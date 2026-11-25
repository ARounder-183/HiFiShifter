//! ARA 文档绑定、region 分配与离线准备。
use super::*;

/// 实例 **writer 身份**的发号器（诊断 / 日志读数，不参与任何行为判定 —— 旁通表决
/// 看的是输入读数，不是身份）。
///
/// 【为什么是模块级而不是函数内 static】此前 ARA 绑定（`bind_to_document`）与
/// Capture 激活（`arm_capture_rendering`）各养了一个计数器 —— 两种形态共存的会话里，
/// 第一个 ARA 实例与第一个 Capture 实例都是 1，旁通表决表、`[ara] writer N …` 日志
/// 与诊断包把两个实例混成同一个身份。
///
/// 【为什么只在 0 时发号】`writer_id` 的语义是"实例身份"，不是"绑定序号"：
/// `arm_capture_rendering` 虽自称幂等，宿主在总线 / 采样率重配时确实会重调
/// `setupProcessing` —— 每次都发新号会让实例身份中途改变，诊断读数前后对不上。
/// 已有身份（非 0）的实例保持不变；发号过程中真正的竞态由 CAS 兜底（输掉的候选
/// 号作废，序列出现空洞 —— 身份互异才是目的，连续并不重要）。
static NEXT_WRITER_ID: AtomicU64 = AtomicU64::new(1);

/// 一次离线准备尝试的结论。
///
/// 【为什么要有第三态而不是 `bool`】"没准备好"与"准备好了"之间的区别对现场毫无
/// 意义 —— 用户看到的是"导出静音"。带上**是哪一段**没准备好（捕获会话/曲线，还是
/// 渲染快照），下一次报障就不必再猜是曲线没加载还是快照没发布。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OfflineReadiness {
    Ready,
    /// 还没准备好，附带未就绪的那一段（只用于日志/错误文本）。
    NotReady(&'static str),
}

impl OfflineReadiness {
    pub(crate) fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }

    pub(crate) fn reason(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NotReady(reason) => reason,
        }
    }
}

/// 给本实例发一个一次性 writer 身份（详见 [`NEXT_WRITER_ID`]）。
fn assign_writer_id(owner: &ExtensionOwner) {
    if owner.writer_id.load(Ordering::Acquire) != 0 {
        return;
    }
    let candidate = NEXT_WRITER_ID.fetch_add(1, Ordering::Relaxed);
    let _ = owner
        .writer_id
        .compare_exchange(0, candidate, Ordering::Release, Ordering::Relaxed);
}

impl ExtensionOwner {
    /// 把租约关联到真实文档；关闭文档时由其同步撤销。
    pub fn bind_to_document(
        self: &Arc<Self>,
        document: Arc<super::document::DocumentSession>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        companion: Option<ara2_bridge::companion::CompanionControllerBinding<'static>>,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        self.startup
            .mark(crate::startup::Stage::AraBind, document.id);
        let _startup_span = crate::startup::Span::new("ara.bind", document.id, 0);
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            // 已绑定再绑定：只有"同一份仍然活着的文档"才是重复绑定。
            //
            // 【为什么不变量从"一个实例只绑一个文档"改成"同一时刻只服务一份活文档"】
            // 宿主顺序不可控：REAPER 可能在旧文档还没关闭时就为新工程发出 bind。
            // 此时 `closed` 仍是 false，若一律拒绝，新文档永远拿不到 renderer
            // （用户报的"重开只显示等待宿主音频，删掉插件再加就好"）。
            //
            // 【为什么不清 `bound_once`】它是历史事实："这个实例绑定过 ARA"。清了它
            // 会让 "document closed" 静默退化成 `ARA_NOT_BOUND`，那是在陈述一件
            // 没发生过的事。
            let same_live_document = self.bound_document_id.load(Ordering::Acquire) == document.id
                && !self.closed.load(Ordering::Acquire);
            if same_live_document {
                return Err(AraError::InvalidState("extension already bound"));
            }
            let previous_document = self
                .document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade);
            let previous_id = previous_document.as_ref().map(|document| document.id);
            if let Some(previous) = previous_document {
                // 旧文档可能仍活着：只摘掉本实例的租约，不关闭整份文档。
                previous.detach_owner(self);
                previous.report_playback_ownership();
            }
            log::warn!(
                "[ara] writer {} moves from document {:?} to {} (previous document {}); \
                 retiring the old extension binding",
                self.writer_id(),
                previous_id,
                document.id,
                if self.closed.load(Ordering::Acquire) {
                    "already closed"
                } else {
                    "still alive"
                }
            );
            self.stop_editor_for_document();
            // 旧扩展实例不能立刻 drop：宿主可能还有最后一次合法回调。
            if let Some(old) = current.take() {
                self.retired.lock_or_recover().push(old);
            }
            self.prepare_rebind_from_previous_document();
            self.assignments.lock_or_recover().clear();
            self.sequences.lock_or_recover().clear();
            self.document.lock_or_recover().take();
            self.rebind_generation.fetch_add(1, Ordering::Relaxed);
            // 【放置位置是"这次绑定"的事实】同一个处理器实例被复用到另一份文档时，
            // 宿主也可能把它从轨道 FX 挪到某条 take 上（用户拖 FX 就会）。留着上一次的
            // 结论，下一次旁通采样会去读**另一条链**。
            if let Some(host) = self.native_host() {
                host.forget_placement();
            }
            self.closed.store(false, Ordering::Release);
        }
        let owner = Arc::downgrade(self);
        let document_id = document.id;
        let observer = Arc::new(
            move |role: ExtensionRoles, keys: &[usize], sequences: &[usize]| {
                if let Some(owner) = owner.upgrade() {
                    let document = owner
                        .document
                        .lock_or_recover()
                        .as_ref()
                        .and_then(std::sync::Weak::upgrade);
                    let Some(document) = document else {
                        return;
                    };
                    // assignment写入、撤销和最终发布必须共用此短事务，消除check→publish竞态。
                    let _transaction = document.transaction.lock_or_recover();
                    let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                    // 【为什么逐条过滤而不是整批作废】`resolve` 是整批语义：一条坏 key
                    // （跨文档/已销毁）就让**整批**返回 `Err`。而宿主会把当前分配整体重发，
                    // 分割后它常把已销毁父段的 key 也带进来 —— 整批作废会连**新 region**
                    // 一起丢掉，表现就是"分割后既不出声也不显示"，且没有任何提示。
                    // 只剔除坏的那条：其余（包括新 region）照常。
                    let (next_keys, dropped): (Vec<RegionKey>, usize) = {
                        let owners = region_owners().lock().unwrap_or_else(|p| p.into_inner());
                        let mut kept = Vec::with_capacity(keys.len());
                        let mut dropped = 0_usize;
                        for key in keys.iter().copied() {
                            if owners.owner_of(key) == Some(document_id) {
                                kept.push(key);
                            } else {
                                dropped += 1;
                            }
                        }
                        (kept, dropped)
                    };
                    let valid = dropped == 0;
                    let count = next_keys.len();
                    let next_sequences =
                        sequences.iter().map(|key| *key as u64).collect::<Vec<_>>();
                    let changed = owner.assignments.lock_or_recover().get(&role.bits())
                        != Some(&next_keys)
                        || owner.sequences.lock_or_recover().get(&role.bits())
                            != Some(&next_sequences);
                    if changed {
                        document.scope_revision.fetch_add(1, Ordering::AcqRel);
                    }
                    owner
                        .assignments
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(role.bits(), next_keys);
                    owner.sequences.lock_or_recover().insert(
                        role.bits(),
                        sequences.iter().map(|key| *key as u64).collect(),
                    );
                    if valid {
                        log::info!(
                            "[ara] renderer assignment role={} regions={count}",
                            role.bits()
                        );
                    } else {
                        log::warn!(
                            "[ara] renderer assignment role={} kept {count} region(s) and \
                             dropped {dropped} unknown or cross-document key(s)",
                            role.bits()
                        );
                    }
                    // 【只有"分配真的变了"才作废】宿主会在一次编辑里把**同一份**分配重发很多遍
                    // （begin_editing 之后、endEditing 收口之前，实测 1 ms 内三次）。而下面这两句的
                    // 代价是"整条轨静音 + 一次完整重渲"：此前这里无条件 clear()，于是每一次重发
                    // 都把可听快照撤成空指针 ⇒ copy_block_at 逐样本写零 —— 用户听到的正是
                    // "每隔一小会儿一次静音断点，又恢复正常"，随后还要再跑一次整条人声的神经合成。
                    // 分配没变 ⇒ 这一轮的渲染输入也没变 ⇒ 什么都不该作废。
                    if changed {
                        owner.cancel_preparation();
                        // 【为什么是 invalidate_generation 而不是 clear】与
                        // DocumentSession::clear_renderers 同一条结论：clear 之后 copy_block_at
                        // 逐样本返回 0.0，而 bound 实例固定 layered == false，没有兜底 —— 那一整段
                        // 就是纯静音。分配变化只该让"这份快照是不是当前内容"失效，不该让它停止可听；
                        // 真正该清空的是"这个实例从此不再渲染"（stop_editor / close）。
                        owner
                            .snapshots
                            .iter()
                            .for_each(|snapshot| snapshot.invalidate_generation());
                    }
                    let owners = document.renderer_owners();
                    for renderer in &owners {
                        if let Err(error) = renderer.merge_pending_restore(&document) {
                            log::warn!("[ara] instance state unresolved: {error}");
                        }
                    }
                    // 某轨首次获得分配可能恢复其state并推进共享revision；其它轨也需要最新任务。
                    drop(_transaction);
                    // 分配变化是角色集合可能改变的第二个时刻：宿主可能在这一步才把
                    // playback 角色交给某个实例。报一次读数，便于从日志判断
                    // "到底有没有实例在渲染"。
                    document.report_playback_ownership();
                    for renderer in owners {
                        renderer.refresh_host_geometry();
                        renderer.prepare();
                    }
                }
            },
        );
        // 【为什么 `EDITOR_VIEW` 必须在 `supported` 里】ARA SDK 对
        // `ARAPlugInExtensionInstance` 的契约是**双向**的：
        //
        //   "Plug-ins **must provide all interfaces that have been requested by the host
        //    through the role assignment**, and suppress interfaces explicitly excluded by
        //    the roles — e.g. if the host did not assign kARAEditorRendererRole even it was
        //    known, editorRendererInterface will be NULL."
        //
        // 此前这里只声明 playback + editor，于是 `enabled = supported & (!known | assigned)`
        // **永远不可能**包含 view ⇒ `editorViewInterface` 恒为 NULL。而 Cubase 实测分配的
        // 正是 `[editor,view]`：它按契约拿 `editorViewInterface` 来通知选区与隐藏
        // （"Selection and hiding of host entities is communicated to editor view plug-ins"），
        // 那正好发生在**打开编辑器**的那一刻 —— 实测崩点就在
        // `IEditController::createView` 之后 7ms，`araservice.dll` 内读空指针。
        //
        // 库那边早就实现好了（`ARAEditorViewInterface` 的 notifySelection /
        // notifyHideRegionSequences），缺的只是我们这一句声明。
        let supported = ExtensionRoles::PLAYBACK_RENDERER
            | ExtensionRoles::EDITOR_RENDERER
            | ExtensionRoles::EDITOR_VIEW;
        let enabled = ExtensionRoles::resolve(known, assigned, supported)?;
        let owned = ExtensionBinding::new_with_renderer_observer(
            generation, known, assigned, supported, observer,
        )?;
        let raw = owned.0.as_raw();
        document.attach(self, owned.1, companion)?;
        *self.document.lock_or_recover() = Some(Arc::downgrade(&document));
        let weak = Arc::downgrade(self);
        // 只登记weak供宿主回调排队；后台计算不形成owner自循环。
        *self.prepare_owner.lock_or_recover() = Some(weak);
        let _ = self.clock.set(document.clock.clone());
        assign_writer_id(self);
        *current = Some(owned.0);
        self.bound_document_id.store(document.id, Ordering::Release);
        // 【T-B0-1】账本里的 `bound` 此前没有任何写者，于是它恒为 false —— 读数在说谎。
        if let Some(ledger) = document.graph_ledger() {
            ledger.note_bound();
        }
        // 绑定成功即置"曾经绑定过"。此后 `editor_document()` 的措辞走 DOCUMENT_CLOSED
        // 而不是 ARA_NOT_BOUND —— 两者对用户的含义完全不同（见该访问器的说明）。
        self.bound_once.store(true, Ordering::Release);
        // 形态也随之确定：这个实例永远走 ARA 通路（`mode()` 先看 `bound_once`，
        // 这里再写原子是为了让不查 `bound_once` 的读者也拿到一致读数）。
        self.note_ara_bound();
        self.roles.store(enabled.bits(), Ordering::Release);
        self.role.store(
            if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                1
            } else {
                2
            },
            Ordering::Release,
        );
        drop(current);
        // 【为什么放在最后】这份读数要看到**全部**实例，包括本实例刚写进去的
        // `writer_id`。角色集合只有两个时刻会变：绑定（这里）与分配变化
        // （见 `new_with_renderer_observer` 的观察器），两处都要报。
        document.report_playback_ownership();
        // 【新文档必须从一次完整枚举开始】`ui_tracks` / `region_items` 是文档级
        // 缓存，重绑后是一份空表；只靠 20ms 定时器要等到节流窗口过期才会填。
        // 这里同步采一次，界面在绑定成功的当帧就能看到 item 与授权媒体。
        for renderer in document.renderer_owners() {
            renderer.refresh_host_geometry();
            renderer.refresh_ui_inventory();
            renderer.prepare();
        }
        // 【触发点②：实例绑定】重开工程时"先给存档、再交付模型"，而存档到达那一刻
        // 还没有任何 owner —— 于是合并的唯一入口（`prepare_renderers`）要等宿主收口。
        // 绑定是**图已经长出来**的第一个确定时刻，在这里补一次尝试。
        if let Err(error) = document.merge_archive_restore() {
            log::warn!("[ara] archive merge after binding failed: {error}");
        }
        // 【T8：把图读数打出来】绑定成功的那一刻宿主的图通常还没建（region 在之后才到），
        // 所以这一行几乎总是全零 —— 它的价值在于**当它停在零**：下一次重启报障时，
        // 若这里 ctx=0，说明宿主要么没发图、要么被我们拒了，不必再从位掩码反推。
        log::info!(
            "[ara] {} (document={})",
            graph_summary_of(&document),
            document.id
        );
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        // 【为什么不是 `stop_editor`】文档结束不等于组件结束：宿主复用同一个
        // 处理器重开工程时，原生扩展适配器必须保留，否则重绑后永远枚举不到
        // 宿主 item（见 `stop_editor_for_document` 的说明）。
        self.stop_editor_for_document();
        // 【为什么这里要重算覆盖】本实例退出后，"文档上还有没有 playback 实例"可能翻转
        // （播放实例先死、留下编辑实例）。`stop_editor` 已把 `closed` 置真，所以
        // `renderer_owners()` 不再包含自己 —— 重算后留下的那个实例会接管渲染，
        // 而不是整档静音。
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            document.report_playback_ownership();
        }
        self.assignments.lock_or_recover().clear();
        self.sequences.lock_or_recover().clear();
        self.document.lock_or_recover().take();
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        // 【刻意不清 `bound_once`】绑定过又关闭就是真的关闭了，措辞必须保持
        // DOCUMENT_CLOSED。清掉它会让 "document closed" 静默退化成
        // ARA_NOT_BOUND（"宿主从未绑定"）—— 那是在陈述一件没发生过的事。
    }

    /// 模型线程展开当前 renderer 的显式区域与 editor sequence，并去重，拒绝跨文档身份。
    pub fn assigned_regions(&self) -> Result<Vec<u64>, AraError> {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or(AraError::InvalidState("document no longer available"))?;
        let role = self.role.load(Ordering::Acquire);
        // 【T1：与枚举/几何路径共用同一个判据】此前这里是第二份实现，且用的是
        // **整批** `resolve` —— 宿主在一次分割里把已销毁父段的 key 一并重发时，
        // 一条陈旧 key 就让整份 scope 失败（→ `ensure_loaded` 整份 Err → 界面回落到
        // 只有宿主清单占位的降级视图）。现在两条路径共用 `assigned_scope`：
        // 逐条过滤、不整批作废。
        //
        // 【为什么这里**不**允许文档级派生】渲染输入是**逐实例**的：REAPER 的一个 ARA
        // 文档上挂着多条轨道的实例，派生出来的并集会让同一段音频被渲染两次。
        // 派生只服务并集语义的调用方（工作区、几何、传输）。
        self.assigned_scope(&document, Some(role as u32), false)
            .map(|scope| scope.keys)
            .map_err(|error| AraError::InvalidState(error.leak()))
    }

    /// 宿主模型callback只撤销/排最新准备任务，不在主线程执行WORLD或PCM混音。
    ///
    /// 【为什么 `fetch_add` 必须在 `renders_playback()` 之后】计数要衡量的是
    /// "播放实例收到了多少次通知" —— 这正是 `render_input_invalidation_…`
    /// 那条不变量测试锁的东西（此前漏掉播放实例的回归就在这里）。放在前面
    /// 会把编辑器实例的 no-op 也数进去，测试就再也分不出"通知了谁"。
    /// 为 Capture 形态准备渲染基础设施。
    ///
    /// 【为什么必须有这一步】`prepare()` 依赖 `prepare_owner` 与 `preparation`，而这两样
    /// 此前**只在 ARA 绑定时**建立（`bind_to_document`）。Capture 实例从不绑定，于是
    /// 它的 `prepare()` 会在第一个早退处静默返回 —— 录制正常、界面正常、就是没有声音。
    /// 这类"线都接好了但没插电"的故障最难归因，所以这里显式补上那部分与 ARA 无关的
    /// 初始化。
    ///
    /// 【为什么可以重复调用】三处都是 `OnceLock` / 覆盖式赋值，重复调用无副作用；
    /// 它由 20ms 的 UI 定时器驱动，因此必须幂等。
    pub fn arm_capture_rendering(self: &Arc<Self>) {
        self.init_block_trace();
        self.ensure_preparation();
        *self.prepare_owner.lock_or_recover() = Some(Arc::downgrade(self));
        // 时钟：Capture 没有 ARA 文档的共享时钟，自己起一个。宿主播放位置由
        // `audio_process_inner` 的 `observe_transport` 照常写入（它与 ARA 无关）。
        let _ = self
            .clock
            .set(Arc::new(super::transport::TransportClock::default()));
        // 身份只发一次：重复调用（宿主重配 / UI 定时器）不得改变实例身份，
        // 见模块级 NEXT_WRITER_ID 的说明。
        assign_writer_id(self);
        // 【为什么在这里也要把会话立起来】处理结果的**唯一**生产者是会话 actor
        // （`start_capture_render` → `capture_prepare_job`），而会话此前只在
        // `webview.rs` 的命令分发里惰性创建 —— 于是"从未打开过插件窗口"的会话里
        // 没有任何东西会把带曲线的渲染排出去：录制正常、时间轴有内容、
        // **播放时永远直通宿主原声**（用户报障）。
        // `setupProcessing` 是 SDK 保证在任何 `process` 之前、且在宿主 UI 线程上被
        // 调用的唯一时刻；actor 由会话创建，与窗口开没开无关，所以建在这里就够了。
        //
        // 【为什么只在有素材时才建】会话会起一个线程、初始化模型并建临时目录
        // （`EditorSession::new_with`）。一个插在母线上、用户早已忘记的实例不该在
        // 每次 `setupProcessing` 上都付这份代价；而没有素材时它也没有任何东西可渲染。
        if self.capture.has_material() || self.pending_capture_edits.lock_or_recover().is_some() {
            if let Ok(session) = self.clone().capture_session() {
                // 让 actor 下一轮加载时间线并排一次渲染（幂等：已加载时它只推票据）。
                session.request_reload();
            }
        }
    }

    /// 首快照同步发布的等待预算。
    ///
    /// 【为什么从 2 s 降到 250 ms】那个上界的本意是"第一块之前必须有内容"，但它支付
    /// 的是**宿主消息泵**的时间。现场一次导入的挂起里，2 s 的上界意味着宿主每问一次
    /// 就卡满两秒，Windows 直接判 `AppHangB1`。250 ms 覆盖实测的宿主预渲染窗口
    /// （REAPER ≈225 ms），超过它的合成本来也不可能在宿主问第一块之前完成 —— 继续等
    /// 只会把宿主拖死，而实时通路的 `layered` 回退已经保证那一段不是静音。
    pub(crate) const FIRST_SNAPSHOT_BUDGET: std::time::Duration =
        std::time::Duration::from_millis(250);

    pub fn prepare(&self) {
        self.prepare_with_budget(Some(Self::FIRST_SNAPSHOT_BUDGET));
    }

    /// 与 [`Self::prepare`] 相同，但可以指定首快照的等待预算（`None` = 只排队）。
    ///
    /// 【谁传 `None`】所有**宿主回调链**：`DocumentSession::prepare_renderers` 从
    /// `end_editing` 进来，写回/分割/剪贴板从宿主命令进来。那些线程上一律不许等 ——
    /// 这是本轮的主题（见 `host::reentrancy` 的模块文档）。
    pub(crate) fn prepare_with_budget(&self, budget: Option<std::time::Duration>) {
        // 【B0】先报告上一批可疑块。放在早退**之前**：Capture 实例在没有素材时
        // `renders()` 为假，而那正是"刚录完、播放中"这个最需要读数的时刻。
        self.log_click_report();
        self.log_capture_report();
        self.log_snapshot_readout();
        if self.is_closed() || !self.renders() {
            return;
        }
        // 【旁通即不渲染】宿主已经把这个 FX 旁通（或离线 / 关掉整条 FX 链）时，
        // 合成结果**没有任何人会取用** —— ARA 的播放渲染器在旁通期间根本不会被调用。
        // 判据只认宿主自己报的读数（见 `sample_host_bypass`）：非 REAPER 宿主恒为
        // Unknown，走的是改动前的行为。
        //
        // 【为什么用 `fresh_host_bypass`】`host_bypass` 给的是"最后一次确定读到的
        // 结论"，它可能已经过期（宿主换轨 / 切工程 / 卸载 FX）。拿过期读数挡渲染，
        // 用户解除旁通后就没有声音了；而"从没读到过"更不是旁通。两者都必须回到
        // 改动前的行为（照常渲染）。
        if self.fresh_host_bypass() == Some(crate::host::reaper::FxBypass::Bypassed) {
            return;
        }
        // 源/assignment撤销在文档事务内清理；此函数只排队，不能在事务外clear旧任务刚发布的音频。
        #[cfg(test)]
        self.prepare_requests.fetch_add(1, Ordering::AcqRel);
        let weak = self.prepare_owner.lock_or_recover().clone();
        let Some(weak) = weak else {
            return;
        };
        if self.is_bound() {
            let role = self.role.load(Ordering::Acquire);
            let regions = self
                .assignments
                .lock_or_recover()
                .get(&role)
                .is_some_and(|keys| !keys.is_empty());
            let sequences = self
                .sequences
                .lock_or_recover()
                .get(&role)
                .is_some_and(|keys| !keys.is_empty());
            if !regions && !sequences {
                return;
            }
        }
        // 【第一次只是"抢一拍"，不再同步】见 `request_first_snapshot`。
        self.request_first_snapshot(budget);
        self.ensure_preparation();
        if let Some(worker) = self.preparation() {
            if let Err(error) = worker.request(Box::new(move |cancel| {
                weak.upgrade()
                    .ok_or("renderer closed".to_owned())?
                    .prepare_job(cancel)
            })) {
                log::warn!("[ara] preparation unavailable: {error}");
            }
        }
    }
    /// 调用者持document事务；完整两输出率发布成功后才登记，不额外持有源或快照。
    pub(crate) fn record_prepared(
        &self,
        model: u64,
        edit: u64,
        epoch: u64,
        scope: u64,
        keys: Vec<u64>,
    ) {
        *self.prepared.lock_or_recover() = Some(PreparedVersion {
            model,
            edit,
            epoch,
            scope,
            keys,
        });
        if let Some(queue) = self.preparation() {
            queue.acknowledge_idle_success();
        }
    }
    /// 离线准备**尝试一次**，不等任何合成。
    ///
    /// 【为什么它必须是非阻塞的】调用者是 VST3 `setupProcessing` / `setProcessing(true)`
    /// 与 CLAP 的对应入口 —— 它们都在**宿主的 UI 线程**上。旧实现在这里 `sleep(20ms)`
    /// 轮询到 15 s：合成一旦不收敛，宿主就冻满 15 秒，导出以"宿主未响应"收场。
    /// 现在只做两件事：把"该准备"排进各自的队列，然后如实回答**此刻**有没有准备好。
    pub(crate) fn try_prepare_offline_once(self: &Arc<Self>) -> Result<OfflineReadiness, String> {
        if self.is_closed() {
            return Err("offline renderer closed".into());
        }
        // 【判据与 `prepare_job` 同一份】离线只跑一趟，没有第二次机会，所以这里更不能
        // 因为"窗口没开过 ⇒ 形态未落定"就把静音烘焙进产物。与实时通路共用
        // "未绑定 + 有素材"这条证据（理由见 `prepare_job` 的同名分支）。
        let needs_capture = !self.is_bound()
            && (self.capture.has_material()
                || self.pending_capture_edits.lock_or_recover().is_some());
        if needs_capture {
            // 【关键：先确保会话存在】离线实例可能从未打开过 GUI，曲线只在
            // `pending_capture_edits` 里。没有会话时捕获通路会退回"只有几何"的时间线，
            // 并把 with_curves=false 发布出去 —— 那正是"参数线不生效"。
            if let Ok(session) = self.clone().capture_session() {
                session.request_reload();
                // 【合成的落点是 actor，不是这条线程】`rearm_render` 只置一个待办位并
                // 唤醒 actor；真正数秒的合成由 actor 的渲染任务执行（它自带在途检查、
                // 取消与重试预算）。
                session.rearm_render();
            }
            if !self.capture_offline_ready()? {
                return Ok(OfflineReadiness::NotReady("capture session/curves"));
            }
        }
        if !self.renders_playback() {
            return Ok(OfflineReadiness::Ready);
        }
        // SDK 离线 setup 在 UI 线程；无 GUI 时也要刷新真实 item 状态，不能依赖 WebView timer。
        self.refresh_host_geometry();
        let document = self.editor_document()?;
        {
            let _transaction = document.transaction.lock_or_recover();
            if !document.is_alive() {
                return Err("offline renderer closed".into());
            }
            if !document.ready.load(Ordering::Acquire) {
                return Err("offline host model/PCM not ready".into());
            }
            let keys = self.assigned_regions().map_err(|e| e.to_string())?;
            let version = PreparedVersion {
                model: document.revision.load(Ordering::Acquire),
                edit: document.edits.lock_or_recover().revision,
                epoch: document.render_epoch.load(Ordering::Acquire),
                scope: document.scope_revision.load(Ordering::Acquire),
                keys,
            };
            if self.prepared.lock_or_recover().as_ref() == Some(&version)
                && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
            {
                return Ok(OfflineReadiness::Ready);
            }
        }
        // 【排一次，不等它】与 `prepare()` 同一个入口：`ensure_preparation` 起队列，
        // `worker.request` 排队；在途时不再重复排队（那一轮会自己看到新版本）。
        self.ensure_preparation();
        let worker = self
            .preparation()
            .ok_or("offline preparation worker missing")?;
        if !worker.state().0 {
            let owner = Arc::downgrade(self);
            worker.request(Box::new(move |cancel| {
                owner
                    .upgrade()
                    .ok_or("renderer closed".to_owned())?
                    .prepare_job(cancel)
            }))?;
        }
        Ok(OfflineReadiness::NotReady("renderer snapshots"))
    }

    /// 离线准备**阻塞等到**就绪（或 deadline / 不收敛）。
    ///
    /// 【只允许在专用 worker 上调用】这是全插件唯一会 `sleep` 的离线入口，而它睡的是
    /// **调用线程**。`in_host_call()` 为真时直接拒绝：那条线程是宿主的消息泵，在这里
    /// 睡满 15 s 就是一次 `AppHangB1`（见 `host::reentrancy` 的模块文档）。
    ///
    /// 【为什么 Capture 要在这里同步准备】bounce / 导出是 Capture 的**第二条通路**，
    /// 而它是宿主跑得最快的一次 —— 插件没有第二次机会，所以后台收尾线程允许自己把
    /// 合成做完（`deadline` 对它无意义：没有文档事务可等，合成本身有界）。
    pub(crate) fn prepare_offline_blocking(
        self: &Arc<Self>,
        deadline: std::time::Instant,
    ) -> Result<(), String> {
        if crate::host::in_host_call() {
            return Err(
                "offline preparation must not block a host callback; it belongs on a worker"
                    .into(),
            );
        }
        let needs_capture = !self.is_bound()
            && (self.capture.has_material()
                || self.pending_capture_edits.lock_or_recover().is_some());
        if needs_capture {
            // 【关键：先确保会话存在】离线实例可能从未打开过 GUI，曲线只在
            // `pending_capture_edits` 里。没有会话时捕获通路会退回“只有几何”的
            // 时间线，并把 with_curves=false 发布出去 —— 那正是“参数线不生效”。
            // capture_session 会把 pending edits 投递给 actor，并请求加载。
            if let Ok(session) = self.clone().capture_session() {
                session.request_reload();
            }
            let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
            loop {
                self.capture_prepare_job_offline(cancel.clone())?;
                if self.capture_offline_ready()? {
                    return Ok(());
                }
                if std::time::Instant::now() >= deadline {
                    return Err(
                        "offline preparation timed out waiting for the capture session/curves"
                            .into(),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        if !self.renders_playback() {
            return Ok(());
        }
        // SDK离线setup在UI线程；无GUI时也刷新真实item状态，不能依赖WebView timer。
        self.refresh_host_geometry();
        let document = self.editor_document()?;
        // 【为什么需要尝试次数上限】若后台准备**成功返回**却仍不能收敛到当前版本
        // （例如 ready 永远不置位），此前会一路空转到 deadline，把宿主线程白等满
        // `OFFLINE_PREPARE_TIMEOUT`。同一个版本连续排队两次仍不匹配就是不可能收敛，
        // 提前返回明确错误，让上层如实记录，而不是静默冻结 15 s。
        let mut attempts = 0usize;
        loop {
            {
                let _transaction = document.transaction.lock_or_recover();
                if self.is_closed() || !document.is_alive() {
                    return Err("offline renderer closed".into());
                }
                if !document.ready.load(Ordering::Acquire) {
                    return Err("offline host model/PCM not ready".into());
                }
                let keys = self.assigned_regions().map_err(|e| e.to_string())?;
                let version = PreparedVersion {
                    model: document.revision.load(Ordering::Acquire),
                    edit: document.edits.lock_or_recover().revision,
                    epoch: document.render_epoch.load(Ordering::Acquire),
                    scope: document.scope_revision.load(Ordering::Acquire),
                    keys,
                };
                if self.prepared.lock_or_recover().as_ref() == Some(&version)
                    && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
                {
                    return Ok(());
                }
            }
            if std::time::Instant::now() >= deadline {
                return Err("offline preparation timed out".into());
            }
            self.ensure_preparation();
            let worker = self
                .preparation()
                .ok_or("offline preparation worker missing")?;
            if !worker.state().0 {
                attempts += 1;
                if attempts > 2 {
                    return Err(
                        "offline preparation did not converge; the current edit cannot be \
                         rendered (host model/PCM or region assignment keeps changing)"
                            .into(),
                    );
                }
                let owner = Arc::downgrade(self);
                worker.request(Box::new(move |cancel| {
                    owner
                        .upgrade()
                        .ok_or("renderer closed".to_owned())?
                        .prepare_job(cancel)
                }))?;
            }
            worker.wait_idle_until(deadline)?;
        }
    }
    /// 宿主拿到 region 之后**立刻**开始渲染，而渲染线程不会等后台队首的任务。
    ///
    /// 【为什么要在 `prepare()` 里抢这一拍】实测（REAPER，48 kHz）：实例建好到第一份
    /// 快照发布之间，宿主已经渲染了 21 个块（≈225 ms），那段窗口里 `copy_block_at`
    /// 没有快照。抢一拍的价值是把这段窗口尽量压短。
    ///
    /// 【为什么**不再**在调用线程上等它做完】旧实现在这里 `wait_idle_until(2 s)`，理由是
    /// "第一块之前必须有内容"。那个理由本身没错，但代价选错了支付方：`prepare()` 的调用者
    /// 几乎全部在**宿主线程**上（ARA 模型回调、写回、视图节拍），于是那 2 秒花掉的是
    /// 宿主的消息泵。现场（2026-11-25，REAPER 7.8.5）`文件 → 导入媒体` 之后主线程就停在
    /// 这里，Windows 记录 `AppHangB1` ——"stopped interacting with Windows and was closed"。
    ///
    /// 【那"第一块之前必须有内容"怎么办】它有两个**不需要阻塞宿主**的替代：
    ///
    ///   1. 实时通路：快照未就绪时 `CaptureOutput::Untouched` 以 `layered: true` 输出，
    ///      回退取宿主输入总线（`snapshot.rs` 的 `input_at`）—— 那是宿主本来会播的东西，
    ///      比一串零诚实得多，也不会被烘进宿主的 ARA 缓存；
    ///   2. 离线/导出通路：`prepare_offline_until` 显式地等（它有一次性的预算与自己的
    ///      收敛判据），那条路不在宿主的消息泵上。
    ///
    /// 【为什么保留一次性闸门】`has_pending_render_sources()` 要取三张表；把它挂在 20 ms
    /// 节拍上没有必要。抢过一拍之后，后续的排队由 `prepare()` 末尾那次 `worker.request`
    /// 负责 —— 它本来就是每拍都在做的。
    /// 首快照这一次允许等多久（`None` = 只排队）。
    ///
    /// 【判据只有一条】**宿主回调链上不许等**。`in_host_call()` 为真说明这条线程是
    /// 宿主同步进入我们的（`UpdateItem` / `UpdateArrange` / `SetItemStateChunk` /
    /// 撤销跳转 / 导入……），它此刻唯一的职责是把控制权还给宿主的消息泵。
    ///
    /// 【为什么抽成纯函数】它是这一轮的核心纪律，必须能被直接断言 —— 藏在
    /// `request_first_snapshot` 里就只能靠"跑一遍再量耗时"来间接证明。
    pub(crate) fn first_snapshot_budget(
        requested: Option<std::time::Duration>,
    ) -> Option<std::time::Duration> {
        if crate::host::in_host_call() {
            None
        } else {
            requested
        }
    }

    fn request_first_snapshot(&self, budget: Option<std::time::Duration>) {
        if !self.is_bound() {
            return;
        }
        // 【为什么只在有原生扩展时做】会提前渲染的宿主，就是会按 region 抢在播放头
        // 之前把音频烘进自己缓存的宿主（REAPER 的 ARA 渲染器实测 100% 走 prefetch）。
        // 没有原生扩展就没有这条渲染通路，"第一块之前必须有内容"也就无从谈起 —— 那
        // 一侧保持改动前的行为，别为一条用不上的前提在别处多算一次完整合成。
        if self.native_host().is_none() {
            return;
        }
        if self.first_snapshot_inlined.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.snapshots.iter().all(|snapshot| snapshot.is_ready()) {
            return;
        }
        // 【素材还没到就别把这一次机会用掉】宿主授权源 PCM 是异步的（实测 32 ms），
        // 而 `prepare_job` 在源还没到时会以 `MissingSource` 失败。这一拍只允许抢一次 ——
        // 在素材到达之前就花掉，等于等宿主真正开始问音频时我们再没有机会。
        if !self.has_pending_render_sources() {
            self.first_snapshot_inlined.store(false, Ordering::Release);
            return;
        }
        let Some(weak) = self.prepare_owner.lock_or_recover().clone() else {
            self.first_snapshot_inlined.store(false, Ordering::Release);
            return;
        };
        self.ensure_preparation();
        let Some(worker) = self.preparation() else {
            self.first_snapshot_inlined.store(false, Ordering::Release);
            return;
        };
        // 【为什么是排队而不是直接调 `prepare_job`】它会在这条线程上跑一次完整合成，
        // 而这条线程是宿主的。让 worker 做；`prepare()` 末尾还会再排一次，两处都只是
        // 排队，不会叠加成"每问一次就阻塞一次"。
        if let Err(error) = worker.request(Box::new(move |cancel| {
            weak.upgrade()
                .ok_or("renderer closed".to_owned())?
                .prepare_job(cancel)
        })) {
            log::warn!("[ara] the first snapshot could not be queued: {error}");
            self.first_snapshot_inlined.store(false, Ordering::Release);
            return;
        }
        // 【纪律的读数】`in_host_call()` 为真 = 我们正处在一次宿主调用里（宿主 API
        // 同步回调进来的）。那种线程上**任何**等待都是在扣宿主的消息泵，所以只排队。
        // `HFS_HOST_WATCHDOG=0` 只关掉看门狗日志，不影响这条判据。
        let budget = Self::first_snapshot_budget(budget);
        let Some(budget) = budget else {
            log::debug!(
                "[ara] the first snapshot was queued; in_host_call={} (this thread must not wait)",
                crate::host::in_host_call()
            );
            return;
        };
        let deadline = std::time::Instant::now() + budget;
        let _ = worker.wait_idle_until(deadline);
        if self.snapshots.iter().all(|snapshot| snapshot.is_ready()) {
            log::info!("[ara] the first snapshot is ready before the host can ask for audio");
            return;
        }
        self.first_snapshot_inlined.store(false, Ordering::Release);
        log::warn!(
            "[ara] the first snapshot did not finish within {} ms; the background worker keeps going",
            budget.as_millis()
        );
    }

    /// 已分配的 region 里，是否至少有一条的源 PCM 已经在手上。
    ///
    /// 【为什么这件事要单独判】见 `request_first_snapshot`：没有源 PCM 的
    /// 同步渲染必然 `MissingSource`，而那次尝试只允许成功一次。三个表分开取锁，
    /// 顺序与 `ara_render_input` 一致（regions → sources → edit_sources），不制造新的
    /// 交叉持锁。
    pub(crate) fn has_pending_render_sources(&self) -> bool {
        let Ok(keys) = self.assigned_regions() else {
            return false;
        };
        let Ok(document) = self.editor_document() else {
            return false;
        };
        let ids: Vec<String> = {
            let regions = document.regions.lock_or_recover();
            keys.iter()
                .filter_map(|key| {
                    regions
                        .get(key)
                        .map(|region| region.audio_source_persistent_id.clone())
                })
                .collect()
        };
        if ids.is_empty() {
            return false;
        }
        let live = {
            let sources = document.sources.lock_or_recover();
            ids.iter().any(|id| sources.contains_key(id))
        };
        if live {
            return true;
        }
        let edits = document.edit_sources.lock_or_recover();
        ids.iter().any(|id| edits.contains_key(id))
    }

    /// 后台冻结/计算/发布三阶段；撤销标记和全部版本在发布短事务内重新验证。
    pub(super) fn prepare_job(
        &self,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), String> {
        // Capture 形态没有 ARA 文档；在取文档**之前**分流，否则它会以
        // "document closed" 收场，而那条路对 Capture 是常态而非错误。
        //
        // 【为什么判据是"未绑定 + 有素材"而不是 `mode()`】形态由 `note_editor_tick`
        // 落定，而它的唯一调用点是 WebView 的 20ms 定时器 —— **窗口不开就没有这个
        // tick**，而且即便开了也要等 `setActive` 之后 5 秒。于是宿主把插件当普通插入、
        // 用户录完关掉窗口再播放的会话里 `mode()` 恒为 `Undetermined` ⇒ 这里永不分流
        // ⇒ 一条处理结果都没合成过 ⇒ 仲裁落到直通。
        // `renders()` 早就因为同一理由绕开了 `mode()`（见它的注释）；准入侧与产出侧
        // 必须问同一个问题，否则"该出声"与"还没到时候"会各自漂移。
        if !self.is_bound() && self.capture.has_material() {
            return self.capture_prepare_job(cancel);
        }
        let Some(document) = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return Err("document closed".into());
        };
        let (model, edit, epoch, scope, keys, input) = {
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire) {
                return Ok(());
            }
            let edits = document.edits.lock_or_recover().clone();
            let (keys, input) = self.ara_render_input(&document, &edits, edits.revision > 0)?;
            // actor可能已发布本代两率结果；同版本重排队不应再争预算或留下伪失败状态。
            let version = PreparedVersion {
                model: document.revision.load(Ordering::Acquire),
                edit: edits.revision,
                epoch: document.render_epoch.load(Ordering::Acquire),
                scope: document.scope_revision.load(Ordering::Acquire),
                keys: keys.clone(),
            };
            if self.prepared.lock_or_recover().as_ref() == Some(&version)
                && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
            {
                return Ok(());
            }
            (
                document.revision.load(Ordering::Acquire),
                edits.revision,
                document.render_epoch.load(Ordering::Acquire),
                document.scope_revision.load(Ordering::Acquire),
                keys,
                input,
            )
        };
        for publisher in &self.snapshots {
            publisher.collect_retired();
        }
        let snapshots = input.render(cancel.clone())?;
        Self::log_snapshot_integrity(&snapshots);
        // 【重排必须发生在**出锁之后**】`prepare()` 会走一串取锁路径（诊断行、形态
        // 判定、准备队列）。在持有 `document.transaction` 时调用它，就是让"文档事务 →
        // 任意其它锁"这条隐藏锁序成立；而 `prepare_job` 自己的 `Retry` 又会被 worker
        // 立刻再跑一遍 —— 两条加起来就是现场那"2 s × N"的空转。
        let superseded = {
            let _transaction = document.transaction.lock_or_recover();
            if cancel.load(Ordering::Acquire) || !document.ready.load(Ordering::Acquire) {
                return Ok(());
            }
            if document.revision.load(Ordering::Acquire) != model
                || document.edits.lock_or_recover().revision != edit
                || document.render_epoch.load(Ordering::Acquire) != epoch
                || document.scope_revision.load(Ordering::Acquire) != scope
                || self.assigned_regions().map_err(|e| e.to_string())? != keys
            {
                // 跨轨接受新参数不一定有新的 model callback；丢弃后补排最新任务，
                // 不能永久留空且假 idle。
                true
            } else {
                // 【空分配不是内容变成静音】ara_render_input 在没有任何被分配 region 时返回一份空的
                // 渲染输入，而它渲出来的是一份 len == 0 的快照。把那份快照发布出去有两个直接后果：
                // is_ready() 变成真（于是 processed_snapshot_ready 也真、layered 退化成 false），而
                // copy_block_at 在这份空快照上逐样本返回 0.0 —— 整条轨就是纯静音，直到下一次真正的
                // 分配把快照换回来。宿主在一次编辑里会短暂地给出空分配
                // （renderer assignment ... regions=0），这段时间用户听到的正是静音断点。
                //
                // 【判据为什么是"文档里还有 region"】自己也一条都没有、文档里也一条都没有 ⇒ 真的没
                // 内容可播，此时发布空快照才是诚实的（否则删掉全部片段之后还在播上一版）。文档里有
                // region、只是这一刻没分配给我们 ⇒ 只是记账上的空窗，保留上一份可听快照。
                if keys.is_empty() && !document.regions.lock_or_recover().is_empty() {
                    hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Debug,
                "ara.prepare.empty-assignment",
                format_args!("[ara] no region is assigned to this renderer right now; the published snapshot is kept instead of replacing it with an empty one"),
            );
                    return Ok(());
                }
                if !self
                    .snapshots
                    .iter()
                    .zip(&snapshots)
                    .all(|(p, s)| p.has_capacity(s))
                {
                    return Err("retired snapshot budget exhausted".into());
                }
                for (publisher, snapshot) in self.snapshots.iter().zip(snapshots) {
                    publisher
                        .publish_shared(snapshot)
                        .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
                }
                self.record_prepared(model, edit, epoch, scope, keys.clone());
                log::info!(
                "[ara] background snapshot ready role={} revision={edit} model={model} regions={}",
                self.role.load(Ordering::Relaxed),
                keys.len()
            );
                false
            }
        };
        if superseded {
            self.requeue_superseded_prepare();
        }
        Ok(())
    }

    /// 版本被取代后的补排 —— **只排队**，且同一批版本变化最多补排几次。
    ///
    /// 【为什么要收敛闸】一次导入会连续触发多条 region 创建/更新，每一条都推进
    /// `document.revision`。没有闸时这条链是：发布成功 → 版本已变 → 补排 → 又变 →
    /// …，每次都完整合成一遍。现场 22:58:08–11 的 6 次 `mixdown` 与 22:58:56 的 3 次
    /// 全是它的脚印，而一行 `[ara] prepared` 都没有 —— 每次都在版本重检处被丢弃。
    ///
    /// 【为什么只记一行就停】继续排不会收敛：宿主那一轮编辑还没结束，版本必然还会变。
    /// 停下来的代价是"这一版快照暂时是旧的"，而下一轮 `endEditing` 会重新驱动它；
    /// 不停下来的代价是宿主线程与 worker 一起被拖死。
    fn requeue_superseded_prepare(&self) {
        let allowed = {
            let mut gate = self.prepare_retry.lock_or_recover();
            gate.allow(std::time::Instant::now())
        };
        if !allowed {
            let report = {
                let mut gate = self.prepare_retry.lock_or_recover();
                gate.should_report()
            };
            if report {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.prepare.non-convergent",
                    format_args!(
                        "[ara] prepare did not converge ({} requeues within {} ms); waiting for the next model change instead of spinning",
                        crate::render::extension::PREPARE_RETRY_LIMIT,
                        crate::render::extension::PREPARE_RETRY_WINDOW.as_millis()
                    ),
                );
            }
            return;
        }
        self.prepare();
    }

    /// 发布前扫一遍快照**内容**：短洞（2 ~ 250 ms、两侧都有信号）是"断断续续"的直接证据。
    ///
    /// 【为什么这个读数必须存在】"播放时有静音断点"有两种成因，而它们在播放侧的所有
    /// 读数里逐样本相同：**内容里就有洞**（快照那几帧真的是零）与**播放没读到内容**
    /// （读指针越界 → 写零）。前者要在合成里修，后者要在拷贝里修 —— 方向完全相反。
    /// 快照是两条路唯一的汇合点，所以判据放在这里：发布时扫一次，把"洞在哪一秒、多宽"
    /// 直接写进日志。
    ///
    /// 【为什么不报正常静音】只看**两侧 100 ms 内都有 >= -40 dBFS 信号**的短洞：
    /// 一句唱完的换气、间奏、句尾收音都长得多或一侧没有信号，不该进这份读数。
    fn log_snapshot_integrity(
        snapshots: &[std::sync::Arc<crate::render::snapshot::PlaybackSnapshot>],
    ) {
        let Some(snapshot) = snapshots.first() else {
            return;
        };
        if snapshot.left.is_empty() || snapshot.sample_rate == 0 {
            return;
        }
        let rate = f64::from(snapshot.sample_rate);
        let frames = snapshot.left.len();
        let level = |index: usize| -> f32 {
            let left = snapshot.left[index].abs();
            let right = snapshot.right.get(index).copied().unwrap_or(0.0).abs();
            left.max(right)
        };
        let has_signal = |lo: usize, hi: usize| -> bool {
            let hi = hi.min(frames);
            lo < hi && (lo..hi).any(|index| level(index) >= SNAPSHOT_HOLE_SIGNAL)
        };
        let mut holes = 0usize;
        let mut silent_frames = 0usize;
        let mut places: Vec<f64> = Vec::new();
        // 【为什么用 `is_finite && <= 地板` 而不是 `> 地板` 取反】NaN 与地板的比较
        // 两边都假：写成 `if value > FLOOR { 前进 } else { 进入静音段 }` 时，一个
        // 非有限采样会让内层循环**一步都不前进** —— 那就是一次不返回的发布
        // （实测：带 NaN 的夹具把整条测试运行卡死）。判据必须先问"这一帧确实是零"。
        let is_silent = |index: usize| -> bool {
            let value = level(index);
            value.is_finite() && value <= SNAPSHOT_HOLE_FLOOR
        };
        let mut index = 0usize;
        while index < frames {
            if !is_silent(index) {
                index += 1;
                continue;
            }
            let start = index;
            while index < frames && is_silent(index) {
                index += 1;
            }
            let len = index - start;
            if !(SNAPSHOT_HOLE_MIN_FRAMES..=SNAPSHOT_HOLE_MAX_FRAMES).contains(&len) {
                continue;
            }
            let before = start.saturating_sub(SNAPSHOT_HOLE_LOOKAROUND);
            if !has_signal(before, start) || !has_signal(index, index + SNAPSHOT_HOLE_LOOKAROUND) {
                continue;
            }
            holes += 1;
            silent_frames += len;
            if places.len() < 8 {
                places.push((snapshot.origin_sample + start as i64) as f64 / rate);
            }
        }
        if holes > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.snapshot.integrity",
                format_args!(
                    "[ara] the published snapshot has {holes} short silent run(s) inside signal ({silent_frames} frames); first at {places:?}s — the content itself is chopped, not the playback path",
                ),
            );
        } else {
            log::info!(
                "[ara] snapshot integrity: no short silent run inside signal ({} frames @ {} Hz, origin={})",
                frames,
                snapshot.sample_rate,
                snapshot.origin_sample
            );
        }
    }
}

/// 判"静音"的地板：低于它就是数字零（或残余噪声底）。
const SNAPSHOT_HOLE_FLOOR: f32 = 1e-6;
/// 判"两侧确实有信号"的门限（满幅 = 1.0）：约 −40 dBFS。
const SNAPSHOT_HOLE_SIGNAL: f32 = 0.01;
/// 短于 2 ms 的洞不可闻（48 kHz 下 96 帧），不报。
const SNAPSHOT_HOLE_MIN_FRAMES: usize = 96;
/// 长于 250 ms 的洞是正常静音（换气 / 间奏），不报。
const SNAPSHOT_HOLE_MAX_FRAMES: usize = 12_000;
/// 两侧看多远（100 ms @ 48 kHz）。
const SNAPSHOT_HOLE_LOOKAROUND: usize = 4_800;
