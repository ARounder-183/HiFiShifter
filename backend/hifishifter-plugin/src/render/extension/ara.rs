//! ARA 文档绑定、region 分配与离线准备。
use super::*;
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
                    owner.cancel_preparation();
                    owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
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
        // 【为什么不再只对 playback 角色初始化】文档上没有任何 playback 实例时，编辑实例会
        // **接管渲染**（见 [`ExtensionOwner::is_editor_only`]），那时它也必须有这条准备队列 ——
        // 只对 playback 初始化会让接管的实例永远排不出任务（`prepare()` 的 `None => {}`
        // 分支会静默吞掉），于是"接管"看起来生效了、声音却一直不更新。
        if !enabled.is_empty() {
            // 【为什么是 ensure 而不是 get_or_init】见 `preparation` 字段说明：
            // 上一份文档关闭时队列已被 close，重绑必须拿到一条**新**队列。
            self.ensure_preparation();
        }
        // 只登记weak供宿主回调排队；后台计算不形成owner自循环。
        *self.prepare_owner.lock_or_recover() = Some(weak);
        let _ = self.clock.set(document.clock.clone());
        static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);
        self.writer_id.store(
            NEXT_WRITER.fetch_add(1, Ordering::Relaxed),
            Ordering::Release,
        );
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
    pub(crate) fn arm_capture_rendering(self: &Arc<Self>) {
        self.init_block_trace();
        self.ensure_preparation();
        *self.prepare_owner.lock_or_recover() = Some(Arc::downgrade(self));
        // 时钟：Capture 没有 ARA 文档的共享时钟，自己起一个。宿主播放位置由
        // `audio_process_inner` 的 `observe_transport` 照常写入（它与 ARA 无关）。
        let _ = self
            .clock
            .set(Arc::new(super::transport::TransportClock::default()));
        static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);
        self.writer_id.store(
            NEXT_WRITER.fetch_add(1, Ordering::Relaxed),
            Ordering::Release,
        );
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
        if self.capture.has_material() {
            if let Ok(session) = self.clone().capture_session() {
                // 让 actor 下一轮加载时间线并排一次渲染（幂等：已加载时它只推票据）。
                session.request_reload();
            }
        }
    }

    pub fn prepare(&self) {
        // 【B0】先报告上一批可疑块。放在早退**之前**：Capture 实例在没有素材时
        // `renders()` 为假，而那正是"刚录完、播放中"这个最需要读数的时刻。
        self.log_click_report();
        self.log_capture_report();
        if !self.renders() {
            return;
        }
        // 源/assignment撤销在文档事务内清理；此函数只排队，不能在事务外clear旧任务刚发布的音频。
        #[cfg(test)]
        self.prepare_requests.fetch_add(1, Ordering::AcqRel);
        let weak = self.prepare_owner.lock_or_recover().clone();
        let Some(weak) = weak else {
            return;
        };
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
    /// 仅SDK规定UI线程的kOffline setup调用；不在事务锁内等待，不把旧快照当最新版本。
    pub(crate) fn prepare_offline_until(&self, deadline: std::time::Instant) -> Result<(), String> {
        // 【为什么 Capture 要在这里同步准备】bounce / 导出是 Capture 的**第二条通路**，
        // 而它是宿主跑得最快的一次 —— 插件没有第二次机会。这里必须同步把结果准备好，
        // 否则产物里是静音。`deadline` 对它无意义：没有文档事务可等，合成本身有界。
        // 【为什么判据与 `prepare_job` 同一份】离线只跑一趟，没有第二次机会，所以这里
        // 更不能因为"窗口没开过 ⇒ 形态未落定"就把静音烘焙进产物。与实时通路共用
        // "未绑定 + 有素材"这条证据（理由见 `prepare_job` 的同名分支）。
        if !self.is_bound() && self.capture.has_material() {
            return self.capture_prepare_job(Arc::new(std::sync::atomic::AtomicBool::new(false)));
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
                self.prepare();
            }
            worker.wait_idle_until(deadline)?;
        }
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
        let mut snapshots = input.render(cancel.clone())?;
        super::snapshot::compact_prepared(&mut snapshots)?;
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
            // 跨轨接受新参数不一定有新的model callback；丢弃后补排最新任务，不能永久留空且假idle。
            self.prepare();
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
                .publish(snapshot)
                .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
        }
        self.record_prepared(model, edit, epoch, scope, keys.clone());
        log::info!(
            "[ara] background snapshot ready role={} revision={edit} model={model} regions={}",
            self.role.load(Ordering::Relaxed),
            keys.len()
        );
        Ok(())
    }
}
