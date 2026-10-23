//! 宿主适配器绑定、region → item 几何求解、清单刷新与退避。
use super::*;
impl ExtensionOwner {
    /// 组件终止独立于Arc尚存与否，旧native entry不能维持可编辑授权。
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    /// 已绑定且实例仍存活时的扩展实例指针。
    ///
    /// 【用途】宿主重复 `bindDocumentToExtension` 时，companion 层会先拒绝第二次绑定
    /// （`companion processor is already bound`），`ara_entry::bind` 据此**幂等复用**
    /// 既有实例而不是返回 null —— 否则宿主认为插件未绑定、插件完全无法加载。这里只
    /// 暴露"复用"所需的只读指针，**不改绑定语义**。
    ///
    /// 【为什么还需要 `_for(document)`】只检查 `is_closed()` 会漏掉更常见的一种
    /// 顺序：宿主先绑新文档、旧文档稍后才关闭。此时实例仍绑在旧文档上，
    /// `existing_binding_raw()` 会把**旧文档的扩展实例指针**交回给新文档 ——
    /// 宿主随后对新文档调用 `addPlaybackRegion`，却落在旧文档的 observer 上。
    /// 调用方必须比对文档身份；`existing_binding_raw()` 只保留给"同文档重复绑定"
    /// 的诊断/测试路径。
    pub(crate) fn existing_binding_raw(
        &self,
    ) -> Option<*const ara2_bridge::sys::ARAPlugInExtensionInstance> {
        if self.is_closed() {
            return None;
        }
        self.binding
            .lock_or_recover()
            .as_ref()
            .map(ara2_bridge::plugin::ExtensionBinding::as_raw)
    }
    /// 仅当"未关闭、且确实绑定在 `document` 上"时复用既有扩展实例。
    pub(crate) fn existing_binding_raw_for(
        &self,
        document: super::ownership::DocumentId,
    ) -> Option<*const ara2_bridge::sys::ARAPlugInExtensionInstance> {
        if self.is_closed()
            || self.bound_document_id.load(Ordering::Acquire) != document
            || self.bound_document_id.load(Ordering::Acquire) == 0
        {
            return None;
        }
        self.existing_binding_raw()
    }
    /// 当前绑定的**活**文档 id；已关闭/从未绑定为 `None`。
    pub(crate) fn bound_document_id(&self) -> Option<super::ownership::DocumentId> {
        let id = self.bound_document_id.load(Ordering::Acquire);
        (id != 0).then_some(id)
    }
    /// 是否还持有一份扩展绑定（不区分它是哪份文档）。
    pub(crate) fn has_binding(&self) -> bool {
        self.binding.lock_or_recover().is_some()
    }
    /// 本实例**是否绑定过** ARA 文档（历史事实，不随文档关闭回落）。
    ///
    /// 【为什么 ara_entry 需要它】companion 层的"already bound"既可能是"宿主重复绑定
    /// 同一文档"（幂等复用即可），也可能是"宿主重开工程、同一实例要求绑新文档"
    /// （必须复位 companion 再绑）。前者复用，后者复位 —— 两种处理不能互换：
    /// 复用会把已销毁文档的绑定交给宿主，复位则会把还在用的绑定拆掉。
    pub(crate) fn was_bound_before(&self) -> bool {
        self.bound_once.load(Ordering::Acquire)
    }
    /// 实时流此刻是否在推进（供"内容更新要不要撤渲染许可"判断）。
    pub(crate) fn realtime_flow_active(&self) -> bool {
        self.realtime_flow.load(Ordering::Acquire)
    }
    /// 读游标接缝读数（诊断线程读；音频线程只写原子）。
    ///
    /// 【为什么要有它】"起播 0.4~0.7 s 一次咔"最直接的量是"读位置跳了多少帧"。
    /// 没有这一行时，判断只能靠听感与代码推理；有了它，一次复现就能回答
    /// "跳了几次、最大多少、宿主纠正过几次"。
    pub(crate) fn seam_summary(&self) -> String {
        let geometry = &self.streams[crate::capture::recorder::ProcessStream::Realtime.index()];
        format!(
            "seam(pos={} corr={} max={} jumps={} last_jump={} counted={} cov(holes={} frames={}))",
            self.rt_last_read_position.load(Ordering::Acquire),
            self.position_corrections.load(Ordering::Relaxed),
            self.max_correction_frames.load(Ordering::Relaxed),
            geometry.jumps.load(Ordering::Acquire),
            geometry.last_jump.load(Ordering::Acquire),
            geometry.counted_blocks.load(Ordering::Acquire),
            self.tape_holes.load(Ordering::Relaxed),
            self.tape_hole_frames.load(Ordering::Relaxed),
        )
    }
    /// 记录一次 companion 绑定复位（见 [`Self::companion_reset_count`]）。
    pub(crate) fn note_companion_reset(&self) {
        self.companion_resets.fetch_add(1, Ordering::Relaxed);
    }
    /// companion 绑定复位的累计次数（诊断）。
    pub(crate) fn companion_reset_count(&self) -> u64 {
        self.companion_resets.load(Ordering::Relaxed)
    }
    /// 记录"有一份持久化状态待恢复"。
    pub(crate) fn note_restore_pending(&self) {
        // 已经从"失败"进入"重试中"时保留错误原文，直到真正成功才清 ——
        // 用户看到的应当是"仍在尝试（上次失败原因：…）"。
        self.restore_status
            .store(RESTORE_PENDING, Ordering::Release);
    }
    /// 记录"仍在等一个明确的时机，不是失败"。
    ///
    /// 【为什么与 `note_restore_pending` 分开】pending 只说明"有载荷"；这一条
    /// 回答"在等什么"（时间线未发布 / 轨道为空 / fade 元数据未就绪）。
    pub(crate) fn note_restore_waiting(&self, reason: &str) {
        self.restore_status
            .store(RESTORE_PENDING, Ordering::Release);
        *self.restore_pending_reason.lock_or_recover() = Some(reason.to_owned());
    }
    /// 记录"持久化状态已成功合并"。
    pub(super) fn note_restore_applied(&self) {
        self.restore_status
            .store(RESTORE_APPLIED, Ordering::Release);
        *self.restore_error.lock_or_recover() = None;
        *self.restore_pending_reason.lock_or_recover() = None;
        self.restore_applied_generation
            .fetch_add(1, Ordering::Relaxed);
    }
    /// 记录"持久化状态合并失败"。
    pub(crate) fn note_restore_failed(&self, error: &str) {
        self.restore_status.store(RESTORE_FAILED, Ordering::Release);
        *self.restore_error.lock_or_recover() = Some(error.to_owned());
        *self.restore_pending_reason.lock_or_recover() = None;
    }
    /// 持久化状态的恢复读数：`(status, error, applied_generation)`。
    pub(crate) fn restore_report(&self) -> (u8, Option<String>, u64) {
        (
            self.restore_status.load(Ordering::Acquire),
            self.restore_error.lock_or_recover().clone(),
            self.restore_applied_generation.load(Ordering::Relaxed),
        )
    }
    /// 恢复状态的语言无关名（前端按它查 catalog）。
    pub(crate) fn restore_status_name(&self) -> &'static str {
        match self.restore_status.load(Ordering::Acquire) {
            RESTORE_APPLIED => "applied",
            RESTORE_PENDING => "pending",
            RESTORE_FAILED => "failed",
            _ => "never",
        }
    }
    /// 恢复仍挂着时的可读原因（成功/失败/无载荷时为 `None`）。
    pub(crate) fn restore_pending_reason(&self) -> Option<String> {
        self.restore_pending_reason.lock_or_recover().clone()
    }
    /// 清空本实例的恢复读数（重绑到新文档时用）。
    pub(super) fn reset_restore_report(&self) {
        self.restore_status.store(RESTORE_NEVER, Ordering::Release);
        *self.restore_error.lock_or_recover() = None;
        *self.restore_pending_reason.lock_or_recover() = None;
        self.restore_applied_generation.store(0, Ordering::Relaxed);
    }
    /// 状态是否落在"失败"档（供 `HostAudioState` 分类）。
    pub(crate) fn restore_failed(&self) -> bool {
        self.restore_status.load(Ordering::Acquire) == RESTORE_FAILED
    }
    pub(crate) fn editor_document(&self) -> Result<Arc<super::document::DocumentSession>, String> {
        if self.is_closed() {
            return Err(crate::editor::errors::FX_PROCESSOR_CLOSED.error());
        }
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .filter(|document| document.is_alive())
            .ok_or_else(|| {
                // "从未绑定"与"绑定过又关闭"是两件不同的事，措辞必须分开：
                // 前者在 Cubase 上重开一万次窗口也一样（宿主根本不走 ARA 路径），
                // 后者才是"会话已结束"。共用一句会把用户引向必然失败的补救动作。
                if self.bound_once.load(Ordering::Acquire) {
                    crate::editor::errors::DOCUMENT_CLOSED.error()
                } else {
                    crate::editor::errors::ARA_NOT_BOUND.error()
                }
            })
    }
    /// 宿主最近一次编辑器视图选择（`notifySelection`），指针键的副本。
    ///
    /// 【为什么可以从任意线程读】`ExtensionState` 只在写入时持锁，这里读的是一份
    /// 已经拷好的值的克隆 —— 与 `assigned_regions()` 同一个口径。宿主的选择通知本身
    /// 不触发 `end_editing`，因此**不能**只在模型线程上采：那样它要等到下一次模型变化
    /// 才可见。编辑器 actor 每次 `ensure_loaded` 采一次即可。
    pub(crate) fn host_view_selection(
        &self,
    ) -> Option<ara2_bridge::plugin::ExtensionViewSelection> {
        if self.is_closed() {
            return None;
        }
        self.binding
            .lock_or_recover()
            .as_ref()
            .and_then(ara2_bridge::plugin::ExtensionBinding::view_selection)
    }
    /// 健康探活快照：**只读原子/弱引用，不创建会话、不取文档事务、不排队**。
    ///
    /// 【为什么需要它】启动探活此前用 `get_runtime_info`，而那条命令会走
    /// `editor_session()` + actor 队列。重开工程时 `EditorSession::new` 的初始化
    /// （以及曾经的同步临时目录清扫）可能远超 30 s IPC 超时，于是用户看到
    /// `Backend invoke failed: plugin:get_runtime_info`。探活只需要回答"会话还在不在、
    /// 形态是什么、有没有片段"，这些都能从原子读数得到。
    pub(crate) fn health_snapshot(&self) -> serde_json::Value {
        let capture = self.capture_session_if_initialized();
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        let document_alive = document
            .as_ref()
            .is_some_and(|document| document.is_alive());
        let clip_count = capture
            .as_ref()
            .map_or(0, |session| session.clip_count())
            .max(
                document
                    .as_ref()
                    .map_or(0, |document| document.editor_clip_count()),
            );
        let mode = match self.mode() {
            InstanceMode::Ara => "ara",
            InstanceMode::Capture => "capture",
            InstanceMode::Undetermined => "undetermined",
        };
        serde_json::json!({
            "ok": true,
            "closed": self.is_closed(),
            "bound": self.is_bound(),
            "bound_once": self.bound_once.load(Ordering::Acquire),
            "mode": mode,
            "document_alive": document_alive,
            "session_initialized": capture.is_some()
                || document.as_ref().is_some_and(|document| document.editor_initialized()),
            "host_kind": self.host_backend().kind().as_str(),
            "model_available": hifishifter_kernel::world_vocoder::is_available(),
            "clip_count": clip_count,
        })
    }

    /// 本实例当前是否**真的**挂在一份活的 ARA 文档上。
    ///
    /// 与 `bound_once` 的区别：那个是"曾经绑定过"的单调历史，这个是"此刻有效"。
    /// 诊断（"宿主是否走了 ARA 路径"）要的是后者。
    pub(crate) fn is_ara_bound(&self) -> bool {
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .is_some_and(|document| document.is_alive())
    }

    /// 本实例的运行形态。**无锁**，可从音频线程调用。
    ///
    /// 【为什么先看 `bound_once` 再看原子】两者本该一致，但 `bound_once` 是既有的、
    /// 已被测试覆盖的读数，而 `binding_mode` 是本轮新增的。取"任一说 ARA 就是 ARA"
    /// 的交集语义，让新字段在任何写入点漏掉时**退化成旧行为**（AraOnly），
    /// 而不是退化成 Capture —— 后者会让一个正常绑定的实例去录制自己的输出。
    pub(crate) fn mode(&self) -> InstanceMode {
        if self.bound_once.load(Ordering::Acquire) {
            return InstanceMode::Ara;
        }
        InstanceMode::from_raw(self.binding_mode.load(Ordering::Acquire))
    }

    /// 记录"宿主把本实例绑到了 ARA 文档"。
    pub(crate) fn note_ara_bound(&self) {
        self.binding_mode
            .store(InstanceMode::Ara.raw(), Ordering::Release);
        // 已打开的视图可能是在绑定之前创建的：让视图定时器补发一次 bootstrap。
        self.bootstrap_changed.store(true, Ordering::Release);
    }

    /// 取走"bootstrap 需要补发"的标记（视图定时器调用）。
    pub(crate) fn take_bootstrap_changed(&self) -> bool {
        self.bootstrap_changed.swap(false, Ordering::AcqRel)
    }

    /// 记录"绑定窗口已关闭且没有绑定" —— 本实例是 Capture 形态。
    ///
    /// 【为什么由调用方判定"窗口已关闭"而不是这里自己看时钟】判据是
    /// `setActive` 之后过了宽限期（见 [`Self::note_editor_tick`]），那需要读
    /// `set_active_at` 这个锁下的值。这里只负责**写入结论**，判定留在唯一的调用点，
    /// 避免同一判据出现两份。
    pub(crate) fn note_binding_window_closed_without_ara(&self) {
        // 已经绑过就不回落：那会把一个正常工作的 ARA 实例降级成 Capture。
        if self.bound_once.load(Ordering::Acquire) {
            return;
        }
        let _ = self.binding_mode.compare_exchange(
            InstanceMode::Undetermined.raw(),
            InstanceMode::Capture.raw(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// 记录宿主调用 `setActive(true)` 的时刻。
    ///
    /// 【为什么在这里记而不是轮询宿主】`setActive` 是 ARA 绑定窗口的**关闭点**：
    /// 契约规定绑定只能发生在它之前。把时刻记下来，UI 定时器就能在宽限期后给出
    /// 一条确定性的诊断，而不是让读者去对比两个会话的日志。
    pub(crate) fn note_component_activated(&self) {
        let mut at = self.set_active_at.lock_or_recover();
        if at.is_none() {
            *at = Some(std::time::Instant::now());
        }
    }

    /// UI 定时器每 tick 调用：宿主激活后仍无 ARA 绑定时，**一次性**给出诊断。
    ///
    /// 【为什么必须有这一条】用户报障：Cubase 里窗口能开、每个命令都失败，但日志里
    /// 没有任何一行说明"宿主根本没走 ARA"。要得出这个结论，得把 Cubase 会话与 REAPER
    /// 会话逐行对比（前者零次 `queryInterface ARA hit`、零次 `document controller
    /// created`）。一条确定性的警告把这件事变成一眼可见。
    ///
    /// 【为什么在 UI 定时器上做而不是开线程】插件 DLL 可能被宿主卸载；一个 sleep 5 秒
    /// 的分离线程会在模块卸载后回到已消失的代码。UI 定时器天然与视图同寿，且已经在
    /// 每 20ms 跑一次。
    /// 绑定窗口是否已经关闭：宿主激活过，且过了宽限期仍没有 ARA 绑定。
    ///
    /// 【为什么不能只看 UI 定时器】UI 定时器要等 WebView 起来才跑，而 bootstrap
    /// （`hostKind` / `captureCaps`）与编辑器会话都在**那之前**就要读形态 ——
    /// 于是"窗口打开的那一刻"读到的永远是 `Undetermined`，`host_backend()` 退成
    /// ARA-only，前端所有按形态分流的地方（面板、探活）在前 5 秒里全是错的。
    /// 判定因此必须能按需求值，而不是只在一个定时器里发生。
    pub(super) fn binding_window_closed(&self) -> bool {
        self.set_active_at
            .lock_or_recover()
            .is_some_and(|at| at.elapsed() >= ARA_BINDING_GRACE)
    }

    /// 在**非音频线程**上把形态判定落定（幂等，可在任何读形态之前调用）。
    ///
    /// 读 `set_active_at` 要取锁，因此绝不能在音频线程调用 —— 音频线程只读
    /// [`Self::mode`] 那个原子结论。
    pub(crate) fn resolve_binding_mode(&self) {
        if self.is_ara_bound() {
            return;
        }
        if self.binding_window_closed() {
            self.note_binding_window_closed_without_ara();
        }
    }

    pub(crate) fn note_editor_tick(self: &Arc<Self>) {
        if self.is_ara_bound() {
            return;
        }
        // 【为什么形态判定在这里落地】这里正是"过了宽限期仍未绑定"这个结论的判定点：
        // `set_active_at` 只有这里与 `binding_window_closed` 读，宽限常数只有这两处用。
        // 形态一旦确定，后端选择、能力位与前端面板都会跟着翻到 Capture（见
        // `host_backend`）。判定写在"只报一次"的早退**之前** —— 形态是事实，与是否要
        // 打日志无关。
        if !self.binding_window_closed() {
            // 宿主还没激活，或还在宽限期内：绑定窗口仍然开着，任何结论都是早的。
            return;
        }
        self.note_binding_window_closed_without_ara();
        // 【为什么这里不再做任何维护】备用缓冲与宿主时钟此前挂在 UI 定时器上，于是
        // "关掉窗口就断录"（备用缓冲 2 粒 × 8 秒 = 24 秒后 `NoSpare`）而且没开窗口的
        // 会话里宿主位置根本没有地方落。现在两者都有与窗口无关的归属：
        //   * 时钟/合成队列 → `setupProcessing`（见 `arm_capture_rendering` 的调用点）；
        //   * 备用缓冲       → 会话 actor（见 `EditorSession::run` 的 `top_up` 调用）。
        // 这里只剩"形态落定 + 一次性说明"，形态本身仍然是这个 tick 的职责 ——
        // 它是唯一同时握有 `setActive` 时刻与宽限期的地方。
        if self
            .ara_binding_absent_reported
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        let backend = self.host_backend();
        let host = backend.app_name().unwrap_or("this host");
        // 诊断里要报"激活之后过了多久"，判据本身只问"窗口关没关"（见 `binding_window_closed`）。
        let elapsed = (*self.set_active_at.lock_or_recover())
            .map(|at| at.elapsed())
            .unwrap_or_default();
        // 【为什么措辞变了、级别也变了】此前这条是 `warn` 且说"宿主把本插件当普通插入
        // 效果器运行"，并把用户引向"请从 ARA 入口插入"。现在普通插入是**受支持**的
        // 形态（Capture 通路，见 `docs/plans/2026-10-15-vst-capture-mode.md`）：
        // 音频从输入总线捕获、按工程时间对到插件时间轴、处理后输出。
        // 因此这不再是一条错误，而是一条**形态说明** —— 级别降到 `info`，
        // 内容改为"本实例走 Capture；想直接编辑宿主片段就从 ARA 入口插入"。
        log::info!(
            "[ara] no ARA binding {:.0}s after setActive: {host} is running this instance as a \
             plain insert, so this instance uses the Capture path (capture the host's playback \
             into the plug-in). Insert the plug-in from the host's ARA entry instead if you want \
             to edit the host's clips directly.",
            elapsed.as_secs_f32()
        );
    }

    /// 初始化在宿主UI/model线程保存有引用的typed扩展，不在actor或process调用REAPER API。
    /// # Safety
    /// context须为宿主初始化期间活FUnknown。
    pub(crate) unsafe fn bind_host_adapter(&self, context: *mut std::ffi::c_void) {
        // SAFETY: 调用方保证 context 是宿主初始化期间存活的 FUnknown。
        unsafe { self.bind_host_adapter_inner(context, false) }
    }

    /// 在 `setActive` 等"宿主明确进入新阶段"的入口强制重建宿主适配器。
    ///
    /// 【为什么需要强制变体】`bind_host_adapter` 在文档已绑定时要求
    /// `host_query_stamp()` 成功；工程刚加载、模型还没发布时它必然失败，于是
    /// 适配器不会被创建。而宿主可能在另一个线程上调用 `setActive`，
    /// 旧适配器的线程亲和性不再成立。强制变体只要求组件未关闭：宿主接口属于
    /// 组件，不属于文档，可以在任何阶段重建。
    ///
    /// # Safety
    /// `context` 必须是本组件持有的、在组件生命周期内有效的宿主接口。
    pub(crate) unsafe fn rebind_host_adapter(&self, context: *mut std::ffi::c_void) {
        // SAFETY: 同 `bind_host_adapter`。
        unsafe { self.bind_host_adapter_inner(context, true) }
    }

    /// 当前宿主适配器是否与调用线程不匹配（`false` = 没有适配器或线程一致）。
    pub(crate) fn host_adapter_thread_mismatch(&self) -> bool {
        self.native_host()
            .is_some_and(|host| host.bound_thread() != std::thread::current().id())
    }

    pub(super) unsafe fn bind_host_adapter_inner(
        &self,
        context: *mut std::ffi::c_void,
        force: bool,
    ) {
        let bound = self.document.lock_or_recover().is_some();
        let stamp = if bound && !force {
            match self.host_query_stamp() {
                Ok(stamp) => Some(stamp),
                Err(_) => return,
            }
        } else {
            None
        };
        let authorized = || {
            !self.is_closed()
                && match &stamp {
                    Some(stamp) => self.host_query_authorized(stamp),
                    // 非强制路径保持既有语义：初始化期间若文档已经出现，说明
                    // 宿主正在绑定，此时不抢建适配器；强制路径（setActive）则
                    // 只要求组件未关闭。
                    None => force || self.document.lock_or_recover().is_none(),
                }
        };
        let host = unsafe { crate::host::NativeHost::from_context(context, authorized) }
            .map(|host| crate::host::adapter::HostBackend::Native(Arc::new(host)));
        if !authorized() {
            return;
        }
        crate::log_line(&format!(
            "native host extension available={} (force={force}, thread={:?})",
            host.is_some(),
            std::thread::current().id()
        ));
        let old = std::mem::replace(&mut *self.host.lock_or_recover(), host);
        drop(old);
        self.host_geometry.lock_or_recover().take();
    }
    /// 短事务冻结真实owner/document/model/scope；整个host调用链都不持内部锁。
    pub(super) fn host_query_stamp(
        &self,
    ) -> Result<(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>), String> {
        let document = match self.editor_document() {
            Ok(document) => document,
            Err(error) => {
                // 【为什么这条静默早退必须留痕】它是枚举、几何刷新与前台自愈的**共同**
                // 第一行。此前它失败时什么都不写，于是"重开工程后界面停在等待宿主音频"
                // 与"宿主没给原生扩展"在日志里完全同形（T6）。
                //
                // 【为什么 Capture 实例不报】它**永远**没有 ARA 文档 —— 那不是故障，
                // 是形态。此前这一行每 10 秒报一次，40 分钟的现场日志里刷了 480 行，
                // 把真正的错误整片淹掉；而"看不到读数"正是本轮要修的第一件事。
                if self.mode() != InstanceMode::Capture {
                    hifishifter_kernel::log_limiter::emit_limited_key(
                        log::Level::Warn,
                        "ara.stamp.no-document",
                        format_args!("[ara] host query skipped (no document): {error}"),
                    );
                }
                return Err(error);
            }
        };
        let _transaction = document.transaction.lock_or_recover();
        if self.is_closed() || !document.is_alive() {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.stamp.closed",
                format_args!(
                    "[ara] host query skipped: owner closed={} document alive={}",
                    self.is_closed(),
                    document.is_alive()
                ),
            );
            return Err("host query document/owner closed".into());
        }
        let keys = match self.host_assigned_regions(&document) {
            Ok(keys) => keys,
            Err(error) => {
                // 【为什么单独一条 key】`assignments` 为空是本轮症状二/三最可疑的
                // 静默早退点（见 `bind_to_document` 的 rebind 分支）。把它与
                // "文档已关闭"分开计数，真机日志才能直接指认。
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.stamp.no-assignment",
                    format_args!("[ara] host query skipped (no usable assignment): {error}"),
                );
                return Err(error);
            }
        };
        let model = document.revision.load(Ordering::Acquire);
        let scope = document.scope_revision.load(Ordering::Acquire);
        drop(_transaction);
        Ok((document, model, scope, keys))
    }
    pub(super) fn host_query_authorized(
        &self,
        stamp: &(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>),
    ) -> bool {
        let (document, model, scope, keys) = stamp;
        let _transaction = document.transaction.lock_or_recover();
        !self.is_closed()
            && document.is_alive()
            && self
                .editor_document()
                .is_ok_and(|current| Arc::ptr_eq(&current, document))
            && document.revision.load(Ordering::Acquire) == *model
            && document.scope_revision.load(Ordering::Acquire) == *scope
            && self
                .host_assigned_regions(document)
                .is_ok_and(|current| current == *keys)
    }
    /// 本实例这一轮应当看到哪些 region —— **全仓库唯一的判据**（T1）。
    ///
    /// 【为什么必须收敛成一个函数】此前有两个各写一遍的实现：`host_assigned_regions`
    /// （inventory / 几何 / 传输的入口，round 17 给它加了"文档级派生"兜底）与
    /// `assigned_regions`（**时间线**与**渲染输入**的入口，没有兜底）。两者在同一个
    /// tick 里给出**互相矛盾**的结论：几何刷新说"这个 item 我认得"，工作区投影说
    /// "这条 region 不归你"。界面于是显示"有 item、有轨道、但 clip 没内容"。
    ///
    /// 【三条纪律】
    /// 1. **逐条过滤，不整批作废**：宿主在一次分割里会把已销毁父段的 key 一并重发
    ///    （`workspace_scope_locked` 的注释早已记录这个形状），整批 `resolve` 会让
    ///    **一条**陈旧 key 把**整份**工作区打成空。
    /// 2. **空集合要区分"宿主没给"与"宿主给了空的"**：前者回退文档级，后者才是真空。
    /// 3. **不可用不是错误**：返回 `AssignedScope`（带来源），由调用方决定怎么呈现。
    ///
    /// 【`allow_document_derived` 为什么是参数而不是常量】只有**并集语义**的调用方
    /// 可以派生：`workspace_scope_locked` 本来就是"本文档全部 owner 的并集"，
    /// 而 REAPER 的一个 ARA 文档上同时挂着多条轨道的实例，派生出来的并集正是它要的。
    /// **渲染输入不能派生** —— 每个实例只许渲染自己的分配，否则同一段音频会被渲染两次。
    /// `role` 为 `None` 时取**全部角色**的并集（枚举/几何路径：它们服务整条轨道），
    /// `Some(bits)` 时只取该角色（**渲染路径**：playback 实例只许渲染自己那一份）。
    pub(crate) fn assigned_scope(
        &self,
        document: &super::document::DocumentSession,
        role: Option<u32>,
        allow_document_derived: bool,
    ) -> Result<AssignedScope, String> {
        let pick = |map: &std::collections::HashMap<i32, Vec<u64>>| -> BTreeSet<u64> {
            match role {
                Some(bits) => map
                    .get(&(bits as i32))
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                None => map.values().flatten().copied().collect(),
            }
        };
        let mut keys = {
            let assignments = self.assignments.lock_or_recover();
            pick(&assignments)
        };
        let declared = {
            let sequences = self.sequences.lock_or_recover();
            pick(&sequences)
        };
        let members = document.sequence_regions.lock_or_recover();
        // 【为什么两者都空才派生】只要宿主给过任何一条线索（region key 或 sequence），
        // 就以它为准 —— 派生只补"本实例还没被告知"，不覆盖"已经被告知为空"。
        let derived = keys.is_empty() && declared.is_empty() && allow_document_derived;
        let mut source = if derived {
            ScopeSource::DocumentDerived
        } else {
            ScopeSource::HostAssignment
        };
        if derived {
            for regions in members.values() {
                keys.extend(regions.iter().copied());
            }
        } else {
            for sequence in declared {
                // 【为什么未知 sequence 只跳过】宿主先建 sequence、后建 region 是正常
                // 时序，其间该键必然缺席。此前 `ok_or(...)?` 让整份 scope 失败。
                if let Some(regions) = members.get(&sequence) {
                    keys.extend(regions.iter().copied());
                }
            }
        }
        drop(members);
        // 【纪律 1：逐条过滤】`resolve` 是**整批**语义，一条坏 key 就整体 Err。
        // 写路径（assignment observer）早已改成逐条 `owner_of`，读路径必须同口径，
        // 否则同一次分割在两处得到不同结论。
        let mut dropped = 0usize;
        let keys = {
            let owners = region_owners().lock_or_recover();
            keys.into_iter()
                .filter(|key| {
                    if owners.owner_of(*key) == Some(document.id) {
                        true
                    } else {
                        dropped += 1;
                        false
                    }
                })
                .collect::<Vec<_>>()
        };
        if dropped > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.scope.dropped",
                format_args!(
                    "[ara] {dropped} assigned region key(s) are not owned by this document \
                     (destroyed, or re-sent by the host after a split); the rest of the scope \
                     is unaffected"
                ),
            );
        }
        if keys.is_empty() {
            source = ScopeSource::Empty;
        }
        Ok(AssignedScope { keys, source })
    }

    /// 几何绑定必须覆盖owner的全部已分配角色，不能只拿playback子集隐藏editor歧义。
    pub(super) fn host_assigned_regions(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<Vec<u64>, String> {
        // 枚举 / 几何 / 传输三条路径都是**读**：它们可以接受文档级派生（见
        // `assigned_scope` 的说明），但绝不能因为一条陈旧 key 就整份失败。
        Ok(self.assigned_scope(document, None, true)?.keys)
    }
    /// 仅UI/model线程读取；唯一assignment先成立，位置/长度仅用于绑定后相容核对。
    ///
    /// 这条路径的身份来自宿主直接 parent take —— 是强证据，因此**只在恰好一个
    /// region 时**成立。多 region（folder 轨上的 FX）走
    /// [`Self::host_geometries`]，那里没有直接 parent 可用，只能按几何唯一匹配。
    pub(crate) fn host_geometry(
        &self,
    ) -> Result<crate::host::geometry::BoundHostGeometry, GeometryFailure> {
        let stamp = self.host_query_stamp()?;
        let [region_key] = stamp.3.as_slice() else {
            return Err("host geometry requires exactly one assigned ARA region".into());
        };
        let region = stamp
            .0
            .regions
            .lock_or_recover()
            .get(region_key)
            .cloned()
            .ok_or("assigned ARA region unavailable")?;
        let host = self.native_host().ok_or("host extension unavailable")?;
        let geometry = host.geometry(|| self.host_query_authorized(&stamp))?;
        if !compatible(geometry.start_sec, region.start_in_playback_time)
            || !compatible(geometry.duration_sec, region.duration_in_playback_time)
        {
            return Err(GeometryFailure::Structural(BINDING_GEOMETRY_INCOMPATIBLE));
        }
        if !self.host_query_authorized(&stamp) {
            return Err("host geometry authorization revoked".into());
        }
        Ok(crate::host::geometry::BoundHostGeometry {
            region_key: *region_key,
            geometry,
        })
    }

    /// 多 region owner（挂在 REAPER folder 轨上的 FX）逐 region 解析宿主几何。
    ///
    /// 【为什么不能退回"取第一个"】后续所有宿主写回都沿 region → item 这条边定位
    /// 对象。拿一个 take 的几何去冒充整组，会把 item GUID 绑到**错误的** region 上，
    /// 于是移动/切分/fade/音量全部可能落到别的 item。因此这里对每个 region 单独
    /// 求唯一匹配，匹配不到就**不绑定**，并在日志里如实报出未绑定的数量。
    ///
    /// 候选 item 与 GUI 清单同源（FX 自己的轨道 + folder 全部后代轨道），沿既有
    /// `ui_folder_tracks` 的线程/授权/预算纪律取得。
    pub(super) fn host_geometries(
        &self,
    ) -> Result<Vec<crate::host::geometry::BoundHostGeometry>, GeometryFailure> {
        let stamp = self.host_query_stamp()?;
        if stamp.3.is_empty() {
            return Err("no assigned ARA region".into());
        }
        let host = self.native_host().ok_or("host extension unavailable")?;
        let authorized = || self.host_query_authorized(&stamp);
        let candidates = host
            .ui_folder_tracks(&authorized)?
            .into_iter()
            .flat_map(|track| track.items.into_iter().map(|item| item.geometry))
            .collect::<Vec<_>>();
        let mut bound = Vec::new();
        let mut unmatched = Vec::new();
        for key in &stamp.3 {
            let region = stamp
                .0
                .regions
                .lock_or_recover()
                .get(key)
                .cloned()
                .ok_or("assigned ARA region unavailable")?;
            // 上一次绑定：给几何全等的候选（复制出来的安全副本等）一个稳定性偏好。
            let preferred = stamp.0.region_items.lock_or_recover().get(key).cloned();
            match unique_geometry_for_region(&region, &candidates, preferred.as_deref()) {
                Some(geometry) => bound.push(crate::host::geometry::BoundHostGeometry {
                    region_key: *key,
                    geometry,
                }),
                None => unmatched.push(*key),
            }
        }
        // 一个宿主 item 只能被**一个** region 认领。两条 region 绑到同一个 item 会让
        // 后续所有写回指向同一对象：移动、切分、fade、音量互相覆盖，且没有任何一侧
        // 会报错。
        //
        // 【为什么不再"全部丢弃"】旧实现把重复认领的候选**全丢**，于是"同一位置两个
        // 几何全等的 item"会两个都不绑定、两个 clip 一起消失（用户报障："片段凭空
        // 没了"）。现在按 region key 的**稳定顺序**保留第一个认领者，其余照常记账 ——
        // 至少不会一次丢掉全部；剩下的那个 region 仍走 unmatched，如实报出。
        let mut seen = std::collections::HashSet::<String>::new();
        let claimed = bound.len();
        bound.retain(|entry| seen.insert(entry.geometry.item_id.clone()));
        // 一次说清、且只在读数变化时输出（见 `report_binding`）。此前这里是两条无条件
        // 日志（INFO + WARN），在 folder 父轨场景下以 10Hz 重放 35 分钟共 16,769 条。
        self.report_binding(&BindingReport {
            assigned: stamp.3.clone(),
            bound: bound.iter().map(|entry| entry.region_key).collect(),
            unmatched: unmatched.clone(),
            dropped: claimed.saturating_sub(bound.len()),
            candidates: candidates.len(),
        });
        if bound.is_empty() {
            return Err(GeometryFailure::Structural(BINDING_NO_UNIQUE_ITEM));
        }
        if !self.host_query_authorized(&stamp) {
            return Err("host geometry authorization revoked".into());
        }
        Ok(bound)
    }

    /// 报告一次 region → 宿主 item 的绑定读数；**只在读数变化时**输出。
    ///
    /// 【为什么必须去重】见 [`BindingReport`]：folder 父轨上的"2 个 region 对 1 个候选"
    /// 是结构性不匹配，100ms 内不会自愈。无条件日志会跟着 UI 定时器无限重放。
    ///
    /// 【为什么还要限流兜底】指纹是"变化检测"，限流是"变化速率上限"。两者叠加覆盖两种
    /// 失败模式：指纹包含会漂移的字段 ⇒ 变化检测失效，由限流兜住；读数在两个值之间
    /// 高频抖动 ⇒ 变化检测每次都为真，同样由限流兜住（键含指纹，两个值各自计数）。
    /// 最坏情况是 10 秒一条，与"刷屏"已经不沾边。
    pub(super) fn report_binding(&self, report: &BindingReport) {
        if !self.binding_report_is_new(report) {
            return;
        }
        let detail = format!(
            "{} assigned, {} bound, {} unmatched, {} dropped (candidates={})",
            report.assigned.len(),
            report.bound.len(),
            report.unmatched.len(),
            report.dropped,
            report.candidates
        );
        // 键取**读数本身**：不同读数各自限流，因此真实变化不会被别的读数挤掉。
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            &format!("ara:binding-report:{detail}"),
            format_args!("[ara] region→item binding: {detail}"),
        );
    }

    /// 这份读数是否**尚未报告过**；是则记下它。纯状态判定，可独立单测。
    ///
    /// 【为什么与输出分开】去重逻辑是这条修复的核心（它决定"刷屏"会不会回来），
    /// 必须能脱离日志后端被断言 —— 与 `log_limiter::limited_decision` 同一手法。
    pub(super) fn binding_report_is_new(&self, report: &BindingReport) -> bool {
        // 全部绑上 = 没有可抱怨的读数。顺手清掉指纹，让下一次真实失败**必然**输出
        // （否则"失败 → 成功 → 同样的失败"会被去重吞掉第二条）。
        if report.unmatched.is_empty() && report.dropped == 0 {
            self.binding_report.lock_or_recover().take();
            return false;
        }
        let mut last = self.binding_report.lock_or_recover();
        if last.as_ref() == Some(report) {
            return false;
        }
        *last = Some(report.clone());
        true
    }

    /// 当前的重枚举间隔：无失败时是 [`HOST_ENUMERATION_INTERVAL`]，连续结构性失败时退避。
    ///
    /// 【为什么用绝对档位而不是乘法】档位表要能一眼看出"最坏情况多久枚举一次"。乘法
    /// 形式（100ms × 2^n）在 n 大时会越过 4 秒上限，得再补一次钳制 —— 不如直接查表。
    pub(super) fn enumeration_interval(&self) -> std::time::Duration {
        match self
            .enumeration_backoff
            .load(std::sync::atomic::Ordering::Acquire)
        {
            0 => HOST_ENUMERATION_INTERVAL,
            1..=2 => std::time::Duration::from_millis(250),
            3..=5 => std::time::Duration::from_secs(1),
            _ => std::time::Duration::from_secs(4),
        }
    }

    /// 按一次几何求解的结果更新退避档位（见 [`ExtensionOwner::enumeration_interval`]）。
    pub(super) fn note_geometry_outcome(
        &self,
        geometry: &Result<Vec<crate::host::geometry::BoundHostGeometry>, GeometryFailure>,
        revision: Option<i32>,
    ) {
        let Err(error) = geometry else {
            self.enumeration_backoff
                .store(0, std::sync::atomic::Ordering::Release);
            *self.last_failed_revision.lock_or_recover() = None;
            return;
        };
        if !error.is_structural() {
            // 授权撤销 / 文档关闭 / 扩展接口消失：状态变化，不是结构性问题。
            // 退避它们只会拖慢恢复，且它们本来就不会以 10Hz 重放。
            return;
        }
        let mut last = self.last_failed_revision.lock_or_recover();
        if *last != Some(revision) {
            // 宿主几何代次变了（用户真的动了工程）⇒ 回到全速，本次不计档。
            // 【为什么不在这里也 +1】那会让"改完工程"的第一次重试落在 250ms 档，
            // 而不是 100ms —— 退避不得牺牲响应速度。
            *last = Some(revision);
            self.enumeration_backoff
                .store(0, std::sync::atomic::Ordering::Release);
            return;
        }
        let _ = self.enumeration_backoff.fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |level| Some(level.saturating_add(1)),
        );
    }

    /// actor/worker只可消费UI冻结的Rust值，不沿此访问器调用host；代次变化明确不可用。
    // 只读几何元数据访问器保留，供worker消费冻结值。
    #[cfg(test)]
    pub(crate) fn host_geometry_metadata(
        &self,
    ) -> Result<Vec<crate::host::geometry::BoundHostGeometry>, String> {
        let stamp = self.host_query_stamp()?;
        let cached = self
            .host_geometry
            .lock_or_recover()
            .clone()
            .ok_or("host geometry has not been sampled on model/UI thread")?;
        if cached.model != stamp.1 || cached.scope != stamp.2 || !self.host_query_authorized(&stamp)
        {
            return Err("host geometry metadata superseded".into());
        }
        cached.value.map_err(GeometryFailure::into_message)
    }
    /// 调用者已持所属doc事务，只消费已授权Rust缓存，不嵌套加锁或调用宿主。
    ///
    /// 返回**当前仍被分配**的 region 各自的绑定；不再要求"恰好一个" —— folder 轨上
    /// 的 FX 会被分配整组 region，要求唯一等于让整条投影链（fade / mute / item 身份）
    /// 全部静默失效。仍然只返回 region 与缓存都同意的那部分：缓存过期、region 已撤销、
    /// 或该 region 没匹配到唯一 item 的，一律不出现。
    pub(crate) fn host_geometries_locked(
        &self,
        document: &super::document::DocumentSession,
    ) -> Vec<crate::host::geometry::BoundHostGeometry> {
        if self.is_closed()
            || !document.is_alive()
            || !self
                .editor_document()
                .is_ok_and(|current| std::ptr::eq(Arc::as_ptr(&current), document))
        {
            return Vec::new();
        }
        let Some(cached) = self.host_geometry.lock_or_recover().clone() else {
            return Vec::new();
        };
        if cached.model != document.revision.load(Ordering::Acquire)
            || cached.scope != document.scope_revision.load(Ordering::Acquire)
        {
            return Vec::new();
        }
        let Ok(keys) = self.host_assigned_regions(document) else {
            return Vec::new();
        };
        match cached.value {
            Ok(bound) => bound
                .into_iter()
                .filter(|entry| keys.contains(&entry.region_key))
                .collect(),
            Err(_) => Vec::new(),
        }
    }
    /// 最近一次清单刷新得出的宿主音频读数；GUI 载荷读它。
    ///
    /// 只读快照，不触发宿主调用 —— 与 `host_geometries_locked` 同一纪律：调用方已经
    /// 在事务里，这里不得再枚举宿主。
    pub(crate) fn host_audio_status(&self) -> HostAudioStatus {
        *self.host_audio.lock_or_recover()
    }
    /// 完整刷新：时钟 + 几何 + 清单。写回路径与测试用这个（必须**同步**生效）。
    pub(crate) fn refresh_host_transport(&self) {
        self.refresh_host_transport_inner(false);
    }

    /// 前台恢复/重新激活时主动重同步一次。
    ///
    /// 【为什么不能等 UI 定时器】后台期间定时器可能停摆；回来后第一次 tick 仍会
    /// 命中"令牌未变 + 1 s 内采样过"的早退，`ui_tracks` 与 `region_items` 可能是
    /// 暂停前的快照。这里清掉采样时刻，强制一次完整枚举，并让每个 renderer 重排
    /// 快照 —— 用户切回前台后不必再等下一次编辑才恢复。
    pub(crate) fn resume_resync(&self) {
        if let Ok(document) = self.editor_document() {
            *document.ui_inventory_sampled_at.lock_or_recover() = None;
        }
        self.refresh_host_geometry();
        self.refresh_ui_inventory();
        if let Ok(document) = self.editor_document() {
            for owner in document.renderer_owners() {
                owner.prepare();
            }
        }
    }

    /// UI 定时器路径：时钟每次跟，几何/清单按 [`ExtensionOwner::enumeration_interval`] 节流。
    ///
    /// 【为什么时钟与枚举分开节流】定时器是 20ms（50Hz）。播放光标必须跟得上
    /// （`host.sample` 很便宜），但几何/清单枚举即使有代次缓存，每 tick 仍要付
    /// `GetProjectStateChangeCount` 等若干宿主调用，还会与 actor/渲染线程争同一把
    /// document 事务。宿主真变了时下一次节流点会捕捉到（≤100ms），用户不可感。
    ///
    /// 【为什么节流间隔是动态的】连续的结构性绑定失败（folder 父轨上"2 个 region 对
    /// 1 个候选"）不会在 100ms 内自愈，却会以这个节奏完整枚举宿主工程。退避把这种
    /// "永不成功"的重试降到 4 秒一次，而宿主几何代次一变就立刻回到 100ms。
    ///
    /// 【为什么不把节流放进 `refresh_host_transport` 本身】写回路径（导入/分割/
    /// 剪贴板）与测试依赖它**同步**把新几何取回来；节流在那里会把"刚写完就读到新值"
    /// 变成"下次再看"。所以节流只属于定时器这一个调用方。
    pub(crate) fn refresh_host_transport_tick(&self) {
        // 【T5：前台恢复的本地信号】编辑器窗口是 `WS_CHILD`，收不到 `WM_ACTIVATEAPP`；
        // 而 `resume_resync()` 的唯一调用点是 `setActive(true)`，宿主切窗口焦点时不会重发。
        // 于是"后台→前台"没有任何入口。这里用**定时器自身的间隔**补上：20 ms 的定时器
        // 被挂起/降频后，恢复后的第一次 tick 与上一次之间的间隔必然远大于一个周期。
        // 它不需要任何宿主配合，也不改变线程模型。
        let resumed = {
            let mut last = self.last_view_tick.lock_or_recover();
            let now = std::time::Instant::now();
            let resumed = last.is_some_and(|at| now.duration_since(at) >= FOREGROUND_RESUME_GAP);
            *last = Some(now);
            resumed
        };
        if resumed {
            log::info!(
                "[ara] the view timer resumed after a long gap; resyncing host geometry and inventory"
            );
            self.resume_resync();
        }
        // 【T4：自愈必须先于 gating】`refresh_host_transport_inner` 的第一行是
        // `host_query_stamp()` 的早退，而自愈原本写在它**之后**。恰恰在重开工程 /
        // `assignments` 为空时 stamp 拿不到 —— 于是最需要自愈的场景一行都不执行，
        // 且完全沉默（这是"round 16 加了自愈却仍然复现"的直接原因）。
        //
        // "我们多久没成功采过样"不需要任何授权：它是本侧的时钟事实。把"记录需求"
        // 与"执行动作"分开之后，随后的枚举仍会被原来的 gating 拦下（正确），
        // 但恢复需求已经被记下来了，下一次 stamp 可用时会真正执行。
        self.heal_stale_inventory_stamp();
        // 【T23】视图定时器是"宿主不再回调"时唯一还会跑的节拍：在这里提出收口请求。
        self.request_settle_if_overdue();
        self.refresh_host_transport_inner(true);
    }

    /// **T23**：把"宿主不再收口"这件事从插件侧提出来。
    ///
    /// 【为什么需要它】`ready` 回到真只依赖宿主发起的四个入口。宿主一旦 `beginEditing`
    /// 之后再也不收口，或者积压了模型变化却不再调 `notifyModelUpdates`，插件就永久
    /// 停在未就绪 —— 用户唯一的出路是删掉插件再加。
    ///
    /// 本函数只**置位**（文档上的两个原子量），真正的收口由 ARA 控制器在它下一次
    /// 回调里执行（见 `ControllerAdapter::service_forced_settle`）。桥的 `PluginRuntime`
    /// 不是 `Send`/`Sync`，从这条线程直接调用它是未定义行为。
    ///
    /// 【为什么用 `try_lock`】音频线程也会调它。拿不到文档槽就跳过 —— 下一次块
    /// （10 ms 后）会再来，而"宽限期"本来就是秒级，漏一拍没有影响。
    pub(crate) fn request_settle_if_overdue(&self) {
        let Ok(slot) = self.document.try_lock() else {
            return;
        };
        let document = slot.as_ref().and_then(std::sync::Weak::upgrade);
        drop(slot);
        let Some(document) = document else {
            return;
        };
        document.request_settle_if_overdue();
    }

    /// 距上次成功采样超过 `STALE_SAMPLE_GRACE` 就清掉采样时刻，让下一次真正执行强制重采。
    ///
    /// 【为什么它不取 `host_query_stamp`】那条路径会因为"没有 document / 没有分配 /
    /// 已关闭"而失败，而这三者正是"后台恢复后界面不刷新"的常见形状。时钟维护
    /// 不能依赖它们。
    pub(super) fn heal_stale_inventory_stamp(&self) {
        let Ok(document) = self.editor_document() else {
            return;
        };
        let stale = document
            .ui_inventory_sampled_at
            .lock_or_recover()
            .is_none_or(|at| at.elapsed() >= STALE_SAMPLE_GRACE);
        if stale {
            *document.ui_inventory_sampled_at.lock_or_recover() = None;
        }
    }

    pub(super) fn refresh_host_transport_inner(&self, throttled: bool) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let host = self.native_host();
        if let Some(host) = host {
            if let Ok((position, playing)) = host.sample(|| self.host_query_authorized(&stamp)) {
                // getter可能重入关闭/改scope，发布与撤销共用最终短事务。
                let _transaction = stamp.0.transaction.lock_or_recover();
                if !self.is_closed()
                    && stamp.0.is_alive()
                    && stamp.0.revision.load(Ordering::Acquire) == stamp.1
                    && stamp.0.scope_revision.load(Ordering::Acquire) == stamp.2
                {
                    if let Some(clock) = self.clock.get() {
                        clock.publish_host(position, playing);
                    }
                }
            }
        }
        if !self.host_query_authorized(&stamp) {
            return;
        }
        // 【前台恢复不依赖 setActive】后台期间定时器可能停摆；超过一个刷新周期
        // 没有成功采样过，就强制一次完整枚举（清掉采样时刻，让 `refresh_ui_inventory`
        // 跳过"令牌未变"的早退），并跳过实例级节流。强制重采在事实未变时无副作用
        // （见 `refresh_ui_inventory` 的 `token_changed` 分支）。
        let stale_sample = stamp
            .0
            .ui_inventory_sampled_at
            .lock_or_recover()
            .is_none_or(|at| at.elapsed() >= STALE_SAMPLE_GRACE);
        if stale_sample {
            *stamp.0.ui_inventory_sampled_at.lock_or_recover() = None;
        }
        if throttled && !stale_sample {
            let mut last = self.last_enumeration.lock_or_recover();
            let now = std::time::Instant::now();
            if last.is_some_and(|last| now.duration_since(last) < self.enumeration_interval()) {
                return;
            }
            *last = Some(now);
        } else if throttled {
            *self.last_enumeration.lock_or_recover() = Some(std::time::Instant::now());
        }
        self.refresh_host_geometry();
        // editor-only通常没有所属take；一个工程GUI必须驱动真正隐藏playback实例的只读采集。
        // renderer_owners仅返回本真实文档的活租约，不扫描全局实例，也不借名称/位置找take。
        for owner in stamp.0.renderer_owners() {
            if !self.host_query_authorized(&stamp) {
                break;
            }
            if owner.renders_playback() && !std::ptr::eq(self, Arc::as_ptr(&owner)) {
                owner.refresh_host_geometry();
            }
        }
        self.refresh_ui_inventory();
    }
    /// GUI清单按项目change/model/scope代次刷新，不每20ms枚举所有item，也不借活动工程。
    pub(crate) fn refresh_ui_inventory(&self) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let Some(host) = self.native_host() else {
            // 没有原生扩展接口 —— 这是 **ARA-only 宿主**（Studio One / Cubase / Nuendo /
            // Logic …）的正常形态，不是"不受支持"：时间线有内容、音频能出、参数能调，
            // 只读的只有**片段几何写回**，而那由能力位门控（`HostCapabilities`），
            // 不需要任何横幅。
            //
            // 【为什么这里不再区分"未识别的宿主"】两者此前共用一个三元表达式，而
            // `ara_only` 那一支**恒为真**：`bind_host_adapter` 只会存 `Some(Native)` 或
            // `None`，而 `host_backend()` 把 `None` 映射成 `AraOnly`。所以"未识别"那一态
            // 从来没有产出者，而它的文案（「本宿主尚不支持 / 时间线会是空的」）对每一个
            // 真能走到编辑器里的宿主都是**假的**。真正"既不提供原生扩展、也不提供 ARA"
            // 的宿主根本不会有文档，插件报的是 `ARA_NOT_BOUND` 那条终态错误，
            // 不会表现为"空白时间线 + 一条猜错方向的横幅"。
            //
            // 因此这里只把读数复位为"无可抱怨"，不再产生任何宿主支持类提示。
            *self.host_audio.lock_or_recover() = HostAudioStatus::default();
            // 【ARA-only 也必须推进 render_epoch】原生扩展缺席不代表"没有渲染输入"：
            // ARA 模型自己就能给出时间拉伸与内容淡化（`RenderItemFacts` 的 ARA 三字段）。
            // 此前这里直接 return ⇒ `render_facts` 永不更新、`render_epoch` 永不推进 ⇒
            // 宿主在 ARA 侧改了这两项，插件继续用旧配置渲染（静默的正确性 bug）。
            // ARA 拿不到方向/循环/声道（ARA 表达不到，见 `ara::mapping::LOST_FIELDS`），
            // 所以这里只登记 ARA 能给出的字段，不假装有原生事实。
            self.refresh_ara_render_facts(&stamp);
            return;
        };
        let allowed = || self.host_query_authorized(&stamp);
        let Ok(change) = host.geometry_revision(allowed) else {
            return;
        };
        let token = (change, stamp.1, stamp.2);
        // 令牌相同 **且** 距上次完整采集还没超过 `FACTS_REFRESH_INTERVAL` 才早退。
        // 光看令牌会漏掉 take/item 级事实变化（倒放 / 声道模式），因为宿主未必为它们
        // 推进 `GetProjectStateChangeCount` —— 那会让 `render_facts` 永不更新，正是
        // "倒放后波形变了、声音没变"的第二层缺口（见 `ui_inventory_sampled_at` 与
        // `FACTS_REFRESH_INTERVAL`）。
        let fresh = stamp
            .0
            .ui_inventory_sampled_at
            .lock_or_recover()
            .is_some_and(|at| at.elapsed() < FACTS_REFRESH_INTERVAL);
        let stored = *stamp.0.ui_inventory_stamp.lock_or_recover();
        if self.host_query_authorized(&stamp) && fresh && stored == Some(token) {
            return;
        }
        // 令牌本身是否变了。强制重采（令牌没变、只是采样过期）时若事实也没变，
        // 就不发布、不 bump `ui_geometry_revision`、不失效渲染 —— 兜底采样必须**无副作用**，
        // 否则空闲工程会被 1Hz 的假变更搅得前端反复重绘。
        let token_changed = stored != Some(token);
        let mut tracks = std::collections::BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        // 诊断计数：多少 item 认领到了 region（没认领的只能是无源占位）。
        let mut item_count = 0usize;
        let mut claimed_items = 0usize;
        // 本实例所在轨道的 `I_FOLDERDEPTH`；读不出来保持 `None`（见 `HostAudioState`）。
        let mut fx_folder_depth: Option<i32> = None;
        for owner in stamp.0.renderer_owners() {
            if !allowed() {
                return;
            }
            let Some(host) = owner.native_host() else {
                continue;
            };
            let own_allowed = || {
                allowed()
                    && !owner.is_closed()
                    && owner
                        .editor_document()
                        .is_ok_and(|doc| Arc::ptr_eq(&doc, &stamp.0))
            };
            let Ok(parent) = host.direct_track_target(&own_allowed) else {
                continue;
            };
            // 只读本实例那条轨道的单值，不枚举整个工程。放在 `seen` 去重之前：本实例
            // 可能不是本轮第一个被处理的 owner，去重会让它被跳过。
            if std::ptr::eq(self, Arc::as_ptr(&owner)) {
                fx_folder_depth = parent.folder_depth(&own_allowed).ok();
            }
            // 父轨已枚举过 ⇒ 它的全部后代也已经在 `tracks` 里，不必重走一遍。
            if seen.contains(parent.inventory_guid()) {
                continue;
            }
            // FX 挂在 folder 轨上时，folder 轨自身通常没有 item：连同后代一起枚举，
            // 否则整组在插件里"看不见"（ARA 侧没有 folder 概念，只能从宿主清单取）。
            let Ok(own_tracks) = host.ui_folder_tracks(&own_allowed) else {
                continue;
            };
            for mut track in own_tracks {
                if !seen.insert(track.guid.clone()) {
                    continue;
                }
                // 【锁序】按 `document.rs` 声明的层级取锁：`timeline`(2) 先于
                // `clip_ids`(4)。此前这里是反的 —— 与 `session.rs::analysis_timeline`
                // 的正序共存，构成真实的死锁窗口。
                let timeline = stamp.0.timeline.lock_or_recover();
                let ids = stamp.0.clip_ids.lock_or_recover();
                item_count += track.items.len();
                for candidate in stamp.0.renderer_owners() {
                    // 该 owner 可能同时持有多个 region（folder 轨上的 FX）：只要本轨有
                    // 任一 item 被它的某个 region 绑定，就把轨道身份认领到那个 clip 上。
                    let matched =
                        candidate
                            .host_geometries_locked(&stamp.0)
                            .into_iter()
                            .find(|bound| {
                                track
                                    .items
                                    .iter()
                                    .any(|item| item.geometry.item_id == bound.geometry.item_id)
                            });
                    if let Some(bound) = matched {
                        claimed_items += track
                            .items
                            .iter()
                            .filter(|item| item.geometry.item_id == bound.geometry.item_id)
                            .count();
                        if let Some(id) = ids.get(&bound.region_key) {
                            if let Some(clip) = timeline
                                .as_ref()
                                .and_then(|t| t.clips.iter().find(|c| &c.id == id))
                            {
                                track.id = clip.track_id.clone();
                                break;
                            }
                        }
                    }
                }
                drop(timeline);
                drop(ids);
                if track.id.starts_with("host-track-") {
                    if let Some(old) = stamp.0.ui_tracks.lock_or_recover().get(&track.guid) {
                        track.id = old.id.clone();
                    }
                }
                tracks.insert(track.guid.clone(), track);
            }
        }
        // 分类只用可核实的事实（见 `classify_host_audio`）。folder 结构读不出来时
        // `None` ⇒ 退化成"等待中"，绝不给出一个可能指错方向的 folder 提示。
        //
        // 绑定侧读数取自 `binding_report`：它与刷屏日志**同源** —— 同一份事实既决定
        // 日志输出，也决定用户看到的提示。`None` 表示"全部绑上了"（或还没求解过），
        // 此时两个 region 计数传 0，分类退回既有的 folder / awaiting 判定。
        let (regions_assigned, regions_unmatched) = self
            .binding_report
            .lock_or_recover()
            .as_ref()
            .map(|report| (report.assigned.len(), report.unmatched.len()))
            .unwrap_or((0, 0));
        let status = classify_host_audio(
            fx_folder_depth.is_some_and(|depth| depth >= 1),
            item_count,
            claimed_items,
            regions_assigned,
            regions_unmatched,
        );
        if !allowed() || host.geometry_revision(allowed).ok() != Some(change) {
            return;
        }
        // 采样会改变渲染结果的宿主事实。**两个来源**：
        //  - 原生扩展：方向 / 循环源 / 声道模式（ARA 表达不到，见 `ara::mapping::LOST_FIELDS`）；
        //  - ARA 模型：时间拉伸 / 基于内容的淡化（ARA region 自己就给出）。
        // 与 `project_host_take_facts_locked` 用**同一个** active-take 口径，否则"清单
        // 说倒放、渲染播种说正放"会再次分叉。
        let mut facts =
            crate::host::geometry::RenderFacts::from_items(tracks.values().flat_map(|track| {
                track.items.iter().map(|item| {
                    let active = item.takes.iter().find(|take| take.active);
                    let geometry = active.map(|take| &take.geometry).unwrap_or(&item.geometry);
                    (
                        item.geometry.item_id.clone(),
                        crate::host::geometry::RenderItemFacts {
                            reversed: active.is_some_and(|take| take.reversed),
                            loop_enabled: geometry.loop_source,
                            channel_mode: geometry.channel_mode,
                            // ARA 侧字段在下面按 region 叠加；这里先给安全默认。
                            timestretch: false,
                            content_fade_head: false,
                            content_fade_tail: false,
                        },
                    )
                })
            }));
        // ARA 侧的细化：把 `region_items` 里已认领的 region 事实叠加到对应 item 上。
        // 这份叠加让"宿主在 ARA 侧改了时间拉伸 / 内容淡化"也进 `RenderFacts` —— 与原生
        // 字段走同一条失效通路，不必再依赖某一处记得手动 bump。
        {
            let regions = stamp.0.regions.lock_or_recover();
            let items = stamp.0.region_items.lock_or_recover();
            for (key, item_id) in items.iter() {
                let (Some(entry), Some(region)) = (facts.items.get_mut(item_id), regions.get(key))
                else {
                    continue;
                };
                entry.timestretch = region.is_timestretch_enabled;
                entry.content_fade_head = region.has_content_based_fade_at_head;
                entry.content_fade_tail = region.has_content_based_fade_at_tail;
            }
        }
        let _transaction = stamp.0.transaction.lock_or_recover();
        if self.is_closed()
            || !stamp.0.is_alive()
            || stamp.0.revision.load(Ordering::Acquire) != stamp.1
            || stamp.0.scope_revision.load(Ordering::Acquire) != stamp.2
        {
            return;
        }
        // 【这是"宿主改了、插件立刻重渲染"的唯一通路】任何一项渲染事实变化都推进
        // `render_epoch`，于是 `PreparedVersion`（渲染缓存键）失效、`prepare_job` 不再
        // 早退。此前只有**淡变长度**有这条通路，倒放/声道/循环源都没有 —— 用户报障
        // "倒放后波形更新了、声音没变，编辑一下才对"正是这个缺口。
        //
        // 【为什么只置标志、通知留给文档级入口】发现变化的实例不一定是持有快照的实例
        // （REAPER 按轨道管理 ARA 实例：定时器在编辑实例上，快照在播放实例上）。
        // `invalidate_render_input()` 会推进 epoch 并通知**全部 owner**；此处只在本
        // 事务内记下"变了"，出事务后再调用它。
        let mut facts_changed = false;
        {
            let mut previous = stamp.0.render_facts.lock_or_recover();
            if *previous != facts {
                *previous = facts;
                facts_changed = true;
            }
        }
        // 强制重采但令牌与事实都没变 ⇒ 这一轮什么也没发生：只刷新采样时刻并返回，
        // 不发布清单、不 bump `ui_geometry_revision`、不失效渲染（见上面的 `token_changed`）。
        if !token_changed && !facts_changed {
            *stamp.0.ui_inventory_sampled_at.lock_or_recover() = Some(std::time::Instant::now());
            return;
        }
        stamp
            .0
            .ui_known_tracks
            .lock_or_recover()
            .extend(tracks.values().map(|track| track.id.clone()));
        *stamp.0.ui_tracks.lock_or_recover() = tracks;
        *stamp.0.ui_inventory_stamp.lock_or_recover() = Some(token);
        // 完整采集已完成：记下时刻，让"令牌不动但 take 级事实变了"这条兜底路径
        // 在下一次 `FACTS_REFRESH_INTERVAL` 到期时强制重采（见 `ui_inventory_sampled_at`）。
        *stamp.0.ui_inventory_sampled_at.lock_or_recover() = Some(std::time::Instant::now());
        stamp.0.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        *self.host_audio.lock_or_recover() = status;
        // 用户报障时最需要的一行：宿主清单里有多少 item，其中多少被 ARA region 认领。
        // 认领不到的那些只能以无源占位出现（见方案 Task 5.1 / R4）。
        if item_count > 0 {
            log::info!(
                "[ara] host inventory: {} track(s), {} item(s), {} claimed by an assigned region",
                stamp.0.ui_tracks.lock_or_recover().len(),
                item_count,
                claimed_items
            );
        }
        if status.state != HostAudioState::Ready {
            // 直读病灶：本实例到底拿到几条 region、FX 轨是不是 folder 父轨。下一次同样
            // 的报障不必再从位掩码与清单反推（见方案 Task 3.3）。
            let assigned = self
                .host_assigned_regions(&stamp.0)
                .map(|keys| keys.len())
                .unwrap_or(0);
            let depth = match fx_folder_depth {
                Some(depth) => depth.to_string(),
                None => "unknown".to_owned(),
            };
            log::info!("[ara] fx track: folderDepth={depth}; assigned regions={assigned}");
            log::info!(
                "[ara] {} clip(s) still waiting for host audio ({})",
                status.waiting_items,
                status.state.as_str()
            );
        }
        drop(_transaction);
        if facts_changed {
            // 渲染输入变了 ⇒ 文档级失效：推进 epoch 并通知**全部 owner**。
            // 不持 `transaction` 调用，避免与 `prepare_job` 的事务重入。
            stamp.0.invalidate_render_input();
        }
    }
    /// ARA-only 后端的渲染事实来源：只登记 ARA region 自己能给出的字段。
    ///
    /// 键用 region key 而非 item GUID —— ARA-only 下没有宿主 item 清单，item GUID 不
    /// 存在；region key 是这一后端里唯一稳定的 region 身份。原生后端另有一套按 item
    /// GUID 的键空间，两者不会同时出现（一个实例的后端在生命周期内固定）。
    pub(super) fn refresh_ara_render_facts(
        &self,
        stamp: &(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>),
    ) {
        let facts = {
            let regions = stamp.0.regions.lock_or_recover();
            crate::host::geometry::RenderFacts::from_items(regions.iter().map(|(key, region)| {
                (
                    format!("ara-region-{key}"),
                    crate::host::geometry::RenderItemFacts {
                        // 方向 / 循环 / 声道是**原生扩展专属**事实，ARA 表达不到 ⇒ 恒为安全默认。
                        reversed: false,
                        loop_enabled: false,
                        channel_mode: 0,
                        timestretch: region.is_timestretch_enabled,
                        content_fade_head: region.has_content_based_fade_at_head,
                        content_fade_tail: region.has_content_based_fade_at_tail,
                    },
                )
            }))
        };
        let _transaction = stamp.0.transaction.lock_or_recover();
        if self.is_closed()
            || !stamp.0.is_alive()
            || stamp.0.revision.load(Ordering::Acquire) != stamp.1
            || stamp.0.scope_revision.load(Ordering::Acquire) != stamp.2
        {
            return;
        }
        let changed = {
            let mut previous = stamp.0.render_facts.lock_or_recover();
            if *previous != facts {
                *previous = facts;
                true
            } else {
                false
            }
        };
        drop(_transaction);
        if changed {
            stamp.0.invalidate_render_input();
        }
    }
    /// 元数据单独采集，不让每个隐藏实例重复写工程clock；宿主调用始终不持内部锁。
    pub(crate) fn refresh_host_state_for_model(&self) {
        self.refresh_host_geometry();
    }
    pub(super) fn refresh_host_geometry(&self) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let host = self.native_host();
        // 【为什么没有原生扩展就直接返回】本函数采集的是**宿主 item 几何**，而
        // `document.region_items` 的唯一写者就在它里面。没有原生扩展时
        // `host_geometry()` / `host_geometries()` 的第一行就
        // `Err("host extension unavailable")` —— 什么都写不成。此前这里依赖
        // `before.is_some()`（`before` 来自原生宿主，缺席时为 `None`）让早退**永不成立**，
        // 于是每个 UI tick 都完整走一遍这条注定失败的路径。早返回等价而省事。
        if host.is_none() {
            return;
        }
        let authorized = || self.host_query_authorized(&stamp);
        let before = host
            .as_ref()
            .and_then(|host| host.geometry_revision(authorized).ok());
        if !authorized() {
            return;
        }
        let unchanged = before.is_some()
            && self
                .host_geometry
                .lock_or_recover()
                .as_ref()
                .is_some_and(|cached| {
                    cached.model == stamp.1
                        && cached.scope == stamp.2
                        && cached.change == before
                        && cached.value.is_ok()
                });
        if unchanged {
            let after = host
                .as_ref()
                .and_then(|host| host.geometry_revision(authorized).ok());
            if before == after && authorized() {
                return;
            }
        }
        if !authorized() {
            return;
        }
        // 唯一 region 时走强身份路径（宿主直接 parent take）；多 region（folder 轨上
        // 的 FX）结构上不可能有直接 parent，只能逐 region 按几何唯一匹配。
        let mut geometry = if stamp.3.len() == 1 {
            self.host_geometry().map(|bound| vec![bound])
        } else {
            self.host_geometries()
        };
        let after = host
            .as_ref()
            .and_then(|host| host.geometry_revision(authorized).ok());
        if geometry.is_ok() && (before.is_none() || before != after) {
            geometry = Err("host project changed during cached geometry refresh".into());
        }
        // 退避档位按**求解结果**更新（见 `enumeration_interval`）。放在缓存事务之前：
        // 它只读结果与代次，不碰文档锁。`after` 是本次尝试结束时的宿主几何代次，
        // 用它判断"宿主工程真的变了"。
        self.note_geometry_outcome(&geometry, after);
        let _transaction = stamp.0.transaction.lock_or_recover();
        let mut prepare_fades = false;
        if !self.is_closed()
            && stamp.0.is_alive()
            && stamp.0.revision.load(Ordering::Acquire) == stamp.1
            && stamp.0.scope_revision.load(Ordering::Acquire) == stamp.2
        {
            let mut cached = self.host_geometry.lock_or_recover();
            let changed = cached.as_ref().is_none_or(|old| {
                old.model != stamp.1 || old.scope != stamp.2 || old.value != geometry
            });
            let delegated = stamp.3.iter().any(|key| {
                stamp
                    .0
                    .regions
                    .lock_or_recover()
                    .get(key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
            });
            if delegated && changed && geometry.is_ok() {
                // 比较**整组**的淡变输入：此前只比四个 fade **长度**，于是改形状 /
                // 曲率 / S 都不推进 `render_epoch` —— 委托端（插件自己渲这条淡变）
                // 改了曲线而声音不变。形状与两根轴同样进指纹。
                let fade_inputs = |value: &[crate::host::geometry::BoundHostGeometry]| {
                    value
                        .iter()
                        .flat_map(|entry| {
                            let g = &entry.geometry;
                            [
                                g.fade_in_sec,
                                g.fade_out_sec,
                                g.auto_fade_in_sec,
                                g.auto_fade_out_sec,
                                g.fade_in_shape,
                                g.fade_out_shape,
                                g.fade_in_dir,
                                g.fade_out_dir,
                                g.fade_in_dir_new,
                                g.fade_out_dir_new,
                                g.fade_in_dir2_new,
                                g.fade_out_dir2_new,
                            ]
                        })
                        .collect::<Vec<_>>()
                };
                prepare_fades = cached
                    .as_ref()
                    .and_then(|old| old.value.as_ref().ok())
                    .is_none_or(|old| fade_inputs(old) != fade_inputs(geometry.as_ref().unwrap()));
                // `prepare_fades` 只记"渲染输入变了"；`render_epoch` 的推进与对全部
                // owner 的通知统一由下面的 `invalidate_render_input()` 完成。
            }
            // 门控是**实例级**的：单 region 时等同于该 item 的 mute（与既有语义一致）；
            // 多 region 时只有全部绑定 item 都静音才静音整组 —— 一个子轨静音不该让
            // 整组哑掉。逐 item 的独立门控需要渲染路径支持，见方案 Task 5.2 的已知边界。
            // 【为什么不给这个读数去抖】"静音 ⇒ 不出声"是一条**基本契约**，
            // 每一次迟滞都会让用户多听一段本该静音的内容。take 切换瞬间的误读是
            // 宿主侧异常，用状态去掩盖它会把"多响了"换成"该静音时还在响"——
            // 后者更难定位。宁可忠实地按宿主读数门控。
            self.host_item_muted.store(
                geometry
                    .as_ref()
                    .is_ok_and(|bound| !bound.is_empty() && bound.iter().all(|e| e.geometry.muted)),
                Ordering::Release,
            );
            if let Ok(bound) = &geometry {
                // 【为什么按**存活 region 集合**回收，而不是按本轮 `bound`】`bound`
                // 是**本实例**分配到的子集：同一文档上的编辑实例与播放实例各有自己
                // 的分配，用其中一个子集做 retain 会把另一个实例的边整批删掉。
                // `regions` 是文档级权威（`destroy_playback_region` 已按 key 删除），
                // 按它回收既清掉陈旧边，又不会误伤兄弟实例。
                let live: std::collections::BTreeSet<u64> =
                    stamp.0.regions.lock_or_recover().keys().copied().collect();
                let mut items = stamp.0.region_items.lock_or_recover();
                items.retain(|key, _| live.contains(key));
                // 【T22：授权记录在这里也有一个写者】此前 `authorized_takes` /
                // `authorized_sources` 只有一个写者（`remap_and_log`），而它要求
                // `region_items` 与 `ui_tracks` **在同一刻都就绪**。重开工程时这个条件
                // 常常不成立（几何缓存被 `clear_renderers` 的 revision 推进作废、
                // `assignments` 还没重发），于是那张表被写成空的 —— 而清单侧把"没有记录"
                // 当成"未授权"，一次性剥掉全部 clip 的源。
                //
                // 【为什么用 `or_insert` 而不是覆盖】这张表的语义是"ARA **授权那一刻**的
                // active take"。每轮清单都覆盖的话，用户在 REAPER 里切换 active take
                // 会被静默接受（GUID 永远相等），`take_switched` 这条真因就再也报不出来。
                // 只在**缺席**时补记 —— 那正是"这一刻才第一次观察到它"的语义。
                //
                // 【锁序】`region_items` → `authorized_takes` → `authorized_sources`，
                // 与 `document.rs` 声明的层级一致。
                let mut takes = stamp.0.authorized_takes.lock_or_recover();
                let mut sources = stamp.0.authorized_sources.lock_or_recover();
                for entry in bound {
                    let key = entry.region_key;
                    let item = entry.geometry.item_id.clone();
                    takes
                        .entry(key)
                        .or_insert_with(|| entry.geometry.take_id.clone());
                    if let Some(name) = entry.geometry.source_file_name.clone() {
                        sources.entry(key).or_insert(name);
                    }
                    items.insert(key, item);
                }
            }
            *cached = Some(CachedHostGeometry {
                model: stamp.1,
                scope: stamp.2,
                change: after,
                value: geometry,
            });
            drop(cached);
            if changed {
                stamp.0.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
            }
        }
        drop(_transaction);
        if prepare_fades {
            stamp.0.invalidate_render_input();
        }
    }
}
