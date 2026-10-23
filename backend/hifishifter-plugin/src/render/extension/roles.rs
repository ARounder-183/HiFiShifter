//! 角色、快照可用性与宿主传输观测。
use super::*;
impl ExtensionOwner {
    /// 本实例是否负责把处理结果作为音频输出（而非透传宿主原声）。
    ///
    /// 判据见 [`ExtensionOwner::is_editor_only`]。**未绑定实例（`roles == 0`）恒为
    /// `false`** —— 这与改动前的 `role == 1` 一致，是 `process_outputs_the_snapshot_*`
    /// 等用例依赖的既有行为。
    pub(crate) fn renders_playback(&self) -> bool {
        self.is_bound() && !self.is_editor_only()
    }
    /// 本实例收到过多少次"重新准备"通知（仅测试构建可见）。
    ///
    /// 【为什么暴露成方法而不是把字段改成 pub(crate)】`prepare_requests` 只在
    /// `#[cfg(test)]` 下存在，字段本身不能带跨模块的 pub 可见性而不连带改变
    /// 非测试构建的形状。给出一个同样只在 test 下编译的读取器，生产构建里
    /// 它随之消失，不会多出一个可见的原子量。
    #[cfg(test)]
    pub(crate) fn prepare_request_count(&self) -> u64 {
        self.prepare_requests.load(Ordering::Acquire)
    }
    /// 等后台准备任务排空（仅测试构建可见）。
    ///
    /// 【为什么需要它】本模块的测试会经 `prepare_renderers` 触发后台任务，而夹具文档
    /// 在用例结束时就被丢弃。给出一个有界的等待，让用例结尾不与仍在运行的任务抢生命周期
    /// （不等待的话会出现偶发的访问违例，且与真正的回归难以区分）。
    #[cfg(test)]
    pub(crate) fn await_preparation_idle(&self, timeout: std::time::Duration) {
        if let Some(worker) = self.preparation() {
            let _ = worker.wait_idle_until(std::time::Instant::now() + timeout);
        }
    }
    /// 当前准备队列（取自锁后的克隆；没有则 `None`）。
    ///
    /// 【为什么返回 `Arc` 而不是 guard】调用方要在队列表上做 `state()` / `request()`
    /// 这类可能短暂阻塞的操作；持着槽位锁做它们会把"换队列"（重绑）拖住。
    pub(super) fn preparation(&self) -> Option<Arc<super::preparation::PreparationQueue>> {
        self.preparation.lock_or_recover().clone()
    }

    /// 确保存在一条**可用**的准备队列；已被 `close()` 的旧队列在这里被替换。
    ///
    /// 【为什么必须检查 `is_closed()`】`stop_editor()` 关闭队列后，`Arc` 仍可能被
    /// 队列自身的 worker 或 UI 读数短暂持有。只判 `Option::is_none()` 会漏掉
    /// "槽位里躺着一个死队列"这一档，而那正是重绑时最常见的形状。
    pub(super) fn ensure_preparation(&self) {
        let mut slot = self.preparation.lock_or_recover();
        if slot.as_ref().is_some_and(|queue| !queue.is_closed()) {
            return;
        }
        match super::preparation::PreparationQueue::new() {
            Ok(queue) => *slot = Some(Arc::new(queue)),
            Err(error) => {
                // 【为什么只记日志】绑定本身仍然有效；没有准备队列只会让快照停在
                // "尚未就绪"，那由 `HostAudioState` 的合成中读数暴露，不需要让绑定失败。
                log::warn!("[ara] preparation worker unavailable: {error}");
                *slot = None;
            }
        }
    }

    /// 本实例是否已拿到角色分配（`roles == 0` 表示尚未绑定）。
    pub(super) fn is_bound(&self) -> bool {
        self.roles.load(Ordering::Acquire) != 0
    }
    /// 本实例是否拿到了宿主分配的 playback 角色。
    pub(crate) fn has_playback_role(&self) -> bool {
        self.roles.load(Ordering::Acquire) & ExtensionRoles::PLAYBACK_RENDERER.bits() != 0
    }
    /// 本实例已拿到角色分配，但其中**没有** playback 角色（等价于旧代码的 `role == 2`）。
    ///
    /// 【为什么与 `is_editor_only()` 分开】后者还要求"文档上另有 playback 实例"，
    /// 因为那才决定**要不要透传**。而"哪个实例挂着用户的窗口"只取决于角色本身 ——
    /// Cubase 的单实例形态下这个实例同时是"唯一的 GUI"和"接管渲染的那个"，
    /// 两个问题在这里答案相同、但判据不同。
    pub(crate) fn is_non_playback(&self) -> bool {
        self.is_bound() && !self.has_playback_role()
    }
    /// 宿主是否把"渲染"这件事交给了**别的**实例。
    ///
    /// 由 [`super::document::DocumentSession::publish_playback_coverage`] 写入；见该函数的说明。
    pub(crate) fn set_playback_covered(&self, covered: bool) {
        self.playback_covered.store(covered, Ordering::Release);
    }
    /// 本实例是否已经有可发布的快照。
    ///
    /// 【为什么提升前要问它】文档级"没有任何实例持 playback 角色"时需要挑一个实例
    /// 接管渲染。挑一个从未 `prepare()` 过的实例只会把静音从"一个实例"搬到"另一个
    /// 实例"；只有已经发布过快照的实例才真的能出声。见
    /// [`super::document::DocumentSession::report_playback_ownership`]。
    pub(crate) fn has_prepared_snapshot(&self) -> bool {
        self.snapshots.iter().any(|publisher| publisher.is_ready())
    }
    /// 处理结果发布器是否对**实际会被读取的那一份**就绪。
    ///
    /// 【为什么不能固定查 `snapshots[0]`】真正被读的是
    /// `snapshots[snapshot_index_for_rate(...)]`：48 kHz 及倍率工程读索引 1。
    /// 探测错发布器会让"实际那份还没就绪"被判成就绪，于是 `layered == false`、
    /// 输出被预清后逐样本写零（用户听到的"有概率完全没声音"）。
    pub(crate) fn processed_snapshot_ready(&self) -> bool {
        let index = self.snapshot_index_for_rate(self.setup_sample_rate());
        self.snapshots[index].is_ready()
    }
    /// 把本实例标记为"接管渲染"（诊断计数 + 日志）。
    pub(crate) fn promote_to_renderer(&self) {
        self.promoted_to_renderer.fetch_add(1, Ordering::Relaxed);
        log::warn!(
            "[ara] writer {} takes over rendering: no instance on this document holds the \
             playback role (roles={:#x})",
            self.writer_id(),
            self.roles.load(Ordering::Acquire)
        );
    }
    /// 文档关闭后重绑到另一份文档的次数（诊断；测试断言用）。
    #[cfg(test)]
    pub(crate) fn rebind_generation_count(&self) -> u64 {
        self.rebind_generation.load(Ordering::Acquire)
    }
    /// 本实例的创建序号（越小越早）；诊断时用来指认实例。
    pub(crate) fn writer_id(&self) -> u64 {
        self.writer_id.load(Ordering::Acquire)
    }
    /// 本实例的角色位集合（诊断用）。
    ///
    /// 【为什么需要它】`bind document` 那一行是 `INFO`，而现场日志可能是 `WARN` 级
    /// （实测 Cubase 会话整段没有一行 INFO）。于是"这个实例到底拿到了什么角色"只在
    /// 报警那条 `WARN` 里才可见 —— 那就得把角色位带上，一条日志说清全部。
    pub(crate) fn roles_bits(&self) -> i32 {
        self.roles.load(Ordering::Acquire)
    }
    /// 记录 `setupProcessing` 声明的采样率（Hz）。非有限/非正数忽略（保持未声明）。
    pub(crate) fn set_setup_sample_rate(&self, rate: f64) {
        let hz = rate.round();
        if hz.is_finite() && hz > 0.0 && hz <= f64::from(i32::MAX) {
            self.setup_sample_rate.store(hz as i32, Ordering::Release);
            // Capture 的段容量也按这个率算（一"段"是 8 **秒**，不是 8 万帧）。
            // 在这里转发而不是让调用点各记一次：宿主声明的工程率只有一个来源，
            // 两处分别记录必然漂移。
            self.capture.note_project_rate(hz);
        }
    }
    /// `setupProcessing` 声明的采样率（Hz；未声明为 `0`）。
    pub(crate) fn setup_sample_rate(&self) -> f64 {
        f64::from(self.setup_sample_rate.load(Ordering::Acquire))
    }
    /// 记录 `setBusArrangements` 协商出的扬声器编排（`kMono` = 1 / `kStereo` = 3），
    /// 打包为 `(输入 << 32) | 输出`。
    pub(crate) fn set_bus_arrangements(&self, inputs: u64, outputs: u64) {
        self.bus_arrangement
            .store((inputs << 32) | (outputs & 0xffff_ffff), Ordering::Release);
    }
    /// 协商出的扬声器编排位掩码（`direction`：`kInput` = 0 / `kOutput` = 1）。
    ///
    /// 声道数 = 掩码置位数（`count_ones`）：kMono=1 ⇒ 1，kStereo=3 ⇒ 2，
    /// 5.1=63 ⇒ 6。尚未协商（`0`）时按默认 `kStereo`(3) 报告 —— 与
    /// `component_get_bus_info` 声明的 2 声道一致，不与宿主协商结果矛盾。
    pub(crate) fn bus_arrangement(&self, direction: i32) -> u64 {
        let packed = self.bus_arrangement.load(Ordering::Acquire);
        if packed == 0 {
            return 3;
        }
        if direction == 0 {
            packed >> 32
        } else {
            packed & 0xffff_ffff
        }
    }
    /// 快照发布者下标：原生 44.1/48k 各一个，其余率取**最合适的已声明原生率**。
    ///
    /// 优先用 `setupProcessing` 的率（权威），未声明时退回 `context_rate`。这里**不**按
    /// 宿主率拒绝：`copy_block` 会把宿主帧线性重采样到该快照的原生率，与
    /// `setupProcessing` 里对非原生率的声明一致。
    ///
    /// 选择次序：先看输出率是不是某个原生率的**整数倍**（整数倍重采样无插值误差，
    /// 如 88.2k→44.1k、96k→48k），再退到最近的原生率。
    pub(crate) fn snapshot_index_for_rate(&self, context_rate: f64) -> usize {
        let declared = self.setup_sample_rate();
        let rate = if declared > 0.0 {
            declared
        } else {
            context_rate
        };
        if !rate.is_finite() || rate <= 0.0 {
            return 0;
        }
        let integral_multiple = |base: f64| {
            let ratio = rate / base;
            ratio >= 1.0 && (ratio - ratio.round()).abs() <= 1e-6
        };
        if integral_multiple(44100.0) {
            0
        } else if integral_multiple(48000.0) {
            1
        } else if (rate - 44100.0).abs() <= (rate - 48000.0).abs() {
            0
        } else {
            1
        }
    }
    /// 最终输出前的有效item静音门；与编辑器内轨道mute不同，不从输入是否全零猜静音。
    pub(crate) fn host_item_muted(&self) -> bool {
        self.host_item_muted.load(Ordering::Acquire)
    }
    /// editor-only必须透传；未绑定角色不在这里虚构为editor，保留原未绑定安全行为。
    ///
    /// 【为什么判据必须是 `role == 2` 而不是"没有 playback 角色"】未绑定的实例
    /// `role == 0`。`role != 1` 会把**未绑定**也判成 editor-only ⇒ `process` 走透传 ⇒
    /// 真正该输出快照的实例什么都不输出（`vst3/audio_boundary_tests` 的
    /// `process_outputs_the_snapshot_...` 等用例正是这个形状）。只有宿主**明确**
    /// 分配了非 playback 角色（`role == 2`）才是 editor-only。
    pub(crate) fn is_editor_only(&self) -> bool {
        self.is_non_playback() && self.playback_covered.load(Ordering::Acquire)
    }
    /// process只写有界原子诊断，JSON/日志全部由actor读取，暂不把候选计数当作根因。
    pub(crate) fn observe_transport(&self, context: &crate::audio_abi::ProcessContext, mode: i32) {
        if let Some(clock) = self.clock.get() {
            clock.observe(
                context,
                self.writer_id.load(Ordering::Relaxed),
                self.roles.load(Ordering::Relaxed),
                mode,
            );
            if mode != 2 {
                clock.update(context);
            }
        }
    }
    /// SDK允许从音频线程调用；记录来源但不做文件IO。
    pub(crate) fn record_processing_stop(&self) {
        if let Some(clock) = self.clock.get() {
            clock.observe_stop(
                self.writer_id.load(Ordering::Relaxed),
                self.roles.load(Ordering::Relaxed),
            );
            clock.stopped();
        }
    }
    /// 真实组件仅作为授权入口，原GUI共享同文档唯一actor/history。
    ///
    /// 【为什么 Capture 走另一支】宿主没给 ARA 时 `editor_document()` 必然失败
    /// （那是 `ARA_NOT_BOUND` 的语义），而 Capture 的编辑界面**必须**能用 ——
    /// 否则"录进来由插件处理"无从发生：用户改不了音高，也就没有处理可言。
    /// 两个分支产出同一个类型，因此命令分发、历史、参数、分析全部共用。
    pub(crate) fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        if !self.is_bound() && !self.bound_once.load(Ordering::Acquire) {
            // 【为什么"形态还没落定"也要给会话】形态要等 `setActive` 之后
            // `ARA_BINDING_GRACE`（5 秒）才落定，而**窗口打开的那一刻必然早于它**：
            // 那时拒绝服务就等于"非 ARA 宿主永远打不开界面"—— 探活拿到
            // `ARA_NOT_BOUND`（终态），整页替换成"插件会话不可用"，而且不可恢复。
            // 从未绑定过 ARA 的实例只有一条能服务界面的路（Capture 会话），所以这里
            // 不按形态分支。宿主**真的**绑上 ARA 之后 `is_bound()` 为真，自然走文档那份。
            return self.capture_session();
        }
        // 绑定过又关闭：那才是真的"会话已结束"，必须如实报错而不是悄悄变成 Capture。
        self.editor_document()?.editor_session()
    }

    /// Capture 形态的会话；每个实例唯一（`OnceLock`）。
    ///
    /// 【为什么与 ARA 会话分开持有】ARA 会话按**文档**唯一（文档上多个实例共享一个
    /// actor，这样同文档的视图能看到同一份编辑）。Capture 实例不在任何文档上，
    /// 因此"每个实例一份"才是与 ARA 同构的粒度（见 Capture 方案 §5.1 第 3 条：
    /// 同一文档上两个实例必须能独立工作）。
    pub(crate) fn capture_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        let session = self
            .capture_session
            .get_or_init(|| crate::editor::session::EditorSession::new_capture(self))
            .clone();
        // 【为什么在这里补投】`setState` 可能早于会话创建（用户从没开过窗口）。
        // 取走即清，幂等；失败只记日志 —— 一份读不出来的存档不该让界面打不开。
        let session = session?;
        if let Some(bytes) = self.pending_capture_edits.lock_or_recover().take() {
            if let Err(error) = session.stash_serialized_capture_edits(&bytes) {
                log::warn!("[capture] restored edits could not be applied: {error}");
            } else {
                session.request_reload();
            }
        }
        Ok(session)
    }

    /// 已经创建过的 Capture 会话；**不会**创建。
    ///
    /// 【为什么 `getState` 必须用这个】存档发生在宿主的主线程上，而创建一个会话会拉起
    /// actor 线程、扫临时目录、初始化模型资源。用户没开过窗口时"没有曲线可存"是正确
    /// 结论，不需要为此付出一次会话的代价。
    pub(crate) fn capture_session_if_initialized(
        &self,
    ) -> Option<Arc<crate::editor::session::EditorSession>> {
        self.capture_session
            .get()
            .and_then(|result| result.as_ref().ok())
            .cloned()
    }

    /// 已经存在的编辑器会话（Capture 或 ARA），**不创建**。
    pub(crate) fn existing_editor_session(
        &self,
    ) -> Option<Arc<crate::editor::session::EditorSession>> {
        if let Some(session) = self.capture_session_if_initialized() {
            return Some(session);
        }
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .and_then(|document| document.existing_editor_session())
    }

    /// 让 Capture 编辑器会话重新加载时间线并重渲染（录制结束 / 切换 take / 清空之后）。
    ///
    /// 【为什么不在命令里直接做】`capture_*` 命令在视图层（UI 线程）执行，而加载要读
    /// 捕获存储、物化分析副本 —— 那是文件 I/O。这里只置一个标志，由会话的 actor 线程
    /// 在下一轮自己加载（加载成功会顺带排一次渲染）。
    pub(crate) fn request_capture_reload(self: &Arc<Self>) {
        // 原始磁带要**立刻**跟着 take 走，不能等合成：用户按下停止之后马上就想听到东西，
        // 而一次神经合成要数秒。这一份也是"换 take 之后绝不播上一份"的保证
        // （见 `raw_tape_ready` 的 take 比对）。
        self.sync_raw_tape();
        // 换 take / 清空 / 录制结束都会走到这里：内容换了，本 pass 锁存的内容源不再有效。
        self.clear_rung_latch();
        if let Ok(session) = self.capture_session() {
            session.request_reload();
        }
    }

    /// 把活动 take 的原始捕获发布成一份**未处理**的快照（见 [`Self::raw_tape`]）。
    ///
    /// 【为什么幂等】它由命令（UI 线程）与合成准备（actor）两处调用；take 号没变时
    /// 直接返回，不重复拷贝几十 MB 的 PCM。
    pub(crate) fn sync_raw_tape(&self) {
        if !self.capture.has_material() {
            // 没有素材（清空 / 还没录 / 全是静音）：撤掉指针，让仲裁落到透传。
            self.raw_tape.clear();
            self.raw_tape_revision.store(0, Ordering::Release);
            return;
        }
        let revision = self.capture.revision();
        if self.raw_tape_revision.load(Ordering::Acquire) == revision {
            return;
        }
        let Some(snapshot) = self.capture.raw_tape() else {
            return;
        };
        self.raw_tape.collect_retired();
        if !self.raw_tape.has_capacity(&snapshot) {
            log::warn!(
                "[capture] raw tape does not fit the snapshot budget; keeping the previous one"
            );
            return;
        }
        match self.raw_tape.publish(snapshot) {
            Ok(()) => self.raw_tape_revision.store(revision, Ordering::Release),
            Err(error) => log::warn!("[capture] raw tape publish failed: {error:?}"),
        }
    }
    /// GUI能力标记只回答是否有官方写API；实际每个clip仍须唯一真实take绑定。
    /// 宿主淡化轴的版本语义；`None` 表示版本读不出来（前端应保持只读）。
    ///
    /// 取**任一** owner 的读数：`fade_axes_v2` 来自 `GetAppVersion`，同一进程里
    /// 所有实例必然一致，不存在"这个 owner 是新轴、那个是旧轴"的情况。
    pub(crate) fn host_fade_axes(&self) -> Option<bool> {
        self.editor_document().ok().and_then(|document| {
            document
                .renderer_owners()
                .iter()
                .find_map(|owner| owner.native_host().and_then(|host| host.fade_axes_v2()))
        })
    }

    /// 只返回同文档活入口的拥有引用接口；其project()仍钉在初始化时的真实所属工程。
    pub(crate) fn project_history_host(&self) -> Option<Arc<crate::host::NativeHost>> {
        let document = self.editor_document().ok()?;
        if let Some(host) = self.native_host().filter(|host| host.has_project_history()) {
            return Some(host);
        }
        document.renderer_owners().iter().find_map(|owner| {
            owner
                .native_host()
                .filter(|host| host.has_project_history())
        })
    }
    /// 当前编辑后端：宿主给了原生扩展就用它，否则退回 ARA-only。
    ///
    /// 【为什么是"一等后端"而不是 `None`】ARA-only 后端照样能看 / 能听 / 能渲染 /
    /// 能调参；它的退化只体现在"片段几何只读"。把这条结论收敛成一个后端值，前端与
    /// 诊断包就能直接读到"我是哪个后端、我能做什么"，而不是从一堆 `Option` 反推。
    /// Capture 通路的捕获存储。
    pub(crate) fn capture(&self) -> &crate::capture::CaptureStore {
        &self.capture
    }

    /// 本实例是否**应当**发布快照。
    ///
    /// 【为什么不能直接改 `renders_playback()`】那个读数的语义是"本文档的播放由本实例
    /// 渲染"，它被 `report_playback_ownership` 与诊断用于回答"文档上还有没有人在渲染"。
    /// Capture 实例**不在任何 ARA 文档上**，把它算进去会让那两处以为文档有播放实例。
    /// 这里另立一个"要不要出音"的判据，只在渲染准入点使用。
    pub(crate) fn renders(&self) -> bool {
        if self.is_bound() {
            return !self.is_editor_only();
        }
        // 【为什么"有素材"本身就是充分证据】素材只有一条产生路径：`capture_start`
        // （ARA 绑定时被显式拒绝，见 `capture/commands.rs`）或组件 state 恢复（绑定的
        // 实例走上面那一支）。因此"未绑定 + 有素材"只可能是 Capture。
        //
        // 【为什么不能只问 `mode()`】形态判定此前完全依赖编辑器窗口的 20ms 定时器
        // （`note_editor_tick`）与 `setActive` 之后 5 秒的宽限期。宿主把插件当普通插入
        // 用时用户很可能**根本没开窗口**，于是 `mode()` 永远是 `Undetermined` ⇒ 这里
        // 恒假 ⇒ 一条快照都不发布 ⇒ 录制正常、界面正常、**播放时全静音**。
        // 素材是比时钟更早、更硬的证据：它已经存在，就说明这个实例在录宿主。
        self.capture.has_material()
    }
}
