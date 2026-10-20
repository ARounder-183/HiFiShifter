//! 拥有文档控制器及安全运行时；对象创建失败必须留下原因，不能只向宿主返回无诊断的空引用。

use crate::ffi::generated_callbacks::ControllerDelegate;
use crate::{
    document_controller_interface, ContentObject, ContentReaderSnapshot, ControllerCapabilities,
    HostAudioSourceRef, HostClients, HostContentScope, HostMusicalContextRef, Plugin, PluginModel,
    PluginRuntime, SemanticCapabilities,
};
use ara2_bridge_core::{
    ApiGeneration, AraError, AudioSourceKind, ContentTimeRange, ContentUpdateScopes,
    DocumentProperties, ForeignSlice, Handle, LicenseRequest, PlaybackTransformationFlags,
    RawHandle, RegionSequenceKind, RestoreFilter, SizedInput, StoreFilter,
};
use ara2_bridge_sys::*;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::mem::offset_of;
use std::ptr::null_mut;

const fn ffi_bool(value: bool) -> ARABool {
    if value {
        kARATrue
    } else {
        kARAFalse
    }
}

/// 被 content reader 挡住的 `endEditing` 最多等多久。
///
/// 【为什么是 2 秒】正常宿主在几十毫秒内释放 reader（一次回调的跨度）；2 秒足够覆盖
/// 慢宿主与重排，同时把"宿主泄漏 reader"和"宿主只是慢"分开。超过它，时间线不发布的
/// 代价已经大于强收口一次映射的代价。
const DEFERRED_EDITING_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// 一次性留痕：`notifyModelUpdates` 想把某个对象的变化推回宿主，但宿主创建它时没给句柄。
///
/// 【为什么只报一次】它是"宿主能力缺失"而不是"每次调用都出错"：Cubase 15 对
/// `createAudioModification` 恒传 null host（见 `PATCHED.md` 补丁 12），于是每次通知都会
/// 命中同一条分支。逐次打印会把日志刷满，而这条事实只需要说一遍。
fn warn_missing_host_reference(kind: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if REPORTED.swap(true, Ordering::AcqRel) {
        return;
    }
    log::warn!(
        "[ara] notifyModelUpdates: the host passed no host reference for an {kind}; changes to \
         that object cannot be pushed back to the host (this message is reported once)"
    );
}

struct ControllerAdapter<P: PluginModel> {
    runtime: Option<PluginRuntime<P>>,
    host: HostClients<'static>,
    capabilities: SemanticCapabilities,
    factory: *const ARAFactory,
    musical_context_hosts: HashMap<RawHandle, ARAMusicalContextHostRef>,
    audio_source_hosts: HashMap<RawHandle, ARAAudioSourceHostRef>,
    audio_modification_hosts: HashMap<RawHandle, ARAAudioModificationHostRef>,
    playback_region_hosts: HashMap<RawHandle, ARAPlaybackRegionHostRef>,
    synthetic_sequences: HashMap<RawHandle, RawHandle>,
    updates: crate::UpdateEmitter,
    analysis: crate::AnalysisCoordinator,
    analysis_events: crate::AnalysisEmitter,
    pending_analysis_starts: Vec<RawHandle>,
    content_readers: HashSet<ARAContentReaderRef>,
    /// 一次被 reader 挡住的 `endEditing` 是否还欠着。
    ///
    /// 【为什么必须记下来】`endEditing` 在 content reader 存活时不能收口，而宿主
    /// **不会**为同一次编辑重发它 —— 此前只 warn 就 return，于是 `runtime.editing`
    /// 永远保持 true：插件侧 `ModelHandle::end_editing` 不执行、`remap_and_log` 不跑、
    /// `ready` 永不回真、`notifyModelUpdates` 与存档回调全部因 `is_editing()` 被拒。
    /// 实例从此砖化，日志里只有那一条 warn。置位后由 `destroy_content_reader` 在
    /// reader 排空的那一刻补跑。
    pending_end_editing: bool,
    /// 这次 `endEditing` 是从哪一个时刻开始被推迟的。
    ///
    /// 【为什么需要时间】"reader 排空"是补跑的正常条件，但宿主**可以**泄漏一个
    /// content reader（从不调 `destroyContentReader`）。那时排空永不到来，补跑也就
    /// 永不发生：`runtime.editing` 保持 true，`remap_and_log` 不跑，时间线永不发布。
    /// 丢一次映射是局部损失，让整个工程不可用是全局损失 —— 超过宽限就必须强制收口。
    deferred_end_editing_since: Option<std::time::Instant>,
    /// 上一次重试失败源读取的时刻（限流；见 `retry_failed_source_reads`）。
    last_source_retry: Option<std::time::Instant>,
    legacy_restore_reader: Option<ARAArchiveReaderHostRef>,
    factory_analyzable_content: HashSet<i32>,
    factory_playback_transformations: PlaybackTransformationFlags,
}

struct ReaderCell {
    snapshot: ContentReaderSnapshot,
}

impl<P: PluginModel> ControllerAdapter<P> {
    fn runtime(&self) -> Result<&PluginRuntime<P>, AraError> {
        self.runtime
            .as_ref()
            .ok_or(AraError::InvalidState("document controller is destroyed"))
    }

    /// 真正收口一次编辑会话；`endEditing` 与"reader 排空后的补跑"共用它。
    ///
    /// 【为什么单独抽出来】`endEditing` 可能因 content reader 存活而推迟，而推迟后
    /// 的补跑必须走**完全相同**的收口路径（同一个 host scope 构造方式），否则两条
    /// 路径会漂移。抽出来只有一处实现。
    fn finish_editing(&mut self) {
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        host.with_end_editing_content(|scope| {
            let _ = runtime.end_callback_editing_with_host(&scope);
        });
    }

    /// 消费插件侧置位的"强制收口"请求（T23）。
    ///
    /// 【为什么必须存在】`ready` 回到真只依赖宿主发起的四个入口（`endEditing`、
    /// `notifyModelUpdates`、样本访问变更、存档恢复）。宿主一旦 `beginEditing` 之后
    /// 再也不收口，或者积压了模型变化却不再调 `notifyModelUpdates`，插件就永久停在
    /// 未就绪 —— 用户唯一的出路是删掉插件再加。这不是某个 bug，是**没有出口**。
    ///
    /// 出口由插件侧提供：它在自己的定时器/音频线程上置一个原子量，这里在**每一次**
    /// 回调入口取走并执行。置位与消费之间不需要任何共享状态。
    fn service_forced_settle(&mut self) {
        let force_close = self
            .runtime
            .as_mut()
            .is_some_and(PluginRuntime::take_forced_edit_close);
        if force_close {
            log::warn!(
                "[ara] the host never closed its editing session; the plugin forced the settle"
            );
            self.finish_editing();
        }
        let force_flush = self
            .runtime
            .as_mut()
            .is_some_and(PluginRuntime::take_forced_model_flush);
        if force_flush && self.runtime.as_ref().is_some_and(|r| !r.is_editing()) {
            if let Some(runtime) = self.runtime.as_mut() {
                if runtime.flush_pending_model_updates() {
                    log::warn!(
                        "[ara] the host never published pending model changes; the plugin \
                         flushed them"
                    );
                }
            }
        }
    }

    fn runtime_mut(&mut self) -> Result<&mut PluginRuntime<P>, AraError> {
        // 【T23】先消费插件侧的强制收口请求 —— 它必须早于"reader 闸门"与任何模型操作，
        // 否则一条卡住的会话会让**每一次**后续回调都在这里返回 Err。
        self.service_forced_settle();
        // 【T5：content reader 不再是硬门】reader 读的是**不可变快照**
        // （`ReaderCell.snapshot`），模型变更既不移动它也不释放它（`publish` 用读者
        // 计数，旧盒只在最后一个读者离开时回收）。把它当硬门等于"宿主多留一个 reader，
        // 插件就再也建不出任何模型对象"—— 而 create/update 返回 null 的后果是
        // **宿主自己停止交付**，现场只剩一行 "document runtime busy"，用户看到的是
        // "所有 Clip 都在等宿主音频，删掉插件再加就好"。
        if !self.content_readers.is_empty() {
            log::warn!(
                "[ara] model operation continues with {} content reader(s) live; readers hold \
                 immutable snapshots and are unaffected by graph changes",
                self.content_readers.len()
            );
        }
        self.runtime
            .as_mut()
            .ok_or(AraError::InvalidState("document controller is destroyed"))
    }

    fn publish_reader(&mut self, snapshot: ContentReaderSnapshot) -> ARAContentReaderRef {
        let pointer = Box::into_raw(Box::new(ReaderCell { snapshot }));
        let reference = pointer.cast();
        let inserted = self.content_readers.insert(reference);
        debug_assert!(
            inserted,
            "fresh allocation has a unique content-reader identity"
        );
        reference
    }

    fn reader(&self, reference: ARAContentReaderRef) -> Option<&ContentReaderSnapshot> {
        if reference.is_null() || !self.content_readers.contains(&reference) {
            return None;
        }
        // SAFETY: membership proves this is a live `ReaderCell` allocation owned by this adapter.
        // SAFETY: the checked allocation remains live until `destroy_reader` or controller teardown.
        Some(unsafe { &(*reference.cast::<ReaderCell>()).snapshot })
    }

    fn destroy_reader(&mut self, reference: ARAContentReaderRef) {
        if self.content_readers.remove(&reference) {
            // SAFETY: successful removal transfers the unique allocation ownership for destruction.
            drop(unsafe { Box::from_raw(reference.cast::<ReaderCell>()) });
        }
    }

    fn clear_readers(&mut self) {
        for reference in self.content_readers.drain() {
            // SAFETY: every drained identity is a distinct adapter-owned `ReaderCell` allocation.
            drop(unsafe { Box::from_raw(reference.cast::<ReaderCell>()) });
        }
    }

    fn audio_source_object(&self, reference: ARAAudioSourceRef) -> Option<ContentObject> {
        let runtime = self.runtime().ok()?;
        let handle = runtime.resolve_audio_source(reference).ok()?;
        runtime
            .audio_source_is_active(handle)
            .ok()?
            .then_some(ContentObject::AudioSource(handle.into_raw()))
    }

    fn audio_modification_object(
        &self,
        reference: ARAAudioModificationRef,
    ) -> Option<ContentObject> {
        let runtime = self.runtime().ok()?;
        let handle = runtime.resolve_audio_modification(reference).ok()?;
        runtime
            .audio_modification_is_active(handle)
            .ok()?
            .then_some(ContentObject::AudioModification(handle.into_raw()))
    }

    fn playback_region_object(&self, reference: ARAPlaybackRegionRef) -> Option<ContentObject> {
        self.runtime()
            .and_then(|runtime| runtime.resolve_playback_region(reference))
            .ok()
            .map(|handle| ContentObject::PlaybackRegion(handle.into_raw()))
    }

    fn create_reader_for(
        &mut self,
        object: ContentObject,
        content_type: ARAContentType,
        range: Option<ContentTimeRange>,
    ) -> ARAContentReaderRef {
        if !self.content_readers.is_empty() {
            return null_mut();
        }
        let Ok(Some(snapshot)) =
            self.capabilities
                .create_content_reader(object, content_type, range)
        else {
            return null_mut();
        };
        if snapshot.content_type() != content_type {
            return null_mut();
        }
        self.publish_reader(snapshot)
    }

    fn read_archive(&self, reader: ARAArchiveReaderHostRef) -> Result<Vec<u8>, AraError> {
        self.host.archives().with_reader(reader, |reader| {
            let mut bytes = vec![0_u8; reader.len()];
            reader.read_at(0, &mut bytes)?;
            Ok(bytes)
        })?
    }

    fn write_archive(&self, writer: ARAArchiveWriterHostRef, bytes: &[u8]) -> Result<(), AraError> {
        self.host
            .archives()
            .with_writer(writer, |writer| writer.write_at(0, bytes))?
    }
}

impl<P: PluginModel> ControllerAdapter<P> {
    /// 补跑一次被 content reader 挡住的 `endEditing`。
    ///
    /// 【为什么需要它】`endEditing` 在 reader 存活时不能收口，而宿主**不会**为同一次
    /// 编辑重发它。不补跑就意味着 `ModelHandle::end_editing` 永不执行 ⇒
    /// `remap_and_log()` 不跑 ⇒ `document.timeline` 永不更新 ⇒ `region_items` 永不
    /// 填充 ⇒ `claimed_items` 保持 0 ⇒ 界面停在"等待宿主音频"。
    ///
    /// 【为什么有多个补跑点】"reader 排空"只是**最可能**的那一刻。宿主可以在 reader
    /// 还活着的时候开启样本访问、或周期性调 `notifyModelUpdates`；这两个入口同样
    /// 意味着"插件侧又可以动了"。任一处能收口就收口，别等那个唯一时刻。
    /// 重试"已授权但读取失败"的源（限流 1 s）。
    ///
    /// 【为什么在 notifyModelUpdates】后台/前台恢复的瞬间，宿主可能还没准备好样本；
    /// 一次失败会被模型记成终态，而之后没有回调保证再次读取。这里是空闲时唯一
    /// 稳定到达、且能构造源访问 scope 的入口。
    fn retry_failed_source_reads(&mut self) {
        if self
            .last_source_retry
            .is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(1))
        {
            return;
        }
        self.last_source_retry = Some(std::time::Instant::now());
        let sources = self
            .audio_source_hosts
            .iter()
            .map(|(handle, host_ref)| (*handle, *host_ref))
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        for (raw, host_ref) in sources {
            let Ok(handle) = Handle::<AudioSourceKind>::try_from_raw(raw) else {
                continue;
            };
            let Some(host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }).ok() else {
                continue;
            };
            host.with_audio_source_management(host_ref, |scope| {
                let _ = runtime.retry_failed_source_read(handle, &scope);
            });
        }
    }

    fn retry_deferred_end_editing(&mut self, reason: &str) {
        if !self.pending_end_editing {
            return;
        }
        let drained = self.content_readers.is_empty();
        if drained {
            self.pending_end_editing = false;
            self.deferred_end_editing_since = None;
            log::info!("[ara] endEditing retried: {reason}");
            self.finish_editing();
            return;
        }
        // 【为什么要强制收口】宿主泄漏 content reader 时"排空"永不到来。继续等下去
        // 等于让 `runtime.editing` 永久为真：时间线永不发布，存档回调也全部被拒。
        // 超过宽限后按事实收口，并把"这是被逼的"写进日志 —— 丢一次映射好过整档砖化。
        let elapsed = self
            .deferred_end_editing_since
            .map_or(std::time::Duration::ZERO, |since| since.elapsed());
        if elapsed >= DEFERRED_EDITING_GRACE {
            log::error!(
                "[ara] forcing endEditing after {:.1}s with {} content reader(s) still live \
                 ({reason}); the host appears to have leaked a reader",
                elapsed.as_secs_f32(),
                self.content_readers.len()
            );
            self.pending_end_editing = false;
            self.deferred_end_editing_since = None;
            self.finish_editing();
        }
    }
}

impl<P: PluginModel> ControllerDelegate for ControllerAdapter<P> {
    fn destroy_document_controller(&mut self) {
        self.clear_readers();
        self.host.revoke_all_audio_readers();
        if let Some(runtime) = self.runtime.take() {
            let _ = runtime.destroy();
        }
    }

    fn get_factory(&mut self) -> *const ARAFactory {
        self.factory
    }

    fn begin_editing(&mut self) {
        // 宿主开始新一次编辑：先看上一次被推迟的收口是否已经等够宽限。
        self.retry_deferred_end_editing("beginEditing observed a new edit session");
        // 【为什么必须在 reader 仍未排空时也强制收口】上一次 `endEditing` 可能因为
        // content reader 还活着而被推迟，`runtime.editing` 仍为 true。新的编辑会话
        // 不能建立在一个仍然 editing 的 runtime 上；让 begin 以
        // "editing is already active" 失败，宿主随后创建的所有对象都会因
        // `callback_edit()` 返回 Err 而被丢弃 —— 整份图建不起来。
        if self.pending_end_editing && self.runtime().is_ok_and(|runtime| runtime.is_editing()) {
            log::warn!(
                "[ara] forcing the previous endEditing before a new edit session \
                 ({} content reader(s) still live)",
                self.content_readers.len()
            );
            self.pending_end_editing = false;
            self.deferred_end_editing_since = None;
            self.finish_editing();
        }
        // 【为什么不再经过 `runtime_mut()`】content reader 是**快照**，它的正确性
        // 不依赖"模型不能变"。SDK 允许 reader 存活期间开始编辑；把它当成全局闸门
        // 会让宿主只要留下一个 reader，`beginEditing` 就静默失效，随后
        // createAudioSource / createPlaybackRegion 全部返回 null。
        // 这里只记一条 WARN，会话照常开始。
        if !self.content_readers.is_empty() {
            log::warn!(
                "[ara] beginEditing with {} live content reader(s); model readers are \
                 snapshots and do not block this edit session",
                self.content_readers.len()
            );
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        if let Err(error) = runtime.begin_callback_editing() {
            log::warn!("[ara] beginEditing had no effect: {error}");
        }
    }

    fn end_editing(&mut self) {
        // 【本地补丁 15】早退不再丢事件：置位待补跑（见 `pending_end_editing`）。
        // 此前这里**无声 return**，宿主只要还有 content reader 活着（长时间播放、
        // 或宿主迟迟不回收），`endEditing` 就永远不会收口 —— 插件停在旧时间线上，
        // 用户看到"编辑操作不更新 ARA"，而日志里什么都没有。
        if !self.content_readers.is_empty() {
            log::warn!(
                "[ara] endEditing deferred: {} content reader(s) still live; it will be                  retried as soon as the last one is released",
                self.content_readers.len()
            );
            if !self.pending_end_editing {
                self.deferred_end_editing_since = Some(std::time::Instant::now());
            }
            self.pending_end_editing = true;
            return;
        }
        self.pending_end_editing = false;
        self.deferred_end_editing_since = None;
        self.finish_editing();
    }

    fn notify_model_updates(&mut self) {
        // 【本地补丁 11】宿主用它让插件把待发的更新推回去（ARA 2.0 起替代旧的归档路径），
        // 并且规定"宿主不编辑时须周期性调用"。早退条件与宿主拿不到我们的更新直接相关，
        // 因此把三条早退原因都记下来。
        log::info!("[ara] host notifyModelUpdates");
        // 【T23】这是宿主空闲时最可靠的一次到达：先把插件侧置位的强制收口消费掉，
        // 再走下面既有的收口/补跑逻辑。
        self.service_forced_settle();
        // 【T2：content reader 不再是硬门】`flush_pending_model_updates` 做的是**重映射**
        // （`remap_and_log`），它不释放、不移动任何 reader 已持有的快照：publish 用读者
        // 计数，`ReadSection` 期间旧盒只在最后一个读者离开时才释放（见
        // `snapshot.rs` 的 `ReadSection`/`Drop`）。因此重映射不存在"reader 正在读半态"
        // 的风险。把它当成硬门，代价是宿主泄漏一个 reader 就终身拿不到任何模型更新 ——
        // 而 reader 的释放时机不在我们的控制之内（`createContentReader` 的宿主调用
        // 完全由它自己决定）。
        if !self.content_readers.is_empty() {
            log::warn!(
                "[ara] notifyModelUpdates continues with {} content reader(s) live; \
                 pending model changes are published from a fresh snapshot while readers \
                 keep the snapshot they already hold",
                self.content_readers.len()
            );
        }
        // 【T2：编辑会话本身也可能永远不关闭】宿主打开 `beginEditing` 之后可以再也不发
        // `endEditing`（工程加载中断 / 后台挂起 / 恢复失败）。那时 `is_editing()` 会永久
        // 挡住唯一的 flush 入口，而 `pending_end_editing` 那条兜底只管"endEditing 来了
        // 但被 reader 挡住"的形状。用同一个宽限期强制收口，让两种"永不收敛"得到同一种处理。
        let stuck_editing = self
            .runtime
            .as_ref()
            .and_then(PluginRuntime::editing_elapsed)
            .is_some_and(|elapsed| elapsed >= DEFERRED_EDITING_GRACE);
        if stuck_editing {
            log::error!(
                "[ara] forcing endEditing after the edit session stayed open longer than {:.1}s; \
                 the host never closed it, so pending model changes could not be published",
                DEFERRED_EDITING_GRACE.as_secs_f32()
            );
            self.finish_editing();
        }
        if self.runtime().map_or(true, PluginRuntime::is_editing) {
            log::warn!("[ara] notifyModelUpdates skipped: the document is in an editing session");
            return;
        }
        // reader 已经排空、文档也不在编辑会话：这正是被推迟的 `endEditing` 能收口的
        // 状态。宿主周期性调用本入口，所以它是独立的第二个补跑点。
        self.retry_deferred_end_editing("notifyModelUpdates found the document idle");
        // 【为什么在这里 flush】宿主可能在工程加载/后台恢复时直接重建图却不发
        // begin/endEditing。`notifyModelUpdates` 是 SDK 保证的空闲入口；只在
        // 委托方确实有积压变化时重算，避免每个心跳都跑一次全量映射。
        if let Some(runtime) = self.runtime.as_mut() {
            if runtime.flush_pending_model_updates() {
                log::info!(
                    "[ara] notifyModelUpdates published model changes that arrived outside \
                     an editing session"
                );
            }
        }
        self.retry_failed_source_reads();
        let Some(host) = self.host.model_updates() else {
            log::warn!(
                "[ara] notifyModelUpdates skipped: the host provided no model-update interface"
            );
            return;
        };
        for source in std::mem::take(&mut self.pending_analysis_starts) {
            if let Some(host_ref) = self.audio_source_hosts.get(&source).copied() {
                let _ = host.notify_analysis_progress(
                    host_ref,
                    kARAAnalysisProgressStarted as ARAAnalysisProgressState,
                    0.0,
                );
            }
        }
        for progress in self.analysis_events.take_pending() {
            match progress {
                crate::analysis::PendingAnalysisProgress::Updated(source, value) => {
                    if self.analysis.update(source, value).is_ok() {
                        if let Some(host_ref) = self.audio_source_hosts.get(&source).copied() {
                            let _ = host.notify_analysis_progress(
                                host_ref,
                                kARAAnalysisProgressUpdated as ARAAnalysisProgressState,
                                value,
                            );
                        }
                    }
                }
                crate::analysis::PendingAnalysisProgress::Completed(source) => {
                    if self.analysis.complete(source).is_ok() {
                        if let Some(host_ref) = self.audio_source_hosts.get(&source).copied() {
                            let _ = host.notify_analysis_progress(
                                host_ref,
                                kARAAnalysisProgressCompleted as ARAAnalysisProgressState,
                                1.0,
                            );
                        }
                    }
                }
            }
        }
        let mut updates = self.updates.take_pending();
        // 【为什么每条通知都要检查 host_ref】宿主可能在 `createXxx` 时传 null host
        // （实测 Cubase 15 对 `createAudioModification` 就是这样，见 `PATCHED.md` 补丁 12）。
        // 那时对象照常建成，但**回推通知没有去处**。此前这里是 `if let Some(..) { .. }`，
        // 没有 else —— 通知被静默丢弃，日志里完全看不出"插件想说、但宿主没给句柄"。
        // 一次性留痕：这条路径今天没有生产调用者（产品侧不修改宿主建模的对象），
        // 但一旦有人开始 `mark_*`，第一个被丢的通知必须可见。
        updates.flush_with(|notification, _| match notification {
            crate::UpdateNotification::AudioSource {
                source,
                range,
                flags,
            } => {
                if let Some(host_ref) = self.audio_source_hosts.get(&source).copied() {
                    let _ = host.notify_audio_source_changed(host_ref, range.as_ref(), flags);
                } else {
                    warn_missing_host_reference("audio source");
                }
            }
            crate::UpdateNotification::AudioModification {
                modification,
                range,
                flags,
            } => {
                if let Some(host_ref) = self.audio_modification_hosts.get(&modification).copied() {
                    let _ = host.notify_audio_modification_changed(host_ref, range.as_ref(), flags);
                } else {
                    warn_missing_host_reference("audio modification");
                }
            }
            crate::UpdateNotification::PlaybackRegion {
                region,
                range,
                flags,
            } => {
                if let Some(host_ref) = self.playback_region_hosts.get(&region).copied() {
                    let _ = host.notify_playback_region_changed(host_ref, range.as_ref(), flags);
                } else {
                    warn_missing_host_reference("playback region");
                }
            }
            crate::UpdateNotification::Document => {
                let _ = host.notify_document_data_changed();
            }
        });
    }

    fn begin_restoring_document_from_archive(
        &mut self,
        reader: ARAArchiveReaderHostRef,
    ) -> ARABool {
        if self.legacy_restore_reader.is_some() {
            return kARAFalse;
        }
        let Ok(bytes) = self.read_archive(reader) else {
            return kARAFalse;
        };
        let Ok(runtime) = self.runtime_mut() else {
            return kARAFalse;
        };
        if runtime.begin_callback_editing().is_err() {
            return kARAFalse;
        }
        if self.capabilities.restore_document(&bytes).is_err() {
            let _ = self
                .runtime_mut()
                .and_then(PluginRuntime::end_callback_editing);
            return kARAFalse;
        }
        self.legacy_restore_reader = Some(reader);
        kARATrue
    }

    fn end_restoring_document_from_archive(&mut self, reader: ARAArchiveReaderHostRef) -> ARABool {
        if self.legacy_restore_reader != Some(reader) {
            return kARAFalse;
        }
        if self
            .runtime_mut()
            .and_then(PluginRuntime::end_callback_editing)
            .is_err()
        {
            return kARAFalse;
        }
        self.legacy_restore_reader = None;
        kARATrue
    }

    fn store_document_to_archive(&mut self, writer: ARAArchiveWriterHostRef) -> ARABool {
        if self.runtime().map_or(true, PluginRuntime::is_editing) {
            // 【为什么这里要强收口】宿主若开启了 `beginRestoringDocumentFromArchive`
            // 却从未配对调用 `endRestoringDocumentFromArchive`，`runtime.editing` 会
            // 永久为真 —— 此后**每一次**存档都返回 `kARAFalse`：用户以为存上了，其实
            // 一次都没存。这种未配对的会话是本插件侧就能判定的事实，按它收口比让存档
            // 静默丢失安全得多。真正的宿主编辑会话（`legacy_restore_reader == None`）
            // 仍然照旧拒绝，避免在编辑中途落盘一份半写状态。
            if self.legacy_restore_reader.is_some() {
                log::error!(
                    "[ara] archive restore was never ended by the host; forcing the editing \
                     session closed so this save can proceed"
                );
                let _ = self
                    .runtime_mut()
                    .and_then(PluginRuntime::end_callback_editing);
                self.legacy_restore_reader = None;
            }
            if self.runtime().map_or(true, PluginRuntime::is_editing) {
                return kARAFalse;
            }
        }
        let Ok(bytes) = self.capabilities.store_document() else {
            return kARAFalse;
        };
        if self.write_archive(writer, &bytes).is_ok() {
            kARATrue
        } else {
            kARAFalse
        }
    }

    fn update_document_properties(&mut self, properties: *const ARADocumentProperties) {
        // SAFETY: the ARA callback contract supplies a complete ephemeral property record.
        let Ok(properties) = (unsafe { DocumentProperties::copy_from_ffi(properties) }) else {
            return;
        };
        let Ok(mut edit) = self.runtime_mut().and_then(PluginRuntime::callback_edit) else {
            return;
        };
        let _ = edit.update_document(properties);
    }

    fn create_musical_context(
        &mut self,
        host_ref: ARAMusicalContextHostRef,
        properties: *const ARAMusicalContextProperties,
    ) -> ARAMusicalContextRef {
        // SAFETY: the ARA callback contract supplies a complete ephemeral property record.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::MusicalContextProperties::copy_from_ffi(properties) })
        else {
            return null_mut();
        };
        let generation = self
            .runtime()
            .map(PluginRuntime::generation)
            .unwrap_or(ApiGeneration::V23Final);
        // 【本地补丁 12】与 createAudioSource 同一修法：必须真正走无宿主作用域，否则
        // 下面 `from_raw` 仍会把 null 拒掉。这里额外保留了 2.0 Draft 之前"补一条隐式
        // region sequence"的行为 —— 那段逻辑与宿主引用无关，退化路径必须原样带上，
        // 否则老代数宿主会丢掉整条轨道。
        if host_ref.is_null() {
            log::warn!(
                "[ara] createMusicalContext: the host passed no host reference; the context is \
                 created without host content access"
            );
            let Ok(runtime) = self.runtime_mut() else {
                log::warn!("[ara] createMusicalContext rejected: document runtime busy or closed");
                return null_mut();
            };
            let result: Result<(RawHandle, ARAMusicalContextRef, Option<RawHandle>), AraError> =
                (|| {
                    let mut edit = runtime.callback_edit()?;
                    let handle = edit.create_musical_context(properties)?;
                    let reference = edit.musical_context_ref(handle)?;
                    let raw_reference = reference.as_raw().cast();
                    let synthetic = if generation < ApiGeneration::V2Draft {
                        let properties = ara2_bridge_core::RegionSequenceProperties::new(
                            None, 0, reference, None,
                        )?;
                        match edit.create_region_sequence(handle, properties) {
                            Ok(sequence) => Some(sequence.into_raw()),
                            Err(error) => {
                                let _ = edit.destroy_musical_context(handle);
                                return Err(error);
                            }
                        }
                    } else {
                        None
                    };
                    Ok((handle.into_raw(), raw_reference, synthetic))
                })();
            let Ok((handle, reference, synthetic)) = result else {
                log::error!("[ara] createMusicalContext callback failure");
                return null_mut();
            };
            self.musical_context_hosts.insert(handle, host_ref);
            if let Some(sequence) = synthetic {
                self.synthetic_sequences.insert(handle, sequence);
            }
            return reference;
        }
        // SAFETY: the non-null callback identity remains live for the document-controller lifetime.
        let Ok(scoped_host_ref) = (unsafe { HostMusicalContextRef::from_raw(host_ref) }) else {
            return null_mut();
        };
        if !self.content_readers.is_empty() {
            return null_mut();
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return null_mut();
        };
        let result = host.with_musical_context_content(scoped_host_ref, |scope| {
            let mut edit = runtime.callback_edit()?;
            let handle = edit.create_musical_context_with_host(properties, &scope)?;
            let reference = edit.musical_context_ref(handle)?;
            let raw_reference = reference.as_raw().cast();
            let synthetic = if generation < ApiGeneration::V2Draft {
                let properties =
                    ara2_bridge_core::RegionSequenceProperties::new(None, 0, reference, None)?;
                match edit.create_region_sequence(handle, properties) {
                    Ok(sequence) => Some(sequence.into_raw()),
                    Err(error) => {
                        let _ = edit.destroy_musical_context(handle);
                        return Err(error);
                    }
                }
            } else {
                None
            };
            Ok((handle.into_raw(), raw_reference, synthetic))
        });
        let Ok((handle, reference, synthetic)) = result else {
            return null_mut();
        };
        self.musical_context_hosts.insert(handle, host_ref);
        if let Some(sequence) = synthetic {
            self.synthetic_sequences.insert(handle, sequence);
        }
        reference
    }

    fn update_musical_context_properties(
        &mut self,
        reference: ARAMusicalContextRef,
        properties: *const ARAMusicalContextProperties,
    ) {
        // SAFETY: the ARA callback contract supplies a complete ephemeral property record.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::MusicalContextProperties::copy_from_ffi(properties) })
        else {
            return;
        };
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_musical_context(reference))
        else {
            return;
        };
        let Some(host_ref) = self.musical_context_hosts.get(&handle.into_raw()).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains controller-owned until destroy.
        let host_ref = unsafe { HostMusicalContextRef::from_raw(host_ref) }.ok();
        if !self.content_readers.is_empty() {
            return;
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let Some(host_ref) = host_ref else {
            // 【为什么 null hostRef 只降级、不丢弃】实测 Cubase 15 会给这些回调传 null
            // hostRef。补丁 12 让 create 容忍它并把 null 存了进来 —— 若这里仍以
            // `from_raw` 失败即裸 `return`，整条生命周期回调会静默失效：宿主改名 / 改
            // 内容 / 换 tempo / 授权 PCM / 销毁源全部不生效（用户看到"授权了却读不到
            // 采样""tempo 永远读不到"）。改为走**无宿主**变体：语义与 create 的
            // `HostContentScope::unavailable()` 一致（本次回调没有宿主内容可查，其余照旧）。
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_musical_context(handle, properties);
            return;
        };
        let host = &self.host;
        host.with_musical_context_content(host_ref, |scope| {
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_musical_context_with_host(handle, properties, &scope);
        });
    }

    fn destroy_musical_context(&mut self, reference: ARAMusicalContextRef) {
        let synthetic = self
            .runtime()
            .and_then(|runtime| runtime.resolve_musical_context(reference))
            .ok()
            .and_then(|handle| self.synthetic_sequences.get(&handle.into_raw()).copied());
        let handle = {
            let Ok(runtime) = self.runtime_mut() else {
                return;
            };
            let Ok(handle) = runtime.resolve_musical_context(reference) else {
                return;
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            if let Some(synthetic) = synthetic {
                let Ok(synthetic) = Handle::<RegionSequenceKind>::try_from_raw(synthetic) else {
                    return;
                };
                if edit.destroy_region_sequence(synthetic).is_err() {
                    return;
                }
            }
            if edit.destroy_musical_context(handle).is_err() {
                return;
            }
            handle.into_raw()
        };
        self.musical_context_hosts.remove(&handle);
        self.synthetic_sequences.remove(&handle);
    }

    fn update_musical_context_content(
        &mut self,
        reference: ARAMusicalContextRef,
        range: *const ARAContentTimeRange,
        flags: ARAContentUpdateFlags,
    ) {
        // SAFETY: the callback contract supplies either null or one live aligned range.
        let Ok(range) = (unsafe { ContentTimeRange::copy_optional_from_ffi(range) }) else {
            return;
        };
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_musical_context(reference))
        else {
            return;
        };
        let Some(host_ref) = self.musical_context_hosts.get(&handle.into_raw()).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains live until destroy.
        let host_ref = unsafe { HostMusicalContextRef::from_raw(host_ref) }.ok();
        if !self.content_readers.is_empty() {
            return;
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let Some(host_ref) = host_ref else {
            // 同上：宿主没给引用时读不到 tempo / 拍号内容，但**必须**照常重读一次
            // （无宿主变体给的是 unavailable scope），否则变速后插件永远不知道。
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_musical_context_content(
                handle,
                range,
                ContentUpdateScopes::from_bits_retain(flags),
            );
            return;
        };
        let host = &self.host;
        host.with_musical_context_content(host_ref, |scope| {
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_musical_context_content_with_host(
                handle,
                range,
                ContentUpdateScopes::from_bits_retain(flags),
                &scope,
            );
        });
    }

    fn create_audio_source(
        &mut self,
        host_ref: ARAAudioSourceHostRef,
        properties: *const ARAAudioSourceProperties,
    ) -> ARAAudioSourceRef {
        // 【本地补丁 12】与 createAudioModification 是同一个误判：把回推通知用的不透明
        // 令牌当成创建模型对象的前提。差别在于这里的 hostRef **确实**参与内容读取
        //（`create_audio_source_with_host` 要用它去宿主侧取内容），所以不能只加一句警告
        // 了事 —— 必须真正改走无宿主作用域，否则紧随其后的 `from_raw` 仍会把 null 拒掉，
        // 日志里那句"created anyway"就是句假话。
        //
        // 退化路径是运行时自带的 `create_audio_source`，它内部用
        // `HostContentScope::unavailable()`，语义明确：本次回调里该源没有宿主内容可查，
        // 其余一切照旧。
        if host_ref.is_null() {
            log::warn!(
                "[ara] createAudioSource: the host passed no host reference; the source is \
                 created without host content access"
            );
            // SAFETY: the ARA callback contract supplies complete ephemeral properties and nested
            // data; this branch parses the same record the non-degraded path would.
            let Ok(parsed) =
                (unsafe { ara2_bridge_core::AudioSourceProperties::copy_from_ffi(properties) })
            else {
                log::error!("[ara] createAudioSource property rejection: null host reference path");
                return null_mut();
            };
            let Ok(runtime) = self.runtime_mut() else {
                log::error!("[ara] createAudioSource rejected: document runtime busy or closed");
                return null_mut();
            };
            let result: Result<(RawHandle, ARAAudioSourceRef), AraError> = (|| {
                let mut edit = runtime.callback_edit()?;
                let handle = edit.create_audio_source(parsed)?;
                let reference = edit.audio_source_ref(handle)?;
                Ok((handle.into_raw(), reference.as_raw().cast()))
            })();
            return match result {
                Ok((handle, reference)) => {
                    self.audio_source_hosts.insert(handle, host_ref);
                    reference
                }
                Err(error) => {
                    log::error!("[ara] createAudioSource callback failure: {error:?}");
                    null_mut()
                }
            };
        }
        // SAFETY: the ARA callback contract supplies complete ephemeral properties and nested data.
        let properties =
            match unsafe { ara2_bridge_core::AudioSourceProperties::copy_from_ffi(properties) } {
                Ok(properties) => properties,
                Err(error) => {
                    log::error!("ARA createAudioSource property rejection: {error:?}");
                    return null_mut();
                }
            };
        // SAFETY: the non-null callback identity remains live for the controller lifetime.
        let Ok(scoped_host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return null_mut();
        };
        if !self.content_readers.is_empty() {
            log::error!(
                "ARA createAudioSource rejected: {} content readers still live",
                self.content_readers.len()
            );
            return null_mut();
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            log::error!("ARA createAudioSource rejected: document runtime closed");
            return null_mut();
        };
        let result: Result<(RawHandle, ARAAudioSourceRef), AraError> = host
            .with_audio_source_content(scoped_host_ref, |scope| {
                let mut edit = runtime.callback_edit()?;
                let handle = edit.create_audio_source_with_host(properties, &scope)?;
                let reference = edit.audio_source_ref(handle)?;
                Ok((handle.into_raw(), reference.as_raw().cast()))
            });
        let (handle, reference) = match result {
            Ok(value) => value,
            Err(error) => {
                log::error!("ARA createAudioSource callback failure: {error:?}");
                return null_mut();
            }
        };
        self.audio_source_hosts.insert(handle, host_ref);
        reference
    }

    fn update_audio_source_properties(
        &mut self,
        reference: ARAAudioSourceRef,
        properties: *const ARAAudioSourceProperties,
    ) {
        // SAFETY: the ARA callback contract supplies complete ephemeral properties and nested data.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::AudioSourceProperties::copy_from_ffi(properties) })
        else {
            return;
        };
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_audio_source(reference))
        else {
            return;
        };
        let Some(host_ref) = self.audio_source_hosts.get(&handle.into_raw()).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains live until destroy.
        let host_ref = unsafe { HostAudioSourceRef::from_raw(host_ref) }.ok();
        if !self.content_readers.is_empty() {
            return;
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let Some(host_ref) = host_ref else {
            // 同上（见 `update_musical_context_properties`）：宿主没给引用时走无宿主变体。
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_audio_source(handle, properties);
            return;
        };
        let host = &self.host;
        host.with_audio_source_content(host_ref, |scope| {
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_audio_source_with_host(handle, properties, &scope);
        });
    }

    fn deactivate_audio_source_for_undo_history(
        &mut self,
        reference: ARAAudioSourceRef,
        deactivate: ARABool,
    ) {
        let deactivating = deactivate != kARAFalse;
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_audio_source(reference))
        else {
            return;
        };
        let Some(host_ref) = self.audio_source_hosts.get(&handle.into_raw()).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains live until destroy.
        let host_ref = unsafe { HostAudioSourceRef::from_raw(host_ref) }.ok();
        // 【T2：reader 不再是硬门】停用/恢复只改一个布尔并（恢复时）重读 PCM；
        // 已发布的快照由读者计数保护，重读只会让**下一次** prepare 用新内容，
        // 不会回收 reader 正在读的那一份。把它当硬门会让"宿主泄漏一个 reader"
        // 变成"源永远停在已停用"，表现正是"后台回来之后 Clip 一直等宿主音频"。
        if !self.content_readers.is_empty() {
            log::warn!(
                "[ara] audio source {} continues with {} content reader(s) live",
                if deactivating { "deactivation" } else { "reactivation" },
                self.content_readers.len()
            );
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let result = match host_ref {
            Some(host_ref) => {
                let host = &self.host;
                host.with_audio_source_management(host_ref, |scope| {
                    let mut edit = runtime.callback_edit()?;
                    edit.set_audio_source_deactivated_with_host(handle, deactivating, &scope)
                })
            }
            // 同上（见 `update_musical_context_properties`）：宿主没给引用时走无宿主变体。
            None => runtime.callback_edit().and_then(|mut edit| {
                edit.set_audio_source_deactivated_with_host(
                    handle,
                    deactivating,
                    &HostContentScope::unavailable(),
                )
            }),
        };
        if result.is_err() {
            return;
        }
        let raw = handle.into_raw();
        if deactivating {
            self.capabilities.cancel_analysis(raw);
            let _ = self.analysis.cancel(raw);
            self.analysis_events.cancel(raw);
            self.pending_analysis_starts.retain(|source| *source != raw);
            // 宿主没给引用时没有 reader 可撤（那一次读取本就走的是无宿主路径）。
            if let Some(host_ref) = host_ref {
                self.host.revoke_audio_source_readers(host_ref);
            }
        }
    }

    fn update_audio_source_content(
        &mut self,
        reference: ARAAudioSourceRef,
        range: *const ARAContentTimeRange,
        flags: ARAContentUpdateFlags,
    ) {
        // SAFETY: the callback contract supplies either null or one live aligned range.
        let Ok(range) = (unsafe { ContentTimeRange::copy_optional_from_ffi(range) }) else {
            return;
        };
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_audio_source(reference))
        else {
            return;
        };
        let Some(host_ref) = self.audio_source_hosts.get(&handle.into_raw()).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains live until destroy.
        let host_ref = unsafe { HostAudioSourceRef::from_raw(host_ref) }.ok();
        if !self.content_readers.is_empty() {
            return;
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let Some(host_ref) = host_ref else {
            // 同上（见 `update_musical_context_properties`）：宿主没给引用时走无宿主变体。
            // 内容版本必须照常推进，否则"素材被替换了"插件完全不知道。
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_audio_source_content(
                handle,
                range,
                ContentUpdateScopes::from_bits_retain(flags),
            );
            return;
        };
        let host = &self.host;
        host.with_audio_source_content(host_ref, |scope| {
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            let _ = edit.update_audio_source_content_with_host(
                handle,
                range,
                ContentUpdateScopes::from_bits_retain(flags),
                &scope,
            );
        });
    }

    fn enable_audio_source_samples_access(
        &mut self,
        reference: ARAAudioSourceRef,
        enable: ARABool,
    ) {
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_audio_source(reference))
        else {
            return;
        };
        let raw = handle.into_raw();
        if enable == kARAFalse {
            self.capabilities.cancel_analysis(raw);
            let _ = self.analysis.cancel(raw);
            self.analysis_events.cancel(raw);
            self.pending_analysis_starts.retain(|source| *source != raw);
        }
        let host_ref = self
            .audio_source_hosts
            .get(&raw)
            .copied()
            .and_then(|host_ref| {
                // SAFETY: the identity was validated on creation and remains live until destroy.
                unsafe { HostAudioSourceRef::from_raw(host_ref) }.ok()
            });
        // 【为什么不再因 content reader 非空而提前返回】reader 读的是**我们的快照**，
        // 与"我们读宿主源 PCM"是两件事。把它当成闸门会让 enable/disable 被静默丢弃，
        // 源的访问状态永久停留在旧值 —— 后台/前台或 undo 之后就是"等待宿主音频"。
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        // 【这一处最要紧】PCM 授权就靠它。宿主没给引用时若丢弃，插件**永远**读不到
        // 素材（波形空白、无声），而日志里只有"授权了却读不出来"。走无宿主变体后，
        // 读取会在 `HostContentScope::unavailable()` 下失败并**记下真因**
        // （`source_failures`），用户至少能看到"源读不出来"而不是一片空白。
        match host_ref {
            Some(host_ref) => {
                let host = &self.host;
                host.with_audio_source_management(host_ref, |scope| {
                    let _ = runtime.set_audio_source_samples_access_with_host(
                        handle,
                        enable != kARAFalse,
                        &scope,
                    );
                });
            }
            None => {
                let _ = runtime.set_audio_source_samples_access_with_host(
                    handle,
                    enable != kARAFalse,
                    &HostContentScope::unavailable(),
                );
            }
        }
        if enable == kARAFalse {
            if let Some(host_ref) = host_ref {
                self.host.revoke_audio_source_readers(host_ref);
            }
        }
        // 样本访问是"素材开始流动/停止流动"的时刻：它也是被推迟的 `endEditing`
        // 的独立补跑点（不依赖 reader 是否排空）。
        self.retry_deferred_end_editing("samples access changed");
    }

    fn destroy_audio_source(&mut self, reference: ARAAudioSourceRef) {
        let Ok(handle) = self
            .runtime()
            .and_then(|runtime| runtime.resolve_audio_source(reference))
        else {
            return;
        };
        if self
            .runtime()
            .and_then(|runtime| runtime.validate_audio_source_destruction(handle))
            .is_err()
        {
            return;
        }
        let raw = handle.into_raw();
        self.capabilities.cancel_analysis(raw);
        let _ = self.analysis.cancel(raw);
        self.analysis_events.cancel(raw);
        self.pending_analysis_starts.retain(|source| *source != raw);
        let Some(host_ref) = self.audio_source_hosts.get(&raw).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains live through this callback.
        let host_ref = unsafe { HostAudioSourceRef::from_raw(host_ref) }.ok();
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        // 【为什么销毁也必须降级】源销毁不了 ⇒ 运行时的 `live_sources` 永不归零 ⇒
        // `PluginRuntime::destroy()` 报错（调用点用 `let _ =` 吞掉）⇒
        // `destroy_document` 不跑 ⇒ 文档关闭时状态残留。
        let result = match host_ref {
            Some(host_ref) => {
                let host = &self.host;
                host.with_audio_source_management(host_ref, |scope| {
                    let mut edit = runtime.callback_edit()?;
                    edit.destroy_audio_source_with_host(handle, &scope)
                })
            }
            // 同上（见 `update_musical_context_properties`）：宿主没给引用时走无宿主变体。
            None => runtime
                .callback_edit()
                .and_then(|mut edit| edit.destroy_audio_source(handle)),
        };
        if result.is_err() {
            return;
        }
        if let Some(host_ref) = host_ref {
            self.host.revoke_audio_source_readers(host_ref);
        }
        self.audio_source_hosts.remove(&raw);
    }

    fn create_audio_modification(
        &mut self,
        source_reference: ARAAudioSourceRef,
        host_ref: ARAAudioModificationHostRef,
        properties: *const ARAAudioModificationProperties,
    ) -> ARAAudioModificationRef {
        // 【本地补丁 10】入口留痕 + 每条静默拒绝的原因。
        //
        // 宿主拿不到 modification 就不会建 playback region，于是**一条片段都没有**，
        // 而这里的每个失败分支原先都只是裸 `return null_mut()`：现场只有"时间线是空的"，
        // 既看不出宿主有没有调用，也看不出为什么被拒。实测 Cubase 15 的读数恒为
        // `sources=2 modifications=0 regionSequences=1 playbackRegions=0`。
        log::info!("[ara] host createAudioModification");
        // 【本地补丁 12】null hostRef 必须容忍，不能拒绝 —— 这是 Cubase 15 里"一条片段
        // 都没有"的**直接成因**：宿主调了本回调（日志里有 `host createAudioModification`），
        // 我们却因为 `hostRef` 为 null 回了 null，于是宿主拿不到 modification，也就永远
        // 不会走到 `createPlaybackRegion`，实测读数恒为
        // `sources=2 modifications=0 regionSequences=1 playbackRegions=0`。
        //
        // 【为什么 null 是合法的】hostRef 是宿主交给插件的**不透明令牌**，插件只在
        // `notifyAudioModificationChanged`、`notifyAnalysisProgress` 这类"回推更新"时原样
        // 交还宿主。它不是创建模型对象的**前提条件**：模型对象归宿主所有，插件只是登记
        // 一个引用。对照本文件里 `createRegionSequence` —— 它把 `_host_ref` 整个忽略掉，
        // 而它在同一次会话里**成功**了（`regionSequences=1`）；同一宿主、同一会话，
        // 唯独把 hostRef 当强校验的两条回调全部落空。这个对照就足以说明问题在我们的
        // 校验强度上，不在宿主身上。
        //
        // 【容忍之后会缺什么】只有"插件主动回推该对象的变更通知"这一项能力缺失，而
        // 本插件对 modification 从不做这种回推（`UpdateNotification::AudioModification`
        // 的发送点不在 v1 的路径上）。代价为零，收益是整条时间线回来了。
        if host_ref.is_null() {
            log::warn!(
                "[ara] createAudioModification: the host passed no host reference; the object is \
                 created anyway, but this plug-in cannot push change notifications for it back \
                 to the host"
            );
        }
        // SAFETY: the ARA callback contract supplies complete ephemeral properties.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::AudioModificationProperties::copy_from_ffi(properties) })
        else {
            log::warn!(
                "[ara] createAudioModification rejected: properties refused (see the core's                  persistent-ID / display-string validation)"
            );
            return null_mut();
        };
        let (handle, reference) = {
            let Ok(runtime) = self.runtime_mut() else {
                log::warn!(
                    "[ara] createAudioModification rejected: document runtime busy or closed"
                );
                return null_mut();
            };
            let Ok(source) = runtime.resolve_audio_source(source_reference) else {
                log::warn!(
                    "[ara] createAudioModification rejected: the host's audio-source reference                      does not resolve in this document"
                );
                return null_mut();
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                log::warn!("[ara] createAudioModification rejected: callback edit already active");
                return null_mut();
            };
            let Ok(handle) = edit.create_audio_modification(source, properties) else {
                log::error!("[ara] createAudioModification rejected: the model refused it");
                return null_mut();
            };
            let Ok(reference) = edit.audio_modification_ref(handle) else {
                log::error!("[ara] createAudioModification rejected: no ARA reference for the new modification");
                return null_mut();
            };
            (handle.into_raw(), reference.as_raw().cast())
        };
        self.audio_modification_hosts.insert(handle, host_ref);
        reference
    }

    fn clone_audio_modification(
        &mut self,
        source_reference: ARAAudioModificationRef,
        host_ref: ARAAudioModificationHostRef,
        properties: *const ARAAudioModificationProperties,
    ) -> ARAAudioModificationRef {
        // 【本地补丁 12】与 createAudioModification 同一缺陷、同一修法：hostRef 只是回推
        // 变更通知用的不透明令牌，不是创建模型对象的前提。把它当强校验会让宿主拿不到
        // 对象，进而整条下游链路（modification ⇒ playback region ⇒ clip）全部落空。
        if host_ref.is_null() {
            log::warn!(
                "[ara] cloneAudioModification: the host passed no host reference; the object is \
                 created anyway, but this plug-in cannot push change notifications for it back \
                 to the host"
            );
        }
        // SAFETY: the ARA callback contract supplies complete ephemeral properties.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::AudioModificationProperties::copy_from_ffi(properties) })
        else {
            return null_mut();
        };
        let (handle, reference) = {
            let Ok(runtime) = self.runtime_mut() else {
                return null_mut();
            };
            let Ok(source) = runtime.resolve_audio_modification(source_reference) else {
                return null_mut();
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                return null_mut();
            };
            let Ok(handle) = edit.clone_audio_modification(source, properties) else {
                return null_mut();
            };
            let Ok(reference) = edit.audio_modification_ref(handle) else {
                return null_mut();
            };
            (handle.into_raw(), reference.as_raw().cast())
        };
        self.audio_modification_hosts.insert(handle, host_ref);
        reference
    }

    fn update_audio_modification_properties(
        &mut self,
        reference: ARAAudioModificationRef,
        properties: *const ARAAudioModificationProperties,
    ) {
        // SAFETY: the ARA callback contract supplies complete ephemeral properties.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::AudioModificationProperties::copy_from_ffi(properties) })
        else {
            return;
        };
        let Ok(runtime) = self.runtime_mut() else {
            return;
        };
        let Ok(handle) = runtime.resolve_audio_modification(reference) else {
            return;
        };
        let Ok(mut edit) = runtime.callback_edit() else {
            return;
        };
        let _ = edit.update_audio_modification(handle, properties);
    }

    fn deactivate_audio_modification_for_undo_history(
        &mut self,
        reference: ARAAudioModificationRef,
        deactivate: ARABool,
    ) {
        let Ok(runtime) = self.runtime_mut() else {
            return;
        };
        let Ok(handle) = runtime.resolve_audio_modification(reference) else {
            return;
        };
        let Ok(mut edit) = runtime.callback_edit() else {
            return;
        };
        let _ = edit.set_audio_modification_deactivated(handle, deactivate != kARAFalse);
    }

    fn destroy_audio_modification(&mut self, reference: ARAAudioModificationRef) {
        let handle = {
            let Ok(runtime) = self.runtime_mut() else {
                return;
            };
            let Ok(handle) = runtime.resolve_audio_modification(reference) else {
                return;
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            if edit.destroy_audio_modification(handle).is_err() {
                return;
            }
            handle.into_raw()
        };
        self.audio_modification_hosts.remove(&handle);
    }

    fn is_audio_source_content_available(
        &mut self,
        reference: ARAAudioSourceRef,
        content_type: ARAContentType,
    ) -> ARABool {
        // 【本地补丁 11】宿主据此判断"这个插件能读这条源的哪类内容"。我们不做分析，
        // 因此一律答 false —— 但必须留痕：如果宿主正是靠它决定要不要把事件交给我们，
        // 那么"一律 false"就是片段消失的原因。
        let available = self
            .audio_source_object(reference)
            .is_some_and(|object| self.capabilities.is_content_available(object, content_type));
        log::info!("[ara] host isAudioSourceContentAvailable(type={content_type}) -> {available}");
        ffi_bool(available)
    }

    fn is_audio_source_content_analysis_incomplete(
        &mut self,
        reference: ARAAudioSourceRef,
        content_type: ARAContentType,
    ) -> ARABool {
        let Some(ContentObject::AudioSource(source)) = self.audio_source_object(reference) else {
            return kARAFalse;
        };
        let incomplete = self.analysis.contains(source, content_type)
            || self
                .capabilities
                .is_analysis_incomplete(source, content_type);
        if incomplete {
            // 【本地补丁 10】宿主会轮询这一项，并在"未完成"时**推迟**后续动作。
            // 只有真的答"未完成"时才记一行 —— 否则会把日志刷满。
            log::warn!(
                "[ara] reporting content analysis INCOMPLETE for type {content_type}; a host that                  waits for analysis will not proceed past this point"
            );
        }
        ffi_bool(incomplete)
    }

    fn request_audio_source_content_analysis(
        &mut self,
        reference: ARAAudioSourceRef,
        content_types_count: ARASize,
        content_types: *const ARAContentType,
    ) {
        let Some(ContentObject::AudioSource(source)) = self.audio_source_object(reference) else {
            return;
        };
        if content_types_count == 0 || content_types_count > 1024 || content_types.is_null() {
            return;
        }
        // SAFETY: ARA supplies `content_types_count` readable values for this callback.
        let content_types =
            unsafe { std::slice::from_raw_parts(content_types, content_types_count) }.to_vec();
        if content_types
            .iter()
            .any(|content_type| !self.factory_analyzable_content.contains(content_type))
        {
            // 【本地补丁 10】宿主请求分析我们**没有广告**的内容类型。忽略是对的（我们不会
            // 做这个分析），但必须留痕：宿主可能正等着它完成才继续。
            log::warn!(
                "[ara] the host requested analysis of {content_types:?}, which this factory does                  not advertise; the request is ignored and will never complete"
            );
            return;
        }
        if self
            .analysis
            .start(source, content_types.iter().copied())
            .is_err()
        {
            return;
        }
        if let Err(error) = self.capabilities.request_analysis(source, &content_types) {
            // 【本地补丁 10】宿主请求了分析、而本插件做不了：库会取消这次分析，宿主随后
            // 得到"没有待办分析"。必须留痕 —— 这条日志出现就说明宿主**确实**想要分析，
            // 而那正是"广告了可分析内容"探针要找的答案（探针开着时请求会走到这里，
            // 而不是走上面那条"未广告"的分支）。
            log::warn!(
                "[ara] the host requested analysis of {content_types:?}, which this build \
                 advertises but does not implement ({error}); the request is cancelled and the \
                 host is told there is nothing pending"
            );
            let _ = self.analysis.cancel(source);
            return;
        }
        self.pending_analysis_starts.push(source);
    }

    fn get_audio_source_content_grade(
        &mut self,
        reference: ARAAudioSourceRef,
        content_type: ARAContentType,
    ) -> ARAContentGrade {
        // 【本地补丁 11】0 = 最低等级（未分析）。宿主可能等它升到"最终"才继续。
        let grade = self.audio_source_object(reference).map_or(0, |object| {
            self.capabilities
                .content_grade(object, content_type)
                .as_raw()
        });
        log::info!("[ara] host getAudioSourceContentGrade(type={content_type}) -> {grade}");
        grade
    }

    fn create_audio_source_content_reader(
        &mut self,
        reference: ARAAudioSourceRef,
        content_type: ARAContentType,
        range: *const ARAContentTimeRange,
    ) -> ARAContentReaderRef {
        // 【本地补丁 11】宿主向我们索取某类内容的读取器。我们不做分析 ⇒ 交 null，
        // 而宿主可能把它当成"这个插件对这条源没有可读内容"。
        let Some(object) = self.audio_source_object(reference) else {
            log::warn!(
                "[ara] createAudioSourceContentReader rejected: unknown audio source reference"
            );
            return null_mut();
        };
        // SAFETY: the callback supplies null or one complete aligned ephemeral range.
        let Ok(range) = (unsafe { ContentTimeRange::copy_optional_from_ffi(range) }) else {
            log::warn!("[ara] createAudioSourceContentReader rejected: malformed range");
            return null_mut();
        };
        let reader = self.create_reader_for(object, content_type, range);
        log::info!(
            "[ara] host createAudioSourceContentReader(type={content_type}) -> {}",
            if reader.is_null() {
                "null (no such content)"
            } else {
                "reader"
            }
        );
        reader
    }

    fn is_audio_modification_content_available(
        &mut self,
        reference: ARAAudioModificationRef,
        content_type: ARAContentType,
    ) -> ARABool {
        ffi_bool(
            self.audio_modification_object(reference)
                .is_some_and(|object| self.capabilities.is_content_available(object, content_type)),
        )
    }

    fn get_audio_modification_content_grade(
        &mut self,
        reference: ARAAudioModificationRef,
        content_type: ARAContentType,
    ) -> ARAContentGrade {
        self.audio_modification_object(reference)
            .map_or(0, |object| {
                self.capabilities
                    .content_grade(object, content_type)
                    .as_raw()
            })
    }

    fn create_audio_modification_content_reader(
        &mut self,
        reference: ARAAudioModificationRef,
        content_type: ARAContentType,
        range: *const ARAContentTimeRange,
    ) -> ARAContentReaderRef {
        let Some(object) = self.audio_modification_object(reference) else {
            return null_mut();
        };
        // SAFETY: the callback supplies null or one complete aligned ephemeral range.
        let Ok(range) = (unsafe { ContentTimeRange::copy_optional_from_ffi(range) }) else {
            return null_mut();
        };
        self.create_reader_for(object, content_type, range)
    }

    fn is_playback_region_content_available(
        &mut self,
        reference: ARAPlaybackRegionRef,
        content_type: ARAContentType,
    ) -> ARABool {
        ffi_bool(
            self.playback_region_object(reference)
                .is_some_and(|object| self.capabilities.is_content_available(object, content_type)),
        )
    }

    fn get_playback_region_content_grade(
        &mut self,
        reference: ARAPlaybackRegionRef,
        content_type: ARAContentType,
    ) -> ARAContentGrade {
        self.playback_region_object(reference).map_or(0, |object| {
            self.capabilities
                .content_grade(object, content_type)
                .as_raw()
        })
    }

    fn create_playback_region_content_reader(
        &mut self,
        reference: ARAPlaybackRegionRef,
        content_type: ARAContentType,
        range: *const ARAContentTimeRange,
    ) -> ARAContentReaderRef {
        let Some(object) = self.playback_region_object(reference) else {
            return null_mut();
        };
        // SAFETY: the callback supplies null or one complete aligned ephemeral range.
        let Ok(range) = (unsafe { ContentTimeRange::copy_optional_from_ffi(range) }) else {
            return null_mut();
        };
        self.create_reader_for(object, content_type, range)
    }

    fn get_content_reader_event_count(&mut self, reference: ARAContentReaderRef) -> ARAInt32 {
        self.reader(reference)
            .and_then(|reader| i32::try_from(reader.len()).ok())
            .unwrap_or(0)
    }

    fn get_content_reader_data_for_event(
        &mut self,
        reference: ARAContentReaderRef,
        event_index: ARAInt32,
    ) -> *const std::ffi::c_void {
        let Ok(index) = usize::try_from(event_index) else {
            return std::ptr::null();
        };
        self.reader(reference)
            .and_then(|reader| reader.event_pointer(index))
            .unwrap_or(std::ptr::null())
    }

    fn destroy_content_reader(&mut self, reference: ARAContentReaderRef) {
        self.destroy_reader(reference);
        self.retry_deferred_end_editing("the last content reader was released");
    }

    fn create_region_sequence(
        &mut self,
        _host_ref: ARARegionSequenceHostRef,
        properties: *const ARARegionSequenceProperties,
    ) -> ARARegionSequenceRef {
        // 【本地补丁 10】与 createAudioModification 同一理由：入口留痕，便于对照宿主
        // 究竟走到哪一步。
        log::info!("[ara] host createRegionSequence");
        let Ok(runtime) = self.runtime_mut() else {
            log::warn!("[ara] createRegionSequence rejected: document runtime busy or closed");
            return null_mut();
        };
        // SAFETY: properties are ephemeral and the resolver accepts only this runtime's live refs.
        let Ok(properties) = (unsafe {
            ara2_bridge_core::RegionSequenceProperties::copy_from_ffi_with_context(
                properties,
                |reference| {
                    runtime
                        .resolve_musical_context(reference)
                        .and_then(|handle| runtime.musical_context_ref(handle))
                },
            )
        }) else {
            return null_mut();
        };
        let context_ref = properties.musical_context();
        let Ok(context) = runtime.resolve_musical_context(context_ref.as_raw().cast()) else {
            return null_mut();
        };
        let Ok(mut edit) = runtime.callback_edit() else {
            return null_mut();
        };
        let Ok(handle) = edit.create_region_sequence(context, properties) else {
            return null_mut();
        };
        edit.region_sequence_ref(handle)
            .map_or(null_mut(), |reference| reference.as_raw().cast())
    }

    fn update_region_sequence_properties(
        &mut self,
        reference: ARARegionSequenceRef,
        properties: *const ARARegionSequenceProperties,
    ) {
        let Ok(runtime) = self.runtime_mut() else {
            return;
        };
        let Ok(handle) = runtime.resolve_region_sequence(reference) else {
            return;
        };
        // SAFETY: properties are ephemeral and the resolver accepts only this runtime's live refs.
        let Ok(properties) = (unsafe {
            ara2_bridge_core::RegionSequenceProperties::copy_from_ffi_with_context(
                properties,
                |raw| {
                    runtime
                        .resolve_musical_context(raw)
                        .and_then(|context| runtime.musical_context_ref(context))
                },
            )
        }) else {
            return;
        };
        let Ok(context) =
            runtime.resolve_musical_context(properties.musical_context().as_raw().cast())
        else {
            return;
        };
        let Ok(mut edit) = runtime.callback_edit() else {
            return;
        };
        let _ = edit.update_region_sequence_with_context(handle, context, properties);
    }

    fn destroy_region_sequence(&mut self, reference: ARARegionSequenceRef) {
        let Ok(runtime) = self.runtime_mut() else {
            return;
        };
        let Ok(handle) = runtime.resolve_region_sequence(reference) else {
            return;
        };
        let Ok(mut edit) = runtime.callback_edit() else {
            return;
        };
        let _ = edit.destroy_region_sequence(handle);
    }

    fn get_playback_region_head_and_tail_time(
        &mut self,
        reference: ARAPlaybackRegionRef,
        head_time: *mut ARATimeDuration,
        tail_time: *mut ARATimeDuration,
    ) {
        if reference.is_null() || head_time.is_null() || tail_time.is_null() {
            return;
        }
        let Some(adapter) = self.capabilities.head_tail() else {
            return;
        };
        let Some((head, tail)) = adapter.query(reference as usize as u64) else {
            return;
        };
        // SAFETY: callback contract supplies both non-null scalar output locations for this call.
        unsafe {
            head_time.write(head);
            tail_time.write(tail);
        }
    }

    fn create_playback_region(
        &mut self,
        modification_reference: ARAAudioModificationRef,
        host_ref: ARAPlaybackRegionHostRef,
        properties: *const ARAPlaybackRegionProperties,
    ) -> ARAPlaybackRegionRef {
        // 【本地补丁 10】入口留痕：这一行出现而后面没有 region，才说明问题在创建逻辑里；
        // 它根本不出现，就说明宿主没走到这一步（modification 那步已经断了）。
        log::info!("[ara] host createPlaybackRegion");
        // 【本地补丁 12】同一缺陷、同一修法：hostRef 是回推变更通知用的不透明令牌。
        // 拒掉它的代价是**整条片段消失**（宿主拿不到 region ⇒ 没有 clip ⇒ 时间线空），
        // 而容忍它的代价只是本插件无法主动回推该 region 的变更通知。
        if host_ref.is_null() {
            log::warn!(
                "[ara] createPlaybackRegion: the host passed no host reference; the region is \
                 created anyway, but this plug-in cannot push change notifications for it back \
                 to the host"
            );
        }
        let (modification, sequence, properties) = {
            let Ok(runtime) = self.runtime() else {
                return null_mut();
            };
            let generation = runtime.generation();
            // SAFETY: properties are ephemeral and both resolvers accept only this runtime's live refs.
            let Ok(properties) = (unsafe {
                ara2_bridge_core::PlaybackRegionProperties::copy_from_ffi_with_refs(
                    properties,
                    generation,
                    |raw| {
                        runtime
                            .resolve_musical_context(raw)
                            .and_then(|handle| runtime.musical_context_ref(handle))
                    },
                    |raw| {
                        runtime
                            .resolve_region_sequence(raw)
                            .and_then(|handle| runtime.region_sequence_ref(handle))
                    },
                )
            }) else {
                return null_mut();
            };
            let requested = PlaybackTransformationFlags::from_bits_retain(
                properties.transformation_flags() as u32,
            );
            if !self.factory_playback_transformations.contains(requested) {
                // 【本地补丁 9，已从"拒绝"改为"降级"】此前这里回 null，代价是**整条
                // playback region 消失**：宿主拿不到 region，于是既没有 clip、也不会调用
                // 任何 renderer 的 addPlaybackRegion（实测 Cubase 15：
                // `sources=1 regionSequences=1 playbackRegions=0`，编辑器里一条片段都没有）。
                //
                // 【为什么降级是对的】ARA 契约规定宿主**只能**设置工厂广告过的位
                // （`ARAPlaybackRegionProperties::transformationFlags` 的说明）。宿主若
                // 越界，正确反应是"按我们能履行的部分渲染并留痕"，而不是把用户的素材
                // 整条丢掉 —— 一位不认识的旗标绝不该赔上一条 region。这也与插件既有的
                // 逐 region 降级纪律一致（见 `log_unknown_transformation_flags`）。
                let unknown = requested.bits() & !self.factory_playback_transformations.bits();
                log::warn!(
                    "[ara] playback region: host flags {:#x} exceed the factory's advertised \
                     {:#x} (unhandled bits {:#x}); the region is created anyway and those bits \
                     are ignored",
                    requested.bits(),
                    self.factory_playback_transformations.bits(),
                    unknown
                );
            }
            let Ok(modification) = runtime.resolve_audio_modification(modification_reference)
            else {
                return null_mut();
            };
            let sequence = if let Some(sequence_ref) = properties.region_sequence() {
                let Ok(sequence) = runtime.resolve_region_sequence(sequence_ref.as_raw().cast())
                else {
                    return null_mut();
                };
                sequence
            } else {
                let Some(context_ref) = properties.musical_context() else {
                    return null_mut();
                };
                let Ok(context) = runtime.resolve_musical_context(context_ref.as_raw().cast())
                else {
                    return null_mut();
                };
                let Some(sequence) = self.synthetic_sequences.get(&context.into_raw()).copied()
                else {
                    return null_mut();
                };
                let Ok(sequence) = Handle::<RegionSequenceKind>::try_from_raw(sequence) else {
                    return null_mut();
                };
                sequence
            };
            (modification, sequence, properties)
        };
        let (handle, reference) = {
            let Ok(runtime) = self.runtime_mut() else {
                return null_mut();
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                return null_mut();
            };
            // 【本地补丁 9】同样静默：模型侧拒绝一条 region 时，宿主只看到 null。
            let handle = match edit.create_playback_region(modification, sequence, properties) {
                Ok(handle) => handle,
                Err(error) => {
                    log::error!(
                        "[ara] playback region rejected: create_playback_region failed: {error}; \
                         the host receives a null region and no clip is created for it"
                    );
                    return null_mut();
                }
            };
            let Ok(reference) = edit.playback_region_ref(handle) else {
                return null_mut();
            };
            (handle.into_raw(), reference.as_raw().cast())
        };
        self.playback_region_hosts.insert(handle, host_ref);
        reference
    }

    fn update_playback_region_properties(
        &mut self,
        reference: ARAPlaybackRegionRef,
        properties: *const ARAPlaybackRegionProperties,
    ) {
        let (handle, sequence, properties) = {
            let Ok(runtime) = self.runtime() else {
                return;
            };
            let Ok(handle) = runtime.resolve_playback_region(reference) else {
                return;
            };
            let generation = runtime.generation();
            // SAFETY: properties are ephemeral and resolvers accept only this runtime's live refs.
            let Ok(properties) = (unsafe {
                ara2_bridge_core::PlaybackRegionProperties::copy_from_ffi_with_refs(
                    properties,
                    generation,
                    |raw| {
                        runtime
                            .resolve_musical_context(raw)
                            .and_then(|h| runtime.musical_context_ref(h))
                    },
                    |raw| {
                        runtime
                            .resolve_region_sequence(raw)
                            .and_then(|h| runtime.region_sequence_ref(h))
                    },
                )
            }) else {
                return;
            };
            let requested = PlaybackTransformationFlags::from_bits_retain(
                properties.transformation_flags() as u32,
            );
            if !self.factory_playback_transformations.contains(requested) {
                // 【本地补丁】与 create 路径同口径：不认识的旗标**降级接受**并留痕，
                // 而不是丢弃整次更新 —— 一次静默丢弃在用户看来就是"编辑莫名不生效"。
                let unknown = requested.bits() & !self.factory_playback_transformations.bits();
                log::warn!(
                    "[ara] updatePlaybackRegion: host flags {:#x} exceed the advertised {:#x} \
                     (unhandled bits {:#x}); the update is applied and those bits are ignored",
                    requested.bits(),
                    self.factory_playback_transformations.bits(),
                    unknown
                );
            }
            let sequence = if let Some(sequence) = properties.region_sequence() {
                let Ok(sequence) = runtime.resolve_region_sequence(sequence.as_raw().cast()) else {
                    return;
                };
                sequence
            } else {
                let Some(context) = properties.musical_context() else {
                    return;
                };
                let Ok(context) = runtime.resolve_musical_context(context.as_raw().cast()) else {
                    return;
                };
                let Some(sequence) = self.synthetic_sequences.get(&context.into_raw()).copied()
                else {
                    return;
                };
                let Ok(sequence) = Handle::<RegionSequenceKind>::try_from_raw(sequence) else {
                    return;
                };
                sequence
            };
            (handle, sequence, properties)
        };
        let Ok(runtime) = self.runtime_mut() else {
            return;
        };
        let Ok(mut edit) = runtime.callback_edit() else {
            return;
        };
        let _ = edit.update_playback_region_with_sequence(handle, sequence, properties);
    }

    fn destroy_playback_region(&mut self, reference: ARAPlaybackRegionRef) {
        let handle = {
            let Ok(runtime) = self.runtime_mut() else {
                return;
            };
            let Ok(handle) = runtime.resolve_playback_region(reference) else {
                return;
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                return;
            };
            if edit.destroy_playback_region(handle).is_err() {
                return;
            }
            handle.into_raw()
        };
        self.playback_region_hosts.remove(&handle);
    }

    fn get_processing_algorithms_count(&mut self) -> i32 {
        self.capabilities
            .algorithms()
            .and_then(|catalog| catalog.len_i32().ok())
            .unwrap_or(0)
    }

    fn restore_objects_from_archive(
        &mut self,
        reader: ARAArchiveReaderHostRef,
        filter: *const ARARestoreObjectsFilter,
    ) -> ARABool {
        if self.runtime().map_or(true, |runtime| !runtime.is_editing()) {
            return kARAFalse;
        }
        // SAFETY: the callback supplies a null/all filter or a complete filter with live nested IDs.
        let Ok(filter) = (unsafe { RestoreFilter::copy_selection_from_ffi(filter) }) else {
            return kARAFalse;
        };
        let Ok(bytes) = self.read_archive(reader) else {
            return kARAFalse;
        };
        if self.capabilities.restore_objects(&filter, &bytes).is_ok() {
            kARATrue
        } else {
            kARAFalse
        }
    }

    fn store_objects_to_archive(
        &mut self,
        writer: ARAArchiveWriterHostRef,
        filter: *const ARAStoreObjectsFilter,
    ) -> ARABool {
        let Ok(runtime) = self.runtime() else {
            return kARAFalse;
        };
        if runtime.is_editing() {
            return kARAFalse;
        }
        let session = runtime.session();
        // SAFETY: the callback supplies a null/all filter or complete live reference arrays; both
        // resolvers reject foreign, stale, and wrong-kind identities before building the filter.
        let Ok(filter) = (unsafe {
            StoreFilter::copy_selection_from_ffi(
                filter,
                session,
                |reference| runtime.resolve_audio_source(reference),
                |reference| runtime.resolve_audio_modification(reference),
            )
        }) else {
            return kARAFalse;
        };
        let Ok(bytes) = self.capabilities.store_objects(&filter) else {
            return kARAFalse;
        };
        if self.write_archive(writer, &bytes).is_ok() {
            kARATrue
        } else {
            kARAFalse
        }
    }

    fn get_processing_algorithm_properties(
        &mut self,
        algorithm_index: ARAInt32,
    ) -> *const ARAProcessingAlgorithmProperties {
        self.capabilities
            .algorithm_properties(algorithm_index)
            .unwrap_or(std::ptr::null())
    }

    fn get_processing_algorithm_for_audio_source(
        &mut self,
        source_reference: ARAAudioSourceRef,
    ) -> ARAInt32 {
        let Ok(runtime) = self.runtime() else {
            return 0;
        };
        let Ok(source) = runtime.resolve_audio_source(source_reference) else {
            return 0;
        };
        self.capabilities
            .active_algorithm(source.into_raw())
            .unwrap_or(0)
    }

    fn request_processing_algorithm_for_audio_source(
        &mut self,
        source_reference: ARAAudioSourceRef,
        algorithm_index: ARAInt32,
    ) {
        if self.runtime().map_or(true, |runtime| !runtime.is_editing()) {
            return;
        }
        let Ok(runtime) = self.runtime_mut() else {
            return;
        };
        let Ok(source) = runtime.resolve_audio_source(source_reference) else {
            return;
        };
        let _ = self
            .capabilities
            .request_algorithm(source.into_raw(), algorithm_index);
    }

    fn is_licensed_for_capabilities(
        &mut self,
        run_modal_activation: ARABool,
        content_types_count: ARASize,
        content_types: *const ARAContentType,
        transformation_flags: ARAPlaybackTransformationFlags,
    ) -> ARABool {
        let Some(supported) = self.capabilities.license_capabilities() else {
            return kARATrue;
        };
        if content_types_count > isize::MAX as usize
            || (content_types_count != 0 && content_types.is_null())
        {
            return kARAFalse;
        }
        let content_types = if content_types_count == 0 {
            &[]
        } else {
            // SAFETY: the ARA callback contract supplies `count` readable content-type values.
            unsafe { std::slice::from_raw_parts(content_types, content_types_count) }
        };
        if content_types
            .iter()
            .any(|content_type| !self.factory_analyzable_content.contains(content_type))
        {
            return kARAFalse;
        }
        let transformations =
            PlaybackTransformationFlags::from_bits_retain(transformation_flags as u32);
        if !self
            .factory_playback_transformations
            .contains(transformations)
        {
            return kARAFalse;
        }
        let Ok(request) = LicenseRequest::new(
            run_modal_activation != kARAFalse,
            content_types.iter().copied(),
            transformations,
            supported,
        ) else {
            return kARAFalse;
        };
        if self.capabilities.is_licensed(&request) {
            kARATrue
        } else {
            kARAFalse
        }
    }

    fn store_audio_source_to_audio_file_chunk(
        &mut self,
        archive_writer: ARAArchiveWriterHostRef,
        source_reference: ARAAudioSourceRef,
        document_archive_id: *mut ARAPersistentID,
        open_automatically: *mut ARABool,
    ) -> ARABool {
        if document_archive_id.is_null() || open_automatically.is_null() {
            return kARAFalse;
        }
        let Ok(runtime) = self.runtime() else {
            return kARAFalse;
        };
        if runtime.is_editing() || !self.content_readers.is_empty() {
            return kARAFalse;
        }
        let Ok(source) = runtime.resolve_audio_source(source_reference) else {
            return kARAFalse;
        };
        let Ok(chunk) = self.capabilities.store_audio_file_chunk(source.into_raw()) else {
            return kARAFalse;
        };
        let write = self
            .host
            .archives()
            .with_writer(archive_writer, |writer| writer.write_at(0, &chunk.bytes));
        if !matches!(write, Ok(Ok(()))) {
            return kARAFalse;
        }
        let Ok(id) = CString::new(chunk.document_archive_id) else {
            return kARAFalse;
        };
        // ARA requires this output to reuse one of the stable archive-ID pointers published by
        // the factory, not merely an equal temporary string.
        // SAFETY: controller creation validated and retained this factory and its metadata backing.
        let factory = unsafe { &*self.factory };
        let mut published_id = std::ptr::null();
        if !factory.documentArchiveID.is_null()
            // SAFETY: the factory contract retains this NUL-terminated ID for its lifetime.
            && unsafe { CStr::from_ptr(factory.documentArchiveID) } == id.as_c_str()
        {
            published_id = factory.documentArchiveID;
        } else if factory.compatibleDocumentArchiveIDsCount != 0
            && !factory.compatibleDocumentArchiveIDs.is_null()
        {
            // SAFETY: the validated factory count/pointer pair remains live for the controller.
            let compatible = unsafe {
                std::slice::from_raw_parts(
                    factory.compatibleDocumentArchiveIDs,
                    factory.compatibleDocumentArchiveIDsCount,
                )
            };
            published_id = compatible
                .iter()
                .copied()
                .find(|candidate| {
                    !candidate.is_null()
                        // SAFETY: every validated compatible ID is retained factory backing.
                        && unsafe { CStr::from_ptr(*candidate) } == id.as_c_str()
                })
                .unwrap_or(std::ptr::null());
        }
        if published_id.is_null() {
            return kARAFalse;
        }
        // SAFETY: non-null output pointers are writable for one scalar by the callback contract;
        // the selected ID pointer remains backed by the factory for its complete lifetime.
        unsafe {
            document_archive_id.write(published_id);
            open_automatically.write(if chunk.open_automatically {
                kARATrue
            } else {
                kARAFalse
            });
        }
        kARATrue
    }

    fn is_audio_modification_preserving_audio_source_signal(
        &mut self,
        modification_reference: ARAAudioModificationRef,
    ) -> ARABool {
        let Ok(runtime) = self.runtime_mut() else {
            return kARAFalse;
        };
        let Ok(modification) = runtime.resolve_audio_modification(modification_reference) else {
            return kARAFalse;
        };
        if self.capabilities.preserves_signal(modification.into_raw()) {
            kARATrue
        } else {
            kARAFalse
        }
    }
}

impl<P: PluginModel + 'static> Plugin<P> {
    /// Creates a stable raw document-controller instance from validated host and document input.
    ///
    /// # Safety
    ///
    /// `host`, `properties`, and `factory` must satisfy their ARA callback contracts. The host and
    /// factory must remain valid until the returned instance is destroyed through its vtable.
    pub unsafe fn create_document_controller(
        self,
        generation: ApiGeneration,
        factory: *const ARAFactory,
        host: *const ARADocumentControllerHostInstance,
        properties: *const ARADocumentProperties,
    ) -> Result<*const ARADocumentControllerInstance, AraError> {
        if factory.is_null() {
            return Err(AraError::InvalidArgument("factory pointer is null"));
        }
        // SAFETY: forwarded caller contract for the factory's advertised immutable prefix.
        let factory_input = unsafe { SizedInput::from_ptr(factory) }?;
        // SAFETY: generated offsets, field types, and extents describe `ARAFactory`.
        let analyzable_count = unsafe {
            factory_input.copy_field::<ARASize>(
                offset_of!(ARAFactory, analyzeableContentTypesCount),
                layout::ARAFACTORY_ANALYZEABLE_CONTENT_TYPES_COUNT,
            )?
        };
        if analyzable_count > 1024 {
            return Err(AraError::InvalidArgument(
                "factory analyzable content-type count exceeds safety limit",
            ));
        }
        // SAFETY: generated offsets, field types, and extents describe `ARAFactory`.
        let analyzable_pointer = unsafe {
            factory_input.copy_field::<*const ARAContentType>(
                offset_of!(ARAFactory, analyzeableContentTypes),
                layout::ARAFACTORY_ANALYZEABLE_CONTENT_TYPES,
            )?
        };
        // SAFETY: the factory contract keeps its represented content-type array readable for the
        // factory lifetime; this operation validates and immediately copies it.
        let factory_analyzable_content =
            unsafe { ForeignSlice::copy_from_raw(analyzable_pointer, analyzable_count)? }
                .into_vec()
                .into_iter()
                .collect::<HashSet<_>>();
        if !factory_analyzable_content.is_empty() && !self.capabilities.has_analysis() {
            return Err(AraError::InvalidArgument(
                "factory advertises analysis without an analysis provider",
            ));
        }
        // SAFETY: generated offsets, field types, and extents describe `ARAFactory`.
        let factory_playback_transformations =
            PlaybackTransformationFlags::from_bits_retain(unsafe {
                factory_input.copy_field::<ARAPlaybackTransformationFlags>(
                    offset_of!(ARAFactory, supportedPlaybackTransformationFlags),
                    layout::ARAFACTORY_SUPPORTED_PLAYBACK_TRANSFORMATION_FLAGS,
                )?
            } as u32);
        let factory_stores_chunks = if factory_input
            .contains_extent(layout::ARAFACTORY_SUPPORTS_STORING_AUDIO_FILE_CHUNKS)
        {
            // SAFETY: generated offsets, field types, and extents describe `ARAFactory`.
            (unsafe {
                factory_input.copy_field::<ARABool>(
                    offset_of!(ARAFactory, supportsStoringAudioFileChunks),
                    layout::ARAFACTORY_SUPPORTS_STORING_AUDIO_FILE_CHUNKS,
                )?
            }) != kARAFalse
        } else {
            false
        };
        if factory_stores_chunks != self.capabilities.has_audio_file_chunks() {
            return Err(AraError::InvalidArgument(
                "factory audio-file chunk flag disagrees with controller capability",
            ));
        }
        // SAFETY: forwarded caller contract; ARA host storage outlives the controller.
        let host = unsafe { HostClients::from_raw(host, generation) }?;
        if let Some(observer) = self.host_playback {
            observer(host.playback().map(|client| client.request_handle()));
        }
        // SAFETY: forwarded caller contract for the ephemeral document properties.
        let properties = unsafe { DocumentProperties::copy_from_ffi(properties) }?;
        // Optional controller-tail capabilities were introduced with ARA 2 Final. A plug-in may
        // still implement them while advertising a generation range that includes ARA 1 or the
        // ARA 2 draft; those older controllers expose only their generation-defined prefix.
        let advertised_capabilities = if generation >= ApiGeneration::V2Final {
            self.controller_capabilities
        } else {
            ControllerCapabilities::default()
        };
        let interface = document_controller_interface(generation, advertised_capabilities)?;
        let runtime = PluginRuntime::new(self.model, generation, properties)?;
        let adapter = ControllerAdapter {
            runtime: Some(runtime),
            // SAFETY: ARA guarantees the validated host instance and nested interfaces remain live
            // for the document-controller lifetime represented by this allocation.
            host: unsafe { std::mem::transmute::<HostClients<'_>, HostClients<'static>>(host) },
            capabilities: self.capabilities,
            factory,
            musical_context_hosts: HashMap::new(),
            audio_source_hosts: HashMap::new(),
            audio_modification_hosts: HashMap::new(),
            playback_region_hosts: HashMap::new(),
            synthetic_sequences: HashMap::new(),
            updates: self.updates,
            analysis: crate::AnalysisCoordinator::default(),
            analysis_events: self.analysis_events,
            pending_analysis_starts: Vec::new(),
            content_readers: HashSet::new(),
            pending_end_editing: false,
            deferred_end_editing_since: None,
            last_source_retry: None,
            legacy_restore_reader: None,
            factory_analyzable_content,
            factory_playback_transformations,
        };
        let instance = crate::ffi::callbacks::controller_instance(Box::new(adapter), interface);
        if let Some(observer) = self.controller_identity {
            // SAFETY: allocation 刚完成且尚未交宿主，复制 packed 身份字段不创建未对齐引用。
            let key = unsafe { (*instance).documentControllerRef } as usize;
            observer(key);
        }
        Ok(instance)
    }
}
