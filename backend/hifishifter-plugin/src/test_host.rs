//! 测试用完整 ARA 宿主边界：稳定接口存储、真实 PCM 读取、reader 生命周期计数，
//! 以及**全部**可选控制器（内容访问 / 模型更新 / 播放）与内存归档。
//!
//! 【为什么要把三个控制器也接上】此前它们全是 `null`：宿主只提供音频访问与归档，
//! 于是"插件到底有没有请求播放/上报模型更新"在测试里**不可观测**，任何一处接线错误
//! （回调签名、hostRef、生命周期）都不会被测出来。接上之后测试可以断言"请求确实到达
//! 宿主"，也能覆盖播放/归档这两条真实路径。
//!
//! 【为什么不追求逐点复刻 ARA 参考宿主】本 fixture 是**最小可观测宿主**：音频读取真实、
//! 计数真实、归档往返真实；内容访问（音符/和声/速度注解）故意回"不可用" —— 那不是
//! 简化，是本仓插件不消费这些注解的诚实表态（真消费时再补）。

use crate::sync::MutexExt;
use ara2_bridge::sys::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

static mut ASSERT_FUNCTION: ARAAssertFunction = None;
/// 相同 ARA generation 的测试工厂必须共享同一 assert-function 地址。
pub(crate) fn assert_address() -> *mut ARAAssertFunction {
    &raw mut ASSERT_FUNCTION
}

pub(crate) struct HostFixture {
    pub planes: Vec<Vec<f32>>,
    pub created: AtomicUsize,
    pub destroyed: AtomicUsize,
    /// 内存归档缓冲：插件写进去的 document 数据从这里原样读回（round-trip）。
    archive_bytes: Mutex<Vec<u8>>,
    /// 播放控制器调用计数与最后一次请求的位置。
    pub playback_start: AtomicUsize,
    pub playback_stop: AtomicUsize,
    pub playback_position: AtomicUsize,
    pub playback_position_bits: AtomicU64,
    /// 模型更新通知计数（任一 notify* 回调）。
    pub model_updates: AtomicUsize,
    /// 内容访问查询计数。
    pub content_queries: AtomicUsize,
    /// 归档读写计数。
    pub archive_writes: AtomicUsize,
    pub archive_reads: AtomicUsize,
    audio: Box<ARAAudioAccessControllerInterface>,
    archiving: Box<ARAArchivingControllerInterface>,
    content: Box<ARAContentAccessControllerInterface>,
    model_update: Box<ARAModelUpdateControllerInterface>,
    playback: Box<ARAPlaybackControllerInterface>,
}

impl HostFixture {
    /// Box 保证宿主上下文地址稳定，即使外层测试移动所有权也不影响 reader。
    pub fn new(planes: Vec<Vec<f32>>) -> Box<Self> {
        Box::new(Self {
            planes,
            created: AtomicUsize::new(0),
            destroyed: AtomicUsize::new(0),
            archive_bytes: Mutex::new(Vec::new()),
            playback_start: AtomicUsize::new(0),
            playback_stop: AtomicUsize::new(0),
            playback_position: AtomicUsize::new(0),
            playback_position_bits: AtomicU64::new(0),
            model_updates: AtomicUsize::new(0),
            content_queries: AtomicUsize::new(0),
            archive_writes: AtomicUsize::new(0),
            archive_reads: AtomicUsize::new(0),
            audio: Box::new(ARAAudioAccessControllerInterface {
                structSize: std::mem::size_of::<ARAAudioAccessControllerInterface>(),
                createAudioReaderForSource: Some(create_reader),
                readAudioSamples: Some(read_samples),
                destroyAudioReader: Some(destroy_reader),
            }),
            archiving: Box::new(ARAArchivingControllerInterface {
                structSize: std::mem::size_of::<ARAArchivingControllerInterface>(),
                getArchiveSize: Some(archive_size),
                readBytesFromArchive: Some(archive_read),
                writeBytesToArchive: Some(archive_write),
                notifyDocumentArchivingProgress: Some(progress),
                notifyDocumentUnarchivingProgress: Some(progress),
                getDocumentArchiveID: Some(archive_id),
            }),
            content: Box::new(ARAContentAccessControllerInterface {
                structSize: std::mem::size_of::<ARAContentAccessControllerInterface>(),
                isMusicalContextContentAvailable: Some(content_unavailable_musical),
                getMusicalContextContentGrade: Some(content_grade_musical),
                createMusicalContextContentReader: Some(create_null_reader_musical),
                isAudioSourceContentAvailable: Some(content_unavailable_audio),
                getAudioSourceContentGrade: Some(content_grade_audio),
                createAudioSourceContentReader: Some(create_null_reader_audio),
                getContentReaderEventCount: Some(content_reader_event_count),
                getContentReaderDataForEvent: Some(content_reader_data),
                destroyContentReader: Some(destroy_content_reader),
            }),
            model_update: Box::new(ARAModelUpdateControllerInterface {
                structSize: std::mem::size_of::<ARAModelUpdateControllerInterface>(),
                notifyAudioSourceAnalysisProgress: Some(model_update_analysis_progress),
                notifyAudioSourceContentChanged: Some(model_update_source),
                notifyAudioModificationContentChanged: Some(model_update_modification),
                notifyPlaybackRegionContentChanged: Some(model_update_region),
                notifyDocumentDataChanged: Some(model_update_document),
            }),
            playback: Box::new(ARAPlaybackControllerInterface {
                structSize: std::mem::size_of::<ARAPlaybackControllerInterface>(),
                requestStartPlayback: Some(playback_start),
                requestStopPlayback: Some(playback_stop),
                requestSetPlaybackPosition: Some(playback_set_position),
                requestSetCycleRange: Some(playback_cycle_range),
                requestEnableCycle: Some(playback_enable_cycle),
            }),
        })
    }

    pub fn instance(&mut self) -> ARADocumentControllerHostInstance {
        let context = self as *mut Self;
        ARADocumentControllerHostInstance {
            structSize: std::mem::size_of::<ARADocumentControllerHostInstance>(),
            audioAccessControllerHostRef: context.cast(),
            audioAccessControllerInterface: &*self.audio,
            archivingControllerHostRef: context.cast(),
            archivingControllerInterface: &*self.archiving,
            contentAccessControllerHostRef: context.cast(),
            contentAccessControllerInterface: &*self.content,
            modelUpdateControllerHostRef: context.cast(),
            modelUpdateControllerInterface: &*self.model_update,
            playbackControllerHostRef: context.cast(),
            playbackControllerInterface: &*self.playback,
        }
    }

    /// 尚未释放的 ARA 音频 reader 数量（必须为 0，否则是对象泄漏）。
    pub fn live_readers(&self) -> usize {
        self.created.load(Ordering::Acquire) - self.destroyed.load(Ordering::Acquire)
    }

    /// 归档缓冲的当前内容（round-trip 断言用）。
    pub fn archived(&self) -> Vec<u8> {
        self.archive_bytes.lock_or_recover().clone()
    }

    /// 上下文指针，供测试直接调用回调。
    pub fn context(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }
}

/// 从宿主回调里的 `controllerHostRef` 取回 fixture。
///
/// # Safety
/// 调用方保证该引用是 `HostFixture::instance()` 装进去的稳定指针，且 fixture 存活。
unsafe fn fixture<'a>(context: *mut c_void) -> &'a HostFixture {
    // SAFETY: 由调用方保证（见函数文档）。
    unsafe { &*context.cast::<HostFixture>() }
}

unsafe extern "C" fn create_reader(
    context: ARAAudioAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    use64: ARABool,
) -> ARAAudioReaderHostRef {
    if use64 != 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: 测试 fixture 持有稳定上下文到控制器销毁。
    unsafe { fixture(context.cast()) }
        .created
        .fetch_add(1, Ordering::Relaxed);
    context.cast()
}
unsafe extern "C" fn read_samples(
    context: ARAAudioAccessControllerHostRef,
    _: ARAAudioReaderHostRef,
    position: i64,
    frames: i64,
    buffers: *const *mut c_void,
) -> ARABool {
    // SAFETY: fixture 及宿主 reader 在调用期间存活；缓冲由 bridge 按通道数创建。
    let fixture = unsafe { fixture(context.cast()) };
    if position < 0 || frames < 0 || buffers.is_null() {
        return 0;
    }
    let Some(end) = (position as usize).checked_add(frames as usize) else {
        return 0;
    };
    if fixture.planes.iter().any(|plane| end > plane.len()) {
        return 0;
    }
    for (channel, plane) in fixture.planes.iter().enumerate() {
        // SAFETY: bridge 提供足够长度的独立输出平面。
        unsafe {
            std::ptr::copy_nonoverlapping(
                plane.as_ptr().add(position as usize),
                (*buffers.add(channel)).cast::<f32>(),
                frames as usize,
            )
        };
    }
    1
}
unsafe extern "C" fn destroy_reader(
    context: ARAAudioAccessControllerHostRef,
    _: ARAAudioReaderHostRef,
) {
    // SAFETY: fixture 保留到全部 reader 已释放。
    unsafe { fixture(context.cast()) }
        .destroyed
        .fetch_add(1, Ordering::Relaxed);
}

// ── 归档：内存 round-trip ─────────────────────────────────────────────────────

unsafe extern "C" fn archive_size(
    context: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
) -> usize {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { fixture(context.cast()) }
        .archive_bytes
        .lock()
        .unwrap()
        .len()
}
unsafe extern "C" fn archive_read(
    context: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
    offset: usize,
    size: usize,
    buffer: *mut u8,
) -> ARABool {
    if buffer.is_null() {
        return 0;
    }
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    let fixture = unsafe { fixture(context.cast()) };
    let archive = fixture.archive_bytes.lock_or_recover();
    let Some(end) = offset.checked_add(size) else {
        return 0;
    };
    if end > archive.len() {
        return 0;
    }
    // SAFETY: 已确认 [offset, end) 落在归档缓冲内，buffer 由宿主提供 size 字节可写空间。
    unsafe {
        std::ptr::copy_nonoverlapping(archive.as_ptr().add(offset), buffer, size);
    }
    fixture.archive_reads.fetch_add(1, Ordering::Relaxed);
    1
}
unsafe extern "C" fn archive_write(
    context: ARAArchivingControllerHostRef,
    _: ARAArchiveWriterHostRef,
    offset: usize,
    size: usize,
    data: *const u8,
) -> ARABool {
    if data.is_null() {
        return 0;
    }
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    let fixture = unsafe { fixture(context.cast()) };
    let mut archive = fixture.archive_bytes.lock_or_recover();
    let Some(end) = offset.checked_add(size) else {
        return 0;
    };
    if archive.len() < end {
        archive.resize(end, 0);
    }
    // SAFETY: 已确认 [offset, end) 在（可能刚扩容的）缓冲内，data 由宿主提供 size 字节可读空间。
    unsafe {
        std::ptr::copy_nonoverlapping(data, archive.as_mut_ptr().add(offset), size);
    }
    fixture.archive_writes.fetch_add(1, Ordering::Relaxed);
    1
}
unsafe extern "C" fn progress(_: ARAArchivingControllerHostRef, _: f32) {}
unsafe extern "C" fn archive_id(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
) -> ARAPersistentID {
    c"org.hifishifter.test".as_ptr()
}

// ── 内容访问：诚实的"不可用" ─────────────────────────────────────────────────

/// 记一次内容查询并回"不可用"（0）。
///
/// # Safety
/// `context` 必须是稳定的 fixture 指针。
unsafe fn count_content_query(context: *mut c_void) {
    // SAFETY: 由调用方保证。
    unsafe { fixture(context) }
        .content_queries
        .fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn content_unavailable_musical(
    context: ARAContentAccessControllerHostRef,
    _: ARAMusicalContextHostRef,
    _: ARAContentType,
) -> ARABool {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_content_query(context.cast()) };
    0
}
unsafe extern "C" fn content_unavailable_audio(
    context: ARAContentAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAContentType,
) -> ARABool {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_content_query(context.cast()) };
    0
}
unsafe extern "C" fn content_grade_musical(
    context: ARAContentAccessControllerHostRef,
    _: ARAMusicalContextHostRef,
    _: ARAContentType,
) -> ARAContentGrade {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_content_query(context.cast()) };
    0
}
unsafe extern "C" fn content_grade_audio(
    context: ARAContentAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAContentType,
) -> ARAContentGrade {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_content_query(context.cast()) };
    0
}
unsafe extern "C" fn create_null_reader_musical(
    context: ARAContentAccessControllerHostRef,
    _: ARAMusicalContextHostRef,
    _: ARAContentType,
    _: *const ARAContentTimeRange,
) -> ARAContentReaderHostRef {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_content_query(context.cast()) };
    std::ptr::null_mut()
}
unsafe extern "C" fn create_null_reader_audio(
    context: ARAContentAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAContentType,
    _: *const ARAContentTimeRange,
) -> ARAContentReaderHostRef {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_content_query(context.cast()) };
    std::ptr::null_mut()
}
unsafe extern "C" fn content_reader_event_count(
    _: ARAContentAccessControllerHostRef,
    _: ARAContentReaderHostRef,
) -> ARAInt32 {
    0
}
unsafe extern "C" fn content_reader_data(
    _: ARAContentAccessControllerHostRef,
    _: ARAContentReaderHostRef,
    _: ARAInt32,
) -> *const c_void {
    std::ptr::null()
}
unsafe extern "C" fn destroy_content_reader(
    _: ARAContentAccessControllerHostRef,
    _: ARAContentReaderHostRef,
) {
}

// ── 模型更新：全部计数 ───────────────────────────────────────────────────────

/// 记一次模型更新通知。
///
/// # Safety
/// `context` 必须是稳定的 fixture 指针。
unsafe fn count_model_update(context: *mut c_void) {
    // SAFETY: 由调用方保证。
    unsafe { fixture(context) }
        .model_updates
        .fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn model_update_analysis_progress(
    context: ARAModelUpdateControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAAnalysisProgressState,
    _: f32,
) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_model_update(context.cast()) };
}
unsafe extern "C" fn model_update_source(
    context: ARAModelUpdateControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: *const ARAContentTimeRange,
    _: ARAContentUpdateFlags,
) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_model_update(context.cast()) };
}
unsafe extern "C" fn model_update_modification(
    context: ARAModelUpdateControllerHostRef,
    _: ARAAudioModificationHostRef,
    _: *const ARAContentTimeRange,
    _: ARAContentUpdateFlags,
) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_model_update(context.cast()) };
}
unsafe extern "C" fn model_update_region(
    context: ARAModelUpdateControllerHostRef,
    _: ARAPlaybackRegionHostRef,
    _: *const ARAContentTimeRange,
    _: ARAContentUpdateFlags,
) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_model_update(context.cast()) };
}
unsafe extern "C" fn model_update_document(context: ARAModelUpdateControllerHostRef) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { count_model_update(context.cast()) };
}

// ── 播放：计数 + 记录最后一次请求 ────────────────────────────────────────────

unsafe extern "C" fn playback_start(context: ARAPlaybackControllerHostRef) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { fixture(context.cast()) }
        .playback_start
        .fetch_add(1, Ordering::Relaxed);
}
unsafe extern "C" fn playback_stop(context: ARAPlaybackControllerHostRef) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { fixture(context.cast()) }
        .playback_stop
        .fetch_add(1, Ordering::Relaxed);
}
unsafe extern "C" fn playback_set_position(
    context: ARAPlaybackControllerHostRef,
    position: ARATimePosition,
) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    let fixture = unsafe { fixture(context.cast()) };
    fixture.playback_position.fetch_add(1, Ordering::Relaxed);
    fixture
        .playback_position_bits
        .store(position.to_bits(), Ordering::Relaxed);
}
unsafe extern "C" fn playback_cycle_range(
    context: ARAPlaybackControllerHostRef,
    _: ARATimePosition,
    _: ARATimeDuration,
) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { fixture(context.cast()) }
        .playback_position
        .fetch_add(1, Ordering::Relaxed);
}
unsafe extern "C" fn playback_enable_cycle(context: ARAPlaybackControllerHostRef, _: ARABool) {
    // SAFETY: 由调用方保证的稳定 fixture 指针。
    unsafe { fixture(context.cast()) }
        .playback_position
        .fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::HostClients;

    /// 归档回调必须真的能往返：写进去多少字节、读到多少、内容一致。
    #[test]
    fn the_in_memory_archive_round_trips_bytes() {
        let mut fixture = HostFixture::new(vec![vec![0.0]]);
        let context = fixture.context();
        // SAFETY: fixture 存活到读取结束；context 是稳定指针。
        unsafe {
            let payload = b"hifishifter-archive";
            assert_eq!(
                archive_write(
                    context.cast(),
                    std::ptr::null_mut(),
                    4,
                    payload.len(),
                    payload.as_ptr()
                ),
                1
            );
            assert_eq!(
                archive_size(context.cast(), std::ptr::null_mut()),
                4 + payload.len()
            );
            let mut out = vec![0u8; payload.len()];
            assert_eq!(
                archive_read(
                    context.cast(),
                    std::ptr::null_mut(),
                    4,
                    payload.len(),
                    out.as_mut_ptr()
                ),
                1
            );
            assert_eq!(&out, payload);
            // 越界读必须失败，而不是读越界内存。
            assert_eq!(
                archive_read(
                    context.cast(),
                    std::ptr::null_mut(),
                    1,
                    payload.len() + 100,
                    out.as_mut_ptr()
                ),
                0
            );
        }
        assert_eq!(fixture.archive_writes.load(Ordering::Acquire), 1);
        assert_eq!(fixture.archive_reads.load(Ordering::Acquire), 1);
        assert!(fixture.archived().len() >= 4);
    }

    /// 播放/模型更新/内容访问回调都必须可达并计数 —— 它们是"插件真的调用了宿主"的证据。
    #[test]
    fn the_optional_controllers_are_reachable_and_counted() {
        let mut fixture = HostFixture::new(vec![vec![0.0]]);
        let context = fixture.context();
        // SAFETY: fixture 存活到调用结束；context 是稳定指针。
        unsafe {
            playback_start(context.cast());
            playback_stop(context.cast());
            playback_set_position(context.cast(), 1.25);
            model_update_document(context.cast());
            content_unavailable_audio(context.cast(), std::ptr::null_mut(), 0);
        }
        assert_eq!(fixture.playback_start.load(Ordering::Acquire), 1);
        assert_eq!(fixture.playback_stop.load(Ordering::Acquire), 1);
        assert_eq!(fixture.playback_position.load(Ordering::Acquire), 1);
        assert_eq!(
            f64::from_bits(fixture.playback_position_bits.load(Ordering::Acquire)),
            1.25
        );
        assert_eq!(fixture.model_updates.load(Ordering::Acquire), 1);
        assert_eq!(fixture.content_queries.load(Ordering::Acquire), 1);
    }

    /// 同一个 fixture 对每个 generation 都必须能构造出可用的 `HostClients`，
    /// 且不留下未释放对象 —— 这是"宿主契约在 2.0 与 2.3 上都成立"的最小矩阵。
    #[test]
    fn every_generation_constructs_host_clients_without_leaking_objects() {
        for generation in [ApiGeneration::V2Final, ApiGeneration::V23Final] {
            let mut fixture = HostFixture::new(vec![vec![0.1, 0.2, 0.3, 0.4]]);
            let host = fixture.instance();
            // SAFETY: fixture 与 host 在 clients 释放前一直存活。
            let clients = unsafe { HostClients::from_raw(&host, generation) };
            assert!(
                clients.is_ok(),
                "host fixture must satisfy the {generation:?} host contract"
            );
            drop(clients);
            assert_eq!(
                fixture.live_readers(),
                0,
                "no ARA reader may outlive the clients"
            );
        }
    }
}
