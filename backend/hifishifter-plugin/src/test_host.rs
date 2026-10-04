//! 测试用完整 ARA 宿主边界：稳定接口存储、真实 PCM 读取和 reader 生命周期计数。

use ara2_bridge::sys::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

static mut ASSERT_FUNCTION: ARAAssertFunction = None;
/// 相同 ARA generation 的测试工厂必须共享同一 assert-function 地址。
pub(crate) fn assert_address() -> *mut ARAAssertFunction {
    &raw mut ASSERT_FUNCTION
}

pub(crate) struct HostFixture {
    pub planes: Vec<Vec<f32>>,
    pub created: AtomicUsize,
    pub destroyed: AtomicUsize,
    audio: Box<ARAAudioAccessControllerInterface>,
    archive: Box<ARAArchivingControllerInterface>,
}

impl HostFixture {
    /// Box 保证宿主上下文地址稳定，即使外层测试移动所有权也不影响 reader。
    pub fn new(planes: Vec<Vec<f32>>) -> Box<Self> {
        Box::new(Self {
            planes,
            created: AtomicUsize::new(0),
            destroyed: AtomicUsize::new(0),
            audio: Box::new(ARAAudioAccessControllerInterface {
                structSize: std::mem::size_of::<ARAAudioAccessControllerInterface>(),
                createAudioReaderForSource: Some(create_reader),
                readAudioSamples: Some(read_samples),
                destroyAudioReader: Some(destroy_reader),
            }),
            archive: Box::new(ARAArchivingControllerInterface {
                structSize: std::mem::size_of::<ARAArchivingControllerInterface>(),
                getArchiveSize: Some(archive_size),
                readBytesFromArchive: Some(archive_read),
                writeBytesToArchive: Some(archive_write),
                notifyDocumentArchivingProgress: Some(progress),
                notifyDocumentUnarchivingProgress: Some(progress),
                getDocumentArchiveID: Some(archive_id),
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
            archivingControllerInterface: &*self.archive,
            contentAccessControllerHostRef: std::ptr::null_mut(),
            contentAccessControllerInterface: std::ptr::null(),
            modelUpdateControllerHostRef: std::ptr::null_mut(),
            modelUpdateControllerInterface: std::ptr::null(),
            playbackControllerHostRef: std::ptr::null_mut(),
            playbackControllerInterface: std::ptr::null(),
        }
    }
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
    unsafe { &*context.cast::<HostFixture>() }
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
    let fixture = unsafe { &*context.cast::<HostFixture>() };
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
    unsafe { &*context.cast::<HostFixture>() }
        .destroyed
        .fetch_add(1, Ordering::Relaxed);
}
unsafe extern "C" fn archive_size(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
) -> usize {
    0
}
unsafe extern "C" fn archive_read(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
    _: usize,
    _: usize,
    _: *mut u8,
) -> ARABool {
    0
}
unsafe extern "C" fn archive_write(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveWriterHostRef,
    _: usize,
    _: usize,
    _: *const u8,
) -> ARABool {
    0
}
unsafe extern "C" fn progress(_: ARAArchivingControllerHostRef, _: f32) {}
unsafe extern "C" fn archive_id(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
) -> ARAPersistentID {
    c"org.hifishifter.test".as_ptr()
}
