//! Host audio-access wrappers and long-lived reader ownership.

use ara2_bridge_core::{AraError, SizedInput};
use ara2_bridge_sys::*;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type Create = unsafe extern "C" fn(
    ARAAudioAccessControllerHostRef,
    ARAAudioSourceHostRef,
    ARABool,
) -> ARAAudioReaderHostRef;
type Read = unsafe extern "C" fn(
    ARAAudioAccessControllerHostRef,
    ARAAudioReaderHostRef,
    ARASamplePosition,
    ARASampleCount,
    *const *mut c_void,
) -> ARABool;
type Destroy = unsafe extern "C" fn(ARAAudioAccessControllerHostRef, ARAAudioReaderHostRef);

/// A sample scalar accepted by the ARA audio-reader service.
pub trait SampleFormat: private::Sealed + Copy {
    #[doc(hidden)]
    const USE_64_BIT: ARABool;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

impl SampleFormat for f32 {
    const USE_64_BIT: ARABool = kARAFalse;
}

impl SampleFormat for f64 {
    const USE_64_BIT: ARABool = kARATrue;
}

/// Required audio-access service supplied by the host.
#[derive(Clone)]
pub struct AudioAccess<'host> {
    host_ref: ARAAudioAccessControllerHostRef,
    create: Create,
    read: Read,
    destroy: Destroy,
    readers: Arc<ReaderRegistry>,
    _lifetime: PhantomData<&'host ()>,
}

impl<'host> AudioAccess<'host> {
    pub(crate) unsafe fn from_raw(
        host_ref: ARAAudioAccessControllerHostRef,
        interface: *const ARAAudioAccessControllerInterface,
    ) -> Result<Self, AraError> {
        if host_ref.is_null() || interface.is_null() {
            return Err(AraError::Abi("required audio-access service is null"));
        }
        // SAFETY: the caller supplies a readable interface for the returned lifetime.
        let input = unsafe { SizedInput::from_ptr(interface) }?;
        // SAFETY: each offset/type/extent triple names its generated packed field.
        let create = unsafe {
            input.copy_field::<Option<Create>>(
                offset_of!(ARAAudioAccessControllerInterface, createAudioReaderForSource),
                ara2_bridge_sys::layout::ARAAUDIO_ACCESS_CONTROLLER_INTERFACE_CREATE_AUDIO_READER_FOR_SOURCE,
            )
        }?
        .ok_or(AraError::Abi("audio reader creation callback is null"))?;
        // SAFETY: generated field triple as above.
        let read = unsafe {
            input.copy_field::<Option<Read>>(
                offset_of!(ARAAudioAccessControllerInterface, readAudioSamples),
                ara2_bridge_sys::layout::ARAAUDIO_ACCESS_CONTROLLER_INTERFACE_READ_AUDIO_SAMPLES,
            )
        }?
        .ok_or(AraError::Abi("audio sample callback is null"))?;
        // SAFETY: generated field triple as above.
        let destroy = unsafe {
            input.copy_field::<Option<Destroy>>(
                offset_of!(ARAAudioAccessControllerInterface, destroyAudioReader),
                ara2_bridge_sys::layout::ARAAUDIO_ACCESS_CONTROLLER_INTERFACE_DESTROY_AUDIO_READER,
            )
        }?
        .ok_or(AraError::Abi("audio reader destruction callback is null"))?;
        Ok(Self {
            host_ref,
            create,
            read,
            destroy,
            readers: Arc::new(ReaderRegistry::default()),
            _lifetime: PhantomData,
        })
    }

    pub(crate) fn reader<S: SampleFormat>(
        &self,
        source: ARAAudioSourceHostRef,
        channels: usize,
    ) -> Result<HostAudioReader<S>, AraError> {
        if source.is_null() || channels == 0 {
            return Err(AraError::InvalidArgument(
                "audio reader requires a source and channel count",
            ));
        }
        // SAFETY: constructor validation retained the callback and host ref; the source is current
        // under the dispatcher-issued scope that calls this private method.
        let reader = unsafe { (self.create)(self.host_ref, source, S::USE_64_BIT) };
        if reader.is_null() {
            return Err(AraError::Peer("host failed to create audio reader"));
        }
        let state = Arc::new(ReaderState {
            inner: Mutex::new(ReaderInner {
                host_ref: self.host_ref,
                reader,
                source_key: source as usize,
                channels,
                read: self.read,
                destroy: self.destroy,
                alive: true,
            }),
        });
        lock(&self.readers.readers).push(Arc::downgrade(&state));
        Ok(HostAudioReader {
            state,
            _sample: PhantomData,
        })
    }

    pub(crate) fn revoke_source(&self, source: ARAAudioSourceHostRef) {
        self.readers.revoke(Some(source as usize));
    }

    pub(crate) fn revoke_all(&self) {
        self.readers.revoke(None);
    }
}

#[derive(Default)]
struct ReaderRegistry {
    readers: Mutex<Vec<Weak<ReaderState>>>,
}

impl ReaderRegistry {
    fn revoke(&self, source_key: Option<usize>) {
        lock(&self.readers).retain(|reader| {
            let Some(reader) = reader.upgrade() else {
                return false;
            };
            let mut inner = lock(&reader.inner);
            if source_key.is_none_or(|source| source == inner.source_key) {
                inner.revoke();
            }
            true
        });
    }
}

struct ReaderState {
    inner: Mutex<ReaderInner>,
}

struct ReaderInner {
    host_ref: ARAAudioAccessControllerHostRef,
    reader: ARAAudioReaderHostRef,
    source_key: usize,
    channels: usize,
    read: Read,
    destroy: Destroy,
    alive: bool,
}

impl ReaderInner {
    fn revoke(&mut self) {
        if self.alive {
            // SAFETY: the registry serializes access and consumes this live host reader once.
            unsafe { (self.destroy)(self.host_ref, self.reader) };
            self.alive = false;
        }
    }
}

impl Drop for ReaderState {
    fn drop(&mut self) {
        self.inner
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .revoke();
    }
}

// SAFETY: opaque ARA reader references may move between non-realtime threads; every operation and
// the unique destruction transition are serialized by `ReaderState::inner`.
unsafe impl Send for ReaderState {}
// SAFETY: shared access cannot cross the host ABI concurrently because all methods lock `inner`.
unsafe impl Sync for ReaderState {}

/// Owned host audio reader synchronously revoked by its controller when source access ends.
pub struct HostAudioReader<S: SampleFormat> {
    state: Arc<ReaderState>,
    _sample: PhantomData<S>,
}

impl<S: SampleFormat> HostAudioReader<S> {
    /// Reads equal-length planar channel buffers from the host.
    pub fn read(&mut self, sample_position: i64, buffers: &mut [&mut [S]]) -> Result<(), AraError> {
        let inner = lock(&self.state.inner);
        if !inner.alive {
            return Err(AraError::InvalidState("host audio reader has been revoked"));
        }
        if buffers.len() != inner.channels || buffers.is_empty() {
            return Err(AraError::InvalidArgument("audio channel count mismatch"));
        }
        let sample_count = buffers[0].len();
        if buffers.iter().any(|buffer| buffer.len() != sample_count) {
            return Err(AraError::InvalidArgument(
                "audio channel buffers must have equal lengths",
            ));
        }
        let count = i64::try_from(sample_count)
            .map_err(|_| AraError::InvalidArgument("audio read length overflow"))?;
        let pointers = buffers
            .iter_mut()
            .map(|buffer| buffer.as_mut_ptr().cast::<c_void>())
            .collect::<Vec<_>>();
        // SAFETY: the owned reader is live and the planar buffers are writable for `count` samples.
        let result = unsafe {
            (inner.read)(
                inner.host_ref,
                inner.reader,
                sample_position,
                count,
                pointers.as_ptr(),
            )
        };
        if result != kARAFalse {
            Ok(())
        } else {
            Err(AraError::Peer("host audio read failed"))
        }
    }
}
