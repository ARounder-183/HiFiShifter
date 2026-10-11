use ara2_bridge_core::{ApiGeneration, AraError};
use ara2_bridge_host::{
    ArchiveReaderId, ArchiveWriterId, ArchivingProvider, AudioAccessProvider, AudioSourceId,
    HostAudioReader, HostServicesBuilder,
};
use ara2_bridge_sys::{
    access::read_field, kARAFalse, kARATrue, ARAAudioAccessControllerHostRef,
    ARAAudioAccessControllerInterface, ARAAudioReaderHostRef, ARAAudioSourceHostRef, ARABool,
    ARASampleCount, ARASamplePosition,
};
use std::ffi::c_void;
use std::mem::offset_of;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

struct EmptyArchive;

impl ArchivingProvider for EmptyArchive {
    fn len(&self, _: ArchiveReaderId) -> Result<usize, AraError> {
        Ok(0)
    }

    fn read_at(&self, _: ArchiveReaderId, _: usize, _: &mut [u8]) -> Result<(), AraError> {
        Ok(())
    }

    fn write_at(&self, _: ArchiveWriterId, _: usize, _: &[u8]) -> Result<(), AraError> {
        Ok(())
    }
}

struct BlockingAudio {
    gate: Arc<(Mutex<(bool, bool)>, Condvar)>,
    reads: Arc<AtomicUsize>,
    source: Arc<Mutex<Option<AudioSourceId>>>,
}

impl AudioAccessProvider for BlockingAudio {
    fn create_reader(
        &self,
        source: AudioSourceId,
        _: bool,
    ) -> Result<Box<dyn HostAudioReader>, AraError> {
        *self.source.lock().unwrap() = Some(source);
        Ok(Box::new(BlockingReader {
            gate: Arc::clone(&self.gate),
            reads: Arc::clone(&self.reads),
        }))
    }
}

struct BlockingReader {
    gate: Arc<(Mutex<(bool, bool)>, Condvar)>,
    reads: Arc<AtomicUsize>,
}

impl HostAudioReader for BlockingReader {
    fn channel_count(&self) -> usize {
        1
    }

    fn sample_count(&self) -> i64 {
        1
    }

    fn read_f32(&mut self, _: i64, buffers: &mut [&mut [f32]]) -> Result<(), AraError> {
        self.reads.fetch_add(1, Ordering::AcqRel);
        let (mutex, condition) = &*self.gate;
        let mut state = mutex.lock().unwrap();
        state.0 = true;
        condition.notify_all();
        while !state.1 {
            state = condition.wait(state).unwrap();
        }
        buffers[0][0] = 1.0;
        Ok(())
    }
}

#[test]
fn access_revocation_waits_for_a_real_in_flight_reader_and_rejects_stale_use() {
    let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let reads = Arc::new(AtomicUsize::new(0));
    let source_id = Arc::new(Mutex::new(None));
    let services = Arc::new(
        HostServicesBuilder::new()
            .audio(BlockingAudio {
                gate: Arc::clone(&gate),
                reads: Arc::clone(&reads),
                source: Arc::clone(&source_id),
            })
            .archiving(EmptyArchive)
            .build(ApiGeneration::V23Final)
            .unwrap(),
    );
    let instance = services.instance();
    let interface = instance.audioAccessControllerInterface;
    let create = callback::<
        unsafe extern "C" fn(
            ARAAudioAccessControllerHostRef,
            ARAAudioSourceHostRef,
            ARABool,
        ) -> ARAAudioReaderHostRef,
    >(
        interface,
        offset_of!(
            ARAAudioAccessControllerInterface,
            createAudioReaderForSource
        ),
    );
    let read = callback::<
        unsafe extern "C" fn(
            ARAAudioAccessControllerHostRef,
            ARAAudioReaderHostRef,
            ARASamplePosition,
            ARASampleCount,
            *const *mut c_void,
        ) -> ARABool,
    >(
        interface,
        offset_of!(ARAAudioAccessControllerInterface, readAudioSamples),
    );
    let source = Arc::new(0_u8);
    let source_ref = Arc::as_ptr(&source).cast_mut().cast();
    // SAFETY: host state and the source allocation remain live through the callback.
    let reader = unsafe { create(instance.audioAccessControllerHostRef, source_ref, kARAFalse) };
    assert!(!reader.is_null());

    let reader_address = reader as usize;
    let host_address = instance.audioAccessControllerHostRef as usize;
    let reading = std::thread::spawn(move || {
        let mut output = [0.0_f32];
        let planes = [output.as_mut_ptr().cast()];
        // SAFETY: reconstructed addresses remain live and `planes` contains one writable sample.
        let result = unsafe {
            read(
                host_address as ARAAudioAccessControllerHostRef,
                reader_address as ARAAudioReaderHostRef,
                0,
                1,
                planes.as_ptr(),
            )
        };
        (result, output)
    });

    let (mutex, condition) = &*gate;
    let mut state = mutex.lock().unwrap();
    while !state.0 {
        state = condition.wait(state).unwrap();
    }
    drop(state);

    let revocation_started = Arc::new(AtomicBool::new(false));
    let revoking_services = Arc::clone(&services);
    let revoking_started = Arc::clone(&revocation_started);
    let source_id = source_id.lock().unwrap().unwrap();
    let revoking = std::thread::spawn(move || {
        revoking_started.store(true, Ordering::Release);
        revoking_services.revoke_audio_source_readers(source_id);
    });
    while !revocation_started.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    for _ in 0..16 {
        std::thread::yield_now();
    }

    let mut state = mutex.lock().unwrap();
    state.1 = true;
    condition.notify_all();
    drop(state);

    let (result, output) = reading.join().unwrap();
    revoking.join().unwrap();
    assert_eq!(result, kARATrue);
    assert_eq!(output, [1.0]);
    assert_eq!(reads.load(Ordering::Acquire), 1);

    let mut stale_output = [9.0_f32];
    let stale_planes = [stale_output.as_mut_ptr().cast()];
    // SAFETY: the stale identity is passed only for registry rejection; output storage is valid.
    assert_eq!(
        unsafe {
            read(
                instance.audioAccessControllerHostRef,
                reader,
                0,
                1,
                stale_planes.as_ptr(),
            )
        },
        kARAFalse
    );
}

fn callback<T: Copy>(interface: *const ARAAudioAccessControllerInterface, offset: usize) -> T {
    // SAFETY: the complete live vtable represents every required audio callback.
    unsafe { read_field::<Option<T>>(interface.cast(), offset) }.unwrap()
}
