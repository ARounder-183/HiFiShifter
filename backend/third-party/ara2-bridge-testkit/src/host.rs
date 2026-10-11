//! Deterministic public-API ARA host used by conformance scenarios.

use ara2_bridge_core::{
    ApiGeneration, AraError, AudioSourceKind, ContentGrade, ContentKind, ContentTimeRange,
    ModelRef, NoteEvent, Notes,
};
use ara2_bridge_host::{
    ArchiveReaderId, ArchiveWriterId, ArchivingProvider, AudioAccessProvider, AudioSourceId,
    ContentAccessProvider, HostAudioReader, HostContentReaderSnapshot, HostContentSnapshot,
    HostServices, HostServicesBuilder, LoadedFactory, ModelUpdateProvider, MusicalContextId,
    PlaybackProvider,
};
use ara2_bridge_plugin::Factory;
use ara2_bridge_sys::{
    access::read_field, kARAFalse, ARAAssertCategory, ARAAudioAccessControllerHostRef,
    ARAAudioAccessControllerInterface, ARAAudioReaderHostRef, ARAAudioSourceHostRef, ARABool,
    ARASampleCount, ARASamplePosition,
};
use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::mem::offset_of;
use std::sync::{Arc, Mutex, MutexGuard};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One deterministic semantic callback observed by [`TestHost`].
#[derive(Clone, Debug, PartialEq)]
pub struct TestHostEvent {
    sequence: usize,
    generation: ApiGeneration,
    operation: &'static str,
    object: Option<usize>,
    detail: String,
}

impl TestHostEvent {
    /// Returns the zero-based event order.
    pub const fn sequence(&self) -> usize {
        self.sequence
    }

    /// Returns the API generation associated with the callback.
    pub const fn generation(&self) -> ApiGeneration {
        self.generation
    }

    /// Returns the semantic operation name.
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the stable scenario-local object number, when applicable.
    pub const fn object(&self) -> Option<usize> {
        self.object
    }

    /// Returns deterministic operation detail.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

#[derive(Default)]
struct TraceState {
    events: Vec<TestHostEvent>,
    source_ids: HashMap<AudioSourceId, usize>,
}

/// Cloneable ordered trace for deterministic TestHost assertions.
#[derive(Clone)]
pub struct TestHostTrace {
    generation: ApiGeneration,
    state: Arc<Mutex<TraceState>>,
}

impl TestHostTrace {
    fn new(generation: ApiGeneration) -> Self {
        Self {
            generation,
            state: Arc::new(Mutex::new(TraceState::default())),
        }
    }

    fn source_object(&self, source: AudioSourceId) -> usize {
        let mut state = lock(&self.state);
        let next = state.source_ids.len();
        *state.source_ids.entry(source).or_insert(next)
    }

    fn record(&self, operation: &'static str, object: Option<usize>, detail: impl Into<String>) {
        let mut state = lock(&self.state);
        let sequence = state.events.len();
        state.events.push(TestHostEvent {
            sequence,
            generation: self.generation,
            operation,
            object,
            detail: detail.into(),
        });
    }

    /// Returns a stable snapshot of all callbacks observed so far.
    pub fn events(&self) -> Vec<TestHostEvent> {
        lock(&self.state).events.clone()
    }

    /// Counts callbacks with one semantic operation name.
    pub fn count(&self, operation: &str) -> usize {
        lock(&self.state)
            .events
            .iter()
            .filter(|event| event.operation == operation)
            .count()
    }
}

#[derive(Clone)]
struct TestAudio {
    trace: TestHostTrace,
}

impl AudioAccessProvider for TestAudio {
    fn create_reader(
        &self,
        source: AudioSourceId,
        use_64_bit_samples: bool,
    ) -> Result<Box<dyn HostAudioReader>, AraError> {
        let object = self.trace.source_object(source);
        self.trace.record(
            "create_audio_reader",
            Some(object),
            if use_64_bit_samples { "f64" } else { "f32" },
        );
        Ok(Box::new(TestAudioReader {
            trace: self.trace.clone(),
            object,
        }))
    }
}

struct TestAudioReader {
    trace: TestHostTrace,
    object: usize,
}

impl HostAudioReader for TestAudioReader {
    fn channel_count(&self) -> usize {
        2
    }

    fn sample_count(&self) -> i64 {
        48_000
    }

    fn read_f32(&mut self, position: i64, buffers: &mut [&mut [f32]]) -> Result<(), AraError> {
        for (channel, buffer) in buffers.iter_mut().enumerate() {
            for (offset, sample) in buffer.iter_mut().enumerate() {
                *sample = channel as f32 + (position + offset as i64) as f32 / 48_000.0;
            }
        }
        self.trace.record(
            "read_audio_samples",
            Some(self.object),
            format!("f32:{position}:{}", buffers[0].len()),
        );
        Ok(())
    }

    fn read_f64(&mut self, position: i64, buffers: &mut [&mut [f64]]) -> Result<(), AraError> {
        for (channel, buffer) in buffers.iter_mut().enumerate() {
            for (offset, sample) in buffer.iter_mut().enumerate() {
                *sample = channel as f64 + (position + offset as i64) as f64 / 48_000.0;
            }
        }
        self.trace.record(
            "read_audio_samples",
            Some(self.object),
            format!("f64:{position}:{}", buffers[0].len()),
        );
        Ok(())
    }
}

#[derive(Clone)]
struct TestArchive {
    trace: TestHostTrace,
    read_bytes: Arc<Mutex<Vec<u8>>>,
    written_bytes: Arc<Mutex<HashMap<ArchiveWriterId, Vec<u8>>>>,
    document_archive_id: Arc<Mutex<String>>,
}

impl ArchivingProvider for TestArchive {
    fn len(&self, _: ArchiveReaderId) -> Result<usize, AraError> {
        let len = lock(&self.read_bytes).len();
        self.trace.record("archive_len", None, len.to_string());
        Ok(len)
    }

    fn read_at(
        &self,
        _: ArchiveReaderId,
        position: usize,
        buffer: &mut [u8],
    ) -> Result<(), AraError> {
        let bytes = lock(&self.read_bytes);
        let end = position
            .checked_add(buffer.len())
            .ok_or(AraError::InvalidArgument("fixture archive read overflow"))?;
        let source = bytes
            .get(position..end)
            .ok_or(AraError::Peer("fixture archive read is out of bounds"))?;
        buffer.copy_from_slice(source);
        self.trace
            .record("read_archive", None, format!("{position}:{}", buffer.len()));
        Ok(())
    }

    fn write_at(
        &self,
        writer: ArchiveWriterId,
        position: usize,
        buffer: &[u8],
    ) -> Result<(), AraError> {
        let end = position
            .checked_add(buffer.len())
            .ok_or(AraError::InvalidArgument("fixture archive write overflow"))?;
        let mut archives = lock(&self.written_bytes);
        let archive = archives.entry(writer).or_default();
        archive.resize(archive.len().max(end), 0);
        archive[position..end].copy_from_slice(buffer);
        self.trace.record(
            "write_archive",
            None,
            format!("{position}:{}", buffer.len()),
        );
        Ok(())
    }

    fn archiving_progress(&self, value: f32) -> Result<(), AraError> {
        self.trace
            .record("archiving_progress", None, value.to_string());
        Ok(())
    }

    fn unarchiving_progress(&self, value: f32) -> Result<(), AraError> {
        self.trace
            .record("unarchiving_progress", None, value.to_string());
        Ok(())
    }

    fn document_archive_id(&self, _: ArchiveReaderId) -> Result<Option<String>, AraError> {
        Ok(Some(lock(&self.document_archive_id).clone()))
    }
}

#[derive(Clone)]
struct TestContent;

impl ContentAccessProvider for TestContent {
    fn musical_context_grade(
        &self,
        _: MusicalContextId,
        _: i32,
    ) -> Result<Option<ContentGrade>, AraError> {
        Ok(None)
    }

    fn musical_context_reader(
        &self,
        _: MusicalContextId,
        _: i32,
        _: Option<ContentTimeRange>,
    ) -> Result<Option<HostContentReaderSnapshot>, AraError> {
        Ok(None)
    }

    fn audio_source_grade(
        &self,
        _: AudioSourceId,
        content_type: i32,
    ) -> Result<Option<ContentGrade>, AraError> {
        Ok((content_type == Notes::RAW_TYPE).then_some(ContentGrade::APPROVED))
    }

    fn audio_source_reader(
        &self,
        _: AudioSourceId,
        content_type: i32,
        _: Option<ContentTimeRange>,
    ) -> Result<Option<HostContentReaderSnapshot>, AraError> {
        if content_type != Notes::RAW_TYPE {
            return Ok(None);
        }
        let note = NoteEvent::new(Some(440.0), Some(69), 1.0, 0.0, 0.0, 1.0, 1.0)?;
        Ok(Some(
            HostContentSnapshot::<Notes>::new([note])?.into_reader(ContentGrade::APPROVED),
        ))
    }
}

impl ModelUpdateProvider for TestHostTrace {
    fn audio_source_analysis_progress(
        &self,
        source: AudioSourceId,
        state: i32,
        value: f32,
    ) -> Result<(), AraError> {
        let object = self.source_object(source);
        self.record(
            "analysis_progress",
            Some(object),
            format!("{state}:{value}"),
        );
        Ok(())
    }

    fn audio_source_content_changed(
        &self,
        source: AudioSourceId,
        _: Option<ContentTimeRange>,
        flags: i32,
    ) -> Result<(), AraError> {
        let object = self.source_object(source);
        self.record("audio_source_changed", Some(object), flags.to_string());
        Ok(())
    }

    fn audio_modification_content_changed(
        &self,
        _: ara2_bridge_host::AudioModificationId,
        _: Option<ContentTimeRange>,
        flags: i32,
    ) -> Result<(), AraError> {
        self.record("audio_modification_changed", None, flags.to_string());
        Ok(())
    }

    fn playback_region_content_changed(
        &self,
        _: ara2_bridge_host::PlaybackRegionId,
        _: Option<ContentTimeRange>,
        flags: i32,
    ) -> Result<(), AraError> {
        self.record("playback_region_changed", None, flags.to_string());
        Ok(())
    }

    fn document_data_changed(&self) -> Result<(), AraError> {
        self.record("document_data_changed", None, "");
        Ok(())
    }
}

impl PlaybackProvider for TestHostTrace {
    fn start(&self) -> Result<(), AraError> {
        self.record("start_playback", None, "");
        Ok(())
    }

    fn stop(&self) -> Result<(), AraError> {
        self.record("stop_playback", None, "");
        Ok(())
    }

    fn set_position(&self, position: f64) -> Result<(), AraError> {
        self.record("set_playback_position", None, position.to_string());
        Ok(())
    }

    fn set_cycle_range(&self, start: f64, duration: f64) -> Result<(), AraError> {
        self.record("set_cycle_range", None, format!("{start}:{duration}"));
        Ok(())
    }

    fn enable_cycle(&self, enable: bool) -> Result<(), AraError> {
        self.record("enable_cycle", None, enable.to_string());
        Ok(())
    }
}

unsafe extern "C" fn assertion(_: ARAAssertCategory, _: *const c_void, _: *const c_char) {}

/// Deterministic host services and trace used by public interoperability scenarios.
pub struct TestHost {
    generation: ApiGeneration,
    services: HostServices,
    trace: TestHostTrace,
    archive_read_bytes: Arc<Mutex<Vec<u8>>>,
    archive_written_bytes: Arc<Mutex<HashMap<ArchiveWriterId, Vec<u8>>>>,
    archive_id: Arc<Mutex<String>>,
}

impl TestHost {
    /// Builds capability-complete deterministic host services for one API generation.
    pub fn new(generation: ApiGeneration) -> Result<Self, AraError> {
        let trace = TestHostTrace::new(generation);
        let archive_read_bytes = Arc::new(Mutex::new(Vec::new()));
        let archive_written_bytes = Arc::new(Mutex::new(HashMap::new()));
        let archive_id = Arc::new(Mutex::new("org.ara2-bridge.test.archive".to_owned()));
        let archive = TestArchive {
            trace: trace.clone(),
            read_bytes: archive_read_bytes.clone(),
            written_bytes: archive_written_bytes.clone(),
            document_archive_id: archive_id.clone(),
        };
        let services = HostServicesBuilder::new()
            .audio(TestAudio {
                trace: trace.clone(),
            })
            .archiving(archive)
            .content(TestContent)
            .model_updates(trace.clone())
            .playback(trace.clone())
            .build(generation)?;
        Ok(Self {
            generation,
            services,
            trace,
            archive_read_bytes,
            archive_written_bytes,
            archive_id,
        })
    }

    /// Returns the host API generation.
    pub const fn generation(&self) -> ApiGeneration {
        self.generation
    }

    /// Returns stable services suitable for document-controller creation.
    pub const fn services(&self) -> &HostServices {
        &self.services
    }

    /// Returns a cloneable ordered callback trace.
    pub fn trace(&self) -> TestHostTrace {
        self.trace.clone()
    }

    /// Replaces the bytes returned by the deterministic archive reader.
    pub fn seed_archive(&self, bytes: impl Into<Vec<u8>>) {
        *lock(&self.archive_read_bytes) = bytes.into();
    }

    /// Selects the document archive ID reported for subsequent fixture readers.
    pub fn set_archive_id(&self, id: impl Into<String>) {
        *lock(&self.archive_id) = id.into();
    }

    /// Returns the sole archive written by the current scenario, if exactly one exists.
    pub fn written_archive(&self) -> Option<Vec<u8>> {
        let archives = lock(&self.archive_written_bytes);
        (archives.len() == 1)
            .then(|| archives.values().next().cloned())
            .flatten()
    }

    /// Loads and initializes a safe Rust fixture factory through only its public ARA pointer.
    pub fn load_factory<'factory>(
        &self,
        factory: &'factory Factory,
    ) -> Result<LoadedFactory<'factory>, AraError> {
        // SAFETY: the returned guard borrows `factory`, whose stable raw record is retained for the
        // guard lifetime; the assertion callback is a process-lifetime function.
        unsafe { LoadedFactory::load(factory.as_raw(), self.generation, Some(assertion)) }
    }

    /// Exercises one real planar f32 host audio read for a checked document source reference.
    pub fn read_source_samples(
        &self,
        source: ModelRef<AudioSourceKind>,
        position: i64,
        sample_count: usize,
    ) -> Result<Vec<Vec<f32>>, AraError> {
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

        let instance = self.services.instance();
        let interface = instance.audioAccessControllerInterface;
        if interface.is_null() {
            return Err(AraError::Abi("TestHost audio interface is null"));
        }
        let base = interface.cast::<u8>();
        // SAFETY: TestHost owns a complete audio interface for its own lifetime.
        let create = unsafe {
            read_field::<Option<Create>>(
                base,
                offset_of!(
                    ARAAudioAccessControllerInterface,
                    createAudioReaderForSource
                ),
            )
        }
        .ok_or(AraError::Abi("TestHost audio create callback is null"))?;
        // SAFETY: same complete owned interface.
        let read = unsafe {
            read_field::<Option<Read>>(
                base,
                offset_of!(ARAAudioAccessControllerInterface, readAudioSamples),
            )
        }
        .ok_or(AraError::Abi("TestHost audio read callback is null"))?;
        // SAFETY: same complete owned interface.
        let destroy = unsafe {
            read_field::<Option<Destroy>>(
                base,
                offset_of!(ARAAudioAccessControllerInterface, destroyAudioReader),
            )
        }
        .ok_or(AraError::Abi("TestHost audio destroy callback is null"))?;
        // SAFETY: the checked model reference and host services remain live for this synchronous
        // create/read/destroy sequence.
        let reader = unsafe {
            create(
                instance.audioAccessControllerHostRef,
                source.as_raw().cast(),
                kARAFalse,
            )
        };
        if reader.is_null() {
            return Err(AraError::Peer("TestHost failed to create an audio reader"));
        }
        let mut channels = vec![vec![0.0_f32; sample_count]; 2];
        let pointers = channels
            .iter_mut()
            .map(|channel| channel.as_mut_ptr().cast())
            .collect::<Vec<*mut c_void>>();
        let sample_count = i64::try_from(sample_count)
            .map_err(|_| AraError::InvalidArgument("sample count exceeds ARA range"))?;
        // SAFETY: both planar buffers contain `sample_count` writable f32 values.
        let accepted = unsafe {
            read(
                instance.audioAccessControllerHostRef,
                reader,
                position,
                sample_count,
                pointers.as_ptr(),
            )
        };
        // SAFETY: the reader was returned by this exact live service instance.
        unsafe { destroy(instance.audioAccessControllerHostRef, reader) };
        if accepted == kARAFalse {
            Err(AraError::Peer("TestHost audio read failed"))
        } else {
            Ok(channels)
        }
    }
}
