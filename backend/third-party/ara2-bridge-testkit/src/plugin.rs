//! Capability-rich safe Rust plug-in fixture used by cross-crate conformance tests.

use ara2_bridge_core::{
    ApiGeneration, AraBool, AraError, AudioModificationProperties, AudioSourceProperties,
    BarSignatureEvent, BarSignatures, ChordEvent, ChordIntervalUsage, ContentGrade,
    ContentTimeRange, ContentUpdateScopes, DocumentProperties, FilterSelection, HeadTailEntry,
    KeySignatureEvent, KeySignatureIntervalUsage, KeySignatures, LicenseCapabilities, NoteEvent,
    Notes, PlaybackRegionProperties, PlaybackTransformationFlags, ProcessingAlgorithmCatalog,
    ProcessingAlgorithmProperties, RawHandle, RegionSequenceProperties, RestoreFilter, SheetChords,
    StaticTuning, StoreFilter, Tempo, TempoEvent, TuningEvent,
};
use ara2_bridge_plugin::{
    AnalysisEmitter, AnalysisProvider, AudioFileChunk, AudioModifications, AudioSources,
    ContentObject, ContentProvider, ContentReaderSnapshot, ContentSnapshot, CreateContext,
    DocumentLifecycle, ExtensionBinding, ExtensionControllerLease, ExtensionRoles, Factory,
    FactoryBuilder, FactoryCapabilities, HostContentScope, MusicalContexts, PartialPersistence,
    Persistence, PlaybackRegions, Plugin, PluginBuilder, RealtimeHeadTailAdapter, RegionSequences,
    UpdateEmitter, UpdateOrigin,
};
use ara2_bridge_sys::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Shared ordered trace emitted by the capability-rich test plug-in.
#[derive(Clone, Default)]
pub struct TestPluginTrace {
    records: Arc<Mutex<Vec<&'static str>>>,
    reject_audio_source_after_host_callback: Arc<AtomicBool>,
    updates: Arc<Mutex<Option<UpdateEmitter>>>,
}

impl TestPluginTrace {
    /// Creates an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a snapshot of all recorded semantic operations.
    pub fn records(&self) -> Vec<&'static str> {
        lock(&self.records).clone()
    }

    /// Returns how often a named semantic operation was recorded.
    pub fn count(&self, name: &str) -> usize {
        lock(&self.records)
            .iter()
            .filter(|record| **record == name)
            .count()
    }

    /// Makes the next audio-source create issue a scoped host callback and then fail.
    pub fn reject_next_audio_source_after_host_callback(&self) {
        self.reject_audio_source_after_host_callback
            .store(true, Ordering::Release);
    }

    /// Queues a plug-in-originated persistent document-data dirtiness notification.
    pub fn mark_document_dirty(&self) -> Result<(), AraError> {
        let updates = lock(&self.updates)
            .clone()
            .ok_or(AraError::InvalidState("test plug-in has no update emitter"))?;
        updates.mark_document(UpdateOrigin::Application);
        Ok(())
    }

    fn install_updates(&self, updates: UpdateEmitter) {
        *lock(&self.updates) = Some(updates);
    }

    fn record(&self, name: &'static str) {
        lock(&self.records).push(name);
    }
}

/// Required model implementation used by [`build_test_plugin`].
pub struct TestPluginModel {
    trace: TestPluginTrace,
    updates: Option<UpdateEmitter>,
    head_tail: Arc<RealtimeHeadTailAdapter>,
    head_tail_entries: Vec<HeadTailEntry>,
}

impl DocumentLifecycle for TestPluginModel {
    type Document = ();

    fn create_document(
        &mut self,
        _: &CreateContext,
        _: DocumentProperties,
    ) -> Result<Self::Document, AraError> {
        self.trace.record("create_document");
        Ok(())
    }

    fn update_document(
        &mut self,
        _: &mut Self::Document,
        _: DocumentProperties,
    ) -> Result<(), AraError> {
        self.trace.record("update_document");
        Ok(())
    }

    fn begin_editing(&mut self, _: &mut Self::Document) -> Result<(), AraError> {
        self.trace.record("begin_editing");
        Ok(())
    }

    fn end_editing(
        &mut self,
        _: &mut Self::Document,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record("end_editing");
        Ok(())
    }

    fn destroy_document(&mut self, _: Self::Document) {
        self.trace.record("destroy_document");
    }
}

impl MusicalContexts for TestPluginModel {
    type MusicalContext = u64;

    fn create_musical_context(
        &mut self,
        context: &CreateContext,
        _: ara2_bridge_core::MusicalContextProperties,
        _: &HostContentScope<'_, '_>,
    ) -> Result<Self::MusicalContext, AraError> {
        self.trace.record("create_musical_context");
        Ok(context.realtime_key().unwrap_or_default())
    }

    fn destroy_musical_context(&mut self, _: Self::MusicalContext) {
        self.trace.record("destroy_musical_context");
    }

    fn update_musical_context(
        &mut self,
        _: &mut Self::MusicalContext,
        _: ara2_bridge_core::MusicalContextProperties,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record("update_musical_context");
        Ok(())
    }

    fn update_musical_context_content(
        &mut self,
        _: &mut Self::MusicalContext,
        _: Option<ContentTimeRange>,
        _: ContentUpdateScopes,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record("update_musical_context_content");
        Ok(())
    }
}

impl RegionSequences for TestPluginModel {
    type RegionSequence = u64;

    fn create_region_sequence(
        &mut self,
        context: &CreateContext,
        _: RegionSequenceProperties,
    ) -> Result<Self::RegionSequence, AraError> {
        self.trace.record("create_region_sequence");
        Ok(context.realtime_key().unwrap_or_default())
    }

    fn destroy_region_sequence(&mut self, _: Self::RegionSequence) {
        self.trace.record("destroy_region_sequence");
    }

    fn update_region_sequence(
        &mut self,
        _: &mut Self::RegionSequence,
        _: RegionSequenceProperties,
    ) -> Result<(), AraError> {
        self.trace.record("update_region_sequence");
        Ok(())
    }
}

impl AudioSources for TestPluginModel {
    type AudioSource = u64;

    fn create_audio_source(
        &mut self,
        context: &CreateContext,
        _: AudioSourceProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<Self::AudioSource, AraError> {
        self.trace.record("create_audio_source");
        if self
            .trace
            .reject_audio_source_after_host_callback
            .swap(false, Ordering::AcqRel)
        {
            let source = host
                .current_audio_source()
                .ok_or(AraError::InvalidState("missing scoped audio source"))?;
            let _ = host.audio_source_grade::<Notes>(source);
            return Err(AraError::Peer("test audio-source create rejection"));
        }
        if let (Some(updates), Some(source)) = (&self.updates, context.object_handle()) {
            updates.mark_source(
                source,
                None,
                ContentUpdateScopes::empty(),
                UpdateOrigin::Application,
            )?;
        }
        Ok(context.realtime_key().unwrap_or_default())
    }

    fn update_audio_source(
        &mut self,
        _: &mut Self::AudioSource,
        _: AudioSourceProperties,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record("update_audio_source");
        Ok(())
    }

    fn update_audio_source_content(
        &mut self,
        _: &mut Self::AudioSource,
        _: Option<ContentTimeRange>,
        _: ContentUpdateScopes,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record("update_audio_source_content");
        Ok(())
    }

    fn enable_audio_source_samples_access(
        &mut self,
        _: &mut Self::AudioSource,
        enable: bool,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record(if enable {
            "enable_audio_source"
        } else {
            "disable_audio_source"
        });
        Ok(())
    }

    fn deactivate_audio_source(
        &mut self,
        _: &mut Self::AudioSource,
        deactivate: bool,
        _: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.trace.record(if deactivate {
            "deactivate_audio_source"
        } else {
            "reactivate_audio_source"
        });
        Ok(())
    }

    fn destroy_audio_source(&mut self, _: Self::AudioSource, _: &HostContentScope<'_, '_>) {
        self.trace.record("destroy_audio_source");
    }
}

impl AudioModifications for TestPluginModel {
    type AudioModification = u64;
    // 本地补丁（HiFiShifter）：`AudioModifications` 多声明了父 source 关联类型，
    // 与 `AudioSources::AudioSource` 钉为同一个 `u64`（见本目录 PATCHED.md）。
    type AudioSource = u64;

    fn create_audio_modification(
        &mut self,
        context: &CreateContext,
        _: &Self::AudioSource,
        _: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        self.trace.record("create_audio_modification");
        Ok(context.realtime_key().unwrap_or_default())
    }

    fn clone_audio_modification(
        &mut self,
        context: &CreateContext,
        _: &Self::AudioModification,
        _: &Self::AudioSource,
        _: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        self.trace.record("clone_audio_modification");
        Ok(context.realtime_key().unwrap_or_default())
    }

    fn update_audio_modification(
        &mut self,
        _: &mut Self::AudioModification,
        _: AudioModificationProperties,
    ) -> Result<(), AraError> {
        self.trace.record("update_audio_modification");
        Ok(())
    }

    fn deactivate_audio_modification(
        &mut self,
        _: &mut Self::AudioModification,
        deactivate: bool,
    ) -> Result<(), AraError> {
        self.trace.record(if deactivate {
            "deactivate_audio_modification"
        } else {
            "reactivate_audio_modification"
        });
        Ok(())
    }

    fn destroy_audio_modification(&mut self, _: Self::AudioModification) {
        self.trace.record("destroy_audio_modification");
    }
}

impl PlaybackRegions for TestPluginModel {
    type PlaybackRegion = u64;
    // 本地补丁（HiFiShifter）：`PlaybackRegions` 多声明了父 modification 与所属
    // sequence 两个关联类型，与"正主" trait 的同名类型钉为同一个 `u64`。
    type AudioModification = u64;
    type RegionSequence = u64;

    fn create_playback_region(
        &mut self,
        context: &CreateContext,
        _: &Self::AudioModification,
        _: &Self::RegionSequence,
        _: PlaybackRegionProperties,
    ) -> Result<Self::PlaybackRegion, AraError> {
        self.trace.record("create_playback_region");
        let key = context.realtime_key().unwrap_or_default();
        self.head_tail_entries
            .push(HeadTailEntry::new(key, 0.125, 0.25)?);
        self.head_tail.install(self.head_tail_entries.clone())?;
        Ok(key)
    }

    fn update_playback_region(
        &mut self,
        _: &mut Self::PlaybackRegion,
        _: PlaybackRegionProperties,
    ) -> Result<(), AraError> {
        self.trace.record("update_playback_region");
        Ok(())
    }

    fn destroy_playback_region(&mut self, _: Self::PlaybackRegion) {
        self.trace.record("destroy_playback_region");
    }
}

struct TestContent {
    trace: TestPluginTrace,
}

impl ContentProvider for TestContent {
    fn is_content_available(&self, _: ContentObject, content_type: i32) -> bool {
        all_content_types().contains(&content_type)
    }

    fn content_grade(&self, _: ContentObject, _: i32) -> ContentGrade {
        ContentGrade::APPROVED
    }

    fn create_content_reader(
        &mut self,
        _: ContentObject,
        content_type: i32,
        _: Option<ContentTimeRange>,
    ) -> Result<Option<ContentReaderSnapshot>, AraError> {
        self.trace.record("create_content_reader");
        let grade = ContentGrade::APPROVED;
        let reader =
            match content_type {
                value if value == kARAContentTypeTempoEntries as i32 => {
                    ContentSnapshot::<Tempo>::new([
                        TempoEvent::new(0.0, 0.0)?,
                        TempoEvent::new(1.0, 2.0)?,
                    ])?
                    .into_reader(grade)
                }
                value if value == kARAContentTypeBarSignatures as i32 => {
                    ContentSnapshot::<BarSignatures>::new([BarSignatureEvent::new(4, 4, 0.0)?])?
                        .into_reader(grade)
                }
                value if value == kARAContentTypeNotes as i32 => {
                    ContentSnapshot::<Notes>::new([NoteEvent::new(
                        Some(440.0),
                        Some(69),
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    )?])?
                    .into_reader(grade)
                }
                value if value == kARAContentTypeStaticTuning as i32 => ContentSnapshot::<
                    StaticTuning,
                >::new([
                    TuningEvent::new(440.0, 0, [0.0; 12], Some("Equal temperament".to_owned()))?,
                ])?
                .into_reader(grade),
                value if value == kARAContentTypeKeySignatures as i32 => {
                    ContentSnapshot::<KeySignatures>::new([KeySignatureEvent::new(
                        0,
                        [KeySignatureIntervalUsage::USED; 12],
                        Some("C".to_owned()),
                        0.0,
                    )?])?
                    .into_reader(grade)
                }
                value if value == kARAContentTypeSheetChords as i32 => {
                    ContentSnapshot::<SheetChords>::new([ChordEvent::new(
                        0,
                        0,
                        [ChordIntervalUsage::USED; 12],
                        Some("C".to_owned()),
                        0.0,
                    )?])?
                    .into_reader(grade)
                }
                _ => return Ok(None),
            };
        Ok(Some(reader))
    }
}

struct TestAnalysis {
    trace: TestPluginTrace,
    emitter: AnalysisEmitter,
}

impl AnalysisProvider for TestAnalysis {
    fn request_analysis(&mut self, source: RawHandle, _: &[i32]) -> Result<(), AraError> {
        self.trace.record("request_analysis");
        self.emitter.update(source, 0.5)?;
        self.emitter.complete(source);
        Ok(())
    }

    fn cancel_analysis(&mut self, _: RawHandle) {
        self.trace.record("cancel_analysis");
    }
}

struct TestPersistence {
    trace: TestPluginTrace,
}

impl Persistence for TestPersistence {
    fn restore_document(&mut self, _: &[u8]) -> Result<(), AraError> {
        self.trace.record("restore_document");
        Ok(())
    }

    fn store_document(&mut self) -> Result<Vec<u8>, AraError> {
        self.trace.record("store_document");
        Ok(vec![0x41, 0x52, 0x41])
    }
}

impl PartialPersistence for TestPersistence {
    fn restore_objects(
        &mut self,
        filter: &FilterSelection<RestoreFilter>,
        _: &[u8],
    ) -> Result<(), AraError> {
        self.trace.record("restore_objects");
        if let FilterSelection::Selected(filter) = filter {
            if !filter.audio_sources().is_empty() {
                self.trace.record("restore_audio_sources");
            }
            if !filter.audio_modifications().is_empty() {
                self.trace.record("restore_audio_modifications");
            }
            if filter.includes_document_data() {
                self.trace.record("restore_document_data");
            }
        }
        Ok(())
    }

    fn store_objects(&mut self, _: &FilterSelection<StoreFilter>) -> Result<Vec<u8>, AraError> {
        self.trace.record("store_objects");
        Ok(vec![0x41, 0x52, 0x41])
    }
}

/// Builds a fresh plug-in with every document-controller capability enabled.
pub fn build_test_plugin(trace: TestPluginTrace) -> Result<Plugin<TestPluginModel>, AraError> {
    let head_tail = Arc::new(RealtimeHeadTailAdapter::new(64)?);
    let mut builder = PluginBuilder::new(TestPluginModel {
        trace: trace.clone(),
        updates: None,
        head_tail: head_tail.clone(),
        head_tail_entries: Vec::new(),
    });
    let updates = builder.update_emitter();
    trace.install_updates(updates.clone());
    builder.model_mut().updates = Some(updates);
    let analysis_emitter = builder.analysis_emitter();
    let algorithms = ProcessingAlgorithmCatalog::new(vec![
        ProcessingAlgorithmProperties::new("test.default", "Test Default")?,
        ProcessingAlgorithmProperties::new("test.polyphonic", "Test Polyphonic")?,
    ])?;
    let transformations = all_transformations();
    let licensing = LicenseCapabilities::new(all_content_types(), transformations)?;
    let chunk_trace = trace.clone();
    let signal_trace = trace.clone();
    builder
        .content(TestContent {
            trace: trace.clone(),
        })
        .analysis(TestAnalysis {
            trace: trace.clone(),
            emitter: analysis_emitter,
        })
        .partial_persistence(TestPersistence {
            trace: trace.clone(),
        })
        .processing_algorithms(algorithms)
        .licensing_for(licensing, |_| true)
        .audio_file_chunks(move |_| {
            chunk_trace.record("store_audio_file_chunk");
            Ok(AudioFileChunk {
                bytes: vec![0x41, 0x52, 0x41],
                document_archive_id: "org.ara2-bridge.test.archive".to_owned(),
                open_automatically: true,
            })
        })
        .signal_preservation(move |_| {
            signal_trace.record("query_signal_preservation");
            true
        })
        .realtime_head_tail(head_tail)
        .build()
}

/// Builds a fresh plug-in with only the generation-required controller surface enabled.
pub fn build_minimal_test_plugin(
    trace: TestPluginTrace,
) -> Result<Plugin<TestPluginModel>, AraError> {
    let head_tail = Arc::new(RealtimeHeadTailAdapter::new(8)?);
    PluginBuilder::new(TestPluginModel {
        trace,
        updates: None,
        head_tail,
        head_tail_entries: Vec::new(),
    })
    .build()
}

/// Builds the capability-rich fixture as an initialized-factory-ready definition.
pub fn build_test_factory(trace: TestPluginTrace) -> Result<Factory, AraError> {
    let minimum_generation = if cfg!(target_arch = "aarch64") {
        ApiGeneration::V2Final
    } else {
        ApiGeneration::V1Final
    };
    FactoryBuilder::new("org.ara2-bridge.test", "org.ara2-bridge.test.archive")
        .display(
            "ARA2 Bridge Test Plug-In",
            "ara2-bridge",
            "https://github.com/entrepeneur4lyf/ara2-bridge",
            env!("CARGO_PKG_VERSION"),
        )
        .generations(minimum_generation, ApiGeneration::V23Final)
        .capabilities(
            FactoryCapabilities::default()
                .with_analyzable_content_types(all_content_types())
                .with_playback_transformations(all_transformations())
                .with_audio_file_chunk_storage(true),
        )
        .document_controller(move || build_test_plugin(trace.clone()))
        .build()
}

/// Builds a factory whose controllers omit every optional capability tail.
pub fn build_minimal_test_factory(trace: TestPluginTrace) -> Result<Factory, AraError> {
    let minimum_generation = if cfg!(target_arch = "aarch64") {
        ApiGeneration::V2Final
    } else {
        ApiGeneration::V1Draft
    };
    FactoryBuilder::new(
        "org.ara2-bridge.test.minimal",
        "org.ara2-bridge.test.minimal.archive",
    )
    .display(
        "ARA2 Bridge Minimal Test Plug-In",
        "ara2-bridge",
        "https://github.com/entrepeneur4lyf/ara2-bridge",
        env!("CARGO_PKG_VERSION"),
    )
    .generations(minimum_generation, ApiGeneration::V23Final)
    .document_controller(move || build_minimal_test_plugin(trace.clone()))
    .build()
}

/// Returns every released content type supported by the fixture.
pub fn all_content_types() -> [i32; 6] {
    [
        kARAContentTypeTempoEntries as i32,
        kARAContentTypeBarSignatures as i32,
        kARAContentTypeNotes as i32,
        kARAContentTypeStaticTuning as i32,
        kARAContentTypeKeySignatures as i32,
        kARAContentTypeSheetChords as i32,
    ]
}

/// Returns all playback transformations supported by the fixture.
pub fn all_transformations() -> PlaybackTransformationFlags {
    PlaybackTransformationFlags::TIMESTRETCH
        | PlaybackTransformationFlags::REFLECT_TEMPO
        | PlaybackTransformationFlags::CONTENT_FADES
}

/// Returns all ARA 2 extension roles supported by the fixture companion binding.
pub fn all_extension_roles() -> ExtensionRoles {
    ExtensionRoles::PLAYBACK_RENDERER
        | ExtensionRoles::EDITOR_RENDERER
        | ExtensionRoles::EDITOR_VIEW
}

/// Creates the fixture's stable companion extension backing from raw ARA role flags.
pub fn build_test_extension(
    generation: ApiGeneration,
    known_roles: i32,
    assigned_roles: i32,
) -> Result<(ExtensionBinding, ExtensionControllerLease), AraError> {
    let known = ExtensionRoles::from_bits(known_roles)
        .ok_or(AraError::InvalidArgument("unknown fixture role flags"))?;
    let assigned = ExtensionRoles::from_bits(assigned_roles)
        .ok_or(AraError::InvalidArgument("unknown fixture role flags"))?;
    ExtensionBinding::new(generation, known, assigned, all_extension_roles())
}

/// Returns the canonical stereo source properties used by raw conformance scenarios.
pub fn test_audio_source_properties() -> Result<AudioSourceProperties, AraError> {
    AudioSourceProperties::new(
        Some("Test Source"),
        "test-source",
        48_000,
        48_000.0,
        2,
        AraBool::new(false),
    )
}

/// Returns the empty update scopes used by fixture host-originated changes.
pub fn test_update_scopes() -> ContentUpdateScopes {
    ContentUpdateScopes::empty()
}
