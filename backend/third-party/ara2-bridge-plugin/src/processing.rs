//! Plug-in composition and optional processing capabilities.

use crate::{
    document_controller_interface, AnalysisEmitter, AnalysisProvider, ContentObject,
    ContentProvider, ContentReaderSnapshot, ControllerCapabilities, ControllerInterface,
    PartialPersistence, Persistence, PluginModel, PluginRuntime, RealtimeHeadTailAdapter,
    UpdateEmitter,
};
use ara2_bridge_core::{
    ApiGeneration, AraError, ContentGrade, ContentTimeRange, DocumentProperties, FilterSelection,
    LicenseCapabilities, LicenseRequest, PlaybackTransformationFlags, ProcessingAlgorithmCatalog,
    RawHandle, RestoreFilter, StoreFilter,
};
use ara2_bridge_sys::{
    kARAContentTypeBarSignatures, kARAContentTypeKeySignatures, kARAContentTypeNotes,
    kARAContentTypeSheetChords, kARAContentTypeStaticTuning, kARAContentTypeTempoEntries,
    ARAProcessingAlgorithmProperties,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Bytes and metadata produced for ARA audio-file chunk persistence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioFileChunk {
    /// Serialized plug-in archive bytes.
    pub bytes: Vec<u8>,
    /// Persistent document archive identifier associated with the bytes.
    pub document_archive_id: String,
    /// Whether a compatible plug-in should open the archive automatically.
    pub open_automatically: bool,
}

type LicenseHandler = Box<dyn FnMut(&LicenseRequest) -> bool>;
type ChunkWriter = Box<dyn FnMut(RawHandle) -> Result<AudioFileChunk, AraError>>;
type SignalQuery = Box<dyn Fn(RawHandle) -> bool>;

enum PersistenceCapability {
    Complete(Box<dyn Persistence>),
    Partial(Box<dyn PartialPersistence>),
}

/// Optional semantic capability state retained beside one plug-in model.
#[derive(Default)]
pub struct SemanticCapabilities {
    algorithms: Option<ProcessingAlgorithmCatalog>,
    algorithm_raw: Box<[ARAProcessingAlgorithmProperties]>,
    active_algorithms: HashMap<RawHandle, i32>,
    license_capabilities: Option<LicenseCapabilities>,
    licensing: Option<LicenseHandler>,
    chunk_writer: Option<ChunkWriter>,
    signal_query: Option<SignalQuery>,
    content: Option<Box<dyn ContentProvider>>,
    analysis: Option<Box<dyn AnalysisProvider>>,
    persistence: Option<PersistenceCapability>,
    head_tail: Option<Arc<RealtimeHeadTailAdapter>>,
}

impl SemanticCapabilities {
    /// Returns the stable processing-algorithm catalog, when enabled.
    pub const fn algorithms(&self) -> Option<&ProcessingAlgorithmCatalog> {
        self.algorithms.as_ref()
    }

    pub(crate) fn algorithm_properties(
        &self,
        index: i32,
    ) -> Result<*const ARAProcessingAlgorithmProperties, AraError> {
        self.algorithms
            .as_ref()
            .ok_or(AraError::Unsupported(
                "processing algorithms are unavailable",
            ))?
            .get(index)?;
        let index = usize::try_from(index)
            .map_err(|_| AraError::InvalidArgument("negative processing algorithm index"))?;
        Ok(std::ptr::from_ref(&self.algorithm_raw[index]))
    }

    /// Selects a validated processing algorithm for one live audio source.
    pub fn request_algorithm(&mut self, source: RawHandle, index: i32) -> Result<(), AraError> {
        self.algorithms
            .as_ref()
            .ok_or(AraError::Unsupported(
                "processing algorithms are unavailable",
            ))?
            .get(index)?;
        self.active_algorithms.insert(source, index);
        Ok(())
    }

    /// Returns the active processing-algorithm index for an audio source.
    pub fn active_algorithm(&self, source: RawHandle) -> Result<i32, AraError> {
        self.active_algorithms
            .get(&source)
            .copied()
            .ok_or(AraError::InvalidState(
                "audio source has no active processing algorithm",
            ))
    }

    /// Evaluates a validated licensing request or applies ARA's permissive default.
    pub fn is_licensed(&mut self, request: &LicenseRequest) -> bool {
        self.licensing
            .as_mut()
            .is_none_or(|handler| handler(request))
    }

    pub(crate) const fn license_capabilities(&self) -> Option<&LicenseCapabilities> {
        self.license_capabilities.as_ref()
    }

    /// Produces an audio-file chunk when that capability is registered.
    pub fn store_audio_file_chunk(
        &mut self,
        source: RawHandle,
    ) -> Result<AudioFileChunk, AraError> {
        let chunk = self.chunk_writer.as_mut().ok_or(AraError::Unsupported(
            "audio-file chunk persistence is unavailable",
        ))?(source)?;
        if chunk.document_archive_id.is_empty() || !chunk.document_archive_id.is_ascii() {
            return Err(AraError::InvalidArgument(
                "audio-file chunk archive ID must be nonempty ASCII",
            ));
        }
        if chunk.document_archive_id.as_bytes().contains(&0) {
            return Err(AraError::InvalidArgument(
                "audio-file chunk archive ID contains NUL",
            ));
        }
        Ok(chunk)
    }

    /// Evaluates the optional signal-preservation query, defaulting to false.
    pub fn preserves_signal(&self, modification: RawHandle) -> bool {
        self.signal_query
            .as_ref()
            .is_some_and(|query| query(modification))
    }

    pub(crate) fn is_content_available(&self, object: ContentObject, content_type: i32) -> bool {
        self.content
            .as_ref()
            .is_some_and(|provider| provider.is_content_available(object, content_type))
    }

    pub(crate) fn content_grade(&self, object: ContentObject, content_type: i32) -> ContentGrade {
        self.content
            .as_ref()
            .map_or(ContentGrade::INITIAL, |provider| {
                provider.content_grade(object, content_type)
            })
    }

    pub(crate) fn create_content_reader(
        &mut self,
        object: ContentObject,
        content_type: i32,
        range: Option<ContentTimeRange>,
    ) -> Result<Option<ContentReaderSnapshot>, AraError> {
        self.content
            .as_mut()
            .ok_or(AraError::Unsupported("content provider is unavailable"))?
            .create_content_reader(object, content_type, range)
    }

    pub(crate) fn is_analysis_incomplete(&self, source: RawHandle, content_type: i32) -> bool {
        self.analysis
            .as_ref()
            .is_some_and(|provider| provider.is_analysis_incomplete(source, content_type))
    }

    pub(crate) fn request_analysis(
        &mut self,
        source: RawHandle,
        content_types: &[i32],
    ) -> Result<(), AraError> {
        self.analysis
            .as_mut()
            .ok_or(AraError::Unsupported("analysis provider is unavailable"))?
            .request_analysis(source, content_types)
    }

    pub(crate) fn cancel_analysis(&mut self, source: RawHandle) {
        if let Some(provider) = self.analysis.as_mut() {
            provider.cancel_analysis(source);
        }
    }

    pub(crate) const fn has_analysis(&self) -> bool {
        self.analysis.is_some()
    }

    pub(crate) const fn has_audio_file_chunks(&self) -> bool {
        self.chunk_writer.is_some()
    }

    pub(crate) fn restore_document(&mut self, bytes: &[u8]) -> Result<(), AraError> {
        match self.persistence.as_mut() {
            Some(PersistenceCapability::Complete(provider)) => provider.restore_document(bytes),
            Some(PersistenceCapability::Partial(provider)) => provider.restore_document(bytes),
            None => Err(AraError::Unsupported("persistence provider is unavailable")),
        }
    }

    pub(crate) fn store_document(&mut self) -> Result<Vec<u8>, AraError> {
        match self.persistence.as_mut() {
            Some(PersistenceCapability::Complete(provider)) => provider.store_document(),
            Some(PersistenceCapability::Partial(provider)) => provider.store_document(),
            None => Err(AraError::Unsupported("persistence provider is unavailable")),
        }
    }

    pub(crate) fn restore_objects(
        &mut self,
        filter: &FilterSelection<RestoreFilter>,
        bytes: &[u8],
    ) -> Result<(), AraError> {
        match self.persistence.as_mut() {
            Some(PersistenceCapability::Complete(provider))
                if matches!(filter, FilterSelection::All) =>
            {
                provider.restore_document(bytes)
            }
            Some(PersistenceCapability::Partial(provider)) => {
                provider.restore_objects(filter, bytes)
            }
            _ => Err(AraError::Unsupported(
                "partial persistence provider is unavailable",
            )),
        }
    }

    pub(crate) fn store_objects(
        &mut self,
        filter: &FilterSelection<StoreFilter>,
    ) -> Result<Vec<u8>, AraError> {
        match self.persistence.as_mut() {
            Some(PersistenceCapability::Complete(provider))
                if matches!(filter, FilterSelection::All) =>
            {
                provider.store_document()
            }
            Some(PersistenceCapability::Partial(provider)) => provider.store_objects(filter),
            _ => Err(AraError::Unsupported(
                "partial persistence provider is unavailable",
            )),
        }
    }

    pub(crate) fn head_tail(&self) -> Option<&RealtimeHeadTailAdapter> {
        self.head_tail.as_deref()
    }
}

/// Builder composing a required model with optional semantic capability groups.
pub struct PluginBuilder<P> {
    model: P,
    capabilities: SemanticCapabilities,
    controller_capabilities: ControllerCapabilities,
    updates: UpdateEmitter,
    analysis_events: AnalysisEmitter,
}

impl<P> PluginBuilder<P> {
    /// Starts a plug-in definition with no optional tail capabilities.
    pub fn new(model: P) -> Self {
        Self {
            model,
            capabilities: SemanticCapabilities::default(),
            controller_capabilities: ControllerCapabilities::default(),
            updates: UpdateEmitter::new(),
            analysis_events: AnalysisEmitter::new(),
        }
    }

    /// Borrows the model while composing optional application services.
    pub fn model_mut(&mut self) -> &mut P {
        &mut self.model
    }

    /// Clones the handle used by application code to queue ARA 2.3 model notifications.
    pub fn update_emitter(&self) -> UpdateEmitter {
        self.updates.clone()
    }

    /// Clones the handle used by analysis workers to queue progress for model-thread delivery.
    pub fn analysis_emitter(&self) -> AnalysisEmitter {
        self.analysis_events.clone()
    }

    /// Registers a stable processing-algorithm catalog.
    pub fn processing_algorithms(mut self, catalog: ProcessingAlgorithmCatalog) -> Self {
        self.capabilities.algorithm_raw = (0..catalog.len_i32().unwrap_or(0))
            .map(|index| {
                catalog
                    .raw(index)
                    .expect("catalog length defines valid indices")
                    .as_ara()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.capabilities.algorithms = Some(catalog);
        self.controller_capabilities = self
            .controller_capabilities
            .with_processing_algorithms(true);
        self
    }

    /// Registers the licensing decision hook; modal activation is already validated in the request.
    pub fn licensing(mut self, handler: impl FnMut(&LicenseRequest) -> bool + 'static) -> Self {
        let all_content = [
            kARAContentTypeTempoEntries as i32,
            kARAContentTypeBarSignatures as i32,
            kARAContentTypeNotes as i32,
            kARAContentTypeStaticTuning as i32,
            kARAContentTypeKeySignatures as i32,
            kARAContentTypeSheetChords as i32,
        ];
        self.capabilities.license_capabilities = LicenseCapabilities::new(
            all_content,
            PlaybackTransformationFlags::from_bits_retain(u32::MAX),
        )
        .ok();
        self.capabilities.licensing = Some(Box::new(handler));
        self.controller_capabilities = self.controller_capabilities.with_licensing(true);
        self
    }

    /// Registers licensing against an explicit supported-capability set.
    pub fn licensing_for(
        mut self,
        capabilities: LicenseCapabilities,
        handler: impl FnMut(&LicenseRequest) -> bool + 'static,
    ) -> Self {
        self.capabilities.license_capabilities = Some(capabilities);
        self.capabilities.licensing = Some(Box::new(handler));
        self.controller_capabilities = self.controller_capabilities.with_licensing(true);
        self
    }

    /// Registers audio-file chunk serialization.
    pub fn audio_file_chunks(
        mut self,
        writer: impl FnMut(RawHandle) -> Result<AudioFileChunk, AraError> + 'static,
    ) -> Self {
        self.capabilities.chunk_writer = Some(Box::new(writer));
        self.controller_capabilities = self
            .controller_capabilities
            .with_audio_file_chunk_storage(true);
        self
    }

    /// Registers the audio-modification signal-preservation query.
    pub fn signal_preservation(mut self, query: impl Fn(RawHandle) -> bool + 'static) -> Self {
        self.capabilities.signal_query = Some(Box::new(query));
        self.controller_capabilities = self.controller_capabilities.with_signal_preservation(true);
        self
    }

    /// Registers immutable content snapshot production for controller reader callbacks.
    pub fn content(mut self, provider: impl ContentProvider + 'static) -> Self {
        self.capabilities.content = Some(Box::new(provider));
        self
    }

    /// Registers asynchronous audio-source analysis lifecycle hooks.
    pub fn analysis(mut self, provider: impl AnalysisProvider + 'static) -> Self {
        self.capabilities.analysis = Some(Box::new(provider));
        self
    }

    /// Registers complete-document persistence for legacy and null-filter archive calls.
    pub fn persistence(mut self, provider: impl Persistence + 'static) -> Self {
        self.capabilities.persistence = Some(PersistenceCapability::Complete(Box::new(provider)));
        self
    }

    /// Registers complete and ARA 2 filtered object persistence.
    pub fn partial_persistence(mut self, provider: impl PartialPersistence + 'static) -> Self {
        self.capabilities.persistence = Some(PersistenceCapability::Partial(Box::new(provider)));
        self
    }

    /// Registers the allocation-free playback-region head/tail snapshot adapter.
    pub fn realtime_head_tail(mut self, adapter: Arc<RealtimeHeadTailAdapter>) -> Self {
        self.capabilities.head_tail = Some(adapter);
        self
    }

    /// Completes the immutable plug-in definition.
    pub fn build(self) -> Result<Plugin<P>, AraError> {
        Ok(Plugin {
            model: self.model,
            capabilities: self.capabilities,
            controller_capabilities: self.controller_capabilities,
            updates: self.updates,
            analysis_events: self.analysis_events,
        })
    }
}

/// Composed plug-in model and optional capability registration.
pub struct Plugin<P> {
    pub(crate) model: P,
    pub(crate) capabilities: SemanticCapabilities,
    pub(crate) controller_capabilities: ControllerCapabilities,
    pub(crate) updates: UpdateEmitter,
    pub(crate) analysis_events: AnalysisEmitter,
}

impl<P> Plugin<P> {
    /// Borrows the retained required model.
    pub const fn model(&self) -> &P {
        &self.model
    }

    /// Borrows optional semantic capability state.
    pub const fn capabilities(&self) -> &SemanticCapabilities {
        &self.capabilities
    }

    /// Mutably borrows optional semantic capability state.
    pub fn capabilities_mut(&mut self) -> &mut SemanticCapabilities {
        &mut self.capabilities
    }

    /// Clones the application update emitter owned by this future controller.
    pub fn update_emitter(&self) -> UpdateEmitter {
        self.updates.clone()
    }

    /// Clones the analysis progress emitter owned by this future controller.
    pub fn analysis_emitter(&self) -> AnalysisEmitter {
        self.analysis_events.clone()
    }

    /// Builds the exact document-controller prefix for a negotiated generation.
    pub fn document_controller_interface(
        &self,
        generation: ApiGeneration,
    ) -> Result<ControllerInterface, AraError> {
        document_controller_interface(generation, self.controller_capabilities)
    }
}

impl<P: PluginModel> Plugin<P> {
    /// Consumes this definition and creates one safe document runtime.
    pub fn into_runtime(
        self,
        generation: ApiGeneration,
        properties: DocumentProperties,
    ) -> Result<(PluginRuntime<P>, SemanticCapabilities), AraError> {
        let runtime = PluginRuntime::new(self.model, generation, properties)?;
        Ok((runtime, self.capabilities))
    }
}
