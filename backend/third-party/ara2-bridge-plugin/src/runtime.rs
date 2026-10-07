//! Typed document graph and edit-session runtime.

use crate::model::{AudioModificationNode, Node, PlaybackRegionNode, RegionSequenceNode};
use crate::{
    AudioModifications, AudioSources, DocumentLifecycle, HostContentScope, MusicalContexts,
    PlaybackRegions, RegionSequences,
};
use ara2_bridge_core::{
    ApiGeneration, AraError, AudioModificationKind, AudioModificationProperties, AudioSourceKind,
    AudioSourceProperties, ContentTimeRange, ContentUpdateScopes, DocumentProperties, Handle,
    ModelRef, MusicalContextKind, MusicalContextProperties, PlaybackRegionKind,
    PlaybackRegionProperties, RawHandle, RegionSequenceKind, RegionSequenceProperties, Registry,
    RegistrySession,
};

/// Context supplied while an application object has a provisional stable identity.
#[derive(Clone, Copy, Debug)]
pub struct CreateContext {
    generation: ApiGeneration,
    object: Option<RawHandle>,
    realtime_key: Option<u64>,
}

impl CreateContext {
    fn document(generation: ApiGeneration) -> Self {
        Self {
            generation,
            object: None,
            realtime_key: None,
        }
    }

    fn object<K: 'static>(generation: ApiGeneration, handle: Handle<K>, realtime_key: u64) -> Self {
        Self {
            generation,
            object: Some(handle.into_raw()),
            realtime_key: Some(realtime_key),
        }
    }

    /// Returns the negotiated API generation.
    pub const fn generation(&self) -> ApiGeneration {
        self.generation
    }

    /// Returns the provisional object identity, or `None` while creating the document.
    pub const fn object_handle(&self) -> Option<RawHandle> {
        self.object
    }

    /// Returns the stable allocation-free lookup key for this provisional model identity.
    pub const fn realtime_key(&self) -> Option<u64> {
        self.realtime_key
    }
}

/// Complete required trait set for the safe document-model runtime.
///
/// 本地补丁（HiFiShifter）：补丁让 `AudioModifications` / `PlaybackRegions`
/// 各自多声明了一份关联类型（为了能接收父子边），这里用等式约束把它们
/// **钉回**各自"正主" trait 的同名类型 —— 于是实现方只需给一套类型。
pub trait PluginModel:
    DocumentLifecycle
    + MusicalContexts
    + RegionSequences
    + AudioSources
    + AudioModifications
    + PlaybackRegions
where
    Self: AudioModifications<AudioSource = <Self as AudioSources>::AudioSource>,
    Self: PlaybackRegions<
        AudioModification = <Self as AudioModifications>::AudioModification,
        RegionSequence = <Self as RegionSequences>::RegionSequence,
    >,
{
}

impl<T> PluginModel for T where
    T: DocumentLifecycle
        + MusicalContexts
        + RegionSequences
        + AudioSources
        + AudioModifications
        + PlaybackRegions
        + AudioModifications<AudioSource = <T as AudioSources>::AudioSource>
        + PlaybackRegions<
            AudioModification = <T as AudioModifications>::AudioModification,
            RegionSequence = <T as RegionSequences>::RegionSequence,
        >
{
}

/// Runtime-owned ARA document graph and application delegate.
pub struct PluginRuntime<P: PluginModel> {
    delegate: P,
    generation: ApiGeneration,
    document: Option<P::Document>,
    contexts: Registry<MusicalContextKind, Node<P::MusicalContext>>,
    // 本地补丁（HiFiShifter）：这几个关联类型现在在多个 trait 里都声明了
    // （为了让 `AudioModifications` / `PlaybackRegions` 的方法能收到父子边），
    // 所以要写全限定路径消歧 —— 它们指向的始终是定义该类型的那一个 trait。
    sequences: Registry<RegionSequenceKind, RegionSequenceNode<<P as RegionSequences>::RegionSequence>>,
    sources: Registry<AudioSourceKind, Node<<P as AudioSources>::AudioSource>>,
    modifications:
        Registry<AudioModificationKind, AudioModificationNode<<P as AudioModifications>::AudioModification>>,
    regions: Registry<PlaybackRegionKind, PlaybackRegionNode<P::PlaybackRegion>>,
    live_contexts: usize,
    live_sequences: usize,
    live_sources: usize,
    live_modifications: usize,
    live_regions: usize,
    editing: bool,
}

impl<P: PluginModel> PluginRuntime<P> {
    /// Creates application document state and empty typed registries.
    pub fn new(
        mut delegate: P,
        generation: ApiGeneration,
        properties: DocumentProperties,
    ) -> Result<Self, AraError> {
        if !generation.supported_on_target() {
            return Err(AraError::Unsupported(
                "runtime generation is unavailable on this target",
            ));
        }
        let document =
            delegate.create_document(&CreateContext::document(generation), properties)?;
        let session = RegistrySession::new();
        Ok(Self {
            delegate,
            generation,
            document: Some(document),
            contexts: Registry::in_session(
                session,
                Registry::<MusicalContextKind, ()>::DEFAULT_CAPACITY,
            ),
            sequences: Registry::in_session(
                session,
                Registry::<RegionSequenceKind, ()>::DEFAULT_CAPACITY,
            ),
            sources: Registry::in_session(
                session,
                Registry::<AudioSourceKind, ()>::DEFAULT_CAPACITY,
            ),
            modifications: Registry::in_session(
                session,
                Registry::<AudioModificationKind, ()>::DEFAULT_CAPACITY,
            ),
            regions: Registry::in_session(
                session,
                Registry::<PlaybackRegionKind, ()>::DEFAULT_CAPACITY,
            ),
            live_contexts: 0,
            live_sequences: 0,
            live_sources: 0,
            live_modifications: 0,
            live_regions: 0,
            editing: false,
        })
    }

    /// Borrows the application delegate for observation.
    pub const fn delegate(&self) -> &P {
        &self.delegate
    }

    /// Returns the negotiated API generation.
    pub const fn generation(&self) -> ApiGeneration {
        self.generation
    }

    pub(crate) fn session(&self) -> RegistrySession {
        self.contexts.session_id()
    }

    pub(crate) const fn is_editing(&self) -> bool {
        self.editing
    }

    /// Begins an exclusive graph mutation session.
    pub fn begin_editing(&mut self) -> Result<EditSession<'_, P>, AraError> {
        if self.editing {
            return Err(AraError::InvalidState("editing is already active"));
        }
        let document = self
            .document
            .as_mut()
            .ok_or(AraError::InvalidState("document has been destroyed"))?;
        self.delegate.begin_editing(document)?;
        self.editing = true;
        Ok(EditSession {
            runtime: self,
            finished: false,
            owns_state: true,
        })
    }

    pub(crate) fn begin_callback_editing(&mut self) -> Result<(), AraError> {
        if self.editing {
            return Err(AraError::InvalidState("editing is already active"));
        }
        let document = self
            .document
            .as_mut()
            .ok_or(AraError::InvalidState("document has been destroyed"))?;
        self.delegate.begin_editing(document)?;
        self.editing = true;
        Ok(())
    }

    pub(crate) fn end_callback_editing(&mut self) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.end_callback_editing_with_host(&host)
    }

    pub(crate) fn end_callback_editing_with_host(
        &mut self,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        if !self.editing {
            return Err(AraError::InvalidState("editing is not active"));
        }
        let document = self
            .document
            .as_mut()
            .ok_or(AraError::InvalidState("document has been destroyed"))?;
        self.delegate.end_editing(document, host)?;
        self.editing = false;
        Ok(())
    }

    pub(crate) fn callback_edit(&mut self) -> Result<EditSession<'_, P>, AraError> {
        if !self.editing {
            return Err(AraError::InvalidState("editing is not active"));
        }
        Ok(EditSession {
            runtime: self,
            finished: false,
            owns_state: false,
        })
    }

    /// Returns the stable model reference for a live musical context.
    pub fn musical_context_ref(
        &self,
        context: Handle<MusicalContextKind>,
    ) -> Result<ModelRef<MusicalContextKind>, AraError> {
        self.contexts.model_ref(context)
    }

    /// Returns the stable model reference for a live region sequence.
    pub fn region_sequence_ref(
        &self,
        sequence: Handle<RegionSequenceKind>,
    ) -> Result<ModelRef<RegionSequenceKind>, AraError> {
        self.sequences.model_ref(sequence)
    }

    pub(crate) fn audio_source_ref(
        &self,
        source: Handle<AudioSourceKind>,
    ) -> Result<ModelRef<AudioSourceKind>, AraError> {
        self.sources.model_ref(source)
    }

    pub(crate) fn audio_modification_ref(
        &self,
        modification: Handle<AudioModificationKind>,
    ) -> Result<ModelRef<AudioModificationKind>, AraError> {
        self.modifications.model_ref(modification)
    }

    pub(crate) fn playback_region_ref(
        &self,
        region: Handle<PlaybackRegionKind>,
    ) -> Result<ModelRef<PlaybackRegionKind>, AraError> {
        self.regions.model_ref(region)
    }

    pub(crate) fn resolve_musical_context(
        &self,
        reference: ara2_bridge_sys::ARAMusicalContextRef,
    ) -> Result<Handle<MusicalContextKind>, AraError> {
        self.contexts.handle_from_opaque(reference.cast())
    }

    pub(crate) fn resolve_region_sequence(
        &self,
        reference: ara2_bridge_sys::ARARegionSequenceRef,
    ) -> Result<Handle<RegionSequenceKind>, AraError> {
        self.sequences.handle_from_opaque(reference.cast())
    }

    pub(crate) fn resolve_audio_source(
        &self,
        reference: ara2_bridge_sys::ARAAudioSourceRef,
    ) -> Result<Handle<AudioSourceKind>, AraError> {
        self.sources.handle_from_opaque(reference.cast())
    }

    pub(crate) fn resolve_audio_modification(
        &self,
        reference: ara2_bridge_sys::ARAAudioModificationRef,
    ) -> Result<Handle<AudioModificationKind>, AraError> {
        self.modifications.handle_from_opaque(reference.cast())
    }

    pub(crate) fn resolve_playback_region(
        &self,
        reference: ara2_bridge_sys::ARAPlaybackRegionRef,
    ) -> Result<Handle<PlaybackRegionKind>, AraError> {
        self.regions.handle_from_opaque(reference.cast())
    }

    pub(crate) fn audio_source_is_active(
        &self,
        source: Handle<AudioSourceKind>,
    ) -> Result<bool, AraError> {
        Ok(self.sources.get(source)?.active)
    }

    pub(crate) fn audio_modification_is_active(
        &self,
        modification: Handle<AudioModificationKind>,
    ) -> Result<bool, AraError> {
        Ok(self.modifications.get(modification)?.node.active)
    }

    pub(crate) fn set_audio_source_samples_access_with_host(
        &mut self,
        source: Handle<AudioSourceKind>,
        enable: bool,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        let state = self.sources.get_mut(source)?.live_value();
        self.delegate
            .enable_audio_source_samples_access(state, enable, host)
    }

    pub(crate) fn validate_audio_source_destruction(
        &self,
        source: Handle<AudioSourceKind>,
    ) -> Result<(), AraError> {
        if !self.editing {
            return Err(AraError::InvalidState(
                "audio-source destruction requires an editing session",
            ));
        }
        if self.sources.get(source)?.children != 0 {
            return Err(AraError::InvalidState(
                "audio source still owns audio modifications",
            ));
        }
        Ok(())
    }

    /// Tears down an empty document and returns its application delegate.
    pub fn destroy(mut self) -> Result<P, AraError> {
        if self.live_contexts
            + self.live_sequences
            + self.live_sources
            + self.live_modifications
            + self.live_regions
            != 0
        {
            return Err(AraError::InvalidState(
                "document graph must be destroyed leaf-first before controller teardown",
            ));
        }
        let document = self
            .document
            .take()
            .expect("live runtime retains document state");
        self.delegate.destroy_document(document);
        Ok(self.delegate)
    }
}

/// Exclusive edit guard through which all graph mutations occur.
pub struct EditSession<'runtime, P: PluginModel> {
    runtime: &'runtime mut PluginRuntime<P>,
    finished: bool,
    owns_state: bool,
}

impl<P: PluginModel> EditSession<'_, P> {
    /// Borrows the delegate for testable state and diagnostics.
    pub const fn delegate(&self) -> &P {
        &self.runtime.delegate
    }

    /// Returns the number of committed live audio sources.
    pub const fn live_audio_source_count(&self) -> usize {
        self.runtime.live_sources
    }

    /// Applies copied document properties during this edit.
    pub fn update_document(&mut self, properties: DocumentProperties) -> Result<(), AraError> {
        let document = self
            .runtime
            .document
            .as_mut()
            .ok_or(AraError::InvalidState("document has been destroyed"))?;
        self.runtime.delegate.update_document(document, properties)
    }

    /// Applies host-originated musical-context content input.
    pub fn update_musical_context_content(
        &mut self,
        context: Handle<MusicalContextKind>,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.update_musical_context_content_with_host(context, range, flags, &host)
    }

    pub(crate) fn update_musical_context_content_with_host(
        &mut self,
        context: Handle<MusicalContextKind>,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        let state = self.runtime.contexts.get_mut(context)?.live_value();
        self.runtime
            .delegate
            .update_musical_context_content(state, range, flags, host)
    }

    /// Applies host-originated audio-source content input.
    pub fn update_audio_source_content(
        &mut self,
        source: Handle<AudioSourceKind>,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.update_audio_source_content_with_host(source, range, flags, &host)
    }

    pub(crate) fn update_audio_source_content_with_host(
        &mut self,
        source: Handle<AudioSourceKind>,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        let source = self.runtime.sources.get_mut(source)?;
        if !source.active {
            return Err(AraError::InvalidState("audio source is deactivated"));
        }
        let state = source.live_value();
        self.runtime
            .delegate
            .update_audio_source_content(state, range, flags, host)
    }

    /// Enables or revokes source sample access after validating source liveness.
    pub fn enable_audio_source_samples_access(
        &mut self,
        source: Handle<AudioSourceKind>,
        enable: bool,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.runtime
            .set_audio_source_samples_access_with_host(source, enable, &host)
    }

    /// Returns a stable reference for a live musical context.
    pub fn musical_context_ref(
        &self,
        context: Handle<MusicalContextKind>,
    ) -> Result<ModelRef<MusicalContextKind>, AraError> {
        self.runtime.musical_context_ref(context)
    }

    /// Returns a stable reference for a live region sequence.
    pub fn region_sequence_ref(
        &self,
        sequence: Handle<RegionSequenceKind>,
    ) -> Result<ModelRef<RegionSequenceKind>, AraError> {
        self.runtime.region_sequence_ref(sequence)
    }

    pub(crate) fn audio_source_ref(
        &self,
        source: Handle<AudioSourceKind>,
    ) -> Result<ModelRef<AudioSourceKind>, AraError> {
        self.runtime.audio_source_ref(source)
    }

    pub(crate) fn audio_modification_ref(
        &self,
        modification: Handle<AudioModificationKind>,
    ) -> Result<ModelRef<AudioModificationKind>, AraError> {
        self.runtime.audio_modification_ref(modification)
    }

    pub(crate) fn playback_region_ref(
        &self,
        region: Handle<PlaybackRegionKind>,
    ) -> Result<ModelRef<PlaybackRegionKind>, AraError> {
        self.runtime.playback_region_ref(region)
    }

    /// Provisions and creates a musical context.
    pub fn create_musical_context(
        &mut self,
        properties: MusicalContextProperties,
    ) -> Result<Handle<MusicalContextKind>, AraError> {
        let host = HostContentScope::unavailable();
        self.create_musical_context_with_host(properties, &host)
    }

    pub(crate) fn create_musical_context_with_host(
        &mut self,
        properties: MusicalContextProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<Handle<MusicalContextKind>, AraError> {
        let handle = self.runtime.contexts.insert(Node::provisional())?;
        let key = self.runtime.contexts.model_ref(handle)?.as_raw() as usize as u64;
        let context = CreateContext::object(self.runtime.generation, handle, key);
        match self
            .runtime
            .delegate
            .create_musical_context(&context, properties, host)
        {
            Ok(value) => {
                self.runtime.contexts.get_mut(handle)?.value = Some(value);
                self.runtime.live_contexts += 1;
                Ok(handle)
            }
            Err(error) => {
                self.runtime.contexts.remove(handle)?;
                Err(error)
            }
        }
    }

    /// Provisions a region sequence after validating its required musical context.
    pub fn create_region_sequence(
        &mut self,
        context_handle: Handle<MusicalContextKind>,
        properties: RegionSequenceProperties,
    ) -> Result<Handle<RegionSequenceKind>, AraError> {
        let expected = self.runtime.contexts.model_ref(context_handle)?;
        if expected.as_raw() != properties.musical_context().as_raw() {
            return Err(AraError::InvalidArgument(
                "region-sequence properties reference a different musical context",
            ));
        }
        if !self.runtime.contexts.get(context_handle)?.active {
            return Err(AraError::InvalidState("musical context is deactivated"));
        }
        let handle = self.runtime.sequences.insert(RegionSequenceNode {
            node: Node::provisional(),
            context: context_handle,
        })?;
        let key = self.runtime.sequences.model_ref(handle)?.as_raw() as usize as u64;
        let create = CreateContext::object(self.runtime.generation, handle, key);
        match self
            .runtime
            .delegate
            .create_region_sequence(&create, properties)
        {
            Ok(value) => {
                self.runtime.sequences.get_mut(handle)?.node.value = Some(value);
                self.runtime.contexts.get_mut(context_handle)?.children += 1;
                self.runtime.live_sequences += 1;
                Ok(handle)
            }
            Err(error) => {
                self.runtime.sequences.remove(handle)?;
                Err(error)
            }
        }
    }

    /// Provisions and creates an audio source.
    pub fn create_audio_source(
        &mut self,
        properties: AudioSourceProperties,
    ) -> Result<Handle<AudioSourceKind>, AraError> {
        let host = HostContentScope::unavailable();
        self.create_audio_source_with_host(properties, &host)
    }

    pub(crate) fn create_audio_source_with_host(
        &mut self,
        properties: AudioSourceProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<Handle<AudioSourceKind>, AraError> {
        let handle = self.runtime.sources.insert(Node::provisional())?;
        let key = self.runtime.sources.model_ref(handle)?.as_raw() as usize as u64;
        let context = CreateContext::object(self.runtime.generation, handle, key);
        match self
            .runtime
            .delegate
            .create_audio_source(&context, properties, host)
        {
            Ok(value) => {
                self.runtime.sources.get_mut(handle)?.value = Some(value);
                self.runtime.live_sources += 1;
                Ok(handle)
            }
            Err(error) => {
                self.runtime.sources.remove(handle)?;
                Err(error)
            }
        }
    }

    /// Provisions an audio modification under a live audio source.
    pub fn create_audio_modification(
        &mut self,
        source: Handle<AudioSourceKind>,
        properties: AudioModificationProperties,
    ) -> Result<Handle<AudioModificationKind>, AraError> {
        if !self.runtime.sources.get(source)?.active {
            return Err(AraError::InvalidState("audio source is deactivated"));
        }
        let handle = self.runtime.modifications.insert(AudioModificationNode {
            node: Node::provisional(),
            source,
        })?;
        let key = self.runtime.modifications.model_ref(handle)?.as_raw() as usize as u64;
        let context = CreateContext::object(self.runtime.generation, handle, key);
        // 本地补丁（HiFiShifter）：把 source 的应有态一并交给委托层 —— 上游把这
        // 条父子边丢掉了，而它是 `Take → Clip.source_path` 的映射输入。
        let result = {
            let (delegate, sources) = (&mut self.runtime.delegate, &self.runtime.sources);
            let source_value = sources
                .get(source)?
                .value
                .as_ref()
                .expect("committed audio source contains application state");
            delegate.create_audio_modification(&context, source_value, properties)
        };
        match result {
            Ok(value) => {
                self.runtime.modifications.get_mut(handle)?.node.value = Some(value);
                self.runtime.sources.get_mut(source)?.children += 1;
                self.runtime.live_modifications += 1;
                Ok(handle)
            }
            Err(error) => {
                self.runtime.modifications.remove(handle)?;
                Err(error)
            }
        }
    }

    /// Provisions a playback region after validating both required parent edges.
    pub fn create_playback_region(
        &mut self,
        modification: Handle<AudioModificationKind>,
        sequence: Handle<RegionSequenceKind>,
        properties: PlaybackRegionProperties,
    ) -> Result<Handle<PlaybackRegionKind>, AraError> {
        let modification_node = self.runtime.modifications.get(modification)?;
        if !modification_node.node.active {
            return Err(AraError::InvalidState("audio modification is deactivated"));
        }
        let sequence_node = self.runtime.sequences.get(sequence)?;
        if !sequence_node.node.active {
            return Err(AraError::InvalidState("region sequence is deactivated"));
        }
        if self.runtime.generation >= ApiGeneration::V2Draft {
            let expected = self.runtime.sequences.model_ref(sequence)?;
            if properties.region_sequence().map(ModelRef::as_raw) != Some(expected.as_raw()) {
                return Err(AraError::InvalidArgument(
                    "playback-region properties reference a different region sequence",
                ));
            }
        } else {
            let expected_context = self.runtime.contexts.model_ref(sequence_node.context)?;
            if properties.musical_context().map(ModelRef::as_raw) != Some(expected_context.as_raw())
            {
                return Err(AraError::InvalidArgument(
                    "ARA 1 playback region references a different musical context",
                ));
            }
        }
        let handle = self.runtime.regions.insert(PlaybackRegionNode {
            node: Node::provisional(),
            modification,
            sequence,
        })?;
        let key = self.runtime.regions.model_ref(handle)?.as_raw() as usize as u64;
        let context = CreateContext::object(self.runtime.generation, handle, key);
        // 本地补丁（HiFiShifter）：把宿主给的两条父子边（modification / sequence）
        // 交给委托层 —— 上游虽然持有它们，却没有往 trait 上暴露。
        let result = {
            let (delegate, modifications, sequences) = (
                &mut self.runtime.delegate,
                &self.runtime.modifications,
                &self.runtime.sequences,
            );
            let modification_value = modifications
                .get(modification)?
                .node
                .value
                .as_ref()
                .expect("committed modification contains application state");
            let sequence_value = sequences
                .get(sequence)?
                .node
                .value
                .as_ref()
                .expect("committed region sequence contains application state");
            delegate.create_playback_region(&context, modification_value, sequence_value, properties)
        };
        match result {
            Ok(value) => {
                self.runtime.regions.get_mut(handle)?.node.value = Some(value);
                self.runtime
                    .modifications
                    .get_mut(modification)?
                    .node
                    .children += 1;
                self.runtime.sequences.get_mut(sequence)?.node.children += 1;
                self.runtime.live_regions += 1;
                Ok(handle)
            }
            Err(error) => {
                self.runtime.regions.remove(handle)?;
                Err(error)
            }
        }
    }

    /// Destroys a leaf playback region and decrements both parent edges.
    pub fn destroy_playback_region(
        &mut self,
        region: Handle<PlaybackRegionKind>,
    ) -> Result<(), AraError> {
        let mut removed = self.runtime.regions.remove(region)?;
        self.runtime
            .modifications
            .get_mut(removed.modification)?
            .node
            .children -= 1;
        self.runtime
            .sequences
            .get_mut(removed.sequence)?
            .node
            .children -= 1;
        self.runtime.delegate.destroy_playback_region(
            removed
                .node
                .value
                .take()
                .expect("committed playback region contains application state"),
        );
        self.runtime.live_regions -= 1;
        Ok(())
    }

    /// Destroys a childless audio modification after invalidating its identity.
    pub fn destroy_audio_modification(
        &mut self,
        modification: Handle<AudioModificationKind>,
    ) -> Result<(), AraError> {
        if self.runtime.modifications.get(modification)?.node.children != 0 {
            return Err(AraError::InvalidState(
                "audio modification still owns playback regions",
            ));
        }
        let removed = self.runtime.modifications.remove(modification)?;
        self.runtime.sources.get_mut(removed.source)?.children -= 1;
        self.runtime.delegate.destroy_audio_modification(
            removed
                .node
                .value
                .expect("committed modification contains application state"),
        );
        self.runtime.live_modifications -= 1;
        Ok(())
    }

    /// Updates a live musical context.
    pub fn update_musical_context(
        &mut self,
        context: Handle<MusicalContextKind>,
        properties: MusicalContextProperties,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.update_musical_context_with_host(context, properties, &host)
    }

    pub(crate) fn update_musical_context_with_host(
        &mut self,
        context: Handle<MusicalContextKind>,
        properties: MusicalContextProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        let state = self.runtime.contexts.get_mut(context)?.live_value();
        self.runtime
            .delegate
            .update_musical_context(state, properties, host)
    }

    /// Updates a live region sequence without changing its required parent edge.
    pub fn update_region_sequence(
        &mut self,
        sequence: Handle<RegionSequenceKind>,
        properties: RegionSequenceProperties,
    ) -> Result<(), AraError> {
        let context = self.runtime.sequences.get(sequence)?.context;
        self.update_region_sequence_with_context(sequence, context, properties)
    }

    pub(crate) fn update_region_sequence_with_context(
        &mut self,
        sequence: Handle<RegionSequenceKind>,
        context: Handle<MusicalContextKind>,
        properties: RegionSequenceProperties,
    ) -> Result<(), AraError> {
        let expected = self.runtime.contexts.model_ref(context)?;
        if expected.as_raw() != properties.musical_context().as_raw() {
            return Err(AraError::InvalidArgument(
                "region-sequence properties reference a different musical context",
            ));
        }
        let previous = self.runtime.sequences.get(sequence)?.context;
        self.runtime.delegate.update_region_sequence(
            self.runtime.sequences.get_mut(sequence)?.node.live_value(),
            properties,
        )?;
        if previous != context {
            self.runtime.contexts.get_mut(previous)?.children -= 1;
            self.runtime.contexts.get_mut(context)?.children += 1;
            self.runtime.sequences.get_mut(sequence)?.context = context;
        }
        Ok(())
    }

    /// Updates a live audio source.
    pub fn update_audio_source(
        &mut self,
        source: Handle<AudioSourceKind>,
        properties: AudioSourceProperties,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.update_audio_source_with_host(source, properties, &host)
    }

    pub(crate) fn update_audio_source_with_host(
        &mut self,
        source: Handle<AudioSourceKind>,
        properties: AudioSourceProperties,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        let source = self.runtime.sources.get_mut(source)?;
        if !source.active {
            return Err(AraError::InvalidState("audio source is deactivated"));
        }
        let state = source.live_value();
        self.runtime
            .delegate
            .update_audio_source(state, properties, host)
    }

    /// Updates a live audio modification.
    pub fn update_audio_modification(
        &mut self,
        modification: Handle<AudioModificationKind>,
        properties: AudioModificationProperties,
    ) -> Result<(), AraError> {
        let modification = self.runtime.modifications.get_mut(modification)?;
        if !modification.node.active {
            return Err(AraError::InvalidState("audio modification is deactivated"));
        }
        let state = modification.node.live_value();
        self.runtime
            .delegate
            .update_audio_modification(state, properties)
    }

    /// Updates a playback region while preserving both parent edges.
    pub fn update_playback_region(
        &mut self,
        region: Handle<PlaybackRegionKind>,
        properties: PlaybackRegionProperties,
    ) -> Result<(), AraError> {
        let sequence = self.runtime.regions.get(region)?.sequence;
        self.update_playback_region_with_sequence(region, sequence, properties)
    }

    pub(crate) fn update_playback_region_with_sequence(
        &mut self,
        region: Handle<PlaybackRegionKind>,
        sequence: Handle<RegionSequenceKind>,
        properties: PlaybackRegionProperties,
    ) -> Result<(), AraError> {
        if self.runtime.generation >= ApiGeneration::V2Draft {
            let expected = self.runtime.sequences.model_ref(sequence)?;
            if properties.region_sequence().map(ModelRef::as_raw) != Some(expected.as_raw()) {
                return Err(AraError::InvalidArgument(
                    "playback-region properties reference a different region sequence",
                ));
            }
        } else {
            let context = self.runtime.sequences.get(sequence)?.context;
            let expected = self.runtime.contexts.model_ref(context)?;
            if properties.musical_context().map(ModelRef::as_raw) != Some(expected.as_raw()) {
                return Err(AraError::InvalidArgument(
                    "ARA 1 playback region references a different musical context",
                ));
            }
        }
        let previous = self.runtime.regions.get(region)?.sequence;
        self.runtime.delegate.update_playback_region(
            self.runtime.regions.get_mut(region)?.node.live_value(),
            properties,
        )?;
        if previous != sequence {
            self.runtime.sequences.get_mut(previous)?.node.children -= 1;
            self.runtime.sequences.get_mut(sequence)?.node.children += 1;
            self.runtime.regions.get_mut(region)?.sequence = sequence;
        }
        Ok(())
    }

    /// Creates an independent modification state by cloning a live sibling.
    pub fn clone_audio_modification(
        &mut self,
        source_modification: Handle<AudioModificationKind>,
        properties: AudioModificationProperties,
    ) -> Result<Handle<AudioModificationKind>, AraError> {
        let source_handle = self.runtime.modifications.get(source_modification)?.source;
        if !self.runtime.sources.get(source_handle)?.active {
            return Err(AraError::InvalidState("audio source is deactivated"));
        }
        let handle = self.runtime.modifications.insert(AudioModificationNode {
            node: Node::provisional(),
            source: source_handle,
        })?;
        let key = self.runtime.modifications.model_ref(handle)?.as_raw() as usize as u64;
        let context = CreateContext::object(self.runtime.generation, handle, key);
        let result = {
            // 本地补丁（HiFiShifter）：克隆出的 modification 也要知道自己的 source，
            // 所以这里还要把 source 的应有态借出来。
            let (delegate, modifications, sources) = (
                &mut self.runtime.delegate,
                &self.runtime.modifications,
                &self.runtime.sources,
            );
            let source = modifications
                .get(source_modification)?
                .node
                .value
                .as_ref()
                .expect("committed modification contains application state");
            let audio_source = sources
                .get(source_handle)?
                .value
                .as_ref()
                .expect("committed audio source contains application state");
            delegate.clone_audio_modification(&context, source, audio_source, properties)
        };
        match result {
            Ok(value) => {
                self.runtime.modifications.get_mut(handle)?.node.value = Some(value);
                self.runtime.sources.get_mut(source_handle)?.children += 1;
                self.runtime.live_modifications += 1;
                Ok(handle)
            }
            Err(error) => {
                self.runtime.modifications.remove(handle)?;
                Err(error)
            }
        }
    }

    /// Deactivates an audio source without destroying its identity or state.
    pub fn deactivate_audio_source(
        &mut self,
        source: Handle<AudioSourceKind>,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.set_audio_source_deactivated_with_host(source, true, &host)
    }

    pub(crate) fn set_audio_source_deactivated_with_host(
        &mut self,
        source: Handle<AudioSourceKind>,
        deactivated: bool,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        if deactivated
            && self
                .runtime
                .modifications
                .values()
                .any(|modification| modification.source == source && modification.node.active)
        {
            return Err(AraError::InvalidState(
                "audio source still owns an active audio modification",
            ));
        }
        let (delegate, sources) = (&mut self.runtime.delegate, &mut self.runtime.sources);
        let source = sources.get_mut(source)?;
        delegate.deactivate_audio_source(source.live_value(), deactivated, host)?;
        source.active = !deactivated;
        Ok(())
    }

    /// Deactivates an audio modification without destroying its identity or state.
    pub fn deactivate_audio_modification(
        &mut self,
        modification: Handle<AudioModificationKind>,
    ) -> Result<(), AraError> {
        self.set_audio_modification_deactivated(modification, true)
    }

    pub(crate) fn set_audio_modification_deactivated(
        &mut self,
        modification: Handle<AudioModificationKind>,
        deactivated: bool,
    ) -> Result<(), AraError> {
        let node = self.runtime.modifications.get(modification)?;
        if deactivated && node.node.children != 0 {
            return Err(AraError::InvalidState(
                "audio modification still owns playback regions",
            ));
        }
        if !deactivated && !self.runtime.sources.get(node.source)?.active {
            return Err(AraError::InvalidState(
                "audio modification cannot reactivate before its audio source",
            ));
        }
        let (delegate, modifications) =
            (&mut self.runtime.delegate, &mut self.runtime.modifications);
        let modification = modifications.get_mut(modification)?;
        delegate.deactivate_audio_modification(modification.node.live_value(), deactivated)?;
        modification.node.active = !deactivated;
        Ok(())
    }

    /// Destroys a childless region sequence after invalidating its identity.
    pub fn destroy_region_sequence(
        &mut self,
        sequence: Handle<RegionSequenceKind>,
    ) -> Result<(), AraError> {
        if self.runtime.sequences.get(sequence)?.node.children != 0 {
            return Err(AraError::InvalidState(
                "region sequence still owns playback regions",
            ));
        }
        let removed = self.runtime.sequences.remove(sequence)?;
        self.runtime.contexts.get_mut(removed.context)?.children -= 1;
        self.runtime.delegate.destroy_region_sequence(
            removed
                .node
                .value
                .expect("committed sequence contains application state"),
        );
        self.runtime.live_sequences -= 1;
        Ok(())
    }

    /// Destroys a childless musical context after invalidating its identity.
    pub fn destroy_musical_context(
        &mut self,
        context: Handle<MusicalContextKind>,
    ) -> Result<(), AraError> {
        if self.runtime.contexts.get(context)?.children != 0 {
            return Err(AraError::InvalidState(
                "musical context still owns region sequences",
            ));
        }
        let removed = self.runtime.contexts.remove(context)?;
        self.runtime.delegate.destroy_musical_context(
            removed
                .value
                .expect("committed context contains application state"),
        );
        self.runtime.live_contexts -= 1;
        Ok(())
    }

    /// Destroys a childless audio source after invalidating its identity.
    pub fn destroy_audio_source(
        &mut self,
        source: Handle<AudioSourceKind>,
    ) -> Result<(), AraError> {
        let host = HostContentScope::unavailable();
        self.destroy_audio_source_with_host(source, &host)
    }

    pub(crate) fn destroy_audio_source_with_host(
        &mut self,
        source: Handle<AudioSourceKind>,
        host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        if self.runtime.sources.get(source)?.children != 0 {
            return Err(AraError::InvalidState(
                "audio source still owns audio modifications",
            ));
        }
        let removed = self.runtime.sources.remove(source)?;
        self.runtime.delegate.destroy_audio_source(
            removed
                .value
                .expect("committed source contains application state"),
            host,
        );
        self.runtime.live_sources -= 1;
        Ok(())
    }

    /// Ends this edit session explicitly.
    pub fn finish(mut self) -> Result<(), AraError> {
        if self.owns_state {
            let host = HostContentScope::unavailable();
            self.runtime.end_callback_editing_with_host(&host)?;
        }
        self.finished = true;
        Ok(())
    }
}

impl<P: PluginModel> Drop for EditSession<'_, P> {
    fn drop(&mut self) {
        if !self.finished && self.owns_state {
            let host = HostContentScope::unavailable();
            let _ = self.runtime.end_callback_editing_with_host(&host);
            self.finished = true;
        }
    }
}
