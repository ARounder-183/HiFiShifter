//! Owned document-controller allocation joining safe runtime state to generated callbacks.

use crate::ffi::generated_callbacks::ControllerDelegate;
use crate::{
    document_controller_interface, ContentObject, ContentReaderSnapshot, ControllerCapabilities,
    HostAudioSourceRef, HostClients, HostMusicalContextRef, Plugin, PluginModel, PluginRuntime,
    SemanticCapabilities,
};
use ara2_bridge_core::{
    ApiGeneration, AraError, ContentTimeRange, ContentUpdateScopes, DocumentProperties,
    ForeignSlice, Handle, LicenseRequest, PlaybackTransformationFlags, RawHandle,
    RegionSequenceKind, RestoreFilter, SizedInput, StoreFilter,
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

    fn runtime_mut(&mut self) -> Result<&mut PluginRuntime<P>, AraError> {
        if !self.content_readers.is_empty() {
            return Err(AraError::InvalidState(
                "model operation conflicts with an active content reader",
            ));
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
        let _ = self
            .runtime_mut()
            .and_then(PluginRuntime::begin_callback_editing);
    }

    fn end_editing(&mut self) {
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        host.with_end_editing_content(|scope| {
            let _ = runtime.end_callback_editing_with_host(&scope);
        });
    }

    fn notify_model_updates(&mut self) {
        if !self.content_readers.is_empty()
            || self.runtime().map_or(true, PluginRuntime::is_editing)
        {
            return;
        }
        let Some(host) = self.host.model_updates() else {
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
        updates.flush_with(|notification, _| match notification {
            crate::UpdateNotification::AudioSource {
                source,
                range,
                flags,
            } => {
                if let Some(host_ref) = self.audio_source_hosts.get(&source).copied() {
                    let _ = host.notify_audio_source_changed(host_ref, range.as_ref(), flags);
                }
            }
            crate::UpdateNotification::AudioModification {
                modification,
                range,
                flags,
            } => {
                if let Some(host_ref) = self.audio_modification_hosts.get(&modification).copied() {
                    let _ = host.notify_audio_modification_changed(host_ref, range.as_ref(), flags);
                }
            }
            crate::UpdateNotification::PlaybackRegion {
                region,
                range,
                flags,
            } => {
                if let Some(host_ref) = self.playback_region_hosts.get(&region).copied() {
                    let _ = host.notify_playback_region_changed(host_ref, range.as_ref(), flags);
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
            return kARAFalse;
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
        if host_ref.is_null() {
            return null_mut();
        }
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
        let Ok(host_ref) = (unsafe { HostMusicalContextRef::from_raw(host_ref) }) else {
            return;
        };
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
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
        let Ok(host_ref) = (unsafe { HostMusicalContextRef::from_raw(host_ref) }) else {
            return;
        };
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
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
        if host_ref.is_null() {
            return null_mut();
        }
        // SAFETY: the ARA callback contract supplies complete ephemeral properties and nested data.
        let Ok(properties) =
            (unsafe { ara2_bridge_core::AudioSourceProperties::copy_from_ffi(properties) })
        else {
            return null_mut();
        };
        // SAFETY: the non-null callback identity remains live for the controller lifetime.
        let Ok(scoped_host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return null_mut();
        };
        if !self.content_readers.is_empty() {
            return null_mut();
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return null_mut();
        };
        let result: Result<(RawHandle, ARAAudioSourceRef), AraError> = host
            .with_audio_source_content(scoped_host_ref, |scope| {
                let mut edit = runtime.callback_edit()?;
                let handle = edit.create_audio_source_with_host(properties, &scope)?;
                let reference = edit.audio_source_ref(handle)?;
                Ok((handle.into_raw(), reference.as_raw().cast()))
            });
        let Ok((handle, reference)) = result else {
            return null_mut();
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
        let Ok(host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return;
        };
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
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
        let Ok(host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return;
        };
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let result = host.with_audio_source_management(host_ref, |scope| {
            let mut edit = runtime.callback_edit()?;
            edit.set_audio_source_deactivated_with_host(handle, deactivating, &scope)
        });
        if result.is_err() {
            return;
        }
        let raw = handle.into_raw();
        if deactivating {
            self.capabilities.cancel_analysis(raw);
            let _ = self.analysis.cancel(raw);
            self.analysis_events.cancel(raw);
            self.pending_analysis_starts.retain(|source| *source != raw);
            self.host.revoke_audio_source_readers(host_ref);
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
        let Ok(host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return;
        };
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
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
        let Some(host_ref) = self.audio_source_hosts.get(&raw).copied() else {
            return;
        };
        // SAFETY: the identity was validated on creation and remains live until destroy.
        let Ok(host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return;
        };
        if !self.content_readers.is_empty() {
            return;
        }
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        host.with_audio_source_management(host_ref, |scope| {
            let _ = runtime.set_audio_source_samples_access_with_host(
                handle,
                enable != kARAFalse,
                &scope,
            );
        });
        if enable == kARAFalse {
            self.host.revoke_audio_source_readers(host_ref);
        }
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
        let Ok(host_ref) = (unsafe { HostAudioSourceRef::from_raw(host_ref) }) else {
            return;
        };
        let host = &self.host;
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let result = host.with_audio_source_management(host_ref, |scope| {
            let mut edit = runtime.callback_edit()?;
            edit.destroy_audio_source_with_host(handle, &scope)
        });
        if result.is_err() {
            return;
        }
        self.host.revoke_audio_source_readers(host_ref);
        self.audio_source_hosts.remove(&raw);
    }

    fn create_audio_modification(
        &mut self,
        source_reference: ARAAudioSourceRef,
        host_ref: ARAAudioModificationHostRef,
        properties: *const ARAAudioModificationProperties,
    ) -> ARAAudioModificationRef {
        if host_ref.is_null() {
            return null_mut();
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
            let Ok(source) = runtime.resolve_audio_source(source_reference) else {
                return null_mut();
            };
            let Ok(mut edit) = runtime.callback_edit() else {
                return null_mut();
            };
            let Ok(handle) = edit.create_audio_modification(source, properties) else {
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

    fn clone_audio_modification(
        &mut self,
        source_reference: ARAAudioModificationRef,
        host_ref: ARAAudioModificationHostRef,
        properties: *const ARAAudioModificationProperties,
    ) -> ARAAudioModificationRef {
        if host_ref.is_null() {
            return null_mut();
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
        ffi_bool(
            self.audio_source_object(reference)
                .is_some_and(|object| self.capabilities.is_content_available(object, content_type)),
        )
    }

    fn is_audio_source_content_analysis_incomplete(
        &mut self,
        reference: ARAAudioSourceRef,
        content_type: ARAContentType,
    ) -> ARABool {
        let Some(ContentObject::AudioSource(source)) = self.audio_source_object(reference) else {
            return kARAFalse;
        };
        ffi_bool(
            self.analysis.contains(source, content_type)
                || self
                    .capabilities
                    .is_analysis_incomplete(source, content_type),
        )
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
            return;
        }
        if self
            .analysis
            .start(source, content_types.iter().copied())
            .is_err()
        {
            return;
        }
        if self
            .capabilities
            .request_analysis(source, &content_types)
            .is_err()
        {
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
        self.audio_source_object(reference).map_or(0, |object| {
            self.capabilities
                .content_grade(object, content_type)
                .as_raw()
        })
    }

    fn create_audio_source_content_reader(
        &mut self,
        reference: ARAAudioSourceRef,
        content_type: ARAContentType,
        range: *const ARAContentTimeRange,
    ) -> ARAContentReaderRef {
        let Some(object) = self.audio_source_object(reference) else {
            return null_mut();
        };
        // SAFETY: the callback supplies null or one complete aligned ephemeral range.
        let Ok(range) = (unsafe { ContentTimeRange::copy_optional_from_ffi(range) }) else {
            return null_mut();
        };
        self.create_reader_for(object, content_type, range)
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
    }

    fn create_region_sequence(
        &mut self,
        _host_ref: ARARegionSequenceHostRef,
        properties: *const ARARegionSequenceProperties,
    ) -> ARARegionSequenceRef {
        let Ok(runtime) = self.runtime_mut() else {
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
        if host_ref.is_null() {
            return null_mut();
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
                return null_mut();
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
            let Ok(handle) = edit.create_playback_region(modification, sequence, properties) else {
                return null_mut();
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
                return;
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
            legacy_restore_reader: None,
            factory_analyzable_content,
            factory_playback_transformations,
        };
        Ok(crate::ffi::callbacks::controller_instance(
            Box::new(adapter),
            interface,
        ))
    }
}
