//! Validated, lifetime-bound clients for services supplied by an ARA host.

mod archive;
mod audio;
mod content;
mod model_update;
mod playback;

pub use archive::{ArchiveAccess, HostArchiveReader, HostArchiveWriter};
pub use audio::{AudioAccess, HostAudioReader, SampleFormat};
pub use content::{HostAudioSourceRef, HostContentReader, HostContentScope, HostMusicalContextRef};
pub use model_update::ModelUpdateAccess;
pub use playback::PlaybackAccess;

use ara2_bridge_core::{ApiGeneration, AraError, SizedInput};
use ara2_bridge_sys::ARADocumentControllerHostInstance;
use std::marker::PhantomData;
use std::mem::offset_of;

/// Validated services supplied to one ARA document controller by its host.
pub struct HostClients<'host> {
    audio: AudioAccess<'host>,
    archives: ArchiveAccess<'host>,
    content: Option<content::ContentAccess<'host>>,
    model_updates: Option<ModelUpdateAccess<'host>>,
    playback: Option<PlaybackAccess<'host>>,
    _host: PhantomData<&'host ARADocumentControllerHostInstance>,
}

impl Drop for HostClients<'_> {
    fn drop(&mut self) {
        self.audio.revoke_all();
    }
}

impl<'host> HostClients<'host> {
    /// Validates a host instance and copies its callable interface prefixes.
    ///
    /// # Safety
    ///
    /// `host` and every non-null interface pointer represented by it must remain readable for
    /// `'host`. Callback implementations and opaque host references must obey the ARA 2.3
    /// contracts for the supplied generation and remain valid until this client is dropped.
    pub unsafe fn from_raw(
        host: *const ARADocumentControllerHostInstance,
        generation: ApiGeneration,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded from this constructor's caller contract.
        let input = unsafe { SizedInput::from_ptr(host) }?;
        macro_rules! field {
            ($name:ident, $extent:ident) => {{
                // SAFETY: offset, type, and generated extent describe the named packed field.
                unsafe {
                    input.copy_field(
                        offset_of!(ARADocumentControllerHostInstance, $name),
                        ara2_bridge_sys::layout::$extent,
                    )
                }?
            }};
        }

        let audio_ref = field!(
            audioAccessControllerHostRef,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_AUDIO_ACCESS_CONTROLLER_HOST_REF
        );
        let audio_interface = field!(
            audioAccessControllerInterface,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_AUDIO_ACCESS_CONTROLLER_INTERFACE
        );
        let archive_ref = field!(
            archivingControllerHostRef,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_ARCHIVING_CONTROLLER_HOST_REF
        );
        let archive_interface = field!(
            archivingControllerInterface,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_ARCHIVING_CONTROLLER_INTERFACE
        );
        let content_ref = field!(
            contentAccessControllerHostRef,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_CONTENT_ACCESS_CONTROLLER_HOST_REF
        );
        let content_interface = field!(
            contentAccessControllerInterface,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_CONTENT_ACCESS_CONTROLLER_INTERFACE
        );
        let model_ref = field!(
            modelUpdateControllerHostRef,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_MODEL_UPDATE_CONTROLLER_HOST_REF
        );
        let model_interface = field!(
            modelUpdateControllerInterface,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_MODEL_UPDATE_CONTROLLER_INTERFACE
        );
        let playback_ref = field!(
            playbackControllerHostRef,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_PLAYBACK_CONTROLLER_HOST_REF
        );
        let playback_interface = field!(
            playbackControllerInterface,
            ARADOCUMENT_CONTROLLER_HOST_INSTANCE_PLAYBACK_CONTROLLER_INTERFACE
        );

        // SAFETY: each interface pointer inherits the outer constructor's lifetime and validity.
        let audio = unsafe { AudioAccess::from_raw(audio_ref, audio_interface) }?;
        // SAFETY: same outer constructor contract.
        let archives =
            unsafe { ArchiveAccess::from_raw(archive_ref, archive_interface, generation) }?;
        // SAFETY: optional interfaces are either null or valid under the same contract.
        let content = unsafe { content::ContentAccess::from_raw(content_ref, content_interface) }?;
        // SAFETY: same optional-interface contract.
        let model_updates = unsafe { ModelUpdateAccess::from_raw(model_ref, model_interface) }?;
        // SAFETY: same optional-interface contract.
        let playback = unsafe { PlaybackAccess::from_raw(playback_ref, playback_interface) }?;
        Ok(Self {
            audio,
            archives,
            content,
            model_updates,
            playback,
            _host: PhantomData,
        })
    }

    /// Returns the required host audio-access client.
    pub const fn audio(&self) -> &AudioAccess<'host> {
        &self.audio
    }

    pub(crate) fn revoke_audio_source_readers(&self, source: HostAudioSourceRef) {
        self.audio.revoke_source(source.as_raw());
    }

    pub(crate) fn revoke_all_audio_readers(&self) {
        self.audio.revoke_all();
    }

    /// Returns the required host archiving client.
    pub const fn archives(&self) -> &ArchiveAccess<'host> {
        &self.archives
    }

    /// Returns no content client outside a dispatcher-issued call scope.
    pub const fn content(&self) -> Option<HostContentScope<'_, 'host>> {
        None
    }

    /// Returns the optional model-update notification client.
    pub const fn model_updates(&self) -> Option<&ModelUpdateAccess<'host>> {
        self.model_updates.as_ref()
    }

    /// Returns the optional playback-request client.
    pub const fn playback(&self) -> Option<&PlaybackAccess<'host>> {
        self.playback.as_ref()
    }

    /// Runs a host-content operation scoped to the current audio source callback.
    pub fn with_audio_source_content<R>(
        &self,
        current: HostAudioSourceRef,
        operation: impl for<'call> FnOnce(HostContentScope<'call, 'host>) -> R,
    ) -> R {
        operation(HostContentScope::for_audio_source(
            self.content.as_ref(),
            &self.audio,
            current,
        ))
    }

    /// Runs an audio-reader operation scoped to any audio-source management callback.
    pub fn with_audio_source_management<R>(
        &self,
        current: HostAudioSourceRef,
        operation: impl for<'call> FnOnce(HostContentScope<'call, 'host>) -> R,
    ) -> R {
        operation(HostContentScope::for_audio_source_management(
            &self.audio,
            current,
        ))
    }

    /// Runs a host-content operation scoped to the current musical-context callback.
    pub fn with_musical_context_content<R>(
        &self,
        current: HostMusicalContextRef,
        operation: impl for<'call> FnOnce(HostContentScope<'call, 'host>) -> R,
    ) -> R {
        operation(HostContentScope::for_musical_context(
            self.content.as_ref(),
            &self.audio,
            current,
        ))
    }

    /// Runs a host-content operation during `endEditing`, where any live object is eligible.
    pub fn with_end_editing_content<R>(
        &self,
        operation: impl for<'call> FnOnce(HostContentScope<'call, 'host>) -> R,
    ) -> R {
        operation(HostContentScope::end_editing(
            self.content.as_ref(),
            &self.audio,
        ))
    }
}
