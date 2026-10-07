//! Optional host model-update notifications with tail-prefix validation.

use ara2_bridge_core::{AraError, ContentTimeRange, ContentUpdateScopes, SizedInput};
use ara2_bridge_sys::*;
use std::marker::PhantomData;
use std::mem::offset_of;

type Analysis = unsafe extern "C" fn(
    ARAModelUpdateControllerHostRef,
    ARAAudioSourceHostRef,
    ARAAnalysisProgressState,
    f32,
);
type Source = unsafe extern "C" fn(
    ARAModelUpdateControllerHostRef,
    ARAAudioSourceHostRef,
    *const ARAContentTimeRange,
    ARAContentUpdateFlags,
);
type Modification = unsafe extern "C" fn(
    ARAModelUpdateControllerHostRef,
    ARAAudioModificationHostRef,
    *const ARAContentTimeRange,
    ARAContentUpdateFlags,
);
type Region = unsafe extern "C" fn(
    ARAModelUpdateControllerHostRef,
    ARAPlaybackRegionHostRef,
    *const ARAContentTimeRange,
    ARAContentUpdateFlags,
);
type Document = unsafe extern "C" fn(ARAModelUpdateControllerHostRef);

/// Optional model-update callbacks represented by the host's advertised prefix.
pub struct ModelUpdateAccess<'host> {
    host_ref: ARAModelUpdateControllerHostRef,
    analysis: Option<Analysis>,
    source: Option<Source>,
    modification: Option<Modification>,
    region: Option<Region>,
    document: Option<Document>,
    _lifetime: PhantomData<&'host ()>,
}

impl<'host> ModelUpdateAccess<'host> {
    pub(crate) unsafe fn from_raw(
        host_ref: ARAModelUpdateControllerHostRef,
        interface: *const ARAModelUpdateControllerInterface,
    ) -> Result<Option<Self>, AraError> {
        if interface.is_null() {
            return Ok(None);
        }
        if host_ref.is_null() {
            return Err(AraError::Abi("model-update host reference is null"));
        }
        // SAFETY: the caller supplies the represented optional interface for the lifetime.
        let input = unsafe { SizedInput::from_ptr(interface) }?;
        macro_rules! optional {
            ($field:ident, $type:ty, $extent:ident) => {{
                let extent = ara2_bridge_sys::layout::$extent;
                if input.contains_extent(extent) {
                    // SAFETY: represented generated offset/type/extent identify this field.
                    unsafe {
                        input.copy_field::<Option<$type>>(
                            offset_of!(ARAModelUpdateControllerInterface, $field),
                            extent,
                        )
                    }?
                } else {
                    None
                }
            }};
        }
        Ok(Some(Self {
            host_ref,
            analysis: optional!(
                notifyAudioSourceAnalysisProgress,
                Analysis,
                ARAMODEL_UPDATE_CONTROLLER_INTERFACE_NOTIFY_AUDIO_SOURCE_ANALYSIS_PROGRESS
            ),
            source: optional!(
                notifyAudioSourceContentChanged,
                Source,
                ARAMODEL_UPDATE_CONTROLLER_INTERFACE_NOTIFY_AUDIO_SOURCE_CONTENT_CHANGED
            ),
            modification: optional!(
                notifyAudioModificationContentChanged,
                Modification,
                ARAMODEL_UPDATE_CONTROLLER_INTERFACE_NOTIFY_AUDIO_MODIFICATION_CONTENT_CHANGED
            ),
            region: optional!(
                notifyPlaybackRegionContentChanged,
                Region,
                ARAMODEL_UPDATE_CONTROLLER_INTERFACE_NOTIFY_PLAYBACK_REGION_CONTENT_CHANGED
            ),
            document: optional!(
                notifyDocumentDataChanged,
                Document,
                ARAMODEL_UPDATE_CONTROLLER_INTERFACE_NOTIFY_DOCUMENT_DATA_CHANGED
            ),
            _lifetime: PhantomData,
        }))
    }

    /// Returns whether the represented prefix supports private-document dirty notification.
    pub const fn supports_document_data_changed(&self) -> bool {
        self.document.is_some()
    }

    /// Notifies the host that private document data changed, when supported.
    pub(crate) fn notify_document_data_changed(&self) -> Result<(), AraError> {
        let callback = self.document.ok_or(AraError::Unsupported(
            "document-data notification is unavailable",
        ))?;
        // SAFETY: callback and host reference were validated during construction.
        unsafe { callback(self.host_ref) };
        Ok(())
    }

    /// Reports ordered audio-source analysis progress when the host represents the callback.
    pub(crate) fn notify_analysis_progress(
        &self,
        source: ARAAudioSourceHostRef,
        state: ARAAnalysisProgressState,
        progress: f32,
    ) -> Result<(), AraError> {
        if source.is_null() || !progress.is_finite() || !(0.0..=1.0).contains(&progress) {
            return Err(AraError::InvalidArgument(
                "invalid analysis progress notification",
            ));
        }
        let callback = self.analysis.ok_or(AraError::Unsupported(
            "analysis-progress notification is unavailable",
        ))?;
        // SAFETY: callback and host/source references were validated by their owners.
        unsafe { callback(self.host_ref, source, state, progress) };
        Ok(())
    }

    /// Notifies the host that persistent audio-source content changed.
    pub(crate) fn notify_audio_source_changed(
        &self,
        source: ARAAudioSourceHostRef,
        range: Option<&ContentTimeRange>,
        flags: ContentUpdateScopes,
    ) -> Result<(), AraError> {
        let callback = self.source.ok_or(AraError::Unsupported(
            "audio-source update notification is unavailable",
        ))?;
        notify_content(self.host_ref, source, range, flags, callback)
    }

    /// Notifies the host that persistent audio-modification content changed.
    pub(crate) fn notify_audio_modification_changed(
        &self,
        modification: ARAAudioModificationHostRef,
        range: Option<&ContentTimeRange>,
        flags: ContentUpdateScopes,
    ) -> Result<(), AraError> {
        let callback = self.modification.ok_or(AraError::Unsupported(
            "audio-modification update notification is unavailable",
        ))?;
        notify_content(self.host_ref, modification, range, flags, callback)
    }

    /// Notifies the host that persistent playback-region content changed.
    pub(crate) fn notify_playback_region_changed(
        &self,
        region: ARAPlaybackRegionHostRef,
        range: Option<&ContentTimeRange>,
        flags: ContentUpdateScopes,
    ) -> Result<(), AraError> {
        let callback = self.region.ok_or(AraError::Unsupported(
            "playback-region update notification is unavailable",
        ))?;
        notify_content(self.host_ref, region, range, flags, callback)
    }
}

fn notify_content<HostRef>(
    host: ARAModelUpdateControllerHostRef,
    object: *mut HostRef,
    range: Option<&ContentTimeRange>,
    flags: ContentUpdateScopes,
    callback: unsafe extern "C" fn(
        ARAModelUpdateControllerHostRef,
        *mut HostRef,
        *const ARAContentTimeRange,
        ARAContentUpdateFlags,
    ),
) -> Result<(), AraError> {
    if object.is_null() {
        return Err(AraError::InvalidArgument(
            "model-update object reference is null",
        ));
    }
    let raw_range = range.map(|range| ARAContentTimeRange {
        start: range.start(),
        duration: range.duration(),
    });
    let pointer = raw_range
        .as_ref()
        .map_or(std::ptr::null(), std::ptr::from_ref);
    // SAFETY: callback and host/object references remain live; the optional local range spans call.
    unsafe { callback(host, object, pointer, flags.bits()) };
    Ok(())
}
