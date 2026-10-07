//! Generation- and capability-aware document-controller prefixes.

use super::generated_callbacks;
use ara2_bridge_core::{ApiGeneration, AraError};
use ara2_bridge_sys::ARADocumentControllerInterface;

/// Optional document-controller tail capabilities affecting the represented prefix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControllerCapabilities {
    processing_algorithms: bool,
    licensing: bool,
    audio_file_chunk_storage: bool,
    signal_preservation: bool,
}

impl ControllerCapabilities {
    /// Enables or disables the processing-algorithm callback group.
    pub const fn with_processing_algorithms(mut self, enabled: bool) -> Self {
        self.processing_algorithms = enabled;
        self
    }

    /// Enables or disables licensing callbacks.
    pub const fn with_licensing(mut self, enabled: bool) -> Self {
        self.licensing = enabled;
        self
    }

    /// Enables or disables audio-file chunk persistence callbacks.
    pub const fn with_audio_file_chunk_storage(mut self, enabled: bool) -> Self {
        self.audio_file_chunk_storage = enabled;
        self
    }

    /// Enables or disables the signal-preservation query.
    pub const fn with_signal_preservation(mut self, enabled: bool) -> Self {
        self.signal_preservation = enabled;
        self
    }

    const fn any(self) -> bool {
        self.processing_algorithms
            || self.licensing
            || self.audio_file_chunk_storage
            || self.signal_preservation
    }
}

/// Stable owned document-controller vtable with one advertised consecutive prefix.
pub struct ControllerInterface {
    raw: Box<ARADocumentControllerInterface>,
    represented_callbacks: usize,
}

impl ControllerInterface {
    /// Returns the stable raw vtable pointer.
    pub fn as_raw(&self) -> *const ARADocumentControllerInterface {
        self.raw.as_ref()
    }

    /// Returns the number of callbacks covered by the advertised prefix.
    pub const fn represented_callback_count(&self) -> usize {
        self.represented_callbacks
    }

    /// Checks that every callback inside the advertised prefix is non-null.
    pub fn represented_callbacks_are_non_null(&self) -> bool {
        generated_callbacks::represented_callbacks_are_non_null(
            &self.raw,
            self.represented_callbacks,
        )
    }

    /// Copies the complete packed record for ABI inspection.
    pub fn raw_copy(&self) -> ARADocumentControllerInterface {
        // SAFETY: the boxed record is fully initialized; unaligned copying supports its packed ABI.
        unsafe { (self.raw.as_ref() as *const ARADocumentControllerInterface).read_unaligned() }
    }
}

/// Builds the generation-required prefix and extends it through the latest enabled capability.
pub fn document_controller_interface(
    generation: ApiGeneration,
    capabilities: ControllerCapabilities,
) -> Result<ControllerInterface, AraError> {
    if !generation.supported_on_target() {
        return Err(AraError::Unsupported(
            "document-controller generation is unavailable on this target",
        ));
    }
    if generation < ApiGeneration::V2Final && capabilities.any() {
        return Err(AraError::Unsupported(
            "ARA 2 document-controller capabilities require ARA 2 Final",
        ));
    }

    let (mut represented_callbacks, mut struct_size) = if generation < ApiGeneration::V2Draft {
        (
            41,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_DESTROY_CONTENT_READER,
        )
    } else if generation < ApiGeneration::V2Final {
        (
            45,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_GET_PLAYBACK_REGION_HEAD_AND_TAIL_TIME,
        )
    } else {
        (
            47,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_STORE_OBJECTS_TO_ARCHIVE,
        )
    };

    for (enabled, count, extent) in [
        (
            capabilities.processing_algorithms,
            51,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_REQUEST_PROCESSING_ALGORITHM_FOR_AUDIO_SOURCE,
        ),
        (
            capabilities.licensing,
            52,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_IS_LICENSED_FOR_CAPABILITIES,
        ),
        (
            capabilities.audio_file_chunk_storage,
            53,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_STORE_AUDIO_SOURCE_TO_AUDIO_FILE_CHUNK,
        ),
        (
            capabilities.signal_preservation,
            54,
            ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_IS_AUDIO_MODIFICATION_PRESERVING_AUDIO_SOURCE_SIGNAL,
        ),
    ] {
        if enabled {
            represented_callbacks = count;
            struct_size = extent;
        }
    }

    Ok(ControllerInterface {
        raw: Box::new(generated_callbacks::raw_interface(struct_size)),
        represented_callbacks,
    })
}
