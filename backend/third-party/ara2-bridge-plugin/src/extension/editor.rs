//! Editor-renderer role callbacks.

use super::{with_state, ExtensionRoles};
use ara2_bridge_sys::*;

pub(crate) unsafe extern "C" fn add_playback_region(
    reference: ARAEditorRendererRef,
    region: ARAPlaybackRegionRef,
) {
    // SAFETY: the role reference is an ExtensionState pointer for the callback lifetime.
    unsafe {
        with_state(reference.cast(), (), |state| {
            let _ = state.add_playback_region(ExtensionRoles::EDITOR_RENDERER, region as usize);
        })
    }
}

pub(crate) unsafe extern "C" fn remove_playback_region(
    reference: ARAEditorRendererRef,
    region: ARAPlaybackRegionRef,
) {
    // SAFETY: same role-reference invariant as `add_playback_region`.
    unsafe {
        with_state(reference.cast(), (), |state| {
            let _ = state.remove_playback_region(ExtensionRoles::EDITOR_RENDERER, region as usize);
        })
    }
}

pub(crate) unsafe extern "C" fn add_region_sequence(
    reference: ARAEditorRendererRef,
    sequence: ARARegionSequenceRef,
) {
    // SAFETY: the role reference is an ExtensionState pointer for the callback lifetime.
    unsafe {
        with_state(reference.cast(), (), |state| {
            if state.require_controller().is_ok()
                && state.enabled.contains(ExtensionRoles::EDITOR_RENDERER)
                && !sequence.is_null()
            {
                state
                    .region_sequences
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(sequence as usize);
            }
        })
    }
}

pub(crate) unsafe extern "C" fn remove_region_sequence(
    reference: ARAEditorRendererRef,
    sequence: ARARegionSequenceRef,
) {
    // SAFETY: same role-reference invariant as `add_region_sequence`.
    unsafe {
        with_state(reference.cast(), (), |state| {
            if state.require_controller().is_ok() {
                state
                    .region_sequences
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&(sequence as usize));
            }
        })
    }
}
