//! Playback-renderer and deprecated ARA 1 callbacks.

use super::{lock, with_state, ExtensionRoles, ExtensionViewSelection};
use ara2_bridge_sys::*;

pub(crate) unsafe extern "C" fn add_playback_region(
    reference: ARAPlaybackRendererRef,
    region: ARAPlaybackRegionRef,
) {
    // SAFETY: the generated role reference is an ExtensionState pointer for this callback lifetime.
    unsafe {
        with_state(reference.cast(), (), |state| {
            let _ = state.add_playback_region(ExtensionRoles::PLAYBACK_RENDERER, region as usize);
        })
    }
}

pub(crate) unsafe extern "C" fn remove_playback_region(
    reference: ARAPlaybackRendererRef,
    region: ARAPlaybackRegionRef,
) {
    // SAFETY: same role-reference invariant as `add_playback_region`.
    unsafe {
        with_state(reference.cast(), (), |state| {
            let _ =
                state.remove_playback_region(ExtensionRoles::PLAYBACK_RENDERER, region as usize);
        })
    }
}

pub(crate) unsafe extern "C" fn legacy_set_playback_region(
    reference: ARAPlugInExtensionRef,
    region: ARAPlaybackRegionRef,
) {
    // SAFETY: the legacy reference is an ExtensionState pointer for this callback lifetime.
    unsafe {
        with_state(reference.cast(), (), |state| {
            if state.require_controller().is_ok() && state.legacy && !region.is_null() {
                let key = region as usize;
                let mut assignments = lock(&state.playback_regions);
                assignments.insert((ExtensionRoles::PLAYBACK_RENDERER.bits(), key));
                assignments.insert((ExtensionRoles::EDITOR_RENDERER.bits(), key));
                *lock(&state.selection) = Some(ExtensionViewSelection {
                    playback_regions: vec![key],
                    region_sequences: Vec::new(),
                    time_range: None,
                });
            }
        })
    }
}

pub(crate) unsafe extern "C" fn legacy_remove_playback_region(
    reference: ARAPlugInExtensionRef,
    region: ARAPlaybackRegionRef,
) {
    // SAFETY: same legacy-reference invariant as `legacy_set_playback_region`.
    unsafe {
        with_state(reference.cast(), (), |state| {
            if state.require_controller().is_ok() && state.legacy {
                let key = region as usize;
                let mut assignments = lock(&state.playback_regions);
                assignments.remove(&(ExtensionRoles::PLAYBACK_RENDERER.bits(), key));
                assignments.remove(&(ExtensionRoles::EDITOR_RENDERER.bits(), key));
                let mut selection = lock(&state.selection);
                if selection
                    .as_ref()
                    .is_some_and(|selection| selection.playback_regions == [key])
                {
                    *selection = None;
                }
            }
        })
    }
}
