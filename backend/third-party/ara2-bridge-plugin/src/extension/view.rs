//! Editor-view role callbacks.

use super::{lock, with_state, ExtensionViewSelection};
use ara2_bridge_core::{ContentTimeRange, ForeignSlice, SizedInput};
use ara2_bridge_sys::*;
use std::mem::offset_of;

const MAX_VIEW_OBJECTS: usize = 1 << 20;

pub(crate) unsafe extern "C" fn notify_selection(
    reference: ARAEditorViewRef,
    selection: *const ARAViewSelection,
) {
    // SAFETY: the role reference is an ExtensionState pointer for the callback lifetime.
    unsafe {
        with_state(reference.cast(), (), |state| {
            if state.require_controller().is_err() || selection.is_null() {
                return;
            }
            // SAFETY: the host callback supplies a complete selection and live nested arrays.
            let Ok(selection) = copy_selection(selection) else {
                return;
            };
            *lock(&state.selection) = Some(selection);
        })
    }
}

pub(crate) unsafe extern "C" fn notify_hide_region_sequences(
    reference: ARAEditorViewRef,
    count: ARASize,
    sequences: *const ARARegionSequenceRef,
) {
    // SAFETY: the role reference is an ExtensionState pointer for the callback lifetime.
    unsafe {
        with_state(reference.cast(), (), |state| {
            if state.require_controller().is_err() || count > MAX_VIEW_OBJECTS {
                return;
            }
            // SAFETY: the callback supplies `count` live sequence references when nonzero.
            let Ok(sequences) = ForeignSlice::copy_from_raw(sequences, count) else {
                return;
            };
            *lock(&state.hidden_sequences) = sequences
                .as_slice()
                .iter()
                .map(|reference| *reference as usize)
                .collect();
        })
    }
}

unsafe fn copy_selection(
    selection: *const ARAViewSelection,
) -> Result<ExtensionViewSelection, ara2_bridge_core::AraError> {
    // SAFETY: forwarded complete selection contract.
    let input = unsafe { SizedInput::from_ptr(selection) }?;
    macro_rules! field {
        ($field:ident, $type:ty, $extent:ident) => {{
            // SAFETY: generated offset/type/extent identify this represented field.
            unsafe {
                input.copy_field::<$type>(
                    offset_of!(ARAViewSelection, $field),
                    ara2_bridge_sys::layout::$extent,
                )
            }?
        }};
    }
    let playback_count = field!(
        playbackRegionRefsCount,
        usize,
        ARAVIEW_SELECTION_PLAYBACK_REGION_REFS_COUNT
    );
    let playback_pointer = field!(
        playbackRegionRefs,
        *const ARAPlaybackRegionRef,
        ARAVIEW_SELECTION_PLAYBACK_REGION_REFS
    );
    let sequence_count = field!(
        regionSequenceRefsCount,
        usize,
        ARAVIEW_SELECTION_REGION_SEQUENCE_REFS_COUNT
    );
    let sequence_pointer = field!(
        regionSequenceRefs,
        *const ARARegionSequenceRef,
        ARAVIEW_SELECTION_REGION_SEQUENCE_REFS
    );
    let time_range = field!(
        timeRange,
        *const ARAContentTimeRange,
        ARAVIEW_SELECTION_TIME_RANGE
    );
    if playback_count > MAX_VIEW_OBJECTS || sequence_count > MAX_VIEW_OBJECTS {
        return Err(ara2_bridge_core::AraError::InvalidArgument(
            "view selection count exceeds limit",
        ));
    }
    // SAFETY: forwarded nested reference-array contract.
    let playback = unsafe { ForeignSlice::copy_from_raw(playback_pointer, playback_count) }?;
    // SAFETY: same nested reference-array contract.
    let sequences = unsafe { ForeignSlice::copy_from_raw(sequence_pointer, sequence_count) }?;
    // SAFETY: forwarded optional nested range contract.
    let time_range = unsafe { ContentTimeRange::copy_optional_from_ffi(time_range) }?;
    Ok(ExtensionViewSelection {
        playback_regions: playback
            .as_slice()
            .iter()
            .map(|reference| *reference as usize)
            .collect(),
        region_sequences: sequences
            .as_slice()
            .iter()
            .map(|reference| *reference as usize)
            .collect(),
        time_range,
    })
}
