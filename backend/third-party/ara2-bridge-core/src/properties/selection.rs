//! Owned editor selection and time-range properties.

use super::{write_raw, zeroed_raw, FfiProperties, PlaybackRegionKind, RegionSequenceKind};
use crate::{ApiGeneration, AraError, ForeignSlice, ModelRef, SizedInput};
use ara2_bridge_sys::{
    layout, ARAContentTimeRange, ARAPlaybackRegionRef, ARARegionSequenceRef, ARASize,
    ARAViewSelection,
};
use std::mem::offset_of;
use std::pin::Pin;

/// A finite half-open time range with nonnegative duration.
#[derive(Clone, Debug)]
pub struct ContentTimeRange {
    raw: Box<ARAContentTimeRange>,
}

impl ContentTimeRange {
    /// Creates a validated time range in seconds.
    pub fn new(start: f64, duration: f64) -> Result<Self, AraError> {
        if !start.is_finite() || !duration.is_finite() || duration < 0.0 {
            return Err(AraError::InvalidArgument(
                "time range must be finite with nonnegative duration",
            ));
        }
        // SAFETY: both raw fields are floats and accept zero.
        let mut raw = unsafe { zeroed_raw::<ARAContentTimeRange>() };
        // SAFETY: generated record field offsets and matching types.
        unsafe {
            write_raw(&mut raw, offset_of!(ARAContentTimeRange, start), start);
            write_raw(
                &mut raw,
                offset_of!(ARAContentTimeRange, duration),
                duration,
            );
        }
        Ok(Self { raw: Box::new(raw) })
    }

    /// Returns the start position in seconds.
    pub fn start(&self) -> f64 {
        // SAFETY: initialized retained raw field with matching generated offset and type.
        unsafe {
            ara2_bridge_sys::access::read_field(
                self.raw.as_ref() as *const ARAContentTimeRange as *const u8,
                offset_of!(ARAContentTimeRange, start),
            )
        }
    }

    /// Returns the duration in seconds.
    pub fn duration(&self) -> f64 {
        // SAFETY: initialized retained raw field with matching generated offset and type.
        unsafe {
            ara2_bridge_sys::access::read_field(
                self.raw.as_ref() as *const ARAContentTimeRange as *const u8,
                offset_of!(ARAContentTimeRange, duration),
            )
        }
    }

    /// Returns a borrowed raw pointer valid for the lifetime of this value.
    pub fn as_ptr(&self) -> *const ARAContentTimeRange {
        self.raw.as_ref()
    }

    /// Copies an optional ephemeral ARA content range.
    ///
    /// # Safety
    ///
    /// A non-null `pointer` must be aligned and readable for one complete
    /// [`ARAContentTimeRange`] during this call.
    pub unsafe fn copy_optional_from_ffi(
        pointer: *const ARAContentTimeRange,
    ) -> Result<Option<Self>, AraError> {
        if pointer.is_null() {
            return Ok(None);
        }
        // SAFETY: forwarded from this method's caller contract.
        unsafe { Self::copy_from_ptr(pointer) }.map(Some)
    }

    unsafe fn copy_from_ptr(pointer: *const ARAContentTimeRange) -> Result<Self, AraError> {
        if pointer.is_null() {
            return Err(AraError::InvalidArgument("null time-range pointer"));
        }
        if pointer as usize % std::mem::align_of::<ARAContentTimeRange>() != 0 {
            return Err(AraError::InvalidArgument("misaligned time-range pointer"));
        }
        let base = pointer.cast::<u8>();
        // SAFETY: the enclosing selection contract guarantees readable nested time-range storage.
        let start = unsafe {
            ara2_bridge_sys::access::read_field(base, offset_of!(ARAContentTimeRange, start))
        };
        // SAFETY: same nested allocation and generated field metadata.
        let duration = unsafe {
            ara2_bridge_sys::access::read_field(base, offset_of!(ARAContentTimeRange, duration))
        };
        Self::new(start, duration)
    }
}

/// Owned editor-view selection with stable reference-array backing.
#[derive(Clone, Debug)]
pub struct ViewSelection {
    playback_regions: Box<[ARAPlaybackRegionRef]>,
    region_sequences: Box<[ARARegionSequenceRef]>,
    time_range: Option<ContentTimeRange>,
}

impl ViewSelection {
    /// Creates a selection from typed live model references.
    pub fn new(
        playback_regions: &[ModelRef<PlaybackRegionKind>],
        region_sequences: &[ModelRef<RegionSequenceKind>],
        time_range: Option<ContentTimeRange>,
    ) -> Result<Self, AraError> {
        let _ = ARASize::try_from(playback_regions.len())
            .map_err(|_| AraError::InvalidArgument("selection count overflow"))?;
        let _ = ARASize::try_from(region_sequences.len())
            .map_err(|_| AraError::InvalidArgument("selection count overflow"))?;
        Ok(Self {
            playback_regions: playback_regions
                .iter()
                .map(|reference| reference.as_raw().cast())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            region_sequences: region_sequences
                .iter()
                .map(|reference| reference.as_raw().cast())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            time_range,
        })
    }

    /// Copies an ephemeral view selection using checked runtime reference resolvers.
    ///
    /// # Safety
    ///
    /// The selection, its advertised prefix, reference arrays, and optional time range must remain
    /// readable and initialized for this call. Resolvers must reject references outside the owning
    /// session and references of the wrong model kind.
    pub unsafe fn copy_from_ffi_with_refs(
        pointer: *const ARAViewSelection,
        resolve_playback: impl Fn(
            ARAPlaybackRegionRef,
        ) -> Result<ModelRef<PlaybackRegionKind>, AraError>,
        resolve_sequence: impl Fn(
            ARARegionSequenceRef,
        ) -> Result<ModelRef<RegionSequenceKind>, AraError>,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded caller-valid storage contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        macro_rules! field {
            ($type:ty, $name:ident, $extent:path) => {{
                // SAFETY: generated offset/type/extent match this record.
                unsafe { input.copy_field::<$type>(offset_of!(ARAViewSelection, $name), $extent)? }
            }};
        }
        let playback_count = field!(
            ARASize,
            playbackRegionRefsCount,
            layout::ARAVIEW_SELECTION_PLAYBACK_REGION_REFS_COUNT
        );
        let playback_pointer = field!(
            *const ARAPlaybackRegionRef,
            playbackRegionRefs,
            layout::ARAVIEW_SELECTION_PLAYBACK_REGION_REFS
        );
        let sequence_count = field!(
            ARASize,
            regionSequenceRefsCount,
            layout::ARAVIEW_SELECTION_REGION_SEQUENCE_REFS_COUNT
        );
        let sequence_pointer = field!(
            *const ARARegionSequenceRef,
            regionSequenceRefs,
            layout::ARAVIEW_SELECTION_REGION_SEQUENCE_REFS
        );
        // SAFETY: the outer contract covers each represented array extent; validation checks
        // null/count agreement, alignment, and arithmetic before copying.
        let playback_raw =
            unsafe { ForeignSlice::copy_from_raw(playback_pointer, playback_count)? };
        // SAFETY: same contract for the region-sequence reference array.
        let sequence_raw =
            unsafe { ForeignSlice::copy_from_raw(sequence_pointer, sequence_count)? };
        let playback_regions = playback_raw
            .as_slice()
            .iter()
            .copied()
            .map(&resolve_playback)
            .collect::<Result<Vec<_>, _>>()?;
        let region_sequences = sequence_raw
            .as_slice()
            .iter()
            .copied()
            .map(&resolve_sequence)
            .collect::<Result<Vec<_>, _>>()?;
        let range_pointer = field!(
            *const ARAContentTimeRange,
            timeRange,
            layout::ARAVIEW_SELECTION_TIME_RANGE
        );
        let time_range = if range_pointer.is_null() {
            None
        } else {
            // SAFETY: the outer contract covers represented nested time-range storage.
            Some(unsafe { ContentTimeRange::copy_from_ptr(range_pointer)? })
        };
        Self::new(&playback_regions, &region_sequences, time_range)
    }

    /// Returns the number of selected playback regions.
    pub fn playback_region_count(&self) -> usize {
        self.playback_regions.len()
    }

    /// Returns the number of selected region sequences.
    pub fn region_sequence_count(&self) -> usize {
        self.region_sequences.len()
    }

    /// Returns the optional selected time range.
    pub const fn time_range(&self) -> Option<&ContentTimeRange> {
        self.time_range.as_ref()
    }

    /// Builds a pinned raw ARA2 selection record.
    pub fn as_ffi(
        &self,
        generation: ApiGeneration,
    ) -> Result<Pin<Box<FfiProperties<'_, ARAViewSelection>>>, AraError> {
        if !generation.supported_on_target() {
            return Err(AraError::Unsupported(
                "API generation is unavailable on this target",
            ));
        }
        if generation < ApiGeneration::V2Draft {
            return Err(AraError::Unsupported(
                "view selection is unavailable in ARA1",
            ));
        }
        // SAFETY: all raw fields accept zero.
        let mut raw = unsafe { zeroed_raw::<ARAViewSelection>() };
        // SAFETY: generated offsets and matching types; retained boxes keep all pointers stable.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARAViewSelection, structSize),
                layout::ARAVIEW_SELECTION_TIME_RANGE as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAViewSelection, playbackRegionRefsCount),
                self.playback_regions.len() as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAViewSelection, playbackRegionRefs),
                if self.playback_regions.is_empty() {
                    std::ptr::null()
                } else {
                    self.playback_regions.as_ptr()
                },
            );
            write_raw(
                &mut raw,
                offset_of!(ARAViewSelection, regionSequenceRefsCount),
                self.region_sequences.len() as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAViewSelection, regionSequenceRefs),
                if self.region_sequences.is_empty() {
                    std::ptr::null()
                } else {
                    self.region_sequences.as_ptr()
                },
            );
            write_raw(
                &mut raw,
                offset_of!(ARAViewSelection, timeRange),
                self.time_range
                    .as_ref()
                    .map_or(std::ptr::null(), ContentTimeRange::as_ptr),
            );
        }
        Ok(FfiProperties::pin(raw))
    }
}
