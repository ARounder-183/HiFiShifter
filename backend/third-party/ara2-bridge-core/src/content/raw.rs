//! Validated copying from caller-owned event storage.

use super::{
    BarSignatureEvent, ChordEvent, ChordIntervalUsage, ContentKind, KeySignatureEvent,
    KeySignatureIntervalUsage, NoteEvent, TempoEvent, TuningEvent,
};
use crate::{AraError, ForeignStr};
use ara2_bridge_sys::{
    kARAInvalidFrequency, kARAInvalidPitchNumber, ARAContentBarSignature, ARAContentChord,
    ARAContentKeySignature, ARAContentNote, ARAContentTempoEntry, ARAContentTuning, ARAContentType,
};
use std::ffi::{c_char, c_void};

const MAX_CONTENT_NAME_BYTES: usize = 1 << 20;

/// Copies one caller-owned ARA content event into its aligned owned Rust representation.
///
/// # Safety
///
/// `pointer` must be non-null and readable for `extent` bytes for the duration of this call. Its
/// storage must contain an event matching `raw_type`; it need not be naturally aligned. Any
/// non-null nested display-name pointer must be readable through a NUL terminator within 1 MiB.
pub unsafe fn copy_event_from_ffi<K: ContentKind>(
    raw_type: ARAContentType,
    pointer: *const c_void,
    extent: usize,
) -> Result<K::Event, AraError> {
    if raw_type != K::RAW_TYPE {
        return Err(AraError::InvalidArgument("content event kind mismatch"));
    }
    if pointer.is_null() {
        return Err(AraError::InvalidArgument("null content event pointer"));
    }
    if extent < K::RAW_EVENT_SIZE {
        return Err(AraError::InvalidArgument(
            "content event storage is truncated",
        ));
    }
    if extent > isize::MAX as usize {
        return Err(AraError::InvalidArgument(
            "content event extent is too large",
        ));
    }
    // SAFETY: the caller contract and checks above establish this kind's complete readable extent.
    unsafe { K::copy_event(pointer.cast::<u8>()) }
}

unsafe fn read<T>(pointer: *const u8) -> T {
    // SAFETY: every caller forwards a complete readable extent for `T`; unaligned input is allowed.
    unsafe { pointer.cast::<T>().read_unaligned() }
}

unsafe fn optional_name(pointer: *const c_char) -> Result<Option<String>, AraError> {
    if pointer.is_null() {
        return Ok(None);
    }
    // SAFETY: the raw event-copy contract covers nested display strings through the documented bound.
    unsafe { ForeignStr::copy_display(pointer, MAX_CONTENT_NAME_BYTES) }
        .map(ForeignStr::into_string)
        .map(Some)
}

pub(super) unsafe fn copy_tempo(pointer: *const u8) -> Result<TempoEvent, AraError> {
    // SAFETY: forwarded from the complete `ARAContentTempoEntry` event contract.
    let raw = unsafe { read::<ARAContentTempoEntry>(pointer) };
    TempoEvent::new(raw.timePosition, raw.quarterPosition)
}

pub(super) unsafe fn copy_bar_signature(pointer: *const u8) -> Result<BarSignatureEvent, AraError> {
    // SAFETY: forwarded from the complete `ARAContentBarSignature` event contract.
    let raw = unsafe { read::<ARAContentBarSignature>(pointer) };
    BarSignatureEvent::new(raw.numerator, raw.denominator, raw.position)
}

pub(super) unsafe fn copy_note(pointer: *const u8) -> Result<NoteEvent, AraError> {
    // SAFETY: forwarded from the complete `ARAContentNote` event contract.
    let raw = unsafe { read::<ARAContentNote>(pointer) };
    let unpitched =
        raw.frequency == kARAInvalidFrequency && raw.pitchNumber == kARAInvalidPitchNumber;
    let partially_unpitched =
        raw.frequency == kARAInvalidFrequency || raw.pitchNumber == kARAInvalidPitchNumber;
    if partially_unpitched && !unpitched {
        return Err(AraError::InvalidArgument(
            "note frequency and pitch invalid sentinels disagree",
        ));
    }
    NoteEvent::new(
        (!unpitched).then_some(raw.frequency),
        (!unpitched).then_some(raw.pitchNumber),
        raw.volume,
        raw.startPosition,
        raw.attackDuration,
        raw.noteDuration,
        raw.signalDuration,
    )
}

pub(super) unsafe fn copy_tuning(pointer: *const u8) -> Result<TuningEvent, AraError> {
    // SAFETY: forwarded from the complete `ARAContentTuning` event and nested-string contract.
    let raw = unsafe { read::<ARAContentTuning>(pointer) };
    // SAFETY: the raw event-copy contract covers this optional nested display string.
    let name = unsafe { optional_name(raw.name) }?;
    TuningEvent::new(raw.concertPitchFrequency, raw.root, raw.tunings, name)
}

pub(super) unsafe fn copy_key_signature(pointer: *const u8) -> Result<KeySignatureEvent, AraError> {
    // SAFETY: forwarded from the complete `ARAContentKeySignature` event and nested-string contract.
    let raw = unsafe { read::<ARAContentKeySignature>(pointer) };
    // SAFETY: the raw event-copy contract covers this optional nested display string.
    let name = unsafe { optional_name(raw.name) }?;
    let intervals = raw.intervals.map(KeySignatureIntervalUsage::from_raw);
    KeySignatureEvent::new(raw.root, intervals, name, raw.position)
}

pub(super) unsafe fn copy_chord(pointer: *const u8) -> Result<ChordEvent, AraError> {
    // SAFETY: forwarded from the complete `ARAContentChord` event and nested-string contract.
    let raw = unsafe { read::<ARAContentChord>(pointer) };
    // SAFETY: the raw event-copy contract covers this optional nested display string.
    let name = unsafe { optional_name(raw.name) }?;
    let intervals = raw
        .intervals
        .map(ChordIntervalUsage::from_raw)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("twelve inputs produce twelve outputs");
    ChordEvent::new(raw.root, raw.bass, intervals, name, raw.position)
}
