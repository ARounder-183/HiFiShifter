//! Sealed content-kind markers.

use super::{raw, validate};
use super::{BarSignatureEvent, ChordEvent, KeySignatureEvent, NoteEvent, TempoEvent, TuningEvent};
use crate::AraError;
use ara2_bridge_sys::{
    kARAContentTypeBarSignatures, kARAContentTypeKeySignatures, kARAContentTypeNotes,
    kARAContentTypeSheetChords, kARAContentTypeStaticTuning, kARAContentTypeTempoEntries,
    ARAContentBarSignature, ARAContentChord, ARAContentKeySignature, ARAContentNote,
    ARAContentTempoEntry, ARAContentTuning, ARAContentType,
};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Associates an ARA content type with its owned Rust event.
pub trait ContentKind: sealed::Sealed + 'static {
    /// The aligned, owned event type.
    type Event: Clone + Send + Sync + 'static;

    /// The raw ARA content-type value.
    const RAW_TYPE: ARAContentType;

    /// The complete raw event extent.
    #[doc(hidden)]
    const RAW_EVENT_SIZE: usize;

    /// Copies a previously validated raw event pointer.
    ///
    /// # Safety
    ///
    /// `pointer` must be readable for [`Self::RAW_EVENT_SIZE`] bytes and contain this kind's event.
    #[doc(hidden)]
    unsafe fn copy_event(pointer: *const u8) -> Result<Self::Event, AraError>;

    /// Validates the count and ordering of an event sequence.
    #[doc(hidden)]
    fn validate_sequence(events: &[Self::Event]) -> Result<(), AraError>;

    /// Validates an event count before peer data is accessed.
    #[doc(hidden)]
    fn validate_count(count: usize) -> Result<(), AraError>;

    /// Validates two consecutively indexed events.
    #[doc(hidden)]
    fn validate_pair(previous: &Self::Event, current: &Self::Event) -> Result<(), AraError>;
}

/// Validates the upstream count and ordering rules for a typed event sequence.
pub fn validate_event_sequence<K: ContentKind>(events: &[K::Event]) -> Result<(), AraError> {
    K::validate_sequence(events)
}

macro_rules! content_kind {
    ($name:ident, $event:ty, $raw_event:ty, $raw:expr, $copy:path, $validate:path, $count:path, $pair:path, $doc:literal) => {
        #[doc = $doc]
        pub enum $name {}
        impl sealed::Sealed for $name {}
        impl ContentKind for $name {
            type Event = $event;
            const RAW_TYPE: ARAContentType = $raw;
            const RAW_EVENT_SIZE: usize = ::std::mem::size_of::<$raw_event>();

            unsafe fn copy_event(pointer: *const u8) -> Result<Self::Event, AraError> {
                // SAFETY: `ContentKind::copy_event` forwards the complete readable-event contract.
                unsafe { $copy(pointer) }
            }

            fn validate_sequence(events: &[Self::Event]) -> Result<(), AraError> {
                $validate(events)
            }

            fn validate_count(count: usize) -> Result<(), AraError> {
                $count(count)
            }

            fn validate_pair(
                previous: &Self::Event,
                current: &Self::Event,
            ) -> Result<(), AraError> {
                $pair(previous, current)
            }
        }
    };
}

content_kind!(
    Tempo,
    TempoEvent,
    ARAContentTempoEntry,
    kARAContentTypeTempoEntries as ARAContentType,
    raw::copy_tempo,
    validate::tempo_sequence,
    validate::tempo_count,
    validate::tempo_pair,
    "Tempo-entry content."
);
content_kind!(
    BarSignatures,
    BarSignatureEvent,
    ARAContentBarSignature,
    kARAContentTypeBarSignatures as ARAContentType,
    raw::copy_bar_signature,
    validate::bar_signature_sequence,
    validate::bar_signature_count,
    validate::bar_signature_pair,
    "Bar-signature content."
);
content_kind!(
    Notes,
    NoteEvent,
    ARAContentNote,
    kARAContentTypeNotes as ARAContentType,
    raw::copy_note,
    validate::note_sequence,
    validate::note_count,
    validate::note_pair,
    "Note content."
);
content_kind!(
    StaticTuning,
    TuningEvent,
    ARAContentTuning,
    kARAContentTypeStaticTuning as ARAContentType,
    raw::copy_tuning,
    validate::tuning_sequence,
    validate::tuning_count,
    validate::tuning_pair,
    "Static-tuning content."
);
content_kind!(
    KeySignatures,
    KeySignatureEvent,
    ARAContentKeySignature,
    kARAContentTypeKeySignatures as ARAContentType,
    raw::copy_key_signature,
    validate::key_signature_sequence,
    validate::key_signature_count,
    validate::key_signature_pair,
    "Key-signature content."
);
content_kind!(
    SheetChords,
    ChordEvent,
    ARAContentChord,
    kARAContentTypeSheetChords as ARAContentType,
    raw::copy_chord,
    validate::chord_sequence,
    validate::chord_count,
    validate::chord_pair,
    "Sheet-chord content."
);
