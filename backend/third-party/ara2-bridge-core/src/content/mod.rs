//! Typed, owned ARA musical-content events.

mod events;
mod kind;
mod raw;
mod reader;
mod validate;

pub use events::{
    BarSignatureEvent, ChordEvent, ChordIntervalUsage, ContentGrade, ContentUpdateScopes,
    KeySignatureEvent, KeySignatureIntervalUsage, NoteEvent, TempoEvent, TuningEvent,
};
pub use kind::{
    validate_event_sequence, BarSignatures, ContentKind, KeySignatures, Notes, SheetChords,
    StaticTuning, Tempo,
};
pub use raw::copy_event_from_ffi;
pub use reader::{
    ContentReader, ContentReaderBackend, ContentReaderGate, ContentReaderLease,
    DynamicContentReader, EventRef, NoContentReaderBackend,
};
