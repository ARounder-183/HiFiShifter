//! Immutable typed-content snapshots produced by safe plug-in delegates.

use ara2_bridge_core::{
    validate_event_sequence, AraError, BarSignatures, ContentGrade, ContentKind, KeySignatures,
    Notes, RawHandle, SheetChords, StaticTuning, Tempo,
};
use ara2_bridge_sys::*;
use std::ffi::{c_void, CString};
use std::marker::PhantomData;

/// Runtime-owned model category passed to content and analysis providers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentObject {
    /// One live audio source.
    AudioSource(RawHandle),
    /// One live audio modification.
    AudioModification(RawHandle),
    /// One live playback region.
    PlaybackRegion(RawHandle),
}

/// Validated immutable events retained for one ARA content reader lifetime.
pub struct ContentSnapshot<K: ContentKind> {
    events: Box<[K::Event]>,
    _kind: PhantomData<K>,
}

impl<K: ContentKind> ContentSnapshot<K> {
    /// Validates count, ordering, and event invariants before publishing a snapshot.
    pub fn new(events: impl IntoIterator<Item = K::Event>) -> Result<Self, AraError> {
        let events = events.into_iter().collect::<Vec<_>>().into_boxed_slice();
        validate_event_sequence::<K>(&events)?;
        Ok(Self {
            events,
            _kind: PhantomData,
        })
    }

    /// Returns the validated event count.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns whether the snapshot contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns one immutable event by checked index.
    pub fn get(&self, index: usize) -> Result<&K::Event, AraError> {
        self.events.get(index).ok_or(AraError::InvalidArgument(
            "content event index out of bounds",
        ))
    }

    /// Returns all validated events in snapshot order.
    pub fn events(&self) -> &[K::Event] {
        &self.events
    }
}

struct NamedEvents<R> {
    _names: Box<[Option<CString>]>,
    raw: Box<[R]>,
}

enum EventStorage {
    Tempo(Box<[ARAContentTempoEntry]>),
    BarSignatures(Box<[ARAContentBarSignature]>),
    Notes(Box<[ARAContentNote]>),
    StaticTuning(NamedEvents<ARAContentTuning>),
    KeySignatures(NamedEvents<ARAContentKeySignature>),
    SheetChords(NamedEvents<ARAContentChord>),
}

/// Type-erased immutable content reader storage retained until the host destroys its reader.
pub struct ContentReaderSnapshot {
    content_type: ARAContentType,
    grade: ContentGrade,
    events: EventStorage,
}

impl ContentReaderSnapshot {
    /// Returns the raw ARA content type represented by this snapshot.
    pub const fn content_type(&self) -> ARAContentType {
        self.content_type
    }

    /// Returns the content quality grade.
    pub const fn grade(&self) -> ContentGrade {
        self.grade
    }

    /// Returns the validated event count.
    pub fn len(&self) -> usize {
        match &self.events {
            EventStorage::Tempo(events) => events.len(),
            EventStorage::BarSignatures(events) => events.len(),
            EventStorage::Notes(events) => events.len(),
            EventStorage::StaticTuning(events) => events.raw.len(),
            EventStorage::KeySignatures(events) => events.raw.len(),
            EventStorage::SheetChords(events) => events.raw.len(),
        }
    }

    /// Returns whether the snapshot contains no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn event_pointer(&self, index: usize) -> Option<*const c_void> {
        macro_rules! pointer {
            ($events:expr) => {
                $events
                    .get(index)
                    .map(|event| std::ptr::from_ref(event).cast())
            };
        }
        match &self.events {
            EventStorage::Tempo(events) => pointer!(events),
            EventStorage::BarSignatures(events) => pointer!(events),
            EventStorage::Notes(events) => pointer!(events),
            EventStorage::StaticTuning(events) => pointer!(events.raw),
            EventStorage::KeySignatures(events) => pointer!(events.raw),
            EventStorage::SheetChords(events) => pointer!(events.raw),
        }
    }
}

impl ContentSnapshot<Tempo> {
    /// Erases a validated tempo snapshot for ABI reader publication.
    pub fn into_reader(self, grade: ContentGrade) -> ContentReaderSnapshot {
        let events = self
            .events
            .iter()
            .map(|event| ARAContentTempoEntry {
                timePosition: event.time_position(),
                quarterPosition: event.quarter_position(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ContentReaderSnapshot {
            content_type: Tempo::RAW_TYPE,
            grade,
            events: EventStorage::Tempo(events),
        }
    }
}

impl ContentSnapshot<BarSignatures> {
    /// Erases a validated bar-signature snapshot for ABI reader publication.
    pub fn into_reader(self, grade: ContentGrade) -> ContentReaderSnapshot {
        let events = self
            .events
            .iter()
            .map(|event| ARAContentBarSignature {
                numerator: event.numerator(),
                denominator: event.denominator(),
                position: event.position(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ContentReaderSnapshot {
            content_type: BarSignatures::RAW_TYPE,
            grade,
            events: EventStorage::BarSignatures(events),
        }
    }
}

impl ContentSnapshot<Notes> {
    /// Erases a validated note snapshot for ABI reader publication.
    pub fn into_reader(self, grade: ContentGrade) -> ContentReaderSnapshot {
        let events = self
            .events
            .iter()
            .map(|event| ARAContentNote {
                frequency: event.frequency().unwrap_or(kARAInvalidFrequency),
                pitchNumber: event.pitch().unwrap_or(kARAInvalidPitchNumber),
                volume: event.volume(),
                startPosition: event.start_position(),
                attackDuration: event.attack_duration(),
                noteDuration: event.note_duration(),
                signalDuration: event.signal_duration(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ContentReaderSnapshot {
            content_type: Notes::RAW_TYPE,
            grade,
            events: EventStorage::Notes(events),
        }
    }
}

fn names(values: impl Iterator<Item = Option<String>>) -> Box<[Option<CString>]> {
    values
        .map(|name| name.map(|name| CString::new(name).expect("validated content name")))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

impl ContentSnapshot<StaticTuning> {
    /// Erases a validated tuning snapshot for ABI reader publication.
    pub fn into_reader(self, grade: ContentGrade) -> ContentReaderSnapshot {
        let names = names(
            self.events
                .iter()
                .map(|event| event.name().map(str::to_owned)),
        );
        let raw = self
            .events
            .iter()
            .zip(names.iter())
            .map(|(event, name)| ARAContentTuning {
                concertPitchFrequency: event.concert_pitch_frequency(),
                root: event.root(),
                tunings: *event.tunings(),
                name: name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ContentReaderSnapshot {
            content_type: StaticTuning::RAW_TYPE,
            grade,
            events: EventStorage::StaticTuning(NamedEvents { _names: names, raw }),
        }
    }
}

impl ContentSnapshot<KeySignatures> {
    /// Erases a validated key-signature snapshot for ABI reader publication.
    pub fn into_reader(self, grade: ContentGrade) -> ContentReaderSnapshot {
        let names = names(
            self.events
                .iter()
                .map(|event| event.name().map(str::to_owned)),
        );
        let raw = self
            .events
            .iter()
            .zip(names.iter())
            .map(|(event, name)| ARAContentKeySignature {
                root: event.root(),
                intervals: event.intervals().map(|usage| usage.as_raw()),
                name: name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr()),
                position: event.position(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ContentReaderSnapshot {
            content_type: KeySignatures::RAW_TYPE,
            grade,
            events: EventStorage::KeySignatures(NamedEvents { _names: names, raw }),
        }
    }
}

impl ContentSnapshot<SheetChords> {
    /// Erases a validated chord snapshot for ABI reader publication.
    pub fn into_reader(self, grade: ContentGrade) -> ContentReaderSnapshot {
        let names = names(
            self.events
                .iter()
                .map(|event| event.name().map(str::to_owned)),
        );
        let raw = self
            .events
            .iter()
            .zip(names.iter())
            .map(|(event, name)| ARAContentChord {
                root: event.root(),
                bass: event.bass(),
                intervals: event.intervals().map(|usage| usage.as_raw()),
                name: name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr()),
                position: event.position(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ContentReaderSnapshot {
            content_type: SheetChords::RAW_TYPE,
            grade,
            events: EventStorage::SheetChords(NamedEvents { _names: names, raw }),
        }
    }
}
