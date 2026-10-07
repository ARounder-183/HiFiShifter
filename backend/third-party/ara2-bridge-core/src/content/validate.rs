//! Upstream-compatible count and ordering validation for typed content.

use super::{BarSignatureEvent, ChordEvent, KeySignatureEvent, NoteEvent, TempoEvent, TuningEvent};
use crate::AraError;

fn invalid(message: &'static str) -> AraError {
    AraError::InvalidArgument(message)
}

pub(super) fn tempo_count(count: usize) -> Result<(), AraError> {
    (count >= 2)
        .then_some(())
        .ok_or_else(|| invalid("tempo content requires at least two events"))
}

pub(super) fn tempo_pair(previous: &TempoEvent, current: &TempoEvent) -> Result<(), AraError> {
    if previous.time_position() >= current.time_position()
        || previous.quarter_position() >= current.quarter_position()
    {
        return Err(invalid("tempo positions must increase strictly"));
    }
    Ok(())
}

pub(super) fn tempo_sequence(events: &[TempoEvent]) -> Result<(), AraError> {
    tempo_count(events.len())?;
    events
        .windows(2)
        .try_for_each(|pair| tempo_pair(&pair[0], &pair[1]))
}

pub(super) fn bar_signature_count(count: usize) -> Result<(), AraError> {
    (count > 0)
        .then_some(())
        .ok_or_else(|| invalid("bar-signature content requires an event"))
}

pub(super) fn bar_signature_pair(
    previous: &BarSignatureEvent,
    current: &BarSignatureEvent,
) -> Result<(), AraError> {
    if previous.position() >= current.position() {
        return Err(invalid("bar-signature positions must increase strictly"));
    }
    let bar_length = f64::from(previous.numerator()) / f64::from(previous.denominator());
    let remainder = (current.position() - previous.position()) % bar_length;
    const ROUNDING_WINDOW: f64 = 1.0 / 32.0;
    if !(remainder < ROUNDING_WINDOW || remainder > bar_length - ROUNDING_WINDOW) {
        return Err(invalid("bar-signature change is not on a bar boundary"));
    }
    Ok(())
}

pub(super) fn bar_signature_sequence(events: &[BarSignatureEvent]) -> Result<(), AraError> {
    bar_signature_count(events.len())?;
    events
        .windows(2)
        .try_for_each(|pair| bar_signature_pair(&pair[0], &pair[1]))
}

pub(super) fn note_count(_count: usize) -> Result<(), AraError> {
    Ok(())
}

pub(super) fn note_pair(previous: &NoteEvent, current: &NoteEvent) -> Result<(), AraError> {
    if previous.start_position() > current.start_position() {
        return Err(invalid("note positions must be nondecreasing"));
    }
    Ok(())
}

pub(super) fn note_sequence(events: &[NoteEvent]) -> Result<(), AraError> {
    events
        .windows(2)
        .try_for_each(|pair| note_pair(&pair[0], &pair[1]))
}

pub(super) fn tuning_count(count: usize) -> Result<(), AraError> {
    (count == 1)
        .then_some(())
        .ok_or_else(|| invalid("static-tuning content requires exactly one event"))
}

pub(super) fn tuning_pair(_previous: &TuningEvent, _current: &TuningEvent) -> Result<(), AraError> {
    Err(invalid("static-tuning content contains multiple events"))
}

pub(super) fn tuning_sequence(events: &[TuningEvent]) -> Result<(), AraError> {
    tuning_count(events.len())
}

pub(super) fn key_signature_count(count: usize) -> Result<(), AraError> {
    (count > 0)
        .then_some(())
        .ok_or_else(|| invalid("key-signature content requires an event"))
}

pub(super) fn key_signature_pair(
    previous: &KeySignatureEvent,
    current: &KeySignatureEvent,
) -> Result<(), AraError> {
    if previous.position() >= current.position() {
        return Err(invalid("key-signature positions must increase strictly"));
    }
    Ok(())
}

pub(super) fn key_signature_sequence(events: &[KeySignatureEvent]) -> Result<(), AraError> {
    key_signature_count(events.len())?;
    events
        .windows(2)
        .try_for_each(|pair| key_signature_pair(&pair[0], &pair[1]))
}

pub(super) fn chord_count(_count: usize) -> Result<(), AraError> {
    Ok(())
}

pub(super) fn chord_pair(previous: &ChordEvent, current: &ChordEvent) -> Result<(), AraError> {
    if previous.position() >= current.position() {
        return Err(invalid("chord positions must increase strictly"));
    }
    Ok(())
}

pub(super) fn chord_sequence(events: &[ChordEvent]) -> Result<(), AraError> {
    events
        .windows(2)
        .try_for_each(|pair| chord_pair(&pair[0], &pair[1]))
}
