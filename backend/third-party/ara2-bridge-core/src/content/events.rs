//! Owned content-event values.

use crate::AraError;
use bitflags::bitflags;

fn finite(value: f64, field: &'static str) -> Result<f64, AraError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(AraError::InvalidArgument(field))
}

fn finite_f32(value: f32, field: &'static str) -> Result<f32, AraError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(AraError::InvalidArgument(field))
}

fn display_name(name: Option<String>) -> Result<Option<String>, AraError> {
    if name.as_ref().is_some_and(|name| name.contains('\0')) {
        return Err(AraError::InvalidArgument("content name contains NUL"));
    }
    Ok(name)
}

/// A tempo synchronization point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoEvent {
    time_position: f64,
    quarter_position: f64,
}

impl TempoEvent {
    /// Creates a finite tempo synchronization point.
    pub fn new(time_position: f64, quarter_position: f64) -> Result<Self, AraError> {
        Ok(Self {
            time_position: finite(time_position, "tempo time is not finite")?,
            quarter_position: finite(quarter_position, "tempo quarter is not finite")?,
        })
    }

    /// Returns the position in seconds.
    pub fn time_position(&self) -> f64 {
        self.time_position
    }

    /// Returns the corresponding position in quarter notes.
    pub fn quarter_position(&self) -> f64 {
        self.quarter_position
    }
}

/// A bar-signature change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarSignatureEvent {
    numerator: i32,
    denominator: i32,
    position: f64,
}

impl BarSignatureEvent {
    /// Creates a positive bar signature at a finite quarter-note position.
    pub fn new(numerator: i32, denominator: i32, position: f64) -> Result<Self, AraError> {
        if numerator <= 0 || denominator <= 0 {
            return Err(AraError::InvalidArgument(
                "bar-signature terms must be positive",
            ));
        }
        Ok(Self {
            numerator,
            denominator,
            position: finite(position, "bar-signature position is not finite")?,
        })
    }

    /// Returns the numerator.
    pub fn numerator(&self) -> i32 {
        self.numerator
    }

    /// Returns the denominator.
    pub fn denominator(&self) -> i32 {
        self.denominator
    }

    /// Returns the position in quarter notes.
    pub fn position(&self) -> f64 {
        self.position
    }
}

/// An analyzed musical note.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteEvent {
    frequency: Option<f32>,
    pitch_number: Option<i32>,
    volume: f32,
    start_position: f64,
    attack_duration: f64,
    note_duration: f64,
    signal_duration: f64,
}

impl NoteEvent {
    /// Creates a validated pitched or unpitched note.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Option<f32>,
        pitch_number: Option<i32>,
        volume: f32,
        start_position: f64,
        attack_duration: f64,
        note_duration: f64,
        signal_duration: f64,
    ) -> Result<Self, AraError> {
        if frequency.is_some() != pitch_number.is_some() {
            return Err(AraError::InvalidArgument(
                "note frequency and pitch must be present together",
            ));
        }
        let frequency = frequency
            .map(|frequency| {
                let frequency = finite_f32(frequency, "note frequency is not finite")?;
                (frequency > 0.0)
                    .then_some(frequency)
                    .ok_or(AraError::InvalidArgument("note frequency must be positive"))
            })
            .transpose()?;
        let volume = finite_f32(volume, "note volume is not finite")?;
        if volume < 0.0 {
            return Err(AraError::InvalidArgument("note volume is negative"));
        }
        let start_position = finite(start_position, "note start is not finite")?;
        let attack_duration = finite(attack_duration, "note attack is not finite")?;
        let note_duration = finite(note_duration, "note duration is not finite")?;
        let signal_duration = finite(signal_duration, "note signal duration is not finite")?;
        if attack_duration < 0.0 || note_duration < 0.0 || signal_duration < 0.0 {
            return Err(AraError::InvalidArgument(
                "note durations must be nonnegative",
            ));
        }
        Ok(Self {
            frequency,
            pitch_number,
            volume,
            start_position,
            attack_duration,
            note_duration,
            signal_duration,
        })
    }

    /// Returns the optional average frequency in hertz.
    pub fn frequency(&self) -> Option<f32> {
        self.frequency
    }

    /// Returns the quantized pitch number or the ARA invalid-pitch sentinel.
    pub fn pitch_number(&self) -> i32 {
        self.pitch_number.unwrap_or(i32::MIN)
    }

    /// Returns the quantized pitch when this is a pitched note.
    pub fn pitch(&self) -> Option<i32> {
        self.pitch_number
    }

    /// Returns the nonnegative perceptual volume.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Returns the start position in seconds.
    pub fn start_position(&self) -> f64 {
        self.start_position
    }

    /// Returns the attack duration in seconds.
    pub fn attack_duration(&self) -> f64 {
        self.attack_duration
    }

    /// Returns the note duration in seconds.
    pub fn note_duration(&self) -> f64 {
        self.note_duration
    }

    /// Returns the full signal duration in seconds.
    pub fn signal_duration(&self) -> f64 {
        self.signal_duration
    }
}

/// A static twelve-tone tuning.
#[derive(Clone, Debug, PartialEq)]
pub struct TuningEvent {
    concert_pitch_frequency: f32,
    root: i32,
    tunings: [f32; 12],
    name: Option<String>,
}

impl TuningEvent {
    /// Creates a validated tuning with optional copied display name.
    pub fn new(
        concert_pitch_frequency: f32,
        root: i32,
        tunings: [f32; 12],
        name: Option<String>,
    ) -> Result<Self, AraError> {
        let concert_pitch_frequency = finite_f32(
            concert_pitch_frequency,
            "concert pitch frequency is not finite",
        )?;
        if concert_pitch_frequency <= 0.0 {
            return Err(AraError::InvalidArgument(
                "concert pitch frequency must be positive",
            ));
        }
        for tuning in tunings {
            finite_f32(tuning, "tuning offset is not finite")?;
        }
        Ok(Self {
            concert_pitch_frequency,
            root,
            tunings,
            name: display_name(name)?,
        })
    }

    /// Returns the concert-pitch frequency.
    pub fn concert_pitch_frequency(&self) -> f32 {
        self.concert_pitch_frequency
    }

    /// Returns the root circle-of-fifths index.
    pub fn root(&self) -> i32 {
        self.root
    }

    /// Returns the chromatic cent offsets.
    pub fn tunings(&self) -> &[f32; 12] {
        &self.tunings
    }

    /// Returns the optional copied display name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// A key-signature interval usage byte, including future values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeySignatureIntervalUsage(u8);

impl KeySignatureIntervalUsage {
    /// The interval is unused.
    pub const UNUSED: Self = Self(0x00);
    /// The interval is used.
    pub const USED: Self = Self(0xFF);

    /// Preserves an ARA usage byte, including future values.
    pub const fn from_raw(raw: u8) -> Self {
        Self(raw)
    }

    /// Returns the preserved usage byte.
    pub const fn as_raw(self) -> u8 {
        self.0
    }
}

/// A key-signature change.
#[derive(Clone, Debug, PartialEq)]
pub struct KeySignatureEvent {
    root: i32,
    intervals: [KeySignatureIntervalUsage; 12],
    name: Option<String>,
    position: f64,
}

impl KeySignatureEvent {
    /// Creates a key signature at a finite quarter-note position.
    pub fn new(
        root: i32,
        intervals: [KeySignatureIntervalUsage; 12],
        name: Option<String>,
        position: f64,
    ) -> Result<Self, AraError> {
        Ok(Self {
            root,
            intervals,
            name: display_name(name)?,
            position: finite(position, "key-signature position is not finite")?,
        })
    }

    /// Returns the root circle-of-fifths index.
    pub fn root(&self) -> i32 {
        self.root
    }

    /// Returns the twelve chromatic interval usages.
    pub fn intervals(&self) -> &[KeySignatureIntervalUsage; 12] {
        &self.intervals
    }

    /// Returns the optional copied display name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the position in quarter notes.
    pub fn position(&self) -> f64 {
        self.position
    }
}

/// A chord interval usage byte.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChordIntervalUsage(u8);

impl ChordIntervalUsage {
    /// The interval is unused.
    pub const UNUSED: Self = Self(0x00);
    /// The interval is used with unknown diatonic function.
    pub const USED: Self = Self(0xFF);

    /// Validates an ARA chord usage byte.
    pub fn from_raw(raw: u8) -> Result<Self, AraError> {
        matches!(raw, 0 | 1..=7 | 9 | 11 | 13 | 0xFF)
            .then_some(Self(raw))
            .ok_or(AraError::InvalidArgument("invalid chord interval usage"))
    }

    /// Returns the usage byte.
    pub const fn as_raw(self) -> u8 {
        self.0
    }
}

/// A lead-sheet chord change.
#[derive(Clone, Debug, PartialEq)]
pub struct ChordEvent {
    root: i32,
    bass: i32,
    intervals: [ChordIntervalUsage; 12],
    name: Option<String>,
    position: f64,
}

impl ChordEvent {
    /// Creates a chord at a finite quarter-note position.
    pub fn new(
        root: i32,
        bass: i32,
        intervals: [ChordIntervalUsage; 12],
        name: Option<String>,
        position: f64,
    ) -> Result<Self, AraError> {
        Ok(Self {
            root,
            bass,
            intervals,
            name: display_name(name)?,
            position: finite(position, "chord position is not finite")?,
        })
    }

    /// Returns the root circle-of-fifths index.
    pub fn root(&self) -> i32 {
        self.root
    }

    /// Returns the bass circle-of-fifths index.
    pub fn bass(&self) -> i32 {
        self.bass
    }

    /// Returns the twelve chord interval usages.
    pub fn intervals(&self) -> &[ChordIntervalUsage; 12] {
        &self.intervals
    }

    /// Returns the optional copied display name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the position in quarter notes.
    pub fn position(&self) -> f64 {
        self.position
    }
}

/// A content grade, including future values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContentGrade(i32);

impl ContentGrade {
    /// Placeholder or unavailable content.
    pub const INITIAL: Self = Self(0);
    /// Automatically detected content.
    pub const DETECTED: Self = Self(1);
    /// User-reviewed or adjusted content.
    pub const ADJUSTED: Self = Self(2);
    /// Explicitly approved content.
    pub const APPROVED: Self = Self(3);

    /// Preserves a raw grade, including future values.
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    /// Returns the preserved raw grade.
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

bitflags! {
    /// Content scopes known to be unchanged, retaining future flag bits.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ContentUpdateScopes: i32 {
        /// The signal scope remains unchanged.
        const SIGNAL_REMAINS_UNCHANGED = 1 << 0;
        /// The note scope remains unchanged.
        const NOTE_REMAINS_UNCHANGED = 1 << 1;
        /// The timing scope remains unchanged.
        const TIMING_REMAINS_UNCHANGED = 1 << 2;
        /// The tuning scope remains unchanged.
        const TUNING_REMAINS_UNCHANGED = 1 << 3;
        /// The harmonic scope remains unchanged.
        const HARMONIC_REMAINS_UNCHANGED = 1 << 4;
    }
}
