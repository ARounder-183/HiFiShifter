//! Circle-of-fifths spelling configuration.

/// Configurable ARA pitch, chord, and key-signature interpreter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PitchInterpreter {
    pub(super) ascii_symbols: bool,
    pub(super) german_note_names: bool,
}

impl PitchInterpreter {
    /// Creates an interpreter with ASCII-symbol and German-name options.
    pub const fn new(ascii_symbols: bool, german_note_names: bool) -> Self {
        Self {
            ascii_symbols,
            german_note_names,
        }
    }

    /// Returns the note name for an ARA circle-of-fifths index.
    pub fn note_name(&self, mut index: i32) -> String {
        const NAMES: [char; 7] = ['F', 'C', 'G', 'D', 'A', 'E', 'B'];
        let mut accidentals = 0_usize;
        let note;
        let accidental;
        if index < 0 {
            while index < -1 {
                accidentals += 1;
                index += 7;
            }
            note = NAMES[(index + 1) as usize];
            if self.german_note_names && note == 'B' {
                accidentals = accidentals.saturating_sub(1);
            }
            accidental = self.flat_symbol();
        } else {
            while index > 5 {
                accidentals += 1;
                index -= 7;
            }
            let english = NAMES[(index + 1) as usize];
            note = if self.german_note_names && english == 'B' {
                'H'
            } else {
                english
            };
            accidental = self.sharp_symbol();
        }
        let mut result = String::with_capacity(1 + accidentals * accidental.len());
        result.push(note);
        for _ in 0..accidentals {
            result.push_str(accidental);
        }
        result
    }

    pub(super) const fn flat_symbol(&self) -> &'static str {
        if self.ascii_symbols {
            "b"
        } else {
            "♭"
        }
    }

    pub(super) const fn sharp_symbol(&self) -> &'static str {
        if self.ascii_symbols {
            "#"
        } else {
            "♯"
        }
    }
}
