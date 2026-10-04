//! Chord and key-signature naming ported from the ARA SDK library.

pub use super::pitch::PitchInterpreter;
use crate::{ChordEvent, KeySignatureEvent};

/// Recognized diatonic scale mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScaleMode {
    /// No standard diatonic mode matches.
    Invalid,
    /// Major scale.
    Ionian,
    /// Dorian mode.
    Dorian,
    /// Phrygian mode.
    Phrygian,
    /// Lydian mode.
    Lydian,
    /// Mixolydian mode.
    Mixolydian,
    /// Natural minor scale.
    Aeolian,
    /// Locrian mode.
    Locrian,
}

impl PitchInterpreter {
    /// Returns whether all chord intervals are unused.
    pub fn is_no_chord(chord: &ChordEvent) -> bool {
        chord.intervals().iter().all(|usage| usage.as_raw() == 0)
    }

    /// Generates the upstream-style chord name from abstract interval data.
    pub fn chord_name(&self, chord: &ChordEvent) -> String {
        if Self::is_no_chord(chord) {
            return "N.C.".to_owned();
        }
        let mut result = self.note_name(chord.root());
        if chord.intervals()[1..]
            .iter()
            .all(|usage| usage.as_raw() == 0)
            && chord.root() == chord.bass()
        {
            result.push_str(" bass");
            return result;
        }
        let mut intervals = chord.intervals().map(|usage| usage.as_raw());
        let mut altered = String::new();
        let mut added = String::new();
        let mut omitted = String::new();
        let mut power = true;
        for (index, usage) in intervals.iter().copied().enumerate().take(10).skip(1) {
            if index != 7 && usage != 0 && (usage == 0xFF || usage < 7) {
                power = false;
                break;
            }
        }
        if !test_clear(&mut intervals, 0, None) {
            append_interval(&mut omitted, Some("omit"), None, "1");
            power = false;
        }
        power &= cleanup_degree(&mut intervals, 1);

        let third = if test_clear(&mut intervals, 4, Some(3)) {
            4
        } else if test_clear(&mut intervals, 3, Some(3)) {
            3
        } else if test_clear(&mut intervals, 5, Some(4)) || test_clear(&mut intervals, 5, Some(3)) {
            5
        } else if test_clear(&mut intervals, 2, Some(2)) || test_clear(&mut intervals, 2, Some(3)) {
            2
        } else if test_clear(&mut intervals, 4, Some(0xFF)) {
            4
        } else if test_clear(&mut intervals, 3, Some(0xFF)) {
            3
        } else if intervals[7] != 0 && test_clear(&mut intervals, 5, Some(0xFF)) {
            5
        } else if intervals[7] != 0 && test_clear(&mut intervals, 2, Some(0xFF)) {
            2
        } else {
            append_interval(&mut omitted, Some("omit"), None, "3");
            0
        };
        power &= third == 0;
        power &= cleanup_degree(&mut intervals, 2);
        power &= cleanup_degree(&mut intervals, 3);
        power &= cleanup_degree(&mut intervals, 4);

        let fifth = if test_clear(&mut intervals, 7, None) {
            7
        } else if test_clear(&mut intervals, 6, Some(5)) {
            6
        } else if test_clear(&mut intervals, 8, Some(5)) {
            8
        } else if test_clear(&mut intervals, 6, Some(0xFF)) {
            6
        } else if test_clear(&mut intervals, 8, Some(0xFF)) {
            8
        } else {
            append_interval(&mut omitted, Some("omit"), None, "5");
            0
        };
        power &= fifth == 7;
        power &= cleanup_degree(&mut intervals, 5);

        let mut seventh = if test_clear(&mut intervals, 11, None) {
            11
        } else if test_clear(&mut intervals, 10, None) {
            10
        } else if test_clear(&mut intervals, 9, Some(7)) {
            9
        } else {
            0
        };
        if test_clear(&mut intervals, 9, Some(6)) {
            if seventh != 0 {
                append_interval(&mut added, Some("add"), None, "6");
            } else {
                seventh = 9;
            }
        } else if test_clear(&mut intervals, 8, Some(6)) {
            if seventh != 0 {
                append_interval(&mut added, Some("add"), Some(self.flat_symbol()), "6");
            } else {
                seventh = 8;
            }
        }
        if seventh == 0 {
            if test_clear(&mut intervals, 8, Some(0xFF)) {
                seventh = 8;
            } else if test_clear(&mut intervals, 9, Some(0xFF)) {
                seventh = 9;
            }
        }
        power &= seventh == 0 || seventh >= 10;
        power &= cleanup_degree(&mut intervals, 6);
        cleanup_degree(&mut intervals, 7);

        let mut pending4 = false;
        let mut pending7 = seventh >= 10;
        let mut implied_flat5 = false;
        let mut implied_sharp5 = false;
        if power {
            result.push('5');
        } else if third == 4 {
            if fifth == 8 {
                result.push('+');
                implied_sharp5 = true;
            }
        } else if third == 3 {
            if fifth == 6 && seventh == 10 {
                result.push_str(if self.ascii_symbols { "halfdim" } else { "ø" });
                pending7 = false;
                implied_flat5 = true;
            } else if fifth == 6 && seventh != 10 {
                result.push_str(if self.ascii_symbols { "dim" } else { "°" });
                implied_flat5 = true;
            } else {
                result.push('m');
            }
        } else if third == 2 {
            result.push_str("sus2");
        } else if third == 5 {
            result.push_str("sus");
            pending4 = seventh == 0;
        }
        if seventh == 11 {
            result.push_str(if self.ascii_symbols { "maj" } else { "∆" });
        } else if seventh == 8 {
            append_interval(&mut result, None, Some(self.flat_symbol()), "6");
        } else if seventh == 9 && (third != 3 || fifth != 6) {
            append_interval(&mut result, None, None, "6");
        }
        if fifth == 6 && !implied_flat5 {
            append_interval(&mut altered, None, Some(self.flat_symbol()), "5");
        }
        if fifth == 8 && !implied_sharp5 {
            append_interval(&mut altered, None, Some(self.sharp_symbol()), "5");
        }

        let mut add9 = seventh == 0;
        let mut pending9 = 0;
        for (index, sign, value) in [
            (2, None, 14),
            (1, Some(self.flat_symbol()), 13),
            (3, Some(self.sharp_symbol()), 15),
        ] {
            if intervals[index] != 0 {
                if add9 {
                    append_interval(&mut added, Some("add"), sign, "9");
                } else {
                    pending9 = value;
                }
                add9 = true;
            }
        }
        if pending9 != 0 {
            pending7 = false;
        }
        let mut add11 = pending9 == 0;
        let mut pending11 = 0;
        for (index, sign, value) in [
            (5, None, 17),
            (4, Some(self.flat_symbol()), 16),
            (6, Some(self.sharp_symbol()), 18),
        ] {
            if intervals[index] != 0 {
                if add11 {
                    append_interval(&mut added, Some("add"), sign, "11");
                } else {
                    pending11 = value;
                }
                add11 = true;
            }
        }
        let mut add13 = pending9 == 0;
        let mut non_add13 = false;
        for (index, sign) in [
            (9, None),
            (8, Some(self.flat_symbol())),
            (10, Some(self.sharp_symbol())),
        ] {
            if intervals[index] != 0 {
                if add13 {
                    append_interval(&mut added, Some("add"), sign, "13");
                } else {
                    append_interval(&mut result, None, sign, "13");
                    non_add13 = true;
                }
                add13 = true;
            }
        }
        if pending11 != 0 || non_add13 {
            if pending9 == 13 {
                append_interval(&mut altered, None, Some(self.flat_symbol()), "9");
            } else if pending9 == 15 {
                append_interval(&mut altered, None, Some(self.sharp_symbol()), "9");
            }
            pending9 = 0;
        }
        if pending4 {
            result.push('4');
        }
        if pending7 {
            append_interval(&mut result, None, None, "7");
        }
        if pending9 != 0 {
            let sign = match pending9 {
                13 => Some(self.flat_symbol()),
                15 => Some(self.sharp_symbol()),
                _ => None,
            };
            append_interval(&mut result, None, sign, "9");
        }
        if pending11 != 0 {
            let sign = match pending11 {
                16 => Some(self.flat_symbol()),
                18 => Some(self.sharp_symbol()),
                _ => None,
            };
            if non_add13 {
                append_interval(&mut added, Some("add"), sign, "11");
            } else {
                append_interval(&mut result, None, sign, "11");
            }
        }
        if !power {
            result.push_str(&altered);
        }
        result.push_str(&added);
        if !power {
            result.push_str(&omitted);
        }
        if chord.root() != chord.bass() {
            result.push('/');
            result.push_str(&self.note_name(chord.bass()));
        }
        result
    }

    /// Identifies the standard diatonic mode represented by a key signature.
    pub fn scale_mode(key: &KeySignatureEvent) -> ScaleMode {
        let mut input = 0_u16;
        for interval in key.intervals() {
            rotate_scale(&mut input);
            if interval.as_raw() != 0 {
                input += 1;
            }
        }
        let mut current = 0xAD5_u16;
        let modes = [
            ScaleMode::Ionian,
            ScaleMode::Dorian,
            ScaleMode::Phrygian,
            ScaleMode::Lydian,
            ScaleMode::Mixolydian,
            ScaleMode::Aeolian,
            ScaleMode::Locrian,
        ];
        for mode in modes {
            if input == current {
                return mode;
            }
            rotate_scale(&mut current);
            if current & (1 << 11) == 0 {
                rotate_scale(&mut current);
            }
        }
        ScaleMode::Invalid
    }

    /// Generates a name for a recognized standard key signature.
    pub fn key_name(&self, key: &KeySignatureEvent) -> Option<String> {
        let note = self.note_name(key.root());
        match Self::scale_mode(key) {
            ScaleMode::Invalid => None,
            ScaleMode::Ionian => Some(note),
            ScaleMode::Dorian => Some(format!("{note} Dorian")),
            ScaleMode::Phrygian => Some(format!("{note} Phrygian")),
            ScaleMode::Lydian => Some(format!("{note} Lydian")),
            ScaleMode::Mixolydian => Some(format!("{note} Mixolydian")),
            ScaleMode::Aeolian => Some(format!("{note}m")),
            ScaleMode::Locrian => Some(format!("{note} Locrian")),
        }
    }
}

fn test_clear(intervals: &mut [u8; 12], index: usize, usage: Option<u8>) -> bool {
    let matches = usage.map_or(intervals[index] != 0, |usage| intervals[index] == usage);
    if matches {
        intervals[index] = 0;
    }
    matches
}

fn cleanup_degree(intervals: &mut [u8; 12], degree: u8) -> bool {
    let mut clean = true;
    for usage in intervals {
        if *usage == degree {
            *usage = 0xFF;
            clean = false;
        }
    }
    clean
}

fn append_interval(text: &mut String, prefix: Option<&str>, sign: Option<&str>, interval: &str) {
    if prefix.is_none()
        && sign.is_none()
        && text
            .chars()
            .last()
            .is_some_and(|character| character.is_ascii_digit())
    {
        text.push('/');
    }
    if let Some(prefix) = prefix {
        text.push_str(prefix);
    }
    if let Some(sign) = sign {
        text.push_str(sign);
    }
    text.push_str(interval);
}

fn rotate_scale(scale: &mut u16) {
    *scale <<= 1;
    if *scale >= 1 << 12 {
        *scale -= (1 << 12) - 1;
    }
}
