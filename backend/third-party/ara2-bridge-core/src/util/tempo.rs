//! Tempo and bar-signature mapping.

use crate::{AraError, BarSignatureEvent, TempoEvent};

/// Validated piecewise-linear tempo map with endpoint extrapolation.
#[derive(Clone, Debug)]
pub struct TempoMap {
    entries: Vec<TempoEvent>,
}

impl TempoMap {
    /// Creates a map from at least two strictly increasing time/quarter pairs.
    pub fn new(entries: Vec<TempoEvent>) -> Result<Self, AraError> {
        if entries.len() < 2 {
            return Err(AraError::InvalidArgument(
                "tempo map requires at least two entries",
            ));
        }
        if entries.windows(2).any(|pair| {
            pair[0].time_position() >= pair[1].time_position()
                || pair[0].quarter_position() >= pair[1].quarter_position()
        }) {
            return Err(AraError::InvalidArgument(
                "tempo entries must increase in time and quarters",
            ));
        }
        Ok(Self { entries })
    }

    /// Converts seconds to quarter notes.
    pub fn quarter_at_time(&self, time: f64) -> Result<f64, AraError> {
        if !time.is_finite() {
            return Err(AraError::InvalidArgument("time is not finite"));
        }
        let (left, right) = self.segment(time, TempoEvent::time_position);
        let slope = (right.quarter_position() - left.quarter_position())
            / (right.time_position() - left.time_position());
        Ok(left.quarter_position() + (time - left.time_position()) * slope)
    }

    /// Converts quarter notes to seconds.
    pub fn time_at_quarter(&self, quarter: f64) -> Result<f64, AraError> {
        if !quarter.is_finite() {
            return Err(AraError::InvalidArgument("quarter position is not finite"));
        }
        let (left, right) = self.segment(quarter, TempoEvent::quarter_position);
        let slope = (right.time_position() - left.time_position())
            / (right.quarter_position() - left.quarter_position());
        Ok(left.time_position() + (quarter - left.quarter_position()) * slope)
    }

    fn segment(&self, position: f64, field: fn(&TempoEvent) -> f64) -> (&TempoEvent, &TempoEvent) {
        let upper = self
            .entries
            .partition_point(|entry| field(entry) <= position);
        let right = upper.clamp(1, self.entries.len() - 1);
        (&self.entries[right - 1], &self.entries[right])
    }
}

/// Validated bar-signature map using ARA beat and bar conventions.
#[derive(Clone, Debug)]
pub struct BarMap {
    entries: Vec<BarSignatureEvent>,
}

impl BarMap {
    /// Creates a nonempty map whose changes occur on whole bar boundaries.
    pub fn new(entries: Vec<BarSignatureEvent>) -> Result<Self, AraError> {
        if entries.is_empty() {
            return Err(AraError::InvalidArgument(
                "bar map requires at least one entry",
            ));
        }
        for pair in entries.windows(2) {
            if pair[0].position() >= pair[1].position() {
                return Err(AraError::InvalidArgument(
                    "bar signatures must increase by quarter position",
                ));
            }
            let bars = (pair[1].position() - pair[0].position()) / quarters_per_bar(&pair[0]);
            if (bars - bars.round()).abs() > 1.0e-9 {
                return Err(AraError::InvalidArgument(
                    "bar-signature change is not on a bar boundary",
                ));
            }
        }
        Ok(Self { entries })
    }

    /// Converts quarter notes to beats.
    pub fn beat_at_quarter(&self, quarter: f64) -> Result<f64, AraError> {
        let (index, start_beat) = self.entry_for_quarter(quarter)?;
        Ok(start_beat
            + (quarter - self.entries[index].position()) * beats_per_quarter(&self.entries[index]))
    }

    /// Converts beats to quarter notes.
    pub fn quarter_at_beat(&self, beat: f64) -> Result<f64, AraError> {
        let (index, start_beat) = self.entry_for_beat(beat)?;
        Ok(self.entries[index].position()
            + (beat - start_beat) / beats_per_quarter(&self.entries[index]))
    }

    /// Returns the zero-based bar index using the upstream rounding convention.
    pub fn bar_index_at_quarter(&self, quarter: f64) -> Result<i32, AraError> {
        let (index, _) = self.entry_for_quarter(quarter)?;
        let mut bars = ((quarter - self.entries[index].position())
            / quarters_per_bar(&self.entries[index]))
        .floor();
        for current in (0..index).rev() {
            bars += (self.entries[current + 1].position() - self.entries[current].position())
                / quarters_per_bar(&self.entries[current]);
        }
        let rounded = bars + 0.5;
        if rounded < i32::MIN as f64 || rounded >= i32::MAX as f64 {
            return Err(AraError::InvalidArgument("bar index is outside i32 range"));
        }
        Ok(rounded as i32)
    }

    /// Returns the quarter position at the start of a bar index.
    pub fn quarter_at_bar_index(&self, bar_index: i32) -> Result<f64, AraError> {
        let mut start_bar = 0_i32;
        for (index, pair) in self.entries.windows(2).enumerate() {
            let distance = (pair[1].position() - pair[0].position()) / quarters_per_bar(&pair[0]);
            let count = round_i32(distance)?;
            let next = start_bar
                .checked_add(count)
                .ok_or(AraError::InvalidArgument("bar index overflow"))?;
            if next > bar_index {
                return Ok(pair[0].position()
                    + f64::from(bar_index - start_bar) * quarters_per_bar(&pair[0]));
            }
            start_bar = next;
            if index + 2 == self.entries.len() {
                let entry = &self.entries[index + 1];
                return Ok(
                    entry.position() + f64::from(bar_index - start_bar) * quarters_per_bar(entry)
                );
            }
        }
        let entry = &self.entries[0];
        Ok(entry.position() + f64::from(bar_index) * quarters_per_bar(entry))
    }

    fn entry_for_quarter(&self, quarter: f64) -> Result<(usize, f64), AraError> {
        if !quarter.is_finite() {
            return Err(AraError::InvalidArgument("quarter position is not finite"));
        }
        let upper = self
            .entries
            .partition_point(|entry| entry.position() <= quarter);
        let index = upper.saturating_sub(1);
        Ok((index, self.start_beat(index)))
    }

    fn entry_for_beat(&self, beat: f64) -> Result<(usize, f64), AraError> {
        if !beat.is_finite() {
            return Err(AraError::InvalidArgument("beat position is not finite"));
        }
        let mut start = 0.0;
        for index in 0..self.entries.len().saturating_sub(1) {
            let next = start
                + (self.entries[index + 1].position() - self.entries[index].position())
                    * beats_per_quarter(&self.entries[index]);
            if beat < next {
                return Ok((index, start));
            }
            start = next.round();
        }
        Ok((self.entries.len() - 1, start))
    }

    fn start_beat(&self, index: usize) -> f64 {
        (0..index)
            .map(|current| {
                (self.entries[current + 1].position() - self.entries[current].position())
                    * beats_per_quarter(&self.entries[current])
            })
            .sum::<f64>()
            .round()
    }
}

fn beats_per_quarter(signature: &BarSignatureEvent) -> f64 {
    f64::from(signature.denominator()) / 4.0
}

fn quarters_per_bar(signature: &BarSignatureEvent) -> f64 {
    f64::from(signature.numerator()) / beats_per_quarter(signature)
}

fn round_i32(value: f64) -> Result<i32, AraError> {
    let rounded = (value + 0.5).floor();
    if rounded < i32::MIN as f64 || rounded >= i32::MAX as f64 {
        return Err(AraError::InvalidArgument(
            "rounded value is outside i32 range",
        ));
    }
    Ok(rounded as i32)
}
