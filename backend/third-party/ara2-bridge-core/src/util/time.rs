//! Sample/time rounding and range helpers.

use crate::{AraError, ContentTimeRange};

/// Converts seconds to an ARA sample position using `floor(x + 0.5)` rounding.
pub fn time_to_sample(time: f64, sample_rate: f64) -> Result<i64, AraError> {
    if !time.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AraError::InvalidArgument(
            "time must be finite and sample rate positive",
        ));
    }
    let continuous = time * sample_rate;
    let rounded = (continuous + 0.5).floor();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
        return Err(AraError::InvalidArgument(
            "sample position is outside i64 range",
        ));
    }
    Ok(rounded as i64)
}

/// Converts a discrete sample position to seconds.
pub fn sample_to_time(sample: i64, sample_rate: f64) -> Result<f64, AraError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AraError::InvalidArgument(
            "sample rate must be finite and positive",
        ));
    }
    Ok(sample as f64 / sample_rate)
}

/// Intersects two finite half-open content ranges.
pub fn intersect_content_ranges(
    left: ContentTimeRange,
    right: ContentTimeRange,
) -> Result<Option<ContentTimeRange>, AraError> {
    let start = left.start().max(right.start());
    let left_end = left
        .start()
        .checked_add(left.duration())
        .ok_or(AraError::InvalidArgument("content range end is not finite"))?;
    let right_end = right
        .start()
        .checked_add(right.duration())
        .ok_or(AraError::InvalidArgument("content range end is not finite"))?;
    let end = left_end.min(right_end);
    if end <= start {
        return Ok(None);
    }
    ContentTimeRange::new(start, end - start).map(Some)
}

trait CheckedFloatAdd {
    fn checked_add(self, other: Self) -> Option<Self>
    where
        Self: Sized;
}

impl CheckedFloatAdd for f64 {
    fn checked_add(self, other: Self) -> Option<Self> {
        let value = self + other;
        value.is_finite().then_some(value)
    }
}
