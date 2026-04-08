#[allow(dead_code)]
pub(crate) fn beat_to_sec(beat: f64, bpm: f64) -> f64 {
    let bpm = if bpm.is_finite() && bpm > 0.0 {
        bpm
    } else {
        120.0
    };
    beat * 60.0 / bpm
}

pub(crate) fn clamp11(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

pub(crate) fn quantize_i64(x: f64, scale: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    (x * scale).round() as i64
}

pub(crate) fn quantize_u32(x: f64, scale: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    let v = (x * scale).round();
    if v <= 0.0 {
        0
    } else if v > (u32::MAX as f64) {
        u32::MAX
    } else {
        v as u32
    }
}
