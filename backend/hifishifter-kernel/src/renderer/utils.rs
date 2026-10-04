//! 渲染器共享辅助函数。
//!
//! 供 [`WorldRenderer`] 和 [`HiFiGanRenderer`] 内部使用，
//! 避免在多个渲染器中重复实现 MIDI 曲线插值逻辑。

/// 在 pitch_edit 曲线中插值，返回目标 MIDI（无编辑时返回 None）。
pub fn edit_midi_at_time_or_none(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    abs_time_sec: f64,
) -> Option<f64> {
    if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
        return None;
    }
    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_time_sec * 1000.0) / fp;
    if !(idx_f.is_finite() && idx_f >= 0.0) {
        return None;
    }
    let i0 = idx_f.floor() as isize;
    if i0 < 0 {
        return None;
    }
    let i0 = i0 as usize;
    if i0 >= pitch_edit.len() {
        return None;
    }
    let i1 = (i0 + 1).min(pitch_edit.len().saturating_sub(1));
    let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

    let e0 = pitch_edit.get(i0).copied().unwrap_or(0.0) as f64;
    let e1 = pitch_edit.get(i1).copied().unwrap_or(0.0) as f64;

    let e0 = if e0.is_finite() && e0 > 0.0 {
        Some(e0)
    } else {
        None
    };
    let e1 = if e1.is_finite() && e1 > 0.0 {
        Some(e1)
    } else {
        None
    };

    match (e0, e1) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(a), Some(b)) => {
            let v = a + (b - a) * frac;
            if v.is_finite() && v > 0.0 {
                Some(v)
            } else {
                None
            }
        }
    }
}

/// 在 clip_midi 曲线中插值，返回原始 MIDI（无效时返回 0.0）。
pub fn clip_midi_at_time(
    frame_period_ms: f64,
    clip_start_sec: f64,
    clip_midi: &[f32],
    abs_time_sec: f64,
) -> f64 {
    if !(abs_time_sec.is_finite() && abs_time_sec >= clip_start_sec) {
        return 0.0;
    }
    let local_sec = abs_time_sec - clip_start_sec;
    let fp = frame_period_ms.max(0.1);
    let idx_f = (local_sec * 1000.0) / fp;
    if !(idx_f.is_finite() && idx_f >= 0.0) {
        return 0.0;
    }
    let i0 = idx_f.floor() as isize;
    if i0 < 0 {
        return 0.0;
    }
    let i0 = i0 as usize;
    if i0 >= clip_midi.len() {
        return 0.0;
    }
    let i1 = (i0 + 1).min(clip_midi.len().saturating_sub(1));
    let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

    let a = clip_midi.get(i0).copied().unwrap_or(0.0) as f64;
    let b = clip_midi.get(i1).copied().unwrap_or(0.0) as f64;

    let mut a = if a.is_finite() && a > 0.0 { a } else { 0.0 };
    let mut b = if b.is_finite() && b > 0.0 { b } else { 0.0 };
    if a <= 0.0 && b > 0.0 {
        a = b;
    }
    if b <= 0.0 && a > 0.0 {
        b = a;
    }
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }

    let v = a + (b - a) * frac;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}
