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

/// 一段源 F0 曲线（`clip_midi`）是否**可用于声码器推理**。
///
/// 【为什么需要这条判据，而不是只判空】NSF-HiFiGAN 是 **F0 驱动的源激励**模型：
/// 有激励才出声。一段"非空但全 0"的曲线与"空"曲线对声码器是**同一个语义** ——
/// 没有可用的激励。此前声码器入口只判 `is_empty()`，于是全 0 的曲线会穿过去，
/// 渲出静音（无分离时）或只剩噪声支（有分离时，见 `chain::process_breath`）。
///
/// 【为什么"全 0"在别处是合法输入，这里却判不可用】真实分析对**整段无声/纯清辅音**
/// 的素材确实可能给出全 0 —— 但那类片段本来就该原样通过（声码器在无激励下也只能
/// 给出静音），返回未处理 PCM 既正确又更省算力。真正危险的是**推导失败**被伪装成
/// 全 0（工程域根轨道数组被宿主编辑清空后按绝对时间采样就是这个形状），
/// 这条判据让两者都落到同一条安全路径。
pub fn pitch_curve_is_renderable(curve: &[f32]) -> bool {
    curve.iter().any(|value| value.is_finite() && *value > 0.0)
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

#[cfg(test)]
mod tests {
    use super::pitch_curve_is_renderable;

    /// 空曲线与"非空但全 0"的曲线对 F0 驱动的声码器是**同一个语义**。
    ///
    /// 【为什么这条判据是回归测试】真机报障（Cubase ARA）：分割 clip 之后大多数片段
    /// 只渲出气声。根因是 F0 源被清空后按绝对时间采样得到一段**非空但全 0** 的曲线，
    /// 它绕过了声码器入口的 `is_empty()` 守卫 —— 谐波支因此渲成静音，而气声分离的
    /// 噪声支按单位增益照常混回。判据必须把两者一起挡住。
    #[test]
    fn an_empty_curve_and_an_all_zero_curve_are_both_unusable() {
        assert!(!pitch_curve_is_renderable(&[]));
        assert!(!pitch_curve_is_renderable(&[0.0; 64]));
    }

    /// 只要有一帧可用的 F0 就算可用（无声前导段是合法输入）。
    #[test]
    fn a_single_voiced_frame_makes_the_curve_usable() {
        let mut curve = vec![0.0; 64];
        curve[40] = 57.0;
        assert!(pitch_curve_is_renderable(&curve));
    }

    /// 非有限值不构成"可用"：NaN 会把噪声传播进 STFT，负音高无意义。
    #[test]
    fn non_finite_or_negative_values_are_not_usable() {
        assert!(!pitch_curve_is_renderable(&[f32::NAN; 8]));
        assert!(!pitch_curve_is_renderable(&[f32::INFINITY; 8]));
        assert!(!pitch_curve_is_renderable(&[-3.0; 8]));
        assert!(!pitch_curve_is_renderable(&[0.0, f32::NAN, -1.0]));
    }
}
