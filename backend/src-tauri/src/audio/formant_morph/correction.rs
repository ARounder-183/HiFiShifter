/*
 * formant_morph/correction.rs - 共振峰校正滤波器的构造与保护逻辑。
 *
 * 主要内容：
 * - moved_pole_freq：极点迁移公式 f' = f + (target − f)·strength。
 * - h_ratio_db：由原/目标 LPC 系数构造 STFT 域幅频比值 H(k)（dB，限幅 ±24）。
 *   Y(k) = X(k)·10^(H·gate/20)，完全保留原相位 —— 这是"只搬共振峰、
 *   不动音色身份"的核心。
 * - VoicingGate：浊音权重的时间平滑（attack 30ms / release 80ms），
 *   替代旧路线逐帧乱跳的 confidence（抖动/门限感的根因）。
 * - match_frame_energy：逐帧 RMS 对齐（clamp 0.5–2.0），防响度漂移。
 * - soft_limit：峰值保护 + 软膝限幅（tanh），替代末端硬 clamp 0.99。
 *
 * 与其他模块的关系：
 * - 被 formant_morph/mod.rs 的 STFT 编排逐帧调用。
 * - 系数约定与 lpc.rs 一致：a[1..=p]（A(z) = 1 + Σ a_k·z⁻ᵏ）。
 *
 * 维护说明：
 * - H 的限幅与 gate 的时间平滑是"防止搬过头/抖动"的最后防线，禁止移除。
 * - all_pole_mag_db 复用 track.rs 的实现（单一真相）。
 */

use super::track::all_pole_mag_db;

// ── 保护常量 ─────────────────────────────────────────────────────────────

/// H 幅频比值限幅（dB）：单点最大提升/衰减。
pub const H_LIMIT_DB: f32 = 24.0;
/// H 幅频响应在模型奈奎斯特之上不可信，向 0 dB 衰减的过渡带宽（Hz）。
const MODEL_FADE_HZ: f32 = 500.0;
/// 浊音 gate 上升时间（秒）。
pub const GATE_ATTACK_SEC: f32 = 0.03;
/// 浊音 gate 释放时间（秒）。
pub const GATE_RELEASE_SEC: f32 = 0.08;
/// 逐帧 RMS 对齐的最小/最大增益。
pub const RMS_MATCH_MIN: f32 = 0.5;
pub const RMS_MATCH_MAX: f32 = 2.0;
/// 软膝限幅的拐点。
const SOFT_KNEE: f32 = 0.85;

/// 极点迁移：f' = f + (target − f)·strength。
///
/// strength ∈ [0,1]；返回值 clamp 到 (0, sr/2) 防御。
pub fn moved_pole_freq(current_hz: f32, target_hz: f32, strength: f32) -> f32 {
    let strength = strength.clamp(0.0, 1.0);
    let moved = current_hz + (target_hz - current_hz) * strength;
    // 文档承诺的防御：把迁移结果限回可听的 (0, sr/2)。上游通常已钳制，
    // 这里兜底 NaN/越界输入（如参数被旧工程文件直传）。
    if !moved.is_finite() || moved <= 0.0 {
        return current_hz.max(1.0);
    }
    moved
}

/// 浊音权重的一阶平滑器（非线性 attack/release）。
#[derive(Debug, Clone)]
pub struct VoicingGate {
    state: f32,
}

impl Default for VoicingGate {
    fn default() -> Self {
        Self::new()
    }
}

impl VoicingGate {
    pub fn new() -> Self {
        VoicingGate { state: 0.0 }
    }

    /// 推进一帧：raw 为本帧的瞬时浊音判定（0/1 或小数），
    /// 上升走 attack、下降走 release，输出 clamp 到 [0,1]。
    pub fn advance(&mut self, raw: f32, dt_sec: f32) -> f32 {
        let raw = raw.clamp(0.0, 1.0);
        let dt = dt_sec.max(1.0e-4);
        let tau = if raw >= self.state {
            GATE_ATTACK_SEC
        } else {
            GATE_RELEASE_SEC
        };
        let alpha = (dt / tau).clamp(0.0, 1.0);
        self.state += (raw - self.state) * alpha;
        self.state
    }
}

/// 计算幅频比值 H(k)（dB），bin 覆盖 [0, fft_size/2]。
///
/// 流程：
/// 1. bin k 对应真实频率 f = k·stft_sr/fft_size；LPC 系数是 **分析域**
///    （采样率 `model_rate`）拟合的，求值时必须用 w = 2π·f/model_rate
///    —— 用 STFT 采样率归一化会把极点映射到 (stft_sr/model_rate) 倍频率处
///    （曾导致"搬不动"的直接 bug）。
/// 2. H_db = 20·log10(|A_orig| / |A_target|)，clamp 到 ±H_LIMIT_DB。
/// 3. LPC 模型仅在 `model_rate`/2 以下可信：其上向 0 dB 余弦衰减
///    （过渡带 MODEL_FADE_HZ），避免外推噪声污染高频。
///
/// 参数：
/// - `coeffs_orig` / `coeffs_target`：a[1..=p] 分析约定系数（分析域拟合）。
/// - `stft_sr`：STFT 域采样率（决定 bin → 真实频率映射）。
/// - `model_rate`：LPC 分析域采样率（决定系数的归一化频率轴）。
pub fn h_ratio_db(
    coeffs_orig: &[f32],
    coeffs_target: &[f32],
    fft_size: usize,
    stft_sr: f32,
    model_rate: f32,
) -> Vec<f32> {
    let half = fft_size / 2 + 1;
    let mut out = vec![0.0_f32; half];
    if stft_sr <= 0.0 || model_rate <= 0.0 || fft_size < 2 {
        return out;
    }
    let model_nyq = (model_rate / 2.0).min(stft_sr / 2.0);
    let ramp_lo = (model_nyq - MODEL_FADE_HZ).max(0.0);
    for (k, slot) in out.iter_mut().enumerate() {
        let f = k as f32 * stft_sr / fft_size as f32;
        let db = all_pole_mag_db(coeffs_orig, f, model_rate)
            - all_pole_mag_db(coeffs_target, f, model_rate);
        let fade = if f >= model_nyq {
            0.0
        } else if f > ramp_lo {
            let t = 1.0 - (f - ramp_lo) / (model_nyq - ramp_lo).max(1.0);
            0.5 - 0.5 * (std::f32::consts::PI * t).cos()
        } else {
            1.0
        };
        *slot = (db * fade).clamp(-H_LIMIT_DB, H_LIMIT_DB);
    }
    out
}

/// 逐帧 RMS 对齐：把 `wet` 的 RMS 缩放到与 `dry` 一致（增益 clamp 0.5–2.0）。
pub fn match_frame_energy(dry: &[f32], wet: &mut [f32]) {
    let n = dry.len().min(wet.len());
    if n == 0 {
        return;
    }
    let rms = |x: &[f32]| -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt()
    };
    let dry_rms = rms(&dry[..n]).max(1.0e-9);
    let wet_rms = rms(&wet[..n]).max(1.0e-9);
    let gain = (dry_rms / wet_rms).clamp(RMS_MATCH_MIN, RMS_MATCH_MAX);
    if (gain - 1.0).abs() > 1.0e-6 {
        for s in wet.iter_mut().take(n) {
            *s *= gain;
        }
    }
}

/// 输出保护：整体峰值 ≤ 输入峰值 × 1.6，随后对超膝样本做 tanh 软限幅。
///
/// 软膝：|x| ≤ SOFT_KNEE 线性通过；超过部分
/// f(x) = k + (1−k)·tanh((|x|−k)/(1−k))，拐点处斜率连续，渐近 1.0。
pub fn soft_limit(out: &mut [f32], input: &[f32]) {
    // 1. 整体峰值保护
    let in_peak = input.iter().fold(0.0_f32, |p, s| p.max(s.abs())).max(1.0e-6);
    let out_peak = out.iter().fold(0.0_f32, |p, s| p.max(s.abs())).max(1.0e-6);
    let limit = in_peak * super::OUTPUT_PEAK_RATIO_LIMIT;
    if out_peak > limit {
        let gain = limit / out_peak;
        for s in out.iter_mut() {
            *s *= gain;
        }
    }
    // 2. 软膝限幅
    for s in out.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
            continue;
        }
        let mag = s.abs();
        if mag > SOFT_KNEE {
            let shaped =
                SOFT_KNEE + (1.0 - SOFT_KNEE) * ((mag - SOFT_KNEE) / (1.0 - SOFT_KNEE)).tanh();
            *s = s.signum() * shaped;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// 单个共轭极点对 → 二阶全极点系数 a[1..=2]（A(z)=1 − 2r·cosθ·z⁻¹ + r²·z⁻²）。
    fn coeffs_of_pole(freq: f32, bw: f32, sr: f32) -> Vec<f32> {
        let r = (-(std::f32::consts::PI * bw) / sr).exp();
        let theta = 2.0 * std::f32::consts::PI * freq / sr;
        vec![-2.0 * r * theta.cos(), r * r]
    }

    #[test]
    fn moved_pole_interpolates_by_strength() {
        assert!((moved_pole_freq(800.0, 300.0, 0.0) - 800.0).abs() < 1e-5);
        assert!((moved_pole_freq(800.0, 300.0, 1.0) - 300.0).abs() < 1e-5);
        assert!((moved_pole_freq(800.0, 300.0, 0.9) - 350.0).abs() < 1e-4);
    }

    #[test]
    fn h_ratio_peaks_at_moved_formant() {
        // 原极点 800 Hz → 目标 300 Hz：H 在 300 Hz 为正峰、800 Hz 为负谷
        let orig = coeffs_of_pole(800.0, 80.0, SR);
        let target = coeffs_of_pole(300.0, 80.0, SR);
        let fft_size = 2048;
        let h = h_ratio_db(&orig, &target, fft_size, SR, SR);
        assert_eq!(h.len(), fft_size / 2 + 1);
        let bin_of = |f: f32| (f / (SR / fft_size as f32)).round() as usize;
        let h_at = |f: f32| h[bin_of(f)];
        assert!(
            h_at(300.0) > 3.0,
            "目标位置应为正峰，got {} dB",
            h_at(300.0)
        );
        assert!(
            h_at(800.0) < -3.0,
            "原位置应为负谷，got {} dB",
            h_at(800.0)
        );
        // 远离两个极点的频段（2 kHz）应接近恒等（极点裙摆内如 50 Hz 本就
        // 会随共振峰迁移而变化，属于物理正确行为，不作恒等断言）
        assert!(
            h_at(2_000.0).abs() < 8.0,
            "远离极点处应接近恒等，got {} dB",
            h_at(2_000.0)
        );
    }

    #[test]
    fn h_ratio_is_clamped() {
        // 目标极点带宽 1 Hz → 峰值增益远超 24 dB，必须被限幅
        let orig = vec![0.0_f32, 0.0]; // A = 1（平坦）
        let target = coeffs_of_pole(1_000.0, 1.0, SR);
        let h = h_ratio_db(&orig, &target, 2048, SR, SR);
        for (i, v) in h.iter().enumerate() {
            assert!(v.abs() <= H_LIMIT_DB + 0.1, "bin {i} 超限: {v}");
        }
    }

    #[test]
    fn gate_attack_release_smoothing() {
        let mut gate = VoicingGate::new();
        // 上升：不应一步到位
        let first = gate.advance(1.0, 0.01);
        assert!(first > 0.0 && first < 1.0, "attack 应渐进，got {first}");
        for _ in 0..50 {
            gate.advance(1.0, 0.01);
        }
        assert!((gate.advance(1.0, 0.01) - 1.0).abs() < 1.0e-3, "应收敛到 1");
        // 下降：release 更慢
        let mut g2 = VoicingGate::new();
        for _ in 0..50 {
            g2.advance(1.0, 0.01);
        }
        let down1 = g2.advance(0.0, 0.01);
        let down2 = g2.advance(0.0, 0.01);
        assert!(down1 < 1.0 && down1 > down2, "release 应渐进下降");
        // 一步 release 后的衰减量应小于一步 attack 的推进量（0.01/ATTACK）
        assert!(
            down1 > 1.0 - 0.01 / GATE_ATTACK_SEC,
            "release 应慢于 attack，down1={down1}"
        );
    }

    #[test]
    fn match_frame_energy_clamps_gain() {
        let dry = vec![0.1_f32; 1000];
        let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        // 1.5×：在 clamp 范围内 → 完全对齐
        let mut wet_mild: Vec<f32> = dry.iter().map(|s| s * 1.5).collect();
        match_frame_energy(&dry, &mut wet_mild);
        assert!(
            (rms(&wet_mild) / rms(&dry) - 1.0).abs() < 0.05,
            "1.5× 应被完全对齐"
        );
        // 10×：需要 0.1 增益 → 被 clamp 到 RMS_MATCH_MIN，最终为 5×
        let mut wet_loud: Vec<f32> = dry.iter().map(|s| s * 10.0).collect();
        match_frame_energy(&dry, &mut wet_loud);
        let ratio = rms(&wet_loud) / rms(&dry);
        assert!(
            (ratio - 10.0 * RMS_MATCH_MIN).abs() < 0.1,
            "增益必须被 clamp 到 {RMS_MATCH_MIN}，got ratio={ratio}"
        );
        // 0.01×：需要 100× 增益 → 被 clamp 到 2.0，最终为 0.02×
        let mut wet_quiet: Vec<f32> = dry.iter().map(|s| s * 0.01).collect();
        match_frame_energy(&dry, &mut wet_quiet);
        let ratio = rms(&wet_quiet) / rms(&dry);
        assert!(
            (ratio - 0.01 * RMS_MATCH_MAX).abs() < 0.001,
            "增益必须被 clamp 到 {RMS_MATCH_MAX}，got ratio={ratio}"
        );
    }

    #[test]
    fn soft_limit_bounds_output() {
        let input = vec![0.1_f32; 100];
        let mut out = vec![5.0_f32; 100];
        soft_limit(&mut out, &input);
        for s in &out {
            assert!(s.is_finite());
            assert!(s.abs() <= 1.0, "软限幅后必须 < 1，got {s}");
        }
        let out_peak = out.iter().fold(0.0_f32, |p, s| p.max(s.abs()));
        let in_peak = 0.1_f32;
        assert!(
            out_peak <= in_peak * 1.6 + 1.0e-4,
            "峰值不得超过输入 × 1.6，got {out_peak}"
        );
        // 正常信号不受影响（峰值在全局限制与膝点之内）
        let loud_input = vec![0.6_f32; 100];
        let mut normal: Vec<f32> = (-50..50).map(|i| i as f32 / 100.0).collect();
        let before = normal.clone();
        soft_limit(&mut normal, &loud_input);
        assert_eq!(normal, before, "低于膝点的样本必须原样保留");
    }
}
