/*
 * formant_morph/track.rs - 共振峰候选筛选与轨迹平滑。
 *
 * 主要内容：
 * - select_f1_f2：从一帧的 LPC 极点中筛选 F1/F2 候选（频率区间 + 带宽区间
 *   + 最小间距 + 相对谱包络的显著性）。筛选失败返回 None —— 调用方必须
 *   恒等直通，严禁用常量兜底（旧路线"假锚点"教训）。
 * - extract_tracks：逐帧候选 → 连续轨迹（未检出帧用最近有效值桥接）。
 * - smooth_tracks：5 帧中值滤波 + 每帧 ±15% 限速，消除逐帧跳变与倍频程错误。
 * - interpolate_at：帧间线性插值，供 STFT hop 网格取值。
 *
 * 与其他模块的关系：
 * - 输入 Pole/FramePoles 来自 lpc.rs；被 mod.rs（DSP 主流程）与
 *   analysis.rs（IPC 源共振峰统计）共用。
 *
 * 维护说明：
 * - 显著性用谱包络 |E(f)| = 1/|A(f)| 的 dB 差衡量（Horner 直接求值，
 *   不需要 FFT）；A 采用分析约定 a[0]=1（系数为 FrameLpc.coeffs）。
 * - 所有阈值集中为本文件常量，调参不散落。
 */

use super::lpc::{FramePoles, Pole};

// ── 筛选阈值（集中管理） ─────────────────────────────────────────────────

/// F1 频率区间（Hz）。
pub const F1_LO_HZ: f32 = 200.0;
pub const F1_HI_HZ: f32 = 1_000.0;
/// F2 频率区间（Hz）。
pub const F2_LO_HZ: f32 = 800.0;
pub const F2_HI_HZ: f32 = 3_000.0;
/// 极点带宽有效区间（Hz）：过宽 ≈ 非共振峰（谐波簇/数值噪声），过窄 ≈ 病态。
pub const BANDWIDTH_MIN_HZ: f32 = 30.0;
pub const BANDWIDTH_MAX_HZ: f32 = 500.0;
/// F2 与 F1 的最小间距（Hz）。
pub const MIN_F1_F2_GAP_HZ: f32 = 200.0;
/// 显著性阈值：峰相对两侧 300 Hz 处谷点 ≥ 3 dB。
const PROMINENCE_MIN_DB: f32 = 3.0;
/// 显著性评估的偏移频率（Hz）。
const PROMINENCE_OFFSET_HZ: f32 = 300.0;
/// 轨迹中值滤波窗长（帧，奇数）。
const MEDIAN_WINDOW: usize = 5;
/// 每帧限速比例（相对前一帧值）。
const MAX_RATE_PER_FRAME: f32 = 0.15;

/// 一帧内筛选出的共振峰候选。
#[derive(Debug, Clone, Copy)]
pub struct FormantCandidate {
    pub f1: Pole,
    pub f2: Pole,
    /// 该帧 LPC 残差能量比（透传给浊音 gate）。
    pub residual_ratio: f32,
}

/// 轨迹采样点（每分析帧一个）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackPoint {
    pub f1_hz: f32,
    pub f2_hz: f32,
    /// 1.0 = 该帧实际检出；0.0 = 桥接填充。gate 依据此值平滑过渡。
    pub voiced: f32,
}

/// 从一帧极点中筛选 F1/F2。
///
/// 流程：
/// 1. 按频率区间 + 带宽区间过滤候选极点。
/// 2. 每个候选计算显著性（|E(f)| 相对 ±300 Hz 谷点 ≥ 3 dB 才保留）。
/// 3. F1 = F1 区间内通过门槛者中带宽最窄的；F2 = F2 区间内带宽最窄
///    且与 F1 间距 ≥ 200 Hz 者。
///
/// 参数：`coeffs` 为该帧 LPC 系数（FrameLpc.coeffs，a[0]=1 约定）。
/// 返回 `None`：无合格候选（该帧视为非元音，调用方恒等直通）。
pub fn select_f1_f2(
    poles: &FramePoles,
    coeffs: &[f32],
    sr: f32,
    residual_ratio: f32,
) -> Option<FormantCandidate> {
    if sr <= 0.0 || coeffs.is_empty() {
        return None;
    }
    // 1. 频率区间 + 带宽区间过滤
    let mut f1_cands: Vec<&Pole> = Vec::new();
    let mut f2_cands: Vec<&Pole> = Vec::new();
    for pole in &poles.pairs {
        if !(BANDWIDTH_MIN_HZ..=BANDWIDTH_MAX_HZ).contains(&pole.bandwidth_hz) {
            continue;
        }
        if (F1_LO_HZ..=F1_HI_HZ).contains(&pole.freq_hz) {
            f1_cands.push(pole);
        } else if (F2_LO_HZ..=F2_HI_HZ).contains(&pole.freq_hz) {
            f2_cands.push(pole);
        }
    }
    if f1_cands.is_empty() {
        return None;
    }

    // 2. 显著性评分：|E(f)| = 1/|A(f)| 相对 ±PROMINENCE_OFFSET 谷点 ≥ 3 dB
    let prominence = |freq: f32| -> f32 {
        let peak_db = all_pole_mag_db(coeffs, freq, sr);
        let valley_lo = all_pole_mag_db(coeffs, (freq - PROMINENCE_OFFSET_HZ).max(80.0), sr);
        let valley_hi = all_pole_mag_db(coeffs, (freq + PROMINENCE_OFFSET_HZ).min(sr * 0.45), sr);
        // |E| = 1/|A| ⇒ prominence(dB) = |A|(谷) − |A|(峰)；取较浅一侧（保守）
        (valley_lo.max(valley_hi) - peak_db).max(0.0)
    };

    // 3. F1/F2 = 区间内通过显著性门槛的候选中带宽最窄者
    //    （越窄越像共振峰；显著性只做门槛不做排序 —— 宽极点在谱包络平台上
    //    的相对突出度反而可能更高，排序不可靠）
    let f1 = **f1_cands
        .iter()
        .filter(|p| prominence(p.freq_hz) >= PROMINENCE_MIN_DB)
        .min_by(|a, b| {
            a.bandwidth_hz
                .partial_cmp(&b.bandwidth_hz)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let f2 = **f2_cands
        .iter()
        .filter(|p| {
            p.freq_hz - f1.freq_hz >= MIN_F1_F2_GAP_HZ
                && prominence(p.freq_hz) >= PROMINENCE_MIN_DB
        })
        .min_by(|a, b| {
            a.bandwidth_hz
                .partial_cmp(&b.bandwidth_hz)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    Some(FormantCandidate {
        f1,
        f2,
        residual_ratio,
    })
}

/// 全极点滤波器 |A(e^{-jw})| 的 dB 值（Horner 直接求值）。
///
/// 约定：`coeffs` 为 a[1..=p]（A(z) = 1 + Σ a_k·z⁻ᵏ）。
/// correction.rs 的 H(k) 计算复用本函数（单一真相）。
pub(crate) fn all_pole_mag_db(coeffs: &[f32], freq_hz: f32, sr: f32) -> f32 {
    let w = 2.0 * std::f32::consts::PI * freq_hz / sr;
    let mut re = 1.0_f32;
    let mut im = 0.0_f32;
    for (k, &a) in coeffs.iter().enumerate() {
        let arg = -((k + 1) as f32) * w;
        re += a * arg.cos();
        im += a * arg.sin();
    }
    20.0 * (re * re + im * im).sqrt().max(1.0e-9).log10()
}

/// 逐帧候选 → 连续轨迹。
///
/// 规则：未检出帧沿用最近有效值（前向桥接；开头未检出用首个有效值回填），
/// `voiced = 0` 标记桥接点，供 gate 平滑降权。
pub fn extract_tracks(cands: &[Option<FormantCandidate>]) -> Vec<TrackPoint> {
    let mut tracks = Vec::with_capacity(cands.len());
    let mut last: Option<(f32, f32)> = None;
    for cand in cands {
        match cand {
            Some(c) => {
                last = Some((c.f1.freq_hz, c.f2.freq_hz));
                tracks.push(TrackPoint {
                    f1_hz: c.f1.freq_hz,
                    f2_hz: c.f2.freq_hz,
                    voiced: 1.0,
                });
            }
            None => {
                // 前向桥接；开头未检出时暂以 0 占位，随后回填
                let (f1, f2) = last.unwrap_or((0.0, 0.0));
                tracks.push(TrackPoint {
                    f1_hz: f1,
                    f2_hz: f2,
                    voiced: 0.0,
                });
            }
        }
    }
    // 回填开头未检出帧
    if let Some(first_valid) = tracks.iter().find(|t| t.voiced > 0.0).copied() {
        for t in tracks.iter_mut() {
            if t.voiced > 0.0 {
                break;
            }
            t.f1_hz = first_valid.f1_hz;
            t.f2_hz = first_valid.f2_hz;
        }
    }
    tracks
}

/// 轨迹平滑：先 5 帧中值（消除孤立跳点），再每帧 ±15% 限速（消除连续漂移）。
///
/// 中值窗口对边界做 clamp；限速基于平滑后的前一帧输出值（累积钳制）。
pub fn smooth_tracks(tracks: &mut Vec<TrackPoint>) {
    if tracks.is_empty() {
        return;
    }
    let radius = MEDIAN_WINDOW / 2;
    let n = tracks.len();

    // 复用同一窗口缓冲区：逐帧 to_vec + 排序在长素材上是十万级的分配开销
    let mut window: Vec<f32> = Vec::with_capacity(MEDIAN_WINDOW);
    let mut median_at = |values: &[f32], idx: usize| -> f32 {
        let lo = idx.saturating_sub(radius);
        let hi = (idx + radius).min(n - 1);
        window.clear();
        window.extend_from_slice(&values[lo..=hi]);
        window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        window[window.len() / 2]
    };

    let f1s: Vec<f32> = tracks.iter().map(|t| t.f1_hz).collect();
    let f2s: Vec<f32> = tracks.iter().map(|t| t.f2_hz).collect();
    for (i, t) in tracks.iter_mut().enumerate() {
        t.f1_hz = median_at(&f1s, i);
        t.f2_hz = median_at(&f2s, i);
    }

    // 限速：相对前一帧输出值，逐帧钳制
    for i in 1..n {
        let prev = tracks[i - 1];
        let cur = tracks[i];
        tracks[i].f1_hz = rate_limit(cur.f1_hz, prev.f1_hz);
        tracks[i].f2_hz = rate_limit(cur.f2_hz, prev.f2_hz);
    }
}

/// 单值限速：|x − prev| ≤ MAX_RATE_PER_FRAME × |prev|。
fn rate_limit(x: f32, prev: f32) -> f32 {
    if !x.is_finite() || !prev.is_finite() || prev.abs() < 1.0e-6 {
        return x;
    }
    let max_delta = prev.abs() * MAX_RATE_PER_FRAME;
    prev + (x - prev).clamp(-max_delta, max_delta)
}

/// 帧间线性插值取轨迹值（`t` 为浮点帧索引，clamp 到有效范围）。
pub fn interpolate_at(tracks: &[TrackPoint], t: f32) -> Option<TrackPoint> {
    if tracks.is_empty() {
        return None;
    }
    let t = t.clamp(0.0, (tracks.len() - 1) as f32);
    let i0 = t.floor() as usize;
    let i1 = (i0 + 1).min(tracks.len() - 1);
    let frac = t - i0 as f32;
    let a = tracks[i0];
    let b = tracks[i1];
    Some(TrackPoint {
        f1_hz: a.f1_hz + (b.f1_hz - a.f1_hz) * frac,
        f2_hz: a.f2_hz + (b.f2_hz - a.f2_hz) * frac,
        voiced: a.voiced + (b.voiced - a.voiced) * frac,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// 测试
// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::formant_morph::lpc::poles_to_coeffs;

    const SR: f32 = 12_000.0;

    /// 用已知极点构造 LPC 系数（分析约定 a[0]=1：z 多项式升幂去尾反序）。
    ///
    /// 注意：A(z) = Π(1 − p_k·z⁻¹) = z⁻ⁿ·P(z)，因此 a_k = c_{n−k}，
    /// **不能**再做任何归一化（会改变根的位置）。
    fn coeffs_from(pairs: &[Pole]) -> Vec<f32> {
        let mono = poles_to_coeffs(pairs, &[], SR);
        mono[..mono.len() - 1].iter().rev().copied().collect()
    }

    fn pole(freq: f32, bw: f32) -> Pole {
        Pole {
            freq_hz: freq,
            bandwidth_hz: bw,
            radius: (-(std::f32::consts::PI * bw) / SR).exp(),
        }
    }

    fn fp(pairs: Vec<Pole>) -> FramePoles {
        FramePoles {
            pairs,
            real_roots: vec![],
        }
    }

    #[test]
    fn select_picks_expected_formants() {
        // 低频宽极点（模拟 F0 区，带宽超标被剔除）+ 三元音极点
        let pairs = vec![
            pole(300.0, 700.0),  // 带宽超标 → 剔除
            pole(800.0, 80.0),   // → F1
            pole(1_200.0, 80.0), // → F2
            pole(2_600.0, 200.0),
        ];
        let coeffs = coeffs_from(&pairs);
        let cand = select_f1_f2(&fp(pairs), &coeffs, SR, 0.1).expect("应选出候选");
        assert!((cand.f1.freq_hz - 800.0).abs() < 1.0, "got {}", cand.f1.freq_hz);
        assert!((cand.f2.freq_hz - 1_200.0).abs() < 1.0, "got {}", cand.f2.freq_hz);
    }

    #[test]
    fn select_rejects_harmonic_like_poles() {
        // F1 区间内唯一候选带宽 900 Hz（超出 500 上限）→ F1 无合格候选 → None
        let pairs = vec![pole(400.0, 900.0), pole(1_500.0, 80.0)];
        let coeffs = coeffs_from(&pairs);
        assert!(
            select_f1_f2(&fp(pairs), &coeffs, SR, 0.1).is_none(),
            "F1 带宽超标必须拒绝"
        );
    }

    #[test]
    fn select_requires_f2_above_f1_plus_gap() {
        // F1=700、F2 候选=850：间距 150 < 200 → F2 被拒 → None
        let pairs = vec![pole(700.0, 80.0), pole(850.0, 80.0)];
        let coeffs = coeffs_from(&pairs);
        assert!(
            select_f1_f2(&fp(pairs), &coeffs, SR, 0.1).is_none(),
            "F2−F1 不足最小间距必须拒绝"
        );
        // 同样极点但 F2 合规（1500）：应选出 700/1500
        let pairs2 = vec![pole(700.0, 80.0), pole(1_500.0, 80.0)];
        let coeffs2 = coeffs_from(&pairs2);
        let cand = select_f1_f2(&fp(pairs2), &coeffs2, SR, 0.1).expect("合规组合应选出");
        assert!((cand.f1.freq_hz - 700.0).abs() < 1.0);
        assert!((cand.f2.freq_hz - 1_500.0).abs() < 1.0);
    }

    #[test]
    fn extract_bridges_unvoiced_frames() {
        let mk = |f1: f32, f2: f32| {
            Some(FormantCandidate {
                f1: pole(f1, 80.0),
                f2: pole(f2, 80.0),
                residual_ratio: 0.1,
            })
        };
        let cands = vec![mk(800.0, 1_200.0), None, None, mk(780.0, 1_180.0)];
        let tracks = extract_tracks(&cands);
        assert_eq!(tracks.len(), 4);
        assert!((tracks[0].f1_hz - 800.0).abs() < 1e-3);
        assert!((tracks[0].voiced - 1.0).abs() < 1e-6);
        // 桥接：沿用最近有效值，voiced=0
        assert!((tracks[1].f1_hz - 800.0).abs() < 1e-3);
        assert!((tracks[1].voiced - 0.0).abs() < 1e-6);
        assert!((tracks[3].voiced - 1.0).abs() < 1e-6);
        // 开头未检出：用首个有效值回填
        let cands2 = vec![None, mk(800.0, 1_200.0)];
        let tracks2 = extract_tracks(&cands2);
        assert!((tracks2[0].f1_hz - 800.0).abs() < 1e-3);
        assert!((tracks2[0].voiced - 0.0).abs() < 1e-6);
    }

    #[test]
    fn smooth_kills_octave_jump() {
        let mut tracks: Vec<TrackPoint> = [800.0, 800.0, 1_600.0, 800.0, 800.0]
            .iter()
            .map(|&f1| TrackPoint {
                f1_hz: f1,
                f2_hz: 1_200.0,
                voiced: 1.0,
            })
            .collect();
        smooth_tracks(&mut tracks);
        for t in &tracks {
            assert!(
                (t.f1_hz - 800.0).abs() / 800.0 < MAX_RATE_PER_FRAME,
                "中值+限速后不允许残留倍频程跳变，got {}",
                t.f1_hz
            );
        }
    }

    #[test]
    fn smooth_limits_continuous_drift() {
        // 每帧 +20%（超限）的连续漂移：限速后每帧增量 ≤ 15%
        let mut tracks: Vec<TrackPoint> = (0..8)
            .map(|i| TrackPoint {
                f1_hz: 500.0 * 1.2_f32.powi(i),
                f2_hz: 1_200.0,
                voiced: 1.0,
            })
            .collect();
        smooth_tracks(&mut tracks);
        for w in tracks.windows(2) {
            let ratio = w[1].f1_hz / w[0].f1_hz;
            assert!(
                (ratio - 1.0).abs() <= MAX_RATE_PER_FRAME + 1.0e-4,
                "帧间变化超限: {ratio}"
            );
        }
    }

    #[test]
    fn interpolate_is_linear_between_frames() {
        let tracks = vec![
            TrackPoint {
                f1_hz: 400.0,
                f2_hz: 1_000.0,
                voiced: 1.0,
            },
            TrackPoint {
                f1_hz: 600.0,
                f2_hz: 2_000.0,
                voiced: 1.0,
            },
        ];
        let mid = interpolate_at(&tracks, 0.5).unwrap();
        assert!((mid.f1_hz - 500.0).abs() < 1e-4);
        assert!((mid.f2_hz - 1_500.0).abs() < 1e-4);
        assert!(interpolate_at(&tracks, -1.0).unwrap().f1_hz - 400.0 < 1e-4);
        assert!(interpolate_at(&tracks, 5.0).unwrap().f1_hz - 600.0 < 1e-4);
    }
}
