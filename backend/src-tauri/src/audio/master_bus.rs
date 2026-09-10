/*
 * audio/master_bus.rs - 主总线（master bus）动态处理。
 *
 * 主要内容：
 * - `soft_clip_sample` / `apply_soft_clip`：软膝（tanh）软削波。
 *
 * 与其他模块的关系：
 * - 被**实时混音**（`audio_engine/mix.rs` 的三个输出回调）与**离线导出**
 *   （`audio/mixdown.rs` 的主混音）共同调用，保证"听到的"与"导出的"
 *   经过完全相同的总线处理（见 P0-2）。
 * - 软膝曲线与 `audio/formant_morph/correction.rs::soft_limit` 同源
 *   （`f(x) = k + (1−k)·tanh((|x|−k)/(1−k))`，拐点处斜率连续、渐近 1.0）。
 *
 * 设计要点：
 * - **为什么需要**：此前主混音是纯求和，唯一保护是写出设备缓冲前的
 *   `clamp11` 硬截断。轨道增益上限为 4.0（`snapshot.rs`），多轨叠加极易
 *   越过 0 dBFS → 硬削波产生强奇次谐波失真。
 * - **软削波买到的到底是什么（实测数据）**：不是"更小的波形偏差"，而是
 *   **高次谐波抑制**。转移曲线在膝点处一阶连续（硬削波的一阶导数是跳变的），
 *   谐波包络衰减更快，听感明显更不刺耳。以 200 Hz 正弦、48 kHz 实测
 *   （`hi` = 13 次以上谐波能量，`lo` = 3 次以上谐波能量）：
 *
 *   | 过载 | 硬削 hi/lo | 软削(膝 0.7) hi/lo | L2 偏差（硬 / 软） |
 *   |---|---|---|---|
 *   | 1.05× | 1.44e-2 | 1.53e-4（≈94× 更低） | 1.12 / 3.16 |
 *   | 1.20× | 3.65e-3 | 1.99e-4（≈18× 更低） | 6.15 / 7.39 |
 *   | 1.50× | 2.26e-3 | 2.87e-4（≈8× 更低） | 18.35 / 18.76 |
 *   | 2.00× | 2.73e-3 | 3.21e-4（≈9× 更低） | 40.75 / 40.86 |
 *
 *   即：**过载 ≥1.2× 时 L2 偏差与硬削波相当**，而高次谐波大幅降低；
 *   轻度过载下软削波的 L2 偏差反而更大（它更早开始压缩），但此时高次谐波
 *   的降幅最大。因此本模块按"抑制刺耳谐波、同时不产生硬削阶跃"定位，
 *   **不以最小化 L2 偏差为目标**。
 * - **膝值选择**：默认 0.7（-3.1 dBFS），与项目既有的
 *   `formant_morph::correction::SOFT_KNEE` 保持一致；实测这是谐波抑制与
 *   "尽量不动正常素材"之间的较优折中（膝 0.9 的抑制效果仅剩 1.2–5×）。
 * - **RT 安全**：不分配、不加锁；配置读取走 `AtomicBool` / `AtomicU32`
 *   （`Relaxed`），且 `|x| <= knee` 的样本直接原样返回，因此实际调用
 *   `tanh` 的只有越界峰值，常规电平下开销接近零。
 * - **输出上界**：对有限输入，输出有界于 ±1。注意 f32 下极大输入
 *   （如 |x| ≥ 10）的 `tanh` 会舍入到恰好 1.0，因此**不能**声称"严格小于 1"；
 *   但仍保证**永不超出** ±1，故其后的 `clamp11` 仅作 NaN / 极端值兜底，
 *   不再承担削波职责。
 */

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// 默认软膝起点：|x| ≤ knee 完全线性通过。
pub(crate) const DEFAULT_SOFT_CLIP_KNEE: f32 = 0.7;

/// 软膝起点可配置范围。
pub(crate) const MIN_SOFT_CLIP_KNEE: f32 = 0.1;
pub(crate) const MAX_SOFT_CLIP_KNEE: f32 = 1.0;

static MASTER_SOFT_CLIP_ENABLED: AtomicBool = AtomicBool::new(true);
static MASTER_SOFT_CLIP_KNEE_BITS: AtomicU32 =
    AtomicU32::new(f32::to_bits(DEFAULT_SOFT_CLIP_KNEE));

/// 环境变量读取一次的初始化标记（避免每次查询都做字符串解析）。
static ENV_APPLIED: AtomicBool = AtomicBool::new(false);

/// 应用一次环境变量覆盖（幂等）。
///
/// - `HIFISHIFTER_MASTER_SOFT_CLIP=0|1`：开关，默认 1（开启）。
/// - `HIFISHIFTER_MASTER_SOFT_CLIP_KNEE=<0.1..1.0>`：软膝起点。
fn apply_env_once() {
    if ENV_APPLIED.swap(true, Ordering::Relaxed) {
        return;
    }
    if let Ok(v) = std::env::var("HIFISHIFTER_MASTER_SOFT_CLIP") {
        let v = v.trim();
        if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off") {
            MASTER_SOFT_CLIP_ENABLED.store(false, Ordering::Relaxed);
        } else if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on") {
            MASTER_SOFT_CLIP_ENABLED.store(true, Ordering::Relaxed);
        }
    }
    if let Ok(v) = std::env::var("HIFISHIFTER_MASTER_SOFT_CLIP_KNEE") {
        if let Ok(k) = v.trim().parse::<f32>() {
            if k.is_finite() {
                set_soft_clip_knee(k);
            }
        }
    }
}

/// 主总线软削波是否启用。
#[inline]
pub(crate) fn soft_clip_enabled() -> bool {
    apply_env_once();
    MASTER_SOFT_CLIP_ENABLED.load(Ordering::Relaxed)
}

/// 设置主总线软削波开关，返回生效值。
pub(crate) fn set_soft_clip_enabled(enabled: bool) -> bool {
    MASTER_SOFT_CLIP_ENABLED.store(enabled, Ordering::Relaxed);
    enabled
}

/// 当前软膝起点。
#[inline]
pub(crate) fn soft_clip_knee() -> f32 {
    apply_env_once();
    f32::from_bits(MASTER_SOFT_CLIP_KNEE_BITS.load(Ordering::Relaxed))
}

/// 设置软膝起点（自动钳制到合法范围），返回生效值。
pub(crate) fn set_soft_clip_knee(knee: f32) -> f32 {
    let k = if knee.is_finite() {
        knee.clamp(MIN_SOFT_CLIP_KNEE, MAX_SOFT_CLIP_KNEE)
    } else {
        DEFAULT_SOFT_CLIP_KNEE
    };
    MASTER_SOFT_CLIP_KNEE_BITS.store(f32::to_bits(k), Ordering::Relaxed);
    k
}

/// 单样本软膝软削波。
///
/// |x| ≤ knee：原样通过（斜率 1）。
/// |x| > knee：`sign(x)·(k + (1−k)·tanh((|x|−k)/(1−k)))`，拐点处一阶连续
/// （硬削波在此处一阶导数跳变，因而产生更多高次谐波）。
/// 输出有界于 ±1 且单调不减；f32 下 |x| 很大时 `tanh` 会舍入到恰好 1.0。
#[inline]
pub(crate) fn soft_clip_sample(x: f32, knee: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let mag = x.abs();
    if mag <= knee {
        return x;
    }
    let over = (mag - knee) / (1.0 - knee);
    let shaped = knee + (1.0 - knee) * over.tanh();
    if x < 0.0 {
        -shaped
    } else {
        shaped
    }
}

/// 就地软削波。`knee` 会被钳制到合法范围；不分配、不加锁。
///
/// `enabled == false` 时不做任何处理（调用方随后仍会做 `clamp11` 硬削，
/// 即回到"无总线处理"的旧行为）。
#[inline]
pub(crate) fn apply_soft_clip(buf: &mut [f32], enabled: bool, knee: f32) {
    if !enabled {
        return;
    }
    let k = knee.clamp(MIN_SOFT_CLIP_KNEE, MAX_SOFT_CLIP_KNEE);
    for s in buf.iter_mut() {
        *s = soft_clip_sample(*s, k);
    }
}

/// 只处理前 `samples` 个元素（用于 `scratch` 这类容量大于当前块长的缓冲）。
#[inline]
pub(crate) fn apply_soft_clip_prefix(buf: &mut [f32], samples: usize, enabled: bool, knee: f32) {
    if !enabled {
        return;
    }
    let n = samples.min(buf.len());
    apply_soft_clip(&mut buf[..n], true, knee);
}

/// 主总线统计：峰值与触发软削波的样本数（用于电平/诊断上报）。
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MasterBusStats {
    pub peak: f32,
    pub limited_samples: u32,
}

/// 就地软削波并统计。供离线导出与诊断使用（实时路径用无统计版本省开销）。
pub(crate) fn apply_soft_clip_with_stats(
    buf: &mut [f32],
    enabled: bool,
    knee: f32,
) -> MasterBusStats {
    let mut stats = MasterBusStats::default();
    if !enabled {
        return stats;
    }
    let k = knee.clamp(MIN_SOFT_CLIP_KNEE, MAX_SOFT_CLIP_KNEE);
    for s in buf.iter_mut() {
        let before = *s;
        if before.is_finite() {
            let a = before.abs();
            if a > stats.peak {
                stats.peak = a;
            }
            if a > k {
                stats.limited_samples = stats.limited_samples.saturating_add(1);
            }
        }
        *s = soft_clip_sample(before, k);
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_knee_passes_through_exactly() {
        let k = DEFAULT_SOFT_CLIP_KNEE;
        for x in [-0.7f32, -0.35, 0.0, 0.35, 0.7] {
            assert_eq!(soft_clip_sample(x, k), x);
        }
    }

    #[test]
    fn above_knee_is_compressed_bounded_and_monotonic() {
        let k = DEFAULT_SOFT_CLIP_KNEE;
        let mut prev = f32::NEG_INFINITY;
        let mut x = 0.0f32;
        while x <= 50.0 {
            let y = soft_clip_sample(x, k);
            assert!(
                y <= 1.0,
                "output must never exceed 1.0, got {y} at x={x}"
            );
            assert!(y >= 0.0, "sign must be preserved, got {y} at x={x}");
            assert!(y >= prev - 1e-6, "transfer must be monotonic at x={x}");
            prev = y;
            x += 0.001;
        }
        // 过膝段确实被压缩
        for x in [0.8f32, 1.0, 1.5, 3.0] {
            let y = soft_clip_sample(x, k);
            assert!(y > k && y < x, "x={x} should be compressed, got {y}");
        }
    }

    #[test]
    fn extrema_are_exactly_bounded_not_over() {
        // f32 下 tanh 在大参数处会舍入到 1.0，因此输出可恰好等于 1.0，
        // 但绝不能超过 —— 这正是"其后 clamp11 只是兜底"的依据。
        let k = DEFAULT_SOFT_CLIP_KNEE;
        for x in [10.0f32, 1e3, 1e30, f32::MAX] {
            let y = soft_clip_sample(x, k);
            assert!(y <= 1.0 && y >= 0.0, "x={x} produced out-of-range {y}");
        }
    }

    #[test]
    fn symmetry_and_sign_preservation() {
        let k = DEFAULT_SOFT_CLIP_KNEE;
        for x in [0.75f32, 1.3, 4.0] {
            let p = soft_clip_sample(x, k);
            let n = soft_clip_sample(-x, k);
            assert!((p + n).abs() < 1e-6, "odd symmetry broken for x={x}");
            assert!(n < 0.0);
        }
    }

    #[test]
    fn slope_is_continuous_at_knee() {
        // 拐点两侧数值导数应接近 1（一阶连续）。
        let k = DEFAULT_SOFT_CLIP_KNEE;
        let h = 1e-4f32;
        let left = (soft_clip_sample(k, k) - soft_clip_sample(k - h, k)) / h;
        let right = (soft_clip_sample(k + h, k) - soft_clip_sample(k, k)) / h;
        assert!(
            (left - 1.0).abs() < 1e-3,
            "left slope should be 1, got {left}"
        );
        assert!(
            (right - 1.0).abs() < 1e-3,
            "right slope should be 1 (C1 continuity), got {right}"
        );
    }

    #[test]
    fn non_finite_becomes_silence() {
        let k = DEFAULT_SOFT_CLIP_KNEE;
        assert_eq!(soft_clip_sample(f32::NAN, k), 0.0);
        assert_eq!(soft_clip_sample(f32::INFINITY, k), 0.0);
        assert_eq!(soft_clip_sample(f32::NEG_INFINITY, k), 0.0);
    }

    #[test]
    fn disabled_is_a_no_op_even_when_overloaded() {
        let mut buf = vec![2.0f32, -3.0, 0.5];
        let original = buf.clone();
        apply_soft_clip(&mut buf, false, DEFAULT_SOFT_CLIP_KNEE);
        assert_eq!(buf, original);
    }

    #[test]
    fn prefix_only_touches_requested_range() {
        let mut buf = vec![2.0f32, 2.0, 0.1, 0.1];
        apply_soft_clip_prefix(&mut buf, 2, true, DEFAULT_SOFT_CLIP_KNEE);
        assert!(buf[0] < 2.0 && buf[1] < 2.0);
        assert_eq!(buf[2], 0.1);
        assert_eq!(buf[3], 0.1);
    }

    #[test]
    fn stats_count_limited_samples_and_peak() {
        let mut buf = vec![0.5f32, 0.8, -1.2, 0.2];
        let st = apply_soft_clip_with_stats(&mut buf, true, DEFAULT_SOFT_CLIP_KNEE);
        assert_eq!(st.limited_samples, 2, "0.8 and -1.2 exceed knee 0.7");
        assert!((st.peak - 1.2).abs() < 1e-6);
    }

    #[test]
    fn knee_setter_clamps_out_of_range_values() {
        let original = soft_clip_knee();
        assert_eq!(set_soft_clip_knee(5.0), MAX_SOFT_CLIP_KNEE);
        assert_eq!(set_soft_clip_knee(-1.0), MIN_SOFT_CLIP_KNEE);
        assert_eq!(set_soft_clip_knee(f32::NAN), DEFAULT_SOFT_CLIP_KNEE);
        set_soft_clip_knee(original);
    }

    /// 计算 `y` 在整数次谐波上（相对 F0）的幅度，第 h 次谐波为 `amps[h-1]`。
    fn harmonic_amplitudes(y: &[f32], f0: f64, sr: f64, max_h: usize) -> Vec<f64> {
        let n = y.len() as f64;
        (1..=max_h)
            .map(|h| {
                let w = 2.0 * std::f64::consts::PI * (h as f64) * f0 / sr;
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (i, v) in y.iter().enumerate() {
                    let a = w * i as f64;
                    re += *v as f64 * a.cos();
                    im += *v as f64 * a.sin();
                }
                2.0 * (re * re + im * im).sqrt() / n
            })
            .collect()
    }

    /// 高次谐波占比：13 次及以上能量 / 3 次及以上能量。数值越小 → 听感越不刺耳。
    fn high_harmonic_ratio(amps: &[f64]) -> f64 {
        let sum = |from: usize| -> f64 {
            amps.iter()
                .enumerate()
                .filter(|(i, _)| i + 1 >= from)
                .map(|(_, a)| a * a)
                .sum()
        };
        let lo = sum(3);
        if lo <= 0.0 {
            0.0
        } else {
            sum(13) / lo
        }
    }

    #[test]
    fn soft_clip_suppresses_high_harmonics_far_more_than_hard_clip() {
        // 这是软削波存在的**真正理由**：不是 L2 偏差更小（见下一个测试），
        // 而是高次谐波显著更低。硬削波的一阶导数在削波点跳变 → 包络 ~1/n²；
        // 软膝使跳变推迟到二阶导数 → 包络衰减更快。
        const SR: f64 = 48_000.0;
        const F0: f64 = 200.0;
        // 4800 样本 = 20 个整周期，避免频谱泄漏。
        let n = 4800usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * F0 * i as f64 / SR).sin() as f32)
            .collect();
        let loud: Vec<f32> = signal.iter().map(|v| v * 1.5).collect();

        let soft: Vec<f32> = loud
            .iter()
            .map(|x| soft_clip_sample(*x, DEFAULT_SOFT_CLIP_KNEE))
            .collect();
        let hard: Vec<f32> = loud.iter().map(|x| x.clamp(-1.0, 1.0)).collect();

        let r_soft = high_harmonic_ratio(&harmonic_amplitudes(&soft, F0, SR, 41));
        let r_hard = high_harmonic_ratio(&harmonic_amplitudes(&hard, F0, SR, 41));

        // 实测（1.5× 过载、膝 0.7）：soft ≈ 2.9e-4，hard ≈ 2.3e-3，比值 ≈ 0.13。
        // 取 0.35 留出充裕余量，同时仍能捕获"退化为硬削波"的回归。
        assert!(
            r_soft < r_hard * 0.35,
            "soft clip should suppress high harmonics: soft={r_soft:.3e} hard={r_hard:.3e}"
        );
    }

    #[test]
    fn soft_clip_l2_deviation_is_comparable_to_hard_clip_at_realistic_overload() {
        // 诚实的取舍记录：软削波**不**保证更小的 L2 偏差。它更早开始压缩，
        // 因此在轻度过载（≈1.05×）下偏差反而大于硬削波；但在 ≥1.2× 的
        // 实际过载区间两者相当（实测 1.5× 时 18.76 vs 18.35）。
        // 本测试锁定这一行为，防止将来误以为"软削波 = 更小的误差能量"。
        const SR: f64 = 48_000.0;
        const F0: f64 = 200.0;
        let n = 4800usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * F0 * i as f64 / SR).sin() as f32)
            .collect();
        let loud: Vec<f32> = signal.iter().map(|v| v * 1.5).collect();

        let soft: Vec<f32> = loud
            .iter()
            .map(|x| soft_clip_sample(*x, DEFAULT_SOFT_CLIP_KNEE))
            .collect();
        let hard: Vec<f32> = loud.iter().map(|x| x.clamp(-1.0, 1.0)).collect();

        let l2 = |y: &[f32]| -> f64 {
            y.iter()
                .zip(loud.iter())
                .map(|(a, b)| {
                    let d = (*a - *b) as f64;
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        };
        let e_soft = l2(&soft);
        let e_hard = l2(&hard);
        assert!(
            e_soft < e_hard * 1.15 && e_soft > e_hard * 0.5,
            "soft clip deviation should be comparable to hard clip: soft={e_soft:.2} hard={e_hard:.2}"
        );
    }
}
