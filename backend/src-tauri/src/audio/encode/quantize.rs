//! f32 → 整数量化，WAV（i16/i24）与 FLAC 共用；支持可选 TPDF 抖动。
//!
//! 缩放约定：满幅 `[-1, 1] × 2^(bits-1)`，round-half-away-from-zero，
//! 结果钳制到 `[-2^(bits-1), 2^(bits-1)-1]`。与重构前 WAV 16/24 导出行为
//! 基本一致（差异仅在满幅附近的缩放基准 `×32767 → ×32768 后钳制`，
//! 幅度差约 0.0003 dB，听感不可辨）。

use super::DitherMode;

/// xorshift64* PRNG：抖动需要确定性输出（同一工程两次渲染逐样本一致），
/// 因此使用固定种子的本地 PRNG，而非全局熵源。
pub(crate) struct DitherState {
    state: u64,
}

impl DitherState {
    pub fn new(seed: u64) -> Self {
        // 全 0 是 xorshift64* 的不动点，强制非零。
        DitherState { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// `[0, 1)` 均匀分布（24-bit 尾数，远超 1 LSB 抖动所需的分辨率）。
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// 量化单个样本：f32 `[-1, 1]` → `bits` 位有符号整数。
///
/// TPDF 抖动在量化前注入：两个均匀分布之差构成 `[-1, 1)` 三角分布，
/// 峰峰 2 LSB，可将量化噪声去相关化为白噪。
pub fn quantize_sample(value: f32, bits: u32, dither: DitherMode, rng: &mut DitherState) -> i32 {
    let half = (1i64 << (bits - 1)) as f32;
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;

    let mut scaled = value.clamp(-1.0, 1.0) * half;
    if matches!(dither, DitherMode::Tpdf) {
        scaled += rng.next_f32() - rng.next_f32();
    }

    (scaled.round() as i64).clamp(min, max) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_DITHER: DitherMode = DitherMode::None;
    const TPDF: DitherMode = DitherMode::Tpdf;

    #[test]
    fn quantize_without_dither_is_deterministic_and_clamped() {
        let mut rng = DitherState::new(1);
        let mut rng2 = DitherState::new(1);
        for bits in [16u32, 24] {
            let a = quantize_sample(0.25, bits, NO_DITHER, &mut rng);
            let b = quantize_sample(0.25, bits, NO_DITHER, &mut rng2);
            assert_eq!(a, b, "无抖动必须逐样本确定");

            let max = quantize_sample(2.0, bits, NO_DITHER, &mut rng);
            let min = quantize_sample(-2.0, bits, NO_DITHER, &mut rng);
            assert_eq!(max, (1i64 << (bits - 1)) as i32 - 1);
            assert_eq!(min, -(1i64 << (bits - 1)) as i32);
        }
    }

    #[test]
    fn quantize_rounds_half_away_from_zero() {
        let mut rng = DitherState::new(7);
        // 0.5 LSB 处四舍五入远离零（0.25 × 32768 = 8192 恰为整数，无舍入；
        // 用 8192.5 的等价值 0.25000763… 验证向上舍入）。
        let v = (8192.5f64 / 32768.0) as f32;
        assert_eq!(quantize_sample(v, 16, NO_DITHER, &mut rng), 8193);
        let v_neg = (-8192.5f64 / 32768.0) as f32;
        assert_eq!(quantize_sample(v_neg, 16, NO_DITHER, &mut rng), -8193);
    }

    #[test]
    fn tpdf_dither_stays_within_one_lsb_and_is_seeded() {
        let mut rng = DitherState::new(42);
        let mut rng2 = DitherState::new(42);
        let v = 0.1234f32;
        let expected: Vec<i32> = (0..4096)
            .map(|_| quantize_sample(v, 16, TPDF, &mut rng))
            .collect();
        let replay: Vec<i32> = (0..4096)
            .map(|_| quantize_sample(v, 16, TPDF, &mut rng2))
            .collect();
        assert_eq!(expected, replay, "同种子抖动序列必须可复现");

        // 无抖动的基准量化值。
        let mut quiet = DitherState::new(1);
        let base = quantize_sample(v, 16, NO_DITHER, &mut quiet);
        let half = 1i64 << 15;
        for q in &expected {
            let diff = (*q as i64) - (base as i64);
            assert!(
                diff >= -1 && diff <= 1,
                "TPDF 抖动偏移必须落在 ±1 LSB 内（得到 {diff}）"
            );
        }
        // 恒定输入下抖动应产生不止一个量化电平（去相关生效）。
        let unique: std::collections::HashSet<i32> = expected.into_iter().collect();
        assert!(unique.len() >= 2, "抖动应在量化台阶间产生变化");
        let _ = half;
    }

    #[test]
    fn dither_state_seed_zero_is_valid() {
        let mut rng = DitherState::new(0);
        // 种子 0 被强制置 1，输出不得恒为 0。
        assert_ne!(rng.next_u64(), 0);
    }
}
