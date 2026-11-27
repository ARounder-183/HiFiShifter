//! Clip / Take / Track 增益的线性值域与分层语义。
//!
//! 【唯一权威】模型层只存线性值：0 是唯一的静音表示，dB 只出现在文本与
//! 输入边界。上界 16 对应 REAPER item 增益上限 +24.08 dB；两层相乘后的
//! 上限由 GAIN_PRODUCT_MAX 单独兜底，不再把乘积钳回单层上限。
//!
//! 【为什么 take 要单独一个钳制入口】REAPER 用负的 take volume 表达极性
//! 反转：-0.5 不是静音，而是半幅反相。用非负的 clamp_gain 处理 take
//! 会把反相 take 静音，波形与听感同时消失。
//!
//! 【前端镜像】frontend/src/utils/gainDb.ts 必须与本文件的常量、边界逐条
//! 一致；两边的数值差异会直接表现为“界面显示的值不是实际渲染的值”。

/// 线性增益下界（0 = −inf dB，静音）。
pub const GAIN_MIN: f32 = 0.0;

/// 线性增益上界（16 = +24.08 dB，与 REAPER 的 item 增益上限一致）。
pub const GAIN_MAX: f32 = 16.0;

/// 显示下界（REAPER 读数下限）：低于它显示 -inf。
pub const GAIN_FLOOR_DB: f64 = -150.0;

/// 显示下界对应的线性值：10^(−150/20) ≈ 3.1622777e−8。
///
/// 低于它的正值统一归零，避免出现“界面显示 −inf、混音却还有值”的僵尸区。
pub const GAIN_FLOOR: f32 = 3.162_277_7e-8;

/// 两层同时提升时的乘积上限（16 × 16）。
///
/// 它只用于防溢出，**不是**对单层的第二次钳制：item +12 dB 与 take +12 dB
/// 叠加后应当得到 +24 dB（16×），而不是被压回 +24.08 dB 的单层上限。
pub const GAIN_PRODUCT_MAX: f32 = GAIN_MAX * GAIN_MAX;

/// 哪一方负责施加增益层。
///
/// 独立 App / 导出自己乘 item × take × track；ARA 插件里这三层都由宿主施加，
/// 插件渲染时必须保持干声（1.0），否则同一个旋钮会被乘两遍。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GainScope {
    /// 独立 App / 导出：所有增益层由本进程施加。
    #[default]
    AppOwned,
    /// ARA 插件：所有增益层由宿主施加；本进程只显示与写回。
    HostOwned,
}

/// 把非负层（item / track）钳进 [0, 16]。
///
/// - NaN → 1.0（不增不减，坏字段不该让整段静音或爆音）；
/// - ≤ 0 或低于显示下界 → 0（唯一的静音表示）；
/// - +inf → 16；−inf → 0。
pub fn clamp_gain(value: f32) -> f32 {
    if value.is_nan() {
        return 1.0;
    }
    if value <= 0.0 || value < GAIN_FLOOR {
        return 0.0;
    }
    value.min(GAIN_MAX)
}

/// 把 take 层钳进 [−16, 16]，**保留符号**（负值 = 极性反转）。
///
/// 幅度低于显示下界时归零；NaN → 1.0。符号不参与钳制方向：−32 → −16。
pub fn clamp_take_gain(value: f32) -> f32 {
    if value.is_nan() {
        return 1.0;
    }
    let magnitude = value.abs();
    if magnitude < GAIN_FLOOR {
        return 0.0;
    }
    value.clamp(-GAIN_MAX, GAIN_MAX)
}

/// 线性 → dB；静音返回 None（不是 −150，避免把“没有声音”说成“很小声”）。
pub fn gain_to_db(linear: f32) -> Option<f64> {
    let gain = clamp_gain(linear);
    if gain <= 0.0 {
        None
    } else {
        // f32 的 GAIN_FLOOR 与 f64 的 −150 端点不是逐位同值；把结果抬到
        // 显示下界，保证“最小可表示值”不会被自己的 dB 边界判成静音。
        Some((20.0 * f64::from(gain).log10()).max(GAIN_FLOOR_DB))
    }
}

/// dB → 线性；低于 −150 dB 或非有限值 → 0（静音）。
///
/// 注意是**严格小于**：−150 dB 本身仍是可表示的最小非零值。
pub fn db_to_gain(db: f64) -> f32 {
    if !db.is_finite() || db < GAIN_FLOOR_DB {
        0.0
    } else {
        (10f64.powf(db / 20.0) as f32).min(GAIN_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_upper_bound_is_the_reaper_item_gain_limit() {
        let db = 20.0 * GAIN_MAX.log10();
        assert!(
            (db - 24.08).abs() < 0.01,
            "GAIN_MAX 应等于 +24.08 dB，实际 {db:.3} dB"
        );
    }

    #[test]
    fn silence_is_the_only_zero() {
        assert_eq!(clamp_gain(0.0), 0.0);
        assert_eq!(clamp_gain(-0.0), 0.0);
        assert_eq!(clamp_gain(1.0), 1.0);
        assert_eq!(clamp_gain(GAIN_MAX), GAIN_MAX);
    }

    #[test]
    fn sub_floor_values_canonicalise_to_silence() {
        assert_eq!(clamp_gain(GAIN_FLOOR * 0.5), 0.0);
        assert!(clamp_gain(GAIN_FLOOR) > 0.0);
        assert_eq!(db_to_gain(GAIN_FLOOR_DB - 0.001), 0.0);
        assert!(db_to_gain(GAIN_FLOOR_DB) > 0.0);
    }

    #[test]
    fn take_gain_keeps_the_polarity_sign() {
        assert_eq!(clamp_take_gain(-0.25), -0.25);
        assert_eq!(clamp_take_gain(-32.0), -GAIN_MAX);
        assert_eq!(clamp_take_gain(32.0), GAIN_MAX);
        assert_eq!(clamp_take_gain(-0.0), 0.0);
    }

    #[test]
    fn take_gain_rejects_out_of_range_magnitudes() {
        assert_eq!(clamp_take_gain(f32::INFINITY), GAIN_MAX);
        assert_eq!(clamp_take_gain(f32::NEG_INFINITY), -GAIN_MAX);
        assert_eq!(clamp_take_gain(GAIN_FLOOR * 0.5), 0.0);
    }

    #[test]
    fn db_roundtrip_is_lossless_above_the_floor() {
        for linear in [1.0f32, 0.5, 2.0, GAIN_MAX, GAIN_FLOOR] {
            let db = gain_to_db(linear).unwrap();
            let back = db_to_gain(db);
            assert!(
                (back - linear).abs() <= linear * 1e-5,
                "{linear} -> {db} -> {back}"
            );
        }
        assert_eq!(gain_to_db(0.0), None);
    }

    #[test]
    fn nan_falls_back_to_unity_on_every_layer() {
        assert_eq!(clamp_gain(f32::NAN), 1.0);
        assert_eq!(clamp_take_gain(f32::NAN), 1.0);
        assert_eq!(db_to_gain(f64::NAN), 0.0);
    }
}
