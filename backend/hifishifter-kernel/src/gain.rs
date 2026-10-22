//! Clip 与轨道增益的**线性值域**，以及把任意输入钳进它的唯一入口。
//!
//! # 为什么值域是 [0, 16]
//!
//! 上界对应 REAPER 的 item 增益上限 **+24.08 dB**（`10^(24.08/20) = 15.998`，取整为 16）。
//! 此前这里是 4.0（= +12.04 dB），而 `reaper_import` 里那段 `convert_volume` 的注释
//! **早就写着**"item trim 与 take volume 相乘后最高可达 +24 dB（16x）" ——
//! 注释承认的上界与钳制不一致，于是 REAPER 工程里一切超过 +12 dB 的 item 在导入后
//! 都被静默压小 12 dB，写回宿主时又会被 `host_edit` 拒绝。
//!
//! 下界 0 对应 −inf dB（静音）：线性增益没有负数，反相是另一回事（REAPER 用负的
//! take volume 表达反相，导入时取绝对值保留响度）。
//!
//! # 为什么钳制必须只有一个入口
//!
//! 这个值域此前以字面量 `4.0` 散落在十一个位置（内核的状态写入、混音、动态分析、
//! 播放快照，以及 app 侧的 REAPER 导入 / 导出、插件侧的宿主写回）。改一处漏一处的
//! 后果不是崩溃，而是**同一个增益在不同路径上被钳到不同值** —— 例如界面显示 +24 dB、
//! 混音按 16x 算，而写回宿主时被拒。收成一个常量与一个函数之后，"哪个值域"这件事
//! 只有一个答案。
//!
//! 【前端有一份镜像】`frontend/src/utils/gainDb.ts`。它同样只做"同一个值域的另一种
//! 表达"，两边的**数值**必须一致 —— 与 `paramRanges.ts` / `DYN_SILENCE_FLOOR` 的
//! 既有约定相同（前端需要同步、同步读取的路径不能等一次 IPC）。

/// 线性增益下界（0 = −inf dB，静音）。
pub const GAIN_MIN: f32 = 0.0;

/// 线性增益上界（16 = +24.08 dB，与 REAPER 的 item 增益上限一致）。
pub const GAIN_MAX: f32 = 16.0;

/// 把任意输入钳进 [`GAIN_MIN`]..=[`GAIN_MAX`]。
///
/// 【为什么只对 NaN 特判、而 ±inf 照常钳制】`f32::clamp` 本身对 ±inf 的行为就是对的
/// （−inf → 下界、+inf → 上界），唯一坏掉的是 NaN —— 它对 NaN 返回 NaN，而 NaN 增益会
/// 一路传播到混音缓冲（整段输出变成 NaN 静音，且没有任何报错）。
///
/// NaN 回退到 **1.0（不增不减）** 而不是 0.0（静音）：一个坏字段不该让整个 clip 变成
/// 静音，更不该让它变成爆音。
pub fn clamp_gain(value: f32) -> f32 {
    if value.is_nan() {
        1.0
    } else {
        value.clamp(GAIN_MIN, GAIN_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 上界必须真的是 REAPER 的 +24.08 dB。
    ///
    /// 【为什么钉住数值而不是"某个大数"】这个常量存在的全部理由就是与宿主对齐；
    /// 它一旦被改回 4.0，导入就会重新丢掉 12 dB，而界面上什么都不会报错。
    #[test]
    fn the_upper_bound_is_the_reaper_item_gain_limit() {
        let db = 20.0 * GAIN_MAX.log10();
        assert!(
            (db - 24.08).abs() < 0.01,
            "GAIN_MAX 应等于 +24.08 dB，实际 {db:.3} dB"
        );
    }

    #[test]
    fn clamping_covers_the_ends_and_rejects_non_finite_values() {
        assert_eq!(clamp_gain(1.0), 1.0);
        assert_eq!(clamp_gain(-3.0), GAIN_MIN);
        assert_eq!(clamp_gain(1e9), GAIN_MAX);
        // NaN ⇒ 不增不减（不能让它传播成 NaN 缓冲，也不能变成静音）；
        // ±inf ⇒ 照常钳到端点（`f32::clamp` 本身就是这样，只有 NaN 需要特判）。
        assert_eq!(clamp_gain(f32::NAN), 1.0);
        assert_eq!(clamp_gain(f32::INFINITY), GAIN_MAX);
        assert_eq!(clamp_gain(f32::NEG_INFINITY), GAIN_MIN);
    }
}
