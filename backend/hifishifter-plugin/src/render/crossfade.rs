//! 两条信号之间的**等功率**过渡。
//!
//! # 为什么单独一个模块
//!
//! "把 A 淡成 B"在插件里有**两处**用途，而它们此前各写了一遍：
//!   * `snapshot.rs::edge_gain` —— 快照覆盖边界（我们算出来的内容 ↔ 宿主原声）；
//!   * `extension.rs::apply_output_declick` —— 输出源接缝（Raw ↔ Processed）。
//!
//! 两处都写着"等功率"，但只有前者真的用了 `cos`/`sin`；后者是线性的
//! `old*(1-t) + current*t`。对两条**互不相关**的信号，线性混合在中点的能量只有
//! −6 dB —— 那不是一次淡化，那是一次可听见的能量凹陷（用户描述为"咔"）。
//!
//! 把数学收敛到一个函数里，两处就不再有"注释说等功率、代码写线性"这种漂移。

/// 等功率交叉淡化的两个增益：`(旧, 新)`，满足 `旧² + 新² = 1`。
///
/// `t` = 进度（0 = 完全是旧信号，1 = 完全是新信号），超出区间会被夹住。
///
/// 【为什么是 cos/sin 而不是 `sqrt`】两者等价（`sqrt(1-t²)` 与 `cos(t·π/2)` 形状相同），
/// 但 `cos`/`sin` 在两端**导数连续**：淡化开始时旧信号的下降斜率与结束时新信号的
/// 上升斜率都是 0，于是接缝处不会有"折点"（折点在听感上是一次轻微的爆音）。
/// 也正因为如此，纯淡入（旧信号恒为 0）用它也是对的：`sin(t·π/2)` 是一条两端平缓的
/// 单调斜坡。
#[inline]
pub(crate) fn equal_power(t: f32) -> (f32, f32) {
    let t = t.clamp(0.0, 1.0);
    let angle = t * std::f32::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 不变量：两个增益的平方和恒为 1（这就是"等功率"的定义）。
    #[test]
    fn the_two_gains_always_sum_to_unit_power() {
        for step in 0..=100 {
            let (old, new) = equal_power(step as f32 / 100.0);
            assert!(
                (old * old + new * new - 1.0).abs() < 1e-5,
                "t={step} old={old} new={new}"
            );
        }
    }

    /// 两端必须是精确的 0/1：否则接缝处会留下一个固定的电平台阶。
    #[test]
    fn the_endpoints_are_exact() {
        assert_eq!(equal_power(0.0), (1.0, 0.0));
        let (old, new) = equal_power(1.0);
        assert!(old.abs() < 1e-7 && (new - 1.0).abs() < 1e-7, "{old} {new}");
    }

    /// 越界进度被夹住（调用方用整数帧算进度，末尾可能多一帧）。
    #[test]
    fn out_of_range_progress_is_clamped() {
        assert_eq!(equal_power(-1.0), (1.0, 0.0));
        let (old, new) = equal_power(2.0);
        assert!(old.abs() < 1e-7 && (new - 1.0).abs() < 1e-7);
    }

    /// **中点能量必须保持**：两条不相关信号在中点的合成 RMS 与两端相同。
    ///
    /// 【为什么这条是回归测试】线性混合在中点是 `0.5+0.5`，两条不相关信号的
    /// 功率相加是 `0.25+0.25 = 0.5` ⇒ −3 dB（电压 −6 dB 的中点，功率 −3 dB）。
    /// 等功率给出 `0.5+0.5 = 1.0` ⇒ 0 dB。用户听到的差别就是"一次凹陷"与"没有凹陷"。
    #[test]
    fn the_midpoint_keeps_the_power_of_two_uncorrelated_signals() {
        let (old, new) = equal_power(0.5);
        let power = old * old + new * new;
        assert!((power - 1.0).abs() < 1e-5, "power={power}");
        // 线性混合的中点功率是 0.5，这正是要避免的那个凹陷。
        let linear_power = 0.5 * 0.5 + 0.5 * 0.5;
        assert!(power > linear_power * 1.9, "等功率必须显著高于线性");
    }
}
