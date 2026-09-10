/*
 * automation.rs - 参数自动化曲线的**唯一**采样实现（见 P3-1 / D1）。
 *
 * 为什么要有这个模块：
 * 实时混音（`audio_engine/mix.rs`）与离线导出（`audio/mixdown.rs`）各自维护过一份
 * `sample_automation_curve`。两份的插值数学相同，**但入参单位不同** —— 一份吃
 * "时间轴绝对帧 + 采样率"，另一份吃"绝对秒"。这个差异此前只靠注释维持同步，
 * 没有任何编译器或测试约束：任何一处改了插值/钳制规则，另一处不会跟着变，
 * 结果就是"监听与导出音量不一致"这类极难定位的问题（D1 列为最高维护风险）。
 *
 * 现在两份实现合并到此，并把单位差异变成**两个具名入口**：
 * - [`sample_curve_at_sec`]：按绝对秒采样；
 * - [`sample_curve_at_frame`]：按绝对帧采样（内部换算成秒后走同一个核心）。
 *
 * 语义（与合并前的两份实现逐位一致）：
 * - 曲线缺失 / 为空 / 索引非有限 → 返回 `default_value`；
 * - 每项代表 `frame_period_ms` 毫秒，线性插值；
 * - 越界**不外推**：钳到末项后 `frac` 会发散，因此必须把 `frac` 也钳到 [0,1]，
 *   从而实现"持有末值"。这一条曾在离线路径上遗漏，导致导出音量在超出曲线时长
 *   后线性爆表 —— 合并后两个路径共用同一份钳制逻辑。
 */

/// 曲线采样时允许的最小帧周期（毫秒），防御 `frame_period_ms <= 0` 导致的除零。
const MIN_FRAME_PERIOD_MS: f64 = 0.1;

/// 按**绝对秒**采样自动化曲线（线性插值，越界持有末值）。
pub(crate) fn sample_curve_at_sec(
    curve: Option<&[f32]>,
    abs_sec: f64,
    frame_period_ms: f64,
    default_value: f32,
) -> f32 {
    let Some(curve) = curve else {
        return default_value;
    };
    if curve.is_empty() {
        return default_value;
    }

    let fp = frame_period_ms.max(MIN_FRAME_PERIOD_MS);
    // `floor().max(0.0)` 而非直接 `as usize`：负值必须钳到 0，而不是依赖 `as`
    // 的截断-向零行为（两者在负值上语义不同，且语义依赖于浮点细节）。
    //
    // ★ 注意 `f64::max` 的 NaN 语义：`NaN.max(0.0)` 返回 **0.0**（IEEE maxNum），
    //   因此 `abs_sec = NaN` 会被当作位置 0 处理，而不是走到下面的
    //   `is_finite()` 兜底。这是合并前**两条路径共有的**既有行为，此处刻意
    //   保持不变并加测试钉住 —— 若改成"NaN 即返回默认值"，导出与监听的行为会
    //   同时变化，那是独立的行为决策，不该混在去重重构里做。
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return default_value;
    }

    let last = curve.len().saturating_sub(1);
    let i0 = (idx_f.floor().max(0.0) as usize).min(last);
    let i1 = (i0 + 1).min(last);
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;

    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 按**时间轴绝对帧**采样自动化曲线（实时路径以帧为位置单位）。
///
/// 存在的意义就是把"帧 → 秒"这一步固定在这里一次，避免每个调用点各写一遍
/// 而写错采样率（这正是两份实现当初分岔的起点）。
pub(crate) fn sample_curve_at_frame(
    curve: Option<&[f32]>,
    abs_frame: u64,
    sample_rate: u32,
    frame_period_ms: f64,
    default_value: f32,
) -> f32 {
    let abs_sec = abs_frame as f64 / sample_rate.max(1) as f64;
    sample_curve_at_sec(curve, abs_sec, frame_period_ms, default_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 两个入口必须对**同一条曲线、同一个时刻**给出完全相同的结果。
    /// 这是本模块存在的全部理由：单位换算不能引入任何差异。
    #[test]
    fn frame_and_sec_entry_points_agree() {
        let curve = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let sr = 44_100u32;
        let fp = 5.0f64;

        // 覆盖帧内插值、整帧边界、超出曲线末端三类位置。
        for frame in [0u64, 44, 100, 220, 221, 440, 44_100, 441_000] {
            let by_frame = sample_curve_at_frame(Some(&curve), frame, sr, fp, 1.0);
            let abs_sec = frame as f64 / sr as f64;
            let by_sec = sample_curve_at_sec(Some(&curve), abs_sec, fp, 1.0);
            assert!(
                (by_frame - by_sec).abs() < 1e-9,
                "frame {frame}: by_frame={by_frame} by_sec={by_sec}"
            );
        }
    }

    #[test]
    fn missing_or_empty_curve_yields_default() {
        assert_eq!(sample_curve_at_sec(None, 1.0, 5.0, 0.25), 0.25);
        assert_eq!(sample_curve_at_sec(Some(&[]), 1.0, 5.0, 0.25), 0.25);
        assert_eq!(sample_curve_at_frame(None, 100, 44_100, 5.0, 0.75), 0.75);
    }

    /// 超出曲线时长后**持有末值**，绝不外推。
    /// 离线路径曾漏掉 `frac` 钳制，表现为导出音量在曲线用尽后线性爆表。
    #[test]
    fn beyond_curve_end_holds_last_value() {
        let curve = vec![0.0f32, 1.0, 4.0];
        // 曲线仅覆盖 3 项 × 5ms = 10ms；100 秒远超其范围。
        let v = sample_curve_at_sec(Some(&curve), 100.0, 5.0, 0.0);
        assert!((v - 4.0).abs() < 1e-6, "expected held last value 4.0, got {v}");
    }

    /// 曲线只有一项时，任何时刻都返回该值（i0 与 i1 同为 0，frac 无意义）。
    #[test]
    fn single_point_curve_is_constant() {
        let curve = vec![0.5f32];
        assert!((sample_curve_at_sec(Some(&curve), 0.0, 5.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((sample_curve_at_sec(Some(&curve), 1_000.0, 5.0, 0.0) - 0.5).abs() < 1e-6);
    }

    /// 非有限输入与非正帧周期都必须安全降级，不得返回 NaN。
    #[test]
    fn non_finite_inputs_and_periods_are_safe() {
        let curve = vec![1.0f32, 2.0];
        // NaN 与 -∞ 都会被 `f64::max` 吞成 0.0 → 当作位置 0 → 取首项
        //（既有行为，见实现注释）。
        assert_eq!(sample_curve_at_sec(Some(&curve), f64::NAN, 5.0, 0.123), 1.0);
        assert_eq!(
            sample_curve_at_sec(Some(&curve), f64::NEG_INFINITY, 5.0, 0.123),
            1.0
        );
        // +∞ 保留为无穷 → idx_f 非有限 → 返回默认值。
        assert_eq!(
            sample_curve_at_sec(Some(&curve), f64::INFINITY, 5.0, 0.123),
            0.123
        );
        // fp = 0 → 使用下限 0.1ms，仍然有限且落在曲线范围内。
        let v = sample_curve_at_sec(Some(&curve), 0.0, 0.0, 0.0);
        assert!(v.is_finite(), "got {v}");
        // 负秒数被钳到 0。
        assert!((sample_curve_at_sec(Some(&curve), -5.0, 5.0, 0.0) - 1.0).abs() < 1e-6);
    }
}
