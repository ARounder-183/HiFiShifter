//! 工程参数帧栅格：秒 ⇄ 帧的**唯一**换算点。
//!
//! ## 栅格是点采样，不是格子
//!
//! 帧 `k` 表示 `t = k · fp / 1000` 秒处的**样本**。渲染端取曲线时按
//! `i0 = floor(t · 1000 / fp)`、`frac = t · 1000 / fp − i0` 在 `i0` 与 `i0+1` 之间
//! 线性插值（`renderer/utils.rs:20-29`、`renderer/common_params.rs:247-253`）。
//! 因此"时刻 t 由哪一帧代表"的答案是**最近**的那一帧 —— [`frame_at_time`] 的
//! `round`，不是 `floor`。`floor` 给的是"不晚于 t 的那一帧"，系统性偏早半帧。
//!
//! ## 为什么必须只有一个换算点
//!
//! 写侧（`pitch_analysis::schedule` 组装 `pitch_orig`、`dyn_analysis` 组装
//! `dyn_orig`）与读侧（导出包络 / 子轨偏移 / 参考线 / 渲染器切片）各写一份取整，
//! 迟早一处 `floor` 一处 `round`。那不是"差几毫秒"，而是**整条曲线平移一帧**：
//! 导出到 REAPER 的 PIT 包络整体晚 5 ms，而参数编辑器的参考线是准的 —— 用户看到的
//! 是"导出的和界面上不一样"，却看不出是差在哪一帧上。
//!
//! 真机现场：`clip.start_sec = 1.0026` 时 `1.0026 × 200 = 200.52`，组装把该片段的
//! 第 0 帧写在工程帧 201，导出却从工程帧 200 读 ⇒ 读到的每个值都属于**前一个**片段帧。
//!
//! ## 两种语义，不要混用
//!
//! - **定位**（[`frame_at_time`] / [`frame_count_for_length`] / [`clip_frame_window`]）：
//!   帧 `k` 与时刻 `k · fp` 对齐。写数据、读数据、切片、参考线、导出全用它。
//!   **非零时长至少占一帧**（片段的起点永远落在某一帧的半帧以内）。
//! - **覆盖**（[`frame_span_covering`] / [`clip_frame_cover`]）：问"哪些帧与这段时间
//!   有交集"，`floor`/`ceil` 才对 —— 用于"这段区间里有没有非默认值""清空这段范围"。
//!
//! 用定位语义去覆盖会漏掉末尾那半帧；用覆盖语义去定位会平移一帧。两者都对，
//! 错的是**拿其中一个去干另一个的活**。

/// 帧周期兜底（毫秒）。`max(0.1)` 同时挡住 `0` 与 `NaN`（Rust 的 `f64::max`
/// 在任一操作数为 NaN 时返回另一个）。
#[inline]
fn period_ms(frame_period_ms: f64) -> f64 {
    frame_period_ms.max(0.1)
}

/// 时刻 → 帧号（**定位**语义：最近的那一帧）。
///
/// 负数与非有限值一律落到帧 0 —— 帧号无负值，而 `as usize` 的饱和转换会把 `-1`
/// 变成 `usize::MAX`（那是"整条曲线跑到工程末尾"级别的破坏）。
pub fn frame_at_time(sec: f64, frame_period_ms: f64) -> usize {
    let frames = (sec * 1000.0) / period_ms(frame_period_ms);
    if !frames.is_finite() || frames <= 0.0 {
        return 0;
    }
    frames.round() as usize
}

/// 时长 → 帧数（**定位**语义：与 [`frame_at_time`] 同一算式）。
///
/// 【为什么不是 `ceil`】组装侧写的就是 `round(时长)` 帧（`schedule.rs` 的
/// `clip_len_frames`）；读侧用 `ceil` 会多读一帧 —— 那一帧属于**下一个**片段
/// （或根本没有数据），表现为导出包络末尾多出一个与内容无关的点。
///
/// 【为什么非零时长至少占一帧】片段的起点永远落在某一帧的**半帧以内**，那一帧就是
/// 它的第一帧（[`frame_at_time`] 的定义）。若把 `round` 得到的 0 直接当成"没有帧"，
/// 就会出现"片段存在、但它的起点那一帧不归它"——读侧于是读 0 帧、写侧也写 0 帧，
/// 而用户点的那一条片段什么都拿不到（真机现场：ARA 里一个 4 样本的极短区间）。
/// 所以规则是：**非零时长 ⇒ 至少 1 帧**；零时长 ⇒ 0 帧。
pub fn frame_count_for_length(length_sec: f64, frame_period_ms: f64) -> usize {
    let frames = (length_sec * 1000.0) / period_ms(frame_period_ms);
    if !frames.is_finite() || frames <= 0.0 {
        return 0;
    }
    (frames.round() as usize).max(1)
}

/// 片段在工程栅格上占的帧区间 `[start, end)`（**定位**语义）。
///
/// 这是"写进去的那几帧"与"读回来的那几帧"必须逐帧相同的唯一出处。
pub fn clip_frame_window(start_sec: f64, length_sec: f64, frame_period_ms: f64) -> (usize, usize) {
    let start = frame_at_time(start_sec, frame_period_ms);
    let count = frame_count_for_length(length_sec, frame_period_ms);
    (start, start.saturating_add(count))
}

/// 帧号 → 时刻（秒）。与 [`frame_at_time`] 互逆到半帧以内。
pub fn time_at_frame(frame: usize, frame_period_ms: f64) -> f64 {
    frame as f64 * period_ms(frame_period_ms) / 1000.0
}

/// 片段窗口的**覆盖并集**：既含与 `[start, start+length)` 有交集的帧，也含片段
/// 自己的那几帧。
///
/// 【为什么"覆盖"单独用不够】覆盖终点由 `ceil((s+L)·k)` 算出，而 `(s+L)` 在浮点里
/// **不等于** `s·k + L·k`。两端各差半个帧时（`round(s)` 进位、`round(L)` 不进位），
/// 覆盖终点会比定位终点 `round(s·k)+round(L·k)` **少一帧** —— "按覆盖清空"于是把片段
/// 自己的最后一帧留下残渣（用户看到"移动之后旧位置还有一小截曲线删不掉"）。
///
/// 用于清空 / 搬走一段参数线：宁可多清一帧，也不能留残渣。
pub fn clip_frame_cover(start_sec: f64, length_sec: f64, frame_period_ms: f64) -> (usize, usize) {
    let (exact_start, exact_end) = clip_frame_window(start_sec, length_sec, frame_period_ms);
    let end_sec = start_sec + length_sec;
    let (cover_start, cover_end) = frame_span_covering(start_sec, end_sec, frame_period_ms);
    (cover_start.min(exact_start), cover_end.max(exact_end))
}

/// 时刻区间 `[start_sec, end_sec)` → 与之**有交集**的全部帧（**覆盖**语义）。
///
/// 与 [`clip_frame_window`] 的区别只在端点：这里 `floor`/`ceil`，宁可多算一帧也不
/// 漏掉半帧。用于"这段区间里有没有非默认值""清空这段范围"这类**不能留残渣**的操作。
pub fn frame_span_covering(start_sec: f64, end_sec: f64, frame_period_ms: f64) -> (usize, usize) {
    let fp = period_ms(frame_period_ms);
    let to_frame = |sec: f64| -> f64 {
        let frames = (sec * 1000.0) / fp;
        if frames.is_finite() && frames > 0.0 {
            frames
        } else {
            0.0
        }
    };
    let (start, end) = (to_frame(start_sec), to_frame(end_sec));
    let start = start.floor() as usize;
    let end = end.ceil() as usize;
    (start, end.max(start))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 【本模块存在的理由】`0.5026 s × 200 = 100.52`：最近的是 101，不晚于它的是 100。
    /// 组装侧取 101；读侧若取 100，整条曲线就早了一帧。
    #[test]
    fn a_time_maps_to_the_nearest_frame_not_the_one_before_it() {
        assert_eq!(frame_at_time(0.5026, 5.0), 101);
        assert_eq!(frame_at_time(0.5023, 5.0), 100);
        // 不钉"正好半帧"的行为：`0.5025 × 200` 在二进制里落在 100.4999…，
        // 断言它等于 101 只是在赌浮点表示，不是在设计。
    }

    /// 帧号与时刻互逆到半帧以内，且再映射一次仍是同一帧（幂等）。
    #[test]
    fn frame_and_time_round_trip_within_half_a_frame() {
        for fp in [5.0_f64, 10.0, 2.5] {
            let half = fp / 2000.0;
            for step in 0..200 {
                let sec = step as f64 * 0.0137;
                let frame = frame_at_time(sec, fp);
                let back = time_at_frame(frame, fp);
                assert!(
                    (back - sec).abs() <= half + 1e-12,
                    "fp={fp} sec={sec} frame={frame} back={back}"
                );
                assert_eq!(frame_at_time(back, fp), frame, "fp={fp} sec={sec}");
            }
        }
    }

    /// 片段窗口必须等于组装侧写进去的那几帧：起点 `round`、长度 `round`。
    #[test]
    fn a_clip_window_is_exactly_what_the_writer_wrote() {
        let fp = 5.0;
        // 起点 1.0026 s ⇒ 帧 201（不是 200）；长度 1.0 s ⇒ 200 帧（不是 201）。
        let (start, end) = clip_frame_window(1.0026, 1.0, fp);
        assert_eq!(start, 201);
        assert_eq!(end - start, 200);
        // 长度 1.0023 s ⇒ `round` 200、`ceil` 201：读侧多读的那一帧不属于本片段。
        assert_eq!(frame_count_for_length(1.0023, fp), 200);
    }

    /// 非零时长**至少占一帧**：片段的起点永远落在某一帧的半帧以内，那一帧就是它的
    /// 第一帧。只有零时长才不占帧。
    #[test]
    fn a_non_zero_length_always_occupies_at_least_one_frame() {
        // 4 样本 @44.1kHz = 0.09 ms，远短于 5 ms 一帧 —— 真机现场就是这种极短区间。
        let four_samples = 4.0 / 44_100.0;
        assert_eq!(frame_count_for_length(four_samples, 5.0), 1);
        let (start, end) = clip_frame_window(0.0, four_samples, 5.0);
        assert_eq!((start, end), (0, 1));
        assert_eq!(frame_count_for_length(0.0, 5.0), 0, "零时长不占帧");
        assert_eq!(frame_count_for_length(-1.0, 5.0), 0);
    }

    /// 覆盖起点永远不晚于定位起点（`floor ≤ round`）。
    #[test]
    fn the_covering_span_starts_at_or_before_the_placement_start() {
        let fp = 5.0;
        for step in 0..50 {
            let start_sec = step as f64 * 0.0131;
            let length_sec = 0.37 + step as f64 * 0.0043;
            let (a, _) = clip_frame_window(start_sec, length_sec, fp);
            let (c, _) = frame_span_covering(start_sec, start_sec + length_sec, fp);
            assert!(c <= a, "covering start {c} > placement {a}");
        }
    }

    /// 【为什么清空范围要用 [`clip_frame_cover`] 而不是 [`frame_span_covering`]】
    /// 覆盖终点由 `ceil((s+L)·k)` 算出，而 `(s+L)` 在浮点里**不等于** `s·k + L·k`：
    /// 两端各差半帧时，覆盖终点可能比定位终点少一帧 —— 清空就会留下片段自己最后
    /// 一帧的残渣。并集窗口把两条都兜住。
    #[test]
    fn the_cover_window_contains_both_the_placement_and_the_covering_window() {
        let fp = 5.0;
        for step in 0..200 {
            let start_sec = step as f64 * 0.0131;
            let length_sec = 0.37 + step as f64 * 0.0043;
            let (a, b) = clip_frame_window(start_sec, length_sec, fp);
            let (c, d) = frame_span_covering(start_sec, start_sec + length_sec, fp);
            let (u, v) = clip_frame_cover(start_sec, length_sec, fp);
            assert!(u <= a && u <= c, "cover start {u} misses {a}/{c}");
            assert!(v >= b && v >= d, "cover end {v} misses {b}/{d}");
        }
    }

    /// 【耦合】`mixdown` 把导出偏移写回 `pitch_edit` 时用的起点（[`frame_at_time`]）必须
    /// 等于导出侧读窗口的起点（[`clip_frame_window`] 的 `.0`）。两者各写一份取整就会
    /// 把"这一帧的原值 + 下一帧的偏移量"拼在一起 —— 这条断言把"同一处"钉住。
    #[test]
    fn the_apply_origin_equals_the_export_window_start() {
        for fp in [5.0_f64, 10.0, 2.5] {
            for step in 0..120 {
                let start_sec = step as f64 * 0.0173;
                let length_sec = 0.41 + step as f64 * 0.0037;
                assert_eq!(
                    frame_at_time(start_sec, fp),
                    clip_frame_window(start_sec, length_sec, fp).0,
                    "fp={fp} start={start_sec}"
                );
            }
        }
    }

    /// 负数 / NaN / 0 帧周期都不能产出巨大的帧号或 panic。
    #[test]
    fn hostile_inputs_clamp_to_the_first_frame() {
        assert_eq!(frame_at_time(-5.0, 5.0), 0);
        assert_eq!(frame_at_time(f64::NAN, 5.0), 0);
        assert_eq!(frame_at_time(f64::INFINITY, 5.0), 0);
        assert_eq!(frame_count_for_length(-1.0, 5.0), 0);
        // 帧周期的兜底沿用全仓库的 `.max(0.1)`（NaN 被 max 吃掉 ⇒ 0.1 ms），
        // 所以 1 s 会算出 10000 帧 —— 荒谬但不 panic，且与其它调用点口径一致。
        assert_eq!(frame_count_for_length(1.0, f64::NAN), 10_000);
        assert_eq!(frame_at_time(1.0, 0.0), 10_000, "0 帧周期按 0.1ms");
        assert_eq!(frame_span_covering(-1.0, -0.5, 5.0), (0, 0));
    }
}
