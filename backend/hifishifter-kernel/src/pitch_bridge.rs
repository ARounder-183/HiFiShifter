//! 参数线 ⇄ 宿主包络的**纯数学**：时间域换算、包络取值、断点简化。
//!
//! 【为什么单独一个模块】同一个换算出现三次：独立 App 导出 `.rpp`、独立 App 导入
//! `.rpp`、以及插件与 ARA 宿主实时互转。三处各写一份的直接后果是"同一段曲线在
//! 导出文件里和插件里位置不同"—— 而那是最难被察觉的一类 bug：听起来只是"偏了一点"。
//! 把时间域换算与断点简化收到这里，三处都只调这一个来源。
//!
//! 【时间域只有一个换算点】宿主包络的横轴 u 是 **take 的播放时间 × 速率**：
//!
//! <center>u = (t − item_start) × |rate|</center>
//!
//! u = 0 落在 take 的**入点**（即 `SOFFS` 所在处），沿播放方向增长；媒体文件时间
//! 是 `SOFFS + u`（倒放 take 沿反方向减少）。
//!
//! 【为什么不是媒体时间】真机渲染实测（REAPER 7.85，
//! `docs/2026-11-17-take-envelope-axis-findings.md`）：同一份 take 几何上，把点写在
//! `u = SOFFS + offset` 上**一个半音都不生效**，写在 `u = offset` 上立刻出现 +12
//! 半音的斜坡；`STARTPOS` / `MODE`（倒放）/ `LOOP` 同样不进坐标，改 `SOFFS` 也不会
//! 让宿主重定基线（`PT` 逐字保留）。所以任何调用点都**不许**再补源起点 —— 多一个
//! `SOFFS` 就是整条曲线平移，听起来是"整个片段跑调"。
//!
//! 倒放 take 也一样 —— u 是**播放顺序**的时间，方向由 SECTION 表达，不改变坐标域。
//! 因此本模块只提供 [offset_to_take_u] / [take_u_to_offset] 一对函数，任何调用点都
//! 不许就地再乘一次速率。

use serde::{Deserialize, Serialize};

/// 音高断点简化的容差（半音）。
///
/// 【为什么是 0.02】逐帧 5 ms 写点，三分钟片段就是 36000 个点 —— REAPER 的包络
/// 编辑器会卡死。0.02 半音 = 2 音分，低于人耳在正常语境下的分辨阈值，同时把
/// 一条平滑曲线压到几十个点。独立 App 的导出与插件的桥接共用这一个数，
/// 否则"导出听起来对、插件里听起来不对"。
pub const SIMPLIFY_TOLERANCE_PITCH: f64 = 0.02;
/// 音量断点简化的容差（倍率）。
pub const SIMPLIFY_TOLERANCE_VOLUME: f64 = 0.005;
/// 声像断点简化的容差（−1..1）。
pub const SIMPLIFY_TOLERANCE_PAN: f64 = 0.005;

/// 宿主包络上的一个点。
///
/// `time` 是**包络自己的时间域**（take 播放时间 × 速率 u，秒；u = 0 在入点，
/// 见模块文档），`value` 是该点的值（音高包络里是半音）。两者都可能是负的：
/// u 在左延伸 item 上确实为负。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnvelopePoint {
    pub time: f64,
    pub value: f64,
}

impl EnvelopePoint {
    pub fn new(time: f64, value: f64) -> Self {
        Self { time, value }
    }
}

/// 工程时间相对 item 起点的偏移 → take 媒体时间 u。
pub fn offset_to_take_u(offset_sec: f64, rate: f64) -> f64 {
    let rate = if rate.is_finite() && rate.abs() > 1e-9 {
        rate.abs()
    } else {
        1.0
    };
    offset_sec * rate
}

/// take 媒体时间 u → 工程时间相对 item 起点的偏移。
pub fn take_u_to_offset(u: f64, rate: f64) -> f64 {
    let rate = if rate.is_finite() && rate.abs() > 1e-9 {
        rate.abs()
    } else {
        1.0
    };
    u / rate
}

/// 在包络上线性插值取值。
///
/// 【为什么区间外取端点值而不是 0】REAPER 的包络在首末点之外**保持**端点值
/// （没有点 = 默认值，有点 = 端点 hold）。按 0 处理会让"整段 +2 半音"在
/// 第一点之前变成 0，用户听到的是开头一跳。
pub fn sample_envelope(points: &[EnvelopePoint], u: f64) -> f64 {
    if points.is_empty() || !u.is_finite() {
        return 0.0;
    }
    let index = points.partition_point(|point| point.time < u);
    if index == 0 {
        return points[0].value;
    }
    if index >= points.len() {
        return points[points.len() - 1].value;
    }
    let (t0, v0) = (points[index - 1].time, points[index - 1].value);
    let (t1, v1) = (points[index].time, points[index].value);
    if (t1 - t0).abs() < 1e-12 {
        return v0;
    }
    let fraction = ((u - t0) / (t1 - t0)).clamp(0.0, 1.0);
    v0 + (v1 - v0) * fraction
}

/// 逐帧偏移 → 包络采样点（帧 0 = item 起点）。
pub fn pitch_samples(offsets: &[f32], frame_period_ms: f64) -> Vec<(f64, f64)> {
    let period = if frame_period_ms.is_finite() && frame_period_ms > 0.0 {
        frame_period_ms
    } else {
        5.0
    };
    offsets
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 * period / 1000.0, *value as f64))
        .collect()
}

/// 逐帧偏移 → 宿主 take 音高包络的断点。
///
/// 三步：采样 → 断点简化 → 换到 u 域，并**强制两端点**（u = 0 与 u = 长度 × 速率）。
/// 【为什么端点必须强制】REAPER 的包络在首个点之前/末个点之后是 hold 语义；
/// 不强制端点就会让"片段开头那一段"和"结尾那一段"继承邻居的值，跨 item 边界
/// 尤其明显（同一轨上前后两个片段会互相污染）。
pub fn build_take_pitch_points(
    offsets: &[f32],
    frame_period_ms: f64,
    item_length_sec: f64,
    rate: f64,
) -> Vec<EnvelopePoint> {
    let samples = pitch_samples(offsets, frame_period_ms);
    let range_end = item_length_sec.max(1e-6);
    let breakpoints = simplify_breakpoints(&samples, 0.0, range_end, 0.0, SIMPLIFY_TOLERANCE_PITCH);
    breakpoints
        .into_iter()
        .map(|(offset, value)| EnvelopePoint::new(offset_to_take_u(offset, rate), value))
        .collect()
}

/// 把逐帧采样压成折线断点。
///
/// 逐帧 5 ms 写点，三分钟片段就是 36000 个点 —— 包络编辑器会卡死。这里的约定：
///
/// - **全部落在容差内等于默认值** ⇒ 空结果（不写包络，保持宿主原状）；
/// - **端点强制**：恒写 `range_start` 与 `range_end` 两端点（不得依赖 hold 语义
///   跨 Item / 段边界）；
/// - **平坦非默认 → 恰好 2 点；斜线 → 仅折点**。
pub fn simplify_breakpoints(
    samples: &[(f64, f64)],
    range_start: f64,
    range_end: f64,
    default_value: f64,
    tolerance: f64,
) -> Vec<(f64, f64)> {
    if samples.is_empty() || range_end <= range_start {
        return Vec::new();
    }
    if samples
        .iter()
        .all(|(_, value)| (value - default_value).abs() <= tolerance)
    {
        return Vec::new();
    }
    if samples.len() == 1 {
        let value = samples[0].1;
        let first = samples[0].0.clamp(range_start, range_end);
        return vec![(first, value), (range_end, value)];
    }
    // O(n) 快速通道：恒平坦 / 全程线性（最常见形态）直接给两端点。
    {
        let (t0, v0) = samples[0];
        let (tn, vn) = samples[samples.len() - 1];
        let span = tn - t0;
        let slope = if span > 1e-12 { (vn - v0) / span } else { 0.0 };
        let mut flat = true;
        let mut linear = true;
        for (t, v) in samples.iter() {
            if (v - v0).abs() > tolerance {
                flat = false;
            }
            if (v - (v0 + slope * (t - t0))).abs() > tolerance {
                linear = false;
            }
            if !flat && !linear {
                break;
            }
        }
        if flat || linear {
            let first_t = t0.max(range_start);
            let last_t = range_end.max(first_t);
            return vec![
                (first_t, v0 + slope * (first_t - t0)),
                (last_t, v0 + slope * (last_t - t0)),
            ];
        }
    }

    let value_at = |t: f64| -> f64 {
        let index = samples.partition_point(|sample| sample.0 < t);
        if index == 0 {
            return samples[0].1;
        }
        if index >= samples.len() {
            return samples[samples.len() - 1].1;
        }
        let (t0, v0) = samples[index - 1];
        let (t1, v1) = samples[index];
        if (t1 - t0).abs() < 1e-12 {
            return v0;
        }
        let fraction = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        v0 + (v1 - v0) * fraction
    };

    let line_fits = |a: usize, f: usize| -> bool {
        let (ta, va) = samples[a];
        let (tf, vf) = samples[f];
        let span = tf - ta;
        if span <= 1e-12 {
            return (vf - va).abs() <= tolerance;
        }
        let slope = (vf - va) / span;
        samples[a + 1..f]
            .iter()
            .all(|(t, v)| (v - (va + slope * (t - ta))).abs() <= tolerance)
    };

    let mut out: Vec<(f64, f64)> = Vec::new();
    let push_point = |t: f64, v: f64, out: &mut Vec<(f64, f64)>| {
        if let Some(last) = out.last_mut() {
            if (last.0 - t).abs() <= 1e-9 {
                last.1 = v;
                return;
            }
        }
        out.push((t, v));
    };

    let first_t = samples[0].0.max(range_start);
    push_point(first_t, value_at(first_t), &mut out);

    let mut anchor = 0usize;
    loop {
        let mut f = anchor + 1;
        let mut farthest = anchor + 1;
        while f < samples.len() {
            if line_fits(anchor, f) {
                farthest = f;
                f += 1;
            } else {
                break;
            }
        }
        let (t, v) = samples[farthest];
        push_point(t, v, &mut out);
        anchor = farthest;
        if anchor + 1 >= samples.len() {
            break;
        }
    }

    // 末端点（强制）。末点值与端值在容差内相等时把末点延伸到 range_end，
    // 而不是追加第四点 —— 保持"线性段只有端点"的精简不变式。
    let end_v = value_at(range_end);
    if let Some(last) = out.last_mut() {
        if (last.1 - end_v).abs() <= tolerance {
            last.0 = range_end;
            last.1 = end_v;
        } else {
            push_point(range_end, end_v, &mut out);
        }
    } else {
        push_point(range_end, end_v, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// u ↔ 偏移必须严格互逆，且速率取绝对值（倒放的负速率不改变坐标域）。
    #[test]
    fn the_take_time_domain_round_trips() {
        for rate in [1.0, 0.5, 2.0, -1.5] {
            for offset in [0.0, 0.25, 3.5] {
                let u = offset_to_take_u(offset, rate);
                assert!((take_u_to_offset(u, rate) - offset).abs() < 1e-12);
            }
        }
        // 非法速率按 1 处理，而不是产生 NaN/0。
        assert_eq!(offset_to_take_u(2.0, 0.0), 2.0);
        assert_eq!(offset_to_take_u(2.0, f64::NAN), 2.0);
    }

    /// 采样在点之间插值，在区间外保持端点值（不是 0）。
    #[test]
    fn sampling_interpolates_and_holds_the_ends() {
        let points = [EnvelopePoint::new(1.0, 2.0), EnvelopePoint::new(3.0, 6.0)];
        assert_eq!(sample_envelope(&points, 0.5), 2.0, "首点之前 hold");
        assert_eq!(sample_envelope(&points, 2.0), 4.0);
        assert_eq!(sample_envelope(&points, 9.0), 6.0, "末点之后 hold");
        assert_eq!(sample_envelope(&[], 1.0), 0.0, "没有包络 = 不改音高");
    }

    /// 全零曲线不产生任何断点（保持宿主原状，不写空包络）。
    #[test]
    fn a_flat_default_curve_produces_no_points() {
        let offsets = vec![0.0_f32; 64];
        assert!(build_take_pitch_points(&offsets, 5.0, 1.0, 1.0).is_empty());
    }

    /// 平坦非零曲线压成恰好两个端点，且换到 u 域（端点 = 0 与 长度×速率）。
    #[test]
    fn a_flat_non_zero_curve_becomes_two_endpoints_in_take_time() {
        let offsets = vec![2.0_f32; 32];
        let points = build_take_pitch_points(&offsets, 5.0, 1.0, 2.0);
        assert_eq!(points.len(), 2, "{points:?}");
        assert!((points[0].time - 0.0).abs() < 1e-12);
        assert!((points[1].time - 2.0).abs() < 1e-12, "u_end = 1.0s × 2.0");
        assert!((points[0].value - 2.0).abs() < 1e-9);
        assert!((points[1].value - 2.0).abs() < 1e-9);
    }

    /// 斜坡只保留折点：点数远小于帧数，且首末值正确。
    #[test]
    fn a_ramp_is_reduced_to_its_corners() {
        let offsets: Vec<f32> = (0..200).map(|index| index as f32 * 0.01).collect();
        let points = build_take_pitch_points(&offsets, 5.0, 1.0, 1.0);
        assert!(points.len() <= 3, "斜坡不该逐帧写点：{}", points.len());
        assert!((points[0].value - 0.0).abs() < 0.03);
        assert!((points[points.len() - 1].value - 1.99).abs() < 0.03);
    }
}
