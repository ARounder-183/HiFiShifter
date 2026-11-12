//! 演唱音节边界估计（纯 Rust，无模型依赖）。
//!
//! ## 为什么不做“歌词音节识别”
//!
//! 没有歌词或音素强制对齐时，无法从音频唯一确定“第几个字”。本模块只估计
//! **发声段内部的音节边界**：以现有逐帧电平做 VAD，用音高稳定度与能量/谱通量
//! 变化挑出候选起点，再用最短/最长段约束收敛。
//!
//! ## 输入
//!
//! `level` 与 `pitch` 同帧率（项目现有分析缓存为 5 ms/帧），覆盖源媒体。
//! `level` 为线性幅值；`pitch` 为 MIDI 音高，0 表示无声帧。两者都可以缺项：
//! 只有电平时退化为 VAD + 能量变化，只有音高时仍可使用音高跳变。
//!
//! ## 输出边界
//!
//! 返回的区间是**源域秒**，首尾相接、不重叠，供 Take 持久化；显示时再按
//! 裁切 / 速率 / 倒放 / Loop 投影到时间轴。

use serde::{Deserialize, Serialize};

/// 算法标识：改算法时必须同时升版本，旧缓存与合并引用才能正确失效。
pub const SYLLABLE_ALGORITHM: &str = "level_vad+pitch_stability+spectral_onset";
pub const SYLLABLE_ALGORITHM_VERSION: u32 = 2;

/// 源域音节区间（秒）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableSpan {
    pub start_sec: f64,
    pub end_sec: f64,
    /// Representative source MIDI note for this span (analysis-time truth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub midi_note: Option<f32>,
    /// Stable identity of the full-source note event. Two clipped views of the
    /// same note must share this id; it is not a Clip window coordinate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

/// Stable identity of a full-source note event. The id is derived from the
/// full-source boundaries, never from a Clip's current audible window.
pub fn source_event_id(start_sec: f64, end_sec: f64) -> String {
    format!("{:.9}:{:.9}", start_sec, end_sec)
}

/// Fill missing event ids and sort full-source events deterministically.
pub fn normalize_source_spans(mut spans: Vec<SyllableSpan>) -> Vec<SyllableSpan> {
    spans.retain(|span| {
        span.start_sec.is_finite() && span.end_sec.is_finite() && span.end_sec > span.start_sec
    });
    for span in &mut spans {
        if span.event_id.is_none() {
            span.event_id = Some(source_event_id(span.start_sec, span.end_sec));
        }
    }
    spans.sort_by(|a, b| {
        a.start_sec
            .total_cmp(&b.start_sec)
            .then(a.end_sec.total_cmp(&b.end_sec))
    });
    spans
}

/// Take 上持久化的音节分析结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TakeSyllableAnalysis {
    pub algorithm: String,
    pub algorithm_version: u32,
    /// 分析所依据的源媒体签名；文件内容或分析帧率变化时整体失效。
    /// Only the source identity; window validity is expressed separately.
    pub source_digest: String,
    pub source_spans_sec: Vec<SyllableSpan>,
    /// Source-domain window covered by this analysis. A narrower current
    /// window is valid; a wider one must trigger reanalysis.
    pub analyzed_window_sec: [f64; 2],
}

/// A completed full-source analysis is usable only when its provenance window
/// contains the current audible window and it has at least one note event.
/// This controls whether another inference can be skipped; it never controls
/// whether still-intersecting cached events are drawn.
pub fn analysis_window_status(
    analyzed: [f64; 2],
    window: (f64, f64),
    has_spans: bool,
) -> &'static str {
    if !has_spans {
        "missing"
    } else if window.0 >= analyzed[0] - 1e-6 && window.1 <= analyzed[1] + 1e-6 {
        "current"
    } else {
        "stale"
    }
}

impl TakeSyllableAnalysis {
    /// True when every position of the window was covered by this analysis.
    /// Narrowing a Clip keeps the cached spans valid; extending it does not.
    pub fn covers_window(&self, window: (f64, f64)) -> bool {
        if !(window.0.is_finite() && window.1.is_finite() && window.1 >= window.0) {
            return false;
        }
        let [start, end] = self.analyzed_window_sec;
        start.is_finite()
            && end.is_finite()
            && end >= start
            && window.0 >= start - 1e-6
            && window.1 <= end + 1e-6
    }

    pub fn is_current_for_window(&self, source_digest: &str, window: (f64, f64)) -> bool {
        self.is_current(source_digest) && self.covers_window(window)
    }

    pub fn is_current(&self, source_digest: &str) -> bool {
        let supported = (self.algorithm == SYLLABLE_ALGORITHM
            && self.algorithm_version == SYLLABLE_ALGORITHM_VERSION)
            || (self.algorithm == crate::game_onnx::GAME_ALGORITHM
                && self.algorithm_version == crate::game_onnx::GAME_ALGORITHM_VERSION);
        supported && self.source_digest == source_digest && !self.source_spans_sec.is_empty()
    }
}

/// 一处候选边界（源域秒）及其证据强度。
#[derive(Debug, Clone, Copy)]
struct BoundaryCandidate {
    sec: f64,
    strength: f64,
}

/// 分析参数集中在一处，便于产品调参与回归。
#[derive(Debug, Clone, Copy)]
pub struct SyllableConfig {
    pub min_span_sec: f64,
    pub max_span_sec: f64,
    pub silence_floor: f64,
    pub pitch_jump_semitones: f64,
    pub min_probation_sec: f64,
    /// 长段内“弱边界”的最小间隔。弱边界只在没有明确音节起点时用于拆开
    /// 持续发声，避免把整句长音压成一个不可编辑的巨块。
    pub weak_boundary_min_gap_sec: f64,
}

impl Default for SyllableConfig {
    fn default() -> Self {
        Self {
            min_span_sec: 0.07,
            max_span_sec: 1.4,
            silence_floor: 0.008,
            pitch_jump_semitones: 0.8,
            min_probation_sec: 0.035,
            weak_boundary_min_gap_sec: 0.28,
        }
    }
}

/// 估计源域音节区间。
///
/// `level` / `pitch` 的长度决定帧率；`frame_period_ms` 必须与它们一致。
/// 电平全静音、数组为空或参数非法时返回空（调用方回退为整个 Clip 一个块）。
pub fn estimate_syllable_spans(
    level: &[f32],
    pitch: &[f32],
    frame_period_ms: f64,
    config: SyllableConfig,
) -> Vec<SyllableSpan> {
    let fp = frame_period_ms.max(0.1);
    let frames = level.len().max(pitch.len());
    if frames < 2 {
        return Vec::new();
    }
    // 两个输入可以缺项（FCPE 缺席时只有 level），但向下取下标时必须有一致长度：
    // 缺失帧按“静音 / 无音高”补齐，不能按原数组长度切。
    let mut level_owned;
    let level: &[f32] = if level.len() == frames {
        level
    } else {
        level_owned = level.to_vec();
        level_owned.resize(frames, 0.0);
        &level_owned
    };
    let mut pitch_owned;
    let pitch: &[f32] = if pitch.len() == frames {
        pitch
    } else {
        pitch_owned = pitch.to_vec();
        pitch_owned.resize(frames, 0.0);
        &pitch_owned
    };
    let fps = 1000.0 / fp;
    let min_frames = ((config.min_span_sec * fps).round() as usize).max(1);
    let max_frames = ((config.max_span_sec * fps).round() as usize).max(min_frames + 1);
    let probation_frames = ((config.min_probation_sec * fps).round() as usize).max(1);

    // 用较高的 VAD 门限找到“确定有声”的核心，再向两侧扩到较低门限；
    // 这样弱起音不会被截掉，也不会把底噪当成整段音节。
    let peak = level
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max);
    if peak <= config.silence_floor as f32 || peak <= 1e-9 {
        return Vec::new();
    }
    let core_floor = (peak as f64 * 0.08).max(config.silence_floor);
    let extend_floor = (peak as f64 * 0.025).max(config.silence_floor * 0.5);
    let has_pitch = pitch.iter().any(|v| *v > 0.0);
    let voiced: Vec<bool> = (0..frames)
        .map(|i| {
            let lv = level.get(i).copied().unwrap_or(0.0) as f64;
            let pv = pitch.get(i).copied().unwrap_or(0.0) as f64;
            lv.is_finite() && lv >= core_floor && (!has_pitch || pv > 0.0)
        })
        .collect();
    let voiced_extended: Vec<bool> = (0..frames)
        .map(|i| {
            let lv = level.get(i).copied().unwrap_or(0.0) as f64;
            lv.is_finite() && lv >= extend_floor
        })
        .collect();

    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < frames {
        if !voiced[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i + 1;
        // 短于 probations 的掉音视为同一次发声内部的波动。
        while end < frames {
            if voiced[end] {
                end += 1;
                continue;
            }
            let gap_start = end;
            while end < frames && !voiced[end] {
                end += 1;
            }
            if end - gap_start > probation_frames {
                break;
            }
        }
        let mut region_start = start;
        let mut region_end = end;
        while region_start > 0 && voiced_extended[region_start - 1] {
            region_start -= 1;
        }
        while region_end < frames && voiced_extended[region_end] {
            region_end += 1;
        }
        if region_end > region_start {
            regions.push((region_start, region_end));
        }
        i = end.max(region_end);
    }

    let mut out = Vec::new();
    for (start, end) in regions {
        let mut candidates = collect_candidates(
            &level[start..end],
            &pitch[start..end],
            start,
            fp,
            config.pitch_jump_semitones,
        );
        candidates.sort_by(|a, b| a.sec.total_cmp(&b.sec));
        let region_start_sec = start as f64 * fp / 1000.0;
        let region_end_sec = end as f64 * fp / 1000.0;
        let mut pieces = split_region(
            region_start_sec,
            region_end_sec,
            &candidates,
            min_frames as f64 * fp / 1000.0,
            max_frames as f64 * fp / 1000.0,
        );
        pieces.retain(|span| {
            span.end_sec > span.start_sec && span.end_sec.is_finite() && span.start_sec.is_finite()
        });
        // A clipped block can have no voiced frames in its visible part. Keep
        // the source note found from the full span so the view never needs to
        // guess a vertical position from the viewport midpoint.
        for span in &mut pieces {
            let first = ((span.start_sec * fps).round() as usize).min(frames);
            let last = ((span.end_sec * fps).round() as usize).min(frames);
            let mut values: Vec<f32> = pitch[first..last.max(first)]
                .iter()
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .collect();
            if values.is_empty() {
                span.midi_note = None;
                continue;
            }
            values.sort_by(f32::total_cmp);
            let middle = values.len() / 2;
            span.midi_note = Some(if values.len().is_multiple_of(2) {
                (values[middle - 1] + values[middle]) * 0.5
            } else {
                values[middle]
            });
        }
        out.extend(pieces);
    }
    merge_touching_spans(out)
}

fn collect_candidates(
    level: &[f32],
    pitch: &[f32],
    frame_offset: usize,
    frame_period_ms: f64,
    pitch_jump: f64,
) -> Vec<BoundaryCandidate> {
    let mut out = Vec::new();
    // 能量与谱通量的代理：|Δlevel| 归一化。无 FFT 也比逐帧零散跳变稳定。
    for i in 2..level.len().saturating_sub(2) {
        let prev = level[i - 1] as f64;
        let cur = level[i] as f64;
        let next = level[i + 1] as f64;
        let rising = cur - prev.max(0.0);
        let local_relief = cur >= (prev + next) * 0.5;
        let energy_strength = if rising > 0.0 && local_relief {
            (rising / cur.max(1e-6)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let p0 = pitch.get(i - 1).copied().unwrap_or(0.0) as f64;
        let p1 = pitch.get(i).copied().unwrap_or(0.0) as f64;
        let p2 = pitch.get(i + 1).copied().unwrap_or(0.0) as f64;
        let pitch_ok = p0 > 0.0 && p1 > 0.0 && p2 > 0.0;
        let pitch_strength = if pitch_ok {
            let jump = (p1 - p0).abs();
            let turn = (p2 - p1).abs();
            if jump >= pitch_jump && jump >= turn * 0.75 {
                (jump / 12.0).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let strength = energy_strength * 0.45 + pitch_strength * 0.75;
        if strength > 0.035 {
            let absolute_frame = frame_offset + i;
            // 帧中心，避免边界整体早半帧。
            out.push(BoundaryCandidate {
                sec: (absolute_frame as f64 + 0.5) * frame_period_ms / 1000.0,
                strength,
            });
        }
    }

    /*
     * 持续发声段往往没有明显能量起音，也没有达到半音级的音高跳变：
     * 一个长元音、连续滑音或长句会被判成一个巨大区间。这里补一组**弱候选**：
     * 音高变化速度的局部峰、音高台阶（不足半音但仍在运动）、以及短时能量起伏。
     *
     * 弱候选从不抢占强候选：后续 `split_region` 先放强边界，弱边界只用于
     * 填充仍然超过最大时长的区段。这样不会把颤音或正常连音切成碎片。
     */
    let mut prev_pitch = 0.0f64;
    for i in 1..pitch.len() {
        let p = pitch[i] as f64;
        if !p.is_finite() || p <= 0.0 {
            prev_pitch = 0.0;
            continue;
        }
        let prev = pitch[i - 1] as f64;
        if !prev.is_finite() || prev <= 0.0 {
            prev_pitch = p;
            continue;
        }
        let delta = p - prev;
        let previous_delta = if prev_pitch > 0.0 {
            prev - prev_pitch
        } else {
            delta
        };
        let turn = (delta - previous_delta).abs();
        let pitch_velocity = delta.abs();
        let pitch_turn_strength = if turn >= 0.035 {
            (turn / 0.35).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let pitch_step_strength = if pitch_velocity >= 0.12 && pitch_velocity < pitch_jump {
            (pitch_velocity / pitch_jump).clamp(0.0, 1.0) * 0.55
        } else {
            0.0
        };
        let level_window = if level.is_empty() {
            0.0
        } else {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(level.len());
            let mean = level[lo..hi]
                .iter()
                .map(|value| (*value as f64).max(0.0))
                .sum::<f64>()
                / (hi - lo).max(1) as f64;
            if mean <= 1e-9 {
                0.0
            } else {
                let variation = level[lo..hi]
                    .iter()
                    .map(|value| ((*value as f64).max(0.0) - mean).abs())
                    .fold(0.0f64, f64::max);
                (variation / mean).clamp(0.0, 1.0) * 0.22
            }
        };
        let strength = (pitch_turn_strength * 0.55 + pitch_step_strength + level_window).min(0.6);
        if strength >= 0.06 {
            out.push(BoundaryCandidate {
                sec: (frame_offset as f64 + i as f64 + 0.5) * frame_period_ms / 1000.0,
                strength,
            });
        }
        prev_pitch = p;
    }
    out
}

/// 在区间内按候选边界切段；候选太密时保留更强的一侧，太稀时按最长段均分。
fn split_region(
    start_sec: f64,
    end_sec: f64,
    candidates: &[BoundaryCandidate],
    min_span_sec: f64,
    max_span_sec: f64,
) -> Vec<SyllableSpan> {
    if !(end_sec > start_sec) {
        return Vec::new();
    }
    let mut cuts: Vec<f64> = Vec::new();
    let mut weak_cuts: Vec<f64> = Vec::new();
    for candidate in candidates {
        let is_weak = candidate.strength < 0.18;
        if candidate.sec <= start_sec + min_span_sec || candidate.sec >= end_sec - min_span_sec {
            continue;
        }
        if is_weak {
            weak_cuts.push(candidate.sec);
            continue;
        }
        if let Some(last) = cuts.last() {
            if candidate.sec - *last < min_span_sec {
                // 保留更强候选：用强度比较替掉刚放进去的弱候选。
                let prev_index = cuts.len() - 1;
                let prev_strength = candidates
                    .iter()
                    .filter(|c| (c.sec - *last).abs() < 1e-9)
                    .map(|c| c.strength)
                    .fold(0.0, f64::max);
                if candidate.strength > prev_strength {
                    cuts[prev_index] = candidate.sec;
                }
                continue;
            }
        }
        cuts.push(candidate.sec);
    }
    cuts.sort_by(f64::total_cmp);
    weak_cuts.sort_by(f64::total_cmp);

    let mut points = Vec::with_capacity(cuts.len() + 2);
    points.push(start_sec);
    let mut last = start_sec;
    for cut in cuts {
        if cut - last >= min_span_sec && end_sec - cut >= min_span_sec {
            points.push(cut);
            last = cut;
        }
    }
    points.push(end_sec);

    /*
     * 强候选切完之后，超长区段优先落在距离中点最近的**弱候选**上；没有弱候选时
     * 才均分。这样连续长音也能获得贴近真实音高拐点的边界，而不是机械等距。
     */
    let mut bounded_points = Vec::with_capacity(points.len());
    bounded_points.push(points[0]);
    for window in points.windows(2) {
        let a = window[0];
        let b = window[1];
        if b - a <= max_span_sec + 1e-9 {
            bounded_points.push(b);
            continue;
        }
        let pieces = ((b - a) / max_span_sec).ceil().max(2.0) as usize;
        let target_step = (b - a) / pieces as f64;
        let min_gap = min_span_sec.max(target_step * 0.35);
        let mut cursor = a;
        for piece in 1..pieces {
            let ideal = a + target_step * piece as f64;
            let picked = weak_cuts
                .iter()
                .copied()
                .filter(|cut| *cut > cursor + min_gap && *cut < b - min_gap)
                .min_by(|left, right| (left - ideal).abs().total_cmp(&(right - ideal).abs()));
            let next = picked.unwrap_or(ideal);
            bounded_points.push(next);
            cursor = next;
        }
        bounded_points.push(b);
    }

    let mut spans = Vec::new();
    for window in bounded_points.windows(2) {
        let a = window[0];
        let b = window[1];
        if !(b > a) {
            continue;
        }
        spans.push(SyllableSpan {
            start_sec: a,
            end_sec: b,
            midi_note: None,
            event_id: None,
        });
    }
    spans
}

fn merge_touching_spans(spans: Vec<SyllableSpan>) -> Vec<SyllableSpan> {
    let mut sorted = spans;
    sorted.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
    let mut out: Vec<SyllableSpan> = Vec::new();
    for span in sorted {
        if let Some(last) = out.last_mut() {
            // 只合并真正重叠的区间。首尾相接是正常的音节边界：把它们并起来
            // 会把整句重新压成一个块，也会让 max_span 的拆分白做。
            if span.start_sec < last.end_sec - 1e-9 {
                last.end_sec = last.end_sec.max(span.end_sec);
                continue;
            }
        }
        out.push(span);
    }
    out
}

/// 把分析得到的发声段补齐到可见窗口，同时保留真实音节边界。
///
/// 这里只做两件事：裁到窗口、填未覆盖的静音间隙。**绝不合并相邻段**：
/// 分析器已经把 `max_span_sec` 内的音节切开，若显示层再把首尾相接的段并起来，
/// 长素材会重新退化成整段一个块。真正重叠的段由调用方或前端最终的
/// “不得重叠”不变量处理，不在这里做二次归并。
pub fn fill_spans_without_merging(
    spans: &[SyllableSpan],
    window_start: f64,
    window_end: f64,
) -> Vec<SyllableSpan> {
    if !(window_start.is_finite() && window_end.is_finite() && window_end > window_start + 1e-9) {
        return Vec::new();
    }
    let mut clipped = spans
        .iter()
        .filter_map(|span| {
            let start = span.start_sec.max(window_start).min(window_end);
            let end = span.end_sec.max(window_start).min(window_end);
            (end > start + 1e-9).then_some(SyllableSpan {
                start_sec: start,
                end_sec: end,
                midi_note: span.midi_note,
                event_id: None,
            })
        })
        .collect::<Vec<_>>();
    clipped.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
    let mut filled = Vec::with_capacity(clipped.len().saturating_mul(2));
    let mut cursor = window_start;
    for span in clipped {
        if span.start_sec > cursor + 1e-9 {
            filled.push(SyllableSpan {
                start_sec: cursor,
                end_sec: span.start_sec,
                midi_note: None,
                event_id: None,
            });
        }
        let start = span.start_sec.max(cursor);
        if span.end_sec > start + 1e-9 {
            filled.push(SyllableSpan {
                start_sec: start,
                end_sec: span.end_sec,
                midi_note: span.midi_note,
                event_id: None,
            });
            cursor = span.end_sec;
        }
    }
    if window_end > cursor + 1e-9 {
        filled.push(SyllableSpan {
            start_sec: cursor,
            end_sec: window_end,
            midi_note: None,
            event_id: None,
        });
    }
    filled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syllable_cache_accepts_game_and_fallback_with_their_own_versions() {
        for (algorithm, version) in [
            (SYLLABLE_ALGORITHM, SYLLABLE_ALGORITHM_VERSION),
            (
                crate::game_onnx::GAME_ALGORITHM,
                crate::game_onnx::GAME_ALGORITHM_VERSION,
            ),
        ] {
            let mut analysis = TakeSyllableAnalysis {
                algorithm: algorithm.into(),
                algorithm_version: version,
                source_digest: "same-pcm".into(),
                source_spans_sec: vec![SyllableSpan {
                    start_sec: 0.0,
                    end_sec: 1.0,
                    midi_note: None,
                    event_id: Some(crate::pitch_analysis::source_event_id(0.0, 1.0)),
                }],
                analyzed_window_sec: [0.0, 1.0],
            };
            assert!(analysis.is_current("same-pcm"), "{algorithm} was rejected");
            assert!(!analysis.is_current("changed-pcm"));
            analysis.algorithm_version += 1;
            assert!(!analysis.is_current("same-pcm"));
            analysis.algorithm_version = version;
            analysis.source_spans_sec.clear();
            assert!(!analysis.is_current("same-pcm"));
        }
    }

    fn level_at(secs: &[f64], fps: f64) -> Vec<f32> {
        let frames = (secs.len() as f64 * fps).ceil() as usize;
        (0..frames)
            .map(|i| {
                let t = i as f64 / fps;
                let mut v = 0.002f32;
                for (start, end, amp) in secs.chunks(3).map(|c| (c[0], c[1], c[2])) {
                    if t >= start && t < end {
                        v = amp as f32;
                    }
                }
                v
            })
            .collect()
    }

    #[test]
    fn silence_has_no_spans() {
        let level = vec![0.0f32; 1000];
        assert!(estimate_syllable_spans(&level, &[], 5.0, SyllableConfig::default()).is_empty());
    }

    #[test]
    fn energy_onsets_split_a_voiced_run() {
        let fps = 200.0;
        let mut level = vec![0.001f32; (fps * 1.2) as usize];
        for (i, v) in level.iter_mut().enumerate() {
            let t = i as f64 / fps;
            *v = if (0.1..0.9).contains(&t) {
                0.2 + if (t * 7.0).fract() < 0.5 { 0.02 } else { 0.0 }
            } else {
                0.001
            };
        }
        let spans = estimate_syllable_spans(&level, &[], 5.0, SyllableConfig::default());
        assert!(!spans.is_empty(), "有声段应产生至少一个块");
        assert!(spans.iter().all(|s| s.end_sec > s.start_sec));
    }

    #[test]
    fn max_span_enforces_bounded_blocks() {
        let fps = 200.0;
        let level = vec![0.3f32; (fps * 4.0) as usize];
        let cfg = SyllableConfig {
            max_span_sec: 0.5,
            ..SyllableConfig::default()
        };
        let spans = estimate_syllable_spans(&level, &[], 5.0, cfg);
        assert!(spans.len() >= 8);
        assert!(spans.iter().all(|s| s.end_sec - s.start_sec <= 0.5 + 1e-9));
    }

    #[test]
    fn spans_are_sorted_and_do_not_overlap() {
        let level = level_at(&[0.1, 0.4, 0.3, 0.5, 0.8, 0.2], 200.0);
        let spans = estimate_syllable_spans(&level, &[], 5.0, SyllableConfig::default());
        for pair in spans.windows(2) {
            assert!(pair[0].end_sec <= pair[1].start_sec + 1e-9);
        }
    }

    #[test]
    fn long_constant_vowel_gets_multiple_blocks() {
        let fps = 200.0;
        let frames = (fps * 8.0) as usize;
        let level = vec![0.3f32; frames];
        let pitch = (0..frames)
            .map(|i| {
                let t = i as f64 / fps;
                60.0 + ((t * 1.7).sin() * 0.6) as f32
            })
            .collect::<Vec<_>>();
        let spans = estimate_syllable_spans(&level, &pitch, 5.0, SyllableConfig::default());
        assert!(
            spans.len() >= 5,
            "8 秒持续长音至少应切成多段，实际 {} 段",
            spans.len()
        );
        assert!(spans
            .iter()
            .all(|span| span.end_sec - span.start_sec <= 1.4 + 1e-9));
    }

    #[test]
    fn analysis_window_must_contain_the_current_window() {
        let analysis = TakeSyllableAnalysis {
            algorithm: SYLLABLE_ALGORITHM.into(),
            algorithm_version: SYLLABLE_ALGORITHM_VERSION,
            source_digest: "same-pcm".into(),
            source_spans_sec: vec![SyllableSpan {
                start_sec: 0.9,
                end_sec: 1.4,
                midi_note: Some(64.0),
                event_id: Some(crate::pitch_analysis::source_event_id(0.9, 1.4)),
            }],
            analyzed_window_sec: [0.9, 2.0],
        };
        assert!(!analysis.is_current_for_window("same-pcm", (0.3, 2.0)));
        assert!(analysis.is_current_for_window("same-pcm", (1.0, 1.8)));
        assert!(!analysis.is_current_for_window("changed-pcm", (1.0, 1.8)));
        let outside = TakeSyllableAnalysis {
            analyzed_window_sec: [1.5, 2.0],
            ..analysis
        };
        assert!(outside.is_current("same-pcm"));
        assert!(!outside.is_current_for_window("same-pcm", (1.0, 1.8)));
    }

    #[test]
    fn normalize_source_spans_assigns_stable_event_ids() {
        let spans = normalize_source_spans(vec![
            SyllableSpan {
                start_sec: 0.2,
                end_sec: 0.5,
                midi_note: Some(62.0),
                event_id: None,
            },
            SyllableSpan {
                start_sec: 0.046,
                end_sec: 0.252,
                midi_note: Some(64.0),
                event_id: None,
            },
        ]);
        assert_eq!(spans[0].start_sec, 0.046);
        assert_eq!(
            spans[0].event_id.as_deref(),
            Some("0.046000000:0.252000000")
        );
        assert_eq!(
            spans[1].event_id.as_deref(),
            Some("0.200000000:0.500000000")
        );
    }

    #[test]
    fn analysis_window_status_gates_cache_adoption_by_provenance() {
        assert_eq!(
            analysis_window_status([1.0, 3.0], (1.0, 3.0), true),
            "current"
        );
        assert_eq!(
            analysis_window_status([1.0, 3.0], (0.5, 3.0), true),
            "stale"
        );
        assert_eq!(
            analysis_window_status([1.0, 3.0], (1.0, 3.0), false),
            "missing"
        );
    }

    #[test]
    fn estimated_spans_keep_the_source_note_median() {
        let fps = 200.0;
        let frames = (fps * 1.0) as usize;
        let level = vec![0.3f32; frames];
        let pitch = vec![61.0f32; frames];
        let spans = estimate_syllable_spans(&level, &pitch, 5.0, SyllableConfig::default());
        assert!(!spans.is_empty());
        assert!(spans.iter().all(|span| span.midi_note == Some(61.0)));
    }

    #[test]
    fn filling_preserves_adjacent_algorithm_boundaries() {
        let spans = vec![
            SyllableSpan {
                start_sec: 0.5,
                end_sec: 1.0,
                midi_note: Some(60.0),
                event_id: Some(crate::pitch_analysis::source_event_id(0.5, 1.0)),
            },
            SyllableSpan {
                start_sec: 1.0,
                end_sec: 1.5,
                midi_note: Some(62.0),
                event_id: Some(crate::pitch_analysis::source_event_id(1.0, 1.5)),
            },
        ];
        let filled = fill_spans_without_merging(&spans, 0.0, 2.0);
        assert_eq!(filled.len(), 4, "首尾静音要补，但两个音节不能被并成一块");
        assert!((filled[1].end_sec - 1.0).abs() < 1e-9);
        assert!((filled[2].start_sec - 1.0).abs() < 1e-9);
    }
}
