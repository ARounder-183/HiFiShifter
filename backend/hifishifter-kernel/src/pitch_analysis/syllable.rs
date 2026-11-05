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
pub const SYLLABLE_ALGORITHM_VERSION: u32 = 1;

/// 源域音节区间（秒）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableSpan {
    pub start_sec: f64,
    pub end_sec: f64,
}

/// Take 上持久化的音节分析结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TakeSyllableAnalysis {
    pub algorithm: String,
    pub algorithm_version: u32,
    /// 分析所依据的源媒体签名；文件内容或分析帧率变化时整体失效。
    pub source_digest: String,
    pub source_spans_sec: Vec<SyllableSpan>,
}

impl TakeSyllableAnalysis {
    pub fn is_current(&self, source_digest: &str) -> bool {
        self.algorithm == SYLLABLE_ALGORITHM
            && self.algorithm_version == SYLLABLE_ALGORITHM_VERSION
            && self.source_digest == source_digest
            && !self.source_spans_sec.is_empty()
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
}

impl Default for SyllableConfig {
    fn default() -> Self {
        Self {
            min_span_sec: 0.07,
            max_span_sec: 1.4,
            silence_floor: 0.008,
            pitch_jump_semitones: 0.8,
            min_probation_sec: 0.035,
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
    for candidate in candidates {
        if candidate.sec <= start_sec + min_span_sec || candidate.sec >= end_sec - min_span_sec {
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

    // 候选切完之后，对每个区间再做一次最长段拆分；否则“整段一个超长块”
    // 会绕过 max_span，长素材因此变成不可操作的巨大块。
    let mut bounded_points = Vec::with_capacity(points.len());
    bounded_points.push(points[0]);
    for window in points.windows(2) {
        let a = window[0];
        let b = window[1];
        let pieces = ((b - a) / max_span_sec).ceil().max(1.0) as usize;
        let step = (b - a) / pieces as f64;
        for index in 1..pieces {
            bounded_points.push(a + step * index as f64);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
