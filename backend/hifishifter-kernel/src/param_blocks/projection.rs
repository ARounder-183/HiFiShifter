//! Clip / Take 源域到工程时间轴的唯一投影实现。
//!
//! 职责：把 `TakeSpan` 投影成工程帧区间，并提供逆投影。
//! 关系：`base` 用它生成 base 块；`compose` 用它投影用户块；`editing` 用它把
//! 工程帧换算回源域。维护约束：所有投影公式只在这里出现一份；前端预览端口必须
//! 与本文件用 golden fixture 对拍。

use super::types::TakeSpan;
use crate::state::{Clip, ClipTake};

pub const MAX_FRAME: u32 = 1_000_000;
/// Shared tolerance for positive source-domain intersections.
pub const SOURCE_EPS: f64 = 1e-12;
/// Shared tolerance for finite window geometry.
pub const WINDOW_EPS: f64 = 1e-9;

/// A positive source/window intersection. Callers must not use a separate
/// epsilon: a left-cut note must survive all the way to its projected frame.
pub fn window_intersection(span: (f64, f64), window: (f64, f64)) -> Option<(f64, f64)> {
    let start = span.0.max(window.0);
    let end = span.1.min(window.1);
    (end > start + SOURCE_EPS).then_some((start, end))
}

/// A positive continuous intersection that rounded to zero frames is still one
/// visible frame; it is not an absent block.
pub fn positive_interval_to_frames(start: u32, end: u32) -> Option<(u32, u32)> {
    (end > start)
        .then_some((start, end))
        .or_else(|| (start < MAX_FRAME).then_some((start, start + 1)))
}

/// Cut flags for a non-loop source/window intersection. Reverse playback swaps
/// the source edges; loop wrapping is handled by the projection's ring rules.
pub fn cut_flags_for_intersection(
    span: (f64, f64),
    window: (f64, f64),
    reversed: bool,
) -> (bool, bool) {
    if reversed {
        (
            span.1 > window.1 + WINDOW_EPS,
            span.0 < window.0 - WINDOW_EPS,
        )
    } else {
        (
            span.0 < window.0 - WINDOW_EPS,
            span.1 > window.1 + WINDOW_EPS,
        )
    }
}

/// 源时间映射扩展点。
///
/// v1 使用线性 rate；未来 `ClipTake.stretch_markers` 非空时用分段线性映射替换。
/// 参数块投影与音频渲染必须消费同一个 map，否则 UI 与声音会再次脱节。
pub trait SourceTimeMap {
    fn source_to_take_media(&self, source_sec: f64) -> Option<f64>;
    fn take_media_to_source(&self, media_sec: f64) -> Option<f64>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearRateMap;

impl SourceTimeMap for LinearRateMap {
    fn source_to_take_media(&self, source_sec: f64) -> Option<f64> {
        source_sec.is_finite().then_some(source_sec)
    }

    fn take_media_to_source(&self, media_sec: f64) -> Option<f64> {
        media_sec.is_finite().then_some(media_sec)
    }
}

#[derive(Clone, Debug)]
pub struct ClipProjection {
    pub clip_id: String,
    pub take_id: String,
    pub project_start_sec: f64,
    pub project_length_sec: f64,
    pub source_window_start_sec: f64,
    pub source_window_end_sec: f64,
    pub media_duration_sec: Option<f64>,
    pub effective_rate: f64,
    pub reversed: bool,
    pub loop_enabled: bool,
    pub loop_cycle_sec: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectedInterval {
    pub cycle: u32,
    pub start_frame: u32,
    pub end_frame_exclusive: u32,
    /// The visible interval starts inside its source span (window/overlap cut).
    pub cut_start: bool,
    /// The visible interval ends inside its source span (window/overlap cut).
    pub cut_end: bool,
}

fn finite_positive(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 1e-9 {
        value
    } else {
        fallback
    }
}

fn take_media_duration(take: &ClipTake) -> Option<f64> {
    if let (Some(frames), Some(sample_rate)) = (take.duration_frames, take.source_sample_rate) {
        if sample_rate > 0 && frames > 0 {
            return Some(frames as f64 / sample_rate as f64);
        }
    }
    if let Some(duration) = take.duration_sec.filter(|d| d.is_finite() && *d > 0.0) {
        return Some(duration);
    }
    take.midi_note_data
        .as_ref()
        .map(|notes| notes.iter().map(|note| note.end_sec).fold(0.0f64, f64::max))
        .filter(|end| end.is_finite() && *end > 0.0)
}

impl ClipProjection {
    pub fn from_clip(clip: &Clip) -> Option<Self> {
        let take = clip.active_take();
        Some(Self::from_clip_take(clip, take))
    }

    pub fn from_clip_take(clip: &Clip, take: &ClipTake) -> Self {
        let item_rate = finite_positive(clip.clip_playback_rate as f64, 1.0);
        let rate = finite_positive(take.playback_rate as f64, 1.0) * item_rate;
        let span = (clip.length_sec.max(0.0) * rate).max(0.0);
        let (window_start, window_end) = if take.loop_enabled {
            (take.source_start_sec, take.source_end_sec)
        } else if take.reversed {
            (take.source_end_sec - span, take.source_end_sec)
        } else {
            (take.source_start_sec, take.source_start_sec + span)
        };
        let media_duration = take_media_duration(take);
        let loop_cycle = if take.loop_enabled {
            clip.loop_period_sec
                .filter(|value| value.is_finite() && *value > 1e-9)
                .or(media_duration)
                .or_else(|| Some((take.source_end_sec - take.source_start_sec).abs()))
                .filter(|value| value.is_finite() && *value > 1e-9)
        } else {
            None
        };
        Self {
            clip_id: clip.id.clone(),
            take_id: take.id.clone(),
            project_start_sec: clip.start_sec,
            project_length_sec: clip.length_sec.max(0.0),
            source_window_start_sec: window_start,
            source_window_end_sec: window_end,
            media_duration_sec: media_duration,
            effective_rate: rate,
            reversed: take.reversed,
            loop_enabled: take.loop_enabled && loop_cycle.is_some(),
            loop_cycle_sec: loop_cycle,
        }
    }

    /// The source-domain window that can currently be heard from this Take.
    ///
    /// Analysis, projection and the UI gate must all use this single function.
    /// A loop Clip wraps through the whole media ring; a section loop whose
    /// anchor fits inside the media uses only its own cycle. Unknown media
    /// duration falls back to the stored source window.
    pub fn audible_source_window(&self) -> (f64, f64) {
        let span = (self.project_length_sec.max(0.0) * self.effective_rate).max(0.0);
        let (start, end) = if self.loop_enabled {
            let anchor = self.loop_anchor();
            let cycle = self.loop_cycle_sec.unwrap_or(0.0);
            let media = self.media_duration_sec;
            let whole_ring = media.filter(|duration| {
                *duration > 1e-9
                    && (cycle >= *duration - 1e-9
                        || anchor + cycle > *duration + 1e-9
                        || anchor < -1e-9)
            });
            if let Some(duration) = whole_ring {
                (0.0, duration)
            } else if cycle > 1e-9 && anchor.is_finite() && anchor >= -1e-9 {
                (anchor, anchor + cycle)
            } else {
                (self.source_window_start_sec, self.source_window_end_sec)
            }
        } else if self.reversed {
            (
                self.source_window_end_sec - span,
                self.source_window_end_sec,
            )
        } else {
            (
                self.source_window_start_sec,
                self.source_window_start_sec + span,
            )
        };
        if !start.is_finite() || !end.is_finite() {
            return (0.0, 0.0);
        }
        (start.min(end), start.max(end))
    }

    pub fn frame_period_sec(&self, frame_period_ms: f64) -> f64 {
        frame_period_ms.max(0.1) / 1000.0
    }

    pub fn to_frame(&self, sec: f64, frame_period_ms: f64) -> u32 {
        let frame = (sec * 1000.0 / frame_period_ms.max(0.1)).round();
        if !frame.is_finite() {
            return 0;
        }
        frame.clamp(0.0, MAX_FRAME as f64) as u32
    }

    pub fn to_sec(&self, frame: u32, frame_period_ms: f64) -> f64 {
        frame as f64 * frame_period_ms.max(0.1) / 1000.0
    }

    fn visible_frames(&self, frame_period_ms: f64) -> (u32, u32) {
        let start = self.to_frame(self.project_start_sec, frame_period_ms);
        let end = self.to_frame(
            self.project_start_sec + self.project_length_sec,
            frame_period_ms,
        );
        (start, end.max(start))
    }

    /// 源域秒 → 工程秒（非 Loop）。
    fn project_non_loop(&self, source_sec: f64) -> f64 {
        if self.reversed {
            self.project_start_sec + (self.source_window_end_sec - source_sec) / self.effective_rate
        } else {
            self.project_start_sec
                + (source_sec - self.source_window_start_sec) / self.effective_rate
        }
    }

    /// 工程秒 → 源域秒（非 Loop）。
    fn inverse_non_loop(&self, project_sec: f64) -> f64 {
        if self.reversed {
            self.source_window_end_sec
                - (project_sec - self.project_start_sec) * self.effective_rate
        } else {
            self.source_window_start_sec
                + (project_sec - self.project_start_sec) * self.effective_rate
        }
    }

    fn loop_anchor(&self) -> f64 {
        if self.reversed {
            self.media_duration_sec
                .map(|duration| self.source_window_end_sec.min(duration))
                .unwrap_or(self.source_window_end_sec)
        } else {
            self.source_window_start_sec
        }
    }
}

/// The authoritative audible source window for a Clip's active Take.
/// Analysis, cache adoption and UI gates all call this function.
pub fn clip_audible_source_window(clip: &crate::state::Clip) -> (f64, f64) {
    if clip.takes.is_empty() {
        let take = crate::state::ClipTake::from_clip(clip);
        return ClipProjection::from_clip_take(clip, &take).audible_source_window();
    }
    ClipProjection::from_clip(clip)
        .map(|projection| projection.audible_source_window())
        .unwrap_or((0.0, 0.0))
}

pub fn project_span(
    projection: &ClipProjection,
    span: &TakeSpan,
    frame_period_ms: f64,
) -> Vec<ProjectedInterval> {
    let frame_period_ms = frame_period_ms.max(0.1);
    let (visible_start, visible_end) = projection.visible_frames(frame_period_ms);
    if visible_end <= visible_start {
        return Vec::new();
    }
    let rate = projection.effective_rate.max(1e-9);
    let mut out = Vec::new();
    if projection.loop_enabled {
        let Some(cycle_sec) = projection.loop_cycle_sec.filter(|value| *value > 1e-9) else {
            return out;
        };
        let anchor = projection.loop_anchor();
        let phase = if projection.reversed {
            (anchor - span.source_end_sec).rem_euclid(cycle_sec)
        } else {
            (span.source_start_sec - anchor).rem_euclid(cycle_sec)
        };
        // 一个成员只能覆盖它所在周期的剩余部分；跨周期必须先 normalize_loop_span。
        let length_sec = ((span.source_end_sec - span.source_start_sec) / rate)
            .min(((cycle_sec - phase) / rate).max(0.0))
            .max(0.0);
        let cycle_timeline_sec = cycle_sec / rate;
        if cycle_timeline_sec <= 1e-9 {
            return out;
        }
        let clip_start = projection.project_start_sec;
        let mut start_local = phase / rate;
        let mut cycle = 0u32;
        // 最多 20000 个周期；异常输入不能变成无限循环。
        while start_local < projection.project_length_sec - 1e-9 && out.len() < 20_000 {
            let start_sec = clip_start + start_local;
            let end_sec = (start_sec + length_sec).min(clip_start + projection.project_length_sec);
            let start = projection.to_frame(start_sec.max(clip_start), frame_period_ms);
            let end = projection.to_frame(end_sec, frame_period_ms);
            let start = start.max(visible_start);
            let end = end.min(visible_end);
            if end > start {
                out.push(ProjectedInterval {
                    cycle,
                    start_frame: start,
                    end_frame_exclusive: end,
                    cut_start: false,
                    cut_end: false,
                });
            }
            start_local += cycle_timeline_sec;
            cycle = cycle.saturating_add(1);
        }
        return out;
    }

    // 非 Loop：区间先按当前消费窗口裁剪，再投影。
    let (window_start, window_end) = if projection.reversed {
        (
            projection.source_window_end_sec - projection.project_length_sec * rate,
            projection.source_window_end_sec,
        )
    } else {
        (
            projection.source_window_start_sec,
            projection.source_window_start_sec + projection.project_length_sec * rate,
        )
    };
    let Some((start_sec, end_sec)) = window_intersection(
        (span.source_start_sec, span.source_end_sec),
        (window_start, window_end),
    ) else {
        return out;
    };
    let a = projection.project_non_loop(start_sec);
    let b = projection.project_non_loop(end_sec);
    let start = projection
        .to_frame(a.min(b), frame_period_ms)
        .max(visible_start);
    let end = projection
        .to_frame(a.max(b), frame_period_ms)
        .min(visible_end);
    let Some((start, end)) = positive_interval_to_frames(start, end) else {
        return out;
    };
    let (cut_start, cut_end) = cut_flags_for_intersection(
        (span.source_start_sec, span.source_end_sec),
        (window_start, window_end),
        projection.reversed,
    );
    out.push(ProjectedInterval {
        cycle: 0,
        start_frame: start,
        end_frame_exclusive: end,
        cut_start,
        cut_end,
    });
    out
}

/// 工程帧 → 源域秒。`cycle` 只对 Loop 有意义；非 Loop 传 0。
pub fn inverse_project(
    projection: &ClipProjection,
    frame: u32,
    cycle: u32,
    frame_period_ms: f64,
) -> Option<f64> {
    let project_sec = frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let local_sec = project_sec - projection.project_start_sec;
    if local_sec < -1e-9 || local_sec > projection.project_length_sec + 1e-9 {
        return None;
    }
    let rate = projection.effective_rate.max(1e-9);
    let mut source = if projection.loop_enabled {
        let cycle_sec = projection.loop_cycle_sec?;
        let cycle_timeline_sec = cycle_sec / rate;
        let phase = (local_sec - cycle as f64 * cycle_timeline_sec) * rate;
        let anchor = projection.loop_anchor();
        if projection.reversed {
            anchor - phase
        } else {
            anchor + phase
        }
    } else {
        projection.inverse_non_loop(project_sec)
    };
    /*
     * 【不能先 clamp 再取模】旧实现把 anchor + phase 直接截到媒体末尾，导致
     * anchor 之前的回绕成员被映射到媒体尾部。正确顺序是：先得到未归一化的
     * 源坐标，再在媒体域做一次归一化（取模），最后才允许等于 mediaDuration
     * 的端点保持原值。
     */
    if let Some(duration) = projection.media_duration_sec.filter(|d| *d > 1e-9) {
        if source < -1e-9 || source > duration + 1e-9 {
            source = source.rem_euclid(duration);
        } else if (source - duration).abs() <= 1e-9 {
            source = duration;
        }
    }
    if let Some(duration) = projection.media_duration_sec {
        source = source.clamp(0.0, duration);
    }
    if source.is_finite() {
        Some(source)
    } else {
        None
    }
}

/**
 * 工程帧 → 源域秒的**局部仿射**映射（不取模、不 clamp）。
 *
 * `cycle` 必须是该投影区间自己的周期号。与 `inverse_project` 的区别：
 * - 不把 anchor + phase 再截到媒体末尾（Loop 回绕会因此跳到媒体尾部）；
 * - 不在周期边界把 end_frame_exclusive 折回 anchor；
 * - 倒放时直接按 `anchor - offset` 计算，方向由成员自己的 frame_source* 表达。
 */
pub fn source_at_projected_frame(
    projection: &ClipProjection,
    frame: u32,
    cycle: u32,
    frame_period_ms: f64,
) -> Option<f64> {
    let project_sec = frame as f64 * frame_period_ms.max(0.1) / 1000.0;
    let local_sec = project_sec - projection.project_start_sec;
    if local_sec < -1e-9 || local_sec > projection.project_length_sec + 1e-9 {
        return None;
    }
    let rate = projection.effective_rate.max(1e-9);
    if projection.loop_enabled {
        let cycle_sec = projection.loop_cycle_sec?;
        let cycle_timeline_sec = cycle_sec / rate;
        let offset_sec = local_sec - cycle as f64 * cycle_timeline_sec;
        let anchor = projection.loop_anchor();
        let mut source = if projection.reversed {
            anchor - offset_sec * rate
        } else {
            anchor + offset_sec * rate
        };
        if let Some(media) = projection.media_duration_sec.filter(|d| *d > 1e-9) {
            if source < -1e-9 || source > media + 1e-9 {
                source = source.rem_euclid(media);
            } else if (source - media).abs() <= 1e-9 {
                source = media;
            }
        }
        return source.is_finite().then_some(source);
    }
    let source = if projection.reversed {
        projection.source_window_end_sec - local_sec * rate
    } else {
        projection.source_window_start_sec + local_sec * rate
    };
    source.is_finite().then_some(source)
}

/// 把一个可能跨 Loop 周期边界的源域区间切成“每段都属于单一周期”的成员。
pub fn normalize_loop_span(projection: &ClipProjection, span: &TakeSpan) -> Vec<TakeSpan> {
    if !projection.loop_enabled {
        return vec![span.clone()];
    }
    let Some(cycle_sec) = projection.loop_cycle_sec.filter(|value| *value > 1e-9) else {
        return vec![span.clone()];
    };
    let media = projection.media_duration_sec.unwrap_or(cycle_sec);
    let anchor = projection.loop_anchor();
    let width = span.source_end_sec - span.source_start_sec;
    if width <= 1e-12 {
        return Vec::new();
    }
    if width > cycle_sec + 1e-9 {
        let mut out = Vec::new();
        let mut start = span.source_start_sec;
        let mut cycle = span.cycle;
        while start < span.source_end_sec - 1e-9 {
            let end = (start + cycle_sec).min(span.source_end_sec);
            let mut part = span.clone();
            part.cycle = cycle;
            part.source_start_sec = start;
            part.source_end_sec = end;
            out.extend(normalize_loop_span(projection, &part));
            start = end;
            cycle = cycle.saturating_add(1);
        }
        return out;
    }
    let mut wrapped: Vec<(f64, f64)> = Vec::new();
    let (lo, hi) = (span.source_start_sec, span.source_end_sec);
    if lo < -1e-9 {
        if hi <= 1e-9 {
            wrapped.push((media + lo, media + hi));
        } else {
            wrapped.push((media + lo, media));
            wrapped.push((0.0, hi));
        }
    } else if lo >= media - 1e-9 {
        wrapped.push((lo - media, hi - media));
    } else if hi > media + 1e-9 {
        wrapped.push((lo, media));
        wrapped.push((0.0, hi - media));
    } else {
        wrapped.push((lo, hi));
    }
    let mut out = Vec::new();
    for (lo, hi) in wrapped {
        if projection.reversed {
            if lo < anchor - 1e-9 && hi > anchor + 1e-9 {
                let mut first = span.clone();
                first.cycle = span.cycle;
                first.source_start_sec = anchor;
                first.source_end_sec = hi;
                let mut second = span.clone();
                second.cycle = span.cycle;
                second.source_start_sec = lo;
                second.source_end_sec = anchor;
                out.push(first);
                out.push(second);
            } else {
                let mut part = span.clone();
                part.cycle = span.cycle;
                part.source_start_sec = lo;
                part.source_end_sec = hi;
                out.push(part);
            }
        } else if lo < anchor - 1e-9 && hi > anchor + 1e-9 {
            let mut first = span.clone();
            first.cycle = span.cycle;
            first.source_start_sec = lo;
            first.source_end_sec = anchor;
            let mut second = span.clone();
            second.cycle = span.cycle.saturating_add(1);
            second.source_start_sec = anchor;
            second.source_end_sec = hi;
            out.push(first);
            out.push(second);
        } else {
            let mut part = span.clone();
            part.cycle = span.cycle;
            part.source_start_sec = lo;
            part.source_end_sec = hi;
            out.push(part);
        }
    }
    out.retain(|part| part.source_end_sec > part.source_start_sec + 1e-12);
    out
}

/// 源域宽度换算成工程帧宽度（用于过渡手柄）。
pub fn source_width_to_frames(
    projection: &ClipProjection,
    width_sec: f64,
    frame_period_ms: f64,
) -> u32 {
    let frames = width_sec.max(0.0) * 1000.0 / frame_period_ms.max(0.1) / projection.effective_rate;
    frames.round().clamp(0.0, MAX_FRAME as f64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Clip;

    fn clip(json: serde_json::Value) -> Clip {
        let mut clip: Clip = serde_json::from_value(json).unwrap();
        clip.normalize_takes();
        clip
    }

    fn span(start: f64, end: f64) -> TakeSpan {
        TakeSpan {
            clip_id: "clip".into(),
            take_id: "take".into(),
            cycle: 0,
            source_start_sec: start,
            source_end_sec: end,
            syllable_id: None,
            label: super::super::types::BlockLabel::Syllable,
        }
    }

    #[test]
    fn project_span_keeps_one_frame_for_a_sub_frame_left_overlap() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":2.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":1.0,
                "source_end_sec":3.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(0.5, 1.001), 5.0);
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start_frame, 0);
        assert_eq!(intervals[0].end_frame_exclusive, 1);
        assert!(intervals[0].cut_start);
    }

    #[test]
    fn project_span_marks_left_cut_forward_and_reverse() {
        let forward = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":1.0,
                "source_end_sec":2.0}]
        }));
        let interval = project_span(
            &ClipProjection::from_clip(&forward).unwrap(),
            &span(0.6, 1.4),
            5.0,
        )[0];
        assert!(interval.cut_start);
        assert!(!interval.cut_end);
        let reverse = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":1.0,
            "reversed":true,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,
                "source_end_sec":2.0,"reversed":true}]
        }));
        let reversed = project_span(
            &ClipProjection::from_clip(&reverse).unwrap(),
            &span(1.6, 2.4),
            5.0,
        )[0];
        assert!(reversed.cut_start);
    }

    #[test]
    fn loop_analysis_window_is_the_audible_ring() {
        let ring = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "loop_enabled":true,"loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,
                "source_end_sec":1.0,"loop_enabled":true,"duration_sec":1.0}]
        }));
        assert_eq!(
            ClipProjection::from_clip(&ring)
                .unwrap()
                .audible_source_window(),
            (0.0, 1.0)
        );
        let section = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "loop_enabled":true,"loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.25,
                "source_end_sec":1.25,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let section_projection = ClipProjection::from_clip(&section).unwrap();
        assert_eq!(section_projection.audible_source_window(), (0.25, 1.25));
    }

    #[test]
    fn audible_window_follows_rate_and_direction() {
        let forward = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":2.0,
            "playback_rate":1.0,"clip_playback_rate":2.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.5,
                "source_end_sec":8.0,"playback_rate":1.0}]
        }));
        assert_eq!(
            ClipProjection::from_clip(&forward)
                .unwrap()
                .audible_source_window(),
            (0.5, 4.5)
        );
        let reverse = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":2.0,
            "reversed":true,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,
                "source_end_sec":4.0,"reversed":true}]
        }));
        assert_eq!(
            ClipProjection::from_clip(&reverse)
                .unwrap()
                .audible_source_window(),
            (2.0, 4.0)
        );
    }

    #[test]
    fn forward_slip_and_rate_move_the_block() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":2.0,"length_sec":2.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":1.0,"source_end_sec":3.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        assert_eq!(
            project_span(&projection, &span(1.2, 1.7), 5.0)[0].start_frame,
            440
        );
        let mut slipped = clip.clone();
        slipped.takes[0].source_start_sec = 1.5;
        slipped.takes[0].source_end_sec = 3.5;
        let take = slipped.takes[0].clone();
        take.apply_to_clip(&mut slipped);
        let projection = ClipProjection::from_clip(&slipped).unwrap();
        assert_eq!(
            project_span(&projection, &span(1.7, 2.2), 5.0)[0].start_frame,
            440
        );
    }

    #[test]
    fn reverse_mirrors_and_round_trips() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":2.0,"reversed":true,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,"source_end_sec":2.0,"reversed":true}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let interval = project_span(&projection, &span(0.2, 0.6), 5.0)[0];
        assert_eq!(interval.start_frame, 280);
        let source = inverse_project(&projection, interval.start_frame, 0, 5.0).unwrap();
        assert!((source - 0.6).abs() < 0.01, "{source}");
    }

    #[test]
    fn inverse_loop_wrap_stays_inside_the_member() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":8.0,
            "loop_enabled":true,"loop_period_sec":2.27,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.266,
                "source_end_sec":2.27,"loop_enabled":true,"duration_sec":2.27}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let span = span(0.046, 0.252);
        let intervals = project_span(&projection, &span, 5.0);
        assert_eq!(intervals[0].start_frame, 410);
        assert_eq!(intervals[0].end_frame_exclusive, 451);
        for frame in intervals[0].start_frame..=intervals[0].end_frame_exclusive {
            let source = inverse_project(&projection, frame, intervals[0].cycle, 5.0).unwrap();
            assert!(
                (0.0459..=0.2521).contains(&source),
                "frame {frame} -> {source}"
            );
        }
    }

    #[test]
    fn loop_emits_each_cycle() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,"loop_enabled":true,
            "loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,"source_end_sec":1.0,"loop_enabled":true}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(0.2, 0.4), 5.0);
        assert_eq!(intervals.len(), 4);
        assert_eq!(intervals[0].start_frame, 40);
        assert_eq!(intervals[3].cycle, 3);
    }
}
