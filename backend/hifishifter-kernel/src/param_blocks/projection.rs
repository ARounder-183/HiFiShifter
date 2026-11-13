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
/// visible frame; it is not an absent block. The fabricated frame must stay
/// inside the clip's visible range: a sliver at the very end rounds past
/// `visible_end`, and emitting `[visible_end, visible_end + 1)` would both draw
/// a block outside the clip and collide with the next clip's first block —
/// `validate_view` then rejects the whole view. The loop path applies the same
/// guard (`start < visible_end`).
pub fn positive_interval_to_frames(start: u32, end: u32, visible_end: u32) -> Option<(u32, u32)> {
    (end > start)
        .then_some((start, end))
        .or_else(|| (start < MAX_FRAME && start < visible_end).then_some((start, start + 1)))
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

/// A source-domain fragment that belongs to a single Loop cycle.
///
/// The cut flags are expressed in timeline orientation: cut_start describes
/// the left edge of the fragment as the user sees it. Reverse playback
/// therefore swaps which source edge maps to which flag.
#[derive(Clone, Debug)]
pub struct LoopSpanPart {
    pub span: TakeSpan,
    pub cut_start: bool,
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
    /// duration falls back to the stored source window. A reversed Take's ring
    /// extends **downward** from its anchor (`source_at_projected_frame`,
    /// `take_geometry::forward_source_window` and the audio wrap in
    /// `mixdown::build_loop_tiled_segment` all agree on that), so its section
    /// window is `[anchor - cycle, anchor]` — the forward arm's
    /// `[anchor, anchor + cycle)` is media that never plays.
    pub fn audible_source_window(&self) -> (f64, f64) {
        let span = (self.project_length_sec.max(0.0) * self.effective_rate).max(0.0);
        let (start, end) = if self.loop_enabled {
            let anchor = self.loop_anchor();
            let cycle = self.loop_cycle_sec.unwrap_or(0.0);
            let media = self.media_duration_sec;
            // The ring wraps once the cycle reaches beyond one side of the
            // media: forward past the end, reversed below zero.
            let wraps = |anchor: f64, cycle: f64, duration: f64| {
                if self.reversed {
                    anchor - cycle < -1e-9 || anchor > duration + 1e-9
                } else {
                    anchor + cycle > duration + 1e-9 || anchor < -1e-9
                }
            };
            let whole_ring = media.filter(|duration| {
                *duration > 1e-9 && (cycle >= *duration - 1e-9 || wraps(anchor, cycle, *duration))
            });
            if let Some(duration) = whole_ring {
                (0.0, duration)
            } else if cycle > 1e-9 && anchor.is_finite() && anchor >= -1e-9 {
                if self.reversed {
                    (anchor - cycle, anchor)
                } else {
                    (anchor, anchor + cycle)
                }
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
    let mut out = Vec::new();
    if projection.loop_enabled {
        // A TakeSpan may arrive unnormalized (the golden test and editing
        // primitives call project_span directly). Splitting is therefore part
        // of projection, not something callers may skip.
        let parts = split_loop_span_parts(projection, span);
        for part in parts {
            out.extend(project_span_part(projection, &part, frame_period_ms));
        }
        out.sort_by_key(|interval| {
            (
                interval.start_frame,
                interval.end_frame_exclusive,
                interval.cycle,
            )
        });
        return out;
    }

    out.extend(project_non_loop_interval(
        projection,
        span,
        frame_period_ms,
        visible_start,
        visible_end,
    ));
    out
}

/// Project one already-normalized Loop fragment. The fragment cycle is the
/// first cycle in which it appears; later cycles repeat at the same phase.
pub fn project_span_part(
    projection: &ClipProjection,
    part: &LoopSpanPart,
    frame_period_ms: f64,
) -> Vec<ProjectedInterval> {
    let frame_period_ms = frame_period_ms.max(0.1);
    let (visible_start, visible_end) = projection.visible_frames(frame_period_ms);
    if visible_end <= visible_start {
        return Vec::new();
    }
    if !projection.loop_enabled {
        let mut intervals = project_non_loop_interval(
            projection,
            &part.span,
            frame_period_ms,
            visible_start,
            visible_end,
        );
        for interval in &mut intervals {
            interval.cut_start |= part.cut_start;
            interval.cut_end |= part.cut_end;
        }
        return intervals;
    }
    let Some(cycle_sec) = projection.loop_cycle_sec.filter(|value| *value > 1e-9) else {
        return Vec::new();
    };
    let rate = projection.effective_rate.max(1e-9);
    let cycle_timeline_sec = cycle_sec / rate;
    if cycle_timeline_sec <= 1e-9 {
        return Vec::new();
    }
    let anchor = projection.loop_anchor();
    let phase = if projection.reversed {
        (anchor - part.span.source_end_sec).rem_euclid(cycle_sec)
    } else {
        (part.span.source_start_sec - anchor).rem_euclid(cycle_sec)
    };
    let width_sec = (part.span.source_end_sec - part.span.source_start_sec).max(0.0);
    let length_sec = (width_sec / rate)
        .min(((cycle_sec - phase) / rate).max(0.0))
        .max(0.0);
    if length_sec <= SOURCE_EPS {
        return Vec::new();
    }
    let clip_start = projection.project_start_sec;
    let clip_end = clip_start + projection.project_length_sec;
    let mut out = Vec::new();
    let mut cycle = part.span.cycle;
    let mut start_local = phase / rate + cycle_timeline_sec * part.span.cycle as f64;
    // At most 20,000 cycles; malformed input must not become an infinite loop.
    while start_local < projection.project_length_sec - WINDOW_EPS && out.len() < 20_000 {
        let raw_start_sec = clip_start + start_local;
        let raw_end_sec = raw_start_sec + length_sec;
        let start = projection
            .to_frame(raw_start_sec, frame_period_ms)
            .max(visible_start);
        let mut end = projection
            .to_frame(raw_end_sec, frame_period_ms)
            .min(visible_end);
        // Rounding may collapse a positive source interval to zero frames.
        // A visible block still deserves one frame of pointer target.
        if end <= start
            && raw_end_sec > raw_start_sec + SOURCE_EPS
            && raw_start_sec < clip_end - WINDOW_EPS
            && raw_end_sec > clip_start + WINDOW_EPS
            && start < visible_end
        {
            end = start.saturating_add(1).min(visible_end);
        }
        if end > start {
            out.push(ProjectedInterval {
                cycle,
                start_frame: start,
                end_frame_exclusive: end,
                cut_start: part.cut_start || raw_start_sec < clip_start - WINDOW_EPS,
                cut_end: part.cut_end || raw_end_sec > clip_end + WINDOW_EPS,
            });
        }
        start_local += cycle_timeline_sec;
        cycle = cycle.saturating_add(1);
    }
    out
}

fn project_non_loop_interval(
    projection: &ClipProjection,
    span: &TakeSpan,
    frame_period_ms: f64,
    visible_start: u32,
    visible_end: u32,
) -> Vec<ProjectedInterval> {
    // The audible source window is the single source of truth for analysis,
    // cache adoption and projection. Never recompute a second affine window.
    let (window_start, window_end) = projection.audible_source_window();
    let Some((start_sec, end_sec)) = window_intersection(
        (span.source_start_sec, span.source_end_sec),
        (window_start, window_end),
    ) else {
        return Vec::new();
    };
    let a = projection.project_non_loop(start_sec);
    let b = projection.project_non_loop(end_sec);
    let start = projection
        .to_frame(a.min(b), frame_period_ms)
        .max(visible_start);
    let end = projection
        .to_frame(a.max(b), frame_period_ms)
        .min(visible_end);
    let Some((start, end)) = positive_interval_to_frames(start, end, visible_end) else {
        return Vec::new();
    };
    let (cut_start, cut_end) = cut_flags_for_intersection(
        (span.source_start_sec, span.source_end_sec),
        (window_start, window_end),
        projection.reversed,
    );
    vec![ProjectedInterval {
        cycle: 0,
        start_frame: start,
        end_frame_exclusive: end,
        cut_start,
        cut_end,
    }]
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
    split_loop_span_parts(projection, span)
        .into_iter()
        .map(|part| part.span)
        .collect()
}

fn mark_low_edge_cut(part: &mut LoopSpanPart, reversed: bool) {
    if reversed {
        part.cut_end = true;
    } else {
        part.cut_start = true;
    }
}

fn mark_high_edge_cut(part: &mut LoopSpanPart, reversed: bool) {
    if reversed {
        part.cut_start = true;
    } else {
        part.cut_end = true;
    }
}

/// Split any TakeSpan into fragments that each live inside one Loop cycle.
///
/// This is the single source of truth for Loop fragmentation. It handles the
/// anchor seam (head and tail belong to the same cycle), media wrap-around,
/// and pathological spans wider than one cycle (deterministic chunking).
///
/// cut_start / cut_end are timeline-oriented: they describe the left and right
/// visible edges. Reverse playback swaps the source edges that map to those
/// flags.
pub fn split_loop_span_parts(projection: &ClipProjection, span: &TakeSpan) -> Vec<LoopSpanPart> {
    if !projection.loop_enabled {
        return vec![LoopSpanPart {
            span: span.clone(),
            cut_start: false,
            cut_end: false,
        }];
    }
    let Some(cycle_sec) = projection.loop_cycle_sec.filter(|value| *value > 1e-9) else {
        return vec![LoopSpanPart {
            span: span.clone(),
            cut_start: false,
            cut_end: false,
        }];
    };
    if !span.source_start_sec.is_finite() || !span.source_end_sec.is_finite() {
        return Vec::new();
    }
    let width = span.source_end_sec - span.source_start_sec;
    if width <= SOURCE_EPS {
        return Vec::new();
    }
    let media = projection
        .media_duration_sec
        .filter(|value| value.is_finite() && *value > WINDOW_EPS)
        .unwrap_or(cycle_sec);
    let anchor = projection.loop_anchor();
    let mut out = Vec::new();
    if width > cycle_sec + WINDOW_EPS {
        let mut start = span.source_start_sec;
        let mut cycle = span.cycle;
        let mut index = 0u32;
        while start < span.source_end_sec - SOURCE_EPS && index < 20_000 {
            let end = (start + cycle_sec).min(span.source_end_sec);
            let last = end >= span.source_end_sec - SOURCE_EPS;
            let (cut_start, cut_end) = if projection.reversed {
                (!last, index > 0)
            } else {
                (index > 0, !last)
            };
            let mut chunk = span.clone();
            chunk.cycle = cycle;
            chunk.source_start_sec = start;
            chunk.source_end_sec = end;
            out.extend(split_loop_piece(
                projection,
                LoopSpanPart {
                    span: chunk,
                    cut_start,
                    cut_end,
                },
                media,
                anchor,
            ));
            start = end;
            cycle = cycle.saturating_add(1);
            index = index.saturating_add(1);
        }
    } else {
        out.extend(split_loop_piece(
            projection,
            LoopSpanPart {
                span: span.clone(),
                cut_start: false,
                cut_end: false,
            },
            media,
            anchor,
        ));
    }
    out.sort_by(|a, b| {
        a.span
            .source_start_sec
            .total_cmp(&b.span.source_start_sec)
            .then(a.span.source_end_sec.total_cmp(&b.span.source_end_sec))
            .then(a.span.cycle.cmp(&b.span.cycle))
    });
    out
}

fn split_loop_piece(
    projection: &ClipProjection,
    part: LoopSpanPart,
    media: f64,
    anchor: f64,
) -> Vec<LoopSpanPart> {
    let reversed = projection.reversed;
    let mut queue = vec![part];
    let mut wrapped = Vec::new();
    let mut guard = 0usize;
    while let Some(current) = queue.pop() {
        guard = guard.saturating_add(1);
        if guard > 20_000 {
            // 有界路径必须有读数：静默 break 会把队列里剩余的片段直接丢掉，
            // 表现为"块少了一截"。只可能由退化输入触发（如 duration_sec≈1e-8
            // 的损坏工程），与 [pitch:trim] / [hnsep] 的有界路径同一待遇。
            // （param_blocks 声明在 lib.rs 的宏定义之前，文本作用域够不到
            // log_warn_limited!，这里直接调用它包装的限流实现。）
            crate::log_limiter::emit_limited(
                log::Level::Warn,
                file!(),
                line!(),
                format_args!(
                    "[param_blocks] loop split hit the 20,000-piece guard: clip '{}' \
                     dropped {} queued fragment(s); media={:.6}s anchor={:.6}s",
                    projection.clip_id,
                    queue.len() + 1,
                    media,
                    anchor
                ),
            );
            break;
        }
        let lo = current.span.source_start_sec;
        let hi = current.span.source_end_sec;
        if hi <= lo + SOURCE_EPS {
            continue;
        }
        if lo < -WINDOW_EPS {
            if hi <= WINDOW_EPS {
                let mut shifted = current;
                shifted.span.source_start_sec = media + lo;
                shifted.span.source_end_sec = media + hi;
                queue.push(shifted);
            } else {
                let mut tail = current.clone();
                tail.span.source_start_sec = media + lo;
                tail.span.source_end_sec = media;
                mark_high_edge_cut(&mut tail, reversed);
                let mut head = current;
                head.span.source_start_sec = 0.0;
                head.span.source_end_sec = hi;
                mark_low_edge_cut(&mut head, reversed);
                queue.push(tail);
                queue.push(head);
            }
        } else if lo >= media - WINDOW_EPS {
            let mut shifted = current;
            shifted.span.source_start_sec = lo - media;
            shifted.span.source_end_sec = hi - media;
            queue.push(shifted);
        } else if hi > media + WINDOW_EPS {
            let mut tail = current.clone();
            tail.span.source_end_sec = media;
            mark_high_edge_cut(&mut tail, reversed);
            let mut head = current;
            head.span.source_start_sec = 0.0;
            head.span.source_end_sec = hi - media;
            mark_low_edge_cut(&mut head, reversed);
            queue.push(tail);
            queue.push(head);
        } else {
            wrapped.push(current);
        }
    }
    let mut out = Vec::new();
    for piece in wrapped {
        let lo = piece.span.source_start_sec;
        let hi = piece.span.source_end_sec;
        if lo < anchor - WINDOW_EPS && hi > anchor + WINDOW_EPS {
            // Forward: [lo, anchor] is the tail at the end of this very cycle,
            // [anchor, hi] is the head at its start. Reverse mirrors the flags
            // but keeps both fragments in the same cycle.
            let mut left = piece.clone();
            left.span.source_end_sec = anchor;
            mark_high_edge_cut(&mut left, reversed);
            let mut right = piece;
            right.span.source_start_sec = anchor;
            mark_low_edge_cut(&mut right, reversed);
            out.push(left);
            out.push(right);
        } else {
            out.push(piece);
        }
    }
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
    fn reversed_section_loop_window_descends_from_the_anchor() {
        // 倒放的 section loop：锚点在窗口末端，环从锚点向下延伸 —— 与
        // source_at_projected_frame / take_geometry / mixdown 三处消费方一致。
        let section = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "reversed":true,"loop_enabled":true,"loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.25,
                "source_end_sec":1.25,"reversed":true,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let projection = ClipProjection::from_clip(&section).unwrap();
        assert_eq!(projection.audible_source_window(), (0.25, 1.25));
        // 越过源起点的 cycle 回绕整圈媒体：环从锚点向下越过 0。
        let wrapped = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "reversed":true,"loop_enabled":true,"loop_period_sec":2.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,
                "source_end_sec":1.0,"reversed":true,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let wrapped_projection = ClipProjection::from_clip(&wrapped).unwrap();
        assert_eq!(wrapped_projection.audible_source_window(), (0.0, 4.0));
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
    fn a_sub_frame_sliver_at_the_clip_end_stays_inside_the_clip() {
        // 交叠 [2.9995, 3.0] 取整后两端都落在 Clip 的最后一帧上：非 Loop 路径
        // 不得把它"提升"成 [visible_end, visible_end+1) —— 那会画出 Clip 之外的
        // 一帧，并与相邻 Clip 的首个块在 validate_view 里撞车（loop 路径
        // 早已用 start < visible_end 守住同一种情况）。
        let edge = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":2.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":1.0,"source_end_sec":3.0}]
        }));
        let projection = ClipProjection::from_clip(&edge).unwrap();
        assert!(project_span(&projection, &span(2.9995, 3.5), 5.0).is_empty());
        // 左缘的亚帧残片仍然保留（fabricate 落在可见范围内）。
        let sliver = project_span(&projection, &span(1.0, 1.0004), 5.0);
        assert_eq!(sliver.len(), 1);
        assert_eq!(
            (sliver[0].start_frame, sliver[0].end_frame_exclusive),
            (0, 1)
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
    fn loop_anchor_crossing_keeps_head_in_current_cycle() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":8.0,
            "loop_enabled":true,"loop_period_sec":2.27,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.266,
                "source_end_sec":2.27,"loop_enabled":true,"duration_sec":2.27}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(0.2, 0.4), 5.0);
        let got: Vec<(u32, u32, bool, bool)> = intervals
            .iter()
            .map(|interval| {
                (
                    interval.start_frame,
                    interval.end_frame_exclusive,
                    interval.cut_start,
                    interval.cut_end,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 27, true, false),
                (441, 454, false, true),
                (454, 481, true, false),
                (895, 908, false, true),
                (908, 935, true, false),
                (1349, 1362, false, true),
                (1362, 1389, true, false),
            ]
        );
    }

    #[test]
    fn loop_left_trim_short_clip_keeps_first_note() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.3,"length_sec":3.7,
            "loop_enabled":true,"loop_period_sec":4.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.3,
                "source_end_sec":4.0,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(0.2, 0.6), 5.0);
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start_frame, 60);
        assert_eq!(intervals[0].end_frame_exclusive, 120);
        assert!(intervals[0].cut_start);
        assert!(!intervals[0].cut_end);
    }

    #[test]
    fn loop_anchor_at_exact_span_edges_does_not_split() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "loop_enabled":true,"loop_period_sec":4.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.3,
                "source_end_sec":4.0,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let start = project_span(&projection, &span(0.3, 0.6), 5.0);
        assert_eq!(start.len(), 1);
        assert_eq!(
            (start[0].start_frame, start[0].end_frame_exclusive),
            (0, 60)
        );
        assert!(!start[0].cut_start && !start[0].cut_end);
        let end = project_span(&projection, &span(0.2, 0.3), 5.0);
        assert_eq!(end.len(), 1);
        assert_eq!((end[0].start_frame, end[0].end_frame_exclusive), (780, 800));
        assert!(!end[0].cut_start && !end[0].cut_end);
    }

    #[test]
    fn loop_sub_frame_head_survives_as_one_frame() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":1.0,
            "loop_enabled":true,"loop_period_sec":4.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.3,
                "source_end_sec":4.0,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(0.2999, 0.3002), 5.0);
        assert_eq!(intervals.len(), 1);
        assert_eq!(
            (intervals[0].start_frame, intervals[0].end_frame_exclusive),
            (0, 1)
        );
        assert!(intervals[0].cut_start);
    }

    #[test]
    fn section_loop_anchor_crossing_uses_the_section_cycle() {
        // SECTION 循环：周期 1s 小于媒体时长 4s，锚点仍按 source_start 起算。
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "loop_enabled":true,"loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.25,
                "source_end_sec":1.25,"loop_enabled":true,"duration_sec":4.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(0.2, 0.4), 5.0);
        let got: Vec<(u32, u32)> = intervals
            .iter()
            .map(|interval| (interval.start_frame, interval.end_frame_exclusive))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 30),
                (190, 200),
                (200, 230),
                (390, 400),
                (400, 430),
                (590, 600),
                (600, 630),
                (790, 800),
            ]
        );
        assert!(intervals[0].cut_start);
        assert!(intervals[1].cut_end);
    }

    #[test]
    fn reversed_loop_anchor_crossing_keeps_both_fragments() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":2.0,
            "reversed":true,"loop_enabled":true,"loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.25,
                "source_end_sec":1.25,"reversed":true,"loop_enabled":true,
                "duration_sec":4.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let intervals = project_span(&projection, &span(1.0, 1.5), 5.0);
        let got: Vec<(u32, u32, bool, bool)> = intervals
            .iter()
            .map(|interval| {
                (
                    interval.start_frame,
                    interval.end_frame_exclusive,
                    interval.cut_start,
                    interval.cut_end,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 50, true, false),
                (150, 200, false, true),
                (200, 250, true, false),
                (350, 400, false, true),
            ]
        );
    }

    #[test]
    fn project_span_starts_at_the_span_cycle_for_user_members() {
        let clip = clip(serde_json::json!({
            "id":"clip","track_id":"root","name":"c","start_sec":0.0,"length_sec":4.0,
            "loop_enabled":true,"loop_period_sec":1.0,
            "takes":[{"id":"take","source_path":"a","source_start_sec":0.0,
                "source_end_sec":1.0,"loop_enabled":true,"duration_sec":1.0}]
        }));
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let mut member = span(0.2, 0.4);
        member.cycle = 2;
        let intervals = project_span(&projection, &member, 5.0);
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].cycle, 2);
        assert_eq!(intervals[0].start_frame, 440);
        assert_eq!(intervals[1].cycle, 3);
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
