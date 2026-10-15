//! 把捕获到的段投影成 [`TimelineState`]：解决覆盖关系、合并相邻段、生成片段。
//!
//! # 为什么这一层必须存在
//!
//! 音频线程只负责"把每一块写到它该在的工程位置上"（`capture.rs`）。段之间**可以
//! 重叠**：宿主 seek 回退、循环回卷、以及**再录一遍**都会让同一段工程时间被录到两次。
//! 合并是**视图**的事，不是音频线程的事 —— 那里既不能分配，也不该承担区间运算。
//!
//! # 覆盖规则：后来的胜出
//!
//! 段按 `(take, segment)` 排序后依次铺开；每铺一段，就把它覆盖到的已有区间**剪掉**。
//! `(take, segment)` 就是**录制顺序**：take 号随每次录制递增，段号在 take 内递增。
//! 于是"第 2 圈覆盖第 1 圈"与"补录的那一句覆盖它盖住的部分"都是同一个规则的自然结果，
//! 不需要为循环或补录写任何特例。
//!
//! # 单位
//!
//! 段是**工程采样域**的（`projectTimeSamples` 的域）。秒 = 帧 / 采样率，三个方向
//! 各自独立换算（段起点、段长、段内偏移），不合到一起算 —— 合起来写会掩盖
//! "用了哪个率"这个唯一重要的问题。

use super::capture::{ClosedReason, Segment, SegmentLoss};
use hifishifter_kernel::state::TimelineState;
use serde_json::json;
use std::ops::Range;

/// Transfer 时间轴上唯一的轨道 id。
///
/// 【为什么只有一条轨】Transfer 捕获的是"到达本插件的混音"这一条流：宿主的输入总线
/// 不给逐轨素材（要逐轨就得多实例）。因此一条轨是诚实的形状，而"多次录制"表达成
/// **时间上的覆盖**（后录的胜出），不是并列的轨。
///
/// 【为什么是常量而不是按宿主派生】宿主连名字都没给我们（没有原生扩展），
/// 而且 Transfer 形态下素材只有一条流。派生一个假名字进日志与工程文件，比一个
/// 诚实的固定名字更糟。
pub(crate) const TRANSFER_TRACK_ID: &str = "transfer-track";
/// 轨道显示名。
pub(crate) const TRANSFER_TRACK_NAME: &str = "Transfer";

/// 跨 take 投影的输入：一个 take 的 id 与它的段。
#[derive(Clone, Copy)]
pub(crate) struct ProjectionTake<'a> {
    pub id: u64,
    pub segments: &'a [Segment],
}

/// 工程区间的帧数。
///
/// 【为什么写成一个函数而不是到处 `(end - start) as usize`】`Range` 没有 `len()`，
/// 而减法在 `i64` 上写错顺序（`start - end`）会得到一个巨大的无符号数。一处定义。
fn frames_of(span: &Range<i64>) -> usize {
    (span.end - span.start).max(0) as usize
}

/// 一段已解决覆盖关系的工程区间，指向某个 take 的某个段的某个偏移。
#[derive(Debug, Clone)]
struct Piece<'a> {
    /// 工程采样区间（左闭右开）。
    span: Range<i64>,
    /// 所属 take 号（= 录制顺序）。
    take: u64,
    /// 素材段。判静音与取采样率都要它。
    segment: &'a Segment,
    /// 该区间起点在段内的帧偏移。
    offset: usize,
}

/// 一次投影的结果摘要（给状态栏与诊断用）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TimelineSummary {
    /// 生成的片段数。
    pub clips: usize,
    /// 片段覆盖的工程帧数（不含静音空洞）。
    pub covered_frames: u64,
    /// 整条时间线跨越的工程帧数（含空洞）。
    pub spanned_frames: u64,
    /// 被丢弃的倒放/非法段数。
    pub dropped_segments: usize,
}

/// 把**全部** take 的段投影成一条时间线。`bpm` 由调用方给出（宿主 tempo 未知时用默认值）。
///
/// 【为什么输入是全部 take 而不是"活动 take"】用户补录一句时的预期是"那一句被替换"，
/// 而不是"之前录的全没了"。把全部段放在同一张覆盖图上，这条语义就是规则本身的结果；
/// 而"活动 take 单选"会让第二次录制把整条时间轴换掉。
///
/// 【为什么非法段是"丢弃"而不是"报错"】一段含 NaN 的音频不该让整条时间线消失 ——
/// 其余段是好的，用户要能听见它们。丢弃并计数，与 `render/input.rs` 的既有纪律一致。
pub(crate) fn project(takes: &[ProjectionTake<'_>], bpm: f64) -> (TimelineState, TimelineSummary) {
    let mut seeds: Vec<Piece<'_>> = Vec::new();
    let mut dropped_segments = 0usize;
    for take in takes {
        for segment in take.segments {
            if segment.span().is_empty() || segment.has_nonfinite() {
                dropped_segments += 1;
                continue;
            }
            seeds.push(Piece {
                span: segment.span(),
                take: take.id,
                segment,
                offset: 0,
            });
        }
    }

    let mut summary = TimelineSummary {
        dropped_segments,
        ..TimelineSummary::default()
    };

    let mut timeline = TimelineState {
        bpm: if bpm.is_finite() && bpm > 0.0 { bpm } else { 120.0 },
        ..TimelineState::default()
    };
    // 【为什么必须先清空】`TimelineState::default()` 自带一条 "Main" 轨（它服务于
    // 独立 App 的"新建工程"）。留着它会多出一条永远没有片段的轨道 —— 用户会看到
    // 一个空轨，并以为录制漏了一条。
    timeline.tracks.clear();
    timeline.selected_track_id = Some(TRANSFER_TRACK_ID.to_owned());
    timeline.tracks.push(
        serde_json::from_value(json!({
            "id": TRANSFER_TRACK_ID,
            "name": TRANSFER_TRACK_NAME,
            "order": 0,
        }))
        .expect("Transfer 轨道的必填字段是固定的"),
    );

    let pieces = coalesce(resolve(seeds));

    let mut index = 0_usize;
    for piece in pieces {
        let segment = piece.segment;
        let rate = segment.sample_rate.max(1) as f64;
        let frames = frames_of(&piece.span);
        summary.spanned_frames += frames as u64;
        // 【为什么逐片判静音而不是逐段】一个段可以是"前半段有声、后半段静音"
        // （用户在某处停止演奏）。按段判定会把有声的部分一起丢掉。
        if is_silent(segment, piece.offset, frames) {
            continue;
        }
        summary.covered_frames += frames as u64;
        index += 1;
        let source = source_id(piece.take, segment.id);
        let start_sec = piece.span.start as f64 / rate;
        let length_sec = frames as f64 / rate;
        let source_start_sec = piece.offset as f64 / rate;
        let source_end_sec = (piece.offset + frames) as f64 / rate;
        let id = clip_id(piece.take, segment.id, piece.offset);
        let name = format!("#{index} @ {start_sec:.3}s");
        let mut clip: hifishifter_kernel::state::Clip = serde_json::from_value(json!({
            "id": id,
            "track_id": TRANSFER_TRACK_ID,
            "name": name,
            "start_sec": start_sec,
            "length_sec": length_sec,
        }))
        .expect("片段的必填字段是固定的");
        let take: hifishifter_kernel::state::ClipTake = serde_json::from_value(json!({
            "id": format!("{id}_take_1"),
            "name": name,
            "source_path": source,
            "source_start_sec": source_start_sec,
            "source_end_sec": source_end_sec,
            "source_sample_rate": segment.sample_rate,
            "duration_sec": frames_of(&segment.span()) as f64 / rate,
            "duration_frames": frames_of(&segment.span()) as u64,
            "playback_rate": 1.0,
        }))
        .expect("take 的必填字段是固定的");
        clip.takes.push(take);
        clip.active_take_id = Some(format!("{id}_take_1"));
        clip.normalize_takes();
        timeline.clips.push(clip);
    }
    timeline.clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
    summary.clips = timeline.clips.len();
    timeline.project_sec = timeline
        .clips
        .iter()
        .map(|clip| clip.start_sec + clip.length_sec)
        .fold(0.0_f64, f64::max);
    (timeline, summary)
}

/// 单个 take 的投影（便利入口；语义与 [`project`] 逐字段一致）。
pub(crate) fn build(segments: &[Segment], take_id: u64, bpm: f64) -> (TimelineState, TimelineSummary) {
    project(
        &[ProjectionTake {
            id: take_id,
            segments,
        }],
        bpm,
    )
}

/// 素材 id。段与素材一一对应：一个段的 PCM 就是一份源。
///
/// 【为什么带 take 号】同一个段的音频在两次 take 里是不同的录制结果，共用一个 id
/// 会让"上一次 take 的源"被这一次引用 —— 而它们的内容不同。
pub(crate) fn source_id(take_id: u64, segment_id: u64) -> String {
    format!("transfer://{take_id}/{segment_id}")
}

/// 片段 id。**稳定**：同一个 take 的同一个段的同一个偏移永远是同一个 id。
///
/// 【为什么键是 `(take, segment, offset)` 而不是投影序号】补录一次会让后面的片整体
/// 后移，序号随之全变 —— 用户在那个片段上的选择与编辑会在重新投影后错位到别的片段上。
/// `(take, segment, offset)` 只描述"这是哪一段素材的哪一截"，与投影结果无关：没被覆盖
/// 的片保持 id 不变，被覆盖的片本来就该换一个身份。
pub(crate) fn clip_id(take_id: u64, segment_id: u64, offset: usize) -> String {
    format!("transfer-clip-{take_id}-{segment_id}-{offset}")
}

/// 区间是否全零。
fn is_silent(segment: &Segment, offset: usize, frames: usize) -> bool {
    let end = (offset + frames).min(segment.planes.first().map_or(0, Vec::len));
    let Some(plane) = segment.planes.first() else {
        return true;
    };
    plane[offset.min(plane.len())..end]
        .iter()
        .all(|sample| *sample == 0.0)
}

/// 按"后来的胜出"把段铺成互不相交的区间。
fn resolve(mut seeds: Vec<Piece<'_>>) -> Vec<Piece<'_>> {
    // 【排序键是录制顺序 `(take, segment)` 而不是 `origin`】"后来者胜出"里的"后来"
    // 指的是**录制顺序**，而 take 号与段号正是录制顺序。按位置排序会把规则弄反：
    // 位置靠前但录得晚的段会先被铺下，再被位置靠后但录得早的段覆盖 —— 于是循环录制的
    // **最后一圈反而输给第一圈**，补录的一句也会输给它替换掉的那一句。
    seeds.sort_by_key(|piece| (piece.take, piece.segment.id));
    let mut pieces: Vec<Piece<'_>> = Vec::new();
    for seed in seeds {
        let span = seed.span.clone();
        let mut next: Vec<Piece<'_>> = Vec::with_capacity(pieces.len() + 1);
        for piece in pieces.drain(..) {
            if piece.span.end <= span.start || piece.span.start >= span.end {
                next.push(piece);
                continue;
            }
            // 左残段：偏移不变（它离段起点还是那么远）。
            if piece.span.start < span.start {
                next.push(Piece {
                    span: piece.span.start..span.start,
                    ..piece.clone()
                });
            }
            // 右残段：偏移要加上被切掉的那一段长度。
            if piece.span.end > span.end {
                let consumed = (span.end - piece.span.start) as usize;
                next.push(Piece {
                    span: span.end..piece.span.end,
                    offset: piece.offset + consumed,
                    ..piece.clone()
                });
            }
        }
        next.push(seed);
        next.sort_by_key(|piece| piece.span.start);
        pieces = next;
    }
    pieces
}

/// 合并首尾相接且源偏移也连续的片：它们是同一个段里的一段连续音频。
///
/// 【为什么要求**两个**方向都连续】只要求工程方向连续会把"工程上相邻但源上跳了"
/// 的两段拼成一个 clip，而 clip 的源窗是**连续**区间 —— 那样源窗会跨过一段不存在
/// 的素材，渲染出错误的内容。
///
/// 【为什么还要求同一个 take】不同 take 的段是两次不同的录制，即使工程上首尾相接、
/// 源偏移也连续，把它们拼成一个 clip 会让"这是哪一次录的"消失 —— 而 take 号是
/// 曲线上与清单里的身份。
fn coalesce(pieces: Vec<Piece<'_>>) -> Vec<Piece<'_>> {
    let mut out: Vec<Piece<'_>> = Vec::new();
    for piece in pieces {
        match out.last_mut() {
            Some(last)
                if last.take == piece.take
                    && last.segment.id == piece.segment.id
                    && last.span.end == piece.span.start
                    && last.offset + frames_of(&last.span) == piece.offset =>
            {
                last.span.end = piece.span.end;
            }
            _ => out.push(piece),
        }
    }
    out
}

/// 段的结束原因是否意味着"用户需要知道"。
pub(crate) fn notable_reasons(segments: &[Segment]) -> Vec<ClosedReason> {
    let mut reasons: Vec<ClosedReason> = segments
        .iter()
        .map(|segment| segment.closed)
        .filter(|reason| reason.is_notable())
        .collect();
    reasons.dedup();
    reasons
}

/// 段集合里有没有"时间倒退"的痕迹（循环回卷）。
pub(crate) fn has_backward(segments: &[Segment]) -> bool {
    segments
        .iter()
        .any(|segment| matches!(segment.loss, SegmentLoss::Backward))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::capture::SegmentLoss;

    fn segment(id: u64, origin: i64, values: &[f32]) -> Segment {
        Segment {
            id,
            origin,
            sample_rate: 1000,
            planes: vec![values.to_vec(), values.to_vec()],
            filled: values.len(),
            loss: SegmentLoss::First,
            peak: [0.0; 2],
            closed: ClosedReason::Stopped,
            _reservation: None,
        }
    }

    /// 单 take 的便利投影。
    fn one(take: u64, segments: &[Segment]) -> (TimelineState, TimelineSummary) {
        build(segments, take, 120.0)
    }

    /// 不相交的两段各生成一个片段，且**位置就是工程位置**。
    #[test]
    fn disjoint_segments_become_clips_at_their_project_positions() {
        let segments = vec![segment(1, 0, &[0.1; 10]), segment(2, 100, &[0.2; 10])];
        let (timeline, summary) = one(7, &segments);
        assert_eq!(summary.clips, 2);
        assert_eq!(timeline.clips.len(), 2);
        // 1000 Hz 下 100 帧 = 0.1 秒。
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!((timeline.clips[0].length_sec - 0.01).abs() < 1e-9);
        assert!((timeline.clips[1].start_sec - 0.1).abs() < 1e-9);
    }

    /// **重叠时后来的胜出**，且被覆盖的部分从胜出者的偏移 0 开始。
    #[test]
    fn a_later_segment_overwrites_an_earlier_overlap() {
        // 段 1 覆盖 [0,20)，段 2 覆盖 [10,30)。
        let segments = vec![segment(1, 0, &[1.0; 20]), segment(2, 10, &[2.0; 20])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 2, "应切成 [0,10) 与 [10,30) 两片");
        // 第一片：段 1 的 [0,10)。
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!((timeline.clips[0].length_sec - 0.01).abs() < 1e-9);
        assert!((timeline.clips[0].source_start_sec - 0.0).abs() < 1e-9);
        // 第二片：段 2 的 [10,30)，偏移 0。
        assert!((timeline.clips[1].start_sec - 0.01).abs() < 1e-9);
        assert!((timeline.clips[1].length_sec - 0.02).abs() < 1e-9);
        assert!((timeline.clips[1].source_start_sec - 0.0).abs() < 1e-9);
        assert!(timeline.clips[1]
            .takes
            .first()
            .is_some_and(|take| take.source_path.as_deref() == Some("transfer://1/2")));
    }

    /// 后来的段完全盖住早先的段 ⇒ 早先那段整个消失。
    #[test]
    fn a_later_segment_can_swallow_an_earlier_one_whole() {
        let segments = vec![segment(1, 10, &[1.0; 10]), segment(2, 0, &[2.0; 40])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].length_sec - 0.04).abs() < 1e-9);
        assert!(timeline.clips[0]
            .takes
            .first()
            .is_some_and(|take| take.source_path.as_deref() == Some("transfer://1/2")));
    }

    /// **补录一句只替换它盖住的那一段**（跨 take 的 later-wins）。
    ///
    /// 【为什么这是本模块最重要的契约】第二次录制此前会**整条换掉**时间轴（活动 take
    /// 单选）：用户补录一句，之前录的全不见了。这里钉住的是"补录的那一句胜出、其余原样
    /// 保留"，并且剩下的片仍然指向**原来那一次**录制的素材。
    #[test]
    fn a_second_take_replaces_only_the_range_it_covers() {
        let first = vec![segment(1, 0, &[1.0; 100])];
        let second = vec![segment(1, 40, &[2.0; 20])];
        let (timeline, summary) = project(
            &[
                ProjectionTake {
                    id: 1,
                    segments: &first,
                },
                ProjectionTake {
                    id: 2,
                    segments: &second,
                },
            ],
            120.0,
        );
        assert_eq!(summary.clips, 3, "头 / 补录 / 尾");
        let spans: Vec<(f64, f64)> = timeline
            .clips
            .iter()
            .map(|clip| (clip.start_sec, clip.length_sec))
            .collect();
        assert!((spans[0].0 - 0.0).abs() < 1e-9 && (spans[0].1 - 0.04).abs() < 1e-9);
        assert!((spans[1].0 - 0.04).abs() < 1e-9 && (spans[1].1 - 0.02).abs() < 1e-9);
        assert!((spans[2].0 - 0.06).abs() < 1e-9 && (spans[2].1 - 0.04).abs() < 1e-9);
        // 补录那一片取第二次录制的素材；两侧仍然是第一次的。
        let sources: Vec<String> = timeline
            .clips
            .iter()
            .map(|clip| clip.active_take().source_path.clone().unwrap_or_default())
            .collect();
        assert_eq!(
            sources,
            vec!["transfer://1/1", "transfer://2/1", "transfer://1/1"],
            "补录只替换它盖住的那一段，两侧仍是第一次录制"
        );
        // 尾片的源偏移必须跟着被切掉的长度走（否则源窗会跨过不存在的素材）。
        let tail = timeline.clips[2].active_take();
        assert!((tail.source_start_sec - 0.06).abs() < 1e-9);
    }

    /// 全零区间不产生片段；有声区间照常。
    #[test]
    fn silent_regions_produce_no_clip() {
        let segments = vec![segment(1, 0, &[0.0; 10]), segment(2, 10, &[0.5; 10])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].start_sec - 0.01).abs() < 1e-9);
    }

    /// 同一个段里连续的片合并成一个片段（不是一片一个）。
    #[test]
    fn contiguous_pieces_from_one_segment_merge_into_a_single_clip() {
        let segments = vec![segment(1, 0, &[0.3; 30])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].length_sec - 0.03).abs() < 1e-9);
    }

    /// 含非有限采样的段被丢弃，其余照常 —— 一段坏音频不该让整条时间线消失。
    #[test]
    fn a_segment_with_nonfinite_samples_is_dropped_without_losing_the_rest() {
        let mut bad = segment(1, 0, &[0.1; 10]);
        bad.planes[0][3] = f32::NAN;
        let segments = vec![bad, segment(2, 20, &[0.2; 10])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.dropped_segments, 1);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].start_sec - 0.02).abs() < 1e-9);
    }

    /// 片段 id 稳定：没被覆盖的片在补录之后仍然是同一个 id（否则用户的编辑会错位）。
    #[test]
    fn clip_ids_are_stable_across_reprojection() {
        let segments = vec![segment(1, 0, &[0.1; 10]), segment(2, 100, &[0.2; 10])];
        let first = one(42, &segments).0;
        let second = one(42, &segments).0;
        let ids = |timeline: &TimelineState| {
            timeline
                .clips
                .iter()
                .map(|clip| clip.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&first), ids(&second));
        assert_eq!(
            ids(&first),
            vec!["transfer-clip-42-1-0", "transfer-clip-42-2-0"]
        );

        // 再录一遍（盖住后半段）：前一片的 id 不变，尾片的偏移变了 ⇒ 换 id。
        let overlay = vec![segment(1, 100, &[0.3; 10])];
        let merged = project(
            &[
                ProjectionTake {
                    id: 42,
                    segments: &segments,
                },
                ProjectionTake {
                    id: 43,
                    segments: &overlay,
                },
            ],
            120.0,
        )
        .0;
        assert_eq!(ids(&merged)[0], "transfer-clip-42-1-0", "未被覆盖的片保持 id");
    }

    /// 投影出的时间线必须能被内核的渲染入口接受（源窗与几何自洽）。
    #[test]
    fn the_projection_is_renderable() {
        let segments = vec![segment(1, 0, &[0.1; 10]), segment(2, 100, &[0.2; 10])];
        let (timeline, _) = one(1, &segments);
        assert_eq!(timeline.tracks.len(), 1);
        for clip in &timeline.clips {
            assert!(clip.length_sec > 0.0 && clip.length_sec.is_finite());
            assert!(clip.start_sec >= 0.0);
            let take = clip.active_take();
            assert!(take.source_path.is_some(), "每个片段必须有源引用");
            assert!(take.source_end_sec > take.source_start_sec);
            // 1:1 映射：clip 长度 == 源窗长度。
            assert!(
                ((take.source_end_sec - take.source_start_sec) - clip.length_sec).abs() < 1e-9,
                "Transfer 是 1:1 映射，源窗长度必须等于片段长度"
            );
        }
        assert!(timeline.project_sec > 0.0);
    }

    /// 没有任何 take（或全是静音）时给出一条**空的**时间线，而不是 `None`。
    #[test]
    fn an_empty_projection_still_yields_a_timeline() {
        let (timeline, summary) = project(&[], 120.0);
        assert!(timeline.clips.is_empty());
        assert_eq!(timeline.tracks.len(), 1, "界面要有一条空轨可以显示");
        assert_eq!(summary.clips, 0);
        let (silent, summary) = one(1, &[segment(1, 0, &[0.0; 32])]);
        assert!(silent.clips.is_empty());
        assert_eq!(summary.spanned_frames, 32, "跨过的工程帧数仍要如实报出");
    }
}
