//! 把捕获到的段投影成 [`TimelineState`]：解决覆盖关系、合并相邻片、生成片段。
//!
//! # 为什么这一层必须存在
//!
//! 音频线程只负责"把每一块写到它该在的工程位置上"（`capture.rs`）。段之间**可以
//! 重叠**：宿主 seek 回退、循环回卷、以及**再录一遍**都会让同一段工程时间被录到两次。
//! 合并是**视图**的事，不是音频线程的事 —— 那里既不能分配，也不该承担区间运算。
//!
//! # 覆盖规则：后来的胜出
//!
//! 段按 `(take, 带内偏移)` 排序后依次铺开；每铺一段，就把它覆盖到的已有区间**剪掉**。
//! 那个排序键就是**录制顺序**：take 号随每次录制递增，带内偏移在一趟连续录制里递增。
//! 于是"第 2 圈覆盖第 1 圈"与"补录的那一句覆盖它盖住的部分"都是同一个规则的自然结果，
//! 不需要为循环或补录写任何特例。
//!
//! # 素材带（[`Tape`]）—— 段是容器，带才是内容
//!
//! 段是**内存容器**（8 秒定长，见 `capture.rs`），而一次录制的内容是一条连续演奏。
//! 一个 clip 只能引用**一份源**与**一个连续源窗**，所以"跨段"的东西必须先有一个连续
//! 的源域 —— 那就是素材带：一个 take 的段按录制顺序首尾相接，段 k 的带内起点等于
//! 前面所有段的帧数之和。
//!
//! 这一层带来两件事，都是用户可见的：
//!
//! 1. **段边界对用户不存在**。40 秒录制在时间轴上是**一个**片段，不是五个。
//!    段容量是内存策略，不该漏到时间线上（它此前正是那样漏出去的：合并要求"同一个段"）。
//! 2. **片段的身份是 `(take, 带内偏移)`** —— "这是哪一次录制的哪一段音频"。它与
//!    "这片由几个段拼成""落在工程时间哪里"都无关，因此重投影不会让用户的编辑错位。
//!
//! # 静音不是洞
//!
//! 静音**留在片段里**（源窗含静音，波形上是一段平的）。它此前被逐片丢弃，于是
//! "这两秒我没出声"从时间轴上消失、后面的内容前移，工程时间与素材对不上。
//! 现在只有**真的没有数据**（宿主停播、位置跳跃）才会切断片段 —— 那时时间线上是
//! 一段**空**，而空是诚实的：那里确实什么都没录。
//!
//! 整段录音全静音时仍然**不产生片段**（素材为空的判据是"有没有可听内容"，
//! 见 [`MaterialSummary::is_silent`](super::MaterialSummary::is_silent)）：一个全静音的
//! take 若算作"有素材"，插件就会把这条轨替换成静音 —— 那是"插件把我的轨道弄哑了"。
//!
//! # 单位
//!
//! 段是**工程采样域**的（`projectTimeSamples` 的域）。秒 = 帧 / 采样率，三个方向
//! 各自独立换算（段起点、段长、段内偏移），不合到一起算 —— 合起来写会掩盖
//! "用了哪个率"这个唯一重要的问题。

use super::capture::Segment;
use crate::render::snapshot::PlaybackSnapshot;
use hifishifter_kernel::state::TimelineState;
use serde_json::json;
use std::collections::HashMap;
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

/// 投影的输入：一个 take 的 id 与它的段（按录制顺序）。
#[derive(Clone, Copy)]
pub(crate) struct ProjectionTake<'a> {
    pub id: u64,
    pub segments: &'a [Segment],
}

/// 一个 take 的**素材带**：它的段按录制顺序首尾相接（见模块头）。
pub(crate) struct Tape<'a> {
    /// 录制序号（take 号）。
    pub(crate) take: u64,
    /// 本带的采样率。同一 take 内必然一致 —— 捕获在采样率变化时停止。
    pub(crate) sample_rate: u32,
    /// 参与本带的段（**已剔除**含 NaN/Inf 的那些）。
    segments: Vec<&'a Segment>,
    /// 每段在带上的起点（帧），与 `segments` 同序。
    offsets: Vec<usize>,
    /// 带的总帧数。
    frames: usize,
}

impl<'a> Tape<'a> {
    /// 带的总帧数。
    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    /// 把带上 `[offset, offset + frames)` 这一段切成"每段一段"的切片。
    ///
    /// 产出 `(段, 段内起点, 段内终点)`，按带内顺序。落在带外的部分被裁掉 ——
    /// 调用方给的窗口永远应当落在带内，这里的裁剪是防御。
    fn slices(
        &self,
        offset: usize,
        frames: usize,
    ) -> impl Iterator<Item = (&'a Segment, usize, usize)> + '_ {
        let end = offset.saturating_add(frames);
        self.segments
            .iter()
            .zip(&self.offsets)
            .filter_map(move |(segment, base)| {
                let base = *base;
                let lo = offset.max(base);
                let hi = end.min(base + segment.filled);
                (lo < hi).then_some((*segment, lo - base, hi - base))
            })
    }

    /// 带上这一段是否**全部**为零（任一声道非零即不是）。
    ///
    /// 【为什么必须看全部声道】此前只看第一条平面 —— 于是"只有右声道有内容"的一段
    /// 会被判成静音而整片丢弃。那是最难排查的一类：用户明明录到了东西，时间轴却空着。
    fn is_silent(&self, offset: usize, frames: usize) -> bool {
        self.slices(offset, frames).all(|(segment, lo, hi)| {
            segment
                .planes
                .iter()
                .all(|plane| plane[lo..hi].iter().all(|sample| *sample == 0.0))
        })
    }

    /// 把整条带拼成可直接交给渲染的平面。
    ///
    /// 【为什么在这里拼而不是让消费方各拼一遍】源 PCM 是渲染与波形共用的输入，
    /// 而"带 = 段的顺序拼接"这条规则只能有一处实现 —— 两处实现必然漂移，
    /// 表现为"听到的和看到的不是同一段音频"。
    pub(crate) fn concat_planes(&self) -> Vec<Vec<f32>> {
        let channels = self
            .segments
            .iter()
            .map(|segment| segment.planes.len())
            .max()
            .unwrap_or(0);
        let mut planes = vec![vec![0.0_f32; self.frames]; channels];
        for (segment, base) in self.segments.iter().zip(&self.offsets) {
            for (channel, plane) in segment.planes.iter().enumerate() {
                planes[channel][*base..*base + segment.filled].copy_from_slice(plane);
            }
        }
        planes
    }
}

/// 由全部 take 建出素材带，并报出被剔除的段数。
///
/// 【为什么剔除段而不是整条 take】含 NaN/Inf 的段不能进渲染，但同一次录制里的其余段
/// 是好的 —— 让一段坏音频抹掉整次录制，与 `render/input.rs` 的既有纪律（逐条降级）不符。
pub(crate) fn build_tapes<'a>(takes: &[ProjectionTake<'a>]) -> (Vec<Tape<'a>>, usize) {
    let mut dropped = 0usize;
    let mut tapes = Vec::with_capacity(takes.len());
    for take in takes {
        let mut segments = Vec::with_capacity(take.segments.len());
        let mut offsets = Vec::with_capacity(take.segments.len());
        let mut frames = 0usize;
        let mut sample_rate = 0u32;
        for segment in take.segments {
            if segment.span().is_empty() || segment.has_nonfinite() {
                dropped += 1;
                continue;
            }
            offsets.push(frames);
            frames += segment.filled;
            sample_rate = sample_rate.max(segment.sample_rate);
            segments.push(segment);
        }
        if segments.is_empty() {
            continue;
        }
        tapes.push(Tape {
            take: take.id,
            sample_rate,
            segments,
            offsets,
            frames,
        });
    }
    (tapes, dropped)
}

/// 工程区间的帧数。
///
/// 【为什么写成一个函数而不是到处 `(end - start) as usize`】`Range` 没有 `len()`，
/// 而减法在 `i64` 上写错顺序（`start - end`）会得到一个巨大的无符号数。一处定义。
fn frames_of(span: &Range<i64>) -> usize {
    (span.end - span.start).max(0) as usize
}

/// 一段已解决覆盖关系的工程区间，指向某条素材带上的一个连续窗口。
#[derive(Debug, Clone)]
struct Piece {
    /// 工程采样区间（左闭右开）。
    span: Range<i64>,
    /// 所属素材带（`tapes` 的下标）。
    tape: usize,
    /// 该区间起点在带内的帧偏移。
    offset: usize,
}

/// 一次录制里的一个连续 pass（遇到 gap / 回卷 / seek 就结束）。
#[derive(Debug, Clone)]
struct Pass {
    /// 所属素材带。
    tape: usize,
    /// 所属 take（跨 take 的显式循环分组用）。
    take: u64,
    /// 段在 `Tape::segments` 中的下标（按录制顺序）。
    segments: Vec<usize>,
    /// 工程区间。
    span: Range<i64>,
    /// 录制顺序 `(take, pass 序号)`。
    order: (u64, usize),
    /// 循环起点（`Some` = 宿主明确给出了 cycle 元数据）。
    loop_start: Option<i64>,
}

/// 由素材带切出 pass。段是内存容器，pass 才是“一次连续演奏”。
fn build_passes(tapes: &[Tape<'_>]) -> Vec<Pass> {
    let mut passes = Vec::new();
    for (tape_index, tape) in tapes.iter().enumerate() {
        let mut start = 0usize;
        let mut pass_index = 0usize;
        let mut previous_end: Option<i64> = None;
        let mut previous_loop: Option<i64> = None;
        for (index, segment) in tape.segments.iter().enumerate() {
            let span = segment.span();
            let begins_new = match previous_end {
                None => true,
                Some(end) => span.start != end || segment.loop_start != previous_loop,
            };
            if begins_new && index > start {
                passes.push(make_pass(tape_index, tape, start, index, pass_index));
                pass_index += 1;
                start = index;
            }
            previous_end = Some(span.end);
            previous_loop = segment.loop_start;
        }
        if start < tape.segments.len() {
            passes.push(make_pass(
                tape_index,
                tape,
                start,
                tape.segments.len(),
                pass_index,
            ));
        }
    }
    passes
}

fn make_pass(
    tape_index: usize,
    tape: &Tape<'_>,
    start: usize,
    end: usize,
    pass_index: usize,
) -> Pass {
    let first = tape.segments[start];
    let last = tape.segments[end - 1];
    Pass {
        tape: tape_index,
        take: tape.take,
        segments: (start..end).collect(),
        span: first.span().start..last.span().end,
        order: (tape.take, pass_index),
        loop_start: first.loop_start,
    }
}

/// 把“同一循环起点上的多圈”收敛成一个原子组：最新一圈替换整组。
///
/// 【为什么必须原子】最后一圈半途停止时，如果只按“后来的胜出”覆盖已到达的部分，
/// 上一圈的尾段会留下来，时间线上就是两个相邻 clip。用户要的是“我刚录的这一遍”，
/// 不是“这一遍 + 上一遍的尾巴”。
fn atomic_loop_groups(passes: Vec<Pass>) -> Vec<Pass> {
    let mut keep = vec![true; passes.len()];

    // 1) 显式分组：宿主给出了 cycle 起点。允许跨 take —— 重录同一循环区间时，
    //    新 take 应原子替换旧组。
    let mut explicit: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, pass) in passes.iter().enumerate() {
        if let Some(loop_start) = pass.loop_start {
            explicit.entry(loop_start).or_default().push(index);
        }
    }
    for group in explicit.values() {
        supersede_group(&passes, group, &mut keep);
    }

    // 2) 隐式分组：没有 cycle 元数据时，同一 take 内两个以上 pass 从同一个工程
    //    位置开始，就当作循环回卷。只跨 pass、不跨 take（避免把普通同点补录误判）。
    let mut implicit: HashMap<(u64, i64), Vec<usize>> = HashMap::new();
    for (index, pass) in passes.iter().enumerate() {
        if pass.loop_start.is_none() {
            implicit
                .entry((pass.take, pass.span.start))
                .or_default()
                .push(index);
        }
    }
    for group in implicit.values() {
        supersede_group(&passes, group, &mut keep);
    }

    passes
        .into_iter()
        .enumerate()
        .filter_map(|(index, pass)| keep[index].then_some(pass))
        .collect()
}

fn supersede_group(passes: &[Pass], group: &[usize], keep: &mut [bool]) {
    if group.len() < 2 {
        return;
    }
    let Some(latest) = group
        .iter()
        .copied()
        .max_by_key(|&index| passes[index].order)
    else {
        return;
    };
    let group_start = group
        .iter()
        .map(|&index| passes[index].span.start)
        .min()
        .unwrap_or(0);
    let group_end = group
        .iter()
        .map(|&index| passes[index].span.end)
        .max()
        .unwrap_or(0);
    for &index in group {
        if index == latest {
            continue;
        }
        let pass = &passes[index];
        // 只移除完全落在循环组内的旧 pass；跨越组边界的 pass 可能还包含组外内容，
        // 交给 later-wins 逐段处理，避免误删。
        if pass.span.start >= group_start && pass.span.end <= group_end {
            keep[index] = false;
        }
    }
}

fn pieces_from_passes(passes: &[Pass], tapes: &[Tape<'_>]) -> Vec<Piece> {
    let mut seeds = Vec::new();
    for pass in passes {
        let tape = &tapes[pass.tape];
        for &index in &pass.segments {
            let segment = tape.segments[index];
            if let Some(piece) = clamp_piece(Piece {
                span: segment.span(),
                tape: pass.tape,
                offset: tape.offsets[index],
            }) {
                seeds.push(piece);
            }
        }
    }
    seeds
}

/// 投影与原始磁带共用的“最终片”入口。
fn resolved_pieces(tapes: &[Tape<'_>]) -> Vec<Piece> {
    let passes = atomic_loop_groups(build_passes(tapes));
    coalesce(resolve(pieces_from_passes(&passes, tapes)))
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
    let (tapes, dropped_segments) = build_tapes(takes);
    let mut summary = TimelineSummary {
        dropped_segments,
        ..TimelineSummary::default()
    };

    let mut timeline = TimelineState {
        bpm: if bpm.is_finite() && bpm > 0.0 {
            bpm
        } else {
            120.0
        },
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
            // 【为什么这条轨道默认开启合成】`compose_enabled` 是"要不要处理这条轨"的
            // 总开关：关着的时候 `hifigan_effect_flags` 一律返回全假、效果曲线被
            // `gate_hifigan_effect_curves` 整条剥掉（设计如此：听到未经改动的原音频）。
            // 而这条轨道**只为被处理而存在** —— 它是插件自己造出来的编辑轨，宿主那边
            // 没有任何"原声"语义需要保留。默认关着会让"画完参数线不生效"成为默认行为，
            // 而界面上只有一个不说原因的空结果。
            //
            // 【为什么默认开启不改变"还没画线时"的声音】`does_clip_need_processor_render`
            // 在没有曲线、没有效果、没有速率变化时返回假 —— 声码器根本不会跑。所以
            // 开启只让"画了线"生效，不让"没画线"变成另一种声音。
            "compose_enabled": true,
        }))
        .expect("Transfer 轨道的必填字段是固定的"),
    );

    let pieces = resolved_pieces(&tapes);

    let mut index = 0_usize;
    for piece in pieces {
        let tape = &tapes[piece.tape];
        let rate = tape.sample_rate.max(1) as f64;
        let frames = frames_of(&piece.span);
        summary.spanned_frames += frames as u64;
        // 【为什么逐片判静音而不是逐段】一个段可以是"前半段有声、后半段静音"
        // （用户在某处停止演奏）。按段判定会把有声的部分一起丢掉。
        //
        // 【为什么判据是"整片"而不是"逐块"】静音**留在片段里**（见模块头）：只有整片
        // 都没有可听内容时才不产生片段，而那种片只可能出现在"两次录制之间"或
        // "整次录制都是静音"这两种场合 —— 前者那里本来就该是空的。
        if tape.is_silent(piece.offset, frames) {
            continue;
        }
        summary.covered_frames += frames as u64;
        index += 1;
        let source = source_id(tape.take);
        let start_sec = piece.span.start as f64 / rate;
        let length_sec = frames as f64 / rate;
        let source_start_sec = piece.offset as f64 / rate;
        let source_end_sec = (piece.offset + frames) as f64 / rate;
        let id = clip_id(tape.take, piece.offset);
        let name = format!("#{index} @ {start_sec:.3}s");
        let mut clip: hifishifter_kernel::state::Clip = serde_json::from_value(json!({
            "id": id,
            "track_id": TRANSFER_TRACK_ID,
            "name": name,
            "start_sec": start_sec,
            "length_sec": length_sec,
        }))
        .expect("片段的必填字段是固定的");
        // 【为什么 `duration_*` 是整条带的长度】它是"这份媒体有多长"，而不是"这一片有多
        // 长"。片段自己的长度在 `length_sec` 里，两者此前混用同一个量，于是源窗一旦
        // 不等于整段（被切过之后就是如此）就自相矛盾。
        let take: hifishifter_kernel::state::ClipTake = serde_json::from_value(json!({
            "id": format!("{id}_take_1"),
            "name": name,
            "source_path": source,
            "source_start_sec": source_start_sec,
            "source_end_sec": source_end_sec,
            "source_sample_rate": tape.sample_rate,
            "duration_sec": tape.frames() as f64 / rate,
            "duration_frames": tape.frames() as u64,
            "playback_rate": 1.0,
        }))
        .expect("take 的必填字段是固定的");
        clip.takes.push(take);
        clip.active_take_id = Some(format!("{id}_take_1"));
        clip.normalize_takes();
        timeline.clips.push(clip);
    }
    timeline
        .clips
        .sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
    summary.clips = timeline.clips.len();
    timeline.project_sec = timeline
        .clips
        .iter()
        .map(|clip| clip.start_sec + clip.length_sec)
        .fold(0.0_f64, f64::max);
    (timeline, summary)
}

/// 素材的**未处理** PCM：把投影出的片段按工程位置原样铺成一份快照（"原始磁带"）。
///
/// 【为什么必须直接拷样本，而不是复用 `mix_plain_regions`】那个函数服务的是 **ARA** 通路，
/// 它的 `start_in_modification_time` 遵循**宿主**的单位约定（见该函数的说明与
/// `the_plain_path_converts_the_project_rate_before_indexing_the_source`）。
/// Transfer 的区间用的是**我们自己的**单位（源速率秒），两者并不相同 —— 实测按它读时
/// 会把"秒"当成"样本下标"，铺出来是一段阶梯状的垃圾：**听感上就是噪音**。
///
/// 原始磁带要表达的东西极其简单："把捕获到的样本原样播一遍"。那就**直接拷样本**：
/// 没有单位换算、没有重采样、没有插值，因而不可能因为约定不一致而出错。
pub(crate) fn raw_tape(takes: &[ProjectionTake<'_>]) -> Option<PlaybackSnapshot> {
    let (tapes, _) = build_tapes(takes);
    let pieces = resolved_pieces(&tapes);
    if pieces.is_empty() {
        return None;
    }
    // 【为什么取最大采样率当输出域】捕获在采样率变化时会停止本次录制，因此同一批素材
    // 的率必然一致；取最大只是兜底，不改变语义。
    let rate = tapes
        .iter()
        .map(|tape| tape.sample_rate)
        .max()
        .filter(|rate| *rate > 0)?;
    let start = pieces.iter().map(|piece| piece.span.start).min()?;
    let end = pieces.iter().map(|piece| piece.span.end).max()?;
    let frames = usize::try_from(end.checked_sub(start)?).ok()?;
    if frames == 0 {
        return None;
    }
    let bytes = frames.checked_mul(8)?;
    let reservation = crate::render::budget::global_budget().reserve(bytes)?;
    let mut left = vec![0.0_f32; frames];
    let mut right = vec![0.0_f32; frames];
    for piece in pieces {
        let tape = &tapes[piece.tape];
        // 逐片按**它自己的**采样率换算到输出域（同率时是恒等，异率时按比例定位）。
        let scale = tape.sample_rate.max(1) as f64 / rate as f64;
        let base = ((piece.span.start - start) as f64 * scale).round() as i64;
        let count = frames_of(&piece.span);
        let mut written = 0usize;
        for (segment, lo, hi) in tape.slices(piece.offset, count) {
            let Some(source_left) = segment.planes.first() else {
                written += hi - lo;
                continue;
            };
            for local in lo..hi {
                let target = base + (written as f64 * scale).round() as i64;
                written += 1;
                if target < 0 || target as usize >= frames {
                    continue;
                }
                let target = target as usize;
                left[target] = source_left[local];
                right[target] = match segment.planes.get(1) {
                    // 单声道捕获：两个输出平面同内容（与 `pass_through` 的展开一致）。
                    Some(plane) if plane.len() > local => plane[local],
                    _ => source_left[local],
                };
            }
        }
    }
    let mut snapshot = PlaybackSnapshot {
        sample_rate: rate,
        origin_sample: start,
        left,
        right,
        _reservation: Some(reservation),
    };
    // 与合成路径同一纪律：逐 bit 相同才归并，且归并会把额度退回去。
    let _ = snapshot.compact_mono();
    Some(snapshot)
}

/// 单个 take 的投影（便利入口；语义与 [`project`] 逐字段一致）。
pub(crate) fn build(
    segments: &[Segment],
    take_id: u64,
    bpm: f64,
) -> (TimelineState, TimelineSummary) {
    project(
        &[ProjectionTake {
            id: take_id,
            segments,
        }],
        bpm,
    )
}

/// 素材 id：**一个 take 一份源**。
///
/// 【为什么不是一段一份】片段可以跨段（见模块头），而一个片段只能引用一份源 ——
/// 段一份源会让跨段片段无法表达。带上 take 号是必须的：两次录制的内容不同，
/// 共用一个 id 会让"上一次 take 的源"被这一次引用。
pub(crate) fn source_id(take_id: u64) -> String {
    format!("transfer://{take_id}")
}

/// 片段 id。**稳定**：同一个 take 的同一段带内音频永远是同一个 id。
///
/// 【为什么键是 `(take, 带内偏移)` 而不是投影序号或段号】补录一次会让后面的片整体
/// 后移，序号随之全变 —— 用户在那个片段上的选择与编辑会在重新投影后错位到别的片段上。
/// 段号则会随**内存容器**的划分变化（8 秒一段），而容器与内容无关。
/// `(take, 带内偏移)` 只描述"这是哪一次录制的哪一段音频"：没被覆盖的片保持 id 不变，
/// 被覆盖的片本来就该换一个身份。
pub(crate) fn clip_id(take_id: u64, offset: usize) -> String {
    format!("transfer-clip-{take_id}-{offset}")
}

/// 把一条**已经组装好的** Transfer 时间线里的负起点 clip 夹到工程 0。
///
/// 【为什么投影之外还要这一层】旧工程/旧会话可能在本次修复之前就投影出了负起点
/// clip。渲染器拿到负起点时只会把窗口起点夹到 0，**不会**平移源窗口；这里在
/// Transfer 的渲染输入上做同样的裁剪（窗口 + 源偏移 + 淡入/吸附偏移同时移动），
/// 于是即使投影这一层被绕过，raw 与 processed 也不会相差一个 pre-roll。
///
/// 返回被裁剪的 clip 数（调用方据此记一次限流日志）。
pub(crate) fn clamp_timeline_to_project(timeline: &mut TimelineState) -> usize {
    let mut trimmed = 0_usize;
    timeline.clips.retain_mut(|clip| {
        if !clip.start_sec.is_finite() || clip.start_sec >= 0.0 {
            return true;
        }
        let pre = -clip.start_sec;
        if !pre.is_finite() || !clip.length_sec.is_finite() || clip.length_sec <= pre {
            return false;
        }
        clip.start_sec = 0.0;
        clip.length_sec -= pre;
        let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 0.0 {
            clip.playback_rate as f64
        } else {
            1.0
        };
        let source_shift = pre * rate;
        if clip.reversed {
            clip.source_end_sec -= source_shift;
        } else {
            clip.source_start_sec += source_shift;
        }
        // 淡入与吸附偏移都相对 clip 起点；起点前移了 pre，它们也前移同样的量。
        clip.fade_in_sec = (clip.fade_in_sec - pre).max(0.0);
        clip.snap_offset_sec = (clip.snap_offset_sec - pre).max(0.0);
        trimmed += 1;
        true
    });
    if trimmed > 0 {
        timeline.project_sec = timeline
            .clips
            .iter()
            .map(|clip| clip.start_sec + clip.length_sec)
            .fold(0.0_f64, f64::max);
    }
    trimmed
}

/// 把负的工程区间夹到素材域 `[0, ∞)`。
///
/// 【为什么投影也要做一次】捕获侧已经不再写负起点（pre-roll 在进入捕获时夹掉），
/// 但旧工程从 sidecar 恢复的 take 仍可能带负 origin。渲染侧把窗口起点夹到 0 时
/// **不会**同步平移源窗口 —— 于是内容会整体前移一个 pre-roll。这里在**唯一的**
/// 投影入口把区间和源偏移一起裁剪，`project` 与 `raw_tape` 因此看到同一份域。
fn clamp_piece(piece: Piece) -> Option<Piece> {
    if piece.span.start >= 0 {
        return Some(piece);
    }
    let frames = frames_of(&piece.span);
    if frames == 0 {
        return None;
    }
    let skip = piece
        .span
        .start
        .saturating_neg()
        .min(i64::try_from(frames).unwrap_or(i64::MAX)) as usize;
    if skip >= frames {
        return None;
    }
    Some(Piece {
        span: 0..piece.span.end,
        offset: piece.offset.saturating_add(skip),
        ..piece
    })
}

/// 按"后来的胜出"把片铺成互不相交的区间。
fn resolve(mut seeds: Vec<Piece>) -> Vec<Piece> {
    // 【排序键是录制顺序 `(take, 带内偏移)` 而不是工程位置】"后来者胜出"里的"后来"
    // 指的是**录制顺序**，而 take 号与带内偏移正是录制顺序（带是段按录制顺序拼起来的）。
    // 按位置排序会把规则弄反：位置靠前但录得晚的片会先被铺下，再被位置靠后但录得早的片
    // 覆盖 —— 于是循环录制的**最后一圈反而输给第一圈**，补录的一句也会输给它替换掉的那一句。
    seeds.sort_by_key(|piece| (piece.tape, piece.offset));
    let mut pieces: Vec<Piece> = Vec::new();
    for seed in seeds {
        let span = seed.span.clone();
        let mut next: Vec<Piece> = Vec::with_capacity(pieces.len() + 1);
        for piece in pieces.drain(..) {
            if piece.span.end <= span.start || piece.span.start >= span.end {
                next.push(piece);
                continue;
            }
            // 左残段：偏移不变（它离带起点还是那么远）。
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

/// 合并工程方向与**带内方向**都首尾相接的片：它们是同一段连续音频。
///
/// 【为什么不再要求"同一个段"】那是把**内存容器**的边界当成了内容边界 —— 一次 40 秒
/// 录制（5 个段）在时间轴上会变成 5 个片段，而"为什么被切成 5 段"对用户毫无意义。
/// 带（[`Tape`]）已经让跨段的偏移连续，因此那一条限制自然消失了。
///
/// 【为什么要求**两个**方向都连续】只要求工程方向连续会把"工程上相邻但带上跳了"
/// 的两段拼成一个 clip，而 clip 的源窗是**连续**区间 —— 那样源窗会跨过一段不存在的
/// 素材，渲染出错误的内容。
///
/// 【为什么还要求同一个 take】不同 take 的段是两次不同的录制，即使工程上首尾相接、
/// 带内偏移也连续，把它们拼成一个 clip 会让"这是哪一次录的"消失 —— 而 take 号是
/// 曲线上与清单里的身份。
fn coalesce(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    for piece in pieces {
        match out.last_mut() {
            Some(last)
                if last.tape == piece.tape
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::capture::{ClosedReason, SegmentLoss};

    fn segment(origin: i64, values: &[f32]) -> Segment {
        Segment {
            origin,
            sample_rate: 1000,
            planes: vec![values.to_vec(), values.to_vec()],
            filled: values.len(),
            loss: SegmentLoss::First,
            loop_start: None,
            peak: [0.0; 2],
            closed: ClosedReason::Stopped,
            _reservation: None,
        }
    }

    /// 带循环起点的段（模拟 capture 在回卷时写下的 `loop_start`）。
    fn segment_loop(origin: i64, values: &[f32], loop_start: i64) -> Segment {
        Segment {
            loop_start: Some(loop_start),
            ..segment(origin, values)
        }
    }

    /// 单 take 的便利投影。
    fn one(take: u64, segments: &[Segment]) -> (TimelineState, TimelineSummary) {
        build(segments, take, 120.0)
    }

    /// 不相交的两段各生成一个片段，且**位置就是工程位置**。
    #[test]
    fn disjoint_segments_become_clips_at_their_project_positions() {
        let segments = vec![segment(0, &[0.1; 10]), segment(100, &[0.2; 10])];
        let (timeline, summary) = one(7, &segments);
        assert_eq!(summary.clips, 2);
        assert_eq!(timeline.clips.len(), 2);
        // 1000 Hz 下 100 帧 = 0.1 秒。
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!((timeline.clips[0].length_sec - 0.01).abs() < 1e-9);
        assert!((timeline.clips[1].start_sec - 0.1).abs() < 1e-9);
    }

    /// **重叠时后来的胜出**，且被覆盖的部分从胜出者的**带内**偏移开始。
    #[test]
    fn a_later_segment_overwrites_an_earlier_overlap() {
        // 段 1 覆盖 [0,20)，段 2 覆盖 [10,30)。
        let segments = vec![segment(0, &[1.0; 20]), segment(10, &[2.0; 20])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 2, "应切成 [0,10) 与 [10,30) 两片");
        // 第一片：段 1 的 [0,10)。
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!((timeline.clips[0].length_sec - 0.01).abs() < 1e-9);
        assert!((timeline.clips[0].source_start_sec - 0.0).abs() < 1e-9);
        // 第二片：段 2 的 [10,30)。它的带内起点是 20（段 1 有 20 帧），因此源窗从
        // **带上**的 20 开始 —— 不是从 0。这是"一个 take 一份源"的直接后果：源域是
        // 整条带，而段 2 落在带的后半。
        assert!((timeline.clips[1].start_sec - 0.01).abs() < 1e-9);
        assert!((timeline.clips[1].length_sec - 0.02).abs() < 1e-9);
        assert!((timeline.clips[1].source_start_sec - 0.02).abs() < 1e-9);
        assert!(timeline.clips[1]
            .takes
            .first()
            .is_some_and(|take| take.source_path.as_deref() == Some("transfer://1")));
    }

    /// 后来的段完全盖住早先的段 ⇒ 早先那段整个消失。
    #[test]
    fn a_later_segment_can_swallow_an_earlier_one_whole() {
        let segments = vec![segment(10, &[1.0; 10]), segment(0, &[2.0; 40])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].length_sec - 0.04).abs() < 1e-9);
        assert!(timeline.clips[0]
            .takes
            .first()
            .is_some_and(|take| take.source_path.as_deref() == Some("transfer://1")));
    }

    /// **补录一句只替换它盖住的那一段**（跨 take 的 later-wins）。
    ///
    /// 【为什么这是本模块最重要的契约】第二次录制此前会**整条换掉**时间轴（活动 take
    /// 单选）：用户补录一句，之前录的全不见了。这里钉住的是"补录的那一句胜出、其余原样
    /// 保留"，并且剩下的片仍然指向**原来那一次**录制的素材。
    #[test]
    fn a_second_take_replaces_only_the_range_it_covers() {
        let first = vec![segment(0, &[1.0; 100])];
        let second = vec![segment(40, &[2.0; 20])];
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
            vec!["transfer://1", "transfer://2", "transfer://1"],
            "补录只替换它盖住的那一段，两侧仍是第一次录制"
        );
        // 尾片的源偏移必须跟着被切掉的长度走（否则源窗会跨过不存在的素材）。
        let tail = timeline.clips[2].active_take();
        assert!((tail.source_start_sec - 0.06).abs() < 1e-9);
    }

    /// **一次连续录制横跨多个段 ⇒ 时间轴上是**一个**片段。
    ///
    /// 【为什么这是本轮最重要的回归测试】段是内存容器（8 秒定长），而"一次演奏"是内容。
    /// 此前合并要求"同一个段"，于是一次 40 秒录制（5 段）在时间轴上是 5 个片段 ——
    /// 用户看到一条被切碎的时间轴，而切分原因对他毫无意义。带（[`Tape`]）让跨段的
    /// 偏移连续之后，这条限制才真正消失。
    #[test]
    fn one_recording_becomes_one_clip_across_many_segments() {
        // 五个首尾相接的段，工程上连续、带内也连续 —— 就是"段满自动切段"的形状。
        let segments = vec![
            segment(0, &[0.1; 8]),
            segment(8, &[0.1; 8]),
            segment(16, &[0.1; 8]),
            segment(24, &[0.1; 8]),
            segment(32, &[0.1; 8]),
        ];
        let (timeline, summary) = one(3, &segments);
        assert_eq!(
            summary.clips, 1,
            "跨段的连续音频必须合成一个片段，而不是一段一个"
        );
        assert_eq!(timeline.clips.len(), 1);
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!(
            (timeline.clips[0].length_sec - 0.04).abs() < 1e-9,
            "片段必须覆盖全部 40 帧，实际 {}",
            timeline.clips[0].length_sec
        );
        // 源窗必须**逐字**覆盖整条带：这正是"跨段也能是一个连续源窗"的证据。
        let take = timeline.clips[0].active_take();
        assert!((take.source_start_sec - 0.0).abs() < 1e-9);
        assert!((take.source_end_sec - 0.04).abs() < 1e-9);
        // 整条带一份源。
        assert_eq!(take.source_path.as_deref(), Some("transfer://3"));
    }

    /// **静音留在片段里**：起唱前的静音不再把乐句切成两半。
    ///
    /// 【为什么这条必须钉住】静音此前被逐片丢弃，于是"我在这两秒停了手"从时间轴上
    /// 消失、后面的内容前移 —— 用户对着自己工程里的位置找不到刚录的东西。
    #[test]
    fn silence_inside_a_continuous_recording_stays_in_the_clip() {
        // 一段全零（用户还没开口）紧接一段有声：两者工程连续、带内也连续。
        let segments = vec![segment(0, &[0.0; 10]), segment(10, &[0.5; 10])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1, "静音不该把乐句切成两半");
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!(
            (timeline.clips[0].length_sec - 0.02).abs() < 1e-9,
            "片段必须从录制起点开始，含前面那段静音"
        );
        assert_eq!(summary.covered_frames, 20);
    }

    /// **只有右声道有内容时不算静音**。
    ///
    /// 【为什么这条必须钉住】静音判据此前只看第一条平面 —— 于是一个"左声道没接上"
    /// 的宿主会让用户录到的内容整片消失，而时间轴上空着、没有任何解释。
    #[test]
    fn content_on_the_second_channel_is_not_silence() {
        let mut segment = segment(0, &[0.0; 10]);
        segment.planes = vec![vec![0.0; 10], vec![0.4; 10]];
        let (timeline, summary) = one(1, &[segment]);
        assert_eq!(summary.clips, 1, "右声道有内容就必须产生片段");
        assert_eq!(timeline.clips.len(), 1);
    }

    /// 同一个段里连续的片合并成一个片段（不是一片一个）。
    #[test]
    fn contiguous_pieces_from_one_segment_merge_into_a_single_clip() {
        let segments = vec![segment(0, &[0.3; 30])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].length_sec - 0.03).abs() < 1e-9);
    }

    /// 含非有限采样的段被丢弃，其余照常 —— 一段坏音频不该让整条时间线消失。
    #[test]
    fn a_segment_with_nonfinite_samples_is_dropped_without_losing_the_rest() {
        let mut bad = segment(0, &[0.1; 10]);
        bad.planes[0][3] = f32::NAN;
        let segments = vec![bad, segment(20, &[0.2; 10])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.dropped_segments, 1);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].start_sec - 0.02).abs() < 1e-9);
    }

    /// 片段 id 稳定：没被覆盖的片在补录之后仍然是同一个 id（否则用户的编辑会错位）。
    #[test]
    fn clip_ids_are_stable_across_reprojection() {
        let segments = vec![segment(0, &[0.1; 10]), segment(100, &[0.2; 10])];
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
        // 键是 `(take, 带内偏移)`：段 2 落在带上的第 10 帧。
        assert_eq!(
            ids(&first),
            vec!["transfer-clip-42-0", "transfer-clip-42-10"]
        );

        // 再录一遍（盖住后半段）：前一片的 id 不变，尾片的偏移变了 ⇒ 换 id。
        let overlay = vec![segment(100, &[0.3; 10])];
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
        assert_eq!(ids(&merged)[0], "transfer-clip-42-0", "未被覆盖的片保持 id");
    }

    /// 投影出的时间线必须能被内核的渲染入口接受（源窗与几何自洽）。
    #[test]
    fn the_projection_is_renderable() {
        let segments = vec![segment(0, &[0.1; 10]), segment(100, &[0.2; 10])];
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

    /// **原始磁带必须是捕获样本的逐样本拷贝。**
    ///
    /// 【为什么这是"录出来是噪音"的回归测试】此前原始磁带用 ARA 通路的
    /// `mix_plain_regions` 铺片，而那个函数的 `start_in_modification_time` 遵循**宿主**的
    /// 单位约定（工程速率秒）。Transfer 给的是**源速率秒**，于是"秒"被当成"样本下标"读：
    /// 一秒之内反复读同一个样本，铺出来是阶梯状的垃圾 —— 听感上就是噪音。
    #[test]
    fn the_raw_tape_is_a_verbatim_copy_of_the_captured_samples() {
        let mut segment = segment(4800, &[0.0; 8]);
        segment.planes = vec![
            (0..8).map(|index| index as f32).collect(),
            (0..8).map(|index| -(index as f32)).collect(),
        ];
        let snapshot = raw_tape(&[ProjectionTake {
            id: 1,
            segments: &[segment],
        }])
        .expect("有段就有原始磁带");
        assert_eq!(snapshot.sample_rate, 1000);
        assert_eq!(snapshot.origin_sample, 4800, "起点必须是工程位置，不是 0");
        assert_eq!(
            snapshot.left,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "原始磁带必须逐样本等于捕获到的内容"
        );
        assert_eq!(
            snapshot.right,
            vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
        );
    }

    /// **原始磁带与时间轴必须给出同一份覆盖关系**（否则"听到的"与"看到的"分叉）。
    #[test]
    fn the_raw_tape_follows_the_same_overlap_rules_as_the_timeline() {
        let first = vec![segment(0, &[1.0; 100])];
        let second = vec![segment(40, &[2.0; 20])];
        let takes = [
            ProjectionTake {
                id: 1,
                segments: &first,
            },
            ProjectionTake {
                id: 2,
                segments: &second,
            },
        ];
        let snapshot = raw_tape(&takes).expect("有段就有原始磁带");
        let (timeline, _) = project(&takes, 120.0);
        assert_eq!(snapshot.left.len(), 100);
        assert_eq!(snapshot.left[..40], [1.0; 40], "补录之前是第一次录制");
        assert_eq!(snapshot.left[40..60], [2.0; 20], "补录覆盖它盖住的那一段");
        assert_eq!(snapshot.left[60..], [1.0; 40], "补录之后回到第一次录制");
        assert_eq!(timeline.clips.len(), 3, "时间轴与它说的是同一件事");
    }

    /// 没有任何 take（或全是静音）时给出一条**空的**时间线，而不是 `None`。
    #[test]
    fn an_empty_projection_still_yields_a_timeline() {
        let (timeline, summary) = project(&[], 120.0);
        assert!(timeline.clips.is_empty());
        assert_eq!(timeline.tracks.len(), 1, "界面要有一条空轨可以显示");
        assert_eq!(summary.clips, 0);
        let (silent, summary) = one(1, &[segment(0, &[0.0; 32])]);
        assert!(
            silent.clips.is_empty(),
            "全静音不产生片段（素材为空的判据）"
        );
        assert_eq!(summary.spanned_frames, 32, "跨过的工程帧数仍要如实报出");
        assert_eq!(summary.covered_frames, 0, "没有任何可听内容");
    }

    /// **Transfer 轨道默认开启合成** —— 否则"画完参数线不生效"是默认行为。
    ///
    /// 【为什么这条必须钉住】`compose_enabled` 是"要不要处理这条轨"的总开关。这条轨道
    /// 只为被处理而存在（它是插件自己造的编辑轨，宿主那边没有"原声"语义要保留），
    /// 所以默认必须是开。默认关着的时候，用户画完线拿到的是一份**静默未处理**的结果。
    #[test]
    fn the_transfer_track_processes_by_default() {
        let (timeline, _) = one(1, &[segment(0, &[0.5; 10])]);
        let track = &timeline.tracks[0];
        assert_eq!(track.id, TRANSFER_TRACK_ID);
        assert!(
            track.compose_enabled,
            "Transfer 轨道默认必须开启合成，否则画完参数线不会有任何变化"
        );
    }
    /// **负 origin 的段在投影时被夹到工程 0，源窗口同步前移。**
    ///
    /// 【为什么必须同时移动源窗口】渲染侧把窗口起点夹到 0 时不会平移源窗口；
    /// 只夹 `span.start` 会让 clip 从源的第 0 帧开始播放 —— 整段内容提前一个 pre-roll。
    #[test]
    fn a_negative_origin_segment_is_clamped_and_keeps_its_source_alignment() {
        let values: Vec<f32> = (0..10).map(|index| index as f32).collect();
        let segments = vec![segment(-5, &values)];
        let (timeline, summary) = one(11, &segments);
        assert_eq!(summary.clips, 1);
        let clip = &timeline.clips[0];
        assert_eq!(clip.start_sec, 0.0, "素材域从工程 0 开始");
        assert_eq!(clip.length_sec, 5.0 / 1000.0, "只保留工程 0 之后的 5 帧");
        assert_eq!(
            clip.source_start_sec,
            5.0 / 1000.0,
            "源窗口必须跳过与 pre-roll 等长的帧数"
        );

        // raw_tape 与投影看到同一个域：工程 0 对应源第 5 帧。
        let raw = raw_tape(&[ProjectionTake {
            id: 11,
            segments: &segments,
        }])
        .expect("非空素材必须生成原始带");
        assert_eq!(raw.origin_sample, 0);
        assert_eq!(raw.left[0], 5.0);
        assert_eq!(raw.left.len(), 5);
    }

    /// **整段都在 0 之前：不产生 clip，也不产生原始带。**
    #[test]
    fn a_fully_negative_segment_projects_to_nothing() {
        let values = [1.0_f32; 4];
        let segments = vec![segment(-4, &values)];
        let (timeline, summary) = one(12, &segments);
        assert!(timeline.clips.is_empty());
        assert_eq!(summary.clips, 0);
        assert!(raw_tape(&[ProjectionTake {
            id: 12,
            segments: &segments,
        }])
        .is_none());
    }

    /// **已在内存里的负起点 clip：渲染输入裁剪会同时移动窗口与源偏移。**
    #[test]
    fn clamp_timeline_to_project_moves_the_window_and_the_source() {
        let mut timeline = TimelineState::default();
        let mut clip: hifishifter_kernel::state::Clip = serde_json::from_value(json!({
            "id": "clip",
            "track_id": "track",
            "name": "clip",
            "start_sec": -0.5,
            "length_sec": 2.0,
            "source_start_sec": 1.0,
            "source_end_sec": 3.0,
            "playback_rate": 1.0,
        }))
        .unwrap();
        clip.fade_in_sec = 0.75;
        clip.snap_offset_sec = 0.25;
        timeline.clips.push(clip);
        assert_eq!(clamp_timeline_to_project(&mut timeline), 1);
        let clip = &timeline.clips[0];
        assert_eq!(clip.start_sec, 0.0);
        assert_eq!(clip.length_sec, 1.5);
        assert_eq!(clip.source_start_sec, 1.5);
        assert_eq!(clip.source_end_sec, 3.0);
        assert_eq!(clip.fade_in_sec, 0.25);
        assert_eq!(clip.snap_offset_sec, 0.0);

        // 倒放：被裁掉的是源窗口的**尾端**。
        let mut reversed: hifishifter_kernel::state::Clip = serde_json::from_value(json!({
            "id": "reversed",
            "track_id": "track",
            "name": "reversed",
            "start_sec": -0.5,
            "length_sec": 2.0,
            "source_start_sec": 1.0,
            "source_end_sec": 3.0,
            "playback_rate": 1.0,
            "reversed": true,
        }))
        .unwrap();
        reversed.normalize_takes();
        let mut timeline = TimelineState::default();
        timeline.clips.push(reversed);
        assert_eq!(clamp_timeline_to_project(&mut timeline), 1);
        assert_eq!(timeline.clips[0].source_end_sec, 2.5);
    }

    /// **最后一圈半途停止：整个循环组原子替换，不留下上一圈的尾巴。**
    #[test]
    fn a_partial_last_loop_pass_replaces_the_whole_group() {
        let segments = vec![
            segment_loop(0, &[1.0; 100], 0),
            segment_loop(0, &[2.0; 40], 0),
        ];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1, "同一循环起点只能有一个可见 clip");
        assert_eq!(timeline.clips.len(), 1);
        assert!((timeline.clips[0].start_sec - 0.0).abs() < 1e-9);
        assert!(
            (timeline.clips[0].length_sec - 0.04).abs() < 1e-9,
            "可见范围到最后一圈实际走到的位置为止"
        );
    }

    /// **两圈完整循环：只留下最后一圈，仍然是一个 clip。**
    #[test]
    fn two_complete_loop_passes_leave_one_clip() {
        let segments = vec![
            segment_loop(0, &[1.0; 100], 0),
            segment_loop(0, &[2.0; 100], 0),
        ];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].length_sec - 0.1).abs() < 1e-9);
        let take = timeline.clips[0].active_take();
        assert!(
            (take.source_start_sec - 0.1).abs() < 1e-9,
            "内容必须来自第二圈（带内偏移 100 帧），而不是第一圈"
        );
    }

    /// **没有 cycle 元数据时，同一 take 内重复起点仍按循环组处理。**
    #[test]
    fn an_implicit_loop_group_requires_two_passes_with_the_same_start() {
        let segments = vec![segment(0, &[1.0; 100]), segment(0, &[2.0; 40])];
        let (timeline, summary) = one(1, &segments);
        assert_eq!(summary.clips, 1);
        assert!((timeline.clips[0].length_sec - 0.04).abs() < 1e-9);
    }

    /// **显式循环起点跨 take：重录同一循环区间会原子替换旧组。**
    #[test]
    fn a_new_take_with_the_same_loop_start_replaces_the_old_group() {
        let first = vec![segment_loop(0, &[1.0; 100], 0)];
        let second = vec![segment_loop(0, &[2.0; 40], 0)];
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
        assert_eq!(summary.clips, 1);
        assert_eq!(timeline.clips.len(), 1);
        assert_eq!(
            timeline.clips[0].active_take().source_path.as_deref(),
            Some("transfer://2")
        );
        assert!((timeline.clips[0].length_sec - 0.04).abs() < 1e-9);
    }
}
