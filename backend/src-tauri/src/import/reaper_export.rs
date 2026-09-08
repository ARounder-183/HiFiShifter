//! Convert selected HiFiShifter clips into REAPERMedia clipboard data.

use crate::reaper_parser::{
    item_time_to_take_env_u, ReaperData, ReaperEnvelope, ReaperIgnTempo, ReaperItem,
    ReaperMidiEvent, ReaperMidiSourceData, ReaperSource, ReaperTrack,
};
use crate::state::{Clip, ClipTake, TimelineState};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct ReaperExportResult {
    pub bytes: Vec<u8>,
    pub exported_clip_count: usize,
    pub skipped_clip_count: usize,
    pub track_count: usize,
}

/// 写出 FADEIN/FADEOUT 数组。
///
/// 布局对照 REAPER 7.x 实测样本（详见 reaper_parser::reaper_fade_shape_dir）：
/// `[shape, 手动长度, 自动交叉淡化长度, 取整形状(镜像槽), 自动selector, 曲率dir, 0]`。
/// 索引 3 在官方样本中存放取整后的基础形状（如形状 1.1 时写 1），保持一致；
/// 第二曲率参数（索引 6）语义未公开，固定写 0。
/// 手动/自动分别来自 Clip 的 `fade_*_sec` 与 `auto_fade_*_sec`，
/// 与应用内"有效 fade"模型一一对应，不再像旧实现那样合并成一个手动值。
pub(crate) fn fade_values(
    shape: f64,
    dir: f64,
    manual_length_sec: f64,
    auto_length_sec: f64,
) -> Vec<f64> {
    let shape = if shape.is_finite() { shape } else { 0.0 };
    let mirror_base = shape.trunc().clamp(0.0, 255.0);
    vec![
        shape,
        manual_length_sec.max(0.0),
        auto_length_sec.max(0.0),
        mirror_base,
        if auto_length_sec > 1e-9 { 1.0 } else { 0.0 },
        dir.clamp(-1.0, 1.0),
        0.0,
    ]
}

fn source_bounds(take: &ClipTake, clip_length_sec: f64) -> (f64, f64) {
    // 正放：保留负 source_start（导出为负 SOFFS = REAPER 左延伸 item 的前导静音）。
    // 倒放走 SECTION 域（RPP 的 SECTION STARTPOS 不能为负），钳制到 ≥0。
    let start = if take.reversed {
        take.source_start_sec.max(0.0)
    } else {
        take.source_start_sec
    };
    let mut end = take.source_end_sec.max(start);
    if end <= start {
        end = take
            .duration_sec
            .filter(|duration| *duration > start)
            .unwrap_or_else(|| start + clip_length_sec * take.playback_rate.max(0.01) as f64);
    }
    (start, end.max(start))
}

/// Map a media source path to the SOURCE type REAPER accepts.
///
/// REAPER only recognises WAVE, MP3 and FLAC as named audio source types.
/// Video containers and every other format (WMA, OGG, M4A, etc.) must be
/// written as VIDEO, otherwise REAPER fails to parse the clipboard data.
fn reaper_source_type(path: &str) -> &'static str {
    let path = std::path::Path::new(path);

    if crate::media::is_video_extension(path) {
        return "VIDEO";
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("wav") => "WAVE",
        Some(ext) if ext.eq_ignore_ascii_case("mp3") => "MP3",
        Some(ext) if ext.eq_ignore_ascii_case("flac") => "FLAC",
        _ => "VIDEO",
    }
}

fn audio_source(take: &ClipTake, _rate: f64, source_span_sec: f64) -> ReaperSource {
    let path = take.source_path.clone().unwrap_or_default();
    let (start, _) = source_bounds(take, 0.0);

    if take.reversed {
        let mut source = ReaperSource::new();
        source.source_type = "SECTION".to_string();
        source.file_path = path.clone();
        source.section_mode = 1;
        source.section_start_sec = Some(start);
        source.section_length_sec = Some(source_span_sec.max(0.0));
        source
    } else {
        let mut source = ReaperSource::new();
        source.source_type = reaper_source_type(&path).to_string();
        source.file_path = path;
        source.section_mode = 0;
        source.section_start_sec = None;
        source.section_length_sec = None;
        source
    }
}

fn midi_source(take: &ClipTake, bpm: f64) -> ReaperSource {
    const PPQ: u32 = 960;

    let notes = take.midi_note_data.as_deref().unwrap_or(&[]);
    let mut events: Vec<ReaperMidiEvent> = Vec::new();
    let mut raw_events: Vec<(u64, u8, u8, u8)> = Vec::new();

    for note in notes {
        let note_number = note.note.clamp(0.0, 127.0).round() as u8;
        let velocity = note.velocity;
        let channel = note.channel.min(15);
        let start_tick =
            (note.start_sec.max(0.0) * PPQ as f64 * bpm.max(1.0) / 60.0).round() as u64;
        let end_tick = ((note.end_sec.max(note.start_sec) * PPQ as f64 * bpm.max(1.0) / 60.0)
            .round() as u64)
            .max(start_tick);
        raw_events.push((start_tick, 0x90 | channel, note_number, velocity));
        raw_events.push((end_tick, 0x80 | channel, note_number, 0));
    }

    raw_events.sort_by_key(|event| (event.0, event.1));
    let mut cumulative_tick = 0_u64;
    for (tick, status, data1, data2) in raw_events {
        events.push(ReaperMidiEvent {
            tick_offset: tick.saturating_sub(cumulative_tick),
            status,
            data1,
            data2,
        });
        cumulative_tick = tick;
    }

    let mut source = ReaperSource::new();
    source.source_type = "MIDI".to_string();
    source.midi_source = Some(ReaperMidiSourceData {
        ticks_per_qn: PPQ,
        events,
        igntempo: Some(ReaperIgnTempo {
            ignore_project: false,
            tempo: bpm.max(1.0),
            beats: 4,
            beat_note: 4,
        }),
    });
    source
}

fn fill_reaper_take(
    dest: &mut crate::reaper_parser::ReaperTake,
    take: &ClipTake,
    clip_length_sec: f64,
    bpm: f64,
    is_item_default: bool,
    output_playback_rate: f32,
) -> bool {
    let rate = output_playback_rate.max(0.01).min(100.0) as f64;
    dest.name = take.name.clone();
    // REAPER 的字段语义：
    // - Item 默认 take：`VOLPAN <item trim> <pan> <take volume> <pan law>`。
    // - 显式 take：`TAKEVOLPAN <pan> <take volume> <pan law>`，
    //   默认值为 `0 1 -1`；不能沿用 Item 默认 take 的四元组写法。
    dest.vol_pan = if is_item_default {
        vec![1.0, 0.0, take.gain as f64, -1.0]
    } else {
        vec![0.0, take.gain as f64, -1.0]
    };
    dest.play_rate = vec![rate, 1.0, 0.0, -1.0, 0.0, 0.0025];
    dest.chan_mode = 0;

    if let Some(ref midi_data) = take.midi_note_data {
        if midi_data.is_empty() {
            return false;
        }
        dest.s_offs = 0.0;
        dest.source = Some(midi_source(take, bpm));
        return true;
    }

    let source_path = take.source_path.as_deref().unwrap_or("").trim();
    if source_path.is_empty() {
        return false;
    }

    if take.reversed && take.loop_enabled {
        // 倒放 + Loop：回绕发生在整个媒体文件上（与引擎及正放 Loop 的
        // "循环原始音频文件"语义一致）。用覆盖全媒体的 SECTION 承载回绕域，
        // SOFFS 承载倒放相位锚点。
        let media_dur = take
            .duration_sec
            .filter(|d| d.is_finite() && *d > 1e-9)
            .or_else(|| {
                take.duration_frames
                    .zip(take.source_sample_rate)
                    .filter(|(frames, sr)| *sr > 0 && *frames > 0)
                    .map(|(frames, sr)| frames as f64 / sr as f64)
            });
        if let Some(d) = media_dur {
            let anchor = take.source_end_sec.min(d).rem_euclid(d);
            let mut source = ReaperSource::new();
            source.source_type = "SECTION".to_string();
            source.file_path = source_path.to_string();
            source.section_mode = 1;
            source.section_start_sec = Some(0.0);
            source.section_length_sec = Some(d);
            dest.s_offs = (d - anchor).rem_euclid(d);
            dest.source = Some(source);
            return true;
        }
        // 媒体时长未知：退化为下方通用反向路径（尽力而为）。
    }

    let (start, end) = source_bounds(take, clip_length_sec);
    let source_span = (end - start).max(0.0);

    let mut source = audio_source(take, rate, source_span);
    if take.reversed {
        // 反向：SECTION MODE 1 承载源窗口，SOFFS 置 0。
        dest.s_offs = 0.0;
    } else {
        // 正向：plain SOURCE + SOFFS 承载进入锚点（可为负 = 前导静音）。
        // Loop 的回绕发生在整个媒体文件上（REAPER 原生 Loop source 语义），
        // 无需 SECTION；非 Loop 超出媒体的 LENGTH 部分由 REAPER 渲染静音。
        source.section_start_sec = None;
        source.section_length_sec = None;
        dest.s_offs = start;
    }
    dest.source = Some(source);
    true
}

fn build_item(
    clip: &Clip,
    bpm: f64,
    curves: &ClipExportCurves,
    qn_at: &dyn Fn(f64) -> f64,
) -> Option<ReaperItem> {
    let mut working = clip.clone();
    working.normalize_takes();
    if working.takes.is_empty() {
        return None;
    }
    let active_idx = working.active_take_index().min(working.takes.len() - 1);

    let mut item = ReaperItem::default();
    item.position = working.start_sec.max(0.0);
    item.length = working.length_sec.max(0.001);
    // 双时基（原生形态）：POSITION/LENGTH 附带 QN 值（Tempo Map 积分）。
    item.position_qn = Some(qn_at(item.position));
    item.length_qn = Some(qn_at(item.position + item.length) - item.position_qn.unwrap_or(0.0));
    // SnapOffset：相对 Clip 起点的偏移，与 REAPER SNAPOFFS 同语义直传。
    item.snap_offs = working.snap_offset_sec.max(0.0);
    // LOOP 是 REAPER 的 Item 级标志：多 take Clip 各 take 的 loop_enabled
    // 不同时，往返只能保留 active take 的值（导入端也按 item 级读取并共享
    // 给全部 take）。这是 RPP 格式的表达力边界，非实现遗漏。
    item.is_loop = working.takes[active_idx].loop_enabled;
    item.all_takes = false;
    // 手动长度与自动交叉淡化长度分槽写出（REAPER 索引 1 / 索引 2 +
    // selector），形状与曲率原样导出，保证 REAPER 渲染与应用一致。
    item.fade_in = fade_values(
        working.fade_in_shape,
        working.fade_in_dir,
        working.fade_in_sec,
        working.auto_fade_in_sec,
    );
    item.fade_out = fade_values(
        working.fade_out_shape,
        working.fade_out_dir,
        working.fade_out_sec,
        working.auto_fade_out_sec,
    );
    item.mute = vec![if working.muted { 1 } else { 0 }, 0];
    item.selected = false;

    let active_take = working.takes[active_idx].clone();
    // RPP 格式没有 Item 级 PLAYRATE 行：take 的 PLAYRATE 就是该 take 的
    // 有效播放速率（导入端也按此读取）。因此：
    // - default take（= active take）必须写 **组合有效速率**
    //   `clip_rate × take_rate` —— 与 develop 单窗口时代导出
    //   `clip.playback_rate` 的行为一致，否则带速率的 item 往返即丢速；
    // - 显式 take 同样写各自组合速率，保证在 REAPER 里切换 take 时
    //   听到的速度与 HiFiShifter 渲染一致。
    let clip_rate = if working.clip_playback_rate.is_finite() && working.clip_playback_rate > 1e-6 {
        working.clip_playback_rate
    } else {
        1.0
    };
    let combined_rate_f32 = clip_rate * active_take.playback_rate;
    let combined_rate = combined_rate_f32 as f64;
    if !fill_reaper_take(
        &mut item.default_take,
        &active_take,
        working.length_sec,
        bpm,
        true,
        combined_rate_f32,
    ) {
        return None;
    }

    // ── 音高参数线 → 活跃 take 的 PITCHENV（default take 的 ITEM 直属块） ──
    // 坐标为 take 媒体时间 u = (t − item_pos) × |PLAYRATE[0]|（唯一转换点
    // item_time_to_take_env_u）；值不钳制（REAPER PIT 值域无限制）。
    // MIDI take 不写音高包络（HiFiShifter 音高线基于音频分析，MIDI take
    // 的 PIT 包络在 REAPER 中无对应渲染语义）。
    if active_take.midi_note_data.is_none() {
        if let Some(env) = build_take_pitch_envelope(curves, item.length, combined_rate) {
            item.default_take.envelopes.push(env);
        }
    }

    // 显式 TAKE 块列出除 active 之外的全部 take（active 已由 default_take
    // 承载）。此前实现固定 skip(1)：当 active 不是第一个 take 时会把
    // active 重复写一份、而真正的第一个 take 被静默丢弃，再导入即得到
    // 错误的 take 集合。所有显式 take 均不打 SEL 标记 —— 导入端在无 SEL
    // 时回退到 default_take，恰好还原当前 active 选择。
    for (idx, take) in working.takes.iter().enumerate() {
        if idx == active_idx {
            continue;
        }
        let mut dest = crate::reaper_parser::ReaperTake::default();
        if fill_reaper_take(
            &mut dest,
            take,
            working.length_sec,
            bpm,
            false,
            clip_rate * take.playback_rate,
        ) {
            dest.selected = false;
            item.takes.push(dest);
        }
    }

    Some(item)
}

// ─── 参数线 → REAPER 包络 ────────────────────────────────────────────────────

/// 精简容差。
const SIMPLIFY_TOLERANCE_VOLUME: f64 = 0.005;
const SIMPLIFY_TOLERANCE_PAN: f64 = 0.005;
const SIMPLIFY_TOLERANCE_PITCH: f64 = 0.02;

/// 每个导出 Clip 的参数线视图（命令层从 live timeline 预解析后传入）。
///
/// 所有切片以"帧 0 = clip 起点"对齐；帧周期统一 `frame_period_ms`。
#[derive(Debug, Clone, Default)]
pub struct ClipExportCurves {
    pub frame_period_ms: f64,
    /// 逐帧半音偏移；**0 = 该帧不做音高修正**（无声/未分析/未编辑帧）。
    pub pitch_offset: Option<Vec<f32>>,
    /// volume 倍率切片（相对轨道推子，默认 1.0）。
    pub volume: Option<Vec<f32>>,
    /// pan 切片（默认 0.0，−1..1）。
    pub pan: Option<Vec<f32>>,
}

impl ClipExportCurves {
    /// 是否携带任何包络数据。
    pub fn is_empty(&self) -> bool {
        self.pitch_offset.is_none() && self.volume.is_none() && self.pan.is_none()
    }
}

fn curve_default_value(key: &str) -> f32 {
    if key == "pan" {
        0.0
    } else {
        1.0
    }
}

/// 曲线切片（帧 0 = clip 起点），不足处补参数默认值（与
/// `extract_linked_params_from_root_range` 的切片语义一致）。
fn slice_curve_with_default(
    curve: &[f32],
    start_frame: usize,
    count: usize,
    default: f32,
) -> Vec<f32> {
    (0..count)
        .map(|i| curve.get(start_frame + i).copied().unwrap_or(default))
        .collect()
}

/// 为导出 clip 预解析参数线视图（渲染同款合并语义：
/// `clip.extra_curves[key]` 按 key 覆盖根轨道曲线）。
pub fn build_clip_export_curves(timeline: &TimelineState, clip: &Clip) -> ClipExportCurves {
    let root = timeline.resolve_root_track_id(&clip.track_id);
    let root_entry = root.as_deref().and_then(|id| timeline.params_by_root_track.get(id));
    // Compose（合成）门控，与渲染端同语义：根轨道未开启 Compose 时音高
    // 曲线不参与渲染，REAPER 剪贴板同样不做音高参数的转换与输出 —— 否则
    // 导出的是渲染中根本不生效的陈旧数据。子轨道沿 parent 链回溯**根轨
    // 道**的 Compose 判定（子轨自身的开关不参与）；根轨道不存在时视为
    // 关闭（无从提供权威音高数据）。
    let root_compose_enabled = root
        .as_deref()
        .and_then(|id| timeline.tracks.iter().find(|t| t.id == id))
        .map(|t| t.compose_enabled)
        .unwrap_or(false);
    let frame_period_ms = root_entry
        .map(|entry| entry.frame_period_ms.max(0.1))
        .unwrap_or(5.0);
    let start_frame = ((clip.start_sec.max(0.0) * 1000.0) / frame_period_ms)
        .floor()
        .max(0.0) as usize;
    let frame_count =
        (((clip.length_sec.max(0.0) * 1000.0) / frame_period_ms).ceil().max(1.0)) as usize;

    let merged_curve = |key: &str| -> Option<Vec<f32>> {
        // 渲染同款合并：clip.extra_curves[key] 覆盖 root[key]（整 key 覆盖）；
        // volume 兼容旧 `hifigan_volume` key（common_volume_curve_for_clip 同款）。
        let legacy_key = if key == "volume" { Some("hifigan_volume") } else { None };
        let clip_override = clip
            .extra_curves
            .as_ref()
            .and_then(|c| c.get(key).or_else(|| legacy_key.and_then(|k| c.get(k))));
        let root_curve = root_entry.and_then(|e| {
            e.extra_curves
                .get(key)
                .or_else(|| legacy_key.and_then(|k| e.extra_curves.get(k)))
        });
        let curve = clip_override.or(root_curve)?;
        Some(slice_curve_with_default(
            curve,
            start_frame,
            frame_count,
            curve_default_value(key),
        ))
    };

    ClipExportCurves {
        frame_period_ms,
        volume: merged_curve("volume"),
        pan: merged_curve("pan"),
        pitch_offset: if root_compose_enabled {
            crate::pitch_editing::compute_clip_export_pitch_offsets(timeline, clip)
                .map(|p| p.offsets)
        } else {
            None
        },
    }
}

/// 视线简化：折线段间所有样本偏差 ≤ tolerance 才延伸，否则落折点。
///
/// - 全默认 → 空表（整条包络不写）；
/// - 端点强制：恒写 range_start 与 range_end 两端点（不得依赖 hold 语义
///   跨 Item / 段边界）；
/// - 平坦非默认 → 恰好 2 点；斜线 → 仅折点。
fn simplify_breakpoints(
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
        .all(|(_, v)| (v - default_value).abs() <= tolerance)
    {
        return Vec::new();
    }

    let value_at = |t: f64| -> f64 {
        let idx = samples.partition_point(|s| s.0 < t);
        if idx == 0 {
            return samples[0].1;
        }
        if idx >= samples.len() {
            return samples[samples.len() - 1].1;
        }
        let (t0, v0) = samples[idx - 1];
        let (t1, v1) = samples[idx];
        if (t1 - t0).abs() < 1e-12 {
            return v0;
        }
        let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        v0 + (v1 - v0) * frac
    };

    // 折线 a→f 是否容纳 (a, f) 内全部样本（端点精确）。
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
            .all(|(tj, vj)| (vj - (va + slope * (tj - ta))).abs() <= tolerance)
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

    // 首端点（强制）。
    let first_t = samples[0].0.max(range_start);
    push_point(first_t, value_at(first_t), &mut out);

    let mut anchor = 0usize;
    loop {
        // 找从 anchor 出发能容纳税入的最远样本。
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

    // 末端点（强制）。若末点值与端值在容差内相等（帧网格取不到精确 range_end
    // 的常态），把末点延伸到 range_end 而不是追加第四点——最后一段的偏差
    // 仍在容差内，且保持"线性段只有端点"的精简不变式。
    let end_t = range_end;
    let end_v = value_at(end_t);
    if let Some(last) = out.last_mut() {
        if (last.1 - end_v).abs() <= tolerance {
            last.0 = end_t;
            last.1 = end_v;
        } else {
            push_point(end_t, end_v, &mut out);
        }
    } else {
        push_point(end_t, end_v, &mut out);
    }
    out
}

/// 音高参数线 → 活跃 take 的 PITCHENV（default take 的 ITEM 直属块）。
///
/// - 坐标 u = t_rel × 组合速率（`item_time_to_take_env_u` 唯一转换点）；
/// - 端点强制（u=0 与 u=item_length×rate）；值为 0 的帧 = 不做音高修正
///  （显式 0 落点，不得依赖 hold 语义跨界或桥接邻近帧）；
/// - 值不钳制（REAPER PIT 值域无限制）；DEFSHAPE 范围按最大偏移写入。
fn build_take_pitch_envelope(
    curves: &ClipExportCurves,
    item_length_sec: f64,
    combined_rate: f64,
) -> Option<ReaperEnvelope> {
    let values = curves.pitch_offset.as_ref()?;
    let frame_period_ms = if curves.frame_period_ms > 0.0 {
        curves.frame_period_ms
    } else {
        5.0
    };
    let samples: Vec<(f64, f64)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64 * frame_period_ms / 1000.0, *v as f64))
        .collect();
    let range_end = item_length_sec.max(1e-6);
    let breakpoints = simplify_breakpoints(&samples, 0.0, range_end, 0.0, SIMPLIFY_TOLERANCE_PITCH);
    if breakpoints.is_empty() {
        return None;
    }

    let points: Vec<Vec<f64>> = breakpoints
        .iter()
        .map(|(t, v)| {
            vec![
                item_time_to_take_env_u(*t, combined_rate),
                *v,
                0.0, // shape = 线性
            ]
        })
        .collect();
    let max_abs = points
        .iter()
        .map(|p| p[1].abs())
        .fold(0.0_f64, f64::max);
    let display_range = (max_abs.ceil() as i64).clamp(3, 96) as f64;

    Some(ReaperEnvelope {
        env_type: "PITCHENV".to_string(),
        act: vec![1, -1],
        seg_range: None,
        points,
        def_shape: Some(vec![0.0, display_range, -1.0]),
    })
}

/// 音量/声像参数线 → ENVSEG 轨道包络段（每个连续 clip 组一条段）。
///
/// - VOLENV2 值 = 轨道推子 × 曲线（绝对量）；PANENV2 值 = 曲线值；
/// - 段范围 = 连续组跨度（gap ≤ 1 帧视为连续；无曲线数据的 clip 打断分组，
///   其范围不属于任何段 → REAPER 段外不生效，保持目标轨既有自动化）；
/// - 段内相对坐标 + 端点强制（相对 0 / 相对 len）；QN 域由 bpm 换算。
fn build_track_envseg_envelopes(
    clip_pairs: &[(&Clip, &ClipExportCurves)],
    track_volume: f64,
    timeline: &TimelineState,
) -> Vec<ReaperEnvelope> {
    // QN 字段按 Tempo Map 积分换算（原生 REAPER 语义：变速工程下 QN ≠
    // 秒 × 常量 BPM 的线性值，否则节拍网格错位）。
    let converter = crate::commands::TempoTickConverter::new(timeline, timeline.bpm);
    let qn_at = move |sec: f64| -> f64 { converter.sec_to_qn(sec) };
    let mut out = Vec::new();
    build_envseg_for_key(
        clip_pairs,
        EnvSegKey::Volume,
        "VOLENV2",
        track_volume,
        SIMPLIFY_TOLERANCE_VOLUME,
        track_volume,
        &qn_at,
        &mut out,
    );
    build_envseg_for_key(
        clip_pairs,
        EnvSegKey::Pan,
        "PANENV2",
        0.0,
        SIMPLIFY_TOLERANCE_PAN,
        1.0,
        &qn_at,
        &mut out,
    );
    out
}

/// ENVSEG 段的参数线种类。
#[derive(Clone, Copy, PartialEq)]
enum EnvSegKey {
    Volume,
    Pan,
}

#[allow(clippy::too_many_arguments)]
fn build_envseg_for_key(
    clip_pairs: &[(&Clip, &ClipExportCurves)],
    key: EnvSegKey,
    env_type: &str,
    default_value: f64,
    tolerance: f64,
    volume_factor: f64,
    qn_at: &dyn Fn(f64) -> f64,
    out: &mut Vec<ReaperEnvelope>,
) {
    let frame_period = 5.0f64;
    let merge_gap_sec = frame_period / 1000.0 + 1e-6;

    // 每个 clip 的采样（绝对工程秒）；无曲线数据的 clip 不产生采样但参与分组。
    let mut entries: Vec<(f64, f64, Option<Vec<(f64, f64)>>)> = clip_pairs
        .iter()
        .map(|(clip, curves)| {
            let clip_start = clip.start_sec.max(0.0);
            let clip_end = clip_start + clip.length_sec.max(0.0);
            let curve = match key {
                EnvSegKey::Volume => curves.volume.as_ref(),
                EnvSegKey::Pan => curves.pan.as_ref(),
            };
            let fp = if curves.frame_period_ms > 0.0 {
                curves.frame_period_ms
            } else {
                frame_period
            };
            let samples = curve.map(|curve| {
                curve
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let t = clip_start + i as f64 * fp / 1000.0;
                        let value = match key {
                            // VOLENV2 = 绝对量（推子 × 曲线倍率）。
                            EnvSegKey::Volume => (volume_factor * *v as f64).clamp(0.0, 4.0),
                            EnvSegKey::Pan => (*v as f64).clamp(-1.0, 1.0),
                        };
                        (t, value)
                    })
                    .filter(|(t, _)| *t <= clip_end + merge_gap_sec)
                    .collect::<Vec<(f64, f64)>>()
            });
            (clip_start, clip_end, samples)
        })
        .collect();
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));

    // 连续组聚合。
    let mut group: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut group_start = 0.0f64;
    let mut group_end = 0.0f64;
    let flush = |group: &mut Vec<Vec<(f64, f64)>>,
                 group_start: &mut f64,
                 group_end: &mut f64,
                 out: &mut Vec<ReaperEnvelope>| {
        if group.is_empty() {
            return;
        }
        let (start, end) = (*group_start, *group_end);
        let mut samples: Vec<(f64, f64)> = group.drain(..).flatten().collect();
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let relative: Vec<(f64, f64)> = samples
            .into_iter()
            .map(|(t, v)| (t - start, v))
            .collect();
        let span_end = end - start;
        let breakpoints = simplify_breakpoints(&relative, 0.0, span_end, default_value, tolerance);
        if !breakpoints.is_empty() {
            // 原生 REAPER 剪贴板语义（样例 ClipboardData/20260908-025849，
            // 多轨双时基形态）：
            // - SEG_RANGE = [起秒, 终秒, 起QN, 终QN]——第二字段是**终点**
            //   而非长度；起点/终点均为工程绝对值（与 item POSITION 同基）；
            // - PT 位置与第 8 字段 QN = **工程绝对值**（start+t / QN(start+t)），
            //   REAPER 粘贴时把复制区首个时间位置对齐到光标（整体平移），
            //   段与 item 的相对关系由此保持。
            let start_qn = qn_at(start);
            let end_qn = qn_at(start + span_end);
            let points = breakpoints
                .iter()
                .map(|(t, v)| {
                    vec![
                        start + t,
                        *v,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        qn_at(start + t),
                    ]
                })
                .collect();
            out.push(ReaperEnvelope {
                env_type: env_type.to_string(),
                act: vec![1, -1],
                seg_range: Some(vec![start, start + span_end, start_qn, end_qn]),
                points,
                def_shape: None,
            });
        }
        *group_start = 0.0;
        *group_end = 0.0;
    };

    for (clip_start, clip_end, samples) in entries {
        match samples {
            Some(samples) if !samples.is_empty() => {
                let contiguous = !group.is_empty() && clip_start <= group_end + merge_gap_sec;
                if !contiguous {
                    flush(&mut group, &mut group_start, &mut group_end, out);
                }
                if group.is_empty() {
                    group_start = clip_start;
                    group_end = clip_end;
                } else {
                    group_end = group_end.max(clip_end);
                }
                group.push(samples);
            }
            _ => {
                // 无曲线数据的 clip 打断分组（其范围不进段）。
                flush(&mut group, &mut group_start, &mut group_end, out);
            }
        }
    }
    flush(&mut group, &mut group_start, &mut group_end, out);
}

pub fn build_reaper_clipboard(
    timeline: &TimelineState,
    clip_ids: &[String],
    curves_by_clip: &BTreeMap<String, ClipExportCurves>,
) -> Result<ReaperExportResult, String> {
    let unique_ids: Vec<&String> = {
        let mut seen = std::collections::HashSet::new();
        clip_ids
            .iter()
            .filter(|id| seen.insert((*id).clone()))
            .collect()
    };

    let mut items_by_track: BTreeMap<usize, Vec<ReaperItem>> = BTreeMap::new();
    // (导出轨道下标 → 该轨道导出 clip 的 (clip, 曲线视图)，用于轨道包络段聚合)
    let mut clips_by_track: BTreeMap<usize, Vec<(&Clip, &ClipExportCurves)>> = BTreeMap::new();
    let mut skipped_clip_count = 0_usize;
    // 无曲线数据 clip 的共享空视图（生命周期覆盖整个导出流程）。
    let empty_curves = ClipExportCurves::default();
    // 双时基 QN 换算（原生 REAPER 剪贴板的 POSITION/LENGTH/ENVSEG 均携带
    // 按 Tempo Map 积分的 QN 值）。
    let qn_converter = crate::commands::TempoTickConverter::new(timeline, timeline.bpm);
    let qn_at = move |sec: f64| -> f64 { qn_converter.sec_to_qn(sec) };
    let qn_at = &qn_at;

    for clip_id in unique_ids {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == *clip_id) else {
            skipped_clip_count += 1;
            continue;
        };
        let Some(track_index) = timeline
            .tracks
            .iter()
            .position(|track| track.id == clip.track_id)
        else {
            skipped_clip_count += 1;
            continue;
        };
        let curves = curves_by_clip.get(clip_id).unwrap_or(&empty_curves);
        let Some(item) = build_item(clip, timeline.bpm, curves, qn_at) else {
            skipped_clip_count += 1;
            continue;
        };
        items_by_track.entry(track_index).or_default().push(item);
        clips_by_track
            .entry(track_index)
            .or_default()
            .push((clip, curves));
    }

    if items_by_track.is_empty() {
        return Err("reaper_export_no_supported_clips".to_string());
    }

    let track_indices: Vec<usize> = items_by_track.keys().copied().collect();
    let first_track_index = track_indices.first().copied().unwrap_or(0);

    let mut data = ReaperData::default();
    data.is_track_data = false;
    for (offset, track_index) in track_indices.iter().enumerate() {
        let mut track = ReaperTrack::default();
        track.items = items_by_track.remove(track_index).unwrap_or_default();
        track.items.sort_by(|left, right| {
            left.position
                .partial_cmp(&right.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // ── 音量/声像参数线 → ENVSEG 轨道包络段（随 item 剪贴板携带） ──
        if let Some(clip_pairs) = clips_by_track.remove(track_index) {
            // VOLENV2 是绝对量 → 用**链式有效推子增益**（根轨×父×子，与
            // 渲染端 compute_track_gains 同款）；组内 clip 单看自身轨道
            // 音量会丢根轨增益。
            let track_volume = timeline
                .tracks
                .get(*track_index)
                .map(|t| timeline.effective_track_volume(&t.id) as f64)
                .unwrap_or(1.0);
            track.envelopes = build_track_envseg_envelopes(&clip_pairs, track_volume, timeline);
        }
        data.tracks.push(track);
        data.track_offsets
            .push(track_index.saturating_sub(first_track_index) as usize);
        if offset == 0 {
            data.track_offsets[0] = 0;
        }
    }

    let exported_clip_count = data.tracks.iter().map(|track| track.items.len()).sum();
    let bytes = crate::reaper_parser::serialize_reaper_clipboard(&data, false);
    Ok(ReaperExportResult {
        bytes,
        exported_clip_count,
        skipped_clip_count,
        track_count: data.tracks.len(),
    })
}

/// Round-trip helper used by tests.
#[cfg(test)]
pub(crate) fn parse_for_test(bytes: &[u8]) -> crate::reaper_parser::ReaperData {
    crate::reaper_parser::parse_clipboard_bytes(bytes).expect("parse exported REAPER clipboard")
}

/// 测试辅助：按导入端约定从 SECTION take 还原倒放锚点（区间末端 − SOFFS）。
#[cfg(test)]
fn compute_anchor_from_section_for_test(take: &crate::reaper_parser::ReaperTake) -> f64 {
    let src = take.source.as_ref().expect("source present");
    let end = src.section_start_sec.unwrap_or(0.0)
        + src
            .section_length_sec
            .filter(|len| len.is_finite() && *len > 0.0)
            .unwrap_or(0.0);
    end - take.s_offs.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_import::MidiNoteEvent;
    use crate::state::TimelineState;

    #[test]
    fn multi_take_export_uses_reaper_takevolpan_layout() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Multi Take".to_string()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            // 直接写投影字段后必须 sync：takes 是磁盘/导出权威，
            // 否则 normalize_takes 会用陈旧 Take 覆盖这些值。
            let mut second = clip.active_take().clone();
            second.id = crate::state::new_id("take");
            second.name = "Second".to_string();
            second.source_path = Some("C:/audio/b.wav".to_string());
            clip.add_take(second);
            clip.gain = 0.75;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        // Item 默认 take：VOLPAN <trim> <pan> <take volume> <pan law>。
        assert_eq!(item.default_take.vol_pan.len(), 4);
        assert!((item.default_take.vol_pan[2] - 0.75).abs() < 1e-9);
        // 显式 take：TAKEVOLPAN <pan> <take volume> <pan law>，默认音量 1。
        assert_eq!(item.takes.len(), 1);
        assert_eq!(item.takes[0].vol_pan.len(), 3);
        assert_eq!(item.takes[0].vol_pan[0], 0.0);
        assert_eq!(item.takes[0].vol_pan[1], 1.0);
        assert_eq!(item.takes[0].vol_pan[2], -1.0);
    }

    #[test]
    fn multi_take_export_preserves_rates_and_nonzero_active_index() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Rates".to_string()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            // takes = [Rates(1.0), Second(1.0)]；随后把 Clip 级倍率设为 2.0，
            // 并给第二个 take 自身速率 1.5（组合有效速率 3.0），再把 active
            // 切到第二个 take —— 覆盖“active 非首位 + 非 1 速率”的导出路径。
            let mut second = clip.active_take().clone();
            second.id = crate::state::new_id("take");
            second.name = "Second".to_string();
            second.source_path = Some("C:/audio/b.wav".to_string());
            clip.add_take(second);
            clip.clip_playback_rate = 2.0;
            clip.playback_rate = 2.0;
            clip.sync_take_from_flat();
            let second_id = clip.takes[1].id.clone();
            clip.switch_active_take(&second_id)
                .expect("second take exists");
            clip.takes[1].playback_rate = 1.5;
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        // active take（Second）作为 default take：PLAYRATE = clip 级 × take 自身。
        assert_eq!(item.default_take.name, "Second");
        assert!((item.default_take.play_rate[0] - 3.0).abs() < 1e-9);
        // 显式 TAKE 块只含非 active 的第一个 take，且不打 SEL 标记：
        // 导入端在无 SEL 时回退 default_take，恰好还原 active 选择。
        assert_eq!(item.takes.len(), 1);
        assert_eq!(item.takes[0].name, "Rates");
        assert!((item.takes[0].play_rate[0] - 2.0).abs() < 1e-9);
        assert!(!item.takes[0].selected);
    }

    #[test]
    fn audio_clip_roundtrips_through_reaper_clipboard() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Test Audio".to_string()),
            Some(1.5),
            Some(2.0),
            Some("C:/audio/test.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.source_start_sec = 0.25;
            clip.source_end_sec = 3.0;
            clip.playback_rate = 1.0;
            clip.gain = 0.8;
            clip.fade_in_sec = 0.01;
            clip.fade_out_sec = 0.02;
            // 本测试考察普通裁剪 Clip 的往返，须显式关闭 add_clip 的
            // 进程级默认 Loop；写完投影统一 sync 回 Take 权威数据。
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        assert_eq!(export.exported_clip_count, 1);
        let parsed = parse_for_test(&export.bytes);
        assert_eq!(parsed.tracks.len(), 1);
        assert_eq!(parsed.tracks[0].items.len(), 1);
        let item = &parsed.tracks[0].items[0];
        assert!((item.position - 1.5).abs() < 1e-9);
        assert!((item.length - 2.0).abs() < 1e-9);
        assert!((item.default_take.s_offs - 0.25).abs() < 1e-9);
        assert_eq!(
            item.default_take.source.as_ref().unwrap().file_path,
            "C:/audio/test.wav"
        );
    }

    fn source_type_for_path(path: &str) -> String {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Source Type Test".to_string()),
            Some(0.0),
            Some(1.0),
            Some(path.to_string()),
        );

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        let parsed = parse_for_test(&export.bytes);
        parsed.tracks[0].items[0]
            .default_take
            .source
            .as_ref()
            .unwrap()
            .source_type
            .clone()
    }

    #[test]
    fn audio_source_type_follows_reaper_clipboard_conventions() {
        assert_eq!(source_type_for_path("C:/audio/song.wav"), "WAVE");
        assert_eq!(source_type_for_path("C:/audio/song.MP3"), "MP3");
        assert_eq!(source_type_for_path("C:/audio/song.flac"), "FLAC");
        assert_eq!(source_type_for_path("C:/video/movie.mp4"), "VIDEO");
        assert_eq!(source_type_for_path("C:/video/movie.mkv"), "VIDEO");
        assert_eq!(source_type_for_path("C:/audio/song.wma"), "VIDEO");
        assert_eq!(source_type_for_path("C:/audio/song.ogg"), "VIDEO");
        assert_eq!(source_type_for_path("C:/audio/song"), "VIDEO");
    }

    #[test]
    fn midi_clip_roundtrips_through_reaper_clipboard() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Test MIDI".to_string()),
            Some(0.0),
            Some(2.0),
            None,
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.midi_note_data = Some(vec![MidiNoteEvent {
                start_sec: 0.5,
                end_sec: 1.0,
                note: 60.0,
                velocity: 100,
                channel: 0,
            }]);
            clip.source_path = None;
            // MIDI 内容写在投影上，须 sync 回 Take 权威数据供导出读取。
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        assert_eq!(export.exported_clip_count, 1);
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        let source = item.default_take.source.as_ref().unwrap();
        assert_eq!(source.source_type, "MIDI");
        assert!(!source.midi_source.as_ref().unwrap().events.is_empty());
    }

    #[test]
    fn unsupported_clip_is_skipped() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("No source".to_string()),
            Some(0.0),
            Some(1.0),
            None,
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.source_path = None;
            clip.midi_note_data = None;
        }
        let result = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn looping_clip_roundtrips_loop_flag_and_section_window() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Looped".to_string()),
            Some(0.0),
            Some(6.0),
            Some("C:/audio/loop.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.duration_sec = Some(4.0);
            // 循环窗口是媒体的一个子区间：导出必须用 SECTION 表达，
            // 否则 REAPER 会在媒体末尾而不是窗口末尾回绕。
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 3.0;
            clip.playback_rate = 1.0;
            clip.loop_enabled = true;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id.clone()], &BTreeMap::new()).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!(item.is_loop, "loop flag must be exported");
        let source = item.default_take.source.as_ref().unwrap();
        // 正向 Loop = plain SOURCE + SOFFS（进入锚点）：
        // REAPER 原生 Loop source 在整个媒体上回绕，无需 SECTION。
        assert_eq!(source.source_type, "WAVE");
        assert!((item.default_take.s_offs - 1.0).abs() < 1e-9);

        // 非 Loop 的同窗口 clip 不应推断出 LOOP。
        {
            let clip = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .expect("clip exists");
            clip.loop_enabled = false;
            clip.length_sec = 2.0;
            // 同步纪律：改写投影后必须写回 Take 权威数据。
            clip.sync_take_from_flat();
        }
        let export2 = build_reaper_clipboard(&timeline, &[clip_id.clone()], &BTreeMap::new()).unwrap();
        let parsed2 = parse_for_test(&export2.bytes);
        let item2 = &parsed2.tracks[0].items[0];
        assert!(!item2.is_loop, "short non-loop clip must not infer loop");
        assert_eq!(
            item2.default_take.source.as_ref().unwrap().source_type,
            "WAVE"
        );
    }

    #[test]
    fn negative_soffs_silence_tail_roundtrips() {
        // LENGTH 覆盖整个可见区间。导出必须逐字保留负 SOFFS 与 LOOP 0 ——
        // 不得把静音尾巴推断成 LOOP，也不得把 SOFFS 钳到 0。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Left Extended".to_string()),
            Some(20.0),
            Some(16.81342267992402),
            Some("C:/audio/Vocal-1-3.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.duration_sec = Some(15.25232042998004);
            clip.source_start_sec = -1.56110224994398;
            clip.source_end_sec = 15.25232042998004;
            clip.playback_rate = 1.0;
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!(
            (item.default_take.s_offs - (-1.56110224994398)).abs() < 1e-9,
            "negative SOFFS must survive export verbatim, got {}",
            item.default_take.s_offs
        );
        assert!(!item.is_loop, "silence-tail non-loop clip must stay LOOP 0");
        assert!((item.length - 16.81342267992402).abs() < 1e-9);
    }

    #[test]
    fn reversed_loop_clip_exports_whole_file_section_with_anchor() {
        // 倒放 + Loop：SECTION 必须覆盖整个媒体文件（回绕域），
        // SOFFS 承载倒放相位锚点：SOFFS = D − floor_mod(min(source_end, D), D)。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Reversed Loop".to_string()),
            Some(0.0),
            Some(6.0),
            Some("C:/audio/loop.wav".to_string()),
        );
        if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
            clip.duration_sec = Some(4.0);
            clip.reversed = true;
            clip.loop_enabled = true;
            // split 可能产生的环绕窗口（start > end）：引擎只按
            // floor_mod(min(source_end, D), D) 取倒放锚点。
            clip.source_start_sec = 3.0;
            clip.source_end_sec = 1.0;
            clip.sync_take_from_flat();
        }

        let export = build_reaper_clipboard(&timeline, &[clip_id], &BTreeMap::new()).unwrap();
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!(item.is_loop, "loop flag must be exported");
        let source = item.default_take.source.as_ref().unwrap();
        assert_eq!(
            source.section_mode, 1,
            "reversal must be expressed via SECTION MODE"
        );
        assert!(
            source.section_start_sec == Some(0.0) && source.section_length_sec == Some(4.0),
            "SECTION must cover the whole media file, got {:?}..{:?}",
            source.section_start_sec,
            source.section_length_sec
        );
        // φ = mod(min(1, 4), 4) = 1 ⇒ SOFFS = 4 − 1 = 3。
        assert!(
            (item.default_take.s_offs - 3.0).abs() < 1e-9,
            "reverse anchor phase must round-trip, got {}",
            item.default_take.s_offs
        );

        // 导入端按同一约定还原倒放锚点：anchor = 区间末端 − SOFFS = 4 − 3 = 1 = φ。
        let take = &item.default_take;
        let anchor = super::compute_anchor_from_section_for_test(take);
        assert!((anchor - 1.0).abs() < 1e-9);
    }

    // ─── 参数线 → 包络导出 ───

    #[test]
    fn reaper_export_track_order_follows_display_order_after_drag() {
        // 拖拽换序后导出：REAPER 剪贴板的 TRACK 顺序必须跟随显示顺序
        //（tracks Vec 顺序，normalize_track_vec 不变式），而非轨道创建顺序。
        let mut timeline = TimelineState::default();
        let track_a = timeline.tracks[0].id.clone();
        let track_b = timeline.add_track(Some("B".into()), None, None);
        let clip_a = timeline.add_clip(
            Some(track_a),
            Some("A".into()),
            Some(0.0),
            Some(1.0),
            Some("C:/audio/a.wav".into()),
        );
        let clip_b = timeline.add_clip(
            Some(track_b.clone()),
            Some("B".into()),
            Some(0.0),
            Some(1.0),
            Some("C:/audio/b.wav".into()),
        );

        // 拖拽：B 移到最上。
        timeline.move_track(&track_b, 0, None);
        assert_eq!(timeline.tracks[0].id, track_b, "Vec 顺序反映拖拽");

        let export = build_reaper_clipboard(&timeline, &[clip_a, clip_b], &BTreeMap::new())
            .expect("export");
        let parsed = parse_for_test(&export.bytes);
        assert_eq!(parsed.tracks.len(), 2);
        // 第一条 TRACK = 显示顺序第一的 B 轨道。
        assert_eq!(
            parsed.tracks[0].items[0]
                .default_take
                .source
                .as_ref()
                .unwrap()
                .file_path,
            "C:/audio/b.wav"
        );
        assert_eq!(
            parsed.tracks[1].items[0]
                .default_take
                .source
                .as_ref()
                .unwrap()
                .file_path,
            "C:/audio/a.wav"
        );
    }

    fn pitch_values(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[test]
    fn simplify_breakpoints_rules() {
        let fp = 0.005f64;
        // 全默认（样本 == 默认值 1.0）→ 空表。
        let default_samples: Vec<(f64, f64)> =
            (0..400).map(|i| (i as f64 * fp, 1.0)).collect();
        assert!(simplify_breakpoints(&default_samples, 0.0, 2.0, 1.0, 0.005).is_empty());
        // 平坦非默认 → 恰 2 端点。
        let flat: Vec<(f64, f64)> = (0..400).map(|i| (i as f64 * fp, 0.8)).collect();
        let out = simplify_breakpoints(&flat, 0.0, 2.0, 1.0, 0.005);
        assert_eq!(out.len(), 2);
        assert!(out[0].0.abs() < 1e-9 && (out[0].1 - 0.8).abs() < 1e-9);
        assert!((out[1].0 - 2.0).abs() < 1e-9);
        // 线性斜线 → 恰 2 端点（中间点全在容差内的直线上）。
        let line: Vec<(f64, f64)> = (0..400)
            .map(|i| {
                let t = i as f64 * fp;
                (t, 1.0 - t / 2.0)
            })
            .collect();
        let out = simplify_breakpoints(&line, 0.0, 2.0, 1.0, 0.005);
        assert_eq!(out.len(), 2);
        // 折线 → 仅折点（slope 变化处）+ 端点。
        let bent: Vec<(f64, f64)> = (0..400)
            .map(|i| {
                let t = i as f64 * fp;
                (t, if t < 1.0 { t } else { 2.0 - t })
            })
            .collect();
        let out = simplify_breakpoints(&bent, 0.0, 2.0, 1.0, 0.005);
        // 斜率在 t=1 处翻转：应产生 (0,0) (≈1,≈1) (2,≈0)（端点强制）。
        assert!(out.len() <= 4, "折点数量应精简，实际 {:?}", out);
        assert!(out.iter().any(|(t, v)| (t - 1.0).abs() < 0.02 && (v - 1.0).abs() < 0.02));
    }

    #[test]
    fn pitch_env_zero_frames_mean_no_correction() {
        // 原始音高 / 当前音高为 0 的帧 → 偏移 = 0（"不做音高修正"），
        // 不得用邻近有声帧的修正值桥接（旧实现把无声区污染成 +2）。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id.clone()),
            Some("Zero".to_string()),
            Some(0.0),
            Some(0.015),
            Some("C:/audio/a.wav".to_string()),
        );
        // 3 帧（5ms）：orig [60, 0, 60]、edit [62, 62, 0] → 偏移 [2, 0, 0]。
        timeline.params_by_root_track.insert(
            track_id,
            crate::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_orig: vec![60.0, 0.0, 60.0],
                pitch_edit: vec![62.0, 62.0, 0.0],
                pitch_edit_user_modified: true,
                ..crate::state::TrackParamsState::default()
            },
        );
        let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
        let result =
            crate::pitch_editing::compute_clip_export_pitch_offsets(&timeline, clip)
                .expect("offsets");
        assert_eq!(result.offsets, vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn clip_in_track_group_resolves_root_track_curves() {
        // 轨道组内的 clip：参数线存于根轨道 entry，曲线视图必须经
        // resolve_root_track_id 命中（修复：导出曾用 fragment timeline 导致
        // 组内 clip 包络完全缺失）。
        let mut timeline = TimelineState::default();
        let root_id = timeline.tracks[0].id.clone();
        // 音高导出门控在根轨道 Compose：命中根轨道解析的前提是根轨已开启。
        timeline.tracks[0].compose_enabled = true;
        timeline.tracks.push(crate::state::Track {
            id: "child_track_t".to_string(),
            name: "Child".to_string(),
            parent_id: Some(root_id.clone()),
            order: 1,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: false,
            pitch_analysis_algo: crate::state::PitchAnalysisAlgo::default(),
            color: "#888888".to_string(),
        });
        let clip_id = timeline.add_clip(
            Some("child_track_t".to_string()),
            Some("Grouped".to_string()),
            Some(0.0),
            Some(0.05),
            Some("C:/audio/a.wav".to_string()),
        );
        timeline.params_by_root_track.insert(
            root_id,
            crate::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_orig: vec![60.0; 10],
                pitch_edit: vec![61.0; 10],
                pitch_edit_user_modified: true,
                extra_curves: {
                    let mut curves = std::collections::HashMap::new();
                    curves.insert("volume".to_string(), vec![0.5f32; 10]);
                    curves
                },
                ..crate::state::TrackParamsState::default()
            },
        );
        let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
        let curves = build_clip_export_curves(&timeline, clip);
        assert!(curves.pitch_offset.is_some(), "音高偏移经根轨道解析命中");
        assert!(curves.volume.is_some(), "音量曲线经根轨道解析命中");
        assert!(
            (curves.pitch_offset.as_ref().unwrap()[0] - 1.0).abs() < 1e-6,
            "偏移 = pitch_edit − pitch_orig = 1"
        );
        assert!((curves.volume.as_ref().unwrap()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reaper_pitch_export_gated_by_root_track_compose() {
        // Compose（合成）门控：根轨道未开启 Compose 时不做音高参数的转换
        // 与输出（pitch_offset = None → take 无 PITCHENV）；音量/声像曲线
        // 不受门控影响。子轨道的 clip 沿 parent 链按**根轨道**判定：
        // 子轨关/根轨开 → 仍导出；子轨开/根轨关 → 不导出。
        let mut timeline = TimelineState::default();
        let root_id = timeline.tracks[0].id.clone();
        // 子轨 compose=false：判定必须回溯根轨道，而不是子轨自身。
        timeline.tracks.push(crate::state::Track {
            id: "child_track_t".to_string(),
            name: "Child".to_string(),
            parent_id: Some(root_id.clone()),
            order: 1,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: false,
            pitch_analysis_algo: crate::state::PitchAnalysisAlgo::default(),
            color: "#888888".to_string(),
        });
        let child_clip_id = timeline.add_clip(
            Some("child_track_t".to_string()),
            Some("Child Clip".to_string()),
            Some(0.0),
            Some(0.01),
            Some("C:/audio/a.wav".to_string()),
        );
        let root_clip_id = timeline.add_clip(
            Some(root_id.clone()),
            Some("Root Clip".to_string()),
            Some(0.0),
            Some(0.01),
            Some("C:/audio/a.wav".to_string()),
        );
        timeline.params_by_root_track.insert(
            root_id.clone(),
            crate::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_orig: vec![60.0; 4],
                pitch_edit: vec![61.0; 4],
                pitch_edit_user_modified: true,
                extra_curves: {
                    let mut curves = std::collections::HashMap::new();
                    curves.insert("volume".to_string(), vec![0.5f32; 4]);
                    curves
                },
                ..crate::state::TrackParamsState::default()
            },
        );

        // 根轨 compose=off：根轨与子轨上的 clip 都不导出音高，音量照常。
        for clip_id in [child_clip_id.as_str(), root_clip_id.as_str()] {
            let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
            let curves = build_clip_export_curves(&timeline, clip);
            assert!(
                curves.pitch_offset.is_none(),
                "根轨 compose=off 时 clip {clip_id} 不得导出音高偏移"
            );
            assert!(
                curves.volume.is_some(),
                "音量/声像曲线不受 compose 门控影响"
            );
        }

        // 根轨 compose=on：子轨（自身 compose=false）的 clip 照常导出音高
        // —— 判定只看根轨道。
        timeline.tracks[0].compose_enabled = true;
        let clip = timeline
            .clips
            .iter()
            .find(|c| c.id == child_clip_id)
            .unwrap();
        let curves = build_clip_export_curves(&timeline, clip);
        assert!(
            curves.pitch_offset.is_some(),
            "根轨 compose=on 时子轨 clip 正常导出音高偏移"
        );

        // 根轨 compose=off、子轨 compose=on：仍不导出（同样只看根轨道）。
        timeline.tracks[0].compose_enabled = false;
        if let Some(child) = timeline.tracks.iter_mut().find(|t| t.id == "child_track_t") {
            child.compose_enabled = true;
        }
        let clip = timeline
            .clips
            .iter()
            .find(|c| c.id == child_clip_id)
            .unwrap();
        let curves = build_clip_export_curves(&timeline, clip);
        assert!(
            curves.pitch_offset.is_none(),
            "子轨自身 compose=on 不能越过根轨道的门控"
        );
    }

    #[test]
    fn export_pitch_offsets_fold_child_scale_and_cents_offsets() {
        // 子轨音分差 +50（= +0.5 半音）作用在编辑后音高上：
        // orig 60、edit 62 → 有效 62.5 → 偏移 = 2.5。
        let mut timeline = TimelineState::default();
        let root_id = timeline.tracks[0].id.clone();
        timeline.tracks.push(crate::state::Track {
            id: "child_track_t".to_string(),
            name: "Child".to_string(),
            parent_id: Some(root_id.clone()),
            order: 1,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: false,
            pitch_analysis_algo: crate::state::PitchAnalysisAlgo::default(),
            color: "#888888".to_string(),
        });
        let clip_id = timeline.add_clip(
            Some("child_track_t".to_string()),
            Some("Child Clip".to_string()),
            Some(0.0),
            Some(0.01),
            Some("C:/audio/a.wav".to_string()),
        );
        timeline.params_by_root_track.insert(
            root_id,
            crate::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_orig: vec![60.0, 60.0],
                pitch_edit: vec![62.0, 62.0],
                pitch_edit_user_modified: true,
                extra_curves: {
                    let mut curves = std::collections::HashMap::new();
                    curves.insert(
                        "child_pitch_offset_cents@child_track_t".to_string(),
                        vec![50.0f32, 50.0],
                    );
                    curves
                },
                ..crate::state::TrackParamsState::default()
            },
        );
        let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
        let result =
            crate::pitch_editing::compute_clip_export_pitch_offsets(&timeline, clip)
                .expect("offsets");
        assert_eq!(result.offsets, vec![2.5, 2.5]);
    }

    #[test]
    fn pitch_env_u_coordinates_scale_with_combined_rate_and_forces_endpoints() {
        // clip 2s、组合速率 0.5 → u 域 0..1；线性偏移 +1 → −1。
        // 端点强制：u=0 与 u_end 恒存在；线性段仅 2 点。
        let n = 400usize;
        let offsets: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.005;
                (1.0 - t) as f32 // (0,+1) → (2,−1) 线性
            })
            .collect();
        let curves = ClipExportCurves {
            frame_period_ms: 5.0,
            pitch_offset: Some(offsets),
            volume: None,
            pan: None,
        };
        let env = build_take_pitch_envelope(&curves, 2.0, 0.5).expect("envelope");
        assert_eq!(env.env_type, "PITCHENV");
        assert_eq!(env.points.len(), 2, "线性段只留端点");
        // u = t × rate：t=0 → u=0；t=2 → u=1。
        assert!((env.points[0][0]).abs() < 1e-9);
        assert!((env.points[1][0] - 1.0).abs() < 1e-9);
        // 值：+1 / ≈−1（末帧 1.995s hold，误差 ≤ 容差量级）。
        assert!((env.points[0][1] - 1.0).abs() < 1e-6);
        assert!((env.points[1][1] + 1.0).abs() < 0.02);
        // DEFSHAPE 显示范围 = max(3, ceil(max|offset|))。
        assert_eq!(env.def_shape.as_ref().map(|s| s[1]), Some(3.0));
    }

    #[test]
    fn pitch_env_values_are_not_clamped() {
        // REAPER PIT 包络值无上下限：平坦 +30 原样写出。
        let curves = ClipExportCurves {
            frame_period_ms: 5.0,
            pitch_offset: Some(pitch_values(&[30.0; 400])),
            volume: None,
            pan: None,
        };
        let env = build_take_pitch_envelope(&curves, 2.0, 1.0).expect("envelope");
        assert_eq!(env.points.len(), 2);
        assert!((env.points[0][1] - 30.0).abs() < 1e-6, "值不钳制");
        assert!((env.points[1][1] - 30.0).abs() < 1e-6);
        // DEFSHAPE 范围随值放大（30 > 3）。
        assert_eq!(env.def_shape.as_ref().map(|s| s[1]), Some(30.0));
    }

    #[test]
    fn pitch_env_skipped_when_all_default() {
        // 全 0 偏移（= 全部"不做修正"）→ 不写包络。
        let curves = ClipExportCurves {
            frame_period_ms: 5.0,
            pitch_offset: Some(pitch_values(&[0.0; 400])),
            volume: None,
            pan: None,
        };
        assert!(build_take_pitch_envelope(&curves, 2.0, 1.0).is_none());
    }

    #[test]
    fn pitch_env_keeps_zero_frames_unbridged() {
        // 首帧原始音高为 0（无声）→ 偏移 0 显式落点，不得桥接为邻近值 +2
        //（旧实现把无声区污染成 +2）。
        let mut values = vec![0.0f32; 400];
        for slot in values.iter_mut().skip(1) {
            *slot = 2.0;
        }
        let curves = ClipExportCurves {
            frame_period_ms: 5.0,
            pitch_offset: Some(values),
            volume: None,
            pan: None,
        };
        let env = build_take_pitch_envelope(&curves, 2.0, 1.0).expect("envelope");
        assert!(
            (env.points[0][1]).abs() < 1e-9,
            "端点 u=0 处偏移必须为 0，实际 {}",
            env.points[0][1]
        );
        // 随后显式升到 +2 并保持到段末。
        assert!((env.points[1][1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn pitch_env_u_domain_independent_of_clip_position() {
        // Take 包络为 u 域（媒体时间），与 clip 的工程位置无关：
        // clip 在 45s 的导出必须与 clip 在 0s 完全一致（PT 0..len×rate），
        // 端点强制落在 item 起止（不得被工程位置污染，也不得依赖 hold）。
        let make_curves = |values: Vec<f32>| ClipExportCurves {
            frame_period_ms: 5.0,
            pitch_offset: Some(values),
            volume: None,
            pan: None,
        };
        let env_at_0 = build_take_pitch_envelope(&make_curves(pitch_values(&[1.0; 400])), 2.0, 1.0)
            .expect("envelope at 0");
        let env_at_45 =
            build_take_pitch_envelope(&make_curves(pitch_values(&[1.0; 400])), 2.0, 1.0)
                .expect("envelope at 45");
        assert_eq!(env_at_0.points, env_at_45.points, "u 域与工程位置无关");
        assert!((env_at_45.points[0][0]).abs() < 1e-9, "起点 = item 起点 u=0");
        assert!(
            (env_at_45.points[1][0] - 2.0).abs() < 1e-9,
            "终点 = item 终点 u=len×rate"
        );
    }

    #[test]
    fn envseg_seg_range_uses_project_absolute_positions() {
        // 原生 REAPER 剪贴板语义（样例 ClipboardData/20260908-025849，双时基
        // 多轨形态）：SEG_RANGE = [起秒, 终秒, 起QN, 终QN]（与 item POSITION
        // 同基），PT 位置与第 8 字段 QN 均为**工程绝对值**。混用相对 PT 会让
        // REAPER 按绝对解释时把包络点落在工程起点附近（非零位置 clip 的
        // 包络整体错位）。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Abs45".to_string()),
            Some(45.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![0.8; 400]),
                pan: None,
            },
        );

        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);
        let item = &parsed.tracks[0].items[0];
        assert!((item.position - 45.0).abs() < 1e-9, "item POSITION 绝对");
        assert!(
            item.position_qn.map(|qn| (qn - 90.0).abs() < 1e-9).unwrap_or(false),
            "双时基 POSITION QN（bpm 120 → 45s = 90 QN）"
        );
        let seg = &parsed.tracks[0].envelopes[0];
        let range = seg.seg_range.as_ref().unwrap();
        assert!(
            (range[0] - 45.0).abs() < 1e-9,
            "SEG_RANGE[0] 必须为工程绝对位置 45，实际 {}",
            range[0]
        );
        assert!(
            (range[1] - 47.0).abs() < 1e-9,
            "SEG_RANGE[1] 为**终点**（45+2=47），实际 {}",
            range[1]
        );
        // QN 字段同基（bpm 120 → 2 QN/s）：起 90，终 94。
        assert!((range[2] - 90.0).abs() < 1e-9);
        assert!((range[3] - 94.0).abs() < 1e-9);
        // PT 绝对：45 → 47（含第 8 字段绝对 QN 90 → 94）。
        assert!((seg.points[0][0] - 45.0).abs() < 1e-9);
        assert!((seg.points[1][0] - 47.0).abs() < 1e-9);
        assert!((seg.points[0][7] - 90.0).abs() < 1e-9);
        assert!((seg.points[1][7] - 94.0).abs() < 1e-9);
    }

    #[test]
    fn envseg_export_writes_seg_range_and_dual_timebase_points() {
        // volume 曲线平坦 0.8（推子 1.0）→ VOLENV2 段 2 点；
        // bpm 120 → QN = 秒 × 2。段范围 = clip 跨度 [0, 2]。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("EnvSeg".to_string()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![0.8; 400]),
                pan: None,
            },
        );

        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);
        let track = &parsed.tracks[0];
        assert_eq!(track.envelopes.len(), 1, "一条 VOLENV2 段");
        let seg = &track.envelopes[0];
        assert_eq!(seg.env_type, "VOLENV2");
        // SEG_RANGE = [start, len, start_qn, len_qn]，bpm 120 → qn = ×2。
        let range = seg.seg_range.as_ref().unwrap();
        assert!(range[0].abs() < 1e-9);
        assert!((range[1] - 2.0).abs() < 1e-9);
        assert!((range[3] - 4.0).abs() < 1e-9);
        // 平坦非默认 → 恰 2 端点；8 值双时基 PT；值 = 推子 × 曲线 = 0.8。
        assert_eq!(seg.points.len(), 2);
        assert_eq!(seg.points[0].len(), 8);
        assert!((seg.points[0][1] - 0.8).abs() < 1e-6);
        assert!((seg.points[1][1] - 0.8).abs() < 1e-6);
        assert!((seg.points[1][0] - 2.0).abs() < 1e-6);
        assert!((seg.points[1][7] - 4.0).abs() < 1e-6, "末位 QN 值");
    }

    #[test]
    fn envelope_round_trip_aligns_first_position_to_paste_cursor() {
        // 导出 clip @45s → 以光标 10s 粘贴：导入端把"数据首个时间位置"
        // （min item POSITION = 45）对齐到光标（time_offset = 10 − 45 = −35），
        // item 与 ENVSEG 段整体平移 → 包络落在 [10, 12]，与 item 同步。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Offset45".into()),
            Some(45.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: Some(pitch_values(&[1.0; 400])),
                volume: Some(vec![0.8; 400]),
                pan: None,
            },
        );
        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);
        // 导出侧：绝对位置。
        assert!((parsed.tracks[0].items[0].position - 45.0).abs() < 1e-9);

        let result = crate::reaper_import::test_import_items_for_round_trip(parsed, 10.0)
            .expect("re-import");
        let clip = &result.timeline.clips[0];
        assert!(
            (clip.start_sec - 10.0).abs() < 1e-6,
            "首个时间位置对齐光标（45 → 10），实际 {}",
            clip.start_sec
        );
        let params = result.timeline.params_by_root_track.values().next().unwrap();
        let fp = 0.005f64;
        let volume = params.extra_curves.get("volume").expect("volume");
        assert!((volume[(10.0 / fp) as usize] - 0.8).abs() < 1e-3, "包络随 item 平移");
        assert!((volume[(12.0 / fp) as usize] - 0.8).abs() < 1e-3);
        assert!(
            (volume[(5.0 / fp) as usize] - 1.0).abs() < 1e-6,
            "段外保持默认"
        );
        let pending = params.pending_pitch_offset.as_ref().expect("pending");
        assert!((pending[(10.0 / fp) as usize] - 1.0).abs() < 1e-3, "音高包络随 item 平移");
    }

    #[test]
    fn envseg_qn_fields_follow_tempo_map() {
        // 变速工程：0-10s @ 120 BPM（2 QN/s），10s 起 60 BPM（1 QN/s）。
        // 段 [10, 12]：start_qn = 20；span_qn = QN(12) − QN(10) = 2
        //（按常量 BPM 线性换算会得 4 —— 变速下节拍网格错位的来源）。
        let mut timeline = TimelineState::default();
        timeline.bpm = 120.0;
        timeline.tempo_map = Some(vec![
            crate::state::TempoPointData {
                id: "tp0".to_string(),
                position_sec: 0.0,
                bpm: 120.0,
                numerator: None,
                denominator: None,
                scale: None,
            },
            crate::state::TempoPointData {
                id: "tp1".to_string(),
                position_sec: 10.0,
                bpm: 60.0,
                numerator: None,
                denominator: None,
                scale: None,
            },
        ]);
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Tempo".into()),
            Some(10.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![0.8; 400]),
                pan: None,
            },
        );

        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);
        let range = parsed.tracks[0].envelopes[0]
            .seg_range
            .as_ref()
            .unwrap();
        assert!((range[0] - 10.0).abs() < 1e-9);
        assert!((range[1] - 12.0).abs() < 1e-9, "终点 = 10+2");
        assert!(
            (range[2] - 20.0).abs() < 1e-6,
            "start_qn = 前 10s 的 120 BPM 积分 = 20 QN，实际 {}",
            range[2]
        );
        assert!(
            (range[3] - 22.0).abs() < 1e-6,
            "end_qn = QN(12) = 20 + 2，实际 {}",
            range[3]
        );
        // PT 第 8 字段（绝对 QN）：20 → 22。
        let points = &parsed.tracks[0].envelopes[0].points;
        assert!((points[0][7] - 20.0).abs() < 1e-6);
        assert!((points[1][7] - 22.0).abs() < 1e-6);
    }

    #[test]
    fn envseg_multi_clip_segments_match_native_sample_layout() {
        // 对照原生样例 ClipboardData/20260908-015013（单轨 5 个 item、每 item
        // 一条 ENVSEG 段、SEG_RANGE == item POSITION、PT 段内相对、末尾单个
        // TRACKSKIP）：同轨两个非相邻 clip → 两条独立段，位置为工程绝对值。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_a = timeline.add_clip(
            Some(track_id.clone()),
            Some("Vocal-5".into()),
            Some(45.0),
            Some(3.824),
            Some("C:/audio/v5.wav".into()),
        );
        let clip_b = timeline.add_clip(
            Some(track_id),
            Some("Vocal-6".into()),
            Some(49.0),
            Some(3.727),
            Some("C:/audio/v6.wav".into()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_a.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![0.8; 765]),
                pan: None,
            },
        );
        curves_by_clip.insert(
            clip_b.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![0.6; 746]),
                pan: None,
            },
        );

        let export =
            build_reaper_clipboard(&timeline, &[clip_a, clip_b], &curves_by_clip).expect("export");
        let text = String::from_utf8_lossy(&export.bytes);
        assert_eq!(text.matches("<ENVSEG VOLENV2").count(), 2, "两条 ENVSEG 段");
        let parsed = parse_for_test(&export.bytes);
        assert_eq!(parsed.tracks.len(), 1, "同轨两个 clip → 同一 TRACK 块");
        let segs = &parsed.tracks[0].envelopes;
        assert_eq!(segs.len(), 2);
        // SEG_RANGE = [起秒, 终秒, …]：起点 = item 的工程绝对位置（45 / 49），
        // 终点 = 起点 + clip 长度；PT 为工程绝对秒。
        assert!((segs[0].seg_range.as_ref().unwrap()[0] - 45.0).abs() < 1e-9);
        assert!((segs[1].seg_range.as_ref().unwrap()[0] - 49.0).abs() < 1e-9);
        assert!((segs[0].points[0][0] - 45.0).abs() < 1e-9, "PT 工程绝对");
        assert!((segs[1].points[0][0] - 49.0).abs() < 1e-9, "PT 工程绝对");
        // 末尾恰好一个 TRACKSKIP。
        assert_eq!(text.matches("TRACKSKIP").count(), 1);
    }

    #[test]
    fn envseg_volume_uses_effective_track_chain_gain() {
        // 组内 clip：根轨推子 0.5 × 子轨推子 1.0 = 链式有效增益 0.5；
        // VOLENV2 绝对值 = 0.5 × 曲线 1.2 = 0.6（旧实现只取子轨 1.0 → 1.2）。
        let mut timeline = TimelineState::default();
        let root_id = timeline.tracks[0].id.clone();
        timeline.tracks[0].volume = 0.5;
        timeline.tracks.push(crate::state::Track {
            id: "child_track_t".to_string(),
            name: "Child".to_string(),
            parent_id: Some(root_id.clone()),
            order: 1,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: false,
            pitch_analysis_algo: crate::state::PitchAnalysisAlgo::default(),
            color: "#888888".to_string(),
        });
        let clip_id = timeline.add_clip(
            Some("child_track_t".to_string()),
            Some("Chain".to_string()),
            Some(0.0),
            Some(1.0),
            Some("C:/audio/a.wav".to_string()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![1.2; 200]),
                pan: None,
            },
        );
        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);
        let seg = &parsed.tracks[0].envelopes[0];
        assert!((seg.points[0][1] - 0.6).abs() < 1e-6, "链式增益 0.5 × 1.2");
    }

    #[test]
    fn envseg_volume_values_scale_with_track_fader() {
        // 推子 0.5 × 曲线 1.2 = 0.6（绝对量写入段）。
        let mut timeline = TimelineState::default();
        timeline.tracks[0].volume = 0.5;
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("Fader".to_string()),
            Some(0.0),
            Some(1.0),
            Some("C:/audio/a.wav".to_string()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: None,
                volume: Some(vec![1.2; 200]),
                pan: None,
            },
        );
        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);
        let seg = &parsed.tracks[0].envelopes[0];
        assert!((seg.points[0][1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn envelope_export_round_trips_to_import_curves() {
        // 导出（PIT + ENVSEG）→ 重解析 → 导入 → 帧域偏移/曲线还原。
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        let clip_id = timeline.add_clip(
            Some(track_id),
            Some("RoundTrip".to_string()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".to_string()),
        );
        let mut curves_by_clip = BTreeMap::new();
        curves_by_clip.insert(
            clip_id.clone(),
            ClipExportCurves {
                frame_period_ms: 5.0,
                pitch_offset: Some(pitch_values(&[1.0; 400])),
                volume: Some(vec![0.8; 400]),
                pan: None,
            },
        );
        let export =
            build_reaper_clipboard(&timeline, &[clip_id], &curves_by_clip).expect("export");
        let parsed = parse_for_test(&export.bytes);

        // 导入端（目标轨推子 1.0 → 除法不变）。
        let result = crate::reaper_import::test_import_items_for_round_trip(parsed, 0.0)
            .expect("re-import");
        let params = result
            .timeline
            .params_by_root_track
            .values()
            .next()
            .expect("params");
        // PIT +1 → pending 偏移帧 ≈ +1。
        let pending = params.pending_pitch_offset.as_ref().expect("pending");
        assert!((pending[0] - 1.0).abs() < 1e-3);
        // ENVSEG +0.8 → volume 曲线帧 ≈ 0.8（推子 1.0 → 不除）。
        let volume = params.extra_curves.get("volume").expect("volume curve");
        assert!((volume[0] - 0.8).abs() < 1e-3);
        assert!((volume[(1.0 / 0.005) as usize] - 0.8).abs() < 1e-3);
    }
}
