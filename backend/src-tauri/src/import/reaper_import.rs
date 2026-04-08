// Reaper 工程 / 剪贴板数据转换为 HiFiShifter 工程
//
// 将 reaper_parser 解析出的 ReaperData 转换为 HiFiShifter 的 TimelineState。

use crate::audio_utils::try_read_audio_header_only;
use crate::models::PitchRange;
use crate::reaper_parser::{
    self, stretch_segments_from_markers, ReaperData, ReaperEnvelope, ReaperItem, ReaperTake,
    ReaperTrack,
};
use crate::state::{Clip, PitchAnalysisAlgo, TimelineState, Track, TrackParamsState};
use std::collections::BTreeMap;
use std::path::Path;

/// HiFiShifter 支持的音频格式扩展名
const SUPPORTED_AUDIO_EXTS: &[&str] = &["wav", "flac", "mp3", "ogg", "m4a"];

/// 帧周期（秒）
const FRAME_PERIOD: f64 = 0.005;

/// 分段重叠上限（秒）
const SEGMENT_OVERLAP_MAX_SEC: f64 = 0.1;

/// 相邻分段过渡长度：取两段中较短者的 50%，并限制在上限内。
fn segment_overlap_sec(left_timeline_sec: f64, right_timeline_sec: f64) -> f64 {
    left_timeline_sec
        .max(0.0)
        .min(right_timeline_sec.max(0.0))
        .mul_add(0.5, 0.0)
        .min(SEGMENT_OVERLAP_MAX_SEC * 0.5)
}

/// 轨道颜色调色板（与 state.rs / vocalshifter_import.rs 一致）
const TRACK_COLORS: &[&str] = &[
    "#4f8ef7", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#38bdf8", "#facc15", "#f87171",
];

fn clip_color() -> String {
    "#4fc3f7".to_string()
}

fn new_track_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn new_clip_id() -> String {
    format!("clip_{}", uuid::Uuid::new_v4())
}

fn is_audio_supported(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            SUPPORTED_AUDIO_EXTS
                .iter()
                .any(|&ext| ext.eq_ignore_ascii_case(e))
        })
        .unwrap_or(false)
}

/// 将 Reaper 音量倍率转换为 HiFiShifter 的 0.0–1.0 范围。
fn convert_volume(vol: f64) -> f32 {
    (vol as f32).clamp(0.0, 1.0)
}

fn reaper_fade_length_sec(values: &[f64]) -> f64 {
    if values.len() >= 2 {
        values[1].max(0.0)
    } else {
        values.first().copied().unwrap_or(0.0).max(0.0)
    }
}

fn reaper_fade_curve(values: &[f64]) -> String {
    let shape = values.first().copied().unwrap_or(0.0).round() as i32;
    match shape {
        0 => "linear",
        1 => "sine",
        2 => "exponential",
        3 => "logarithmic",
        4 => "scurve",
        5 => "exponential",
        6 => "logarithmic",
        _ => "sine",
    }
    .to_string()
}

fn derive_fades_from_item_volume_envelope(item: &ReaperItem, item_length: f64) -> (Option<f64>, Option<f64>) {
    let mut points: Vec<(f64, f64)> = item
        .envelopes
        .iter()
        .filter(|env| {
            let t = env.env_type.to_uppercase();
            env.act.first().copied().unwrap_or(1) != 0 && (t.contains("VOLENV") || t == "VOLENV")
        })
        .flat_map(|env| {
            env.points.iter().filter_map(|pt| {
                if pt.len() >= 2 {
                    let t = pt[0];
                    let v = pt[1];
                    if t.is_finite() && v.is_finite() {
                        Some((t.clamp(0.0, item_length.max(0.0)), v))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        })
        .collect();

    if points.len() < 2 || item_length <= 0.0 {
        return (None, None);
    }

    points.sort_by(|a, b| a.0.total_cmp(&b.0));

    let peak = points
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() || peak <= 0.0 {
        return (None, None);
    }

    // Reaper item volume envelope does not always plateau at exactly 1.0
    // (e.g. when item/take gain is already attenuated). Use a relative peak.
    let unity_threshold = peak * 0.98;
    let edge_sec = item_length.mul_add(0.05, 0.0).max(0.05);

    let first = points.first().copied();
    let last = points.last().copied();

    let fade_in = first.and_then(|(t0, v0)| {
        if t0 <= edge_sec && v0 < unity_threshold {
            points
                .iter()
                .find(|(t, v)| *t > t0 && *v >= unity_threshold)
                .map(|(t, _)| t.clamp(0.0, item_length))
        } else {
            None
        }
    });

    let fade_out = last.and_then(|(t1, v1)| {
        if item_length - t1 <= edge_sec && v1 < unity_threshold {
            points
                .iter()
                .rev()
                .find(|(t, v)| *t < t1 && *v >= unity_threshold)
                .map(|(t, _)| (item_length - *t).clamp(0.0, item_length))
        } else {
            None
        }
    });

    (fade_in, fade_out)
}

fn effective_item_fades(item: &ReaperItem, take: &ReaperTake, item_length: f64) -> (f64, f64) {
    let max_len = item_length.max(0.0);
    let take_fade_in = reaper_fade_length_sec(&take.fade_in);
    let take_fade_out = reaper_fade_length_sec(&take.fade_out);

    let mut fade_in_sec = if take_fade_in > 1e-9 {
        take_fade_in.clamp(0.0, max_len)
    } else {
        reaper_fade_length_sec(&item.fade_in).clamp(0.0, max_len)
    };
    let mut fade_out_sec = if take_fade_out > 1e-9 {
        take_fade_out.clamp(0.0, max_len)
    } else {
        reaper_fade_length_sec(&item.fade_out).clamp(0.0, max_len)
    };
    let (env_fade_in, env_fade_out) = derive_fades_from_item_volume_envelope(item, item_length.max(0.0));

    // 仅在显式 fade 长度缺失时才从音量包络推导，避免覆盖 Reaper 的直接 FADEIN/FADEOUT。
    if fade_in_sec <= 1e-9 {
        if let Some(v) = env_fade_in {
            fade_in_sec = v.clamp(0.0, item_length.max(0.0));
        }
    }
    if fade_out_sec <= 1e-9 {
        if let Some(v) = env_fade_out {
            fade_out_sec = v.clamp(0.0, item_length.max(0.0));
        }
    }

    (fade_in_sec, fade_out_sec)
}

fn compute_take_source_bounds_sec(
    take: &ReaperTake,
    source_duration_sec: Option<f64>,
) -> (f64, f64, bool) {
    let section_start = take
        .source
        .as_ref()
        .and_then(|src| src.section_start_sec)
        .unwrap_or(0.0);
    let section_length = take
        .source
        .as_ref()
        .and_then(|src| src.section_length_sec)
        .filter(|len| len.is_finite() && *len > 0.0);

    let mut min_bound = 0.0;
    let mut max_bound = f64::INFINITY;
    let has_section = take
        .source
        .as_ref()
        .and_then(|src| src.section_start_sec)
        .is_some();

    if has_section {
        min_bound = section_start.max(0.0);
        if let Some(section_len) = section_length {
            max_bound = (section_start + section_len).max(min_bound);
        }
    }

    if let Some(total_sec) = source_duration_sec.filter(|v| v.is_finite() && *v > 0.0) {
        max_bound = max_bound.min(total_sec);
    }

    (min_bound, max_bound, has_section)
}

fn compute_take_source_anchor_sec(
    take: &ReaperTake,
    min_bound: f64,
    max_bound: f64,
    has_section: bool,
    is_reversed: bool,
) -> f64 {
    let section_start = take
        .source
        .as_ref()
        .and_then(|src| src.section_start_sec)
        .unwrap_or(0.0);
    let soffs_nonneg = take.s_offs.max(0.0);

    let primary_anchor = if has_section {
        if is_reversed {
            max_bound - soffs_nonneg
        } else {
            section_start + soffs_nonneg
        }
    } else if is_reversed {
        if max_bound.is_finite() {
            max_bound - soffs_nonneg
        } else {
            take.s_offs
        }
    } else {
        take.s_offs
    };

    let mut anchor = primary_anchor.clamp(min_bound, max_bound);

    if has_section {
        // 兼容部分工程里 SOFFS 已经是绝对源坐标的写法。
        let alt_anchor = soffs_nonneg.clamp(min_bound, max_bound);
        let primary_span = if is_reversed {
            anchor - min_bound
        } else {
            max_bound - anchor
        };
        let alt_span = if is_reversed {
            alt_anchor - min_bound
        } else {
            max_bound - alt_anchor
        };
        if alt_span > primary_span {
            anchor = alt_anchor;
        }
    }

    anchor
}

fn take_linear_gain(item: &ReaperItem, take: &ReaperTake) -> f64 {
    let vol = take
        .vol_pan
        .first()
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(1.0);
    if vol > 0.0 {
        return vol;
    }

    // 兼容部分 Reaper 多 Take 工程：非主 take 的 VOLPAN 可能写成 0，
    // 但实际可听音量继承自主 take。此处仅对“显式 take”做回退。
    let explicit_take = item.takes.iter().any(|candidate| std::ptr::eq(candidate, take));
    if !explicit_take {
        return vol.max(0.0);
    }

    let fallback = item
        .default_take
        .vol_pan
        .first()
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(1.0);
    if fallback > 0.0 {
        fallback
    } else {
        vol.max(0.0)
    }
}

fn compute_item_source_window_sec(
    take: &ReaperTake,
    consumed_sec: f64,
    source_duration_sec: Option<f64>,
    is_reversed: bool,
) -> (f64, f64) {
    let (min_bound, max_bound, has_section) =
        compute_take_source_bounds_sec(take, source_duration_sec);

    let consumed = consumed_sec.max(0.0);
    let anchor = compute_take_source_anchor_sec(take, min_bound, max_bound, has_section, is_reversed);

    if is_reversed {
        let end = anchor;
        let start = (end - consumed).max(min_bound).min(end);
        (start, end)
    } else {
        let start = anchor;
        let end = (start + consumed).min(max_bound).max(start);
        (start, end)
    }
}

pub struct ReaperImportResult {
    pub timeline: TimelineState,
    pub skipped_files: Vec<String>,
    pub beats_per_bar: u32,
}

/// 导入 Reaper 工程文件（.rpp）。
pub fn import_rpp(path: &Path) -> Result<ReaperImportResult, String> {
    let data = reaper_parser::parse_rpp_file(path)?;
    let rpp_dir = path.parent().unwrap_or_else(|| Path::new("."));
    convert_reaper_data(data, Some(rpp_dir))
}

/// 导入 Reaper 剪贴板数据。
///
/// - `playhead_sec`: 当前光标位置
/// - `selected_track_idx`: 用户选中的轨道在 `ordered_track_ids` 中的下标
/// - `ordered_track_ids`: 按 order 排序的现有轨道 ID 列表
pub fn import_reaper_clipboard(
    data: &[u8],
    playhead_sec: f64,
    selected_track_idx: usize,
    ordered_track_ids: &[String],
) -> Result<ReaperImportResult, String> {
    let reaper_data = reaper_parser::parse_clipboard_bytes(data)?;
    convert_reaper_data_clipboard(
        reaper_data,
        playhead_sec,
        selected_track_idx,
        ordered_track_ids,
    )
}

/// 剪贴板导入逻辑：
/// - 有 Track 块：创建新轨道（.rpp 完整工程方式）
/// - 纯 Item 数据（含 TRACKSKIP）：粘贴到选中轨道及其下方现有轨道，偏移到光标位置
fn convert_reaper_data_clipboard(
    data: ReaperData,
    playhead_sec: f64,
    selected_track_idx: usize,
    ordered_track_ids: &[String],
) -> Result<ReaperImportResult, String> {
    if data.is_track_data {
        // 有 Track 信息，创建新轨道
        convert_reaper_data(data, None)
    } else {
        // 纯 Item（可能含 TRACKSKIP）：粘贴到现有轨道，偏移到光标
        convert_reaper_items_to_existing_tracks(
            data,
            playhead_sec,
            selected_track_idx,
            ordered_track_ids,
        )
    }
}

/// 将纯 Item 剪贴板数据粘贴到现有轨道。
///
/// - 首个音频块的开始位置对齐到光标
/// - TRACKSKIP 的 offset 用于映射到 selected_track 下方的现有轨道
fn convert_reaper_items_to_existing_tracks(
    data: ReaperData,
    playhead_sec: f64,
    selected_track_idx: usize,
    ordered_track_ids: &[String],
) -> Result<ReaperImportResult, String> {
    let mut skipped_files: Vec<String> = Vec::new();
    let mut clips: Vec<Clip> = Vec::new();
    let mut new_tracks: Vec<Track> = Vec::new();
    // 新建轨道映射：target_track_idx → track_id
    let mut created_track_ids: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();
    // track_id → pitch offset accumulator
    let mut pitch_offset_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();

    // 当前已有轨道的最大 order，用于分配新轨道 order
    let mut next_order = ordered_track_ids.len() as i32;

    // 计算所有 item 中最小的 position，用于 offset 到 playhead
    let min_position = data
        .tracks
        .iter()
        .flat_map(|t| t.items.iter())
        .map(|item| item.position)
        .fold(f64::MAX, f64::min);
    let time_offset = if min_position.is_finite() {
        playhead_sec - min_position
    } else {
        0.0
    };

    for (track_idx, reaper_track) in data.tracks.iter().enumerate() {
        // 查找此 Reaper track 对应的 HiFiShifter 轨道
        let track_offset = data
            .track_offsets
            .get(track_idx)
            .copied()
            .unwrap_or(track_idx);
        let target_track_idx = selected_track_idx + track_offset;
        let target_track_id = if target_track_idx < ordered_track_ids.len() {
            ordered_track_ids[target_track_idx].clone()
        } else if let Some(id) = created_track_ids.get(&target_track_idx) {
            // 已经为此下标创建过轨道
            id.clone()
        } else {
            // 超出现有轨道范围，创建新轨道
            let tid = new_track_id();
            let color_idx = (ordered_track_ids.len() + new_tracks.len()) % TRACK_COLORS.len();
            new_tracks.push(Track {
                id: tid.clone(),
                name: format!("Track {}", next_order + 1),
                parent_id: None,
                order: next_order,
                muted: false,
                solo: false,
                volume: 1.0,
                compose_enabled: false,
                pitch_analysis_algo: PitchAnalysisAlgo::default(),
                color: TRACK_COLORS[color_idx].to_string(),
            });
            created_track_ids.insert(target_track_idx, tid.clone());
            next_order += 1;
            tid
        };

        let track_pitch_accum = pitch_offset_by_track
            .entry(target_track_id.clone())
            .or_default();

        for item in &reaper_track.items {
            process_item(
                item,
                &target_track_id,
                None, // no base dir for clipboard
                time_offset,
                &mut clips,
                &mut skipped_files,
                track_pitch_accum,
            );
        }
    }

    // 构建待应用的 pitch 偏移数据
    let project_end = clips
        .iter()
        .map(|c| c.start_sec + c.length_sec)
        .fold(32.0_f64, f64::max);
    let frame_period_ms = FRAME_PERIOD * 1000.0;
    let total_frames = ((project_end * 1000.0 / frame_period_ms).ceil() as usize).max(1);

    let mut params_by_root_track: BTreeMap<String, TrackParamsState> = BTreeMap::new();
    for (track_id, accum) in &pitch_offset_by_track {
        if accum.is_empty() || track_id.is_empty() {
            continue;
        }
        let offset_frames = build_pitch_frames(accum, total_frames);
        // 只在有非零偏移时才记录
        if offset_frames.iter().any(|&v| v.abs() > 1e-6) {
            params_by_root_track.insert(
                track_id.clone(),
                TrackParamsState {
                    frame_period_ms,
                    pitch_orig: Vec::new(),
                    pitch_edit: Vec::new(),
                    pitch_edit_user_modified: false,
                    tension_orig: Vec::new(),
                    tension_edit: Vec::new(),
                    pitch_orig_key: None,
                    pending_pitch_offset: Some(offset_frames),
                    extra_curves: Default::default(),
                    extra_params: Default::default(),
                },
            );
        }
    }

    let timeline = TimelineState {
        tracks: new_tracks,
        clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm: 120.0,
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        project_scale_notes: vec![0, 2, 4, 5, 7, 9, 11],
        next_track_order: next_order,
    };

    Ok(ReaperImportResult {
        timeline,
        skipped_files,
        beats_per_bar: data.tempo.as_ref().map(|t| t.beats_per_bar).unwrap_or(4),
    })
}

// ─── 轨道层级辅助函数 ───

/// 根据 ISBUS 字段计算每条 Reaper 轨道的深度。
///
/// 层级公式：L[0] = 0，L[i] = max(0, L[i-1] + isbus[i-1][1])
/// 其中 isbus[i][1] 是第 i 条轨道的 ISBUS 第二个数值。
fn compute_track_depths(tracks: &[ReaperTrack]) -> Vec<i32> {
    let mut depths = Vec::with_capacity(tracks.len());
    let mut current_depth: i32 = 0;
    for track in tracks {
        depths.push(current_depth);
        let delta = track.isbus.get(1).copied().unwrap_or(0);
        current_depth = (current_depth + delta).max(0);
    }
    depths
}

/// 根据深度列表和轨道 ID 列表，为每条轨道分配父轨道 ID。
///
/// 使用栈算法：当轨道深度为 D 时，弹出栈中深度 >= D 的条目，
/// 栈顶即为父轨道（深度为 D-1）。
fn compute_parent_ids(depths: &[i32], track_ids: &[String]) -> Vec<Option<String>> {
    let mut parent_ids = Vec::with_capacity(depths.len());
    // 栈中存储 (depth, track_index)
    let mut stack: Vec<(i32, usize)> = Vec::new();

    for (i, &depth) in depths.iter().enumerate() {
        // 弹出深度 >= 当前深度的元素
        while let Some(&(d, _)) = stack.last() {
            if d >= depth {
                stack.pop();
            } else {
                break;
            }
        }
        let parent_id = stack.last().map(|&(_, idx)| track_ids[idx].clone());
        parent_ids.push(parent_id);
        stack.push((depth, i));
    }
    parent_ids
}

/// 将含有 Track 信息的 Reaper 数据转换为完整 TimelineState。
fn convert_reaper_data(
    data: ReaperData,
    base_dir: Option<&Path>,
) -> Result<ReaperImportResult, String> {
    let mut hs_tracks: Vec<Track> = Vec::new();
    let mut hs_clips: Vec<Clip> = Vec::new();
    let mut skipped_files: Vec<String> = Vec::new();
    let mut track_order: i32 = 0;

    // track_id → pitch accumulator
    let mut pitch_data_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();

    // 预分配 UUID、计算深度和父子关系（两道步）
    let track_ids: Vec<String> = (0..data.tracks.len()).map(|_| new_track_id()).collect();
    let depths = compute_track_depths(&data.tracks);
    let parent_ids = compute_parent_ids(&depths, &track_ids);

    for (i, reaper_track) in data.tracks.iter().enumerate() {
        let track_id = &track_ids[i];
        let volume = if !reaper_track.vol_pan.is_empty() {
            convert_volume(reaper_track.vol_pan[0])
        } else {
            0.9
        };
        let muted = reaper_track.mute_solo.first().copied().unwrap_or(0) != 0;
        let solo = reaper_track.mute_solo.get(1).copied().unwrap_or(0) != 0;

        hs_tracks.push(Track {
            id: track_id.clone(),
            name: if reaper_track.name.is_empty() {
                format!("Track {}", track_order + 1)
            } else {
                reaper_track.name.clone()
            },
            parent_id: parent_ids[i].clone(),
            order: track_order,
            muted,
            solo,
            volume,
            compose_enabled: false,
            pitch_analysis_algo: PitchAnalysisAlgo::default(),
            color: TRACK_COLORS[hs_tracks.len() % TRACK_COLORS.len()].to_string(),
        });

        let mut track_pitch_accum: Vec<PitchFrameAccumulator> = Vec::new();

        for item in &reaper_track.items {
            process_item(
                item,
                track_id,
                base_dir,
                0.0, // .rpp 导入不做时间偏移
                &mut hs_clips,
                &mut skipped_files,
                &mut track_pitch_accum,
            );
        }

        if !track_pitch_accum.is_empty() {
            pitch_data_by_track.insert(track_id.clone(), track_pitch_accum);
        }

        track_order += 1;
    }

    // 计算工程时长
    let project_end = hs_clips
        .iter()
        .map(|c| c.start_sec + c.length_sec)
        .fold(32.0_f64, f64::max);

    // 构建 pitch 参数
    let mut params_by_root_track: BTreeMap<String, TrackParamsState> = BTreeMap::new();
    let frame_period_ms = FRAME_PERIOD * 1000.0;

    for track in &hs_tracks {
        if let Some(points) = pitch_data_by_track.get(&track.id) {
            if points.is_empty() {
                continue;
            }
            let total_frames = ((project_end * 1000.0 / frame_period_ms).ceil() as usize).max(1);
            let offset_frames = build_pitch_frames(points, total_frames);

            // 只在有非零偏移时才记录
            if offset_frames.iter().any(|&v| v.abs() > 1e-6) {
                params_by_root_track.insert(
                    track.id.clone(),
                    TrackParamsState {
                        frame_period_ms,
                        pitch_orig: Vec::new(),
                        pitch_edit: Vec::new(),
                        pitch_edit_user_modified: false,
                        tension_orig: Vec::new(),
                        tension_edit: Vec::new(),
                        pitch_orig_key: None,
                        pending_pitch_offset: Some(offset_frames),
                        extra_curves: Default::default(),
                        extra_params: Default::default(),
                    },
                );
            }
        }
    }

    // 从解析的 TEMPO 中获取 BPM（无则默认 120）
    let bpm = data.tempo.as_ref().map(|t| t.bpm).unwrap_or(120.0);

    let timeline = TimelineState {
        tracks: hs_tracks,
        clips: hs_clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm,
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        project_scale_notes: vec![0, 2, 4, 5, 7, 9, 11],
        next_track_order: track_order,
    };

    Ok(ReaperImportResult {
        timeline,
        skipped_files,
        beats_per_bar: data.tempo.as_ref().map(|t| t.beats_per_bar).unwrap_or(4),
    })
}

// ─── Item 处理 ───

#[derive(Default, Clone, Copy)]
struct PitchFrameAccumulator {
    sum: f64,
    weight: f64,
}

/// 处理一个 Reaper Item，生成一个或多个 HiFiShifter Clip。
///
/// `time_offset`: 时间偏移量（用于将剪贴板数据对齐到光标位置），.rpp 导入时为 0。
fn process_item(
    item: &ReaperItem,
    track_id: &str,
    base_dir: Option<&Path>,
    time_offset: f64,
    clips: &mut Vec<Clip>,
    skipped_files: &mut Vec<String>,
    pitch_accum: &mut Vec<PitchFrameAccumulator>,
) {
    let take = item.active_take();

    // 获取音频文件路径
    let raw_path = match &take.source {
        Some(src) => src.resolved_path().to_string(),
        None => return, // skip MIDI or empty items
    };
    if raw_path.is_empty() {
        return;
    }

    // 如果使用相对路径且有 base_dir，拼接成绝对路径
    let audio_path = resolve_path(&raw_path, base_dir);

    // 检查格式支持
    if !is_audio_supported(&audio_path) {
        skipped_files.push(raw_path);
        return;
    }

    // 检查文件存在
    if !Path::new(&audio_path).exists() {
        skipped_files.push(raw_path);
        return;
    }

    // 读取音频文件信息
    // 只读 header/codec params 获取时长与采样率，不生成 waveform_preview（避免全量解码）。
    // 波形数据由前端按需通过当前 waveform API 懒加载。
    let audio_info = try_read_audio_header_only(Path::new(&audio_path));
    let (duration_sec, duration_frames, source_sr) = match &audio_info {
        Some(info) => (
            Some(info.duration_sec),
            Some(info.total_frames),
            Some(info.sample_rate),
        ),
        None => (None, None, None),
    };

    // 获取 take 参数
    let raw_play_rate = take.play_rate.first().copied().unwrap_or(1.0);
    let source_section_reversed = take
        .source
        .as_ref()
        .map(|src| src.section_mode > 0)
        .unwrap_or(false);
    let item_reversed = raw_play_rate < 0.0 || source_section_reversed;
    let play_rate = raw_play_rate.abs().max(0.01);
    let item_pitch_semitones = take.play_rate.get(2).copied().unwrap_or(0.0); // 整体音高偏移
    let take_gain = take_linear_gain(item, take);
    let item_muted = item.mute.first().copied().unwrap_or(0) != 0;
    let s_offs = take.s_offs; // source offset (seconds)
    let item_pos = item.position; // timeline position (seconds)
    let item_length = item.length; // visible length (seconds)
    let (fade_in_sec, fade_out_sec) = effective_item_fades(item, take, item_length.max(0.0));
    let fade_in_curve = if reaper_fade_length_sec(&take.fade_in) > 1e-9 {
        reaper_fade_curve(&take.fade_in)
    } else {
        reaper_fade_curve(&item.fade_in)
    };
    let fade_out_curve = if reaper_fade_length_sec(&take.fade_out) > 1e-9 {
        reaper_fade_curve(&take.fade_out)
    } else {
        reaper_fade_curve(&item.fade_out)
    };

    // 获取音高包络（如果有）
    let pitch_envelope = find_pitch_envelope(&item.envelopes);

    // ─── 处理 Stretch Markers ───
    let segments = stretch_segments_from_markers(&item.stretch_markers);

    if !segments.is_empty() {
        // 有 stretch markers：拆分为多段
        // effective rate = segment_avg_rate * item_play_rate（源消耗速率）
        let seg_count = segments.len();
        let mut segment_clip_indices: Vec<usize> = Vec::with_capacity(seg_count);
        let mut segment_actual_pre_tl: Vec<f64> = Vec::with_capacity(seg_count);
        let mut segment_actual_post_tl: Vec<f64> = Vec::with_capacity(seg_count);
        let seg_timeline_durations: Vec<f64> = segments
            .iter()
            .map(|seg| (seg.offset_length() / play_rate).max(0.001))
            .collect();
        let mut current_timeline_pos = item_pos + time_offset;
        let mut cumulative_source_pos: f64 = 0.0;
        let (source_min_bound, source_max_bound, has_source_section) =
            compute_take_source_bounds_sec(take, duration_sec);
        let source_anchor = compute_take_source_anchor_sec(
            take,
            source_min_bound,
            source_max_bound,
            has_source_section,
            item_reversed,
        );

        for (seg_idx, seg) in segments.iter().enumerate() {
            let seg_avg_rate = seg.velocity_average().max(0.01);
            let effective_rate = seg_avg_rate * play_rate;
            let seg_timeline_duration = seg_timeline_durations[seg_idx];
            // 源消耗量 = 时间线时长 × 播放速率
            let seg_source_duration = seg_timeline_duration * effective_rate;

            // 分段重叠与淡入淡出
            let want_pre = if seg_idx > 0 {
                segment_overlap_sec(seg_timeline_durations[seg_idx - 1], seg_timeline_duration)
            } else {
                0.0
            };
            let want_post = if seg_idx + 1 < seg_count {
                segment_overlap_sec(seg_timeline_duration, seg_timeline_durations[seg_idx + 1])
            } else {
                0.0
            };
            let actual_pre_src = (want_pre * effective_rate).min(cumulative_source_pos);
            let actual_post_src = want_post * effective_rate;
            let actual_pre_tl = actual_pre_src / effective_rate;
            let actual_post_tl = actual_post_src / effective_rate;

            let (clip_src_start, clip_src_end) = if item_reversed {
                let raw_start =
                    source_anchor - cumulative_source_pos - seg_source_duration - actual_post_src;
                let raw_end = source_anchor - cumulative_source_pos + actual_pre_src;
                let start = raw_start.max(source_min_bound).min(source_max_bound);
                let end = raw_end.max(start).min(source_max_bound);
                (start, end)
            } else {
                let start = (s_offs + cumulative_source_pos - actual_pre_src)
                    .max(source_min_bound)
                    .min(source_max_bound);
                let raw_end = s_offs + cumulative_source_pos + seg_source_duration + actual_post_src;
                let end = raw_end.max(start).min(source_max_bound);
                (start, end)
            };
            let clip_start = current_timeline_pos - actual_pre_tl;
            let clip_length = (seg_timeline_duration + actual_pre_tl + actual_post_tl).max(0.001);

            let clip_name = clip_name_from_path(&audio_path);
            let clip_id = new_clip_id();
            let clip_index = clips.len();

            clips.push(Clip {
                id: clip_id.clone(),
                track_id: track_id.to_string(),
                name: if seg_count > 1 {
                    format!("{} ({})", clip_name, seg_idx + 1)
                } else {
                    clip_name
                },
                start_sec: clip_start,
                length_sec: clip_length,
                color: clip_color(),
                source_path: Some(audio_path.clone()),
                source_path_relative: None,
                duration_sec,
                duration_frames,
                source_sample_rate: source_sr,
                waveform_preview: None,
                pitch_range: Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                }),
                gain: convert_volume(take_gain),
                muted: item_muted,
                source_start_sec: clip_src_start.max(0.0),
                source_end_sec: clip_src_end,
                playback_rate: (effective_rate as f32).clamp(0.1, 10.0),
                reversed: item_reversed,
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_curve: "sine".to_string(),
                fade_out_curve: "sine".to_string(),
                extra_curves: None,
                extra_params: None,
            });
            segment_clip_indices.push(clip_index);
            segment_actual_pre_tl.push(actual_pre_tl);
            segment_actual_post_tl.push(actual_post_tl);

            // 写入 pitch 偏移数据
            write_pitch_for_clip(
                pitch_accum,
                clip_start,
                clip_length,
                clip_src_start,
                effective_rate,
                item_pitch_semitones,
                pitch_envelope.as_ref(),
                item_pos + time_offset,
                item_length,
            );

            current_timeline_pos += seg_timeline_duration;
            cumulative_source_pos += seg_source_duration;
        }

        for seg_idx in 0..seg_count {
            let clip_idx = segment_clip_indices[seg_idx];
            let Some(clip) = clips.get_mut(clip_idx) else {
                continue;
            };

            let fade_in_sec = if seg_idx > 0 {
                (segment_actual_pre_tl[seg_idx] + segment_actual_post_tl[seg_idx - 1])
                    .min(clip.length_sec.max(0.0))
            } else {
                fade_in_sec.min(clip.length_sec.max(0.0))
            };
            let fade_out_sec = if seg_idx + 1 < seg_count {
                (segment_actual_post_tl[seg_idx] + segment_actual_pre_tl[seg_idx + 1])
                    .min(clip.length_sec.max(0.0))
            } else {
                fade_out_sec.min(clip.length_sec.max(0.0))
            };

            let fade_in_curve_name = if seg_idx == 0 {
                fade_in_curve.clone()
            } else {
                "sine".to_string()
            };
            let fade_out_curve_name = if seg_idx + 1 == seg_count {
                fade_out_curve.clone()
            } else {
                "sine".to_string()
            };

            clip.fade_in_sec = fade_in_sec;
            clip.fade_out_sec = fade_out_sec;
            clip.fade_in_curve = fade_in_curve_name;
            clip.fade_out_curve = fade_out_curve_name;
        }
    } else {
        // 无 stretch markers：使用 take 的 play_rate
        let effective_rate = play_rate;
        let (mut source_start, mut source_end) = compute_item_source_window_sec(
            take,
            item_length * effective_rate,
            duration_sec,
            item_reversed,
        );

        // 兜底：若窗口被裁成零长度，回退到基于 SOFFS 的正向区间，避免导入后静音。
        if source_end - source_start <= 1e-9 {
            let consumed = item_length * effective_rate;
            let (min_bound, max_bound, has_section) =
                compute_take_source_bounds_sec(take, duration_sec);
            let anchor =
                compute_take_source_anchor_sec(take, min_bound, max_bound, has_section, item_reversed);
            let (fallback_start, fallback_end) = if item_reversed {
                let end = anchor;
                let start = (end - consumed).max(min_bound).min(end);
                (start, end)
            } else {
                let start = anchor;
                let end = (start + consumed).min(max_bound).max(start);
                (start, end)
            };
            if fallback_end - fallback_start > source_end - source_start {
                source_start = fallback_start;
                source_end = fallback_end;
            }
        }
        let clip_name = clip_name_from_path(&audio_path);
        let clip_id = new_clip_id();
        let clip_start = item_pos + time_offset;

        clips.push(Clip {
            id: clip_id.clone(),
            track_id: track_id.to_string(),
            name: clip_name,
            start_sec: clip_start,
            length_sec: item_length,
            color: clip_color(),
            source_path: Some(audio_path.clone()),
            source_path_relative: None,
            duration_sec,
            duration_frames,
            source_sample_rate: source_sr,
            waveform_preview: None,
            pitch_range: Some(PitchRange {
                min: -24.0,
                max: 24.0,
            }),
            gain: convert_volume(take_gain),
            muted: item_muted,
            source_start_sec: source_start.max(0.0),
            source_end_sec: source_end,
            playback_rate: (effective_rate as f32).clamp(0.1, 10.0),
            reversed: item_reversed,
            fade_in_sec,
            fade_out_sec,
            fade_in_curve,
            fade_out_curve,
            extra_curves: None,
            extra_params: None,
        });

        // 写入 pitch 偏移数据
        write_pitch_for_clip(
            pitch_accum,
            clip_start,
            item_length,
            source_start,
            effective_rate,
            item_pitch_semitones,
            pitch_envelope.as_ref(),
            clip_start,
            item_length,
        );
    }
}

// ─── Pitch 处理 ───

/// 在 item 的 envelopes 中查找音高包络。
/// Reaper 的音高包络类型为 "ENVSEG" 且通常是 "PITCHENV" 或以 "PITCH" 开头。
/// 也可能直接作为 item level 的 envelope 出现。
fn find_pitch_envelope(envelopes: &[ReaperEnvelope]) -> Option<Vec<(f64, f64)>> {
    for env in envelopes {
        let t = env.env_type.to_uppercase();
        // 在 item 级别的 pitch envelope 通常类型名包含 "PITCH"
        // 但 Reaper 也可能使用 ENVSEG
        if t.contains("PITCH") || t == "ENVSEG" {
            // 检查 act[0] 是否启用（默认 act=[1, -1]）
            if env.act.first().copied().unwrap_or(1) == 0 {
                continue;
            }
            let mut points = Vec::new();
            for pt in &env.points {
                if pt.len() >= 2 {
                    // pt[0] = time (seconds, relative to item start)
                    // pt[1] = value (semitones for pitch envelope, range typically -24..+24)
                    points.push((pt[0], pt[1]));
                }
            }
            if !points.is_empty() {
                return Some(points);
            }
        }
    }
    None
}

/// 在音高包络上插值取得指定时间点的值。
fn interpolate_pitch_envelope(points: &[(f64, f64)], time_sec: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // 二分查找
    let idx = points.partition_point(|p| p.0 < time_sec);

    if idx == 0 {
        return points[0].1;
    }
    if idx == points.len() {
        return points[points.len() - 1].1;
    }

    let (t0, v0) = points[idx - 1];
    let (t1, v1) = points[idx];
    let dt = t1 - t0;

    if dt.abs() < 1e-12 {
        return v0;
    }

    let t = (time_sec - t0) / dt;
    v0 + (v1 - v0) * t
}

/// 将 pitch 数据写入帧级别的 accumulator。
/// Reaper 的音高是"相对于原始"的半音偏移，要叠加到原始音高上。
/// 但由于 HiFiShifter 导入时还没有分析原始音高，这里先记录偏移量，
/// 后续在 pitch params 构建阶段会将它写入 pitch_edit。
///
/// 实现策略：由于 Reaper 的音高是偏移量（相对原始），而 HiFiShifter 的 pitch_edit 是绝对值，
/// 在导入时我们暂时记录偏移量，等 HiFiShifter 进行音高分析后会用 pitch_orig + offset 来计算。
/// 如果没有偏移（0半音），则不写入 pitch 数据，让 HiFiShifter 的后续音高分析流程来处理。
fn write_pitch_for_clip(
    accum: &mut Vec<PitchFrameAccumulator>,
    clip_start_sec: f64,
    clip_length_sec: f64,
    _source_start_sec: f64,
    _play_rate: f64,
    item_pitch_semitones: f64,
    pitch_envelope: Option<&Vec<(f64, f64)>>,
    item_start_sec: f64,
    item_length_sec: f64,
) {
    // 如果没有任何音高偏移，跳过（让 HiFiShifter 默认处理）
    let has_pitch_shift = item_pitch_semitones.abs() > 1e-6;
    let has_envelope = pitch_envelope.map(|e| !e.is_empty()).unwrap_or(false);

    if !has_pitch_shift && !has_envelope {
        return;
    }

    let clip_end_sec = clip_start_sec + clip_length_sec;
    let start_frame = (clip_start_sec / FRAME_PERIOD).floor().max(0.0) as usize;
    let end_frame = (clip_end_sec / FRAME_PERIOD).ceil().max(0.0) as usize;

    for frame_idx in start_frame..=end_frame {
        let frame_time = frame_idx as f64 * FRAME_PERIOD;
        // 相对于 item 开始的时间
        let time_in_item = frame_time - item_start_sec;

        if time_in_item < 0.0 || time_in_item > item_length_sec {
            continue;
        }

        // 计算音高偏移 = 整体偏移 + 包络偏移
        let mut pitch_offset = item_pitch_semitones;
        if let Some(env_points) = pitch_envelope {
            pitch_offset += interpolate_pitch_envelope(env_points, time_in_item);
        }

        if frame_idx >= accum.len() {
            accum.resize(frame_idx + 1, PitchFrameAccumulator::default());
        }
        let entry = &mut accum[frame_idx];
        entry.sum += pitch_offset;
        entry.weight += 1.0;
    }
}

/// 从 accumulator 构建 pitch_edit 帧数组。
/// 值是半音偏移量（会在后续音高分析后叠加到 pitch_orig 上）。
fn build_pitch_frames(accum: &[PitchFrameAccumulator], total_frames: usize) -> Vec<f32> {
    let mut frames = vec![0.0f32; total_frames];
    for (idx, acc) in accum.iter().enumerate() {
        if idx < total_frames && acc.weight > 0.0 {
            frames[idx] = (acc.sum / acc.weight) as f32;
        }
    }
    frames
}

// ─── 辅助函数 ───

fn clip_name_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Audio")
        .to_string()
}

fn resolve_path(raw_path: &str, base_dir: Option<&Path>) -> String {
    let p = Path::new(raw_path);
    if p.is_absolute() {
        return raw_path.to_string();
    }
    if let Some(dir) = base_dir {
        let resolved = dir.join(p);
        return resolved.to_string_lossy().to_string();
    }
    raw_path.to_string()
}
