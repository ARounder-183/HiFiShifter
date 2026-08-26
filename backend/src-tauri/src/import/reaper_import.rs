// Reaper 工程 / 剪贴板数据转换为 HiFiShifter 工程
//
// 将 reaper_parser 解析出的 ReaperData 转换为 HiFiShifter 的 TimelineState。

use crate::audio_utils::try_read_audio_header_only;
use crate::midi_import::MidiNoteEvent;
use crate::models::PitchRange;
use crate::reaper_parser::{
    self, reaper_fade_auto_length_sec, reaper_fade_effective_length_sec,
    reaper_fade_manual_length_sec, stretch_segments_from_markers, ReaperData, ReaperEnvelope,
    ReaperItem, ReaperMidiEvent, ReaperMidiSourceData, ReaperTake, ReaperTrack,
};
use crate::state::{
    Clip, PitchAnalysisAlgo, TempoPointData, TimelineState, Track, TrackParamsState,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

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
    "#4a8fd1", "#7b6bc4", "#43a875", "#cf6f2e", "#f087b5", "#b845a5", "#f0d25e", "#d94f4a",
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
    crate::media::is_media_extension(Path::new(path))
}

/// 将 Reaper 音量倍率转换为 HiFiShifter 的 0.0–1.0 范围。
fn convert_volume(vol: f64) -> f32 {
    (vol as f32).clamp(0.0, 1.0)
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

fn derive_fades_from_item_volume_envelope(
    item: &ReaperItem,
    item_length: f64,
) -> (Option<f64>, Option<f64>) {
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

/// 计算 item 的淡入淡出，拆分为「手动淡化长度」与「自动交叉淡化长度」。
///
/// 返回值：`(manual_fade_in_sec, manual_fade_out_sec, auto_fade_in_sec, auto_fade_out_sec)`。
///
/// REAPER 的 FADEIN/FADEOUT 同时携带两个长度（见 reaper_parser 的说明）：
/// - 索引 1 = 手动淡化长度（用户手动设置、永久保留）；
/// - 索引 2 = 自动交叉淡化长度（自动标记开启时才有，通常 = 与相邻 item 的重叠量）。
/// 因此这里把二者分别解析出来：手动值写入 `fade_in_sec / fade_out_sec`，
/// 自动值写入 `auto_fade_in_sec / auto_fade_out_sec`。这样当 clip 被分开
/// （自动交叉淡化归零）后，REAPER 原来的手动淡化长度能正确恢复。
fn effective_item_fades(
    item: &ReaperItem,
    take: &ReaperTake,
    item_length: f64,
) -> (f64, f64, f64, f64) {
    let max_len = item_length.max(0.0);

    // 淡化的存在性与“来源选择”（take 优先，其次 item）按“有效长度”判定
    // （自动交叉淡化生效时用自动长度，否则用手动长度）。
    let (mut fade_in_manual, fade_in_auto);
    if reaper_fade_effective_length_sec(&take.fade_in) > 1e-9 {
        fade_in_manual = reaper_fade_manual_length_sec(&take.fade_in);
        fade_in_auto = reaper_fade_auto_length_sec(&take.fade_in);
    } else {
        fade_in_manual = reaper_fade_manual_length_sec(&item.fade_in);
        fade_in_auto = reaper_fade_auto_length_sec(&item.fade_in);
    }
    let (mut fade_out_manual, fade_out_auto);
    if reaper_fade_effective_length_sec(&take.fade_out) > 1e-9 {
        fade_out_manual = reaper_fade_manual_length_sec(&take.fade_out);
        fade_out_auto = reaper_fade_auto_length_sec(&take.fade_out);
    } else {
        fade_out_manual = reaper_fade_manual_length_sec(&item.fade_out);
        fade_out_auto = reaper_fade_auto_length_sec(&item.fade_out);
    }

    // 显式 FADEIN/FADEOUT 完全缺失（手动与自动都为 0）时才从音量包络推导；
    // 包络推导出的淡化没有 REAPER 的自动标记，一律作为手动淡化。
    let (env_fade_in, env_fade_out) =
        derive_fades_from_item_volume_envelope(item, item_length.max(0.0));
    if fade_in_manual <= 1e-9 && fade_in_auto <= 1e-9 {
        if let Some(v) = env_fade_in {
            fade_in_manual = v.clamp(0.0, max_len.max(0.0));
        }
    }
    if fade_out_manual <= 1e-9 && fade_out_auto <= 1e-9 {
        if let Some(v) = env_fade_out {
            fade_out_manual = v.clamp(0.0, max_len.max(0.0));
        }
    }

    (
        fade_in_manual.clamp(0.0, max_len),
        fade_out_manual.clamp(0.0, max_len),
        fade_in_auto.clamp(0.0, max_len),
        fade_out_auto.clamp(0.0, max_len),
    )
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
        // 兼容 REAPER 左延伸 item（负 SOFFS = 前导静音）：正放无 SECTION 时
        // 保留原始符号，仅在媒体时长已知时对上界钳制。引擎侧以
        // pre-silence / Loop 回绕锚点两种方式都原生支持负 source_start。
        take.s_offs.clamp(-1_000_000.0, max_bound)
    };

    let mut anchor = if has_section || is_reversed {
        primary_anchor.clamp(min_bound, max_bound)
    } else {
        // 正放无 SECTION：下界允许为负（前导静音），只做有限性兜底。
        if primary_anchor.is_finite() {
            primary_anchor
        } else {
            0.0
        }
    };

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

fn reaper_take_volume(values: &[f64]) -> f64 {
    // Item 默认 take：VOLPAN <trim> <pan> <volume> <pan law> → volume = [2]。
    // 显式 take：TAKEVOLPAN <pan> <volume> <pan law> → volume = [1]。
    // 兜底候选同样跳过 index 0（trim/pan 不是增益）：显式 take 的
    // [pan=0.5, vol=0, law] 若回退到 index 0 会把 pan 当成 0.5 倍增益。
    let candidates: &[usize] = if values.len() >= 4 { &[2, 1] } else { &[1] };
    for &idx in candidates {
        if let Some(v) = values.get(idx).copied() {
            if v.is_finite() && v > 0.0 {
                return v;
            }
        }
    }
    1.0
}

fn take_linear_gain(item: &ReaperItem, take: &ReaperTake) -> f64 {
    let vol = reaper_take_volume(&take.vol_pan);
    if vol > 0.0 {
        return vol;
    }

    // 兼容部分 Reaper 多 Take 工程：非主 take 的 TAKEVOLPAN 可能写成 0，
    // 但实际可听音量继承自主 take。此处仅对“显式 take”做回退。
    let explicit_take = item
        .takes
        .iter()
        .any(|candidate| std::ptr::eq(candidate, take));
    if !explicit_take {
        return vol.max(0.0);
    }

    let fallback = reaper_take_volume(&item.default_take.vol_pan);
    if fallback > 0.0 {
        fallback
    } else {
        vol.max(0.0)
    }
}

/// 返回 REAPER Item 中 active take 在 `[default_take] + takes` 列表中的下标。
/// 与 `ReaperItem::active_take()` 的选择规则保持一致。
fn reaper_active_take_index(item: &ReaperItem) -> usize {
    for (idx, take) in item.takes.iter().enumerate() {
        if take.selected {
            return idx + 1;
        }
    }
    if item.default_take.source.is_some() {
        return 0;
    }
    for (idx, take) in item.takes.iter().enumerate() {
        if take.source.is_some() {
            return idx + 1;
        }
    }
    0
}

/// 把单个 REAPER 音频 Take 转换为 HiFiShifter 的 `ClipTake`。
///
/// 即使源文件缺失/不支持也会返回带路径的 Take（切换到时渲染为静音并进入
/// missing_files 提示），从而保持 REAPER 的 take 数量与顺序。
fn build_audio_clip_take(
    item: &ReaperItem,
    take: &ReaperTake,
    base_dir: Option<&Path>,
    skipped_files: &mut Vec<String>,
) -> crate::state::ClipTake {
    let item_loop = if item.has_loop_token {
        item.is_loop
    } else {
        crate::config::loop_new_clips_default()
    };
    let raw_play_rate = take.play_rate.first().copied().unwrap_or(1.0);
    let source_section_reversed = take
        .source
        .as_ref()
        .map(|src| src.section_mode > 0)
        .unwrap_or(false);
    let item_reversed = raw_play_rate < 0.0 || source_section_reversed;
    let play_rate = raw_play_rate.abs().max(0.01);

    let raw_path = take
        .source
        .as_ref()
        .map(|src| src.resolved_path().to_string())
        .unwrap_or_default();
    let audio_path = resolve_path(&raw_path, base_dir);

    let mut duration_sec = None;
    let mut duration_frames = None;
    let mut source_sample_rate = None;
    let mut pitch_range = None;
    if !raw_path.is_empty() && Path::new(&audio_path).exists() {
        if !is_audio_supported(&audio_path) {
            skipped_files.push(raw_path.clone());
        } else if let Some(info) = try_read_audio_header_only(Path::new(&audio_path)) {
            duration_sec = Some(info.duration_sec);
            duration_frames = Some(info.total_frames);
            source_sample_rate = Some(info.sample_rate);
            pitch_range = Some(PitchRange {
                min: -24.0,
                max: 24.0,
            });
        } else {
            skipped_files.push(raw_path.clone());
        }
    } else if !raw_path.is_empty() {
        skipped_files.push(raw_path.clone());
    }

    let (mut source_start, mut source_end) = compute_item_source_window_sec(
        take,
        item.length.max(0.0) * play_rate,
        duration_sec,
        item_reversed,
        item_loop,
    );
    // 兜底：与 flat/active 导入路径一致 —— 窗口被 SECTION/SOFFS 组合裁成
    // 零长度时回退到基于锚点的正向区间，避免 inactive take 切过去即静音。
    if source_end - source_start <= 1e-9 {
        let consumed = item.length.max(0.0) * play_rate;
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

    let name = if take.name.trim().is_empty() {
        clip_name_from_path(&audio_path)
    } else {
        take.name.clone()
    };

    crate::state::ClipTake {
        id: new_clip_id().replace("clip_", "take_"),
        name,
        gain: convert_volume(take_linear_gain(item, take)),
        source_path: if raw_path.is_empty() {
            None
        } else {
            Some(audio_path)
        },
        source_path_relative: None,
        duration_sec,
        duration_frames,
        source_sample_rate,
        source_file_fingerprint: None,
        source_file_mtime: None,
        source_file_size: None,
        waveform_preview: None,
        pitch_range,
        source_start_sec: source_start,
        source_end_sec: source_end,
        playback_rate: (play_rate as f32).clamp(0.1, 10.0),
        reversed: item_reversed,
        loop_enabled: item_loop,
        midi_note_data: None,
        midi_fill_gaps: false,
        stretch_markers: Vec::new(),
        envelopes: None,
    }
}

fn compute_item_source_window_sec(
    take: &ReaperTake,
    consumed_sec: f64,
    source_duration_sec: Option<f64>,
    is_reversed: bool,
    is_loop: bool,
) -> (f64, f64) {
    let (min_bound, max_bound, has_section) =
        compute_take_source_bounds_sec(take, source_duration_sec);

    let consumed = consumed_sec.max(0.0);
    let anchor =
        compute_take_source_anchor_sec(take, min_bound, max_bound, has_section, is_reversed);

    if is_loop {
        // LOOP=1（循环源，REAPER "Loop source" 语义）：
        // 源窗口取"锚点 → 可用边界"的完整区间（正向到媒体/SECTION 末尾，
        // 反向回退到区间起点），不被 ITEM LENGTH 钳制 —— 超出窗口的播放
        // 时间由引擎/渲染按该窗口周期回绕产生循环内容。
        if is_reversed {
            let start = min_bound.min(anchor);
            return (start, anchor);
        }
        let end = if max_bound.is_finite() {
            max_bound.max(anchor)
        } else {
            // 媒体时长未知时退化为按消耗量截取（无回绕素材可用）。
            anchor + consumed
        };
        return (anchor, end);
    }

    if is_reversed {
        let end = anchor;
        let start = (end - consumed).max(min_bound).min(end);
        (start, end)
    } else {
        // 非 Loop 正放：派生窗口 —— 终点 = 起点 + 消耗量，**不**按媒体时长
        // 钳制。REAPER 中 LENGTH 大于可用源的 item 其超出部分为静音尾巴，
        // 导入必须保真（渲染管线自行把越界区间处理为静音；派生模型下
        // source_end 与 length 保持一致，Slip/拖边的语义才成立）。
        let start = anchor;
        let end = start + consumed;
        (start, end)
    }
}

pub struct ReaperImportResult {
    pub timeline: TimelineState,
    pub skipped_files: Vec<String>,
    pub beats_per_bar: u32,
    /// 由工程 TEMPO / TEMPOENVEX 数据构建的 Tempo Map（None = 无实际变化）。
    pub tempo_map: Option<Vec<TempoPointData>>,
}

/// 导入 Reaper 工程文件（.rpp）。
pub fn import_rpp(path: &Path) -> Result<ReaperImportResult, String> {
    let data = reaper_parser::parse_rpp_file(path)?;
    let rpp_dir = path.parent().unwrap_or_else(|| Path::new("."));
    convert_reaper_data(data, Some(rpp_dir), 120.0)
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
    project_bpm: f64,
) -> Result<ReaperImportResult, String> {
    let reaper_data = reaper_parser::parse_clipboard_bytes(data)?;
    convert_reaper_data_clipboard(
        reaper_data,
        playhead_sec,
        selected_track_idx,
        ordered_track_ids,
        project_bpm,
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
    project_bpm: f64,
) -> Result<ReaperImportResult, String> {
    if data.is_track_data {
        // 有 Track 信息，创建新轨道；clipboard 数据可能无 TEMPO 行，传入 fallback BPM
        convert_reaper_data(data, None, project_bpm)
    } else {
        // 纯 Item（可能含 TRACKSKIP）：粘贴到现有轨道，偏移到光标
        convert_reaper_items_to_existing_tracks(
            data,
            playhead_sec,
            selected_track_idx,
            ordered_track_ids,
            project_bpm,
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
    project_bpm: f64,
) -> Result<ReaperImportResult, String> {
    let mut skipped_files: Vec<String> = Vec::new();
    let mut clips: Vec<Clip> = Vec::new();
    let mut reaper_group_map: HashMap<i32, Vec<String>> = HashMap::new();
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
                project_bpm,
                &mut reaper_group_map,
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
                    has_pitch_adjustment_active: false,
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

    let mut timeline = TimelineState {
        tracks: new_tracks,
        clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm: 120.0,
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        project_scale_notes: vec![0, 2, 4, 5, 7, 9, 11],
        tempo_map: None,
        next_track_order: next_order,
        disabled_group_ids: HashSet::new(),
    };
    timeline.normalize_clip_takes();

    // 将相同 Reaper GROUP 编号的 clip 编组
    for clip_ids in reaper_group_map.values() {
        timeline.group_clips(clip_ids);
    }

    Ok(ReaperImportResult {
        timeline,
        skipped_files,
        beats_per_bar: data.tempo.as_ref().map(|t| t.beats_per_bar).unwrap_or(4),
        tempo_map: None,
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

/// 将 REAPER 工程的 TEMPO 行 + TEMPOENVEX 包络转换为 HiFiShifter Tempo Map。
///
/// - 点位置为秒（时间锚定，与 REAPER 存储一致）；
/// - 线性渐变段（shape=0）采样为若干阶梯点，近似保留拍数；
/// - 拍号来自 TEMPO 行（初始）与包络点第 4 个值（slowcurv 打包）；
/// - REAPER 无工程调号概念，音阶全部为“跟随工程音阶”（None）；
/// - 仅当存在 0 之后的实际变化时返回 Some。
fn build_tempo_map_from_reaper(
    data: &ReaperData,
    fallback_bpm: f64,
) -> Option<Vec<TempoPointData>> {
    let initial = data.tempo.as_ref();
    let initial_bpm = initial
        .map(|t| t.bpm)
        .unwrap_or(fallback_bpm)
        .clamp(10.0, 960.0);
    let initial_numerator = initial.map(|t| t.beats_per_bar).unwrap_or(4).clamp(1, 32);
    let initial_denominator = initial
        .map(|t| {
            if matches!(t.beat_note, 1 | 2 | 4 | 8 | 16 | 32) {
                t.beat_note
            } else {
                4
            }
        })
        .unwrap_or(4);

    let mut points: Vec<TempoPointData> = Vec::new();
    let mut push = |position_sec: f64, bpm: f64, numerator: u32, denominator: u32| {
        let position_sec = position_sec.max(0.0);
        if let Some(last) = points.last_mut() {
            if (last.position_sec - position_sec).abs() < 1e-6 {
                last.bpm = bpm;
                last.numerator = Some(numerator);
                last.denominator = Some(denominator);
                return;
            }
        }
        points.push(TempoPointData {
            id: format!("reaper_tp_{}", points.len()),
            position_sec,
            bpm,
            numerator: Some(numerator),
            denominator: Some(denominator),
            scale: None,
        });
    };

    push(0.0, initial_bpm, initial_numerator, initial_denominator);

    let Some(envelope) = data.tempo_envelope.as_ref() else {
        return None;
    };
    if envelope.points.is_empty() {
        return None;
    }

    let mut cur_numerator = initial_numerator;
    let mut cur_denominator = initial_denominator;

    for (i, pt) in envelope.points.iter().enumerate() {
        let bpm = pt.bpm.clamp(10.0, 960.0);
        if let Some(num) = pt.numerator {
            cur_numerator = num.clamp(1, 32);
        }
        if let Some(den) = pt.denominator {
            cur_denominator = den;
        }
        push(pt.position_sec, bpm, cur_numerator, cur_denominator);

        // 线性渐变（shape=0）：到下一个点的速度渐变，采样为阶梯点近似。
        if pt.shape == 0 {
            if let Some(next) = envelope.points.get(i + 1) {
                let next_bpm = next.bpm.clamp(10.0, 960.0);
                if (next_bpm - bpm).abs() > 0.5 && next.position_sec > pt.position_sec + 1e-6 {
                    const SAMPLES: usize = 4;
                    for k in 1..SAMPLES {
                        let t = pt.position_sec
                            + (next.position_sec - pt.position_sec) * (k as f64 / SAMPLES as f64);
                        let bpm_k = bpm + (next_bpm - bpm) * (k as f64 / SAMPLES as f64);
                        push(t, bpm_k, cur_numerator, cur_denominator);
                    }
                }
            }
        }
    }

    if points.len() > 1 {
        Some(points)
    } else {
        None
    }
}

/// 将含有 Track 信息的 Reaper 数据转换为完整 TimelineState。
fn convert_reaper_data(
    data: ReaperData,
    base_dir: Option<&Path>,
    fallback_bpm: f64,
) -> Result<ReaperImportResult, String> {
    let mut hs_tracks: Vec<Track> = Vec::new();
    let mut hs_clips: Vec<Clip> = Vec::new();
    let mut skipped_files: Vec<String> = Vec::new();
    let mut track_order: i32 = 0;
    let mut reaper_group_map: HashMap<i32, Vec<String>> = HashMap::new();

    // track_id → pitch accumulator
    let mut pitch_data_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();

    // 从解析的 TEMPO 中获取 BPM（无则用 fallback），后续 MIDI 转换需要
    let bpm = data.tempo.as_ref().map(|t| t.bpm).unwrap_or(fallback_bpm);
    // 由 TEMPO / TEMPOENVEX 构建 Tempo Map（有实际变化时才返回 Some）。
    let tempo_map = build_tempo_map_from_reaper(&data, fallback_bpm);

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
                bpm,
                &mut reaper_group_map,
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
    let total_frames = ((project_end * 1000.0 / frame_period_ms).ceil() as usize).max(1);

    for track in &hs_tracks {
        if let Some(points) = pitch_data_by_track.get(&track.id) {
            if points.is_empty() {
                continue;
            }
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
                        has_pitch_adjustment_active: false,
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

    // REAPER 轨道 pan（VOLPAN 第二个值）导入为共通声像曲线；
    // 音量（VOLPAN 第一个值）已经作为 Track.volume 导入，无需重复。
    for (i, reaper_track) in data.tracks.iter().enumerate() {
        let pan = reaper_track.vol_pan.get(1).copied().unwrap_or(0.0);
        if !(pan.is_finite() && pan.abs() > 1e-9) {
            continue;
        }
        let Some(track_id) = track_ids.get(i) else {
            continue;
        };
        let entry = params_by_root_track
            .entry(track_id.clone())
            .or_insert_with(|| TrackParamsState {
                frame_period_ms,
                ..TrackParamsState::default()
            });
        let pan_curve = entry
            .extra_curves
            .entry("pan".to_string())
            .or_insert_with(|| vec![0.0f32; total_frames]);
        pan_curve.resize(total_frames, 0.0);
        pan_curve.fill(pan.clamp(-1.0, 1.0) as f32);
    }

    let mut timeline = TimelineState {
        tracks: hs_tracks,
        clips: hs_clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm,
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        project_scale_notes: vec![0, 2, 4, 5, 7, 9, 11],
        tempo_map: tempo_map.clone(),
        next_track_order: track_order,
        disabled_group_ids: HashSet::new(),
    };
    timeline.normalize_clip_takes();

    // 将相同 Reaper GROUP 编号的 clip 编组
    for clip_ids in reaper_group_map.values() {
        timeline.group_clips(clip_ids);
    }

    Ok(ReaperImportResult {
        timeline,
        skipped_files,
        beats_per_bar: data.tempo.as_ref().map(|t| t.beats_per_bar).unwrap_or(4),
        tempo_map,
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
    project_bpm: f64,
    reaper_group_map: &mut HashMap<i32, Vec<String>>,
) {
    let take = item.active_take();

    // 检查 MIDI 源
    if let Some(ref src) = take.source {
        if src.source_type.eq_ignore_ascii_case("MIDI") {
            // 混合 take 的 item（MIDI active + 音频 inactive）当前只导入
            // active take：显式记录这一限制，避免静默丢数据无人察觉。
            if !item.takes.is_empty() {
                eprintln!(
                    "reaper_import: item at {} has {} non-active take(s) dropped (mixed MIDI/audio takes are not fully supported yet)",
                    item.position, item.takes.len()
                );
            }
            if let Some(ref midi_data) = src.midi_source {
                process_midi_item(
                    item,
                    take,
                    track_id,
                    time_offset,
                    midi_data,
                    project_bpm,
                    clips,
                    reaper_group_map,
                );
            }
            return; // MIDI item 已处理或跳过（空 MIDI）
        }
    }

    // 获取音频文件路径
    let raw_path = match &take.source {
        Some(src) => src.resolved_path().to_string(),
        None => return,
    };
    if raw_path.is_empty() {
        return;
    }

    // 如果使用相对路径且有 base_dir，拼接成绝对路径
    let audio_path = resolve_path(&raw_path, base_dir);

    // 检查格式支持；不支持的 take 仍保留在 Clip 中（切换后渲染为静音），
    // 与 REAPER 的静音/空 take 语义一致。
    let source_readable =
        !raw_path.is_empty() && is_audio_supported(&audio_path) && Path::new(&audio_path).exists();

    // 读取音频文件信息
    // 只读 header/codec params 获取时长与采样率，不生成 waveform_preview（避免全量解码）。
    // 波形数据由前端按需通过当前 waveform API 懒加载。
    let audio_info = if source_readable {
        try_read_audio_header_only(Path::new(&audio_path))
    } else {
        None
    };
    if !source_readable || audio_info.is_none() {
        skipped_files.push(raw_path.clone());
    }
    let duration_sec = audio_info.as_ref().map(|info| info.duration_sec);
    let duration_frames = audio_info.as_ref().map(|info| info.total_frames);
    let source_sr = audio_info.as_ref().map(|info| info.sample_rate);

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
    // LOOP 标记：REAPER 显式写出时以其为准；缺失（极老工程/第三方生成器）
    // 时回退到"为新的音频块启用循环"设置。
    let item_loop = if item.has_loop_token {
        item.is_loop
    } else {
        crate::config::loop_new_clips_default()
    };
    let s_offs = take.s_offs; // source offset (seconds)
    let item_pos = item.position; // timeline position (seconds)
    let item_length = item.length; // visible length (seconds)
    let (manual_fade_in_sec, manual_fade_out_sec, auto_fade_in_sec, auto_fade_out_sec) =
        effective_item_fades(item, take, item_length.max(0.0));
    let fade_in_curve = if reaper_fade_effective_length_sec(&take.fade_in) > 1e-9 {
        reaper_fade_curve(&take.fade_in)
    } else {
        reaper_fade_curve(&item.fade_in)
    };
    let fade_out_curve = if reaper_fade_effective_length_sec(&take.fade_out) > 1e-9 {
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
        // v4 边界：拆段路径按 active take 展开成多个单 take 段 Clip，
        // 其余 take 无法随之拆分、被静默丢弃 —— 显式告警避免无人察觉。
        if !item.takes.is_empty() {
            eprintln!(
                "reaper_import: item at {} has {} non-active take(s) dropped (stretch-marker items import the active take only)",
                item.position, item.takes.len()
            );
        }
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
                // 正放：允许段起点为负（REAPER 左延伸 item = 前导静音），
                // 仅对上界与有限性做钳制；下界不再强制 ≥0。
                let raw_start =
                    (s_offs + cumulative_source_pos - actual_pre_src).min(source_max_bound);
                let start = if raw_start.is_finite() {
                    raw_start
                } else {
                    0.0
                };
                let raw_end =
                    s_offs + cumulative_source_pos + seg_source_duration + actual_post_src;
                // 派生窗口：终点不按媒体时长钳制（超出部分 = 静音尾巴）。
                let clamped_end = if raw_end.is_finite() { raw_end } else { start };
                let end = clamped_end.max(start);
                // 整段完全落在媒体起点之前的病态 item：回退为非负窗口，
                // 避免零长度/全负窗口。
                if end - start <= 1e-9 {
                    let fallback_start = start.max(0.0);
                    (fallback_start, end.max(fallback_start))
                } else {
                    (start, end)
                }
            };
            let clip_start = current_timeline_pos - actual_pre_tl;
            let clip_length = (seg_timeline_duration + actual_pre_tl + actual_post_tl).max(0.001);

            let clip_name = clip_name_from_path(&audio_path);
            let clip_id = new_clip_id();
            let clip_index = clips.len();

            clips.push(Clip {
                takes: vec![],
                active_take_id: None,
                id: clip_id.clone(),
                group_id: None,
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
                source_file_mtime: None,
                source_file_size: None,
                source_file_fingerprint: None,
                waveform_preview: None,
                pitch_range: Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                }),
                gain: convert_volume(take_gain),
                muted: item_muted,
                // 兼容 REAPER 左延伸 item：负 SOFFS 保留为前导静音
                // （引擎/离线渲染/音高分析均原生支持负 source_start_sec）。
                source_start_sec: clip_src_start,
                source_end_sec: clip_src_end,
                playback_rate: (effective_rate as f32).clamp(0.1, 10.0),
                clip_playback_rate: 1.0,
                reversed: item_reversed,
                loop_enabled: item_loop,
                // REAPER SNAPOFFS = 相对 item 起点的偏移（项目时间轴秒）。
                // 拉伸分段只落在第一段；越界钳制到段长。
                snap_offset_sec: if seg_idx == 0 {
                    item.snap_offs.max(0.0).min(clip_length.max(0.0))
                } else {
                    0.0
                },
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_curve: "sine".to_string(),
                fade_out_curve: "sine".to_string(),
                auto_fade_in_sec: 0.0,
                auto_fade_out_sec: 0.0,
                extra_curves: None,
                extra_params: None,
                formant_morph: None,
                midi_note_data: None,
                midi_fill_gaps: false,
            });
            if let Some(gid) = item.group_id {
                reaper_group_map.entry(gid).or_default().push(clip_id);
            }
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
                manual_fade_in_sec.min(clip.length_sec.max(0.0))
            };
            let fade_out_sec = if seg_idx + 1 < seg_count {
                (segment_actual_post_tl[seg_idx] + segment_actual_pre_tl[seg_idx + 1])
                    .min(clip.length_sec.max(0.0))
            } else {
                manual_fade_out_sec.min(clip.length_sec.max(0.0))
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

            // item 自身首/尾缘淡化：手动长度写入 fade_*（来自 REAPER 索引 1），
            // 自动交叉淡化长度写入 auto_fade_*（来自 REAPER 索引 2，仅该侧有自动标记）。
            // 段间合成淡化保持手动（auto 保持 0）。
            if seg_idx == 0 {
                clip.auto_fade_in_sec = auto_fade_in_sec.min(clip.length_sec.max(0.0));
            }
            if seg_idx + 1 == seg_count {
                clip.auto_fade_out_sec = auto_fade_out_sec.min(clip.length_sec.max(0.0));
            }
        }
    } else {
        // 无 stretch markers：使用 take 的 play_rate
        let effective_rate = play_rate;
        let (mut source_start, mut source_end) = compute_item_source_window_sec(
            take,
            item_length * effective_rate,
            duration_sec,
            item_reversed,
            item_loop,
        );

        // 兜底：若窗口被裁成零长度，回退到基于 SOFFS 的正向区间，避免导入后静音。
        if source_end - source_start <= 1e-9 {
            let consumed = item_length * effective_rate;
            let (min_bound, max_bound, has_section) =
                compute_take_source_bounds_sec(take, duration_sec);
            let anchor = compute_take_source_anchor_sec(
                take,
                min_bound,
                max_bound,
                has_section,
                item_reversed,
            );
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
        // 构造全部 take（default + 显式 TAKE 块），active take 由选择规则决定。
        let mut hs_takes: Vec<crate::state::ClipTake> = Vec::new();
        for reaper_take in std::iter::once(&item.default_take).chain(item.takes.iter()) {
            hs_takes.push(build_audio_clip_take(
                item,
                reaper_take,
                base_dir,
                skipped_files,
            ));
        }
        if hs_takes.is_empty() {
            hs_takes.push(build_audio_clip_take(item, take, base_dir, skipped_files));
        }
        let active_take_idx = reaper_active_take_index(item).min(hs_takes.len() - 1);
        let active_take_id = Some(hs_takes[active_take_idx].id.clone());
        let clip_name = if hs_takes[active_take_idx].name.trim().is_empty() {
            clip_name_from_path(&audio_path)
        } else {
            hs_takes[active_take_idx].name.clone()
        };
        let clip_id = new_clip_id();
        let clip_start = item_pos + time_offset;

        let mut clip = Clip {
            takes: hs_takes,
            active_take_id,
            id: clip_id.clone(),
            group_id: None,
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
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: Some(PitchRange {
                min: -24.0,
                max: 24.0,
            }),
            gain: convert_volume(take_gain),
            muted: item_muted,
            // 兼容 REAPER 左延伸 item：负 SOFFS 保留为前导静音
            // （引擎/离线渲染/音高分析均原生支持负 source_start_sec）。
            source_start_sec: source_start,
            source_end_sec: source_end,
            playback_rate: (effective_rate as f32).clamp(0.1, 10.0),
            clip_playback_rate: 1.0,
            reversed: item_reversed,
            loop_enabled: item_loop,
            // REAPER SNAPOFFS：相对 item 起点的偏移，钳制到 Clip 长度。
            snap_offset_sec: item.snap_offs.max(0.0).min(item_length.max(0.0)),
            // 手动淡化长度写入 fade_*（REAPER 索引 1），自动交叉淡化长度写入
            // auto_fade_*（REAPER 索引 2）。分开后自动值归零、手动值正确恢复。
            fade_in_sec: manual_fade_in_sec,
            fade_out_sec: manual_fade_out_sec,
            fade_in_curve,
            fade_out_curve,
            auto_fade_in_sec,
            auto_fade_out_sec,
            extra_curves: None,
            extra_params: None,
            formant_morph: None,
            midi_note_data: None,
            midi_fill_gaps: false,
        };
        // active take 的投影以现有计算（含零窗口兜底）为准写回。
        clip.sync_take_from_flat();
        clips.push(clip);

        if let Some(gid) = item.group_id {
            reaper_group_map.entry(gid).or_default().push(clip_id);
        }

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

// ─── MIDI 转换辅助函数 ───

fn midi_ticks_to_seconds(ticks: u64, ticks_per_qn: u32, bpm: f64) -> f64 {
    if ticks_per_qn == 0 || bpm <= 0.0 {
        return 0.0;
    }
    ticks as f64 / ticks_per_qn as f64 * (60.0 / bpm)
}

fn resolve_midi_bpm(midi_source: &ReaperMidiSourceData, project_bpm: f64) -> f64 {
    if let Some(ref igntempo) = midi_source.igntempo {
        if igntempo.ignore_project {
            return igntempo.tempo.max(1.0);
        }
    }
    project_bpm.max(1.0)
}

fn reaper_midi_events_to_notes(
    events: &[ReaperMidiEvent],
    ticks_per_qn: u32,
    bpm: f64,
) -> Vec<MidiNoteEvent> {
    let mut notes: Vec<MidiNoteEvent> = Vec::new();
    // Key: (channel << 7) | note_number, Value: (cumulative_tick_start, velocity)
    let mut active: std::collections::HashMap<u16, (u64, u8)> = std::collections::HashMap::new();
    let mut cumulative_ticks: u64 = 0;

    for event in events {
        cumulative_ticks += event.tick_offset;
        let channel = event.status & 0x0F;
        let msg_type = event.status & 0xF0;

        match msg_type {
            0x90 => {
                // Note On
                let note = event.data1;
                let velocity = event.data2;
                let key = ((channel as u16) << 7) | (note as u16);

                if velocity == 0 {
                    // Note On with velocity 0 = Note Off
                    if let Some((start_tick, start_vel)) = active.remove(&key) {
                        let start_sec = midi_ticks_to_seconds(start_tick, ticks_per_qn, bpm);
                        let end_sec = midi_ticks_to_seconds(cumulative_ticks, ticks_per_qn, bpm);
                        notes.push(MidiNoteEvent {
                            start_sec,
                            end_sec,
                            note: note as f32,
                            velocity: start_vel,
                            channel,
                        });
                    }
                } else {
                    // Note On: 如果已有同键活跃音符则先关闭
                    if let Some((start_tick, start_vel)) = active.remove(&key) {
                        let start_sec = midi_ticks_to_seconds(start_tick, ticks_per_qn, bpm);
                        let end_sec = midi_ticks_to_seconds(cumulative_ticks, ticks_per_qn, bpm);
                        notes.push(MidiNoteEvent {
                            start_sec,
                            end_sec,
                            note: note as f32,
                            velocity: start_vel,
                            channel,
                        });
                    }
                    active.insert(key, (cumulative_ticks, velocity));
                }
            }
            0x80 => {
                // Note Off
                let note = event.data1;
                let key = ((channel as u16) << 7) | (note as u16);
                if let Some((start_tick, start_vel)) = active.remove(&key) {
                    let start_sec = midi_ticks_to_seconds(start_tick, ticks_per_qn, bpm);
                    let end_sec = midi_ticks_to_seconds(cumulative_ticks, ticks_per_qn, bpm);
                    notes.push(MidiNoteEvent {
                        start_sec,
                        end_sec,
                        note: note as f32,
                        velocity: start_vel,
                        channel,
                    });
                }
            }
            _ => {
                // CC, pitch bend, program change 等暂不处理
            }
        }
    }

    // 关闭仍然活跃的音符
    let remaining: Vec<(u16, u64, u8)> = active
        .into_iter()
        .map(|(k, (tick, vel))| (k, tick, vel))
        .collect();
    for (key, start_tick, velocity) in remaining {
        let note = (key & 0x7F) as u8;
        let channel = (key >> 7) as u8;
        let start_sec = midi_ticks_to_seconds(start_tick, ticks_per_qn, bpm);
        let end_sec = midi_ticks_to_seconds(cumulative_ticks, ticks_per_qn, bpm);
        notes.push(MidiNoteEvent {
            start_sec,
            end_sec,
            note: note as f32,
            velocity,
            channel,
        });
    }

    notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    notes
}

fn process_midi_item(
    item: &ReaperItem,
    take: &ReaperTake,
    track_id: &str,
    time_offset: f64,
    midi_source: &ReaperMidiSourceData,
    project_bpm: f64,
    clips: &mut Vec<Clip>,
    reaper_group_map: &mut HashMap<i32, Vec<String>>,
) {
    let bpm = resolve_midi_bpm(midi_source, project_bpm);

    let mut notes = reaper_midi_events_to_notes(&midi_source.events, midi_source.ticks_per_qn, bpm);

    if notes.is_empty() {
        return;
    }

    // 应用 SOFFS 和 PLAYRATE
    let soffs = take.s_offs.max(0.0);
    let raw_play_rate = take.play_rate.first().copied().unwrap_or(1.0);
    let play_rate = raw_play_rate.abs().max(0.01);

    for note in &mut notes {
        note.start_sec = note.start_sec - soffs;
        note.end_sec = note.end_sec - soffs;
    }

    let item_length = item.length.max(0.0);

    // 过滤掉完全在窗口外的音符（使用源时间窗口）
    notes.retain(|n| n.end_sec > 0.0 && n.start_sec < item_length * play_rate);

    if notes.is_empty() {
        return;
    }

    // 归一化使最早音符的起始时间为 clip-relative 0
    let first_start = notes
        .iter()
        .map(|n| n.start_sec)
        .fold(f64::INFINITY, f64::min);
    for note in &mut notes {
        note.start_sec -= first_start;
        note.end_sec -= first_start;
    }

    let min_note = notes.iter().fold(127.0f32, |m, n| m.min(n.note));
    let max_note = notes.iter().fold(0.0f32, |m, n| m.max(n.note));

    let item_muted = item.mute.first().copied().unwrap_or(0) != 0;
    let take_gain = take_linear_gain(item, take);
    let clip_start = item.position + time_offset;

    let clip_id = new_clip_id();
    let clip_name = if take.name.is_empty() {
        let short_id = clip_id.strip_prefix("clip_").unwrap_or(&clip_id);
        format!("MIDI {}", short_id)
    } else {
        take.name.clone()
    };

    clips.push(Clip {
        takes: vec![],
        active_take_id: None,
        id: clip_id.clone(),
        group_id: None,
        track_id: track_id.to_string(),
        name: clip_name,
        start_sec: clip_start,
        length_sec: item_length.max(0.1),
        color: "cyan".to_string(),
        source_path: None,
        source_path_relative: None,
        duration_sec: None,
        duration_frames: None,
        source_sample_rate: None,
        source_file_mtime: None,
        source_file_size: None,
        source_file_fingerprint: None,
        waveform_preview: None,
        pitch_range: Some(PitchRange {
            min: min_note,
            max: max_note,
        }),
        gain: convert_volume(take_gain),
        muted: item_muted,
        source_start_sec: 0.0,
        source_end_sec: item_length * play_rate,
        playback_rate: play_rate as f32,
        clip_playback_rate: 1.0,
        reversed: false,
        // MIDI item 没有源媒体可循环；Loop 属性保持关闭。
        loop_enabled: false,
        // REAPER SNAPOFFS：相对 item 起点的偏移，钳制到 Clip 长度。
        snap_offset_sec: item.snap_offs.max(0.0).min(item_length.max(0.0)),
        fade_in_sec: 0.0,
        fade_out_sec: 0.0,
        fade_in_curve: "sine".to_string(),
        fade_out_curve: "sine".to_string(),
        auto_fade_in_sec: 0.0,
        auto_fade_out_sec: 0.0,
        extra_curves: None,
        extra_params: None,
        formant_morph: None,
        midi_note_data: Some(notes),
        midi_fill_gaps: false,
    });

    if let Some(gid) = item.group_id {
        reaper_group_map.entry(gid).or_default().push(clip_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaper_parser::ReaperSource;

    #[test]
    fn loop_source_window_extends_to_media_end() {
        // 正向 LOOP：窗口 = [SOFFS, 媒体末尾]，不被 LENGTH 钳制。
        let take = ReaperTake {
            s_offs: 1.0,
            ..ReaperTake::default()
        };
        let (start, end) = compute_item_source_window_sec(&take, 10.0, Some(4.0), false, true);
        assert!((start - 1.0).abs() < 1e-9);
        assert!((end - 4.0).abs() < 1e-9);

        // 非 LOOP：窗口被消耗量钳制。
        let (start2, end2) = compute_item_source_window_sec(&take, 1.5, Some(4.0), false, false);
        assert!((end2 - start2 - 1.5).abs() < 1e-9);

        // 反向 LOOP：窗口 = [区间起点, 锚点]，锚点 = max_bound − SOFFS。
        let (start3, end3) = compute_item_source_window_sec(&take, 10.0, Some(4.0), true, true);
        assert!(start3.abs() < 1e-9);
        assert!((end3 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn reaper_take_volume_reads_explicit_takevolpan_layout() {
        let item = ReaperItem::default();
        // 显式 take 的 TAKEVOLPAN：<pan> <volume> <pan law>。
        let explicit = ReaperTake {
            vol_pan: vec![0.0, 1.25, -1.0],
            ..ReaperTake::default()
        };
        assert!((reaper_take_volume(&explicit.vol_pan) - 1.25).abs() < 1e-9);
        assert!((take_linear_gain(&item, &explicit) - 1.25).abs() < 1e-9);

        // Item 默认 take 的 VOLPAN：<trim> <pan> <volume> <pan law>。
        let default = ReaperTake {
            vol_pan: vec![1.0, 0.0, 0.8, -1.0],
            ..ReaperTake::default()
        };
        assert!((reaper_take_volume(&default.vol_pan) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn loop_window_falls_back_to_consumed_when_duration_unknown() {
        let take = ReaperTake {
            s_offs: 2.0,
            ..ReaperTake::default()
        };
        let (start, end) = compute_item_source_window_sec(&take, 3.0, None, false, true);
        assert!((start - 2.0).abs() < 1e-9);
        assert!((end - 5.0).abs() < 1e-9);
    }

    #[test]
    fn multi_take_item_imports_all_takes_and_active_selection() {
        let mut item = ReaperItem {
            position: 0.0,
            length: 1.0,
            has_loop_token: true,
            is_loop: false,
            default_take: ReaperTake {
                name: "Default".to_string(),
                s_offs: 0.0,
                source: None,
                ..ReaperTake::default()
            },
            ..ReaperItem::default()
        };
        fn wave_source(path: &str) -> ReaperSource {
            let mut src = ReaperSource::new();
            src.source_type = "WAVE".to_string();
            src.file_path = path.to_string();
            src
        }
        item.default_take.source = Some(wave_source("C:/missing/default.wav"));

        let alt = ReaperTake {
            selected: true,
            name: "Alt".to_string(),
            s_offs: 2.0,
            source: Some(wave_source("C:/missing/alt.wav")),
            ..ReaperTake::default()
        };
        item.takes.push(alt.clone());

        let mut clips = Vec::new();
        let mut skipped = Vec::new();
        let mut pitch = Vec::new();
        let mut groups = HashMap::new();
        process_item(
            &item,
            "track_1",
            None,
            0.0,
            &mut clips,
            &mut skipped,
            &mut pitch,
            120.0,
            &mut groups,
        );

        assert_eq!(clips.len(), 1, "one item -> one clip");
        let clip = &clips[0];
        assert_eq!(clip.takes.len(), 2, "default + explicit take");
        assert_eq!(clip.takes[1].id, clip.active_take_id.as_deref().unwrap());
        assert_eq!(clip.takes[1].name, "Alt");
        assert_eq!(clip.takes[1].source_start_sec, 2.0);
        assert!(!skipped.is_empty(), "missing sources are reported");
    }
}
