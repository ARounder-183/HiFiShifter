// Reaper 工程 / 剪贴板数据转换为 HiFiShifter 工程
//
// 将 reaper_parser 解析出的 ReaperData 转换为 HiFiShifter 的 TimelineState。

use crate::audio_utils::try_read_audio_header_only;
use crate::midi_import::MidiNoteEvent;
use crate::models::PitchRange;
use crate::reaper_parser::{
    self, item_time_to_take_env_u, reaper_fade_auto_length_sec, reaper_fade_effective_length_sec,
    reaper_fade_manual_length_sec, stretch_segments_full_cover, ReaperData, ReaperEnvelope,
    ReaperItem, ReaperMidiEvent, ReaperMidiSourceData, ReaperTake, ReaperTrack,
};
use crate::state::{
    Clip, PitchAnalysisAlgo, TempoPointData, TimelineState, Track, TrackParamsState,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

/// 帧周期（秒）
const FRAME_PERIOD: f64 = 0.005;

/// RPP 数值防御上限（秒）：损坏 / 伪造数据的极端 POSITION / LENGTH /
/// 包络点位置不得把帧向量 resize 到不可分配的规模（24h ≈ 1.7e7 帧）。
/// 合法工程远低于该值；超界输入按上限处理并表现为内容截断。
const MAX_IMPORT_ITEM_SEC: f64 = 86_400.0;
const MAX_IMPORT_FRAME_IDX: usize = (MAX_IMPORT_ITEM_SEC / FRAME_PERIOD) as usize;

fn clamp_import_position(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(-MAX_IMPORT_ITEM_SEC, MAX_IMPORT_ITEM_SEC)
    } else {
        0.0
    }
}

fn clamp_import_length(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, MAX_IMPORT_ITEM_SEC)
    } else {
        0.0
    }
}

/// 帧号钳制：把任意有限秒值折进安全的帧下标范围。
fn clamp_import_frame(sec: f64) -> usize {
    if !sec.is_finite() {
        return 0;
    }
    ((sec / FRAME_PERIOD).floor().max(0.0) as usize).min(MAX_IMPORT_FRAME_IDX)
}

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

/// 轨道颜色调色板（与 state.rs / vocalshifter_import.rs 一致，灰色开头）
const TRACK_COLORS: &[&str] = &[
    "#74787e", "#4a8fd1", "#7b6bc4", "#43a875", "#cf6f2e", "#f087b5", "#b845a5", "#f0d25e",
    "#d94f4a",
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

/// 将 Reaper 的线性音量倍率转换为 HiFiShifter 的增益（0.0–4.0）。
///
/// REAPER 的音量是线性倍率（0.5 = −6 dB、1.0 = 0 dB、2.0 = +6 dB），
/// item trim 与 take volume 相乘后最高可达 +24 dB（16×）；负值表示相位
/// 反转。HiFiShifter 的 gain 支持 [0, 4]（0..+12 dB）且无反相概念：
/// 负值取绝对值保留响度，超出上限截断到 4.0。
fn convert_volume(vol: f64) -> f32 {
    (vol.abs() as f32).clamp(0.0, 4.0)
}

/// 将 REAPER fade 数组换算为（浮点形状 id, 曲率）。
///
/// 历史：旧实现把形状数字映射到内部命名曲线枚举（"sine"/"exponential"…），
/// v4 起 Clip 直接保存 REAPER 的形状/曲率原值，无需再经过命名枚举中转。
use super::reaper_parser::reaper_fade_shape_dir;

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
///
/// 注意：take 音量包络（VOLENV）不再用于"推导淡化"——包络现在被真实导入
/// 为轨道音量参数线（边缘 ramp 即曲线值，听感等价且保真更高）；旧的
/// env→fade 启发式是包络不被支持时代的补偿，已删除。淡化只来自
/// FADEIN/FADEOUT 行。
fn effective_item_fades(
    item: &ReaperItem,
    take: &ReaperTake,
    item_length: f64,
) -> (f64, f64, f64, f64) {
    let max_len = item_length.max(0.0);

    // 淡化的存在性与“来源选择”（take 优先，其次 item）按“有效长度”判定
    // （自动交叉淡化生效时用自动长度，否则用手动长度）。
    let (fade_in_manual, fade_in_auto);
    if reaper_fade_effective_length_sec(&take.fade_in) > 1e-9 {
        fade_in_manual = reaper_fade_manual_length_sec(&take.fade_in);
        fade_in_auto = reaper_fade_auto_length_sec(&take.fade_in);
    } else {
        fade_in_manual = reaper_fade_manual_length_sec(&item.fade_in);
        fade_in_auto = reaper_fade_auto_length_sec(&item.fade_in);
    }
    let (fade_out_manual, fade_out_auto);
    if reaper_fade_effective_length_sec(&take.fade_out) > 1e-9 {
        fade_out_manual = reaper_fade_manual_length_sec(&take.fade_out);
        fade_out_auto = reaper_fade_auto_length_sec(&take.fade_out);
    } else {
        fade_out_manual = reaper_fade_manual_length_sec(&item.fade_out);
        fade_out_auto = reaper_fade_auto_length_sec(&item.fade_out);
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

fn reaper_take_volume(values: &[f64], explicit_take: bool) -> f64 {
    // 布局由“默认 take vs 显式 take”决定（不能靠长度启发式）：
    // - 默认 take（ITEM 的 VOLPAN）：<item trim> <pan> <take volume> <pan law>
    //   → volume = [2]；
    // - 显式 take（TAKEVOLPAN）：<pan> <take volume> <pan law> → volume = [1]。
    // volume 可以合法地为 0（静音）、负（反相）或 >1（提升），只排除非有限值。
    let idx = if explicit_take { 1 } else { 2 };
    values
        .get(idx)
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(1.0)
}

fn take_linear_gain(item: &ReaperItem, take: &ReaperTake, take_volume_overridden: bool) -> f64 {
    // REAPER 的可听增益 = item trim（VOLPAN[0]，item 级） × take volume
    // （VOLPAN[2] / TAKEVOLPAN[1]）。只取 take volume 会丢掉用户在 item
    // 音量把手上设置的音量（REAPER 把它写进 trim）。
    //
    // take_volume_overridden：该 take 存在激活的 VOLENV——REAPER 的绝对
    // 包络语义下包络取代 take 音量钮（item trim 不受影响），因此钮值按
    // 1.0 中性化，包络本身晋升为轨道音量曲线，避免双重施加。
    let explicit_take = item
        .takes
        .iter()
        .any(|candidate| std::ptr::eq(candidate, take));
    let trim = item
        .default_take
        .vol_pan
        .first()
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(1.0);

    let vol = if take_volume_overridden {
        1.0
    } else {
        reaper_take_volume(&take.vol_pan, explicit_take)
    };
    if vol != 0.0 {
        return trim * vol;
    }

    // 兼容部分 Reaper 多 Take 工程：非主 take 的 TAKEVOLPAN 可能写成 0，
    // 但实际可听音量继承自主 take。此处仅对“显式 take”做回退。
    if !explicit_take {
        return 0.0;
    }

    let fallback = reaper_take_volume(&item.default_take.vol_pan, false);
    if fallback != 0.0 {
        trim * fallback
    } else {
        0.0
    }
}

/// 该 take 是否存在激活的音量包络（绝对语义 → take 音量钮被取代）。
fn take_volume_envelope_active(take: &ReaperTake) -> bool {
    find_active_envelope(&take.envelopes, VOL_ENV_TYPES).is_some()
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
    // 该 take 自身存在激活 VOLENV 时其音量钮被取代（各 take 独立判定）。
    let take_volume_overridden = take_volume_envelope_active(take);
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
        gain: convert_volume(take_linear_gain(item, take, take_volume_overridden)),
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
    ordered_track_volumes: &[f32],
    project_bpm: f64,
    next_track_order: i32,
) -> Result<ReaperImportResult, String> {
    let reaper_data = reaper_parser::parse_clipboard_bytes(data)?;
    convert_reaper_data_clipboard(
        reaper_data,
        playhead_sec,
        selected_track_idx,
        ordered_track_ids,
        ordered_track_volumes,
        project_bpm,
        next_track_order,
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
    ordered_track_volumes: &[f32],
    project_bpm: f64,
    next_track_order: i32,
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
            ordered_track_volumes,
            project_bpm,
            next_track_order,
        )
    }
}

/// 测试辅助（仅 test 构建）：把已解析的剪贴板数据按 item 模式导入单一目标轨
///（推子 1.0），供导出↔导入往返测试使用。
#[cfg(test)]
pub(crate) fn test_import_items_for_round_trip(
    data: ReaperData,
    playhead_sec: f64,
) -> Result<ReaperImportResult, String> {
    convert_reaper_items_to_existing_tracks(
        data,
        playhead_sec,
        0,
        &["track_1".to_string()],
        &[1.0],
        120.0,
        0,
    )
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
    ordered_track_volumes: &[f32],
    project_bpm: f64,
    next_track_order: i32,
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
    // track_id → take 包络晋升 / ENVSEG 段曲线累积器
    let mut curves_by_track: std::collections::HashMap<String, CurveAccum> =
        std::collections::HashMap::new();

    // 自动新建轨道的起始 order：order 允许稀疏，不能假设它等于轨道数
    // （调用方传入 max(next_track_order, 现有最大 order + 1)），
    // 否则新建轨道可能插进显示序列中部而非"下方扩展"。
    let mut next_order = next_track_order.max(ordered_track_ids.len() as i32);

    // 计算所有 item 中最小的 position，用于 offset 到 playhead
    // （空 item 集合 → INFINITY → 不可数 → offset 取 0；不能用 f64::MAX
    // 兜底，那会算出 ~-1.8e308 的垃圾偏移把包络段整体丢弃）。
    let min_position = data
        .tracks
        .iter()
        .flat_map(|t| t.items.iter())
        .map(|item| item.position)
        .fold(f64::INFINITY, f64::min);
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

        {
            let track_pitch_accum = pitch_offset_by_track
                .entry(target_track_id.clone())
                .or_default();
            let track_curves_accum = curves_by_track
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
                    track_curves_accum,
                    project_bpm,
                    &mut reaper_group_map,
                );
            }
        }

        // ── ENVSEG 轨道包络段（item 剪贴板携带的轨道包络） ──
        // REAPER 复制 item 时把覆盖其跨度的轨道包络写成 <ENVSEG <TYPE> +
        // SEG_RANGE 段；坐标为段内相对时间（项目位置 = seg_range[0] + PT）。
        // 附带 SEG_RANGE 的视为段；无 SEG_RANGE 的裸 ENVSEG/普通类型块按
        // 绝对工程秒解释。
        for env in &reaper_track.envelopes {
            if !envelope_is_active(env) {
                continue;
            }
            let kind = if env_type_matches(&env.env_type, VOL_ENV_TYPES) {
                Some("volume")
            } else if env_type_matches(&env.env_type, PAN_ENV_TYPES) {
                Some("pan")
            } else if env_type_matches(&env.env_type, MUTE_ENV_TYPES) {
                Some(MUTE_GATE_KEY)
            } else {
                None
            };
            let Some(key) = kind else {
                continue;
            };
            let mut points = reaper_env_points(env);
            if points.is_empty() {
                continue;
            }
            // ENVSEG 段的 VOLENV2 是绝对量；HiFiShifter 曲线相对轨道推子 →
            // 相对化除法（take 晋升曲线在 process_item 中已是相对量，不经此处）。
            if key == "volume" {
                let track_volume = ordered_track_volumes
                    .get(target_track_idx)
                    .copied()
                    .unwrap_or(1.0);
                if track_volume.abs() < 1e-6 {
                    log::warn!(
                        "reaper_import: target track volume ~0, ENVSEG volume kept absolute"
                    );
                } else {
                    points = points
                        .into_iter()
                        .map(|(p, v, s)| (p, v / track_volume as f64, s))
                        .collect();
                }
            }
            let curves = curves_by_track.entry(target_track_id.clone()).or_default();
            match env.seg_range.as_ref() {
                Some(range) => {
                    // 原生 REAPER 语义（样例 ClipboardData/20260908-025849）：
                    // SEG_RANGE = [起秒, 终秒, 起QN, 终QN]（第二字段是**终点**
                    // 而非长度），PT 为**工程绝对秒**。
                    //
                    // M2（主路径，原生形态）：seg_start > 0 且所有点 ≥ seg_start
                    // → 点为绝对工程秒，相对化平移 seg_start（段随 items 一起
                    // 平移 time_offset）。
                    //
                    // M1（回退，旧/第三方形态）：点 < seg_start → 点为段内
                    // 相对时间，段起点 = seg_range[0]。
                    let seg_start = range.first().copied().unwrap_or(0.0).max(0.0);
                    let absolute = seg_start > 1e-6
                        && points.iter().all(|(p, _, _)| *p >= seg_start - 1e-6);
                    let points = if absolute {
                        points
                            .into_iter()
                            .map(|(p, v, s)| (p - seg_start, v, s))
                            .collect()
                    } else {
                        points
                    };
                    let span_start = seg_start + time_offset;
                    // SEG_RANGE[1]：> seg_start → 终点（原生）；≤ seg_start →
                    // 长度（旧形态回退）。缺省回退到点跨度。
                    let span_end = match range.get(1).copied() {
                        Some(end) if end.is_finite() && end > seg_start + 1e-9 => {
                            end + time_offset
                        }
                        Some(len) if len.is_finite() && len > 0.0 => span_start + len,
                        _ => span_start + points.last().map(|(p, _, _)| *p).unwrap_or(0.0),
                    };
                    overwrite_accum_span(curves, key, &points, span_start, span_end, span_start);
                }
                None => {
                    // 无 SEG_RANGE：绝对工程秒（+ 光标对齐偏移）。
                    let span_start = (points.first().map(|(p, _, _)| *p).unwrap_or(0.0)
                        + time_offset)
                        .max(0.0);
                    let span_end = points.last().map(|(p, _, _)| *p).unwrap_or(0.0) + time_offset;
                    overwrite_accum_span(curves, key, &points, span_start, span_end, 0.0);
                }
            }
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
            params_by_root_track
                .entry(track_id.clone())
                .or_insert_with(|| TrackParamsState {
                    frame_period_ms,
                    ..TrackParamsState::default()
                })
                .pending_pitch_offset = Some(offset_frames);
        }
    }

    // take 包络晋升 + ENVSEG 段 → extra_curves（MUTE 门控乘入 volume）。
    for (track_id, accum) in &curves_by_track {
        let curves = build_extra_curves_from_accum(accum, total_frames);
        if curves.is_empty() {
            continue;
        }
        params_by_root_track
            .entry(track_id.clone())
            .or_insert_with(|| TrackParamsState {
                frame_period_ms,
                ..TrackParamsState::default()
            })
            .extra_curves
            .extend(curves);
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
        beats_per_bar: data
            .tempo
            .as_ref()
            .map(|t| t.beats_per_bar.clamp(1, 32))
            .unwrap_or(4),
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

    // track_id → pitch accumulator / take 包络晋升曲线累积器
    let mut pitch_data_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();
    let mut curves_by_track: std::collections::HashMap<String, CurveAccum> =
        std::collections::HashMap::new();
    // track_id → 轨道包络（vol, pan, mute gate），待 total_frames 已知后覆写
    let mut track_envs_by_track: std::collections::HashMap<
        String,
        (Option<ReaperEnvelope>, Option<ReaperEnvelope>, Option<ReaperEnvelope>),
    > = std::collections::HashMap::new();

    // 从解析的 TEMPO 中获取 BPM（无则用 fallback），后续 MIDI 转换需要
    let bpm = data.tempo.as_ref().map(|t| t.bpm).unwrap_or(fallback_bpm);
    // 由 TEMPO / TEMPOENVEX 构建 Tempo Map（有实际变化时才返回 Some）。
    let tempo_map = build_tempo_map_from_reaper(&data, fallback_bpm);

    // 预分配 UUID、计算深度和父子关系（两道步）
    let track_ids: Vec<String> = (0..data.tracks.len()).map(|_| new_track_id()).collect();
    let depths = compute_track_depths(&data.tracks);
    let parent_ids = compute_parent_ids(&depths, &track_ids);

    // 轨道包络预检：REAPER 绝对包络语义下，激活的 VOLENV2 取代轨道推子、
    // PANENV2 取代声像钮（静态 pan 常量曲线随之抑制）。
    let track_vol_envs: Vec<Option<ReaperEnvelope>> = data
        .tracks
        .iter()
        .map(|t| find_active_envelope(&t.envelopes, VOL_ENV_TYPES).cloned())
        .collect();
    let track_pan_envs: Vec<Option<ReaperEnvelope>> = data
        .tracks
        .iter()
        .map(|t| find_active_envelope(&t.envelopes, PAN_ENV_TYPES).cloned())
        .collect();
    let track_gate_envs: Vec<Option<ReaperEnvelope>> = data
        .tracks
        .iter()
        .map(|t| find_active_envelope(&t.envelopes, MUTE_ENV_TYPES).cloned())
        .collect();

    for (i, reaper_track) in data.tracks.iter().enumerate() {
        let track_id = &track_ids[i];
        // VOLENV2 激活 → 推子被包络取代，Track.volume 中性化为 1.0，
        // 包络值全量进 volume 曲线（构建阶段覆写），避免双重施加。
        let volume = if track_vol_envs[i].is_some() {
            1.0
        } else if !reaper_track.vol_pan.is_empty() {
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
        let mut track_curves_accum: CurveAccum = BTreeMap::new();

        for item in &reaper_track.items {
            process_item(
                item,
                track_id,
                base_dir,
                0.0, // .rpp 导入不做时间偏移
                &mut hs_clips,
                &mut skipped_files,
                &mut track_pitch_accum,
                &mut track_curves_accum,
                bpm,
                &mut reaper_group_map,
            );
        }

        // 轨道包络（绝对语义 + hold 全程覆盖）暂存，待 total_frames 已知后
        // 统一覆写 take 晋升数据（见下方轨道包络覆写段）。
        track_envs_by_track.insert(track_id.clone(), (track_vol_envs[i].clone(), track_pan_envs[i].clone(), track_gate_envs[i].clone()));

        if !track_pitch_accum.is_empty() {
            pitch_data_by_track.insert(track_id.clone(), track_pitch_accum);
        }
        if !track_curves_accum.is_empty() {
            curves_by_track.insert(track_id.clone(), track_curves_accum);
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

    // 轨道包络优先（绝对语义 + hold 全程覆盖）：覆写 take 晋升数据。
    // 轨道包络点是工程绝对秒，直接采样整条轨道帧域 [0, project_end]。
    for (track_id, (vol_env, pan_env, gate_env)) in &track_envs_by_track {
        if vol_env.is_none() && pan_env.is_none() && gate_env.is_none() {
            continue;
        }
        let accum = curves_by_track.entry(track_id.clone()).or_default();
        if let Some(env) = vol_env {
            overwrite_accum_span(accum, "volume", &reaper_env_points(env), 0.0, project_end, 0.0);
        }
        if let Some(env) = pan_env {
            overwrite_accum_span(accum, "pan", &reaper_env_points(env), 0.0, project_end, 0.0);
        }
        if let Some(env) = gate_env {
            overwrite_accum_span(accum, MUTE_GATE_KEY, &reaper_env_points(env), 0.0, project_end, 0.0);
        }
    }

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

    // take/轨道包络晋升曲线 → extra_curves（MUTE 门控乘入 volume）。
    for track in &hs_tracks {
        let Some(accum) = curves_by_track.get(&track.id) else {
            continue;
        };
        let curves = build_extra_curves_from_accum(accum, total_frames);
        if curves.is_empty() {
            continue;
        }
        let entry = params_by_root_track
            .entry(track.id.clone())
            .or_insert_with(|| TrackParamsState {
                frame_period_ms,
                ..TrackParamsState::default()
            });
        entry.extra_curves.extend(curves);
    }

    // REAPER 轨道 pan（VOLPAN 第二个值）导入为共通声像曲线；
    // 音量（VOLPAN 第一个值）已经作为 Track.volume 导入，无需重复。
    // PANENV2 激活时取代声像钮 → 静态 pan 常量曲线被抑制（上面已由
    // 包络覆写写入动态 pan 曲线）。
    for (i, reaper_track) in data.tracks.iter().enumerate() {
        if track_pan_envs[i].is_some() {
            continue;
        }
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
        beats_per_bar: data
            .tempo
            .as_ref()
            .map(|t| t.beats_per_bar.clamp(1, 32))
            .unwrap_or(4),
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
///
/// `pitch_accum` / `curves_accum`：本 item 活跃 take 的音高偏移与
/// VOL/PAN/MUTE 包络晋升数据（按目标轨道累积，构建阶段转成帧曲线）。
fn process_item(
    item: &ReaperItem,
    track_id: &str,
    base_dir: Option<&Path>,
    time_offset: f64,
    clips: &mut Vec<Clip>,
    skipped_files: &mut Vec<String>,
    pitch_accum: &mut Vec<PitchFrameAccumulator>,
    curves_accum: &mut BTreeMap<String, Vec<CurveFrameAccumulator>>,
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
                log::warn!(
                    "reaper_import: item at {} has {} non-active take(s) dropped (mixed MIDI/audio takes are not fully supported yet)",
                    item.position, item.takes.len()
                );
            }
            // MIDI take 的包络（v1 边界）不消费：HiFiShifter 的音高线依赖
            // 音频分析（pitch_orig），MIDI take 无音频可分析，偏移无落点。
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
    // 活跃 take 存在激活 VOLENV 时其音量钮被取代（绝对包络语义），
    // 包络本身由 write_take_envelope_frames 晋升为轨道音量曲线。
    let take_volume_overridden = take_volume_envelope_active(take);
    let take_gain = take_linear_gain(item, take, take_volume_overridden);
    let item_muted = item.mute.first().copied().unwrap_or(0) != 0;
    // LOOP 标记：REAPER 显式写出时以其为准；缺失（极老工程/第三方生成器）
    // 时回退到"为新的音频块启用循环"设置。
    let item_loop = if item.has_loop_token {
        item.is_loop
    } else {
        crate::config::loop_new_clips_default()
    };
    let s_offs = take.s_offs; // source offset (seconds)

    // 防御性钳制：损坏 / 伪造 RPP 的极端 POSITION/LENGTH 不得进入下游
    // 帧向量分配（合法工程远低于一天）。
    let item_pos = clamp_import_position(item.position);
    let item_length = clamp_import_length(item.length);
    let (manual_fade_in_sec, manual_fade_out_sec, auto_fade_in_sec, auto_fade_out_sec) =
        effective_item_fades(item, take, item_length.max(0.0));
    // 形状/曲率与长度使用同一 take-vs-item 优先规则。
    let (fade_in_shape, fade_in_dir) = if reaper_fade_effective_length_sec(&take.fade_in) > 1e-9 {
        reaper_fade_shape_dir(&take.fade_in)
    } else {
        reaper_fade_shape_dir(&item.fade_in)
    };
    let (fade_out_shape, fade_out_dir) = if reaper_fade_effective_length_sec(&take.fade_out) > 1e-9
    {
        reaper_fade_shape_dir(&take.fade_out)
    } else {
        reaper_fade_shape_dir(&item.fade_out)
    };

    // 获取活跃 take 的包络（只取活跃 take；item 直属包络 = 默认 take 的包络）。
    // 时间基准：take 媒体时间 u，经 take_env_u_to_item_time 映射到 item 时间线。
    let take_envs = item.active_take_envelopes();
    let pitch_envelope = find_active_envelope(take_envs, PITCH_ENV_TYPES);
    let take_vol_env = if take_volume_overridden {
        find_active_envelope(take_envs, VOL_ENV_TYPES)
    } else {
        None
    };
    let take_pan_env = find_active_envelope(take_envs, PAN_ENV_TYPES);
    let take_mute_env = find_active_envelope(take_envs, MUTE_ENV_TYPES);

    // ─── 处理 Stretch Markers ───
    // 拉伸标记是 Take 属性：先有标记、后有 Item 裁断。分段映射以全部标记
    // 为锚点（含窗口外锚点），再按 item 窗口裁断出各段；倒放 take 的标记
    // 源坐标在构建时镜像回原始媒体坐标。缺媒体时长（无法镜像）时回退到
    // 单 clip 路径，显式告警避免静默丢速率。
    if item_reversed
        && !item.stretch_markers.is_empty()
        && !duration_sec.is_some_and(|d| d.is_finite() && d > 0.0)
    {
        log::warn!(
            "reaper_import: reversed item at {} has {} stretch marker(s) but media duration is unknown; ignoring markers",
            item.position,
            item.stretch_markers.len()
        );
    }
    let segments = stretch_segments_full_cover(
        &item.stretch_markers,
        s_offs,
        play_rate,
        item_length,
        duration_sec,
        item_reversed,
    );

    if !segments.is_empty() {
        // 有 stretch markers：拆分为多段
        // v4 边界：拆段路径按 active take 展开成多个单 take 段 Clip，
        // 其余 take 无法随之拆分、被静默丢弃 —— 显式告警避免无人察觉。
        if !item.takes.is_empty() {
            log::warn!(
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

        for (seg_idx, seg) in segments.iter().enumerate() {
            let seg_avg_rate = seg.velocity_average().max(0.01);
            let effective_rate = seg_avg_rate * play_rate;
            let seg_timeline_duration = seg_timeline_durations[seg_idx];

            // 分段重叠与淡入淡出（重叠伸入相邻段的内容，总是可行的）
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
            let actual_pre_src = want_pre * effective_rate;
            let actual_post_src = want_post * effective_rate;
            let actual_pre_tl = actual_pre_src / effective_rate;
            let actual_post_tl = actual_post_src / effective_rate;

            // 段源窗口由标记映射直接给出（原始媒体坐标，start ≤ end）。
            // 倒放段：时间线推进时源位置自 end 向 start 递减，pre 重叠
            // 伸向时间线上更早一段的内容 = 窗口高端之外，post 反之。
            let (clip_src_start, clip_src_end) = if item_reversed {
                (
                    seg.src_start - actual_post_src,
                    (seg.src_end + actual_pre_src).max(seg.src_start - actual_post_src),
                )
            } else {
                (
                    seg.src_start - actual_pre_src,
                    (seg.src_end + actual_post_src).max(seg.src_start - actual_pre_src),
                )
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
                fade_in_shape: 0.0,
                fade_out_shape: 0.0,
                fade_in_dir: 0.0,
                fade_out_dir: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
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

            current_timeline_pos += seg_timeline_duration;
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

            // 段间合成淡化用线性（REAPER 拉伸段默认）；item 首尾保留导入形状/曲率。
            let fade_in_shape = if seg_idx == 0 {
                Some(fade_in_shape)
            } else {
                None
            };
            let fade_out_shape = if seg_idx + 1 == seg_count {
                Some(fade_out_shape)
            } else {
                None
            };

            clip.fade_in_sec = fade_in_sec;
            clip.fade_out_sec = fade_out_sec;
            if let Some(shape) = fade_in_shape {
                clip.fade_in_shape = shape;
                clip.fade_in_dir = fade_in_dir;
            } else {
                clip.fade_in_shape = 0.0;
                clip.fade_in_dir = 0.0;
            }
            if let Some(shape) = fade_out_shape {
                clip.fade_out_shape = shape;
                clip.fade_out_dir = fade_out_dir;
            } else {
                clip.fade_out_shape = 0.0;
                clip.fade_out_dir = 0.0;
            }

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
        // 非活跃 take 的包络当前不进 ClipTake（v4 边界）：留痕避免静默
        // 丢用户数据（活跃 take 的包络已晋升为轨道曲线 / PITCHENV）。
        let dropped_take_envs: usize = (active_take_idx != 0)
            .then(|| item.default_take.envelopes.len())
            .unwrap_or(0)
            + item
                .takes
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx + 1 != active_take_idx)
                .map(|(_, t)| t.envelopes.len())
                .sum::<usize>();
        if dropped_take_envs > 0 {
            log::warn!(
                "reaper_import: item at {} drops {dropped_take_envs} envelope(s) on non-active take(s) (ClipTake has no envelope container yet)",
                item.position
            );
        }
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
            fade_in_shape,
            fade_out_shape,
            fade_in_dir,
            fade_out_dir,
            fade_in_curve: String::new(),
            fade_out_curve: String::new(),
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
    }

    // ─── 写入 take 包络晋升数据（pitch / volume / pan / mute gate） ───
    // 音高与包络使用同一全局 u↔t 仿射映射（take 速率，见 reaper_parser
    // 的唯一转换点），与拉伸分段无关；帧范围 = item 时间线跨度（分段
    // clip 恰好铺满该跨度，故逐 item 写一次即可）。
    write_take_envelope_frames(
        pitch_accum,
        curves_accum,
        item_pos + time_offset,
        item_length,
        play_rate,
        item_pitch_semitones,
        pitch_envelope,
        take_vol_env,
        take_pan_env,
        take_mute_env,
    );
}

// ─── 包络类型匹配与采样 ───

pub(crate) const PITCH_ENV_TYPES: &[&str] = &["PITCHENV"];
pub(crate) const VOL_ENV_TYPES: &[&str] = &["VOLENV", "VOLENV2"];
pub(crate) const PAN_ENV_TYPES: &[&str] = &["PANENV", "PANENV2"];
pub(crate) const MUTE_ENV_TYPES: &[&str] = &["MUTEENV", "MUTEENV2"];

/// 曲线 key：静音门控（内部中间 key，构建阶段乘入 volume 后剥离）。
const MUTE_GATE_KEY: &str = "@mute_gate";

fn env_type_matches(env_type: &str, kinds: &[&str]) -> bool {
    kinds.iter().any(|k| env_type.eq_ignore_ascii_case(k))
}

/// `ACT` 首值判定（缺省 1 = 激活，REAPER 剪贴板 ENVSEG 段常无 ACT 行）。
fn envelope_is_active(env: &ReaperEnvelope) -> bool {
    env.act.first().copied().unwrap_or(1) != 0
}

/// 在包络列表中查找第一个「类型匹配 + 激活 + 有点」的包络。
///
/// 注意只匹配精确类型别名：旧实现的 `contains("PITCH") || == "ENVSEG"`
/// 会把 ENVSEG 误判为音高包络（ENVSEG 是剪贴板轨道包络段的块名，不是
/// 音高包络）。
fn find_active_envelope<'a>(envs: &'a [ReaperEnvelope], kinds: &[&str]) -> Option<&'a ReaperEnvelope> {
    envs.iter().find(|env| {
        env_type_matches(&env.env_type, kinds) && envelope_is_active(env) && !env.points.is_empty()
    })
}

/// 将包络 PT 行预处理为 (位置, 值, 形状) 三元组（按位置升序）。
pub(crate) fn reaper_env_points(env: &ReaperEnvelope) -> Vec<(f64, f64, i32)> {
    let mut points: Vec<(f64, f64, i32)> = env
        .points
        .iter()
        .filter(|pt| pt.len() >= 2)
        .filter(|pt| pt[0].is_finite() && pt[1].is_finite())
        .map(|pt| (pt[0], pt[1], pt.get(2).copied().unwrap_or(0.0) as i32))
        .collect();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    points
}

/// REAPER 包络采样：shape=1（方波）阶梯保持；其它形状线性近似；
/// 首点前 / 末点后按 REAPER hold 语义延续首/末点值。
///
/// 边界归属：`pos ≤ t` 分段——t 恰落在某点上时取该点值（方波点切换、
/// 线性点取值都符合 REAPER 语义）。
pub(crate) fn sample_reaper_envelope(points: &[(f64, f64, i32)], t: f64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let idx = points.partition_point(|p| p.0 <= t);
    if idx == 0 {
        return Some(points[0].1);
    }
    if idx == points.len() {
        return Some(points[points.len() - 1].1);
    }
    let (t0, v0, shape0) = points[idx - 1];
    let (t1, v1, _) = points[idx];
    if shape0 == 1 || (t1 - t0).abs() < 1e-12 {
        return Some(v0);
    }
    let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    Some(v0 + (v1 - v0) * frac)
}

// ─── 曲线帧累积器 ───

#[derive(Default, Clone, Copy)]
pub(crate) struct CurveFrameAccumulator {
    sum: f64,
    weight: f64,
}

type CurveAccum = BTreeMap<String, Vec<CurveFrameAccumulator>>;

/// 用包络在 [start_sec, end_sec] 的采样值**覆写**累积器帧（weight=1）。
///
/// 优先级机制：轨道包络 / ENVSEG 段覆写 take 晋升数据——先铺 take 贡献，
/// 再用本函数覆写轨道（或段）覆盖的范围。
fn overwrite_accum_span(
    accum: &mut CurveAccum,
    key: &str,
    points: &[(f64, f64, i32)],
    span_start_sec: f64,
    span_end_sec: f64,
    // 包络点坐标 → 采样时刻的偏移（轨道包络为绝对时间传 0；ENVSEG 段传段起点）
    env_time_origin: f64,
) {
    if points.is_empty() || span_end_sec <= span_start_sec {
        return;
    }
    let start_frame = clamp_import_frame(span_start_sec);
    let end_frame =
        ((span_end_sec / FRAME_PERIOD).ceil().max(0.0) as usize).min(MAX_IMPORT_FRAME_IDX);
    let entry = accum.entry(key.to_string()).or_default();
    for frame_idx in start_frame..=end_frame {
        let t = frame_idx as f64 * FRAME_PERIOD;
        if t < span_start_sec || t > span_end_sec {
            continue;
        }
        let Some(value) = sample_reaper_envelope(points, t - env_time_origin) else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        if frame_idx >= entry.len() {
            entry.resize(frame_idx + 1, CurveFrameAccumulator::default());
        }
        entry[frame_idx] = CurveFrameAccumulator { sum: value, weight: 1.0 };
    }
}

// ─── Take 包络晋升写入 ───

/// 将活跃 take 的包络写入帧累积器（pitch 偏移 + volume/pan/mute gate）。
///
/// 时间映射：u = (t − item_start) × take_rate（全局仿射，与拉伸分段无关；
/// 帧范围 = item 时间线跨度）。音高 = 静态偏移 + PITCHENV 采样（半音，
/// 叠加语义）；volume/pan 采样值进入轨道曲线；MUTEENV 进入内部门控 key。
#[allow(clippy::too_many_arguments)]
fn write_take_envelope_frames(
    pitch_accum: &mut Vec<PitchFrameAccumulator>,
    curves_accum: &mut CurveAccum,
    item_start_tl: f64,
    item_length_tl: f64,
    take_rate: f64,
    static_pitch_semitones: f64,
    pitch_env: Option<&ReaperEnvelope>,
    vol_env: Option<&ReaperEnvelope>,
    pan_env: Option<&ReaperEnvelope>,
    mute_env: Option<&ReaperEnvelope>,
) {
    let item_end_tl = item_start_tl + item_length_tl.max(0.0);
    let has_pitch_shift = static_pitch_semitones.abs() > 1e-6;
    let pitch_points = pitch_env.map(reaper_env_points);
    let has_pitch_env = pitch_points.as_ref().is_some_and(|p| !p.is_empty());

    if has_pitch_shift || has_pitch_env {
        let static_semitones = static_pitch_semitones;
        for (frame_idx, entry) in pitch_frame_span(pitch_accum, item_start_tl, item_end_tl) {
            let t = frame_idx as f64 * FRAME_PERIOD;
            let time_in_item = t - item_start_tl;
            if time_in_item < 0.0 || time_in_item > item_length_tl.max(0.0) {
                continue;
            }
            let u = item_time_to_take_env_u(time_in_item, take_rate);
            let mut offset = static_semitones;
            if let Some(points) = pitch_points.as_ref() {
                if let Some(v) = sample_reaper_envelope(points, u) {
                    offset += v;
                }
            }
            entry.sum += offset;
            entry.weight += 1.0;
        }
    }

    let (vol_points, pan_points, gate_points) = (
        vol_env.map(reaper_env_points),
        pan_env.map(reaper_env_points),
        mute_env.map(reaper_env_points),
    );
    if let Some(points) = vol_points.as_ref().filter(|p| !p.is_empty()) {
        accumulate_take_env(accum_for(curves_accum, "volume"), points, item_start_tl, item_end_tl, take_rate);
    }
    if let Some(points) = pan_points.as_ref().filter(|p| !p.is_empty()) {
        accumulate_take_env(accum_for(curves_accum, "pan"), points, item_start_tl, item_end_tl, take_rate);
    }
    if let Some(points) = gate_points.as_ref().filter(|p| !p.is_empty()) {
        accumulate_take_env(accum_for(curves_accum, MUTE_GATE_KEY), points, item_start_tl, item_end_tl, take_rate);
    }
}

/// 迭代 [start, end] 覆盖的帧（自动扩容），返回 (帧号, 可变槽位)。
///
/// 直接对目标子切片迭代：此前 `iter_mut().enumerate().filter(range)` 会
/// 为每个 item 扫过整个轨道级累积器（O(items × 工程帧数)，长工程导入
/// 秒级卡顿）；resize 后目标区间必在界内。
fn pitch_frame_span<'a>(
    accum: &'a mut Vec<PitchFrameAccumulator>,
    start_sec: f64,
    end_sec: f64,
) -> impl Iterator<Item = (usize, &'a mut PitchFrameAccumulator)> + 'a {
    let start_frame = clamp_import_frame(start_sec);
    let end_frame = ((end_sec / FRAME_PERIOD).ceil().max(0.0) as usize).min(MAX_IMPORT_FRAME_IDX);
    let end_frame = end_frame.max(start_frame);
    if end_frame >= accum.len() {
        accum.resize(end_frame + 1, PitchFrameAccumulator::default());
    }
    accum[start_frame..=end_frame]
        .iter_mut()
        .enumerate()
        .map(move |(i, slot)| (i + start_frame, slot))
}

/// take 包络按 u↔t 映射累积到帧（value = 包络在 u 处的采样）。
fn accumulate_take_env(
    entry: &mut Vec<CurveFrameAccumulator>,
    points: &[(f64, f64, i32)],
    item_start_tl: f64,
    item_end_tl: f64,
    take_rate: f64,
) {
    let start_frame = clamp_import_frame(item_start_tl);
    let end_frame =
        ((item_end_tl / FRAME_PERIOD).ceil().max(0.0) as usize).min(MAX_IMPORT_FRAME_IDX);
    for frame_idx in start_frame..=end_frame {
        let t = frame_idx as f64 * FRAME_PERIOD;
        let time_in_item = t - item_start_tl;
        if time_in_item < 0.0 || (item_end_tl > item_start_tl && time_in_item > item_end_tl - item_start_tl) {
            continue;
        }
        let u = item_time_to_take_env_u(time_in_item, take_rate);
        let Some(value) = sample_reaper_envelope(points, u) else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        if frame_idx >= entry.len() {
            entry.resize(frame_idx + 1, CurveFrameAccumulator::default());
        }
        let slot = &mut entry[frame_idx];
        slot.sum += value;
        slot.weight += 1.0;
    }
}

fn accum_for<'a>(accum: &'a mut CurveAccum, key: &str) -> &'a mut Vec<CurveFrameAccumulator> {
    accum.entry(key.to_string()).or_default()
}

/// 从累积器构建 `extra_curves`（mute gate 乘入 volume；全默认剪枝）。
pub(crate) fn build_extra_curves_from_accum(
    curves_accum: &CurveAccum,
    total_frames: usize,
) -> BTreeMap<String, Vec<f32>> {
    let mut out = BTreeMap::new();
    let vol = curves_accum.get("volume");
    let gate = curves_accum.get(MUTE_GATE_KEY);
    let has_vol = vol.is_some_and(|v| v.iter().any(|a| a.weight > 0.0));
    let has_gate = gate.is_some_and(|v| v.iter().any(|a| a.weight > 0.0));
    if has_vol || has_gate {
        let mut curve = Vec::with_capacity(total_frames);
        let mut non_default = false;
        for idx in 0..total_frames {
            let mut value = 1.0f64;
            if has_vol {
                if let Some(slot) = vol.and_then(|v| v.get(idx)) {
                    if slot.weight > 0.0 {
                        value = slot.sum / slot.weight;
                    }
                }
            }
            if has_gate {
                if let Some(slot) = gate.and_then(|g| g.get(idx)) {
                    if slot.weight > 0.0 {
                        value *= (slot.sum / slot.weight).clamp(0.0, 1.0);
                    }
                }
            }
            let value = (value as f32).clamp(0.0, 4.0);
            if (value - 1.0).abs() > 1e-6 {
                non_default = true;
            }
            curve.push(value);
        }
        if non_default {
            out.insert("volume".to_string(), curve);
        }
    }
    if let Some(pan) = curves_accum.get("pan") {
        if pan.iter().any(|a| a.weight > 0.0) {
            let mut curve = Vec::with_capacity(total_frames);
            let mut non_default = false;
            for idx in 0..total_frames {
                let value = match pan.get(idx) {
                    Some(slot) if slot.weight > 0.0 => (slot.sum / slot.weight) as f32,
                    _ => 0.0,
                };
                let value = value.clamp(-1.0, 1.0);
                if value.abs() > 1e-6 {
                    non_default = true;
                }
                curve.push(value);
            }
            if non_default {
                out.insert("pan".to_string(), curve);
            }
        }
    }
    out
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
    let reversed_playback = raw_play_rate < 0.0;
    let play_rate = raw_play_rate.abs().max(0.01);

    for note in &mut notes {
        note.start_sec = note.start_sec - soffs;
        note.end_sec = note.end_sec - soffs;
    }

    let item_length = clamp_import_length(item.length);

    // 过滤掉完全在窗口外的音符（使用源时间窗口）
    notes.retain(|n| n.end_sec > 0.0 && n.start_sec < item_length * play_rate);

    // 倒放（PLAYRATE < 0）：源窗口内容镜像逆序 —— 与音频路径"倒放消费
    // 窗口"语义一致，绝不静默丢弃 PLAYRATE 的符号（此前导入为正放）。
    if reversed_playback {
        let span = item_length * play_rate;
        for note in &mut notes {
            let s = note.start_sec.clamp(0.0, span);
            let e = note.end_sec.clamp(0.0, span);
            note.start_sec = span - e;
            note.end_sec = span - s;
        }
        notes.retain(|n| n.end_sec > n.start_sec + 1e-6);
        notes.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

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
    // MIDI take 的音量包络 v1 不消费（见 process_item 的 MIDI 分支说明），
    // 但音量钮语义一致：存在激活 VOLENV 时同样按"包络取代钮"中性化。
    let take_gain = take_linear_gain(item, take, take_volume_envelope_active(take));
    let clip_start = clamp_import_position(item.position) + time_offset;

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
        fade_in_shape: 0.0,
        fade_out_shape: 0.0,
        fade_in_dir: 0.0,
        fade_out_dir: 0.0,
        fade_in_curve: String::new(),
        fade_out_curve: String::new(),
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
    use crate::reaper_parser::ReaperStretchMarker;

    // ─── 包络适配测试辅助（自编简洁数值，不引用真实工程样例） ───

    fn wave_source(path: &str) -> ReaperSource {
        let mut src = ReaperSource::new();
        src.source_type = "WAVE".to_string();
        src.file_path = path.to_string();
        src
    }

    fn audio_item(position: f64, length: f64) -> ReaperItem {
        let mut item = ReaperItem {
            position,
            length,
            has_loop_token: true,
            is_loop: false,
            ..ReaperItem::default()
        };
        item.default_take.source = Some(wave_source("C:/missing/vocal.wav"));
        item
    }

    fn env(env_type: &str, points: Vec<(f64, f64)>) -> ReaperEnvelope {
        ReaperEnvelope {
            env_type: env_type.to_string(),
            act: vec![1, -1],
            points: points
                .into_iter()
                .map(|(p, v)| vec![p, v, 0.0])
                .collect(),
            ..ReaperEnvelope::default()
        }
    }

    fn run_process_item(
        item: &ReaperItem,
    ) -> (
        Vec<Clip>,
        Vec<PitchFrameAccumulator>,
        CurveAccum,
        Vec<String>,
    ) {
        let mut clips = Vec::new();
        let mut skipped = Vec::new();
        let mut pitch = Vec::new();
        let mut curves: CurveAccum = BTreeMap::new();
        let mut groups = HashMap::new();
        process_item(
            item,
            "track_1",
            None,
            0.0,
            &mut clips,
            &mut skipped,
            &mut pitch,
            &mut curves,
            120.0,
            &mut groups,
        );
        (clips, pitch, curves, skipped)
    }

    fn curve_value_at(
        curves: &CurveAccum,
        key: &str,
        frame: usize,
    ) -> f64 {
        curves
            .get(key)
            .and_then(|slots| slots.get(frame))
            .filter(|slot| slot.weight > 0.0)
            .map(|slot| slot.sum / slot.weight)
            .unwrap_or(f64::NAN)
    }

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
        let mut item = ReaperItem::default();
        // 显式 take 的 TAKEVOLPAN：<pan> <volume> <pan law> → volume = [1]。
        let explicit = ReaperTake {
            vol_pan: vec![0.0, 1.25, -1.0],
            ..ReaperTake::default()
        };
        item.takes.push(explicit);
        assert!((reaper_take_volume(&item.takes[0].vol_pan, true) - 1.25).abs() < 1e-9);
        assert!((take_linear_gain(&item, &item.takes[0], false) - 1.25).abs() < 1e-9);

        // Item 默认 take 的 VOLPAN：<trim> <pan> <volume> <pan law> → volume = [2]。
        let mut item2 = ReaperItem::default();
        item2.default_take.vol_pan = vec![1.0, 0.0, 0.8, -1.0];
        assert!((reaper_take_volume(&item2.default_take.vol_pan, false) - 0.8).abs() < 1e-9);
        assert!((take_linear_gain(&item2, &item2.default_take, false) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn take_linear_gain_combines_item_trim_with_take_volume() {
        // REAPER 可听增益 = item trim（VOLPAN[0]）× take volume（VOLPAN[2]）。
        // 用户拖动 item 音量把手写的是 trim —— 只取 take volume 会丢增益。
        let mut item = ReaperItem::default();
        item.default_take.vol_pan = vec![0.5, 0.0, 1.0, -1.0]; // trim -6dB
        assert!((take_linear_gain(&item, &item.default_take, false) - 0.5).abs() < 1e-9);

        // 提升：trim 与 volume 都 >1（+6dB × +6dB = +12dB）。
        let mut boost = ReaperItem::default();
        boost.default_take.vol_pan = vec![2.0, 0.0, 2.0, -1.0];
        assert!((take_linear_gain(&boost, &boost.default_take, false) - 4.0).abs() < 1e-9);

        // 负 volume（REAPER 反相）：响度取绝对值保留。
        let mut inverted = ReaperItem::default();
        inverted.default_take.vol_pan = vec![1.0, 0.0, -0.8, -1.0];
        assert!((take_linear_gain(&inverted, &inverted.default_take, false) - -0.8).abs() < 1e-9);
        assert!(
            (convert_volume(take_linear_gain(&inverted, &inverted.default_take, false)) - 0.8).abs()
                < 1e-6
        );

        // 显式 take 音量写 0 时回退主 take 音量（多 take 兼容）。
        let mut multi = ReaperItem::default();
        multi.default_take.vol_pan = vec![1.0, 0.0, 0.9, -1.0];
        multi.takes.push(ReaperTake {
            vol_pan: vec![0.0, 0.0, -1.0],
            ..ReaperTake::default()
        });
        assert!((take_linear_gain(&multi, &multi.takes[0], false) - 0.9).abs() < 1e-9);

        // 显式 take 的 TAKEVOLPAN 音量同样乘以 item trim。
        let mut explicit_trim = ReaperItem::default();
        explicit_trim.default_take.vol_pan = vec![0.5, 0.0, 1.0, -1.0];
        explicit_trim.takes.push(ReaperTake {
            vol_pan: vec![0.0, 1.5, -1.0],
            ..ReaperTake::default()
        });
        assert!(
            (take_linear_gain(&explicit_trim, &explicit_trim.takes[0], false) - 0.75).abs() < 1e-9,
            "explicit take gain must include item trim"
        );
    }

    #[test]
    fn convert_volume_keeps_boost_and_inverts_phase_to_amplitude() {
        // HiFiShifter 增益范围 [0, 4]：0 dB → 1.0、+6 dB → 2.0 原样保留；
        // 负值（REAPER 相位反转）取绝对值保留响度；超出 +12 dB（4×）截断。
        assert!((convert_volume(1.0) - 1.0).abs() < 1e-6);
        assert!((convert_volume(0.5) - 0.5).abs() < 1e-6);
        assert!((convert_volume(2.0) - 2.0).abs() < 1e-6);
        assert!((convert_volume(-0.8) - 0.8).abs() < 1e-6);
        assert!((convert_volume(16.0) - 4.0).abs() < 1e-6);
        assert!((convert_volume(0.0) - 0.0).abs() < 1e-6);
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
            &mut BTreeMap::new(),
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

    #[test]
    fn stretch_marker_item_splits_with_leading_segment_at_base_rate() {
        let mut item = ReaperItem {
            position: 10.0,
            length: 53.18616059322192,
            has_loop_token: true,
            is_loop: true,
            default_take: ReaperTake {
                name: "Vocal-5-1.wav".to_string(),
                s_offs: 0.0,
                source: {
                    let mut src = ReaperSource::new();
                    src.source_type = "WAVE".to_string();
                    src.file_path = "C:/missing/Vocal-5-1.wav".to_string();
                    Some(src)
                },
                ..ReaperTake::default()
            },
            ..ReaperItem::default()
        };
        for (offset, position, vc) in [
            (2.05, 2.05, 0.502487562),
            (2.971878688, 3.902976163, 0.0),
            (8.142757375, 6.240799448, 0.509465607),
            (10.770669133, 10.578918316, 0.0),
            (28.138154992, 31.798583333, 0.0),
        ] {
            item.stretch_markers.push(ReaperStretchMarker {
                offset,
                position,
                velocity_change: vc,
            });
        }

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
            &mut BTreeMap::new(),
            120.0,
            &mut groups,
        );

        assert_eq!(clips.len(), 6, "前导段 + 4 个标记段 + 尾段");

        // 前导段：落在 item 起点，基准速率，源从 SOFFS 起；
        // 源终点 = 首标记源位置 + 0.05s 段间重叠。
        let lead = &clips[0];
        assert!((lead.start_sec - 10.0).abs() < 1e-9);
        assert!((lead.playback_rate - 1.0).abs() < 1e-6);
        assert!((lead.source_start_sec - 0.0).abs() < 1e-9);
        assert!((lead.source_end_sec - 2.10).abs() < 1e-6);

        // 首个标记段：速率 = Δ源/Δtake媒体 ≈ 2.01，源窗口覆盖标记锚点区间。
        let second = &clips[1];
        let expected_rate = (3.902976163 - 2.05) / (2.971878688 - 2.05);
        assert!((second.playback_rate - expected_rate).abs() < 1e-6);
        assert!(second.source_start_sec < 2.05 && second.source_end_sec > 3.902976163,
            "源窗口必须锚定标记的绝对源位置（允许段间重叠外扩）");

        // 中间段抽查：压缩段速率 < 1。
        let compressed = &clips[2];
        assert!((compressed.playback_rate - (6.240799448 - 3.902976163)
            / (8.142757375 - 2.971878688))
        .abs()
            < 1e-6);

        // 尾段：基准速率外推，且 clips 链一直铺到 item 末端
        //（旧实现丢失末标记之后的 25s）。
        let tail = clips.last().unwrap();
        assert!((tail.playback_rate - 1.0).abs() < 1e-6);
        let tail_end = tail.start_sec + tail.length_sec;
        assert!((tail_end - (10.0 + 53.18616059322192)).abs() < 1e-9);

        // 段间时间线连续（重叠只发生在相邻段的共享边界上）。
        for pair in clips.windows(2) {
            let prev_end = pair[0].start_sec + pair[0].length_sec;
            assert!(
                (pair[1].start_sec - prev_end).abs() < pair[1].length_sec.max(0.05) + 1e-6,
                "相邻段起点应落在前段范围内"
            );
        }
    }

    // ─── Take 包络 → HiFiShifter 参数线 ───

    #[test]
    fn take_pitch_envelope_samples_in_u_domain_scaled_by_play_rate() {
        // item length 4s、take rate 0.5 → u 域 0..2。
        // PITCHENV: (u=0, +1) (u=2, −1) → t=0 → +1、t=2s → 0、t=4s → −1
        //（u→t 斜率 = 1/rate；旧实现把包络坐标直接当 timeline 时间会挤在
        // 前 2s 内）。
        let mut item = audio_item(0.0, 4.0);
        item.default_take.play_rate = vec![0.5, 1.0, 0.0, -1.0, 0.0, 0.0025];
        item.default_take
            .envelopes
            .push(env("PITCHENV", vec![(0.0, 1.0), (2.0, -1.0)]));

        let (_, pitch, _, _) = run_process_item(&item);
        let frames = build_pitch_frames(&pitch, 801);
        assert!((frames[0] - 1.0).abs() < 1e-6, "t=0 → u=0 → +1");
        assert!(frames[400].abs() < 1e-6, "t=2s → u=1 → 0");
        assert!((frames[800] + 1.0).abs() < 1e-6, "t=4s → u=2 → −1");
    }

    #[test]
    fn only_active_take_envelope_is_imported() {
        // 两个 take 各带 PITCHENV（数值刻意区分：default +1、Alt +5）。
        // 无 TAKE SEL → 活跃 take = default take → 偏移 +1；
        // TAKE SEL → 活跃 take = Alt → 偏移 +5。
        let mut item = audio_item(0.0, 2.0);
        item.default_take
            .envelopes
            .push(env("PITCHENV", vec![(0.0, 1.0)]));
        item.takes.push(ReaperTake {
            selected: false,
            name: "Alt".to_string(),
            source: Some(wave_source("C:/missing/alt.wav")),
            envelopes: vec![env("PITCHENV", vec![(0.0, 5.0)])],
            ..ReaperTake::default()
        });

        let (_, pitch, _, _) = run_process_item(&item);
        let frames = build_pitch_frames(&pitch, 401);
        assert!((frames[0] - 1.0).abs() < 1e-6, "无 SEL → default take 的包络");

        item.takes[0].selected = true;
        let (_, pitch, _, _) = run_process_item(&item);
        let frames = build_pitch_frames(&pitch, 401);
        assert!((frames[0] - 5.0).abs() < 1e-6, "TAKE SEL → 显式 take 的包络");
    }

    #[test]
    fn take_volume_envelope_neutralizes_knob_and_promotes_curve() {
        // take volume 钮 = 2.0 + 激活 VOLENV（0.5 → 1.5）：
        // 绝对包络语义 → 钮被取代（gain = trim × 1.0），包络晋升为音量曲线。
        let mut item = audio_item(0.0, 2.0);
        item.default_take.vol_pan = vec![1.0, 0.0, 2.0, -1.0];
        item.default_take
            .envelopes
            .push(env("VOLENV", vec![(0.0, 0.5), (2.0, 1.5)]));

        let (clips, _, curves, _) = run_process_item(&item);
        assert!(
            (clips[0].gain - 1.0).abs() < 1e-6,
            "gain = trim × 1.0（钮被包络取代），而非 × 2.0"
        );
        assert!((curve_value_at(&curves, "volume", 0) - 0.5).abs() < 1e-6);
        assert!((curve_value_at(&curves, "volume", 400) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn inactive_envelopes_are_ignored() {
        // ACT 0 的 PITCHENV / VOLENV 均忽略：无 pitch 偏移、无音量曲线，
        // 音量钮照常生效（gain = trim × 钮值）。
        let mut item = audio_item(0.0, 2.0);
        item.default_take.vol_pan = vec![1.0, 0.0, 2.0, -1.0];
        item.default_take.envelopes.push(ReaperEnvelope {
            env_type: "PITCHENV".to_string(),
            act: vec![0, -1],
            points: vec![vec![0.0, 3.0, 0.0]],
            ..ReaperEnvelope::default()
        });
        item.default_take.envelopes.push(ReaperEnvelope {
            env_type: "VOLENV".to_string(),
            act: vec![0, -1],
            points: vec![vec![0.0, 0.5, 0.0]],
            ..ReaperEnvelope::default()
        });

        let (clips, pitch, curves, _) = run_process_item(&item);
        assert!((clips[0].gain - 2.0).abs() < 1e-6, "ACT 0 → 钮生效");
        assert!(pitch.is_empty(), "ACT 0 → 无音高偏移");
        assert!(curves.get("volume").is_none(), "ACT 0 → 无音量曲线");
    }

    #[test]
    fn take_pan_envelope_promotes_to_pan_curve() {
        let mut item = audio_item(0.0, 2.0);
        item.default_take
            .envelopes
            .push(env("PANENV", vec![(0.0, -1.0), (2.0, 1.0)]));

        let (_, _, curves, _) = run_process_item(&item);
        assert!((curve_value_at(&curves, "pan", 0) + 1.0).abs() < 1e-6);
        assert!((curve_value_at(&curves, "pan", 400) - 1.0).abs() < 1e-6);
    }

    // ─── 轨道包络（.rpp / TRACK 块）→ 曲线 ───

    #[test]
    fn track_envelopes_override_take_and_neutralize_fader() {
        // 轨道：VOLPAN [0.5, 0.8] + VOLENV2 (2.0 → 1.0) + PANENV2 (0.5)
        // + MUTEENV 方波门控（1s..3s 静音）。
        // - VOLENV2 绝对语义 → Track.volume 中性化为 1.0（推子被取代）；
        // - PANENV2 取代声像钮 → 静态 0.8 抑制，曲线 = 0.5；
        // - MUTEENV 乘入门控。
        let mut data = ReaperData::default();
        data.is_track_data = true;
        let mut track = ReaperTrack::default();
        track.vol_pan = vec![0.5, 0.8, -1.0, -1.0, 1.0];
        track.envelopes.push(env("VOLENV2", vec![(0.0, 2.0), (4.0, 1.0)]));
        track.envelopes.push(env("PANENV2", vec![(0.0, 0.5)]));
        track.envelopes.push(ReaperEnvelope {
            env_type: "MUTEENV".to_string(),
            act: vec![1, -1],
            points: vec![
                vec![0.0, 1.0, 1.0],
                vec![1.0, 0.0, 1.0],
                vec![3.0, 1.0, 1.0],
            ],
            ..ReaperEnvelope::default()
        });
        data.tracks.push(track);

        let result = convert_reaper_data(data, None, 120.0).unwrap();
        let hs_track = &result.timeline.tracks[0];
        assert!(
            (hs_track.volume - 1.0).abs() < 1e-9,
            "VOLENV2 取代推子 → Track.volume 中性化"
        );
        let params = result.timeline.params_by_root_track.values().next().unwrap();
        let volume = params.extra_curves.get("volume").unwrap();
        let pan = params.extra_curves.get("pan").unwrap();
        let fp = FRAME_PERIOD;
        // 音量：包络绝对值（非推子 0.5 的乘积）；末端 = 包络末值 1.0。
        assert!((volume[0] - 2.0).abs() < 1e-3);
        assert!((volume[(4.0 / fp) as usize] - 1.0).abs() < 1e-3);
        // 声像：PANENV2 抑制静态 0.8。
        assert!((pan[0] - 0.5).abs() < 1e-3);
        // MUTEENV 方波：t∈[1s,3s) 门控 0；t=3s 起恢复。
        // t=3s 处音量包络本身为 1.25（2→1 线性），门控恢复后 = 1.25。
        assert!((volume[(1.0 / fp) as usize] - 0.0).abs() < 1e-3);
        assert!((volume[(2.0 / fp) as usize] - 0.0).abs() < 1e-3);
        assert!((volume[(3.0 / fp) as usize] - 1.25).abs() < 1e-3);
    }

    // ─── ENVSEG 轨道包络段（item 剪贴板） ───

    fn envseg_data(seg_start: f64, seg_len: f64, points: Vec<(f64, f64)>) -> ReaperData {
        let mut data = ReaperData::default();
        data.is_track_data = false;
        data.track_offsets = vec![0];
        let mut track = ReaperTrack::default();
        track.items.push(audio_item(0.0, 2.0));
        track.envelopes.push(ReaperEnvelope {
            env_type: "VOLENV2".to_string(),
            act: vec![1, -1],
            seg_range: Some(vec![
                seg_start,
                seg_len,
                seg_start * 2.0,
                seg_len * 2.0,
            ]),
            points: points
                .into_iter()
                .map(|(p, v)| vec![p, v, 0.0])
                .collect(),
            ..ReaperEnvelope::default()
        });
        data.tracks.push(track);
        data
    }

    fn run_envseg_import(data: ReaperData) -> crate::state::TrackParamsState {
        let result = convert_reaper_items_to_existing_tracks(
            data,
            0.0,
            0,
            &["track_1".to_string()],
            &[1.0],
            120.0,
            0,
        )
        .unwrap();
        result
            .timeline
            .params_by_root_track
            .get("track_1")
            .cloned()
            .unwrap()
    }

    #[test]
    fn envseg_clipboard_segment_imports_to_target_track_curves() {
        // <ENVSEG VOLENV2 SEG_RANGE 10 2 ...>，段内相对点 (0, 0.8) (2, 0.4)。
        // M1（段内相对）：项目位置 = seg_range[0] + PT → t=10..12s。
        let params = run_envseg_import(envseg_data(
            10.0,
            2.0,
            vec![(0.0, 0.8), (2.0, 0.4)],
        ));
        let volume = params.extra_curves.get("volume").unwrap();
        let fp = FRAME_PERIOD;
        assert!((volume[(10.0 / fp) as usize] - 0.8).abs() < 1e-3);
        assert!((volume[(12.0 / fp) as usize] - 0.4).abs() < 1e-3);
        // 段外为默认（1.0）。
        assert!((volume[(5.0 / fp) as usize] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn envseg_absolute_points_fall_back_to_absolute_interpretation() {
        // seg_range[0]=10 且所有点 ≥ 10（相对解释下不可能出现的形态）→
        // M2 判别：点为绝对工程秒，相对化后段同样落在 t=10..12s。
        let params = run_envseg_import(envseg_data(
            10.0,
            2.0,
            vec![(10.0, 0.8), (12.0, 0.4)],
        ));
        let volume = params.extra_curves.get("volume").unwrap();
        let fp = FRAME_PERIOD;
        assert!((volume[(10.0 / fp) as usize] - 0.8).abs() < 1e-3);
        assert!((volume[(12.0 / fp) as usize] - 0.4).abs() < 1e-3);
    }

    #[test]
    fn envseg_native_dual_timebase_multitrack_imports_aligned() {
        // 原生多轨形态（ClipboardData/20260908-025849）：
        // - SEG_RANGE = [起秒, 终秒, 起QN, 终QN]（第二字段为终点）；
        // - PT 为工程绝对秒；
        // - 各轨 clip 位置不共享时间范围（轨 1 在 0、轨 2 在 77）。
        // 粘贴对齐（首个位置 → 光标 10s）后：轨 2 的包络必须仍与其 item
        // 同步（87..90.824），不得被相对解释拉回光标附近。
        let mut data = ReaperData::default();
        data.is_track_data = false;
        data.track_offsets = vec![0, 1];
        let mut track0 = ReaperTrack::default();
        track0.items.push(audio_item(0.0, 2.0));
        track0.envelopes.push(ReaperEnvelope {
            env_type: "VOLENV2".to_string(),
            act: vec![1, -1],
            seg_range: Some(vec![0.0, 2.0, 0.0, 4.0]),
            points: vec![vec![0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         vec![2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]],
            ..ReaperEnvelope::default()
        });
        let mut track1 = ReaperTrack::default();
        track1.items.push(audio_item(77.0, 3.824479166667));
        track1.envelopes.push(ReaperEnvelope {
            env_type: "VOLENV2".to_string(),
            act: vec![1, -1],
            seg_range: Some(vec![77.0, 80.824479166667, 154.0, 161.648958333334]),
            points: vec![vec![77.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 154.0],
                         vec![80.824479166667, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 161.648958333334]],
            ..ReaperEnvelope::default()
        });
        data.tracks.push(track0);
        data.tracks.push(track1);

        let result = convert_reaper_items_to_existing_tracks(
            data,
            10.0,
            0,
            &["track_1".to_string(), "track_2".to_string()],
            &[1.0, 1.0],
            120.0,
            0,
        )
        .unwrap();
        let fp = FRAME_PERIOD;
        // item 落点：轨 1 → 10..12；轨 2 → 87..90.824（首个位置对齐光标）。
        let clip_t1 = result
            .timeline
            .clips
            .iter()
            .find(|c| c.track_id == "track_1")
            .unwrap();
        let clip_t2 = result
            .timeline
            .clips
            .iter()
            .find(|c| c.track_id == "track_2")
            .unwrap();
        assert!((clip_t1.start_sec - 10.0).abs() < 1e-6);
        assert!((clip_t2.start_sec - 87.0).abs() < 1e-6);

        let params_t1 = result.timeline.params_by_root_track.get("track_1").unwrap();
        let volume_t1 = params_t1.extra_curves.get("volume").unwrap();
        assert!((volume_t1[(10.0 / fp) as usize] - 0.5).abs() < 1e-3);
        assert!((volume_t1[(12.0 / fp) as usize] - 1.5).abs() < 1e-3);

        // 关键：轨 2 的包络必须落在 item 范围内（87..90.824），而不是被
        // 相对解释拉到光标附近。
        let params_t2 = result.timeline.params_by_root_track.get("track_2").unwrap();
        let volume_t2 = params_t2.extra_curves.get("volume").unwrap();
        assert!((volume_t2[(87.0 / fp) as usize] - 0.5).abs() < 1e-3, "轨 2 包络起点 = item 起点");
        // 段末（80.824+10 = 90.824s）为端点值 1.5；t=90 处为线性中间值 ≈1.284。
        let end_frame = (90.824479166667f64 / fp).floor() as usize;
        assert!(
            (volume_t2[end_frame] - 1.5).abs() < 5e-3,
            "轨 2 包络终点 = item 终点（frame {} 实际 {}）",
            end_frame,
            volume_t2[end_frame]
        );
        let mid = volume_t2[(90.0 / fp) as usize];
        assert!(
            (mid - 1.2845).abs() < 5e-3,
            "线性插值中间值，实际 {}",
            mid
        );
        assert!(
            (volume_t2[(10.0 / fp) as usize] - 1.0).abs() < 1e-6,
            "轨 2 的段不得出现在光标附近"
        );
    }
}
