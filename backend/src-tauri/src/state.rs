use crate::audio_engine::AudioEngine;
use crate::audio_utils::try_read_wav_info;
use crate::clip_pitch_cache::ClipPitchCache;
use crate::midi_import::MidiNoteEvent;
use crate::models::{
    ModelConfig, ModelConfigPayload, PitchRange, ProjectMetaPayload, RuntimeInfoPayload,
    TimelineClip, TimelineStatePayload, TimelineTrack,
};
use crate::project::CustomScale;
use crate::time_stretch::UserStretchAlgorithm;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use uuid::Uuid;

fn default_frame_period_ms() -> f64 {
    5.0
}

fn default_project_scale_notes() -> Vec<u8> {
    vec![0, 2, 4, 5, 7, 9, 11]
}

fn default_formant_target_f1_hz() -> f64 {
    800.0
}

fn default_formant_target_f2_hz() -> f64 {
    1400.0
}

fn default_formant_strength() -> f64 {
    0.50
}

fn default_gain() -> f32 {
    1.0
}

fn default_playback_rate() -> f32 {
    1.0
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Clip 源媒体的总时长（秒）：优先 `duration_frames / source_sample_rate`，
/// 回退 `duration_sec`；均不可用时返回 None。
///
/// Loop（循环源）模式的回绕周期即该值 —— "循环原始音频文件"。
///
/// 纯音高参考块（Pitch Reference Clip，无源媒体）的内容时长 = 音符内容的
/// 最大结束时间 —— 使其循环逻辑与普通媒体 Clip **完全一致**：Loop 回绕
/// 整个音符内容（而非窗口跨度），窗口之外的部分为静音/无音高。
pub(crate) fn clip_source_media_duration_sec(clip: &Clip) -> Option<f64> {
    if let (Some(frames), Some(sample_rate)) = (clip.duration_frames, clip.source_sample_rate) {
        if sample_rate > 0 && frames > 0 {
            return Some(frames as f64 / sample_rate as f64);
        }
    }
    if let Some(duration) = clip.duration_sec.filter(|d| d.is_finite() && *d > 0.0) {
        return Some(duration);
    }
    if let Some(notes) = clip.midi_note_data.as_ref() {
        let max_end = notes.iter().map(|n| n.end_sec).fold(0.0f64, f64::max);
        if max_end.is_finite() && max_end > 0.0 {
            return Some(max_end);
        }
    }
    None
}

/// Loop（循环源）回绕周期的统一取值（秒）：媒体时长已知时取媒体时长，
/// 否则退化为 `max(source_end, source_start)`（兜底仍为非负、有限的键值）。
///
/// 拉伸缓存的**生产者**（schedule_stretch_jobs 的 StretchJob）与**消费者**
/// （build_snapshot 的换入查找）必须使用同一函数 —— 此前两处回退链不一致
/// （一边 `duration_sec.unwrap_or(max(end,start))` 且不过滤非有限值，
/// 另一边 `clip_source_media_duration_sec(...).unwrap_or(source_end)`），
/// 在元数据缺失/异常时会生成互不匹配的缓存键，导致拉伸结果永远无法命中。
pub(crate) fn clip_loop_wrap_total_sec(clip: &Clip) -> f64 {
    clip_source_media_duration_sec(clip)
        .unwrap_or_else(|| clip.source_end_sec.max(clip.source_start_sec))
        .max(0.0)
}

/// 非 Loop 正放 Clip 的**派生源终点**（REAPER 派生窗口模型）：
/// se' = source_start + length×rate。
///
/// 消费区间为 [source_start, se')，落在媒体之外的部分渲染为静音
/// （右缘延伸的尾部静音 / REAPER 左延伸的前导静音）。存储的
/// `source_end_sec` 在循环开关、历史数据等场景下可能与长度脱钩 ——
/// 所有切片/窗口消费点都应使用本函数取**有效**终点，避免陈旧窗口
/// 冻结静音区或截断音频。Loop（回绕锚点语义）与倒放（反向锚点）
/// 保持存储字段不变。
pub(crate) fn clip_effective_source_end_sec(clip: &Clip) -> f64 {
    if !clip.loop_enabled && !clip.reversed {
        clip.source_start_sec + clip.length_sec.max(0.0) * clip.playback_rate as f64
    } else {
        clip.source_end_sec
    }
}

/// 非 Loop Clip 的**实际消费窗口**（源域秒，方向无关的统一模型）：
///
///   正放 win = [source_start, source_start + length×rate)
///   倒放 win = [source_end − length×rate, source_end)
///
/// 消费方向：正放自 win 起点升、倒放自 win 终点降；win 落在媒体
/// [0, D) 之外的部分渲染为静音 —— 倒放的 `source_end > D` 产生前导
/// 静音、窗口下探 <0 产生尾部静音（与正放 `source_start < 0` 的前导
/// 静音对称）。倒放的 `source_start` 只是历史/编辑字段，**不参与**
/// 消费数学 —— 音频切片、波形取窗、音高曲线必须全部使用本函数，
/// 否则编辑器写入的域外锚点会让内容错位（波形与音频对不上、该有声
/// 处被静音吞掉）。
///
/// Loop（回绕锚点语义）不适用本模型 —— 返回原始存储窗口，调用方须
/// 自行走锚点回绕分支。
#[allow(clippy::needless_return)]
pub(crate) fn clip_playback_window_sec(clip: &Clip) -> (f64, f64) {
    let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 1e-6 {
        clip.playback_rate as f64
    } else {
        1.0
    };
    let span = clip.length_sec.max(0.0) * rate;
    if clip.loop_enabled {
        return (clip.source_start_sec, clip.source_end_sec);
    }
    if clip.reversed {
        let end = clip.source_end_sec;
        (end - span, end)
    } else {
        let start = clip.source_start_sec;
        (start, start + span)
    }
}

/// 非 Loop Clip 消费方向的**前导静音**（时间线秒）：
///   正放：窗口起点越过媒体起点（source_start < 0）→ 前导静音；
///   倒放：窗口终点越过媒体末端（source_end > D）→ 前导静音。
/// `media_total_sec` = 解码媒体总时长（秒）；未知时传 None（按无前导
/// 静音处理）。尾部静音无需显式表达 —— 切片自然短于 clip 长度。
/// Loop 的负 source_start 是环绕锚点，不产生前导静音（返回 0）。
pub(crate) fn clip_leading_silence_sec(clip: &Clip, media_total_sec: Option<f64>) -> f64 {
    if clip.loop_enabled {
        return 0.0;
    }
    let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 1e-6 {
        clip.playback_rate as f64
    } else {
        1.0
    };
    if clip.reversed {
        let d = media_total_sec.filter(|v| v.is_finite() && *v > 0.0);
        match d {
            Some(d) => (clip.source_end_sec - d).max(0.0) / rate,
            None => 0.0,
        }
    } else {
        (-clip_playback_window_sec(clip).0).max(0.0) / rate
    }
}

/// `trim_and_resample_midi` 的源窗口实参（非 Loop 倒放专用重定向）：
///
/// 该函数内部始终按"升序窗口 → 输出再由调用方翻转"的方式处理非 Loop
/// 曲线，因此倒放 Clip 必须传入真实消费窗口 `[se−len·r, se]`（而非存储
/// 的 `[ss, se]`），否则陈旧/延伸过的窗口会把曲线拉伸到错误区域。其余
/// 模式原样透传 `(source_start, source_end)`。
pub(crate) fn clip_pitch_trim_window_sec(clip: &Clip) -> (f64, f64) {
    if !clip.loop_enabled && clip.reversed {
        let (win_start, win_end) = clip_playback_window_sec(clip);
        (win_start, win_end)
    } else {
        (clip.source_start_sec, clip.source_end_sec)
    }
}

/// 非 Loop Clip 存储字段的**加载期规范化**：使存储窗口 == 消费窗口。
///
///   正放：source_end := source_start + len·r
///   倒放：source_start := source_end − len·r
///
/// 与所有消费端的派生值逐字段一致，功能零变化；作用是让历史版本写入的
/// 陈旧/发散字段不再保留在工程数据里（避免误导任何直接读原始字段的
/// 路径，如 REAPER 导出、上下文菜单显示等）。Loop 的字段承载锚点相位，
/// 不适用本规范化。
pub(crate) fn normalize_nonloop_source_window(clip: &mut Clip) {
    if clip.loop_enabled {
        return;
    }
    let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 1e-6 {
        clip.playback_rate as f64
    } else {
        1.0
    };
    let span = clip.length_sec.max(0.0) * rate;
    if clip.reversed {
        clip.source_start_sec = clip.source_end_sec - span;
    } else {
        clip.source_end_sec = clip.source_start_sec + span;
    }
}

/// 对 Clip 的**全部 Take** 应用非 Loop 存储窗口规范化。
///
/// 与 `normalize_nonloop_source_window` 同一不变式（存储窗口 == 消费窗口），
/// 但每个 Take 的消费速率是组合速率（clip 倍率 × take 速率）。作用：打开
/// 工程时自愈 inactive take 的陈旧/发散源窗口 —— 消费端虽按派生模型不受
/// 影响，但原始字段会流向前端 take-lane 显示与 REAPER 导出的 SECTION 计算，
/// 保留陈旧值会造成显示/导出与实际可听内容脱节。Loop take 的字段承载锚点
/// 相位，不触碰。
pub(crate) fn normalize_nonloop_all_take_windows(clip: &mut Clip) {
    let clip_rate = if clip.clip_playback_rate.is_finite() && clip.clip_playback_rate > 1e-6 {
        clip.clip_playback_rate as f64
    } else {
        1.0
    };
    for take in &mut clip.takes {
        if take.loop_enabled {
            continue;
        }
        let take_rate = if take.playback_rate.is_finite() && take.playback_rate > 1e-6 {
            take.playback_rate as f64
        } else {
            1.0
        };
        let span = clip.length_sec.max(0.0) * clip_rate * take_rate;
        if take.reversed {
            take.source_start_sec = take.source_end_sec - span;
        } else {
            take.source_end_sec = take.source_start_sec + span;
        }
    }
}

/// Loop（循环源）下**音符内容**的回绕周期（源域秒）。
///
/// 音频 clip 的实际声音按整个媒体时长 D 回绕（mix / snapshot / mixdown 的
/// 锚点数学），因此其派生音符（音高编辑回写的 `midi_note_data`）也必须按
/// D 平铺才能与音频保持同相位。音高参考块（Pitch Reference，无源媒体）
/// 与普通媒体 Clip 完全一致：D = 音符内容总时长（最大结束时间），回绕
/// 整个内容；仅当连音符内容都无法确定时才退化为窗口跨度。
///
/// 返回 `None` 表示未启用 Loop 或周期无效 —— 调用方应走非循环路径。
pub(crate) fn clip_loop_cycle_span_sec(clip: &Clip) -> Option<f64> {
    if !clip.loop_enabled {
        return None;
    }
    let span = clip_source_media_duration_sec(clip)
        .unwrap_or_else(|| (clip.source_end_sec - clip.source_start_sec).abs());
    if span.is_finite() && span > 1e-9 {
        Some(span)
    } else {
        None
    }
}

/// Loop（循环源）下单个音符事件映射到时间线帧坐标的出现位置。
///
/// 与音频渲染的 floor_mod 锚点回绕逐帧一致：
///   正放 s(u) = floor_mod(source_start + u, D)，u 为已消费源秒；
///   倒放 s(u) = floor_mod(min(source_end, D) − u, D)。
/// 音符 [n0, n1) 在每个周期内恰好占据一段连续区间：
///   正放首现于 u₀ = (n0 − source_start) mod D；
///   倒放首现于 u₀ = (min(source_end,D) − n1) mod D。
/// 当 D 等于窗口跨度且音符落在窗口内时，本函数与既有的"窗口内相对偏移 +
/// 镜像"算法完全等价（严格泛化，不改变既有用例）。
///
/// 返回 `(首个出现的起始帧, 单次出现长度帧, 重复周期帧)`；`None` 表示
/// 未启用 Loop、周期无效或音符为空 —— 调用方走非循环路径。
pub(crate) struct LoopNotePlacement {
    pub first_start_frame: usize,
    pub len_frames: usize,
    pub cycle_frames: usize,
}

/// `place_note_occurrence_in_loop` 的原始参数核心：供无法直接持有 `&Clip`
/// 的调用方（如 `build_fallback_pitch_from_midi`）复用同一套锚点数学。
/// 返回 `(首个出现的起始帧, 单次出现长度帧, 重复周期帧)`。
pub(crate) fn place_note_occurrence_frames(
    reversed: bool,
    playback_rate: f64,
    frame_period_ms: f64,
    fwd_anchor_sec: f64,
    rev_anchor_end_sec: f64,
    cycle_src_sec: f64,
    note_start_sec: f64,
    note_end_sec: f64,
) -> Option<LoopNotePlacement> {
    if !(cycle_src_sec.is_finite() && cycle_src_sec > 1e-9) {
        return None;
    }
    let len_sec = note_end_sec - note_start_sec;
    if !(len_sec > 1e-9) {
        return None;
    }
    let rate = if playback_rate.is_finite() && playback_rate > 1e-6 {
        playback_rate
    } else {
        1.0
    };
    let fp = frame_period_ms.max(0.1);
    // 源域秒 → 时间线帧：sec / rate 秒的时间线时长 → ×1000/fp 帧。
    let to_frames = |source_sec: f64| -> usize {
        (((source_sec / rate) * 1000.0 / fp).round().max(0.0)) as usize
    };
    let u0 = if reversed {
        (rev_anchor_end_sec - note_end_sec).rem_euclid(cycle_src_sec)
    } else {
        // 正放锚点取**原始** source_start（可为负），与音频回绕保持一致。
        (note_start_sec - fwd_anchor_sec).rem_euclid(cycle_src_sec)
    };
    Some(LoopNotePlacement {
        first_start_frame: to_frames(u0),
        len_frames: to_frames(len_sec),
        cycle_frames: to_frames(cycle_src_sec).max(1),
    })
}

pub(crate) fn place_note_occurrence_in_loop(
    clip: &Clip,
    note_start_sec: f64,
    note_end_sec: f64,
    frame_period_ms: f64,
) -> Option<LoopNotePlacement> {
    // 周期来源决定倒放锚点的 clamp 规则：
    // - 周期 = 媒体时长（已知元数据）：锚点 clamp 到媒体时长（与音频渲染的
    //   min(source_end, D) 一致，防止异常超界 source_end 错相位）；
    // - 周期退化为窗口跨度（纯 MIDI / 无元数据）：**不能**用跨度 clamp ——
    //   窗口 [2,7] 的跨度 5 < end=7 会把倒放锚点从 7 改写成 5，使所有音符
    //   相位平移 ss=2s。此时保持原始 source_end（窗口 exclusive 末端）。
    let media_total = clip_source_media_duration_sec(clip).filter(|d| *d > 1e-9);
    let cycle_src_sec = clip_loop_cycle_span_sec(clip)?;
    // 倒放锚点与音频路径（mix/snapshot/mixdown）同约定：min(source_end, D)
    // 后**不做** max(0) —— 负 source_end（slip/split + Loop 组合可达）由
    // rem_euclid 统一环绕，避免音符域与音频域出现恒定相位差。
    let rev_anchor_end = match media_total {
        Some(d) => clip.source_end_sec.min(d),
        None => clip.source_end_sec,
    };
    place_note_occurrence_frames(
        clip.reversed,
        clip.playback_rate as f64,
        frame_period_ms,
        // 正放锚点：原始 source_start（可为负，floor_mod 环绕）。
        clip.source_start_sec,
        rev_anchor_end,
        cycle_src_sec,
        note_start_sec,
        note_end_sec,
    )
}

/// 把区间 `[start, end)` 按模 `modulus` 的回绕边界拆分成若干不跨界的子区间，
/// 每个子区间以"周期内相位"坐标返回（值域 `[0, modulus)`）。
///
/// 用于 Loop（循环源）把超出一个循环周期的内容（如 MIDI 音符时间）
/// 回绕映射回源窗口，并在回绕边界处拆分跨越边界的音符。
fn split_range_into_periods(start: f64, end: f64, modulus: f64) -> Vec<(f64, f64)> {
    let mut segs: Vec<(f64, f64)> = Vec::new();
    if !(end > start) || !(modulus > 1e-9) || !start.is_finite() || !end.is_finite() {
        return segs;
    }
    let mut cursor = start;
    let mut guard = 0usize;
    while cursor < end - 1e-9 && guard < 1_000_000 {
        guard += 1;
        let phase = cursor.rem_euclid(modulus);
        let dist_to_boundary = modulus - phase;
        let seg_end = (cursor + dist_to_boundary).min(end);
        if seg_end - cursor > 1e-9 {
            segs.push((phase, phase + (seg_end - cursor)));
        }
        cursor = seg_end;
        if dist_to_boundary <= 1e-12 {
            break;
        }
    }
    segs
}

fn is_default_frame_period(value: &f64) -> bool {
    *value == default_frame_period_ms()
}

fn btree_map_string_track_params_is_empty(value: &BTreeMap<String, TrackParamsState>) -> bool {
    value.is_empty()
}

fn hash_set_string_is_empty(value: &HashSet<String>) -> bool {
    value.is_empty()
}

/// 内置音阶键名 → 音级集合（未知键名返回 None）。
pub(crate) fn scale_notes_for_key(scale: &str) -> Option<Vec<u8>> {
    let notes = match scale {
        "C" => vec![0, 2, 4, 5, 7, 9, 11],
        "Db" => vec![1, 3, 5, 6, 8, 10, 0],
        "D" => vec![2, 4, 6, 7, 9, 11, 1],
        "Eb" => vec![3, 5, 7, 8, 10, 0, 2],
        "E" => vec![4, 6, 8, 9, 11, 1, 3],
        "F" => vec![5, 7, 9, 10, 0, 2, 4],
        "Gb" => vec![6, 8, 10, 11, 1, 3, 5],
        "G" => vec![7, 9, 11, 0, 2, 4, 6],
        "Ab" => vec![8, 10, 0, 1, 3, 5, 7],
        "A" => vec![9, 11, 1, 2, 4, 6, 8],
        "Bb" => vec![10, 0, 2, 3, 5, 7, 9],
        "B" => vec![11, 1, 3, 4, 6, 8, 10],
        _ => return None,
    };
    Some(notes)
}

/// 全部内置音阶键名（与 scale_notes_for_key 对应）。
pub(crate) const SCALE_KEYS: [&str; 12] = [
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
];

/// 由音级集合反查内置音阶键名（归一化后完全一致才匹配）。
pub(crate) fn key_for_scale_notes(notes: &[u8]) -> Option<String> {
    let mut normalized: Vec<u8> = notes.iter().map(|v| v % 12).collect();
    normalized.sort_unstable();
    normalized.dedup();
    for key in SCALE_KEYS {
        if let Some(n) = scale_notes_for_key(key) {
            let mut m: Vec<u8> = n.iter().map(|v| v % 12).collect();
            m.sort_unstable();
            m.dedup();
            if m == normalized {
                return Some(key.to_string());
            }
        }
    }
    None
}

/// 工程音阶 → Tempo Map 初始点音阶数据（初始点即工程基准记录，含自定义音阶名）。
pub(crate) fn tempo_scale_data_from_project(p: &ProjectState) -> TempoScaleData {
    if p.use_custom_scale {
        if let Some(custom) = p.custom_scale.as_ref() {
            return TempoScaleData {
                key: None,
                name: Some(custom.name.clone()),
                notes: Some(custom.notes.clone()),
            };
        }
    }
    TempoScaleData {
        key: Some(p.base_scale.clone()),
        name: None,
        notes: None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PitchAnalysisAlgo {
    WorldDll,
    #[default]
    NsfHifiganOnnx,
    #[serde(rename = "vslib")]
    VocalShifterVslib,
    None,
    #[serde(other)]
    Unknown,
}

/// 合成链路类型，独立于 PitchAnalysisAlgo，面向声码器选择。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthPipelineKind {
    WorldVocoder,
    NsfHifiganOnnx,
    /// VocalShifter vslib 原生声码器（仅限 Windows，需 vslib feature）。
    #[cfg(feature = "vslib")]
    VocalShifterVslib,
}

impl SynthPipelineKind {
    /// 从 Track 的分析算法推断合成链路类型。
    pub fn from_track_algo(algo: &PitchAnalysisAlgo) -> Self {
        match algo {
            PitchAnalysisAlgo::NsfHifiganOnnx => Self::NsfHifiganOnnx,
            #[cfg(feature = "vslib")]
            PitchAnalysisAlgo::VocalShifterVslib => Self::VocalShifterVslib,
            _ => Self::WorldVocoder,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrackParamsState {
    #[serde(
        default = "default_frame_period_ms",
        skip_serializing_if = "is_default_frame_period"
    )]
    pub frame_period_ms: f64,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pitch_orig: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pitch_edit: Vec<f32>,

    #[serde(default, skip_serializing_if = "is_false")]
    pub pitch_edit_user_modified: bool,

    /// 是否有活跃的音高参考块（非静音 MIDI clip）在此轨道组中
    #[serde(skip)]
    pub has_pitch_adjustment_active: bool,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tension_orig: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tension_edit: Vec<f32>,

    #[serde(skip)]
    pub pitch_orig_key: Option<String>,

    /// 由 Reaper 导入产生的待应用音高偏移（半音）。
    /// 当 pitch_orig 分析完成后，pitch_edit = pitch_orig + 此偏移。
    #[serde(skip)]
    pub pending_pitch_offset: Option<Vec<f32>>,

    /// 自动化曲线（key = ParamDescriptor::id）。
    /// 多数曲线是声码器专属的；`volume` / `pan` 是所有算法共通的混音参数，
    /// 切换算法时保留同一条曲线。缺失 key = 使用参数默认值。
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_curves: HashMap<String, Vec<f32>>,

    /// 声码器专属静态参数（key = ParamDescriptor::id，值为枚举整数转 f64）。
    /// 例："synth_mode" = 1.0（SYNTHMODE_MF）。
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_params: HashMap<String, f64>,
}

impl TrackParamsState {
    /// 保存工程文件时，没有任何用户数据需要持久化的状态。
    ///
    /// 全零/空的 pitch、tension 和自动化曲线在反序列化后与默认状态语义一致，
    /// 因此整条 root-track 参数记录都可以省略。
    pub fn is_empty_project_data(&self) -> bool {
        self.pitch_orig.is_empty()
            && self.pitch_edit.is_empty()
            && !self.pitch_edit_user_modified
            && self.tension_orig.is_empty()
            && self.tension_edit.is_empty()
            && self.extra_curves.is_empty()
            && self.extra_params.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LinkedParamCurvesPayload {
    #[serde(
        default = "default_frame_period_ms",
        skip_serializing_if = "is_default_frame_period"
    )]
    pub frame_period_ms: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pitch_edit: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tension_edit: Vec<f32>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_curves: HashMap<String, Vec<f32>>,
}

/// 单个剪辑拉伸前后的几何范围（秒），用于"锁定参数线"的时域映射。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StretchLinkedRangeSec {
    pub old_start_sec: f64,
    pub old_length_sec: f64,
    pub new_start_sec: f64,
    pub new_length_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ClipFormantMorph {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_formant_target_f1_hz")]
    pub target_f1_hz: f64,
    #[serde(default = "default_formant_target_f2_hz")]
    pub target_f2_hz: f64,
    #[serde(default = "default_formant_strength")]
    pub strength: f64,
}

impl Default for ClipFormantMorph {
    fn default() -> Self {
        Self {
            enabled: false,
            target_f1_hz: default_formant_target_f1_hz(),
            target_f2_hz: default_formant_target_f2_hz(),
            strength: default_formant_strength(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveClipPayload {
    pub clip_id: String,
    pub start_sec: f64,
    #[serde(default)]
    pub track_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BulkClipStatePatch {
    pub clip_id: String,
    #[serde(flatten)]
    pub patch: ClipStatePatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DuplicateClipsTrackMode {
    SameTrack,
    OffsetTracks { offset: i32 },
    ExplicitMapping { mapping: HashMap<String, String> },
    NewTracks,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuplicateClipsBulkPayload {
    pub source_clip_ids: Vec<String>,
    pub delta_sec: f64,
    pub track_mode: DuplicateClipsTrackMode,
    #[serde(default)]
    pub copy_linked_params: bool,
    #[serde(default)]
    pub select_created_clips: bool,
    #[serde(default)]
    pub apply_auto_crossfade: bool,
    #[serde(default)]
    pub place_on_selected_track: bool,
    pub rename_copies: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateClipTemplatePayload {
    pub track_id: String,
    pub name: String,
    pub start_sec: f64,
    pub length_sec: f64,
    #[serde(default)]
    pub source_clip_id: Option<String>,
    pub source_path: Option<String>,
    pub gain: Option<f32>,
    pub muted: Option<bool>,
    pub source_start_sec: Option<f64>,
    pub source_end_sec: Option<f64>,
    pub playback_rate: Option<f32>,
    pub reversed: Option<bool>,
    #[serde(default)]
    pub loop_enabled: Option<bool>,
    #[serde(default)]
    pub snap_offset_sec: Option<f64>,
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_shape: Option<f64>,
    pub fade_out_shape: Option<f64>,
    pub fade_in_dir: Option<f64>,
    pub fade_out_dir: Option<f64>,
    #[serde(default)]
    pub auto_fade_in_sec: Option<f64>,
    #[serde(default)]
    pub auto_fade_out_sec: Option<f64>,
    pub linked_params: Option<LinkedParamCurvesPayload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub midi_note_data: Option<Vec<MidiNoteEvent>>,
    #[serde(default)]
    pub midi_fill_gaps: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateClipsBulkPayload {
    pub templates: Vec<CreateClipTemplatePayload>,
    #[serde(default)]
    pub select_created_clips: bool,
}

/// Take 级拉伸标记（Phase 1 仅持久化/导入导出，引擎消费在后续阶段接入）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipStretchMarker {
    pub offset_sec: f64,
    pub position_sec: f64,
    pub velocity_change: f64,
}

/// Take 级包络集合（Phase 1 仅预留结构，曲线按 Take 绑定）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipTakeEnvelopeSet {
    pub frame_period_ms: f64,
    pub curves: BTreeMap<String, Vec<f32>>,
}

/// Clip 的内容层：对齐 REAPER Take / VEGAS Take。
///
/// 磁盘工程中这是媒体相关字段的唯一权威来源；`Clip` 上的同名字段是
/// active take 的内存投影，通过 `normalize_takes()` 物化，并在工程保存时
/// 以 `#[serde(skip_serializing)]` 省略。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipTake {
    pub id: String,
    #[serde(default)]
    pub name: String,

    /// Take 级音量倍率（REAPER TAKEVOLPAN[0]；旧工程的 Clip.gain 迁移到此处）。
    #[serde(default = "default_gain")]
    pub gain: f32,

    // ── 媒体引用与元数据 ──
    #[serde(default)]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path_relative: Option<String>,
    #[serde(default)]
    pub duration_sec: Option<f64>,
    #[serde(default)]
    pub duration_frames: Option<u64>,
    #[serde(default)]
    pub source_sample_rate: Option<u32>,
    /// 源文件内容指纹（随工程持久化，用于外部文件变更检测与重匹配）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file_fingerprint: Option<u64>,
    /// 运行时文件元数据（不持久化），用于本会话的外部文件变更检测。
    #[serde(skip)]
    pub source_file_mtime: Option<u64>,
    #[serde(skip)]
    pub source_file_size: Option<u64>,

    /// 波形预览与音高范围缓存（保存前由 `prepare_timeline_for_project_save` 清空）。
    #[serde(default)]
    pub waveform_preview: Option<Vec<f32>>,
    #[serde(default)]
    pub pitch_range: Option<PitchRange>,

    // ── 内容编辑参数 ──
    #[serde(alias = "trim_start_sec", default)]
    pub source_start_sec: f64,
    #[serde(alias = "trim_end_sec", default)]
    pub source_end_sec: f64,
    #[serde(default = "default_playback_rate")]
    pub playback_rate: f32,
    #[serde(default)]
    pub reversed: bool,
    #[serde(default)]
    pub loop_enabled: bool,

    // ── MIDI 内容（无音频源时） ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub midi_note_data: Option<Vec<MidiNoteEvent>>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub midi_fill_gaps: bool,

    // ── Take 级拉伸标记 / 包络（预留，Phase 2 起消费） ──
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stretch_markers: Vec<ClipStretchMarker>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelopes: Option<ClipTakeEnvelopeSet>,
}

impl ClipTake {
    /// 从 Clip 的 active-take 内存投影生成一个 Take。
    ///
    /// - takes 为空（旧工程 / 新建扁平 Clip）：`clip.playback_rate` 就是原
    ///   Take 的速率；
    /// - takes 非空：`clip.playback_rate` 是 Item×Take 的有效投影，反除
    ///   Item 速率以保留 Take 自身速率。
    pub fn from_clip(clip: &Clip) -> Self {
        let playback_rate = if clip.takes.is_empty() {
            clip.playback_rate
        } else {
            let clip_rate = if clip.clip_playback_rate.is_finite() && clip.clip_playback_rate > 1e-6
            {
                clip.clip_playback_rate
            } else {
                1.0
            };
            clip.playback_rate / clip_rate
        };
        Self {
            id: format!("{}_take_1", clip.id),
            name: if clip.takes.is_empty() {
                clip.name.clone()
            } else {
                clip.active_take().name.clone()
            },
            gain: clip.gain,
            source_path: clip.source_path.clone(),
            source_path_relative: clip.source_path_relative.clone(),
            duration_sec: clip.duration_sec,
            duration_frames: clip.duration_frames,
            source_sample_rate: clip.source_sample_rate,
            source_file_fingerprint: clip.source_file_fingerprint,
            source_file_mtime: clip.source_file_mtime,
            source_file_size: clip.source_file_size,
            waveform_preview: clip.waveform_preview.clone(),
            pitch_range: clip.pitch_range.clone(),
            source_start_sec: clip.source_start_sec,
            source_end_sec: clip.source_end_sec,
            playback_rate,
            reversed: clip.reversed,
            loop_enabled: clip.loop_enabled,
            midi_note_data: clip.midi_note_data.clone(),
            midi_fill_gaps: clip.midi_fill_gaps,
            stretch_markers: Vec::new(),
            envelopes: None,
        }
    }

    /// 把该 Take 的内容写入 Clip 的 active-take 内存投影。
    /// `Clip.playback_rate` 始终保存 Clip×Take 的有效播放倍率。
    pub fn apply_to_clip(&self, clip: &mut Clip) {
        let clip_rate = if clip.clip_playback_rate.is_finite() && clip.clip_playback_rate > 1e-6 {
            clip.clip_playback_rate
        } else {
            1.0
        };
        clip.gain = self.gain;
        clip.source_path = self.source_path.clone();
        clip.source_path_relative = self.source_path_relative.clone();
        clip.duration_sec = self.duration_sec;
        clip.duration_frames = self.duration_frames;
        clip.source_sample_rate = self.source_sample_rate;
        clip.source_file_fingerprint = self.source_file_fingerprint;
        clip.source_file_mtime = self.source_file_mtime;
        clip.source_file_size = self.source_file_size;
        clip.waveform_preview = self.waveform_preview.clone();
        clip.pitch_range = self.pitch_range.clone();
        clip.source_start_sec = self.source_start_sec;
        clip.source_end_sec = self.source_end_sec;
        clip.playback_rate = clip_rate * self.playback_rate;
        clip.reversed = self.reversed;
        clip.loop_enabled = self.loop_enabled;
        clip.midi_note_data = self.midi_note_data.clone();
        clip.midi_fill_gaps = self.midi_fill_gaps;
    }
}

/// Take 自身媒体内容的总时长；Loop 回绕周期优先使用该值。
fn clip_take_media_duration_sec(take: &ClipTake) -> Option<f64> {
    if let (Some(frames), Some(sample_rate)) = (take.duration_frames, take.source_sample_rate) {
        if sample_rate > 0 && frames > 0 {
            return Some(frames as f64 / sample_rate as f64);
        }
    }
    if let Some(duration) = take.duration_sec.filter(|d| d.is_finite() && *d > 0.0) {
        return Some(duration);
    }
    if let Some(notes) = take.midi_note_data.as_ref() {
        let max_end = notes.iter().map(|n| n.end_sec).fold(0.0f64, f64::max);
        if max_end.is_finite() && max_end > 0.0 {
            return Some(max_end);
        }
    }
    None
}

/// 方向翻转（正放 ↔ 倒放）时保持**消费内容**不变的源窗口/锚点换算。
///
/// 非 Loop 的消费窗口正放为 `[ss, ss+span)`、倒放为 `[se−span, se)`，翻转
/// 方向必须以**翻转前的消费窗口**为准推导新方向的锚点字段 —— 派生窗口
/// 模型下非锚定方向的存储字段不参与消费数学、可能是陈旧值，直接翻转布尔
/// 会让消费内容跳变（如裁剪过的 Clip 倒放后播到陈旧 se 所指的文件末段）：
///
///   - 翻为倒放：`se := ss + span`（原正放消费窗口的终点成为倒放锚点）；
///   - 翻为正放：`ss := se − span`（原倒放消费窗口的起点成为正放锚点）。
///
/// Loop 的字段承载回绕锚点（引擎正放自 `mod(ss, D)` 升、倒放自 `mod(se, D)`
/// 降），换算同样以原方向的消费区间为准：
///
///   - 翻为倒放：原正放自锚点升 span 秒，消费终点为 `ss + span`；倒放自
///     锚点**下降**，故新锚 = `mod(ss + span, D)` —— 直接取 `mod(ss, D)`
///     会从原正放的起点降奏，播到锚点下方的另一段内容；
///   - 翻为正放：原倒放自锚点降 span 秒，消费起点为 `se − span`；正放自
///     锚点**上升**，故新锚 = `mod(se − span, D)`。
///
/// Loop 锚点按整文件回绕归一 —— 引擎对倒放锚先 `min(se, D)` 再 rem_euclid，
/// 存储 se 必须已落在 [0, D) 内，否则超界锚点被钳到 D 会错相。媒体时长未知
/// 时退化为原始字段直算（引擎侧 rem_euclid 仍会回绕）。
///
/// `span_sec` = length × clip_rate × take_rate（该 Take 的组合消费速率）。
pub(crate) fn flip_direction_source_window(
    flip_to_reversed: bool,
    loop_enabled: bool,
    span_sec: f64,
    media_total_sec: Option<f64>,
    source_start_sec: &mut f64,
    source_end_sec: &mut f64,
) {
    if loop_enabled {
        match media_total_sec {
            Some(d) if d.is_finite() && d > 1e-9 => {
                if flip_to_reversed {
                    *source_end_sec = (*source_start_sec + span_sec).rem_euclid(d);
                } else {
                    *source_start_sec = (*source_end_sec - span_sec).rem_euclid(d);
                }
            }
            _ => {
                if flip_to_reversed {
                    *source_end_sec = *source_start_sec + span_sec;
                } else {
                    *source_start_sec = *source_end_sec - span_sec;
                }
            }
        }
        return;
    }
    if flip_to_reversed {
        *source_end_sec = *source_start_sec + span_sec;
    } else {
        *source_start_sec = *source_end_sec - span_sec;
    }
}

/// [`flip_direction_source_window`] 的 Take 便捷封装：span 按该 Take 的
/// 组合消费速率（clip_rate × take.playback_rate）计算。在改写
/// `take.reversed` **之前**调用 —— 内部按“翻转前方向”读取消费窗口。
pub(crate) fn flip_take_playback_direction(take: &mut ClipTake, length_sec: f64, clip_rate: f64) {
    let take_rate = if take.playback_rate.is_finite() && take.playback_rate > 1e-6 {
        take.playback_rate as f64
    } else {
        1.0
    };
    let clip_rate = if clip_rate.is_finite() && clip_rate > 1e-6 {
        clip_rate
    } else {
        1.0
    };
    let span = length_sec.max(0.0) * clip_rate * take_rate;
    let media_total = clip_take_media_duration_sec(take);
    flip_direction_source_window(
        !take.reversed,
        take.loop_enabled,
        span,
        media_total,
        &mut take.source_start_sec,
        &mut take.source_end_sec,
    );
}

/// 对单个 Take 应用分割几何。
///
/// `clip_rate` 是 Clip 级倍率；每个 Take 的实际消费速率为
/// `clip_rate × take.playback_rate`。音频源窗口与 MIDI 音符都会按切割点
/// 分成左右两段；该函数总是执行，不受“同步编辑所有 Take”设置影响。
fn split_clip_take_window(
    take: &mut ClipTake,
    clip_rate: f64,
    left_len_sec: f64,
    right_len_sec: f64,
    is_right_side: bool,
) {
    // 先快照原始窗口：左右两侧必须都从同一个原始 Take 推导，
    // 不能在处理同一份字段时把左侧结果当作右侧输入。
    let orig_start = take.source_start_sec;
    let orig_end = take.source_end_sec;
    let take_rate = if take.playback_rate.is_finite() && take.playback_rate > 1e-6 {
        take.playback_rate as f64
    } else {
        1.0
    };
    let rate = (clip_rate.max(1e-6) * take_rate).max(1e-6);

    if take.loop_enabled {
        // 与旧版 Clip 分割完全一致：
        // - 左段保留原锚点与窗口，仅由容器长度截短；
        // - 右段锚点按左段消费量对完整媒体时长回绕。
        if is_right_side {
            let media_total =
                clip_take_media_duration_sec(take).unwrap_or_else(|| orig_end.max(orig_start));
            if media_total.is_finite() && media_total > 1e-9 {
                let mut consumed = left_len_sec * rate;
                consumed %= media_total;
                if consumed < 0.0 {
                    consumed += media_total;
                }
                if take.reversed {
                    let wrapped = (orig_end - consumed).rem_euclid(media_total);
                    take.source_end_sec = if wrapped <= 0.0 { media_total } else { wrapped };
                } else {
                    take.source_start_sec = (orig_start + consumed).rem_euclid(media_total);
                }
            }
        }
        // 左侧 Loop：不修改 source 窗口。
    } else if take.reversed {
        // 倒放消费窗口 [se−len·r, se)，锚定原始 se。
        if is_right_side {
            let new_end = orig_end - left_len_sec * rate;
            take.source_start_sec = new_end - right_len_sec * rate;
            take.source_end_sec = new_end;
        } else {
            take.source_start_sec = orig_end - left_len_sec * rate;
            take.source_end_sec = orig_end;
        }
    } else {
        // 正放派生窗口 [ss, ss+len·r)。
        if is_right_side {
            take.source_start_sec = orig_start + left_len_sec * rate;
            take.source_end_sec = take.source_start_sec + right_len_sec * rate;
        } else {
            take.source_start_sec = orig_start;
            take.source_end_sec = orig_start + left_len_sec * rate;
        }
    }

    // MIDI 音符坐标相对 Clip 起点；按对应侧保留并重定基。
    if let Some(notes) = take.midi_note_data.as_mut() {
        if is_right_side {
            for note in notes.iter_mut() {
                note.start_sec -= left_len_sec;
                note.end_sec -= left_len_sec;
            }
            notes.retain(|note| note.end_sec > 1e-9 && note.start_sec < right_len_sec - 1e-9);
        } else {
            notes.retain(|note| note.start_sec < left_len_sec - 1e-9 && note.end_sec > 1e-9);
        }
        let bound = if is_right_side {
            right_len_sec
        } else {
            left_len_sec
        };
        for note in notes.iter_mut() {
            note.start_sec = note.start_sec.max(0.0);
            note.end_sec = note.end_sec.min(bound.max(0.0));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub id: String,
    pub name: String,
    /// 轨道分组父级；`None` 表示根轨道。始终序列化以兼容旧版本读取。
    #[serde(default)]
    pub parent_id: Option<String>,
    pub order: i32,
    /// 轨道音量/静音/独奏/合成开关/分析算法等基础参数属于工程语义内容，
    /// 始终序列化，避免依赖"缺省 = 默认值"的隐式规则（默认值可能在跨版本间变化）。
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub solo: bool,
    #[serde(default = "default_gain")]
    pub volume: f32,

    #[serde(default)]
    pub compose_enabled: bool,

    #[serde(default)]
    pub pitch_analysis_algo: PitchAnalysisAlgo,

    /// 轨道主题色，hex 字符串，如 "#4f8ef7"
    #[serde(default)]
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    pub track_id: String,
    pub name: String,
    pub start_sec: f64,
    pub length_sec: f64,
    #[serde(default = "default_clip_color")]
    pub color: String,

    /// Take 集合：磁盘工程中媒体相关字段的权威来源。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub takes: Vec<ClipTake>,
    /// 当前活跃 take；旧工程反序列化后由 `normalize_takes()` 补齐。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_take_id: Option<String>,
    /// Clip 级播放倍率（对标 REAPER 的 Item 拉伸）。实际消费速率为
    /// `clip_playback_rate × active_take.playback_rate`。
    #[serde(default = "default_playback_rate")]
    pub clip_playback_rate: f32,

    // ────────────────────────────────────────────────────────────────────────
    // 以下媒体/内容字段是 active take 的内存投影：运行时兼容既有消费者，
    // 磁盘序列化已用 `skip_serializing` 省略（权威数据只保存在 `takes` 中）。
    // 修改后必须调用 `sync_take_from_flat()`；加载后调用 `normalize_takes()`。
    // ────────────────────────────────────────────────────────────────────────
    #[serde(default, skip_serializing)]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing)]
    pub source_path_relative: Option<String>,
    #[serde(default, skip_serializing)]
    pub duration_sec: Option<f64>,
    #[serde(default, skip_serializing)]
    pub duration_frames: Option<u64>,
    #[serde(default, skip_serializing)]
    pub source_sample_rate: Option<u32>,
    /// 文件导入时的 mtime（Unix 时间戳，秒），用于检测外部文件替换/删除。
    /// 仅在程序运行期间有效，不持久化到工程文件。
    #[serde(skip)]
    pub source_file_mtime: Option<u64>,
    /// 源文件大小（字节），与 mtime 一起作为第一层元数据比对。
    /// 仅在程序运行期间有效，不持久化到工程文件。
    #[serde(skip)]
    pub source_file_size: Option<u64>,
    /// 源文件内容指纹（active take 投影；权威值保存在对应 `ClipTake`）。
    #[serde(default, skip_serializing)]
    pub source_file_fingerprint: Option<u64>,
    /// 波形预览（active take 投影，保存前清空）。
    #[serde(default, skip_serializing)]
    pub waveform_preview: Option<Vec<f32>>,
    #[serde(default, skip_serializing)]
    pub pitch_range: Option<PitchRange>,

    /// 增益（active take 投影；Take 级音量）。
    #[serde(default = "default_gain", skip_serializing)]
    pub gain: f32,
    #[serde(default)]
    pub muted: bool,
    #[serde(alias = "trim_start_sec", default, skip_serializing)]
    pub source_start_sec: f64,
    #[serde(alias = "trim_end_sec", default, skip_serializing)]
    pub source_end_sec: f64,
    #[serde(default = "default_playback_rate", skip_serializing)]
    pub playback_rate: f32,
    #[serde(default, skip_serializing)]
    pub reversed: bool,
    /// Loop（循环源）属性，对齐 REAPER / VEGAS 的 item LOOP 语义：
    ///
    /// 启用后对**整个原始媒体文件**做模运算回绕（"循环原始音频文件"）：
    ///   正放 src(t) = floor_mod(source_start + t·rate, D)
    ///   倒放 src(t) = floor_mod(source_end   − t·rate, D)
    /// 其中 D = 完整媒体时长。例：媒体 10s、锚点 2s 时，clip 的
    /// [0,8) 对应源 2~10s，[8,18) 对应源 0~10s，以此类推。
    ///
    /// - 延伸/裁短等操作不受源媒体长度限制；向左延伸会回退锚点并环绕；
    /// - 该字段随工程文件持久化；旧版本工程缺失时按"为新的音频块启用循环"
    ///   设置迁移（见 open_project 的 v4 迁移逻辑）。
    #[serde(default, skip_serializing)]
    pub loop_enabled: bool,
    /// 吸附偏移（秒）：相对 Clip 起点的偏移，默认 0。与倒放无关 ——
    /// 倒放时它依然表示"距 Clip 起点偏移 X"的位置。
    /// 对标 REAPER / VEGAS 的 item snap offset；旧工程缺失时按 0 补齐。
    #[serde(default)]
    pub snap_offset_sec: f64,
    #[serde(default)]
    pub fade_in_sec: f64,
    #[serde(default)]
    pub fade_out_sec: f64,
    /// 淡入曲线类型（旧版命名枚举；仅作读取兼容保留，不再序列化写出）。
    #[serde(default = "default_fade_curve", skip_serializing)]
    pub fade_in_curve: String,
    /// 淡出曲线类型（旧版命名枚举；仅作读取兼容保留，不再序列化写出）。
    #[serde(default = "default_fade_curve", skip_serializing)]
    pub fade_out_curve: String,
    /// 淡入形状：REAPER 的浮点形状 id（整数 0..6 为标准七预设：
    /// 0=线性、1/2=轻微凸/凹、3/4=陡峭凸/凹、5/6=轻微/锐利 S；
    /// 小数变体（如 1.1、5.1）为 REAPER 扩展编码，原样透传保存。
    /// 对标 REAPER FADEIN 行首槽位）。默认 0（线性）。
    #[serde(default)]
    pub fade_in_shape: f64,
    /// 淡出形状（语义同 [`Clip::fade_in_shape`]）。
    #[serde(default)]
    pub fade_out_shape: f64,
    /// 淡入曲率，对标 REAPER `D_FADEINDIR`，范围 [-1, 1]；默认 0。
    #[serde(default)]
    pub fade_in_dir: f64,
    /// 淡出曲率，对标 REAPER `D_FADEOUTDIR`，范围 [-1, 1]；默认 0。
    #[serde(default)]
    pub fade_out_dir: f64,

    /// 自动交叉淡化长度（秒，由剪辑重叠派生；**不覆盖手动 fade**）。
    ///
    /// 对齐 REAPER 的存储方式：手动 fade（`fade_in_sec` / `fade_out_sec`）始终保留，
    /// 自动交叉淡化独立记录在 `auto_fade_*_sec`。渲染/显示使用“有效 fade”：
    /// `auto_fade_*_sec > 0` 时用自动值，否则用手动值。分离后自动值归 0，
    /// 手动 fade 自然恢复。
    ///
    /// 旧版本工程没有这两个字段（serde default = 0），有效 fade = 手动值，完全兼容。
    #[serde(default)]
    pub auto_fade_in_sec: f64,
    #[serde(default)]
    pub auto_fade_out_sec: f64,

    /// Clip 级别的声码器曲线覆盖（None = 使用 Track 级别的 extra_curves）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_curves: Option<HashMap<String, Vec<f32>>>,

    /// Clip 级别的声码器静态参数覆盖（None = 使用 Track 级别的 extra_params）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_params: Option<HashMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formant_morph: Option<ClipFormantMorph>,

    /// MIDI 音符数据（active take 投影；仅用于 MIDI clip，无音频源）。
    /// 音符时间相对于 clip 起点（0 = clip 起点）。
    #[serde(default, skip_serializing)]
    pub midi_note_data: Option<Vec<MidiNoteEvent>>,

    /// 是否在 pitch_orig 组装时填补 MIDI 音符之间的空隙。
    #[serde(default, skip_serializing)]
    pub midi_fill_gaps: bool,
}

/// 旧命名曲线枚举 → （REAPER 浮点形状 id, 曲率）。
///
/// 仅用于读取兼容迁移（v3 与早期开发版 v4 工程里的
/// `fade_in_curve`/`fade_out_curve` 字符串）。旧曲线只影响渲染装饰，
/// 音频引擎从未应用过，因此近似映射不构成听感回退：
/// - linear → 形状 0（线性）
/// - exponential（t² 起伏偏晚）→ 形状 2（轻微凹）
/// - logarithmic（√t 起伏偏早）→ 形状 1（轻微凸）
/// - sine / scurve（对称平滑）→ 形状 5（轻微 S）
pub fn legacy_curve_to_fade_spec(curve: &str) -> (f64, f64) {
    match curve {
        "exponential" => (2.0, 0.0),
        "logarithmic" => (1.0, 0.0),
        "sine" | "scurve" => (5.0, 0.0),
        _ => (0.0, 0.0),
    }
}

/// 把可能带负号的曲率夹紧到 REAPER 允许的范围 [-1, 1]。
pub fn clamp_fade_dir(dir: f64) -> f64 {
    dir.clamp(-1.0, 1.0)
}

/// 新版“分割过渡淡化曲线”预设 id → （形状 id, 淡入默认曲率, 淡出默认曲率）。
///
/// id 与前端 `reaperFade.ts` 的 `FADE_PRESETS` / `DEFAULT_FADE_DIR_BY_SHAPE`
/// 对应（四元组顺序：shape / dir-in / dir-out）。`"keep"`（分割后保留原
/// Clip 曲线，不修改）由调用方在转换前拦截为 `curve=None`，本函数只认
/// 预设 id；未知值返回 None（调用方按“不修改”处理）。
pub fn split_transition_curve_spec(curve: &str) -> Option<(f64, f64, f64)> {
    match curve {
        "linear" => Some((0.0, 0.0, 0.0)),
        "convexSlight" => Some((1.0, 0.0, 0.0)),
        "lateSlight" => Some((2.0, 1.0, -1.0)),
        "convexSharp" => Some((3.0, -1.0, 1.0)),
        "lateSharp" => Some((4.0, 1.0, -1.0)),
        "sSlight" => Some((5.0, 0.0, 0.0)),
        "sSharp" => Some((6.0, 0.0, 0.0)),
        _ => None,
    }
}

impl Clip {
    /// 读取期兼容：把旧命名曲线字符串换算成 (shape, dir)。
    ///
    /// 规则：字符串为空（VocalShifter 导入等哨兵）或新字段已由工程文件显式
    /// 提供且非默认时不动 —— 这里的实现采用"空字符串才视为未设置"的策略，
    /// 因为 `#[serde(default)]` 无法区分"缺省 0"与"显式 0"，而旧数据里
    /// 只要写过淡变就必然带有非空曲线字符串。
    pub fn reconcile_legacy_fade_fields(&mut self) {
        let (in_shape, in_dir) = legacy_curve_to_fade_spec(&self.fade_in_curve);
        if !self.fade_in_curve.is_empty() {
            self.fade_in_shape = in_shape;
            self.fade_in_dir = in_dir;
            self.fade_in_curve.clear();
        }
        let (out_shape, out_dir) = legacy_curve_to_fade_spec(&self.fade_out_curve);
        if !self.fade_out_curve.is_empty() {
            self.fade_out_shape = out_shape;
            self.fade_out_dir = out_dir;
            self.fade_out_curve.clear();
        }
        self.fade_in_dir = clamp_fade_dir(self.fade_in_dir);
        self.fade_out_dir = clamp_fade_dir(self.fade_out_dir);
    }

    /// 有效淡入长度：自动交叉淡化启用时用自动值，否则用手动值。
    pub fn effective_fade_in_sec(&self) -> f64 {
        if self.auto_fade_in_sec > 0.0 {
            self.auto_fade_in_sec
        } else {
            self.fade_in_sec
        }
    }

    /// 有效淡出长度：自动交叉淡化启用时用自动值，否则用手动值。
    pub fn effective_fade_out_sec(&self) -> f64 {
        if self.auto_fade_out_sec > 0.0 {
            self.auto_fade_out_sec
        } else {
            self.fade_out_sec
        }
    }

    /// 当前活跃 take 的下标；takes 为空时返回 0（调用方应先 `normalize_takes()`）。
    pub fn active_take_index(&self) -> usize {
        self.active_take_id
            .as_deref()
            .and_then(|id| self.takes.iter().position(|t| t.id == id))
            .unwrap_or(0)
    }

    /// 当前活跃 take。调用前必须保证 takes 非空（反序列化/构造边界统一调用
    /// `normalize_takes()`）。
    pub fn active_take(&self) -> &ClipTake {
        assert!(
            !self.takes.is_empty(),
            "active_take() on empty takes — normalize_takes() must run at load/construct boundaries"
        );
        let idx = self.active_take_index().min(self.takes.len() - 1);
        &self.takes[idx]
    }

    pub fn take(&self, take_id: &str) -> Option<&ClipTake> {
        self.takes.iter().find(|t| t.id == take_id)
    }

    /// 把 active-take 内存投影写回 `takes` 中的对应条目；takes 为空时
    /// 由投影生成一个 Take。用于旧工程反序列化与运行时字段修改后的同步。
    pub fn sync_take_from_flat(&mut self) {
        let mut projection = ClipTake::from_clip(self);
        if self.takes.is_empty() {
            self.active_take_id = Some(projection.id.clone());
            self.takes.push(projection);
            return;
        }
        let idx = self.active_take_index().min(self.takes.len() - 1);
        let existing = self.takes[idx].clone();
        projection.id = existing.id.clone();
        projection.name = existing.name.clone();
        projection.stretch_markers = existing.stretch_markers;
        projection.envelopes = existing.envelopes;
        if existing.source_file_mtime.is_some() {
            projection.source_file_mtime = existing.source_file_mtime;
        }
        if existing.source_file_size.is_some() {
            projection.source_file_size = existing.source_file_size;
        }
        if projection.source_file_fingerprint.is_none() {
            projection.source_file_fingerprint = existing.source_file_fingerprint;
        }
        self.active_take_id = Some(existing.id);
        self.takes[idx] = projection;
    }

    /// 统一规范化：旧工程（无 takes）由投影生成单 Take；新工程（有 takes）
    /// 把 active take 物化到投影。任何加载/合并/构造边界都应调用。
    pub fn normalize_takes(&mut self) {
        if self.takes.is_empty() {
            self.sync_take_from_flat();
            return;
        }
        let idx = self.active_take_index().min(self.takes.len() - 1);
        let take = self.takes[idx].clone();
        self.active_take_id = Some(take.id.clone());
        take.apply_to_clip(self);
    }

    /// 清空波形预览缓存（active 投影 + 全部 Take）。
    ///
    /// 片段导出/跨进程剪贴板在序列化前调用：只清投影不够 —— `takes` 是
    /// 磁盘权威，残留的 Take 预览会在下一次 normalize/sync 时回流，
    /// 且会随片段载荷一起序列化（体积膨胀）。
    pub fn clear_waveform_preview_caches(&mut self) {
        self.waveform_preview = None;
        for take in &mut self.takes {
            take.waveform_preview = None;
        }
    }

    /// 切换 active take，并把选中 take 物化到内存投影。
    pub fn switch_active_take(&mut self, take_id: &str) -> Result<(), String> {
        if self.takes.is_empty() {
            self.sync_take_from_flat();
        }
        let idx = self
            .takes
            .iter()
            .position(|t| t.id == take_id)
            .ok_or_else(|| format!("take not found: {}", take_id))?;
        self.active_take_id = Some(self.takes[idx].id.clone());
        let take = self.takes[idx].clone();
        take.apply_to_clip(self);
        Ok(())
    }

    /// 循环切换 active take。单 take 时为 no-op。
    pub fn cycle_active_take(&mut self, direction: i32) -> bool {
        if self.takes.len() <= 1 {
            return false;
        }
        let current = self.active_take_index();
        let next = if direction >= 0 {
            (current + 1) % self.takes.len()
        } else {
            (current + self.takes.len() - 1) % self.takes.len()
        };
        let next_id = self.takes[next].id.clone();
        self.active_take_id = Some(next_id.clone());
        if let Some(take) = self.take(&next_id).cloned() {
            take.apply_to_clip(self);
        }
        true
    }

    /// 为复制/粘贴/跨工程合并生成全新的 take id，并保持 active 指向。
    pub fn remap_take_ids(&mut self) {
        let old_active = self.active_take_id.clone();
        let mut new_active = None;
        for take in &mut self.takes {
            let old_id = take.id.clone();
            let new_id = new_id("take");
            if old_active.as_deref() == Some(old_id.as_str()) {
                new_active = Some(new_id.clone());
            }
            take.id = new_id;
        }
        if let Some(active) = new_active {
            self.active_take_id = Some(active);
        } else if !self.takes.is_empty() {
            self.active_take_id = Some(self.takes[0].id.clone());
        }
    }

    /// 追加一个 Take；返回其 id。
    pub fn add_take(&mut self, take: ClipTake) -> String {
        let id = if take.id.is_empty() {
            new_id("take")
        } else {
            take.id.clone()
        };
        let mut take = take;
        take.id = id.clone();
        self.takes.push(take);
        id
    }

    /// 删除 Take；最后一个 Take 不可删除。删除 active take 时自动切到第一个。
    pub fn remove_take(&mut self, take_id: &str) -> Result<(), String> {
        if self.takes.len() <= 1 {
            return Err("cannot remove the last take".to_string());
        }
        let idx = self
            .takes
            .iter()
            .position(|t| t.id == take_id)
            .ok_or_else(|| format!("take not found: {}", take_id))?;
        let removing_active = self.active_take_id.as_deref() == Some(take_id);
        self.takes.remove(idx);
        if removing_active {
            let first_id = self.takes[0].id.clone();
            self.switch_active_take(&first_id)?;
        }
        Ok(())
    }

    /// 重命名 Take。
    pub fn rename_take(&mut self, take_id: &str, name: &str) -> Result<(), String> {
        let found = self
            .takes
            .iter_mut()
            .find(|t| t.id == take_id)
            .ok_or_else(|| format!("take not found: {}", take_id))?;
        found.name = name.trim().to_string();
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClipStatePatch {
    pub name: Option<String>,
    pub start_sec: Option<f64>,
    pub length_sec: Option<f64>,
    pub gain: Option<f32>,
    pub muted: Option<bool>,
    pub source_start_sec: Option<f64>,
    pub source_end_sec: Option<f64>,
    pub playback_rate: Option<f32>,
    /// Clip 级播放倍率；修饰键拉伸修改的是这一层，不改动 Take 自身速率。
    pub clip_playback_rate: Option<f32>,
    pub reversed: Option<bool>,
    #[serde(default)]
    pub loop_enabled: Option<bool>,
    pub snap_offset_sec: Option<f64>,
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_shape: Option<f64>,
    pub fade_out_shape: Option<f64>,
    pub fade_in_dir: Option<f64>,
    pub fade_out_dir: Option<f64>,
    pub auto_fade_in_sec: Option<f64>,
    pub auto_fade_out_sec: Option<f64>,
    pub color: Option<String>,
    pub formant_morph: Option<ClipFormantMorph>,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeState {
    pub device: String,
    pub model_loaded: bool,
    pub audio_loaded: bool,
    pub has_synthesized: bool,

    pub synthesized_wav_path: Option<String>,
}

/// Tempo Map 变化点携带的音阶覆盖数据（None = 跟随工程音阶）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TempoScaleData {
    /// 内置音阶键名（如 "C"、"Db"）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// 自定义音阶名称。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 自定义音阶音级集合（0-11）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<Vec<u8>>,
}

/// Tempo Map 变化点（时间锚定：position_sec 绝对秒）。
/// 拍号为 None 表示“跟随之前的拍号”（0 位置初始点必须显式携带拍号）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TempoPointData {
    pub id: String,
    pub position_sec: f64,
    pub bpm: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerator: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denominator: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<TempoScaleData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineState {
    pub tracks: Vec<Track>,
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub selected_track_id: Option<String>,
    #[serde(default)]
    pub selected_clip_id: Option<String>,
    pub bpm: f64,
    #[serde(default)]
    pub playhead_sec: f64,
    pub project_sec: f64,

    #[serde(
        default,
        skip_serializing_if = "btree_map_string_track_params_is_empty"
    )]
    pub params_by_root_track: BTreeMap<String, TrackParamsState>,

    #[serde(
        default = "default_project_scale_notes",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub project_scale_notes: Vec<u8>,

    /// Tempo Map（None = 无 Tempo Map，使用全局 BPM/拍号/音阶）。
    /// 点按 position_sec 升序；第一个点必须位于 0。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo_map: Option<Vec<TempoPointData>>,

    #[serde(default = "default_next_track_order")]
    pub next_track_order: i32,

    #[serde(default, skip_serializing_if = "hash_set_string_is_empty")]
    pub disabled_group_ids: HashSet<String>,
}

/// Deserialization default (older project files lack the field): the next
/// track order starts at 1, matching `TimelineState::default()`.
fn default_next_track_order() -> i32 {
    1
}

const MAX_UNDO_HISTORY: usize = 100;

#[derive(Debug, Clone, Default)]
pub struct TimelineHistory {
    pub undo: VecDeque<TimelineState>,
    pub redo: Vec<TimelineState>,
}

#[derive(Debug, Clone)]
pub struct ProjectState {
    pub name: String,
    pub path: Option<String>,
    pub dirty: bool,
    pub recent: Vec<String>,
    pub notes_markdown: String,
    pub base_scale: String,
    pub use_custom_scale: bool,
    pub custom_scale: Option<CustomScale>,
    pub beats_per_bar: u32,
    /// 工程基准拍号分母（1/2/4/8/16/32）。
    pub time_signature_denominator: u32,
    pub grid_size: String,
    pub stretch_algorithm_override: Option<UserStretchAlgorithm>,
    pub hifigan_mel_stretch_override: Option<bool>,
    #[allow(dead_code)]
    pub allow_close: bool,
}

impl Default for ProjectState {
    fn default() -> Self {
        Self {
            name: "Untitled".to_string(),
            path: None,
            dirty: false,
            recent: Vec::new(),
            notes_markdown: String::new(),
            base_scale: "C".to_string(),
            use_custom_scale: false,
            custom_scale: None,
            beats_per_bar: 4,
            time_signature_denominator: 4,
            grid_size: "1/4".to_string(),
            stretch_algorithm_override: None,
            hifigan_mel_stretch_override: None,
            allow_close: false,
        }
    }
}

impl Default for TimelineState {
    fn default() -> Self {
        let track_id = "track_main".to_string();
        Self {
            tracks: vec![Track {
                id: track_id.clone(),
                name: "Main".to_string(),
                parent_id: None,
                order: 0,
                muted: false,
                solo: false,
                volume: 1.0,

                compose_enabled: false,
                pitch_analysis_algo: PitchAnalysisAlgo::default(),
                color: track_palette_color(0),
            }],
            clips: vec![],
            selected_track_id: Some(track_id),
            selected_clip_id: None,
            bpm: 120.0,
            playhead_sec: 0.0,
            project_sec: 32.0, // 64 beats @ 120 BPM = 32 sec

            params_by_root_track: BTreeMap::new(),
            project_scale_notes: default_project_scale_notes(),
            tempo_map: None,
            next_track_order: 1,
            disabled_group_ids: HashSet::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitTransitionMode {
    FadeOnly,
    ExtendOverlap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitTransitionDurationUnit {
    Seconds,
    Percent,
}

/// 分割过渡选项，由全局 UI 设置解析而来，并在 `split_clip_with_transition` 中使用。
#[derive(Debug, Clone)]
pub struct SplitTransitionOptions {
    pub enabled: bool,
    pub mode: SplitTransitionMode,
    pub duration_unit: SplitTransitionDurationUnit,
    pub duration_sec: f64,
    pub duration_percent: f64,
    pub curve: Option<String>,
    /// 延伸重叠模式下，是否同时为重叠区域设置淡入淡出。
    pub overlap_fades: bool,
}

/// 波纹编辑（自动跟进）模式，对应 REAPER 的 Ripple Editing。
///
/// - `Off`：关闭。编辑只影响被编辑对象本身，后续剪辑保持原位（默认，与 REAPER 默认一致）。
/// - `Track`：仅被编辑剪辑所在轨道上的后续剪辑一起平移（对应 REAPER“per selected track”）。
/// - `All`：所有轨道上位于编辑点之后的剪辑一起平移，保持多轨内容时间对齐（对应 REAPER“all tracks”）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RippleMode {
    Off,
    Track,
    All,
}

impl RippleMode {
    /// 从持久化字符串解析；未知值回退为 `Off`。
    pub fn from_str(value: &str) -> Self {
        match value {
            "track" => Self::Track,
            "all" => Self::All,
            _ => Self::Off,
        }
    }
}

impl TimelineState {
    fn clip_frame_bounds(
        &self,
        start_sec: f64,
        length_sec: f64,
        frame_period_ms: f64,
    ) -> (usize, usize) {
        let fp = frame_period_ms.max(0.1);
        let start_frame = ((start_sec.max(0.0) * 1000.0) / fp).floor() as usize;
        let frame_len = ((length_sec.max(0.0) * 1000.0) / fp).ceil().max(1.0) as usize;
        (start_frame, start_frame.saturating_add(frame_len))
    }

    fn root_track_kind(&self, root_track_id: &str) -> SynthPipelineKind {
        self.tracks
            .iter()
            .find(|track| track.id == root_track_id)
            .map(|track| SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
            .unwrap_or(SynthPipelineKind::WorldVocoder)
    }

    fn linked_param_frame_len(linked_params: &LinkedParamCurvesPayload) -> usize {
        let mut frame_len = linked_params
            .pitch_edit
            .len()
            .max(linked_params.tension_edit.len());
        for curve in linked_params.extra_curves.values() {
            frame_len = frame_len.max(curve.len());
        }
        frame_len.max(1)
    }

    fn clear_curve_range(
        curve: &mut Vec<f32>,
        required_len: usize,
        start_frame: usize,
        end_frame: usize,
        default_value: f32,
    ) {
        if curve.len() < required_len {
            curve.resize(required_len, default_value);
        }
        let start = start_frame.min(curve.len());
        let end = end_frame.min(curve.len());
        if start >= end {
            return;
        }
        for value in &mut curve[start..end] {
            *value = default_value;
        }
    }

    fn write_curve_range(
        curve: &mut Vec<f32>,
        required_len: usize,
        start_frame: usize,
        values: &[f32],
        default_value: f32,
    ) {
        if curve.len() < required_len {
            curve.resize(required_len, default_value);
        }
        for (offset, value) in values.iter().copied().enumerate() {
            let idx = start_frame.saturating_add(offset);
            if idx >= curve.len() {
                break;
            }
            curve[idx] = value;
        }
    }

    pub(crate) fn extract_linked_params_from_root_range(
        &mut self,
        root_track_id: &str,
        start_sec: f64,
        length_sec: f64,
    ) -> Option<LinkedParamCurvesPayload> {
        self.ensure_params_for_root(root_track_id);
        let frame_period_ms = self.frame_period_ms().max(0.1);
        let (start_frame, end_frame) =
            self.clip_frame_bounds(start_sec, length_sec, frame_period_ms);
        let kind = self.root_track_kind(root_track_id);
        let entry = self.params_by_root_track.get(root_track_id)?;

        // ⚠ 必须用带填充的切片，禁止 `get(start..end)`：
        // extra_curves 不是工程长度 —— set_param_frames 只把它增长到最后一次
        // 写入的帧，`ensure_params_for_root` 也不补齐它们。因此当剪辑范围
        // 超出曲线数组末尾（即剪辑包含该参数的最后一个有效点、其后全是
        // 默认值）时，`get()` 返回 None → 提取出空曲线 → 移动/复制时旧范围
        // 被清除而新范围什么都没写，用户数据被整体销毁（表现为"被初始化"）。
        // pitch/tension 始终保持工程长度所以未受影响；这里统一改为
        // 不足处补参考值（pitch/tension 补 0，extra 补参数默认值）。
        let frame_count = end_frame.saturating_sub(start_frame);
        let pitch_edit = if entry.pitch_edit_user_modified {
            Self::curve_slice(&entry.pitch_edit, start_frame, frame_count, 0.0)
        } else {
            Vec::new()
        };
        let tension_edit = Self::curve_slice(&entry.tension_edit, start_frame, frame_count, 0.0);
        let extra_curves = entry
            .extra_curves
            .iter()
            .map(|(param, curve)| {
                let default_value =
                    crate::renderer::automation_curve_default_value(kind, param).unwrap_or(0.0);
                (
                    param.clone(),
                    Self::curve_slice(curve, start_frame, frame_count, default_value),
                )
            })
            .collect();

        Some(LinkedParamCurvesPayload {
            frame_period_ms,
            pitch_edit,
            tension_edit,
            extra_curves,
        })
    }

    fn clear_linked_params_in_root_range(
        &mut self,
        root_track_id: &str,
        start_sec: f64,
        length_sec: f64,
        clear_pitch: bool,
        extra_curve_keys: Option<&[String]>,
    ) {
        self.ensure_params_for_root(root_track_id);
        let frame_period_ms = self.frame_period_ms().max(0.1);
        let (start_frame, end_frame) =
            self.clip_frame_bounds(start_sec, length_sec, frame_period_ms);
        let kind = self.root_track_kind(root_track_id);
        let Some(entry) = self.params_by_root_track.get_mut(root_track_id) else {
            return;
        };

        let required_len = entry.pitch_edit.len().max(end_frame);
        if clear_pitch {
            Self::clear_curve_range(
                &mut entry.pitch_edit,
                required_len,
                start_frame,
                end_frame,
                0.0,
            );
        }
        Self::clear_curve_range(
            &mut entry.tension_edit,
            required_len,
            start_frame,
            end_frame,
            0.0,
        );

        let keys = extra_curve_keys
            .map(|keys| keys.to_vec())
            .unwrap_or_else(|| entry.extra_curves.keys().cloned().collect());
        for key in keys {
            let default_value =
                crate::renderer::automation_curve_default_value(kind, &key).unwrap_or(0.0);
            let curve = entry
                .extra_curves
                .entry(key)
                .or_insert_with(|| vec![default_value; required_len]);
            Self::clear_curve_range(curve, required_len, start_frame, end_frame, default_value);
        }

        if clear_pitch {
            entry.pitch_edit_user_modified = true;
        }
    }

    pub(crate) fn apply_linked_params_to_root_range(
        &mut self,
        root_track_id: &str,
        start_sec: f64,
        linked_params: &LinkedParamCurvesPayload,
    ) {
        self.ensure_params_for_root(root_track_id);
        let frame_period_ms = self.frame_period_ms().max(0.1);
        let start_frame = ((start_sec.max(0.0) * 1000.0) / frame_period_ms).floor() as usize;
        let frame_len = Self::linked_param_frame_len(linked_params);
        let end_frame = start_frame.saturating_add(frame_len);
        let kind = self.root_track_kind(root_track_id);
        let has_pitch = !linked_params.pitch_edit.is_empty();

        // 粘贴/导入带 pitch_edit 的曲线时，目标根轨道必须进入合成模式，
        // 否则后续异步 pitch_orig 分析会跳过该轨道，声码器也会直接返回原声。
        if has_pitch {
            if let Some(track) = self
                .tracks
                .iter_mut()
                .find(|track| track.id == root_track_id)
            {
                track.compose_enabled = true;
            }
        }

        let target_existing_keys = self
            .params_by_root_track
            .get(root_track_id)
            .map(|entry| entry.extra_curves.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        let Some(entry) = self.params_by_root_track.get_mut(root_track_id) else {
            return;
        };

        let required_len = entry.pitch_edit.len().max(end_frame);
        if has_pitch {
            Self::clear_curve_range(
                &mut entry.pitch_edit,
                required_len,
                start_frame,
                end_frame,
                0.0,
            );
        }
        Self::clear_curve_range(
            &mut entry.tension_edit,
            required_len,
            start_frame,
            end_frame,
            0.0,
        );
        if has_pitch {
            Self::write_curve_range(
                &mut entry.pitch_edit,
                required_len,
                start_frame,
                &linked_params.pitch_edit,
                0.0,
            );
        }
        Self::write_curve_range(
            &mut entry.tension_edit,
            required_len,
            start_frame,
            &linked_params.tension_edit,
            0.0,
        );

        let mut all_keys = target_existing_keys;
        for key in linked_params.extra_curves.keys() {
            if !all_keys.iter().any(|existing| existing == key) {
                all_keys.push(key.clone());
            }
        }
        for key in &all_keys {
            let default_value =
                crate::renderer::automation_curve_default_value(kind, key).unwrap_or(0.0);
            let curve = entry
                .extra_curves
                .entry(key.clone())
                .or_insert_with(|| vec![default_value; required_len]);
            Self::clear_curve_range(curve, required_len, start_frame, end_frame, default_value);
        }
        for (key, values) in &linked_params.extra_curves {
            let default_value =
                crate::renderer::automation_curve_default_value(kind, key).unwrap_or(0.0);
            let curve = entry
                .extra_curves
                .entry(key.clone())
                .or_insert_with(|| vec![default_value; required_len]);
            Self::write_curve_range(curve, required_len, start_frame, values, default_value);
        }

        if has_pitch {
            entry.pitch_edit_user_modified = true;
        }
    }

    pub fn extract_clip_linked_params(
        &mut self,
        clip_id: &str,
    ) -> Option<LinkedParamCurvesPayload> {
        let clip = self.clips.iter().find(|clip| clip.id == clip_id)?;
        let root_track_id = self.resolve_root_track_id(&clip.track_id)?;
        self.extract_linked_params_from_root_range(&root_track_id, clip.start_sec, clip.length_sec)
    }

    pub fn apply_linked_params_to_clip(
        &mut self,
        clip_id: &str,
        linked_params: &LinkedParamCurvesPayload,
    ) -> bool {
        let Some(clip) = self.clips.iter().find(|clip| clip.id == clip_id) else {
            return false;
        };
        let Some(root_track_id) = self.resolve_root_track_id(&clip.track_id) else {
            return false;
        };
        self.apply_linked_params_to_root_range(&root_track_id, clip.start_sec, linked_params);
        true
    }

    /// 把旧范围内的曲线段取出（不足处补 pad 值），与 get_param_frames 的
    /// 越界回退语义一致。
    fn curve_slice(curve: &[f32], start: usize, count: usize, pad: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        for offset in 0..count {
            out.push(curve.get(start.saturating_add(offset)).copied().unwrap_or(pad));
        }
        out
    }

    /// 线性时域重采样。`pitch_zero_semantics` 时 0 视为"无声帧"，
    /// 不跨 0 插值（与前端旧 stretchLinkedParams 的行为逐位一致）。
    fn resample_curve(values: &[f32], target_len: usize, pitch_zero_semantics: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; target_len];
        if values.is_empty() || target_len == 0 {
            return out;
        }
        let old_max_idx = values.len() - 1;
        let new_max_idx: f32 = if target_len > 1 { (target_len - 1) as f32 } else { 1.0 };
        let ratio = old_max_idx as f32 / new_max_idx;
        for (i, slot) in out.iter_mut().enumerate() {
            let old_idxf = i as f32 * ratio;
            let lo = (old_idxf as usize).min(old_max_idx);
            let hi = if lo < old_max_idx { lo + 1 } else { old_max_idx };
            let frac = old_idxf - lo as f32;
            let lo_val = values[lo];
            let hi_val = values[hi];
            *slot = if pitch_zero_semantics {
                if lo_val == 0.0 && hi_val == 0.0 {
                    0.0
                } else if lo_val == 0.0 {
                    0.0
                } else if hi_val == 0.0 {
                    if frac < 0.5 {
                        lo_val
                    } else {
                        0.0
                    }
                } else {
                    lo_val + (hi_val - lo_val) * frac
                }
            } else {
                lo_val + (hi_val - lo_val) * frac
            };
        }
        out
    }

    /// 计算"旧范围中不被任何新范围覆盖"的帧段（闭区间），用于恢复参考值。
    /// 与前端旧 subtractIntervals 的批量语义一致：先写全部新范围，再恢复
    /// 未被覆盖的旧帧，避免相邻剪辑的恢复互相擦除新写入的值。
    fn uncovered_old_segments(
        mappings: &[(usize, usize, usize, usize)],
    ) -> Vec<(usize, usize)> {
        // (old_start, old_count, new_start, new_count)
        let mut segments: Vec<(usize, usize)> = Vec::new();
        for &(old_start, old_count, _, _) in mappings {
            let old_end = old_start.saturating_add(old_count).saturating_sub(1);
            let mut excluded: Vec<(usize, usize)> = mappings
                .iter()
                .map(|&(_, _, new_start, new_count)| {
                    (new_start, new_start.saturating_add(new_count).saturating_sub(1))
                })
                .filter(|&(s, e)| e >= old_start && s <= old_end)
                .collect();
            excluded.sort_unstable();
            let mut cursor = old_start;
            for (s, e) in excluded {
                if cursor > old_end {
                    break;
                }
                if s > cursor {
                    let end = (s - 1).min(old_end);
                    if end >= cursor {
                        segments.push((cursor, end));
                    }
                }
                cursor = cursor.max(e.saturating_add(1));
            }
            if cursor <= old_end {
                segments.push((cursor, old_end));
            }
        }
        segments
    }

    /// "锁定参数线"：剪辑拉伸后把旧范围内的参数曲线时域映射到新范围。
    ///
    /// 支持该剪辑范围涉及到的**所有**参数曲线——无论参数是否在 UI 中被
    /// 激活、是否有描述符、是否已有用户数据：
    /// - `pitch`：仅当用户手动编辑过时映射（否则由后端按剪辑几何重建）；
    /// - `tension`：始终映射；
    /// - `extra_curves`：映射所有已存在的曲线（volume/pan/气声/子轨道偏移等）。
    ///
    /// 批量语义：先写全部新范围，再把不被任何新范围覆盖的旧范围帧恢复为
    /// 参考值（pitch → pitch_orig；tension/extra → 参数默认值）。
    pub(crate) fn stretch_linked_params_in_root_range(
        &mut self,
        root_track_id: &str,
        mappings: &[StretchLinkedRangeSec],
    ) {
        if mappings.is_empty() {
            return;
        }
        self.ensure_params_for_root(root_track_id);
        let fp = self.frame_period_ms().max(0.1);

        // 秒 → 帧；跳过几何没有变化的映射。
        let mut frame_mappings: Vec<(usize, usize, usize, usize)> = Vec::new();
        for m in mappings {
            let (old_start_sec, old_len, new_start_sec, new_len) =
                (m.old_start_sec, m.old_length_sec, m.new_start_sec, m.new_length_sec);
            if !old_len.is_finite() || !new_len.is_finite() {
                continue;
            }
            let old_start = (old_start_sec.max(0.0) * 1000.0 / fp).round() as usize;
            let old_end =
                ((old_start_sec.max(0.0) + old_len.max(0.0)) * 1000.0 / fp).round() as usize;
            let new_start = (new_start_sec.max(0.0) * 1000.0 / fp).round() as usize;
            let new_end =
                ((new_start_sec.max(0.0) + new_len.max(0.0)) * 1000.0 / fp).round() as usize;
            let old_count = old_end.saturating_sub(old_start).max(1);
            let new_count = new_end.saturating_sub(new_start).max(1);
            if old_start == new_start && old_count == new_count {
                continue;
            }
            frame_mappings.push((old_start, old_count, new_start, new_count));
        }
        if frame_mappings.is_empty() {
            return;
        }

        let max_new_end = frame_mappings
            .iter()
            .map(|&(_, _, new_start, new_count)| new_start.saturating_add(new_count))
            .max()
            .unwrap_or(0);
        let kind = self.root_track_kind(root_track_id);
        let pitch_user_modified = self
            .params_by_root_track
            .get(root_track_id)
            .map(|entry| entry.pitch_edit_user_modified)
            .unwrap_or(false);

        let Some(entry) = self.params_by_root_track.get_mut(root_track_id) else {
            return;
        };

        // 写入前先取出旧范围快照，避免写/读互相覆盖。
        let pitch_sources: Vec<Vec<f32>> = if pitch_user_modified {
            frame_mappings
                .iter()
                .map(|&(old_start, old_count, _, _)| {
                    Self::curve_slice(&entry.pitch_edit, old_start, old_count, 0.0)
                })
                .collect()
        } else {
            Vec::new()
        };
        let tension_sources: Vec<Vec<f32>> = frame_mappings
            .iter()
            .map(|&(old_start, old_count, _, _)| {
                Self::curve_slice(&entry.tension_edit, old_start, old_count, 0.0)
            })
            .collect();
        let extra_keys: Vec<String> = entry.extra_curves.keys().cloned().collect();
        let extra_sources: Vec<(String, f32, Vec<Vec<f32>>)> = extra_keys
            .iter()
            .map(|key| {
                let default_value =
                    crate::renderer::automation_curve_default_value(kind, key).unwrap_or(0.0);
                let slices = frame_mappings
                    .iter()
                    .map(|&(old_start, old_count, _, _)| {
                        entry
                            .extra_curves
                            .get(key)
                            .map(|curve| {
                                Self::curve_slice(curve, old_start, old_count, default_value)
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                (key.clone(), default_value, slices)
            })
            .collect();

        let required_len = entry
            .pitch_edit
            .len()
            .max(entry.tension_edit.len())
            .max(max_new_end);

        // 写入阶段：先把所有新范围写满，恢复阶段才不会擦掉新值。
        if pitch_user_modified {
            if entry.pitch_edit.len() < required_len {
                entry.pitch_edit.resize(required_len, 0.0);
            }
            for (slice, &(.., new_start, new_count)) in
                pitch_sources.iter().zip(frame_mappings.iter())
            {
                let values = Self::resample_curve(slice, new_count, true);
                for (offset, value) in values.into_iter().enumerate() {
                    entry.pitch_edit[new_start.saturating_add(offset)] = value;
                }
            }
        }
        if entry.tension_edit.len() < required_len {
            entry.tension_edit.resize(required_len, 0.0);
        }
        for (slice, &(.., new_start, new_count)) in
            tension_sources.iter().zip(frame_mappings.iter())
        {
            let values = Self::resample_curve(slice, new_count, false);
            for (offset, value) in values.into_iter().enumerate() {
                entry.tension_edit[new_start.saturating_add(offset)] = value;
            }
        }
        for (key, default_value, slices) in &extra_sources {
            let curve = entry
                .extra_curves
                .entry(key.clone())
                .or_insert_with(|| vec![*default_value; required_len]);
            if curve.len() < required_len {
                curve.resize(required_len, *default_value);
            }
            for (slice, &(.., new_start, new_count)) in slices.iter().zip(frame_mappings.iter()) {
                let values = Self::resample_curve(slice, new_count, false);
                for (offset, value) in values.into_iter().enumerate() {
                    curve[new_start.saturating_add(offset)] = value;
                }
            }
        }

        // 恢复阶段：旧范围中不被任何新范围覆盖的帧还原为参考值。
        let segments = Self::uncovered_old_segments(&frame_mappings);
        if pitch_user_modified {
            for (start, end) in &segments {
                for idx in *start..=*end {
                    if idx >= entry.pitch_edit.len() {
                        break;
                    }
                    entry.pitch_edit[idx] = entry.pitch_orig.get(idx).copied().unwrap_or(0.0);
                }
            }
        }
        for (start, end) in &segments {
            for idx in *start..=(*end).min(entry.tension_edit.len().saturating_sub(1)) {
                entry.tension_edit[idx] = 0.0;
            }
        }
        for (key, default_value, _) in &extra_sources {
            if let Some(curve) = entry.extra_curves.get_mut(key) {
                for (start, end) in &segments {
                    for idx in *start..=(*end).min(curve.len().saturating_sub(1)) {
                        curve[idx] = *default_value;
                    }
                }
            }
        }

        // 与 restore_param_frames 一致：恢复后重算用户编辑标记。
        if pitch_user_modified {
            let len = entry.pitch_orig.len().min(entry.pitch_edit.len());
            let mut modified = false;
            for i in 0..len {
                let o = entry.pitch_orig[i];
                let e = entry.pitch_edit[i];
                if (e.is_finite() && e > 0.0)
                    && (!(o.is_finite() && o > 0.0) || (e - o).abs() > 1e-3)
                {
                    modified = true;
                    break;
                }
            }
            entry.pitch_edit_user_modified = modified;
        }

        // 与 apply_linked_params_to_root_range 一致：写入了用户 pitch 数据时，
        // 根轨道必须处于合成模式，否则曲线不会参与渲染。
        if pitch_user_modified {
            if let Some(track) = self
                .tracks
                .iter_mut()
                .find(|track| track.id == root_track_id)
            {
                track.compose_enabled = true;
            }
        }
    }

    pub fn resolve_root_track_id(&self, track_id: &str) -> Option<String> {
        if track_id.trim().is_empty() {
            return None;
        }
        let mut cur = track_id.to_string();
        let mut safety = 0;
        loop {
            let parent = self
                .tracks
                .iter()
                .find(|t| t.id == cur)
                .and_then(|t| t.parent_id.clone());
            match parent {
                Some(p) if !p.trim().is_empty() => {
                    cur = p;
                }
                _ => return Some(cur),
            }
            safety += 1;
            if safety > 2048 {
                return Some(cur);
            }
        }
    }

    pub fn frame_period_ms(&self) -> f64 {
        default_frame_period_ms()
    }

    pub fn project_duration_sec(&self) -> f64 {
        self.project_sec.max(0.0)
    }

    pub fn target_param_frames(&self, frame_period_ms: f64) -> usize {
        let fp = frame_period_ms.max(0.1);
        let sec = self.project_duration_sec();
        let frames = (sec * 1000.0 / fp).ceil();
        if !(frames.is_finite() && frames > 0.0) {
            return 1;
        }
        (frames as usize).max(1)
    }

    pub fn ensure_params_for_root(&mut self, root_track_id: &str) {
        let fp = self.frame_period_ms();
        let target = self.target_param_frames(fp);

        // Calculate expected cache key to detect when timeline changed
        let expected_key = crate::pitch_analysis::build_root_pitch_key(self, root_track_id);

        let entry = self
            .params_by_root_track
            .entry(root_track_id.to_string())
            .or_insert_with(|| TrackParamsState {
                frame_period_ms: fp,
                ..TrackParamsState::default()
            });

        // 旧工程使用算法专有的 `hifigan_volume`；统一迁移到共通 `volume`，
        // 保证切换算法时曲线仍然生效。
        Self::migrate_legacy_common_curves_in_entry(entry);

        entry.frame_period_ms = fp;

        // CRITICAL FIX: Detect stale pitch curves and clear them when clip/timeline changes.
        // This prevents old pitch data from being displayed after clip replacement or timeline edits.
        let key_changed = entry.pitch_orig_key.as_deref() != Some(&expected_key);

        if key_changed && entry.pitch_orig_key.is_some() {
            // Timeline/clip configuration changed - clear orig curves to force re-analysis
            entry.pitch_orig.clear();
            entry.pitch_orig_key = None;
            // 仅当用户未手动编辑时才清空 pitch_edit，保护用户的编辑成果
            if !entry.pitch_edit_user_modified {
                entry.pitch_edit.clear();
            }

            if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                log::warn!(
                    "state: [INVALIDATE] Cleared stale pitch curves for root_track={} (key changed, user_modified={})",
                    root_track_id, entry.pitch_edit_user_modified
                );
            }
        }

        #[allow(clippy::ptr_arg)]
        fn resize_curve(v: &mut Vec<f32>, target: usize, fill: f32) {
            if v.len() < target {
                v.extend(std::iter::repeat_n(fill, target - v.len()));
            } else if v.len() > target {
                v.truncate(target);
            }
        }

        resize_curve(&mut entry.pitch_orig, target, 0.0);
        resize_curve(&mut entry.pitch_edit, target, 0.0);
        resize_curve(&mut entry.tension_orig, target, 0.0);
        resize_curve(&mut entry.tension_edit, target, 0.0);

        // Backward compatibility: older projects didn't have `pitch_edit_user_modified`.
        // Infer it if we detect a meaningful difference between edit and orig.
        if !entry.pitch_edit_user_modified {
            let len = entry.pitch_orig.len().min(entry.pitch_edit.len());
            let mut i = 0usize;
            let stride = 1usize; // keep it simple; curves are not huge.
            while i < len {
                let o = entry.pitch_orig[i];
                let e = entry.pitch_edit[i];
                if e.is_finite() && e > 0.0 {
                    if !(o.is_finite() && o > 0.0) {
                        entry.pitch_edit_user_modified = true;
                        break;
                    }
                    if (e - o).abs() > 1e-3 {
                        entry.pitch_edit_user_modified = true;
                        break;
                    }
                }
                i += stride;
            }
        }
    }

    fn migrate_legacy_common_curves_in_entry(entry: &mut TrackParamsState) {
        // `hifigan_volume` → 共通 `volume`。
        if !entry.extra_curves.contains_key("volume") {
            if let Some(legacy) = entry.extra_curves.remove("hifigan_volume") {
                entry.extra_curves.insert("volume".to_string(), legacy);
            }
        } else {
            // 已存在共通曲线时，旧键不再参与渲染，直接移除避免缓存键重复计算。
            entry.extra_curves.remove("hifigan_volume");
        }
    }

    /// 工程加载/导入时执行参数迁移：
    /// 旧版 NSF-HiFiGAN 的 `hifigan_volume` 曲线统一迁移到共通 `volume` 曲线。
    pub fn migrate_legacy_common_param_curves(&mut self) {
        for entry in self.params_by_root_track.values_mut() {
            Self::migrate_legacy_common_curves_in_entry(entry);
        }
        for clip in &mut self.clips {
            if let Some(curves) = clip.extra_curves.as_mut() {
                if !curves.contains_key("volume") {
                    if let Some(legacy) = curves.remove("hifigan_volume") {
                        curves.insert("volume".to_string(), legacy);
                    }
                } else {
                    curves.remove("hifigan_volume");
                }
            }
        }
    }
}

/// Timeline snapshot for incremental pitch refresh
///
/// Stores a snapshot of the timeline state at the time of last pitch analysis
/// to enable detection of which clips have changed and need re-analysis.
#[derive(Debug, Clone)]
pub struct TimelineSnapshot {
    /// Mapping from clip ID to cache key
    pub clips: HashMap<String, String>,
    /// BPM at the time of analysis
    pub bpm: f64,
    /// Frame period used for analysis
    pub frame_period_ms: f64,
}

pub struct AppState {
    pub timeline: std::sync::Mutex<TimelineState>,
    pub timeline_version: std::sync::atomic::AtomicU64,
    pub timeline_history: std::sync::Mutex<TimelineHistory>,
    pub project: std::sync::Mutex<ProjectState>,
    pub runtime: std::sync::Mutex<RuntimeState>,

    /// Current UI locale reported by the frontend (e.g. "en-US", "zh-CN").
    /// Used to localize native dialogs implemented in Rust.
    pub ui_locale: RwLock<String>,

    /// When true, `checkpoint_timeline` calls are suppressed.
    /// Used by begin_undo_group / end_undo_group to group multiple
    /// backend operations into a single undo entry.
    pub suppress_checkpoints: std::sync::atomic::AtomicBool,

    pub waveform_cache_dir: std::sync::Mutex<PathBuf>,

    /// V2 多级 mipmap 波形缓存 (key = source_path)
    pub waveform_cache_v2:
        std::sync::Mutex<crate::hfspeaks_v2::WaveformPeakCache>,

    /// Inflight deduplication for waveform peak computation.
    /// When a file is being computed, its source_path is in this set.
    /// Other threads calling get_or_compute for the same path will wait
    /// on the Condvar until computation finishes, then read from cache.
    pub waveform_inflight: std::sync::Mutex<std::collections::HashSet<String>>,
    pub waveform_inflight_cv: std::sync::Condvar,

    /// In-memory cache of clipboard MIDI bytes, keyed by GUID (first 8 bytes of blake3 hash as hex).
    pub clipboard_midi_cache: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,

    // Set in Tauri setup. Used for async notifications.
    pub app_handle: OnceLock<tauri::AppHandle>,

    // De-dup background pitch analysis jobs (keyed by rootTrackId + analysis key).
    pub pitch_inflight: std::sync::Mutex<std::collections::HashSet<String>>,

    // Current pitch analysis progress (for polling from frontend)
    pub pitch_analysis_progress:
        std::sync::RwLock<Option<crate::pitch_analysis::PitchOrigAnalysisProgressEvent>>,

    // Clip-level pitch analysis cache for performance optimization
    pub clip_pitch_cache: Arc<Mutex<ClipPitchCache>>,

    // Timeline snapshot for incremental pitch refresh (keyed by root_track_id)
    pub pitch_timeline_snapshot: Mutex<HashMap<String, TimelineSnapshot>>,

    pub audio_engine: AudioEngine,

    /// 正在进行的录音会话（非录制时为 None）。
    pub recording: std::sync::Mutex<Option<crate::recording::ActiveRecording>>,

    /// App config directory for persisting recent projects etc.
    pub config_dir: OnceLock<std::path::PathBuf>,

    /// 启动参数传入的待打开工程路径（一次性消费）。
    pub pending_startup_project_path: Mutex<Option<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            timeline: std::sync::Mutex::new(TimelineState::default()),
            timeline_version: std::sync::atomic::AtomicU64::new(0),
            timeline_history: std::sync::Mutex::new(TimelineHistory::default()),
            project: std::sync::Mutex::new(ProjectState::default()),
            runtime: std::sync::Mutex::new(RuntimeState {
                device: "tauri".to_string(),
                synthesized_wav_path: None,
                ..RuntimeState::default()
            }),

            ui_locale: RwLock::new("en-US".to_string()),

            suppress_checkpoints: std::sync::atomic::AtomicBool::new(false),

            waveform_cache_dir: std::sync::Mutex::new(crate::hfspeaks_v2::default_cache_dir()),
            waveform_cache_v2: std::sync::Mutex::new(
                crate::hfspeaks_v2::WaveformPeakCache::default(),
            ),

            waveform_inflight: std::sync::Mutex::new(std::collections::HashSet::new()),
            waveform_inflight_cv: std::sync::Condvar::new(),
            clipboard_midi_cache: std::sync::Mutex::new(std::collections::HashMap::new()),

            app_handle: OnceLock::new(),
            pitch_inflight: std::sync::Mutex::new(std::collections::HashSet::new()),
            pitch_analysis_progress: std::sync::RwLock::new(None),
            clip_pitch_cache: Arc::new(Mutex::new(ClipPitchCache::new(100))),
            pitch_timeline_snapshot: Mutex::new(HashMap::new()),

            audio_engine: AudioEngine::new(),
            recording: std::sync::Mutex::new(None),
            config_dir: OnceLock::new(),
            pending_startup_project_path: Mutex::new(None),
        }
    }
}

impl AppState {
    pub fn bump_timeline_version(&self) -> u64 {
        self.timeline_version
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            .saturating_add(1)
    }

    pub fn set_pending_startup_project_path(&self, path: Option<String>) {
        let mut guard = self
            .pending_startup_project_path
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *guard = path;
    }

    pub fn take_pending_startup_project_path(&self) -> Option<String> {
        let mut guard = self
            .pending_startup_project_path
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.take()
    }

    /// 使指定源路径的波形峰值缓存失效（内存缓存 + inflight 标记）。
    ///
    /// 当源文件被替换（即使路径相同但内容变化）时调用，确保下次请求波形
    /// 数据时重新从磁盘/文件计算，而非返回旧文件的缓存峰值。
    pub fn invalidate_waveform_cache_for_path(&self, source_path: &str) {
        {
            let mut cache_v2 = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            cache_v2.remove(source_path);
        }
        // 同时移除 inflight 标记（如果正在计算中，也让它失效重新计算）
        self.remove_waveform_inflight(source_path);
    }

    pub fn clear_waveform_cache(&self) -> crate::hfspeaks_v2::ClearStats {
        // 清理 v2 内存缓存
        {
            let mut cache_v2 = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            cache_v2.clear();
        }

        let cache_dir = {
            self.waveform_cache_dir
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner())
                .clone()
        };
        crate::hfspeaks_v2::clear_cache_dir(&cache_dir)
    }

    /// 获取或计算 v2 多级 mipmap 峰值数据
    ///
    /// 优先从内存缓存读取，其次从磁盘缓存读取，最后计算。
    /// 使用 inflight 去重：如果另一线程正在计算同一文件，当前线程会等待
    /// 其完成后直接从缓存读取，避免重复计算和重复进度事件。
    /// 首次计算时会通过 Tauri 事件推送进度（waveform_analysis_progress）
    pub fn get_or_compute_waveform_peaks_v2(
        &self,
        source_path: &str,
    ) -> Result<std::sync::Arc<crate::hfspeaks_v2::HfsPeakFile>, String> {
        if source_path.trim().is_empty() {
            return Err("empty source_path".to_string());
        }

        // ── 1. 检查内存缓存 ──
        {
            let mut cache = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            if let Some(found) = cache.get(source_path) {
                // 缓存命中：发送 cached 状态事件
                if let Some(handle) = self.app_handle.get() {
                    use tauri::Emitter;
                    let _ = handle.emit(
                        "waveform_analysis_progress",
                        serde_json::json!({
                            "sourcePath": source_path,
                            "progress": 1.0,
                            "status": "cached",
                        }),
                    );
                }
                return Ok(found.clone() as std::sync::Arc<crate::hfspeaks_v2::HfsPeakFile>);
            }
        }

        // ── 2. Inflight 去重检查 ──
        // 如果另一线程已在计算同一文件，等待它完成后从缓存读取
        {
            let mut inflight = self
                .waveform_inflight
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            if inflight.contains(source_path) {
                // 另一线程正在计算此文件，等待 Condvar 通知
                let key = source_path.to_string();
                let _guard = self
                    .waveform_inflight_cv
                    .wait_while(inflight, |set| set.contains(&*key))
                    .unwrap_or_else(|e| e.into_inner());

                // 计算已完成，从缓存读取
                let mut cache = self
                    .waveform_cache_v2
                    .lock()
                    .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
                if let Some(found) = cache.get(source_path) {
                    if let Some(handle) = self.app_handle.get() {
                        use tauri::Emitter;
                        let _ = handle.emit(
                            "waveform_analysis_progress",
                            serde_json::json!({
                                "sourcePath": source_path,
                                "progress": 1.0,
                                "status": "cached",
                            }),
                        );
                    }
                    return Ok(found.clone());
                }
                // 极端情况：前一线程计算失败未放入缓存，继续往下重新计算
            } else {
                // 标记当前线程为此文件的计算者
                inflight.insert(source_path.to_string());
            }
        }

        // ── 3. 磁盘缓存 ──
        let cache_dir = {
            self.waveform_cache_dir
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        };

        let hfs_cache = crate::hfspeaks_v2::HfsPeaksCache::new(cache_dir);
        let path = std::path::Path::new(source_path);

        // 尝试从磁盘加载
        if let Some(cached) = hfs_cache.try_load(path) {
            let cached: std::sync::Arc<crate::hfspeaks_v2::HfsPeakFile> =
                std::sync::Arc::new(cached);
            {
                let mut cache = self
                    .waveform_cache_v2
                    .lock()
                    .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
                cache.insert(source_path, cached.clone());
            }
            // 磁盘缓存命中：发送 cached 状态事件
            if let Some(handle) = self.app_handle.get() {
                use tauri::Emitter;
                let _ = handle.emit(
                    "waveform_analysis_progress",
                    serde_json::json!({
                        "sourcePath": source_path,
                        "progress": 1.0,
                        "status": "cached",
                    }),
                );
            }
            // 移除 inflight 标记并通知等待线程
            self.remove_waveform_inflight(source_path);
            return Ok(cached);
        }

        // ── 4. 计算新的峰值数据 ──
        // 发送 computing 状态事件（进度 0）
        let source_path_owned = source_path.to_string();
        if let Some(handle) = self.app_handle.get() {
            use tauri::Emitter;
            let _ = handle.emit(
                "waveform_analysis_progress",
                serde_json::json!({
                    "sourcePath": &source_path_owned,
                    "progress": 0.0,
                    "status": "computing",
                }),
            );
        }

        // 构建进度回调：通过 app_handle emit 事件
        let app_handle_for_cb = self.app_handle.get().cloned();
        let source_path_for_cb = source_path_owned.clone();
        let progress_cb = move |progress: f32| {
            if let Some(ref handle) = app_handle_for_cb {
                use tauri::Emitter;
                let _ = handle.emit(
                    "waveform_analysis_progress",
                    serde_json::json!({
                        "sourcePath": &source_path_for_cb,
                        "progress": progress.clamp(0.0, 1.0),
                        "status": "computing",
                    }),
                );
            }
        };

        // 计算新的峰值数据（带进度回调）
        let result =
            crate::hfspeaks_v2::compute_mipmap_peaks_with_progress(path, Some(progress_cb));

        // 如果计算失败，移除 inflight 标记并返回错误
        let peaks = match result {
            Ok(p) => p,
            Err(e) => {
                // 必须发送终态事件：前端依赖 done/failed 隐藏“正在分析波形”状态，
                // 缺失/无法读取的文件如果没有终态事件会导致进度提示永久停留。
                if let Some(handle) = self.app_handle.get() {
                    use tauri::Emitter;
                    let _ = handle.emit(
                        "waveform_analysis_progress",
                        serde_json::json!({
                            "sourcePath": &source_path_owned,
                            "progress": 1.0,
                            "status": "failed",
                            "error": e,
                        }),
                    );
                }
                self.remove_waveform_inflight(source_path);
                return Err(e);
            }
        };

        // 保存到磁盘缓存
        if let Err(e) = hfs_cache.save(path, &peaks) {
            log::error!("Warning: failed to save v2 peaks cache: {}", e);
        }

        // 发送 done 状态事件
        if let Some(handle) = self.app_handle.get() {
            use tauri::Emitter;
            let _ = handle.emit(
                "waveform_analysis_progress",
                serde_json::json!({
                    "sourcePath": &source_path_owned,
                    "progress": 1.0,
                    "status": "done",
                }),
            );
        }

        let peaks = std::sync::Arc::new(peaks);
        {
            let mut cache = self
                .waveform_cache_v2
                .lock()
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
            cache.insert(source_path, peaks.clone());
        }
        // 移除 inflight 标记并通知等待线程
        self.remove_waveform_inflight(source_path);
        Ok(peaks)
    }

    /// 辅助方法：从 inflight 集合中移除 source_path 并通知所有等待线程
    fn remove_waveform_inflight(&self, source_path: &str) {
        let mut inflight = self
            .waveform_inflight
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        inflight.remove(source_path);
        self.waveform_inflight_cv.notify_all();
    }

    pub fn project_meta_payload(&self) -> ProjectMetaPayload {
        let p = self
            .project
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        ProjectMetaPayload {
            name: p.name,
            path: p.path,
            dirty: p.dirty,
            recent: p.recent,
            notes_markdown: p.notes_markdown,
            base_scale: p.base_scale,
            use_custom_scale: p.use_custom_scale,
            custom_scale: p.custom_scale,
            beats_per_bar: p.beats_per_bar,
            time_signature_denominator: p.time_signature_denominator,
            grid_size: p.grid_size,
            stretch_algorithm_override: p.stretch_algorithm_override,
            hifigan_mel_stretch_override: p.hifigan_mel_stretch_override,
        }
    }

    pub fn checkpoint_timeline(&self, snapshot: &TimelineState) {
        // When suppress_checkpoints is active (inside an undo group),
        // skip pushing to the undo stack so multiple operations become
        // a single undo entry.
        if self
            .suppress_checkpoints
            .load(std::sync::atomic::Ordering::Acquire)
        {
            // Still mark project dirty
            let (name, was_clean) = {
                let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
                let was_clean = !p.dirty;
                p.dirty = true;
                (p.name.clone(), was_clean)
            };
            if was_clean {
                if let Some(handle) = self.app_handle.get() {
                    use tauri::Manager;
                    if let Some(win) = handle.get_webview_window("main") {
                        let title = format!("HiFiShifter - {}*", name);
                        let _ = win.set_title(&title);
                    }
                }
            }
            return;
        }
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.undo.push_back(snapshot.clone());
        if h.undo.len() > MAX_UNDO_HISTORY {
            h.undo.pop_front();
        }
        h.redo.clear();
        drop(h);

        self.bump_timeline_version();

        let (name, was_clean) = {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            let was_clean = !p.dirty;
            p.dirty = true;
            (p.name.clone(), was_clean)
        };

        // 仅在首次变脏时更新窗口标题（添加 * 号）
        if was_clean {
            if let Some(handle) = self.app_handle.get() {
                use tauri::Manager;
                if let Some(win) = handle.get_webview_window("main") {
                    let title = format!("HiFiShifter - {}*", name);
                    let _ = win.set_title(&title);
                }
            }
        }
    }

    pub fn clear_history(&self) {
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.undo.clear();
        h.redo.clear();
    }

    /// Begin an undo group: push the current state once and suppress further checkpoints.
    pub fn begin_undo_group(&self) -> TimelineStatePayload {
        let tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        // Force a checkpoint even if suppress was already active (defensive)
        self.suppress_checkpoints
            .store(false, std::sync::atomic::Ordering::Release);
        self.checkpoint_timeline(&tl);
        self.suppress_checkpoints
            .store(true, std::sync::atomic::Ordering::Release);
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        payload
    }

    /// End the undo group: re-enable checkpoints.
    pub fn end_undo_group(&self) -> serde_json::Value {
        self.suppress_checkpoints
            .store(false, std::sync::atomic::Ordering::Release);
        serde_json::json!({ "ok": true })
    }

    pub fn undo_timeline(&self) -> TimelineStatePayload {
        let mut tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(prev) = h.undo.pop_back() else {
            let mut payload = tl.to_payload();
            payload.project = Some(self.project_meta_payload());
            return payload;
        };
        let scale_before = tl.render_scale_signature();
        let current = std::mem::replace(&mut *tl, prev);
        h.redo.push(current);
        drop(h);
        self.bump_timeline_version();
        // 恢复的快照可能改变 clip 的 active take：hnsep 分离缓存键只含
        // clip_id+采样率+样本数，等长 Take 会命中彼此的 harmonic/noise
        // stem（气声路径串音）。撤销/重做属低频操作，整体清空与
        // set_clip_active_take 命令路径的失效策略一致。
        crate::hnsep_onnx::clear_separation_cache();
        // 恢复的时间线快照可能带有 Tempo Map 初始点（工程基准记录）：
        // 同步工程 BPM/拍号/音阶，避免撤销后工程记录与 Tempo Map 分叉。
        {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            self.sync_project_record_from_tempo_map(&mut tl, &mut p);
        }
        self.audio_engine.update_timeline(tl.clone());
        self.invalidate_render_caches_if_scale_changed(&tl, &scale_before);
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        payload
    }

    pub fn redo_timeline(&self) -> TimelineStatePayload {
        let mut tl = self.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = self
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(next) = h.redo.pop() else {
            let mut payload = tl.to_payload();
            payload.project = Some(self.project_meta_payload());
            return payload;
        };
        let scale_before = tl.render_scale_signature();
        let current = std::mem::replace(&mut *tl, next);
        h.undo.push_back(current);
        drop(h);
        self.bump_timeline_version();
        // 与 undo_timeline 一致：active take 可能随快照回退，须清分离缓存。
        crate::hnsep_onnx::clear_separation_cache();
        // 与 undo_timeline 一致：重做后同步工程基准记录。
        {
            let mut p = self.project.lock().unwrap_or_else(|e| e.into_inner());
            self.sync_project_record_from_tempo_map(&mut tl, &mut p);
        }
        self.audio_engine.update_timeline(tl.clone());
        self.invalidate_render_caches_if_scale_changed(&tl, &scale_before);
        let mut payload = tl.to_payload();
        payload.project = Some(self.project_meta_payload());
        payload
    }

    /// 实际生效音阶发生变化时失效所有渲染缓存，并在「后台预渲染」启用时
    /// 触发后台渲染（与直接编辑 Tempo Map 的路径一致；撤销/重做恢复的快照
    /// 同样走这里 —— 引擎的 clip 差分检测不会覆盖 Tempo Map）。
    fn invalidate_render_caches_if_scale_changed(&self, tl: &TimelineState, scale_before: &str) {
        let scale_after = tl.render_scale_signature();
        if scale_before == scale_after {
            return;
        }
        for clip in &tl.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        if let Some(handle) = self.app_handle.get() {
            let _ = crate::commands::playback::request_background_render(handle);
        }
    }

    /// 从 Tempo Map 0 位置初始点同步“工程基准记录”（BPM / 拍号 / 音阶）。
    ///
    /// 初始点即工程基准记录（与 `set_timeline_tempo_map` 的双向同步约定一致），
    /// 撤销/重做恢复时间线快照后也必须重新同步，否则工程记录与 Tempo Map
    /// 会永久分叉（例如撤销音阶修改后工程仍显示旧音阶，保存/重开也无法自愈）。
    /// 仅在实际值变化时写回并标记工程 dirty。
    pub fn sync_project_record_from_tempo_map(&self, tl: &mut TimelineState, p: &mut ProjectState) {
        let Some(first) = tl
            .tempo_map
            .as_ref()
            .and_then(|points| points.first())
            .cloned()
        else {
            return;
        };
        let mut changed = false;

        let bpm = first.bpm.clamp(10.0, 960.0);
        if (tl.bpm - bpm).abs() > 1e-9 {
            tl.bpm = bpm;
            changed = true;
        }
        let beats = first.numerator.unwrap_or(4).clamp(1, 32);
        if p.beats_per_bar != beats {
            p.beats_per_bar = beats;
            changed = true;
        }
        let denominator = match first.denominator {
            Some(d) if matches!(d, 1 | 2 | 4 | 8 | 16 | 32) => d,
            _ => p.time_signature_denominator,
        };
        if p.time_signature_denominator != denominator {
            p.time_signature_denominator = denominator;
            changed = true;
        }

        if let Some(scale) = first.scale.as_ref() {
            if let Some(key) = scale.key.as_deref() {
                if p.base_scale != key || p.use_custom_scale {
                    p.base_scale = key.to_string();
                    p.use_custom_scale = false;
                    p.custom_scale = None;
                    changed = true;
                }
                if let Some(notes) = scale_notes_for_key(key) {
                    tl.project_scale_notes = notes;
                }
            } else if let Some(notes) = scale.notes.as_ref() {
                let mut normalized: Vec<u8> = notes.iter().map(|n| n % 12).collect();
                normalized.sort_unstable();
                normalized.dedup();
                if !normalized.is_empty() {
                    let name = scale
                        .name
                        .clone()
                        .filter(|n| !n.trim().is_empty())
                        .unwrap_or_else(|| "Custom Scale".to_string());
                    let same = p.use_custom_scale
                        && p.custom_scale.as_ref().map(|c| (&c.name, &c.notes))
                            == Some((&name, &normalized));
                    if !same {
                        p.custom_scale = Some(crate::project::CustomScale {
                            id: p
                                .custom_scale
                                .as_ref()
                                .map(|c| c.id.clone())
                                .unwrap_or_else(|| new_id("cs")),
                            name,
                            notes: normalized.clone(),
                        });
                        p.use_custom_scale = true;
                        changed = true;
                    }
                    tl.project_scale_notes = normalized;
                }
            }
        }
        if changed {
            p.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 串行化触及进程级“同步编辑所有 Take”开关的测试：该设置是全局原子，
    /// 无 per-test 作用域，并行测试会互相踩踏（一个测试把开关改回 true 时，
    /// 另一个正在断言“关闭同步”行为的测试就会读到错误的值）。
    static SYNC_EDITS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn find_clip_start(timeline: &TimelineState, clip_id: &str) -> f64 {
        timeline
            .clips
            .iter()
            .find(|clip| clip.id == clip_id)
            .map(|clip| clip.start_sec)
            .unwrap_or(f64::NAN)
    }

    #[test]
    fn track_color_rotation_starts_with_gray_and_cycles() {
        // 调色板以灰色开头（与前端取色器顺序一致：灰→蓝→紫→绿→橙→
        // 粉→洋红→黄→红）：新建工程的初始 Main 轨道 = palette[0]（灰）。
        let mut tl = TimelineState::default();
        assert_eq!(tl.tracks[0].color, track_palette_color(0));
        assert_eq!(tl.tracks[0].color, "#74787e");

        // 「添加轨道」按当前轨道总数轮转：蓝 → 紫 → … → 红 → 灰 → 蓝…
        // 验证前 12 条新轨道（覆盖完整一轮 + 回到灰色后再出发）。
        let mut ids = Vec::new();
        for i in 1..=12 {
            let id = tl.add_track(Some(format!("T{i}")), None, None);
            let color = tl.tracks.iter().find(|t| t.id == id).unwrap().color.clone();
            let expected = track_palette_color(i); // Main 已占 index 0
            assert_eq!(color, expected, "add_track #{i} 的颜色");
            ids.push(id);
        }

        // 完整一轮：第 9 条新轨道回到灰色（palette 长度 = 9）。
        assert_eq!(
            tl.tracks.iter().find(|t| t.id == ids[8]).unwrap().color,
            "#74787e"
        );
        // 第 10 条新轨道重新从蓝色开始。
        assert_eq!(
            tl.tracks.iter().find(|t| t.id == ids[9]).unwrap().color,
            "#4a8fd1"
        );
    }

    #[test]
    fn clip_take_switch_cycle_and_remove_keep_projection_in_sync() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("TakeTest".into()),
                Some(0.0),
                Some(4.0),
                Some("C:/audio/a.wav".into()),
            )
            .clone();

        let second = {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            let mut take = clip.active_take().clone();
            take.id = new_id("take");
            take.name = "Take B".to_string();
            take.source_start_sec = 3.0;
            take.source_end_sec = 7.0;
            take
        };
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.add_take(second.clone());
        }

        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            let original_active = clip.active_take_id.clone().unwrap();
            clip.switch_active_take(&second.id).unwrap();
            assert_eq!(clip.source_start_sec, 3.0);
            assert_eq!(clip.source_end_sec, 7.0);
            clip.cycle_active_take(1);
            assert_eq!(
                clip.active_take_id.as_deref(),
                Some(original_active.as_str())
            );
            assert_eq!(clip.source_start_sec, 0.0);
            assert!(clip.remove_take(&second.id).is_ok());
            assert_eq!(clip.takes.len(), 1);
            let last_id = clip.takes[0].id.clone();
            assert!(
                clip.remove_take(&last_id).is_err(),
                "last take cannot be removed"
            );
        }
    }

    #[test]
    fn split_clip_splits_every_take_independent_of_sync_setting() {
        let _sync_guard = SYNC_EDITS_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("Multi".into()),
                Some(0.0),
                Some(4.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        let take_ids = {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            let first_id = clip.takes[0].id.clone();
            let mut second = clip.active_take().clone();
            second.id = new_id("take");
            second.name = "Take B".to_string();
            second.source_path = Some("C:/b.wav".to_string());
            second.duration_sec = Some(30.0);
            second.duration_frames = None;
            second.source_sample_rate = None;
            second.source_start_sec = 20.0;
            second.source_end_sec = 30.0;
            // 本测试考察非 Loop 分割：克隆自默认 Loop 的 active take 会带上
            // loop=true，必须显式关闭。
            second.loop_enabled = false;
            clip.add_take(second);
            // 本测试考察非 Loop 分割的窗口数学：add_clip 按进程级默认开启
            // Loop，须显式关闭并 sync 回全部 Take 后再分割。
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
            vec![first_id, clip.takes[1].id.clone()]
        };

        // 显式关闭进程级同步设置：分割仍然必须作用于全部 Take。
        crate::config::set_sync_edits_across_takes(false);
        let right_id = tl.split_clip(&clip_id, 2.0).unwrap();

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert_eq!(left.takes.len(), 2);
        assert_eq!(right.takes.len(), 2);

        assert!((left.takes[0].source_start_sec - 0.0).abs() < 1e-9);
        assert!((left.takes[0].source_end_sec - 2.0).abs() < 1e-9);
        assert!((right.takes[0].source_start_sec - 2.0).abs() < 1e-9);
        assert!((right.takes[0].source_end_sec - 4.0).abs() < 1e-9);

        assert!(
            (left.takes[1].source_start_sec - 20.0).abs() < 1e-9,
            "left.take1 ss = {}",
            left.takes[1].source_start_sec
        );
        assert!(
            (left.takes[1].source_end_sec - 22.0).abs() < 1e-9,
            "left.take1 se = {} (loop={})",
            left.takes[1].source_end_sec,
            left.takes[1].loop_enabled
        );
        assert!(
            (right.takes[1].source_start_sec - 22.0).abs() < 1e-9,
            "right.take1 ss = {}",
            right.takes[1].source_start_sec
        );
        assert!(
            (right.takes[1].source_end_sec - 24.0).abs() < 1e-9,
            "right.take1 se = {}",
            right.takes[1].source_end_sec
        );

        assert_eq!(
            take_ids.len(),
            2,
            "original ids captured for mapping sanity"
        );
        crate::config::set_sync_edits_across_takes(true);
    }

    #[test]
    fn pack_and_explode_clips_into_takes_roundtrip_geometry() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let a = tl
            .add_clip(
                Some(track_id.clone()),
                Some("A".into()),
                Some(1.0),
                Some(2.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        let b = tl
            .add_clip(
                Some(track_id),
                Some("B".into()),
                Some(0.0),
                Some(2.5),
                Some("C:/b.wav".into()),
            )
            .clone();
        // Clip 级拉伸速率应并入聚合后的 Take 自身速率，新 Clip 级速率为 1。
        // 平铺 playback_rate 是组合有效速率（权威）：倍率与有效速率须一致地
        // 设为 2，sync 后 Take 自身速率保持 1，pack 才能把组合速率 2 并入。
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == a).unwrap();
            clip.clip_playback_rate = 2.0;
            clip.playback_rate = 2.0;
            clip.sync_take_from_flat();
        }

        let packed = tl.pack_clips_into_takes(&[a.clone(), b.clone()]).unwrap();
        let clip = tl.clips.iter().find(|c| c.id == packed).unwrap();
        assert_eq!(clip.takes.len(), 2);
        assert!((clip.start_sec - 0.0).abs() < 1e-9);
        // 打包容器覆盖源 Clip 的并集区间：A [1,3] ∪ B [0,2.5] → [0,3]。
        assert!((clip.length_sec - 3.0).abs() < 1e-9);
        assert!((clip.clip_playback_rate - 1.0).abs() < f32::EPSILON);
        assert!(
            (clip.takes[0].playback_rate - 2.0).abs() < f32::EPSILON,
            "A 的 Clip×Take 速率应并入第一个 Take，实际 = {}",
            clip.takes[0].playback_rate
        );
        assert_eq!(
            tl.clips.iter().filter(|c| c.id == a || c.id == b).count(),
            0
        );

        let exploded = tl.explode_clip_takes(&packed);
        assert_eq!(exploded.len(), 2);
        assert_eq!(tl.clips.len(), 2);
    }

    /// "将 Take 展开为独立音频块"向下展开回归：第 idx 个 Take 放到源轨道
    /// 可视顺序下方第 idx 行；下方没有现成轨道时克隆源轨道设置新建。
    #[test]
    fn explode_clip_takes_places_takes_downward_across_tracks() {
        let mut tl = TimelineState::default();
        let track_a = tl.tracks[0].id.clone();
        let track_b = tl.add_track(Some("B".into()), None, None);
        let track_c = tl.add_track(Some("C".into()), None, None);

        let clip_id = tl.add_clip(
            Some(track_a.clone()),
            Some("Multi".into()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            for n in 1..4 {
                let mut take = clip.active_take().clone();
                take.id = new_id("take");
                take.name = format!("Take {n}");
                clip.add_take(take);
            }
        }
        assert_eq!(tl.tracks.len(), 3);

        let exploded = tl.explode_clip_takes(&clip_id);
        assert_eq!(exploded.len(), 4);

        let track_of = |cid: &String| {
            tl.clips
                .iter()
                .find(|c| c.id == *cid)
                .map(|c| c.track_id.clone())
                .unwrap()
        };
        assert_eq!(track_of(&exploded[0]), track_a, "take 1 stays on track A");
        assert_eq!(track_of(&exploded[1]), track_b, "take 2 lands on track B");
        assert_eq!(track_of(&exploded[2]), track_c, "take 3 lands on track C");

        // 第 4 个 Take：下方没有现成轨道 → 新建轨道 D（克隆 A 的设置）。
        let track_d = track_of(&exploded[3]);
        assert_ne!(track_d, track_a, "take 4 must get a new track");
        assert_eq!(tl.tracks.len(), 4, "one track auto-created for take 4");
        let track_a_meta = tl.tracks.iter().find(|t| t.id == track_a).unwrap();
        let track_d_meta = tl.tracks.iter().find(|t| t.id == track_d).unwrap();
        assert_eq!(track_d_meta.parent_id, track_a_meta.parent_id);
        assert_eq!(
            track_d_meta.pitch_analysis_algo,
            track_a_meta.pitch_analysis_algo
        );
        assert_eq!(track_d_meta.name, track_a_meta.name);
        assert_eq!(track_d_meta.volume, track_a_meta.volume);

        // 可视顺序自上而下为 A、B、C、D。
        assert_eq!(
            tl.visual_track_ids(),
            vec![track_a, track_b, track_c, track_d]
        );
    }

    #[test]
    fn clip_take_remap_preserves_active_pointer() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(Some(track_id), Some("R".into()), Some(0.0), Some(2.0), None)
            .clone();
        let old_ids = {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            let mut take = clip.active_take().clone();
            take.id = new_id("take");
            let second_id = take.id.clone();
            clip.add_take(take);
            clip.switch_active_take(&second_id).unwrap();
            let ids: Vec<String> = clip.takes.iter().map(|t| t.id.clone()).collect();
            ids
        };
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.remap_take_ids();
            let new_ids: Vec<String> = clip.takes.iter().map(|t| t.id.clone()).collect();
            assert_ne!(old_ids, new_ids);
            assert_eq!(
                clip.active_take_id.as_deref(),
                Some(new_ids[1].as_str()),
                "active 指向重映射后的第二个 take"
            );
        }
    }

    // ── 方向翻转（倒放）的源窗口/锚点换算 ──────────────────────────────

    /// 核心回归：非 Loop 正放 Clip 的存储 se 可能是陈旧值（派生窗口模型下
    /// 不参与消费数学）。右键“倒放”翻转方向时必须以原消费窗口 [ss, ss+len·r)
    /// 推导新锚点 se —— 否则裁剪过的 Clip 倒放后会播到陈旧 se 所指的文件末段。
    #[test]
    fn reverse_flip_preserves_consumed_window_with_stale_source_end() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("R".into()),
                Some(0.0),
                Some(8.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
            clip.source_start_sec = 10.0;
            // 模拟陈旧存储：长度 8、组合速率 1 的有效消费终点是 18，
            // 存储 se 停留在导入期的文件末端值。
            clip.source_end_sec = 30.0;
            clip.sync_take_from_flat();
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(clip.reversed);
        assert!(
            (clip.source_end_sec - 18.0).abs() < 1e-9,
            "se 应换算为原消费窗口终点 18，实际 {}",
            clip.source_end_sec
        );
        assert!((clip.source_start_sec - 10.0).abs() < 1e-9);
        let (win_start, win_end) = clip_playback_window_sec(clip);
        assert!((win_start - 10.0).abs() < 1e-9);
        assert!((win_end - 18.0).abs() < 1e-9);
        let take = clip.active_take();
        assert!(take.reversed);
        assert!((take.source_end_sec - 18.0).abs() < 1e-9);
    }

    /// 倒放 → 正放的对称方向：以倒放消费窗口 [se−len·r, se) 的起点换算
    /// 新锚点 ss，存储中的陈旧 ss（不参与倒放消费数学）必须被覆盖。
    #[test]
    fn unreverse_flip_preserves_consumed_window_with_stale_source_start() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("R".into()),
                Some(0.0),
                Some(8.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip.reversed = true;
            clip.source_end_sec = 18.0;
            // 陈旧 ss：倒放消费窗口 [10, 18) 不读该字段。
            clip.source_start_sec = 999.0;
            clip.sync_take_from_flat();
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(false),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(!clip.reversed);
        assert!(
            (clip.source_start_sec - 10.0).abs() < 1e-9,
            "ss 应换算为原倒放窗口起点 10，实际 {}",
            clip.source_start_sec
        );
        let (win_start, win_end) = clip_playback_window_sec(clip);
        assert!((win_start - 10.0).abs() < 1e-9);
        assert!((win_end - 18.0).abs() < 1e-9);
    }

    /// “同步编辑所有 Take”下的方向翻转：每个 Take 按自身组合消费速率
    /// （clip 倍率 × take 速率）换算窗口，不能共享 flat 的同一窗口。
    #[test]
    fn reverse_flip_sync_converts_every_take_with_own_rate() {
        let _sync_guard = SYNC_EDITS_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::config::set_sync_edits_across_takes(true);
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("Multi".into()),
                Some(0.0),
                Some(4.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
            clip.source_start_sec = 10.0;
            clip.source_end_sec = 14.0;
            let mut second = clip.active_take().clone();
            second.id = new_id("take");
            second.playback_rate = 2.0;
            second.source_start_sec = 5.0;
            second.source_end_sec = 20.0;
            clip.add_take(second);
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(clip.reversed);
        let take1 = &clip.takes[0];
        assert!(take1.reversed);
        assert!(
            (take1.source_end_sec - 14.0).abs() < 1e-9,
            "take1 se = {}",
            take1.source_end_sec
        );
        let take2 = &clip.takes[1];
        assert!(take2.reversed);
        assert!(
            (take2.source_end_sec - 13.0).abs() < 1e-9,
            "take2（rate 2）se 应为 5 + 4×2 = 13，实际 {}",
            take2.source_end_sec
        );
        crate::config::set_sync_edits_across_takes(true);
    }

    /// 关闭同步设置时，Clip 级倒放只翻转 active take；inactive take 的
    /// 方向与窗口保持不变。
    #[test]
    fn reverse_flip_without_sync_touches_active_take_only() {
        let _sync_guard = SYNC_EDITS_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::config::set_sync_edits_across_takes(false);
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("Multi".into()),
                Some(0.0),
                Some(4.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
            clip.source_start_sec = 10.0;
            clip.source_end_sec = 14.0;
            let mut second = clip.active_take().clone();
            second.id = new_id("take");
            second.source_start_sec = 20.0;
            second.source_end_sec = 24.0;
            clip.add_take(second);
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(clip.reversed);
        assert!(clip.takes[0].reversed);
        assert!((clip.takes[0].source_end_sec - 14.0).abs() < 1e-9);
        assert!(!clip.takes[1].reversed, "inactive take 不被翻转");
        assert!((clip.takes[1].source_start_sec - 20.0).abs() < 1e-9);
        assert!((clip.takes[1].source_end_sec - 24.0).abs() < 1e-9);
        crate::config::set_sync_edits_across_takes(true);
    }

    /// 单 Take 倒放命令：翻转目标 Take 的窗口按其自身速率换算；inactive
    /// take 不动 flat 投影，active take 物化到 flat。
    #[test]
    fn set_clip_take_reversed_flips_only_target_take() {
        let _sync_guard = SYNC_EDITS_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::config::set_sync_edits_across_takes(false);
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("Multi".into()),
                Some(0.0),
                Some(4.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        let take2_id = {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
            clip.source_start_sec = 10.0;
            clip.source_end_sec = 14.0;
            clip.sync_take_from_flat();
            let mut second = clip.active_take().clone();
            second.id = new_id("take");
            second.playback_rate = 2.0;
            second.source_start_sec = 5.0;
            second.source_end_sec = 20.0;
            let id = second.id.clone();
            clip.add_take(second);
            id
        };
        // 翻转 inactive take：返回 false，flat 与其它 take 不变。
        let flipped_active = tl
            .set_clip_take_reversed(&clip_id, &take2_id, true)
            .unwrap();
        assert!(!flipped_active);
        {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            assert!(!clip.reversed, "inactive take 翻转不改 flat 投影");
            assert!(!clip.takes[0].reversed);
            let take2 = &clip.takes[1];
            assert!(take2.reversed);
            assert!(
                (take2.source_end_sec - 13.0).abs() < 1e-9,
                "take2 se 应为 5 + 4×1×2 = 13，实际 {}",
                take2.source_end_sec
            );
        }
        // 翻转 active take：返回 true，flat 投影随 take 物化。
        let take1_id = {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            clip.takes[0].id.clone()
        };
        let flipped_active = tl
            .set_clip_take_reversed(&clip_id, &take1_id, true)
            .unwrap();
        assert!(flipped_active);
        {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            assert!(clip.reversed);
            assert!((clip.source_end_sec - 14.0).abs() < 1e-9);
            assert!((clip.source_start_sec - 10.0).abs() < 1e-9);
            assert!(clip.active_take().reversed);
        }
        // 幂等：目标方向与当前一致时不改动窗口。
        let flipped_active = tl
            .set_clip_take_reversed(&clip_id, &take1_id, true)
            .unwrap();
        assert!(flipped_active);
        {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            assert!((clip.source_end_sec - 14.0).abs() < 1e-9);
        }
        crate::config::set_sync_edits_across_takes(true);
    }

    /// Loop take 的方向翻转以**原方向消费区间**为准换算锚点：正放锚 12、
    /// D=30、span=4 → 翻为倒放后锚定消费终点 mod(12+4, 30) = 16（自 16 降奏
    /// 覆盖 [12,16) 的镜像），而非正放起点 12；翻回正放还原 mod(16−4, 30)=12。
    #[test]
    fn reverse_flip_loop_anchors_at_consumption_end() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("L".into()),
                Some(0.0),
                Some(4.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = true;
            clip.duration_sec = Some(30.0);
            clip.sync_take_from_flat();
            clip.source_start_sec = 12.0;
            clip.sync_take_from_flat();
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                ..Default::default()
            },
        );
        {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            assert!(clip.reversed);
            assert!(
                (clip.source_end_sec - 16.0).abs() < 1e-9,
                "Loop 倒放锚应为消费终点 mod(12+4, 30) = 16，实际 {}",
                clip.source_end_sec
            );
            assert!((clip.source_start_sec - 12.0).abs() < 1e-9);
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(false),
                ..Default::default()
            },
        );
        {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            assert!(!clip.reversed);
            assert!(
                (clip.source_start_sec - 12.0).abs() < 1e-9,
                "Loop 正放锚应还原为消费起点 mod(16−4, 30) = 12，实际 {}",
                clip.source_start_sec
            );
        }
    }

    /// 用户场景回归：10s 媒体、Loop 开启（应用默认）、Clip 修剪为源 [2,4)
    /// （存储 se 停留在导入期的 10），倒放必须锚定消费终点 4 —— 自 4 降奏
    /// 覆盖 4~2s，而不是从正放锚 2 向下降奏到 [0,2)。
    #[test]
    fn reverse_flip_loop_trimmed_region_plays_backwards_in_place() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("L".into()),
                Some(0.0),
                Some(2.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = true;
            clip.duration_sec = Some(10.0);
            clip.sync_take_from_flat();
            clip.source_start_sec = 2.0;
            // 陈旧存储：导入期的文件末端值。
            clip.source_end_sec = 10.0;
            clip.sync_take_from_flat();
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(clip.reversed);
        assert!(
            (clip.source_end_sec - 4.0).abs() < 1e-9,
            "Loop 倒放锚应为 mod(2+2, 10) = 4，实际 {}",
            clip.source_end_sec
        );
        assert!((clip.source_start_sec - 2.0).abs() < 1e-9);
    }

    /// 变速下的 Loop 翻转：span 按组合速率换算 —— 源窗口锚 2、组合速率 2
    /// （clip 长 2s 消费 4 源秒）→ 倒放锚 mod(2 + 2×2, 10) = 6。
    #[test]
    fn reverse_flip_loop_with_rate_uses_combined_span() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("L".into()),
                Some(0.0),
                Some(2.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = true;
            clip.duration_sec = Some(10.0);
            clip.sync_take_from_flat();
            clip.source_start_sec = 2.0;
            clip.playback_rate = 2.0;
            clip.sync_take_from_flat();
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(clip.reversed);
        assert!(
            (clip.source_end_sec - 6.0).abs() < 1e-9,
            "Loop 倒放锚应为 mod(2+2×2, 10) = 6，实际 {}",
            clip.source_end_sec
        );
    }

    /// 显式携带源窗口的 reversed 请求（粘贴/导入模板路径）以调用方窗口为准，
    /// 不做方向换算。
    #[test]
    fn reverse_patch_with_explicit_window_skips_conversion() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("R".into()),
                Some(0.0),
                Some(8.0),
                Some("C:/a.wav".into()),
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip.sync_take_from_flat();
            clip.source_start_sec = 10.0;
            clip.source_end_sec = 30.0;
            clip.sync_take_from_flat();
        }
        tl.patch_clip_state(
            &clip_id,
            ClipStatePatch {
                reversed: Some(true),
                source_start_sec: Some(2.0),
                source_end_sec: Some(6.0),
                ..Default::default()
            },
        );
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        assert!(clip.reversed);
        assert!((clip.source_start_sec - 2.0).abs() < 1e-9);
        assert!((clip.source_end_sec - 6.0).abs() < 1e-9);
    }

    /// Loop 音符放置：D == 窗口时与"窗口内相对偏移 + 镜像 + 周期平铺"的
    /// 既有约定等价（fp=10ms、rate=1）。
    #[test]
    fn loop_note_placement_matches_window_semantics_when_cycle_equals_window() {
        // 窗口 [2, 6)，D = 窗口跨度 4s（纯 MIDI clip 场景）。
        let cycle = 4.0f64;
        let fwd_anchor = 2.0f64;
        // 倒放锚点传**原始** source_end（函数内部用 rem_euclid 归一），
        // 与音频路径约定一致 —— 不得在调用方预 clamp。
        let rev_anchor_end = 6.0f64;
        let fp = 10.0f64;

        // 正放：音符源位置 3.0~4.5 → 首现于消费 1.0s（= 帧 100）。
        let p = place_note_occurrence_frames(
            false,
            1.0,
            fp,
            fwd_anchor,
            rev_anchor_end,
            cycle,
            3.0,
            4.5,
        )
        .expect("valid placement");
        assert_eq!(p.first_start_frame, 100);
        assert_eq!(p.len_frames, 150);
        assert_eq!(p.cycle_frames, 400);

        // 倒放：(6−4.5) mod 4 = 1.5s（= 帧 150）—— 与窗口镜像语义等价。
        let p_rev = place_note_occurrence_frames(
            true,
            1.0,
            fp,
            fwd_anchor,
            rev_anchor_end,
            cycle,
            3.0,
            4.5,
        )
        .expect("valid placement");
        assert_eq!(p_rev.first_start_frame, 150);

        // 负锚点环绕：anchor=-1 对 D=4 → 首现于消费 1.0s（floor_mod(-1+u)）。
        let p_neg =
            place_note_occurrence_frames(false, 1.0, fp, -1.0, rev_anchor_end, cycle, -0.5, 0.5)
                .expect("valid placement");
        assert_eq!(p_neg.first_start_frame, 50);
    }

    /// 纯 MIDI（无媒体元数据）slip 窗口的倒放锚点不得被跨度 clamp 改写：
    /// 窗口 [2,7] 的跨度是 5，若把 rev 锚点 clamp 成 min(7,5)=5，
    /// 结束于 7s 的音符首现于 (5−7) mod 5 = 3s；正确行为是保持锚点 7 →
    /// 首现于 0s（窗口 exclusive 末端即入口）。
    #[test]
    fn loop_placement_reversed_window_span_keeps_raw_source_end_anchor() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("M".into()),
                Some(0.0),
                Some(10.0),
                None,
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = true;
            clip.reversed = true;
            clip.source_start_sec = 2.0;
            clip.source_end_sec = 7.0;
            // 无 duration_sec / duration_frames 元数据 → 周期退化为窗口跨度。
        }
        let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let p = place_note_occurrence_in_loop(clip, 6.0, 7.0, 10.0).expect("valid placement");
        assert_eq!(
            p.first_start_frame, 0,
            "note ending at window end is heard first"
        );
        assert_eq!(p.cycle_frames, 500);
    }

    /// Loop 音符放置：裁剪窗口（窗口 ≠ 媒体时长）时按媒体 D 的锚点回绕 ——
    /// 与音频渲染 floor_mod(anchor ± u, D) 同相位。
    #[test]
    fn loop_note_placement_uses_media_cycle_for_trimmed_windows() {
        // 媒体 10s，窗口 [1, 3]；split 右段 start=3.5 > end=3.0（环绕窗口）。
        // 音频在消费 u 秒处播放 floor_mod(3.5 + u, 10)：
        //   u ∈ [0, 6.5) → 源 3.5~10；u ∈ [6.5, 10) → 源 0~3.5。
        let cycle = 10.0f64;
        let fwd_anchor = 3.5f64;
        let fp = 10.0f64;

        // 音符源 8.0~9.5：出现在 u = 4.5s（帧 450），每 10s 重复一次。
        let p = place_note_occurrence_frames(false, 1.0, fp, fwd_anchor, 9.9, cycle, 8.0, 9.5)
            .expect("valid placement");
        assert_eq!(p.first_start_frame, 450);
        assert_eq!(p.cycle_frames, 1000);

        // 音符源 1.0~2.0：环绕后首现于 u = floor_mod(1−3.5, 10) = 7.5s（帧 750）。
        let p2 = place_note_occurrence_frames(false, 1.0, fp, fwd_anchor, 9.9, cycle, 1.0, 2.0)
            .expect("valid placement");
        assert_eq!(p2.first_start_frame, 750);

        // 倒放锚点末端 min(source_end=3.0, D)：音符源 1.0~2.0 首现于
        // w = 3.0 − 2.0 = 1.0s（帧 100）。
        let pr = place_note_occurrence_frames(true, 1.0, fp, fwd_anchor, 3.0, cycle, 1.0, 2.0)
            .expect("valid placement");
        assert_eq!(pr.first_start_frame, 100);
    }

    #[test]
    fn clip_loop_cycle_span_prefers_media_duration_and_falls_back_to_window() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(
                Some(track_id),
                Some("L".into()),
                Some(0.0),
                Some(12.0),
                None,
            )
            .clone();
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = true;
            // 无媒体元数据：退化为窗口跨度。
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 5.0;
        }
        let span_window = {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            clip_loop_cycle_span_sec(clip)
        };
        assert!((span_window.unwrap() - 4.0).abs() < 1e-9);

        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 有媒体元数据：取媒体总时长（与音频整文件回绕一致）。
            clip.duration_sec = Some(10.0);
        }
        let span_media = {
            let clip = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
            clip_loop_cycle_span_sec(clip)
        };
        assert!((span_media.unwrap() - 10.0).abs() < 1e-9);

        // 未启用 Loop：None。
        let span_off = {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.loop_enabled = false;
            clip_loop_cycle_span_sec(clip)
        };
        assert!(span_off.is_none());
    }

    /// 消费窗口模型：正放锚定 ss、倒放锚定 se —— 倒放的 ss 不参与消费数学，
    /// 域外窗口（se>D / 窗口下探 <0）由方向性前导静音表达。
    #[test]
    fn playback_window_and_leading_silence_directional() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl
            .add_clip(Some(track_id), Some("W".into()), Some(0.0), Some(4.0), None)
            .clone();
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.source_start_sec = 2.0;
            c.source_end_sec = 6.0;
            c.duration_sec = Some(10.0);
            c.playback_rate = 1.0;
            // add_clip 按进程级默认（loop_new_clips_default）开启 Loop；
            // 本测试考察的是**非 Loop** 消费窗口/前导静音模型，须显式关闭
            // （末尾 Loop 分支会再显式置 true）。否则前导静音恒为 0、
            // 倒放窗口断言走锚点分支 —— 与被测语义无关地失败。
            c.loop_enabled = false;
        }
        fn read<'a>(tl: &'a TimelineState, cid: &str) -> &'a super::Clip {
            tl.clips.iter().find(|c| c.id == cid).unwrap()
        }

        // 正放：win=[ss, ss+len·r)。
        let (ws, we) = clip_playback_window_sec(read(&tl, &clip_id));
        assert!((ws - 2.0).abs() < 1e-9 && (we - 6.0).abs() < 1e-9);
        // 正放 ss<0 → 前导静音。
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.source_start_sec = -1.0;
        }
        assert!((clip_leading_silence_sec(read(&tl, &clip_id), Some(10.0)) - 1.0).abs() < 1e-9);

        // 倒放：win=[se−len·r, se)，ss 完全不参与（即使为负/陈旧）。
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.reversed = true;
            c.source_start_sec = -99.0; // 历史字段：不得影响窗口
            c.source_end_sec = 6.0;
        }
        let (ws, we) = clip_playback_window_sec(read(&tl, &clip_id));
        assert!((ws - 2.0).abs() < 1e-9 && (we - 6.0).abs() < 1e-9);
        // 倒放窗口在媒体内 → 无前导静音；ss<0 是尾部静音，不产生前导静音。
        assert!(clip_leading_silence_sec(read(&tl, &clip_id), Some(10.0)).abs() < 1e-12);

        // 倒放 se 越过媒体末端（trim_left 延伸/split 过渡可达）→ 前导静音。
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.source_end_sec = 12.0;
        }
        let (ws, we) = clip_playback_window_sec(read(&tl, &clip_id));
        assert!((ws - 8.0).abs() < 1e-9 && (we - 12.0).abs() < 1e-9);
        assert!((clip_leading_silence_sec(read(&tl, &clip_id), Some(10.0)) - 2.0).abs() < 1e-9);

        // Loop：恒无前导静音；窗口原样返回（调用方走锚点回绕分支）。
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.loop_enabled = true;
            c.source_start_sec = -5.0;
        }
        assert!(clip_leading_silence_sec(read(&tl, &clip_id), Some(10.0)).abs() < 1e-12);

        // trim 窗口重定向：非 Loop 倒放传 [se−len·r, se]，其余透传。
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.loop_enabled = false;
        }
        let (ts, te) = clip_pitch_trim_window_sec(read(&tl, &clip_id));
        assert!((ts - 8.0).abs() < 1e-9 && (te - 12.0).abs() < 1e-9);
        {
            let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            c.reversed = false;
            c.source_start_sec = 3.0;
            c.source_end_sec = 7.0;
        }
        let (ts, te) = clip_pitch_trim_window_sec(read(&tl, &clip_id));
        assert!((ts - 3.0).abs() < 1e-9 && (te - 7.0).abs() < 1e-9);
    }

    /// 加载期规范化：存储字段 == 消费窗口（正放派生终点、倒放派生起点、
    /// Loop 不触碰）。
    #[test]
    fn normalize_nonloop_source_window_matches_consumed_window() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        fn mk(tl: &mut TimelineState, track_id: &str, rev: bool) -> String {
            let id = tl.add_clip(
                Some(track_id.to_string()),
                Some("N".into()),
                Some(0.0),
                Some(3.0),
                None,
            );
            {
                let c = tl.clips.iter_mut().find(|c| c.id == id).unwrap();
                c.reversed = rev;
                c.loop_enabled = false;
                c.playback_rate = 1.0;
            }
            id
        }
        let fwd_id = mk(&mut tl, &track_id, false);
        let rev_id = mk(&mut tl, &track_id, true);
        {
            let c = tl.clips.iter_mut().find(|c| c.id == fwd_id).unwrap();
            c.source_start_sec = 2.0;
            c.source_end_sec = 99.0; // 陈旧值
        }
        {
            let c = tl.clips.iter_mut().find(|c| c.id == rev_id).unwrap();
            c.source_start_sec = 99.0; // 陈旧值
            c.source_end_sec = -1.5; // 媒体下方静音段（合法）
        }
        let mut fwd = tl.clips.iter().find(|c| c.id == fwd_id).unwrap().clone();
        normalize_nonloop_source_window(&mut fwd);
        assert!(
            (fwd.source_end_sec - 5.0).abs() < 1e-9,
            "forward end derives from start+len"
        );
        assert!((fwd.source_start_sec - 2.0).abs() < 1e-9);

        let mut rev = tl.clips.iter().find(|c| c.id == rev_id).unwrap().clone();
        normalize_nonloop_source_window(&mut rev);
        assert!(
            (rev.source_start_sec - (-4.5)).abs() < 1e-9,
            "reversed start derives from end−len"
        );
        assert!(
            (rev.source_end_sec - (-1.5)).abs() < 1e-9,
            "negative anchor preserved"
        );

        // Loop：字段承载锚点相位，规范化不得触碰。
        let mut lp = rev.clone();
        lp.loop_enabled = true;
        lp.source_start_sec = -77.0;
        normalize_nonloop_source_window(&mut lp);
        assert!((lp.source_start_sec - (-77.0)).abs() < 1e-12);
    }

    #[test]
    fn render_scale_signature_ignores_non_scale_tempo_map_changes() {
        let base = TimelineState::default();
        let no_map = base.render_scale_signature();

        let initial = TempoPointData {
            id: "initial".to_string(),
            position_sec: 0.0,
            bpm: 120.0,
            numerator: Some(4),
            denominator: Some(4),
            scale: Some(TempoScaleData {
                key: Some("C".to_string()),
                name: None,
                notes: Some(base.project_scale_notes.clone()),
            }),
        };

        // 仅创建“初始点 = 工程基准”的 Tempo Map：实际生效音阶不变。
        let mut initial_only = base.clone();
        initial_only.tempo_map = Some(vec![initial.clone()]);
        assert_eq!(no_map, initial_only.render_scale_signature());

        // 只有 BPM / 拍号变化、没有音阶变化的变化点：实际生效音阶不变。
        let mut tempo_only = base.clone();
        tempo_only.tempo_map = Some(vec![
            initial.clone(),
            TempoPointData {
                id: "tempo".to_string(),
                position_sec: 5.0,
                bpm: 90.0,
                numerator: Some(3),
                denominator: Some(4),
                scale: None,
            },
        ]);
        assert_eq!(no_map, tempo_only.render_scale_signature());

        let scale_at = |sec: f64, key: &str| TempoPointData {
            id: format!("scale_{sec}"),
            position_sec: sec,
            bpm: 120.0,
            numerator: Some(4),
            denominator: Some(4),
            scale: Some(TempoScaleData {
                key: Some(key.to_string()),
                name: None,
                notes: None,
            }),
        };

        // 添加真正的音阶变化点：签名必须改变。
        let mut scaled = base.clone();
        scaled.tempo_map = Some(vec![initial.clone(), scale_at(5.0, "G")]);
        let scaled_sig = scaled.render_scale_signature();
        assert_ne!(no_map, scaled_sig);

        // 挪动真正的音阶变化点：签名必须改变。
        let mut moved = base.clone();
        moved.tempo_map = Some(vec![initial.clone(), scale_at(6.0, "G")]);
        assert_ne!(scaled_sig, moved.render_scale_signature());

        // 清除 Tempo Map：回到工程基准签名。
        let mut cleared = scaled.clone();
        cleared.tempo_map = None;
        assert_eq!(no_map, cleared.render_scale_signature());
    }

    #[test]
    fn tempo_map_normalize_materializes_initial_time_signature() {
        let mut timeline = TimelineState::default();
        timeline.bpm = 150.0;
        timeline.tempo_map = Some(vec![
            TempoPointData {
                id: "a".to_string(),
                position_sec: 0.0,
                bpm: 120.0,
                numerator: None,
                denominator: None,
                scale: None,
            },
            TempoPointData {
                id: "b".to_string(),
                position_sec: 2.0,
                bpm: 120.0,
                numerator: None,
                denominator: None,
                scale: None,
            },
            TempoPointData {
                id: "c".to_string(),
                position_sec: 4.0,
                bpm: 120.0,
                numerator: Some(6),
                denominator: Some(8),
                scale: None,
            },
        ]);
        timeline.normalize_tempo_map();
        let points = timeline.tempo_map.as_ref().unwrap();
        // 初始点必须显式携带拍号（工程基准记录不存在“之前”可跟随）。
        assert_eq!(points[0].numerator, Some(4));
        assert_eq!(points[0].denominator, Some(4));
        // 其它点保持“跟随之前的拍号”。
        assert_eq!(points[1].numerator, None);
        assert_eq!(points[1].denominator, None);
        // 生效拍号解析：跟随点解析为 4/4，显式点解析为 6/8。
        assert_eq!(
            TimelineState::effective_time_signature_at(points, 1),
            (4, 4)
        );
        assert_eq!(
            TimelineState::effective_time_signature_at(points, 2),
            (6, 8)
        );
    }

    #[test]
    fn patch_clips_state_updates_multiple_clips_in_one_pass() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.add_track(Some("Track".to_string()), None, None);
        timeline.add_clip(
            Some(track_id.clone()),
            Some("A".into()),
            Some(0.0),
            Some(1.0),
            None,
        );
        timeline.add_clip(Some(track_id), Some("B".into()), Some(1.0), Some(1.0), None);

        let ids: Vec<String> = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
        timeline.patch_clips_state(&[
            BulkClipStatePatch {
                clip_id: ids[0].clone(),
                patch: ClipStatePatch {
                    gain: Some(1.5),
                    ..Default::default()
                },
            },
            BulkClipStatePatch {
                clip_id: ids[1].clone(),
                patch: ClipStatePatch {
                    muted: Some(true),
                    fade_in_sec: Some(0.25),
                    ..Default::default()
                },
            },
        ]);

        assert_eq!(timeline.clips[0].gain, 1.5);
        assert!(timeline.clips[1].muted);
        assert_eq!(timeline.clips[1].fade_in_sec, 0.25);
    }

    #[test]
    fn ripple_track_mode_moves_following_clips_on_same_track_only() {
        let mut timeline = TimelineState::default();
        let track_a = timeline.add_track(Some("A".into()), None, None);
        let track_b = timeline.add_track(Some("B".into()), None, None);

        let a0 = timeline.add_clip(
            Some(track_a.clone()),
            Some("a0".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
        let a1 = timeline.add_clip(
            Some(track_a.clone()),
            Some("a1".into()),
            Some(2.0),
            Some(2.0),
            None,
        );
        let a2 = timeline.add_clip(
            Some(track_a.clone()),
            Some("a2".into()),
            Some(4.0),
            Some(2.0),
            None,
        );
        let b0 = timeline.add_clip(
            Some(track_b.clone()),
            Some("b0".into()),
            Some(2.0),
            Some(2.0),
            None,
        );

        let edited: Vec<&str> = vec![a1.as_str()];
        let affected: HashSet<String> = HashSet::from([track_a]);
        let shifted = timeline.ripple_shift_clips(&edited, Some(&affected), 2.0, 1.0, false);

        // A 轨上的后续剪辑 a2 右移 1s。
        assert!(shifted.contains(&a2));
        // 被编辑的 a1 与更早的 a0 不动。
        assert!(!shifted.contains(&a1));
        assert!(!shifted.contains(&a0));
        // B 轨上的 b0（与 a1 同时刻）不动（Track 模式）。
        assert!(!shifted.contains(&b0));

        assert_eq!(find_clip_start(&timeline, &a0), 0.0);
        assert_eq!(find_clip_start(&timeline, &a1), 2.0);
        assert_eq!(find_clip_start(&timeline, &a2), 5.0);
        assert_eq!(find_clip_start(&timeline, &b0), 2.0);
    }

    #[test]
    fn ripple_all_mode_moves_following_clips_on_every_track() {
        let mut timeline = TimelineState::default();
        let track_a = timeline.add_track(Some("A".into()), None, None);
        let track_b = timeline.add_track(Some("B".into()), None, None);

        let a1 = timeline.add_clip(
            Some(track_a.clone()),
            Some("a1".into()),
            Some(2.0),
            Some(2.0),
            None,
        );
        let a2 = timeline.add_clip(
            Some(track_a.clone()),
            Some("a2".into()),
            Some(4.0),
            Some(2.0),
            None,
        );
        let b0 = timeline.add_clip(
            Some(track_b.clone()),
            Some("b0".into()),
            Some(2.0),
            Some(4.0),
            None,
        );

        // 删除 a1（2~4s）：All 模式下所有轨道上 start >= 2 的后续剪辑左移 2s。
        let delta = 2.0 - 4.0; // origin - old_right_edge
        let edited: Vec<&str> = vec![a1.as_str()];
        let shifted = timeline.ripple_shift_clips(&edited, None, 2.0, delta, false);

        assert_eq!(find_clip_start(&timeline, &a2), 2.0);
        // b0 起点 2 >= origin 2，被平移左移 2s → 0。
        assert_eq!(find_clip_start(&timeline, &b0), 0.0);
        assert!(shifted.contains(&a2));
        assert!(shifted.contains(&b0));
    }

    #[test]
    fn ripple_off_and_zero_delta_are_noops() {
        let mut timeline = TimelineState::default();
        let track = timeline.add_track(Some("A".into()), None, None);
        let _a0 = timeline.add_clip(
            Some(track.clone()),
            Some("a0".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
        let a1 = timeline.add_clip(
            Some(track.clone()),
            Some("a1".into()),
            Some(2.0),
            Some(2.0),
            None,
        );
        let a2 = timeline.add_clip(
            Some(track.clone()),
            Some("a2".into()),
            Some(4.0),
            Some(2.0),
            None,
        );

        // delta = 0（如“锁定参数线关闭时的纯纵向/无位移编辑”）不产生平移。
        let shifted = timeline.ripple_shift_clips(&[a1.as_str()], None, 2.0, 0.0, false);
        assert!(shifted.is_empty());
        assert_eq!(find_clip_start(&timeline, &a2), 4.0);
    }

    #[test]
    fn create_clips_bulk_creates_multiple_snapshot_clips() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.add_track(Some("Track".to_string()), None, None);

        let created = timeline.create_clips_bulk(&CreateClipsBulkPayload {
            templates: vec![
                CreateClipTemplatePayload {
                    track_id: track_id.clone(),
                    name: "Snap A".into(),
                    start_sec: 1.0,
                    length_sec: 2.0,
                    source_clip_id: None,
                    source_path: Some("a.wav".into()),
                    gain: Some(1.25),
                    muted: Some(true),
                    source_start_sec: Some(0.3),
                    source_end_sec: Some(1.8),
                    playback_rate: Some(0.8),
                    reversed: Some(true),
                    loop_enabled: None,
                    snap_offset_sec: None,
                    fade_in_sec: Some(0.15),
                    fade_out_sec: Some(0.25),
                    fade_in_shape: Some(5.0),
                    fade_out_shape: Some(1.0),
                    fade_in_dir: Some(-0.25),
                    fade_out_dir: None,
                    auto_fade_in_sec: None,
                    auto_fade_out_sec: None,
                    linked_params: None,
                    midi_fill_gaps: Some(false),
                    midi_note_data: None,
                },
                CreateClipTemplatePayload {
                    track_id,
                    name: "Snap B".into(),
                    start_sec: 4.0,
                    length_sec: 1.5,
                    source_clip_id: None,
                    source_path: None,
                    gain: Some(0.9),
                    muted: Some(false),
                    source_start_sec: Some(0.0),
                    source_end_sec: Some(1.5),
                    playback_rate: Some(1.0),
                    reversed: Some(false),
                    loop_enabled: None,
                    snap_offset_sec: None,
                    fade_in_sec: Some(0.05),
                    fade_out_sec: Some(0.1),
                    fade_in_shape: Some(0.0),
                    fade_out_shape: Some(5.0),
                    fade_in_dir: None,
                    fade_out_dir: None,
                    auto_fade_in_sec: None,
                    auto_fade_out_sec: None,
                    linked_params: None,
                    midi_fill_gaps: Some(false),
                    midi_note_data: None,
                },
            ],
            select_created_clips: true,
        });

        assert_eq!(created.len(), 2);
        let first = timeline
            .clips
            .iter()
            .find(|clip| clip.id == created[0])
            .expect("first created clip");
        assert_eq!(first.name, "Snap A");
        assert!((first.start_sec - 1.0).abs() < 1e-6);
        assert_eq!(first.gain, 1.25);
        assert!(first.muted);
        assert!((first.source_start_sec - 0.3).abs() < 1e-6);
        assert!((first.source_end_sec - 1.8).abs() < 1e-6);
        assert_eq!(first.fade_in_shape, 5.0);
        assert_eq!(first.fade_out_shape, 1.0);
        assert_eq!(
            timeline.selected_clip_id.as_deref(),
            Some(created[0].as_str())
        );
    }

    #[test]
    fn create_clips_bulk_uses_source_clip_id_when_available() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.add_track(Some("Track".to_string()), None, None);
        let source_clip_id = timeline.add_clip(
            Some(track_id.clone()),
            Some("Source".into()),
            Some(0.5),
            Some(1.0),
            Some("source.wav".into()),
        );

        timeline.patch_clip_state(
            &source_clip_id,
            ClipStatePatch {
                gain: Some(1.5),
                muted: Some(true),
                source_start_sec: Some(0.25),
                source_end_sec: Some(0.9),
                playback_rate: Some(0.8),
                reversed: Some(true),
                fade_in_sec: Some(0.1),
                fade_out_sec: Some(0.2),
                fade_in_dir: Some(-0.5),
                fade_out_dir: Some(0.75),
                ..Default::default()
            },
        );
        if let Some(source_clip) = timeline
            .clips
            .iter_mut()
            .find(|clip| clip.id == source_clip_id)
        {
            source_clip.color = "amber".into();
        }

        let created = timeline.create_clips_bulk(&CreateClipsBulkPayload {
            templates: vec![CreateClipTemplatePayload {
                track_id: track_id.clone(),
                name: "Pasted".into(),
                start_sec: 3.0,
                length_sec: 1.0,
                source_clip_id: Some(source_clip_id.clone()),
                source_path: Some("source.wav".into()),
                gain: Some(1.5),
                muted: Some(true),
                source_start_sec: Some(0.25),
                source_end_sec: Some(0.9),
                playback_rate: Some(0.8),
                reversed: Some(true),
                loop_enabled: None,
                snap_offset_sec: None,
                fade_in_sec: Some(0.1),
                fade_out_sec: Some(0.2),
                fade_in_shape: Some(6.0),
                fade_out_shape: Some(3.0),
                fade_in_dir: Some(0.4),
                fade_out_dir: None,
                auto_fade_in_sec: None,
                auto_fade_out_sec: None,
                linked_params: None,
                midi_fill_gaps: Some(false),
                midi_note_data: None,
            }],
            select_created_clips: true,
        });

        let pasted = timeline
            .clips
            .iter()
            .find(|clip| clip.id == created[0])
            .expect("pasted clip");
        assert_eq!(pasted.name, "Pasted");
        assert!((pasted.start_sec - 3.0).abs() < 1e-6);
        assert_eq!(pasted.color, "amber");
        assert_eq!(pasted.source_path.as_deref(), Some("source.wav"));
        assert_eq!(pasted.gain, 1.5);
        assert!(pasted.muted);
        assert!(pasted.reversed);
        // Paste/clone path: explicit template fields override the source, dir is
        // clamped to [-1, 1].
        assert_eq!(pasted.fade_out_shape, 3.0);
        assert!((pasted.fade_in_dir - 0.4).abs() < 1e-9);
        assert!((pasted.fade_out_dir - 0.75).abs() < 1e-9);
    }

    #[test]
    fn duplicate_clips_bulk_duplicates_multiple_clips_with_delta() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.add_track(Some("Track".to_string()), None, None);
        timeline.add_clip(
            Some(track_id.clone()),
            Some("A".into()),
            Some(0.0),
            Some(1.0),
            None,
        );
        timeline.add_clip(Some(track_id), Some("B".into()), Some(2.0), Some(1.5), None);

        let source_ids: Vec<String> = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
        let created = timeline.duplicate_clips_bulk(&DuplicateClipsBulkPayload {
            source_clip_ids: source_ids,
            delta_sec: 1.25,
            track_mode: DuplicateClipsTrackMode::SameTrack,
            copy_linked_params: false,
            select_created_clips: true,
            apply_auto_crossfade: false,
            place_on_selected_track: false,
            rename_copies: None,
        });

        assert_eq!(created.len(), 2);
        assert_eq!(timeline.clips.len(), 4);
        assert!(timeline
            .clips
            .iter()
            .any(|clip| (clip.start_sec - 1.25).abs() < 1e-6));
        assert!(timeline
            .clips
            .iter()
            .any(|clip| (clip.start_sec - 3.25).abs() < 1e-6));
        assert!(timeline.clips.iter().any(|clip| clip.name == "A Copy"));
    }

    #[test]
    fn duplicate_clips_bulk_can_preserve_source_names() {
        let mut timeline = TimelineState::default();
        let track_id = timeline.add_track(Some("Track".to_string()), None, None);
        timeline.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(1.0), None);

        let source_clip_id = timeline.clips[0].id.clone();
        let created = timeline.duplicate_clips_bulk(&DuplicateClipsBulkPayload {
            source_clip_ids: vec![source_clip_id],
            delta_sec: 1.0,
            track_mode: DuplicateClipsTrackMode::SameTrack,
            copy_linked_params: false,
            select_created_clips: true,
            apply_auto_crossfade: false,
            place_on_selected_track: false,
            rename_copies: Some(false),
        });

        let duplicated = timeline
            .clips
            .iter()
            .find(|clip| clip.id == created[0])
            .expect("duplicated clip");
        assert_eq!(duplicated.name, "A");
    }

    #[test]
    fn duplicate_clips_bulk_new_tracks_follow_source_track_order() {
        let mut timeline = TimelineState::default();
        let low_track_id = timeline.add_track(Some("Low".to_string()), None, None);
        let high_track_id = timeline.add_track(Some("High".to_string()), None, None);
        timeline.add_clip(
            Some(low_track_id.clone()),
            Some("Low Clip".into()),
            Some(0.0),
            Some(1.0),
            None,
        );
        timeline.add_clip(
            Some(high_track_id.clone()),
            Some("High Clip".into()),
            Some(0.5),
            Some(1.0),
            None,
        );

        let low_clip_id = timeline
            .clips
            .iter()
            .find(|clip| clip.track_id == low_track_id)
            .map(|clip| clip.id.clone())
            .expect("low clip");
        let high_clip_id = timeline
            .clips
            .iter()
            .find(|clip| clip.track_id == high_track_id)
            .map(|clip| clip.id.clone())
            .expect("high clip");

        let created = timeline.duplicate_clips_bulk(&DuplicateClipsBulkPayload {
            source_clip_ids: vec![high_clip_id, low_clip_id],
            delta_sec: 2.0,
            track_mode: DuplicateClipsTrackMode::NewTracks,
            copy_linked_params: false,
            select_created_clips: false,
            apply_auto_crossfade: false,
            place_on_selected_track: false,
            rename_copies: None,
        });

        assert_eq!(created.len(), 2);

        let original_track_count = 3usize;
        let new_tracks = &timeline.tracks[original_track_count..];
        assert_eq!(new_tracks.len(), 2);

        let low_duplicate = timeline
            .clips
            .iter()
            .find(|clip| created.contains(&clip.id) && clip.name == "Low Clip Copy")
            .expect("low duplicate");
        let high_duplicate = timeline
            .clips
            .iter()
            .find(|clip| created.contains(&clip.id) && clip.name == "High Clip Copy")
            .expect("high duplicate");

        assert_eq!(low_duplicate.track_id, new_tracks[0].id);
        assert_eq!(high_duplicate.track_id, new_tracks[1].id);
    }

    #[test]
    fn clip_formant_morph_defaults_to_disabled_when_missing() {
        // Clip 的持久化格式是 snake_case（无 rename_all）；
        // formant_morph 缺失时必须反序列化为 None → 默认禁用。
        let json = serde_json::json!({
            "id": "clip-1",
            "track_id": "track_main",
            "name": "clip",
            "start_sec": 0.0,
            "length_sec": 1.0,
            "color": "#fff",
            "source_path": "demo.wav",
            "source_start_sec": 0.0,
            "source_end_sec": 1.0,
            "playback_rate": 1.0,
            "gain": 1.0,
            "muted": false,
            "fade_in_sec": 0.0,
            "fade_out_sec": 0.0,
            "fade_in_curve": "sine",
            "fade_out_curve": "sine"
        });

        let clip: Clip = serde_json::from_value(json).expect("clip should deserialize");
        let morph = clip.formant_morph.unwrap_or_default();
        assert!(!morph.enabled);
        assert_eq!(morph.target_f1_hz, 800.0);
        assert_eq!(morph.target_f2_hz, 1400.0);
        assert!((morph.strength - 0.50).abs() < 1e-6);
    }

    #[test]
    fn pitch_reference_content_duration_uses_note_extent() {
        // 纯音高参考块（无源媒体）的内容时长 = 音符内容最大结束时间，
        // 使其 Loop 周期与普通媒体 Clip 完全一致（回绕整个内容，
        // 而非窗口跨度 [source_start, source_end]）。
        let json = serde_json::json!({
            "id": "pref-1",
            "track_id": "track_main",
            "name": "ref",
            "start_sec": 0.0,
            "length_sec": 4.5,
            "source_start_sec": 1.0,
            "source_end_sec": 2.0,
            "playback_rate": 1.0,
            "gain": 1.0,
            "muted": false,
            "fade_in_sec": 0.0,
            "fade_out_sec": 0.0,
            "fade_in_curve": "sine",
            "fade_out_curve": "sine",
            "midi_note_data": [
                {"startSec": 1.0, "endSec": 2.0, "note": 60.0, "velocity": 100, "channel": 0},
                {"startSec": 2.0, "endSec": 4.25, "note": 62.0, "velocity": 100, "channel": 0}
            ]
        });
        let mut clip: Clip = serde_json::from_value(json).expect("clip should deserialize");
        assert!(clip.source_path.is_none());
        assert_eq!(
            clip_source_media_duration_sec(&clip),
            Some(4.25),
            "content duration must come from note extent"
        );
        assert_ne!(
            clip_loop_cycle_span_sec(&clip),
            Some(1.0),
            "must NOT use the window span as cycle"
        );
        clip.loop_enabled = true;
        assert_eq!(clip_loop_cycle_span_sec(&clip), Some(4.25));
    }

    // ── split_clips_at tests ──────────────────────────────────────

    /// Split a grouped clip: left half keeps original group_id, right half gets new group_id.
    #[test]
    fn split_clips_at_basic() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);
        let c1 = tl.add_clip(
            Some(tid.clone()),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
        let c2 = tl.add_clip(Some(tid), Some("B".into()), Some(3.0), Some(2.0), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        let orig_group = tl
            .clips
            .iter()
            .find(|c| c.id == c1)
            .unwrap()
            .group_id
            .clone();
        assert!(orig_group.is_some());

        tl.split_clips_at(&[c1.clone()], 1.0);

        // Left half (start_sec ≈ 0.0) keeps original group
        let left = tl
            .clips
            .iter()
            .find(|c| c.start_sec < 0.5 && c.id != c2)
            .unwrap();
        assert_eq!(left.group_id, orig_group);

        // Right half (start_sec ≈ 1.0) gets new group
        let right = tl
            .clips
            .iter()
            .find(|c| c.start_sec >= 0.5 && c.id != c2)
            .unwrap();
        assert!(right.group_id.is_some());
        assert_ne!(right.group_id, orig_group);
    }

    /// Right-side group with only 1 member gets dissolved after split.
    #[test]
    fn split_clips_at_dissolves_small_groups() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);
        // c1 at 0.0..2.0, c2 at 0.5..0.8 (entirely left of split point at 1.0)
        let c1 = tl.add_clip(
            Some(tid.clone()),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
        let c2 = tl.add_clip(Some(tid), Some("B".into()), Some(0.5), Some(0.3), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        // Split c1 at 1.0. c2 starts at 0.5 < 1.0 so stays in original (left) group.
        // Left group: left half of c1 + c2 = 2 members → survives.
        // Right group: right half of c1 only → 1 member → dissolved.
        tl.split_clips_at(&[c1.clone()], 1.0);

        // Right half of c1 should have no group (dissolved)
        let right_half = tl
            .clips
            .iter()
            .find(|c| c.start_sec >= 0.9 && c.id != c2)
            .unwrap();
        assert!(
            right_half.group_id.is_none(),
            "right half should have no group"
        );

        // Left half and c2 should still be in the original group
        let left_half = tl
            .clips
            .iter()
            .find(|c| c.start_sec < 0.5 && c.id != c2)
            .unwrap();
        assert!(left_half.group_id.is_some(), "left half should keep group");
        assert!(
            tl.clips
                .iter()
                .find(|c| c.id == c2)
                .unwrap()
                .group_id
                .is_some(),
            "c2 should keep group"
        );
    }

    /// Unsplit clip entirely to the right of the split point moves to the new group.
    #[test]
    fn split_clips_at_unsplit_member_to_right() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);
        let c1 = tl.add_clip(
            Some(tid.clone()),
            Some("left".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
        let c2 = tl.add_clip(Some(tid), Some("right".into()), Some(2.5), Some(1.0), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        let orig_group = tl
            .clips
            .iter()
            .find(|c| c.id == c1)
            .unwrap()
            .group_id
            .clone();

        // Split c1 at 1.0; c2 is at 2.5 > 1.0 so it goes to the right group
        tl.split_clips_at(&[c1.clone()], 1.0);

        // c2 should have moved to the new right group
        let c2_after = tl.clips.iter().find(|c| c.id == c2).unwrap();
        assert!(c2_after.group_id.is_some());
        assert_ne!(c2_after.group_id, orig_group);
    }

    /// Unsplit clip to the left of the split point keeps the original group.
    #[test]
    fn split_clips_at_unsplit_member_stays_left() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);
        let c1 = tl.add_clip(
            Some(tid.clone()),
            Some("early".into()),
            Some(0.0),
            Some(1.0),
            None,
        );
        let c2 = tl.add_clip(Some(tid), Some("later".into()), Some(3.0), Some(2.0), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        let orig_group = tl
            .clips
            .iter()
            .find(|c| c.id == c1)
            .unwrap()
            .group_id
            .clone();

        // Split c2 at 4.0; c1 is at 0.0 < 4.0 so it stays in original group
        tl.split_clips_at(&[c2.clone()], 4.0);

        let c1_after = tl.clips.iter().find(|c| c.id == c1).unwrap();
        assert_eq!(c1_after.group_id, orig_group);
    }

    /// Splitting clips from different groups simultaneously assigns distinct right-side groups.
    #[test]
    fn split_clips_at_mixed_groups() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);

        // Group A
        let a1 = tl.add_clip(
            Some(tid.clone()),
            Some("A1".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
        let a2 = tl.add_clip(
            Some(tid.clone()),
            Some("A2".into()),
            Some(3.0),
            Some(1.0),
            None,
        );
        tl.group_clips(&[a1.clone(), a2.clone()]);
        let group_a = tl
            .clips
            .iter()
            .find(|c| c.id == a1)
            .unwrap()
            .group_id
            .clone();

        // Group B
        let b1 = tl.add_clip(Some(tid), Some("B1".into()), Some(5.0), Some(2.0), None);
        let b2 = tl.add_clip(None, Some("B2".into()), Some(8.0), Some(1.0), None);
        tl.group_clips(&[b1.clone(), b2.clone()]);
        let group_b = tl
            .clips
            .iter()
            .find(|c| c.id == b1)
            .unwrap()
            .group_id
            .clone();

        assert_ne!(group_a, group_b);

        // Split one clip from each group（切割点必须落在对应 clip 范围内，
        // 否则 split 是无操作、不会产生右组 —— 见 split_clips_at 的 clamp 规则）
        tl.split_clips_at(&[a1.clone()], 1.0);
        tl.split_clips_at(&[b1.clone()], 6.0);

        // Each group should have at least 2 distinct group_ids after split (original + new)
        let mut groups: std::collections::HashSet<String> = std::collections::HashSet::new();
        for clip in &tl.clips {
            if let Some(ref gid) = clip.group_id {
                groups.insert(gid.clone());
            }
        }
        // Original group_a, new right group for A, original group_b, new right group for B
        assert!(
            groups.len() >= 4,
            "expected >=4 groups, got {}",
            groups.len()
        );
    }

    #[test]
    fn split_clip_with_transition_fade_only_sets_boundary_fades() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 本用例验证**非 Loop** 的分割窗口推进语义（新建 Clip 默认 Loop 开启）。
            clip.loop_enabled = false;
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 5.0;
            clip.playback_rate = 2.0;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::FadeOnly,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: Some("lateSlight".to_string()),
            overlap_fades: false,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert!((left.length_sec - 0.5).abs() < 1e-9);
        assert!((left.fade_out_sec - 0.1).abs() < 1e-9);
        assert!((left.source_end_sec - 2.0).abs() < 1e-9);
        assert!((right.start_sec - 0.5).abs() < 1e-9);
        assert!((right.length_sec - 1.5).abs() < 1e-9);
        assert!((right.fade_in_sec - 0.1).abs() < 1e-9);
        assert!((right.source_start_sec - 2.0).abs() < 1e-9);
        // 新版预设（lateSlight = 形状 2）：淡出侧默认曲率 -1、淡入侧 +1。
        assert_eq!(left.fade_out_shape, 2.0);
        assert_eq!(left.fade_out_dir, -1.0);
        assert_eq!(right.fade_in_shape, 2.0);
        assert_eq!(right.fade_in_dir, 1.0);
    }

    #[test]
    fn split_transition_keep_preserves_original_fade_curves() {
        // "keep"（curve=None）= 分割后保留原 Clip 的淡化曲线类型，不再修改：
        // 只写淡化时长，不碰 shape/dir。
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.fade_out_shape = 3.5; // 小数变体（基础族 3）
            clip.fade_out_dir = 0.6;
            clip.fade_in_shape = 5.1;
            clip.fade_in_dir = -0.4;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::FadeOnly,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: None, // "keep" → 调用层映射为 None
            overlap_fades: false,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        // 时长照常写入……
        assert!((left.fade_out_sec - 0.1).abs() < 1e-9);
        assert!((right.fade_in_sec - 0.1).abs() < 1e-9);
        // ……但曲线类型（形状/曲率）原样保留，包括小数变体。
        assert_eq!(left.fade_out_shape, 3.5);
        assert_eq!(left.fade_out_dir, 0.6);
        assert_eq!(right.fade_in_shape, 5.1);
        assert_eq!(right.fade_in_dir, -0.4);
    }

    #[test]
    fn split_clip_with_transition_extend_overlap_preserves_source_position() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 非 Loop 语义用例（新建 Clip 默认 Loop 开启）。
            clip.loop_enabled = false;
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 5.0;
            clip.playback_rate = 2.0;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::ExtendOverlap,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: Some("sSlight".to_string()),
            overlap_fades: true,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert!((left.length_sec - 0.6).abs() < 1e-9);
        assert!((left.source_end_sec - 2.2).abs() < 1e-9);
        assert!((left.auto_fade_out_sec - 0.2).abs() < 1e-9);
        assert!((left.fade_out_sec - 0.0).abs() < 1e-9);
        assert!((right.start_sec - 0.4).abs() < 1e-9);
        assert!((right.length_sec - 1.6).abs() < 1e-9);
        assert!((right.source_start_sec - 1.8).abs() < 1e-9);
        assert!((right.auto_fade_in_sec - 0.2).abs() < 1e-9);
        assert!((right.fade_in_sec - 0.0).abs() < 1e-9);

        // At the split point, both clips must still reference the same source position.
        let left_source_at_split = left.source_end_sec - (left.length_sec - 0.5) * 2.0;
        let right_source_at_split = right.source_start_sec + (0.5 - right.start_sec) * 2.0;
        assert!((left_source_at_split - 2.0).abs() < 1e-9);
        assert!((right_source_at_split - 2.0).abs() < 1e-9);
    }

    #[test]
    fn split_clip_loop_keeps_window_and_wraps_right_piece() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("Looped".into()),
            Some(0.0),
            Some(6.0),
            None,
        );
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 媒体总长 10s；Clip 锚点从 1s 进入，窗口 [1,3]。
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 3.0;
            clip.duration_sec = Some(10.0);
            clip.playback_rate = 1.0;
            clip.loop_enabled = true;
        }

        let options = SplitTransitionOptions {
            enabled: false,
            mode: SplitTransitionMode::FadeOnly,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.0,
            duration_percent: 1.0,
            curve: None,
            overlap_fades: false,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 2.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();

        // 左段：保持锚点与窗口不变，仅缩短长度。
        assert!((left.length_sec - 2.5).abs() < 1e-9);
        assert!((left.source_start_sec - 1.0).abs() < 1e-9);
        assert!((left.source_end_sec - 3.0).abs() < 1e-9);
        assert!(left.loop_enabled);

        // 右段：切割点在整个媒体上的环绕位置 = floor_mod(1 + 2.5, 10) = 3.5
        // （Loop 模式下音频只依赖锚点与媒体时长，另一端字段保持原值）。
        assert!((right.start_sec - 2.5).abs() < 1e-9);
        assert!((right.length_sec - 3.5).abs() < 1e-9);
        assert!(right.loop_enabled);
        assert!((right.source_start_sec - 3.5).abs() < 1e-9);
        assert!((right.source_end_sec - 3.0).abs() < 1e-9);
    }

    #[test]
    fn split_transition_extend_overlap_allows_loop_growth_beyond_media() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 媒体只有 2 秒（duration 未设置时 source_file_duration_sec 回退
            // 到 source_end），窗口即整个媒体；Loop 下延伸不再受其限制。
            clip.source_start_sec = 0.0;
            clip.source_end_sec = 2.0;
            clip.duration_sec = Some(2.0);
            clip.loop_enabled = true;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::ExtendOverlap,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.5,
            duration_percent: 1.0,
            curve: None,
            overlap_fades: false,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 1.0, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        // 左段向右延伸 0.5s：源窗口保持不变（循环回绕填充）。
        assert!((left.length_sec - 1.5).abs() < 1e-9);
        assert!((left.source_start_sec - 0.0).abs() < 1e-9);
        assert!((left.source_end_sec - 2.0).abs() < 1e-9);
        // 右段向左延伸 0.5s：锚点沿消费方向**等价回退并环绕**
        // （floor_mod(1.0 − 0.5, 2) = 0.5），保持原有内容的绝对时间线位置
        // —— 与非 Loop 分支"扩展窗口保内容位置"的语义一致；否则重叠区内
        // 左尾与右头会播放错相的内容。另一端字段在 Loop 下不参与渲染。
        assert!((right.start_sec - 0.5).abs() < 1e-9);
        assert!((right.length_sec - 1.5).abs() < 1e-9);
        assert!((right.source_start_sec - 0.5).abs() < 1e-9);
        assert!((right.source_end_sec - 2.0).abs() < 1e-9);
    }

    #[test]
    fn split_clip_with_transition_percent_uses_combined_clip_length() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::FadeOnly,
            duration_unit: SplitTransitionDurationUnit::Percent,
            duration_sec: 999.0,
            duration_percent: 1.0,
            curve: None,
            overlap_fades: false,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        // 前 0.5s + 后 1.5s = 2.0s，1% = 0.02s
        assert!((left.fade_out_sec - 0.02).abs() < 1e-9);
        assert!((right.fade_in_sec - 0.02).abs() < 1e-9);
        assert!((left.length_sec - 0.5).abs() < 1e-9);
        assert!((right.length_sec - 1.5).abs() < 1e-9);
    }

    #[test]
    fn split_clip_with_transition_overlap_without_fades_keeps_zero_fades() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::ExtendOverlap,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: None,
            overlap_fades: false,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert!((left.fade_out_sec - 0.0).abs() < 1e-9);
        assert!((right.fade_in_sec - 0.0).abs() < 1e-9);
        assert!((left.length_sec - 0.6).abs() < 1e-9);
        assert!((right.start_sec - 0.4).abs() < 1e-9);
        assert!((left.auto_fade_out_sec - 0.0).abs() < 1e-9);
        assert!((right.auto_fade_in_sec - 0.0).abs() < 1e-9);
    }

    #[test]
    fn split_clip_with_transition_overlap_handles_reversed_clips() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 非 Loop 语义用例（新建 Clip 默认 Loop 开启）。
            clip.loop_enabled = false;
            clip.source_start_sec = 3.0;
            clip.source_end_sec = 7.0;
            clip.playback_rate = 2.0;
            clip.reversed = true;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::ExtendOverlap,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: Some("sSlight".to_string()),
            overlap_fades: true,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.5, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert!((left.length_sec - 0.6).abs() < 1e-9);
        assert!((left.source_start_sec - 5.8).abs() < 1e-9);
        assert!((right.start_sec - 0.4).abs() < 1e-9);
        assert!((right.length_sec - 1.6).abs() < 1e-9);
        assert!((right.source_end_sec - 6.2).abs() < 1e-9);
    }

    #[test]
    fn split_clip_with_transition_overlap_clamps_near_timeline_start() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(2.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            clip.source_start_sec = 1.0;
            clip.source_end_sec = 5.0;
            clip.playback_rate = 2.0;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::ExtendOverlap,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: None,
            overlap_fades: true,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.005, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert!((left.length_sec - 0.105).abs() < 1e-9);
        assert!((right.start_sec - 0.0).abs() < 1e-9);
        assert!((right.length_sec - 2.0).abs() < 1e-9);
        assert!((right.source_start_sec - 1.0).abs() < 1e-9);
        assert!((left.auto_fade_out_sec - 0.105).abs() < 1e-9);
        assert!((right.auto_fade_in_sec - 0.105).abs() < 1e-9);
    }

    #[test]
    fn split_clip_with_transition_overlap_grows_into_silence_beyond_material() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("A".into()), Some(0.0), Some(1.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 非 Loop：向左/向右延伸均无界 —— 左段尾部越过素材末尾(3.0)的
            // 部分渲染为静音；不再按素材可用量钳制。
            clip.loop_enabled = false;
            clip.source_start_sec = 2.0;
            clip.source_end_sec = 3.0;
            clip.duration_sec = Some(3.0);
            clip.playback_rate = 1.0;
        }

        let options = SplitTransitionOptions {
            enabled: true,
            mode: SplitTransitionMode::ExtendOverlap,
            duration_unit: SplitTransitionDurationUnit::Seconds,
            duration_sec: 0.1,
            duration_percent: 1.0,
            curve: None,
            overlap_fades: true,
        };
        let right_id = tl
            .split_clip_with_transition(&clip_id, 0.95, &options)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();
        assert!((left.length_sec - 1.05).abs() < 1e-9);
        assert!((left.source_end_sec - 3.05).abs() < 1e-9);
        assert!((right.start_sec - 0.85).abs() < 1e-9);
        // 右段原长 0.05 + 头部延伸 0.1 = 0.15（尾部保持在原 clip 末端 1.0，
        // 延伸只向外侧扩展）。
        assert!((right.length_sec - 0.15).abs() < 1e-9);
        assert!((right.source_start_sec - 2.85).abs() < 1e-9);
        // 重叠 = 左尾增长 + 右头增长 = 0.2；写入时按各自长度钳制
        //（右段长 0.15 < 重叠 0.2）。
        assert!((left.auto_fade_out_sec - 0.2).abs() < 1e-9);
        assert!((right.auto_fade_in_sec - 0.15).abs() < 1e-9);
    }

    #[test]
    fn split_clip_clears_auto_fades_on_cut_edges() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(Some(track_id), Some("B".into()), Some(0.0), Some(3.0), None);
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // B 与左侧邻居的自动交叉淡化在 fadeIn，与右侧邻居的自动交叉淡化在 fadeOut。
            clip.fade_in_sec = 0.1;
            clip.fade_out_sec = 0.2;
            clip.auto_fade_in_sec = 0.4;
            clip.auto_fade_out_sec = 0.5;
        }

        let right_id = tl
            .split_clip(&clip_id, 1.5)
            .expect("split should create right clip");

        let left = tl.clips.iter().find(|c| c.id == clip_id).unwrap();
        let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();

        // 切割产生的新边缘（左 clip 右缘、右 clip 左缘）不继承任何淡化（手动/自动）。
        assert!((left.fade_out_sec - 0.0).abs() < 1e-9);
        assert!((left.auto_fade_out_sec - 0.0).abs() < 1e-9);
        assert!((right.fade_in_sec - 0.0).abs() < 1e-9);
        assert!((right.auto_fade_in_sec - 0.0).abs() < 1e-9);

        // 外缘仍然保留对应侧淡化，并按新长度钳制。
        assert!((left.fade_in_sec - 0.1).abs() < 1e-9);
        assert!((left.auto_fade_in_sec - 0.4).abs() < 1e-9);
        assert!((right.fade_out_sec - 0.2).abs() < 1e-9);
        assert!((right.auto_fade_out_sec - 0.5).abs() < 1e-9);
    }

    #[test]
    fn check_source_files_changed_uses_persisted_fingerprint_after_open_baseline_refresh() {
        let test_path = std::env::temp_dir().join(format!(
            "hifishifter_fingerprint_check_{}.bin",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&test_path);

        // 保存工程时磁盘上是内容 A。
        std::fs::write(&test_path, b"content-A").unwrap();
        let saved_fingerprint =
            crate::audio_utils::compute_file_fingerprint(&test_path).expect("fingerprint A");

        // 关闭工程后，用户在资源管理器中用内容 B 替换了同名文件。
        std::fs::write(&test_path, b"content-B-different").unwrap();

        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(1.0),
            Some(test_path.display().to_string()),
        );

        {
            let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
            // 模拟打开工程后的状态：持久化指纹仍是 A，但 mtime/size 已按 B 刷新。
            let meta = std::fs::metadata(&test_path).unwrap();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            clip.source_file_fingerprint = Some(saved_fingerprint);
            clip.source_file_size = Some(meta.len());
            clip.source_file_mtime = mtime;
            clip.sync_take_from_flat();
        }

        let changed = tl.check_source_files_changed().changed;
        assert!(
            changed.iter().any(|item| {
                item.clip_id == clip_id
                    && item.source_path == test_path.display().to_string()
                    && item.change == "modified"
            }),
            "fingerprint change must be detected even when mtime/size match the just-opened file"
        );

        // 用户重新加载 B 后，运行时指纹更新为 B，不应再报告变更。
        if let Some(clip) = tl.clips.iter_mut().find(|c| c.id == clip_id) {
            clip.source_file_fingerprint = crate::audio_utils::compute_file_fingerprint(&test_path);
            clip.sync_take_from_flat();
        }
        let changed = tl.check_source_files_changed().changed;
        assert!(
            changed.is_empty(),
            "same fingerprint must not report a change"
        );

        let _ = std::fs::remove_file(&test_path);
    }

    /// 验证"锁定参数线"移动剪辑时，pitch / tension / 自动化曲线（extra curves）
    /// 都必须搬运到新位置，旧位置恢复默认。
    #[test]
    fn move_clips_linked_params_carry_all_curves() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl.add_clip(
            Some(track_id.clone()),
            Some("MoveLinked".into()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );
        let root_track_id = tl.resolve_root_track_id(&track_id).unwrap();
        tl.ensure_params_for_root(&root_track_id);

        // 构造：用户编辑过的 pitch、非默认 tension、非默认 volume 自动化曲线。
        {
            let entry = tl.params_by_root_track.get_mut(&root_track_id).unwrap();
            let frames = (2.0f64 * 1000.0 / 5.0).ceil() as usize; // 2s @ 5ms = 400 帧
            for i in 0..frames {
                entry.pitch_edit[i] = 60.0 + (i as f32) * 0.1;
                entry.tension_edit[i] = 10.0 + (i as f32) * 0.05;
            }
            entry
                .extra_curves
                .insert("volume".to_string(), vec![0.8; frames]);
            entry.pitch_edit_user_modified = true;
        }

        // 向右移动 1 秒（锁定参数线开启）。
        tl.move_clips(
            &[MoveClipPayload {
                clip_id: clip_id.clone(),
                start_sec: 1.0,
                track_id: None,
            }],
            true,
        );

        let fp = 5.0f64;
        let old_start = 0.0f64;
        let new_start = 1.0f64;
        let frames = (2.0 * 1000.0 / fp).ceil() as usize;
        let old_idx = (old_start * 1000.0 / fp) as usize; // 0
        let new_idx = (new_start * 1000.0 / fp) as usize; // 200

        let entry = tl.params_by_root_track.get(&root_track_id).unwrap();

        // 新范围：pitch / tension / volume 都应搬运到位。
        assert!(
            (entry.pitch_edit[new_idx] - 60.0).abs() < 1e-4,
            "pitch must follow the clip, got {}",
            entry.pitch_edit[new_idx]
        );
        assert!(
            (entry.tension_edit[new_idx] - 10.0).abs() < 1e-4,
            "tension must follow the clip, got {}",
            entry.tension_edit[new_idx]
        );
        let volume = entry
            .extra_curves
            .get("volume")
            .expect("volume curve must exist");
        assert!(
            (volume[new_idx] - 0.8).abs() < 1e-4,
            "volume curve must follow the clip, got {}",
            volume[new_idx]
        );

        // 旧范围：应恢复默认（tension=0、volume 默认值；pitch 在用户编辑过时被清 0）。
        assert!(
            entry.pitch_edit[old_idx] == 0.0,
            "vacated pitch range must be cleared, got {}",
            entry.pitch_edit[old_idx]
        );
        assert!(
            entry.tension_edit[old_idx] == 0.0,
            "vacated tension range must be cleared, got {}",
            entry.tension_edit[old_idx]
        );
        assert!(
            (volume[old_idx] - 1.0).abs() < 1e-4,
            "vacated volume range must be restored to default 1.0, got {}",
            volume[old_idx]
        );

        let _ = frames;
    }

    /// 验证"锁定参数线"拉伸剪辑时，pitch / tension / 自动化曲线都被时域映射
    /// 到新范围，且旧范围中不再被新范围覆盖的帧恢复为参考值。
    /// 回归背景：旧前端实现只映射 pitch+tension，其余自动化曲线遗留在旧位置。
    #[test]
    fn stretch_linked_params_maps_all_curves() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let _clip_id = tl.add_clip(
            Some(track_id.clone()),
            Some("StretchLinked".into()),
            Some(1.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );
        let root_track_id = tl.resolve_root_track_id(&track_id).unwrap();
        tl.ensure_params_for_root(&root_track_id);

        // 在 1~3s（帧 200..600）写入非默认曲线：pitch=60+0.1i、tension=10、volume=0.8。
        {
            let entry = tl.params_by_root_track.get_mut(&root_track_id).unwrap();
            let start_f = 200usize;
            for i in 0..400usize {
                entry.pitch_edit[start_f + i] = 60.0 + (i as f32) * 0.1;
                entry.tension_edit[start_f + i] = 10.0;
            }
            let mut volume = vec![1.0f32; entry.pitch_edit.len()];
            for value in &mut volume[start_f..start_f + 400] {
                *value = 0.8;
            }
            entry.extra_curves.insert("volume".to_string(), volume);
            entry.pitch_edit_user_modified = true;
        }

        // 拉伸并左移：[1.0s, 3.0s) → [0.0s, 1.0s)。
        tl.stretch_linked_params_in_root_range(
            &root_track_id,
            &[StretchLinkedRangeSec {
                old_start_sec: 1.0,
                old_length_sec: 2.0,
                new_start_sec: 0.0,
                new_length_sec: 1.0,
            }],
        );

        let entry = tl.params_by_root_track.get(&root_track_id).unwrap();
        // 新范围帧 0..200 应有映射后的值；旧范围帧 200..600 中不再被覆盖的
        // 部分应恢复参考值（tension→0、volume→1.0、pitch→pitch_orig=0）。
        assert!(
            (entry.tension_edit[0] - 10.0).abs() < 1e-4 && (entry.tension_edit[100] - 10.0).abs() < 1e-4,
            "tension must follow the stretched clip"
        );
        assert!(
            entry.tension_edit[250] == 0.0,
            "vacated tension range must be reset, got {}",
            entry.tension_edit[250]
        );

        let volume = entry
            .extra_curves
            .get("volume")
            .expect("volume curve must exist");
        assert!(
            (volume[0] - 0.8).abs() < 1e-4 && (volume[100] - 0.8).abs() < 1e-4,
            "volume curve must follow the stretched clip"
        );
        assert!(
            (volume[250] - 1.0).abs() < 1e-4,
            "vacated volume range must be restored to default, got {}",
            volume[250]
        );

        assert!(
            (entry.pitch_edit[0] - 60.0).abs() < 1e-4,
            "pitch must follow the stretched clip, got {}",
            entry.pitch_edit[0]
        );
        assert!(
            entry.pitch_edit[250] == 0.0,
            "vacated pitch range must be restored to orig, got {}",
            entry.pitch_edit[250]
        );
        assert!(
            entry.pitch_edit_user_modified,
            "mapped pitch edits must keep the user-modified flag"
        );

        // 没有数据的参数不应被映射过程物化出来。
        assert!(
            !entry.extra_curves.contains_key("breath_gain"),
            "stretch mapping must not materialize curves for untouched params"
        );
    }

    /// 跨根轨道移动：曲线必须从旧 root 搬到新 root（旧 root 恢复默认）。
    #[test]
    fn move_clips_linked_params_cross_root_track() {
        let mut tl = TimelineState::default();
        let track_a = tl.tracks[0].id.clone();
        let track_b = tl.add_track(Some("B".to_string()), None, None);
        let clip_id = tl.add_clip(
            Some(track_a.clone()),
            Some("CrossRoot".into()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );

        let root_a = tl.resolve_root_track_id(&track_a).unwrap();
        let root_b = tl.resolve_root_track_id(&track_b).unwrap();
        tl.ensure_params_for_root(&root_a);
        {
            let entry = tl.params_by_root_track.get_mut(&root_a).unwrap();
            for i in 0..400usize {
                entry.tension_edit[i] = 10.0;
            }
            entry
                .extra_curves
                .insert("volume".to_string(), vec![0.8; 400]);
        }

        tl.move_clips(
            &[MoveClipPayload {
                clip_id: clip_id.clone(),
                start_sec: 0.0,
                track_id: Some(track_b.clone()),
            }],
            true,
        );

        let entry_a = tl.params_by_root_track.get(&root_a).unwrap();
        let entry_b = tl.params_by_root_track.get(&root_b).unwrap();
        assert!(
            (entry_b.tension_edit[0] - 10.0).abs() < 1e-4,
            "tension must follow the clip to the new root track"
        );
        assert!(
            (entry_b
                .extra_curves
                .get("volume")
                .expect("volume must exist on new root")[0]
                - 0.8)
                .abs()
                < 1e-4,
            "volume must follow the clip to the new root track"
        );
        assert!(
            entry_a.tension_edit[0] == 0.0,
            "old root range must be cleared after cross-root move"
        );
    }

    /// 拖拽复制（copyLinkedParams）：副本必须携带全部曲线，且不影响源曲线。
    #[test]
    fn duplicate_clips_bulk_copies_all_linked_curves() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl.add_clip(
            Some(track_id.clone()),
            Some("DupLinked".into()),
            Some(0.0),
            Some(2.0),
            Some("C:/audio/a.wav".into()),
        );
        let root = tl.resolve_root_track_id(&track_id).unwrap();
        tl.ensure_params_for_root(&root);
        {
            let entry = tl.params_by_root_track.get_mut(&root).unwrap();
            for i in 0..400usize {
                entry.tension_edit[i] = 10.0;
            }
            entry
                .extra_curves
                .insert("volume".to_string(), vec![0.8; 400]);
        }

        let created = tl.duplicate_clips_bulk(&DuplicateClipsBulkPayload {
            source_clip_ids: vec![clip_id],
            delta_sec: 3.0,
            track_mode: DuplicateClipsTrackMode::SameTrack,
            copy_linked_params: true,
            select_created_clips: true,
            apply_auto_crossfade: false,
            place_on_selected_track: false,
            rename_copies: None,
        });
        assert_eq!(created.len(), 1);

        let entry = tl.params_by_root_track.get(&root).unwrap();
        // 源范围保持不变。
        assert!((entry.tension_edit[0] - 10.0).abs() < 1e-4, "source tension intact");
        // 新范围（3s → 帧 600）携带副本曲线。
        assert!(
            (entry.tension_edit[600] - 10.0).abs() < 1e-4,
            "duplicate must carry tension, got {}",
            entry.tension_edit[600]
        );
        assert!(
            (entry
                .extra_curves
                .get("volume")
                .expect("volume must exist")[600]
                - 0.8)
                .abs()
                < 1e-4,
            "duplicate must carry volume curve"
        );
    }

    /// ⭐ 核心回归（用户实测场景）：extra 曲线不是工程长度 ——
    /// `set_param_frames` 只把它增长到最后一次写入的帧。当剪辑范围超出
    /// 曲线数组末尾（剪辑包含该参数的最后一个有效点、其后全是默认值）时，
    /// 旧实现的 `get(start..end)` 返回 None → 提取为空 → 移动时旧范围被
    /// 清除而新范围什么都没写，曲线被整体销毁（表现为"被初始化"）。
    #[test]
    fn move_clips_survives_extra_curve_shorter_than_clip_range() {
        let mut tl = TimelineState::default();
        let track_id = tl.tracks[0].id.clone();
        let clip_id = tl.add_clip(
            Some(track_id.clone()),
            Some("ShortCurve".into()),
            Some(0.0),
            Some(2.0), // 剪辑 [0s, 2s) = 帧 0..400
            Some("C:/audio/a.wav".into()),
        );
        let root_track_id = tl.resolve_root_track_id(&track_id).unwrap();
        tl.ensure_params_for_root(&root_track_id);

        {
            let entry = tl.params_by_root_track.get_mut(&root_track_id).unwrap();
            // tension 是工程长度曲线：整个剪辑范围都有数据。
            for value in entry.tension_edit.iter_mut().take(400) {
                *value = 10.0;
            }
            // volume 模拟 set_param_frames 的增长语义：只写到 0..1s（帧 0..200），
            // 数组长度 = 200。剪辑后半段（200..400）超出数组 = 默认值 1.0，
            // 且 200 就是"最后一个有效参数点"之后的数组末尾。
            entry
                .extra_curves
                .insert("volume".to_string(), vec![0.5; 200]);
        }

        // 向右移动 1s：剪辑 [1s, 3s) = 帧 200..600。
        tl.move_clips(
            &[MoveClipPayload {
                clip_id: clip_id.clone(),
                start_sec: 1.0,
                track_id: None,
            }],
            true,
        );

        let entry = tl.params_by_root_track.get(&root_track_id).unwrap();
        // 有效数据搬运到新范围前半段（帧 200..400）。
        let volume = entry
            .extra_curves
            .get("volume")
            .expect("volume curve must exist");
        assert!(
            (volume[200] - 0.5).abs() < 1e-4 && (volume[399] - 0.5).abs() < 1e-4,
            "drawn volume data must survive the move, got {}..{}",
            volume[200],
            volume[399]
        );
        // 剪辑内有效点之后的部分仍是默认值（帧 400..600）。
        assert!(
            (volume[500] - 1.0).abs() < 1e-4,
            "frames past the last valid point must stay default, got {}",
            volume[500]
        );
        // 旧范围恢复默认。
        assert!(
            (volume[0] - 1.0).abs() < 1e-4 && (volume[100] - 1.0).abs() < 1e-4,
            "vacated range must be restored to default"
        );
        // tension 照常搬运。
        assert!(
            (entry.tension_edit[200] - 10.0).abs() < 1e-4,
            "tension must follow the clip"
        );
    }
}

pub(crate) fn new_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

const TRACK_COLOR_PALETTE: &[&str] = &[
    "#74787e", // 灰（初始 Main 轨道色；也是整条轮转的起点）
    "#4a8fd1", // 蓝
    "#7b6bc4", // 紫
    "#43a875", // 绿
    "#cf6f2e", // 橙
    "#f087b5", // 粉
    "#b845a5", // 洋红
    "#f0d25e", // 黄
    "#d94f4a", // 红
];

fn track_palette_color(index: usize) -> String {
    TRACK_COLOR_PALETTE[index % TRACK_COLOR_PALETTE.len()].to_string()
}

fn default_clip_color() -> String {
    "emerald".to_string()
}

/// Deserialization default for `fade_in_curve`/`fade_out_curve`.
///
/// Must be the empty string: `reconcile_legacy_fade_fields` treats an empty
/// curve string as "not set" (see [`Clip::reconcile_legacy_fade_fields`]).
/// If old projects lacking the field defaulted to a named curve such as
/// "sine", the legacy migration would wrongly overwrite the new shape/dir
/// model (e.g. 1.1 → 5.0 on load). Real legacy data carries non-empty curve
/// strings in the file, and engine/import construction sites
/// (audio_engine, pitch_editing, ...) write literal strings, never going
/// through this default.
fn default_fade_curve() -> String {
    String::new()
}

impl TimelineState {
    pub(crate) fn ensure_project_end_sec(&mut self, end_sec: f64) {
        if !(end_sec.is_finite()) {
            return;
        }
        // Only extend; never shrink automatically.
        // Use ceil so the ruler/grid has room for the full clip.
        let target = end_sec.max(4.0).ceil();
        if target > self.project_sec {
            self.project_sec = target;
        }
    }

    /// 全部 Clip 统一规范化 Take：旧工程（takes 为空）由 active 投影生成
    /// 单 Take；新工程把 active take 物化到内存投影。
    ///
    /// 同时执行旧 Fade 曲线字段的一次性兼容迁移（v3 / 早期开发版 v4 的
    /// 命名曲线字符串 → REAPER 形状/曲率模型）。工程加载、剪贴板片段
    /// 合并等所有反序列化边界都会经过这里，确保迁移不遗漏。
    pub fn normalize_clip_takes(&mut self) {
        for clip in &mut self.clips {
            clip.reconcile_legacy_fade_fields();
            clip.normalize_takes();
        }
    }

    /// 把全部 Clip 的 active 投影写回对应 Take（运行时字段修改后的同步）。
    pub fn sync_clip_takes_from_flat(&mut self) {
        for clip in &mut self.clips {
            clip.sync_take_from_flat();
        }
    }

    pub fn to_payload(&self) -> TimelineStatePayload {
        let tracks_payload = build_track_payload(&self.tracks);
        let clips_payload = self
            .clips
            .iter()
            .map(|c| TimelineClip {
                id: c.id.clone(),
                group_id: c.group_id.clone(),
                track_id: c.track_id.clone(),
                name: c.name.clone(),
                start_sec: c.start_sec,
                length_sec: c.length_sec,
                color: c.color.clone(),
                takes: c
                    .takes
                    .iter()
                    .map(crate::models::TimelineClipTake::from)
                    .collect(),
                active_take_id: c.active_take_id.clone(),
                source_path: c.source_path.clone(),
                source_path_relative: c.source_path_relative.clone(),
                duration_sec: c.duration_sec,
                duration_frames: c.duration_frames,
                source_sample_rate: c.source_sample_rate,
                waveform_preview: c.waveform_preview.clone(),
                pitch_range: c.pitch_range.clone(),
                gain: Some(c.gain),
                muted: Some(c.muted),
                source_start_sec: Some(c.source_start_sec),
                source_end_sec: Some(c.source_end_sec),
                playback_rate: Some(c.playback_rate),
                clip_playback_rate: Some(c.clip_playback_rate),
                reversed: Some(c.reversed),
                loop_enabled: c.loop_enabled,
                snap_offset_sec: Some(c.snap_offset_sec),
                fade_in_sec: Some(c.fade_in_sec),
                fade_out_sec: Some(c.fade_out_sec),
                fade_in_shape: Some(c.fade_in_shape),
                fade_out_shape: Some(c.fade_out_shape),
                fade_in_dir: Some(c.fade_in_dir),
                fade_out_dir: Some(c.fade_out_dir),                auto_fade_in_sec: Some(c.auto_fade_in_sec),
                auto_fade_out_sec: Some(c.auto_fade_out_sec),
                formant_morph: c
                    .formant_morph
                    .as_ref()
                    .map(crate::models::ClipFormantMorphPayload::from),
                midi_note_count: c.midi_note_data.as_ref().map(|n| n.len()),
                midi_note_data: c.midi_note_data.clone(),
                midi_fill_gaps: if c.midi_note_data.is_some() {
                    Some(c.midi_fill_gaps)
                } else {
                    None
                },
            })
            .collect::<Vec<_>>();

        TimelineStatePayload {
            ok: true,
            tracks: tracks_payload,
            clips: clips_payload,
            created_clip_ids: None,
            created_track_ids: None,
            selected_track_id: self.selected_track_id.clone(),
            selected_clip_id: self.selected_clip_id.clone(),
            bpm: self.bpm,
            playhead_sec: self.playhead_sec,
            project_sec: Some(self.project_sec),
            project: None,
            missing_files: None,
            tempo_map: self.tempo_map_payload(),
            disabled_group_ids: {
                let mut ids: Vec<String> = self.disabled_group_ids.iter().cloned().collect();
                ids.sort();
                ids
            },
        }
    }

    /// Lightweight version of `to_payload()` for regular frontend polls.
    ///
    /// Skips expensive per-clip fields (waveform_preview, pitch_range, midi_note_data)
    /// that the frontend caches separately. Reduces clone+serialize cost for
    /// projects with many clips.
    pub fn to_payload_lite(&self) -> TimelineStatePayload {
        let tracks_payload = build_track_payload(&self.tracks);
        let clips_payload = self
            .clips
            .iter()
            .map(|c| TimelineClip {
                id: c.id.clone(),
                group_id: c.group_id.clone(),
                track_id: c.track_id.clone(),
                name: c.name.clone(),
                start_sec: c.start_sec,
                length_sec: c.length_sec,
                color: c.color.clone(),
                // lite 轮询同样剥离 take 级 midi_note_data（顶层已剥离，
                // 每个 take 再带一份完整音符数组会让轮询载荷随 take 数膨胀）。
                takes: c
                    .takes
                    .iter()
                    .map(|t| crate::models::TimelineClipTake::from_take(t, false))
                    .collect(),
                active_take_id: c.active_take_id.clone(),
                source_path: c.source_path.clone(),
                source_path_relative: c.source_path_relative.clone(),
                duration_sec: c.duration_sec,
                duration_frames: c.duration_frames,
                source_sample_rate: c.source_sample_rate,
                waveform_preview: None, // skip — frontend caches separately
                pitch_range: None,      // skip — frontend caches separately
                gain: Some(c.gain),
                muted: Some(c.muted),
                source_start_sec: Some(c.source_start_sec),
                source_end_sec: Some(c.source_end_sec),
                playback_rate: Some(c.playback_rate),
                clip_playback_rate: Some(c.clip_playback_rate),
                reversed: Some(c.reversed),
                loop_enabled: c.loop_enabled,
                snap_offset_sec: Some(c.snap_offset_sec),
                fade_in_sec: Some(c.fade_in_sec),
                fade_out_sec: Some(c.fade_out_sec),
                fade_in_shape: Some(c.fade_in_shape),
                fade_out_shape: Some(c.fade_out_shape),
                fade_in_dir: Some(c.fade_in_dir),
                fade_out_dir: Some(c.fade_out_dir),                auto_fade_in_sec: Some(c.auto_fade_in_sec),
                auto_fade_out_sec: Some(c.auto_fade_out_sec),
                formant_morph: c
                    .formant_morph
                    .as_ref()
                    .map(crate::models::ClipFormantMorphPayload::from),
                midi_note_count: c.midi_note_data.as_ref().map(|n| n.len()),
                midi_note_data: None, // skip — frontend caches separately
                midi_fill_gaps: if c.midi_note_data.is_some() {
                    Some(c.midi_fill_gaps)
                } else {
                    None
                },
            })
            .collect::<Vec<_>>();

        TimelineStatePayload {
            ok: true,
            tracks: tracks_payload,
            clips: clips_payload,
            created_clip_ids: None,
            created_track_ids: None,
            selected_track_id: self.selected_track_id.clone(),
            selected_clip_id: self.selected_clip_id.clone(),
            bpm: self.bpm,
            playhead_sec: self.playhead_sec,
            project_sec: Some(self.project_sec),
            project: None,
            missing_files: None,
            tempo_map: self.tempo_map_payload(),
            disabled_group_ids: {
                let mut ids: Vec<String> = self.disabled_group_ids.iter().cloned().collect();
                ids.sort();
                ids
            },
        }
    }

    // ─── Tempo Map ────────────────────────────────────────────────

    /// 序列化 Tempo Map 供前端载荷使用（None = 无 Tempo Map）。
    fn tempo_map_payload(&self) -> Option<Vec<crate::models::TempoPointPayload>> {
        self.tempo_map.as_ref().map(|points| {
            points
                .iter()
                .map(|p| crate::models::TempoPointPayload {
                    id: p.id.clone(),
                    position_sec: p.position_sec,
                    bpm: p.bpm,
                    numerator: p.numerator,
                    denominator: p.denominator,
                    scale: p.scale.as_ref().map(|s| crate::models::TempoScalePayload {
                        key: s.key.clone(),
                        name: s.name.clone(),
                        notes: s.notes.clone(),
                    }),
                })
                .collect()
        })
    }

    /// 规范化 Tempo Map：排序、钳制、确保首点位于 0；无有效点返回 None。
    /// 0 位置初始点即工程基准记录，必须显式携带拍号（缺失时按 4/4 物化）。
    pub fn normalize_tempo_map(&mut self) {
        let Some(points) = self.tempo_map.take() else {
            return;
        };
        let mut valid: Vec<TempoPointData> = Vec::new();
        for mut p in points {
            if p.id.trim().is_empty() {
                continue;
            }
            p.position_sec = p.position_sec.max(0.0);
            p.bpm = p.bpm.clamp(10.0, 960.0);
            p.numerator = p.numerator.map(|n| n.clamp(1, 32));
            p.denominator = match p.denominator {
                Some(d) if matches!(d, 1 | 2 | 4 | 8 | 16 | 32) => Some(d),
                Some(_) => Some(4),
                None => None,
            };
            valid.push(p);
        }
        // 先排序再去重：若在排序前去重，输入乱序时相邻的重复点（如 [2,5,2]）
        // 会同时保留，破坏“位置严格递增”的不变量（下游二分查找/积分依赖它）。
        valid.sort_by(|a, b| {
            a.position_sec
                .partial_cmp(&b.position_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        valid.dedup_by(|a, b| (a.position_sec - b.position_sec).abs() < 1e-6);
        if valid.is_empty() {
            self.tempo_map = None;
            return;
        }
        if valid[0].position_sec > 1e-9 {
            valid.insert(
                0,
                TempoPointData {
                    id: new_id("tp"),
                    position_sec: 0.0,
                    bpm: self.bpm,
                    numerator: Some(4),
                    denominator: Some(4),
                    // 初始点即工程基准记录：携带工程音阶（内置键反查，否则保留音级集合）。
                    scale: Some(TempoScaleData {
                        key: key_for_scale_notes(&self.project_scale_notes),
                        name: None,
                        notes: Some(self.project_scale_notes.clone()),
                    }),
                },
            );
        }
        valid[0].position_sec = 0.0;
        // 初始点必须显式携带拍号（工程基准记录不存在“之前”可跟随）。
        if valid[0].numerator.is_none() || valid[0].denominator.is_none() {
            valid[0].numerator = Some(valid[0].numerator.unwrap_or(4));
            valid[0].denominator = Some(valid[0].denominator.unwrap_or(4));
        }
        self.tempo_map = Some(valid);
    }

    /// 下标处变化点的生效拍号（跟随之前的拍号时向前解析为实际值）。
    /// 0 位置初始点由规范化保证显式携带拍号，因此任何下标都有确定值。
    pub fn effective_time_signature_at(points: &[TempoPointData], index: usize) -> (u32, u32) {
        let mut carry: (u32, u32) = (4, 4);
        for (i, point) in points.iter().enumerate() {
            if let (Some(n), Some(d)) = (point.numerator, point.denominator) {
                carry = (n.clamp(1, 32), d);
            }
            if i >= index {
                break;
            }
        }
        carry
    }

    /// 逐段生效音阶（(段起始秒, 音级集合)，按时间升序；首段从 0 开始）。
    /// 供逐帧渲染路径使用（帧时间单调递增，可用游标快速查询）。
    pub fn scale_segments(&self) -> Vec<(f64, Vec<u8>)> {
        let Some(points) = self.tempo_map.as_ref() else {
            return vec![(0.0, self.project_scale_notes.clone())];
        };
        let mut segments: Vec<(f64, Vec<u8>)> = Vec::new();
        let mut current: Option<Vec<u8>> = None;
        let mut last_sec = 0.0f64;
        for point in points {
            let sec = point.position_sec.max(last_sec);
            if sec > last_sec + 1e-9 {
                let notes = current
                    .clone()
                    .unwrap_or_else(|| self.project_scale_notes.clone());
                segments.push((last_sec, notes));
            }
            last_sec = sec;
            if let Some(scale) = point.scale.as_ref() {
                if let Some(key) = scale.key.as_deref() {
                    if let Some(notes) = scale_notes_for_key(key) {
                        current = Some(notes);
                    }
                }
                if let Some(notes) = scale.notes.as_ref() {
                    let mut normalized: Vec<u8> = notes.iter().map(|v| v % 12).collect();
                    normalized.sort_unstable();
                    normalized.dedup();
                    if !normalized.is_empty() {
                        current = Some(normalized);
                    }
                }
            }
        }
        let notes = current.unwrap_or_else(|| self.project_scale_notes.clone());
        if segments.is_empty() || (segments.last().map(|(sec, _)| *sec) < Some(last_sec)) {
            segments.push((last_sec, notes));
        }
        segments
    }

    /// 渲染相关的“生效音阶”签名。
    ///
    /// 该签名基于 `scale_segments()` 的实际生效音阶，并压缩相邻相同音阶段。
    /// 与只统计“显式音阶变化点”不同，因此：
    /// - 创建 / 清除只含“初始点 = 工程基准”的 Tempo Map 不会产生签名变化；
    /// - 移动只有 BPM / 拍号、没有音阶变化的变化点不会产生签名变化；
    /// - 音阶键、音级集合、音阶生效位置或工程基准音阶变化时签名会变化。
    pub fn render_scale_signature(&self) -> String {
        let mut segments: Vec<String> = Vec::new();
        let mut last_notes: Option<Vec<u8>> = None;
        for (sec, notes) in self.scale_segments() {
            if last_notes.as_ref() == Some(&notes) {
                continue;
            }
            let notes_text = notes
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            segments.push(format!("{:.6}:{}", sec, notes_text));
            last_notes = Some(notes);
        }
        segments.join("|")
    }

    pub fn add_track(
        &mut self,
        name: Option<String>,
        parent_track_id: Option<String>,
        index: Option<usize>,
    ) -> String {
        let id = new_id("track");
        let order = self.next_track_order;
        self.next_track_order += 1;

        // 调色板第一个颜色就是灰色（初始 Main 轨道色）：新建工程 Main = 灰
        // （palette[0]），此后添加的轨道按 蓝 → 紫 → … → 红 → 灰 → … 循环。
        let color = track_palette_color(self.tracks.len());

        let track = Track {
            id: id.clone(),
            name: name.unwrap_or_else(|| "Track".to_string()),
            parent_id: parent_track_id,
            order,
            muted: false,
            solo: false,
            volume: 1.0,

            compose_enabled: false,
            pitch_analysis_algo: PitchAnalysisAlgo::default(),
            color,
        };
        self.tracks.push(track);

        // Best-effort insert ordering: we encode ordering using `order`, but for now
        // we accept `index` by nudging orders for the same parent.
        if let Some(i) = index {
            self.reorder_siblings(&id, i);
        }

        self.selected_track_id = Some(id.clone());
        id
    }

    /// 克隆轨道：
    /// - 普通子轨道：创建新子轨道（同 parent），克隆所有 clip
    /// - 根轨道：创建整个轨道组（根 + 后代），克隆所有 clip + params_by_root_track
    pub fn duplicate_track(&mut self, track_id: &str) -> Vec<String> {
        use std::collections::HashMap;

        let source = match self.tracks.iter().find(|t| t.id == track_id) {
            Some(t) => t.clone(),
            None => return vec![],
        };

        let is_root = source.parent_id.is_none();

        // 显示顺序由树形 DFS 决定（每层按 order 排序），因此“紧贴源轨道
        // 之后”= 同级（同 parent）中把源轨道之后的 order 整体后移一位，
        // 克隆占据 source.order + 1。不跨父级重编号，避免影响其他分组。

        if is_root {
            // ── 根轨道：收集整棵子树 ──
            let mut all_ids = vec![track_id.to_string()];
            let mut idx = 0;
            while idx < all_ids.len() {
                let cur = all_ids[idx].clone();
                for child in self
                    .tracks
                    .iter()
                    .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
                    .map(|t| t.id.clone())
                    .collect::<Vec<_>>()
                {
                    all_ids.push(child);
                }
                idx += 1;
                if idx > 4096 {
                    break;
                }
            }

            // 根层级：位于源根之后的根轨道整体后移一位，为克隆子树腾位。
            let src_root_order = source.order;
            for t in self.tracks.iter_mut() {
                if t.parent_id.is_none() && t.id != track_id && t.order > src_root_order {
                    t.order += 1;
                }
            }

            // old_id → new_id 映射
            let id_map: HashMap<String, String> = all_ids
                .iter()
                .map(|old| (old.clone(), new_id("track")))
                .collect();

            let mut new_track_ids = Vec::new();

            // 克隆轨道。子树内部保持原有相对顺序：
            // 克隆根 = 源根 order + 1，后代 = 源对应轨道 order + 1。
            for old_id in &all_ids {
                let src_track = match self.tracks.iter().find(|t| &t.id == old_id) {
                    Some(t) => t,
                    None => continue,
                };
                let new_tid = id_map[old_id].clone();
                let new_parent = src_track
                    .parent_id
                    .as_ref()
                    .and_then(|pid| id_map.get(pid))
                    .cloned();

                let order = src_track.order + 1;

                let mut cloned = src_track.clone();
                cloned.id = new_tid.clone();
                cloned.parent_id = new_parent;
                cloned.order = order;
                // 根轨道名称加 " (Copy)" 后缀
                if old_id == track_id {
                    cloned.name = format!("{} (Copy)", cloned.name);
                }
                self.tracks.push(cloned);
                new_track_ids.push(new_tid);
            }

            // 克隆所有 clip
            let clips_to_clone: Vec<Clip> = self
                .clips
                .iter()
                .filter(|c| all_ids.contains(&c.track_id))
                .cloned()
                .collect();
            for clip in clips_to_clone {
                let new_cid = new_id("clip");
                let new_tid = id_map[&clip.track_id].clone();
                let mut cloned = clip;
                cloned.id = new_cid;
                cloned.track_id = new_tid;
                self.clips.push(cloned);
            }

            // 克隆 params_by_root_track
            let new_root_id = id_map[track_id].clone();
            if let Some(params) = self.params_by_root_track.get(track_id).cloned() {
                self.params_by_root_track
                    .insert(new_root_id.clone(), params);
            }

            self.selected_track_id = Some(new_root_id);
            new_track_ids
        } else {
            // ── 普通子轨道：只克隆单个轨道 + 其 clip ──
            // 同级中位于源轨道之后的全部后移一位，克隆紧贴源轨道之后。
            let src_order = source.order;
            for t in self.tracks.iter_mut() {
                if t.parent_id == source.parent_id && t.id != track_id && t.order > src_order {
                    t.order += 1;
                }
            }

            let new_tid = new_id("track");

            let mut cloned = source.clone();
            cloned.id = new_tid.clone();
            cloned.name = format!("{} (Copy)", cloned.name);
            cloned.order = src_order + 1;
            self.tracks.push(cloned);

            // 克隆 clip
            let clips_to_clone: Vec<Clip> = self
                .clips
                .iter()
                .filter(|c| c.track_id == track_id)
                .cloned()
                .collect();
            for clip in clips_to_clone {
                let new_cid = new_id("clip");
                let mut cloned = clip;
                cloned.id = new_cid;
                cloned.track_id = new_tid.clone();
                self.clips.push(cloned);
            }

            self.selected_track_id = Some(new_tid.clone());
            vec![new_tid]
        }
    }

    /// 克隆轨道并把克隆子树放置到指定位置（目标父级 + 同级 index）。
    ///
    /// 用于“复制拖动”修饰键下的轨道头拖拽：克隆内容与 `duplicate_track`
    /// 一致（根轨道含整棵子树），随后把克隆子树移动到用户拖放的位置。
    /// 目标位置以“不含被拖拽源轨道”的当前树为基准计算（前端
    /// `computeDropSpec` 的语义），因此不存在自嵌套环。
    pub fn duplicate_track_to(
        &mut self,
        track_id: &str,
        parent_track_id: Option<String>,
        target_index: usize,
    ) -> Vec<String> {
        let new_ids = self.duplicate_track(track_id);
        if let Some(new_root) = new_ids.first().cloned() {
            self.move_track(&new_root, target_index, parent_track_id);
        }
        new_ids
    }

    fn reorder_siblings(&mut self, track_id: &str, target_index: usize) {
        let parent_id = self
            .tracks
            .iter()
            .find(|t| t.id == track_id)
            .and_then(|t| t.parent_id.clone());
        let mut siblings: Vec<_> = self
            .tracks
            .iter()
            .filter(|t| t.parent_id == parent_id && t.id != track_id)
            .cloned()
            .collect();
        siblings.sort_by_key(|t| t.order);
        let target_index = target_index.min(siblings.len());

        // Pull this track out and rebuild orders.
        let mut rebuilt: Vec<String> = siblings.into_iter().map(|t| t.id).collect();
        rebuilt.insert(target_index, track_id.to_string());

        for (i, tid) in rebuilt.iter().enumerate() {
            if let Some(t) = self.tracks.iter_mut().find(|t| &t.id == tid) {
                t.order = i as i32;
            }
        }
        self.next_track_order = rebuilt.len() as i32 + 1;
    }

    pub fn remove_track(&mut self, track_id: &str) {
        // 守卫：如果目标是根轨道且只剩最后一个根轨道，禁止删除。
        let target = self.tracks.iter().find(|t| t.id == track_id);
        let is_root = target.map_or(false, |t| t.parent_id.is_none());
        if is_root {
            let root_count = self.tracks.iter().filter(|t| t.parent_id.is_none()).count();
            if root_count <= 1 {
                return;
            }
        }

        // BFS 收集要删除的轨道及其所有后代。
        let mut to_remove = vec![track_id.to_string()];
        let mut idx = 0;
        while idx < to_remove.len() {
            let cur = to_remove[idx].clone();
            for child in self
                .tracks
                .iter()
                .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
                .map(|t| t.id.clone())
                .collect::<Vec<_>>()
            {
                to_remove.push(child);
            }
            idx += 1;
        }

        // Remove clips belonging to the removed tracks.
        let remove_set: std::collections::HashSet<&str> =
            to_remove.iter().map(|s| s.as_str()).collect();
        self.clips
            .retain(|c| !remove_set.contains(c.track_id.as_str()));

        self.tracks.retain(|t| !remove_set.contains(t.id.as_str()));

        if self.selected_track_id.as_deref() == Some(track_id) {
            self.selected_track_id = self.tracks.first().map(|t| t.id.clone());
        }
        if let Some(cid) = self.selected_clip_id.clone() {
            if !self.clips.iter().any(|c| c.id == cid) {
                self.selected_clip_id = None;
            }
        }
    }

    pub fn move_track(
        &mut self,
        track_id: &str,
        target_index: usize,
        parent_track_id: Option<String>,
    ) {
        // 环与存在性校验：parent 必须存在，且不能是 track_id 自身或其后代。
        // 否则后续按 parent 链的遍历（如导出排序的递归 DFS）会无限递归
        // 直至栈溢出，且坏数据会随撤销历史/保存文件持久化。
        if let Some(ref pid) = parent_track_id {
            if pid == track_id {
                return;
            }
            if !self.tracks.iter().any(|t| t.id == *pid) {
                return;
            }
            let mut cursor = pid.clone();
            let mut safety = 0;
            loop {
                let next = self
                    .tracks
                    .iter()
                    .find(|t| t.id == cursor)
                    .and_then(|t| t.parent_id.clone());
                match next {
                    Some(p) => {
                        if p == track_id {
                            // pid 位于 track_id 的子树内 → 会成环，拒绝移动。
                            return;
                        }
                        cursor = p;
                    }
                    None => break,
                }
                safety += 1;
                if safety > self.tracks.len() {
                    // 既有数据已存在环，同样拒绝移动。
                    return;
                }
            }
        }
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            t.parent_id = parent_track_id;
        }
        self.reorder_siblings(track_id, target_index);
    }

    pub fn set_track_state(
        &mut self,
        track_id: &str,
        muted: Option<bool>,
        solo: Option<bool>,
        volume: Option<f32>,
        compose_enabled: Option<bool>,
        pitch_analysis_algo: Option<PitchAnalysisAlgo>,
        color: Option<String>,
        name: Option<String>,
    ) {
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            if let Some(v) = muted {
                t.muted = v;
            }
            if let Some(v) = solo {
                t.solo = v;
            }
            if let Some(v) = volume {
                t.volume = v.clamp(0.0, 4.0);
            }

            if let Some(v) = compose_enabled {
                t.compose_enabled = v;
            }
            if let Some(v) = pitch_analysis_algo {
                t.pitch_analysis_algo = v;
            }
            if let Some(v) = color {
                t.color = v;
            }
            if let Some(v) = name {
                let trimmed = v.trim().to_string();
                if !trimmed.is_empty() {
                    t.name = trimmed;
                }
            }
        }
    }

    pub fn select_track(&mut self, track_id: &str) {
        if self.tracks.iter().any(|t| t.id == track_id) {
            self.selected_track_id = Some(track_id.to_string());
        }
    }

    pub fn set_project_length(&mut self, project_sec: f64) {
        if project_sec.is_finite() {
            self.project_sec = project_sec.max(4.0);
        }
    }

    /// 从 `duration_frames` / `source_sample_rate` 重建被省略的 `duration_sec`。
    ///
    /// 工程文件保存时会省略可精确推导的 `duration_sec`，这里在加载/导入时
    /// 恢复它，保证所有读取路径看到的仍是完整 Clip。
    pub fn restore_derived_clip_fields(&mut self) {
        for clip in &mut self.clips {
            if clip.duration_sec.is_some() {
                continue;
            }
            if let (Some(frames), Some(sample_rate)) =
                (clip.duration_frames, clip.source_sample_rate)
            {
                if sample_rate > 0 {
                    clip.duration_sec = Some(frames as f64 / sample_rate as f64);
                }
            }
        }
    }

    /// 根据 clip 的 source_path 从磁盘读取文件元数据 + 内容指纹，
    /// 填充 `source_file_size`、`source_file_mtime`、`source_file_fingerprint`。
    ///
    /// 内容指纹优先保留工程文件中已持久化的值：它代表“用于匹配的原始文件
    /// 哈希”。仅当工程中没有保存指纹时，才用当前磁盘文件计算一次。
    pub fn populate_clip_file_metadata(clip: &mut Clip) {
        for take in &mut clip.takes {
            let Some(ref source_path) = take.source_path else {
                continue;
            };
            let p = std::path::Path::new(source_path);
            if !p.exists() {
                continue;
            }
            if let Ok(meta) = std::fs::metadata(p) {
                take.source_file_size = Some(meta.len());
                take.source_file_mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());
            }
            if take.source_file_fingerprint.is_none() {
                if let Some(fp) = crate::audio_utils::compute_file_fingerprint(p) {
                    take.source_file_fingerprint = Some(fp);
                }
            }
        }
        let take: ClipTake = clip.active_take().clone();
        take.apply_to_clip(clip);
    }

    pub fn add_clip(
        &mut self,
        track_id: Option<String>,
        name: Option<String>,
        start_sec: Option<f64>,
        length_sec: Option<f64>,
        source_path: Option<String>,
    ) -> String {
        let track_id = track_id
            .or_else(|| self.selected_track_id.clone())
            .or_else(|| self.tracks.first().map(|t| t.id.clone()))
            .unwrap_or_else(|| self.add_track(Some("Main".to_string()), None, None));

        if !self.tracks.iter().any(|t| t.id == track_id) {
            // Create missing track.
            self.tracks.push(Track {
                id: track_id.clone(),
                name: "Track".to_string(),
                parent_id: None,
                order: self.next_track_order,
                muted: false,
                solo: false,
                volume: 1.0,

                compose_enabled: false,
                pitch_analysis_algo: PitchAnalysisAlgo::default(),
                color: track_palette_color(self.tracks.len()),
            });
            self.next_track_order += 1;
        }

        // If this is a new clip referencing an existing audio source, inherit cached metadata
        // (duration + waveform preview) from any existing clip that already has it.
        let inherited = source_path.as_deref().and_then(|sp| {
            self.clips
                .iter()
                .find(|c| c.source_path.as_deref() == Some(sp) && c.waveform_preview.is_some())
                .map(|c| {
                    (
                        c.duration_sec,
                        c.duration_frames,
                        c.source_sample_rate,
                        c.waveform_preview.clone(),
                        c.pitch_range.clone(),
                    )
                })
        });

        let id = new_id("clip");
        let ss = start_sec.unwrap_or(self.playhead_sec).max(0.0);
        let ls = length_sec.unwrap_or(4.0).max(0.01);
        self.ensure_project_end_sec(ss + ls);

        // If no inherited metadata (duration / waveform) is available for this
        // source_path, try to read basic audio info and a preview from the file
        // so newly created clips (e.g. pasted ones) display waveforms.
        let mut computed_duration_sec = inherited.as_ref().and_then(|v| v.0);
        let mut computed_duration_frames = inherited.as_ref().and_then(|v| v.1);
        let mut computed_source_sr = inherited.as_ref().and_then(|v| v.2);
        let mut computed_waveform = inherited.as_ref().and_then(|v| v.3.clone());
        let mut computed_mtime: Option<u64> = None;
        let mut computed_size: Option<u64> = None;
        let mut computed_fp: Option<u64> = None;

        if computed_waveform.is_none() {
            if let Some(sp) = source_path.as_deref() {
                let p = std::path::Path::new(sp);
                if p.exists() {
                    // 记录文件元数据 + 内容指纹，用于检测外部修改
                    if let Ok(meta) = std::fs::metadata(p) {
                        computed_size = Some(meta.len());
                        computed_mtime = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs());
                    }
                    computed_fp = crate::audio_utils::compute_file_fingerprint(p);
                    // 视频文件只做 O(1) header 探测：try_read_wav_info 会为生成
                    // preview 全量解码整条音轨，大视频导入时会造成长时间阻塞，
                    // 且该结果随后会被 import_audio_item 的元数据覆盖（纯浪费）。
                    // 波形由前端按需异步请求峰值缓存生成。
                    let info = if crate::media::is_video_extension(p) {
                        crate::audio_utils::try_read_audio_header_only(p)
                    } else {
                        crate::audio_utils::try_read_wav_info(p, 4096)
                    };
                    if let Some(info) = info {
                        computed_duration_sec = Some(info.duration_sec);
                        computed_duration_frames = Some(info.total_frames);
                        computed_source_sr = Some(info.sample_rate);
                        computed_waveform = Some(info.waveform_preview);
                    }
                }
            }
        }

        let clip = Clip {
            takes: vec![],
            active_take_id: None,
            id: id.clone(),
            group_id: None,
            track_id: track_id.clone(),
            name: name.unwrap_or_else(|| "Clip".to_string()),
            start_sec: ss,
            length_sec: ls,
            color: default_clip_color(),
            source_path,
            source_path_relative: None,
            duration_sec: computed_duration_sec,
            duration_frames: computed_duration_frames,
            source_sample_rate: computed_source_sr,
            source_file_mtime: computed_mtime,
            source_file_size: computed_size,
            source_file_fingerprint: computed_fp,
            waveform_preview: computed_waveform,
            pitch_range: inherited
                .as_ref()
                .and_then(|v| v.4.clone())
                .or(Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                })),
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: computed_duration_sec.unwrap_or(ls),
            playback_rate: 1.0,
            clip_playback_rate: 1.0,
            reversed: false,
            // 新 Clip 的 Loop 属性跟随"为新的音频块启用循环"设置
            //（导入/录音/MIDI-as-clip/add_clip 等所有创建路径统一生效）。
            loop_enabled: crate::config::loop_new_clips_default(),
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_shape: 1.0,
            fade_out_shape: 1.0,
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
        };
        self.clips.push(clip);
        // 确保文件元数据始终被填充（包括继承 waveform 但未计算 metadata 的情况）
        if let Some(last) = self.clips.last_mut() {
            last.sync_take_from_flat();
            Self::populate_clip_file_metadata(last);
        }
        self.selected_clip_id = Some(id.clone());
        self.playhead_sec = ss;
        id
    }

    /// 波纹编辑（自动跟进）：把“编辑点（origin）之后、且不属于被编辑集合的剪辑”
    /// 整体平移 `delta_sec`（秒，可正可负）。
    ///
    /// - `edited_ids`：本次编辑直接作用到的剪辑（被移动/删除/重设尺寸），排除在平移之外；
    /// - `affected_tracks`：`Some(轨道集合)` 时只平移这些轨道上的后续剪辑（Track 模式）；
    ///   `None` 表示所有轨道（All 模式）。`Off` 模式由调用方直接跳过，无需传入；
    /// - `move_linked_params`：是否把后续剪辑携带的轨道组参数线一起平移
    ///   （与普通拖拽移动的 `move_linked_params` / “锁定参数线” 语义一致）。
    ///
    /// 返回实际被平移的剪辑 id 列表（供调用方调度音高重分析）。
    #[allow(clippy::too_many_arguments)]
    pub fn ripple_shift_clips(
        &mut self,
        edited_ids: &[&str],
        affected_tracks: Option<&HashSet<String>>,
        origin: f64,
        delta_sec: f64,
        move_linked_params: bool,
    ) -> Vec<String> {
        if !delta_sec.is_finite() || delta_sec.abs() < 1e-9 {
            return Vec::new();
        }
        let edited: HashSet<&str> = edited_ids.iter().copied().collect();
        let origin_ok = origin.is_finite();

        // 收集需要平移的剪辑 id 与目标位置（所有被波及剪辑共用同一平移量）。
        let mut moves: Vec<MoveClipPayload> = Vec::new();
        let mut shifted_ids: Vec<String> = Vec::new();
        for clip in &self.clips {
            if edited.contains(clip.id.as_str()) {
                continue;
            }
            if let Some(ref tracks) = affected_tracks {
                if !tracks.contains(&clip.track_id) {
                    continue;
                }
            }
            if !origin_ok || clip.start_sec + 1e-9 < origin {
                continue;
            }
            let next_start = (clip.start_sec + delta_sec).max(0.0);
            moves.push(MoveClipPayload {
                clip_id: clip.id.clone(),
                start_sec: next_start,
                track_id: None,
            });
            shifted_ids.push(clip.id.clone());
        }
        if moves.is_empty() {
            return Vec::new();
        }
        self.move_clips(&moves, move_linked_params);
        shifted_ids
    }

    /// 批量删除多个 clip，只触发一次状态变更
    pub fn remove_clips(&mut self, clip_ids: &[String]) {
        let id_set: HashSet<&str> = clip_ids.iter().map(|s| s.as_str()).collect();
        self.clips.retain(|c| !id_set.contains(c.id.as_str()));
        if let Some(ref sel) = self.selected_clip_id {
            if id_set.contains(sel.as_str()) {
                self.selected_clip_id = None;
            }
        }
    }

    pub fn move_clips(&mut self, moves: &[MoveClipPayload], move_linked_params: bool) {
        #[derive(Debug)]
        struct LinkedMovePlan {
            old_root_track_id: String,
            old_start_sec: f64,
            clip_length_sec: f64,
            source_extra_keys: Vec<String>,
            new_root_track_id: String,
            new_start_sec: f64,
            linked_params: LinkedParamCurvesPayload,
        }

        #[derive(Debug)]
        struct MovePlan {
            clip_id: String,
            new_track_id: String,
            new_start_sec: f64,
            new_end_sec: f64,
            linked_move: Option<LinkedMovePlan>,
        }

        let mut seen_clip_ids = HashSet::new();
        let mut plans = Vec::new();

        for requested_move in moves {
            if !seen_clip_ids.insert(requested_move.clip_id.clone()) {
                continue;
            }

            let Some((old_track_id, old_start_sec, clip_length_sec)) = self
                .clips
                .iter()
                .find(|clip| clip.id == requested_move.clip_id)
                .map(|clip| {
                    (
                        clip.track_id.clone(),
                        clip.start_sec,
                        clip.length_sec.max(0.0),
                    )
                })
            else {
                continue;
            };

            let new_start_sec = requested_move.start_sec.max(0.0);
            let new_track_id = requested_move
                .track_id
                .clone()
                .filter(|track_id| self.tracks.iter().any(|track| track.id == *track_id))
                .unwrap_or_else(|| old_track_id.clone());

            let linked_move = if move_linked_params && clip_length_sec > 0.0 {
                let old_root_track_id = self.resolve_root_track_id(&old_track_id);
                let new_root_track_id = self.resolve_root_track_id(&new_track_id);
                match (old_root_track_id, new_root_track_id) {
                    (Some(old_root_track_id), Some(new_root_track_id))
                        if old_root_track_id != new_root_track_id
                            || (new_start_sec - old_start_sec).abs() > f64::EPSILON =>
                    {
                        let source_extra_keys = self
                            .params_by_root_track
                            .get(&old_root_track_id)
                            .map(|entry| entry.extra_curves.keys().cloned().collect::<Vec<_>>())
                            .unwrap_or_default();
                        self.extract_linked_params_from_root_range(
                            &old_root_track_id,
                            old_start_sec,
                            clip_length_sec,
                        )
                        .map(|linked_params| LinkedMovePlan {
                            old_root_track_id,
                            old_start_sec,
                            clip_length_sec,
                            source_extra_keys,
                            new_root_track_id,
                            new_start_sec,
                            linked_params,
                        })
                    }
                    _ => None,
                }
            } else {
                None
            };

            plans.push(MovePlan {
                clip_id: requested_move.clip_id.clone(),
                new_track_id,
                new_start_sec,
                new_end_sec: new_start_sec + clip_length_sec,
                linked_move,
            });
        }

        for plan in &plans {
            if let Some(clip) = self.clips.iter_mut().find(|clip| clip.id == plan.clip_id) {
                clip.start_sec = plan.new_start_sec;
                clip.track_id = plan.new_track_id.clone();
            }
            self.ensure_project_end_sec(plan.new_end_sec);
        }

        let linked_moves: Vec<&LinkedMovePlan> = plans
            .iter()
            .filter_map(|plan| plan.linked_move.as_ref())
            .collect();
        for linked_move in &linked_moves {
            self.clear_linked_params_in_root_range(
                &linked_move.old_root_track_id,
                linked_move.old_start_sec,
                linked_move.clip_length_sec,
                !linked_move.linked_params.pitch_edit.is_empty(),
                Some(&linked_move.source_extra_keys),
            );
        }
        for linked_move in linked_moves {
            self.apply_linked_params_to_root_range(
                &linked_move.new_root_track_id,
                linked_move.new_start_sec,
                &linked_move.linked_params,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn set_clip_state(
        &mut self,
        clip_id: &str,
        length_sec: Option<f64>,
        gain: Option<f32>,
        muted: Option<bool>,
        source_start_sec: Option<f64>,
        source_end_sec: Option<f64>,
        playback_rate: Option<f32>,
        reversed: Option<bool>,
        fade_in_sec: Option<f64>,
        fade_out_sec: Option<f64>,
    ) {
        self.patch_clip_state(
            clip_id,
            ClipStatePatch {
                name: None,
                start_sec: None,
                length_sec,
                gain,
                muted,
                source_start_sec,
                source_end_sec,
                playback_rate,
                clip_playback_rate: None,
                reversed,
                loop_enabled: None,
                snap_offset_sec: None,
                fade_in_sec,
                fade_out_sec,
                fade_in_shape: None,
                fade_out_shape: None,
                fade_in_dir: None,
                fade_out_dir: None,
                auto_fade_in_sec: None,
                auto_fade_out_sec: None,
                color: None,
                formant_morph: None,
            },
        );
    }

    pub fn patch_clip_state(&mut self, clip_id: &str, patch: ClipStatePatch) {
        let content_sync_patch = patch.clone();
        let mut end_sec: Option<f64> = None;
        if let Some(c) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            if let Some(v) = patch.name {
                c.name = v;
            }
            if let Some(v) = patch.start_sec {
                c.start_sec = v.max(0.0);
            }
            if let Some(v) = patch.length_sec {
                c.length_sec = v.max(0.0);
            }
            if let Some(v) = patch.gain {
                c.gain = v.clamp(0.0, 4.0);
            }
            if let Some(v) = patch.muted {
                c.muted = v;
            }
            if let Some(v) = patch.source_start_sec {
                if v.is_finite() {
                    // Negative values are allowed (slip-edit past the source start -> leading silence).
                    // Keep a reasonable bound to avoid accidental extreme values.
                    c.source_start_sec = v.clamp(-1_000_000.0, 1_000_000.0);
                }
            }
            if let Some(v) = patch.source_end_sec {
                // 与前端契约一致：source_end < 0 是合法状态（倒放 Clip 的消费窗口
                // 锚定 se；整窗在媒体下方的静音段）。旧实现 v.max(0.0) 会把
                // 前导静音/负窗口钳回 0，造成"拖拽时正常、后端快照回灌后跳变"。
                if v.is_finite() {
                    c.source_end_sec = v.clamp(-1_000_000.0, 1_000_000.0);
                }
            }
            if let Some(v) = patch.playback_rate {
                c.playback_rate = v.clamp(0.1, 10.0);
            }
            if let Some(v) = patch.clip_playback_rate {
                let next_clip_rate = v.clamp(0.1, 10.0);
                // Clip 级倍率变化只改乘数：以 active Take 的自身速率重算
                // 有效速率（不污染 Take 速率本身）。takes 为空（旧扁平数据）
                // 时按“有效速率 ÷ 旧倍率”还原 Take 速率。
                let old_clip_rate =
                    if c.clip_playback_rate.is_finite() && c.clip_playback_rate > 1e-6 {
                        c.clip_playback_rate
                    } else {
                        1.0
                    };
                let take_rate = if c.takes.is_empty() {
                    c.playback_rate / old_clip_rate
                } else {
                    c.active_take().playback_rate
                };
                let take_rate = if take_rate.is_finite() && take_rate > 1e-6 {
                    take_rate
                } else {
                    1.0
                };
                c.clip_playback_rate = next_clip_rate;
                c.playback_rate = next_clip_rate * take_rate;
                // 同一请求同时携带目标有效速率时以其为准 —— Take 自身速率
                // 在下方“同步编辑所有 Take”阶段按新倍率统一反推，保证
                // active / inactive take 口径一致。
                if let Some(v) = patch.playback_rate {
                    c.playback_rate = v.clamp(0.1, 10.0);
                }
            }
            if let Some(v) = patch.reversed {
                // 方向翻转（且本请求未显式指定源窗口，如右键“倒放”）时，
                // 按翻转前的消费窗口换算新方向的锚点字段 —— 派生窗口模型下
                // 非锚定方向的存储字段可能陈旧，直接翻转布尔会让消费内容
                // 跳变（裁剪过的 Clip 倒放后播到陈旧 se 所指的文件末段）。
                // 显式携带 source_start/end 的请求（粘贴/导入等）以调用方
                // 给出的窗口为准，不做换算。
                if c.reversed != v
                    && patch.source_start_sec.is_none()
                    && patch.source_end_sec.is_none()
                {
                    let rate = if c.playback_rate.is_finite() && c.playback_rate > 1e-6 {
                        c.playback_rate as f64
                    } else {
                        1.0
                    };
                    let span = c.length_sec.max(0.0) * rate;
                    let media_total = clip_source_media_duration_sec(c);
                    flip_direction_source_window(
                        v,
                        c.loop_enabled,
                        span,
                        media_total,
                        &mut c.source_start_sec,
                        &mut c.source_end_sec,
                    );
                }
                c.reversed = v;
            }
            if let Some(v) = patch.loop_enabled {
                c.loop_enabled = v;
            }
            // SnapOffset 必须落在 [0, length] 内：负值无意义，超出 Clip
            // 长度的偏移点不可见也不可吸附。trim/拉伸改写长度而未携带
            // snap_offset 时，同步下钳已有偏移 —— 否则残留 offset > length
            // 的"幻影吸附目标"（屏幕上不可见，但其它 Clip 仍会吸附到该位置）。
            match patch.snap_offset_sec {
                Some(v) if v.is_finite() => {
                    c.snap_offset_sec = v.clamp(0.0, c.length_sec.max(0.0));
                }
                _ if patch.length_sec.is_some() => {
                    c.snap_offset_sec = c.snap_offset_sec.clamp(0.0, c.length_sec.max(0.0));
                }
                _ => {}
            }
            if let Some(v) = patch.fade_in_sec {
                c.fade_in_sec = v.max(0.0);
            }
            if let Some(v) = patch.fade_out_sec {
                c.fade_out_sec = v.max(0.0);
            }
            if let Some(v) = patch.fade_in_shape {
                c.fade_in_shape = v;
            }
            if let Some(v) = patch.fade_out_shape {
                c.fade_out_shape = v;
            }
            if let Some(v) = patch.fade_in_dir {
                c.fade_in_dir = clamp_fade_dir(v);
            }
            if let Some(v) = patch.fade_out_dir {
                c.fade_out_dir = clamp_fade_dir(v);
            }
            if let Some(v) = patch.auto_fade_in_sec {
                c.auto_fade_in_sec = v.max(0.0);
            }
            if let Some(v) = patch.auto_fade_out_sec {
                c.auto_fade_out_sec = v.max(0.0);
            }
            if let Some(v) = patch.color {
                c.color = v;
            }
            if let Some(v) = patch.formant_morph {
                c.formant_morph = Some(v);
            }
            // “同步编辑所有 Take”：内容级编辑（源偏移/速率/倒放/Loop/增益）
            // 同步到该 Clip 的全部 Take；容器级属性（位置/长度/fade/颜色等）保持
            // Clip 级语义，不参与同步。
            if crate::config::sync_edits_across_takes() {
                // playback_rate 请求的是“组合有效速率”（clip 倍率 × take 速率），
                // 写入各 Take 自身速率前必须按当前倍率反推 —— 否则 inactive take
                // 在切换后有效速率会被放大 clip_rate 倍（与 from_clip 对 active
                // take 的口径不一致）。
                let clip_rate_for_sync =
                    if c.clip_playback_rate.is_finite() && c.clip_playback_rate > 1e-6 {
                        c.clip_playback_rate
                    } else {
                        1.0
                    };
                let clip_len_for_sync = c.length_sec;
                for take in &mut c.takes {
                    if let Some(v) = content_sync_patch.gain {
                        take.gain = v.clamp(0.0, 4.0);
                    }
                    if let Some(v) = content_sync_patch.source_start_sec {
                        if v.is_finite() {
                            take.source_start_sec = v.clamp(-1_000_000.0, 1_000_000.0);
                        }
                    }
                    if let Some(v) = content_sync_patch.source_end_sec {
                        // 与 flat 投影同口径：se<0 合法（倒放负窗口/前导静音），
                        // 不得钳回 0（详情见 patch_clip_state 中 source_end_sec）。
                        if v.is_finite() {
                            take.source_end_sec = v.clamp(-1_000_000.0, 1_000_000.0);
                        }
                    }
                    if let Some(v) = content_sync_patch.playback_rate {
                        take.playback_rate =
                            (v.clamp(0.1, 10.0) / clip_rate_for_sync).clamp(0.1, 10.0);
                    }
                    if let Some(v) = content_sync_patch.reversed {
                        // 与 flat 投影同口径：方向翻转时按各 Take **自身**的
                        // 组合消费速率换算源窗口/锚点，保持每个 Take 的消费
                        // 内容不变（不同 Take 速率/窗口各不相同，不能只翻布尔）。
                        if take.reversed != v
                            && content_sync_patch.source_start_sec.is_none()
                            && content_sync_patch.source_end_sec.is_none()
                        {
                            flip_take_playback_direction(
                                take,
                                clip_len_for_sync,
                                clip_rate_for_sync as f64,
                            );
                        }
                        take.reversed = v;
                    }
                    if let Some(v) = content_sync_patch.loop_enabled {
                        take.loop_enabled = v;
                    }
                }
            }
            // active 投影已更新，写回 Take 权威数据。
            c.sync_take_from_flat();

            end_sec = Some(c.start_sec + c.length_sec);
        }

        if let Some(v) = end_sec {
            self.ensure_project_end_sec(v);
        }
    }

    pub fn patch_clips_state(&mut self, updates: &[BulkClipStatePatch]) {
        for update in updates {
            self.patch_clip_state(&update.clip_id, update.patch.clone());
        }
    }

    pub fn create_clips_bulk(&mut self, payload: &CreateClipsBulkPayload) -> Vec<String> {
        let mut created_clip_ids = Vec::with_capacity(payload.templates.len());

        for template in &payload.templates {
            let created_id = if let Some(source_clip_id) = template.source_clip_id.as_ref() {
                if let Some(source_clip) = self.clips.iter().find(|clip| clip.id == *source_clip_id)
                {
                    let mut duplicated = source_clip.clone();
                    duplicated.id = new_id("clip");
                    // 克隆后以投影为准写回 Take：运行时编辑全部发生在投影上，
                    // 若存在尚未 sync 的最新修改，normalize_takes() 会用旧 Take
                    // 数据覆盖它（如录音自动增益被回退）。
                    duplicated.sync_take_from_flat();
                    duplicated.remap_take_ids();
                    duplicated.group_id = None;
                    duplicated.track_id = template.track_id.clone();
                    duplicated.name = template.name.clone();
                    duplicated.start_sec = template.start_sec;
                    duplicated.length_sec = template.length_sec;
                    self.ensure_project_end_sec(duplicated.start_sec + duplicated.length_sec);
                    let created_id = duplicated.id.clone();
                    self.clips.push(duplicated);
                    created_id
                } else {
                    self.add_clip(
                        Some(template.track_id.clone()),
                        Some(template.name.clone()),
                        Some(template.start_sec),
                        Some(template.length_sec),
                        template.source_path.clone(),
                    )
                }
            } else {
                self.add_clip(
                    Some(template.track_id.clone()),
                    Some(template.name.clone()),
                    Some(template.start_sec),
                    Some(template.length_sec),
                    template.source_path.clone(),
                )
            };

            self.patch_clip_state(
                &created_id,
                ClipStatePatch {
                    name: Some(template.name.clone()),
                    start_sec: Some(template.start_sec),
                    length_sec: Some(template.length_sec),
                    gain: template.gain,
                    muted: template.muted,
                    source_start_sec: template.source_start_sec,
                    source_end_sec: template.source_end_sec,
                    playback_rate: template.playback_rate,
                    clip_playback_rate: None,
                    reversed: template.reversed,
                    loop_enabled: template.loop_enabled,
                    snap_offset_sec: template.snap_offset_sec,
                    fade_in_sec: template.fade_in_sec,
                    fade_out_sec: template.fade_out_sec,
                    fade_in_shape: template.fade_in_shape,
                    fade_out_shape: template.fade_out_shape,
                    fade_in_dir: template.fade_in_dir,
                    fade_out_dir: template.fade_out_dir,
                    auto_fade_in_sec: template.auto_fade_in_sec,
                    auto_fade_out_sec: template.auto_fade_out_sec,
                    color: None,
                    formant_morph: None,
                },
            );

            if let Some(linked_params) = template.linked_params.as_ref() {
                self.apply_linked_params_to_clip(&created_id, linked_params);
            }

            if template.midi_note_data.is_some() || template.midi_fill_gaps.is_some() {
                if let Some(clip) = self.clips.iter_mut().find(|c| c.id == created_id) {
                    if let Some(ref midi_data) = template.midi_note_data {
                        clip.midi_note_data = Some(midi_data.clone());
                        clip.source_path = None;
                        clip.source_path_relative = None;
                        clip.color = "cyan".to_string();
                        clip.pitch_range = Some(PitchRange {
                            min: 0.0,
                            max: 127.0,
                        });
                    }
                    if let Some(fill_gaps) = template.midi_fill_gaps {
                        clip.midi_fill_gaps = fill_gaps;
                    }
                    // MIDI 内容写在投影上，写回 Take 权威数据。
                    clip.sync_take_from_flat();
                }
            }

            created_clip_ids.push(created_id);
        }

        if payload.select_created_clips {
            self.selected_clip_id = created_clip_ids.first().cloned();
            if let Some(first_created_clip) = created_clip_ids
                .first()
                .and_then(|id| self.clips.iter().find(|clip| clip.id == *id))
            {
                self.selected_track_id = Some(first_created_clip.track_id.clone());
                self.playhead_sec = first_created_clip.start_sec;
            }
        }

        created_clip_ids
    }

    /// 翻转单个 Take 的播放方向（正放 ↔ 倒放）。
    ///
    /// 与 [`TimelineState::patch_clip_state`] 的方向翻转同口径：按翻转前的
    /// 消费窗口换算该 Take 的源窗口/Loop 锚点（span 用该 Take 的组合消费
    /// 速率），使消费内容不变。**不**受“同步编辑所有 Take”影响 —— 这是
    /// 针对单个 Take 的内容操作。
    ///
    /// 返回该 Take 是否为 active take（调用方据此决定是否需要重调度音频
    /// 分析/共振峰重建 —— inactive take 的翻转不改变当前可听内容）。
    pub fn set_clip_take_reversed(
        &mut self,
        clip_id: &str,
        take_id: &str,
        reversed: bool,
    ) -> Result<bool, String> {
        let clip = self
            .clips
            .iter_mut()
            .find(|c| c.id == clip_id)
            .ok_or_else(|| format!("clip not found: {clip_id}"))?;
        let clip_len = clip.length_sec;
        let clip_rate = clip.clip_playback_rate;
        let is_active = clip.active_take_id.as_deref() == Some(take_id);
        if is_active {
            // active take 以内存投影为消费权威：先把 flat 物化到 Take 条目，
            // 避免条目滞后的窗口被当作翻转前的消费窗口。
            clip.sync_take_from_flat();
        }
        let take = clip
            .takes
            .iter_mut()
            .find(|t| t.id == take_id)
            .ok_or_else(|| format!("take not found: {take_id}"))?;
        if take.reversed != reversed {
            flip_take_playback_direction(take, clip_len, clip_rate as f64);
            take.reversed = reversed;
        }
        if is_active {
            // active take 的翻转改变可听内容：物化到内存投影。
            let take = take.clone();
            take.apply_to_clip(clip);
        }
        Ok(is_active)
    }

    /// 把选中的多个 Clip 聚合为一个多 Take Clip。
    ///
    /// - 结果 Clip 的时间范围为所有源 Clip 的最小起点～最大终点；
    /// - 每个源 Clip 的全部 Take 都进入结果 Clip，并按源 Clip 在时间轴上的
    ///   原始位置换算 source 窗口（正放/倒放/Loop 分别处理），使内容在
    ///   时间轴上保持原对齐；
    /// - 结果 Clip 放在 `clip_ids` 第一个源 Clip 的轨道上，并删除所有源 Clip。
    pub fn pack_clips_into_takes(&mut self, clip_ids: &[String]) -> Option<String> {
        let mut sources: Vec<Clip> = Vec::new();
        for id in clip_ids {
            if let Some(clip) = self.clips.iter().find(|c| c.id == *id) {
                if !sources.iter().any(|c| c.id == clip.id) {
                    sources.push(clip.clone());
                }
            }
        }
        if sources.len() < 2 {
            return None;
        }

        let track_id = sources[0].track_id.clone();
        let start = sources
            .iter()
            .map(|c| c.start_sec)
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        let end = sources
            .iter()
            .map(|c| c.start_sec + c.length_sec)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(start + 0.01);
        let length = (end - start).max(0.01);

        let mut packed_takes: Vec<ClipTake> = Vec::new();
        for source in &mut sources {
            source.normalize_takes();
            // 源 Clip 在合并区间内的本地偏移（start 为全部源的最小起点，
            // 故 delta ≥ 0）。打包后每个 Take 的内容必须仍在自己原本的
            // 时间位置出声：
            // - 非 Loop 正放：窗口 [ss,se) 整体前移 δ·rate —— 新起点
            //   ss−δ·r 使内容在 t=δ 处开始播放（负起点 = 前导静音），
            //   窗口长度保持源自身的裁剪长度，其余区段渲染静音；
            // - 非 Loop 倒放：消费锚点是 source_end，整体后移 δ·rate；
            // - Loop：回绕发生在整个媒体文件上，按锚点在 mod-D 域内
            //   平移（正放减、倒放加），容器全长由回绕铺满。
            // 注意：与 split 不同，这里**有意**不继承源 Clip 的 fade/
            // muted/snap_offset/formant 覆盖 —— 打包产物是全新的多 Take
            // 容器，这些 Item 级属性以第一个源之外的默认值起步。
            let delta = (source.start_sec - start).max(0.0);
            for take in &source.takes {
                let source_clip_rate =
                    if source.clip_playback_rate.is_finite() && source.clip_playback_rate > 1e-6 {
                        source.clip_playback_rate as f64
                    } else {
                        1.0
                    };
                let take_rate = if take.playback_rate.is_finite() && take.playback_rate > 1e-6 {
                    take.playback_rate as f64
                } else {
                    1.0
                };
                // 原 Clip 级速率并入新 Take；新 Clip 级速率归一为 1，
                // 因此时间轴消费速率保持不变。
                let rate = source_clip_rate * take_rate;
                let mut packed = take.clone();
                packed.id = new_id("take");
                if source.takes.len() > 1 {
                    packed.name = format!("{} · {}", source.name, packed.name);
                } else if !source.name.is_empty() {
                    packed.name = source.name.clone();
                }
                packed.playback_rate = (rate as f32).clamp(0.1, 10.0);
                if take.loop_enabled {
                    match clip_take_media_duration_sec(take).filter(|d| d.is_finite() && *d > 1e-9)
                    {
                        Some(media_total) => {
                            let wrap = |value: f64| -> f64 {
                                let m = value % media_total;
                                if m < 0.0 {
                                    m + media_total
                                } else {
                                    m
                                }
                            };
                            if take.reversed {
                                packed.source_end_sec = wrap(take.source_end_sec + delta * rate);
                            } else {
                                packed.source_start_sec =
                                    wrap(take.source_start_sec - delta * rate);
                            }
                        }
                        None => {
                            // 媒体时长未知：退化为不回绕的相位平移（尽力而为）。
                            if take.reversed {
                                packed.source_end_sec = take.source_end_sec + delta * rate;
                            } else {
                                packed.source_start_sec = take.source_start_sec - delta * rate;
                            }
                        }
                    }
                } else if take.reversed {
                    packed.source_end_sec = take.source_end_sec + delta * rate;
                    packed.source_start_sec = take.source_start_sec + delta * rate;
                } else {
                    packed.source_start_sec = take.source_start_sec - delta * rate;
                    packed.source_end_sec = take.source_end_sec - delta * rate;
                }
                if let Some(notes) = packed.midi_note_data.as_mut() {
                    for note in notes {
                        note.start_sec += delta;
                        note.end_sec += delta;
                    }
                }
                packed_takes.push(packed);
            }
        }
        if packed_takes.is_empty() {
            return None;
        }

        let active_take_id = Some(packed_takes[0].id.clone());
        let packed_id = new_id("clip");
        let name = sources[0].name.clone();
        let color = sources[0].color.clone();
        let mut packed = Clip {
            id: packed_id.clone(),
            group_id: None,
            track_id,
            name,
            start_sec: start,
            length_sec: length,
            color,
            takes: packed_takes,
            active_take_id,
            clip_playback_rate: 1.0,
            source_path: None,
            source_path_relative: None,
            duration_sec: None,
            duration_frames: None,
            source_sample_rate: None,
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: length,
            playback_rate: 1.0,
            reversed: false,
            loop_enabled: false,
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_shape: 1.0,
            fade_out_shape: 1.0,
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
        };
        packed.normalize_takes();

        let remove: std::collections::HashSet<&str> =
            sources.iter().map(|c| c.id.as_str()).collect();
        self.clips.retain(|c| !remove.contains(c.id.as_str()));
        self.ensure_project_end_sec(start + length);
        self.clips.push(packed);
        self.selected_clip_id = Some(packed_id.clone());
        self.selected_track_id = Some(
            self.clips
                .iter()
                .find(|c| c.id == packed_id)?
                .track_id
                .clone(),
        );
        Some(packed_id)
    }

    /// 与 build_track_payload 一致的 DFS 可视顺序（自上而下的轨道 id 列表）。
    fn visual_track_ids(&self) -> Vec<String> {
        let mut by_parent: HashMap<Option<String>, Vec<&Track>> = HashMap::new();
        for t in &self.tracks {
            by_parent.entry(t.parent_id.clone()).or_default().push(t);
        }
        for group in by_parent.values_mut() {
            group.sort_by_key(|t| t.order);
        }

        let mut out: Vec<String> = Vec::with_capacity(self.tracks.len());
        fn walk(
            t: &Track,
            by_parent: &HashMap<Option<String>, Vec<&Track>>,
            out: &mut Vec<String>,
        ) {
            out.push(t.id.clone());
            if let Some(children) = by_parent.get(&Some(t.id.clone())) {
                for child in children {
                    walk(child, by_parent, out);
                }
            }
        }
        if let Some(roots) = by_parent.get(&None) {
            for root in roots {
                walk(root, &by_parent, &mut out);
            }
        }
        out
    }

    /// 把一个多 Take Clip 展开为多个独立 Clip（每个 Take 一个），
    /// 保留原 Clip 的几何位置、长度与容器级属性。返回新 Clip id 列表。
    ///
    /// 放置采用"向下展开"：第 idx 个 Take 放到源轨道可视顺序下方第 idx 行的
    /// 轨道上（idx=0 即源轨道本身）；下方没有现成轨道时，克隆源轨道的设置
    /// （父轨道 / 算法 / 音量 / 颜色等）在可视顺序底部新建轨道，保证每个
    /// Take 独占一行，而不是全部挤在源轨道上。
    pub fn explode_clip_takes(&mut self, clip_id: &str) -> Vec<String> {
        let Some(mut source) = self.clips.iter().find(|c| c.id == clip_id).cloned() else {
            return Vec::new();
        };
        source.normalize_takes();
        let source_track_id = source.track_id.clone();
        let source_track = self
            .tracks
            .iter()
            .find(|t| t.id == source_track_id)
            .cloned();
        let mut visual_ids = self.visual_track_ids();
        let source_row = visual_ids
            .iter()
            .position(|id| *id == source_track_id)
            .unwrap_or(0);

        let mut created = Vec::new();
        for (idx, take) in source.takes.iter().enumerate() {
            // 目标行 = 源轨道下方第 idx 行；没有现成轨道则新建（克隆源轨道
            // 设置，追加到可视顺序末尾）并重算可视行表。
            let target_row = source_row + idx;
            while visual_ids.len() <= target_row {
                let Some(template) = source_track.as_ref() else {
                    break;
                };
                let track = Track {
                    id: new_id("track"),
                    name: template.name.clone(),
                    parent_id: template.parent_id.clone(),
                    order: self.next_track_order,
                    muted: template.muted,
                    solo: template.solo,
                    volume: template.volume,
                    compose_enabled: template.compose_enabled,
                    pitch_analysis_algo: template.pitch_analysis_algo.clone(),
                    color: template.color.clone(),
                };
                self.next_track_order += 1;
                self.tracks.push(track);
                visual_ids = self.visual_track_ids();
            }
            let Some(target_track_id) = visual_ids.get(target_row).cloned() else {
                continue;
            };

            let mut clip = source.clone();
            clip.id = new_id("clip");
            clip.group_id = None;
            clip.track_id = target_track_id;
            if idx > 0 {
                clip.name = if take.name.trim().is_empty() {
                    format!("{} {}", source.name, idx + 1)
                } else {
                    take.name.clone()
                };
            }
            clip.takes = vec![take.clone()];
            clip.active_take_id = Some(take.id.clone());
            clip.normalize_takes();
            let id = clip.id.clone();
            self.ensure_project_end_sec(clip.start_sec + clip.length_sec);
            self.clips.push(clip);
            created.push(id);
        }
        self.clips.retain(|c| c.id != clip_id);
        if let Some(first) = created.first() {
            self.selected_clip_id = Some(first.clone());
            if let Some(clip) = self.clips.iter().find(|c| c.id == *first) {
                self.selected_track_id = Some(clip.track_id.clone());
            }
        }
        created
    }

    pub fn duplicate_clips_bulk(&mut self, payload: &DuplicateClipsBulkPayload) -> Vec<String> {
        let unique_source_ids: Vec<String> = {
            let mut seen = HashSet::new();
            payload
                .source_clip_ids
                .iter()
                .filter(|id| seen.insert((*id).clone()))
                .cloned()
                .collect()
        };
        let source_clips: Vec<Clip> = unique_source_ids
            .iter()
            .filter_map(|id| self.clips.iter().find(|clip| clip.id == *id).cloned())
            .collect();
        if source_clips.is_empty() {
            return Vec::new();
        }

        // Capture original group IDs before source_clips is consumed
        let original_group_ids: Vec<Option<String>> =
            source_clips.iter().map(|c| c.group_id.clone()).collect();

        let source_track_order = self
            .tracks
            .iter()
            .map(|track| track.id.clone())
            .collect::<Vec<_>>();
        let source_track_index_by_id = source_track_order
            .iter()
            .enumerate()
            .map(|(index, id)| (id.clone(), index))
            .collect::<HashMap<_, _>>();
        let ordered_source_track_ids = {
            let mut seen = HashSet::new();
            let mut track_ids = source_clips
                .iter()
                .filter_map(|clip| {
                    if seen.insert(clip.track_id.clone()) {
                        Some(clip.track_id.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            track_ids.sort_by_key(|track_id| {
                source_track_index_by_id
                    .get(track_id)
                    .copied()
                    .unwrap_or(usize::MAX)
            });
            track_ids
        };

        let mut explicit_mapping = HashMap::new();
        if payload.place_on_selected_track {
            if let Some(selected_track_id) = self.selected_track_id.clone() {
                let track_order = self
                    .tracks
                    .iter()
                    .map(|track| track.id.clone())
                    .collect::<Vec<_>>();
                if let Some(selected_index) =
                    track_order.iter().position(|id| *id == selected_track_id)
                {
                    let needed_last_index =
                        selected_index + ordered_source_track_ids.len().saturating_sub(1);
                    while self.tracks.len() <= needed_last_index {
                        self.add_track(Some("Track".to_string()), None, None);
                    }

                    for (offset, source_track_id) in ordered_source_track_ids.iter().enumerate() {
                        if let Some(target_track) = self.tracks.get(selected_index + offset) {
                            explicit_mapping
                                .insert(source_track_id.clone(), target_track.id.clone());
                        }
                    }
                }
            }
        }

        let mut new_track_mapping = HashMap::new();
        if matches!(payload.track_mode, DuplicateClipsTrackMode::NewTracks) {
            for source_track_id in &ordered_source_track_ids {
                let new_track_id = self.add_track(Some("Track".to_string()), None, None);
                new_track_mapping.insert(source_track_id.clone(), new_track_id);
            }
        }

        // 在改动曲线之前一次性预提取全部源剪辑的联动参数线：应用某个副本的
        // 曲线会写入目标范围，若与后续源剪辑的范围重叠，串行"提取→应用"会
        // 让后续提取读到被覆盖后的值。
        let mut linked_params_by_index: Vec<Option<LinkedParamCurvesPayload>> =
            Vec::with_capacity(source_clips.len());
        for source in &source_clips {
            let linked = if payload.copy_linked_params && source.length_sec > 0.0 {
                self.resolve_root_track_id(&source.track_id).and_then(|root_track_id| {
                    self.extract_linked_params_from_root_range(
                        &root_track_id,
                        source.start_sec,
                        source.length_sec,
                    )
                })
            } else {
                None
            };
            linked_params_by_index.push(linked);
        }

        let mut created_clip_ids = Vec::new();
        for (source_index, source) in source_clips.into_iter().enumerate() {
            let target_track_id = if let Some(mapped) = explicit_mapping.get(&source.track_id) {
                mapped.clone()
            } else {
                match &payload.track_mode {
                    DuplicateClipsTrackMode::SameTrack => source.track_id.clone(),
                    DuplicateClipsTrackMode::OffsetTracks { offset } => {
                        let source_index = source_track_index_by_id
                            .get(&source.track_id)
                            .copied()
                            .unwrap_or(0) as i32;
                        let target_index = (source_index + *offset)
                            .clamp(0, self.tracks.len().saturating_sub(1) as i32)
                            as usize;
                        self.tracks
                            .get(target_index)
                            .map(|track| track.id.clone())
                            .unwrap_or_else(|| source.track_id.clone())
                    }
                    DuplicateClipsTrackMode::ExplicitMapping { mapping } => mapping
                        .get(&source.track_id)
                        .cloned()
                        .unwrap_or_else(|| source.track_id.clone()),
                    DuplicateClipsTrackMode::NewTracks => new_track_mapping
                        .get(&source.track_id)
                        .cloned()
                        .unwrap_or_else(|| source.track_id.clone()),
                }
            };

            let new_root_track_id = self.resolve_root_track_id(&target_track_id);
            let linked_params = linked_params_by_index[source_index].take();

            let mut duplicated = source.clone();
            duplicated.id = new_id("clip");
            // 同 create_clips_bulk：克隆后以 active 投影写回 Take，
            // 避免未同步的最新投影修改被旧 Take 数据覆盖。
            duplicated.sync_take_from_flat();
            duplicated.remap_take_ids();
            duplicated.track_id = target_track_id;
            duplicated.start_sec = (duplicated.start_sec + payload.delta_sec).max(0.0);
            if payload.rename_copies.unwrap_or(true) {
                duplicated.name = format!("{} Copy", duplicated.name);
            }
            self.ensure_project_end_sec(duplicated.start_sec + duplicated.length_sec);
            created_clip_ids.push(duplicated.id.clone());
            self.clips.push(duplicated.clone());

            if let (Some(linked_params), Some(new_root_track_id)) =
                (linked_params, new_root_track_id)
            {
                self.apply_linked_params_to_root_range(
                    &new_root_track_id,
                    duplicated.start_sec,
                    &linked_params,
                );
            }
        }

        // Remap group IDs: each original group gets a new unique group ID
        // so duplicates are in independent groups from the originals.
        {
            let mut group_remap: HashMap<String, String> = HashMap::new();
            for gid_opt in &original_group_ids {
                if let Some(ref gid) = gid_opt {
                    group_remap
                        .entry(gid.clone())
                        .or_insert_with(|| Uuid::new_v4().to_string());
                }
            }
            if !group_remap.is_empty() {
                for clip in &mut self.clips {
                    if created_clip_ids.contains(&clip.id) {
                        if let Some(ref old_gid) = clip.group_id {
                            if let Some(new_gid) = group_remap.get(old_gid) {
                                clip.group_id = Some(new_gid.clone());
                            }
                        }
                    }
                }
            }
        }

        if payload.select_created_clips {
            self.selected_clip_id = created_clip_ids.first().cloned();
            if let Some(first_created_clip) = created_clip_ids
                .first()
                .and_then(|id| self.clips.iter().find(|clip| clip.id == *id))
            {
                self.selected_track_id = Some(first_created_clip.track_id.clone());
                self.playhead_sec = first_created_clip.start_sec;
            }
        }

        created_clip_ids
    }

    /// 把 SnapOffset 按标记点的**时间线绝对位置**归属到分割后的两段之一：
    /// 包含标记点的那段继承"标记位置 − 该段起点"（重新以段起点为基准），
    /// 另一段为 0。右段优先（延伸重叠可能使右段起点越过标记点）。
    ///
    /// 调用方必须在两段的**最终几何**确定之后调用：
    /// - 普通 `split_clip`：切割边界即最终边界；
    /// - `split_clip_with_transition`："延伸重叠"会平移右段起点/延伸左段
    ///   终点，须在过渡应用后再调用。
    fn assign_snap_offset_to_split(&mut self, left_id: &str, right_id: &str, marker_pos_sec: f64) {
        let Some(l_idx) = self.clips.iter().position(|c| c.id == left_id) else {
            return;
        };
        let Some(r_idx) = self.clips.iter().position(|c| c.id == right_id) else {
            return;
        };
        let m = marker_pos_sec.max(0.0);
        let (r_start, r_len) = {
            let r = &self.clips[r_idx];
            (r.start_sec, r.length_sec)
        };
        let (l_start, l_len) = {
            let l = &self.clips[l_idx];
            (l.start_sec, l.length_sec)
        };
        if m >= r_start - 1e-9 {
            self.clips[r_idx].snap_offset_sec = (m - r_start).min(r_len.max(0.0));
            self.clips[l_idx].snap_offset_sec = 0.0;
        } else {
            self.clips[l_idx].snap_offset_sec = (m - l_start).clamp(0.0, l_len.max(0.0));
            self.clips[r_idx].snap_offset_sec = 0.0;
        }
    }

    pub fn split_clip(&mut self, clip_id: &str, split_sec: f64) -> Option<String> {
        let Some(idx) = self.clips.iter().position(|c| c.id == clip_id) else {
            return None;
        };

        // 分割是 Clip 容器级操作：无论“同步编辑所有 Take”是否启用，
        // 都必须把每个 Take 的 source 窗口 / MIDI 内容切到对应侧。
        let mut left = self.clips[idx].clone();
        // 先把 active 投影中的最新修改写回对应 Take；这样既保留其它 Take，
        // 也保持旧逻辑对 active 字段直接修改的兼容行为。
        left.sync_take_from_flat();
        let start = left.start_sec;
        let end = start + left.length_sec;
        let split = split_sec.clamp(start, end);
        if split <= start + 1e-6 || split >= end - 1e-6 {
            return None;
        }

        self.ensure_project_end_sec(end);

        let left_len = split - start;
        let right_len = end - split;
        let clip_rate = if left.clip_playback_rate.is_finite() && left.clip_playback_rate > 1e-6 {
            left.clip_playback_rate as f64
        } else {
            1.0
        };

        // 右侧也必须从分割前的同一份 Take 快照推导；
        // 绝不能从已经改写过的左侧 Take 再二次推导。
        let source_clip = left.clone();

        left.length_sec = left_len;
        for take in &mut left.takes {
            split_clip_take_window(take, clip_rate, left_len, right_len, false);
        }

        // Fade semantics on split:
        // - fade-in is anchored to the original start, so only the left clip should keep it.
        // - fade-out is anchored to the original end, so only the right clip should keep it.
        // - 切割产生的新边缘（左 clip 的右缘、右 clip 的左缘）**不继承任何淡化**，
        //   包括自动交叉淡化与手动淡化。
        left.fade_in_sec = left.fade_in_sec.min(left_len.max(0.0));
        left.fade_out_sec = 0.0;
        left.auto_fade_out_sec = 0.0;
        left.auto_fade_in_sec = left.auto_fade_in_sec.min(left_len.max(0.0));

        // SnapOffset 归属在两段最终几何确定后统一处理。
        let marker_pos_sec = start + left.snap_offset_sec;

        let mut right = source_clip.clone();
        right.id = new_id("clip");
        right.remap_take_ids();
        right.start_sec = split;
        right.length_sec = right_len;
        right.fade_in_sec = 0.0;
        right.fade_out_sec = right.fade_out_sec.min(right_len.max(0.0));
        right.auto_fade_in_sec = 0.0;
        right.auto_fade_out_sec = right.auto_fade_out_sec.min(right_len.max(0.0));
        for take in &mut right.takes {
            split_clip_take_window(take, clip_rate, left_len, right_len, true);
        }

        // Propagate group_id to the split-off right clip
        right.group_id = left.group_id.clone();

        // 把 active Take 物化到内存投影，供既有渲染/编辑消费者使用。
        left.normalize_takes();
        right.normalize_takes();
        self.clips[idx] = left;

        let right_id = right.id.clone();
        self.clips.push(right);
        self.assign_snap_offset_to_split(clip_id, &right_id, marker_pos_sec);
        Some(right_id)
    }
    /// 分割 clip，并在分割完成后根据全局“分割过渡”设置应用淡入淡出或延伸重叠。
    pub fn split_clip_with_transition(
        &mut self,
        clip_id: &str,
        split_sec: f64,
        opts: &SplitTransitionOptions,
    ) -> Option<String> {
        // 快照分割前的几何与 SnapOffset："延伸重叠"过渡会平移右段起点、
        // 延伸左段终点 —— SnapOffset 的归属必须以**过渡后的最终几何**
        // 重新计算（split_clip 内部按切割边界做的预分配会被此处覆盖）。
        let (orig_start_sec, orig_snap_offset_sec) = {
            let c = self.clips.iter().find(|c| c.id == clip_id)?;
            (c.start_sec, c.snap_offset_sec)
        };
        let right_id = self.split_clip(clip_id, split_sec)?;
        let effective_duration_sec = match opts.duration_unit {
            SplitTransitionDurationUnit::Seconds => opts.duration_sec,
            SplitTransitionDurationUnit::Percent => {
                let left_len = self
                    .clips
                    .iter()
                    .find(|c| c.id == clip_id)
                    .map(|c| c.length_sec)
                    .unwrap_or(0.0);
                let right_len = self
                    .clips
                    .iter()
                    .find(|c| c.id == right_id)
                    .map(|c| c.length_sec)
                    .unwrap_or(0.0);
                (left_len + right_len) * opts.duration_percent / 100.0
            }
        };
        if opts.enabled && effective_duration_sec.is_finite() && effective_duration_sec > 0.0 {
            self.apply_split_transition(clip_id, &right_id, opts, effective_duration_sec);
        }

        // SnapOffset 重算："延伸重叠"已平移右段起点/延伸左段终点 ——
        // 以最终几何按同一归属规则重新分配（覆盖 split_clip 的预分配）。
        let marker_pos_sec = orig_start_sec + orig_snap_offset_sec;
        self.assign_snap_offset_to_split(clip_id, &right_id, marker_pos_sec);
        Some(right_id)
    }

    fn apply_split_transition(
        &mut self,
        left_id: &str,
        right_id: &str,
        opts: &SplitTransitionOptions,
        duration_sec: f64,
    ) {
        let Some(left_idx) = self.clips.iter().position(|c| c.id == left_id) else {
            return;
        };
        let Some(right_idx) = self.clips.iter().position(|c| c.id == right_id) else {
            return;
        };

        let duration = duration_sec.max(0.0);
        if duration <= 0.0 {
            return;
        }

        // 仅淡入淡出模式：切割处创建的是“手动淡化”（不随重叠自动变化）。
        // 淡化曲线：opts.curve=None（“keep”）时**不修改**两段的 shape/dir，
        // 保留原 Clip 的曲线类型；否则按新版预设写入（含各侧默认曲率）。
        let set_manual_fade = |left: &mut Clip, right: &mut Clip, fade_len: f64| {
            left.fade_out_sec = fade_len.min(left.length_sec);
            right.fade_in_sec = fade_len.min(right.length_sec);
            left.auto_fade_out_sec = 0.0;
            right.auto_fade_in_sec = 0.0;
            if let Some((shape, dir_in, dir_out)) =
                opts.curve.as_deref().and_then(split_transition_curve_spec)
            {
                left.fade_out_shape = shape;
                left.fade_out_dir = dir_out;
                right.fade_in_shape = shape;
                right.fade_in_dir = dir_in;
            }
        };
        // 延伸重叠模式：重叠区的交叉淡化写入“自动交叉淡化”长度（跟随重叠，
        // 分开后自动归零、手动 fade 恢复），适配新的自动交叉淡化模型。
        let set_auto_fade = |left: &mut Clip, right: &mut Clip, fade_len: f64| {
            left.auto_fade_out_sec = fade_len.min(left.length_sec);
            right.auto_fade_in_sec = fade_len.min(right.length_sec);
            left.fade_out_sec = 0.0;
            right.fade_in_sec = 0.0;
            if let Some((shape, dir_in, dir_out)) =
                opts.curve.as_deref().and_then(split_transition_curve_spec)
            {
                left.fade_out_shape = shape;
                left.fade_out_dir = dir_out;
                right.fade_in_shape = shape;
                right.fade_in_dir = dir_in;
            }
        };

        match opts.mode {
            SplitTransitionMode::FadeOnly => {
                let (left, right) = self.clips.split_at_mut(right_idx);
                set_manual_fade(&mut left[left_idx], &mut right[0], duration);
            }
            SplitTransitionMode::ExtendOverlap => {
                // 前 clip 与后 clip 各向外延长 X，形成 2X 秒的重叠区域。
                // 源媒体可用量不再设限（产品决策：向左/向右延伸均无界）——
                // 越出源素材的部分由渲染管线按静音填充；仅保留时间轴约束
                //（后 clip 不能越过时间轴起点 0）。
                let left_rate = {
                    let left = &self.clips[left_idx];
                    if left.playback_rate.is_finite() && left.playback_rate > 0.0 {
                        left.playback_rate as f64
                    } else {
                        1.0
                    }
                };
                let left_grow = duration.max(0.0);

                let right_rate = {
                    let right = &self.clips[right_idx];
                    if right.playback_rate.is_finite() && right.playback_rate > 0.0 {
                        right.playback_rate as f64
                    } else {
                        1.0
                    }
                };
                let right_grow = duration.min(self.clips[right_idx].start_sec).max(0.0);

                let overlap_sec = left_grow + right_grow;
                if overlap_sec <= 0.0 {
                    return;
                }

                // 前 clip 末尾向后延长 left_grow，同时扩展 source 范围，
                // 保证素材内容在时间轴上的位置不变（等价于拖拽 clip 末尾）。
                // Loop（循环源）clip：延长部分由循环回绕内容填充，不改源窗口。
                // 非 Loop：正放终点/倒放起点随增长派生，越出媒体的部分为静音。
                //
                // 与 split 相同的纪律：分割过渡是 Clip 容器级操作，必须把每个
                // Take 的窗口按其自身组合速率（clip 倍率 × take 速率）做同一
                // 变换；否则切换到 inactive 倒放/Loop take 时内容会错位
                // grow×rate 秒。
                {
                    let left = &mut self.clips[left_idx];
                    let clip_rate =
                        if left.clip_playback_rate.is_finite() && left.clip_playback_rate > 1e-6 {
                            left.clip_playback_rate as f64
                        } else {
                            1.0
                        };
                    let combined = |take_rate: f32| -> f64 {
                        let tr = if take_rate.is_finite() && take_rate > 1e-6 {
                            take_rate as f64
                        } else {
                            1.0
                        };
                        clip_rate * tr
                    };
                    left.length_sec += left_grow;
                    if !left.loop_enabled {
                        if left.reversed {
                            for take in &mut left.takes {
                                take.source_start_sec -= left_grow * combined(take.playback_rate);
                            }
                            left.source_start_sec = left.source_start_sec - left_grow * left_rate;
                        } else {
                            for take in &mut left.takes {
                                take.source_end_sec += left_grow * combined(take.playback_rate);
                            }
                            left.source_end_sec = left.source_end_sec + left_grow * left_rate;
                        }
                    }
                    // 源窗口写在 active 投影上，立即写回 Take 权威数据；
                    // 否则后续克隆路径（复制/粘贴/合并）会用旧 Take 窗口
                    // 覆盖这次延伸。
                    left.sync_take_from_flat();
                }

                // 后 clip 起始位置向前延长 right_grow，同时扩展 source 范围。
                {
                    let media_dur = self.source_file_duration_sec(&self.clips[right_idx]);
                    // Loop（循环源）锚点环绕：媒体时长未知时原样保留。
                    let wrap_anchor = |value: f64| -> f64 {
                        match media_dur.filter(|d| d.is_finite() && *d > 1e-9) {
                            Some(d) => {
                                let m = value % d;
                                if m < 0.0 {
                                    m + d
                                } else {
                                    m
                                }
                            }
                            None => value,
                        }
                    };
                    let right = &mut self.clips[right_idx];
                    let clip_rate = if right.clip_playback_rate.is_finite()
                        && right.clip_playback_rate > 1e-6
                    {
                        right.clip_playback_rate as f64
                    } else {
                        1.0
                    };
                    let combined = |take_rate: f32| -> f64 {
                        let tr = if take_rate.is_finite() && take_rate > 1e-6 {
                            take_rate as f64
                        } else {
                            1.0
                        };
                        clip_rate * tr
                    };
                    right.start_sec = (right.start_sec - right_grow).max(0.0);
                    right.length_sec += right_grow;
                    if right.loop_enabled {
                        // Loop：头部增长若不动源窗口，新起点会从旧相位重新消费，
                        // 内容整体后移并与左尾交叉淡化错相。锚点须按增长量沿
                        // 消费方向回退并环绕（正放减 / 倒放加），保持内容的
                        // 绝对时间线位置不变 —— 与非 Loop 分支的语义一致。
                        if right.reversed {
                            right.source_end_sec =
                                wrap_anchor(right.source_end_sec + right_grow * right_rate);
                        } else {
                            right.source_start_sec =
                                wrap_anchor(right.source_start_sec - right_grow * right_rate);
                        }
                        // 每个 Take 的回绕周期是它自己的媒体时长。
                        for take in &mut right.takes {
                            let r = combined(take.playback_rate);
                            let take_media = clip_take_media_duration_sec(take)
                                .filter(|d| d.is_finite() && *d > 1e-9);
                            let take_wrap = |value: f64| -> f64 {
                                match take_media {
                                    Some(d) => {
                                        let m = value % d;
                                        if m < 0.0 {
                                            m + d
                                        } else {
                                            m
                                        }
                                    }
                                    None => value,
                                }
                            };
                            if right.reversed {
                                take.source_end_sec =
                                    take_wrap(take.source_end_sec + right_grow * r);
                            } else {
                                take.source_start_sec =
                                    take_wrap(take.source_start_sec - right_grow * r);
                            }
                        }
                    } else if right.reversed {
                        // 倒放非 Loop：头部延伸使锚点(source_end)越过媒体时长
                        // → 前导静音，不再按媒体时长钳制。窗口起点随新锚点/
                        // 长度同步派生，保持存储字段 == 消费窗口。
                        for take in &mut right.takes {
                            let r = combined(take.playback_rate);
                            take.source_end_sec += right_grow * r;
                            take.source_start_sec = take.source_end_sec - right.length_sec * r;
                        }
                        right.source_end_sec = right.source_end_sec + right_grow * right_rate;
                        right.source_start_sec =
                            right.source_end_sec - right.length_sec * right_rate;
                    } else {
                        // 正放非 Loop：头部延伸使起点向下穿越媒体起点 → 前导
                        // 静音（派生窗口），不再钳制到 0。终点同步派生，
                        // 保持存储字段 == 消费窗口。
                        for take in &mut right.takes {
                            let r = combined(take.playback_rate);
                            take.source_start_sec -= right_grow * r;
                            take.source_end_sec = take.source_start_sec + right.length_sec * r;
                        }
                        right.source_start_sec = right.source_start_sec - right_grow * right_rate;
                        right.source_end_sec =
                            right.source_start_sec + right.length_sec * right_rate;
                    }
                    // 与左侧同理：源窗口修改必须写回 Take 权威数据。
                    right.sync_take_from_flat();
                }

                if opts.overlap_fades {
                    let (left, right) = self.clips.split_at_mut(right_idx);
                    set_auto_fade(&mut left[left_idx], &mut right[0], overlap_sec);
                }
            }
        }

        if let Some(clip) = self.clips.get(left_idx) {
            self.ensure_project_end_sec(clip.start_sec + clip.length_sec);
        }
    }

    fn source_file_duration_sec(&self, clip: &Clip) -> Option<f64> {
        if let (Some(frames), Some(sample_rate)) = (clip.duration_frames, clip.source_sample_rate) {
            if sample_rate > 0 && frames > 0 {
                return Some(frames as f64 / sample_rate as f64);
            }
        }
        if let Some(duration) = clip.duration_sec {
            if duration.is_finite() && duration > 0.0 {
                return Some(duration);
            }
        }
        None
    }

    /// Split multiple clips at the same position.
    /// - Left halves keep the original group_id.
    /// - Right halves get a new group_id (per original group).
    /// - Unsplit clips in affected groups are assigned to left or right side by position.
    /// - Groups with fewer than 2 members after reassignment are dissolved.
    #[allow(dead_code)]
    pub fn split_clips_at(&mut self, clip_ids: &[String], split_sec: f64) {
        self.split_clips_at_with_options(clip_ids, split_sec, None);
    }

    /// 同 `split_clips_at`，但在每个分割上应用“分割过渡”设置。
    pub fn split_clips_at_with_transition(
        &mut self,
        clip_ids: &[String],
        split_sec: f64,
        opts: &SplitTransitionOptions,
    ) {
        self.split_clips_at_with_options(clip_ids, split_sec, Some(opts));
    }

    fn split_clips_at_with_options(
        &mut self,
        clip_ids: &[String],
        split_sec: f64,
        opts: Option<&SplitTransitionOptions>,
    ) {
        // 1. Collect affected group IDs from input clips
        let affected_groups: HashSet<Option<String>> = clip_ids
            .iter()
            .filter_map(|cid| {
                self.clips
                    .iter()
                    .find(|c| c.id == *cid)
                    .map(|c| c.group_id.clone())
            })
            .collect();

        // 2. Record clip IDs before split
        let before_ids: HashSet<String> = self.clips.iter().map(|c| c.id.clone()).collect();

        // 3. Split each input clip
        for cid in clip_ids {
            if let Some(opts) = opts {
                let _ = self.split_clip_with_transition(cid, split_sec, opts);
            } else {
                let _ = self.split_clip(cid, split_sec);
            }
        }

        // 4. Identify newly created clip IDs (right halves)
        let new_ids: HashSet<String> = self
            .clips
            .iter()
            .map(|c| c.id.clone())
            .filter(|id| !before_ids.contains(id))
            .collect();

        if new_ids.is_empty() {
            return;
        }

        // 5. 仅对**确实发生了分割**的组生成右组 UUID。
        //    输入 clip 的切割点可能被 clamp 在其范围外（返回 None、无新 clip），
        //    这类组不得发生右侧迁移 —— 否则整个组会被错误地搬进一个空降的新组。
        //    判定依据：该组内出现了本次新增的 clip（右半继承原组 id，见 split_clip）。
        let mut right_group_map: HashMap<String, Option<String>> = HashMap::new();
        for gid_opt in &affected_groups {
            let Some(ref gid) = gid_opt else { continue };
            let has_actual_split = self
                .clips
                .iter()
                .any(|clip| new_ids.contains(&clip.id) && clip.group_id.as_ref() == Some(gid));
            if has_actual_split {
                right_group_map.insert(gid.clone(), Some(new_id("group")));
            }
        }

        // 6. 右侧成员迁移（仅限实际发生分割的组）：新右半必然进入新组；
        //    未被切开但完全位于切割点右侧的同组成员也迁入新组；
        //    位于左侧的成员保留原组。
        for (gid, new_gid) in &right_group_map {
            let Some(ref migrated_gid) = new_gid else {
                continue;
            };
            for clip in self.clips.iter_mut() {
                if clip.group_id.as_ref() != Some(gid) {
                    continue;
                }
                // 新右半必然进入新组；未被切开但完全位于切割点右侧的成员同样迁入。
                if new_ids.contains(&clip.id) || clip.start_sec >= split_sec - 1e-6 {
                    clip.group_id = Some(migrated_gid.clone());
                }
            }
        }

        // 7. Dissolve single-member groups（仅限本次操作涉及的组）
        // 只溶解"本次分割新建"的单成员组；用户原有的编组即使因成员迁移到
        // 右组而暂时只剩 1 个成员也不得拆散。此外必须**只扫描本次涉及**的组：
        // 连续多次 split 时，前一次调用合法保留下来的单成员原组，
        // 不能被后一次调用的全局溶解误伤。
        let protected_original_groups: HashSet<&String> = affected_groups
            .iter()
            .filter_map(|gid_opt| gid_opt.as_ref())
            .collect();
        let mut involved_groups: HashSet<&String> = protected_original_groups.clone();
        for gid in right_group_map.values().flatten() {
            involved_groups.insert(gid);
        }
        let mut group_counts: HashMap<String, usize> = HashMap::new();
        for clip in &self.clips {
            if let Some(ref gid) = clip.group_id {
                *group_counts.entry(gid.clone()).or_default() += 1;
            }
        }

        for clip in &mut self.clips {
            if let Some(ref gid) = clip.group_id {
                if !involved_groups.contains(gid) {
                    continue;
                }
                if protected_original_groups.contains(gid) {
                    continue;
                }
                if group_counts.get(gid).copied().unwrap_or(0) < 2 {
                    clip.group_id = None;
                }
            }
        }
    }

    pub fn glue_clips(&mut self, clip_ids: &[String]) {
        if clip_ids.len() < 2 {
            return;
        }
        let mut selected: Vec<Clip> = self
            .clips
            .iter()
            .filter(|c| clip_ids.contains(&c.id))
            .cloned()
            .collect();
        if selected.len() < 2 {
            return;
        }
        let track_id = selected[0].track_id.clone();
        if selected.iter().any(|c| c.track_id != track_id) {
            return;
        }
        selected.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));

        // 针对音高参考块的胶合流程
        let all_pitch = selected.iter().all(|c| c.midi_note_data.is_some());
        if all_pitch {
            let original_ids: Vec<String> = clip_ids.to_vec();
            self.glue_pitch_clips(&selected, &original_ids);
            return;
        }

        let Some(first) = selected.first() else {
            return;
        };
        let start = first.start_sec;
        let end = selected
            .iter()
            .map(|c| c.start_sec + c.length_sec)
            .fold(start, f64::max);

        self.ensure_project_end_sec(end);

        let mut glued = first.clone();
        glued.id = new_id("clip");
        glued.group_id = None; // Glued clip starts a new identity, not part of any group
        glued.name = "Glued".to_string();
        glued.start_sec = start;
        glued.length_sec = (end - start).max(0.01);
        // Loop（循环源）：无论烘焙成功与否，胶合产物都不应再按首个片段的
        // 窗口回绕 —— 失败兜底路径会保留 first 的源窗口铺满合并跨度，
        // 若保留 loop=true 会把首段内容循环成整段（凭空捏造内容）。
        // 成功路径下方也会重置；此处提前统一关闭。
        glued.loop_enabled = false;

        // Render selected clips into one baked audio file so glue includes all selected data,
        // not only the first clip's source payload.
        let selected_id_set: HashSet<String> = selected.iter().map(|c| c.id.clone()).collect();

        let temp_glue_path = crate::temp_manager::hifishifter_temp_dir()
            .map(|dir| dir.join(format!("glue_{}.wav", Uuid::new_v4().simple())));

        if let Ok(glue_path) = temp_glue_path {
            let mut render_timeline = self.clone();
            render_timeline
                .clips
                .retain(|c| selected_id_set.contains(&c.id));

            for tr in &mut render_timeline.tracks {
                if tr.id == track_id {
                    tr.muted = false;
                    tr.solo = false;
                    tr.volume = 1.0;
                } else {
                    tr.muted = true;
                    tr.solo = false;
                    tr.volume = 0.0;
                }
            }

            let render_result = crate::mixdown::render_mixdown_wav(
                &render_timeline,
                &glue_path,
                crate::mixdown::MixdownOptions {
                    sample_rate: 44_100,
                    start_sec: start,
                    end_sec: Some(end),
                    stretch: crate::time_stretch::resolved_external_stretch_algorithm(),
                    apply_pitch_edit: true,
                    export_format: crate::mixdown::ExportFormat::Wav32f,
                    quality_preset: crate::mixdown::QualityPreset::Export,
                    cancel_flag: None,
                },
            );

            if render_result.is_ok() {
                let info = try_read_wav_info(&glue_path, 4096);
                let rendered_duration_sec = info
                    .as_ref()
                    .map(|v| v.duration_sec)
                    .unwrap_or(glued.length_sec);

                glued.source_path = Some(glue_path.to_string_lossy().to_string());
                glued.duration_sec = Some(rendered_duration_sec);
                glued.duration_frames = info.as_ref().map(|v| v.total_frames);
                glued.source_sample_rate = info.as_ref().map(|v| v.sample_rate);
                glued.waveform_preview = info.map(|v| v.waveform_preview);
                glued.source_start_sec = 0.0;
                glued.source_end_sec = rendered_duration_sec;
                glued.playback_rate = 1.0;
                glued.reversed = false;
                // 胶合产物是完整烘焙的独立源文件（覆盖原循环内容），
                // Loop 属性不再有意义，重置为关闭。
                glued.loop_enabled = false;
                glued.gain = 1.0;
                glued.muted = false;
                glued.fade_in_sec = 0.0;
                glued.fade_out_sec = 0.0;
                glued.fade_in_shape = 0.0;
                glued.fade_out_shape = 0.0;
                glued.fade_in_dir = 0.0;
                glued.fade_out_dir = 0.0;
                glued.extra_curves = None;
                glued.extra_params = None;
                glued.pitch_range = Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                });
            }
        }

        // 胶合产物是"烘焙出的单 Take 新 Clip"（对齐 REAPER glue 语义）：
        // 清空源 takes 集合，由当前投影重建唯一 active Take。若保留继承来的
        // inactive takes，它们会带着旧源文件的旧窗口套在合并后的新几何上，
        // 用户切一下 take 就会听到与胶合结果完全错位的原始内容。
        glued.takes.clear();
        glued.active_take_id = None;
        glued.sync_take_from_flat();
        self.clips.retain(|c| !clip_ids.contains(&c.id));
        self.clips.push(glued.clone());
        self.selected_clip_id = Some(glued.id);
    }

    /// 将选中的音频块编入同一组。
    pub fn group_clips(&mut self, clip_ids: &[String]) {
        if clip_ids.len() < 2 {
            return;
        }
        let clip_id_set: HashSet<&str> = clip_ids.iter().map(|s| s.as_str()).collect();
        let group_id = Uuid::new_v4().to_string();
        for c in &mut self.clips {
            if clip_id_set.contains(c.id.as_str()) {
                c.group_id = Some(group_id.clone());
            }
        }
    }

    /// 将选中音频块从其所属组中移除（解组）。
    /// 仅移除指定音频块的组关系，若某组剩余成员 ≤1 则自动解散该组。
    pub fn ungroup_clips(&mut self, clip_ids: &[String]) {
        let clip_id_set: HashSet<&str> = clip_ids.iter().map(|s| s.as_str()).collect();

        // 收集受影响的组 ID
        let affected_groups: HashSet<String> = self
            .clips
            .iter()
            .filter(|c| clip_id_set.contains(c.id.as_str()) && c.group_id.is_some())
            .filter_map(|c| c.group_id.clone())
            .collect();

        if affected_groups.is_empty() {
            return;
        }

        // 仅移除指定音频块的 group_id
        for c in &mut self.clips {
            if clip_id_set.contains(c.id.as_str()) {
                c.group_id = None;
            }
        }

        // 自动解散成员数 ≤1 的组
        for gid in &affected_groups {
            let count = self
                .clips
                .iter()
                .filter(|c| c.group_id.as_deref() == Some(gid.as_str()))
                .count();
            if count <= 1 {
                for c in &mut self.clips {
                    if c.group_id.as_deref() == Some(gid.as_str()) {
                        c.group_id = None;
                    }
                }
                self.disabled_group_ids.remove(gid);
            }
        }
    }

    /// 切换编组的禁用状态。返回新的状态（true = 已禁用）。
    pub fn toggle_group_disabled(&mut self, group_id: &str) -> bool {
        if self.disabled_group_ids.contains(group_id) {
            self.disabled_group_ids.remove(group_id);
            false
        } else {
            self.disabled_group_ids.insert(group_id.to_string());
            true
        }
    }

    /// 将指定的常规音频块转换为音高参考块。
    /// 获取每个音频块的原始音高数据，转换为 midi_note_data，
    /// 并清除 source_path 使其成为纯音高参考块。
    pub fn convert_clips_to_pitch_reference(&mut self, clip_ids: &[String]) {
        let fp_ms = default_frame_period_ms();
        let fp_sec = fp_ms / 1000.0;

        for clip_id in clip_ids {
            let clip = match self.clips.iter().find(|c| c.id == *clip_id) {
                Some(c) => c,
                None => continue,
            };

            // 跳过已经是音高块的 clip
            if clip.midi_note_data.is_some() {
                continue;
            }

            // 获取原始音高数据
            let root_track_id = match self.resolve_root_track_id(&clip.track_id) {
                Some(id) => id,
                None => continue,
            };

            let pitch_midi: Vec<f32> =
                match crate::pitch_clip::compute_clip_pitch_midi(self, clip, &root_track_id, fp_ms)
                {
                    Some(curve) if !curve.is_empty() => curve,
                    _ => continue,
                };

            // 将 pitch 曲线转换为 midiNoteData（合并相邻的相同音符）
            let midi_notes = Self::pitch_curve_to_midi_notes(&pitch_midi, fp_sec, clip.length_sec);

            // 更新 clip
            if let Some(clip) = self.clips.iter_mut().find(|c| c.id == *clip_id) {
                clip.midi_note_data = Some(midi_notes);
                clip.midi_fill_gaps = true;
                clip.source_path = None;
                clip.source_path_relative = None;
                clip.duration_sec = None;
                clip.duration_frames = None;
                clip.waveform_preview = None;
                clip.color = "cyan".to_string();
                clip.pitch_range = Some(PitchRange {
                    min: 0.0,
                    max: 127.0,
                });
                // 音频 → MIDI 的内容替换写在投影上；必须写回 Take 权威数据，
                // 否则 Take 里残留的旧音频 source_path 会在克隆/切换时"复活"。
                clip.sync_take_from_flat();
                // 转换产物是"由当前可听内容派生的单 Take 音高参考块"
                // （与 glue 的单 Take 语义对齐）：清空源 takes 集合后由投影
                // 重建唯一 active Take。若保留继承来的 inactive 音频 take，
                // 用户切一下 take 就会把旧音频内容整体物化回来，转换静默丢失。
                clip.takes.clear();
                clip.active_take_id = None;
                clip.sync_take_from_flat();
            }
        }
    }

    /// 将所选音高参考块的 midi_note_data 更新为轨道上对应范围内的 pitch_edit 曲线。
    /// 对未编辑区域（pitch_edit 中为 0 的帧）回退到 clip 原有的 midi_note_data。
    /// 正确处理 stretch（playback_rate）和倒放（reversed）的时间映射。
    pub fn update_pitch_reference_from_track_params(&mut self, clip_ids: &[String]) {
        let fp = self.frame_period_ms();
        let fp_sec = fp / 1000.0;

        // 先收集每个 clip 的必要信息，避免同时持有不可变借用和可变借用
        struct ClipMeta {
            clip_id: String,
            root: String,
            start_sec: f64,
            length_sec: f64,
            playback_rate: f32,
            reversed: bool,
            /// Loop（循环源）回绕描述：(周期D, 正放锚点, 倒放锚点末端)；None = 非 Loop。
            loop_cycle: Option<(f64, f64, f64)>,
            src_start: f64,
            src_end: f64,
            original_midi: Option<Vec<MidiNoteEvent>>,
        }

        let clip_infos: Vec<ClipMeta> = clip_ids
            .iter()
            .filter_map(|clip_id| {
                let clip = self.clips.iter().find(|c| c.id == *clip_id)?;
                if clip.midi_note_data.is_none() {
                    return None;
                }
                let root = self.resolve_root_track_id(&clip.track_id)?;
                let src_end = if clip.source_end_sec > 0.0 {
                    clip.source_end_sec
                } else {
                    clip.length_sec
                };
                // 消费窗口（非 Loop 倒放锚定 se：win=[se−len·r, se]，可为负，
                // 域外为静音）—— fallback 曲线与 remap 写回共用同一坐标系。
                let src_start = if !clip.loop_enabled && clip.reversed {
                    let r = if clip.playback_rate.is_finite() && clip.playback_rate > 1e-6 {
                        clip.playback_rate as f64
                    } else {
                        1.0
                    };
                    src_end - clip.length_sec.max(0.0) * r
                } else {
                    clip.source_start_sec
                };
                // 倒放锚点 clamp 规则与 place_note_occurrence_in_loop 一致：
                // 周期来自媒体时长时 clamp 到媒体时长；周期退化为窗口跨度时
                // 保持原始 source_end（否则 slip 窗口的倒放相位被错误平移）。
                let clip_media_total = clip_source_media_duration_sec(clip).filter(|d| *d > 1e-9);
                let loop_cycle = clip_loop_cycle_span_sec(clip).map(|cycle| {
                    (
                        cycle,
                        clip.source_start_sec,
                        match clip_media_total {
                            Some(d) => clip.source_end_sec.min(d),
                            None => clip.source_end_sec,
                        }
                        .max(0.0),
                    )
                });
                Some(ClipMeta {
                    clip_id: clip_id.clone(),
                    root,
                    start_sec: clip.start_sec,
                    length_sec: clip.length_sec,
                    playback_rate: clip.playback_rate,
                    reversed: clip.reversed,
                    loop_cycle,
                    src_start,
                    src_end,
                    original_midi: clip.midi_note_data.clone(),
                })
            })
            .collect();

        for info in &clip_infos {
            // Step 1: 从 clip 原有的 midi_note_data 构建回退音高曲线
            let fallback_curve = Self::build_fallback_pitch_from_midi(
                info.length_sec,
                fp,
                info.original_midi.as_deref().unwrap_or(&[]),
                info.playback_rate,
                info.reversed,
                info.src_start,
                info.src_end,
                info.loop_cycle,
            );

            // Step 2: 同步 params 并读取 pitch_edit
            self.ensure_params_for_root(&info.root);

            let entry = match self.params_by_root_track.get(&info.root) {
                Some(e) => e,
                None => continue,
            };

            let (start_frame, end_frame) =
                self.clip_frame_bounds(info.start_sec, info.length_sec, fp);

            let extracted: Vec<f32> = entry
                .pitch_edit
                .get(start_frame..end_frame)
                .unwrap_or(&[])
                .to_vec();

            if extracted.is_empty() {
                continue;
            }

            // Step 3: 合并 pitch_edit 与回退曲线 ——
            // 帧中 pitch_edit 值 <= 0 表示未被编辑或无效，回退到原有 midi 数据
            let merged: Vec<f32> = extracted
                .iter()
                .enumerate()
                .map(|(i, &e)| {
                    if e > 0.0 {
                        e
                    } else {
                        fallback_curve.get(i).copied().unwrap_or(0.0)
                    }
                })
                .collect();

            // Step 4: 转换为 MIDI 音符事件
            let midi_notes = Self::pitch_curve_to_midi_notes(&merged, fp_sec, info.length_sec);

            // Step 5: 根据 stretch / reverse / loop 重映射音符时间
            let remapped = Self::remap_midi_note_times(
                midi_notes,
                info.length_sec,
                info.src_start,
                info.src_end,
                info.playback_rate,
                info.reversed,
                info.loop_cycle,
            );

            // Step 6: 写回 clip，同时重新计算 pitch_range
            if let Some(clip) = self.clips.iter_mut().find(|c| c.id == info.clip_id) {
                let min_note = remapped.iter().fold(127.0f32, |m, n| m.min(n.note));
                let max_note = remapped.iter().fold(0.0f32, |m, n| m.max(n.note));
                let padding = 2.0f32;
                clip.pitch_range = Some(PitchRange {
                    min: (min_note - padding).max(0.0),
                    max: (max_note + padding).min(127.0),
                });
                clip.midi_note_data = Some(remapped);
                // midi_note_data 是 active-take 投影字段，写回 Take 权威数据。
                clip.sync_take_from_flat();
            }
        }

        // Step 7: 为每个受影响的 root track 立即重组 pitch_orig / pitch_edit
        let roots: std::collections::HashSet<&str> =
            clip_infos.iter().map(|info| info.root.as_str()).collect();
        for root in &roots {
            let (curve, _all_cache_hit, has_pitch_adjustment) =
                match crate::pitch_analysis::schedule::assemble_pitch_orig_from_cache(self, root) {
                    Some(v) => v,
                    None => continue,
                };

            self.ensure_params_for_root(root);
            let key = crate::pitch_analysis::build_root_pitch_key(self, root);

            if let Some(entry) = self.params_by_root_track.get_mut(*root) {
                entry.pitch_orig = curve;
                entry.pitch_orig_key = Some(key);
                entry.has_pitch_adjustment_active = has_pitch_adjustment;
                if !entry.pitch_edit_user_modified {
                    entry.pitch_edit.clone_from(&entry.pitch_orig);
                }
            }
        }
    }

    /// 从 midi_note_data 构建一段 clip 时长内的回退音高曲线（单位：MIDI note number）。
    /// 时间映射与 `assemble_pitch_orig_from_cache` 中的 MIDI clip 路径保持一致。
    /// Loop（循环源）启用时，音符按媒体时长 D 的锚点回绕重复铺满整个 clip 长度
    /// （与音频渲染的 floor_mod 映射一致；纯 MIDI clip 无媒体 → D 退化为窗口跨度）。
    /// `loop_cycle = Some((周期D, 正放锚点, 倒放锚点末端))`；None 为非循环路径。
    fn build_fallback_pitch_from_midi(
        length_sec: f64,
        fp: f64,
        midi_notes: &[MidiNoteEvent],
        playback_rate: f32,
        reversed: bool,
        src_start: f64,
        src_end: f64,
        loop_cycle: Option<(f64, f64, f64)>,
    ) -> Vec<f32> {
        let fp_sec = fp / 1000.0;
        let total_frames = ((length_sec.max(0.0) * 1000.0) / fp).ceil().max(1.0) as usize;
        let mut curve = vec![0.0f32; total_frames];
        let pr = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate as f64
        } else {
            1.0
        };
        let src_total = src_end - src_start;

        for note in midi_notes {
            let note_value = note.note as f32;

            // Loop（循环源）：媒体时长锚点回绕放置（不能用窗口比较过滤可见性 ——
            // split 的环绕窗口 start > end 会把音符全部误判为越界）。
            if let Some((cycle_src_sec, fwd_anchor, rev_anchor_end)) = loop_cycle {
                if let Some(placement) = crate::state::place_note_occurrence_frames(
                    reversed,
                    playback_rate as f64,
                    fp,
                    fwd_anchor,
                    rev_anchor_end,
                    cycle_src_sec,
                    note.start_sec,
                    note.end_sec,
                ) {
                    let mut cycle_offset = 0usize;
                    while cycle_offset < total_frames {
                        let write_start = cycle_offset + placement.first_start_frame;
                        let write_end =
                            (cycle_offset + placement.first_start_frame + placement.len_frames)
                                .min(total_frames);
                        if write_start >= write_end {
                            break;
                        }
                        for f in write_start..write_end {
                            if note_value > curve[f] || curve[f] <= 0.0 {
                                curve[f] = note_value;
                            }
                        }
                        cycle_offset += placement.cycle_frames;
                    }
                    continue;
                }
            }

            let rel_start = (note.start_sec - src_start).max(0.0);
            let rel_end = (note.end_sec - src_start).min(src_total);
            if rel_end <= rel_start {
                continue;
            }

            let (eff_start, eff_end) = if reversed {
                (
                    (src_total - rel_end).max(0.0),
                    (src_total - rel_start).min(src_total),
                )
            } else {
                (rel_start, rel_end)
            };
            if eff_end <= eff_start {
                continue;
            }

            let frame_start = ((eff_start / pr) / fp_sec).round() as usize;
            let frame_end_raw = (eff_end / pr) / fp_sec;
            let frame_end_raw = frame_end_raw.round() as usize;
            {
                // 非 Loop：单次写入（Loop 已在上方 placement 分支处理）。
                let frame_end = frame_end_raw.min(total_frames);
                if frame_start < frame_end {
                    for f in frame_start..frame_end {
                        if note_value > curve[f] || curve[f] <= 0.0 {
                            curve[f] = note_value;
                        }
                    }
                }
            }
        }

        curve
    }

    /// 将 pitch_curve_to_midi_notes 输出的"提取时间"坐标重映射为 source-time 坐标，
    /// 以匹配 stretch（playback_rate）和倒放（reversed）参数。
    ///
    /// Loop（循环源）启用时（`loop_cycle = Some((周期D, 正放锚点, 倒放锚点末端))`），
    /// 提取时间按音频渲染的 floor_mod 锚点映射回**媒体文件域** `[0, D)`：
    ///   正放 s(proj) = floor_mod(source_start + proj·rate, D)
    ///   倒放 s(proj) = floor_mod(min(source_end, D) − proj·rate, D)
    /// 跨越回绕边界的音符会被拆分，保证存储坐标始终落在 [0, D) 内 —— 与
    /// assemble / emit / build_fallback 的音符放置算法使用同一坐标系。
    /// 非 Loop 保持既有窗口映射约定不变。
    fn remap_midi_note_times(
        notes: Vec<MidiNoteEvent>,
        length_sec: f64,
        src_start: f64,
        src_end: f64,
        playback_rate: f32,
        reversed: bool,
        loop_cycle: Option<(f64, f64, f64)>,
    ) -> Vec<MidiNoteEvent> {
        let mut out: Vec<MidiNoteEvent> = Vec::with_capacity(notes.len());
        // 速率净化提前：Loop 分支同样使用净化后的速率（非有限/过小的速率按
        // 1.0 处理），避免 NaN/负值把音符映射成垃圾坐标后被静默丢弃。
        let pr = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate as f64
        } else {
            1.0
        };

        for note in notes {
            let proj_start = note.start_sec; // 当前为提取时间（相对 clip 起点的项目时间）
            let proj_end = note.end_sec;

            if let Some((cycle, fwd_anchor, rev_anchor_end)) = loop_cycle {
                if !reversed {
                    // 正放：v = 锚点 + 消费量，直接对 D 取模得文件域坐标。
                    let v_start = proj_start * pr + fwd_anchor;
                    let v_end = proj_end * pr + fwd_anchor;
                    for (s, e) in split_range_into_periods(v_start, v_end, cycle) {
                        out.push(MidiNoteEvent {
                            start_sec: s,
                            end_sec: e,
                            ..note.clone()
                        });
                    }
                } else {
                    // 倒放：w = 倒放锚点 − 消费量，对 D 取模得文件域坐标。
                    let w_lo = rev_anchor_end - proj_end * pr;
                    let w_hi = rev_anchor_end - proj_start * pr;
                    for (s, e) in split_range_into_periods(w_lo, w_hi, cycle) {
                        out.push(MidiNoteEvent {
                            start_sec: s,
                            end_sec: e,
                            ..note.clone()
                        });
                    }
                }
                continue;
            }

            let src_total = src_end - src_start;

            let (new_start, new_end) = if reversed {
                let eff_start = (length_sec - proj_end) * pr;
                let eff_end = (length_sec - proj_start) * pr;
                let src_note_start = (src_total - eff_end).max(0.0);
                let src_note_end = (src_total - eff_start).min(src_total);
                (src_note_start + src_start, src_note_end + src_start)
            } else {
                (proj_start * pr + src_start, proj_end * pr + src_start)
            };

            out.push(MidiNoteEvent {
                start_sec: new_start,
                end_sec: new_end,
                ..note
            });
        }

        out
    }

    /// 将 pitch 曲线（Vec<f32> of MIDI note numbers）转换为 MidiNoteEvent 列表。
    /// 使用原始的浮点音高值，不进行半音量化。
    /// 相邻帧中音高差异极小时合并为一个事件。
    fn pitch_curve_to_midi_notes(
        curve: &[f32],
        frame_period_sec: f64,
        _total_length_sec: f64,
    ) -> Vec<MidiNoteEvent> {
        if curve.is_empty() {
            return vec![];
        }

        let mut notes: Vec<MidiNoteEvent> = Vec::new();
        let mut seg_start_frame: usize = 0;
        let mut current_note: f32 = curve[0];

        for i in 1..curve.len() {
            let note: f32 = curve[i];
            // 仅当音高差异大于 0.001 半音时才分段
            if (note - current_note).abs() > 0.001 {
                let start_sec = seg_start_frame as f64 * frame_period_sec;
                let end_sec = i as f64 * frame_period_sec;
                notes.push(MidiNoteEvent {
                    start_sec,
                    end_sec,
                    note: current_note,
                    velocity: 100,
                    channel: 0,
                });
                seg_start_frame = i;
                current_note = note;
            }
        }

        // 最后一段
        let end_sec = curve.len() as f64 * frame_period_sec;
        notes.push(MidiNoteEvent {
            start_sec: seg_start_frame as f64 * frame_period_sec,
            end_sec,
            note: current_note,
            velocity: 100,
            channel: 0,
        });

        notes
    }

    /// 胶合音高参考块：合并多个同轨道音高参考块的 midi_note_data，
    /// 用空音高填充间隙，生成新的音高参考块。
    fn glue_pitch_clips(&mut self, selected: &[Clip], original_clip_ids: &[String]) {
        let start = selected.first().map(|c| c.start_sec).unwrap_or(0.0);
        let end = selected
            .iter()
            .map(|c| c.start_sec + c.length_sec)
            .fold(start, f64::max);
        let length = (end - start).max(0.01);

        self.ensure_project_end_sec(end);

        // 合并所有 midiNoteData，按时间偏移
        let mut merged_notes: Vec<MidiNoteEvent> = Vec::new();
        for clip in selected {
            let offset = clip.start_sec - start;
            if let Some(ref notes) = clip.midi_note_data {
                for note in notes {
                    merged_notes.push(MidiNoteEvent {
                        start_sec: note.start_sec + offset,
                        end_sec: note.end_sec + offset,
                        note: note.note,
                        velocity: note.velocity,
                        channel: note.channel,
                    });
                }
            }
        }

        // 按 start_sec 排序
        merged_notes.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 填补间隙：如果两个连续音符之间存在空隙，插入 rest 音符
        let mut filled_notes: Vec<MidiNoteEvent> = Vec::new();
        let mut cursor = 0.0f64;
        for note in &merged_notes {
            if note.start_sec > cursor + 0.001 {
                // 间隙 > 1ms，插入 rest 音符
                filled_notes.push(MidiNoteEvent {
                    start_sec: cursor,
                    end_sec: note.start_sec,
                    note: 0.0, // rest
                    velocity: 0,
                    channel: 0,
                });
            }
            filled_notes.push(note.clone());
            cursor = note.end_sec;
        }

        // 如果末尾有剩余空间，也插入 rest
        if cursor < length - 0.001 {
            filled_notes.push(MidiNoteEvent {
                start_sec: cursor,
                end_sec: length,
                note: 0.0,
                velocity: 0,
                channel: 0,
            });
        }

        // 创建新的胶合音高参考块
        let mut glued = selected[0].clone();
        glued.id = new_id("clip");
        glued.name = "Glued".to_string();
        glued.start_sec = start;
        glued.length_sec = length;
        glued.midi_note_data = Some(filled_notes);
        glued.midi_fill_gaps = true;
        glued.color = "cyan".to_string();
        glued.source_path = None;
        glued.source_path_relative = None;
        // 重置 source trim 和播放参数，使 midiNoteData 完整映射到 clip 时间轴
        glued.source_start_sec = 0.0;
        glued.source_end_sec = length;
        glued.playback_rate = 1.0;
        glued.reversed = false;
        // 音符已铺满整个胶合区间（含 rest），Loop 回绕不再有意义。
        glued.loop_enabled = false;
        glued.duration_sec = None;
        glued.duration_frames = None;
        glued.waveform_preview = None;
        glued.pitch_range = Some(PitchRange {
            min: 0.0,
            max: 127.0,
        });

        // 与音频 glue 一致：胶合产物是单 Take 新 Clip，清空继承的 takes 集合。
        glued.takes.clear();
        glued.active_take_id = None;
        glued.sync_take_from_flat();
        self.clips.retain(|c| !original_clip_ids.contains(&c.id));
        self.clips.push(glued.clone());
        self.selected_clip_id = Some(glued.id);
    }

    pub fn select_clip(&mut self, clip_id: Option<String>) {
        match clip_id {
            None => self.selected_clip_id = None,
            Some(id) => {
                if let Some(track_id) = self
                    .clips
                    .iter()
                    .find(|c| c.id == id)
                    .map(|c| c.track_id.clone())
                {
                    self.selected_clip_id = Some(id);
                    self.selected_track_id = Some(track_id);
                }
            }
        }
    }

    pub fn import_audio_item(
        &mut self,
        audio_path: &str,
        track_id: Option<String>,
        start_sec: Option<f64>,
    ) {
        let name = Path::new(audio_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Audio")
            .to_string();

        let mut duration_sec: Option<f64> = None;
        let mut duration_frames: Option<u64> = None;
        let mut source_sample_rate: Option<u32> = None;
        let mut waveform_preview: Option<Vec<f32>> = None;

        // 视频文件导入只做 O(1) header 探测，不在导入线程里全量解码；
        // 波形由前端按需异步请求峰值缓存生成。
        let is_video_source = crate::media::is_video_extension(Path::new(audio_path));
        let import_info = if is_video_source {
            crate::audio_utils::try_read_audio_header_only(Path::new(audio_path))
        } else {
            crate::audio_utils::try_read_wav_info(Path::new(audio_path), 4096)
        };

        match import_info {
            Some(info) => {
                if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                    let mut max_amp = 0.0f32;
                    for &v in info.waveform_preview.iter() {
                        if v.is_finite() {
                            max_amp = max_amp.max(v.abs());
                        }
                    }
                    let head: Vec<String> = info
                        .waveform_preview
                        .iter()
                        .take(8)
                        .map(|v| format!("{:.4}", v))
                        .collect();
                    log::warn!(
                        "import_audio_item: audio_info ok: total_frames={}, sample_rate={}, duration_sec={:.6}, preview_len={}, preview_max={:.4}, preview_head=[{}]",
                        info.total_frames,
                        info.sample_rate,
                        info.duration_sec,
                        info.waveform_preview.len(),
                        max_amp,
                        head.join(", ")
                    );
                }
                duration_sec = Some(info.duration_sec);
                duration_frames = Some(info.total_frames);
                source_sample_rate = Some(info.sample_rate);
                waveform_preview = Some(info.waveform_preview);
            }
            None => {
                if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                    let exists = Path::new(audio_path).exists();
                    log::error!(
                        "import_audio_item: audio_info FAILED: path_exists={} path={}",
                        exists, audio_path
                    );
                }
            }
        }

        // 使用精确的frame计算length_sec（直接用秒，不依赖BPM）
        let computed_length_sec =
            if let (Some(frames), Some(sr)) = (duration_frames, source_sample_rate) {
                frames as f64 / sr as f64
            } else {
                duration_sec.unwrap_or(4.0)
            };

        let clip_id = self.add_clip(
            track_id,
            Some(name),
            start_sec,
            Some(computed_length_sec),
            Some(audio_path.to_string()),
        );

        // DEBUG: 打印导入clip时的关键参数
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            log::warn!(
                "import_audio_item: clip created: clip_id={}, duration_frames={:?}, sample_rate={:?}, computed_length_sec={:.6}",
                &clip_id[..8.min(clip_id.len())],
                duration_frames,
                source_sample_rate,
                computed_length_sec
            );
        }

        if let Some(c) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            c.duration_sec = duration_sec;
            c.duration_frames = duration_frames;
            c.source_sample_rate = source_sample_rate;
            c.waveform_preview = waveform_preview;
            // 文件元数据 + 内容指纹已由 add_clip → populate_clip_file_metadata 填充，
            // 此处只需确保 waveform_preview 等音频信息正确落盘即可。
            // 投影字段被直接覆盖，必须写回 Take 权威数据 —— 否则 take 里的
            // 旧元数据会在下次 normalize/sync 时回流覆盖这里的值（例如视频
            // header-only 探测与 add_clip 内完整解码结果存在差异时）。
            c.sync_take_from_flat();
        }
    }

    /// 将目标 Clip 的源媒体替换为新文件。
    ///
    /// 语义边界：只替换 **active take** 的媒体（经投影写入后 sync 回该 Take）；
    /// inactive takes 保留各自的旧源文件。这与"切换 take 对比不同素材"的
    /// 多 take 用法一致 —— 整体替换全部 take 属于破坏性操作，如需提供应
    /// 作为独立命令并向前端明示。
    pub fn replace_clip_sources(
        &mut self,
        clip_ids: &[String],
        new_source_path: &str,
        replace_same_source: bool,
    ) -> usize {
        if clip_ids.is_empty() || new_source_path.trim().is_empty() {
            return 0;
        }

        let target_id_set: HashSet<&str> = clip_ids.iter().map(|id| id.as_str()).collect();
        let mut old_source_set: HashSet<String> = HashSet::new();
        for clip in &self.clips {
            if target_id_set.contains(clip.id.as_str()) {
                if let Some(path) = clip.source_path.as_ref() {
                    old_source_set.insert(path.clone());
                }
            }
        }

        let info = try_read_wav_info(Path::new(new_source_path), 4096);
        let duration_sec = info.as_ref().map(|v| v.duration_sec);
        let duration_frames = info.as_ref().map(|v| v.total_frames);
        let source_sample_rate = info.as_ref().map(|v| v.sample_rate);
        let waveform_preview = info.map(|v| v.waveform_preview);

        // 记录新源文件的元数据 + 内容指纹
        let new_meta = std::fs::metadata(new_source_path).ok();
        let new_mtime = new_meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        let new_size = new_meta.as_ref().map(|m| m.len());
        let new_fp = crate::audio_utils::compute_file_fingerprint(Path::new(new_source_path));
        if new_fp.is_none() {
            log::warn!(
                "[replace_clip_sources] WARNING: compute_file_fingerprint failed for path={} (file may be locked), keeping old fingerprint",
                new_source_path
            );
        }

        let mut changed = 0usize;
        for clip in &mut self.clips {
            let direct_match = target_id_set.contains(clip.id.as_str());
            let same_source_match = replace_same_source
                && clip
                    .source_path
                    .as_ref()
                    .map(|p| old_source_set.contains(p))
                    .unwrap_or(false);

            if !direct_match && !same_source_match {
                continue;
            }

            clip.source_path = Some(new_source_path.to_string());
            clip.source_path_relative = None;
            clip.duration_sec = duration_sec;
            clip.duration_frames = duration_frames;
            clip.source_sample_rate = source_sample_rate;
            clip.source_file_mtime = new_mtime;
            clip.source_file_size = new_size;
            // 仅在新指纹成功计算时才更新，避免因文件锁等原因丢失指纹数据
            if let Some(fp) = new_fp {
                clip.source_file_fingerprint = Some(fp);
            }
            clip.waveform_preview = waveform_preview.clone();
            clip.sync_take_from_flat();
            changed += 1;
        }

        changed
    }

    /// 检查所有有 source_path 的 clip，使用分层策略检测外部文件变更。
    ///
    /// 第 1 层：存在性检查。
    /// 第 2 层：元数据比对（文件大小 + 修改时间），均未变 → 跳过。
    /// 第 3 层：内容指纹验证（头 64KB + 尾 64KB FNV-1a），
    ///          若指纹一致 → 仅元数据变化（如云同步/touch），静默更新；
    ///          若指纹不一致 → 内容确实被修改 → 报告 "modified"。
    ///
    /// 对 GB 级音频文件也只读取最多 128KB，IO 开销可忽略。
    pub fn check_source_files_changed(&self) -> crate::models::CheckSourceFilesChangedPayload {
        let mut changed: Vec<crate::models::SourceFileChangePayload> = Vec::new();
        let mut reported_paths: HashSet<String> = HashSet::new();

        for clip in &self.clips {
            for take in &clip.takes {
                let source_path = match take.source_path.as_ref() {
                    Some(p) => p,
                    None => continue,
                };
                if reported_paths.contains(source_path) {
                    continue;
                }

                let path = std::path::Path::new(source_path);

                // ── 第 1 层：存在性检查 ──────────────────────────────────────
                if !path.exists() {
                    reported_paths.insert(source_path.to_string());
                    changed.push(crate::models::SourceFileChangePayload {
                        clip_id: clip.id.clone(),
                        clip_name: clip.name.clone(),
                        source_path: source_path.to_string(),
                        change: "deleted".to_string(),
                    });
                    continue;
                }

                // ── 第 2 层：元数据快速比对 ─────────────────────────────────
                let current_meta = std::fs::metadata(path).ok();
                let current_size = current_meta.as_ref().map(|m| m.len());
                let current_mtime = current_meta
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());

                let old_mtime = take.source_file_mtime;
                let old_size = take.source_file_size;
                let old_fp = take.source_file_fingerprint;

                // 若大小和 mtime 均与记录一致，通常可跳过。
                // 但工程文件可能保存了旧的源文件指纹：例如用户在关闭工程后手动替换了
                // 同名文件，重新打开工程时 mtime/size 会以“新文件”为基线刷新，因此
                // 元数据一致并不代表内容与工程保存时一致。只要存在指纹，就必须进入
                // 指纹验证层，用工程中保存的哈希重新判断内容是否发生变化。
                if current_size == old_size && current_mtime == old_mtime && old_fp.is_none() {
                    continue;
                }

                // 旧工程既无元数据也无指纹 → 跳过检测（无法判断是否变更）
                if old_mtime.is_none() && old_size.is_none() && old_fp.is_none() {
                    continue;
                }

                // ── 第 3 层：内容指纹验证 ────────────────────────────────────
                let current_fp = crate::audio_utils::compute_file_fingerprint(path);
                if current_fp.is_some() && current_fp == old_fp {
                    // 内容未变，仅元数据被修改（touch、云同步等）→ 静默更新记录，不打扰用户
                    // Note: 此处为只读引用，无法原地更新 clip 的元数据。
                    //       元数据将在下次 reload/replace 时自然更新。
                    continue;
                }

                // 内容确实发生变化
                reported_paths.insert(source_path.to_string());
                changed.push(crate::models::SourceFileChangePayload {
                    clip_id: clip.id.clone(),
                    clip_name: clip.name.clone(),
                    source_path: source_path.to_string(),
                    change: "modified".to_string(),
                });
            }
        }

        crate::models::CheckSourceFilesChangedPayload { changed }
    }
}

fn build_track_payload(tracks: &[Track]) -> Vec<TimelineTrack> {
    // Group by parent and keep stable ordering by `order`.
    let mut by_parent: HashMap<Option<String>, Vec<Track>> = HashMap::new();
    for t in tracks.iter().cloned() {
        by_parent.entry(t.parent_id.clone()).or_default().push(t);
    }
    for v in by_parent.values_mut() {
        v.sort_by_key(|t| t.order);
    }

    // Roots in order.
    let roots = by_parent.get(&None).cloned().unwrap_or_else(Vec::new);

    let mut out: Vec<TimelineTrack> = Vec::with_capacity(tracks.len());

    fn dfs(
        t: &Track,
        depth: u32,
        by_parent: &HashMap<Option<String>, Vec<Track>>,
        out: &mut Vec<TimelineTrack>,
    ) {
        fn algo_name(a: &PitchAnalysisAlgo) -> String {
            match a {
                PitchAnalysisAlgo::WorldDll => "world_dll".to_string(),
                PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx".to_string(),
                PitchAnalysisAlgo::VocalShifterVslib => "vslib".to_string(),
                PitchAnalysisAlgo::None => "none".to_string(),
                PitchAnalysisAlgo::Unknown => "unknown".to_string(),
            }
        }

        let children = by_parent
            .get(&Some(t.id.clone()))
            .cloned()
            .unwrap_or_else(Vec::new);
        let child_ids = children.iter().map(|c| c.id.clone()).collect::<Vec<_>>();

        out.push(TimelineTrack {
            id: t.id.clone(),
            name: t.name.clone(),
            parent_id: t.parent_id.clone(),
            depth: Some(depth),
            child_track_ids: Some(child_ids),
            muted: t.muted,
            solo: t.solo,
            volume: t.volume,
            compose_enabled: t.compose_enabled,
            pitch_analysis_algo: algo_name(&t.pitch_analysis_algo),
            color: t.color.clone(),
        });

        for c in children {
            dfs(&c, depth + 1, by_parent, out);
        }
    }

    for r in roots {
        dfs(&r, 0, &by_parent, &mut out);
    }

    // Any orphans (missing parent) appended.
    if out.len() != tracks.len() {
        let mut seen: BTreeMap<String, bool> = BTreeMap::new();
        for t in &out {
            seen.insert(t.id.clone(), true);
        }
        for t in tracks {
            if !seen.contains_key(&t.id) {
                out.push(TimelineTrack {
                    id: t.id.clone(),
                    name: t.name.clone(),
                    parent_id: t.parent_id.clone(),
                    depth: Some(0),
                    child_track_ids: Some(vec![]),
                    muted: t.muted,
                    solo: t.solo,
                    volume: t.volume,
                    compose_enabled: t.compose_enabled,
                    pitch_analysis_algo: match t.pitch_analysis_algo {
                        PitchAnalysisAlgo::WorldDll => "world_dll".to_string(),
                        PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx".to_string(),
                        PitchAnalysisAlgo::VocalShifterVslib => "vslib".to_string(),
                        PitchAnalysisAlgo::None => "none".to_string(),
                        PitchAnalysisAlgo::Unknown => "unknown".to_string(),
                    },
                    color: t.color.clone(),
                });
            }
        }
    }

    out
}

impl AppState {
    pub fn runtime_info(&self) -> RuntimeInfoPayload {
        let rt = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        let pb = self.audio_engine.snapshot_state();

        // Report the execution provider the live vocoder session actually runs
        // on.  This used to be a compile-time constant, so the menu claimed
        // "GPU (CoreML)" even when every session had fallen back to CPU.
        let gpu_backend = crate::nsf_hifigan_onnx::active_backend_name();

        RuntimeInfoPayload {
            ok: true,
            device: rt.device.clone(),
            model_loaded: rt.model_loaded,
            audio_loaded: rt.audio_loaded,
            has_synthesized: rt.has_synthesized,
            is_playing: Some(pb.is_playing),
            playback_target: pb.target.clone(),
            timeline: None,
            gpu_backend: gpu_backend.to_string(),
        }
    }

    pub fn model_config_ok(&self) -> ModelConfigPayload {
        ModelConfigPayload {
            ok: true,
            config: ModelConfig {
                audio_sample_rate: 44100,
                audio_num_mel_bins: 128,
                hop_size: 512,
                fmin: 40.0,
                fmax: 16000.0,
            },
        }
    }
}
