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
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_curve: Option<String>,
    pub fade_out_curve: Option<String>,
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
    /// Clip 颜色、来源路径、时长/采样率等基础参数始终序列化，兼容旧版本读取；
    /// `source_path_relative` 为可推导的派生路径，仍按需省略。
    #[serde(default = "default_clip_color")]
    pub color: String,

    #[serde(default)]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path_relative: Option<String>,
    #[serde(default)]
    pub duration_sec: Option<f64>, // 兼容性保留
    #[serde(default)]
    pub duration_frames: Option<u64>, // 精确的frame总数
    #[serde(default)]
    pub source_sample_rate: Option<u32>, // 源文件采样率
    /// 文件导入时的 mtime（Unix 时间戳，秒），用于检测外部文件替换/删除。
    /// None 表示运行时从字节流导入（无磁盘文件）或尚未初始化。
    /// 仅在程序运行期间有效，不持久化到工程文件。
    #[serde(skip)]
    pub source_file_mtime: Option<u64>,
    /// 源文件大小（字节），与 mtime 一起作为第一层元数据比对。
    /// 仅在程序运行期间有效，不持久化到工程文件。
    #[serde(skip)]
    pub source_file_size: Option<u64>,
    /// 源文件内容指纹（头 64KB + 尾 64KB FNV-1a 64-bit）。
    ///
    /// 用于：
    /// 1. 元数据变化后的第二层内容确认；
    /// 2. 源文件缺失时按文件名搜索候选文件并进行哈希匹配。
    ///
    /// 该字段随工程文件持久化；打开工程时优先使用工程中保存的值，
    /// 即使源文件当前缺失，也能用于后续重新匹配。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file_fingerprint: Option<u64>,
    /// 波形预览与音高范围属于可重新生成的缓存，仅在保存时以 None/null 形式
    /// 落盘（保持字段存在以兼容旧版本读取，但不携带任何波形数据）。
    #[serde(default)]
    pub waveform_preview: Option<Vec<f32>>,
    #[serde(default)]
    pub pitch_range: Option<PitchRange>,

    /// 增益/静音/裁剪/播放速率/反向/淡入淡出/曲线类型等 clip 基础参数
    /// 始终序列化，避免依赖"缺省 = 默认值"的隐式规则。
    #[serde(default = "default_gain")]
    pub gain: f32,
    #[serde(default)]
    pub muted: bool,
    #[serde(alias = "trim_start_sec", default)]
    pub source_start_sec: f64,
    #[serde(alias = "trim_end_sec")]
    pub source_end_sec: f64,
    #[serde(default = "default_playback_rate")]
    pub playback_rate: f32,
    #[serde(default)]
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
    #[serde(default)]
    pub loop_enabled: bool,
    #[serde(default)]
    pub fade_in_sec: f64,
    #[serde(default)]
    pub fade_out_sec: f64,
    /// 淡入曲线类型（linear/sine/exponential/logarithmic/scurve），默认 sine
    #[serde(default = "default_fade_curve")]
    pub fade_in_curve: String,
    /// 淡出曲线类型（linear/sine/exponential/logarithmic/scurve），默认 sine
    #[serde(default = "default_fade_curve")]
    pub fade_out_curve: String,

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

    /// MIDI 音符数据（仅用于 MIDI clip，无音频源）。
    /// 音符时间相对于 clip 起点（0 = clip 起点）。
    /// 当 Some 时，source_path 应为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub midi_note_data: Option<Vec<MidiNoteEvent>>,

    /// 是否在 pitch_orig 组装时填补 MIDI 音符之间的空隙。
    #[serde(default, skip_serializing_if = "is_false")]
    pub midi_fill_gaps: bool,
}

impl Clip {
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
    pub reversed: Option<bool>,
    #[serde(default)]
    pub loop_enabled: Option<bool>,
    pub fade_in_sec: Option<f64>,
    pub fade_out_sec: Option<f64>,
    pub fade_in_curve: Option<String>,
    pub fade_out_curve: Option<String>,
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

    pub next_track_order: i32,

    #[serde(default, skip_serializing_if = "hash_set_string_is_empty")]
    pub disabled_group_ids: HashSet<String>,
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

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Track => "track",
            Self::All => "all",
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
        let entry = self.params_by_root_track.get(root_track_id)?;

        let pitch_edit = if entry.pitch_edit_user_modified {
            entry
                .pitch_edit
                .get(start_frame..end_frame)
                .unwrap_or(&[])
                .to_vec()
        } else {
            Vec::new()
        };
        let tension_edit = entry
            .tension_edit
            .get(start_frame..end_frame)
            .unwrap_or(&[])
            .to_vec();
        let extra_curves = entry
            .extra_curves
            .iter()
            .map(|(param, curve)| {
                (
                    param.clone(),
                    curve.get(start_frame..end_frame).unwrap_or(&[]).to_vec(),
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
            if let Some(track) = self.tracks.iter_mut().find(|track| track.id == root_track_id) {
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
                eprintln!(
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
            eprintln!("Warning: failed to save v2 peaks cache: {}", e);
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

    fn find_clip_start(timeline: &TimelineState, clip_id: &str) -> f64 {
        timeline
            .clips
            .iter()
            .find(|clip| clip.id == clip_id)
            .map(|clip| clip.start_sec)
            .unwrap_or(f64::NAN)
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
        let p = place_note_occurrence_frames(false, 1.0, fp, fwd_anchor, rev_anchor_end, cycle, 3.0, 4.5)
            .expect("valid placement");
        assert_eq!(p.first_start_frame, 100);
        assert_eq!(p.len_frames, 150);
        assert_eq!(p.cycle_frames, 400);

        // 倒放：(6−4.5) mod 4 = 1.5s（= 帧 150）—— 与窗口镜像语义等价。
        let p_rev =
            place_note_occurrence_frames(true, 1.0, fp, fwd_anchor, rev_anchor_end, cycle, 3.0, 4.5)
                .expect("valid placement");
        assert_eq!(p_rev.first_start_frame, 150);

        // 负锚点环绕：anchor=-1 对 D=4 → 首现于消费 1.0s（floor_mod(-1+u)）。
        let p_neg = place_note_occurrence_frames(false, 1.0, fp, -1.0, rev_anchor_end, cycle, -0.5, 0.5)
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
            .add_clip(Some(track_id), Some("M".into()), Some(0.0), Some(10.0), None)
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
        assert_eq!(p.first_start_frame, 0, "note ending at window end is heard first");
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
            .add_clip(Some(track_id), Some("L".into()), Some(0.0), Some(12.0), None)
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
            assert!((fwd.source_end_sec - 5.0).abs() < 1e-9, "forward end derives from start+len");
            assert!((fwd.source_start_sec - 2.0).abs() < 1e-9);

            let mut rev = tl.clips.iter().find(|c| c.id == rev_id).unwrap().clone();
            normalize_nonloop_source_window(&mut rev);
            assert!((rev.source_start_sec - (-4.5)).abs() < 1e-9, "reversed start derives from end−len");
            assert!((rev.source_end_sec - (-1.5)).abs() < 1e-9, "negative anchor preserved");

            // Loop：字段承载锚点相位，规范化不得触碰。
            let mut lp = rev.clone();
            lp.loop_enabled = true;
            lp.source_start_sec = -77.0;
            normalize_nonloop_source_window(&mut lp);
            assert!((lp.source_start_sec - (-77.0)).abs() < 1e-12);
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

        let a0 = timeline.add_clip(Some(track_a.clone()), Some("a0".into()), Some(0.0), Some(2.0), None);
        let a1 = timeline.add_clip(Some(track_a.clone()), Some("a1".into()), Some(2.0), Some(2.0), None);
        let a2 = timeline.add_clip(Some(track_a.clone()), Some("a2".into()), Some(4.0), Some(2.0), None);
        let b0 = timeline.add_clip(Some(track_b.clone()), Some("b0".into()), Some(2.0), Some(2.0), None);

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

        let a1 = timeline.add_clip(Some(track_a.clone()), Some("a1".into()), Some(2.0), Some(2.0), None);
        let a2 = timeline.add_clip(Some(track_a.clone()), Some("a2".into()), Some(4.0), Some(2.0), None);
        let b0 = timeline.add_clip(Some(track_b.clone()), Some("b0".into()), Some(2.0), Some(4.0), None);

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
        let _a0 = timeline.add_clip(Some(track.clone()), Some("a0".into()), Some(0.0), Some(2.0), None);
        let a1 = timeline.add_clip(Some(track.clone()), Some("a1".into()), Some(2.0), Some(2.0), None);
        let a2 = timeline.add_clip(Some(track.clone()), Some("a2".into()), Some(4.0), Some(2.0), None);

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
                    fade_in_sec: Some(0.15),
                    fade_out_sec: Some(0.25),
                    fade_in_curve: Some("sine".into()),
                    fade_out_curve: Some("logarithmic".into()),
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
                    fade_in_sec: Some(0.05),
                    fade_out_sec: Some(0.1),
                    fade_in_curve: Some("linear".into()),
                    fade_out_curve: Some("scurve".into()),
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
        assert_eq!(first.fade_in_curve, "sine");
        assert_eq!(first.fade_out_curve, "logarithmic");
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
                fade_in_curve: Some("linear".into()),
                fade_out_curve: Some("scurve".into()),
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
                fade_in_sec: Some(0.1),
                fade_out_sec: Some(0.2),
                fade_in_curve: Some("linear".into()),
                fade_out_curve: Some("scurve".into()),
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
        assert_eq!(pasted.fade_out_curve, "scurve");
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
        let c1 = tl.add_clip(Some(tid.clone()), Some("A".into()), Some(0.0), Some(2.0), None);
        let c2 = tl.add_clip(Some(tid), Some("B".into()), Some(3.0), Some(2.0), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        let orig_group = tl.clips.iter().find(|c| c.id == c1).unwrap().group_id.clone();
        assert!(orig_group.is_some());

        tl.split_clips_at(&[c1.clone()], 1.0);

        // Left half (start_sec ≈ 0.0) keeps original group
        let left = tl.clips.iter().find(|c| c.start_sec < 0.5 && c.id != c2).unwrap();
        assert_eq!(left.group_id, orig_group);

        // Right half (start_sec ≈ 1.0) gets new group
        let right = tl.clips.iter().find(|c| c.start_sec >= 0.5 && c.id != c2).unwrap();
        assert!(right.group_id.is_some());
        assert_ne!(right.group_id, orig_group);
    }

    /// Right-side group with only 1 member gets dissolved after split.
    #[test]
    fn split_clips_at_dissolves_small_groups() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);
        // c1 at 0.0..2.0, c2 at 0.5..0.8 (entirely left of split point at 1.0)
        let c1 = tl.add_clip(Some(tid.clone()), Some("A".into()), Some(0.0), Some(2.0), None);
        let c2 = tl.add_clip(Some(tid), Some("B".into()), Some(0.5), Some(0.3), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        // Split c1 at 1.0. c2 starts at 0.5 < 1.0 so stays in original (left) group.
        // Left group: left half of c1 + c2 = 2 members → survives.
        // Right group: right half of c1 only → 1 member → dissolved.
        tl.split_clips_at(&[c1.clone()], 1.0);

        // Right half of c1 should have no group (dissolved)
        let right_half = tl.clips.iter().find(|c| c.start_sec >= 0.9 && c.id != c2).unwrap();
        assert!(right_half.group_id.is_none(), "right half should have no group");

        // Left half and c2 should still be in the original group
        let left_half = tl.clips.iter().find(|c| c.start_sec < 0.5 && c.id != c2).unwrap();
        assert!(left_half.group_id.is_some(), "left half should keep group");
        assert!(tl.clips.iter().find(|c| c.id == c2).unwrap().group_id.is_some(), "c2 should keep group");
    }

    /// Unsplit clip entirely to the right of the split point moves to the new group.
    #[test]
    fn split_clips_at_unsplit_member_to_right() {
        let mut tl = TimelineState::default();
        let tid = tl.add_track(Some("T1".into()), None, None);
        let c1 = tl.add_clip(Some(tid.clone()), Some("left".into()), Some(0.0), Some(2.0), None);
        let c2 = tl.add_clip(Some(tid), Some("right".into()), Some(2.5), Some(1.0), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        let orig_group = tl.clips.iter().find(|c| c.id == c1).unwrap().group_id.clone();

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
        let c1 = tl.add_clip(Some(tid.clone()), Some("early".into()), Some(0.0), Some(1.0), None);
        let c2 = tl.add_clip(Some(tid), Some("later".into()), Some(3.0), Some(2.0), None);
        tl.group_clips(&[c1.clone(), c2.clone()]);

        let orig_group = tl.clips.iter().find(|c| c.id == c1).unwrap().group_id.clone();

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
        let a1 = tl.add_clip(Some(tid.clone()), Some("A1".into()), Some(0.0), Some(2.0), None);
        let a2 = tl.add_clip(Some(tid.clone()), Some("A2".into()), Some(3.0), Some(1.0), None);
        tl.group_clips(&[a1.clone(), a2.clone()]);
        let group_a = tl.clips.iter().find(|c| c.id == a1).unwrap().group_id.clone();

        // Group B
        let b1 = tl.add_clip(Some(tid), Some("B1".into()), Some(5.0), Some(2.0), None);
        let b2 = tl.add_clip(None, Some("B2".into()), Some(8.0), Some(1.0), None);
        tl.group_clips(&[b1.clone(), b2.clone()]);
        let group_b = tl.clips.iter().find(|c| c.id == b1).unwrap().group_id.clone();

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
        assert!(groups.len() >= 4, "expected >=4 groups, got {}", groups.len());
    }

    #[test]
    fn split_clip_with_transition_fade_only_sets_boundary_fades() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
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
            curve: Some("sine".to_string()),
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
        assert_eq!(left.fade_out_curve, "sine");
        assert_eq!(right.fade_in_curve, "sine");
    }

    #[test]
    fn split_clip_with_transition_extend_overlap_preserves_source_position() {
        let mut tl = TimelineState::default();
        let track_id = tl.add_track(Some("Track".to_string()), None, None);
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
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
            curve: Some("sine".to_string()),
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
        let right_source_at_split =
            right.source_start_sec + (0.5 - right.start_sec) * 2.0;
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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );

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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );

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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
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
            curve: Some("sine".to_string()),
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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(2.0),
            None,
        );
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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("A".into()),
            Some(0.0),
            Some(1.0),
            None,
        );
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
        let clip_id = tl.add_clip(
            Some(track_id),
            Some("B".into()),
            Some(0.0),
            Some(3.0),
            None,
        );
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
            clip.source_file_fingerprint = Some(saved_fingerprint);
            let meta = std::fs::metadata(&test_path).unwrap();
            clip.source_file_size = Some(meta.len());
            clip.source_file_mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
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
            clip.source_file_fingerprint =
                crate::audio_utils::compute_file_fingerprint(&test_path);
        }
        let changed = tl.check_source_files_changed().changed;
        assert!(changed.is_empty(), "same fingerprint must not report a change");

        let _ = std::fs::remove_file(&test_path);
    }
}

pub(crate) fn new_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

const TRACK_COLOR_PALETTE: &[&str] = &[
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

fn default_fade_curve() -> String {
    "sine".to_string()
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
                reversed: Some(c.reversed),
                loop_enabled: c.loop_enabled,
                fade_in_sec: Some(c.fade_in_sec),
                fade_out_sec: Some(c.fade_out_sec),
                fade_in_curve: Some(c.fade_in_curve.clone()),
                fade_out_curve: Some(c.fade_out_curve.clone()),
                auto_fade_in_sec: Some(c.auto_fade_in_sec),
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
                reversed: Some(c.reversed),
                loop_enabled: c.loop_enabled,
                fade_in_sec: Some(c.fade_in_sec),
                fade_out_sec: Some(c.fade_out_sec),
                fade_in_curve: Some(c.fade_in_curve.clone()),
                fade_out_curve: Some(c.fade_out_curve.clone()),
                auto_fade_in_sec: Some(c.auto_fade_in_sec),
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
        valid.sort_by(|a, b| a.position_sec.partial_cmp(&b.position_sec).unwrap_or(std::cmp::Ordering::Equal));
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

    /// 某绝对秒位置生效的音阶音级集合（Tempo Map 音阶覆盖优先，否则工程音阶）。
    /// 语义与前端 `effectiveScaleAtSec` 及 `scale_segments` 一致：
    /// 音阶为 null 的变化点表示“跟随之前的音阶”（透明），需继续向前寻找
    /// 最近一个显式携带音阶的变化点；找不到才回退工程音阶。
    pub fn effective_scale_notes_at_sec(&self, sec: f64) -> Vec<u8> {
        let Some(points) = self.tempo_map.as_ref() else {
            return self.project_scale_notes.clone();
        };
        let target = sec.max(0.0);
        for point in points.iter().rev() {
            if point.position_sec > target + 1e-9 {
                continue;
            }
            if let Some(scale) = point.scale.as_ref() {
                if let Some(key) = scale.key.as_deref() {
                    if let Some(notes) = scale_notes_for_key(key) {
                        return notes;
                    }
                }
                if let Some(notes) = scale.notes.as_ref() {
                    let mut normalized: Vec<u8> = notes
                        .iter()
                        .map(|v| v % 12)
                        .collect();
                    normalized.sort_unstable();
                    normalized.dedup();
                    if !normalized.is_empty() {
                        return normalized;
                    }
                }
            }
            // 该点音阶为 null（跟随之前的音阶）：继续向前寻找。
        }
        self.project_scale_notes.clone()
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
                if t.parent_id == source.parent_id
                    && t.id != track_id
                    && t.order > src_order
                {
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
        let Some(ref source_path) = clip.source_path else {
            return;
        };
        let p = std::path::Path::new(source_path);
        if !p.exists() {
            return;
        }
        if let Ok(meta) = std::fs::metadata(p) {
            clip.source_file_size = Some(meta.len());
            clip.source_file_mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
        }
        if clip.source_file_fingerprint.is_none() {
            if let Some(fp) = crate::audio_utils::compute_file_fingerprint(p) {
                clip.source_file_fingerprint = Some(fp);
            }
        }
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
            reversed: false,
            // 新 Clip 的 Loop 属性跟随"为新的音频块启用循环"设置
            //（导入/录音/MIDI-as-clip/add_clip 等所有创建路径统一生效）。
            loop_enabled: crate::config::loop_new_clips_default(),
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: default_fade_curve(),
            fade_out_curve: default_fade_curve(),
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

    pub fn remove_clip(&mut self, clip_id: &str) {
        self.clips.retain(|c| c.id != clip_id);
        if self.selected_clip_id.as_deref() == Some(clip_id) {
            self.selected_clip_id = None;
        }
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

    pub fn move_clip(
        &mut self,
        clip_id: &str,
        start_sec: f64,
        track_id: Option<String>,
        move_linked_params: bool,
    ) {
        self.move_clips(
            &[MoveClipPayload {
                clip_id: clip_id.to_string(),
                start_sec,
                track_id,
            }],
            move_linked_params,
        );
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
                reversed,
                loop_enabled: None,
                fade_in_sec,
                fade_out_sec,
                fade_in_curve: None,
                fade_out_curve: None,
                auto_fade_in_sec: None,
                auto_fade_out_sec: None,
                color: None,
                formant_morph: None,
            },
        );
    }

    pub fn patch_clip_state(&mut self, clip_id: &str, patch: ClipStatePatch) {
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
                c.source_end_sec = v.max(0.0);
            }
            if let Some(v) = patch.playback_rate {
                c.playback_rate = v.clamp(0.1, 10.0);
            }
            if let Some(v) = patch.reversed {
                c.reversed = v;
            }
            if let Some(v) = patch.loop_enabled {
                c.loop_enabled = v;
            }
            if let Some(v) = patch.fade_in_sec {
                c.fade_in_sec = v.max(0.0);
            }
            if let Some(v) = patch.fade_out_sec {
                c.fade_out_sec = v.max(0.0);
            }
            if let Some(v) = patch.fade_in_curve {
                c.fade_in_curve = v;
            }
            if let Some(v) = patch.fade_out_curve {
                c.fade_out_curve = v;
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
                    reversed: template.reversed,
                    loop_enabled: template.loop_enabled,
                    fade_in_sec: template.fade_in_sec,
                    fade_out_sec: template.fade_out_sec,
                    fade_in_curve: template.fade_in_curve.clone(),
                    fade_out_curve: template.fade_out_curve.clone(),
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
        let original_group_ids: Vec<Option<String>> = source_clips.iter().map(|c| c.group_id.clone()).collect();

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

        let mut created_clip_ids = Vec::new();
        for source in source_clips {
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

            let old_root_track_id = self.resolve_root_track_id(&source.track_id);
            let new_root_track_id = self.resolve_root_track_id(&target_track_id);
            let linked_params = if payload.copy_linked_params && source.length_sec > 0.0 {
                old_root_track_id.as_ref().and_then(|root_track_id| {
                    self.extract_linked_params_from_root_range(
                        root_track_id,
                        source.start_sec,
                        source.length_sec,
                    )
                })
            } else {
                None
            };

            let mut duplicated = source.clone();
            duplicated.id = new_id("clip");
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
                    group_remap.entry(gid.clone()).or_insert_with(|| Uuid::new_v4().to_string());
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

    pub fn split_clip(&mut self, clip_id: &str, split_sec: f64) -> Option<String> {
        let Some(idx) = self.clips.iter().position(|c| c.id == clip_id) else {
            return None;
        };
        let clip = self.clips[idx].clone();
        let start = clip.start_sec;
        let end = clip.start_sec + clip.length_sec;
        let split = split_sec.clamp(start, end);
        if split <= start + 1e-6 || split >= end - 1e-6 {
            return None;
        }

        self.ensure_project_end_sec(end);

        let left_len = split - start;
        let right_len = end - split;

        // 计算左 clip 的 playback_rate，用于更新 source_end_sec
        let left_rate = {
            let r = self.clips[idx].playback_rate as f64;
            if r.is_finite() && r > 0.0 {
                r
            } else {
                1.0
            }
        };

        self.clips[idx].length_sec = left_len;
        // 更新左 clip 的源区间（**消费窗口模型**，两方向严格镜像）：
        // - Loop（循环源）：左右两段都保留完整循环窗口 —— 左段仅缩短 length，
        //   内容仍按窗口周期回绕；切割点之后的回绕位置由右段自己的窗口表达。
        // - 正放：消费 [ss, ss+S·r) —— 左段终点派生为 ss+S·r。
        //   **不得按旧存储 se 钳制**：延伸/陈旧工程的 se 与长度脱钩，
        //   钳制会把左段窗口错误截短（该有声处变静音）。
        // - 倒放：消费 [se−S·r, se)，锚定 se —— 左段锚点保持不变；
        //   ss 卫生化为派生窗口起点 se−S·r。
        //   **不得按旧 ss 钳制**（同理）。
        {
            let orig_src_start = self.clips[idx].source_start_sec;
            let orig_src_end = self.clips[idx].source_end_sec;
            if self.clips[idx].loop_enabled {
                // 保留窗口不变（见上方注释）。
            } else if self.clips[idx].reversed {
                let new_win_start = orig_src_end - left_len * left_rate;
                self.clips[idx].source_start_sec = new_win_start;
                self.clips[idx].source_end_sec = orig_src_end;
            } else {
                self.clips[idx].source_start_sec = orig_src_start;
                self.clips[idx].source_end_sec = orig_src_start + left_len * left_rate;
            }
        }
        // Fade semantics on split:
        // - fade-in is anchored to the original start, so only the left clip should keep it.
        // - fade-out is anchored to the original end, so only the right clip should keep it.
        // - 切割产生的新边缘（左 clip 的右缘、右 clip 的左缘）**不继承任何淡化**，
        //   包括自动交叉淡化与手动淡化。
        // Clamp fades to the new clip lengths.
        self.clips[idx].fade_in_sec = self.clips[idx].fade_in_sec.min(left_len.max(0.0));
        self.clips[idx].fade_out_sec = 0.0;
        self.clips[idx].auto_fade_out_sec = 0.0;
        self.clips[idx].auto_fade_in_sec = self.clips[idx].auto_fade_in_sec.min(left_len.max(0.0));

        let mut right = clip;
        right.id = new_id("clip");
        right.start_sec = split;
        right.length_sec = right_len;
        right.fade_in_sec = 0.0;
        right.fade_out_sec = right.fade_out_sec.min(right_len.max(0.0));
        right.auto_fade_in_sec = 0.0;
        right.auto_fade_out_sec = right.auto_fade_out_sec.min(right_len.max(0.0));

        // Preserve the original audio offset: the right clip should continue from where the left ended.
        // trim_* are in sec (source time), while playback_rate scales source progress per timeline time.
        let rate = right.playback_rate as f64;
        let rate = if rate.is_finite() && rate > 0.0 {
            rate
        } else {
            1.0
        };
        if right.loop_enabled {
            // Loop（循环源）：右段锚点推进到切割点在**整个媒体文件**上的环绕
            // 位置（对齐 REAPER 切割循环项的行为）：
            //   正放 new_start = floor_mod(start + u, D)
            //   倒放 new_end   = floor_mod(end − u, D)
            // 其中 u = left_len·rate，D = 完整媒体时长。Loop 模式下音频只依赖
            // 锚点与 D，另一端字段保持原值即可；左段窗口不变。
            // 周期 D：优先媒体元数据；缺失（纯 MIDI/音高参考块、无元数据音频）
            // 时兜底 clip_loop_wrap_total_sec（回退 max(end,start)）——
            // 新建 clip 默认 loop=true，若因缺元数据整段跳过推进，右段会从
            // 窗口相位 0 重新开始而非从切割点继续。
            let wrap_total = crate::state::clip_source_media_duration_sec(&right)
                .unwrap_or_else(|| right.source_end_sec.max(right.source_start_sec));
            if let Some(d) = self
                .source_file_duration_sec(&right)
                .or(Some(wrap_total))
                .filter(|d| *d > 1e-9 && d.is_finite())
            {
                let mut u = left_len * rate;
                u %= d;
                if u < 0.0 {
                    u += d;
                }
                if right.reversed {
                    let wrapped = (right.source_end_sec - u).rem_euclid(d);
                    right.source_end_sec = if wrapped <= 0.0 { d } else { wrapped };
                } else {
                    right.source_start_sec =
                        (right.source_start_sec + u).rem_euclid(d);
                }
            }
        } else if right.reversed {
            // 倒放（锚定 source_end）：右段消费窗口为 [ss₀, se−S·r) ——
            // 锚点沿消费方向下移 S·r。**绝不能按旧 ss 上钳**：延伸/陈旧
            // 工程的 ss 可能大于新锚点（用户工程实测：钳制把 1.15 抬回
            // 5.53，右段窗口几乎全落在媒体外 → 偏移彻底错乱）。
            // ss 卫生化为新窗口起点（se'−R·r，恰等于原真实窗口起点），
            // 使存储字段 == 消费窗口，工程数据自洽。
            let new_end = right.source_end_sec - left_len * rate;
            right.source_start_sec = new_end - right_len * rate;
            right.source_end_sec = new_end;
        } else if right.source_start_sec.is_finite() {
            // 正放（派生窗口）：右段消费区间为 [ss+S·r, ss+S·r+R·r)。
            right.source_start_sec =
                (right.source_start_sec + left_len * rate).clamp(-1_000_000.0, 1_000_000.0);
            right.source_end_sec = right.source_start_sec + right_len * rate;
        }
        // Propagate group_id to the split-off right clip
        right.group_id = self.clips[idx].group_id.clone();
        let right_id = right.id.clone();
        self.clips.push(right);
        Some(right_id)
    }

    /// 分割 clip，并在分割完成后根据全局“分割过渡”设置应用淡入淡出或延伸重叠。
    pub fn split_clip_with_transition(
        &mut self,
        clip_id: &str,
        split_sec: f64,
        opts: &SplitTransitionOptions,
    ) -> Option<String> {
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
        let set_manual_fade = |left: &mut Clip, right: &mut Clip, fade_len: f64| {
            left.fade_out_sec = fade_len.min(left.length_sec);
            right.fade_in_sec = fade_len.min(right.length_sec);
            left.auto_fade_out_sec = 0.0;
            right.auto_fade_in_sec = 0.0;
            if let Some(curve) = opts.curve.as_deref() {
                left.fade_out_curve = curve.to_string();
                right.fade_in_curve = curve.to_string();
            }
        };
        // 延伸重叠模式：重叠区的交叉淡化写入“自动交叉淡化”长度（跟随重叠，
        // 分开后自动归零、手动 fade 恢复），适配新的自动交叉淡化模型。
        let set_auto_fade = |left: &mut Clip, right: &mut Clip, fade_len: f64| {
            left.auto_fade_out_sec = fade_len.min(left.length_sec);
            right.auto_fade_in_sec = fade_len.min(right.length_sec);
            left.fade_out_sec = 0.0;
            right.fade_in_sec = 0.0;
            if let Some(curve) = opts.curve.as_deref() {
                left.fade_out_curve = curve.to_string();
                right.fade_in_curve = curve.to_string();
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
                let right_grow = duration
                    .min(self.clips[right_idx].start_sec)
                    .max(0.0);

                let overlap_sec = left_grow + right_grow;
                if overlap_sec <= 0.0 {
                    return;
                }

                // 前 clip 末尾向后延长 left_grow，同时扩展 source 范围，
                // 保证素材内容在时间轴上的位置不变（等价于拖拽 clip 末尾）。
                // Loop（循环源）clip：延长部分由循环回绕内容填充，不改源窗口。
                // 非 Loop：正放终点/倒放起点随增长派生，越出媒体的部分为静音。
                {
                    let left = &mut self.clips[left_idx];
                    left.length_sec += left_grow;
                    if !left.loop_enabled {
                        if left.reversed {
                            left.source_start_sec =
                                left.source_start_sec - left_grow * left_rate;
                        } else {
                            left.source_end_sec =
                                left.source_end_sec + left_grow * left_rate;
                        }
                    }
                }

                // 后 clip 起始位置向前延长 right_grow，同时扩展 source 范围。
                {
                    let media_dur = self.source_file_duration_sec(&self.clips[right_idx]);
                    // Loop（循环源）锚点环绕：媒体时长未知时原样保留。
                    let wrap_anchor = |value: f64| -> f64 {
                        match media_dur.filter(|d| d.is_finite() && *d > 1e-9) {
                            Some(d) => {
                                let m = value % d;
                                if m < 0.0 { m + d } else { m }
                            }
                            None => value,
                        }
                    };
                    let right = &mut self.clips[right_idx];
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
                    } else if right.reversed {
                        // 倒放非 Loop：头部延伸使锚点(source_end)越过媒体时长
                        // → 前导静音，不再按媒体时长钳制。窗口起点随新锚点/
                        // 长度同步派生，保持存储字段 == 消费窗口。
                        right.source_end_sec =
                            right.source_end_sec + right_grow * right_rate;
                        right.source_start_sec =
                            right.source_end_sec - right.length_sec * right_rate;
                    } else {
                        // 正放非 Loop：头部延伸使起点向下穿越媒体起点 → 前导
                        // 静音（派生窗口），不再钳制到 0。终点同步派生，
                        // 保持存储字段 == 消费窗口。
                        right.source_start_sec =
                            right.source_start_sec - right_grow * right_rate;
                        right.source_end_sec =
                            right.source_start_sec + right.length_sec * right_rate;
                    }
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
        if let (Some(frames), Some(sample_rate)) =
            (clip.duration_frames, clip.source_sample_rate)
        {
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
            let has_actual_split = self.clips.iter().any(|clip| {
                new_ids.contains(&clip.id) && clip.group_id.as_ref() == Some(gid)
            });
            if has_actual_split {
                right_group_map.insert(gid.clone(), Some(new_id("group")));
            }
        }

        // 6. 右侧成员迁移（仅限实际发生分割的组）：新右半必然进入新组；
        //    未被切开但完全位于切割点右侧的同组成员也迁入新组；
        //    位于左侧的成员保留原组。
        for (gid, new_gid) in &right_group_map {
            let Some(ref migrated_gid) = new_gid else { continue };
            for clip in self.clips.iter_mut() {
                if clip.group_id.as_ref() != Some(gid) {
                    continue;
                }
                if new_ids.contains(&clip.id) {
                    clip.group_id = Some(migrated_gid.clone());
                } else if clip.start_sec >= split_sec - 1e-6 {
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
                glued.fade_in_curve = default_fade_curve();
                glued.fade_out_curve = default_fade_curve();
                glued.extra_curves = None;
                glued.extra_params = None;
                glued.pitch_range = Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                });
            }
        }

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

            let pitch_midi: Vec<f32> = match crate::pitch_clip::compute_clip_pitch_midi(
                self,
                clip,
                &root_track_id,
                fp_ms,
            ) {
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
                let min_note = remapped
                    .iter()
                    .fold(127.0f32, |m, n| m.min(n.note));
                let max_note = remapped
                    .iter()
                    .fold(0.0f32, |m, n| m.max(n.note));
                let padding = 2.0f32;
                clip.pitch_range = Some(PitchRange {
                    min: (min_note - padding).max(0.0),
                    max: (max_note + padding).min(127.0),
                });
                clip.midi_note_data = Some(remapped);
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
                    eprintln!(
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
                    eprintln!(
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
            eprintln!(
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
        }
    }

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
            eprintln!(
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
    pub fn check_source_files_changed(
        &self,
    ) -> crate::models::CheckSourceFilesChangedPayload {
        let mut changed: Vec<crate::models::SourceFileChangePayload> = Vec::new();
        let mut reported_paths: HashSet<String> = HashSet::new();

        for clip in &self.clips {
            let source_path = match clip.source_path.as_ref() {
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

            let old_mtime = clip.source_file_mtime;
            let old_size = clip.source_file_size;
            let old_fp = clip.source_file_fingerprint;

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

        let gpu_backend = {
            #[cfg(target_os = "windows")] { "DirectML" }
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))] { "WebGPU" }
            #[cfg(all(target_os = "linux", not(target_arch = "x86_64")))] { "" }
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))] { "CoreML" }
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))] { "" }
        };

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
