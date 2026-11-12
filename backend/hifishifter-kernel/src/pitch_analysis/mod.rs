// pitch_analysis — 音高分析主模块
// 工具函数、类型定义、公开 API。
// 核心分析流水线见 analysis.rs，调度逻辑见 schedule.rs。

use crate::state::{Clip, PitchAnalysisAlgo, TimelineState};
use serde::Serialize;
use std::path::Path;

pub mod analysis;
pub mod dyn_analysis;
pub mod schedule;
pub mod syllable;

// 公开 API — 供 crate 内其他模块使用
pub use schedule::maybe_schedule_pitch_orig;

// DYN（动态）原声电平基线的组装入口。
pub use dyn_analysis::{assemble_dyn_orig_for_engine, maybe_schedule_dyn_orig};
pub use syllable::{
    analysis_window_status, clip_spans_to_window, estimate_syllable_spans,
    fill_spans_without_merging, SyllableConfig, SyllableSpan, TakeSyllableAnalysis,
    SYLLABLE_ALGORITHM, SYLLABLE_ALGORITHM_VERSION,
};

pub fn hz_to_midi(hz: f64) -> f32 {
    if !(hz.is_finite() && hz > 1e-6) {
        return 0.0;
    }
    // 利用对数公式抹除浮点除法
    // 69.0 - 12.0 * log2(440.0) ≈ -36.3763165622959
    let midi = 12.0 * hz.log2() - 36.3763165622959;
    if midi.is_finite() {
        midi as f32
    } else {
        0.0
    }
}

pub fn quantize_i64(x: f64, scale: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    (x * scale).round() as i64
}

pub fn quantize_u32(x: f64, scale: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    let v = (x * scale).round();
    if v <= 0.0 {
        0
    } else if v > (u32::MAX as f64) {
        u32::MAX
    } else {
        v as u32
    }
}

pub fn file_sig(path: &Path) -> (u64, u64) {
    // (len_bytes, modified_ms_since_epoch)
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return (0, 0),
    };
    let len = meta.len();
    let mtime_ms = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    (len, mtime_ms)
}

/// clip id 的**布局身份**：剥掉宿主/命名空间前缀，只留 ARA 标签之后的部分。
///
/// 【为什么必须剥】同一份布局在两份时间线里 id 不同：文档是 `ara-item-{guid}`，
/// 编辑器是 `hfs-ui-{pid}-{n}-ara-item-{guid}`。布局键若直接哈希 id，两份时间线
/// 会算出不同的键 —— 文档刚发布的基线在编辑器侧立刻被判"布局变了"，于是每帧
/// 重建一次；反之亦然。按标签后缀取键（与 `workspace::host_item_guid` 同一手法）
/// 让两边得到同一个键。
fn clip_layout_identity(clip_id: &str) -> &str {
    for tag in ["ara-item-", "ara-region-", "ara-clip-"] {
        if let Some(index) = clip_id.rfind(tag) {
            return &clip_id[index + tag.len()..];
        }
    }
    clip_id
}

/// 根布局键：clip 身份 / 工程位置 / 源窗口 / 速率 / 方向。只用于判断
/// “是否需要从内容缓存重新组装根数组”，绝不用于判断“是否需要重跑音频分析”。
pub fn build_root_pitch_layout_key(tl: &TimelineState, root_track_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pitch_root_layout_v1");
    hasher.update(&quantize_u32(tl.frame_period_ms(), 1000.0).to_le_bytes());
    let mut clips: Vec<&Clip> = tl
        .clips
        .iter()
        .filter(|c| tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id))
        .collect();
    clips.sort_by(|a, b| {
        a.start_sec
            .total_cmp(&b.start_sec)
            .then_with(|| a.id.cmp(&b.id))
    });
    for clip in clips {
        hasher.update(clip_layout_identity(&clip.id).as_bytes());
        hasher.update(&quantize_u32(clip.start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(clip.length_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(clip.playback_rate as f64, 10000.0).to_le_bytes());
        hasher.update(&quantize_i64(clip.source_start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(clip.source_end_sec, 1000.0).to_le_bytes());
        hasher.update(&[u8::from(clip.muted)]);
        hasher.update(&[u8::from(clip.loop_enabled)]);
        hasher.update(&[u8::from(clip.reversed)]);
    }
    hasher.finalize().to_hex().to_string()
}

pub fn build_root_pitch_key(tl: &TimelineState, root_track_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pitch_orig_v3_content");
    hasher.update(&quantize_u32(tl.frame_period_ms(), 1000.0).to_le_bytes());

    let (compose, algo) = tl
        .tracks
        .iter()
        .find(|t| t.id == root_track_id)
        .map(|t| (t.compose_enabled, t.pitch_analysis_algo.clone()))
        .unwrap_or((false, PitchAnalysisAlgo::default()));
    hasher.update(&[u8::from(compose)]);
    hasher.update(algo.effective().id().as_bytes());
    if matches!(
        algo.effective(),
        PitchAnalysisAlgo::WorldDll | PitchAnalysisAlgo::NsfHifiganOnnx
    ) {
        hasher.update(&[u8::from(crate::fcpe_onnx::is_available())]);
    }

    // 内容集合按（工程起点、源路径、源窗口）排序，但**不写入**工程位置。
    // 移动 / 分割 / 裁剪因此不会让分析缓存失效；重新组装由布局键负责。
    let mut clips: Vec<&Clip> = tl
        .clips
        .iter()
        .filter(|c| tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id))
        .collect();
    clips.sort_by(|a, b| {
        a.source_path
            .cmp(&b.source_path)
            .then_with(|| a.source_start_sec.total_cmp(&b.source_start_sec))
            .then_with(|| a.source_end_sec.total_cmp(&b.source_end_sec))
            .then_with(|| a.id.cmp(&b.id))
    });
    for clip in clips {
        hasher.update(&quantize_i64(clip.source_start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(clip.source_end_sec, 1000.0).to_le_bytes());
        hasher.update(&[u8::from(clip.loop_enabled)]);
        hasher.update(&[u8::from(clip.reversed)]);
        if let Some(notes) = clip.midi_note_data.as_ref() {
            for note in notes {
                hasher.update(&quantize_u32(note.start_sec, 1000.0).to_le_bytes());
                hasher.update(&quantize_u32(note.end_sec, 1000.0).to_le_bytes());
                hasher.update(&quantize_i64(note.note as f64, 10000.0).to_le_bytes());
                hasher.update(&[note.velocity, note.channel]);
            }
        }
        // 【内容身份优先】见 `Clip::content_id`：Capture 的素材是会话私有临时文件，
        // 路径里的命名空间每个会话都不同。把它哈希进键，等于"每开一次窗口就把整份
        // 音频重分析一遍"。发布方给的稳定标识才是内容身份；缺失时退回路径 + 文件
        // 签名（老工程、外部拖入），行为与改动前一致。
        if let Some(content_id) = clip.content_id.as_deref() {
            hasher.update(b"(content)");
            hasher.update(content_id.as_bytes());
        } else if let Some(sp) = clip.source_path.as_deref() {
            hasher.update(sp.as_bytes());
            let (len, mtime) = file_sig(Path::new(sp));
            hasher.update(&len.to_le_bytes());
            hasher.update(&mtime.to_le_bytes());
            hasher.update(&clip.source_file_fingerprint.unwrap_or(0).to_le_bytes());
        } else {
            hasher.update(b"(no_source)");
        }
    }
    hasher.finalize().to_hex().to_string()
}

/// 一次 `pitch_orig` 装配任务的最小描述。
///
/// 只携带调用方真正需要的两项：目标根轨道，以及本次装配对应的缓存 key。
///
/// 【历史】它曾额外携带 `timeline`（根子树快照）以及 `frame_period_ms` /
/// `target_frames` / `algo`。前三者只被已经删除的旧分析流水线读取，`timeline` 更是
/// 让每次 `maybe_schedule_pitch_orig` 都白克隆一份完整 `TimelineState` —— 而该函数
/// 在时间线编辑路径上有数十个调用点。这些都随旧流水线一并移除。
#[derive(Debug, Clone)]
pub struct PitchJob {
    pub root_track_id: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchOrigUpdatedEvent {
    pub root_track_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchOrigAnalysisStartedEvent {
    pub root_track_id: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchOrigAnalysisProgressEvent {
    pub root_track_id: String,
    pub progress: f32,
    /// 当前正在分析?clip 名称（None 表示未知或已完成?
    pub current_clip_name: Option<String>,
    /// 已完成的 clip 数量
    pub completed_clips: u32,
    /// 需要分析的 clip 总数
    pub total_clips: u32,
}

/// 音节 / 音符分块分析进度（独立于音高原线批次）。
///
/// 前端状态栏只显示这一个快照；`algorithm` 让用户与诊断能看出当前是
/// GAME 模型还是 VAD 回退，不再把两种质量完全不同的结果混为一谈。
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableAnalysisProgress {
    pub active: bool,
    pub clip_id: String,
    pub clip_name: String,
    pub progress: f32,
    pub stage: String,
    pub algorithm: String,
    pub message: Option<String>,
}

fn syllable_progress_slot() -> &'static std::sync::Mutex<Option<SyllableAnalysisProgress>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<SyllableAnalysisProgress>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

pub fn set_syllable_analysis_progress(progress: Option<SyllableAnalysisProgress>) {
    if let Ok(mut slot) = syllable_progress_slot().lock() {
        *slot = progress;
    }
}

pub fn syllable_analysis_progress() -> Option<SyllableAnalysisProgress> {
    syllable_progress_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.clone())
}

pub fn resample_curve_linear(values: &[f32], out_len: usize) -> Vec<f32> {
    if out_len == 0 {
        return vec![];
    }
    if values.is_empty() {
        return vec![0.0; out_len];
    }
    if values.len() == out_len {
        return values.to_vec();
    }
    if values.len() == 1 {
        return vec![values[0]; out_len];
    }
    if out_len == 1 {
        return vec![values[0]];
    }

    let in_len = values.len();
    let scale = (in_len - 1) as f64 / (out_len - 1) as f64;

    // 使用迭代器直接分配并写入，消灭 vec![0.0] 造成的额外 memset
    (0..out_len)
        .map(|of| {
            let t_in = (of as f64) * scale;
            let i0 = t_in.floor() as usize;
            let i1 = (i0 + 1).min(in_len - 1);
            let frac = (t_in - (i0 as f64)) as f32;
            let a = values[i0];
            let b = values[i1];
            a + (b - a) * frac
        })
        .collect()
}

// Task 3.6: PitchProgressPayload for frontend API
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchProgressPayload {
    pub root_track_id: String,
    pub progress: f32,
    pub eta_seconds: Option<f64>,
    /// 当前正在分析?clip 名称
    pub current_clip_name: Option<String>,
    /// 已完成的 clip 数量
    pub completed_clips: u32,
    /// 需要分析的 clip 总数
    pub total_clips: u32,
}

impl From<&PitchOrigAnalysisProgressEvent> for PitchProgressPayload {
    fn from(event: &PitchOrigAnalysisProgressEvent) -> Self {
        Self {
            root_track_id: event.root_track_id.clone(),
            progress: event.progress,
            eta_seconds: None,
            current_clip_name: event.current_clip_name.clone(),
            completed_clips: event.completed_clips,
            total_clips: event.total_clips,
        }
    }
}
