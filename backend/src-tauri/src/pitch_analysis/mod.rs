// pitch_analysis — 音高分析主模块
// 工具函数、类型定义、公开 API。
// 核心分析流水线见 analysis.rs，调度逻辑见 schedule.rs。

use crate::state::{Clip, PitchAnalysisAlgo, TimelineState};
use serde::Serialize;
use std::path::Path;

pub(crate) mod analysis;
pub(crate) mod schedule;

// 公开 API — 供 crate 内其他模块使用
pub use schedule::maybe_schedule_pitch_orig;

#[allow(dead_code)]
pub(crate) fn hz_to_midi(hz: f64) -> f32 {
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

pub(crate) fn quantize_i64(x: f64, scale: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    (x * scale).round() as i64
}

pub(crate) fn quantize_u32(x: f64, scale: f64) -> u32 {
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

pub(crate) fn file_sig(path: &Path) -> (u64, u64) {
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

pub(crate) fn build_root_pitch_key(tl: &TimelineState, root_track_id: &str) -> String {
    let bpm = if tl.bpm.is_finite() && tl.bpm > 0.0 {
        tl.bpm
    } else {
        120.0
    };

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pitch_orig_v2_clip_fuse");
    hasher.update(root_track_id.as_bytes());
    hasher.update(&quantize_u32(bpm, 1000.0).to_le_bytes());
    hasher.update(&quantize_u32(tl.frame_period_ms(), 1000.0).to_le_bytes());

    // Include track-level analysis config.
    let (compose, algo) = tl
        .tracks
        .iter()
        .find(|t| t.id == root_track_id)
        .map(|t| (t.compose_enabled, t.pitch_analysis_algo.clone()))
        .unwrap_or((false, PitchAnalysisAlgo::Unknown));
    hasher.update(&[if compose { 1 } else { 0 }]);
    hasher.update(match algo {
        PitchAnalysisAlgo::WorldDll => b"world_dll",
        PitchAnalysisAlgo::NsfHifiganOnnx => b"nsf_hifigan_onnx",
        PitchAnalysisAlgo::VocalShifterVslib => b"vslib",
        PitchAnalysisAlgo::None => b"none",
        PitchAnalysisAlgo::Unknown => b"unknown",
    });

    // Include detector availability so unavailable states can be cached and
    // recomputed when the detector becomes available.
    if matches!(
        algo,
        PitchAnalysisAlgo::WorldDll
            | PitchAnalysisAlgo::NsfHifiganOnnx
            | PitchAnalysisAlgo::Unknown
    ) {
        hasher.update(&[if crate::fcpe_onnx::is_available() { 1 } else { 0 }]);
    }

    // Include each clip mapped to this root track.
    // Sort by clip id for stability.
    let mut clips: Vec<&Clip> = tl
        .clips
        .iter()
        .filter(|c| tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id))
        .collect();
    clips.sort_by(|a, b| a.id.cmp(&b.id));

    for c in clips {
        hasher.update(c.id.as_bytes());
        hasher.update(&quantize_u32(c.start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(c.length_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(c.playback_rate as f64, 10000.0).to_le_bytes());
        hasher.update(&quantize_i64(c.source_start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(c.source_end_sec, 1000.0).to_le_bytes());
        if let Some(sp) = c.source_path.as_deref() {
            hasher.update(sp.as_bytes());
            let p = Path::new(sp);
            let (len, mtime) = file_sig(p);
            hasher.update(&len.to_le_bytes());
            hasher.update(&mtime.to_le_bytes());
        } else {
            hasher.update(b"(no_source)");
        }
    }

    hasher.finalize().to_hex().to_string()
}

#[derive(Debug, Clone)]
pub(crate) struct PitchJob {
    pub(crate) root_track_id: String,
    pub(crate) key: String,
    #[allow(dead_code)]
    pub(crate) frame_period_ms: f64,
    #[allow(dead_code)]
    pub(crate) target_frames: usize,
    #[allow(dead_code)]
    pub(crate) algo: PitchAnalysisAlgo,

    /// Root-subtree timeline snapshot used for root-mix analysis.
    /// This matches what the parameter panel background waveform shows.
    #[allow(dead_code)]
    pub(crate) timeline: TimelineState,
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

#[allow(dead_code)]
pub(crate) fn resample_curve_linear(values: &[f32], out_len: usize) -> Vec<f32> {
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
