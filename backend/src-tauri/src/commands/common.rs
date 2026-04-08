// 命令层共用工具函数
use crate::state::AppState;
use serde::Serialize;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PlaybackRenderingStateEvent {
    pub(crate) active: bool,
    pub(crate) progress: Option<f64>,
    pub(crate) target: Option<String>,
}

pub(crate) fn guard_json_command(
    name: &str,
    f: impl FnOnce() -> serde_json::Value,
) -> serde_json::Value {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("command panicked: {name}");
            serde_json::json!({"ok": false, "error": format!("panic in command: {name}")})
        }
    }
}

pub(crate) fn guard_waveform_command(
    name: &str,
    f: impl FnOnce() -> super::waveform::WaveformPeaksSegmentPayload,
) -> super::waveform::WaveformPeaksSegmentPayload {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("command panicked: {name}");
            super::waveform::WaveformPeaksSegmentPayload {
                ok: false,
                min: vec![],
                max: vec![],
            }
        }
    }
}

pub(crate) fn ok_bool() -> serde_json::Value {
    serde_json::json!({ "ok": true })
}

pub(crate) fn ensure_temp_dir() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("hifishifter");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub(crate) fn new_temp_wav_path(prefix: &str) -> Result<PathBuf, String> {
    let dir = ensure_temp_dir().map_err(|e| e.to_string())?;
    Ok(dir.join(format!("{}_{}.wav", prefix, Uuid::new_v4().simple())))
}

pub(crate) fn render_timeline_to_wav(
    state: &AppState,
    output_path: &Path,
    start_sec: f64,
    end_sec: Option<f64>,
) -> Result<crate::mixdown::MixdownResult, String> {
    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    crate::mixdown::render_mixdown_wav(
        &timeline,
        output_path,
        crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec,
            stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            apply_pitch_edit: true,
            // 导出时使用最高质量：32-bit float + Export 预设。
            export_format: crate::mixdown::ExportFormat::Wav32f,
            quality_preset: crate::mixdown::QualityPreset::Export,
            cancel_flag: None,
        },
    )
}
