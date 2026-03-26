use crate::audio_utils::try_read_wav_info;
use crate::models::{ProcessAudioPayload, SynthesizePayload};
use crate::state::AppState;
use std::fs;
use std::path::Path;
use tauri::State;

use super::common::{new_temp_wav_path, render_timeline_to_wav};

// ===================== model / processing / synthesis =====================

pub(super) fn load_default_model(state: State<'_, AppState>) -> crate::models::ModelConfigPayload {
    {
        let mut rt = state.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.model_loaded = true;
    }
    state.model_config_ok()
}

pub(super) fn load_model(
    state: State<'_, AppState>,
    model_dir: String,
) -> crate::models::ModelConfigPayload {
    let _ = model_dir;
    {
        let mut rt = state.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.model_loaded = true;
    }
    state.model_config_ok()
}

pub(super) fn set_pitch_shift(semitones: f64) -> serde_json::Value {
    serde_json::json!({"ok": true, "pitch_shift": semitones, "frames": 0})
}

pub(super) fn process_audio(state: State<'_, AppState>, audio_path: String) -> ProcessAudioPayload {
    let path = Path::new(&audio_path);
    let mut duration_sec = 0.0f64;
    let mut sample_rate = 44100u32;
    let mut waveform_preview: Option<Vec<f32>> = None;

    if let Some(info) = try_read_wav_info(path, 4096) {
        duration_sec = info.duration_sec;
        sample_rate = info.sample_rate;
        waveform_preview = Some(info.waveform_preview);
    }

    {
        let mut rt = state.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.audio_loaded = true;
    }

    ProcessAudioPayload {
        ok: true,
        audio: Some(crate::models::ProcessedAudio {
            path: audio_path,
            sample_rate,
            duration_sec,
        }),
        feature: Some(crate::models::AudioFeature {
            mel_shape: None,
            f0_frames: None,
            segment_count: None,
            segments_preview: None,
            waveform_preview,
            pitch_range: Some(crate::models::PitchRange {
                min: -24.0,
                max: 24.0,
            }),
        }),
        timeline: None,
    }
}

pub(super) fn synthesize(state: State<'_, AppState>) -> SynthesizePayload {
    // 删除上一次的 synth 临时文件，避免磁盘泄漏
    {
        let rt = state.runtime.lock().unwrap_or_else(|e| e.into_inner());
        crate::temp_manager::remove_old_synth_temp(rt.synthesized_wav_path.as_deref());
    }

    let out_path = match new_temp_wav_path("synth") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("synthesize: temp path error: {e}");
            return SynthesizePayload {
                ok: false,
                sample_rate: 44100,
                num_samples: 0,
                duration_sec: 0.0,
            };
        }
    };

    let result = match render_timeline_to_wav(&state, &out_path, 0.0, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("synthesize: render error: {e}");
            return SynthesizePayload {
                ok: false,
                sample_rate: 44100,
                num_samples: 0,
                duration_sec: 0.0,
            };
        }
    };

    {
        let mut rt = state.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.has_synthesized = true;
        rt.synthesized_wav_path = Some(out_path.display().to_string());
    }

    let num_samples = (result.duration_sec * result.sample_rate as f64)
        .round()
        .max(0.0) as u32;

    SynthesizePayload {
        ok: true,
        sample_rate: result.sample_rate,
        num_samples,
        duration_sec: result.duration_sec,
    }
}

pub(super) fn save_synthesized(
    state: State<'_, AppState>,
    output_path: String,
) -> serde_json::Value {
    let out_path = Path::new(&output_path);

    // 1. 获取当前最新的时间轴状态
    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    // 2. 构造与分轨导出完全一致的高质量渲染选项
    let opts = crate::mixdown::MixdownOptions {
        sample_rate: 44100,
        start_sec: 0.0,
        end_sec: None,
        stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
        apply_pitch_edit: true,
        export_format: crate::mixdown::ExportFormat::Wav32f,
        quality_preset: crate::mixdown::QualityPreset::Export,
    };

    // 3. 直接调用 mixdown 模块进行高质量重新渲染并写入目标路径
    match crate::mixdown::render_mixdown_wav(&timeline, out_path, opts) {
        Ok(result) => {
            let num_samples = (result.duration_sec * result.sample_rate as f64)
                .round()
                .max(0.0) as u32;

            serde_json::json!({
                "ok": true,
                "path": output_path,
                "sample_rate": result.sample_rate,
                "num_samples": num_samples
            })
        }
        Err(e) => {
            eprintln!("save_synthesized: render failed: {e}");
            serde_json::json!({
                "ok": false,
                "path": output_path,
                "error": e
            })
        }
    }
}

/// 按 root track 分轨导出音频到指定目录。
///
/// 每个 root track（`parent_id == None` 的轨道）以及它的所有子轨道的音频
/// 会被混缩成一个独立的 WAV 文件，文件名为 `{TrackIndex}_{TrackName}.wav`。
pub(super) fn save_separated(state: State<'_, AppState>, output_dir: String) -> serde_json::Value {
    let out_dir = Path::new(&output_dir);
    if !out_dir.exists() {
        if let Err(e) = fs::create_dir_all(out_dir) {
            eprintln!("save_separated: create dir failed: {e}");
            return serde_json::json!({"ok": false, "error": format!("Cannot create directory: {e}")});
        }
    }

    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    // 找到所有 root track（parent_id 为 None）
    let root_tracks: Vec<&crate::state::Track> = timeline
        .tracks
        .iter()
        .filter(|t| t.parent_id.is_none())
        .collect();

    if root_tracks.is_empty() {
        return serde_json::json!({"ok": false, "error": "No root tracks found"});
    }

    // 收集某个 root 下所有后代 track id（包括自身）
    fn collect_descendants(
        tracks: &[crate::state::Track],
        root_id: &str,
    ) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        set.insert(root_id.to_string());
        let mut queue = vec![root_id.to_string()];
        while let Some(cur) = queue.pop() {
            for t in tracks {
                if t.parent_id.as_deref() == Some(cur.as_str()) && !set.contains(&t.id) {
                    set.insert(t.id.clone());
                    queue.push(t.id.clone());
                }
            }
        }
        set
    }

    // 文件名安全化：去掉路径分隔符和不合法字符
    fn sanitize_filename(name: &str) -> String {
        name.chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => c,
            })
            .collect::<String>()
            .trim()
            .to_string()
    }

    let mut results = Vec::new();
    let export_roots: Vec<&crate::state::Track> =
        root_tracks.iter().copied().filter(|t| !t.muted).collect();

    let index_width = {
        let count = export_roots.len();
        if count <= 1 {
            1
        } else {
            (count - 1).to_string().len()
        }
    };

    for (track_idx, root) in export_roots.iter().enumerate() {
        let included = collect_descendants(&timeline.tracks, &root.id);

        // 构建子 timeline：仅保留该 root 分支下的、未 mute 的 tracks 和对应 clips
        let mut sub_tl = timeline.clone();
        sub_tl
            .tracks
            .retain(|t| included.contains(&t.id) && !t.muted);
        let active_track_ids: std::collections::HashSet<&str> =
            sub_tl.tracks.iter().map(|t| t.id.as_str()).collect();
        sub_tl
            .clips
            .retain(|c| active_track_ids.contains(c.track_id.as_str()));

        let safe_name = sanitize_filename(&root.name);
        let normalized_name = if safe_name.is_empty() {
            format!("track_{}", root.id)
        } else {
            safe_name
        };
        let file_name = format!(
            "{:0width$}_{}.wav",
            track_idx,
            normalized_name,
            width = index_width,
        );
        let out_path = out_dir.join(&file_name);

        let opts = crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec: 0.0,
            end_sec: None,
            stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            apply_pitch_edit: true,
            export_format: crate::mixdown::ExportFormat::Wav32f,
            quality_preset: crate::mixdown::QualityPreset::Export,
        };

        match crate::mixdown::render_mixdown_wav(&sub_tl, &out_path, opts) {
            Ok(result) => {
                let num_samples = (result.duration_sec * result.sample_rate as f64)
                    .round()
                    .max(0.0) as u32;
                results.push(serde_json::json!({
                    "track_id": root.id,
                    "name": root.name,
                    "path": out_path.display().to_string(),
                    "ok": true,
                    "sample_rate": result.sample_rate,
                    "num_samples": num_samples,
                }));
            }
            Err(e) => {
                eprintln!(
                    "save_separated: render failed for track '{}': {e}",
                    root.name
                );
                results.push(serde_json::json!({
                    "track_id": root.id,
                    "name": root.name,
                    "ok": false,
                    "error": e,
                }));
            }
        }
    }

    let all_ok = results.iter().all(|r| r["ok"].as_bool().unwrap_or(false));
    serde_json::json!({
        "ok": all_ok,
        "count": results.len(),
        "tracks": results,
        "output_dir": output_dir,
    })
}
