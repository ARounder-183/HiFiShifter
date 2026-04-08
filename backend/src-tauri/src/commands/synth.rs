use crate::audio_utils::try_read_wav_info;
use crate::models::{ProcessAudioPayload, SynthesizePayload};
use crate::state::AppState;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{Emitter, Manager, State};

use super::common::{new_temp_wav_path, render_timeline_to_wav};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ExportAudioMode {
    Project,
    Separated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ExportTimeRangeKind {
    All,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportTimeRange {
    pub kind: ExportTimeRangeKind,
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SeparatedExportTargetKind {
    #[serde(alias = "main")]
    Root,
    Sub,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SeparatedExportTarget {
    pub kind: SeparatedExportTargetKind,
    pub track_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportAudioRequest {
    pub mode: ExportAudioMode,
    pub range: ExportTimeRange,
    pub project_output_dir: Option<String>,
    pub project_file_name: Option<String>,
    pub project_output_path: Option<String>,
    pub separated_output_dir: Option<String>,
    pub separated_name_pattern: Option<String>,
    #[serde(default)]
    pub separated_targets: Vec<SeparatedExportTarget>,
    #[serde(default)]
    pub overwrite_existing_paths: Vec<String>,
    #[serde(default)]
    pub skip_existing_paths: Vec<String>,
    pub sample_rate: Option<u32>,
    pub bit_depth: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportAudioProgressEvent {
    pub active: bool,
    pub mode: Option<ExportAudioMode>,
    pub progress: Option<f64>,
    pub current: Option<usize>,
    pub total: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportAudioDefaultsPayload {
    pub ok: bool,
    pub project_name: String,
    pub documents_dir: String,
    pub project_output_dir: String,
    pub project_file_name: String,
    pub separated_output_dir: String,
    pub separated_file_name: String,
    pub sample_rate: u32,
    pub bit_depth: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportAudioPlanItem {
    pub track_id: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportAudioPlanPayload {
    pub ok: bool,
    pub mode: ExportAudioMode,
    pub targets: Vec<ExportAudioPlanItem>,
    pub existing_paths: Vec<String>,
}

#[derive(Debug, Clone)]
struct ResolvedSeparatedTarget {
    kind: SeparatedExportTargetKind,
    track_id: String,
    track_name: String,
    track_index: usize,
    included_track_ids: HashSet<String>,
}

fn export_cancel_slot() -> &'static Mutex<Vec<Arc<AtomicBool>>> {
    static SLOT: OnceLock<Mutex<Vec<Arc<AtomicBool>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(Vec::new()))
}

struct ExportCancelGuard {
    flag: Arc<AtomicBool>,
}

fn install_export_cancel_flag(flag: Arc<AtomicBool>) -> ExportCancelGuard {
    let mut slot = export_cancel_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    slot.push(flag.clone());
    ExportCancelGuard { flag }
}

impl Drop for ExportCancelGuard {
    fn drop(&mut self) {
        let mut slot = export_cancel_slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        slot.retain(|f| !Arc::ptr_eq(f, &self.flag));
    }
}

pub(super) fn cancel_export_audio() -> serde_json::Value {
    let flags = {
        export_cancel_slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    };

    let active = !flags.is_empty();
    for flag in flags.iter() {
        flag.store(true, Ordering::Relaxed);
    }

    serde_json::json!({
        "ok": true,
        "active": active,
    })
}

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
        cancel_flag: None,
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
            cancel_flag: None,
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

pub(super) fn get_export_audio_defaults(state: State<'_, AppState>) -> ExportAudioDefaultsPayload {
    let project_name = resolve_project_name(&state);
    let documents_dir = resolve_documents_dir(&state).unwrap_or_else(|| PathBuf::from("."));

    let defaults_project_dir = PathBuf::from("<ProjectFolder>");
    let defaults_project_file_name = "<ProjectName>.wav".to_string();
    let defaults_separated_dir = PathBuf::from("<ProjectFolder>").join("<ProjectName>");
    let defaults_separated_file_name = "<ExportIndex>_<TrackName>.wav".to_string();

    let export_settings = state
        .config_dir
        .get()
        .map(|config_dir| crate::config::load_export_settings(config_dir))
        .unwrap_or_default();

    let project_output_dir = export_settings
        .project_output_dir
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| defaults_project_dir.display().to_string());

    let project_file_name = export_settings
        .project_file_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or(defaults_project_file_name);

    let separated_output_dir = export_settings
        .separated_output_dir
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| defaults_separated_dir.display().to_string());

    let separated_file_name = export_settings
        .separated_file_name_pattern
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or(defaults_separated_file_name);

    let sample_rate = normalize_export_sample_rate(export_settings.sample_rate);
    let bit_depth = normalize_export_bit_depth(export_settings.bit_depth);

    ExportAudioDefaultsPayload {
        ok: true,
        project_name,
        documents_dir: documents_dir.display().to_string(),
        project_output_dir,
        project_file_name,
        separated_output_dir,
        separated_file_name,
        sample_rate,
        bit_depth,
    }
}

pub(super) fn preview_export_audio_plan(
    state: State<'_, AppState>,
    request: ExportAudioRequest,
) -> ExportAudioPlanPayload {
    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let project_name = resolve_project_name(&state);
    let project_folder = resolve_project_folder(&state).display().to_string();
    let export_start_time = Local::now();

    match request.mode {
        ExportAudioMode::Project => {
            let Ok((path, _, _)) = resolve_project_output_path(
                &state,
                &request,
                &project_name,
                export_start_time,
            ) else {
                return ExportAudioPlanPayload {
                    ok: false,
                    mode: ExportAudioMode::Project,
                    targets: vec![],
                    existing_paths: vec![],
                };
            };
            let path_text = path.display().to_string();
            let existing_paths = if path.exists() {
                vec![path_text.clone()]
            } else {
                vec![]
            };
            ExportAudioPlanPayload {
                ok: true,
                mode: ExportAudioMode::Project,
                targets: vec![ExportAudioPlanItem {
                    track_id: None,
                    path: path_text,
                }],
                existing_paths,
            }
        }
        ExportAudioMode::Separated => {
            let output_dir_template = request
                .separated_output_dir
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or("");
            if output_dir_template.is_empty() {
                return ExportAudioPlanPayload {
                    ok: false,
                    mode: ExportAudioMode::Separated,
                    targets: vec![],
                    existing_paths: vec![],
                };
            }

            let output_dir = match resolve_output_dir_template(
                output_dir_template,
                &project_name,
                &project_folder,
                export_start_time,
            ) {
                Ok(value) => value,
                Err(_) => {
                    return ExportAudioPlanPayload {
                        ok: false,
                        mode: ExportAudioMode::Separated,
                        targets: vec![],
                        existing_paths: vec![],
                    };
                }
            };
            let out_dir = PathBuf::from(output_dir);
            let display_tracks = build_display_track_order(&timeline.tracks);
            let Ok(resolved_targets) = resolve_separated_targets(
                &timeline.tracks,
                &request.separated_targets,
                &display_tracks,
            ) else {
                return ExportAudioPlanPayload {
                    ok: false,
                    mode: ExportAudioMode::Separated,
                    targets: vec![],
                    existing_paths: vec![],
                };
            };

            let pattern = request
                .separated_name_pattern
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or("<ExportIndex>_<TrackName>.wav");
            let index_width = display_tracks.len().max(1).to_string().len();
            let mut used_names = HashSet::new();
            let mut targets = Vec::with_capacity(resolved_targets.len());
            let mut existing_paths = Vec::new();

            for (target_index, target) in resolved_targets.into_iter().enumerate() {
                let track_type = match target.kind {
                    SeparatedExportTargetKind::Root => "Root",
                    SeparatedExportTargetKind::Sub => "Sub",
                };
                let relative = match build_unique_export_file_name(
                    pattern,
                    target.track_index,
                    target_index,
                    index_width,
                    &target.track_name,
                    track_type,
                    &target.track_id,
                    &project_name,
                    Local::now(),
                    &mut used_names,
                ) {
                    Ok(value) => value,
                    Err(_) => {
                        return ExportAudioPlanPayload {
                            ok: false,
                            mode: ExportAudioMode::Separated,
                            targets: vec![],
                            existing_paths: vec![],
                        };
                    }
                };
                let path = out_dir.join(relative);
                let path_text = path.display().to_string();
                if path.exists() {
                    existing_paths.push(path_text.clone());
                }
                targets.push(ExportAudioPlanItem {
                    track_id: Some(target.track_id),
                    path: path_text,
                });
            }

            ExportAudioPlanPayload {
                ok: true,
                mode: ExportAudioMode::Separated,
                targets,
                existing_paths,
            }
        }
    }
}

pub(super) fn export_audio_advanced(
    state: State<'_, AppState>,
    request: ExportAudioRequest,
) -> serde_json::Value {
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let _cancel_guard = install_export_cancel_flag(cancel_flag.clone());

    let requested_sample_rate = normalize_export_sample_rate(request.sample_rate.unwrap_or(44_100));
    let requested_bit_depth = normalize_export_bit_depth(request.bit_depth.unwrap_or(32));
    let requested_export_format = export_format_from_bit_depth(requested_bit_depth);

    let overwrite_path_keys: HashSet<String> = request
        .overwrite_existing_paths
        .iter()
        .map(|path| normalize_path_key(path))
        .collect();
    let skip_path_keys: HashSet<String> = request
        .skip_existing_paths
        .iter()
        .map(|path| normalize_path_key(path))
        .collect();

    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    let (start_sec, end_sec) = match normalize_export_range(&timeline, &request.range) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({
                "ok": false,
                "error": e,
            });
        }
    };

    match request.mode {
        ExportAudioMode::Project => {
            let project_name = resolve_project_name(&state);
            let export_start_time = Local::now();
            let (out_path, output_dir_value, file_name_value) =
                match resolve_project_output_path(&state, &request, &project_name, export_start_time) {
                    Ok(value) => value,
                    Err(error) => {
                        return serde_json::json!({
                            "ok": false,
                            "mode": "project",
                            "error": error,
                        });
                    }
                };

            if let Some(parent) = out_path.parent() {
                if let Err(error) = fs::create_dir_all(parent) {
                    return serde_json::json!({
                        "ok": false,
                        "mode": "project",
                        "path": out_path.display().to_string(),
                        "error": format!("Cannot create directory: {error}"),
                    });
                }
            }

            let out_path_key = normalize_path_key(&out_path.display().to_string());
            if out_path.exists() {
                if skip_path_keys.contains(&out_path_key) {
                    emit_export_audio_progress(
                        &state,
                        ExportAudioProgressEvent {
                            active: false,
                            mode: Some(ExportAudioMode::Project),
                            progress: Some(1.0),
                            current: Some(1),
                            total: Some(1),
                        },
                    );
                    return serde_json::json!({
                        "ok": true,
                        "mode": "project",
                        "path": out_path.display().to_string(),
                        "skipped": true,
                    });
                }
                if !overwrite_path_keys.contains(&out_path_key) {
                    return serde_json::json!({
                        "ok": false,
                        "mode": "project",
                        "path": out_path.display().to_string(),
                        "error": "export_target_exists",
                    });
                }
            }

            emit_export_audio_progress(
                &state,
                ExportAudioProgressEvent {
                    active: true,
                    mode: Some(ExportAudioMode::Project),
                    progress: Some(0.0),
                    current: Some(0),
                    total: Some(1),
                },
            );

            let opts = crate::mixdown::MixdownOptions {
                sample_rate: requested_sample_rate,
                start_sec,
                end_sec,
                stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
                apply_pitch_edit: true,
                export_format: requested_export_format,
                quality_preset: crate::mixdown::QualityPreset::Export,
                cancel_flag: Some(cancel_flag.clone()),
            };

            match crate::mixdown::render_mixdown_wav(&timeline, &out_path, opts) {
                Ok(result) => {
                    let num_samples = (result.duration_sec * result.sample_rate as f64)
                        .round()
                        .max(0.0) as u32;

                    persist_successful_export_settings(
                        &state,
                        Some(output_dir_value),
                        Some(file_name_value),
                        None,
                        None,
                        Some(requested_sample_rate),
                        Some(requested_bit_depth),
                    );

                    emit_export_audio_progress(
                        &state,
                        ExportAudioProgressEvent {
                            active: false,
                            mode: Some(ExportAudioMode::Project),
                            progress: Some(1.0),
                            current: Some(1),
                            total: Some(1),
                        },
                    );

                    serde_json::json!({
                        "ok": true,
                        "mode": "project",
                        "path": out_path.display().to_string(),
                        "sample_rate": result.sample_rate,
                        "num_samples": num_samples,
                    })
                }
                Err(e) => {
                    if e == "export_cancelled" {
                        emit_export_audio_progress(
                            &state,
                            ExportAudioProgressEvent {
                                active: false,
                                mode: Some(ExportAudioMode::Project),
                                progress: Some(1.0),
                                current: Some(1),
                                total: Some(1),
                            },
                        );

                        return serde_json::json!({
                            "ok": false,
                            "mode": "project",
                            "path": out_path.display().to_string(),
                            "cancelled": true,
                            "error": "export_cancelled",
                        });
                    }

                    emit_export_audio_progress(
                        &state,
                        ExportAudioProgressEvent {
                            active: false,
                            mode: Some(ExportAudioMode::Project),
                            progress: Some(1.0),
                            current: Some(1),
                            total: Some(1),
                        },
                    );

                    serde_json::json!({
                        "ok": false,
                        "mode": "project",
                        "path": out_path.display().to_string(),
                        "error": e,
                    })
                }
            }
        }
        ExportAudioMode::Separated => {
            let project_name = resolve_project_name(&state);
            let project_folder = resolve_project_folder(&state).display().to_string();
            let export_start_time = Local::now();

            let output_dir_template = match request
                .separated_output_dir
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
            {
                Some(v) => v,
                None => {
                    return serde_json::json!({
                        "ok": false,
                        "error": "separated_output_dir is required",
                    });
                }
            };

            let output_dir = match resolve_output_dir_template(
                output_dir_template,
                &project_name,
                &project_folder,
                export_start_time,
            ) {
                Ok(value) => value,
                Err(error) => {
                    return serde_json::json!({
                        "ok": false,
                        "error": error,
                    });
                }
            };
            if output_dir.trim().is_empty() {
                return serde_json::json!({
                    "ok": false,
                    "error": "Invalid separated_output_dir",
                });
            }

            let out_dir = Path::new(&output_dir);
            if !out_dir.exists() {
                if let Err(e) = fs::create_dir_all(out_dir) {
                    return serde_json::json!({
                        "ok": false,
                        "error": format!("Cannot create directory: {e}"),
                    });
                }
            }

            let display_tracks = build_display_track_order(&timeline.tracks);
            let resolved_targets = match resolve_separated_targets(
                &timeline.tracks,
                &request.separated_targets,
                &display_tracks,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return serde_json::json!({
                        "ok": false,
                        "error": e,
                    });
                }
            };

            if resolved_targets.is_empty() {
                return serde_json::json!({
                    "ok": false,
                    "error": "No separated export targets selected",
                });
            }

            let total_targets = resolved_targets.len();
            emit_export_audio_progress(
                &state,
                ExportAudioProgressEvent {
                    active: true,
                    mode: Some(ExportAudioMode::Separated),
                    progress: Some(0.0),
                    current: Some(0),
                    total: Some(total_targets),
                },
            );

            let pattern = request
                .separated_name_pattern
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or("<ExportIndex>_<TrackName>.wav");

            let index_width = display_tracks.len().max(1).to_string().len();
            let mut used_names: HashSet<String> = HashSet::new();
            let mut results = Vec::with_capacity(resolved_targets.len());

            for (target_index, target) in resolved_targets.into_iter().enumerate() {
                if cancel_flag.load(Ordering::Relaxed) {
                    results.push(serde_json::json!({
                        "track_id": target.track_id,
                        "track_index": target.track_index,
                        "name": target.track_name,
                        "ok": false,
                        "cancelled": true,
                        "error": "export_cancelled",
                    }));
                    break;
                }

                let track_type = match target.kind {
                    SeparatedExportTargetKind::Root => "Root",
                    SeparatedExportTargetKind::Sub => "Sub",
                };
                let relative_path = match build_unique_export_file_name(
                    pattern,
                    target.track_index,
                    target_index,
                    index_width,
                    &target.track_name,
                    track_type,
                    &target.track_id,
                    &project_name,
                    Local::now(),
                    &mut used_names,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        results.push(serde_json::json!({
                            "track_id": target.track_id,
                            "track_index": target.track_index,
                            "track_type": track_type,
                            "name": target.track_name,
                            "ok": false,
                            "error": error,
                        }));

                        let current = target_index + 1;
                        let progress = if total_targets == 0 {
                            1.0
                        } else {
                            current as f64 / total_targets as f64
                        };
                        emit_export_audio_progress(
                            &state,
                            ExportAudioProgressEvent {
                                active: true,
                                mode: Some(ExportAudioMode::Separated),
                                progress: Some(progress),
                                current: Some(current),
                                total: Some(total_targets),
                            },
                        );
                        continue;
                    }
                };
                let out_path = out_dir.join(&relative_path);
                if let Some(parent) = out_path.parent() {
                    if let Err(error) = fs::create_dir_all(parent) {
                        results.push(serde_json::json!({
                            "track_id": target.track_id,
                            "track_index": target.track_index,
                            "track_type": track_type,
                            "name": target.track_name,
                            "path": out_path.display().to_string(),
                            "ok": false,
                            "error": format!("Cannot create directory: {error}"),
                        }));
                        continue;
                    }
                }

                let out_path_key = normalize_path_key(&out_path.display().to_string());
                if out_path.exists() {
                    if skip_path_keys.contains(&out_path_key) {
                        results.push(serde_json::json!({
                            "track_id": target.track_id,
                            "track_index": target.track_index,
                            "track_type": track_type,
                            "name": target.track_name,
                            "path": out_path.display().to_string(),
                            "ok": true,
                            "skipped": true,
                        }));

                        let current = target_index + 1;
                        let progress = if total_targets == 0 {
                            1.0
                        } else {
                            current as f64 / total_targets as f64
                        };
                        emit_export_audio_progress(
                            &state,
                            ExportAudioProgressEvent {
                                active: true,
                                mode: Some(ExportAudioMode::Separated),
                                progress: Some(progress),
                                current: Some(current),
                                total: Some(total_targets),
                            },
                        );
                        continue;
                    }
                    if !overwrite_path_keys.contains(&out_path_key) {
                        results.push(serde_json::json!({
                            "track_id": target.track_id,
                            "track_index": target.track_index,
                            "track_type": track_type,
                            "name": target.track_name,
                            "path": out_path.display().to_string(),
                            "ok": false,
                            "error": "export_target_exists",
                        }));

                        let current = target_index + 1;
                        let progress = if total_targets == 0 {
                            1.0
                        } else {
                            current as f64 / total_targets as f64
                        };
                        emit_export_audio_progress(
                            &state,
                            ExportAudioProgressEvent {
                                active: true,
                                mode: Some(ExportAudioMode::Separated),
                                progress: Some(progress),
                                current: Some(current),
                                total: Some(total_targets),
                            },
                        );
                        continue;
                    }
                }

                let mut sub_timeline = timeline.clone();
                sub_timeline
                    .tracks
                    .retain(|track| target.included_track_ids.contains(&track.id));
                let active_track_ids: HashSet<&str> =
                    sub_timeline.tracks.iter().map(|track| track.id.as_str()).collect();
                sub_timeline
                    .clips
                    .retain(|clip| active_track_ids.contains(clip.track_id.as_str()));

                let opts = crate::mixdown::MixdownOptions {
                    sample_rate: requested_sample_rate,
                    start_sec,
                    end_sec,
                    stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
                    apply_pitch_edit: true,
                    export_format: requested_export_format,
                    quality_preset: crate::mixdown::QualityPreset::Export,
                    cancel_flag: Some(cancel_flag.clone()),
                };

                match crate::mixdown::render_mixdown_wav(&sub_timeline, &out_path, opts) {
                    Ok(result) => {
                        let num_samples = (result.duration_sec * result.sample_rate as f64)
                            .round()
                            .max(0.0) as u32;
                        results.push(serde_json::json!({
                            "track_id": target.track_id,
                            "track_index": target.track_index,
                            "track_type": track_type,
                            "name": target.track_name,
                            "path": out_path.display().to_string(),
                            "ok": true,
                            "sample_rate": result.sample_rate,
                            "num_samples": num_samples,
                        }));
                    }
                    Err(e) => {
                        let cancelled = e == "export_cancelled";
                        results.push(serde_json::json!({
                            "track_id": target.track_id,
                            "track_index": target.track_index,
                            "track_type": track_type,
                            "name": target.track_name,
                            "path": out_path.display().to_string(),
                            "ok": false,
                            "cancelled": cancelled,
                            "error": e,
                        }));

                        if cancelled {
                            break;
                        }
                    }
                }

                let current = target_index + 1;
                let progress = if total_targets == 0 {
                    1.0
                } else {
                    current as f64 / total_targets as f64
                };
                emit_export_audio_progress(
                    &state,
                    ExportAudioProgressEvent {
                        active: true,
                        mode: Some(ExportAudioMode::Separated),
                        progress: Some(progress),
                        current: Some(current),
                        total: Some(total_targets),
                    },
                );
            }

            let all_ok = results.iter().all(|r| r["ok"].as_bool().unwrap_or(false));
            let cancelled = results
                .iter()
                .any(|r| r["cancelled"].as_bool().unwrap_or(false));

            if all_ok {
                persist_successful_export_settings(
                    &state,
                    None,
                    None,
                    Some(output_dir_template.to_string()),
                    Some(pattern.to_string()),
                    Some(requested_sample_rate),
                    Some(requested_bit_depth),
                );
            }

            emit_export_audio_progress(
                &state,
                ExportAudioProgressEvent {
                    active: false,
                    mode: Some(ExportAudioMode::Separated),
                    progress: Some(1.0),
                    current: Some(total_targets),
                    total: Some(total_targets),
                },
            );

            serde_json::json!({
                "ok": all_ok,
                "mode": "separated",
                "cancelled": cancelled,
                "count": results.len(),
                "tracks": results,
                "output_dir": output_dir,
            })
        }
    }
}

fn normalize_export_range(
    timeline: &crate::state::TimelineState,
    range: &ExportTimeRange,
) -> Result<(f64, Option<f64>), String> {
    let project_sec = latest_clip_end_sec(timeline).max(0.0);
    match range.kind {
        ExportTimeRangeKind::All => Ok((0.0, Some(project_sec))),
        ExportTimeRangeKind::Custom => {
            let start_sec = range.start_sec.unwrap_or(0.0);
            let end_sec = range.end_sec.unwrap_or(project_sec);
            if !start_sec.is_finite() || !end_sec.is_finite() {
                return Err("Invalid custom export range".to_string());
            }

            let clamped_start = start_sec.clamp(0.0, project_sec);
            let clamped_end = end_sec.clamp(clamped_start, project_sec);
            if clamped_end <= clamped_start {
                return Err("Custom export range must have end > start".to_string());
            }

            Ok((clamped_start, Some(clamped_end)))
        }
    }
}

fn build_display_track_order(tracks: &[crate::state::Track]) -> Vec<crate::state::Track> {
    let mut by_parent: HashMap<Option<String>, Vec<crate::state::Track>> = HashMap::new();
    for track in tracks.iter().cloned() {
        by_parent.entry(track.parent_id.clone()).or_default().push(track);
    }
    for siblings in by_parent.values_mut() {
        siblings.sort_by_key(|track| track.order);
    }

    let mut out = Vec::with_capacity(tracks.len());
    let roots = by_parent.get(&None).cloned().unwrap_or_default();

    fn dfs(
        track: &crate::state::Track,
        by_parent: &HashMap<Option<String>, Vec<crate::state::Track>>,
        out: &mut Vec<crate::state::Track>,
    ) {
        out.push(track.clone());
        if let Some(children) = by_parent.get(&Some(track.id.clone())) {
            for child in children {
                dfs(child, by_parent, out);
            }
        }
    }

    for root in roots {
        dfs(&root, &by_parent, &mut out);
    }

    if out.len() != tracks.len() {
        let mut seen: HashSet<String> = out.iter().map(|track| track.id.clone()).collect();
        for track in tracks {
            if !seen.contains(&track.id) {
                out.push(track.clone());
                seen.insert(track.id.clone());
            }
        }
    }

    out
}

fn resolve_separated_targets(
    tracks: &[crate::state::Track],
    targets: &[SeparatedExportTarget],
    display_tracks: &[crate::state::Track],
) -> Result<Vec<ResolvedSeparatedTarget>, String> {
    let mut track_by_id: HashMap<&str, &crate::state::Track> = HashMap::new();
    for track in tracks {
        track_by_id.insert(track.id.as_str(), track);
    }

    let mut children_by_parent: HashMap<Option<&str>, Vec<&crate::state::Track>> = HashMap::new();
    for track in tracks {
        children_by_parent
            .entry(track.parent_id.as_deref())
            .or_default()
            .push(track);
    }
    for siblings in children_by_parent.values_mut() {
        siblings.sort_by_key(|track| track.order);
    }

    let display_index: HashMap<&str, usize> = display_tracks
        .iter()
        .enumerate()
        .map(|(idx, track)| (track.id.as_str(), idx + 1))
        .collect();

    let mut normalized_targets: Vec<ResolvedSeparatedTarget> = Vec::new();
    let mut dedup: HashSet<(SeparatedExportTargetKind, String)> = HashSet::new();

    for target in targets {
        let selected_track = match track_by_id.get(target.track_id.as_str()) {
            Some(track) => *track,
            None => {
                return Err(format!("Track not found: {}", target.track_id));
            }
        };

        let normalized_track = match target.kind {
            SeparatedExportTargetKind::Root => {
                let mut cursor = selected_track;
                let mut safety = 0usize;
                while let Some(parent_id) = cursor.parent_id.as_deref() {
                    if let Some(parent) = track_by_id.get(parent_id) {
                        cursor = parent;
                    } else {
                        break;
                    }
                    safety += 1;
                    if safety > tracks.len() {
                        break;
                    }
                }
                cursor
            }
            SeparatedExportTargetKind::Sub => selected_track,
        };

        if !dedup.insert((target.kind.clone(), normalized_track.id.clone())) {
            continue;
        }

        let track_index = display_index
            .get(normalized_track.id.as_str())
            .copied()
            .unwrap_or(1);

        let included_track_ids = match target.kind {
            SeparatedExportTargetKind::Root => {
                let mut collected: HashSet<String> = HashSet::new();
                let mut queue = vec![normalized_track.id.as_str()];
                while let Some(current_id) = queue.pop() {
                    if !collected.insert(current_id.to_string()) {
                        continue;
                    }
                    if let Some(children) = children_by_parent.get(&Some(current_id)) {
                        for child in children {
                            queue.push(child.id.as_str());
                        }
                    }
                }
                collected
            }
            SeparatedExportTargetKind::Sub => {
                let mut collected = HashSet::new();
                let mut cursor = Some(normalized_track);
                let mut safety = 0usize;
                while let Some(track) = cursor {
                    if !collected.insert(track.id.clone()) {
                        break;
                    }
                    cursor = track
                        .parent_id
                        .as_deref()
                        .and_then(|parent_id| track_by_id.get(parent_id).copied());
                    safety += 1;
                    if safety > tracks.len() {
                        break;
                    }
                }
                collected
            }
        };

        normalized_targets.push(ResolvedSeparatedTarget {
            kind: target.kind.clone(),
            track_id: normalized_track.id.clone(),
            track_name: normalized_track.name.clone(),
            track_index,
            included_track_ids,
        });
    }

    normalized_targets.sort_by(|a, b| {
        a.track_index
            .cmp(&b.track_index)
            .then_with(|| match (&a.kind, &b.kind) {
                (SeparatedExportTargetKind::Root, SeparatedExportTargetKind::Sub) => {
                    std::cmp::Ordering::Less
                }
                (SeparatedExportTargetKind::Sub, SeparatedExportTargetKind::Root) => {
                    std::cmp::Ordering::Greater
                }
                _ => std::cmp::Ordering::Equal,
            })
    });

    Ok(normalized_targets)
}

fn sanitize_file_name_segment(raw: &str) -> String {
    raw.chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

fn emit_export_audio_progress(state: &AppState, payload: ExportAudioProgressEvent) {
    if let Some(handle) = state.app_handle.get() {
        let _ = handle.emit("export_audio_progress", payload);
    }
}

fn resolve_project_name(state: &AppState) -> String {
    let project = state.project.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(path) = project.path.as_deref() {
        let stem = Path::new(path)
            .file_stem()
            .and_then(|value| value.to_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("Untitled");
        return sanitize_file_name_segment(stem);
    }

    let name = project.name.trim();
    if name.is_empty() {
        "Untitled".to_string()
    } else {
        sanitize_file_name_segment(name)
    }
}

fn latest_clip_end_sec(timeline: &crate::state::TimelineState) -> f64 {
    timeline
        .clips
        .iter()
        .map(|clip| (clip.start_sec + clip.length_sec).max(0.0))
        .fold(0.0_f64, f64::max)
}

fn normalize_export_sample_rate(sample_rate: u32) -> u32 {
    match sample_rate {
        8_000 | 11_025 | 12_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000
        | 88_200 | 96_000 | 176_400 | 192_000 => sample_rate,
        _ => 44_100,
    }
}

fn normalize_export_bit_depth(bit_depth: u32) -> u32 {
    match bit_depth {
        16 | 24 | 32 => bit_depth,
        _ => 32,
    }
}

fn export_format_from_bit_depth(bit_depth: u32) -> crate::mixdown::ExportFormat {
    match normalize_export_bit_depth(bit_depth) {
        16 => crate::mixdown::ExportFormat::Wav16,
        24 => crate::mixdown::ExportFormat::Wav24,
        _ => crate::mixdown::ExportFormat::Wav32f,
    }
}

fn try_apply_time_format(template: &str, time: chrono::DateTime<Local>) -> Result<String, String> {
    let direct = std::panic::catch_unwind(|| time.format(template).to_string());
    if let Ok(value) = direct {
        return Ok(value);
    }

    let escaped = template.replace('%', "%%");
    let escaped_try = std::panic::catch_unwind(|| time.format(&escaped).to_string());
    if let Ok(value) = escaped_try {
        return Ok(value);
    }

    Err("export_invalid_time_format".to_string())
}

fn resolve_project_folder(state: &AppState) -> PathBuf {
    let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(path) = project.path.as_deref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            return parent.to_path_buf();
        }
    }
    resolve_documents_dir(state).unwrap_or_else(|| PathBuf::from("."))
}

fn resolve_documents_dir(state: &AppState) -> Option<PathBuf> {
    if let Some(handle) = state.app_handle.get() {
        if let Ok(dir) = handle.path().document_dir() {
            return Some(dir);
        }
    }

    if cfg!(target_os = "windows") {
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            return Some(PathBuf::from(profile).join("Documents"));
        }
    }

    std::env::var_os("HOME").map(PathBuf::from)
}

fn resolve_project_output_path(
    state: &AppState,
    request: &ExportAudioRequest,
    project_name: &str,
    export_start_time: chrono::DateTime<Local>,
) -> Result<(PathBuf, String, String), String> {
    let project_folder = resolve_project_folder(state).display().to_string();
    let project_output_dir = request
        .project_output_dir
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let project_file_name_pattern = request
        .project_file_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("<ProjectName>.wav");

    if let Some(output_dir_template) = project_output_dir {
        let output_dir = resolve_output_dir_template(
            output_dir_template,
            project_name,
            &project_folder,
            export_start_time,
        )?;
        if output_dir.trim().is_empty() {
            return Err("project_output_dir resolves to empty path".to_string());
        }

        let file_name_template = project_file_name_pattern
            .replace("<ProjectName>", project_name)
            .replace("<ProjectFolder>", &project_folder)
            .trim()
            .to_string();
        let mut file_name = try_apply_time_format(&file_name_template, Local::now())?;

        file_name = sanitize_file_name_segment(&file_name);
        if file_name.is_empty() {
            file_name = format!("{project_name}.wav");
        }
        if !file_name.to_ascii_lowercase().ends_with(".wav") {
            file_name.push_str(".wav");
        }

        return Ok((
            Path::new(&output_dir).join(&file_name),
            output_dir_template.to_string(),
            project_file_name_pattern.to_string(),
        ));
    }

    if let Some(output_path) = request
        .project_output_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let out_path = PathBuf::from(output_path);
        let output_dir = out_path
            .parent()
            .map(|parent| parent.display().to_string())
            .unwrap_or_default();
        let file_name = out_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("output.wav")
            .to_string();
        return Ok((out_path, output_dir, file_name));
    }

    Err("project_output_dir and project_file_name are required".to_string())
}

fn resolve_output_dir_template(
    output_dir_template: &str,
    project_name: &str,
    project_folder: &str,
    export_start_time: chrono::DateTime<Local>,
) -> Result<String, String> {
    let replaced = output_dir_template
        .replace("<ProjectName>", project_name)
        .replace("<ProjectFolder>", project_folder)
        .trim()
        .to_string();
    try_apply_time_format(&replaced, export_start_time)
}

fn normalize_export_relative_path(path: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for segment in path.split(['/', '\\']) {
        let cleaned = sanitize_file_name_segment(segment);
        if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
            continue;
        }
        out.push(cleaned);
    }
    out
}

fn normalize_path_key(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    if cfg!(target_os = "windows") {
        normalized.to_ascii_lowercase()
    } else {
        normalized
    }
}

fn persist_successful_export_settings(
    state: &AppState,
    project_output_dir: Option<String>,
    project_file_name: Option<String>,
    separated_output_dir: Option<String>,
    separated_file_name_pattern: Option<String>,
    sample_rate: Option<u32>,
    bit_depth: Option<u32>,
) {
    let Some(config_dir) = state.config_dir.get() else {
        return;
    };

    let mut settings = crate::config::load_export_settings(config_dir);

    if let Some(value) = project_output_dir {
        settings.project_output_dir = Some(value);
    }
    if let Some(value) = project_file_name {
        settings.project_file_name = Some(value);
    }
    if let Some(value) = separated_output_dir {
        settings.separated_output_dir = Some(value);
    }
    if let Some(value) = separated_file_name_pattern {
        settings.separated_file_name_pattern = Some(value);
    }
    if let Some(value) = sample_rate {
        settings.sample_rate = normalize_export_sample_rate(value);
    }
    if let Some(value) = bit_depth {
        settings.bit_depth = normalize_export_bit_depth(value);
    }

    crate::config::save_export_settings(config_dir, &settings);
}

fn build_unique_export_file_name(
    pattern: &str,
    track_index: usize,
    export_index: usize,
    index_width: usize,
    track_name: &str,
    track_type: &str,
    track_id: &str,
    project_name: &str,
    export_time: chrono::DateTime<Local>,
    used_names: &mut HashSet<String>,
) -> Result<PathBuf, String> {
    let mut rendered = pattern.to_string();
    let index_token = format!("{:0width$}", track_index, width = index_width);
    let export_index_token = format!("{:0width$}", export_index, width = index_width);
    rendered = rendered.replace("<TrackIndex>", &index_token);
    rendered = rendered.replace("<ExportIndex>", &export_index_token);
    rendered = rendered.replace("<TrackName>", track_name);
    rendered = rendered.replace("<TrackType>", track_type);
    rendered = rendered.replace("<TrackId>", track_id);
    rendered = rendered.replace("<ProjectName>", project_name);
    rendered = try_apply_time_format(&rendered, export_time)?;

    let mut relative = normalize_export_relative_path(&rendered);
    if relative.as_os_str().is_empty() {
        relative.push(format!(
            "{}_{}.wav",
            export_index_token,
            sanitize_file_name_segment(track_name)
        ));
    }
    if relative.extension().and_then(|ext| ext.to_str()).map(|ext| ext.eq_ignore_ascii_case("wav")) != Some(true) {
        let stem = relative
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("track")
            .to_string();
        let mut new_file_name = sanitize_file_name_segment(&stem);
        if new_file_name.is_empty() {
            new_file_name = format!("{}_{}.wav", export_index_token, sanitize_file_name_segment(track_name));
        } else if !new_file_name.to_ascii_lowercase().ends_with(".wav") {
            new_file_name.push_str(".wav");
        }
        relative.set_file_name(new_file_name);
    }

    let candidate = relative.to_string_lossy().to_string();
    if used_names.insert(candidate.clone()) {
        return Ok(PathBuf::from(candidate));
    }

    let parent = PathBuf::from(&candidate)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let file_name = PathBuf::from(&candidate)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("track.wav")
        .to_string();
    let lower = file_name.to_ascii_lowercase();
    let stem = if lower.ends_with(".wav") {
        file_name[..file_name.len() - 4].to_string()
    } else {
        file_name
    };

    let mut seq = 2usize;
    loop {
        let file = format!("{}_{}.wav", stem, seq);
        let next = if parent.as_os_str().is_empty() {
            file
        } else {
            parent.join(file).to_string_lossy().to_string()
        };
        if used_names.insert(next.clone()) {
            return Ok(PathBuf::from(next));
        }
        seq += 1;
    }
}
