use crate::project::{
    load_project_file, prepare_timeline_for_project_save, project_name_from_path,
    read_project_file_version, resolve_source_paths_on_open, serialize_project_file_for_path,
    CustomScale, ProjectFile, CURRENT_PROJECT_FILE_VERSION,
};
use crate::state::AppState;
use crate::synth_clip_cache;
use crate::time_stretch::{
    update_project_stretch_overrides, update_runtime_stretch_settings, UserStretchAlgorithm,
};
use chrono::Local;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::{Manager, State, Window};
use zip::write::FileOptions;

fn normalize_scale_key(raw: &str) -> String {
    const SCALE_KEYS: [&str; 12] = [
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
    ];
    if SCALE_KEYS.contains(&raw) {
        return raw.to_string();
    }
    "C".to_string()
}

fn normalize_custom_scale(input: Option<CustomScale>) -> Option<CustomScale> {
    input.map(|s| s.normalized())
}

fn base_scale_notes(scale: &str) -> Vec<u8> {
    match normalize_scale_key(scale).as_str() {
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
        _ => vec![0, 2, 4, 5, 7, 9, 11],
    }
}

fn effective_scale_notes(
    base_scale: &str,
    use_custom_scale: bool,
    custom_scale: Option<&CustomScale>,
) -> Vec<u8> {
    if use_custom_scale {
        if let Some(custom) = custom_scale {
            let normalized = custom.normalized();
            if !normalized.notes.is_empty() {
                return normalized.notes;
            }
        }
    }
    base_scale_notes(base_scale)
}

fn normalize_beats_per_bar(raw: u32) -> u32 {
    raw.clamp(1, 32)
}

fn normalize_grid_size(raw: &str) -> String {
    const VALID: [&str; 21] = [
        "1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/1d", "1/2d", "1/4d", "1/8d",
        "1/16d", "1/32d", "1/64d", "1/1t", "1/2t", "1/4t", "1/8t", "1/16t", "1/32t", "1/64t",
    ];
    if VALID.contains(&raw) {
        return raw.to_string();
    }
    "1/4".to_string()
}

use super::common::ok_bool;
use super::core::{get_timeline_state, get_timeline_state_from_ref};

fn update_window_title(window: &Window, name: &str, dirty: bool) {
    let suffix = if dirty { "*" } else { "" };
    let title = format!("HiFiShifter - {}{}", name, suffix);
    let _ = window.set_title(&title);
}

fn latest_clip_end_sec(timeline: &crate::state::TimelineState) -> f64 {
    timeline
        .clips
        .iter()
        .map(|clip| (clip.start_sec + clip.length_sec).max(0.0))
        .fold(0.0_f64, f64::max)
}

fn is_zip_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("zip"))
        .unwrap_or(false)
}

fn save_recent_projects(state: &AppState) {
    let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(dir) = state.config_dir.get() {
        crate::config::save_recent(dir, &p.recent);
    }
}

fn load_auto_backup_settings(state: &AppState) -> crate::config::AutoBackupSettings {
    if let Some(config_dir) = state.config_dir.get() {
        return crate::config::load_auto_backup_settings(config_dir);
    }
    crate::config::AutoBackupSettings::default()
}

fn load_ui_stretch_defaults(state: &AppState) -> (UserStretchAlgorithm, bool) {
    if let Some(config_dir) = state.config_dir.get() {
        let settings = crate::config::load_ui_settings(config_dir);
        return (
            settings.default_stretch_algorithm,
            settings.default_hifigan_mel_stretch,
        );
    }
    (UserStretchAlgorithm::default(), true)
}

fn sync_runtime_stretch_settings(state: &AppState) {
    let (default_algorithm, default_hifigan_mel_stretch) = load_ui_stretch_defaults(state);
    let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
    update_runtime_stretch_settings(
        default_algorithm,
        default_hifigan_mel_stretch,
        project.stretch_algorithm_override,
        project.hifigan_mel_stretch_override,
    );
}

fn is_hifishifter_project_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "hshp" | "hsp"))
        .unwrap_or(false)
}

fn save_on_save_backup_path(path: &Path) -> PathBuf {
    // 保留原文件名与扩展名，在其后追加 "-bak"，例如：
    // "project.hshp" -> "project.hshp-bak"
    // "project.hsp"  -> "project.hsp-bak"
    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
        let backup_name = format!("{}-bak", file_name);
        if let Some(parent) = path.parent() {
            return parent.join(backup_name);
        } else {
            return PathBuf::from(backup_name);
        }
    }
    // fallback: 保持兼容旧实现
    path.with_extension("hshp-bak")
}

fn rotate_existing_project_file_for_backup(path: &Path) -> Result<Option<PathBuf>, String> {
    if !path.exists() {
        return Ok(None);
    }

    let backup_path = save_on_save_backup_path(path);
    if backup_path.exists() {
        fs::remove_file(&backup_path).map_err(|e| {
            format!(
                "Failed to remove stale save backup {:?}: {}",
                backup_path, e
            )
        })?;
    }

    fs::rename(path, &backup_path).map_err(|e| {
        format!(
            "Failed to rotate existing project file {:?} to backup {:?}: {}",
            path, backup_path, e
        )
    })?;

    Ok(Some(backup_path))
}

fn restore_rotated_project_backup(backup_path: Option<&PathBuf>, project_path: &Path) {
    let Some(backup_path) = backup_path else {
        return;
    };

    if backup_path.exists() && !project_path.exists() {
        let _ = fs::rename(backup_path, project_path);
    }
}

pub(crate) fn sanitize_file_name_segment(raw: &str) -> String {
    raw.chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        })
        .collect::<String>()
        .trim()
        .to_string()
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

pub(crate) fn resolve_project_folder_for_backup(state: &AppState) -> PathBuf {
    let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(path) = project.path.as_deref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            return parent.to_path_buf();
        }
    }
    resolve_documents_dir(state).unwrap_or_else(|| PathBuf::from("."))
}

pub(crate) fn resolve_project_name_for_backup(state: &AppState) -> String {
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

pub(crate) fn try_apply_time_format_with_fallback(
    template: &str,
    time: chrono::DateTime<Local>,
) -> Result<(String, bool), String> {
    let direct = std::panic::catch_unwind(|| time.format(template).to_string());
    if let Ok(value) = direct {
        return Ok((value, false));
    }

    let escaped = template.replace('%', "%%");
    let escaped_try = std::panic::catch_unwind(|| time.format(&escaped).to_string());
    if let Ok(value) = escaped_try {
        return Ok((value, true));
    }

    Err("auto_backup_invalid_time_format".to_string())
}

fn resolve_timed_backup_output_path(
    state: &AppState,
    template: &str,
    now: chrono::DateTime<Local>,
) -> Result<(PathBuf, bool), String> {
    let normalized_template = if template.trim().is_empty() {
        crate::config::AutoBackupSettings::default().timed_backup_path_template
    } else {
        template.trim().to_string()
    };

    let project_folder = resolve_project_folder_for_backup(state)
        .display()
        .to_string();
    let project_name = resolve_project_name_for_backup(state);

    let replaced = normalized_template
        .replace("<ProjectFolder>", &project_folder)
        .replace("<ProjectName>", &project_name);

    let (formatted, fallback_used) = try_apply_time_format_with_fallback(&replaced, now)?;
    let trimmed = formatted.trim();
    if trimmed.is_empty() {
        return Err("timed_backup_path_resolves_to_empty".to_string());
    }

    let mut output_path = PathBuf::from(trimmed);
    if output_path.is_relative() {
        output_path = resolve_project_folder_for_backup(state).join(output_path);
    }

    Ok((output_path, fallback_used))
}

fn atomic_write_project_snapshot_to_path(
    state: &AppState,
    output_path: &Path,
) -> Result<(), String> {
    let output_name = project_name_from_path(output_path);
    let project_file = build_project_file_snapshot(state, output_path, &output_name);
    let bytes = serialize_project_file_for_path(&project_file, output_path)?;

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create backup directory {:?}: {}", parent, e))?;
        }
    }

    let tmp_path = {
        let ext = output_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if ext.is_empty() {
            output_path.with_extension("tmp_save")
        } else {
            output_path.with_extension(format!("{}.tmp_save", ext))
        }
    };

    let old_path = {
        let ext = output_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if ext.is_empty() {
            output_path.with_extension("old")
        } else {
            output_path.with_extension(format!("{}.old", ext))
        }
    };

    fs::write(&tmp_path, &bytes)
        .map_err(|e| format!("Failed to write temporary backup {:?}: {}", tmp_path, e))?;

    let had_existing_output = output_path.exists();
    if had_existing_output {
        if old_path.exists() {
            fs::remove_file(&old_path).map_err(|e| {
                format!(
                    "Failed to remove stale rollback backup {:?}: {}",
                    old_path, e
                )
            })?;
        }

        fs::rename(output_path, &old_path).map_err(|e| {
            let _ = fs::remove_file(&tmp_path);
            format!(
                "Failed to move existing backup {:?} to rollback path {:?}: {}",
                output_path, old_path, e
            )
        })?;
    }

    if let Err(e) = fs::rename(&tmp_path, output_path) {
        let _ = fs::remove_file(&tmp_path);

        if had_existing_output {
            let _ = fs::rename(&old_path, output_path);
        }

        return Err(format!(
            "Failed to move temporary backup {:?} to {:?}: {}",
            tmp_path, output_path, e
        ));
    }

    if had_existing_output && old_path.exists() {
        let _ = fs::remove_file(&old_path);
    }

    Ok(())
}

fn build_project_file_snapshot(
    state: &AppState,
    project_path: &Path,
    project_name: &str,
) -> ProjectFile {
    let mut tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    tl.project_sec = latest_clip_end_sec(&tl).max(4.0).ceil();

    let (
        base_scale,
        use_custom_scale,
        custom_scale,
        beats_per_bar,
        time_signature_denominator,
        grid_size,
        notes_markdown,
        stretch_algorithm_override,
        hifigan_mel_stretch_override,
    ) = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        (
            normalize_scale_key(&p.base_scale),
            p.use_custom_scale,
            normalize_custom_scale(p.custom_scale.clone()),
            normalize_beats_per_bar(p.beats_per_bar),
            if matches!(p.time_signature_denominator, 1 | 2 | 4 | 8 | 16 | 32) {
                p.time_signature_denominator
            } else {
                4
            },
            normalize_grid_size(&p.grid_size),
            p.notes_markdown.clone(),
            p.stretch_algorithm_override,
            p.hifigan_mel_stretch_override,
        )
    };

    let tl_saved = prepare_timeline_for_project_save(tl, project_path);
    let mut pf = ProjectFile::new(
        project_name.to_string(),
        tl_saved,
        base_scale,
        beats_per_bar,
        time_signature_denominator,
        grid_size,
    );
    pf.use_custom_scale = use_custom_scale && custom_scale.is_some();
    pf.custom_scale = custom_scale;
    pf.notes_markdown = notes_markdown;
    pf.synth_config.stretch_algorithm_override = stretch_algorithm_override;
    pf.synth_config.hifigan_mel_stretch_override = hifigan_mel_stretch_override;
    pf
}

fn unique_entry_path(desired: &str, used_paths: &mut std::collections::HashSet<String>) -> String {
    let path = Path::new(desired);
    let parent = path
        .parent()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let stem = path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("file")
        .to_string();
    let ext = path.extension().and_then(|v| v.to_str()).unwrap_or("");

    let mk = |index: usize| -> String {
        let filename = if index == 0 {
            if ext.is_empty() {
                stem.clone()
            } else {
                format!("{}.{}", stem, ext)
            }
        } else if ext.is_empty() {
            format!("{} ({})", stem, index)
        } else {
            format!("{} ({}).{}", stem, index, ext)
        };
        if parent.is_empty() {
            filename
        } else {
            format!("{}/{}", parent.trim_end_matches('/'), filename)
        }
    };

    let mut idx = 0usize;
    loop {
        let candidate = mk(idx);
        if used_paths.insert(candidate.clone()) {
            return candidate;
        }
        idx += 1;
    }
}

fn save_project_archive_to_zip_inner(
    state: &AppState,
    zip_path: &Path,
) -> Result<crate::models::TimelineStatePayload, String> {
    let project_name = project_name_from_path(zip_path);
    let project_entry_name = format!("{}.hshp", project_name);
    let archive_project_virtual_path = PathBuf::from(&project_entry_name);

    let mut pf = build_project_file_snapshot(state, &archive_project_virtual_path, &project_name);

    let current_project_dir = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.path
            .as_deref()
            .map(PathBuf::from)
            .and_then(|v| v.parent().map(|x| x.to_path_buf()))
    };

    let mut used_zip_paths = std::collections::HashSet::<String>::new();
    used_zip_paths.insert(project_entry_name.clone());

    let mut source_to_entry = std::collections::HashMap::<String, String>::new();
    let mut archive_logs: Vec<String> = Vec::new();
    archive_logs.push(format!(
        "Archive started at {}",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    ));
    archive_logs.push(format!("Target zip: {}", zip_path.display()));
    archive_logs.push(format!("Embedded project file: {}", project_entry_name));

    pf.timeline.sync_clip_takes_from_flat();
    for clip in pf.timeline.clips.iter_mut() {
        for take in &mut clip.takes {
            let Some(source_path) = take.source_path.clone() else {
                continue;
            };
            if source_path.trim().is_empty() {
                take.source_path = None;
                continue;
            }

            if let Some(existing) = source_to_entry.get(&source_path) {
                take.source_path_relative = Some(existing.clone());
                take.source_path = None;
                continue;
            }

            let abs_path = PathBuf::from(&source_path);
            if !abs_path.is_absolute() || !abs_path.exists() {
                archive_logs.push(format!(
                    "Skip missing or non-absolute source: {} (clip={}, take={})",
                    source_path, clip.id, take.id
                ));
                take.source_path = None;
                continue;
            }

            let relative_candidate = current_project_dir
                .as_ref()
                .and_then(|base_dir| abs_path.strip_prefix(base_dir).ok())
                .map(|p| p.to_string_lossy().replace('\\', "/"));

            let desired_entry_path = if let Some(rel) = relative_candidate {
                rel
            } else {
                let file_name = abs_path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("audio.wav");
                format!("Archived/{}", file_name)
            };

            let unique_entry = unique_entry_path(&desired_entry_path, &mut used_zip_paths);
            if unique_entry != desired_entry_path {
                archive_logs.push(format!(
                    "Name collision resolved: {} -> {}",
                    desired_entry_path, unique_entry
                ));
            }

            source_to_entry.insert(source_path.clone(), unique_entry.clone());
            take.source_path_relative = Some(unique_entry.clone());
            take.source_path = None;
            archive_logs.push(format!(
                "Archive source: {} -> {}",
                source_path, unique_entry
            ));
        }
        let active_take = clip
            .active_take_id
            .as_deref()
            .and_then(|id| clip.takes.iter().find(|t| t.id == id))
            .or_else(|| clip.takes.first())
            .cloned();
        if let Some(take) = active_take {
            take.apply_to_clip(clip);
        }
    }

    let bytes = serialize_project_file_for_path(&pf, Path::new(&project_entry_name))?;

    // 为了保证保存的原子性，先写入临时文件，成功后再重命名为最终路径。
    let tmp_path = {
        let ext = zip_path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext.is_empty() {
            // 没有扩展名的情况，直接追加后缀
            zip_path.with_extension("tmp_save")
        } else {
            // 保留原有扩展名，并追加 .tmp_save 后缀，例如 .zip.tmp_save
            let new_ext = format!("{}.tmp_save", ext);
            zip_path.with_extension(new_ext)
        }
    };

    let write_result: Result<(), String> = (|| {
        let file = fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipWriter::new(file);
        let options = FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        zip.start_file(project_entry_name.clone(), options)
            .map_err(|e| e.to_string())?;
        zip.write_all(&bytes).map_err(|e| e.to_string())?;

        let mut written_entries = std::collections::HashSet::<String>::new();
        for (source_path, zip_entry) in &source_to_entry {
            if !written_entries.insert(zip_entry.clone()) {
                continue;
            }
            // 使用流式写入，避免将整个文件读入内存。
            let mut src_file = fs::File::open(source_path).map_err(|e| e.to_string())?;
            zip.start_file(zip_entry, options)
                .map_err(|e| e.to_string())?;
            std::io::copy(&mut src_file, &mut zip).map_err(|e| e.to_string())?;
        }

        let log_name = format!(
            "{}_{}.log",
            project_name,
            Local::now().format("%Y%m%d_%H%M%S")
        );
        archive_logs.push(format!(
            "Archive completed at {}",
            Local::now().format("%Y-%m-%d %H:%M:%S")
        ));
        zip.start_file(log_name.clone(), options)
            .map_err(|e| e.to_string())?;
        let mut log_text = archive_logs.join("\n");
        log_text.push('\n');
        zip.write_all(log_text.as_bytes())
            .map_err(|e| e.to_string())?;

        // 确保 ZipWriter 正常完成写入并刷新到底层文件。
        zip.finish().map_err(|e| e.to_string())?;
        Ok(())
    })();

    match write_result {
        Ok(()) => {
            // 写入成功后，用可回滚的方式替换最终 zip 文件，兼容 Windows 上目标已存在时
            // `fs::rename` 不能直接覆盖的问题。
            let backup_path = {
                let file_name = zip_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("archive.zip");
                zip_path.with_file_name(format!("{file_name}.replace_backup"))
            };

            let destination_existed = zip_path.exists();
            let mut backup_created = false;

            if destination_existed {
                if backup_path.exists() {
                    fs::remove_file(&backup_path).map_err(|e| {
                        format!(
                            "Failed to remove stale archive backup {:?}: {}",
                            backup_path, e
                        )
                    })?;
                }

                fs::rename(zip_path, &backup_path).map_err(|e| {
                    format!(
                        "Failed to move existing archive {:?} to backup {:?}: {}",
                        zip_path, backup_path, e
                    )
                })?;
                backup_created = true;
            }

            if let Err(rename_err) = fs::rename(&tmp_path, zip_path) {
                let _ = fs::remove_file(&tmp_path);

                if backup_created {
                    let _ = fs::remove_file(zip_path);
                    let _ = fs::rename(&backup_path, zip_path);
                }

                return Err(format!(
                    "Failed to replace archive {:?} with temporary file {:?}: {}",
                    zip_path, tmp_path, rename_err
                ));
            }

            if backup_created {
                fs::remove_file(&backup_path).map_err(|e| {
                    format!(
                        "Archive replaced, but failed to remove backup {:?}: {}",
                        backup_path, e
                    )
                })?;
            }
        }
        Err(e) => {
            // 写入失败，尝试清理临时文件，然后返回原始错误。
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
    }

    Ok(get_timeline_state_from_ref(state))
}

pub(crate) fn save_project_to_path_inner(
    state: &AppState,
    window: &Window,
    project_path: String,
) -> Result<crate::models::TimelineStatePayload, String> {
    let path = PathBuf::from(&project_path);
    let name = project_name_from_path(&path);
    let pf = build_project_file_snapshot(state, &path, &name);
    let bytes = serialize_project_file_for_path(&pf, &path)?;

    let auto_backup_settings = load_auto_backup_settings(state);
    let rotated_backup =
        if auto_backup_settings.save_on_save_enabled && is_hifishifter_project_path(&path) {
            rotate_existing_project_file_for_backup(&path)?
        } else {
            None
        };

    // 使用原子保存，防止程序崩溃或断电导致工程文件损坏
    let tmp_path = path.with_extension("tmp_save");
    let write_result: Result<(), String> = (|| {
        fs::write(&tmp_path, &bytes).map_err(|e| e.to_string())?;
        fs::rename(&tmp_path, &path).map_err(|e| {
            let _ = fs::remove_file(&tmp_path);
            e.to_string()
        })?;
        Ok(())
    })();

    if let Err(err) = write_result {
        restore_rotated_project_backup(rotated_backup.as_ref(), &path);
        return Err(err);
    }

    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.name = name;
        p.path = Some(project_path.clone());
        p.dirty = false;
        p.recent.retain(|x| x != &project_path);
        p.recent.insert(0, project_path.clone());
        if p.recent.len() > 10 {
            p.recent.truncate(10);
        }
        update_window_title(window, &p.name, p.dirty);
    }

    // 持久化最近工程列表
    save_recent_projects(state);

    Ok(get_timeline_state_from_ref(state))
}

pub(super) fn get_project_meta(state: State<'_, AppState>) -> crate::models::ProjectMetaPayload {
    state.project_meta_payload()
}

pub(super) fn get_auto_backup_settings(
    state: State<'_, AppState>,
) -> crate::config::AutoBackupSettings {
    load_auto_backup_settings(state.inner())
}

pub(super) fn save_auto_backup_settings(
    state: State<'_, AppState>,
    settings: crate::config::AutoBackupSettings,
) -> serde_json::Value {
    let normalized = settings.normalized();
    if let Some(config_dir) = state.config_dir.get() {
        crate::config::save_auto_backup_settings(config_dir, &normalized);
    }
    serde_json::json!({ "ok": true, "settings": normalized })
}

pub(super) fn run_timed_auto_backup(
    state: State<'_, AppState>,
    path_template: String,
) -> serde_json::Value {
    let now = Local::now();
    let (mut output_path, fallback_used) =
        match resolve_timed_backup_output_path(state.inner(), &path_template, now) {
            Ok(value) => value,
            Err(error) => {
                return serde_json::json!({ "ok": false, "error": error });
            }
        };

    if output_path.extension().is_none() {
        output_path.set_extension("hshp");
    }

    match atomic_write_project_snapshot_to_path(state.inner(), &output_path) {
        Ok(()) => serde_json::json!({
            "ok": true,
            "path": output_path.display().to_string(),
            "formatFallbackApplied": fallback_used,
        }),
        Err(error) => serde_json::json!({ "ok": false, "error": error }),
    }
}

pub(super) fn new_project(
    state: State<'_, AppState>,
    window: Window,
) -> crate::models::TimelineStatePayload {
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        *tl = crate::state::TimelineState::default();
        state.audio_engine.update_timeline(tl.clone());
    }
    state.clear_history();
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.name = "Untitled".to_string();
        p.path = None;
        p.dirty = false;
        p.notes_markdown = String::new();
        p.base_scale = "C".to_string();
        p.use_custom_scale = false;
        p.custom_scale = None;
        p.beats_per_bar = 4;
        p.grid_size = "1/4".to_string();
        p.stretch_algorithm_override = None;
        p.hifigan_mel_stretch_override = None;
    }
    sync_runtime_stretch_settings(state.inner());
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.project_scale_notes = base_scale_notes("C");
        state.audio_engine.update_timeline(tl.clone());
    }
    update_window_title(&window, "Untitled", false);
    get_timeline_state(state)
}

pub(super) fn open_project_dialog() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("HiFiShifter Project", &["hshp", "hsp"])
        .add_filter("JSON Project", &["json"])
        .pick_file();
    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}

pub(super) fn open_project(
    state: State<'_, AppState>,
    window: Window,
    project_path: String,
    force: Option<bool>,
) -> crate::models::OpenProjectPayload {
    let path = PathBuf::from(&project_path);
    // 读取字节流，自动检测 MessagePack（v3）或 JSON（v1/v2 兼容）格式。
    // 读取失败不再静默当作空文件：把 io 错误带回给前端展示。
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            let mut payload = get_timeline_state(state);
            payload.ok = false;
            return crate::models::OpenProjectPayload {
                timeline: payload,
                error: Some(format!("failed to read project file: {e}")),
                project_version_too_new: None,
                project_file_version: None,
                current_project_file_version: None,
            };
        }
    };
    // 先只读取版本号：即使未来版本工程因结构变化而无法完整解析，
    // 也能在尝试加载前给出明确的“可能不兼容”警告。
    let project_file_version = read_project_file_version(&bytes).unwrap_or(0);
    if project_file_version > CURRENT_PROJECT_FILE_VERSION && !force.unwrap_or(false) {
        let mut payload = get_timeline_state(state);
        payload.ok = true;
        return crate::models::OpenProjectPayload {
            timeline: payload,
            error: None,
            project_version_too_new: Some(true),
            project_file_version: Some(project_file_version),
            current_project_file_version: Some(CURRENT_PROJECT_FILE_VERSION),
        };
    }

    let parsed = load_project_file(&bytes);
    let Ok(mut pf) = parsed else {
        let parse_error = parsed
            .err()
            .map(|e| format!("failed to parse project file: {e}"))
            .unwrap_or_else(|| "failed to parse project file".to_string());
        let mut payload = get_timeline_state(state);
        payload.ok = false;
        return crate::models::OpenProjectPayload {
            timeline: payload,
            error: Some(parse_error),
            project_version_too_new: None,
            project_file_version: None,
            current_project_file_version: None,
        };
    };

    let (resolved_timeline, missing_files) = resolve_source_paths_on_open(pf.timeline, &path);
    pf.timeline = resolved_timeline;
    // 旧项目兼容迁移（仅 v3 及更早）：source_end_sec == 0.0 曾表示"到源文件
    // 末尾"，新语义要求它是真实的结束时间，此处自动修正为 duration_sec 或
    // length_sec。v4+ 工程的 se 恒为真实坐标（可为 0/负值，如倒放静音段），
    // 不得改写。
    if pf.version < 4 {
        for clip in &mut pf.timeline.clips {
            if clip.source_end_sec == 0.0 {
                clip.source_end_sec = clip.duration_sec.unwrap_or(clip.length_sec);
            }
        }
    }
    // v4 迁移：v3 及更早的工程 Clip 不携带 loop_enabled（Loop / 循环源）字段，
    // 按当前"为新的音频块启用循环"设置作为这些既有**音频** Clip 的 Loop 属性；
    // 绝不改动已显式携带该字段的 v4+ 工程。
    // 纯 MIDI / 音高参考块（无源媒体路径）不参与 Loop 迁移 —— 与各格式导入器
    // 显式创建 `loop_enabled=false` 的 MIDI 块约定保持一致（REAPER 的 LOOP
    // 语义只作用于音频 item）。
    if pf.version < 4 {
        let default_loop = crate::config::loop_new_clips_default();
        for clip in &mut pf.timeline.clips {
            if clip.source_path.is_some() {
                clip.loop_enabled = default_loop;
            }
        }
    }
    // 非 Loop 存储窗口规范化（对**所有版本**生效）：使存储字段 == 消费
    // 窗口（正放 se:=ss+len·r；倒放 ss:=se−len·r），与消费端派生值一致、
    // 功能零变化 —— 用于自愈历史版本写入的陈旧/发散源窗口。
    // 同一不变式也应用到全部 take（组合速率口径），避免 inactive take 的
    // 陈旧窗口流向前端 take-lane 显示与 REAPER 导出。
    for clip in &mut pf.timeline.clips {
        crate::state::normalize_nonloop_source_window(clip);
        crate::state::normalize_nonloop_all_take_windows(clip);
    }
    // 迁移/规范化都发生在 active take 内存投影上，写回 Take 权威数据。
    pf.timeline.sync_clip_takes_from_flat();

    // 打开工程时清除所有渲染缓存，确保旧的预渲染结果不会影响新的播放。
    // 这是修复"音高分析未完成时播放导致音高编辑不生效"问题的关键步骤。
    eprintln!("[open_project] Clearing all render caches before loading project...");
    // hnsep 分离缓存键只含 clip_id+采样率+样本数：换工程后同 id/等长 clip
    // 会命中上一个工程的 stems，必须一并清空（低频操作，整体清空可接受）。
    crate::hnsep_onnx::clear_separation_cache();
    for clip in &pf.timeline.clips {
        synth_clip_cache::invalidate_clip_all_caches(&clip.id);
    }

    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        *tl = pf.timeline.clone();
        // 规范化 Tempo Map（排序/钳制/补 0 位置点），并同步工程基准 BPM。
        tl.normalize_tempo_map();
        if let Some(points) = tl.tempo_map.as_ref() {
            if let Some(first) = points.first() {
                tl.bpm = first.bpm.clamp(10.0, 960.0);
            }
        }
        // 打开工程时为所有含 source_path 的 clip 初始化文件元数据 + 内容指纹，
        // 用于本会话中的外部文件变更检测。此数据仅在程序运行期间有效，不持久化。
        for clip in &mut tl.clips {
            crate::state::TimelineState::populate_clip_file_metadata(clip);
        }
        tl.sync_clip_takes_from_flat();
        let normalized_base_scale = normalize_scale_key(&pf.base_scale);
        let normalized_custom_scale = normalize_custom_scale(pf.custom_scale.clone());
        let normalized_use_custom_scale = pf.use_custom_scale && normalized_custom_scale.is_some();
        tl.project_scale_notes = effective_scale_notes(
            &normalized_base_scale,
            normalized_use_custom_scale,
            normalized_custom_scale.as_ref(),
        );
        state.audio_engine.update_timeline(tl.clone());
    }
    // Tempo Map 存在时，工程基准拍号同步为 0 位置点的拍号（分子/分母）。
    let tempo_map_initial_signature = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .tempo_map
        .as_ref()
        .and_then(|points| points.first())
        .map(|first| (first.numerator, first.denominator));
    state.clear_history();
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.name = project_name_from_path(&path);
        p.path = Some(project_path.clone());
        p.dirty = false;
        p.notes_markdown = pf.notes_markdown;
        p.base_scale = normalize_scale_key(&pf.base_scale);
        p.custom_scale = normalize_custom_scale(pf.custom_scale);
        p.use_custom_scale = pf.use_custom_scale && p.custom_scale.is_some();
        p.beats_per_bar = normalize_beats_per_bar(pf.beats_per_bar);
        p.time_signature_denominator =
            if matches!(pf.time_signature_denominator, 1 | 2 | 4 | 8 | 16 | 32) {
                pf.time_signature_denominator
            } else {
                4
            };
        if let Some((initial_numerator, initial_denominator)) = tempo_map_initial_signature {
            p.beats_per_bar = initial_numerator.unwrap_or(4).clamp(1, 32);
            p.time_signature_denominator = initial_denominator.unwrap_or(4);
        }
        p.grid_size = normalize_grid_size(&pf.grid_size);
        p.stretch_algorithm_override = pf.synth_config.stretch_algorithm_override;
        p.hifigan_mel_stretch_override = pf.synth_config.hifigan_mel_stretch_override;
        // recent list (in-memory)
        p.recent.retain(|x| x != &project_path);
        p.recent.insert(0, project_path.clone());
        if p.recent.len() > 10 {
            p.recent.truncate(10);
        }
        update_window_title(&window, &p.name, p.dirty);
    }
    // 防御性修复旧版本工程文件可能存在的“工程音阶与 Tempo Map 初始点分叉”
    // （早期撤销路径不回写工程记录，保存的文件可能带有不一致的 base_scale）：
    // 初始点即工程基准记录，加载后以它为准同步工程记录（含 BPM/拍号/音阶）。
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        state.sync_project_record_from_tempo_map(&mut tl, &mut p);
    }
    sync_runtime_stretch_settings(state.inner());
    if let Some(handle) = state.app_handle.get() {
        crate::commands::playback::request_background_render(handle);
    }

    // 持久化最近工程列表
    save_recent_projects(state.inner());

    let mut payload = get_timeline_state(state);
    if !missing_files.is_empty() {
        payload.missing_files = Some(missing_files);
    }
    crate::models::OpenProjectPayload {
        timeline: payload,
        error: None,
        project_version_too_new: None,
        project_file_version: None,
        current_project_file_version: None,
    }
}

pub(super) fn save_project(
    state: State<'_, AppState>,
    window: Window,
    notes_markdown: Option<String>,
) -> serde_json::Value {
    if let Some(notes_markdown) = notes_markdown {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notes_markdown = notes_markdown;
    }
    let existing_path = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.path.clone()
    };
    if let Some(path) = existing_path {
        return save_project_to_path(state, window, path, None, None);
    }
    // No path yet -> Save As
    save_project_as(state, window, None)
}

pub(super) fn save_project_as(
    state: State<'_, AppState>,
    window: Window,
    notes_markdown: Option<String>,
) -> serde_json::Value {
    if let Some(notes_markdown) = notes_markdown {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notes_markdown = notes_markdown;
    }
    let default_name = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        if p.name.trim().is_empty() {
            "Untitled".to_string()
        } else {
            p.name.clone()
        }
    };
    let picked = rfd::FileDialog::new()
        .add_filter("HiFiShifter Project", &["hshp", "hsp"])
        .add_filter("JSON Project", &["json"])
        .add_filter("Archive Zip", &["zip"])
        .set_file_name(format!("{}.hshp", default_name))
        .save_file();
    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => save_project_to_path(state, window, path.display().to_string(), None, None),
    }
}

/// 读取目标路径已有 HiFiShifter 工程文件的版本号（仅当目标确实是工程文件时）。
///
/// - 目标不存在、不是工程文件或读取失败时返回 `None`（无需覆盖警告）。
/// - ZIP 归档是打包容器而非直接的工程文件，跳过检查。
fn detect_save_target_version_conflict(path: &Path) -> Option<u32> {
    if !path.exists() {
        return None;
    }
    if is_zip_path(path) {
        return None;
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    if !matches!(ext.as_deref(), Some("hshp") | Some("hsp") | Some("json")) {
        return None;
    }
    let bytes = fs::read(path).unwrap_or_default();
    if bytes.is_empty() {
        return None;
    }
    read_project_file_version(&bytes)
}

/// 保存到指定路径；若目标已存在版本不同的 HiFiShifter 工程文件，先返回
/// 版本冲突信号而不直接覆盖，由前端弹出确认窗口（force=true 表示已确认）。
pub(super) fn save_project_to_path(
    state: State<'_, AppState>,
    window: Window,
    project_path: String,
    notes_markdown: Option<String>,
    force: Option<bool>,
) -> serde_json::Value {
    if let Some(notes_markdown) = notes_markdown {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notes_markdown = notes_markdown;
    }
    let path = PathBuf::from(&project_path);

    if !force.unwrap_or(false) {
        if let Some(existing_version) = detect_save_target_version_conflict(&path) {
            if existing_version != CURRENT_PROJECT_FILE_VERSION {
                return serde_json::json!({
                    "ok": false,
                    "canceled": false,
                    "versionConflict": true,
                    "path": project_path,
                    "existingVersion": existing_version,
                    "currentVersion": CURRENT_PROJECT_FILE_VERSION,
                    "existingIsNewer": existing_version > CURRENT_PROJECT_FILE_VERSION,
                });
            }
        }
    }

    if is_zip_path(&path) {
        match save_project_archive_to_zip_inner(state.inner(), &path) {
            Ok(timeline) => {
                return serde_json::json!({
                    "ok": true,
                    "canceled": false,
                    "path": project_path,
                    "archived": true,
                    "timeline": timeline
                });
            }
            Err(e) => {
                return serde_json::json!({"ok": false, "error": e});
            }
        }
    }

    match save_project_to_path_inner(state.inner(), &window, project_path.clone()) {
        Ok(timeline) => {
            serde_json::json!({"ok": true, "canceled": false, "path": project_path, "timeline": timeline })
        }
        Err(e) => serde_json::json!({"ok": false, "error": e}),
    }
}

pub(super) fn close_window(window: Window) -> serde_json::Value {
    let _ = window.close();
    ok_bool()
}

pub(super) fn set_project_base_scale(
    state: State<'_, AppState>,
    base_scale: String,
) -> serde_json::Value {
    let normalized = normalize_scale_key(&base_scale);
    let (name, changed, was_clean) = {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        if p.base_scale == normalized && !p.use_custom_scale {
            return serde_json::json!({ "ok": true, "base_scale": p.base_scale });
        }
        let was_clean = !p.dirty;
        p.base_scale = normalized.clone();
        p.use_custom_scale = false;
        p.dirty = true;
        (p.name.clone(), true, was_clean)
    };

    if changed && was_clean {
        if let Some(handle) = state.app_handle.get() {
            use tauri::Manager;
            if let Some(win) = handle.get_webview_window("main") {
                let title = format!("HiFiShifter - {}*", name);
                let _ = win.set_title(&title);
            }
        }
    }

    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        // 工程音阶变化会同步到 Tempo Map 初始点：先打撤销快照，
        // 否则该 Tempo Map 变化无法撤销（与 set_timeline_tempo_map 的
        // “工程影响性变化先 checkpoint”约定一致）。
        if tl.tempo_map.is_some() {
            state.checkpoint_timeline(&tl);
        }
        tl.project_scale_notes = base_scale_notes(&normalized);
        // 初始点即工程基准记录：工程音阶变化同步到 Tempo Map 初始点。
        if let Some(points) = tl.tempo_map.as_mut() {
            if let Some(first) = points.first_mut() {
                first.scale = Some(crate::state::TempoScaleData {
                    key: Some(normalized.clone()),
                    name: None,
                    notes: None,
                });
            }
        }
        state.audio_engine.update_timeline(tl.clone());
        // 工程音阶变化影响子轨道“度数差”等依赖音阶的渲染（未被子轨道 Tempo Map 音阶覆盖的区域），
        // 失效所有渲染缓存并触发后台预渲染。
        for clip in &tl.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
    }
    if let Some(handle) = state.app_handle.get() {
        crate::commands::playback::request_background_render(handle);
    }

    let payload = state.project_meta_payload();
    serde_json::json!({ "ok": true, "project": payload })
}

pub(super) fn set_project_custom_scale(
    state: State<'_, AppState>,
    custom_scale: CustomScale,
) -> serde_json::Value {
    let normalized = custom_scale.normalized();
    let (name, changed, was_clean) = {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let changed = p.custom_scale.as_ref().map(|s| (&s.id, &s.name, &s.notes))
            != Some((&normalized.id, &normalized.name, &normalized.notes))
            || !p.use_custom_scale;
        if !changed {
            return serde_json::json!({ "ok": true, "project": state.project_meta_payload() });
        }
        let was_clean = !p.dirty;
        p.custom_scale = Some(normalized.clone());
        p.use_custom_scale = true;
        p.dirty = true;
        (p.name.clone(), true, was_clean)
    };

    if changed && was_clean {
        if let Some(handle) = state.app_handle.get() {
            use tauri::Manager;
            if let Some(win) = handle.get_webview_window("main") {
                let title = format!("HiFiShifter - {}*", name);
                let _ = win.set_title(&title);
            }
        }
    }

    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        // 与 set_project_base_scale 一致：Tempo Map 初始点会被改写，先打撤销快照。
        if tl.tempo_map.is_some() {
            state.checkpoint_timeline(&tl);
        }
        tl.project_scale_notes = normalized.notes.clone();
        // 初始点即工程基准记录：工程音阶变化同步到 Tempo Map 初始点。
        if let Some(points) = tl.tempo_map.as_mut() {
            if let Some(first) = points.first_mut() {
                first.scale = Some(crate::state::TempoScaleData {
                    key: None,
                    name: Some(normalized.name.clone()),
                    notes: Some(normalized.notes.clone()),
                });
            }
        }
        state.audio_engine.update_timeline(tl.clone());
        // 工程音阶变化影响子轨道“度数差”等依赖音阶的渲染（未被子轨道 Tempo Map 音阶覆盖的区域），
        // 失效所有渲染缓存并触发后台预渲染。
        for clip in &tl.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
    }
    if let Some(handle) = state.app_handle.get() {
        crate::commands::playback::request_background_render(handle);
    }

    serde_json::json!({ "ok": true, "project": state.project_meta_payload() })
}

pub(super) fn set_project_timeline_settings(
    state: State<'_, AppState>,
    beats_per_bar: u32,
    time_signature_denominator: u32,
    grid_size: String,
) -> serde_json::Value {
    let normalized_beats = normalize_beats_per_bar(beats_per_bar);
    let normalized_denominator = if matches!(time_signature_denominator, 1 | 2 | 4 | 8 | 16 | 32) {
        time_signature_denominator
    } else {
        4
    };
    let normalized_grid = normalize_grid_size(&grid_size);

    let (name, changed, was_clean) = {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let changed = p.beats_per_bar != normalized_beats
            || p.time_signature_denominator != normalized_denominator
            || p.grid_size != normalized_grid;
        if !changed {
            return serde_json::json!({ "ok": true, "project": state.project_meta_payload() });
        }
        let was_clean = !p.dirty;
        p.beats_per_bar = normalized_beats;
        p.time_signature_denominator = normalized_denominator;
        p.grid_size = normalized_grid;
        p.dirty = true;
        (p.name.clone(), true, was_clean)
    };

    // Tempo Map 存在时，工程基准拍号变化同步到 0 位置点（保持“删除 Tempo Map 后回退一致”）。
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        // 初始点会被改写：先打撤销快照（工程影响性变化，与上面两个命令一致）。
        if tl.tempo_map.is_some() {
            state.checkpoint_timeline(&tl);
        }
        if let Some(points) = tl.tempo_map.as_mut() {
            if let Some(first) = points.first_mut() {
                first.numerator = Some(normalized_beats);
                first.denominator = Some(normalized_denominator);
            }
            state.audio_engine.update_timeline(tl.clone());
        }
    }

    if changed && was_clean {
        if let Some(handle) = state.app_handle.get() {
            use tauri::Manager;
            if let Some(win) = handle.get_webview_window("main") {
                let title = format!("HiFiShifter - {}*", name);
                let _ = win.set_title(&title);
            }
        }
    }

    serde_json::json!({ "ok": true, "project": state.project_meta_payload() })
}

pub(super) fn set_project_stretch_settings(
    state: State<'_, AppState>,
    stretch_algorithm_override: Option<UserStretchAlgorithm>,
    hifigan_mel_stretch_override: Option<bool>,
) -> serde_json::Value {
    let (name, changed, was_clean) = {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let changed = p.stretch_algorithm_override != stretch_algorithm_override
            || p.hifigan_mel_stretch_override != hifigan_mel_stretch_override;
        if !changed {
            return serde_json::json!({ "ok": true, "project": state.project_meta_payload() });
        }
        let was_clean = !p.dirty;
        p.stretch_algorithm_override = stretch_algorithm_override;
        p.hifigan_mel_stretch_override = hifigan_mel_stretch_override;
        p.dirty = true;
        (p.name.clone(), true, was_clean)
    };

    if changed && was_clean {
        if let Some(handle) = state.app_handle.get() {
            use tauri::Manager;
            if let Some(win) = handle.get_webview_window("main") {
                let title = format!("HiFiShifter - {}*", name);
                let _ = win.set_title(&title);
            }
        }
    }

    update_project_stretch_overrides(stretch_algorithm_override, hifigan_mel_stretch_override);
    {
        let timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for clip in &timeline.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        state.audio_engine.update_timeline(timeline);
    }
    if let Some(handle) = state.app_handle.get() {
        crate::commands::playback::request_background_render(handle);
    }
    serde_json::json!({ "ok": true, "project": state.project_meta_payload() })
}
