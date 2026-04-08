use crate::project::{
    load_project_file, prepare_source_paths_for_save, project_name_from_path,
    resolve_source_paths_on_open, serialize_project_file_for_path, CustomScale, ProjectFile,
};
use crate::state::AppState;
use crate::synth_clip_cache;
use chrono::Local;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::{State, Window};
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

fn build_project_file_snapshot(state: &AppState, project_path: &Path, project_name: &str) -> ProjectFile {
    let mut tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    tl.project_sec = latest_clip_end_sec(&tl).max(4.0).ceil();

    let (base_scale, use_custom_scale, custom_scale, beats_per_bar, grid_size) = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        (
            normalize_scale_key(&p.base_scale),
            p.use_custom_scale,
            normalize_custom_scale(p.custom_scale.clone()),
            normalize_beats_per_bar(p.beats_per_bar),
            normalize_grid_size(&p.grid_size),
        )
    };

    let tl_saved = prepare_source_paths_for_save(tl, project_path);
    let mut pf = ProjectFile::new(
        project_name.to_string(),
        tl_saved,
        base_scale,
        beats_per_bar,
        grid_size,
    );
    pf.use_custom_scale = use_custom_scale && custom_scale.is_some();
    pf.custom_scale = custom_scale;
    pf
}

fn unique_entry_path(
    desired: &str,
    used_paths: &mut std::collections::HashSet<String>,
) -> String {
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

    for clip in pf.timeline.clips.iter_mut() {
        let Some(source_path) = clip.source_path.clone() else {
            continue;
        };
        if source_path.trim().is_empty() {
            clip.source_path = None;
            continue;
        }

        if let Some(existing) = source_to_entry.get(&source_path) {
            clip.source_path_relative = Some(existing.clone());
            clip.source_path = None;
            continue;
        }

        let abs_path = PathBuf::from(&source_path);
        if !abs_path.is_absolute() || !abs_path.exists() {
            archive_logs.push(format!(
                "Skip missing or non-absolute source: {} (clip={})",
                source_path, clip.id
            ));
            clip.source_path = None;
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
        clip.source_path_relative = Some(unique_entry.clone());
        clip.source_path = None;
        archive_logs.push(format!(
            "Archive source: {} -> {}",
            source_path, unique_entry
        ));
    }

    let bytes = serialize_project_file_for_path(&pf, Path::new(&project_entry_name))?;

    // 为了保证保存的原子性，先写入临时文件，成功后再重命名为最终路径。
    let tmp_path = {
        let ext = zip_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
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
        let options =
            FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

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
            zip.start_file(zip_entry, options).map_err(|e| e.to_string())?;
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
                        format!("Failed to remove stale archive backup {:?}: {}", backup_path, e)
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
    // 使用原子保存，防止程序崩溃或断电导致工程文件损坏
    let tmp_path = path.with_extension("tmp_save");
    fs::write(&tmp_path, &bytes).map_err(|e| e.to_string())?;
    // fs::rename 在主流操作系统下会原子性地替换目标文件
    fs::rename(&tmp_path, &path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path); // 如果 rename 失败，顺手清理临时文件
        e.to_string()
    })?;

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
        p.base_scale = "C".to_string();
        p.use_custom_scale = false;
        p.custom_scale = None;
        p.beats_per_bar = 4;
        p.grid_size = "1/4".to_string();
    }
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
) -> crate::models::TimelineStatePayload {
    let path = PathBuf::from(&project_path);
    // 读取字节流，自动检测 MessagePack（v2）或 JSON（v1 兼容）格式。
    let bytes = fs::read(&path).unwrap_or_default();
    let parsed = load_project_file(&bytes);
    let Ok(mut pf) = parsed else {
        let mut payload = get_timeline_state(state);
        payload.ok = false;
        return payload;
    };

    let (resolved_timeline, missing_files) = resolve_source_paths_on_open(pf.timeline, &path);
    pf.timeline = resolved_timeline;
    // 旧项目兼容迁移：source_end_sec == 0.0 曾表示"到源文件末尾"，
    // 新语义要求它是真实的结束时间，此处自动修正为 duration_sec 或 length_sec。
    for clip in &mut pf.timeline.clips {
        if clip.source_end_sec == 0.0 {
            clip.source_end_sec = clip.duration_sec.unwrap_or(clip.length_sec);
        }
    }

    // 打开工程时清除所有渲染缓存，确保旧的预渲染结果不会影响新的播放。
    // 这是修复"音高分析未完成时播放导致音高编辑不生效"问题的关键步骤。
    eprintln!("[open_project] Clearing all render caches before loading project...");
    for clip in &pf.timeline.clips {
        synth_clip_cache::invalidate_clip_all_caches(&clip.id);
    }

    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        *tl = pf.timeline.clone();
        let normalized_base_scale = normalize_scale_key(&pf.base_scale);
        let normalized_custom_scale = normalize_custom_scale(pf.custom_scale.clone());
        let normalized_use_custom_scale =
            pf.use_custom_scale && normalized_custom_scale.is_some();
        tl.project_scale_notes = effective_scale_notes(
            &normalized_base_scale,
            normalized_use_custom_scale,
            normalized_custom_scale.as_ref(),
        );
        state.audio_engine.update_timeline(tl.clone());
    }
    state.clear_history();
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.name = project_name_from_path(&path);
        p.path = Some(project_path.clone());
        p.dirty = false;
        p.base_scale = normalize_scale_key(&pf.base_scale);
        p.custom_scale = normalize_custom_scale(pf.custom_scale);
        p.use_custom_scale = pf.use_custom_scale && p.custom_scale.is_some();
        p.beats_per_bar = normalize_beats_per_bar(pf.beats_per_bar);
        p.grid_size = normalize_grid_size(&pf.grid_size);
        // recent list (in-memory)
        p.recent.retain(|x| x != &project_path);
        p.recent.insert(0, project_path.clone());
        if p.recent.len() > 10 {
            p.recent.truncate(10);
        }
        update_window_title(&window, &p.name, p.dirty);
    }

    // 持久化最近工程列表
    save_recent_projects(state.inner());

    let mut payload = get_timeline_state(state);
    if !missing_files.is_empty() {
        payload.missing_files = Some(missing_files);
    }
    payload
}

pub(super) fn save_project(state: State<'_, AppState>, window: Window) -> serde_json::Value {
    let existing_path = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.path.clone()
    };
    if let Some(path) = existing_path {
        return save_project_to_path(state, window, path);
    }
    // No path yet -> Save As
    save_project_as(state, window)
}

pub(super) fn save_project_as(state: State<'_, AppState>, window: Window) -> serde_json::Value {
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
        Some(path) => save_project_to_path(state, window, path.display().to_string()),
    }
}

fn save_project_to_path(
    state: State<'_, AppState>,
    window: Window,
    project_path: String,
) -> serde_json::Value {
    let path = PathBuf::from(&project_path);
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
        tl.project_scale_notes = base_scale_notes(&normalized);
        state.audio_engine.update_timeline(tl.clone());
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
        tl.project_scale_notes = normalized.notes.clone();
        state.audio_engine.update_timeline(tl.clone());
    }

    serde_json::json!({ "ok": true, "project": state.project_meta_payload() })
}

pub(super) fn set_project_timeline_settings(
    state: State<'_, AppState>,
    beats_per_bar: u32,
    grid_size: String,
) -> serde_json::Value {
    let normalized_beats = normalize_beats_per_bar(beats_per_bar);
    let normalized_grid = normalize_grid_size(&grid_size);

    let (name, changed, was_clean) = {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let changed = p.beats_per_bar != normalized_beats || p.grid_size != normalized_grid;
        if !changed {
            return serde_json::json!({ "ok": true, "project": state.project_meta_payload() });
        }
        let was_clean = !p.dirty;
        p.beats_per_bar = normalized_beats;
        p.grid_size = normalized_grid;
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

    serde_json::json!({ "ok": true, "project": state.project_meta_payload() })
}
