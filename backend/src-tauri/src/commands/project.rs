use crate::project::{
    load_project_file, make_paths_relative, project_name_from_path, resolve_paths_relative,
    CustomScale, ProjectFile,
};
use crate::state::AppState;
use crate::synth_clip_cache;
use std::fs;
use std::path::PathBuf;
use tauri::{State, Window};

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

pub(crate) fn save_project_to_path_inner(
    state: &AppState,
    window: &Window,
    project_path: String,
) -> Result<crate::models::TimelineStatePayload, String> {
    let path = PathBuf::from(&project_path);
    let name = project_name_from_path(&path);

    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
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
    let tl_rel = make_paths_relative(tl, &path);
    let mut pf = ProjectFile::new(name.clone(), tl_rel, base_scale, beats_per_bar, grid_size);
    pf.use_custom_scale = use_custom_scale && custom_scale.is_some();
    pf.custom_scale = custom_scale;
    // 使用 MessagePack 格式保存（v2），体积更小、解析更快。
    let bytes = rmp_serde::to_vec_named(&pf).map_err(|e| e.to_string())?;
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
    {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dir) = state.config_dir.get() {
            crate::config::save_recent(dir, &p.recent);
        }
    }

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
    update_window_title(&window, "Untitled", false);
    get_timeline_state(state)
}

pub(super) fn open_project_dialog() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("HiFiShifter Project", &["hshp", "hsp", "json"])
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

    pf.timeline = resolve_paths_relative(pf.timeline, &path);
    let missing_files: Vec<String> = pf
        .timeline
        .clips
        .iter()
        .filter_map(|clip| clip.source_path.as_deref()) // 转为借用，消除 99% 的堆分配
        .filter(|sp| !sp.trim().is_empty() && !std::path::Path::new(sp).exists())
        .map(|sp| sp.to_string()) // 只有确诊丢失的文件，才分配内存收集
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
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
    {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dir) = state.config_dir.get() {
            crate::config::save_recent(dir, &p.recent);
        }
    }

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
        .add_filter("HiFiShifter Project", &["hshp", "hsp", "json"])
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
        p.custom_scale = Some(normalized);
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
