// Reaper 剪贴板粘贴命令
//
// 从系统剪贴板读取 Reaper 的 "REAPERMedia" 自定义格式数据，
// 解析并导入到当前时间线。
// 优先检测 "Standard MIDI File" 格式并作为 MIDI 导入。
// 支持 Windows / macOS / Linux (X11 & Wayland)。

use crate::pitch_analysis;
use crate::reaper_import;
use crate::state::AppState;

use super::core::get_timeline_state_from_ref;

// ---------------------------------------------------------------------------
// 平台特定的剪贴板读取
// ---------------------------------------------------------------------------

/// Windows: 通过 clipboard-win 读取自定义格式 "REAPERMedia"。
#[cfg(target_os = "windows")]
fn read_reaper_clipboard() -> Result<Vec<u8>, String> {
    use clipboard_win::{register_format, Clipboard};

    let _clipboard =
        Clipboard::new_attempts(10).map_err(|e| format!("clipboard_open_failed: {}", e))?;

    let format =
        register_format("REAPERMedia").ok_or_else(|| "clipboard_format_not_found".to_string())?;

    let size =
        clipboard_win::raw::size(format.get()).ok_or_else(|| "clipboard_empty".to_string())?;

    let mut buf = vec![0u8; size.get()];
    let bytes_read = clipboard_win::raw::get(format.get(), &mut buf)
        .map_err(|e| format!("clipboard_read_failed: {}", e))?;

    buf.truncate(bytes_read);
    Ok(buf)
}

/// macOS: 通过 NSPasteboard 读取自定义类型 "REAPERMedia"。
#[cfg(target_os = "macos")]
fn read_reaper_clipboard() -> Result<Vec<u8>, String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::NSString;

    let pasteboard = unsafe { NSPasteboard::generalPasteboard() };
    let pb_type = NSString::from_str("REAPERMedia");

    let data =
        unsafe { pasteboard.dataForType(&pb_type) }.ok_or_else(|| "clipboard_empty".to_string())?;

    let len = data.length();
    if len == 0 {
        return Err("clipboard_empty".to_string());
    }
    // `objc2` / `objc2-foundation` may expose different helper methods across
    // versions; prefer converting via a safe slice view if available.
    // Try calling `bytes` via objc runtime as a fallback for compatibility
    use objc2::msg_send;
    use std::ffi::c_void;
    // `Retained<NSData>` does not implement `MessageReceiver`; take a reference
    // to the underlying object so `msg_send!` accepts it (e.g. `&T`).
    let raw_ptr: *const c_void = unsafe { msg_send![&*data, bytes] };
    let ptr = raw_ptr as *const u8;
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bytes.to_vec())
}

/// Linux: 通过 wl-paste (Wayland) 或 xclip (X11) 读取自定义目标 "REAPERMedia"。
#[cfg(target_os = "linux")]
fn read_reaper_clipboard() -> Result<Vec<u8>, String> {
    use std::process::Command;

    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();

    let output = if is_wayland {
        Command::new("wl-paste")
            .args(["--type", "REAPERMedia"])
            .output()
    } else {
        Command::new("xclip")
            .args(["-selection", "clipboard", "-target", "REAPERMedia", "-o"])
            .output()
    };

    let output = output.map_err(|e| {
        let tool = if is_wayland { "wl-paste" } else { "xclip" };
        format!("clipboard_read_failed: failed to run {}: {}", tool, e)
    })?;

    if !output.status.success() {
        return Err("clipboard_empty".to_string());
    }

    if output.stdout.is_empty() {
        return Err("clipboard_empty".to_string());
    }

    Ok(output.stdout)
}

/// 不支持的平台回退。
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn read_reaper_clipboard() -> Result<Vec<u8>, String> {
    Err("clipboard_unsupported_platform".to_string())
}

// ---------------------------------------------------------------------------
// 平台特定的 MIDI 剪贴板读取 ("Standard MIDI File")
// ---------------------------------------------------------------------------

/// Windows: 通过 clipboard-win 读取自定义格式 "Standard MIDI File"。
#[cfg(target_os = "windows")]
pub(crate) fn read_midi_clipboard() -> Result<Vec<u8>, String> {
    use clipboard_win::{register_format, Clipboard};

    let _clipboard =
        Clipboard::new_attempts(10).map_err(|e| format!("clipboard_open_failed: {}", e))?;

    let format = register_format("Standard MIDI File")
        .ok_or_else(|| "midi_clipboard_format_not_found".to_string())?;

    let size =
        clipboard_win::raw::size(format.get()).ok_or_else(|| "midi_clipboard_empty".to_string())?;

    let mut buf = vec![0u8; size.get()];
    let bytes_read = clipboard_win::raw::get(format.get(), &mut buf)
        .map_err(|e| format!("midi_clipboard_read_failed: {}", e))?;

    buf.truncate(bytes_read);
    Ok(buf)
}

/// macOS: 通过 NSPasteboard 读取自定义类型 "Standard MIDI File"。
#[cfg(target_os = "macos")]
pub(crate) fn read_midi_clipboard() -> Result<Vec<u8>, String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::NSString;

    let pasteboard = unsafe { NSPasteboard::generalPasteboard() };
    let pb_type = NSString::from_str("Standard MIDI File");

    let data = unsafe { pasteboard.dataForType(&pb_type) }
        .ok_or_else(|| "midi_clipboard_empty".to_string())?;

    let len = data.length();
    if len == 0 {
        return Err("midi_clipboard_empty".to_string());
    }
    use objc2::msg_send;
    use std::ffi::c_void;
    let raw_ptr: *const c_void = unsafe { msg_send![&*data, bytes] };
    let ptr = raw_ptr as *const u8;
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bytes.to_vec())
}

/// Linux: 通过 wl-paste (Wayland) 或 xclip (X11) 读取自定义目标 "Standard MIDI File"。
#[cfg(target_os = "linux")]
pub(crate) fn read_midi_clipboard() -> Result<Vec<u8>, String> {
    use std::process::Command;

    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();

    let output = if is_wayland {
        Command::new("wl-paste")
            .args(["--type", "Standard MIDI File"])
            .output()
    } else {
        Command::new("xclip")
            .args([
                "-selection",
                "clipboard",
                "-target",
                "Standard MIDI File",
                "-o",
            ])
            .output()
    };

    let output = output.map_err(|e| {
        let tool = if is_wayland { "wl-paste" } else { "xclip" };
        format!("midi_clipboard_read_failed: failed to run {}: {}", tool, e)
    })?;

    if !output.status.success() {
        return Err("midi_clipboard_empty".to_string());
    }

    if output.stdout.is_empty() {
        return Err("midi_clipboard_empty".to_string());
    }

    Ok(output.stdout)
}

/// 不支持的平台回退。
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub(crate) fn read_midi_clipboard() -> Result<Vec<u8>, String> {
    Err("clipboard_unsupported_platform".to_string())
}

/// 粘贴 Reaper 剪贴板数据到当前选中的轨道。
///
/// 不再自动处理 MIDI 剪贴板；前端应先通过 `read_midi_clipboard_to_memory` 检测
/// Standard MIDI File 并弹出统一导入弹窗。
pub(super) fn paste_reaper_clipboard(
    state: &AppState,
    _selection_start_frame: Option<usize>,
    _selection_max_frames: Option<usize>,
) -> serde_json::Value {
    // REAPERMedia 剪贴板
    let data = match read_reaper_clipboard() {
        Ok(d) => d,
        Err(e) => {
            return serde_json::json!({"ok": false, "error": e});
        }
    };

    // 从当前 timeline 读取光标位置、选中轨道、轨道顺序、BPM
    let (playhead_sec, selected_track_idx, ordered_track_ids, project_bpm) = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        // 按 order 排序的轨道 ID
        let mut sorted_tracks: Vec<_> = tl.tracks.iter().collect();
        sorted_tracks.sort_by_key(|t| t.order);
        let ordered: Vec<String> = sorted_tracks.iter().map(|t| t.id.clone()).collect();

        // 选中轨道的下标
        let sel_idx = tl
            .selected_track_id
            .as_ref()
            .and_then(|sel| ordered.iter().position(|id| id == sel))
            .unwrap_or(0);

        (tl.playhead_sec, sel_idx, ordered, tl.bpm)
    };

    // 解析并转换
    let result = match reaper_import::import_reaper_clipboard(
        &data,
        playhead_sec,
        selected_track_idx,
        &ordered_track_ids,
        project_bpm,
    ) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::json!({"ok": false, "error": format!("parse_failed: {}", e)});
        }
    };

    // 需要开启 compose_enabled 的轨道
    let tracks_needing_compose: Vec<String> = result
        .timeline
        .params_by_root_track
        .keys()
        .filter(|tid| {
            result
                .timeline
                .params_by_root_track
                .get(*tid)
                .and_then(|p| p.pending_pitch_offset.as_ref())
                .map(|offsets| offsets.iter().any(|&v| v.abs() > 1e-6))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    // 应用到 AppState
    state.begin_undo_group();
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        if !result.timeline.tracks.is_empty() {
            // 有新轨道：合并到现有 timeline
            for track in &result.timeline.tracks {
                tl.tracks.push(track.clone());
            }
            tl.next_track_order = tl
                .next_track_order
                .max(tl.tracks.iter().map(|t| t.order).max().unwrap_or(0) + 1);
        }

        // 合并 clips
        for clip in &result.timeline.clips {
            let mut c = clip.clone();
            crate::state::TimelineState::populate_clip_file_metadata(&mut c);
            tl.clips.push(c);
        }

        // 合并 pitch params（pending_pitch_offset 需要合并到已有的 entry）
        for (track_id, new_params) in &result.timeline.params_by_root_track {
            if let Some(existing) = tl.params_by_root_track.get_mut(track_id) {
                // 轨道已有 pitch 数据 → 只设置 pending offset
                if let Some(ref offsets) = new_params.pending_pitch_offset {
                    existing.pending_pitch_offset = Some(offsets.clone());
                }
            } else {
                tl.params_by_root_track
                    .insert(track_id.clone(), new_params.clone());
            }
        }

        // 为含音高偏移的轨道开启 compose_enabled
        for track in &mut tl.tracks {
            if tracks_needing_compose.contains(&track.id) && !track.compose_enabled {
                track.compose_enabled = true;
            }
        }

        // 更新工程时长（如果需要）
        let max_end = tl
            .clips
            .iter()
            .map(|c| c.start_sec + c.length_sec)
            .fold(tl.project_sec, f64::max);
        tl.project_sec = max_end;

        state.audio_engine.update_timeline(tl.clone());

        // 为导入的 MIDI 音高参考块触发 pitch 分析
        let midi_root_tracks: Vec<String> = tl
            .clips
            .iter()
            .filter(|c| c.midi_note_data.is_some())
            .filter_map(|c| tl.resolve_root_track_id(&c.track_id))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        drop(tl);
        for root_id in &midi_root_tracks {
            pitch_analysis::maybe_schedule_pitch_orig(state, root_id);
        }
    }
    let _ = state.end_undo_group();

    let payload = get_timeline_state_from_ref(state);
    let mut json = serde_json::to_value(&payload).unwrap_or_default();

    if !result.skipped_files.is_empty() {
        json["skipped_files"] = serde_json::json!(result.skipped_files);
    }

    json
}
