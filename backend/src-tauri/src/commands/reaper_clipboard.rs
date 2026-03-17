// Reaper 剪贴板粘贴命令
//
// 从系统剪贴板读取 Reaper 的 "REAPERMedia" 自定义格式数据，
// 解析并导入到当前时间线。
// 优先检测 "Standard MIDI File" 格式并作为 MIDI 导入。
// 支持 Windows / macOS / Linux (X11 & Wayland)。

use crate::midi_import;
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
    let ptr = data.bytes().cast::<u8>();
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
fn read_midi_clipboard() -> Result<Vec<u8>, String> {
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
fn read_midi_clipboard() -> Result<Vec<u8>, String> {
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
    let ptr = data.bytes().cast::<u8>();
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bytes.to_vec())
}

/// Linux: 通过 wl-paste (Wayland) 或 xclip (X11) 读取自定义目标 "Standard MIDI File"。
#[cfg(target_os = "linux")]
fn read_midi_clipboard() -> Result<Vec<u8>, String> {
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
fn read_midi_clipboard() -> Result<Vec<u8>, String> {
    Err("clipboard_unsupported_platform".to_string())
}

// ---------------------------------------------------------------------------
// MIDI 剪贴板粘贴实现
// ---------------------------------------------------------------------------

/// 将 MIDI 剪贴板数据写入当前选中轨道的 pitch_edit。
///
/// 使用工程 BPM 作为 Tempo 回退，导入起始点为当前光标位置。
/// 若提供了 selection_start_frame / selection_max_frames，则以选区起始帧作为偏移起点，
/// 超出选区范围的音符不写入。
fn paste_midi_clipboard_inner(
    state: &AppState,
    midi_data: &[u8],
    selection_start_frame: Option<usize>,
    selection_max_frames: Option<usize>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

    let bpm = tl.bpm;
    let playhead_sec = tl.playhead_sec;
    let frame_period_ms_raw = tl.frame_period_ms().max(0.1);

    // 解析 MIDI 数据，使用工程 BPM 作为 fallback tempo
    let parse_result = match midi_import::parse_midi_bytes(midi_data, Some(bpm)) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::json!({"ok": false, "error": format!("midi_parse_failed: {}", e)});
        }
    };

    // 合并所有轨道的音符
    let mut all_notes: Vec<midi_import::MidiNoteEvent> =
        parse_result.track_notes.into_iter().flatten().collect();
    all_notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if all_notes.is_empty() {
        return serde_json::json!({"ok": false, "error": "midi_no_notes"});
    }

    // 确定目标轨道
    let Some(selected_track_id) = tl.selected_track_id.clone() else {
        return serde_json::json!({"ok": false, "error": "no_track_selected"});
    };

    let Some(root_track_id) = tl.resolve_root_track_id(&selected_track_id) else {
        return serde_json::json!({"ok": false, "error": "no_track_selected"});
    };

    tl.ensure_params_for_root(&root_track_id);
    let frame_period_ms = tl.frame_period_ms().max(0.1);

    state.checkpoint_timeline(&tl);

    let Some(entry) = tl.params_by_root_track.get_mut(&root_track_id) else {
        return serde_json::json!({"ok": false, "error": "params_missing"});
    };

    // 根据是否有选区约束决定偏移和写入范围
    let touched = if let Some(sel_start) = selection_start_frame {
        // 以选区起始帧对应秒作为偏移
        let offset_sec = (sel_start as f64 * frame_period_ms_raw) / 1000.0;
        let max_frame = sel_start + selection_max_frames.unwrap_or(usize::MAX - sel_start);
        // 限制 pitch_edit 的写入范围
        let clamp_len = max_frame.min(entry.pitch_edit.len());
        midi_import::write_notes_to_pitch_edit(
            &all_notes,
            frame_period_ms,
            &mut entry.pitch_edit[..clamp_len],
            offset_sec,
        )
    } else {
        // 默认以光标位置作为偏移写入 pitch_edit
        midi_import::write_notes_to_pitch_edit(
            &all_notes,
            frame_period_ms,
            &mut entry.pitch_edit,
            playhead_sec,
        )
    };

    if touched > 0 {
        entry.pitch_edit_user_modified = true;
    }

    state.audio_engine.update_timeline(tl.clone());
    drop(tl);

    let payload = get_timeline_state_from_ref(state);
    let mut json = serde_json::to_value(&payload).unwrap_or_default();
    json["midi_imported"] = serde_json::json!({
        "notes": all_notes.len(),
        "frames_touched": touched,
    });
    json
}

/// 粘贴 Reaper 剪贴板数据到当前选中的轨道。
/// 优先检测 "Standard MIDI File" 格式，若存在则作为 MIDI 导入到当前轨道的 pitch_edit。
pub(super) fn paste_reaper_clipboard(
    state: &AppState,
    selection_start_frame: Option<usize>,
    selection_max_frames: Option<usize>,
) -> serde_json::Value {
    // 优先尝试 MIDI 剪贴板
    if let Ok(midi_data) = read_midi_clipboard() {
        return paste_midi_clipboard_inner(
            state,
            &midi_data,
            selection_start_frame,
            selection_max_frames,
        );
    }

    // 回退到 REAPERMedia 剪贴板
    let data = match read_reaper_clipboard() {
        Ok(d) => d,
        Err(e) => {
            return serde_json::json!({"ok": false, "error": e});
        }
    };

    // 从当前 timeline 读取光标位置、选中轨道、轨道顺序
    let (playhead_sec, selected_track_idx, ordered_track_ids) = {
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

        (tl.playhead_sec, sel_idx, ordered)
    };

    // 解析并转换
    let result = match reaper_import::import_reaper_clipboard(
        &data,
        playhead_sec,
        selected_track_idx,
        &ordered_track_ids,
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
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        state.checkpoint_timeline(&tl);

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
            tl.clips.push(clip.clone());
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
    }

    let payload = get_timeline_state_from_ref(state);
    let mut json = serde_json::to_value(&payload).unwrap_or_default();

    if !result.skipped_files.is_empty() {
        json["skipped_files"] = serde_json::json!(result.skipped_files);
    }

    json
}
