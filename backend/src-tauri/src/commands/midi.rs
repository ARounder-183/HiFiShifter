// MIDI 导入命令
//
// 提供两个 Tauri 命令：
// - get_midi_tracks: 解析 MIDI 文件并返回轨道列表（供前端轨道选择面板使用）
// - import_midi_to_pitch: 将选中的 MIDI 轨道音符写入 pitch_edit

use crate::midi_import::{self, MidiTrackInfo};
use crate::state::{AppState, PitchAnalysisAlgo, Track};

fn midi_log(message: impl AsRef<str>) {
    eprintln!("[midi_import] {}", message.as_ref());
}

fn validate_midi_import_target(track: &Track) -> Result<(), &'static str> {
    if !track.compose_enabled {
        return Err("pitch_requires_compose");
    }

    if matches!(track.pitch_analysis_algo, PitchAnalysisAlgo::None) {
        return Err("pitch_requires_algo");
    }

    Ok(())
}

/// 读取 MIDI 文件并返回轨道摘要列表。
pub(super) fn get_midi_tracks(midi_path: String) -> serde_json::Value {
    let path = std::path::Path::new(&midi_path);
    midi_log(format!("get_midi_tracks: path={midi_path}"));

    if !path.exists() {
        midi_log("get_midi_tracks: file_not_found");
        return serde_json::json!({"ok": false, "error": "file_not_found"});
    }

    match midi_import::parse_midi_file(path, None) {
        Ok(result) => {
            // 只返回有音符的轨道
            let tracks_with_notes: Vec<&MidiTrackInfo> =
                result.tracks.iter().filter(|t| t.note_count > 0).collect();

            midi_log(format!(
                "get_midi_tracks: parsed tracks_total={} tracks_with_notes={}",
                result.tracks.len(),
                tracks_with_notes.len()
            ));

            serde_json::json!({
                "ok": true,
                "tracks": tracks_with_notes,
            })
        }
        Err(e) => {
            midi_log(format!("get_midi_tracks: parse_error={e}"));
            serde_json::json!({"ok": false, "error": e})
        }
    }
}

/// 将 MIDI 文件中指定轨道的音符写入当前选中根轨的 pitch_edit。
///
/// 导入逻辑与 `paste_midi_clipboard_inner`（Reaper 剪贴板 Standard MIDI File）完全一致：
/// - 使用工程 BPM 作为 Tempo 回退
/// - 偏移量为光标位置或选区起始帧对应秒，**第一个音符对齐该偏移量**（即所有音符整体平移）
/// - 支持选区约束（selection_start_frame / selection_max_frames）
pub(super) fn import_midi_to_pitch(
    state: &AppState,
    midi_path: String,
    track_index: Option<usize>,
    selection_start_frame: Option<usize>,
    selection_max_frames: Option<usize>,
) -> serde_json::Value {
    let path = std::path::Path::new(&midi_path);
    midi_log(format!(
        "import_midi_to_pitch: path={} track_index={:?} sel_start={:?} sel_max={:?}",
        midi_path, track_index, selection_start_frame, selection_max_frames
    ));

    if !path.exists() {
        midi_log("import_midi_to_pitch: file_not_found");
        return serde_json::json!({"ok": false, "error": "file_not_found"});
    }

    // 先锁 timeline 读取 bpm / playhead / 选中轨道等信息（与 paste_midi_clipboard_inner 一致）
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

    let bpm = tl.bpm;
    let playhead_sec = tl.playhead_sec;
    let frame_period_ms_raw = tl.frame_period_ms().max(0.1);

    // 使用工程 BPM 作为 fallback tempo（与 Reaper 剪贴板路径一致）
    let parse_result = match midi_import::parse_midi_file(path, Some(bpm)) {
        Ok(r) => r,
        Err(e) => {
            midi_log(format!("import_midi_to_pitch: parse_error={e}"));
            return serde_json::json!({"ok": false, "error": e});
        }
    };

    // 收集要写入的音符：如果指定了 track_index 则只取该轨道，否则合并所有轨道
    let notes: Vec<midi_import::MidiNoteEvent> = match track_index {
        Some(idx) => {
            if idx >= parse_result.track_notes.len() {
                midi_log(format!(
                    "import_midi_to_pitch: track_index_out_of_range idx={} available={}",
                    idx,
                    parse_result.track_notes.len()
                ));
                return serde_json::json!({"ok": false, "error": "track_index_out_of_range"});
            }
            parse_result.track_notes[idx].clone()
        }
        None => {
            // 合并所有轨道的音符（与 paste_midi_clipboard_inner 一致）
            let mut all_notes: Vec<midi_import::MidiNoteEvent> =
                parse_result.track_notes.into_iter().flatten().collect();
            all_notes.sort_by(|a, b| {
                a.start_sec
                    .partial_cmp(&b.start_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_notes
        }
    };

    if notes.is_empty() {
        midi_log("import_midi_to_pitch: no_notes_in_track");
        return serde_json::json!({"ok": false, "error": "no_notes_in_track"});
    }

    midi_log(format!(
        "import_midi_to_pitch: notes_selected={} first_start={:.3} last_end={:.3}",
        notes.len(),
        notes.first().map(|n| n.start_sec).unwrap_or(0.0),
        notes.last().map(|n| n.end_sec).unwrap_or(0.0),
    ));

    // 确定目标轨道
    let Some(selected_track_id) = tl.selected_track_id.clone() else {
        midi_log("import_midi_to_pitch: no_pitch_line_selected (selected_track_id missing)");
        return serde_json::json!({"ok": false, "error": "no_pitch_line_selected"});
    };

    let Some(root_track_id) = tl.resolve_root_track_id(&selected_track_id) else {
        midi_log(format!(
            "import_midi_to_pitch: no_pitch_line_selected (resolve_root_track_id failed for selected_track_id={})",
            selected_track_id
        ));
        return serde_json::json!({"ok": false, "error": "no_pitch_line_selected"});
    };

    let Some(root_track) = tl.tracks.iter().find(|track| track.id == root_track_id) else {
        midi_log(format!(
            "import_midi_to_pitch: no_pitch_line_selected (root_track missing root_track_id={})",
            root_track_id
        ));
        return serde_json::json!({"ok": false, "error": "no_pitch_line_selected"});
    };

    if let Err(error) = validate_midi_import_target(root_track) {
        midi_log(format!(
            "import_midi_to_pitch: validation_failed error={error}"
        ));
        return serde_json::json!({"ok": false, "error": error});
    }

    tl.ensure_params_for_root(&root_track_id);
    let frame_period_ms = tl.frame_period_ms().max(0.1);

    state.checkpoint_timeline(&tl);

    let Some(entry) = tl.params_by_root_track.get_mut(&root_track_id) else {
        midi_log(format!(
            "import_midi_to_pitch: params_missing root_track_id={}",
            root_track_id
        ));
        return serde_json::json!({"ok": false, "error": "params_missing"});
    };

    // 根据是否有选区约束决定偏移和写入范围（与 paste_midi_clipboard_inner 完全一致）
    let touched = if let Some(sel_start) = selection_start_frame {
        // 以选区起始帧对应秒作为目标偏移，所有音符整体平移使第一个音符对齐该偏移
        let offset_sec = (sel_start as f64 * frame_period_ms_raw) / 1000.0;
        let first_start = notes
            .iter()
            .map(|n| n.start_sec)
            .fold(f64::INFINITY, f64::min);
        let align_offset = offset_sec - first_start;
        let max_frame = sel_start + selection_max_frames.unwrap_or(usize::MAX - sel_start);
        let clamp_len = max_frame.min(entry.pitch_edit.len());
        midi_log(format!(
            "import_midi_to_pitch: selection mode offset_sec={:.3} align_offset={:.3} clamp_len={}",
            offset_sec, align_offset, clamp_len
        ));
        midi_import::write_notes_to_pitch_edit(
            &notes,
            frame_period_ms,
            &mut entry.pitch_edit[..clamp_len],
            align_offset,
        )
    } else {
        // 以光标位置为目标偏移，所有音符整体平移使第一个音符对齐该偏移
        let first_start = notes
            .iter()
            .map(|n| n.start_sec)
            .fold(f64::INFINITY, f64::min);
        let align_offset = playhead_sec - first_start;
        midi_log(format!(
            "import_midi_to_pitch: playhead mode offset_sec={:.3} align_offset={:.3}",
            playhead_sec, align_offset
        ));
        midi_import::write_notes_to_pitch_edit(
            &notes,
            frame_period_ms,
            &mut entry.pitch_edit,
            align_offset,
        )
    };

    if touched > 0 {
        entry.pitch_edit_user_modified = true;
        midi_log(format!(
            "import_midi_to_pitch: success frames_touched={} notes_imported={}",
            touched,
            notes.len()
        ));
    } else {
        midi_log(format!(
            "import_midi_to_pitch: no_frames_touched notes={} pitch_edit_len={} frame_period_ms={:.3}",
            notes.len(), entry.pitch_edit.len(), frame_period_ms
        ));
        return serde_json::json!({"ok": false, "error": "no_frames_touched"});
    }

    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({
        "ok": true,
        "notes_imported": notes.len(),
        "frames_touched": touched,
    })
}
