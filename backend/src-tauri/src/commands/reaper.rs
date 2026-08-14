// Reaper 工程文件导入命令
//
// 提供两个操作：
// - open_reaper_dialog: 打开文件选择对话框（.rpp）
// - import_reaper_project: 解析并导入 .rpp 工程

use crate::pitch_analysis;
use crate::reaper_import;
use crate::state::AppState;
use std::path::Path;
use tauri::Window;

use super::core::get_timeline_state_from_ref;

fn update_window_title(window: &Window, name: &str, dirty: bool) {
    let suffix = if dirty { "*" } else { "" };
    let title = format!("HiFiShifter - {}{}", name, suffix);
    let _ = window.set_title(&title);
}

/// 弹出文件选择对话框，选择 .rpp 文件。
pub(super) fn open_reaper_dialog() -> serde_json::Value {
    let picked = rfd::FileDialog::new()
        .add_filter("Reaper Project", &["rpp", "RPP"])
        .pick_file();

    match picked {
        None => serde_json::json!({"ok": true, "canceled": true}),
        Some(path) => {
            serde_json::json!({"ok": true, "canceled": false, "path": path.display().to_string()})
        }
    }
}

/// 解析 Reaper 工程并导入到 HiFiShifter。
pub(super) fn import_reaper_project(
    state: &AppState,
    window: &Window,
    rpp_path: String,
) -> serde_json::Value {
    let path = Path::new(&rpp_path);

    let result = match reaper_import::import_rpp(path) {
        Ok(r) => r,
        Err(_e) => {
            let mut payload = get_timeline_state_from_ref(state);
            payload.ok = false;
            let mut json = serde_json::to_value(&payload).unwrap_or_default();
            json["ok"] = serde_json::json!(false);
            json["error"] = serde_json::json!("import_parse_failed");
            return json;
        }
    };

    // 应用到 AppState —— 合并到现有工程（不替换）
    state.begin_undo_group();
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        // 计算现有轨道的最大 order
        let max_existing_order = tl.tracks.iter().map(|t| t.order).max().unwrap_or(-1);
        let mut order_offset = max_existing_order + 1;

        // 应用工程 BPM（如果现有工程为空则直接应用；否则覆盖写入）
        if result.timeline.bpm != 120.0 || tl.tracks.is_empty() {
            tl.bpm = result.timeline.bpm;
        }
        // 导入文件不含 Tempo Map 时，保持工程现有 Tempo Map 与工程 BPM 一致。
        if result.tempo_map.is_none() {
            let project_bpm = tl.bpm;
            if let Some(points) = tl.tempo_map.as_mut() {
                if let Some(first) = points.first_mut() {
                    first.bpm = project_bpm.clamp(10.0, 960.0);
                }
            }
        }

        // 导入的 .rpp 包含 Tempo Map（变速/变拍）时替换工程现有 Tempo Map。
        // 工程级 TEMPO/TEMPOENVEX 数据才是 Tempo Map 的来源；REAPER MIDI item
        // 内部的变速（IGNTEMPO）不参与。
        if result.tempo_map.is_some() {
            tl.tempo_map = result.tempo_map.clone();
            tl.normalize_tempo_map();
            // 初始点即工程基准记录：音阶为空时物化为工程音阶。
            {
                let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(points) = tl.tempo_map.as_mut() {
                    if let Some(first) = points.first_mut() {
                        if first.scale.is_none() {
                            first.scale = Some(crate::state::tempo_scale_data_from_project(&p));
                        }
                    }
                }
            }
            if let Some(first) = tl.tempo_map.as_ref().and_then(|p| p.first()) {
                tl.bpm = first.bpm.clamp(10.0, 960.0);
            }
            // 失效渲染缓存（音阶部分可能变化，且时间轴结构变化）。
            for clip in &tl.clips {
                crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
            }
            if let Some(handle) = state.app_handle.get() {
                crate::commands::playback::request_background_render(handle);
            }
        }

        // 合并轨道（调整 order 使其排在现有轨道之后）
        for mut track in result.timeline.tracks {
            track.order = order_offset;
            order_offset += 1;
            tl.tracks.push(track);
        }
        tl.next_track_order = order_offset;

        // 合并 clips
        for mut clip in result.timeline.clips {
            crate::state::TimelineState::populate_clip_file_metadata(&mut clip);
            tl.clips.push(clip);
        }

        // 合并 pitch params
        for (track_id, params) in result.timeline.params_by_root_track {
            tl.params_by_root_track.insert(track_id, params);
        }

        // 更新工程时长
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

    // 更新工程元信息
    {
        let p = &mut *state.project.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(first) = result
            .tempo_map
            .as_ref()
            .and_then(|points| points.first())
        {
            p.beats_per_bar = first.numerator.unwrap_or(4).clamp(1, 32);
            p.time_signature_denominator = first.denominator.unwrap_or(4);
        } else {
            p.beats_per_bar = result.beats_per_bar.clamp(1, 32);
        }
        update_window_title(window, &p.name, p.dirty);
    }

    let payload = get_timeline_state_from_ref(state);
    let mut json = serde_json::to_value(&payload).unwrap_or_default();

    if !result.skipped_files.is_empty() {
        json["skipped_files"] = serde_json::json!(result.skipped_files);
    }

    json
}
