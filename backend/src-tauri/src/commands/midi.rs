// MIDI 导入命令
//
// - get_midi_tracks: 解析 MIDI 文件并返回轨道列表（供前端轨道选择面板使用）
// - read_midi_clipboard_to_memory: 读系统剪贴板的 "Standard MIDI File" 并缓存
// - import_midi_to_pitch: 将选中的 MIDI 轨道音符写入 pitch_edit
// - import_midi_as_clip / replace_midi_clip_data: 建/改"只有音符、没有音频源"的片段
//
// 【为什么这里只剩薄适配】解析、选区窗口、写曲线三步已搬进内核
// （`hifishifter_kernel::editor::midi_import`），ARA 插件跑的是同一份实现。留在
// App 侧的只有它独有的那件事：本地 MIDI 片段 —— 插件的时间线是宿主清单的投影
// （`workspace_timeline_locked` 只保留已分配 region 的 clip），造不出本地片段。

use crate::midi_import;
use crate::state::AppState;

use super::param_selection_window::SelectionFrameRange;

// 内核 MIDI 导入命令的 App 侧适配；唯一差异是剪贴板载荷缓存归谁持有。
pub(crate) use hifishifter_kernel::editor::midi_import as shared;

impl shared::MidiImportHost for AppState {
    fn clipboard_midi(&self) -> &std::sync::Mutex<std::collections::VecDeque<(String, Vec<u8>)>> {
        &self.clipboard_midi_cache
    }
}

fn midi_log(message: impl AsRef<str>) {
    log::error!("[midi_import] {}", message.as_ref());
}

fn error_payload(error: &str) -> crate::models::TimelineStatePayload {
    crate::models::TimelineStatePayload {
        ok: false,
        tracks: vec![],
        clips: vec![],
        created_clip_ids: None,
        created_track_ids: None,
        selected_track_id: None,
        selected_clip_id: None,
        bpm: 120.0,
        playhead_sec: 0.0,
        project_sec: None,
        project: None,
        missing_files: Some(vec![error.to_string()]),
        disabled_group_ids: vec![],
        tempo_map: None,
        undo_depth: None,
        redo_depth: None,
        notes_markdown: None,
        param_selection_restore: None,
    }
}

/// 读取 MIDI 文件（或剪贴板缓存）并返回轨道摘要列表。
pub(super) fn get_midi_tracks(
    state: &AppState,
    midi_path: String,
    clipboard_guid: Option<String>,
) -> serde_json::Value {
    shared::get_midi_tracks(state, midi_path, clipboard_guid)
}

/// 从系统剪贴板读取 "Standard MIDI File" 格式数据，存入内存缓存并解析。
///
/// 平台读取留在 `reaper_clipboard`（App 与插件共用 `hifishifter-clipboard`），
/// 解析、GUID 与缓存全在共享内核里。
pub(super) fn read_midi_clipboard_to_memory(state: &AppState) -> serde_json::Value {
    shared::read_midi_clipboard_to_memory(
        state,
        crate::commands::reaper_clipboard::read_midi_clipboard(),
    )
}

/// 剪贴板 MIDI 载荷的消费与来源解析都在共享内核里；这里只留调用别名，
/// 免得 `import_midi_as_clip` / `replace_midi_clip_data` 到处写长路径。
use shared::{resolve_midi_source, set_project_bpm_syncing_tempo_map, take_clipboard_midi};

/// 将 MIDI 文件中指定轨道的音符写入当前选中根轨的 pitch_edit。
///
/// 语义（BPM 回退、选区对齐、多选区掩码回滚）全部在内核共享实现里；这里只把
/// Tauri 的位置参数装成请求体。
pub(super) fn import_midi_to_pitch(
    state: &AppState,
    midi_path: String,
    track_indices: Vec<usize>,
    selection_ranges: Option<Vec<SelectionFrameRange>>,
    fill_gaps: Option<bool>,
    note_bpm_mode: Option<String>,
    specified_bpm: Option<f64>,
    import_midi_bpm_as_project: Option<bool>,
    clipboard_guid: Option<String>,
    close_leading_gap: Option<bool>,
) -> serde_json::Value {
    shared::import_midi_to_pitch(
        state,
        shared::ImportMidiToPitchRequest {
            midi_path,
            track_indices,
            selection_ranges,
            fill_gaps,
            note_bpm_mode,
            specified_bpm,
            import_midi_bpm_as_project,
            clipboard_guid,
            close_leading_gap,
        },
    )
}

/// 导入 MIDI 文件为时间线上的 MIDI clip（无音频源）。
///
/// 创建一���特殊的 clip，其中 `source_path` 为 None，`midi_note_data` 包含
/// 从 MIDI 文件提取的音符事件。clip 的长度由 MIDI 中最后一个音符的结束时间决定。
///
/// 返回完整的 timeline state payload，以便前端更新 Redux store。
pub(super) fn import_midi_as_clip(
    state: &AppState,
    midi_path: String,
    track_indices: Vec<usize>,
    track_id: Option<String>,
    start_sec: f64,
    fill_gaps: Option<bool>,
    multi_track_merge: Option<bool>,
    note_bpm_mode: Option<String>,
    specified_bpm: Option<f64>,
    import_midi_bpm_as_project: Option<bool>,
    clipboard_guid: Option<String>,
    close_leading_gap: Option<bool>,
    import_midi_as_tempo_map: Option<bool>,
    import_midi_tempo: Option<bool>,
    import_midi_time_signature: Option<bool>,
    import_midi_key_signature: Option<bool>,
) -> crate::models::TimelineStatePayload {
    midi_log(format!(
        "import_midi_as_clip: path={} clipboard_guid={:?} track_indices={:?} track_id={:?} start_sec={:.3} fill_gaps={:?} multi_track_merge={:?} note_bpm_mode={:?} specified_bpm={:?} import_midi_bpm_as_project={:?} close_leading_gap={:?} import_as_tempo_map={:?} import_tempo={:?} import_ts={:?} import_key={:?}",
        midi_path, clipboard_guid, track_indices, track_id, start_sec, fill_gaps, multi_track_merge, note_bpm_mode, specified_bpm, import_midi_bpm_as_project, close_leading_gap, import_midi_as_tempo_map, import_midi_tempo, import_midi_time_signature, import_midi_key_signature
    ));

    // 先短暂锁定读取 bpm / 拍号；MIDI 磁盘解析放在锁外 —— 本命令是同步命令
    // （主线程），持锁解析会在阻塞主线程的同时冻结所有其他命令与 UI 轮询。
    let (project_bpm, project_beats_per_bar, project_time_signature_denominator) = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let bpm = tl.bpm;
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        (bpm, p.beats_per_bar, p.time_signature_denominator)
    };

    let (mut parse_result, source_stem) = match resolve_midi_source(
        state,
        Some(&midi_path).filter(|s| !s.is_empty()),
        clipboard_guid.as_deref(),
        Some(project_bpm),
    ) {
        Ok((r, stem)) => (r, stem),
        Err(e) => {
            midi_log(format!("import_midi_as_clip: parse_error={}", e));
            return error_payload(&e);
        }
    };

    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

    let file_stem = source_stem.unwrap_or_else(|| "MIDI".to_string());

    let initial_bpm = parse_result.initial_bpm;

    // ── BPM 导入与重映射 ──
    let import_as_project = import_midi_bpm_as_project.unwrap_or(false);
    if import_as_project {
        set_project_bpm_syncing_tempo_map(&mut tl, initial_bpm);
        midi_log(format!(
            "import_midi_as_clip: set_project_bpm from {project_bpm} to {initial_bpm}"
        ));
    }

    let mode = note_bpm_mode.as_deref().unwrap_or("midi");
    let target_bpm: Option<f64> = match mode {
        "project" => {
            if import_as_project {
                None // 工程 BPM 已设为 MIDI BPM，视为"MIDI 自身 BPM"
            } else {
                Some(project_bpm)
            }
        }
        "specified" => specified_bpm.filter(|&b| b > 0.0 && b.is_finite()),
        _ => None, // "midi" 模式：不重映射
    };

    // 应用 BPM 重映射到所有轨道的音符
    if let Some(tbpm) = target_bpm {
        let scale = initial_bpm / tbpm;
        midi_log(format!(
            "import_midi_as_clip: bpm_remap initial_bpm={initial_bpm:.2} target_bpm={tbpm:.2} scale={scale:.6}"
        ));
        for track_notes in &mut parse_result.track_notes {
            for note in track_notes {
                note.start_sec *= scale;
                note.end_sec *= scale;
            }
        }
    }

    // ── 导入为 Tempo Map（替换工程现有 Tempo Map） ──
    if import_midi_as_tempo_map.unwrap_or(false) {
        let points = midi_import::build_tempo_map_points_from_midi(
            &parse_result,
            import_midi_tempo.unwrap_or(true),
            import_midi_time_signature.unwrap_or(true),
            import_midi_key_signature.unwrap_or(false),
            tl.bpm,
            project_beats_per_bar,
            // 工程拍号分母透传：MIDI 无拍号事件时回退到工程基准
            // （否则 6/8 等工程会被静默重置为 6/4）。
            project_time_signature_denominator,
        );
        let render_scale_signature_before = tl.render_scale_signature();

        // 先判断是否真的会产生变化，再决定是否打撤销快照（与
        // set_timeline_tempo_map 的幂等约定一致：无变化不产生撤销步）。
        let will_change = match &points {
            Some(pts) => {
                let has_change = pts.iter().any(|p| p.position_sec > 1e-9) || pts.len() > 1;
                if has_change {
                    true
                } else {
                    // 无 0 之后的变化：工程基准值（BPM/拍号）是否与现状不同。
                    pts.first()
                        .map(|first| {
                            (tl.bpm - first.bpm.clamp(10.0, 960.0)).abs() > 1e-9
                                || project_beats_per_bar
                                    != first.numerator.unwrap_or(4).clamp(1, 32)
                                || project_time_signature_denominator
                                    != first.denominator.unwrap_or(4)
                        })
                        .unwrap_or(false)
                }
            }
            None => tl.tempo_map.is_some(),
        };
        if will_change {
            state.checkpoint_timeline(&tl, crate::state::HistoryOp::ImportMidi);
        }

        // 无变化分支把 0 位置点的音阶（调号）应用到工程时可能改变工程音阶；
        // render_scale_signature 会覆盖该情况，无需单独跟踪 project_scale_changed。
        match points {
            Some(points) => {
                let has_change = points.iter().any(|p| p.position_sec > 1e-9) || points.len() > 1;
                if has_change {
                    tl.tempo_map = Some(points);
                    tl.normalize_tempo_map();
                    // 初始点即工程基准记录：音阶为空时物化为工程音阶。
                    {
                        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(first_points) = tl.tempo_map.as_mut() {
                            if let Some(first) = first_points.first_mut() {
                                if first.scale.is_none() {
                                    first.scale =
                                        Some(crate::state::tempo_scale_data_from_project(&p));
                                }
                            }
                        }
                    }
                    // 同步工程基准 BPM / 拍号 / 音阶（含导入的调号，与
                    // set_timeline_tempo_map 的双向同步约定一致）。
                    {
                        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                        state.sync_project_record_from_tempo_map(&mut tl, &mut p);
                    }
                    midi_log(format!(
                        "import_midi_as_clip: tempo_map imported with {} points",
                        tl.tempo_map.as_ref().map(|p| p.len()).unwrap_or(0)
                    ));
                } else {
                    // 无 0 之后的变化：仅把 0 位置点的参数应用到工程基准（速度/拍号/音阶）。
                    if let Some(first) = points.first() {
                        tl.bpm = first.bpm.clamp(10.0, 960.0);
                        if let Ok(mut p) = state.project.lock() {
                            p.beats_per_bar = first.numerator.unwrap_or(4).clamp(1, 32);
                            p.time_signature_denominator = first.denominator.unwrap_or(4);
                            if let Some(scale) = first.scale.as_ref() {
                                if let Some(key) = scale.key.as_deref() {
                                    let key = key.to_string();
                                    if p.base_scale != key || p.use_custom_scale {
                                        // 工程基准音阶变化已由 render_scale_signature 捕获。
                                    }
                                    tl.project_scale_notes =
                                        crate::state::scale_notes_for_key(&key)
                                            .unwrap_or_else(|| vec![0, 2, 4, 5, 7, 9, 11]);
                                    p.base_scale = key;
                                    p.use_custom_scale = false;
                                    p.custom_scale = None;
                                    p.dirty = true;
                                }
                            }
                            p.dirty = true;
                        }
                    }
                    tl.tempo_map = None;
                    midi_log("import_midi_as_clip: no tempo map change (initial values applied to project)");
                }
            }
            None => {
                tl.tempo_map = None;
                midi_log("import_midi_as_clip: no tempo map events found; cleared");
            }
        }
        let render_scale_signature_after = tl.render_scale_signature();
        if render_scale_signature_before != render_scale_signature_after {
            for clip in &tl.clips {
                crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
            }
            if let Some(handle) = state.app_handle.get() {
                crate::commands::playback::request_background_render(handle);
            }
        }
    }

    let fill = fill_gaps.unwrap_or(false);
    let multi = multi_track_merge.unwrap_or(true);
    let close_gap = close_leading_gap.unwrap_or(true);

    if multi {
        // ── 合并模式：将所有选中轨道的音符合并为单个 clip ──
        let notes: Vec<midi_import::MidiNoteEvent> = {
            let mut all: Vec<_> = if track_indices.is_empty() {
                parse_result.track_notes.into_iter().flatten().collect()
            } else {
                track_indices
                    .iter()
                    .filter_map(|&idx| parse_result.track_notes.get(idx))
                    .flatten()
                    .cloned()
                    .collect()
            };
            all.sort_by(|a, b| {
                a.start_sec
                    .partial_cmp(&b.start_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all
        };

        if notes.is_empty() {
            midi_log("import_midi_as_clip: no_notes");
            return error_payload("no_notes_in_track");
        }

        let last_end = notes.iter().map(|n| n.end_sec).fold(0.0f64, f64::max);

        let first_start = notes
            .iter()
            .map(|n| n.start_sec)
            .fold(f64::INFINITY, f64::min);
        let length_sec = if close_gap {
            (last_end - first_start).max(0.1)
        } else {
            last_end.max(0.1)
        };
        let normalized_notes: Vec<midi_import::MidiNoteEvent> = notes
            .into_iter()
            .map(|n| midi_import::MidiNoteEvent {
                start_sec: if close_gap {
                    n.start_sec - first_start
                } else {
                    n.start_sec
                },
                end_sec: if close_gap {
                    n.end_sec - first_start
                } else {
                    n.end_sec
                },
                note: n.note,
                velocity: n.velocity,
                channel: n.channel,
            })
            .collect();

        let pitch_range = {
            let min_note = normalized_notes.iter().fold(127.0f32, |m, n| m.min(n.note));
            let max_note = normalized_notes.iter().fold(0.0f32, |m, n| m.max(n.note));
            Some(crate::models::PitchRange {
                min: min_note,
                max: max_note,
            })
        };

        state.checkpoint_timeline(&tl, crate::state::HistoryOp::ImportMidi);

        let clip_id = tl.add_clip(
            track_id,
            Some(file_stem),
            Some(start_sec),
            Some(length_sec),
            None,
        );

        if let Some(clip) = tl.clips.iter_mut().find(|c| c.id == clip_id) {
            clip.midi_note_data = Some(normalized_notes);
            clip.midi_fill_gaps = fill;
            clip.pitch_range = pitch_range;
            clip.color = "cyan".to_string();
            clip.source_path = None;
            clip.source_path_relative = None;
        }

        midi_log(format!(
            "import_midi_as_clip: created clip_id={} length_sec={:.3} notes={}",
            clip_id,
            length_sec,
            tl.clips
                .iter()
                .find(|c| c.id == clip_id)
                .and_then(|c| c.midi_note_data.as_ref())
                .map(|n| n.len())
                .unwrap_or(0)
        ));

        let root_track_id = tl.resolve_root_track_id(
            &tl.clips
                .iter()
                .find(|c| c.id == clip_id)
                .map(|c| c.track_id.clone())
                .unwrap_or_default(),
        );
        tl.sync_clip_takes_from_flat();
        state.audio_engine.update_timeline(tl.clone());
        let mut payload = tl.to_payload();
        payload.created_clip_ids = Some(vec![clip_id]);
        payload.project = Some(state.project_meta_payload());
        drop(tl);
        if let Some(root) = root_track_id {
            crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, &root);
        }
        payload
    } else {
        // ── 非合并模式：每条轨道独立处理，重叠音符拆分为不同 clip ──
        let resolved_indices: Vec<usize> = if track_indices.is_empty() {
            (0..parse_result.track_notes.len()).collect()
        } else {
            track_indices
                .iter()
                .filter(|&&idx| idx < parse_result.track_notes.len())
                .copied()
                .collect()
        };

        if resolved_indices.is_empty() {
            midi_log("import_midi_as_clip: no_tracks");
            return error_payload("no_notes_in_track");
        }

        state.checkpoint_timeline(&tl, crate::state::HistoryOp::ImportMidi);

        let mut created_clip_ids: Vec<String> = vec![];
        let mut created_track_ids: Vec<String> = vec![];
        let mut current_track_id = track_id;

        for (ti, &track_idx) in resolved_indices.iter().enumerate() {
            let track_notes = &parse_result.track_notes[track_idx];
            if track_notes.is_empty() {
                continue;
            }

            let track_info = &parse_result.tracks[track_idx];
            let track_name = if track_info.name.is_empty() {
                format!("Track {}", track_idx + 1)
            } else {
                track_info.name.clone()
            };

            let groups = midi_import::split_notes_into_non_overlapping_groups(track_notes);

            for (gi, group) in groups.iter().enumerate() {
                if group.is_empty() {
                    continue;
                }

                if current_track_id.is_none() {
                    let new_id = tl.add_track(Some(file_stem.clone()), None, None);
                    created_track_ids.push(new_id.clone());
                    current_track_id = Some(new_id);
                }

                let first_start = group
                    .iter()
                    .map(|n| n.start_sec)
                    .fold(f64::INFINITY, f64::min);
                let last_end = group.iter().map(|n| n.end_sec).fold(0.0f64, f64::max);
                let normalized: Vec<midi_import::MidiNoteEvent> = group
                    .iter()
                    .map(|n| midi_import::MidiNoteEvent {
                        start_sec: if close_gap {
                            n.start_sec - first_start
                        } else {
                            n.start_sec
                        },
                        end_sec: if close_gap {
                            n.end_sec - first_start
                        } else {
                            n.end_sec
                        },
                        note: n.note,
                        velocity: n.velocity,
                        channel: n.channel,
                    })
                    .collect();

                let pitch_range = {
                    let min_note = normalized.iter().fold(127.0f32, |m, n| m.min(n.note));
                    let max_note = normalized.iter().fold(0.0f32, |m, n| m.max(n.note));
                    Some(crate::models::PitchRange {
                        min: min_note,
                        max: max_note,
                    })
                };

                let clip_name = if groups.len() > 1 {
                    format!("{} - {} #{}", file_stem, track_name, gi + 1)
                } else {
                    format!("{} - {}", file_stem, track_name)
                };

                let clip_id = tl.add_clip(
                    current_track_id.clone(),
                    Some(clip_name),
                    Some(start_sec),
                    Some(if close_gap {
                        (last_end - first_start).max(0.1)
                    } else {
                        last_end.max(0.1)
                    }),
                    None,
                );

                if let Some(clip) = tl.clips.iter_mut().find(|c| c.id == clip_id) {
                    clip.midi_note_data = Some(normalized);
                    clip.midi_fill_gaps = fill;
                    clip.pitch_range = pitch_range;
                    clip.color = "cyan".to_string();
                    clip.source_path = None;
                    clip.source_path_relative = None;
                }

                created_clip_ids.push(clip_id);

                if gi + 1 < groups.len() {
                    let insert_pos = current_track_id
                        .as_ref()
                        .and_then(|tid| tl.tracks.iter().position(|t| t.id == *tid))
                        .map(|pos| pos + 1);
                    let new_name = if groups.len() > 1 {
                        format!("{} - {} #{}", file_stem, track_name, gi + 2)
                    } else {
                        format!("{} - {}", file_stem, track_name)
                    };
                    let new_id = tl.add_track(Some(new_name), None, insert_pos);
                    created_track_ids.push(new_id.clone());
                    current_track_id = Some(new_id);
                }
            }

            if ti + 1 < resolved_indices.len() {
                let insert_pos = current_track_id
                    .as_ref()
                    .and_then(|tid| tl.tracks.iter().position(|t| t.id == *tid))
                    .map(|pos| pos + 1);
                let new_id = tl.add_track(Some(file_stem.clone()), None, insert_pos);
                created_track_ids.push(new_id.clone());
                current_track_id = Some(new_id);
            }
        }

        midi_log(format!(
            "import_midi_as_clip: multi_track_merge=false created_clips={} created_tracks={}",
            created_clip_ids.len(),
            created_track_ids.len()
        ));

        // Auto-group all created pitch reference clips when multi_track_merge is disabled
        if created_clip_ids.len() >= 2 {
            tl.group_clips(&created_clip_ids);
        }

        let mut root_track_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for clip_id in &created_clip_ids {
            if let Some(clip) = tl.clips.iter().find(|c| c.id == *clip_id) {
                if let Some(root) = tl.resolve_root_track_id(&clip.track_id) {
                    root_track_ids.insert(root);
                }
            }
        }
        tl.sync_clip_takes_from_flat();
        state.audio_engine.update_timeline(tl.clone());
        let mut payload = tl.to_payload();
        payload.created_clip_ids = Some(created_clip_ids);
        payload.created_track_ids = Some(created_track_ids);
        payload.project = Some(state.project_meta_payload());
        drop(tl);
        for root in &root_track_ids {
            crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, root);
        }

        if let Some(ref guid) = clipboard_guid {
            if !guid.is_empty() {
                take_clipboard_midi(state, guid);
            }
        }

        payload
    }
}

/// 替换指定 MIDI clip 的音符数据。
///
/// 从新的 MIDI 文件中解析音符，替换已有 MIDI clip 的 `midi_note_data`、
/// `midi_pitch_bends`、`pitch_range` 和时长。
///
/// 返回完整的 timeline state payload，以便前端更新 Redux store。
pub(super) fn replace_midi_clip_data(
    state: &AppState,
    clip_id: String,
    midi_path: String,
    track_indices: Vec<usize>,
    fill_gaps: Option<bool>,
    note_bpm_mode: Option<String>,
    specified_bpm: Option<f64>,
    import_midi_bpm_as_project: Option<bool>,
    clipboard_guid: Option<String>,
    close_leading_gap: Option<bool>,
) -> crate::models::TimelineStatePayload {
    midi_log(format!(
        "replace_midi_clip_data: clip_id={} path={} track_indices={:?} fill_gaps={:?} close_leading_gap={:?}",
        clip_id, midi_path, track_indices, fill_gaps, close_leading_gap
    ));

    // 先短暂锁定读取 bpm；MIDI 磁盘解析放在锁外（同 import_midi_to_pitch：
    // 同步命令持锁解析会冻结主线程与其他所有命令）。
    let project_bpm = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.bpm
    };

    let (mut parse_result, _source_stem) = match resolve_midi_source(
        state,
        Some(&midi_path).filter(|s| !s.is_empty()),
        clipboard_guid.as_deref(),
        Some(project_bpm),
    ) {
        Ok((r, stem)) => (r, stem),
        Err(e) => {
            midi_log(format!("replace_midi_clip_data: parse_error={}", e));
            return error_payload(&e);
        }
    };

    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

    let initial_bpm = parse_result.initial_bpm;

    // ── BPM 重映射 ──
    let import_as_project = import_midi_bpm_as_project.unwrap_or(false);
    if import_as_project {
        set_project_bpm_syncing_tempo_map(&mut tl, initial_bpm);
    }

    let mode = note_bpm_mode.as_deref().unwrap_or("midi");
    let target_bpm: Option<f64> = match mode {
        "project" => {
            if import_as_project {
                None
            } else {
                Some(project_bpm)
            }
        }
        "specified" => specified_bpm.filter(|&b| b > 0.0 && b.is_finite()),
        _ => None,
    };

    if let Some(tbpm) = target_bpm {
        let scale = initial_bpm / tbpm;
        for track_notes in &mut parse_result.track_notes {
            for note in track_notes {
                note.start_sec *= scale;
                note.end_sec *= scale;
            }
        }
    }

    let file_stem = _source_stem.unwrap_or_else(|| "MIDI".to_string());

    let fill = fill_gaps.unwrap_or(false);

    // 合并选中轨道的音符
    let notes: Vec<midi_import::MidiNoteEvent> = {
        let mut all: Vec<_> = if track_indices.is_empty() {
            parse_result.track_notes.into_iter().flatten().collect()
        } else {
            track_indices
                .iter()
                .filter_map(|&idx| parse_result.track_notes.get(idx))
                .flatten()
                .cloned()
                .collect()
        };
        all.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    };

    if notes.is_empty() {
        midi_log("replace_midi_clip_data: no_notes");
        return error_payload("no_notes_in_track");
    }

    let close_gap = close_leading_gap.unwrap_or(true);
    let last_end = notes.iter().map(|n| n.end_sec).fold(0.0f64, f64::max);
    let first_start = notes
        .iter()
        .map(|n| n.start_sec)
        .fold(f64::INFINITY, f64::min);
    let length_sec = if close_gap {
        (last_end - first_start).max(0.1)
    } else {
        last_end.max(0.1)
    };
    let normalized_notes: Vec<midi_import::MidiNoteEvent> = notes
        .into_iter()
        .map(|n| midi_import::MidiNoteEvent {
            start_sec: if close_gap {
                n.start_sec - first_start
            } else {
                n.start_sec
            },
            end_sec: if close_gap {
                n.end_sec - first_start
            } else {
                n.end_sec
            },
            note: n.note,
            velocity: n.velocity,
            channel: n.channel,
        })
        .collect();

    let pitch_range = {
        let min_note = normalized_notes.iter().fold(127.0f32, |m, n| m.min(n.note));
        let max_note = normalized_notes.iter().fold(0.0f32, |m, n| m.max(n.note));
        Some(crate::models::PitchRange {
            min: min_note,
            max: max_note,
        })
    };

    state.checkpoint_timeline(&tl, crate::state::HistoryOp::EditMidi);

    // 找到目标 clip 并替换其 MIDI 数据
    if let Some(clip) = tl.clips.iter_mut().find(|c| c.id == clip_id) {
        clip.name = file_stem;
        clip.length_sec = length_sec;
        clip.midi_note_data = Some(normalized_notes);
        clip.midi_fill_gaps = fill;
        clip.pitch_range = pitch_range;
        clip.source_path = None;
        clip.source_path_relative = None;
        // 内容整体换成了"新文件的正序音符 [0, length_sec]"（不变式 PN 的源域
        // 坐标），因此描述**旧内容**的消费参数必须一并归零：否则旧窗口
        // （例如 `source_start_sec = 3.0`）会与新音符错位，渲染端按窗口求交
        // 时音高线整体偏移或消失。取值与 `import_midi_as_clip` 的新建路径、
        // 以及 `glue_pitch_clips` 重新谱写音符时的做法完全一致。
        clip.source_start_sec = 0.0;
        clip.source_end_sec = length_sec;
        clip.playback_rate = 1.0;
        clip.reversed = false;
    } else {
        midi_log(format!(
            "replace_midi_clip_data: clip_not_found clip_id={}",
            clip_id
        ));
        return error_payload("clip_not_found");
    }

    midi_log(format!(
        "replace_midi_clip_data: replaced clip_id={} length_sec={:.3}",
        clip_id, length_sec
    ));

    let root_track_id = tl.resolve_root_track_id(
        &tl.clips
            .iter()
            .find(|c| c.id == clip_id)
            .map(|c| c.track_id.clone())
            .unwrap_or_default(),
    );
    tl.sync_clip_takes_from_flat();
    state.audio_engine.update_timeline(tl.clone());
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    drop(tl);
    if let Some(root) = root_track_id {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, &root);
    }

    if let Some(ref guid) = clipboard_guid {
        if !guid.is_empty() {
            take_clipboard_midi(state, guid);
        }
    }

    payload
}
