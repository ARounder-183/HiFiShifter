//! MIDI 导入命令：解析 → 算帧窗口 → 写 `pitch_edit`，独立 App 与 ARA 插件共用。
//!
//! 【为什么在 kernel】整条链路只依赖 `TimelineState` 与 `crate::midi_import` 的纯
//! 函数，既不碰音频设备也不碰宿主写 API。两边各写一份的代价是同一个 MIDI 文件会落到
//! 不同的帧上 —— 与 `midi_export` 被搬进内核是同一条理由。
//!
//! 【为什么只有一个 trait 差异】宿主差异只有两处：剪贴板 MIDI 载荷缓存归谁持有，
//! 以及曲线写完后谁去发布（App 推给音频引擎，插件只排后台渲染）。后者已经在
//! [`ParamHost::publish_timeline`] 里，所以这里只需要补上前者。
//!
//! 【为什么不做 `import_midi_as_clip`】它创建的是"只有音符、没有音频源"的片段，而
//! 插件的时间线是**宿主清单的投影**（`workspace_timeline_locked` 只保留已分配 region
//! 的 clip）。在插件里凭空造一个本地片段，下一次宿主同步就会把它抹掉 —— 那不是
//! 未实现，是做不到。写进参数曲线则完全属于插件自己的权威范围。

use std::collections::VecDeque;
use std::sync::Mutex;

use super::{ParamHost, ParamSelectionWindow, SelectionFrameRange};
use crate::midi_import::{self, MidiTrackInfo};
use crate::state::{HistoryOp, PitchAnalysisAlgo, TimelineState, Track};

fn midi_log(message: impl AsRef<str>) {
    // 沿用 App 侧既有的 tag 与级别：这条链路的信息量集中在失败原因上，
    // 默认级别就要看得见，否则用户报障时无从下手。
    log::error!("[midi_import] {}", message.as_ref());
}

/// 剪贴板 MIDI 载荷的内存缓存上限（条）。
pub const CLIPBOARD_MIDI_CACHE_MAX: usize = 16;

/// MIDI 导入所需的最小宿主边界。
///
/// 除剪贴板缓存外全部复用 [`ParamHost`]：`timeline` 取曲线、`checkpoint_timeline`
/// 记撤销步、`publish_timeline` 把结果交给各自的消费者。
pub trait MidiImportHost: ParamHost {
    /// 剪贴板 MIDI 载荷缓存。App 与插件各自持有同一形状的 LRU。
    fn clipboard_midi(&self) -> &Mutex<VecDeque<(String, Vec<u8>)>>;
}

/// 读取剪贴板 MIDI 载荷（不消费；导入命令完成后再 take 掉）。
pub fn peek_clipboard_midi(host: &impl MidiImportHost, guid: &str) -> Option<Vec<u8>> {
    let cache = host
        .clipboard_midi()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .iter()
        .find(|(key, _)| key == guid)
        .map(|(_, bytes)| bytes.clone())
}

/// 移除剪贴板 MIDI 载荷（导入完成后一次性消费）。
pub fn take_clipboard_midi(host: &impl MidiImportHost, guid: &str) {
    let mut cache = host
        .clipboard_midi()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = cache.iter().position(|(key, _)| key == guid) {
        cache.remove(pos);
    }
}

/// 存入剪贴板 MIDI 载荷，按**插入顺序**淘汰，最多保留 [`CLIPBOARD_MIDI_CACHE_MAX`] 条。
///
/// 【为什么不能用 HashMap】曾经的实现用 `HashMap::keys().next()` 当"最旧"，
/// 但那只是任意顺序：缓存溢出时可能把**刚插入**的条目自己挤掉，后续
/// `import_midi_*` 立刻报 `midi_clipboard_guid_not_found`。
pub fn put_clipboard_midi(host: &impl MidiImportHost, guid: String, bytes: Vec<u8>) {
    let mut cache = host
        .clipboard_midi()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = cache.iter().position(|(key, _)| *key == guid) {
        cache.remove(pos);
    }
    cache.push_back((guid, bytes));
    while cache.len() > CLIPBOARD_MIDI_CACHE_MAX {
        cache.pop_front();
    }
}

/// 解析 MIDI 来源：可以是文件路径或剪贴板 GUID。
///
/// 返回 `(MidiParseResult, Option<显示名称>)`。显示名称仅在文件来源时有值（文件 stem）。
pub fn resolve_midi_source(
    host: &impl MidiImportHost,
    midi_path: Option<&String>,
    clipboard_guid: Option<&str>,
    fallback_bpm: Option<f64>,
) -> Result<(midi_import::MidiParseResult, Option<String>), String> {
    if let Some(guid) = clipboard_guid {
        if guid.is_empty() {
            return Err("invalid_guid".to_string());
        }
        let bytes = peek_clipboard_midi(host, guid)
            .ok_or_else(|| "midi_clipboard_guid_not_found".to_string())?;
        let result = midi_import::parse_midi_bytes(&bytes, fallback_bpm)?;
        Ok((result, None))
    } else if let Some(path) = midi_path {
        if path.is_empty() {
            return Err("file_not_found".to_string());
        }
        let p = std::path::Path::new(path.as_str());
        if !p.exists() {
            return Err("file_not_found".to_string());
        }
        let file_stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string());
        let result = midi_import::parse_midi_file(p, fallback_bpm)?;
        Ok((result, file_stem))
    } else {
        Err("no_source_specified".to_string())
    }
}

/// 设置工程 BPM，并在 Tempo Map 存在时同步 0 位置点（保持回退一致）。
pub fn set_project_bpm_syncing_tempo_map(tl: &mut TimelineState, bpm: f64) {
    let clamped = bpm.clamp(10.0, 960.0);
    tl.bpm = clamped;
    if let Some(points) = tl.tempo_map.as_mut() {
        if let Some(first) = points.first_mut() {
            first.bpm = clamped;
        }
    }
}

/// 目标根轨必须既在合成、又有音高分析算法 —— 否则写进去的曲线没有任何消费者。
pub fn validate_midi_import_target(track: &Track) -> Result<(), &'static str> {
    if !track.compose_enabled {
        return Err("pitch_requires_compose");
    }

    if matches!(track.pitch_analysis_algo, PitchAnalysisAlgo::None) {
        return Err("pitch_requires_algo");
    }

    Ok(())
}

/// 读取 MIDI 文件（或剪贴板缓存）并返回轨道摘要列表。
pub fn get_midi_tracks(
    host: &impl MidiImportHost,
    midi_path: String,
    clipboard_guid: Option<String>,
) -> serde_json::Value {
    midi_log(format!(
        "get_midi_tracks: path={midi_path} clipboard_guid={:?}",
        clipboard_guid
    ));

    let parse_result = match resolve_midi_source(
        host,
        Some(&midi_path).filter(|s| !s.is_empty()),
        clipboard_guid.as_deref(),
        None,
    ) {
        Ok((r, _)) => r,
        Err(e) => {
            midi_log(format!("get_midi_tracks: error={e}"));
            return serde_json::json!({"ok": false, "error": e});
        }
    };

    let tracks_with_notes: Vec<&MidiTrackInfo> = parse_result
        .tracks
        .iter()
        .filter(|t| t.note_count > 0)
        .collect();

    midi_log(format!(
        "get_midi_tracks: parsed tracks_total={} tracks_with_notes={}",
        parse_result.tracks.len(),
        tracks_with_notes.len()
    ));

    serde_json::json!({
        "ok": true,
        "tracks": tracks_with_notes,
        "initial_bpm": parse_result.initial_bpm,
        "has_bpm": parse_result.has_tempo,
        "has_time_signature": !parse_result.time_signature_events.is_empty(),
        "has_key_signature": !parse_result.key_signature_events.is_empty(),
        "tempo_point_count": parse_result.tempo_events.len(),
        "time_signature_count": parse_result.time_signature_events.len(),
        "key_signature_count": parse_result.key_signature_events.len(),
    })
}

/// 把系统剪贴板里的 "Standard MIDI File" 字节存进内存缓存并解析。
///
/// 不创建临时文件。返回 GUID、轨道列表和初始 BPM，供前端弹窗展示；MIDI 原始字节
/// 留在缓存里，后续导入命令通过 GUID 引用。
///
/// `midi_data` 是调用方（平台层）读到的结果：任何读取失败都折成
/// `midi_clipboard_empty`，真实原因只进日志 —— 前端对这个错误只有一句"剪贴板里没有
/// MIDI"，把内部错误码透出去只会变成无法据以行动的技术信息。
pub fn read_midi_clipboard_to_memory(
    host: &impl MidiImportHost,
    midi_data: Result<Vec<u8>, String>,
) -> serde_json::Value {
    midi_log("read_midi_clipboard_to_memory: start");

    let midi_data = match midi_data {
        Ok(data) => data,
        Err(e) => {
            midi_log(format!(
                "read_midi_clipboard_to_memory: clipboard_error={e}"
            ));
            return serde_json::json!({"ok": false, "error": "midi_clipboard_empty"});
        }
    };

    if midi_data.is_empty() {
        midi_log("read_midi_clipboard_to_memory: clipboard_empty");
        return serde_json::json!({"ok": false, "error": "midi_clipboard_empty"});
    }

    // 用 blake3 哈希生成 GUID（前 8 字节 → 16 位 hex）
    let hash = blake3::hash(&midi_data);
    let guid: String = hash.as_bytes()[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    midi_log(format!("read_midi_clipboard_to_memory: guid={guid}"));

    let parse_result = match midi_import::parse_midi_bytes(&midi_data, None) {
        Ok(r) => r,
        Err(e) => {
            midi_log(format!("read_midi_clipboard_to_memory: parse_error={e}"));
            return serde_json::json!({"ok": false, "error": format!("midi_parse_error: {}", e)});
        }
    };

    put_clipboard_midi(host, guid.clone(), midi_data);

    let tracks_with_notes: Vec<&MidiTrackInfo> = parse_result
        .tracks
        .iter()
        .filter(|t| t.note_count > 0)
        .collect();

    midi_log(format!(
        "read_midi_clipboard_to_memory: parsed tracks_total={} tracks_with_notes={} initial_bpm={:.2}",
        parse_result.tracks.len(),
        tracks_with_notes.len(),
        parse_result.initial_bpm
    ));

    serde_json::json!({
        "ok": true,
        "guid": guid,
        "tracks": tracks_with_notes,
        "initial_bpm": parse_result.initial_bpm,
        "has_bpm": parse_result.has_tempo,
        "has_time_signature": !parse_result.time_signature_events.is_empty(),
        "has_key_signature": !parse_result.key_signature_events.is_empty(),
        "tempo_point_count": parse_result.tempo_events.len(),
        "time_signature_count": parse_result.time_signature_events.len(),
        "key_signature_count": parse_result.key_signature_events.len(),
    })
}

/// `import_midi_to_pitch` 的全部入参（前端 `importMidiToPitch` 的 camelCase 形状）。
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportMidiToPitchRequest {
    #[serde(default)]
    pub midi_path: String,
    #[serde(default)]
    pub track_indices: Vec<usize>,
    #[serde(default)]
    pub selection_ranges: Option<Vec<SelectionFrameRange>>,
    #[serde(default)]
    pub fill_gaps: Option<bool>,
    #[serde(default)]
    pub note_bpm_mode: Option<String>,
    #[serde(default)]
    pub specified_bpm: Option<f64>,
    #[serde(default)]
    pub import_midi_bpm_as_project: Option<bool>,
    #[serde(default)]
    pub clipboard_guid: Option<String>,
    #[serde(default)]
    pub close_leading_gap: Option<bool>,
}

/// 将 MIDI 文件中指定轨道的音符写入当前选中根轨的 `pitch_edit`。
///
/// - 使用工程 BPM 作为 Tempo 回退
/// - 偏移量为光标位置或选区首段起点对应秒，**第一个音符对齐该偏移量**（整体平移）
/// - 支持多选区约束：只写入落在任一段内的帧，断层保持原值（见下方的掩码回滚）
pub fn import_midi_to_pitch(
    host: &impl MidiImportHost,
    request: ImportMidiToPitchRequest,
) -> serde_json::Value {
    let ImportMidiToPitchRequest {
        midi_path,
        track_indices,
        selection_ranges,
        fill_gaps,
        note_bpm_mode,
        specified_bpm,
        import_midi_bpm_as_project,
        clipboard_guid,
        close_leading_gap,
    } = request;
    midi_log(format!(
        "import_midi_to_pitch: path={} clipboard_guid={:?} track_indices={:?} selection_ranges={:?} fill_gaps={:?} note_bpm_mode={:?} specified_bpm={:?} import_midi_bpm_as_project={:?} close_leading_gap={:?}",
        midi_path, clipboard_guid, track_indices, selection_ranges, fill_gaps, note_bpm_mode, specified_bpm, import_midi_bpm_as_project, close_leading_gap
    ));

    // 先短暂锁定读取 bpm / playhead 等信息；MIDI 磁盘解析放在锁外 ——
    // 本命令是同步命令（主线程），持锁解析会在阻塞主线程的同时冻结
    // 所有其他命令与 UI 轮询。
    let (project_bpm, playhead_sec, frame_period_ms_raw) = {
        let tl = host.timeline().lock().unwrap_or_else(|e| e.into_inner());
        (tl.bpm, tl.playhead_sec, tl.frame_period_ms().max(0.1))
    };

    // 使用工程 BPM 作为 fallback tempo（与 Reaper 剪贴板路径一致）
    let parse_result = match resolve_midi_source(
        host,
        Some(&midi_path).filter(|s| !s.is_empty()),
        clipboard_guid.as_deref(),
        Some(project_bpm),
    ) {
        Ok((r, _)) => r,
        Err(e) => {
            midi_log(format!("import_midi_to_pitch: parse_error={e}"));
            return serde_json::json!({"ok": false, "error": e});
        }
    };

    let mut tl = host.timeline().lock().unwrap_or_else(|e| e.into_inner());

    let initial_bpm = parse_result.initial_bpm;

    // ── BPM 导入与重映射 ──
    let import_as_project = import_midi_bpm_as_project.unwrap_or(false);
    if import_as_project {
        set_project_bpm_syncing_tempo_map(&mut tl, initial_bpm);
        midi_log(format!(
            "import_midi_to_pitch: set_project_bpm from {project_bpm} to {initial_bpm}"
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

    // 收集要写入的音符：合并所有选中轨道的音符
    let mut notes: Vec<midi_import::MidiNoteEvent> = {
        let mut all: Vec<midi_import::MidiNoteEvent> = if track_indices.is_empty() {
            // 未指定轨道则合并所有轨道
            parse_result.track_notes.into_iter().flatten().collect()
        } else {
            track_indices
                .iter()
                .filter_map(|&idx| parse_result.track_notes.get(idx))
                .flatten()
                .cloned()
                .collect()
        };
        if all.is_empty() {
            midi_log("import_midi_to_pitch: no_notes_in_track");
            return serde_json::json!({"ok": false, "error": "no_notes_in_track"});
        }
        all.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    };

    // 应用 BPM 重映射
    if let Some(tbpm) = target_bpm {
        let scale = initial_bpm / tbpm;
        midi_log(format!(
            "import_midi_to_pitch: bpm_remap initial_bpm={initial_bpm:.2} target_bpm={tbpm:.2} scale={scale:.6}"
        ));
        for note in &mut notes {
            note.start_sec *= scale;
            note.end_sec *= scale;
        }
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

    host.checkpoint_timeline(&tl, HistoryOp::ImportMidi);

    let Some(entry) = tl.params_by_root_track.get_mut(&root_track_id) else {
        midi_log(format!(
            "import_midi_to_pitch: params_missing root_track_id={}",
            root_track_id
        ));
        return serde_json::json!({"ok": false, "error": "params_missing"});
    };

    // 计算对齐偏移量和写入范围
    //
    // 多选区（selection_ranges）：对齐偏移仍取**首段起点**，写入范围为末段末尾；
    // 写完后再按窗口做一次**掩码回滚** —— 落在断层里的帧恢复导入前的值，
    // 保证"导入不会把两段之间的缺口填上"。
    let close_gap = close_leading_gap.unwrap_or(true);
    let first_start = notes
        .iter()
        .map(|n| n.start_sec)
        .fold(f64::INFINITY, f64::min);
    let selection_window = ParamSelectionWindow::new(selection_ranges.clone(), None, None);
    let (align_offset, clamp_range_end) = if let Some(sel_start) = selection_window.origin() {
        let offset_sec = (sel_start as f64 * frame_period_ms_raw) / 1000.0;
        let ao = if close_gap {
            offset_sec - first_start
        } else {
            offset_sec
        };
        let max_frame = selection_window
            .end_bound()
            .unwrap_or(usize::MAX)
            .min(entry.pitch_edit.len());
        midi_log(format!(
            "import_midi_to_pitch: selection mode offset_sec={:.3} align_offset={:.3} clamp_len={} close_gap={} multi_range={}",
            offset_sec,
            ao,
            max_frame,
            close_gap,
            !selection_window.is_empty() && selection_ranges.is_some()
        ));
        (ao, Some(max_frame))
    } else {
        let ao = if close_gap {
            playhead_sec - first_start
        } else {
            playhead_sec
        };
        midi_log(format!(
            "import_midi_to_pitch: playhead mode offset_sec={:.3} align_offset={:.3} close_gap={}",
            playhead_sec, ao, close_gap
        ));
        (ao, None)
    };

    // 多选区掩码回滚用的导入前快照（仅在确实存在多段约束时才需要）
    let mask_before: Option<Vec<f32>> =
        if selection_ranges.is_some() && !selection_window.is_empty() {
            Some(entry.pitch_edit.clone())
        } else {
            None
        };

    let target_slice = if let Some(clamp_len) = clamp_range_end {
        &mut entry.pitch_edit[..clamp_len]
    } else {
        &mut entry.pitch_edit[..]
    };

    // 先清除目标范围，避免已有编辑阻挡新导入的 MIDI 音符
    midi_import::clear_pitch_edit_range_for_notes(
        &notes,
        frame_period_ms,
        target_slice,
        align_offset,
    );

    let touched =
        midi_import::write_notes_to_pitch_edit(&notes, frame_period_ms, target_slice, align_offset);

    // 填补音符之间的空隙（仅在导入的音符范围内）
    if fill_gaps.unwrap_or(false) {
        // 计算导入音符的实际帧范围，避免 fill_gaps_in_pitch_edit
        // 在已有非零音高值的历史编辑区域产生意外的填充
        let mut min_frame = usize::MAX;
        let mut max_frame = 0usize;
        for note in &notes {
            let start_sec = note.start_sec + align_offset;
            let end_sec = note.end_sec + align_offset;
            if start_sec < 0.0 || !start_sec.is_finite() || !end_sec.is_finite() {
                continue;
            }
            let sf = ((start_sec * 1000.0) / frame_period_ms).round() as usize;
            let ef = ((end_sec * 1000.0) / frame_period_ms).round() as usize;
            if sf < entry.pitch_edit.len() {
                min_frame = min_frame.min(sf);
                max_frame = max_frame.max(ef.min(entry.pitch_edit.len()));
            }
        }
        if min_frame < max_frame && max_frame <= entry.pitch_edit.len() {
            let filled =
                midi_import::fill_gaps_in_pitch_edit(&mut entry.pitch_edit[min_frame..max_frame]);
            if filled > 0 {
                midi_log(format!("import_midi_to_pitch: fill_gaps filled={}", filled));
            }
        }
    }

    // 多选区掩码回滚：断层（未选中）的帧恢复导入前的值 —— 导入同样不得
    // 把两段之间的缺口填平（与前端"不合并断层"语义一致）。
    if let Some(before) = mask_before {
        let mut reverted = 0usize;
        for idx in 0..entry.pitch_edit.len() {
            if selection_window.allows(idx) {
                continue;
            }
            if entry.pitch_edit[idx] != before[idx] {
                entry.pitch_edit[idx] = before[idx];
                reverted += 1;
            }
        }
        if reverted > 0 {
            midi_log(format!(
                "import_midi_to_pitch: multi-range mask reverted frames={}",
                reverted
            ));
        }
    }

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

    tl.sync_clip_takes_from_flat();
    host.publish_timeline(tl.clone());

    if let Some(ref guid) = clipboard_guid {
        if !guid.is_empty() {
            take_clipboard_midi(host, guid);
        }
    }

    serde_json::json!({
        "ok": true,
        "notes_imported": notes.len(),
        "frames_touched": touched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TimelineState;

    /// 最小合法 SMF：格式 0、单轨、480 ticks/四分音符，C4 从 0 弹到半秒。
    ///
    /// 【为什么手写字节而不是带 fixture 文件】这条链路的测试要跑在 App 与插件两侧，
    /// 而两侧的工作目录不同；把二十个字节写在测试里，比让两侧各猜一次相对路径稳。
    fn minimal_midi() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes()); // format 0
        bytes.extend_from_slice(&1u16.to_be_bytes()); // one track
        bytes.extend_from_slice(&480u16.to_be_bytes()); // ticks per quarter note
        let mut track = Vec::new();
        track.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on C4
        track.extend_from_slice(&[0x83, 0x60, 0x80, 0x3C, 0x40]); // +480 ticks, note off
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]); // end of track
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&track);
        bytes
    }

    fn midi_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("hfs-midi-import-kernel");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_midi(name: &str) -> String {
        let path = midi_dir().join(name);
        std::fs::write(&path, minimal_midi()).unwrap();
        path.to_string_lossy().into_owned()
    }

    struct Host {
        timeline: Mutex<TimelineState>,
        clipboard: Mutex<VecDeque<(String, Vec<u8>)>>,
        checkpoints: Mutex<Vec<HistoryOp>>,
        published: Mutex<usize>,
    }

    impl ParamHost for Host {
        fn timeline(&self) -> &Mutex<TimelineState> {
            &self.timeline
        }
        fn checkpoint_timeline(&self, _timeline: &TimelineState, operation: HistoryOp) {
            self.checkpoints.lock().unwrap().push(operation);
        }
        fn mark_dirty(&self) {}
        fn publish_timeline(&self, _timeline: TimelineState) {
            *self.published.lock().unwrap() += 1;
        }
    }

    impl MidiImportHost for Host {
        fn clipboard_midi(&self) -> &Mutex<VecDeque<(String, Vec<u8>)>> {
            &self.clipboard
        }
    }

    fn host(compose: bool, algo: &str) -> Host {
        let timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"shared midi","order":0,
                       "compose_enabled":compose,"pitch_analysis_algo":algo}],
            "clips":[],"bpm":120.0,"project_sec":1.0,
            "selected_track_id":"track","playhead_sec":0.0
        }))
        .unwrap();
        Host {
            timeline: Mutex::new(timeline),
            clipboard: Mutex::new(VecDeque::new()),
            checkpoints: Mutex::new(Vec::new()),
            published: Mutex::new(0),
        }
    }

    fn request(midi_path: String) -> ImportMidiToPitchRequest {
        ImportMidiToPitchRequest {
            midi_path,
            ..Default::default()
        }
    }

    /// 共享编排必须真的写进曲线、记一次撤销步、并发布给宿主。
    #[test]
    fn imports_notes_into_the_pitch_curve() {
        let host = host(true, "nsf_hifigan_onnx");
        let path = write_midi("note.mid");

        let result = import_midi_to_pitch(&host, request(path));
        assert_eq!(result["ok"], true, "{result}");
        assert_eq!(result["notes_imported"], 1);

        let timeline = host.timeline.lock().unwrap();
        let curve = &timeline.params_by_root_track["track"];
        // 480 ticks/四分音符 + 回退 120 BPM → 音符持续 0.5 s；5 ms/帧 → 前 100 帧。
        assert_eq!(curve.pitch_edit[0], 60.0);
        assert_eq!(curve.pitch_edit[99], 60.0);
        assert_eq!(curve.pitch_edit[100], 0.0);
        assert!(curve.pitch_edit_user_modified);
        drop(timeline);

        assert_eq!(
            host.checkpoints.lock().unwrap().as_slice(),
            &[HistoryOp::ImportMidi]
        );
        assert_eq!(*host.published.lock().unwrap(), 1);
    }

    /// 目标根轨不能接收曲线时，必须给出**可据以行动**的错误码，而不是静默成功。
    #[test]
    fn rejects_targets_that_cannot_carry_a_curve() {
        let path = write_midi("note.mid");

        let quiet = host(false, "nsf_hifigan_onnx");
        assert_eq!(
            import_midi_to_pitch(&quiet, request(path.clone()))["error"],
            "pitch_requires_compose"
        );

        let algo_none = host(true, "none");
        assert_eq!(
            import_midi_to_pitch(&algo_none, request(path.clone()))["error"],
            "pitch_requires_algo"
        );

        let host = host(true, "nsf_hifigan_onnx");
        host.timeline.lock().unwrap().selected_track_id = None;
        assert_eq!(
            import_midi_to_pitch(&host, request(path))["error"],
            "no_pitch_line_selected"
        );
        // 被拒绝时不得留下撤销步或发布。
        assert!(host.checkpoints.lock().unwrap().is_empty());
        assert_eq!(*host.published.lock().unwrap(), 0);
    }

    /// 剪贴板来源按 GUID 取字节；GUID 失效必须是独立错误码，不能退化成"文件不存在"。
    #[test]
    fn resolves_clipboard_payloads_by_guid() {
        let host = host(true, "nsf_hifigan_onnx");
        assert_eq!(
            import_midi_to_pitch(
                &host,
                ImportMidiToPitchRequest {
                    clipboard_guid: Some("nope".into()),
                    ..request(String::new())
                }
            )["error"],
            "midi_clipboard_guid_not_found"
        );

        let guid = "abc123".to_string();
        put_clipboard_midi(&host, guid.clone(), minimal_midi());
        let tracks = get_midi_tracks(&host, String::new(), Some(guid.clone()));
        assert_eq!(tracks["ok"], true, "{tracks}");
        assert_eq!(tracks["tracks"].as_array().unwrap().len(), 1);
        assert_eq!(tracks["tracks"][0]["note_count"], 1);

        let result = import_midi_to_pitch(
            &host,
            ImportMidiToPitchRequest {
                clipboard_guid: Some(guid.clone()),
                ..request(String::new())
            },
        );
        assert_eq!(result["ok"], true, "{result}");
        // 导入成功后载荷被消费，同一条 GUID 不再可用（避免过期字节被反复导入）。
        assert!(peek_clipboard_midi(&host, &guid).is_none());
    }

    /// 剪贴板读失败只对外报"剪贴板里没有 MIDI"，真实原因留在日志里。
    #[test]
    fn clipboard_read_failure_reports_an_empty_clipboard() {
        let host = host(true, "nsf_hifigan_onnx");
        let failed = read_midi_clipboard_to_memory(&host, Err("format not registered".into()));
        assert_eq!(failed["error"], "midi_clipboard_empty");
        let empty = read_midi_clipboard_to_memory(&host, Ok(Vec::new()));
        assert_eq!(empty["error"], "midi_clipboard_empty");

        let parsed = read_midi_clipboard_to_memory(&host, Ok(minimal_midi()));
        assert_eq!(parsed["ok"], true, "{parsed}");
        let guid = parsed["guid"].as_str().unwrap().to_string();
        assert!(peek_clipboard_midi(&host, &guid).is_some());
    }

    /// 多选区导入不得把两段之间的缺口填上（与前端"不合并断层"同一语义）。
    #[test]
    fn selection_window_keeps_the_gap_untouched() {
        let host = host(true, "nsf_hifigan_onnx");
        let path = write_midi("note.mid");

        // 只允许 [0,10) 与 [100,110) 两段：音符覆盖 0..100 帧，缺口 10..100 必须保持 0。
        let result = import_midi_to_pitch(
            &host,
            ImportMidiToPitchRequest {
                selection_ranges: Some(vec![
                    SelectionFrameRange {
                        start_frame: 0,
                        frame_count: 10,
                    },
                    SelectionFrameRange {
                        start_frame: 100,
                        frame_count: 10,
                    },
                ]),
                close_leading_gap: Some(true),
                ..request(path)
            },
        );
        assert_eq!(result["ok"], true, "{result}");

        let timeline = host.timeline.lock().unwrap();
        let curve = &timeline.params_by_root_track["track"].pitch_edit;
        assert_eq!(curve[0], 60.0);
        assert_eq!(curve[9], 60.0);
        assert_eq!(curve[10], 0.0);
        assert_eq!(curve[99], 0.0);
    }

    /// 剪贴板 LRU 只保留最近 16 条，且按**插入顺序**淘汰（不是任意顺序）。
    #[test]
    fn clipboard_cache_evicts_the_oldest_entry() {
        let host = host(true, "nsf_hifigan_onnx");
        for index in 0..CLIPBOARD_MIDI_CACHE_MAX + 2 {
            put_clipboard_midi(&host, format!("g{index}"), vec![index as u8]);
        }
        assert!(peek_clipboard_midi(&host, "g0").is_none());
        assert!(peek_clipboard_midi(&host, "g1").is_none());
        assert!(peek_clipboard_midi(&host, "g2").is_some());
        assert_eq!(
            host.clipboard.lock().unwrap().len(),
            CLIPBOARD_MIDI_CACHE_MAX
        );
    }
}
