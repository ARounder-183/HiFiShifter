//! Clip 音节块的查询与按需重分析。

use crate::state::AppState;
use hifishifter_kernel::pitch_analysis::syllable::{
    estimate_syllable_spans, SyllableConfig, TakeSyllableAnalysis, SYLLABLE_ALGORITHM,
    SYLLABLE_ALGORITHM_VERSION,
};
use hifishifter_kernel::pitch_analysis::{
    set_syllable_analysis_progress, SyllableAnalysisProgress,
};
use serde::Deserialize;
use serde::Serialize;
use tauri::Manager;
use tauri::State;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableSpanPayload {
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipSyllableAnalysisPayload {
    pub ok: bool,
    pub clip_id: String,
    pub take_id: String,
    /// 分析所得源域区间；为空表示回退整段一个块。
    pub source_spans_sec: Vec<SyllableSpanPayload>,
    pub algorithm: String,
    pub algorithm_version: u32,
    pub analyzed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamBlockMemberRefPayload {
    pub clip_id: String,
    pub take_id: String,
    pub syllable_id: String,
    #[serde(default)]
    pub source_start_sec: f64,
    #[serde(default)]
    pub source_end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamBlockMergeRecordPayload {
    pub merged_group_id: String,
    #[serde(default)]
    pub member_refs: Vec<ParamBlockMemberRefPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamBlockIntervalEditPayload {
    pub edit_id: String,
    #[serde(default)]
    pub member_refs: Vec<ParamBlockMemberRefPayload>,
    pub start_sec: f64,
    pub end_sec: f64,
    #[serde(default)]
    pub member_start_sec: f64,
    #[serde(default)]
    pub member_end_sec: f64,
    #[serde(default)]
    pub block_ids: Vec<String>,
    #[serde(default)]
    pub explicit: bool,
    #[serde(default)]
    pub deleted: bool,
    #[serde(default)]
    pub manual: bool,
    #[serde(default)]
    pub value: Option<f64>,
    #[serde(default)]
    pub transition_sec: f64,
}

fn source_digest(clip: &crate::state::Clip, frame_period_ms: f64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"take_syllable_v1");
    hasher.update(clip.source_path.as_deref().unwrap_or("").as_bytes());
    hasher.update(&clip.source_file_fingerprint.unwrap_or(0).to_le_bytes());
    hasher.update(&clip.source_file_size.unwrap_or(0).to_le_bytes());
    hasher.update(&clip.source_file_mtime.unwrap_or(0).to_le_bytes());
    hasher.update(&clip.duration_frames.unwrap_or(0).to_le_bytes());
    hasher.update(&clip.source_sample_rate.unwrap_or(0).to_le_bytes());
    hasher.update(&((frame_period_ms.max(0.1) * 1000.0).round() as u32).to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

fn payload(
    clip: &crate::state::Clip,
    take_id: &str,
    analysis: Option<&TakeSyllableAnalysis>,
) -> ClipSyllableAnalysisPayload {
    let spans = analysis
        .map(|a| {
            a.source_spans_sec
                .iter()
                .map(|s| SyllableSpanPayload {
                    start_sec: s.start_sec,
                    end_sec: s.end_sec,
                })
                .collect()
        })
        .unwrap_or_default();
    ClipSyllableAnalysisPayload {
        ok: true,
        clip_id: clip.id.clone(),
        take_id: take_id.to_string(),
        source_spans_sec: spans,
        algorithm: analysis
            .map(|a| a.algorithm.clone())
            .unwrap_or_else(|| SYLLABLE_ALGORITHM.to_string()),
        algorithm_version: analysis
            .map(|a| a.algorithm_version)
            .unwrap_or(SYLLABLE_ALGORITHM_VERSION),
        analyzed: analysis.is_some(),
    }
}

/// 后台补齐一个已导入 Clip 的 GAME/VAD 区间。
///
/// 导入路径只负责把媒体落盘与创建 Clip；模型推理不在命令线程里跑，也不依赖
/// 前端是否停留在参数面板。调用方先检查根轨道 Compose，本函数只处理“有源、
/// 非 MIDI、尚无当前分析缓存”的 Clip。
pub(super) fn schedule_clip_syllables(state: &AppState, clip_id: String) {
    /*
     * 导入完成时根轨道通常还没开 Compose（用户是导入后才去参数面板打开）。
     * 但“打开块层”不能因此变成永久不分析：先只登记一次待分析意图；真正
     * 触发仍由参数面板在 Compose 打开且块可见时发起。这样未使用的轨道不会
     * 预跑 GAME，而正在编辑的轨道也不会卡在“没有块”。
     */
    let Some(clip) = state
        .timeline
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clips
        .iter()
        .find(|clip| clip.id == clip_id)
        .cloned()
    else {
        return;
    };
    if clip.source_path.is_none() || clip.midi_note_data.is_some() {
        return;
    }
    log::info!("[syllable] queued pending analysis for clip {}", clip.id);
}

/// 真正启动后台音节分析（由“块可见 + Compose 开启”的显式请求调用）。
pub(super) fn start_clip_syllables(state: &AppState, clip_id: String) {
    let Some(app) = state.app_handle.get().cloned() else {
        return;
    };
    tauri::async_runtime::spawn_blocking(move || {
        let state: State<'_, AppState> = app.state();
        if let Err(error) = analyze_clip_syllables_inner(state, clip_id) {
            log::warn!("background syllable analysis failed: {error}");
        }
    });
}

/// 只读查询：没有已存分析时不会触发重新计算。
pub(super) fn get_clip_syllable_spans(
    state: State<'_, AppState>,
    clip_id: String,
) -> Result<ClipSyllableAnalysisPayload, String> {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let clip = tl
        .clips
        .iter()
        .find(|c| c.id == clip_id)
        .ok_or_else(|| "clip_not_found".to_string())?;
    let take = clip.active_take();
    let fp = tl.frame_period_ms();
    let digest = source_digest(clip, fp);
    let analysis = take
        .syllable_analysis
        .as_ref()
        .filter(|a| a.is_current(&digest));
    Ok(payload(clip, &take.id, analysis))
}

/// 显式重分析：使用现有电平 / 音高缓存；F0 缺失时仍只靠电平估计。
pub(super) fn analyze_clip_syllables(
    state: State<'_, AppState>,
    clip_id: String,
) -> Result<ClipSyllableAnalysisPayload, String> {
    analyze_clip_syllables_inner(state, clip_id)
}

fn analyze_clip_syllables_inner(
    state: State<'_, AppState>,
    clip_id: String,
) -> Result<ClipSyllableAnalysisPayload, String> {
    let (tl_snapshot, clip) = {
        let guard = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let clip = guard
            .clips
            .iter()
            .find(|c| c.id == clip_id)
            .cloned()
            .ok_or_else(|| "clip_not_found".to_string())?;
        (guard.clone(), clip)
    };
    let mut tl = tl_snapshot;
    let take = clip.active_take().clone();
    let fp = tl.frame_period_ms().max(0.1);
    let digest = source_digest(&clip, fp);
    let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();

    // MIDI 内容已有精确音符边界；不要用音频估计覆盖它。
    if take.midi_note_data.as_ref().is_some_and(|n| !n.is_empty()) || clip.source_path.is_none() {
        set_syllable_analysis_progress(None);
        return Ok(payload(&clip, &take.id, take.syllable_analysis.as_ref()));
    }
    /*
     * 幂等：时间线刷新可能在同一 Clip 上重复发起分析。已有当前缓存时
     * 直接返回，不重复加载 GAME，也不把 `active` 进度重新打回 2%。
     */
    if let Some(analysis) = take
        .syllable_analysis
        .as_ref()
        .filter(|a| a.is_current(&digest))
    {
        if !analysis.source_spans_sec.is_empty() {
            set_syllable_analysis_progress(None);
            return Ok(payload(&clip, &take.id, Some(analysis)));
        }
    }

    let window = hifishifter_kernel::state::clip_playback_window_sec(&clip);
    set_syllable_analysis_progress(Some(SyllableAnalysisProgress {
        active: true,
        clip_id: clip.id.clone(),
        clip_name: clip.name.clone(),
        progress: 0.02,
        stage: "decoding".into(),
        algorithm: hifishifter_kernel::game_onnx::GAME_ALGORITHM.into(),
        message: None,
    }));
    let game = hifishifter_kernel::game_onnx::resolve_model_dir().map(|dir| {
        hifishifter_kernel::game_onnx::analyze_source_file(
            std::path::Path::new(clip.source_path.as_deref().unwrap_or("")),
            &dir,
            hifishifter_kernel::game_onnx::GameInferenceConfig::default(),
        )
    });
    let (spans, algorithm, version) = match game {
        Some(Ok(events)) => {
            set_syllable_analysis_progress(Some(SyllableAnalysisProgress {
                active: true,
                clip_id: clip.id.clone(),
                clip_name: clip.name.clone(),
                progress: 0.94,
                stage: "projecting".into(),
                algorithm: hifishifter_kernel::game_onnx::GAME_ALGORITHM.into(),
                message: None,
            }));
            (
                hifishifter_kernel::game_onnx::syllable_spans_for_window(
                    &events,
                    window.0.min(window.1),
                    window.0.max(window.1),
                ),
                hifishifter_kernel::game_onnx::GAME_ALGORITHM.to_string(),
                hifishifter_kernel::game_onnx::GAME_ALGORITHM_VERSION,
            )
        }
        Some(Err(error)) => {
            log::warn!("GAME syllable analysis failed, falling back to VAD: {error}");
            fallback_vad_spans(&tl, &clip, &root, fp)
        }
        None => fallback_vad_spans(&tl, &clip, &root, fp),
    };
    let analysis = if spans.is_empty() {
        None
    } else {
        Some(TakeSyllableAnalysis {
            algorithm,
            algorithm_version: version,
            source_digest: digest,
            source_spans_sec: spans
                .into_iter()
                .map(
                    |(start_sec, end_sec)| hifishifter_kernel::pitch_analysis::SyllableSpan {
                        start_sec,
                        end_sec,
                    },
                )
                .collect(),
        })
    };
    let Some(clip_mut) = tl.clips.iter_mut().find(|c| c.id == clip_id) else {
        return Err("clip_not_found".to_string());
    };
    if let Some(index) = clip_mut
        .takes
        .iter()
        .position(|t| t.id == take.id)
        .or_else(|| (!clip_mut.takes.is_empty()).then_some(0))
    {
        let take_mut = &mut clip_mut.takes[index];
        take_mut.syllable_analysis = analysis.clone();
    }
    {
        let mut guard = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        *guard = tl;
    }
    // 音节分析会进入工程文件，必须提示保存；但不属于用户编辑，不进撤销栈。
    state.bump_timeline_version();
    state.mark_project_dirty_and_retitle();
    set_syllable_analysis_progress(None);
    Ok(payload(&clip, &take.id, analysis.as_ref()))
}

fn fallback_vad_spans(
    tl: &crate::state::TimelineState,
    clip: &crate::state::Clip,
    root: &str,
    fp: f64,
) -> (Vec<(f64, f64)>, String, u32) {
    set_syllable_analysis_progress(Some(SyllableAnalysisProgress {
        active: true,
        clip_id: clip.id.clone(),
        clip_name: clip.name.clone(),
        progress: 0.55,
        stage: "vad_fallback".into(),
        algorithm: SYLLABLE_ALGORITHM.into(),
        message: Some("GAME model unavailable; using VAD fallback".into()),
    }));
    let (level, pitch) =
        crate::pitch_clip::analyze_clip_for_syllables(tl, clip, root, fp).unwrap_or_default();
    let spans = if level.is_empty() {
        Vec::new()
    } else {
        estimate_syllable_spans(&level, &pitch, fp, SyllableConfig::default())
    };
    (
        spans
            .into_iter()
            .map(|span| (span.start_sec, span.end_sec))
            .collect(),
        SYLLABLE_ALGORITHM.to_string(),
        SYLLABLE_ALGORITHM_VERSION,
    )
}

pub(super) fn get_param_block_merges(
    state: State<'_, AppState>,
    root_track_id: String,
) -> Result<
    (
        Vec<ParamBlockMergeRecordPayload>,
        Vec<ParamBlockIntervalEditPayload>,
    ),
    String,
> {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let records = tl
        .params_by_root_track
        .get(&root_track_id)
        .map(|entry| {
            entry
                .param_block_merges
                .iter()
                .map(|record| ParamBlockMergeRecordPayload {
                    merged_group_id: record.merged_group_id.clone(),
                    member_refs: record
                        .member_refs
                        .iter()
                        .map(|r| ParamBlockMemberRefPayload {
                            clip_id: r.clip_id.clone(),
                            take_id: r.take_id.clone(),
                            syllable_id: r.syllable_id.clone(),
                            source_start_sec: r.source_start_sec,
                            source_end_sec: r.source_end_sec,
                        })
                        .collect(),
                })
                .collect()
        })
        .unwrap_or_default();
    let interval_edits = tl
        .params_by_root_track
        .get(&root_track_id)
        .map(|entry| {
            entry
                .param_block_interval_edits
                .iter()
                .map(|edit| ParamBlockIntervalEditPayload {
                    edit_id: edit.edit_id.clone(),
                    member_refs: edit
                        .member_refs
                        .iter()
                        .map(|r| ParamBlockMemberRefPayload {
                            clip_id: r.clip_id.clone(),
                            take_id: r.take_id.clone(),
                            syllable_id: r.syllable_id.clone(),
                            source_start_sec: r.source_start_sec,
                            source_end_sec: r.source_end_sec,
                        })
                        .collect(),
                    start_sec: edit.start_sec,
                    end_sec: edit.end_sec,
                    member_start_sec: edit.member_start_sec,
                    member_end_sec: edit.member_end_sec,
                    block_ids: edit.block_ids.clone(),
                    explicit: edit.explicit,
                    deleted: edit.deleted,
                    manual: edit.manual,
                    value: edit.value,
                    transition_sec: edit.transition_sec,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok((records, interval_edits))
}

pub(super) fn set_param_block_merges(
    state: State<'_, AppState>,
    root_track_id: String,
    records: Vec<ParamBlockMergeRecordPayload>,
    interval_edits: Vec<ParamBlockIntervalEditPayload>,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let before = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if checkpoint.unwrap_or(true) {
        state.checkpoint_timeline(&before, crate::state::HistoryOp::ParamBlockMerge);
    }
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.ensure_params_for_root(&root_track_id);
        if let Some(entry) = tl.params_by_root_track.get_mut(&root_track_id) {
            entry.param_block_merges = records
                .into_iter()
                .map(|r| crate::state::ParamBlockMergeRecord {
                    merged_group_id: r.merged_group_id,
                    member_refs: r
                        .member_refs
                        .into_iter()
                        .map(|m| crate::state::ParamBlockMemberRef {
                            clip_id: m.clip_id,
                            take_id: m.take_id,
                            syllable_id: m.syllable_id,
                            source_start_sec: m.source_start_sec,
                            source_end_sec: m.source_end_sec,
                        })
                        .collect(),
                })
                .filter(|r| r.member_refs.len() >= 2)
                .collect();
            entry.param_block_interval_edits = interval_edits
                .into_iter()
                .map(|edit| crate::state::ParamBlockIntervalEdit {
                    edit_id: edit.edit_id,
                    member_refs: edit
                        .member_refs
                        .into_iter()
                        .map(|m| crate::state::ParamBlockMemberRef {
                            clip_id: m.clip_id,
                            take_id: m.take_id,
                            syllable_id: m.syllable_id,
                            source_start_sec: m.source_start_sec,
                            source_end_sec: m.source_end_sec,
                        })
                        .collect(),
                    start_sec: edit.start_sec,
                    end_sec: edit.end_sec,
                    member_start_sec: edit.member_start_sec,
                    member_end_sec: edit.member_end_sec,
                    block_ids: edit.block_ids,
                    explicit: edit.explicit,
                    deleted: edit.deleted,
                    manual: edit.manual,
                    value: edit.value,
                    transition_sec: edit.transition_sec,
                })
                .filter(|edit| {
                    let width = edit.end_sec - edit.start_sec;
                    !edit.member_refs.is_empty()
                        && edit.edit_id.len() <= 512
                        && width.is_finite()
                        && width >= 0.02
                        && edit.start_sec.is_finite()
                        && edit.end_sec.is_finite()
                })
                .collect();
        }
        let clips = tl.clips.clone();
        if let Some(entry) = tl.params_by_root_track.get_mut(&root_track_id) {
            entry.prune_param_block_merges(&clips);
        }
        // 合并记录虽不直接参与混音，但属于工程状态；引擎快照必须与磁盘
        // 工程同源，避免下一次快照重建用旧记录覆盖用户的合并意图。
        state.audio_engine.update_timeline(tl.clone());
    }
    state.mark_project_dirty_and_retitle();
    serde_json::json!({ "ok": true })
}
