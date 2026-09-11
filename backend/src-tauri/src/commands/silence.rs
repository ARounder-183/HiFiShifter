// 静音检测（Silence Detection）命令实现。
//
// - `analyze_clip_silence`：干跑分析，供设置对话框实时预览，不修改状态；
// - `remove_clip_silence`：执行切除——分析（不持锁）→ 一次撤销检查点内
//   完成"切分 → 删除 → 闭合 → 切边淡化"（`TimelineState::apply_silence_removal`），
//   引擎更新 + 受影响根轨道音高重分析调度。
//
// 分析在**不持有时间线锁**的情况下进行（解码是慢 IO）；命令入口先在短锁内
// 抓取所需 Clip/Take 元数据快照，再逐 Take 解码分析。

use std::collections::HashSet;

use tauri::State;

use crate::models::{
    ClipSilenceReportPayload, RemoveSilenceResultPayload, SilenceAnalyzeResultPayload,
    SilenceDetectOptionsPayload, SilenceRegionPayload, TimelineStatePayload,
};
use crate::state::{AppState, SilenceRemovalAction, TimelineState};

/// 检测选项载荷 → 分析选项。
fn detect_options(p: &SilenceDetectOptionsPayload) -> crate::silence_detect::SilenceDetectOptions {
    crate::silence_detect::SilenceDetectOptions {
        use_peak: p.method == "peak",
        threshold_db: p.threshold_db,
        adaptive: p.adaptive,
        min_silence_ms: p.min_silence_ms.max(0.0),
        min_sound_ms: p.min_sound_ms.max(0.0),
        padding_ms: p.padding_ms.max(0.0),
    }
}

/// 待分析 Clip 的元数据快照（避免在解码期间持锁）。
struct ClipMeta {
    clip_id: String,
    clip_start_sec: f64,
    clip_length_sec: f64,
    /// (source_path, source_start, source_end, rate, reversed, loop)
    takes: Vec<(String, f64, f64, f64, bool, bool)>,
}

fn snapshot_clips(tl: &TimelineState, clip_ids: &[String], all_takes: bool) -> Vec<ClipMeta> {
    let mut metas = Vec::with_capacity(clip_ids.len());
    for clip_id in clip_ids {
        let Some(clip) = tl.clips.iter().find(|c| c.id == *clip_id) else {
            continue;
        };
        let clip_rate = if clip.clip_playback_rate.is_finite() && clip.clip_playback_rate > 1e-6 {
            clip.clip_playback_rate as f64
        } else {
            1.0
        };
        let takes: Vec<&crate::state::ClipTake> = if all_takes {
            clip.takes.iter().collect()
        } else {
            vec![clip.active_take()]
        };
        let mut take_rows = Vec::new();
        for take in takes {
            let Some(path) = take.source_path.as_deref() else {
                continue;
            };
            if path.trim().is_empty() {
                continue;
            }
            let take_rate = if take.playback_rate.is_finite() && take.playback_rate > 1e-6 {
                take.playback_rate as f64
            } else {
                1.0
            };
            take_rows.push((
                path.to_string(),
                take.source_start_sec,
                take.source_end_sec,
                clip_rate * take_rate,
                take.reversed,
                take.loop_enabled,
            ));
        }
        metas.push(ClipMeta {
            clip_id: clip.id.clone(),
            clip_start_sec: clip.start_sec,
            clip_length_sec: clip.length_sec,
            takes: take_rows,
        });
    }
    metas
}

/// 合并两个升序区间列表（并集，交叠/贴合区间合并）。
fn merge_region_lists(a: Vec<(f64, f64)>, b: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut all: Vec<(f64, f64)> = a.into_iter().chain(b).collect();
    all.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(all.len());
    for (s, e) in all {
        match merged.last_mut() {
            Some(last) if s <= last.1 + 1e-4 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
}

/// 对一批 Clip 做静音分析（不持锁、不修改状态）。
fn analyze_clips(
    state: &State<'_, AppState>,
    clip_ids: &[String],
    options: &SilenceDetectOptionsPayload,
) -> Vec<ClipSilenceReportPayload> {
    let opts = detect_options(options);
    let metas = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        snapshot_clips(&tl, clip_ids, options.sync_all_takes)
    };
    let mut reports = Vec::with_capacity(clip_ids.len());
    let mut reported = HashSet::new();
    for meta in &metas {
        reported.insert(meta.clip_id.clone());
        if meta.takes.is_empty() {
            reports.push(ClipSilenceReportPayload {
                clip_id: meta.clip_id.clone(),
                ok: false,
                message: Some("no_audio_source".to_string()),
                fully_silent: false,
                total_silent_sec: 0.0,
                regions: Vec::new(),
            });
            continue;
        }
        let mut union: Vec<(f64, f64)> = Vec::new();
        let mut error: Option<String> = None;
        for (path, ss, se, rate, reversed, loop_enabled) in &meta.takes {
            match crate::silence_detect::analyze_take_silence(
                path,
                *ss,
                *se,
                *rate,
                *reversed,
                *loop_enabled,
                meta.clip_start_sec,
                meta.clip_length_sec,
                &opts,
            ) {
                Ok(regions) => union = merge_region_lists(union, regions),
                Err(msg) => {
                    error = Some(msg);
                    break;
                }
            }
        }
        if let Some(msg) = error {
            reports.push(ClipSilenceReportPayload {
                clip_id: meta.clip_id.clone(),
                ok: false,
                message: Some(msg),
                fully_silent: false,
                total_silent_sec: 0.0,
                regions: Vec::new(),
            });
            continue;
        }
        let total = union.iter().map(|(s, e)| (e - s).max(0.0)).sum();
        // padding 把整段静音的区间两端各内缩 pad 秒，判定"全静音"必须
        // 把 pad 计入容差，否则带 padding 时 fully_silent 永远为 false。
        let pad = (opts.padding_ms / 1000.0).max(0.0);
        reports.push(ClipSilenceReportPayload {
            clip_id: meta.clip_id.clone(),
            ok: true,
            message: None,
            fully_silent: union
                .first()
                .is_some_and(|(s, _)| *s <= meta.clip_start_sec + pad + 1e-4)
                && union.last().is_some_and(|(_, e)| {
                    *e >= meta.clip_start_sec + meta.clip_length_sec - pad - 1e-4
                }),
            total_silent_sec: total,
            regions: union
                .into_iter()
                .map(|(s, e)| SilenceRegionPayload {
                    start_sec: s,
                    end_sec: e,
                })
                .collect(),
        });
    }
    // 找不到的 clip id（并发编辑中被删除等）：补报，保证 reports 与
    // clip_ids 一一对应，前端不至于静默少一条。
    for clip_id in clip_ids {
        if reported.insert(clip_id.clone()) {
            reports.push(ClipSilenceReportPayload {
                clip_id: clip_id.clone(),
                ok: false,
                message: Some("source_missing".to_string()),
                fully_silent: false,
                total_silent_sec: 0.0,
                regions: Vec::new(),
            });
        }
    }
    reports
}

/// 静音检测（干跑）：返回每个 Clip 的静音区间报告，不修改任何状态。
pub(super) fn analyze_clip_silence(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
    options: SilenceDetectOptionsPayload,
) -> SilenceAnalyzeResultPayload {
    SilenceAnalyzeResultPayload {
        ok: true,
        reports: analyze_clips(&state, &clip_ids, &options),
    }
}

/// 静音切除：分析 + 一次撤销检查点内完成"切分 → 删除 → 闭合 → 切边淡化"。
pub(super) fn remove_clip_silence(
    state: State<'_, AppState>,
    clip_ids: Vec<String>,
    options: SilenceDetectOptionsPayload,
) -> RemoveSilenceResultPayload {
    // 1) 先分析（不持锁）。
    let reports = analyze_clips(&state, &clip_ids, &options);

    let action = match options.action.as_str() {
        "keep" => SilenceRemovalAction::Keep,
        "split" => SilenceRemovalAction::Split,
        _ => SilenceRemovalAction::Close,
    };
    let link = state.ui_settings_snapshot().lock_param_lines;

    // 2) 组装待变换计划（只包含分析成功且有静音区的 Clip）。
    let per_clip: Vec<(String, Vec<(f64, f64)>)> = reports
        .iter()
        .filter(|r| r.ok && !r.regions.is_empty())
        .map(|r| {
            (
                r.clip_id.clone(),
                r.regions.iter().map(|g| (g.start_sec, g.end_sec)).collect(),
            )
        })
        .collect();

    // 3) 一次检查点内完成全部几何变换。
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    state.checkpoint_timeline(&tl, crate::state::HistoryOp::DeleteSilence);
    let outcome = tl.apply_silence_removal(
        &per_clip,
        action,
        options.delete_silent_clips,
        options.cut_fade_ms.max(0.0) / 1000.0,
        link,
    );
    state.audio_engine.update_timeline(tl.clone());
    let mut timeline_payload: TimelineStatePayload = tl.to_payload();
    timeline_payload.project = Some(state.project_meta_payload());
    drop(tl);

    // 4) 受影响根轨道调度音高重分析。
    for root_id in &outcome.touched_root_track_ids {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, root_id);
    }

    RemoveSilenceResultPayload {
        ok: true,
        timeline: timeline_payload,
        reports,
        kept_clip_ids: outcome.kept_clip_ids,
        removed_clip_ids: outcome.removed_clip_ids,
    }
}
