// pitch_analysis::schedule — 缓存组装 + 调度
// assemble_pitch_orig_from_cache: 从 per-clip 缓存同步拼装整体音高线。
// maybe_schedule_pitch_orig: 对外公开的调度入口。

use crate::state::AppState;
use tauri::Emitter;

use super::analysis::build_pitch_job;
use super::{build_root_pitch_key, resample_curve_linear, PitchOrigUpdatedEvent};

fn assemble_pitch_orig_from_cache(
    tl: &crate::state::TimelineState,
    root_track_id: &str,
) -> Option<(Vec<f32>, bool)> {
    let fp = tl.frame_period_ms();
    let target_frames = tl.target_param_frames(fp);
    let bpm = if tl.bpm.is_finite() && tl.bpm > 0.0 {
        tl.bpm
    } else {
        120.0
    };
    let bs = 60.0 / bpm;

    // 收集属于?root track 的所?clip（保?tl.clips 原始顺序 = z-order?
    let clips: Vec<&crate::state::Clip> = tl
        .clips
        .iter()
        .filter(|c| {
            tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id)
                && !c.muted
                && c.source_path.is_some()
        })
        .collect();

    if clips.is_empty() {
        // 没有 clip，直接返回全零曲线（视为全部命中?
        return Some((vec![0.0f32; target_frames], true));
    }

    let mut out = vec![0.0f32; target_frames];
    let mut all_cache_hit = true;

    // ?z-order 从低到高（tl.clips 顺序）写入，后面?clip 覆盖前面?
    for clip in &clips {
        let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
        let cached =
            match crate::pitch_clip::get_or_compute_clip_pitch_midi_global(tl, clip, &root, fp) {
                Some(c) => c,
                None => {
                    all_cache_hit = false;
                    continue;
                }
            };

        // 计算 clip ?timeline 中的起始?
        let clip_start_sec = clip.start_sec.max(0.0);
        let clip_start_frame = ((clip_start_sec * 1000.0) / fp).round().max(0.0) as usize;

        // 判断是否为全量源音频缓存（playback_rate == 1?
        let pr = clip.playback_rate as f64;
        let is_full_source = pr.is_finite() && pr > 0.0 && (pr - 1.0).abs() <= 1e-6;

        // 缓存中始终是全量源音频的 MIDI 曲线?
        // is_full_source (rate==1)：从 source_start_sec 处偏移截取，直接写入 out
        // !is_full_source (rate!=1)：从全量曲线中截?source range 区间 ?resample ?clip timeline 长度 ?写入 out
        let clip_len_sec = clip.length_sec.max(0.0);
        let clip_len_frames = ((clip_len_sec * 1000.0) / fp).round().max(0.0) as usize;

        if is_full_source {
            // rate==1：从 source_start_sec 处偏移截取，直接写入
            let src_offset = {
                let source_start_sec = clip.source_start_sec.max(0.0);
                ((source_start_sec * 1000.0) / fp).round().max(0.0) as usize
            };
            for local_i in 0..clip_len_frames {
                let src_i = src_offset + local_i;
                let global_i = clip_start_frame + local_i;
                if global_i >= target_frames {
                    break;
                }
                let pitch = cached.midi.get(src_i).copied().unwrap_or(0.0);
                if pitch.is_finite() && pitch > 0.0 {
                    out[global_i] = pitch;
                } else {
                    out[global_i] = 0.0;
                }
            }
        } else {
            // rate!=1：从全量曲线中截?source range 区间 ?resample ?clip timeline 长度
            let pr_valid = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
            let resampled = crate::pitch_clip::trim_and_resample_midi(
                &cached.midi,
                fp,
                clip.source_start_sec,
                clip.source_end_sec,
                pr_valid,
                clip_len_sec,
            );
            for local_i in 0..clip_len_frames {
                let global_i = clip_start_frame + local_i;
                if global_i >= target_frames {
                    break;
                }
                let pitch = resampled.get(local_i).copied().unwrap_or(0.0);
                if pitch.is_finite() && pitch > 0.0 {
                    out[global_i] = pitch;
                } else {
                    out[global_i] = 0.0;
                }
            }
        }
    }

    Some((out, all_cache_hit))
}

/// Returns whether pitch analysis is currently pending (scheduled or already inflight).
pub fn maybe_schedule_pitch_orig(state: &AppState, root_track_id: &str) -> bool {
    // 单次 lock 保证 build_pitch_job ?assemble ?写入 的原子性，
    // 避免多次 lock 之间 state.timeline 被前端命令修改导?key 不一致?
    let mut should_emit = false;
    let mut emit_root_track_id = String::new();
    {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        // 检查是否需要更新（compose_enabled、algo 等前置条件）
        let job = match build_pitch_job(&tl, root_track_id) {
            Some(j) => j,
            None => return false,
        };

        // 直接?per-clip 缓存同步组装整体音高线（不再重新分析音频?
        let (curve, all_cache_hit) = match assemble_pitch_orig_from_cache(&tl, root_track_id) {
            Some(v) => v,
            None => {
                // assemble_pitch_orig_from_cache 目前永远返回 Some，此分支保留作为安全兜底
                return true;
            }
        };

        // 将组装好的曲线写?state
        tl.ensure_params_for_root(&job.root_track_id);
        let current_key = build_root_pitch_key(&tl, &job.root_track_id);
        if current_key == job.key {
            if let Some(entry) = tl.params_by_root_track.get_mut(&job.root_track_id) {
                if all_cache_hit {
                    // 全部命中：写入曲线、标记完成、通知前端
                    entry.pitch_orig = curve;
                    entry.pitch_orig_key = Some(job.key.clone());

                    // 应用 Reaper 导入的待定音高偏移
                    if let Some(offsets) = entry.pending_pitch_offset.take() {
                        let len = entry.pitch_orig.len().min(offsets.len());
                        // 若已有用户编辑的 pitch_edit，在其基础上叠加偏移，避免重置其他片段的音高线
                        if entry.pitch_edit.is_empty() || !entry.pitch_edit_user_modified {
                            entry.pitch_edit = entry.pitch_orig.clone();
                        }
                        // 确保 pitch_edit 长度与 pitch_orig 一致
                        if entry.pitch_edit.len() < entry.pitch_orig.len() {
                            let old_len = entry.pitch_edit.len();
                            entry.pitch_edit.resize(entry.pitch_orig.len(), 0.0);
                            // 新增区间以 pitch_orig 作为 baseline，避免 baseline 变为 0
                            for i in old_len..entry.pitch_orig.len() {
                                entry.pitch_edit[i] = entry.pitch_orig[i];
                            }
                        }
                        for i in 0..len {
                            if offsets[i].abs() > 1e-6 && entry.pitch_orig[i].abs() > 1e-6 {
                                entry.pitch_edit[i] = entry.pitch_orig[i] + offsets[i];
                            }
                        }
                        entry.pitch_edit_user_modified = true;
                    } else if !entry.pitch_edit_user_modified {
                        entry.pitch_edit = entry.pitch_orig.clone();
                    }
                    should_emit = true;
                    emit_root_track_id = job.root_track_id.clone();
                } else {
                    // 部分命中：仅当曲线内容确实发生变化时才更新并通知前端。
                    // 否则跳过 emit，防止"fetch -> emit -> fetch"无限循环。
                    if entry.pitch_orig != curve {
                        entry.pitch_orig = curve;
                        entry.pitch_orig_key = None;
                        if !entry.pitch_edit_user_modified {
                            entry.pitch_edit = entry.pitch_orig.clone();
                        }
                        should_emit = true;
                        emit_root_track_id = job.root_track_id.clone();
                    }
                    // else: 曲线未变化，跳过通知，避免死循环
                }
            }
        }
    }
    // lock 释放后再 emit，避免持锁时发事?
    if should_emit {
        if let Some(app) = state.app_handle.get() {
            let _ = app.emit(
                "pitch_orig_updated",
                PitchOrigUpdatedEvent {
                    root_track_id: emit_root_track_id,
                },
            );
        }
    }

    false // 同步完成，不?pending
}
