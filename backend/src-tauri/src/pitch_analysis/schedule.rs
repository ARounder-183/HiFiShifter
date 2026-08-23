// pitch_analysis::schedule — 缓存组装 + 调度
// assemble_pitch_orig_from_cache: 从 per-clip 缓存同步拼装整体音高线。
// maybe_schedule_pitch_orig: 对外公开的调度入口。

use crate::state::AppState;
use tauri::Emitter;

use super::analysis::build_pitch_job;
use super::{build_root_pitch_key, PitchOrigUpdatedEvent};

pub(crate) fn assemble_pitch_orig_from_cache(
    tl: &crate::state::TimelineState,
    root_track_id: &str,
) -> Option<(Vec<f32>, bool, bool)> {
    let fp = tl.frame_period_ms();
    let target_frames = tl.target_param_frames(fp);
    let bpm = if tl.bpm.is_finite() && tl.bpm > 0.0 {
        tl.bpm
    } else {
        120.0
    };
    let _bs = 60.0 / bpm;

    // 收集属于 root track 的所有 clip（同时包含有音频源的 clip 和带 midi_note_data 的 MIDI clip）
    // 按轨道由上至下的顺序排列：上方轨道的 clip 优先于下方轨道。
    // 先收集整个轨道组内的所有轨道（任何深度），按 tl.tracks 顺序（即 UI 从上到下）排列。
    let mut group_track_ids: Vec<&str> = Vec::new();
    for track in &tl.tracks {
        if tl.resolve_root_track_id(&track.id).as_deref() == Some(root_track_id) {
            group_track_ids.push(track.id.as_str());
        }
    }

    // 按 tl.tracks 中的顺序排列轨道（即 UI 中从上到下的顺序）
    let mut ordered_track_ids: Vec<&str> = Vec::new();
    for track in &tl.tracks {
        if group_track_ids.contains(&track.id.as_str()) {
            ordered_track_ids.push(track.id.as_str());
        }
    }

    // 收集所有 clip，按轨道分组
    let mut track_clips: std::collections::HashMap<&str, Vec<&crate::state::Clip>> =
        std::collections::HashMap::new();
    let mut has_pitch_adjustment = false;
    for clip in &tl.clips {
        let clip_root = tl.resolve_root_track_id(&clip.track_id);
        if clip_root.as_deref() != Some(root_track_id) {
            continue;
        }
        if clip.muted {
            continue;
        }
        if clip.source_path.is_none() && clip.midi_note_data.is_none() {
            continue;
        }
        if clip.midi_note_data.is_some() {
            has_pitch_adjustment = true;
        }
        track_clips
            .entry(clip.track_id.as_str())
            .or_default()
            .push(clip);
    }

    let all_empty = track_clips.is_empty();
    if all_empty {
        // 没有 clip，直接返回全零曲线（视为全部命中）
        return Some((vec![0.0f32; target_frames], true, false));
    }

    let mut out = vec![0.0f32; target_frames];
    let mut all_cache_hit = true;

    // 按轨道从下到上遍历（下方轨道先写入，上方轨道后写入以覆盖）
    // 同一轨道内按 z-order 从低到高（tl.clips 顺序）写入
    let mut processed_clips = Vec::new();
    for track_id in ordered_track_ids.iter().rev() {
        if let Some(clips) = track_clips.get(track_id) {
            for clip in clips {
                processed_clips.push(*clip);
            }
        }
    }

    for clip in &processed_clips {
        // 计算 clip 在 timeline 中的起始帧（MIDI clip 和音频 clip 共用）
        let clip_start_sec = clip.start_sec.max(0.0);
        let clip_start_frame = ((clip_start_sec * 1000.0) / fp).round().max(0.0) as usize;
        let clip_len_sec = clip.length_sec.max(0.0);
        let clip_len_frames = ((clip_len_sec * 1000.0) / fp).round().max(0.0) as usize;

        // ── MIDI clip 路径：直接从 midi_note_data 生成音高帧 ──
        if let Some(ref notes) = clip.midi_note_data {
            let pr = clip.playback_rate as f64;
            let pr_valid = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
            let src_start = clip.source_start_sec.max(0.0);
            let src_end = if clip.source_end_sec > 0.0 {
                clip.source_end_sec
            } else {
                clip_len_sec
            };
            let src_total_len = src_end - src_start;

            // Loop（循环源）：单个循环周期占用的帧数。
            // 启用时音符内容按该周期重复铺满整个 clip 可见区间
            //（音高参考块被延伸后循环 ori pitch）。
            let loop_cycle_frames: Option<usize> = if clip.loop_enabled && src_total_len > 1e-9 {
                Some(((src_total_len / pr_valid * 1000.0) / fp).round().max(1.0) as usize)
            } else {
                None
            };
            let clip_visible_frames =
                clip_len_frames.min(target_frames.saturating_sub(clip_start_frame));

            // 收集已写入音符的帧范围（用于后续填补空隙）和音高值
            let mut note_ranges: Vec<(usize, usize, f32)> = Vec::new();

            for note in notes {
                if note.end_sec <= src_start || note.start_sec >= src_end {
                    continue; // 音符在可见范围之外
                }
                let rel_start = (note.start_sec - src_start).max(0.0);
                let rel_end = (note.end_sec - src_start).min(src_end - src_start);
                if rel_end <= rel_start {
                    continue;
                }
                // 倒放时镜像音符在 source range 内的位置
                let (effective_rel_start, effective_rel_end) = if clip.reversed {
                    (
                        (src_total_len - rel_end).max(0.0),
                        (src_total_len - rel_start).min(src_total_len),
                    )
                } else {
                    (rel_start, rel_end)
                };
                if effective_rel_end <= effective_rel_start {
                    continue;
                }
                // 通过 playback_rate 缩放以支持拉伸/压缩
                let note_start_frame =
                    ((effective_rel_start / pr_valid * 1000.0) / fp).round() as usize;
                let note_end_frame =
                    ((effective_rel_end / pr_valid * 1000.0) / fp).round() as usize;
                let note_value = note.note as f32;
                match loop_cycle_frames {
                    Some(cycle_frames) => {
                        // Loop：按周期重复写入，直到铺满 clip 可见区间。
                        let mut cycle_offset = 0usize;
                        while cycle_offset < clip_visible_frames {
                            let write_start =
                                clip_start_frame + cycle_offset + note_start_frame;
                            let write_end = (clip_start_frame + cycle_offset + note_end_frame)
                                .min(clip_start_frame + clip_visible_frames);
                            if write_start >= write_end {
                                break;
                            }
                            for frame in write_start..write_end {
                                let current = out[frame];
                                if note_value > current || current <= 0.0 {
                                    out[frame] = note_value;
                                }
                            }
                            note_ranges.push((write_start, write_end, note_value));
                            cycle_offset += cycle_frames;
                        }
                    }
                    None => {
                        let write_start = clip_start_frame.saturating_add(note_start_frame);
                        let write_end = clip_start_frame
                            .saturating_add(note_end_frame)
                            .min(target_frames);
                        if write_start < write_end {
                            for frame in write_start..write_end {
                                let current = out[frame];
                                if note_value > current || current <= 0.0 {
                                    out[frame] = note_value;
                                }
                            }
                            note_ranges.push((write_start, write_end, note_value));
                        }
                    }
                }
            }

            // 填补音符之间的空隙（仅在 clip 区间内）
            // 使用已写入的音符帧范围精确填充，覆盖子轨音频块的原始音高值
            if clip.midi_fill_gaps && !note_ranges.is_empty() {
                note_ranges.sort_by_key(|&(s, _, _)| s);
                // 合并时间上重叠的音符范围（保留最高音高，与写入逻辑一致）
                let mut merged: Vec<(usize, usize, f32)> = Vec::new();
                for (s, e, v) in note_ranges {
                    if let Some(last) = merged.last_mut() {
                        if s < last.1 {
                            last.1 = last.1.max(e);
                            last.2 = last.2.max(v);
                            continue;
                        }
                    }
                    merged.push((s, e, v));
                }
                // 在相邻的音符范围之间填充空隙
                for w in merged.windows(2) {
                    let (_, end_prev, pitch_prev) = w[0];
                    let (start_next, _, _) = w[1];
                    if end_prev < start_next {
                        for frame in end_prev..start_next {
                            out[frame] = pitch_prev;
                        }
                    }
                }
            }

            continue; // 跳过音频缓存路径
        }

        // ── 音频 clip 路径：从缓存获取 FCPE 分析结果 ──
        let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
        let cached =
            match crate::pitch_clip::get_or_compute_clip_pitch_midi_global(tl, clip, &root, fp) {
                Some(c) => c,
                None => {
                    all_cache_hit = false;
                    continue;
                }
            };

        // 判断是否为全量源音频缓存（playback_rate == 1）
        let pr = clip.playback_rate as f64;
        let is_full_source = pr.is_finite() && pr > 0.0 && (pr - 1.0).abs() <= 1e-6;

        // is_full_source (rate==1)：从 source_start_sec 处偏移截取，直接写入 out
        // !is_full_source (rate!=1)：从全量曲线中截取 source range 区间并 resample 到 clip timeline 长度再写入 out

        if is_full_source {
            let src_offset = ((clip.source_start_sec.max(0.0) * 1000.0) / fp)
                .round()
                .max(0.0) as usize;

            // 预计算切片安全边界，消除内部循环的所有越界检查、if 判断和解包
            let write_len = clip_len_frames.min(target_frames.saturating_sub(clip_start_frame));

            if clip.loop_enabled {
                // Loop（循环源）：对整个媒体文件做模运算回绕
                // idx(i) = floor_mod(anchor ± i, N)，N = 完整媒体时长对应帧数。
                let media_total = crate::state::clip_source_media_duration_sec(clip);
                // 媒体时长未知时回退为"整条缓存曲线即回绕周期"（缓存本身就是
                // 全量源音频曲线）。此前回退值是 1 —— 所有帧都映射到第 0 帧，
                // 曲线塌缩成恒定音高；缓存为空时更会直接越界 panic。
                let n_frames = media_total
                    .map(|total| (((total * 1000.0) / fp).round() as usize).min(cached.midi.len()))
                    .filter(|n| *n > 0)
                    .unwrap_or(cached.midi.len());
                // 正放锚点用**原始** source_start_sec（可为负，floor_mod 环绕），
                // 与音频回绕（mix/snapshot/mixdown）保持一致。
                let anchor_f = ((clip.source_start_sec * 1000.0) / fp).round() as i64;
                let end_eff = clip
                    .source_end_sec
                    .min(media_total.unwrap_or(f64::INFINITY))
                    .max(0.0);
                let anchor_r = ((end_eff * 1000.0) / fp).round() as i64;
                if write_len > 0 && !cached.midi.is_empty() {
                    let n_frames = n_frames.max(1);
                    let dst_slice = &mut out[clip_start_frame..clip_start_frame + write_len];
                    for (i, dst) in dst_slice.iter_mut().enumerate() {
                        let idx_i = if clip.reversed {
                            anchor_r - 1 - i as i64
                        } else {
                            anchor_f + i as i64
                        };
                        let idx = idx_i.rem_euclid(n_frames as i64) as usize;
                        let pitch = cached.midi[idx];
                        *dst = if pitch.is_finite() && pitch > 0.0 {
                            pitch
                        } else {
                            0.0
                        };
                    }
                }
            } else {
                let read_len =
                    write_len.min(cached.midi.len().saturating_sub(src_offset));
                if read_len > 0 {
                    let dst_slice = &mut out[clip_start_frame..clip_start_frame + read_len];
                    let src_slice = &cached.midi[src_offset..src_offset + read_len];

                    for (dst, &pitch) in dst_slice.iter_mut().zip(src_slice.iter()) {
                        *dst = if pitch.is_finite() && pitch > 0.0 {
                            pitch
                        } else {
                            0.0
                        };
                    }
                }
            }
        } else {
            let pr_valid = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
            let resampled = crate::pitch_clip::trim_and_resample_midi(
                &cached.midi,
                fp,
                clip.source_start_sec,
                clip.source_end_sec,
                pr_valid,
                clip_len_sec,
                clip.loop_enabled,
                crate::state::clip_source_media_duration_sec(clip),
                clip.reversed && clip.loop_enabled,
            );

            // 预计算边界并进行迭代覆盖
            let write_len = clip_len_frames.min(target_frames.saturating_sub(clip_start_frame));
            let read_len = write_len.min(resampled.len());

            if read_len > 0 {
                let dst_slice = &mut out[clip_start_frame..clip_start_frame + read_len];
                let src_slice = &resampled[..read_len];

                for (dst, &pitch) in dst_slice.iter_mut().zip(src_slice.iter()) {
                    *dst = if pitch.is_finite() && pitch > 0.0 {
                        pitch
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    Some((out, all_cache_hit, has_pitch_adjustment))
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

        // 直接 per-clip 缓存同步组装整体音高线（不再重新分析音频）
        let (curve, all_cache_hit, has_pitch_adjustment) =
            match assemble_pitch_orig_from_cache(&tl, root_track_id) {
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
                    entry.has_pitch_adjustment_active = has_pitch_adjustment;

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
                            // 单次 memcpy 直接拷入，消灭 memset 和 越界检查
                            entry
                                .pitch_edit
                                .extend_from_slice(&entry.pitch_orig[old_len..]);
                        }
                        for i in 0..len {
                            if offsets[i].abs() > 1e-6 && entry.pitch_orig[i].abs() > 1e-6 {
                                entry.pitch_edit[i] = entry.pitch_orig[i] + offsets[i];
                            }
                        }
                        entry.pitch_edit_user_modified = true;
                    } else if !entry.pitch_edit_user_modified {
                        // 复用已有内存，避免昂贵的重新分配
                        entry.pitch_edit.clone_from(&entry.pitch_orig);
                    }
                    should_emit = true;
                    emit_root_track_id = job.root_track_id.clone();
                } else {
                    // 部分命中：仅当曲线内容确实发生变化时才更新并通知前端。
                    // 否则跳过 emit，防止"fetch -> emit -> fetch"无限循环。
                    if entry.pitch_orig != curve {
                        entry.pitch_orig = curve;
                        entry.pitch_orig_key = None;
                        entry.has_pitch_adjustment_active = has_pitch_adjustment;
                        if !entry.pitch_edit_user_modified {
                            // 复用已有内存，避免昂贵的重新分配
                            entry.pitch_edit.clone_from(&entry.pitch_orig);
                        }
                        should_emit = true;
                        emit_root_track_id = job.root_track_id.clone();
                    }
                    // else: 曲线未变化，跳过通知，避免死循环
                }
            }
        }
    }
    // lock 释放后再 emit，避免持锁时发事件
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
