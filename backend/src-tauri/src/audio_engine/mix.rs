use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;

use super::types::EngineClip;
use super::types::EngineSnapshot;
use super::types::TrackMeterValue;
use super::util::clamp11;

const SNAPSHOT_XFADE_FRAMES: usize = 256;

#[derive(Default)]
pub(crate) struct SnapshotTransitionState {
    current_snapshot: Option<Arc<EngineSnapshot>>,
    fade_from_snapshot: Option<Arc<EngineSnapshot>>,
    fade_remaining_frames: usize,
}

#[derive(Default)]
pub(crate) struct TrackMeterScratch {
    per_track_mix: std::collections::HashMap<String, Vec<f32>>,
    active_track_ids: Vec<String>,
}

fn sample_automation_curve(
    curve: Option<&Vec<f32>>,
    abs_frame: u64,
    sample_rate: u32,
    frame_period_ms: f64,
    default_value: f32,
) -> f32 {
    let Some(curve) = curve else {
        return default_value;
    };
    if curve.is_empty() {
        return default_value;
    }

    let fp = frame_period_ms.max(0.1);
    let abs_sec = abs_frame as f64 / sample_rate.max(1) as f64;
    let idx_f = (abs_sec * 1000.0) / fp;
    if !idx_f.is_finite() {
        return default_value;
    }
    let i0 = idx_f.floor().max(0.0) as usize;
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 采样 clip 在 local 帧处的原始 PCM（不含 gain/fade）。
/// 返回 None 表示该帧应静音（越界、leading silence 等）。
#[inline]
fn sample_clip_pcm(clip: &EngineClip, local: u64, local_adj: f64) -> Option<(f32, f32)> {
    // 最高优先级：预渲染 PCM（有 pitch edit 时由后台线程渲染）
    if let Some(ref rendered) = clip.rendered_pcm {
        let idx = (local as usize) * 2;
        if idx + 1 < rendered.len() {
            let mut left = rendered[idx];
            let mut right = rendered[idx + 1];
            if let Some(ref breath_noise) = clip.breath_noise_pcm {
                if idx + 1 < breath_noise.len() {
                    let gain = sample_automation_curve(
                        clip.breath_curve.as_deref(),
                        clip.start_frame.saturating_add(local),
                        clip.src.sample_rate,
                        clip.breath_curve_frame_period_ms,
                        1.0,
                    );
                    left += breath_noise[idx] * gain;
                    right += breath_noise[idx + 1] * gain;
                }
            }
            // 应用 volume 曲线（不触发重渲染，实时乘到最终输出）
            let vol = sample_automation_curve(
                clip.volume_curve.as_deref(),
                clip.start_frame.saturating_add(local),
                clip.src.sample_rate,
                clip.volume_curve_frame_period_ms,
                1.0,
            );
            return Some((left * vol, right * vol));
        }
        // rendered_pcm 存在但越界时返回静音
        return None;
    }

    // 若该 clip 需要合成（pitch edit）但尚未渲染完成，静音等待
    if clip.needs_synthesis {
        return None;
    }

    // 无需合成：直接回退到源 PCM（支持 playback_rate 采样）
    let src_frame_f = local_adj * clip.playback_rate;
    let src_frame = src_frame_f.round() as u64;
    let range = clip.src_end_frame.saturating_sub(clip.src_start_frame);
    if range == 0 {
        return None;
    }
    let src_abs = if clip.reversed {
        if src_frame >= range {
            clip.src_end_frame
        } else {
            clip.src_end_frame
                .saturating_sub(1)
                .saturating_sub(src_frame)
        }
    } else {
        src_frame.saturating_add(clip.src_start_frame)
    };
    if src_abs >= clip.src_end_frame {
        if clip.repeat {
            let src_off = src_frame % range;
            let looped = if clip.reversed {
                clip.src_end_frame
                    .saturating_sub(1)
                    .saturating_sub(src_off)
            } else {
                clip.src_start_frame + src_off
            };
            let idx = (looped as usize) * 2;
            if idx + 1 < clip.src.pcm.len() {
                return Some((clip.src.pcm[idx], clip.src.pcm[idx + 1]));
            }
        }
        return None;
    }
    let idx = (src_abs as usize) * 2;
    if idx + 1 < clip.src.pcm.len() {
        Some((clip.src.pcm[idx], clip.src.pcm[idx + 1]))
    } else {
        None
    }
}

pub(crate) fn mix_snapshot_clips_into_scratch(
    frames: usize,
    snap: &EngineSnapshot,
    pos0: u64,
    pos1: u64,
    scratch: &mut [f32],
    meter_scratch: Option<&mut TrackMeterScratch>,
) {
    let mut meter_scratch = meter_scratch;
    if let Some(ms) = meter_scratch.as_deref_mut() {
        for track_id in ms.active_track_ids.drain(..) {
            if let Some(buf) = ms.per_track_mix.get_mut(&track_id) {
                buf.resize(frames * 2, 0.0);
                buf.fill(0.0);
            }
        }
    }

    for clip in snap.clips.iter() {
        let clip_start = clip.start_frame;
        let clip_end = clip.start_frame.saturating_add(clip.length_frames);
        if clip_end <= pos0 || clip_start >= pos1 {
            continue;
        }

        let overlap_start = clip_start.max(pos0);
        let overlap_end = clip_end.min(pos1);
        if overlap_end <= overlap_start {
            continue;
        }

        let out_off = (overlap_start - pos0) as usize;
        let clip_off = overlap_start - clip_start;
        let mix_frames = (overlap_end - overlap_start) as usize;

        for f in 0..mix_frames {
            let local = clip_off + f as u64;

            let local_i64 = if local > i64::MAX as u64 {
                continue;
            } else {
                local as i64
            };
            let local_adj_i64 = local_i64.saturating_add(clip.local_src_offset_frames);
            if local_adj_i64 < 0 {
                continue;
            }
            let local_adj = local_adj_i64 as f64;

            let mut g = clip.gain;
            if clip.fade_in_frames > 0 && local < clip.fade_in_frames {
                // Use frame-centered fade-in so the first frame is not hard-zeroed.
                g *= ((local + 1) as f32 / clip.fade_in_frames as f32).clamp(0.0, 1.0);
            }
            if clip.fade_out_frames > 0 && local + clip.fade_out_frames > clip.length_frames {
                let remain = clip.length_frames.saturating_sub(local);
                g *= (remain as f32 / clip.fade_out_frames as f32).clamp(0.0, 1.0);
            }
            if g <= 0.0 {
                continue;
            }

            let Some((l, r)) = sample_clip_pcm(clip, local, local_adj) else {
                continue;
            };

            let oi = (out_off + f) * 2;
            let mixed_l = l * g;
            let mixed_r = r * g;
            scratch[oi] += mixed_l;
            scratch[oi + 1] += mixed_r;
            if let Some(ms) = meter_scratch.as_deref_mut() {
                let first_use_this_block = !ms.active_track_ids.iter().any(|id| id == &clip.track_id);
                let track_buf = ms
                    .per_track_mix
                    .entry(clip.track_id.clone())
                    .or_insert_with(|| vec![0.0; frames * 2]);
                if track_buf.len() != frames * 2 {
                    track_buf.resize(frames * 2, 0.0);
                }
                if first_use_this_block {
                    ms.active_track_ids.push(clip.track_id.clone());
                }
                track_buf[oi] += mixed_l;
                track_buf[oi + 1] += mixed_r;
            }
        }
    }
}

fn snapshot_has_pending_clip(snap: &EngineSnapshot, pos0: u64, pos1: u64) -> bool {
    snap.clips.iter().any(|clip| {
        if !clip.needs_synthesis || clip.rendered_pcm.is_some() {
            return false;
        }
        let clip_end = clip.start_frame.saturating_add(clip.length_frames);
        clip.start_frame < pos1 && clip_end > pos0
    })
}

fn render_snapshot_window(
    frames: usize,
    snap: &EngineSnapshot,
    pos0: u64,
    pos1: u64,
    scratch: &mut Vec<f32>,
) -> bool {
    scratch.resize(frames * 2, 0.0);
    scratch.fill(0.0);

    if snapshot_has_pending_clip(snap, pos0, pos1) {
        return false;
    }

    mix_snapshot_clips_into_scratch(
        frames,
        snap,
        pos0,
        pos1,
        scratch.as_mut_slice(),
        None,
    );
    true
}

fn collect_track_meter_block(
    frames: usize,
    snap: &EngineSnapshot,
    pos0: u64,
    pos1: u64,
    meter_scratch: &mut TrackMeterScratch,
) {
    for track_id in meter_scratch.active_track_ids.drain(..) {
        if let Some(buf) = meter_scratch.per_track_mix.get_mut(&track_id) {
            buf.resize(frames * 2, 0.0);
            buf.fill(0.0);
        }
    }

    for clip in snap.clips.iter() {
        let clip_start = clip.start_frame;
        let clip_end = clip.start_frame.saturating_add(clip.length_frames);
        if clip_end <= pos0 || clip_start >= pos1 {
            continue;
        }

        let overlap_start = clip_start.max(pos0);
        let overlap_end = clip_end.min(pos1);
        if overlap_end <= overlap_start {
            continue;
        }

        let out_off = (overlap_start - pos0) as usize;
        let clip_off = overlap_start - clip_start;
        let mix_frames = (overlap_end - overlap_start) as usize;

        let first_use_this_block = !meter_scratch
            .active_track_ids
            .iter()
            .any(|id| id == &clip.track_id);
        let track_buf = meter_scratch
            .per_track_mix
            .entry(clip.track_id.clone())
            .or_insert_with(|| vec![0.0; frames * 2]);
        if track_buf.len() != frames * 2 {
            track_buf.resize(frames * 2, 0.0);
        }
        if first_use_this_block {
            meter_scratch.active_track_ids.push(clip.track_id.clone());
        }

        for f in 0..mix_frames {
            let local = clip_off + f as u64;
            let local_i64 = if local > i64::MAX as u64 {
                continue;
            } else {
                local as i64
            };
            let local_adj_i64 = local_i64.saturating_add(clip.local_src_offset_frames);
            if local_adj_i64 < 0 {
                continue;
            }
            let local_adj = local_adj_i64 as f64;

            let mut g = clip.gain;
            if clip.fade_in_frames > 0 && local < clip.fade_in_frames {
                // Keep meter path consistent with audio callback fade behavior.
                g *= ((local + 1) as f32 / clip.fade_in_frames as f32).clamp(0.0, 1.0);
            }
            if clip.fade_out_frames > 0 && local + clip.fade_out_frames > clip.length_frames {
                let remain = clip.length_frames.saturating_sub(local);
                g *= (remain as f32 / clip.fade_out_frames as f32).clamp(0.0, 1.0);
            }
            if g <= 0.0 {
                continue;
            }

            let Some((l, r)) = sample_clip_pcm(clip, local, local_adj) else {
                continue;
            };

            let oi = (out_off + f) * 2;
            track_buf[oi] += l * g;
            track_buf[oi + 1] += r * g;
        }
    }
}

fn update_track_meter_state(
    snap: &EngineSnapshot,
    meter_scratch: &TrackMeterScratch,
    meter_state: &Arc<Mutex<std::collections::HashMap<String, TrackMeterValue>>>,
    meter_generation: &AtomicU64,
) {
    let mut next = std::collections::HashMap::with_capacity(snap.track_ids.len());
    if let Ok(mut state) = meter_state.lock() {
        for track_id in snap.track_ids.iter() {
            let block_peak = meter_scratch
                .per_track_mix
                .get(track_id)
                .map(|buf| {
                    buf.iter()
                        .fold(0.0f32, |acc, sample| acc.max(sample.abs()))
                })
                .unwrap_or(0.0);
            let prev = state.get(track_id).copied().unwrap_or_default();
            next.insert(
                track_id.clone(),
                TrackMeterValue {
                    peak_linear: block_peak,
                    max_peak_linear: prev.max_peak_linear.max(block_peak),
                    clipped: prev.clipped || block_peak >= 1.0,
                },
            );
        }
        *state = next;
        meter_generation.fetch_add(1, Ordering::Relaxed);
    }
}

fn blend_snapshot_windows_in_place(
    current_and_out: &mut [f32],
    from: &[f32],
    fade_remaining_frames: usize,
) {
    let total = SNAPSHOT_XFADE_FRAMES.max(1);
    let already_blended = total.saturating_sub(fade_remaining_frames);
    let frames = (current_and_out.len() / 2).min(from.len() / 2);

    for frame in 0..frames {
        let t = ((already_blended + frame + 1).min(total) as f32) / total as f32;
        let from_gain = 1.0 - t;
        let to_gain = t;
        let base = frame * 2;
        // 在 current_and_out 内部完成读取与复写
        current_and_out[base] = from[base] * from_gain + current_and_out[base] * to_gain;
        current_and_out[base + 1] =
            from[base + 1] * from_gain + current_and_out[base + 1] * to_gain;
    }
}

fn advance_playback_position(
    frames: usize,
    is_playing: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
) {
    let pos0 = position_frames.load(Ordering::Relaxed);
    let new_pos = pos0.saturating_add(frames as u64);
    position_frames.store(new_pos, Ordering::Relaxed);

    let dur = duration_frames.load(Ordering::Relaxed);
    if dur > 0 && new_pos >= dur {
        is_playing.store(false, Ordering::Relaxed);
    }
}

fn mix_into_scratch_stereo(
    frames: usize,
    snapshot: &Arc<ArcSwap<EngineSnapshot>>,
    is_playing: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
) {
    scratch.resize(frames * 2, 0.0);
    scratch.fill(0.0);

    if !is_playing.load(Ordering::Relaxed) {
        return;
    }

    let snap = snapshot.load_full();
    let pos0 = position_frames.load(Ordering::Relaxed);
    let pos1 = pos0.saturating_add(frames as u64);

    let snap_ptr = Arc::as_ptr(&snap) as usize;
    let current_ptr = transition
        .current_snapshot
        .as_ref()
        .map(|current| Arc::as_ptr(current) as usize)
        .unwrap_or(0);
    if current_ptr != 0 && current_ptr != snap_ptr {
        transition.fade_from_snapshot = transition.current_snapshot.take();
        transition.fade_remaining_frames = SNAPSHOT_XFADE_FRAMES;
    }
    transition.current_snapshot = Some(snap.clone());

    let current_ready = render_snapshot_window(frames, &snap, pos0, pos1, scratch);

    if !current_ready && transition.fade_from_snapshot.is_none() {
        // cursor 暂停，不推进 position，输出静音等待
        // 调试：每隔约 1s 打印一次
        static DEBUG_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DEBUG_LOG.get_or_init(|| {
            std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1")
        }) {
            static LAST_LOG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let now = pos0 / 44100; // rough seconds
            let last = LAST_LOG.load(Ordering::Relaxed);
            if now != last {
                LAST_LOG.store(now, Ordering::Relaxed);
                for clip in snap.clips.iter() {
                    if clip.needs_synthesis && clip.rendered_pcm.is_none() {
                        let clip_end = clip.start_frame.saturating_add(clip.length_frames);
                        if clip.start_frame < pos1 && clip_end > pos0 {
                            eprintln!(
                                "[mix] PENDING clip_id={} needs_synthesis=true rendered_pcm=None pos={}",
                                clip.clip_id, pos0
                            );
                        }
                    }
                }
            }
        }
        return;
    }

    if let Some(from_snapshot) = transition.fade_from_snapshot.as_ref() {
        let from_ready =
            render_snapshot_window(frames, from_snapshot, pos0, pos1, scratch_fade_from);

        if from_ready && !current_ready {
            scratch.resize(scratch_fade_from.len(), 0.0);
            scratch.copy_from_slice(scratch_fade_from.as_slice());
            advance_playback_position(frames, is_playing, position_frames, duration_frames);
            return;
        }

        if from_ready && current_ready && transition.fade_remaining_frames > 0 {
            // 直接就地混合，删掉极其耗时的 scratch.clone()
            blend_snapshot_windows_in_place(
                scratch.as_mut_slice(),
                scratch_fade_from.as_slice(),
                transition.fade_remaining_frames,
            );
            transition.fade_remaining_frames =
                transition.fade_remaining_frames.saturating_sub(frames);
            if transition.fade_remaining_frames == 0 {
                transition.fade_from_snapshot = None;
            }
        } else if current_ready {
            transition.fade_from_snapshot = None;
            transition.fade_remaining_frames = 0;
        }
    }

    if current_ready || transition.fade_from_snapshot.is_some() {
        advance_playback_position(frames, is_playing, position_frames, duration_frames);
    }
}

pub(crate) fn render_callback_f32(
    data: &mut [f32],
    out_channels: usize,
    snapshot: &Arc<ArcSwap<EngineSnapshot>>,
    is_playing: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter_scratch: &mut TrackMeterScratch,
    meter_state: &Arc<Mutex<std::collections::HashMap<String, TrackMeterValue>>>,
    meter_generation: &AtomicU64,
) {
    let frames = if out_channels == 0 {
        0
    } else {
        data.len() / out_channels
    };
    if frames == 0 {
        return;
    }

    let was_playing = is_playing.load(Ordering::Relaxed);
    if !was_playing {
        data.fill(0.0);
        return;
    }

    mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
    );
    let snap = snapshot.load_full();
    let pos1 = position_frames.load(Ordering::Relaxed);
    let pos0 = pos1.saturating_sub(frames as u64);
    collect_track_meter_block(frames, &snap, pos0, pos1, meter_scratch);
    update_track_meter_state(&snap, meter_scratch, meter_state, meter_generation);

    for f in 0..frames {
        let l = clamp11(scratch[f * 2]);
        let r = clamp11(scratch[f * 2 + 1]);
        if out_channels == 1 {
            data[f] = (l + r) * 0.5;
        } else {
            let base = f * out_channels;
            data[base] = l;
            data[base + 1] = r;
            for ch in 2..out_channels {
                data[base + ch] = 0.0;
            }
        }
    }
}

pub(crate) fn render_callback_i16(
    data: &mut [i16],
    out_channels: usize,
    snapshot: &Arc<ArcSwap<EngineSnapshot>>,
    is_playing: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter_scratch: &mut TrackMeterScratch,
    meter_state: &Arc<Mutex<std::collections::HashMap<String, TrackMeterValue>>>,
    meter_generation: &AtomicU64,
) {
    let frames = if out_channels == 0 {
        0
    } else {
        data.len() / out_channels
    };
    if frames == 0 {
        return;
    }

    if !is_playing.load(Ordering::Relaxed) {
        data.fill(0);
        return;
    }

    mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
    );
    let snap = snapshot.load_full();
    let pos1 = position_frames.load(Ordering::Relaxed);
    let pos0 = pos1.saturating_sub(frames as u64);
    collect_track_meter_block(frames, &snap, pos0, pos1, meter_scratch);
    update_track_meter_state(&snap, meter_scratch, meter_state, meter_generation);

    for f in 0..frames {
        let l = clamp11(scratch[f * 2]);
        let r = clamp11(scratch[f * 2 + 1]);
        if out_channels == 1 {
            let v = clamp11((l + r) * 0.5);
            data[f] = (v * i16::MAX as f32) as i16;
        } else {
            let base = f * out_channels;
            data[base] = (l * i16::MAX as f32) as i16;
            data[base + 1] = (r * i16::MAX as f32) as i16;
            for ch in 2..out_channels {
                data[base + ch] = 0;
            }
        }
    }
}

pub(crate) fn render_callback_u16(
    data: &mut [u16],
    out_channels: usize,
    snapshot: &Arc<ArcSwap<EngineSnapshot>>,
    is_playing: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter_scratch: &mut TrackMeterScratch,
    meter_state: &Arc<Mutex<std::collections::HashMap<String, TrackMeterValue>>>,
    meter_generation: &AtomicU64,
) {
    let frames = if out_channels == 0 {
        0
    } else {
        data.len() / out_channels
    };
    if frames == 0 {
        return;
    }

    if !is_playing.load(Ordering::Relaxed) {
        data.fill(u16::MAX / 2);
        return;
    }

    mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
    );
    let snap = snapshot.load_full();
    let pos1 = position_frames.load(Ordering::Relaxed);
    let pos0 = pos1.saturating_sub(frames as u64);
    collect_track_meter_block(frames, &snap, pos0, pos1, meter_scratch);
    update_track_meter_state(&snap, meter_scratch, meter_state, meter_generation);

    for f in 0..frames {
        let l = clamp11(scratch[f * 2]);
        let r = clamp11(scratch[f * 2 + 1]);
        if out_channels == 1 {
            let v = clamp11((l + r) * 0.5);
            // 用 Rust 自带的安全强转，不需要边界检测与 round 了
            data[f] = ((v * 0.5 + 0.5) * u16::MAX as f32) as u16;
        } else {
            let base = f * out_channels;
            data[base] = ((l * 0.5 + 0.5) * u16::MAX as f32) as u16;
            data[base + 1] = ((r * 0.5 + 0.5) * u16::MAX as f32) as u16;
            for ch in 2..out_channels {
                data[base + ch] = u16::MAX / 2;
            }
        }
    }
}
