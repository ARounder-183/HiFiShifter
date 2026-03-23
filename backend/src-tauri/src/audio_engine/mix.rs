// 实时音频混音引擎核心。
// 在 cpal 音频回调线程中运行，负责将 EngineSnapshot 中的 clips 混音为立体声输出。
// 当存在 per-track VST FX 链时，按轨道分组混音后逐轨道应用 VST 处理。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use super::types::EngineClip;
use super::types::EngineSnapshot;
use super::util::clamp11;

#[cfg(feature = "vst")]
use std::collections::HashMap;

const SNAPSHOT_XFADE_FRAMES: usize = 256;

#[derive(Default)]
pub(crate) struct SnapshotTransitionState {
    current_snapshot: Option<Arc<EngineSnapshot>>,
    fade_from_snapshot: Option<Arc<EngineSnapshot>>,
    fade_remaining_frames: usize,
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
    let src_abs = src_frame.saturating_add(clip.src_start_frame);
    if src_abs >= clip.src_end_frame {
        if clip.repeat {
            let range = clip.src_end_frame.saturating_sub(clip.src_start_frame);
            if range == 0 {
                return None;
            }
            let looped = clip.src_start_frame + ((src_abs - clip.src_start_frame) % range);
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
    _frames: usize,
    snap: &EngineSnapshot,
    pos0: u64,
    pos1: u64,
    scratch: &mut [f32],
) {
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
                g *= (local as f32 / clip.fade_in_frames as f32).clamp(0.0, 1.0);
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
            scratch[oi] += l * g;
            scratch[oi + 1] += r * g;
        }
    }
}

// ─── per-track VST 混音 ─────────────────────────────────────────────────────

/// 按轨道分组混音 + 逐轨道应用 VST FX 链，结果写入 scratch。
///
/// 工作流程：
/// 1. 收集所有含 VST 链的轨道 ID
/// 2. 对有 VST 链的轨道：逐轨道混音 → VST 处理 → 累加到 scratch
/// 3. 对无 VST 链的轨道：直接混音到 scratch（与原始路径一致）
///
/// VST 插件实例使用 `try_lock` 非阻塞获取锁：
/// - 成功：正常处理音频
/// - 失败：跳过 VST 处理，直接使用干信号（避免音频回调卡顿）
#[cfg(feature = "vst")]
pub(crate) fn mix_snapshot_clips_with_vst(
    frames: usize,
    snap: &EngineSnapshot,
    pos0: u64,
    pos1: u64,
    scratch: &mut [f32],
) {
    // 如果没有任何 VST stages，走快速路径
    if snap.vst_stages.is_empty() {
        mix_snapshot_clips_into_scratch(frames, snap, pos0, pos1, scratch);
        return;
    }

    // 按轨道分组 clips
    let mut clips_by_track: HashMap<&str, Vec<&EngineClip>> = HashMap::new();
    for clip in snap.clips.iter() {
        clips_by_track
            .entry(clip.track_id.as_str())
            .or_default()
            .push(clip);
    }

    let buf_len = frames * 2; // stereo interleaved

    for (track_id, clips) in &clips_by_track {
        let has_vst = snap.vst_stages.contains_key(*track_id);

        if has_vst {
            // 分配轨道临时缓冲区
            let mut track_buf = vec![0.0f32; buf_len];

            // 混音该轨道的所有 clips 到临时缓冲区
            for clip in clips {
                mix_single_clip_into_buffer(clip, pos0, pos1, &mut track_buf);
            }

            // 应用 VST FX 链（try_lock 非阻塞）
            if let Some(stages) = snap.vst_stages.get(*track_id) {
                apply_vst_chain_to_buffer_trylock(
                    &stages.instances,
                    &mut track_buf,
                    frames,
                    snap.sample_rate,
                );
            }

            // 累加到最终 scratch
            for (s, &t) in scratch.iter_mut().zip(track_buf.iter()) {
                *s += t;
            }
        } else {
            // 无 VST 链：直接混入 scratch
            for clip in clips {
                mix_single_clip_into_buffer(clip, pos0, pos1, scratch);
            }
        }
    }
}

/// 将单个 clip 混入 buffer（与 mix_snapshot_clips_into_scratch 中的单 clip 逻辑一致）。
fn mix_single_clip_into_buffer(
    clip: &EngineClip,
    pos0: u64,
    pos1: u64,
    buffer: &mut [f32],
) {
    let clip_start = clip.start_frame;
    let clip_end = clip.start_frame.saturating_add(clip.length_frames);
    if clip_end <= pos0 || clip_start >= pos1 {
        return;
    }

    let overlap_start = clip_start.max(pos0);
    let overlap_end = clip_end.min(pos1);
    if overlap_end <= overlap_start {
        return;
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
            g *= (local as f32 / clip.fade_in_frames as f32).clamp(0.0, 1.0);
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
        buffer[oi] += l * g;
        buffer[oi + 1] += r * g;
    }
}

/// 对立体声交错缓冲区应用 VST 插件链（非阻塞 try_lock）。
///
/// 对每个插件实例：
/// - `try_lock` 成功 → 正常处理音频
/// - `try_lock` 失败 → 跳过该插件（pass-through），避免实时线程卡顿
#[cfg(feature = "vst")]
fn apply_vst_chain_to_buffer_trylock(
    instances: &[Arc<std::sync::Mutex<crate::vst_host::plugin_instance::VstPluginInstance>>],
    buffer: &mut [f32],
    frames: usize,
    sample_rate: u32,
) {
    for instance_arc in instances {
        // 非阻塞尝试获取锁
        let mut inst = match instance_arc.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                // 锁被占用（可能 GUI 线程在操作），跳过此插件
                continue;
            }
        };

        if inst.bypassed {
            continue;
        }

        // 确保采样率正确
        let sr = sample_rate as f32;
        if (inst.sample_rate - sr).abs() > 1.0 {
            inst.set_sample_rate(sr);
        }

        let block_size = inst.block_size.max(64);
        let num_inputs = inst.num_inputs.max(1) as usize;
        let num_outputs = inst.num_outputs.max(1) as usize;

        // 从 interleaved stereo 拆分为 per-channel buffers
        let mut input_left = Vec::with_capacity(frames);
        let mut input_right = Vec::with_capacity(frames);
        for f in 0..frames {
            input_left.push(buffer[f * 2]);
            input_right.push(buffer[f * 2 + 1]);
        }

        // 适配插件输入通道数
        let mut input_channels: Vec<Vec<f32>> = Vec::with_capacity(num_inputs);
        if num_inputs >= 2 {
            input_channels.push(input_left);
            input_channels.push(input_right);
            for _ in 2..num_inputs {
                input_channels.push(vec![0.0f32; frames]);
            }
        } else {
            // 单声道输入：混合 L/R
            let mono: Vec<f32> = (0..frames)
                .map(|f| (buffer[f * 2] + buffer[f * 2 + 1]) * 0.5)
                .collect();
            input_channels.push(mono);
        }

        // 准备输出通道
        let mut output_channels: Vec<Vec<f32>> = (0..num_outputs)
            .map(|_| vec![0.0f32; frames])
            .collect();

        // 分块处理
        let mut offset = 0;
        while offset < frames {
            let chunk_len = (frames - offset).min(block_size);

            let input_chunks: Vec<Vec<f32>> = input_channels
                .iter()
                .map(|ch| ch[offset..offset + chunk_len].to_vec())
                .collect();

            let mut output_chunks: Vec<Vec<f32>> = (0..num_outputs)
                .map(|_| vec![0.0f32; chunk_len])
                .collect();

            inst.process(&input_chunks, &mut output_chunks);

            for (ch_idx, chunk) in output_chunks.iter().enumerate() {
                if ch_idx < output_channels.len() {
                    output_channels[ch_idx][offset..offset + chunk_len]
                        .copy_from_slice(&chunk[..chunk_len]);
                }
            }

            offset += chunk_len;
        }

        // 将处理后的音频写回 interleaved buffer
        if num_outputs >= 2 {
            for f in 0..frames {
                buffer[f * 2] = output_channels[0][f];
                buffer[f * 2 + 1] = output_channels[1][f];
            }
        } else if num_outputs == 1 {
            // 单声道输出 → 复制到双声道
            for f in 0..frames {
                buffer[f * 2] = output_channels[0][f];
                buffer[f * 2 + 1] = output_channels[0][f];
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

    // 当 VST feature 启用且存在 VST stages 时，使用 per-track VST 混音路径
    #[cfg(feature = "vst")]
    {
        if !snap.vst_stages.is_empty() {
            mix_snapshot_clips_with_vst(frames, snap, pos0, pos1, scratch.as_mut_slice());
            return true;
        }
    }

    mix_snapshot_clips_into_scratch(frames, snap, pos0, pos1, scratch.as_mut_slice());
    true
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
