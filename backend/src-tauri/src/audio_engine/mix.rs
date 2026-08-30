use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use super::types::EngineClip;
use super::types::EngineSnapshot;
use super::util::clamp11;

const SNAPSHOT_XFADE_FRAMES: usize = 256;

/// Unsigned 16-bit silence level (0x8000), keeping the waveform centered.
const U16_SILENCE: u16 = 32768;

/// Map a [-1, 1] sample to unsigned 16-bit with 0x8000 as the zero point.
#[inline]
fn f32_to_u16(v: f32) -> u16 {
    ((v * 32768.0) + 32768.0).round().clamp(0.0, 65535.0) as u16
}

#[derive(Default)]
pub(crate) struct SnapshotTransitionState {
    current_snapshot: Option<Arc<EngineSnapshot>>,
    fade_from_snapshot: Option<Arc<EngineSnapshot>>,
    fade_remaining_frames: usize,
}

/// RT-local per-track meter scratch. Lives entirely inside the audio
/// callback thread: `track_peaks` holds the running max amplitude per
/// track slot (index = `snap.track_ids` position) for the current block.
#[derive(Default)]
pub(crate) struct TrackMeterScratch {
    track_peaks: Vec<f32>,
}

impl TrackMeterScratch {
    fn reset(&mut self, track_count: usize) {
        if self.track_peaks.len() < track_count {
            // Grows only when the project gains tracks; rare and bounded.
            self.track_peaks.resize(track_count, 0.0);
        }
        for p in self.track_peaks.iter_mut() {
            *p = 0.0;
        }
    }
}

/// Lock-free handoff of per-track block peaks from the audio callback to
/// the meter thread. The RT side only writes fixed atomic slots (no locks,
/// no allocation); the meter thread polls `generation` and publishes the
/// values into the shared `meter_state` map, keeping stderr logging and
/// map rebuilds off the RT thread.
pub(crate) struct TrackMeterBus {
    /// f32 bits of each track slot's latest block peak.
    slots: Vec<AtomicU32>,
    generation: AtomicU64,
    /// Position of the block that auto-paused playback during a pending
    /// background render (0 = none). Drained + logged by the meter thread.
    auto_pause_pos: AtomicU64,
    /// Position of the last block that rendered silence while a clip was
    /// still pending synthesis. Debug aid, drained by the meter thread.
    pending_pos: AtomicU64,
}

impl TrackMeterBus {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            generation: AtomicU64::new(0),
            auto_pause_pos: AtomicU64::new(0),
            pending_pos: AtomicU64::new(0),
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    pub(crate) fn slot_peak(&self, slot: usize) -> f32 {
        f32::from_bits(
            self.slots
                .get(slot)
                .map(|s| s.load(Ordering::Relaxed))
                .unwrap_or(0),
        )
    }

    /// Drain the recorded auto-pause position (0 if none recorded).
    pub(crate) fn take_auto_pause_pos(&self) -> u64 {
        self.auto_pause_pos.swap(0, Ordering::Relaxed)
    }

    /// Drain the recorded pending-clip position (0 if none recorded).
    pub(crate) fn take_pending_pos(&self) -> u64 {
        self.pending_pos.swap(0, Ordering::Relaxed)
    }

    /// RT side: publish one block of per-track peaks (0.0 for silent slots).
    fn publish_block(&self, peaks: &[f32], track_count: usize) {
        for (slot, peak) in peaks.iter().take(track_count).enumerate() {
            if let Some(s) = self.slots.get(slot) {
                s.store(peak.to_bits(), Ordering::Relaxed);
            }
        }
        for slot in peaks.len()..track_count {
            if let Some(s) = self.slots.get(slot) {
                s.store(0.0f32.to_bits(), Ordering::Relaxed);
            }
        }
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

fn sample_automation_curve(
    curve: Option<&[f32]>,
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
    let i0 = (idx_f.floor().max(0.0) as usize).min(curve.len().saturating_sub(1));
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 线性平衡式声像：center 保持两声道均为 1.0，硬左/硬右时关闭对侧声道。
#[inline]
fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    if pan <= 0.0 {
        (1.0, 1.0 + pan)
    } else {
        (1.0 - pan, 1.0)
    }
}

/// 对一帧 PCM 应用共通音量/声像自动化（vslib 路径的曲线为 None，由处理器内部完成）。
#[inline]
fn apply_mix_automation(clip: &EngineClip, abs_frame: u64, l: f32, r: f32) -> (f32, f32) {
    let vol = sample_automation_curve(
        clip.volume_curve.as_deref().map(|v| v.as_slice()),
        abs_frame,
        clip.src.sample_rate,
        clip.volume_curve_frame_period_ms,
        1.0,
    );
    let pan = sample_automation_curve(
        clip.pan_curve.as_deref().map(|v| v.as_slice()),
        abs_frame,
        clip.src.sample_rate,
        clip.pan_curve_frame_period_ms,
        0.0,
    );
    let (left_gain, right_gain) = pan_gains(pan);
    (l * vol * left_gain, r * vol * right_gain)
}

/// 采样 clip 在 local 帧处的原始 PCM（不含 gain/fade，但含 volume/pan 自动化）。
/// 返回 None 表示该帧应静音（越界、leading silence 等）。
#[inline]
fn sample_clip_pcm(clip: &EngineClip, local: u64, local_adj: f64) -> Option<(f32, f32)> {
    let abs_frame = clip.start_frame.saturating_add(local);
    let raw = if let Some(ref rendered) = clip.rendered_pcm {
        let idx = (local as usize) * 2;
        if idx + 1 >= rendered.len() {
            // rendered_pcm 存在但越界时返回静音
            None
        } else {
            let mut left = rendered[idx];
            let mut right = rendered[idx + 1];
            if let Some(ref breath_noise) = clip.breath_noise_pcm {
                if idx + 1 < breath_noise.len() {
                    let gain = sample_automation_curve(
                        clip.breath_curve.as_deref().map(|v| v.as_slice()),
                        abs_frame,
                        clip.src.sample_rate,
                        clip.breath_curve_frame_period_ms,
                        1.0,
                    );
                    left += breath_noise[idx] * gain;
                    right += breath_noise[idx + 1] * gain;
                }
            }
            Some((left, right))
        }
    } else {
        // 若该 clip 需要合成（pitch edit）但尚未渲染完成，按调用约定渲染分支
        // 已经返回；此处仅处理“无需合成，直接采样源 PCM”的路径。
        let src_frame_f = local_adj * clip.playback_rate;
        let src_frame = src_frame_f.round();

        // ── Loop（循环源）：对整个媒体缓冲做模运算回绕 ─────────────────────
        // 语义：src(t) = floor_mod(anchor ± t·rate, D)。正放从 source_start
        // 向上、倒放从 source_end 向下；越过文件边界后环绕到另一侧继续 ——
        // 即"循环原始音频文件"（对齐 REAPER 的 Loop source 行为）。
        if let Some(anchor) = clip.loop_anchor_frame {
            let total = clip.src.frames.max(1) as i64;
            let idx_i = if clip.reversed {
                anchor - src_frame as i64
            } else {
                anchor + src_frame as i64
            };
            let idx = idx_i.rem_euclid(total) as usize;
            let base = idx * 2;
            if base + 1 < clip.src.pcm.len() {
                Some((clip.src.pcm[base], clip.src.pcm[base + 1]))
            } else {
                None
            }
        } else {
            let src_frame_u = if src_frame >= 0.0 {
                src_frame as u64
            } else {
                0
            };
            let range = clip.src_end_frame.saturating_sub(clip.src_start_frame);
            if range == 0 {
                return None;
            }
            let src_abs = if clip.reversed {
                if src_frame_u >= range {
                    clip.src_end_frame
                } else {
                    clip.src_end_frame
                        .saturating_sub(1)
                        .saturating_sub(src_frame_u)
                }
            } else {
                src_frame_u.saturating_add(clip.src_start_frame)
            };
            if src_abs >= clip.src_end_frame {
                if clip.repeat {
                    let src_off = src_frame_u % range;
                    let looped = if clip.reversed {
                        clip.src_end_frame.saturating_sub(1).saturating_sub(src_off)
                    } else {
                        clip.src_start_frame + src_off
                    };
                    let idx = (looped as usize) * 2;
                    if idx + 1 < clip.src.pcm.len() {
                        Some((clip.src.pcm[idx], clip.src.pcm[idx + 1]))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                let idx = (src_abs as usize) * 2;
                if idx + 1 < clip.src.pcm.len() {
                    Some((clip.src.pcm[idx], clip.src.pcm[idx + 1]))
                } else {
                    None
                }
            }
        }
    };

    raw.map(|(left, right)| apply_mix_automation(clip, abs_frame, left, right))
}

pub(crate) fn mix_snapshot_clips_into_scratch(
    _frames: usize,
    snap: &EngineSnapshot,
    pos0: u64,
    pos1: u64,
    scratch: &mut [f32],
    meter: Option<&mut TrackMeterScratch>,
) {
    let mut meter = meter;
    let has_meter = meter.is_some();

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

        // Meter slot lookup happens once per clip (not per frame); it only
        // costs a few short string compares against snap.track_ids.
        let meter_slot = if has_meter {
            snap.track_ids.iter().position(|id| id == &clip.track_id)
        } else {
            None
        };
        if let (Some(m), Some(slot)) = (meter.as_deref_mut(), meter_slot) {
            if m.track_peaks.len() <= slot {
                m.track_peaks.resize(slot + 1, 0.0);
            }
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
                g *= match &clip.fade_in_lut {
                    Some(lut) => crate::fade_curves::sample_fade_lut(
                        lut,
                        ((local + 1) as f64 / clip.fade_in_frames as f64)
                            * crate::fade_curves::FADE_LUT_SIZE as f64,
                    ),
                    None => ((local + 1) as f32 / clip.fade_in_frames as f32).clamp(0.0, 1.0),
                };
            }
            if clip.fade_out_frames > 0 && local + clip.fade_out_frames > clip.length_frames {
                let remain = clip.length_frames.saturating_sub(local);
                // 淡出表按【区间内时间进度】下降采样（t=0 处 1 → t=1 处 0），
                // 因此必须用"已消耗进度"索引。剩余比例的走向恰好相反，
                // 用它做索引会把淡出整体反成淡入（历史 bug）。线性分支的
                // remain/N 与补号形式恒等，保持不动。
                let consumed = 1.0 - remain as f64 / clip.fade_out_frames as f64;
                g *= match &clip.fade_out_lut {
                    Some(lut) => crate::fade_curves::sample_fade_lut(
                        lut,
                        consumed * crate::fade_curves::FADE_LUT_SIZE as f64,
                    ),
                    None => (remain as f32 / clip.fade_out_frames as f32).clamp(0.0, 1.0),
                };
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
            if let (Some(m), Some(slot)) = (meter.as_deref_mut(), meter_slot) {
                let peak = &mut m.track_peaks[slot];
                let l_abs = mixed_l.abs();
                let r_abs = mixed_r.abs();
                if l_abs > *peak {
                    *peak = l_abs;
                }
                if r_abs > *peak {
                    *peak = r_abs;
                }
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
    meter: Option<&mut TrackMeterScratch>,
) -> bool {
    scratch.resize(frames * 2, 0.0);
    scratch.fill(0.0);

    if snapshot_has_pending_clip(snap, pos0, pos1) {
        return false;
    }

    mix_snapshot_clips_into_scratch(frames, snap, pos0, pos1, scratch.as_mut_slice(), meter);
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

/// Outcome of one callback block, consumed by the RT-side meter publish.
/// Blocks muted while waiting on a pending render carry zeroed peaks, so
/// publishing them keeps the meters at silence for the duration.
pub(crate) struct BlockRender {
    snapshot: Arc<EngineSnapshot>,
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
    meter: &mut TrackMeterScratch,
    bus: &TrackMeterBus,
) -> Option<BlockRender> {
    scratch.resize(frames * 2, 0.0);
    scratch.fill(0.0);

    if !is_playing.load(Ordering::Relaxed) {
        return None;
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

    meter.reset(snap.track_ids.len());
    let current_ready =
        render_snapshot_window(frames, &snap, pos0, pos1, scratch, Some(&mut *meter));

    if !current_ready && transition.fade_from_snapshot.is_none() {
        // 若后台预渲染激活，自动暂停播放（而非无限静音等待）。
        // 已渲染完成的 clip 在后台渲染线程存入缓存后，
        // 用户可手动再次按下播放键继续。
        let bg_render =
            crate::commands::playback::BG_RENDER_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
        if bg_render {
            is_playing.store(false, std::sync::atomic::Ordering::Relaxed);
            // stderr I/O must stay off the RT thread: record the position and
            // let the meter thread log it.
            bus.auto_pause_pos.store(pos0, Ordering::Relaxed);
        }
        // Debug aid: report the pending clip via the meter thread as well.
        bus.pending_pos.store(pos0, Ordering::Relaxed);
        // cursor 暂停，不推进 position，输出静音等待
        return Some(BlockRender { snapshot: snap });
    }

    if let Some(from_snapshot) = transition.fade_from_snapshot.as_ref() {
        let from_ready = if current_ready {
            render_snapshot_window(frames, from_snapshot, pos0, pos1, scratch_fade_from, None)
        } else {
            // Output this block comes from the fade-from snapshot; meter it.
            meter.reset(from_snapshot.track_ids.len());
            render_snapshot_window(
                frames,
                from_snapshot,
                pos0,
                pos1,
                scratch_fade_from,
                Some(&mut *meter),
            )
        };

        if from_ready && !current_ready {
            scratch.resize(scratch_fade_from.len(), 0.0);
            scratch.copy_from_slice(scratch_fade_from.as_slice());
            advance_playback_position(frames, is_playing, position_frames, duration_frames);
            return Some(BlockRender { snapshot: snap });
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
    Some(BlockRender { snapshot: snap })
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
    meter_bus: &TrackMeterBus,
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

    let block = mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
        &mut *meter_scratch,
        meter_bus,
    );
    if let Some(block) = block.as_ref() {
        // Publish per-track peaks so meters always mirror the output. The
        // peaks were zeroed by reset() inside mix_into_scratch_stereo before
        // mixing, so a block muted while waiting on a pending render
        // publishes zeros here — do NOT reset again or real peaks are lost.
        meter_bus.publish_block(&meter_scratch.track_peaks, block.snapshot.track_ids.len());
    }

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
    meter_bus: &TrackMeterBus,
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

    let block = mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
        &mut *meter_scratch,
        meter_bus,
    );
    if let Some(block) = block.as_ref() {
        // Publish per-track peaks so meters always mirror the output. The
        // peaks were zeroed by reset() inside mix_into_scratch_stereo before
        // mixing, so a block muted while waiting on a pending render
        // publishes zeros here — do NOT reset again or real peaks are lost.
        meter_bus.publish_block(&meter_scratch.track_peaks, block.snapshot.track_ids.len());
    }

    for f in 0..frames {
        let l = clamp11(scratch[f * 2]);
        let r = clamp11(scratch[f * 2 + 1]);
        if out_channels == 1 {
            let v = clamp11((l + r) * 0.5);
            data[f] = (v * i16::MAX as f32).round() as i16;
        } else {
            let base = f * out_channels;
            data[base] = (l * i16::MAX as f32).round() as i16;
            data[base + 1] = (r * i16::MAX as f32).round() as i16;
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
    meter_bus: &TrackMeterBus,
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

    let block = mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
        &mut *meter_scratch,
        meter_bus,
    );
    if let Some(block) = block.as_ref() {
        // Publish per-track peaks so meters always mirror the output. The
        // peaks were zeroed by reset() inside mix_into_scratch_stereo before
        // mixing, so a block muted while waiting on a pending render
        // publishes zeros here — do NOT reset again or real peaks are lost.
        meter_bus.publish_block(&meter_scratch.track_peaks, block.snapshot.track_ids.len());
    }

    for f in 0..frames {
        let l = clamp11(scratch[f * 2]);
        let r = clamp11(scratch[f * 2 + 1]);
        if out_channels == 1 {
            let v = clamp11((l + r) * 0.5);
            data[f] = f32_to_u16(v);
        } else {
            let base = f * out_channels;
            data[base] = f32_to_u16(l);
            data[base + 1] = f32_to_u16(r);
            for ch in 2..out_channels {
                data[base + ch] = U16_SILENCE;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_mix_automation, sample_automation_curve};
    use crate::audio_engine::types::{EngineClip, ResampledStereo};
    use std::sync::Arc;

    fn clip_with_curves(volume_curve: Option<Vec<f32>>) -> EngineClip {
        let pcm = Arc::new(vec![1.0f32; 8]);
        EngineClip {
            clip_id: "clip-a".to_string(),
            track_id: "track-a".to_string(),
            start_frame: 0,
            length_frames: 4,
            src: ResampledStereo {
                sample_rate: 44_100,
                frames: 4,
                pcm,
            },
            src_start_frame: 0,
            src_end_frame: 4,
            reversed: false,
            playback_rate: 1.0,
            local_src_offset_frames: 0,
            repeat: false,
            loop_anchor_frame: None,
            fade_in_frames: 0,
            fade_out_frames: 0,
            fade_in_lut: None,
            fade_out_lut: None,
            gain: 1.0,
            rendered_pcm: None,
            breath_noise_pcm: None,
            breath_curve: None,
            breath_curve_frame_period_ms: 5.0,
            volume_curve: volume_curve.map(Arc::new),
            volume_curve_frame_period_ms: 5.0,
            pan_curve: None,
            pan_curve_frame_period_ms: 5.0,
            needs_synthesis: false,
        }
    }

    #[test]
    fn volume_curve_scales_mixed_output() {
        // 曲线第 0 帧为 0.5：0 号样本必须被压到一半。
        let clip = clip_with_curves(Some(vec![0.5f32]));
        let (l, r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 0.5).abs() < 1e-6, "left got {l}");
        assert!((r - 0.5).abs() < 1e-6, "right got {r}");
    }

    #[test]
    fn missing_volume_curve_is_unity() {
        let clip = clip_with_curves(None);
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-6, "left got {l}");
    }

    #[test]
    fn volume_curve_samples_at_timeline_absolute_frame() {
        // 曲线按**绝对时间**索引：fp=5ms → 每秒 200 帧。
        // 断言采样索引随绝对时间线性推进（而非 clip 局部时间）。
        let curve = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let at = |abs_frame: u64| {
            sample_automation_curve(Some(&curve), abs_frame, 44_100, 5.0, 1.0)
        };
        assert!(at(0) < 1e-6, "abs 0s reads curve frame 0");
        // 1ms = 44.1 样本 → 曲线帧 0.2
        assert!((at(44) - 0.2).abs() < 0.05, "got {}", at(44));
        // 5ms = 220.5 样本 → 曲线帧 1
        assert!((at(220) - 1.0).abs() < 0.05, "got {}", at(220));
        // 曲线末尾之后保持末值（不回落到 default）
        assert!((at(44_100) - 4.0).abs() < 1e-6, "got {}", at(44_100));
    }
}
