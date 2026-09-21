use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use super::metronome::MetronomeRt;
use super::metronome::MetronomeVoices;
use super::types::EngineClip;
use super::types::EngineSnapshot;
use super::util::clamp11;

const SNAPSHOT_XFADE_FRAMES: usize = 256;

/// Unsigned 16-bit silence level (0x8000), keeping the waveform centered.
// u16 输出格式的静音样本（f32 0.0 → i16/u16 中点）。pub 供 engine.rs 的
// panic 恢复路径复用，保证与正常静音同值（32767 会产生 1 LSB 阶跃）。
pub(crate) const U16_SILENCE: u16 = 32768;

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
    /// Position of the block that entered "transport waiting for render"
    /// （原地等待后台渲染：位置冻结、静音输出，0 = none）。
    /// Drained + logged by the meter thread.
    transport_wait_pos: AtomicU64,
    /// Position of the last block that rendered silence while a clip was
    /// still pending synthesis. Debug aid, drained by the meter thread.
    pending_pos: AtomicU64,
}

impl TrackMeterBus {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            generation: AtomicU64::new(0),
            transport_wait_pos: AtomicU64::new(0),
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

    /// Drain the recorded transport-wait position (0 if none recorded).
    pub(crate) fn take_transport_wait_pos(&self) -> u64 {
        self.transport_wait_pos.swap(0, Ordering::Relaxed)
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

/// 采样动态（DYN）曲线在绝对帧处的增益。
///
/// 语义：`增益 = 目标电平 / max(原声电平, 静音下限)`（两者同处倍率域），
/// 并带三类保护 ——
/// - **曲线不存在** → 1.0（该轨道组根本没在用动态）；
/// - **原声基线不存在**（分析未就绪）→ 1.0，绝不凭空造增益；
/// - 原声低于静音下限 → 分母钳到下限，增益**有界**（不放大无内容帧）；衰减/
///   静音照常生效（见 `compute_dyn_gain`）；
/// - 目标为 `DYN_FOLLOW_ORIG` 哨兵 → 1.0（**防御性兜底**：正常路径下曲线已在
///   装配期解析（见 `resolve_dyn_sentinels_for_audio`），引擎不该再看到哨兵）。
///
/// 采样规则与 volume/pan 完全一致（`sample_automation_curve`，越界钳制到末值），
/// 保证三者在同一时间轴上对齐。
#[inline]
fn dyn_gain_at(clip: &EngineClip, abs_frame: u64) -> f32 {
    if clip.dyn_curve.as_ref().is_none_or(|c| c.is_empty()) {
        return 1.0;
    }
    // 用 DYN 专用采样器：越界回落到"沿用原声"哨兵，而不是持有末值 ——
    // 否则用户画的最后一个目标电平会被 hold 到曲线尽头，对后续所有音频
    // 逐帧强加同一个目标电平（末点污染）。
    let target = crate::renderer::common_params::sample_dyn_curve_at_frame(
        clip.dyn_curve.as_deref().map(|v| v.as_slice()),
        abs_frame,
        clip.src.sample_rate,
        clip.dyn_curve_frame_period_ms,
    );
    // 哨兵：沿用原声，不做任何改变（无需再采样原声曲线）。
    if target < 0.0 {
        return 1.0;
    }
    // 基线缺失（或空曲线）时不得推导增益：采样器的兜底默认值只是占位，
    // 拿它去算 `目标 / 兜底` 会产生一个凭空的增益（曾表现为 max gain）。
    let Some(orig_curve) = clip.dyn_orig_curve.as_deref().filter(|c| !c.is_empty()) else {
        return 1.0;
    };
    let orig = sample_automation_curve(
        Some(orig_curve),
        abs_frame,
        clip.src.sample_rate,
        clip.dyn_curve_frame_period_ms,
        // 0.0 = "该帧无基线数据"（与 DYN_FOLLOW_ORIG 同名不同域：这是分母侧）。
        // 不能用 DYN_SILENCE_FLOOR：那会让"无数据帧"被当成无内容帧，于是任何
        // 超过 −60 dBFS 的目标都被读成放大请求而被拒绝 —— 曲线在基线缺口处
        // 会静默失效。
        0.0,
    );
    crate::renderer::common_params::compute_dyn_gain(target, orig)
}

/// 对一帧 PCM 应用共通音量/声像/动态自动化。
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
    let dyn_gain = dyn_gain_at(clip, abs_frame);
    let (left_gain, right_gain) = pan_gains(pan);
    let gain = vol * dyn_gain;
    (l * gain * left_gain, r * gain * right_gain)
}

/// 采样 clip 在 local 帧处的原始 PCM（不含 gain/fade，但含 volume/pan 自动化）。
/// 返回 None 表示该帧应静音（越界、leading silence 等）。
#[inline]
/// Take 声道模式的实时采样映射（与离线 `condition_take_channels` 语义一致）。
/// 仅用于**源 PCM** 读取路径；Swap/MonoLeft/MonoRight 是纯平面选择，
/// MonoMix 为每样本一次加法 —— 均为零分配。
fn apply_take_channel_mode(left: f32, right: f32, mode: crate::channel_mode::TakeChannelMode) -> (f32, f32) {
    match mode {
        crate::channel_mode::TakeChannelMode::Normal => (left, right),
        crate::channel_mode::TakeChannelMode::Swap => (right, left),
        crate::channel_mode::TakeChannelMode::MonoMix => {
            let v = (left + right) * 0.5;
            (v, v)
        }
        crate::channel_mode::TakeChannelMode::MonoLeft => (left, left),
        crate::channel_mode::TakeChannelMode::MonoRight => (right, right),
    }
}

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
                Some(apply_take_channel_mode(
                    clip.src.pcm[base],
                    clip.src.pcm[base + 1],
                    clip.channel_mode,
                ))
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
                        Some(apply_take_channel_mode(
                            clip.src.pcm[idx],
                            clip.src.pcm[idx + 1],
                            clip.channel_mode,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                let idx = (src_abs as usize) * 2;
                if idx + 1 < clip.src.pcm.len() {
                    Some(apply_take_channel_mode(
                        clip.src.pcm[idx],
                        clip.src.pcm[idx + 1],
                        clip.channel_mode,
                    ))
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

        // 淡出（端点锁定 + 内容耗尽收缩）：
        // - 端点锁定：N 帧区间的最后一帧（clip 末帧）进度恰为 1 → 增益精确
        //   0。旧公式 `1-remain/N` 让末帧进度停在 1-1/N，对“先慢后快”
        //   曲线（e<1）留下 (1/N)^e 级增益阶跃 → Click。
        // - 内容耗尽：深拉伸（播放速率 0.3~0.5）的 clip 被拉长到超出源窗口
        //   时，源内容会在淡出区之前/内部耗尽（sample_clip_pcm 返回 None）。
        //   若直接静音，会在增益还很大时硬切（与曲线形状无关）。处理：
        //   淡出区间收缩为 [E-N, L]（E=内容末端），内容末端之后按淡出增益
        //   保持末帧衰减 —— 全程无阶跃。
        let default_zone_start = clip.length_frames.saturating_sub(clip.fade_out_frames);
        let content_end = clip_content_end_frame(clip);
        let fade_zone_start = match content_end {
            Some(end) if end < default_zone_start => end.saturating_sub(clip.fade_out_frames),
            _ => default_zone_start,
        };
        let mut last_l: f32 = 0.0;
        let mut last_r: f32 = 0.0;
        let mut has_last: bool = false;

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
            if clip.fade_out_frames > 0 && local >= fade_zone_start && local < clip.length_frames {
                // 淡出表按【区间内时间进度】下降采样（t=0 处 1 → t=1 处 0），
                // 因此必须用"已消耗进度"索引。剩余比例的走向恰好相反，
                // 用它做索引会把淡出整体反成淡入（历史 bug）。进度起点
                // 1/N、终点 1（端点锁定），区间由 content_end 决定。
                let progress =
                    (local - fade_zone_start + 1) as f64 / clip.fade_out_frames.max(1) as f64;
                if progress <= 1.0 {
                    g *= match &clip.fade_out_lut {
                        Some(lut) => crate::fade_curves::sample_fade_lut(
                            lut,
                            progress * crate::fade_curves::FADE_LUT_SIZE as f64,
                        ),
                        None => (progress as f32).clamp(0.0, 1.0),
                    };
                } else {
                    // 收缩后的淡出在这帧之前已走完 → 静音。
                    g = 0.0;
                }
            }
            if g <= 0.0 {
                continue;
            }

            let oi = (out_off + f) * 2;
            let Some((l, r)) = sample_clip_pcm(clip, local, local_adj) else {
                // 内容耗尽：淡出激活时保持末帧内容按淡出增益继续衰减（E
                // 处及其后增益从 ~1 平滑走到 0），杜绝“增益还很大时内容
                // 硬切”的 Click。淡出未激活时维持越界静音语义。
                if clip.fade_out_frames > 0
                    && local >= fade_zone_start
                    && local < clip.length_frames
                    && has_last
                {
                    let mixed_l = last_l * g;
                    let mixed_r = last_r * g;
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
                continue;
            };
            last_l = l;
            last_r = r;
            has_last = true;
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

/// 估算 clip 内容末端（输出帧域，第一个"无内容"帧；deep-stretch 等造成
/// 内容不足时 < clip.length_frames）。`None` = 内容不会在此 clip 内耗尽
/// （repeat / Loop 回绕 / 内容覆盖整条 clip）。
fn clip_content_end_frame(clip: &EngineClip) -> Option<u64> {
    if clip.repeat {
        return None;
    }
    if clip.loop_anchor_frame.is_some() {
        return None;
    }
    if let Some(pcm) = clip.rendered_pcm.as_ref() {
        // 合成（pitch edit）clip：渲染缓冲即内容源。
        return Some((pcm.len() / 2) as u64);
    }
    let window = clip.src_end_frame.saturating_sub(clip.src_start_frame);
    if window == 0 {
        return Some(0);
    }
    let rate = if clip.playback_rate.is_finite() && clip.playback_rate > 0.0 {
        clip.playback_rate
    } else {
        1.0
    };
    // 映射：src_frame = round((local + offset) * rate)，内容为 src 域 [0, window)。
    // 首个越界帧 ≈ window/rate - offset（±1 帧误差由主循环 hold 兜底）。
    // offset 为负（前导静音，snapshot 的既有约定）时内容末端应后移 |offset|。
    let end = ((window as f64) / rate).floor() as u64;
    let offset = clip.local_src_offset_frames;
    Some(if offset >= 0 {
        end.saturating_sub(offset as u64)
    } else {
        end.saturating_add(offset.unsigned_abs())
    })
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
    if scratch.len() == frames * 2 {
        scratch.fill(0.0);
    } else {
        scratch.clear();
        scratch.resize(frames * 2, 0.0);
    }

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
    play_start_wait: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter: &mut TrackMeterScratch,
    bus: &TrackMeterBus,
    metro: &MetronomeRt,
    metro_voices: &mut MetronomeVoices,
) -> Option<BlockRender> {
    if scratch.len() == frames * 2 {
        scratch.fill(0.0);
    } else {
        scratch.clear();
        scratch.resize(frames * 2, 0.0);
    }

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

    // ── 原地等待渲染（Case A + Case B 统一）──────────────────────────────
    // 引擎快照是唯一真相，本回调是其**纯函数**：
    //   - 窗口未就绪（覆盖该位置的 Clip 尚在渲染）→ 冻结：保持 is_playing、
    //     不推进位置、静音输出（pending 窗口的 scratch 已是全零），并丢弃
    //     xfade 来源，等待期间绝不播放编辑前的陈旧内容；
    //   - 窗口就绪 → 正常推进并出声。
    // 因此等待的解除无需任何主动信号：渲染线程把结果发布给引擎
    // （`RenderedClipsChanged`）→ worker 换入新快照 → 本判定在下一个音频块
    // 自动通过，播放随之开始/继续 —— 既等价于"关闭后台预渲染时先渲染完再
    // 起播"，又无需阻塞、无需用户手动重按播放。
    if !current_ready {
        play_start_wait.store(true, Ordering::Relaxed);
        transition.fade_from_snapshot = None;
        transition.fade_remaining_frames = 0;
        // stderr I/O must stay off the RT thread: report via the meter thread
        // (diagnostics only — the resume is driven by the render threads).
        bus.transport_wait_pos.store(pos0, Ordering::Relaxed);
        bus.pending_pos.store(pos0, Ordering::Relaxed);
        return Some(BlockRender { snapshot: snap });
    }
    play_start_wait.store(false, Ordering::Relaxed);

    // 到达此处 current_ready 必为 true（未就绪窗口已在上方冻结早退）。
    // 旧实现"fade-from 快照就绪而当前快照未就绪时播陈旧内容推进"的分支随
    // 无条件冻结的引入而不可达，已删除：等待期间绝不播放编辑前的陈旧内容。

    if let Some(from_snapshot) = transition.fade_from_snapshot.as_ref() {
        let from_ready =
            render_snapshot_window(frames, from_snapshot, pos0, pos1, scratch_fade_from, None);
        if from_ready && transition.fade_remaining_frames > 0 {
            // 直接就地混合，删掉极其耗时的 scratch.clone()
            blend_snapshot_windows_in_place(
                scratch.as_mut_slice(),
                scratch_fade_from.as_slice(),
                transition.fade_remaining_frames,
            );
            transition.fade_remaining_frames =
                transition.fade_remaining_frames.saturating_sub(frames);
        }
        if !from_ready || transition.fade_remaining_frames == 0 {
            // fade-from 未就绪（陈旧内容绝不参与出声）或交叉淡化已完成：
            // 结束过渡，后续块按当前快照推进。
            transition.fade_from_snapshot = None;
            transition.fade_remaining_frames = 0;
        }
    }

    // 节拍器与实际出声的块同步叠加：未播放 / 冻结等待的静音块已在上方早退，
    // 到达此处必然出声。
    metro_voices.mix(scratch, metro, pos0, pos1, snap.sample_rate);

    advance_playback_position(frames, is_playing, position_frames, duration_frames);
    Some(BlockRender { snapshot: snap })
}

pub(crate) fn render_callback_f32(
    data: &mut [f32],
    out_channels: usize,
    snapshot: &Arc<ArcSwap<EngineSnapshot>>,
    is_playing: &AtomicBool,
    play_start_wait: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter_scratch: &mut TrackMeterScratch,
    meter_bus: &TrackMeterBus,
    metro: &MetronomeRt,
    metro_voices: &mut MetronomeVoices,
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
        play_start_wait,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
        &mut *meter_scratch,
        meter_bus,
        metro,
        metro_voices,
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
    play_start_wait: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter_scratch: &mut TrackMeterScratch,
    meter_bus: &TrackMeterBus,
    metro: &MetronomeRt,
    metro_voices: &mut MetronomeVoices,
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
        play_start_wait,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
        &mut *meter_scratch,
        meter_bus,
        metro,
        metro_voices,
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
    play_start_wait: &AtomicBool,
    position_frames: &AtomicU64,
    duration_frames: &AtomicU64,
    scratch: &mut Vec<f32>,
    scratch_fade_from: &mut Vec<f32>,
    transition: &mut SnapshotTransitionState,
    meter_scratch: &mut TrackMeterScratch,
    meter_bus: &TrackMeterBus,
    metro: &MetronomeRt,
    metro_voices: &mut MetronomeVoices,
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
        // 与正常静音路径同一常量（32768）：panic 恢复路径若用 u16::MAX/2
        // （=32767）会在切换瞬间产生 1 LSB 的可闻阶跃。
        data.fill(U16_SILENCE);
        return;
    }

    let block = mix_into_scratch_stereo(
        frames,
        snapshot,
        is_playing,
        play_start_wait,
        position_frames,
        duration_frames,
        scratch,
        scratch_fade_from,
        transition,
        &mut *meter_scratch,
        meter_bus,
        metro,
        metro_voices,
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

    #[test]
    fn sample_clip_pcm_applies_take_channel_mode() {
        // 源 PCM：帧 i = [L=i/4, R=-i/4]（左右可分辨）。
        let mut pcm = Vec::new();
        for i in 0..4 {
            let f = i as f32;
            pcm.push(f / 4.0);
            pcm.push(-f / 4.0);
        }
        let mut clip = clip_with_dyn(None, None);
        clip.src = ResampledStereo {
            sample_rate: 44_100,
            frames: 4,
            pcm: Arc::new(pcm),
        };

        // Normal：原样。
        clip.channel_mode = crate::channel_mode::TakeChannelMode::Normal;
        let (l, r) = super::sample_clip_pcm(&clip, 2, 2.0).unwrap();
        assert!((l - 0.5).abs() < 1e-6 && (r - (-0.5)).abs() < 1e-6);

        // Swap：平面互换。
        clip.channel_mode = crate::channel_mode::TakeChannelMode::Swap;
        let (l, r) = super::sample_clip_pcm(&clip, 2, 2.0).unwrap();
        assert!((l - (-0.5)).abs() < 1e-6 && (r - 0.5).abs() < 1e-6);

        // MonoLeft：左平面复制。
        clip.channel_mode = crate::channel_mode::TakeChannelMode::MonoLeft;
        let (l, r) = super::sample_clip_pcm(&clip, 2, 2.0).unwrap();
        assert!((l - 0.5).abs() < 1e-6 && (r - 0.5).abs() < 1e-6);

        // MonoRight：右平面复制。
        clip.channel_mode = crate::channel_mode::TakeChannelMode::MonoRight;
        let (l, r) = super::sample_clip_pcm(&clip, 2, 2.0).unwrap();
        assert!((l - (-0.5)).abs() < 1e-6 && (r - (-0.5)).abs() < 1e-6);

        // MonoMix：两平面均值。
        clip.channel_mode = crate::channel_mode::TakeChannelMode::MonoMix;
        let (l, r) = super::sample_clip_pcm(&clip, 2, 2.0).unwrap();
        assert!(l.abs() < 1e-6 && r.abs() < 1e-6);
    }

    fn clip_with_curves(volume_curve: Option<Vec<f32>>) -> EngineClip {
        let mut clip = clip_with_dyn(None, None);
        clip.volume_curve = volume_curve.map(Arc::new);
        clip
    }

    /// 构造一个只关心动态曲线的 clip（其余曲线全为 None）。
    fn clip_with_dyn(dyn_curve: Option<Vec<f32>>, dyn_orig: Option<Vec<f32>>) -> EngineClip {
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
        channel_mode: crate::channel_mode::TakeChannelMode::Normal,
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
            volume_curve: None,
            volume_curve_frame_period_ms: 5.0,
            pan_curve: None,
            pan_curve_frame_period_ms: 5.0,
            dyn_curve: dyn_curve.map(Arc::new),
            dyn_orig_curve: dyn_orig.map(Arc::new),
            dyn_curve_frame_period_ms: 5.0,
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
        let at =
            |abs_frame: u64| sample_automation_curve(Some(&curve), abs_frame, 44_100, 5.0, 1.0);
        assert!(at(0) < 1e-6, "abs 0s reads curve frame 0");
        // 1ms = 44.1 样本 → 曲线帧 0.2
        assert!((at(44) - 0.2).abs() < 0.05, "got {}", at(44));
        // 5ms = 220.5 样本 → 曲线帧 1
        assert!((at(220) - 1.0).abs() < 0.05, "got {}", at(220));
        // 曲线末尾之后保持末值（不回落到 default）
        assert!((at(44_100) - 4.0).abs() < 1e-6, "got {}", at(44_100));
    }

    #[test]
    fn dyn_gain_is_target_over_original() {
        // 目标 0.5 / 原声 1.0 → 增益 0.5。
        let clip = clip_with_dyn(Some(vec![0.5f32]), Some(vec![1.0f32]));
        let (l, r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 0.5).abs() < 1e-6, "left got {l}");
        assert!((r - 0.5).abs() < 1e-6, "right got {r}");
    }

    #[test]
    fn dyn_gain_boosts_quiet_frames() {
        // 原声只有 0.25、目标 1.0 → +12 dB（×4），这是"抬安静段"的核心场景。
        let clip = clip_with_dyn(Some(vec![1.0f32]), Some(vec![0.25f32]));
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 4.0).abs() < 1e-6, "left got {l}");
    }

    /// ★ 回归：真实素材的安静段必须被提升到**用户画的目标**（门限曾是 −26 dBFS、
    /// 上限曾仅 ×4，两者叠加使安静段"怎么编辑都提不上去"）。
    #[test]
    fn dyn_gain_boosts_very_quiet_content() {
        // −40 dBFS（0.01）画目标 −4.7 dBFS（0.582）→ 需 ×58.2，必须精确兑现。
        let clip = clip_with_dyn(Some(vec![0.582f32]), Some(vec![0.01f32]));
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!(
            (l - 58.2).abs() < 0.1,
            "安静段必须被精确提升到目标（×58.2），实测 {l}"
        );

        // −55 dBFS（0.00178）→ 需 ×327，同样必须兑现（上限只是数值兜底）。
        let clip = clip_with_dyn(Some(vec![0.582f32]), Some(vec![0.00178f32]));
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 327.0).abs() < 1.0, "实测 {l}");

        // 上限的确切位置：从下限兑现到值域顶端。
        assert_eq!(
            crate::renderer::common_params::DYN_MAX_GAIN,
            1.0 / crate::renderer::common_params::DYN_SILENCE_FLOOR
        );
    }

    #[test]
    fn dyn_follow_orig_sentinel_is_unity() {
        // 哨兵（负值）= 沿用原声 → 增益 1.0，即便原声很小。
        let clip = clip_with_dyn(
            Some(vec![crate::renderer::common_params::DYN_FOLLOW_ORIG]),
            Some(vec![0.0001f32]),
        );
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-6, "left got {l}");
    }

    /// ★ 相邻点交界咔哒声回归（三类不连续一起钉住）。
    ///
    /// 这是最常见的编辑形态（在一条连续曲线的**中途**画一段），因此交界在整条
    /// 轨道上成对出现。引擎逐 PCM 样本在相邻帧之间线性插值，而存储形态的动态
    /// 曲线对未画帧存的是 `DYN_FOLLOW_ORIG`（−1）哨兵 —— 一个**非数值**。把它
    /// 交给插值会制造三类不连续（详见 `resolve_dyn_curves_for_audio`）：
    ///
    /// ① 穿过 0 → 掉到静音（0 = 画静音）；② 分子/分母下钳速率不一致 → 先冲高
    /// 再塌陷；③ 曲线末尾之外回落哨兵 → 增益硬跳（用户报告的"前帧已编辑、后帧
    /// 未编辑"就是它：动态数组的长度 = 最后写入帧 + 1，笔画结尾的下一帧即数组
    /// 之外）。
    ///
    /// 未解析的曲线必须出现①的跌落（解析存在的理由）；解析后必须：不跌落、
    /// 过交界**单调**（不得冲高再塌陷）、且正常基线下相邻样本变化极小。
    #[test]
    fn dyn_sentinel_boundary_must_stay_continuous() {
        use crate::renderer::common_params::{resolve_dyn_curves_for_audio, DYN_FOLLOW_ORIG};

        // 44.1 kHz + 5 ms 帧周期 ⇒ 一帧 ≈ 220.5 个样本。
        let last_sample = 662u64;
        let gain_at = |clip: &EngineClip, frame: u64| apply_mix_automation(clip, frame, 1.0, 1.0).0;
        let gains_of = |stored: &[f32], baseline: &[f32]| -> Vec<f32> {
            let r = resolve_dyn_curves_for_audio(stored, Some(baseline));
            let clip = clip_with_dyn(Some(r.target), Some(r.baseline));
            (0..=last_sample).map(|f| gain_at(&clip, f)).collect()
        };

        // ① 未解析（存储形态直接交给引擎）：出现"掉到静音"的跌落。
        let raw = clip_with_dyn(Some(vec![0.5, DYN_FOLLOW_ORIG]), Some(vec![0.1, 0.1]));
        let raw_min = (0..=last_sample)
            .map(|f| gain_at(&raw, f))
            .fold(f32::INFINITY, f32::min);
        assert!(
            raw_min < 0.05,
            "未解析的哨兵曲线在交界处本应出现掉音跌落（这正是咔哒声的根因），实测最小增益 {raw_min}"
        );

        // ② 解析后，交界两侧都必须满足"不跌落 + 单调"。
        // 覆盖：编辑→未编辑（正常基线）、编辑→未编辑（后帧静音）、
        //       编辑延续到曲线末尾之外（③的场景）。
        let cases: Vec<(&str, Vec<f32>, Vec<f32>, bool)> = vec![
            (
                "编辑→未编辑（正常基线）",
                vec![DYN_FOLLOW_ORIG, 0.5, DYN_FOLLOW_ORIG],
                vec![0.1, 0.1, 0.1],
                true,
            ),
            (
                "编辑→未编辑（后帧静音）",
                vec![DYN_FOLLOW_ORIG, 0.5, DYN_FOLLOW_ORIG],
                vec![0.1, 0.1, 0.0],
                false,
            ),
            (
                "编辑延到曲线末尾之外",
                vec![DYN_FOLLOW_ORIG, 0.5],
                vec![0.1, 0.1],
                true,
            ),
        ];
        for (label, stored, baseline, expect_smooth) in cases {
            let gains = gains_of(&stored, &baseline);
            assert!(
                (gains[0] - 1.0).abs() < 1e-6,
                "{label}：未画帧增益仍应为 1，实测 {}",
                gains[0]
            );
            let min_gain = gains.iter().cloned().fold(f32::INFINITY, f32::min);
            assert!(min_gain > 0.5, "{label}：不得跌落，实测最小增益 {min_gain}");
            // 编辑收尾段不得"冲高再塌陷"（那是分子/分母下钳折点造成的：
            // 修复前该段的增益会先多涨 ≈1.0 再在格末崩到 1）。
            let tail: Vec<f32> = gains[220..=441].to_vec();
            let tail_max = tail.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                tail_max <= tail[0] + 0.1,
                "{label}：收尾不得冲高，起点 {} 峰值 {tail_max}",
                tail[0]
            );
            assert!(
                (tail[tail.len() - 1] - 1.0).abs() < 1e-3,
                "{label}：收尾应回到未画帧的增益 1，实测 {}",
                tail[tail.len() - 1]
            );
            if expect_smooth {
                // 基线正常时，相邻样本的增益变化必须极小（平滑斜坡）。
                let max_jump = gains
                    .windows(2)
                    .map(|p| (p[1] - p[0]).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_jump < 0.05,
                    "{label}：解析后相邻样本增益变化应极小，实测 {max_jump}"
                );
            }
        }

        // ③ 末尾之外：留的那一格让"编辑收尾"有一格过渡，增益因此不再硬跳。
        let with_tail = gains_of(&[DYN_FOLLOW_ORIG, 0.5], &[0.1, 0.1]);
        let jumps_after_end: f32 = with_tail[441..]
            .windows(2)
            .map(|p| (p[1] - p[0]).abs())
            .fold(0.0, f32::max);
        assert!(
            jumps_after_end < 0.05,
            "曲线末尾之外不得出现增益硬跳，实测 {jumps_after_end}"
        );
    }

    #[test]
    fn dyn_missing_orig_curve_is_unity() {
        // 分析未就绪（无基线）时不得放大：增益必须是 1.0，而不是 目标/兜底。
        let clip = clip_with_dyn(Some(vec![1.0f32]), None);
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-6, "left got {l}");
    }

    #[test]
    fn dyn_gain_bounds_no_content_frames_instead_of_rejecting() {
        // 无内容帧（−80 dBFS，抖动噪声量级）：增益有界（分母钳到下限），
        // 不会无限放大 —— 但也不再是"拒绝放大返回 1"（那会造成门限处阶跃，
        // 即近零处的随机伪影，见 DYN_SILENCE_FLOOR 的说明）。
        let clip = clip_with_dyn(Some(vec![1.0f32]), Some(vec![0.0001f32]));
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!(
            (l - crate::renderer::common_params::DYN_MAX_GAIN).abs() < 1e-3,
            "无内容帧的增益应恰好钳到上限，实测 {l}"
        );
    }

    #[test]
    fn dyn_and_volume_multiply() {
        // 两个参数是独立乘性增益：0.5（音量）× 2.0（动态）= 1.0。
        let mut clip = clip_with_dyn(Some(vec![1.0f32]), Some(vec![0.5f32]));
        clip.volume_curve = Some(Arc::new(vec![0.5f32]));
        clip.volume_curve_frame_period_ms = 5.0;
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-6, "left got {l}");
    }

    /// 末点污染回归：曲线越界后必须回落到"沿用原声"，而不是持有末值。
    ///
    /// 用户只在开头画了一个目标电平 2.0，后续所有音频都不该被它影响。
    /// 若采样器持有末值，越界帧会持续算 `2.0 / dyn_orig(帧)`，
    /// 把整段后续音频强行拉到同一个目标电平 —— 这是历史上共振峰参数
    /// 出现过的同类 bug（见 `renderer/chain.rs::sample_curve_at_abs_sec`）。
    #[test]
    fn dyn_curve_beyond_last_point_does_not_pollute() {
        // 曲线只有 1 帧：第 0 帧目标 2.0。
        let clip = clip_with_dyn(Some(vec![2.0f32]), Some(vec![1.0f32]));
        // 第 0 帧（曲线内）：2.0 / 1.0 = ×2。
        let (l, _r) = apply_mix_automation(&clip, 0, 1.0, 1.0);
        assert!((l - 2.0).abs() < 1e-6, "曲线内应生效，got {l}");
        // 越界帧：曲线只有 1 帧 = 5ms = 220.5 采样 @44.1k，
        // 因此 >221 帧才算真正越界。
        for frame in [221u64, 300, 1_000, 44_100] {
            let (l, _r) = apply_mix_automation(&clip, frame, 1.0, 1.0);
            assert!((l - 1.0).abs() < 1e-6, "帧 {frame} 被末点污染，got {l}");
        }
    }

    /// 末元素自身的保持区间仍应生效：`[len-1, len)` 内取末值。
    ///
    /// 与共振峰修复的边界约定一致（chain.rs 的测试同样断言了这一点）：
    /// 越界判定是 `idx >= len`，不是 `idx > len-1`。
    #[test]
    fn dyn_curve_last_element_holds_within_its_own_frame() {
        // 曲线 4 帧，末值 2.0；帧周期 5ms → 末点覆盖 [15ms, 20ms)。
        let clip = clip_with_dyn(Some(vec![1.0f32, 1.0, 1.0, 2.0]), Some(vec![1.0f32; 4]));
        let sr = clip.src.sample_rate;
        // 末点处（含插值邻域）应接近 2.0：曲线是 [1,1,1,2]，故 idx 3.0 处
        // 恰为 2.0，而 idx 2.5~3.0 之间是 1→2 的插值过渡。
        let (l, _r) = apply_mix_automation(&clip, (0.015 * sr as f64) as u64, 1.0, 1.0);
        assert!((l - 2.0).abs() < 0.05, "末点处应生效，got {l}");
        // idx 3.5 → 17.5ms（保持区间内，无下一元素可插值 → 恒为末值 2.0）
        let (l, _r) = apply_mix_automation(&clip, (0.0175 * sr as f64) as u64, 1.0, 1.0);
        assert!((l - 2.0).abs() < 1e-6, "末点保持区间内应生效，got {l}");
        // idx 4.5 → 22.5ms（越界）→ no-op
        let (l, _r) = apply_mix_automation(&clip, (0.0225 * sr as f64) as u64, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-6, "越界后不得污染，got {l}");
    }

}
