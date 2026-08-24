use crate::state::{TimelineState, Track};
use crate::time_stretch::{time_stretch_interleaved, StretchAlgorithm};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

// ─── 导出格式与质量预设 ────────────────────────────────────────────────────────

/// 导出音频格式（位深）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    /// 16-bit 整型（默认，向后兼容，用于实时预览）。
    #[default]
    Wav16,
    /// 24-bit 整型（高质量存档）。
    #[allow(dead_code)]
    Wav24,
    /// 32-bit 浮点（最高质量，用于最终导出）。
    Wav32f,
}

/// 质量预设，区分实时预览和最终导出场景。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityPreset {
    /// 快速模式，用于播放预览（默认）。
    #[default]
    Realtime,
    /// 最高质量模式，用于最终导出。
    Export,
}

#[derive(Debug, Clone)]
pub struct MixdownOptions {
    pub sample_rate: u32,
    pub start_sec: f64,
    pub end_sec: Option<f64>,
    pub stretch: StretchAlgorithm,
    pub apply_pitch_edit: bool,
    /// 导出格式（位深），默认 [`ExportFormat::Wav16`]。
    pub export_format: ExportFormat,
    /// 质量预设，默认 [`QualityPreset::Realtime`]。
    #[allow(dead_code)]
    pub quality_preset: QualityPreset,
    /// 可选取消标记：为 true 时中断渲染并返回 `export_cancelled`。
    pub cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
}

#[derive(Debug, Clone)]
pub struct MixdownResult {
    pub sample_rate: u32,
    pub duration_sec: f64,
}

fn mixdown_cancelled(opts: &MixdownOptions) -> bool {
    opts.cancel_flag
        .as_ref()
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

#[allow(dead_code)]
fn beat_sec(bpm: f64) -> f64 {
    60.0 / bpm.max(1e-6)
}

fn clamp_track_volume(x: f32) -> f32 {
    x.clamp(0.0, 4.0)
}

fn clamp11(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// 在 mixdown 中采样自动化曲线（与 mix.rs 中的 sample_automation_curve 逻辑一致）。
fn sample_automation_curve_at_sec(
    curve: Option<&[f32]>,
    abs_sec: f64,
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
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return default_value;
    }
    let i0 = (idx_f as usize).min(curve.len().saturating_sub(1));
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64) as f32; // fraction 在 [0, 1) 内，无需 clamp
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

pub(crate) fn linear_resample_interleaved(
    input: &[f32],
    channels: usize,
    in_rate: u32,
    out_rate: u32,
) -> Vec<f32> {
    if input.is_empty() || channels == 0 {
        return vec![];
    }
    if in_rate == out_rate {
        return input.to_vec();
    }

    let in_frames = input.len() / channels;
    if in_frames < 2 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_frames = ((in_frames as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_frames * channels];

    for of in 0..out_frames {
        let t_in = (of as f64) / ratio;
        let mut i0 = t_in as usize; //  向下取整
        let frac = (t_in - (i0 as f64)) as f32;
        i0 = i0.min(in_frames - 1); //  限制上限即可
        let i1 = (i0 + 1).min(in_frames - 1);

        // 提取乘法基址到声道循环外部
        let base0 = i0 * channels;
        let base1 = i1 * channels;
        let out_base = of * channels;

        for ch in 0..channels {
            let a = input[base0 + ch];
            let b = input[base1 + ch];
            out[out_base + ch] = a + (b - a) * frac;
        }
    }

    out
}

pub(crate) fn reverse_interleaved_frames(samples: &mut [f32], channels: usize) {
    if channels == 0 {
        return;
    }
    let frames = samples.len() / channels;
    for i in 0..(frames / 2) {
        let li = i * channels;
        let ri = (frames - 1 - i) * channels;
        for ch in 0..channels {
            samples.swap(li + ch, ri + ch);
        }
    }
}

/// Loop（循环源）：从完整媒体 PCM 构建按**整个文件**模运算回绕的片段。
///
/// 映射（f 为片段内已消费的源帧序号）：
///   正放 idx(f) = floor_mod(anchor + f, total)
///   倒放 idx(f) = floor_mod(anchor − 1 − f, total)
/// 其中正放锚点 = `round(source_start·in_rate)`、倒放锚点 =
/// `round(source_end·in_rate)`（exclusive 末端）。越过文件边界后环绕到
/// 另一侧继续 —— 即"循环原始音频文件"，与 REAPER Loop source 一致。
/// 输出保持自然时间顺序（倒放的方向已体现在索引递减中，
/// 调用方无需再做整体反转）。
///
/// 实现：在回绕点之间源索引是连续的，因此按"整段拷贝到边界"的方式用
/// `extend_from_slice` 分块复制，而不是逐帧取模 + 逐样本 push —— 长输出
/// （循环多个周期）下每帧成本从"取模+分支+逐样本写"降为 memcpy 级别。
pub(crate) fn build_loop_tiled_segment(
    pcm: &[f32],
    channels: usize,
    anchor_frame_exclusive: i64,
    reversed: bool,
    out_source_frames: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    if channels == 0 || pcm.is_empty() {
        return out;
    }
    let total = (pcm.len() / channels) as i64;
    if total <= 0 {
        return out;
    }
    out.reserve(out_source_frames.saturating_mul(channels));
    // 起始索引（首帧实际写入的源帧号，已归一化进 [0, total)）：
    let mut idx = if reversed {
        (anchor_frame_exclusive - 1).rem_euclid(total)
    } else {
        anchor_frame_exclusive.rem_euclid(total)
    };
    let mut remaining = out_source_frames as i64;
    while remaining > 0 {
        // 本轮可连续拷贝的帧数：到达文件边界的距离 与 剩余需求 取小。
        // 正放连续区间向上：[idx, idx+run)；倒放连续区间向下：
        // [idx−run+1, idx]（随后帧序就地反转，通道内样本保持配对）。
        let run = if reversed {
            (idx + 1).min(remaining)
        } else {
            (total - idx).min(remaining)
        };
        let (base, end_base) = if reversed {
            let start_frame = (idx + 1 - run) as usize;
            (start_frame * channels, (idx as usize + 1) * channels)
        } else {
            ((idx as usize) * channels, (idx as usize + run as usize) * channels)
        };
        out.extend_from_slice(&pcm[base..end_base]);
        if reversed {
            // 就地反转刚追加的 run 个帧的帧序（每帧 channels 个样本整体交换）。
            let len = out.len();
            let block = &mut out[len - (run as usize) * channels ..];
            let half = run as usize / 2;
            for f in 0..half {
                let a = f * channels;
                let b = (run as usize - 1 - f) * channels;
                for c in 0..channels {
                    block.swap(a + c, b + c);
                }
            }
        }
        remaining -= run;
        // 推进索引并环绕（正放越过末尾回到 0；倒放越过 0 回到末尾）。
        idx = if reversed {
            (idx - run).rem_euclid(total)
        } else {
            (idx + run) % total
        };
    }
    out
}

fn build_parent_map(tracks: &[Track]) -> HashMap<String, Option<String>> {
    let mut map = HashMap::new();
    for t in tracks {
        map.insert(t.id.clone(), t.parent_id.clone());
    }
    map
}

fn track_lineage(track_id: &str, parent_map: &HashMap<String, Option<String>>) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = Some(track_id.to_string());
    let mut safety = 0;
    while let Some(id) = cur {
        out.push(id.clone());
        cur = parent_map.get(&id).and_then(|p| p.clone());
        safety += 1;
        if safety > 2048 {
            break;
        }
    }
    out
}

fn compute_track_gains(tracks: &[Track]) -> HashMap<String, (f32, bool, bool)> {
    let parent_map = build_parent_map(tracks);
    let by_id: HashMap<&str, &Track> = tracks.iter().map(|t| (t.id.as_str(), t)).collect();

    let any_solo = tracks.iter().any(|t| t.solo);
    let mut out = HashMap::new();

    for t in tracks {
        let lineage = track_lineage(&t.id, &parent_map);

        let mut gain = 1.0f32;
        let mut muted = false;
        let mut soloed = false;
        for id in &lineage {
            if let Some(node) = by_id.get(id.as_str()) {
                gain *= clamp_track_volume(node.volume);
                muted |= node.muted;
                soloed |= node.solo;
            }
        }

        // Solo overrides mute: when a track (or its ancestor) is soloed,
        // its own mute flag is ignored so that solo always wins.
        let effective_muted = if any_solo && soloed { false } else { muted };

        if any_solo {
            out.insert(t.id.clone(), (gain, effective_muted, soloed));
        } else {
            out.insert(t.id.clone(), (gain, effective_muted, true));
        }
    }

    out
}

pub(crate) fn clip_duration_sec_from_wav(
    sample_rate: u32,
    channels: u16,
    pcm: &[f32],
) -> Option<f64> {
    let ch = channels as usize;
    if sample_rate == 0 || ch == 0 {
        return None;
    }
    let frames = pcm.len() / ch;
    if frames == 0 {
        return None;
    }
    Some(frames as f64 / sample_rate as f64)
}

pub fn render_mixdown_wav(
    timeline: &TimelineState,
    output_path: &Path,
    opts: MixdownOptions,
) -> Result<MixdownResult, String> {
    if mixdown_cancelled(&opts) {
        return Err("export_cancelled".to_string());
    }

    let (out_rate, out_channels, duration_sec, mix) =
        render_mixdown_interleaved(timeline, opts.clone())?;

    if mixdown_cancelled(&opts) {
        return Err("export_cancelled".to_string());
    }

    // 根据 export_format 选择 WavSpec。
    let spec = match opts.export_format {
        ExportFormat::Wav16 => WavSpec {
            channels: out_channels,
            sample_rate: out_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
        ExportFormat::Wav24 => WavSpec {
            channels: out_channels,
            sample_rate: out_rate,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        },
        ExportFormat::Wav32f => WavSpec {
            channels: out_channels,
            sample_rate: out_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    };
    let mut writer = WavWriter::create(output_path, spec).map_err(|e| e.to_string())?;

    match opts.export_format {
        ExportFormat::Wav16 => {
            for (idx, s) in mix.into_iter().enumerate() {
                if idx % 8192 == 0 && mixdown_cancelled(&opts) {
                    drop(writer);
                    let _ = std::fs::remove_file(output_path);
                    return Err("export_cancelled".to_string());
                }
                let v = clamp11(s);
                let i = (v * i16::MAX as f32) as i16;
                writer.write_sample(i).map_err(|e| e.to_string())?;
            }
        }
        ExportFormat::Wav24 => {
            // hound 的 24-bit int 写入使用 i32，有效范围 [-8388608, 8388607]。
            const MAX24: f32 = 8_388_607.0;
            for (idx, s) in mix.into_iter().enumerate() {
                if idx % 8192 == 0 && mixdown_cancelled(&opts) {
                    drop(writer);
                    let _ = std::fs::remove_file(output_path);
                    return Err("export_cancelled".to_string());
                }
                let v = clamp11(s);
                let i = (v * MAX24) as i32;
                writer.write_sample(i).map_err(|e| e.to_string())?;
            }
        }
        ExportFormat::Wav32f => {
            for (idx, s) in mix.into_iter().enumerate() {
                if idx % 8192 == 0 && mixdown_cancelled(&opts) {
                    drop(writer);
                    let _ = std::fs::remove_file(output_path);
                    return Err("export_cancelled".to_string());
                }
                writer.write_sample(s).map_err(|e| e.to_string())?;
            }
        }
    }
    writer.finalize().map_err(|e| e.to_string())?;

    Ok(MixdownResult {
        sample_rate: out_rate,
        duration_sec,
    })
}

pub fn render_mixdown_interleaved(
    timeline: &TimelineState,
    opts: MixdownOptions,
) -> Result<(u32, u16, f64, Vec<f32>), String> {
    if mixdown_cancelled(&opts) {
        return Err("export_cancelled".to_string());
    }

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");

    let mut clips_considered: u32 = 0;
    let mut clips_decoded: u32 = 0;
    let mut clips_mixed: u32 = 0;

    let bpm = timeline.bpm;
    if !(bpm.is_finite() && bpm > 0.0) {
        return Err("invalid bpm".to_string());
    }

    let out_rate = opts.sample_rate.max(8000);
    let out_channels: u16 = 2;

    let project_sec = timeline.project_sec.max(0.0);
    let start_sec = opts.start_sec.max(0.0);
    let end_sec = opts.end_sec.unwrap_or(project_sec).max(start_sec);
    let duration_sec = (end_sec - start_sec).max(0.0);
    let out_frames = (duration_sec * out_rate as f64).round().max(1.0) as usize;
    let mut mix = vec![0.0f32; out_frames * out_channels as usize];

    let track_gain = compute_track_gains(&timeline.tracks);

    // Precompute audible tracks set.
    let mut audible_tracks: HashSet<String> = HashSet::new();
    for (tid, (_gain, muted, solo_ok)) in &track_gain {
        if !*muted && *solo_ok {
            audible_tracks.insert(tid.clone());
        }
    }

    for clip in &timeline.clips {
        if mixdown_cancelled(&opts) {
            return Err("export_cancelled".to_string());
        }

        if clip.muted {
            continue;
        }
        if !audible_tracks.contains(&clip.track_id) {
            continue;
        }
        let Some(source_path) = clip.source_path.as_ref() else {
            continue;
        };

        clips_considered = clips_considered.saturating_add(1);

        let (track_gain_value, _tmuted, _solo_ok) = track_gain
            .get(&clip.track_id)
            .cloned()
            .unwrap_or((1.0, false, true));
        let gain = (clip.gain.max(0.0) * track_gain_value).clamp(0.0, 4.0);
        if gain <= 0.0 {
            continue;
        }

        // Timeline placement.
        let clip_start_sec = clip.start_sec.max(0.0);
        let clip_timeline_len_sec = clip.length_sec.max(0.0);
        if !(clip_timeline_len_sec.is_finite() && clip_timeline_len_sec > 0.0) {
            continue;
        }
        let clip_end_sec = clip_start_sec + clip_timeline_len_sec;

        // Check overlap with requested render window.
        if clip_end_sec <= start_sec || clip_start_sec >= end_sec {
            continue;
        }

        let playback_rate = clip.playback_rate as f64;
        let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate
        } else {
            1.0
        };

        // Decode audio (WAV fast-path; otherwise Symphonia).
        let (in_rate, in_channels, pcm) =
            match crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path)) {
                Ok(v) => v,
                Err(e) => {
                    if debug {
                        eprintln!(
                            "mixdown: decode failed; clip_id={} track_id={} path={} err={}",
                            clip.id, clip.track_id, source_path, e
                        );
                    }
                    continue;
                }
            };

        clips_decoded = clips_decoded.saturating_add(1);

        let in_channels_usize = in_channels as usize;
        let in_frames = pcm.len() / in_channels_usize;
        if in_frames < 2 {
            continue;
        }

        // Source trimming is expressed in source-domain absolute seconds.
        // 非 Loop 统一使用**消费窗口模型**（clip_playback_window_sec）：
        //   正放 win = [ss, ss+len·r)；倒放 win = [se−len·r, se)。
        // win ∉ [0, D) 的部分渲染静音：正放 ss<0 / 倒放 se>D → 前导静音
        //（方向不同！倒放的 ss<0 是尾部静音，切片自然变短即可，绝不能
        // 再触发前导静音 —— 否则内容整体后移、该有声处被静音吞掉）。
        let loop_mode = clip.loop_enabled;

        let total_sec = match clip_duration_sec_from_wav(in_rate, in_channels, &pcm) {
            Some(v) => v,
            None => continue,
        };
        if !(total_sec.is_finite() && total_sec > 0.0) {
            continue;
        }

        let (win_start_sec, win_end_sec) = crate::state::clip_playback_window_sec(clip);
        let pre_silence_sec =
            crate::state::clip_leading_silence_sec(clip, Some(total_sec)) / playback_rate.max(1e-6);

        let src_end_limit_sec = win_end_sec.min(total_sec).max(win_start_sec.max(0.0));
        let slice_start_sec = win_start_sec.max(0.0);
        if !loop_mode && src_end_limit_sec - slice_start_sec <= 1e-9 {
            continue;
        }

        // ── 片段构建 ─────────────────────────────────────────────────────────
        // Loop（循环源）：从完整媒体按整文件模运算回绕生成片段
        //   正放 idx(f) = floor_mod(source_start + f, D_frames)
        //   倒放 idx(f) = floor_mod(source_end − 1 − f, D_frames)
        // 即"循环原始音频文件"：先消费 source_start → 文件末尾，
        // 之后每个周期都是整个文件（对齐 REAPER Loop source）。
        // 锚点直接取原始字段：正放可为负（floor_mod 环绕到末尾一侧）；
        // 倒放只把末端 clamp 到媒体时长 —— 不能用含 `.max(source_start)`
        // 的 src_end_limit_sec（那是为非 Loop 切片准备的），否则 Loop 下
        // split 产生的"环绕窗口"会把倒放锚点错误地推回窗口起点。
        // 非 Loop 保持原窗口切片行为。
        let anchor_frame: i64 = if clip.reversed {
            (clip.source_end_sec.min(total_sec) * in_rate as f64).round() as i64
        } else {
            (clip.source_start_sec * in_rate as f64).round() as i64
        };
        // Loop（循环源）：只物化【导出窗口 ∩ clip】对应的消费量 —— 整条 clip
        // 的平铺段在"导出局部区间 / 长循环 clip"场景会产生多份全尺寸缓冲的
        // 瞬时峰值（tiled 段 + resample 副本 + formant 产物），分配失败即
        // 进程 abort。锚点按窗口起点前移等量消费帧，内容相位不变。
        //
        // 窗口起点/终点量化到固定网格（1s）：波形 peaks 与区间导出以任意
        // 浮点窗口反复调用本函数，若直接使用原始窗口，Loop+Formant 的缓存
        // key 会随每次滚动/缩放变化 → 全量 Formant DSP 重算并冲刷 LRU。
        // 量化后滑动窗口只命中小集合 key；多消费的边界帧由下方
        // 【导出窗口 ∩ clip】交集裁掉，不影响输出内容与淡化相位。
        const LOOP_SEG_QUANTUM_SEC: f64 = 1.0;
        let (loop_seg_local_start_sec, loop_seg_len_sec) = if loop_mode {
            let local_start = (start_sec - clip_start_sec).max(0.0);
            let local_end = (end_sec - clip_start_sec).min(clip_timeline_len_sec);
            let q_start =
                (local_start - local_start % LOOP_SEG_QUANTUM_SEC).max(0.0);
            let q_end = ((local_end / LOOP_SEG_QUANTUM_SEC).ceil() * LOOP_SEG_QUANTUM_SEC)
                .min(clip_timeline_len_sec.max(0.0));
            (q_start, (q_end - q_start).max(0.0))
        } else {
            (0.0, clip_timeline_len_sec)
        };
        // Loop（循环源）平铺段几何 —— 只计算一次，片段构建与 Formant 缓存键
        // 必须共享同一组数值（此前两处各算一遍，一旦某处改动就会静默漂移：
        // 键与内容不再对应，缓存互相投毒/永不命中）。
        let (loop_advanced_anchor, loop_out_source_frames) = if loop_mode {
            let skip_src_frames =
                (loop_seg_local_start_sec * playback_rate * in_rate as f64).round() as i64;
            let advanced_anchor = if clip.reversed {
                anchor_frame - skip_src_frames
            } else {
                anchor_frame + skip_src_frames
            };
            let out_source_frames = ((loop_seg_len_sec.max(0.0)
                * playback_rate
                * in_rate as f64)
                .ceil()
                .max(2.0)) as usize;
            (advanced_anchor, out_source_frames)
        } else {
            (anchor_frame, 0usize)
        };
        let segment: Vec<f32> = if loop_mode {
            build_loop_tiled_segment(
                &pcm,
                in_channels_usize,
                loop_advanced_anchor,
                clip.reversed,
                loop_out_source_frames,
            )
        } else {
            // 非 Loop：按消费窗口切片（正放 [ss, ss+len·r)、倒放
            // [se−len·r, se)，均 clamp 到媒体内；域外部分由前导/尾部静音表达）。
            let src_i0 = (slice_start_sec * in_rate as f64).floor().max(0.0) as usize;
            let src_i1 = (src_end_limit_sec * in_rate as f64)
                .ceil()
                .max(src_i0 as f64) as usize;
            let src_i1 = src_i1.min(in_frames);
            if src_i1 <= src_i0 + 1 {
                continue;
            }
            pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)].to_vec()
        };

        let mut segment =
            linear_resample_interleaved(&segment, in_channels_usize, in_rate, out_rate);

        // Loop 模式的倒放方向已由回绕索引体现，不再整体反转。
        if !loop_mode && clip.reversed {
            reverse_interleaved_frames(&mut segment, in_channels_usize);
        }

        // Convert to stereo if needed.
        let segment = if in_channels == 1 {
            let frames = segment.len();
            let mut stereo = Vec::with_capacity(frames * 2);
            for s in segment {
                stereo.push(s);
                stereo.push(s);
            }
            stereo
        } else if in_channels >= 2 {
            // Use first two channels.
            let frames = segment.len() / in_channels_usize;
            let mut stereo = Vec::with_capacity(frames * 2);
            for f in 0..frames {
                stereo.push(segment[f * in_channels_usize]);
                stereo.push(segment[f * in_channels_usize + 1]);
            }
            stereo
        } else {
            continue;
        };
        let mut segment = segment;

        if let Some(params) = clip.formant_morph.as_ref().filter(|params| params.enabled) {
            // Loop（循环源）键必须编码**实际消费的平铺区间**（锚点推进量 + 消费
            // 帧数）：平铺段内容随导出窗口 [start_sec, end_sec] 变化，若键固定取
            // [0, total_sec]，不同导出窗口会命中同一条目 —— 先渲染的一方把错误
            // 长度/内容的结果投毒给另一方（get_or_compute 不做长度校验）。
            // 用"归一化锚点帧 + 消费帧数"（换算为秒）唯一确定 segment 内容。
            let (key_start_sec, key_end_sec) = if loop_mode {
                // 与上方片段构建共享同一组几何数值（loop_advanced_anchor /
                // loop_out_source_frames），键与 segment 内容严格对应。
                let start_frame = loop_advanced_anchor.rem_euclid(
                    ((total_sec * in_rate as f64).round() as i64).max(1),
                );
                (
                    start_frame as f64 / in_rate as f64,
                    (start_frame + loop_out_source_frames as i64) as f64 / in_rate as f64,
                )
            } else {
                // 非 Loop：键编码实际消费窗口（正放/倒放统一取自
                // clip_playback_window_sec，与 snapshot 实时域查找键成对）。
                (slice_start_sec, win_end_sec)
            };
            let key = crate::formant_cache::make_formant_cache_key(
                &clip.id,
                Path::new(source_path),
                out_rate,
                key_start_sec,
                key_end_sec,
                clip.reversed && !loop_mode,
                // 离线 Loop 的处理对象是"回绕平铺 segment"（锚点起、长度为
                // clip 消费量），与实时域的完整文件自然顺序内容不同 —— 必须
                // 用 tiled_wrap 域判别隔离，避免两个域互相毒化缓存。
                loop_mode,
                params,
            );
            match crate::formant_cache::get_or_compute_formant_audio(key, &segment, out_rate, params)
            {
                Ok(entry) => {
                    segment = entry.pcm_stereo.as_ref().clone();
                }
                Err(err) => {
                    if debug {
                        eprintln!(
                            "mixdown: formant morph failed; clip_id={} path={} err={}",
                            clip.id, source_path, err
                        );
                    }
                }
            }
        }

        // Pitch-preserving time-stretch:
        // - playback_rate == 1: keep source window duration as-is.
        // - playback_rate != 1: stretch the trimmed window to (src_len / playback_rate) in timeline time.
        // 若合成处理器声明自己处理时间拉伸（handles_time_stretch = true，如 vslib），
        // 则跳过此处外部拉伸，由 pitch edit 阶段的处理器内部完成。
        let processor_handles_stretch = timeline
            .resolve_root_track_id(&clip.track_id)
            .and_then(|root| timeline.tracks.iter().find(|t| t.id == root))
            .map(|t| {
                let kind = crate::state::SynthPipelineKind::from_track_algo(&t.pitch_analysis_algo);
                crate::renderer::processor_handles_time_stretch(kind, t.compose_enabled)
            })
            .unwrap_or(false);
        // 外部 SoundTouch 拉伸的执行条件：
        //   !processor_handles_stretch → 处理器不内部拉伸（World/HiFiGAN chain 内有 TimeStretchStage，vslib 原生拉伸）
        //   !opts.apply_pitch_edit    → pitch edit 链不会运行，内部拉伸无法触发，需回退到外部拉伸
        if (playback_rate - 1.0).abs() > 1e-6
            && (!processor_handles_stretch || !opts.apply_pitch_edit)
        {
            let seg_frames_in = segment.len() / 2;
            let target_frames = ((seg_frames_in as f64) / playback_rate).round().max(2.0) as usize;
            segment = time_stretch_interleaved(&segment, 2, out_rate, target_frames, opts.stretch);
        }

        // Loop（循环源）：整文件回绕已在片段构建阶段完成（见上方 build_loop_tiled_segment），
        // 此处 segment 天然覆盖整条 clip 的消费量，参数线阶段按绝对帧读取曲线即可。

        // Apply pitch edit per-clip (v2) if enabled.
        if opts.apply_pitch_edit {
            let seg_start_sec =
                clip_start_sec + pre_silence_sec + loop_seg_local_start_sec;
            let mut seg = segment;
            let applied = crate::pitch_editing::maybe_apply_pitch_edit_to_clip_segment(
                timeline,
                clip,
                clip_start_sec,
                seg_start_sec,
                out_rate,
                &mut seg,
            );
            match applied {
                Ok(true) => {
                    segment = seg;
                }
                Ok(false) => {
                    segment = seg;
                }
                Err(e) => {
                    eprintln!("[pitch_edit] clip_id={} ERROR: {e}", clip.id);
                    segment = seg;
                }
            }
        }

        // 提取共通 volume / pan 曲线（与 snapshot.rs 的逻辑对应）。
        // vslib 在合成阶段通过自己的控制点消费 volume/pan，mixdown 跳过避免二次应用。
        let (volume_curve, volume_curve_frame_period_ms, pan_curve, pan_curve_frame_period_ms) =
            timeline
                .resolve_root_track_id(&clip.track_id)
                .and_then(|root| {
                    let entry = timeline.params_by_root_track.get(&root)?;
                    let track = timeline.tracks.iter().find(|t| t.id == root)?;
                    let kind = crate::state::SynthPipelineKind::from_track_algo(
                        &track.pitch_analysis_algo,
                    );
                    if crate::pitch_editing::processor_bakes_common_mix_curves(kind) {
                        return None;
                    }
                    let volume = crate::pitch_editing::common_volume_curve_for_clip(entry, clip);
                    let pan = crate::pitch_editing::common_pan_curve_for_clip(entry, clip);
                    Some((
                        volume,
                        entry.frame_period_ms.max(0.1),
                        pan,
                        entry.frame_period_ms.max(0.1),
                    ))
                })
                .unwrap_or((None, 5.0, None, 5.0));

        // Apply fades (linear) and gain (timeline-referenced).
        let fade_in_frames = (clip.effective_fade_in_sec().max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let fade_out_frames = (clip.effective_fade_out_sec().max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as usize;

        let seg_frames = segment.len() / 2;
        let clip_total_frames = (clip_timeline_len_sec * out_rate as f64).round().max(1.0) as usize;
        let pre_silence_frames = (pre_silence_sec * out_rate as f64).round().max(0.0) as usize;

        // Mix into output, considering overlap window.
        // The audio segment starts after pre_silence_sec (Loop：再叠加窗口
        // 起点的 clip 局部偏移 —— 平铺段只覆盖窗口交集，见上方) and lasts seg_frames/out_rate.
        let seg_start_sec =
            clip_start_sec + pre_silence_sec + loop_seg_local_start_sec;
        let seg_end_sec = seg_start_sec + (seg_frames as f64) / out_rate as f64;

        // Loop（循环源）：平铺段只覆盖【导出窗口 ∩ clip】，seg 内的帧偏移是
        // "窗口内相对位置"；淡化 / 音量 / 声像曲线必须按 **clip 局部绝对位置**
        // 求值 —— 否则局部导出（后台预渲染、区间导出、波形 peaks）会在每个
        // 窗口边界重新触发 fade-in。非 Loop 时该偏移为 0，行为不变。
        let loop_local_offset_frames = if loop_mode {
            ((loop_seg_local_start_sec * out_rate as f64).round().max(0.0)) as usize
        } else {
            0usize
        };

        let clip_window_start = seg_start_sec.max(start_sec);
        let clip_window_end = seg_end_sec.min(end_sec).min(clip_end_sec);
        let window_len_sec = (clip_window_end - clip_window_start).max(0.0);
        if window_len_sec <= 1e-9 {
            continue;
        }

        let out_offset_frames = ((clip_window_start - start_sec) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let seg_offset_frames = ((clip_window_start - seg_start_sec) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let frames_to_mix = ((window_len_sec) * out_rate as f64).round().max(0.0) as usize;

        let max_frames_to_mix = frames_to_mix
            .min(out_frames.saturating_sub(out_offset_frames))
            .min(seg_frames.saturating_sub(seg_offset_frames));
        if max_frames_to_mix == 0 {
            continue;
        }

        clips_mixed = clips_mixed.saturating_add(1);

        let has_volume_curve = volume_curve.is_some() && !volume_curve.as_ref().unwrap().is_empty();
        let has_pan_curve = pan_curve.is_some() && !pan_curve.as_ref().unwrap().is_empty();
        for f in 0..max_frames_to_mix {
            if f % 4096 == 0 && mixdown_cancelled(&opts) {
                return Err("export_cancelled".to_string());
            }
            let oi = (out_offset_frames + f) * 2;
            let si = (seg_offset_frames + f) * 2;

            // Local position inside the CLIP (timeline), used for fades.
            // Loop：叠加窗口起点的 clip 局部偏移（见上方 loop_local_offset_frames）。
            let local_in_clip = pre_silence_frames
                .saturating_add(loop_local_offset_frames)
                .saturating_add(seg_offset_frames + f);
            if local_in_clip >= clip_total_frames {
                break;
            }

            let mut g = gain;
            if fade_in_frames > 0 && local_in_clip < fade_in_frames {
                g *= (local_in_clip as f32 / fade_in_frames as f32).clamp(0.0, 1.0);
            }
            if fade_out_frames > 0 && local_in_clip + fade_out_frames > clip_total_frames {
                let remain = clip_total_frames.saturating_sub(local_in_clip);
                g *= (remain as f32 / fade_out_frames as f32).clamp(0.0, 1.0);
            }
            if g <= 0.0 {
                continue;
            }

            // 只有真存在曲线时才计算
            let mut final_g = g;
            let abs_sec = clip_start_sec + (local_in_clip as f64 / out_rate as f64);
            if has_volume_curve {
                let vol = sample_automation_curve_at_sec(
                    volume_curve,
                    abs_sec,
                    volume_curve_frame_period_ms,
                    1.0,
                );
                final_g *= vol;
            }

            let pan = if has_pan_curve {
                sample_automation_curve_at_sec(pan_curve, abs_sec, pan_curve_frame_period_ms, 0.0)
            } else {
                0.0
            }
            .clamp(-1.0, 1.0);
            // 线性平衡：center 保持两声道增益为 1，避免中心衰减。
            let (left_gain, right_gain) = if pan <= 0.0 {
                (1.0, 1.0 + pan)
            } else {
                (1.0 - pan, 1.0)
            };

            mix[oi] += segment[si] * final_g * left_gain;
            mix[oi + 1] += segment[si + 1] * final_g * right_gain;
        }
    }

    if debug {
        let mut max_abs = 0.0f32;
        for &v in &mix {
            let a = v.abs();
            if a.is_finite() && a > max_abs {
                max_abs = a;
            }
        }
        eprintln!(
            "mixdown: rendered window start_sec={:.3} end_sec={:.3} sr={} frames={} max_abs={:.6} clips_considered={} clips_decoded={} clips_mixed={}",
            start_sec,
            end_sec,
            out_rate,
            out_frames,
            max_abs,
            clips_considered,
            clips_decoded,
            clips_mixed
        );
    }

    Ok((out_rate, out_channels, duration_sec, mix))
}

#[cfg(test)]
mod tests {
    use super::build_loop_tiled_segment;

    /// 交错 PCM：帧 i 的样本值为 [i as f32, i as f32 + 0.5]。
    fn make_pcm(frames: usize, channels: usize) -> Vec<f32> {
        let mut pcm = Vec::with_capacity(frames * channels);
        for f in 0..frames {
            for c in 0..channels {
                pcm.push(f as f32 + if channels > 1 { c as f32 * 0.5 } else { 0.0 });
            }
        }
        pcm
    }

    #[test]
    fn loop_tiled_forward_wraps_over_whole_file() {
        // 5 帧立体声媒体，锚点 3：期望序列 3,4,0,1,2,3,4,0
        let channels = 2;
        let pcm = make_pcm(5, channels);
        let out = build_loop_tiled_segment(&pcm, channels, 3, false, 8);
        assert_eq!(out.len(), 8 * channels);
        let expected = [3.0f32, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0];
        for (i, f) in expected.iter().enumerate() {
            assert_eq!(out[i * 2], *f, "forward frame {i} left");
            assert_eq!(out[i * 2 + 1], *f + 0.5, "forward frame {i} right");
        }
    }

    #[test]
    fn loop_tiled_reverse_descends_and_wraps() {
        // 锚点 exclusive=4（即从帧 3 开始向下）：3,2,1,0,4,3,2
        let pcm = make_pcm(5, 1);
        let out = build_loop_tiled_segment(&pcm, 1, 4, true, 7);
        let expected = [3.0f32, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0];
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn loop_tiled_negative_forward_anchor_wraps_to_tail() {
        // 负锚点 -2 对 5 帧媒体 → floor_mod(-2,5)=3：序列 3,4,0,1,2
        let pcm = make_pcm(5, 1);
        let out = build_loop_tiled_segment(&pcm, 1, -2, false, 5);
        let expected = [3.0f32, 4.0, 0.0, 1.0, 2.0];
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn loop_tiled_matches_per_frame_floor_mod_reference() {
        // 与逐帧 floor_mod 参考实现对拍（多周期 + 大锚点偏移）
        let total = 37i64;
        let pcm = make_pcm(total as usize, 2);
        for &anchor in &[-50i64, 0, 1, 19, 36, 1000] {
            for &reversed in &[false, true] {
                let n = 100usize;
                let out = build_loop_tiled_segment(&pcm, 2, anchor, reversed, n);
                assert_eq!(out.len(), n * 2);
                for f in 0..n {
                    let fi = f as i64;
                    let expect = if reversed {
                        (anchor - 1 - fi).rem_euclid(total)
                    } else {
                        (anchor + fi).rem_euclid(total)
                    } as usize;
                    assert_eq!(out[f * 2], expect as f32);
                    assert_eq!(out[f * 2 + 1], expect as f32 + 0.5);
                }
            }
        }
    }

    #[test]
    fn loop_tiled_handles_empty_and_degenerate_inputs() {
        assert!(build_loop_tiled_segment(&[], 2, 0, false, 10).is_empty());
        let pcm = make_pcm(4, 2);
        assert_eq!(build_loop_tiled_segment(&pcm, 2, 0, false, 0).len(), 0);
        // 单帧媒体也能循环铺满
        let one = vec![0.5f32, 0.25f32];
        let out = build_loop_tiled_segment(&one, 2, 1234567, false, 3);
        assert_eq!(out, vec![0.5, 0.25, 0.5, 0.25, 0.5, 0.25]);
    }
}
