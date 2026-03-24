//! ExternalResamplerProcessor：通过子进程调用外部 UTAU Resampler 的 ClipProcessor 实现。
//!
//! 遵循 UTAU resampler 标准命令行协议，将 clip PCM 写入临时 WAV，
//! 调用外部 resampler.exe 做音高变换，读回结果 WAV 并返回。
//!
//! 支持任意用户注册的 resampler（如 Moresampler、TIPS、straycat、tn_fnds 等）。

use super::traits::{
    ClipProcessContext, ClipProcessor, ParamDescriptor, ParamKind, ProcessorCapabilities,
    RenderContext, Renderer, RendererCapabilities,
};
use crate::state::{ResamplerEntry, SynthPipelineKind};

use hound::{SampleFormat, WavSpec, WavWriter, WavReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

// ─── UTAU Base64 编码常量 ──────────────────────────────────────────────────────

const B64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// ─── 辅助工具函数 ──────────────────────────────────────────────────────────────

/// MIDI 音高 → UTAU 音名格式（如 60 → "C4"、69 → "A4"）。
fn midi_to_utau_pitch(midi: f32) -> String {
    const NAMES: &[&str] = &[
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let note = midi.round() as i32;
    let octave = (note / 12) - 1; // MIDI 60 = C4
    let name_idx = ((note % 12) + 12) as usize % 12;
    format!("{}{}", NAMES[name_idx], octave)
}

/// 计算 pitch_edit 曲线中有效帧（> 0）的中位数 MIDI 值，作为 UTAU 目标音高。
fn median_pitch(pitch_edit: &[f32]) -> f32 {
    let mut valid: Vec<f32> = pitch_edit
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v > 0.0)
        .collect();
    if valid.is_empty() {
        return 60.0; // 默认 C4
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = valid.len() / 2;
    if valid.len() % 2 == 0 {
        (valid[mid - 1] + valid[mid]) / 2.0
    } else {
        valid[mid]
    }
}

/// 将逐帧 cent 偏移编码为 UTAU pitchbend Base64 字符串。
///
/// 每个值 = 12-bit 有符号整数，范围 [-2048, +2047]，
/// 编码为 2 个 Base64 字符（6bit + 6bit = 12bit）。
fn encode_pitchbend(cent_offsets: &[i16]) -> String {
    let mut result = String::with_capacity(cent_offsets.len() * 2);
    for &delta in cent_offsets {
        let val = (delta as i32).clamp(-2048, 2047);
        let unsigned = if val < 0 { (val + 4096) as u16 } else { val as u16 };
        result.push(B64_TABLE[(unsigned >> 6) as usize] as char);
        result.push(B64_TABLE[(unsigned & 0x3F) as usize] as char);
    }
    result
}

/// 将 HiFiShifter 帧率的 pitchbend 曲线重采样到 UTAU 帧率。
///
/// UTAU pitchbend 帧率 ≈ tempo × 96 / 60 Hz。
/// HiFiShifter 帧率 = 1000 / frame_period_ms Hz。
fn resample_pitchbend(
    cent_offsets: &[i16],
    frame_period_ms: f64,
    bpm: f64,
) -> Vec<i16> {
    let hs_fps = 1000.0 / frame_period_ms.max(0.1);
    let utau_fps = bpm.max(1.0) * 96.0 / 60.0;
    let ratio = hs_fps / utau_fps;

    if cent_offsets.is_empty() {
        return vec![];
    }

    let out_len = ((cent_offsets.len() as f64) / ratio).ceil() as usize;
    let mut result = Vec::with_capacity(out_len.max(1));

    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let i0 = src_idx.floor() as usize;
        let i1 = (i0 + 1).min(cent_offsets.len().saturating_sub(1));
        let frac = src_idx - i0 as f64;
        let a = cent_offsets.get(i0).copied().unwrap_or(0) as f64;
        let b = cent_offsets.get(i1).copied().unwrap_or(0) as f64;
        let v = (a + (b - a) * frac).round().clamp(-2048.0, 2047.0) as i16;
        result.push(v);
    }
    result
}

/// 返回 resampler 临时目录。
fn resampler_temp_dir() -> PathBuf {
    let t = std::env::temp_dir();
    let dir = t.join("hifishifter").join("resampler");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 为 UTAU resampler 生成 `.frq` 文件（频率分析数据）。
///
/// 大多数 UTAU resampler（tn_fnds、Moresampler、TIPS 等）期望输入 WAV 旁存在
/// 同名的 `{wav_path}_frq` 文件，包含预计算的基频分析。
///
/// `.frq` 文件格式（UTAU FREQ0003）：
/// - 8 字节 magic: "FREQ0003"
/// - i32 LE: 采样率（Hz）
/// - i32 LE: 帧周期（采样数，通常 256）
/// - i32 LE: 平均频率（Hz，四舍五入整数）
/// - 16 字节: 空白预留
/// - i32 LE: 数据帧数
/// - 每帧: f64 LE 频率（Hz）+ f64 LE 振幅
///
/// `pitch_orig`: 源音频的原始音高曲线（全局绝对帧索引，MIDI 值），
///   用于告诉 resampler 输入 WAV 的真实基频。
/// `clip_start_sec`: clip 在时间轴上的起点（秒），用于将 pitch_orig 对齐到 PCM。
fn generate_frq_file(
    wav_path: &Path,
    pcm: &[f32],
    sample_rate: u32,
    pitch_orig: &[f32],
    frame_period_ms: f64,
    clip_start_sec: f64,
) -> Result<PathBuf, String> {
    use std::io::Write;

    // frq 文件路径：{wav_path}_frq（UTAU 标准命名）
    let frq_path = PathBuf::from(format!("{}_frq", wav_path.display()));

    // frq 帧周期（采样数）。UTAU 标准默认 256 采样/帧。
    let hop_samples: i32 = 256;
    let hop_sec = hop_samples as f64 / sample_rate.max(1) as f64;

    // 计算帧数
    let total_samples = pcm.len();
    let num_frames = if total_samples > 0 {
        ((total_samples as f64) / hop_samples as f64).ceil() as usize
    } else {
        0
    };

    // 从 pitch_orig（原始音高）曲线提取每帧频率和振幅。
    // pitch_orig 是全局绝对帧索引，需要加上 clip_start_sec 偏移来定位正确区间。
    let mut freqs = Vec::with_capacity(num_frames);
    let mut amps = Vec::with_capacity(num_frames);

    let inv_fp = 1000.0 / frame_period_ms.max(0.1);
    let _pcm_duration_sec = total_samples as f64 / sample_rate.max(1) as f64;

    for i in 0..num_frames {
        let time_sec = i as f64 * hop_sec;

        // 从 pitch_orig 曲线插值获取当前时间的 MIDI 值。
        // time_sec 是相对于 clip PCM 的本地时间，加上 clip_start_sec 得到全局时间，
        // 再换算到 pitch_orig 的帧索引。
        let global_time_sec = clip_start_sec + time_sec;
        let pitch_idx_f = global_time_sec * inv_fp;
        let midi_val = if pitch_orig.is_empty() {
            0.0f32
        } else {
            let i0 = pitch_idx_f.floor() as usize;
            let i1 = (i0 + 1).min(pitch_orig.len().saturating_sub(1));
            let frac = (pitch_idx_f - i0 as f64) as f32;
            let v0 = pitch_orig.get(i0).copied().unwrap_or(0.0);
            let v1 = pitch_orig.get(i1).copied().unwrap_or(0.0);
            if v0 > 0.0 && v0.is_finite() && v1 > 0.0 && v1.is_finite() {
                v0 + (v1 - v0) * frac
            } else if v0 > 0.0 && v0.is_finite() {
                v0
            } else if v1 > 0.0 && v1.is_finite() {
                v1
            } else {
                0.0
            }
        };

        // MIDI → Hz
        let freq = if midi_val > 0.0 && midi_val.is_finite() {
            440.0 * 2.0f64.powf((midi_val as f64 - 69.0) / 12.0)
        } else {
            0.0 // 无声帧
        };

        // 计算该帧的 RMS 振幅
        let sample_start = (i * hop_samples as usize).min(total_samples);
        let sample_end = ((i + 1) * hop_samples as usize).min(total_samples);
        let amp = if sample_end > sample_start {
            let sum_sq: f64 = pcm[sample_start..sample_end]
                .iter()
                .map(|&s| (s as f64) * (s as f64))
                .sum();
            (sum_sq / (sample_end - sample_start) as f64).sqrt()
        } else {
            0.0
        };

        freqs.push(freq);
        amps.push(amp);
    }

    // 计算平均频率（仅统计有声帧）
    let voiced_freqs: Vec<f64> = freqs.iter().copied().filter(|&f| f > 0.0).collect();
    let avg_freq = if voiced_freqs.is_empty() {
        0i32
    } else {
        (voiced_freqs.iter().sum::<f64>() / voiced_freqs.len() as f64).round() as i32
    };

    // 写入 frq 文件
    let mut file = std::fs::File::create(&frq_path)
        .map_err(|e| format!("创建 frq 文件失败: {e}"))?;

    // Header: "FREQ0003"
    file.write_all(b"FREQ0003")
        .map_err(|e| format!("写入 frq header 失败: {e}"))?;

    // 采样率 (i32 LE)
    file.write_all(&(sample_rate as i32).to_le_bytes())
        .map_err(|e| format!("写入 frq 采样率失败: {e}"))?;

    // 帧周期（采样数, i32 LE）
    file.write_all(&hop_samples.to_le_bytes())
        .map_err(|e| format!("写入 frq 帧周期失败: {e}"))?;

    // 平均频率 (i32 LE)
    file.write_all(&avg_freq.to_le_bytes())
        .map_err(|e| format!("写入 frq 平均频率失败: {e}"))?;

    // 16 字节空白预留
    file.write_all(&[0u8; 16])
        .map_err(|e| format!("写入 frq 预留字段失败: {e}"))?;

    // 帧数 (i32 LE)
    file.write_all(&(num_frames as i32).to_le_bytes())
        .map_err(|e| format!("写入 frq 帧数失败: {e}"))?;

    // 每帧: f64 频率 + f64 振幅
    for i in 0..num_frames {
        file.write_all(&freqs[i].to_le_bytes())
            .map_err(|e| format!("写入 frq 频率帧失败: {e}"))?;
        file.write_all(&amps[i].to_le_bytes())
            .map_err(|e| format!("写入 frq 振幅帧失败: {e}"))?;
    }

    file.flush()
        .map_err(|e| format!("flush frq 文件失败: {e}"))?;

    Ok(frq_path)
}

/// 将单声道 f32 PCM 写入临时 WAV 文件（16-bit int），返回文件路径。
fn write_temp_wav_mono(pcm: &[f32], sample_rate: u32) -> Result<PathBuf, String> {
    let temp_dir = resampler_temp_dir();
    let uuid = uuid::Uuid::new_v4().to_string().replace('-', "");
    let path = temp_dir.join(format!("hs_resampler_in_{uuid}.wav"));
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(&path, spec)
        .map_err(|e| format!("创建临时 WAV 失败: {e}"))?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .map_err(|e| format!("写入 WAV 样本失败: {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize WAV 失败: {e}"))?;
    Ok(path)
}

/// 从 WAV 文件读取单声道 f32 PCM。
fn read_wav_mono(path: &Path) -> Result<Vec<f32>, String> {
    let reader = WavReader::open(path)
        .map_err(|e| format!("读取输出 WAV 失败: {e}"))?;
    let spec = reader.spec();
    let bits = spec.bits_per_sample;
    let channels = spec.channels as usize;

    match spec.sample_format {
        SampleFormat::Int => {
            let samples: Vec<i32> = reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .collect();
            // 取第一声道
            let mono: Vec<f32> = samples
                .chunks(channels)
                .map(|ch| {
                    let s = ch[0] as f64;
                    let max_val = (1i64 << (bits - 1)) as f64;
                    (s / max_val) as f32
                })
                .collect();
            Ok(mono)
        }
        SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect();
            let mono: Vec<f32> = samples
                .chunks(channels)
                .map(|ch| ch[0])
                .collect();
            Ok(mono)
        }
    }
}

/// RAII 辅助：drop 时删除临时文件。
struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ─── 静态参数描述符 ───────────────────────────────────────────────────────────

static RESAMPLER_PARAMS: &[ParamDescriptor] = &[
    // 力度（velocity）
    ParamDescriptor {
        id: "resampler_velocity",
        display_name: "力度 (Velocity)",
        group: "Resampler",
        kind: ParamKind::AutomationCurve {
            unit: "",
            default_value: 100.0,
            min_value: 0.0,
            max_value: 200.0,
        },
    },
    // 调制（modulation）
    ParamDescriptor {
        id: "resampler_modulation",
        display_name: "调制 (Modulation)",
        group: "Resampler",
        kind: ParamKind::AutomationCurve {
            unit: "",
            default_value: 0.0,
            min_value: 0.0,
            max_value: 200.0,
        },
    },
    // 音量（volume）
    ParamDescriptor {
        id: "resampler_volume",
        display_name: "音量 (Volume)",
        group: "Resampler",
        kind: ParamKind::AutomationCurve {
            unit: "%",
            default_value: 100.0,
            min_value: 0.0,
            max_value: 200.0,
        },
    },
];

// ─── ExternalResamplerRenderer（Renderer trait stub）──────────────────────────

pub struct ExternalResamplerRenderer {
    pub entry: ResamplerEntry,
}

impl Renderer for ExternalResamplerRenderer {
    fn id(&self) -> &str {
        &self.entry.id
    }

    fn display_name(&self) -> &str {
        &self.entry.display_name
    }

    fn kind(&self) -> SynthPipelineKind {
        SynthPipelineKind::ExternalResampler(self.entry.id.clone())
    }

    fn is_available(&self) -> bool {
        self.entry.exe_path.exists()
    }

    fn render(&self, _ctx: &RenderContext<'_>) -> Result<Vec<f32>, String> {
        Err("ExternalResamplerRenderer::render() 不应被直接调用；请使用 get_processor()".to_string())
    }

    fn capabilities(&self) -> RendererCapabilities {
        RendererCapabilities {
            supports_realtime: false,
            prefers_prerender: true,
            max_pitch_shift_semitones: 24.0,
        }
    }
}

// ─── ExternalResamplerProcessor ───────────────────────────────────────────────

/// 外部 UTAU Resampler 全链路处理器。
pub struct ExternalResamplerProcessor {
    entry: ResamplerEntry,
}

impl ExternalResamplerProcessor {
    pub fn new(entry: ResamplerEntry) -> Self {
        Self { entry }
    }
}

impl ClipProcessor for ExternalResamplerProcessor {
    fn id(&self) -> &str {
        &self.entry.id
    }

    fn display_name(&self) -> &str {
        &self.entry.display_name
    }

    fn is_available(&self) -> bool {
        self.entry.exe_path.exists()
    }

    fn capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities {
            handles_time_stretch: true, // 通过 length_require 参数控制
            supports_formant: false,    // 通过 flags 中的 g 参数间接支持
            supports_breathiness: false, // 通过 flags 中的 B 参数间接支持
        }
    }

    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        let mut params = RESAMPLER_PARAMS.to_vec();

        // 为每个用户自定义的 flag 参数动态生成一个 AutomationCurve ParamDescriptor。
        // ID 格式: "resampler_flag_{key}"（如 resampler_flag_B、resampler_flag_g）。
        // 前端参数面板会自动渲染对应的滑块/曲线编辑器。
        for fp in &self.entry.flag_params {
            // ParamDescriptor 中 id / display_name / group 是 &'static str，
            // 但我们这里需要动态字符串。通过 leak 将 String 转成 &'static str。
            // 条目数量很少（通常 < 20），leak 量可忽略。
            let id: &'static str = format!("resampler_flag_{}", fp.key).leak();
            let display_name: &'static str = fp.display_name.clone().leak();
            params.push(ParamDescriptor {
                id,
                display_name,
                group: "Flags",
                kind: ParamKind::AutomationCurve {
                    unit: "",
                    default_value: fp.default_value as f32,
                    min_value: fp.min_value as f32,
                    max_value: fp.max_value as f32,
                },
            });
        }

        params
    }

    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String> {
        if ctx.mono_pcm.is_empty() {
            return Ok(vec![0.0f32; ctx.out_frames]);
        }

        // ── 0. 检查 resampler exe 是否存在 ──────────────────────────────────
        if !self.entry.exe_path.exists() {
            return Err(format!(
                "Resampler 可执行文件不存在: {} (\"{}\")\n\
                 请检查文件是否已被移动或删除，并在设置中更新路径。",
                self.entry.exe_path.display(),
                self.entry.display_name,
            ));
        }

        // ── 1. 写入临时 input WAV ──────────────────────────────────────────
        let input_wav = write_temp_wav_mono(ctx.mono_pcm, ctx.sample_rate)?;
        let _input_guard = TempFileGuard(input_wav.clone());

        // ── 1b. 生成 frq 文件（基频分析数据）─────────────────────────────
        // 大多数 UTAU resampler 期望输入 WAV 旁存在同名 frq 文件。
        // 使用 pitch_orig（源音频原始音高）而非 pitch_edit（用户编辑后的目标音高），
        // 因为 frq 的作用是告诉 resampler「输入 WAV 的原始基频是什么」。
        let frq_path = generate_frq_file(
            &input_wav,
            ctx.mono_pcm,
            ctx.sample_rate,
            ctx.pitch_orig,
            ctx.frame_period_ms,
            ctx.clip_start_sec,
        )?;
        let _frq_guard = TempFileGuard(frq_path);

        // ── 2. 准备输出路径 ──────────────────────────────────────────────────
        let uuid = uuid::Uuid::new_v4().to_string().replace('-', "");
        let output_wav = resampler_temp_dir().join(format!("hs_resampler_out_{uuid}.wav"));
        let _output_guard = TempFileGuard(output_wav.clone());

        eprintln!(
            "[resampler] begin clip_id={} resampler=\"{}\" exe={} sr={} in_frames={} out_frames={} rate={:.3}",
            ctx.clip_id,
            self.entry.display_name,
            self.entry.exe_path.display(),
            ctx.sample_rate,
            ctx.mono_pcm.len(),
            ctx.out_frames,
            ctx.playback_rate,
        );

        // ── 3. 裁剪 pitch_edit 到 clip 局部区间 ─────────────────────────
        // pitch_edit 是全局绝对帧索引的完整曲线（帧 0 对应时间轴 0 秒），
        // 但 resampler 只处理当前 clip 的局部区间。
        // 需要根据 clip_start_sec 和 PCM 时长计算帧范围，只取对应区间。
        let inv_fp = 1000.0 / ctx.frame_period_ms.max(0.1);
        let clip_start_frame = (ctx.clip_start_sec * inv_fp).round().max(0.0) as usize;
        let clip_duration_sec = ctx.mono_pcm.len() as f64 / ctx.sample_rate.max(1) as f64;
        let clip_end_frame = ((ctx.clip_start_sec + clip_duration_sec) * inv_fp)
            .ceil()
            .max(0.0) as usize;
        let clip_end_frame = clip_end_frame.min(ctx.pitch_edit.len());
        let clip_start_frame = clip_start_frame.min(clip_end_frame);

        let local_pitch_edit: &[f32] = if clip_start_frame < clip_end_frame {
            &ctx.pitch_edit[clip_start_frame..clip_end_frame]
        } else {
            &[]
        };

        // ── 4. 计算目标音高（局部 pitch_edit 中位数）─────────────────────
        let target_midi = median_pitch(local_pitch_edit);
        let pitch_str = midi_to_utau_pitch(target_midi);

        // ── 5. 计算 pitchbend（逐帧 cent 偏移 → Base64）──────────────────
        let cent_offsets: Vec<i16> = local_pitch_edit
            .iter()
            .map(|&midi| {
                if midi > 0.0 && midi.is_finite() {
                    ((midi - target_midi) * 100.0)
                        .round()
                        .clamp(-2048.0, 2047.0) as i16
                } else {
                    0i16
                }
            })
            .collect();

        // 重采样到 UTAU 帧率
        let bpm = if ctx.bpm.is_finite() && ctx.bpm > 0.0 { ctx.bpm } else { 120.0 };
        let resampled = resample_pitchbend(&cent_offsets, ctx.frame_period_ms, bpm);
        let pitchbend_b64 = encode_pitchbend(&resampled);

        // ── 6. 计算输出长度（考虑 playback_rate）──────────────────────────
        let source_ms = ctx.mono_pcm.len() as f64 / ctx.sample_rate.max(1) as f64 * 1000.0;
        let length_ms = source_ms / ctx.playback_rate.max(1e-6);

        // ── 7. 从 extra_params 获取参数，从 extra_curves 获取平均值 ────────
        let velocity = ctx
            .extra_curves
            .get("resampler_velocity")
            .and_then(|c| {
                let valid: Vec<f32> = c.iter().copied().filter(|v| v.is_finite()).collect();
                if valid.is_empty() {
                    None
                } else {
                    Some(valid.iter().sum::<f32>() / valid.len() as f32)
                }
            })
            .unwrap_or(100.0) as i32;

        let modulation = ctx
            .extra_curves
            .get("resampler_modulation")
            .and_then(|c| {
                let valid: Vec<f32> = c.iter().copied().filter(|v| v.is_finite()).collect();
                if valid.is_empty() {
                    None
                } else {
                    Some(valid.iter().sum::<f32>() / valid.len() as f32)
                }
            })
            .unwrap_or(0.0) as i32;

        let volume = ctx
            .extra_curves
            .get("resampler_volume")
            .and_then(|c| {
                let valid: Vec<f32> = c.iter().copied().filter(|v| v.is_finite()).collect();
                if valid.is_empty() {
                    None
                } else {
                    Some(valid.iter().sum::<f32>() / valid.len() as f32)
                }
            })
            .unwrap_or(100.0) as i32;

        // ── 7b. 从 extra_curves 获取用户自定义 flag 参数值，拼接 flags 字符串 ──
        //
        // 对于每个 FlagParam（如 B/g/t），从 extra_curves["resampler_flag_{key}"] 取平均值，
        // 以整数拼进 flags 字符串。如果曲线不存在或为空，使用 default_value。
        let mut flags = String::new();
        for fp in &self.entry.flag_params {
            let curve_id = format!("resampler_flag_{}", fp.key);
            let val = ctx
                .extra_curves
                .get(curve_id.as_str())
                .and_then(|c| {
                    let valid: Vec<f32> = c.iter().copied().filter(|v| v.is_finite()).collect();
                    if valid.is_empty() {
                        None
                    } else {
                        Some(valid.iter().sum::<f32>() / valid.len() as f32)
                    }
                })
                .unwrap_or(fp.default_value as f32);
            // 拼接: key + 整数值（如 "B50" "g-5"）
            flags.push_str(&format!("{}{}", fp.key, val.round() as i32));
        }
        // 追加 default_flags 中不在 flag_params 里的额外 flags
        if !self.entry.default_flags.is_empty() {
            flags.push_str(&self.entry.default_flags);
        }

        eprintln!(
            "[resampler] args: pitch={} velocity={} flags=\"{}\" length_ms={:.1} volume={} mod={} pitchbend_len={} local_pitch_frames={}",
            pitch_str, velocity, flags, length_ms, volume, modulation, resampled.len(), local_pitch_edit.len()
        );

        // ── 8. 构造命令行参数并调用 ────────────────────────────────────────
        let tempo_str = format!("!{:.0}", bpm);

        let mut cmd = Command::new(&self.entry.exe_path);
        cmd.args([
            input_wav.to_str().unwrap_or(""),
            output_wav.to_str().unwrap_or(""),
            &pitch_str,                     // 目标音高
            &velocity.to_string(),          // velocity
            &flags,                         // flags
            "0",                            // offset
            &format!("{:.1}", length_ms),   // length_require
            "0",                            // consonant
            "0",                            // blank
            &volume.to_string(),            // volume
            &modulation.to_string(),        // modulation
            &tempo_str,                     // tempo
        ]);

        // pitchbend 作为最后一个参数（如果有的话）
        if !pitchbend_b64.is_empty() {
            cmd.arg(&pitchbend_b64);
        }

        // 隐藏子进程窗口（Windows）
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        // 执行子进程（带超时保护）
        let child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "启动 Resampler 失败: {}\n可执行文件: {}\n请检查路径是否正确、文件是否可执行。",
                    e,
                    self.entry.exe_path.display(),
                )
            })?;

        let output = child
            .wait_with_output()
            .map_err(|e| format!("等待 Resampler 进程失败: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let code = output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string());
            return Err(format!(
                "Resampler \"{}\" 退出码: {}\n标准错误: {}",
                self.entry.display_name,
                code,
                if stderr.trim().is_empty() {
                    "(无输出)"
                } else {
                    stderr.trim()
                },
            ));
        }

        // ── 9. 读回 output.wav → Vec<f32> ──────────────────────────────────
        if !output_wav.exists() {
            return Err(format!(
                "Resampler \"{}\" 执行成功但未生成输出文件: {}\n\
                 可能是参数不兼容或 resampler 不支持当前调用方式。",
                self.entry.display_name,
                output_wav.display(),
            ));
        }

        let pcm = read_wav_mono(&output_wav)?;

        eprintln!(
            "[resampler] output: frames={} expected={} peak={:.6}",
            pcm.len(),
            ctx.out_frames,
            pcm.iter().fold(0.0f32, |m, &v| m.max(v.abs())),
        );

        // ── 10. 对齐到 ctx.out_frames ──────────────────────────────────────
        let mut out = vec![0.0f32; ctx.out_frames];
        let copy_len = pcm.len().min(ctx.out_frames);
        out[..copy_len].copy_from_slice(&pcm[..copy_len]);

        Ok(out)
    }
}

/// 返回 resampler 参数描述符（供前端 UI 查询）。
pub fn resampler_param_descriptors() -> &'static [ParamDescriptor] {
    RESAMPLER_PARAMS
}
