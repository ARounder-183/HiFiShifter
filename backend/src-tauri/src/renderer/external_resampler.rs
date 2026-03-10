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

        // ── 3. 计算目标音高（pitch_edit 中位数）──────────────────────────────
        let target_midi = median_pitch(ctx.pitch_edit);
        let pitch_str = midi_to_utau_pitch(target_midi);

        // ── 4. 计算 pitchbend（逐帧 cent 偏移 → Base64）──────────────────
        let cent_offsets: Vec<i16> = ctx
            .pitch_edit
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
        let bpm = 120.0; // 默认 BPM（从 ctx 取不到，使用安全默认值）
        let resampled = resample_pitchbend(&cent_offsets, ctx.frame_period_ms, bpm);
        let pitchbend_b64 = encode_pitchbend(&resampled);

        // ── 5. 计算输出长度（考虑 playback_rate）──────────────────────────
        let source_ms = ctx.mono_pcm.len() as f64 / ctx.sample_rate.max(1) as f64 * 1000.0;
        let length_ms = source_ms / ctx.playback_rate.max(1e-6);

        // ── 6. 从 extra_params 获取参数，从 extra_curves 获取平均值 ────────
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

        // ── 6b. 从 extra_curves 获取用户自定义 flag 参数值，拼接 flags 字符串 ──
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
            "[resampler] args: pitch={} velocity={} flags=\"{}\" length_ms={:.1} volume={} mod={} pitchbend_len={}",
            pitch_str, velocity, flags, length_ms, volume, modulation, resampled.len()
        );

        // ── 7. 构造命令行参数并调用 ────────────────────────────────────────
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

        // ── 8. 读回 output.wav → Vec<f32> ──────────────────────────────────
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

        // ── 9. 对齐到 ctx.out_frames ──────────────────────────────────────
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
