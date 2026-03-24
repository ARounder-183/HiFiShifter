#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StretchAlgorithm {
    /// Current fallback: linear resampling in the time domain.
    /// NOTE: This changes pitch/formants when the ratio != 1.
    LinearResample,

    /// High-quality time-stretch (pitch-preserving) via Signalsmith Stretch (MIT).
    ///
    /// Implementation uses a C wrapper over the header-only C++ library,
    /// statically linked at compile time. Always available.
    SignalsmithStretch,

    /// Desired: zplane Elastique (Soloist) time-stretch preserving pitch + formants.
    /// This requires integrating the Elastique SDK (commercial).
    ElastiqueSoloist,
}

const STRETCH_SILENCE_WINDOW_MS: f64 = 10.0;
const STRETCH_MIN_SILENCE_MS: f64 = 20.0;
const STRETCH_SILENCE_RMS: f32 = 1.0e-4;

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
}

fn preserve_hard_silence_after_stretch(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    sample_rate: u32,
) {
    if input.is_empty() || output.is_empty() || channels == 0 {
        return;
    }

    let in_frames = input.len() / channels;
    let out_frames = output.len() / channels;
    if in_frames == 0 || out_frames == 0 {
        return;
    }

    let silence_rms = env_f32("HIFISHIFTER_STRETCH_SILENCE_RMS")
        .unwrap_or(STRETCH_SILENCE_RMS)
        .max(0.0);
    let window_frames = ((sample_rate.max(1) as f64) * (STRETCH_SILENCE_WINDOW_MS / 1000.0))
        .round()
        .max(1.0) as usize;
    let min_silence_blocks = (STRETCH_MIN_SILENCE_MS / STRETCH_SILENCE_WINDOW_MS)
        .round()
        .max(1.0) as usize;

    let block_count = in_frames.div_ceil(window_frames);
    let mut silent_blocks = vec![false; block_count];

    for (block_index, silent) in silent_blocks.iter_mut().enumerate() {
        let start_frame = block_index.saturating_mul(window_frames);
        let end_frame = (start_frame + window_frames).min(in_frames);
        if start_frame >= end_frame {
            continue;
        }

        let mut energy = 0.0f64;
        let mut sample_count = 0usize;
        for frame in start_frame..end_frame {
            let base = frame * channels;
            for channel in 0..channels {
                let sample = input[base + channel] as f64;
                energy += sample * sample;
                sample_count += 1;
            }
        }

        if sample_count == 0 {
            continue;
        }

        let rms = (energy / sample_count as f64).sqrt() as f32;
        *silent = rms <= silence_rms;
    }

    let mut run_start: Option<usize> = None;
    for index in 0..=silent_blocks.len() {
        let is_silent = silent_blocks.get(index).copied().unwrap_or(false);
        match (run_start, is_silent) {
            (None, true) => run_start = Some(index),
            (Some(start), false) => {
                if index.saturating_sub(start) < min_silence_blocks {
                    for block in &mut silent_blocks[start..index] {
                        *block = false;
                    }
                }
                run_start = None;
            }
            _ => {}
        }
    }

    if !silent_blocks.iter().any(|&silent| silent) {
        return;
    }

    let scale = if out_frames <= 1 || in_frames <= 1 {
        0.0
    } else {
        (in_frames - 1) as f64 / (out_frames - 1) as f64
    };

    for out_frame in 0..out_frames {
        let source_frame = if out_frames <= 1 || in_frames <= 1 {
            0
        } else {
            ((out_frame as f64) * scale)
                .round()
                .clamp(0.0, (in_frames - 1) as f64) as usize
        };
        let block_index = (source_frame / window_frames).min(silent_blocks.len() - 1);
        if !silent_blocks[block_index] {
            continue;
        }
        let base = out_frame * channels;
        for channel in 0..channels {
            output[base + channel] = 0.0;
        }
    }
}

pub fn time_stretch_interleaved(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    out_frames: usize,
    algorithm: StretchAlgorithm,
) -> Vec<f32> {
    match algorithm {
        StretchAlgorithm::LinearResample => {
            linear_time_stretch_interleaved(input, channels, out_frames)
        }
        StretchAlgorithm::SignalsmithStretch => {
            // Signalsmith Stretch: time ratio = out / in.
            let in_frames = if channels == 0 {
                0
            } else {
                input.len() / channels
            };
            if in_frames < 2 || out_frames < 2 {
                return linear_time_stretch_interleaved(input, channels, out_frames);
            }
            let ratio = (out_frames as f64) / (in_frames as f64);

            // 优先使用实时模式，与 stretch_stream 路径统一。
            // 若实时模式失败，回退到离线模式。
            let result = crate::sstretch::try_time_stretch_interleaved_realtime(
                input,
                channels,
                sample_rate.max(1),
                ratio,
                out_frames,
            )
            .or_else(|_| {
                crate::sstretch::try_time_stretch_interleaved_offline(
                    input,
                    channels,
                    sample_rate.max(1),
                    ratio,
                    out_frames,
                )
            });

            match result {
                Ok(mut out) => {
                    // 确保输出长度精确匹配请求
                    preserve_hard_silence_after_stretch(
                        input,
                        &mut out,
                        channels,
                        sample_rate.max(1),
                    );
                    out.resize(out_frames * channels, 0.0);
                    out
                }
                Err(e) => {
                    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                        eprintln!("time_stretch: SignalsmithStretch failed, falling back: {e}");
                    }
                    linear_time_stretch_interleaved(input, channels, out_frames)
                }
            }
        }
        StretchAlgorithm::ElastiqueSoloist => {
            // TODO: integrate Elastique SDK and implement true pitch/formant-preserving stretch.
            // For now, fall back to the existing linear method to keep the app functional.
            linear_time_stretch_interleaved(input, channels, out_frames)
        }
    }
}

fn linear_time_stretch_interleaved(input: &[f32], channels: usize, out_frames: usize) -> Vec<f32> {
    if input.is_empty() || channels == 0 {
        return vec![];
    }
    let in_frames = input.len() / channels;
    if in_frames == 0 {
        return vec![];
    }
    if in_frames == out_frames {
        return input.to_vec();
    }
    if out_frames <= 1 || in_frames <= 1 {
        let mut out = vec![0.0f32; out_frames * channels];
        let copy_frames = in_frames.min(out_frames);
        out[..copy_frames * channels].copy_from_slice(&input[..copy_frames * channels]);
        return out;
    }

    let mut out = vec![0.0f32; out_frames * channels];
    let scale = (in_frames - 1) as f64 / (out_frames - 1) as f64;

    for of in 0..out_frames {
        let t_in = (of as f64) * scale;
        let i0 = t_in as usize;
        let i1 = (i0 + 1).min(in_frames - 1);
        let frac = (t_in - (i0 as f64)) as f32;

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
