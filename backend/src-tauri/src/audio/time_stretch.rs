use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserStretchAlgorithm {
    Linear,
    Signalsmith,
    Soundtouch,
}

impl Default for UserStretchAlgorithm {
    fn default() -> Self {
        Self::Signalsmith
    }
}

impl UserStretchAlgorithm {
    pub fn to_runtime(self) -> StretchAlgorithm {
        match self {
            Self::Linear => StretchAlgorithm::LinearResample,
            Self::Signalsmith => StretchAlgorithm::SignalsmithStretch,
            Self::Soundtouch => StretchAlgorithm::SoundTouchDll,
        }
    }
}

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

    /// Default time-stretch implementation via SoundTouch Windows DLL.
    SoundTouchDll,

    /// Desired: zplane Elastique (Soloist) time-stretch preserving pitch + formants.
    /// This requires integrating the Elastique SDK (commercial) — deliberately
    /// kept as a roadmap placeholder until that integration happens.
    #[allow(dead_code)]
    ElastiqueSoloist,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeStretchSettings {
    pub default_algorithm: UserStretchAlgorithm,
    pub default_hifigan_mel_stretch: bool,
    pub project_algorithm_override: Option<UserStretchAlgorithm>,
    pub project_hifigan_mel_stretch_override: Option<bool>,
}

impl Default for RuntimeStretchSettings {
    fn default() -> Self {
        Self {
            default_algorithm: UserStretchAlgorithm::default(),
            default_hifigan_mel_stretch: true,
            project_algorithm_override: None,
            project_hifigan_mel_stretch_override: None,
        }
    }
}

impl RuntimeStretchSettings {
    pub fn effective_algorithm(self) -> UserStretchAlgorithm {
        self.project_algorithm_override
            .unwrap_or(self.default_algorithm)
    }

    pub fn effective_hifigan_mel_stretch(self) -> bool {
        self.project_hifigan_mel_stretch_override
            .unwrap_or(self.default_hifigan_mel_stretch)
    }
}

fn runtime_stretch_settings_cell() -> &'static Mutex<RuntimeStretchSettings> {
    static CELL: OnceLock<Mutex<RuntimeStretchSettings>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(RuntimeStretchSettings::default()))
}

pub fn current_runtime_stretch_settings() -> RuntimeStretchSettings {
    *runtime_stretch_settings_cell()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub fn update_runtime_stretch_settings(
    default_algorithm: UserStretchAlgorithm,
    default_hifigan_mel_stretch: bool,
    project_algorithm_override: Option<UserStretchAlgorithm>,
    project_hifigan_mel_stretch_override: Option<bool>,
) {
    let mut settings = runtime_stretch_settings_cell()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    settings.default_algorithm = default_algorithm;
    settings.default_hifigan_mel_stretch = default_hifigan_mel_stretch;
    settings.project_algorithm_override = project_algorithm_override;
    settings.project_hifigan_mel_stretch_override = project_hifigan_mel_stretch_override;
}

pub fn update_global_stretch_defaults(
    default_algorithm: UserStretchAlgorithm,
    default_hifigan_mel_stretch: bool,
) {
    let mut settings = runtime_stretch_settings_cell()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    settings.default_algorithm = default_algorithm;
    settings.default_hifigan_mel_stretch = default_hifigan_mel_stretch;
}

pub fn update_project_stretch_overrides(
    project_algorithm_override: Option<UserStretchAlgorithm>,
    project_hifigan_mel_stretch_override: Option<bool>,
) {
    let mut settings = runtime_stretch_settings_cell()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    settings.project_algorithm_override = project_algorithm_override;
    settings.project_hifigan_mel_stretch_override = project_hifigan_mel_stretch_override;
}

pub fn resolved_external_stretch_algorithm() -> StretchAlgorithm {
    current_runtime_stretch_settings()
        .effective_algorithm()
        .to_runtime()
}

pub fn resolved_user_external_stretch_algorithm() -> UserStretchAlgorithm {
    current_runtime_stretch_settings().effective_algorithm()
}

pub fn should_use_hifigan_mel_stretch() -> bool {
    current_runtime_stretch_settings().effective_hifigan_mel_stretch()
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
        StretchAlgorithm::SoundTouchDll => {
            let in_frames = if channels == 0 {
                0
            } else {
                input.len() / channels
            };
            if in_frames < 2 || out_frames < 2 {
                return linear_time_stretch_interleaved(input, channels, out_frames);
            }
            let ratio = (out_frames as f64) / (in_frames as f64);
            let result = crate::soundtouch::try_time_stretch_interleaved_realtime(
                input,
                channels,
                sample_rate.max(1),
                ratio,
                out_frames,
            )
            .or_else(|_| {
                crate::soundtouch::try_time_stretch_interleaved_offline(
                    input,
                    channels,
                    sample_rate.max(1),
                    ratio,
                    out_frames,
                )
            });

            match result {
                Ok(mut out) => {
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
                        eprintln!("time_stretch: SoundTouch failed, falling back: {e}");
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

#[cfg(test)]
mod tests {
    use super::{
        current_runtime_stretch_settings, resolved_external_stretch_algorithm,
        should_use_hifigan_mel_stretch, time_stretch_interleaved, update_runtime_stretch_settings,
        StretchAlgorithm, UserStretchAlgorithm,
    };

    #[test]
    fn soundtouch_fallback_keeps_requested_length() {
        let input = vec![0.0f32, 0.5, 0.25, -0.25];
        let out = time_stretch_interleaved(&input, 1, 44_100, 8, StretchAlgorithm::SoundTouchDll);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn default_algorithm_symbol_exists() {
        let algo = StretchAlgorithm::SoundTouchDll;
        assert!(matches!(algo, StretchAlgorithm::SoundTouchDll));
    }

    #[test]
    fn project_override_inherits_and_resolves_from_global_defaults() {
        update_runtime_stretch_settings(UserStretchAlgorithm::Signalsmith, true, None, None);
        let settings = current_runtime_stretch_settings();
        assert_eq!(
            settings.effective_algorithm(),
            UserStretchAlgorithm::Signalsmith
        );
        assert!(settings.effective_hifigan_mel_stretch());
        assert!(matches!(
            resolved_external_stretch_algorithm(),
            StretchAlgorithm::SignalsmithStretch
        ));
        assert!(should_use_hifigan_mel_stretch());

        update_runtime_stretch_settings(
            UserStretchAlgorithm::Signalsmith,
            true,
            Some(UserStretchAlgorithm::Linear),
            Some(false),
        );
        let settings = current_runtime_stretch_settings();
        assert_eq!(settings.effective_algorithm(), UserStretchAlgorithm::Linear);
        assert!(!settings.effective_hifigan_mel_stretch());
        assert!(matches!(
            resolved_external_stretch_algorithm(),
            StretchAlgorithm::LinearResample
        ));
        assert!(!should_use_hifigan_mel_stretch());
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
