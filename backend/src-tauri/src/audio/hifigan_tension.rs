use num_complex::Complex32;
use rustfft::FftPlanner;

const WINDOW_SIZE: usize = 2048;
const HOP_SIZE: usize = 1024;
const MAX_TENSION_DB: f64 = 17.0;
const OUTPUT_CEILING: f32 = 0.98;

fn hann_window() -> Vec<f32> {
    (0..WINDOW_SIZE)
        .map(|index| {
            let phase = (2.0 * std::f64::consts::PI * index as f64) / WINDOW_SIZE as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

fn sample_curve_at_abs_sec(
    curve: Option<&Vec<f32>>,
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

    let i0 = idx_f.floor().max(0.0) as usize;
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

fn sample_target_midi_at_abs_sec(
    pitch_orig: &[f32],
    pitch_edit: &[f32],
    abs_sec: f64,
    frame_period_ms: f64,
) -> Option<f32> {
    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return None;
    }

    let i0 = idx_f.floor().max(0.0) as usize;
    let i1 = i0 + 1;
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;

    let sample_curve = |curve: &[f32]| -> Option<f32> {
        if curve.is_empty() || i0 >= curve.len() {
            return None;
        }
        let a = *curve.get(i0)?;
        let b = curve.get(i1).copied().unwrap_or(a);
        let value = a + (b - a) * frac;
        (value.is_finite() && value > 0.0).then_some(value)
    };

    sample_curve(pitch_edit).or_else(|| sample_curve(pitch_orig))
}

fn midi_to_hz(midi: f32) -> f64 {
    440.0 * (2.0f64).powf((midi as f64 - 69.0) / 12.0)
}

fn tension_center_hz(midi: f32) -> f64 {
    midi_to_hz(midi).clamp(100.0, 1000.0) * 2.0
}

fn apply_tension_to_channel(
    samples: &[f32],
    sample_rate: u32,
    clip_start_sec: f64,
    frame_period_ms: f64,
    pitch_orig: &[f32],
    pitch_edit: &[f32],
    tension_curve: Option<&Vec<f32>>,
) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return samples.to_vec();
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);
    let ifft = planner.plan_fft_inverse(WINDOW_SIZE);
    let window = hann_window();

    let mut output = vec![0.0f32; samples.len() + WINDOW_SIZE];
    let mut norm = vec![0.0f32; samples.len() + WINDOW_SIZE];
    let mut spectrum = vec![Complex32::new(0.0, 0.0); WINDOW_SIZE];
    let frame_count = if samples.len() <= WINDOW_SIZE {
        1
    } else {
        (samples.len() - 1) / HOP_SIZE + 1
    };

    for frame_index in 0..frame_count {
        let offset = frame_index * HOP_SIZE;
        spectrum.fill(Complex32::new(0.0, 0.0));

        for index in 0..WINDOW_SIZE {
            if let Some(&sample) = samples.get(offset + index) {
                spectrum[index].re = sample * window[index];
            }
        }

        fft.process(&mut spectrum);

        let center_sec =
            clip_start_sec + (offset + WINDOW_SIZE / 2) as f64 / sample_rate.max(1) as f64;
        let tension =
            sample_curve_at_abs_sec(tension_curve, center_sec, frame_period_ms, 0.0) as f64;
        let max_gain_db = (tension / 100.0) * MAX_TENSION_DB;

        if max_gain_db.abs() > 1e-4 {
            if let Some(midi) =
                sample_target_midi_at_abs_sec(pitch_orig, pitch_edit, center_sec, frame_period_ms)
            {
                let freq_hz = tension_center_hz(midi);
                let freq_bin = freq_hz * WINDOW_SIZE as f64 / sample_rate.max(1) as f64;
                if freq_bin.is_finite() && freq_bin > 1e-3 {
                    let clamp_db = max_gain_db.abs();
                    for (bin, value) in spectrum.iter_mut().enumerate() {
                        let mirror_bin = bin.min(WINDOW_SIZE.saturating_sub(bin)) as f64;
                        let shaped_db = (max_gain_db * (mirror_bin / freq_bin - 1.0))
                            .clamp(-clamp_db, clamp_db);
                        let linear = 10.0f32.powf(shaped_db as f32 / 20.0);
                        *value *= linear;
                    }
                }
            }
        }

        ifft.process(&mut spectrum);

        for index in 0..WINDOW_SIZE {
            if offset + index >= output.len() {
                break;
            }
            let sample = (spectrum[index].re / WINDOW_SIZE as f32) * window[index];
            output[offset + index] += sample;
            norm[offset + index] += window[index] * window[index];
        }
    }

    let mut normalized = vec![0.0f32; samples.len()];
    for (index, value) in normalized.iter_mut().enumerate() {
        let denom = norm[index];
        *value = if denom > 1e-6 {
            output[index] / denom
        } else {
            samples[index]
        };
    }

    normalized
}

fn apply_output_ceiling(samples: &mut [f32], ceiling: f32) {
    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if peak <= ceiling || peak <= 1e-6 {
        return;
    }

    let gain = ceiling / peak;
    for sample in samples {
        *sample *= gain;
    }
}

pub fn apply_tension_to_stereo(
    stereo_pcm: &[f32],
    sample_rate: u32,
    clip_start_sec: f64,
    frame_period_ms: f64,
    pitch_orig: &[f32],
    pitch_edit: &[f32],
    tension_curve: Option<&Vec<f32>>,
) -> Result<Vec<f32>, String> {
    if stereo_pcm.len() % 2 != 0 {
        return Err("hifigan tension expects stereo interleaved pcm".to_string());
    }
    if stereo_pcm.is_empty() {
        return Ok(Vec::new());
    }

    let frames = stereo_pcm.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for frame in 0..frames {
        left.push(stereo_pcm[frame * 2]);
        right.push(stereo_pcm[frame * 2 + 1]);
    }

    let left_processed = apply_tension_to_channel(
        &left,
        sample_rate,
        clip_start_sec,
        frame_period_ms,
        pitch_orig,
        pitch_edit,
        tension_curve,
    );
    let right_processed = apply_tension_to_channel(
        &right,
        sample_rate,
        clip_start_sec,
        frame_period_ms,
        pitch_orig,
        pitch_edit,
        tension_curve,
    );

    let mut interleaved = Vec::with_capacity(stereo_pcm.len());
    for frame in 0..frames {
        interleaved.push(left_processed[frame]);
        interleaved.push(right_processed[frame]);
    }
    // The DAW reference patch can exceed 0 dB internally, but our playback callback
    // hard-clamps to [-1, 1]. Apply a clip-level ceiling here to avoid audible crackle.
    apply_output_ceiling(&mut interleaved, OUTPUT_CEILING);
    Ok(interleaved)
}

#[cfg(test)]
mod tests {
    use super::{apply_output_ceiling, apply_tension_to_stereo, tension_center_hz, OUTPUT_CEILING};

    #[test]
    fn zero_tension_is_near_identity() {
        let mut stereo = Vec::new();
        for index in 0..4096 {
            let sample = ((index as f32 / 32.0).sin()) * 0.2;
            stereo.push(sample);
            stereo.push(sample);
        }
        let pitch_orig = vec![69.0f32; 512];
        let pitch_edit = vec![0.0f32; 512];
        let tension_curve = vec![0.0f32; 512];

        let processed = apply_tension_to_stereo(
            &stereo,
            44_100,
            0.0,
            5.0,
            &pitch_orig,
            &pitch_edit,
            Some(&tension_curve),
        )
        .expect("tension processing should succeed");

        assert_eq!(processed.len(), stereo.len());
        let max_diff = processed
            .iter()
            .zip(stereo.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max_diff={max_diff}");
    }

    #[test]
    fn output_ceiling_limits_peak() {
        let mut samples = vec![0.25f32, -0.5, 1.4, -1.2, 0.7];
        apply_output_ceiling(&mut samples, OUTPUT_CEILING);

        let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!(peak <= OUTPUT_CEILING + 1e-6, "peak={peak}");
    }

    #[test]
    fn tension_center_matches_pd_patch_behavior() {
        assert!((tension_center_hz(69.0) - 880.0).abs() < 1e-3);
        assert!((tension_center_hz(24.0) - 200.0).abs() < 1e-3);
        assert!((tension_center_hz(120.0) - 2000.0).abs() < 1e-3);
    }
}
