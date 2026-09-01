// FCPE ONNX pitch detector.
//
// This module provides F0 extraction for pitch analysis, replacing WORLD
// Harvest/DIO in clip-level pitch detection.

use num_complex::Complex32;
use ort::session::Session;
use ort::value::Tensor;
use ort::value::TensorElementType;
use rustfft::FftPlanner;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// FCPE model frequency range — must match the model's training parameters.
/// Source: HachiTune FCPEPitchDetector.h (open-ai-tuning/HachiTune)
/// f0ToCent(32.7) → centToF0 for 360-bin output layer.
pub const FCPE_F0_MIN_HZ: f64 = 32.7;
pub const FCPE_F0_MAX_HZ: f64 = 1975.5;

/// Precomputed cent table matching HachiTune FCPEPitchDetector::initCentTable().
/// centTable[i] = cent(32.7) + (cent(1975.5) - cent(32.7)) * i / (n_bins - 1)
static CENT_TABLE: OnceLock<Vec<f64>> = OnceLock::new();

fn get_cent_table(n_bins: usize) -> &'static [f64] {
    CENT_TABLE.get_or_init(|| {
        let n = n_bins.max(2);
        let cent_min = 1200.0 * (FCPE_F0_MIN_HZ / 10.0).log2();
        let cent_max = 1200.0 * (FCPE_F0_MAX_HZ / 10.0).log2();
        let span = cent_max - cent_min;
        (0..n)
            .map(|i| cent_min + span * (i as f64) / ((n - 1) as f64))
            .collect()
    })
}

/// Convert cent to Hz (matching HachiTune FCPEPitchDetector::centToF0).
fn cent_to_hz(cent: f64) -> f64 {
    10.0 * (2.0f64).powf(cent / 1200.0)
}

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static SHARED_SESSION: OnceLock<Mutex<Option<Arc<Mutex<Session>>>>> = OnceLock::new();
static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

fn ensure_ort_init() -> Result<(), String> {
    match ORT_INIT.get_or_init(|| {
        // commit() returns false if the global ORT environment was already
        // committed by another module — that's fine, the active environment
        // remains valid. We still need to ensure the OrtEnv is created.
        ort::init().with_name("hifishifter").commit();

        if let Err(e) = ort::environment::Environment::current() {
            return Err(format!("failed to create ORT environment: {e}"));
        }
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

fn debug_enabled() -> bool {
    std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1")
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

fn default_model_guess() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let bundled = manifest
        .join("resources")
        .join("models")
        .join("fcpe")
        .join("fcpe.onnx");
    if bundled.is_file() {
        return Some(bundled);
    }

    let root_model = manifest.join("..").join("..").join("fcpe.onnx");
    if root_model.is_file() {
        return Some(root_model);
    }

    // 发布/便携环境：模型位于可执行文件同级的 models/ 目录中
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let p = exe_dir.join("models").join("fcpe").join("fcpe.onnx");
            if p.is_file() {
                return Some(p);
            }
        }
    }

    None
}

fn resolve_model_path() -> Result<PathBuf, String> {
    if let Some(onnx) = env_path("HIFISHIFTER_FCPE_ONNX") {
        return Ok(onnx);
    }

    if let Some(onnx) = crate::fcpe_onnx_path().map(|p| p.to_path_buf()) {
        return Ok(onnx);
    }

    if let Some(dir) = env_path("HIFISHIFTER_FCPE_MODEL_DIR") {
        let p = dir.join("fcpe.onnx");
        if p.is_file() {
            return Ok(p);
        }
    }

    default_model_guess().ok_or_else(|| {
        "FCPE ONNX model not found. Set HIFISHIFTER_FCPE_ONNX or HIFISHIFTER_FCPE_MODEL_DIR."
            .to_string()
    })
}

fn build_session_with_ep(onnx_path: &Path) -> Result<Session, String> {
    let (session, _ep) = crate::vocoder_ort_session::build_ort_session(
        onnx_path,
        crate::vocoder_ort_session::OrtSessionRole::PitchDetector,
    )?;
    Ok(session)
}

fn get_or_init_shared_session() -> Result<Arc<Mutex<Session>>, String> {
    let mutex = SHARED_SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = mutex
        .lock()
        .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?;
    if let Some(ref session) = *guard {
        return Ok(Arc::clone(session));
    }
    ensure_ort_init()?;
    let onnx_path = resolve_model_path()?;
    let session = build_session_with_ep(&onnx_path)?;
    let arc = Arc::new(Mutex::new(session));
    *guard = Some(Arc::clone(&arc));
    Ok(arc)
}

/// Drop the shared session to release GPU/CPU memory. Called on app exit.
pub fn drop_shared_session() {
    if let Some(mutex) = SHARED_SESSION.get() {
        for _ in 0..10 {
            if let Ok(mut guard) = mutex.try_lock() {
                *guard = None;
                log::error!("[fcpe] shared session dropped");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        log::error!("[fcpe] WARNING: could not acquire SHARED_SESSION lock at shutdown — giving up");
    }
}

/// Reset the shared session so the next inference rebuilds it with the
/// current EP choice (e.g. after user switches from WebGPU to DirectML).
pub fn update_ort_ep(_choice: &str, _device_id: Option<i32>) {
    crate::vocoder_ort_session::set_runtime_ep_override(Some(_choice.to_string()));
    crate::vocoder_ort_session::set_runtime_dml_device_id(_device_id);
    if let Some(mutex) = SHARED_SESSION.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = None;
        }
    }
}

fn probe() -> &'static Result<(), String> {
    PROBE.get_or_init(|| get_or_init_shared_session().map(|_| ()))
}

pub fn is_available() -> bool {
    match probe() {
        Ok(()) => true,
        Err(e) => {
            if debug_enabled() && !LOGGED_UNAVAILABLE.swap(true, Ordering::Relaxed) {
                log::warn!("fcpe_onnx: unavailable: {e}");
            }
            false
        }
    }
}

fn resample_f0_linear(values: &[f64], out_len: usize) -> Vec<f64> {
    if out_len == 0 {
        return Vec::new();
    }
    if values.is_empty() {
        return vec![0.0; out_len];
    }
    if values.len() == out_len {
        return values.to_vec();
    }
    if values.len() == 1 {
        return vec![values[0]; out_len];
    }
    if out_len == 1 {
        return vec![values[0]];
    }

    let in_len = values.len();
    let scale = (in_len - 1) as f64 / (out_len - 1) as f64;
    let mut out = vec![0.0f64; out_len];
    for (of, out_v) in out.iter_mut().enumerate() {
        let t_in = (of as f64) * scale;
        let i0 = t_in.floor() as usize;
        let i1 = (i0 + 1).min(in_len - 1);
        let frac = t_in - (i0 as f64);
        let a = values[i0];
        let b = values[i1];
        *out_v = a + (b - a) * frac;
    }
    out
}

fn sanitize_f0(mut f0: Vec<f64>, f0_floor: f64, f0_ceil: f64) -> Vec<f64> {
    let floor = f0_floor.max(1.0);
    let ceil = f0_ceil.max(floor);
    for v in &mut f0 {
        if !v.is_finite() || *v <= 0.0 {
            *v = 0.0;
            continue;
        }
        *v = v.clamp(floor, ceil);
    }
    f0
}

static CACHED_FCPE_SR: OnceLock<u32> = OnceLock::new();
static CACHED_FCPE_HOP: OnceLock<usize> = OnceLock::new();
static CACHED_FCPE_N_FFT: OnceLock<usize> = OnceLock::new();
static CACHED_FCPE_WIN: OnceLock<usize> = OnceLock::new();
static CACHED_FCPE_FMIN: OnceLock<f32> = OnceLock::new();
static CACHED_FCPE_FMAX: OnceLock<f32> = OnceLock::new();

fn env_fcpe_sr() -> u32 {
    *CACHED_FCPE_SR.get_or_init(|| {
        std::env::var("HIFISHIFTER_FCPE_SAMPLE_RATE")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(16_000)
    })
}

fn env_fcpe_hop() -> usize {
    *CACHED_FCPE_HOP.get_or_init(|| {
        std::env::var("HIFISHIFTER_FCPE_HOP")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(160)
    })
}

fn env_fcpe_n_fft() -> usize {
    *CACHED_FCPE_N_FFT.get_or_init(|| {
        std::env::var("HIFISHIFTER_FCPE_N_FFT")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024)
    })
}

fn env_fcpe_win() -> usize {
    *CACHED_FCPE_WIN.get_or_init(|| {
        std::env::var("HIFISHIFTER_FCPE_WIN_SIZE")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024)
    })
}

fn env_fcpe_fmin() -> f32 {
    *CACHED_FCPE_FMIN.get_or_init(|| {
        std::env::var("HIFISHIFTER_FCPE_FMIN")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0)
    })
}

fn env_fcpe_fmax(sr: u32) -> f32 {
    *CACHED_FCPE_FMAX.get_or_init(|| {
        std::env::var("HIFISHIFTER_FCPE_FMAX")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or((sr as f32) * 0.5)
    })
}

fn build_mel_from_waveform(
    waveform: &[f32],
    in_sr: u32,
    n_mels: usize,
) -> Result<(Vec<f32>, usize), String> {
    let target_sr = env_fcpe_sr();
    let hop = env_fcpe_hop();
    let n_fft = env_fcpe_n_fft();
    let win_size = env_fcpe_win();
    let fmin = env_fcpe_fmin();
    let fmax = env_fcpe_fmax(target_sr);

    if hop == 0 || n_fft == 0 || win_size == 0 || n_mels == 0 {
        return Err("fcpe mel config invalid".to_string());
    }

    let y = crate::mel_utils::linear_resample_mono(waveform, in_sr, target_sr);
    let pad_left = ((win_size as isize - hop as isize) / 2).max(0) as usize;
    let pad_right = ((win_size as isize - hop as isize + 1) / 2).max(0) as usize;
    let y = crate::mel_utils::reflect_pad(&y, pad_left, pad_right);
    if y.len() < win_size {
        return Ok((vec![(1e-9f32).ln(); n_mels], 1));
    }

    let n_frames = 1 + (y.len().saturating_sub(win_size)) / hop;
    let n_freqs = n_fft / 2 + 1;
    let window = crate::mel_utils::hann_window(win_size);
    let fb = crate::mel_utils::mel_filterbank_slaney(target_sr, n_fft, n_mels, fmin, fmax);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut spec = vec![0.0f32; n_freqs * n_frames];

    for frame in 0..n_frames {
        let start = frame * hop;
        for i in 0..win_size {
            fft_buf[i] = Complex32::new(y[start + i] * window[i], 0.0);
        }
        for c in &mut fft_buf[win_size..] {
            *c = Complex32::new(0.0, 0.0);
        }
        fft.process(&mut fft_buf);

        for f in 0..n_freqs {
            let c = fft_buf[f];
            spec[f * n_frames + frame] = (c.re * c.re + c.im * c.im).sqrt();
        }
    }

    let mut mel = vec![0.0f32; n_mels * n_frames];
    for m in 0..n_mels {
        for t in 0..n_frames {
            let mut acc = 0.0f32;
            for f in 0..n_freqs {
                acc += fb[[m, f]] * spec[f * n_frames + t];
            }
            mel[m * n_frames + t] = (acc.max(1e-9)).ln();
        }
    }

    Ok((mel, n_frames))
}

fn decode_model_output_to_f0_hz(
    shape: &ort::value::Shape,
    data: &[f32],
    _f0_floor: f64,
    _f0_ceil: f64,
) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    // Direct F0 output: [T], [1,T] or [B,T].
    let dims: &[i64] = &**shape;

    if dims.len() <= 2 {
        return data
            .iter()
            .map(|&v| {
                if v.is_finite() && v > 0.0 {
                    v as f64
                } else {
                    0.0
                }
            })
            .collect();
    }

    // Class/logit output (commonly 360 bins): [B,T,C] or [B,C,T].
    if dims.len() == 3 {
        let b = (dims[0].max(1)) as usize;
        let d1 = (dims[1].max(1)) as usize;
        let d2 = (dims[2].max(1)) as usize;

        let (t, c, btc_layout) = if d2 >= 64 {
            (d1, d2, true) // [B,T,C]
        } else if d1 >= 64 {
            (d2, d1, false) // [B,C,T]
        } else {
            (d1.max(d2), d1.min(d2).max(1), true)
        };

        if t == 0 || c == 0 {
            return Vec::new();
        }

        let total = b.saturating_mul(t).saturating_mul(c);
        if total == 0 || data.len() < total {
            return Vec::new();
        }

        // Precompute cent table matching HachiTune's initCentTable()
        let cent_table = get_cent_table(c);
        // HachiTune uses confidence threshold 0.05
        let threshold: f32 = 0.05;

        let mut out = Vec::with_capacity(t);
        for ti in 0..t {
            // Step 1: find global argmax (confidence check)
            let mut best_k = 0usize;
            let mut best_v = f32::NEG_INFINITY;

            for k in 0..c {
                let idx = if btc_layout { ti * c + k } else { k * t + ti };
                let v = data[idx];
                if v > best_v {
                    best_v = v;
                    best_k = k;
                }
            }

            // Confidence threshold (matching HachiTune)
            if best_v <= threshold {
                out.push(0.0);
                continue;
            }

            // Step 2: local weighted average in cent space (±4 bins)
            let local_start = best_k.saturating_sub(4);
            let local_end = (best_k + 4).min(c.saturating_sub(1));

            let mut weighted_sum = 0.0f64;
            let mut weight_sum = 0.0f64;

            for k in local_start..=local_end {
                let idx = if btc_layout { ti * c + k } else { k * t + ti };
                let v = data[idx] as f64;
                weighted_sum += cent_table[k] * v;
                weight_sum += v;
            }

            let hz = if weight_sum > 1e-9 {
                let cent = weighted_sum / weight_sum;
                cent_to_hz(cent)
            } else {
                0.0
            };
            out.push(hz);
        }
        return out;
    }

    data.iter()
        .map(|&v| {
            if v.is_finite() && v > 0.0 {
                v as f64
            } else {
                0.0
            }
        })
        .collect()
}

fn tensor_rank_from_outlet(outlet: &ort::value::Outlet) -> usize {
    outlet.dtype().tensor_shape().map(|s| s.len()).unwrap_or(0)
}

fn build_waveform_tensor_for_rank(rank: usize, waveform: Vec<f32>) -> Result<Tensor<f32>, String> {
    match rank {
        1 => Tensor::from_array(([waveform.len()], waveform.into_boxed_slice()))
            .map_err(|e| format!("build FCPE input [T] failed: {e}")),
        2 => Tensor::from_array(([1usize, waveform.len()], waveform.into_boxed_slice()))
            .map_err(|e| format!("build FCPE input [1,T] failed: {e}")),
        _ => Tensor::from_array((
            [1usize, 1usize, waveform.len()],
            waveform.into_boxed_slice(),
        ))
        .map_err(|e| format!("build FCPE input [1,1,T] failed: {e}")),
    }
}

fn run_with_named_inputs(
    session: &mut Session,
    waveform: &[f32],
    sample_rate: u32,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<Vec<f64>, String> {
    let input_meta: Vec<(String, usize, Option<TensorElementType>)> = session
        .inputs()
        .iter()
        .map(|o| {
            (
                o.name().to_string(),
                tensor_rank_from_outlet(o),
                o.dtype().tensor_type(),
            )
        })
        .collect();

    if input_meta.is_empty() {
        return Err("FCPE model has no inputs".to_string());
    }

    let io_summary = {
        let ins: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| {
                let ty = o
                    .dtype()
                    .tensor_type()
                    .map(|t| format!("{t:?}"))
                    .unwrap_or_else(|| "unknown".to_string());
                let shape = o
                    .dtype()
                    .tensor_shape()
                    .map(|s| format!("{:?}", &**s))
                    .unwrap_or_else(|| "[]".to_string());
                format!("{}:{ty}:{shape}", o.name())
            })
            .collect();
        let outs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| {
                let ty = o
                    .dtype()
                    .tensor_type()
                    .map(|t| format!("{t:?}"))
                    .unwrap_or_else(|| "unknown".to_string());
                let shape = o
                    .dtype()
                    .tensor_shape()
                    .map(|s| format!("{:?}", &**s))
                    .unwrap_or_else(|| "[]".to_string());
                format!("{}:{ty}:{shape}", o.name())
            })
            .collect();
        format!("inputs=[{}], outputs=[{}]", ins.join(", "), outs.join(", "))
    };

    if debug_enabled() {
        log::warn!("fcpe_onnx: model io = {io_summary}");
    }

    if input_meta.len() == 1 {
        let (first_name, rank, _) = &input_meta[0];
        if first_name.eq_ignore_ascii_case("mel") && *rank == 3 {
            let mel_shape: Vec<i64> = session
                .inputs()
                .get(0)
                .and_then(|o| o.dtype().tensor_shape())
                .map(|s| s.iter().copied().collect())
                .unwrap_or_else(|| vec![-1, -1, 128]);

            let mel_axis = mel_shape.iter().position(|&d| d == 128).unwrap_or(2);
            let n_mels = 128usize;

            let (mel, t) = build_mel_from_waveform(waveform, sample_rate, n_mels)
                .map_err(|e| format!("{e}; {io_summary}"))?;

            let mel_tensor = if mel_axis == 2 {
                // Model expects [B, T, M] where M=128.
                let mut btm = vec![0.0f32; t * n_mels];
                for m in 0..n_mels {
                    for ti in 0..t {
                        btm[ti * n_mels + m] = mel[m * t + ti];
                    }
                }
                Tensor::from_array(([1usize, t, n_mels], btm.into_boxed_slice()))
                    .map_err(|e| format!("build FCPE mel tensor [B,T,M] failed: {e}"))?
            } else {
                // Fallback to [B, M, T].
                Tensor::from_array(([1usize, n_mels, t], mel.into_boxed_slice()))
                    .map_err(|e| format!("build FCPE mel tensor [B,M,T] failed: {e}"))?
            };

            let outputs = session
                .run(ort::inputs![first_name.as_str() => mel_tensor])
                .map_err(|e| format!("FCPE run failed (mel-input): {e}; {io_summary}"))?;
            let first_out = outputs
                .into_iter()
                .next()
                .ok_or_else(|| "FCPE returned no outputs".to_string())?;
            let (_shape, data) = first_out
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract FCPE output failed: {e}"))?;
            let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
            return Ok(hz);
        }

        let audio = build_waveform_tensor_for_rank(*rank, waveform.to_vec())?;
        let outputs = session
            .run(ort::inputs![first_name.as_str() => audio])
            .map_err(|e| format!("FCPE run failed (single-input): {e}; {io_summary}"))?;
        let first_out = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "FCPE returned no outputs".to_string())?;
        let (_shape, data) = first_out
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract FCPE output failed: {e}"))?;
        let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
        return Ok(hz);
    }

    if input_meta.len() == 2 {
        let (first_name, first_rank, _) = &input_meta[0];
        let (second_name, _, second_ty) = &input_meta[1];

        if first_name.eq_ignore_ascii_case("mel") && *first_rank == 3 {
            let mel_shape: Vec<i64> = session
                .inputs()
                .get(0)
                .and_then(|o| o.dtype().tensor_shape())
                .map(|s| s.iter().copied().collect())
                .unwrap_or_else(|| vec![-1, -1, 128]);
            let mel_axis = mel_shape.iter().position(|&d| d == 128).unwrap_or(2);
            let n_mels = 128usize;

            let (mel, t) = build_mel_from_waveform(waveform, sample_rate, n_mels)
                .map_err(|e| format!("{e}; {io_summary}"))?;

            let mel_tensor = if mel_axis == 2 {
                let mut btm = vec![0.0f32; t * n_mels];
                for m in 0..n_mels {
                    for ti in 0..t {
                        btm[ti * n_mels + m] = mel[m * t + ti];
                    }
                }
                Tensor::from_array(([1usize, t, n_mels], btm.into_boxed_slice()))
                    .map_err(|e| format!("build FCPE mel tensor [B,T,M] failed: {e}"))?
            } else {
                Tensor::from_array(([1usize, n_mels, t], mel.into_boxed_slice()))
                    .map_err(|e| format!("build FCPE mel tensor [B,M,T] failed: {e}"))?
            };

            match second_ty {
                Some(TensorElementType::Int64) => {
                    let sr = Tensor::from_array(((), vec![env_fcpe_sr() as i64]))
                        .map_err(|e| format!("build FCPE sr(int64) failed: {e}"))?;
                    let outputs = session
                        .run(ort::inputs![first_name.as_str() => mel_tensor, second_name.as_str() => sr])
                        .map_err(|e| format!("FCPE run failed (mel+sr:int64): {e}; {io_summary}"))?;
                    let first_out = outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| "FCPE returned no outputs".to_string())?;
                    let (_shape, data) = first_out
                        .1
                        .try_extract_tensor::<f32>()
                        .map_err(|e| format!("extract FCPE output failed: {e}"))?;
                    let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
                    return Ok(hz);
                }
                Some(TensorElementType::Int32) => {
                    let sr = Tensor::from_array(((), vec![env_fcpe_sr() as i32]))
                        .map_err(|e| format!("build FCPE sr(int32) failed: {e}"))?;
                    let outputs = session
                        .run(ort::inputs![first_name.as_str() => mel_tensor, second_name.as_str() => sr])
                        .map_err(|e| format!("FCPE run failed (mel+sr:int32): {e}; {io_summary}"))?;
                    let first_out = outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| "FCPE returned no outputs".to_string())?;
                    let (_shape, data) = first_out
                        .1
                        .try_extract_tensor::<f32>()
                        .map_err(|e| format!("extract FCPE output failed: {e}"))?;
                    let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
                    return Ok(hz);
                }
                _ => {
                    let aux = Tensor::from_array(((), vec![0.0f32]))
                        .map_err(|e| format!("build FCPE aux(float32) failed: {e}"))?;
                    let outputs = session
                        .run(ort::inputs![first_name.as_str() => mel_tensor, second_name.as_str() => aux])
                        .map_err(|e| format!("FCPE run failed (mel+aux): {e}; {io_summary}"))?;
                    let first_out = outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| "FCPE returned no outputs".to_string())?;
                    let (_shape, data) = first_out
                        .1
                        .try_extract_tensor::<f32>()
                        .map_err(|e| format!("extract FCPE output failed: {e}"))?;
                    let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
                    return Ok(hz);
                }
            }
        }

        let audio = build_waveform_tensor_for_rank(*first_rank, waveform.to_vec())?;

        // Common FCPE exports use (audio, sr) where sr is int scalar/tensor.
        match second_ty {
            Some(TensorElementType::Int64) => {
                let sr = Tensor::from_array(((), vec![sample_rate as i64]))
                    .map_err(|e| format!("build FCPE sr(int64) failed: {e}"))?;
                let outputs = session
                    .run(ort::inputs![first_name.as_str() => audio, second_name.as_str() => sr])
                    .map_err(|e| format!("FCPE run failed (audio+sr:int64): {e}; {io_summary}"))?;
                let first_out = outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| "FCPE returned no outputs".to_string())?;
                let (_shape, data) = first_out
                    .1
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("extract FCPE output failed: {e}"))?;
                let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
                return Ok(hz);
            }
            Some(TensorElementType::Int32) => {
                let sr = Tensor::from_array(((), vec![sample_rate as i32]))
                    .map_err(|e| format!("build FCPE sr(int32) failed: {e}"))?;
                let outputs = session
                    .run(ort::inputs![first_name.as_str() => audio, second_name.as_str() => sr])
                    .map_err(|e| format!("FCPE run failed (audio+sr:int32): {e}; {io_summary}"))?;
                let first_out = outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| "FCPE returned no outputs".to_string())?;
                let (_shape, data) = first_out
                    .1
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("extract FCPE output failed: {e}"))?;
                let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
                return Ok(hz);
            }
            _ => {
                // Fallback: pass zero scalar as second input for models expecting threshold/config.
                let aux = Tensor::from_array(((), vec![0.0f32]))
                    .map_err(|e| format!("build FCPE aux(float32) failed: {e}"))?;
                let outputs = session
                    .run(ort::inputs![first_name.as_str() => audio, second_name.as_str() => aux])
                    .map_err(|e| format!("FCPE run failed (audio+aux): {e}; {io_summary}"))?;
                let first_out = outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| "FCPE returned no outputs".to_string())?;
                let (_shape, data) = first_out
                    .1
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("extract FCPE output failed: {e}"))?;
                let hz = decode_model_output_to_f0_hz(_shape, data, f0_floor, f0_ceil);
                return Ok(hz);
            }
        }
    }

    Err(format!(
        "Unsupported FCPE input arity: {} (expected 1 or 2); {}",
        input_meta.len(),
        io_summary
    ))
}

pub fn infer_f0_hz(
    mono: &[f64],
    sample_rate: u32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<Vec<f64>, String> {
    if mono.is_empty() {
        return Ok(Vec::new());
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let target_frames = ((mono.len() as f64) / (sample_rate.max(1) as f64) * 1000.0 / fp)
        .round()
        .max(1.0) as usize;

    let waveform: Vec<f32> = mono.iter().map(|&v| v as f32).collect();
    let shared = get_or_init_shared_session()?;
    let mut session = shared
        .lock()
        .map_err(|e| format!("FCPE session lock poisoned: {e}"))?;

    let output_values =
        run_with_named_inputs(&mut session, &waveform, sample_rate, f0_floor, f0_ceil)?;

    if output_values.is_empty() {
        return Ok(vec![0.0; target_frames]);
    }

    let resized = resample_f0_linear(&output_values, target_frames);
    Ok(sanitize_f0(resized, f0_floor, f0_ceil))
}
