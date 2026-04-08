// FCPE ONNX pitch detector.
//
// This module provides F0 extraction for pitch analysis, replacing WORLD
// Harvest/DIO in clip-level pitch detection.

use ndarray::Array2;
use num_complex::Complex32;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use ort::value::TensorElementType;
use rustfft::FftPlanner;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static SHARED_SESSION: OnceLock<Arc<Mutex<Session>>> = OnceLock::new();
static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrtExecutionProviderChoice {
    Auto,
    Cpu,
    Cuda,
}

fn ensure_ort_init() -> Result<(), String> {
    match ORT_INIT.get_or_init(|| {
        ort::init().with_name("hifishifter").commit();
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

fn env_ep_choice() -> OrtExecutionProviderChoice {
    let v = std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .unwrap_or_else(|| "auto".to_string());
    match v.trim().to_ascii_lowercase().as_str() {
        "cpu" => OrtExecutionProviderChoice::Cpu,
        "cuda" => OrtExecutionProviderChoice::Cuda,
        _ => OrtExecutionProviderChoice::Auto,
    }
}

fn env_i32(name: &str) -> Option<i32> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
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

    None
}

fn resolve_model_path() -> Result<PathBuf, String> {
    if let Some(onnx) = env_path("HIFISHIFTER_FCPE_ONNX") {
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
    let mut builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    let choice = env_ep_choice();
    let device_id = env_i32("HIFISHIFTER_ORT_CUDA_DEVICE_ID").unwrap_or(0);

    match choice {
        OrtExecutionProviderChoice::Cpu => {}
        OrtExecutionProviderChoice::Cuda => {
            builder = builder
                .with_execution_providers([ep::CUDA::default().with_device_id(device_id).build()])
                .map_err(|e| format!("enable CUDA EP failed: {e}"))?;
        }
        OrtExecutionProviderChoice::Auto => {
            if let Ok(b) = builder
                .clone()
                .with_execution_providers([ep::CUDA::default().with_device_id(device_id).build()])
            {
                builder = b;
            }
        }
    }

    builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load FCPE onnx into ort session failed: {e}"))
}

fn get_or_init_shared_session() -> Result<Arc<Mutex<Session>>, String> {
    ensure_ort_init()?;
    let onnx_path = resolve_model_path()?;
    let session = build_session_with_ep(&onnx_path)?;
    let arc = Arc::new(Mutex::new(session));
    Ok(Arc::clone(SHARED_SESSION.get_or_init(|| Arc::clone(&arc))))
}

fn probe() -> &'static Result<(), String> {
    PROBE.get_or_init(|| get_or_init_shared_session().map(|_| ()))
}

pub fn is_available() -> bool {
    match probe() {
        Ok(()) => true,
        Err(e) => {
            if debug_enabled() && !LOGGED_UNAVAILABLE.swap(true, Ordering::Relaxed) {
                eprintln!("fcpe_onnx: unavailable: {e}");
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

fn env_fcpe_sr() -> u32 {
    std::env::var("HIFISHIFTER_FCPE_SAMPLE_RATE")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16_000)
}

fn env_fcpe_hop() -> usize {
    std::env::var("HIFISHIFTER_FCPE_HOP")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(160)
}

fn env_fcpe_n_fft() -> usize {
    std::env::var("HIFISHIFTER_FCPE_N_FFT")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1024)
}

fn env_fcpe_win() -> usize {
    std::env::var("HIFISHIFTER_FCPE_WIN_SIZE")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1024)
}

fn env_fcpe_fmin() -> f32 {
    std::env::var("HIFISHIFTER_FCPE_FMIN")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(30.0)
}

fn env_fcpe_fmax(sr: u32) -> f32 {
    std::env::var("HIFISHIFTER_FCPE_FMAX")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or((sr as f32) * 0.5)
}

fn linear_resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if input.is_empty() || in_rate == 0 || out_rate == 0 || in_rate == out_rate {
        return input.to_vec();
    }
    if input.len() < 2 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src = idx as f64 / ratio;
        let i0 = src.floor().max(0.0) as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = (src - i0 as f64) as f32;
        let a = input[i0];
        let b = input[i1];
        out.push(a + (b - a) * frac);
    }
    out
}

fn hann_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }

    let denom = (len - 1) as f32;
    (0..len)
        .map(|n| {
            let x = (2.0 * std::f32::consts::PI * n as f32) / denom;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn reflect_index(i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * ((len as isize) - 1);
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m < len as isize {
        m as usize
    } else {
        (period - m) as usize
    }
}

fn reflect_pad(y: &[f32], left: usize, right: usize) -> Vec<f32> {
    if y.is_empty() {
        return vec![0.0; left + right];
    }

    let len = y.len();
    let mut out = Vec::with_capacity(left + len + right);
    for i in -(left as isize)..0 {
        out.push(y[reflect_index(i, len)]);
    }
    out.extend_from_slice(y);
    for i in (len as isize)..((len as isize) + (right as isize)) {
        out.push(y[reflect_index(i, len)]);
    }
    out
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

fn mel_filterbank_slaney(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.max(fmin + 1.0));

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_min + (mel_max - mel_min) * t);
    }

    let mut hz_points = Vec::with_capacity(n_mels + 2);
    for &m in &mel_points {
        hz_points.push(mel_to_hz_slaney(m));
    }

    let mut fftfreqs = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        fftfreqs.push((i as f32) * (sr as f32) / (n_fft as f32));
    }

    let mut weights = Array2::<f32>::zeros((n_mels, n_freqs));
    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];
        let dl = (f_center - f_left).max(1e-6);
        let dr = (f_right - f_center).max(1e-6);

        for (i, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - f_left) / dl;
            let upper = (f_right - f) / dr;
            weights[[m, i]] = lower.min(upper).max(0.0);
        }

        let enorm = 2.0 / (f_right - f_left).max(1e-6);
        for i in 0..n_freqs {
            weights[[m, i]] *= enorm;
        }
    }
    weights
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

    let y = linear_resample_mono(waveform, in_sr, target_sr);
    let pad_left = ((win_size as isize - hop as isize) / 2).max(0) as usize;
    let pad_right = ((win_size as isize - hop as isize + 1) / 2).max(0) as usize;
    let y = reflect_pad(&y, pad_left, pad_right);
    if y.len() < win_size {
        return Ok((vec![(1e-9f32).ln(); n_mels], 1));
    }

    let n_frames = 1 + (y.len().saturating_sub(win_size)) / hop;
    let n_freqs = n_fft / 2 + 1;
    let window = hann_window(win_size);
    let fb = mel_filterbank_slaney(target_sr, n_fft, n_mels, fmin, fmax);

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
    f0_floor: f64,
    f0_ceil: f64,
) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    // Direct F0 output: [T], [1,T] or [B,T].
    let dims: &[i64] = &**shape;

    if dims.len() <= 2 {
        return data
            .iter()
            .map(|&v| if v.is_finite() && v > 0.0 { v as f64 } else { 0.0 })
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

        let min_hz = f0_floor.max(1.0);
        let max_hz = f0_ceil.max(min_hz + 1e-6);
        let log_span = (max_hz / min_hz).ln();

        let mut out = Vec::with_capacity(t);
        for ti in 0..t {
            let mut best_k = 0usize;
            let mut best_v = f32::NEG_INFINITY;

            for k in 0..c {
                let idx = if btc_layout {
                    // [B,T,C] for first batch only.
                    ti * c + k
                } else {
                    // [B,C,T] for first batch only.
                    k * t + ti
                };
                let v = data[idx];
                if v > best_v {
                    best_v = v;
                    best_k = k;
                }
            }

            let ratio = if c <= 1 {
                0.0
            } else {
                best_k as f64 / (c - 1) as f64
            };
            let hz = min_hz * (log_span * ratio).exp();
            out.push(hz);
        }
        return out;
    }

    data
        .iter()
        .map(|&v| if v.is_finite() && v > 0.0 { v as f64 } else { 0.0 })
        .collect()
}

fn tensor_rank_from_outlet(outlet: &ort::value::Outlet) -> usize {
    outlet
        .dtype()
        .tensor_shape()
        .map(|s| s.len())
        .unwrap_or(0)
}

fn build_waveform_tensor_for_rank(rank: usize, waveform: Vec<f32>) -> Result<Tensor<f32>, String> {
    match rank {
        1 => Tensor::from_array(([waveform.len()], waveform.into_boxed_slice()))
            .map_err(|e| format!("build FCPE input [T] failed: {e}")),
        2 => Tensor::from_array(([1usize, waveform.len()], waveform.into_boxed_slice()))
            .map_err(|e| format!("build FCPE input [1,T] failed: {e}")),
        _ => Tensor::from_array(([1usize, 1usize, waveform.len()], waveform.into_boxed_slice()))
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
        eprintln!("fcpe_onnx: model io = {io_summary}");
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

            let mel_axis = mel_shape
                .iter()
                .position(|&d| d == 128)
                .unwrap_or(2);
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
            let mel_axis = mel_shape
                .iter()
                .position(|&d| d == 128)
                .unwrap_or(2);
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

    let output_values = run_with_named_inputs(
        &mut session,
        &waveform,
        sample_rate,
        f0_floor,
        f0_ceil,
    )?;

    if output_values.is_empty() {
        return Ok(vec![0.0; target_frames]);
    }

    let resized = resample_f0_linear(&output_values, target_frames);
    Ok(sanitize_f0(resized, f0_floor, f0_ceil))
}
