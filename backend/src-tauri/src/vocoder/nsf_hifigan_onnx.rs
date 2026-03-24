use ndarray::Array2;
use num_complex::Complex32;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::Fft;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_ort_init() -> Result<(), String> {
    match ORT_INIT.get_or_init(|| {
        ort::init().with_name("hifishifter").commit();
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrtExecutionProviderChoice {
    Auto,
    Cpu,
    Cuda,
}

fn env_ep_choice() -> OrtExecutionProviderChoice {
    let v = std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .unwrap_or_else(|| "auto".to_string());
    match v.trim().to_ascii_lowercase().as_str() {
        "cpu" => OrtExecutionProviderChoice::Cpu,
        "cuda" => OrtExecutionProviderChoice::Cuda,
        "auto" | "" => OrtExecutionProviderChoice::Auto,
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

fn build_session_with_ep(onnx_path: &Path) -> Result<Session, String> {
    let mut builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    let choice = env_ep_choice();
    let device_id = env_i32("HIFISHIFTER_ORT_CUDA_DEVICE_ID").unwrap_or(0);

    let selected: &str;

    match choice {
        OrtExecutionProviderChoice::Cpu => {
            // Default session uses CPU EP.
            selected = "cpu";
        }
        OrtExecutionProviderChoice::Cuda => {
            builder = builder
                .with_execution_providers([ep::CUDA::default().with_device_id(device_id).build()])
                .map_err(|e| format!("enable CUDA EP failed: {e}"))?;
            selected = "cuda";
        }
        OrtExecutionProviderChoice::Auto => {
            // Try CUDA first, then fall back to CPU.
            match builder
                .clone()
                .with_execution_providers([ep::CUDA::default().with_device_id(device_id).build()])
            {
                Ok(b) => {
                    builder = b;
                    selected = "cuda";
                }
                Err(e) => {
                    if debug_enabled() {
                        eprintln!(
                            "nsf_hifigan_onnx: CUDA EP unavailable, falling back to CPU: {e}"
                        );
                    }
                    selected = "cpu";
                }
            }
        }
    }

    if debug_enabled() {
        eprintln!("nsf_hifigan_onnx: ort ep={selected} (HIFISHIFTER_ORT_EP={:?}, cuda_device_id={device_id})", choice);
    }

    builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load onnx into ort session failed: {e}"))
}

#[derive(Debug, Clone, Deserialize)]
struct NsfHifiganConfig {
    sampling_rate: u32,
    num_mels: usize,
    hop_size: usize,
    n_fft: usize,
    win_size: usize,
    fmin: f32,
    fmax: f32,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

fn default_model_dir_guess() -> Option<PathBuf> {
    // 开发环境：模型位于 CARGO_MANIFEST_DIR/resources/models/nsf_hifigan/
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .join("resources")
        .join("models")
        .join("nsf_hifigan");
    if p.join("pc_nsf_hifigan.onnx").is_file() && p.join("config.json").is_file() {
        return Some(p);
    }

    None
}

fn resolve_model_paths() -> Result<(PathBuf, PathBuf), String> {
    // Returns (onnx_path, config_path)
    if let Some(onnx) = env_path("HIFISHIFTER_NSF_HIFIGAN_ONNX") {
        let dir = onnx.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let cfg = env_path("HIFISHIFTER_NSF_HIFIGAN_CONFIG")
            .or_else(|| {
                let p = dir.join("config.json");
                if p.is_file() {
                    Some(p)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| dir.join("config.json"));
        return Ok((onnx, cfg));
    }

    if let Some(dir) =
        env_path("HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR").or_else(default_model_dir_guess)
    {
        let onnx = dir.join("pc_nsf_hifigan.onnx");
        let cfg = dir.join("config.json");
        if onnx.is_file() && cfg.is_file() {
            return Ok((onnx, cfg));
        }
    }

    Err(
        "NSF-HiFiGAN ONNX model not found. Set HIFISHIFTER_NSF_HIFIGAN_ONNX (or HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR)."
            .to_string(),
    )
}

fn read_config(path: &Path) -> Result<NsfHifiganConfig, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read config.json failed: {e}"))?;
    serde_json::from_slice::<NsfHifiganConfig>(&bytes)
        .map_err(|e| format!("parse config.json failed: {e}"))
}

pub(crate) fn probe_load() -> Result<String, String> {
    ensure_ort_init()?;
    let (onnx_path, cfg_path) = resolve_model_paths()?;
    let cfg = read_config(&cfg_path)?;

    // Create a session (this also validates that the model is loadable by ORT).
    let mut session = build_session_with_ep(&onnx_path)?;

    // Best-effort smoke run to ensure inputs/outputs are compatible.
    // Model expects mel: (1, n_mels, T) and f0: (1, T).
    let t = 10usize;
    let mel = vec![0.0f32; cfg.num_mels.saturating_mul(t)];
    let f0 = vec![0.0f32; t];
    let mel_tensor = Tensor::from_array(([1usize, cfg.num_mels, t], mel.into_boxed_slice()))
        .map_err(|e| format!("build mel tensor failed: {e}"))?;
    let f0_tensor = Tensor::from_array(([1usize, t], f0.into_boxed_slice()))
        .map_err(|e| format!("build f0 tensor failed: {e}"))?;
    let outputs = session
        .run(ort::inputs![mel_tensor, f0_tensor])
        .map_err(|e| format!("ort session run failed: {e}"))?;
    let output0 = outputs
        .into_iter()
        .next()
        .ok_or_else(|| "ort returned no outputs".to_string())?;
    let (_shape, data) = output0
        .1
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort output extract failed: {e}"))?;
    if data.is_empty() {
        return Err("ort output tensor is empty".to_string());
    }

    Ok(format!(
        "nsf_hifigan_onnx: OK\n  onnx: {}\n  cfg: {}\n  sr={} mels={} hop={} n_fft={} win={} fmin={} fmax={}",
        onnx_path.display(),
        cfg_path.display(),
        cfg.sampling_rate,
        cfg.num_mels,
        cfg.hop_size,
        cfg.n_fft,
        cfg.win_size,
        cfg.fmin,
        cfg.fmax
    ))
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

fn reflect_pad_into(y: &[f32], left: usize, right: usize, out: &mut Vec<f32>) {
    out.clear();
    if y.is_empty() {
        out.resize(left + right, 0.0);
        return;
    }
    let len = y.len();
    out.reserve(left + len + right);
    for i in -(left as isize)..0 {
        out.push(y[reflect_index(i, len)]);
    }
    // 中间主体数据直接内存拷贝
    out.extend_from_slice(y);
    for i in (len as isize)..((len as isize) + (right as isize)) {
        out.push(y[reflect_index(i, len)]);
    }
}

fn hann_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return vec![];
    }
    if len == 1 {
        return vec![1.0];
    }

    let denom = (len - 1) as f32;
    let mut w = Vec::with_capacity(len);
    for n in 0..len {
        let x = (2.0 * std::f32::consts::PI * (n as f32)) / denom;
        w.push(0.5 - 0.5 * x.cos());
    }
    w
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
    let mel_max = hz_to_mel_slaney(fmax.max(fmin));

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

        let fdiff_left = (f_center - f_left).max(1e-6);
        let fdiff_right = (f_right - f_center).max(1e-6);

        for (i, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - f_left) / fdiff_left;
            let upper = (f_right - f) / fdiff_right;
            weights[[m, i]] = lower.min(upper).max(0.0);
        }

        // Slaney normalization.
        let enorm = 2.0 / (f_right - f_left).max(1e-6);
        for i in 0..n_freqs {
            weights[[m, i]] *= enorm;
        }
    }

    weights
}

#[allow(dead_code)]
fn stft_magnitude(
    y: &[f32],
    n_fft: usize,
    win_size: usize,
    hop: usize,
    window: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    if win_size == 0 || hop == 0 || n_fft == 0 {
        return Err("stft: invalid params".to_string());
    }
    if window.len() != win_size {
        return Err("stft: window length mismatch".to_string());
    }

    let n_freqs = n_fft / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    if y.len() < win_size {
        return Ok(vec![vec![0.0; 1]; n_freqs]);
    }

    let n_frames = 1 + (y.len().saturating_sub(win_size)) / hop;
    let mut out = vec![vec![0.0f32; n_frames]; n_freqs];

    let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_fft];

    for frame in 0..n_frames {
        let start = frame * hop;
        let windowed = &y[start..start + win_size];
        for (buf_c, (&v, &win)) in buf[..win_size].iter_mut().zip(windowed.iter().zip(window)) {
            *buf_c = Complex32::new(v * win, 0.0);
        }
        buf[win_size..n_fft].fill(Complex32::new(0.0, 0.0));

        fft.process(&mut buf);

        for f in 0..n_freqs {
            let c = buf[f];
            out[f][frame] = (c.re * c.re + c.im * c.im).sqrt();
        }
    }

    Ok(out)
}

fn dynamic_range_compression_ln(x: f32) -> f32 {
    (x.max(1e-9)).ln()
}

fn midi_to_hz(midi: f64) -> f32 {
    if !(midi.is_finite() && midi > 0.0) {
        return 0.0;
    }
    let hz = 440.0 * (2.0f64).powf((midi - 69.0) / 12.0);
    if hz.is_finite() {
        hz as f32
    } else {
        0.0
    }
}

fn linear_resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    if in_rate == out_rate {
        return input.to_vec();
    }
    if input.len() < 2 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_frames = ((input.len() as f64) * ratio).round().max(1.0) as usize;

    // 利用 collect() 直接分配好容量并写入，消除内存开销
    (0..out_frames)
        .map(|of| {
            let t_in = (of as f64) / ratio;
            let i0 = t_in.floor() as isize;
            let frac = (t_in - (i0 as f64)) as f32;
            let i0 = i0.clamp(0, (input.len() - 1) as isize) as usize;
            let i1 = (i0 + 1).min(input.len() - 1);
            let a = input[i0];
            let b = input[i1];
            a + (b - a) * frac
        })
        .collect()
}

fn linear_resample_mono_into(input: &[f32], in_rate: u32, out_rate: u32, out: &mut Vec<f32>) {
    out.clear();
    if input.is_empty() {
        return;
    }

    if in_rate == out_rate || input.len() < 2 {
        out.extend_from_slice(input);
        return;
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_frames = ((input.len() as f64) * ratio).round().max(1.0) as usize;

    // 利用 extend() 推入缓冲，消除 resize(0.0) 的 memset 填零损耗
    out.extend((0..out_frames).map(|of| {
        let t_in = (of as f64) / ratio;
        let i0 = t_in.floor() as isize;
        let frac = (t_in - (i0 as f64)) as f32;
        let i0 = i0.clamp(0, (input.len() - 1) as isize) as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let a = input[i0];
        let b = input[i1];
        a + (b - a) * frac
    }));
}

/// 进程级全局共享的 ORT Session。
/// ORT Session::run() 需要 &mut self，因此用 Mutex 保护。
static SHARED_SESSION: OnceLock<Arc<Mutex<Session>>> = OnceLock::new();

/// 初始化（或获取已有的）全局 Session。
fn get_or_init_shared_session() -> Result<Arc<Mutex<Session>>, String> {
    if let Some(s) = SHARED_SESSION.get() {
        return Ok(Arc::clone(s));
    }
    ensure_ort_init()?;
    let (onnx_path, _cfg_path) = resolve_model_paths()?;
    let session = build_session_with_ep(&onnx_path)?;
    let arc = Arc::new(Mutex::new(session));
    // get_or_init 保证只有一个线程真正写入，其余线程拿到同一个 Arc。
    Ok(Arc::clone(SHARED_SESSION.get_or_init(|| Arc::clone(&arc))))
}

pub struct NsfHifiganOnnx {
    cfg: NsfHifiganConfig,
    /// Mel 滤波器组矩阵，shape: [n_mels, n_freqs]，预计算后只读。
    mel_fb_matrix: Array2<f32>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_buf: Vec<Complex32>,
    pad_buf: Vec<f32>,
    audio_resample_buf: Vec<f32>,
    /// 共享的 ORT Session，Arc<Mutex<>> 保证多线程安全复用。
    session: Arc<Mutex<Session>>,
}

impl NsfHifiganOnnx {
    fn load() -> Result<Self, String> {
        let (_onnx_path, cfg_path) = resolve_model_paths()?;
        let cfg = read_config(&cfg_path)?;

        if cfg.sampling_rate == 0 || cfg.num_mels == 0 || cfg.hop_size == 0 || cfg.n_fft == 0 {
            return Err("invalid NSF-HiFiGAN config.json".to_string());
        }

        // 获取（或初始化）全局共享 Session，消除每线程冷启动。
        let session = get_or_init_shared_session()?;

        let mel_fb_matrix = mel_filterbank_slaney(
            cfg.sampling_rate,
            cfg.n_fft,
            cfg.num_mels,
            cfg.fmin,
            cfg.fmax,
        );

        let window = hann_window(cfg.win_size);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        let fft_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); cfg.n_fft];

        Ok(Self {
            cfg,
            mel_fb_matrix,
            window,
            fft,
            fft_buf,
            pad_buf: Vec::new(),
            audio_resample_buf: Vec::new(),
            session,
        })
    }

    fn mel_from_audio_fast(&mut self, audio: &[f32]) -> Result<Vec<f32>, String> {
        let hop = self.cfg.hop_size;
        let win_size = self.cfg.win_size;
        let n_fft = self.cfg.n_fft;

        if win_size == 0 || hop == 0 || n_fft == 0 {
            return Err("mel: invalid config".to_string());
        }
        if self.window.len() != win_size {
            return Err("mel: window length mismatch".to_string());
        }
        if self.fft_buf.len() != n_fft {
            return Err("mel: fft buffer length mismatch".to_string());
        }

        let pad_left = ((win_size as isize - hop as isize) / 2).max(0) as usize;
        let pad_right = ((win_size as isize - hop as isize + 1) / 2).max(0) as usize;
        reflect_pad_into(audio, pad_left, pad_right, &mut self.pad_buf);
        let y: &[f32] = self.pad_buf.as_slice();

        let n_freqs = n_fft / 2 + 1;

        if y.len() < win_size {
            // 空音频：返回全零（经 log 压缩后为 ln(1e-9)）的 mel 矩阵。
            let n_frames = 1usize;
            let fill = dynamic_range_compression_ln(0.0);
            return Ok(vec![fill; self.cfg.num_mels * n_frames]);
        }

        let n_frames = 1 + (y.len().saturating_sub(win_size)) / hop;

        // 将所有帧的幅度谱累积为矩阵 mag_matrix: [n_freqs, n_frames]，
        // 然后用一次矩阵乘法替代双重循环，利用 SIMD 自动向量化。
        let mut mag_matrix = Array2::<f32>::zeros((n_freqs, n_frames));

        for frame in 0..n_frames {
            let start = frame * hop;

            let windowed = &y[start..start + win_size];
            for (buf_c, (&v, &win)) in self.fft_buf[..win_size]
                .iter_mut()
                .zip(windowed.iter().zip(&self.window))
            {
                *buf_c = Complex32::new(v * win, 0.0);
            }
            self.fft_buf[win_size..n_fft].fill(Complex32::new(0.0, 0.0));

            self.fft.process(&mut self.fft_buf);

            for f in 0..n_freqs {
                let c = self.fft_buf[f];
                mag_matrix[[f, frame]] = (c.re * c.re + c.im * c.im).sqrt();
            }
        }

        // 对每个元素应用动态范围压缩，并展平为 [n_mels * n_frames] 的 Vec<f32>。
        let mel: Vec<f32> = self
            .mel_fb_matrix
            .dot(&mag_matrix)
            .into_iter()
            .map(|v| dynamic_range_compression_ln(v))
            .collect();

        Ok(mel)
    }

    #[allow(dead_code)]
    fn mel_from_audio(&self, audio: &[f32], key_shift_semitones: f32) -> Result<Vec<f32>, String> {
        // Replicates utils/wav2mel.py (PitchAdjustableMelSpectrogram + log compression),
        // but we currently only use key_shift=0 in the app.
        let factor = 2.0f32.powf(key_shift_semitones / 12.0);
        let n_fft_new = ((self.cfg.n_fft as f32) * factor).round().max(1.0) as usize;
        let win_size_new = ((self.cfg.win_size as f32) * factor).round().max(1.0) as usize;
        let hop = self.cfg.hop_size;

        let pad_left = ((win_size_new as isize - hop as isize) / 2).max(0) as usize;
        let pad_right = ((win_size_new as isize - hop as isize + 1) / 2).max(0) as usize;
        let y = reflect_pad(audio, pad_left, pad_right);

        let window = hann_window(win_size_new);
        let mut spec = stft_magnitude(&y, n_fft_new, win_size_new, hop, &window)?;

        // Handle pitch shift by resizing frequency bins (python behavior).
        if key_shift_semitones.abs() > 1e-6 {
            let size = self.cfg.n_fft / 2 + 1;
            let resize = spec.len();
            if resize < size {
                spec.extend(std::iter::repeat(vec![0.0f32; spec[0].len()]).take(size - resize));
            }
            spec.truncate(size);
            let scale = (self.cfg.win_size as f32) / (win_size_new as f32);
            for row in &mut spec {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Mel projection.
        let n_freqs = self.cfg.n_fft / 2 + 1;
        if spec.len() != n_freqs {
            return Err(format!(
                "mel: unexpected spec bins (got {}, expected {})",
                spec.len(),
                n_freqs
            ));
        }
        let n_frames = spec[0].len();
        // 将 spec（Vec<Vec<f32>>，[n_freqs][n_frames]）转为 Array2 后做矩阵乘法。
        let mut mag_matrix = Array2::<f32>::zeros((n_freqs, n_frames));
        for f in 0..n_freqs {
            for t in 0..n_frames {
                mag_matrix[[f, t]] = spec[f][t];
            }
        }
        let mel_result = self.mel_fb_matrix.dot(&mag_matrix);
        let mel: Vec<f32> = mel_result
            .iter()
            .map(|&v| dynamic_range_compression_ln(v))
            .collect();
        Ok(mel)
    }

    fn env_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
    }

    fn run_model(&mut self, mel: Vec<f32>, f0: Vec<f32>, t: usize) -> Result<Vec<f32>, String> {
        let mel_tensor =
            Tensor::from_array(([1usize, self.cfg.num_mels, t], mel.into_boxed_slice()))
                .map_err(|e| format!("build mel tensor failed: {e}"))?;
        let f0_tensor = Tensor::from_array(([1usize, t], f0.into_boxed_slice()))
            .map_err(|e| format!("build f0 tensor failed: {e}"))?;

        // 通过 Mutex 获取 &mut Session 来调用 run()。
        // 用块作用域限制 guard 的生命周期，确保 lock 尽快释放。
        let result: Vec<f32> = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| format!("ort session lock poisoned: {e}"))?;
            let outputs = session_guard
                .run(ort::inputs![mel_tensor, f0_tensor])
                .map_err(|e| format!("ort run failed: {e}"))?;
            let output0 = outputs
                .into_iter()
                .next()
                .ok_or_else(|| "onnx returned no outputs".to_string())?;
            let (_shape, data) = output0
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("ort output type mismatch: {e}"))?;
            data.to_vec()
        };
        Ok(result)
    }

    pub fn infer_from_audio_and_midi(
        &mut self,
        audio_mono: &[f32],
        sample_rate: u32,
        start_sec: f64,
        midi_at_time: impl Fn(f64) -> f64,
        formant_shift_at_time: impl Fn(f64) -> f32,
    ) -> Result<Vec<f32>, String> {
        let model_sr = self.cfg.sampling_rate;

        // 利用 std::mem::take 绕过借用冲突，实现 0 拷贝缓冲区复用
        let mut mel = if sample_rate == model_sr {
            self.mel_from_audio_fast(audio_mono)?
        } else {
            let mut resample_buf = std::mem::take(&mut self.audio_resample_buf);
            linear_resample_mono_into(audio_mono, sample_rate, model_sr, &mut resample_buf);
            let mel_result = self.mel_from_audio_fast(&resample_buf);
            self.audio_resample_buf = resample_buf; // 将容量归还给 self 以供下次复用
            mel_result?
        };

        // mel is stored as (n_mels, T) contiguous. Build f0 (1, T) in Hz.
        let t = mel.len() / self.cfg.num_mels;
        if t == 0 {
            return Ok(vec![0.0; audio_mono.len()]);
        }

        // 应用共振峰偏移（在 mel 域沿频率轴做线性插值）
        let hop_sec = (self.cfg.hop_size as f64) / (model_sr.max(1) as f64);
        let formant_shifts: Vec<f32> = (0..t)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                formant_shift_at_time(abs_t)
            })
            .collect();
        let has_formant_shift = formant_shifts.iter().any(|s| s.abs() >= 0.5);
        if has_formant_shift {
            shift_mel_formant(
                &mut mel,
                self.cfg.num_mels,
                t,
                &formant_shifts,
                self.cfg.fmin,
                self.cfg.fmax,
            );
        }

        let f0: Vec<f32> = (0..t)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                midi_to_hz(midi_at_time(abs_t))
            })
            .collect();

        // Optional experimental segmented inference (stream-like).
        // This can reduce peak latency / memory for very long buffers at the cost of extra overlap compute.
        // Disabled by default to avoid boundary artifacts.
        let seg_frames = Self::env_usize("HIFISHIFTER_NSF_HIFIGAN_SEGMENT_FRAMES").unwrap_or(0);
        let overlap_frames = Self::env_usize("HIFISHIFTER_NSF_HIFIGAN_OVERLAP_FRAMES").unwrap_or(8);

        let y_vec: Vec<f32> = if seg_frames >= 16 && t > seg_frames {
            let overlap_frames = overlap_frames.min(seg_frames.saturating_sub(1));
            let step = seg_frames.saturating_sub(overlap_frames).max(1);

            let expected_total = (t as usize).saturating_mul(self.cfg.hop_size).max(1);
            let mut out = vec![0.0f32; expected_total];
            let mut wsum = vec![0.0f32; expected_total];

            let mut s = 0usize;
            while s < t {
                let end = (s + seg_frames).min(t);
                let seg_t = end.saturating_sub(s).max(1);

                // Slice mel: original layout is (n_mels, T)
                let mut mel_seg = vec![0.0f32; self.cfg.num_mels * seg_t];
                for m in 0..self.cfg.num_mels {
                    let src = &mel[m * t + s..m * t + end];
                    let dst = &mut mel_seg[m * seg_t..(m + 1) * seg_t];
                    dst.copy_from_slice(src);
                }
                let f0_seg = f0[s..end].to_vec();

                let y_seg = self.run_model(mel_seg, f0_seg, seg_t)?;
                let seg_expected = seg_t.saturating_mul(self.cfg.hop_size).max(1);
                let seg_samples = y_seg.len().min(seg_expected);

                let overlap_samples = overlap_frames.saturating_mul(self.cfg.hop_size);
                let base = s.saturating_mul(self.cfg.hop_size);

                for i in 0..seg_samples {
                    let g = base + i;
                    if g >= out.len() {
                        break;
                    }
                    let mut w = 1.0f32;
                    if overlap_samples > 0 {
                        if s > 0 && i < overlap_samples {
                            w = (i as f32) / (overlap_samples as f32);
                        }
                        if end < t && seg_samples > overlap_samples {
                            let tail = seg_samples.saturating_sub(1).saturating_sub(i);
                            if tail < overlap_samples {
                                let w_out = (tail as f32) / (overlap_samples as f32);
                                w = w.min(w_out);
                            }
                        }
                    }

                    out[g] += y_seg[i] * w;
                    wsum[g] += w;
                }

                if end >= t {
                    break;
                }
                s += step;
            }

            for i in 0..out.len() {
                let w = wsum[i];
                if w > 1e-6 {
                    out[i] /= w;
                }
            }
            out
        } else {
            self.run_model(mel, f0, t)?
        };

        // Resample back to mixdown rate if needed.
        let mut out = if model_sr == sample_rate {
            y_vec
        } else {
            linear_resample_mono(&y_vec, model_sr, sample_rate)
        };

        // Force length to match input buffer for in-place mixdown.
        let target = audio_mono.len();
        if out.len() > target {
            out.truncate(target);
        } else if out.len() < target {
            out.resize(target, 0.0);
        }
        Ok(out)
    }
}

static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TLS_SESSION: RefCell<Option<Result<NsfHifiganOnnx, String>>> = RefCell::new(None);
}

fn probe() -> &'static Result<(), String> {
    PROBE.get_or_init(|| {
        // probe() 同时触发 SHARED_SESSION 的初始化，
        // 确保后续 TLS load() 直接复用，不再重复加载 ONNX 文件。
        get_or_init_shared_session().map(|_| ())
    })
}

pub fn is_available() -> bool {
    match probe() {
        Ok(()) => true,
        Err(e) => {
            let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
            if debug && !LOGGED_UNAVAILABLE.swap(true, Ordering::Relaxed) {
                eprintln!("nsf_hifigan_onnx: unavailable: {e}");
            }
            false
        }
    }
}

pub fn infer_pitch_edit_mono(
    audio_mono: &[f32],
    sample_rate: u32,
    start_sec: f64,
    midi_at_time: impl Fn(f64) -> f64,
    formant_shift_at_time: impl Fn(f64) -> f32,
) -> Result<Vec<f32>, String> {
    if let Err(e) = probe() {
        return Err(e.clone());
    }

    TLS_SESSION.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(NsfHifiganOnnx::load());
        }
        let sess = opt
            .as_mut()
            .expect("TLS_SESSION just initialized")
            .as_mut()
            .map_err(|e| e.clone())?;

        sess.infer_from_audio_and_midi(
            audio_mono,
            sample_rate,
            start_sec,
            midi_at_time,
            formant_shift_at_time,
        )
    })
}

// Helper functions for diagnostics
pub fn compiled() -> bool {
    true
}

pub fn model_load_error() -> Option<String> {
    match probe() {
        Ok(()) => None,
        Err(e) => Some(e.clone()),
    }
}

pub fn ep_choice() -> String {
    match env_ep_choice() {
        OrtExecutionProviderChoice::Auto => "auto".to_string(),
        OrtExecutionProviderChoice::Cpu => "cpu".to_string(),
        OrtExecutionProviderChoice::Cuda => "cuda".to_string(),
    }
}

// Task 1.9: ONNX diagnostic info
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnnxDiagnosticInfo {
    pub compiled: bool,
    pub available: bool,
    pub error: Option<String>,
    pub ep_choice: String,
    pub onnx_version: Option<String>,
    pub providers: Option<Vec<String>>,
}

pub fn diagnose_onnx_availability() -> OnnxDiagnosticInfo {
    let compiled = compiled();
    let ep_choice_val = ep_choice();

    if !compiled {
        return OnnxDiagnosticInfo {
            compiled: false,
            available: false,
            error: Some("ONNX feature not compiled".to_string()),
            ep_choice: "disabled".to_string(),
            onnx_version: None,
            providers: None,
        };
    }

    let available = is_available();
    let error = if !available { model_load_error() } else { None };

    // Try to get ONNX runtime version and available providers
    let (onnx_version, providers) = if let Ok(()) = ensure_ort_init() {
        let version = Some(format!("ort {}", env!("CARGO_PKG_VERSION")));
        let providers_list = Some(vec![ep_choice_val.clone()]);
        (version, providers_list)
    } else {
        (None, None)
    };

    OnnxDiagnosticInfo {
        compiled,
        available,
        error,
        ep_choice: ep_choice_val,
        onnx_version,
        providers,
    }
}

// ─── 分块推理环境变量辅助（任务 2.5）──────────────────────────────────────────

/// 从环境变量 `HIFISHIFTER_ONNX_CHUNK_SEC` 读取单块最大时长（秒），默认 10.0。
pub fn env_chunk_sec() -> f64 {
    std::env::var("HIFISHIFTER_ONNX_CHUNK_SEC")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(10.0)
}

/// 从环境变量 `HIFISHIFTER_ONNX_OVERLAP_SEC` 读取相邻块重叠时长（秒），默认 0.1。
pub fn env_overlap_sec() -> f64 {
    std::env::var("HIFISHIFTER_ONNX_OVERLAP_SEC")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.1)
}

// ─── 分块推理（任务 2.1-2.4）──────────────────────────────────────────────────

/// 对长 clip 进行分块推理，每块调用 [`infer_pitch_edit_mono`]，
/// 相邻块之间使用等功率 crossfade 拼接，消除块边界伪影。
///
/// # 参数
///
/// - `mono_pcm`：单声道 PCM 输入（f32，已归一化）
/// - `sample_rate`：采样率（Hz）
/// - `start_sec`：该片段在时间轴上的起始时间（秒），用于 `midi_at_time` 对齐
/// - `midi_at_time`：返回目标绝对 MIDI 的回调（0.0 表示静音/无效）
/// - `chunk_sec`：单块最大时长（秒），建议 5.0–15.0
/// - `overlap_sec`：相邻块的重叠时长（秒），用于等功率 crossfade
///
/// # 行为
///
/// - 若 `mono_pcm` 时长 ≤ `chunk_sec`，等价于直接调用 `infer_pitch_edit_mono`
/// - 最后一块不足 `chunk_sec` 时直接推理，无需额外 padding
/// - 输出长度与输入 `mono_pcm` 严格一致
pub fn infer_pitch_edit_chunked(
    mono_pcm: &[f32],
    sample_rate: u32,
    start_sec: f64,
    midi_at_time: impl Fn(f64) -> f64 + Clone,
    formant_shift_at_time: impl Fn(f64) -> f32 + Clone,
    chunk_sec: f64,
    overlap_sec: f64,
) -> Result<Vec<f32>, String> {
    if mono_pcm.is_empty() {
        return Ok(vec![]);
    }

    let sr = sample_rate.max(1) as f64;
    let total_samples = mono_pcm.len();
    let chunk_samples = ((chunk_sec * sr).round() as usize).max(1);
    let overlap_samples =
        ((overlap_sec * sr).round() as usize).min(chunk_samples.saturating_sub(1));

    // 单块情况：直接调用 infer_pitch_edit_mono，无额外开销
    if total_samples <= chunk_samples {
        return infer_pitch_edit_mono(
            mono_pcm,
            sample_rate,
            start_sec,
            midi_at_time,
            formant_shift_at_time,
        );
    }

    // 多块情况：分块推理 + 等功率 crossfade 拼接
    let mut out = vec![0.0f32; total_samples];

    // 步长 = 块长 - 重叠长，保证相邻块有 overlap_samples 的重叠区
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);

    let mut chunk_start = 0usize;
    // 记录上一块推理结果的末尾（用于 crossfade），以及其在 out 中的起始位置
    let mut prev_chunk_out: Option<(Vec<f32>, usize)> = None;

    loop {
        let chunk_end = (chunk_start + chunk_samples).min(total_samples);
        let chunk_pcm = &mono_pcm[chunk_start..chunk_end];
        let chunk_start_sec = start_sec + (chunk_start as f64) / sr;

        let chunk_result = infer_pitch_edit_mono(
            chunk_pcm,
            sample_rate,
            chunk_start_sec,
            midi_at_time.clone(),
            formant_shift_at_time.clone(),
        )?;

        // 确保推理结果长度与输入一致（infer_pitch_edit_mono 保证这一点）
        let chunk_len = chunk_result.len().min(chunk_end - chunk_start);

        if let Some((prev_out, prev_start)) = prev_chunk_out.take() {
            // crossfade 区域：当前块的前 overlap_samples 与上一块的后 overlap_samples 混合
            // 等功率权重：w_curr = sin(t·π/2)，w_prev = cos(t·π/2)，满足 w²+w²=1
            let xfade_len = overlap_samples.min(chunk_len).min(
                prev_out
                    .len()
                    .saturating_sub(chunk_start.saturating_sub(prev_start)),
            );

            for i in 0..xfade_len {
                let t = (i as f64 + 0.5) / (xfade_len as f64).max(1.0);
                let angle = t * std::f64::consts::FRAC_PI_2;
                let w_curr = angle.sin() as f32;
                let w_prev = angle.cos() as f32;

                let out_idx = chunk_start + i;
                if out_idx >= total_samples {
                    break;
                }
                // 上一块在该位置的值（已写入 out）
                let prev_val = out[out_idx];
                // 当前块在该位置的值
                let curr_val = chunk_result.get(i).copied().unwrap_or(0.0);
                out[out_idx] = prev_val * w_prev + curr_val * w_curr;
            }

            // crossfade 区域之后：直接写入当前块的剩余部分
            for i in xfade_len..chunk_len {
                let out_idx = chunk_start + i;
                if out_idx >= total_samples {
                    break;
                }
                out[out_idx] = chunk_result.get(i).copied().unwrap_or(0.0);
            }

            // 保存当前块供下一次 crossfade 使用（仅保留末尾 overlap 区域）
            prev_chunk_out = Some((chunk_result, chunk_start));
        } else {
            // 第一块：直接写入，无需 crossfade
            for i in 0..chunk_len {
                let out_idx = chunk_start + i;
                if out_idx >= total_samples {
                    break;
                }
                out[out_idx] = chunk_result.get(i).copied().unwrap_or(0.0);
            }
            prev_chunk_out = Some((chunk_result, chunk_start));
        }

        if chunk_end >= total_samples {
            break;
        }
        chunk_start += step;
    }

    Ok(out)
}

// ─── Mel 共振峰偏移（频率轴线性插值）──────────────────────────────────────────

/// 对 mel 矩阵逐帧应用共振峰偏移。
///
/// `mel`: `[n_mels * t]` 行优先展平数据（n_mels 行 × t 列）。
/// `formant_shifts`: `[t]`，每帧的共振峰偏移量（单位：cents）。
/// `fmin` / `fmax`：mel filterbank 的频率范围（Hz），必须与提取 mel 时使用的参数一致。
///
/// 对每帧，根据 shift 值计算频率缩放因子 `ratio = 2^(shift/1200)`，
/// 然后在 **Hz 域**对每个输出 mel bin 查找对应源 bin（正确处理 Slaney mel 的非线性刻度）：
///   source_hz = center_hz(output_bin) / ratio  →  source_bin = hz_to_mel_bin(source_hz)
///
/// - 正值 → 共振峰上移 → 声音变细
/// - 负值 → 共振峰下移 → 声音变粗
fn shift_mel_formant(
    mel: &mut [f32],
    n_mels: usize,
    t: usize,
    formant_shifts: &[f32],
    fmin: f32,
    fmax: f32,
) {
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.max(fmin + 1.0));
    let mel_range = (mel_max - mel_min).max(1e-9);
    let n_mels_f = n_mels as f32;
    let silence = (1e-9_f32).ln();

    let mut col_buf = vec![0.0f32; n_mels];

    // 提取常量表达式，消除指数运算
    let hz_m_table: Vec<f32> = (0..n_mels)
        .map(|m| {
            let mel_center = mel_min + (m as f32 + 1.0) * mel_range / (n_mels_f + 1.0);
            mel_to_hz_slaney(mel_center)
        })
        .collect();

    for frame in 0..t {
        let shift = formant_shifts.get(frame).copied().unwrap_or(0.0);
        if shift.abs() < 0.5 {
            continue;
        }

        let ratio = 2.0f32.powf(shift / 1200.0);
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }

        for m in 0..n_mels {
            col_buf[m] = mel[m * t + frame];
        }

        for m in 0..n_mels {
            let hz_m = hz_m_table[m]; // 直接查表，复杂度 O(1)
            let hz_src = hz_m / ratio;
            let mel_src = hz_to_mel_slaney(hz_src.max(0.0));
            let src_bin_f = (mel_src - mel_min) / mel_range * (n_mels_f + 1.0) - 1.0;

            let i0 = src_bin_f.floor() as isize;
            let frac = (src_bin_f - i0 as f32).clamp(0.0, 1.0);

            let v = if i0 < 0 {
                // 低于 fmin：静音填充（共振峰上移时低频端留空）
                silence
            } else {
                let i0u = i0 as usize;
                if i0u >= n_mels {
                    // 高于 fmax：静音填充（共振峰下移时高频端留空，避免引入伪高频能量）
                    silence
                } else if i0u == n_mels - 1 {
                    col_buf[i0u]
                } else {
                    let a = col_buf[i0u];
                    let b = col_buf[i0u + 1];
                    a + (b - a) * frac
                }
            };

            mel[m * t + frame] = v;
        }
    }
}

// ─── Mel 时间轴线性插值 + HiFiGAN 推理（mel stretch 方案）─────────────────────

/// 沿时间轴对 mel 矩阵做线性插值。
///
/// 输入: `mel` 为 `[n_mels * t_in]` 的行优先（n_mels 行 × t_in 列）展平数据。
/// 输出: `[n_mels * t_out]`，同样行优先。
///
/// 当 `t_in == t_out` 时直接返回输入的拷贝。
fn interpolate_mel_time(mel: &[f32], n_mels: usize, t_in: usize, t_out: usize) -> Vec<f32> {
    if t_in == t_out {
        return mel.to_vec();
    }
    if t_in == 0 || t_out == 0 {
        return vec![0.0f32; n_mels * t_out];
    }

    let mut out = Vec::with_capacity(n_mels * t_out);
    let scale = if t_out <= 1 {
        0.0
    } else {
        (t_in as f64 - 1.0) / (t_out as f64 - 1.0)
    };

    for m in 0..n_mels {
        let src_row = m * t_in;
        for j in 0..t_out {
            let t_src = (j as f64) * scale;
            let i0 = t_src.floor() as usize;
            let i1 = (i0 + 1).min(t_in - 1);
            let frac = (t_src - i0 as f64) as f32;
            let a = mel[src_row + i0];
            let b = mel[src_row + i1];
            out.push(a + (b - a) * frac);
        }
    }
    out
}

impl NsfHifiganOnnx {
    /// 从原始 PCM 提取 mel → 沿时间轴插值到目标长度 → 构建 F0 → 推理输出波形。
    ///
    /// 与 [`infer_from_audio_and_midi`] 的区别：不需要预先对 PCM 做时间拉伸，
    /// 而是在 mel 域完成时间拉伸，由 HiFiGAN 直接从插值后的 mel 合成波形。
    ///
    /// # 参数
    /// - `audio_mono`：**源速率**的原始 PCM（未拉伸）
    /// - `sample_rate`：PCM 采样率
    /// - `playback_rate`：播放速率（> 1.0 快放/缩短，< 1.0 慢放/拉长）
    /// - `start_sec`：该段在**时间轴**上的起始秒（已考虑拉伸后坐标）
    /// - `midi_at_time`：回调，参数为时间轴绝对时间（秒），返回目标 MIDI 值
    pub fn infer_from_audio_and_midi_mel_stretch(
        &mut self,
        audio_mono: &[f32],
        sample_rate: u32,
        playback_rate: f64,
        start_sec: f64,
        midi_at_time: impl Fn(f64) -> f64,
        formant_shift_at_time: impl Fn(f64) -> f32,
    ) -> Result<Vec<f32>, String> {
        let model_sr = self.cfg.sampling_rate;

        // 1. 重采样到模型采样率、从原始 PCM 提取 mel [n_mels, T_orig]
        let mel_orig = if sample_rate == model_sr {
            self.mel_from_audio_fast(audio_mono)?
        } else {
            let mut resample_buf = std::mem::take(&mut self.audio_resample_buf);
            linear_resample_mono_into(audio_mono, sample_rate, model_sr, &mut resample_buf);
            let mel_result = self.mel_from_audio_fast(&resample_buf);
            self.audio_resample_buf = resample_buf;
            mel_result?
        };
        let t_orig = mel_orig.len() / self.cfg.num_mels;
        if t_orig == 0 {
            // 拉伸后的目标 PCM 长度
            let target_len = ((audio_mono.len() as f64) / playback_rate).round().max(0.0) as usize;
            return Ok(vec![0.0; target_len]);
        }

        // 2. 计算拉伸后的目标帧数 T_new = T_orig / playback_rate
        let t_new = ((t_orig as f64) / playback_rate).round().max(1.0) as usize;

        // 3. mel 时间轴线性插值 [n_mels, T_orig] → [n_mels, T_new]
        let mut mel_stretched = if (playback_rate - 1.0).abs() <= 1e-6 {
            mel_orig
        } else {
            interpolate_mel_time(&mel_orig, self.cfg.num_mels, t_orig, t_new)
        };

        // 4. 应用共振峰偏移（在 mel 域沿频率轴做线性插值）
        let hop_sec = (self.cfg.hop_size as f64) / (model_sr.max(1) as f64);
        let formant_shifts: Vec<f32> = (0..t_new)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                formant_shift_at_time(abs_t)
            })
            .collect();
        let has_formant_shift = formant_shifts.iter().any(|s| s.abs() >= 0.5);
        if has_formant_shift {
            shift_mel_formant(
                &mut mel_stretched,
                self.cfg.num_mels,
                t_new,
                &formant_shifts,
                self.cfg.fmin,
                self.cfg.fmax,
            );
        }

        // 5. 构建 F0 [T_new]
        // F0 直接按时间轴坐标查询，pitch_edit / clip_midi 已与时间轴对齐
        let f0: Vec<f32> = (0..t_new)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                midi_to_hz(midi_at_time(abs_t))
            })
            .collect();

        // 6. 分段推理（复用现有环境变量控制的段式推理逻辑）
        let seg_frames = Self::env_usize("HIFISHIFTER_NSF_HIFIGAN_SEGMENT_FRAMES").unwrap_or(0);
        let overlap_frames = Self::env_usize("HIFISHIFTER_NSF_HIFIGAN_OVERLAP_FRAMES").unwrap_or(8);

        let y_vec: Vec<f32> = if seg_frames >= 16 && t_new > seg_frames {
            let overlap_frames = overlap_frames.min(seg_frames.saturating_sub(1));
            let step = seg_frames.saturating_sub(overlap_frames).max(1);

            let expected_total = t_new.saturating_mul(self.cfg.hop_size).max(1);
            let mut out = vec![0.0f32; expected_total];
            let mut wsum = vec![0.0f32; expected_total];

            let mut s = 0usize;
            while s < t_new {
                let end = (s + seg_frames).min(t_new);
                let seg_t = end.saturating_sub(s).max(1);

                let mut mel_seg = vec![0.0f32; self.cfg.num_mels * seg_t];
                for m in 0..self.cfg.num_mels {
                    let src = &mel_stretched[m * t_new + s..m * t_new + end];
                    let dst = &mut mel_seg[m * seg_t..(m + 1) * seg_t];
                    dst.copy_from_slice(src);
                }
                let f0_seg = f0[s..end].to_vec();

                let y_seg = self.run_model(mel_seg, f0_seg, seg_t)?;
                let seg_expected = seg_t.saturating_mul(self.cfg.hop_size).max(1);
                let seg_samples = y_seg.len().min(seg_expected);

                let overlap_samples = overlap_frames.saturating_mul(self.cfg.hop_size);
                let base = s.saturating_mul(self.cfg.hop_size);

                for i in 0..seg_samples {
                    let g = base + i;
                    if g >= out.len() {
                        break;
                    }
                    let mut w = 1.0f32;
                    if overlap_samples > 0 {
                        if s > 0 && i < overlap_samples {
                            w = (i as f32) / (overlap_samples as f32);
                        }
                        if end < t_new && seg_samples > overlap_samples {
                            let tail = seg_samples.saturating_sub(1).saturating_sub(i);
                            if tail < overlap_samples {
                                let w_out = (tail as f32) / (overlap_samples as f32);
                                w = w.min(w_out);
                            }
                        }
                    }

                    out[g] += y_seg[i] * w;
                    wsum[g] += w;
                }

                if end >= t_new {
                    break;
                }
                s += step;
            }

            for i in 0..out.len() {
                let w = wsum[i];
                if w > 1e-6 {
                    out[i] /= w;
                }
            }
            out
        } else {
            self.run_model(mel_stretched, f0, t_new)?
        };

        // 7. 重采样回原始采样率
        let mut out = if model_sr == sample_rate {
            y_vec
        } else {
            linear_resample_mono(&y_vec, model_sr, sample_rate)
        };

        // 8. 对齐到拉伸后的目标长度
        let target_len = ((audio_mono.len() as f64) / playback_rate).round().max(0.0) as usize;
        if out.len() > target_len {
            out.truncate(target_len);
        } else if out.len() < target_len {
            out.resize(target_len, 0.0);
        }
        Ok(out)
    }
}

/// 单次 mel stretch 推理入口（thread-local session）。
///
/// 参数语义与 [`infer_pitch_edit_mono`] 相似，但额外接收 `playback_rate`
/// 并在 mel 域完成时间拉伸，省去外部预处理。
pub fn infer_pitch_edit_mono_mel_stretch(
    audio_mono: &[f32],
    sample_rate: u32,
    playback_rate: f64,
    start_sec: f64,
    midi_at_time: impl Fn(f64) -> f64,
    formant_shift_at_time: impl Fn(f64) -> f32,
) -> Result<Vec<f32>, String> {
    if let Err(e) = probe() {
        return Err(e.clone());
    }

    TLS_SESSION.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(NsfHifiganOnnx::load());
        }
        let sess = opt
            .as_mut()
            .expect("TLS_SESSION just initialized")
            .as_mut()
            .map_err(|e| e.clone())?;

        sess.infer_from_audio_and_midi_mel_stretch(
            audio_mono,
            sample_rate,
            playback_rate,
            start_sec,
            midi_at_time,
            formant_shift_at_time,
        )
    })
}

/// 分块 mel stretch 推理：对长 clip 分块调用 [`infer_pitch_edit_mono_mel_stretch`]，
/// 相邻块之间使用等功率 crossfade 拼接。
pub fn infer_pitch_edit_chunked_mel_stretch(
    mono_pcm: &[f32],
    sample_rate: u32,
    playback_rate: f64,
    start_sec: f64,
    midi_at_time: impl Fn(f64) -> f64 + Clone,
    formant_shift_at_time: impl Fn(f64) -> f32 + Clone,
    chunk_sec: f64,
    overlap_sec: f64,
) -> Result<Vec<f32>, String> {
    if mono_pcm.is_empty() {
        return Ok(vec![]);
    }

    let sr = sample_rate.max(1) as f64;
    let total_samples = mono_pcm.len();
    // chunk_samples 基于源 PCM 长度（未拉伸）
    let chunk_samples = ((chunk_sec * sr * playback_rate).round() as usize).max(1);
    // overlap_samples 也基于源 PCM
    let overlap_samples =
        ((overlap_sec * sr * playback_rate).round() as usize).min(chunk_samples.saturating_sub(1));

    // 拉伸后的总目标长度
    let target_total = ((total_samples as f64) / playback_rate).round().max(0.0) as usize;

    // 单块情况
    if total_samples <= chunk_samples {
        return infer_pitch_edit_mono_mel_stretch(
            mono_pcm,
            sample_rate,
            playback_rate,
            start_sec,
            midi_at_time,
            formant_shift_at_time,
        );
    }

    // 多块情况：按源 PCM 分块，每块独立做 mel stretch，然后拼接
    let mut out = vec![0.0f32; target_total];
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);

    let mut chunk_start = 0usize;
    let mut prev_chunk_out: Option<(Vec<f32>, usize)> = None;

    loop {
        let chunk_end = (chunk_start + chunk_samples).min(total_samples);
        let chunk_pcm = &mono_pcm[chunk_start..chunk_end];

        // 该块在时间轴上的起始时间
        let chunk_start_sec = start_sec + (chunk_start as f64) / sr / playback_rate;

        let chunk_result = infer_pitch_edit_mono_mel_stretch(
            chunk_pcm,
            sample_rate,
            playback_rate,
            chunk_start_sec,
            midi_at_time.clone(),
            formant_shift_at_time.clone(),
        )?;

        // 该块在输出中的起始位置
        let out_start = ((chunk_start as f64) / playback_rate).round() as usize;
        let chunk_len = chunk_result.len();

        // 重叠区域的输出样本数
        let overlap_out_samples = ((overlap_samples as f64) / playback_rate).round() as usize;

        if let Some((_prev_out, _prev_start)) = prev_chunk_out.take() {
            // crossfade 区域
            let xfade_len = overlap_out_samples.min(chunk_len);

            for i in 0..xfade_len {
                let t = (i as f64 + 0.5) / (xfade_len as f64).max(1.0);
                let angle = t * std::f64::consts::FRAC_PI_2;
                let w_curr = angle.sin() as f32;
                let w_prev = angle.cos() as f32;

                let out_idx = out_start + i;
                if out_idx >= target_total {
                    break;
                }
                let prev_val = out[out_idx];
                let curr_val = chunk_result.get(i).copied().unwrap_or(0.0);
                out[out_idx] = prev_val * w_prev + curr_val * w_curr;
            }

            // crossfade 之后的部分
            for i in xfade_len..chunk_len {
                let out_idx = out_start + i;
                if out_idx >= target_total {
                    break;
                }
                out[out_idx] = chunk_result.get(i).copied().unwrap_or(0.0);
            }

            prev_chunk_out = Some((chunk_result, out_start));
        } else {
            // 第一块
            for i in 0..chunk_len {
                let out_idx = out_start + i;
                if out_idx >= target_total {
                    break;
                }
                out[out_idx] = chunk_result.get(i).copied().unwrap_or(0.0);
            }
            prev_chunk_out = Some((chunk_result, out_start));
        }

        if chunk_end >= total_samples {
            break;
        }
        chunk_start += step;
    }

    Ok(out)
}
