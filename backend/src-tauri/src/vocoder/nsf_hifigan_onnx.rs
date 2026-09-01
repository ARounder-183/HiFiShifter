use ndarray::Array2;
use num_complex::Complex32;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::Fft;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

#[derive(Debug)]
enum ModelRunError {
    Message(String),
    TimedOut,
}

fn run_session_once(
    session: &Arc<Mutex<Session>>,
    n_mels: usize,
    mel_buf: Vec<f32>,
    f0_buf: Vec<f32>,
    t: usize,
    timeout: std::time::Duration,
) -> Result<Vec<f32>, ModelRunError> {
    let sess = Arc::clone(session);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = (|| -> Result<Vec<f32>, String> {
            let mel_tensor = Tensor::from_array(([1usize, n_mels, t], mel_buf.into_boxed_slice()))
                .map_err(|e| format!("build mel tensor failed: {e}"))?;
            let f0_tensor = Tensor::from_array(([1usize, t], f0_buf.into_boxed_slice()))
                .map_err(|e| format!("build f0 tensor failed: {e}"))?;
            let mut session_guard = sess
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
            Ok(data.to_vec())
        })();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(e)) => Err(ModelRunError::Message(e)),
        Err(_) => Err(ModelRunError::TimedOut),
    }
}

fn reset_shared_session() {
    if let Some(mutex) = SHARED_SESSION.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = None;
        }
    }
}

/// Global progress callback for chunk rendering. Set before render, cleared after.
static CHUNK_PROGRESS_CB: OnceLock<Mutex<Option<Box<dyn Fn(f64) + Send + Sync>>>> = OnceLock::new();
static CHUNK_PROGRESS_TOTAL: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();
static CHUNK_PROGRESS_DONE: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();

pub fn set_chunk_progress_callback(cb: Option<Box<dyn Fn(f64) + Send + Sync>>) {
    let slot = CHUNK_PROGRESS_CB.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap() = cb;
}

pub fn reset_chunk_progress(total: usize) {
    CHUNK_PROGRESS_TOTAL
        .get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
        .store(total, std::sync::atomic::Ordering::Relaxed);
    CHUNK_PROGRESS_DONE
        .get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

fn emit_chunk_progress(_local: f64) {
    let done = CHUNK_PROGRESS_DONE
        .get()
        .map(|a| a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1)
        .unwrap_or(0);
    let total = CHUNK_PROGRESS_TOTAL
        .get()
        .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(1);
    let progress = if total > 0 {
        done as f64 / total as f64
    } else {
        0.0
    };
    if let Some(slot) = CHUNK_PROGRESS_CB.get() {
        if let Some(cb) = slot.lock().unwrap().as_ref() {
            cb(progress);
        }
    }
}

/// Tracks which execution provider the live session actually uses.
///
/// Backed by an `RwLock` rather than a `OnceLock`: the EP can change while the
/// process is running (`update_ort_ep()` rebuilds the session when the user
/// switches device in the UI), and a `OnceLock` would keep reporting the very
/// first EP forever.
static ACTIVE_EP: OnceLock<RwLock<String>> = OnceLock::new();

/// Record the EP a freshly built session actually ended up on.
fn set_active_ep(ep: &str) {
    let slot = ACTIVE_EP.get_or_init(|| RwLock::new("unknown".to_string()));
    if let Ok(mut guard) = slot.write() {
        *guard = ep.to_string();
    }
}

/// Returns the EP the live session actually uses — e.g. `"coreml"` on macOS
/// ARM64, `"directml"` on Windows, `"webgpu"`, or `"cpu"`.  Returns
/// `"unknown"` before the first session has been built.
pub fn active_ep() -> String {
    ACTIVE_EP
        .get()
        .and_then(|slot| slot.read().ok().map(|g| g.clone()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Human-readable display name for [`active_ep`], for the UI's device readout.
///
/// `"CoreML"` / `"WebGPU"` / `"DirectML"` / `"CPU"`, or `""` when no session
/// has been built yet.  This must stay a runtime value — a hard-coded
/// compile-time backend name is what made the menu claim "GPU (CoreML)" while
/// inference was actually falling back to CPU.
pub fn active_backend_name() -> &'static str {
    match active_ep().as_str() {
        "coreml" => "CoreML",
        "webgpu" => "WebGPU",
        "directml" => "DirectML",
        "cpu" => "CPU",
        _ => "",
    }
}

fn ensure_ort_init() -> Result<(), String> {
    match ORT_INIT.get_or_init(|| {
        // Try to commit our desired environment config (name, etc.).
        // If commit() returns false, the environment was already committed
        // by another module (e.g. FCPE or HNSEP init ran first) — that's
        // perfectly fine; the active environment is still valid.
        ort::init().with_name("hifishifter").commit();

        // Ensure the environment is actually created before we proceed.
        // Environment::current() lazily creates the OrtEnv from the committed
        // options and caches it for all subsequent calls.
        if let Err(e) = ort::environment::Environment::current() {
            return Err(format!("failed to create ORT environment: {e}"));
        }

        log::warn!("[ort] initialized: {}", ort::info());
        let providers = crate::vocoder_ort_session::diagnose_available_providers();
        log::warn!("[ort] available providers: {providers:?}");
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

fn build_session_with_ep(onnx_path: &Path) -> Result<Session, String> {
    let (session, ep) = crate::vocoder_ort_session::build_ort_session(
        onnx_path,
        crate::vocoder_ort_session::OrtSessionRole::Vocoder,
    )?;
    set_active_ep(&ep);
    Ok(session)
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

/// On macOS the CoreML-compatible model variant (static Pad pads) is used
/// because the stock model's dynamic Pad input cannot be compiled by the
/// CoreML EP ("output_features has no value for 'Sub_output_0'").  The
/// variant is numerically identical to the stock model, so Intel macOS
/// (CPU-only) can use it too, and macOS bundles only this single model.
fn vocoder_model_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "pc_nsf_hifigan_coreml.onnx"
    } else {
        "pc_nsf_hifigan.onnx"
    }
}

fn default_model_dir_guess() -> Option<PathBuf> {
    // 开发环境：模型位于 CARGO_MANIFEST_DIR/resources/models/nsf_hifigan/
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .join("resources")
        .join("models")
        .join("nsf_hifigan");
    let has_model =
        p.join(vocoder_model_filename()).is_file() || p.join("pc_nsf_hifigan.onnx").is_file();
    if has_model && p.join("config.json").is_file() {
        return Some(p);
    }

    // 发布/便携环境：模型位于可执行文件同级的 models/ 目录中
    // （CARGO_MANIFEST_DIR 在发布构建中指向构建机器的路径，在用户机器上不存在）
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let p = exe_dir.join("models").join("nsf_hifigan");
            let has_model = p.join(vocoder_model_filename()).is_file()
                || p.join("pc_nsf_hifigan.onnx").is_file();
            if has_model && p.join("config.json").is_file() {
                return Some(p);
            }
        }
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

    if let Some(dir) = crate::nsf_hifigan_model_dir()
        .map(|p| p.to_path_buf())
        .or_else(|| env_path("HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR"))
        .or_else(default_model_dir_guess)
    {
        let preferred = dir.join(vocoder_model_filename());
        let onnx = if preferred.is_file() {
            preferred
        } else {
            dir.join("pc_nsf_hifigan.onnx")
        };
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
    // Build test tensors from the session's actual input metadata so the
    // probe works for both dynamic-shape sessions (CPU/WebGPU, tiny test
    // frames) and fixed-shape CoreML sessions (4096 frames).
    use ort::value::{Tensor, ValueType};
    let mut input_pairs: Vec<(String, ort::value::Value)> = Vec::new();
    for input in session.inputs() {
        let (tensor_ty, shape) = match input.dtype() {
            ValueType::Tensor { ty, shape, .. } => (ty, shape),
            _ => continue,
        };
        if *tensor_ty != ort::value::TensorElementType::Float32 {
            continue;
        }
        if shape.iter().any(|&d| d == 0) {
            continue;
        }
        let test_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0 {
                    d as usize
                } else if i == 0 {
                    1
                } else {
                    4
                }
            })
            .collect();
        let total: usize = test_shape.iter().product::<usize>().max(1);
        let data: Vec<f32> = vec![0.0f32; total];
        let tensor = Tensor::from_array((test_shape, data.into_boxed_slice()))
            .map_err(|e| format!("probe: tensor '{}' creation failed: {e}", input.name()))?;
        input_pairs.push((input.name().to_string(), tensor.into()));
    }
    if input_pairs.is_empty() {
        return Err("probe: no f32 tensor inputs found".to_string());
    }
    let outputs = session
        .run(input_pairs)
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

/// 进程级全局共享的 ORT Session 容器。
/// 使用 Mutex 允许我们在运行时修改 Session 以切换 EPs。
static SHARED_SESSION: OnceLock<Mutex<Option<Arc<Mutex<Session>>>>> = OnceLock::new();

/// 递增此 Epoch 可以促使所有 Thread Local 重新加载 ONNX 实例以同步 EP 切换。
static SESSION_EPOCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Drop the shared ORT session to release GPU memory. Called on app exit.
///
/// Uses try_lock with a short spin to avoid blocking indefinitely if
/// another thread is stuck holding the session lock (e.g. during a
/// hung GPU operation on WSL2/Lavapipe).
pub fn drop_shared_session() {
    if let Some(mutex) = SHARED_SESSION.get() {
        // Try to acquire the lock for up to ~500ms before giving up.
        // At shutdown we don't want to block the main thread forever.
        for _ in 0..10 {
            if let Ok(mut guard) = mutex.try_lock() {
                *guard = None;
                log::warn!("[nsf_hifigan] shared session dropped");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        log::error!(
            "[nsf_hifigan] WARNING: could not acquire SHARED_SESSION lock at shutdown — giving up"
        );
    }
}

/// 初始化（或获取已有的）全局 Session。
fn get_or_init_shared_session() -> Result<Arc<Mutex<Session>>, String> {
    let mutex = SHARED_SESSION.get_or_init(|| Mutex::new(None));
    let mut guard = mutex
        .lock()
        .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?;
    if let Some(ref session) = *guard {
        return Ok(Arc::clone(session));
    }
    ensure_ort_init()?;
    let (onnx_path, _cfg_path) = resolve_model_paths()?;
    let session = build_session_with_ep(&onnx_path)?;
    let arc = Arc::new(Mutex::new(session));
    *guard = Some(Arc::clone(&arc));
    Ok(arc)
}

pub fn update_ort_ep(choice: &str, device_id: Option<i32>) {
    let ep_str = choice.trim().to_lowercase();

    // 写入运行时 EP 覆盖设置（存储在 ort_session 模块中）
    crate::vocoder_ort_session::set_runtime_ep_override(Some(ep_str));
    // 写入 DirectML 设备 ID 覆盖
    crate::vocoder_ort_session::set_runtime_dml_device_id(device_id);

    // 重置全局 Session，下一次渲染请求时将自动使用新 EP 重新创建
    if let Some(mutex) = SHARED_SESSION.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = None;
        }
    }

    // 更新 Epoch，这会告知所有的 TLS 缓存将他们的本地 NsfHifiganOnnx 实例作废并重新载入
    SESSION_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    // 清空 Active EP：会话是惰性重建的，在下一次渲染真正把新会话建起来之前
    // 不能继续上报旧 EP（那正是菜单里"显示 GPU 但实际在跑 CPU"的原因）。
    set_active_ep("unknown");
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
    /// 标记当前实例是在哪个 Epoch 加载的。用于检测重新加载。
    epoch: usize,
    /// True when the ORT session's batch dimension is pinned to 1
    /// (DirectML session builder overrides batch=1; CoreML also pins batch=1).
    /// When true, batched tensors with B>1 are invalid and must run sequentially.
    batch_pinned_to_one: bool,
}

/// Detect whether the current ONNX session has its batch dimension pinned to 1.
///
/// DirectML sessions are built with `.with_dimension_override("batch", 1)` to
/// avoid dynamic-shape GPU shaders, so they cannot accept a batched input with
/// B>1. CoreML sessions are likewise pinned to batch=1. CPU/WebGPU sessions
/// usually keep the model's dynamic batch dimension and can run real batches.
fn session_batch_pinned_to_one(session: &Arc<Mutex<Session>>) -> bool {
    let Ok(guard) = session.lock() else {
        // If we cannot inspect the session, prefer the safe sequential path.
        return true;
    };

    let mut has_fixed_batch = false;
    let mut has_dynamic_batch = false;
    for input in guard.inputs() {
        if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
            match shape.first().copied() {
                Some(1) => has_fixed_batch = true,
                Some(-1) => has_dynamic_batch = true,
                _ => {}
            }
        }
    }
    has_fixed_batch && !has_dynamic_batch
}

impl NsfHifiganOnnx {
    fn load() -> Result<Self, String> {
        let current_epoch = SESSION_EPOCH.load(std::sync::atomic::Ordering::SeqCst);
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

        let batch_pinned_to_one = session_batch_pinned_to_one(&session);

        Ok(Self {
            cfg,
            mel_fb_matrix,
            window,
            fft,
            fft_buf,
            pad_buf: Vec::new(),
            audio_resample_buf: Vec::new(),
            session,
            batch_pinned_to_one,
            epoch: current_epoch,
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

    /// Run the vocoder on one mel/f0 pair covering `t` mel frames.
    ///
    /// Sessions keep the model's dynamic `time` axis on every platform, so the
    /// inputs go through verbatim and the output needs no trimming.  The
    /// Windows DirectML builder still pins `time` for shader specialisation,
    /// but a dimension override only drives graph specialisation — the session
    /// accepts any runtime length, and this path never padded for it.
    fn run_model(&mut self, mel: Vec<f32>, f0: Vec<f32>, t: usize) -> Result<Vec<f32>, String> {
        let n_mels = self.cfg.num_mels;
        let timeout = std::time::Duration::from_secs(120);

        // Clone the inputs before the first attempt.  If a GPU EP hangs, the
        // first attempt owns the original buffers on its worker thread and we
        // need fresh copies for the automatic EP-fallback retry.
        let retry_mel = mel.clone();
        let retry_f0 = f0.clone();

        let first = run_session_once(&self.session, n_mels, mel, f0, t, timeout);
        match first {
            Ok(data) => Ok(data),
            Err(ModelRunError::Message(e)) => Err(e),
            Err(ModelRunError::TimedOut) => {
                log::warn!(
                    "[nsf_hifigan] model inference timed out after {timeout:?}; disabling the hung EP and retrying with a fresh session"
                );
                crate::vocoder_ort_session::disable_coreml("vocoder inference timed out");
                reset_shared_session();
                self.session = get_or_init_shared_session()?;

                match run_session_once(
                    &self.session,
                    n_mels,
                    retry_mel,
                    retry_f0,
                    t,
                    std::time::Duration::from_secs(120),
                ) {
                    Ok(data) => Ok(data),
                    Err(ModelRunError::Message(e)) => Err(e),
                    Err(ModelRunError::TimedOut) => Err(format!(
                        "model inference timed out again after EP fallback ({t} frames)"
                    )),
                }
            }
        }
    }

    /// Each item is (mel_vec, f0_vec, t) where t is the mel frame count.
    /// Returns Vec of output waveforms, each trimmed to its original expected length.
    fn run_model_batch(
        &mut self,
        items: &[(Vec<f32>, Vec<f32>, usize)],
    ) -> Result<Vec<Vec<f32>>, String> {
        if items.is_empty() {
            return Ok(vec![]);
        }
        if items.len() == 1 {
            let (mel, f0, t) = &items[0];
            return self.run_model(mel.clone(), f0.clone(), *t).map(|v| vec![v]);
        }
        let n = items.len();
        // Every item is zero-padded up to the longest one so the batch shares
        // a single rectangular tensor; results are trimmed back afterwards.
        let max_t = items.iter().map(|(_, _, t)| *t).max().unwrap_or(1);

        // Batched inference is only exact when no item needs padding.  The
        // model's f0 source-generator subgraph runs across the whole time
        // axis, so feeding a chunk zero-padded to a longer length changes the
        // audio in the *valid* region too (measured rel_l2 up to ~8% for a
        // 256-frame chunk padded to 1024 — and identically on CPU and CoreML,
        // so this is a property of the model, not of the execution provider).
        // Items of unequal length therefore run one at a time, which is also
        // what DirectML requires because its builder pins batch=1.
        let uniform_length = items.iter().all(|(_, _, t)| *t == max_t);
        if self.batch_pinned_to_one || !uniform_length {
            let mut results = Vec::with_capacity(items.len());
            for (mel, f0, t) in items {
                results.push(self.run_model(mel.clone(), f0.clone(), *t)?);
            }
            return Ok(results);
        }

        let n_mels = self.cfg.num_mels;
        let hop = self.cfg.hop_size;

        // Build batched mel [B, n_mels, max_t] and f0 [B, max_t], zero-padded
        let mut mel_batch = vec![0.0f32; n * n_mels * max_t];
        let mut f0_batch = vec![0.0f32; n * max_t];
        let mut out_lengths = Vec::with_capacity(n);

        for (i, (mel, f0, t)) in items.iter().enumerate() {
            // mel is (n_mels, t) column-major
            for m in 0..n_mels {
                let src_offset = m * t;
                let dst_offset = (i * n_mels + m) * max_t;
                mel_batch[dst_offset..dst_offset + t]
                    .copy_from_slice(&mel[src_offset..src_offset + t]);
            }
            f0_batch[i * max_t..i * max_t + t].copy_from_slice(f0);
            out_lengths.push(t * hop);
        }

        let mel_tensor = Tensor::from_array(([n, n_mels, max_t], mel_batch.into_boxed_slice()))
            .map_err(|e| format!("build batched mel tensor failed: {e}"))?;
        let f0_tensor = Tensor::from_array(([n, max_t], f0_batch.into_boxed_slice()))
            .map_err(|e| format!("build batched f0 tensor failed: {e}"))?;

        let all_output: Vec<f32> = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| format!("ort session lock poisoned: {e}"))?;
            let outputs = session_guard
                .run(ort::inputs![mel_tensor, f0_tensor])
                .map_err(|e| format!("ort batch run failed: {e}"))?;
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

        // Split batched output back into per-clip results
        let max_out_t = max_t * hop;
        let mut results = Vec::with_capacity(n);
        for (i, &expected_len) in out_lengths.iter().enumerate() {
            let start = i * max_out_t;
            let end = (start + expected_len).min(all_output.len());
            results.push(all_output[start..end].to_vec());
        }
        Ok(results)
    }
}

static PROBE: OnceLock<Mutex<Option<Result<(), String>>>> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TLS_SESSION: RefCell<Option<Result<NsfHifiganOnnx, String>>> = RefCell::new(None);
}

fn probe() -> Result<(), String> {
    let mutex = PROBE.get_or_init(|| Mutex::new(None));
    let mut guard = mutex
        .lock()
        .map_err(|e| format!("PROBE lock poisoned: {e}"))?;
    if let Some(ref res) = *guard {
        return res.clone();
    }
    let res = get_or_init_shared_session().map(|_| ());
    *guard = Some(res.clone());
    res
}

pub fn is_available() -> bool {
    match probe() {
        Ok(()) => true,
        Err(e) => {
            let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
            if debug && !LOGGED_UNAVAILABLE.swap(true, Ordering::Relaxed) {
                log::warn!("nsf_hifigan_onnx: unavailable: {e}");
            }
            false
        }
    }
}

// Helper functions for diagnostics
pub fn compiled() -> bool {
    true
}

pub fn model_load_error() -> Option<String> {
    match probe() {
        Ok(()) => None,
        Err(e) => Some(e),
    }
}

pub fn ep_choice() -> String {
    std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_ascii_lowercase()
}

// Task 1.9: ONNX diagnostic info
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnnxDiagnosticInfo {
    pub compiled: bool,
    pub available: bool,
    pub error: Option<String>,
    pub ep_choice: String,
    pub active_ep: String,
    pub onnx_version: Option<String>,
    pub providers: Option<Vec<String>>,
    /// Full GPU diagnostic info (available providers, smoke test, etc.)
    pub gpu_diagnostic: Option<crate::vocoder_ort_session::GpuDiagnostic>,
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
            active_ep: "none".to_string(),
            onnx_version: None,
            providers: None,
            gpu_diagnostic: None,
        };
    }

    let available = is_available();
    let error = if !available { model_load_error() } else { None };

    // Gather provider info
    let providers = if ensure_ort_init().is_ok() {
        Some(crate::vocoder_ort_session::diagnose_available_providers())
    } else {
        None
    };

    let onnx_version = Some(format!("ort {}", env!("CARGO_PKG_VERSION")));

    // Gather GPU diagnostic
    let gpu_diagnostic = if ensure_ort_init().is_ok() {
        Some(crate::vocoder_ort_session::diagnose_gpu())
    } else {
        None
    };

    OnnxDiagnosticInfo {
        compiled,
        available,
        error,
        ep_choice: ep_choice_val,
        active_ep: active_ep(),
        onnx_version,
        providers,
        gpu_diagnostic,
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

// ─── 帧级分块常量────────────────────────────────────────────

/// 单块最大 mel 帧数：4096 帧（≈47s @ hop=512, sr=44100）。
/// 覆盖大多数 clip 为单 chunk，最小化 GPU 启动开销。
/// 每个 chunk 的 mel 输入约 2MB，波形输出约 8MB，12GB GPU 轻松容纳。
const CHUNK_MAX_FRAMES: usize = 4096;

// ─── 帧级分块推理优化──────────────────────────────────────

/// 优化版长音频分块推理：预提取全段 mel 一次，按帧切片推理，线性 crossfade 拼接。
///
/// 与旧版秒级分块实现的区别：
/// - mel 只提取一次，按帧切片（而非每块独立提取）
/// - 使用帧级常量 `CHUNK_MAX_FRAMES` 分块
/// - 线性 crossfade
/// - 支持分块级缓存回调，参数变动时只重渲染脏 chunk
///
/// `chunk_cache_get(mel_start_frame, mel_end_frame)` → 命中时返回缓存的 mono PCM，
/// `chunk_cache_put(mel_start_frame, mel_end_frame, waveform)` → 写入波形到缓存。
/// 帧号相对于 `mono_pcm` 的起始（0-based mel frame index）。
pub fn infer_pitch_edit_chunked_optimized(
    mono_pcm: &[f32],
    sample_rate: u32,
    start_sec: f64,
    midi_at_time: impl Fn(f64) -> f64 + Clone,
    formant_shift_at_time: impl Fn(f64) -> f32 + Clone,
    chunk_cache_get: &dyn Fn(usize, usize) -> Option<Vec<f32>>,
    chunk_cache_put: &dyn Fn(usize, usize, Vec<f32>),
) -> Result<Vec<f32>, String> {
    if mono_pcm.is_empty() {
        return Ok(vec![]);
    }
    if let Err(e) = probe() {
        return Err(e.clone());
    }

    TLS_SESSION.with(|cell| {
        let mut opt = cell.borrow_mut();

        let current_epoch = SESSION_EPOCH.load(std::sync::atomic::Ordering::SeqCst);
        if let Some(Ok(ref sess)) = *opt {
            if sess.epoch != current_epoch {
                *opt = None;
            }
        }

        if opt.is_none() {
            *opt = Some(NsfHifiganOnnx::load());
        }
        let sess = opt
            .as_mut()
            .expect("TLS_SESSION just initialized")
            .as_mut()
            .map_err(|e| e.clone())?;

        let model_sr = sess.cfg.sampling_rate;
        let hop = sess.cfg.hop_size;

        // 1. 重采样到模型采样率，提取完整 mel
        let mut mel_full = if sample_rate == model_sr {
            sess.mel_from_audio_fast(mono_pcm)?
        } else {
            let mut resample_buf = std::mem::take(&mut sess.audio_resample_buf);
            linear_resample_mono_into(mono_pcm, sample_rate, model_sr, &mut resample_buf);
            let mel = sess.mel_from_audio_fast(&resample_buf);
            sess.audio_resample_buf = resample_buf;
            mel?
        };

        let t = mel_full.len() / sess.cfg.num_mels;
        if t == 0 {
            return Ok(vec![0.0; mono_pcm.len()]);
        }

        // 2. 构建 F0 + 共振峰偏移
        let hop_sec = (hop as f64) / (model_sr.max(1) as f64);
        let f0_full: Vec<f32> = (0..t)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                midi_to_hz(midi_at_time(abs_t))
            })
            .collect();

        // 3. 应用共振峰偏移（原地修改 mel_full）
        let formant_shifts: Vec<f32> = (0..t)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                formant_shift_at_time(abs_t)
            })
            .collect();
        let has_formant_shift = formant_shifts.iter().any(|s| s.abs() >= 0.5);
        if has_formant_shift {
            shift_mel_formant(
                &mut mel_full,
                sess.cfg.num_mels,
                t,
                &formant_shifts,
                sess.cfg.fmin,
                sess.cfg.fmax,
            );
        }

        // 4. 分块迭代 — 批量推理：收集所有未命中缓存的 chunk，一次 GPU 调用处理
        // 输出长度 = 重建内容真实长度（t×hop）。mel 提取的尾部窗损失使
        // t×hop < 输入长度；尾部对齐（含斜坡收尾）在步骤 5 统一完成，
        // 若此处直接按输入长度初始化并留零，会预先制造"内容↔零"缺口。
        let total_samples = t.saturating_mul(hop).max(1);
        let mut out = vec![0.0f32; total_samples];

        // 4a. 分离已缓存和需要推理的 chunk
        let mut cached_chunks: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut needs_inference: Vec<usize> = Vec::new();

        let mut frame_off = 0usize;
        while frame_off < t {
            let chunk_end = (frame_off + CHUNK_MAX_FRAMES).min(t);
            if let Some(cached) = chunk_cache_get(frame_off, chunk_end) {
                cached_chunks.push((frame_off, cached));
            } else {
                needs_inference.push(frame_off);
            }
            frame_off = chunk_end;
        }

        let total_chunks = (t + CHUNK_MAX_FRAMES - 1) / CHUNK_MAX_FRAMES;
        let processed_before = cached_chunks.len();
        debug_eprintln!(
            "[nsf_hifigan] chunked_opt: t={} chunks={} cached={} infer={}",
            t,
            total_chunks,
            cached_chunks.len(),
            needs_inference.len()
        );

        // Report progress for cached chunks
        if !cached_chunks.is_empty() {
            emit_chunk_progress(cached_chunks.len() as f64 / total_chunks as f64);
        }

        // 4b. 批量推理需要推理的 chunk
        if !needs_inference.is_empty() {
            let batch_items: Vec<(Vec<f32>, Vec<f32>, usize)> = needs_inference
                .iter()
                .map(|&fi| {
                    let chunk_end = (fi + CHUNK_MAX_FRAMES).min(t);
                    let chunk_t = chunk_end - fi;
                    let mut mel_seg = vec![0.0f32; sess.cfg.num_mels * chunk_t];
                    for m in 0..sess.cfg.num_mels {
                        let src = &mel_full[m * t + fi..m * t + chunk_end];
                        let dst = &mut mel_seg[m * chunk_t..(m + 1) * chunk_t];
                        dst.copy_from_slice(src);
                    }
                    let f0_seg = f0_full[fi..chunk_end].to_vec();
                    (mel_seg, f0_seg, chunk_t)
                })
                .collect();

            let t_batch = std::time::Instant::now();
            let batch_results = sess.run_model_batch(&batch_items)?;
            debug_eprintln!(
                "[nsf_hifigan] chunked_opt: batch_gpu={}ms for {} chunks",
                t_batch.elapsed().as_millis(),
                batch_results.len()
            );

            for (i, wf) in batch_results.into_iter().enumerate() {
                let fi = needs_inference[i];
                let chunk_end = (fi + CHUNK_MAX_FRAMES).min(t);
                chunk_cache_put(fi, chunk_end, wf.clone());
                cached_chunks.push((fi, wf));
                emit_chunk_progress((processed_before + i + 1) as f64 / total_chunks as f64);
            }
        }

        // 4c. 按帧偏移排序并合并到输出
        cached_chunks.sort_by_key(|&(fi, _)| fi);
        for (fi, wf) in &cached_chunks {
            let base_out = fi * hop;
            for (i, &sample) in wf.iter().enumerate() {
                let g = base_out + i;
                if g < out.len() {
                    out[g] = sample;
                }
            }
        }

        // 5. 重采样回原始采样率
        let mut out = if model_sr == sample_rate {
            out
        } else {
            linear_resample_mono(&out, model_sr, sample_rate)
        };

        // 对齐到输入长度。mel 提取的尾部窗损失（最后不足一帧 hop 的内容
        // 没有帧覆盖）使重建内容比输入短 1~2 帧（模型 hop），裸 resize 补
        // 零 / truncate 会在"内容末端 ↔ 补零"交界留下单帧硬切 —— 淡出增益
        // 在淡出区前段仍接近 1 时即末尾 Click（与 mel stretch 路径同根因）。
        // 修复：对齐边界处做 ~2 帧 hop 的线性收尾，让内容平滑落到 0。
        let target = mono_pcm.len();
        let ramp_samples = tail_ramp_samples(hop, model_sr, sample_rate);
        smooth_tail_then_align(&mut out, target, ramp_samples);

        Ok(out)
    })
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
#[allow(dead_code)]
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
    /// 与已移除的 `infer_from_audio_and_midi`（df4e17b4 删除）的思路一致，
    /// 但不需要预先对 PCM 做时间拉伸：而是在 mel 域完成时间拉伸，由 HiFiGAN
    /// 直接从插值后的 mel 合成波形。
    ///
    /// # 参数
    /// - `audio_mono`：**源速率**的原始 PCM（未拉伸）
    /// - `sample_rate`：PCM 采样率
    /// - `playback_rate`：播放速率（> 1.0 快放/缩短，< 1.0 慢放/拉长）
    /// - `start_sec`：该段在**时间轴**上的起始秒（已考虑拉伸后坐标）
    /// - `midi_at_time`：回调，参数为时间轴绝对时间（秒），返回目标 MIDI 值
    #[allow(dead_code)]
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

        // 8. 对齐到拉伸后的目标长度。
        // mel 提取的尾部窗损失（最后不足一帧 hop 的内容没有帧覆盖）与帧数
        // 取整会让重建输出比严格目标短 1~2 帧（模型 hop≈10ms）。裸 resize
        // 补零 / truncate 会在"内容末端 ↔ 补零/截断"的交界留下单帧硬切 ——
        // 淡出增益在淡出区前段仍接近 1（尤其"先慢后快"曲线），即末尾 Click
        //（HiFiGAN Mel Stretch 特有；外部精确拉伸器输出铺满目标，无此偏差）。
        // 修复：对齐边界处做 ~2 帧 hop 的线性收尾，让内容平滑落到 0，
        // 后续所有 pad/truncate（pitch_editing / 渲染装配）都切在 ≈0 上。
        let target_len = ((audio_mono.len() as f64) / playback_rate).round().max(0.0) as usize;
        let ramp_samples = tail_ramp_samples(self.cfg.hop_size, model_sr, sample_rate);
        smooth_tail_then_align(&mut out, target_len, ramp_samples);
        Ok(out)
    }
}

/// 模型 hop 换算到目标采样率后的 2 倍 —— 重建内容末端的线性收尾斜坡长度
/// （覆盖 mel 提取尾部窗损失 + 帧数取整的最大偏差）。
fn tail_ramp_samples(hop: usize, model_sr: u32, out_sr: u32) -> usize {
    let hop_out = (hop as u64)
        .saturating_mul(out_sr.max(1) as u64)
        .div_ceil(model_sr.max(1) as u64) as usize;
    hop_out.saturating_mul(2)
}

/// 对齐输出长度到 `target_len`，并把"内容末端 ↔ 补零/截断"的边界做成
/// 线性收尾（边界处 ≈0），避免重建内容在 fade 增益仍大时硬切。
fn smooth_tail_then_align(out: &mut Vec<f32>, target_len: usize, ramp: usize) {
    if out.len() > target_len {
        // 截断前：把 [target-ramp, target) 线性压到 ≈0，截断点落在收尾内。
        let start = target_len.saturating_sub(ramp);
        let n = target_len.saturating_sub(start);
        if n >= 2 {
            for i in start..target_len {
                let k = (target_len - i) as f32 / n as f32;
                out[i] *= k;
            }
        }
        out.truncate(target_len);
    } else if out.len() < target_len {
        // 补零前：把内容末端 [len-ramp, len) 线性压到 ≈0，补零区从 ≈0 开始。
        let start = out.len().saturating_sub(ramp);
        let n = out.len().saturating_sub(start);
        if n >= 2 {
            for i in start..out.len() {
                let k = (out.len() - i) as f32 / n as f32;
                out[i] *= k;
            }
        }
        out.resize(target_len, 0.0);
    }
}

/// 单次 mel stretch 推理入口（thread-local session）。
///
/// 参数语义与已移除的 `infer_pitch_edit_mono`（df4e17b4 删除）相似，但额外
/// 接收 `playback_rate` 并在 mel 域完成时间拉伸，省去外部预处理。
#[allow(dead_code)]
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
#[allow(dead_code)]
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

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResults {
    pub cpu_median_ms: f64,
    pub cpu_rt_factor: f64,
    pub gpu_median_ms: Option<f64>,
    pub gpu_rt_factor: Option<f64>,
    pub dml_median_ms: Option<f64>,
    pub dml_rt_factor: Option<f64>,
    pub benchmark_samples: usize,
    /// True when WebGPU EP was available for the GPU benchmark.
    pub gpu_available: bool,
    /// Display name of the GPU backend used by the benchmark ("CoreML" on
    /// macOS ARM64, "WebGPU" on Linux x86_64).
    pub gpu_backend_name: String,
    /// Detailed error message when the GPU benchmark could not be completed
    /// (None when the GPU benchmark succeeded or was not attempted).
    pub gpu_error: Option<String>,
    /// True when DirectML EP was available for the benchmark.
    pub dml_available: bool,
    /// GPU device ID that was used (0 if GPU not available).
    pub gpu_device_id: i32,
    /// Execution providers available in the ONNX Runtime DLL.
    pub available_providers: Vec<String>,
    /// ORT build info string.
    pub ort_build_info: String,
    /// All GPUs discovered via NVML (name, memory, device ID).
    pub gpu_devices: Vec<crate::gpu_info::GpuDeviceInfo>,
    /// All DirectML-compatible GPU adapters discovered via DXGI.
    pub dml_adapters: Vec<crate::dml_adapters::DmlAdapterInfo>,
}

/// Build input tensors for a session using its declared metadata, filling
/// dynamic dimensions with the benchmark's frame count (batch=1).
fn build_benchmark_inputs(
    session: &Session,
    frames: usize,
) -> Result<Vec<(String, ort::value::Value)>, String> {
    use ort::value::{Tensor, ValueType};
    let mut pairs = Vec::new();
    for input in session.inputs() {
        let (ty, shape) = match input.dtype() {
            ValueType::Tensor { ty, shape, .. } => (ty, shape),
            _ => continue,
        };
        if *ty != ort::value::TensorElementType::Float32 {
            continue;
        }
        let test_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0 {
                    d as usize
                } else if i == 0 {
                    1
                } else {
                    frames
                }
            })
            .collect();
        let total: usize = test_shape.iter().product::<usize>().max(1);
        // Non-zero f0 keeps the model's f0 differential (Pad data) valid.
        let fill = if input.name() == "f0" {
            440.0f32
        } else {
            0.0f32
        };
        let data: Vec<f32> = vec![fill; total];
        let tensor = Tensor::from_array((test_shape, data.into_boxed_slice()))
            .map_err(|e| format!("build benchmark tensor '{}' failed: {e}", input.name()))?;
        pairs.push((input.name().to_string(), tensor.into()));
    }
    if pairs.is_empty() {
        return Err("benchmark: no f32 tensor inputs found".to_string());
    }
    Ok(pairs)
}

/// Run one session inference on a helper thread with a timeout so a hung GPU
/// backend can never freeze the benchmark.  Returns Ok(Some(ms)) on success,
/// Ok(None) on timeout, Err on inference failure.
fn timed_session_run(
    session: &Arc<Mutex<Session>>,
    input_pairs: Vec<(String, ort::value::Value)>,
    timeout: std::time::Duration,
) -> Result<Option<f64>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let sess = Arc::clone(session);
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let result = sess
            .lock()
            .map_err(|e| e.to_string())
            .and_then(|mut guard| {
                guard
                    .run(input_pairs)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            });
        let _ = tx.send((t0.elapsed(), result));
    });
    match rx.recv_timeout(timeout) {
        Ok((elapsed, Ok(()))) => Ok(Some(elapsed.as_secs_f64() * 1000.0)),
        Ok((_, Err(e))) => Err(e),
        Err(_) => Ok(None),
    }
}

/// Mel frames fed to the model by the built-in benchmark.
///
/// Sessions keep the model's dynamic `time` axis on every platform, so a
/// single budget works everywhere.  1024 frames (~11.9 s of audio at hop 512
/// / 44.1 kHz) keeps one CPU run under a second while still being large
/// enough to amortise GPU dispatch overhead.
const BENCHMARK_FRAMES: usize = 1024;

/// The GPU execution providers the benchmark should try, in priority order.
///
/// This is deliberately not derived from `diagnose_available_providers()`
/// alone: that helper only reports whether an EP *registers*, and the macOS
/// WebGPU EP registers fine yet fails every inference.  Ordering matters too
/// — on Apple Silicon CoreML is the primary GPU path and must be tried before
/// Dawn/Metal.  Windows is excluded because DirectML is benchmarked
/// separately (see the `dml_*` result fields).
fn gpu_ep_candidates() -> Vec<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return vec!["coreml", "webgpu"];
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return vec!["webgpu"];
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    return vec![];
}

/// True when ORT's runtime probe reported `provider` as usable.
fn provider_probed_available(available_providers: &[String], provider: &str) -> bool {
    match provider {
        "coreml" => available_providers
            .iter()
            .any(|p| p == "CoreMLExecutionProvider"),
        "webgpu" => available_providers
            .iter()
            .any(|p| p == "WebGpuExecutionProvider"),
        _ => false,
    }
}

/// Benchmark a single GPU execution provider for the vocoder.
///
/// Returns `(median_ms, rt_factor, ep_name)` when the EP both registered and
/// completed every timed run, otherwise a human-readable reason why it could
/// not be measured.  A hung CoreML EP is disabled process-wide so it cannot
/// stall a later render.
fn benchmark_gpu_ep(
    onnx_path: &Path,
    frames: usize,
    audio_sec: f64,
    runs: usize,
    gpu_ep_choice: &str,
) -> Result<(f64, f64, String), String> {
    let gpu_session_res = {
        let _guard = crate::vocoder_ort_session::EpOverrideGuard::new(gpu_ep_choice.to_string());
        crate::vocoder_ort_session::build_ort_session(
            onnx_path,
            crate::vocoder_ort_session::OrtSessionRole::Vocoder,
        )
    };

    let (gpu_session, ep) = gpu_session_res.map_err(|e| {
        log::error!("[benchmark] GPU session creation FAILED for '{gpu_ep_choice}': {e}");
        e
    })?;

    if ep != gpu_ep_choice {
        return Err(format!(
            "GPU session creation fell back to CPU (requested {gpu_ep_choice}, got {ep}). \
             Check the application log for the detailed error."
        ));
    }

    log::warn!("[benchmark] GPU session created: ep={ep}");
    let gpu_session = Arc::new(Mutex::new(gpu_session));
    let timeout = std::time::Duration::from_secs(120);

    // Warmup on a helper thread (same execution model as the session smoke
    // test) so a hung EP inference can never freeze the benchmark.
    {
        let guard = gpu_session.lock().map_err(|e| e.to_string())?;
        let inputs = build_benchmark_inputs(&guard, frames)?;
        drop(guard);
        match timed_session_run(&gpu_session, inputs, timeout) {
            Ok(Some(_)) => {}
            Ok(None) => {
                let msg = format!("{ep} warmup inference timed out after {timeout:?}");
                log::warn!("[benchmark] WARNING: {msg}");
                if ep == "coreml" {
                    crate::vocoder_ort_session::disable_coreml("benchmark warmup timed out");
                }
                return Err(msg);
            }
            Err(e) => {
                let msg = format!("{ep} warmup inference failed: {e}");
                log::warn!("[benchmark] WARNING: {msg}");
                return Err(msg);
            }
        }
    }

    let mut gpu_times = Vec::new();
    for _ in 0..runs {
        let guard = gpu_session.lock().map_err(|e| e.to_string())?;
        let inputs = build_benchmark_inputs(&guard, frames)?;
        drop(guard);
        match timed_session_run(&gpu_session, inputs, timeout) {
            Ok(Some(ms)) => gpu_times.push(ms),
            Ok(None) => {
                let msg = format!("{ep} inference timed out after {timeout:?}");
                log::warn!("[benchmark] WARNING: {msg}");
                if ep == "coreml" {
                    crate::vocoder_ort_session::disable_coreml("benchmark inference timed out");
                }
                return Err(msg);
            }
            Err(e) => {
                let msg = format!("{ep} inference failed: {e}");
                log::warn!("[benchmark] WARNING: {msg}");
                return Err(msg);
            }
        }
    }

    if gpu_times.len() < 2 {
        return Err(format!("{ep} did not complete any timed run"));
    }

    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = gpu_times[gpu_times.len() / 2];
    let rtf = audio_sec / (median / 1000.0);
    log::warn!("[benchmark] GPU({ep}): median={median:.1}ms rtf={rtf:.3}x");
    Ok((median, rtf, ep))
}

pub fn run_benchmark() -> Result<BenchmarkResults, String> {
    ensure_ort_init()?;
    let (onnx_path, cfg_path) = resolve_model_paths()?;
    let cfg = read_config(&cfg_path)?;

    // Sessions keep the model's dynamic `time` axis on every platform, so the
    // benchmark is free to pick a single frame budget for all of them.  1024
    // frames (~11.9 s of audio at hop 512 / 44.1 kHz) keeps one run well under
    // a second on CPU while staying large enough to amortise GPU dispatch.
    let frames = BENCHMARK_FRAMES;
    let audio_sec = (frames as f64) * (cfg.hop_size as f64) / (cfg.sampling_rate as f64);
    let runs = 5;

    // Collect diagnostic info before benchmark
    let available_providers = crate::vocoder_ort_session::diagnose_available_providers();
    let gpu_device_id = crate::vocoder_ort_session::diagnose_gpu().gpu_device_id;
    let ort_build_info = std::panic::catch_unwind(|| ort::info().to_string())
        .unwrap_or_else(|_| "ort::info() unavailable".to_string());
    // A GPU EP counts as available only when this platform has a candidate
    // for it AND ORT's runtime probe reported it.  Deriving this from
    // `available_providers` alone was wrong: the list is platform-agnostic, so
    // a machine with working CoreML but a failing WebGPU probe used to skip
    // the GPU benchmark entirely.
    let gpu_candidates = gpu_ep_candidates();
    let gpu_available = gpu_candidates
        .iter()
        .any(|ep| provider_probed_available(&available_providers, ep));
    let gpu_devices = crate::gpu_info::enumerate_gpus().devices;
    let dml_adapters = crate::dml_adapters::enumerate_dml_adapters().adapters;
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);

    log::warn!("[benchmark] ========================================");
    log::warn!(
        "[benchmark] model={} frames={frames} audio_sec={audio_sec:.2}s runs={runs}",
        onnx_path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default()
    );
    log::warn!(
        "[benchmark] model_sr={} num_mels={} hop={} n_fft={}",
        cfg.sampling_rate, cfg.num_mels, cfg.hop_size, cfg.n_fft
    );
    log::warn!("[benchmark] cpu_cores={cpu_cores} ort={ort_build_info}");
    log::warn!("[benchmark] providers={available_providers:?}");
    log::warn!("[benchmark] dml_adapters={dml_adapters:?}");
    log::warn!("[benchmark] gpu_devices(NVML)={gpu_devices:?}");
    log::warn!("[benchmark] env HIFISHIFTER_ORT_EP={:?} HIFISHIFTER_HIFIGAN_ORT_EP={:?} HIFISHIFTER_DML_DEVICE_ID={:?}",
        std::env::var("HIFISHIFTER_ORT_EP").ok(),
        std::env::var("HIFISHIFTER_HIFIGAN_ORT_EP").ok(),
        std::env::var("HIFISHIFTER_DML_DEVICE_ID").ok());

    // 1. Benchmark CPU
    let mut cpu_times = Vec::new();
    let t_cpu_total = std::time::Instant::now();
    {
        let _guard = crate::vocoder_ort_session::EpOverrideGuard::new("cpu".to_string());
        let t_session = std::time::Instant::now();
        let (mut cpu_session, _) = crate::vocoder_ort_session::build_ort_session(
            &onnx_path,
            crate::vocoder_ort_session::OrtSessionRole::Vocoder,
        )?;
        log::warn!(
            "[benchmark] CPU session created in {}ms",
            t_session.elapsed().as_millis()
        );

        // Warmup
        let mel = vec![0.0f32; cfg.num_mels * frames];
        let f0 = vec![440.0f32; frames];
        let mt = Tensor::from_array(([1, cfg.num_mels, frames], mel.clone().into_boxed_slice()))
            .unwrap();
        let ft = Tensor::from_array(([1, frames], f0.clone().into_boxed_slice())).unwrap();
        let _ = cpu_session.run(ort::inputs![mt, ft]).unwrap();

        for _ in 0..runs {
            let mt =
                Tensor::from_array(([1, cfg.num_mels, frames], mel.clone().into_boxed_slice()))
                    .unwrap();
            let ft = Tensor::from_array(([1, frames], f0.clone().into_boxed_slice())).unwrap();
            let t = std::time::Instant::now();
            let _ = cpu_session.run(ort::inputs![mt, ft]).unwrap();
            cpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_median = cpu_times[cpu_times.len() / 2];
    let cpu_rt_factor = audio_sec / (cpu_median / 1000.0);
    log::warn!(
        "[benchmark] CPU: total={}ms runs={:?} median={cpu_median:.1}ms rtf={cpu_rt_factor:.3}x",
        t_cpu_total.elapsed().as_millis(),
        cpu_times
    );

    // 2. Benchmark GPU.  Candidates come from `gpu_ep_candidates()` so the
    // platform's primary GPU EP is tried first and a broken secondary EP
    // (macOS WebGPU registers but cannot infer) never masks a working one.
    let mut gpu_median = None;
    let mut gpu_rt_factor = None;
    let mut gpu_actually_working = false;
    let mut gpu_ep_name = String::new();
    let mut gpu_error: Option<String> = None;

    if gpu_available {
        for candidate in &gpu_candidates {
            match benchmark_gpu_ep(&onnx_path, frames, audio_sec, runs, candidate) {
                Ok((median, rtf, ep)) => {
                    gpu_median = Some(median);
                    gpu_rt_factor = Some(rtf);
                    gpu_ep_name = ep;
                    gpu_actually_working = true;
                    gpu_error = None;
                    break;
                }
                Err(e) => {
                    log::error!("[benchmark] GPU candidate '{candidate}' unusable: {e}");
                    gpu_error = Some(e);
                }
            }
        }
    }

    // 3. Benchmark DirectML if available
    let dml_available = available_providers.iter().any(|p| p.contains("Dml"));
    let mut dml_median = None;
    let mut dml_rt_factor = None;

    if dml_available {
        let dml_total = std::time::Instant::now();
        let dml_session_res = {
            let _guard = crate::vocoder_ort_session::EpOverrideGuard::new("directml".to_string());
            crate::vocoder_ort_session::build_ort_session(
                &onnx_path,
                crate::vocoder_ort_session::OrtSessionRole::Vocoder,
            )
        };

        if let Ok((mut dml_session, ep)) = dml_session_res {
            log::warn!(
                "[benchmark] DirectML session created in {}ms (ep={ep})",
                dml_total.elapsed().as_millis()
            );
            if ep == "directml" {
                let mut dml_times = Vec::new();
                let mel = vec![0.0f32; cfg.num_mels * frames];
                let f0 = vec![440.0f32; frames];

                // Warmup
                let mt =
                    Tensor::from_array(([1, cfg.num_mels, frames], mel.clone().into_boxed_slice()))
                        .unwrap();
                let ft = Tensor::from_array(([1, frames], f0.clone().into_boxed_slice())).unwrap();
                if dml_session.run(ort::inputs![mt, ft]).is_ok() {
                    for run_i in 0..runs {
                        let mt = Tensor::from_array((
                            [1, cfg.num_mels, frames],
                            mel.clone().into_boxed_slice(),
                        ))
                        .unwrap();
                        let ft = Tensor::from_array(([1, frames], f0.clone().into_boxed_slice()))
                            .unwrap();
                        let t = std::time::Instant::now();
                        let _ = dml_session.run(ort::inputs![mt, ft]).unwrap();
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        log::warn!("[benchmark] DirectML run {run_i}: {ms:.1}ms");
                        dml_times.push(ms);
                    }
                    dml_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = dml_times[dml_times.len() / 2];
                    dml_median = Some(median);
                    dml_rt_factor = Some(audio_sec / (median / 1000.0));
                    log::info!("[benchmark] DirectML: total={}ms runs={dml_times:?} median={median:.1}ms rtf={:.3}x",
                        dml_total.elapsed().as_millis(), audio_sec / (median / 1000.0));
                } else {
                    log::error!(
                        "[benchmark] WARNING: DirectML EP registered but warmup inference FAILED."
                    );
                }
            }
        } else {
            log::error!("[benchmark] DirectML session creation FAILED");
        }
    }

    // Log diagnostic info for debugging
    log::warn!(
        "[benchmark] Providers: {:?} | GPU device_id: {} | GPU works: {} | DirectML available: {}",
        available_providers, gpu_device_id, gpu_actually_working, dml_available
    );

    let gpu_backend_name = match gpu_ep_name.as_str() {
        "coreml" => "CoreML",
        "webgpu" => "WebGPU",
        _ => {
            if cfg!(target_os = "macos") {
                "CoreML"
            } else {
                "WebGPU"
            }
        }
    }
    .to_string();

    Ok(BenchmarkResults {
        cpu_median_ms: cpu_median,
        cpu_rt_factor,
        gpu_median_ms: gpu_median,
        gpu_rt_factor,
        dml_median_ms: dml_median,
        dml_rt_factor,
        benchmark_samples: runs,
        gpu_available,
        gpu_backend_name,
        gpu_error,
        dml_available,
        gpu_device_id,
        available_providers,
        ort_build_info,
        gpu_devices,
        dml_adapters,
    })
}

#[cfg(test)]
mod tests {
    use super::smooth_tail_then_align;

    /// mel stretch 对齐收尾：补零前内容末端平滑落到 ≈0，补零区从 ≈0 开始，
    /// 边界不再有单帧硬切（修复"HiFiGAN Mel Stretch 尾部 Click"的根因）。
    #[test]
    fn align_pads_short_output_with_smooth_tail() {
        let mut out = vec![1.0f32; 100];
        // 内容 100 → 目标 250，收尾 50：末尾从 1 平滑降到 ~0.02，然后补零。
        smooth_tail_then_align(&mut out, 250, 50);
        assert_eq!(out.len(), 250);
        // 前半保持原样
        assert_eq!(out[0], 1.0);
        assert_eq!(out[49], 1.0);
        // 收尾严格单调递减
        for i in 50..99 {
            assert!(
                out[i] > out[i + 1],
                "tail must be strictly decreasing at {i}: {} vs {}",
                out[i],
                out[i + 1]
            );
        }
        // 末帧（收尾终点）与补零区起点都 ≈0
        assert!(out[99].abs() < 0.03, "last content frame: {}", out[99]);
        assert_eq!(out[100], 0.0);
        assert_eq!(out[249], 0.0);
    }

    #[test]
    fn align_truncates_long_output_on_smooth_tail() {
        let mut out = vec![1.0f32; 300];
        // 内容 300 → 目标 200，收尾 50：截断点落在收尾末端（≈0），
        // 尾后即输出边界 —— 与组装层后续 pad/truncate 的切点一致。
        smooth_tail_then_align(&mut out, 200, 50);
        assert_eq!(out.len(), 200);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[149], 1.0);
        for i in 150..199 {
            assert!(
                out[i] > out[i + 1],
                "tail must be strictly decreasing at {i}: {} vs {}",
                out[i],
                out[i + 1]
            );
        }
        assert!(out[199].abs() < 0.03, "truncated edge: {}", out[199]);
    }

    #[test]
    fn align_exact_length_is_untouched() {
        let mut out = vec![0.5f32; 120];
        smooth_tail_then_align(&mut out, 120, 50);
        assert_eq!(out.len(), 120);
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn align_tiny_ramp_degrades_gracefully() {
        // ramp 过小（或不合理输入）时不得 panic/破坏长度。
        let mut out = vec![1.0f32; 4];
        smooth_tail_then_align(&mut out, 2, 0);
        assert_eq!(out.len(), 2);
        let mut out2 = vec![1.0f32; 2];
        smooth_tail_then_align(&mut out2, 6, 4);
        assert_eq!(out2.len(), 6);
    }
}
