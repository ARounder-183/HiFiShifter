//! NSF-HiFiGAN真实模型、mel分析与有界神经分块；HNSEP仍由独立模块整段处理。

use ndarray::Array2;
use num_complex::Complex32;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::Fft;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static NEURAL_RUNS: AtomicU64 = AtomicU64::new(0);

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
            NEURAL_RUNS.fetch_add(1,Ordering::Relaxed);
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

    // 开发树兜底：模型仍放在 app crate 的 `resources/` 下（那才是打包时带的东西）。
    // 内核是从 app 里搬出来的，所以必须回看得到它 —— 否则内核自己的测试会因为
    // 「模型不在内核目录下」而失败，而那不是被测代码的问题。
    if let Some(app_p) = manifest
        .parent()
        .map(|p| p.join("src-tauri").join("resources").join("models").join("nsf_hifigan"))
    {
        let has_model = app_p.join(vocoder_model_filename()).is_file()
            || app_p.join("pc_nsf_hifigan.onnx").is_file();
        if has_model && app_p.join("config.json").is_file() {
            return Some(app_p);
        }
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

pub fn probe_load() -> Result<String, String> {
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

/// keyShift 的量化步长（半音）。
///
/// 1/4 半音 = 25 cents。取值权衡：越细越接近 OpenUtau 的连续曲线，但档位数
/// 直接决定 [`ShiftPlan`] 的数量（每档一个 FFT plan）。±12 半音 → 97 档，
/// 25 cents 的量化误差远低于人耳对共振峰位置的 JND。
const KEY_SHIFT_QUANTUM_SEMITONES: f32 = 0.25;

/// 量化 keyShift 到有限档位。
///
/// 【为什么必须量化】连续曲线会让每帧产出不同的 `n_fft`，导致逐帧重建 FFT plan。
/// 量化把 plan 总数收敛到常数（档数）。量化在**keyShift 域**而非 `n_fft` 域进行，
/// 保证同一档位总是映射到同一 `n_fft`。
pub fn quantize_key_shift(key_shift_semitones: f32) -> f32 {
    if !key_shift_semitones.is_finite() {
        return 0.0;
    }
    (key_shift_semitones / KEY_SHIFT_QUANTUM_SEMITONES).round() * KEY_SHIFT_QUANTUM_SEMITONES
}

/// 由量化后的 keyShift 求该档位的 FFT 长度。
///
/// `n_fft_new = round(n_fft * 2^(keyShift/12))`，下限 4（`rustfft` 对小尺寸的
/// 可用性下限；OpenUtau 同样要求 `length >= 4`），并保证不超过两倍基准长度
/// （keyShift 已钳在 ±12 半音内，此处只是数值兜底）。
pub fn shift_n_fft(key_shift_semitones: f32, base_n_fft: usize, base_win: usize) -> usize {
    let factor = 2.0f32.powf(key_shift_semitones / 12.0);
    let n = (base_n_fft as f32 * factor).round();
    let n = if n.is_finite() && n >= 4.0 {
        n as usize
    } else {
        4
    };
    n.min(base_n_fft.max(base_win).saturating_mul(2).max(4))
}

/// 一个量化 keyShift 档位对应的分析配置（FFT plan + 窗 + 幅度缩放 + reflect pad）。
///
/// 与 [`NsfHifiganOnnx::mel_from_audio_fast`] 的固定 2048 plan 不同，
/// 逐帧 keyShift 需要按 `n_fft` 变化的 plan；本结构把每档的构造结果缓存下来。
struct ShiftPlan {
    /// 该档位的 FFT 长度（= 窗长，OpenUtau 中 `n_fft == win_size`）。
    n_fft: usize,
    fft: Arc<dyn Fft<f32>>,
    /// 居中到 `n_fft` 长度的窗（窗长可能小于 `n_fft`，torch 会居中补齐）。
    window: Vec<f32>,
    frame_buf: Vec<Complex32>,
    /// `base_win / win_size_new`：补偿窗长变化对幅度谱的整体缩放。
    bin_scale: f32,
    /// reflect pad 左端长度 `(win_size_new - hop) / 2`。
    pad_left: usize,
}

impl ShiftPlan {
    /// 构造一档分析配置。
    ///
    /// 窗长 `win_size_new = round(base_win * factor)`，FFT 长度取同一个值
    /// （OpenUtau 的 `PitchAdjustableMelSpectrogram` 中两者始终相等）；
    /// 窗按 torch 约定居中放入 `n_fft` 长度的帧缓冲。
    /// 构造一档分析配置。
    ///
    /// **窗长恒等于 FFT 长度**（`win_size_new == n_fft_new`），这是 OpenUtau
    /// `PitchAdjustableMelSpectrogram` 的契约：两者都由 `round(2048 * factor)`
    /// 得到。该等式同时保证**帧数与 keyShift 无关**——
    /// `padded_len = n + win_new - hop`，`frames = 1 + (padded_len - nfft_new)/hop
    /// = 1 + (n - hop)/hop`，`win_new == nfft_new` 时 `factor` 被约掉。
    /// 若两者不等，帧数会随曲线变化，mel 与 f0 的时间轴将无法对齐。
    fn new(
        key_shift_semitones: f32,
        n_fft: usize,
        base_win: usize,
        hop: usize,
    ) -> Result<Self, String> {
        if hop == 0 || n_fft == 0 {
            return Err("mel: invalid config".to_string());
        }
        // 窗长 = FFT 长度（见上方契约说明）。窗口直接用对称 Hann，
        // 不做居中补齐——两者等长时居中偏移为 0。
        let win_size = n_fft;
        let window = hann_window(win_size);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // binScale 补偿窗长变化：OpenUtau 用 winSize / winSizeNew。
        // keyShift == 0 时为 1.0，保证与 mel_from_audio_fast 一致。
        let bin_scale = if key_shift_semitones.abs() > 1e-6 {
            base_win.max(1) as f32 / win_size.max(1) as f32
        } else {
            1.0
        };

        let pad_left = ((win_size as isize - hop as isize) / 2).max(0) as usize;

        Ok(Self {
            n_fft,
            fft,
            window,
            frame_buf: vec![Complex32::new(0.0, 0.0); n_fft],
            bin_scale,
            pad_left,
        })
    }
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
struct SharedVocoder {
    runtime:Arc<Mutex<Session>>,
    cfg:NsfHifiganConfig,
    identity:blake3::Hash,
}
static SHARED_SESSION: OnceLock<Mutex<Option<Arc<SharedVocoder>>>> = OnceLock::new();

/// 流式哈希模型/config完整内容；只在worker建会话时运行，不靠路径/大小/首尾猜版本。
fn digest_model_files(onnx:&Path,config:&Path)->Result<blake3::Hash,String> {
    use std::io::Read;
    let mut hash=blake3::Hasher::new();hash.update(b"hifigan-model-and-config-v1");
    for path in [onnx,config] {
        let mut file=std::fs::File::open(path).map_err(|e|format!("vocoder identity open failed: {e}"))?;
        let length=file.metadata().map_err(|e|format!("vocoder identity metadata failed: {e}"))?.len();
        hash.update(&length.to_le_bytes());let mut buffer=[0_u8;64*1024];
        loop {let count=file.read(&mut buffer).map_err(|e|format!("vocoder identity read failed: {e}"))?;
            if count==0 {break;}hash.update(&buffer[..count]);}
    }Ok(hash.finalize())
}

/// 真实已加载model/config/EP的内容命名空间；调用方仅在离线worker进入处理器时读取。
pub fn cache_identity()->Result<String,String> {Ok(get_or_init_shared_session()?.identity.to_hex().to_string())}
/// 实际模型run次数（包含失败尝试，不含建会话烟测），用于非实时性能验收。
pub fn inference_runs()->u64 {NEURAL_RUNS.load(Ordering::Relaxed)}

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
///
/// ★ 构建流程的关键约束：**容器锁绝不跨构建持有**。会话构建是秒级操作
/// （DirectML shader 预编译、CoreML 图编译），且可能因驱动问题挂起 ——
/// 旧实现持容器锁构建，导致构建期间设备切换（`update_ort_ep`）、其他渲染
/// 线程、乃至关机清理全部卡死（实测：设备切换后渲染进度永久卡在 0%，
/// 退出时 `could not acquire SHARED_SESSION lock at shutdown`）。
///
/// 正确流程：容器锁快速检查 → **释放容器锁** → 全局构建锁内构建
/// （与 FCPE / HNSEP 的构建串行化，避免并发 D3D12 初始化在驱动层死锁）
/// → 容器锁写回。构建前后比对 EP 设置代数：构建期间用户再次切换设备时
/// 丢弃陈旧结果并按当前设置重建（最多重试 3 次）。D3D12/DML 的**创建**由
/// `session_build_lock` 在构建器内部串行化，**烟测在创建锁之外**执行。
fn get_or_init_shared_session() -> Result<Arc<SharedVocoder>, String> {
    let mutex = SHARED_SESSION.get_or_init(|| Mutex::new(None));
    // 快路径：已有会话直接克隆返回。
    if let Some(session) = mutex
        .lock()
        .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?
        .clone()
    {
        return Ok(session);
    }

    // ★ 全局单飞锁：同一时刻全局只允许一次会话构建（创建 + 烟测整体）。
    // DirectML 的设备创建与首次推理都不能与其他 DML 操作并发 —— 两种并发
    // 都会在驱动层挂起（见 ort_session::session_build_lock 的说明）。同一
    // 模块的第二个到达者在此等待，构建完成后经下方的双重检查直接复用结果，
    // 因此**不会出现同一模块并发构建**。
    // 挂起由烟测超时兜底：持有者最多持锁一个 SMOKE_TEST_TIMEOUT
    // （DirectML 10s），随后 DirectML 被禁用、等待者立即以 CPU 继续 ——
    // 等待有界，绝不永久卡死。
    // 有界等待（20s）：持有者的构建挂起时不再让渲染线程永久阻塞 ——
    // 超时返回 Err，本次会话加载失败 → 该 Clip 失败但 pass 继续推进。
    let _build_flight =
        crate::vocoder_ort_session::acquire_session_build_lock(std::time::Duration::from_secs(20))?;
    // 双重检查：等待期间其他线程（设备切换的异步预热）可能已完成构建。
    if let Some(session) = mutex
        .lock()
        .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?
        .clone()
    {
        return Ok(session);
    }

    // 构建期间用户再次切换设备（EP 设置代数推进）会让本次构建结果过时 ——
    // 丢弃并按当前设置重建，最多 3 次。
    for _ in 0..3 {
        let generation_before = crate::vocoder_ort_session::ep_settings_generation();
        ensure_ort_init()?;
        let (onnx_path,cfg_path) = resolve_model_paths()?;
        let digest=digest_model_files(&onnx_path,&cfg_path)?;let cfg=read_config(&cfg_path)?;
        let (session,ep)=crate::vocoder_ort_session::build_ort_session(&onnx_path,crate::vocoder_ort_session::OrtSessionRole::Vocoder)?;
        if digest_model_files(&onnx_path,&cfg_path)?!=digest {return Err("vocoder model/config changed during session build".into());}
        let mut identity=blake3::Hasher::new();identity.update(digest.as_bytes());identity.update(ep.as_bytes());
        identity.update(&crate::synth_clip_cache::RENDER_PIPELINE_VERSION.to_le_bytes());
        identity.update(&chunk_max_frames().to_le_bytes());
        set_active_ep(&ep);
        if crate::vocoder_ort_session::ep_settings_generation() != generation_before {
            log::warn!("[nsf_hifigan] inference device changed during session build — rebuilding with the new EP");
            continue;
        }
        let mut guard = mutex
            .lock()
            .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?;
        // 其他线程（设备切换的异步预热 / 并发的渲染线程）可能已完成构建：
        // 复用已存在的会话，避免无谓替换与重复烟测。
        if let Some(existing) = guard.as_ref() {
            return Ok(existing.clone());
        }
        let arc = Arc::new(SharedVocoder {runtime:Arc::new(Mutex::new(session)),cfg,identity:identity.finalize()});
        *guard = Some(arc.clone());
        return Ok(arc);
    }
    Err("inference device kept changing during session build".to_string())
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

    // 异步预热：立即在后台用新 EP 重建会话。构建可能耗时数秒（DML shader
    // 预编译 / CoreML 图编译）甚至因驱动问题失败 —— 预热把它移出渲染热路径，
    // 渲染线程只需等构建锁（且双重检查后直接复用预热结果，不再重复构建）。
    // 失败仅记录日志：下一次渲染请求的 load 会再次尝试构建。
    std::thread::Builder::new()
        .name("ort-session-prewarm-vocoder".into())
        .spawn(|| {
            if let Err(e) = get_or_init_shared_session() {
                log::warn!("[nsf_hifigan] session pre-warm failed: {e}");
            }
        })
        .ok();
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
    /// 逐帧 keyShift（gender / 共振峰偏移）用的分析配置缓存，按 `n_fft` 索引。
    /// keyShift 已被 [`quantize_key_shift`] 量化，因此条目数上界 = 档位数（常数）。
    shift_plans: HashMap<usize, ShiftPlan>,
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
        let shared=get_or_init_shared_session()?;
        let cfg=shared.cfg.clone();

        if cfg.sampling_rate == 0 || cfg.num_mels == 0 || cfg.hop_size == 0 || cfg.n_fft == 0 {
            return Err("invalid NSF-HiFiGAN config.json".to_string());
        }

        // 获取（或初始化）全局共享 Session，消除每线程冷启动。
        let session = shared.runtime.clone();

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
            shift_plans: HashMap::new(),
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

    /// 逐帧 keyShift 的 mel 提取（gender / 共振峰偏移）。
    ///
    /// 对齐 OpenUtau hifisampler `PitchAdjustableMelSpectrogram`：按
    /// `factor = 2^(keyShift/12)` 伸缩 **FFT 长度与窗长**（hop 不变），用**原始
    /// mel 基**投影，幅度谱多截少补零并乘 `binScale`。分析窗被伸缩使频谱包络
    /// 整体平移，从而实现共振峰移动。
    ///
    /// 调用方须保证 `shifts.len() == 帧数`（帧数见 [`mel_frame_count`]）；
    /// 全 0 时调用方应改走 [`Self::mel_from_audio_fast`]（见
    /// [`extract_mel_with_shifts`]），以保证零偏移路径逐样本不变。
    ///
    /// # 与 OpenUtau 的两处有意差异
    /// - **窗函数用对称 Hann**（HiFiShifter 全链路约定），非 OpenUtau 的周期 Hann。
    /// - **keyShift 已由调用方量化**到有限档位，避免逐帧重建 FFT plan。
    fn mel_from_audio_shifted(
        &mut self,
        audio: &[f32],
        shifts: &[f32],
    ) -> Result<Vec<f32>, String> {
        compute_shifted_mel(
            audio,
            shifts,
            &self.cfg,
            &self.mel_fb_matrix,
            &mut self.shift_plans,
        )
    }

    #[allow(dead_code)]
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
                self.session = get_or_init_shared_session()?.runtime.clone();

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
            NEURAL_RUNS.fetch_add(1,Ordering::Relaxed);
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

/// 逐帧 keyShift 的 mel 计算核心（独立于 ORT session，便于测试）。
///
/// 由 [`NsfHifiganOnnx::mel_from_audio_shifted`] 委托。参数化 `cfg` /
/// `mel_fb_matrix` / `shift_plans` 使单元测试无需构建推理会话即可验证
/// 频谱搬移行为。
fn compute_shifted_mel(
    audio: &[f32],
    shifts: &[f32],
    cfg: &NsfHifiganConfig,
    mel_fb_matrix: &Array2<f32>,
    shift_plans: &mut HashMap<usize, ShiftPlan>,
) -> Result<Vec<f32>, String> {
    let hop = cfg.hop_size;
    let n_freqs = cfg.n_fft / 2 + 1;
    let n_mels = cfg.num_mels;

    if audio.is_empty() {
        let fill = dynamic_range_compression_ln(0.0);
        return Ok(vec![fill; n_mels]);
    }
    if hop == 0 || cfg.n_fft == 0 {
        return Err("mel: invalid config".to_string());
    }

    let n_frames = mel_frame_count(audio.len(), hop);
    if shifts.len() != n_frames {
        return Err(format!(
            "mel: {} key shifts for {} frames",
            shifts.len(),
            n_frames
        ));
    }

    // 阶段 1：确保本段用到的每个档位都有 plan。
    let mut frame_nfft = Vec::with_capacity(n_frames);
    for &shift in shifts {
        let n_fft = shift_n_fft(shift, cfg.n_fft, cfg.win_size);
        if !shift_plans.contains_key(&n_fft) {
            let plan = ShiftPlan::new(shift, n_fft, cfg.win_size, hop)?;
            shift_plans.insert(n_fft, plan);
        }
        frame_nfft.push(n_fft);
    }

    // 阶段 2：逐帧幅度谱按 [n_freqs, n_frames] 累积。
    // 复用同一条固定 mel 基矩阵做矩阵乘法（阶段 3），与快速路径一致，
    // 避免手写逐帧双循环丢掉 ndarray 的优化。
    let mut mag_matrix = Array2::<f32>::zeros((n_freqs, n_frames));
    for (frame, &n_fft) in frame_nfft.iter().enumerate() {
        // 临时取出该档 plan 以获得可变借用（`frame_buf` 是每帧复用的 scratch）；
        // 用完立即放回，保证缓存不丢。
        let mut plan = shift_plans
            .remove(&n_fft)
            .ok_or_else(|| "mel: shift plan missing".to_string())?;

        // 帧 frame 的首个样本在**未 pad** 坐标系中的位置。reflect pad 使
        // 越界样本回绕，与 mel_from_audio_fast 的 reflect_pad_into 一致。
        let start = frame as isize * hop as isize - plan.pad_left as isize;
        let len = audio.len();
        for i in 0..plan.n_fft {
            let j = start + i as isize;
            let sample = if j >= 0 && (j as usize) < len {
                audio[j as usize]
            } else {
                audio[reflect_index(j, len)]
            };
            plan.frame_buf[i] = Complex32::new(sample * plan.window[i], 0.0);
        }
        plan.fft.process(&mut plan.frame_buf);

        // 幅度谱 resize 回固定 `n_freqs`：多截、少补零（OpenUtau 同此）。
        let usable = (plan.n_fft / 2 + 1).min(n_freqs);
        for k in 0..usable {
            let c = plan.frame_buf[k];
            mag_matrix[[k, frame]] = (c.re * c.re + c.im * c.im).sqrt() * plan.bin_scale;
        }
        shift_plans.insert(n_fft, plan);
    }

    // 阶段 3：固定 mel 基投影 + 对数压缩。
    let mel: Vec<f32> = mel_fb_matrix
        .dot(&mag_matrix)
        .into_iter()
        .map(dynamic_range_compression_ln)
        .collect();
    Ok(mel)
}

/// 后台预热结果：None = 尚未尝试/进行中；Some(Ok) = 就绪；Some(Err) = 失败。
/// `is_available()` 的非阻塞依据。
static PREWARM: OnceLock<Mutex<Option<Result<(), String>>>> = OnceLock::new();
static PREWARM_STARTED: AtomicBool = AtomicBool::new(false);
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TLS_SESSION: RefCell<Option<Result<NsfHifiganOnnx, String>>> = RefCell::new(None);
}

/// 幂等地启动一次后台会话预热（**绝不阻塞**调用线程）。
///
/// App启动流程显式调用；`is_available()`只读结果，不启动预热。会话构建可能耗时数秒
/// （DirectML 首次推理 1.5s+，CPU 仅数十毫秒），绝不能发生在 UI 线程 /
/// 前端初始化命令 / 引擎 worker / 快照构建上 —— 实测 GPU 设备下启动会被
/// 阻塞约 4 秒、期间前端无法交互。
pub fn ensure_background_prewarm() {
    if PREWARM_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("ort-prewarm-nsf-hifigan".to_string())
        .spawn(|| {
            let res = get_or_init_shared_session().map(|_| ());
            *PREWARM
                .get_or_init(|| Mutex::new(None))
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(res);
        });
}

/// 已明确判定不可用时的错误信息（"尚未尝试"不算不可用）。
fn confirmed_unavailable() -> Option<String> {
    let guard = PREWARM
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(Err(e)) => Some(e.clone()),
        _ => None,
    }
}

/// **非阻塞**可用性查询：绝不构建会话（构建交给后台预热 / 显式加载路径）。
///
/// 尚未预热完成时返回 true（乐观）—— 调用方在 false 时会**跳过处理器渲染**
///（产出未处理音频）或跳过分离/分析，乐观是安全侧；真实失败会在首次实际
/// 使用（load / 预热完成）时暴露并缓存为 false。
pub fn is_available() -> bool {
    // 可用性只读：快照/参数查询不应偷偷启动ORT线程。App已在登记模型后显式预热，
    // 插件真实worker使用get_or_init路径加载；避免短查询结束时线程撞上进程/DLL退出。
    match confirmed_unavailable() {
        Some(e) => {
            let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
            if debug && !LOGGED_UNAVAILABLE.swap(true, Ordering::Relaxed) {
                log::warn!("nsf_hifigan_onnx: unavailable: {e}");
            }
            false
        }
        None => true,
    }
}

// Helper functions for diagnostics
pub fn compiled() -> bool {
    true
}

pub fn model_load_error() -> Option<String> {
    confirmed_unavailable()
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

/// 单块最大 mel 帧数默认值：**512 帧（≈5.9s @ hop=512, sr=44100）**。
///
/// # 为什么是 512 而不是更大
/// 该值同时影响**冷渲染吞吐**与**编辑后的重渲染量**。实测（60s clip、CoreML、
/// 预热后；配合"曲线按块区间切片"的哈希，见 `compute_param_hash`）：
///
/// | 块大小 | 冷渲染 | 改 1 秒张力后 |
/// |---|---|---|
/// | 4096 帧（47.6s） | 36.9 ms | 996.6 ms |
/// | **512 帧（5.9s）** | **36.3 ms** | **283.8 ms** |
/// | 256 帧（2.97s） | 36.9 ms | 3156.6 ms |
///
/// 512 帧的冷渲染与 4096 帧**持平**（差异在噪声内），而编辑延迟降到约 1/3.5。
/// 再往下（256 帧）反而急剧劣化：块数太多，每块的固定开销与残余块占比上升，
/// 收益被吃掉。故 512 是实测甜点。
///
/// # 输出会变（已按版本失效处理）
/// 块大小决定分块位置，因而改变输出波形。实测差异是**纯相位/时移**性质：
/// 幅度谱余弦相似度 1.000000、逐块 RMS 比 1.002~1.008、最佳时移对齐后残差
/// 降至原 RMS 的 0.25 —— **音色与能量不变**，但 PCM 逐样本不同。
/// `RENDER_PIPELINE_VERSION` 已因此递增（v6），使磁盘缓存整体失效，
/// 避免新旧相位基准混用。
///
/// # 与 HNSEP 无关
/// HNSEP 含双向 LSTM、反向状态依赖整条序列，**保持整段推理、不分块**。
/// 本常量仅作用于 HiFiGAN。
///
/// 可用 `HIFISHIFTER_ONNX_CHUNK_FRAMES` 覆盖。
const CHUNK_MAX_FRAMES_DEFAULT: usize = 512;
/// 神经批量有固定上限；整段mel仍一次计算，不能把它宣称为常量总内存。
const CHUNK_BATCH_MAX: usize = 4;

static CHUNK_MAX_FRAMES_OVERRIDE: OnceLock<usize> = OnceLock::new();

/// 单块最大 mel 帧数，可由 `HIFISHIFTER_ONNX_CHUNK_FRAMES` 覆盖。
///
/// 详见 [`CHUNK_MAX_FRAMES_DEFAULT`] 的取值权衡。
fn chunk_max_frames() -> usize {
    *CHUNK_MAX_FRAMES_OVERRIDE.get_or_init(|| {
        std::env::var("HIFISHIFTER_ONNX_CHUNK_FRAMES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(CHUNK_MAX_FRAMES_DEFAULT)
    })
}

/// 把 mel 帧区间 `[mel_lo, mel_hi)` 换算成**绝对时间窗（秒）**。
///
/// `mel_frame * hop / model_sr` 是模型域的时间轴（见本文件构建 f0 处
/// `start_sec + i * hop_sec`）。调用方再乘自己的输出采样率即可得到帧号，
/// 从而避免把模型域样本数直接加到输出域帧号上（那会在非 44100 输出时错位）。
fn chunk_time_span(start_sec: f64, hop_sec: f64, mel_lo: usize, mel_hi: usize) -> (f64, f64) {
    (
        start_sec + mel_lo as f64 * hop_sec,
        start_sec + mel_hi as f64 * hop_sec,
    )
}

/// 固定数量的块查缓存/推理后立即拼入输出，不同时持有整段所有块的输入与输出副本。
/// 缓存损坏按miss重算；新推理输出必须完整且有限，失败绝不写成功缓存。
fn assemble_bounded_chunks(
    frames:usize,hop:usize,chunk_frames:usize,batch_max:usize,
    mut get:impl FnMut(usize,usize)->Option<Vec<f32>>,
    mut infer:impl FnMut(&[(usize,usize)])->Result<Vec<Vec<f32>>,String>,
    mut put:impl FnMut(usize,usize,Vec<f32>),
) -> Result<Vec<f32>,String> {
    if hop==0||chunk_frames==0||batch_max==0 {return Err("invalid HiFiGAN chunk geometry".into());}
    let samples=frames.checked_mul(hop).ok_or("HiFiGAN output length overflow")?;
    let mut out=vec![0_f32;samples];let mut offset=0;let mut completed=0;
    let total=frames.div_ceil(chunk_frames);
    while offset<frames {
        let mut missing=Vec::with_capacity(batch_max);
        for _ in 0..batch_max {
            if offset>=frames {break;}
            let end=offset.saturating_add(chunk_frames).min(frames);let expected=(end-offset)*hop;
            if let Some(cached)=get(offset,end).filter(|wf|wf.len()==expected&&wf.iter().all(|v|v.is_finite())) {
                out[offset*hop..end*hop].copy_from_slice(&cached);completed+=1;
            } else {missing.push((offset,end));}
            offset=end;
        }
        if !missing.is_empty() {
            let outputs=infer(&missing)?;
            if outputs.len()!=missing.len() {return Err("HiFiGAN batch output count mismatch".into());}
            // 先验证整批再写cache，不发布同一失败批的半成品。
            if outputs.iter().zip(&missing).any(|(wf,(start,end))|wf.len()!=(end-start)*hop||wf.iter().any(|v|!v.is_finite())) {
                return Err("invalid HiFiGAN chunk output".into());
            }
            for ((start,end),wf) in missing.into_iter().zip(outputs) {
                out[start*hop..end*hop].copy_from_slice(&wf);put(start,end,wf);completed+=1;
            }
        }
        crate::renderer::progress::report_clip_progress(completed as f64/total.max(1) as f64);
    }
    Ok(out)
}

// ─── 帧级分块推理优化──────────────────────────────────────

/// 优化版长音频分块推理：预提取全段 mel 一次，按帧切片推理，按帧偏移顺序拼接。
///
/// 与旧版秒级分块实现的区别：
/// - mel 只提取一次，按帧切片（而非每块独立提取）
/// - 使用帧级常量 `chunk_max_frames()` 分块
/// - 支持分块级缓存回调，参数变动时只重渲染脏 chunk
///
/// # ⚠ 拼接方式：直接覆盖，**没有** overlap / crossfade
/// 旧版本注释在此处声称"线性 crossfade"，但代码执行的是
/// `out[base_out + i] = sample` —— 块之间既不重叠也不加权。
/// 块边界因此存在轻微不连续，实测其跳变约为信号自身相邻跳变的 p99 的 0.3 倍
/// **低于**正常信号起伏，听感上不构成 click。
///
/// **不要照搬本函数修 crossfade**：块大小一变，接缝位置就变，输出随之改变
/// （实测不同块大小之间 rel_l2 6~9%）。也就是说"补 crossfade"会改变所有既有
/// 工程的渲染结果，属于需要用户知情的行为变更。
/// 需要 crossfade 的实现可参考 [`infer_pitch_edit_chunked_mel_stretch`]
/// （rate≠1 路径，用 sin/cos 等功率加权）。
///
/// `chunk_cache_get(mel_start, mel_end, chunk_start_sec, chunk_end_sec)` → 命中时返回缓存的 mono PCM，
/// `chunk_cache_put(mel_start, mel_end, chunk_start_sec, chunk_end_sec, waveform)` → 写入波形到缓存。
///
/// # ⚠ 为什么必须把时间窗以**秒**传给回调（曾有真实 bug）
/// mel 帧索引是**模型域**的（`hop`/`model_sr`，本工程为 512/44100），而调用方缓存的
/// 键与曲线切片用的是**输出采样率域**的帧号。调用方若自行用 `mel_start * 512`
/// 换算，就会把模型域的偏移加到输出域的起点上 —— 二者仅在 44100 输出时相等，
/// 在 48000 等常见设备采样率下会错位 `sr/44100`（约 8%），导致"块尾部渲染了、
/// 却不在自己的哈希窗口内"，编辑该处会命中陈旧块。
/// 因此时间窗由本函数（唯一同时知道 `hop`、`model_sr`、`start_sec` 的地方）
/// 算好并以秒给出，调用方只做 `秒 × 输出采样率`，杜绝单位混淆。
/// 帧号相对于 `mono_pcm` 的起始（0-based mel frame index）。
pub fn infer_pitch_edit_chunked_optimized(
    mono_pcm: &[f32],
    sample_rate: u32,
    start_sec: f64,
    midi_at_time: impl Fn(f64) -> f64 + Clone,
    formant_shift_at_time: impl Fn(f64) -> f32 + Clone,
    chunk_cache_get: &dyn Fn(usize, usize, f64, f64) -> Option<Vec<f32>>,
    chunk_cache_put: &dyn Fn(usize, usize, f64, f64, Vec<f32>),
) -> Result<Vec<f32>, String> {
    if mono_pcm.is_empty() {
        return Ok(vec![]);
    }
    // 非阻塞可用性检查：真正的会话构建由下方 TLS_SESSION 加载完成
    //（在该调用线程上按需构建，不在 UI/命令/快照构建路径上同步构建）。
    if !is_available() {
        return Err("nsf_hifigan: model unavailable".to_string());
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
        //
        // 共振峰（gender）在 **mel 提取阶段** 生效（伸缩分析窗），因此必须先算出
        // 帧数 → 采样出逐帧 keyShift → 再提取 mel。帧数只由 `hop` 与音频长度决定
        // （与 keyShift 无关），顺序调整不改变 `t`。
        let mel_full = {
            let mut resample_buf = std::mem::take(&mut sess.audio_resample_buf);
            let model_audio: &[f32] = if sample_rate == model_sr {
                mono_pcm
            } else {
                linear_resample_mono_into(mono_pcm, sample_rate, model_sr, &mut resample_buf);
                &resample_buf
            };

            let frame_count = mel_frame_count(model_audio.len(), hop);
            let hop_sec = (hop as f64) / (model_sr.max(1) as f64);
            let shifts =
                formant_shifts_for_frames(&formant_shift_at_time, start_sec, hop_sec, frame_count);
            let mel = extract_mel_with_shifts(sess, model_audio, &shifts);
            sess.audio_resample_buf = resample_buf;
            mel
        }?;

        let t = mel_full.len() / sess.cfg.num_mels;
        if t == 0 {
            return Ok(vec![0.0; mono_pcm.len()]);
        }

        // 2. 构建 F0
        let hop_sec = (hop as f64) / (model_sr.max(1) as f64);
        let f0_full: Vec<f32> = (0..t)
            .map(|i| {
                let abs_t = start_sec + (i as f64) * hop_sec;
                midi_to_hz(midi_at_time(abs_t))
            })
            .collect();

        // 3. 共振峰偏移已在步骤 1 的 mel 提取阶段生效（见 mel_from_audio_shifted），
        //    此处不再做 mel 域的二次插值。

        // 4. 每批最多四块；保留既有mel网格/尾块长度，不跨不同长度补零改变模型语义。
        // 输出长度 = 重建内容真实长度（t×hop）。mel 提取的尾部窗损失使
        // t×hop < 输入长度；尾部对齐（含斜坡收尾）在步骤 5 统一完成，
        // 若此处直接按输入长度初始化并留零，会预先制造"内容↔零"缺口。
        let out=assemble_bounded_chunks(t,hop,chunk_max_frames(),CHUNK_BATCH_MAX,
            |fi,end| {let (c0,c1)=chunk_time_span(start_sec,hop_sec,fi,end);chunk_cache_get(fi,end,c0,c1)},
            |ranges| {
                let batch_items:Vec<(Vec<f32>,Vec<f32>,usize)>=ranges.iter().map(|&(fi,chunk_end)| {
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
                let began=std::time::Instant::now();let results=sess.run_model_batch(&batch_items)?;
                debug_eprintln!("[nsf_hifigan] bounded_batch={}ms chunks={}",began.elapsed().as_millis(),results.len());
                Ok(results)
            },
            |fi,end,wf| {let (c0,c1)=chunk_time_span(start_sec,hop_sec,fi,end);chunk_cache_put(fi,end,c0,c1,wf)},
        )?;

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

// ─── Mel 共振峰偏移（gender / PitchAdjustableMelSpectrogram）─────────────────

/// mel 矩阵的帧数（与 keyShift 无关）。
///
/// reflect pad 左右共 `win - hop` 个样本，因此 `padded_len = n + win - hop`，
/// 帧数 = `1 + (padded_len - win) / hop = 1 + (n - hop) / hop`。
/// `n < hop` 时窗口连一帧都放不满，返回 1 帧（与 `mel_from_audio_fast` 的
/// 空音频分支一致，保证两者帧数契约相同）。
pub fn mel_frame_count(n: usize, hop: usize) -> usize {
    if hop == 0 {
        return 1;
    }
    if n < hop {
        return 1;
    }
    1 + (n - hop) / hop
}

/// 逐帧采样共振峰偏移曲线（cents），并换算成 keyShift（半音）。
///
/// # 符号（关键，易写反）
/// `keyShift = +cents / 100`。两者**同号**：mel 基按原始 `n_fft` 建立，而
/// `n_fft_new = round(n_fft * 2^(k/12))`，频率 `f` 的内容被基读成
/// `f * 2^(k/12)`，故 `k = +12` → 内容出现在 `2f` → 共振峰**上移**，
/// 与 `formant_shift_cents` 既有语义（正 = 上移）一致。
///
/// 返回值已按 [`quantize_key_shift`] 量化，使下游 FFT plan 数量有界。
pub fn formant_shifts_for_frames(
    formant_shift_at_time: &(impl Fn(f64) -> f32 + ?Sized),
    start_sec: f64,
    hop_sec: f64,
    frames: usize,
) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            let abs_t = start_sec + (i as f64) * hop_sec;
            quantize_key_shift(formant_shift_at_time(abs_t) / 100.0)
        })
        .collect()
}

/// 有 keyShift 时走逐帧分析，否则走既有的定长快速路径。
///
/// 【为什么保留双路径】`mel_from_audio_fast` 是零偏移的唯一基准实现，
/// 全 0 时走它可保证既有工程（未画共振峰曲线）的渲染结果**逐样本不变**；
/// 逐帧路径只在确有偏移时启用。判定阈值与旧实现一致（0.5 cents）。
pub fn extract_mel_with_shifts(
    sess: &mut NsfHifiganOnnx,
    audio: &[f32],
    shifts: &[f32],
) -> Result<Vec<f32>, String> {
    // shifts 已量化：只要有一帧非 0 就必须走逐帧路径。
    let has_shift = shifts.iter().any(|&s| s != 0.0);
    if !has_shift {
        return sess.mel_from_audio_fast(audio);
    }
    sess.mel_from_audio_shifted(audio, shifts)
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
        //
        // 共振峰（gender）在 mel 提取阶段生效（伸缩分析窗），因此先按**源帧**采样
        // keyShift 再提取。曲线按源 PCM 的时间轴查询（本函数的 `start_sec` 即源段
        // 在时间轴上的起点，拉伸由 mel 时间轴插值在其后完成）。
        let hop_sec_src = (self.cfg.hop_size as f64) / (model_sr.max(1) as f64);
        let mel_orig = {
            let mut resample_buf = std::mem::take(&mut self.audio_resample_buf);
            let model_audio: &[f32] = if sample_rate == model_sr {
                audio_mono
            } else {
                linear_resample_mono_into(audio_mono, sample_rate, model_sr, &mut resample_buf);
                &resample_buf
            };
            let frame_count = mel_frame_count(model_audio.len(), self.cfg.hop_size);
            let shifts = formant_shifts_for_frames(
                &formant_shift_at_time,
                start_sec,
                hop_sec_src,
                frame_count,
            );
            let mel = extract_mel_with_shifts(self, model_audio, &shifts);
            self.audio_resample_buf = resample_buf;
            mel?
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
        let mel_stretched = if (playback_rate - 1.0).abs() <= 1e-6 {
            mel_orig
        } else {
            interpolate_mel_time(&mel_orig, self.cfg.num_mels, t_orig, t_new)
        };

        // 4. 共振峰偏移已在步骤 1 的 mel 提取阶段生效（见 mel_from_audio_shifted），
        //    时间轴插值（步骤 3）后无需再做频率轴处理：拉伸与共振峰是正交的两轴。

        // 5. 构建 F0 [T_new]
        // F0 直接按时间轴坐标查询，pitch_edit / clip_midi 已与时间轴对齐
        let hop_sec = hop_sec_src;
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
    // 非阻塞可用性检查：真正的会话构建由下方 TLS_SESSION 加载完成
    //（在该调用线程上按需构建，不在 UI/命令/快照构建路径上同步构建）。
    if !is_available() {
        return Err("nsf_hifigan: model unavailable".to_string());
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
        cfg.sampling_rate,
        cfg.num_mels,
        cfg.hop_size,
        cfg.n_fft
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
        available_providers,
        gpu_device_id,
        gpu_actually_working,
        dml_available
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
    use super::{
        formant_shifts_for_frames, mel_frame_count, quantize_key_shift, shift_n_fft,
        KEY_SHIFT_QUANTUM_SEMITONES,
    };

    /// 只读能力轮询不能隐式启动模型线程；实际合成仍由TLS加载路径负责。
    #[test]
    fn availability_query_does_not_start_model_prewarm() {
        let started = super::PREWARM_STARTED.load(std::sync::atomic::Ordering::Acquire);
        let session_exists = super::SHARED_SESSION.get().is_some();
        for _ in 0..1000 {
            let _ = super::is_available();
        }
        assert_eq!(super::PREWARM_STARTED.load(std::sync::atomic::Ordering::Acquire), started);
        assert_eq!(super::SHARED_SESSION.get().is_some(), session_exists);
    }

    #[test]
    fn model_cache_digest_changes_for_equal_length_middle_bytes_and_configuration() {
        let dir=std::env::temp_dir().join(format!("hfs-model-identity-{}",uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();let model=dir.join("model.onnx");let config=dir.join("config.json");
        let mut bytes=vec![3_u8;1024];std::fs::write(&model,&bytes).unwrap();std::fs::write(&config,b"config-a").unwrap();
        let before=super::digest_model_files(&model,&config).unwrap();bytes[512]=4;std::fs::write(&model,&bytes).unwrap();
        let changed=super::digest_model_files(&model,&config).unwrap();assert_ne!(before,changed);
        std::fs::write(&config,b"config-b").unwrap();assert_ne!(changed,super::digest_model_files(&model,&config).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 字面序列验证长素材批数、混合命中、最后不足整块的样本及cache写回契约。
    #[test]
    fn bounded_chunks_cover_all_samples_and_never_collect_an_unbounded_batch() {
        let calls=std::cell::RefCell::new(Vec::new());let saved=std::cell::RefCell::new(Vec::new());
        let out=super::assemble_bounded_chunks(19,2,3,2,
            |start,end|if start==6 {Some(vec![9.;(end-start)*2])} else {None},
            |ranges| {calls.borrow_mut().push(ranges.to_vec());Ok(ranges.iter().map(|&(start,end)|vec![start as f32;(end-start)*2]).collect())},
            |start,end,wf|saved.borrow_mut().push((start,end,wf.len())),
        ).unwrap();
        assert_eq!(calls.into_inner(),vec![vec![(0,3),(3,6)],vec![(9,12)],vec![(12,15),(15,18)],vec![(18,19)]]);
        assert_eq!(out,vec![0.,0.,0.,0.,0.,0.,3.,3.,3.,3.,3.,3.,9.,9.,9.,9.,9.,9.,9.,9.,9.,9.,9.,9.,12.,12.,12.,12.,12.,12.,15.,15.,15.,15.,15.,15.,18.,18.]);
        assert_eq!(saved.borrow().last(),Some(&(18,19,2)));
    }
    #[test]
    fn bounded_chunks_rebuild_corrupt_cache_and_do_not_store_invalid_model_output() {
        let writes=std::cell::Cell::new(0);
        let err=super::assemble_bounded_chunks(6,2,3,2,|_,_|Some(vec![f32::NAN;6]),
            |_|Ok(vec![vec![0.;6],vec![0.;5]]),|_,_,_|writes.set(writes.get()+1)).unwrap_err();
        assert!(err.contains("invalid HiFiGAN chunk"));assert_eq!(writes.get(),0);
        let out=super::assemble_bounded_chunks(1,2,3,2,|_,_|Some(vec![0.;1]),
            |_|Ok(vec![vec![0.25,0.5]]),|_,_,_|writes.set(writes.get()+1)).unwrap();
        assert_eq!(out,vec![0.25,0.5]);assert_eq!(writes.get(),1);
    }
    /// 真模型超过旧30秒限制，尾块/48k秒域/暖命中/接缝均验证，不将HNSEP切块。
    #[test]
    #[ignore = "真实CPU HiFiGAN长素材诊断：显式运行，不自动重复"]
    fn real_model_long_chunks_and_warm_cache_cover_the_complete_clip() {
        crate::vocoder_ort_session::set_runtime_ep_override(Some("cpu".into()));
        let rate=48000u32;let seconds=35;let samples=seconds*rate as usize;
        let input:Vec<f32>=(0..samples).map(|i| {
            let phase=2.*std::f64::consts::PI*220.*i as f64/rate as f64;
            (1..=12).map(|k|0.15/k as f64*(phase*k as f64).sin()).sum::<f64>() as f32
        }).collect();
        let cache=std::cell::RefCell::new(std::collections::BTreeMap::new());let writes=std::cell::RefCell::new(Vec::new());
        let get=|start:usize,end:usize,_:f64,_:f64|cache.borrow().get(&(start,end)).cloned();
        let put=|start,end,c0,c1,wf:Vec<f32>| {writes.borrow_mut().push((start,end,c0,c1));cache.borrow_mut().insert((start,end),wf);};
        let began=std::time::Instant::now();
        let cold=super::infer_pitch_edit_chunked_optimized(&input,rate,3.,|_|60.,|_|0.,&get,&put).unwrap();
        let cold_ms=began.elapsed().as_millis();let chunks=writes.borrow().clone();let began=std::time::Instant::now();
        let warm=super::infer_pitch_edit_chunked_optimized(&input,rate,3.,|_|60.,|_|0.,&get,&put).unwrap();
        let warm_ms=began.elapsed().as_millis();assert_eq!(cold,warm);assert_eq!(writes.borrow().len(),chunks.len());
        assert_eq!(cold.len(),samples);assert!(cold.iter().all(|v|v.is_finite()));assert!(chunks.len()>super::CHUNK_BATCH_MAX);
        for (index,(start,end,c0,c1)) in chunks.iter().copied().enumerate() {
            assert!(end-start<=super::chunk_max_frames());if index>0 {assert_eq!(start,chunks[index-1].1);}
            assert!((c0-(3.+start as f64*512./44100.)).abs()<1e-10);assert!((c1-(3.+end as f64*512./44100.)).abs()<1e-10);
        }
        assert!(chunks.last().unwrap().1-chunks.last().unwrap().0<super::chunk_max_frames());
        for second in 0..seconds {
            let lo=second*rate as usize;let rms=(cold[lo..lo+rate as usize].iter().map(|&v|(v as f64).powi(2)).sum::<f64>()/rate as f64).sqrt();
            assert!(rms>1e-4,"第{second}秒整段静音: {rms}");
        }
        let mut jumps=cold.windows(2).map(|v|(v[1]-v[0]).abs()).collect::<Vec<_>>();jumps.sort_by(f32::total_cmp);
        let p99=jumps[jumps.len()*99/100];let mut boundary_max=0_f32;
        for (_,_,_,c1) in chunks.iter().take(chunks.len()-1) {
            let at=((c1-3.)*rate as f64).round() as usize;
            boundary_max=boundary_max.max((cold[at]-cold[at-1]).abs());
        }
        assert!(boundary_max<p99*8.+1e-4,"异常接缝: max={boundary_max}, p99={p99}");
        println!("HIFIGAN_LONG rate={rate} seconds={seconds} chunks={} batch_limit={} tail_frames={} cold_ms={cold_ms} warm_ms={warm_ms} warm_extra_runs=0 boundary_max={boundary_max:e} sample_jump_p99={p99:e}",
            chunks.len(),super::CHUNK_BATCH_MAX,chunks.last().unwrap().1-chunks.last().unwrap().0);
    }

    // ─── gender / 共振峰偏移（PitchAdjustableMelSpectrogram 移植）──────────────

    /// 测试用配置：与 resources/models/nsf_hifigan/config.json 一致。
    fn test_cfg() -> super::NsfHifiganConfig {
        super::NsfHifiganConfig {
            sampling_rate: 44_100,
            num_mels: 128,
            hop_size: 512,
            n_fft: 2048,
            win_size: 2048,
            fmin: 40.0,
            fmax: 16_000.0,
        }
    }

    fn test_basis(cfg: &super::NsfHifiganConfig) -> ndarray::Array2<f32> {
        super::mel_filterbank_slaney(
            cfg.sampling_rate,
            cfg.n_fft,
            cfg.num_mels,
            cfg.fmin,
            cfg.fmax,
        )
    }

    fn sine(n: usize, hz: f64, sr: f64) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / sr).sin() as f32)
            .collect()
    }

    /// mel 频谱（[n_mels, frames] 行优先）在给定帧上能量最强的 bin 序号。
    fn peak_bin(mel: &[f32], n_mels: usize, frame: usize) -> usize {
        let frames = mel.len() / n_mels;
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for m in 0..n_mels {
            let v = mel[m * frames + frame];
            if v > best_v {
                best_v = v;
                best = m;
            }
        }
        best
    }

    /// **核心行为**：正 keyShift 必须把频谱峰值搬到更高的 mel bin，负值搬到更低的。
    ///
    /// 这是整个 gender 迁移的成败判据 —— `formant_shift_sign_is_positive_up` 只验证
    /// 「cents → keyShift」的换算，本测试才验证「keyShift → 频谱真的移动了」。
    #[test]
    fn shifted_mel_moves_spectral_peak() {
        let cfg = test_cfg();
        let basis = test_basis(&cfg);
        let n = 44_100; // 1 秒
        let audio = sine(n, 1000.0, 44_100.0);
        let frames = super::mel_frame_count(n, cfg.hop_size);
        let mid = frames / 2;

        let run = |key_shift: f32| -> usize {
            let mut plans = std::collections::HashMap::new();
            let shifts = vec![super::quantize_key_shift(key_shift); frames];
            let mel =
                super::compute_shifted_mel(&audio, &shifts, &cfg, &basis, &mut plans).unwrap();
            peak_bin(&mel, cfg.num_mels, mid)
        };

        let base = run(0.0);
        let up = run(12.0); // +1 八度
        let down = run(-12.0);

        assert!(
            up > base,
            "+12 semitones must move the peak UP: base bin {base}, up bin {up}"
        );
        assert!(
            down < base,
            "-12 semitones must move the peak DOWN: base bin {base}, down bin {down}"
        );
        // 只做量级下界断言，避免把测试绑死在具体 mel 刻度实现上。
        assert!(
            up - base >= 8,
            "octave up should move at least 8 mel bins, moved {}",
            up - base
        );
        assert!(
            base - down >= 3,
            "octave down should move at least 3 mel bins, moved {}",
            base - down
        );
    }

    /// keyShift = 0 时逐帧路径必须与既有快速路径**逐样本一致**。
    ///
    /// 【为什么必须钉住】既有工程（未画共振峰曲线）走 `mel_from_audio_fast`；
    /// 若零偏移下两条路径不等，升级会让所有旧工程的渲染结果发生无谓变化，
    /// 且与磁盘缓存里的旧结果对不上。
    #[test]
    fn zero_shift_matches_fast_path() {
        let cfg = test_cfg();
        let basis = test_basis(&cfg);
        let n = 8192;
        let audio = sine(n, 440.0, 44_100.0);
        let frames = super::mel_frame_count(n, cfg.hop_size);

        let mut plans = std::collections::HashMap::new();
        let shifts = vec![0.0f32; frames];
        let shifted =
            super::compute_shifted_mel(&audio, &shifts, &cfg, &basis, &mut plans).unwrap();

        // 手写快速路径的等效实现（不依赖 ORT session）。
        let pad_left = ((cfg.win_size as isize - cfg.hop_size as isize) / 2).max(0) as usize;
        let pad_right =
            ((cfg.win_size as isize - cfg.hop_size as isize + 1) / 2).max(0) as usize;
        let mut padded = Vec::with_capacity(pad_left + n + pad_right);
        for i in -(pad_left as isize)..0 {
            padded.push(audio[super::reflect_index(i, n)]);
        }
        padded.extend_from_slice(&audio);
        for i in n..(n + pad_right) {
            padded.push(audio[super::reflect_index(i as isize, n)]);
        }

        let window = super::hann_window(cfg.win_size);
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        let n_freqs = cfg.n_fft / 2 + 1;
        let mut expected = vec![0.0f32; cfg.num_mels * frames];
        for frame in 0..frames {
            let mut buf = vec![num_complex::Complex32::new(0.0, 0.0); cfg.n_fft];
            for i in 0..cfg.win_size {
                buf[i] =
                    num_complex::Complex32::new(padded[frame * cfg.hop_size + i] * window[i], 0.0);
            }
            fft.process(&mut buf);
            let mut col = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let c = buf[k];
                col[k] = (c.re * c.re + c.im * c.im).sqrt();
            }
            for m in 0..cfg.num_mels {
                let row = &basis.row(m);
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += row[k] * col[k];
                }
                expected[m * frames + frame] = super::dynamic_range_compression_ln(sum);
            }
        }

        assert_eq!(shifted.len(), expected.len());
        let mut max_err = 0.0f32;
        for (a, b) in shifted.iter().zip(expected.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 1e-3,
            "zero-shift path must match the fast path, max err {max_err}"
        );
    }

    /// 端到端：非零 keyShift 时 mel 帧数必须与快速路径一致。
    ///
    /// `mel_from_audio_shifted` 依赖"帧数与 keyShift 无关"这一契约；若它不成立，
    /// mel 与 f0 的时间轴会错位（输出整体变速/静音）。这里用真实配置对比两条路径。
    #[test]
    fn shifted_and_fast_paths_agree_on_frame_count() {
        let cfg = test_cfg();
        let basis = test_basis(&cfg);
        for &n in &[0usize, 100, 512, 513, 4096, 44_100, 48_000] {
            let audio = sine(n.max(1), 440.0, 44_100.0);
            let want = super::mel_frame_count(audio.len(), cfg.hop_size);

            for key_shift in [-12.0f32, -3.0, 0.0, 3.0, 12.0] {
                let shifts = vec![super::quantize_key_shift(key_shift); want];
                let mut plans = std::collections::HashMap::new();
                let mel =
                    super::compute_shifted_mel(&audio, &shifts, &cfg, &basis, &mut plans).unwrap();
                assert_eq!(
                    mel.len(),
                    cfg.num_mels * want,
                    "frames drifted at n={n}, keyShift={key_shift}"
                );
            }
        }
    }

    /// 窗口长度必须等于 FFT 长度 —— 这是帧数与 keyShift 无关的前提。
    #[test]
    fn shift_plan_window_equals_fft_length() {
        for step in -48i32..=48 {
            let k = step as f32 * KEY_SHIFT_QUANTUM_SEMITONES;
            let n_fft = shift_n_fft(k, 2048, 2048);
            let plan = super::ShiftPlan::new(k, n_fft, 2048, 512).unwrap();
            assert_eq!(
                plan.window.len(),
                plan.n_fft,
                "window len must equal n_fft at keyShift {k}"
            );
            assert_eq!(plan.frame_buf.len(), plan.n_fft);
        }
    }

    /// shifts 长度与帧数不符时必须报错，而不是静默产出错位的 mel。
    #[test]
    fn shifted_mel_rejects_wrong_shift_count() {
        let cfg = test_cfg();
        let basis = test_basis(&cfg);
        let audio = sine(8192, 440.0, 44_100.0);
        let mut plans = std::collections::HashMap::new();
        let wrong = vec![0.0f32; 3];
        assert!(super::compute_shifted_mel(&audio, &wrong, &cfg, &basis, &mut plans).is_err());
    }

    /// keyShift 量化必须落在有限档位上，且档位数有界。
    ///
    /// 【为什么重要】连续曲线会让每帧产出不同的 `n_fft`，进而逐帧重建 FFT plan。
    /// 量化把 plan 数收敛为档数；档数爆炸等于量化失效。
    #[test]
    fn key_shift_quantization_buckets_are_bounded() {
        // ±12 半音、步长 1/4 → 97 档
        let expected = (24.0 / KEY_SHIFT_QUANTUM_SEMITONES) as usize + 1;
        assert_eq!(expected, 97);

        let mut nffts = std::collections::HashSet::new();
        // 扫过整个值域，步长远小于量化步长，模拟连续曲线
        let mut x = -12.0f32;
        while x <= 12.0 + 1e-6 {
            let q = quantize_key_shift(x);
            assert!(
                (q / KEY_SHIFT_QUANTUM_SEMITONES).fract().abs() < 1e-4,
                "quantized value {q} is off-grid"
            );
            nffts.insert(shift_n_fft(q, 2048, 2048));
            x += 0.001;
        }
        assert!(
            nffts.len() <= 97,
            "distinct n_fft values {} exceeds bucket count",
            nffts.len()
        );
    }

    /// 非有限输入必须收敛到 0，不得产生 NaN 档位。
    #[test]
    fn key_shift_quantization_handles_non_finite() {
        assert_eq!(quantize_key_shift(f32::NAN), 0.0);
        assert_eq!(quantize_key_shift(f32::INFINITY), 0.0);
        assert_eq!(quantize_key_shift(f32::NEG_INFINITY), 0.0);
    }

    /// `shift_n_fft` 恒 >= 4（rustfft 可用性下限），且随 keyShift 单调。
    #[test]
    fn shift_n_fft_is_safe_and_monotonic() {
        assert_eq!(shift_n_fft(0.0, 2048, 2048), 2048);
        assert!(shift_n_fft(-12.0, 2048, 2048) >= 4);
        assert!(shift_n_fft(12.0, 2048, 2048) >= 4);
        // 极端值不得 panic / 归零
        assert!(shift_n_fft(-999.0, 2048, 2048) >= 4);
        assert!(shift_n_fft(999.0, 2048, 2048) >= 4);

        let mut prev = 0usize;
        for step in -48i32..=48 {
            let k = step as f32 * KEY_SHIFT_QUANTUM_SEMITONES;
            let n = shift_n_fft(k, 2048, 2048);
            assert!(n >= prev, "not monotonic at keyShift {k}: {n} < {prev}");
            prev = n;
        }
    }

    /// 帧数必须与 keyShift 无关。
    ///
    /// 【为什么必须钉住】`ShiftPlan` 要求窗长 == FFT 长度；只有满足该等式，
    /// `frames = 1 + (n - hop) / hop` 才与 `factor` 无关。若帧数随曲线变化，
    /// mel 与 f0 的时间轴会错位，输出整体变速。
    #[test]
    fn mel_frame_count_is_shift_invariant() {
        for &n in &[0usize, 100, 512, 513, 4096, 44_100] {
            let base = mel_frame_count(n, 512);
            for step in -48i32..=48 {
                let k = step as f32 * KEY_SHIFT_QUANTUM_SEMITONES;
                let nfft = shift_n_fft(k, 2048, 2048);
                // 窗长 == FFT 长度时，padded = n + nfft - hop，
                // frames = 1 + (padded - nfft) / hop = 1 + (n - hop) / hop
                let frames = if n < 512 { 1 } else { 1 + (n - 512) / 512 };
                assert_eq!(
                    frames, base,
                    "frame count changed for n={n}, keyShift={k}, nfft={nfft}"
                );
            }
        }
    }

    /// 符号契约：**正 cents → 正 keyShift → 共振峰上移**。
    ///
    /// 【为什么必须钉住】这个符号极易写反（OpenUtau 的 `gender` 与
    /// `formant_shift_cents` 符号**相反**）。mel 基按原始 n_fft 建立，而
    /// `nfft_new = round(n_fft * 2^(k/12))`，频率 f 的内容被基读成
    /// `f * 2^(k/12)` ⇒ k>0 时内容出现在更高的频率 ⇒ 上移。
    #[test]
    fn formant_shift_sign_is_positive_up() {
        let curve = |_: f64| -> f32 { 1200.0 }; // +1200 cents
        let shifts = formant_shifts_for_frames(&curve, 0.0, 0.01, 4);
        assert_eq!(shifts.len(), 4);
        for s in shifts {
            assert!(
                (s - 12.0).abs() < 1e-4,
                "+1200 cents must map to keyShift +12 (formant UP), got {s}"
            );
            assert!(shift_n_fft(s, 2048, 2048) > 2048, "n_fft must grow for +12");
        }

        let curve_down = |_: f64| -> f32 { -1200.0 };
        let shifts = formant_shifts_for_frames(&curve_down, 0.0, 0.01, 4);
        for s in shifts {
            assert!(
                (s + 12.0).abs() < 1e-4,
                "-1200 cents must map to keyShift -12 (formant DOWN), got {s}"
            );
            assert!(shift_n_fft(s, 2048, 2048) < 2048, "n_fft must shrink for -12");
        }
    }

    /// 曲线按绝对时间采样，且量化生效。
    #[test]
    fn formant_shifts_sample_by_absolute_time() {
        // 前半 +600 cents、后半 -600 cents（按绝对时间 0.05s 分界）
        let curve = |t: f64| -> f32 {
            if t < 0.05 {
                600.0
            } else {
                -600.0
            }
        };
        // start_sec = 0, hop = 0.01s → 帧 0..4 在分界前，帧 5..9 在分界后
        let shifts = formant_shifts_for_frames(&curve, 0.0, 0.01, 10);
        for (i, &s) in shifts.iter().enumerate() {
            let want = if i < 5 { 6.0 } else { -6.0 };
            assert!(
                (s - want).abs() < 1e-4,
                "frame {i}: expected keyShift {want}, got {s}"
            );
        }

        // 起点偏移后采样窗口随之平移
        let shifted = formant_shifts_for_frames(&curve, 0.05, 0.01, 4);
        for s in shifted {
            assert!(
                (s + 6.0).abs() < 1e-4,
                "expected -6 after start offset, got {s}"
            );
        }
    }

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

    // ─── 分块参数（与编辑延迟契约）────────────────────────────────────────

    /// 默认块大小必须保持在实测甜点区间内。
    ///
    /// 【为什么钉住】该值同时决定**冷渲染吞吐**与**改一个音后的重渲染量**，
    /// 且输出波形会随它改变（不同块大小差异为纯相位，`RENDER_PIPELINE_VERSION`
    /// 需同步递增）。实测（60s clip、CoreML、预热后、曲线按块切片）：
    ///
    /// | 块大小 | 冷渲染 | 改 1 秒后 |
    /// |---|---|---|
    /// | 4096 帧 | 36.9 ms | 996.6 ms |
    /// | 512 帧  | 36.3 ms | 283.8 ms |
    /// | 256 帧  | 36.9 ms | 3156.6 ms |
    ///
    /// 512 帧冷渲染与 4096 帧持平、编辑延迟约 1/3.5；256 帧反而急剧劣化
    /// （块数过多，固定开销与残余块占比上升）。故默认应落在 [256, 1024]，
    /// 且不得回到 4096。
    #[test]
    fn default_chunk_frames_stay_in_the_measured_sweet_spot() {
        let d = super::CHUNK_MAX_FRAMES_DEFAULT;
        assert!(
            (256..=1024).contains(&d),
            "default chunk {d} frames is outside the measured sweet spot [256, 1024]"
        );
        // 47.6s/块（4096 帧）会让 60s clip 只有 2 块，改一小段近乎整段重推理。
        assert!(
            d < 4096,
            "must not regress to the 4096-frame (47.6s) chunking that caused long edit waits"
        );
        // 每块时长（hop=512, sr=44100）
        let secs = d as f64 * 512.0 / 44_100.0;
        assert!(
            (2.0..=12.0).contains(&secs),
            "chunk duration {secs:.2}s should stay in a sane range"
        );
    }

    // ─── 分块时间窗的采样率域（曾因混淆而真实出错）──────────────────────

    /// 块的时间窗必须经**秒**换算到输出采样率，而不是把模型域样本数直接
    /// 当成输出域帧号。
    ///
    /// 【曾经的真实 bug】调用方用 `seg_start + mel_start * 512` 计算哈希窗口，
    /// 其中 `seg_start` 是**输出采样率**的帧号，而 `mel_start * 512` 是
    /// **模型采样率**（44100）的样本数。二者只在输出也是 44100 时相等；
    /// 在 48000（Windows WASAPI 共享模式等常见默认）下窗口会错位约 8%，
    /// 使每个块的尾部落在自己的哈希窗口之外 —— 编辑该处不会让渲染它的块失效，
    /// 于是命中陈旧音频，正是"改一小段却听到旧声音"的成因。
    #[test]
    fn chunk_time_span_uses_model_domain_then_converts_through_seconds() {
        let model_sr = 44_100.0f64;
        let hop = 512usize;
        let hop_sec = hop as f64 / model_sr;

        // mel 帧 [512, 1024) 的绝对时间窗
        let (lo, hi) = super::chunk_time_span(0.0, hop_sec, 512, 1024);
        assert!((lo - 512.0 * hop_sec).abs() < 1e-12, "lo={lo}");
        assert!((hi - 1024.0 * hop_sec).abs() < 1e-12, "hi={hi}");

        // 48000 输出下，正确帧号 = 秒 × 输出采样率
        let out_sr = 48_000.0f64;
        let correct = (lo * out_sr) as u64;
        assert_eq!(correct, (512.0 * hop_sec * out_sr) as u64);

        // 错误做法（模型域样本数直接当输出域帧号）会得到显著不同的值
        let wrong = (512 * hop) as u64;
        assert_ne!(
            correct, wrong,
            "output-rate frame number must differ from the model-domain sample count"
        );
        // 偏差约 8%（48000/44100 - 1），即约一个块的尾部长度量级
        let rel = (correct as f64 - wrong as f64).abs() / correct as f64;
        assert!(rel > 0.05, "expected a large (>5%) discrepancy, got {rel}");

        // start_sec 不为 0（片段不在时间轴原点）时同样成立
        let (lo2, _) = super::chunk_time_span(3.0, hop_sec, 512, 1024);
        assert!((lo2 - (3.0 + 512.0 * hop_sec)).abs() < 1e-12);
    }

}
