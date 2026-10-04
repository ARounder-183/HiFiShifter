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

    // 开发树兜底：模型仍放在 app crate 的 `resources/` 下（那才是打包时带的东西）。
    // 见 `nsf_hifigan_onnx.rs` 里同一处的说明。
    let app_bundled = manifest
        .parent()
        .map(|p| p.join("src-tauri").join("resources").join("models").join("fcpe").join("fcpe.onnx"));
    if let Some(app_bundled) = app_bundled {
        if app_bundled.is_file() {
            return Some(app_bundled);
        }
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
    // 快路径：已有会话直接克隆返回。
    if let Some(session) = mutex
        .lock()
        .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?
        .clone()
    {
        return Ok(session);
    }
    // 慢路径：构建在全局构建锁内进行（与 Vocoder / HNSEP 串行化，避免并发
    // D3D12 初始化在驱动层死锁），且**不持有容器锁** —— 构建是秒级操作，
    // 持容器锁构建会让设备切换与关机清理全部卡死。
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
    // 构建期间 EP 设置再次变化（用户连续切换设备）→ 丢弃陈旧结果并重建。
    for _ in 0..3 {
        let generation_before = crate::vocoder_ort_session::ep_settings_generation();
        ensure_ort_init()?;
        let onnx_path = resolve_model_path()?;
        let session = build_session_with_ep(&onnx_path)?;
        if crate::vocoder_ort_session::ep_settings_generation() != generation_before {
            log::warn!(
                "[fcpe] inference device changed during session build — rebuilding with the new EP"
            );
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
        let arc = Arc::new(Mutex::new(session));
        *guard = Some(arc.clone());
        return Ok(arc);
    }
    Err("inference device kept changing during session build".to_string())
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
        log::error!(
            "[fcpe] WARNING: could not acquire SHARED_SESSION lock at shutdown — giving up"
        );
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
    // 异步预热：立即在后台用新 EP 重建会话，把秒级构建移出 pitch 分析热路径。
    std::thread::Builder::new()
        .name("ort-session-prewarm-fcpe".into())
        .spawn(|| {
            if let Err(e) = get_or_init_shared_session() {
                log::warn!("[fcpe] session pre-warm failed: {e}");
            }
        })
        .ok();
}

/// 后台预热结果：None = 尚未尝试/进行中；Some(Ok) = 就绪；Some(Err) = 失败。
/// `is_available()` 的非阻塞依据。
static PREWARM: OnceLock<Mutex<Option<Result<(), String>>>> = OnceLock::new();
static PREWARM_STARTED: AtomicBool = AtomicBool::new(false);

/// 幂等地启动一次后台会话预热（**绝不阻塞**调用线程）。
///
/// `is_available()` 与启动流程用它预热：会话构建 + 烟测可能耗时数秒
/// （GPU EP 的首次推理尤其慢），绝不能发生在 UI 线程 / 前端初始化命令 /
/// 引擎 worker / 快照构建上 —— 那会阻塞前端初始化与全部 IPC。
pub fn ensure_background_prewarm() {
    if PREWARM_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("ort-prewarm-fcpe".to_string())
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
/// 尚未预热完成时返回 true（乐观）—— 调用方在 false 时会跳过处理器/分离/
/// 分析路径，乐观是安全侧；真实失败会在首次实际使用或预热完成时暴露并缓存。
pub fn is_available() -> bool {
    ensure_background_prewarm();
    match confirmed_unavailable() {
        Some(e) => {
            if debug_enabled() && !LOGGED_UNAVAILABLE.swap(true, Ordering::Relaxed) {
                log::warn!("fcpe_onnx: unavailable: {e}");
            }
            false
        }
        None => true,
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

/// 每次分析都要用到的、只由进程级常量决定的 STFT 前置产物。
///
/// `n_fft` / `win_size` 来自 `CACHED_FCPE_*`（env 读一次即固定），因此窗口系数与
/// FFT plan 在整个进程内只需构建一次。原实现每次推理都 `FftPlanner::new()` +
/// `hann_window()` 重建，分块后每 30 s 调一次，属于纯浪费。
struct MelStftPlan {
    n_fft: usize,
    win_size: usize,
    window: Arc<Vec<f32>>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

static MEL_STFT_PLAN: OnceLock<Mutex<Option<Arc<MelStftPlan>>>> = OnceLock::new();

fn mel_stft_plan_cached(n_fft: usize, win_size: usize) -> Arc<MelStftPlan> {
    let slot = MEL_STFT_PLAN.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(plan) = guard.as_ref() {
        if plan.n_fft == n_fft && plan.win_size == win_size {
            // 取 Arc 后即释放锁：整段 mel 计算不该被这把锁串行化。
            return Arc::clone(plan);
        }
    }
    let mut planner = FftPlanner::<f32>::new();
    let plan = Arc::new(MelStftPlan {
        n_fft,
        win_size,
        window: Arc::new(crate::mel_utils::hann_window(win_size)),
        fft: planner.plan_fft_forward(n_fft),
    });
    *guard = Some(Arc::clone(&plan));
    plan
}

/// mel 滤波bank，展平为行主序 `[n_mels × n_freqs]`。
///
/// 同样只由进程级常量决定，构建一次即可。展平（而非保留 `Array2`）是为了让
/// "某一 mel 带与功率谱的点积"退化成一次连续切片的 zip 求和 —— `Array2` 的
/// `[[m, f]]` 逐元素索引在 128 × 513 的双重循环里无法向量化。
struct MelFilterbank {
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    flat: Arc<Vec<f32>>,
}

static MEL_FILTERBANK: OnceLock<Mutex<Option<MelFilterbank>>> = OnceLock::new();

fn mel_filterbank_cached(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Arc<Vec<f32>> {
    let slot = MEL_FILTERBANK.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(cached) = guard.as_ref() {
        if cached.sr == sr
            && cached.n_fft == n_fft
            && cached.n_mels == n_mels
            && cached.fmin == fmin
            && cached.fmax == fmax
        {
            return Arc::clone(&cached.flat);
        }
    }
    let n_freqs = n_fft / 2 + 1;
    let bank = crate::mel_utils::mel_filterbank_slaney(sr, n_fft, n_mels, fmin, fmax);
    let mut flat = Vec::with_capacity(n_mels * n_freqs);
    for m in 0..n_mels {
        for f in 0..n_freqs {
            flat.push(bank[[m, f]]);
        }
    }
    let flat = Arc::new(flat);
    *guard = Some(MelFilterbank {
        sr,
        n_fft,
        n_mels,
        fmin,
        fmax,
        flat: Arc::clone(&flat),
    });
    flat
}

/// 计算 log-mel 频谱，返回 `(mel, n_frames)`。
///
/// `time_major` 决定输出布局：`true` → `[n_frames × n_mels]`（一帧的 mel 连续），
/// `false` → `[n_mels × n_frames]`。由调用方按模型期望的输入轴直接索取，因此
/// `[B,T,M]` 的模型不再需要一次整块转置拷贝。
///
/// ## 为什么按帧融合，而不是先算整张频谱再乘 bank
///
/// 原实现先把功率谱写进一张 `n_freqs × n_frames`（= 513 × T）矩阵，第二遍才乘
/// 滤波bank。那张矩阵是分析期单项最大的分配：16 kHz / hop 160 下 1 小时素材是
/// 36 万帧 → 513 × 360k × 4B ≈ **739 MB**；而它每一列（一帧）算完即被消费、
/// 从不回看。融合后只需 `n_freqs` 长度的功率 scratch，内存与素材长度彻底解耦，
/// 顺带消除了"逐帧跨 513 行跳写"的糟糕缓存局部性。
fn build_mel_from_waveform(
    waveform: &[f32],
    in_sr: u32,
    n_mels: usize,
    time_major: bool,
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

    let plan = mel_stft_plan_cached(n_fft, win_size);
    let fb = mel_filterbank_cached(target_sr, n_fft, n_mels, fmin, fmax);

    let mut fft_buf = vec![Complex32::new(0.0, 0.0); n_fft];
    // 单帧功率谱 scratch：整段分析里唯一与帧数无关的工作缓冲。
    let mut power = vec![0.0f32; n_freqs];
    let mut frame_mel = vec![0.0f32; n_mels];
    let mut mel = vec![0.0f32; n_mels * n_frames];

    for frame in 0..n_frames {
        let start = frame * hop;
        for i in 0..win_size {
            fft_buf[i] = Complex32::new(y[start + i] * plan.window[i], 0.0);
        }
        for c in &mut fft_buf[win_size..] {
            *c = Complex32::new(0.0, 0.0);
        }
        plan.fft.process(&mut fft_buf);

        for (f, p) in power.iter_mut().enumerate() {
            let c = fft_buf[f];
            *p = (c.re * c.re + c.im * c.im).sqrt();
        }

        for (m, out) in frame_mel.iter_mut().enumerate() {
            let band = &fb[m * n_freqs..(m + 1) * n_freqs];
            let mut acc = 0.0f32;
            for (b, p) in band.iter().zip(power.iter()) {
                acc += b * p;
            }
            *out = (acc.max(1e-9)).ln();
        }

        if time_major {
            mel[frame * n_mels..(frame + 1) * n_mels].copy_from_slice(&frame_mel);
        } else {
            for (m, v) in frame_mel.iter().enumerate() {
                mel[m * n_frames + frame] = *v;
            }
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

            // 直接按模型期望的轴序产出 mel，省掉一次整块转置拷贝。
            let time_major = mel_axis == 2;
            let (mel, t) = build_mel_from_waveform(waveform, sample_rate, n_mels, time_major)
                .map_err(|e| format!("{e}; {io_summary}"))?;

            let mel_tensor = if time_major {
                // Model expects [B, T, M] where M=128.
                Tensor::from_array(([1usize, t, n_mels], mel.into_boxed_slice()))
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

            // 同上：按模型期望的轴序直接产出，免去转置拷贝。
            let time_major = mel_axis == 2;
            let (mel, t) = build_mel_from_waveform(waveform, sample_rate, n_mels, time_major)
                .map_err(|e| format!("{e}; {io_summary}"))?;

            let mel_tensor = if time_major {
                Tensor::from_array(([1usize, t, n_mels], mel.into_boxed_slice()))
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

/// F0 推理（单声道 f32 输入）。
///
/// 分析流水线本身就以 f32 持有单声道素材（FCPE 的通道输入也是 f32），因此这是
/// 首选入口 —— 走它就不会为了调用而把整段素材转成 f64 再转回来。
pub fn infer_f0_hz_f32(
    mono: &[f32],
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

    let shared = get_or_init_shared_session()?;
    let mut session = shared
        .lock()
        .map_err(|e| format!("FCPE session lock poisoned: {e}"))?;

    let output_values = run_with_named_inputs(&mut session, mono, sample_rate, f0_floor, f0_ceil)?;

    if output_values.is_empty() {
        return Ok(vec![0.0; target_frames]);
    }

    let resized = resample_f0_linear(&output_values, target_frames);
    Ok(sanitize_f0(resized, f0_floor, f0_ceil))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 融合改造前 `build_mel_from_waveform` 的逐字复制：先把整张功率谱写进
    /// `n_freqs × n_frames` 矩阵，第二遍才逐 (mel 带, 帧) 乘滤波bank。
    ///
    /// 保留它作为行为基准 —— mel 计算的任何改动都必须与此逐位一致，否则
    /// 音高曲线会整体偏移，而这类偏差极难从听感或 UI 上发现。
    fn reference_mel_two_pass(waveform: &[f32], in_sr: u32, n_mels: usize) -> (Vec<f32>, usize) {
        let target_sr = env_fcpe_sr();
        let hop = env_fcpe_hop();
        let n_fft = env_fcpe_n_fft();
        let win_size = env_fcpe_win();
        let fmin = env_fcpe_fmin();
        let fmax = env_fcpe_fmax(target_sr);

        let y = crate::mel_utils::linear_resample_mono(waveform, in_sr, target_sr);
        let pad_left = ((win_size as isize - hop as isize) / 2).max(0) as usize;
        let pad_right = ((win_size as isize - hop as isize + 1) / 2).max(0) as usize;
        let y = crate::mel_utils::reflect_pad(&y, pad_left, pad_right);
        if y.len() < win_size {
            return (vec![(1e-9f32).ln(); n_mels], 1);
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
        (mel, n_frames)
    }

    fn test_waveform(sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                // 两个非谐波分量 + 慢速幅度调制：既覆盖安静的窄带，也覆盖
                // 会让滤波bank 多带同时非零的富谐波情形。
                (0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1310.0 * t).sin())
                    * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin())
            })
            .collect()
    }

    #[test]
    fn fused_mel_is_bit_identical_to_two_pass() {
        let sr = 44_100u32;
        let waveform = test_waveform(sr, 0.5);
        let n_mels = 128usize;

        let (reference, ref_frames) = reference_mel_two_pass(&waveform, sr, n_mels);
        let (freq_major, fm_frames) =
            build_mel_from_waveform(&waveform, sr, n_mels, false).unwrap();

        assert_eq!(ref_frames, fm_frames);
        assert_eq!(reference.len(), freq_major.len());
        // 融合版对每个 (mel 带, 帧) 仍按 f 升序累加，与两遍版求和顺序一致，
        // 因此这里要求逐位相同，而不是"近似相等"。
        assert_eq!(
            reference, freq_major,
            "fused mel diverged from the two-pass reference"
        );
        assert!(reference.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn time_major_layout_is_the_same_values_transposed() {
        let sr = 44_100u32;
        let waveform = test_waveform(sr, 0.5);
        let n_mels = 128usize;

        let (reference, frames) = reference_mel_two_pass(&waveform, sr, n_mels);
        let (time_major, tm_frames) = build_mel_from_waveform(&waveform, sr, n_mels, true).unwrap();

        assert_eq!(frames, tm_frames);
        assert_eq!(time_major.len(), frames * n_mels);
        for t in 0..frames {
            for m in 0..n_mels {
                assert_eq!(
                    time_major[t * n_mels + m],
                    reference[m * frames + t],
                    "time-major value mismatch at (t={t}, m={m})"
                );
            }
        }
    }

    /// 短于一个 FFT 窗的输入走早退分支，两种布局都必须仍是 `n_mels` 长。
    #[test]
    fn too_short_input_returns_single_frame() {
        let sr = 44_100u32;
        let waveform = vec![0.1f32; 16];
        let (mel, frames) = build_mel_from_waveform(&waveform, sr, 128, true).unwrap();
        assert_eq!(frames, 1);
        assert_eq!(mel.len(), 128);
    }
}
