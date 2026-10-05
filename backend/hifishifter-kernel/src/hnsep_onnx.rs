//! 谐波/噪声分离（HNSEP）的会话、缓存与推理编排。
//!
//! # 主要内容
//! - 模型路径解析（env → `hnsep_model_dir()` → 开发树 → 可执行文件同级）
//! - 全局共享 ORT 会话（`Separator` 角色）与 EP 选择
//! - 完整内容/model 权威、按字节有界的整段分离 LRU 缓存
//! - [`infer_harmonic_noise_mono`]：重采样 → STFT → mask 网络 → ISTFT → 重采样回
//!
//! # 模型形态：频谱域（mask-only）
//! 本模块使用**频谱域**模型：ONNX 里只有 mask 网络，输入 `spec [1,2,1025,T]`、
//! 输出 `mask [1,2,1025,T]`；STFT/ISTFT 由 [`crate::hnsep_dsp`] 在 Rust 侧完成。
//!
//! 早期版本用的是**波形域**模型（图内含 STFT + 24 LSTM + ISTFT，输入输出都是
//! `[1,N]`），它在 GPU 上几乎没有收益（实测 CoreML 1.02~1.04x）：LSTM 串行不可
//! 并行，且图内的 STFT/ISTFT 要么不被 EP 支持而回退 CPU，要么把图切碎。
//! 把 DSP 移出图外后，GPU 上剩下的是密集卷积网络，才有加速空间
//! （与 OpenUtau 的 `Hnsep.cs` 同一做法）。
//!
//! # 与其他模块的关系
//! - [`crate::hnsep_dsp`]：STFT / mask 施加 / ISTFT 的纯 DSP 实现。
//! - `hnsep_onnx_stub.rs`：`onnx` feature 关闭时的空实现。
//! - 上层调用方：`renderer::chain` 的 `HiFiGanStage::process_breath`。
//!
//! # 维护说明
//! 模型的采样率/FFT 参数必须与 `hnsep_dsp` 的调用参数一致；
//! 分离缓存的键**不含**任何曲线参数（分离只取决于源音频），
//! 改动键的构成会让改参数时白跑一次推理，或更糟 —— 命中错源的 stem。
//!
//! ## 采样率契约（曾因违反而真实出错）
//! [`infer_harmonic_noise_mono`] 的输入与两条 stem **必须都是调用方给的
//! `sample_rate`**。模型只在 44100 上工作，所以中间要把输入重采样到 44100，
//! 分离后**必须把 h 与 n 各自重采样回 `sample_rate`** —— 只做 truncate /
//! zero-pad 是错的：48000 下内容会只占前 44100/48000 = 91.87%（时间压缩 8.1%），
//! 尾部 8.1% 变成纯静音。
//! 回归测试：`hnsep_dsp::e2e_hnsep::stems_are_resampled_back_to_the_output_rate`。

use lru::LruCache;
use ort::session::Session;
use ort::value::Tensor;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
struct SeparatorSession {
    runtime: Mutex<Session>,
    identity: blake3::Hash,
}
static SHARED_SESSION: OnceLock<Mutex<Option<Arc<SeparatorSession>>>> = OnceLock::new();
/// Cached EP name selected at session build time (for diagnostic reporting).
static SELECTED_EP: OnceLock<String> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

const HNSEP_MODEL_SR: u32 = 44_100;
/// STFT 长度：与 `hnsep.yaml` 的 `n_fft` 一致（模型固定 1025 个频点）。
const HNSEP_N_FFT: usize = 2048;
/// STFT 帧步长：与 `hnsep.yaml` 的 `hop_length` 一致。
const HNSEP_HOP: usize = 512;
/// HNSEP 分离缓存默认容量（可通过环境变量 HIFISHIFTER_HNSEP_CACHE_CAPACITY 覆盖）。
const HNSEP_CACHE_CAPACITY_DEFAULT: usize = 128;
/// 缓存持有的两条stem按实际字节收费；这是kernel模型缓存上限，额外占用须与插件预算分别报告。
const HNSEP_CACHE_BYTES_DEFAULT: usize = 128 * 1024 * 1024;
static SEPARATION_FLIGHT: Mutex<()> = Mutex::new(());
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static INFERENCE_RUNS: AtomicU64 = AtomicU64::new(0);

/// 读取环境变量或使用默认值获取 HNSEP 缓存初始容量。
fn hnsep_cache_initial_capacity() -> usize {
    std::env::var("HIFISHIFTER_HNSEP_CACHE_CAPACITY")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(HNSEP_CACHE_CAPACITY_DEFAULT)
}

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

fn default_model_dir_guess() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest.join("resources").join("models").join("hnsep");
    if p.join("hnsep.onnx").is_file() {
        return Some(p);
    }

    // 开发树兜底：模型仍放在 app crate 的 `resources/` 下（那才是打包时带的东西）。
    // 见 `nsf_hifigan_onnx.rs` 里同一处的说明。
    if let Some(app_p) = manifest
        .parent()
        .map(|p| p.join("src-tauri").join("resources").join("models").join("hnsep"))
    {
        if app_p.join("hnsep.onnx").is_file() {
            return Some(app_p);
        }
    }

    // 发布/便携环境：模型位于可执行文件同级的 models/ 目录中
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let p = exe_dir.join("models").join("hnsep");
            if p.join("hnsep.onnx").is_file() {
                return Some(p);
            }
        }
    }

    None
}

fn resolve_model_path() -> Result<PathBuf, String> {
    if let Some(onnx) = env_path("HIFISHIFTER_HNSEP_ONNX") {
        return Ok(onnx);
    }

    if let Some(dir) = crate::hnsep_model_dir()
        .map(|p| p.to_path_buf())
        .or_else(|| env_path("HIFISHIFTER_HNSEP_MODEL_DIR"))
        .or_else(default_model_dir_guess)
    {
        let onnx = dir.join("hnsep.onnx");
        if onnx.is_file() {
            return Ok(onnx);
        }
    }

    Err(
        "HNSEP ONNX model not found. Set HIFISHIFTER_HNSEP_ONNX or HIFISHIFTER_HNSEP_MODEL_DIR."
            .to_string(),
    )
}

/// 会话与实际模型完整摘要绑定；模型加载过程中变化就拒绝，不缓存不明版本stem。
fn model_digest(path: &Path) -> Result<blake3::Hash, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e|format!("hnsep model open failed: {e}"))?;
    let mut digest = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e|format!("hnsep model read failed: {e}"))?;
        if count == 0 {break;}
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize())
}

fn build_session_with_ep(onnx_path: &Path) -> Result<SeparatorSession, String> {
    let digest = model_digest(onnx_path)?;
    let (session, ep) = crate::vocoder_ort_session::build_ort_session(
        onnx_path,
        crate::vocoder_ort_session::OrtSessionRole::Separator,
    )?;
    // Cache the EP name so diagnostics can report whether HNSEP is on GPU.
    if model_digest(onnx_path)? != digest {return Err("hnsep model changed during session build".into());}
    let _ = SELECTED_EP.set(ep.clone());
    let mut identity = blake3::Hasher::new();
    identity.update(b"hnsep-mask-stft-v2");identity.update(digest.as_bytes());identity.update(ep.as_bytes());
    Ok(SeparatorSession {runtime:Mutex::new(session),identity:identity.finalize()})
}

/// Returns the EP that was actually selected for the HNSEP session (for diagnostics).
#[allow(dead_code)]
pub fn selected_ep_name() -> Option<&'static str> {
    SELECTED_EP.get().map(|s| s.as_str())
}

/// 持久内容缓存绑定实际已加载模型/EP，不把用户切换设备的历史次数当模型身份。
pub fn cache_identity()->Result<String,String> {Ok(get_or_init_shared_session()?.identity.to_hex().to_string())}

fn get_or_init_shared_session() -> Result<Arc<SeparatorSession>, String> {
    let mutex = SHARED_SESSION.get_or_init(|| Mutex::new(None));
    // 快路径：已有会话直接克隆返回。
    if let Some(session) = mutex
        .lock()
        .map_err(|e| format!("SHARED_SESSION lock poisoned: {e}"))?
        .clone()
    {
        return Ok(session);
    }
    // 慢路径：构建在全局构建锁内进行（与 Vocoder / FCPE 串行化，避免并发
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
            log::warn!("[hnsep] inference device changed during session build — rebuilding with the new EP");
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
        let arc = Arc::new(session);
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
                log::error!("[hnsep] shared session dropped");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        log::error!(
            "[hnsep] WARNING: could not acquire SHARED_SESSION lock at shutdown — giving up"
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
    // 异步预热：立即在后台用新 EP 重建会话，把秒级构建移出渲染热路径。
    std::thread::Builder::new()
        .name("ort-session-prewarm-hnsep".into())
        .spawn(|| {
            if let Err(e) = get_or_init_shared_session() {
                log::warn!("[hnsep] session pre-warm failed: {e}");
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
        .name("ort-prewarm-hnsep".to_string())
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
                log::warn!("hnsep_onnx: unavailable: {e}");
            }
            false
        }
        None => true,
    }
}

/// 加载自检：建会话并跑一次**真实形态**的推理，用于诊断面板/人工排查。
///
/// 【为什么要跑真实形态】历史实现喂的是波形 `[1, N]` 并断言"至少 2 个输出" ——
/// 那是**波形域**模型的接口。换成 mask-only 模型后，那段代码若被调用会立刻失败
/// （输入名/形状不符、输出只有 1 个），表现为"模型坏了"的错误结论。
/// 自检必须走与生产一致的路径（`hnsep_dsp` 的 STFT + mask 网络），否则它证明不了
/// 任何事。这里刻意只跑一个最小 segment（32 帧）以保持自检轻量。
#[allow(dead_code)]
pub fn probe_load() -> Result<String, String> {
    ensure_ort_init()?;
    let onnx_path = resolve_model_path()?;
    let session = Arc::new(build_session_with_ep(&onnx_path)?);

    // 一个 segment 的静音：长度 = SEGMENT_FRAMES * hop，足够走完整流水线。
    let samples = crate::hnsep_dsp::SEGMENT_FRAMES * HNSEP_HOP;
    let audio = vec![0.0f32; samples];
    let window = crate::hnsep_dsp::periodic_hann(HNSEP_N_FFT);
    let bins = HNSEP_N_FFT / 2 + 1;

    let harmonic = crate::hnsep_dsp::separate(
        &audio,
        HNSEP_N_FFT,
        HNSEP_HOP,
        &window,
        |mask_input| {
            let frames = mask_input.len() / 2 / bins;
            let tensor = Tensor::from_array((
                [1usize, 2, bins, frames],
                mask_input.to_vec().into_boxed_slice(),
            ))
            .map_err(|e| format!("build spec tensor failed: {e}"))?;
            let mut guard = session.runtime
                .lock()
                .map_err(|e| format!("hnsep session lock poisoned: {e}"))?;
            let outputs = guard
                .run(ort::inputs![tensor])
                .map_err(|e| format!("hnsep ort session run failed: {e}"))?;
            let output = outputs
                .into_iter()
                .next()
                .ok_or_else(|| "hnsep ort returned no output".to_string())?;
            let (_, mask) = output
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("hnsep mask extract failed: {e}"))?;
            Ok(mask.to_vec())
        },
    )?;

    if harmonic.len() != samples {
        return Err(format!(
            "hnsep probe: expected {samples} samples, got {}",
            harmonic.len()
        ));
    }
    if !harmonic.iter().all(|v| v.is_finite()) {
        return Err("hnsep probe: non-finite output".to_string());
    }
    Ok(format!(
        "hnsep_onnx: OK (mask-only)\n  onnx: {}\n  sr={} n_fft={} hop={}",
        onnx_path.display(),
        HNSEP_MODEL_SR,
        HNSEP_N_FFT,
        HNSEP_HOP
    ))
}

#[derive(Clone)]
struct HnsepCacheEntry {
    harmonic: Arc<Vec<f32>>,
    noise: Arc<Vec<f32>>,
}

type SeparationKey = [u8;32];
struct SeparationCache {
    entries: LruCache<SeparationKey,HnsepCacheEntry>,
    bytes: usize,
    max_bytes: usize,
}
impl SeparationCache {
    fn new(capacity:usize,max_bytes:usize)->Self {
        Self {entries:LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap()),bytes:0,max_bytes}
    }
    fn entry_bytes(entry:&HnsepCacheEntry)->usize {
        (entry.harmonic.len()+entry.noise.len()).saturating_mul(std::mem::size_of::<f32>())
    }
    /// 成功stem仅以有界强引用缓存；超预算条目可返回给调用者，但不挤走其它有用缓存。
    fn put(&mut self,key:SeparationKey,entry:HnsepCacheEntry) {
        let bytes=Self::entry_bytes(&entry);
        if bytes>self.max_bytes {return;}
        if let Some(old)=self.entries.pop(&key) {self.bytes-=Self::entry_bytes(&old);}
        while self.bytes.saturating_add(bytes)>self.max_bytes||self.entries.len()==self.entries.cap().get() {
            let Some((_,old))=self.entries.pop_lru() else {break;};self.bytes-=Self::entry_bytes(&old);
        }
        self.entries.put(key,entry);self.bytes+=bytes;
    }
}
static HNSEP_CACHE: OnceLock<Mutex<SeparationCache>> = OnceLock::new();

fn global_cache() -> &'static Mutex<SeparationCache> {
    HNSEP_CACHE.get_or_init(|| {
        let cap = hnsep_cache_initial_capacity();
        log::warn!("[hnsep] LRU cache initialized with capacity={cap}, bytes={HNSEP_CACHE_BYTES_DEFAULT}");
        Mutex::new(SeparationCache::new(cap,HNSEP_CACHE_BYTES_DEFAULT))
    })
}

/// 确保 HNSEP 分离缓存容量不小于给定值（仅增不减）。
///
/// 渲染线程可在开始批量渲染前调用此函数，根据轨道上的 clip 数量动态扩容，
/// 避免在大量切片场景下因 LRU 容量不足导致缓存驱逐和重复推理。
pub fn ensure_cache_capacity(min_capacity: usize) {
    let next = min_capacity.max(1);
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    let current_cap = cache.entries.cap().get();
    if next > current_cap {
        cache.entries.resize(NonZeroUsize::new(next).unwrap());
        log::warn!("[hnsep] LRU cache resized: {current_cap} -> {next}");
    }
}

/// 清空全部谐波/噪声分离缓存。
///
/// 兼容原App的低频整体清理入口；正常查找已经由完整内容摘要自然防止等长换源误命中。
pub fn clear_separation_cache() {
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    let cleared = cache.entries.len();
    cache.entries.clear();cache.bytes=0;
    if cleared > 0 {
        log::warn!("[hnsep] separation cache cleared ({cleared} entries)");
    }
}

/// 分离仅由实际完整mono PCM、采样率与已加载model/EP决定，不由clip/view/轨道名决定。
/// 相同内容跨组件共享；同路径等长换源（包括中间样本变化）必须失效。
fn separation_cache_key(
    audio: &[f32],
    sample_rate: u32,
    model: blake3::Hash,
) -> SeparationKey {
    let mut hash=blake3::Hasher::new();hash.update(b"hnsep-complete-mono-pcm-v2");
    hash.update(model.as_bytes());hash.update(&sample_rate.to_le_bytes());hash.update(&(audio.len() as u64).to_le_bytes());
    for sample in audio {hash.update(&sample.to_bits().to_le_bytes());}
    *hash.finalize().as_bytes()
}

/// 只在非实时诊断/worker读取；次数不包含session构建的烟测。
pub fn separation_cache_stats()->(u64,u64,usize) {
    let bytes=global_cache().lock().unwrap_or_else(|e|e.into_inner()).bytes;
    (CACHE_HITS.load(Ordering::Relaxed),INFERENCE_RUNS.load(Ordering::Relaxed),bytes)
}

/// Get the noise stem for a clip, for use in computing breath_noise_stereo
/// without a second full ProcessorChain pass.
///
/// This is a convenience wrapper around [`infer_harmonic_noise_mono`] that
/// only returns the noise component. Used by `render_single_clip` to avoid
/// double HiFiGAN rendering in the breath path.
pub fn infer_noise_mono(
    clip_id: &str,
    audio_mono: &[f32],
    sample_rate: u32,
    channel_index: u16,
    source_fingerprint: Option<u64>,
) -> Result<Arc<Vec<f32>>, String> {
    infer_harmonic_noise_mono(
        clip_id,
        audio_mono,
        sample_rate,
        channel_index,
        source_fingerprint,
    )
    .map(|(_, noise)| noise)
}

/// 整段分离保持原DSP语义；完整内容缓存与single-flight都在worker，绝不切块HNSEP。
pub fn infer_harmonic_noise_mono(
    _clip_id: &str,
    audio_mono: &[f32],
    sample_rate: u32,
    _channel_index: u16,
    _source_fingerprint: Option<u64>,
) -> Result<(Arc<Vec<f32>>, Arc<Vec<f32>>), String> {
    // 非阻塞可用性检查：真正的会话构建由下方加载路径完成（在该调用线程上
    // 按需构建，不在 UI/命令/快照构建路径上同步构建）。
    if !is_available() {
        return Err("hnsep: model unavailable".to_string());
    }

    if sample_rate==0||audio_mono.iter().any(|v|!v.is_finite()) {return Err("hnsep: invalid PCM".into());}
    let audio_len = audio_mono.len();
    let session = get_or_init_shared_session()?;
    let cache_key = separation_cache_key(audio_mono,sample_rate,session.identity);
    // ORT本就串行使用唯一separator；把单飞边界前移到cache复查，避免并发重复STFT/推理。
    let _flight=SEPARATION_FLIGHT.lock().map_err(|e|format!("hnsep flight lock poisoned: {e}"))?;
    {
        let mut cache = global_cache()
            .lock()
            .map_err(|e| format!("hnsep cache lock poisoned: {e}"))?;
        if let Some(entry) = cache.entries.get(&cache_key) {
            // Verify length consistency before returning cached result.
            // If the cached audio differs in length from the request, treat as miss
            // (can happen when clip is trimmed/stretched after caching).
            if entry.harmonic.len()==audio_len&&entry.noise.len()==audio_len {
                CACHE_HITS.fetch_add(1,Ordering::Relaxed);
                return Ok((entry.harmonic.clone(), entry.noise.clone()));
            }
            // Cached result too short → remove and re-infer below.
            if let Some(old)=cache.entries.pop(&cache_key) {cache.bytes-=SeparationCache::entry_bytes(&old);}
        }
    }

    let model_audio = if sample_rate == HNSEP_MODEL_SR {
        audio_mono.to_vec()
    } else {
        crate::mel_utils::linear_resample_mono(audio_mono, sample_rate, HNSEP_MODEL_SR)
    };

    // ── 频谱域分离：STFT → mask 网络 → ISTFT ─────────────────────────────
    //
    // 模型只做 mask；STFT/ISTFT 在 Rust 侧（`hnsep_dsp`）。这与 OpenUtau 的
    // `Hnsep.cs` 一致，也是让 GPU 有可加速算子的前提（见模块头说明）。
    let window = crate::hnsep_dsp::periodic_hann(HNSEP_N_FFT);
    let harmonic = crate::hnsep_dsp::separate(
        &model_audio,
        HNSEP_N_FFT,
        HNSEP_HOP,
        &window,
        |mask_input| {
            let sep_frames = mask_input.len() / 2 / (HNSEP_N_FFT / 2 + 1);
            let spec_tensor = Tensor::from_array((
                [1usize, 2, HNSEP_N_FFT / 2 + 1, sep_frames],
                mask_input.to_vec().into_boxed_slice(),
            ))
            .map_err(|e| format!("build hnsep spec tensor failed: {e}"))?;

            let mut session_guard = session.runtime
                .lock()
                .map_err(|e| format!("hnsep ort session lock poisoned: {e}"))?;
            INFERENCE_RUNS.fetch_add(1,Ordering::Relaxed);
            let outputs = session_guard
                .run(ort::inputs![spec_tensor])
                .map_err(|e| format!("hnsep ort run failed: {e}"))?;
            // 先取出拥有所有权的 output，再借用其中的张量（否则临时值提前释放）。
            let output = outputs
                .into_iter()
                .next()
                .ok_or_else(|| "hnsep ort returned no output".to_string())?;
            let (_, mask_tensor) = output
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("hnsep mask output extract failed: {e}"))?;
            Ok(mask_tensor.to_vec())
        },
    )?;

    // 噪声 = 原信号 − 谐波（在**模型采样率**下相减，与波形域模型的定义一致）。
    // 必须在重采样回输出采样率**之前**做：此处 model_audio 与 harmonic 同域同长，
    // 相减结果精确满足 `h + n == model_audio`。
    let noise: Vec<f32> = model_audio
        .iter()
        .zip(harmonic.iter())
        .map(|(a, h)| a - h)
        .collect();

    // ── 重采样回输出采样率 ────────────────────────────────────────────────
    //
    // 【为什么必须重采样回去，而不是直接截断/补零】上面把输入重采样到了模型原生
    // 的 44100；分离得到的 stem 因此也是 **44100** 的。若只按输出长度 truncate /
    // zero-pad，就等于把 44100 的波形当成输出采样率的波形使用：
    // 48000 下内容只占前 44100/48000 = 91.87%，**时间被压缩 8.1%**、尾部 8.1%
    // 变成纯静音；再叠加声码器自身的 48k→44.1k 重采样，谐波包络与 F0 时间轴
    // 错位，听感即"跑调 / 整体不对"。模块头注释一直写着"…→ ISTFT → 重采样回"，
    // 但代码里缺了这一步（已有回归测试 `stems_are_resampled_back_to_the_output_rate`）。
    //
    // 线性重采样是**线性**算子，h 与 n 同长、用同一插值网格，故
    // `resample(h) + resample(n) == resample(h+n) == resample(model_audio)`，
    // 即 `h + n == x` 在非原生采样率下**同样成立**（仅差浮点误差）。
    let (harmonic, noise) = if sample_rate == HNSEP_MODEL_SR {
        (harmonic, noise)
    } else {
        (
            crate::mel_utils::linear_resample_mono(&harmonic, HNSEP_MODEL_SR, sample_rate),
            crate::mel_utils::linear_resample_mono(&noise, HNSEP_MODEL_SR, sample_rate),
        )
    };
    let (mut harmonic, mut noise) = (harmonic, noise);

    // Length normalization: ensure output matches input length exactly.
    // Resampling can produce ±1 sample drift; truncate or zero-pad as needed.
    let target_len = audio_mono.len();
    if harmonic.len() < target_len {
        harmonic.resize(target_len, 0.0);
    } else if harmonic.len() > target_len {
        harmonic.truncate(target_len);
    }
    if noise.len() < target_len {
        noise.resize(target_len, 0.0);
    } else if noise.len() > target_len {
        noise.truncate(target_len);
    }

    let harmonic_arc = Arc::new(harmonic);
    let noise_arc = Arc::new(noise);
    if harmonic_arc.iter().chain(noise_arc.iter()).any(|v|!v.is_finite()) {
        return Err("hnsep: non-finite separation output".into());
    }
    let entry = HnsepCacheEntry {
        harmonic: harmonic_arc.clone(),
        noise: noise_arc.clone(),
    };
    let mut cache = global_cache()
        .lock()
        .map_err(|e| format!("hnsep cache lock poisoned: {e}"))?;
    cache.put(cache_key, entry);

    Ok((harmonic_arc, noise_arc))
}

#[cfg(test)]
mod content_cache_tests {
    use super::*;
    fn entry(samples:usize,value:f32)->HnsepCacheEntry {
        HnsepCacheEntry {harmonic:Arc::new(vec![value;samples]),noise:Arc::new(vec![0.;samples])}
    }
    #[test]
    fn complete_content_and_loaded_model_define_separation_identity() {
        let model=blake3::hash(b"model-a");let original=vec![0.25;1024];let mut changed=original.clone();changed[512]=0.5;
        let key=separation_cache_key(&original,44100,model);
        assert_ne!(key,separation_cache_key(&changed,44100,model),"等长换源中间变化不能命中旧stem");
        assert_ne!(key,separation_cache_key(&original,48000,model));
        assert_ne!(key,separation_cache_key(&original,44100,blake3::hash(b"model-b")));
        assert_eq!(key,separation_cache_key(&original.clone(),44100,model),"身份不依赖轨道/view/clip名");
    }
    #[test]
    fn stem_cache_evicts_by_bytes_and_preserves_referenced_audio() {
        let mut cache=SeparationCache::new(128,32);let old=entry(2,0.25);let held=old.harmonic.clone();
        cache.put([1;32],old);cache.put([2;32],entry(2,0.5));assert_eq!(cache.bytes,32);
        cache.entries.get(&[1;32]);cache.put([3;32],entry(2,0.75));
        assert!(cache.entries.contains(&[1;32]));assert!(!cache.entries.contains(&[2;32]));
        cache.put([1;32],entry(1,1.));assert_eq!(cache.bytes,24);
        cache.put([4;32],entry(5,2.));assert_eq!(cache.bytes,24,"超预算条目不冲掉已有缓存");
        cache.put([5;32],entry(3,3.));assert_eq!(cache.bytes,32);
        assert_eq!(held.as_slice(),&[0.25,0.25],"驱逐只释放缓存引用，不损坏活worker的Arc");
    }
    #[test]
    #[ignore = "真实CPU模型诊断：显式运行，缺模型必须失败而非skip"]
    fn real_model_content_cache_is_single_flight_and_rejects_same_length_changed_source() {
        crate::vocoder_ort_session::set_runtime_ep_override(Some("cpu".into()));
        clear_separation_cache();
        let input:Arc<Vec<f32>>=Arc::new((0..44100).map(|i| {
            let phase=2.*std::f64::consts::PI*220.*i as f64/44100.;
            (1..=12).map(|k|0.15/k as f64*(phase*k as f64).sin()).sum::<f64>() as f32
        }).collect());
        let before=separation_cache_stats();let began=std::time::Instant::now();
        let jobs=(0..2).map(|index| {let input=input.clone();std::thread::spawn(move ||
            infer_harmonic_noise_mono(&format!("owner-{index}"),&input,44100,index,Some(7)).unwrap())}).collect::<Vec<_>>();
        let mut outputs=jobs.into_iter().map(|job|job.join().unwrap());let a=outputs.next().unwrap();let b=outputs.next().unwrap();
        let cold=began.elapsed();let after=separation_cache_stats();
        assert_eq!(after.1-before.1,1);assert_eq!(after.0-before.0,1);
        assert!(Arc::ptr_eq(&a.0,&b.0));assert!(Arc::ptr_eq(&a.1,&b.1));
        let reconstruction=a.0.iter().zip(a.1.iter()).zip(input.iter()).map(|((h,n),x)|(h+n-x).abs()).fold(0_f32,f32::max);
        assert!(reconstruction<1e-5);assert_eq!(a.0.len(),44100);
        let mut changed=input.as_ref().clone();changed[22050]+=0.1;
        let c=infer_harmonic_noise_mono("owner-0",&changed,44100,0,Some(7)).unwrap();
        assert!(!Arc::ptr_eq(&a.0,&c.0));assert_eq!(separation_cache_stats().1-after.1,1);
        println!("HNSEP_REAL cold_single_flight_ms={} runs={} hits={} cache_bytes={} reconstruction_max={reconstruction:e} changed_source_runs=1",
            cold.as_millis(),after.1-before.1,after.0-before.0,after.2);
    }
}
