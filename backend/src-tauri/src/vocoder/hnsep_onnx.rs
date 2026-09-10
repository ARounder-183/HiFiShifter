use lru::LruCache;
use ort::session::Session;
use ort::value::Tensor;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static SHARED_SESSION: OnceLock<Mutex<Option<Arc<Mutex<Session>>>>> = OnceLock::new();
/// Cached EP name selected at session build time (for diagnostic reporting).
static SELECTED_EP: OnceLock<String> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

const HNSEP_MODEL_SR: u32 = 44_100;
/// HNSEP 分离缓存默认容量（可通过环境变量 HIFISHIFTER_HNSEP_CACHE_CAPACITY 覆盖）。
const HNSEP_CACHE_CAPACITY_DEFAULT: usize = 128;

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

fn build_session_with_ep(onnx_path: &Path) -> Result<Session, String> {
    let (session, ep) = crate::vocoder_ort_session::build_ort_session(
        onnx_path,
        crate::vocoder_ort_session::OrtSessionRole::Separator,
    )?;
    // Cache the EP name so diagnostics can report whether HNSEP is on GPU.
    let _ = SELECTED_EP.set(ep);
    Ok(session)
}

/// Returns the EP that was actually selected for the HNSEP session (for diagnostics).
#[allow(dead_code)]
pub fn selected_ep_name() -> Option<&'static str> {
    SELECTED_EP.get().map(|s| s.as_str())
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
    let _build_flight = crate::vocoder_ort_session::acquire_session_build_lock(
        std::time::Duration::from_secs(20),
    )?;
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
                log::error!("[hnsep] shared session dropped");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        log::error!("[hnsep] WARNING: could not acquire SHARED_SESSION lock at shutdown — giving up");
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

#[allow(dead_code)]
pub fn probe_load() -> Result<String, String> {
    ensure_ort_init()?;
    let onnx_path = resolve_model_path()?;
    let mut session = build_session_with_ep(&onnx_path)?;

    let waveform = vec![0.0f32; HNSEP_MODEL_SR as usize / 10];
    let waveform_tensor =
        Tensor::from_array(([1usize, waveform.len()], waveform.into_boxed_slice()))
            .map_err(|e| format!("build waveform tensor failed: {e}"))?;
    let outputs = session
        .run(ort::inputs![waveform_tensor])
        .map_err(|e| format!("hnsep ort session run failed: {e}"))?;
    if outputs.len() < 2 {
        return Err("hnsep ort returned fewer than 2 outputs".to_string());
    }
    Ok(format!(
        "hnsep_onnx: OK\n  onnx: {}\n  sr={}",
        onnx_path.display(),
        HNSEP_MODEL_SR
    ))
}

#[derive(Clone)]
struct HnsepCacheEntry {
    harmonic: Arc<Vec<f32>>,
    noise: Arc<Vec<f32>>,
}

static HNSEP_CACHE: OnceLock<Mutex<LruCache<u64, HnsepCacheEntry>>> = OnceLock::new();

fn global_cache() -> &'static Mutex<LruCache<u64, HnsepCacheEntry>> {
    HNSEP_CACHE.get_or_init(|| {
        let cap = hnsep_cache_initial_capacity();
        log::warn!("[hnsep] LRU cache initialized with capacity={cap}");
        Mutex::new(LruCache::new(
            NonZeroUsize::new(cap).expect("HNSEP cache capacity must be non-zero"),
        ))
    })
}

/// 确保 HNSEP 分离缓存容量不小于给定值（仅增不减）。
///
/// 渲染线程可在开始批量渲染前调用此函数，根据轨道上的 clip 数量动态扩容，
/// 避免在大量切片场景下因 LRU 容量不足导致缓存驱逐和重复推理。
pub fn ensure_cache_capacity(min_capacity: usize) {
    let next = min_capacity.max(1);
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    let current_cap = cache.cap().get();
    if next > current_cap {
        cache.resize(NonZeroUsize::new(next).unwrap());
        log::warn!("[hnsep] LRU cache resized: {current_cap} -> {next}");
    }
}

/// 清空全部谐波/噪声分离缓存。
///
/// 自 v1.1（P0-6）起缓存 key **包含音频内容摘要**，因此等长不同 Take 不会再
/// 命中彼此的 stem —— 本函数不再是正确性的必要条件，仅作为内存回收 /
/// 工程切换时的清理手段保留（HNSEP 结果重建代价较高，故不作过度清理）。
pub fn clear_separation_cache() {
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    let cleared = cache.len();
    cache.clear();
    if cleared > 0 {
        log::warn!("[hnsep] separation cache cleared ({cleared} entries)");
    }
}

#[inline]
fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    h
}

/// Compute a clip-level cache key.
///
/// 组成：`clip_id` + `sample_rate` + `audio_len` + **音频内容摘要**。
///
/// 为什么必须包含内容摘要：仅用 `clip_id` 的 key 无法区分**同一 clip 的多个
/// Take**。当两个 Take 长度恰好相同时，后一个 Take 会命中前一个 Take 的
/// 分离结果，导致气声/张力路径串音。此前靠"切换 Take 时整体清空缓存"规避，
/// 但那只是约定级防护，任何新增的绕行编辑路径都会破坏它（见 P0-6）。
///
/// 开销：FNV-1a 对每个样本 4 字节做一次乘加，数分钟 mono 音频约 10^7 次
/// 操作（十毫秒量级），相对 HNSEP 推理本身可忽略。
fn separation_cache_key(clip_id: &str, sample_rate: u32, audio: &[f32]) -> u64 {
    // FNV-1a 64-bit
    let mut h: u64 = 14695981039346656037u64;
    h = fnv1a_update(h, clip_id.as_bytes());
    h = fnv1a_update(h, &sample_rate.to_le_bytes());
    h = fnv1a_update(h, &(audio.len() as u64).to_le_bytes());
    for &s in audio {
        // 用位模式而非数值：需要区分任何不同的浮点表示。
        h = fnv1a_update(h, &s.to_bits().to_le_bytes());
    }
    h
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
) -> Result<Arc<Vec<f32>>, String> {
    infer_harmonic_noise_mono(clip_id, audio_mono, sample_rate).map(|(_, noise)| noise)
}

/// Pre-populate the HNSEP cache with a harmonic+noise pair for a given clip.
///
/// This allows the caller to perform HNSEP separation once at the clip level
/// and ensure subsequent per-segment ProcessorChain calls hit the cache,
/// avoiding redundant HNSEP inference for every segment of the same clip.
#[allow(dead_code)]
pub fn cache_separation(
    clip_id: &str,
    sample_rate: u32,
    audio_mono: &[f32],
    harmonic: Arc<Vec<f32>>,
    noise: Arc<Vec<f32>>,
) {
    let cache_key = separation_cache_key(clip_id, sample_rate, audio_mono);
    let entry = HnsepCacheEntry { harmonic, noise };
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    cache.put(cache_key, entry);
}

pub fn infer_harmonic_noise_mono(
    clip_id: &str,
    audio_mono: &[f32],
    sample_rate: u32,
) -> Result<(Arc<Vec<f32>>, Arc<Vec<f32>>), String> {
    // 非阻塞可用性检查：真正的会话构建由下方加载路径完成（在该调用线程上
    // 按需构建，不在 UI/命令/快照构建路径上同步构建）。
    if !is_available() {
        return Err("hnsep: model unavailable".to_string());
    }

    let audio_len = audio_mono.len();
    let cache_key = separation_cache_key(clip_id, sample_rate, audio_mono);
    {
        let mut cache = global_cache()
            .lock()
            .map_err(|e| format!("hnsep cache lock poisoned: {e}"))?;
        if let Some(entry) = cache.get(&cache_key) {
            // Verify length consistency before returning cached result.
            // If the cached audio differs in length from the request, treat as miss
            // (can happen when clip is trimmed/stretched after caching).
            if entry.harmonic.len().max(entry.noise.len()) >= audio_len {
                return Ok((entry.harmonic.clone(), entry.noise.clone()));
            }
            // Cached result too short → remove and re-infer below.
            cache.pop(&cache_key);
        }
    }

    let model_audio = if sample_rate == HNSEP_MODEL_SR {
        audio_mono.to_vec()
    } else {
        crate::mel_utils::linear_resample_mono(audio_mono, sample_rate, HNSEP_MODEL_SR)
    };

    let waveform_tensor =
        Tensor::from_array(([1usize, model_audio.len()], model_audio.into_boxed_slice()))
            .map_err(|e| format!("build hnsep waveform tensor failed: {e}"))?;

    let session = get_or_init_shared_session()?;
    let (mut harmonic, mut noise): (Vec<f32>, Vec<f32>) = {
        let mut session_guard = session
            .lock()
            .map_err(|e| format!("hnsep ort session lock poisoned: {e}"))?;
        let outputs = session_guard
            .run(ort::inputs![waveform_tensor])
            .map_err(|e| format!("hnsep ort run failed: {e}"))?;
        if outputs.len() < 2 {
            return Err("hnsep ort returned fewer than 2 outputs".to_string());
        }

        let mut iter = outputs.into_iter();
        let harmonic_output = iter
            .next()
            .ok_or_else(|| "hnsep ort missing harmonic output".to_string())?
            .1;
        let (_, harmonic_tensor) = harmonic_output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("hnsep harmonic output extract failed: {e}"))?;
        let noise_output = iter
            .next()
            .ok_or_else(|| "hnsep ort missing noise output".to_string())?
            .1;
        let (_, noise_tensor) = noise_output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("hnsep noise output extract failed: {e}"))?;
        (harmonic_tensor.to_vec(), noise_tensor.to_vec())
    };
    // Drop session lock before resampling (CPU work)

    if sample_rate != HNSEP_MODEL_SR {
        harmonic = crate::mel_utils::linear_resample_mono(&harmonic, HNSEP_MODEL_SR, sample_rate);
        noise = crate::mel_utils::linear_resample_mono(&noise, HNSEP_MODEL_SR, sample_rate);
    }

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
