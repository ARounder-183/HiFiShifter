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
static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
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
                eprintln!("[hnsep] shared session dropped");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        eprintln!("[hnsep] WARNING: could not acquire SHARED_SESSION lock at shutdown — giving up");
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
                eprintln!("hnsep_onnx: unavailable: {e}");
            }
            false
        }
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
        eprintln!("[hnsep] LRU cache initialized with capacity={cap}");
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
        eprintln!("[hnsep] LRU cache resized: {current_cap} -> {next}");
    }
}

/// 清空全部谐波/噪声分离缓存。
///
/// 缓存 key 是 `clip_id+采样率+样本数` 的单向哈希，无法按 clip 反查剔除；
/// 而多 Take 切换后同一 clip_id 会对应不同源内容（等长 Take 尤其会命中
/// 旧 Take 的 stem，造成气声路径串音）。Take 切换/增删属于低频用户操作，
/// 直接整体清空、由后续渲染重建是正确且代价最小的失效策略。
pub fn clear_separation_cache() {
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    let cleared = cache.len();
    cache.clear();
    if cleared > 0 {
        eprintln!("[hnsep] separation cache cleared ({cleared} entries)");
    }
}

/// Compute a clip-level cache key using only identity fields (not audio content).
///
/// Uses FNV-1a 64-bit for low overhead. The key is based on `clip_id` + `sample_rate` +
/// `audio_len`, which is sufficient for per-clip HNSEP separation caching — two calls
/// for the same clip with the same clipped audio length will produce the same separation
/// output, regardless of which segment is being rendered.
fn separation_cache_key(clip_id: &str, sample_rate: u32, audio_len: usize) -> u64 {
    // FNV-1a 64-bit initial value
    let mut h: u64 = 14695981039346656037u64;

    for &b in clip_id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    for &b in &sample_rate.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    for &b in &(audio_len as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
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
    audio_len: usize,
    harmonic: Arc<Vec<f32>>,
    noise: Arc<Vec<f32>>,
) {
    let cache_key = separation_cache_key(clip_id, sample_rate, audio_len);
    let entry = HnsepCacheEntry { harmonic, noise };
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    cache.put(cache_key, entry);
}

pub fn infer_harmonic_noise_mono(
    clip_id: &str,
    audio_mono: &[f32],
    sample_rate: u32,
) -> Result<(Arc<Vec<f32>>, Arc<Vec<f32>>), String> {
    if let Err(e) = probe() {
        return Err(e.clone());
    }

    let audio_len = audio_mono.len();
    let cache_key = separation_cache_key(clip_id, sample_rate, audio_len);
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
