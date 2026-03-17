use blake3::Hasher;
use lru::LruCache;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static SHARED_SESSION: OnceLock<Arc<Mutex<Session>>> = OnceLock::new();
static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
static LOGGED_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

const HNSEP_MODEL_SR: u32 = 44_100;
const HNSEP_CACHE_CAPACITY: usize = 32;

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

fn default_model_dir_guess() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest.join("resources").join("models").join("hnsep");
    if p.join("hnsep.onnx").is_file() {
        return Some(p);
    }
    None
}

fn resolve_model_path() -> Result<PathBuf, String> {
    if let Some(onnx) = env_path("HIFISHIFTER_HNSEP_ONNX") {
        return Ok(onnx);
    }

    if let Some(dir) = env_path("HIFISHIFTER_HNSEP_MODEL_DIR").or_else(default_model_dir_guess) {
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
        .map_err(|e| format!("load HNSEP onnx into ort session failed: {e}"))
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
                eprintln!("hnsep_onnx: unavailable: {e}");
            }
            false
        }
    }
}

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

fn linear_resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if in_rate == out_rate || in_rate == 0 || out_rate == 0 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_len];
    for (index, sample) in out.iter_mut().enumerate() {
        let src = index as f64 / ratio;
        let i0 = src.floor().max(0.0) as usize;
        let i1 = (i0 + 1).min(input.len().saturating_sub(1));
        let frac = (src - i0 as f64).clamp(0.0, 1.0) as f32;
        let a = input[i0];
        let b = input[i1];
        *sample = a + (b - a) * frac;
    }
    out
}

#[derive(Clone)]
struct HnsepCacheEntry {
    harmonic: Arc<Vec<f32>>,
    noise: Arc<Vec<f32>>,
}

static HNSEP_CACHE: OnceLock<Mutex<LruCache<u64, HnsepCacheEntry>>> = OnceLock::new();

fn global_cache() -> &'static Mutex<LruCache<u64, HnsepCacheEntry>> {
    HNSEP_CACHE.get_or_init(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(HNSEP_CACHE_CAPACITY).expect("HNSEP cache capacity must be non-zero"),
        ))
    })
}

fn separation_cache_key(clip_id: &str, sample_rate: u32, audio_mono: &[f32]) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(clip_id.as_bytes());
    hasher.update(&sample_rate.to_le_bytes());
    hasher.update(&(audio_mono.len() as u64).to_le_bytes());
    for sample in audio_mono {
        hasher.update(&sample.to_bits().to_le_bytes());
    }
    let bytes = hasher.finalize();
    let mut key = [0u8; 8];
    key.copy_from_slice(&bytes.as_bytes()[..8]);
    u64::from_le_bytes(key)
}

pub fn infer_harmonic_noise_mono(
    clip_id: &str,
    audio_mono: &[f32],
    sample_rate: u32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if let Err(e) = probe() {
        return Err(e.clone());
    }

    let cache_key = separation_cache_key(clip_id, sample_rate, audio_mono);
    {
        let mut cache = global_cache()
            .lock()
            .map_err(|e| format!("hnsep cache lock poisoned: {e}"))?;
        if let Some(entry) = cache.get(&cache_key) {
            return Ok((
                entry.harmonic.as_ref().clone(),
                entry.noise.as_ref().clone(),
            ));
        }
    }

    let model_audio = if sample_rate == HNSEP_MODEL_SR {
        audio_mono.to_vec()
    } else {
        linear_resample_mono(audio_mono, sample_rate, HNSEP_MODEL_SR)
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

    if sample_rate != HNSEP_MODEL_SR {
        harmonic = linear_resample_mono(&harmonic, HNSEP_MODEL_SR, sample_rate);
        noise = linear_resample_mono(&noise, HNSEP_MODEL_SR, sample_rate);
    }

    harmonic.resize(audio_mono.len(), 0.0);
    noise.resize(audio_mono.len(), 0.0);
    if harmonic.len() > audio_mono.len() {
        harmonic.truncate(audio_mono.len());
    }
    if noise.len() > audio_mono.len() {
        noise.truncate(audio_mono.len());
    }

    let entry = HnsepCacheEntry {
        harmonic: Arc::new(harmonic.clone()),
        noise: Arc::new(noise.clone()),
    };
    let mut cache = global_cache()
        .lock()
        .map_err(|e| format!("hnsep cache lock poisoned: {e}"))?;
    cache.put(cache_key, entry);

    Ok((harmonic, noise))
}
