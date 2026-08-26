#[allow(dead_code)]
pub fn probe_load() {
    // ONNX feature disabled.
}

pub fn is_available() -> bool {
    false
}

// Stub for ONNX diagnostic info
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
    /// Always None in stub (ONNX not compiled). Real type: GpuDiagnostic.
    pub gpu_diagnostic: Option<serde_json::Value>,
}

pub fn diagnose_onnx_availability() -> OnnxDiagnosticInfo {
    OnnxDiagnosticInfo {
        compiled: false,
        available: false,
        error: Some("ONNX feature not compiled".to_string()),
        ep_choice: "disabled".to_string(),
        active_ep: "none".to_string(),
        onnx_version: None,
        providers: None,
        gpu_diagnostic: None,
    }
}

pub fn compiled() -> bool {
    false
}

pub fn model_load_error() -> Option<String> {
    Some("ONNX feature not compiled".to_string())
}

pub fn ep_choice() -> String {
    "disabled".to_string()
}

pub fn active_ep() -> String {
    "none".to_string()
}

pub fn update_ort_ep(_choice: &str, _device_id: Option<i32>) {}

pub fn drop_shared_session() {}

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
    pub gpu_available: bool,
    pub dml_available: bool,
    pub gpu_device_id: i32,
    pub available_providers: Vec<String>,
    pub ort_build_info: String,
    pub gpu_devices: Vec<serde_json::Value>,
}

pub fn run_benchmark() -> Result<BenchmarkResults, String> {
    Err("ONNX feature not compiled".to_string())
}

pub fn set_chunk_progress_callback(_cb: Option<Box<dyn Fn(f64) + Send + Sync>>) {}
pub fn reset_chunk_progress(_total: usize) {}

// ─── 分块推理 stub（与 nsf_hifigan_onnx.rs 接口保持一致）──────────────────────

pub fn env_chunk_sec() -> f64 {
    10.0
}

pub fn env_overlap_sec() -> f64 {
    0.1
}

pub fn infer_pitch_edit_chunked(
    mono: &[f32],
    _sample_rate: u32,
    _start_sec: f64,
    _midi_at_time: impl Fn(f64) -> f64 + Clone,
    _formant_shift_fn: impl Fn(f64) -> f32 + Clone,
    _chunk_sec: f64,
    _overlap_sec: f64,
) -> Result<Vec<f32>, String> {
    // ONNX feature disabled: bypass.
    Ok(mono.to_vec())
}

/// Mel-stretch variant used when caller performs time-stretch in mel-domain
/// before invoking the vocoder. Stubbed to match real interface when ONNX
/// feature is disabled so `cargo check` / builds that compile without the
/// `onnx` feature still succeed.
pub fn infer_pitch_edit_chunked_mel_stretch(
    mono: &[f32],
    _sample_rate: u32,
    _playback_rate: f64,
    _start_sec: f64,
    _midi_at_time: impl Fn(f64) -> f64 + Clone,
    _formant_shift_fn: impl Fn(f64) -> f32 + Clone,
    _chunk_sec: f64,
    _overlap_sec: f64,
) -> Result<Vec<f32>, String> {
    // ONNX feature disabled: bypass. We ignore playback_rate and formant
    // adjustments in the stub and return input PCM unchanged.
    Ok(mono.to_vec())
}

pub fn infer_pitch_edit_chunked_optimized(
    mono_pcm: &[f32],
    _sample_rate: u32,
    _start_sec: f64,
    _midi_at_time: impl Fn(f64) -> f64 + Clone,
    _formant_shift_at_time: impl Fn(f64) -> f32 + Clone,
    _chunk_cache_get: &dyn Fn(usize, usize) -> Option<Vec<f32>>,
    _chunk_cache_put: &dyn Fn(usize, usize, Vec<f32>),
) -> Result<Vec<f32>, String> {
    Ok(mono_pcm.to_vec())
}
