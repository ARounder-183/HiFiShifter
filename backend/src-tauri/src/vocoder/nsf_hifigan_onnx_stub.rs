#[allow(dead_code)]
pub fn probe_load() {
    // ONNX feature disabled.
}

pub fn is_available() -> bool {
    false
}

pub fn infer_pitch_edit_mono(
    mono: &[f32],
    _sample_rate: u32,
    _start_sec: f64,
    _target_midi_at_time: impl Fn(f64) -> f64,
) -> Result<Vec<f32>, String> {
    // When ONNX is unavailable, behave as bypass (no pitch edit).
    Ok(mono.to_vec())
}

// Stub for ONNX diagnostic info
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
    OnnxDiagnosticInfo {
        compiled: false,
        available: false,
        error: Some("ONNX feature not compiled".to_string()),
        ep_choice: "disabled".to_string(),
        onnx_version: None,
        providers: None,
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
