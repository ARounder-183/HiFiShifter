pub fn is_available() -> bool {
    false
}

pub fn probe_load() -> Result<String, String> {
    Err("onnx feature disabled".to_string())
}

pub fn infer_harmonic_noise_mono(
    _clip_id: &str,
    _audio_mono: &[f32],
    _sample_rate: u32,
) -> Result<(std::sync::Arc<Vec<f32>>, std::sync::Arc<Vec<f32>>), String> {
    Err("onnx feature disabled".to_string())
}
