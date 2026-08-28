pub fn is_available() -> bool {
    false
}

/// 与 onnx 变体同名；无 onnx 构建下无调用方，保留以维持接口一致。
#[allow(dead_code)]
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

pub fn infer_noise_mono(
    _clip_id: &str,
    _audio_mono: &[f32],
    _sample_rate: u32,
) -> Result<std::sync::Arc<Vec<f32>>, String> {
    Err("onnx feature disabled".to_string())
}

pub fn drop_shared_session() {}

pub fn update_ort_ep(_choice: &str, _device_id: Option<i32>) {}

pub fn ensure_cache_capacity(_min_capacity: usize) {}

/// 与 onnx 变体同名：take 切换等场景的缓存失效在 stub 下为 no-op。
pub fn clear_separation_cache() {}

/// 仅返回噪声 stem 的便捷封装；stub 下与完整分离一致地报错，
/// 调用方（气声路径）已有降级处理。
pub fn infer_noise_mono(
    _clip_id: &str,
    _audio_mono: &[f32],
    _sample_rate: u32,
) -> Result<std::sync::Arc<Vec<f32>>, String> {
    Err("onnx feature disabled".to_string())
}
