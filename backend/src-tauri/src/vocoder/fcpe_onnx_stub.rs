// FCPE ONNX pitch detector stub (used when `onnx` feature is disabled).

pub const FCPE_F0_MIN_HZ: f64 = 32.7;
pub const FCPE_F0_MAX_HZ: f64 = 1975.5;

pub fn is_available() -> bool {
    false
}

pub fn infer_f0_hz(
    _mono: &[f64],
    _sample_rate: u32,
    _frame_period_ms: f64,
    _f0_floor: f64,
    _f0_ceil: f64,
) -> Result<Vec<f64>, String> {
    Err("ONNX feature not compiled".to_string())
}

pub fn drop_shared_session() {}

pub fn update_ort_ep(_choice: &str, _device_id: Option<i32>) {}
