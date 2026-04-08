// FCPE ONNX pitch detector stub (used when `onnx` feature is disabled).

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
