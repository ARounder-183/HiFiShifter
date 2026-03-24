use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnnxStatusPayload {
    /// True when the `onnx` Cargo feature is compiled in.
    pub compiled: bool,
    /// True when the model file loaded successfully.
    pub available: bool,
    /// Human-readable error string when unavailable, null otherwise.
    pub error: Option<String>,
    /// Requested execution provider choice (cpu/cuda/auto/disabled).
    pub ep_choice: String,
}

pub(super) fn get_onnx_status() -> OnnxStatusPayload {
    let compiled = crate::nsf_hifigan_onnx::compiled();
    let available = if compiled {
        crate::nsf_hifigan_onnx::is_available()
    } else {
        false
    };
    let error = if available {
        None
    } else {
        crate::nsf_hifigan_onnx::model_load_error()
    };

    OnnxStatusPayload {
        compiled,
        available,
        error,
        ep_choice: crate::nsf_hifigan_onnx::ep_choice(),
    }
}

/// 获取ONNX诊断信息（详细版本）
pub(super) fn get_onnx_diagnostic_info() -> crate::nsf_hifigan_onnx::OnnxDiagnosticInfo {
    crate::nsf_hifigan_onnx::diagnose_onnx_availability()
}
