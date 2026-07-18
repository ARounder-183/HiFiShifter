//! Shared ORT session builder with consistent optimization policy.
//!
//! All three ONNX models (NSF-HiFiGAN, FCPE, HNSEP) should use the same
//! optimization stack: Level3 graph opt, memory pattern, OpenCL/DirectML EP with
//! limits, and fallback to CPU. This module centralizes that configuration so
//! individual vocoder modules don't drift.

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde::Serialize;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Runtime override for EP choice. Set by `set_runtime_ep_override()`.
/// Takes precedence over the `HIFISHIFTER_ORT_EP` env var.
static RUNTIME_EP_OVERRIDE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

/// Set the runtime EP override. Pass `None` to clear the override.
pub fn set_runtime_ep_override(ep: Option<String>) {
    if let Ok(mut guard) = RUNTIME_EP_OVERRIDE.get_or_init(|| Mutex::new(None)).lock() {
        *guard = ep;
    }
}

/// Returns the runtime EP override if set, otherwise falls back to env var.
fn ep_choice() -> String {
    RUNTIME_EP_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_else(|| env_ep_choice())
}

fn env_ep_choice() -> String {
    std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_ascii_lowercase()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtSessionRole {
    /// NSF-HiFiGAN vocoder — full GPU budget, aggressive optimizations.
    Vocoder,
    /// FCPE pitch analysis — smaller model, share GPU with vocoder.
    PitchDetector,
    /// HNSEP harmonic+noise separation — medium model, share GPU with vocoder.
    Separator,
}

// ── EP registration helpers (feature-gated) ──────────────────────────────

/// Try to register OpenCL EP on a session builder.
/// Uses ONNX Runtime's generic `SessionOptionsAppendExecutionProvider` API
/// to register "OpenCLExecutionProvider" by name.  Works if the ORT DLL has
/// the OpenCL provider compiled in; returns a clear error otherwise.
#[cfg(feature = "opencl")]
fn try_register_opencl_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    use ort::AsPointer;
    use std::ffi::CString;

    eprintln!("ort_session: try_register_opencl_ep");

    let mut builder = builder;
    let api = ort::api();
    let provider_name =
        CString::new("OpenCLExecutionProvider").map_err(|e| format!("invalid provider name: {e}"))?;

    // Call the generic provider registration API.  This registers by name
    // rather than requiring a specific symbol.  No provider options needed.
    let status = unsafe {
        (api.SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            provider_name.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };

    if status.0.is_null() {
        Ok((builder, "opencl"))
    } else {
        Err(
            "OpenCL EP not available in this ONNX Runtime build (ORT was not compiled with --use_opencl)"
                .to_string(),
        )
    }
}

#[cfg(not(feature = "opencl"))]
fn try_register_opencl_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let _ = (builder, _role);
    Err("OpenCL EP not compiled in this build".to_string())
}

/// Try to register DirectML EP on a session builder.
/// DirectML uses DirectX 12 to accelerate ONNX models on any GPU (NVIDIA, AMD, Intel Arc).
/// It is Windows-only and requires no additional SDK or runtime DLLs beyond the ORT provider DLL.
#[cfg(feature = "directml")]
fn try_register_directml_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    eprintln!("ort_session: try_register_directml_ep");
    let ep = ort::ep::DirectML::default().build();
    builder
        .with_execution_providers([ep])
        .map(|b| (b, "directml"))
        .map_err(|e| format!("enable DirectML EP failed: {e}"))
}

#[cfg(not(feature = "directml"))]
fn try_register_directml_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let _ = (builder, _role);
    Err("DirectML EP not compiled in this build".to_string())
}

/// Build an ORT session with the full optimization policy.
///
/// All three models (NSF-HiFiGAN, FCPE, HNSEP) should call this instead of
/// building sessions ad-hoc with inconsistent settings.
///
/// Returns `(Session, selected_ep_name)`.
pub fn build_ort_session(onnx_path: &Path, role: OrtSessionRole) -> Result<(Session, String), String> {
    let mut builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    let choice = ep_choice();
    let mut selected: &str = "cpu";

    match choice.as_str() {
        "cpu" => {
            selected = "cpu";
        }
        _ => {
            // "auto" / "gpu" / "directml" / "opencl" / "cuda" →
            // try the platform-specific GPU backend, fall back to CPU.
            let mut tried = false;
            #[cfg(feature = "directml")]
            {
                match try_register_directml_ep(builder.clone(), role) {
                    Ok((b, ep)) => { builder = b; selected = ep; tried = true; }
                    Err(e) => { eprintln!("ort_session: {e}"); }
                }
            }
            #[cfg(feature = "opencl")]
            if !tried {
                match try_register_opencl_ep(builder.clone(), role) {
                    Ok((b, ep)) => { builder = b; selected = ep; tried = true; }
                    Err(e) => { eprintln!("ort_session: {e}"); }
                }
            }
            if !tried {
                selected = "cpu";
            }
        }
    }

    eprintln!(
        "ort_session: role={role:?} ep={selected} (ep_choice={choice:?}, env={:?})",
        env_ep_choice(),
    );

    // Level3: full graph optimization (operator fusion, constant folding, layout opt).
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?;

    // Pre-allocate tensor memory to avoid per-inference realloc.
    builder = builder
        .with_memory_pattern(true)
        .map_err(|e| format!("set memory pattern failed: {e}"))?;

    // Thread config: GPU path uses 1 thread (GPU parallelism), CPU path uses half cores.
    let threads = if selected == "cpu" {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(2)
    } else {
        1
    };
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let session = builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load onnx into ort session failed: {e}"))?;

    Ok((session, selected.to_string()))
}

// ─── GPU Diagnostic & Provider Enumeration ────────────────────────────────

/// Diagnostic info about GPU setup for user-facing reporting.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuDiagnostic {
    /// List of all ONNX Runtime execution provider names available in the DLL.
    pub available_providers: Vec<String>,
    /// The EP that was actually selected (e.g. "directml", "opencl", "cpu").
    pub selected_ep: String,
    /// GPU device ID that was requested (from env or default 0).
    pub gpu_device_id: i32,
    /// The ONNX Runtime build info string.
    pub ort_build_info: String,
}

/// Enumerate available ONNX Runtime execution providers.
///
/// Checks each provider by attempting to query its availability through ORT.
/// This is simpler and safer than using the raw C API for GetAvailableProviders.
pub fn diagnose_available_providers() -> Vec<String> {
    let mut providers = vec!["CPUExecutionProvider".to_string()];

    // DirectML is Windows-only
    if probe_directml_ep_available() {
        providers.push("DmlExecutionProvider".to_string());
    }

    // OpenCL works across platforms
    if probe_opencl_ep_available() {
        providers.push("OpenCLExecutionProvider".to_string());
    }

    providers
}

/// Quick check: try registering DirectML EP on a temporary session builder.
/// Returns true if DirectML EP is available in the loaded ORT DLL.
#[cfg(feature = "directml")]
fn probe_directml_ep_available() -> bool {
    match Session::builder() {
        Ok(builder) => {
            let ep = ort::ep::DirectML::default().build();
            match builder.with_execution_providers([ep]) {
                Ok(_) => true,
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

/// Stub: DirectML EP not compiled in.
#[cfg(not(feature = "directml"))]
const fn probe_directml_ep_available() -> bool {
    false
}

/// Quick check: try registering "OpenCLExecutionProvider" by name via the
/// generic `SessionOptionsAppendExecutionProvider` API.  Returns true if
/// the provider registered successfully (meaning ORT was compiled with
/// OpenCL support and the provider DLL/backend is available).
#[cfg(feature = "opencl")]
fn probe_opencl_ep_available() -> bool {
    use ort::AsPointer;
    use std::ffi::CString;

    // Safety: ort::api() panics if called before ort::init().  This probe is
    // only called from diagnose_available_providers() which is only reached
    // after ensure_ort_init() succeeds.
    let Ok(builder) = Session::builder() else {
        return false;
    };
    let Ok(provider_name) = CString::new("OpenCLExecutionProvider") else {
        return false;
    };
    let api = ort::api();
    let append_fn = api.SessionOptionsAppendExecutionProvider;
    let mut builder = builder;
    let status = unsafe {
        append_fn(
            builder.ptr_mut(),
            provider_name.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    let ok = status.0.is_null();
    eprintln!(
        "ort_session: probe_opencl_ep — {}",
        if ok { "AVAILABLE" } else { "NOT available (ORT built without --use_opencl)" }
    );
    ok
}

/// Stub: OpenCL EP not compiled in.
#[cfg(not(feature = "opencl"))]
const fn probe_opencl_ep_available() -> bool {
    false
}

/// Full GPU diagnostic: providers, device info.
///
/// Does NOT include a smoke test (that requires a model, handled by nsf_hifigan_onnx).
pub fn diagnose_gpu() -> GpuDiagnostic {
    let available_providers = diagnose_available_providers();
    let selected_ep = ep_choice();
    let gpu_device_id = 0;
    let ort_build_info = ort::info().to_string();

    GpuDiagnostic {
        available_providers,
        selected_ep,
        gpu_device_id,
        ort_build_info,
    }
}

/// RAII guard that temporarily overrides the EP choice for the duration of a
/// benchmark session, restoring the previous value on drop.
///
/// Used by `run_benchmark()` so it can force a specific EP (e.g. "cpu" or
/// "opencl") without permanently changing the process-wide setting.
///
/// Only sets the runtime override (mutex-protected), NOT the env var.
/// The runtime override takes precedence in `ep_choice()` over the env var,
/// so setting the env var is unnecessary (and would be unsafe in Rust).
pub struct EpOverrideGuard {
    prev_override: Option<String>,
}

impl EpOverrideGuard {
    pub fn new(ep: String) -> Self {
        // Save previous runtime override
        let prev_override = RUNTIME_EP_OVERRIDE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .ok()
            .and_then(|g| g.clone());

        // Set new runtime override
        set_runtime_ep_override(Some(ep.clone()));

        Self { prev_override }
    }
}

impl Drop for EpOverrideGuard {
    fn drop(&mut self) {
        // Restore runtime override
        set_runtime_ep_override(self.prev_override.clone());
    }
}
