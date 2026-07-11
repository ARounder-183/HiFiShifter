//! Shared ORT session builder with consistent optimization policy.
//!
//! All three ONNX models (NSF-HiFiGAN, FCPE, HNSEP) should use the same
//! optimization stack: Level3 graph opt, memory pattern, CUDA EP with arena
//! limits, TF32, and heuristic conv search. This module centralizes that
//! configuration so individual vocoder modules don't drift.

use ort::ep;
use ort::ep::ExecutionProviderDispatch;
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

/// Runtime override for CUDA device ID. Set by `set_runtime_cuda_device_id()`.
/// Takes precedence over the `HIFISHIFTER_ORT_CUDA_DEVICE_ID` env var.
static RUNTIME_CUDA_DEVICE_ID: OnceLock<Mutex<Option<i32>>> = OnceLock::new();

/// Set the runtime CUDA device ID override. Pass `None` to clear the override.
/// This is called when the user changes the GPU device in the UI settings.
pub fn set_runtime_cuda_device_id(device_id: i32) {
    if let Ok(mut guard) = RUNTIME_CUDA_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
        *guard = Some(device_id);
    }
}

/// Clear the runtime CUDA device ID override (revert to env var or default 0).
pub fn clear_runtime_cuda_device_id() {
    if let Ok(mut guard) = RUNTIME_CUDA_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
        *guard = None;
    }
}

/// Returns the effective CUDA device ID, checking runtime override first,
/// then env var, then defaulting to 0.
fn cuda_device_id() -> i32 {
    RUNTIME_CUDA_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| *g)
        .unwrap_or_else(|| env_i32("HIFISHIFTER_ORT_CUDA_DEVICE_ID").unwrap_or(0))
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

fn env_i32(name: &str) -> Option<i32> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
}

/// Read `HIFISHIFTER_ORT_CUDA_MEM_LIMIT_MB` (default 8192 = 8 GB).
/// Returns bytes.
fn env_cuda_mem_limit_bytes() -> u64 {
    let mb = std::env::var("HIFISHIFTER_ORT_CUDA_MEM_LIMIT_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(8192);
    mb.saturating_mul(1024 * 1024)
}

fn build_cuda_ep(role: OrtSessionRole) -> ExecutionProviderDispatch {
    let device_id = cuda_device_id();
    let arena_bytes = env_cuda_mem_limit_bytes();

    eprintln!(
        "ort_session: build_cuda_ep role={role:?} device_id={device_id} arena_mb={}",
        arena_bytes / (1024 * 1024)
    );

    let mut ep = ep::CUDA::default()
        .with_device_id(device_id)
        .with_memory_limit(arena_bytes as usize)
        .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
        .with_tf32(true);

    // For the main vocoder, use larger arena; for analysis models, cap lower
    // to avoid VRAM contention when all sessions are alive.
    if matches!(role, OrtSessionRole::PitchDetector | OrtSessionRole::Separator) {
        // Analysis models are smaller; halve the arena to leave room for the vocoder.
        let reduced = arena_bytes / 2;
        ep = ep::CUDA::default()
            .with_device_id(device_id)
            .with_memory_limit(reduced as usize)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_tf32(true);
    }

    ep.build()
}

#[cfg(feature = "tensorrt")]
fn build_trt_ep(role: OrtSessionRole) -> ExecutionProviderDispatch {
    let device_id = cuda_device_id();
    let arena_bytes = env_cuda_mem_limit_bytes();

    let mut ep = ep::TensorRT::default()
        .with_device_id(device_id)
        .with_max_workspace_size(arena_bytes as usize)
        .with_fp16(true)
        .with_engine_cache(true)
        .with_engine_cache_path("K:\\REALAI\\dev\\trt_engine_cache")
        .with_timing_cache(true)
        .with_timing_cache_path("K:\\REALAI\\dev\\trt_timing_cache")
        .with_force_timing_cache(true)
        .with_build_heuristics(true)
        .with_builder_optimization_level(3);

    // For analysis models, cap workspace lower
    if matches!(role, OrtSessionRole::PitchDetector | OrtSessionRole::Separator) {
        ep = ep.with_max_workspace_size(arena_bytes as usize / 2);
    }

    ep.build()
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
    let selected: &str;

    match choice.as_str() {
        "cpu" => {
            selected = "cpu";
        }
        "cuda" => {
            builder = builder
                .with_execution_providers([build_cuda_ep(role)])
                .map_err(|e| format!("enable CUDA EP failed: {e}"))?;
            selected = "cuda";
        }
        #[cfg(feature = "tensorrt")]
        "trt" => {
            builder = builder
                .with_execution_providers([build_trt_ep(role), build_cuda_ep(role)])
                .map_err(|e| format!("enable TRT EP failed: {e}"))?;
            selected = "trt";
        }
        _ => {
            // "auto": try TRT first, then CUDA, fall back to CPU.
            #[cfg(feature = "tensorrt")]
            {
                match builder
                    .clone()
                    .with_execution_providers([build_trt_ep(role), build_cuda_ep(role)])
                {
                    Ok(b) => {
                        builder = b;
                        selected = "trt";
                    }
                    Err(e) => {
                        eprintln!("ort_session: TRT EP unavailable for {role:?}, trying CUDA: {e}");
                        match builder
                            .clone()
                            .with_execution_providers([build_cuda_ep(role)])
                        {
                            Ok(b) => {
                                builder = b;
                                selected = "cuda";
                            }
                            Err(e) => {
                                eprintln!("ort_session: CUDA EP unavailable for {role:?}, falling back to CPU: {e}");
                                selected = "cpu";
                            }
                        }
                    }
                }
            }
            #[cfg(not(feature = "tensorrt"))]
            {
                // TRT not available, try CUDA then CPU.
                match builder
                    .clone()
                    .with_execution_providers([build_cuda_ep(role)])
                {
                    Ok(b) => {
                        builder = b;
                        selected = "cuda";
                    }
                    Err(e) => {
                        eprintln!(
                            "ort_session: CUDA EP unavailable for {role:?}, falling back to CPU: {e}"
                        );
                        selected = "cpu";
                    }
                }
            }
        }
    }

    eprintln!(
        "ort_session: role={role:?} ep={selected} (ep_choice={choice:?}, env={:?}, cuda_mem_limit={}MB)",
        env_ep_choice(),
        env_cuda_mem_limit_bytes() / (1024 * 1024)
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

// ─── CUDA Diagnostic & Provider Enumeration ────────────────────────────────

/// Diagnostic info about CUDA/GPU setup for user-facing reporting.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CudaDiagnostic {
    /// List of all ONNX Runtime execution provider names available in the DLL.
    pub available_providers: Vec<String>,
    /// The EP that was actually selected (e.g. "cuda", "cpu").
    pub selected_ep: String,
    /// CUDA device ID that was requested (from env or default 0).
    pub cuda_device_id: i32,
    /// Whether a CUDA smoke test passed (GPU is actually executing).
    pub cuda_smoke_test_passed: bool,
    /// Human-readable error if smoke test failed.
    pub cuda_smoke_test_error: Option<String>,
    /// The ONNX Runtime build info string.
    pub ort_build_info: String,
    /// Critical CUDA DLLs found on disk (cublas, cudnn). Each entry is (dll_name, found).
    pub cuda_dll_status: Vec<(String, bool)>,
}

/// Enumerate available ONNX Runtime execution providers.
///
/// Checks each provider by attempting to query its availability through ORT.
/// This is simpler and safer than using the raw C API for GetAvailableProviders.
pub fn diagnose_available_providers() -> Vec<String> {
    let mut providers = vec!["CPUExecutionProvider".to_string()];

    // Try to register CUDA EP on a dummy session builder to check availability
    if probe_cuda_ep_available() {
        providers.push("CUDAExecutionProvider".to_string());
    }

    #[cfg(feature = "tensorrt")]
    {
        // TensorRT depends on CUDA being available
        if probe_cuda_ep_available() {
            providers.push("TensorrtExecutionProvider".to_string());
        }
    }

    providers
}

/// Quick check: try registering CUDA EP on a temporary session builder.
/// Returns true if CUDA EP is available in the loaded ORT DLL.
fn probe_cuda_ep_available() -> bool {
    // Try to build a minimal CUDA EP and check if the registration API is available
    match Session::builder() {
        Ok(builder) => {
            let ep = ep::CUDA::default()
                .with_device_id(0)
                .build();
            match builder.with_execution_providers([ep]) {
                Ok(_) => true,
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

/// Check if critical CUDA DLLs can be found via the system DLL search path.
/// Public for use in benchmarks and diagnostics.
pub fn probe_cuda_dlls() -> Vec<(String, bool)> {
    #[cfg(windows)]
    {
        let critical = [
            "cudart64_12.dll",  // CUDA Runtime (GPU communication)
            "cublas64_12.dll",  // cuBLAS (matrix ops)
            "cufft64_11.dll",   // cuFFT (FFT ops)
            "cudnn64_9.dll",    // cuDNN (deep neural network ops)
        ];
        critical
            .iter()
            .map(|name| {
                let found = probe_dll(name);
                (name.to_string(), found)
            })
            .collect()
    }
    #[cfg(not(windows))]
    {
        let critical = [
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcudart.so.12",
        ];
        critical
            .iter()
            .map(|name| {
                let found = probe_dll(name);
                (name.to_string(), found)
            })
            .collect()
    }
}

/// Check if a DLL/SO file exists in common locations on disk.
#[cfg(windows)]
fn probe_dll(name: &str) -> bool {
    // Check common locations
    let paths: Vec<Option<std::path::PathBuf>> = vec![
        // Next to the executable
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(name))),
        // ORT_LIB_LOCATION
        std::env::var("ORT_LIB_LOCATION")
            .ok()
            .map(|d| std::path::PathBuf::from(d).join(name)),
        // CUDA Toolkit v12.6
        Some(std::path::PathBuf::from(format!(
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\{name}"
        ))),
        // CUDA Toolkit v12.5
        Some(std::path::PathBuf::from(format!(
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin\\{name}"
        ))),
        // CUDA Toolkit v12.4
        Some(std::path::PathBuf::from(format!(
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin\\{name}"
        ))),
        // System32
        Some(std::path::PathBuf::from(format!(
            "C:\\Windows\\System32\\{name}"
        ))),
    ];
    for path in paths.iter().flatten() {
        if path.exists() {
            return true;
        }
    }
    false
}

#[cfg(not(windows))]
fn probe_dll(name: &str) -> bool {
    // On Linux, check common paths
    let paths = [
        format!("/usr/lib/x86_64-linux-gnu/{name}"),
        format!("/usr/local/cuda/lib64/{name}"),
        std::env::var("ORT_LIB_LOCATION")
            .ok()
            .map(|d| format!("{d}/{name}"))
            .unwrap_or_default(),
    ];
    for path in &paths {
        if std::path::Path::new(path).exists() {
            return true;
        }
    }
    false
}

/// Full CUDA diagnostic: providers, DLL status, device info.
///
/// Does NOT include a smoke test (that requires a model, handled by nsf_hifigan_onnx).
pub fn diagnose_cuda() -> CudaDiagnostic {
    let available_providers = diagnose_available_providers();
    let selected_ep = ep_choice();
    let dev_id = cuda_device_id();
    let ort_build_info = ort::info().to_string();
    let cuda_dll_status = probe_cuda_dlls();

    // Smoke test must be done with an actual model — handled by the caller
    let cuda_available = available_providers.iter().any(|p| p.contains("CUDA"));
    let (cuda_smoke_test_passed, cuda_smoke_test_error) = if cuda_available {
        (false, Some("Smoke test requires model — run benchmark to verify GPU execution".to_string()))
    } else {
        (false, Some("CUDAExecutionProvider not available in ONNX Runtime DLL".to_string()))
    };

    CudaDiagnostic {
        available_providers,
        selected_ep,
        cuda_device_id: dev_id,
        cuda_smoke_test_passed,
        cuda_smoke_test_error,
        ort_build_info,
        cuda_dll_status,
    }
}

/// RAII guard that temporarily overrides the EP choice for the duration of a
/// benchmark session, restoring the previous value on drop.
///
/// Used by `run_benchmark()` so it can force a specific EP (e.g. "cpu" or
/// "cuda") without permanently changing the process-wide setting.
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
