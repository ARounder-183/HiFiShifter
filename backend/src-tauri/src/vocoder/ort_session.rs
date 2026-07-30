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

/// Runtime override for DirectML device ID. Set by `set_runtime_dml_device_id()`.
/// Takes precedence over the `HIFISHIFTER_DML_DEVICE_ID` env var.
static RUNTIME_DML_DEVICE_ID: OnceLock<Mutex<Option<i32>>> = OnceLock::new();

/// Set the runtime EP override. Pass `None` to clear the override.
pub fn set_runtime_ep_override(ep: Option<String>) {
    if let Ok(mut guard) = RUNTIME_EP_OVERRIDE.get_or_init(|| Mutex::new(None)).lock() {
        *guard = ep;
    }
}

/// Set the runtime DirectML device ID override. Pass `None` to clear.
pub fn set_runtime_dml_device_id(device_id: Option<i32>) {
    if let Ok(mut guard) = RUNTIME_DML_DEVICE_ID.get_or_init(|| Mutex::new(None)).lock() {
        *guard = device_id;
    }
}

/// Resolve the DirectML device ID to use, in priority order:
/// 1. Runtime override (set via UI/settings)
/// 2. `HIFISHIFTER_DML_DEVICE_ID` env var
/// 3. Auto-detect via DXGI: pick the GPU with most VRAM
///
/// Always returns an explicit device_id. This uses `with_device_id(n)`
/// which calls `SessionOptionsAppendExecutionProvider_DML` (old API).
/// The newer `SessionOptionsAppendExecutionProvider_DML2` (used by
/// filter/preference options when device_id is None) has been observed
/// to create DML devices with significantly worse performance on
/// Ada Lovelace (RTX 4060) GPUs despite passing HighPerformance hints.
/// The old API with explicit device_id performs consistently across
/// all GPU architectures tested (Ampere, Ada, Pascal).
fn resolve_dml_device_id() -> Option<i32> {
    // 1. Runtime override (set via UI/settings) — user explicitly chose
    if let Some(id) = RUNTIME_DML_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| *g)
    {
        return Some(id);
    }
    // 2. Env var — explicit override
    if let Ok(val) = std::env::var("HIFISHIFTER_DML_DEVICE_ID") {
        if let Ok(id) = val.trim().parse::<i32>() {
            return Some(id);
        }
    }
    // 3. Auto-detect: pick the GPU with most VRAM.
    //    Always use explicit device_id (old DML API) regardless of
    //    whether the GPU is device 0 or not. The old API consistently
    //    outperforms DML2 across architectures.
    let adapters = crate::dml_adapters::enumerate_dml_adapters().adapters;
    if let Some(best) = adapters.first() {
        let device_id = best.device_id as i32;
        eprintln!(
            "ort_session: auto-detected DML device_id={device_id} name='{}' vram={}MB",
            best.name, best.dedicated_video_memory_mb
        );
        return Some(device_id);
    }
    // 4. No DXGI adapters — must rely on DML2 as last resort
    None
}

/// Returns the runtime EP override if set, otherwise falls back to per-role env var,
/// then global env var.
fn ep_choice_for_role(role: OrtSessionRole) -> String {
    // HNSEP (Separator) 强制使用 CPU。
    // GPU (DirectML/OpenCL) 对该模型的算子支持不完整，ORT 会将不支持的算子
    // 回退到 CPU 并插入 GPU↔CPU 数据搬运节点，导致推理反而比纯 CPU 更慢。
    // 这是一个临时方案，待 HNSEP 模型升级或 ORT 算子覆盖完善后可移除。
    if matches!(role, OrtSessionRole::Separator) {
        return "cpu".to_string();
    }

    // 1. Runtime override (set via UI or benchmark) — highest priority
    if let Some(ov) = RUNTIME_EP_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| g.clone())
    {
        return ov.to_ascii_lowercase();
    }
    // 2. Per-model env var (e.g. HIFISHIFTER_HNSEP_ORT_EP=cpu)
    let role_env = match role {
        OrtSessionRole::Vocoder => "HIFISHIFTER_HIFIGAN_ORT_EP",
        OrtSessionRole::PitchDetector => "HIFISHIFTER_FCPE_ORT_EP",
        OrtSessionRole::Separator => "HIFISHIFTER_HNSEP_ORT_EP",
    };
    if let Ok(val) = std::env::var(role_env) {
        let v = val.trim().to_ascii_lowercase();
        if !v.is_empty() {
            return v;
        }
    }
    // 3. Global env var fallback
    env_ep_choice()
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
///
/// # Safety
///
/// `ort::api()` panics if `ort::init()` has not been called.  Callers must
/// ensure `ensure_ort_init()` has succeeded before calling this function.
#[cfg(target_os = "linux")]
fn try_register_opencl_ep(
    mut builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    use ort::AsPointer;
    use std::ffi::CString;

    // Safety: caller must have called ort::init() successfully first.
    let api = ort::api();
    let provider_name =
        CString::new("OpenCLExecutionProvider").map_err(|e| format!("invalid provider name: {e}"))?;

    // Call the generic provider registration API.  This registers by name
    // rather than requiring a specific symbol.  No provider options needed.
    // NOTE: The standard ONNX Runtime Linux binary from the ort crate's
    // `download-binaries` feature is CPU-only — it does NOT include the
    // OpenCL provider.  This call will fail gracefully on default builds.
    // GPU acceleration on Linux requires either:
    //   a) A custom ORT build with --use_opencl, or
    //   b) The CUDA ort feature (requires NVIDIA GPU + CUDA Toolkit at runtime).
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
        eprintln!("ort_session: OpenCL EP registered successfully");
        Ok((builder, "opencl"))
    } else {
        Err(
            "OpenCL EP not available in this ONNX Runtime build (ORT was not compiled with --use_opencl). \
             Falling back to CPU."
                .to_string(),
        )
    }
}

#[cfg(not(target_os = "linux"))]
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
///
/// Registers BOTH DirectML AND CPU EP explicitly. When only DirectML is registered,
/// ORT implicitly adds CPU as a fallback but the graph partitioner may not make
/// optimal partitioning decisions — it doesn't "know" CPU is available as a target
/// until after the first partitioning pass. Explicit registration of both EPs lets
/// the partitioner plan the full EP assignment upfront, reducing partition boundaries.
#[cfg(target_os = "windows")]
fn try_register_directml_ep(
    builder: ort::session::builder::SessionBuilder,
    role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    eprintln!("ort_session: try_register_directml_ep");
    let device_id = resolve_dml_device_id();
    let dml = if let Some(id) = device_id {
        eprintln!("ort_session[{role:?}]: DirectML device_id={id} (old API)");
        ort::ep::DirectML::default()
            .with_device_id(id)
            .build()
    } else {
        // Last resort: no DXGI adapters found, fall back to DML2
        eprintln!("ort_session[{role:?}]: DirectML auto-select (DML2 fallback)");
        ort::ep::DirectML::default()
            .with_performance_preference(ort::ep::directml::PerformancePreference::HighPerformance)
            .with_device_filter(ort::ep::directml::DeviceFilter::Gpu)
            .build()
    };
    builder
        .with_execution_providers([dml])
        .map(|b| (b, "directml"))
        .map_err(|e| format!("enable DirectML EP failed: {e}"))
}

#[cfg(not(target_os = "windows"))]
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
    let choice = ep_choice_for_role(role);

    if choice == "cpu" || matches!(role, OrtSessionRole::Separator) {
        return build_cpu_session(onnx_path, role, &choice);
    }

    // ── GPU path: try DirectML first (Windows) ──────────────────────────
    #[cfg(target_os = "windows")]
    {
        // Attempt 1: strict DirectML (no CPU fallback).
        // If all operators in the model have native DirectML support,
        // the entire graph runs on GPU with ZERO partition boundaries.
        match build_dml_session_inner(onnx_path, role, &choice, true) {
            Ok((session, ep)) => return Ok((session, ep)),
            Err(e) => eprintln!(
                "ort_session[{role:?}]: strict DirectML failed (will retry with CPU fallback): {e}"
            ),
        }
        // Attempt 2: DirectML with CPU fallback.
        match build_dml_session_inner(onnx_path, role, &choice, false) {
            Ok((session, ep)) => return Ok((session, ep)),
            Err(e) => eprintln!("ort_session[{role:?}]: DirectML with fallback failed: {e}"),
        }
    }

    // ── GPU path: try OpenCL (Linux) ───────────────────────────────────
    #[cfg(target_os = "linux")]
    {
        match Session::builder() {
            Ok(mut builder) => {
                match try_register_opencl_ep(builder, role) {
                    Ok((b, ep)) => {
                        builder = b;
                        builder = builder
                            .with_optimization_level(GraphOptimizationLevel::Level3)
                            .map_err(|e| format!("set graph optimization level failed: {e}"))?
                            .with_memory_pattern(true)
                            .map_err(|e| format!("set memory pattern failed: {e}"))?;
                        let cores = std::thread::available_parallelism()
                            .map(|n| n.get()).unwrap_or(4).max(2);
                        let threads = match role {
                            OrtSessionRole::Separator => cores,
                            OrtSessionRole::Vocoder => (cores / 2).max(2),
                            OrtSessionRole::PitchDetector => (cores / 2).max(2),
                        };
                        builder = builder
                            .with_intra_threads(threads)
                            .map_err(|e| format!("set intra op threads failed: {e}"))?;
                        let session = builder
                            .commit_from_file(onnx_path)
                            .map_err(|e| format!("load onnx into ort session failed: {e}"))?;
                        eprintln!("ort_session[{role:?}]: OpenCL session created successfully");
                        return Ok((session, ep.to_string()));
                    }
                    Err(e) => eprintln!("ort_session[{role:?}]: OpenCL unavailable — {e}"),
                }
            }
            Err(e) => eprintln!(
                "ort_session[{role:?}]: failed to create session builder for OpenCL — {e}; falling back to CPU"
            ),
        }
    }

    // ── Fallback: CPU ──────────────────────────────────────────────────
    build_cpu_session(onnx_path, role, &choice)
}

/// Build a DirectML session with optional strict mode (no CPU fallback).
#[cfg(target_os = "windows")]
fn build_dml_session_inner(
    onnx_path: &Path,
    role: OrtSessionRole,
    choice: &str,
    strict: bool,
) -> Result<(Session, String), String> {
    let mut builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    let (builder, selected) = try_register_directml_ep(builder, role)?;

    eprintln!(
        "ort_session[{role:?}]: model={} ep={selected} strict={strict} (choice={choice}, global_env={})",
        onnx_path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        env_ep_choice(),
    );

    // ── DirectML-specific config ────────────────────────────────────────
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        .with_memory_pattern(false)
        .map_err(|e| format!("set memory pattern failed: {e}"))?
        .with_device_allocated_initializers()
        .map_err(|e| format!("enable device allocated initializers failed: {e}"))?
        .with_flush_to_zero()
        .map_err(|e| format!("enable flush-to-zero failed: {e}"))?
        .with_prepacking(false)
        .map_err(|e| format!("disable prepacking failed: {e}"))?
        // Override dynamic dimensions to fixed values. DirectML performs
        // best when shapes are known at session creation time because it
        // can pre-compile shaders and optimize GPU memory layouts. Dynamic
        // dimensions force DML to use less-optimized generic kernels.
        .with_dimension_override("batch", 1)
        .map_err(|e| format!("override batch dim failed: {e}"))?
        .with_dimension_override_by_denotation("time", 4096)
        .map_err(|e| format!("override time dim failed: {e}"))?;

    if strict {
        // Disable CPU fallback: if ANY op can't run on DirectML, session
        // creation FAILS. If it succeeds, the ENTIRE graph runs on GPU
        // with ZERO partition boundaries → no GPU↔CPU copies → maximum
        // throughput. This is the key to unlocking Pascal (GTX 10xx)
        // performance where ORT's partitioner otherwise sends too many
        // ops to CPU.
        builder = builder
            .with_disable_cpu_fallback()
            .map_err(|e| format!("disable cpu fallback failed: {e}"))?;
    } else {
        builder = builder
            .with_parallel_execution(true)
            .map_err(|e| format!("enable parallel execution failed: {e}"))?
            .with_inter_threads(2)
            .map_err(|e| format!("set inter threads failed: {e}"))?;
    }

    // ── Thread config ───────────────────────────────────────────────────
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2);
    let threads = if strict {
        // Strict mode: ALL ops on GPU. CPU threads are irrelevant —
        // keep 2 to avoid overhead.
        2
    } else {
        // Fallback mode: CPU handles unsupported ops.
        // Max threads for fastest CPU fallback throughput.
        cores.max(4)
    };

    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let t_create = std::time::Instant::now();
    let session = builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load onnx into ort session failed: {e}"))?;
    let create_ms = t_create.elapsed().as_millis();

    // ── Detailed diagnostic logging ────────────────────────────────────
    eprintln!(
        "ort_session[{role:?}]: created session ep={selected} strict={strict} intra_threads={threads} commit_ms={create_ms}",
    );
    // Log session I/O metadata (names, shapes, types)
    for input in session.inputs() {
        eprintln!(
            "ort_session[{:?}]:   input name='{}' dtype={:?}",
            role, input.name(), input.dtype()
        );
    }
    for output in session.outputs() {
        eprintln!(
            "ort_session[{:?}]:   output name='{}' dtype={:?}",
            role, output.name(), output.dtype()
        );
    }

    Ok((session, selected.to_string()))
}

/// Build a pure CPU session (used for "cpu" choice or fallback).
fn build_cpu_session(
    onnx_path: &Path,
    role: OrtSessionRole,
    choice: &str,
) -> Result<(Session, String), String> {
    let mut builder =
        Session::builder().map_err(|e| format!("create ort session builder failed: {e}"))?;

    eprintln!(
        "ort_session[{role:?}]: model={} ep=cpu (choice={choice}, global_env={})",
        onnx_path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        env_ep_choice(),
    );

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        .with_memory_pattern(true)
        .map_err(|e| format!("set memory pattern failed: {e}"))?;

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2);
    let threads = match role {
        OrtSessionRole::Separator => cores,
        OrtSessionRole::Vocoder => (cores / 2).max(2),
        OrtSessionRole::PitchDetector => (cores / 2).max(2),
    };
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let t_create = std::time::Instant::now();
    let session = builder
        .commit_from_file(onnx_path)
        .map_err(|e| format!("load onnx into ort session failed: {e}"))?;
    let create_ms = t_create.elapsed().as_millis();

    eprintln!(
        "ort_session[{role:?}]: created session ep=cpu intra_threads={threads} commit_ms={create_ms}",
    );

    Ok((session, "cpu".to_string()))
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
#[cfg(target_os = "windows")]
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
#[cfg(not(target_os = "windows"))]
const fn probe_directml_ep_available() -> bool {
    false
}

/// Quick check: try registering "OpenCLExecutionProvider" by name via the
/// generic `SessionOptionsAppendExecutionProvider` API.  Returns true if
/// the provider registered successfully (meaning ORT was compiled with
/// OpenCL support and the provider DLL/backend is available).
#[cfg(target_os = "linux")]
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
#[cfg(not(target_os = "linux"))]
const fn probe_opencl_ep_available() -> bool {
    false
}

/// Full GPU diagnostic: providers, device info.
///
/// Does NOT include a smoke test (that requires a model, handled by nsf_hifigan_onnx).
pub fn diagnose_gpu() -> GpuDiagnostic {
    let available_providers = diagnose_available_providers();
    let selected_ep = env_ep_choice();
    let gpu_device_id = 0;
    let ort_build_info = std::panic::catch_unwind(|| ort::info().to_string())
        .unwrap_or_else(|_| "ort::info() unavailable".to_string());

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
