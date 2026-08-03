//! Shared ORT session builder with consistent optimization policy.
//!
//! All three ONNX models (NSF-HiFiGAN, FCPE, HNSEP) should use the same
//! optimization stack. GPU acceleration is provided by:
//!   - WebGPU EP (Dawn backend) — cross-platform (Windows/Linux/macOS)
//!   - DirectML EP (DX12)        — Windows only, fallback
//!
//! On platforms without GPU prebuilt binaries, sessions gracefully fall
//! back to CPU. This module centralizes EP registration so individual
//! vocoder modules don't drift.

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde::Serialize;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Fixed mel-frame length used by CoreML sessions on macOS ARM64.
///
/// The NSF-HiFiGAN model has a dynamic `time` axis and its f0
/// pre-processing subgraph contains `Shape`/`Gather`/`Mod`/`ConstantOfShape`
/// operators.  ONNX Runtime's CoreML EP cannot always resolve shapes at
/// graph-compilation time when `time` is dynamic ("Unable to get shape for
/// output" / `model_builder` failures), even though the same model loads
/// fine on CPU/WebGPU.  Pinning `time` to this constant makes every shape in
/// the graph statically resolvable, which lets the CoreML EP compile and run
/// the model.  All inference entry points pad mel/f0 to this length and trim
/// the output back to the requested chunk length.
pub const COREML_FIXED_TIME_FRAMES: usize = 4096;

/// Stores whether the most recently created CoreML session for each role is
/// pinned to [`COREML_FIXED_TIME_FRAMES`].  Only the vocoder model requires
/// the fixed dimension; FCPE/HNSEP keep their dynamic shapes.
static COREML_PINNED_BY_ROLE: OnceLock<Mutex<Vec<(OrtSessionRole, bool)>>> = OnceLock::new();

/// Set once a CoreML smoke test times out or fails hard.  The CoreML EP is
/// then skipped for the rest of the process (WebGPU/CPU take over) so a
/// hung CoreML inference can never block the benchmark or rendering again.
static COREML_DISABLED: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();

fn coreml_disabled() -> bool {
    COREML_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .load(std::sync::atomic::Ordering::Relaxed)
}

fn disable_coreml(reason: &str) {
    eprintln!("ort_session: disabling CoreML EP for this process: {reason}");
    COREML_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Returns `true` when the current process is a macOS ARM64 build running
/// with the CoreML execution provider (i.e. sessions are pinned to
/// [`COREML_FIXED_TIME_FRAMES`] for the given role and inference inputs must
/// be padded).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn coreml_active(role: OrtSessionRole) -> bool {
    COREML_PINNED_BY_ROLE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .ok()
        .map(|g| g.iter().any(|(r, pinned)| *r == role && *pinned))
        .unwrap_or(false)
}

/// Stub for non-macOS-ARM64 builds.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn coreml_active(_role: OrtSessionRole) -> bool {
    false
}

/// Clear the CoreML pinned-state cache (e.g. when the runtime EP is changed,
/// so stale "coreml" state does not keep padding after switching to CPU).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn reset_coreml_pinned_state() {
    if let Ok(mut guard) = COREML_PINNED_BY_ROLE.get_or_init(|| Mutex::new(Vec::new())).lock() {
        guard.clear();
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn reset_coreml_pinned_state() {}

/// Record the CoreML pinned state for a role (macOS ARM64 only).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn set_coreml_pinned(role: OrtSessionRole, pinned: bool) {
    if let Ok(mut guard) = COREML_PINNED_BY_ROLE.get_or_init(|| Mutex::new(Vec::new())).lock() {
        if let Some(entry) = guard.iter_mut().find(|(r, _)| *r == role) {
            entry.1 = pinned;
        } else {
            guard.push((role, pinned));
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn set_coreml_pinned(_role: OrtSessionRole, _pinned: bool) {}

/// Build a CoreML execution provider with the options that make the
/// NSF-HiFiGAN model compile reliably on Apple Silicon.
///
/// - `MLProgram` format: supports more operators and is required for many
///   models that the legacy NeuralNetwork format rejects.
/// - `CPUAndNeuralEngine`: prefer the ANE for real-time vocoding, with CPU
///   fallback for unsupported ops.
/// - A persistent model cache avoids recompiling the CoreML model on every
///   session creation (can take tens of seconds for this 56 MB model).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_coreml_ep() -> ort::ep::CoreML {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};

    let mut ep = ort::ep::CoreML::default()
        .with_compute_units(ComputeUnits::CPUAndNeuralEngine)
        .with_model_format(ModelFormat::MLProgram)
        .with_static_input_shapes(false);

    // Cache compiled CoreML programs under ~/Library/Caches/HiFiShifter so
    // repeated session creation (e.g. after an EP switch) does not recompile
    // the whole model.  If the cache directory cannot be created, continue
    // without caching -- it is purely an optimization.
    if let Some(home) = std::env::var_os("HOME") {
        let cache = std::path::PathBuf::from(home)
            .join("Library")
            .join("Caches")
            .join("HiFiShifter")
            .join("coreml");
        if std::fs::create_dir_all(&cache).is_ok() {
            // with_model_cache_dir takes `impl ToString`, so convert the
            // PathBuf explicitly (PathBuf itself does not implement
            // ToString).
            ep = ep.with_model_cache_dir(cache.to_string_lossy().into_owned());
        }
    }

    ep
}

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
    // HNSEP (Separator) forces CPU.
    // GPU (WebGPU/DirectML) has incomplete operator coverage for this model;
    // ORT inserts GPU↔CPU data transfer nodes for unsupported ops, making
    // inference slower than pure CPU. This is a temporary workaround until
    // the HNSEP model is upgraded or ORT operator coverage improves.
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

// ── EP registration helpers ──────────────────────────────────────────────

/// Try to register WebGPU EP on a session builder.
///
/// Uses the ort crate's high-level `ep::WebGPU` type. The WebGPU EP
/// leverages Dawn (Google's WebGPU implementation) with platform-native
/// backends: Vulkan on Linux, Metal on macOS.
///
/// **Not compiled on Windows.** The Dawn/D3D12 backend in the `+wgpu`
/// ORT static binary can cause native crashes during D3D12 device
/// initialization on some GPU/driver combinations. Windows uses
/// DirectML instead, which is the mature, stable GPU path.
///
/// On WSL2, Vulkan hardware acceleration is not directly available.
/// Mesa's Lavapipe software renderer may work but with poor performance.
/// The EP will gracefully fall back to CPU if Dawn cannot find a
/// usable Vulkan device.
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "aarch64")))]
fn try_register_webgpu_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    // Detect WSL2 for diagnostic purposes
    let is_wsl = is_wsl2();

    // On Linux, Dawn uses the Vulkan backend. Configure it explicitly
    // and disable features that may cause issues on software renderers.
    //
    // On Windows, Dawn auto-selects D3D12. We don't try to force a
    // backend because the option string format differs between ORT
    // versions and an incorrect value can crash Dawn internally.
    //
    // IMPORTANT: WebGPU EP registration calls into Dawn native code.
    // On some GPU/driver combinations this can crash at the C level.
    // We wrap the registration in catch_unwind, but note that C-level
    // SIGSEGV cannot be caught — the best defense is to not auto-probe
    // WebGPU on Windows (which we already avoid).
    let wgpu = {
        let result = std::panic::catch_unwind(|| {
            if cfg!(target_os = "linux") {
                ort::ep::WebGPU::default()
                    .with_dawn_backend_type(ort::ep::webgpu::DawnBackendType::Vulkan)
                    .with_validation_mode(ort::ep::webgpu::ValidationMode::Disabled)
                    .with_enable_graph_capture(false)
                    .build()
            } else {
                ort::ep::WebGPU::default().build()
            }
        });
        match result {
            Ok(ep) => ep,
            Err(panic) => {
                let msg = format!(
                    "WebGPU EP build panicked: {}",
                    panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
                );
                eprintln!("ort_session: {msg}");
                return Err(msg);
            }
        }
    };

    let wsl_note = if is_wsl {
        " (WSL2: Vulkan may not be available; ensure mesa-vulkan-drivers are installed)"
    } else {
        ""
    };

    // Wrap the actual EP registration in catch_unwind as well.
    // Dawn init happens inside with_execution_providers() → register().
    let register_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        builder.with_execution_providers([wgpu.clone()])
    }));

    match register_result {
        Ok(Ok(b)) => {
            eprintln!("ort_session: WebGPU EP registered successfully{wsl_note}");
            Ok((b, "webgpu"))
        }
        Ok(Err(e)) => {
            let msg = format!(
                "WebGPU EP registration failed: {e}{wsl_note}"
            );
            eprintln!("ort_session: {msg}");
            Err(msg)
        }
        Err(panic) => {
            let msg = format!(
                "WebGPU EP registration panicked (likely Dawn/D3D12 init crash): {}{wsl_note}",
                panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
            );
            eprintln!("ort_session: {msg}");
            log_vulkan_diagnostics();
            Err(msg)
        }
    }
}

/// Stub: WebGPU EP not compiled on this platform (Windows, Linux ARM64, macOS x86_64).
/// On these platforms, DirectML (Windows) or CPU fallback is used instead.
#[cfg(not(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "aarch64"))))]
fn try_register_webgpu_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let _ = (builder, _role);
    Err("WebGPU EP is not compiled on this platform. Use DirectML (Windows) or CPU.".to_string())
}

/// Try to register the CoreML EP on a session builder (macOS ARM64).
///
/// CoreML is the primary GPU/Neural Engine path on Apple Silicon: it uses
/// Apple's Core ML framework (CPU + Neural Engine + GPU depending on the
/// selected compute units) instead of Dawn/WebGPU.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn try_register_coreml_ep(
    builder: ort::session::builder::SessionBuilder,
    _role: OrtSessionRole,
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let build_result = std::panic::catch_unwind(|| {
        build_coreml_ep().build()
    });
    let ep = match build_result {
        Ok(ep) => ep,
        Err(panic) => {
            let msg = format!(
                "CoreML EP build panicked: {}",
                panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
            );
            eprintln!("ort_session: {msg}");
            return Err(msg);
        }
    };

    let register_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        builder.with_execution_providers([ep.clone()])
    }));

    match register_result {
        Ok(Ok(b)) => {
            eprintln!("ort_session: CoreML EP registered successfully");
            Ok((b, "coreml"))
        }
        Ok(Err(e)) => {
            let msg = format!("CoreML EP registration failed: {e}");
            eprintln!("ort_session: {msg}");
            Err(msg)
        }
        Err(panic) => {
            let msg = format!(
                "CoreML EP registration panicked: {}",
                panic.downcast_ref::<&str>().copied().unwrap_or("unknown")
            );
            eprintln!("ort_session: {msg}");
            Err(msg)
        }
    }
}

/// Check if we're running under WSL2 (Windows Subsystem for Linux).
/// WSL2 does not expose native hardware Vulkan to Linux guests — the
/// Windows GPU driver provides D3D12/DirectX passthrough via /dev/dxg.
/// Mesa's Lavapipe software renderer may be available but offers poor
/// performance and limited SPIR-V feature support.
fn is_wsl2() -> bool {
    if cfg!(target_os = "linux") {
        if let Ok(version) = std::fs::read_to_string("/proc/version") {
            return version.to_lowercase().contains("microsoft")
                || version.to_lowercase().contains("wsl");
        }
    }
    false
}

/// Emit diagnostic info about the Vulkan environment. Helps debug
/// WebGPU/Dawn initialization failures, especially in WSL2 where
/// Vulkan ICD availability is limited.
fn log_vulkan_diagnostics() {
    // Check for Vulkan ICDs
    for icd_dir in ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"] {
        if let Ok(entries) = std::fs::read_dir(icd_dir) {
            eprintln!(
                "ort_session: vulkan ICD dir {icd_dir}: {} entries",
                entries.count()
            );
        }
    }
    // Check if /dev/dxg is present (WSL2 D3D12 passthrough)
    if std::path::Path::new("/dev/dxg").exists() {
        eprintln!("ort_session: WSL2 D3D12 passthrough (/dev/dxg) is available — WebGPU/Dawn cannot use this on Linux (Vulkan backend only)");
    }
    if is_wsl2() {
        eprintln!("ort_session: WSL2 detected — Vulkan hardware acceleration is not available from the Windows GPU driver.");
        eprintln!("ort_session: Only Mesa Lavapipe (software Vulkan) may be available, which is slow and may lack required Dawn features.");
        eprintln!("ort_session: For GPU acceleration on WSL2, use the Windows build of HiFiShifter with DirectML instead.");
    }
}

/// Try to register DirectML EP on a session builder.
///
/// DirectML uses DirectX 12 to accelerate ONNX models on any GPU
/// (NVIDIA, AMD, Intel Arc). It is Windows-only and requires no
/// additional SDK or runtime DLLs beyond the ORT provider DLL.
///
/// Registers BOTH DirectML AND CPU EP explicitly. When only DirectML
/// is registered, ORT implicitly adds CPU as a fallback but the graph
/// partitioner may not make optimal partitioning decisions. Explicit
/// registration of both EPs lets the partitioner plan the full EP
/// assignment upfront, reducing partition boundaries.
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
/// EP selection priority (for choice="auto"):
///   Windows:  1. DirectML (DX12, proven stable)  2. CPU fallback
///   Linux:    1. WebGPU (Dawn/Vulkan)            2. CPU fallback
///   macOS ARM64: 1. CoreML (Neural Engine/GPU)   2. WebGPU (Dawn/Metal)
///               3. CPU fallback
///
/// WebGPU on Windows is only used when explicitly selected
/// (choice="webgpu"), because Dawn/D3D12 probing can crash on some
/// GPU/driver combinations.  On Linux, Dawn/Vulkan is the primary
/// GPU path since there is no DirectML alternative.
///
/// Returns `(Session, selected_ep_name)`.
pub fn build_ort_session(onnx_path: &Path, role: OrtSessionRole) -> Result<(Session, String), String> {
    let choice = ep_choice_for_role(role);

    if choice == "cpu" || matches!(role, OrtSessionRole::Separator) {
        return build_cpu_session(onnx_path, role, &choice);
    }

    // ── Windows: DirectML first (proven stable, no crash risk) ─────────
    #[cfg(target_os = "windows")]
    if choice == "auto" || choice == "directml" || choice == "gpu" {
        // DirectML with strict mode first, then with fallback
        match build_dml_session_inner(onnx_path, role, &choice, true) {
            Ok((session, ep)) => return Ok((session, ep)),
            Err(e) => eprintln!(
                "ort_session[{role:?}]: strict DirectML failed (will retry with CPU fallback): {e}"
            ),
        }
        match build_dml_session_inner(onnx_path, role, &choice, false) {
            Ok((session, ep)) => return Ok((session, ep)),
            Err(e) => eprintln!("ort_session[{role:?}]: DirectML with fallback failed: {e}"),
        }
    }

    // ── Windows: WebGPU not compiled (uses DirectML instead) ──────────
    #[cfg(target_os = "windows")]
    if choice == "webgpu" {
        eprintln!("ort_session[{role:?}]: WebGPU is not available on Windows. Use DirectML ('auto' or 'directml' or 'gpu') for GPU acceleration, or 'cpu' for CPU-only.");
        // Fall through to CPU below.
    }

    // ── Linux x86_64: WebGPU first, then CPU ───────────────────────────
    // NOTE: Linux ARM64 does NOT have the webgpu feature (no prebuilt
    // ORT binary for aarch64+wgpu), so this block is excluded there.
    // NOTE: WSL2 is skipped because Dawn/Vulkan init hangs at shutdown
    // even when WebGPU sessions are never used for inference.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if (choice == "auto" || choice == "webgpu" || choice == "gpu") && !is_wsl2() {
        match Session::builder() {
            Ok(builder) => {
                match try_register_webgpu_ep(builder, role) {
                    Ok((b, ep)) => {
                        let session = build_gpu_session_finalize(b, onnx_path, role, "WebGPU")?;
                        return Ok((session, ep.to_string()));
                    }
                    Err(e) => {
                        eprintln!("ort_session[{role:?}]: WebGPU unavailable — {e}");
                        log_vulkan_diagnostics();
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "ort_session[{role:?}]: failed to create session builder for WebGPU — {e}"
                );
                log_vulkan_diagnostics();
            }
        }
    }

    // ── Fallback: CPU ──────────────────────────────────────────────────
    // macOS ARM64: CoreML (Neural Engine/GPU) first, WebGPU fallback, then CPU.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if choice == "auto" || choice == "coreml" || choice == "webgpu" || choice == "gpu" || choice == "directml" {
        // CoreML is the primary GPU path on Apple Silicon. An explicit
        // "webgpu" selection skips CoreML and goes straight to Dawn/Metal.
        if choice != "webgpu" && !coreml_disabled() {
            match Session::builder() {
                Ok(builder) => match try_register_coreml_ep(builder, role) {
                    Ok((b, ep)) => {
                        // Pin the vocoder model's dynamic `time` and `batch`
                        // dimensions.  Without this, the CoreML EP fails to
                        // compile this model ("Unable to get shape for
                        // output") because the f0 subgraph
                        // (Shape/Gather/Mod/ConstantOfShape) cannot resolve
                        // dynamic shapes, and a dynamic batch is a known
                        // CoreML EP crash/hang source.  FCPE/HNSEP keep
                        // dynamic shapes (they are not padded downstream).
                        let pinned = matches!(role, OrtSessionRole::Vocoder);
                        let pinned_builder = if pinned {
                            b.with_dimension_override("time", COREML_FIXED_TIME_FRAMES as i64)?
                                .with_dimension_override("batch", 1)
                        } else {
                            Ok(b)
                        };
                        match pinned_builder {
                            Ok(b) => {
                                match build_gpu_session_finalize(b, onnx_path, role, "CoreML") {
                                    Ok(session) => {
                                        set_coreml_pinned(role, pinned);
                                        return Ok((session, ep.to_string()));
                                    }
                                    Err(e) => eprintln!(
                                        "ort_session[{role:?}]: CoreML session creation failed (will try WebGPU): {e}"
                                    ),
                                }
                            }
                            Err(e) => eprintln!(
                                "ort_session[{role:?}]: failed to pin CoreML time dimension (will try WebGPU): {e}"
                            ),
                        }
                    }
                    Err(e) => eprintln!("ort_session[{role:?}]: CoreML unavailable: {e}"),
                },
                Err(e) => eprintln!("ort_session[{role:?}]: failed to create session builder for CoreML: {e}"),
            }
        }
        match Session::builder() {
            Ok(builder) => match try_register_webgpu_ep(builder, role) {
                Ok((b, ep)) => {
                    let session = build_gpu_session_finalize(b, onnx_path, role, "WebGPU")?;
                    return Ok((session, ep.to_string()));
                }
                Err(e) => eprintln!("ort_session[{role:?}]: WebGPU unavailable: {e}"),
            },
            Err(e) => eprintln!("ort_session[{role:?}]: failed to create session builder for WebGPU: {e}"),
        }
    }

    build_cpu_session(onnx_path, role, &choice)
}

/// Run a minimal inference through a newly-created GPU session to
/// verify that the EP can actually execute compute shaders.  On some
/// platforms (WSL2 with Lavapipe, misconfigured drivers, headless
/// systems) the EP registers successfully but runtime inference fails
/// — this catches that early so we can fall back to CPU.
fn smoke_test_gpu_session(
    mut session: Session,
    role: OrtSessionRole,
    ep_name: &str,
) -> Result<Session, String> {
    use ort::value::{Tensor, ValueType};

    // Collect f32 tensor input metadata first so the `session` borrow ends
    // before the session is moved into the helper thread below.
    let mut plans: Vec<(String, Vec<usize>)> = Vec::new();
    for input in session.inputs() {
        let (tensor_ty, shape) = match input.dtype() {
            ValueType::Tensor { ty, shape, .. } => (ty, shape),
            _ => continue,
        };
        if *tensor_ty != ort::value::TensorElementType::Float32 {
            continue;
        }
        if shape.iter().any(|&d| d == 0) {
            continue; // scalar or zero-dim - skip
        }
        // Replace dynamic dimensions (-1) with small test values:
        //   dim 0 -> 1 (batch),  other dims -> 4 (one upsampling hop).
        let test_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if d > 0 { d as usize } else if i == 0 { 1 } else { 4 })
            .collect();
        plans.push((input.name().to_string(), test_shape));
    }
    let mut input_pairs: Vec<(String, ort::value::Value)> = Vec::with_capacity(plans.len());
    for (name, test_shape) in plans {
        let total: usize = test_shape.iter().product::<usize>().max(1);
        let data: Vec<f32> = vec![0.0f32; total];
        let tensor = Tensor::from_array((test_shape, data.into_boxed_slice()))
            .map_err(|e| format!("smoke test: tensor '{name}' creation failed: {e}"))?;
        input_pairs.push((name, tensor.into()));
    }

    if input_pairs.is_empty() {
        eprintln!("ort_session[{role:?}]: GPU smoke test skipped (no f32 tensor inputs)");
        return Ok(session);
    }

    // CoreML can take a long time (or hang forever) on its first inference:
    // MLProgram compilation is deferred to the first run, and a dynamic
    // batch is a known hang source.  Run the smoke test on a helper thread
    // with a generous timeout so a stuck CoreML session can never freeze the
    // benchmark.  If it times out we return an error and the caller falls
    // back to WebGPU/CPU; the orphaned thread (if truly hung) is harmless.
    let (tx, rx) = std::sync::mpsc::channel();
    let timeout = std::time::Duration::from_secs(60);
    let run_thread = std::thread::spawn(move || {
        // Run inside a block so the returned SessionOutputs (which borrow
        // the session) are dropped before we move the session back.
        let result = {
            let outputs = session.run(input_pairs);
            outputs.map(|_| ())
        };
        let _ = tx.send((result, session));
    });

    match rx.recv_timeout(timeout) {
        Ok((result, session)) => {
            let _ = run_thread.join();
            match result {
                Ok(_) => {
                    eprintln!("ort_session[{role:?}]: {ep_name} smoke test passed - EP is functional");
                    Ok(session)
                }
                Err(e) => Err(format!(
                    "{ep_name} inference is not functional on this system.                      The EP registered but compute shader execution failed: {e}.                      Falling back to CPU."
                )),
            }
        }
        Err(_) => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            disable_coreml(&format!(
                "{ep_name} smoke test exceeded {timeout:?} (first inference hung)"
            ));
            Err(format!(
                "{ep_name} smoke test timed out after {timeout:?}.                  The EP registered but the first inference did not complete.                  Falling back to WebGPU/CPU."
            ))
        }
    }
}


/// Finalize a GPU session (CoreML / WebGPU) with appropriate optimization
/// settings, then smoke-test it so broken GPU backends fall back to CPU early.
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "aarch64")))]
fn build_gpu_session_finalize(
    mut builder: ort::session::builder::SessionBuilder,
    onnx_path: &Path,
    role: OrtSessionRole,
    ep_name: &str,
) -> Result<Session, String> {
    let model_name = onnx_path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
    eprintln!(
        "ort_session[{role:?}]: model={model_name} ep={ep_name} (global_env={})",
        env_ep_choice(),
    );

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

    let t_create = std::time::Instant::now();
    let mut session = builder
        .commit_from_file(onnx_path)
        .map_err(|e| {
            let msg = format!(
                "load onnx into {ep_name} ort session failed: {e}"
            );
            eprintln!("ort_session[{role:?}]: {msg}");
            msg
        })?;
    let create_ms = t_create.elapsed().as_millis();

    eprintln!(
        "ort_session[{role:?}]: created session ep={ep_name} intra_threads={threads} commit_ms={create_ms}",
    );

    // Log session I/O metadata
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

    // ── Smoke test: verify WebGPU can actually run inference ──────────
    // On some platforms (WSL2 with Lavapipe, headless systems, etc.)
    // the WebGPU EP registers successfully (Dawn finds a Vulkan device)
    // but compute shader execution fails at runtime.  Running a tiny
    // inference now catches this early and lets us fall back to CPU
    // instead of silently returning a broken session.
    match smoke_test_gpu_session(session, role, ep_name) {
        Ok(s) => session = s,
        Err(e) => {
            eprintln!("ort_session[{role:?}]: {ep_name} smoke test failed, discarding session and falling back to CPU: {e}");
            return Err(e);
        }
    }

    Ok(session)
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
    let mut session = builder
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

    // ── Smoke test: verify DirectML can actually run inference ─────────
    match smoke_test_gpu_session(session, role, "DirectML") {
        Ok(s) => session = s,
        Err(e) => {
            eprintln!("ort_session[{role:?}]: DirectML smoke test failed, discarding session and falling back to CPU: {e}");
            return Err(e);
        }
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
    /// The EP that was actually selected (e.g. "webgpu", "directml", "cpu").
    pub selected_ep: String,
    /// GPU device ID that was requested (from env or default 0).
    pub gpu_device_id: i32,
    /// The ONNX Runtime build info string.
    pub ort_build_info: String,
}

/// Enumerate available ONNX Runtime execution providers.
///
/// Checks each provider by attempting to query its availability through ORT.
///
/// NOTE: WebGPU probing is ONLY performed on Linux, where Dawn uses the
/// Vulkan backend which is safe to probe. On Windows, Dawn uses D3D12
/// and probing can trigger native crashes on some GPU/driver combos.
/// WebGPU on Windows is only used when the user explicitly selects it.
pub fn diagnose_available_providers() -> Vec<String> {
    let mut providers = vec!["CPUExecutionProvider".to_string()];

    // WebGPU — compiled on Linux x86_64 / macOS ARM64 only.
    // Excluded: Windows (Dawn/D3D12 crash risk), Linux ARM64 (no prebuilt binary),
    //           WSL2 (Vulkan not available; Dawn init hangs at shutdown).
    #[cfg(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "aarch64")))]
    if !is_wsl2() && probe_webgpu_ep_available() {
        providers.push("WebGpuExecutionProvider".to_string());
    }

    // CoreML -- macOS ARM64 only (Apple Neural Engine / GPU).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if probe_coreml_ep_available() {
        providers.push("CoreMLExecutionProvider".to_string());
    }

    // DirectML — Windows only
    if probe_directml_ep_available() {
        providers.push("DmlExecutionProvider".to_string());
    }

    providers
}

/// Quick check: try registering WebGPU EP on a temporary session builder.
/// Returns true if WebGPU EP is available in the loaded ORT binary.
///
/// Wrapped in catch_unwind because Dawn native code (Vulkan init)
/// can crash on some platforms. Only compiled on Linux/macOS ARM
/// where Dawn/Vulkan and Dawn/Metal are stable backends.
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "macos", target_arch = "aarch64")))]
fn probe_webgpu_ep_available() -> bool {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match Session::builder() {
            Ok(builder) => {
                let wgpu = if cfg!(target_os = "linux") {
                    ort::ep::WebGPU::default()
                        .with_dawn_backend_type(ort::ep::webgpu::DawnBackendType::Vulkan)
                        .build()
                } else {
                    ort::ep::WebGPU::default().build()
                };
                match builder.with_execution_providers([wgpu]) {
                    Ok(_) => {
                        eprintln!("ort_session: probe_webgpu_ep — AVAILABLE");
                        true
                    }
                    Err(e) => {
                        eprintln!("ort_session: probe_webgpu_ep — NOT available: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                eprintln!("ort_session: probe_webgpu_ep — session builder failed: {e}");
                false
            }
        }
    }));

    match result {
        Ok(available) => available,
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().copied().unwrap_or("unknown");
            eprintln!("ort_session: probe_webgpu_ep — PANICKED: {msg}");
            log_vulkan_diagnostics();
            false
        }
    }
}

/// Quick check: try registering the CoreML EP on a temporary session builder.
/// Returns true if CoreML EP is available in the loaded ORT binary.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn probe_coreml_ep_available() -> bool {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match Session::builder() {
            Ok(builder) => {
                let ep = build_coreml_ep().build();
                match builder.with_execution_providers([ep]) {
                    Ok(_) => {
                        eprintln!("ort_session: probe_coreml_ep - AVAILABLE");
                        true
                    }
                    Err(e) => {
                        eprintln!("ort_session: probe_coreml_ep - NOT available: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                eprintln!("ort_session: probe_coreml_ep - session builder failed: {e}");
                false
            }
        }
    }));

    match result {
        Ok(available) => available,
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().copied().unwrap_or("unknown");
            eprintln!("ort_session: probe_coreml_ep - PANICKED: {msg}");
            false
        }
    }
}

/// Quick check: try registering DirectML EP on a temporary session builder.
/// Returns true if DirectML EP is available in the loaded ORT binary.
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
/// "webgpu") without permanently changing the process-wide setting.
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
