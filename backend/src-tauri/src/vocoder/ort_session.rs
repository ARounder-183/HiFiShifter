//! Shared ORT session builder with consistent optimization policy.
//!
//! All three ONNX models (NSF-HiFiGAN, FCPE, HNSEP) should use the same
//! optimization stack. GPU acceleration is provided by:
//!   - CoreML EP (Metal / Neural Engine) — macOS ARM64, primary GPU path
//!   - WebGPU EP (Dawn backend)          — Linux x86_64 (Vulkan), macOS (Metal)
//!   - DirectML EP (DX12)                — Windows only
//!
//! On platforms without GPU prebuilt binaries, sessions gracefully fall
//! back to CPU. This module centralizes EP registration so individual
//! vocoder modules don't drift.
//!
//! # Shape policy
//!
//! Sessions keep the models' dynamic axes (`batch`, `time`) **unpinned**.
//! An earlier revision pinned the vocoder's `time` axis to a constant so the
//! CoreML EP could compile the graph, but that made CoreML ~1.7x *slower*
//! than CPU (measured: 10.2 s vs 6.2 s for 47.6 s of audio) and forced every
//! chunk — however short — to be padded up to that constant.  On ONNX Runtime
//! 1.28 the CoreML EP compiles the dynamic graph without help and runs it
//! ~7.7x faster than CPU with bit-comparable output.  See
//! `docs/hifigan-gpu-acceleration.md` for the full measurement table.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use serde::Serialize;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Mel-frame length used by the post-creation GPU smoke test.
///
/// The NSF-HiFiGAN model's f0 pre-processing subgraph derives a `Pad` size
/// from the f0 tensor, so tiny inputs make ORT's buffer-reuse optimizer
/// collide with the model's fixed intermediate shapes ("{1,4,1} !=
/// {1,2048,1}").  Probing at the renderer's chunk size keeps every
/// intermediate shape valid.  This is *only* a smoke-test probe length —
/// sessions themselves stay fully dynamic.
pub const SMOKE_TEST_FRAMES: usize = 4096;

/// Set once a CoreML smoke test times out or fails hard.  The CoreML EP is
/// then skipped for the rest of the process (WebGPU/CPU take over) so a
/// hung CoreML inference can never block the benchmark or rendering again.
static COREML_DISABLED: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn coreml_disabled() -> bool {
    COREML_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn disable_coreml(reason: &str) {
    eprintln!("ort_session: disabling CoreML EP for this process: {reason}");
    COREML_DISABLED
        .get_or_init(|| std::sync::atomic::AtomicBool::new(false))
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Build a CoreML execution provider with the options that make the
/// NSF-HiFiGAN model compile reliably on Apple Silicon.
///
/// - `MLProgram` format: supports more operators and is required for many
///   models that the legacy NeuralNetwork format rejects.
/// - `CPUAndGPU`: prefer the Metal GPU for real-time vocoding, with CPU
///   fallback for unsupported ops.
/// - Dynamic input shapes: the models keep their dynamic `batch`/`time`
///   axes.  Requiring static shapes here was measured to make the vocoder
///   ~1.7x *slower* than the CPU EP (see the module docs).
/// - A persistent model cache avoids recompiling the CoreML model on every
///   session creation (can take tens of seconds for this 56 MB model).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_coreml_ep() -> ort::ep::CoreML {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};

    let mut ep = ort::ep::CoreML::default()
        // Use the GPU instead of the Neural Engine: HiFi-GAN's ConvTranspose
        // upsampling layers (stride/kernel 16/8/4/2) are known to hang the
        // ANE compiler (see Apple Developer Forums: ConvTranspose2d with
        // stride(16,1) kernel(16,1) breaks the ANE).  CPUAndGPU keeps the
        // model on the Metal GPU where these layers run correctly.
        .with_compute_units(ComputeUnits::CPUAndGPU)
        .with_model_format(ModelFormat::MLProgram)
        // Keep FP32 accumulation: HiFi-GAN audio quality is sensitive to
        // low-precision GPU accumulation.
        .with_low_precision_accumulation_on_gpu(false)
        // Every role keeps the model's dynamic `batch`/`time` axes.
        // Requiring static shapes forces CoreML to compile one fully-static
        // MLProgram, which benchmarked ~1.7x SLOWER than the CPU EP.
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
            // Versioned separately from the old ORT 1.24 cache: compiled
            // CoreML artifacts from the previous runtime can reuse stale
            // partitions and must not be loaded by the new build.
            .join("coreml-ort1.28");
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
    if let Ok(mut guard) = RUNTIME_DML_DEVICE_ID
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
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

/// Default EP for each role when nothing is explicitly configured.
///
/// Only HNSEP (Separator) deviates: it defaults to CPU because the CoreML EP
/// was measured to give it essentially no speedup (5 s clip: 310 ms CPU vs
/// 304 ms CoreML; 10 s clip: 553 ms vs 531 ms — ~2-4%) while adding 0.6-1.2 s
/// of one-off CoreML model compilation.  HNSEP separation is cached per clip
/// and therefore runs once per clip, so the extra compilation cost is not
/// amortised.  See `docs/hifigan-gpu-acceleration.md`.
fn default_ep_for_role(role: OrtSessionRole) -> &'static str {
    match role {
        OrtSessionRole::Separator => "cpu",
        OrtSessionRole::Vocoder | OrtSessionRole::PitchDetector => "auto",
    }
}

/// Resolve the EP choice for one model, in priority order:
/// 1. Per-model env var (e.g. `HIFISHIFTER_HNSEP_ORT_EP=coreml`) — an
///    explicit, per-model request always wins, including for HNSEP.
/// 2. The process-wide runtime override (set by the UI settings or the
///    benchmark's [`EpOverrideGuard`]).
/// 3. The global `HIFISHIFTER_ORT_EP` env var.
/// 4. The role's default (see [`default_ep_for_role`]).
fn ep_choice_for_role(role: OrtSessionRole) -> String {
    // 1. Per-model env var — highest priority so a user can opt any single
    //    model (including HNSEP) onto the GPU even though its default is CPU.
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

    // 2. Runtime override (set via UI or benchmark)
    if let Some(ov) = RUNTIME_EP_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|g| g.clone())
    {
        let ov = ov.to_ascii_lowercase();
        // HNSEP stays on its default unless the request names a concrete EP,
        // because its per-clip GPU gain does not repay the compile cost.
        if !matches!(role, OrtSessionRole::Separator)
            || !matches!(ov.as_str(), "auto" | "gpu" | "directml")
        {
            return ov;
        }
    }

    // 3. Global env var, else 4. the role default.
    global_env_ep_choice().unwrap_or_else(|| default_ep_for_role(role).to_string())
}

/// The global `HIFISHIFTER_ORT_EP` env var, normalised, when it is set to a
/// non-empty value.  `None` means "no global preference configured".
fn global_env_ep_choice() -> Option<String> {
    std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
}

/// The global env var, defaulting to `"auto"`.  Only used for diagnostics.
fn env_ep_choice() -> String {
    global_env_ep_choice().unwrap_or_else(|| "auto".to_string())
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
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
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
            let msg = format!("WebGPU EP registration failed: {e}{wsl_note}");
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
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
#[allow(dead_code)] // call sites live in linux-x86_64 / macos-ARM64 branches only
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
) -> Result<(ort::session::builder::SessionBuilder, &'static str), String> {
    let build_result = std::panic::catch_unwind(|| build_coreml_ep().build());
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
#[allow(dead_code)] // call sites live in linux-x86_64-only branches
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
#[allow(dead_code)] // call sites live in linux-x86_64-only branches
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
        ort::ep::DirectML::default().with_device_id(id).build()
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
pub fn build_ort_session(
    onnx_path: &Path,
    role: OrtSessionRole,
) -> Result<(Session, String), String> {
    let choice = ep_choice_for_role(role);

    // Only an explicit "cpu" choice short-circuits here.  HNSEP resolves to
    // "cpu" by default via `ep_choice_for_role`, but a per-model override
    // (HIFISHIFTER_HNSEP_ORT_EP=coreml) must be able to reach the GPU path.
    if choice == "cpu" {
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
            Ok(builder) => match try_register_webgpu_ep(builder, role) {
                Ok((b, ep)) => {
                    let session = build_gpu_session_finalize(b, onnx_path, role, "WebGPU")?;
                    return Ok((session, ep.to_string()));
                }
                Err(e) => {
                    eprintln!("ort_session[{role:?}]: WebGPU unavailable — {e}");
                    log_vulkan_diagnostics();
                }
            },
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
    if choice == "auto"
        || choice == "coreml"
        || choice == "webgpu"
        || choice == "gpu"
        || choice == "directml"
    {
        // CoreML is the primary GPU path on Apple Silicon. An explicit
        // "webgpu" selection skips CoreML and goes straight to Dawn/Metal.
        //
        // The session deliberately keeps the model's dynamic `batch`/`time`
        // axes: ONNX Runtime 1.28's CoreML EP compiles the dynamic graph
        // without help, and pinning `time` to a constant benchmarked ~1.7x
        // SLOWER than the CPU EP (see the module docs).
        if choice != "webgpu" && !coreml_disabled() {
            match Session::builder() {
                Ok(builder) => match try_register_coreml_ep(builder) {
                    Ok((b, ep)) => {
                        match build_gpu_session_finalize(b, onnx_path, role, "CoreML") {
                            Ok(session) => return Ok((session, ep.to_string())),
                            Err(e) => eprintln!(
                                "ort_session[{role:?}]: CoreML session creation failed (will try WebGPU): {e}"
                            ),
                        }
                    }
                    Err(e) => eprintln!("ort_session[{role:?}]: CoreML unavailable: {e}"),
                },
                Err(e) => eprintln!(
                    "ort_session[{role:?}]: failed to create session builder for CoreML: {e}"
                ),
            }
        }
        match Session::builder() {
            Ok(builder) => match try_register_webgpu_ep(builder, role) {
                Ok((b, ep)) => match build_gpu_session_finalize(b, onnx_path, role, "WebGPU") {
                    Ok(session) => return Ok((session, ep.to_string())),
                    Err(e) => eprintln!(
                        "ort_session[{role:?}]: WebGPU session creation failed (will try CPU): {e}"
                    ),
                },
                Err(e) => eprintln!("ort_session[{role:?}]: WebGPU unavailable: {e}"),
            },
            Err(e) => {
                eprintln!("ort_session[{role:?}]: failed to create session builder for WebGPU: {e}")
            }
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
        // Replace dynamic dimensions (-1) with realistic test values:
        //   dim 0 -> 1 (batch),  other dims -> SMOKE_TEST_FRAMES.
        // Tiny values (e.g. 4) make ORT's buffer-reuse optimizer collide
        // with the model's fixed intermediate shapes ("{1,4,1} !=
        // {1,2048,1}"), so use the same frame length the app actually runs.
        let fallback_dim = SMOKE_TEST_FRAMES as i64;
        let test_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0 {
                    d as usize
                } else if i == 0 {
                    1
                } else {
                    fallback_dim as usize
                }
            })
            .collect();
        plans.push((input.name().to_string(), test_shape));
    }
    let mut input_pairs: Vec<(String, ort::value::Value)> = Vec::with_capacity(plans.len());
    for (name, test_shape) in plans {
        let total: usize = test_shape.iter().product::<usize>().max(1);
        // Use a non-zero f0: this model's f0 pre-processing computes a
        // differential that becomes the Pad "pads" input, and an all-zero f0
        // produces an empty/zero-sized tensor that crashes ORT's buffer
        // reuse ("{1,0,112}", "{1,4096,1} vs {1,4096,4096}").  440 Hz is a
        // realistic mid-range pitch and keeps every intermediate shape valid.
        let fill = if name == "f0" { 440.0f32 } else { 0.0f32 };
        let data: Vec<f32> = vec![fill; total];
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
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn build_gpu_session_finalize(
    mut builder: ort::session::builder::SessionBuilder,
    onnx_path: &Path,
    role: OrtSessionRole,
    ep_name: &str,
) -> Result<Session, String> {
    let model_name = onnx_path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    eprintln!(
        "ort_session[{role:?}]: model={model_name} ep={ep_name} (global_env={})",
        env_ep_choice(),
    );

    let coreml_session = ep_name == "CoreML";
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        // CoreML MLProgram sessions have their own execution queues.  ORT's
        // CPU memory-pattern/parallel-execution optimizers have been observed
        // to leave CoreML sessions stuck on repeated runs, so disable both for
        // CoreML and let Apple's runtime manage its buffers.
        .with_memory_pattern(!coreml_session)
        .map_err(|e| format!("set memory pattern failed: {e}"))?
        .with_parallel_execution(!coreml_session)
        .map_err(|e| format!("set parallel execution failed: {e}"))?;

    // CoreML keeps ORT's default (`0`): the reference benchmark of the
    // dynamic-shape CoreML session was taken with this setting and it is the
    // configuration the ~59x rtf was measured on.
    let threads = if coreml_session {
        0
    } else {
        cpu_intra_threads(role)
    };
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| format!("set intra op threads failed: {e}"))?;

    let t_create = std::time::Instant::now();
    let mut session = builder.commit_from_file(onnx_path).map_err(|e| {
        let msg = format!("load onnx into {ep_name} ort session failed: {e}");
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
            role,
            input.name(),
            input.dtype()
        );
    }
    for output in session.outputs() {
        eprintln!(
            "ort_session[{:?}]:   output name='{}' dtype={:?}",
            role,
            output.name(),
            output.dtype()
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
    let builder =
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
            role,
            input.name(),
            input.dtype()
        );
    }
    for output in session.outputs() {
        eprintln!(
            "ort_session[{:?}]:   output name='{}' dtype={:?}",
            role,
            output.name(),
            output.dtype()
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

/// Intra-op thread budget for sessions whose ops run on the CPU (the whole
/// graph for a CPU session, or the unsupported ops of a GPU session).
///
/// Apple Silicon uses every core: measured on an 8-core M-series with 1024 mel
/// frames, `intra=cores/2` took 1586 ms vs `intra=cores` at 1202 ms (−24%).
/// Apple's performance and efficiency cores are homogeneous enough that using
/// all of them is a straight win.
///
/// Windows and Linux keep `cores/2`: Intel hybrid CPUs mix P-cores with much
/// slower E-cores and many workstations are multi-socket/NUMA, so pinning
/// intra-op work to every logical core can oversubscribe the slow cores and
/// regress. The conservative half-core budget stays there until it is measured
/// on those topologies.
fn cpu_intra_threads(role: OrtSessionRole) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2);
    match role {
        OrtSessionRole::Separator => cores,
        OrtSessionRole::Vocoder | OrtSessionRole::PitchDetector => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                cores
            } else {
                (cores / 2).max(2)
            }
        }
    }
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
        onnx_path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default(),
        env_ep_choice(),
    );

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("set graph optimization level failed: {e}"))?
        .with_memory_pattern(true)
        .map_err(|e| format!("set memory pattern failed: {e}"))?;

    let threads = cpu_intra_threads(role);
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
/// NOTE: WebGPU probing is performed on Linux x86_64 and macOS ARM64,
/// where Dawn uses the Vulkan/Metal backend which is safe to probe.
/// On Windows, Dawn uses D3D12 and probing can trigger native crashes on
/// some GPU/driver combos; WebGPU on Windows is only used when the user
/// explicitly selects it.
pub fn diagnose_available_providers() -> Vec<String> {
    let mut providers = vec!["CPUExecutionProvider".to_string()];

    // WebGPU — compiled on Linux x86_64 / macOS ARM64 only.
    // Excluded: Windows (Dawn/D3D12 crash risk), Linux ARM64 (no prebuilt binary),
    //           WSL2 (Vulkan not available; Dawn init hangs at shutdown).
    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    ))]
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
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn probe_webgpu_ep_available() -> bool {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match Session::builder() {
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
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match Session::builder() {
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
