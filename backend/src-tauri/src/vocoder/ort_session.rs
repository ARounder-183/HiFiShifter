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
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtSessionRole {
    /// NSF-HiFiGAN vocoder — full GPU budget, aggressive optimizations.
    Vocoder,
    /// FCPE pitch analysis — smaller model, share GPU with vocoder.
    PitchDetector,
    /// HNSEP harmonic+noise separation — medium model, share GPU with vocoder.
    Separator,
}

fn env_ep_choice() -> String {
    std::env::var("HIFISHIFTER_ORT_EP")
        .ok()
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_ascii_lowercase()
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
    let device_id = env_i32("HIFISHIFTER_ORT_CUDA_DEVICE_ID").unwrap_or(0);
    let arena_bytes = env_cuda_mem_limit_bytes();

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
    let device_id = env_i32("HIFISHIFTER_ORT_CUDA_DEVICE_ID").unwrap_or(0);
    let arena_bytes = env_cuda_mem_limit_bytes();

    let mut ep = ep::TensorRT::default()
        .with_device_id(device_id)
        .with_max_workspace_size(arena_bytes as usize)
        .with_fp16(true)
        .with_engine_cache(true)
        .with_engine_cache_path("/tmp/hifishifter/trt_engine_cache")
        .with_timing_cache(true)
        .with_timing_cache_path("/tmp/hifishifter/trt_timing_cache")
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

    let choice = env_ep_choice();
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
        "ort_session: role={role:?} ep={selected} (HIFISHIFTER_ORT_EP={choice:?}, cuda_mem_limit={}MB)",
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

/// RAII guard that temporarily overrides the `HIFISHIFTER_ORT_EP` env var for
/// the duration of a benchmark session, restoring the previous value on drop.
///
/// Used by `run_benchmark()` so it can force a specific EP (e.g. "cpu" or
/// "cuda") without permanently changing the process-wide setting.
pub struct EpOverrideGuard {
    prev: Option<String>,
}

impl EpOverrideGuard {
    pub fn new(ep: String) -> Self {
        let prev = std::env::var("HIFISHIFTER_ORT_EP").ok();
        // Safety: benchmark serialises session creation; env mutation is
        // single-threaded at the point this guard is held.
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("HIFISHIFTER_ORT_EP", &ep);
        }
        Self { prev }
    }
}

impl Drop for EpOverrideGuard {
    fn drop(&mut self) {
        #[allow(unused_unsafe)]
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var("HIFISHIFTER_ORT_EP", v),
                None => std::env::remove_var("HIFISHIFTER_ORT_EP"),
            }
        }
    }
}
