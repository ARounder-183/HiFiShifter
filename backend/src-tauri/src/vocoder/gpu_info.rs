//! GPU enumeration via NVML (NVIDIA Management Library).
//!
//! Dynamically loads `nvml.dll` to discover all NVIDIA GPUs, their names,
//! memory sizes, and compute capabilities. NVML device indices match
//! GPU device indices on standard Windows/WDDM configurations (which
//! covers all laptop and most desktop setups).
//!
//! This is used by the frontend to let users select which GPU to use for
//! ONNX Runtime GPU inference (DirectML, OpenCL, etc.), and by the benchmark
//! to report which physical GPU participated in the test.

use serde::Serialize;
use std::ffi::CStr;

// ── NVML type aliases ──────────────────────────────────────────────────────

type NvmlReturn = i32;
const NVML_SUCCESS: NvmlReturn = 0;
const NVML_DEVICE_NAME_BUFFER_SIZE: u32 = 96;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

// NVML function pointer types
type NvmlInitFn = unsafe extern "C" fn() -> NvmlReturn;
type NvmlShutdownFn = unsafe extern "C" fn() -> NvmlReturn;
type NvmlDeviceGetCountFn = unsafe extern "C" fn(count: *mut u32) -> NvmlReturn;
type NvmlDeviceGetHandleByIndexFn =
    unsafe extern "C" fn(index: u32, handle: *mut *mut std::ffi::c_void) -> NvmlReturn;
type NvmlDeviceGetNameFn =
    unsafe extern "C" fn(handle: *mut std::ffi::c_void, name: *mut u8, length: u32) -> NvmlReturn;
type NvmlDeviceGetMemoryInfoFn =
    unsafe extern "C" fn(handle: *mut std::ffi::c_void, memory: *mut NvmlMemory) -> NvmlReturn;
type NvmlDeviceGetCudaComputeCapabilityFn =
    unsafe extern "C" fn(handle: *mut std::ffi::c_void, major: *mut i32, minor: *mut i32) -> NvmlReturn;

// ── Public types ────────────────────────────────────────────────────────────

/// Information about a single NVIDIA GPU discovered via NVML.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuDeviceInfo {
    /// NVML device index (matches GPU device ID on standard configs).
    pub device_id: u32,
    /// Human-readable GPU name, e.g. "NVIDIA GeForce RTX 3050 Laptop GPU".
    pub name: String,
    /// Total GPU memory in megabytes.
    pub memory_mb: u64,
    /// Compute capability major version (e.g. 8 for Ampere).
    pub compute_major: i32,
    /// Compute capability minor version (e.g. 6 for RTX 3050: 8.6).
    pub compute_minor: i32,
    /// Whether this device supports GPU compute (based on NVML compute capability query).
    pub cuda_capable: bool,
}

/// All NVIDIA GPUs discovered in the system.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuEnumerationResult {
    pub devices: Vec<GpuDeviceInfo>,
    /// Simple user-facing diagnostic message.
    pub note: Option<String>,
}

// ── Implementation ──────────────────────────────────────────────────────────

/// Enumerate all NVIDIA GPUs via NVML. Returns an empty list (not an error)
/// if NVML is not available (e.g., no NVIDIA driver installed).
pub fn enumerate_gpus() -> GpuEnumerationResult {
    match try_enumerate() {
        Ok(devices) => {
            let note = if devices.is_empty() {
                Some("No NVIDIA GPUs found via NVML. Is the NVIDIA driver installed?".to_string())
            } else {
                None
            };
            GpuEnumerationResult { devices, note }
        }
        Err(e) => GpuEnumerationResult {
            devices: vec![],
            note: Some(format!("NVML enumeration failed: {e}")),
        },
    }
}

fn try_enumerate() -> Result<Vec<GpuDeviceInfo>, String> {
    // 1. Load nvml.dll (always present when NVIDIA driver is installed)
    let nvml = unsafe {
        libloading::Library::new("nvml.dll")
            .map_err(|e| format!("Cannot load nvml.dll: {e} — is NVIDIA driver installed?"))?
    };

    // Safety helper: get a function pointer or return error
    // `.into_raw()` returns `*mut T`; we dereference with `*` to get the fn value.
    macro_rules! get_fn {
        ($lib:expr, $name:expr, $t:ty) => {{
            unsafe {
                *$lib.get::<$t>($name.as_bytes())
                    .map_err(|_| format!("nvml.dll missing symbol: {}", $name))?
                    .into_raw()
            }
        }};
    }

    // 2. Get function pointers
    let nvml_init: NvmlInitFn = get_fn!(nvml, "nvmlInit_v2", NvmlInitFn);
    let nvml_shutdown: NvmlShutdownFn = get_fn!(nvml, "nvmlShutdown", NvmlShutdownFn);
    let nvml_device_get_count: NvmlDeviceGetCountFn =
        get_fn!(nvml, "nvmlDeviceGetCount_v2", NvmlDeviceGetCountFn);
    let nvml_device_get_handle_by_index: NvmlDeviceGetHandleByIndexFn =
        get_fn!(nvml, "nvmlDeviceGetHandleByIndex_v2", NvmlDeviceGetHandleByIndexFn);
    let nvml_device_get_name: NvmlDeviceGetNameFn =
        get_fn!(nvml, "nvmlDeviceGetName", NvmlDeviceGetNameFn);
    let nvml_device_get_memory_info: NvmlDeviceGetMemoryInfoFn =
        get_fn!(nvml, "nvmlDeviceGetMemoryInfo", NvmlDeviceGetMemoryInfoFn);
    let nvml_device_get_cuda_cc: NvmlDeviceGetCudaComputeCapabilityFn =
        get_fn!(nvml, "nvmlDeviceGetCudaComputeCapability", NvmlDeviceGetCudaComputeCapabilityFn);

    // 3. Initialize NVML (wrapped in RAII guard for cleanup)
    let ret = unsafe { nvml_init() };
    if ret != NVML_SUCCESS {
        return Err(format!("nvmlInit_v2 failed with code {ret}"));
    }

    struct NvmlGuard {
        shutdown: NvmlShutdownFn,
        _lib: libloading::Library, // keep library alive
    }
    impl Drop for NvmlGuard {
        fn drop(&mut self) {
            unsafe { (self.shutdown)(); }
        }
    }
    let _guard = NvmlGuard {
        shutdown: nvml_shutdown,
        _lib: nvml,
    };

    // 4. Get device count
    let mut device_count: u32 = 0;
    let ret = unsafe { nvml_device_get_count(&mut device_count) };
    if ret != NVML_SUCCESS {
        return Err(format!("nvmlDeviceGetCount_v2 failed with code {ret}"));
    }

    // 5. Enumerate each device
    let mut devices = Vec::with_capacity(device_count as usize);
    for idx in 0..device_count {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let ret = unsafe { nvml_device_get_handle_by_index(idx, &mut handle) };
        if ret != NVML_SUCCESS {
            continue; // Skip device we can't open
        }

        // Get GPU name
        let mut name_buf = [0u8; NVML_DEVICE_NAME_BUFFER_SIZE as usize];
        let ret = unsafe {
            nvml_device_get_name(handle, name_buf.as_mut_ptr(), NVML_DEVICE_NAME_BUFFER_SIZE)
        };
        let name = if ret == NVML_SUCCESS {
            let cstr = CStr::from_bytes_until_nul(&name_buf)
                .unwrap_or(CStr::from_bytes_with_nul(b"Unknown\0").unwrap());
            cstr.to_str().unwrap_or("Unknown").to_string()
        } else {
            "Unknown".to_string()
        };

        // Get memory info
        let mut mem = NvmlMemory::default();
        let _ = unsafe { nvml_device_get_memory_info(handle, &mut mem) };
        let memory_mb = mem.total / (1024 * 1024);

        // Get compute capability
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        let cc_ret = unsafe { nvml_device_get_cuda_cc(handle, &mut major, &mut minor) };
        let cuda_capable = cc_ret == NVML_SUCCESS;

        devices.push(GpuDeviceInfo {
            device_id: idx,
            name,
            memory_mb,
            compute_major: major,
            compute_minor: minor,
            cuda_capable,
        });
    }

    Ok(devices)
}
