//! GPU enumeration via NVML (NVIDIA Management Library).
//!
//! Dynamically loads `nvml.dll` to discover all NVIDIA GPUs, their names,
//! memory sizes, and compute capabilities. NVML device indices match
//! GPU device indices on standard Windows/WDDM configurations (which
//! covers all laptop and most desktop setups).
//!
//! This is used by the frontend to let users select which GPU to use for
//! ONNX Runtime GPU inference (DirectML, WebGPU, etc.), and by the benchmark
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
type NvmlDeviceGetCudaComputeCapabilityFn = unsafe extern "C" fn(
    handle: *mut std::ffi::c_void,
    major: *mut i32,
    minor: *mut i32,
) -> NvmlReturn;

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

/// 载入 NVML 动态库。
///
/// 【为什么不能用裸文件名】libloading 的 `Library::new("nvml.dll")` 走的是操作系统
/// 默认的 DLL 搜索顺序：进程当前工作目录与 `PATH` 里的目录都排在 System32 之前。
/// 用户从下载目录、工程目录或 U 盘启动本程序时，同目录下任何一个同名 `nvml.dll`
/// 都会被当作 NVIDIA 驱动加载并执行其 `DllMain` —— 典型的 DLL 预加载/劫持路径
/// （CWE-426，以及 CWE-427 的不受控搜索路径）。NVML 只可能来自驱动安装的固定位置，
/// 因此这里只接受**绝对路径**，一处都没有就按"没有 NVIDIA 驱动"降级。
fn load_nvml() -> Result<libloading::Library, String> {
    let candidates = nvml_candidates();
    let mut last_error: Option<String> = None;
    for path in &candidates {
        if !path.is_file() {
            continue;
        }
        match unsafe { libloading::Library::new(path) } {
            Ok(library) => return Ok(library),
            Err(error) => last_error = Some(format!("{}: {error}", path.display())),
        }
    }
    Err(match last_error {
        Some(error) => format!("Cannot load NVML: {error}"),
        None => format!(
            "Cannot load NVML: not found in {}",
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

/// NVML 的候选**绝对路径**（驱动安装器写下的固定位置）。
///
/// 绝不包含相对路径或裸文件名：`load_nvml` 的整个安全前提就是"只加载驱动自己
/// 装下的那一个文件"。
fn nvml_candidates() -> Vec<std::path::PathBuf> {
    #[cfg(windows)]
    {
        let mut candidates = Vec::new();
        if let Some(root) = std::env::var_os("SystemRoot") {
            candidates.push(
                std::path::PathBuf::from(root)
                    .join("System32")
                    .join("nvml.dll"),
            );
        }
        if let Some(program_files) = std::env::var_os("ProgramFiles") {
            candidates.push(
                std::path::PathBuf::from(program_files)
                    .join("NVIDIA Corporation")
                    .join("NVSMI")
                    .join("nvml.dll"),
            );
        }
        candidates
    }
    #[cfg(target_os = "linux")]
    {
        [
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
        ]
        .into_iter()
        .map(std::path::PathBuf::from)
        .collect()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Vec::new()
    }
}

fn try_enumerate() -> Result<Vec<GpuDeviceInfo>, String> {
    // 1. Load NVML from an absolute, driver-installed path (see `load_nvml`).
    let nvml = load_nvml()?;

    // Safety helper: get a function pointer or return error
    // `.into_raw()` returns `*mut T`; we dereference with `*` to get the fn value.
    macro_rules! get_fn {
        ($lib:expr, $name:expr, $t:ty) => {{
            unsafe {
                *$lib
                    .get::<$t>($name.as_bytes())
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
    let nvml_device_get_handle_by_index: NvmlDeviceGetHandleByIndexFn = get_fn!(
        nvml,
        "nvmlDeviceGetHandleByIndex_v2",
        NvmlDeviceGetHandleByIndexFn
    );
    let nvml_device_get_name: NvmlDeviceGetNameFn =
        get_fn!(nvml, "nvmlDeviceGetName", NvmlDeviceGetNameFn);
    let nvml_device_get_memory_info: NvmlDeviceGetMemoryInfoFn =
        get_fn!(nvml, "nvmlDeviceGetMemoryInfo", NvmlDeviceGetMemoryInfoFn);
    let nvml_device_get_cuda_cc: NvmlDeviceGetCudaComputeCapabilityFn = get_fn!(
        nvml,
        "nvmlDeviceGetCudaComputeCapability",
        NvmlDeviceGetCudaComputeCapabilityFn
    );

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
            unsafe {
                (self.shutdown)();
            }
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
            let cstr = CStr::from_bytes_until_nul(&name_buf).unwrap_or(c"Unknown");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **不变量：NVML 的候选路径必须全部是绝对路径。**
    ///
    /// 这条测试守的是"动态库搜索路径劫持"的回归点：任何人把裸文件名
    /// （`nvml.dll`）或相对路径塞回候选表，`is_absolute()` 会立刻失败。
    #[test]
    fn nvml_candidates_are_always_absolute() {
        for candidate in nvml_candidates() {
            assert!(
                candidate.is_absolute(),
                "NVML candidate must be absolute, got {}",
                candidate.display()
            );
        }
    }

    /// 驱动不存在时必须返回可读错误（列出搜过的路径），而不是 panic 或静默成功。
    #[test]
    fn a_missing_driver_reports_where_it_looked() {
        match load_nvml() {
            // 机器上确实装了 NVIDIA 驱动：这条断言无从验证，跳过。
            Ok(_) => {}
            Err(error) => assert!(error.starts_with("Cannot load NVML"), "{error}"),
        }
    }
}
