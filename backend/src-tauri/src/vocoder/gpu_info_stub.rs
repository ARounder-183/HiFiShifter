//! Stub for non-Windows platforms where NVML is not available.
//! Always returns an empty device list.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub memory_mb: u64,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub cuda_capable: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuEnumerationResult {
    pub devices: Vec<GpuDeviceInfo>,
    pub note: Option<String>,
}

pub fn enumerate_gpus() -> GpuEnumerationResult {
    GpuEnumerationResult {
        devices: vec![],
        note: Some("GPU enumeration via NVML is only available on Windows".to_string()),
    }
}
