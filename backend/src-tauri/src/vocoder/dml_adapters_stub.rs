//! Stub: DirectML adapter enumeration is only available on Windows.
//! On other platforms, returns an empty list.

use serde::Serialize;

/// Information about a single DXGI GPU adapter.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DmlAdapterInfo {
    pub device_id: u32,
    pub name: String,
    pub dedicated_video_memory_mb: u64,
    pub shared_system_memory_mb: u64,
    pub vendor_id: u32,
    pub device_id_pci: u32,
}

/// All DXGI GPU adapters discovered in the system.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DmlAdapterList {
    pub adapters: Vec<DmlAdapterInfo>,
    pub note: Option<String>,
}

/// Enumerate all DXGI GPU adapters (stub — Windows only).
pub fn enumerate_dml_adapters() -> DmlAdapterList {
    DmlAdapterList {
        adapters: vec![],
        note: Some("DirectML adapter enumeration is only available on Windows.".to_string()),
    }
}
