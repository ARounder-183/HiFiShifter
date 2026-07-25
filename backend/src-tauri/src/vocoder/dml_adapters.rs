//! DirectML adapter enumeration via DXGI.
//!
//! DirectML uses DXGI adapter indices as device IDs. This module enumerates
//! all DXGI adapters (GPU, not NPU) that DirectML can use for hardware
//! acceleration. Unlike NVML (NVIDIA-only), DXGI covers all GPU vendors:
//! NVIDIA, AMD, Intel Arc, and integrated GPUs.
//!
//! The adapter index from DXGI matches the `device_id` parameter accepted
//! by `ort::ep::DirectML::with_device_id()` on standard WDDM configurations.

use serde::Serialize;
use windows::core::Interface;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_DESC1,
};

/// Information about a single DXGI GPU adapter.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DmlAdapterInfo {
    /// DXGI adapter index (= DirectML device_id).
    pub device_id: u32,
    /// Human-readable GPU description, e.g. "NVIDIA GeForce RTX 3050 Laptop GPU".
    pub name: String,
    /// Dedicated video memory in megabytes.
    pub dedicated_video_memory_mb: u64,
    /// Shared system memory in megabytes.
    pub shared_system_memory_mb: u64,
    /// PCI vendor ID (e.g. 0x10DE for NVIDIA).
    pub vendor_id: u32,
    /// PCI device ID.
    pub device_id_pci: u32,
}

/// All DXGI GPU adapters discovered in the system.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DmlAdapterList {
    pub adapters: Vec<DmlAdapterInfo>,
    /// Human-readable note, set on failure.
    pub note: Option<String>,
}

/// Enumerate all DXGI GPU adapters.
///
/// Returns an empty list (not an error) if DXGI enumeration fails.
/// DirectML device IDs match the returned `device_id` values (0-based index).
pub fn enumerate_dml_adapters() -> DmlAdapterList {
    match try_enumerate_dxgi() {
        Ok(adapters) => {
            let note = if adapters.is_empty() {
                Some("No DXGI GPU adapters found.".to_string())
            } else {
                None
            };
            DmlAdapterList { adapters, note }
        }
        Err(e) => DmlAdapterList {
            adapters: vec![],
            note: Some(format!("DXGI enumeration failed: {e}")),
        },
    }
}

fn try_enumerate_dxgi() -> Result<Vec<DmlAdapterInfo>, String> {
    // Create DXGI factory — this is always available on Windows 8+
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }
        .map_err(|e| format!("CreateDXGIFactory1 failed: {e}"))?;

    let mut adapters = Vec::new();
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut index: u32 = 0;

    loop {
        let adapter_result = unsafe { factory.EnumAdapters1(index) };
        match adapter_result {
            Ok(a) => {
                let desc: DXGI_ADAPTER_DESC1 = unsafe { a.GetDesc1() }
                    .map_err(|e| format!("GetDesc1 failed for adapter {index}: {e}"))?;

                let name = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_string();

                // Skip software/Microsoft Basic Render Driver adapters
                let is_software = desc.VendorId == 0x1414;
                // Deduplicate: same physical GPU can appear multiple times in DXGI
                let key = (desc.VendorId, desc.DeviceId);
                let is_duplicate = !seen.insert(key);

                if !is_software && !name.is_empty() && !is_duplicate {
                    adapters.push(DmlAdapterInfo {
                        device_id: index,
                        name,
                        dedicated_video_memory_mb: (desc.DedicatedVideoMemory as u64) / (1024 * 1024),
                        shared_system_memory_mb: (desc.SharedSystemMemory as u64) / (1024 * 1024),
                        vendor_id: desc.VendorId,
                        device_id_pci: desc.DeviceId,
                    });
                }

                index += 1;
            }
            Err(e) => {
                // DXGI_ERROR_NOT_FOUND (0x887A0002) = end of enumeration
                if e.code().0 as u32 == 0x887A0002 {
                    break;
                }
                eprintln!("dml_adapters: EnumAdapters1({index}) failed: {e:?}");
                break;
            }
        }
    }

    // Sort by VRAM descending so the dGPU (most VRAM) is listed first in UI.
    // IMPORTANT: device_id retains the ACTUAL DXGI adapter index regardless of sort.
    adapters.sort_by(|a, b| {
        b.dedicated_video_memory_mb
            .cmp(&a.dedicated_video_memory_mb)
    });

    // After sorting, first adapter = recommended (most VRAM = dGPU).
    // The device_id field still holds the true DXGI index for DirectML.

    eprintln!(
        "dml_adapters: enumerated {} unique DXGI GPU adapters ({} total indices scanned)",
        adapters.len(),
        index
    );
    Ok(adapters)
}
