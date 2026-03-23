//! VST2/VST3 插件加载与卸载逻辑。
//!
//! 提供 `load_vst2` 和 `load_vst3` 函数，分别加载 VST2 `.dll` 和 VST3 `.vst3` 插件，
//! 返回统一的 `VstPluginInstance` 封装。

use std::path::Path;
use std::sync::{Arc, Mutex};

use super::plugin_instance::{VstPluginBackend, VstPluginInstance};
use super::VstFormat;

// ─── VST2 加载 ──────────────────────────────────────────────────────────────

/// 加载 VST2 插件（`.dll` / `.so` / `.vst`）。
pub fn load_vst2(
    path: &Path,
    sample_rate: f32,
    block_size: i64,
) -> Result<Arc<Mutex<VstPluginInstance>>, String> {
    #[cfg(feature = "vst")]
    {
        use super::plugin_instance::SimpleVst2Host;
        use vst2::host::PluginLoader;

        let host = Arc::new(Mutex::new(SimpleVst2Host));

        let mut loader = PluginLoader::load(path, host.clone())
            .map_err(|e| format!("VST2 load failed: {}", e))?;

        let mut plugin = loader
            .instance()
            .map_err(|e| format!("VST2 instance failed: {}", e))?;

        let info = plugin.get_info();
        plugin.init();
        plugin.set_sample_rate(sample_rate);
        plugin.set_block_size(block_size);
        plugin.resume();

        let instance = VstPluginInstance {
            name: info.name.clone(),
            vendor: info.vendor.clone(),
            format: VstFormat::Vst2,
            path: path.to_path_buf(),
            sample_rate,
            block_size: block_size as usize,
            num_inputs: info.inputs as u32,
            num_outputs: info.outputs as u32,
            bypassed: false,
            backend: VstPluginBackend::Vst2 {
                plugin: Box::new(plugin),
                _host: host,
            },
        };

        Ok(Arc::new(Mutex::new(instance)))
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (path, sample_rate, block_size);
        Err("VST feature is not enabled".to_string())
    }
}

// ─── VST3 加载 ──────────────────────────────────────────────────────────────

/// 加载 VST3 插件（`.vst3` 模块）。
///
/// 使用 `libloading` 加载共享库，通过 COM 接口初始化。
pub fn load_vst3(
    path: &Path,
    sample_rate: f64,
    block_size: i32,
) -> Result<Arc<Mutex<VstPluginInstance>>, String> {
    #[cfg(feature = "vst")]
    {
        let module_path = resolve_vst3_module_path(path)?;

        let lib = unsafe { libloading::Library::new(&module_path) }
            .map_err(|e| format!("VST3 load library failed: {}", e))?;

        type GetFactoryFn = unsafe extern "system" fn() -> *mut std::ffi::c_void;
        let get_factory: libloading::Symbol<GetFactoryFn> = unsafe {
            lib.get(b"GetPluginFactory\0")
                .map_err(|e| format!("VST3 GetPluginFactory not found: {}", e))?
        };

        let factory_ptr = unsafe { get_factory() };
        if factory_ptr.is_null() {
            return Err("VST3 GetPluginFactory returned null".to_string());
        }

        let instance = VstPluginInstance {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown VST3")
                .to_string(),
            vendor: String::new(),
            format: VstFormat::Vst3,
            path: path.to_path_buf(),
            sample_rate: sample_rate as f32,
            block_size: block_size as usize,
            num_inputs: 2,
            num_outputs: 2,
            bypassed: false,
            backend: VstPluginBackend::Vst3 {
                _lib: lib,
                factory_ptr,
                component_ptr: std::ptr::null_mut(),
                processor_ptr: std::ptr::null_mut(),
                initialized: false,
            },
        };

        Ok(Arc::new(Mutex::new(instance)))
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (path, sample_rate, block_size);
        Err("VST feature is not enabled".to_string())
    }
}

/// 解析 `.vst3` bundle 内部的实际共享库路径。
#[cfg(feature = "vst")]
fn resolve_vst3_module_path(bundle_path: &Path) -> Result<std::path::PathBuf, String> {
    if bundle_path.is_file() {
        return Ok(bundle_path.to_path_buf());
    }

    let name = bundle_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("plugin");

    #[cfg(target_os = "windows")]
    {
        let p = bundle_path
            .join("Contents")
            .join("x86_64-win")
            .join(format!("{}.vst3", name));
        if p.exists() {
            return Ok(p);
        }
        let p2 = bundle_path.join(format!("{}.vst3", name));
        if p2.exists() {
            return Ok(p2);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let p = bundle_path.join("Contents").join("MacOS").join(name);
        if p.exists() {
            return Ok(p);
        }
    }

    #[cfg(target_os = "linux")]
    {
        let p = bundle_path
            .join("Contents")
            .join("x86_64-linux")
            .join(format!("{}.so", name));
        if p.exists() {
            return Ok(p);
        }
    }

    Err(format!(
        "VST3 module not found in bundle: {}",
        bundle_path.display()
    ))
}
