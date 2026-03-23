//! VST2/VST3 插件加载与卸载逻辑。
//!
//! 提供 `load_vst2` 和 `load_vst3` 函数，分别加载 VST2 `.dll` 和 VST3 `.vst3` 插件，
//! 返回统一的 `VstPluginInstance` 封装。

use std::path::Path;
use std::sync::{Arc, Mutex};

use super::plugin_instance::{VstPluginBackend, VstPluginInstance};
use super::VstFormat;
#[cfg(feature = "vst")]
use vst2::plugin::Plugin;

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
/// 使用 `vst3` crate 的类型安全 COM 接口完成完整初始化：
/// 1. 加载 DLL → GetPluginFactory → IPluginFactory
/// 2. getClassInfo → 找到音频处理器组件
/// 3. createInstance → IComponent
/// 4. IComponent::initialize → IAudioProcessor::setupProcessing
/// 5. IComponent::setActive(true) → IAudioProcessor::setProcessing(true)
/// 6. 尝试获取 IEditController（参数管理和 GUI）
pub fn load_vst3(
    path: &Path,
    sample_rate: f64,
    block_size: i32,
) -> Result<Arc<Mutex<VstPluginInstance>>, String> {
    #[cfg(feature = "vst")]
    {
        let module_path = resolve_vst3_module_path(path)?;

        let (vst3_instance, plugin_name, plugin_vendor, num_inputs, num_outputs) =
            super::vst3_com::Vst3Instance::load(&module_path, sample_rate, block_size)?;

        let name = if plugin_name.is_empty() {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown VST3")
                .to_string()
        } else {
            plugin_name
        };

        let instance = VstPluginInstance {
            name,
            vendor: plugin_vendor,
            format: VstFormat::Vst3,
            path: path.to_path_buf(),
            sample_rate: sample_rate as f32,
            block_size: block_size as usize,
            num_inputs,
            num_outputs,
            bypassed: false,
            backend: VstPluginBackend::Vst3 {
                instance: vst3_instance,
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
///
/// 对 Windows/macOS/Linux 不同的 bundle 结构进行路径解析。
/// 如果输入路径本身是文件（单文件 .vst3），直接返回。
#[cfg(feature = "vst")]
pub(super) fn resolve_vst3_module_path(bundle_path: &Path) -> Result<std::path::PathBuf, String> {
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
