//! 插件资源位置：先于ARA建图/分析登记原FCPE及声码器模型，不使用REAPER的resource_dir。
use std::path::PathBuf;
/// 模型只登记一次，不初始化推理设备，不改变用户文件或系统路径。
pub(crate) fn initialize_models() {
    // FCPE自身还会异步预热；其迟到线程与WebView回调一样不能进入已卸载的DLL。
    #[cfg(all(windows, not(test)))]
    if let Err(error) = pin_analysis_code() {
        crate::log_line(&format!("Plugin module pin failed: {error}"));
        return;
    }
    let directory = model_directory();
    if let Ok(directory) = directory {
        let fcpe = directory.join("fcpe/fcpe.onnx");
        if fcpe.is_file() {
            hifishifter_kernel::model_paths::set_fcpe_onnx_path(fcpe);
        } else {
            crate::log_line("Plugin FCPE model missing; pitch analysis is unavailable");
        }
        let nsf = directory.join("nsf_hifigan");
        if nsf.join("pc_nsf_hifigan.onnx").is_file() {
            hifishifter_kernel::model_paths::set_nsf_hifigan_model_dir(nsf);
        }
        let hnsep = directory.join("hnsep");
        if hnsep.join("hnsep.onnx").is_file() {
            hifishifter_kernel::model_paths::set_hnsep_model_dir(hnsep);
        }
    }
}
/// 固定本引擎到REAPER退出；不保留UI/实例worker，不支持运行中热更新。
#[cfg(all(windows, not(test)))]
fn pin_analysis_code() -> Result<(), String> {
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::*;
    static PIN: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    PIN.get_or_init(|| {
        let mut module = HMODULE::default();
        unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                PCWSTR(initialize_models as *const () as *const u16),
                &mut module,
            )
        }
        .map_err(|e| e.to_string())
    })
    .clone()
}
fn model_directory() -> Result<PathBuf, String> {
    #[cfg(test)]
    {
        return Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src-tauri/resources/models"));
    }
    #[cfg(all(windows, not(test)))]
    {
        use windows::core::PCWSTR;
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::System::LibraryLoader::*;
        let mut module = HMODULE::default();
        unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                PCWSTR(initialize_models as *const () as *const u16),
                &mut module,
            )
        }
        .map_err(|e| e.to_string())?;
        let mut name = vec![0_u16; 32768];
        let count = unsafe { GetModuleFileNameW(Some(module), &mut name) } as usize;
        if count == 0 || count >= name.len() {
            return Err("plugin module path unavailable".into());
        }
        let path = PathBuf::from(String::from_utf16_lossy(&name[..count]));
        Ok(path
            .parent()
            .and_then(|p| p.parent())
            .ok_or("VST3 Contents unavailable")?
            .join("Resources/models"))
    }
    #[cfg(all(not(windows), not(test)))]
    {
        Err("embedded plugin model paths are Windows-only".into())
    }
}
