//! 插件资源位置：先于ARA建图/分析登记原FCPE及声码器模型，不使用REAPER的resource_dir。
//!
//! 解析本身交给内核的共享模型库（`hifishifter_kernel::model_store`）：它优先用与独立
//! App 共用的那一份，库里没有就从本 bundle 建立（同卷硬链接）。插件这边只负责回答
//! "我的 bundle 在哪"。
use std::path::PathBuf;
/// 模型只登记一次，不初始化推理设备，不改变用户文件或系统路径。
pub(crate) fn initialize_models() {
    // FCPE自身还会异步预热；其迟到线程与WebView回调一样不能进入已卸载的DLL。
    #[cfg(all(windows, not(test)))]
    if let Err(error) = pin_analysis_code() {
        crate::log_line(&format!("Plugin module pin failed: {error}"));
        return;
    }
    let Ok(directory) = model_directory() else {
        crate::log_line(
            "Plugin model directory unavailable; pitch analysis and vocoder are unavailable",
        );
        return;
    };
    match hifishifter_kernel::model_store::resolve_and_register(&directory) {
        Ok(origin) => crate::log_line(&format!(
            "models: {} ({:?}{})",
            origin.dir.display(),
            origin.source,
            if origin.published {
                ", just published"
            } else {
                ""
            }
        )),
        // 模型不可用不该让插件窗口起不来：编辑与界面照常，推理相关的操作各自报错。
        Err(error) => crate::log_line(&format!("models unavailable: {error}")),
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
        Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src-tauri/resources/models"))
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
    // 【Unix】用 dladdr 反查本模块自己的路径：宿主加载的是
    // `<bundle>/Contents/<arch>/HiFiShifter.so`（macOS 是 `Contents/MacOS/HiFiShifter`），
    // 往上两级就是 `Contents` —— 与 Windows 的 `Contents\Resources\models` 同构，
    // 因此模型清单与 `model_store` 一个字都不用改。
    //
    // 【为什么不是 `std::env::current_exe`】插件跑在宿主进程里，那是**宿主**的路径。
    // 【为什么不用 `/proc/self/maps`】只有 Linux 有；dladdr 两个平台都有。
    #[cfg(all(unix, not(test)))]
    {
        use std::ffi::CStr;
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        // SAFETY: dladdr 只写它自己的 out 参数；地址取自本 crate 内的函数，
        // 必然落在本模块的映射区间里。
        let found = unsafe {
            libc::dladdr(
                initialize_models as *const () as *const libc::c_void,
                info.as_mut_ptr(),
            )
        };
        if found == 0 {
            return Err("plugin module path unavailable".into());
        }
        // SAFETY: found != 0 时 dladdr 已把 info 初始化完毕。
        let info = unsafe { info.assume_init() };
        if info.dli_fname.is_null() {
            return Err("plugin module path unavailable".into());
        }
        // SAFETY: dli_fname 指向 dladdr 内部的 NUL 结尾字符串。
        let module = PathBuf::from(
            unsafe { CStr::from_ptr(info.dli_fname) }
                .to_string_lossy()
                .into_owned(),
        );
        Ok(module
            .parent()
            .and_then(|p| p.parent())
            .ok_or("VST3 Contents unavailable")?
            .join("Resources/models"))
    }
}

/// 前端产物目录（`Contents/Resources/frontend`）。
///
/// 【为什么与模型目录同构】三个平台的"资源在哪"只该有一个答案：
/// `<bundle>/Contents/Resources/<名字>`。Windows 用 `GetModuleFileNameW`、Unix 用
/// `dladdr` 找到本模块，再往上两级 —— 于是 `assets()` 的实现也只有一份。
///
/// 【`HIFISHIFTER_PLUGIN_EDITOR_ASSETS` 覆盖】开发时前端产物在 `frontend/dist`，
/// 与 bundle 的相对位置无关。这个变量**只接受绝对路径**：相对路径会随宿主的 CWD
/// 变化，而那正是"开发机能开、用户机白屏"的经典成因。
pub(crate) fn frontend_directory() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("HIFISHIFTER_PLUGIN_EDITOR_ASSETS") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err("editor asset override must be absolute".into());
        }
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.join("plugin.html").is_file() {
            return Err("editor plugin.html missing".into());
        }
        return Ok(path);
    }
    let folder = bundle_resources_directory()?.join("frontend");
    if !folder.join("plugin.html").is_file() {
        return Err(format!(
            "plugin frontend assets missing: {}",
            folder.display()
        ));
    }
    Ok(folder)
}

/// 本模块所在的 `Contents` 目录（`<bundle>/Contents`）。
///
/// 【为什么不用 `std::env::current_exe`】插件跑在宿主进程里，那是**宿主**的路径。
/// 【为什么 Windows 不用 `GetModuleFileNameW(None)`】那同样返回宿主 exe。
fn bundle_contents_directory() -> Result<PathBuf, String> {
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
        path.parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .ok_or_else(|| "VST3 Contents unavailable".to_string())
    }
    #[cfg(all(unix, not(test)))]
    {
        use std::ffi::CStr;
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        // SAFETY: dladdr 只写它自己的 out 参数；地址取自本 crate 内的函数。
        let found = unsafe {
            libc::dladdr(
                initialize_models as *const () as *const libc::c_void,
                info.as_mut_ptr(),
            )
        };
        if found == 0 {
            return Err("plugin module path unavailable".into());
        }
        // SAFETY: found != 0 时 dladdr 已把 info 初始化完毕。
        let info = unsafe { info.assume_init() };
        if info.dli_fname.is_null() {
            return Err("plugin module path unavailable".into());
        }
        // SAFETY: dli_fname 指向 dladdr 内部的 NUL 结尾字符串。
        let module = PathBuf::from(
            unsafe { CStr::from_ptr(info.dli_fname) }
                .to_string_lossy()
                .into_owned(),
        );
        return module
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .ok_or_else(|| "VST3 Contents unavailable".to_string());
    }
    #[cfg(test)]
    {
        Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."))
    }
}

/// bundle 资源根（`Contents/Resources`）。
pub(crate) fn bundle_resources_directory() -> Result<PathBuf, String> {
    Ok(bundle_contents_directory()?.join("Resources"))
}
