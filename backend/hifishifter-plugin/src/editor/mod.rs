//! 插件内原GUI宿主；原生视图不启动独立app，也不接管宿主消息循环。
mod browser_files;
pub(crate) mod commands;
pub(crate) mod connection;
pub(crate) mod errors;
mod events;
mod formant;
// 宿主专属快捷键的让路策略（跨平台：Policy 无 cfg，原语按平台各一份）。
pub(crate) mod host_key_relay;
pub(crate) mod media_identity;
mod notebook;
// 只在非测试构建里存在：唯一调用点 `EditorSession::run` 的空闲分支本身是
// `#[cfg(not(test))]`（测试进程不该扫描/删除真实的用户临时目录）。
pub(crate) mod capture_edits;
pub(crate) mod parameter_atlas;
// GUI 的另一半：文件对话框 / 在文件管理器中显示 / 用默认程序打开（按平台各一份）。
pub(crate) mod gui;
#[cfg(not(windows))]
pub(crate) mod platform_services;
mod plugin_diagnostics;
pub(crate) mod private_groups;
pub(crate) mod resources;
pub(crate) mod routing;
pub(crate) mod session;
pub(crate) mod slow_lane;
#[cfg(not(test))]
mod temp_sweep;
mod view;
pub(crate) mod workspace;
pub(crate) use errors::classify_error;
pub(crate) use view::create_view_with_link;

#[cfg(test)]
mod tests {
    use super::classify_error;

    /// 【为什么钉死】前端据 `terminal` **永久停止轮询**。一旦这里把终态错误判成
    /// `transient`，那个"把终态错误答 19,056 次"的缺陷就会以另一种形式回来。
    #[test]
    fn session_teardown_errors_are_terminal() {
        for error in [
            "document closed",
            "FX processor closed",
            "editor view closed or unknown",
            "editor view identity mismatch",
            "editor route changed while request queued",
            "editor route belongs to another document",
            "editor lease revoked",
            "plugin instance closed",
            "plug-in instance is not bound to an ARA document",
        ] {
            assert_eq!(
                classify_error(error),
                "terminal",
                "{error} must be terminal"
            );
        }
    }

    /// 【为什么钉死】"尚未连接"与"宿主拒绝"这类是**暂时**的：编辑器视图在组件
    /// 创建与绑定之间必然经历它，判成终态会让界面刚打开就宣称"会话已结束"。
    #[test]
    fn recoverable_conditions_stay_transient() {
        for error in [
            "FX editor not connected to its processor yet",
            "host clip changed before GUI commit; refresh required",
            "host does not provide ARA playback control",
            "invalid editor arguments",
        ] {
            assert_eq!(
                classify_error(error),
                "transient",
                "{error} must be transient"
            );
        }
    }
}

/// 卸载前撤销本模块注册的窗口类 / 停掉私有 GUI 线程；尚有窗口时明确拒绝正常卸载。
pub(crate) fn shutdown() -> bool {
    gui::shutdown()
}

/// 把本模块钉进宿主进程：引用计数永久 +1，宿主随后的 `FreeLibrary`/`dlclose`
/// 不再真正卸载本模块的代码页。
///
/// 【为什么需要它】三类后台实体的存活期**长于**任何一次卸载协商，而宿主并不
/// 知情：慢车道任务以分钟计（基准测试实测 31–134 秒）、Linux 远程编辑器的
/// `hfs-editor` 驱动线程 20ms 轮询、被 abandon 的私有 GTK 线程（`gui::shutdown`
/// 等 5 秒就放弃）。它们任何一个在 `ExitDll` 之后的下一次执行都会落到已卸载的
/// 代码页 —— 宿主崩溃（第 5 轮审查 §3）。与其和宿主协商"请别卸载"（`ExitDll`
/// 返回假并不约束宿主），不如让卸载变成无害的引用计数递减。
///
/// 【为什么是不可逆的】`PIN`/`RTLD_NODELETE` 没有反向操作。这是既定取舍的
/// 推广：WebView2 视图一旦打开就已经同样钉住（`gui/windows.rs` 的
/// `pin_callback_code`，注释写明"开发升级需要正常退出宿主"）。触发点取
/// "后台工作真正开始"的那一刻：慢车道任务开始执行、远程/嵌入式编辑器附着。
#[allow(unused)]
pub(crate) fn pin_module() {
    #[cfg(windows)]
    {
        use windows::core::PCWSTR;
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::System::LibraryLoader::{
            GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            GET_MODULE_HANDLE_EX_FLAG_PIN,
        };
        static PINNED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
        let result = PINNED.get_or_init(|| {
            let mut module = HMODULE::default();
            unsafe {
                GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    PCWSTR(pin_module as *const () as *const u16),
                    &mut module,
                )
            }
            .map_err(|error| error.to_string())
        });
        if let Err(error) = result {
            log::warn!("[ara] could not pin the module: {error}");
        }
    }
    #[cfg(unix)]
    {
        static PINNED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
        let result = PINNED.get_or_init(|| unsafe {
            // 先用 dladdr 拿本模块的装载路径（传一个本 crate 内函数的地址），
            // 再以 RTLD_NODELETE 重开它：引用计数 +1 且此后 dlclose 不卸载。
            let symbol = pin_module as *const () as *const libc::c_void;
            let mut info: libc::Dl_info = std::mem::zeroed();
            if libc::dladdr(symbol, &mut info) == 0 {
                return Err("dladdr could not locate this module".to_string());
            }
            let path = std::ffi::CStr::from_ptr(info.dli_fname);
            let handle = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_NODELETE);
            if handle.is_null() {
                return Err("dlopen(self, RTLD_NODELETE) failed".to_string());
            }
            Ok(())
        });
        if let Err(error) = result {
            log::warn!("[ara] could not pin the module: {error}");
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        // 该目标没有已知的驻留原语：后台工作窗口内的卸载风险维持原状（记录在案）。
    }
}
