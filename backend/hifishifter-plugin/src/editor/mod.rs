//! 插件内原GUI宿主；原生视图不启动独立app，也不接管宿主消息循环。
mod browser_files;
pub(crate) mod commands;
pub(crate) mod connection;
pub(crate) mod errors;
mod events;
mod formant;
mod source_files;
// 宿主专属快捷键的让路策略（跨平台：Policy 无 cfg，原语按平台各一份）。
pub(crate) mod host_key_relay;
pub(crate) mod host_threads;
pub(crate) mod media_identity;
mod notebook;
// 只在非测试构建里存在：唯一调用点 `EditorSession::run` 的空闲分支本身是
// `#[cfg(not(test))]`（测试进程不该扫描/删除真实的用户临时目录）。
pub(crate) mod capture_edits;
mod param_curves;
// 内嵌编辑器：**格式无关**的 Policy + 三个平台原语。VST3 的 `IPlugView` 与 CLAP 的
// `clap_plugin_gui` 都只经 `gui::create_view` 建视图（见 `gui::EditorView`）。
pub mod gui;
#[cfg(not(windows))]
pub(crate) mod platform_services;
mod plugin_diagnostics;
pub(crate) mod private_groups;
pub(crate) mod resources;
pub(crate) mod routing;
pub mod port;
pub mod session;
pub(crate) mod slow_lane;
#[cfg(not(test))]
mod temp_sweep;

mod view;
pub(crate) mod workspace;

/// 临时目录回收表：「本地数据子目录 → 目录名前缀」。
///
/// 【为什么放在这里而不是 `temp_sweep` 里】那个模块只在非测试构建里编译（测试进程
/// 不该去删真实的用户目录），放在它里面的常量与测试都跟着不参与 `cargo test` ——
/// 于是"少回收一类目录"这种缺陷不会有任何测试变红。表放在 `editor` 这一层，
/// 判据（[`TEMP_SWEEP_TARGETS`] 的条目）就能被普通单元测试守住。
///
/// 【为什么 webview 有两个前缀】`view-<pid>` 是两个格式**共用**的 WebView2 profile
/// （同一个共享视图栈，见 `gui/windows.rs::profile_directory`）；`clap-<pid>` 是 CLAP
/// 外壳独立实现时期留下的旧目录名 —— 仍然必须扫，否则那些机器上的旧目录永远不会被回收。
/// 两者同根、判据（属主进程已退出）相同，所以由同一趟扫描处理。
pub(crate) const TEMP_SWEEP_TARGETS: &[(&str, &str)] = &[
    ("webview", "view-"),
    ("webview", "clap-"),
    ("pcm", "hfs-ui-"),
];

pub(crate) use errors::classify_error;
/// 视图 bootstrap 的能力快照（两种格式共用同一份事实）。
pub use gui::client::bootstrap_capabilities;
pub use gui::client::log_frontend_error;
pub(crate) use view::create_view_with_link;

/// 建视图的唯一入口（父句柄 + 尺寸 + 实例链路）。
pub use gui::create_view as create_editor_view;
/// 一个已建立的嵌入式视图；两个格式共用同一份实现。
pub use gui::EditorView;
/// 编辑器链路：把视图绑到一个 `ExtensionOwner` 上。
pub use routing::EditorLink;
/// 实例在进程内的租约；持有它 = 让 `live_owners()` 看得见这个实例。
pub use routing::RouteLease;

/// 宿主主线程的周期工作：投递泵 + 宿主时钟/Undo/setDirty 收口。
///
/// 【为什么只有这一份】VST3 挂在 REAPER 的 `plugin_register("timer")` 上，CLAP 挂在
/// `clap.timer-support` 的 `on_timer` 与 `on_main_thread` 上 —— 驱动方式不同，但
/// **每次节拍要做的事**完全相同。此前这段顺序写在 `host/reaper/host_timer.rs` 里，
/// CLAP 没有对应物，于是它的宿主侧周期工作一个环节都不跑。
pub fn host_main_thread_tick() {
    // 1) 先排空投递队列：编辑器命令要回到这条线程上执行。
    crate::host::marshal::pump();
    // 2) 再跑宿主侧周期工作（时钟采样、Undo 收口、setDirty 冲刷）。
    gui::client::host_heartbeat();
}

/// 只排空宿主线程投递队列（不跑心跳）。
///
/// 【为什么与 `host_main_thread_tick` 分开】VST3 的 `IPlugView` 回调（`getSize` /
/// `onSize` / `canResize`）与 CLAP 的 `clap.gui` 回调都是"宿主此刻就在我们的 UI 线程上"
/// 的证据 —— 那里只该泵队列（微秒级），不该跑整个心跳（时钟采样 + 枚举 + JSON）。
/// 此前 CLAP 的 gui 回调连泵都不跑，于是没有 timer-support 的宿主上编辑器命令
/// 只能等用户拖动窗口。
pub fn pump_host_thread() {
    crate::host::marshal::pump();
}


#[cfg(test)]
mod tests {
    use super::classify_error;

    /// 回收表必须覆盖两个格式的 WebView2 profile（含 CLAP 独立实现时期的旧前缀）。
    ///
    /// 【为什么这条要单独守】漏掉一个前缀不会报错：宿主每次启动都会多留一份几百 MB 的
    /// 用户数据目录，直到用户自己发现（VST3 曾经就是这样攒出 147 个目录、7.1 GiB）。
    #[test]
    fn the_temp_sweep_table_covers_both_webview_profiles() {
        assert!(
            super::TEMP_SWEEP_TARGETS.contains(&("webview", "clap-")),
            "CLAP profiles are never reclaimed: {:?}",
            super::TEMP_SWEEP_TARGETS
        );
        // VST3 那一列不能在加 CLAP 的时候被顺手弄丢。
        assert!(super::TEMP_SWEEP_TARGETS.contains(&("webview", "view-")));
        assert!(super::TEMP_SWEEP_TARGETS.contains(&("pcm", "hfs-ui-")));
    }

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
pub fn shutdown() -> bool {
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
