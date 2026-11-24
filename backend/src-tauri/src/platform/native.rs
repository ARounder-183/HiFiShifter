//! 原生桥（Android）：系统文件选择器、安全区、系统栏。
//!
//! 【这个模块解决的三件事，以及为什么它们都只能在这里做】
//!
//! 1. **系统文件选择器**。`rfd` 在移动目标上编译不过（后端只有 gtk/macos/win32），
//!    wry 0.57 的 `RustWebChromeClient` 也没有实现 `onShowFileChooser` —— 于是
//!    `<input type="file">` 在 Android 上点了毫无反应，而 `FileDialog::pick_file`
//!    在移动端返回 `None`。**导入媒体、打开工程、另存为** 全都依赖选择器，
//!    这就是"移动端大量功能不工作"的根因。SAF 只有原生侧能发起。
//! 2. **安全区**。`enableEdgeToEdge()` 让 WebView 铺到状态栏与手势条底下，而
//!    Android WebView 的 `env(safe-area-inset-*)` 只反映刘海、不反映系统栏 ——
//!    CSS 里那套 `--qt-safe-*` 在真机上恒为 0。真实内边距只能从 `WindowInsets` 取。
//! 3. **系统栏图标明暗**。应用主题与系统夜间模式无关，必须由应用自己告诉系统。
//!
//! ## 边界纪律
//!
//! 本模块是"平台差异的最后一厘米"：它**只**做上面三件事，不做业务判断，也不
//! 决定文件该落在哪个目录（那是 [`super::paths`] 的事）。Kotlin 侧把 URI 物化成
//! 缓存里的真实文件，Rust 侧再决定把它归类到哪里 —— 两处各管一件事，不会分叉。
//!
//! ## 线程约束（**最容易踩的坑**）
//!
//! **绝对不能从主线程调用 `run_mobile_plugin`。** wry 的 `dispatch` 只是把闭包塞进
//! `MainPipe` 再写一个字节到管道（`wry/src/android/main_pipe.rs`），由**主线程**的
//! 消息循环取出来执行；而 `run_mobile_plugin` 会在**调用线程**上阻塞等待结果。
//! 于是主线程上调用 = 自己等自己 —— 应用直接冻死，没有任何日志。
//!
//! Tauri 的**同步**命令就跑在主线程（Android 上 WebView 的 IPC 线程就是主线程），
//! 所以凡是会触到原生桥的命令都必须是 `async` + `spawn_blocking`
//! （与 `commands.rs` 的 dialogs 段同一条理由）。本模块里：
//!
//! - `pick_files` / `save_to_location`：只在阻塞池里被调用；
//! - `insets` / `set_system_bars`：由 `get_surface_profile` / `get_surface_insets` /
//!   `set_system_bars` 三个 **async** 命令调用；
//! - `export_dir`：**只读缓存**（见 [`EXPORT_DIR`]），缓存由 setup 起的后台线程预取。
//!   它被 `platform::paths::documents_dir` 这类可能在同步命令里执行的地方使用，
//!   因此绝不能阻塞。
//!
//! ## 桌面与 iOS
//!
//! 桌面目标上 `init()` 返回一个**空插件**（不注册任何原生类），所有查询返回
//! 零值/错误；iOS 上同样返回错误（本仓还没有 iOS 原生桥）。调用点全部位于
//! `#[cfg(not(desktop))]` 分支内，桌面路径一个字节都不变。

use std::path::{Path, PathBuf};

use serde::Serialize;
// 反序列化只用在原生桥的响应体上，而它们全部是 Android 专属。
#[cfg(target_os = "android")]
use serde::Deserialize;
use tauri::{
    plugin::{Builder, TauriPlugin},
    Wry,
};
// 只有 Android 分支会碰句柄与 `Manager`；桌面上它们未使用，因此单独引入。
#[cfg(target_os = "android")]
use tauri::{plugin::PluginHandle, Manager};

/// 窗口的安全区内边距，单位是 **CSS px（= Android 的 dp）**。
///
/// 【为什么是整数而不是浮点】它只会被写进 CSS 自定义属性（`--qt-inset-top: 24px`），
/// 亚像素在这里没有意义，而整数让"相等"可以直接比较 —— 监听器靠它避免重复下发。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Insets {
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub left: u32,
}

/// 原生桥的句柄。注册失败（例如 `gen/android` 是旧版本、没有 Kotlin 类）时不放进
/// 状态，所有调用点因此拿到 `None` 而不是崩溃 —— 移动端的降级是"功能不可用 +
/// 一条 error 日志"，不是白屏。
#[cfg(target_os = "android")]
struct Bridge {
    handle: PluginHandle<Wry>,
}

/// 注册原生桥插件。**必须在 `Builder::plugin` 里注册**，否则 Rust 侧拿不到句柄。
pub fn init() -> TauriPlugin<Wry> {
    Builder::new("hifishifter-native")
        .setup(|app, api| {
            // 桌面上没有任何原生类要注册，`api` 因此未使用 —— 显式消费掉，
            // 而不是把它改名成 `_api`（那样 Android 分支里又要改回来）。
            #[cfg(not(target_os = "android"))]
            let _ = &api;
            #[cfg(target_os = "android")]
            {
                // 类名拼成 `<identifier 的点换斜杠>/<Class>`，由 tauri 用 activity 的
                // ClassLoader 解析（见 tauri 的 `register_android_plugin`）。
                match api.register_android_plugin("com.arounder.hifishifter", "NativeBridgePlugin")
                {
                    Ok(handle) => {
                        app.manage(Bridge { handle });
                        // 导出目录在应用生命周期内不变，启动时预取一次即可 ——
                        // 且必须放到后台线程（见 `warm_export_dir` 的说明）。
                        warm_export_dir(app);
                    }
                    Err(error) => {
                        // 【为什么不 panic】缺少 Kotlin 类通常意味着 `gen/android` 是
                        // 上一轮生成物（没有 NativeBridgePlugin.kt）。此时应用仍应启动：
                        // 界面能打开，用户才能看到"导入不可用"的提示，而不是一片黑屏。
                        log::error!("[native] android bridge unavailable: {error}");
                    }
                }
            }
            let _ = app;
            Ok(())
        })
        .build()
}

#[cfg(target_os = "android")]
fn bridge(app: &tauri::AppHandle) -> Option<&Bridge> {
    app.try_state::<Bridge>().map(|state| state.inner())
}

// ── 安全区 ──────────────────────────────────────────────────────────────────

#[cfg(target_os = "android")]
#[derive(Debug, Default, Deserialize)]
struct InsetsResponse {
    #[serde(default)]
    top: u32,
    #[serde(default)]
    right: u32,
    #[serde(default)]
    bottom: u32,
    #[serde(default)]
    left: u32,
}

/// 当前窗口的安全区。桌面与 iOS 返回零值（CSS 的 `env()` 在那里本来就是对的）。
pub fn insets(app: &tauri::AppHandle) -> Insets {
    #[cfg(target_os = "android")]
    {
        if let Some(bridge) = bridge(app) {
            // 这个命令是**同步**的（Kotlin 侧直接返回缓存值），因此不会阻塞等待
            // 用户操作；即便如此它也在 tokio 线程上被调用，不在主线程。
            if let Ok(response) = bridge
                .handle
                .run_mobile_plugin::<InsetsResponse>("insets", ())
            {
                return Insets {
                    top: response.top,
                    right: response.right,
                    bottom: response.bottom,
                    left: response.left,
                };
            }
        }
    }
    let _ = app;
    Insets::default()
}

// ── 系统栏 ──────────────────────────────────────────────────────────────────

#[cfg(target_os = "android")]
#[derive(Debug, Serialize)]
struct SystemBarsRequest {
    dark: bool,
}

/// 按应用主题切换状态栏/导航栏图标的明暗。
pub fn set_system_bars(app: &tauri::AppHandle, dark: bool) {
    #[cfg(target_os = "android")]
    {
        if let Some(bridge) = bridge(app) {
            if let Err(error) = bridge
                .handle
                .run_mobile_plugin::<serde_json::Value>("setSystemBars", SystemBarsRequest { dark })
            {
                log::warn!("[native] setSystemBars failed: {error}");
            }
        }
    }
    let _ = (app, dark);
}

// ── 导出目录 ────────────────────────────────────────────────────────────────

#[cfg(target_os = "android")]
#[derive(Debug, Deserialize)]
struct PathResponse {
    path: PathBuf,
}

/// 导出目录的进程内缓存。
///
/// 【为什么缓存而不是每次问原生】`export_dir` 会被 `platform::paths::documents_dir`
/// 调用，而后者在 `get_export_audio_defaults` / `preview_export_audio_plan` 这类
/// **同步**命令里执行 —— 主线程上调用原生桥必然死锁（见模块头的线程约束）。
/// 目录本身在应用生命周期内不会变，一次预取足够。
static EXPORT_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// 后台预取导出目录（插件 setup 调用一次）。
///
/// 【为什么起线程】setup 跑在主线程上，直接调用会死锁。放到后台线程之后，
/// 桥的 `dispatch` 排进主线程队列、主线程继续往下走（setup 返回、事件循环起来），
/// 消息被取出来执行，线程拿到结果写进缓存。
#[cfg(target_os = "android")]
fn warm_export_dir(app: &tauri::AppHandle) {
    #[cfg(target_os = "android")]
    {
        let Some(bridge) = app
            .try_state::<Bridge>()
            .map(|state| state.inner().handle.clone())
        else {
            return;
        };
        std::thread::spawn(move || {
            match bridge.run_mobile_plugin::<PathResponse>("exportDir", ()) {
                Ok(response) => {
                    let _ = EXPORT_DIR.set(response.path);
                }
                Err(error) => log::warn!("[native] exportDir failed: {error}"),
            }
        });
    }
    let _ = app;
}

/// 用户能在系统文件管理器里找到的目录。
///
/// 【为什么由原生给而不是 `app.path().document_dir()`】后者是
/// `Android/data/<pkg>/files/Documents`，从 Android 11 起在系统"文件"应用里
/// **不可见**；原生给的是 `Android/media/<pkg>/HiFiShifter`，同一个应用、同样
/// 零权限，但不隐藏。见 `NativeBridgePlugin.exportDir` 的注释。
///
/// **只读缓存**：冷缓存时返回 `None`，调用方退回 `document_dir`（旧行为）。
/// 缓存由 [`warm_export_dir`] 在启动时预热，用户打开导出对话框时早已就位。
pub fn export_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    let _ = app;
    if let Some(dir) = EXPORT_DIR.get() {
        return Some(dir.clone());
    }
    log::debug!("[native] export dir is not cached yet; the caller falls back");
    None
}

// ── 文件选择器 ──────────────────────────────────────────────────────────────

#[cfg(target_os = "android")]
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PickFilesRequest {
    mime_types: Vec<String>,
    multiple: bool,
}

#[cfg(target_os = "android")]
#[derive(Debug, Deserialize)]
struct PickFilesResponse {
    #[serde(default)]
    paths: Vec<PathBuf>,
    /// 用户取消时原生侧返回 `{ "canceled": true }`，而不是一个错误。
    ///
    /// 【为什么不用 reject】`invoke.reject` 在 Rust 侧只剩一个字符串错误，
    /// 与"读文件失败"无法区分 —— 那会把一次取消变成界面上的一条红条。
    #[serde(default)]
    canceled: bool,
}

/// 打开系统文件选择器；用户取消时返回空列表。
pub fn pick_files(
    app: &tauri::AppHandle,
    mime_types: &[String],
    multiple: bool,
) -> Result<Vec<PathBuf>, String> {
    #[cfg(target_os = "android")]
    {
        let Some(bridge) = bridge(app) else {
            return Err("native file picker is unavailable on this build".to_string());
        };
        let request = PickFilesRequest {
            mime_types: mime_types.to_vec(),
            multiple,
        };
        // 【为什么这里会阻塞】用户要在系统选择器里做选择，Rust 侧必须等结果。
        // 调用点全部在 `spawn_blocking` 或 `async` 命令里（从不在主线程），
        // 与桌面上 `rfd::FileDialog::pick_file` 的阻塞语义完全对称。
        let response = bridge
            .handle
            .run_mobile_plugin::<PickFilesResponse>("pickFiles", request)
            .map_err(|error| error.to_string())?;
        if response.canceled {
            return Ok(Vec::new());
        }
        // 这个块就是函数体（`#[cfg(not(target_os = "android"))]` 的兄弟块被编译掉了），
        // 因此用表达式而不是 `return` —— clippy 的 `needless_return` 盯的就是这里。
        Ok(response.paths)
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, mime_types, multiple);
        Err("native file picker is unavailable on this platform".to_string())
    }
}

#[cfg(target_os = "android")]
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SaveToLocationRequest {
    source_path: String,
    file_name: String,
    mime_type: String,
}

#[cfg(target_os = "android")]
#[derive(Debug, Deserialize)]
struct SaveToLocationResponse {
    #[serde(default)]
    canceled: bool,
    #[serde(default)]
    name: String,
}

/// 把已经写好的文件交给用户选定的位置（SAF）。返回目标处的显示名；取消返回 `None`。
pub fn save_to_location(
    app: &tauri::AppHandle,
    source: &Path,
    file_name: &str,
    mime_type: &str,
) -> Result<Option<String>, String> {
    #[cfg(target_os = "android")]
    {
        let Some(bridge) = bridge(app) else {
            return Err("native file picker is unavailable on this build".to_string());
        };
        let request = SaveToLocationRequest {
            source_path: source.to_string_lossy().into_owned(),
            file_name: file_name.to_string(),
            mime_type: mime_type.to_string(),
        };
        let response = bridge
            .handle
            .run_mobile_plugin::<SaveToLocationResponse>("saveToLocation", request)
            .map_err(|error| error.to_string())?;
        if response.canceled {
            return Ok(None);
        }
        Ok(Some(response.name))
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, source, file_name, mime_type);
        Err("native file picker is unavailable on this platform".to_string())
    }
}

/// 这个构建是否真的挂上了原生桥。
pub fn is_available(app: &tauri::AppHandle) -> bool {
    #[cfg(target_os = "android")]
    {
        bridge(app).is_some()
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = app;
        false
    }
}
