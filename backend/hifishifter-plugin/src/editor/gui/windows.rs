//! Windows 原语：一个子 `HWND` + WebView2 + 20 ms `WM_TIMER` 泵。
//!
//! 【这一份与 `webview.rs` 的差别必须逐块可对照】搬迁是**纯机械**的：判定逻辑一个字
//! 都没改，只是搬到了 `client.rs`。这里保留的三件事全是"只有 Windows 会做"的：
//! 建子窗口、把虚拟主机映射到 `Resources/frontend`、把 `WebMessageReceived` 转给
//! Policy。评审时对着 `git show` 逐块比即可。
//!
//! 【为什么 `ORIGIN` 只在 Windows 上存在】WebView2 用
//! `SetVirtualHostNameToFolderMapping` 把 `hifishifter.invalid` 映射到一个本地目录，
//! 于是"资源在哪"由宿主进程决定；WebKit 两个平台走自定义 scheme（见 `protocol.rs`）。

use super::client::{self, EditorClient};
use super::port::{EditorHost, EditorPort, GuiFidelity, NativeWindow};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use webview2_com::Microsoft::Web::WebView2::Win32::*;
use webview2_com::{
    AddScriptToExecuteOnDocumentCreatedCompletedHandler, CoTaskMemPWSTR,
    CoreWebView2EnvironmentOptions, CreateCoreWebView2ControllerCompletedHandler,
    CreateCoreWebView2EnvironmentCompletedHandler, NavigationCompletedEventHandler,
    NavigationStartingEventHandler, NewWindowRequestedEventHandler, ProcessFailedEventHandler,
    WebMessageReceivedEventHandler,
};
use windows::core::{w, Interface, PCWSTR, PWSTR};
use windows::Win32::Foundation::{
    COLORREF, HANDLE, HINSTANCE, HMODULE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, CreateSolidBrush, MonitorFromWindow, MONITOR_DEFAULTTONEAREST,
};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::LibraryLoader::GET_MODULE_HANDLE_EX_FLAG_PIN;
use windows::Win32::System::LibraryLoader::{
    GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::HiDpi::{
    AreDpiAwarenessContextsEqual, GetDpiForMonitor, GetDpiForWindow, GetWindowDpiAwarenessContext,
    SetThreadDpiAwarenessContext, DPI_AWARENESS_CONTEXT,
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, DPI_AWARENESS_CONTEXT_UNAWARE, MDT_EFFECTIVE_DPI,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{GetFocus, SetFocus};
use windows::Win32::UI::WindowsAndMessaging::*;

/// 内嵌 WebView 的虚拟 origin。
///
/// 【为什么用 `.invalid`】RFC 2606 把 `.invalid` 保留为**永不解析**的 TLD，所以即使宿主
/// 机器上有恶意 DNS/代理，这个 origin 也不可能指向真实站点 —— 它只是本进程内
/// `WebResourceRequested` 拦截器（见下面的路由表）用的命名空间。不要改成任何可注册域名。
const ORIGIN: &str = "https://hifishifter.invalid/";
static WINDOWS: AtomicUsize = AtomicUsize::new(0);
/// 窗口属性 token：只用来判定 "这个句柄还是不是我的那个窗口"。
static NEXT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static CLASS_LOCK: Mutex<bool> = Mutex::new(false);
static WEBVIEW_MODULE_PIN: OnceLock<Result<(), String>> = OnceLock::new();

/// 编辑器加载期的底色（RGB 30/30/34）。
///
/// 【为什么要有它】控制器创建到首帧之间 WebView2 画的是**默认白底**；现场
/// （VEGAS 2026-11）重开面板时会白屏数秒，与编辑器的深色主题形成刺眼反差。
/// 窗口类背景与 `SetDefaultBackgroundColor` 用同一个颜色，等待期就是一块
/// 与编辑器同色的面板，而不是白屏。
const EDITOR_BACKGROUND: COLORREF = COLORREF(0x0022_1E1E); // B=0x22, G=0x1E, R=0x1E
const EDITOR_BACKGROUND_RGB: (u8, u8, u8) = (30, 30, 34);

/// 停放的编辑器窗口：宿主关掉插件面板后**不销毁**的那一份原生视图。
///
/// 【为什么停放而不是销毁】WebView2 在"上一个控制器刚关闭"之后创建新控制器，
/// 现场实测要等 ~5.6 s（VEGAS 2026-11-22：每次重开面板 `createView` → `page ready`
/// 正好 5619/5629 ms，期间面板一片白）。而窗口、控制器、页面本身都可复用：
/// 宿主给的下一个 `attached` 只是同一控制器（`EditorLink` 是同一个 `Arc`）的另一个
/// 视图。把 HWND 挂到 message-only 窗口下隐藏，下一次 attached 原样挂回新父窗口，
/// 重开瞬时完成，页面状态（滚动位置、展开的面板）也一并保留。
///
/// 【为什么只停放嵌入形态】popup 是宿主窗口树的"替身"，它的几何/可见性逐拍跟随
/// 宿主；停放它等于把一个顶层窗口藏起来再改回子窗口，收益（ACID 重开本来就快）
/// 远小于风险。popup 形态仍按原路径销毁重建。
struct ParkedWindow {
    /// 只存数值，不把 `HWND` 当 `Send`：真正的 Win32 调用都在 `thread` 上做。
    hwnd: usize,
    thread: u32,
    token: usize,
    /// 与宿主控制器的编辑器链路同一份 —— 只有同一个链路才允许复用。
    link: Arc<crate::editor::routing::EditorLink>,
    /// 创建时的父窗口是不是 DPI-unaware。上下文不同（感知父窗口 vs 不感知父窗口）
    /// 不能复用同一个子窗口：子窗口的感知上下文在创建时定死。
    unaware_parent: bool,
}
static PARKED: Mutex<Vec<ParkedWindow>> = Mutex::new(Vec::new());
/// 最多停放几个窗口。正常一个插件实例一个；上限只是防止异常宿主把内存堆起来。
const MAX_PARKED_WINDOWS: usize = 4;

// 进程内**唯一**的 WebView2 环境。
//
// 【为什么必须池化】此前每个视图各调一次 `CreateCoreWebView2EnvironmentWithOptions`
// —— 一次会话 31 次开窗就是 31 次环境创建。每次创建都会拉起（或接管）一个浏览器
// 进程并初始化一整套缓存；而 WebView2 的环境本身就是**可以**在同进程内被多个控制器
// 共享的，文档里的约束只是"一个用户数据目录不能被两个**进程**同时打开"。于是那 31 次
// 里只有第 1 次是必要的，其余 30 次只是在重建同一份东西。
//
// 池化之后还有两个附带收益：① 浏览器进程、HTTP 缓存、GPU shader 缓存跨窗口复用，
// 关掉再打开插件窗口不再从冷缓存起步；② 渲染进程死亡后的自动重建（见 [`rebuild_page`]）
// 变成毫秒级 —— 不必再等一次环境初始化。
//
// 【为什么是 thread_local】`ICoreWebView2Environment` 不是 `Send`/`Sync`：COM 的
// 单元线程模型要求它在创建它的 STA 上使用。而所有视图都在宿主 UI 线程上创建，
// 所以"UI 线程独占一份"既满足 COM 的约束，也满足共享的需要。
thread_local! {
    static ENVIRONMENT: RefCell<Option<ICoreWebView2Environment>> = const { RefCell::new(None) };
    /// 本窗口已经自动重建过几次（连续计数，页面成功加载后归零）。
    static PAGE_REBUILDS: Cell<u32> = const { Cell::new(0) };
}

/// 连续自动重建的上限。
///
/// 【为什么要有上限】重建解决的是"渲染进程被 OOM 杀掉"这种**偶发**故障。若页面每次
/// 都立刻再死一次（比如某个必现的崩溃），无限重建就是一块永远在闪的面板加一份永远在
/// 涨的日志。三次之后停下来并明确告诉用户"重开编辑器"，比空转诚实。
const MAX_PAGE_REBUILDS: u32 = 3;

/// 交给 Chromium 的附加开关。
///
/// 【`--enable-precise-memory-info`】没有它 `performance.memory` 会被模糊到 100 MiB
/// 粒度，看门狗只能看出"涨了 100 MiB 以上"，看不出斜率。它只影响我们自己创建的环境，
/// 与宿主无关。
///
/// 【`--disk-cache-size`】单个 profile 的磁盘缓存上限。实测单份最大 115 MiB，而这个
/// 数字乘上"曾经攒下的 147 份"就是 7 GiB。给一个明确上限之后，即使清扫失效，一份
/// profile 也不会再长到那个量级。
const BROWSER_ARGUMENTS: &str = "--enable-precise-memory-info --disk-cache-size=67108864";

/// 把本线程临时切到"与参照窗口同一坐标语义"的作用域守卫。
///
/// 【这条守卫回答的问题】创建子窗口 / 移动子窗口 / 读客户区，Win32 都按**调用线程**
/// 的感知上下文解释坐标；而嵌入插件的参照物（宿主父窗口）有它自己的上下文。两者
/// 不一致时，同一组数字会被解释成两种长度 —— ACID Pro 11 实测（125% 缩放，2026-11）：
/// `sfvstwrap` 包裹窗口在不感知线程上创建（父窗口 96-DPI），它的客户区被系统虚拟化
/// 成 96 空间，而感知宿主按物理像素给它留位 —— 子窗口若按 `getSize` 的数字创建，
/// 要么被再放大 1.25×（溢出 ⇒ 宿主滚动条、右/下被裁），要么被再缩小（页面留白）。
///
/// 【守卫的规则：跟父窗口走，不跟"最优感知"走】
///   * 父窗口 DPI-unaware（96 且上下文不感知）⇒ 切到 **unaware**。unaware 父窗口
///     里嵌感知子窗口是平台**不支持**的组合（实测 2026-11：几何错位且内容仍被 DWM
///     拉伸）—— 上下文一致才能让"父窗口给多大、子窗口占多大"成立。模糊依旧，但
///     那是包裹层整体被拉伸的结果，属于宿主桥的限制（根因修复见 `host_identity.rs`
///     的 DllMain 线程提升）；
///   * 其余（感知父窗口）⇒ 切 **PMv2**。感知父窗口与 PMv2 子窗口是平台支持的混合
///     模式（VEGAS Pro 实测父窗口 120/system-aware、子窗口 PMv2，几何与清晰度都
///     正确）；PMv2 让 WebView2 按 `devicePixelRatio=1.25` 渲染，物理像素准确。
///
/// 子窗口的感知上下文在**创建时**定死且随后不可改，所以"创建与摆放"必须发生在
/// 与父窗口一致的上下文里；宿主自身的感知策略一概不动。
///
/// 【为什么是线程级而不是进程级】进程/线程的感知策略是**宿主**的资产，永久改掉
/// 会连带改变宿主其余窗口的缩放行为；`SetThreadDpiAwarenessContext` 只在守卫
/// 存活期间生效，出作用域即还原。
///
/// 【逃生门】`HIFISHIFTER_NO_DPI_AWARE=1` 回到"继承宿主上下文"的旧行为 —— 与
/// `HIFISHIFTER_NO_ARA`（`host_identity.rs`）同一模式，给现场排查留退路。
struct HostContextScope {
    previous: DPI_AWARENESS_CONTEXT,
}

impl HostContextScope {
    /// 把线程切到与 `reference` 一致的坐标语义；`None` = 保持原状（逃生门开启，
    /// 参照窗口读不到，或线程切换失败）。
    fn enter(reference: HWND) -> Option<Self> {
        if std::env::var_os("HIFISHIFTER_NO_DPI_AWARE").is_some_and(|value| value == "1") {
            return None;
        }
        // SAFETY: 只读查询；伪句柄查询对任意 HWND 安全。
        let unaware = unsafe {
            AreDpiAwarenessContextsEqual(
                GetWindowDpiAwarenessContext(reference),
                DPI_AWARENESS_CONTEXT_UNAWARE,
            )
        }
        .as_bool();
        let target = if unaware {
            DPI_AWARENESS_CONTEXT_UNAWARE
        } else {
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        };
        // SAFETY: 参数是常量伪句柄；失败返回 null 伪句柄，按"没切成"处理。
        let previous = unsafe { SetThreadDpiAwarenessContext(target) };
        if previous.0.is_null() {
            return None;
        }
        Some(Self { previous })
    }
    /// 本线程当前的感知上下文是否已经是 PMv2（诊断读数，读后保持原状）。
    fn thread_is_pm_v2() -> bool {
        // SAFETY: 查询 = 切过去拿旧值再立刻还原；伪句柄常量，无副作用。
        let previous =
            unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
        unsafe { SetThreadDpiAwarenessContext(previous) };
        unsafe {
            AreDpiAwarenessContextsEqual(previous, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        }
        .as_bool()
    }
}

impl Drop for HostContextScope {
    fn drop(&mut self) {
        // SAFETY: `previous` 来自 enter() 的成对返回值；还原失败无可用恢复手段。
        unsafe { SetThreadDpiAwarenessContext(self.previous) };
    }
}

impl HostContextScope {
    /// 把线程临时切到 **PMv2**（不参照任何宿主窗口）。
    ///
    /// 【为什么需要它】顶层窗口（popup）的感知上下文在创建时定死，而它**不该**跟
    /// 包裹层走：popup 拥有独立的 DWM surface，按物理像素渲染 —— 这正是它能锐利的
    /// 原因（见 `attach` 的 popup 分支说明）。
    fn force_pm_v2() -> Option<Self> {
        if std::env::var_os("HIFISHIFTER_NO_DPI_AWARE").is_some_and(|value| value == "1") {
            return None;
        }
        // SAFETY: 参数是常量伪句柄；失败返回 null 伪句柄，按"没切成"处理。
        let previous =
            unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
        if previous.0.is_null() {
            return None;
        }
        Some(Self { previous })
    }
}

/// 包裹窗口客户区的**物理**屏幕矩形（popup 的摆放依据）。
///
/// 【为什么能拿到物理值】坐标按**调用线程**的感知上下文虚拟化：PMv2 线程去查
/// DPI-unaware 的包裹窗，系统把它的逻辑矩形按屏幕缩放换算回物理值返回 —— 于是
/// "包裹窗在屏幕上到底占了多大"与"popup 摆多大"是同一把尺子。
fn wrapper_physical_client(parent: HWND) -> Option<RECT> {
    let _scope = HostContextScope::force_pm_v2()?;
    // SAFETY: 只读查询；失败按"拿不到"处理。
    let mut client = RECT::default();
    if unsafe { GetClientRect(parent, &mut client) }.is_err()
        || client.right <= 0
        || client.bottom <= 0
    {
        return None;
    }
    let mut origin = POINT::default();
    if unsafe { ClientToScreen(parent, &mut origin) }.as_bool() {
        Some(RECT {
            left: origin.x,
            top: origin.y,
            right: origin.x + client.right,
            bottom: origin.y + client.bottom,
        })
    } else {
        None
    }
}

/// 是否**显式**要求 DPI popup 形态（`HIFISHIFTER_DPI_POPUP=1`）。
///
/// 【为什么默认关闭】popup 是顶层窗口，不在宿主的窗口树里：它换来了 125% 缩放下的
/// 清晰度，代价是**脱离宿主裁剪与布局** —— 现场（ACID Pro 2026-11-22）表现为覆盖
/// 宿主 FX 链/预设栏、宿主缩放面板时它停在原地。作为插件窗口，这比"宿主包裹层被
/// DWM 拉伸导致的模糊"更不可接受：模糊是宿主的 DPI 虚拟化对**所有**子窗口的统一
/// 结果，而覆盖宿主控件直接让用户无法管理 FX。默认形态因此是宿主父窗口里的普通
/// 子窗口；需要清晰度的用户显式打开这个逃生门，并接受浮窗行为。
fn dpi_popup_requested() -> bool {
    std::env::var_os("HIFISHIFTER_DPI_POPUP").is_some_and(|value| value == "1")
}

/// popup 形态的判据：宿主显式要求，且包裹窗 DPI-unaware、屏幕缩放 > 1。
///
/// 【为什么要这一层】ACID 的 `sfvstwrap` 在不感知上下文里创建包裹窗，宿主进程本身
/// 是感知的。子窗口无论跟包裹层走（96 DPI 渲染 + DWM 1.25× 拉伸 = 模糊）还是强制
/// PMv2（几何被两把尺子撕开 = 滚动条 + 裁切），都修不好"模糊" —— 因为模糊的根源
/// 是**包裹窗的 DWM surface 只有 96 DPI**，子窗口共享它的 surface。popup 是顶层
/// 窗口：它有自己的 surface、自己的感知上下文，是包裹层不感知时唯一能锐利的形态。
///
/// 返回 popup 应用的缩放读数（仅用于日志）；`None` = 走默认的子窗口路径。
fn popup_scale_for(parent: HWND) -> Option<f64> {
    if !window_is_unaware(parent) {
        return None;
    }
    monitor_scale_for(parent).filter(|scale| *scale > 1.0)
}

/// 窗口所在显示器的有效缩放（`DPI / 96`）。
///
/// 【为什么不能只用 `GetDpiForWindow`】它对 DPI-unaware 窗口恒回 96 —— 那正是
/// ACID 包裹层的形态，而 DWM 实际按 125% 合成它。显示器缩放是**物理事实**，
/// 与窗口/线程的感知上下文无关；`GetDpiForMonitor` 只回答 PMv2 上下文里的调用，
/// 所以查询前后要切一次（见 `HostContextScope::force_pm_v2`）。
fn monitor_scale_for(window: HWND) -> Option<f64> {
    let monitor = unsafe { MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST) };
    if monitor.is_invalid() {
        return None;
    }
    let _scope = HostContextScope::force_pm_v2()?;
    let mut dpi_x = 0_u32;
    let mut dpi_y = 0_u32;
    // SAFETY: 只读查询；失败按"拿不到缩放"处理。
    if unsafe { GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y) }.is_err() {
        return None;
    }
    (dpi_y > 0).then(|| f64::from(dpi_y) / 96.0)
}

/// 参照窗口的父级（嵌入形态里就是宿主包裹窗口）；拿不到时退回自身。
///
/// 【为什么退回自身】浮窗等非嵌入形态没有宿主父级；此时子窗口自己的上下文就是
/// 事实参照 —— 守卫按它匹配，行为与"继承"一致。
fn context_reference(hwnd: HWND) -> HWND {
    // SAFETY: 查询失败返回 null，调用方按自身处理。GA_PARENT（直接父级）而不是
    // GA_ROOT：坐标语义的参照物是**直接包裹**子窗口的那层（感知宿主里的不感知
    // 包裹窗），根窗口的上下文回答不了"这层包裹怎么解释坐标"。
    let parent = unsafe { GetAncestor(hwnd, GA_PARENT) };
    if parent.0.is_null() || parent == hwnd {
        hwnd
    } else {
        parent
    }
}

/// 附着点的一次性 DPI 事实：宿主父窗口的 DPI 与感知上下文、本线程切换前的状态。
///
/// 【为什么值得一行日志】ACID 的两个症状都源自"包裹层上下文与感知宿主不一致"，
/// 而修复走到了哪一层只有现场日志才能回答：
///   * 父窗口 DPI=120 且父上下文非 PMv2 ⇒ 感知宿主里的 system-aware 包裹窗（VEGAS
///     形态），PMv2 子窗口是平台支持的组合；
///   * 父窗口 DPI=96 ⇒ 包裹窗 DPI-unaware（ACID 形态）——守卫因此按父窗口匹配
///     （见 `HostContextScope`）；`loader-promote` 则回答 DllMain 的线程提升
///     （见 `host_identity.rs`）有没有赶在包裹窗创建之前生效。
fn log_host_dpi_facts(parent: HWND) {
    let thread = unsafe { GetCurrentThreadId() };
    let (dpi, pm_v2, unaware, thread_pm_v2) = {
        // SAFETY: 只读查询；`AreDpiAwarenessContextsEqual` 对任意伪句柄安全。
        let context = unsafe { GetWindowDpiAwarenessContext(parent) };
        let (dpi, pm_v2, unaware) = unsafe {
            (
                GetDpiForWindow(parent),
                AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
                    .as_bool(),
                AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_UNAWARE).as_bool(),
            )
        };
        (dpi, pm_v2, unaware, HostContextScope::thread_is_pm_v2())
    };
    crate::gui_log_line(&format!(
        "editor DPI facts: parent-dpi={dpi} parent-pmv2={pm_v2} parent-unaware={unaware} \
         thread-pmv2={thread_pm_v2} thread=0x{thread:x} loader-promote={}",
        crate::host_identity::wrapper_thread_dpi_promote_outcome()
    ));
}

fn cached_environment() -> Option<ICoreWebView2Environment> {
    ENVIRONMENT.with(|slot| slot.borrow().clone())
}

fn store_environment(environment: &ICoreWebView2Environment) {
    ENVIRONMENT.with(|slot| *slot.borrow_mut() = Some(environment.clone()));
}

/// 丢掉池化的环境（它随浏览器进程一起死掉时）。
fn drop_environment() {
    ENVIRONMENT.with(|slot| *slot.borrow_mut() = None);
}

/// 构造环境选项：附加开关走这里。
fn environment_options() -> ICoreWebView2EnvironmentOptions {
    let options = CoreWebView2EnvironmentOptions::default();
    // SAFETY: 只写本对象自己的 `UnsafeCell<String>`，没有别名借用。
    unsafe { options.set_additional_browser_arguments(BROWSER_ARGUMENTS.to_string()) };
    options.into()
}

/// 本进程的 WebView2 用户数据目录。
///
/// 【为什么不再用 `%TEMP%`】磁盘清理会在会话进行中删掉 `%TEMP%` 下的内容，
/// 正在运行的 WebView2 用户数据目录被删掉会直接崩 —— 而不只是"缓存丢了"。
/// 放在本地数据目录下，生命周期由我们自己掌握。
///
/// 【为什么是**每进程**一个目录，而不是每视图一个】WebView2 不支持两个进程同时打开
/// 同一个用户数据目录（所以必须带 PID），但同进程内多个控制器可以共享一个环境。
/// 此前每个视图各建一个目录，视图一多就攒出一堆（实测 41 个目录 838 MB），而且只在
/// `Drop` 时清理 —— 宿主被强杀就永久留下。现在按进程复用一份，并由
/// `temp_sweep::sweep_step` 周期回收已退出进程的那份。
fn profile_directory() -> Result<Vec<u16>, String> {
    let profile = hifishifter_kernel::config_location::local_data_subdir("webview")
        .join(format!("view-{}", std::process::id()));
    std::fs::create_dir_all(&profile).map_err(|e| e.to_string())?;
    Ok(wide(&profile.to_string_lossy()))
}

/// 窗口是不是 DPI-unaware（子窗口的感知上下文必须与父窗口同尺，见 `HostContextScope`）。
fn window_is_unaware(window: HWND) -> bool {
    // SAFETY: 只读查询；伪句柄比较对任意 HWND 安全。
    unsafe {
        AreDpiAwarenessContextsEqual(
            GetWindowDpiAwarenessContext(window),
            DPI_AWARENESS_CONTEXT_UNAWARE,
        )
    }
    .as_bool()
}

/// 取一个可复用的停放窗口；顺手清掉已经死掉的条目。
///
/// 复用条件（缺一不可）：同一线程、同一个 `EditorLink`（同一个宿主控制器）、
/// 宿主这次要的仍是嵌入形态、父窗口的 DPI 感知上下文与创建时一致。
/// 条件不满足的条目**留在池里**（宿主下次可能换回原来的形态），死掉的才销毁。
fn take_parked_window(
    link: &Arc<crate::editor::routing::EditorLink>,
    parent: HWND,
) -> Option<ParkedWindow> {
    let thread = unsafe { GetCurrentThreadId() };
    let adopt_unaware = window_is_unaware(parent);
    let popup_wanted = dpi_popup_requested() && popup_scale_for(parent).is_some();
    let mut dead = Vec::new();
    let taken = {
        let mut parked = PARKED.lock().unwrap_or_else(|e| e.into_inner());
        let mut index = 0;
        let mut found = None;
        while index < parked.len() {
            let entry = &parked[index];
            let key = WindowKey {
                hwnd: entry.hwnd,
                thread: entry.thread,
                token: entry.token,
            };
            let alive = entry.thread == thread
                && key.valid()
                && key.client().is_some_and(|client| {
                    let client = client.borrow();
                    // 宿主销毁 FX 实例后链路 owner 会消失：这种窗口没有复用价值，
                    // 留着只会把一份 WebView 挂在 message-only 窗口下。
                    !client.is_dead() && client.link.owner().is_ok()
                });
            if !alive {
                dead.push(parked.remove(index));
                continue;
            }
            if found.is_none()
                && !popup_wanted
                && entry.unaware_parent == adopt_unaware
                && Arc::ptr_eq(&entry.link, link)
            {
                found = Some(parked.remove(index));
                continue;
            }
            index += 1;
        }
        found
    };
    for entry in dead {
        destroy_parked(entry);
    }
    taken
}

/// 销毁一个停放窗口（必须在 `thread` 上调用；跨线程的条目投 `WM_CLOSE`）。
fn destroy_parked(entry: ParkedWindow) {
    let hwnd = HWND(entry.hwnd as *mut c_void);
    if unsafe { GetCurrentThreadId() } == entry.thread {
        let _ = unsafe { DestroyWindow(hwnd) };
    } else {
        let _ = unsafe { PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0)) };
    }
}

/// 把窗口挂到 message-only 窗口下隐藏；成功返回 `true`（调用方不要 `DestroyWindow`）。
///
/// 【为什么不销毁】见 [`ParkedWindow`]：复用同一个 HWND/控制器/页面，下一次
/// `attached` 是毫秒级的；销毁重建会让 WebView2 在"控制器刚关闭"时冷启动 ~5.6 s。
fn park_window(hwnd: HWND, thread: u32, token: usize) -> bool {
    let key = WindowKey {
        hwnd: hwnd.0 as usize,
        thread,
        token,
    };
    let Some(client) = key.client() else {
        return false;
    };
    {
        let client = client.borrow();
        // 页面已死，或宿主正在销毁 FX 实例（链路 owner 已消失）：没有复用价值。
        if client.is_dead() || client.link.owner().is_err() {
            return false;
        }
    }
    let pointer = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const WindowData;
    if pointer.is_null() {
        return false;
    }
    // popup 形态不停放（见 `ParkedWindow` 的说明）。
    if unsafe { !(*pointer).popup_anchor.get().0.is_null() } {
        return false;
    }
    let Ok(parent) = (unsafe { GetParent(hwnd) }) else {
        return false;
    };
    let unaware_parent = window_is_unaware(parent);
    let link = client.borrow().link.clone();
    let port = unsafe { (*pointer).port.clone() };
    // 先让页面停止渲染，再隐藏并挂走：宿主随后销毁父窗口时，窗口树里已经没有我们。
    port.set_page_visible(false);
    let _ = unsafe { KillTimer(Some(hwnd), 0x4853) };
    unsafe {
        let _ = ShowWindow(hwnd, SW_HIDE);
        if SetParent(hwnd, Some(HWND_MESSAGE)).is_err() {
            // 挂不走就保持原状，让 Drop 走销毁路径。
            let _ = SetParent(hwnd, Some(parent));
            return false;
        }
    }
    let entry = ParkedWindow {
        hwnd: hwnd.0 as usize,
        thread,
        token,
        link,
        unaware_parent,
    };
    let evicted = {
        let mut parked = PARKED.lock().unwrap_or_else(|e| e.into_inner());
        let evicted = if parked.len() >= MAX_PARKED_WINDOWS {
            Some(parked.remove(0))
        } else {
            None
        };
        parked.push(entry);
        evicted
    };
    if let Some(evicted) = evicted {
        destroy_parked(evicted);
    }
    true
}

/// 把停放的窗口挂回宿主父窗口并显示。
fn adopt_parked(
    parked: ParkedWindow,
    parent: HWND,
    width: i32,
    height: i32,
) -> Result<WindowsHost, String> {
    let hwnd = HWND(parked.hwnd as *mut c_void);
    let attached = (|| -> Result<(), String> {
        let _dpi_scope = HostContextScope::enter(parent);
        // 与新建路径同一把尺子：填满宿主父窗口客户区（见 `attach` 的说明）。
        let mut parent_client = RECT::default();
        let (width, height) = if unsafe { GetClientRect(parent, &mut parent_client) }.is_ok()
            && parent_client.right > 0
            && parent_client.bottom > 0
        {
            (parent_client.right, parent_client.bottom)
        } else {
            (width, height)
        };
        unsafe { SetParent(hwnd, Some(parent)) }.map_err(|e| e.to_string())?;
        unsafe { MoveWindow(hwnd, 0, 0, width, height, true) }.map_err(|e| e.to_string())?;
        let _ = unsafe { ShowWindow(hwnd, SW_SHOWNA) };
        Ok(())
    })();
    if let Err(error) = attached {
        // 已经脱离停放池：挂不回去就销毁，不能让它悬在 message-only 窗口下。
        destroy_parked(parked);
        return Err(error);
    }
    let pointer = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const WindowData;
    if pointer.is_null() {
        destroy_parked(parked);
        return Err("parked window lost its data".into());
    }
    // 页面在停放期间被 `put_IsVisible(false)`；恢复可见并按新父窗口尺寸对齐。
    let port = unsafe { (*pointer).port.clone() };
    port.set_page_visible(true);
    port.sync_bounds(hwnd);
    // 停放前可能留着一张错误提示 STATIC；恢复时它不该继续盖在页面上。
    port.clear_error();
    if unsafe { SetTimer(Some(hwnd), 0x4853, 20, None) } == 0 {
        port.show_error("UI response timer unavailable");
    }
    crate::gui_log_line(&format!(
        "editor view resumed (parked window 0x{:x} reused)",
        parked.hwnd
    ));
    Ok(WindowsHost {
        key: WindowKey {
            hwnd: parked.hwnd,
            thread: parked.thread,
            token: parked.token,
        },
    })
}

/// 销毁所有停放窗口（`ExitDll` 前调用）。
///
/// 【为什么必须显式收尾】它们不在宿主窗口树里，宿主退出时不会替我们销毁；留着它们
/// 会让 `WINDOWS` 永远非零，`ExitDll` 永远被拒（模块无法卸载）。
fn destroy_parked_windows() {
    let entries: Vec<ParkedWindow> = {
        let mut parked = PARKED.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *parked)
    };
    for entry in entries {
        destroy_parked(entry);
    }
}

/// 挂在 `GWLP_USERDATA` 上的窗口数据：它是**唯一的**"从 HWND 找回判定状态"的入口。
///
/// 【为什么仍然用窗口属性而不是全局表】`WM_NCDESTROY` 是系统给的、与窗口同寿的析构
/// 点；用它释放数据，就不存在"窗口没了而表里还留着"的悬垂项。
struct WindowData {
    client: Rc<RefCell<EditorClient>>,
    port: Rc<WindowsPort>,
    capturered: Rc<Cell<bool>>,
    token: usize,
    /// popup 形态的几何参照：宿主交给 `IPlugView::attached` 的**父窗口**
    /// （ACID 的 `sfvstwrap` 子窗口；子窗口形态为默认值）。
    ///
    /// 【为什么不是顶层祖先】顶层祖先是宿主整个插件窗口（含 FX 链、预设栏等
    /// 宿主自有控件），popup 覆盖它的客户区会把宿主控件一起盖住 —— 现场表现
    /// 就是"插件替换掉了宿主窗口"。几何参照必须是宿主**留给插件的那块矩形**；
    /// 顶层祖先只用作 popup 的 owner（z 序/最小化跟随），两者不是同一个窗口。
    ///
    /// 【为什么放这里】同步发生在 20 ms 定时器上，那一条只拿 `WindowData`；
    /// 参照物属于"窗口的事实"，随窗口数据同生命周期最自然。
    popup_anchor: Cell<HWND>,
}
impl Drop for WindowData {
    fn drop(&mut self) {
        // 【顺序】先让判定层收尾（撤销本视图持有的宿主 Undo 记账、停掉出站），
        // 再关 COM。反过来的话，关闭过程中的迟到回调会撞上一个已经宣称关闭的状态。
        self.client.borrow_mut().close();
        self.port.close();
        let _ = unsafe { KillTimer(Some(self.port.hwnd.get()), 0x4853) };
        // 【为什么不在这里删用户数据目录】目录现在是**每进程**一份、被本进程的其它视图
        // 共享（见 `begin_browser`），关掉一个视图不能把还在用的那份删掉。回收改由
        // `temp_sweep::sweep_stale_temp_dirs` 在下一次启动时按"属主进程已退出"进行。
        unsafe {
            CoUninitialize();
        }
    }
}

/// WebView2 的出站端口：页面还没建好时 `post_message` 返回 `false`。
pub(super) struct WindowsPort {
    /// 建窗之后才填（`WindowData` 必须先于 `CreateWindowExW` 存在）。
    hwnd: Cell<HWND>,
    /// 控制器是**异步**到的；它到达之前一切投递都只能报"投不出去"。
    controller: RefCell<Option<ICoreWebView2Controller>>,
    /// 最近一次真的交给 WebView2 的（物理宽、物理高、rasterization scale 位）。
    ///
    /// 【为什么缓存】`WM_SIZE` 在宿主拖动面板时每像素来一次；尺寸没变就不要再
    /// `SetBounds`/`SetRasterizationScale`，否则每次重排都要多一次跨进程 COM。
    /// 控制器重建时 `attach` 会清空它（新控制器必须重新收一遍边界）。
    bounds: Cell<Option<(i32, i32, u64)>>,
    /// 错误提示用的 `STATIC` 子窗口（自有 child 没有标题栏，不能只写 window text）。
    error_window: RefCell<Option<HWND>>,
}
impl EditorPort for WindowsPort {
    fn post_message(&self, text: &str) -> bool {
        let controller = self.controller.borrow().clone();
        let Some(controller) = controller else {
            return false;
        };
        let Ok(browser) = (unsafe { controller.CoreWebView2() }) else {
            return false;
        };
        let text = wide(text);
        unsafe { browser.PostWebMessageAsJson(PCWSTR(text.as_ptr())) }.is_ok()
    }

    fn show_error(&self, text: &str) {
        let hwnd = self.hwnd.get();
        let text = wide(&format!("HiFiShifter editor: {text}"));
        let existing = *self.error_window.borrow();
        if let Some(existing) = existing {
            let _ = unsafe { SetWindowTextW(existing, PCWSTR(text.as_ptr())) };
            return;
        }
        // 自有 child 没有标题栏；用真实 STATIC 控件显示错误，不能只写不可见 window text。
        // 【为什么切上下文】错误框的坐标按调用线程解释 —— 与包裹层不一致时会把
        // 它放大 1.25×，正好在"界面已经坏了"的时刻再添一层错位（见 `HostContextScope`）。
        let _dpi_scope = HostContextScope::enter(context_reference(hwnd));
        let created = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("STATIC"),
                PCWSTR(text.as_ptr()),
                WS_CHILD | WS_VISIBLE,
                12,
                12,
                600,
                120,
                Some(hwnd),
                None,
                None,
                None,
            )
        };
        drop(_dpi_scope);
        if let Ok(window) = created {
            *self.error_window.borrow_mut() = Some(window);
        }
    }

    fn close(&self) {
        // Close 会同步回调；此时不持有任何 `RefCell` 借用。
        let controller = self.controller.borrow_mut().take();
        if let Some(controller) = controller {
            let _ = unsafe { controller.Close() };
        }
    }
}

/// 只保存窗口身份，不把非 Send COM 接口从 UI 线程带走。资源由自有 WndProc 析构。
pub(super) struct WindowsHost {
    key: WindowKey,
}
#[derive(Clone, Copy)]
pub(super) struct WindowKey {
    hwnd: usize,
    thread: u32,
    token: usize,
}
impl WindowKey {
    /// 句柄被系统复用后不能移动/关闭另一实例窗口；属性 token 在 `WM_NCDESTROY` 时撤销。
    fn valid(self) -> bool {
        let hwnd = HWND(self.hwnd as *mut c_void);
        unsafe {
            IsWindow(Some(hwnd)).as_bool()
                && GetPropW(hwnd, w!("HiFiShifter.ARA.ViewToken")).0 as usize == self.token
        }
    }
    /// 从窗口属性取回判定状态（**只克隆 `Rc`**，不返回指向窗口数据的引用）。
    fn client(self) -> Option<Rc<RefCell<EditorClient>>> {
        if !self.valid() {
            return None;
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let pointer = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const WindowData;
        if pointer.is_null() {
            return None;
        }
        Some(unsafe { (*pointer).client.clone() })
    }
    /// 只在宿主 UI 线程把焦点交给自有编辑器；已在浏览器内时不重置 DOM 焦点。
    fn focus(self) -> Result<(), String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("focus on stale window or off UI thread".into());
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let current = unsafe { GetFocus() };
        if current == hwnd || unsafe { IsChild(hwnd, current) }.as_bool() {
            return Ok(());
        }
        let result = unsafe { SetFocus(Some(hwnd)) };
        // 首次 SetFocus 的旧句柄可能为 null，Win32 包装返回 Err 但真实焦点已成功设置。
        let focused = unsafe { GetFocus() };
        if focused == hwnd || unsafe { IsChild(hwnd, focused) }.as_bool() {
            Ok(())
        } else {
            result.map(|_| ()).map_err(|e| e.to_string())
        }
    }
    /// popup 形态的几何参照（子窗口形态为 `None`）。
    fn popup_anchor(self) -> Option<HWND> {
        if !self.valid() {
            return None;
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        let pointer = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const WindowData;
        if pointer.is_null() {
            return None;
        }
        // SAFETY: pointer 来自窗口属性，窗口存活期间有效（valid() 已验证）。
        let anchor = unsafe { (*pointer).popup_anchor.get() };
        (!anchor.0.is_null()).then_some(anchor)
    }
    /// 可复制的窗口身份不持有 COM 资源，允许调用前释放 view 锁以防 Win32 同步重入。
    fn resize(self, width: i32, height: i32) -> Result<(), String> {
        if unsafe { GetCurrentThreadId() } != self.thread || !self.valid() {
            return Err("resize on stale window or off UI thread".into());
        }
        let hwnd = HWND(self.hwnd as *mut c_void);
        // 【popup 形态不按宿主数字摆放】宿主 onSize 传来的数字属于宿主的坐标语义；
        // popup 的唯一正确尺寸是**宿主留给插件那块矩形**的物理客户区 —— 同步一次即可。
        if let Some(anchor) = self.popup_anchor() {
            sync_popup_to_anchor(hwnd, anchor);
            return Ok(());
        }
        // 【为什么按父窗口客户区收口，而不是宿主给的 (width, height)】嵌入子窗口的
        // 正确尺寸是"填满父窗口客户区"。宿主给的数字在**宿主的上下文**里有意义，
        // 换一层上下文就换一种长度：ACID 的包裹窗在不感知上下文里把感知宿主按
        // 物理像素留出的客户区虚拟化成 96 值，`onSize` 传下来的数字因此比实际
        // 客户区大 1.25× —— 按它移动，子窗口必然溢出（实测：宿主滚动条、右/下被裁；
        // 2026-11 第一份现场截图正是这个形状）。`fit_child_to_parent` 在守卫上下文
        // 里量父窗口客户区，量出的数字与 MoveWindow 的解释**同一把尺子**；感知宿主
        // （REAPER / VEGAS）里父窗口客户区与宿主数字一致，行为不变。
        fit_child_to_parent(hwnd, Some((width, height))).map(|_| ())
    }
}
impl EditorHost for WindowsHost {
    fn fidelity(&self) -> GuiFidelity {
        GuiFidelity::Embedded
    }

    fn resize(&self, width: i32, height: i32) -> Result<(), String> {
        self.key.resize(width, height)
    }

    fn focus(&self) -> Result<(), String> {
        self.key.focus()
    }

    /// 只向真实活 WebView 交付共享编辑键；不向外层 HWND 伪造无效 Win32 键盘消息。
    fn forward_edit_key(&self, key: char, mods: i16, down: bool) -> bool {
        if unsafe { GetCurrentThreadId() } != self.key.thread {
            return false;
        }
        let Some(client) = self.key.client() else {
            return false;
        };
        let handled = client.borrow_mut().forward_edit_key(key, mods, down);
        handled
    }

    /// 20 ms 定时器本来就在宿主 UI 线程上，不需要"补做"。
    fn host_tick(&self) {}

    fn close(&self) {
        // 析构由 `Drop` 负责（它知道该 DestroyWindow 还是 PostMessage 给窗口线程）。
    }
}
impl Drop for WindowsHost {
    fn drop(&mut self) {
        let hwnd = HWND(self.key.hwnd as *mut c_void);
        if !self.key.valid() {
            return;
        }
        if unsafe { GetCurrentThreadId() } != self.key.thread {
            // COM 由自有窗口线程 Close，不在错误线程释放。正常 REAPER UI 析构走上分支。
            let _ = unsafe { PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0)) };
            return;
        }
        // 【先试停放】宿主关面板 ≠ 插件实例销毁：同一个控制器很快会再 attached
        // 一个新视图（同一个 `EditorLink`）。把 HWND+控制器+页面停放到 message-only
        // 窗口下，下一次 attached 复用 —— 避免 WebView2 在"控制器刚关闭"时创建新
        // 控制器的 ~5.6 s 冷启动（VEGAS 现场白屏）。popup / 页面已死 / 挂不走时
        // 仍走销毁路径。
        if park_window(hwnd, self.key.thread, self.key.token) {
            return;
        }
        let _ = unsafe { DestroyWindow(hwnd) };
    }
}

/// 获取本DLL而非REAPER模块；不依据项目CWD定位资产。
fn module() -> Result<HMODULE, String> {
    let mut module = HMODULE::default();
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(module_address as *const () as *const u16),
            &mut module,
        )
    }
    .map_err(|e| e.to_string())?;
    Ok(module)
}
fn module_address() {}
/// WebView2的环境创建没有取消API，迟到COM回调仍会进入本DLL；固定模块到宿主退出。
/// 只固定这一模块一次，不保留窗口/浏览器/worker；开发升级需要正常退出REAPER。
fn pin_callback_code() -> Result<(), String> {
    WEBVIEW_MODULE_PIN
        .get_or_init(|| {
            let mut handle = HMODULE::default();
            unsafe {
                GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    PCWSTR(module_address as *const () as *const u16),
                    &mut handle,
                )
            }
            .map_err(|e| e.to_string())
        })
        .clone()
}
/// 前端产物目录：与 macOS/Linux **同一份**实现（`editor::resources`）。
///
/// 【为什么不在这里再写一遍 `GetModuleFileNameW`】"资源在哪"只该有一个答案；
/// 两份实现里只要有一份忘了往上两级，另一个平台就会白屏，而日志里看不出来。
fn assets() -> Result<PathBuf, String> {
    crate::editor::resources::frontend_directory()
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

fn register_class() -> Result<(), String> {
    let mut registered = CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if *registered {
        return Ok(());
    }
    let class = WNDCLASSW {
        lpfnWndProc: Some(window_proc),
        hInstance: HINSTANCE(module()?.0),
        lpszClassName: w!("HiFiShifter.ARA.Editor"),
        // 【深色加载底】控制器创建到首帧之间 WebView2 画的是默认白底；窗口类背景
        // 与 `SetDefaultBackgroundColor` 用同一个颜色，等待期不再闪白屏。
        hbrBackground: unsafe { CreateSolidBrush(EDITOR_BACKGROUND) },
        ..Default::default()
    };
    if unsafe { RegisterClassW(&class) } == 0 {
        // windows 0.62 把 `Error::from_win32` 改名为 `from_thread`（语义不变：
        // 取本线程的 GetLastError）。
        return Err(windows::core::Error::from_thread().to_string());
    }
    *registered = true;
    Ok(())
}
/// 不留指向已卸载DLL的window class；打开中的child明确阻止正常卸载。
pub(super) fn shutdown() -> bool {
    // 【先收停放窗口】它们不在宿主窗口树里，宿主退出时不会替我们销毁；留着它们
    // 会让 `WINDOWS` 永远非零，`ExitDll` 永远被拒（模块无法卸载）。
    destroy_parked_windows();
    if WINDOWS.load(Ordering::Acquire) != 0 {
        crate::gui_log_line("ExitDll refused: native editor windows still alive");
        return false;
    }
    let mut registered = CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if *registered {
        let Ok(module) = module() else {
            return false;
        };
        if unsafe { UnregisterClassW(w!("HiFiShifter.ARA.Editor"), Some(HINSTANCE(module.0))) }
            .is_err()
        {
            return false;
        }
        *registered = false;
    }
    true
}

impl WindowsHost {
    /// 同线程创建子窗口，异步浏览器由宿主消息循环完成，禁止嵌套消息泵。
    pub(super) fn attach(
        parent: NativeWindow,
        width: i32,
        height: i32,
        link: Arc<crate::editor::routing::EditorLink>,
    ) -> Result<Self, String> {
        if let Ok(owner) = link.owner() {
            owner.startup_stage(crate::startup::Stage::Attached);
        }
        let parent = HWND(parent as *mut c_void);
        if !unsafe { IsWindow(Some(parent)) }.as_bool() {
            return Err("parent is not a live HWND".into());
        }
        let thread = unsafe { GetCurrentThreadId() };
        if unsafe { GetWindowThreadProcessId(parent, None) } != thread {
            return Err("IPlugView attached off the parent UI thread".into());
        }
        register_class()?;
        // 【为什么在创建前记 DPI 事实】父窗口的感知上下文决定"尺寸数字到底落在
        // 多少物理像素上"；没有这一行，ACID 的滚动条/模糊只能靠猜（见
        // `HostContextScope` 的文档）。
        log_host_dpi_facts(parent);
        // 【复用优先】同一个宿主控制器重开面板时，停放窗口里已经有活的 HWND /
        // WebView2 控制器 / 页面。必须在 `CoInitializeEx` 与新建 client 之前分流：
        // 复用路径不新建任何东西（COM 也已经在那个窗口的线程上初始化过）。
        if let Some(parked) = take_parked_window(&link, parent) {
            match adopt_parked(parked, parent, width, height) {
                Ok(host) => return Ok(host),
                Err(error) => crate::gui_log_line(&format!(
                    "parked editor window unusable ({error}); creating a fresh one"
                )),
            }
        }
        let instance = HINSTANCE(module()?.0);
        let initialized = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        initialized
            .ok()
            .map_err(|e| format!("WebView2 STA unavailable: {e}"))?;
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) as usize;
        // 判定状态先建：窗口数据要持有它，而窗口数据在 CreateWindowExW 之前就要存在
        // （WM_NCCREATE 会立刻取它）。
        let client = EditorClient::open(link);
        let port = Rc::new(WindowsPort {
            hwnd: Cell::new(HWND::default()),
            controller: RefCell::new(None),
            bounds: Cell::new(None),
            error_window: RefCell::new(None),
        });
        let capturered = Rc::new(Cell::new(false));
        let data = Box::into_raw(Box::new(WindowData {
            client: client.clone(),
            port: port.clone(),
            capturered: capturered.clone(),
            token,
            popup_anchor: Cell::new(HWND::default()),
        }));
        // ── 默认形态：宿主父窗口里的普通子窗口 ──────────────────────────────
        //
        // 【为什么默认不再 popup】宿主交给 `attached` 的父窗口就是"插件的那块矩形"：
        // 子窗口天然跟随宿主的裁剪、移动、缩放与 z 序，这是插件窗口唯一可靠的形态。
        // 曾经的默认 popup 只在"包裹窗 DPI-unaware + 屏幕缩放 > 1"（ACID 的
        // `sfvstwrap`）时启用，用独立顶层 surface 换清晰度；但顶层窗口不被宿主裁剪、
        // 也不随宿主布局 —— 现场（2026-11-22）就是覆盖宿主 FX 链/预设栏、宿主缩放
        // 面板时停在原地，用户连删除 FX 都做不到。这比模糊更不可接受。
        //
        // 【模糊的归属】包裹窗不感知时，DWM 把整棵窗口树（含子窗口）按屏幕缩放拉伸；
        // 任何子窗口形态都共享这个 surface。它不是本插件能修的 —— 宿主自己创建的
        // 包裹窗决定了 surface 的 DPI。要清晰度就得离开窗口树，而离开窗口树就不再是
        // 插件窗口。因此默认选宿主集成，popup 只作为显式逃生门
        // （`HIFISHIFTER_DPI_POPUP=1`，见 `dpi_popup_requested`）。
        // 【但尺寸必须与它同尺】模糊可以忍，"内容被放大两轮、右/下永远滚不到"不行：
        // WebView 的 raw surface 必须按窗口坐标下发（见 `content_geometry`），
        // 交给 DWM 只拉伸一次。
        //
        // 【子窗口的尺寸纪律】`resize` 与 20 ms 定时器都把子窗口收口到**父窗口客户区**
        // （见 `fit_child_to_parent`）：宿主给的 `onSize` 数字属于宿主的坐标语义，
        // 包裹层与宿主上下文不一致时按它摆放会溢出；量父窗口客户区与 `MoveWindow`
        // 是同一把尺子。定时器兜底还覆盖"宿主缩放了包裹窗却没调用 `onSize`"的宿主。
        if dpi_popup_requested() {
            if let Some(scale) = popup_scale_for(parent) {
                return Self::attach_popup(
                    parent, data, port, client, capturered, scale, token, thread, instance, width,
                    height,
                );
            }
        }
        // 【为什么这里切上下文】子窗口的感知上下文在创建时定死；与父窗口不一致的
        // 线程创建出的子窗口，坐标语义就和包裹层对不上 —— 滚动条 + 裁切 + 模糊
        // 的根因（见 `HostContextScope`）。守卫覆盖 WM_NCCREATE/WM_CREATE 的同步
        // 送达；宿主的上下文在守卫析构后原样还原。
        let _dpi_scope = HostContextScope::enter(parent);
        // 【为什么按父窗口客户区定初始尺寸】`getSize` 协商出的 (width, height) 属于
        // 宿主的上下文；包裹层与宿主上下文不一致时（ACID：感知宿主 + 不感知包裹窗），
        // 按它创建的子窗口会被系统再放大 1.25× —— 2026-11 实测的"宿主滚动条 +
        // 右/下被裁"正是这个错位。守卫上下文里量出的父窗口客户区与 CreateWindow
        // 的解释同尺；感知宿主里它与宿主给的数字一致，行为不变。
        let mut parent_client = RECT::default();
        let (create_width, create_height) = if unsafe { GetClientRect(parent, &mut parent_client) }
            .is_ok()
            && parent_client.right > 0
            && parent_client.bottom > 0
        {
            (parent_client.right, parent_client.bottom)
        } else {
            (width, height)
        };
        let result = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("HiFiShifter.ARA.Editor"),
                w!("HiFiShifter plugin editor initializing…"),
                WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                0,
                0,
                create_width,
                create_height,
                Some(parent),
                None,
                Some(instance),
                Some(data.cast()),
            )
        };
        drop(_dpi_scope);
        // 【几何现场】不感知包裹层里客户区是窗口坐标（1100×720），DWM 会把它
        // 按屏幕缩放合成为物理值（125% → 1375×900）；感知包裹层里客户区就是
        // 物理值。这两类窗口的 WebView 内容面不同（见 `content_geometry`），
        // 所以把判据与屏幕缩放一起记下来。
        crate::gui_log_line(&format!(
            "editor embed: parent-client={create_width}x{create_height} \
             parent-unaware={} display-scale={:.2}",
            window_is_unaware(parent),
            monitor_scale_for(parent).unwrap_or(1.0)
        ));
        let hwnd = match result {
            Ok(hwnd) => hwnd,
            Err(error) => {
                // WM_NCCREATE 之前失败时系统未接管 data；之后失败则 WM_NCDESTROY 已负责释放。
                if !capturered.get() {
                    unsafe {
                        drop(Box::from_raw(data));
                    }
                }
                return Err(format!("create child window: {error}"));
            }
        };
        port.hwnd.set(hwnd);
        client.borrow_mut().set_window(hwnd.0 as usize);
        let host = Self {
            key: WindowKey {
                hwnd: hwnd.0 as usize,
                thread,
                token,
            },
        };
        if unsafe { SetTimer(Some(hwnd), 0x4853, 20, None) } == 0 {
            port.show_error("UI response timer unavailable");
        }
        if let Err(error) = begin_browser(&client, &port) {
            port.show_error(&error);
            // attach 半失败：没有可投递的 browser，泵无事可做、视图已死，
            // 20ms 定时器只会空转（50Hz 全速泵一块错误 STATIC）。停掉它。
            unsafe {
                let _ = KillTimer(Some(hwnd), 0x4853);
            }
        }
        Ok(host)
    }

    /// popup 形态的附着（**显式 opt-in**，见 `dpi_popup_requested`）：顶层窗口 +
    /// 逐拍同步到**宿主留给插件那块矩形**的物理客户区。
    ///
    /// 【为什么初始尺寸不取 (width, height)】那两个数是宿主 `getSize` 协商出的，属于
    /// **宿主**的坐标语义；popup 按物理像素摆放，唯一的正确初始值就是宿主父窗口
    /// （插件区域）客户区的物理矩形。attach 这一刻宿主可能还没定尺寸 —— 拿不到就先以
    /// 协商值近似（按屏幕缩放换算成物理），20 ms 定时器会在下一拍把它校正到真值。
    ///
    /// 【owner 与几何参照必须分开】owner 是**顶层祖先**（popup 跟着宿主最小化/还原、
    /// z 序在宿主之上）；几何参照是宿主交给 `attached` 的**父窗口**。ACID 的父窗口是
    /// 顶层窗口里的 `sfvstwrap` 子窗口，顶层窗口还画着宿主自己的 FX 链/预设栏 ——
    /// 用顶层祖先的客户区摆 popup，就会把宿主控件一起盖住（用户报"插件替换了宿主
    /// 窗口，连删除 FX 都做不到"）。
    #[allow(clippy::too_many_arguments)]
    fn attach_popup(
        parent: HWND,
        data: *mut WindowData,
        port: Rc<WindowsPort>,
        client: Rc<RefCell<EditorClient>>,
        capturered: Rc<Cell<bool>>,
        scale: f64,
        token: usize,
        thread: u32,
        instance: HINSTANCE,
        width: i32,
        height: i32,
    ) -> Result<Self, String> {
        // owner 只负责窗口归属（z 序/最小化）；几何一律看 anchor。
        let root = unsafe { GetAncestor(parent, GA_ROOT) };
        let owner = if root.0.is_null() || root == parent {
            parent
        } else {
            root
        };
        let anchor = parent;
        let physical = wrapper_physical_client(anchor);
        let (x, y, width, height) = physical
            .map(|rect| {
                (
                    rect.left,
                    rect.top,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                )
            })
            .unwrap_or_else(|| {
                // 协商值属于宿主上下文；屏幕缩放>1 时它按物理像素理解，换算成 popup
                // 的物理像素再摆 —— 误差由下一拍同步收敛。
                let scaled = ((width as f64) * scale).round().max(1.0) as i32;
                let scaled_height = ((height as f64) * scale).round().max(1.0) as i32;
                let mut origin = POINT::default();
                {
                    let _scope = HostContextScope::force_pm_v2();
                    // 换算失败只影响这一拍的近似位置；下一拍同步会收敛到真值。
                    let _ = unsafe { ClientToScreen(anchor, &mut origin) };
                }
                (origin.x, origin.y, scaled, scaled_height)
            });
        crate::gui_log_line(&format!(
            "editor popup: anchor=0x{:x} owner=0x{:x} scale={scale:.2} rect=({x},{y},{width},{height})",
            anchor.0 as usize, owner.0 as usize
        ));
        // 【为什么这里强制 PMv2】popup 是顶层窗口，感知上下文不跟包裹层走：PMv2 让
        // WebView2 以 devicePixelRatio=屏幕缩放 渲染物理像素，DWM 不再拉伸 —— 模糊
        // 由此消失。创建与摆放必须在同一上下文里。
        let _dpi_scope = HostContextScope::force_pm_v2();
        let result = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(WS_EX_TOOLWINDOW.0),
                w!("HiFiShifter.ARA.Editor"),
                w!("HiFiShifter plugin editor initializing…"),
                WS_POPUP | WS_VISIBLE | WS_CLIPCHILDREN,
                x,
                y,
                width,
                height,
                Some(owner),
                None,
                Some(instance),
                Some(data.cast()),
            )
        };
        drop(_dpi_scope);
        let hwnd = match result {
            Ok(hwnd) => hwnd,
            Err(error) => {
                // WM_NCCREATE 之前失败时系统未接管 data；之后失败则 WM_NCDESTROY 已负责释放。
                if !capturered.get() {
                    unsafe {
                        drop(Box::from_raw(data));
                    }
                }
                return Err(format!("create editor popup: {error}"));
            }
        };
        // SAFETY: data 由本函数分配；WM_NCCREATE 之前写入是安全的（窗口尚未取走所有权）。
        unsafe { (*data).popup_anchor.set(anchor) };
        port.hwnd.set(hwnd);
        client.borrow_mut().set_window(hwnd.0 as usize);
        let host = Self {
            key: WindowKey {
                hwnd: hwnd.0 as usize,
                thread,
                token,
            },
        };
        if unsafe { SetTimer(Some(hwnd), 0x4853, 20, None) } == 0 {
            port.show_error("UI response timer unavailable");
        }
        if let Err(error) = begin_browser(&client, &port) {
            port.show_error(&error);
            unsafe {
                let _ = KillTimer(Some(hwnd), 0x4853);
            }
        }
        Ok(host)
    }
}

/// 把嵌入形态的子窗口收口到**宿主父窗口客户区**；`Ok(true)` = 发生了摆放。
///
/// 【为什么是父窗口客户区，而不是宿主 `onSize` 的数字】两者在宿主自己的上下文里
/// 一致，但在"感知宿主 + 不感知包裹窗"的混合形态里不是同一把尺子（ACID 现场）：
/// `onSize` 的数字会被系统按另一层上下文解释，按它摆放必然溢出（宿主滚动条 +
/// 右/下被裁）。在父窗口的上下文里量客户区、在同一个上下文里 `MoveWindow`，
/// 量多大就占多大。
///
/// 【为什么定时器也要做】`WM_SIZE` 只会在**我们自己的**窗口尺寸变化时到来；宿主
/// 缩放了包裹窗却没调用 `IPlugView::onSize`（或调用晚于布局）时，子窗口就停在
/// 旧尺寸上。20 ms 一拍的空查询是两次 `GetClientRect`，只有真变了才 `MoveWindow`。
///
/// `fallback` 是"量不到父窗口客户区"时的退路（`resize` 用宿主给的协商值；定时器
/// 不传，直接放弃这一拍）。
fn fit_child_to_parent(hwnd: HWND, fallback: Option<(i32, i32)>) -> Result<bool, String> {
    let parent = match unsafe { GetParent(hwnd) } {
        Ok(parent) if !parent.0.is_null() => parent,
        _ => {
            return match fallback {
                Some((width, height)) => unsafe { MoveWindow(hwnd, 0, 0, width, height, true) }
                    .map(|_| true)
                    .map_err(|error| error.to_string()),
                None => Ok(false),
            };
        }
    };
    // 子窗口的感知上下文在创建时已按父窗口匹配（见 `attach`），这里再切一次是为了
    // 让"量客户区"与"MoveWindow"都落在与父窗口一致的坐标语义里（宿主线程的上下文
    // 可能已被宿主改过）。
    let _dpi_scope = HostContextScope::enter(parent);
    let mut client = RECT::default();
    let (width, height) = if unsafe { GetClientRect(parent, &mut client) }.is_ok()
        && client.right > 0
        && client.bottom > 0
    {
        (client.right, client.bottom)
    } else if let Some(size) = fallback {
        size
    } else {
        return Ok(false);
    };
    let mut current = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut current) }.is_ok()
        && current.right == width
        && current.bottom == height
    {
        return Ok(false);
    }
    unsafe { MoveWindow(hwnd, 0, 0, width, height, true) }
        .map(|_| true)
        .map_err(|error| error.to_string())
}

/// 一次窗口测量得出的 **WebView 内容面**：窗口客户区、交给 `SetBounds` 的原始
/// 像素面、交给 `SetRasterizationScale` 的比例。
struct EditorGeometry {
    /// 窗口坐标里的客户区（只用于诊断日志）。
    client: (i32, i32),
    /// 交给 `SetBounds` 的原始像素面（`USE_RAW_PIXELS` 的含义见 [`content_geometry`]）。
    content: (i32, i32),
    /// 交给 `SetRasterizationScale` 的比例。
    scale: f64,
    /// 窗口是否 DPI-unaware。
    unaware: bool,
}

/// 由测量值决定 WebView 的内容面 —— **判据只有一条：这块 surface 会不会被 DWM 拉伸**。
///
/// 【DPI-unaware 窗口：内容面 = 窗口坐标，缩放 = 1】DWM 把整棵窗口树按屏幕缩放
/// 拉伸。ACID 的 `sfvstwrap` 就是这种：窗口坐标里客户区 1100×720，屏幕上合成为
/// 1375×900（125%）。WebView 的 surface 是这棵树的一部分，它的“raw pixel”就是
/// **窗口坐标里的像素**，不是屏幕上的物理像素。把 PMv2 量到的 1375×900 交给它，
/// surface 会比窗口大 25%，DWM 再拉伸一次 —— 于是内容被放大两轮、状态栏被推出
/// 窗口外，滚到最右下也看不见（2026-11-22 第二次打开 ACID 的现场）。缩放固定 1.0：
/// 窗口自己的 DPI 就是 96（unaware 的 `GetDpiForWindow` 恒回 96），屏幕缩放由 DWM
/// 在窗口外完成；再乘一次 1.25 只会让文字再次变大。
///
/// 【DPI-aware 窗口：内容面 = 物理客户区，缩放 = 显示器缩放】窗口自己按物理像素
/// 合成，DWM 不再拉伸。`USE_RAW_PIXELS` 要的正是物理值，rasterization scale 取
/// 显示器缩放，页面拿到 `物理 / 缩放` 的 CSS 视口（第一次打开 ACID：1100×720 物理
/// → 880×576 CSS）。
fn content_geometry(
    client: (i32, i32),
    physical: (i32, i32),
    unaware: bool,
    monitor_scale: f64,
) -> EditorGeometry {
    if unaware {
        return EditorGeometry {
            client,
            content: client,
            scale: 1.0,
            unaware,
        };
    }
    EditorGeometry {
        client,
        content: physical,
        scale: if monitor_scale > 0.0 {
            monitor_scale
        } else {
            1.0
        },
        unaware,
    }
}

/// 量一次编辑器窗口，按它的感知上下文给出 WebView 该铺多大的内容面。
///
/// 两种窗口读同一组 `GetClientRect` 会得到不同含义的值，所以**先判感知上下文，
/// 再决定要不要读物理值**：unaware 窗口的物理值（PMv2 里量到的 1375×900）恰恰
/// 是绝不能交给 WebView 的那一个。
fn editor_geometry(window: HWND) -> Option<EditorGeometry> {
    // 【窗口自己的坐标】不感知窗口里是 96-DPI 逻辑值；感知窗口里就是物理值。
    let mut client = RECT::default();
    {
        let _scope = HostContextScope::enter(window);
        if unsafe { GetClientRect(window, &mut client) }.is_err() {
            return None;
        }
    }
    if client.right <= 0 || client.bottom <= 0 {
        return None;
    }
    let client = (client.right, client.bottom);
    if window_is_unaware(window) {
        // 不感知窗口：内容面与窗口同尺，屏幕缩放交给 DWM —— 不能再读物理值。
        return Some(content_geometry(client, client, true, 1.0));
    }
    // 【感知窗口的物理值】窗口自己按物理像素合成；从 PMv2 再量一次覆盖宿主线程
    // 上下文被临时改过的情形。逃生门（`HIFISHIFTER_NO_DPI_AWARE=1`）下退回当前
    // 上下文，行为与改动前一致。
    let mut physical = RECT::default();
    {
        let _scope = HostContextScope::force_pm_v2();
        if unsafe { GetClientRect(window, &mut physical) }.is_err() {
            return None;
        }
    }
    if physical.right <= 0 || physical.bottom <= 0 {
        return None;
    }
    let physical = (physical.right, physical.bottom);
    Some(content_geometry(
        client,
        physical,
        false,
        monitor_scale_for(window).unwrap_or(1.0),
    ))
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    /// 不感知窗口的内容面必须是**窗口坐标**，不能是 PMv2 量到的物理值。
    ///
    /// 【为什么这条是回归】把 1375×900 交给 WebView 后，DWM 会把这块 surface 再
    /// 按 1.25× 拉伸 —— 用户看到的是"内容又放大一圈、填满窗口，状态栏被推出窗口
    /// 外，滚到最右下也看不见"（ACID 第二次打开，2026-11-22）。内容面与窗口同尺
    /// 之后，DWM 只拉伸一次：文字回到与第一次打开相同的物理大小，底部与右侧也能
    /// 被宿主滚动条完整取到。
    #[test]
    fn an_unaware_window_sizes_the_view_in_window_coordinates() {
        let geometry = content_geometry((1100, 720), (1375, 900), true, 1.25);
        assert_eq!(
            geometry.content,
            (1100, 720),
            "不感知窗口的 raw surface 是窗口坐标，不是屏幕物理值"
        );
        assert_eq!(
            geometry.scale, 1.0,
            "屏幕缩放由 DWM 完成，WebView 再乘一次就是二次放大"
        );
    }

    /// 感知窗口没有 DWM 拉伸：内容面取物理客户区，缩放取显示器缩放。
    ///
    /// 【为什么让两个输入不同】这样断言才区分得出实现选了哪一个；真实 PMv2
    /// 窗口里两者相等，选错也看不出来。
    #[test]
    fn an_aware_window_sizes_the_view_in_physical_pixels() {
        let geometry = content_geometry((880, 576), (1100, 720), false, 1.25);
        assert_eq!(geometry.content, (1100, 720), "感知窗口按物理像素合成");
        assert_eq!(geometry.scale, 1.25, "rasterization scale 跟随显示器");
    }

    /// 拿不到显示器缩放时退回 1.0，不把 0 当作比例交给 WebView。
    #[test]
    fn a_missing_monitor_scale_falls_back_to_one() {
        let geometry = content_geometry((1100, 720), (1100, 720), false, 0.0);
        assert_eq!(geometry.scale, 1.0);
    }
}

/// 把 popup 逐拍摆回**宿主留给插件那块矩形**的物理客户区；返回是否发生了摆放。
///
/// 【为什么是"变了才摆"】SetWindowPos 即使矩形不变也会打扰窗口（窗口消息、可能
/// 的重绘），20 ms 一拍的空摆是纯损耗；先比较，只在真变了的时候动。
///
/// 【为什么可见性也跟随】宿主最小化/隐藏面板时，popup 不在 owner 的窗口树里，
/// 系统不会自动隐藏它 —— 不跟随就是"宿主窗口没了，插件还浮在桌面上"。
fn sync_popup_to_anchor(hwnd: HWND, anchor: HWND) -> bool {
    let Some(rect) = wrapper_physical_client(anchor) else {
        return false;
    };
    let shown = unsafe { IsWindowVisible(anchor) }.as_bool();
    let _scope = HostContextScope::force_pm_v2();
    // SAFETY: 只读查询；popup 与 anchor 都是活窗口（定时器回调里已验证）。
    let mut current = RECT::default();
    let same = unsafe { GetWindowRect(hwnd, &mut current) }.is_ok()
        && current.left == rect.left
        && current.top == rect.top
        && current.right == rect.right
        && current.bottom == rect.bottom;
    if !same {
        unsafe {
            let _ = SetWindowPos(
                hwnd,
                None,
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
                SWP_NOACTIVATE | SWP_NOZORDER | SWP_NOOWNERZORDER,
            );
        }
    }
    let visible = unsafe { IsWindowVisible(hwnd) }.as_bool();
    if shown && !visible {
        unsafe {
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        }
    } else if !shown && visible {
        unsafe {
            let _ = ShowWindow(hwnd, SW_HIDE);
        }
    }
    true
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // 所有可能 panic 的自有逻辑都在边界内，不能越过 Win32 回调栈。
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if message == WM_NCCREATE {
            let create = &*(lparam.0 as *const CREATESTRUCTW);
            let data = &*(create.lpCreateParams as *const WindowData);
            data.capturered.set(true);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, create.lpCreateParams as isize);
            WINDOWS.fetch_add(1, Ordering::AcqRel);
            if SetPropW(
                hwnd,
                w!("HiFiShifter.ARA.ViewToken"),
                Some(HANDLE(data.token as *mut c_void)),
            )
            .is_err()
            {
                return LRESULT(0);
            }
        }
        let pointer = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowData;
        if !pointer.is_null() {
            if message == WM_TIMER && wparam.0 == 0x4853 {
                // 宿主 getter 可能重入并关闭窗口：先持 Rc，不再在调用后解引用 WindowData。
                let client = (*pointer).client.clone();
                // 【几何参照也在 tick 之前取】tick 可能重入宿主并销毁本窗口
                // （WM_NCDESTROY 会释放 WindowData）；之后再解引用 pointer 就是 UAF。
                let anchor = (*pointer).popup_anchor.get();
                client::tick(&client);
                // 【几何跟随】popup 不在宿主窗口树里，宿主移动/缩放面板不会通知
                // 我们 —— 参照**宿主留给插件那块矩形**的物理客户区，把 popup 摆回去
                // （见 `sync_popup_to_anchor` 的"变了才摆"纪律）。嵌入形态是宿主窗口
                // 树里的子窗口，裁剪/移动天然跟随，但**尺寸**仍可能停在旧值上：宿主
                // 缩放了包裹窗却没调用 `IPlugView::onSize`。定时器用同一个"变了才摆"
                // 的判据兜底（见 `fit_child_to_parent`）。
                if !anchor.0.is_null() {
                    sync_popup_to_anchor(hwnd, anchor);
                } else {
                    let _ = fit_child_to_parent(hwnd, None);
                }
                // 【为什么先重建而不是直接关窗】页面进程崩溃 / OOM 之后所有投递都会
                // 失败，界面已是一张死图。此前唯一的处置是**关掉窗口** —— 用户看到的
                // 是一块空白，而且必须自己重开插件（这正是用户报的"关掉再开就好"）。
                // 现在先在同一个 HWND 上重建（毫秒级，因为环境是池化的）；只有连续
                // 失败到上限才真的关，并把原因写在界面与日志里。
                if client.borrow().is_dead() {
                    let port = (*pointer).port.clone();
                    if let Err(reason) = rebuild_page(&client, &port) {
                        crate::gui_log_line(&format!("closing the editor view: {reason}"));
                        let _ = DestroyWindow(hwnd);
                    }
                }
                return LRESULT(0);
            } else if message == WM_KILLFOCUS {
                // 【为什么记这里】wParam 是**即将获得焦点**的那个窗口 —— 用户点了宿主
                // 的哪一块，这里就是哪一块。"把键盘交给宿主"要把焦点还回原处，靠的就是
                // 这一笔（见 `EditorClient::previous_focus`）。
                (*pointer).client.borrow_mut().note_previous_focus(wparam.0);
            } else if message == WM_SETFOCUS {
                // 宿主 Tab/onFocus 进入自有 HWND 后，使用标准 WebView2 接口进入 HTML 控件。
                // MoveFocus 可能同步通知宿主；调用前不持任何借用。
                //
                // 【为什么在这里清"宿主自己处理这个键"的记账】用户从别的窗口回到插件的
                // 这一刻，是"我接下来想让插件处理这些键"最明确的一刻。那条记账来自一次
                // 只看 `playing` 翻转的探针，会被工程播完 / 循环回卷污染 —— 不清它，
                // 一次误判就让某个键在整场会话里都送不出去。
                let client = (*pointer).client.clone();
                client.borrow_mut().note_focus_regained();
                let port = (*pointer).port.clone();
                port.focus_page();
                return LRESULT(0);
            } else if message == WM_SIZE {
                let port = (*pointer).port.clone();
                port.sync_bounds(hwnd);
            } else if message == WM_DPICHANGED || message == WM_DPICHANGED_AFTERPARENT {
                // 【为什么 DPI 变化也要重下边界】感知窗口换了显示器 ⇒ 物理内容面与
                // rasterization scale 都要更新；不感知窗口的缩放固定 1.0，但窗口
                // 尺寸本身会被重新协商，内容面同样要跟着走。子窗口收到的是
                // `WM_DPICHANGED_AFTERPARENT`；顶层窗口是 `WM_DPICHANGED`。
                let port = (*pointer).port.clone();
                port.sync_bounds(hwnd);
            } else if message == WM_CLOSE {
                let _ = DestroyWindow(hwnd);
                return LRESULT(0);
            } else if message == WM_NCDESTROY {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                let _ = RemovePropW(hwnd, w!("HiFiShifter.ARA.ViewToken"));
                drop(Box::from_raw(pointer));
                WINDOWS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        DefWindowProcW(hwnd, message, wparam, lparam)
    }))
    .unwrap_or(LRESULT(0))
}

/// 把一条 `import_native_audio_file` 改写成普通的 `import_audio_item`。
///
/// 【为什么必须在平台侧做】真实磁盘路径只在 WebView2 的 `AdditionalObjects` 上，
/// 而那是 COM 参数 —— Policy 只收字符串。改写在**进入** Policy 之前完成，于是
/// "拖进来的文件"和"从菜单选的路径"在 Policy 里是同一条命令。
///
/// 返回 `None` = 这条消息不是拖放导入，原样交给 Policy。
fn native_file_request(
    args: &ICoreWebView2WebMessageReceivedEventArgs,
    view_id: &str,
    text: &str,
) -> Option<Result<String, serde_json::Value>> {
    if !text.contains("import_native_audio_file") {
        return None;
    }
    let mut request = serde_json::from_str::<serde_json::Value>(text).ok()?;
    if request["command"] != "import_native_audio_file" {
        return None;
    }
    let path = (|| -> Result<String, String> {
        if request
            .get("args")
            .is_some_and(|value| !value.is_object() && !value.is_null())
        {
            return Err("native File args must be an object".into());
        }
        let extra: ICoreWebView2WebMessageReceivedEventArgs2 =
            args.cast().map_err(|_| "native File bridge unavailable")?;
        let objects = unsafe { extra.AdditionalObjects() }.map_err(|e| e.to_string())?;
        let mut count = 0;
        unsafe { objects.Count(&mut count) }.map_err(|e| e.to_string())?;
        if count != 1 {
            return Err("exactly one native File is required".into());
        }
        let file: ICoreWebView2File = unsafe { objects.GetValueAtIndex(0) }
            .map_err(|e| e.to_string())?
            .cast()
            .map_err(|_| "additional object is not a disk File")?;
        let mut path = PWSTR::null();
        unsafe { file.Path(&mut path) }.map_err(|e| e.to_string())?;
        let path = CoTaskMemPWSTR::from(path).to_string();
        if !std::path::Path::new(&path).is_absolute() {
            return Err("dragged File has no native disk path; use File menu import".into());
        }
        Ok(path)
    })();
    match path {
        Ok(path) => {
            request["command"] = serde_json::json!("import_audio_item");
            request["args"]["audioPath"] = serde_json::json!(path);
            Some(Ok(request.to_string()))
        }
        Err(error) => {
            // 失败也走同一条回执通道 —— Policy 不知道 Windows 的 COM 参数，
            // 所以这条错误只能在这里成形。
            let id = request["id"].as_u64().unwrap_or_default();
            Some(Err(
                serde_json::json!({"version":1,"viewId":view_id,"id":id,
                "ok":false,"errorKind":"transient","error":error}),
            ))
        }
    }
}

/// 建（或复用）WebView2 环境，然后在本窗口的 HWND 上挂一个控制器。
///
/// 【为什么分成"环境"与"控制器"两段】环境是**进程级**资源（见 [`ENVIRONMENT`]），
/// 控制器是**窗口级**资源。拆开之后，重建一个死掉的页面只需要重做第二段。
fn begin_browser(state: &Rc<RefCell<EditorClient>>, port: &Rc<WindowsPort>) -> Result<(), String> {
    let folder = assets()?;
    pin_callback_code()?;
    let started = std::time::Instant::now();
    // 已经有环境：直接建控制器。这条路是"开第二个窗口"与"重建"的共同路径。
    if let Some(environment) = cached_environment() {
        crate::gui_log_line("webview: environment reused (per-thread pool)");
        return create_controller(&environment, state, port, folder, true);
    }
    crate::gui_log_line("webview: creating the environment");
    let profile = profile_directory()?;
    let weak = Rc::downgrade(state);
    let outer_port = port.clone();
    let completed = CreateCoreWebView2EnvironmentCompletedHandler::create(Box::new(
        move |result, environment| {
            let Some(state) = weak.upgrade() else {
                return Ok(());
            };
            if state.borrow().closed {
                return Ok(());
            }
            let environment = match (result, environment) {
                (Ok(()), Some(environment)) => environment,
                (Err(error), _) => {
                    crate::gui_log_line(&format!(
                        "webview: environment failed after {}ms: {error}",
                        started.elapsed().as_millis()
                    ));
                    outer_port.show_error(&error.to_string());
                    return Ok(());
                }
                _ => {
                    outer_port.show_error("WebView2 returned no environment");
                    return Ok(());
                }
            };
            crate::gui_log_line(&format!(
                "webview: environment ready after {}ms",
                started.elapsed().as_millis()
            ));
            store_environment(&environment);
            if let Err(error) =
                create_controller(&environment, &state, &outer_port, folder.clone(), false)
            {
                outer_port.show_error(&error.to_string());
            }
            Ok(())
        },
    ));
    let options = environment_options();
    unsafe {
        CreateCoreWebView2EnvironmentWithOptions(
            PCWSTR::null(),
            PCWSTR(profile.as_ptr()),
            Some(&options),
            &completed,
        )
    }
    .map_err(|e| e.to_string())
}

/// 在给定环境上为本窗口创建控制器（异步；结果在回调里挂端口并配置浏览器）。
///
/// `from_cache` 记录这次用的是不是池化的环境 —— 若是，而控制器创建失败，那只可能是
/// 那个环境已经随浏览器进程一起死了，必须丢掉它，否则之后每次重建都会撞上同一个死环境。
fn create_controller(
    environment: &ICoreWebView2Environment,
    state: &Rc<RefCell<EditorClient>>,
    port: &Rc<WindowsPort>,
    folder: PathBuf,
    from_cache: bool,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let weak = Rc::downgrade(state);
    let outer_port = port.clone();
    let controller_ready = CreateCoreWebView2ControllerCompletedHandler::create(Box::new(
        move |result, controller| {
            let Some(state) = weak.upgrade() else {
                if let Some(controller) = controller {
                    let _ = unsafe { controller.Close() };
                }
                return Ok(());
            };
            if state.borrow().closed {
                if let Some(controller) = controller {
                    let _ = unsafe { controller.Close() };
                }
                return Ok(());
            }
            if let Err(error) = result {
                // 【为什么这里必须把视图重新判死】池化的环境可能已经随浏览器进程一起
                // 死了；控制器建不出来时页面不会自己好起来，而"没死"就意味着定时器
                // 再也不会来重建。判死 + 丢掉环境，下一次 tick 就会用新环境重试。
                crate::gui_log_line(&format!(
                    "webview: controller failed after {}ms (from_cache={from_cache}): {error}",
                    started.elapsed().as_millis()
                ));
                if from_cache {
                    drop_environment();
                    state.borrow_mut().mark_dead("controller creation failed");
                }
                outer_port.show_error(&error.to_string());
                return Ok(());
            }
            let Some(controller) = controller else {
                outer_port.show_error("WebView2 returned no controller");
                return Ok(());
            };
            crate::gui_log_line(&format!(
                "webview: controller ready after {}ms",
                started.elapsed().as_millis()
            ));
            outer_port.attach(controller.clone());
            // 控制器起来了 = 这一代页面已经接手这块矩形；旧的错误提示必须撤掉，
            // 否则它会一直盖在刚恢复的界面上。
            outer_port.clear_error();
            // 【为什么端口在这里才挂上】页面就绪前的出站消息由 Policy 排队；
            // 没有端口时它连"投不出去"都判断不了。
            let dyn_port: Rc<dyn EditorPort> = outer_port.clone();
            state.borrow_mut().set_port(dyn_port);
            if let Err(error) = configure_browser(&state, &outer_port, &controller, &folder) {
                outer_port.show_error(&error.to_string());
            }
            Ok(())
        },
    ));
    let hwnd = port.hwnd.get();
    unsafe { environment.CreateCoreWebView2Controller(hwnd, &controller_ready) }
        .map_err(|e| e.to_string())
}

/// 页面死了：关掉旧控制器，在**同一个 HWND** 上重新走一次引导。
///
/// 【为什么不重建窗口】窗口是宿主交给我们的、由 `IPlugView` 管尺寸与焦点的那一块矩形。
/// 换一个 HWND 就等于把宿主手里的句柄换掉，而宿主并不知道。控制器是"谁在这个矩形里
/// 画东西"，它可以随便换 —— 换的时候用户只会看到画面闪一下。
///
/// 【为什么不重建环境】见 [`ENVIRONMENT`]：环境是进程级的，重建它反而要等一次初始化。
fn rebuild_page(state: &Rc<RefCell<EditorClient>>, port: &Rc<WindowsPort>) -> Result<(), String> {
    let attempt = PAGE_REBUILDS.with(|count| {
        let next = count.get().saturating_add(1);
        count.set(next);
        next
    });
    if attempt > MAX_PAGE_REBUILDS {
        return Err(format!(
            "the embedded page failed {MAX_PAGE_REBUILDS} times in a row; reopen the editor to try again"
        ));
    }
    crate::gui_log_line(&format!(
        "rebuilding the embedded page (attempt {attempt}/{MAX_PAGE_REBUILDS})"
    ));
    // 【顺序】先让判定层放掉旧端口，再关控制器，最后重置就绪门。反过来的话，关闭
    // 过程中的迟到回调会往一个"已经准备好"的队列里塞消息。
    port.close();
    state.borrow_mut().reset_page();
    begin_browser(state, port)
}

impl WindowsPort {
    fn attach(&self, controller: ICoreWebView2Controller) {
        *self.controller.borrow_mut() = Some(controller);
        // 新控制器还没有收过任何边界：下一次 sync_bounds 必须真的下发一次。
        self.bounds.set(None);
    }

    /// 撤掉错误提示。
    ///
    /// 【为什么必须显式撤】错误提示是一块盖在 WebView 上的 `STATIC` 子窗口。
    /// 页面重建成功之后它仍然留在最上层 —— 于是"已经恢复"的界面看起来还是一块
    /// 错误文字，用户唯一的结论仍然是"得重开插件"。重建成功即撤掉它。
    fn clear_error(&self) {
        let existing = self.error_window.borrow_mut().take();
        if let Some(window) = existing {
            let _ = unsafe { DestroyWindow(window) };
        }
    }

    /// 宿主把焦点交给本窗口后，让 WebView2 把焦点放进 HTML 控件。
    fn focus_page(&self) {
        let controller = self.controller.borrow().clone();
        if let Some(controller) = controller {
            let _ = unsafe { controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT) };
        }
    }

    /// 子窗口尺寸变了：把它的**内容面**交给控制器，并显式固定渲染比例。
    ///
    /// 【为什么不能只看 `GetClientRect` 的数字】同一个客户区在两种感知上下文里
    /// 是两种长度：ACID 的 unaware 包裹层报 1100×720（窗口坐标，DWM 会拉伸到
    /// 1375×900 物理），感知窗口报的就是物理值。WebView 的 raw surface 必须与
    /// **窗口 surface** 同尺，否则 DWM 会把它再拉伸一次 —— 内容放大、右/下被裁。
    /// 判据在 [`content_geometry`] 里只有一条：窗口会不会被 DWM 拉伸。
    ///
    /// 现在两边都显式：BoundsMode=RAW_PIXELS、Bounds=`content_geometry` 给出的
    /// 内容面、RasterizationScale 也由它给出。页面视口于是始终是 `内容面 / 缩放`，
    /// 与窗口的合成方式一致。
    fn sync_bounds(&self, hwnd: HWND) {
        let controller = self.controller.borrow().clone();
        let Some(controller) = controller else {
            return;
        };
        let Some(geometry) = editor_geometry(hwnd) else {
            return;
        };
        let (width, height) = geometry.content;
        let scale = geometry.scale;
        let key = (width, height, scale.to_bits());
        let previous = self.bounds.get();
        if previous == Some(key) {
            return;
        }
        let Ok(controller3) = controller.cast::<ICoreWebView2Controller3>() else {
            // 【旧运行时没有 BoundsMode】交窗口上下文里的客户区原值：unaware 窗口
            // 里它正好是内容面；感知窗口里它是物理值。它不能显式固定缩放，但也
            // 不会把已知可用的宿主弄坏。
            let mut bounds = RECT::default();
            let _scope = HostContextScope::enter(hwnd);
            if unsafe { GetClientRect(hwnd, &mut bounds) }.is_ok() {
                let _ = unsafe { controller.SetBounds(bounds) };
            }
            return;
        };
        // 【为什么关掉自动检测】不感知窗口的 `GetDpiForWindow` 恒为 96，自动检测
        // 会让缩放与窗口坐标脱节；缩放该取多少由 [`content_geometry`] 按 surface
        // 是否被 DWM 拉伸决定，这里显式给。
        let _ = unsafe { controller3.SetShouldDetectMonitorScaleChanges(false) };
        let _ = unsafe { controller3.SetBoundsMode(COREWEBVIEW2_BOUNDS_MODE_USE_RAW_PIXELS) };
        let _ = unsafe { controller3.SetRasterizationScale(scale) };
        let bounds = RECT {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        };
        if unsafe { controller.SetBounds(bounds) }.is_ok() {
            self.bounds.set(Some(key));
            // 尺寸变化在拖动分隔条时会来很多次；只在首次配置 / 缩放变化时留一行，
            // 诊断"子窗口跟随了没有、WebView 拿到了多大的面"已经足够。
            let scale_changed = previous.is_none_or(|previous| previous.2 != key.2);
            if scale_changed {
                let css_width = (f64::from(width) / scale).round() as i32;
                let css_height = (f64::from(height) / scale).round() as i32;
                crate::gui_log_line(&format!(
                    "editor webview bounds: client={}x{} content={width}x{height} \
                     css={css_width}x{css_height} scale={scale:.2} mode=raw-pixels \
                     unaware={}",
                    geometry.client.0, geometry.client.1, geometry.unaware
                ));
            }
        }
    }

    /// 停放/恢复页面渲染。停放期间页面不必画，也避免 Chromium 在隐藏窗口上
    /// 继续合成；恢复时由 `sync_bounds` 按新父窗口尺寸对齐。
    fn set_page_visible(&self, visible: bool) {
        let controller = self.controller.borrow().clone();
        if let Some(controller) = controller {
            let _ = unsafe { controller.SetIsVisible(visible) };
        }
    }
}

/// `COREWEBVIEW2_PROCESS_FAILED_KIND` → 可 grep 的名字。
///
/// 【为什么不用 `{:?}`】这些是 `windows-core` 的 newtype，`Debug` 打出来是
/// `COREWEBVIEW2_PROCESS_FAILED_KIND(5)` —— 一条要靠查表才能读懂的日志。名字是给人
/// 看的，序号随时可能随 SDK 版本变。
fn process_failed_kind(args: &ICoreWebView2ProcessFailedEventArgs) -> &'static str {
    let mut raw = COREWEBVIEW2_PROCESS_FAILED_KIND::default();
    if unsafe { args.ProcessFailedKind(&mut raw) }.is_err() {
        return "unknown";
    }
    let value = raw.0;
    if value == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED.0 {
        "render-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE.0 {
        "render-process-unresponsive"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED.0 {
        "browser-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED.0 {
        "gpu-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED.0 {
        "frame-render-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_UTILITY_PROCESS_EXITED.0 {
        "utility-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_SANDBOX_HELPER_PROCESS_EXITED.0 {
        "sandbox-helper-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_PPAPI_PLUGIN_PROCESS_EXITED.0 {
        "ppapi-plugin-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_PPAPI_BROKER_PROCESS_EXITED.0 {
        "ppapi-broker-process-exited"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_KIND_UNKNOWN_PROCESS_EXITED.0 {
        "unknown-process-exited"
    } else {
        "unknown"
    }
}

/// `COREWEBVIEW2_PROCESS_FAILED_REASON` → 可 grep 的名字。
///
/// 【为什么 OOM 这一条最关键】用户报的就是 "Out of Memory"。把它单独命名之后，
/// "这一次到底是内存耗尽、还是崩溃、还是被外部杀掉"在日志里一眼可辨 —— 而这三者的
/// 处置完全不同（前者要降载荷，后者要查崩溃，第三种可能是宿主主动结束）。
fn process_failed_reason(raw: COREWEBVIEW2_PROCESS_FAILED_REASON) -> &'static str {
    let value = raw.0;
    if value == COREWEBVIEW2_PROCESS_FAILED_REASON_OUT_OF_MEMORY.0 {
        "out-of-memory"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_CRASHED.0 {
        "crashed"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_TERMINATED.0 {
        "terminated"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_UNRESPONSIVE.0 {
        "unresponsive"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_LAUNCH_FAILED.0 {
        "launch-failed"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_PROFILE_DELETED.0 {
        "profile-deleted"
    } else if value == COREWEBVIEW2_PROCESS_FAILED_REASON_UNEXPECTED.0 {
        "unexpected"
    } else {
        "unknown"
    }
}

fn configure_browser(
    state: &Rc<RefCell<EditorClient>>,
    port: &Rc<WindowsPort>,
    controller: &ICoreWebView2Controller,
    folder: &std::path::Path,
) -> windows::core::Result<()> {
    let hwnd = port.hwnd.get();
    {
        // 【深色加载底】控制器刚创建、页面还没画第一帧时 WebView2 用的是默认白底；
        // 把它设成编辑器主题色，等待期不再闪白屏（与窗口类背景同一个颜色）。
        if let Ok(controller2) = controller.cast::<ICoreWebView2Controller2>() {
            let color = COREWEBVIEW2_COLOR {
                A: 255,
                R: EDITOR_BACKGROUND_RGB.0,
                G: EDITOR_BACKGROUND_RGB.1,
                B: EDITOR_BACKGROUND_RGB.2,
            };
            let _ = unsafe { controller2.SetDefaultBackgroundColor(color) };
        }
        // 【初始边界走与 WM_SIZE 完全相同的一条路】raw-pixels + 显式缩放的决定只该
        // 有一份；在这里重写一遍 `GetClientRect/SetBounds` 会让第二次打开（不感知
        // 包裹层）重新落回"逻辑值当物理值"的错误。`sync_bounds` 在控制器刚 attach
        // 时缓存为空，因此这里一定会真的下发一次。
        port.sync_bounds(hwnd);
        unsafe {
            controller.SetIsVisible(true)?;
        }
    }
    // 异步创建完成时仅延续宿主已交给本窗口的焦点，不能抢另一个 FX/应用的焦点。
    if unsafe { GetFocus() } == hwnd {
        unsafe {
            controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT)?;
        }
    }
    let browser = unsafe { controller.CoreWebView2()? };
    // 真实文件拖入由 WebView2 原生 OLE 接收，再通过 AdditionalObjects/File.Path 交给宿主。
    if let Ok(external) = controller.cast::<ICoreWebView2Controller4>() {
        unsafe {
            external.SetAllowExternalDrop(true)?;
        }
    }
    let mapping: ICoreWebView2_3 = browser.cast()?;
    let path = wide(&folder.to_string_lossy());
    unsafe {
        mapping.SetVirtualHostNameToFolderMapping(
            w!("hifishifter.invalid"),
            PCWSTR(path.as_ptr()),
            COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_DENY_CORS,
        )?;
    }
    let settings = unsafe { browser.Settings()? };
    unsafe {
        settings.SetAreHostObjectsAllowed(false)?;
        settings.SetAreDefaultContextMenusEnabled(false)?;
        // 【为什么必须关掉浏览器加速键】默认值下 WebView2 自己消费一批按键
        // （`F5` 重载页面、`F3` / `F7`、`Ctrl+F`、`Alt+←` …），页面根本收不到
        // `keydown`。而 `F1-F24` 明确在可让路键表里 —— 用户按 F5 期望宿主动作，
        // 实际发生的是**插件页面被重载**（顺带清掉前端的"本会话不再尝试"闩锁，
        // 于是同一个 bug 会"偶尔自愈"，更难查）。
        //
        // 【为什么选"关掉"而不是订阅 `AcceleratorKeyPressed`】两者都能到达 DOM，
        // 但订阅那条路要自己维护一份"哪些键允许浏览器处理"的白名单，而那份名单
        // 会与页面里的键表漂移成两份；关掉之后**所有**按键都进页面，让路的判据
        // 只剩一处（前端 `hostKeyRelay`）。方案 §3.5.5 允许二选一。
        // 该成员在 `ICoreWebView2Settings3` 上（WebView2 ≥ 1.0.864），基础接口没有。
        let settings3: webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2Settings3 =
            settings.cast()?;
        settings3.SetAreBrowserAcceleratorKeysEnabled(false)?;
    }
    let mut token = 0;
    unsafe {
        browser.add_NavigationStarting(
            &NavigationStartingEventHandler::create(Box::new(|_, args| {
                if let Some(args) = args {
                    let mut uri = PWSTR::null();
                    args.Uri(&mut uri)?;
                    if !CoTaskMemPWSTR::from(uri).to_string().starts_with(ORIGIN) {
                        args.SetCancel(true)?;
                    }
                }
                Ok(())
            })),
            &mut token,
        )?;
        browser.add_NewWindowRequested(
            &NewWindowRequestedEventHandler::create(Box::new(|_, args| {
                if let Some(args) = args {
                    args.SetHandled(true)?;
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    // 就绪门（见方案 `4.3`）：导航完成之前投递会被丢弃，所以 Policy 先排队。
    //
    // 【为什么用 NavigationCompleted 而不是"脚本注入完成"】后者只证明脚本装好了，
    // 页面本身还在解析；前者才是"监听器已在运行"的可靠信号。
    // 【为什么必须订阅 ProcessFailed】这是 WebView2 唯一一条"渲染进程 / GPU 进程 /
    // 浏览器进程没了"的**显式**信号。没有它，"页面被 OOM 杀掉"在插件侧与"用户一直没
    // 操作"完全同形 —— 全仓此前 0 次出现，所以用户报的 Out of Memory 在日志里连一行
    // 痕迹都没有。它同时也是自动重建的触发器。
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_ProcessFailed(
            &ProcessFailedEventHandler::create(Box::new(move |_, args| {
                let Some(args) = args else {
                    return Ok(());
                };
                let kind = process_failed_kind(&args);
                // 详细原因与退出码在 ICoreWebView2ProcessFailedEventArgs2 上；
                // 老版本运行时没有它，那就只报 kind。
                let (reason, exit) = match args.cast::<ICoreWebView2ProcessFailedEventArgs2>() {
                    Ok(detailed) => {
                        let mut raw = COREWEBVIEW2_PROCESS_FAILED_REASON::default();
                        let reason = if detailed.Reason(&mut raw).is_ok() {
                            process_failed_reason(raw)
                        } else {
                            "unknown"
                        };
                        let mut exit = 0i32;
                        let exit = if detailed.ExitCode(&mut exit).is_ok() {
                            exit
                        } else {
                            0
                        };
                        (reason, exit)
                    }
                    Err(_) => ("unknown", 0),
                };
                if let Some(state) = weak.upgrade() {
                    state.borrow_mut().on_process_failed(kind, reason, exit);
                } else {
                    crate::gui_log_line(&format!(
                        "renderer failed kind={kind} reason={reason} exit=0x{exit:08X}"
                    ));
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_NavigationCompleted(
            &NavigationCompletedEventHandler::create(Box::new(move |_, args| {
                let Some(args) = args else {
                    return Ok(());
                };
                let mut ok = windows::core::BOOL(0);
                let success = args.IsSuccess(&mut ok).is_ok() && ok.as_bool();
                if success {
                    // 页面真的跑起来了：连续重建计数归零（下一次故障重新给三次机会）。
                    PAGE_REBUILDS.with(|count| count.set(0));
                    if let Some(state) = weak.upgrade() {
                        state.borrow_mut().on_page_ready();
                    }
                } else {
                    // 【为什么失败也要落一行】导航失败此前完全静默：界面停在上一帧，
                    // 日志里一行都没有，于是"白屏"只能靠猜。状态码是唯一能区分
                    // "资源缺失"与"渲染进程没了"的读数。
                    let mut status = COREWEBVIEW2_WEB_ERROR_STATUS::default();
                    let status = if args.WebErrorStatus(&mut status).is_ok() {
                        format!("0x{:08X}", status.0)
                    } else {
                        "unknown".to_string()
                    };
                    crate::gui_log_line(&format!("navigation failed status={status}"));
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    let weak = Rc::downgrade(state);
    unsafe {
        browser.add_WebMessageReceived(
            &WebMessageReceivedEventHandler::create(Box::new(move |_, args| {
                let (Some(state), Some(args)) = (weak.upgrade(), args) else {
                    return Ok(());
                };
                let view_id = {
                    let state = state.borrow();
                    if state.closed {
                        return Ok(());
                    }
                    state.view_id.clone()
                };
                // 来源校验：只有本 bundle 的虚拟主机可以发消息。
                let mut source = PWSTR::null();
                args.Source(&mut source)?;
                if !CoTaskMemPWSTR::from(source).to_string().starts_with(ORIGIN) {
                    return Ok(());
                }
                let mut json = PWSTR::null();
                args.WebMessageAsJson(&mut json)?;
                let text = CoTaskMemPWSTR::from(json).to_string();
                // 【Windows 专属的入口改写】拖进来的真实磁盘文件只能从 COM 参数读到。
                match native_file_request(&args, &view_id, &text) {
                    None => client::on_message(&state, &text),
                    Some(Ok(rewritten)) => client::on_message(&state, &rewritten),
                    Some(Err(reply)) => {
                        state.borrow_mut().post(reply);
                    }
                }
                Ok(())
            })),
            &mut token,
        )?;
    }
    let boot = {
        let state = state.borrow();
        client::bootstrap_json(&state.link, &state.view_id)
    };
    let script = wide(&format!("window.__HFS_PLUGIN_BOOTSTRAP__={boot};"));
    let weak = Rc::downgrade(state);
    let navigate = browser.clone();
    let ready_port = port.clone();
    let ready =
        AddScriptToExecuteOnDocumentCreatedCompletedHandler::create(Box::new(move |result, _| {
            let Some(state) = weak.upgrade() else {
                return Ok(());
            };
            if state.borrow().closed {
                return Ok(());
            }
            if let Err(error) = result {
                ready_port.show_error(&error.to_string());
                return Ok(());
            }
            unsafe {
                navigate.Navigate(w!("https://hifishifter.invalid/plugin.html"))?;
            }
            if let Ok(owner) = state.borrow().link.owner() {
                owner.startup_stage(crate::startup::Stage::Navigation);
            }
            crate::gui_log_line("frontend navigation started (commands pending session binding)");
            Ok(())
        }));
    unsafe { browser.AddScriptToExecuteOnDocumentCreated(PCWSTR(script.as_ptr()), &ready) }
}

#[cfg(test)]
mod focus_tests {
    use super::*;
    use std::ffi::c_void;
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetActiveWindow, SetActiveWindow};

    /// 用真实自有HWND验证Tab边界，不凭源码是否包含某个style判断可达性。
    #[test]
    fn native_editor_is_a_tab_stop_in_its_host_parent() {
        // 【为什么在 PMv2 作用域里建夹具父窗】本用例验证的是**感知包裹窗里的 child
        // 焦点语义**（宿主 Tab 链）。默认形态已经是宿主父窗口里的子窗口（popup 只在
        // 显式 `HIFISHIFTER_DPI_POPUP=1` 时启用，见 `dpi_popup_requested`），这里仍用
        // PMv2 父窗，覆盖"感知宿主 + 子窗口"这条最常见路径。
        // SAFETY: 夹具父窗口仅在本线程创建/销毁，不操作任何用户或REAPER窗口。
        let parent = {
            let _dpi_scope = HostContextScope::force_pm_v2();
            unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE::default(),
                    w!("STATIC"),
                    w!("HiFiShifter focus fixture"),
                    WS_OVERLAPPEDWINDOW,
                    0,
                    0,
                    1100,
                    720,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .unwrap()
        };
        let host = WindowsHost::attach(parent.0 as usize, 1100, 720, Arc::new(Default::default()))
            .unwrap();
        let child = HWND(host.key.hwnd as *mut c_void);
        // `SetFocus` 的前置条件是目标窗口所属线程有一个**活动窗口**：刚创建的
        // 夹具窗口尚未激活，此时 `SetFocus` 返回 `ERROR_INVALID_PARAMETER`
        // (0x80070057)。真实宿主里插件窗口始终位于活动窗口内，这里补上同一前置
        // 条件。
        //
        // 但"能否激活"取决于会话：当本进程不是前台进程时（无头/服务/沙箱，或
        // 用户正聚焦别的应用）`SetActiveWindow` 会被系统拒绝。此时**焦点投递**
        // 类断言无从成立，跳过它们；本用例的本职 —— Tab 顺序可达性 —— 不依赖
        // 激活，照常执行。这样 CI（有交互桌面）保持全量覆盖，无前台窗口的会话
        // 也不会假失败。
        // `SetActiveWindow` 返回先前的活动窗口（没有则为 null → Err），仅用于还原。
        let previous_active = unsafe { SetActiveWindow(parent) }.ok();
        let can_take_activation = unsafe { GetActiveWindow() } == parent;
        if !can_take_activation {
            eprintln!(
                "[focus_tests] 本会话不允许本进程激活窗口，跳过焦点投递断言（Tab 顺序断言仍执行）"
            );
        }
        // GetNextDlgTabItem在没有任何Tab stop时会回退到第一个child，单看句柄会假通过。
        let tab_stop = unsafe { GetWindowLongPtrW(child, GWL_STYLE) } as u32 & WS_TABSTOP.0 != 0;
        let key = host.key;
        assert!(
            std::thread::spawn(move || key.focus().is_err())
                .join()
                .unwrap(),
            "不能从非UI线程触碰宿主焦点"
        );
        let focus_result = key.focus();
        let focused = unsafe { GetFocus() };
        // SAFETY: 两个窗口均为本线程且存活；仅查询标准Windows对话框导航结果。
        let target = unsafe { GetNextDlgTabItem(parent, None, false) }.ok();
        drop(host);
        assert!(key.focus().is_err(), "关闭后的窗口租约不能再夺取焦点");
        // 把线程的活动窗口还回去，免得把即将销毁的夹具窗口留给后续用例。
        if let Some(previous) = previous_active {
            // SAFETY: 还原本线程先前的活动窗口。
            let _ = unsafe { SetActiveWindow(previous) };
        }
        // SAFETY: 清理测试创建的唯一父窗口，子窗口已由NativeEditor析构。
        unsafe { DestroyWindow(parent) }.unwrap();
        assert!(
            tab_stop,
            "真实编辑器HWND必须接受宿主Tab停留，不能依赖导航的首child回退"
        );
        if can_take_activation {
            assert!(
                focus_result.is_ok(),
                "首次无旧焦点时仍应成功: {focus_result:?}"
            );
            assert_eq!(focused, child, "宿主焦点请求必须进入自有窗口");
        }
        assert_eq!(target, Some(child), "宿主Tab顺序必须能到达真实编辑器子窗口");
    }
}
