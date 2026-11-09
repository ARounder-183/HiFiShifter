//! CLAP 的 `clap.gui` 扩展：把宿主给的父窗口交给**共享编辑器栈**。
//!
//! 【为什么这里只有 ABI 翻译】内嵌编辑器（WebView2 / WKWebView / WebKitGTK）的全部判定都在
//! `hifishifter_plugin::editor::gui`：页面协议、命令分发、原生文件对话框、
//! 渲染进程崩溃后的页面重建、20 ms 泵、模块钉住、Tab 焦点。VST3 的 `IPlugView` 与
//! 本扩展建的是**同一个** `EditorView`（同一份 Policy、同一份平台原语），
//! 因此"同一个命令在两种格式下有两种行为"这一类分叉从结构上不存在。
//!
//! 【为什么视图推迟到 set_parent 才建】CLAP 的 `create` 只给 API 名，父窗口在
//! `set_parent` 才到；而三个平台的视图都必须建在一个活的父句柄里
//! （WebView2 的 `CreateCoreWebView2Controller`、GTK 的 `GtkPlug`）。
//! 把创建推迟一步，就不必为"先建后挂"再写一套父窗口迁移逻辑。
//!
//! 【非 Windows/macOS/Linux】`is_api_supported` 如实返回 false，绝不交一个空白视图。

use crate::abi::{clap_plugin_t, CLAP_EXT_GUI};
use hifishifter_plugin::editor::{EditorLink, EditorView};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};

/// CLAP 的父窗口句柄（clap.window）。
#[repr(C)]
pub struct clap_window_t {
    pub api: *const c_char,
    pub handle: *mut c_void,
}
/// `clap_plugin_gui_t`。
#[repr(C)]
pub struct clap_plugin_gui_t {
    pub is_api_supported:
        Option<unsafe extern "C" fn(*const clap_plugin_t, *const c_char, bool) -> bool>,
    pub get_preferred_api:
        Option<unsafe extern "C" fn(*const clap_plugin_t, *mut *const c_char, *mut bool) -> bool>,
    pub create: Option<unsafe extern "C" fn(*const clap_plugin_t, *const c_char, bool) -> bool>,
    pub destroy: Option<unsafe extern "C" fn(*const clap_plugin_t)>,
    pub set_scale: Option<unsafe extern "C" fn(*const clap_plugin_t, f64) -> bool>,
    pub get_size: Option<unsafe extern "C" fn(*const clap_plugin_t, *mut u32, *mut u32) -> bool>,
    pub can_resize: Option<unsafe extern "C" fn(*const clap_plugin_t) -> bool>,
    pub get_resize_hints: Option<unsafe extern "C" fn(*const clap_plugin_t, *mut c_void) -> bool>,
    pub adjust_size: Option<unsafe extern "C" fn(*const clap_plugin_t, *mut u32, *mut u32) -> bool>,
    pub set_size: Option<unsafe extern "C" fn(*const clap_plugin_t, u32, u32) -> bool>,
    pub set_parent:
        Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_window_t) -> bool>,
    pub set_transient:
        Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_window_t) -> bool>,
    pub suggest_title: Option<unsafe extern "C" fn(*const clap_plugin_t, *const c_char)>,
    pub show: Option<unsafe extern "C" fn(*const clap_plugin_t) -> bool>,
    pub hide: Option<unsafe extern "C" fn(*const clap_plugin_t) -> bool>,
}

/// 宿主 GUI 回调；本插件只用到 `closed` 之外的 `request_resize` 形状。
#[repr(C)]
pub struct clap_host_gui_t {
    pub resize_hints_changed: Option<unsafe extern "C" fn(*const c_void)>,
    pub request_resize: Option<unsafe extern "C" fn(*const c_void, u32, u32) -> bool>,
    pub request_show: Option<unsafe extern "C" fn(*const c_void) -> bool>,
    pub request_hide: Option<unsafe extern "C" fn(*const c_void) -> bool>,
    pub closed: Option<unsafe extern "C" fn(*const c_void, bool)>,
}

pub static GUI_EXT: clap_plugin_gui_t = clap_plugin_gui_t {
    is_api_supported: Some(is_api_supported),
    get_preferred_api: Some(get_preferred_api),
    create: Some(create),
    destroy: Some(destroy),
    set_scale: Some(set_scale),
    get_size: Some(get_size),
    can_resize: Some(can_resize),
    get_resize_hints: None,
    adjust_size: Some(adjust_size),
    set_size: Some(set_size),
    set_parent: Some(set_parent),
    set_transient: Some(set_transient),
    suggest_title: Some(suggest_title),
    show: Some(show),
    hide: Some(hide),
};

/// 编辑器尺寸策略（**与 VST3 共用同一份**：`hifishifter_plugin::editor::gui`）。
///
/// 【为什么不在外壳里再定义一遍】尺寸是产品决策，不是 ABI 知识。此前 VST3 的
/// `IPlugView::constrain` 用 640..=16384、这里用 720..=4096 × 480..=2400 —— 同一个
/// 用户拖同一个窗口，在两种外壳里得到不同的边界，而两处注释都写着"这是产品的下限"。
pub use hifishifter_plugin::editor::gui::{
    clamp_editor_size, EDITOR_DEFAULT_SIZE as DEFAULT_EDITOR_SIZE,
    EDITOR_MAX_SIZE as MAX_EDITOR_SIZE, EDITOR_MIN_SIZE as MIN_EDITOR_SIZE,
};
/// 本平台接受的父窗口 API 名。
fn platform_api() -> Option<&'static str> {
    #[cfg(windows)]
    {
        Some("win32")
    }
    #[cfg(target_os = "macos")]
    {
        Some("cocoa")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Some("x11")
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

/// 一个实例的视图槽位。
///
/// 【为什么按插件指针索引】CLAP 的 gui 回调只带 `clap_plugin_t*`；视图本身是
/// `Rc<dyn EditorHost>`（!Send，只属于宿主 UI 线程）。thread_local 表既满足
/// 线程亲和，也保证"宿主忘了 destroy 时随 plugin.destroy 一起收掉"。
struct Slot {
    size: (u32, u32),
    /// 本视图的实例链路。生命周期与槽位一致（视图重建时复用同一条链路）。
    link: std::sync::Arc<EditorLink>,
    /// 【为什么是 `Rc`】尺寸/显示回调需要先把句柄取出来、**释放 `VIEWS` 借用**，
    /// 再调用平台原语（`resize` 会同步派发 `WM_SIZE`，那条路径可能再进本模块）。
    view: Option<std::rc::Rc<EditorView>>,
}

thread_local! {
    static VIEWS: RefCell<BTreeMap<usize, Slot>> = const { RefCell::new(BTreeMap::new()) };
}

fn key(plugin: *const clap_plugin_t) -> usize {
    plugin as usize
}

fn with_slot<R>(plugin: *const clap_plugin_t, f: impl FnOnce(&mut Slot) -> R) -> Option<R> {
    VIEWS.with(|views| views.borrow_mut().get_mut(&key(plugin)).map(f))
}

unsafe extern "C" fn is_api_supported(
    _p: *const clap_plugin_t,
    api: *const c_char,
    floating: bool,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if api.is_null() || floating || !hifishifter_plugin::editor::gui::available() {
            return false;
        }
        let Ok(api) = unsafe { CStr::from_ptr(api) }.to_str() else {
            return false;
        };
        platform_api() == Some(api)
    })
}

unsafe extern "C" fn get_preferred_api(
    _p: *const clap_plugin_t,
    api: *mut *const c_char,
    floating: *mut bool,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if api.is_null() || floating.is_null() {
            return false;
        }
        let Some(name) = platform_api() else {
            return false;
        };
        let name = match name {
            "win32" => c"win32",
            "cocoa" => c"cocoa",
            _ => c"x11",
        };
        unsafe {
            *api = name.as_ptr();
            *floating = false;
        }
        true
    })
}

unsafe extern "C" fn create(p: *const clap_plugin_t, api: *const c_char, floating: bool) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if !unsafe { is_api_supported(p, api, floating) } {
            return false;
        }
        // 【临时探针】与 VST3 的 `createView` 同名同义：HIFISHIFTER_NO_VIEW=1 时
        // 完全不产生视图，用来判定"宿主冻结"是否出在我们这套视图实现上。
        // 两个格式必须能被同一把探针分开，否则"CLAP 上不冻结、VST3 上冻结"这种
        // 结论无法排除"根本没建视图"。
        if std::env::var("HIFISHIFTER_NO_VIEW").is_ok_and(|value| value.trim() == "1") {
            log::warn!("[clap][gui] create suppressed by HIFISHIFTER_NO_VIEW");
            return false;
        }
        let Some(instance) = (unsafe { crate::instance(p) }) else {
            return false;
        };
        // 【为什么在这里建链路而不是建窗口】父窗口还没到。链路与实例同生共死，
        // 与 VST3 的 `IPlugView` 在 `attached` 之前就已持有 link 完全同构。
        let link = match EditorLink::for_owner(&instance.owner) {
            Ok(link) => link,
            Err(error) => {
                log::error!("[clap][gui] editor route rejected: {error}");
                return false;
            }
        };
        // 重复 create（宿主没先 destroy）时先收掉旧视图，避免两个窗口叠在同一个父窗口里。
        // 【为什么把旧槽位拿到借用之外再 drop】drop 会 DestroyWindow，窗口过程可能同步回调
        // 本模块；在 RefCell 借用里做这件事等于把重入风险留给下一个人。
        let previous = VIEWS.with(|views| {
            views.borrow_mut().insert(
                key(p),
                Slot {
                    size: DEFAULT_EDITOR_SIZE,
                    link,
                    view: None,
                },
            )
        });
        drop(previous);
        log::warn!(
            "[clap][gui] editor created ({}x{})",
            DEFAULT_EDITOR_SIZE.0,
            DEFAULT_EDITOR_SIZE.1
        );
        true
    })
}

unsafe extern "C" fn destroy(p: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        let slot = VIEWS.with(|views| views.borrow_mut().remove(&key(p)));
        if let Some(slot) = slot {
            if let Some(view) = slot.view {
                // Drop 会关窗；显式调一次让"关窗"这条语义在日志里可见。
                view.close();
            }
            log::info!("[clap][gui] editor destroyed");
        }
    })
}

/// 插件销毁时的兜底清理：宿主没按契约先调 `gui.destroy` 时，窗口与链路
/// 必须随实例一起消失（否则窗口类留在已卸载的模块里，下一条消息就是野跳转）。
/// # Safety
/// `plugin` 必须是本模块创建、尚未 destroy 的插件记录（或空指针）。
pub unsafe fn forget(plugin: *const clap_plugin_t) {
    unsafe { destroy(plugin) }
}

unsafe extern "C" fn set_scale(_p: *const clap_plugin_t, _scale: f64) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}

unsafe extern "C" fn get_size(p: *const clap_plugin_t, width: *mut u32, height: *mut u32) -> bool {
    // 【宿主线程泵】宿主只在它自己的 UI 线程上调用视图回调 —— 这是"不依赖宿主支持
    // 任何额外能力"的那条泵路径（VST3 的 `IPlugView::getSize` 调的是同一个函数）。
    hifishifter_plugin::editor::pump_host_thread();
    hifishifter_plugin_core::guard_ffi(false, || {
        if width.is_null() || height.is_null() {
            return false;
        }
        let Some(size) = with_slot(p, |slot| slot.size) else {
            return false;
        };
        unsafe {
            *width = size.0;
            *height = size.1;
        }
        true
    })
}

unsafe extern "C" fn can_resize(_p: *const clap_plugin_t) -> bool {
    // 同 `get_size`：宿主尺寸协商期间也会走这里。
    hifishifter_plugin::editor::pump_host_thread();
    hifishifter_plugin_core::guard_ffi(false, || {
        // 【临时探针】与 VST3 的 `IPlugView::canResize` 同名同义：声明"不可缩放"后
        // 宿主完全不进尺寸协商回路 —— 用来判定"打开 FX 窗口就冻结"是否来自这条回路。
        if std::env::var("HIFISHIFTER_FIXED_SIZE").is_ok_and(|value| value.trim() == "1") {
            return false;
        }
        true
    })
}

unsafe extern "C" fn adjust_size(_p: *const clap_plugin_t, w: *mut u32, h: *mut u32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if w.is_null() || h.is_null() {
            return false;
        }
        // 【为什么这里绝不能返回固定尺寸】宿主在用户拖拽边框时调用本函数，把请求的
        // 宽高交给插件"调整后"使用。此前这里无条件写回 960×640 —— 于是窗口**永远
        // 变不大也变不小**，WebView 也就永远收不到尺寸变化（用户报的"调整窗口大小
        // 前端不更新"）。这里只做夹取，原样放行用户要的尺寸。
        let (width, height) = clamp_editor_size(unsafe { *w }, unsafe { *h });
        unsafe {
            *w = width;
            *h = height;
        }
        true
    })
}

unsafe extern "C" fn set_size(p: *const clap_plugin_t, w: u32, h: u32) -> bool {
    // 同 `get_size`：尺寸协商是宿主 UI 线程上最频繁的进入点。
    hifishifter_plugin::editor::pump_host_thread();
    hifishifter_plugin_core::guard_ffi(false, || {
        let (width, height) = clamp_editor_size(w, h);
        let view = with_slot(p, |slot| {
            slot.size = (width, height);
            slot.view.clone()
        })
        .flatten();
        match view {
            // 已经有视图：把尺寸交给共享原语（它会 SetBounds 并让 WebView 重排）。
            Some(view) => view.resize(width as i32, height as i32).is_ok(),
            // 还没有视图：尺寸先记账，set_parent 时按它建。
            None => true,
        }
    })
}

unsafe extern "C" fn set_parent(p: *const clap_plugin_t, window: *const clap_window_t) -> bool {
    // 同 `get_size`：宿主把父窗口交给我们时，它就在自己的 UI 线程上。
    hifishifter_plugin::editor::pump_host_thread();
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(window) = (unsafe { window.as_ref() }) else {
            return false;
        };
        if window.handle.is_null() {
            return false;
        }
        let Ok(api) = unsafe { CStr::from_ptr(window.api) }.to_str() else {
            return false;
        };
        if platform_api() != Some(api) {
            log::error!("[clap][gui] set_parent with an unexpected window api {api}");
            return false;
        }
        // 宿主可能重新挂父窗口：先关旧视图，再用**同一条链路**在新父窗口里重建。
        // 两步都在 `VIEWS` 借用之外 —— `close` 会 DestroyWindow，窗口过程可能同步回调。
        let previous = with_slot(p, |slot| slot.view.take()).flatten();
        if let Some(view) = previous {
            view.close();
        }
        let Some((size, link)) = with_slot(p, |slot| (slot.size, std::sync::Arc::clone(&slot.link)))
        else {
            return false;
        };
        match hifishifter_plugin::editor::create_editor_view(
            window.handle as usize,
            size.0 as i32,
            size.1 as i32,
            link,
        ) {
            Ok(view) => {
                log::warn!(
                    "[clap][gui] embedded view attached ({}x{}, fidelity={})",
                    size.0,
                    size.1,
                    view.fidelity()
                );
                let view = std::rc::Rc::new(view);
                with_slot(p, |slot| slot.view = Some(view));
                true
            }
            Err(error) => {
                log::error!("[clap][gui] embedded view failed: {error}");
                false
            }
        }
    })
}

unsafe extern "C" fn set_transient(_p: *const clap_plugin_t, _w: *const clap_window_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}

unsafe extern "C" fn suggest_title(_p: *const clap_plugin_t, _title: *const c_char) {
    hifishifter_plugin_core::guard_ffi((), || ())
}

unsafe extern "C" fn show(p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        // 子窗口以 WS_VISIBLE 建在宿主的父窗口里；宿主隐藏父窗口时它随之不可见。
        with_slot(p, |slot| slot.view.is_some()).unwrap_or(false)
    })
}

unsafe extern "C" fn hide(p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        with_slot(p, |slot| slot.view.is_some()).unwrap_or(false)
    })
}

/// `CLAP_EXT_GUI` 的字节串（给 `get_extension` 比较用）。
pub const GUI_ID: &[u8] = CLAP_EXT_GUI;

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for clap_plugin_gui_t {}
unsafe impl Send for clap_plugin_gui_t {}
// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for clap_host_gui_t {}
unsafe impl Send for clap_host_gui_t {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 宿主拖拽边框时**必须原样放行**用户要的尺寸（只夹到合法区间）。
    ///
    /// 【为什么这条是回归测试】此前 `adjust_size` 无条件写回 960×640 —— 窗口拖不动，
    /// WebView 也永远收不到尺寸变化（用户报的"调整窗口大小前端不更新"）。
    #[test]
    fn resize_requests_survive_adjustment() {
        assert_eq!(clamp_editor_size(1200, 800), (1200, 800));
        assert_eq!(
            clamp_editor_size(10, 10),
            MIN_EDITOR_SIZE,
            "below the floor the window must not collapse"
        );
        assert_eq!(
            clamp_editor_size(9000, 9000),
            (9000, 9000),
            "共享上限（16384）之内的尺寸必须原样放行 —— 那是 VST3 的同一条判据"
        );
        assert_eq!(clamp_editor_size(20_000, 20_000), MAX_EDITOR_SIZE);
        let (mut w, mut h) = (1280_u32, 720_u32);
        assert!(unsafe { adjust_size(std::ptr::null(), &mut w, &mut h) });
        assert_eq!((w, h), (1280, 720), "the requested size must pass through");
    }

    #[test]
    fn the_gui_extension_is_complete_and_offers_only_embedded_views() {
        assert!(GUI_EXT.is_api_supported.is_some());
        assert!(GUI_EXT.get_preferred_api.is_some());
        assert!(GUI_EXT.create.is_some());
        assert!(GUI_EXT.destroy.is_some());
        assert!(GUI_EXT.get_size.is_some());
        assert!(GUI_EXT.set_size.is_some());
        assert!(GUI_EXT.set_parent.is_some());
        assert!(GUI_EXT.show.is_some());
        assert!(GUI_EXT.hide.is_some());
        assert!(
            !unsafe { is_api_supported(std::ptr::null(), c"win32".as_ptr(), true) },
            "floating windows are not offered"
        );
    }

    /// 未知 API 与浮窗一律拒绝：宿主据此回退到"这个插件没有界面"。
    #[test]
    fn only_the_embedded_native_api_is_offered() {
        assert!(!unsafe { create(std::ptr::null(), c"unknown".as_ptr(), false) });
        assert!(!unsafe { create(std::ptr::null(), c"win32".as_ptr(), true) });
        assert!(!unsafe { set_parent(std::ptr::null(), std::ptr::null()) });
    }

    /// 没有 `create` 的实例不能凭空报告尺寸或"已显示"。
    #[test]
    fn an_instance_without_a_view_reports_nothing() {
        let (mut w, mut h) = (0_u32, 0_u32);
        assert!(!unsafe { get_size(std::ptr::null(), &mut w, &mut h) });
        assert!(!unsafe { show(std::ptr::null()) });
    }
}
