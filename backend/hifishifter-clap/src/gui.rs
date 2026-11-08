//! CLAP `clap.gui` 扩展：把同一份前端产物嵌进宿主给的父窗口。
//!
//! 【为什么不复用 VST3 的原语】VST3 的 `editor/gui` 与 ARA 会话、219 条命令、
//! REAPER 原生扩展绑在一起；CLAP 外壳复用共享编辑会话，不另起一套后端。这里走
//! `editor/windows.rs` 的精简 WebView2 宿主 + `editor/protocol.rs` 的结构化回执：
//! 页面是真的；宿主把实例绑到 ARA 文档后命令由共享会话服务，未绑定时
//! `hostCapabilities` 如实表达能力边界。
//!
//! 【非 Windows】当前只有 Windows 提供内嵌 WebView2；其它平台 `is_api_supported`
//! 如实返回 false，绝不交一个空白视图让宿主以为成功。

use crate::abi::{clap_plugin_t, CLAP_EXT_GUI};
use std::ffi::{c_char, c_void, CStr};

/// CLAP 的父窗口句柄（clap.window）。
#[repr(C)]
pub struct clap_window_t {
    pub api: *const c_char,
    pub handle: *mut c_void,
}
/// `clap_plugin_gui_t` 的最小可用实现。
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

/// 宿主 GUI 回调；只用 request_resize。
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

unsafe extern "C" fn is_api_supported(
    _p: *const clap_plugin_t,
    api: *const c_char,
    floating: bool,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if api.is_null() || !crate::editor::is_supported() {
            return false;
        }
        let Ok(api) = unsafe { CStr::from_ptr(api) }.to_str() else {
            return false;
        };
        #[cfg(windows)]
        {
            crate::editor::windows::is_api_supported(api, floating)
        }
        #[cfg(not(windows))]
        {
            let _ = (api, floating);
            false
        }
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
        #[cfg(windows)]
        {
            let Some((name, is_floating)) = crate::editor::windows::preferred_api() else {
                return false;
            };
            unsafe {
                *api = match name {
                    "win32" => c"win32".as_ptr(),
                    "cocoa" => c"cocoa".as_ptr(),
                    "x11" => c"x11".as_ptr(),
                    _ => c"wayland".as_ptr(),
                };
                *floating = is_floating;
            }
            true
        }
        #[cfg(not(windows))]
        {
            false
        }
    })
}
unsafe extern "C" fn create(p: *const clap_plugin_t, _api: *const c_char, floating: bool) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        #[cfg(windows)]
        {
            // SAFETY: create 是 clap.gui 的回调，p 是本实例的活记录。
            match unsafe { crate::editor::windows::create(p, floating) } {
                Ok(()) => {
                    log::info!("[clap][gui] embedded view created");
                    true
                }
                Err(error) => {
                    log::error!("[clap][gui] create failed: {error}");
                    false
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (p, floating);
            log::warn!("[clap][gui] {}", crate::editor::unavailable_reason());
            false
        }
    })
}
unsafe extern "C" fn destroy(p: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || {
        #[cfg(windows)]
        // SAFETY: `p` 是本模块创建、尚未销毁的插件实例（CLAP 保证 gui.destroy 早于
        // plugin.destroy）。
        unsafe {
            crate::editor::windows::destroy(p)
        };
        #[cfg(not(windows))]
        let _ = p;
    })
}
unsafe extern "C" fn set_scale(_p: *const clap_plugin_t, _scale: f64) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn get_size(p: *const clap_plugin_t, width: *mut u32, height: *mut u32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if width.is_null() || height.is_null() {
            return false;
        }
        #[cfg(windows)]
        if let Some((w, h)) = crate::editor::windows::get_size(p) {
            unsafe {
                *width = w;
                *height = h;
            }
            return true;
        }
        #[cfg(not(windows))]
        let _ = p;
        false
    })
}
unsafe extern "C" fn can_resize(_p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || true)
}
/// 编辑器允许的最小 / 最大像素尺寸。
///
/// 【为什么最小不是 0】参数编辑器与时间轴在这个宽度以下会互相压扁（工具栏换行、
/// 轨道头吃掉全部横向空间）。给宿主一个下限，比让它把窗口拖成一条缝再画不出来好。
pub const MIN_EDITOR_SIZE: (u32, u32) = (720, 480);
/// 上限只是"别让一次误拖把窗口设成 8K"的理智线。
pub const MAX_EDITOR_SIZE: (u32, u32) = (4096, 2400);

/// 把宿主请求的尺寸夹到合法区间。
pub fn clamp_editor_size(width: u32, height: u32) -> (u32, u32) {
    (
        width.clamp(MIN_EDITOR_SIZE.0, MAX_EDITOR_SIZE.0),
        height.clamp(MIN_EDITOR_SIZE.1, MAX_EDITOR_SIZE.1),
    )
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
    hifishifter_plugin_core::guard_ffi(false, || {
        let (width, height) = clamp_editor_size(w, h);
        #[cfg(windows)]
        {
            crate::editor::windows::set_size(p, width, height)
        }
        #[cfg(not(windows))]
        {
            let _ = (p, width, height);
            false
        }
    })
}
unsafe extern "C" fn set_parent(p: *const clap_plugin_t, window: *const clap_window_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        #[cfg(windows)]
        {
            unsafe { crate::editor::windows::set_parent(p, window) }
        }
        #[cfg(not(windows))]
        {
            let _ = (p, window);
            false
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
        #[cfg(windows)]
        {
            crate::editor::windows::show(p)
        }
        #[cfg(not(windows))]
        {
            let _ = p;
            false
        }
    })
}
unsafe extern "C" fn hide(p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        #[cfg(windows)]
        {
            crate::editor::windows::hide(p)
        }
        #[cfg(not(windows))]
        {
            let _ = p;
            false
        }
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

    /// 扩展表的槽位必须齐全：宿主靠它们建窗口、挂父窗口、取尺寸与销毁。
    ///
    /// 【为什么只测这一层】真正的创建路径在 `editor/windows.rs`（子 HWND + WebView2），
    /// 真机验收由 `scripts/clap/probe.c` 的黑盒 GUI 流程负责。这里钉的是"表是完整的、
    /// 并且只提供 win32 嵌入而不是浮窗"这两条契约。
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
        assert_eq!(clamp_editor_size(9000, 9000), MAX_EDITOR_SIZE);
        let (mut w, mut h) = (1280_u32, 720_u32);
        assert!(unsafe { adjust_size(std::ptr::null(), &mut w, &mut h) });
        assert_eq!((w, h), (1280, 720), "the requested size must pass through");
    }

    #[test]
    fn the_gui_extension_is_complete_and_offers_only_embedded_win32() {
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
}
