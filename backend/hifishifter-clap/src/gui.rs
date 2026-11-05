//! CLAP GUI 扩展的诚实声明。
//!
//! 【为什么阶段 1 不嵌 WebView】core 的 WebView 原语依赖 VST3 的 `IRunLoop` /
//! `IPlugFrame` 连接点；CLAP 用 `clap_host_gui` + `timer-support`，是另一套协议。
//! 在共享 GUI 抽象落地前，明确报告「本版本只提供浮窗 API」比返回一个空白嵌入视图诚实。

use crate::abi::{clap_plugin_t, CLAP_EXT_GUI};
use hifishifter_plugin_core::format::GuiAttach;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};

static FLOATING_WINDOW_OPEN: AtomicBool = AtomicBool::new(false);

/// GUI 是否已创建。用于 `on_main_thread` 心跳与状态诊断。
pub fn is_created() -> bool {
    FLOATING_WINDOW_OPEN.load(Ordering::Acquire)
}

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

/// 平台父窗口类型：core 不硬编码 "HWND"。
pub struct ClapGuiAttach {
    pub api: &'static str,
    pub parent: *mut c_void,
    pub host_gui: *const clap_host_gui_t,
}
unsafe impl Send for ClapGuiAttach {}
impl GuiAttach for ClapGuiAttach {
    fn parent(&self) -> *mut c_void {
        self.parent
    }
    fn platform_kind(&self) -> &str {
        self.api
    }
    fn embedded(&self) -> bool {
        false
    }
    fn frame(&self) -> usize {
        0
    }
    fn request_resize(&self, w: i32, h: i32) -> bool {
        let Some(host) = (unsafe { self.host_gui.as_ref() }) else {
            return false;
        };
        let Some(request) = host.request_resize else {
            return false;
        };
        if w <= 0 || h <= 0 {
            return false;
        }
        unsafe { request(self.host_gui.cast(), w as u32, h as u32) }
    }
    fn on_host_tick(&self) {}
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
        floating
            && !api.is_null()
            && unsafe { CStr::from_ptr(api) }.to_bytes() == b"clap:unavailable"
    })
}
unsafe extern "C" fn get_preferred_api(
    _p: *const clap_plugin_t,
    _api: *mut *const c_char,
    _floating: *mut bool,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn create(_p: *const clap_plugin_t, _api: *const c_char, floating: bool) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        // 只承认浮窗；没有嵌入父窗口协议时不谎报成功。
        if !floating {
            return false;
        }
        FLOATING_WINDOW_OPEN.store(true, Ordering::Release);
        true
    })
}
unsafe extern "C" fn destroy(_p: *const clap_plugin_t) {
    hifishifter_plugin_core::guard_ffi((), || FLOATING_WINDOW_OPEN.store(false, Ordering::Release))
}
unsafe extern "C" fn set_scale(_p: *const clap_plugin_t, _scale: f64) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn get_size(_p: *const clap_plugin_t, width: *mut u32, height: *mut u32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if width.is_null() || height.is_null() {
            return false;
        }
        unsafe {
            *width = 960;
            *height = 640
        };
        true
    })
}
unsafe extern "C" fn can_resize(_p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn adjust_size(_p: *const clap_plugin_t, w: *mut u32, h: *mut u32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if w.is_null() || h.is_null() {
            false
        } else {
            unsafe {
                *w = 960;
                *h = 640
            };
            true
        }
    })
}
unsafe extern "C" fn set_size(_p: *const clap_plugin_t, _w: u32, _h: u32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn set_parent(_p: *const clap_plugin_t, _w: *const clap_window_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn set_transient(_p: *const clap_plugin_t, _w: *const clap_window_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || true)
}
unsafe extern "C" fn suggest_title(_p: *const clap_plugin_t, _title: *const c_char) {
    hifishifter_plugin_core::guard_ffi((), || ())
}
unsafe extern "C" fn show(_p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, is_created)
}
unsafe extern "C" fn hide(_p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || true)
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
    #[test]
    fn gui_platform_kind_is_supplied_by_the_format_layer() {
        let attach = ClapGuiAttach {
            api: "clap:win32",
            parent: std::ptr::null_mut(),
            host_gui: std::ptr::null(),
        };
        assert_eq!(attach.platform_kind(), "clap:win32");
        assert!(!attach.embedded());
        assert_eq!(attach.frame(), 0);
    }
}
