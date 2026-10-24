//! 按锁定iplugview.h实现的原生VST3视图ABI；只操作自有子窗口。
use crate::vst3::{
    iid_matches, TResult, K_INVALID_ARGUMENT, K_NO_INTERFACE, K_RESULT_FALSE, K_RESULT_OK,
};
use std::ffi::{c_char, c_void, CStr};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

const IID_VIEW: [u32; 4] = [0x5BC32507, 0xD06049EA, 0xA6151B52, 0x2B755B29];
const IID_UNKNOWN: [u32; 4] = [0, 0, 0xC0000000, 0x46];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ViewRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}
impl Default for ViewRect {
    fn default() -> Self {
        Self {
            left: 0,
            top: 0,
            right: 1100,
            bottom: 720,
        }
    }
}
impl ViewRect {
    /// 防止减法溢出及极端宿主尺寸；物理像素不修改宿主DPI策略。
    fn dimensions(&self) -> Option<(i32, i32)> {
        let width = self.right.checked_sub(self.left)?;
        let height = self.bottom.checked_sub(self.top)?;
        ((640..=16384).contains(&width) && (400..=16384).contains(&height))
            .then_some((width, height))
    }
}

#[repr(C)]
struct ViewVtbl {
    query: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    platform: unsafe extern "system" fn(*mut c_void, *const c_char) -> TResult,
    attached: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_char) -> TResult,
    removed: unsafe extern "system" fn(*mut c_void) -> TResult,
    wheel: unsafe extern "system" fn(*mut c_void, f32) -> TResult,
    key_down: unsafe extern "system" fn(*mut c_void, u16, i16, i16) -> TResult,
    key_up: unsafe extern "system" fn(*mut c_void, u16, i16, i16) -> TResult,
    get_size: unsafe extern "system" fn(*mut c_void, *mut ViewRect) -> TResult,
    on_size: unsafe extern "system" fn(*mut c_void, *mut ViewRect) -> TResult,
    focus: unsafe extern "system" fn(*mut c_void, u8) -> TResult,
    set_frame: unsafe extern "system" fn(*mut c_void, *mut c_void) -> TResult,
    can_resize: unsafe extern "system" fn(*mut c_void) -> TResult,
    constrain: unsafe extern "system" fn(*mut c_void, *mut ViewRect) -> TResult,
}

#[derive(Default)]
struct ViewState {
    rect: ViewRect,
    frame: usize,
    attaching: bool,
    generation: u64,
    /// 平台视图句柄。§None§ = 尚未附着或已移除。
    ///
    /// 【为什么是 §Rc§ 而不是 §Box§】几何/焦点/按键三处都需要**先取出来、再释放 view 锁**
    /// 才能调用它（宿主回调会同步重入）。§Rc§ 让"取出来"是一次克隆。
    native: Option<Rc<dyn super::gui::EditorHost>>,
}
#[repr(C)]
struct View {
    vtbl: *const ViewVtbl,
    refs: AtomicU32,
    state: Mutex<ViewState>,
    link: std::sync::Arc<super::routing::EditorLink>,
}

/// 返回一个具有宿主owning引用的IPlugView；实例路由由后续connection接线提供。
#[cfg(test)]
fn create_view() -> *mut c_void {
    create_view_with_link(std::sync::Arc::new(Default::default()))
}
/// 每个FX视图继承其真实controller connection，不做任何全局实例搜索。
pub(crate) fn create_view_with_link(
    link: std::sync::Arc<super::routing::EditorLink>,
) -> *mut c_void {
    Box::into_raw(Box::new(View {
        vtbl: &VTBL,
        refs: AtomicU32::new(1),
        state: Mutex::new(ViewState::default()),
        link,
    }))
    .cast()
}

/// UI ABI中的panic不能越过FFI杀死宿主；错误统一转为明确失败。
fn boundary(f: impl FnOnce() -> TResult) -> TResult {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(K_RESULT_FALSE)
}
unsafe fn view<'a>(this: *mut c_void) -> &'a View {
    unsafe { &*this.cast::<View>() }
}
unsafe extern "system" fn query(
    this: *mut c_void,
    iid: *const u8,
    out: *mut *mut c_void,
) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if out.is_null() || this.is_null() {
                return K_INVALID_ARGUMENT;
            }
            unsafe {
                *out = std::ptr::null_mut();
            }
            if unsafe { iid_matches(iid, IID_VIEW) || iid_matches(iid, IID_UNKNOWN) } {
                unsafe {
                    add_ref(this);
                    *out = this;
                }
                K_RESULT_OK
            } else {
                K_NO_INTERFACE
            }
        })
    }
}
unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            unsafe { view(this) }.refs.fetch_add(1, Ordering::Relaxed) + 1
        })
    }
}
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `0` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(0, || {
            let previous = unsafe { view(this) }.refs.fetch_sub(1, Ordering::AcqRel);
            if previous == 1 {
                unsafe {
                    drop(Box::from_raw(this.cast::<View>()));
                }
            }
            previous - 1
        })
    }
}
unsafe extern "system" fn platform(_this: *mut c_void, kind: *const c_char) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if kind.is_null() {
                return K_INVALID_ARGUMENT;
            }
            let kind = unsafe { CStr::from_ptr(kind) }.to_bytes();
            // 【为什么记这一行】"宿主问了哪个平台类型、我们答了什么"是空面板的第一现场：
            // 如果 REAPER 问的类型我们全都不认，它**根本不会调 attached**，日志里于是
            // 什么都不出现 —— 那正是最难自查的形态。按类型各自限流，稳态不会刷屏。
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                &format!("gui.platform.{}", String::from_utf8_lossy(kind)),
                format_args!(
                    "[gui] host asked platform type {:?}",
                    String::from_utf8_lossy(kind)
                ),
            );
            #[cfg(windows)]
            if kind == b"HWND" {
                return K_RESULT_OK;
            }
            // macOS：宿主给的 `NSView*`，坐标是**逻辑点**（iplugview.h）。
            #[cfg(target_os = "macos")]
            if kind == b"NSView" {
                return K_RESULT_OK;
            }
            // Linux：支持 XEmbed 的 X11 Window ID，坐标是**物理像素**。
            #[cfg(all(unix, not(target_os = "macos")))]
            if kind == b"X11EmbedWindowID" {
                return K_RESULT_OK;
            }
            // Wayland（SDK 3.8 的 `WaylandSurfaceID`）：接受，但只能开浮窗。
            //
            // 【为什么不返回 `K_RESULT_FALSE`】WebKitGTK 没有"把 `WebKitWebView` 渲染到
            // 外部 `wl_surface`"的公开 API（见方案 `6.9`）。拒绝平台类型会让宿主面板
            // 直接空着、用户没有任何界面；接受它则我们开一个自己的顶层窗口 ——
            // 功能可用，只是不嵌在宿主里。这个取舍写在 `GuiFidelity::FloatingWindow` 上。
            #[cfg(all(unix, not(target_os = "macos")))]
            if kind == b"WaylandSurfaceID" {
                return K_RESULT_OK;
            }
            K_RESULT_FALSE
        })
    }
}

unsafe extern "system" fn attached(
    this: *mut c_void,
    parent: *mut c_void,
    kind: *const c_char,
) -> TResult {
    let _guard = EnterExit::new("attached");
    if parent.is_null() || unsafe { platform(this, kind) } != K_RESULT_OK {
        // 【为什么要记】走到这里说明**宿主给了一个我们不认的平台类型**（或者 parent
        // 是空）。这是一条"宿主想开窗、我们拒绝了"的现场证据，此前完全静默。
        crate::gui_log_line(&format!(
            "attached rejected: parent={:?} type={:?}",
            parent,
            if kind.is_null() {
                "<null>".to_string()
            } else {
                unsafe { CStr::from_ptr(kind) }
                    .to_string_lossy()
                    .into_owned()
            }
        ));
        return K_INVALID_ARGUMENT;
    }
    crate::gui_log_line(&format!(
        "attached: parent=0x{:x} type={:?}",
        parent as usize,
        unsafe { CStr::from_ptr(kind) }.to_string_lossy()
    ));
    boundary(|| {
        let (width, height, generation) = {
            let mut state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if state.native.is_some() || state.attaching {
                return K_RESULT_FALSE;
            }
            let Some((width, height)) = state.rect.dimensions() else {
                return K_INVALID_ARGUMENT;
            };
            state.attaching = true;
            state.generation = state.generation.wrapping_add(1);
            (width, height, state.generation)
        };
        // 建原生视图会同步通知宿主（CreateWindowEx 的 WM_PARENTNOTIFY、XEmbed 握手），
        // 不能持有 view 锁。
        // 宿主给的是"可嵌入的父窗口"还是"我们只能自己开窗"（Wayland）。
        let embedded = unsafe { CStr::from_ptr(kind) }.to_bytes() == b"X11EmbedWindowID";
        // 宿主的 `IPlugFrame*`：Linux 用它查 `Linux::IRunLoop`（见 gui/linux.rs），
        // 另外两个平台忽略。典型宿主在 `attached` 之前就调过 `setFrame`。
        let frame = {
            let state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.frame
        };
        let native = super::gui::create_for(
            parent as usize,
            width,
            height,
            embedded,
            frame,
            unsafe { view(this) }.link.clone(),
        );
        let mut state = unsafe { view(this) }
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // 创建期间宿主可能已经 removed（或又 attached 了一次）：那这一份就是过期产物。
        if state.generation != generation || !state.attaching {
            drop(state);
            drop(native);
            return K_RESULT_FALSE;
        }
        state.attaching = false;
        match native {
            // 【Unavailable 是"结论"，不是错误】宿主拿不到 GUI 时看到的是空面板 +
            // 日志里一行可搜到的原因；§attached§ 必须如实返回失败，否则宿主会以为
            // 面板里有一个能用的编辑器（见方案 §3.4）。
            Ok(host) if matches!(host.fidelity(), super::gui::GuiFidelity::Unavailable(_)) => {
                let reason = host.fidelity().as_str();
                drop(state);
                drop(host);
                crate::gui_log_line(&format!("no embedded editor: {reason}"));
                K_RESULT_FALSE
            }
            Ok(host) => {
                let fidelity = host.fidelity();
                state.native = Some(host);
                drop(state);
                crate::gui_log_line(&format!("editor view attached ({})", fidelity.as_str()));
                // 【为什么要单独一行】"浮窗"是一种**有损**的降级：泵不在宿主线程上，
                // 宿主 Undo 的合并粒度会变粗。用户报"撤销一次退两步"时这行是关键线索。
                if matches!(fidelity, super::gui::GuiFidelity::FloatingWindow) {
                    crate::gui_log_line(
                        "running on a private GUI thread; host undo coalescing degraded",
                    );
                }
                K_RESULT_OK
            }
            Err(error) => {
                drop(state);
                crate::log_line(&format!("IPlugView attach failed: {error}"));
                K_RESULT_FALSE
            }
        }
    })
}

unsafe extern "system" fn removed(this: *mut c_void) -> TResult {
    let _guard = EnterExit::new("removed");
    boundary(|| {
        // 先移出锁，再关闭。DestroyWindow/COM Close 可能同步重入宿主。
        let native = {
            let mut state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.generation = state.generation.wrapping_add(1);
            state.attaching = false;
            state.native.take()
        };
        if let Some(host) = &native {
            host.close();
        }
        drop(native);
        K_RESULT_OK
    })
}

unsafe extern "system" fn wheel(_this: *mut c_void, _distance: f32) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || K_RESULT_FALSE)
    }
}
/// 按锁定SDK的kCommandKey/ASCII虚拟键约定识别共享编辑键，不消费宿主其它快捷键。
fn edit_key(key: u16, code: i16, mods: i16) -> Option<char> {
    if mods & 4 == 0 || mods & 2 != 0 {
        return None;
    }
    let character = if key == 0 && code >= 128 {
        code as u16 - 128 + 0x30
    } else if (1..=26).contains(&key) {
        key + b'a' as u16 - 1
    } else {
        key
    };
    let character = char::from_u32(character as u32)?.to_ascii_lowercase();
    matches!(character, 'c' | 'x' | 'v' | 'z' | 'y').then_some(character)
}
/// 宿主交付的编辑按键走实例内消息，交给原GUI的焦点、用户键位与撤销/粘贴路径。
unsafe fn forward_edit_key(
    this: *mut c_void,
    key: u16,
    code: i16,
    mods: i16,
    down: bool,
) -> TResult {
    boundary(|| {
        if let Some(character) = edit_key(key, code, if down { mods } else { mods | 4 }) {
            let native = {
                let state = unsafe { view(this) }
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                state.native.clone()
            };
            if let Some(host) = native {
                if host.forward_edit_key(character, mods, down) {
                    return K_RESULT_OK;
                }
            }
        }
        K_RESULT_FALSE
    })
}
unsafe extern "system" fn key_down(this: *mut c_void, key: u16, code: i16, mods: i16) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || unsafe {
            forward_edit_key(this, key, code, mods, true)
        })
    }
}
unsafe extern "system" fn key_up(this: *mut c_void, key: u16, code: i16, mods: i16) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || unsafe {
            forward_edit_key(this, key, code, mods, false)
        })
    }
}
/// 入口守卫：进入打一行、离开打一行。
///
/// 【为什么需要它】"宿主冻住"时日志会停在**最后一个进入、却没有对应离开**的入口上 ——
/// 那一行就是答案，不需要再猜是哪个回调、也不需要复现时的现场观察。
struct EnterExit(&'static str);

impl EnterExit {
    fn new(name: &'static str) -> Self {
        crate::gui_log_line(&format!("-> {name}"));
        Self(name)
    }
}

impl Drop for EnterExit {
    fn drop(&mut self) {
        crate::gui_log_line(&format!("<- {}", self.0));
    }
}

unsafe extern "system" fn get_size(this: *mut c_void, rect: *mut ViewRect) -> TResult {
    let _guard = EnterExit::new("get_size");
    if rect.is_null() {
        return K_INVALID_ARGUMENT;
    }
    boundary(|| {
        unsafe {
            *rect = view(this)
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .rect;
        }
        K_RESULT_OK
    })
}
unsafe extern "system" fn on_size(this: *mut c_void, rect: *mut ViewRect) -> TResult {
    let _guard = EnterExit::new("on_size");
    if rect.is_null() {
        return K_INVALID_ARGUMENT;
    }
    let new = unsafe { *rect };
    // ── 先把新矩形落盘，再做任何可能失败的事 ─────────────────────────────────
    //
    // 【这一行是"宿主冻住"的根因】宿主是靠 `getSize` / `onSize` 往返**收敛**尺寸的：
    // 它 `onSize` 一个矩形，随后 `getSize` 必须回同一个。此前这里一旦走任何提前返回
    // （矩形不合法、或 `resize` 失败），`state.rect` 就停在旧值，宿主于是永远看到
    // "我设了尺寸但插件不认" —— 它会**无限重试**，而重试跑在宿主的 UI 线程上。
    //
    // 实测日志就是这个循环：`get_size × N -> can_resize -> on_size -> get_size …`，
    // 宿主看起来"卡死"（CPU 不高、不处理输入），实际上是在这条循环里出不来。
    //
    // 所以顺序必须是：**先记录，再尽力而为地应用**，且永远回 `K_RESULT_OK` ——
    // 尺寸协商是宿主的控制回路，插件不该有能力把它卡住。
    {
        let mut state = unsafe { view(this) }
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.rect = new;
    }
    let Some((width, height)) = new.dimensions() else {
        // 矩形不合法（宽高非正）：已经如实记录，宿主下轮会自己修正。
        return K_RESULT_OK;
    };
    boundary(|| {
        let native = {
            let state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.native.clone()
        };
        if let Some(host) = native {
            // 泵不在宿主线程上时（浮窗形态），这是宿主唯一还会回来的节拍。
            host.host_tick();
            if let Err(error) = host.resize(width, height) {
                // 【为什么不回失败】见上：让宿主把"应用失败"当成"协商失败"只会让它重试
                // 到死。记录原因，返回成功，界面最多是尺寸没跟上。
                crate::log_line(&format!("IPlugView resize failed: {error}"));
            }
        }
        K_RESULT_OK
    })
}

/// 宿主明确交付焦点时进入自有WebView，不主动改变宿主失焦后的焦点归属。
unsafe extern "system" fn focus(this: *mut c_void, focused: u8) -> TResult {
    if focused == 0 {
        return K_RESULT_OK;
    }
    boundary(|| {
        let native = {
            let state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.native.clone()
        };
        if let Some(host) = native {
            host.host_tick();
            if let Err(error) = host.focus() {
                crate::log_line(&format!("IPlugView focus failed: {error}"));
                return K_RESULT_FALSE;
            }
        }
        K_RESULT_OK
    })
}

#[repr(C)]
struct UnknownVtbl {
    query: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> TResult,
    add: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}
unsafe fn frame_add(frame: usize) {
    if frame != 0 {
        let ptr = frame as *mut c_void;
        unsafe {
            ((**ptr.cast::<*const UnknownVtbl>()).add)(ptr);
        }
    }
}
unsafe fn frame_release(frame: usize) {
    if frame != 0 {
        let ptr = frame as *mut c_void;
        unsafe {
            ((**ptr.cast::<*const UnknownVtbl>()).release)(ptr);
        }
    }
}
unsafe extern "system" fn set_frame(this: *mut c_void, frame: *mut c_void) -> TResult {
    let _guard = EnterExit::new("set_frame");
    boundary(|| {
        unsafe {
            frame_add(frame as usize);
        }
        let old = {
            let mut state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            std::mem::replace(&mut state.frame, frame as usize)
        };
        unsafe {
            frame_release(old);
        }
        // 宿主换了 frame（换实例 / 重新停靠）也是一次"宿主回来了"的节拍。
        let native = {
            let state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.native.clone()
        };
        if let Some(host) = native {
            host.host_tick();
        }
        K_RESULT_OK
    })
}
unsafe extern "system" fn can_resize(_this: *mut c_void) -> TResult {
    let _guard = EnterExit::new("can_resize");
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        // 【探针】HIFISHIFTER_FIXED_SIZE=1 时声明"不可缩放"，宿主于是完全不进
        // 尺寸协商回路 —— 用来判定"打开 FX 窗口就冻结"是否来自这条回路。
        if std::env::var("HIFISHIFTER_FIXED_SIZE").is_ok_and(|value| value.trim() == "1") {
            return K_RESULT_FALSE;
        }
        crate::guard_ffi(K_RESULT_FALSE, || K_RESULT_OK)
    }
}
unsafe extern "system" fn constrain(_this: *mut c_void, rect: *mut ViewRect) -> TResult {
    {
        // 【FFI 守卫】本入口被宿主调用：panic 越界 = abort（宿主崩溃）。
        // 拦成 `K_RESULT_FALSE` 并记日志 —— 全部入口的统一纪律（thread-model.md）。
        crate::guard_ffi(K_RESULT_FALSE, || {
            if rect.is_null() {
                return K_INVALID_ARGUMENT;
            }
            let rect = unsafe { &mut *rect };
            let (Some(width), Some(height)) = (
                rect.right.checked_sub(rect.left),
                rect.bottom.checked_sub(rect.top),
            ) else {
                return K_INVALID_ARGUMENT;
            };
            if width <= 0 || height <= 0 {
                return K_INVALID_ARGUMENT;
            }
            let (Some(right), Some(bottom)) = (
                rect.left.checked_add(width.clamp(640, 16384)),
                rect.top.checked_add(height.clamp(400, 16384)),
            ) else {
                return K_INVALID_ARGUMENT;
            };
            rect.right = right;
            rect.bottom = bottom;
            K_RESULT_OK
        })
    }
}
impl Drop for View {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        drop(state.native.take());
        unsafe {
            frame_release(std::mem::take(&mut state.frame));
        }
    }
}
static VTBL: ViewVtbl = ViewVtbl {
    query,
    add_ref,
    release,
    platform,
    attached,
    removed,
    wheel,
    key_down,
    key_up,
    get_size,
    on_size,
    focus,
    set_frame,
    can_resize,
    constrain,
};

#[cfg(test)]
mod tests {
    use super::*;
    /// REAPER交付Unicode/控制字符/SDK虚拟键时，同一Ctrl+V得到同一参数粘贴入口。
    #[test]
    fn host_edit_keys_follow_sdk_modifiers_and_character_encodings() {
        for key in [b'v' as u16, b'V' as u16, 22] {
            assert_eq!(edit_key(key, 0, 4), Some('v'));
        }
        assert_eq!(edit_key(0, 166, 4), Some('v'));
        assert_eq!(edit_key(b'z' as u16, 0, 5), Some('z'));
        assert_eq!(edit_key(b'y' as u16, 0, 4), Some('y'));
        assert_eq!(edit_key(b'v' as u16, 0, 0), None);
        assert_eq!(edit_key(b'v' as u16, 0, 6), None);
        assert_eq!(edit_key(32, 0, 4), None);
    }
    #[test]
    fn actual_view_vtable_has_safe_size_and_lifetime_contracts() {
        let ptr = create_view();
        let table = unsafe { &**ptr.cast::<*const ViewVtbl>() };
        let mut rect = ViewRect::default();
        unsafe {
            assert_eq!((table.get_size)(ptr, &mut rect), K_RESULT_OK);
            assert_eq!(rect.dimensions(), Some((1100, 720)));
            assert_eq!(
                (table.on_size)(ptr, std::ptr::null_mut()),
                K_INVALID_ARGUMENT
            );
            rect.right = i32::MIN;
            assert_eq!((table.on_size)(ptr, &mut rect), K_INVALID_ARGUMENT);
            assert_eq!((table.platform)(ptr, c"NSView".as_ptr()), K_RESULT_FALSE);
            assert_eq!(
                (table.attached)(ptr, std::ptr::null_mut(), c"HWND".as_ptr()),
                K_INVALID_ARGUMENT
            );
            assert_eq!((table.key_down)(ptr, 32, 0, 0), K_RESULT_FALSE);
            assert_eq!((table.removed)(ptr), K_RESULT_OK);
            assert_eq!((table.removed)(ptr), K_RESULT_OK);
            assert_eq!((table.add_ref)(ptr), 2);
            assert_eq!((table.release)(ptr), 1);
            assert_eq!((table.release)(ptr), 0);
        }
        assert_eq!(std::mem::size_of::<ViewRect>(), 16);
    }
    #[test]
    fn constraints_enlarge_small_size_but_refuse_overflow() {
        let mut rect = ViewRect {
            left: 10,
            top: 20,
            right: 30,
            bottom: 40,
        };
        unsafe {
            assert_eq!(constrain(std::ptr::null_mut(), &mut rect), K_RESULT_OK);
        }
        assert_eq!((rect.right, rect.bottom), (650, 420));
        rect.left = i32::MIN;
        rect.right = i32::MAX;
        unsafe {
            assert_eq!(
                constrain(std::ptr::null_mut(), &mut rect),
                K_INVALID_ARGUMENT
            );
        }
    }
}
