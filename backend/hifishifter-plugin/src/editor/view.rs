//! 按锁定iplugview.h实现的原生VST3视图ABI；只操作自有子窗口。
use crate::vst3::{
    iid_matches, TResult, K_INVALID_ARGUMENT, K_NO_INTERFACE, K_RESULT_FALSE, K_RESULT_OK,
};
use std::ffi::{c_char, c_void, CStr};
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
    #[cfg(windows)]
    native: Option<super::webview::NativeEditor>,
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
}
unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    unsafe { view(this) }.refs.fetch_add(1, Ordering::Relaxed) + 1
}
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    let previous = unsafe { view(this) }.refs.fetch_sub(1, Ordering::AcqRel);
    if previous == 1 {
        unsafe {
            drop(Box::from_raw(this.cast::<View>()));
        }
    }
    previous - 1
}
unsafe extern "system" fn platform(_this: *mut c_void, kind: *const c_char) -> TResult {
    if kind.is_null() {
        return K_INVALID_ARGUMENT;
    }
    #[cfg(windows)]
    if unsafe { CStr::from_ptr(kind) }.to_bytes() == b"HWND" {
        return K_RESULT_OK;
    }
    K_RESULT_FALSE
}
unsafe extern "system" fn attached(
    this: *mut c_void,
    parent: *mut c_void,
    kind: *const c_char,
) -> TResult {
    if parent.is_null() || unsafe { platform(this, kind) } != K_RESULT_OK {
        return K_INVALID_ARGUMENT;
    }
    boundary(|| {
        #[cfg(windows)]
        {
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
            // CreateWindowEx会同步通知宿主父窗口，不能持有view锁。
            let native = super::webview::NativeEditor::attach(
                parent,
                width,
                height,
                unsafe { view(this) }.link.clone(),
            );
            let mut state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if state.generation != generation || !state.attaching {
                drop(state);
                drop(native);
                return K_RESULT_FALSE;
            }
            state.attaching = false;
            match native {
                Ok(native) => {
                    state.native = Some(native);
                    K_RESULT_OK
                }
                Err(error) => {
                    crate::log_line(&format!("IPlugView attach failed: {error}"));
                    K_RESULT_FALSE
                }
            }
        }
        #[cfg(not(windows))]
        {
            K_RESULT_FALSE
        }
    })
}
unsafe extern "system" fn removed(this: *mut c_void) -> TResult {
    boundary(|| {
        #[cfg(windows)]
        {
            // 先移出锁，再关闭。DestroyWindow/COM Close可能同步重入宿主。
            let native = {
                let mut state = unsafe { view(this) }
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                state.generation = state.generation.wrapping_add(1);
                state.attaching = false;
                state.native.take()
            };
            drop(native);
        }
        K_RESULT_OK
    })
}
unsafe extern "system" fn wheel(_this: *mut c_void, _distance: f32) -> TResult {
    K_RESULT_FALSE
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
        #[cfg(windows)]
        if let Some(character) = edit_key(key, code, if down { mods } else { mods | 4 }) {
            let native = {
                let state = unsafe { view(this) }
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                state
                    .native
                    .as_ref()
                    .map(super::webview::NativeEditor::window_key)
            };
            if let Some(native) = native {
                if native.forward_edit_key(character, mods, down) {
                    return K_RESULT_OK;
                }
            }
        }
        K_RESULT_FALSE
    })
}
unsafe extern "system" fn key_down(this: *mut c_void, key: u16, code: i16, mods: i16) -> TResult {
    unsafe { forward_edit_key(this, key, code, mods, true) }
}
unsafe extern "system" fn key_up(this: *mut c_void, key: u16, code: i16, mods: i16) -> TResult {
    unsafe { forward_edit_key(this, key, code, mods, false) }
}
unsafe extern "system" fn get_size(this: *mut c_void, rect: *mut ViewRect) -> TResult {
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
    if rect.is_null() {
        return K_INVALID_ARGUMENT;
    }
    let new = unsafe { *rect };
    let Some((_width, _height)) = new.dimensions() else {
        return K_INVALID_ARGUMENT;
    };
    boundary(|| {
        #[cfg(windows)]
        let native = {
            let state = unsafe { view(this) }
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state
                .native
                .as_ref()
                .map(super::webview::NativeEditor::window_key)
        };
        #[cfg(windows)]
        if let Some(native) = native {
            if let Err(error) = native.resize(_width, _height) {
                crate::log_line(&format!("IPlugView resize failed: {error}"));
                return K_RESULT_FALSE;
            }
        }
        let mut state = unsafe { view(this) }
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.rect = new;
        K_RESULT_OK
    })
}
/// 宿主明确交付焦点时进入自有WebView，不主动改变宿主失焦后的焦点归属。
unsafe extern "system" fn focus(this: *mut c_void, focused: u8) -> TResult {
    if focused == 0 {
        return K_RESULT_OK;
    }
    boundary(|| {
        #[cfg(windows)]
        {
            let native = {
                let state = unsafe { view(this) }
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                state
                    .native
                    .as_ref()
                    .map(super::webview::NativeEditor::window_key)
            };
            if let Some(native) = native {
                if let Err(error) = native.focus() {
                    crate::log_line(&format!("IPlugView focus failed: {error}"));
                    return K_RESULT_FALSE;
                }
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
        K_RESULT_OK
    })
}
unsafe extern "system" fn can_resize(_this: *mut c_void) -> TResult {
    K_RESULT_OK
}
unsafe extern "system" fn constrain(_this: *mut c_void, rect: *mut ViewRect) -> TResult {
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
}
impl Drop for View {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        #[cfg(windows)]
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
