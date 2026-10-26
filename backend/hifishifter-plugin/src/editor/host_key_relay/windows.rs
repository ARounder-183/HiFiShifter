//! Windows 原语：把按键**重投**到宿主顶层窗口的消息队列。
//!
//! 【为什么必须这么做】Windows 的键盘消息**不向父窗口冒泡**。焦点落在 WebView2 的孙窗口
//! 上时，`WM_KEYDOWN` 由 Chromium 收下并变成 DOM 事件；宿主的消息循环拿到的是
//! `msg.hwnd = WebView2 子窗口`，`DispatchMessage` 之后没有任何机制把按键
//! "还给"宿主。VST3 的 `IPlugView::onKeyDown` 也帮不上：它只在宿主**主动调用**时
//! 才有机会，而 HWND 型视图的按键根本不经过宿主。
//!
//! 【为什么是 `PostMessage` 而不是 `SendInput`】`SendInput` 要求先把焦点抢到宿主
//! 窗口 —— 用户按一次 Space，焦点就被搬走，之后想继续编辑必须先点回来。保真度不值得
//! 用焦点稳定性去换：`PostMessage` 已经足够让宿主执行动作，且不动焦点。
//!
//! 【为什么保真度只敢报 `Synthetic`】投递消息不更新线程的异步键状态，
//! 也不保证宿主一定会处理（有的宿主只看自己的前台焦点窗口）。因此 Policy 仍要先探针、
//! 之后再观察窗口判定"成没成"，而不是假定它一定灵。

use super::{EscapeError, Fidelity, HostKeyEscape, KeyToken, Stroke};
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetFocus, MapVirtualKeyW, SetFocus, MAPVK_VK_TO_VSC, VIRTUAL_KEY, VK_BACK, VK_DELETE, VK_DOWN,
    VK_END, VK_ESCAPE, VK_HOME, VK_INSERT, VK_LEFT, VK_NEXT, VK_PRIOR, VK_RETURN, VK_RIGHT,
    VK_SPACE, VK_TAB, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetAncestor, GetForegroundWindow, GetWindowThreadProcessId, IsChild, IsWindow, IsWindowVisible,
    PostMessageW, SetForegroundWindow, GA_ROOT, GA_ROOTOWNER, WM_KEYDOWN, WM_KEYUP,
};

pub(crate) struct WindowsEscape;

impl HostKeyEscape for WindowsEscape {
    fn fidelity(&self) -> Fidelity {
        Fidelity::Synthetic
    }

    fn escape(&mut self, window: usize, stroke: &Stroke) -> Result<(), EscapeError> {
        // 前端已经挡过带修饰的键；这里再挡一次 —— 原生入口不能假定调用方守规矩。
        if !stroke.is_bare() {
            return Err(EscapeError::UnsupportedKey);
        }
        let key = virtual_key(stroke.token).ok_or(EscapeError::UnsupportedKey)?;
        let editor = HWND(window as *mut core::ffi::c_void);
        let host = host_window(editor)?;
        // SAFETY: 两个句柄都来自本进程的窗口树；两个调用都是纯查询，不改宿主状态。
        if !unsafe { IsWindowVisible(host) }.as_bool() {
            // 投给一个不可见窗口等于投进虚空（桥接进程的隐藏窗口正是这种）。
            return Err(EscapeError::NoHostWindow);
        }
        if unsafe { GetForegroundWindow() } != host {
            // 宿主不在前台时投递，键可能落到别人身上。
            return Err(EscapeError::NotForeground);
        }
        post(host, key, false)?;
        post(host, key, true)?;
        Ok(())
    }

    fn focus_host(&mut self, window: usize, previous: Option<usize>) -> Result<(), EscapeError> {
        let editor = HWND(window as *mut core::ffi::c_void);
        let host = host_window(editor)?;
        let target = previous
            .map(|raw| HWND(raw as *mut core::ffi::c_void))
            .filter(|candidate| focus_target_belongs_to(*candidate, editor, host))
            .unwrap_or(host);
        hand_focus(target, editor)
    }
}

/// 宿主窗口：编辑器 HWND 所属的、真正处理用户快捷键的那个窗口。
///
/// 【为什么不是 `GA_ROOT`】插件视图的父窗口在多数宿主里**本身就是顶层窗口**
/// （REAPER 的 FX 链窗口、VEGAS 的插件浮窗都是）。`GA_ROOT` 会停在那里，于是
/// 「把键盘交给宿主」把焦点交给了一个不翻译快捷键的窗口 —— 用户按 Space 依旧没反应，
/// 而功能看起来"完全没生效"。
///
/// `GA_ROOTOWNER` 会沿 **owner 链**继续向上，拿到真正的主窗口。取不到时才退回
/// `GA_ROOT`；两者都拿不到、或拿到的就是编辑器自己（视图还没挂进宿主窗口树），
/// 则如实失败。
fn host_window(editor: HWND) -> Result<HWND, EscapeError> {
    // SAFETY: 纯查询，返回本线程窗口树里的祖先句柄。
    let owner = unsafe { GetAncestor(editor, GA_ROOTOWNER) };
    let host = if owner.0.is_null() || owner == editor {
        // SAFETY: 同上。
        unsafe { GetAncestor(editor, GA_ROOT) }
    } else {
        owner
    };
    if host.0.is_null() || host == editor {
        return Err(EscapeError::NoHostWindow);
    }
    Ok(host)
}

/// 记下来的"上一个焦点窗口"还能不能当目标。
///
/// 判据只有三条：还活着、不在我们的子树里、与宿主主窗口同属一棵 owner 树。
/// 少任何一条都可能把焦点交给一个已销毁或与我们无关的窗口 —— 那比不交还更糟。
fn focus_target_belongs_to(candidate: HWND, editor: HWND, host: HWND) -> bool {
    // SAFETY: 全部是纯查询。
    unsafe {
        if !IsWindow(Some(candidate)).as_bool() {
            return false;
        }
        if candidate == editor || IsChild(editor, candidate).as_bool() {
            return false;
        }
        let candidate_root = GetAncestor(candidate, GA_ROOTOWNER);
        candidate_root == host || GetAncestor(candidate, GA_ROOT) == host
    }
}

/// 把焦点交给 `target`，并在返回前**验证**它真的落到了宿主侧。
///
/// 【为什么需要 `AttachThreadInput`】`SetFocus` 只对调用线程自己的窗口队列生效。
/// 宿主主窗口与插件视图可能不属于同一条线程（桥接形态尤其如此），直接调用会
/// 返回失败而焦点纹丝不动。临时挂到目标线程的输入队列上，是这个 API 的标准用法，
/// 成功后必须立刻解挂 —— 挂着会让两条线程共享输入状态。
///
/// 【为什么要 `SetForegroundWindow`】焦点不能落到一个不在前台的窗口上；宿主在
/// 后台时这一步会被系统拒绝，但那不是错误：目标窗口本身就是前台的宿主主窗口时，
/// 它只是返回假，随后的 `SetFocus` 仍然成立。
fn hand_focus(target: HWND, editor: HWND) -> Result<(), EscapeError> {
    // SAFETY: 全部是 Win32 的纯查询 / 焦点操作，句柄来自本进程的窗口树。
    unsafe {
        let current = GetCurrentThreadId();
        let target_thread = GetWindowThreadProcessId(target, None);
        let attached = target_thread != 0
            && target_thread != current
            && AttachThreadInput(current, target_thread, true).as_bool();
        let _ = SetForegroundWindow(target);
        let result = SetFocus(Some(target));
        if attached {
            let _ = AttachThreadInput(current, target_thread, false);
        }
        let focused = GetFocus();
        let landed = !focused.0.is_null()
            && focused != editor
            && !IsChild(editor, focused).as_bool()
            && (focused == target || IsChild(target, focused).as_bool());
        if landed {
            return Ok(());
        }
        // 失败时把 Win32 的错误留在结构化的原因里，而不是只回一个布尔。
        result.map(|_| ()).map_err(|_| EscapeError::PostFailed)?;
        Err(EscapeError::PostFailed)
    }
}

/// 一个要投递的键：虚拟键码 + 扫描码 + 是否扩展键。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct VirtualKey {
    code: u16,
    scan: u32,
    extended: bool,
}

/// token 到虚拟键码。
///
/// 【为什么只有这张表】只让**与键盘布局无关**的具名键（见 `strokes::RELAYABLE_TOKENS`）。
/// 表外一律 `None`：不猜、不近似。
fn virtual_key(token: KeyToken) -> Option<VirtualKey> {
    let (key, extended) = match token {
        KeyToken::Space => (VK_SPACE, false),
        KeyToken::Enter => (VK_RETURN, false),
        KeyToken::Tab => (VK_TAB, false),
        KeyToken::Escape => (VK_ESCAPE, false),
        KeyToken::Backspace => (VK_BACK, false),
        KeyToken::Insert => (VK_INSERT, true),
        KeyToken::Delete => (VK_DELETE, true),
        KeyToken::Home => (VK_HOME, true),
        KeyToken::End => (VK_END, true),
        KeyToken::PageUp => (VK_PRIOR, true),
        KeyToken::PageDown => (VK_NEXT, true),
        KeyToken::ArrowLeft => (VK_LEFT, true),
        KeyToken::ArrowUp => (VK_UP, true),
        KeyToken::ArrowRight => (VK_RIGHT, true),
        KeyToken::ArrowDown => (VK_DOWN, true),
    };
    Some(VirtualKey {
        code: key.0,
        scan: scan_code(key),
        extended,
    })
}

/// 虚拟键码到扫描码（set 1）。读不到就填 0：宿主普遍不依赖它，但填对比填错好。
fn scan_code(key: VIRTUAL_KEY) -> u32 {
    // SAFETY: 纯查询，无副作用。
    (unsafe { MapVirtualKeyW(u32::from(key.0), MAPVK_VK_TO_VSC) }) & 0xff
}

/// 组装 `WM_KEYDOWN` / `WM_KEYUP` 的 `lParam`（MSDN 的位域）。
///
/// bit0-15 重复次数、16-23 扫描码、24 扩展键、29 上下文（Alt，恒 0）、
/// 30 前一状态（按下为 0）、31 转换状态（抬起为 1）。
///
/// 【为什么在 `u32` 域里拼完再转 `isize`】bit31 置位时若按 `i32` 计算会得到负数，
/// `as isize` 会把高位一起符号扩展，宿主收到的是一个带 0xFFFFFFFF 前缀的
/// `lParam`。先在无符号域里拼好，语义才对。
fn key_lparam(key: VirtualKey, up: bool) -> isize {
    let repeat: u32 = 1;
    let scan = (key.scan & 0xff) << 16;
    let extended = u32::from(key.extended) << 24;
    let previous = u32::from(up) << 30;
    let transition = u32::from(up) << 31;
    (repeat | scan | extended | previous | transition) as isize
}

fn post(host: HWND, key: VirtualKey, up: bool) -> Result<(), EscapeError> {
    let message = if up { WM_KEYUP } else { WM_KEYDOWN };
    // SAFETY: 同进程投递，不阻塞、不等待；目标句柄在上面的可见性/前台检查里已确认。
    unsafe {
        PostMessageW(
            Some(host),
            message,
            WPARAM(key.code as usize),
            LPARAM(key_lparam(key, up)),
        )
    }
    .map_err(|_| EscapeError::PostFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare(token: KeyToken) -> Stroke {
        Stroke {
            token,
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }
    }

    /// **不变量：lParam 的位域逐位正确。**
    ///
    /// 【为什么钉死】宿主用 lParam 判断"这是不是抬起""是不是扩展键"。位域错一位，
    /// 宿主会把它当成另一个键 —— 而表现只是"按了没反应"，最难查的一类。
    #[test]
    fn the_lparam_bitfields_match_the_win32_contract() {
        let space = VirtualKey {
            code: VK_SPACE.0,
            scan: 0x39,
            extended: false,
        };
        // 按下：重复 1、扫描码 0x39、非扩展、前一状态 0、转换状态 0。
        assert_eq!(key_lparam(space, false), 0x0039_0001);
        // 抬起：前一状态与转换状态都置 1，最高位因此为 1 —— 必须仍是**无符号**拼法。
        assert_eq!(key_lparam(space, true) as u32 as u64, 0xC039_0001);

        let left = VirtualKey {
            code: VK_LEFT.0,
            scan: 0x4B,
            extended: true,
        };
        assert_eq!(key_lparam(left, false), 0x014B_0001, "扩展键必须置 bit24");
        assert_eq!(key_lparam(left, true) as u32 as u64, 0xC14B_0001);
    }

    /// **不变量：token 表里每一项都有虚拟键码。**
    ///
    /// 【为什么是回归测试】两处各写一张表时，最容易发生的漂移是"token 收进来了、
    /// 键码忘了加"，表现是那个键按了毫无反应。
    #[test]
    fn every_relayable_token_has_a_virtual_key() {
        for token in super::super::strokes::relayable_tokens() {
            let parsed = KeyToken::parse(token).expect("表里的每一项都必须能解析");
            assert!(
                virtual_key(parsed).is_some(),
                "{token} 在 token 表里却没有虚拟键码"
            );
        }
    }

    /// 方向键/翻页键是扩展键，文本类不是 —— 宿主编排逻辑依赖这个分类。
    #[test]
    fn navigation_keys_are_extended_and_text_keys_are_not() {
        for token in [
            KeyToken::Space,
            KeyToken::Enter,
            KeyToken::Tab,
            KeyToken::Escape,
            KeyToken::Backspace,
        ] {
            assert!(!virtual_key(token).unwrap().extended, "{token:?}");
        }
        for token in [
            KeyToken::Insert,
            KeyToken::Delete,
            KeyToken::Home,
            KeyToken::End,
            KeyToken::PageUp,
            KeyToken::PageDown,
            KeyToken::ArrowLeft,
            KeyToken::ArrowUp,
            KeyToken::ArrowRight,
            KeyToken::ArrowDown,
        ] {
            assert!(virtual_key(token).unwrap().extended, "{token:?}");
        }
    }

    /// 投递前的拒绝必须给出**结构化**原因，而不是 panic 或静默。
    #[test]
    fn a_dead_window_is_refused_rather_than_posted_to() {
        // 0 号句柄不是窗口：必须得到明确原因，不能投递、不能 panic。
        let mut escape = WindowsEscape;
        assert_eq!(
            escape.escape(0, &bare(KeyToken::Space)),
            Err(EscapeError::NoHostWindow)
        );
    }

    /// **不变量：拿不到宿主窗口时，交还必须失败而不是假装成功。**
    ///
    /// 【为什么钉死】"把键盘交给宿主"唯一的失败形态就是**什么都没发生**。若这里
    /// 返回 `Ok`，调用方会告诉用户"已交还"，而焦点其实还在插件里 —— 那正是本次
    /// 报障的原始体验。
    #[test]
    fn an_empty_window_handle_is_refused_instead_of_reported_as_handed_back() {
        let mut escape = WindowsEscape;
        assert_eq!(escape.focus_host(0, None), Err(EscapeError::NoHostWindow));
    }

    /// 记下来的"上一个焦点窗口"若已销毁，绝不能被当成目标。
    #[test]
    fn a_dead_previous_focus_target_is_never_used() {
        let dead = HWND(std::ptr::null_mut());
        let editor = HWND(std::ptr::null_mut());
        let host = HWND(std::ptr::null_mut());
        assert!(!focus_target_belongs_to(dead, editor, host));
    }

    /// 带修饰的键在原生侧同样被拒 —— 不能假定调用方守规矩。
    #[test]
    fn a_modified_stroke_is_refused() {
        let mut escape = WindowsEscape;
        let stroke = Stroke {
            token: KeyToken::Space,
            ctrl: true,
            shift: false,
            alt: false,
            meta: false,
        };
        assert_eq!(escape.escape(0, &stroke), Err(EscapeError::UnsupportedKey));
    }
}
