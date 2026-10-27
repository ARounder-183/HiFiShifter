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

use super::{EscapeError, Fidelity, HostKeyEscape, KeyCode, Stroke, TargetKind};
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetFocus, MapVirtualKeyW, SendInput, SetFocus, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT,
    KEYBD_EVENT_FLAGS, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, MAPVK_VK_TO_VSC, VIRTUAL_KEY,
    VK_ADD, VK_BACK, VK_CONTROL, VK_DECIMAL, VK_DELETE, VK_DIVIDE, VK_DOWN, VK_END, VK_ESCAPE,
    VK_HOME, VK_INSERT, VK_LEFT, VK_MENU, VK_MULTIPLY, VK_NEXT, VK_PRIOR, VK_RETURN, VK_RIGHT,
    VK_SHIFT, VK_SPACE, VK_SUBTRACT, VK_TAB, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetAncestor, GetForegroundWindow, GetGUIThreadInfo, GetWindowThreadProcessId, IsChild,
    IsWindow, IsWindowVisible, PostMessageW, SetForegroundWindow, GA_ROOT, GA_ROOTOWNER,
    GUITHREADINFO, WM_KEYDOWN, WM_KEYUP,
};

pub(crate) struct WindowsEscape;

impl HostKeyEscape for WindowsEscape {
    fn fidelity(&self) -> Fidelity {
        Fidelity::Synthetic
    }

    fn escape(
        &mut self,
        window: usize,
        previous: Option<usize>,
        stroke: &Stroke,
        prefer_injected: bool,
    ) -> Result<(TargetKind, Fidelity), EscapeError> {
        // `Meta`（Win 键）不外送：它带系统级副作用，而宿主快捷键里几乎不用它。
        if stroke.meta {
            return Err(EscapeError::UnsupportedKey);
        }
        let key = virtual_key(stroke.code).ok_or(EscapeError::UnsupportedKey)?;
        let editor = HWND(window as *mut core::ffi::c_void);
        let host = host_window(editor)?;
        let previous = previous
            .filter(|raw| *raw != 0)
            .map(|raw| HWND(raw as *mut core::ffi::c_void));
        let (target, kind) = keyboard_target(editor, host, previous)?;
        // SAFETY: 两个句柄都来自本进程的窗口树；两个调用都是纯查询，不改宿主状态。
        if !unsafe { IsWindowVisible(target) }.as_bool() {
            // 投给一个不可见窗口等于投进虚空（桥接进程的隐藏窗口正是这种）。
            return Err(EscapeError::NoHostWindow);
        }
        // 【为什么不再要求"前台窗口 == 宿主主窗口"】插件通常活在一个**自己的顶层窗口**
        // 里（宿主的 FX 链 / 插件浮窗），那个窗口才是前台。要求相等，等于在浮窗形态下
        // 恒拒绝 —— 而这正是"让路从来没生效过"的直接原因：目标窗口取的是 owner 链顶端
        // 的主窗口，前台窗口却是宿主给插件开的那个浮窗，两者必然不同。
        if !belongs_to_host_tree(unsafe { GetForegroundWindow() }, editor, host) {
            // 前台是别的应用：此时投递，键可能落到别人身上。
            return Err(EscapeError::NotForeground);
        }
        // 【一行能定死"容器 vs 主窗口"的读数】真机上一次会话就能回答"投给谁"，
        // 不必再从"没反应"反推。限流只压重复，不压信号。
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "hostKeyRelay.target",
            format_args!(
                "[relay] target={} hwnd={:#x} class={} parent={:#x}",
                kind.as_str(),
                target.0 as usize,
                window_class(target),
                unsafe { GetAncestor(target, GA_ROOT) }.0 as usize,
            ),
        );
        // 【两条路怎么选】注入会租借焦点（用户可能看到一瞬变化），合成不会；但合成
        // 在两类情形下**注定**无效：
        //   ① 这个键的合成已经试过一次、宿主毫无反应（`prefer_injected`）；
        //   ② 带修饰键 —— `PostMessage` 不更新线程键状态，宿主会把 Ctrl+S 读成 S。
        // 其余情况仍然先走合成：便宜的先用，只有证据说不灵才升级。
        if prefer_injected || !stroke.is_bare() {
            match inject_stroke(editor, target, key, stroke) {
                Ok(()) => return Ok((kind, Fidelity::Injected)),
                // 组合键没有第二条路：合成送出的会是一个**错**的键。
                Err(error) if !stroke.is_bare() => return Err(error),
                Err(_) => {}
            }
        }
        post_stroke(target, key, stroke)?;
        Ok((kind, Fidelity::Synthetic))
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
/// 宿主里**真正会收到这个键**的那个窗口。
///
/// 【为什么不再有"Parent"这一级】上一版在这里加过第三级：焦点在我们子树里时，沿父链
/// 走出我们子树的第一层（宿主给插件开的容器窗口）。它的前提是"宿主把键位表挂在他给
/// 插件的容器上"，而这条前提**没有任何证据**，并且可以构造性证伪：
///
///   * 编辑器是 `WS_CHILD`（见 `gui/windows.rs`），所以 `GetParent(editor)` 与
///     `editor` 同属一个顶层窗口 ⇒ `belongs_to_host_tree(parent, …)` **恒真**；
///   * 于是"焦点在我们子树里"这一支**恒返回 Parent**，`RootOwner` 永远不可达；
///   * 而 `Focus` 级要求焦点不在我们子树里，可 WebView2 只在拿到焦点时才把键变成
///     DOM 事件（`WM_SETFOCUS → port.focus_page()`）—— 用户在插件里按键时它必然不可达。
///
/// 三条合起来的实际效果是：**Space 恒被投给宿主为装插件视图而建的容器子窗口**，而那个
/// 窗口的 `WndProc` 里没有宿主的键位表 ⇒ 按键进了虚空，日志里还是一行 `relayed`。
///
/// 现在只有两级，且每一级都有明确的证据：
///  1. `Focus` —— 宿主线程的焦点窗口（`GetGUIThreadInfo`），且**不在我们子树里**
///     （"用户此刻按键会去哪"；此时我们本来就不是接收方，投给它最保真）；
///  2. `RootOwner` —— owner 链顶端，也就是宿主主窗口。`ARAPlugInExtensionInstance`
///     与 VST3 视图都挂在它下面，宿主的快捷键表在主窗口的消息过程里。
fn keyboard_target(
    editor: HWND,
    host: HWND,
    previous: Option<HWND>,
) -> Result<(HWND, TargetKind), EscapeError> {
    // 1) 焦点不在我们子树里 ⇒ 就是"用户此刻按键会去的地方"。
    if let Some(focused) = focused_window(host) {
        // SAFETY: 纯查询。
        let ours = focused == editor || unsafe { IsChild(editor, focused) }.as_bool();
        if !ours && belongs_to_host_tree(focused, editor, host) {
            return Ok((focused, TargetKind::Focus));
        }
    }
    // 2) 焦点在我们子树里（= 用户就在插件里画曲线）⇒ **上一个宿主焦点窗口**。
    //
    // 【为什么它比主窗口更可能命中】宿主的快捷键表通常挂在"用户当时正在用的那个
    // 窗口"的消息过程里（轨道视图 / 编曲区 / 混音台），而不是主框架。这个句柄本来就
    // 记着（`WM_KILLFOCUS` 的 wParam），此前只是没有投递路径用它。
    if let Some(previous) = previous {
        if focus_target_belongs_to(previous, editor, host) {
            return Ok((previous, TargetKind::PreviousFocus));
        }
    }
    // 3) 兜底：owner 链顶端。焦点落在我们完全不认识的窗口上时也走这里 ——
    //    "投给未知窗口"等于投进虚空，主窗口至少是宿主自己的。
    Ok((host, TargetKind::RootOwner))
}
/// 宿主线程的焦点窗口；读不到时 `None`。
/// 窗口类名（诊断用）。读不到时返回询问符，绝不 panic。
///
/// 【为什么值一行日志】"投给容器还是投给主窗口"这个问题只能靠真机会话回答，而类名
/// （`REAPERwnd` / WebView 的容器类 / `#32770` …）是其中最直接的那一个事实。
fn window_class(window: HWND) -> String {
    use windows::Win32::UI::WindowsAndMessaging::GetClassNameW;
    let mut buffer = [0u16; 256];
    // SAFETY: 纯查询；缓冲区长度按元素个数传入。
    let len = unsafe { GetClassNameW(window, &mut buffer) };
    if len <= 0 {
        return "?".to_owned();
    }
    String::from_utf16_lossy(&buffer[..len as usize])
}

fn focused_window(host: HWND) -> Option<HWND> {
    // SAFETY: 纯查询。
    let thread = unsafe { GetWindowThreadProcessId(host, None) };
    if thread == 0 {
        return None;
    }
    let mut info = GUITHREADINFO {
        cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: 纯查询；`cbSize` 已按结构体大小填好。
    if unsafe { GetGUIThreadInfo(thread, &mut info) }.is_err() {
        return None;
    }
    (!info.hwndFocus.0.is_null()).then_some(info.hwndFocus)
}

/// 前台窗口是不是"宿主的"那一棵窗口树里的。
///
/// 【为什么不是判等】插件常驻在自己的顶层窗口里（宿主的 FX 链窗口 / 插件浮窗），
/// 那个窗口才是前台，而它的 **root owner** 才是宿主主窗口 —— 也就是我们真正要投递的
/// 目标。判等等于在浮窗形态下恒拒绝，而那恰恰是最常见的形态。
///
/// 【为什么这四条就够】它们覆盖了"宿主自己的窗口"的三种形态（主窗口本身、主窗口的
/// 子窗口、与编辑器同一条 owner 链上的兄弟浮窗）加上"主窗口就是顶层"的普通情形。
/// 前台属于**别的应用**时四条全假 —— 那种情况下投递会把键送到别人身上。
fn belongs_to_host_tree(foreground: HWND, editor: HWND, host: HWND) -> bool {
    if foreground.0.is_null() || host.0.is_null() {
        return false;
    }
    if foreground == host || foreground == editor {
        return true;
    }
    // SAFETY: 全部是纯查询。
    unsafe {
        IsChild(host, foreground).as_bool()
            || GetAncestor(foreground, GA_ROOTOWNER) == host
            || GetAncestor(foreground, GA_ROOT) == host
    }
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

/// DOM `code` 到虚拟键码。
///
/// 【为什么可以直接查表】`KeyboardEvent.code` 是**与键盘布局无关的物理键名**，
/// 与 VK 一一对应；需要 `VkKeyScanW` 那种布局换算的只有可打印字符，而它们不在
/// 可让路的表里（见 `strokes`）。
///
/// 【表外一律 None】不猜、不近似。
fn virtual_key(code: KeyCode) -> Option<VirtualKey> {
    let (key, extended) = match code.as_str() {
        "Space" => (VK_SPACE, false),
        "Enter" => (VK_RETURN, false),
        "Tab" => (VK_TAB, false),
        "Escape" => (VK_ESCAPE, false),
        "Backspace" => (VK_BACK, false),
        "Insert" => (VK_INSERT, true),
        "Delete" => (VK_DELETE, true),
        "Home" => (VK_HOME, true),
        "End" => (VK_END, true),
        "PageUp" => (VK_PRIOR, true),
        "PageDown" => (VK_NEXT, true),
        "ArrowLeft" => (VK_LEFT, true),
        "ArrowUp" => (VK_UP, true),
        "ArrowRight" => (VK_RIGHT, true),
        "ArrowDown" => (VK_DOWN, true),
        // 小键盘：数字区与主键盘同码不同键，靠扩展位区分。
        "NumpadEnter" => (VK_RETURN, true),
        "NumpadAdd" => (VK_ADD, false),
        "NumpadSubtract" => (VK_SUBTRACT, false),
        "NumpadMultiply" => (VK_MULTIPLY, false),
        "NumpadDivide" => (VK_DIVIDE, true),
        "NumpadDecimal" => (VK_DECIMAL, false),
        other => {
            // F1–F24 是连续区间（0x70 起），Numpad0–9 也是（0x60 起）——
            // 把 34 个常量逐个 import 只会让这张表更难读，区间表达更贴近 VK 的定义。
            if let Some(number) = other
                .strip_prefix('F')
                .and_then(|rest| rest.parse::<u16>().ok())
            {
                if !(1..=24).contains(&number) {
                    return None;
                }
                (VIRTUAL_KEY(0x70 + number - 1), false)
            } else {
                let digit = other
                    .strip_prefix("Numpad")
                    .and_then(|rest| rest.parse::<u16>().ok())?;
                if digit > 9 {
                    return None;
                }
                (VIRTUAL_KEY(0x60 + digit), false)
            }
        }
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

/// 把一次按键（含修饰位）按**同一条消息流**投出去。
///
/// 【为什么要补修饰键的按下/抬起】`PostMessage` 不更新线程的异步键状态，所以
/// `GetKeyState(VK_CONTROL)` 永远是"没按"。但**按消息流维护自己修饰状态的宿主**会
/// 正确看到 `Ctrl` —— 顺序固定为「修饰按下 → 键按下 → 键抬起 → 修饰抬起」，宿主
/// 不可能收到半个组合（除非它丢消息，而那与单键投递是同一个风险）。
///
/// 【已知边界】直接读 `GetKeyState` 的宿主仍然看不到修饰位。那条路只有 `SendInput`
/// 能走通，而它要求先把焦点抢到宿主窗口 —— 本功能明确不做。
fn post_stroke(host: HWND, key: VirtualKey, stroke: &Stroke) -> Result<(), EscapeError> {
    let mut modifiers: Vec<VIRTUAL_KEY> = Vec::new();
    if stroke.ctrl {
        modifiers.push(VK_CONTROL);
    }
    if stroke.shift {
        modifiers.push(VK_SHIFT);
    }
    if stroke.alt {
        modifiers.push(VK_MENU);
    }
    for modifier in &modifiers {
        post(host, modifier_key(*modifier), false)?;
    }
    post(host, key, false)?;
    post(host, key, true)?;
    for modifier in modifiers.iter().rev() {
        post(host, modifier_key(*modifier), true)?;
    }
    Ok(())
}

/// 修饰键的键描述（扫描码取不到时填 0，宿主普遍不依赖它）。
fn modifier_key(key: VIRTUAL_KEY) -> VirtualKey {
    VirtualKey {
        code: key.0,
        scan: scan_code(key),
        extended: false,
    }
}

/// 一次"把键盘借给宿主"的作用域（RAII：离开时一定还回去）。
///
/// 【为什么必须借】`SendInput` 把键投给**前台焦点窗口所属线程**的输入队列。用户此刻
/// 的焦点在 WebView 上，直接注入等于把键送回插件自己。借的动作是
/// `AttachThreadInput` + `SetFocus`：前者让跨线程的 `SetFocus` 真的生效。
///
/// 【为什么必须还】这是"租"不是"给"。不还的话用户下一次按键会落进宿主 —— 界面上
/// 看不出任何异常，只是插件忽然"不理人"了。
struct FocusLoan {
    editor: HWND,
    attached: Option<u32>,
}

impl FocusLoan {
    fn take(editor: HWND, target: HWND) -> Option<Self> {
        // SAFETY: 全部是 Win32 的焦点/线程查询与设置，句柄来自本进程窗口树。
        unsafe {
            let current = GetCurrentThreadId();
            let target_thread = GetWindowThreadProcessId(target, None);
            if target_thread == 0 {
                return None;
            }
            let attached = (target_thread != current
                && AttachThreadInput(current, target_thread, true).as_bool())
            .then_some(target_thread);
            let _ = SetFocus(Some(target));
            let focused = GetFocus();
            let landed =
                !focused.0.is_null() && (focused == target || IsChild(target, focused).as_bool());
            if !landed {
                if let Some(thread) = attached {
                    let _ = AttachThreadInput(current, thread, false);
                }
                return None;
            }
            Some(Self { editor, attached })
        }
    }
}

impl Drop for FocusLoan {
    fn drop(&mut self) {
        // SAFETY: 同 `take`；失败也不做任何补救 —— 用户点一下插件焦点就回来了。
        unsafe {
            let _ = SetFocus(Some(self.editor));
            if let Some(thread) = self.attached.take() {
                let _ = AttachThreadInput(GetCurrentThreadId(), thread, false);
            }
        }
    }
}

/// 一条 `SendInput` 键盘事件。
///
/// 【为什么用虚拟键码而不是扫描码】宿主判定快捷键读的是 VK；扫描码只在少数
/// 游戏/低层输入里才有意义，而 `MapVirtualKeyW` 对某些键（多媒体键）会返回 0。
/// 填 VK 是这里唯一稳的选择。
fn key_input(key: VirtualKey, up: bool) -> INPUT {
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if key.extended {
        flags.0 |= KEYEVENTF_EXTENDEDKEY.0;
    }
    if up {
        flags.0 |= KEYEVENTF_KEYUP.0;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(key.code),
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

/// 用**真实输入**投递一次按键。
///
/// 【为什么它比 PostMessage 可靠】`SendInput` 走系统输入队列：宿主线程的键状态、
/// 焦点窗口、`TranslateAccelerator` 全都看得到它 —— 而 `PostMessage` 只把一条消息
/// 塞进目标窗口的队列，绕过上述全部判据。VEGAS / Cubase 这类在自己消息循环里按焦点
/// 判快捷键的宿主，只认前者。
///
/// 【为什么整组一次提交】修饰按下 → 键按下 → 键抬起 → 修饰抬起 必须是一个原子序列；
/// 分成多次调用时宿主可能在中途插进来，看到"半个组合键"。
fn inject_stroke(
    editor: HWND,
    target: HWND,
    key: VirtualKey,
    stroke: &Stroke,
) -> Result<(), EscapeError> {
    let _loan = FocusLoan::take(editor, target).ok_or(EscapeError::PostFailed)?;
    let mut events: Vec<INPUT> = Vec::with_capacity(8);
    if stroke.ctrl {
        events.push(key_input(modifier_key(VK_CONTROL), false));
    }
    if stroke.shift {
        events.push(key_input(modifier_key(VK_SHIFT), false));
    }
    if stroke.alt {
        events.push(key_input(modifier_key(VK_MENU), false));
    }
    events.push(key_input(key, false));
    events.push(key_input(key, true));
    if stroke.alt {
        events.push(key_input(modifier_key(VK_MENU), true));
    }
    if stroke.shift {
        events.push(key_input(modifier_key(VK_SHIFT), true));
    }
    if stroke.ctrl {
        events.push(key_input(modifier_key(VK_CONTROL), true));
    }
    // SAFETY: 事件数组是本地 `Vec`，长度与 `cbSize` 都是编译期常量结构体的大小。
    let sent = unsafe { SendInput(&events, std::mem::size_of::<INPUT>() as i32) };
    if sent != events.len() as u32 {
        return Err(EscapeError::PostFailed);
    }
    Ok(())
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

    fn code(name: &str) -> KeyCode {
        KeyCode::parse(name).expect(name)
    }

    fn bare(name: &str) -> Stroke {
        Stroke {
            code: code(name),
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }
    }

    /// **不变量：记下来的"上一个宿主焦点窗口"优先于主窗口。**
    ///
    /// 【为什么钉死】VEGAS 的日志证明目标**已经是**主窗口（`class=Vegas.Class.Frame`）
    /// 却仍然 `no-response` —— 说明"投给谁"还有一层没用上：用户按键之前正在用的那个
    /// 宿主窗口。这个句柄本来就记着（`WM_KILLFOCUS` 的 wParam），此前没有投递路径用过。
    ///
    /// 句柄 0 与非法句柄必须落回 `RootOwner`，而不是"投进虚空"。
    #[test]
    fn a_missing_previous_focus_falls_back_to_the_root_owner() {
        let editor = HWND(std::ptr::null_mut());
        let host = HWND(std::ptr::dangling_mut::<core::ffi::c_void>());
        assert_eq!(
            keyboard_target(editor, host, None).unwrap().1,
            TargetKind::RootOwner
        );
        let stranger = HWND(std::ptr::without_provenance_mut(2));
        assert_eq!(
            keyboard_target(editor, host, Some(stranger)).unwrap().1,
            TargetKind::RootOwner,
            "认不出来的句柄必须被拒绝，而不是当成目标"
        );
    }

    /// **不变量：租不到焦点就不注入，而且不留下任何状态。**
    ///
    /// 【为什么钉死】`FocusLoan` 是 RAII：失败路径必须自己把 `AttachThreadInput` 解掉。
    /// 这里用非法句柄走一遍失败分支 —— 它同时证明"拿不到焦点时返回 None"这条契约。
    #[test]
    fn a_focus_loan_is_refused_when_the_target_is_not_a_window() {
        let editor = HWND(std::ptr::null_mut());
        let target = HWND(std::ptr::without_provenance_mut(2));
        assert!(FocusLoan::take(editor, target).is_none());
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

    /// **不变量：键表里每一项都有虚拟键码。**
    ///
    /// 【为什么是回归测试】表与换算各写一份时，最容易发生的漂移是"表里加了、换算忘了"，
    /// 表现是那个键按了毫无反应、且没有任何提示。
    #[test]
    fn every_relayable_code_has_a_virtual_key() {
        for name in super::super::strokes::relayable_codes() {
            let parsed = KeyCode::parse(name).expect("表里的每一项都必须能解析");
            assert!(
                virtual_key(parsed).is_some(),
                "{name} 在键表里却没有虚拟键码"
            );
        }
    }

    /// 方向键/翻页键是扩展键，文本类不是 —— 宿主编排逻辑依赖这个分类。
    #[test]
    fn navigation_keys_are_extended_and_text_keys_are_not() {
        for name in ["Space", "Enter", "Tab", "Escape", "Backspace"] {
            assert!(!virtual_key(code(name)).unwrap().extended, "{name}");
        }
        for name in [
            "Insert",
            "Delete",
            "Home",
            "End",
            "PageUp",
            "PageDown",
            "ArrowLeft",
            "ArrowUp",
            "ArrowRight",
            "ArrowDown",
        ] {
            assert!(virtual_key(code(name)).unwrap().extended, "{name}");
        }
    }

    /// 功能键与小键盘落在 VK 的两个连续区间上。
    #[test]
    fn function_and_numpad_keys_land_on_the_vk_ranges() {
        assert_eq!(virtual_key(code("F1")).unwrap().code, 0x70);
        assert_eq!(virtual_key(code("F24")).unwrap().code, 0x87);
        assert_eq!(virtual_key(code("Numpad0")).unwrap().code, 0x60);
        assert_eq!(virtual_key(code("Numpad9")).unwrap().code, 0x69);
        // 小键盘回车与主键盘回车同码，只有扩展位能区分。
        assert_eq!(virtual_key(code("NumpadEnter")).unwrap().code, VK_RETURN.0);
        assert!(virtual_key(code("NumpadEnter")).unwrap().extended);
        assert!(!virtual_key(code("Enter")).unwrap().extended);
    }

    /// 投递前的拒绝必须给出**结构化**原因，而不是 panic 或静默。
    #[test]
    fn a_dead_window_is_refused_rather_than_posted_to() {
        // 0 号句柄不是窗口：必须得到明确原因，不能投递、不能 panic。
        let mut escape = WindowsEscape;
        assert_eq!(
            escape.escape(0, None, &bare("Space"), false),
            Err(EscapeError::NoHostWindow)
        );
    }

    /// **不变量：拿不到宿主窗口时，交还必须失败而不是假装成功。**
    ///
    /// 【为什么钉死】"把键盘交给宿主"唯一的失败形态就是**什么都没发生**。若这里
    /// 返回 `Ok`，调用方会告诉用户"已交还"，而焦点其实还在插件里。
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

    /// **不变量：前台判据看的是"窗口树归属"，不是"等于主窗口"。**
    ///
    /// 【为什么这是本轮的关键回归】插件通常活在自己的顶层窗口里（宿主的 FX 链 /
    /// 插件浮窗），那个窗口才是前台；而投递目标是它的 **root owner**（宿主主窗口）。
    /// 两者恒不相等 —— 旧判据因此在浮窗形态下**一次都没投递过**，而那正是"让路按了
    /// 没反应"的直接原因。
    #[test]
    fn a_foreground_window_is_accepted_by_tree_membership_not_by_equality() {
        // 句柄 0 一律拒绝（没有前台窗口 / 没有宿主）。
        let host = HWND(std::ptr::dangling_mut::<core::ffi::c_void>());
        assert!(!belongs_to_host_tree(HWND::default(), host, host));
        assert!(!belongs_to_host_tree(host, host, HWND::default()));
        // 非窗口句柄：四个查询都会失败或返回 null ⇒ 必须拒绝，而不是"碰巧通过"。
        let stranger = HWND(std::ptr::without_provenance_mut(2));
        assert!(
            !belongs_to_host_tree(stranger, host, host),
            "不认识的前台窗口必须被拒绝"
        );
    }

    /// **不变量：带修饰的组合允许外送（走注入），`Meta` 不允许。**
    ///
    /// 【为什么与上一轮不同】合成消息确实不能忠实转交组合键（不更新异步键状态），
    /// 但注入可以 —— 所以"带修饰"不再是**拒绝**的理由，而是"必须走注入"的理由
    /// （见 `escape` 里的两条路）。`Meta`（Win 键）仍然拒绝：它带系统级副作用，
    /// 外送一次就可能把开始菜单弹出来。
    #[test]
    fn a_modified_stroke_is_relayed_but_meta_is_refused() {
        let mut escape = WindowsEscape;
        let mut stroke = bare("Space");
        stroke.ctrl = true;
        // 句柄 0 不是窗口 ⇒ 走到「找不到宿主窗口」，而不是「这个键不支持」。
        assert_eq!(
            escape.escape(0, None, &stroke, false),
            Err(EscapeError::NoHostWindow)
        );
        stroke.meta = true;
        assert_eq!(
            escape.escape(0, None, &stroke, false),
            Err(EscapeError::UnsupportedKey)
        );
    }
}
