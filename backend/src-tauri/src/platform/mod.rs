//! 平台收敛层（platform convergence layer）。
//!
//! 【这个模块存在的唯一理由】把"这台设备是什么"与"因此哪些能力成立"收在**一处**，
//! 让业务代码里不再出现 `cfg(target_os = ...)`。移动适配真正的风险不是写不出移动
//! 分支，而是同一个判断被抄成十几份、各自漂移一格 —— 到那时"桌面零回归"就无法用
//! 一次阅读来验证了。
//!
//! ## 三条纪律（设计文档 §5.3）
//!
//! 1. **cfg 只允许出现在本目录内**。业务代码问形态用 [`form`] / [`is_mobile`]，
//!    问能力用 [`profile`]。`scripts/checks/check-platform-boundary.py` 会拦下越界。
//! 2. **本目录不实现业务**。它只做三件事：回答形态、回答能力、把平台差异包成
//!    与桌面签名相同的门面（[`dialog`] / [`share`] / [`trash`] / [`resources`]）。
//! 3. **桌面路径逐字节不变**。每一个门面的桌面分支都是"转发到改动前的那次调用"，
//!    没有额外分支、没有额外日志、没有额外分配。
//!
//! ## 为什么能力是数据而不是谓词
//!
//! 移动端"没有右键"这件事，如果在每个交互面写一次 `if (isMobile)`，那么新增一个
//! 交互面就一定会漏掉一次。做成一张表之后，新增交互面只是**读表**；而表本身只有
//! 一份，漏读是可被静态门与测试发现的。前端的对应物是 `services/surfaceCapabilities.ts`
//! —— 两侧 id 必须逐字一致，由 `surfaceProfileContract` 测试钉住。

use serde::Serialize;

pub mod audio_session;
pub mod dialog;
pub mod logging;
pub mod mime;
pub mod native;
pub mod paths;
pub mod resources;
pub mod share;
pub mod trash;
pub mod window;

// ── 形态与目标系统 ──────────────────────────────────────────────────────────

/// 应用形态。
///
/// 【为什么不复用 `hostMode()`】`HostMode` 回答的是"宿主是谁"（独立 App / ARA 插件），
/// 这里回答的是"窗口有多小、输入是什么"。两者正交：插件跑在桌面（有键鼠、可多窗口），
/// 移动端是独立 App（无键鼠、单窗口）。把它们合成一个枚举，任何一个新组合都要改枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Form {
    Desktop,
    Mobile,
}

/// 目标操作系统。
///
/// 【为什么不用 `std::env::consts::OS`】它是运行期字符串，拼错不报错、比较大小写敏感，
/// 而且 `"android"` / `"ios"` 只在**目标**上成立 —— 交叉编译时宿主与目标是两个答案。
/// 编译期常量在这里是唯一正确的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OsId {
    Windows,
    Macos,
    Linux,
    Android,
    Ios,
}

/// 当前形态。**全仓唯一的形态判据**。
pub fn form() -> Form {
    if cfg!(any(target_os = "android", target_os = "ios")) {
        Form::Mobile
    } else {
        Form::Desktop
    }
}

/// 是否移动形态。等价于 `form() == Form::Mobile`，只是调用点更短。
pub fn is_mobile() -> bool {
    matches!(form(), Form::Mobile)
}

/// 当前目标系统。
pub fn os() -> OsId {
    if cfg!(target_os = "windows") {
        OsId::Windows
    } else if cfg!(target_os = "macos") {
        OsId::Macos
    } else if cfg!(target_os = "android") {
        OsId::Android
    } else if cfg!(target_os = "ios") {
        OsId::Ios
    } else {
        OsId::Linux
    }
}

// ── 能力表 ──────────────────────────────────────────────────────────────────

/// 一条能力的判定结果。
///
/// 【为什么带 `reason` 而不是只有 bool】界面上每一个被禁用的控件都欠用户一句话。
/// 布尔能回答"能不能"，回答不了"为什么"，于是界面只能退化成一句通用文案 —— 这与
/// 插件侧 `hostCapabilities` 已经踩过的坑是同一个，因此这里从一开始就带上理由。
#[derive(Debug, Clone, Serialize)]
pub struct CapabilityDecision {
    pub allowed: bool,
    /// 不可用时的理由键（前端 `i18n` 的 `MessageKey`）；可用时为 `None`。
    pub reason: Option<&'static str>,
}

impl CapabilityDecision {
    const fn yes() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }

    const fn no(reason: &'static str) -> Self {
        Self {
            allowed: false,
            reason: Some(reason),
        }
    }
}

/// 输入通道剖面。
#[derive(Debug, Clone, Serialize)]
pub struct InputSurface {
    /// 主指针通道：`"pointer"`（鼠标/笔）或 `"touch"`。
    pub primary: &'static str,
    pub hover: bool,
    pub wheel: bool,
    pub secondary_button: bool,
    pub modifier_keys: bool,
    pub pressure: bool,
    /// 设备能否同时报多个触点（硬件事实，两端都为真）。
    pub multi_touch: bool,
    /// 画布是否实现了**双指捏合缩放 + 平移**（软件事实）。
    ///
    /// 【为什么与 `multi_touch` 分开】`multi_touch` 说的是「屏幕能不能同时
    /// 报出两根手指」，那是硬件事实，桌面触摸屏也有。这一位说的是「我们的
    /// 画布有没有实现那套手势」—— 它是**产品决定**，移动端为真、桌面为假。
    /// 前端的触摸输入层只认这一位：桌面因此不注册任何手势监听器，
    /// 交互路径逐字节不变。
    pub two_finger_gesture: bool,
    /// 长按是否等价于右键（弹出对象菜单）。移动端为真。
    ///
    /// 【为什么要单独一位】长按合成 `contextmenu` 的实现早已存在
    /// （`utils/useLongPress.ts`），但「长按算不算右键」是形态约定，
    /// 不是实现细节：桌面长按左键是拖拽，绝不能弹菜单。
    pub long_press_menu: bool,
}

/// 视口剖面。
#[derive(Debug, Clone, Serialize)]
pub struct ViewportSurface {
    /// 是否存在需要让位的系统安全区（刘海 / 手势条 / 状态栏）。
    pub safe_area: bool,
    /// 真实的安全区内边距（CSS px）。
    ///
    /// 【为什么必须由后端给，而不是只用 CSS 的 `env(safe-area-inset-*)`】
    /// Android WebView 的 `env()` 只反映**刘海**（display cutout），不反映状态栏与
    /// 导航栏的高度。`enableEdgeToEdge()` 之后 WebView 铺到系统栏底下，而 CSS 拿到的
    /// 内边距是 0 —— 于是汉堡菜单、抽屉的关闭按钮、对话框标题全都压在状态栏下面，
    /// 既看不见也点不到。真实值只能从 `WindowInsets` 取（见 [`native`]）。
    ///
    /// 桌面与 iOS 恒为零：桌面没有系统栏，iOS 的 `env()` 是准的。
    pub insets: native::Insets,
    /// 用户能否自由改变窗口尺寸。
    pub resizable: bool,
    /// 能否再开第二个窗口。
    pub multi_window: bool,
}

/// 存储剖面。
#[derive(Debug, Clone, Serialize)]
pub struct StorageSurface {
    /// 能否按任意绝对路径读写（移动端的沙箱让这一条为假）。
    pub path_access: bool,
    /// 是否处于应用沙箱内（写出去的文件只有本应用能按路径再读回来）。
    pub sandboxed: bool,
    /// 导入是否必然产生一份副本（移动端没有"引用原位置"这个选项）。
    pub import_copies: bool,
    /// 能否在系统文件管理器里定位产物。
    pub reveal_in_folder: bool,
    /// 能否把产物交给系统分享面板。
    pub share_sheet: bool,
}

/// 音频剖面。
#[derive(Debug, Clone, Serialize)]
pub struct AudioSurface {
    pub output: bool,
    pub microphone: bool,
    /// 能否在后台继续出声（移动端需要额外的系统声明与审核理由）。
    pub background: bool,
    /// 系统是否会把"被来电/其它应用打断"告诉我们。
    pub focus_events: bool,
}

/// 交给前端的完整形态剖面。
///
/// 【为什么一次给全而不是逐条命令】前端在挂载期就需要它（决定外壳、令牌、停靠剖面），
/// 逐条查询会变成 N 次 IPC 且必然有人在第 N 次之前就读到了"还没有值"。一次给全，
/// 前端只需要等一次。
#[derive(Debug, Clone, Serialize)]
pub struct SurfaceProfile {
    pub form: Form,
    pub os: OsId,
    pub input: InputSurface,
    pub viewport: ViewportSurface,
    pub storage: StorageSurface,
    pub audio: AudioSurface,
    /// 能力表：id → 判定。id 与前端 `SurfaceCapabilityId` 逐字一致。
    pub capabilities: std::collections::BTreeMap<&'static str, CapabilityDecision>,
}

// 能力 id。与前端 `services/surfaceCapabilities.ts` 的 `SurfaceCapabilityId` 一一对应；
// 改名必须两侧同时改，`surfaceProfileContract.test.ts` 会核对。
const CAP_INPUT_HOVER: &str = "input.hover";
const CAP_INPUT_WHEEL: &str = "input.wheel";
const CAP_INPUT_SECONDARY: &str = "input.secondaryButton";
const CAP_INPUT_MODIFIERS: &str = "input.modifierKeys";
const CAP_INPUT_PRESSURE: &str = "input.pressure";
const CAP_INPUT_MULTI_TOUCH: &str = "input.multiTouch";
const CAP_WINDOW_MULTI: &str = "window.multi";
const CAP_WINDOW_RESIZABLE: &str = "window.resizable";
const CAP_FS_PATH_ACCESS: &str = "fs.pathAccess";
const CAP_EXPORT_SHARE: &str = "export.share";
const CAP_EXPORT_REVEAL: &str = "export.revealInFolder";
const CAP_AUDIO_BACKGROUND: &str = "audio.background";
const CAP_AUDIO_FOCUS: &str = "audio.focusEvents";
const CAP_DIALOG_NATIVE_PICKER: &str = "dialog.nativeFilePicker";

// 理由键。前端 `messages.ts` 里有同名条目（`surface_reason_*`）。
const REASON_TOUCH_INPUT: &str = "surface_reason_touch_input";
const REASON_NO_PATH_ACCESS: &str = "surface_reason_no_path_access";
const REASON_NOT_ON_THIS_SURFACE: &str = "surface_reason_not_on_this_surface";

/// 当前形态的完整剖面。
pub fn profile(app: &tauri::AppHandle) -> SurfaceProfile {
    // 【为什么要问原生】只有它知道状态栏与导航栏的真实高度（见 [`native`]）。
    // 桌面与 iOS 直接给零值 —— 那里的 `env(safe-area-inset-*)` 本来就是准的。
    let insets = if is_mobile() {
        native::insets(app)
    } else {
        native::Insets::default()
    };
    profile_with_insets(insets)
}

/// 不访问原生桥的剖面（安全区按零处理）。
///
/// 【为什么需要它】`profile()` 要问原生，而原生桥不能在主线程上调用（见
/// [`native`] 的线程约束）。取剖面的命令因此在阻塞池里跑；万一那个任务 panic，
/// 返回值仍然要是**一份完整的剖面**，而不是让启动路径上少一个字段。
pub fn profile_fallback() -> SurfaceProfile {
    profile_with_insets(native::Insets::default())
}

fn profile_with_insets(insets: native::Insets) -> SurfaceProfile {
    let mobile = is_mobile();
    let mut capabilities = std::collections::BTreeMap::new();

    // ── 输入 ────────────────────────────────────────────────────────────
    // 移动端的三条硬事实：没有悬停、没有滚轮、没有右键。修饰键也不存在 ——
    // 外接键盘确实可能有，但那不是可以依赖的默认状态（P3 的"修饰键行"就是为此而设）。
    capabilities.insert(
        CAP_INPUT_HOVER,
        if mobile {
            CapabilityDecision::no(REASON_TOUCH_INPUT)
        } else {
            CapabilityDecision::yes()
        },
    );
    capabilities.insert(
        CAP_INPUT_WHEEL,
        if mobile {
            CapabilityDecision::no(REASON_TOUCH_INPUT)
        } else {
            CapabilityDecision::yes()
        },
    );
    capabilities.insert(
        CAP_INPUT_SECONDARY,
        if mobile {
            CapabilityDecision::no(REASON_TOUCH_INPUT)
        } else {
            CapabilityDecision::yes()
        },
    );
    capabilities.insert(
        CAP_INPUT_MODIFIERS,
        if mobile {
            CapabilityDecision::no(REASON_TOUCH_INPUT)
        } else {
            CapabilityDecision::yes()
        },
    );
    // 压力：桌面数位板有，移动端屏幕没有（iOS 3D Touch 已从新机型移除）。
    capabilities.insert(CAP_INPUT_PRESSURE, CapabilityDecision::yes());
    // 多指：移动端**有**，而且它是移动端唯一的缩放通道。
    capabilities.insert(CAP_INPUT_MULTI_TOUCH, CapabilityDecision::yes());

    // ── 窗口 ────────────────────────────────────────────────────────────
    capabilities.insert(
        CAP_WINDOW_MULTI,
        if mobile {
            CapabilityDecision::no(REASON_NOT_ON_THIS_SURFACE)
        } else {
            CapabilityDecision::yes()
        },
    );
    capabilities.insert(
        CAP_WINDOW_RESIZABLE,
        if mobile {
            CapabilityDecision::no(REASON_NOT_ON_THIS_SURFACE)
        } else {
            CapabilityDecision::yes()
        },
    );

    // ── 存储 ────────────────────────────────────────────────────────────
    // 移动端的沙箱让"按绝对路径读写任意位置"不成立；文件来自系统选择器给出的
    // content URI，必须先物化成副本。这条能力为假时，前端的文件浏览器退化成
    // "工作区浏览器"（见设计文档 §5.10），导入改走 WebView 的文件选择器。
    capabilities.insert(
        CAP_FS_PATH_ACCESS,
        if mobile {
            CapabilityDecision::no(REASON_NO_PATH_ACCESS)
        } else {
            CapabilityDecision::yes()
        },
    );

    // ── 导出交付 ────────────────────────────────────────────────────────
    // 【为什么 share 在两端都"允许"但走不同实现】桌面是"在文件管理器里定位"，
    // 移动端是"写进应用文档目录并告诉用户去哪找"。两者都是"把产物交到用户手上"，
    // 因此是同一个能力位；差别在实现，不在能力。
    capabilities.insert(CAP_EXPORT_SHARE, CapabilityDecision::yes());
    capabilities.insert(
        CAP_EXPORT_REVEAL,
        if mobile {
            CapabilityDecision::no(REASON_NOT_ON_THIS_SURFACE)
        } else {
            CapabilityDecision::yes()
        },
    );

    // ── 音频 ────────────────────────────────────────────────────────────
    capabilities.insert(
        CAP_AUDIO_BACKGROUND,
        CapabilityDecision::no(REASON_NOT_ON_THIS_SURFACE),
    );
    // 【为什么两端都为假】桌面没有"音频焦点"这个概念（Windows 的 ducking 是系统
    // 单向行为，没有可订阅的回调）；移动端**有**这个机制，但把它接进引擎需要原生
    // 通道（见 `audio_session.rs` 的边界说明）。两端都为假时前端就不必去接一个
    // 还不存在的事件流 —— 能力表如实反映现状，而不是反映愿望。
    capabilities.insert(
        CAP_AUDIO_FOCUS,
        CapabilityDecision::no(REASON_NOT_ON_THIS_SURFACE),
    );

    // ── 原生文件选择器 ──────────────────────────────────────────────────
    // 移动端不用 `rfd`（它在移动目标上**编译不过**，见 `dialog.rs` 的说明），
    // 而是走 SAF（`ACTION_OPEN_DOCUMENT` / `ACTION_CREATE_DOCUMENT`），由
    // [`native`] 的原生桥发起。两端都有真正的系统选择器，因此这条能力两端都为真 ——
    // 它回答的是"这个形态有没有系统选择器"，而不是"是不是 rfd"。
    capabilities.insert(CAP_DIALOG_NATIVE_PICKER, CapabilityDecision::yes());

    SurfaceProfile {
        form: form(),
        os: os(),
        input: InputSurface {
            primary: if mobile { "touch" } else { "pointer" },
            hover: !mobile,
            wheel: !mobile,
            secondary_button: !mobile,
            modifier_keys: !mobile,
            // 压力通道：桌面（数位板）与移动端都不作为**必需**能力，保持 true 让
            // 前端沿用既有的"设备自报"判定（`penInput.ts`），不要在这里二次判断。
            pressure: true,
            multi_touch: true,
            // 触摸输入层（平移 / 捏合 / 长按菜单）只在移动形态挂载。
            two_finger_gesture: mobile,
            long_press_menu: mobile,
        },
        viewport: ViewportSurface {
            safe_area: mobile,
            insets,
            resizable: !mobile,
            multi_window: !mobile,
        },
        storage: StorageSurface {
            path_access: !mobile,
            sandboxed: mobile,
            import_copies: mobile,
            reveal_in_folder: !mobile,
            share_sheet: true,
        },
        audio: AudioSurface {
            output: true,
            microphone: true,
            background: false,
            focus_events: false,
        },
        capabilities,
    }
}

/// 应用启动期的一次性平台准备。
///
/// 【为什么在 `setup` 里调而不是 `run` 开头】音频会话、窗口与资源物化都需要
/// `AppHandle`；而日志初始化必须在 `Builder` 之前（它要在第一条 Tauri 日志之前
/// 就位）。两件事因此分成两个入口：[`logging::init`] 与这里。
pub fn init(app: &tauri::AppHandle) {
    audio_session::configure(app);
    window::prepare_main_window(app);
    log::info!(
        "surface: form={:?} os={:?} mobile={}",
        form(),
        os(),
        is_mobile()
    );
}
