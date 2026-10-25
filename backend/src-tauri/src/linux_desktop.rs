//! Linux 桌面会话的启动前准备：逃生开关、会话诊断、EGL 预检。
//!
//! 【为什么需要它】Linux 上的启动失败几乎从来不会留下线索：WebKitGTK 在 EGL 初始化
//! 失败时直接 abort（打印一句 `Could not create default EGL display:
//! EGL_BAD_PARAMETER. Aborting...` 就没了），用户看到的是"双击没反应"。而这一类
//! 故障多半与"哪个后端在跑、哪一份 libEGL 被加载"有关 —— 恰恰是日志里从来没有的
//! 信息。0.1.0-beta.13 的用户反馈（Arch/CachyOS + AMD RX560）就是这样：只能靠
//! 用户自己在终端里反复试 LD_PRELOAD 才定位到 Wayland 库上。
//!
//! 所以这里做三件不侵入的事：
//!
//! 1. 把两个逃生开关翻译成 GTK / WebKit 认的环境变量：
//!    * `HIFISHIFTER_GDK_BACKEND=x11|wayland` —— 覆盖 AppImage 里写死的
//!      `GDK_BACKEND=x11`（它会强制 Wayland 会话走 XWayland）；
//!    * `HIFISHIFTER_SAFE_GL=1` —— 关掉 WebKit 的合成器并强制软件 GL，
//!      用于 GPU 驱动有问题时"先让界面出来"。
//!
//! 2. 启动时打一行会话摘要（后端、DISPLAY、WAYLAND_DISPLAY、是否 AppImage）。
//!    下一次有人报"起不来"，这一行就能把范围缩到后端/驱动上。
//!
//! 3. 预检 EGL：能拿到 vendor/version 就打一行 Info；拿不到就打 Warn 并给出
//!    `HIFISHIFTER_SAFE_GL=1` 的提示；如果连 `libEGL.so.1` 都加载不了（WebKit
//!    必然 abort），则直接降级为软件合成，让界面先出来。
//!
//! 【为什么预检不 dlclose】探针之后 WebKit 立刻要用同一个库；卸载再加载会把 Mesa
//! 的全局状态多折腾一遍，收益为零。探针创建的 EGLDisplay 也保持存活：它等价于
//! WebKit 自己那条 surfaceless 路径，重复初始化是幂等的。

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// 覆盖 `GDK_BACKEND`（`x11` / `wayland`）。
pub const ENV_GDK_BACKEND: &str = "HIFISHIFTER_GDK_BACKEND";
/// 强制软件渲染：关掉 WebKit 合成器 + `LIBGL_ALWAYS_SOFTWARE`。
pub const ENV_SAFE_GL: &str = "HIFISHIFTER_SAFE_GL";
/// 置 0 可跳过 EGL 预检（探针本身出问题时的后备）。
pub const ENV_EGL_PREFLIGHT: &str = "HIFISHIFTER_EGL_PREFLIGHT";

const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
const EGL_VENDOR: u32 = 0x3053;
const EGL_VERSION: u32 = 0x3054;
const EGL_DEFAULT_DISPLAY: *mut c_void = ptr::null_mut();

type EglGetPlatformDisplay = unsafe extern "C" fn(u32, *mut c_void, *const isize) -> *mut c_void;
type EglGetDisplay = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type EglInitialize = unsafe extern "C" fn(*mut c_void, *mut i32, *mut i32) -> u32;
type EglQueryString = unsafe extern "C" fn(*mut c_void, u32) -> *const c_char;
type EglGetError = unsafe extern "C" fn() -> i32;

/// `true` / `1` / `yes` / `on` 视为开；空值与其他一律视为关。
fn truthy(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// `0` / `false` / `no` / `off` 视为显式关闭。
fn falsy(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("false") | Some("no") | Some("off")
    )
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// 会话摘要：全部字段在 `prepare()` 里一次性采样，便于测试与日志复现。
#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionFacts {
    gdk_backend: Option<String>,
    gdk_backend_source: &'static str,
    display: Option<String>,
    wayland_display: Option<String>,
    session_type: Option<String>,
    appimage: Option<String>,
    safe_gl: bool,
}

impl SessionFacts {
    fn describe(&self) -> String {
        let backend = match (&self.gdk_backend, self.gdk_backend_source) {
            (Some(value), source) => format!("{value} ({source})"),
            (None, source) => format!("<unset> ({source})"),
        };
        format!(
            "gdk_backend={backend} display={} wayland_display={} session_type={} appimage={} safe_gl={}",
            self.display.as_deref().unwrap_or("<unset>"),
            self.wayland_display.as_deref().unwrap_or("<unset>"),
            self.session_type.as_deref().unwrap_or("<unset>"),
            self.appimage.as_deref().unwrap_or("<none>"),
            if self.safe_gl { "on" } else { "off" },
        )
    }
}

/// 启动前调用一次（`main` 里、日志初始化之后、GTK 初始化之前）。
pub fn prepare() {
    let gdk_backend_source = apply_gdk_backend_override();
    let safe_gl = apply_safe_gl();

    let facts = SessionFacts {
        gdk_backend: env_nonempty("GDK_BACKEND"),
        gdk_backend_source,
        display: env_nonempty("DISPLAY"),
        wayland_display: env_nonempty("WAYLAND_DISPLAY"),
        session_type: env_nonempty("XDG_SESSION_TYPE"),
        appimage: env_nonempty("APPIMAGE").or_else(|| env_nonempty("APPDIR")),
        safe_gl,
    };
    log::info!("[linux] session: {}", facts.describe());

    if safe_gl {
        log::warn!(
            "[linux] {ENV_SAFE_GL} 已生效：WebKit 合成器关闭、GL 走软件渲染（界面可用，但绘制会变慢）"
        );
    }

    check_egl(safe_gl);
}

/// 处理 [`ENV_GDK_BACKEND`]，返回 `GDK_BACKEND` 的来源，供日志标注。
fn apply_gdk_backend_override() -> &'static str {
    let Some(raw) = env_nonempty(ENV_GDK_BACKEND) else {
        return if env_nonempty("GDK_BACKEND").is_some() {
            "from environment"
        } else {
            "unset"
        };
    };

    let normalized = raw.trim().to_ascii_lowercase();
    let value = match normalized.as_str() {
        "x11" => "x11",
        "wayland" => "wayland",
        other => {
            log::warn!(
                "[linux] {ENV_GDK_BACKEND}={other} 无法识别（只接受 x11 / wayland），已忽略"
            );
            return if env_nonempty("GDK_BACKEND").is_some() {
                "from environment"
            } else {
                "unset"
            };
        }
    };

    std::env::set_var("GDK_BACKEND", value);
    "override"
}

/// 处理 [`ENV_SAFE_GL`]，返回是否要求软件渲染。
fn apply_safe_gl() -> bool {
    if !truthy(std::env::var(ENV_SAFE_GL).ok().as_deref()) {
        return false;
    }
    std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
    std::env::set_var("LIBGL_ALWAYS_SOFTWARE", "1");
    true
}

struct EglReport {
    vendor: String,
    version: String,
}

fn check_egl(safe_gl: bool) {
    if safe_gl {
        // 已经按用户要求降级，再探一次没有意义。
        return;
    }
    if falsy(std::env::var(ENV_EGL_PREFLIGHT).ok().as_deref()) {
        log::info!("[linux] egl: 预检已通过 {ENV_EGL_PREFLIGHT}=0 关闭");
        return;
    }

    match probe_egl() {
        Ok(report) => {
            log::info!("[linux] egl: {} / {}", report.vendor, report.version);
        }
        Err(reason) => {
            if reason.lib_missing {
                // WebKit 拿不到 libEGL 必然 abort；这里先降级，让界面能出来。
                std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
                log::warn!(
                    "[linux] egl: {} —— 已自动降级为软件合成（WEBKIT_DISABLE_COMPOSITING_MODE=1）",
                    reason.message
                );
            } else {
                log::warn!(
                    "[linux] egl: {} —— 若界面起不来，请用 {ENV_SAFE_GL}=1 启动",
                    reason.message
                );
            }
        }
    }
}

struct EglFailure {
    message: String,
    /// 连 `libEGL.so.1` 都加载不了：WebKit 没有任何活路，必须降级。
    lib_missing: bool,
}

fn probe_egl() -> Result<EglReport, EglFailure> {
    // SAFETY：见 unsafe_probe_egl 的说明 —— 全部指针都由 libc 提供，失败即返回。
    unsafe { unsafe_probe_egl() }
}

/// 真正干活的部分：dlopen 一份 libEGL，建一个 surfaceless display 并初始化。
///
/// SAFETY：所有指针都来自 libc 的 dlopen/dlsym，函数指针只在非空时解引用；
/// 字符串在返回前复制成 `String`，不越过库的生命周期。
unsafe fn unsafe_probe_egl() -> Result<EglReport, EglFailure> {
    let lib_name = CString::new("libEGL.so.1").expect("static string");
    let handle = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
    if handle.is_null() {
        return Err(EglFailure {
            message: format!("libEGL.so.1 无法加载（{}）", last_dl_error()),
            lib_missing: true,
        });
    }

    let get_platform_display: Option<EglGetPlatformDisplay> =
        load_symbol(handle, "eglGetPlatformDisplay");
    let get_display: Option<EglGetDisplay> = load_symbol(handle, "eglGetDisplay");
    let initialize: EglInitialize = match load_symbol(handle, "eglInitialize") {
        Some(f) => f,
        None => {
            return Err(EglFailure {
                message: "libEGL.so.1 缺少 eglInitialize".into(),
                lib_missing: true,
            })
        }
    };
    let query_string: Option<EglQueryString> = load_symbol(handle, "eglQueryString");
    let get_error: Option<EglGetError> = load_symbol(handle, "eglGetError");

    let error_code = || -> String {
        match get_error {
            Some(f) => format!("0x{:04X}", f()),
            None => "unknown".into(),
        }
    };

    // 优先走 surfaceless：它不需要 X11/Wayland 连接，能单独回答"驱动这一层通不通"。
    let display = match get_platform_display {
        Some(f) => f(
            EGL_PLATFORM_SURFACELESS_MESA,
            EGL_DEFAULT_DISPLAY,
            ptr::null(),
        ),
        None => match get_display {
            Some(f) => f(EGL_DEFAULT_DISPLAY),
            None => {
                return Err(EglFailure {
                    message: "libEGL.so.1 既没有 eglGetPlatformDisplay 也没有 eglGetDisplay".into(),
                    lib_missing: true,
                })
            }
        },
    };
    if display.is_null() {
        return Err(EglFailure {
            message: format!("eglGetPlatformDisplay 返回空（错误码 {}）", error_code()),
            lib_missing: false,
        });
    }

    if initialize(display, ptr::null_mut(), ptr::null_mut()) == 0 {
        return Err(EglFailure {
            message: format!("eglInitialize 失败（错误码 {}）", error_code()),
            lib_missing: false,
        });
    }

    let read = |which: u32| -> String {
        match query_string {
            Some(f) => {
                let raw = f(display, which);
                if raw.is_null() {
                    "<unknown>".into()
                } else {
                    CStr::from_ptr(raw).to_string_lossy().into_owned()
                }
            }
            None => "<unknown>".into(),
        }
    };

    Ok(EglReport {
        vendor: read(EGL_VENDOR),
        version: read(EGL_VERSION),
    })
}

/// 从已加载的库里取一个函数指针。
///
/// SAFETY：调用方保证 `handle` 来自 `dlopen`；`transmute_copy` 的前提是目标类型与
/// `*mut c_void` 同为指针宽度（下面用到的 EGL 入口都是）。
unsafe fn load_symbol<T>(handle: *mut c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let symbol = libc::dlsym(handle, cname.as_ptr());
    if symbol.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy::<*mut c_void, T>(&symbol))
    }
}

fn last_dl_error() -> String {
    unsafe {
        let raw = libc::dlerror();
        if raw.is_null() {
            "未知原因".into()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_accepts_the_documented_spellings() {
        for raw in ["1", "true", "TRUE", "yes", "on", " 1 "] {
            assert!(truthy(Some(raw)), "{raw} 应当为真");
        }
        for raw in ["", "0", "false", "no", "off", "auto"] {
            assert!(!truthy(Some(raw)), "{raw} 应当为假");
        }
        assert!(!truthy(None));
    }

    #[test]
    fn falsy_only_matches_explicit_off() {
        for raw in ["0", "false", "OFF", "no"] {
            assert!(falsy(Some(raw)), "{raw} 应当被识别为显式关闭");
        }
        for raw in ["", "1", "true"] {
            assert!(!falsy(Some(raw)));
        }
        assert!(!falsy(None));
    }

    #[test]
    fn session_facts_are_readable_when_everything_is_unset() {
        let facts = SessionFacts {
            gdk_backend: None,
            gdk_backend_source: "unset",
            display: None,
            wayland_display: None,
            session_type: None,
            appimage: None,
            safe_gl: false,
        };
        let line = facts.describe();
        assert!(line.contains("gdk_backend=<unset> (unset)"));
        assert!(line.contains("display=<unset>"));
        assert!(line.contains("appimage=<none>"));
        assert!(line.contains("safe_gl=off"));
    }

    #[test]
    fn session_facts_report_override_and_appimage() {
        let facts = SessionFacts {
            gdk_backend: Some("wayland".into()),
            gdk_backend_source: "override",
            display: Some(":0".into()),
            wayland_display: Some("wayland-0".into()),
            session_type: Some("wayland".into()),
            appimage: Some("/tmp/x.AppImage".into()),
            safe_gl: true,
        };
        let line = facts.describe();
        assert!(line.contains("gdk_backend=wayland (override)"));
        assert!(line.contains("wayland_display=wayland-0"));
        assert!(line.contains("safe_gl=on"));
    }
}
