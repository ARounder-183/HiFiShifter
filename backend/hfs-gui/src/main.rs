//! `hfs-gui` —— 编辑器渲染进程。
//!
//! 【为什么需要它】实测：**只要在 REAPER 的进程里创建一个 GTK 窗口**，那个进程的
//! 绘制就整个废掉 —— 判据是 `xwd -id <REAPER 主窗口>` 从 3.1 MB 变成 0 字节，连
//! REAPER 自己的对话框都变黑。窗口里只放一个 `gtk::Label`（完全没有 WebKit）也复现。
//!
//! 所以 GTK 与 WebKit 活在**这个进程**里，只把自己的 X 窗口嵌进宿主给的
//! `X11EmbedWindowID`；宿主进程里一个 GTK 符号都不碰。
//!
//! 与宿主走 stdin/stdout，一行一个 JSON：
//!
//! | 方向 | 消息 |
//! |---|---|
//! | 宿主 → 本进程 | `eval` / `resize` / `focus` / `close` |
//! | 本进程 → 宿主 | `ready` / `message` / `error` |
//!
//! 【为什么用管道而不是 socket】宿主本来就要 spawn 子进程；管道天然带生命周期
//! （父进程死了管道就断），不需要额外清理逻辑，也不会在 `/tmp` 里留垃圾。

use std::io::{BufRead, Write};

use gtk::glib::translate::ToGlibPtr;
use gtk::prelude::*;
use javascriptcore::ValueExt;
use webkit2gtk::{
    NavigationPolicyDecisionExt, PolicyDecisionExt, SecurityManagerExt, SettingsExt,
    URIRequestExt, URISchemeRequestExt, UserContentManagerExt, WebContextExt, WebViewExt,
};

/// 与插件里同一份桥接脚本（WebKitGTK 上它负责提供 `window.chrome.webview`）。
const BRIDGE: &str = include_str!("../../hifishifter-plugin/src/editor/gui/bridge.js");

mod x11 {
    use std::ffi::c_void;

    pub type XWindow = libc::c_ulong;

    #[link(name = "X11")]
    extern "C" {
        pub fn gdk_x11_get_default_xdisplay() -> *mut c_void;
        pub fn gdk_x11_window_get_xid(window: *mut c_void) -> XWindow;
        pub fn gdk_window_set_override_redirect(window: *mut c_void, override_redirect: i32);
        pub fn XReparentWindow(
            display: *mut c_void,
            window: XWindow,
            parent: XWindow,
            x: i32,
            y: i32,
        ) -> i32;
        pub fn XMoveResizeWindow(
            display: *mut c_void,
            window: XWindow,
            x: i32,
            y: i32,
            width: u32,
            height: u32,
        ) -> i32;
        pub fn XSync(display: *mut c_void, discard: i32) -> i32;
    }
}

fn send(value: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

fn mime_for(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html",
        "js" | "mjs" => "text/javascript",
        "css" => "text/css",
        "json" => "application/json",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "map" => "application/json",
        "ico" => "image/x-icon",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut parent: u64 = 0;
    let mut width: i32 = 900;
    let mut height: i32 = 600;
    let mut root = String::new();
    let mut bootstrap = String::from("{}");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            // 宿主给的是十六进制（`0x60002c`），`u64::from_str` 不认 `0x` 前缀。
            "--parent" => {
                parent = args
                    .next()
                    .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                    .unwrap_or(0)
            }
            "--width" => width = args.next().and_then(|v| v.parse().ok()).unwrap_or(900),
            "--height" => height = args.next().and_then(|v| v.parse().ok()).unwrap_or(600),
            "--root" => root = args.next().unwrap_or_default(),
            "--bootstrap-file" => {
                if let Some(path) = args.next() {
                    bootstrap = std::fs::read_to_string(path).unwrap_or_else(|_| "{}".into());
                }
            }
            _ => {}
        }
    }

    // 【必须先钉后端】只要会话里存在 `WAYLAND_DISPLAY`，GDK 就默认选 Wayland ——
    // 那时我们的 GdkWindow 不是 X11 窗口，既拿不到 XID 也无法被 `XReparentWindow`
    // 搬进宿主的窗口（`gdk_x11_window_get_xid` 会直接断言失败）。
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        std::env::set_var("GDK_BACKEND", "x11");
    }
    if gtk::init().is_err() {
        send(serde_json::json!({"type":"error","text":"gtk_init failed"}));
        std::process::exit(1);
    }

    let context = webkit2gtk::WebContext::new_ephemeral();
    if let Some(security) = context.security_manager() {
        security.register_uri_scheme_as_secure("hfs-plugin");
        security.register_uri_scheme_as_cors_enabled("hfs-plugin");
    }
    {
        let root = root.clone();
        context.register_uri_scheme("hfs-plugin", move |request| {
            let uri = request.uri().unwrap_or_default();
            // `hfs-plugin://local/<path>` → 文件系统路径；`..` 一律拒绝。
            let path = uri
                .split("://")
                .nth(1)
                .and_then(|rest| rest.split_once('/').map(|(_, path)| path.to_string()))
                .unwrap_or_default();
            let path = path.split(['?', '#']).next().unwrap_or("");
            if path.contains("..") {
                request.finish_error(&mut gtk::glib::Error::new(
                    gtk::gio::IOErrorEnum::InvalidArgument,
                    "invalid path",
                ));
                return;
            }
            let full = std::path::Path::new(&root).join(path);
            match std::fs::read(&full) {
                Ok(bytes) => {
                    let mime = mime_for(path);
                    let stream =
                        gtk::gio::MemoryInputStream::from_bytes(&gtk::glib::Bytes::from(&bytes));
                    request.finish(&stream, bytes.len() as i64, Some(mime));
                }
                Err(error) => request.finish_error(&mut gtk::glib::Error::new(
                    gtk::gio::IOErrorEnum::NotFound,
                    &format!("{error}: {}", full.display()),
                )),
            }
        });
    }

    let webview = webkit2gtk::WebView::with_context(&context);
    if let Some(settings) = WebViewExt::settings(&webview) {
        // 与插件里同样的理由：宿主/环境的 GL 不可假设，CPU 合成最稳。
        settings.set_hardware_acceleration_policy(webkit2gtk::HardwareAccelerationPolicy::Never);
    }
    let Some(manager) = webview.user_content_manager() else {
        send(serde_json::json!({"type":"error","text":"no user content manager"}));
        std::process::exit(1);
    };
    manager.add_script(&webkit2gtk::UserScript::new(
        BRIDGE,
        webkit2gtk::UserContentInjectedFrames::TopFrame,
        webkit2gtk::UserScriptInjectionTime::Start,
        &[],
        &[],
    ));
    manager.add_script(&webkit2gtk::UserScript::new(
        &format!("window.__HFS_PLUGIN_BOOTSTRAP__={bootstrap};"),
        webkit2gtk::UserContentInjectedFrames::TopFrame,
        webkit2gtk::UserScriptInjectionTime::Start,
        &[],
        &[],
    ));
    if manager.register_script_message_handler("hfs") {
        manager.connect_script_message_received(None, move |_, result| {
            // 只接受字符串：结构化值在这里没有意义，而字符串是三条链路上唯一逐字节
            // 一致的形式（见插件里 protocol.rs 的理由）。
            let Some(value) = result.js_value() else { return };
            send(serde_json::json!({"type":"message","text": value.to_str().to_string()}));
        });
    }
    {
        let webview = webview.clone();
        webview.connect_decide_policy(|_, decision, kind| {
            if kind == webkit2gtk::PolicyDecisionType::NavigationAction {
                if let Some(navigation) =
                    decision.downcast_ref::<webkit2gtk::NavigationPolicyDecision>()
                {
                    let allowed = navigation
                        .navigation_action()
                        .and_then(|action| action.request())
                        .and_then(|request| request.uri())
                        .is_some_and(|uri| uri.starts_with("hfs-plugin://local/"));
                    if !allowed {
                        decision.ignore();
                        return true;
                    }
                }
            }
            false
        });
        webview.connect_create(|_, _| None::<gtk::Widget>);
        webview.connect_load_changed(move |_, event| {
            if event == webkit2gtk::LoadEvent::Finished {
                send(serde_json::json!({"type":"page-ready"}));
            }
        });
        webview.connect_web_process_terminated(|_, reason| {
            send(serde_json::json!({"type":"error","text":format!("web process terminated: {reason:?}")}));
        });
    }

    let window = gtk::Window::new(gtk::WindowType::Toplevel);
    window.set_decorated(false);
    window.set_default_size(width, height);
    window.add(&webview);
    window.realize();

    let mut xid = 0u64;
    if let Some(gdk_window) = window.window() {
        let raw: *mut gtk::gdk::ffi::GdkWindow = gdk_window.to_glib_none().0;
        unsafe {
            x11::gdk_window_set_override_redirect(raw.cast(), 1);
            xid = x11::gdk_x11_window_get_xid(raw.cast());
        }
    }
    if parent != 0 && xid != 0 {
        unsafe {
            let dpy = x11::gdk_x11_get_default_xdisplay();
            x11::XReparentWindow(dpy, xid, parent as x11::XWindow, 0, 0);
            x11::XMoveResizeWindow(dpy, xid, 0, 0, width as u32, height as u32);
            x11::XSync(dpy, 0);
        }
    }
    window.show_all();
    send(serde_json::json!({"type":"ready","xid": xid}));
    webview.load_uri("hfs-plugin://local/plugin.html");

    // 宿主 → 本进程：stdin 线程只负责**解析**，真正的 GTK 调用必须回到主线程
    // （Gtk 对象不是 `Send`，把控件搬进另一个线程的闭包会直接编译不过 —— 这也是
    // 为什么这里走通道而不是直接 `idle_add_local_once`）。
    let (commands, incoming) = std::sync::mpsc::channel::<serde_json::Value>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if commands.send(value).is_err() {
                break;
            }
        }
    });
    gtk::glib::timeout_add_local(std::time::Duration::from_millis(16), move || {
        while let Ok(value) = incoming.try_recv() {
            match value["type"].as_str().unwrap_or_default() {
                "eval" => {
                    if let Some(js) = value["js"].as_str() {
                        webview.evaluate_javascript(
                            js,
                            None,
                            None,
                            None::<&gtk::gio::Cancellable>,
                            |_| {},
                        );
                    }
                }
                "resize" => {
                    let w = value["width"].as_i64().unwrap_or(0) as i32;
                    let h = value["height"].as_i64().unwrap_or(0) as i32;
                    if w > 0 && h > 0 {
                        window.resize(w, h);
                        if xid != 0 {
                            unsafe {
                                let dpy = x11::gdk_x11_get_default_xdisplay();
                                x11::XMoveResizeWindow(dpy, xid, 0, 0, w as u32, h as u32);
                                x11::XSync(dpy, 0);
                            }
                        }
                    }
                }
                "focus" => {
                    window.grab_focus();
                }
                "close" => {
                    gtk::main_quit();
                    return gtk::glib::ControlFlow::Break;
                }
                _ => {}
            }
        }
        gtk::glib::ControlFlow::Continue
    });

    gtk::main();
}
