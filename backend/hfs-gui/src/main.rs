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
    NavigationPolicyDecisionExt, PolicyDecisionExt, SecurityManagerExt, SettingsExt, URIRequestExt,
    URISchemeRequestExt, UserContentManagerExt, WebContextExt, WebViewExt,
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

/// 与插件侧 `editor/gui/protocol.rs::resolve_asset` 同一份契约的渲染进程实现。
///
/// 【为什么是拷贝而不是依赖】hfs-gui 是独立工作区的子进程，不依赖插件 crate；
/// bridge.js 走 `include_str!` 共享，文本契约没法这样共享 —— 这里逐条对齐，
/// 任何一侧改动判据都必须同步另一侧（对拍由插件侧 `protocol` 的测试与本
/// 函数的一致性评审共同守卫）。
fn resolve_asset(root: &std::path::Path, request_path: &str) -> Option<std::path::PathBuf> {
    let relative = request_path.trim_start_matches('/');
    if relative.is_empty() || relative.contains('\\') || relative.contains('\0') {
        return None;
    }
    let mut path = root.to_path_buf();
    for segment in relative.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return None;
        }
        path.push(segment);
    }
    let canonical_root = root.canonicalize().ok()?;
    let canonical = path.canonicalize().ok()?;
    if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
        return None;
    }
    Some(canonical)
}

/// 把最终生效的加速策略写进日志：A/B 时"到底用的哪一个"不能靠记忆。
fn log_policy(policy: webkit2gtk::HardwareAccelerationPolicy) {
    let name = match policy {
        webkit2gtk::HardwareAccelerationPolicy::Never => "never",
        webkit2gtk::HardwareAccelerationPolicy::Always => "always",
        _ => "on-demand",
    };
    send(serde_json::json!({
        "type": "note",
        "text": format!("hardware acceleration policy: {name}")
    }));
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
            // `hfs-plugin://local/<path>` → 文件系统路径。判据与插件侧的
            // `editor/gui/protocol.rs::resolve_asset` 是**同一份契约**（bridge.js 已用
            // include_str! 共享，同理这里逐条对齐）：拒绝绝对路径、反斜杠、NUL、空段、
            // `.`/`..` 段，canonicalize 后必须仍在根内且是文件。
            //
            // 【为什么不能只拦 `..`】`Path::join` 对以 `/` 开头的 path 会**整体替换根**
            // —— `hfs-plugin://local//etc/passwd` 原样放行；导航白名单拦不住 scheme
            // 内部的子资源请求。页面可被注入脚本改写是插件侧写明的威胁模型，
            // 渲染进程是它的延伸（第 5 轮审查 P2-7）。
            let path = uri
                .split("://")
                .nth(1)
                .and_then(|rest| rest.split_once('/').map(|(_, path)| path.to_string()))
                .unwrap_or_default();
            let path = path.split(['?', '#']).next().unwrap_or("");
            // `root` 是命令行给的字符串；`resolve_asset` 要的是 `&Path`。
            let resolved = resolve_asset(std::path::Path::new(&root), path);
            match resolved {
                Some(full) => match std::fs::read(&full) {
                    Ok(bytes) => {
                        let mime = mime_for(path);
                        let stream = gtk::gio::MemoryInputStream::from_bytes(
                            &gtk::glib::Bytes::from(&bytes),
                        );
                        request.finish(&stream, bytes.len() as i64, Some(mime));
                    }
                    Err(error) => request.finish_error(&mut gtk::glib::Error::new(
                        gtk::gio::IOErrorEnum::NotFound,
                        &format!("{error}: {}", full.display()),
                    )),
                },
                None => request.finish_error(&mut gtk::glib::Error::new(
                    gtk::gio::IOErrorEnum::InvalidArgument,
                    "invalid path",
                )),
            }
        });
    }

    let webview = webkit2gtk::WebView::with_context(&context);
    if let Some(settings) = WebViewExt::settings(&webview) {
        // 【为什么是 OnDemand 而不是 Never】时间轴与参数编辑器各有一块 WebGL2 画布，
        // 而标尺、Clip 体、网格是 DOM。`Never` 强制软件光栅：DOM 层由 rAF 立刻平移，
        // 画布却要等一次软件重绘才跟上，于是水平滚动时标尺与网格/Clip/参数线肉眼可见
        // 地分离、不同步。
        //
        // 【为什么 OnDemand 仍然守住了原来的顾虑】它是 WebKitGTK 的默认策略：能用 GPU
        // 就用，不能就自动退回软件路径。原来那条 `Never` 想避免的是「假设 GL 可用」，
        // 而 `OnDemand` 从不做这个假设 —— 它只是不在 GL 可用时白白放弃它。
        // 【为什么留一个环境变量开关】OnDemand 修好了水平滚动时各层分离（软件光栅下
        // DOM 与 WebGL2 画布不在同一个合成事务里）。但真机上又出现了"标签页切回来只
        // 剩部分图层"的现象，而它既可能是画布 backing store 被丢弃，也可能是嵌入 X
        // 窗口里的 GL 合成掉了。这两条修法完全不同，靠猜只会再来一轮。
        //
        // `HFS_GUI_ACCEL=never` 让现场直接 A/B：
        //   · 设上之后问题消失 ⇒ 是 GL 合成的事，未走的路线是"让 OnDemand 与重绘共存"；
        //   · 设上之后问题照旧 ⇒ 是各画布需要自己重绘，与合成无关。
        let policy = match std::env::var("HFS_GUI_ACCEL").ok().as_deref() {
            Some("never") => webkit2gtk::HardwareAccelerationPolicy::Never,
            _ => webkit2gtk::HardwareAccelerationPolicy::OnDemand,
        };
        log_policy(policy);
        settings.set_hardware_acceleration_policy(policy);
    }
    let Some(manager) = webview.user_content_manager() else {
        send(serde_json::json!({"type":"error","text":"no user content manager"}));
        std::process::exit(1);
    };
    // 【为什么 bootstrap 要序列化成字面量再 JSON.parse】
    //
    // 直接把 bootstrap 文本拼进 JS 源码，等于把「载荷」和「代码」放进同一个语法
    // 单元：载荷里任何一个字符让整条脚本语法失败，注入的就是**空**，而 UserScript
    // 的语法错误不报错、不抛异常、不留痕迹，症状只有「页面看不到这个全局」——
    // 前端 `getPluginHost()` 随即静默返回 null，整个应用退化成 standalone。
    // 经 serde 序列化成一个字符串字面量，载荷就只是数据，不可能破坏脚本结构。
    let bootstrap_literal =
        serde_json::to_string(&bootstrap).unwrap_or_else(|_| "\"{}\"".to_owned());
    // 【为什么合成一条脚本】bootstrap 是前端的入口条件（`getPluginHost()` 看不到
    // 它就返回 null），它必须与桥**同生共死**。分两条 `add_script` 时，第二条在真机上
    // 没有生效过（桥的探针每次都到，单独的 bootstrap 探针一次都没到）；与其继续猜
    // WebKit 为什么不执行第二条，不如把它并进那条已被证明会执行的那一条。
    manager.add_script(&webkit2gtk::UserScript::new(
        &format!(
            "{BRIDGE}\n\
             try{{window.__HFS_PLUGIN_BOOTSTRAP__=JSON.parse({bootstrap_literal});}}\
             catch(e){{window.__HFS_PLUGIN_BOOTSTRAP__={{}};}}\n"
        ),
        webkit2gtk::UserContentInjectedFrames::TopFrame,
        webkit2gtk::UserScriptInjectionTime::Start,
        &[],
        &[],
    ));
    // 【为什么失败必须报出来】注册失败时 `window.webkit.messageHandlers.hfs` 是
    // undefined，页面里 `postMessage` 抛 TypeError 并被前端自己的 try/catch 吞掉 ——
    // 表现是「界面能显示，点什么都没反应」，而这是**唯一**的开关。成功时不说话：
    // 一次会话一行噪声换不来任何信息。
    if !manager.register_script_message_handler("hfs") {
        send(serde_json::json!({
            "type": "error",
            "text": "the webview refused the 'hfs' script message handler; \
                     the editor cannot send any command"
        }));
    } else {
        manager.connect_script_message_received(None, move |_, result| {
            // 只接受字符串：结构化值在这里没有意义，而字符串是三条链路上唯一逐字节
            // 一致的形式（见插件里 protocol.rs 的理由）。
            let Some(value) = result.js_value() else {
                return;
            };
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
        // 【为什么加载完成后再补一次 bootstrap】
        //
        // UserScript 的注入世界与页面自己的执行上下文未必相同 —— 真机上出现过
        // `window.__HFS_PLUGIN_BOOTSTRAP__` 在页面里不存在、而同一通道的桥探针却到达
        // 的情形。前端的 bootstrap 是**按需读取**的（`getPluginHost()` 每次调用现读
        // 全局），所以晚到一拍也能被看到；而加载完成是这条链上还握在我们手里的时机。
        // 与 UserScript 那条用同一个字面量，两处不可能给出不同的载荷。
        let boot_webview = webview.clone();
        let boot_script = format!(
            "window.__HFS_PLUGIN_BOOTSTRAP__=JSON.parse({bootstrap_literal});"
        );
        webview.connect_load_changed(move |_, event| {
            if event == webkit2gtk::LoadEvent::Finished {
                boot_webview.evaluate_javascript(
                    &boot_script,
                    None,
                    None,
                    None::<&gtk::gio::Cancellable>,
                    |_result| {},
                );
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
