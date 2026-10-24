//! `hfs-gui` —— 编辑器渲染进程（原型）。
//!
//! 【为什么需要它】实测：**只要在 REAPER 的进程里创建一个 GTK 窗口**，那个进程的
//! 绘制就整个废掉 —— 判据是 `xwd -id <REAPER 主窗口>` 从 3.1 MB 变成 0 字节，连
//! REAPER 自己的对话框都变黑。窗口里只放一个 `gtk::Label`（完全没有 WebKit）也复现。
//!
//! 所以 GTK 必须活在另一个进程里，只把它的 X 窗口嵌进宿主给的 `X11EmbedWindowID`。
//!
//! 用法：`hfs-gui --parent <xid> --width <w> --height <h>`
//!
//! 本原型只建窗口 + 嵌入，用来验证"宿主是否保持健康"；页面与 IPC 随后接上。

use gtk::glib::translate::ToGlibPtr;
use gtk::prelude::*;

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

fn main() {
    let mut args = std::env::args().skip(1);
    let mut parent: u64 = 0;
    let mut width: i32 = 900;
    let mut height: i32 = 600;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--parent" => {
                // 宿主给的是十六进制（§0x60002c§），§u64::from_str§ 不认 §0x§ 前缀。
                parent = args
                    .next()
                    .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                    .unwrap_or(0)
            }
            "--width" => width = args.next().and_then(|v| v.parse().ok()).unwrap_or(900),
            "--height" => height = args.next().and_then(|v| v.parse().ok()).unwrap_or(600),
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
        eprintln!("gtk_init failed");
        std::process::exit(1);
    }
    let window = gtk::Window::new(gtk::WindowType::Toplevel);
    window.set_decorated(false);
    window.set_default_size(width, height);
    window.add(&gtk::Label::new(Some("HiFiShifter helper")));
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
    println!("ready xid=0x{xid:x} parent=0x{parent:x}");
    gtk::main();
}
