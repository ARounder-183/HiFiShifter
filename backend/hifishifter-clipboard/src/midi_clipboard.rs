//! 系统剪贴板里的 "Standard MIDI File"（REAPER 复制 MIDI item 时写入的格式）。
//!
//! 【为什么在共享剪贴板 crate 里】它和本 crate 的其它格式是同一件事：打开系统剪贴板、
//! 按平台私有格式名取一段字节、失败时给出可判定的错误。App 与 ARA 插件都要读它
//! （插件里"粘贴 MIDI"必须和独立 App 行为一致），格式名与平台分支因此只能有一份。
//!
//! 平台格式名：
//! - Windows：注册格式 `Standard MIDI File`；
//! - macOS：NSPasteboard 类型 `Standard MIDI File`；
//! - Linux：REAPER 用 `application/swell-Standard MIDI File`，旧版短名作回退。

/// Windows 注册格式名；macOS 的 pasteboard 类型名与之相同。
pub const STANDARD_MIDI_FILE_FORMAT: &str = "Standard MIDI File";

/// 读取系统剪贴板中的 "Standard MIDI File" 字节。
///
/// 返回 `Err("midi_clipboard_empty")` 表示剪贴板里没有这个格式（不是失败）；
/// 调用方通常把它呈现为"剪贴板里没有 MIDI"。
#[cfg(target_os = "windows")]
pub fn read_standard_midi_file() -> Result<Vec<u8>, String> {
    use clipboard_win::{raw, register_format};

    let format = register_format(STANDARD_MIDI_FILE_FORMAT)
        .ok_or_else(|| "midi_clipboard_format_not_found".to_string())?;

    crate::system_clipboard::clipboard_session(|_clip| {
        let size = raw::size(format.get()).ok_or_else(|| "midi_clipboard_empty".to_string())?;

        let mut buf = vec![0u8; size.get()];
        let bytes_read = raw::get(format.get(), &mut buf)
            .map_err(|e| format!("midi_clipboard_read_failed: {}", e))?;

        buf.truncate(bytes_read);
        Ok(buf)
    })
}

/// macOS：通过 NSPasteboard 读取自定义类型 "Standard MIDI File"。
#[cfg(target_os = "macos")]
pub fn read_standard_midi_file() -> Result<Vec<u8>, String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::NSString;

    let pasteboard = unsafe { NSPasteboard::generalPasteboard() };
    let pb_type = NSString::from_str(STANDARD_MIDI_FILE_FORMAT);

    let data = unsafe { pasteboard.dataForType(&pb_type) }
        .ok_or_else(|| "midi_clipboard_empty".to_string())?;

    let len = data.length();
    if len == 0 {
        return Err("midi_clipboard_empty".to_string());
    }
    use objc2::msg_send;
    use std::ffi::c_void;
    let raw_ptr: *const c_void = unsafe { msg_send![&*data, bytes] };
    let ptr = raw_ptr as *const u8;
    // Objective-C 消息可能返回 nil（如数据被外部释放），
    // 对 null 指针调用 from_raw_parts 是未定义行为，必须先判空。
    if ptr.is_null() {
        return Err("clipboard_data_unavailable".to_string());
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bytes.to_vec())
}

/// Linux：通过 wl-paste (Wayland) 或 xclip (X11) 读取 Standard MIDI File。
/// 优先读取 REAPER Linux 使用的 `application/swell-Standard MIDI File`，
/// 同时保留旧版短目标 `Standard MIDI File` 作为回退。
#[cfg(target_os = "linux")]
pub fn read_standard_midi_file() -> Result<Vec<u8>, String> {
    use std::process::Command;

    let is_wayland = crate::linux_clipboard::is_wayland_session();
    let targets = [
        crate::linux_clipboard::STANDARD_MIDI_FILE_LINUX_FORMAT,
        crate::linux_clipboard::STANDARD_MIDI_FILE_LEGACY_FORMAT,
    ];

    for target in targets {
        let output = if is_wayland {
            Command::new("wl-paste").args(["--type", target]).output()
        } else {
            Command::new("xclip")
                .args(["-selection", "clipboard", "-target", target, "-o"])
                .output()
        };

        match output {
            Err(error) => {
                let tool = if is_wayland { "wl-paste" } else { "xclip" };
                return Err(format!(
                    "midi_clipboard_read_failed: failed to run {}: {}",
                    tool, error
                ));
            }
            Ok(output) if output.status.success() && !output.stdout.is_empty() => {
                return Ok(output.stdout);
            }
            Ok(_) => continue,
        }
    }

    Err("midi_clipboard_empty".to_string())
}

/// 不支持的平台回退。
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn read_standard_midi_file() -> Result<Vec<u8>, String> {
    Err("clipboard_unsupported_platform".to_string())
}
