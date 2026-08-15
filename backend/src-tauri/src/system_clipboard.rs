//! Native system clipboard transport for HiFiShifter object data.
//!
//! WebView clipboard APIs are permission-gated and unreliable on some
//! platforms, so all structured copy/paste data is written through this
//! module.  The payload is stored twice:
//!
//! 1. As a platform-native custom format (where available).
//! 2. As a base64 text envelope (`HIFISHIFTER_CLIPBOARD_V1:<base64>`).
//!
//! Another HiFiShifter process prefers the custom format and falls back to
//! the text envelope, which makes cross-process copy/paste work on all
//! supported desktop platforms.

use base64::Engine;

pub const OBJECT_FORMAT: &str = "application/x-hifishifter-object";
const TEXT_PREFIX: &str = "HIFISHIFTER_CLIPBOARD_V1:";

fn encode_text_envelope(bytes: &[u8]) -> String {
    format!(
        "{}{}",
        TEXT_PREFIX,
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn decode_text_envelope(text: &str) -> Option<Vec<u8>> {
    let body = text.strip_prefix(TEXT_PREFIX)?;
    base64::engine::general_purpose::STANDARD.decode(body).ok()
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
pub fn write_bytes(bytes: &[u8]) -> Result<(), String> {
    use clipboard_win::{raw, register_format, Clipboard};

    let _clipboard =
        Clipboard::new_attempts(10).map_err(|e| format!("clipboard_open_failed: {}", e))?;

    raw::empty().map_err(|e| format!("clipboard_empty_failed: {}", e))?;

    if let Some(format) = register_format(OBJECT_FORMAT) {
        raw::set_without_clear(format.get(), bytes)
            .map_err(|e| format!("clipboard_write_custom_failed: {}", e))?;
    }

    let text = encode_text_envelope(bytes);
    raw::set_string_with(&text, clipboard_win::options::NoClear)
        .map_err(|e| format!("clipboard_write_text_failed: {}", e))?;

    Ok(())
}

#[cfg(target_os = "windows")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use clipboard_win::{raw, register_format, Clipboard};

    let _clipboard =
        Clipboard::new_attempts(10).map_err(|e| format!("clipboard_open_failed: {}", e))?;

    if let Some(format) = register_format(OBJECT_FORMAT) {
        if raw::is_format_avail(format.get()) {
            let size = raw::size(format.get()).ok_or_else(|| "clipboard_empty".to_string())?;
            let mut buf = vec![0u8; size.get()];
            let bytes_read = raw::get(format.get(), &mut buf)
                .map_err(|e| format!("clipboard_read_failed: {}", e))?;
            buf.truncate(bytes_read);
            if !buf.is_empty() {
                return Ok(Some(buf));
            }
        }
    }

    use clipboard_win::Getter;
    let mut text = String::new();
    match clipboard_win::formats::Unicode.read_clipboard(&mut text) {
        Ok(_) => Ok(decode_text_envelope(text.trim())),
        Err(_) => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// macOS
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub fn write_bytes(bytes: &[u8]) -> Result<(), String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::{NSData, NSString};

    let pasteboard = NSPasteboard::generalPasteboard();
    let _ = pasteboard.clearContents();

    let text = encode_text_envelope(bytes);
    let text_ns = NSString::from_str(&text);
    let text_type = NSString::from_str("public.utf8-plain-text");
    if !pasteboard.setString_forType(&text_ns, &text_type) {
        return Err("clipboard_write_text_failed".to_string());
    }

    let data = NSData::with_bytes(bytes);
    let format_ns = NSString::from_str(OBJECT_FORMAT);
    let _ = pasteboard.setData_forType(Some(&data), &format_ns);
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::NSString;

    let pasteboard = NSPasteboard::generalPasteboard();

    let format_ns = NSString::from_str(OBJECT_FORMAT);
    if let Some(data) = pasteboard.dataForType(&format_ns) {
        let bytes = data.to_vec();
        if !bytes.is_empty() {
            return Ok(Some(bytes));
        }
    }

    let text_type = NSString::from_str("public.utf8-plain-text");
    if let Some(text) = pasteboard.stringForType(&text_type) {
        return Ok(decode_text_envelope(text.to_string().trim()));
    }
    Ok(None)
}

#[cfg(target_os = "linux")]
fn is_wayland_session() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
        || std::env::var("XDG_SESSION_TYPE")
            .map(|v| v.eq_ignore_ascii_case("wayland"))
            .unwrap_or(false)
}

#[cfg(target_os = "linux")]
pub fn write_bytes(bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let text = encode_text_envelope(bytes);
    let is_wayland = is_wayland_session();

    let mut child = if is_wayland {
        Command::new("wl-copy")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    } else {
        Command::new("xclip")
            .args(["-selection", "clipboard", "-i"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    }
    .map_err(|e| {
        let tool = if is_wayland { "wl-copy" } else { "xclip" };
        format!("clipboard_write_failed: failed to run {}: {}", tool, e)
    })?;

    child
        .stdin
        .take()
        .ok_or_else(|| "clipboard_write_failed: no stdin".to_string())?
        .write_all(text.as_bytes())
        .map_err(|e| format!("clipboard_write_failed: {}", e))?;

    let status = child
        .wait()
        .map_err(|e| format!("clipboard_write_failed: {}", e))?;
    if status.success() {
        Ok(())
    } else {
        Err("clipboard_write_failed".to_string())
    }
}

#[cfg(target_os = "linux")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use std::process::Command;

    let is_wayland = is_wayland_session();
    let output = if is_wayland {
        Command::new("wl-paste").args(["--no-newline"]).output()
    } else {
        Command::new("xclip")
            .args(["-selection", "clipboard", "-o"])
            .output()
    };

    let output = output.map_err(|e| {
        let tool = if is_wayland { "wl-paste" } else { "xclip" };
        format!("clipboard_read_failed: failed to run {}: {}", tool, e)
    })?;

    if !output.status.success() {
        return Ok(None);
    }
    Ok(decode_text_envelope(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

// ---------------------------------------------------------------------------
// Unsupported platform
// ---------------------------------------------------------------------------

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn write_bytes(_bytes: &[u8]) -> Result<(), String> {
    Err("clipboard_unsupported_platform".to_string())
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    Err("clipboard_unsupported_platform".to_string())
}
