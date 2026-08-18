//! Native system clipboard transport for HiFiShifter object data.
//!
//! WebView clipboard APIs are permission-gated and unreliable on some
//! platforms, so all structured copy/paste data is written through this
//! module.
//!
//! The binary payload is stored in a platform-native private clipboard
//! format.  The normal `text/plain` slot is deliberately *not* used for the
//! binary payload: instead it receives a short human-readable summary such
//! as "HiFiShifter: 3 clips copied".  Pasting into a regular text box
//! therefore never produces base64 garbage.
//!
//! For backwards compatibility, readers still understand the old
//! `HIFISHIFTER_CLIPBOARD_V1:<base64>` text envelope.

use base64::Engine;

pub const OBJECT_FORMAT: &str = "application/x-hifishifter-object";
#[cfg(not(target_os = "linux"))]
pub const REAPER_MEDIA_FORMAT: &str = "REAPERMedia";
const TEXT_PREFIX: &str = "HIFISHIFTER_CLIPBOARD_V1:";

fn decode_text_envelope(text: &str) -> Option<Vec<u8>> {
    let body = text.strip_prefix(TEXT_PREFIX)?;
    base64::engine::general_purpose::STANDARD.decode(body).ok()
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn write_contents(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    use clipboard_win::{raw, register_format, Clipboard};

    let _clipboard =
        Clipboard::new_attempts(10).map_err(|e| format!("clipboard_open_failed: {}", e))?;

    // Both HiFiShifter and REAPER formats must share one clipboard ownership
    // session: `empty()` once, then set each format with `NoClear` semantics.
    raw::empty().map_err(|e| format!("clipboard_empty_failed: {}", e))?;

    if let Some(format) = register_format(OBJECT_FORMAT) {
        raw::set_without_clear(format.get(), bytes)
            .map_err(|e| format!("clipboard_write_custom_failed: {}", e))?;
    }

    if let Some(reaper_bytes) = reaper_bytes {
        let format = register_format(REAPER_MEDIA_FORMAT)
            .ok_or_else(|| "reaper_clipboard_format_not_found".to_string())?;
        raw::set_without_clear(format.get(), reaper_bytes)
            .map_err(|e| format!("reaper_clipboard_write_failed: {}", e))?;
    }

    raw::set_string_with(text_summary, clipboard_win::options::NoClear)
        .map_err(|e| format!("clipboard_write_text_failed: {}", e))?;

    Ok(())
}

#[cfg(target_os = "windows")]
pub fn write_bytes(bytes: &[u8], text_summary: &str) -> Result<(), String> {
    write_contents(bytes, text_summary, None)
}

#[cfg(target_os = "windows")]
pub fn write_bytes_with_reaper(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    write_contents(bytes, text_summary, reaper_bytes)
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
fn write_contents(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::{NSData, NSString};

    let pasteboard = NSPasteboard::generalPasteboard();
    let _ = pasteboard.clearContents();

    let text_ns = NSString::from_str(text_summary);
    let text_type = NSString::from_str("public.utf8-plain-text");
    if !pasteboard.setString_forType(&text_ns, &text_type) {
        return Err("clipboard_write_text_failed".to_string());
    }

    let data = NSData::with_bytes(bytes);
    let format_ns = NSString::from_str(OBJECT_FORMAT);
    let _ = pasteboard.setData_forType(Some(&data), &format_ns);

    if let Some(reaper_bytes) = reaper_bytes {
        let reaper_data = NSData::with_bytes(reaper_bytes);
        let reaper_format_ns = NSString::from_str(REAPER_MEDIA_FORMAT);
        if !pasteboard.setData_forType(Some(&reaper_data), &reaper_format_ns) {
            return Err("reaper_clipboard_write_failed".to_string());
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub fn write_bytes(bytes: &[u8], text_summary: &str) -> Result<(), String> {
    write_contents(bytes, text_summary, None)
}

#[cfg(target_os = "macos")]
pub fn write_bytes_with_reaper(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    write_contents(bytes, text_summary, reaper_bytes)
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
pub fn write_bytes(bytes: &[u8], text_summary: &str) -> Result<(), String> {
    let formats: Vec<(&str, &[u8])> = vec![
        (OBJECT_FORMAT, bytes),
        ("UTF8_STRING", text_summary.as_bytes()),
        ("text/plain", text_summary.as_bytes()),
    ];
    crate::linux_clipboard::write_multi(&formats)
}

/// Linux now co-writes REAPERMedia in the same clipboard ownership session.
/// Both the SWELL MIME name used by REAPER on Linux and the legacy short
/// target are advertised so old and new readers can paste the data.
#[cfg(target_os = "linux")]
pub fn write_bytes_with_reaper(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    let mut formats: Vec<(&str, &[u8])> = vec![(OBJECT_FORMAT, bytes)];
    if let Some(reaper_bytes) = reaper_bytes {
        formats.push((
            crate::linux_clipboard::REAPER_MEDIA_LINUX_FORMAT,
            reaper_bytes,
        ));
        formats.push((
            crate::linux_clipboard::REAPER_MEDIA_LEGACY_FORMAT,
            reaper_bytes,
        ));
    }
    formats.push(("UTF8_STRING", text_summary.as_bytes()));
    formats.push(("text/plain", text_summary.as_bytes()));
    crate::linux_clipboard::write_multi(&formats)
}

#[cfg(target_os = "linux")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use std::process::Command;

    let is_wayland = is_wayland_session();

    let custom = if is_wayland {
        Command::new("wl-paste")
            .args(["--type", OBJECT_FORMAT])
            .output()
    } else {
        Command::new("xclip")
            .args(["-selection", "clipboard", "-target", OBJECT_FORMAT, "-o"])
            .output()
    };

    if let Ok(output) = custom {
        if output.status.success() && !output.stdout.is_empty() {
            return Ok(Some(output.stdout));
        }
    }

    let text = if is_wayland {
        Command::new("wl-paste").args(["--no-newline"]).output()
    } else {
        Command::new("xclip").args(["-selection", "clipboard", "-o"]).output()
    };

    match text {
        Ok(output) if output.status.success() => {
            Ok(decode_text_envelope(String::from_utf8_lossy(&output.stdout).trim()))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Unsupported platform
// ---------------------------------------------------------------------------

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn write_bytes(_bytes: &[u8], _text_summary: &str) -> Result<(), String> {
    Err("clipboard_unsupported_platform".to_string())
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn write_bytes_with_reaper(
    _bytes: &[u8],
    _text_summary: &str,
    _reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    Err("clipboard_unsupported_platform".to_string())
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    Err("clipboard_unsupported_platform".to_string())
}
