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
//!
//! ## Concurrency and contention
//!
//! The Windows clipboard is a process-global, exclusive resource:
//! `OpenClipboard` fails (ERROR_ACCESS_DENIED) while *any* other window —
//! in this process or another — holds it open.  Clipboard managers, RDP /
//! Citrix sessions, screen readers, and apps doing delayed rendering
//! (Office, Explorer, WebView2) routinely hold it for tens to hundreds of
//! milliseconds, which is exactly the kind of transient contention that used
//! to make Clip copy/cut fail at random.
//!
//! Two defenses are implemented here:
//! - `CLIPBOARD_LOCK` serializes every clipboard session inside this
//!   process (a second `OpenClipboard` from our own process would fail).
//! - `open_clipboard_with_retry` retries `OpenClipboard` with *real* sleeps
//!   (`clipboard-win`'s `new_attempts` only yields with `Sleep(0)`, which is
//!   effectively no wait at all), so momentary external holders almost never
//!   surface to the user.
//!
//! Additionally `has_timeline_clipboard` is answered from a sequence-number
//! keyed cache (`clipboard_seq_num` / `CLIPBOARD_CACHE`) so the frontend's
//! periodic availability poll stops opening the OS clipboard at all unless
//! its content actually changed.

use base64::Engine;
use std::sync::Mutex;

pub const OBJECT_FORMAT: &str = "application/x-hifishifter-object";
#[cfg(not(target_os = "linux"))]
pub const REAPER_MEDIA_FORMAT: &str = "REAPERMedia";
const TEXT_PREFIX: &str = "HIFISHIFTER_CLIPBOARD_V1:";

/// Serializes every clipboard access in this process.  The Windows clipboard
/// can only be opened by one thread at a time — even within the same process
/// a second `OpenClipboard` fails with ERROR_ACCESS_DENIED.  The guard is
/// held only around the raw Win32 session, never during encode/decode.
static CLIPBOARD_LOCK: Mutex<()> = Mutex::new(());

/// `clipboard-win::Clipboard::new_attempts` retries with `Sleep(0)` — it
/// yields the scheduler but waits virtually no time, so genuine contention
/// (a clipboard manager / RDP / WebView2 / another app's delayed rendering)
/// fails all attempts instantly.  We retry with growing real sleeps instead.
#[cfg(target_os = "windows")]
const OPEN_RETRIES: u32 = 12;
#[cfg(target_os = "windows")]
const OPEN_RETRY_BASE_MS: u64 = 25;
#[cfg(target_os = "windows")]
const OPEN_RETRY_MAX_MS: u64 = 150;
/// Whole write sessions (open → empty → set) are retried as one atomic unit:
/// a failure mid-session leaves the clipboard already emptied, so only a full
/// re-run can guarantee a complete write.
#[cfg(target_os = "windows")]
const WRITE_SESSION_ATTEMPTS: u32 = 3;

fn decode_text_envelope(text: &str) -> Option<Vec<u8>> {
    let body = text.strip_prefix(TEXT_PREFIX)?;
    base64::engine::general_purpose::STANDARD.decode(body).ok()
}

// ---------------------------------------------------------------------------
// Shared open/backoff helpers (Windows)
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn sleep_backoff(attempt: u32) {
    let ms = (OPEN_RETRY_BASE_MS << attempt.min(5)).min(OPEN_RETRY_MAX_MS);
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

/// Open the clipboard with real backoff so transient external contention
/// (clipboard managers, RDP, other apps' delayed rendering, WebView2)
/// almost never fails a copy.  Worst case ≈ 800ms before reporting failure.
#[cfg(target_os = "windows")]
fn open_clipboard_with_retry() -> Result<clipboard_win::Clipboard, String> {
    use clipboard_win::Clipboard;
    let mut last_error: Option<String> = None;
    for attempt in 0..OPEN_RETRIES {
        match Clipboard::new() {
            Ok(clip) => return Ok(clip),
            Err(error) => last_error = Some(format!("{}", error)),
        }
        if attempt + 1 < OPEN_RETRIES {
            sleep_backoff(attempt);
        }
    }
    Err(format!(
        "clipboard_open_failed: {}",
        last_error.unwrap_or_default()
    ))
}

/// Run a raw Win32 clipboard session under the process-wide lock.
/// `f` receives the open `Clipboard` handle (kept alive for the whole
/// session so `clipboard_win` raw fns see an open clipboard).
#[cfg(target_os = "windows")]
pub(crate) fn clipboard_session<T>(
    f: impl FnOnce(&clipboard_win::Clipboard) -> Result<T, String>,
) -> Result<T, String> {
    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let clip = open_clipboard_with_retry()?;
    f(&clip)
}

// ---------------------------------------------------------------------------
// Sequence-number keyed cache (for the has_* availability polls)
// ---------------------------------------------------------------------------

/// OS clipboard sequence number — cheap, never opens the clipboard.
/// (Windows: GetClipboardSequenceNumber; macOS: NSPasteboard changeCount;
/// Linux: no equivalent, returns None.)
pub fn clipboard_seq_num() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        Some(u64::from(clipboard_win::seq_num().map(|n| n.get()).unwrap_or(0)))
    }
    #[cfg(target_os = "macos")]
    {
        use objc2_app_kit::NSPasteboard;
        let pasteboard = NSPasteboard::generalPasteboard();
        Some(pasteboard.changeCount() as u64)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        None
    }
}

/// Last known clipboard state recorded by this process, keyed to the OS
/// clipboard sequence number.  Lets `has_timeline_clipboard` answer without
/// opening the OS clipboard when nothing changed since the last evaluation.
#[derive(Clone, Debug)]
pub struct ClipboardCacheEntry {
    pub seq: u64,
    pub hifi_kind: Option<String>,
    pub hifi_clip_count: u64,
    pub hifi_track_count: u64,
    pub hifi_source_project: Option<String>,
    pub reaper_available: bool,
}

static CLIPBOARD_CACHE: Mutex<Option<ClipboardCacheEntry>> = Mutex::new(None);

pub fn write_clipboard_cache(entry: ClipboardCacheEntry) {
    if let Ok(mut cache) = CLIPBOARD_CACHE.lock() {
        *cache = Some(entry);
    }
}

/// Drop any cached clipboard state (e.g. after an untracked write).
pub fn invalidate_clipboard_cache() {
    if let Ok(mut cache) = CLIPBOARD_CACHE.lock() {
        *cache = None;
    }
}

/// Returns the cached entry when the OS clipboard has not changed since it
/// was recorded; returns None otherwise (caller must perform a real read).
pub fn read_clipboard_cache_if_current() -> Option<ClipboardCacheEntry> {
    let seq = clipboard_seq_num()?;
    let cache = CLIPBOARD_CACHE.lock().ok()?;
    let entry = cache.as_ref()?;
    if entry.seq == seq {
        Some(entry.clone())
    } else {
        None
    }
}

/// Whether the REAPERMedia format is present on the clipboard right now.
/// Used when refreshing the availability cache after a real read, so the
/// cached `reaper_available` flag stays accurate for foreign copies too.
pub fn has_reaper_format() -> bool {
    #[cfg(target_os = "windows")]
    {
        use clipboard_win::{raw, register_format};
        let format = match register_format(REAPER_MEDIA_FORMAT) {
            Some(format) => format,
            None => return false,
        };
        clipboard_session(|_clip| Ok(raw::is_format_avail(format.get()))).unwrap_or(false)
    }
    #[cfg(target_os = "macos")]
    {
        use objc2_app_kit::NSPasteboard;
        use objc2_foundation::NSString;
        let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pasteboard = NSPasteboard::generalPasteboard();
        let pb_type = NSString::from_str(REAPER_MEDIA_FORMAT);
        pasteboard.dataForType(&pb_type).is_some()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        false
    }
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
    use clipboard_win::{raw, register_format};

    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Format registration is cheap and does not require an open clipboard.
    let object_format =
        register_format(OBJECT_FORMAT).ok_or_else(|| "clipboard_format_not_found".to_string())?;
    let reaper_format = reaper_bytes
        .map(|_| register_format(REAPER_MEDIA_FORMAT))
        .flatten();

    let mut last_error: Option<String> = None;

    // The whole open → empty → set session is retried as one unit: a failed
    // mid-session set leaves the clipboard already emptied, so only a full
    // re-run can guarantee a complete write.
    for attempt in 0..WRITE_SESSION_ATTEMPTS {
        let _clipboard = match open_clipboard_with_retry() {
            Ok(clip) => clip,
            Err(error) => {
                last_error = Some(error);
                sleep_backoff(attempt);
                continue;
            }
        };

        if let Err(error) = raw::empty() {
            last_error = Some(format!("clipboard_empty_failed: {}", error));
            drop(_clipboard);
            sleep_backoff(attempt);
            continue;
        }

        if let Err(error) = raw::set_without_clear(object_format.get(), bytes) {
            last_error = Some(format!("clipboard_write_custom_failed: {}", error));
            drop(_clipboard);
            sleep_backoff(attempt);
            continue;
        }

        // REAPERMedia and the text/plain summary are best-effort: the
        // HiFiShifter data is already committed, so a failure in an
        // auxiliary format must never fail the user's copy.
        if let (Some(format), Some(reaper_bytes)) = (reaper_format.as_ref(), reaper_bytes) {
            if let Err(error) = raw::set_without_clear(format.get(), reaper_bytes) {
                log::error!("[hifishifter] reaper clipboard write failed: {error}");
            }
        }
        if let Err(error) = raw::set_string_with(text_summary, clipboard_win::options::NoClear) {
            log::error!("[hifishifter] clipboard text write failed: {error}");
        }

        return Ok(());
    }

    Err(last_error.unwrap_or_else(|| "clipboard_open_failed".to_string()))
}

#[cfg(target_os = "windows")]
pub fn write_bytes(bytes: &[u8], text_summary: &str) -> Result<(), String> {
    write_contents(bytes, text_summary, None)?;
    invalidate_clipboard_cache();
    Ok(())
}

#[cfg(target_os = "windows")]
pub fn write_bytes_with_reaper(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    write_contents(bytes, text_summary, reaper_bytes)?;
    invalidate_clipboard_cache();
    Ok(())
}

#[cfg(target_os = "windows")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use clipboard_win::{raw, register_format};

    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _clipboard = open_clipboard_with_retry()?;

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

    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

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

    // REAPERMedia is best-effort on macOS as well.
    if let Some(reaper_bytes) = reaper_bytes {
        let reaper_data = NSData::with_bytes(reaper_bytes);
        let reaper_format_ns = NSString::from_str(REAPER_MEDIA_FORMAT);
        if !pasteboard.setData_forType(Some(&reaper_data), &reaper_format_ns) {
            log::error!("[hifishifter] reaper clipboard write failed");
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub fn write_bytes(bytes: &[u8], text_summary: &str) -> Result<(), String> {
    write_contents(bytes, text_summary, None)?;
    invalidate_clipboard_cache();
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn write_bytes_with_reaper(
    bytes: &[u8],
    text_summary: &str,
    reaper_bytes: Option<&[u8]>,
) -> Result<(), String> {
    write_contents(bytes, text_summary, reaper_bytes)?;
    invalidate_clipboard_cache();
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use objc2_app_kit::NSPasteboard;
    use objc2_foundation::NSString;

    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

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

// ---------------------------------------------------------------------------
// Linux
// ---------------------------------------------------------------------------

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
    let result = crate::linux_clipboard::write_multi(&formats);
    if result.is_ok() {
        invalidate_clipboard_cache();
    }
    result
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
    let result = crate::linux_clipboard::write_multi(&formats);
    if result.is_ok() {
        invalidate_clipboard_cache();
    }
    result
}

#[cfg(target_os = "linux")]
pub fn read_bytes() -> Result<Option<Vec<u8>>, String> {
    use std::process::Command;

    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

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
        Command::new("xclip")
            .args(["-selection", "clipboard", "-o"])
            .output()
    };

    match text {
        Ok(output) if output.status.success() => Ok(decode_text_envelope(
            String::from_utf8_lossy(&output.stdout).trim(),
        )),
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