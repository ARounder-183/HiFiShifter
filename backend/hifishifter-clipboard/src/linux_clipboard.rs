//! 共享Linux多格式剪贴板，保留原X11/Wayland实现。
//! Linux multi-format system clipboard transport.
//!
//! `xclip` / `wl-copy` can only offer a single payload per selection, which
//! is not enough for the native HiFiShifter copy operation: it must publish
//! both `application/x-hifishifter-object` and REAPERMedia (plus a readable
//! text summary) in the same clipboard ownership session.
//!
//! X11: one long-lived XCB clipboard owner window advertises every target and
//! serves selection requests itself (including INCR for large payloads).
//! Wayland: `wl-clipboard-rs` offers multiple independent MIME sources.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread;

use xcb;
use xcb::x;

/// REAPER's Linux builds (SWELL) expose clipboard formats under this MIME
/// prefix. Keep the short name as a fallback for older HiFiShifter versions.
pub const REAPER_MEDIA_LINUX_FORMAT: &str = "application/swell-REAPERMedia";
pub const REAPER_MEDIA_LEGACY_FORMAT: &str = "REAPERMedia";
pub const STANDARD_MIDI_FILE_LINUX_FORMAT: &str = "application/swell-Standard MIDI File";
pub const STANDARD_MIDI_FILE_LEGACY_FORMAT: &str = "Standard MIDI File";

const INCR_CHUNK_SIZE: usize = 4000;

pub fn is_wayland_session() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
        || std::env::var("XDG_SESSION_TYPE")
            .map(|value| value.eq_ignore_ascii_case("wayland"))
            .unwrap_or(false)
}

/// Publish every `(target, bytes)` pair in one atomic clipboard update.
pub fn write_multi(formats: &[(&str, &[u8])]) -> Result<(), String> {
    if is_wayland_session() {
        write_wayland_multi(formats)
    } else {
        write_x11_multi(formats)
    }
}

// ---------------------------------------------------------------------------
// X11 multi-target owner
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ClipboardEntry {
    target: x::Atom,
    data: Vec<u8>,
}

struct IncrTransfer {
    requestor: x::Window,
    property: x::Atom,
    target: x::Atom,
    pos: usize,
}

struct X11Atoms {
    clipboard: x::Atom,
    targets: x::Atom,
    incr: x::Atom,
}

struct X11Context {
    connection: xcb::Connection,
    window: x::Window,
    atoms: X11Atoms,
}

/// intern 一个 atom。
///
/// xcb 1.x 起不再有 `intern_atom()` 这类自由函数：改成发 `x::InternAtom`
/// 请求拿 cookie，再 `wait_for_reply`。
fn intern_atom(connection: &xcb::Connection, name: &str) -> Result<x::Atom, String> {
    let cookie = connection.send_request(&x::InternAtom {
        only_if_exists: false,
        name: name.as_bytes(),
    });
    connection
        .wait_for_reply(cookie)
        .map(|reply| reply.atom())
        .map_err(|error| format!("clipboard_x11_atom_failed ({name}): {error:?}"))
}

impl X11Context {
    fn new() -> Result<Self, String> {
        let (connection, screen) = xcb::Connection::connect(None)
            .map_err(|error| format!("clipboard_x11_open_failed: {error:?}"))?;
        let window = connection.generate_id();

        {
            let root = connection
                .get_setup()
                .roots()
                .nth(screen as usize)
                .ok_or_else(|| "clipboard_x11_no_screen".to_string())?;
            connection.send_request(&x::CreateWindow {
                depth: x::COPY_FROM_PARENT as u8,
                wid: window,
                parent: root.root(),
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                border_width: 0,
                class: x::WindowClass::InputOutput,
                visual: root.root_visual(),
                value_list: &[x::Cw::EventMask(x::EventMask::PROPERTY_CHANGE)],
            });
            let _ = connection.flush();
        }

        let atoms = X11Atoms {
            clipboard: intern_atom(&connection, "CLIPBOARD")?,
            targets: intern_atom(&connection, "TARGETS")?,
            incr: intern_atom(&connection, "INCR")?,
        };

        Ok(Self {
            connection,
            window,
            atoms,
        })
    }

    fn get_atom(&self, name: &str) -> Result<x::Atom, String> {
        intern_atom(&self.connection, name)
    }
}

struct X11Owner {
    context: Arc<X11Context>,
    entries: Arc<RwLock<Vec<ClipboardEntry>>>,
    refresh: Sender<()>,
}

static X11_OWNER: OnceLock<Mutex<X11Owner>> = OnceLock::new();

fn x11_owner() -> Result<&'static Mutex<X11Owner>, String> {
    if let Some(owner) = X11_OWNER.get() {
        return Ok(owner);
    }

    let context = Arc::new(X11Context::new()?);
    let entries = Arc::new(RwLock::new(Vec::<ClipboardEntry>::new()));
    let (refresh, refresh_receiver) = mpsc::channel::<()>();

    let serve_context = Arc::clone(&context);
    let serve_entries = Arc::clone(&entries);
    thread::Builder::new()
        .name("hifishifter-x11-clipboard".to_string())
        .spawn(move || x11_serve(serve_context, serve_entries, refresh_receiver))
        .map_err(|error| format!("clipboard_x11_thread_failed: {error}"))?;

    let owner = X11Owner {
        context,
        entries,
        refresh,
    };
    match X11_OWNER.set(Mutex::new(owner)) {
        Ok(()) => X11_OWNER
            .get()
            .ok_or_else(|| "clipboard_x11_init_failed".to_string()),
        Err(_) => X11_OWNER
            .get()
            .ok_or_else(|| "clipboard_x11_init_failed".to_string()),
    }
}

fn write_x11_multi(formats: &[(&str, &[u8])]) -> Result<(), String> {
    let owner_guard = x11_owner()?;
    let owner = owner_guard
        .lock()
        .map_err(|_| "clipboard_x11_lock_failed".to_string())?;

    let mut next_entries = Vec::with_capacity(formats.len());
    for (target, bytes) in formats {
        let atom = owner
            .context
            .get_atom(target)
            .map_err(|error| format!("clipboard_x11_atom_failed ({target}): {error:?}"))?;
        next_entries.push(ClipboardEntry {
            target: atom,
            data: bytes.to_vec(),
        });
    }

    *owner
        .entries
        .write()
        .map_err(|_| "clipboard_x11_entries_lock_failed".to_string())? = next_entries;
    let _ = owner.refresh.send(());

    let connection = &owner.context.connection;
    let window = owner.context.window;
    let clipboard = owner.context.atoms.clipboard;

    connection.send_request(&x::SetSelectionOwner {
        owner: window,
        selection: clipboard,
        time: x::Time::CurrentTime as u32,
    });
    let _ = connection.flush();

    let cookie = connection.send_request(&x::GetSelectionOwner {
        selection: clipboard,
    });
    match connection.wait_for_reply(cookie) {
        Ok(reply) if reply.owner() == window => Ok(()),
        Ok(_) => Err("clipboard_x11_owner_failed".to_string()),
        Err(error) => Err(format!("clipboard_x11_owner_failed: {error:?}")),
    }
}

fn x11_serve(
    context: Arc<X11Context>,
    entries: Arc<RwLock<Vec<ClipboardEntry>>>,
    refresh: Receiver<()>,
) {
    let mut incr_transfers: HashMap<x::Atom, IncrTransfer> = HashMap::new();

    loop {
        while let Ok(()) = refresh.try_recv() {
            incr_transfers.clear();
        }

        // xcb 1.x：事件是 `xcb::Event::X(x::Event::…)` 枚举，不再靠
        // `response_type()` + `cast_event` 手工还原。
        let Ok(event) = context.connection.wait_for_event() else {
            continue;
        };

        match event {
            xcb::Event::X(x::Event::SelectionRequest(event)) => {
                let Ok(current_entries) = entries.read() else {
                    continue;
                };

                let mut success = false;
                if event.target() == context.atoms.targets {
                    let mut targets = Vec::with_capacity(current_entries.len() + 1);
                    targets.push(context.atoms.targets);
                    targets.extend(current_entries.iter().map(|entry| entry.target));
                    context.connection.send_request(&x::ChangeProperty {
                        mode: x::PropMode::Replace,
                        window: event.requestor(),
                        property: event.property(),
                        r#type: x::ATOM_ATOM,
                        data: &targets,
                    });
                    success = true;
                } else if let Some(entry) = current_entries
                    .iter()
                    .find(|entry| entry.target == event.target())
                {
                    let max_length = context.connection.get_maximum_request_length() as usize * 4;
                    if entry.data.len() < max_length.saturating_sub(24) {
                        context.connection.send_request(&x::ChangeProperty {
                            mode: x::PropMode::Replace,
                            window: event.requestor(),
                            property: event.property(),
                            r#type: event.target(),
                            data: &entry.data,
                        });
                        success = true;
                    } else {
                        // INCR: the requestor deletes the property after each
                        // chunk and we answer with the next chunk on
                        // PropertyNotify(PROPERTY_DELETE).
                        context.connection.send_request(&x::ChangeWindowAttributes {
                            window: event.requestor(),
                            value_list: &[x::Cw::EventMask(x::EventMask::PROPERTY_CHANGE)],
                        });
                        context.connection.send_request(&x::ChangeProperty {
                            mode: x::PropMode::Replace,
                            window: event.requestor(),
                            property: event.property(),
                            r#type: context.atoms.incr,
                            data: &[entry.data.len().min(u32::MAX as usize) as u32],
                        });
                        incr_transfers.insert(
                            event.property(),
                            IncrTransfer {
                                requestor: event.requestor(),
                                property: event.property(),
                                target: event.target(),
                                pos: 0,
                            },
                        );
                        success = true;
                    }
                }

                let property = if success {
                    event.property()
                } else {
                    x::ATOM_NONE
                };
                let notify = x::SelectionNotifyEvent::new(
                    event.time(),
                    event.requestor(),
                    event.selection(),
                    event.target(),
                    property,
                );
                context.connection.send_request(&x::SendEvent {
                    propagate: false,
                    destination: x::SendEventDest::Window(event.requestor()),
                    event_mask: x::EventMask::empty(),
                    event: &notify,
                });
                let _ = context.connection.flush();
            }
            xcb::Event::X(x::Event::PropertyNotify(event)) => {
                if event.state() != x::Property::Delete {
                    continue;
                }

                let Some(transfer) = incr_transfers.get_mut(&event.atom()) else {
                    continue;
                };
                let Ok(current_entries) = entries.read() else {
                    continue;
                };
                let Some(entry) = current_entries
                    .iter()
                    .find(|entry| entry.target == transfer.target)
                else {
                    incr_transfers.remove(&event.atom());
                    continue;
                };

                let remaining = entry.data.len().saturating_sub(transfer.pos);
                let chunk_len = remaining.min(INCR_CHUNK_SIZE);
                context.connection.send_request(&x::ChangeProperty {
                    mode: x::PropMode::Replace,
                    window: transfer.requestor,
                    property: transfer.property,
                    r#type: transfer.target,
                    data: &entry.data[transfer.pos..transfer.pos + chunk_len],
                });
                transfer.pos += chunk_len;
                if chunk_len == 0 {
                    incr_transfers.remove(&event.atom());
                }
                let _ = context.connection.flush();
            }
            xcb::Event::X(x::Event::SelectionClear(event)) => {
                if event.selection() == context.atoms.clipboard {
                    if let Ok(mut current_entries) = entries.write() {
                        current_entries.clear();
                    }
                    incr_transfers.clear();
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Wayland multi-source writer
// ---------------------------------------------------------------------------

fn write_wayland_multi(formats: &[(&str, &[u8])]) -> Result<(), String> {
    use wl_clipboard_rs::copy::{self, MimeSource, MimeType, Options, Source};

    let mut options = Options::new();
    options.clipboard(copy::ClipboardType::Regular);
    options.foreground(false);

    let mut sources = Vec::with_capacity(formats.len());
    for (mime, bytes) in formats {
        let mime_type = if *mime == "text/plain" || *mime == "UTF8_STRING" {
            MimeType::Text
        } else {
            MimeType::Specific((*mime).to_string())
        };
        sources.push(MimeSource {
            source: Source::Bytes(bytes.to_vec().into_boxed_slice()),
            mime_type,
        });
    }

    options
        .copy_multi(sources)
        .map_err(|error| format!("clipboard_write_failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn xclip_read(target: &str) -> Option<Vec<u8>> {
        let output = Command::new("xclip")
            .args(["-selection", "clipboard", "-target", target, "-o"])
            .output()
            .ok()?;
        output.status.success().then_some(output.stdout)
    }

    /// Manual integration test (needs a running X server and modifies the
    /// X11 clipboard): `HIFISHIFTER_CLIPBOARD_TEST=1 cargo test \
    /// linux_clipboard::tests::x11_multi_target_clipboard_roundtrip`.
    #[test]
    fn x11_multi_target_clipboard_roundtrip() {
        if std::env::var("HIFISHIFTER_CLIPBOARD_TEST").as_deref() != Ok("1") {
            log::warn!("skipped: set HIFISHIFTER_CLIPBOARD_TEST=1 to run");
            return;
        }
        if is_wayland_session() {
            log::warn!("skipped: test only covers the X11 backend");
            return;
        }

        let object = b"hifishifter-object-payload";
        let reaper = b"reapermedia-payload";
        let summary = b"hifishifter-clipboard-test";
        // Deliberately larger than the X11 maximum request length so the
        // INCR transfer path is exercised as well.
        let large = vec![0x5a_u8; 400_000];

        write_multi(&[
            (crate::system_clipboard::OBJECT_FORMAT, &object[..]),
            (REAPER_MEDIA_LINUX_FORMAT, &reaper[..]),
            (REAPER_MEDIA_LEGACY_FORMAT, &reaper[..]),
            ("UTF8_STRING", &summary[..]),
            ("text/plain", &summary[..]),
            ("application/x-hifishifter-incremental-test", &large),
        ])
        .expect("write_multi should succeed");

        let targets =
            String::from_utf8_lossy(&xclip_read("TARGETS").expect("read TARGETS")).to_string();
        assert!(
            targets.contains(crate::system_clipboard::OBJECT_FORMAT),
            "TARGETS missing object format: {targets:?}"
        );
        assert!(
            targets.contains(REAPER_MEDIA_LINUX_FORMAT),
            "TARGETS missing SWELL REAPERMedia format: {targets:?}"
        );
        assert!(
            targets.contains(REAPER_MEDIA_LEGACY_FORMAT),
            "TARGETS missing legacy REAPERMedia format: {targets:?}"
        );

        assert_eq!(
            xclip_read(crate::system_clipboard::OBJECT_FORMAT).expect("read object"),
            object
        );
        assert_eq!(
            xclip_read(REAPER_MEDIA_LINUX_FORMAT).expect("read SWELL REAPERMedia"),
            reaper
        );
        assert_eq!(
            xclip_read(REAPER_MEDIA_LEGACY_FORMAT).expect("read legacy REAPERMedia"),
            reaper
        );
        assert_eq!(
            xclip_read("text/plain").expect("read text summary"),
            summary
        );
        assert_eq!(
            xclip_read("application/x-hifishifter-incremental-test").expect("read large payload"),
            large
        );
    }
}
