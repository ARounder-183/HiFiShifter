// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(target_os = "linux")]
fn sanitize_gtk_modules_for_appimage() {
    // Some Linux environments inject xapp-gtk3-module globally, but the module
    // is optional and may be unavailable in AppImage runtime.
    if std::env::var_os("APPIMAGE").is_none() {
        return;
    }

    let raw = std::env::var("GTK_MODULES").unwrap_or_default();
    let filtered = raw
        .split(':')
        .map(str::trim)
        .filter(|m| !m.is_empty() && *m != "xapp-gtk3-module")
        .collect::<Vec<_>>()
        .join(":");

    std::env::set_var("GTK_MODULES", filtered);
}

/// In release builds on Windows, redirect stderr to a log file next to the exe
/// so that all `eprintln!` diagnostics are captured for debugging distributed builds.
/// In release builds on Windows, redirect stderr to a log file next to the exe
/// so that all diagnostics are captured for debugging distributed builds.
/// Only compiled when the `logging` feature is enabled.
#[cfg(all(feature = "logging", windows, not(debug_assertions)))]
fn init_file_log() {
    use std::fs::File;
    use std::io::Write;

    // Place log next to the executable (portable layout).
    let log_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("log.txt");

    if let Ok(mut file) = File::create(&log_path) {
        let _ = writeln!(file, "[log] HiFiShifter portable log started at {:?}", std::time::SystemTime::now());
        if let Ok(exe) = std::env::current_exe() {
            let _ = writeln!(file, "[log] exe: {}", exe.display());
        }
        let _ = file.flush();

        // Redirect stderr (fd 2) to the log file using low-level Windows API.
        // We use _open_osfhandle to get a CRT fd from the Win32 file handle,
        // then dup2 to replace fd 2.
        unsafe {
            use std::os::windows::io::AsRawHandle;
            let raw_handle = file.as_raw_handle();
            let crt_fd = libc::open_osfhandle(raw_handle as isize, 0);
            if crt_fd >= 0 {
                libc::dup2(crt_fd, 2);
                // Don't close crt_fd — it owns the handle now.
                // Leak the File so it stays open for the process lifetime.
                std::mem::forget(file);
            }
        }
    }
}

#[cfg(not(all(feature = "logging", windows, not(debug_assertions))))]
fn init_file_log() {}

fn main() {
    #[cfg(target_os = "linux")]
    sanitize_gtk_modules_for_appimage();

    init_file_log();

    backend_lib::run()
}
