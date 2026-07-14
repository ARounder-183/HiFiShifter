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

/// In release builds on Windows with the `logging` feature, capture all
/// stderr output to a `log.txt` next to the executable.  Each line is
/// prefixed with a `[HH:MM:SS.mmm]` timestamp so that issues can be
/// correlated with user reports.
///
/// Implementation:  a pipe intercepts fd 2; a background thread reads from
/// the pipe, timestamps every line, and writes the result to both the log
/// file and the *real* console stderr.  This way nothing is lost — the user
/// still sees stderr, and a timestamped copy lands on disk.
#[cfg(all(feature = "logging", windows, not(debug_assertions)))]
fn init_file_log() {
    use std::io::{BufRead, BufReader, Write};

    let log_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("log.txt");

    // Save original stderr so we can still echo output to the console.
    let saved = unsafe { libc::dup(2) };
    if saved < 0 {
        return;
    }

    // Anonymous pipe — everything written to fd 2 will come out the read end.
    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        unsafe { libc::close(saved); }
        return;
    }

    // Replace fd 2 with the pipe's write end.
    unsafe { libc::dup2(fds[1], 2); }
    unsafe { libc::close(fds[1]); } // our copy — fd 2 still points to the pipe

    // ---- background thread: timestamp & tee to file + console ----
    let log_path_copy = log_path.clone();
    std::thread::spawn(move || {
        use std::os::windows::io::FromRawHandle;
        let pipe_reader = unsafe { std::fs::File::from_raw_handle(fds[0]) };
        let reader = BufReader::new(pipe_reader);
        let mut out_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path_copy)
            .expect("failed to open log file");
        // Initial banner so we know when this session started.
        let _ = writeln!(
            out_file,
            "==== HiFiShifter log started at {} ====",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        );
        let _ = out_file.flush();

        let mut console = unsafe { std::fs::File::from_raw_handle(saved) };
        let mut line_buf = String::new();

        loop {
            line_buf.clear();
            match reader.read_line(&mut line_buf) {
                Ok(0) => break, // pipe closed (process exiting)
                Ok(_) => {
                    let ts = chrono::Local::now().format("%H:%M:%S%.3f");
                    // `read_line` includes the trailing `\n`; strip it so we
                    // can add our own uniform line ending.
                    let body = line_buf.trim_end_matches('\n');
                    let stamped = format!("[{}] {}\n", ts, body);
                    let _ = out_file.write_all(stamped.as_bytes());
                    let _ = out_file.flush();
                    let _ = console.write_all(stamped.as_bytes());
                    let _ = console.flush();
                }
                Err(_) => break,
            }
        }
    });
}

#[cfg(not(all(feature = "logging", windows, not(debug_assertions))))]
fn init_file_log() {}

fn main() {
    #[cfg(target_os = "linux")]
    sanitize_gtk_modules_for_appimage();

    init_file_log();

    backend_lib::run()
}
