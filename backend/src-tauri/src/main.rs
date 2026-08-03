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
    use std::os::windows::io::FromRawHandle;

    let log_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("log.txt");

    // Save original stderr fd so we can still echo to the real console.
    let saved = unsafe { libc::dup(2) };
    if saved < 0 {
        return;
    }

    // Anonymous pipe — the CRT's _pipe creates a pair of fds.
    // Use a 1 MB buffer so that even verbose ORT init output does not block
    // the main thread while the background reader is still getting started.
    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr(), 1048576, libc::O_BINARY) } != 0 {
        unsafe { libc::close(saved); }
        return;
    }

    // Replace fd 2 with the pipe's write end.
    unsafe { libc::dup2(fds[1], 2); }
    unsafe { libc::close(fds[1]); } // our copy — fd 2 still points to the pipe

    // Convert the read-end fd to a Windows HANDLE, then to a Rust File.
    let read_handle = unsafe { libc::get_osfhandle(fds[0]) };
    let console_handle = unsafe { libc::get_osfhandle(saved) };
    if read_handle == -1 || console_handle == -1 {
        return;
    }

    // ---- background thread: timestamp & tee to file + console ----
    let log_path_copy = log_path.clone();
    std::thread::spawn(move || {
        let pipe_reader = unsafe { std::fs::File::from_raw_handle(read_handle as *mut _) };
        let mut reader = BufReader::new(pipe_reader);
        let mut out_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path_copy)
            .expect("failed to open log file");
        let _ = writeln!(
            out_file,
            "==== HiFiShifter log started at {} ====",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        );
        let _ = out_file.flush();

        let mut console = unsafe { std::fs::File::from_raw_handle(console_handle as *mut _) };
        let mut line_buf = String::new();

        loop {
            line_buf.clear();
            match reader.read_line(&mut line_buf) {
                Ok(0) => break,
                Ok(_) => {
                    let ts = chrono::Local::now().format("%H:%M:%S%.3f");
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

#[cfg(all(feature = "logging", unix, not(debug_assertions)))]
fn init_file_log() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::fd::AsRawFd;

    // macOS: ~/Library/Logs/HiFiShifter.log; Linux: log.txt next to the exe.
    let log_path = if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .map(|h| h.join("Library").join("Logs").join("HiFiShifter.log"))
            .unwrap_or_else(|| std::path::PathBuf::from("HiFiShifter.log"))
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("log.txt")
    };
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Save the real stderr, then redirect fd 2 into a pipe.  A background
    // thread timestamps every line and tees it to the log file and console.
    let saved = unsafe { libc::dup(2) };
    if saved < 0 {
        return;
    }
    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        unsafe { libc::close(saved) };
        return;
    }
    unsafe {
        libc::dup2(fds[1], 2);
        libc::close(fds[1]);
    }

    let log_path_copy = log_path.clone();
    std::thread::spawn(move || {
        use std::os::fd::FromRawFd;

        let reader = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let mut reader = BufReader::new(reader);
        let mut out_file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path_copy)
        {
            Ok(f) => f,
            Err(_) => return,
        };
        let _ = writeln!(
            out_file,
            "==== HiFiShifter log started at {} ====",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        );
        let _ = out_file.flush();
        let mut console = unsafe { std::fs::File::from_raw_fd(saved) };
        let mut line_buf = String::new();

        loop {
            line_buf.clear();
            match reader.read_line(&mut line_buf) {
                Ok(0) => break,
                Ok(_) => {
                    let ts = chrono::Local::now().format("%H:%M:%S%.3f");
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

#[cfg(not(all(feature = "logging", unix, not(debug_assertions))))]
#[cfg(not(all(feature = "logging", windows, not(debug_assertions))))]
fn init_file_log() {}

fn main() {
    #[cfg(target_os = "linux")]
    sanitize_gtk_modules_for_appimage();

    init_file_log();

    // CLI diagnostics: `HiFiShifter --benchmark` runs the inference-device
    // benchmark and prints the JSON result, then exits.  Useful for
    // debugging GPU/CoreML issues from a terminal without opening the GUI.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--benchmark" || a == "--diagnose") {
        match backend_lib::run_vocoder_benchmark_cli() {
            Ok(json) => {
                println!("{json}");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("benchmark failed: {e}");
                std::process::exit(1);
            }
        }
    }

    backend_lib::run()
}
