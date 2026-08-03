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

/// Parse `--log-file <path>` / `--log-file=<path>` from the command line.
///
/// Returns `None` when no log file was requested (no file logging at all).
/// A path of `-` explicitly disables file logging (console-only).
fn parse_log_file_arg(args: &[String]) -> Option<std::path::PathBuf> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if let Some(path) = arg.strip_prefix("--log-file=") {
            if path.is_empty() || path == "-" {
                return None;
            }
            return Some(std::path::PathBuf::from(path));
        }
        if arg == "--log-file" {
            match iter.next() {
                Some(path) if !path.is_empty() && path != "-" => {
                    return Some(std::path::PathBuf::from(path));
                }
                _ => return None,
            }
        }
    }
    None
}

/// Capture stderr into a pipe and tee it to both the real stderr (console)
/// and the requested log file, prefixing every line with a timestamp.
///
/// When `log_path` is `None` (no `--log-file` on the command line) nothing
/// is written to disk at all.  This implementation is shared by every
/// platform; only the raw pipe API differs (CRT fds on Windows vs POSIX
/// fds on Unix), so both variants keep the exact same control flow.
#[cfg(windows)]
fn init_file_log(log_path: Option<std::path::PathBuf>) {
    use std::io::{BufRead, BufReader, Write};
    use std::os::windows::io::FromRawHandle;

    let log_path = match log_path {
        Some(p) => p,
        None => return, // No --log-file: do not write any log file.
    };
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Save the original stderr so the tee thread can still echo to console.
    let saved = unsafe { libc::dup(2) };
    if saved < 0 {
        return;
    }

    // Anonymous pipe (CRT _pipe).  1 MiB buffer keeps verbose ORT init output
    // from blocking the main thread while the reader thread starts up.
    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr(), 1048576, libc::O_BINARY) } != 0 {
        unsafe { libc::close(saved) };
        return;
    }

    // Replace fd 2 with the pipe's write end.
    unsafe {
        libc::dup2(fds[1], 2);
        libc::close(fds[1]);
    }

    let read_handle = unsafe { libc::get_osfhandle(fds[0]) };
    let console_handle = unsafe { libc::get_osfhandle(saved) };
    if read_handle == -1 || console_handle == -1 {
        return;
    }

    std::thread::spawn(move || {
        let pipe_reader = unsafe { std::fs::File::from_raw_handle(read_handle as *mut _) };
        let mut reader = BufReader::new(pipe_reader);
        let mut out_file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
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

#[cfg(unix)]
fn init_file_log(log_path: Option<std::path::PathBuf>) {
    use std::io::{BufRead, BufReader, Write};
    use std::os::fd::FromRawFd;

    let log_path = match log_path {
        Some(p) => p,
        None => return, // No --log-file: do not write any log file.
    };
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Save the original stderr so the tee thread can still echo to console.
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

    std::thread::spawn(move || {
        let reader = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let mut reader = BufReader::new(reader);
        let mut out_file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
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

fn main() {
    #[cfg(target_os = "linux")]
    sanitize_gtk_modules_for_appimage();

    let args: Vec<String> = std::env::args().collect();
    init_file_log(parse_log_file_arg(&args));

    // CLI diagnostics: `HiFiShifter --benchmark` runs the inference-device
    // benchmark and prints the JSON result, then exits.  Useful for
    // debugging GPU/CoreML issues from a terminal without opening the GUI.
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
