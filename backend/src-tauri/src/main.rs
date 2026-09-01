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

fn main() {
    #[cfg(target_os = "linux")]
    sanitize_gtk_modules_for_appimage();

    let args: Vec<String> = std::env::args().collect();

    // 日志初始化（stderr logger + panic hook + stderr→文件 tee）：
    // `--log-file=<path>` 指定文件，`--log-file=-` 显式关闭，
    // 缺省写入平台标准日志目录（详见 logging 模块文档）。
    backend_lib::logging::init_logging(backend_lib::logging::choice_from_args(&args));

    // CLI diagnostics: `HiFiShifter --benchmark` runs the inference-device
    // benchmark and prints the JSON result, then exits.  Useful for
    // debugging GPU/CoreML issues from a terminal without opening the GUI.
    // JSON 结果输出到 stdout（不经 tee），供脚本/用户直接解析。
    if args.iter().any(|a| a == "--benchmark" || a == "--diagnose") {
        match backend_lib::run_vocoder_benchmark_cli() {
            Ok(json) => {
                println!("{json}");
                std::process::exit(0);
            }
            Err(e) => {
                log::error!("benchmark failed: {e}");
                std::process::exit(1);
            }
        }
    }

    backend_lib::run()
}
