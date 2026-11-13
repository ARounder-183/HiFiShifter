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

    // ── 用户级目录的一次性布局迁移 ──────────────────────────────────────
    //
    // 【为什么必须在日志初始化之前】`init_logging` 会创建
    // `%LOCALAPPDATA%\HiFiShifter\logs` 并立刻写文件。迁移的判据是"新旧根同时存在
    // 就绝不合并"，所以日志目录一旦先被建出来，旧日志目录就会被判成冲突、永久
    // 搬不过来。这是整个启动链上唯一能保证顺序的位置。
    let layout = backend_lib::app_paths_migrate::ensure_layout_v2();

    let args: Vec<String> = std::env::args().collect();

    // 日志初始化（stderr logger + panic hook + stderr→文件 tee）：
    // `--log-file=<path>` 指定文件，`--log-file=-` 显式关闭，
    // 缺省写入平台标准日志目录（详见 logging 模块文档）。
    backend_lib::logging::init_logging(backend_lib::logging::choice_from_args(&args), &args);

    // 迁移发生在日志初始化之前，结局必须在这里补记（否则那段时间里发生的事
    // 在日志里完全不可见 —— 迁移恰恰是"用户数据动了位置"的大事）。
    layout.log_summary("app");

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

    // Linux：把逃生开关翻译成 GTK/WebKit 认的环境变量，并把会话与 EGL 状态写进
    // 日志。必须赶在 run() 之前 —— GTK 一旦初始化，GDK_BACKEND 就不再生效。
    #[cfg(target_os = "linux")]
    backend_lib::linux_desktop::prepare();

    backend_lib::run()
}
