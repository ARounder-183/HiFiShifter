//! 移动形态的日志初始化。
//!
//! 【为什么需要这一层】桌面的日志由 `main.rs` 初始化 —— 但移动端**没有 `main.rs`**：
//! Tauri 的移动入口是 `lib.rs` 的 `run()`（`#[cfg_attr(mobile, tauri::mobile_entry_point)]`），
//! `main.rs` 根本不参与链接。于是移动端此前一行日志都没有：不建目录、不装 logger、
//! 不装 panic hook。用户报障时没有任何现场，这是移动适配里最容易被忽略、代价却最高
//! 的一处。
//!
//! 【为什么是幂等的】桌面路径上 `main.rs` 已经初始化过一次；`run()` 里再兜一次是为了
//! 让"任何启动路径"都不漏（测试、将来的 CLI）。`log::set_logger` 第二次会返回 Err，
//! 因此这里用 `Once` 而不是靠它自己容错。

use std::sync::Once;

use crate::logging::{choice_from_args, init_logging, LogFileChoice};

static INIT: Once = Once::new();

/// 幂等初始化。桌面路径上是一次空操作。
pub fn init() {
    INIT.call_once(|| {
        // 移动端没有命令行参数（`std::env::args()` 只有可执行名），因此走默认档位：
        // 写到 `app_paths::log_dir()`（Android `files/HiFiShifter/logs`、
        // iOS `Library/Logs/HiFiShifter`）。
        let args: Vec<String> = std::env::args().collect();
        init_logging(choice_from_args(&args), &args);
        if crate::platform::is_mobile() {
            // 一行"我们在哪"：Android 上 `adb logcat` 直接能看到这行，是排查
            // "应用起来了但界面白屏"时唯一的第一手线索。
            log::info!(
                "mobile surface: os={:?} log_dir={:?}",
                crate::platform::os(),
                crate::logging::log_dir()
            );
        }
    });
}

/// 移动端日志的对外出口：把当前日志目录交给"导出诊断包"。
///
/// 桌面上"打开日志文件夹"是有意义的（资源管理器能定位）；移动端没有这个动作，
/// 用户能做的只有"把日志导出去"。这里只回答目录，怎么交出去是 `share` 的事。
pub fn log_dir_for_export() -> Option<std::path::PathBuf> {
    crate::logging::log_dir().map(std::path::Path::to_path_buf)
}

/// 桌面的 `--log-file` 语义在移动端不成立；保留一个显式入口给未来的设置项。
pub fn set_choice(choice: LogFileChoice) {
    // 只有首次初始化生效（`Once`）：运行期改档位走 `log_settings` 那条既有通道。
    let _ = choice;
}
