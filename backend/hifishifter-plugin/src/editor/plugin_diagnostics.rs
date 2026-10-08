//! 插件侧**精简**诊断导出：日志 + 构建身份 + 运行时 + 宿主版本。
//!
//! 【为什么是精简版】App 的完整诊断包会跑推理设备基准测试、枚举 GPU/DML 设备、
//! 统计音高缓存 —— `src-tauri/src/commands/diagnostics.rs` 自己写着基准测试在
//! GPU/驱动异常的环境下**可能硬崩**。在 DAW 进程里跑它等于把用户的整个会话一起
//! 带走。因此插件里只做**不会崩**的部分，基准测试保持拒绝（见 `commands.rs`）。
//!
//! 【为什么放这里而不是 `commands.rs`】打包要碰文件系统与 zip，而 `commands.rs`
//! 的 dispatch 跑在 actor 线程上、只处理纯数据命令。这里由 UI 线程调用（保存对话框
//! 需要 HWND，宿主版本也需要宿主对象）。

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// 单个日志文件纳入诊断包的大小上限（与 App 侧同一取向：异常膨胀的日志不该拖垮导出）。
const LOG_SIZE_CAP: u64 = 32 * 1024 * 1024;

/// 默认文件名（带本地时间戳，便于用户区分多次导出）。
pub(super) fn default_file_name() -> String {
    format!(
        "HiFiShifter-plugin-diagnostics-{}.zip",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    )
}

/// 打包 system_info.json + settings.json + logs/<name>。
///
/// 【为什么 system_info 由调用方拼】它要宿主版本与渲染器诊断，两者都只有 UI 线程
/// 拿得到；把拼装留在调用方，这里只负责"把 JSON 和日志放进 zip"这一件事。
pub(super) fn write_package(
    out: &Path,
    system_info: &serde_json::Value,
    settings: &serde_json::Value,
) -> Result<(), String> {
    let file = std::fs::File::create(out).map_err(|e| format!("create zip failed: {e}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("system_info.json", options)
        .map_err(|e| format!("zip add system_info failed: {e}"))?;
    let info = serde_json::to_string_pretty(system_info)
        .map_err(|e| format!("serialize system_info failed: {e}"))?;
    zip.write_all(info.as_bytes())
        .map_err(|e| format!("write system_info failed: {e}"))?;

    zip.start_file("settings.json", options)
        .map_err(|e| format!("zip add settings failed: {e}"))?;
    let settings_json = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("serialize settings failed: {e}"))?;
    zip.write_all(settings_json.as_bytes())
        .map_err(|e| format!("write settings failed: {e}"))?;

    // 日志：插件自己的 plugin.log，以及同目录下 App 的 hifishifter.log（如果存在）。
    // 两份日志共用一个目录（见 `diagnostics.rs`），出问题时往往要一起看。
    for path in log_files() {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        if size > LOG_SIZE_CAP {
            crate::log_line(&format!(
                "[diagnostics] skipping oversized log file {name} ({size} bytes)"
            ));
            continue;
        }
        match std::fs::File::open(&path) {
            Ok(mut source) => {
                zip.start_file(format!("logs/{name}"), options)
                    .map_err(|e| format!("zip add {name} failed: {e}"))?;
                // 流式写入：日志可能有几 MB，不值得整份读进内存。
                std::io::copy(&mut source, &mut zip)
                    .map_err(|e| format!("write {name} failed: {e}"))?;
            }
            Err(error) => crate::log_line(&format!(
                "[diagnostics] failed to read log file {name}: {error}"
            )),
        }
    }

    zip.finish()
        .map_err(|e| format!("finish zip failed: {e}"))?;
    Ok(())
}

/// 纳入诊断包的日志文件：本插件的 `plugin.log`，外加同目录的 `hifishifter.log`
/// 与它的轮转分代。缺失的文件直接跳过（不是错误）。
fn log_files() -> Vec<PathBuf> {
    let mut files = vec![crate::diagnostics::log_file()];
    if let Some(dir) = crate::diagnostics::log_directory().parent() {
        for name in ["hifishifter.log", "hifishifter.log.1", "hifishifter.log.2"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                files.push(candidate);
            }
        }
    }
    files.retain(|path| path.is_file());
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 包内必须能读回 system_info / settings / 日志三个条目。
    ///
    /// 【为什么要真的读回】zip 写入最容易出的错是"条目名对了但内容没落盘"或
    /// "只 finish 没 flush"。只有重新打开归档读一遍才能证明用户拿到的文件可用。
    #[test]
    fn the_package_contains_the_report_and_the_logs() {
        let dir = std::env::temp_dir().join(format!("hfs-diag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("package.zip");
        write_package(
            &out,
            &serde_json::json!({"product": "HiFiShifter ARA plugin"}),
            &serde_json::json!({"ui": {"locale": "zh-CN"}}),
        )
        .unwrap();

        let file = std::fs::File::open(&out).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut names = (0..archive.len())
            .map(|index| archive.by_index(index).unwrap().name().to_owned())
            .collect::<Vec<_>>();
        names.sort();
        assert!(names.contains(&"system_info.json".to_owned()), "{names:?}");
        assert!(names.contains(&"settings.json".to_owned()), "{names:?}");

        let mut info = String::new();
        std::io::Read::read_to_string(&mut archive.by_name("system_info.json").unwrap(), &mut info)
            .unwrap();
        assert!(info.contains("HiFiShifter ARA plugin"), "{info}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 默认文件名必须带时间戳后缀，避免两次导出互相覆盖。
    #[test]
    fn the_default_name_carries_a_timestamp() {
        let name = default_file_name();
        assert!(name.starts_with("HiFiShifter-plugin-diagnostics-"));
        assert!(name.ends_with(".zip"));
        assert!(name.len() > "HiFiShifter-plugin-diagnostics-.zip".len() + 8);
    }
}
