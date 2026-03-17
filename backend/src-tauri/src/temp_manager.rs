//! 临时文件统一管理模块。
//!
//! 所有 HiFiShifter 产生的临时文件统一放在 `%TEMP%/hifishifter/` 下，
//! 应用启动时自动清理上次遗留的临时文件。

use std::fs;
use std::path::{Path, PathBuf};

/// 返回 HiFiShifter 统一临时目录：`%TEMP%/hifishifter/`。
///
/// 若目录不存在会自动创建。
pub fn hifishifter_temp_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join("hifishifter");
    fs::create_dir_all(&dir).map_err(|e| format!("创建临时目录失败: {e}"))?;
    Ok(dir)
}

/// 在创建新的 synth 临时 WAV 前，删除旧的临时文件。
///
/// `old_path` 是 `runtime.synthesized_wav_path` 中保存的上一次路径。
pub fn remove_old_synth_temp(old_path: Option<&str>) {
    if let Some(p) = old_path {
        let path = Path::new(p);
        if path.exists() {
            match fs::remove_file(path) {
                Ok(()) => {
                    eprintln!("[temp_manager] 已删除旧 synth 临时文件: {}", path.display());
                }
                Err(e) => {
                    eprintln!(
                        "[temp_manager] 删除旧 synth 临时文件失败: {} — {}",
                        path.display(),
                        e
                    );
                }
            }
        }
    }
}

/// 应用启动时清理遗留的临时文件。
///
/// 清理范围：
/// - `%TEMP%/hifishifter/synth_*.wav`（合成临时文件）
/// - `%TEMP%/hifishifter/import_*.*`（导入临时文件）
/// - `%TEMP%/hs_vslib_*.wav`（vslib 崩溃残留）
///
/// 此函数不会阻塞，内部 spawn 后台线程执行。
pub fn cleanup_stale_temp_files() {
    std::thread::spawn(|| {
        let mut total_removed = 0u64;
        let mut total_bytes = 0u64;

        // 1. 清理 %TEMP%/hifishifter/ 下的 synth_*.wav 和 import_*.*
        if let Ok(hs_dir) = hifishifter_temp_dir() {
            let (removed, bytes) = cleanup_dir_by_prefix(&hs_dir, &["synth_", "import_"]);
            total_removed += removed;
            total_bytes += bytes;
        }

        // 2. 清理 %TEMP% 下的 hs_vslib_*.wav（进程崩溃残留）
        let sys_temp = std::env::temp_dir();
        let (removed, bytes) = cleanup_dir_by_prefix(&sys_temp, &["hs_vslib_"]);
        total_removed += removed;
        total_bytes += bytes;

        if total_removed > 0 {
            eprintln!(
                "[temp_manager] 启动清理完成: 删除 {} 个遗留临时文件, 释放 {:.1} KB",
                total_removed,
                total_bytes as f64 / 1024.0,
            );
        }
    });
}

/// 扫描目录，删除文件名以指定前缀开头的文件。
///
/// 返回 `(删除文件数, 释放字节数)`。
fn cleanup_dir_by_prefix(dir: &Path, prefixes: &[&str]) -> (u64, u64) {
    let mut removed = 0u64;
    let mut bytes = 0u64;

    let entries = match fs::read_dir(dir) {
        Ok(v) => v,
        Err(_) => return (0, 0),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let matches = prefixes.iter().any(|prefix| file_name.starts_with(prefix));
        if !matches {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            bytes += meta.len();
        }
        match fs::remove_file(&path) {
            Ok(()) => {
                removed += 1;
            }
            Err(e) => {
                eprintln!("[temp_manager] 清理失败: {} — {}", path.display(), e);
            }
        }
    }

    (removed, bytes)
}
