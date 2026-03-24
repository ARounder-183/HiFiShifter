//! 缓存管理命令。
//!
//! 提供前端可调用的 `clear_cache` 命令，清除合成结果缓存并返回释放的字节数。

use crate::state::AppState;
use tauri::State;

/// 清除合成结果缓存（内存 + 磁盘）。
///
/// # 行为
/// 1. 清空 [`SynthClipCache`] 内存条目
/// 2. 删除 `<exe_dir>/cache/synth/` 下所有缓存文件
/// 3. 不删除 `vslib_tmp/`（运行中可能仍有临时文件）
///
/// # 返回值
/// 估算释放的字节数（内存 PCM 数据 + 磁盘文件字节数之和）。
pub(super) fn clear_cache(_state: State<'_, AppState>) -> Result<u64, String> {
    let mut total_bytes: u64 = 0;

    // 1. 清空内存缓存，估算释放字节数
    {
        let mut cache = crate::synth_clip_cache::global_synth_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        total_bytes += cache.clear_and_estimate_bytes();
    }

    // 2. 删除磁盘缓存目录 <exe_dir>/cache/synth/
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let synth_cache_dir = exe_dir.join("cache").join("synth");
            if synth_cache_dir.is_dir() {
                match std::fs::read_dir(&synth_cache_dir) {
                    Ok(entries) => {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_file() {
                                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                                if std::fs::remove_file(&path).is_ok() {
                                    total_bytes += size;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("clear_cache: cannot read synth cache dir: {}", e);
                    }
                }
            }
        }
    }

    Ok(total_bytes)
}
