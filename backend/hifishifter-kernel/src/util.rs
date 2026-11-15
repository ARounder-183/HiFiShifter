//! 与领域无关的通用小工具。
//!
//! 【为什么把原子替换放在这里】`config.rs` 的 `app_config.json` 与
//! `log_settings.rs` 的 `log_settings.json` 都要在 Windows 上做**原子替换**；
//! 两份实现会各自漂移（尤其是 `MoveFileExW` 的替换标志与 rename 回退）。
//! 日志策略在 UI 之前就要读写，不能反过来依赖 `config`，所以提升到 `util`。

pub fn beat_to_sec(beat: f64, bpm: f64) -> f64 {
    let bpm = if bpm.is_finite() && bpm > 0.0 {
        bpm
    } else {
        120.0
    };
    beat * 60.0 / bpm
}

pub fn clamp11(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

pub fn quantize_i64(x: f64, scale: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    (x * scale).round() as i64
}

pub fn quantize_u32(x: f64, scale: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    let v = (x * scale).round();
    if v <= 0.0 {
        0
    } else if v > (u32::MAX as f64) {
        u32::MAX
    } else {
        v as u32
    }
}

/// 把临时文件原子地替换为目标文件。
///
/// 【为什么不能直接 `fs::rename`】Windows 上 rename 拒绝覆盖已存在的目标，因此
/// 原先的写法必须先 `remove_file` 再 rename —— 那中间的窗口里**目标文件是不存在
/// 的**。两个进程同时保存（独立 App 与 REAPER 里的插件现在共用配置目录）时，
/// 恰好落在窗口里的那次读取会拿不到任何设置。删除也不原子：删掉之后 rename
/// 失败（磁盘满、被杀进程）就只剩 `.bak` 能救。
///
/// `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` 是同一卷内的原子替换；
/// 非 Windows 平台的 `fs::rename` 本身就是原子替换。
pub fn replace_file_atomically(
    tmp: &std::path::Path,
    target: &std::path::Path,
) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows::core::PCWSTR;
        use windows::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_REPLACE_EXISTING};

        let wide = |path: &std::path::Path| -> Vec<u16> {
            path.as_os_str().encode_wide().chain(Some(0)).collect()
        };
        let tmp_wide = wide(tmp);
        let target_wide = wide(target);
        // SAFETY: 两个缓冲区都是 NUL 结尾的宽字符串，且在本调用期间存活。
        let moved = unsafe {
            MoveFileExW(
                PCWSTR(tmp_wide.as_ptr()),
                PCWSTR(target_wide.as_ptr()),
                MOVEFILE_REPLACE_EXISTING,
            )
        };
        if moved.is_ok() {
            return Ok(());
        }
        // 目标不存在时 MOVEFILE_REPLACE_EXISTING 也可用，但仍回退一次 rename：
        // 某些网络/可移动卷上的 MoveFileEx 会失败，而那里的 rename 通常可行。
        std::fs::rename(tmp, target)
    }
    #[cfg(not(windows))]
    {
        std::fs::rename(tmp, target)
    }
}

#[cfg(test)]
mod atomic_replace_tests {
    use super::replace_file_atomically;

    #[test]
    fn replacing_an_existing_file_keeps_it_readable() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-atomic-replace-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("settings.json");
        let tmp = dir.join("settings.json.tmp");
        std::fs::write(&target, "old").unwrap();
        std::fs::write(&tmp, "new").unwrap();
        replace_file_atomically(&tmp, &target).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        assert!(!tmp.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
