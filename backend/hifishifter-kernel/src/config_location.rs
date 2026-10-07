//! 用户配置目录的解析：独立 App 与 ARA 插件必须落在**同一个目录**。
//!
//! 【为什么需要它】App 通过 Tauri 的 `app_config_dir()` 拿路径，而插件没有 Tauri
//! （依赖树守卫明令禁止）。插件此前干脆不落盘 —— 设置纯内存，换个 REAPER 工程或
//! 重启一次就回到出厂默认。让插件自己"猜"一个路径同样不行：猜错就等于两套设置，
//! 用户在 App 里调好的外观与快捷键在插件里全部失效。
//!
//! 因此路径只有这一处计算，两个宿主都调用它。各平台的结果与 Tauri 的
//! `app_config_dir()/HiFiShifter` 逐一对应：
//!
//! | 平台 | 路径 |
//! | --- | --- |
//! | Windows | `%APPDATA%\com.arounder.hifishifter\HiFiShifter` |
//! | macOS | `~/Library/Application Support/com.arounder.hifishifter/HiFiShifter` |
//! | Linux | `$XDG_CONFIG_HOME/com.arounder.hifishifter/HiFiShifter`（缺省 `~/.config`） |
//!
//! 【为什么放在 Roaming 而不是 Local】这是既有布局（App 一直写在这里），改位置
//! 等于让存量用户丢一次设置；插件加入不该有这种代价。

use std::path::{Path, PathBuf};

/// Tauri 的 `identifier`，也是本目录的一级名字。
const APP_IDENTIFIER: &str = "com.arounder.hifishifter";
/// 配置子目录名（与 App 历史布局一致）。
const CONFIG_SUBDIR: &str = "HiFiShifter";

/// 覆盖配置目录的环境变量。
///
/// 【为什么必须有】插件与内核的单元测试跑在开发机上，如果它们写真实的用户配置，
/// 一次 `cargo test` 就会改掉开发者自己的设置；并行测试之间也会互相踩。有了这个
/// 变量，每个测试指向自己的临时目录，落盘逻辑才可测。
pub const CONFIG_DIR_ENV: &str = "HIFISHIFTER_CONFIG_DIR";

/// 平台默认的配置目录；无法确定时返回 `None`（不 panic —— 设置存不下不该让
/// 宿主进程起不来）。
pub fn default_config_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let base = std::env::var_os("APPDATA")?;
        Some(PathBuf::from(base).join(APP_IDENTIFIER).join(CONFIG_SUBDIR))
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")?;
        Some(
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join(APP_IDENTIFIER)
                .join(CONFIG_SUBDIR),
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join(APP_IDENTIFIER).join(CONFIG_SUBDIR))
    }
}

/// 解析配置目录：显式参数 > 环境变量 > 平台默认。
///
/// `explicit` 供调用方传入自己已经算好的位置（历史上 App 从 Tauri 拿），
/// 优先级最高；测试与排障走环境变量。
pub fn resolve(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    if let Some(raw) = std::env::var_os(CONFIG_DIR_ENV) {
        let raw = raw.to_string_lossy().trim().to_owned();
        if !raw.is_empty() {
            return Some(PathBuf::from(raw));
        }
    }
    default_config_dir()
}

/// 解析并确保目录存在（失败时返回原因，调用方决定是报错还是静默降级）。
pub fn resolve_and_create(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let dir = resolve(explicit).ok_or("user config directory is unavailable on this platform")?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("create config dir {}: {e}", dir.display()))?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 显式参数优先于环境变量：App 传自己的路径时必须用它。
    #[test]
    fn explicit_path_wins_over_the_environment() {
        let explicit = PathBuf::from("C:/explicit/config");
        std::env::set_var(CONFIG_DIR_ENV, "C:/from-env/config");
        let resolved = resolve(Some(&explicit));
        std::env::remove_var(CONFIG_DIR_ENV);
        assert_eq!(resolved, Some(explicit));
    }

    /// 环境变量优先于平台默认：测试与排障依赖这一点。
    #[test]
    fn environment_overrides_the_platform_default() {
        let override_dir = std::env::temp_dir().join("hfs-config-location-test");
        std::env::set_var(CONFIG_DIR_ENV, &override_dir);
        let resolved = resolve(None);
        std::env::remove_var(CONFIG_DIR_ENV);
        assert_eq!(resolved, Some(override_dir));
    }

    /// 空的环境变量视为未设置，不能解析成一个空路径。
    #[test]
    fn a_blank_environment_value_falls_through() {
        std::env::set_var(CONFIG_DIR_ENV, "   ");
        let resolved = resolve(None);
        std::env::remove_var(CONFIG_DIR_ENV);
        assert_ne!(resolved, Some(PathBuf::new()));
        assert_eq!(resolved, default_config_dir());
    }

    /// 平台默认路径必须以 `<identifier>/HiFiShifter` 收尾 —— 这正是 App 一直
    /// 使用的目录；插件算错一层就会与 App 分家，而症状只是"设置不共享"。
    #[test]
    fn the_platform_default_matches_the_historical_app_layout() {
        let Some(dir) = default_config_dir() else {
            // 环境缺少 HOME/APPDATA 时无从判定，跳过（CI 上两者都在）。
            return;
        };
        assert!(
            dir.ends_with(Path::new(APP_IDENTIFIER).join(CONFIG_SUBDIR)),
            "配置目录 {dir:?} 不再以 {APP_IDENTIFIER}/{CONFIG_SUBDIR} 收尾，插件会与 App 分家"
        );
    }

    /// 目录不存在时 `resolve_and_create` 会创建它。
    #[test]
    fn resolve_and_create_creates_the_directory() {
        let dir = std::env::temp_dir().join(format!("hfs-config-create-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let created = resolve_and_create(Some(&dir)).expect("create config dir");
        assert!(created.is_dir());
        std::fs::remove_dir_all(&dir).ok();
    }
}
