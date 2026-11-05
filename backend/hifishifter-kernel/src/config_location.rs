//! 用户配置目录的解析：独立 App 与 ARA 插件必须落在**同一个目录**。
//!
//! 【这个模块现在只做转发】路径的**唯一**计算点是 [crate::app_paths]：那里同时
//! 定义设置、本地数据、缓存、日志与共享库五个根，并给出三平台一张表。本模块保留
//! 既有公开函数名（`default_config_dir` / `resolve` / `local_data_subdir` …），因此
//! 调用点一个都不用改 —— 但**没有第二份实现**可以漂移。
//!
//! 【为什么这件事重要】App 通过 Tauri 拿路径，而插件没有 Tauri（依赖树守卫明令禁止）。
//! 插件此前干脆不落盘 —— 设置纯内存，换个 REAPER 工程或重启一次就回到出厂默认。
//! 让插件自己"猜"一个路径同样不行：猜错就等于两套设置，用户在 App 里调好的外观与
//! 快捷键在插件里全部失效。路径只有这一处计算，两个宿主都调用它。
//!
//! 各平台结果见 [crate::app_paths] 的模块注释（Windows 是
//! `%APPDATA%\HiFiShifter`，macOS 是 `~/Library/Application Support/HiFiShifter`，
//! Linux 是 `$XDG_CONFIG_HOME/HiFiShifter`）。

use std::path::{Path, PathBuf};

pub use crate::app_paths::CONFIG_DIR_ENV;

/// 平台默认的配置目录；无法确定时返回 `None`（不 panic —— 设置存不下不该让
/// 宿主进程起不来）。
pub fn default_config_dir() -> Option<PathBuf> {
    crate::app_paths::config_dir()
}

/// 解析配置目录：显式参数 > 环境变量 > 平台默认。
///
/// `explicit` 供调用方传入自己已经算好的位置（历史上 App 从 Tauri 拿），
/// 优先级最高；测试与排障走环境变量。
pub fn resolve(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    crate::app_paths::config_dir()
}

/// 解析并确保目录存在（失败时返回原因，调用方决定是报错还是静默降级）。
pub fn resolve_and_create(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let dir = resolve(explicit).ok_or("user config directory is unavailable on this platform")?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("create config dir {}: {e}", dir.display()))?;
    Ok(dir)
}

/// 平台默认的**本地**数据目录（缓存 / 临时工作文件），与漫游的配置目录分开。
///
/// 见 [crate::app_paths::local_data_dir]：Windows 是 `%LOCALAPPDATA%\HiFiShifter`，
/// macOS 是 `~/Library/Application Support/HiFiShifter`，Linux 是
/// `$XDG_DATA_HOME/HiFiShifter`。
pub fn default_local_data_dir() -> Option<PathBuf> {
    crate::app_paths::local_data_dir()
}

/// 本地数据目录下的一个子目录；解析不出来时退回系统临时目录。
///
/// 【为什么保留临时目录兜底】缺 `LOCALAPPDATA` 时宁可用临时目录也不能让插件起不来 ——
/// 一个跑得起来但清理不完美的会话，好过一个打不开的窗口。
pub fn local_data_subdir(name: &str) -> PathBuf {
    crate::app_paths::local_data_subdir(name)
}

/// 平台默认的**全机共享**数据目录（只有 Windows 有，其余为 `None`）。
///
/// 理由见 [crate::app_paths::shared_data_dir] 与 [crate::model_store] 的模块文档。
pub fn default_shared_data_dir() -> Option<PathBuf> {
    crate::app_paths::shared_data_dir()
}

/// 共享数据目录下的一个子目录；非 Windows 或不可用时为 `None`。
pub fn shared_data_subdir(name: &str) -> Option<PathBuf> {
    crate::app_paths::shared_data_subdir(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `HIFISHIFTER_CONFIG_DIR` 是**进程级**全局量，而 cargo 默认并行跑用例：一个用例
    /// `set_var`、另一个同时把它设成空值、第一个再去读，就会落到平台默认 —— 表现为
    /// 随机失败的"环境变量没生效"。改动它的用例都先取这把锁（**全 crate 共用一把**，
    /// 见 `crate::test_env`）。
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env::guard()
    }

    /// 显式参数优先于环境变量：App 传自己的路径时必须用它。
    #[test]
    fn explicit_path_wins_over_the_environment() {
        let _guard = env_guard();
        let explicit = PathBuf::from("C:/explicit/config");
        std::env::set_var(CONFIG_DIR_ENV, "C:/from-env/config");
        let resolved = resolve(Some(&explicit));
        std::env::remove_var(CONFIG_DIR_ENV);
        assert_eq!(resolved, Some(explicit));
    }

    /// 环境变量优先于平台默认：测试与排障依赖这一点。
    #[test]
    fn environment_overrides_the_platform_default() {
        let _guard = env_guard();
        let override_dir = std::env::temp_dir().join("hfs-config-location-test");
        std::env::set_var(CONFIG_DIR_ENV, &override_dir);
        let resolved = resolve(None);
        std::env::remove_var(CONFIG_DIR_ENV);
        assert_eq!(resolved, Some(override_dir));
    }

    /// 空的环境变量视为未设置，不能解析成一个空路径。
    #[test]
    fn a_blank_environment_value_falls_through() {
        let _guard = env_guard();
        std::env::set_var(CONFIG_DIR_ENV, "   ");
        let resolved = resolve(None);
        std::env::remove_var(CONFIG_DIR_ENV);
        assert_ne!(resolved, Some(PathBuf::new()));
        assert_eq!(resolved, crate::app_paths::config_dir());
    }

    /// 平台默认路径必须以产品目录名收尾 —— 这正是两个宿主共用的目录；算错一层
    /// 就会与对方分家，而症状只是"设置不共享"。
    #[test]
    fn the_platform_default_names_the_product() {
        let Some(dir) = default_config_dir() else {
            // 环境缺少 HOME/APPDATA 时无从判定，跳过（CI 上两者都在）。
            return;
        };
        assert!(
            dir.ends_with(crate::app_paths::PRODUCT_DIR),
            "配置目录 {dir:?} 不再以 {} 收尾，插件会与 App 分家",
            crate::app_paths::PRODUCT_DIR
        );
        // 历史的一层 identifier 目录必须**不再出现**：它是 v2 布局的重点。
        assert!(
            !dir.to_string_lossy()
                .contains(crate::app_paths::LEGACY_IDENTIFIER),
            "配置目录 {dir:?} 仍带着历史 identifier 一层"
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

    /// App 与插件解析到的必须是**同一个**目录 —— 这是本模块存在的全部理由。
    ///
    /// 【为什么这条不测两个 crate】两个宿主调用的就是同一个函数；能漂移的只有
    /// "谁算了第二遍"。这条把"转发"钉死：只要 config_location 还是一条转发，
    /// 两侧就不可能分家。
    #[test]
    fn the_app_and_the_plugin_resolve_the_same_directory() {
        assert_eq!(default_config_dir(), crate::app_paths::config_dir());
        assert_eq!(
            crate::logfile::log_dir(),
            crate::app_paths::log_dir(),
            "日志目录也必须由同一个根算出来（两个宿主写进同一目录）"
        );
    }
}
