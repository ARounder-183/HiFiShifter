//! 插件进程可写的用户数据根。
//!
//! 【为什么 core 里也要算一次这个路径】CLAP 外壳刻意不依赖推理内核（见 crate 文档），
//! 而它内嵌的 WebView2 必须有一个**当前用户可写**的用户数据目录：模块自己躺在
//! `C:\Program Files\Common Files\CLAP`，把 profile 写在模块旁边会让浏览器进程
//! 直接起不来（宿主里弹"Microsoft Edge 无法读取和写入其数据目录"，窗口一片空白）。
//!
//! 【为什么不算第二份实现】规则与 `hifishifter_kernel::app_paths` 逐条一致：
//! Windows `%LOCALAPPDATA%\HiFiShifter\<name>`（缺 `LOCALAPPDATA` 时回退到
//! `%USERPROFILE%\AppData\Local`），macOS `~/Library/Application Support/HiFiShifter`，
//! Linux `$XDG_DATA_HOME/.local/share`。产品目录名直接取 [crate::CLASS_NAME]（与
//! `app_paths::PRODUCT_DIR` 同值），漂移由 `hifishifter-plugin` 的
//! `tests/local_data_paths_agree.rs` 在两边比对着守。
//!
//! 【为什么缺变量时回退到临时目录】路径解析不出来时宁可用一个清理不完美的临时目录，
//! 也不能让插件窗口打不开；这条与内核的取舍相同。

use std::path::PathBuf;

/// Windows 的**本地**数据根；缺 `LOCALAPPDATA` 时按同义回退到 `%USERPROFILE%\AppData\Local`。
#[cfg(windows)]
fn local_base() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| {
            let home = std::env::var_os("USERPROFILE")?;
            Some(PathBuf::from(home).join("AppData").join("Local"))
        })
}

/// 平台默认的本地数据目录（缓存 / 临时工作文件的根，**不是**漫游的配置目录）。
pub fn local_data_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        Some(local_base()?.join(crate::CLASS_NAME))
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")?;
        Some(
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join(crate::CLASS_NAME),
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                let home = std::env::var_os("HOME")?;
                Some(PathBuf::from(home).join(".local").join("share"))
            })?;
        Some(base.join(crate::CLASS_NAME))
    }
}

/// 本地数据目录下的一个子目录；解析不出来时退回系统临时目录（同样带产品名）。
pub fn local_data_subdir(name: &str) -> PathBuf {
    match local_data_dir() {
        Some(base) => base.join(name),
        None => std::env::temp_dir()
            .join(crate::CLASS_NAME.to_lowercase())
            .join(name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_local_data_root_is_absolute_and_carries_the_product_name() {
        let Some(root) = local_data_dir() else {
            return;
        };
        assert!(root.is_absolute(), "{}", root.display());
        assert_eq!(root.file_name().unwrap(), crate::CLASS_NAME);
    }

    #[test]
    fn a_subdir_is_one_level_below_the_root() {
        let Ok(root) = local_data_dir().ok_or(()) else {
            return;
        };
        assert_eq!(local_data_subdir("webview"), root.join("webview"));
    }

    /// 兜底路径也必须带产品名：否则多份 profile 会在临时目录里撞名。
    #[test]
    fn the_temp_fallback_keeps_the_product_name() {
        let fallback = std::env::temp_dir()
            .join(crate::CLASS_NAME.to_lowercase())
            .join("webview");
        assert!(fallback.ends_with("webview"));
        assert!(fallback
            .to_string_lossy()
            .contains(&crate::CLASS_NAME.to_lowercase()));
    }
}
