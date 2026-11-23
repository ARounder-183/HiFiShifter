//! 移动形态的目录布局。
//!
//! 【为什么不复用 `app_paths`】`app_paths` 回答的是"**设置**放在哪"（配置 / 缓存 /
//! 日志），而这里回答的是"**用户的素材**放在哪"。两者在桌面上恰好都落在
//! `%LOCALAPPDATA%\HiFiShifter` 附近，于是很容易被混为一谈 —— 但它们的生命周期
//! 完全不同：设置可以删、素材不能；移动端的沙箱也把两者分到了不同的系统目录。
//!
//! ## 移动端的存储模型（设计文档 §5.10）
//!
//! 桌面是"**引用**用户的文件"：工程里存的是绝对路径，素材留在原地。移动端没有这个
//! 选项 —— 系统选择器给出的是 content URI / 安全作用域 URL，应用拿不到长期有效的
//! 路径。因此移动端一律"**导入即拷贝**"：字节进 `media_dir`，工程只引用沙箱内的副本。
//!
//! 这条差异不是实现细节，它改变了产品的语义：移动端的文件浏览器是**工作区浏览器**
//! （只看得见本应用导入过的素材），而不是整机文件浏览器。

use std::path::{Path, PathBuf};

use tauri::Manager;

/// 用户素材的根目录（导入即拷贝的落点）。
///
/// 桌面上返回 `None`：桌面走"引用原文件"，没有工作区这个概念，凭空造一个空目录
/// 只会让用户困惑（"为什么多了一个文件夹"）。
pub fn media_root(app: &tauri::AppHandle) -> Option<PathBuf> {
    if !crate::platform::is_mobile() {
        return None;
    }
    let base = app.path().app_data_dir().ok()?;
    Some(base.join("media"))
}

/// 导出产物的落点。
///
/// 【为什么移动端是 `document_dir` 而不是 `app_data_dir`】导出物的唯一价值是
/// **离开这个应用**。`app_data_dir` 在 iOS 上是 `Library/Application Support`，
/// 用户在"文件"App 里看不到它 —— 导出的文件等于扔进了黑洞。
///
/// - iOS：`document_dir` = `Documents/`，配合 `UIFileSharingEnabled` 与
///   `LSSupportsOpeningDocumentsInPlace`（见 `tauri.ios.conf.json`）后直接出现在
///   "文件 → 我的 iPhone → HiFiShifter"。
/// - Android：`document_dir` = `getExternalFilesDir(DIRECTORY_DOCUMENTS)`
///   （见 tauri 的 `PathPlugin.kt`），经"文件"App 的 `Android/data/<pkg>/files`
///   可达，且**不需要任何存储权限**。
pub fn export_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if crate::platform::is_mobile() {
        let base = app
            .path()
            .document_dir()
            .or_else(|_| app.path().app_data_dir())
            .map_err(|error| format!("export dir unresolved: {error}"))?;
        return Ok(base.join("HiFiShifter"));
    }
    // 桌面没有"默认导出目录"这个概念：用户每次都在对话框里选。返回配置目录下的
    // 一个子目录只是给"未选路径"的兜底场景（自动备份等）一个确定落点。
    let base = hifishifter_kernel::app_paths::local_data_dir()
        .ok_or_else(|| "local data dir unresolved".to_string())?;
    Ok(base.join("exports"))
}

/// 建目录；已存在时是空操作。
pub fn ensure_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| format!("create {}: {error}", path.display()))
}

/// 把用户选中的文件拷进工作区，返回副本路径。
///
/// 【为什么保留原文件名、冲突时加序号】用户是按文件名认素材的。哈希命名"更安全"，
/// 但会让文件浏览器里全是 `a3f9...wav` —— 那不是可用性上的取舍，是直接把功能做没了。
/// 冲突处理用 `name (2).ext`，与系统文件管理器的惯例一致。
pub fn import_copy(source: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    ensure_dir(dest_dir)?;
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("media");
    let extension = source.extension().and_then(|value| value.to_str());
    let mut candidate = dest_dir.join(source.file_name().unwrap_or_default());
    let mut index = 2u32;
    while candidate.exists() {
        let name = match extension {
            Some(extension) => format!("{stem} ({index}).{extension}"),
            None => format!("{stem} ({index})"),
        };
        candidate = dest_dir.join(name);
        index += 1;
        if index > 10_000 {
            return Err("import copy: too many name collisions".to_string());
        }
    }
    std::fs::copy(source, &candidate)
        .map_err(|error| format!("copy {}: {error}", source.display()))?;
    Ok(candidate)
}
