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
/// 【为什么移动端不能直接用 `document_dir`】导出物的唯一价值是**离开这个应用**。
/// `document_dir` 在 Android 上是 `getExternalFilesDir(DIRECTORY_DOCUMENTS)`
/// —— `Android/data/<pkg>/files/Documents`。从 Android 11 起，系统"文件"应用
/// **明确隐藏 `Android/data`**，于是导出的音频在用户眼里根本不存在：导出"成功"了，
/// 文件却谁也打不开。这就是"导出音频不工作"的真实成因。
///
/// 因此：
/// - **Android**：由原生桥给出 `Android/media/<pkg>/HiFiShifter`（[`super::native::export_dir`]）。
///   同一个应用、同样零权限，但**不隐藏** —— 文件管理器与 MTP（USB 连电脑）都看得见。
/// - **iOS**：`document_dir` = `Documents/`，配合 `UIFileSharingEnabled` 与
///   `LSSupportsOpeningDocumentsInPlace` 后直接出现在
///   "文件 → 我的 iPhone → HiFiShifter"。
/// - 原生桥不可用（极少数无外部存储的设备 / 旧构建）时退回 `document_dir`：
///   至少文件还在，应用内的"保存到其它位置"仍能把它交出去。
pub fn export_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if crate::platform::is_mobile() {
        if let Some(dir) = super::native::export_dir(app) {
            return Ok(dir);
        }
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

/// 工程库的落点（移动端）。
///
/// 【为什么工程要有一个「库」而不是继续用 SAF 选路径】移动端的「保存」与
/// 「打开」此前是**不对称**的：保存写死到一个固定目录，打开却要用户自己走
/// SAF 找一个 `.hshp`。用户看不到那个固定目录，于是「我保存的工程去哪了」
/// 与「我该打开哪个文件」成了两个都无法回答的问题。
///
/// 库把它变成**一个列表**：保存进库、打开时看库。SAF 仍然保留，但只用于
/// 「从设备导入…」（把外部 `.hshp` 收编进库）与「导出 / 分享」。
///
/// 【为什么放在 `export_dir` 之下】那是移动端唯一**用户看得见**的应用目录
/// （见 [`export_dir`] 的说明）。放在 `app_data_dir` 里，工程文件会与导出物
/// 分居两处，而其中一处谁也打不开。
pub fn projects_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = export_dir(app)?.join("projects");
    ensure_dir(&dir)?;
    Ok(dir)
}

/// "用户能看到的文档目录"。
///
/// 【为什么需要这一层】全应用有若干处默认落点（导出对话框的默认输出目录、
/// "另存为"的起始位置）走的是 `document_dir()`。桌面那是 `~/Documents` ——
/// 用户天天在用；Android 那是 `Android/data/<pkg>/files/Documents` ——
/// 系统"文件"应用从 Android 11 起就把它藏起来了。同一个调用在两端得到
/// "用户能找到"与"用户找不到"两种结果，而调用点看不出差别。
///
/// 移动端因此统一改指向 [`export_dir`]（Android/media，可见）。桌面上仍然是
/// 系统的文档目录 —— 行为一个字节都不变。
pub fn documents_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    if crate::platform::is_mobile() {
        return export_dir(app).ok();
    }
    app.path().document_dir().ok()
}

/// 建目录；已存在时是空操作。
pub fn ensure_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| format!("create {}: {error}", path.display()))
}

/// 导入文件的归类目标。
///
/// 【为什么要分两类而不是全丢进一个目录】移动端的文件浏览器只浏览 `media/`
/// （见 [`media_root`]）。把工程文件、MIDI、ReaScript 之类混进去，用户会在
/// "媒体"面板里看到一堆 `.hshp`，而真正的素材被挤到后面。分开之后，
/// "我在手机上导入的音频去哪了"只有一个答案。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportTarget {
    /// 音频 / 视频素材：进工作区（`media/`），文件浏览器可见。
    Media,
    /// 工程 / MIDI / 文本等文档：进 `imports/`，不进媒体面板。
    #[default]
    Document,
}

/// 导入文件的落点目录。
///
/// 桌面上返回 `None`：桌面走"引用原文件"，导入不产生副本（见模块头）。
pub fn import_root(app: &tauri::AppHandle, target: ImportTarget) -> Option<PathBuf> {
    if !crate::platform::is_mobile() {
        return None;
    }
    let base = app.path().app_data_dir().ok()?;
    Some(match target {
        ImportTarget::Media => base.join("media"),
        ImportTarget::Document => base.join("imports"),
    })
}

/// 把选择器物化出来的文件收进工作区，返回收编后的路径。
///
/// 【为什么先试 `rename`】原生桥把 SAF 的 `content://` 物化到 `cacheDir`，与
/// `app_data_dir` 同在 `/data/user/0/<pkg>/` 这一个文件系统上，因此 `rename`
/// 成功就意味着**零拷贝**。用户的素材动辄几百 MB，第二次拷贝既慢又让峰值磁盘
/// 占用翻倍。跨文件系统（极端情况）才退回拷贝。
///
/// 【为什么保留原文件名、冲突时加序号】用户是按文件名认素材的。哈希命名"更安全"，
/// 但会让文件浏览器里全是 `a3f9...wav` —— 那不是可用性上的取舍，是直接把功能做没了。
/// 冲突处理用 `name (2).ext`，与系统文件管理器的惯例一致。
pub fn adopt_import(source: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
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
    if std::fs::rename(source, &candidate).is_ok() {
        return Ok(candidate);
    }
    std::fs::copy(source, &candidate)
        .map_err(|error| format!("copy {}: {error}", source.display()))?;
    // 缓存副本已经没用了；删不掉也不算失败（系统迟早会回收 cacheDir）。
    let _ = std::fs::remove_file(source);
    Ok(candidate)
}
