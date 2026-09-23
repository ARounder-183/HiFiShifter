//! 记事本附件存储：图片与 HiFiShifter 剪贴板载荷。
//!
//! ## 为什么需要附件而不是把字节塞进 Markdown
//!
//! 记事本的正文是 Markdown 字符串，每次编辑都要经 IPC 写回后端并登记为
//! 撤销步。若把图片或剪贴板载荷的字节以 base64 内嵌进正文，则**每次按键
//! 都要搬运兆级文本**，打字会被 IPC 拖垮。因此正文只保存轻量引用
//! （`hifi-asset://<id>.<ext>` 与 ```hifi-clip 围栏），字节落在附件目录里，
//! 由本模块统一管理。
//!
//! ## 落点
//!
//! - 工程已落盘：`<工程文件全名>-assets/`（与 `-UNDO` 伴生文件的命名同构，
//!   见 `commands/undo_history_file.rs`），随工程一起搬迁；
//! - 工程尚未落盘：`%TEMP%/hifishifter/notebook_staging/`，首次保存时整体
//!   迁入旁挂目录。
//!
//! ## 生命周期
//!
//! 附件**只增不删**：会话内删除引用只是把登记项标记为 `orphaned`，真正的
//! 文件清理发生在保存时（`prune_orphans`）。这样撤销/重做把引用恢复回来时
//! 附件依然在，不会出现"撤销后图片变成空白"。

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// 工程旁挂附件目录后缀，与 `-UNDO` 同构。
pub const ASSET_DIR_SUFFIX: &str = "-assets";

/// 附件种类。决定前端用什么 UI 呈现，也决定读取时给什么 MIME 兜底。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotebookAssetKind {
    /// 图片（Markdown 里是 `![alt](hifi-asset://<id>.<ext>)`）。
    Image,
    /// HiFiShifter 剪贴板载荷的原始字节（Markdown 里是 ```hifi-clip 围栏）。
    ClipPayload,
}

impl Default for NotebookAssetKind {
    fn default() -> Self {
        NotebookAssetKind::Image
    }
}

/// 一条附件登记项。字节在磁盘上，这里只存元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct NotebookAsset {
    #[serde(default)]
    pub kind: NotebookAssetKind,
    /// 文件扩展名（不含点）。同时是磁盘文件名的后缀。
    pub ext: String,
    /// MIME 类型，读取时直接回给前端。
    #[serde(default)]
    pub mime: String,
    #[serde(default)]
    pub byte_len: u64,
    #[serde(default)]
    pub created_at_ms: u64,
    /// 引用已消失，等待保存时清理。会话内不硬删，见模块头注释。
    #[serde(default)]
    pub orphaned: bool,
    /// 自由元数据：
    /// - Image：`{ width, height, originalName }`
    /// - ClipPayload：`{ clipKind, title, clipCount, trackCount, durationSec, sourceProject, preview }`
    #[serde(default)]
    pub meta: serde_json::Value,
}

/// 附件 id 允许的字符集：只允许字母数字与 `-` `_`。
///
/// id 由前端生成（内容哈希），但要经 IPC 到达这里并参与路径拼接 ——
/// 必须在此收口，否则 `../` 之类能逃出附件目录。
pub fn sanitize_asset_id(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("notebook_asset_id_empty".to_string());
    }
    if trimmed.len() > 128 {
        return Err("notebook_asset_id_too_long".to_string());
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!("notebook_asset_id_invalid: {trimmed}"));
    }
    Ok(trimmed.to_string())
}

/// 扩展名白名单（字符集层面的收口，避免路径拼接被注入）。
pub fn sanitize_ext(raw: &str) -> String {
    let cleaned: String = raw
        .trim()
        .trim_start_matches('.')
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(16)
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        "bin".to_string()
    } else {
        cleaned
    }
}

/// 工程旁挂附件目录：`<工程文件全名>-assets/`。
pub fn asset_dir_for_project(project_path: &Path) -> PathBuf {
    let file_name = project_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project");
    project_path.with_file_name(format!("{file_name}{ASSET_DIR_SUFFIX}"))
}

/// 未落盘工程的附件暂存目录。
pub fn staging_dir() -> Result<PathBuf, String> {
    let dir = crate::temp_manager::hifishifter_temp_dir()?.join("notebook_staging");
    fs::create_dir_all(&dir).map_err(|e| format!("创建记事本暂存目录失败: {e}"))?;
    Ok(dir)
}

/// 确保目录存在。
pub fn ensure_dir(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("创建附件目录 {:?} 失败: {}", dir, e))
}

/// 写入一条附件的字节（覆盖同名）。
pub fn write_asset_bytes(dir: &Path, id: &str, ext: &str, bytes: &[u8]) -> Result<PathBuf, String> {
    let id = sanitize_asset_id(id)?;
    let ext = sanitize_ext(ext);
    ensure_dir(dir)?;
    let path = dir.join(format!("{id}.{ext}"));
    fs::write(&path, bytes).map_err(|e| format!("写入附件 {:?} 失败: {}", path, e))?;
    Ok(path)
}

/// 读取一条附件的字节。
pub fn read_asset_bytes(dir: &Path, id: &str, ext: &str) -> Result<Vec<u8>, String> {
    let id = sanitize_asset_id(id)?;
    let ext = sanitize_ext(ext);
    let path = dir.join(format!("{id}.{ext}"));
    fs::read(&path).map_err(|e| format!("读取附件 {:?} 失败: {}", path, e))
}

/// 删除一条附件的文件（不存在视为成功）。
pub fn remove_asset_file(dir: &Path, id: &str, ext: &str) -> Result<(), String> {
    let id = sanitize_asset_id(id)?;
    let ext = sanitize_ext(ext);
    let path = dir.join(format!("{id}.{ext}"));
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("删除附件 {:?} 失败: {}", path, e)),
    }
}

/// 在目录中按 id 定位附件文件，返回 `(路径, 扩展名)`。
///
/// 登记表里的 ext 理论上就是权威，但工程文件可能被手工改动或跨版本，
/// 因此读取时以磁盘实际内容为准兜底。
pub fn locate_asset_file(dir: &Path, id: &str) -> Option<(PathBuf, String)> {
    let id = sanitize_asset_id(id).ok()?;
    let entries = fs::read_dir(dir).ok()?;
    let prefix = format!("{id}.");
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(ext) = name.strip_prefix(&prefix) {
            if entry.path().is_file() {
                return Some((entry.path(), ext.to_string()));
            }
        }
    }
    None
}

/// 把 `from` 目录中的全部附件文件迁入 `to`（覆盖同名），用于首次保存时
/// 把暂存目录里的附件搬进工程旁挂目录。
pub fn migrate_dir(from: &Path, to: &Path) -> Result<usize, String> {
    if from == to {
        return Ok(0);
    }
    if !from.is_dir() {
        return Ok(0);
    }
    ensure_dir(to)?;
    let mut moved = 0usize;
    for entry in fs::read_dir(from)
        .map_err(|e| format!("读取暂存目录 {:?} 失败: {}", from, e))?
        .flatten()
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name() else { continue };
        let target = to.join(name);
        // 先删目标再改名：Windows 上 rename 不覆盖已存在文件。
        let _ = fs::remove_file(&target);
        match fs::rename(&path, &target) {
            Ok(()) => moved += 1,
            Err(_) => {
                // 跨卷时 rename 会失败，退回"复制 + 删除"。
                if fs::copy(&path, &target).is_ok() {
                    let _ = fs::remove_file(&path);
                    moved += 1;
                }
            }
        }
    }
    Ok(moved)
}

/// 清空暂存目录（新建/打开工程时调用，避免上一份未落盘工程的附件残留）。
pub fn clear_staging_dir() {
    if let Ok(dir) = staging_dir() {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

/// 按"仍被引用的 id 集合"清理目录，返回删除数量。
///
/// 与登记表的 `orphaned` 标记双管齐下：登记表负责语义（谁还被引用），
/// 目录清理负责兜底（登记表丢了但文件还在的残渣）。
pub fn prune_dir(dir: &Path, keep: &HashSet<String>) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let id = name.split('.').next().unwrap_or_default().to_string();
        if keep.contains(&id) {
            continue;
        }
        if fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// 登记表的别名，避免调用点到处写完整泛型。
pub type NotebookAssetMap = BTreeMap<String, NotebookAsset>;

/// 正文里的一处 `hifi-asset://` 引用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetRef {
    /// 引用在正文中的字节区间（用于拼接替换）。
    pub start: usize,
    pub end: usize,
    pub id: String,
    /// 引用里显式写出的扩展名（可缺省，读取时以登记表为准）。
    pub ext: Option<String>,
    /// `#w=` 片段携带的显示宽度。
    pub width: Option<u32>,
}

/// 扫描正文中全部 `hifi-asset://` 引用。
///
/// 语法（全部可选部分都容错）：
///
/// ```text
/// hifi-asset://<id>[.<ext>][#w=<px>]
/// ```
///
/// 手写扫描而非正则：id 与扩展名的字符集都是封闭的（见
/// `sanitize_asset_id` / `sanitize_ext`），逐字符推进比正则更易读，也不会
/// 因正文里的正则元字符而退化。
pub fn scan_asset_refs(content: &str) -> Vec<AssetRef> {
    const PREFIX: &str = "hifi-asset://";
    let bytes = content.as_bytes();
    let mut refs = Vec::new();
    let mut cursor = 0usize;

    while let Some(rel) = content[cursor..].find(PREFIX) {
        let start = cursor + rel;
        let mut pos = start + PREFIX.len();

        let id_start = pos;
        while pos < bytes.len() && is_id_byte(bytes[pos]) {
            pos += 1;
        }
        if pos == id_start {
            cursor = pos.max(start + PREFIX.len());
            continue;
        }
        let id = content[id_start..pos].to_string();

        let mut ext = None;
        if pos < bytes.len() && bytes[pos] == b'.' {
            let ext_start = pos + 1;
            let mut ext_end = ext_start;
            while ext_end < bytes.len() && bytes[ext_end].is_ascii_alphanumeric() {
                ext_end += 1;
            }
            if ext_end > ext_start {
                ext = Some(content[ext_start..ext_end].to_ascii_lowercase());
                pos = ext_end;
            }
        }

        let mut width = None;
        if content[pos..].starts_with("#w=") {
            let digits_start = pos + 3;
            let mut digits_end = digits_start;
            while digits_end < bytes.len() && bytes[digits_end].is_ascii_digit() {
                digits_end += 1;
            }
            if digits_end > digits_start {
                width = content[digits_start..digits_end].parse::<u32>().ok();
                pos = digits_end;
            }
        }

        refs.push(AssetRef {
            start,
            end: pos,
            id,
            ext,
            width,
        });
        cursor = pos;
    }

    refs
}

fn is_id_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

/// 从 Markdown 正文中抽出被引用的附件 id。
///
/// 覆盖两种引用形式：
/// - 图片：`hifi-asset://<id>[.<ext>]`（见 `scan_asset_refs`）
/// - 剪贴板暂存块：```hifi-clip 围栏里的 `id: <id>`
pub fn referenced_asset_ids(markdown: &str) -> HashSet<String> {
    let mut ids: HashSet<String> = scan_asset_refs(markdown).into_iter().map(|r| r.id).collect();

    for line in markdown.lines() {
        let trimmed = line.trim();
        let Some(value) = trimmed.strip_prefix("id:") else {
            continue;
        };
        let value = value.trim();
        if !value.is_empty() && sanitize_asset_id(value).is_ok() {
            ids.insert(value.to_string());
        }
    }

    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_rejects_traversal() {
        assert!(sanitize_asset_id("../etc/passwd").is_err());
        assert!(sanitize_asset_id("a/b").is_err());
        assert!(sanitize_asset_id("").is_err());
        assert_eq!(sanitize_asset_id("abc-123_x").unwrap(), "abc-123_x");
    }

    #[test]
    fn sanitize_ext_strips_dots_and_junk() {
        assert_eq!(sanitize_ext(".PNG"), "png");
        assert_eq!(sanitize_ext("we b.p"), "webp");
        assert_eq!(sanitize_ext(""), "bin");
    }

    #[test]
    fn referenced_ids_cover_images_and_clip_blocks() {
        let md = "正文\n\n![图](hifi-asset://abc123.webp#w=640)\n\n```hifi-clip\nid: 7c1e9a4b2d\nkind: clips\n```\n";
        let ids = referenced_asset_ids(md);
        assert!(ids.contains("abc123"), "image id missing: {ids:?}");
        assert!(ids.contains("7c1e9a4b2d"), "clip id missing: {ids:?}");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn referenced_ids_ignore_plain_text() {
        let ids = referenced_asset_ids("hifi-asset 不是引用；id: 也不是\n");
        assert!(ids.is_empty(), "unexpected: {ids:?}");
    }

    #[test]
    fn scan_refs_reads_ext_and_width() {
        let refs = scan_asset_refs("前 ![图](hifi-asset://abc123.webp#w=640) 后");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].id, "abc123");
        assert_eq!(refs[0].ext.as_deref(), Some("webp"));
        assert_eq!(refs[0].width, Some(640));
        // 区间必须精确覆盖引用本身，调用方靠它做拼接替换。
        assert_eq!(&"前 ![图](hifi-asset://abc123.webp#w=640) 后"[refs[0].start..refs[0].end], "hifi-asset://abc123.webp#w=640");
    }

    #[test]
    fn scan_refs_tolerates_partial_forms() {
        let refs = scan_asset_refs("a hifi-asset://onlyid b hifi-asset://x.png#w= c");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, "onlyid");
        assert_eq!(refs[0].ext, None);
        assert_eq!(refs[0].width, None);
        assert_eq!(refs[1].id, "x");
        assert_eq!(refs[1].ext.as_deref(), Some("png"));
        assert_eq!(refs[1].width, None);
    }

    #[test]
    fn scan_refs_ignores_bare_prefix() {
        assert!(scan_asset_refs("hifi-asset:// 后面没有 id").is_empty());
        assert!(scan_asset_refs("hifi-asset://../escape").is_empty());
    }

    #[test]
    fn asset_dir_mirrors_undo_sidecar_naming() {
        let dir = asset_dir_for_project(Path::new("/tmp/My Song.hshp"));
        assert!(dir.to_string_lossy().ends_with("My Song.hshp-assets"));
    }
}
