//! 记事本附件登记表：图片与 HiFiShifter 剪贴板载荷。
//!
//! ## 字节内嵌在工程文件里
//!
//! 附件（图片、剪贴板载荷）的字节以 base64 直接存在工程文件的
//! `notebook_assets` 表里，**不落任何旁挂目录**：
//!
//! - 工程自包含：把 `.hshp` 单独拷走、发出去、放进压缩包，图片都跟着走，
//!   不会出现"文件在、图没了"；
//! - 没有旁挂目录、没有暂存目录、没有"另存为时要复制附件"这一整套生命周期；
//! - 定时备份天然自带附件（备份写的就是工程文件本身）。
//!
//! 代价是工程文件会变大。抵消手段在**写入前**：图片先按长边上限缩放、转成
//! WebP/JPEG，并按内容哈希去重（同一张图重复拖入只存一份）。
//!
//! base64 而不是 msgpack 原生字节：工程文件既可能是 `.hshp`（MessagePack）
//! 也可能是 `.json`，后者用 `Vec<u8>` 会被序列化成数字数组（体积爆炸）。
//! base64 在两种格式里都是紧凑字符串。
//!
//! ## 生命周期
//!
//! 附件**只增不删**：删除引用（删掉正文里的图）只是让它在保存时被判为
//! "未被引用"，真正的清理发生在保存时（`prune_unreferenced`）。这样撤销/
//! 重做把引用恢复回来时附件仍在，不会出现"撤销后图片变成空白"。

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

/// 附件种类。决定前端用什么 UI 呈现。
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

/// 一条附件：元数据 + 字节（base64）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct NotebookAsset {
    #[serde(default)]
    pub kind: NotebookAssetKind,
    /// 文件扩展名（不含点）。导出/另存为时用来拼文件名。
    pub ext: String,
    /// MIME 类型，读取时直接回给前端。
    #[serde(default)]
    pub mime: String,
    /// 解码后的字节数（不是 base64 长度），用于展示占用与上限判断。
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
    /// 字节本体（base64）。
    ///
    /// 带 `serde(default)`：登记项若因手工编辑或部分写入而缺这个字段，工程
    /// 仍能打开（该图显示为"附件缺失"占位），而不是整个文件反序列化失败。
    #[serde(default)]
    pub data: String,
}

/// 附件 id 允许的字符集：只允许字母数字与 `-` `_`。
///
/// id 由前端生成（内容哈希），但要经 IPC 到达这里并出现在正文里，因此必须
/// 在此收口 —— 否则正文里一个 `../` 就能让导出路径拼到目录外。
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

/// 扩展名白名单（字符集层面的收口，避免拼文件名时被注入）。
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
/// 手写扫描而非正则：id 与扩展名的字符集都是封闭的（见 `sanitize_asset_id`
/// / `sanitize_ext`），逐字符推进比正则更易读，也不会因正文里的正则元字符
/// 而退化。
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

/// 按"仍被引用"清理登记表，返回移除条数。
pub fn prune_unreferenced(map: &mut NotebookAssetMap, keep: &HashSet<String>) -> usize {
    let stale: Vec<String> = map
        .iter()
        .filter(|(id, asset)| asset.orphaned || !keep.contains(*id))
        .map(|(id, _)| id.clone())
        .collect();
    for id in &stale {
        map.remove(id);
    }
    stale.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(ext: &str) -> NotebookAsset {
        NotebookAsset {
            kind: NotebookAssetKind::Image,
            ext: ext.to_string(),
            mime: "image/png".to_string(),
            byte_len: 3,
            created_at_ms: 0,
            orphaned: false,
            meta: serde_json::Value::Null,
            data: "AAAA".to_string(),
        }
    }

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
        assert_eq!(
            &"前 ![图](hifi-asset://abc123.webp#w=640) 后"[refs[0].start..refs[0].end],
            "hifi-asset://abc123.webp#w=640"
        );
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
    fn prune_keeps_referenced_and_drops_the_rest() {
        let mut map = NotebookAssetMap::new();
        map.insert("keep1".into(), asset("png"));
        map.insert("keep2".into(), asset("webp"));
        map.insert("gone".into(), asset("png"));
        let mut orphaned = asset("png");
        orphaned.orphaned = true;
        map.insert("explicit".into(), orphaned);

        let keep: HashSet<String> = ["keep1", "keep2"].into_iter().map(String::from).collect();
        let removed = prune_unreferenced(&mut map, &keep);
        assert_eq!(removed, 2);
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("keep1"));
        assert!(map.contains_key("keep2"));
    }

    #[test]
    fn asset_round_trips_through_msgpack_with_embedded_bytes() {
        let mut map = NotebookAssetMap::new();
        map.insert("abc".into(), asset("png"));
        let bytes = rmp_serde::to_vec_named(&map).expect("serialize");
        let back: NotebookAssetMap = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(back["abc"].data, "AAAA");
        assert_eq!(back["abc"].ext, "png");
    }

    #[test]
    fn asset_without_data_field_still_deserializes() {
        // `data` 带 serde(default)：手工编辑或部分写入的工程也要能打开，
        // 只是那条附件显示为"内容缺失"，而不是整个工程反序列化失败。
        let json = r#"{"abc":{"kind":"image","ext":"png","mime":"image/png","byte_len":3}}"#;
        let map: NotebookAssetMap = serde_json::from_str(json).expect("deserialize partial shape");
        assert_eq!(map["abc"].data, "");
        assert_eq!(map["abc"].ext, "png");
    }

}
