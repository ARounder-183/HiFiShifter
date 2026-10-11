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
#[derive(Default)]
pub enum NotebookAssetKind {
    /// 图片（Markdown 里是 `![alt](hifi-asset://<id>.<ext>)`）。
    #[default]
    Image,
    /// HiFiShifter 剪贴板载荷的原始字节（Markdown 里是 ```hifi-clip 围栏）。
    ClipPayload,
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

// ─── 磁盘读取通道（App 与插件共用的安全收口） ────────────────────────────────
//
// `notebook_read_file_base64` 是 IPC 可达的读盘通道。收口必须**只有一份**：
// App 侧先加固了（扩展名白名单 + 不可抬高的上限 + is_file），插件侧的复制体
// 没跟上，漏出一条"前端被注入后可在 DAW 进程内读任意文件"的通道。既然
// `sanitize_asset_id` / `sanitize_ext` 早已在内核，读取通道也下沉到这里。

/// `read_image_file` 允许读取的扩展名白名单。
///
/// 该命令按任意路径读盘并回传 base64，不加收口就是一条任意文件读通道：
/// 收到 IPC 后只放行图片格式（音频导入有专用命令，`hsf` 载荷只来自
/// 剪贴板而非磁盘），即使前端被攻破也读不出工程外的任意文件。
pub const READABLE_IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "avif"];

/// `read_image_file` 的服务端硬上限（字节）。生效上限 =
/// `min(调用方值或缺省, READ_FILE_MAX_BYTES)` —— 调用方传更大的值抬不高上限。
pub const READ_FILE_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// `read_image_file` 的拒绝理由。`Display` 输出**稳定错误码**（与 App 侧
/// `commands/notebook.rs` 对前端的既有约定一致：前端按码做 i18n 与归类）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadImageFileError {
    /// 目标不是常规文件（detail: 路径）。
    NotAFile(String),
    /// 扩展名不在 [`READABLE_IMAGE_EXTS`]（detail: 扩展名）。
    UnsupportedExt(String),
    /// 超过大小上限（`byte_len` 是实测文件大小）。
    TooLarge {
        /// 实测文件字节数（供响应体附加 `byteLen` 字段）。
        byte_len: u64,
    },
    /// 磁盘 IO 失败（detail: io 错误串）。
    Io(String),
}

impl std::fmt::Display for ReadImageFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAFile(path) => write!(f, "notebook_not_a_file: {path}"),
            Self::UnsupportedExt(ext) => write!(f, "notebook_unsupported_image_ext: {ext}"),
            Self::TooLarge { byte_len } => write!(f, "notebook_file_too_large: {byte_len}"),
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

/// 一次成功读取：扩展名、MIME 兜底与原始字节（base64 编码留给调用方 ——
/// 两侧各自的编码器与响应组装已经存在，不必把 JSON 形状搬进内核）。
#[derive(Debug, Clone)]
pub struct ReadImageFile {
    /// 小写扩展名（白名单内）。
    pub ext: String,
    /// 按扩展名给出的 MIME 兜底（前端一般会显式传）。
    pub mime: &'static str,
    /// 文件原始字节。
    pub bytes: Vec<u8>,
}

/// 白名单内扩展名的 MIME 兜底。
pub fn image_mime_for_ext(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "avif" => "image/avif",
        _ => "application/octet-stream",
    }
}

/// 读一个磁盘图片文件（拖入图片的读盘通道；App 与插件共用此实现）。
///
/// 三条服务端收口缺一不可：
/// 1. **扩展名白名单**（[`READABLE_IMAGE_EXTS`]）——否则即任意文件读；
/// 2. **大小上限不可协商**：生效上限 = `min(调用方值或缺省, [`READ_FILE_MAX_BYTES`])`；
/// 3. **只读常规文件**（`is_file`）。
pub fn read_image_file(
    path: &str,
    max_bytes: Option<u64>,
) -> Result<ReadImageFile, ReadImageFileError> {
    let path = std::path::Path::new(path);
    if !path.is_file() {
        return Err(ReadImageFileError::NotAFile(path.display().to_string()));
    }
    let limit = max_bytes
        .unwrap_or(READ_FILE_MAX_BYTES)
        .min(READ_FILE_MAX_BYTES);
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if !READABLE_IMAGE_EXTS.contains(&ext.as_str()) {
        return Err(ReadImageFileError::UnsupportedExt(ext));
    }
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > limit => {
            return Err(ReadImageFileError::TooLarge {
                byte_len: meta.len(),
            })
        }
        Ok(_) => {}
        Err(error) => return Err(ReadImageFileError::Io(error.to_string())),
    }
    let bytes = std::fs::read(path).map_err(|e| ReadImageFileError::Io(e.to_string()))?;
    Ok(ReadImageFile {
        // mime 先算：`ext` 还要在下面移进字段。
        mime: image_mime_for_ext(&ext),
        ext,
        bytes,
    })
}

/// 清洗导出/另存为对话框的默认文件名（App 与插件共用此实现）。
///
/// 名字来自前端或正文引用，最终交给系统保存对话框：路径分隔符能让对话框预导航
/// 到任意目录，Windows 保留字符在多数文件系统上非法。与 `sanitize_asset_id`
/// 同风格 —— 删非法字符、去首尾空白；清完为空时回落到调用方给的通用名。
pub fn sanitize_export_file_name(raw: &str, fallback: &str) -> String {
    let cleaned: String = raw
        .chars()
        .filter(|c| {
            !matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') && !c.is_control()
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
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
    let mut ids: HashSet<String> = scan_asset_refs(markdown)
        .into_iter()
        .map(|r| r.id)
        .collect();

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

    /// 读盘通道的三条收口逐条验证（App 与插件共用这份实现，测一处即测两处；
    /// 两侧各自的包装只应做错误码到响应 JSON 的映射，见各自的包装测试）。
    #[test]
    fn read_image_file_enforces_whitelist_size_and_file_checks() {
        let dir = std::env::temp_dir().join(format!("hfs-kernel-readimage-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("x.png");
        std::fs::write(&png, b"png-bytes").unwrap();
        let png = png.display().to_string();

        let read = read_image_file(&png, None).expect("png passes the whitelist");
        assert_eq!(read.ext, "png");
        assert_eq!(read.mime, "image/png");
        assert_eq!(read.bytes, b"png-bytes");

        // 白名单外一律拒绝（这是"任意文件读"的收口本身）。
        let txt = dir.join("x.txt");
        std::fs::write(&txt, b"secret").unwrap();
        match read_image_file(&txt.display().to_string(), None) {
            Err(ReadImageFileError::UnsupportedExt(ext)) => assert_eq!(ext, "txt"),
            other => panic!("expected UnsupportedExt, got {other:?}"),
        }

        // 非 is_file（目录、不存在的路径）拒绝。
        match read_image_file(&dir.display().to_string(), None) {
            Err(ReadImageFileError::NotAFile(_)) => {}
            other => panic!("expected NotAFile, got {other:?}"),
        }

        // 大小上限不可协商：调用方给的值只在 min 里生效，抬不高上限本身。
        match read_image_file(&png, Some(2)) {
            Err(ReadImageFileError::TooLarge { byte_len }) => assert_eq!(byte_len, 9),
            other => panic!("expected TooLarge, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 文件名清洗：分隔符与 Windows 保留字符全删，清空回落到通用名。
    #[test]
    fn export_file_names_never_escape_the_dialog() {
        assert_eq!(sanitize_export_file_name("../evil", "untitled"), "..evil");
        assert_eq!(sanitize_export_file_name("a:b*c?", "untitled"), "abc");
        assert_eq!(sanitize_export_file_name("   ", "untitled"), "untitled");
        assert_eq!(sanitize_export_file_name("line
break", "untitled"), "linebreak");
    }
}
