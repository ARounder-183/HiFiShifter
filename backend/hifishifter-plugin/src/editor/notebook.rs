//! 插件侧记事本持久化：笔记正文 + 附件字节。
//!
//! ## 为什么必须有
//!
//! 独立 App 把笔记与附件都写进工程文件（`.hshp`）。插件**没有**工程文件，而
//! Notebook 面板在插件里完全可达可编辑 —— 不落盘就等于"用户写的每一句话、贴的
//! 每一张图都会在关窗后消失"。此前这些命令全部落到 `Command unavailable`，
//! 前端又把失败吞掉（`NotebookPanel` 的 `.catch(() => {})`），所以症状是
//! **面板看起来能用，其实是空的**。
//!
//! ## 存在哪里
//!
//! `config_location::local_data_subdir("plugin-notebook")` —— 与插件自己的 PCM
//! 临时目录同一命名空间，不与独立 App 的工程文件混在一起。
//!
//! ## 一个必须说清楚的语义差异
//!
//! 这份存储是**进程级、全局一份**：同一台机器上所有 HiFiShifter 实例共用它。
//! 插件拿不到宿主的工程文件路径（没有绑定相关 API），因此无法把笔记挂到"某个工程"
//! 上；而按实例临时 id 建目录会在每次 REAPER 重启后换一个目录 —— 那是**静默丢
//! 数据**，比"共用一份"糟得多。共用至少可预期：写进去的东西下次还在。
//!
//! ## 格式
//!
//! `notes.md` 存正文，`assets.json` 存 `NotebookAssetMap`（含 base64 字节）。
//! 两者都是**整份读写** —— 记事本数据量小（附件另有 32MB 上限），增量写入只会
//! 带来损坏风险，换不来什么。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use hifishifter_kernel::notebook_assets::{
    prune_unreferenced, referenced_asset_ids, sanitize_asset_id, sanitize_ext, NotebookAsset,
    NotebookAssetKind, NotebookAssetMap,
};

/// 单条附件的解码后字节上限（与 App 侧同值）。
const MAX_ASSET_BYTES: usize = 32 * 1024 * 1024;
/// `notebook_read_file_base64` 的单文件读取上限（与 App 侧同值）。
const READ_FILE_MAX_BYTES: u64 = 64 * 1024 * 1024;

const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

/// 记事本存储。目录显式传入，便于测试用临时目录（进程级单例见 [`store`]）。
pub(super) struct NotebookStore {
    dir: PathBuf,
}

impl NotebookStore {
    pub(super) fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn notes_path(&self) -> PathBuf {
        self.dir.join("notes.md")
    }

    fn assets_path(&self) -> PathBuf {
        self.dir.join("assets.json")
    }

    // ─────────────────────────── 笔记正文 ───────────────────────────

    /// 读回持久化的笔记正文；文件不存在/读不出来时返回空串。
    ///
    /// 【为什么不报错】笔记读不出来不该让整个面板打不开 —— 空正文 + 可写，比一句
    /// 内部错误更有用（用户至少能把内容重新贴进去）。
    pub(super) fn notes(&self) -> String {
        std::fs::read_to_string(self.notes_path()).unwrap_or_default()
    }

    pub(super) fn set_notes(&self, markdown: &str) -> Result<(), String> {
        let path = self.notes_path();
        let temp = path.with_extension("md.tmp");
        write_atomically(&temp, &path, markdown.as_bytes(), "notebook_notes")
    }

    // ─────────────────────────── 附件登记表 ───────────────────────────

    fn load_assets(&self) -> NotebookAssetMap {
        let Ok(text) = std::fs::read_to_string(self.assets_path()) else {
            return NotebookAssetMap::new();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    fn save_assets(&self, map: &NotebookAssetMap) -> Result<(), String> {
        let text =
            serde_json::to_string(map).map_err(|e| format!("notebook_assets_encode: {e}"))?;
        let path = self.assets_path();
        let temp = path.with_extension("json.tmp");
        write_atomically(&temp, &path, text.as_bytes(), "notebook_assets")
    }

    /// 一条附件（字节已解码）—— 导出/另存为要的就是它。
    pub(super) fn stored_asset(&self, id: &str) -> Option<StoredAsset> {
        let asset = self.load_assets().get(id).cloned()?;
        let bytes = B64.decode(asset.data.as_bytes()).ok()?;
        Some(StoredAsset {
            bytes,
            ext: sanitize_ext(&asset.ext),
        })
    }

    pub(super) fn put_asset(
        &self,
        asset_id: &str,
        kind: &str,
        ext: &str,
        mime: Option<&str>,
        data_base64: &str,
        meta: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        let id = sanitize_asset_id(asset_id)?;
        let bytes = B64
            .decode(data_base64.as_bytes())
            .map_err(|e| format!("notebook_asset_base64_invalid: {e}"))?;
        if bytes.len() > MAX_ASSET_BYTES {
            return Err("notebook_asset_too_large".into());
        }
        let mut map = self.load_assets();
        map.insert(
            id.clone(),
            NotebookAsset {
                kind: kind_of(kind),
                ext: sanitize_ext(ext),
                mime: mime.unwrap_or_default().to_string(),
                byte_len: bytes.len() as u64,
                created_at_ms: now_ms(),
                orphaned: false,
                meta: meta.unwrap_or(serde_json::Value::Null),
                data: data_base64.to_string(),
            },
        );
        self.save_assets(&map)?;
        Ok(serde_json::json!({"ok": true, "assetId": id, "byteLen": bytes.len()}))
    }

    pub(super) fn read_asset(&self, asset_id: &str) -> serde_json::Value {
        let Ok(id) = sanitize_asset_id(asset_id) else {
            return serde_json::json!({"ok": false, "error": "notebook_asset_id_invalid"});
        };
        match self.load_assets().get(&id) {
            Some(asset) if !asset.data.is_empty() => serde_json::json!({
                "ok": true,
                "mime": asset.mime,
                "base64": asset.data,
            }),
            _ => serde_json::json!({"ok": true, "missing": true}),
        }
    }

    pub(super) fn list_assets(&self) -> serde_json::Value {
        let assets = self
            .load_assets()
            .iter()
            .map(|(id, asset)| {
                serde_json::json!({
                    "id": id,
                    "kind": asset.kind,
                    "ext": asset.ext,
                    "mime": asset.mime,
                    "byteLen": asset.byte_len,
                    "createdAtMs": asset.created_at_ms,
                    "meta": asset.meta,
                    // 登记项里是否真的有字节：手工编辑过文件时为 false，前端据此显示
                    // "附件缺失"而不是一张破图。
                    "hasData": !asset.data.is_empty(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({"ok": true, "assets": assets})
    }

    pub(super) fn remove_asset(&self, asset_id: &str) -> serde_json::Value {
        let Ok(id) = sanitize_asset_id(asset_id) else {
            return serde_json::json!({"ok": true, "removed": false});
        };
        let mut map = self.load_assets();
        let removed = map.remove(&id).is_some();
        if removed {
            if let Err(error) = self.save_assets(&map) {
                crate::log_line(&format!("[notebook] asset removal not persisted: {error}"));
            }
        }
        serde_json::json!({"ok": true, "removed": removed})
    }

    /// 按正文引用清理孤儿附件。
    ///
    /// 【为什么以**持久化的**正文为准】prune 在保存前调用，此时正文可能还没写完；
    /// 但更糟的是拿一个"调用方传来的"正文去删别人的附件。这里一律读自己存的那份 ——
    /// 单一事实来源，删错不可逆。
    pub(super) fn prune_assets(&self) -> serde_json::Value {
        let markdown = self.notes();
        let keep = referenced_asset_ids(&markdown);
        let mut map = self.load_assets();
        let removed = prune_unreferenced(&mut map, &keep);
        if removed > 0 {
            if let Err(error) = self.save_assets(&map) {
                crate::log_line(&format!("[notebook] prune not persisted: {error}"));
            }
        }
        serde_json::json!({"ok": true, "removed": removed})
    }

    /// 全部附件（导出时用来把 `hifi-asset://` 换成 data URI）。
    pub(super) fn asset_data_map(&self) -> BTreeMap<String, NotebookAsset> {
        self.load_assets()
    }
}

/// 一条附件（字节已解码）。
pub(super) struct StoredAsset {
    pub bytes: Vec<u8>,
    pub ext: String,
}

/// 导出正文：把 `hifi-asset://` 引用改写成内嵌 data URI。
///
/// 【为什么内嵌】导出物是**单个自包含文件** —— 与工程本身的存储模型一致：内容
/// 跟着文件走，拷到哪里都能看，不生成任何旁挂目录（也就不存在"另存为要复制附件"
/// 那套生命周期）。
///
/// 返回 `(改写后的正文, 缺失的附件 id)`：缺字节的引用**原样留着**并把 id 报出去，
/// 让调用方告诉用户"这几张图没找到"，而不是静默产出坏图。
pub(super) fn render_export(content: &str, store: &NotebookStore) -> (String, Vec<String>) {
    let assets = store.asset_data_map();
    let mut out = content.to_string();
    let mut missing = Vec::new();
    // 从后往前替换，避免前面的替换让后面的下标失效。
    for asset_ref in hifishifter_kernel::notebook_assets::scan_asset_refs(content)
        .into_iter()
        .rev()
    {
        let Some(asset) = assets.get(&asset_ref.id) else {
            missing.push(asset_ref.id.clone());
            continue;
        };
        if asset.data.is_empty() {
            missing.push(asset_ref.id.clone());
            continue;
        }
        let mime = if asset.mime.is_empty() {
            mime_for_ext(&asset.ext)
        } else {
            asset.mime.clone()
        };
        out.replace_range(
            asset_ref.start..asset_ref.end,
            &format!("data:{mime};base64,{}", asset.data),
        );
    }
    (out, missing)
}

/// 进程级单例：存储目录只解析一次（与 `settings_store` 同一取向）。
pub(super) fn store() -> NotebookStore {
    NotebookStore::new(hifishifter_kernel::config_location::local_data_subdir(
        "plugin-notebook",
    ))
}

/// 需要系统保存对话框的两条命令（导出正文 / 附件另存为）。
///
/// 【为什么放在这里而不是 dispatch】对话框要 HWND，而 HWND 只有 UI 线程持有；
/// `commands.rs` 的 dispatch 跑在 actor 线程上。逻辑本体（改写引用、取字节）仍
/// 留在本模块，这里只负责"问用户要一个路径，然后写文件"。
#[cfg(windows)]
pub(super) fn export_with_dialog(
    hwnd: windows::Win32::Foundation::HWND,
    command: &str,
    args: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match command {
        "notebook_export_document" => {
            let ext = sanitize_ext(args["extension"].as_str().unwrap_or("md"));
            let content = args["content"].as_str().unwrap_or_default();
            let base = sanitize_export_file_name(
                args["suggestedName"].as_str().unwrap_or_default(),
                "untitled",
            );
            let Some(path) =
                super::browser_files::pick_save_path(hwnd, &format!("{base}.{ext}"), &ext)?
            else {
                return Ok(serde_json::json!({"ok": true, "canceled": true}));
            };
            // 用户可能把扩展名改掉；改掉就拒绝，免得产出名不副实的文件。
            if extension_of(&path) != ext {
                return Ok(
                    serde_json::json!({"ok": false, "error": "notebook_export_extension_mismatch"}),
                );
            }
            let (rendered, missing) = render_export(content, &store());
            write_output(&path, rendered.as_bytes())?;
            Ok(serde_json::json!({
                "ok": true,
                "canceled": false,
                "path": path.display().to_string(),
                "missingAssets": missing,
            }))
        }
        "notebook_save_asset_as" => {
            let id = args["assetId"].as_str().unwrap_or_default();
            let Some(asset) = store().stored_asset(id) else {
                return Ok(serde_json::json!({"ok": false, "error": "notebook_asset_not_found"}));
            };
            // 名字来自前端（不可信）与 asset_id（也可能被手工编辑过）；统一过一遍清洗器。
            let fallback = format!("{id}.{}", asset.ext);
            let suggested = args["suggestedName"]
                .as_str()
                .filter(|name| !name.trim().is_empty())
                .unwrap_or(&fallback);
            let name = sanitize_export_file_name(suggested, "asset");
            let Some(path) = super::browser_files::pick_save_path(hwnd, &name, &asset.ext)? else {
                return Ok(serde_json::json!({"ok": true, "canceled": true}));
            };
            write_output(&path, &asset.bytes)?;
            Ok(serde_json::json!({
                "ok": true,
                "canceled": false,
                "path": path.display().to_string(),
            }))
        }
        _ => Err(format!("unknown notebook dialog command: {command}")),
    }
}

#[cfg(windows)]
fn write_output(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("notebook_export_mkdir_failed: {e}"))?;
        }
    }
    std::fs::write(path, bytes).map_err(|e| format!("notebook_export_write_failed: {e}"))
}

/// 先写临时文件再改名：进程在写入中途被杀时，用户不会得到一个半截的文件。
fn write_atomically(temp: &Path, path: &Path, bytes: &[u8], tag: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{tag}_mkdir_failed: {e}"))?;
    }
    std::fs::write(temp, bytes).map_err(|e| format!("{tag}_write_failed: {e}"))?;
    std::fs::rename(temp, path).map_err(|e| format!("{tag}_commit_failed: {e}"))
}

// ─────────────────────────── 其它（无状态） ───────────────────────────

/// 读一个磁盘文件为 base64（拖入图片走这条）。
pub(super) fn read_file_base64(path: &str, max_bytes: Option<u64>) -> serde_json::Value {
    let limit = max_bytes
        .unwrap_or(READ_FILE_MAX_BYTES)
        .min(READ_FILE_MAX_BYTES);
    let source = Path::new(path);
    let size = match std::fs::metadata(source) {
        Ok(meta) => meta.len(),
        Err(error) => return serde_json::json!({"ok": false, "error": error.to_string()}),
    };
    if size > limit {
        return serde_json::json!({"ok": false, "error": "notebook_read_file_too_large"});
    }
    match std::fs::read(source) {
        Ok(bytes) => {
            let ext = extension_of(source);
            serde_json::json!({
                "ok": true,
                "mime": mime_for_ext(&ext),
                "ext": ext,
                "byteLen": bytes.len(),
                "base64": B64.encode(&bytes),
            })
        }
        Err(error) => serde_json::json!({"ok": false, "error": error.to_string()}),
    }
}

/// 读系统剪贴板里的位图（CF_DIB）。前端把它包成 BMP 再走统一图片流水线。
pub(super) fn read_clipboard_image() -> serde_json::Value {
    match hifishifter_clipboard::system_clipboard::read_bitmap_dib() {
        Ok(Some(bitmap)) => serde_json::json!({
            "ok": true,
            "available": true,
            "width": bitmap.width,
            "height": bitmap.height,
            "bitsPerPixel": bitmap.bits_per_pixel,
            "base64": B64.encode(&bitmap.bytes),
        }),
        Ok(None) => serde_json::json!({"ok": true, "available": false}),
        Err(error) => serde_json::json!({"ok": false, "error": error}),
    }
}

/// `seal_project_notes_history`：独立 App 用它关闭"连续输入合并成一步撤销"的窗口。
///
/// 【插件里为什么是 no-op】插件的笔记写在 `session.project`（`ProjectState`）里，
/// 而它**不参与**插件的撤销栈（`session.history` 只记时间轴与参数）。既然没有合并
/// 窗口可关，这里如实回报成功，而不是报一个用户看不懂的"不支持"。
pub(super) fn seal_notes_history() -> serde_json::Value {
    serde_json::json!({"ok": true})
}

// ─────────────────────────── 小工具 ───────────────────────────

fn kind_of(raw: &str) -> NotebookAssetKind {
    match raw {
        "clip_payload" => NotebookAssetKind::ClipPayload,
        _ => NotebookAssetKind::Image,
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .unwrap_or_default()
}

pub(super) fn mime_for_ext(ext: &str) -> String {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "avif" => "image/avif",
        "svg" => "image/svg+xml",
        "hsf" => "application/x-hifishifter-fragment",
        "hsp" => "application/json",
        _ => "application/octet-stream",
    }
    .to_string()
}

/// 清洗导出/另存为对话框的默认文件名。
///
/// 名字来自前端或正文引用，最终交给系统保存对话框：路径分隔符能让对话框预导航
/// 到任意目录，Windows 保留字符在多数文件系统上非法。与 `sanitize_asset_id`
/// 同风格 —— 删非法字符、去首尾空白；清完为空时回落到调用方给的通用名。
pub(super) fn sanitize_export_file_name(raw: &str, fallback: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> NotebookStore {
        let dir = std::env::temp_dir().join(format!("hfs-notebook-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        NotebookStore::new(dir)
    }

    /// 附件写入 → 列出 → 读回 → 删除，全程经过磁盘。
    ///
    /// 【为什么要走真文件】这一层的全部价值就是"关窗后还在"。只测内存映射等于
    /// 没测到持久化，而症状恰恰是"面板看起来能用、重开是空的"。
    #[test]
    fn assets_survive_a_round_trip_through_disk() {
        let store = scratch("roundtrip");
        let payload = B64.encode(b"hello notebook");
        store
            .put_asset("asset-1", "image", "png", Some("image/png"), &payload, None)
            .unwrap();

        let listed = store.list_assets();
        assert_eq!(listed["assets"].as_array().unwrap().len(), 1);
        assert_eq!(listed["assets"][0]["id"], "asset-1");
        assert_eq!(listed["assets"][0]["hasData"], true);
        assert_eq!(listed["assets"][0]["byteLen"], 14);

        let read = store.read_asset("asset-1");
        assert_eq!(read["ok"], true);
        assert_eq!(read["base64"], payload);

        // 真的落盘了：换一个实例读同一个目录，内容还在。
        let reopened = NotebookStore::new(store.dir.clone());
        assert_eq!(reopened.read_asset("asset-1")["base64"], payload);

        assert_eq!(store.remove_asset("asset-1")["removed"], true);
        assert_eq!(store.read_asset("asset-1")["missing"], true);
    }

    /// 非法 id 必须被挡在存储之外（正文里的 `../` 不能变成路径）。
    #[test]
    fn invalid_asset_ids_never_reach_the_store() {
        let store = scratch("badid");
        assert!(store
            .put_asset("../escape", "image", "png", None, "AA==", None)
            .is_err());
        assert_eq!(store.read_asset("../escape")["ok"], false);
        assert_eq!(store.remove_asset("../escape")["removed"], false);
    }

    /// 正文里不再引用的附件在 prune 时被清掉，仍被引用的保留。
    #[test]
    fn pruning_follows_the_persisted_markdown() {
        let store = scratch("prune");
        store
            .put_asset("kept", "image", "png", None, "AA==", None)
            .unwrap();
        store
            .put_asset("dropped", "image", "png", None, "AA==", None)
            .unwrap();
        store.set_notes("![x](hifi-asset://kept.png)").unwrap();

        assert_eq!(store.prune_assets()["removed"], 1);
        assert!(store.asset_data_map().contains_key("kept"));
        assert!(!store.asset_data_map().contains_key("dropped"));
        assert_eq!(store.notes(), "![x](hifi-asset://kept.png)");
    }

    /// 笔记正文跨实例持久化。
    #[test]
    fn notes_survive_a_reopen() {
        let store = scratch("notes");
        assert_eq!(store.notes(), "");
        store.set_notes("# 标题\n正文").unwrap();
        assert_eq!(
            NotebookStore::new(store.dir.clone()).notes(),
            "# 标题\n正文"
        );
    }

    /// 导出文件名清洗：路径分隔符与保留字符被删掉，空名回落到通用名。
    #[test]
    fn export_names_cannot_navigate_the_save_dialog() {
        assert_eq!(sanitize_export_file_name("../evil", "untitled"), "..evil");
        assert_eq!(sanitize_export_file_name("a:b*c?", "untitled"), "abc");
        assert_eq!(sanitize_export_file_name("   ", "untitled"), "untitled");
    }
}
