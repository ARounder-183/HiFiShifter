//! 记事本后端：附件存储、剪贴板载荷暂存、文档导出。
//!
//! 前端只做"呈现与交互"，所有落到磁盘或系统剪贴板的动作都收敛在这里：
//!
//! - **附件**（图片 / 剪贴板载荷）：字节写进工程旁挂目录，登记表随工程文件
//!   持久化（见 `crate::notebook_assets` 的模块注释）。
//! - **剪贴板载荷暂存**：把 `application/x-hifishifter-object` 槽位里的原始
//!   字节原样取出来交给前端保存；恢复时再把同一份字节原样写回。
//!   **不做任何重新序列化** —— 载荷是 MessagePack，重新编码会随版本漂移，
//!   原样字节才能保证"暂存 → 恢复"是无损的。
//! - **导出**：把正文里的 `hifi-asset://` 引用改写为内嵌 data URI 或旁挂
//!   相对路径后写盘。

use crate::notebook_assets::{
    self, scan_asset_refs, NotebookAsset, NotebookAssetKind, NotebookAssetMap,
};
use crate::project_fragment::{ProjectFragment, ProjectFragmentKind};
use crate::state::AppState;
use base64::Engine as _;
use serde_json::json;
use std::path::{Path, PathBuf};
use tauri::State;

const B64: base64::engine::general_purpose::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn b64_encode(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

fn b64_decode(value: &str) -> Result<Vec<u8>, String> {
    B64.decode(value.trim())
        .map_err(|e| format!("附件数据不是合法的 base64: {e}"))
}

/// 按扩展名给出 MIME 兜底（前端一般会显式传，这里只兜底）。
fn mime_for_ext(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "avif" => "image/avif",
        "hsf" => "application/x-hifishifter-fragment",
        _ => "application/octet-stream",
    }
}

fn kind_from_str(value: &str) -> NotebookAssetKind {
    match value {
        "clip_payload" | "clipPayload" | "clip" => NotebookAssetKind::ClipPayload,
        _ => NotebookAssetKind::Image,
    }
}

// ─── 附件 ─────────────────────────────────────────────────────────────────────

pub(super) fn put_asset(
    state: State<'_, AppState>,
    asset_id: String,
    kind: String,
    ext: String,
    mime: Option<String>,
    data_base64: String,
    meta: Option<serde_json::Value>,
) -> serde_json::Value {
    let id = match notebook_assets::sanitize_asset_id(&asset_id) {
        Ok(id) => id,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let bytes = match b64_decode(&data_base64) {
        Ok(bytes) => bytes,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let dir = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let ext = notebook_assets::sanitize_ext(&ext);
    if let Err(error) = notebook_assets::write_asset_bytes(&dir, &id, &ext, &bytes) {
        return json!({ "ok": false, "error": error });
    }

    let asset = NotebookAsset {
        kind: kind_from_str(&kind),
        ext: ext.clone(),
        mime: mime.unwrap_or_else(|| mime_for_ext(&ext).to_string()),
        byte_len: bytes.len() as u64,
        created_at_ms: crate::state::now_unix_ms(),
        orphaned: false,
        meta: meta.unwrap_or(serde_json::Value::Null),
    };
    let byte_len = asset.byte_len;
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notebook_assets.insert(id.clone(), asset);
        // 附件是工程内容的一部分（正文引用了它），落盘后必须标脏。
        p.dirty = true;
    }

    json!({ "ok": true, "assetId": id, "byteLen": byte_len })
}

pub(super) fn read_asset(state: State<'_, AppState>, asset_id: String) -> serde_json::Value {
    let asset = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notebook_assets.get(&asset_id).cloned()
    };
    let Some(asset) = asset else {
        return json!({ "ok": false, "error": "notebook_asset_not_found" });
    };
    let dir = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    match notebook_assets::read_asset_bytes(&dir, &asset_id, &asset.ext) {
        Ok(bytes) => json!({
            "ok": true,
            "mime": asset.mime,
            "byteLen": bytes.len(),
            "base64": b64_encode(&bytes),
        }),
        Err(error) => json!({ "ok": false, "error": error, "missing": true }),
    }
}

pub(super) fn list_assets(state: State<'_, AppState>) -> serde_json::Value {
    let assets = state.notebook_assets_snapshot();
    let dir = state.notebook_asset_dir().unwrap_or_else(|_| PathBuf::new());
    let entries: Vec<serde_json::Value> = assets
        .iter()
        .map(|(id, asset)| {
            let exists = notebook_assets::locate_asset_file(&dir, id).is_some();
            json!({
                "id": id,
                "kind": match asset.kind {
                    NotebookAssetKind::Image => "image",
                    NotebookAssetKind::ClipPayload => "clip_payload",
                },
                "ext": asset.ext,
                "mime": asset.mime,
                "byteLen": asset.byte_len,
                "createdAtMs": asset.created_at_ms,
                "meta": asset.meta,
                "exists": exists,
            })
        })
        .collect();
    json!({ "ok": true, "assets": entries })
}

/// 从登记表移除一条附件并删除文件。
pub(super) fn remove_asset(state: State<'_, AppState>, asset_id: String) -> serde_json::Value {
    let dir = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let removed = {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        match p.notebook_assets.remove(&asset_id) {
            Some(asset) => {
                let _ = notebook_assets::remove_asset_file(&dir, &asset_id, &asset.ext);
                p.dirty = true;
                true
            }
            None => false,
        }
    };
    json!({ "ok": true, "removed": removed })
}

/// 按正文引用清理孤儿附件（保存前调用）。
pub(super) fn prune_assets(state: State<'_, AppState>) -> serde_json::Value {
    let removed = state.prune_notebook_assets();
    json!({ "ok": true, "removed": removed })
}

/// 附件在磁盘上的绝对路径（"在文件管理器中显示"用）。
pub(super) fn asset_path(state: State<'_, AppState>, asset_id: String) -> serde_json::Value {
    let dir = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    match notebook_assets::locate_asset_file(&dir, &asset_id) {
        Some((path, _)) => json!({ "ok": true, "path": path.display().to_string() }),
        None => json!({ "ok": false, "error": "notebook_asset_file_missing" }),
    }
}

/// 把磁盘上任意一个文件读成 base64（拖入的图片走这条；音频导入有另一条命令）。
pub(super) fn read_file_base64(path: String, max_bytes: Option<u64>) -> serde_json::Value {
    let path = PathBuf::from(&path);
    if !path.is_file() {
        return json!({ "ok": false, "error": format!("Not a file: {}", path.display()) });
    }
    let limit = max_bytes.unwrap_or(64 * 1024 * 1024);
    match std::fs::metadata(&path) {
        Ok(meta) if meta.len() > limit => {
            return json!({ "ok": false, "error": "notebook_file_too_large", "byteLen": meta.len() })
        }
        Ok(_) => {}
        Err(error) => return json!({ "ok": false, "error": error.to_string() }),
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match std::fs::read(&path) {
        Ok(bytes) => json!({
            "ok": true,
            "mime": mime_for_ext(&ext),
            "ext": ext,
            "byteLen": bytes.len(),
            "base64": b64_encode(&bytes),
        }),
        Err(error) => json!({ "ok": false, "error": error.to_string() }),
    }
}

// ─── 剪贴板载荷 ───────────────────────────────────────────────────────────────

fn fragment_preview(fragment: &ProjectFragment) -> (serde_json::Value, f64) {
    let mut rows = Vec::new();
    let mut end_sec = 0.0f64;
    for clip in &fragment.timeline.clips {
        let track_name = fragment
            .timeline
            .tracks
            .iter()
            .find(|t| t.id == clip.track_id)
            .map(|t| t.name.clone())
            .unwrap_or_default();
        end_sec = end_sec.max(clip.start_sec + clip.length_sec);
        rows.push(json!({
            "trackId": clip.track_id,
            "trackName": track_name,
            "name": clip.name,
            "startSec": clip.start_sec,
            "lengthSec": clip.length_sec,
        }));
    }
    (json!(rows), end_sec)
}

/// 参数线载荷的摘要：参数名 + 总帧数 + 降采样后的曲线（供暂存块画迷你曲线）。
fn param_preview(value: &serde_json::Value) -> serde_json::Value {
    let param = value
        .get("param")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let frame_period_ms = value
        .get("frame_period_ms")
        .or_else(|| value.get("framePeriodMs"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    // v2 是 segments，v1 是单段 values —— 两种都要能画。
    let mut total_frames = 0usize;
    let mut collected: Vec<f64> = Vec::new();
    let segments: Vec<&serde_json::Value> = match value.get("segments").and_then(|v| v.as_array()) {
        Some(list) => list.iter().collect(),
        None => value.get("values").map(|_| vec![value]).unwrap_or_default(),
    };
    for segment in segments {
        if let Some(values) = segment.get("values").and_then(|v| v.as_array()) {
            total_frames += values.len();
            for v in values {
                collected.push(v.as_f64().unwrap_or(0.0));
            }
        }
    }
    // 降采样到 ≤ 200 点：暂存块里的迷你曲线不需要原始精度，而载荷可能有
    // 上万帧 —— 把整条曲线交给前端会让 IPC 白白搬运几百 KB。
    const MAX_POINTS: usize = 200;
    let step = (collected.len() / MAX_POINTS).max(1);
    let sparkline: Vec<f64> = collected.iter().step_by(step).copied().collect();

    json!({
        "param": param,
        "framePeriodMs": frame_period_ms,
        "frameCount": total_frames,
        "sparkline": sparkline,
    })
}

/// 读出系统剪贴板里的 HiFiShifter 载荷（含时间轴片段与参数线两种编码）。
///
/// 这是现状缺口的关键补齐：`read_system_clipboard_object` 只能读 UTF-8，
/// 而时间轴片段是 MessagePack，因此记事本此前完全感知不到剪贴板里有数据。
pub(super) fn read_clipboard_payload() -> serde_json::Value {
    let bytes = match crate::system_clipboard::read_bytes() {
        Ok(Some(bytes)) if !bytes.is_empty() => bytes,
        Ok(_) => return json!({ "ok": true, "available": false }),
        Err(error) => return json!({ "ok": false, "error": error }),
    };

    if let Ok(fragment) = ProjectFragment::decode(&bytes) {
        let (preview, duration_sec) = fragment_preview(&fragment);
        let kind = match fragment.kind {
            ProjectFragmentKind::Clips => "clips",
            ProjectFragmentKind::Tracks => "tracks",
            ProjectFragmentKind::Project => "project",
        };
        let summary = json!({
            "clipKind": kind,
            "clipCount": fragment.timeline.clips.len(),
            "trackCount": fragment.timeline.tracks.len(),
            "sourceProject": fragment.source_project_name,
            "durationSec": duration_sec,
            "preview": preview,
        });
        return json!({
            "ok": true,
            "available": true,
            "kind": kind,
            "encoding": "fragment",
            "ext": "hsf",
            "mime": mime_for_ext("hsf"),
            "byteLen": bytes.len(),
            "base64": b64_encode(&bytes),
            "summary": summary,
        });
    }

    if let Some(value) = super::timeline_clipboard::param_payload_value(&bytes) {
        let preview = param_preview(&value);
        let summary = json!({
            "clipKind": "param",
            "clipCount": 0,
            "trackCount": 0,
            "sourceProject": serde_json::Value::Null,
            "durationSec": 0.0,
            "preview": [],
            "param": preview,
        });
        return json!({
            "ok": true,
            "available": true,
            "kind": "param",
            "encoding": "param",
            "ext": "hsp",
            "mime": "application/json",
            "byteLen": bytes.len(),
            "base64": b64_encode(&bytes),
            "summary": summary,
        });
    }

    json!({ "ok": true, "available": false })
}

/// 把暂存的载荷字节原样写回系统剪贴板。
///
/// `write_system_clipboard_object` 只能写 UTF-8 字符串，MessagePack 片段过不去；
/// 这里走字节通道，并同步刷新剪贴板缓存，使 `clipboard_kind` 的快速路径立刻
/// 反映新内容（否则时间轴的"可粘贴"提示会滞后到下一次真实读取）。
pub(super) fn write_clipboard_payload(
    payload_base64: String,
    text_summary: Option<String>,
) -> serde_json::Value {
    let bytes = match b64_decode(&payload_base64) {
        Ok(bytes) => bytes,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let summary = text_summary
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "HiFiShifter data restored. Paste in HiFiShifter.".to_string());
    match crate::system_clipboard::write_bytes(&bytes, &summary) {
        Ok(()) => {
            if let Some(seq) = crate::system_clipboard::clipboard_seq_num() {
                let (kind, clip_count, track_count, source_project) =
                    if let Ok(fragment) = ProjectFragment::decode(&bytes) {
                        (
                            Some(match fragment.kind {
                                ProjectFragmentKind::Clips => "clips".to_string(),
                                ProjectFragmentKind::Tracks => "tracks".to_string(),
                                ProjectFragmentKind::Project => "project".to_string(),
                            }),
                            fragment.timeline.clips.len() as u64,
                            fragment.timeline.tracks.len() as u64,
                            Some(fragment.source_project_name.clone()),
                        )
                    } else if super::timeline_clipboard::param_payload_value(&bytes).is_some()
                    {
                        (Some("param".to_string()), 0, 0, None)
                    } else {
                        (None, 0, 0, None)
                    };
                crate::system_clipboard::write_clipboard_cache(
                    crate::system_clipboard::ClipboardCacheEntry {
                        seq,
                        hifi_kind: kind,
                        hifi_clip_count: clip_count,
                        hifi_track_count: track_count,
                        hifi_source_project: source_project,
                        reaper_available: crate::system_clipboard::has_reaper_format(),
                    },
                );
            }
            json!({ "ok": true })
        }
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

/// 读系统剪贴板里的位图（CF_DIB）。前端把它包成 BMP 再走统一图片流水线。
pub(super) fn read_clipboard_image() -> serde_json::Value {
    match crate::system_clipboard::read_bitmap_dib() {
        Ok(Some(bitmap)) => json!({
            "ok": true,
            "available": true,
            "width": bitmap.width,
            "height": bitmap.height,
            "bitsPerPixel": bitmap.bits_per_pixel,
            "base64": b64_encode(&bitmap.bytes),
        }),
        Ok(None) => json!({ "ok": true, "available": false }),
        Err(error) => json!({ "ok": false, "error": error }),
    }
}

// ─── 撤销分节 ─────────────────────────────────────────────────────────────────

pub(super) fn seal_notes_history(state: State<'_, AppState>) -> serde_json::Value {
    state.seal_notes_history();
    json!({ "ok": true })
}

// ─── 导出 ─────────────────────────────────────────────────────────────────────

fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default()
}

/// 导出正文：把 `hifi-asset://` 引用改写成内嵌 data URI 或旁挂相对路径。
///
/// `image_mode`：
/// - `embed`（默认）：字节内嵌为 data URI，产物是单个自包含文件；
/// - `copyFolder`：附件复制到 `<导出名>.assets/` 并与导出文件并列，
///   引用改写为相对路径（`.md` 在别的编辑器里也能看到图）。
pub(super) fn export_document(
    state: State<'_, AppState>,
    suggested_name: String,
    extension: String,
    content: String,
    image_mode: Option<String>,
) -> serde_json::Value {
    let ext = notebook_assets::sanitize_ext(&extension);
    let mode = image_mode.unwrap_or_else(|| "embed".to_string());
    let filter_label = match ext.as_str() {
        "html" => "HTML Document",
        _ => "Markdown Document",
    };

    let picked = rfd::FileDialog::new()
        .add_filter(filter_label, &[ext.as_str()])
        .set_file_name(format!("{suggested_name}.{ext}"))
        .save_file();
    let Some(output_path) = picked else {
        return json!({ "ok": true, "canceled": true });
    };
    if extension_of(&output_path) != ext {
        return json!({ "ok": false, "error": "notebook_export_extension_mismatch" });
    }

    let dir = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let assets = state.notebook_assets_snapshot();

    // 旁挂模式需要先知道导出文件的主名，才能拼出 `<主名>.assets/`。
    let asset_folder = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|stem| format!("{stem}.assets"))
        .unwrap_or_else(|| "assets".to_string());
    let asset_dir_out = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&asset_folder);
    if mode == "copyFolder" {
        if let Err(error) = std::fs::create_dir_all(&asset_dir_out) {
            return json!({ "ok": false, "error": format!("创建导出附件目录失败: {error}") });
        }
    }

    // 从后往前替换，避免前面的替换使后面的下标失效。
    let refs = scan_asset_refs(&content);
    let mut out = content;
    let mut missing = Vec::new();
    let mut copied = 0usize;
    for asset_ref in refs.into_iter().rev() {
        let Some(asset) = assets.get(&asset_ref.id) else {
            missing.push(asset_ref.id.clone());
            continue;
        };
        let bytes = match notebook_assets::read_asset_bytes(&dir, &asset_ref.id, &asset.ext) {
            Ok(bytes) => bytes,
            Err(_) => {
                missing.push(asset_ref.id.clone());
                continue;
            }
        };
        let replacement = if mode == "copyFolder" {
            let file_name = format!("{}.{}", asset_ref.id, asset.ext);
            let target = asset_dir_out.join(&file_name);
            if std::fs::write(&target, &bytes).is_ok() {
                copied += 1;
            } else {
                missing.push(asset_ref.id.clone());
                continue;
            }
            // 导出目录里的相对引用：用正斜杠，跨平台可移植。
            format!("./{asset_folder}/{file_name}")
        } else {
            format!("data:{};base64,{}", asset.mime, b64_encode(&bytes))
        };
        out.replace_range(asset_ref.start..asset_ref.end, &replacement);
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                return json!({ "ok": false, "error": format!("创建导出目录失败: {error}") });
            }
        }
    }
    match std::fs::write(&output_path, out.as_bytes()) {
        Ok(()) => json!({
            "ok": true,
            "canceled": false,
            "path": output_path.display().to_string(),
            "missingAssets": missing,
            "copiedAssets": copied,
        }),
        Err(error) => json!({ "ok": false, "error": error.to_string() }),
    }
}

/// 把一条附件另存到用户选定路径。
pub(super) fn save_asset_as(
    state: State<'_, AppState>,
    asset_id: String,
    suggested_name: Option<String>,
) -> serde_json::Value {
    let (asset, dir) = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let Some(asset) = p.notebook_assets.get(&asset_id).cloned() else {
            return json!({ "ok": false, "error": "notebook_asset_not_found" });
        };
        drop(p);
        let dir = match state.notebook_asset_dir() {
            Ok(dir) => dir,
            Err(error) => return json!({ "ok": false, "error": error }),
        };
        (asset, dir)
    };
    let bytes = match notebook_assets::read_asset_bytes(&dir, &asset_id, &asset.ext) {
        Ok(bytes) => bytes,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let name = suggested_name
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("{asset_id}.{}", asset.ext));
    let picked = rfd::FileDialog::new().set_file_name(name).save_file();
    let Some(output_path) = picked else {
        return json!({ "ok": true, "canceled": true });
    };
    match std::fs::write(&output_path, &bytes) {
        Ok(()) => json!({ "ok": true, "canceled": false, "path": output_path.display().to_string() }),
        Err(error) => json!({ "ok": false, "error": error.to_string() }),
    }
}

/// 保存时的附件整理入口（各保存路径共用）。
pub(crate) fn prepare_assets_for_save(state: &AppState, project_path: &Path) {
    if let Err(error) = state.bind_notebook_asset_dir(project_path) {
        log::warn!("[notebook] 附件目录迁移失败: {error}");
    }
    let removed = state.prune_notebook_assets();
    if removed > 0 {
        log::info!("[notebook] 保存时清理了 {removed} 条未引用附件");
    }
}

/// 把附件**复制**一份到 `project_path` 的旁挂目录，用于定时备份。
///
/// 与 `prepare_assets_for_save` 的区别是"只复制、不迁移、不改绑定"：备份写入
/// 的是另一个路径，若在那里绑定/迁移，就会把工程的附件从工程旁边搬走。
pub(crate) fn mirror_assets_for_backup(state: &AppState, project_path: &Path) {
    let source = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(_) => return,
    };
    let target = notebook_assets::asset_dir_for_project(project_path);
    if source == target {
        return;
    }
    if notebook_assets::ensure_dir(&target).is_err() {
        return;
    }
    for (id, asset) in state.notebook_assets_snapshot() {
        let Ok(bytes) = notebook_assets::read_asset_bytes(&source, &id, &asset.ext) else {
            continue;
        };
        let _ = notebook_assets::write_asset_bytes(&target, &id, &asset.ext, &bytes);
    }
}

/// 归档（ZIP）保存时把附件一并打包：返回 `(压缩包内条目名, 源文件路径)` 列表。
pub(crate) fn archive_asset_entries(state: &AppState, folder: &str) -> Vec<(String, PathBuf)> {
    let dir = match state.notebook_asset_dir() {
        Ok(dir) => dir,
        Err(_) => return Vec::new(),
    };
    let mut entries = Vec::new();
    for (id, _asset) in state.notebook_assets_snapshot() {
        let Some((path, ext)) = notebook_assets::locate_asset_file(&dir, &id) else {
            continue;
        };
        entries.push((format!("{folder}/{id}.{ext}"), path));
    }
    entries
}

/// 打开工程时装载附件登记表并绑定落点。
pub(crate) fn bind_assets_on_open(state: &AppState, project_path: &Path, assets: NotebookAssetMap) {
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notebook_assets = assets;
    }
    if let Err(error) = state.bind_notebook_asset_dir(project_path) {
        log::warn!("[notebook] 绑定附件目录失败: {error}");
    }
}
