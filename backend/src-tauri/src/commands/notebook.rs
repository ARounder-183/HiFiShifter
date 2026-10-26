//! 记事本后端：附件登记、剪贴板载荷暂存、文档导出。
//!
//! 前端只做"呈现与交互"，所有落盘或系统剪贴板的动作都收敛在这里：
//!
//! - **附件**（图片 / 剪贴板载荷）：字节以 base64 **内嵌在工程文件里**，
//!   没有任何旁挂目录（见 `crate::notebook_assets` 的模块注释）。
//! - **剪贴板载荷暂存**：把 `application/x-hifishifter-object` 槽位里的原始
//!   字节原样取出来交给前端保存；恢复时再把同一份字节原样写回。
//!   **不做任何重新序列化** —— 载荷是 MessagePack，重新编码会随版本漂移，
//!   原样字节才能保证"暂存 → 恢复"是无损的。
//! - **导出**：把正文里的 `hifi-asset://` 引用改写为内嵌 data URI 后写盘，
//!   导出物是单个自包含文件。
//!
//! ## 稳定错误码
//!
//! 本模块对前端的 `error` 字段统一返回**稳定错误码**，供前端做 i18n 映射与
//! 精确分支（如 `notebook_file_too_large` 的 "too-large" 归类）。带上下文的
//! 错误用 `code:detail` 形式（前端按 `code` 前缀匹配，冒号后为人类可读细节）：
//!
//! - `notebook_bad_base64` —— 附件/载荷不是合法 base64（detail: 解码器报错）
//! - `notebook_not_a_file` —— 读取目标不是常规文件（detail: 路径）
//! - `notebook_file_too_large` —— 拖入文件超过大小上限（附加 `byteLen` 字段）
//! - `notebook_unsupported_image_ext` —— 拖入文件扩展名不在图片白名单（detail: 扩展名）
//! - `notebook_asset_too_large` —— 附件解码后超过大小上限（附加 `byteLen` 字段）
//! - `notebook_export_mkdir_failed` —— 创建导出目录失败（detail: IO 错误）
//! - `notebook_read_task_failed` —— 读取任务本身失败（detail: join 错误）
//!
//! 另有 `notebook_asset_not_found` / `notebook_asset_empty` /
//! `notebook_export_extension_mismatch`（原样返回，无 detail）与
//! `notebook_asset_id_*`（见 `notebook_assets::sanitize_asset_id`）。

use crate::notebook_assets::{
    self, scan_asset_refs, NotebookAsset, NotebookAssetKind, NotebookAssetMap,
};
use crate::project_fragment::{ProjectFragment, ProjectFragmentKind};
use crate::state::AppState;
use base64::Engine as _;
use serde_json::json;
use std::path::Path;
use tauri::State;

const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

// `read_file_base64` 的扩展名白名单与服务端硬上限在内核
// `notebook_assets::READABLE_IMAGE_EXTS` / `READ_FILE_MAX_BYTES`
// （App 与插件共用的唯一实现，注释与测试都在那里）。

/// 单条附件解码后的字节上限。
///
/// 附件以 base64 内嵌进工程文件：一条超大附件会同时撑爆内存里的工程、
/// 每次保存的写盘量与撤销/备份的克隆成本，且没有"事后瘦身"手段
/// （会话内只增不删）。上限必须在写入前拒绝。
const MAX_ASSET_BYTES: usize = 32 * 1024 * 1024;

/// `data_base64` 文本的长度上限（解码后不超过 [`MAX_ASSET_BYTES`]）。
///
/// 【为什么先看长度再解码】解码会立刻分配 3/4 的输入长度：一段 1 GiB 的
/// `data_base64` 会先吃掉 750 MiB 内存，然后才被上面那条上限拒掉。门禁必须在
/// 分配之前。
const MAX_ASSET_BASE64_LEN: usize = MAX_ASSET_BYTES / 3 * 4 + 4;

fn b64_encode(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

fn b64_decode(value: &str) -> Result<Vec<u8>, String> {
    B64.decode(value.trim())
        .map_err(|e| format!("notebook_bad_base64: {e}"))
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
    // 长度先看：见 `MAX_ASSET_BASE64_LEN` 的理由（门禁在分配之前）。
    if data_base64.len() > MAX_ASSET_BASE64_LEN {
        return json!({
            "ok": false,
            "error": "notebook_asset_too_large",
            "byteLen": data_base64.len() / 4 * 3,
        });
    }
    // 再解码校验：宁可在这里拒绝，也不要把一段坏 base64 写进工程文件 ——
    // 那会让工程在后续每次保存/读取时都带着一颗雷。
    let byte_len = match b64_decode(&data_base64) {
        Ok(bytes) => bytes.len(),
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    // 大小上限在服务端收口：`data_base64` 的长度由调用方决定，前端自己的
    // 预检查只是体验优化，不能当作安全边界。
    if byte_len > MAX_ASSET_BYTES {
        return json!({ "ok": false, "error": "notebook_asset_too_large", "byteLen": byte_len });
    }
    let ext = notebook_assets::sanitize_ext(&ext);
    // MIME 收口在内核（`sanitize_asset_mime`）：它是导出物里 `data:{mime};base64,…`
    // 的类型段，前端可控，不能原样落盘。
    let mime = notebook_assets::sanitize_asset_mime(mime.as_deref().unwrap_or_default(), &ext);

    let asset = NotebookAsset {
        kind: kind_from_str(&kind),
        ext,
        mime,
        byte_len: byte_len as u64,
        created_at_ms: crate::state::now_unix_ms(),
        orphaned: false,
        meta: meta.unwrap_or(serde_json::Value::Null),
        data: data_base64,
    };
    {
        let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notebook_assets.insert(id.clone(), asset);
        // 附件是工程内容的一部分（正文引用了它），必须标脏。
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
        return json!({ "ok": false, "error": "notebook_asset_not_found", "missing": true });
    };
    if asset.data.is_empty() {
        // 登记项在但内容为空（工程被手工编辑过等）：按"附件缺失"处理。
        return json!({ "ok": false, "error": "notebook_asset_empty", "missing": true });
    }
    json!({
        "ok": true,
        "mime": asset.mime,
        "byteLen": asset.byte_len,
        "base64": asset.data,
    })
}

pub(super) fn list_assets(state: State<'_, AppState>) -> serde_json::Value {
    let assets = state.notebook_assets_snapshot();
    let entries: Vec<serde_json::Value> = assets
        .iter()
        .map(|(id, asset)| {
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
                // 有登记项但内容为空（工程被手工编辑过等）时为 false。
                "hasData": !asset.data.is_empty(),
            })
        })
        .collect();
    json!({ "ok": true, "assets": entries })
}

/// 标记一条附件为待清理。
///
/// 只打标记、不删字节：撤销/重做可能把引用恢复回来，真正的清理放在保存时
/// （`prune_assets`）。这与"会话内附件只增不删"的整体约定一致。
pub(super) fn remove_asset(state: State<'_, AppState>, asset_id: String) -> serde_json::Value {
    let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
    match p.notebook_assets.get_mut(&asset_id) {
        Some(asset) => {
            asset.orphaned = true;
            p.dirty = true;
            json!({ "ok": true, "removed": true })
        }
        None => json!({ "ok": true, "removed": false }),
    }
}

/// 按正文引用清理孤儿附件（保存前调用）。
pub(super) fn prune_assets(state: State<'_, AppState>) -> serde_json::Value {
    let removed = state.prune_notebook_assets();
    json!({ "ok": true, "removed": removed })
}

/// 把磁盘上的一个图片文件读成 base64（拖入的图片走这条；音频导入有另一条命令）。
///
/// 这是 IPC 可达的读盘通道，三条服务端收口缺一不可：
/// 1. **扩展名白名单**（`READABLE_IMAGE_EXTS`）——否则即任意文件读；
/// 2. **大小上限不可协商**：生效上限 = min(调用方值或缺省,
///    [`READ_FILE_MAX_BYTES`])，调用方传更大的值抬不高上限；
/// 3. **阻塞线程池**：文件读取 + base64 编码可能达数十 MB，包装命令把它放进
///    `spawn_blocking`（见 commands.rs 的 `notebook_read_file_base64`）。
/// `read_file_base64` 允许读取的扩展名白名单与服务端硬上限在内核
/// `notebook_assets::read_image_file`（App 与插件共用的唯一实现）；
/// 本包装只负责把错误码映射成响应 JSON，并把重活放进 `spawn_blocking`。
///
/// 这是 IPC 可达的读盘通道，三条服务端收口（扩展名白名单、不可协商的大小
/// 上限、is_file）都在共享实现里逐条收紧 —— 缺任何一条就是任意文件读。
pub(super) fn read_file_base64(path: String, max_bytes: Option<u64>) -> serde_json::Value {
    match notebook_assets::read_image_file(&path, max_bytes) {
        Ok(read) => json!({
            "ok": true,
            "mime": read.mime,
            "ext": read.ext,
            "byteLen": read.bytes.len(),
            "base64": b64_encode(&read.bytes),
        }),
        Err(notebook_assets::ReadImageFileError::TooLarge { byte_len }) => {
            json!({ "ok": false, "error": "notebook_file_too_large", "byteLen": byte_len })
        }
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
                    } else if super::timeline_clipboard::param_payload_value(&bytes).is_some() {
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

/// 导出正文：把 `hifi-asset://` 引用改写成内嵌 data URI。
///
/// 导出物是**单个自包含文件**（图片以 data URI 内嵌），不生成任何旁挂目录 ——
/// 与工程本身的存储模型一致：内容跟着文件走，拷到哪里都能看。
pub(super) fn export_document(
    state: State<'_, AppState>,
    suggested_name: String,
    extension: String,
    content: String,
) -> serde_json::Value {
    let ext = notebook_assets::sanitize_ext(&extension);
    let filter_label = match ext.as_str() {
        "html" => "HTML Document",
        _ => "Markdown Document",
    };
    // suggested_name 由前端传入，先清洗再拼扩展名：路径分隔符可以让对话框
    // 预导航到任意目录。
    let base_name = notebook_assets::sanitize_export_file_name(&suggested_name, "untitled");

    let picked = rfd::FileDialog::new()
        .add_filter(filter_label, &[ext.as_str()])
        .set_file_name(format!("{base_name}.{ext}"))
        .save_file();
    let Some(output_path) = picked else {
        return json!({ "ok": true, "canceled": true });
    };
    if extension_of(&output_path) != ext {
        return json!({ "ok": false, "error": "notebook_export_extension_mismatch" });
    }

    let assets = state.notebook_assets_snapshot();
    // 从后往前替换，避免前面的替换使后面的下标失效。
    let refs = scan_asset_refs(&content);
    let mut out = content;
    let mut missing = Vec::new();
    for asset_ref in refs.into_iter().rev() {
        let Some(asset) = assets.get(&asset_ref.id) else {
            missing.push(asset_ref.id.clone());
            continue;
        };
        if asset.data.is_empty() {
            missing.push(asset_ref.id.clone());
            continue;
        }
        // 工程文件里的 mime 可能是旧版本写下的任意值：导出前按同一份收口过滤，
        // 否则 `data:{mime};base64,…` 会把一条图片引用变成一段可执行文档。
        let mime = notebook_assets::sanitize_asset_mime(&asset.mime, &asset.ext);
        out.replace_range(
            asset_ref.start..asset_ref.end,
            &format!("data:{mime};base64,{}", asset.data),
        );
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                return json!({
                    "ok": false,
                    "error": format!("notebook_export_mkdir_failed: {error}")
                });
            }
        }
    }
    match std::fs::write(&output_path, out.as_bytes()) {
        Ok(()) => json!({
            "ok": true,
            "canceled": false,
            "path": output_path.display().to_string(),
            "missingAssets": missing,
        }),
        Err(error) => json!({ "ok": false, "error": error.to_string() }),
    }
}

/// 把一条附件的内容另存到用户选定路径（从内嵌数据里"取出"一个文件）。
pub(super) fn save_asset_as(
    state: State<'_, AppState>,
    asset_id: String,
    suggested_name: Option<String>,
) -> serde_json::Value {
    let asset = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        p.notebook_assets.get(&asset_id).cloned()
    };
    let Some(asset) = asset else {
        return json!({ "ok": false, "error": "notebook_asset_not_found" });
    };
    let bytes = match b64_decode(&asset.data) {
        Ok(bytes) => bytes,
        Err(error) => return json!({ "ok": false, "error": error }),
    };
    let fallback = format!("{asset_id}.{}", notebook_assets::sanitize_ext(&asset.ext));
    // suggested_name 来自前端；asset_id 来自工程文件（可能被手工编辑过）。
    // 两者都不可信，最终统一过一遍清洗器。
    let name = notebook_assets::sanitize_export_file_name(
        suggested_name
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(&fallback),
        "asset",
    );
    let picked = rfd::FileDialog::new().set_file_name(name).save_file();
    let Some(output_path) = picked else {
        return json!({ "ok": true, "canceled": true });
    };
    match std::fs::write(&output_path, &bytes) {
        Ok(()) => {
            json!({ "ok": true, "canceled": false, "path": output_path.display().to_string() })
        }
        Err(error) => json!({ "ok": false, "error": error.to_string() }),
    }
}

/// 保存时的附件整理入口（各保存路径共用）。
///
/// 内嵌模型下只剩一件事：把不再被引用的条目丢掉，让工程文件别无限膨胀。
/// 字节已经跟着工程文件走，没有目录要绑定、没有附件要镜像到备份。
pub(crate) fn prepare_assets_for_save(state: &AppState) {
    let removed = state.prune_notebook_assets();
    if removed > 0 {
        log::info!("[notebook] 保存时清理了 {removed} 条未引用附件");
    }
}

/// 打开工程时装载附件登记表（字节已随工程文件一起读入）。
pub(crate) fn bind_assets_on_open(state: &AppState, assets: NotebookAssetMap) {
    let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
    p.notebook_assets = assets;
}
