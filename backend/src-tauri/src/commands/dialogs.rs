// 文件对话框命令的实现。
//
// 【为什么全部经过 `platform::dialog`】`rfd` 在移动目标上**编译不过**（后端只有
// gtk3/linux/macos/wasm/win32），它已从全局依赖移到桌面专属的 target 段。本文件的
// 每一个调用点都改走收敛层门面：桌面分支是逐字节的 `rfd` 转发，移动分支是
// 系统选择器（SAF）+ 导入即拷贝（见门面的模块文档）。
//
// 【为什么返回值形状一个字都不能改】前端 `services/api/*` 解析的是
// `{ok, canceled, path|paths}`；多一个字段可以（前端忽略未知字段），少一个字段就会
// 让"用户取消"被当成"失败"弹红条。因此两端的 JSON 与重构前**逐字节相同**。

use std::path::PathBuf;

use crate::platform::dialog::FileDialog;
use crate::platform::paths::ImportTarget;

const MEDIA_DIALOG_EXTENSIONS: &[&str] = &[
    "wav", "flac", "mp3", "ogg", "oga", "opus", "aac", "m4a", "aif", "aiff", "wma", "ac3", "eac3",
    "ape", "wv", "mp2", "mpa", "dts", "amr", "mp4", "m4v", "mov", "mkv", "webm", "avi", "flv",
    "wmv", "ts", "mts", "m2ts", "vob", "mpg", "mpeg", "3gp", "3g2", "ogv", "rm", "rmvb",
];

/// 「用户取消」的结果。
fn canceled() -> serde_json::Value {
    serde_json::json!({ "ok": true, "canceled": true })
}

fn picked(path: PathBuf) -> serde_json::Value {
    serde_json::json!({ "ok": true, "canceled": false, "path": path.display().to_string() })
}

pub(super) fn open_audio_dialog(app: &tauri::AppHandle) -> serde_json::Value {
    let picked_path = FileDialog::new()
        .add_filter("Media Files", MEDIA_DIALOG_EXTENSIONS)
        // 移动端：选中的素材收编进工作区（`media/`），文件浏览器因此看得见它。
        .set_import_target(ImportTarget::Media)
        .pick_file(app);

    match picked_path {
        None => canceled(),
        Some(path) => picked(path),
    }
}

/// 为"源文件已变更"窗口的单个缺失文件选择替代文件。
///
/// 通过对话框标题与预设文件名明确告知用户当前正在为哪一个文件选择替代，
/// 并尽量从原路径所在目录开始浏览。
pub(super) fn open_audio_dialog_for_source(
    app: &tauri::AppHandle,
    source_path: String,
    dialog_title: String,
) -> serde_json::Value {
    use std::path::Path;

    let source_path_trimmed = source_path.trim();
    let source = Path::new(source_path_trimmed);

    let mut dialog = FileDialog::new()
        .add_filter("Media Files", MEDIA_DIALOG_EXTENSIONS)
        .set_import_target(ImportTarget::Media);
    if let Some(parent) = source.parent() {
        if parent.is_dir() {
            dialog = dialog.set_directory(parent);
        }
    }
    if let Some(file_name) = source.file_name() {
        dialog = dialog.set_file_name(file_name.to_string_lossy().to_string());
    }
    let title = dialog_title.trim();
    if !title.is_empty() {
        dialog = dialog.set_title(title);
    }

    match dialog.pick_file(app) {
        None => canceled(),
        Some(path) => picked(path),
    }
}

pub(super) fn open_audio_dialog_multi(app: &tauri::AppHandle) -> serde_json::Value {
    let picked_paths = FileDialog::new()
        .add_filter("Media Files", MEDIA_DIALOG_EXTENSIONS)
        .set_import_target(ImportTarget::Media)
        .pick_files(app);

    match picked_paths {
        None => canceled(),
        Some(paths) => {
            let path_list: Vec<String> = paths
                .into_iter()
                .map(|path| path.display().to_string())
                .collect();
            serde_json::json!({ "ok": true, "canceled": false, "paths": path_list })
        }
    }
}

/// 选择导出目标。
///
/// 【为什么两端共用同一条代码路径】桌面上 `save_file` 弹 `rfd` 的保存框；
/// 移动端同一个调用返回 [`crate::platform::paths::export_dir`] 下的确定路径
/// （用户能在文件管理器里找到的那个目录）。门面已经吃掉了差异，这里不需要分支。
pub(super) fn pick_output_path(app: &tauri::AppHandle) -> serde_json::Value {
    let picked_path = FileDialog::new()
        .add_filter("WAV", &["wav"])
        .set_file_name("output.wav")
        .save_file(app);

    match picked_path {
        None => canceled(),
        Some(path) => picked(path),
    }
}

/// 选择目录。
///
/// 【移动端返回工作区根】移动端的"选目录"语义在系统选择器里是 SAF 的树选择器
/// （Android）或安全作用域 URL（iOS），拿到之后仍然要逐文件物化。而本应用在移动端
/// 的素材**本来就都在工作区里**（导入即拷贝），因此"选目录"实际要回答的是
/// "文件浏览器从哪开始" —— 那就是工作区根。
pub(super) fn pick_directory(app: &tauri::AppHandle) -> serde_json::Value {
    if crate::platform::is_mobile() {
        return match crate::platform::paths::media_root(app) {
            Some(root) => {
                if let Err(error) = crate::platform::paths::ensure_dir(&root) {
                    return serde_json::json!({ "ok": false, "canceled": false, "error": error });
                }
                picked(root)
            }
            None => serde_json::json!({
                "ok": false,
                "canceled": false,
                "error": "workspace root unavailable on this surface",
            }),
        };
    }

    match FileDialog::new().pick_folder(app) {
        None => canceled(),
        Some(path) => picked(path),
    }
}

pub(super) fn pick_midi_output_path(app: &tauri::AppHandle) -> serde_json::Value {
    let file_type = &hifishifter_kernel::file_types::MIDI;

    let picked_path = FileDialog::new()
        .add_filter(file_type.name(), &file_type.extension_refs())
        .set_file_name("export.mid")
        .save_file(app);

    match picked_path {
        None => canceled(),
        Some(path) => picked(path),
    }
}

pub(super) fn open_midi_dialog(app: &tauri::AppHandle) -> serde_json::Value {
    let file_type = &hifishifter_kernel::file_types::MIDI;
    let picked_path = FileDialog::new()
        .add_filter(file_type.name(), &file_type.extension_refs())
        .set_import_target(ImportTarget::Document)
        .pick_file(app);

    match picked_path {
        None => canceled(),
        Some(path) => picked(path),
    }
}

/// 打开系统文件选择器，把选中的文件**读成字节**返回。
///
/// 【为什么需要它】WebView 的 `<input type="file">` 在 Android 上点了毫无反应 ——
/// wry 0.57 的 `RustWebChromeClient` 没有实现 `onShowFileChooser`。而本应用有
/// 三处"选一个小文件读进内存"的入口（导入外观主题 / 导入布局 / 往记事本插入图片），
/// 它们要的是字节而不是路径。桌面仍走原生 `<input>`（零 IPC、零拷贝），
/// 移动端走这条命令。
///
/// 【为什么限制大小】它是"把整个文件塞进一次 IPC 的 base64 字符串"的形状，
/// 只适合几 MB 的文档与图片。误选一个 200 MB 的 WAV 时应当明确报错，
/// 而不是让 WebView 因为一个 270 MB 的字符串而 OOM。
pub(super) fn pick_files_bytes(
    app: &tauri::AppHandle,
    extensions: Vec<String>,
    multiple: bool,
    max_bytes: u64,
) -> serde_json::Value {
    use base64::Engine as _;

    let picked = match crate::platform::native::pick_files(app, &extensions, multiple) {
        Ok(paths) => paths,
        Err(error) => return serde_json::json!({ "ok": false, "canceled": false, "error": error }),
    };
    if picked.is_empty() {
        return serde_json::json!({ "ok": true, "canceled": true, "files": [] });
    }

    let mut files = Vec::with_capacity(picked.len());
    for path in picked {
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                return serde_json::json!({
                    "ok": false,
                    "canceled": false,
                    "error": format!("cannot read {}: {error}", path.display()),
                })
            }
        };
        if metadata.len() > max_bytes {
            return serde_json::json!({
                "ok": false,
                "canceled": false,
                "error": format!(
                    "the selected file is too large ({} bytes, limit {max_bytes})",
                    metadata.len()
                ),
            });
        }
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                return serde_json::json!({
                    "ok": false,
                    "canceled": false,
                    "error": format!("cannot read {}: {error}", path.display()),
                })
            }
        };
        let name = path
            .file_name()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|| "import".to_string());
        files.push(serde_json::json!({
            "name": name,
            "base64": base64::engine::general_purpose::STANDARD.encode(&bytes),
        }));
        // 缓存里的临时副本已经读进内存，可以立刻回收。
        let _ = std::fs::remove_file(&path);
    }
    serde_json::json!({ "ok": true, "canceled": false, "files": files })
}

/// 把已经写好的产物交给用户选定的位置（移动端 SAF 创建文档）。
///
/// 【为什么这是一个独立命令而不是塞进导出流程】导出是**长操作**，失败与成功都要先
/// 落盘；把"选位置"塞进导出中间会让失败留下空文件。独立命令让界面可以在导出成功
/// 之后问"要存到哪"，而桌面上它直接返回 `ok: true, skipped: true`（桌面本来就有
/// 保存对话框）。
pub(super) fn save_output_to_location(app: &tauri::AppHandle, path: String) -> serde_json::Value {
    let source = PathBuf::from(path.trim());
    if !source.is_file() {
        return serde_json::json!({ "ok": false, "error": "the file no longer exists" });
    }
    match crate::platform::dialog::save_output_to_location(app, &source, None) {
        Ok(Some(name)) => serde_json::json!({ "ok": true, "canceled": false, "name": name }),
        Ok(None) => {
            if crate::platform::is_mobile() {
                serde_json::json!({ "ok": true, "canceled": true })
            } else {
                serde_json::json!({ "ok": true, "canceled": false, "skipped": true })
            }
        }
        Err(error) => serde_json::json!({ "ok": false, "canceled": false, "error": error }),
    }
}
