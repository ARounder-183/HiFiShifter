// 文件对话框命令的实现。
//
// 【为什么全部经过 `platform::dialog`】`rfd` 在移动目标上**编译不过**（后端只有
// gtk3/linux/macos/wasm/win32），它已从全局依赖移到桌面专属的 target 段。本文件的
// 每一个调用点都改走收敛层门面：桌面分支是逐字节的 `rfd` 转发，移动分支见门面文档。
//
// 【为什么返回值形状一个字都不能改】前端 `services/api/*` 解析的是
// `{ok, canceled, path|paths}`；多一个字段可以（前端忽略未知字段），少一个字段就会
// 让"用户取消"被当成"失败"弹红条。因此桌面路径的 JSON 与重构前**逐字节相同**，
// 只在移动端补一个 `reason`（见 `canceled()`）。

use std::path::PathBuf;

use crate::platform::dialog::FileDialog;

const MEDIA_DIALOG_EXTENSIONS: &[&str] = &[
    "wav", "flac", "mp3", "ogg", "oga", "opus", "aac", "m4a", "aif", "aiff", "wma", "ac3", "eac3",
    "ape", "wv", "mp2", "mpa", "dts", "amr", "mp4", "m4v", "mov", "mkv", "webm", "avi", "flv",
    "wmv", "ts", "mts", "m2ts", "vob", "mpg", "mpeg", "3gp", "3g2", "ogv", "rm", "rmvb",
];

/// 「用户取消」的结果。
///
/// 【为什么桌面端不追加 `reason`】桌面上用户按取消就是取消，没有别的解释；多一个
/// 恒为 `null` 的字段会让"桌面逐字节不变"这条纪律失去可验证性（对比 JSON 时会
/// 到处是噪声差异）。移动端才补 `reason`，因为那里"什么都没发生"需要一个解释。
fn canceled() -> serde_json::Value {
    let mut value = serde_json::json!({ "ok": true, "canceled": true });
    if let Some(reason) = crate::platform::dialog::unavailable_reason() {
        value["reason"] = serde_json::Value::from(reason);
    }
    value
}

fn picked(path: PathBuf) -> serde_json::Value {
    serde_json::json!({ "ok": true, "canceled": false, "path": path.display().to_string() })
}

pub(super) fn open_audio_dialog() -> serde_json::Value {
    let picked_path = FileDialog::new()
        .add_filter("Media Files", MEDIA_DIALOG_EXTENSIONS)
        .pick_file();

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
    source_path: String,
    dialog_title: String,
) -> serde_json::Value {
    use std::path::Path;

    let source_path_trimmed = source_path.trim();
    let source = Path::new(source_path_trimmed);

    let mut dialog = FileDialog::new().add_filter("Media Files", MEDIA_DIALOG_EXTENSIONS);
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

    match dialog.pick_file() {
        None => canceled(),
        Some(path) => picked(path),
    }
}

pub(super) fn open_audio_dialog_multi() -> serde_json::Value {
    let picked_paths = FileDialog::new()
        .add_filter("Media Files", MEDIA_DIALOG_EXTENSIONS)
        .pick_files();

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
/// 【为什么需要 `app`】移动端没有可以任意写的目录树，"让用户在一个沙箱路径上做
/// 文件系统导航"是没有意义的。移动端直接给出**默认落点**（系统可见的文档目录），
/// 于是"导出"在移动端是一步完成，而不是"先弹一个不能用的对话框"。
pub(super) fn pick_output_path(
    app: &tauri::AppHandle,
    suggested_name: Option<&str>,
) -> serde_json::Value {
    if crate::platform::is_mobile() {
        let name = suggested_name.unwrap_or("output.wav");
        return match crate::platform::dialog::default_export_path(app, name) {
            Ok(path) => picked(path),
            Err(error) => serde_json::json!({ "ok": false, "canceled": false, "error": error }),
        };
    }

    let picked_path = FileDialog::new()
        .add_filter("WAV", &["wav"])
        .set_file_name("output.wav")
        .save_file();

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

    match FileDialog::new().pick_folder() {
        None => canceled(),
        Some(path) => picked(path),
    }
}

pub(super) fn pick_midi_output_path(
    app: &tauri::AppHandle,
    suggested_name: Option<&str>,
) -> serde_json::Value {
    let file_type = &hifishifter_kernel::file_types::MIDI;

    if crate::platform::is_mobile() {
        let name = suggested_name.unwrap_or("export.mid");
        return match crate::platform::dialog::default_export_path(app, name) {
            Ok(path) => picked(path),
            Err(error) => serde_json::json!({ "ok": false, "canceled": false, "error": error }),
        };
    }

    let picked_path = FileDialog::new()
        .add_filter(file_type.name(), &file_type.extension_refs())
        .set_file_name("export.mid")
        .save_file();

    match picked_path {
        None => canceled(),
        Some(path) => picked(path),
    }
}

pub(super) fn open_midi_dialog() -> serde_json::Value {
    let file_type = &hifishifter_kernel::file_types::MIDI;
    let picked_path = FileDialog::new()
        .add_filter(file_type.name(), &file_type.extension_refs())
        .pick_file();

    match picked_path {
        None => canceled(),
        Some(path) => picked(path),
    }
}
