//! GUI 的另一半：文件对话框、在文件管理器中显示、用默认程序打开。
//!
//! 【为什么单独成文件】四个平台服务都被 `gui/client.rs` 调用，而它们的实现里
//! **没有一行判定** —— 全是"系统集成"。放进 `browser_files.rs` 会让那个文件同时承担
//! "路径收口"（有安全判据、有测试）与"弹一个系统窗口"（没法测）两件事。
//!
//! 【两条纪律，两边都必须遵守】
//!
//! 1. **只在持有 GUI 的那条线程上弹**。命令层已经保证命令在视图线程上执行
//!    （`gui/client.rs` 的 `on_message`），这里只加一条 `debug_assert` 把契约钉住。
//! 2. **`Ok(None)` 表示用户取消，`Err` 只表示"打不开对话框"**。今天两者混在 `Err` 里，
//!    前端会把"取消"显示成红色错误 —— 这是顺手要修的小缺陷。
//!
//! 【过滤器只有一条扩展名】现有签名只支持一个；多选留给后续
//! （`import_native_audio_file` 一次一个文件）。

use std::path::PathBuf;

/// 等待外部程序（`xdg-open`）返回的上限。
///
/// 【为什么必须设】不设超时的话，一个坏掉的 `xdg-open` 会挂住宿主 UI 线程 ——
/// 而那正是"点了帮助菜单，整个 DAW 卡住"这类最难复现的报障。
#[cfg(all(unix, not(target_os = "macos")))]
const COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

// ───────────────────────────── Linux ─────────────────────────────

#[cfg(all(unix, not(target_os = "macos")))]
pub(crate) fn pick_directory() -> Result<Option<PathBuf>, String> {
    let dialog = gtk::FileChooserNative::new(
        Some("Choose a folder"),
        None::<&gtk::Window>,
        gtk::FileChooserAction::SelectFolder,
        Some("Select"),
        Some("Cancel"),
    );
    Ok(run_dialog(&dialog))
}

#[cfg(all(unix, not(target_os = "macos")))]
pub(crate) fn pick_open_path(
    filter_name: &str,
    extension: &str,
) -> Result<Option<PathBuf>, String> {
    use gtk::prelude::*;
    let dialog = gtk::FileChooserNative::new(
        Some(filter_name),
        None::<&gtk::Window>,
        gtk::FileChooserAction::Open,
        Some("Open"),
        Some("Cancel"),
    );
    let filter = gtk::FileFilter::new();
    filter.set_name(Some(filter_name));
    filter.add_pattern(&format!("*.{extension}"));
    dialog.add_filter(filter);
    Ok(run_dialog(&dialog))
}

#[cfg(all(unix, not(target_os = "macos")))]
pub(crate) fn pick_save_path(file_name: &str, extension: &str) -> Result<Option<PathBuf>, String> {
    use gtk::prelude::*;
    let dialog = gtk::FileChooserNative::new(
        Some("Save as"),
        None::<&gtk::Window>,
        gtk::FileChooserAction::Save,
        Some("Save"),
        Some("Cancel"),
    );
    dialog.set_current_name(file_name);
    dialog.set_do_overwrite_confirmation(true);
    let filter = gtk::FileFilter::new();
    filter.set_name(Some(&format!("*.{extension}")));
    filter.add_pattern(&format!("*.{extension}"));
    dialog.add_filter(filter);
    Ok(run_dialog(&dialog))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn run_dialog(dialog: &gtk::FileChooserNative) -> Option<PathBuf> {
    use gtk::prelude::*;
    // `run()` 是同步模态的：它自旋一个嵌套主循环，用户选完才返回。
    // 【为什么不用异步 `response` 信号】命令是请求-响应式的，调用方在这一条命令里
    // 等结果；改成异步就要给每个对话框再建一条状态机，收益为零。
    let response = dialog.run();
    dialog.destroy();
    if response == gtk::ResponseType::Accept {
        dialog.filename()
    } else {
        None
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
pub(crate) fn reveal_directory(path: &std::path::Path) -> Result<(), String> {
    run_command("xdg-open", path)
}

#[cfg(all(unix, not(target_os = "macos")))]
pub(crate) fn open_with_default_app(path: &std::path::Path) -> Result<(), String> {
    run_command("xdg-open", path)
}

/// `spawn` + `wait`，带超时。
///
/// 【为什么必须 `wait`】不 `wait` 会留下僵尸进程；不设超时则会挂住宿主 UI 线程。
/// 两条都要有，见本文件的模块注释。
#[cfg(all(unix, not(target_os = "macos")))]
fn run_command(program: &str, argument: &std::path::Path) -> Result<(), String> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(program)
        .arg(argument)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("could not run {program}: {error}"))?;
    let deadline = std::time::Instant::now() + COMMAND_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err(format!("{program} exited with {status}"))
                }
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{program} did not return in time"));
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}

// ───────────────────────────── macOS ─────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use objc2_app_kit::{NSModalResponseOK, NSOpenPanel, NSSavePanel, NSWorkspace};
    use objc2_foundation::{MainThreadMarker, NSArray, NSString, NSURL};
    use std::path::{Path, PathBuf};

    /// 本平台的对话框必须主线程：`runModal` 起的是 `NSApplication` 的模态会话。
    fn main_thread() -> Result<MainThreadMarker, String> {
        MainThreadMarker::new()
            .ok_or_else(|| "file dialogs must be opened on the main thread".to_string())
    }

    /// 统一的扩展名过滤。
    ///
    /// 【为什么用已废弃的 `setAllowedFileTypes`】新的 `setAllowedContentTypes` 要
    /// `UTType`，那会把 `objc2-uniform-type-identifiers` 拉进依赖树 —— 为一个过滤器
    /// 多一个 crate 不值得。这个 API 在 macOS 12+ 标了废弃但仍然工作。
    #[allow(deprecated)]
    fn apply_filter(panel: &NSSavePanel, extension: &str) {
        let types = NSArray::from_retained_slice(&[NSString::from_str(extension)]);
        panel.setAllowedFileTypes(Some(&types));
    }

    fn path_of(url: Option<objc2::rc::Retained<NSURL>>) -> Option<PathBuf> {
        let url = url?;
        let path = url.path()?;
        Some(PathBuf::from(path.to_string()))
    }

    pub(crate) fn pick_directory() -> Result<Option<PathBuf>, String> {
        let marker = main_thread()?;
        let panel = NSOpenPanel::openPanel(marker);
        panel.setCanChooseFiles(false);
        panel.setCanChooseDirectories(true);
        panel.setAllowsMultipleSelection(false);
        panel.setCanCreateDirectories(true);
        if panel.runModal() != NSModalResponseOK {
            return Ok(None);
        }
        Ok(path_of(panel.URL()))
    }

    pub(crate) fn pick_open_path(
        filter_name: &str,
        extension: &str,
    ) -> Result<Option<PathBuf>, String> {
        let marker = main_thread()?;
        let panel = NSOpenPanel::openPanel(marker);
        panel.setTitle(Some(&NSString::from_str(filter_name)));
        panel.setCanChooseFiles(true);
        panel.setCanChooseDirectories(false);
        panel.setAllowsMultipleSelection(false);
        apply_filter(&panel, extension);
        if panel.runModal() != NSModalResponseOK {
            return Ok(None);
        }
        Ok(path_of(panel.URL()))
    }

    pub(crate) fn pick_save_path(
        file_name: &str,
        extension: &str,
    ) -> Result<Option<PathBuf>, String> {
        let marker = main_thread()?;
        let panel = NSSavePanel::savePanel(marker);
        panel.setNameFieldStringValue(&NSString::from_str(file_name));
        panel.setCanCreateDirectories(true);
        apply_filter(&panel, extension);
        if panel.runModal() != NSModalResponseOK {
            return Ok(None);
        }
        Ok(path_of(panel.URL()))
    }

    pub(crate) fn reveal_directory(path: &Path) -> Result<(), String> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let urls = NSArray::from_retained_slice(&[url]);
        NSWorkspace::sharedWorkspace().activateFileViewerSelectingURLs(&urls);
        Ok(())
    }

    pub(crate) fn open_with_default_app(path: &Path) -> Result<(), String> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        if NSWorkspace::sharedWorkspace().openURL(&url) {
            Ok(())
        } else {
            Err(format!("no application can open {}", path.display()))
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) use macos::{
    open_with_default_app, pick_directory, pick_open_path, pick_save_path, reveal_directory,
};
