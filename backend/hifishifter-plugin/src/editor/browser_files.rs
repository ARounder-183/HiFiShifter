//! 插件文件浏览器：只对用户显式选择过的真实目录生效（`browser_roots`）。
//! 目录/元信息读取与建目录/改名/删除/显示都限制在这些根之内；音频仍由 REAPER 导入、
//! ARA 供源，这里不读 PCM。
use super::session::EditorSession;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

impl EditorSession {
    /// 系统选择器成功后登记规范路径，符号链接跳出所选目录仍拒绝。
    pub(super) fn grant_browser_directory(&self, path: &Path) -> Result<String, String> {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.is_dir() {
            return Err("not a real directory".into());
        }
        let text = path
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_owned();
        self.browser_roots.lock().unwrap().push(path);
        Ok(text)
    }
    /// 浏览授权只用于目录/元信息；音频仍由REAPER导入和ARA供源。
    fn browser_path(&self, input: &str) -> Result<PathBuf, String> {
        let path = Path::new(input).canonicalize().map_err(|e| e.to_string())?;
        if !self
            .browser_roots
            .lock()
            .unwrap()
            .iter()
            .any(|root| path.starts_with(root))
        {
            return Err("select this folder in the native folder picker first".into());
        }
        Ok(path)
    }
    pub(super) fn browser_command(&self, command: &str, input: &Value) -> Result<Value, String> {
        match command {
            "list_directory" | "search_files_recursive" => {
                let dir =
                    self.browser_path(input["dirPath"].as_str().ok_or("directory missing")?)?;
                let hidden = input["options"]["includeHidden"].as_bool().unwrap_or(false);
                let mut result = Vec::new();
                let mut pending = vec![dir];
                let query = input["query"].as_str().unwrap_or("").to_lowercase();
                let mut visited = std::collections::BTreeSet::new();
                while let Some(dir) = pending.pop() {
                    if !visited.insert(dir.clone()) {
                        continue;
                    }
                    if visited.len() > 4096 {
                        return Err("folder search budget exceeded".into());
                    }
                    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
                        if result.len() >= 10000 {
                            return Err("directory entry budget exceeded".into());
                        }
                        let entry = entry.map_err(|e| e.to_string())?;
                        let name = entry.file_name().to_string_lossy().into_owned();
                        if !hidden && name.starts_with('.') {
                            continue;
                        }
                        let Ok(path) = self.browser_path(&entry.path().to_string_lossy()) else {
                            continue;
                        };
                        let meta = path.metadata().map_err(|e| e.to_string())?;
                        if command == "search_files_recursive" && meta.is_dir() {
                            pending.push(path.clone());
                        }
                        if command == "search_files_recursive"
                            && !name.to_lowercase().contains(&query)
                        {
                            continue;
                        }
                        let modified = meta
                            .modified()
                            .ok()
                            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs_f64());
                        let text = path
                            .to_string_lossy()
                            .trim_start_matches(r"\\?\")
                            .to_owned();
                        result.push(json!({"name":name,"path":text,"isDir":meta.is_dir(),"size":if meta.is_file(){Some(meta.len())}else{None},
                        "extension":path.extension().map(|e|e.to_string_lossy().to_string()),"modifiedTime":modified}));
                    }
                }
                result.sort_by_key(|entry| {
                    (
                        !entry["isDir"].as_bool().unwrap_or(false),
                        entry["name"].as_str().unwrap_or("").to_lowercase(),
                    )
                });
                Ok(json!(result))
            }
            "get_audio_file_info" => {
                let path = self.browser_path(input["filePath"].as_str().ok_or("file missing")?)?;
                let info = hifishifter_kernel::audio_utils::try_read_wav_info(&path, 0)
                    .ok_or("unsupported audio metadata")?;
                Ok(
                    json!({"sampleRate":info.sample_rate,"channels":info.channels,"durationSec":info.duration_sec,"totalFrames":info.total_frames}),
                )
            }
            "stat_paths" => {
                let paths = input["paths"].as_array().ok_or("paths missing")?;
                if paths.len() > 512 {
                    return Err("path list budget exceeded".into());
                }
                Ok(json!(paths
                    .iter()
                    .map(|input| {
                        let text = input.as_str().unwrap_or("");
                        match self.browser_path(text) {
                            Ok(path) => json!({"path":text,"exists":true,"isDir":path.is_dir()}),
                            Err(_) => json!({"path":text,"exists":false,"isDir":false}),
                        }
                    })
                    .collect::<Vec<_>>()))
            }
            // ── 写操作 ──
            //
            // 【为什么插件里可以做】这些操作改的是**用户在文件浏览器里自己选中的
            // 素材目录**，不是宿主工程、也不是 ARA 授权范围。此前它们落到
            // "file browser operation unavailable"，于是浏览器里的"新建文件夹/
            // 重命名/删除"看起来能用、点了什么也不会发生。
            //
            // 边界仍是 `browser_path`：路径必须真实存在且落在用户显式选择过的根之内。
            // 新建目录的目标还不存在，所以校验的是**父目录** + 名字本身。
            "create_directory" => {
                let parent = self.browser_path(
                    input["parentDir"]
                        .as_str()
                        .ok_or("parent directory missing")?,
                )?;
                if !parent.is_dir() {
                    return Err("parent is not a directory".into());
                }
                let name = validate_entry_name(input["name"].as_str().unwrap_or_default())?;
                let target = parent.join(&name);
                if target.exists() {
                    return Err("name_exists".into());
                }
                std::fs::create_dir(&target).map_err(|e| e.to_string())?;
                Ok(json!(display_path(&target)))
            }
            "rename_path" => {
                let source = self.browser_path(input["path"].as_str().ok_or("path missing")?)?;
                let name = validate_entry_name(input["newName"].as_str().unwrap_or_default())?;
                let parent = source
                    .parent()
                    .ok_or_else(|| "cannot rename a root path".to_string())?;
                let target = parent.join(&name);
                if target == source {
                    // 名字没变：当作成功（用户可能只改了大小写，Windows 上这两者相同）。
                    return Ok(json!(display_path(&source)));
                }
                if target.exists() {
                    return Err("name_exists".into());
                }
                std::fs::rename(&source, &target).map_err(|e| e.to_string())?;
                Ok(json!(display_path(&target)))
            }
            "delete_paths" => {
                let paths = input["paths"].as_array().ok_or("paths missing")?;
                if paths.len() > 512 {
                    return Err("path list budget exceeded".into());
                }
                // 缺省是**回收站**而不是永久删除：浏览器里删的是用户素材，不在本
                // 应用的撤销栈里，永久删除不可逆（前端 Shift+Delete 才传 permanent）。
                let permanent = input["permanent"].as_bool().unwrap_or(false);
                let mut deleted = 0usize;
                let mut errors = Vec::new();
                for raw in paths {
                    let Some(text) = raw.as_str() else { continue };
                    let Ok(path) = self.browser_path(text) else {
                        errors.push(format!("not authorized: {text}"));
                        continue;
                    };
                    let result = if permanent {
                        if path.is_dir() {
                            std::fs::remove_dir_all(&path)
                        } else {
                            std::fs::remove_file(&path)
                        }
                    } else {
                        trash::delete(&path).map_err(|e| std::io::Error::other(e.to_string()))
                    };
                    match result {
                        Ok(()) => deleted += 1,
                        Err(error) => errors.push(format!("{text}: {error}")),
                    }
                }
                Ok(if errors.is_empty() {
                    json!({"ok": true, "deleted": deleted})
                } else {
                    json!({"ok": false, "deleted": deleted, "error": errors.join("; ")})
                })
            }
            "reveal_paths_in_file_manager" => {
                let paths = input["paths"].as_array().ok_or("paths missing")?;
                if paths.len() > 64 {
                    return Err("path list budget exceeded".into());
                }
                let mut opened = 0usize;
                let mut errors = Vec::new();
                for raw in paths {
                    let Some(text) = raw.as_str() else { continue };
                    let Ok(path) = self.browser_path(text) else {
                        errors.push(format!("not authorized: {text}"));
                        continue;
                    };
                    // 目录本身直接打开；文件则打开它所在目录（够用且不需要
                    // `SHOpenFolderAndSelectItems` 的额外 COM 编排）。
                    let target = if path.is_dir() {
                        path.clone()
                    } else {
                        path.parent().map(Path::to_path_buf).unwrap_or(path.clone())
                    };
                    match reveal_directory(&target) {
                        Ok(()) => opened += 1,
                        Err(error) => errors.push(format!("{text}: {error}")),
                    }
                }
                Ok(json!({"ok": true, "count": opened, "error": errors.join("; ")}))
            }
            "open_path_with_default_app" => {
                let path = self.browser_path(input["path"].as_str().ok_or("path missing")?)?;
                open_with_default_app(&path)?;
                Ok(json!({"ok": true}))
            }
            _ => Err("file browser operation unavailable in plugin".into()),
        }
    }
}

/// 显示用路径：去掉 Windows 扩展长度前缀，与 `grant_browser_directory` 同一形式。
fn display_path(path: &Path) -> String {
    path.to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_owned()
}

/// 新建/重命名时对**单个名字**的收口。
///
/// 名字来自用户输入，最终会 `join` 到一个已授权目录上：路径分隔符、`.`/`..`
/// 都能让目标跳出该目录；Windows 保留字符在多数文件系统上非法。
fn validate_entry_name(raw: &str) -> Result<String, String> {
    let name = raw.trim();
    if name.is_empty() {
        return Err("name_empty".into());
    }
    if name == "." || name == ".." {
        return Err("name_invalid".into());
    }
    if name.len() > 255 {
        return Err("name_too_long".into());
    }
    if name.chars().any(|c| {
        matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()
    }) {
        return Err("name_invalid".into());
    }
    Ok(name.to_owned())
}

/// 用系统默认程序打开一个文件（Windows）。
#[cfg(windows)]
fn open_with_default_app(path: &Path) -> Result<(), String> {
    use windows::core::HSTRING;
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let operation = HSTRING::from("open");
    let target = HSTRING::from(path.to_string_lossy().as_ref());
    // SAFETY: 两个 HSTRING 在本调用期间存活；其余参数按文档传 NULL。
    let result = unsafe { ShellExecuteW(None, &operation, &target, None, None, SW_SHOWNORMAL) };
    // 微软文档：返回值 ≤ 32 表示失败（那是一个错误码而不是 HINSTANCE）。
    if result.0 as usize <= 32 {
        return Err(format!(
            "could not open the file (code {})",
            result.0 as usize
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn open_with_default_app(_path: &Path) -> Result<(), String> {
    Err("opening files with the default app is not implemented on this platform".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 选择真实目录后能列出中文文件，未选择目录拒绝；不暴露任意文件读写。
    #[test]
    fn browser_directory_grant_lists_real_files_and_keeps_scope_boundary() {
        let (_model, owner, _) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        let dir = std::env::temp_dir().join(format!("hfs-browser-contract-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("元音.wav");
        std::fs::write(&file, b"header fixture").unwrap();
        assert!(editor
            .browser_command("list_directory", &json!({"dirPath":dir}))
            .is_err());
        let selected = editor.grant_browser_directory(&dir).unwrap();
        assert!(Path::new(&selected).is_dir());
        let list = editor
            .browser_command("list_directory", &json!({"dirPath":selected}))
            .unwrap();
        assert!(list
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["name"] == "元音.wav" && entry["isDir"] == false));
        assert!(editor
            .browser_path(&dir.parent().unwrap().to_string_lossy())
            .is_err());
        // 授权根**之外**的写操作被拒绝（以结果里的 error 形式，不是 panic）。
        let outside =
            std::env::temp_dir().join(format!("hfs-browser-outside-{}.txt", std::process::id()));
        std::fs::write(&outside, b"keep me").unwrap();
        let denied = editor
            .browser_command(
                "delete_paths",
                &json!({"paths":[outside.to_string_lossy()],"permanent":true}),
            )
            .unwrap();
        assert_eq!(denied["ok"], false, "{denied}");
        assert!(outside.exists(), "未授权路径不得被删除");
        std::fs::remove_file(&outside).unwrap();
        std::fs::remove_file(file).unwrap();
        std::fs::remove_dir(dir).unwrap();
        editor.close();
    }

    /// 建目录 / 改名 / 删除 / 名字收口：全部限制在授权根之内。
    ///
    /// 【为什么这一组必须覆盖】这些是**破坏性**操作。收口一旦漏掉，浏览器里一个
    /// `../` 就能把文件写到用户目录之外 —— 而"看起来正常返回"会让用户以为没事。
    #[test]
    fn browser_write_operations_stay_inside_the_granted_root() {
        let (_model, owner, _) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        let dir = std::env::temp_dir().join(format!("hfs-browser-write-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let root = editor.grant_browser_directory(&dir).unwrap();

        // 建目录。
        let created = editor
            .browser_command("create_directory", &json!({"parentDir":root,"name":"素材"}))
            .unwrap();
        let created_dir = PathBuf::from(created.as_str().unwrap());
        assert!(created_dir.is_dir());

        // 重名拒绝。
        assert!(editor
            .browser_command("create_directory", &json!({"parentDir":root,"name":"素材"}))
            .is_err());
        // 路径分隔符与 `..` 不得出现在名字里。
        for name in ["../escape", "a/b", "..", "", "  ", "a:b"] {
            assert!(
                editor
                    .browser_command("create_directory", &json!({"parentDir":root,"name":name}))
                    .is_err(),
                "名字 {name:?} 必须被拒绝"
            );
        }

        // 改名。
        let renamed = editor
            .browser_command(
                "rename_path",
                &json!({"path":created_dir.to_string_lossy(),"newName":"素材2"}),
            )
            .unwrap();
        let renamed_dir = PathBuf::from(renamed.as_str().unwrap());
        assert!(renamed_dir.is_dir());
        assert!(!created_dir.exists());

        // 写一个文件进去，再永久删除它。
        let file = renamed_dir.join("a.txt");
        std::fs::write(&file, b"x").unwrap();
        let deleted = editor
            .browser_command(
                "delete_paths",
                &json!({"paths":[file.to_string_lossy()],"permanent":true}),
            )
            .unwrap();
        assert_eq!(deleted["ok"], true);
        assert_eq!(deleted["deleted"], 1);
        assert!(!file.exists());

        // 未授权路径改名同样拒绝。
        assert!(editor
            .browser_command(
                "rename_path",
                &json!({"path":std::env::temp_dir().to_string_lossy(),"newName":"nope"})
            )
            .is_err());

        std::fs::remove_dir_all(&dir).ok();
        editor.close();
    }
}

#[cfg(windows)]
/// 原生真实文件夹选择器，不用选文件后取父目录，也不接受虚拟Shell位置。
pub(super) fn pick_directory(
    hwnd: windows::Win32::Foundation::HWND,
) -> Result<Option<PathBuf>, String> {
    use windows::Win32::System::Com::{CoCreateInstance, CoTaskMemFree, CLSCTX_INPROC_SERVER};
    use windows::Win32::UI::Shell::{
        FileOpenDialog, IFileOpenDialog, FOS_FORCEFILESYSTEM, FOS_PATHMUSTEXIST, FOS_PICKFOLDERS,
        SIGDN_FILESYSPATH,
    };
    unsafe {
        let dialog: IFileOpenDialog = CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER)
            .map_err(|e| e.to_string())?;
        dialog
            .SetOptions(
                dialog.GetOptions().map_err(|e| e.to_string())?
                    | FOS_PICKFOLDERS
                    | FOS_FORCEFILESYSTEM
                    | FOS_PATHMUSTEXIST,
            )
            .map_err(|e| e.to_string())?;
        if let Err(error) = dialog.Show(Some(hwnd)) {
            if error.code().0 as u32 == 0x800704c7 {
                return Ok(None);
            }
            return Err(error.to_string());
        }
        let item = dialog.GetResult().map_err(|e| e.to_string())?;
        let value = item
            .GetDisplayName(SIGDN_FILESYSPATH)
            .map_err(|e| e.to_string())?;
        let text = value.to_string().map_err(|e| e.to_string());
        CoTaskMemFree(Some(value.0.cast()));
        Ok(Some(PathBuf::from(text?)))
    }
}

/// 原生"另存为"对话框（诊断导出用），与 [`pick_directory`] 同一套 COM 模式。
///
/// 【为什么不用 rfd】插件不引入 Tauri，而 `IFileSaveDialog` 就在同一个
/// `Win32_UI_Shell` 特性里 —— 只为弹一次窗多拉一个对话框库不值得。
/// 取消（`0x800704c7`）返回 `Ok(None)`，与文件夹选择器同一约定。
#[cfg(windows)]
pub(super) fn pick_save_path(
    hwnd: windows::Win32::Foundation::HWND,
    file_name: &str,
    extension: &str,
) -> Result<Option<PathBuf>, String> {
    use windows::core::HSTRING;
    use windows::Win32::System::Com::{CoCreateInstance, CoTaskMemFree, CLSCTX_INPROC_SERVER};
    use windows::Win32::UI::Shell::{
        FileSaveDialog, IFileSaveDialog, FOS_FORCEFILESYSTEM, FOS_OVERWRITEPROMPT,
        SIGDN_FILESYSPATH,
    };
    unsafe {
        let dialog: IFileSaveDialog = CoCreateInstance(&FileSaveDialog, None, CLSCTX_INPROC_SERVER)
            .map_err(|e| e.to_string())?;
        dialog
            .SetOptions(
                dialog.GetOptions().map_err(|e| e.to_string())?
                    | FOS_FORCEFILESYSTEM
                    | FOS_OVERWRITEPROMPT,
            )
            .map_err(|e| e.to_string())?;
        dialog
            .SetFileName(&HSTRING::from(file_name))
            .map_err(|e| e.to_string())?;
        dialog
            .SetDefaultExtension(&HSTRING::from(extension))
            .map_err(|e| e.to_string())?;
        if let Err(error) = dialog.Show(Some(hwnd)) {
            // 0x800704c7 = ERROR_CANCELLED。
            if error.code().0 as u32 == 0x800704c7 {
                return Ok(None);
            }
            return Err(error.to_string());
        }
        let item = dialog.GetResult().map_err(|e| e.to_string())?;
        let value = item
            .GetDisplayName(SIGDN_FILESYSPATH)
            .map_err(|e| e.to_string())?;
        let text = value.to_string().map_err(|e| e.to_string());
        CoTaskMemFree(Some(value.0.cast()));
        Ok(Some(PathBuf::from(text?)))
    }
}

/// 非 Windows 平台暂不实现 —— 明确报错，不让调用方以为用户取消了。
#[cfg(not(windows))]
pub(super) fn pick_save_path(
    _hwnd: (),
    _file_name: &str,
    _extension: &str,
) -> Result<Option<PathBuf>, String> {
    Err("the save dialog is not implemented on this platform".into())
}

/// 在系统文件管理器中打开一个目录（Windows）。
///
/// 【为什么这不是"宿主限制"】插件**已经**在 REAPER 进程里开过原生文件夹选择器
/// （[`pick_directory`]，用的是同一个 `Win32_UI_Shell` 特性）。此前 Help 菜单只回报
/// 路径、不打开，理由写的是"不该在宿主进程里拉起外部程序" —— 那条策略与已经在跑的
/// 文件夹选择器自相矛盾，而且让用户不得不手抄路径。
///
/// 因此这里真的打开它；失败时调用方仍会把路径回报给用户（可选中、可复制）。
#[cfg(windows)]
pub(super) fn reveal_directory(path: &Path) -> Result<(), String> {
    use windows::core::HSTRING;
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let operation = HSTRING::from("open");
    let target = HSTRING::from(path.to_string_lossy().as_ref());
    // SAFETY: 两个 HSTRING 在本调用期间存活；其余参数按文档传 NULL。
    let result = unsafe { ShellExecuteW(None, &operation, &target, None, None, SW_SHOWNORMAL) };
    // 微软文档：返回值 ≤ 32 表示失败（那是一个错误码而不是 HINSTANCE）。
    if result.0 as usize <= 32 {
        return Err(format!(
            "could not open the file manager (code {})",
            result.0 as usize
        ));
    }
    Ok(())
}

/// 非 Windows 平台暂不实现 —— 明确报错，不让调用方以为打开成功了。
#[cfg(not(windows))]
pub(super) fn reveal_directory(path: &Path) -> Result<(), String> {
    let _ = path;
    Err("opening the file manager is not implemented on this platform".into())
}
