//! 插件文件浏览器：只对用户显式选择过的真实目录生效（`browser_roots`）。
//! 目录/元信息读取与建目录/改名/删除/显示都限制在这些根之内；音频仍由 REAPER 导入、
//! ARA 供源。唯一的例外是三个**只读媒体探测**命令（元信息 / 解码前缀 / 容器音轨表），
//! 它们额外接受用户显式指过的绝对媒体路径 —— 判据见 `browser_media_path`。
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
    /// 允许**只读探测**一个媒体文件的路径判据（元信息 / 解码前缀 / 容器音轨表）。
    ///
    /// 【与 `browser_path` 的分工】后者是"列表与写入"的边界：路径必须落在用户在
    /// 原生选择器里显式选过的目录内，且**会枚举目录**。本判据只服务三个只读命令：
    /// 它们要么读文件头、要么解前几秒 PCM、要么列容器里的音轨，**从不枚举目录**，
    /// 也从不写盘。
    ///
    /// 【为什么不能只认授权根】调用点来自用户**显式指过**的文件：原生"导入媒体"
    /// 选择器返回的路径、拖入的文件、或文件浏览器里列出的条目 —— 前两者都不在任何
    /// 授权根里。只认根会让多音轨视频的音轨选择静默退回默认音轨、拖放预览拿不到
    /// 时长（两者都是"看起来能用、其实没生效"）。
    fn browser_media_path(&self, input: &str) -> Result<PathBuf, String> {
        if let Ok(path) = self.browser_path(input) {
            return Ok(path);
        }
        let path = Path::new(input);
        if !path.is_absolute() || !hifishifter_kernel::media::is_media_extension(path) {
            return Err("an absolute media path is required outside a selected folder".into());
        }
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.is_file() {
            return Err("not a file".into());
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
                        // 隐藏判据与内核/独立 App **同一份**：Windows 的隐藏是一个文件
                        // 属性，只看点开头等于对 Windows 用户完全没生效。
                        if !hidden {
                            let is_hidden = match entry.metadata() {
                                Ok(meta) => {
                                    hifishifter_kernel::folder_scan::is_hidden_entry(&meta, &name)
                                }
                                Err(_) => name.starts_with('.'),
                            };
                            if is_hidden {
                                continue;
                            }
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
                let path =
                    self.browser_media_path(input["filePath"].as_str().ok_or("file missing")?)?;
                let info = hifishifter_kernel::audio_utils::try_read_wav_info(&path, 0)
                    .ok_or("unsupported audio metadata")?;
                Ok(
                    json!({"sampleRate":info.sample_rate,"channels":info.channels,"durationSec":info.duration_sec,"totalFrames":info.total_frames}),
                )
            }
            // ── 只读媒体探测 ──
            //
            // 这三条此前在插件里完全没有，于是：文件浏览器的试听点了没声、拖放预览
            // 拿不到时长、多音轨视频的音轨选择框永远不出现。它们都只读，判据见
            // `browser_media_path`。
            "read_audio_preview" => {
                let path =
                    self.browser_media_path(input["filePath"].as_str().ok_or("file missing")?)?;
                // 上限与独立 App 同量级：试听只需前几秒，一条 1 小时音轨整解会吃掉
                // 上 GB 内存。`maxFrames` 由前端给（缺省 10 秒 @48k）。
                let max = input["maxFrames"]
                    .as_u64()
                    .unwrap_or(480_000)
                    .clamp(1, 4_800_000) as usize;
                let (sample_rate, channels, samples) =
                    hifishifter_kernel::media::decode_media_audio_prefix_f32(&path, None, max)?;
                let channels = channels.max(1) as usize;
                let frames = (samples.len() / channels).min(max);
                let bytes = samples[..frames * channels]
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect::<Vec<u8>>();
                use base64::Engine as _;
                Ok(json!({
                    "sampleRate": sample_rate,
                    "channels": channels,
                    "pcmBase64": base64::engine::general_purpose::STANDARD.encode(&bytes),
                }))
            }
            "get_media_audio_streams" => {
                let path =
                    self.browser_media_path(input["filePath"].as_str().ok_or("file missing")?)?;
                Ok(
                    serde_json::to_value(hifishifter_kernel::media::list_audio_streams(&path)?)
                        .map_err(|e| e.to_string())?,
                )
            }
            // ── 目录导入的扫描 ──
            //
            // 【为什么准入必须是 `browser_path`】扫描会**递归枚举整个子树**并回报真实
            // 文件路径 —— 这正是只读探测不能放宽的那一半。授权根来自用户在原生选择器
            // 里显式选过的目录，与列表/写入同一条边界。
            //
            // 【为什么输出要过 `display_path`】`browser_path` 走 `canonicalize()`，路径
            // 因此带扩展长度前缀（`\\?\C:\…`）；而这些路径随后会被交回 REAPER 建 item。
            // 去掉前缀后与文件浏览器显示、与宿主选择器返回的形式一致。
            "collect_folder_media" => {
                let dirs = input["dirs"].as_array().ok_or("directory list missing")?;
                if dirs.len() > 64 {
                    return Err("directory list budget exceeded".into());
                }
                let mut granted = Vec::with_capacity(dirs.len());
                for raw in dirs {
                    let Some(text) = raw.as_str() else { continue };
                    granted.push(display_path(&self.browser_path(text)?));
                }
                let options = match input.get("options") {
                    Some(value) if !value.is_null() => Some(
                        serde_json::from_value::<
                            hifishifter_kernel::folder_scan::CollectFolderMediaOptions,
                        >(value.clone())
                        .map_err(|e| e.to_string())?,
                    ),
                    _ => None,
                };
                let scan = hifishifter_kernel::folder_scan::collect_folder_media(granted, options);
                Ok(serde_json::to_value(scan).map_err(|e| e.to_string())?)
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

    /// 最小可解码 WAV（16-bit PCM 单声道）。
    ///
    /// 【为什么要真造一个】媒体探测走的是真解码器：随便写几个字节只会得到"看起来
    /// 像 WAV"的文件，解码器会如实拒绝，测试于是测不到正例。
    fn write_minimal_wav(path: &Path, frames: usize) {
        let sample_rate = 8000_u32;
        let data_len = (frames * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for index in 0..frames {
            bytes.extend_from_slice(&((index % 97) as i16 * 100 - 4800).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// 只读媒体探测：授权根内一律可读；根外**只有媒体扩展名**可读，其余拒绝。
    ///
    /// 【为什么这条边界值得钉住】`browser_path`（列表/写入）与 `browser_media_path`
    /// （只读探测）是两条不同的口子。放宽的那条必须只放宽"读用户显式指过的媒体文件"
    /// 这一件事 —— 一旦顺手把目录枚举也放开，插件就成了任意磁盘读取的入口。
    #[test]
    fn browser_media_probe_widens_only_media_reads() {
        let (_model, owner, _) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        let dir = std::env::temp_dir().join(format!("hfs-media-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let inside = dir.join("元音.wav");
        write_minimal_wav(&inside, 800);

        // 授权根内：元信息 / 试听 / 容器音轨表都可读。
        let root = editor.grant_browser_directory(&dir).unwrap();
        assert_eq!(
            editor
                .browser_command("get_audio_file_info", &json!({"filePath":inside}))
                .unwrap()["sampleRate"],
            8000
        );
        let preview = editor
            .browser_command(
                "read_audio_preview",
                &json!({"filePath":inside,"maxFrames":100}),
            )
            .unwrap();
        assert_eq!(preview["channels"], 1);
        assert!(
            !preview["pcmBase64"].as_str().unwrap().is_empty(),
            "试听必须真的解出 PCM"
        );
        assert!(!editor
            .browser_command("get_media_audio_streams", &json!({"filePath":inside}))
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty());

        // 授权根外：媒体文件仍可探测（导入选择器与拖入的文件不在任何授权根里）。
        let outside_dir =
            std::env::temp_dir().join(format!("hfs-media-out-{}", std::process::id()));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let outside_wav = outside_dir.join("别处.wav");
        write_minimal_wav(&outside_wav, 400);
        assert!(editor
            .browser_command(
                "read_audio_preview",
                &json!({"filePath":outside_wav,"maxFrames":50})
            )
            .is_ok());
        // 非媒体扩展名在根外仍然拒绝 —— 放宽的只有媒体。
        let outside_txt = outside_dir.join("笔记.txt");
        std::fs::write(&outside_txt, b"secret").unwrap();
        assert!(editor
            .browser_command("read_audio_preview", &json!({"filePath":outside_txt}))
            .is_err());
        // 相对路径同样拒绝：宿主进程的工作目录不可预测，不能作为基准。
        assert!(editor
            .browser_command("get_audio_file_info", &json!({"filePath":"元音.wav"}))
            .is_err());
        // 目录不是文件。
        assert!(editor
            .browser_command("read_audio_preview", &json!({"filePath":root}))
            .is_err());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&outside_dir).ok();
        editor.close();
    }

    /// 目录导入的扫描：授权根内递归分组；未授权目录整条命令拒绝。
    ///
    /// 【为什么未授权是整条拒绝而不是塞进 `rejected`】`rejected` 是给用户看的
    /// "这个路径不能导入"（盘符根 / 不存在），而"这个目录你从没选过"是策略违规：
    /// 静默跳过会让调用方以为扫完了。
    #[test]
    fn browser_folder_scan_stays_inside_granted_roots() {
        let (_model, owner, _) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        let dir = std::env::temp_dir().join(format!("hfs-folder-scan-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("Takes")).unwrap();
        write_minimal_wav(&dir.join("主歌.wav"), 64);
        write_minimal_wav(&dir.join("Takes").join("take2.wav"), 64);
        std::fs::write(dir.join("说明.txt"), b"x").unwrap();

        assert!(
            editor
                .browser_command(
                    "collect_folder_media",
                    &json!({"dirs":[dir],"options":{"recursive":true}})
                )
                .is_err(),
            "未授权的目录不得被枚举"
        );

        let root = editor.grant_browser_directory(&dir).unwrap();
        let flat = editor
            .browser_command("collect_folder_media", &json!({"dirs":[root]}))
            .unwrap();
        assert_eq!(flat["totalFiles"], 1, "非递归只收本层媒体");
        assert_eq!(flat["groups"][0]["hasSubdirs"], true);
        assert!(flat["groups"][0]["paths"][0]
            .as_str()
            .unwrap()
            .ends_with("主歌.wav"));

        let deep = editor
            .browser_command(
                "collect_folder_media",
                &json!({"dirs":[root],"options":{"recursive":true}}),
            )
            .unwrap();
        assert_eq!(deep["totalFiles"], 2);
        assert_eq!(deep["groups"].as_array().unwrap().len(), 2);

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
