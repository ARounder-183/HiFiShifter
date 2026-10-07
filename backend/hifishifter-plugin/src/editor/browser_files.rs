//! 插件文件浏览器只读目录；用户显式选择的真实目录才授权，不开放删除/重命名或任意磁盘读取。
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
            _ => Err("file browser operation unavailable in plugin".into()),
        }
    }
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
        assert!(editor
            .browser_command("delete_paths", &json!({"paths":[file]}))
            .is_err());
        std::fs::remove_file(file).unwrap();
        std::fs::remove_dir(dir).unwrap();
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
