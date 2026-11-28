//! 文件对话框门面。
//!
//! 【这个模块解决的具体问题】`rfd` 0.17 **在移动目标上编译不过** —— 它的后端只有
//! gtk3 / linux / macos / wasm / win32，没有任何 android/ios 实现，于是
//! `cargo build --target aarch64-linux-android` 在链接期才会炸，而错误信息指向的是
//! 一个与本项目无关的 crate。这不是"运行期功能缺失"，是**整包编不出来**。
//!
//! 因此 `rfd` 从全局依赖移到了 `[target.'cfg(not(any(android, ios)))'.dependencies]`，
//! 而全部调用点改为经过本门面。桌面分支是**逐字节的转发**（同一个 `rfd::FileDialog`、
//! 同一串方法调用），所以桌面上这次重构的行为差异为零。
//!
//! ## 移动端：系统选择器 + 导入即拷贝
//!
//! 移动端没有 `rfd`，但有系统选择器（SAF / UIDocumentPicker），由 [`super::native`]
//! 的原生桥发起。两者语义的差别只有一条：**系统选择器给出的是 `content://` URI，
//! 不是路径**。应用其余部分全部按路径工作（导入、工程、波形缓存、编码器），
//! 因此在边界处一次性物化：
//!
//! - **打开**（导入媒体、打开工程、导入 MIDI…）：原生桥把 URI 拷到 `cacheDir`，
//!   本模块再把结果收编进工作区（[`super::paths::adopt_import`]，同一文件系统上
//!   是一次 `rename`）。下游拿到的是一条**稳定、可长期引用**的路径。
//! - **保存**（导出音频、另存为…）：给出 `export_dir` 下的确定路径。移动端没有
//!   "任意可写目录树"，让用户在一个沙箱路径上做文件系统导航是没有意义的；
//!   需要交给别的应用时走"保存到其它位置 / 分享"（[`save_output_to_location`]）。
//! - **选目录**：返回 `export_dir`（导出目标）或工作区根（文件浏览器起点）。
//!
//! 也就是说：移动端**永远不会**走到"弹一个不存在的对话框"这条路上。

use std::path::{Path, PathBuf};

use super::paths::ImportTarget;

/// 一个文件对话框请求。
///
/// 【为什么自己存字段而不是直接持有 `rfd::FileDialog`】`rfd::FileDialog` 在移动端
/// 不存在；把字段摊平之后，本类型在所有目标上都可构造，桌面分支只是在 `pick_*`
/// 里把它"重放"成一个 `rfd::FileDialog`。
#[derive(Default)]
pub struct FileDialog {
    title: Option<String>,
    directory: Option<PathBuf>,
    file_name: Option<String>,
    filters: Vec<(String, Vec<String>)>,
    /// 移动端导入的归类目标（桌面忽略）。
    target: ImportTarget,
}

impl FileDialog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_filter(mut self, name: &str, extensions: &[&str]) -> Self {
        self.filters.push((
            name.to_string(),
            extensions.iter().map(|value| value.to_string()).collect(),
        ));
        self
    }

    pub fn set_directory(mut self, directory: impl AsRef<Path>) -> Self {
        self.directory = Some(directory.as_ref().to_path_buf());
        self
    }

    pub fn set_file_name(mut self, name: impl Into<String>) -> Self {
        self.file_name = Some(name.into());
        self
    }

    pub fn set_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// 移动端：导入结果归到哪一类目录。桌面上是空操作（桌面不拷贝）。
    pub fn set_import_target(mut self, target: ImportTarget) -> Self {
        self.target = target;
        self
    }

    pub fn pick_file(self, app: &tauri::AppHandle) -> Option<PathBuf> {
        #[cfg(desktop)]
        let picked = {
            let _ = app;
            self.into_rfd().pick_file()
        };
        #[cfg(not(desktop))]
        let picked = self.pick_mobile(app, false).and_then(|mut paths| {
            if paths.is_empty() {
                None
            } else {
                Some(paths.remove(0))
            }
        });
        picked
    }

    pub fn pick_files(self, app: &tauri::AppHandle) -> Option<Vec<PathBuf>> {
        #[cfg(desktop)]
        let picked = {
            let _ = app;
            self.into_rfd().pick_files()
        };
        #[cfg(not(desktop))]
        let picked = self
            .pick_mobile(app, true)
            .filter(|paths| !paths.is_empty());
        picked
    }

    pub fn pick_folder(self, app: &tauri::AppHandle) -> Option<PathBuf> {
        #[cfg(desktop)]
        let picked = {
            let _ = app;
            self.into_rfd().pick_folder()
        };
        #[cfg(not(desktop))]
        let picked = {
            let _ = self;
            super::paths::export_dir(app).ok()
        };
        picked
    }

    pub fn save_file(self, app: &tauri::AppHandle) -> Option<PathBuf> {
        #[cfg(desktop)]
        let picked = {
            let _ = app;
            self.into_rfd().save_file()
        };
        #[cfg(not(desktop))]
        let picked = {
            // 移动端没有"任意可写目录"，但**必须**有一个用户找得到、且我们能写的
            // 落点。`export_dir` 就是那个落点（Android 上是
            // `Android/media/<pkg>/HiFiShifter`，文件管理器可见）。
            let name = self
                .file_name
                .clone()
                .unwrap_or_else(|| "output".to_string());
            match super::paths::export_dir(app) {
                Ok(dir) => {
                    if let Err(error) = super::paths::ensure_dir(&dir) {
                        log::error!("[dialog] save_file: {error}");
                        return None;
                    }
                    Some(dir.join(name))
                }
                Err(error) => {
                    log::error!("[dialog] save_file: {error}");
                    None
                }
            }
        };
        picked
    }

    /// 移动端的打开流程：系统选择器 → 收编进工作区。
    #[cfg(not(desktop))]
    fn pick_mobile(self, app: &tauri::AppHandle, multiple: bool) -> Option<Vec<PathBuf>> {
        let extensions: Vec<String> = self
            .filters
            .iter()
            .flat_map(|(_, extensions)| extensions.iter().cloned())
            .collect();
        let picked = match super::native::pick_files(app, &extensions, multiple) {
            Ok(paths) => paths,
            Err(error) => {
                log::error!("[dialog] the system file picker failed: {error}");
                return None;
            }
        };
        if picked.is_empty() {
            return None;
        }
        let Some(root) = super::paths::import_root(app, self.target) else {
            // 没有工作区（不该发生：移动端一定有）→ 退回缓存里的临时路径，
            // 至少本次会话能用；比"什么都不返回"好。
            log::warn!("[dialog] workspace unavailable; using the cache copy as-is");
            return Some(picked);
        };
        let mut adopted = Vec::with_capacity(picked.len());
        for source in picked {
            match super::paths::adopt_import(&source, &root) {
                Ok(path) => adopted.push(path),
                Err(error) => {
                    // 单个文件失败不能连累同批次其余文件（多选导入时用户会一次选
                    // 十几首）。
                    log::error!("[dialog] could not adopt {}: {error}", source.display());
                }
            }
        }
        Some(adopted)
    }

    #[cfg(desktop)]
    fn into_rfd(self) -> rfd::FileDialog {
        let mut dialog = rfd::FileDialog::new();
        for (name, extensions) in &self.filters {
            let refs: Vec<&str> = extensions.iter().map(String::as_str).collect();
            dialog = dialog.add_filter(name, &refs);
        }
        if let Some(directory) = &self.directory {
            dialog = dialog.set_directory(directory);
        }
        if let Some(name) = &self.file_name {
            dialog = dialog.set_file_name(name.clone());
        }
        if let Some(title) = &self.title {
            dialog = dialog.set_title(title.clone());
        }
        dialog
    }
}

/// 移动端导出的默认落点。
///
/// 【为什么是"默认落点"而不是"让用户选"】移动端没有可以任意写的目录树，让用户在
/// 一个沙箱路径上做文件系统导航是没有意义的。产物统一进 [`super::paths::export_dir`]，
/// 界面告诉用户去哪找（以及"保存到其它位置"）。
pub fn default_export_path(app: &tauri::AppHandle, file_name: &str) -> Result<PathBuf, String> {
    let dir = super::paths::export_dir(app)?;
    super::paths::ensure_dir(&dir)?;
    Ok(dir.join(file_name))
}

/// 把已经写好的产物交给用户选定的位置（移动端：SAF 创建文档）。
///
/// 【为什么是"先写后拷"】本应用的导出/保存全部按路径写（`std::fs`、编码器、zip），
/// 无法直接写 `content://`。倒过来做只需一次文件到文件的拷贝，而且拷贝发生在
/// 导出**成功之后** —— 不会出现"用户选好了位置、导出却失败，目标处留下一个空文件"。
///
/// 桌面上返回 `Ok(None)`：桌面本来就有保存对话框，不需要二次交付。
pub fn save_output_to_location(
    app: &tauri::AppHandle,
    source: &Path,
    file_name: Option<&str>,
) -> Result<Option<String>, String> {
    #[cfg(desktop)]
    {
        let _ = (app, source, file_name);
        Ok(None)
    }
    #[cfg(not(desktop))]
    {
        if !crate::platform::is_mobile() {
            return Ok(None);
        }
        let name = file_name
            .map(str::to_string)
            .or_else(|| {
                source
                    .file_name()
                    .map(|value| value.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "output".to_string());
        super::native::save_to_location(app, source, &name, mime_type_for(&name))
    }
}

/// 按扩展名猜 MIME 类型。SAF 用它决定目标文档的后缀提示。
///
/// 【为什么只覆盖这几种】它是**提示**，不是校验：猜错只会让系统选择器少一个默认
/// 后缀，而导出本身的正确性不依赖它。覆盖音频/MIDI/工程/JSON 就够本应用的全部
/// 产出了。
#[cfg(not(desktop))]
fn mime_type_for(file_name: &str) -> &'static str {
    let extension = Path::new(file_name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    // 同一张表也决定"系统选择器里哪些文件可选"（见 `platform::mime`）：
    // 导出与导入各写一份，迟早会有一边漏掉某种格式。
    super::mime::for_extension(extension).unwrap_or("application/octet-stream")
}
