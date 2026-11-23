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
//! ## 移动端为什么没有原生选择器
//!
//! 系统选择器在移动端当然存在（SAF / UIDocumentPicker），但把它的结果交回 Rust 需要
//! 一个原生桥（Tauri 移动插件 + Kotlin/Swift）。而**导入**这条路上根本不需要它：
//! WebView 的 `<input type="file">` 由系统接管，选完之后 JS 直接拿到 `File` 对象的
//! 字节，随后走既有的 `import_audio_bytes` 命令 —— 零原生代码、零权限声明。
//!
//! 于是移动端的策略是：
//! - **导入**：前端用 `<input type="file">`（见 `services/mobileFilePicker.ts`），
//!   根本不调用本门面；
//! - **导出**：`pick_output_path` 在移动端直接返回 `export_dir` 里的默认路径
//!   （[\`default_export_path\`]），文件落进系统可见的文档目录；
//! - **选目录**：返回工作区根（`media_root`），文件浏览器因此变成工作区浏览器。
//!
//! 也就是说：移动端**永远不会**走到"弹一个不存在的对话框"这条路上 —— 门面在移动端
//! 的 `pick_*` 返回 `None` 只是最后一道防线，而不是正常路径。

use std::path::{Path, PathBuf};

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

    pub fn pick_file(self) -> Option<PathBuf> {
        #[cfg(desktop)]
        let picked = self.into_rfd().pick_file();
        #[cfg(not(desktop))]
        let picked = {
            self.log_unsupported("pick_file");
            None
        };
        picked
    }

    pub fn pick_files(self) -> Option<Vec<PathBuf>> {
        #[cfg(desktop)]
        let picked = self.into_rfd().pick_files();
        #[cfg(not(desktop))]
        let picked = {
            self.log_unsupported("pick_files");
            None
        };
        picked
    }

    pub fn pick_folder(self) -> Option<PathBuf> {
        #[cfg(desktop)]
        let picked = self.into_rfd().pick_folder();
        #[cfg(not(desktop))]
        let picked = {
            self.log_unsupported("pick_folder");
            None
        };
        picked
    }

    pub fn save_file(self) -> Option<PathBuf> {
        #[cfg(desktop)]
        let picked = self.into_rfd().save_file();
        #[cfg(not(desktop))]
        let picked = {
            self.log_unsupported("save_file");
            None
        };
        picked
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

    #[cfg(not(desktop))]
    fn log_unsupported(&self, action: &str) {
        // 【为什么是 warn 而不是 error】移动端的正常路径根本不调用这里（见模块头）。
        // 走到这里说明某个调用点漏了能力判断 —— 值得留痕，但不是用户可见的故障。
        log::warn!(
            "[dialog] {action} is not available on this surface (mobile uses the web picker); request dropped"
        );
    }
}

/// 取消原因。
///
/// 桌面返回 `None`：用户按了取消就是取消，没有别的解释，命令的 JSON 里也不该多出
/// 一个字段（那会破坏"桌面逐字节不变"）。移动端返回一个原因键，界面才能说
/// "这个入口在移动端走系统选择器"而不是静默什么都不发生。
pub fn unavailable_reason() -> Option<&'static str> {
    if crate::platform::is_mobile() {
        // 理由键由 `platform::mod` 统一定义（前端 `surfaceCapabilities.ts` 的白名单
        // 与它逐字一致，由 `surfaceProfileContract.test.ts` 守住）。
        Some(super::REASON_USE_SYSTEM_PICKER)
    } else {
        None
    }
}

/// 移动端导出的默认落点。
///
/// 【为什么是"默认落点"而不是"让用户选"】移动端没有可以任意写的目录树，让用户在
/// 一个沙箱路径上做文件系统导航是没有意义的。产物统一进文档目录，界面告诉用户
/// 去哪找（以及"用其它应用打开"）。
pub fn default_export_path(app: &tauri::AppHandle, file_name: &str) -> Result<PathBuf, String> {
    let dir = super::paths::export_dir(app)?;
    super::paths::ensure_dir(&dir)?;
    Ok(dir.join(file_name))
}
