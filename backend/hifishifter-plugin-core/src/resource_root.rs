//! 按插件格式解析资源根（D9）。
//!
//! VST3 / macOS CLAP / AU 是 bundle（`Contents/Resources`），Linux CLAP 是单文件
//! （同级 `HiFiShifter-resources`）。Windows 安装器版是**例外**：CLAP 公共插件目录
//! 只放 `.clap` 一个文件，资源改放全机共享目录（`%PROGRAMDATA%\HiFiShifter\plugin-resources`），
//! 见 `shared_plugin_resources`。模型目录统一再交给 `model_store`，本模块只回答路径。

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// 资源布局。每个 cdylib 在模块入口登记一次。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLayout {
    /// `<bundle>/Contents/Resources`。
    BundleContents,
    /// `<module-dir>/<stem>-resources`。
    Sibling,
}

impl ResourceLayout {
    /// CLAP 的布局随 OS 变：macOS 是 bundle（`Contents/Resources`），Linux 与 Windows
    /// 是单文件模块（同级 `<stem>-resources`）。把这条规则放在这里而不是散在入口里，
    /// 是因为它同时被两处消费：模块入口登记布局、测试断言布局。
    pub const fn for_clap() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::BundleContents
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::Sibling
        }
    }
}

static LAYOUT: OnceLock<ResourceLayout> = OnceLock::new();
static MODULE_PATH: OnceLock<PathBuf> = OnceLock::new();

/// 登记本模块的资源布局；重复登记不同值返回错误，避免两个格式在同一进程里互相覆盖。
pub fn set_layout(layout: ResourceLayout) -> Result<(), String> {
    match LAYOUT.set(layout) {
        Ok(()) => Ok(()),
        Err(existing) if existing == layout => Ok(()),
        Err(existing) => Err(format!(
            "resource layout already registered as {existing:?}, refusing {layout:?}"
        )),
    }
}

/// 登记本 cdylib 的装载路径。解析规则本身保持纯函数，便于独立验证。
pub fn set_module_path(module_path: impl Into<PathBuf>) -> Result<(), String> {
    let module_path = module_path.into();
    if !module_path.is_absolute() {
        return Err("module path must be absolute".into());
    }
    match MODULE_PATH.set(module_path.clone()) {
        Ok(()) => Ok(()),
        Err(existing) if *existing == module_path => Ok(()),
        Err(_) => Err("module path already registered for this process".into()),
    }
}

/// 已登记的布局；未登记时用 bundle 形态（VST3 的历史默认）。
pub fn layout() -> ResourceLayout {
    LAYOUT
        .get()
        .copied()
        .unwrap_or(ResourceLayout::BundleContents)
}

/// 校验前端产物覆盖路径：只接受绝对路径，且必须含 `plugin.html`。
pub fn editor_override(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("editor asset override must be absolute".into());
    }
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    if !crate::has_frontend_entry(&path) {
        return Err("editor plugin.html missing".into());
    }
    Ok(path)
}

/// `HIFISHIFTER_PLUGIN_EDITOR_ASSETS` 覆盖优先于模块布局。
pub fn frontend_dir() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("HIFISHIFTER_PLUGIN_EDITOR_ASSETS") {
        return editor_override(Path::new(&path));
    }
    let folder = resources_dir()?.join("frontend");
    if !crate::has_frontend_entry(&folder) {
        return Err(format!(
            "plugin frontend assets missing: {}",
            folder.display()
        ));
    }
    Ok(folder)
}

/// 模型清单目录（包内只带 `models.json`，权重由共享模型库提供）。
pub fn models_dir() -> Result<PathBuf, String> {
    Ok(resources_dir()?.join("models"))
}

/// 由模块路径与布局解析资源根；不触碰进程级状态。
pub fn resources_dir_for(module_path: &Path, layout: ResourceLayout) -> Result<PathBuf, String> {
    match layout {
        ResourceLayout::BundleContents => module_path
            .parent()
            .and_then(Path::parent)
            .map(|contents| contents.join("Resources"))
            .ok_or_else(|| "bundle Contents unavailable".to_string()),
        ResourceLayout::Sibling => {
            let dir = module_path.parent().ok_or("module directory unavailable")?;
            let stem = module_path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("module file stem unavailable")?;
            Ok(dir.join(format!("{stem}-resources")))
        }
    }
}

/// Windows 安装器把插件资源写到哪儿。
///
/// 【为什么不在 CLAP 目录里】`C:\Program Files\Common Files\CLAP` 是**全机共享**的插件
/// 目录，规范里只应该出现 `.clap` 文件本身；别人的插件也装在这里。把我们自己的
/// `frontend/` 与 `models.json` 摆在旁边，用户看到的是一堆看不出归属的文件，删除时
/// 也无从判断该不该动。放这里之后，CLAP 目录里只有 `HiFiShifter.clap` 一个条目。
///
/// 【为什么仍然先看同级目录】手动拷贝与便携包是自包含的（`HiFiShifter.clap` +
/// 同级 `HiFiShifter-resources`），那份必须优先，否则用户把整套拷到别处就跑不起来。
/// 共享目录只是安装器版的兜底。
#[cfg(windows)]
fn shared_plugin_resources_at(programdata: &Path) -> PathBuf {
    programdata.join(crate::CLASS_NAME).join("plugin-resources")
}

#[cfg(windows)]
fn shared_plugin_resources() -> Option<PathBuf> {
    Some(shared_plugin_resources_at(Path::new(&std::env::var_os(
        "PROGRAMDATA",
    )?)))
}

/// 当前进程登记后的资源根。
///
/// 顺序：同级 `<stem>-resources`（bundle / 便携 / 手动拷贝）> Windows 全机共享目录。
pub fn resources_dir() -> Result<PathBuf, String> {
    let module = MODULE_PATH
        .get()
        .ok_or_else(|| "module path is not registered".to_string())?;
    let sibling = resources_dir_for(module, layout())?;
    if crate::has_frontend_entry(&sibling) {
        return Ok(sibling);
    }
    #[cfg(windows)]
    {
        if let Some(shared) = shared_plugin_resources() {
            if crate::has_frontend_entry(&shared) {
                return Ok(shared);
            }
        }
    }
    Ok(sibling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_layout_resolves_contents_resources_frontend() {
        let module = Path::new("/plug/HiFiShifter.vst3/Contents/x86_64-linux/HiFiShifter.so");
        let frontend = resources_dir_for(module, ResourceLayout::BundleContents)
            .unwrap()
            .join("frontend");
        assert_eq!(
            frontend,
            Path::new("/plug/HiFiShifter.vst3/Contents/Resources/frontend")
        );
    }

    #[test]
    fn clap_layout_follows_the_platform_bundle_rule() {
        // macOS 的 .clap 是 bundle，资源在 Contents/Resources；Linux/Windows 是单文件，
        // 资源在同级 <stem>-resources。写错这一条的后果不是报错，而是"窗口一片空白"。
        #[cfg(target_os = "macos")]
        assert_eq!(ResourceLayout::for_clap(), ResourceLayout::BundleContents);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(ResourceLayout::for_clap(), ResourceLayout::Sibling);
    }

    #[test]
    fn sibling_layout_never_contains_contents() {
        let module = Path::new("/usr/lib/clap/HiFiShifter.clap");
        let resources = resources_dir_for(module, ResourceLayout::Sibling).unwrap();
        assert_eq!(resources, Path::new("/usr/lib/clap/HiFiShifter-resources"));
        assert!(!resources.to_string_lossy().contains("Contents"));
    }

    #[test]
    fn relative_editor_asset_override_is_rejected() {
        let error = editor_override(Path::new("frontend/dist")).unwrap_err();
        assert!(error.contains("absolute"), "{error}");
    }

    /// 安装器版：CLAP 目录里只有 `.clap`，资源落在全机共享目录。
    /// 这条路径必须与安装器的 `HFS_CLAP_RES_DIR` 逐字一致，否则插件窗口一片空白。
    #[cfg(windows)]
    #[test]
    fn the_shared_plugin_resources_dir_matches_the_installer_layout() {
        let dir = shared_plugin_resources_at(Path::new("C:/ProgramData"));
        assert_eq!(
            dir,
            Path::new("C:/ProgramData/HiFiShifter/plugin-resources")
        );
    }

    /// 同级目录优先：便携包 / 手动拷贝是自包含的，不能被共享目录抢走。
    #[test]
    fn a_sibling_resources_dir_wins_over_the_shared_one() {
        let module = Path::new("/plug/HiFiShifter.clap");
        let sibling = resources_dir_for(module, ResourceLayout::Sibling).unwrap();
        assert_eq!(sibling, Path::new("/plug/HiFiShifter-resources"));
    }

    #[test]
    fn relative_module_path_is_rejected() {
        assert!(set_module_path("relative/HiFiShifter.clap").is_err());
    }
}
