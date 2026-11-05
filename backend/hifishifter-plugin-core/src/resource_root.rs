//! 按插件格式解析资源根（D9）。
//!
//! VST3 / macOS CLAP / AU 是 bundle（`Contents/Resources`），Linux/Windows CLAP 是单文件
//! （同级 `HiFiShifter-resources`）。模型目录统一再交给 `model_store`，本模块只回答路径。

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

/// 当前进程登记后的资源根。
pub fn resources_dir() -> Result<PathBuf, String> {
    let module = MODULE_PATH
        .get()
        .ok_or_else(|| "module path is not registered".to_string())?;
    resources_dir_for(module, layout())
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

    #[test]
    fn relative_module_path_is_rejected() {
        assert!(set_module_path("relative/HiFiShifter.clap").is_err());
    }
}
