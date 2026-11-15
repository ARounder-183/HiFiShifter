//! macOS 原生菜单：只保留 App 与最小 Edit 标准角色。
//!
//! 项目完整菜单由窗口内的 React MenuBar 提供；这里不复制任何项目命令。
//! 关闭 Tauri 默认菜单后，Cmd+Q / Hide / About 与文本框复制粘贴仍需要
//! 这些标准角色。

use tauri::menu::{Menu, MenuBuilder, SubmenuBuilder};
use tauri::{AppHandle, Runtime};

/// 原生菜单允许出现的顶层角色（测试用清单）。
pub(crate) const NATIVE_TOP_LEVEL_ROLES: &[&str] = &["app", "edit"];

pub(crate) fn build<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let app_menu = SubmenuBuilder::new(app, "HiFiShifter")
        .about(None)
        .separator()
        .services()
        .separator()
        .hide()
        .hide_others()
        .separator()
        .quit()
        .build()?;
    let edit_menu = SubmenuBuilder::new(app, "Edit")
        .undo()
        .redo()
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .build()?;
    MenuBuilder::new(app)
        .item(&app_menu)
        .item(&edit_menu)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_menu_contains_only_app_and_edit_roles() {
        assert_eq!(NATIVE_TOP_LEVEL_ROLES, &["app", "edit"]);
        for forbidden in ["file", "view", "window", "help", "track", "options"] {
            assert!(!NATIVE_TOP_LEVEL_ROLES.contains(&forbidden));
        }
    }
}
