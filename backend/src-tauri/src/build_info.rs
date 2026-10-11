//! 构建期注入的 git 信息（由 build.rs 通过 rustc-env 烘进二进制）。
//!
//! 读取器住在 `hifishifter_kernel::build_info` —— 插件要报同一份构建身份，
//! 两边各写一套必然会分叉（插件那边就因此长期缺 commit 与仓库链接）。
//! 本模块只补上「产品版本号」，其余一律转发。
//!
//! 【为什么版本号不是 `CARGO_PKG_VERSION`】产品版本的**单一来源**是
//! `src-tauri/tauri.conf.json`，由 `build.rs` 注入为 `HFS_PRODUCT_VERSION`。读本 crate
//! 的 `CARGO_PKG_VERSION` 会引入第二份副本 —— 独立 App 与插件的「关于」对话框各读
//! 自己那份，改一处就会报出不同版本（已经发生过一次）。其余副本由
//! `hifishifter-plugin/tests/version_parity.rs` 钉住。

pub(crate) use hifishifter_kernel::build_info::{commit_full, commit_short, dirty, repo_url};

/// 产品版本号（单一来源：`src-tauri/tauri.conf.json`）。
pub(crate) fn version() -> &'static str {
    env!("HFS_PRODUCT_VERSION")
}

/// 用户可见的版本展示串：
/// `0.1.0-beta.14` / `0.1.0-beta.14 (34d4ac89)` / `0.1.0-beta.14 (34d4ac89 dirty)`。
pub(crate) fn display_version() -> String {
    hifishifter_kernel::build_info::display_version(version())
}
