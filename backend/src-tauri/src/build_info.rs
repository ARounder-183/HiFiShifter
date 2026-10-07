//! 构建期注入的 git 信息（由 build.rs 通过 rustc-env 烘进二进制）。
//!
//! 读取器住在 `hifishifter_kernel::build_info` —— 插件要报同一份构建身份，
//! 两边各写一套必然会分叉（插件那边就因此长期缺 commit 与仓库链接）。
//! 本模块只补上「这个 crate 自己的版本号」，其余一律转发。
//!
//! 【为什么版本号在这里读】`env!("CARGO_PKG_VERSION")` 取的是**当前 crate** 的
//! 版本。内核里读它只会得到内核版本（`0.1.0`），而这里读到的才是产品版本
//! （`0.1.0-beta.15`）。插件同理读自己的，两者相等由
//! `hifishifter-plugin/tests/version_parity.rs` 保证。

pub(crate) use hifishifter_kernel::build_info::{commit_full, commit_short, dirty, repo_url};

/// 版本号（本 crate 的 Cargo.toml package.version）。
pub(crate) fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// 用户可见的版本展示串：
/// `0.1.0-beta.14` / `0.1.0-beta.14 (34d4ac89)` / `0.1.0-beta.14 (34d4ac89 dirty)`。
pub(crate) fn display_version() -> String {
    hifishifter_kernel::build_info::display_version(version())
}
