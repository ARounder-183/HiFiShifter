//! 插件引擎的构建脚本。
//!
//! 只有一件事：把构建身份（commit / 脏标志 / GitHub 仓库链接）烘进 DLL。
//!
//! 【为什么现在才有】在此之前插件完全没有 build script，于是它的「关于」对话框
//! 只能报一个 crate 版本号 —— 没有 commit、没有仓库链接，用户报 bug 时无法定位到
//! 具体构建。实现与 App 共用 `backend/build-support/git_info.rs`，避免两份逻辑分叉。
//!
//! 监视集只有 `src` 与 `native`：它们是本 crate 会进入二进制的输入。模型与前端
//! 产物由打包阶段决定、不编译进 DLL，因此不参与脏标志判定。

#[path = "../build-support/git_info.rs"]
mod git_info;

fn main() {
    git_info::emit(&["src", "native"]);
}
