//! 构建期注入的 git 信息（由 build.rs 通过 rustc-env 烘进二进制）。
//!
//! 非 git 构建（如 GitHub 源码 zip）时对应变量为空，各读取器返回 None，
//! 调用方回退为纯版本号 / 固定仓库链接。

/// 版本号（Cargo.toml 的 package.version）。
pub(crate) fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn non_empty(value: Option<&'static str>) -> Option<&'static str> {
    value.filter(|v| !v.trim().is_empty())
}

/// 完整 commit 哈希（40 位）；非 git 构建为 None。
pub(crate) fn commit_full() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_COMMIT"))
}

/// 短 commit 哈希（≥9 位）；非 git 构建为 None。
pub(crate) fn commit_short() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_COMMIT_SHORT"))
}

/// 构建时工作区是否脏（有未提交修改）。
pub(crate) fn dirty() -> bool {
    option_env!("HIFISHIFTER_GIT_DIRTY") == Some("true")
}

/// 构建时的 GitHub 仓库主页链接（由 remote.origin.url 归一化而来）；
/// 上游不是 GitHub 或非 git 构建为 None。
pub(crate) fn repo_url() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_REPO_URL"))
}

/// 用户可见的版本展示串：
/// `0.1.0-beta.14` / `0.1.0-beta.14 (34d4ac89)` / `0.1.0-beta.14 (34d4ac89 dirty)`。
pub(crate) fn display_version() -> String {
    match commit_short() {
        Some(short) if dirty() => format!("{} ({short} dirty)", version()),
        Some(short) => format!("{} ({short})", version()),
        None => version().to_string(),
    }
}
