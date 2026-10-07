//! 构建期注入的 git 信息（由各 crate 的 build.rs 通过 rustc-env 烘进二进制）。
//!
//! 【为什么在内核里】这段读取逻辑原先只存在于独立 App（`src-tauri/src/build_info.rs`），
//! 于是插件无从取用 —— 它的「关于」对话框因此只能报一个光秃秃的 crate 版本号，
//! 没有 commit、没有脏标志、仓库链接也永远是前端写死的兜底值。而这两个形态
//! **必须报同一个产品身份**：用户拿着插件的截图报 bug 时，开发者要能定位到同一份源码。
//!
//! 【为什么版本号要当参数传进来，而不是在这里 `env!("CARGO_PKG_VERSION")`】
//! 内核是一个**依赖**，它的 `CARGO_PKG_VERSION` 是内核自己的版本（`0.1.0`），
//! 不是产品版本（`0.1.0-beta.15`）。在内核里读它会让 App 与插件的「关于」对话框
//! 都退回内核版本 —— 那正是本模块要修掉的那类错误，只是换了个地方犯。
//! 因此版本号由**调用方**传入：App 传自己的 crate 版本，插件传自己的。
//! 两者的相等由 `hifishifter-plugin/tests/version_parity.rs` 钉死。
//!
//! 非 git 构建（如 GitHub 源码 zip）时对应变量为空，各读取器返回 None，
//! 调用方回退为纯版本号 / 固定仓库链接。

fn non_empty(value: Option<&'static str>) -> Option<&'static str> {
    value.filter(|v| !v.trim().is_empty())
}

/// 完整 commit 哈希（40 位）；非 git 构建为 None。
pub fn commit_full() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_COMMIT"))
}

/// 短 commit 哈希（≥9 位）；非 git 构建为 None。
pub fn commit_short() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_COMMIT_SHORT"))
}

/// 构建时工作区是否脏（有未提交修改）。
pub fn dirty() -> bool {
    option_env!("HIFISHIFTER_GIT_DIRTY") == Some("true")
}

/// 构建时的 GitHub 仓库主页链接（由 remote.origin.url 归一化而来）；
/// 上游不是 GitHub 或非 git 构建为 None。
pub fn repo_url() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_REPO_URL"))
}

/// 用户可见的版本展示串：
/// `0.1.0-beta.14` / `0.1.0-beta.14 (34d4ac89)` / `0.1.0-beta.14 (34d4ac89 dirty)`。
pub fn display_version(version: &str) -> String {
    match commit_short() {
        Some(short) if dirty() => format!("{version} ({short} dirty)"),
        Some(short) => format!("{version} ({short})"),
        None => version.to_string(),
    }
}

/// 「关于」对话框 / 诊断导出共用的构建身份对象。
///
/// 【为什么做成一个函数而不是让两个宿主各自拼 JSON】插件的 `get_about_info` 与
/// App 的必须**逐字段同形**：前端 `AboutDialog` 只认这一组键，任何一处少一个键
/// 都会让该字段静默变空（原先插件返回 `{name, version, host}`，于是版本号错、
/// commit 恒空、仓库链接恒为兜底值）。放在这里就没有第二处可以写歪。
///
/// `host` 是给人看的形态标识（如 `ARA plugin`），App 传空串或自身标识。
pub fn about_payload(host: &str, version: &str) -> serde_json::Value {
    serde_json::json!({
        "version": version,
        "commit": commit_full(),
        "commitShort": commit_short(),
        "dirty": dirty(),
        "repoUrl": repo_url(),
        "host": host,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构建身份对象必须带齐前端 `AboutDialog` 读取的全部键。
    ///
    /// 【为什么要钉死键名】前端读的是 `info.version` / `info.commitShort` /
    /// `info.dirty` / `info.repoUrl`，缺失不报错、只是不显示 —— 这正是插件此前
    /// 「关于」对话框少半屏信息却无人发现的原因。
    #[test]
    fn about_payload_carries_every_key_the_dialog_reads() {
        let payload = about_payload("ARA plugin", "1.2.3");
        for key in [
            "version",
            "commit",
            "commitShort",
            "dirty",
            "repoUrl",
            "host",
        ] {
            assert!(
                payload.get(key).is_some(),
                "about_payload 缺少键 `{key}`：前端 AboutDialog 会静默显示为空"
            );
        }
        assert_eq!(payload["version"], "1.2.3");
        assert_eq!(payload["host"], "ARA plugin");
    }

    /// 版本展示串始终以传入的版本号开头（commit 只是后缀）。
    #[test]
    fn display_version_starts_with_the_given_version() {
        assert!(display_version("1.2.3").starts_with("1.2.3"));
    }
}
