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

/// 构建期 git 探测的结果。
///
/// 取值（由 `build-support/git_info.rs` 的 `emit` 写入）：
/// - `Some("git")`        —— 注入路径通着，且这次构建拿到了 commit；
/// - `Some("no-git")`     —— `git` 可执行文件不在 PATH 上（**构建环境**问题）；
/// - `Some("not-a-repo")` —— git 在，但当前目录不是仓库（源码 zip，**预期**行为）；
/// - `None`               —— 没有任何 build script 注入过这个变量。
///
/// 【为什么需要它】`commit_full()` 为 `None` 有两种成因，处置完全不同：
/// "这台机器没装 git"不该让门禁测试变红，而"build script 根本没接上"必须让它变红。
/// 只有把"探测跑过没有"单独记下来，测试才能区分二者 —— 而"内核 crate 从来没有
/// build script"正是本仓库真实发生过的缺陷。
pub fn probe() -> Option<&'static str> {
    non_empty(option_env!("HIFISHIFTER_GIT_PROBE"))
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

    /// **git 构建身份必须真的到达「内核」这个编译单元。**
    ///
    /// 【为什么这条测试是必需的】`build_info` 住在内核里，而 `rustc-env` 是**按 crate**
    /// 注入的。此前只有 app 与插件的 build script 调 `git_info::emit`，内核没有 ——
    /// 于是这四个变量在**读取者所在的编译单元**里恒未定义，独立 App 与插件的「关于」
    /// 对话框、日志会话头、诊断包里的 commit 全部为空。
    ///
    /// 上一条测试只钉住 `about_payload` 的**键名**（"字段都在"），钉不住**值**
    /// （"字段里有东西"）—— 这正是那个缺陷能长期存活的原因：界面不报错，只是少半屏。
    ///
    /// 【为什么要环境变量豁免】从 GitHub 源码 zip 构建时没有 `.git`，
    /// `git_info::emit` 按设计注入空值、`display_version` 回退纯版本号。那种构建里
    /// "没有 commit"是**正确行为**，不是缺陷。CI 与本地都在仓库里构建，所以默认断言非空。
    #[test]
    fn the_git_identity_reaches_the_kernel_crate() {
        // 【第一步：接线必须在】`None` 只可能意味着**没有任何 build script 注入过**
        // 这个变量 —— 也就是内核的 build.rs 根本没调 `git_info::emit`。
        // 这一条与环境无关，任何机器上都该成立，所以它是最强的那道闸。
        let probe = probe().expect(
            "HIFISHIFTER_GIT_PROBE 未定义 —— 内核 crate 的 build.rs 没有调用 \
             git_info::emit（这正是 commit 长期为空的根因）",
        );

        match probe {
            // 构建机没装 git / git 不在 PATH：环境问题，不判失败。
            // 但仍要确认"空值"这条路径自洽（否则运行时会显示半个身份）。
            "no-git" => {
                assert!(commit_full().is_none(), "git 不可用时不得有 commit");
                assert!(!dirty(), "git 不可用时不得声称工作区是脏的");
            }
            // 源码 zip：预期行为，运行期回退纯版本号。
            "not-a-repo" => {
                assert!(commit_full().is_none(), "非 git 构建不得有 commit");
                assert_eq!(
                    display_version("9.9.9"),
                    "9.9.9",
                    "非 git 构建的展示串必须回退成纯版本号"
                );
            }
            // 正常 git 构建：四个字段必须齐。
            "git" => {
                let full = commit_full().expect("probe=git 但 commit 为空");
                let short = commit_short().expect("probe=git 但短 commit 为空");
                assert!(
                    full.starts_with(short),
                    "短 commit 必须是完整 commit 的前缀（两者应由同一次 rev-parse 派生）"
                );
                assert!(
                    display_version("9.9.9").starts_with("9.9.9 ("),
                    "git 构建的展示串必须带上 commit 后缀"
                );
            }
            other => panic!("未知的 git probe 取值 {other:?}"),
        }
    }
}
