//! 版本一致性守卫：插件 crate 必须与独立 App 共用同一个产品版本号。
//!
//! 【为什么需要这条测试】插件的版本是**用户可见**的：ARA 编辑器的「关于」对话框
//! 报它，安装器把它写进「程序和功能」的 DisplayVersion。而
//! `scripts/release/set-version.ps1` 长期只改 App 那四个文件，插件一直停在 `0.1.0` ——
//! 于是同一个产品在两个形态里报出两个版本号，安装器记的版本与 App 也对不上。
//!
//! 【为什么用测试而不是只改脚本】改脚本只解决"这次改对了"；漏改一次不会有任何
//! 症状，直到用户在关于对话框里看到错误的版本。这条测试让漏改立刻变红。
//!
//! 判据（两层）：
//! 1. `build.rs` 注入的 `HFS_PRODUCT_VERSION`（两个形态「关于」对话框的实际取值）
//!    必须等于**单一来源** `src-tauri/tauri.conf.json`；
//! 2. 其余副本（`tauri.conf.dist-dev.json` / 两个 `Cargo.toml` / `package.json` /
//!    `package-lock.json`）必须都等于插件的 `CARGO_PKG_VERSION`。
//!
//! 任何一处不一致都说明 `scripts/release/set-version.ps1` 漏了一个文件。

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // tests/ -> crate 根 -> backend/ -> 仓库根
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repository root")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// 取文件里第一个版本赋值，兼容 JSON（`"version": "x"`）与 TOML（`version = "x"`）。
///
/// 两个 `[package]` 段与 `tauri.conf.json` / `package.json` 都把版本放在文件靠前处，
/// 且都是"第一个出现的 version 键"，因此不需要真正的解析器。
fn first_version_assignment(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim().trim_start_matches('"');
        let Some(rest) = line.strip_prefix("version") else {
            continue;
        };
        // JSON 的键名以引号收尾，TOML 的键名直接跟分隔符。
        let rest = rest.trim_start().trim_start_matches('"').trim_start();
        let Some(rest) = rest.strip_prefix('=').or_else(|| rest.strip_prefix(':')) else {
            continue;
        };
        // JSON 的值以 `",` 收尾，TOML 的以 `"` 收尾 —— 一并削掉。
        let value = rest.trim().trim_matches(|c| c == '"' || c == ',');
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

#[test]
fn every_version_source_agrees_with_the_plugin_crate() {
    let root = repo_root();
    let plugin_version = env!("CARGO_PKG_VERSION");

    // 产品版本的**单一来源**是 `src-tauri/tauri.conf.json`；`build.rs` 把它注入为
    // `HFS_PRODUCT_VERSION`，两个形态的「关于」对话框都读这一个值。这里先钉住注入值
    // 与权威文件一致，再钉住其余副本 —— 任何一处漏改都会立刻变红。
    let authority = {
        let text = read(&root.join("backend/src-tauri/tauri.conf.json"));
        first_version_assignment(&text).expect("tauri.conf.json must declare a version")
    };
    assert_eq!(
        env!("HFS_PRODUCT_VERSION"),
        authority,
        "build.rs 注入的 HFS_PRODUCT_VERSION 必须来自 tauri.conf.json（产品版本的单一来源）"
    );

    let sources = [
        "backend/src-tauri/tauri.conf.json",
        "backend/src-tauri/tauri.conf.dist-dev.json",
        "backend/src-tauri/Cargo.toml",
        "frontend/package.json",
        "frontend/package-lock.json",
    ];

    let mut mismatches = Vec::new();
    for relative in sources {
        let path = root.join(relative);
        let text = read(&path);
        match first_version_assignment(&text) {
            Some(version) if version == plugin_version => {}
            Some(version) => mismatches.push(format!("{relative} = {version}")),
            None => mismatches.push(format!("{relative} = <version not found>")),
        }
    }

    assert!(
        mismatches.is_empty(),
        "版本号不一致：hifishifter-plugin = {plugin_version}，但 {}。\
         运行 scripts/release/set-version.ps1 -Version {plugin_version} 同步全部来源。",
        mismatches.join("、")
    );
}
