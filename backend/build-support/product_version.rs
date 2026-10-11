//! 产品版本的**单一来源**：`src-tauri/tauri.conf.json`。
//!
//! 【为什么需要它】产品版本此前散在 `tauri.conf.json` / `tauri.conf.dist-dev.json` /
//! 两个 `Cargo.toml` / `package.json` / `package-lock.json` 里，而两个形态的「关于」
//! 对话框各自读**本 crate** 的 `CARGO_PKG_VERSION` —— 只改一处就会让插件与独立 App
//! 报出不同版本（已经真实发生过一次）。这里把"报告用的版本"收敛成**一个文件**：
//! 两个 build script 都从这里读，其余副本由 `tests/version_parity.rs` 钉住。
//!
//! 【为什么是源码级 include 而不是 `use`】与 `git_info.rs` 同理：build script 在
//! 所属 crate 之前编译，不能依赖 workspace 里的任何 crate。两个 `build.rs` 各自用
//! `#[path = "…"] mod product_version;` 引入本文件，共享的是源码而不是依赖。
//!
//! 使用方式：
//! ```ignore
//! #[path = "../build-support/product_version.rs"]
//! mod product_version;
//!
//! fn main() {
//!     product_version::emit(&std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")), env!("CARGO_PKG_VERSION"));
//! }
//! ```

use std::path::Path;

/// 读取产品版本并以 `rustc-env` 注入，供 `env!("HFS_PRODUCT_VERSION")` 取用。
///
/// `manifest_dir` 是本 crate 的 `CARGO_MANIFEST_DIR`；`tauri.conf.json` 位于
/// 与之同级的 `src-tauri/`（两个 build script 的相对路径都是 `../src-tauri/…`）。
/// 文件缺失或读不出时**回退**到 `fallback`（本 crate 的 `CARGO_PKG_VERSION`），
/// 绝不因为版本读取失败而中断构建 —— 版本号不是构建正确性的前提。
pub fn emit(manifest_dir: &Path, fallback: &str) {
    let config = manifest_dir.join("../src-tauri/tauri.conf.json");
    println!("cargo:rerun-if-changed={}", config.display());
    let version = std::fs::read_to_string(&config)
        .ok()
        .and_then(|text| first_json_version(&text))
        .unwrap_or_else(|| fallback.to_string());
    println!("cargo:rustc-env=HFS_PRODUCT_VERSION={version}");
}

/// 取 JSON 文本里第一个 `"version": "…"` 的值。
///
/// 只需要够用：`tauri.conf.json` 是受控文件，版本键在文件靠前处且是第一个 `version`。
fn first_json_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("\"version\"") else {
            continue;
        };
        let rest = rest.trim_start().strip_prefix(':')?;
        let value = rest.trim().trim_start_matches('"');
        let end = value.find('"')?;
        let version = &value[..end];
        if !version.is_empty() {
            return Some(version.to_string());
        }
    }
    None
}
