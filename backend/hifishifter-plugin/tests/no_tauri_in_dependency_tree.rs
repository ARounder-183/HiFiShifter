//! 架构不变量的守卫：**插件进程里不出现 Tauri / WebView2**（设计 §1 判据 A5）。
//!
//! 【为什么用测试而不是人看】这条约束会随着后续搬迁被无意破坏
//! （某个模块为了图方便去 `use backend_lib::…`），而症状要到 DAW 里才显形 ——
//! 那时插件进程里已经塞进了一个 WebView 宿主。放在测试里，破坏的第一时间就报。
//!
//! 手段是 `cargo tree`：只要插件的**依赖图**里没有 `tauri` / `wry` / `webview2-com` /
//! app 本体（`HiFiShifter`），就说明它的依赖闭包是干净的内核。
//!
//! 【为什么不用 `cargo metadata`】它列的是**整个 workspace 的成员**，不是某个包的
//! 依赖图 —— app 本体必然出现在里面，判据会被自己的工具否定。实测踩过。

use std::process::Command;

#[test]
fn the_plugin_dependency_tree_contains_no_tauri_stack() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "hifishifter-plugin",
            "--edges",
            "normal",
            "--prefix",
            "none",
            "--offline",
            "--manifest-path",
        ])
        .arg(format!("{manifest}/Cargo.toml"))
        .output()
        .expect("cargo tree must run");
    assert!(output.status.success(), "cargo tree failed: {output:?}");

    let text = String::from_utf8_lossy(&output.stdout);
    // 每行形如 `name v1.2.3`，所以按「行首的包名」匹配，避免误伤子串。
    for banned in ["tauri", "wry", "webview2-com", "HiFiShifter"] {
        let hit = text
            .lines()
            .any(|line| line.split_whitespace().next() == Some(banned));
        assert!(
            !hit,
            "插件依赖树里出现了 `{banned}` —— 插件会把 DAW 进程污染成 WebView 宿主（判据 A5）"
        );
    }
}
