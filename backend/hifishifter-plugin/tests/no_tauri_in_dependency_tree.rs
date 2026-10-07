//! 二期架构守卫：插件不引入Tauri/wry/app/cpal；允许原生WebView2内嵌原GUI。
//!
//! 【为什么用测试而不是人看】这条约束会随着后续搬迁被无意破坏
//! （某个模块为了图方便去 `use backend_lib::…`），而症状要到 DAW 里才显形 ——
//! 那时插件已经初始化了独立app事件循环或设备。二期原生WebView2是有意允许的例外。
//!
//! 手段是cargo tree：插件依赖图禁止tauri/wry/app本体/cpal，原生webview2-com不在禁表。
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
    for banned in ["tauri", "wry", "HiFiShifter", "cpal"] {
        let hit = text
            .lines()
            .any(|line| line.split_whitespace().next() == Some(banned));
        assert!(
            !hit,
            "插件依赖树里出现了 `{banned}` —— 二期只能引入原生WebView2，不可引入app事件循环/设备（判据 A5-v2）"
        );
    }
}
