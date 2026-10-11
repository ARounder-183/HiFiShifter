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
    // 每行形如 `name v1.2.3`，所以按「行首包名前缀」匹配。此前是行首**全等**，
    // 漏掉 `tauri-utils` / `tauri-runtime` 这类 tauri 家族的传递依赖 —— 禁表的本意
    // 是"整个 tauri 栈"，不是恰好叫 tauri 的那一个 crate。
    for banned in ["tauri", "wry", "HiFiShifter", "cpal"] {
        let hit = text.lines().any(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|name| name.starts_with(banned))
        });
        assert!(
            !hit,
            "插件依赖树里出现了 `{banned}` 家族 —— 二期只能引入原生WebView2，不可引入app事件循环/设备（判据 A5-v2）"
        );
    }
}

/// `--edges normal` 只覆盖默认 feature 的依赖边：可选依赖在非默认 feature 下被
/// 引入时这条测试看不见。加一条 --all-features 变体把那扇门也钉住。
#[test]
fn the_plugin_dependency_tree_contains_no_tauri_stack_with_all_features() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "hifishifter-plugin",
            "--edges",
            "normal",
            "--all-features",
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
    for banned in ["tauri", "wry", "HiFiShifter", "cpal"] {
        let hit = text.lines().any(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|name| name.starts_with(banned))
        });
        assert!(
            !hit,
            "插件依赖树（--all-features）里出现了 `{banned}` 家族 —— 可选 feature 也不得引入app事件循环/设备"
        );
    }
}
