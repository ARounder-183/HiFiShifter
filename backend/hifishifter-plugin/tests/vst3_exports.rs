//! VST3 模块入口的守卫：导出的入口符号必须存在且拼写正确。
//!
//! 【为什么值得一条测试】宿主查不到入口符号时**不会报错**，表现就是"这个文件 0 个类"，
//! 与符号拼错、IID 布局写错完全同一个症状（探针期实测过）。那是这类问题里最难定位的
//! 一种，所以用测试把每个平台的入口清单钉住。
//!
//! 前置：先 `cargo build -p hifishifter-plugin`（cdylib 由它产出，`cargo test` 不产出）。

use std::path::{Path, PathBuf};
use std::process::Command;

/// 各平台的模块产物文件名。
fn module_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "hifishifter_plugin.dll"
    } else if cfg!(target_os = "macos") {
        "libhifishifter_plugin.dylib"
    } else {
        "libhifishifter_plugin.so"
    }
}

/// 各平台的入口清单。
///
/// - Windows：`InitDll` / `ExitDll`（`LoadLibrary` 之后显式调用）。
/// - Linux：`ModuleEntry` / `ModuleExit`（SDK 的 `module_linux.cpp` 会查）。
/// - macOS：`bundleEntry` / `bundleExit`（CFBundle 那一套）。
///
/// `GetPluginFactory` 三平台都要 —— 它是唯一真正必须的符号。
fn expected_entry_points() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["GetPluginFactory", "InitDll", "ExitDll"]
    } else if cfg!(target_os = "macos") {
        &["GetPluginFactory", "bundleEntry", "bundleExit"]
    } else {
        &["GetPluginFactory", "ModuleEntry", "ModuleExit"]
    }
}

/// 列出动态导出表。
#[cfg(target_os = "windows")]
fn exported_symbols(artifact: &Path) -> String {
    let output = Command::new("dumpbin")
        .args(["/exports", artifact.to_str().unwrap()])
        .output()
        .expect("需要 MSVC 的 dumpbin（先点源 scripts/env/msvc-env.ps1）");
    assert!(output.status.success(), "dumpbin 失败: {output:?}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[cfg(target_os = "linux")]
fn exported_symbols(artifact: &Path) -> String {
    let output = Command::new("nm")
        .args(["-D", "--defined-only", artifact.to_str().unwrap()])
        .output()
        .expect("需要 nm（binutils）");
    assert!(output.status.success(), "nm 失败: {output:?}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[cfg(target_os = "macos")]
fn exported_symbols(artifact: &Path) -> String {
    let output = Command::new("nm")
        .args(["-gU", artifact.to_str().unwrap()])
        .output()
        .expect("需要 nm（Xcode Command Line Tools）");
    assert!(output.status.success(), "nm 失败: {output:?}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn the_vst3_module_entry_points_are_exported() {
    let artifact = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/debug")
        .join(module_file_name());
    assert!(
        artifact.is_file(),
        "找不到 {} —— 先跑 `cargo build -p hifishifter-plugin` 再跑测试（cdylib 不由 cargo test 产出）",
        artifact.display()
    );

    let text = exported_symbols(&artifact);
    for name in expected_entry_points() {
        assert!(text.contains(name), "缺少 VST3 导出符号 `{name}`：\n{text}");
    }
}
