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

/// `dumpbin /dependents` 的解析：取出导入的 DLL 名（小写）。
#[cfg(target_os = "windows")]
fn imported_dlls(artifact: &Path) -> Vec<String> {
    let output = Command::new("dumpbin")
        .args(["/dependents"])
        .arg(artifact)
        .output()
        .expect("需要 MSVC 的 dumpbin（先点源 scripts/env/msvc-env.ps1）");
    assert!(output.status.success(), "dumpbin /dependents 失败: {output:?}");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let name = line.trim();
            (name.len() > 4
                && name.to_ascii_lowercase().ends_with(".dll")
                && !name.contains(char::is_whitespace))
                .then(|| name.to_ascii_lowercase())
        })
        .collect()
}

/// Windows 系统模块（含 CRT 与 api-ms-win-* 转发层）。
#[cfg(target_os = "windows")]
fn is_system_module(name: &str) -> bool {
    name.starts_with("api-ms-win-")
        || matches!(
            name,
            "kernel32.dll"
                | "user32.dll"
                | "ole32.dll"
                | "oleaut32.dll"
                | "shell32.dll"
                | "advapi32.dll"
                | "combase.dll"
                | "ntdll.dll"
                | "dbghelp.dll"
                | "setupapi.dll"
                | "d3d12.dll"
                | "dxgi.dll"
                | "bcryptprimitives.dll"
                | "msvcp140.dll"
                | "msvcp140_1.dll"
                | "vcruntime140.dll"
                | "vcruntime140_1.dll"
        )
}
/// 引擎的**运行时依赖**必须全部在打包白名单里。
///
/// 【为什么这条必须存在】`InitDll` 返回 false 是"宿主里插件不见了"的唯一现场表现，
/// 而真因（引擎导入了一个没随包分发的 DLL）在宿主日志里一个字都没有 —— 那一刻
/// `guard_ffi` 的 logger 还没装。这条用例把"能编译"与"能加载"之间的缝钉住：
/// 任何新增的链接期依赖都必须同时进 `build.ps1` 的拷贝白名单，或者根本不该加。
#[cfg(target_os = "windows")]
#[test]
fn the_engine_only_imports_dlls_the_bundle_ships() {
    let artifact = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/debug")
        .join(module_file_name());
    assert!(artifact.is_file(), "找不到 {} —— 先跑 cargo build -p hifishifter-plugin", artifact.display());
    // 与 `scripts/build/build.ps1` 的 `foreach ($dll in ...)` 白名单逐字对应。
    const SHIPPED: &[&str] = &["soundtouchdll.dll", "directml.dll", "onnxruntime.dll"];
    let unexpected: Vec<String> = imported_dlls(&artifact)
        .into_iter()
        .filter(|name| !is_system_module(name) && !SHIPPED.contains(&name.as_str()))
        .collect();
    assert!(
        unexpected.is_empty(),
        "引擎导入了没有随包分发的 DLL：{unexpected:?}。\n
         要么把它加进 build.ps1 的拷贝白名单，要么关掉引入它的 feature —— \n
         宿主的 LoadLibrary 会直接失败，表现是 InitDll 返回 false，日志里没有任何原因。"
    );
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
