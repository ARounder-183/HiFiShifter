//! CLAP 产物导出符号门禁：`clap_entry` 是宿主唯一必须找得到的动态库入口。
//!
//! 【为什么要构建产物再查】`cargo test` 只产出 rlib 测试二进制，不会生成 cdylib；
//! 与 `vst3_exports.rs` 一样，先 `cargo build -p hifishifter-clap`。

use std::path::{Path, PathBuf};
use std::process::Command;

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
/// CLAP 模块是**单文件**产物：除了系统库，它不许导入任何 DLL。
///
/// 【为什么比 VST3 更严】`.clap` 与 `HiFiShifter-resources` 同级分发，旁边没有
/// 任何第三方 DLL。一个链接期依赖（曾经的 `vslib_x64.dll`）会让宿主的
/// `LoadLibrary` 直接失败，而日志里连一行都没有。
#[cfg(target_os = "windows")]
#[test]
fn the_clap_module_imports_nothing_beyond_system_libraries() {
    let artifact = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/debug")
        .join("hifishifter_clap.dll");
    assert!(artifact.is_file(), "找不到 {} —— 先跑 cargo build -p hifishifter-clap", artifact.display());
    let unexpected: Vec<String> = imported_dlls(&artifact)
        .into_iter()
        .filter(|name| !is_system_module(name))
        .collect();
    assert!(
        unexpected.is_empty(),
        "CLAP 模块导入了非系统 DLL：{unexpected:?}。\n
         `.clap` 是单文件产物，旁边不会放任何第三方 DLL —— 宿主加载会失败。"
    );
}

#[test]
fn the_clap_module_entry_point_is_exported() {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/debug");
    let (name, dll) = if cfg!(windows) {
        ("hifishifter_clap.dll", "hifishifter_clap.dll")
    } else if cfg!(target_os = "macos") {
        ("libhifishifter_clap.dylib", "libhifishifter_clap.dylib")
    } else {
        ("libhifishifter_clap.so", "libhifishifter_clap.so")
    };
    let artifact = target.join(dll);
    assert!(
        artifact.is_file(),
        "缺少 {} —— 先跑 cargo build -p hifishifter-clap",
        artifact.display()
    );
    let output = if cfg!(windows) {
        Command::new("dumpbin")
            .args(["/exports"])
            .arg(&artifact)
            .output()
    } else if cfg!(target_os = "macos") {
        Command::new("nm").args(["-gU"]).arg(&artifact).output()
    } else {
        Command::new("nm")
            .args(["-D", "--defined-only"])
            .arg(&artifact)
            .output()
    }
    .expect("symbol listing tool must run");
    assert!(output.status.success(), "symbol listing failed: {output:?}");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.contains("clap_entry"),
        "{name} 没有导出 clap_entry: {text}"
    );
}
