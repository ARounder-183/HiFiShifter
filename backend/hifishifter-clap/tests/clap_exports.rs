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
                // ONNX Runtime 的 DirectML 执行提供器静态链接部分。它和 d3d12 / dxgi
                // 一样是 Windows 系统组件（Win10 1903+ / Win11 自带，System32 里就有），
                // 不是本产品分发的第三方 DLL —— 宿主按绝对路径加载 .clap 时由系统解析。
                // 【为什么 CLAP 现在会导入它】曾把 onnx 一起关掉以避开这个导入，代价是
                // fcpe_onnx 退化成 stub：捕获后音高分析只算出 DYN 电平、F0 恒为空，
                // 原线永远不出现、渲染门禁 pending 60 s。真实 F0 比少一个系统导入重要。
                | "directml.dll"
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

/// **CLAP 必须链接真正的 FCPE 估计器，而不是 onnx 关闭时的 stub。**
///
/// 【为什么这条是那次报障的正身】`default-features = false` 曾把内核的 `onnx` 一起
/// 关掉 ⇒ `fcpe_onnx` 编译成 stub（`is_available() = false`、推理恒失败）。捕获结束后
/// 那次「音高分析」于是只产出 DYN 电平（`has_data = true` 的假象），F0 恒为空：
/// `pitch_orig` 组装出来是全 0、`pitch_orig_key` 永不置位 ⇒ 原线不出现、渲染门禁
/// pending 60 s 后超时、用户画完参数线才触发一次 `f0=0` 的直通。三个现象同源。
/// VST3/App 一直带 onnx，只有 CLAP 这条通路坏 —— 所以门禁钉在 CLAP 这一层。
///
/// 【判据不碰模型】空输入时 stub 返回 `Err("ONNX feature not compiled")`，真实实现
/// 在 `mono.is_empty()` 处直接 `Ok(vec![])`。不加载会话、不读文件、不依赖 GPU。
#[test]
fn the_clap_module_links_the_real_fcpe_estimator() {
    let result = hifishifter_kernel::fcpe_onnx::infer_f0_hz_f32(
        &[],
        44_100,
        5.0,
        hifishifter_kernel::fcpe_onnx::FCPE_F0_MIN_HZ,
        hifishifter_kernel::fcpe_onnx::FCPE_F0_MAX_HZ,
    );
    assert!(
        result.is_ok(),
        "CLAP 必须启用 hifishifter-plugin/onnx：FCPE stub 产不出音高原线，\
         捕获后的分析会退化为只算电平（{result:?}）"
    );
}
