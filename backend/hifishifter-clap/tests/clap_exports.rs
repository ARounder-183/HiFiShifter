//! CLAP 产物导出符号门禁：`clap_entry` 是宿主唯一必须找得到的动态库入口。
//!
//! 【为什么要构建产物再查】`cargo test` 只产出 rlib 测试二进制，不会生成 cdylib；
//! 与 `vst3_exports.rs` 一样，先 `cargo build -p hifishifter-clap`。

use std::path::PathBuf;
use std::process::Command;

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
