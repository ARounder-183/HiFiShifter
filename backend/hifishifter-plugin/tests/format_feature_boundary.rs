//! 格式特性边界：ara2-bridge 的 companion/格式特性只由**最终 cdylib** 决定。
//!
//! 【为什么用测试而不是人看】共享层（`hifishifter-plugin`）曾钉死
//! `ara2-bridge features = ["vst3"]`，于是每个格式外壳都被迫携带 ARA↔VST3 的
//! C++ companion shim 并检出 VST3 SDK —— CLAP 的 CI checkout vst3sdk 正是这么来
//! 的（旧注释自己承认）。裸 `companion` 特性（`third-party/ara2-bridge/PATCHED.md`）
//! 把"只要 binding 层、不要任何格式"变成可能之后，这条边界必须被钉住，否则下一次
//! "顺手加一个特性"就会把 SDK 需求带回每个外壳。
//!
//! 手段与 `no_tauri_in_dependency_tree.rs` 相同：`cargo tree` 而非
//! `cargo metadata`（后者列整个 workspace，判据会被自己否定）。
//!
//! 【为什么判据只对单 `-p` 构建图成立】Cargo 按整次调用的构建图取特性**并集**：
//! `cargo build -p hifishifter-vst3 -p hifishifter-clap` 一起跑时，ara2-bridge 同时
//! 拿到 `vst3 + clap`，这是**契约内的并集**（build.ps1"一次编译两处组装"正是依赖它）。
//! 边界判据看的是**单个包自身的声明**，所以每个判据都只 `-p` 一个包。

use std::path::PathBuf;
use std::process::Command;

/// `cargo tree -p <pkg> -e features` 的输出（offline；失败即断言失败）。
fn feature_tree(package: &str) -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args(["tree", "--manifest-path"])
        .arg(&manifest)
        .args(["--edges", "features", "--offline", "-p", package])
        .output()
        .expect("cargo tree must run");
    assert!(
        output.status.success(),
        "cargo tree -p {package} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        // 只看 ara2-bridge 系的行；其它 crate 的同名特性（如有）不归本判据管。
        .filter(|line| line.contains("ara2-bridge"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 共享层不得启用任何**格式**特性：它只需要 `plugin`（语义 trait）与裸 `companion`
/// （binding 层）。SDK 需求（vst3 / audio-unit-v2 的 C++ shim）从此进不了共享层的
/// 依赖图 —— 本 crate 无 SDK 可编译，CI 也因此不需要 ARA/VST3 检出。
#[test]
fn the_implementation_layer_enables_no_format_feature() {
    let tree = feature_tree("hifishifter-plugin");
    for forbidden in [
        "feature \"vst3\"",
        "feature \"clap\"",
        "feature \"audio-unit-v2\"",
    ] {
        assert!(
            !tree.contains(forbidden),
            "hifishifter-plugin 的依赖图把 ara2-bridge 的格式特性 {forbidden} 拉了进来：\
             格式特性由最终 cdylib 决定（见各外壳的 Cargo.toml），共享层只要 plugin + companion"
        );
    }
}

/// CLAP 外壳只启用 `clap` 格式特性。companion 的 clap 分支是纯 Rust —— 判据的意义是
/// "CLAP 的构建不需要 VST3/ARA SDK"这一承诺不会回退。
#[test]
fn the_clap_shell_enables_only_the_clap_feature() {
    let tree = feature_tree("hifishifter-clap");
    assert!(
        tree.contains("feature \"clap\"") || tree.contains("companion feature \"clap\""),
        "hifishifter-clap 的依赖图没有启用 ara2-bridge/clap：ARA-over-CLAP 的入口类型来自它"
    );
    for forbidden in ["feature \"vst3\"", "feature \"audio-unit-v2\""] {
        assert!(
            !tree.contains(forbidden),
            "hifishifter-clap 的依赖图把 {forbidden} 拉了进来：\
             CLAP 外壳不得携带其它格式的 companion（SDK 需求会随之回来）"
        );
    }
}

/// `test-support` 特性只允许出现在外壳的 `[dev-dependencies]` 里。
///
/// 【为什么钉死】它把共享层的测试夹具（test_allocator / test_host / MockHost /
/// `session::tests::workspace_fixture`）暴露给消费者。产品构建的**常规依赖**一旦
/// 带上它，测试专用的进程钩子与死代码豁免就进了产物。
#[test]
fn test_support_is_never_a_normal_dependency() {
    for package in ["hifishifter-plugin", "hifishifter-clap"] {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Cargo.toml");
        let output = Command::new(env!("CARGO"))
            .args(["tree", "--manifest-path"])
            .arg(&manifest)
            .args(["--edges", "normal", "--offline", "-p", package])
            .output()
            .expect("cargo tree must run");
        assert!(output.status.success(), "cargo tree -p {package} failed");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            !text.contains("test-support"),
            "{package} 的常规依赖图出现了 test-support：它只允许在外壳的 \
             [dev-dependencies] 里启用（见 hifishifter-plugin::test_support 的文档）"
        );
    }
}
