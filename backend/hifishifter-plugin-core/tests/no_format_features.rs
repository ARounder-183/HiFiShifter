//! 格式专属 feature 只能出现在最终 cdylib（D4）。
//!
//! `ara2-bridge` 的 `vst3` / `clap` / `audio-unit-v2` 会拉 companion 与对应 SDK；
//! core 一旦带上任何一项，每个格式都会被拖进别人的 SDK。

use std::process::Command;

#[test]
fn core_declares_no_companion_features() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--manifest-path",
        ])
        .arg(format!("{manifest}/Cargo.toml"))
        .output()
        .expect("cargo metadata must run");
    assert!(output.status.success(), "cargo metadata failed: {output:?}");
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let package = json["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "hifishifter-plugin-core")
        .expect("core package");
    let companion = package["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["name"] == "ara2-bridge");
    assert!(
        companion.is_none(),
        "core must not declare ara2-bridge directly"
    );
}
