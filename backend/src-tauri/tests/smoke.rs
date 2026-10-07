//! Minimal integration test target.
//!
//! Background: the dependency tree (winit/tauri dialog) statically imports
//! `comctl32.dll!TaskDialogIndirect`, which exists only in the v6
//! side-by-side assembly. The main binary gets a Common-Controls v6 manifest
//! from tauri_build; this target and the other integration tests get the
//! same manifest from `build.rs` via `cargo:rustc-link-arg-tests` (which
//! requires the project to have at least one integration test target — this
//! file is that target, and keeps any future dialog call bound to v6).
//!
//! The **lib unit-test harness** has no cargo link-arg channel; it is covered
//! by the `comctl32.dll` delay-load (/DELAYLOAD) setup in the repo-root
//! `.cargo/config.toml`: the harness never binds comctl32 at startup and
//! unit tests never open dialogs, so `cargo test --lib` runs directly on
//! Windows. (Historically this needed RUSTFLAGS-injected /MANIFEST:EMBED or
//! post-build manifest injection via mt.exe.)

#[test]
fn smoke_test_target_exists() {
    // 本测试**没有断言**，这是刻意的：它的存在本身就是断言 —— 目标能被 cargo
    // 发现、能链接、能启动到跑完这个函数，就是它要证明的全部。
    //
    // 【为什么删掉原来的 `assert!(true)`】它是恒真式，clippy 会（正确地）报
    // `assertions_on_constants`；而把它换成任何"看起来有意义"的断言都只是在给
    // 一个占位测试化妆。空体表达同一个意图，且不再是一条永远不会失败的断言。
}
