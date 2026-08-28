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
    // 占位断言：保证该目标可被 cargo 发现并运行。
    assert!(true);
}
