//! 最小集成测试目标。
//!
//! 存在意义：`build.rs` 通过 `cargo:rustc-link-arg-tests` 为测试二进制嵌入
//! ComCtl32 v6 应用清单（修复 Windows 上 cargo test 的测试 exe 因静态导入
//! `TaskDialogIndirect` 而在进程初始化阶段 STATUS_ENTRYPOINT_NOT_FOUND 的
//! 问题）。该 link-arg 指令要求项目至少存在一个测试目标 —— 此文件即该目标。
//!
//! 注意：`rustc-link-arg-tests` 只作用于 tests/ 下的集成测试目标；
//! **lib 单元测试 harness 在 Windows 上无法直接启动**（cargo 无对应通道）。
//! 本地如需运行 `cargo test --lib`，可在构建后用 Win32 `UpdateResource`
//! 向 `target/debug/deps/backend_lib-*.exe` 注入 RT_MANIFEST(24) 资源，
//! 并把 `target/debug` 加入 PATH（vslib/SoundTouch DLL 所在），然后直接
//! 运行该 exe；CI 上则由 Linux 的 backend-test job 覆盖 lib 单测。

#[test]
fn smoke_test_target_exists() {
    // 占位断言：保证该目标可被 cargo 发现并运行。
    assert!(true);
}
