//! 最小集成测试目标。
//!
//! 存在意义：`build.rs` 通过 `cargo:rustc-link-arg-tests` 为测试二进制嵌入
//! ComCtl32 v6 应用清单（修复 Windows 上 cargo test 的测试 exe 因静态导入
//! `TaskDialogIndirect` 而在进程初始化阶段 STATUS_ENTRYPOINT_NOT_FOUND 的
//! 问题）。该 link-arg 指令要求项目至少存在一个测试目标 —— 此文件即该目标。
//!
//! 真正的单元测试位于各源码模块的 `#[cfg(test)] mod tests` 中
//! （`cargo test --lib` 运行，同样受益于清单嵌入）。

#[test]
fn smoke_test_target_exists() {
    // 占位断言：保证该目标可被 cargo 发现并运行。
    assert!(true);
}
