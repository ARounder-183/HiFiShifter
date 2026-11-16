# 本地补丁说明（HiFiShifter）

这份源码是 `ara2-bridge` 0.3.0（聚合门面 crate）的拷贝 + 下表所列补丁。
与 `ara2-bridge-core` / `ara2-bridge-plugin` 的补丁同属一个家族：`[patch.crates-io]`
指向本目录，升级 0.4 时逐项核对后删除。

## 补丁清单（升级 0.4 时逐项核对；"行为"= 不只是新增代码）

| # | 位置 | 增量/行为 | 内容 | 上游 0.4 状态 |
|---|---|---|---|---|
| 1 | `Cargo.toml` `[features]` | 增量 | 新增裸 `companion = ["dep:ara2-bridge-companion"]`：只启用格式中立的 binding/lifecycle 层（`CompanionFactory` / `CompanionProcessorBinding` / `CompanionControllerBinding` / `CompanionRoles`），不启用任何格式子模块，因此不触发任何 SDK 检出、不编译任何 C++ shim（companion 的 vst3 / audio-unit-v2 分支各自要求 SDK，clap 分支纯 Rust） | 待查 |
| 2 | `src/lib.rs` 的 companion 再导出门 | 增量 | `#[cfg(any(...))]` 增加 `feature = "companion"`，使裸 companion 特性下 `ara2_bridge::companion` 可达 | 待查 |

## 为什么需要它

【动机】`hifishifter-plugin`（格式无关实现层）的 `ara/binder.rs` 使用 companion 的
`CompanionFactory` / `CompanionProcessorBinding` / `CompanionControllerBinding` /
`CompanionRoles`——这些类型在 0.3.0 里只能经 `vst3` / `clap` / `audio-unit-v2`
三个格式特性进入。于是共享实现层被迫钉死一个格式特性（历史上是 `vst3`），后果：

1. 每个格式外壳（CLAP / 未来的 AU）构建时都携带 ARA↔VST3 的 C++ companion shim，
   并被迫检出 VST3 SDK（见 `.github/workflows/clap.yml` 旧注释的自述）；
2. 与 `hifishifter-plugin-core` Cargo.toml 声明的原则"feature 由最终 cdylib 决定"
   直接矛盾。

裸 `companion` 特性补上"只要 binding 层、不要任何格式"这一档，让格式特性真正
回到各壳 crate 的依赖表上。详细设计见
`docs/plans/2026-11-17-vst3-shell-extraction-plan.md` §6.1。

## 上游修复后的撤销方式

上游若提供等价特性（例如 `companion` 或按需的 `dep:ara2-bridge-companion` 直连），
删除本目录与 `backend/Cargo.toml` `[patch.crates-io]` 里的
`ara2-bridge = { path = "third-party/ara2-bridge" }` 一行，改用上游特性即可。
两个消费声明需要同步：

- `backend/hifishifter-plugin/Cargo.toml`：`features = ["plugin", "companion"]`
- `backend/hifishifter-vst3/Cargo.toml`：`features = ["plugin", "companion", "vst3"]`
- `backend/hifishifter-clap/Cargo.toml`：`features = ["plugin", "companion", "clap"]`
