//! 时间线状态：数据模型与运行时容器。
//!
//! 【本文件的存在意义】原先 `state.rs` 是一个 11826 行的单文件，把"纯数据模型"
//! 与"运行时容器 `AppState`"混在一起。`AppState` 持有 Tauri 句柄与设备层句柄，
//! 于是任何从 `state` 出发的依赖闭包都会被非内核模块污染 —— 这是内核抽取撞墙的根因
//! （实测：从 `mixdown` 出发的闭包会因为 `state` 而卷进 `project` / `notebook_assets` /
//! `hfspeaks_v2` / `temp_manager` / `media` / `recording`，共 43 模块 / 2.57 MB）。
//!
//! 拆成多个文件、用一个 `mod.rs` 再导出，是为了让 `crate::state::X` 的**全部既有路径
//! 保持不变**（内核搬迁的通用做法：搬内容，接路径）。
//!
//! 拆分分两步走，两步都已落地：
//! 1. 目录化：`state.rs` → `state/original.rs`，零改写。
//! 2. 切分：`AppState`、`RuntimeState`、`WaveformInflightGuard`、`SourceFileCheckItem`
//!    及其 `impl`、以及依赖 `AppState` 的 3 条测试移进 `app.rs`；
//!    其余（`Clip` / `Track` / `TimelineState` / `ClipTake` / 历史……）是 `model.rs`。

mod app;
mod model;

pub use app::*;
pub use model::*;
