//! 时间线数据模型。
//!
//! 从 app 的 `state.rs` 拆出来的纯模型部分 —— 不含 `AppState`
//! （那是持有 Tauri 句柄与设备层句柄的运行时容器，留在 app 层）。

pub mod model;

pub use model::*;
