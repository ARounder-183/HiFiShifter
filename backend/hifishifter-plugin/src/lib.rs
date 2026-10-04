//! HiFiShifter 的 ARA 插件侧适配层。
//!
//! 目标（设计文档 §5.1）：让 HiFiShifter 以 ARA2 插件形式挂在 DAW 里，时间线内容
//! 随 DAW 走，并**复用本体已经存在的离线内核**，而不是另写一套渲染。
//!
//! 本 crate 目前只有两块：
//! - [`ara`]：ARA 文档模型 → `TimelineState` 的映射（探针 Task 3 的产品化版本）；
//! - [`render`]：把 `TimelineState` 交给本体内核离线渲染的薄封装。
//!
//! 边界：这里**不**做设备 I/O，也**不**依赖 Tauri IPC。插件**只**依赖
//! `hifishifter-kernel` —— 这是"插件不把 WebView2 / Tauri 带进 DAW 进程"真正成立的时刻，
//! 由 `tests/no_tauri_in_dependency_tree.rs` 钉住。宿主回调替代 cpal 的职责是下一步的事。

pub mod ara;
pub mod render;
