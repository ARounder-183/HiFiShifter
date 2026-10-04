//! HiFiShifter 的离线音频内核。
//!
//! 这里放**不依赖 Tauri、也不依赖音频设备**的那部分：混音、拉伸、淡入淡出曲线、
//! 渲染缓存、音高编辑等。ARA 插件侧（`hifishifter-plugin`）只依赖这一层，
//! 于是插件不会把 WebView2 / Tauri 运行时带进 DAW 进程。
//!
//! ## 搬迁状态
//!
//! 内核目前仍主要住在 `backend/src-tauri/src/` 里（那里有 43 个模块属于内核闭包，
//! 但只有 5 个模块碰到 `tauri::`）。搬迁是**逐步**进行的：每搬一个模块，
//! `backend_lib` 就用 `pub use` 把它再接回来，于是 `crate::<模块名>::…` 的路径
//! 在 app 侧完全不变，基线测试始终可跑。
//!
//! 已迁入：
//! - [`fade_curves`]：REAPER 风格淡入淡出曲线数学核心（自包含、无内部依赖）。
//!
//! 待迁入：`state`（模型）、`mixdown`、`renderer`、`render_cache`、`render_key`、
//! `encode`、`time_stretch`、`vocoder` 系、`pitch` 系等 —— 以及先把
//! `state` / `pitch_clip` / `recording` / `pitch_analysis` 里那 32 处
//! `tauri::Emitter` 调用换成注入式的事件出口。

pub mod fade_curves;

pub mod byte_budget_cache;
pub mod events;
pub mod util;
