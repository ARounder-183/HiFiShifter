//! ARA 宿主适配层。
//!
//! - [`mapping`]：ARA 文档模型 → `TimelineState`（探针 Task 3 验证过的映射，产品化版本）；
//! - [`model`]：实现 `ara2-bridge` 的 `PluginModel` 语义 trait，把宿主的模型回调
//!   **累积成一份 [`mapping::AraDocument`]**，供映射层消费。
//!
//! 【为什么分两层】映射层只依赖一个纯数据的 `AraDocument`，因此它可以在没有宿主、
//! 没有 ARA 运行时的前提下被逐样本测试（`tests/ara_mapping.rs`）。模型层是它与
//! 真实 ARA 之间的唯一接触面。

pub mod mapping;
pub mod model;
pub(crate) mod time_map;

pub use mapping::*;
