//! ARA 宿主适配层。
//!
//! - [`mapping`]：ARA 文档模型 → `TimelineState`。**实现已搬到 `hifishifter-ara`**
//!   （与格式无关，CLAP 侧复用同一份），这里只做再导出，既有调用点不用改；
//! - [`model`]：实现 `ara2-bridge` 的 `PluginModel` 语义 trait，把宿主的模型回调
//!   **累积成一份 [`mapping::AraDocument`]**，供映射层消费。
//!
//! 【为什么分两层】映射层只依赖一个纯数据的 `AraDocument`，因此它可以在没有宿主、
//! 没有 ARA 运行时的前提下被逐样本测试（`tests/ara_mapping.rs`）。模型层是它与
//! 真实 ARA 之间的唯一接触面。

pub use hifishifter_ara::mapping;
/// 格式无关的 ARA 实例绑定（VST3 与 CLAP 共用同一份决策逻辑）。
pub mod binder;
pub mod model;
pub mod persistence;

pub use mapping::*;
