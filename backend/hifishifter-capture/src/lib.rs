//! 捕获运行时：把宿主播放中的音频按**工程位置**录进插件自己的时间线。
//!
//! # 分层
//!
//! | 层 | 模块 | 职责 |
//! |---|---|---|
//! | 捕获 | [`recorder`] | 音频线程：按工程位置写定长段，零分配、零阻塞 |
//! | 存储 | [`store`] | take 管理、选项、sidecar 落盘、状态读数 |
//! | 投影 | [`timeline`] | 非音频线程：解决覆盖关系、生成 `TimelineState` |
//! | 落盘 | [`sidecar`] | 把段写成可恢复的音频 + 清单 |
//! | 预算 | [`budget`] | 进程级 PCM 内存预算（与渲染共用） |
//!
//! # 为什么宿主无关
//!
//! 输入只有 [`recorder::InputBlock`]（工程位置、帧数、采样率、两个浮点平面、是否在
//! 播放）与 [`recorder::BlockContext`]（哪条进程流、传输相位）。VST3 从
//! `ProcessContext` 构造它们，CLAP 从 `clap_event_transport_t` / `steady_time`
//! 构造它们 —— 判定逻辑一份，格式外壳只做一次转换。
//!
//! # 与渲染的关系
//!
//! 本 crate **不认识**渲染。它交出的是数据：[`types::CaptureTape`]（原始磁带）与
//! [`types::CapturePcm`]（源 PCM）、[`types::CaptureRegion`]（1:1 播放区间）。
//! 各格式外壳自己把它们接进渲染管线或直接交给编辑器。
// 【为什么允许缺文档】这些类型原来是 VST3 crate 的 `pub(crate)` 实现细节（逐行注释
// 在代码里），搬来之后为了两个格式外壳共享才变成 `pub`。要求它们逐个补齐字段级文档
// 只会造出一堆复述字段名的噪音；真正需要解释的判定仍写在原处。
#[allow(missing_docs)]
pub mod budget;
#[allow(missing_docs)]
pub mod recorder;
#[allow(missing_docs)]
pub mod sidecar;
#[allow(missing_docs)]
pub mod status;
pub mod store;
#[allow(missing_docs)]
pub mod timeline;
pub mod transport;
pub mod types;

pub use store::{
    CaptureOptions, CaptureStatus, CaptureStore, MaterialSpan, MaterialSummary, MissingCapture,
    RecordingStatus, TakeSummary,
};
pub use types::{CapturePcm, CaptureRegion, CaptureTape};
