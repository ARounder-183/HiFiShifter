//! HiFiShifter 的 ARA 文档模型：宿主回调累积出的图 → `TimelineState`。
//!
//! 【边界】本 crate 只认识 ARA 的**语义**（源、修改、区间序列、播放区间）与内核的
//! 时间线，不认识任何格式 ABI：没有 `ARAFactory`、没有 `ProcessContext`、没有
//! `clap_process_t`。格式外壳负责把宿主回调变成 [`mapping::AraDocument`]，映射只有
//! 这一份。
//!
//! 【为什么值得独立】映射决定「宿主里的 item 落在时间轴哪里、按什么速率播放、
//! 哪些字段丢失」。VST3 与 CLAP 各写一份，用户就会看到同一个工程在两种格式下
//! 片段位置不同 —— 那是无法用测试发现、只能在现场被用户发现的错误。

#![deny(missing_docs)]

pub mod mapping;
