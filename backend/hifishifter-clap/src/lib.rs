//! HiFiShifter 的 CLAP 插件外壳。
//!
//! 【边界】本 crate 只认识 CLAP 1.1.9 的模块入口与扩展；时间/缓冲/状态/资源规则
//! 一律经 `hifishifter-plugin-core`。
//!
//! 【能力边界】阶段 1 是 Capture-only：descriptor 不声明 `ara:supported`，
//! entry 不暴露 ARA 工厂，实例不返回 ARA 扩展。共享 ARA 文档模型仍在 VST3 外壳里。
//! 编辑器窗口在 Windows 上经 `clap.gui` 内嵌同一份前端产物；其它平台如实降级。
//! 不链接 VST3 SDK、不带 Tauri/wry/app runtime。

#![deny(missing_docs)]
// ABI 结构体是 C 头文件的逐字段镜像；字段语义在对应 CLAP 头文件里，不在 Rust 侧重复。
#![allow(missing_docs)]
#![allow(
    clippy::items_after_test_module,
    clippy::new_without_default,
    // 按声道下标逐帧读取/写入是实时缓冲的语义本身；迭代器会掩盖通道索引。
    clippy::needless_range_loop,
)]
pub mod abi;
pub mod ara;
pub mod audio;
pub mod capture;
pub mod editor;
pub mod entry;
pub mod gui;
pub mod latency_tail;
pub mod log_host;
pub mod ports;
pub mod render;
pub mod render_ext;
pub mod state;
pub mod timer;

pub use entry::{instance, instance_mut, Instance};

/// 黑盒探针读取的构建身份（CLAP 1.1.9 / 产品版本 / 目标）。
pub fn build_identity() -> String {
    format!(
        "HiFiShifter {} clap-1.1.9 {}-{}",
        entry::VERSION,
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}
