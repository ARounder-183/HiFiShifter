//! HiFiShifter 的 CLAP 插件外壳。
//!
//! 【边界】本 crate 只认识 CLAP 1.1.9 的模块入口与扩展；时间/缓冲/状态/资源规则
//! 一律经 `hifishifter-plugin-core`。ARA 工厂通过 `ara2-bridge` 的 CLAP companion 暴露，
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
pub mod entry;
pub mod gui;
pub mod ports;
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
