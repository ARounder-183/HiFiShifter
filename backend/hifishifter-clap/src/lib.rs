//! HiFiShifter 的 CLAP 插件外壳。
//!
//! 【边界】本 crate 只认识 CLAP 1.1.9 的模块入口与扩展；时间/缓冲/状态/资源规则
//! 一律经 `hifishifter-plugin-core`。
//!
//! 【能力边界】descriptor 声明 `ara:supported`（不声明 `ara:required`）：entry 暴露共享
//! ARA 工厂，实例暴露 ARA 插件扩展并在宿主 bind 时接到同一份文档模型 —— 仅限按
//! ARACLAP.h 定稿协议发现的宿主。REAPER 6.71+ 用的是 2022 私有草案，其宿主实现在
//! 需要真正渲染时会崩在自身代码里（见 `ara` 模块文档），因此不应答该 ID；REAPER 里
//! 的 CLAP 插件以 Capture 形态稳定运行，需要 ARA 时用 VST3 形态。
//! 没有 ARA 的宿主里，它仍是可用的普通效果器（Capture 形态）。
//! 两种形态的音频、捕获与编辑都走
//! `hifishifter-plugin` 里那一份实现，本 crate 只做 ABI 翻译。
//! 编辑器窗口经 `clap.gui` 内嵌**与 VST3 同一个**共享视图栈（`hifishifter-plugin` 的
//! `editor::gui`）：同一份 Policy、同一份平台原语、同一份前端产物。
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
pub mod latency_tail;
pub mod log_host;
pub mod ports;
pub mod process;
pub mod render_ext;
pub mod state;
pub mod timer;

/// 【为什么测试二进制要在进入 main 之前关掉异步预热】预热线程在"建 ONNX 会话"
/// 中途被进程退出拆掉时，测试二进制以 `0xC0000005` 退出 —— 即使每条用例都通过
/// （plugin crate 里 `settings_store` 的注释记载了同一签名；本 crate 的测试没有
/// 设置下发路径可挂靠，因此用 `ctor` 在 CRT 初始化时执行进程级兜底）。
#[cfg(test)]
mod test_prewarm_guard {
    #[ctor::ctor(unsafe)]
    fn disarm_async_session_prewarm() {
        hifishifter_kernel::vocoder_ort_session::set_async_session_prewarm(false);
    }
}

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
