//! 共享原生剪贴板；沿用本应用自定义格式、文本兼容和争用重试，不初始化GUI运行时。
#[cfg(target_os = "linux")]
pub mod linux_clipboard;
pub mod system_clipboard;
pub use system_clipboard::*;
