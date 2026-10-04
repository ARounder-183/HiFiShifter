//! 内核向 UI 发事件的出口。
//!
//! 内核不认识 Tauri。宿主（Tauri app、将来的 ARA 插件）各自注入自己的实现，
//! 内核只依赖这个 trait —— 这是"内核可以脱离 Tauri 单独编译"的前提。
//!
//! 设计细节：方法名刻意叫 `emit`、载荷用 `serde_json::Value`，于是从
//! `tauri::Emitter::emit` 迁过来的调用点**可以原样不动**，改动只落在
//! "这个 handle 从哪来"上。

use std::sync::Arc;

/// 内核向 UI 发送事件的出口。
pub trait EventSink: Send + Sync + 'static {
    /// 发送一个事件。
    ///
    /// 实现方必须自行容忍失败：UI 不在线（或宿主根本没有 UI）时，后台任务
    /// 不应因此受影响。
    fn emit(&self, event: &str, payload: serde_json::Value);
}

/// 可跨线程持有的事件出口。
pub type SharedEventSink = Arc<dyn EventSink>;

/// 什么都不做的出口。给"没有 UI"的宿主（如插件侧、测试）用。
pub struct NullEventSink;

impl EventSink for NullEventSink {
    fn emit(&self, _event: &str, _payload: serde_json::Value) {}
}
