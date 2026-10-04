//! 内核向 UI 发事件的出口。
//!
//! 内核不认识 Tauri。宿主（Tauri app、将来的 ARA 插件）各自注入自己的实现，
//! 内核只依赖这个 trait —— 这是"内核可以脱离 Tauri 单独编译"的前提。
//!
//! 设计细节：方法名刻意叫 `emit`、载荷用 `serde_json::Value`，于是从
//! `tauri::Emitter::emit` 迁过来的调用点**可以原样不动**，改动只落在
//! "这个 handle 从哪来"上。

use std::sync::Arc;
use std::sync::OnceLock;

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

/// 进程级的事件出口。
///
/// 【为什么需要它，而不只是 `AppState.events`】内核模块散落在调用栈深处
/// （音高分析的后台线程、逐 clip 的分析线程），把事件出口一路当参数传下去
/// 要污染十几个签名，而且调用点会退化成 `if let Some(app) = …` 这种形状 ——
/// 那正是它们现在离不开 `tauri::AppHandle` 的原因。
/// 宿主在一个进程里只有一个，所以进程级单例贴合事实，与 [`crate::host::host`] 同一做法。
pub struct Events {
    inner: OnceLock<SharedEventSink>,
}

impl Events {
    /// 构造一个空的出口（`const` 以便放进 `static`）。
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// 安装事件出口。返回 `false` 表示已经装过（**不替换**）。
    pub fn install(&self, sink: SharedEventSink) -> bool {
        self.inner.set(sink).is_ok()
    }

    /// 是否已装配。
    pub fn is_installed(&self) -> bool {
        self.inner.get().is_some()
    }

    /// 发送一个事件。**未装配时静默丢弃**（内核会被单测与插件直接使用，那时没有 UI）。
    ///
    /// 载荷由 `serde` 序列化；序列化失败时退化为 `null` 载荷而不是丢弃事件 ——
    /// 事件本身（"某根轨的音高线更新了"）比它的载荷更重要。
    pub fn emit<T: serde::Serialize>(&self, event: &str, payload: T) {
        if let Some(sink) = self.inner.get() {
            let value = serde_json::to_value(payload).unwrap_or(serde_json::Value::Null);
            sink.emit(event, value);
        }
    }
}

impl Default for Events {
    fn default() -> Self {
        Self::new()
    }
}

/// 进程级事件出口。
pub fn events() -> &'static Events {
    static EVENTS: Events = Events::new();
    &EVENTS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingSink {
        seen: Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl EventSink for RecordingSink {
        fn emit(&self, event: &str, payload: serde_json::Value) {
            self.seen.lock().unwrap().push((event.to_string(), payload));
        }
    }

    /// 未装配时静默丢弃：内核会被单测与插件直接使用，那时根本没有 UI。
    #[test]
    fn an_uninstalled_outlet_drops_events_silently() {
        let events = Events::new();
        assert!(!events.is_installed());
        events.emit("some_event", serde_json::json!({"a": 1}));
    }

    /// 装配后必须原样转发，且载荷可被序列化。
    #[test]
    fn an_installed_outlet_forwards_and_serializes() {
        let events = Events::new();
        let sink = Arc::new(RecordingSink::default());
        assert!(events.install(sink.clone()));

        #[derive(serde::Serialize)]
        struct Payload {
            root_track_id: String,
        }
        events.emit(
            "pitch_orig_updated",
            Payload {
                root_track_id: "root-1".to_string(),
            },
        );

        let seen = sink.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "pitch_orig_updated");
        assert_eq!(seen[0].1["root_track_id"], "root-1");
    }
}
