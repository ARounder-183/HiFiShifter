//! app 层的事件出口与 `AppHandle` 持有者。
//!
//! 背景：内核模块（`state` / `pitch_clip` / `recording` / `pitch_analysis` /
//! `audio_engine`）里那几十处 `tauri::Emitter` 调用，是"内核离不开 Tauri"的唯一原因。
//! 这里把 Tauri 的两件事收口到 app 层：
//!
//! 1. 把 `AppHandle` 包成内核认识的 `EventSink`，供内核模块使用；
//! 2. 给 app 层自己留一个进程级 `AppHandle` 出口 —— 因为 app 层还需要它做
//!    窗口（`get_webview_window`）、路径（`path`）、状态（`state`）这些**不只是发事件**
//!    的事，那些调用点本来就该留在 app 层。
//!
//! 迁移方向：内核模块改从 `EventSink` 取值；app 层改用 [`app_handle`]；
//! 两边都改完之后，`AppState.app_handle` 这个字段就可以删掉，内核侧不再出现 `tauri::`。

use hifishifter_kernel::events::{EventSink, SharedEventSink};
use std::sync::{Arc, OnceLock};
use tauri::{AppHandle, Emitter};

/// 进程级 `AppHandle`：由 Tauri setup 注册一次。
static APP_HANDLE: OnceLock<AppHandle> = OnceLock::new();
/// 进程级内核事件出口。
static EVENT_SINK: OnceLock<SharedEventSink> = OnceLock::new();

/// 把 `AppHandle` 适配成内核的 `EventSink`。
struct TauriEventSink(AppHandle);

impl EventSink for TauriEventSink {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        let _ = self.0.emit(event, payload);
    }
}

/// 由 Tauri setup 调用一次，注册 `AppHandle` 与内核事件出口。
pub fn install(handle: &AppHandle) {
    let _ = APP_HANDLE.set(handle.clone());
    let _ = EVENT_SINK.set(Arc::new(TauriEventSink(handle.clone())));
}

/// app 层自己的 `AppHandle` 出口（窗口 / 路径 / state）。
pub fn app_handle() -> Option<&'static AppHandle> {
    APP_HANDLE.get()
}

/// 内核事件出口；供 `AppState` 持有一份，内核模块据此发事件。
pub fn event_sink() -> Option<SharedEventSink> {
    EVENT_SINK.get().cloned()
}
