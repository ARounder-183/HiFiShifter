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
use hifishifter_kernel::host::{HostCallbacks, SharedHostCallbacks};
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

// ── 内核找宿主的出口 ────────────────────────────────────────────────────────
//
// 【为什么不放在 AppState 里】`AppState` 是 app 层的运行时容器，而这两个回调
// 在装配期就要可用；进程级单例与上面的 `APP_HANDLE` 同一做法。

/// app 层对内核宿主回调的实现。
///
/// 【为什么只做这三件事】实测：内核闭包里唯一一处**生产代码**越界，就是
/// `pitch_analysis/schedule.rs` 去够 `commands::playback` 的后台渲染开关与请求函数。
/// 把这三件事收成回调之后，音高分析不再认识 `commands`，
/// 内核闭包里那 6 个 app 层模块（`commands` / `recording` / `search` /
/// `system_clipboard` / `linux_clipboard` 及 `commands` 子模块）就全掉出去了。
struct AppHostCallbacks;

impl HostCallbacks for AppHostCallbacks {
    fn auto_background_render_enabled(&self) -> bool {
        crate::commands::playback::AUTO_BG_RENDER_ENABLED
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn take_pitch_pending_flag(&self) -> bool {
        // 「读并清零」必须是一次原子操作：拆开会让它被消费两次（补触发两次渲染）。
        crate::commands::playback::BG_RENDER_PITCH_PENDING
            .swap(false, std::sync::atomic::Ordering::AcqRel)
    }

    fn request_background_render(&self) {
        if let Some(handle) = app_handle() {
            let _ = crate::commands::playback::request_background_render(handle);
        }
    }
}

/// 装配内核的宿主出口。由 Tauri setup 调用一次（在 [`install`] 之后）。
pub fn install_host_callbacks() -> bool {
    let callbacks: SharedHostCallbacks = Arc::new(AppHostCallbacks);
    hifishifter_kernel::host::host().install(callbacks)
}
