//! 内核异步事件按文档私有ID/源路径分流；同文档全部活view共享事件，不跨工程广播。
use super::session::EditorSession;
use std::sync::{Arc, Mutex, OnceLock, Weak};
struct Router {
    sessions: Mutex<Vec<Weak<EditorSession>>>,
}
/// 一条内核事件该不该投递给某个会话（纯判据，便于单测）。
///
/// `text` 是序列化后的 payload，`namespace` 是该会话的私有 id 前缀，
/// `session_count` 是此刻活着的会话数。
fn should_route(event: &str, text: &str, namespace: &str, session_count: usize) -> bool {
    // Capture 形态的根 id 不带命名空间前缀（`analysis_timeline` 在没有文档时不做
    // id 重写），而分析完成事件的 payload 只有 `rootTrackId`。命名空间一刀切会把
    // `pitch_orig_updated` 全部丢掉 —— 原线算完却永远不通知前端。
    let is_analysis_event = matches!(
        event,
        "pitch_orig_updated"
            | "pitch_orig_analysis_started"
            | "pitch_orig_analysis_progress"
            | "dyn_orig_updated"
    );
    (is_analysis_event && session_count == 1) || text.contains(namespace)
}

impl hifishifter_kernel::events::EventSink for Router {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        let sessions = {
            let mut known = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            known.retain(|s| s.strong_count() > 0);
            known.iter().filter_map(Weak::upgrade).collect::<Vec<_>>()
        };
        let text = payload.to_string();
        // 这几类事件由会话自己的分析快照产生：单会话时归属无歧义，直接投递；
        // 多会话时仍按命名空间精确分流，绝不跨工程广播。
        let session_count = sessions.len();
        for session in sessions {
            if should_route(event, &text, &session.namespace, session_count) {
                session.emit(event, payload.clone());
            }
        }
    }
}
/// 注册weak，不增加会话寿命；进程级内核出口只安装一次，不覆盖其它已安装宿主。
pub(super) fn register(session: &Arc<EditorSession>) {
    static ROUTER: OnceLock<Arc<Router>> = OnceLock::new();
    let router = ROUTER.get_or_init(|| {
        let router = Arc::new(Router {
            sessions: Mutex::new(vec![]),
        });
        if !hifishifter_kernel::events::events().install(router.clone()) {
            crate::log_line("kernel event sink already installed; embedded routing unavailable");
        }
        router
    });
    router
        .sessions
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(session));
}

#[cfg(test)]
mod tests {
    use super::should_route;

    /// **Capture 形态：分析完成事件必须送达，尽管 rootTrackId 不带命名空间。**
    ///
    /// 【为什么这条是现场报障的正身】捕获结束后 `pitch_clip` 算完、`pitch_orig`
    /// 也写进时间线了，但 `pitch_orig_updated` 的 payload 只有
    /// `{"rootTrackId":"capture-track"}`。旧判据 `text.contains(namespace)` 恒假 ⇒
    /// 事件被静默丢弃 ⇒ 前端从不重取原线，用户只能自己画一笔触发刷新。
    #[test]
    fn a_capture_pitch_event_reaches_its_only_session() {
        let payload = r#"{"rootTrackId":"capture-track"}"#;
        assert!(
            should_route("pitch_orig_updated", payload, "hfs-ui-1234-1-", 1),
            "唯一的 Capture 会话必须收到自己的分析完成事件"
        );
        assert!(should_route(
            "pitch_orig_analysis_progress",
            payload,
            "hfs-ui-1234-1-",
            1
        ));
    }

    /// 普通事件仍按命名空间分流；多会话时不能靠"唯一会话"兜底跨工程广播。
    #[test]
    fn ordinary_events_and_multi_session_stay_scoped() {
        let payload = r#"{"rootTrackId":"capture-track"}"#;
        assert!(!should_route(
            "plugin_host_changed",
            payload,
            "hfs-ui-1234-1-",
            1
        ));
        assert!(!should_route(
            "pitch_orig_updated",
            payload,
            "hfs-ui-1234-1-",
            2
        ));
        // ARA 形态的根 id 带前缀，多会话下也能精确命中。
        assert!(should_route(
            "pitch_orig_updated",
            r#"{"rootTrackId":"hfs-ui-1234-1-track"}"#,
            "hfs-ui-1234-1-",
            2
        ));
    }
}
