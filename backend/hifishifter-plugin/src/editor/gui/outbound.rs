//! 出站桥：把"要发给页面的消息"分类、合并、并按预算取批。
//!
//! 【为什么需要它】`tick()` 每 20 ms 在**宿主 UI 线程**上跑一次，末尾把回执与事件
//! 逐条 `post_message` 给 WebView2。现场一次 REAPER 会话读到的是：
//!
//! ```text
//! 02:08:22.445  [gui] outbound big message bytes=1915519
//! 02:08:53.139  [gui] outbound bytes=64526480 over 60.0s (1075133 B/s)
//! ```
//!
//! 而同一台机器上空闲基线是 **19 400 B/s** —— 高出 55 倍。`PostWebMessageAsJson` 是
//! 同步调用：浏览器进程正在解析 1.9 MB 的 JSON、跑 React、画 canvas 时，这条调用会
//! 阻塞投递方，也就是宿主 UI 线程。宿主线程一旦停在那里就不再泵消息，而 WebView2 的
//! 投递正靠它泵 —— 于是"我们等页面"与"页面等我们"可以互相咬住。
//!
//! 【为什么不是"起一条发送线程"】`EditorPort` 是 `Rc<dyn EditorPort>`：三个平台的
//! 视图句柄都**只属于宿主 UI 线程**（见 `port.rs` 的说明）。把它变成 `Send` 等于逼
//! macOS / Linux 的实现去藏 `Rc`，换不来任何真实保证。所以这一层解决的是**量**，
//! 不是线程：同一批字节不再重复发、状态消息只留最新、每拍有硬预算。
//!
//! 【三类消息的区别】
//!
//! | 类别 | 判据 | 能不能丢 | 为什么 |
//! |---|---|---|---|
//! | `Reply` | 带 `id` | **永不** | 丢一条回执 = 前端那个 promise 挂到 30 s 超时，并占着在途预算 |
//! | `State` | `event` 是 `plugin_apply_state` / `plugin_host_changed` | 合并到最新 | 它们是"去重取一次"的提示；旧的那条只会让前端多取一次 |
//! | `Bulk` | 其余 | 预算内保留，超了丢最旧 | 它们是纯通知（导入进度、诊断读数），旧的没有价值 |

use serde_json::Value;

/// 每拍允许投递的字节上限（20 ms 一拍 ⇒ 最坏 13 MB/s，实测远低于此）。
///
/// 【为什么是 256 KB】它同时是"单条消息不该超过"的量级（`BIG_MESSAGE_BYTES` 是 1 MiB，
/// 那条读数存在的理由就是 1.9 MB 那种消息）。把一拍限制在这个量级，宿主线程每次
/// `post_message` 的阻塞时间就有上界。
pub(crate) const OUTBOUND_BATCH_BYTES: usize = 256 * 1024;

/// 一拍最多投递多少条回执（与 `pump` 里原来的 `take(32)` 同值）。
pub(crate) const OUTBOUND_BATCH_REPLIES: usize = 32;

/// 一条出站消息的类别。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutboundKind {
    Reply,
    State,
    Bulk,
}

impl OutboundKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Reply => "reply",
            Self::State => "state",
            Self::Bulk => "bulk",
        }
    }
}

/// 给一条出站消息分类。
pub(crate) fn classify(value: &Value) -> OutboundKind {
    if value.get("id").is_some() {
        return OutboundKind::Reply;
    }
    match value.get("event").and_then(Value::as_str) {
        // 【为什么这两条算 State】它们是"时间线/状态变了，去取一次"的提示。前端收到
        // 之后做的是**整份重取**；连发 N 条与发 1 条的结果逐字相同。
        Some("plugin_apply_state") | Some("plugin_host_changed") => OutboundKind::State,
        _ => OutboundKind::Bulk,
    }
}

/// 合并一批事件：状态只留最新，其余按字节预算保留。
///
/// 返回 `(要投递的事件, 因为预算被丢掉的事件数)`。
///
/// 【为什么不排序】事件的顺序是它到达的顺序；合并只删除**被取代**的那几条，
/// 不改变仍然存在的那几条的相对顺序。
pub(crate) fn coalesce_events(events: Vec<Value>, budget_bytes: usize) -> (Vec<Value>, u64) {
    let mut newest_state: Option<Value> = None;
    let mut kept: Vec<Value> = Vec::new();
    let mut dropped = 0_u64;
    let mut bytes = 0_usize;
    for event in events {
        // 【回执永远保留】丢一条回执 = 前端那个 promise 挂到 30 s 超时，并一直占着
        // 在途预算（32 条用满之后本视图拒绝一切后续命令）。所以它们不参与任何预算。
        if classify(&event) == OutboundKind::Reply {
            kept.push(event);
            continue;
        }
        if classify(&event) == OutboundKind::State {
            // 旧的那条被这一条取代：它要表达的事实已经在新的里面。
            if newest_state.is_some() {
                dropped += 1;
            }
            newest_state = Some(event);
            continue;
        }
        let size = event.to_string().len();
        if bytes.saturating_add(size) > budget_bytes && !kept.is_empty() {
            // 超预算：丢掉**最旧**的一条，给新的腾位置。旧通知没有价值。
            let oldest = kept.remove(0);
            bytes = bytes.saturating_sub(oldest.to_string().len());
            dropped += 1;
        }
        bytes = bytes.saturating_add(size);
        kept.push(event);
    }
    if let Some(state) = newest_state {
        kept.push(state);
    }
    (kept, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn state(version: u64) -> Value {
        json!({"version":1,"viewId":"v","event":"plugin_host_changed","payload":{"version":version}})
    }

    fn bulk(text: &str) -> Value {
        json!({"version":1,"viewId":"v","event":"import_progress","payload":{"text":text}})
    }

    /// **不变量：连发 100 条状态，只留最新的一条。**
    #[test]
    fn state_messages_coalesce() {
        let events = (0..100).map(state).collect::<Vec<_>>();
        let (kept, dropped) = coalesce_events(events, OUTBOUND_BATCH_BYTES);
        assert_eq!(kept.len(), 1, "状态消息必须只留一条");
        assert_eq!(dropped, 99);
        assert_eq!(
            kept[0]["payload"]["version"],
            json!(99),
            "留下的必须是最后一个值"
        );
    }

    /// **不变量：回执永远不被合并、不被丢弃。**
    #[test]
    fn replies_are_never_dropped() {
        let mut events = (0..200)
            .map(|index| json!({"version":1,"viewId":"v","id":index,"ok":true}))
            .collect::<Vec<_>>();
        events.push(state(1));
        let (kept, dropped) = coalesce_events(events, 1);
        assert_eq!(dropped, 0, "回执不参与合并");
        assert_eq!(kept.len(), 201, "200 条回执 + 1 条状态，一条都不能少");
    }

    /// 批量通知按字节预算丢最旧的，而不是丢最新的。
    #[test]
    fn bulk_is_bounded_and_drops_the_oldest() {
        let events = (0..50).map(|_| bulk("xxxxxxxxxx")).collect::<Vec<_>>();
        let (kept, dropped) = coalesce_events(events, 200);
        assert!(dropped > 0, "超预算必须真的丢掉一些");
        assert!(
            kept.iter()
                .map(|event| event.to_string().len())
                .sum::<usize>()
                <= 200,
            "留下的事件必须落在预算内"
        );
    }

    /// 分类判据本身。
    #[test]
    fn the_kind_follows_the_payload_shape() {
        assert_eq!(classify(&json!({"id":1,"ok":true})), OutboundKind::Reply);
        assert_eq!(classify(&state(1)), OutboundKind::State);
        assert_eq!(
            classify(&json!({"event":"plugin_apply_state","payload":{}})),
            OutboundKind::State
        );
        assert_eq!(classify(&bulk("x")), OutboundKind::Bulk);
    }
}
