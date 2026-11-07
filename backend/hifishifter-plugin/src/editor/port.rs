//! 页面命令的**唯一**入口：任何格式外壳都经它把命令送进共享编辑会话。
//!
//! 【为什么需要这一层】编辑命令的语义只有一套（219 条，全在 [`crate::editor::commands`]
//! 与 [`crate::editor::session::EditorSession`] 里）：排队、票据、历史、渲染失效、
//! 宿主写回。而**页面到会话**的胶水此前按格式各写一份 —— VST3 的 GUI 线程里一份，
//! CLAP 外壳里另一份（那份只会回"未接线"）。两份胶水的差别本来只是"回执信封长什么样"，
//! 却让同一个命令在两种格式下有两种行为。
//!
//! 这里把胶水收成一处：[`EditorPort::request`] 只做"排队 + 等回执 + 拆信封"，
//! 格式外壳负责把自己的页面消息翻译成 [`command`, `args`]，再把结果包回自己的信封。
//! 命令本身一条都不在这里实现 —— 那是唯一实现该在的地方。

use crate::editor::session::{EditorSession, UiRequest, UiSink};
use crate::render::extension::ExtensionOwner;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// 等一条命令回执的上限。
///
/// 【为什么与 VST3 页面同值】前端自己的超时（`commandTimeoutMs`）比它长，因此插件先
/// 超时会把一条**确定性的错误**交给页面；反过来则页面先放弃、命令仍在跑，用户重试时
/// 会叠出第二条。
const REPLY_TIMEOUT: Duration = Duration::from_secs(30);

/// 一个编辑器视图的命令端口。
pub struct EditorPort {
    owner: Arc<ExtensionOwner>,
    sink: UiSink,
    replies: Mutex<Receiver<Value>>,
    events: Mutex<Receiver<Value>>,
    next_id: AtomicU64,
}

impl EditorPort {
    /// 为一个视图建立端口。
    ///
    /// `view_id` 会出现在每条事件与回执里，页面据此丢弃别的视图的消息。
    pub fn new(owner: Arc<ExtensionOwner>, view_id: &str) -> Self {
        let (reply, replies) = mpsc::channel();
        let (events, event_receiver) = mpsc::sync_channel(256);
        Self {
            owner,
            sink: UiSink {
                view_id: view_id.to_owned(),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            },
            replies: Mutex::new(replies),
            events: Mutex::new(event_receiver),
            next_id: AtomicU64::new(1),
        }
    }

    /// 本端口此刻是否挂在一份活文档上。
    ///
    /// 【为什么由调用方先问】Capture 形态没有 ARA 文档，编辑会话无从建立；外壳必须
    /// 能区分"本形态没有编辑后端"与"命令失败"，否则用户看到的是一句误导的错误。
    pub fn is_bound(&self) -> bool {
        self.owner.is_ara_bound()
    }

    /// 把一条命令送进共享编辑会话并等回执。
    ///
    /// 返回的是**命令的返回值**（回执信封由外壳按自己的页面协议包），错误是可直接展示
    /// 的字符串 —— 与 VST3 页面收到的同一条文案，因为它们出自同一个 `dispatch`。
    pub fn request(&self, command: &str, args: Value) -> Result<Value, String> {
        // 【为什么 Capture 命令在这里拦下】它的全部状态都在 `ExtensionOwner` 上（捕获
        // 存储、take、观察位），而会话分发只拿得到 `EditorSession`（它到不了 owner）。
        // VST3 的视图层走的是同一个 `capture::commands::handle` —— 两种格式因此不会在
        // “按录制到底做了什么”上分叉。
        if crate::capture::commands::is_capture_command(command) {
            return crate::capture::commands::handle(&self.owner, command, &args);
        }
        let session: Arc<EditorSession> = self.owner.editor_session()?;
        let id = self.next_id.fetch_add(1, Ordering::AcqRel);
        session.enqueue(UiRequest {
            id,
            command: command.to_owned(),
            args,
            sink: self.sink.clone(),
            link: None,
        })?;
        let replies = self.replies.lock().map_err(|_| "editor reply channel poisoned")?;
        loop {
            let reply = replies
                .recv_timeout(REPLY_TIMEOUT)
                .map_err(|_| format!("editor command {command} timed out"))?;
            // 【为什么按 id 过滤】同一视图上可能有多条命令在途（页面并行发），回执到达
            // 顺序不保证；把别人的回执当自己的会返回**另一个命令的**结果。
            if reply.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            return if reply.get("ok").and_then(Value::as_bool) == Some(true) {
                Ok(reply.get("value").cloned().unwrap_or(Value::Null))
            } else {
                Err(reply
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("editor command failed")
                    .to_owned())
            };
        }
    }

    /// 取走本视图待发的页面事件（分析进度、渲染状态等）。
    ///
    /// 【为什么是轮询而不是回调】事件必须由外壳在**它自己的**线程上投递给页面
    /// （WebView2 的 `PostWebMessageAsJson` 有线程亲和性），而编辑会话在 actor 线程上
    /// 产生事件。轮询把线程边界留在外壳里，端口不假设任何 UI 运行时。
    pub fn poll_events(&self) -> Vec<Value> {
        let events = match self.events.lock() {
            Ok(events) => events,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut drained = Vec::new();
        while let Ok(event) = events.try_recv() {
            drained.push(event);
        }
        drained
    }

    /// 关闭端口：之后到达的事件直接丢弃，页面不会再收到已消失视图的消息。
    pub fn close(&self) {
        self.sink.closed.store(true, Ordering::Release);
    }
}

impl Drop for EditorPort {
    fn drop(&mut self) {
        self.close();
    }
}
