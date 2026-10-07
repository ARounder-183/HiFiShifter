//! 文档共享宿主Undo调度：GUI只负责主线程API，actor在原生历史收尾前暂缓神经合成。
use super::reaper::{HostUndoBlock, ReaperHost};
use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    block: Option<HostUndoBlock>,
    requests: HashSet<(String, u64)>,
    native_requests: HashSet<(String, u64)>,
    released: HashSet<String>,
    histories: HashSet<(String, u64)>,
    groups: HashMap<String, u32>,
    idle: Option<Instant>,
    starting: bool,
    closing: bool,
}
#[derive(Default)]
pub(crate) struct HostUndo {
    state: Mutex<State>,
    /// 只控制自动合成，不阻塞参数命令/flush屏障，否则宿主getState会与actor循环等待。
    pub pending: AtomicBool,
}
impl HostUndo {
    pub(crate) fn begin_group(&self, view: &str) {
        let mut state = self.state.lock().unwrap();
        *state.groups.entry(view.into()).or_default() += 1;
        state.idle = None;
    }
    pub(crate) fn end_group(&self, view: &str) {
        {
            let mut state = self.state.lock().unwrap();
            if let Some(depth) = state.groups.get_mut(view) {
                *depth = depth.saturating_sub(1);
                if *depth == 0 {
                    state.groups.remove(view);
                }
            }
        }
        self.finish_if_idle(false);
    }
    /// 新检查点先收尾上次非分组操作；分块尾笔checkpoint=false留在同一150ms批次中。
    pub(crate) fn begin_request(
        &self,
        view: &str,
        id: u64,
        checkpoint: bool,
        native: bool,
        host: &Arc<ReaperHost>,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if checkpoint {
            self.finish_if_idle(false);
        }
        let start = {
            let mut state = self.state.lock().unwrap();
            if state.starting || state.closing {
                return Err("host undo reentry while starting/closing block".into());
            }
            if state.requests.len() >= 128 || !state.requests.insert((view.into(), id)) {
                return Err("host undo request budget/identity rejected".into());
            }
            if native {
                state.native_requests.insert((view.into(), id));
            }
            state.idle = None;
            let start = state.block.is_none();
            if start {
                state.starting = true;
            }
            start
        };
        if start {
            self.pending.store(true, Ordering::Release);
            let block = host.begin_project_undo(authorized);
            let mut state = self.state.lock().unwrap();
            state.starting = false;
            match block {
                Ok(block) => state.block = Some(block),
                Err(error) => {
                    state.requests.remove(&(view.into(), id));
                    state.native_requests.remove(&(view.into(), id));
                    self.pending
                        .store(!state.histories.is_empty(), Ordering::Release);
                    return Err(error);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn finish_request(&self, view: &str, id: u64) {
        let mut state = self.state.lock().unwrap();
        state.requests.remove(&(view.into(), id));
        state.native_requests.remove(&(view.into(), id));
        if state.requests.is_empty() {
            state.idle = Some(Instant::now() + Duration::from_millis(150));
        }
        let released = state.released.contains(view);
        if !state.requests.iter().any(|(owner, _)| owner == view) {
            state.released.remove(view);
        }
        drop(state);
        if released {
            self.finish_if_idle(false);
        }
    }
    /// UI timer收尾时不持Mutex调用宿主；pending到Undo_EndBlock2返回后才撤销。
    pub(crate) fn tick(&self) {
        self.finish_if_idle(true);
    }
    /// Ctrl+Z先等已入队写入的响应，再结束活动手势组；返回false表示尚有在途请求。
    pub(crate) fn finish_for_history(&self) -> bool {
        {
            let mut state = self.state.lock().unwrap();
            if state.starting || state.closing || !state.requests.is_empty() {
                return false;
            }
            state.groups.clear();
        }
        self.finish_if_idle(false);
        true
    }
    pub(crate) fn finish_if_idle(&self, wait_deadline: bool) {
        let block = {
            let mut state = self.state.lock().unwrap();
            if state.starting
                || state.closing
                || !state.requests.is_empty()
                || !state.groups.is_empty()
                || wait_deadline && state.idle.is_none_or(|deadline| deadline > Instant::now())
            {
                return;
            }
            state.idle = None;
            let block = state.block.take();
            state.closing = block.is_some();
            block
        };
        if block.is_some() {
            drop(block);
            let mut state = self.state.lock().unwrap();
            state.closing = false;
            self.pending
                .store(!state.histories.is_empty(), Ordering::Release);
        }
    }
    pub(crate) fn begin_history(&self, view: &str, id: u64) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if state.histories.len() >= 32 || !state.histories.insert((view.into(), id)) {
            return Err("host history request budget/identity rejected".into());
        }
        self.pending.store(true, Ordering::Release);
        Ok(())
    }
    pub(crate) fn finish_history(&self, view: &str, id: u64) {
        let mut state = self.state.lock().unwrap();
        state.histories.remove(&(view.into(), id));
        self.pending.store(
            state.block.is_some() || state.starting || state.closing || !state.histories.is_empty(),
            Ordering::Release,
        );
    }
    /// 窗口关闭不能遗留整个工程的Undo组；已排队参数尾笔由getState的actor屏障保存。
    pub(crate) fn release_view(&self, view: &str) {
        {
            let mut state = self.state.lock().unwrap();
            state.groups.remove(view);
            let native = state
                .native_requests
                .iter()
                .filter(|(owner, _)| owner == view)
                .cloned()
                .collect::<HashSet<_>>();
            state
                .requests
                .retain(|(owner, id)| owner != view || native.contains(&(owner.clone(), *id)));
            if !native.is_empty() {
                state.released.insert(view.into());
            }
            state.histories.retain(|(owner, _)| owner != view);
        }
        self.finish_if_idle(false);
        let state = self.state.lock().unwrap();
        self.pending.store(
            state.block.is_some() || state.starting || state.closing || !state.histories.is_empty(),
            Ordering::Release,
        );
    }
    pub(crate) fn close(&self) {
        let block = {
            let mut state = self.state.lock().unwrap();
            state.groups.clear();
            state.requests.clear();
            state.native_requests.clear();
            state.released.clear();
            state.histories.clear();
            state.idle = None;
            state.closing = true;
            state.block.take()
        };
        drop(block);
        self.pending.store(false, Ordering::Release);
        self.state.lock().unwrap().closing = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 一次多块参数提交/显式几何组只有一个原生Undo块，收尾回调期间pending仍有效。
    #[test]
    fn host_history_groups_chunks_and_closes_outside_mutex() {
        let fixture = crate::host::reaper::ReaperFixture::new();
        fixture.enable_writer();
        let host = Arc::new(fixture.client());
        let undo = Arc::new(HostUndo::default());
        undo.begin_group("view");
        undo.begin_request("view", 1, true, false, &host, &|| true)
            .unwrap();
        undo.finish_request("view", 1);
        undo.begin_request("view", 2, false, false, &host, &|| true)
            .unwrap();
        undo.finish_request("view", 2);
        let checked = undo.clone();
        *fixture.hook.borrow_mut() = Some((
            "undo-end".into(),
            Box::new(move || {
                assert!(checked.pending.load(Ordering::Acquire));
                assert!(
                    checked.state.try_lock().is_ok(),
                    "调用宿主End不能持有自身mutex"
                );
            }),
        ));
        undo.end_group("view");
        assert!(!undo.pending.load(Ordering::Acquire));
        let calls = fixture.calls();
        assert_eq!(
            calls.iter().filter(|s| s.as_str() == "undo-begin").count(),
            1
        );
        assert_eq!(calls.iter().filter(|s| s.as_str() == "undo-end").count(), 1);
        let snapshot = host.project_history(&|| true).unwrap();
        assert_eq!(snapshot["undoDepth"], 1);
        assert!(host.history_jump(false, &|| true).unwrap());
        assert_eq!(host.project_history(&|| true).unwrap()["redoDepth"], 1);
        host.history_jump_to(1, &|| true).unwrap();
        assert_eq!(host.project_history(&|| true).unwrap()["position"], 1);
    }
    /// 关闭UI发生在原生setter内部时，不能提前结束Undo让余下setter落到历史之外。
    #[test]
    fn host_history_view_close_waits_for_native_writer_but_releases_queued_actor_tail() {
        let fixture = crate::host::reaper::ReaperFixture::new();
        fixture.enable_writer();
        let host = Arc::new(fixture.client());
        let undo = HostUndo::default();
        undo.begin_request("view", 1, true, true, &host, &|| true)
            .unwrap();
        undo.release_view("view");
        assert!(undo.pending.load(Ordering::Acquire));
        assert!(!fixture.calls().contains(&"undo-end".into()));
        undo.finish_request("view", 1);
        assert!(!undo.pending.load(Ordering::Acquire));
        undo.begin_request("other", 2, true, false, &host, &|| true)
            .unwrap();
        undo.release_view("other");
        assert!(!undo.pending.load(Ordering::Acquire));
        assert_eq!(host.project_history(&|| true).unwrap()["undoDepth"], 2);
    }
    /// 宿主getState使用的flush屏障在Undo打开期间仍能执行；只有自动重推理被延后。
    #[test]
    fn host_history_pending_defers_automatic_apply_not_editor_commands_or_flush() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let fixture = crate::host::reaper::ReaperFixture::new();
        fixture.enable_writer();
        let host = Arc::new(fixture.client());
        let (reply, rx) = std::sync::mpsc::channel();
        let (events, _) = std::sync::mpsc::sync_channel(32);
        let sink = crate::editor::session::UiSink {
            view_id: "history-actor".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let call = |id, command: &str, args| {
            editor
                .enqueue(crate::editor::session::UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sink.clone(),
                    link: None,
                })
                .unwrap();
            let response: serde_json::Value = rx.recv_timeout(Duration::from_secs(3)).unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let timeline = call(1, "get_timeline_state", serde_json::json!({}));
        document
            .host_undo
            .begin_request("history-actor", 2, true, false, &host, &|| {
                document.is_alive()
            })
            .unwrap();
        call(
            2,
            "set_track_state",
            serde_json::json!({"trackId":timeline["tracks"][0]["id"],"volume":0.5}),
        );
        std::thread::sleep(Duration::from_millis(250));
        let state = call(3, "plugin_get_apply_state", serde_json::json!({}));
        assert_eq!(state["generation"], 1);
        assert_eq!(state["applied_generation"], 0);
        editor.flush().unwrap();
        document.host_undo.finish_request("history-actor", 2);
        assert!(document.host_undo.finish_for_history());
        assert!(!document.host_undo.pending.load(Ordering::Acquire));
        document.close();
    }
}
