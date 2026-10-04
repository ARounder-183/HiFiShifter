//! 内核异步事件按实例私有ID/源路径分流；不把另一FX的音高分析通知广播给当前GUI。
use super::session::EditorSession;
use std::sync::{Arc,Mutex,OnceLock,Weak};
struct Router {sessions:Mutex<Vec<Weak<EditorSession>>>}
impl hifishifter_kernel::events::EventSink for Router {
    fn emit(&self,event:&str,payload:serde_json::Value) {
        let sessions={let mut known=self.sessions.lock().unwrap_or_else(|e|e.into_inner());
            known.retain(|s|s.strong_count()>0);known.iter().filter_map(Weak::upgrade).collect::<Vec<_>>()};
        let text=payload.to_string();
        for session in sessions {if text.contains(&session.namespace) {session.emit(event,payload.clone());}}
    }
}
/// 注册weak，不增加会话寿命；进程级内核出口只安装一次，不覆盖其它已安装宿主。
pub(super) fn register(session:&Arc<EditorSession>) {
    static ROUTER:OnceLock<Arc<Router>>=OnceLock::new();
    let router=ROUTER.get_or_init(|| {
        let router=Arc::new(Router {sessions:Mutex::new(vec![])});
        if !hifishifter_kernel::events::events().install(router.clone()) {
            crate::log_line("kernel event sink already installed; embedded routing unavailable");
        }
        router
    });
    router.sessions.lock().unwrap_or_else(|e|e.into_inner()).push(Arc::downgrade(session));
}
