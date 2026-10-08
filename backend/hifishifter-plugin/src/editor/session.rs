//! 真实ARA文档共享原编辑命令actor：UI只排队，分析/文件/离线渲染均不在音频或UI回调执行。
use crate::render::document::DocumentSession;
use hifishifter_kernel::editor::{
    history,
    host_pcm::{materialize_with_byte_limit, PcmView},
    ParamHost,
};
use hifishifter_kernel::state::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, Weak};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub(crate) struct UiSink {
    pub view_id: String,
    pub reply: mpsc::Sender<Value>,
    pub events: mpsc::SyncSender<Value>,
    pub closed: Arc<AtomicBool>,
}
impl UiSink {
    fn response(&self, value: Value) {
        if !self.closed.load(Ordering::Acquire) {
            let _ = self.reply.send(value);
        }
    }
    fn event(&self, value: Value) {
        if !self.closed.load(Ordering::Acquire) {
            let _ = self.events.try_send(value);
        }
    }
}
pub(crate) struct UiRequest {
    pub id: u64,
    pub command: String,
    pub args: Value,
    pub sink: UiSink,
    pub link: Option<Arc<super::routing::EditorLink>>,
}
enum Job {
    Request(UiRequest, Option<u64>),
    Barrier(mpsc::Sender<()>),
    Close,
}
/// 单个可取消DSP任务；编辑actor只收结果，计算线程不改GUI历史或参数权威。
struct RenderTask {
    ticket: u64,
    generation: u64,
    cancel: Arc<AtomicBool>,
    reply: mpsc::Receiver<Result<(), String>>,
    worker: std::thread::JoinHandle<()>,
}
impl RenderTask {
    fn spawn(
        ticket: u64,
        generation: u64,
        cancel: Arc<AtomicBool>,
        work: impl FnOnce() -> Result<(), String> + Send + 'static,
    ) -> Result<Self, String> {
        let (sender, reply) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("hfs-editor-dsp".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .unwrap_or_else(|_| Err("automatic render panicked".into()));
                let _ = sender.send(result);
            })
            .map_err(|error| format!("create editor DSP worker: {error}"))?;
        Ok(Self {
            ticket,
            generation,
            cancel,
            reply,
            worker,
        })
    }
}
struct RegisteredView {
    sink: UiSink,
    route: Option<(Arc<super::routing::EditorLink>, u64)>,
}
#[derive(Default)]
struct Loaded {
    initialized: bool,
    edit: u64,
    model: u64,
    scope: u64,
    audio: u64,
    projection: String,
    reverse_paths: HashMap<String, String>,
}
pub(crate) struct EditorSession {
    pub(super) document: Weak<DocumentSession>,
    pub(super) timeline: Mutex<TimelineState>,
    pub(super) history: Mutex<TimelineHistory>,
    pub(super) project: Mutex<ProjectState>,
    // 设置**不在**这里：它属于用户，不属于某一个 ARA 文档。此前它是本结构体的字段，
    // 于是每个工程一份、初始化为出厂默认，换工程即丢。现在由进程级的
    // `crate::settings_store` 持有（与独立 App 共用同一份配置文件）。
    loaded: Mutex<Loaded>,
    pub(super) namespace: String,
    pub(super) browser_roots: Mutex<Vec<PathBuf>>,
    // 只保留已获ARA授权后生成的GUI媒体元信息/路径，不保留或复用实时播放PCM。
    display_waveforms: Mutex<HashMap<String, (String, hifishifter_kernel::state::Clip)>>,
    pcm_dir: PathBuf,
    pub(super) peaks: Mutex<HashMap<String, Arc<hifishifter_kernel::hfspeaks_v2::HfsPeakFile>>>,
    queue: mpsc::SyncSender<Job>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    analysis_sender: mpsc::Sender<hifishifter_kernel::engine_command::EngineCommand>,
    analysis_updates: Mutex<mpsc::Receiver<hifishifter_kernel::engine_command::EngineCommand>>,
    analysis_cancel: Arc<AtomicBool>,
    analysis_workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    views: Mutex<HashMap<String, RegisteredView>>,
    pub(super) generation: AtomicU64,
    submitted: AtomicU64,
    processed: AtomicU64,
    pub(super) applied: AtomicU64,
    host_version: AtomicU64,
    ui_geometry_version: AtomicU64,
    project_duration: AtomicU64,
    pub(super) error: Mutex<Option<String>>,
    render_error: Mutex<Option<String>>,
    render_requested: AtomicBool,
    render_task: Mutex<Option<RenderTask>>,
    closed: AtomicBool,
    transport_probe: bool,
    transport_probe_next: Mutex<Instant>,
    pub(super) suppress_history: AtomicBool,
}

impl Drop for EditorSession {
    /// 会话结束时清掉自己的 PCM 暂存目录。
    ///
    /// 【为什么必须有】这个目录是按命名空间（含 pid 与序号）建的，宿主每次启动都会
    /// 得到新目录；没有回收点就会在磁盘上无限累积渲染用的 PCM。清理失败只留日志，
    /// 不影响关闭流程。
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.pcm_dir) {
            // 目录可能从未被创建过（没有渲染发生）—— 那不算错误。
            if error.kind() != std::io::ErrorKind::NotFound {
                crate::log_line(&format!("plugin PCM scratch cleanup failed: {error}"));
            }
        }
    }
}

impl EditorSession {
    /// 导入目标只接受当前原GUI已授权轨道；未指定时由native直接parent决定空轨首次导入。
    pub(super) fn host_track_id(&self, id: &str) -> Result<String, String> {
        if !self
            .timeline
            .lock()
            .unwrap()
            .tracks
            .iter()
            .any(|track| track.id == id)
        {
            return Err("unknown editor import track".into());
        }
        id.strip_prefix(&self.namespace)
            .map(str::to_owned)
            .ok_or_else(|| "import track belongs to another editor".into())
    }
    pub(super) fn ui_clip_id(&self, native: &str) -> String {
        format!("{}{native}", self.namespace)
    }
    /// 原App轨道拖动只形成插件参数分组；一次事务保存GUID关系，不调用任何宿主轨道setter。
    pub(super) fn move_private_track(
        &self,
        id: &str,
        index: usize,
        parent: Option<String>,
    ) -> Result<(), String> {
        let document = self.document.upgrade().ok_or("document closed")?;
        let before = self.timeline.lock().unwrap().clone();
        if !before.tracks.iter().any(|track| track.id == id) || index > 10000 {
            return Err("unknown private track or invalid index".into());
        }
        let mut candidate = before.clone();
        candidate.move_track(id, index, parent.clone());
        if candidate
            .tracks
            .iter()
            .find(|track| track.id == id)
            .unwrap()
            .parent_id
            != parent
        {
            return Err("private track parent is missing or cyclic".into());
        }
        let groups = super::private_groups::TrackGroups::capture(
            &candidate,
            &document.group_aliases(&self.namespace),
        )?;
        let versions = document.editor_versions()?;
        let selected = candidate
            .selected_clip_id
            .as_deref()
            .map(|id| id.strip_prefix(&self.namespace).unwrap_or(id).to_owned());
        let (_, grouped) = document.private_parameter_views(selected, Some(&groups))?;
        let revision = {
            let _transaction = document.transaction.lock().unwrap();
            let mut edits = document.edits.lock().unwrap();
            if (
                edits.revision,
                document.revision.load(Ordering::Acquire),
                document.scope_revision.load(Ordering::Acquire),
            ) != versions
            {
                return Err("Conflict: host changed during private track grouping".into());
            }
            let revision = edits
                .revision
                .checked_add(1)
                .ok_or("edit revision exhausted")?;
            edits.groups = groups;
            edits.revision = revision;
            revision
        };
        candidate.params_by_root_track = grouped
            .params_by_root_track
            .into_iter()
            .map(|(root, params)| (format!("{}{root}", self.namespace), params))
            .collect();
        self.checkpoint_timeline(&before, HistoryOp::MoveTrack);
        *self.timeline.lock().unwrap() = candidate;
        {
            let mut loaded = self.loaded.lock().unwrap();
            loaded.edit = revision;
            loaded.projection = document.workspace_projection()?;
        }
        self.mark_dirty();
        self.render_requested.store(true, Ordering::Release);
        self.schedule_analysis();
        self.notify_timeline();
        Ok(())
    }
    /// 每个真实文档唯一actor；worker与会话均用weak，不形成document→actor→document循环。
    pub(crate) fn new(document: &Arc<DocumentSession>) -> Result<Arc<Self>, String> {
        super::resources::initialize_models();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let namespace = format!(
            "hfs-ui-{}-{}-",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let (queue, receiver) = mpsc::sync_channel(32);
        let (analysis_sender, analysis_updates) = mpsc::channel();
        // 【为什么从设置里播种网格】网格是 HiFiShifter 自有的编辑设置（宿主没有对应
        // 概念），存在进程级的 `settings_store` 里。`ProjectState::default()` 只带出厂
        // 值，不播种的话用户在插件里设过的网格会在换工程/重启后归零 —— 而设置本该
        // 属于用户、不属于某一个工程。
        let initial_project = {
            let settings = crate::settings_store::settings();
            let mut project = ProjectState::default();
            project.grid_size =
                hifishifter_kernel::config::TimelineSnapSettings::normalize_grid_size(
                    &settings.grid_size,
                );
            // 音阶是 HiFiShifter 自有的设置（宿主没有对应概念），必须从用户设置里
            // 播种 —— 否则用户在插件里选过的音阶会在换工程/重启后归零，而渲染缓存键
            // 与级数渲染都锚定它。
            let musical = settings.plugin_musical_context;
            project.base_scale =
                hifishifter_kernel::state::model::normalize_scale_key(&musical.base_scale);
            project.use_custom_scale = musical.use_custom_scale && musical.custom_scale.is_some();
            project.custom_scale = musical.custom_scale.map(|scale| scale.normalized());
            project
        };
        let session = Arc::new(Self {
            document: Arc::downgrade(document),
            timeline: Mutex::new(TimelineState::default()),
            history: Mutex::new(Default::default()),
            project: Mutex::new(initial_project),
            loaded: Mutex::new(Default::default()),
            // 渲染用的 PCM 暂存同样搬出 `%TEMP%`：磁盘清理在会话进行中删掉这些
            // 文件会让正在进行的渲染失败。命名空间含 pid 与序号，多个宿主进程
            // 之间不会互相覆盖。
            pcm_dir: hifishifter_kernel::config_location::local_data_subdir("pcm").join(&namespace),
            namespace,
            browser_roots: Mutex::new(Vec::new()),
            display_waveforms: Mutex::new(HashMap::new()),
            peaks: Mutex::new(HashMap::new()),
            queue,
            worker: Mutex::new(None),
            analysis_sender,
            analysis_updates: Mutex::new(analysis_updates),
            views: Mutex::new(HashMap::new()),
            analysis_cancel: Arc::new(AtomicBool::new(false)),
            analysis_workers: Mutex::new(Vec::new()),
            generation: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            applied: AtomicU64::new(0),
            host_version: AtomicU64::new(0),
            ui_geometry_version: AtomicU64::new(0),
            project_duration: AtomicU64::new(0),
            error: Mutex::new(None),
            render_error: Mutex::new(None),
            render_requested: AtomicBool::new(false),
            render_task: Mutex::new(None),
            closed: AtomicBool::new(false),
            suppress_history: AtomicBool::new(false),
            transport_probe: std::env::var_os("HIFISHIFTER_ARA_TRANSPORT_PROBE").is_some(),
            transport_probe_next: Mutex::new(Instant::now()),
        });
        let weak = Arc::downgrade(&session);
        let worker = std::thread::Builder::new()
            .name("hfs-embedded-editor".into())
            .spawn(move || Self::run(weak, receiver))
            .map_err(|e| format!("create editor command worker: {e}"))?;
        *session.worker.lock().unwrap() = Some(worker);
        super::events::register(&session);
        Ok(session)
    }
    /// 有界32任务FIFO；关view取消回信，真实组件入口撤销则拒绝排队请求执行。
    pub fn enqueue(&self, request: UiRequest) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Err("FX processor closed".into());
        }
        let document = self.document.upgrade().ok_or("document closed")?;
        let lease = request
            .link
            .as_ref()
            .map(|link| link.authorize(&document))
            .transpose()?;
        if request.sink.view_id.is_empty()
            || (request.link.is_some() && request.sink.closed.load(Ordering::Acquire))
        {
            return Err("editor view closed or unknown".into());
        }
        let mut views = self.views.lock().unwrap();
        if self.closed.load(Ordering::Acquire) {
            return Err("editor session closed".into());
        }
        if request.link.is_some()
            && views
                .get(&request.sink.view_id)
                .is_some_and(|view| !Arc::ptr_eq(&view.sink.closed, &request.sink.closed))
        {
            return Err("editor view identity mismatch".into());
        }
        let sink = request.sink.clone();
        let route = request.link.clone().zip(lease);
        // 播放观察不排到神经合成后面。这里只读原子缓存，不执行actor命令/host API/IO。
        // 原始view身份及同文档route授权仍与普通队列入口一致。
        if request.command == "get_playback_state" {
            if !document.is_alive() {
                return Err("document closed".into());
            }
            let value = self.playback_state();
            if let Some(link) = &request.link {
                if Some(link.authorize(&document)?) != lease {
                    return Err("editor route changed".into());
                }
            }
            views.insert(
                sink.view_id.clone(),
                RegisteredView {
                    sink: sink.clone(),
                    route,
                },
            );
            sink.response(
                json!({"version":1,"viewId":sink.view_id,"id":request.id,"ok":true,"value":value}),
            );
            return Ok(());
        }
        let mutates = super::commands::mutates_audio(&request.command);
        // 与worker的检查共用views锁；只有成功入队才登记view和推进合并票据。
        self.queue
            .try_send(Job::Request(request, lease))
            .map_err(|e| format!("editor queue unavailable: {e}"))?;
        views.insert(sink.view_id.clone(), RegisteredView { sink, route });
        if mutates {
            self.submitted.fetch_add(1, Ordering::AcqRel);
            self.cancel_render();
        }
        Ok(())
    }
    /// getState的非实时屏障：先排完已收到的尾块，不需要等音频快照才持久化曲线。
    pub fn flush(&self) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let (sender, receiver) = mpsc::channel();
        self.queue
            .send(Job::Barrier(sender))
            .map_err(|e| e.to_string())?;
        receiver
            .recv_timeout(Duration::from_secs(30))
            .map_err(|e| format!("editor flush: {e}"))
    }
    /// 在组件stop返回前撤销其回信/订阅；排队任务仍会独立核对原始route代次。
    pub(crate) fn revoke_closed_views(&self) {
        let document = self.document.upgrade();
        self.views.lock().unwrap().retain(|_, view| {
            let valid = view.route.as_ref().is_none_or(|(link, lease)| {
                document.as_ref().is_some_and(|document| {
                    link.authorize(document)
                        .is_ok_and(|current| current == *lease)
                })
            });
            if !valid {
                view.sink.closed.store(true, Ordering::Release);
            }
            valid
        });
    }
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        {
            let mut views = self.views.lock().unwrap();
            for view in views.values() {
                view.sink.closed.store(true, Ordering::Release);
            }
            views.clear();
        }
        self.analysis_cancel.store(true, Ordering::Release);
        self.cancel_render();
        self.submitted.fetch_add(1, Ordering::AcqRel);
        let _ = self.queue.send(Job::Close);
        let worker = self.worker.lock().unwrap().take();
        if let Some(worker) = worker {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
        let render = self.render_task.lock().unwrap().take();
        if let Some(render) = render {
            render.cancel.store(true, Ordering::Release);
            if render.worker.thread().id() != std::thread::current().id() {
                let _ = render.worker.join();
            }
        }
        for worker in self.analysis_workers.lock().unwrap().drain(..) {
            let _ = worker.join();
        }
        // 旧view可继续持有已关闭actor的Arc；join之后释放分析投影/历史/波形缓存。
        *self.timeline.lock().unwrap() = Default::default();
        *self.history.lock().unwrap() = Default::default();
        *self.loaded.lock().unwrap() = Default::default();
        self.peaks.lock().unwrap().clear();
    }
    fn run(weak: Weak<Self>, receiver: mpsc::Receiver<Job>) {
        let mut deadline: Option<Instant> = None;
        loop {
            if let Some(session) = weak.upgrade() {
                if session.closed.load(Ordering::Acquire) {
                    break;
                }
                session.refresh_host();
                if let Some(render) = session.render_task.lock().unwrap().as_ref() {
                    if render.ticket != session.submitted.load(Ordering::Acquire)
                        || render.generation != session.generation.load(Ordering::Acquire)
                        || session
                            .document
                            .upgrade()
                            .is_none_or(|doc| doc.host_undo.pending.load(Ordering::Acquire))
                    {
                        render.cancel.store(true, Ordering::Release);
                    }
                }
                if let Some((generation, ticket, cancelled, result)) = session.complete_render() {
                    session.emit("playback_rendering_state",json!({"active":false,"progress":if result.is_ok()&&!cancelled {Some(1.0)} else {None::<f64>},"target":"background"}));
                    let current = !cancelled
                        && ticket == session.submitted.load(Ordering::Acquire)
                        && generation == session.generation.load(Ordering::Acquire);
                    if current {
                        match result {
                            Ok(()) => {
                                session.applied.store(generation, Ordering::Release);
                                session.render_requested.store(false, Ordering::Release);
                                *session.render_error.lock().unwrap() = None;
                            }
                            Err(error)
                                if error == "pitch analysis pending"
                                    || error == "automatic apply superseded" =>
                            {
                                deadline = Some(Instant::now() + Duration::from_millis(150));
                            }
                            Err(error) => {
                                *session.render_error.lock().unwrap() = Some(error);
                                deadline = Some(Instant::now() + Duration::from_secs(1));
                            }
                        }
                    } else {
                        deadline = Some(Instant::now() + Duration::from_millis(150));
                    }
                    session.emit_state();
                }
                session.report_transport_probe();
                if session.refresh_analysis() {
                    deadline = Some(Instant::now() + Duration::from_millis(150));
                }
                if deadline.is_none()
                    && session.render_requested.load(Ordering::Acquire)
                    && session.error.lock().unwrap().is_none()
                {
                    deadline = Some(Instant::now() + Duration::from_millis(150));
                }
                // 到期应用先于下一条只读轮询；持续get_playback_state不能令合成饥饿。
                if deadline.is_some_and(|d| d <= Instant::now()) {
                    let ticket = session.submitted.load(Ordering::Acquire);
                    if session.render_task.lock().unwrap().is_some()
                        || session
                            .document
                            .upgrade()
                            .is_some_and(|doc| doc.host_undo.pending.load(Ordering::Acquire))
                    {
                        deadline = Some(Instant::now() + Duration::from_millis(50));
                    } else if session.processed.load(Ordering::Acquire) != ticket {
                        deadline = Some(Instant::now() + Duration::from_millis(1));
                    } else {
                        deadline = None;
                        if let Err(error) = session.start_render(ticket) {
                            session.emit("playback_rendering_state",json!({"active":false,"progress":None::<f64>,"target":"background"}));
                            if error == "pitch analysis pending" {
                                deadline = Some(Instant::now() + Duration::from_millis(150));
                            } else {
                                *session.render_error.lock().unwrap() = Some(error);
                                deadline = Some(Instant::now() + Duration::from_secs(1));
                            }
                            session.emit_state();
                        }
                    }
                }
            } else {
                break;
            }
            let wait = deadline
                .map(|d| d.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::from_millis(100));
            match receiver.recv_timeout(wait) {
                Ok(Job::Close) => break,
                Ok(Job::Barrier(sender)) => {
                    let _ = sender.send(());
                }
                Ok(Job::Request(request, lease)) => {
                    let Some(session) = weak.upgrade() else {
                        break;
                    };
                    if session.closed.load(Ordering::Acquire) {
                        break;
                    }
                    let before = session.generation.load(Ordering::Acquire);
                    let mutates = super::commands::mutates_audio(&request.command);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if let Some(link) = &request.link {
                            let document = session.document.upgrade().ok_or("document closed")?;
                            if Some(link.authorize(&document)?) != lease {
                                return Err("editor route changed while request queued".into());
                            }
                            let views = session.views.lock().unwrap();
                            // view关闭仅取消回信；已入队尾笔仍持enqueue验证的身份与route代次。
                            // emit可能已移除closed view，但同名新view不能替代这次原始请求。
                            match views.get(&request.sink.view_id) {
                                Some(view)
                                    if !Arc::ptr_eq(&view.sink.closed, &request.sink.closed) =>
                                {
                                    return Err("editor view identity mismatch".into())
                                }
                                None if !request.sink.closed.load(Ordering::Acquire) => {
                                    return Err("editor view closed or unknown".into())
                                }
                                _ => {}
                            }
                        }
                        super::commands::dispatch(&session, &request.command, request.args)
                    }))
                    .unwrap_or_else(|_| Err("editor command panicked".into()));
                    if mutates {
                        session.processed.fetch_add(1, Ordering::AcqRel);
                    }
                    session.schedule_analysis();
                    if session.generation.load(Ordering::Acquire) != before {
                        deadline = Some(Instant::now() + Duration::from_millis(150));
                    }
                    let response = match result {
                        Ok(value) => {
                            json!({"version":1,"viewId":request.sink.view_id,"id":request.id,"ok":true,"value":value})
                        }
                        Err(error) => {
                            json!({"version":1,"viewId":request.sink.view_id,"id":request.id,"ok":false,"error":error})
                        }
                    };
                    request.sink.response(response);
                    session.emit_state();
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // 回到顶部统一调度，不要求命令队列出现空闲间隙。
                }
            }
        }
    }
    /// 新写入只设取消标记，不join或等待DSP，保证FIFO参数读写/flush始终可运行。
    fn cancel_render(&self) {
        if let Some(render) = self.render_task.lock().unwrap().as_ref() {
            render.cancel.store(true, Ordering::Release);
        }
    }
    fn complete_render(&self) -> Option<(u64, u64, bool, Result<(), String>)> {
        let mut task = self.render_task.lock().unwrap();
        let current = task.as_ref()?;
        let result = match current.reply.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Err("DSP worker disconnected".into()),
        };
        let current = task.take().unwrap();
        drop(task);
        let cancelled = current.cancel.load(Ordering::Acquire);
        let _ = current.worker.join();
        Some((current.generation, current.ticket, cancelled, result))
    }
    /// 原线缓存组装保留在actor；只把已确认的版本票据交给DSP线程，不并发修改timeline。
    fn prepare_apply(&self) -> Result<(Arc<DocumentSession>, u64, u64, String), String> {
        if let Some(error) = self.error.lock().unwrap().clone() {
            return Err(error);
        }
        self.complete_cached_analysis();
        if let Some(error) = self.error.lock().unwrap().clone() {
            return Err(error);
        }
        // 全零占位原线并非已完成的清音分析；收敛前保留旧快照及dirty代次。
        if Self::requires_analysis(&self.timeline.lock().unwrap()) {
            return Err("pitch analysis pending".into());
        }
        let (edit, model, projection) = {
            let loaded = self.loaded.lock().unwrap();
            (loaded.edit, loaded.model, loaded.projection.clone())
        };
        let document = self.document.upgrade().ok_or("document closed")?;
        Ok((document, edit, model, projection))
    }
    fn start_render(self: &Arc<Self>, ticket: u64) -> Result<(), String> {
        let (document, edit, model, projection) = self.prepare_apply()?;
        let generation = self.generation.load(Ordering::Acquire);
        let cancel = Arc::new(AtomicBool::new(false));
        let current = self.clone();
        let job_cancel = cancel.clone();
        let progress_session = self.clone();
        let progress_cancel = cancel.clone();
        let progress_last = AtomicU64::new(0);
        let progress = hifishifter_kernel::mixdown::ProgressCallback::new(move |value| {
            if progress_cancel.load(Ordering::Acquire)
                || progress_session.closed.load(Ordering::Acquire)
                || progress_session.submitted.load(Ordering::Acquire) != ticket
                || progress_session.generation.load(Ordering::Acquire) != generation
            {
                return;
            }
            let value = value.clamp(0.0, 1.0);
            let previous = f64::from_bits(progress_last.load(Ordering::Acquire));
            if value + 1e-6 < previous {
                return;
            }
            progress_last.store(value.to_bits(), Ordering::Release);
            progress_session.emit(
                "playback_rendering_state",
                json!({"active":true,"progress":value,"target":"background"}),
            );
        });
        self.emit(
            "playback_rendering_state",
            json!({"active":true,"progress":0.0,"target":"background"}),
        );
        let worker = RenderTask::spawn(ticket, generation, cancel, move || {
            document.apply_workspace_edits_with_progress(
                edit,
                model,
                &projection,
                job_cancel.clone(),
                || {
                    !job_cancel.load(Ordering::Acquire)
                        && !current.closed.load(Ordering::Acquire)
                        && current.submitted.load(Ordering::Acquire) == ticket
                        && current.generation.load(Ordering::Acquire) == generation
                        && !document.host_undo.pending.load(Ordering::Acquire)
                },
                Some(progress),
            )
        })?;
        *self.render_task.lock().unwrap() = Some(worker);
        Ok(())
    }
    #[cfg(test)]
    fn apply(
        &self,
        ticket: u64,
        progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
    ) -> Result<(), String> {
        let (document, edit, model, projection) = self.prepare_apply()?;
        document.apply_workspace_edits_with_progress(
            edit,
            model,
            &projection,
            self.analysis_cancel.clone(),
            || {
                !self.closed.load(Ordering::Acquire)
                    && self.submitted.load(Ordering::Acquire) == ticket
            },
            progress,
        )
    }
    /// 音高和气声等处理器效果都依赖完成的原线；纯混音/禁用算法不受此门禁阻塞。
    fn requires_analysis(timeline: &TimelineState) -> bool {
        timeline.params_by_root_track.iter().any(|(root, params)| {
            params.pitch_orig_key.is_none()
                && timeline.clips.iter().any(|clip| {
                    timeline.resolve_root_track_id(&clip.track_id).as_deref() == Some(root.as_str())
                        && hifishifter_kernel::pitch_editing::does_clip_need_processor_render(
                            timeline,
                            clip,
                            clip.start_sec,
                        )
                })
                && timeline.tracks.iter().any(|t| {
                    t.id == *root && !matches!(t.pitch_analysis_algo, PitchAnalysisAlgo::None)
                })
        })
    }
    /// 投影撤销项目缓存键时，源分析可能早已全量命中；由actor组装，不等GUI读取/新通知。
    /// 只消费clip分析缓存，不做神经推理；尚未齐全仍pending，用户目标曲线保持。
    fn complete_cached_analysis(&self) {
        let roots = {
            let mut timeline = self.timeline.lock().unwrap();
            // 首次载入尚未get_param_frames时也必须有根状态，否则完成通知找不到接收原线的条目。
            let active_roots = timeline
                .clips
                .iter()
                .filter_map(|clip| timeline.resolve_root_track_id(&clip.track_id))
                .collect::<std::collections::BTreeSet<_>>();
            for root in active_roots {
                timeline.ensure_params_for_root(&root);
            }
            timeline
                .params_by_root_track
                .iter()
                .filter(|(_, params)| {
                    params.pitch_orig_key.is_none() || params.dyn_orig_key.is_none()
                })
                .map(|(root, _)| root.clone())
                .collect::<Vec<_>>()
        };
        if roots.is_empty() {
            return;
        }
        let analysis = Mutex::new(self.analysis_timeline());
        {
            let mut timeline = analysis.lock().unwrap();
            if let Some(selected) = timeline.selected_clip_id.clone() {
                if let Some(index) = timeline.clips.iter().position(|clip| clip.id == selected) {
                    let clip = timeline.clips.remove(index);
                    timeline.clips.push(clip);
                }
            }
        }
        for root in &roots {
            hifishifter_kernel::pitch_analysis::maybe_schedule_pitch_orig(&analysis, root);
            hifishifter_kernel::pitch_analysis::maybe_schedule_dyn_orig(&analysis, root);
        }
        let params = analysis.into_inner().unwrap().params_by_root_track;
        let mut changed = false;
        {
            let mut timeline = self.timeline.lock().unwrap();
            for root in roots {
                let Some(next) = params.get(&root) else {
                    continue;
                };
                let Some(current) = timeline.params_by_root_track.get_mut(&root) else {
                    continue;
                };
                if current.pitch_orig_key.is_none() && next.pitch_orig_key.is_some() {
                    current.pitch_orig = next.pitch_orig.clone();
                    if !current.pitch_edit_user_modified {
                        current.pitch_edit = next.pitch_edit.clone();
                    }
                    current.pitch_orig_key = next.pitch_orig_key.clone();
                    current.has_pitch_adjustment_active = next.has_pitch_adjustment_active;
                    changed = true;
                }
                if current.dyn_orig_key.is_none() && next.dyn_orig_key.is_some() {
                    current.dyn_orig = next.dyn_orig.clone();
                    current.dyn_orig_key = next.dyn_orig_key.clone();
                    changed = true;
                }
            }
        }
        if changed {
            let updated = self.timeline.lock().unwrap().clone();
            self.publish_timeline(updated);
            self.render_requested.store(true, Ordering::Release);
        }
    }
    pub(crate) fn ensure_loaded(&self, force: bool) -> Result<(), String> {
        let document = self.document.upgrade().ok_or("document closed")?;
        // GUI清单有独立代次：粘贴/删除可以先于ARA模型回调，不能被模型缓存早退吞掉。
        if self.loaded.lock().unwrap().initialized
            && document.renderer_owners().is_empty()
            && !self.timeline.lock().unwrap().clips.is_empty()
        {
            return Err("Conflict: no active renderer; local curves preserved".into());
        }
        if self.loaded.lock().unwrap().initialized {
            self.refresh_ui_geometry(&document);
        }
        let versions = document.editor_versions()?;
        {
            let mut loaded = self.loaded.lock().unwrap();
            let audio_current =
                loaded.audio == document.editor_audio_revision.load(Ordering::Acquire);
            if loaded.initialized
                && (loaded.edit, loaded.model, loaded.scope) == versions
                && audio_current
                && !force
            {
                return Ok(());
            }
            if loaded.initialized
                && loaded.model == versions.1
                && loaded.scope == versions.2
                && audio_current
                && !force
                && document.workspace_projection()? == loaded.projection
            {
                loaded.edit = versions.0;
                return Ok(());
            }
        }
        // generation 未应用不等于宿主冲突：workspace_snapshot 会把本地曲线
        // 叠加到最新宿主几何上。只有 ARA 还没有重新交付输入时才暂缓重载。
        if !force
            && !document.ready.load(Ordering::Acquire)
            && !document.workspace_timeline()?.clips.is_empty()
        {
            return Err("Conflict: host changed; local curves preserved, host is not ready".into());
        }
        let (snapshot, scope, projection) = document.workspace_snapshot()?;
        let mut timeline = snapshot.timeline;
        // 快照冻结take权威；先重建扁平投影，再判断真实的倒放/组合倍率。
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        // 正向倍率交给原kernel保调处理，不能再把全部拉伸挡在GUI外；倒放仍按用户范围拒绝。
        if timeline.clips.iter().any(|clip| clip.reversed) {
            let error = "Unsupported ARA geometry: reverse is not supported".to_owned();
            *self.error.lock().unwrap() = Some(error.clone());
            return Err(error);
        }
        if timeline.target_param_frames(timeline.frame_period_ms()) > 1_000_000 {
            return Err("ARA editor parameter frame budget exceeded for project span".into());
        }
        let views: Vec<_> = snapshot
            .sources
            .iter()
            .map(|(id, pcm)| PcmView {
                persistent_id: id,
                sample_rate: pcm.sample_rate,
                planes: &pcm.planes,
            })
            .collect();
        // 内容hash决定私有分析路径；model/edit代次和项目位置不应制造新的F0缓存身份。
        let dir = self.pcm_dir.join("sources");
        let (mut timeline, reverse_paths) = materialize_with_byte_limit(
            timeline,
            &views,
            &dir,
            crate::render::budget::global_budget().limit(),
        )?;
        // 旧状态或同URI换音频的原线不能假报就绪；已有完整clip cache由actor重新组装。
        for params in timeline.params_by_root_track.values_mut() {
            params.pitch_orig_key = None;
            params.dyn_orig_key = None;
        }
        self.rewrite_ids(&mut timeline, true);
        self.retain_display_waveforms(&mut timeline, false);
        document.present_host_inventory(&mut timeline, &self.namespace);
        document.present_private_groups(&mut timeline, &self.namespace);
        self.retain_display_waveforms(&mut timeline, true);
        // 音频迟到只能补齐资源，不能把用户已选中的新粘贴片段跳回首个clip。
        {
            let previous = self.timeline.lock().unwrap();
            if previous
                .selected_track_id
                .as_ref()
                .is_some_and(|id| timeline.tracks.iter().any(|track| &track.id == id))
            {
                timeline.selected_track_id = previous.selected_track_id.clone();
            }
            if previous
                .selected_clip_id
                .as_ref()
                .is_some_and(|id| timeline.clips.iter().any(|clip| &clip.id == id))
            {
                timeline.selected_clip_id = previous.selected_clip_id.clone();
            }
        }
        if timeline.selected_track_id.is_none() {
            timeline.selected_track_id = timeline.tracks.first().map(|t| t.id.clone());
        }
        if timeline.selected_clip_id.is_none() {
            timeline.selected_clip_id = timeline.clips.first().map(|c| c.id.clone());
        }
        self.project_duration
            .store(timeline.project_sec.to_bits(), Ordering::Release);
        *self.timeline.lock().unwrap() = timeline;
        {
            let loaded = self.loaded.lock().unwrap();
            if force
                || !loaded.initialized
                || loaded.model != snapshot.model_revision
                || loaded.scope != scope
            {
                *self.history.lock().unwrap() = Default::default();
            }
        }
        *self.loaded.lock().unwrap() = Loaded {
            initialized: true,
            edit: snapshot.revision,
            model: snapshot.model_revision,
            scope,
            audio: snapshot.audio_revision,
            reverse_paths,
            projection,
        };
        *self.error.lock().unwrap() = None;
        *self.render_error.lock().unwrap() = None;
        self.applied
            .store(self.generation.load(Ordering::Acquire), Ordering::Release);
        self.peaks.lock().unwrap().clear();
        // snapshot返回后UI缓存可能再次更新，强制下一轮按短事务版本同步，不能吞掉该变化。
        self.ui_geometry_version.store(0, Ordering::Release);
        self.complete_cached_analysis();
        self.schedule_analysis();
        self.render_requested.store(true, Ordering::Release);
        self.host_version.fetch_add(1, Ordering::AcqRel);
        self.emit(
            "plugin_host_changed",
            json!({"version":self.host_version.load(Ordering::Acquire)}),
        );
        Ok(())
    }
    /// 清单结构与普通fade独立同步；ARA未ready时也通知新增/删除，不清空曲线、选择或历史。
    pub(super) fn refresh_ui_geometry(&self, document: &DocumentSession) {
        let version = document.ui_geometry_revision.load(Ordering::Acquire);
        if version == self.ui_geometry_version.load(Ordering::Acquire) {
            return;
        }
        // 先冻结可用fade，再合并清单；只确认开始时的代次，期间的新变化留到下一轮。
        let fades = document.ui_fade_projection().ok();
        {
            let mut timeline = self.timeline.lock().unwrap();
            self.retain_display_waveforms(&mut timeline, false);
            document.present_host_inventory(&mut timeline, &self.namespace);
            document.present_private_groups(&mut timeline, &self.namespace);
            self.retain_display_waveforms(&mut timeline, true);
            if let Some((_, host)) = fades {
                for clip in &mut timeline.clips {
                    if let Some(original) = host
                        .clips
                        .iter()
                        .find(|original| format!("{}{}", self.namespace, original.id) == clip.id)
                    {
                        clip.muted = original.muted;
                        clip.gain = original.gain;
                        for take in &mut clip.takes {
                            take.gain = original.gain;
                        }
                        clip.snap_offset_sec = original.snap_offset_sec;
                        clip.fade_in_sec = original.fade_in_sec;
                        clip.fade_out_sec = original.fade_out_sec;
                        clip.auto_fade_in_sec = original.auto_fade_in_sec;
                        clip.auto_fade_out_sec = original.auto_fade_out_sec;
                        clip.fade_in_shape = original.fade_in_shape;
                        clip.fade_out_shape = original.fade_out_shape;
                        clip.fade_in_dir = original.fade_in_dir;
                        clip.fade_out_dir = original.fade_out_dir;
                    }
                }
            }
            self.project_duration
                .store(timeline.project_sec.to_bits(), Ordering::Release);
        }
        self.ui_geometry_version.store(version, Ordering::Release);
        self.notify_timeline();
    }
    /// 稳定模型变化在后台自动读取；pending曲线遇到真实冲突保留，不静默强制重载。
    fn refresh_host(&self) {
        let Some(document) = self.document.upgrade() else {
            return;
        };
        let (initialized, edit, model, scope, audio) = {
            let loaded = self.loaded.lock().unwrap();
            (
                loaded.initialized,
                loaded.edit,
                loaded.model,
                loaded.scope,
                loaded.audio,
            )
        };
        if !initialized {
            return;
        }
        self.refresh_ui_geometry(&document);
        // 原生媒体写入的短Undo窗口先回执GUI结构；音频物化留给写入收尾后的后台加载。
        if !document.host_undo.pending.load(Ordering::Acquire)
            && (document.editor_audio_revision.load(Ordering::Acquire) != audio
                || document
                    .editor_versions()
                    .is_ok_and(|versions| versions != (edit, model, scope)))
        {
            if let Err(error) = self.ensure_loaded(false) {
                if error.starts_with("Conflict") || error.starts_with("Unsupported") {
                    let changed = {
                        let mut current = self.error.lock().unwrap();
                        let changed = current.as_deref() != Some(error.as_str());
                        *current = Some(error);
                        changed
                    };
                    if changed {
                        self.emit_state();
                    }
                }
            }
        }
        // 宿主音乐上下文：BPM 与拍号都来自 VST3 进程上下文，同为**只读**的宿主权威
        // 读数。两者各自取锁（不嵌套），最后合成一次变更通知 —— 一次回调触发两次
        // 刷新会让 GUI 做两遍工作。
        let tempo_changed = match document.clock.tempo() {
            Some(tempo) => {
                let mut timeline = self.timeline.lock().unwrap();
                let changed = (timeline.bpm - tempo).abs() > 1e-6;
                timeline.bpm = tempo;
                changed
            }
            None => false,
        };
        // 【为什么拍号也写进来】此前只有 BPM 被下发，拍号虽然早在 `audio_abi` 的
        // 进程上下文里解析好了，却从来没人读 —— 于是插件里的拍号永远是工程默认值，
        // 而用户改不了它（那是宿主的东西）。读出来至少让界面说的是实话。
        let meter_changed = match document.clock.time_signature() {
            Some((numerator, denominator)) => {
                let mut project = self.project.lock().unwrap();
                // 与 `normalize_tempo_map` 同口径地夹取，不让宿主读数的越界值
                // 进入工程状态。
                let numerator = (numerator as u32).clamp(1, 32);
                let denominator = (denominator as u32).clamp(1, 32);
                let changed = project.beats_per_bar != numerator
                    || project.time_signature_denominator != denominator;
                project.beats_per_bar = numerator;
                project.time_signature_denominator = denominator;
                changed
            }
            None => false,
        };
        if tempo_changed || meter_changed {
            self.host_version.fetch_add(1, Ordering::AcqRel);
            self.emit(
                "plugin_host_changed",
                json!({"version":self.host_version.load(Ordering::Acquire)}),
            );
        }
    }
    /// 波形入口只接受本会话从宿主PCM生成的路径，不能让JS任意读取本机文件。
    pub(super) fn check_source(&self, path: &str) -> Result<(), String> {
        if self.loaded.lock().unwrap().reverse_paths.contains_key(path)
            || self
                .display_waveforms
                .lock()
                .unwrap()
                .values()
                .any(|(_, clip)| clip.source_path.as_deref() == Some(path))
        {
            Ok(())
        } else {
            Err("source is not authorized by this ARA session".into())
        }
    }
    /// mute只撤销播放分配，不撤销已经生成的显示波形；take更换/真实删除不沿旧缓存猜。
    fn retain_display_waveforms(&self, timeline: &mut TimelineState, restore: bool) {
        let Some(document) = self.document.upgrade() else {
            return;
        };
        let tracks = document.ui_tracks.lock().unwrap();
        let mut cache = self.display_waveforms.lock().unwrap();
        let actual = tracks
            .values()
            .flat_map(|track| &track.items)
            .map(|item| {
                (
                    format!("{}ara-item-{}", self.namespace, item.geometry.item_id),
                    item.geometry.take_id.clone(),
                )
            })
            .collect::<HashMap<_, _>>();
        cache.retain(|id, (take, _)| actual.get(id) == Some(take));
        for clip in &mut timeline.clips {
            let Some(take) = actual.get(&clip.id) else {
                continue;
            };
            if !restore && clip.source_path.is_some() {
                cache.insert(clip.id.clone(), (take.clone(), clip.clone()));
            }
            if restore && clip.muted && clip.source_path.is_none() {
                if let Some((_, old)) = cache.get(&clip.id) {
                    clip.source_path = old.source_path.clone();
                    clip.duration_sec = old.duration_sec;
                    clip.duration_frames = old.duration_frames;
                    clip.source_sample_rate = old.source_sample_rate;
                    clip.source_channels = old.source_channels;
                    clip.waveform_preview = old.waveform_preview.clone();
                    for current in &mut clip.takes {
                        current.source_path = old.source_path.clone();
                        current.duration_sec = old.duration_sec;
                        current.duration_frames = old.duration_frames;
                        current.source_sample_rate = old.source_sample_rate;
                        current.source_channels = old.source_channels;
                        current.waveform_preview = old.waveform_preview.clone();
                    }
                }
            }
        }
    }
    /// 手绘pitch在compose关闭时仍需原线；只在分析快照中打开门禁，不改宿主轨道。
    fn analysis_timeline(&self) -> TimelineState {
        let mut timeline = self.timeline.lock().unwrap().clone();
        if let Some(document) = self.document.upgrade() {
            let ids = document.clip_ids.lock().unwrap();
            timeline.clips.retain(|clip| {
                ids.values()
                    .any(|id| format!("{}{id}", self.namespace) == clip.id)
            });
        }
        for track in &mut timeline.tracks {
            if timeline
                .params_by_root_track
                .get(&track.id)
                .is_some_and(|p| p.pitch_edit_user_modified)
            {
                track.compose_enabled = true;
            }
        }
        timeline
    }
    /// 原分析实现可取消/join；每会话持有自己的worker，不推进全局generation。
    fn schedule_analysis(&self) {
        if self.closed.load(Ordering::Acquire) || !self.loaded.lock().unwrap().initialized {
            return;
        }
        let mut workers = self.analysis_workers.lock().unwrap();
        let mut pending = Vec::new();
        for worker in workers.drain(..) {
            if worker.is_finished() {
                let _ = worker.join();
            } else {
                pending.push(worker);
            }
        }
        pending.extend(
            hifishifter_kernel::pitch_clip::schedule_clip_pitch_jobs_scoped(
                &self.analysis_timeline(),
                &self.analysis_sender,
                self.analysis_cancel.clone(),
            ),
        );
        *workers = pending;
    }
    /// 原设备worker负责的ClipPitchReady现在由本实例actor接收，不创建cpal设备。
    fn refresh_analysis(&self) -> bool {
        let updates = self
            .analysis_updates
            .lock()
            .unwrap()
            .try_iter()
            .collect::<Vec<_>>();
        let mut roots = std::collections::BTreeSet::new();
        {
            let mut timeline = self.timeline.lock().unwrap();
            for update in updates {
                if let hifishifter_kernel::engine_command::EngineCommand::ClipPitchReady {
                    clip_id,
                } = update
                {
                    if let Some(clip) = timeline.clips.iter().find(|c| c.id == clip_id) {
                        if let Some(root) = timeline.resolve_root_track_id(&clip.track_id) {
                            roots.insert(root);
                        }
                    }
                }
            }
            for root in &roots {
                if let Some(params) = timeline.params_by_root_track.get_mut(root) {
                    params.pitch_orig_key = None;
                    params.dyn_orig_key = None;
                }
            }
        }
        if roots.is_empty() {
            return false;
        }
        self.complete_cached_analysis();
        // 保存冷恢复没有本地generation，原线完成仍须重应用，不能永久沿旧源F0快照。
        true
    }
    /// GUI读取真实宿主时钟，不驱动独立设备；未收到process上下文时保留本地查看游标。
    pub(super) fn transport(&self) -> (f64, bool) {
        self.document
            .upgrade()
            .filter(|document| document.is_alive())
            .map(|document| document.clock.read())
            .unwrap_or_else(|| (self.timeline.lock().unwrap().playhead_sec, false))
    }
    /// 新淡化轴只装饰发给原GUI的JSON，不进入算法状态、撤销历史或音频缓存键。
    pub(super) fn decorate_host_fades(&self, payload: &mut Value) {
        if let Some(document) = self.document.upgrade() {
            document.decorate_host_fades(payload, &self.namespace);
        }
    }
    /// 宿主音频读数：与 `decorate_host_fades` 同一路径，只多一个语言无关的分类字段。
    pub(super) fn decorate_host_audio(&self, payload: &mut Value) {
        if let Some(document) = self.document.upgrade() {
            document.decorate_host_audio(payload);
        }
    }
    /// 只读宿主原子时钟；重合成阻塞actor时UI仍取得新鲜播放态，不借编辑timeline锁。
    pub(super) fn playback_state(&self) -> Value {
        let (position, playing) = self
            .document
            .upgrade()
            .filter(|document| document.is_alive())
            .map(|document| document.clock.read())
            .unwrap_or((0., false));
        json!({"ok":true,"is_playing":playing,"waiting_for_render":false,
            "target":if playing {Some("synthesized")} else {None},"base_sec":0.,
            "position_sec":position,"duration_sec":f64::from_bits(self.project_duration.load(Ordering::Acquire)),
            "host_authoritative":true})
    }
    /// 一次性时钟探针最多每秒写一条；只在命令actor线程调用，不在音频callback写文件。
    pub(super) fn report_transport_probe(&self) {
        if !self.transport_probe {
            return;
        }
        let now = Instant::now();
        let mut next = self.transport_probe_next.lock().unwrap();
        if now < *next {
            return;
        }
        *next = now + Duration::from_secs(1);
        drop(next);
        if let Some(clock) = self
            .document
            .upgrade()
            .map(|document| document.clock.clone())
        {
            log::info!("[ara] transport diagnostic {}", clock.diagnostics());
        }
    }
    fn rewrite_ids(&self, timeline: &mut TimelineState, to_ui: bool) {
        let id = |value: &str| {
            if to_ui {
                format!("{}{value}", self.namespace)
            } else {
                value
                    .strip_prefix(&self.namespace)
                    .unwrap_or(value)
                    .to_owned()
            }
        };
        for track in &mut timeline.tracks {
            track.id = id(&track.id);
            track.parent_id = track.parent_id.as_deref().map(id);
        }
        for clip in &mut timeline.clips {
            clip.id = id(&clip.id);
            clip.track_id = id(&clip.track_id);
        }
        timeline.selected_track_id = timeline.selected_track_id.as_deref().map(id);
        timeline.selected_clip_id = timeline.selected_clip_id.as_deref().map(id);
        timeline.params_by_root_track = std::mem::take(&mut timeline.params_by_root_track)
            .into_iter()
            .map(|(key, value)| (id(&key), value))
            .collect();
    }
    /// 只将已授权私有分析路径还原到ARA身份，未知路径明确失败。
    fn native_timeline(&self, mut timeline: TimelineState) -> Result<TimelineState, String> {
        self.rewrite_ids(&mut timeline, false);
        if let Some(document) = self.document.upgrade() {
            if !document.ui_tracks.lock().unwrap().is_empty() {
                // GUI占位、名字/排序及可见总时长不进入音频权威；只取参数/轨道可编辑值。
                let mut host = document.workspace_timeline()?;
                if !document.edits.lock().unwrap().groups.is_empty() {
                    let ids = host
                        .tracks
                        .iter()
                        .map(|track| track.id.clone())
                        .collect::<std::collections::BTreeSet<_>>();
                    let (mut flat, grouped) = document
                        .private_parameter_views(timeline.selected_clip_id.clone(), None)?;
                    super::private_groups::expand_parameter_changes(
                        &timeline, &grouped, &mut flat,
                    )?;
                    flat.tracks.retain(|track| ids.contains(&track.id));
                    flat.params_by_root_track.retain(|id, _| ids.contains(id));
                    flat.selected_track_id =
                        timeline.selected_track_id.filter(|id| ids.contains(id));
                    return Ok(flat);
                }
                for track in &mut host.tracks {
                    if let Some(client) = timeline.tracks.iter().find(|t| t.id == track.id) {
                        track.volume = client.volume;
                        track.muted = client.muted;
                        track.solo = client.solo;
                        track.compose_enabled = client.compose_enabled;
                        track.pitch_analysis_algo = client.pitch_analysis_algo.clone();
                    }
                }
                host.params_by_root_track = timeline
                    .params_by_root_track
                    .into_iter()
                    .filter(|(id, _)| host.tracks.iter().any(|t| &t.id == id))
                    .collect();
                host.selected_track_id = timeline
                    .selected_track_id
                    .filter(|id| host.tracks.iter().any(|t| &t.id == id));
                host.selected_clip_id = timeline
                    .selected_clip_id
                    .filter(|id| host.clips.iter().any(|clip| &clip.id == id));
                return Ok(host);
            }
        }
        // 宿主显示占位没有ARA音频授权，只提交真实活动图，不能把占位送入分析/合成。
        if let Some(document) = self.document.upgrade() {
            let ids = document.clip_ids.lock().unwrap();
            timeline
                .clips
                .retain(|clip| ids.values().any(|id| id == &clip.id));
            drop(ids);
            let host = document.timeline.lock().unwrap();
            if let Some(host) = host.as_ref() {
                timeline
                    .tracks
                    .retain(|t| host.tracks.iter().any(|h| h.id == t.id));
                timeline
                    .params_by_root_track
                    .retain(|id, _| host.tracks.iter().any(|t| &t.id == id));
            }
        }
        let loaded = self.loaded.lock().unwrap();
        for clip in &mut timeline.clips {
            // 原GUI普通宿主fade仅装饰，参数提交还原到ARA内核域，避免二次淡化。
            clip.fade_in_sec = 0.;
            clip.fade_out_sec = 0.;
            clip.auto_fade_in_sec = 0.;
            clip.auto_fade_out_sec = 0.;
            clip.fade_in_shape = 0.;
            clip.fade_out_shape = 0.;
            clip.fade_in_dir = 0.;
            clip.fade_out_dir = 0.;
            // 宿主item音量只在GUI显示，提交参数前还原原始ARA源域，不能重复烘焙。
            clip.gain = 1.;
            for take in &mut clip.takes {
                take.gain = 1.;
            }
            for path in std::iter::once(&mut clip.source_path)
                .chain(clip.takes.iter_mut().map(|t| &mut t.source_path))
            {
                let local = path.as_ref().ok_or("clip analysis source missing")?;
                *path = Some(
                    loaded
                        .reverse_paths
                        .get(local)
                        .ok_or("unknown analysis path")?
                        .clone(),
                );
            }
            clip.normalize_takes();
        }
        Ok(timeline)
    }
    /// 选region后沿当前真实doc取只读投影，防止重叠区仍显示前一素材的曲线。
    pub(super) fn select_source_projection(&self) -> Result<(), String> {
        if self
            .error
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|error| error.starts_with("Conflict"))
        {
            return Ok(());
        }
        let document = self.document.upgrade().ok_or("document closed")?;
        let current = self.native_timeline(self.timeline.lock().unwrap().clone())?;
        let roots = document.selected_source_parameters(&current)?;
        let mut timeline = self.timeline.lock().unwrap();
        timeline.params_by_root_track.extend(
            roots
                .into_iter()
                .map(|(root, params)| (format!("{}{root}", self.namespace), params)),
        );
        Ok(())
    }
    pub(super) fn emit(&self, event: &str, payload: Value) {
        let document = self.document.upgrade();
        let views = {
            let mut views = self.views.lock().unwrap();
            views.retain(|_, view| {
                !view.sink.closed.load(Ordering::Acquire)
                    && view.route.as_ref().is_none_or(|(link, lease)| {
                        document.as_ref().is_some_and(|document| {
                            link.authorize(document)
                                .is_ok_and(|current| current == *lease)
                        })
                    })
            });
            views
                .values()
                .map(|view| view.sink.clone())
                .collect::<Vec<_>>()
        };
        for sink in views {
            sink.event(json!({"version":1,"viewId":sink.view_id,"event":event,"payload":payload}));
        }
    }
    /// 复用原GUI已订阅的刷新事件；共享选轨/曲线变化让所有视图重取同一个权威payload。
    pub(super) fn notify_timeline(&self) {
        self.host_version.fetch_add(1, Ordering::AcqRel);
        self.emit(
            "plugin_host_changed",
            json!({"version":self.host_version.load(Ordering::Acquire)}),
        );
    }
    pub(super) fn state(&self) -> Value {
        let generation = self.generation.load(Ordering::Acquire);
        let applied = self.applied.load(Ordering::Acquire);
        let states = self
            .document
            .upgrade()
            .map(|document| {
                document
                    .renderer_owners()
                    .into_iter()
                    .map(|owner| owner.preparation_state())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let host_pending = states.iter().any(|state| state.0);
        let host_error = states.into_iter().find_map(|state| state.1);
        let analysis_pending = Self::requires_analysis(&self.timeline.lock().unwrap());
        let rendering = self.render_task.lock().unwrap().is_some();
        let inventory_pending = self
            .timeline
            .lock()
            .unwrap()
            .clips
            .iter()
            .any(|clip| !clip.muted && clip.source_path.is_none());
        json!({"generation":generation,"applied_generation":applied,"pending":generation!=applied||host_pending||inventory_pending||analysis_pending||rendering||self.render_requested.load(Ordering::Acquire),
            "host_version":self.host_version.load(Ordering::Acquire),
            "error":self.error.lock().unwrap().clone().or_else(||self.render_error.lock().unwrap().clone()).or(host_error),"connected":!self.closed.load(Ordering::Acquire),
            "ready":self.loaded.lock().unwrap().initialized})
    }
    fn emit_state(&self) {
        self.emit("plugin_apply_state", self.state());
    }
}
impl ParamHost for EditorSession {
    fn timeline(&self) -> &Mutex<TimelineState> {
        &self.timeline
    }
    fn checkpoint_timeline(&self, timeline: &TimelineState, operation: HistoryOp) {
        if self.suppress_history.load(Ordering::Acquire) {
            return;
        }
        history::checkpoint(
            &mut self.history.lock().unwrap(),
            timeline,
            operation.key().into(),
            || None,
        );
    }
    fn mark_dirty(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let views = self.views.lock().unwrap();
        for view in views.values() {
            if let Some((link, _)) = &view.route {
                link.mark_dirty();
            }
        }
    }
    fn publish_timeline(&self, timeline: TimelineState) {
        let result = (|| {
            let document = self.document.upgrade().ok_or("document closed")?;
            let groups = document.edits.lock().unwrap().groups.clone();
            let groups =
                if groups.is_empty() {
                    None
                } else {
                    Some(groups.with_processing_from(
                        &timeline,
                        &document.group_aliases(&self.namespace),
                    )?)
                };
            let timeline = self.native_timeline(timeline)?;
            let (edit, model, projection) = {
                let loaded = self.loaded.lock().unwrap();
                (loaded.edit, loaded.model, loaded.projection.clone())
            };
            let versions = document.accept_workspace_edits_with_groups(
                edit,
                model,
                &timeline,
                &projection,
                groups,
            )?;
            let mut loaded = self.loaded.lock().unwrap();
            loaded.edit = versions.0;
            loaded.model = versions.1;
            loaded.projection = versions.2;
            *self.error.lock().unwrap() = None;
            Ok::<_, String>(())
        })();
        if let Err(error) = result {
            *self.error.lock().unwrap() = Some(error);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ara::model::ModelHandle;
    use crate::render::extension::ExtensionOwner;
    use crate::render::ownership::region_owners;
    use crate::render::source::SourcePcm;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::ExtensionRoles;
    /// 参数专用Undo/Redo确实走actor历史，而不是宿主几何；同一分组中的多块曲线只撤一次。
    #[test]
    fn parameter_history_commands_restore_grouped_curve_and_redo_without_geometry_changes() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(64);
        let sink = UiSink {
            view_id: "parameter-history".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let before = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let track = before["tracks"][0]["id"].clone();
        call(
            &editor,
            &sink,
            &rx,
            2,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        call(
            &editor,
            &sink,
            &rx,
            3,
            "begin_undo_group",
            json!({"label":"curve"}),
        );
        call(
            &editor,
            &sink,
            &rx,
            4,
            "set_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"values":[35.,35.],"checkpoint":false}),
        );
        call(
            &editor,
            &sink,
            &rx,
            5,
            "set_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":2,"values":[45.,45.],"checkpoint":false}),
        );
        call(&editor, &sink, &rx, 6, "end_undo_group", json!({}));
        let undo = call(&editor, &sink, &rx, 7, "undo_parameter_edit", json!({}));
        assert_eq!(
            undo["clips"][0]["start_sec"],
            before["clips"][0]["start_sec"]
        );
        let cleared = call(
            &editor,
            &sink,
            &rx,
            8,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        assert!(cleared["edit"]
            .as_array()
            .unwrap()
            .iter()
            .all(|value| value.as_f64() == Some(0.)));
        call(&editor, &sink, &rx, 9, "redo_parameter_edit", json!({}));
        let restored = call(
            &editor,
            &sink,
            &rx,
            10,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        assert_eq!(restored["edit"], json!([35., 35., 45., 45.]));
        document.close();
    }

    struct ReleaseDsp(Option<mpsc::Sender<()>>);
    impl Drop for ReleaseDsp {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }
    /// 实际actor在DSP线程受控阻塞时仍处理读参、写参和getState屏障；不是神经模型速度测试。
    #[test]
    fn async_dsp_does_not_block_parameter_commands_or_state_barriers() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let track = editor.timeline.lock().unwrap().tracks[0].id.clone();
        let (release, wait) = mpsc::channel();
        let mut release = ReleaseDsp(Some(release));
        let (started, ready) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = RenderTask::spawn(
            editor.submitted.load(Ordering::Acquire),
            editor.generation.load(Ordering::Acquire),
            cancel.clone(),
            move || {
                started.send(()).unwrap();
                wait.recv_timeout(Duration::from_secs(3))
                    .map_err(|error| error.to_string())?;
                Ok(())
            },
        )
        .unwrap();
        ready.recv_timeout(Duration::from_secs(1)).unwrap();
        {
            let mut active = editor.render_task.lock().unwrap();
            *active = Some(job);
            document.host_undo.pending.store(false, Ordering::Release);
        }
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(64);
        let sink = UiSink {
            view_id: "async-dsp-diagnostic".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let started = Instant::now();
        let frames = call(
            &editor,
            &sink,
            &rx,
            1,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        assert_eq!(frames["ok"], true);
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":track,"volume":0.5}),
        );
        editor.flush().unwrap();
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "编辑命令不应等待受控DSP任务: {:?}",
            started.elapsed()
        );
        assert!(
            cancel.load(Ordering::Acquire),
            "成功入队的新编辑必须取消旧DSP任务"
        );
        release.0.take().unwrap().send(()).unwrap();
        document.close();
        assert!(
            editor.render_task.lock().unwrap().is_none(),
            "关闭完成前必须回收DSP任务"
        );
    }

    /// 完成结果与计算线程独立，取消位始终保留，不能把被新编辑取代的成功计算标成已应用。
    #[test]
    fn async_dsp_completion_retains_ticket_generation_and_cancellation() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let job = RenderTask::spawn(41, 17, cancel, || Ok(())).unwrap();
        {
            let mut active = editor.render_task.lock().unwrap();
            assert!(active.is_none());
            *active = Some(job);
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some((generation, ticket, cancelled, result)) = editor.complete_render() {
                assert_eq!((generation, ticket), (17, 41));
                assert!(cancelled);
                assert!(result.is_ok());
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        document.close();
    }

    /// 宿主音量只影响GUI；参数提交不能把item增益再交给内核烘焙。
    #[test]
    fn host_clip_gain_display_is_removed_before_parameter_commit() {
        let (_model, owner, _id) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let mut display = editor.timeline.lock().unwrap().clone();
        display.clips[0].gain = 0.5;
        display.clips[0].takes[0].gain = 0.5;
        let native = editor.native_timeline(display).unwrap();
        assert_eq!(native.clips[0].gain, 1.);
        assert_eq!(native.clips[0].takes[0].gain, 1.);
        editor.close();
    }

    /// 只替代真实DAW的建图/授权边界；使用真正owner、native扩展、actor与state编码。
    pub(crate) fn fixture() -> (ModelHandle, Arc<ExtensionOwner>, Box<u8>) {
        let model = ModelHandle::new();
        let document = model.session();
        let mut timeline:TimelineState=serde_json::from_value(json!({
            "tracks":[{"id":"track","name":"host","order":0,"pitch_analysis_algo":"none"}],
            "clips":[{"id":"clip","track_id":"track","name":"source","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","source_path":"ara://source","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":1
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock().unwrap() = Some(timeline);
        document.track_bindings.lock().unwrap().insert(
            "track".into(),
            vec![("modification".into(), "ara://source".into())],
        );
        document.edit_sources.lock().unwrap().insert(
            "ara://source".into(),
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
                version: 0,
                _reservation: None,
            }),
        );
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        document.clip_ids.lock().unwrap().insert(key, "clip".into());
        document.regions.lock().unwrap().insert(
            key,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "ara://source".into(),
                audio_modification_persistent_id: "modification".into(),
                duration_in_modification_time: 4.0 / 44100.0,
                duration_in_playback_time: 4.0 / 44100.0,
                ..Default::default()
            },
        );
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: identity和扩展owner在所有测试调用期间存活。
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        document.ready.store(true, Ordering::Release);
        (model, owner, identity)
    }
    fn call(
        editor: &EditorSession,
        sink: &UiSink,
        receiver: &mpsc::Receiver<Value>,
        id: u64,
        command: &str,
        args: Value,
    ) -> Value {
        editor
            .enqueue(UiRequest {
                id,
                command: command.into(),
                args,
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let result = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(result["id"], id);
        assert_eq!(result["ok"], true, "{result}");
        result["value"].clone()
    }
    /// 两真实组件使用同源的不同区域，测试原actor而不是拼接UI快照。
    pub(crate) fn workspace_fixture() -> (ModelHandle, Vec<Arc<ExtensionOwner>>, Vec<Box<u8>>) {
        let (model, a, first) = fixture();
        let document = model.session();
        {
            let mut host = document.timeline.lock().unwrap();
            let host = host.as_mut().unwrap();
            let mut track = host.tracks[0].clone();
            track.id = "b".into();
            track.order = 1;
            host.tracks.push(track);
            let mut clip = host.clips[0].clone();
            clip.id = "cb".into();
            clip.track_id = "b".into();
            host.clips.push(clip);
        }
        document.track_bindings.lock().unwrap().insert(
            "b".into(),
            vec![("modification-b".into(), "ara://source".into())],
        );
        let second = Box::new(0_u8);
        let key = (&*second as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 1)
            .unwrap();
        document.clip_ids.lock().unwrap().insert(key, "cb".into());
        let mut region = document
            .regions
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .clone();
        region.audio_modification_persistent_id = "modification-b".into();
        document.regions.lock().unwrap().insert(key, region);
        let b = Arc::new(ExtensionOwner::default());
        let raw = b
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: 模型、两owner和真实region身份由返回值保留。
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        (model, vec![a, b], vec![first, second])
    }
    /// 两真实组件统一根编辑后仍按物理区域保存参数；GUI分组/拆组不改宿主轨道图。
    #[test]
    fn goal_feedback_private_group_actor_edit_and_state_roundtrip() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let api = Arc::new(host.client());
        let base = api.ui_track(&|| true).unwrap();
        for (id, guid, order) in [
            ("track", "{11111111-1111-1111-1111-111111111111}", 0),
            ("b", "{22222222-2222-2222-2222-222222222222}", 1),
        ] {
            let mut track = base.clone();
            track.id = id.into();
            track.guid = guid.into();
            track.order = order;
            track.items.clear();
            document
                .ui_tracks
                .lock()
                .unwrap()
                .insert(guid.into(), track);
        }
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "private-groups".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let initial = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let a = initial["tracks"][0]["id"].as_str().unwrap().to_owned();
        let b = initial["tracks"][1]["id"].as_str().unwrap().to_owned();
        for (id, track, value) in [(2, &a, 10.), (3, &b, 20.)] {
            call(
                &editor,
                &sink,
                &rx,
                id,
                "set_param_frames",
                json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"values":[value],"checkpoint":true}),
            );
        }
        let grouped = call(
            &editor,
            &sink,
            &rx,
            4,
            "move_track",
            json!({"trackId":b,"targetIndex":0,"parentTrackId":a}),
        );
        assert_eq!(grouped["tracks"][1]["parent_id"], a);
        assert!(
            document
                .timeline
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .tracks
                .iter()
                .all(|track| track.parent_id.is_none()),
            "REAPER图必须保持原样"
        );
        assert_eq!(
            document.edits.lock().unwrap().params["b"].extra_curves["hifigan_tension"][0],
            20.,
            "仅分组不覆盖子轨曲线"
        );
        call(
            &editor,
            &sink,
            &rx,
            5,
            "set_param_frames",
            json!({"trackId":a,"param":"hifigan_tension","startFrame":0,"values":[65.],"checkpoint":true}),
        );
        for id in ["track", "b"] {
            assert_eq!(
                document.edits.lock().unwrap().params[id].extra_curves["hifigan_tension"][0],
                65.
            );
        }
        let encoded = owners[1].encode_state().unwrap();
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&encoded).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded).unwrap()["version"],
            5
        );
        assert_eq!(
            restored.groups,
            document.edits.lock().unwrap().groups,
            "组件保存必须保留共享GUID分组"
        );
        let parent_state = owners[0].encode_state().unwrap();
        let (cold_model, cold_owners, _cold_ids) = workspace_fixture();
        let cold_document = cold_model.session();
        cold_document
            .host_undo
            .pending
            .store(true, Ordering::Release);
        *cold_document.ui_tracks.lock().unwrap() = document.ui_tracks.lock().unwrap().clone();
        cold_owners[0].restore_state(&parent_state).unwrap();
        cold_owners[1].restore_state(&encoded).unwrap();
        let cold_editor = cold_owners[0].editor_session().unwrap();
        cold_editor.ensure_loaded(false).unwrap();
        let cold = cold_editor.timeline.lock().unwrap().clone();
        let cold_b = cold
            .tracks
            .iter()
            .find(|track| track.id.ends_with("-b"))
            .unwrap();
        assert!(
            cold_b
                .parent_id
                .as_ref()
                .is_some_and(|id| id.ends_with("-track")),
            "冷恢复后GUID分组必须重投影到新GUI namespace"
        );
        assert_eq!(
            cold_document.edits.lock().unwrap().params["b"].extra_curves["hifigan_tension"][0],
            65.
        );
        cold_document.close();
        let ungrouped = call(
            &editor,
            &sink,
            &rx,
            6,
            "move_track",
            json!({"trackId":b,"targetIndex":1,"parentTrackId":null}),
        );
        assert!(ungrouped["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|track| track["parent_id"].is_null()));
        document.close();
        assert!(
            !host
                .calls()
                .iter()
                .any(|call| call.starts_with("set-") || call.starts_with("undo-")),
            "私有分组不能调用宿主setter"
        );
    }
    /// 与已有WORLD oracle相同的谐波源，双轨同源但持久modification身份不同。
    fn task34_world_fixture() -> (ModelHandle, Vec<Arc<ExtensionOwner>>, Vec<Box<u8>>) {
        let (model, owners, ids) = workspace_fixture();
        let document = model.session();
        {
            let mut known = document.timeline.lock().unwrap();
            let timeline = known.as_mut().unwrap();
            timeline.project_sec = 2.;
            for track in &mut timeline.tracks {
                track.compose_enabled = true;
                track.pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            }
            for clip in &mut timeline.clips {
                clip.length_sec = 2.;
                clip.takes[0].source_end_sec = 2.;
                clip.normalize_takes();
            }
        }
        for region in document.regions.lock().unwrap().values_mut() {
            region.duration_in_modification_time = 2.;
            region.duration_in_playback_time = 2.;
        }
        let samples = (0..88200)
            .map(|n| {
                let phase = 2. * std::f64::consts::PI * 220. * n as f64 / 44100.;
                (1..=16)
                    .map(|harmonic| (phase * harmonic as f64).sin() * 0.2 / harmonic as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        let pcm = Arc::new(SourcePcm {
            sample_rate: 44100,
            planes: vec![samples],
            version: 0,
            _reservation: None,
        });
        document
            .edit_sources
            .lock()
            .unwrap()
            .insert("ara://source".into(), pcm.clone());
        document
            .sources
            .lock()
            .unwrap()
            .insert("ara://source".into(), pcm);
        (model, owners, ids)
    }
    /// 生产route尾块在view关闭后仍先入权威；render失败也能保存，共享undo保存后无GUI恢复供音。
    #[test]
    fn task34_tail_and_shared_undo_save_restore_both_world_tracks_without_gui() {
        let (model, owners, _ids) = task34_world_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let leases = owners
            .iter()
            .map(super::super::routing::RouteLease::new)
            .collect::<Vec<_>>();
        let links = leases
            .iter()
            .map(|lease| {
                let link = Arc::new(super::super::routing::EditorLink::default());
                link.bind(std::process::id() as i64, lease.token()).unwrap();
                link
            })
            .collect::<Vec<_>>();
        let mut sinks = Vec::new();
        let mut receivers = Vec::new();
        for index in 0..2 {
            let (reply, rx) = mpsc::channel();
            let (events, _) = mpsc::sync_channel(128);
            sinks.push(UiSink {
                view_id: format!("task34-world-{index}"),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            });
            receivers.push(rx);
        }
        let request = |index: usize, id: u64, command: &str, args: Value| {
            editor
                .enqueue(UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sinks[index].clone(),
                    link: Some(links[index].clone()),
                })
                .unwrap();
            let response = receivers[index]
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let timeline = request(0, 1, "get_timeline_state", json!({}));
        let ta = timeline["tracks"][0]["id"].as_str().unwrap();
        let tb = timeline["tracks"][1]["id"].as_str().unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        for track in [ta, tb] {
            loop {
                let frames = request(
                    0,
                    2,
                    "get_param_frames",
                    json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}),
                );
                if frames["orig"]
                    .as_array()
                    .is_some_and(|orig| orig.iter().filter_map(Value::as_f64).any(|p| p > 30.))
                {
                    break;
                }
                assert!(Instant::now() < deadline, "WORLD原线未完成: {frames}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        request(
            0,
            3,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":false}),
        );
        request(
            0,
            4,
            "set_param_frames",
            json!({"trackId":ta,"param":"pitch","startFrame":0,"values":vec![60.;400],"checkpoint":true}),
        );
        request(
            1,
            5,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":vec![67.;400],"checkpoint":true}),
        );
        // 真实不支持的宿主fade使合成失败；权威的尾笔不能依赖快照发布成功。
        for region in document.regions.lock().unwrap().values_mut() {
            region.has_content_based_fade_at_head = true;
        }
        let held = editor.timeline.lock().unwrap();
        editor
            .enqueue(UiRequest {
                id: 6,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sinks[1].clone(),
                link: Some(links[1].clone()),
            })
            .unwrap();
        editor.enqueue(UiRequest {id:7,command:"set_param_frames".into(),args:json!({"trackId":tb,"param":"pitch","startFrame":399,"values":[69.],"checkpoint":false}),sink:sinks[1].clone(),link:Some(links[1].clone())}).unwrap();
        sinks[1].closed.store(true, Ordering::Release);
        drop(held);
        let saved = [
            owners[0].encode_state().unwrap(),
            owners[1].encode_state().unwrap(),
        ];
        assert!(
            editor
                .apply(editor.submitted.load(Ordering::Acquire), None)
                .is_err(),
            "真实不支持fade必须拒绝合成"
        );
        let b: Value = serde_json::from_slice(&saved[1]).unwrap();
        assert_eq!(b["edits"]["params"]["b"]["pitch_edit"][399], 69.);
        assert_eq!(
            editor.history.lock().unwrap().position,
            2,
            "尾笔不新增undo步"
        );
        assert!(editor
            .enqueue(UiRequest {
                id: 8,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sinks[1].clone(),
                link: Some(links[1].clone())
            })
            .is_err());
        request(0, 9, "undo_timeline", json!({}));
        let undone = [
            owners[0].encode_state().unwrap(),
            owners[1].encode_state().unwrap(),
        ];
        for (states, b_midi) in [(saved, 67.), (undone, 64.)] {
            let (cold, cold_owners, _cold_ids) = task34_world_fixture();
            let cold_document = cold.session();
            // 不创建editor_session，不轮询GUI命令，直接驱动原setState和真实publisher供音。
            cold_document.ready.store(false, Ordering::Release);
            for (owner, bytes) in cold_owners.iter().zip(&states) {
                owner.restore_state(bytes).unwrap();
            }
            cold_document.prepare_renderers();
            let deadline = Instant::now() + Duration::from_secs(15);
            for (index, owner) in cold_owners.iter().enumerate() {
                while owner.preparation_state().0 {
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(10));
                }
                assert!(
                    owner.preparation_state().1.is_none(),
                    "{:?}",
                    owner.preparation_state()
                );
                let mut left = vec![0.; 8820];
                let mut right = vec![0.; 8820];
                let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut bus = crate::audio_abi::AudioBusBuffers {
                    num_channels: 2,
                    silence_flags: 0,
                    channel_buffers: planes.as_mut_ptr(),
                };
                // SAFETY: 两个8820帧平面；oracle读取实际后台准备发布的PCM。
                assert!(unsafe { owner.snapshots[0].copy_block(22050, 44100, &mut bus, 8820) });
                assert!(left.iter().all(|sample| sample.is_finite()));
                assert_eq!(left, right);
                let correlation = |lag: usize| {
                    let mut xy = 0.;
                    let mut xx = 0.;
                    let mut yy = 0.;
                    for n in 0..left.len() - lag {
                        let a = left[n] as f64;
                        let b = left[n + lag] as f64;
                        xy += a * b;
                        xx += a * a;
                        yy += b * b;
                    }
                    xy / (xx * yy).sqrt().max(1e-20)
                };
                let target =
                    440. * 2_f64.powf(((if index == 0 { 60. } else { b_midi }) - 69.) / 12.);
                let expected_lag = 44100. / target;
                let first = (expected_lag * 0.8) as usize;
                let last = (expected_lag * 1.2) as usize;
                let lag = (first..=last)
                    .max_by(|a, b| correlation(*a).total_cmp(&correlation(*b)))
                    .unwrap();
                let hz = 44100. / lag as f64;
                eprintln!("Task34 cold track={index} B_midi={b_midi} frequency={hz:.3}Hz target={target:.3}Hz");
                assert!(
                    (hz - target).abs() < 8.,
                    "cold WORLD pitch mismatch {hz} vs {target}"
                );
                let local: Value = serde_json::from_slice(&owner.encode_state().unwrap()).unwrap();
                assert_eq!(local["edits"]["tracks"].as_array().unwrap().len(), 1);
                assert_eq!(local["edits"]["params"].as_object().unwrap().len(), 1);
                assert_eq!(
                    local["edits"]["params"][if index == 0 { "track" } else { "b" }]["pitch_edit"]
                        [100],
                    if index == 0 { 60. } else { b_midi }
                );
            }
            cold_document.close();
        }
        // 模型变化必须拒绝旧actor投影，不能把曲线保存误报成音频已经同步。
        document.clear_renderers();
        editor
            .enqueue(UiRequest {
                id: 10,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sinks[0].clone(),
                link: Some(links[0].clone()),
            })
            .unwrap();
        let rejected = receivers[0].recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(rejected["ok"], false);
        assert!(rejected["error"]
            .as_str()
            .unwrap()
            .contains("local curves preserved"));
        let local = editor.timeline.lock().unwrap();
        assert_eq!(local.params_by_root_track[ta].pitch_edit[100], 60.);
        assert_eq!(local.params_by_root_track[tb].pitch_edit[100], 64.);
        drop(local);
        assert_eq!(editor.state()["pending"], true);
        assert!(owners[0].encode_state().unwrap_err().contains("not ready"));
        document.close();
    }
    /// 最后一个renderer入口撤销后，actor屏障/空范围应用/文档join都不等待不存在的renderer。
    #[test]
    fn task34_no_renderer_does_not_retry_or_fall_back_to_the_whole_document() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "task34-no-renderer".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":timeline["tracks"][0]["id"],"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),
        );
        for owner in &owners {
            owner.stop_editor();
        }
        let projection = document.workspace_projection().unwrap();
        let workspace = document.workspace_timeline().unwrap();
        assert!(workspace.tracks.is_empty());
        assert!(workspace.clips.is_empty());
        let began = Instant::now();
        editor.flush().unwrap();
        let edit = document.edits.lock().unwrap().revision;
        document
            .apply_workspace_edits(
                edit,
                document.revision.load(Ordering::Acquire),
                &projection,
                Arc::new(AtomicBool::new(false)),
                || true,
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while editor.error.lock().unwrap().is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(editor
            .error
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains("local curves preserved"));
        assert_eq!(
            editor.state()["pending"],
            true,
            "失去全部renderer不能把保存曲线伪报成已渲染"
        );
        assert!(
            !editor.closed.load(Ordering::Acquire),
            "组件入口都撤销也不能误关文档actor"
        );
        document.close();
        assert!(began.elapsed() < Duration::from_secs(3));
        assert!(editor.worker.lock().unwrap().is_none());
    }
    /// 旧view仍持actor Arc时，文档close也必须在join之后释放分析投影/历史和持久权威缓存。
    #[test]
    fn task34_document_close_releases_state_even_when_the_closed_actor_is_retained() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "task34-retained-actor".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":timeline["tracks"][0]["id"],"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),
        );
        let authority = document.edits.clone();
        document.close();
        assert!(editor.worker.lock().unwrap().is_none());
        assert!(editor.analysis_workers.lock().unwrap().is_empty());
        assert!(
            editor.timeline.lock().unwrap().clips.is_empty(),
            "join后旧actor不得继续持有分析投影"
        );
        assert!(editor.history.lock().unwrap().records.is_empty());
        assert!(editor.peaks.lock().unwrap().is_empty());
        assert!(!editor.loaded.lock().unwrap().initialized);
        assert!(document.track_bindings.lock().unwrap().is_empty());
        assert!(authority.lock().unwrap().params.is_empty());
        assert!(authority.lock().unwrap().tracks.is_empty());
    }
    #[test]
    fn task34_component_stop_revokes_its_views_and_output_before_returning() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let leases = owners
            .iter()
            .map(super::super::routing::RouteLease::new)
            .collect::<Vec<_>>();
        let mut sinks = Vec::new();
        let mut receivers = Vec::new();
        let mut links = Vec::new();
        for (index, lease) in leases.iter().enumerate() {
            let link = Arc::new(super::super::routing::EditorLink::default());
            link.bind(std::process::id() as i64, lease.token()).unwrap();
            let (reply, rx) = mpsc::channel();
            let (events, _) = mpsc::sync_channel(8);
            let sink = UiSink {
                view_id: format!("task34-stop-{index}"),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            };
            editor
                .enqueue(UiRequest {
                    id: 1,
                    command: "get_ui_settings".into(),
                    args: json!({}),
                    sink: sink.clone(),
                    link: Some(link.clone()),
                })
                .unwrap();
            assert_eq!(rx.recv_timeout(Duration::from_secs(3)).unwrap()["ok"], true);
            owners[index].snapshots[0]
                .publish(crate::render::snapshot::PlaybackSnapshot {
                    sample_rate: 44100,
                    origin_sample: 0,
                    left: vec![0.25; 4],
                    right: vec![0.5; 4],
                    _reservation: None,
                })
                .unwrap();
            sinks.push(sink);
            receivers.push(rx);
            links.push(link);
        }
        owners[0].stop_editor();
        // 失败也先join文档worker，避免把RED断言变成线程退出问题。
        let revoked = sinks[0].closed.load(Ordering::Acquire);
        let other_live = !sinks[1].closed.load(Ordering::Acquire);
        let output = owners
            .iter()
            .map(|owner| {
                let mut left = [9.; 4];
                let mut right = [9.; 4];
                let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut bus = crate::audio_abi::AudioBusBuffers {
                    num_channels: 2,
                    silence_flags: 0,
                    channel_buffers: planes.as_mut_ptr(),
                };
                // SAFETY: 两个平面各四帧；观察真实发布输出是否已经撤销。
                unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) }
            })
            .collect::<Vec<_>>();
        assert!(editor
            .enqueue(UiRequest {
                id: 2,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sinks[0].clone(),
                link: Some(links[0].clone())
            })
            .is_err());
        editor
            .enqueue(UiRequest {
                id: 3,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sinks[1].clone(),
                link: Some(links[1].clone()),
            })
            .unwrap();
        assert_eq!(
            receivers[1].recv_timeout(Duration::from_secs(3)).unwrap()["ok"],
            true
        );
        document.close();
        assert!(revoked, "组件stop必须同步撤销自己的view");
        assert!(other_live, "另一组件view仍有租约");
        assert_eq!(
            output,
            [false, true],
            "组件stop必须撤销自己的输出，不能清除另一renderer"
        );
    }
    /// 原actor选择重叠region只换其参数投影；不能生成编辑/history或覆盖另一素材。
    #[test]
    fn atlas_selection_commands_project_overlaps_without_audio_or_history_changes() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let mut original = document.timeline.lock().unwrap().as_ref().unwrap().clone();
        for (index, track) in original.tracks.iter().enumerate() {
            original.params_by_root_track.insert(
                track.id.clone(),
                hifishifter_kernel::state::TrackParamsState {
                    frame_period_ms: 5.,
                    pitch_edit: vec![if index == 0 { 60. } else { 67. }; 2],
                    ..Default::default()
                },
            );
        }
        let identities = document.parameter_identities_locked(&original).unwrap();
        let atlas = super::super::parameter_atlas::ParameterAtlas::default()
            .capture(&original, &identities)
            .unwrap();
        let root = original.tracks[0].id.clone();
        let ca = original.clips[0].id.clone();
        let cb = original.clips[1].id.clone();
        original.clips[1].track_id = root.clone();
        original.tracks.truncate(1);
        original.selected_clip_id = Some(ca.clone());
        let atlas = atlas.follow_geometry(&original, &identities).unwrap();
        original.params_by_root_track = atlas.project_roots(&original, &identities).unwrap();
        *document.timeline.lock().unwrap() = Some(original.clone());
        document.track_bindings.lock().unwrap().insert(
            root.clone(),
            vec![
                ("modification".into(), "ara://source".into()),
                ("modification-b".into(), "ara://source".into()),
            ],
        );
        {
            let mut edits = document.edits.lock().unwrap();
            edits.params = original.params_by_root_track;
            edits.atlas = atlas;
        }
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "atlas-select".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let root = loaded["tracks"][0]["id"].as_str().unwrap();
        let ca = format!("{}{ca}", editor.namespace);
        let cb = format!("{}{cb}", editor.namespace);
        let before = (
            document.edits.lock().unwrap().revision,
            editor.generation.load(Ordering::Acquire),
        );
        call(&editor, &sink, &rx, 2, "select_clip", json!({"clipId":cb}));
        let b = editor.timeline.lock().unwrap().params_by_root_track[root].pitch_edit[0];
        call(&editor, &sink, &rx, 3, "select_clip", json!({"clipId":ca}));
        let a = editor.timeline.lock().unwrap().params_by_root_track[root].pitch_edit[0];
        let after = (
            document.edits.lock().unwrap().revision,
            editor.generation.load(Ordering::Acquire),
        );
        let history = editor.history.lock().unwrap().records.len();
        document.close();
        assert_eq!((a, b), (60., 67.));
        assert_eq!(before, after);
        assert_eq!(history, 0);
    }
    #[test]
    fn workspace_components_share_the_original_actor_and_cross_track_history() {
        let (_model, owners, _ids) = workspace_fixture();
        let a = owners[0].editor_session().unwrap();
        let b = owners[1].editor_session().unwrap();
        assert!(Arc::ptr_eq(&a, &b), "同文档必须只有一个actor/history权威");
        let (_other, owner, _id) = fixture();
        assert!(!Arc::ptr_eq(&a, &owner.editor_session().unwrap()));
        let (reply, rx) = mpsc::channel();
        let (events, event_rx) = mpsc::sync_channel(128);
        let sa = UiSink {
            view_id: "workspace-a".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let (reply, rb) = mpsc::channel();
        let (events, eb) = mpsc::sync_channel(128);
        let sb = UiSink {
            view_id: "workspace-b".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let leases = owners
            .iter()
            .map(super::super::routing::RouteLease::new)
            .collect::<Vec<_>>();
        let links = leases
            .iter()
            .map(|lease| {
                let link = Arc::new(super::super::routing::EditorLink::default());
                link.bind(std::process::id() as i64, lease.token()).unwrap();
                link
            })
            .collect::<Vec<_>>();
        let call = |editor: &EditorSession,
                    sink: &UiSink,
                    receiver: &mpsc::Receiver<Value>,
                    id: u64,
                    command: &str,
                    args: Value| {
            let link = links[usize::from(sink.view_id == "workspace-b")].clone();
            editor
                .enqueue(UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sink.clone(),
                    link: Some(link),
                })
                .unwrap();
            let result = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(result["id"], id);
            assert_eq!(result["ok"], true, "{result}");
            result["value"].clone()
        };
        let loaded = call(&a, &sa, &rx, 1, "get_timeline_state", json!({}));
        assert_eq!(loaded["tracks"].as_array().unwrap().len(), 2);
        let ta = loaded["tracks"][0]["id"].as_str().unwrap();
        let tb = loaded["tracks"][1]["id"].as_str().unwrap();
        call(
            &a,
            &sa,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":[64.],"checkpoint":false}),
        );
        call(&a, &sa, &rx, 3, "select_track", json!({"trackId":tb}));
        call(
            &a,
            &sa,
            &rx,
            4,
            "set_param_frames",
            json!({"trackId":ta,"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),
        );
        let second = call(&b, &sb, &rb, 5, "get_timeline_state", json!({}));
        assert_eq!(second["selected_track_id"], tb);
        assert_eq!(second["undo_depth"], 1);
        event_rx.try_iter().for_each(drop);
        eb.try_iter().for_each(drop);
        call(
            &b,
            &sb,
            &rb,
            6,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":[67.],"checkpoint":true}),
        );
        assert!(
            event_rx
                .try_iter()
                .any(|e| e["event"] == "plugin_host_changed"),
            "其它view需要收到原GUI时间线刷新事件"
        );
        assert!(eb.try_iter().any(|e| e["event"] == "plugin_host_changed"));
        call(&a, &sa, &rx, 7, "undo_timeline", json!({}));
        assert_eq!(
            a.timeline.lock().unwrap().params_by_root_track[ta].pitch_edit[0],
            60.
        );
        assert_eq!(
            a.timeline.lock().unwrap().params_by_root_track[tb].pitch_edit[0],
            64.
        );
        call(&b, &sb, &rb, 8, "redo_timeline", json!({}));
        assert_eq!(
            a.timeline.lock().unwrap().params_by_root_track[tb].pitch_edit[0],
            67.
        );
        assert!(event_rx.try_iter().any(|e| e["event"] == "history_state"));
        assert!(eb.try_iter().any(|e| e["event"] == "history_state"));
        owners[0].stop_editor();
        assert!(
            !a.closed.load(Ordering::Acquire),
            "关闭组件不能停止共享actor"
        );
        assert!(call(&b, &sb, &rb, 9, "get_ui_settings", json!({})).is_object());
        a.close();
    }
    #[test]
    fn workspace_queued_requests_revalidate_original_routes_and_reject_unknown_views() {
        use super::super::routing::{EditorLink, RouteLease};
        let (model, owners, _ids) = workspace_fixture();
        let editor = owners[0].editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let ra = RouteLease::new(&owners[0]);
        let rb = RouteLease::new(&owners[1]);
        let link = Arc::new(EditorLink::default());
        link.bind(std::process::id() as i64, ra.token()).unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-route".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        // 真实命令在timeline锁上停住，随后命令一定排队；clear/bind不需持document事务。
        let held = editor.timeline.lock().unwrap();
        editor
            .enqueue(UiRequest {
                id: 1,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link.clone()),
            })
            .unwrap();
        let track = held.tracks[0].id.clone();
        editor.enqueue(UiRequest {id:2,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
        link.clear();
        link.bind(std::process::id() as i64, rb.token()).unwrap();
        drop(held);
        let mut results = [
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
        ];
        results.sort_by_key(|r| r["id"].as_u64().unwrap());
        assert_eq!(
            results[1]["ok"], false,
            "排队入口不能重新绑定别的组件继续写入"
        );
        assert_eq!(model.session().edits.lock().unwrap().revision, 0);
        drop(rb);
        assert!(
            editor
                .enqueue(UiRequest {
                    id: 3,
                    command: "get_ui_settings".into(),
                    args: json!({}),
                    sink: sink.clone(),
                    link: Some(link.clone())
                })
                .is_err(),
            "过期route拒绝入队"
        );
        let unknown = Arc::new(EditorLink::default());
        assert!(editor
            .enqueue(UiRequest {
                id: 4,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(unknown)
            })
            .is_err());
        link.bind(std::process::id() as i64, ra.token()).unwrap();
        editor
            .enqueue(UiRequest {
                id: 5,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link.clone()),
            })
            .unwrap();
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(5)).unwrap()["ok"],
            true
        );
        let (reply, _) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(8);
        let forged = UiSink {
            view_id: sink.view_id.clone(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        assert!(editor
            .enqueue(UiRequest {
                id: 6,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: forged,
                link: Some(link.clone())
            })
            .is_err());
        sink.closed.store(true, Ordering::Release);
        assert!(editor
            .enqueue(UiRequest {
                id: 7,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink,
                link: Some(link)
            })
            .is_err());
        editor.close();
    }
    #[test]
    fn workspace_queued_origin_close_cannot_borrow_the_other_live_component() {
        let (_model, owners, _ids) = workspace_fixture();
        let editor = owners[0].editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let lease = super::super::routing::RouteLease::new(&owners[0]);
        let link = Arc::new(super::super::routing::EditorLink::default());
        link.bind(std::process::id() as i64, lease.token()).unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, event_rx) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-origin-close".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let held = editor.timeline.lock().unwrap();
        editor
            .enqueue(UiRequest {
                id: 1,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link.clone()),
            })
            .unwrap();
        editor
            .enqueue(UiRequest {
                id: 2,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link),
            })
            .unwrap();
        owners[0].stop_editor();
        drop(held);
        editor.flush().unwrap();
        assert!(
            sink.closed.load(Ordering::Acquire),
            "组件关闭同步撤销旧view回信"
        );
        assert!(receiver.try_iter().next().is_none());
        assert!(!editor.closed.load(Ordering::Acquire));
        assert!(Arc::ptr_eq(&editor, &owners[1].editor_session().unwrap()));
        assert!(
            event_rx.try_iter().next().is_none(),
            "入口撤销后不能通过共享actor继续订阅事件"
        );
        editor.close();
    }
    #[test]
    fn workspace_document_close_joins_actor_revokes_views_and_releases_model_pcm() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let weak = Arc::downgrade(&document);
        let editor = owners[0].editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-lifecycle".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let started = Instant::now();
        document.close();
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "document close应join而不持事务等待actor"
        );
        assert!(editor.worker.lock().unwrap().is_none());
        assert!(editor.analysis_workers.lock().unwrap().is_empty());
        assert!(document.timeline.lock().unwrap().is_none());
        assert!(document.edit_sources.lock().unwrap().is_empty());
        assert!(
            sink.closed.load(Ordering::Acquire),
            "文档关闭撤销全部view回信/事件"
        );
        assert!(owners[1].editor_session().is_err());
        assert!(document.workspace_projection().is_err());
        drop(document);
        drop(model);
        assert!(weak.upgrade().is_none(), "actor不能强持document");
    }
    #[test]
    fn workspace_scope_and_geometry_checks_reject_partial_or_stale_transactions() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let host = document.workspace_timeline().unwrap();
        let previous = document.workspace_projection().unwrap();
        let version = document.revision.load(Ordering::Acquire);
        for changed in 0..5 {
            let mut client = host.clone();
            match changed {
                0 => {
                    client.tracks.pop();
                }
                1 => client.tracks[1] = client.tracks[0].clone(),
                2 => {
                    client.clips.pop();
                }
                3 => client.clips[0].takes[0].source_path = Some("C:/private.wav".into()),
                _ => client.clips[0].takes[0].playback_rate = 2.,
            }
            assert!(document
                .accept_workspace_edits(0, version, &client, &previous)
                .is_err());
            assert_eq!(
                document.edits.lock().unwrap().revision,
                0,
                "拒绝前不能部分提交"
            );
        }
        owners[0].stop_editor();
        assert_eq!(document.revision.load(Ordering::Acquire), version);
        assert!(
            document
                .accept_workspace_edits(0, version, &host, &previous)
                .unwrap_err()
                .starts_with("Conflict"),
            "scope必须独立于model校验"
        );
        assert_eq!(document.edits.lock().unwrap().revision, 0);
    }
    #[test]
    fn workspace_accept_rejects_forged_geometry_before_changing_document_edits() {
        let (model, _owners, _ids) = workspace_fixture();
        let document = model.session();
        let projection = document.workspace_projection().unwrap();
        let mut client = document.workspace_timeline().unwrap();
        client.clips[0].start_sec = 42.;
        assert!(
            document
                .accept_workspace_edits(
                    0,
                    document.revision.load(Ordering::Acquire),
                    &client,
                    &projection
                )
                .is_err(),
            "宿主几何必须拒绝而不是忽略"
        );
        assert_eq!(document.edits.lock().unwrap().revision, 0);
    }
    #[test]
    fn workspace_global_solo_keeps_each_renderer_output_independent() {
        let (_model, owners, _ids) = workspace_fixture();
        let editor = owners[0].editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-solo".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_track_state",
            json!({"trackId":track,"solo":true,"volume":0.5}),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while editor.applied.load(Ordering::Acquire) < 1 {
            assert!(
                Instant::now() < deadline,
                "{:?}",
                editor.error.lock().unwrap()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let outputs = owners
            .iter()
            .map(|owner| {
                let mut left = [9.; 4];
                let mut right = [9.; 4];
                let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut bus = crate::audio_abi::AudioBusBuffers {
                    num_channels: 2,
                    silence_flags: 0,
                    channel_buffers: planes.as_mut_ptr(),
                };
                // SAFETY: 每个publisher仅写本次提供的两个四帧平面。
                assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
                left
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], [0.05, 0.1, 0.15, 0.2]);
        assert_eq!(outputs[1], [0.; 4], "其它轨solo不能在本renderer投影后丢失");
        editor.close();
    }
    #[test]
    fn actor_accepts_original_curve_and_save_flush_keeps_tail_after_ui_closed() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "test-view".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        assert!(track.starts_with(&editor.namespace));
        assert_ne!(timeline["clips"][0]["source_path"], "ara://source");
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_param_frames",
            json!({"trackId":track,"param":"pitch","startFrame":0,"values":[60.0],"checkpoint":true}),
        );
        sink.closed.store(true, Ordering::Release);
        editor.enqueue(UiRequest {id:3,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":1,"values":[64.0],"checkpoint":false}),sink,link:None}).unwrap();
        let encoded = owner.encode_state().unwrap(); // 真getState路径会flush，不只测手工barrier。
        let saved: Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            saved["version"], 3,
            "新源basis使用v3，旧v2解码/范围回归另行保留"
        );
        assert_eq!(
            &saved["edits"]["params"]["track"]["pitch_edit"]
                .as_array()
                .unwrap()[..2],
            &[json!(60.), json!(64.)]
        );
        assert_eq!(
            editor.history.lock().unwrap().position,
            1,
            "尾块不得增加undo步"
        );
        editor.close();
    }
    #[test]
    fn invalid_track_patch_cannot_partially_mutate_volume_or_create_history() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let track = editor.timeline.lock().unwrap().tracks[0].id.clone();
        let result = super::super::commands::dispatch(
            &editor,
            "set_track_state",
            json!({"trackId":track,"volume":0.5,"pitchAnalysisAlgo":"not-an-algorithm"}),
        );
        assert!(result.is_err());
        assert_eq!(editor.timeline.lock().unwrap().tracks[0].volume, 1.);
        assert!(editor.history.lock().unwrap().records.is_empty());
        assert_eq!(editor.generation.load(Ordering::Acquire), 0);
        assert!(editor.check_source("C:/Users/user/private.wav").is_err());
        editor.close();
    }
    /// 原前端以base+position消费；停播seek也必须得到宿主绝对位置。
    #[test]
    fn host_cursor_is_absolute_once_and_tracks_stopped_seeks() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        // 时钟投影与分析无关；标记夹具已加载，避免为四帧源启动异步ONNX预热。
        editor.loaded.lock().unwrap().initialized = true;
        owner
            .clock
            .get()
            .unwrap()
            .update(&crate::audio_abi::ProcessContext {
                state: 0,
                sample_rate: 44100.,
                project_time_samples: 88200,
                ..Default::default()
            });
        let playback =
            super::super::commands::dispatch(&editor, "get_playback_state", json!({})).unwrap();
        let timeline =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        editor.close();
        assert_eq!(
            playback["base_sec"].as_f64().unwrap() + playback["position_sec"].as_f64().unwrap(),
            2.
        );
        assert_eq!(timeline["playhead_sec"], 2.);
    }
    /// 宿主tempo/geometry通知必须在无人点击重新载入时进入原GUI。
    #[test]
    fn host_changes_and_bpm_sync_automatically_without_reloading() {
        let (model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "host-sync".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        owner
            .clock
            .get()
            .unwrap()
            .update(&crate::audio_abi::ProcessContext {
                state: 1 << 10,
                sample_rate: 44100.,
                tempo: 150.,
                ..Default::default()
            });
        let document = model.session();
        document.timeline.lock().unwrap().as_mut().unwrap().clips[0].start_sec = 2.;
        document.revision.fetch_add(1, Ordering::AcqRel);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let synced = editor.timeline.lock().unwrap().clone();
            if synced.clips[0].start_sec == 2. && synced.bpm == 150. {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "宿主改动没有自动同步: start={} bpm={}",
                synced.clips[0].start_sec,
                synced.bpm
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        editor.close();
    }
    /// 首次遇到用户明确暂不做的倒放必须显示原因，不能只永远显示等待音频。
    #[test]
    fn unsupported_first_load_surfaces_geometry_error() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let mut timeline = document.timeline.lock().unwrap();
        let clip = &mut timeline.as_mut().unwrap().clips[0];
        clip.reversed = true;
        clip.takes[0].reversed = true;
        drop(timeline);
        let editor = owner.editor_session().unwrap();
        assert!(editor
            .ensure_loaded(false)
            .unwrap_err()
            .starts_with("Unsupported"));
        owner
            .clock
            .get()
            .unwrap()
            .update(&crate::audio_abi::ProcessContext {
                state: 1 << 1,
                sample_rate: 44100.,
                project_time_samples: 44100,
                ..Default::default()
            });
        let playing =
            super::super::commands::dispatch(&editor, "get_playback_state", json!({})).unwrap();
        assert_eq!(playing["is_playing"], true);
        assert_eq!(playing["position_sec"], 1.);
        owner.clock.get().unwrap().stopped();
        assert_eq!(
            super::super::commands::dispatch(&editor, "get_playback_state", json!({})).unwrap()
                ["is_playing"],
            false
        );
        let state = editor.state();
        editor.close();
        assert!(state["error"].as_str().unwrap().starts_with("Unsupported"));
        assert_eq!(state["ready"], false);
    }
    /// 同一已打开会话只改变GUI清单代次，读取仍采纳新增/删除；不依赖重开或定时器抢跑。
    #[test]
    fn inventory_only_read_refreshes_existing_editor_without_resetting_parameter_history() {
        let (model, owner, _) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        // 暂停idle pump，使测试准确检查命令自身的缓存早退，而非碰巧被后台刷新。
        editor.queue.send(Job::Close).unwrap();
        editor
            .worker
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .join()
            .unwrap();
        let loaded =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        let root = loaded["tracks"][0]["id"].as_str().unwrap().to_owned();
        super::super::commands::dispatch(
            &editor,
            "set_param_frames",
            json!({"trackId":root,"param":"hifigan_tension",
            "startFrame":0,"values":[35.,45.],"checkpoint":true}),
        )
        .unwrap();
        let versions = document.editor_versions().unwrap();
        let history =
            super::super::commands::dispatch(&editor, "get_history_state", json!({})).unwrap();
        let selected = editor.timeline.lock().unwrap().selected_clip_id.clone();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let host_api = Arc::new(host.client());
        let mut track = host_api.ui_track(&|| true).unwrap();
        track.id = "track".into();
        let item = track.items[0].clone();
        track.items.clear();
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track.clone());
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        track.items.push(item.clone());
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track.clone());
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        let added =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        let id = format!("{}ara-item-{}", editor.namespace, item.geometry.item_id);
        assert!(
            added["clips"]
                .as_array()
                .unwrap()
                .iter()
                .any(|clip| clip["id"] == id),
            "当前UI应立刻看见新item"
        );
        assert_eq!(
            document.editor_versions().unwrap(),
            versions,
            "清单不伪造ARA/edit/scope代次"
        );
        assert_eq!(editor.timeline.lock().unwrap().selected_clip_id, selected);
        track.items.clear();
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        let removed =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        assert!(!removed["clips"]
            .as_array()
            .unwrap()
            .iter()
            .any(|clip| clip["id"] == id));
        assert_eq!(
            super::super::commands::dispatch(&editor, "get_history_state", json!({})).unwrap(),
            history
        );
        let params = super::super::commands::dispatch(
            &editor,
            "get_param_frames",
            json!({"trackId":root,"param":"hifigan_tension",
            "startFrame":0,"frameCount":2,"binary":false}),
        )
        .unwrap();
        assert_eq!(params["edit"], json!([35., 45.]));
        document.close();
    }
    /// 已授权生成的波形在mute显示占位期间保留；take更换后不错误沿用旧图。
    #[test]
    fn muted_display_keeps_authorized_waveform_without_reentering_audio_assignment() {
        let (model, owner, _) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let host_api = Arc::new(host.client());
        let mut track = host_api.ui_track(&|| true).unwrap();
        let mut audio = editor.timeline.lock().unwrap().clips[0].clone();
        let path = audio.source_path.clone().unwrap();
        audio.id = format!(
            "{}ara-item-{}",
            editor.namespace, track.items[0].geometry.item_id
        );
        audio.track_id = format!("{}{}", editor.namespace, track.id);
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track.clone());
        let mut visible = TimelineState::default();
        visible.tracks.clear();
        visible.clips = vec![audio];
        editor.retain_display_waveforms(&mut visible, false);
        track.items[0].geometry.muted = true;
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track.clone());
        visible.clips.clear();
        document.present_host_inventory(&mut visible, &editor.namespace);
        editor.retain_display_waveforms(&mut visible, true);
        assert!(visible.clips[0].muted);
        assert_eq!(visible.clips[0].source_path.as_ref(), Some(&path));
        assert!(editor.check_source(&path).is_ok());
        track.items[0].geometry.take_id = "different-take".into();
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);
        visible.clips.clear();
        document.present_host_inventory(&mut visible, &editor.namespace);
        editor.retain_display_waveforms(&mut visible, true);
        assert!(visible.clips[0].source_path.is_none());
        document.close();
    }
    /// 真实宿主正向倍率不再被GUI挡住；take的源窗口和项目时长保留各自坐标。
    #[test]
    fn forward_time_stretch_loads_the_real_host_geometry_in_original_gui_state() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        {
            let mut timeline = document.timeline.lock().unwrap();
            let clip = &mut timeline.as_mut().unwrap().clips[0];
            clip.length_sec = 8.0 / 44100.0;
            clip.takes[0].playback_rate = 0.5;
            clip.normalize_takes();
        }
        {
            let mut regions = document.regions.lock().unwrap();
            for region in regions.values_mut() {
                region.duration_in_playback_time = 8.0 / 44100.0;
                region.is_timestretch_enabled = true;
            }
        }
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let timeline = editor.timeline.lock().unwrap();
        assert_eq!(timeline.clips[0].playback_rate, 0.5);
        assert_eq!(timeline.clips[0].length_sec, 8.0 / 44100.0);
        drop(timeline);
        editor.close();
    }
    /// 两根的源分析已缓存后切回另一素材，再改隐藏根；不调用reload或原线getter推动应用。
    #[test]
    fn cached_other_track_pitch_applies_after_selection_without_reload_or_parameter_polling() {
        let (model, owners, _ids) = task34_world_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "cached-other-track".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let roots = loaded["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|track| track["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let ca = loaded["clips"][0]["id"].as_str().unwrap();
        let began = Instant::now();
        loop {
            for root in &roots {
                call(
                    &editor,
                    &sink,
                    &rx,
                    2,
                    "get_param_frames",
                    json!({"trackId":root,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}),
                );
            }
            if roots.iter().all(|root| {
                editor.timeline.lock().unwrap().params_by_root_track[root]
                    .pitch_orig_key
                    .is_some()
            }) {
                break;
            }
            assert!(began.elapsed() < Duration::from_secs(20));
            std::thread::sleep(Duration::from_millis(10));
        }
        let wait = |target| {
            let began = Instant::now();
            while editor.applied.load(Ordering::Acquire) < target {
                assert!(
                    began.elapsed() < Duration::from_secs(10),
                    "{:?}",
                    editor.error.lock().unwrap()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        call(
            &editor,
            &sink,
            &rx,
            3,
            "set_param_frames",
            json!({"trackId":roots[1],"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":true}),
        );
        wait(1);
        call(&editor, &sink, &rx, 4, "select_clip", json!({"clipId":ca}));
        call(
            &editor,
            &sink,
            &rx,
            5,
            "set_param_frames",
            json!({"trackId":roots[1],"param":"pitch","startFrame":0,"values":vec![67.;400],"checkpoint":true}),
        );
        wait(2);
        let restored = owners[1].encode_state().unwrap();
        let accepted = document.edits.lock().unwrap().params["b"].pitch_edit[100];
        document.close();
        assert_eq!(accepted, 67.);
        assert!(!restored.is_empty());
    }
    /// GUI保持连续只读请求时，150ms到期的自动应用仍应先行，不能等FIFO空闲才渲染。
    #[test]
    fn automatic_apply_is_not_starved_by_continuous_read_only_requests() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "read-flood".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":loaded["tracks"][0]["id"],"volume":0.5}),
        );
        let running = Arc::new(AtomicBool::new(true));
        let flag = running.clone();
        let client = editor.clone();
        let view = sink.clone();
        let poller = std::thread::spawn(move || {
            while flag.load(Ordering::Acquire) {
                let _ = client.enqueue(UiRequest {
                    id: 99,
                    command: "get_ui_settings".into(),
                    args: json!({}),
                    sink: view.clone(),
                    link: None,
                });
                std::thread::yield_now();
            }
        });
        let began = Instant::now();
        while editor.applied.load(Ordering::Acquire) < 1 && began.elapsed() < Duration::from_secs(5)
        {
            rx.try_iter().for_each(drop);
            std::thread::sleep(Duration::from_millis(5));
        }
        let applied = editor.applied.load(Ordering::Acquire);
        running.store(false, Ordering::Release);
        poller.join().unwrap();
        document.close();
        assert!(applied >= 1, "不断只读轮询不能饿死已经到期的应用");
    }
    /// 实际合成尚占actor时，播放态查询不得等待编辑timeline锁；参数权威仍走原队列。
    #[test]
    fn playback_observation_does_not_wait_for_actor_timeline_or_render() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        document.clock.publish_host(3.125, true);
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(4);
        let sink = UiSink {
            view_id: "fast-transport".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let held = editor.timeline.lock().unwrap();
        editor
            .enqueue(UiRequest {
                id: 1,
                command: "get_playback_state".into(),
                args: json!({}),
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let response = rx.recv_timeout(Duration::from_millis(200)).unwrap();
        drop(held);
        assert_eq!(response["value"]["position_sec"], 3.125);
        assert_eq!(response["value"]["is_playing"], true);
        document.close();
        assert!(editor
            .enqueue(UiRequest {
                id: 2,
                command: "get_playback_state".into(),
                args: json!({}),
                sink,
                link: None
            })
            .is_err());
    }
    /// 暂时渲染失败只影响就绪状态；宿主恢复后自动重试，不能要求用户手工reload。
    #[test]
    fn automatic_apply_recovers_from_render_failure_without_reload() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "render-retry".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        for region in document.regions.lock().unwrap().values_mut() {
            region.has_content_based_fade_at_head = true;
        }
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":loaded["tracks"][0]["id"],"volume":0.5}),
        );
        let began = Instant::now();
        while editor.render_error.lock().unwrap().is_none() {
            assert!(began.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(editor.error.lock().unwrap().is_none());
        assert_eq!(editor.applied.load(Ordering::Acquire), 0);
        for region in document.regions.lock().unwrap().values_mut() {
            region.has_content_based_fade_at_head = false;
        }
        let began = Instant::now();
        while editor.applied.load(Ordering::Acquire) < 1 {
            assert!(
                began.elapsed() < Duration::from_secs(5),
                "{:?}",
                editor.state()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(editor.render_error.lock().unwrap().is_none());
        document.close();
    }
    /// 没有手绘pitch的HiFiGAN气声/张力/共振峰也要等原线；None与纯混音仍能立即应用。
    #[test]
    fn effect_only_analysis_gate_does_not_block_none_or_plain_mix() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let mut timeline = editor.timeline.lock().unwrap().clone();
        let root = timeline.tracks[0].id.clone();
        timeline.ensure_params_for_root(&root);
        timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::NsfHifiganOnnx;
        timeline.tracks[0].compose_enabled = true;
        for effect in ["breath_enabled", "hifigan_tension", "formant_shift_cents"] {
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.extra_params.clear();
            params.extra_curves.clear();
            if effect == "breath_enabled" {
                params.extra_params.insert(effect.into(), 1.);
            } else {
                params.extra_curves.insert(effect.into(), vec![100.; 200]);
                if effect == "hifigan_tension" {
                    params.extra_params.insert("breath_enabled".into(), 1.);
                }
            }
            params.pitch_orig_key = None;
            assert!(!params.pitch_edit_user_modified);
            assert!(
                EditorSession::requires_analysis(&timeline),
                "{effect}缺原线不能先发布未处理PCM"
            );
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::None;
            assert!(!EditorSession::requires_analysis(&timeline));
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::NsfHifiganOnnx;
        }
        let params = timeline.params_by_root_track.get_mut(&root).unwrap();
        params.extra_params.clear();
        params.extra_curves.clear();
        assert!(!EditorSession::requires_analysis(&timeline));
        editor.close();
    }
    /// 三分钟真实授权PCM同进程Arc冻结，原GUI流式分析副本不受旧30秒门禁或整源复制影响。
    #[test]
    fn long_host_source_loads_original_gui_through_shared_pcm_without_ipc_copy() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let seconds = 180.;
        let frames = 44100 * 180;
        let budget = crate::render::budget::global_budget();
        let baseline = budget.used();
        let pcm = Arc::new(SourcePcm {
            sample_rate: 44100,
            planes: vec![vec![0.125; frames]],
            version: 1,
            _reservation: Some(budget.reserve(frames * 4).unwrap()),
        });
        document
            .sources
            .lock()
            .unwrap()
            .insert("ara://source".into(), pcm.clone());
        document
            .edit_sources
            .lock()
            .unwrap()
            .insert("ara://source".into(), pcm.clone());
        {
            let mut timeline = document.timeline.lock().unwrap();
            let timeline = timeline.as_mut().unwrap();
            timeline.project_sec = seconds;
            for clip in &mut timeline.clips {
                clip.length_sec = seconds;
                clip.takes[0].source_end_sec = seconds;
                clip.normalize_takes();
            }
        }
        for region in document.regions.lock().unwrap().values_mut() {
            region.duration_in_modification_time = seconds;
            region.duration_in_playback_time = seconds;
        }
        let (snapshot, _, _) = document.workspace_snapshot().unwrap();
        assert!(Arc::ptr_eq(&snapshot.sources[0].1, &pcm));
        assert_eq!(
            budget.used(),
            baseline + frames * 4,
            "冻结不能复制一份整源PCM"
        );
        drop(snapshot);
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let timeline = editor.timeline.lock().unwrap();
        assert_eq!(timeline.clips[0].length_sec, seconds);
        assert_eq!(timeline.clips[0].duration_frames, Some(frames as u64));
        let path = timeline.clips[0].source_path.clone().unwrap();
        drop(timeline);
        use std::io::{Read, Seek};
        let mut wav = std::fs::File::open(path).unwrap();
        assert!(wav.metadata().unwrap().len() >= frames as u64 * 4);
        wav.seek(std::io::SeekFrom::End(-4)).unwrap();
        let mut tail = [0_u8; 4];
        wav.read_exact(&mut tail).unwrap();
        assert_eq!(f32::from_le_bytes(tail), 0.125);
        document.close();
        drop(pcm);
        assert_eq!(budget.used(), baseline);
    }
    /// 双轨各三分钟保留旧就绪快照时，修改另一轨仍能原子更新；双侧输出不能相加/串轨。
    #[test]
    fn two_long_tracks_update_atomically_with_old_ready_audio_inside_pcm_budget() {
        let (model, owners, ids) = workspace_fixture();
        let document = model.session();
        document.ready.store(false, Ordering::Release);
        let budget = crate::render::budget::global_budget();
        let baseline = budget.used();
        let frames = 44100 * 180;
        let seconds = 180.;
        let source = |value| {
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![value; frames]],
                version: 1,
                _reservation: Some(budget.reserve(frames * 4).unwrap()),
            })
        };
        let a = source(0.25);
        let b = source(0.5);
        {
            let _transaction = document.transaction.lock().unwrap();
            let mut timeline = document.timeline.lock().unwrap();
            let timeline = timeline.as_mut().unwrap();
            timeline.project_sec = seconds;
            for (index, clip) in timeline.clips.iter_mut().enumerate() {
                clip.length_sec = seconds;
                clip.takes[0].source_end_sec = seconds;
                clip.takes[0].source_path = Some(
                    if index == 0 {
                        "ara://source"
                    } else {
                        "ara://source-b"
                    }
                    .into(),
                );
                clip.normalize_takes();
            }
            for (index, id) in ids.iter().enumerate() {
                let key = (&**id as *const u8) as u64;
                let mut regions = document.regions.lock().unwrap();
                let region = regions.get_mut(&key).unwrap();
                region.duration_in_modification_time = seconds;
                region.duration_in_playback_time = seconds;
                if index == 1 {
                    region.audio_source_persistent_id = "ara://source-b".into();
                }
            }
            document.track_bindings.lock().unwrap().insert(
                "b".into(),
                vec![("modification-b".into(), "ara://source-b".into())],
            );
            for sources in [&document.sources, &document.edit_sources] {
                let mut sources = sources.lock().unwrap();
                sources.insert("ara://source".into(), a.clone());
                sources.insert("ara://source-b".into(), b.clone());
            }
            document.revision.fetch_add(1, Ordering::AcqRel);
            document.render_epoch.fetch_add(1, Ordering::AcqRel);
            document.ready.store(true, Ordering::Release);
        }
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "long-multitrack".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let wait = |generation| {
            let began = Instant::now();
            loop {
                let pending = editor.state()["pending"].as_bool().unwrap();
                if !pending && editor.applied.load(Ordering::Acquire) >= generation {
                    break;
                }
                assert!(
                    began.elapsed() < Duration::from_secs(30),
                    "{:?}",
                    editor.state()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        wait(0);
        let check = |index: usize, rate: usize, value: f32| {
            let mut left = [9_f32; 513];
            let mut right = [9_f32; 513];
            let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut bus = crate::audio_abi::AudioBusBuffers {
                num_channels: 2,
                silence_flags: 0,
                channel_buffers: planes.as_mut_ptr(),
            };
            // SAFETY: 两个513帧平面存活；逐轨读取尾部511帧，剩余2帧必须补零。
            assert!(unsafe {
                owners[index].snapshots[usize::from(rate == 48000)].copy_block(
                    (rate * 180 - 511) as i64,
                    rate as u32,
                    &mut bus,
                    513,
                )
            });
            assert_eq!(&left[..511], &[value; 511]);
            assert_eq!(right, left);
            assert_eq!(&left[511..], &[0.; 2]);
        };
        for rate in [44100, 48000] {
            check(0, rate, 0.25);
            check(1, rate, 0.5);
        }
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":loaded["tracks"][1]["id"],"volume":0.5}),
        );
        wait(1);
        for rate in [44100, 48000] {
            check(0, rate, 0.25);
            check(1, rate, 0.25);
        }
        for owner in &owners {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
        }
        let ready_bytes = (44100 + 48000) * 180 * 4 * 2;
        let curve_bytes = document.edits.lock().unwrap().atlas.accounted_curve_bytes();
        assert_eq!(
            budget.used() - baseline,
            frames * 4 * 2 + ready_bytes + curve_bytes
        );
        assert_eq!(editor.state()["error"], Value::Null);
        eprintln!("two-long-tracks seconds=180 tracks=2 ready_bytes={ready_bytes} curve_bytes={curve_bytes} accounted_peak={} hard_limit={} automatic_other_track=true",budget.peak(),budget.limit());
        document.close();
        drop(a);
        drop(b);
        for owner in &owners {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
        }
        assert_eq!(budget.used(), baseline);
    }
    /// 无本地落笔的宿主移动也要重新准备；cache全命中不会发ClipPitchReady，不能静默遗漏。
    #[test]
    fn host_move_without_local_generation_reapplies_and_reuses_analysis_file() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let before_path = editor.timeline.lock().unwrap().clips[0]
            .source_path
            .clone()
            .unwrap();
        let modified = std::fs::metadata(&before_path).unwrap().modified().unwrap();
        {
            let _transaction = document.transaction.lock().unwrap();
            let mut timeline = document.timeline.lock().unwrap();
            let timeline = timeline.as_mut().unwrap();
            timeline.clips[0].start_sec = 1.;
            timeline.project_sec = 2.;
            for region in document.regions.lock().unwrap().values_mut() {
                region.start_in_playback_time = 1.;
            }
            document.revision.fetch_add(1, Ordering::AcqRel);
            document.render_epoch.fetch_add(1, Ordering::AcqRel);
        }
        let mut left = [0_f32; 4];
        let mut right = [0_f32; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        let began = Instant::now();
        loop {
            unsafe {
                owner.snapshots[0].copy_block(44100, 44100, &mut bus, 4);
            }
            if left == [0.1, 0.2, 0.3, 0.4] {
                break;
            }
            assert!(
                began.elapsed() < Duration::from_secs(5),
                "{:?}",
                editor.state()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let after_path = editor.timeline.lock().unwrap().clips[0]
            .source_path
            .clone()
            .unwrap();
        assert_eq!(after_path, before_path);
        assert_eq!(
            std::fs::metadata(after_path).unwrap().modified().unwrap(),
            modified
        );
        assert_eq!(editor.generation.load(Ordering::Acquire), 0);
        document.close();
    }
    /// PCM 晚到而 edit/model/scope 不变时，当前窗口仍补波形，保留选择和参数历史。
    #[test]
    fn late_host_pcm_refreshes_existing_editor_without_model_change() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let root = editor.timeline.lock().unwrap().tracks[0].id.clone();
        super::super::commands::dispatch(
            &editor,
            "set_track_state",
            json!({"trackId":root,"volume":0.5}),
        )
        .unwrap();
        let versions = document.editor_versions().unwrap();
        let selected = editor.timeline.lock().unwrap().selected_clip_id.clone();
        let history = editor.history.lock().unwrap().position;
        assert!(history > 0);
        {
            let mut timeline = editor.timeline.lock().unwrap();
            timeline.clips[0].source_path = None;
            for take in &mut timeline.clips[0].takes {
                take.source_path = None;
            }
        }
        document.publish_source_pcm(
            "ara://source".into(),
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![0.4, 0.3, 0.2, 0.1]],
                version: 0,
                _reservation: None,
            }),
        );
        assert_eq!(document.editor_versions().unwrap(), versions);
        editor.refresh_host();
        let timeline = editor.timeline.lock().unwrap();
        let clip = &timeline.clips[0];
        let path = clip
            .source_path
            .as_ref()
            .expect("late PCM must fill the visible clip");
        assert!(std::path::Path::new(path).is_file());
        assert_eq!(clip.takes[0].source_path.as_ref(), Some(path));
        assert_eq!(timeline.selected_clip_id, selected);
        assert_eq!(timeline.tracks[0].volume, 0.5);
        drop(timeline);
        assert_eq!(editor.history.lock().unwrap().position, history);
        document.close();
    }

    /// 同一ARA源ID换为不同PCM后，原线从57重建为69，保留目标60；不靠读取原线推动分析。
    #[test]
    fn same_host_source_identity_changed_pcm_invalidates_original_pitch_without_reload() {
        let (model, owners, _ids) = task34_world_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "same-uri-change".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let root = loaded["tracks"][0]["id"].as_str().unwrap();
        let wait = |predicate: &dyn Fn() -> bool| {
            let began = Instant::now();
            while !predicate() {
                assert!(
                    began.elapsed() < Duration::from_secs(25),
                    "{:?}",
                    editor.state()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        wait(&|| {
            editor
                .timeline
                .lock()
                .unwrap()
                .params_by_root_track
                .get(root)
                .is_some_and(|params| params.pitch_orig_key.is_some())
        });
        let old_path = editor.timeline.lock().unwrap().clips[0]
            .source_path
            .clone()
            .unwrap();
        assert!(
            (editor.timeline.lock().unwrap().params_by_root_track[root].pitch_orig[100] - 57.)
                .abs()
                < 1.
        );
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":root,"param":"pitch","startFrame":0,"values":vec![60.;400],"checkpoint":true}),
        );
        wait(&|| editor.applied.load(Ordering::Acquire) >= 1);
        let samples = (0..88200)
            .map(|n| {
                let phase = 2. * std::f64::consts::PI * 440. * n as f64 / 44100.;
                (1..=16)
                    .map(|harmonic| (phase * harmonic as f64).sin() * 0.2 / harmonic as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        {
            let _transaction = document.transaction.lock().unwrap();
            let pcm = Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![samples],
                version: 1,
                _reservation: None,
            });
            document
                .edit_sources
                .lock()
                .unwrap()
                .insert("ara://source".into(), pcm.clone());
            document
                .sources
                .lock()
                .unwrap()
                .insert("ara://source".into(), pcm);
            document.revision.fetch_add(1, Ordering::AcqRel);
            document.render_epoch.fetch_add(1, Ordering::AcqRel);
        }
        wait(&|| {
            let timeline = editor.timeline.lock().unwrap();
            timeline
                .params_by_root_track
                .get(root)
                .is_some_and(|params| {
                    params.pitch_orig_key.is_some()
                        && params
                            .pitch_orig
                            .get(100)
                            .is_some_and(|note| (*note - 69.).abs() < 1.)
                })
        });
        wait(&|| {
            document
                .edits
                .lock()
                .unwrap()
                .params
                .get("track")
                .is_some_and(|params| {
                    params
                        .pitch_orig
                        .get(100)
                        .is_some_and(|note| (*note - 69.).abs() < 1.)
                })
        });
        let timeline = editor.timeline.lock().unwrap();
        assert_ne!(
            timeline.clips[0].source_path.as_deref(),
            Some(old_path.as_str())
        );
        assert_eq!(timeline.params_by_root_track[root].pitch_edit[100], 60.);
        drop(timeline);
        document.close();
    }
    /// 真actor的自动调度消费快速写入/undo/redo，不调用手工render函数冒充自动应用。
    #[test]
    fn automatic_audio_uses_latest_edit_and_follows_undo_redo_without_ui() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "audio-view".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_track_state",
            json!({"trackId":track,"volume":0.25}),
        );
        call(
            &editor,
            &sink,
            &receiver,
            3,
            "set_track_state",
            json!({"trackId":track,"volume":0.5}),
        );
        let wait = |expected: u64| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while editor.applied.load(Ordering::Acquire) < expected {
                assert!(
                    Instant::now() < deadline,
                    "automatic apply failed: {:?}",
                    editor.error.lock().unwrap()
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let output = || {
            let mut left = [9_f32; 4];
            let mut right = [9_f32; 4];
            let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut bus = crate::audio_abi::AudioBusBuffers {
                num_channels: 2,
                silence_flags: 0,
                channel_buffers: planes.as_mut_ptr(),
            };
            // SAFETY: 两个四帧可写平面完整覆盖copy_block，owner与snapshot仍存活。
            assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
            (left, right)
        };
        wait(2);
        for plane in [output().0, output().1] {
            for (actual, want) in plane.into_iter().zip([0.05, 0.1, 0.15, 0.2]) {
                assert!((actual - want).abs() < 1e-6);
            }
        }
        call(&editor, &sink, &receiver, 4, "undo_timeline", json!({}));
        wait(3);
        for (actual, want) in output().0.into_iter().zip([0.025, 0.05, 0.075, 0.1]) {
            assert!((actual - want).abs() < 1e-6);
        }
        call(&editor, &sink, &receiver, 5, "redo_timeline", json!({}));
        wait(4);
        for (actual, want) in output().0.into_iter().zip([0.05, 0.1, 0.15, 0.2]) {
            assert!((actual - want).abs() < 1e-6);
        }
        sink.closed.store(true, Ordering::Release);
        editor
            .enqueue(UiRequest {
                id: 6,
                command: "set_track_state".into(),
                args: json!({"trackId":track,"muted":true}),
                sink,
                link: None,
            })
            .unwrap();
        wait(5);
        assert_eq!(
            output().0,
            [0.; 4],
            "关UI取消响应，不能取消已排参数或后台供音"
        );
        editor.close();
    }
    /// WORLD自动作业实际改变音高；频率从输出PCM独立自相关估算，不读参数分析结果。
    #[test]
    fn automatic_world_pitch_changes_real_pcm_without_manual_submit() {
        world_pitch_case(true);
    }
    /// 分析前落笔也需自动收敛，不依赖GUI不停请求get_param_frames才能供修音音频。
    #[test]
    fn early_pitch_waits_for_analysis_and_applies_without_ui_polling() {
        world_pitch_case(false);
    }
    fn world_pitch_case(wait_for_analysis: bool) {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        {
            let mut known = document.timeline.lock().unwrap();
            let timeline = known.as_mut().unwrap();
            timeline.project_sec = 2.;
            timeline.tracks[0].compose_enabled = true;
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            timeline.clips[0].length_sec = 2.;
            timeline.clips[0].takes[0].source_end_sec = 2.;
            timeline.clips[0].normalize_takes();
        }
        for region in document.regions.lock().unwrap().values_mut() {
            region.duration_in_modification_time = 2.;
            region.duration_in_playback_time = 2.;
        }
        let samples: Vec<f32> = (0..88200)
            .map(|n| {
                let phase = 2. * std::f64::consts::PI * 220. * n as f64 / 44100.;
                // 与既有WORLD oracle相同的有声谐波结构；三谐波被Harvest判为清音。
                (1..=16)
                    .map(|harmonic| (phase * harmonic as f64).sin() * 0.2 / harmonic as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        document.edit_sources.lock().unwrap().insert(
            "ara://source".into(),
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![samples],
                version: 0,
                _reservation: None,
            }),
        );
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "world-view".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        assert_eq!(timeline["clips"].as_array().unwrap().len(), 1);
        {
            let live = editor.timeline.lock().unwrap();
            assert!(live.tracks[0].compose_enabled);
            assert!(!live.clips[0].muted);
            assert_eq!(live.clips[0].track_id, live.tracks[0].id);
            assert!(std::path::Path::new(live.clips[0].source_path.as_ref().unwrap()).is_file());
        }
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        let analysis_deadline = Instant::now() + Duration::from_secs(15);
        while wait_for_analysis {
            let frames = call(
                &editor,
                &sink,
                &receiver,
                10,
                "get_param_frames",
                json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}),
            );
            if frames["orig"]
                .as_array()
                .is_some_and(|orig| orig.iter().filter_map(Value::as_f64).any(|p| p > 30.))
            {
                break;
            }
            assert!(
                Instant::now() < analysis_deadline,
                "原GUI真实get_param_frames分析未完成: {frames}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_param_frames",
            json!({"trackId":track,"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":true}),
        );
        {
            let accepted = owner.edit_state();
            let accepted = accepted.lock().unwrap();
            assert!(
                accepted.params["track"].pitch_edit_user_modified,
                "参数写入必须保留用户修改标记"
            );
            assert!(
                matches!(
                    accepted.tracks[0].pitch_analysis_algo,
                    PitchAnalysisAlgo::WorldDll
                ),
                "接受后的算法必须WORLD"
            );
            assert_eq!(accepted.params["track"].pitch_edit[100], 64.);
            if wait_for_analysis {
                assert!(
                    accepted.params["track"].pitch_orig[100] > 30.,
                    "自动渲染的原线必须分析完成"
                );
            }
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        while editor.applied.load(Ordering::Acquire) < 1 {
            assert!(
                Instant::now() < deadline,
                "WORLD automatic apply failed: {:?}",
                editor.error.lock().unwrap()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut left = vec![0_f32; 8820];
        let mut right = vec![0_f32; 8820];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 两个8820帧平面存活，读取实际已经由actor发布的0.5秒起输出。
        assert!(unsafe { owner.snapshots[0].copy_block(22050, 44100, &mut bus, 8820) });
        let correlation = |lag: usize| {
            let mut cross = 0_f64;
            let mut first = 0_f64;
            let mut second = 0_f64;
            for i in 0..left.len() - lag {
                let a = left[i] as f64;
                let b = left[i + lag] as f64;
                cross += a * b;
                first += a * a;
                second += b * b;
            }
            cross / (first * second).sqrt().max(1e-20)
        };
        let lag = (88..=245)
            .max_by(|a, b| correlation(*a).total_cmp(&correlation(*b)))
            .unwrap();
        let frequency = 44100. / lag as f64;
        eprintln!("automatic WORLD output frequency={frequency:.3}Hz target=329.63Hz");
        assert!(
            (frequency - 329.63).abs() < 8.,
            "MIDI64应约329.63Hz，实际{frequency}Hz；不能只改变gain"
        );
        editor.close();
    }
    /// 缺少原声分析时不能把用户的pitch标成已应用；曲线权威仍可保存。
    #[test]
    fn pending_pitch_analysis_does_not_publish_raw_audio_as_applied() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        {
            let mut timeline = editor.timeline.lock().unwrap();
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            timeline.tracks[0].compose_enabled = true;
            let root = timeline.tracks[0].id.clone();
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.pitch_edit_user_modified = true;
            params.pitch_edit = vec![64.; 200];
            params.pitch_orig = vec![0.; 200];
            params.pitch_orig_key = None;
            editor.publish_timeline(timeline.clone());
        }
        let outcome = editor.apply(0, None);
        assert!(outcome.is_err(), "未分析的全零原线不能渲染原声并假报成功");
        assert!(outcome.unwrap_err().contains("analysis"));
        let bytes = owner.encode_state().unwrap();
        let saved: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(saved["edits"]["params"]["track"]["pitch_edit"][0], 64.);
        editor.close();
    }
}
