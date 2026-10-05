//! 真实ARA文档共享原编辑命令actor：UI只排队，分析/文件/离线渲染均不在音频或UI回调执行。
use crate::render::document::DocumentSession;
use hifishifter_kernel::editor::{history,host_pcm::{materialize,PcmView},ParamHost};
use hifishifter_kernel::state::*;
use serde_json::{json,Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc,Weak,Mutex,mpsc};
use std::sync::atomic::{AtomicBool,AtomicU64,Ordering};
use std::time::{Duration,Instant};

#[derive(Clone)]
pub(crate) struct UiSink {
    pub view_id:String,
    pub reply:mpsc::Sender<Value>,
    pub events:mpsc::SyncSender<Value>,
    pub closed:Arc<AtomicBool>,
}
impl UiSink {
    fn response(&self,value:Value) {
        if !self.closed.load(Ordering::Acquire) { let _=self.reply.send(value); }
    }
    fn event(&self,value:Value) {
        if !self.closed.load(Ordering::Acquire) { let _=self.events.try_send(value); }
    }
}
pub(crate) struct UiRequest {pub id:u64,pub command:String,pub args:Value,pub sink:UiSink,pub link:Option<Arc<super::routing::EditorLink>>}
enum Job {Request(UiRequest,Option<u64>),Barrier(mpsc::Sender<()>),Close}
struct RegisteredView {sink:UiSink,route:Option<(Arc<super::routing::EditorLink>,u64)>}
#[derive(Default)]
struct Loaded {
    initialized:bool,edit:u64,model:u64,scope:u64,
    projection:String,
    reverse_paths:HashMap<String,String>,
}
pub(crate) struct EditorSession {
    document:Weak<DocumentSession>,
    pub(super) timeline:Mutex<TimelineState>,
    pub(super) history:Mutex<TimelineHistory>,
    pub(super) project:Mutex<ProjectState>,
    pub(super) settings:Mutex<hifishifter_kernel::config::UiSettings>,
    loaded:Mutex<Loaded>,
    pub(super) namespace:String,
    pcm_dir:PathBuf,
    pub(super) peaks:Mutex<HashMap<String,Arc<hifishifter_kernel::hfspeaks_v2::HfsPeakFile>>>,
    queue:mpsc::SyncSender<Job>,worker:Mutex<Option<std::thread::JoinHandle<()>>>,
    analysis_sender:mpsc::Sender<hifishifter_kernel::engine_command::EngineCommand>,
    analysis_updates:Mutex<mpsc::Receiver<hifishifter_kernel::engine_command::EngineCommand>>,
    analysis_cancel:Arc<AtomicBool>,analysis_workers:Mutex<Vec<std::thread::JoinHandle<()>>>,
    views:Mutex<HashMap<String,RegisteredView>>,
    pub(super) generation:AtomicU64,
    submitted:AtomicU64,
    pub(super) applied:AtomicU64,
    host_version:AtomicU64,
    pub(super) error:Mutex<Option<String>>,
    closed:AtomicBool,
    transport_probe:bool,
    transport_probe_next:Mutex<Instant>,
    pub(super) suppress_history:AtomicBool,
}
impl EditorSession {
    /// 每个真实文档唯一actor；worker与会话均用weak，不形成document→actor→document循环。
    pub(crate) fn new(document:&Arc<DocumentSession>)->Result<Arc<Self>,String> {
        super::resources::initialize_models();
        static NEXT:AtomicU64=AtomicU64::new(1);
        let namespace=format!("hfs-ui-{}-{}-",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed));
        let (queue,receiver)=mpsc::sync_channel(32);
        let (analysis_sender,analysis_updates)=mpsc::channel();
        let session=Arc::new(Self {document:Arc::downgrade(document),timeline:Mutex::new(TimelineState::default()),
            history:Mutex::new(Default::default()),project:Mutex::new(ProjectState::default()),
            settings:Mutex::new(hifishifter_kernel::config::UiSettings::default()),loaded:Mutex::new(Default::default()),
            pcm_dir:std::env::temp_dir().join("hifishifter-plugin-pcm").join(&namespace),namespace,
            peaks:Mutex::new(HashMap::new()),queue,worker:Mutex::new(None),analysis_sender,analysis_updates:Mutex::new(analysis_updates),views:Mutex::new(HashMap::new()),
            analysis_cancel:Arc::new(AtomicBool::new(false)),analysis_workers:Mutex::new(Vec::new()),
            generation:AtomicU64::new(0),submitted:AtomicU64::new(0),applied:AtomicU64::new(0),
            host_version:AtomicU64::new(0),
            error:Mutex::new(None),closed:AtomicBool::new(false),suppress_history:AtomicBool::new(false),
            transport_probe:std::env::var_os("HIFISHIFTER_ARA_TRANSPORT_PROBE").is_some(),transport_probe_next:Mutex::new(Instant::now())});
        let weak=Arc::downgrade(&session);
        let worker=std::thread::Builder::new().name("hfs-embedded-editor".into()).spawn(move ||Self::run(weak,receiver))
            .map_err(|e|format!("create editor command worker: {e}"))?;
        *session.worker.lock().unwrap()=Some(worker);
        super::events::register(&session);
        Ok(session)
    }
    /// 有界32任务FIFO；关view取消回信，真实组件入口撤销则拒绝排队请求执行。
    pub fn enqueue(&self,request:UiRequest)->Result<(),String> {
        if self.closed.load(Ordering::Acquire) {return Err("FX processor closed".into());}
        let document=self.document.upgrade().ok_or("document closed")?;
        let lease=request.link.as_ref().map(|link|link.authorize(&document)).transpose()?;
        if request.sink.view_id.is_empty() || (request.link.is_some() && request.sink.closed.load(Ordering::Acquire)) {return Err("editor view closed or unknown".into());}
        let mut views=self.views.lock().unwrap();
        if self.closed.load(Ordering::Acquire) {return Err("editor session closed".into());}
        if request.link.is_some() && views.get(&request.sink.view_id).is_some_and(|view|!Arc::ptr_eq(&view.sink.closed,&request.sink.closed)) {
            return Err("editor view identity mismatch".into());}
        let sink=request.sink.clone();let route=request.link.clone().zip(lease);
        let mutates=super::commands::mutates_audio(&request.command);
        // 与worker的检查共用views锁；只有成功入队才登记view和推进合并票据。
        self.queue.try_send(Job::Request(request,lease)).map_err(|e|format!("editor queue unavailable: {e}"))?;
        views.insert(sink.view_id.clone(),RegisteredView {sink,route});
        if mutates {self.submitted.fetch_add(1,Ordering::AcqRel);}Ok(())
    }
    /// getState的非实时屏障：先排完已收到的尾块，不需要等音频快照才持久化曲线。
    pub fn flush(&self)->Result<(),String> {
        if self.closed.load(Ordering::Acquire) {return Ok(());}
        let (sender,receiver)=mpsc::channel();
        self.queue.send(Job::Barrier(sender)).map_err(|e|e.to_string())?;
        receiver.recv_timeout(Duration::from_secs(30)).map_err(|e|format!("editor flush: {e}"))
    }
    /// 在组件stop返回前撤销其回信/订阅；排队任务仍会独立核对原始route代次。
    pub(crate) fn revoke_closed_views(&self) {
        let document=self.document.upgrade();
        self.views.lock().unwrap().retain(|_,view| {
            let valid=view.route.as_ref().is_none_or(|(link,lease)|document.as_ref()
                .is_some_and(|document|link.authorize(document).is_ok_and(|current|current==*lease)));
            if !valid {view.sink.closed.store(true,Ordering::Release);}valid
        });
    }
    pub fn close(&self) {
        if self.closed.swap(true,Ordering::AcqRel) {return;}
        {let mut views=self.views.lock().unwrap();for view in views.values() {view.sink.closed.store(true,Ordering::Release);}views.clear();}
        self.analysis_cancel.store(true,Ordering::Release);
        self.submitted.fetch_add(1,Ordering::AcqRel);
        let _=self.queue.send(Job::Close);
        let worker=self.worker.lock().unwrap().take();
        if let Some(worker)=worker {if worker.thread().id()!=std::thread::current().id() {let _=worker.join();}}
        for worker in self.analysis_workers.lock().unwrap().drain(..) {let _=worker.join();}
        // 旧view可继续持有已关闭actor的Arc；join之后释放分析投影/历史/波形缓存。
        *self.timeline.lock().unwrap()=Default::default();
        *self.history.lock().unwrap()=Default::default();
        *self.loaded.lock().unwrap()=Default::default();
        self.peaks.lock().unwrap().clear();
    }
    fn run(weak:Weak<Self>,receiver:mpsc::Receiver<Job>) {
        let mut deadline:Option<Instant>=None;
        loop {
            if let Some(session)=weak.upgrade() {
                if session.closed.load(Ordering::Acquire) {break;}
                session.refresh_host();
                if session.refresh_analysis() {deadline=Some(Instant::now()+Duration::from_millis(150));}
            } else {break;}
            let wait=deadline.map(|d|d.saturating_duration_since(Instant::now())).unwrap_or(Duration::from_millis(100));
            match receiver.recv_timeout(wait) {
                Ok(Job::Close)=>break,
                Ok(Job::Barrier(sender))=>{let _=sender.send(());},
                Ok(Job::Request(request,lease))=>{
                    let Some(session)=weak.upgrade() else {break;};
                    if session.closed.load(Ordering::Acquire) {break;}
                    let before=session.generation.load(Ordering::Acquire);
                    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if let Some(link)=&request.link {
                            let document=session.document.upgrade().ok_or("document closed")?;
                            if Some(link.authorize(&document)?)!=lease {return Err("editor route changed while request queued".into());}
                            let views=session.views.lock().unwrap();
                            // view关闭仅取消回信；已入队尾笔仍持enqueue验证的身份与route代次。
                            // emit可能已移除closed view，但同名新view不能替代这次原始请求。
                            match views.get(&request.sink.view_id) {
                                Some(view) if !Arc::ptr_eq(&view.sink.closed,&request.sink.closed)=>return Err("editor view identity mismatch".into()),
                                None if !request.sink.closed.load(Ordering::Acquire)=>return Err("editor view closed or unknown".into()),
                                _=>{},
                            }
                        }
                        super::commands::dispatch(&session,&request.command,request.args)
                    }))
                        .unwrap_or_else(|_|Err("editor command panicked".into()));
                    session.schedule_analysis();
                    if session.generation.load(Ordering::Acquire)!=before {deadline=Some(Instant::now()+Duration::from_millis(150));}
                    let response=match result {
                        Ok(value)=>json!({"version":1,"viewId":request.sink.view_id,"id":request.id,"ok":true,"value":value}),
                        Err(error)=>json!({"version":1,"viewId":request.sink.view_id,"id":request.id,"ok":false,"error":error}),
                    };
                    request.sink.response(response);
                    session.emit_state();
                },
                Err(mpsc::RecvTimeoutError::Disconnected)=>break,
                Err(mpsc::RecvTimeoutError::Timeout)=>{
                    let Some(session)=weak.upgrade() else {break;};
                    if session.closed.load(Ordering::Acquire) {break;}
                    if deadline.is_some_and(|d|d<=Instant::now()) {
                        deadline=None;
                        let ticket=session.submitted.load(Ordering::Acquire);
                        let generation=session.generation.load(Ordering::Acquire);
                        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||session.apply(ticket)))
                            .unwrap_or_else(|_|Err("automatic render panicked".into()));
                        match result {
                            Ok(())=>{session.applied.store(generation,Ordering::Release);*session.error.lock().unwrap()=None;},
                            Err(error) if error=="pitch analysis pending"=>{deadline=Some(Instant::now()+Duration::from_millis(150));},
                            Err(error) if error=="automatic apply superseded"=>{deadline=Some(Instant::now()+Duration::from_millis(150));},
                            Err(error)=>{*session.error.lock().unwrap()=Some(error);},
                        }
                        session.emit_state();
                    }
                },
            }
        }
    }
    fn apply(&self,ticket:u64)->Result<(),String> {
        if let Some(error)=self.error.lock().unwrap().clone() {return Err(error);}
        // 全零占位原线并非已完成的清音分析；收敛前保留旧快照及dirty代次。
        {
            let timeline=self.timeline.lock().unwrap();
            if timeline.params_by_root_track.iter().any(|(root,params)| {
                params.pitch_edit_user_modified && params.pitch_edit.iter().any(|p|*p>0.) && params.pitch_orig_key.is_none()
                    && timeline.tracks.iter().any(|t|t.id==*root && !matches!(t.pitch_analysis_algo,PitchAnalysisAlgo::None))
            }) {return Err("pitch analysis pending".into());}
        }
        let (edit,model,projection)={let loaded=self.loaded.lock().unwrap();(loaded.edit,loaded.model,loaded.projection.clone())};
        self.document.upgrade().ok_or("document closed")?.apply_workspace_edits(edit,model,
            &projection,
            self.analysis_cancel.clone(),
            ||!self.closed.load(Ordering::Acquire) && self.submitted.load(Ordering::Acquire)==ticket)
    }
    pub(super) fn ensure_loaded(&self,force:bool)->Result<(),String> {
        let document=self.document.upgrade().ok_or("document closed")?;
        let versions=document.editor_versions()?;
        {
            let mut loaded=self.loaded.lock().unwrap();
            if loaded.initialized && (loaded.edit,loaded.model,loaded.scope)==versions && !force {return Ok(());}
            if loaded.initialized && loaded.model==versions.1 && loaded.scope==versions.2 && !force && document.workspace_projection()?==loaded.projection {
                loaded.edit=versions.0;return Ok(());
            }
            if loaded.initialized && !force && self.generation.load(Ordering::Acquire)!=self.applied.load(Ordering::Acquire) {
                return Err("Conflict: host changed; local curves preserved, reload explicitly".into());
            }
        }
        let (snapshot,scope,projection)=document.workspace_snapshot()?;
        if !snapshot.ok {return Err(snapshot.error.unwrap_or_else(||"host snapshot unavailable".into()));}
        let mut timeline:TimelineState=serde_json::from_value(snapshot.timeline.ok_or("host timeline missing")?).map_err(|e|e.to_string())?;
        // 快照只序列化take权威；先重建扁平投影，再判断真实的倒放/组合倍率。
        for clip in &mut timeline.clips {clip.normalize_takes();}
        // 正向倍率交给原kernel保调处理，不能再把全部拉伸挡在GUI外；倒放仍按用户范围拒绝。
        if timeline.clips.iter().any(|clip|clip.reversed) {
            let error="Unsupported ARA geometry: reverse is not supported".to_owned();
            *self.error.lock().unwrap()=Some(error.clone());return Err(error);
        }
        if timeline.target_param_frames(timeline.frame_period_ms())>1_000_000 {return Err("ARA editor parameter frame budget exceeded for project span".into());}
        let views:Vec<_>=snapshot.sources.iter().map(|pcm|PcmView {persistent_id:&pcm.persistent_id,sample_rate:pcm.sample_rate,planes:&pcm.planes}).collect();
        let dir=self.pcm_dir.join(format!("m{}-e{}",snapshot.model_revision,snapshot.revision));
        let (mut timeline,reverse_paths)=materialize(timeline,&views,&dir)?;
        self.rewrite_ids(&mut timeline,true);
        if timeline.selected_track_id.is_none() {timeline.selected_track_id=timeline.tracks.first().map(|t|t.id.clone());}
        if timeline.selected_clip_id.is_none() {timeline.selected_clip_id=timeline.clips.first().map(|c|c.id.clone());}
        *self.timeline.lock().unwrap()=timeline;
        *self.history.lock().unwrap()=Default::default();
        *self.loaded.lock().unwrap()=Loaded {initialized:true,edit:snapshot.revision,model:snapshot.model_revision,scope,reverse_paths,projection};
        *self.error.lock().unwrap()=None;
        self.applied.store(self.generation.load(Ordering::Acquire),Ordering::Release);
        self.peaks.lock().unwrap().clear();
        self.schedule_analysis();
        self.host_version.fetch_add(1,Ordering::AcqRel);
        self.emit("plugin_host_changed",json!({"version":self.host_version.load(Ordering::Acquire)}));
        Ok(())
    }
    /// 稳定模型变化在后台自动读取；pending曲线遇到真实冲突保留，不静默强制重载。
    fn refresh_host(&self) {
        let Some(document)=self.document.upgrade() else {return;};
        let (initialized,edit,model,scope)={let loaded=self.loaded.lock().unwrap();(loaded.initialized,loaded.edit,loaded.model,loaded.scope)};
        if !initialized {return;}
        if document.editor_versions().is_ok_and(|versions|versions!=(edit,model,scope)) {
            if let Err(error)=self.ensure_loaded(false) {
                if error.starts_with("Conflict") || error.starts_with("Unsupported") {
                    let changed={let mut current=self.error.lock().unwrap();let changed=current.as_deref()!=Some(error.as_str());*current=Some(error);changed};
                    if changed {self.emit_state();}
                }
            }
        }
        if let Some(tempo)=document.clock.tempo() {
            let changed={let mut timeline=self.timeline.lock().unwrap();let changed=(timeline.bpm-tempo).abs()>1e-6;timeline.bpm=tempo;changed};
            if changed {self.host_version.fetch_add(1,Ordering::AcqRel);self.emit("plugin_host_changed",json!({"version":self.host_version.load(Ordering::Acquire)}));}
        }
    }
    /// 波形入口只接受本会话从宿主PCM生成的路径，不能让JS任意读取本机文件。
    pub(super) fn check_source(&self,path:&str)->Result<(),String> {
        if self.loaded.lock().unwrap().reverse_paths.contains_key(path) {Ok(())} else {Err("source is not authorized by this ARA session".into())}
    }
    /// 手绘pitch在compose关闭时仍需原线；只在分析快照中打开门禁，不改宿主轨道。
    fn analysis_timeline(&self)->TimelineState {
        let mut timeline=self.timeline.lock().unwrap().clone();
        for track in &mut timeline.tracks {
            if timeline.params_by_root_track.get(&track.id).is_some_and(|p|p.pitch_edit_user_modified) {track.compose_enabled=true;}
        }
        timeline
    }
    /// 原分析实现可取消/join；每会话持有自己的worker，不推进全局generation。
    fn schedule_analysis(&self) {
        if self.closed.load(Ordering::Acquire) || !self.loaded.lock().unwrap().initialized {return;}
        let mut workers=self.analysis_workers.lock().unwrap();
        let mut pending=Vec::new();
        for worker in workers.drain(..) {if worker.is_finished() {let _=worker.join();} else {pending.push(worker);}}
        pending.extend(hifishifter_kernel::pitch_clip::schedule_clip_pitch_jobs_scoped(
            &self.analysis_timeline(),&self.analysis_sender,self.analysis_cancel.clone()));
        *workers=pending;
    }
    /// 原设备worker负责的ClipPitchReady现在由本实例actor接收，不创建cpal设备。
    fn refresh_analysis(&self)->bool {
        let updates=self.analysis_updates.lock().unwrap().try_iter().collect::<Vec<_>>();
        let mut roots=std::collections::BTreeSet::new();
        {
            let mut timeline=self.timeline.lock().unwrap();
            for update in updates {
                if let hifishifter_kernel::engine_command::EngineCommand::ClipPitchReady {clip_id}=update {
                    if let Some(clip)=timeline.clips.iter().find(|c|c.id==clip_id) {
                        if let Some(root)=timeline.resolve_root_track_id(&clip.track_id) {roots.insert(root);}
                    }
                }
            }
            for root in &roots {if let Some(params)=timeline.params_by_root_track.get_mut(root) {params.pitch_orig_key=None;params.dyn_orig_key=None;}}
        }
        if roots.is_empty() {return false;}
        let analysis=Mutex::new(self.analysis_timeline());
        for root in &roots {
            hifishifter_kernel::pitch_analysis::maybe_schedule_pitch_orig(&analysis,root);
            hifishifter_kernel::pitch_analysis::maybe_schedule_dyn_orig(&analysis,root);
        }
        let params=analysis.into_inner().unwrap().params_by_root_track;
        self.timeline.lock().unwrap().params_by_root_track=params;
        if self.generation.load(Ordering::Acquire)>0 {
            let timeline=self.timeline.lock().unwrap().clone();
            self.publish_timeline(timeline);return true;
        }
        false
    }
    /// GUI读取真实宿主时钟，不驱动独立设备；未收到process上下文时保留本地查看游标。
    pub(super) fn transport(&self)->(f64,bool) {
        self.document.upgrade().filter(|document|document.is_alive()).map(|document|document.clock.read())
            .unwrap_or_else(||(self.timeline.lock().unwrap().playhead_sec,false))
    }
    /// 一次性时钟探针最多每秒写一条；只在命令actor线程调用，不在音频callback写文件。
    pub(super) fn report_transport_probe(&self) {
        if !self.transport_probe {return;}
        let now=Instant::now();let mut next=self.transport_probe_next.lock().unwrap();
        if now<*next {return;}*next=now+Duration::from_secs(1);drop(next);
        if let Some(clock)=self.document.upgrade().map(|document|document.clock.clone()) {
            log::info!("[ara] transport diagnostic {}",clock.diagnostics());
        }
    }
    fn rewrite_ids(&self,timeline:&mut TimelineState,to_ui:bool) {
        let id=|value:&str|if to_ui {format!("{}{value}",self.namespace)} else {value.strip_prefix(&self.namespace).unwrap_or(value).to_owned()};
        for track in &mut timeline.tracks {track.id=id(&track.id);track.parent_id=track.parent_id.as_deref().map(id);}
        for clip in &mut timeline.clips {clip.id=id(&clip.id);clip.track_id=id(&clip.track_id);}
        timeline.selected_track_id=timeline.selected_track_id.as_deref().map(id);
        timeline.selected_clip_id=timeline.selected_clip_id.as_deref().map(id);
        timeline.params_by_root_track=std::mem::take(&mut timeline.params_by_root_track).into_iter().map(|(key,value)|(id(&key),value)).collect();
    }
    /// 只将已授权私有分析路径还原到ARA身份，未知路径明确失败。
    fn native_timeline(&self,mut timeline:TimelineState)->Result<TimelineState,String> {
        self.rewrite_ids(&mut timeline,false);
        let loaded=self.loaded.lock().unwrap();
        for clip in &mut timeline.clips {
            for path in std::iter::once(&mut clip.source_path).chain(clip.takes.iter_mut().map(|t|&mut t.source_path)) {
                let local=path.as_ref().ok_or("clip analysis source missing")?;
                *path=Some(loaded.reverse_paths.get(local).ok_or("unknown analysis path")?.clone());
            }
            clip.normalize_takes();
        }
        Ok(timeline)
    }
    /// 选region后沿当前真实doc取只读投影，防止重叠区仍显示前一素材的曲线。
    pub(super) fn select_source_projection(&self)->Result<(),String> {
        if self.error.lock().unwrap().as_ref().is_some_and(|error|error.starts_with("Conflict")) {return Ok(());}
        let document=self.document.upgrade().ok_or("document closed")?;
        let current=self.native_timeline(self.timeline.lock().unwrap().clone())?;
        let roots=document.selected_source_parameters(&current)?;
        let mut timeline=self.timeline.lock().unwrap();
        timeline.params_by_root_track.extend(roots.into_iter().map(|(root,params)|(format!("{}{root}",self.namespace),params)));
        Ok(())
    }
    pub(super) fn emit(&self,event:&str,payload:Value) {
        let document=self.document.upgrade();
        let views={let mut views=self.views.lock().unwrap();views.retain(|_,view|!view.sink.closed.load(Ordering::Acquire)
            && view.route.as_ref().is_none_or(|(link,lease)|document.as_ref().is_some_and(|document|link.authorize(document).is_ok_and(|current|current==*lease))));
            views.values().map(|view|view.sink.clone()).collect::<Vec<_>>()};
        for sink in views {sink.event(json!({"version":1,"viewId":sink.view_id,"event":event,"payload":payload}));}
    }
    /// 复用原GUI已订阅的刷新事件；共享选轨/曲线变化让所有视图重取同一个权威payload。
    pub(super) fn notify_timeline(&self) {
        self.host_version.fetch_add(1,Ordering::AcqRel);
        self.emit("plugin_host_changed",json!({"version":self.host_version.load(Ordering::Acquire)}));
    }
    pub(super) fn state(&self)->Value {
        let generation=self.generation.load(Ordering::Acquire);let applied=self.applied.load(Ordering::Acquire);
        let states=self.document.upgrade().map(|document|document.renderer_owners().into_iter().map(|owner|owner.preparation_state()).collect::<Vec<_>>()).unwrap_or_default();
        let host_pending=states.iter().any(|state|state.0);let host_error=states.into_iter().find_map(|state|state.1);
        json!({"generation":generation,"applied_generation":applied,"pending":generation!=applied||host_pending,
            "host_version":self.host_version.load(Ordering::Acquire),
            "error":self.error.lock().unwrap().clone().or(host_error),"connected":!self.closed.load(Ordering::Acquire),
            "ready":self.loaded.lock().unwrap().initialized})
    }
    fn emit_state(&self) {self.emit("plugin_apply_state",self.state());}
}
impl ParamHost for EditorSession {
    fn timeline(&self)->&Mutex<TimelineState> {&self.timeline}
    fn checkpoint_timeline(&self,timeline:&TimelineState,operation:HistoryOp) {
        if self.suppress_history.load(Ordering::Acquire) {return;}
        history::checkpoint(&mut self.history.lock().unwrap(),timeline,operation.key().into(),||None);
    }
    fn mark_dirty(&self) {self.generation.fetch_add(1,Ordering::AcqRel);}
    fn publish_timeline(&self,timeline:TimelineState) {
        let result=(|| {
            let timeline=self.native_timeline(timeline)?;
            let (edit,model,projection)={let loaded=self.loaded.lock().unwrap();(loaded.edit,loaded.model,loaded.projection.clone())};
            let versions=self.document.upgrade().ok_or("document closed")?.accept_workspace_edits(edit,model,&timeline,&projection)?;
            let mut loaded=self.loaded.lock().unwrap();loaded.edit=versions.0;loaded.model=versions.1;loaded.projection=versions.2;
            *self.error.lock().unwrap()=None;
            Ok::<_,String>(())
        })();
        if let Err(error)=result {*self.error.lock().unwrap()=Some(error);}
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::render::extension::ExtensionOwner;
    use crate::ara::model::ModelHandle;
    use crate::render::ownership::region_owners;
    use crate::render::source::SourcePcm;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::ExtensionRoles;

    /// 只替代真实DAW的建图/授权边界；使用真正owner、native扩展、actor与state编码。
    fn fixture()->(ModelHandle,Arc<ExtensionOwner>,Box<u8>) {
        let model=ModelHandle::new();let document=model.session();
        let mut timeline:TimelineState=serde_json::from_value(json!({
            "tracks":[{"id":"track","name":"host","order":0,"pitch_analysis_algo":"none"}],
            "clips":[{"id":"clip","track_id":"track","name":"source","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","source_path":"ara://source","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":1
        })).unwrap();for clip in &mut timeline.clips {clip.normalize_takes();}
        *document.timeline.lock().unwrap()=Some(timeline);
        document.track_bindings.lock().unwrap().insert("track".into(),vec![("modification".into(),"ara://source".into())]);
        document.edit_sources.lock().unwrap().insert("ara://source".into(),Arc::new(SourcePcm {
            sample_rate:44100,planes:vec![vec![0.1,0.2,0.3,0.4]],version:0,_reservation:None,
        }));
        let identity=Box::new(0_u8);let key=(&*identity as *const u8) as u64;
        region_owners().lock().unwrap().register(key,document.id,0).unwrap();
        document.clip_ids.lock().unwrap().insert(key,"clip".into());
        document.regions.lock().unwrap().insert(key,crate::ara::AraPlaybackRegion {
            audio_source_persistent_id:"ara://source".into(),audio_modification_persistent_id:"modification".into(),duration_in_modification_time:4.0/44100.0,
            duration_in_playback_time:4.0/44100.0,..Default::default()
        });
        let owner=Arc::new(ExtensionOwner::default());
        let raw=owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        // SAFETY: identity和扩展owner在所有测试调用期间存活。
        unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
        document.ready.store(true,Ordering::Release);
        (model,owner,identity)
    }
    fn call(editor:&EditorSession,sink:&UiSink,receiver:&mpsc::Receiver<Value>,id:u64,command:&str,args:Value)->Value {
        editor.enqueue(UiRequest {id,command:command.into(),args,sink:sink.clone(),link:None}).unwrap();
        let result=receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(result["id"],id);assert_eq!(result["ok"],true,"{result}");result["value"].clone()
    }
    /// 两真实组件使用同源的不同区域，测试原actor而不是拼接UI快照。
    pub(crate) fn workspace_fixture()->(ModelHandle,Vec<Arc<ExtensionOwner>>,Vec<Box<u8>>) {
        let (model,a,first)=fixture();let document=model.session();
        {let mut host=document.timeline.lock().unwrap();let host=host.as_mut().unwrap();
            let mut track=host.tracks[0].clone();track.id="b".into();track.order=1;host.tracks.push(track);
            let mut clip=host.clips[0].clone();clip.id="cb".into();clip.track_id="b".into();host.clips.push(clip);}
        document.track_bindings.lock().unwrap().insert("b".into(),vec![("modification-b".into(),"ara://source".into())]);
        let second=Box::new(0_u8);let key=(&*second as *const u8) as u64;
        region_owners().lock().unwrap().register(key,document.id,1).unwrap();
        document.clip_ids.lock().unwrap().insert(key,"cb".into());
        let mut region=document.regions.lock().unwrap().values().next().unwrap().clone();region.audio_modification_persistent_id="modification-b".into();document.regions.lock().unwrap().insert(key,region);
        let b=Arc::new(ExtensionOwner::default());let raw=b.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        // SAFETY: 模型、两owner和真实region身份由返回值保留。
        unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
        (model,vec![a,b],vec![first,second])
    }
    /// 与已有WORLD oracle相同的谐波源，双轨同源但持久modification身份不同。
    fn task34_world_fixture()->(ModelHandle,Vec<Arc<ExtensionOwner>>,Vec<Box<u8>>) {
        let (model,owners,ids)=workspace_fixture();let document=model.session();
        {let mut known=document.timeline.lock().unwrap();let timeline=known.as_mut().unwrap();timeline.project_sec=2.;
            for track in &mut timeline.tracks {track.compose_enabled=true;track.pitch_analysis_algo=PitchAnalysisAlgo::WorldDll;}
            for clip in &mut timeline.clips {clip.length_sec=2.;clip.takes[0].source_end_sec=2.;clip.normalize_takes();}}
        for region in document.regions.lock().unwrap().values_mut() {region.duration_in_modification_time=2.;region.duration_in_playback_time=2.;}
        let samples=(0..88200).map(|n| {let phase=2.*std::f64::consts::PI*220.*n as f64/44100.;
            (1..=16).map(|harmonic|(phase*harmonic as f64).sin()*0.2/harmonic as f64).sum::<f64>() as f32}).collect();
        let pcm=Arc::new(SourcePcm {sample_rate:44100,planes:vec![samples],version:0,_reservation:None});
        document.edit_sources.lock().unwrap().insert("ara://source".into(),pcm.clone());
        document.sources.lock().unwrap().insert("ara://source".into(),pcm);
        (model,owners,ids)
    }
    /// 生产route尾块在view关闭后仍先入权威；render失败也能保存，共享undo保存后无GUI恢复供音。
    #[test]
    fn task34_tail_and_shared_undo_save_restore_both_world_tracks_without_gui() {
        let (model,owners,_ids)=task34_world_fixture();let document=model.session();let editor=owners[0].editor_session().unwrap();
        let leases=owners.iter().map(|owner|super::super::routing::RouteLease::new(owner)).collect::<Vec<_>>();
        let links=leases.iter().map(|lease| {let link=Arc::new(super::super::routing::EditorLink::default());
            link.bind(std::process::id() as i64,lease.token()).unwrap();link}).collect::<Vec<_>>();
        let mut sinks=Vec::new();let mut receivers=Vec::new();
        for index in 0..2 {let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
            sinks.push(UiSink {view_id:format!("task34-world-{index}"),reply,events,closed:Arc::new(AtomicBool::new(false))});receivers.push(rx);}
        let request=|index:usize,id:u64,command:&str,args:Value| {
            editor.enqueue(UiRequest {id,command:command.into(),args,sink:sinks[index].clone(),link:Some(links[index].clone())}).unwrap();
            let response=receivers[index].recv_timeout(Duration::from_secs(10)).unwrap();assert_eq!(response["ok"],true,"{response}");response["value"].clone()
        };
        let timeline=request(0,1,"get_timeline_state",json!({}));let ta=timeline["tracks"][0]["id"].as_str().unwrap();let tb=timeline["tracks"][1]["id"].as_str().unwrap();
        let deadline=Instant::now()+Duration::from_secs(15);
        for track in [ta,tb] {loop {
            let frames=request(0,2,"get_param_frames",json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}));
            if frames["orig"].as_array().is_some_and(|orig|orig.iter().filter_map(Value::as_f64).any(|p|p>30.)) {break;}
            assert!(Instant::now()<deadline,"WORLD原线未完成: {frames}");std::thread::sleep(Duration::from_millis(20));
        }}
        request(0,3,"set_param_frames",json!({"trackId":tb,"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":false}));
        request(0,4,"set_param_frames",json!({"trackId":ta,"param":"pitch","startFrame":0,"values":vec![60.;400],"checkpoint":true}));
        request(1,5,"set_param_frames",json!({"trackId":tb,"param":"pitch","startFrame":0,"values":vec![67.;400],"checkpoint":true}));
        // 真实不支持的宿主fade使合成失败；权威的尾笔不能依赖快照发布成功。
        for region in document.regions.lock().unwrap().values_mut() {region.has_content_based_fade_at_head=true;}
        let held=editor.timeline.lock().unwrap();
        editor.enqueue(UiRequest {id:6,command:"get_timeline_state".into(),args:json!({}),sink:sinks[1].clone(),link:Some(links[1].clone())}).unwrap();
        editor.enqueue(UiRequest {id:7,command:"set_param_frames".into(),args:json!({"trackId":tb,"param":"pitch","startFrame":399,"values":[69.],"checkpoint":false}),sink:sinks[1].clone(),link:Some(links[1].clone())}).unwrap();
        sinks[1].closed.store(true,Ordering::Release);drop(held);
        let saved=[owners[0].encode_state().unwrap(),owners[1].encode_state().unwrap()];
        assert!(editor.apply(editor.submitted.load(Ordering::Acquire)).is_err(),"真实不支持fade必须拒绝合成");
        let b:Value=serde_json::from_slice(&saved[1]).unwrap();assert_eq!(b["edits"]["params"]["b"]["pitch_edit"][399],69.);
        assert_eq!(editor.history.lock().unwrap().position,2,"尾笔不新增undo步");
        assert!(editor.enqueue(UiRequest {id:8,command:"get_ui_settings".into(),args:json!({}),sink:sinks[1].clone(),link:Some(links[1].clone())}).is_err());
        request(0,9,"undo_timeline",json!({}));
        let undone=[owners[0].encode_state().unwrap(),owners[1].encode_state().unwrap()];
        for (states,b_midi) in [(saved,67.),(undone,64.)] {
            let (cold,cold_owners,_cold_ids)=task34_world_fixture();let cold_document=cold.session();
            // 不创建editor_session，不轮询GUI命令，直接驱动原setState和真实publisher供音。
            cold_document.ready.store(false,Ordering::Release);
            for (owner,bytes) in cold_owners.iter().zip(&states) {owner.restore_state(bytes).unwrap();}
            cold_document.prepare_renderers();let deadline=Instant::now()+Duration::from_secs(15);
            for (index,owner) in cold_owners.iter().enumerate() {
                while owner.preparation_state().0 {assert!(Instant::now()<deadline);std::thread::sleep(Duration::from_millis(10));}
                assert!(owner.preparation_state().1.is_none(),"{:?}",owner.preparation_state());
                let mut left=vec![0.;8820];let mut right=vec![0.;8820];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
                let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
                // SAFETY: 两个8820帧平面；oracle读取实际后台准备发布的PCM。
                assert!(unsafe {owner.snapshots[0].copy_block(22050,44100,&mut bus,8820)});
                assert!(left.iter().all(|sample|sample.is_finite()));assert_eq!(left,right);
                let correlation=|lag:usize| {let mut xy=0.;let mut xx=0.;let mut yy=0.;
                    for n in 0..left.len()-lag {let a=left[n] as f64;let b=left[n+lag] as f64;xy+=a*b;xx+=a*a;yy+=b*b;}xy/(xx*yy).sqrt().max(1e-20)};
                let target=440.*2_f64.powf(((if index==0 {60.} else {b_midi})-69.)/12.);
                let expected_lag=44100./target;let first=(expected_lag*0.8) as usize;let last=(expected_lag*1.2) as usize;
                let lag=(first..=last).max_by(|a,b|correlation(*a).total_cmp(&correlation(*b))).unwrap();let hz=44100./lag as f64;
                eprintln!("Task34 cold track={index} B_midi={b_midi} frequency={hz:.3}Hz target={target:.3}Hz");
                assert!((hz-target).abs()<8.,"cold WORLD pitch mismatch {hz} vs {target}");
                let local:Value=serde_json::from_slice(&owner.encode_state().unwrap()).unwrap();
                assert_eq!(local["edits"]["tracks"].as_array().unwrap().len(),1);
                assert_eq!(local["edits"]["params"].as_object().unwrap().len(),1);
                assert_eq!(local["edits"]["params"][if index==0 {"track"} else {"b"}]["pitch_edit"][100],if index==0 {60.} else {b_midi});
            }
            cold_document.close();
        }
        // 模型变化必须拒绝旧actor投影，不能把曲线保存误报成音频已经同步。
        document.clear_renderers();
        editor.enqueue(UiRequest {id:10,command:"get_timeline_state".into(),args:json!({}),sink:sinks[0].clone(),link:Some(links[0].clone())}).unwrap();
        let rejected=receivers[0].recv_timeout(Duration::from_secs(5)).unwrap();assert_eq!(rejected["ok"],false);
        assert!(rejected["error"].as_str().unwrap().contains("local curves preserved"));
        let local=editor.timeline.lock().unwrap();assert_eq!(local.params_by_root_track[ta].pitch_edit[100],60.);
        assert_eq!(local.params_by_root_track[tb].pitch_edit[100],64.);drop(local);
        assert_eq!(editor.state()["pending"],true);assert!(owners[0].encode_state().unwrap_err().contains("not ready"));
        document.close();
    }
    /// 最后一个renderer入口撤销后，actor屏障/空范围应用/文档join都不等待不存在的renderer。
    #[test]
    fn task34_no_renderer_does_not_retry_or_fall_back_to_the_whole_document() {
        let (model,owners,_ids)=workspace_fixture();let document=model.session();let editor=owners[0].editor_session().unwrap();
        let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"task34-no-renderer".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let timeline=call(&editor,&sink,&rx,1,"get_timeline_state",json!({}));
        call(&editor,&sink,&rx,2,"set_param_frames",json!({"trackId":timeline["tracks"][0]["id"],"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}));
        for owner in &owners {owner.stop_editor();}
        let projection=document.workspace_projection().unwrap();let workspace=document.workspace_timeline().unwrap();
        assert!(workspace.tracks.is_empty());assert!(workspace.clips.is_empty());
        let began=Instant::now();editor.flush().unwrap();
        let edit=document.edits.lock().unwrap().revision;
        document.apply_workspace_edits(edit,document.revision.load(Ordering::Acquire),&projection,Arc::new(AtomicBool::new(false)),||true).unwrap();
        let deadline=Instant::now()+Duration::from_secs(3);
        while editor.error.lock().unwrap().is_none() {assert!(Instant::now()<deadline);std::thread::sleep(Duration::from_millis(10));}
        assert!(editor.error.lock().unwrap().as_ref().unwrap().contains("local curves preserved"));
        assert_eq!(editor.state()["pending"],true,"失去全部renderer不能把保存曲线伪报成已渲染");
        assert!(!editor.closed.load(Ordering::Acquire),"组件入口都撤销也不能误关文档actor");
        document.close();assert!(began.elapsed()<Duration::from_secs(3));assert!(editor.worker.lock().unwrap().is_none());
    }
    /// 旧view仍持actor Arc时，文档close也必须在join之后释放分析投影/历史和持久权威缓存。
    #[test]
    fn task34_document_close_releases_state_even_when_the_closed_actor_is_retained() {
        let (model,owners,_ids)=workspace_fixture();let document=model.session();let editor=owners[0].editor_session().unwrap();
        let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"task34-retained-actor".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let timeline=call(&editor,&sink,&rx,1,"get_timeline_state",json!({}));
        call(&editor,&sink,&rx,2,"set_param_frames",json!({"trackId":timeline["tracks"][0]["id"],"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}));
        let authority=document.edits.clone();document.close();
        assert!(editor.worker.lock().unwrap().is_none());assert!(editor.analysis_workers.lock().unwrap().is_empty());
        assert!(editor.timeline.lock().unwrap().clips.is_empty(),"join后旧actor不得继续持有分析投影");
        assert!(editor.history.lock().unwrap().records.is_empty());assert!(editor.peaks.lock().unwrap().is_empty());
        assert!(!editor.loaded.lock().unwrap().initialized);assert!(document.track_bindings.lock().unwrap().is_empty());
        assert!(authority.lock().unwrap().params.is_empty());assert!(authority.lock().unwrap().tracks.is_empty());
    }
    #[test]
    fn task34_component_stop_revokes_its_views_and_output_before_returning() {
        let (model,owners,_ids)=workspace_fixture();let document=model.session();
        let editor=owners[0].editor_session().unwrap();
        let leases=owners.iter().map(|owner|super::super::routing::RouteLease::new(owner)).collect::<Vec<_>>();
        let mut sinks=Vec::new();let mut receivers=Vec::new();let mut links=Vec::new();
        for (index,lease) in leases.iter().enumerate() {
            let link=Arc::new(super::super::routing::EditorLink::default());link.bind(std::process::id() as i64,lease.token()).unwrap();
            let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(8);
            let sink=UiSink {view_id:format!("task34-stop-{index}"),reply,events,closed:Arc::new(AtomicBool::new(false))};
            editor.enqueue(UiRequest {id:1,command:"get_ui_settings".into(),args:json!({}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
            assert_eq!(rx.recv_timeout(Duration::from_secs(3)).unwrap()["ok"],true);
            owners[index].snapshots[0].publish(crate::render::snapshot::PlaybackSnapshot {
                sample_rate:44100,origin_sample:0,left:vec![0.25;4],right:vec![0.5;4],_reservation:None}).unwrap();
            sinks.push(sink);receivers.push(rx);links.push(link);
        }
        owners[0].stop_editor();
        // 失败也先join文档worker，避免把RED断言变成线程退出问题。
        let revoked=sinks[0].closed.load(Ordering::Acquire);let other_live=!sinks[1].closed.load(Ordering::Acquire);
        let output=owners.iter().map(|owner| {
            let mut left=[9.;4];let mut right=[9.;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
            let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
            // SAFETY: 两个平面各四帧；观察真实发布输出是否已经撤销。
            unsafe {owner.snapshots[0].copy_block(0,44100,&mut bus,4)}
        }).collect::<Vec<_>>();
        assert!(editor.enqueue(UiRequest {id:2,command:"get_ui_settings".into(),args:json!({}),sink:sinks[0].clone(),link:Some(links[0].clone())}).is_err());
        editor.enqueue(UiRequest {id:3,command:"get_ui_settings".into(),args:json!({}),sink:sinks[1].clone(),link:Some(links[1].clone())}).unwrap();
        assert_eq!(receivers[1].recv_timeout(Duration::from_secs(3)).unwrap()["ok"],true);
        document.close();
        assert!(revoked,"组件stop必须同步撤销自己的view");assert!(other_live,"另一组件view仍有租约");
        assert_eq!(output,[false,true],"组件stop必须撤销自己的输出，不能清除另一renderer");
    }
    /// 原actor选择重叠region只换其参数投影；不能生成编辑/history或覆盖另一素材。
    #[test]
    fn atlas_selection_commands_project_overlaps_without_audio_or_history_changes() {
        let (model,owners,_ids)=workspace_fixture();let document=model.session();
        let mut original=document.timeline.lock().unwrap().as_ref().unwrap().clone();
        for (index,track) in original.tracks.iter().enumerate() {
            original.params_by_root_track.insert(track.id.clone(),hifishifter_kernel::state::TrackParamsState {
                frame_period_ms:5.,pitch_edit:vec![if index==0 {60.} else {67.};2],..Default::default()});
        }
        let identities=document.parameter_identities_locked(&original).unwrap();
        let atlas=super::super::parameter_atlas::ParameterAtlas::default().capture(&original,&identities).unwrap();
        let root=original.tracks[0].id.clone();let ca=original.clips[0].id.clone();let cb=original.clips[1].id.clone();
        original.clips[1].track_id=root.clone();original.tracks.truncate(1);original.selected_clip_id=Some(ca.clone());
        let atlas=atlas.follow_geometry(&original,&identities).unwrap();
        original.params_by_root_track=atlas.project_roots(&original,&identities).unwrap();
        *document.timeline.lock().unwrap()=Some(original.clone());
        document.track_bindings.lock().unwrap().insert(root.clone(),vec![("modification".into(),"ara://source".into()),("modification-b".into(),"ara://source".into())]);
        {let mut edits=document.edits.lock().unwrap();edits.params=original.params_by_root_track;edits.atlas=atlas;}
        let editor=owners[0].editor_session().unwrap();
        let (reply,rx)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"atlas-select".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let loaded=call(&editor,&sink,&rx,1,"get_timeline_state",json!({}));
        let root=loaded["tracks"][0]["id"].as_str().unwrap();let ca=format!("{}{ca}",editor.namespace);let cb=format!("{}{cb}",editor.namespace);
        let before=(document.edits.lock().unwrap().revision,editor.generation.load(Ordering::Acquire));
        call(&editor,&sink,&rx,2,"select_clip",json!({"clipId":cb}));
        let b=editor.timeline.lock().unwrap().params_by_root_track[root].pitch_edit[0];
        call(&editor,&sink,&rx,3,"select_clip",json!({"clipId":ca}));
        let a=editor.timeline.lock().unwrap().params_by_root_track[root].pitch_edit[0];
        let after=(document.edits.lock().unwrap().revision,editor.generation.load(Ordering::Acquire));
        let history=editor.history.lock().unwrap().records.len();
        document.close();
        assert_eq!((a,b),(60.,67.));assert_eq!(before,after);assert_eq!(history,0);
    }
    #[test]
    fn workspace_components_share_the_original_actor_and_cross_track_history() {
        let (_model,owners,_ids)=workspace_fixture();let a=owners[0].editor_session().unwrap();let b=owners[1].editor_session().unwrap();
        assert!(Arc::ptr_eq(&a,&b),"同文档必须只有一个actor/history权威");
        let (_other,owner,_id)=fixture();assert!(!Arc::ptr_eq(&a,&owner.editor_session().unwrap()));
        let (reply,rx)=mpsc::channel();let (events,event_rx)=mpsc::sync_channel(128);
        let sa=UiSink {view_id:"workspace-a".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let (reply,rb)=mpsc::channel();let (events,eb)=mpsc::sync_channel(128);
        let sb=UiSink {view_id:"workspace-b".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let leases=owners.iter().map(|owner|super::super::routing::RouteLease::new(owner)).collect::<Vec<_>>();
        let links=leases.iter().map(|lease| {let link=Arc::new(super::super::routing::EditorLink::default());
            link.bind(std::process::id() as i64,lease.token()).unwrap();link}).collect::<Vec<_>>();
        let call=|editor:&EditorSession,sink:&UiSink,receiver:&mpsc::Receiver<Value>,id:u64,command:&str,args:Value| {
            let link=links[usize::from(sink.view_id=="workspace-b")].clone();
            editor.enqueue(UiRequest {id,command:command.into(),args,sink:sink.clone(),link:Some(link)}).unwrap();
            let result=receiver.recv_timeout(Duration::from_secs(5)).unwrap();assert_eq!(result["id"],id);assert_eq!(result["ok"],true,"{result}");result["value"].clone()
        };
        let loaded=call(&a,&sa,&rx,1,"get_timeline_state",json!({}));assert_eq!(loaded["tracks"].as_array().unwrap().len(),2);
        let ta=loaded["tracks"][0]["id"].as_str().unwrap();let tb=loaded["tracks"][1]["id"].as_str().unwrap();
        call(&a,&sa,&rx,2,"set_param_frames",json!({"trackId":tb,"param":"pitch","startFrame":0,"values":[64.],"checkpoint":false}));
        call(&a,&sa,&rx,3,"select_track",json!({"trackId":tb}));
        call(&a,&sa,&rx,4,"set_param_frames",json!({"trackId":ta,"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}));
        let second=call(&b,&sb,&rb,5,"get_timeline_state",json!({}));assert_eq!(second["selected_track_id"],tb);assert_eq!(second["undo_depth"],1);
        event_rx.try_iter().for_each(drop);eb.try_iter().for_each(drop);
        call(&b,&sb,&rb,6,"set_param_frames",json!({"trackId":tb,"param":"pitch","startFrame":0,"values":[67.],"checkpoint":true}));
        assert!(event_rx.try_iter().any(|e|e["event"]=="plugin_host_changed"),"其它view需要收到原GUI时间线刷新事件");
        assert!(eb.try_iter().any(|e|e["event"]=="plugin_host_changed"));
        call(&a,&sa,&rx,7,"undo_timeline",json!({}));
        assert_eq!(a.timeline.lock().unwrap().params_by_root_track[ta].pitch_edit[0],60.);
        assert_eq!(a.timeline.lock().unwrap().params_by_root_track[tb].pitch_edit[0],64.);
        call(&b,&sb,&rb,8,"redo_timeline",json!({}));assert_eq!(a.timeline.lock().unwrap().params_by_root_track[tb].pitch_edit[0],67.);
        assert!(event_rx.try_iter().any(|e|e["event"]=="history_state"));assert!(eb.try_iter().any(|e|e["event"]=="history_state"));
        owners[0].stop_editor();assert!(!a.closed.load(Ordering::Acquire),"关闭组件不能停止共享actor");
        assert!(call(&b,&sb,&rb,9,"get_ui_settings",json!({})).is_object());
        a.close();
    }
    #[test]
    fn workspace_queued_requests_revalidate_original_routes_and_reject_unknown_views() {
        use super::super::routing::{EditorLink,RouteLease};
        let (model,owners,_ids)=workspace_fixture();let editor=owners[0].editor_session().unwrap();editor.ensure_loaded(false).unwrap();
        let ra=RouteLease::new(&owners[0]);let rb=RouteLease::new(&owners[1]);
        let link=Arc::new(EditorLink::default());link.bind(std::process::id() as i64,ra.token()).unwrap();
        let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"workspace-route".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        // 真实命令在timeline锁上停住，随后命令一定排队；clear/bind不需持document事务。
        let held=editor.timeline.lock().unwrap();
        editor.enqueue(UiRequest {id:1,command:"get_timeline_state".into(),args:json!({}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
        let track=held.tracks[0].id.clone();
        editor.enqueue(UiRequest {id:2,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
        link.clear();link.bind(std::process::id() as i64,rb.token()).unwrap();drop(held);
        let mut results=vec![receiver.recv_timeout(Duration::from_secs(5)).unwrap(),receiver.recv_timeout(Duration::from_secs(5)).unwrap()];
        results.sort_by_key(|r|r["id"].as_u64().unwrap());assert_eq!(results[1]["ok"],false,"排队入口不能重新绑定别的组件继续写入");
        assert_eq!(model.session().edits.lock().unwrap().revision,0);
        drop(rb);
        assert!(editor.enqueue(UiRequest {id:3,command:"get_ui_settings".into(),args:json!({}),sink:sink.clone(),link:Some(link.clone())}).is_err(),"过期route拒绝入队");
        let unknown=Arc::new(EditorLink::default());
        assert!(editor.enqueue(UiRequest {id:4,command:"get_ui_settings".into(),args:json!({}),sink:sink.clone(),link:Some(unknown)}).is_err());
        link.bind(std::process::id() as i64,ra.token()).unwrap();
        editor.enqueue(UiRequest {id:5,command:"get_ui_settings".into(),args:json!({}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(5)).unwrap()["ok"],true);
        let (reply,_)=mpsc::channel();let (events,_)=mpsc::sync_channel(8);
        let forged=UiSink {view_id:sink.view_id.clone(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        assert!(editor.enqueue(UiRequest {id:6,command:"get_ui_settings".into(),args:json!({}),sink:forged,link:Some(link.clone())}).is_err());
        sink.closed.store(true,Ordering::Release);
        assert!(editor.enqueue(UiRequest {id:7,command:"get_ui_settings".into(),args:json!({}),sink,link:Some(link)}).is_err());
        editor.close();
    }
    #[test]
    fn workspace_queued_origin_close_cannot_borrow_the_other_live_component() {
        let (_model,owners,_ids)=workspace_fixture();let editor=owners[0].editor_session().unwrap();editor.ensure_loaded(false).unwrap();
        let lease=super::super::routing::RouteLease::new(&owners[0]);let link=Arc::new(super::super::routing::EditorLink::default());
        link.bind(std::process::id() as i64,lease.token()).unwrap();
        let (reply,receiver)=mpsc::channel();let (events,event_rx)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"workspace-origin-close".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let held=editor.timeline.lock().unwrap();
        editor.enqueue(UiRequest {id:1,command:"get_timeline_state".into(),args:json!({}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
        editor.enqueue(UiRequest {id:2,command:"get_ui_settings".into(),args:json!({}),sink:sink.clone(),link:Some(link)}).unwrap();
        owners[0].stop_editor();drop(held);
        editor.flush().unwrap();
        assert!(sink.closed.load(Ordering::Acquire),"组件关闭同步撤销旧view回信");
        assert!(receiver.try_iter().next().is_none());assert!(!editor.closed.load(Ordering::Acquire));
        assert!(Arc::ptr_eq(&editor,&owners[1].editor_session().unwrap()));
        assert!(event_rx.try_iter().next().is_none(),"入口撤销后不能通过共享actor继续订阅事件");editor.close();
    }
    #[test]
    fn workspace_document_close_joins_actor_revokes_views_and_releases_model_pcm() {
        let (model,owners,_ids)=workspace_fixture();let document=model.session();let weak=Arc::downgrade(&document);
        let editor=owners[0].editor_session().unwrap();let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"workspace-lifecycle".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        call(&editor,&sink,&receiver,1,"get_timeline_state",json!({}));
        let started=Instant::now();document.close();
        assert!(started.elapsed()<Duration::from_secs(3),"document close应join而不持事务等待actor");
        assert!(editor.worker.lock().unwrap().is_none());assert!(editor.analysis_workers.lock().unwrap().is_empty());
        assert!(document.timeline.lock().unwrap().is_none());assert!(document.edit_sources.lock().unwrap().is_empty());
        assert!(sink.closed.load(Ordering::Acquire),"文档关闭撤销全部view回信/事件");
        assert!(owners[1].editor_session().is_err());assert!(document.workspace_projection().is_err());
        drop(document);drop(model);assert!(weak.upgrade().is_none(),"actor不能强持document");
    }
    #[test]
    fn workspace_scope_and_geometry_checks_reject_partial_or_stale_transactions() {
        let (model,owners,_ids)=workspace_fixture();let document=model.session();let host=document.workspace_timeline().unwrap();
        let previous=document.workspace_projection().unwrap();let version=document.revision.load(Ordering::Acquire);
        for changed in 0..5 {
            let mut client=host.clone();match changed {
                0=>{client.tracks.pop();},1=>client.tracks[1]=client.tracks[0].clone(),
                2=>{client.clips.pop();},3=>client.clips[0].takes[0].source_path=Some("C:/private.wav".into()),
                _=>client.clips[0].takes[0].playback_rate=2.,
            }
            assert!(document.accept_workspace_edits(0,version,&client,&previous).is_err());
            assert_eq!(document.edits.lock().unwrap().revision,0,"拒绝前不能部分提交");
        }
        owners[0].stop_editor();assert_eq!(document.revision.load(Ordering::Acquire),version);
        assert!(document.accept_workspace_edits(0,version,&host,&previous).unwrap_err().starts_with("Conflict"),"scope必须独立于model校验");
        assert_eq!(document.edits.lock().unwrap().revision,0);
    }
    #[test]
    fn workspace_accept_rejects_forged_geometry_before_changing_document_edits() {
        let (model,_owners,_ids)=workspace_fixture();let document=model.session();
        let projection=document.workspace_projection().unwrap();
        let mut client=document.workspace_timeline().unwrap();
        client.clips[0].start_sec=42.;
        assert!(document.accept_workspace_edits(0,document.revision.load(Ordering::Acquire),&client,&projection).is_err(),"宿主几何必须拒绝而不是忽略");
        assert_eq!(document.edits.lock().unwrap().revision,0);
    }
    #[test]
    fn workspace_global_solo_keeps_each_renderer_output_independent() {
        let (_model,owners,_ids)=workspace_fixture();let editor=owners[0].editor_session().unwrap();
        let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"workspace-solo".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let timeline=call(&editor,&sink,&receiver,1,"get_timeline_state",json!({}));
        let track=timeline["tracks"][0]["id"].as_str().unwrap();
        call(&editor,&sink,&receiver,2,"set_track_state",json!({"trackId":track,"solo":true,"volume":0.5}));
        let deadline=Instant::now()+Duration::from_secs(5);
        while editor.applied.load(Ordering::Acquire)<1 {assert!(Instant::now()<deadline,"{:?}",editor.error.lock().unwrap());std::thread::sleep(Duration::from_millis(5));}
        let outputs=owners.iter().map(|owner| {
            let mut left=[9.;4];let mut right=[9.;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
            let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
            // SAFETY: 每个publisher仅写本次提供的两个四帧平面。
            assert!(unsafe {owner.snapshots[0].copy_block(0,44100,&mut bus,4)});left
        }).collect::<Vec<_>>();
        assert_eq!(outputs[0],[0.05,0.1,0.15,0.2]);assert_eq!(outputs[1],[0.;4],"其它轨solo不能在本renderer投影后丢失");
        editor.close();
    }
    #[test]
    fn actor_accepts_original_curve_and_save_flush_keeps_tail_after_ui_closed() {
        let (_model,owner,_identity)=fixture();let editor=owner.editor_session().unwrap();
        let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"test-view".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let timeline=call(&editor,&sink,&receiver,1,"get_timeline_state",json!({}));
        let track=timeline["tracks"][0]["id"].as_str().unwrap();
        assert!(track.starts_with(&editor.namespace));
        assert_ne!(timeline["clips"][0]["source_path"],"ara://source");
        call(&editor,&sink,&receiver,2,"set_param_frames",json!({"trackId":track,"param":"pitch","startFrame":0,"values":[60.0],"checkpoint":true}));
        sink.closed.store(true,Ordering::Release);
        editor.enqueue(UiRequest {id:3,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":1,"values":[64.0],"checkpoint":false}),sink,link:None}).unwrap();
        let encoded=owner.encode_state().unwrap(); // 真getState路径会flush，不只测手工barrier。
        let saved:Value=serde_json::from_slice(&encoded).unwrap();
        assert_eq!(saved["version"],3,"新源basis使用v3，旧v2解码/范围回归另行保留");
        assert_eq!(&saved["edits"]["params"]["track"]["pitch_edit"].as_array().unwrap()[..2],&[json!(60.),json!(64.)]);
        assert_eq!(editor.history.lock().unwrap().position,1,"尾块不得增加undo步");
        editor.close();
    }
    #[test]
    fn invalid_track_patch_cannot_partially_mutate_volume_or_create_history() {
        let (_model,owner,_identity)=fixture();let editor=owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let track=editor.timeline.lock().unwrap().tracks[0].id.clone();
        let result=super::super::commands::dispatch(&editor,"set_track_state",json!({"trackId":track,"volume":0.5,"pitchAnalysisAlgo":"not-an-algorithm"}));
        assert!(result.is_err());assert_eq!(editor.timeline.lock().unwrap().tracks[0].volume,1.);
        assert!(editor.history.lock().unwrap().records.is_empty());
        assert_eq!(editor.generation.load(Ordering::Acquire),0);
        assert!(editor.check_source("C:/Users/user/private.wav").is_err());
        editor.close();
    }
    /// 原前端以base+position消费；停播seek也必须得到宿主绝对位置。
    #[test]
    fn host_cursor_is_absolute_once_and_tracks_stopped_seeks() {
        let (_model,owner,_identity)=fixture();let editor=owner.editor_session().unwrap();
        // 时钟投影与分析无关；标记夹具已加载，避免为四帧源启动异步ONNX预热。
        editor.loaded.lock().unwrap().initialized=true;
        owner.clock.get().unwrap().update(&crate::audio_abi::ProcessContext {
            state:0,sample_rate:44100.,project_time_samples:88200,..Default::default()
        });
        let playback=super::super::commands::dispatch(&editor,"get_playback_state",json!({})).unwrap();
        let timeline=super::super::commands::dispatch(&editor,"get_timeline_state",json!({})).unwrap();
        editor.close();
        assert_eq!(playback["base_sec"].as_f64().unwrap()+playback["position_sec"].as_f64().unwrap(),2.);
        assert_eq!(timeline["playhead_sec"],2.);
    }
    /// 宿主tempo/geometry通知必须在无人点击重新载入时进入原GUI。
    #[test]
    fn host_changes_and_bpm_sync_automatically_without_reloading() {
        let (model,owner,_identity)=fixture();let editor=owner.editor_session().unwrap();
        let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"host-sync".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        call(&editor,&sink,&receiver,1,"get_timeline_state",json!({}));
        owner.clock.get().unwrap().update(&crate::audio_abi::ProcessContext {
            state:1<<10,sample_rate:44100.,tempo:150.,..Default::default()
        });
        let document=model.session();
        document.timeline.lock().unwrap().as_mut().unwrap().clips[0].start_sec=2.;
        document.revision.fetch_add(1,Ordering::AcqRel);
        let deadline=Instant::now()+Duration::from_secs(5);
        loop {
            let synced=editor.timeline.lock().unwrap().clone();
            if synced.clips[0].start_sec==2. && synced.bpm==150. {break;}
            assert!(Instant::now()<deadline,"宿主改动没有自动同步: start={} bpm={}",synced.clips[0].start_sec,synced.bpm);
            std::thread::sleep(Duration::from_millis(20));
        }
        editor.close();
    }
    /// 首次遇到用户明确暂不做的倒放必须显示原因，不能只永远显示等待音频。
    #[test]
    fn unsupported_first_load_surfaces_geometry_error() {
        let (model,owner,_identity)=fixture();
        let document=model.session();
        let mut timeline=document.timeline.lock().unwrap();
        let clip=&mut timeline.as_mut().unwrap().clips[0];
        clip.reversed=true;clip.takes[0].reversed=true;drop(timeline);
        let editor=owner.editor_session().unwrap();
        assert!(editor.ensure_loaded(false).unwrap_err().starts_with("Unsupported"));
        owner.clock.get().unwrap().update(&crate::audio_abi::ProcessContext {
            state:1<<1,sample_rate:44100.,project_time_samples:44100,..Default::default()
        });
        let playing=super::super::commands::dispatch(&editor,"get_playback_state",json!({})).unwrap();
        assert_eq!(playing["is_playing"],true);assert_eq!(playing["position_sec"],1.);
        owner.clock.get().unwrap().stopped();
        assert_eq!(super::super::commands::dispatch(&editor,"get_playback_state",json!({})).unwrap()["is_playing"],false);
        let state=editor.state();editor.close();
        assert!(state["error"].as_str().unwrap().starts_with("Unsupported"));
        assert_eq!(state["ready"],false);
    }
    /// 真实宿主正向倍率不再被GUI挡住；take的源窗口和项目时长保留各自坐标。
    #[test]
    fn forward_time_stretch_loads_the_real_host_geometry_in_original_gui_state() {
        let (model,owner,_identity)=fixture();let document=model.session();
        {let mut timeline=document.timeline.lock().unwrap();let clip=&mut timeline.as_mut().unwrap().clips[0];
            clip.length_sec=8.0/44100.0;clip.takes[0].playback_rate=0.5;clip.normalize_takes();}
        {let mut regions=document.regions.lock().unwrap();for region in regions.values_mut() {
            region.duration_in_playback_time=8.0/44100.0;region.is_timestretch_enabled=true;}}
        let editor=owner.editor_session().unwrap();editor.ensure_loaded(false).unwrap();
        let timeline=editor.timeline.lock().unwrap();assert_eq!(timeline.clips[0].playback_rate,0.5);
        assert_eq!(timeline.clips[0].length_sec,8.0/44100.0);drop(timeline);editor.close();
    }
    /// 真actor的自动调度消费快速写入/undo/redo，不调用手工render函数冒充自动应用。
    #[test]
    fn automatic_audio_uses_latest_edit_and_follows_undo_redo_without_ui() {
        let (_model,owner,_identity)=fixture();let editor=owner.editor_session().unwrap();
        let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"audio-view".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let timeline=call(&editor,&sink,&receiver,1,"get_timeline_state",json!({}));
        let track=timeline["tracks"][0]["id"].as_str().unwrap();
        call(&editor,&sink,&receiver,2,"set_track_state",json!({"trackId":track,"volume":0.25}));
        call(&editor,&sink,&receiver,3,"set_track_state",json!({"trackId":track,"volume":0.5}));
        let wait=|expected:u64| {
            let deadline=Instant::now()+Duration::from_secs(5);
            while editor.applied.load(Ordering::Acquire)<expected {
                assert!(Instant::now()<deadline,"automatic apply failed: {:?}",editor.error.lock().unwrap());
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let output=|| {
            let mut left=[9_f32;4];let mut right=[9_f32;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
            let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
            // SAFETY: 两个四帧可写平面完整覆盖copy_block，owner与snapshot仍存活。
            assert!(unsafe {owner.snapshots[0].copy_block(0,44100,&mut bus,4)});
            (left,right)
        };
        wait(2);
        for plane in [output().0,output().1] {for (actual,want) in plane.into_iter().zip([0.05,0.1,0.15,0.2]) {assert!((actual-want).abs()<1e-6);}}
        call(&editor,&sink,&receiver,4,"undo_timeline",json!({}));wait(3);
        for (actual,want) in output().0.into_iter().zip([0.025,0.05,0.075,0.1]) {assert!((actual-want).abs()<1e-6);}
        call(&editor,&sink,&receiver,5,"redo_timeline",json!({}));wait(4);
        for (actual,want) in output().0.into_iter().zip([0.05,0.1,0.15,0.2]) {assert!((actual-want).abs()<1e-6);}
        sink.closed.store(true,Ordering::Release);
        editor.enqueue(UiRequest {id:6,command:"set_track_state".into(),args:json!({"trackId":track,"muted":true}),sink,link:None}).unwrap();
        wait(5);assert_eq!(output().0,[0.;4],"关UI取消响应，不能取消已排参数或后台供音");
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
    fn world_pitch_case(wait_for_analysis:bool) {
        let (model,owner,_identity)=fixture();let document=model.session();
        {
            let mut known=document.timeline.lock().unwrap();let timeline=known.as_mut().unwrap();
            timeline.project_sec=2.;timeline.tracks[0].compose_enabled=true;
            timeline.tracks[0].pitch_analysis_algo=PitchAnalysisAlgo::WorldDll;
            timeline.clips[0].length_sec=2.;timeline.clips[0].takes[0].source_end_sec=2.;timeline.clips[0].normalize_takes();
        }
        for region in document.regions.lock().unwrap().values_mut() {region.duration_in_modification_time=2.;region.duration_in_playback_time=2.;}
        let samples:Vec<f32>=(0..88200).map(|n|{
            let phase=2.*std::f64::consts::PI*220.*n as f64/44100.;
            // 与既有WORLD oracle相同的有声谐波结构；三谐波被Harvest判为清音。
            (1..=16).map(|harmonic|(phase*harmonic as f64).sin()*0.2/harmonic as f64).sum::<f64>() as f32
        }).collect();
        document.edit_sources.lock().unwrap().insert("ara://source".into(),Arc::new(SourcePcm {
            sample_rate:44100,planes:vec![samples],version:0,_reservation:None,
        }));
        let editor=owner.editor_session().unwrap();
        let (reply,receiver)=mpsc::channel();let (events,_)=mpsc::sync_channel(128);
        let sink=UiSink {view_id:"world-view".into(),reply,events,closed:Arc::new(AtomicBool::new(false))};
        let timeline=call(&editor,&sink,&receiver,1,"get_timeline_state",json!({}));
        assert_eq!(timeline["clips"].as_array().unwrap().len(),1);
        {
            let live=editor.timeline.lock().unwrap();
            assert!(live.tracks[0].compose_enabled);
            assert!(!live.clips[0].muted);
            assert_eq!(live.clips[0].track_id,live.tracks[0].id);
            assert!(std::path::Path::new(live.clips[0].source_path.as_ref().unwrap()).is_file());
        }
        let track=timeline["tracks"][0]["id"].as_str().unwrap();
        let analysis_deadline=Instant::now()+Duration::from_secs(15);
        while wait_for_analysis {
            let frames=call(&editor,&sink,&receiver,10,"get_param_frames",json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}));
            if frames["orig"].as_array().is_some_and(|orig|orig.iter().filter_map(Value::as_f64).any(|p|p>30.)) {break;}
            assert!(Instant::now()<analysis_deadline,"原GUI真实get_param_frames分析未完成: {frames}");
            std::thread::sleep(Duration::from_millis(20));
        }
        call(&editor,&sink,&receiver,2,"set_param_frames",json!({"trackId":track,"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":true}));
        {
            let accepted=owner.edit_state();let accepted=accepted.lock().unwrap();
            assert!(accepted.params["track"].pitch_edit_user_modified,"参数写入必须保留用户修改标记");
            assert!(matches!(accepted.tracks[0].pitch_analysis_algo,PitchAnalysisAlgo::WorldDll),"接受后的算法必须WORLD");
            assert_eq!(accepted.params["track"].pitch_edit[100],64.);
            if wait_for_analysis {assert!(accepted.params["track"].pitch_orig[100]>30.,"自动渲染的原线必须分析完成");}
        }
        let deadline=Instant::now()+Duration::from_secs(30);
        while editor.applied.load(Ordering::Acquire)<1 {
            assert!(Instant::now()<deadline,"WORLD automatic apply failed: {:?}",editor.error.lock().unwrap());
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut left=vec![0_f32;8820];let mut right=vec![0_f32;8820];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        // SAFETY: 两个8820帧平面存活，读取实际已经由actor发布的0.5秒起输出。
        assert!(unsafe {owner.snapshots[0].copy_block(22050,44100,&mut bus,8820)});
        let correlation=|lag:usize| {
            let mut cross=0_f64;let mut first=0_f64;let mut second=0_f64;
            for i in 0..left.len()-lag {let a=left[i] as f64;let b=left[i+lag] as f64;cross+=a*b;first+=a*a;second+=b*b;}
            cross/(first*second).sqrt().max(1e-20)
        };
        let lag=(88..=245).max_by(|a,b|correlation(*a).total_cmp(&correlation(*b))).unwrap();
        let frequency=44100./lag as f64;
        eprintln!("automatic WORLD output frequency={frequency:.3}Hz target=329.63Hz");
        assert!((frequency-329.63).abs()<8.,"MIDI64应约329.63Hz，实际{frequency}Hz；不能只改变gain");
        editor.close();
    }
    /// 缺少原声分析时不能把用户的pitch标成已应用；曲线权威仍可保存。
    #[test]
    fn pending_pitch_analysis_does_not_publish_raw_audio_as_applied() {
        let (_model,owner,_identity)=fixture();let editor=owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        {
            let mut timeline=editor.timeline.lock().unwrap();
            timeline.tracks[0].pitch_analysis_algo=PitchAnalysisAlgo::WorldDll;
            timeline.tracks[0].compose_enabled=true;
            let root=timeline.tracks[0].id.clone();timeline.ensure_params_for_root(&root);
            let params=timeline.params_by_root_track.get_mut(&root).unwrap();
            params.pitch_edit_user_modified=true;params.pitch_edit=vec![64.;200];params.pitch_orig=vec![0.;200];
            params.pitch_orig_key=None;
            editor.publish_timeline(timeline.clone());
        }
        let outcome=editor.apply(0);
        assert!(outcome.is_err(),"未分析的全零原线不能渲染原声并假报成功");
        assert!(outcome.unwrap_err().contains("analysis"));
        let bytes=owner.encode_state().unwrap();
        let saved:Value=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(saved["edits"]["params"]["track"]["pitch_edit"][0],64.);
        editor.close();
    }
}
