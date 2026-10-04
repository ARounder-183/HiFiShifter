//! 插件实例的原编辑命令actor：UI只排队，分析/文件/离线渲染均不在音频或UI回调执行。
use crate::render::extension::ExtensionOwner;
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
pub(crate) struct UiRequest {pub id:u64,pub command:String,pub args:Value,pub sink:UiSink}
enum Job {Request(UiRequest),Barrier(mpsc::Sender<()>),Close}
#[derive(Default)]
struct Loaded {
    initialized:bool,edit:u64,model:u64,
    reverse_paths:HashMap<String,String>,
}
pub(crate) struct EditorSession {
    owner:Weak<ExtensionOwner>,
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
    views:Mutex<HashMap<String,UiSink>>,
    pub(super) generation:AtomicU64,
    submitted:AtomicU64,
    pub(super) applied:AtomicU64,
    pub(super) error:Mutex<Option<String>>,
    closed:AtomicBool,
    pub(super) suppress_history:AtomicBool,
}
impl EditorSession {
    /// 每个组件唯一actor；worker只在收到任务时升级weak，不形成Arc自循环。
    pub fn new(owner:&Arc<ExtensionOwner>)->Result<Arc<Self>,String> {
        super::resources::initialize_models();
        static NEXT:AtomicU64=AtomicU64::new(1);
        let namespace=format!("hfs-ui-{}-{}-",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed));
        let (queue,receiver)=mpsc::sync_channel(32);
        let (analysis_sender,analysis_updates)=mpsc::channel();
        let session=Arc::new(Self {owner:Arc::downgrade(owner),timeline:Mutex::new(TimelineState::default()),
            history:Mutex::new(Default::default()),project:Mutex::new(ProjectState::default()),
            settings:Mutex::new(hifishifter_kernel::config::UiSettings::default()),loaded:Mutex::new(Default::default()),
            pcm_dir:std::env::temp_dir().join("hifishifter-plugin-pcm").join(&namespace),namespace,
            peaks:Mutex::new(HashMap::new()),queue,worker:Mutex::new(None),analysis_sender,analysis_updates:Mutex::new(analysis_updates),views:Mutex::new(HashMap::new()),
            analysis_cancel:Arc::new(AtomicBool::new(false)),analysis_workers:Mutex::new(Vec::new()),
            generation:AtomicU64::new(0),submitted:AtomicU64::new(0),applied:AtomicU64::new(0),
            error:Mutex::new(None),closed:AtomicBool::new(false),suppress_history:AtomicBool::new(false)});
        let weak=Arc::downgrade(&session);
        let worker=std::thread::Builder::new().name("hfs-embedded-editor".into()).spawn(move ||Self::run(weak,receiver))
            .map_err(|e|format!("create editor command worker: {e}"))?;
        *session.worker.lock().unwrap()=Some(worker);
        super::events::register(&session);
        Ok(session)
    }
    /// 有界32任务FIFO；已排队的曲线写入在关FX后仍接受，只有响应/事件取消。
    pub fn enqueue(&self,request:UiRequest)->Result<(),String> {
        if self.closed.load(Ordering::Acquire) {return Err("FX processor closed".into());}
        if super::commands::mutates_audio(&request.command) {self.submitted.fetch_add(1,Ordering::AcqRel);}
        self.views.lock().unwrap().insert(request.sink.view_id.clone(),request.sink.clone());
        self.queue.try_send(Job::Request(request)).map_err(|e|format!("editor queue unavailable: {e}"))
    }
    /// getState的非实时屏障：先排完已收到的尾块，不需要等音频快照才持久化曲线。
    pub fn flush(&self)->Result<(),String> {
        if self.closed.load(Ordering::Acquire) {return Ok(());}
        let (sender,receiver)=mpsc::channel();
        self.queue.send(Job::Barrier(sender)).map_err(|e|e.to_string())?;
        receiver.recv_timeout(Duration::from_secs(30)).map_err(|e|format!("editor flush: {e}"))
    }
    pub fn close(&self) {
        if self.closed.swap(true,Ordering::AcqRel) {return;}
        self.analysis_cancel.store(true,Ordering::Release);
        self.submitted.fetch_add(1,Ordering::AcqRel);
        let _=self.queue.send(Job::Close);
        let worker=self.worker.lock().unwrap().take();
        if let Some(worker)=worker {if worker.thread().id()!=std::thread::current().id() {let _=worker.join();}}
        for worker in self.analysis_workers.lock().unwrap().drain(..) {let _=worker.join();}
        self.views.lock().unwrap().clear();
    }
    fn run(weak:Weak<Self>,receiver:mpsc::Receiver<Job>) {
        let mut deadline:Option<Instant>=None;
        loop {
            if let Some(session)=weak.upgrade() {
                if session.closed.load(Ordering::Acquire) {break;}
                if session.refresh_analysis() {deadline=Some(Instant::now()+Duration::from_millis(150));}
            } else {break;}
            let wait=deadline.map(|d|d.saturating_duration_since(Instant::now())).unwrap_or(Duration::from_millis(100));
            match receiver.recv_timeout(wait) {
                Ok(Job::Close)=>break,
                Ok(Job::Barrier(sender))=>{let _=sender.send(());},
                Ok(Job::Request(request))=>{
                    let Some(session)=weak.upgrade() else {break;};
                    if session.closed.load(Ordering::Acquire) {break;}
                    let before=session.generation.load(Ordering::Acquire);
                    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||super::commands::dispatch(&session,&request.command,request.args)))
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
        let (edit,model)={let loaded=self.loaded.lock().unwrap();(loaded.edit,loaded.model)};
        self.owner.upgrade().ok_or("processor closed")?.apply_editor_edits(edit,model,
            ||!self.closed.load(Ordering::Acquire) && self.submitted.load(Ordering::Acquire)==ticket)
    }
    pub(super) fn ensure_loaded(&self,force:bool)->Result<(),String> {
        let owner=self.owner.upgrade().ok_or("processor closed")?;
        let versions=owner.editor_versions()?;
        {
            let loaded=self.loaded.lock().unwrap();
            if loaded.initialized && (loaded.edit,loaded.model)==versions && !force {return Ok(());}
            if loaded.initialized && !force && self.generation.load(Ordering::Acquire)!=self.applied.load(Ordering::Acquire) {
                return Err("Conflict: host changed; local curves preserved, reload explicitly".into());
            }
        }
        let snapshot=owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        if !snapshot.ok {return Err(snapshot.error.unwrap_or_else(||"host snapshot unavailable".into()));}
        let timeline:TimelineState=serde_json::from_value(snapshot.timeline.ok_or("host timeline missing")?).map_err(|e|e.to_string())?;
        if timeline.target_param_frames(timeline.frame_period_ms())>1_000_000 {return Err("ARA editor parameter frame budget exceeded for project span".into());}
        let views:Vec<_>=snapshot.sources.iter().map(|pcm|PcmView {persistent_id:&pcm.persistent_id,sample_rate:pcm.sample_rate,planes:&pcm.planes}).collect();
        let dir=self.pcm_dir.join(format!("m{}-e{}",snapshot.model_revision,snapshot.revision));
        let (mut timeline,reverse_paths)=materialize(timeline,&views,&dir)?;
        self.rewrite_ids(&mut timeline,true);
        if timeline.selected_track_id.is_none() {timeline.selected_track_id=timeline.tracks.first().map(|t|t.id.clone());}
        if timeline.selected_clip_id.is_none() {timeline.selected_clip_id=timeline.clips.first().map(|c|c.id.clone());}
        *self.timeline.lock().unwrap()=timeline;
        *self.history.lock().unwrap()=Default::default();
        *self.loaded.lock().unwrap()=Loaded {initialized:true,edit:snapshot.revision,model:snapshot.model_revision,reverse_paths};
        *self.error.lock().unwrap()=None;
        self.applied.store(self.generation.load(Ordering::Acquire),Ordering::Release);
        self.peaks.lock().unwrap().clear();
        self.schedule_analysis();
        Ok(())
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
        self.owner.upgrade().and_then(|o|o.clock.get().map(|clock|clock.read()))
            .unwrap_or_else(||(self.timeline.lock().unwrap().playhead_sec,false))
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
    pub(super) fn emit(&self,event:&str,payload:Value) {
        let views={let mut views=self.views.lock().unwrap();views.retain(|_,sink|!sink.closed.load(Ordering::Acquire));views.values().cloned().collect::<Vec<_>>()};
        for sink in views {sink.event(json!({"version":1,"viewId":sink.view_id,"event":event,"payload":payload}));}
    }
    pub(super) fn state(&self)->Value {
        let generation=self.generation.load(Ordering::Acquire);let applied=self.applied.load(Ordering::Acquire);
        json!({"generation":generation,"applied_generation":applied,"pending":generation!=applied,
            "error":self.error.lock().unwrap().clone(),"connected":!self.closed.load(Ordering::Acquire),
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
            let (edit,model)={let loaded=self.loaded.lock().unwrap();(loaded.edit,loaded.model)};
            let versions=self.owner.upgrade().ok_or("processor closed")?.accept_editor_edits(edit,model,&timeline)?;
            let mut loaded=self.loaded.lock().unwrap();loaded.edit=versions.0;loaded.model=versions.1;
            *self.error.lock().unwrap()=None;
            Ok::<_,String>(())
        })();
        if let Err(error)=result {*self.error.lock().unwrap()=Some(error);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            audio_source_persistent_id:"ara://source".into(),duration_in_modification_time:4.0/44100.0,
            duration_in_playback_time:4.0/44100.0,..Default::default()
        });
        let owner=Arc::new(ExtensionOwner::default());
        let raw=owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        // SAFETY: identity和扩展owner在所有测试调用期间存活。
        unsafe {let ext=&*raw;((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(ext.editorRendererRef,key as *mut _);}
        document.ready.store(true,Ordering::Release);
        (model,owner,identity)
    }
    fn call(editor:&EditorSession,sink:&UiSink,receiver:&mpsc::Receiver<Value>,id:u64,command:&str,args:Value)->Value {
        editor.enqueue(UiRequest {id,command:command.into(),args,sink:sink.clone()}).unwrap();
        let result=receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(result["id"],id);assert_eq!(result["ok"],true,"{result}");result["value"].clone()
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
        editor.enqueue(UiRequest {id:3,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":1,"values":[64.0],"checkpoint":false}),sink}).unwrap();
        let encoded=owner.encode_state().unwrap(); // 真getState路径会flush，不只测手工barrier。
        let saved:Value=serde_json::from_slice(&encoded).unwrap();
        assert_eq!(saved["version"],2);
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
        editor.enqueue(UiRequest {id:6,command:"set_track_state".into(),args:json!({"trackId":track,"muted":true}),sink}).unwrap();
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
