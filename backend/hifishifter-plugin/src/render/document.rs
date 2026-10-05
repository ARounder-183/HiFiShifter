//! 按真实 controller 身份索引文档生命周期，不从最后一个创建的文档猜测归属。

use super::extension::ExtensionOwner;
use super::ownership::DocumentId;
use ara2_bridge::companion::CompanionControllerBinding;
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::ExtensionControllerLease;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

struct RendererLease {
    lease: ExtensionControllerLease,
    _companion: Option<CompanionControllerBinding<'static>>,
    owner: Weak<ExtensionOwner>,
}

#[derive(Default)]
pub(crate) struct DocumentSession {
    controller: AtomicUsize,
    alive: AtomicBool,
    renderers: Mutex<Vec<RendererLease>>,
    pub generation: Mutex<Option<ApiGeneration>>,
    pub sequence_regions: Mutex<HashMap<u64, HashSet<u64>>>,
    pub regions: Mutex<HashMap<u64, crate::ara::AraPlaybackRegion>>,
    pub clip_ids: Mutex<HashMap<u64, String>>,
    pub sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    pub edit_sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    pub timeline: Mutex<Option<hifishifter_kernel::state::TimelineState>>,
    pub track_bindings: Mutex<crate::state_channel::TrackBindings>,
    pub revision: AtomicU64,
    pub render_epoch:AtomicU64,
    pub scope_revision:AtomicU64,
    pub ready: AtomicBool,
    pub transaction: Mutex<()>,
    pub edits: Arc<Mutex<crate::state_channel::EditState>>,
    pub id: DocumentId,
    pub clock:Arc<super::transport::TransportClock>,
    pub playback:Mutex<Option<ara2_bridge::plugin::PlaybackRequestHandle>>,
    editor:OnceLock<Result<Arc<crate::editor::session::EditorSession>,String>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    /// 真正的工厂必须在把控制器交给宿主前登记身份，空文档也必须可绑定。
    #[test]
    fn native_factory_registers_the_actual_controller_identity() {
        let factory = FactoryBuilder::new("org.hfs.identity", "org.hfs.identity.archive")
            .display("identity", "HiFiShifter", "https://example.invalid", "1")
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap();
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"empty document".as_ptr(),
        };
        // SAFETY: factory、fixture、host 和属性在控制器整个生命周期内存活。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 工厂成功返回稳定 controller instance。
        let instance = unsafe { raw.read_unaligned() };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_some());
        // SAFETY: 唯一终止回调；之后不再访问控制器。
        unsafe {
            ((*instance.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(instance.documentControllerRef)
        };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_none());
        factory.entry().uninitialize().unwrap();
    }
}

fn controllers() -> &'static Mutex<HashMap<usize, Weak<DocumentSession>>> {
    static DOCUMENTS: OnceLock<Mutex<HashMap<usize, Weak<DocumentSession>>>> = OnceLock::new();
    DOCUMENTS.get_or_init(Default::default)
}

impl DocumentSession {
    /// 文档惰性持有唯一原编辑actor；actor只弱引用文档，组件不持第二份history。
    pub(crate) fn editor_session(self:&Arc<Self>)->Result<Arc<crate::editor::session::EditorSession>,String> {
        let _transaction=self.transaction.lock().unwrap();
        if !self.is_alive() {return Err("document closed".into());}
        self.editor.get_or_init(||crate::editor::session::EditorSession::new(self)).clone()
    }
    pub(crate) fn flush_editor(&self)->Result<(),String> {
        if let Some(Ok(editor))=self.editor.get() {editor.flush()?;}Ok(())
    }
    /// 组件撤销只关闭失效入口的view，不创建actor也不停止同文档其它入口。
    pub(crate) fn revoke_editor_views(&self) {
        if let Some(Ok(editor))=self.editor.get() {editor.revoke_closed_views();}
    }
    /// edit/model/scope分开计数；assignment变更不能由相同model误判成未变。
    pub(crate) fn editor_versions(&self)->Result<(u64,u64,u64),String> {
        let _transaction=self.transaction.lock().unwrap();
        if !self.is_alive() {return Err("document closed".into());}
        Ok((self.edits.lock().unwrap().revision,self.revision.load(Ordering::Acquire),self.scope_revision.load(Ordering::Acquire)))
    }
    fn workspace_projection_locked(&self,edits:&crate::state_channel::EditState)->Result<String,String> {
        let mut timeline=self.workspace_timeline_locked()?;edits.apply(&mut timeline);
        let bytes=serde_json::to_vec(&(timeline.params_by_root_track,timeline.tracks)).map_err(|e|e.to_string())?;
        Ok(format!("{}:{}",self.scope_revision.load(Ordering::Acquire),blake3::hash(&bytes).to_hex()))
    }
    /// scope版本参与投影指纹；区域退出后即便edit/model没动也不能接受旧工作区。
    pub(crate) fn workspace_projection(&self)->Result<String,String> {
        let _transaction=self.transaction.lock().unwrap();self.workspace_projection_locked(&self.edits.lock().unwrap())
    }
    /// 短事务同时冻结授权、参数、PCM和三个版本，分析副本的文件IO由actor随后执行。
    pub(crate) fn workspace_snapshot(&self)->Result<(hifishifter_ara_ipc::Response,u64,String),String> {
        let _transaction=self.transaction.lock().unwrap();
        for owner in self.renderer_owners() {owner.merge_pending_restore(self)?;}
        let mut timeline=self.workspace_timeline_locked()?;let mut edits=self.edits.lock().unwrap();
        edits.reconcile(&self.track_bindings.lock().unwrap())?;edits.apply(&mut timeline);
        for clip in &mut timeline.clips {clip.normalize_takes();}
        let available=self.edit_sources.lock().unwrap();
        let ids=timeline.clips.iter().flat_map(|clip|std::iter::once(&clip.source_path).chain(clip.takes.iter().map(|take|&take.source_path)))
            .filter_map(Option::as_ref).collect::<std::collections::BTreeSet<_>>();
        let sources=ids.into_iter().map(|id| {
            let pcm=available.get(id).ok_or_else(||format!("host PCM unavailable: {id}"))?;
            Ok(hifishifter_ara_ipc::HostPcm {persistent_id:id.clone(),sample_rate:pcm.sample_rate,
                fingerprint:super::extension::pcm_fingerprint(pcm),planes:pcm.planes.clone()})
        }).collect::<Result<Vec<_>,String>>()?;
        let projection=self.workspace_projection_locked(&edits)?;
        Ok((hifishifter_ara_ipc::Response {ok:true,timeline:Some(serde_json::to_value(timeline).map_err(|e|e.to_string())?),sources,
            revision:edits.revision,model_revision:self.revision.load(Ordering::Acquire),..Default::default()},self.scope_revision.load(Ordering::Acquire),projection))
    }
    /// 原分析副本可补媒体元信息，除此之外clip、take及轨道结构全部必须与宿主一致。
    fn validate_workspace_geometry(host:&hifishifter_kernel::state::TimelineState,client:&hifishifter_kernel::state::TimelineState)->Result<(),String> {
        let normalize=|timeline:&hifishifter_kernel::state::TimelineState|->Result<serde_json::Value,String> {
            let mut timeline=timeline.clone();for clip in &mut timeline.clips {clip.normalize_takes();}
            let mut value=serde_json::to_value(timeline).map_err(|e|e.to_string())?;
            let metadata=["source_path_relative","duration_sec","duration_frames","source_sample_rate","source_channels",
                "source_file_fingerprint","source_file_mtime","source_file_size","waveform_preview","pitch_range"];
            if let Some(clips)=value["clips"].as_array_mut() {for clip in clips {
                for key in metadata {clip.as_object_mut().unwrap().remove(key);}
                if let Some(takes)=clip["takes"].as_array_mut() {for take in takes {for key in metadata {take.as_object_mut().unwrap().remove(key);}}}
            }}
            if let Some(tracks)=value["tracks"].as_array_mut() {for track in tracks {for key in ["volume","muted","solo","compose_enabled","pitch_analysis_algo"] {track.as_object_mut().unwrap().remove(key);}}}
            Ok(serde_json::json!({"tracks":value["tracks"],"clips":value["clips"],"project_sec":value["project_sec"]}))
        };
        if normalize(host)?!=normalize(client)? {return Err("host workspace geometry is read-only".into());}Ok(())
    }
    /// 全工作区在单事务校验scope/模型/曲线并接受；未知范围或几何不静默忽略。
    pub(crate) fn accept_workspace_edits(&self,base_edit:u64,base_model:u64,
        client:&hifishifter_kernel::state::TimelineState,previous:&str)->Result<(u64,u64,String),String> {
        let _transaction=self.transaction.lock().unwrap();
        let model=self.revision.load(Ordering::Acquire);
        if model!=base_model {return Err("Conflict: host model changed; local curves preserved".into());}
        let host=self.workspace_timeline_locked()?;
        let mut edits=self.edits.lock().unwrap();
        if self.workspace_projection_locked(&edits)?!=previous || base_edit>edits.revision {return Err("Conflict: workspace scope or curves changed; local curves preserved".into());}
        Self::validate_workspace_geometry(&host,client)?;
        let mut candidate=edits.merge(&host,client,edits.revision)?;candidate.reconcile(&self.track_bindings.lock().unwrap())?;
        let projection=self.workspace_projection_locked(&candidate)?;*edits=candidate;
        Ok((edits.revision,model,projection))
    }
    /// 合成只在短事务外计算；最终再次核对文档、scope、assignment与编辑代次。
    pub(crate) fn apply_workspace_edits(&self,base_edit:u64,base_model:u64,previous:&str,
        cancel:Arc<AtomicBool>,current:impl Fn()->bool)->Result<(),String> {
        let (edit,epoch,scope,inputs)={
            let _transaction=self.transaction.lock().unwrap();
            if !self.is_alive() || !self.ready.load(Ordering::Acquire) || self.revision.load(Ordering::Acquire)!=base_model {
                return Err("Conflict: host model changed during automatic apply".into());}
            let edits=self.edits.lock().unwrap().clone();
            if edits.revision!=base_edit || self.workspace_projection_locked(&edits)?!=previous {return Err("Conflict: workspace changed during automatic apply".into());}
            let inputs=self.renderer_owners().into_iter().filter(|owner|owner.renders_playback()).map(|owner| {
                let (keys,input)=owner.capture_render_input(self,&edits,true)?;Ok((owner,keys,input))
            }).collect::<Result<Vec<_>,String>>()?;
            (edits.revision,self.render_epoch.load(Ordering::Acquire),self.scope_revision.load(Ordering::Acquire),inputs)
        };
        if !current() {return Err("automatic apply superseded".into());}
        let mut prepared=Vec::new();for (owner,keys,input) in inputs {
            for publisher in &owner.snapshots {publisher.collect_retired();}
            prepared.push((owner,keys,input.render(cancel.clone())?));
        }
        let _transaction=self.transaction.lock().unwrap();
        if !self.is_alive() || !self.ready.load(Ordering::Acquire) || self.revision.load(Ordering::Acquire)!=base_model {
            return Err("Conflict: host model changed during automatic apply".into());}
        if !current() || self.edits.lock().unwrap().revision!=edit || self.render_epoch.load(Ordering::Acquire)!=epoch
            || self.scope_revision.load(Ordering::Acquire)!=scope {return Err("automatic apply superseded".into());}
        for (owner,keys,snapshots) in &prepared {
            if owner.is_closed() || owner.assigned_regions().map_err(|e|e.to_string())?!=*keys {return Err("automatic apply superseded".into());}
            if !owner.snapshots.iter().zip(snapshots).all(|(publisher,snapshot)|publisher.has_capacity(snapshot)) {return Err("retired snapshot budget exhausted; reopen instance".into());}
        }
        for (owner,_,snapshots) in prepared {for (publisher,snapshot) in owner.snapshots.iter().zip(snapshots) {
            publisher.publish(snapshot).map_err(|e|format!("snapshot publish failed: {e:?}"))?;
        }}Ok(())
    }
    /// 宿主专属API调用前核对真实文档仍存活，销毁后的view不能沿旧project指针查询。
    pub(crate) fn is_alive(&self)->bool {self.alive.load(Ordering::Acquire)}
    /// 会话与模型同寿，controller 地址在工厂 allocation 完成后登记。
    pub fn new(id: DocumentId) -> Arc<Self> {
        Arc::new(Self {
            id,
            alive: AtomicBool::new(true),
            ..Default::default()
        })
    }

    /// 工厂成功返回前登记真实 controllerRef，失败或销毁时撤销。
    pub fn register(self: &Arc<Self>, key: usize) {
        self.controller.store(key, Ordering::Release);
        controllers()
            .lock()
            .unwrap()
            .insert(key, Arc::downgrade(self));
    }

    /// 同步关闭文档；保持 renderer 的原生接口存储，但撤销模型操作许可。
    pub fn close(&self) {
        {let _transaction=self.transaction.lock().unwrap();
            if !self.alive.swap(false,Ordering::AcqRel) {return;}
            self.scope_revision.fetch_add(1,Ordering::AcqRel);
        }
        // 不持transaction join：actor可能正在收尾短事务；先停止全部编辑/分析再清宿主图。
        if let Some(Ok(editor))=self.editor.get() {editor.close();}
        self.render_epoch.fetch_add(1,Ordering::AcqRel);
        self.playback.lock().unwrap().take();
        self.ready.store(false, Ordering::Release);
        let leases = {
            let mut renderers = self.renderers.lock().unwrap();
            self.alive.store(false, Ordering::Release);
            std::mem::take(&mut *renderers)
        };
        let mut controllers = controllers().lock().unwrap();
        let key = self.controller.load(Ordering::Acquire);
        if controllers
            .get(&key)
            .and_then(Weak::upgrade)
            .is_some_and(|current| current.id == self.id)
        {
            controllers.remove(&key);
        }
        drop(controllers);
        for renderer in leases {
            renderer.lease.destroy();
            if let Some(owner) = renderer.owner.upgrade() {
                owner.document_closed();
            }
            // companion guard 随 renderer 释放，禁止继续借用即将销毁的 controller。
        }
        self.sequence_regions.lock().unwrap().clear();
        self.regions.lock().unwrap().clear();
        self.clip_ids.lock().unwrap().clear();
        self.timeline.lock().unwrap().take();
        self.track_bindings.lock().unwrap().clear();
        *self.edits.lock().unwrap()=Default::default();
        self.sources.lock().unwrap().clear();
        self.edit_sources.lock().unwrap().clear();
    }

    /// 控制器侧独立保留 lease，companion 先释放时 raw extension 存储仍须存活。
    pub fn attach(
        &self,
        owner: &Arc<ExtensionOwner>,
        lease: ExtensionControllerLease,
        companion: Option<CompanionControllerBinding<'static>>,
    ) -> Result<(), AraError> {
        let mut renderers = self.renderers.lock().unwrap();
        if !self.alive.load(Ordering::Acquire) {
            return Err(AraError::InvalidState("document is destroyed"));
        }
        renderers.push(RendererLease {
            lease,
            _companion: companion,
            owner: Arc::downgrade(owner),
        });
        Ok(())
    }

    /// 模型线程刷新所有仍存活 renderer，任何 source/geometry/访问变化都触发重新准备。
    pub fn prepare_renderers(&self) {
        let _transaction=self.transaction.lock().unwrap();
        self.ready.store(true, Ordering::Release);
        let owners = self
            .renderers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|lease| lease.owner.upgrade()).filter(|owner|!owner.is_closed())
            .collect::<Vec<_>>();
        // 先统一恢复全部组件权威，再允许任何后台任务捕获共享revision。
        for owner in &owners {
            if let Err(error)=owner.merge_pending_restore(self) {log::warn!("[ara] instance state unresolved: {error}");}
        }
        for owner in owners {
            owner.prepare();
        }
    }

    /// 同一ARA文档的编辑权威共享，输出仍按各renderer分配隔离。
    pub fn renderer_owners(&self) -> Vec<Arc<ExtensionOwner>> {
        self.renderers.lock().unwrap().iter().filter_map(|lease| lease.owner.upgrade()).filter(|owner|!owner.is_closed()).collect()
    }

    /// 在旧内容可能被改变前立即撤销发布；不回收实时读者可能仍持有的旧快照。
    pub fn clear_renderers(&self) {
        let _transaction = self.transaction.lock().unwrap();
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.revoke_snapshots();
    }

    /// 授权开关只撤销播放许可，不改变宿主内容/布局的乐观并发版本。
    pub fn revoke_renderers(&self) {
        let _transaction = self.transaction.lock().unwrap();
        self.revoke_snapshots();
    }

    fn revoke_snapshots(&self) {
        self.render_epoch.fetch_add(1,Ordering::AcqRel);
        self.ready.store(false, Ordering::Release);
        let owners = self
            .renderers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .collect::<Vec<_>>();
        for owner in owners {
            owner.cancel_preparation();
            owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
    }

    /// 只借出仍存活的文档，拒绝已销毁或非本插件的控制器。
    pub fn lookup(key: usize) -> Option<Arc<Self>> {
        controllers()
            .lock()
            .unwrap()
            .get(&key)
            .and_then(Weak::upgrade)
            .filter(|session| session.alive.load(Ordering::Acquire))
    }
}
