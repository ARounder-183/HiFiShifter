//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 全PCM内容参与身份，重开时同路径不同音频不能命中旧缓存。
pub(super) fn pcm_fingerprint(pcm: &super::source::SourcePcm) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(&pcm.sample_rate.to_le_bytes());
    hash.update(&(pcm.planes.len() as u64).to_le_bytes());
    for plane in &pcm.planes { for sample in plane { hash.update(&sample.to_le_bytes()); } }
    hash.finalize().to_hex().to_string()
}

/// 只读元数据供非实时读者消费；UI/model采集，音频线程不访问此缓存或锁owner。
#[derive(Clone)]
struct CachedHostGeometry {
    model:u64,scope:u64,change:Option<i32>,
    value:Result<crate::host::geometry::BoundHostGeometry,String>,
}
#[derive(Clone,PartialEq,Eq)]
struct PreparedVersion {model:u64,edit:u64,epoch:u64,scope:u64,keys:Vec<u64>}

/// 原生接口与只读元数据属于真实组件，缓存不跨其文档/分配/host变更复用。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<ExtensionBinding>>,
    document: Mutex<Option<std::sync::Weak<super::document::DocumentSession>>>,
    assignments: Mutex<HashMap<i32, Vec<RegionKey>>>,
    sequences: Mutex<HashMap<i32, Vec<u64>>>,
    role: AtomicI32,
    roles:AtomicI32,
    closed:std::sync::atomic::AtomicBool,
    writer_id:AtomicU64,
    reaper:Mutex<Option<Arc<crate::host::reaper::ReaperHost>>>,
    host_geometry:Mutex<Option<CachedHostGeometry>>,
    prepared:Mutex<Option<PreparedVersion>>,
    pub snapshots: [super::snapshot::SnapshotPublisher; 2],
    pub(crate) edits: Arc<Mutex<crate::state_channel::EditState>>,
    pending_restore:Mutex<Option<crate::state_channel::EditState>>,
    channel: Mutex<Option<hifishifter_ara_ipc::Server>>,
    preparation:std::sync::OnceLock<Result<super::preparation::PreparationQueue,String>>,
    prepare_owner:Mutex<Option<std::sync::Weak<ExtensionOwner>>>,
    pub(crate) clock:std::sync::OnceLock<Arc<super::transport::TransportClock>>,
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::ara_entry::HostEntry;
    use ara2_bridge::companion::vst3::ffi::{ara2_vst3_plugin_entry_bind, ARA2_VST3_OK};
    use ara2_bridge::companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    /// v3源basis按组件真实区域保存，冷恢复不能借另一轨道的atlas或沿旧session key。
    #[test]
    fn source_parameter_atlas_state_is_scoped_and_rebound_without_gui() {
        let (model,owners,_ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();
        let mut client=document.workspace_timeline().unwrap();
        for (root,note) in [("track",60.),("b",67.)] {client.params_by_root_track.insert(root.into(),hifishifter_kernel::state::TrackParamsState {
            frame_period_ms:5.,pitch_edit_user_modified:true,pitch_orig:vec![57.,57.],pitch_edit:vec![note,note],..Default::default()});}
        document.accept_workspace_edits(0,document.revision.load(Ordering::Acquire),&client,&document.workspace_projection().unwrap()).unwrap();
        let saved=owners.iter().map(|owner|owner.encode_state().unwrap()).collect::<Vec<_>>();
        let payloads=saved.iter().map(|bytes|serde_json::from_slice::<serde_json::Value>(bytes).unwrap()).collect::<Vec<_>>();document.close();
        for payload in &payloads {assert_eq!(payload["version"],3);assert_eq!(payload["edits"]["atlas"]["regions"].as_object().unwrap().len(),1,"不能保存其它组件区域");}
        let (cold,restored,_cold_ids)=crate::editor::session::tests::workspace_fixture();let cold_doc=cold.session();
        for index in 0..2 {restored[index].restore_state(&saved[index]).unwrap();}
        cold_doc.prepare_renderers();let atlas=cold_doc.edits.lock().unwrap().atlas.clone();cold_doc.close();
        assert_eq!(atlas.regions.len(),2);assert!(atlas.regions.values().all(|record|record.identity.key!=0),"JSON旧key不得冒充新会话身份");
    }

    /// 只打开editor-only入口时，也必须采集同文档隐藏playback owner的元数据。
    #[test]
    fn task38b_one_editor_refresh_collects_hidden_playback_metadata() {
        let (model,owners,ids)=crate::editor::session::tests::workspace_fixture();
        let document=model.session();let first=(&*ids[0] as *const u8) as u64;
        {let mut regions=document.regions.lock().unwrap();let region=regions.get_mut(&first).unwrap();
            region.start_in_playback_time=1.0;region.duration_in_playback_time=4.0;}
        let editor=Arc::new(ExtensionOwner::default());
        let raw=editor.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        // SAFETY: fixture保留真实region、document及extension；editor入口没有自己的take。
        unsafe {let ext=&*raw;for key in [&*ids[0] as *const u8,&*ids[1] as *const u8] {
            ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(ext.editorRendererRef,key.cast_mut().cast());
        }}
        let host=crate::host::reaper::ReaperFixture::new();
        unsafe {owners[0].bind_reaper_host(host.context());}
        host.reset();editor.refresh_reaper_transport();
        let metadata=owners[0].host_geometry_metadata();
        let entry_metadata=editor.host_geometry_metadata();
        document.close();
        let metadata=metadata.expect("单一editor入口必须让隐藏playback的只读数据就绪");
        assert_eq!(metadata.region_key,first);assert_eq!(metadata.geometry.fade_in_sec,0.2);
        assert!(entry_metadata.is_err(),"多区域editor不能冒充唯一take绑定");
    }

    /// 隐藏getter重入关闭唯一GUI入口后，不能借另一活owner继续该入口的采集批次。
    #[test]
    fn task38b_hidden_getter_revokes_closed_editor_batch_before_next_renderer() {
        let (model,owners,ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();
        let editor=Arc::new(ExtensionOwner::default());
        let raw=editor.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        unsafe {let ext=&*raw;((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(ext.editorRendererRef,(&*ids[0] as *const u8).cast_mut().cast());}
        let host=crate::host::reaper::ReaperFixture::new();unsafe {owners[0].bind_reaper_host(host.context());}
        host.reset();let weak=Arc::downgrade(&editor);
        *host.hook.borrow_mut()=Some(("D_LENGTH".into(),Box::new(move||weak.upgrade().unwrap().stop_editor())));
        editor.refresh_reaper_transport();let calls=host.calls();
        let sampled_next=owners[1].host_geometry.lock().unwrap().is_some();
        let sampled_first=owners[0].host_geometry.lock().unwrap().is_some();document.close();
        assert_eq!(calls.last().map(String::as_str),Some("D_LENGTH"));
        assert!(!sampled_first,"撤销后的第一份数据不能发布");
        assert!(!sampled_next,"入口关闭后不能继续读取另一活renderer");
    }

    /// 稳定工程只检查版本，不在每个UI tick重复读取整组markers；fade改变仍刷新。
    #[test]
    fn task38b_stable_geometry_skips_raw_fields_but_fade_change_and_model_refresh_do_not() {
        let (model,owners,ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();let owner=&owners[0];
        let first=(&*ids[0] as *const u8) as u64;
        {let mut regions=document.regions.lock().unwrap();let region=regions.get_mut(&first).unwrap();region.start_in_playback_time=1.;region.duration_in_playback_time=4.;}
        let host=crate::host::reaper::ReaperFixture::new();unsafe {owner.bind_reaper_host(host.context());}
        owner.refresh_reaper_transport();let first_metadata=owner.host_geometry_metadata().unwrap();
        host.reset();owner.refresh_reaper_transport();let same=owner.host_geometry_metadata().unwrap();let stable_calls=host.calls();
        host.set_value("D_FADEINLEN",0.75);host.reset();owner.refresh_reaper_transport();let changed=owner.host_geometry_metadata().unwrap();
        let changed_calls=host.calls();host.reset();document.revision.fetch_add(1,Ordering::AcqRel);owner.refresh_reaper_transport();let model_calls=host.calls();document.close();
        assert_eq!(same,first_metadata);assert!(!stable_calls.iter().any(|name|name=="count"||name=="D_FADEINLEN"),"稳定UI tick不能重读全量几何: {stable_calls:?}");
        assert_eq!(changed.geometry.fade_in_sec,0.75);assert!(changed_calls.iter().any(|name|name=="D_FADEINLEN"));
        assert!(model_calls.iter().any(|name|name=="D_FADEINLEN"),"相同project counter但新ARA model仍须重新核对绑定");
    }

    /// 原生getter可同步调用真实doc.close/owner.stop/assignment回调，不能持任何内部锁。
    #[test]
    fn task38a_owner_gate_rechecks_document_close_owner_scope_and_model_after_transport_state() {
        for change in ["document","owner","scope","model"] {
            let (model,owners,_ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();let owner=owners[0].clone();
            let host=crate::host::reaper::ReaperFixture::new();
            unsafe {owner.bind_reaper_host(host.context());}host.reset();
            let weak=Arc::downgrade(&owner);let doc=document.clone();
            *host.hook.borrow_mut()=Some(("state".into(),Box::new(move ||match change {
                "document"=>doc.close(),"owner"=>weak.upgrade().unwrap().stop_editor(),
                "scope"=>{doc.scope_revision.fetch_add(1,Ordering::AcqRel);},_=>{doc.revision.fetch_add(1,Ordering::AcqRel);},
            })));
            owner.refresh_reaper_transport();let calls=host.calls();let pose=document.clock.diagnostics()["reaper_position_authority"].clone();
            document.close();assert_eq!(calls,["validate:ReaProject*","state"],"{change}: revoked batch must not continue any getter");assert_eq!(pose,false,"{change}: revoked result cannot publish");
        }
    }

    /// 同一owner只有唯一真实region才有typed几何，位置相等不会建立额外身份关系。
    #[test]
    fn task38a_owner_geometry_requires_unique_actual_assignment_and_compatible_playback_window() {
        let (model,owners,ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();let owner=&owners[0];
        let first=(&*ids[0] as *const u8) as u64;let second=(&*ids[1] as *const u8) as u64;
        let host=crate::host::reaper::ReaperFixture::new();unsafe {owner.bind_reaper_host(host.context());}host.reset();
        assert!(owner.reaper_geometry().unwrap_err().contains("incompatible"));
        {let mut regions=document.regions.lock().unwrap();let region=regions.get_mut(&first).unwrap();region.start_in_playback_time=1.;region.duration_in_playback_time=4.;}
        assert_eq!(owner.reaper_geometry().unwrap().region_key,first);
        let raw=owner.binding.lock().unwrap().as_ref().unwrap().as_raw();
        // playback仍只有first，但editor另一region也属于此owner，不能忽略它伪造唯一性。
        unsafe {let ext=&*raw;((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(ext.editorRendererRef,second as *mut _);}
        host.reset();assert!(owner.reaper_geometry().unwrap_err().contains("exactly one"));assert!(host.calls().is_empty());
        unsafe {let ext=&*raw;((*ext.editorRendererInterface).removePlaybackRegion.unwrap())(ext.editorRendererRef,second as *mut _);}
        unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,second as *mut _);}
        host.reset();assert!(owner.reaper_geometry().unwrap_err().contains("exactly one"));assert!(host.calls().is_empty());
        unsafe {let ext=&*raw;for key in [first,second] {((*ext.playbackRendererInterface).removePlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}}
        host.reset();assert!(owner.reaper_geometry().unwrap_err().contains("exactly one"));assert!(host.calls().is_empty());document.close();
    }

    #[test]
    fn task38a_owner_geometry_getter_reentry_discards_closed_or_changed_scope() {
        for change in ["document","owner","scope","model"] {
            let (model,owners,_ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();let owner=owners[0].clone();
            let host=crate::host::reaper::ReaperFixture::new();unsafe {owner.bind_reaper_host(host.context());}host.reset();
            let weak=Arc::downgrade(&owner);let doc=document.clone();
            *host.hook.borrow_mut()=Some(("D_LENGTH".into(),Box::new(move ||match change {
                "document"=>doc.close(),"owner"=>weak.upgrade().unwrap().stop_editor(),
                "scope"=>{doc.scope_revision.fetch_add(1,Ordering::AcqRel);},_=>{doc.revision.fetch_add(1,Ordering::AcqRel);},
            })));
            let result=owner.reaper_geometry();let calls=host.calls();document.close();
            assert!(result.unwrap_err().contains("authorization revoked"));assert_eq!(calls.last().unwrap(),"D_LENGTH","{change}");
        }
    }

    /// 可消费副本只含Rust数据；普通fade变化可在UI重新采集，代次变化/关闭不返回旧值。
    #[test]
    fn task38a_owner_cached_geometry_is_read_only_and_revocable_without_host_calls() {
        let (model,owners,ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();let owner=&owners[0];
        let first=(&*ids[0] as *const u8) as u64;
        {let mut regions=document.regions.lock().unwrap();let region=regions.get_mut(&first).unwrap();region.start_in_playback_time=1.;region.duration_in_playback_time=4.;}
        let host=crate::host::reaper::ReaperFixture::new();unsafe {owner.bind_reaper_host(host.context());}
        assert!(owner.host_geometry_metadata().is_err());owner.refresh_reaper_transport();host.reset();
        let cached=owner.host_geometry_metadata().unwrap();assert_eq!(cached.region_key,first);
        assert_eq!(cached.geometry.fade_in_sec,0.2);assert!(host.calls().is_empty());
        std::thread::scope(|scope| {scope.spawn(|| {assert_eq!(owner.host_geometry_metadata().unwrap(),cached);assert!(owner.reaper_geometry().is_err());}).join().unwrap();});
        assert!(host.calls().is_empty());let revision=document.revision.load(Ordering::Acquire);
        host.set_value("D_FADEINLEN",0.75);owner.refresh_reaper_transport();
        assert_eq!(document.revision.load(Ordering::Acquire),revision);assert_eq!(owner.host_geometry_metadata().unwrap().geometry.fade_in_sec,0.75);
        document.scope_revision.fetch_add(1,Ordering::AcqRel);assert!(owner.host_geometry_metadata().is_err());
        document.close();assert!(owner.host_geometry_metadata().is_err());assert!(owner.host_geometry.lock().unwrap().is_none());
    }

    /// 测试专用同步驱动：冻结阶段持锁，真正内核计算阶段不借文档事务。
    fn render_test_edits(owner:&ExtensionOwner,document:&super::super::document::DocumentSession,edits:&crate::state_channel::EditState)
        ->Result<Vec<super::super::snapshot::PlaybackSnapshot>,String> {
        let input={let _transaction=document.transaction.lock().unwrap();owner.capture_render_input(document,edits,true)?.1};
        input.render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }

    /// 只读提取已提交REAPER旧归档的原始组件JSON，不调用当前encoder生成兼容证据。
    fn task34_archived_v2_states()->Vec<Vec<u8>> {
        use base64::Engine as _;
        let mut chunks=Vec::new();let mut states=Vec::new();let mut inside=false;
        for line in include_str!("../../../../probe/ara/captures/gui-keyboard-edited.RPP").lines().map(str::trim) {
            if line.starts_with("<VST ") {inside=true;chunks.clear();continue;}
            if !inside {continue;}
            if line==">" {
                let start=chunks.windows(b"{\"edits\":".len()).position(|bytes|bytes==b"{\"edits\":").unwrap();
                let length=u32::from_le_bytes(chunks[start-4..start].try_into().unwrap()) as usize;
                states.push(chunks[start..start+length].to_vec());inside=false;
            } else {chunks.extend(base64::engine::general_purpose::STANDARD.decode(line).unwrap());}
        }
        assert_eq!(states.iter().map(Vec::len).collect::<Vec<_>>(),[481,22516]);states
    }

    /// 两条归档记录共享完全相同的旧source/modification身份，必须由真实assignment限定恢复目标。
    #[test]
    fn task34_archived_v2_bytes_rebind_only_inside_the_component_scope() {
        let states=task34_archived_v2_states();
        let (model,owners,_ids)=crate::editor::session::tests::workspace_fixture();let document=model.session();
        let archived:serde_json::Value=serde_json::from_slice(&states[0]).unwrap();
        let identity:Vec<(String,String)>=serde_json::from_value(archived["edits"]["bindings"]["ara-track-0"].clone()).unwrap();
        *document.track_bindings.lock().unwrap()=std::collections::BTreeMap::from([("track".into(),identity.clone()),("b".into(),identity.clone())]);
        // 与宿主冷重建一致：完整图ready前暂存全部组件原始bytes，随后统一合入权威。
        document.ready.store(false,Ordering::Release);
        for (owner,bytes) in owners.iter().zip(&states) {owner.restore_state(bytes).unwrap();}
        document.prepare_renderers();
        let accepted=document.edits.lock().unwrap().clone();
        assert!(accepted.params.get("track").is_none(),"旧A无曲线，不能借旧轨序号取到B曲线");
        assert_eq!(accepted.params["b"].pitch_edit.len(),800);assert_eq!(&accepted.params["b"].pitch_edit[200..212],&[64.;12]);
        assert!(accepted.tracks.iter().all(|track|track.id=="track"||track.id=="b"));
        for (index,id) in ["track","b"].into_iter().enumerate() {
            let bytes=owners[index].encode_state().unwrap();let saved:serde_json::Value=serde_json::from_slice(&bytes).unwrap();
            assert_eq!(saved["version"],2);assert_eq!(saved["edits"]["tracks"].as_array().unwrap().len(),1);
            assert_eq!(saved["edits"]["tracks"][0]["id"],id);assert_eq!(saved["edits"]["bindings"].as_object().unwrap().len(),1);
            assert_eq!(saved["edits"]["params"].as_object().unwrap().len(),index);
        }
        // 真正扩大同一组件assignment，使两个完全相同身份同时进入候选集，必须拒绝合入。
        let key=(&*_ids[1] as *const u8) as u64;
        let raw=owners[0].binding.lock().unwrap().as_ref().unwrap().as_raw();
        unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
        let before=document.edits.lock().unwrap().revision;
        owners[0].restore_state(&states[1]).unwrap();assert!(owners[0].encode_state().unwrap_err().contains("ambiguous"));
        assert_eq!(document.edits.lock().unwrap().revision,before);assert_eq!(document.edits.lock().unwrap().params["b"].pitch_edit[200],64.);
        // 缺少真实身份也不能按归档旧序号恢复；既有曲线留在权威中。
        document.track_bindings.lock().unwrap().insert("track".into(),vec![("unknown-mod".into(),"ara://source".into())]);
        document.track_bindings.lock().unwrap().insert("b".into(),vec![("unknown-b".into(),"ara://source".into())]);
        assert!(owners[0].encode_state().unwrap_err().contains("identity"));
        assert_eq!(document.edits.lock().unwrap().params["b"].pitch_edit[200],64.);document.close();
    }

    /// 真实自动应用在计算前可接受另一组件/模型修改；旧作业不能覆盖新权威或已撤销输出。
    #[test]
    fn automatic_apply_releases_the_transaction_and_rechecks_model_and_edit_versions() {
        for model_changed in [true,false] {
            let model=crate::ara::model::ModelHandle::new();let document=model.session();
            *document.timeline.lock().unwrap()=Some(serde_json::from_value(serde_json::json!({
                "tracks":[],"clips":[],"bpm":120,"project_sec":0})).unwrap());
            document.ready.store(true,Ordering::Release);
            let owner=Arc::new(ExtensionOwner::default());
            owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::EDITOR_RENDERER,None).unwrap();
            owner.snapshots[0].publish(super::super::snapshot::PlaybackSnapshot {
                sample_rate:44100,origin_sample:0,left:vec![0.25;4],right:vec![0.5;4],_reservation:None}).unwrap();
            let projection=document.workspace_projection().unwrap();let calls=std::sync::atomic::AtomicUsize::new(0);
            let result=document.apply_workspace_edits(0,document.revision.load(Ordering::Acquire),&projection,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),|| {
                    if calls.fetch_add(1,Ordering::AcqRel)==0 {
                        assert!(document.transaction.try_lock().is_ok(),"冻结后必须先释放事务才能计算");
                        if model_changed {document.clear_renderers();} else {
                            let timeline=document.timeline.lock().unwrap().clone().unwrap();
                            document.accept_workspace_edits(0,document.revision.load(Ordering::Acquire),&timeline,&projection).unwrap();
                        }
                    }
                    true
                });
            assert!(result.is_err());
            assert!(result.unwrap_err().contains(if model_changed {"host model changed"} else {"superseded"}));
            assert_eq!(document.edits.lock().unwrap().revision,if model_changed {0} else {1});
            let mut left=[9.0;4];let mut right=[9.0;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
            let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
            // SAFETY: 两个平面都是四帧；只检查真实publisher，不能由计算出的expected镜像掩盖覆盖。
            let available=unsafe {owner.snapshots[0].copy_block(0,44100,&mut bus,4)};
            assert_eq!(available,!model_changed);
            assert_eq!(left,if model_changed {[0.0;4]} else {[0.25;4]});
            drop(model);
        }
    }

    /// 即使其它线程暂持模型事务，真实prepare callback也必须返回，并在后台最终发布。
    #[test]
    fn prepare_callback_does_not_wait_for_a_model_transaction() {
        let model=crate::ara::model::ModelHandle::new();let document=model.session();
        *document.timeline.lock().unwrap()=Some(serde_json::from_value(serde_json::json!({
            "tracks":[],"clips":[],"bpm":120,"project_sec":0})).unwrap());
        document.ready.store(true,Ordering::Release);
        let owner=Arc::new(ExtensionOwner::default());
        owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        let held=document.transaction.lock().unwrap();let (sent,received)=std::sync::mpsc::channel();
        let requesting=owner.clone();let caller=std::thread::spawn(move ||{requesting.prepare();sent.send(()).unwrap();});
        let returned=received.recv_timeout(std::time::Duration::from_secs(3)).is_ok();
        drop(held);caller.join().unwrap();assert!(returned,"宿主prepare不得等待文档事务/合成");
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(3);
        while owner.preparation_state().0 {assert!(std::time::Instant::now()<deadline);std::thread::yield_now();}
        assert!(owner.preparation_state().1.is_none());
        let mut left=[9.0;4];let mut right=[9.0;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        // SAFETY: 四帧平面存活；验证worker真的发布空分配快照，不接受“只是忽略prepare”。
        assert!(unsafe {owner.snapshots[0].copy_block(0,44100,&mut bus,4)});assert_eq!(left,[0.0;4]);
        drop(model);
    }

    /// 冷恢复在运行任务被阻塞时也要先合入全部组件状态，不能由各worker逐轨推进revision。
    #[test]
    fn cold_restores_are_merged_document_wide_before_background_jobs_start() {
        let model=crate::ara::model::ModelHandle::new();let document=model.session();
        let mut timeline:hifishifter_kernel::state::TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0},{"id":"b","name":"B","order":1}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":4.0/44100.0,"takes":[{"id":"ta","name":"A","source_path":"source","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"cb","name":"B","track_id":"b","start_sec":0,"length_sec":4.0/44100.0,"takes":[{"id":"tb","name":"B","source_path":"source","source_start_sec":0,"source_end_sec":4.0/44100.0}]}]
        })).unwrap();for clip in &mut timeline.clips {clip.normalize_takes();}
        *document.timeline.lock().unwrap()=Some(timeline);
        let pcm=Arc::new(super::super::source::SourcePcm {sample_rate:44100,planes:vec![vec![0.1,0.2,0.3,0.4]],version:0,_reservation:None});
        document.edit_sources.lock().unwrap().insert("source".into(),pcm.clone());document.sources.lock().unwrap().insert("source".into(),pcm);
        *document.track_bindings.lock().unwrap()=std::collections::BTreeMap::from([
            ("a".into(),vec![("ma".into(),"source".into())]),("b".into(),vec![("mb".into(),"source".into())])]);
        let identities=[Box::new(0_u8),Box::new(0_u8)];let mut owners=Vec::new();let mut gates=Vec::new();
        for (index,clip) in ["ca","cb"].into_iter().enumerate() {
            let key=(&*identities[index] as *const u8) as u64;region_owners().lock().unwrap().register(key,document.id,index).unwrap();
            document.clip_ids.lock().unwrap().insert(key,clip.into());
            document.regions.lock().unwrap().insert(key,crate::ara::AraPlaybackRegion {audio_source_persistent_id:"source".into(),
                duration_in_modification_time:4.0/44100.0,duration_in_playback_time:4.0/44100.0,..Default::default()});
            let owner=Arc::new(ExtensionOwner::default());let raw=owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER,None).unwrap();
            // SAFETY: identities/raw extension保留到doc销毁；这是实际宿主assignment入口。
            unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
            let (release,gate)=std::sync::mpsc::channel();let (started,entered)=std::sync::mpsc::channel();
            owner.preparation.get().unwrap().as_ref().unwrap().request(Box::new(move |_|{
                started.send(()).unwrap();gate.recv_timeout(std::time::Duration::from_secs(3)).unwrap();Ok(())
            })).unwrap();entered.recv_timeout(std::time::Duration::from_secs(3)).unwrap();gates.push(release);
            let host=owner.assigned_timeline(&document).unwrap();let mut client=host.clone();client.tracks[0].volume=if index==0 {0.5} else {0.25};
            let mut saved=crate::state_channel::EditState::default().merge(&host,&client,0).unwrap();
            let id=if index==0 {"a"} else {"b"};saved.reconcile(&std::collections::BTreeMap::from([(id.into(),document.track_bindings.lock().unwrap()[id].clone())])).unwrap();
            owner.restore_state(&saved.encode().unwrap()).unwrap();owners.push(owner);
        }
        document.prepare_renderers();let accepted=document.edits.lock().unwrap().clone();
        // 先释放测试屏障再断言，旧实现失败也不能把真实worker留在无限等待里。
        for gate in gates {gate.send(()).unwrap();}
        assert_eq!(accepted.tracks.iter().find(|t|t.id=="a").map(|t|t.volume),Some(0.5));
        assert_eq!(accepted.tracks.iter().find(|t|t.id=="b").map(|t|t.volume),Some(0.25));
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(3);
        for (index,owner) in owners.iter().enumerate() {
            while owner.preparation_state().0 {assert!(std::time::Instant::now()<deadline);std::thread::yield_now();}
            assert!(owner.preparation_state().1.is_none());let mut left=[9.0;4];let mut right=[9.0;4];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
            let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
            // SAFETY: 两个四帧平面；保证无GUI冷恢复确实供音，而非只合入了state。
            assert!(unsafe {owner.snapshots[0].copy_block(0,44100,&mut bus,4)});
            for (actual,expected) in left.iter().zip(if index==0 {[0.05,0.1,0.15,0.2]} else {[0.025,0.05,0.075,0.1]}) {assert!((*actual-expected).abs()<1e-6);}
        }
        drop(model);
    }

    /// 实际assignment observer必须与发布使用同一事务；不能校验后移除、再发布旧区域。
    #[test]
    fn native_assignment_observer_serializes_with_snapshot_publication_transaction() {
        let model=crate::ara::model::ModelHandle::new();let document=model.session();
        let owner=Arc::new(ExtensionOwner::default());let raw=owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::EDITOR_RENDERER,None).unwrap();
        let identity=Box::new(0_u8);let key=(&*identity as *const u8) as u64;region_owners().lock().unwrap().register(key,document.id,0).unwrap();
        let (ready,entered)=std::sync::mpsc::channel();let (release,released)=std::sync::mpsc::channel();
        let blocking=document.clone();let holder=std::thread::spawn(move || {
            let _held=blocking.transaction.lock().unwrap();ready.send(()).unwrap();
            let _=released.recv_timeout(std::time::Duration::from_millis(300));
        });
        entered.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        let began=std::time::Instant::now();
        // SAFETY: 在绑定时的真实model线程驱动API；其它线程只持Rust事务，不能绕过桥接线程检查。
        unsafe {let ext=&*raw;((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(ext.editorRendererRef,key as *mut _);}
        let elapsed=began.elapsed();let _=release.send(());holder.join().unwrap();
        assert!(elapsed>=std::time::Duration::from_millis(150),"assignment不得绕开发布的文档事务");
        assert_eq!(owner.assignments.lock().unwrap()[&2],[key]);drop(model);
    }

    #[test]
    fn gui_commit_changes_assigned_pcm_and_persisted_state_restores_it() {
        use crate::render::source::SourcePcm;
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let mut timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"one","track_id":"track","name":"one","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","name":"one","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"other","track_id":"track","name":"other","start_sec":1,"length_sec":4.0/44100.0,
                "takes":[{"id":"take2","name":"other","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":2
        })).unwrap();
        for clip in &mut timeline.clips { clip.normalize_takes(); }
        *document.timeline.lock().unwrap() = Some(timeline);
        document.track_bindings.lock().unwrap().insert("track".into(), vec![("host-mod".into(),"ara://pcm".into())]);
        document.edit_sources.lock().unwrap().insert("ara://pcm".into(), Arc::new(SourcePcm { sample_rate:44100,
            planes:vec![vec![0.1,0.2,0.3,0.4]], version:0, _reservation:None }));
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners().lock().unwrap().register(key, document.id, 0).unwrap();
        document.clip_ids.lock().unwrap().insert(key, "one".into());
        document.regions.lock().unwrap().insert(key, crate::ara::AraPlaybackRegion {
            audio_source_persistent_id:"ara://pcm".into(),audio_modification_persistent_id:"host-mod".into(), duration_in_modification_time:4.0/44100.0,
            duration_in_playback_time:4.0/44100.0, ..Default::default() });
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner.bind_to_document(document.clone(), ApiGeneration::V2Final, ExtensionRoles::all(), ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER, None).unwrap();
        unsafe { let ext = &*raw; ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef, key as *mut _); }
        let snapshot = owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let mut client = snapshot.timeline.unwrap();
        assert_eq!(client["clips"].as_array().unwrap().len(), 1);
        client["tracks"][0]["volume"] = serde_json::json!(0.5);
        let response = owner.handle_request(hifishifter_ara_ipc::Request::Commit {
            base_revision:snapshot.revision, model_revision:snapshot.model_revision, timeline:client });
        assert!(response.ok, "{:?}", response.error);
        let companion = Arc::new(ExtensionOwner::default());
        companion.bind_to_document(document.clone(), ApiGeneration::V2Final, ExtensionRoles::all(), ExtensionRoles::EDITOR_RENDERER, None).unwrap();
        assert_eq!(companion.edit_state().lock().unwrap().revision, response.revision);
        assert_eq!(companion.edit_state().lock().unwrap().tracks[0].volume, 0.5);
        let state = owner.edit_state().lock().unwrap().encode().unwrap();
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&state).unwrap();
        let output = render_test_edits(&owner,&document,&restored).unwrap();
        assert_eq!(output[0].left.len(), 4);
        for (actual, expected) in output[0].left.iter().zip([0.05_f32,0.1,0.15,0.2]) { assert!((*actual-expected).abs()<1e-6); }
        assert!(restored.restore(b"bad state").is_err());
        drop(model);
    }

    #[test]
    fn gui_request_commits_parameters_and_rejects_stale_document_revision() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[],"clips":[],"bpm":120,"project_sec":0
        })).unwrap();
        *document.timeline.lock().unwrap() = Some(timeline);
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        owner.bind_to_document(document.clone(), ApiGeneration::V2Final, ExtensionRoles::all(), ExtensionRoles::EDITOR_RENDERER, None).unwrap();
        let snapshot = owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let client = snapshot.timeline.unwrap();
        let commit = hifishifter_ara_ipc::Request::Commit { base_revision: snapshot.revision, model_revision: snapshot.model_revision, timeline: client };
        let response = owner.handle_request(commit.clone());
        assert!(response.ok, "{:?}", response.error);
        assert_eq!(response.revision, 1);
        assert!(!owner.handle_request(commit).ok);
        document.clear_renderers();
        assert!(!owner.handle_request(hifishifter_ara_ipc::Request::Snapshot).ok);
        drop(model);
    }

    /// 两个renderer先后提交局部视图，第二次刷新后不能抹掉第一轨PCM或持久曲线。
    #[test]
    fn two_renderer_commits_keep_both_curves_volumes_pcm_and_saved_state() {
        two_renderer_case(false);
    }
    /// REAPER复制轨道共享modification/source；状态与恢复仍必须限制在宿主实例分配内。
    #[test]
    fn copied_tracks_share_sources_but_keep_instance_state_isolated() {
        two_renderer_case(true);
    }
    fn two_renderer_case(shared_identity:bool) {
        use crate::render::source::SourcePcm;
        use hifishifter_ara_ipc::Request;
        let model=crate::ara::model::ModelHandle::new(); let document=model.session();
        let mut timeline:hifishifter_kernel::state::TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0},{"id":"b","name":"B","order":1}],"bpm":120,"project_sec":1,
            "clips":[{"id":"clip-a","track_id":"a","name":"A","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take-a","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"clip-b","track_id":"b","name":"B","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take-b","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}]
        })).unwrap(); for clip in &mut timeline.clips { clip.normalize_takes(); }
        *document.timeline.lock().unwrap()=Some(timeline);
        document.edit_sources.lock().unwrap().insert("ara://pcm".into(),Arc::new(SourcePcm { sample_rate:44100,planes:vec![vec![0.1,0.2,0.3,0.4]],version:0,_reservation:None }));
        let bindings=std::collections::BTreeMap::from([("a".into(),vec![("mod-a".into(),"ara://pcm".into())]),("b".into(),vec![(if shared_identity {"mod-a"} else {"mod-b"}.into(),"ara://pcm".into())])]);
        *document.track_bindings.lock().unwrap()=bindings.clone(); document.ready.store(true,Ordering::Release);
        let identities=[Box::new(0_u8),Box::new(0_u8)]; let mut owners=Vec::new();
        for (index,id) in ["clip-a","clip-b"].into_iter().enumerate() {
            let key=(&*identities[index] as *const u8) as u64;
            region_owners().lock().unwrap().register(key,document.id,index).unwrap();
            document.clip_ids.lock().unwrap().insert(key,id.into());
            document.regions.lock().unwrap().insert(key,crate::ara::AraPlaybackRegion { audio_source_persistent_id:"ara://pcm".into(),
                audio_modification_persistent_id:if index==0||shared_identity {"mod-a"} else {"mod-b"}.into(),
                duration_in_modification_time:4.0/44100.0,duration_in_playback_time:4.0/44100.0,..Default::default() });
            let owner=Arc::new(ExtensionOwner::default());
            let raw=owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER|ExtensionRoles::EDITOR_RENDERER,None).unwrap();
            // SAFETY: identities及native扩展都保留到owner/document销毁。
            unsafe { let ext=&*raw; ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _); }
            owners.push(owner);
        }
        for (index,id) in ["a","b"].into_iter().enumerate() {
            let snapshot=owners[index].handle_request(Request::Snapshot); assert!(snapshot.ok,"{:?}",snapshot.error);
            assert_eq!(snapshot.revision,index as u64,"B先刷新最新共享revision");
            let mut client=snapshot.timeline.unwrap(); client["tracks"][0]["volume"]=serde_json::json!(if index==0 {0.5} else {0.25});
            client["params_by_root_track"]=serde_json::json!({id:{"frame_period_ms":5.0,"pitch_edit":[61.0+index as f32,63.0+index as f32]}});
            let response=owners[index].handle_request(Request::Commit { base_revision:snapshot.revision,model_revision:snapshot.model_revision,timeline:client }); assert!(response.ok,"{:?}",response.error);
        }
        for (index,id) in ["a","b"].into_iter().enumerate() {
            let saved:serde_json::Value=serde_json::from_slice(&owners[index].encode_state().unwrap()).unwrap();
            assert_eq!(saved["edits"]["params"].as_object().unwrap().len(),1,"单个组件不能保存整个ARA文档的其它轨道");
            assert!(saved["edits"]["params"].get(id).is_some());
        }
        let mut restored=document.edits.lock().unwrap().clone();restored.reconcile(&bindings).unwrap();
        for (index,id) in ["a","b"].into_iter().enumerate() {
            assert_eq!(restored.params[id].pitch_edit,[61.0+index as f32,63.0+index as f32]);
            let output=render_test_edits(&owners[index],&document,&restored).unwrap();
            let factor=if index==0 {0.5} else {0.25};
            for (actual,original) in output[0].left.iter().zip([0.1_f32,0.2,0.3,0.4]) { assert!((*actual-original*factor).abs()<1e-6); }
            let current=owners[index].handle_request(Request::Snapshot); assert!(current.ok);
            assert_eq!(current.timeline.unwrap()["tracks"][0]["volume"],serde_json::json!(factor));
        }
        if shared_identity {
            let original=owners[0].encode_state().unwrap();let second=owners[1].encode_state().unwrap();
            // REAPER复制插件：同一源身份由另一实例的实际assignment限定到B，不能改写A。
            owners[1].restore_state(&original).unwrap();
            let a=owners[0].handle_request(Request::Snapshot);let b=owners[1].handle_request(Request::Snapshot);
            assert!(a.ok && b.ok,"{:?} {:?}",a.error,b.error);
            assert_eq!(a.timeline.unwrap()["tracks"][0]["volume"],0.5);
            assert_eq!(b.timeline.unwrap()["tracks"][0]["volume"],0.5);
            assert_eq!(document.edits.lock().unwrap().params["b"].pitch_edit,[61.,63.]);
            owners[1].restore_state(&second).unwrap();
            assert_eq!(document.edits.lock().unwrap().params["a"].pitch_edit,[61.,63.]);
            assert_eq!(document.edits.lock().unwrap().params["b"].pitch_edit,[62.,64.]);
            let editors:Vec<_>=owners.iter().map(|owner|owner.editor_session().unwrap()).collect();
            let call=|editor:&Arc<crate::editor::session::EditorSession>,command:&str,args:serde_json::Value| {
                let (reply,received)=std::sync::mpsc::channel();let (events,_)=std::sync::mpsc::sync_channel(128);
                let sink=crate::editor::session::UiSink {view_id:"dual-actor".into(),reply,events,closed:Arc::new(std::sync::atomic::AtomicBool::new(false))};
                editor.enqueue(crate::editor::session::UiRequest {id:1,command:command.into(),args,sink,link:None}).unwrap();
                let response=received.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                assert_eq!(response["ok"],true,"{response}");response["value"].clone()
            };
            for editor in &editors {call(editor,"get_timeline_state",serde_json::json!({}));}
            // 同一次debounce窗口内两轨落笔；B的共享revision变化不能让A的最新作业Conflict。
            for (index,editor) in editors.iter().enumerate() {
                let timeline=call(editor,"get_timeline_state",serde_json::json!({}));
                call(editor,"set_track_state",serde_json::json!({
                    "trackId":timeline["tracks"][index]["id"],"volume":if index==0 {0.75} else {0.125}
                }));
            }
            let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
            for editor in &editors {
                loop {let state=call(editor,"plugin_get_apply_state",serde_json::json!({}));assert!(state["error"].is_null(),"{state}");
                    if state["pending"]==false {break;}assert!(std::time::Instant::now()<deadline,"{state}");
                    std::thread::sleep(std::time::Duration::from_millis(20));}
            }
            editors[0].close(); // 现在两个真实入口共享actor，只在全部回归请求完成后关闭。
            assert_eq!(document.edits.lock().unwrap().tracks.iter().find(|t|t.id=="a").unwrap().volume,0.75);
            assert_eq!(document.edits.lock().unwrap().tracks.iter().find(|t|t.id=="b").unwrap().volume,0.125);
        }
    }

    /// 真实空文档绑定也必须收到销毁；不依赖 first region assignment 猜 document。
    #[test]
    fn bound_native_entry_is_tombstoned_by_actual_document_teardown() {
        let factory = Box::leak(Box::new(
            FactoryBuilder::new("org.hfs.bound", "org.hfs.bound.archive")
                .display("bound", "HiFiShifter", "https://example.invalid", "1")
                .document_controller(|| {
                    let model = crate::ara::model::ModelHandle::new();
                    let session = model.session();
                    PluginBuilder::new(model)
                        .controller_identity(move |key| session.register(key))
                        .build()
                })
                .build()
                .unwrap(),
        ));
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"bound empty".as_ptr(),
        };
        // SAFETY: test host、factory 和 properties 保持到实际 controller 终止。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 原生工厂返回完整 packed 实例。
        let controller = unsafe { raw.read_unaligned() };
        let document = crate::render::document::DocumentSession::lookup(
            controller.documentControllerRef as usize,
        )
        .unwrap();
        // SAFETY: test factory backing 已保留到进程结束。
        let association =
            unsafe { CompanionFactory::from_raw("bound", &*factory.as_raw()) }.unwrap();
        let processor =
            CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
        let probe = processor.lifetime_probe();
        let owner = Arc::new(ExtensionOwner::default());
        let entry = HostEntry::new(processor, "bound", owner.clone()).unwrap();
        let mut extension_raw = std::ptr::null();
        // SAFETY: real C++ shim 调真实产品 bind，controller 存活且已登记。
        assert_eq!(
            unsafe {
                ara2_vst3_plugin_entry_bind(
                    entry.as_raw(),
                    controller.documentControllerRef.cast(),
                    7,
                    1,
                    1,
                    &raw mut extension_raw,
                )
            },
            ARA2_VST3_OK
        );
        assert!(!extension_raw.is_null());
        assert!(probe.controller_alive());
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        // SAFETY: extension 与 controller 保留；region 仅作为不透明身份。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert_eq!(owner.assignments.lock().unwrap().get(&1).unwrap(), &[key]);
        // SAFETY: actual controller destructor invokes ModelHandle::destroy_document。
        unsafe {
            ((*controller.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(controller.documentControllerRef)
        };
        assert!(owner.assignments.lock().unwrap().is_empty());
        assert!(owner.assigned_regions().is_err());
        assert!(!probe.controller_alive());
        // SAFETY: native entry 仍保留 companion storage；销毁文档后回调必须被 tombstone 拒绝。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert!(owner.assignments.lock().unwrap().is_empty());
        drop(entry);
        drop(owner);
        factory.entry().uninitialize().unwrap();
    }

    /// sequence 预览只展开本 sequence；显式区域重复分配不能导致双倍混音。
    #[test]
    fn editor_sequence_selection_is_expanded_and_deduplicated() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let region_a = Box::new(0_u8);
        let region_b = Box::new(0_u8);
        let sequence = Box::new(0_u8);
        let a = (&*region_a as *const u8) as u64;
        let b = (&*region_b as *const u8) as u64;
        let seq = (&*sequence as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(a, document.id, 0)
            .unwrap();
        region_owners()
            .lock()
            .unwrap()
            .register(b, document.id, 1)
            .unwrap();
        document
            .sequence_regions
            .lock()
            .unwrap()
            .insert(seq, [a, b].into_iter().collect());
        // SAFETY: binding、session 和不透明身份在调用期间均存活。
        unsafe {
            let extension = &*raw;
            let api = &*extension.editorRendererInterface;
            api.addPlaybackRegion.unwrap()(extension.editorRendererRef, a as *mut _);
            api.addRegionSequence.unwrap()(extension.editorRendererRef, seq as *mut _);
        }
        let mut expected = vec![a, b];
        expected.sort_unstable();
        assert_eq!(owner.assigned_regions().unwrap(), expected);
        document
            .sequence_regions
            .lock()
            .unwrap()
            .get_mut(&seq)
            .unwrap()
            .remove(&b);
        region_owners().lock().unwrap().remove(b);
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        // SAFETY: lease 保留接口，remove sequence 不会移除显式 region。
        unsafe {
            let extension = &*raw;
            ((*extension.editorRendererInterface)
                .removeRegionSequence
                .unwrap())(extension.editorRendererRef, seq as *mut _);
        }
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        drop(model);
        assert!(owner.assigned_regions().is_err());
    }

    /// companion 先销毁时 controller lease 必须独立保留真实 raw storage。
    #[test]
    fn actual_document_retains_bound_storage_after_owner_is_dropped() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document,
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .unwrap();
        drop(owner);
        let mut identity = 0_u8;
        // SAFETY: actual model 的 session 独立持有控制器 lease，owner 已 drop 也不释放 storage。
        unsafe {
            let extension = raw.read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
            ((*extension.playbackRendererInterface)
                .removePlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
        }
        drop(model);
        // 模型终止后所有 owner 都已释放，此后不得再访问 raw。
    }
}

impl ExtensionOwner {
    /// 组件终止独立于Arc尚存与否，旧native entry不能维持可编辑授权。
    pub(crate) fn is_closed(&self)->bool {self.closed.load(Ordering::Acquire)}
    pub(crate) fn editor_document(&self)->Result<Arc<super::document::DocumentSession>,String> {
        if self.is_closed() {return Err("FX processor closed".into());}
        self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade).filter(|document|document.is_alive())
            .ok_or_else(||"document closed".into())
    }
    /// 初始化在宿主UI/model线程保存有引用的typed扩展，不在actor或process调用REAPER API。
    /// # Safety
    /// context须为宿主初始化期间活FUnknown。
    pub(crate) unsafe fn bind_reaper_host(&self,context:*mut std::ffi::c_void) {
        let bound=self.document.lock().unwrap().is_some();
        let stamp=if bound {match self.host_query_stamp() {Ok(stamp)=>Some(stamp),Err(_)=>return}} else {None};
        let authorized=||!self.is_closed()&&match &stamp {Some(stamp)=>self.host_query_authorized(stamp),None=>self.document.lock().unwrap().is_none()};
        let host=unsafe {crate::host::reaper::ReaperHost::from_context(context,authorized)}.map(Arc::new);
        if !authorized() {return;}
        crate::log_line(&format!("REAPER host extension available={}",host.is_some()));
        let old=std::mem::replace(&mut *self.reaper.lock().unwrap(),host);drop(old);
        self.host_geometry.lock().unwrap().take();
    }
    /// 短事务冻结真实owner/document/model/scope；整个host调用链都不持内部锁。
    fn host_query_stamp(&self)->Result<(Arc<super::document::DocumentSession>,u64,u64,Vec<u64>),String> {
        let document=self.editor_document()?;
        let _transaction=document.transaction.lock().unwrap();
        if self.is_closed()||!document.is_alive() {return Err("host query document/owner closed".into());}
        let keys=self.host_assigned_regions(&document)?;
        let model=document.revision.load(Ordering::Acquire);let scope=document.scope_revision.load(Ordering::Acquire);
        drop(_transaction);Ok((document,model,scope,keys))
    }
    fn host_query_authorized(&self,stamp:&(Arc<super::document::DocumentSession>,u64,u64,Vec<u64>))->bool {
        let (document,model,scope,keys)=stamp;
        let _transaction=document.transaction.lock().unwrap();
        !self.is_closed()&&document.is_alive()
            && self.editor_document().is_ok_and(|current|Arc::ptr_eq(&current,document))
            && document.revision.load(Ordering::Acquire)==*model&&document.scope_revision.load(Ordering::Acquire)==*scope
            && self.host_assigned_regions(document).is_ok_and(|current|current==*keys)
    }
    /// 几何绑定必须覆盖owner的全部已分配角色，不能只拿playback子集隐藏editor歧义。
    fn host_assigned_regions(&self,document:&super::document::DocumentSession)->Result<Vec<u64>,String> {
        let mut keys=self.assignments.lock().unwrap().values().flatten().copied().collect::<BTreeSet<_>>();
        let sequences=self.sequences.lock().unwrap().values().flatten().copied().collect::<BTreeSet<_>>();
        let members=document.sequence_regions.lock().unwrap();
        for sequence in sequences {keys.extend(members.get(&sequence).ok_or("unknown host assigned sequence")?.iter().copied());}
        drop(members);let keys=keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty()&&region_owners().lock().unwrap().resolve(&keys).map_err(|_|"invalid host assigned region")?.0!=document.id {
            return Err("host assigned region belongs to another document".into());
        }Ok(keys)
    }
    /// 仅UI/model线程读取；唯一assignment先成立，位置/长度仅用于绑定后相容核对。
    pub(crate) fn reaper_geometry(&self)->Result<crate::host::geometry::BoundHostGeometry,String> {
        let stamp=self.host_query_stamp()?;
        let [region_key]=stamp.3.as_slice() else {return Err("REAPER geometry requires exactly one assigned ARA region".into());};
        let region=stamp.0.regions.lock().unwrap().get(region_key).cloned().ok_or("assigned ARA region unavailable")?;
        let host=self.reaper.lock().unwrap().clone().ok_or("REAPER host extension unavailable")?;
        let geometry=host.geometry(||self.host_query_authorized(&stamp))?;
        let compatible=|a:f64,b:f64|a.is_finite()&&b.is_finite()
            &&(a-b).abs()<=1e-7+8.*f64::EPSILON*a.abs().max(b.abs());
        if !compatible(geometry.start_sec,region.start_in_playback_time)||!compatible(geometry.duration_sec,region.duration_in_playback_time) {
            return Err("direct take geometry incompatible with assigned ARA playback window".into());
        }
        if !self.host_query_authorized(&stamp) {return Err("host geometry authorization revoked".into());}
        Ok(crate::host::geometry::BoundHostGeometry {region_key:*region_key,geometry})
    }
    /// actor/worker只可消费UI冻结的Rust值，不沿此访问器调用host；代次变化明确不可用。
    pub(crate) fn host_geometry_metadata(&self)->Result<crate::host::geometry::BoundHostGeometry,String> {
        let stamp=self.host_query_stamp()?;
        let cached=self.host_geometry.lock().unwrap().clone().ok_or("REAPER geometry has not been sampled on model/UI thread")?;
        if cached.model!=stamp.1||cached.scope!=stamp.2||!self.host_query_authorized(&stamp) {
            return Err("REAPER geometry metadata superseded".into());
        }cached.value
    }
    /// 只由已验证的native view UI timer调用；释放内部锁后才调用可能重入的host函数。
    pub(crate) fn refresh_reaper_transport(&self) {
        let Ok(stamp)=self.host_query_stamp() else {return;};
        let host={self.reaper.lock().unwrap().clone()};
        if let Some(host)=host {if let Ok((position,playing))=host.sample(||self.host_query_authorized(&stamp)) {
            // getter可能重入关闭/改scope，发布与撤销共用最终短事务。
            let _transaction=stamp.0.transaction.lock().unwrap();
            if !self.is_closed()&&stamp.0.is_alive()&&stamp.0.revision.load(Ordering::Acquire)==stamp.1
                &&stamp.0.scope_revision.load(Ordering::Acquire)==stamp.2 {
                if let Some(clock)=self.clock.get() {clock.publish_host(position,playing);}
            }
        }}
        if !self.host_query_authorized(&stamp) {return;}
        self.refresh_reaper_geometry();
        // editor-only通常没有所属take；一个工程GUI必须驱动真正隐藏playback实例的只读采集。
        // renderer_owners仅返回本真实文档的活租约，不扫描全局实例，也不借名称/位置找take。
        for owner in stamp.0.renderer_owners() {
            if !self.host_query_authorized(&stamp) {break;}
            if owner.renders_playback()&&!std::ptr::eq(self,Arc::as_ptr(&owner)) {owner.refresh_reaper_geometry();}
        }
    }
    /// 元数据单独采集，不让每个隐藏实例重复写工程clock；宿主调用始终不持内部锁。
    fn refresh_reaper_geometry(&self) {
        let Ok(stamp)=self.host_query_stamp() else {return;};
        let host={self.reaper.lock().unwrap().clone()};
        let authorized=||self.host_query_authorized(&stamp);
        let before=host.as_ref().and_then(|host|host.geometry_revision(authorized).ok());
        if !authorized() {return;}
        let unchanged=before.is_some()&&self.host_geometry.lock().unwrap().as_ref().is_some_and(|cached|
            cached.model==stamp.1&&cached.scope==stamp.2&&cached.change==before&&cached.value.is_ok());
        if unchanged {
            let after=host.as_ref().and_then(|host|host.geometry_revision(authorized).ok());
            if before==after&&authorized() {return;}
        }
        if !authorized() {return;}
        let mut geometry=self.reaper_geometry();
        let after=host.as_ref().and_then(|host|host.geometry_revision(authorized).ok());
        if geometry.is_ok()&&(before.is_none()||before!=after) {
            geometry=Err("REAPER project changed during cached geometry refresh".into());
        }
        let _transaction=stamp.0.transaction.lock().unwrap();
        if !self.is_closed()&&stamp.0.is_alive()&&stamp.0.revision.load(Ordering::Acquire)==stamp.1
            &&stamp.0.scope_revision.load(Ordering::Acquire)==stamp.2 {
            *self.host_geometry.lock().unwrap()=Some(CachedHostGeometry {model:stamp.1,scope:stamp.2,change:after,value:geometry});
        }
    }
    /// realtime只读角色原子值；只有playback角色负责替换歌曲音频。
    pub(crate) fn renders_playback(&self)->bool {self.role.load(Ordering::Acquire)==1}
    /// editor-only必须透传；未绑定角色不在这里虚构为editor，保留原未绑定安全行为。
    pub(crate) fn is_editor_only(&self)->bool {self.role.load(Ordering::Acquire)==2}
    /// process只写有界原子诊断，JSON/日志全部由actor读取，暂不把候选计数当作根因。
    pub(crate) fn observe_transport(&self,context:&crate::audio_abi::ProcessContext,mode:i32) {
        if let Some(clock)=self.clock.get() {
            clock.observe(context,self.writer_id.load(Ordering::Relaxed),self.roles.load(Ordering::Relaxed),mode);
            if mode!=2 {clock.update(context);}
        }
    }
    /// SDK允许从音频线程调用；记录来源但不做文件IO。
    pub(crate) fn record_processing_stop(&self) {
        if let Some(clock)=self.clock.get() {
            clock.observe_stop(self.writer_id.load(Ordering::Relaxed),self.roles.load(Ordering::Relaxed));clock.stopped();
        }
    }
    /// 真实组件仅作为授权入口，原GUI共享同文档唯一actor/history。
    pub(crate) fn editor_session(self:&Arc<Self>)->Result<Arc<crate::editor::session::EditorSession>,String> {
        self.editor_document()?.editor_session()
    }
    /// 组件/文档终止时停止实例worker；不要在音频process调用。
    pub(crate) fn stop_editor(&self) {
        let document={self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade)};
        if let Some(document)=document {
            {let _transaction=document.transaction.lock().unwrap();
                if !self.closed.swap(true,Ordering::AcqRel) {document.scope_revision.fetch_add(1,Ordering::AcqRel);}
                self.snapshots.iter().for_each(|snapshot|snapshot.clear());
            }
            // 不持文档事务取得views锁，避免enqueue/worker授权反向等待。
            document.revoke_editor_views();
        } else {
            self.closed.store(true,Ordering::Release);
            self.snapshots.iter().for_each(|snapshot|snapshot.clear());
        }
        let host=self.reaper.lock().unwrap().take();drop(host);
        self.host_geometry.lock().unwrap().take();
        self.cancel_preparation();
        if let Some(Ok(worker))=self.preparation.get() {worker.close();}
    }
    /// 模型撤销时立即取消待准备作业；保持worker可供后续重新授权使用。
    pub(crate) fn cancel_preparation(&self) {
        if let Some(Ok(worker))=self.preparation.get() {worker.cancel();}
    }
    /// 原GUI合并后台宿主准备状态，不能把冷恢复尚未准备完写成已应用。
    pub(crate) fn preparation_state(&self)->(bool,Option<String>) {
        if self.is_editor_only() {
            if let Some(document)=self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade) {
                let states=document.renderer_owners().into_iter().filter(|owner|owner.renders_playback()).map(|owner|owner.local_preparation_state()).collect::<Vec<_>>();
                return (states.iter().any(|state|state.0),states.into_iter().find_map(|state|state.1));
            }
        }
        self.local_preparation_state()
    }
    fn local_preparation_state(&self)->(bool,Option<String>) {
        match self.preparation.get() {Some(Ok(worker))=>worker.state(),Some(Err(error))=>(false,Some(error.clone())),None=>(false,None)}
    }
    /// native主线程取得宿主给本ARA文档的可撤销播放租约；不寻找全局REAPER窗口。
    pub(crate) fn host_playback(&self)->Option<ara2_bridge::plugin::PlaybackRequestHandle> {
        self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade)
            .and_then(|document|document.playback.lock().unwrap().clone())
    }
    /// 组件释放/文档关闭前结束后台服务，避免DLL卸载后线程仍执行插件代码。
    pub fn stop_channel(&self) { let channel = self.channel.lock().unwrap().take(); drop(channel); }

    /// 未绑定时暂存组件state；绑定后所有处理器读取同一文档的参数权威。
    pub fn edit_state(&self) -> Arc<Mutex<crate::state_channel::EditState>> {
        self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade)
            .map(|document| document.edits.clone()).unwrap_or_else(|| self.edits.clone())
    }

    /// 恢复/undo组件状态后刷新整张文档，各renderer保持原分配。
    pub fn refresh_document(&self) {
        let document = self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            {let _transaction=document.transaction.lock().unwrap();
                let owners=document.renderer_owners();
                for owner in &owners {
                    if let Err(error)=owner.merge_pending_restore(&document) {log::warn!("[ara] instance state unresolved: {error}");}
                }
                for owner in owners {owner.prepare();}
            }
        }
    }

    /// setState只暂存本组件恢复；宿主明确分配区域且模型ready后再合入共享权威。
    pub(crate) fn restore_state(&self,bytes:&[u8])->Result<(),String> {
        if bytes.is_empty() {return Ok(());}
        let mut restored=crate::state_channel::EditState::default();
        restored.restore(bytes)?;
        *self.pending_restore.lock().unwrap()=Some(restored);
        self.refresh_document();
        Ok(())
    }
    /// 调用方持文档transaction；恢复候选集由宿主区域分配收窄，不按名称/旧序号猜测。
    pub(super) fn merge_pending_restore(&self,document:&super::document::DocumentSession)->Result<(),String> {
        let mut pending=self.pending_restore.lock().unwrap();
        let Some(saved)=pending.as_ref() else {return Ok(());};
        if !document.ready.load(Ordering::Acquire) {return Ok(());}
        let host=self.assigned_timeline(document)?;
        if host.tracks.is_empty() {return Ok(());}
        let allowed:BTreeSet<_>=host.tracks.iter().map(|t|t.id.clone()).collect();
        let bindings=document.track_bindings.lock().unwrap().iter().filter(|(id,_)|allowed.contains(*id))
            .map(|(id,identity)|(id.clone(),identity.clone())).collect();
        let mut restored=saved.clone();restored.reconcile(&bindings)?;
        let mut changed_geometry=false;
        if !restored.atlas.is_empty() {
            let rebound=restored.atlas.rebind(&host,&document.parameter_identities_locked(&host)?)?;
            changed_geometry=!restored.atlas.same_layout(&rebound);restored.atlas=rebound;
        }
        let mut client=host.clone();restored.apply(&mut client);
        if changed_geometry {client.params_by_root_track.extend(restored.atlas.project_roots(&host,&document.parameter_identities_locked(&host)?)?);}
        let mut edits=document.edits.lock().unwrap();
        let mut merged=edits.merge(&host,&client,edits.revision)?;
        let clip_ids=host.clips.iter().map(|clip|clip.id.clone()).collect::<BTreeSet<_>>();
        merged.atlas.regions.retain(|id,_|!clip_ids.contains(id));merged.atlas.regions.extend(restored.atlas.regions);
        merged.reconcile(&document.track_bindings.lock().unwrap())?;
        *edits=merged;*pending=None;Ok(())
    }

    /// 保存前重新核对完整宿主图；歧义或中间编辑态不能伪装成可恢复的state。
    pub(crate) fn encode_state(&self) -> Result<Vec<u8>, String> {
        let document = self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            document.flush_editor()?;
            let _transaction = document.transaction.lock().unwrap();
            if !document.ready.load(Ordering::Acquire) { return Err("host graph not ready; cannot save ARA edits".into()); }
            self.merge_pending_restore(&document)?;
            let mut edits = document.edits.lock().unwrap();
            edits.reconcile(&document.track_bindings.lock().unwrap())?;
            let host=self.assigned_timeline(&document)?;
            let allowed:BTreeSet<_>=host.tracks.iter().map(|t|t.id.clone()).collect();
            let mut local=edits.clone();
            local.params.retain(|id,_|allowed.contains(id));local.tracks.retain(|t|allowed.contains(&t.id));
            let clips=host.clips.iter().map(|clip|clip.id.clone()).collect::<BTreeSet<_>>();local.atlas.regions.retain(|id,_|clips.contains(id));
            local.bindings.retain(|id,_|allowed.contains(id));local.encode()
        } else { self.pending_restore.lock().unwrap().as_ref().unwrap_or(&self.edits.lock().unwrap()).encode() }
    }

    /// 参数/快照在短事务内校验；外部提交合成同样不能持锁阻塞宿主模型回调。
    pub(crate) fn handle_request(&self, request: hifishifter_ara_ipc::Request) -> hifishifter_ara_ipc::Response {
        use hifishifter_ara_ipc::{Request, Response, HostPcm};
        let document = self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade);
        let Some(document) = document else { return Response { error: Some("document closed".into()), ..Default::default() }; };
        if let Request::Commit {base_revision,model_revision,timeline}=request {
            return self.commit_request(&document,base_revision,model_revision,timeline);
        }
        let _transaction = document.transaction.lock().unwrap();
        if let Err(error)=self.merge_pending_restore(&document) {return Response {error:Some(error),..Default::default()};}
        let model_revision = document.revision.load(Ordering::Acquire);
        let mut edits = document.edits.lock().unwrap();
        let outcome = (|| {
            if !document.ready.load(Ordering::Acquire) { return Err("host model not ready; refresh after editing".into()); }
            edits.reconcile(&document.track_bindings.lock().unwrap())?;
            let mut timeline = self.assigned_timeline(&document)?;
            match request {
                Request::Snapshot => {
                    edits.apply(&mut timeline);
                    let available = document.edit_sources.lock().unwrap();
                    let mut sources = Vec::new();
                    for id in timeline.clips.iter().filter_map(|clip| clip.source_path.as_ref()).collect::<BTreeSet<_>>() {
                        let pcm = available.get(id).ok_or_else(|| format!("host PCM unavailable: {id}"))?;
                        sources.push(HostPcm { persistent_id: id.clone(), sample_rate: pcm.sample_rate,
                            fingerprint: pcm_fingerprint(pcm), planes: pcm.planes.clone() });
                    }
                    Ok(Response { ok: true, timeline: Some(serde_json::to_value(&timeline).map_err(|e| e.to_string())?), sources, ..Default::default() })
                }
                Request::Commit {..}=>unreachable!("commit dispatched before snapshot transaction"),
            }
        })();
        let mut response = outcome.unwrap_or_else(|error| Response { error: Some(error), ..Default::default() });
        response.revision = edits.revision;
        response.model_revision = model_revision;
        response
    }

    /// 一期外部客户端保持同步结果契约，但捕获/计算/提交分开，失败不改变参数权威。
    fn commit_request(&self,document:&Arc<super::document::DocumentSession>,base_edit:u64,base_model:u64,
        client:serde_json::Value)->hifishifter_ara_ipc::Response {
        let outcome=(|| {
            let (candidate,epoch,scope,inputs)={
                let _transaction=document.transaction.lock().unwrap();self.merge_pending_restore(document)?;
                if !document.ready.load(Ordering::Acquire) {return Err("host model not ready; refresh after editing".into());}
                if document.revision.load(Ordering::Acquire)!=base_model {return Err("Conflict: host model changed; refresh".into());}
                let mut edits=document.edits.lock().unwrap();edits.reconcile(&document.track_bindings.lock().unwrap())?;
                let client=serde_json::from_value(client).map_err(|e|e.to_string())?;
                let mut candidate=edits.merge(&self.assigned_timeline(document)?,&client,base_edit)?;
                let mut curve_timeline=self.assigned_timeline(document)?;candidate.apply(&mut curve_timeline);
                candidate.atlas=edits.atlas.capture(&curve_timeline,&document.parameter_identities_locked(&curve_timeline)?)?;
                candidate.reconcile(&document.track_bindings.lock().unwrap())?;
                drop(edits);
                let inputs=document.renderer_owners().into_iter().filter(|owner|owner.renders_playback()).map(|owner| {
                    let (keys,input)=owner.capture_render_input(document,&candidate,true)?;Ok((owner,keys,input))
                }).collect::<Result<Vec<_>,String>>()?;
                (candidate,document.render_epoch.load(Ordering::Acquire),document.scope_revision.load(Ordering::Acquire),inputs)
            };
            let mut prepared=Vec::new();
            for (owner,keys,input) in inputs {
                for publisher in &owner.snapshots {publisher.collect_retired();}
                prepared.push((owner,keys,input.render(Arc::new(std::sync::atomic::AtomicBool::new(false)))?));
            }
            let _transaction=document.transaction.lock().unwrap();
            if !document.ready.load(Ordering::Acquire) || document.revision.load(Ordering::Acquire)!=base_model {
                return Err("Conflict: host model changed during commit".into());
            }
            let mut edits=document.edits.lock().unwrap();
            if edits.revision!=base_edit {return Err("Conflict: edit revision changed during commit".into());}
            if document.render_epoch.load(Ordering::Acquire)!=epoch||document.scope_revision.load(Ordering::Acquire)!=scope {return Err("host audio access/scope changed during commit; retry".into());}
            for (owner,keys,snapshots) in &prepared {
                if owner.assigned_regions().map_err(|e|e.to_string())?!=*keys {return Err("Conflict: assigned regions changed during commit".into());}
                if !owner.snapshots.iter().zip(snapshots).all(|(p,s)|p.has_capacity(s)) {return Err("retired snapshot budget exhausted; reopen instance".into());}
            }
            for (owner,keys,snapshots) in prepared {
                for (publisher,snapshot) in owner.snapshots.iter().zip(snapshots) {publisher.publish(snapshot).map_err(|e|format!("snapshot publish failed: {e:?}"))?;}
                owner.record_prepared(base_model,candidate.revision,epoch,scope,keys);
            }
            *edits=candidate;log::info!("[ara] GUI commit ready revision={} model={base_model}",edits.revision);
            Ok::<(),String>(())
        })();
        hifishifter_ara_ipc::Response {ok:outcome.is_ok(),error:outcome.err(),revision:document.edits.lock().unwrap().revision,
            model_revision:document.revision.load(Ordering::Acquire),..Default::default()}
    }

    /// 调用方在短事务内冻结几何/参数/PCM，只复制本renderer授权源，不执行合成。
    pub(super) fn capture_render_input(&self,document:&super::document::DocumentSession,edits:&crate::state_channel::EditState,edited:bool)
        ->Result<(Vec<u64>,super::input::RenderInput),String> {
        if !document.ready.load(Ordering::Acquire) {return Err("host PCM/model not ready".into());}
        let keys=self.assigned_regions().map_err(|e|e.to_string())?;
        let mut timeline = self.assigned_timeline(document)?;
        // 所有授权轨道保留原kernel的全局solo/父链判定，clip仍只有本renderer分配区域。
        timeline.tracks=document.workspace_timeline_locked()?.tracks;
        let mut resolved = edits.clone();
        resolved.reconcile(&document.track_bindings.lock().unwrap())?;
        resolved.apply(&mut timeline);
        let geometry = document.regions.lock().unwrap();
        let regions = keys.iter().map(|key| geometry.get(key).cloned().ok_or("assigned region disappeared"))
            .collect::<Result<Vec<_>, _>>()?;
        drop(geometry);
        let stretch=regions.iter().any(|region|(region.duration_in_modification_time-region.duration_in_playback_time).abs()>1e-9);
        let kernel_render=edited||stretch;
        let clip_parameters=if resolved.atlas.is_empty() {Default::default()} else {
            resolved.atlas.project_local(&timeline,&document.parameter_identities_locked(&timeline)?)?
        };
        let available=if kernel_render {document.edit_sources.lock().unwrap()} else {document.sources.lock().unwrap()};
        let sources=regions.iter().map(|region|region.audio_source_persistent_id.clone()).collect::<BTreeSet<_>>()
            .into_iter().filter_map(|id|available.get(&id).map(|pcm|(id.clone(),pcm.clone()))).collect();
        Ok((keys,super::input::RenderInput {timeline:kernel_render.then_some(timeline),regions,sources,clip_parameters}))
    }

    /// 从宿主时间线筛选实际分配的区域，而不是让每个处理器混整张文档。
    fn assigned_timeline(&self, document: &super::document::DocumentSession) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        let identities = document.clip_ids.lock().unwrap();
        let ids = keys.iter().map(|key| identities.get(key).cloned().ok_or("missing assigned clip identity"))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut timeline = document.timeline.lock().unwrap().clone().ok_or("host timeline unavailable")?;
        if let Some(tempo)=document.clock.tempo() {timeline.bpm=tempo;}
        timeline.clips.retain(|clip| ids.contains(&clip.id));
        // 零分配也只能看零轨道，不能意外把整张文档交给空renderer编辑。
        timeline.tracks.retain(|track| timeline.clips.iter().any(|clip| clip.track_id == track.id));
        Ok(timeline)
    }

    /// 把租约关联到真实文档；关闭文档时由其同步撤销。
    pub fn bind_to_document(
        self: &Arc<Self>,
        document: Arc<super::document::DocumentSession>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        companion: Option<ara2_bridge::companion::CompanionControllerBinding<'static>>,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            return Err(AraError::InvalidState("extension already bound"));
        }
        let owner = Arc::downgrade(self);
        let document_id = document.id;
        let observer = Arc::new(
            move |role: ExtensionRoles, keys: &[usize], sequences: &[usize]| {
                if let Some(owner) = owner.upgrade() {
                    let document=owner.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade);
                    let Some(document)=document else {return;};
                    // assignment写入、撤销和最终发布必须共用此短事务，消除check→publish竞态。
                    let _transaction=document.transaction.lock().unwrap();
                    let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                    let valid = keys.is_empty()
                        || region_owners()
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .resolve(&keys)
                            .is_ok_and(|(owner, _)| owner == document_id);
                    let count = keys.len();
                    let next_keys=if valid {keys.clone()} else {Vec::new()};
                    let next_sequences=sequences.iter().map(|key|*key as u64).collect::<Vec<_>>();
                    let changed=owner.assignments.lock().unwrap().get(&role.bits())!=Some(&next_keys)
                        || owner.sequences.lock().unwrap().get(&role.bits())!=Some(&next_sequences);
                    if changed {document.scope_revision.fetch_add(1,Ordering::AcqRel);}
                    owner
                        .assignments
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(role.bits(), if valid { keys } else { Vec::new() });
                    owner.sequences.lock().unwrap().insert(
                        role.bits(),
                        sequences.iter().map(|key| *key as u64).collect(),
                    );
                    if valid {
                        log::info!(
                            "[ara] renderer assignment role={} regions={count}",
                            role.bits()
                        );
                    } else {
                        log::warn!("[ara] rejected unknown or cross-document renderer assignment");
                    }
                    owner.cancel_preparation();
                    owner.snapshots.iter().for_each(|snapshot|snapshot.clear());
                    let owners=document.renderer_owners();
                    for renderer in &owners {
                        if let Err(error)=renderer.merge_pending_restore(&document) {log::warn!("[ara] instance state unresolved: {error}");}
                    }
                    // 某轨首次获得分配可能恢复其state并推进共享revision；其它轨也需要最新任务。
                    for renderer in owners {renderer.prepare();}
                }
            },
        );
        let supported = ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER;
        let enabled = ExtensionRoles::resolve(known, assigned, supported)?;
        let owned = ExtensionBinding::new_with_renderer_observer(
            generation, known, assigned, supported, observer,
        )?;
        let raw = owned.0.as_raw();
        document.attach(self, owned.1, companion)?;
        *self.document.lock().unwrap() = Some(Arc::downgrade(&document));
        let weak=Arc::downgrade(self);
        if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
            let _=self.preparation.get_or_init(||super::preparation::PreparationQueue::new());
        }
        // 只登记weak供宿主回调排队；后台计算不形成owner自循环。
        *self.prepare_owner.lock().unwrap()=Some(weak);
        let _=self.clock.set(document.clock.clone());
        static NEXT_WRITER:AtomicU64=AtomicU64::new(1);
        self.writer_id.store(NEXT_WRITER.fetch_add(1,Ordering::Relaxed),Ordering::Release);
        *current = Some(owned.0);
        self.roles.store(enabled.bits(),Ordering::Release);
        self.role.store(
            if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                1
            } else {
                2
            },
            Ordering::Release,
        );
        drop(current);
        if enabled.contains(ExtensionRoles::EDITOR_RENDERER) {
            let weak = Arc::downgrade(self);
            match hifishifter_ara_ipc::Server::start("HiFiShifter / REAPER".into(), move |request| {
                weak.upgrade().map(|owner| owner.handle_request(request)).unwrap_or_else(|| hifishifter_ara_ipc::Response {
                    error: Some("plugin instance closed".into()), ..Default::default()
                })
            }) {
                Ok(server) => { log::info!("[ara] GUI channel ready instance={}", server.record().instance_id); *self.channel.lock().unwrap() = Some(server); }
                Err(error) => log::warn!("[ara] GUI channel unavailable: {error}"),
            }
        }
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        self.stop_editor();
        self.stop_channel();
        self.assignments.lock().unwrap().clear();
        self.sequences.lock().unwrap().clear();
        self.document.lock().unwrap().take();
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
    }

    /// 模型线程展开当前 renderer 的显式区域与 editor sequence，并去重，拒绝跨文档身份。
    pub fn assigned_regions(&self) -> Result<Vec<u64>, AraError> {
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or(AraError::InvalidState("document no longer available"))?;
        let role = self.role.load(Ordering::Acquire);
        let mut keys = self
            .assignments
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default();
        let members = document.sequence_regions.lock().unwrap();
        for sequence in sequences {
            keys.extend(
                members
                    .get(&sequence)
                    .ok_or(AraError::InvalidState("unknown assigned sequence"))?
                    .iter()
                    .copied(),
            );
        }
        drop(members);
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty() {
            let (owner, _) = region_owners()
                .lock()
                .unwrap()
                .resolve(&keys)
                .map_err(|_| AraError::InvalidState("invalid assigned region"))?;
            if owner != document.id {
                return Err(AraError::InvalidState(
                    "assigned region belongs to another document",
                ));
            }
        }
        Ok(keys)
    }

    /// 宿主模型callback只撤销/排最新准备任务，不在主线程执行WORLD或PCM混音。
    pub fn prepare(&self) {
        if !self.renders_playback() {return;}
        // 源/assignment撤销在文档事务内清理；此函数只排队，不能在事务外clear旧任务刚发布的音频。
        let weak=self.prepare_owner.lock().unwrap().clone();
        let Some(weak)=weak else {return;};
        match self.preparation.get() {
            Some(Ok(worker))=>if let Err(error)=worker.request(Box::new(move |cancel|{
                weak.upgrade().ok_or("renderer closed".to_owned())?.prepare_job(cancel)
            })) {log::warn!("[ara] preparation unavailable: {error}");},
            Some(Err(error))=>log::warn!("[ara] preparation unavailable: {error}"),
            None=>{},
        }
    }
    /// 调用者持document事务；完整两输出率发布成功后才登记，不额外持有源或快照。
    pub(crate) fn record_prepared(&self,model:u64,edit:u64,epoch:u64,scope:u64,keys:Vec<u64>) {
        *self.prepared.lock().unwrap()=Some(PreparedVersion {model,edit,epoch,scope,keys});
    }
    /// 仅SDK规定UI线程的kOffline setup调用；不在事务锁内等待，不把旧快照当最新版本。
    pub(crate) fn prepare_offline_until(&self,deadline:std::time::Instant)->Result<(),String> {
        if !self.renders_playback() {return Ok(());}
        let document=self.editor_document()?;
        loop {
            {
                let _transaction=document.transaction.lock().unwrap();
                if self.is_closed()||!document.is_alive() {return Err("offline renderer closed".into());}
                if !document.ready.load(Ordering::Acquire) {return Err("offline host model/PCM not ready".into());}
                let keys=self.assigned_regions().map_err(|e|e.to_string())?;
                let version=PreparedVersion {model:document.revision.load(Ordering::Acquire),edit:document.edits.lock().unwrap().revision,
                    epoch:document.render_epoch.load(Ordering::Acquire),scope:document.scope_revision.load(Ordering::Acquire),keys};
                if self.prepared.lock().unwrap().as_ref()==Some(&version)&&self.snapshots.iter().all(|snapshot|snapshot.is_ready()) {return Ok(());}
            }
            if std::time::Instant::now()>=deadline {return Err("offline preparation timed out".into());}
            let worker=self.preparation.get().ok_or("offline preparation worker missing")?.as_ref().map_err(Clone::clone)?;
            if !worker.state().0 {self.prepare();}
            worker.wait_idle_until(deadline)?;
        }
    }
    /// 后台冻结/计算/发布三阶段；撤销标记和全部版本在发布短事务内重新验证。
    fn prepare_job(&self,cancel:Arc<std::sync::atomic::AtomicBool>)->Result<(),String> {
        let Some(document) = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return Err("document closed".into());
        };
        let (model,edit,epoch,scope,keys,input)={
            let _transaction=document.transaction.lock().unwrap();
            if !document.ready.load(Ordering::Acquire) {return Ok(());}
            let edits=document.edits.lock().unwrap().clone();
            let (keys,input)=self.capture_render_input(&document,&edits,edits.revision>0)?;
            (document.revision.load(Ordering::Acquire),edits.revision,document.render_epoch.load(Ordering::Acquire),document.scope_revision.load(Ordering::Acquire),keys,input)
        };
        for publisher in &self.snapshots {publisher.collect_retired();}
        let snapshots=input.render(cancel.clone())?;
        let _transaction=document.transaction.lock().unwrap();
        if cancel.load(Ordering::Acquire) || !document.ready.load(Ordering::Acquire) {return Ok(());}
        if document.revision.load(Ordering::Acquire)!=model || document.edits.lock().unwrap().revision!=edit
            || document.render_epoch.load(Ordering::Acquire)!=epoch||document.scope_revision.load(Ordering::Acquire)!=scope
            || self.assigned_regions().map_err(|e|e.to_string())?!=keys {
            // 跨轨接受新参数不一定有新的model callback；丢弃后补排最新任务，不能永久留空且假idle。
            self.prepare();return Ok(());
        }
        if !self.snapshots.iter().zip(&snapshots).all(|(p,s)|p.has_capacity(s)) {return Err("retired snapshot budget exhausted".into());}
        for (publisher,snapshot) in self.snapshots.iter().zip(snapshots) {publisher.publish(snapshot).map_err(|e|format!("snapshot publish failed: {e:?}"))?;}
        self.record_prepared(model,edit,epoch,scope,keys.clone());
        log::info!("[ara] background snapshot ready role={} revision={edit} model={model} regions={}",self.role.load(Ordering::Relaxed),keys.len());
        Ok(())
    }
}
