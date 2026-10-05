//! 工程编辑权限是同一活ARA文档renderer区域并集，音频输出仍按各renderer原分配隔离。
use crate::render::document::DocumentSession;
use crate::render::ownership::region_owners;
use hifishifter_kernel::state::TimelineState;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

#[derive(Debug,Clone,PartialEq,Eq)]
pub(crate) struct WorkspaceScope {pub regions:BTreeSet<u64>}
impl DocumentSession {
    /// UI专用原始曲率不写进kernel状态或参数权威；普通fade最终声音仍由宿主负责。
    pub(crate) fn decorate_host_fades_locked(&self,payload:&mut serde_json::Value,namespace:&str) {
        let identities=self.clip_ids.lock().unwrap().clone();
        let Some(clips)=payload["clips"].as_array_mut() else {return;};
        for owner in self.renderer_owners() {
            let Some(bound)=owner.host_geometry_metadata_locked(self) else {continue;};
            let Some(id)=identities.get(&bound.region_key) else {continue;};let ui_id=format!("{namespace}{id}");
            let Some(clip)=clips.iter_mut().find(|clip|clip["id"]==ui_id) else {continue;};let g=bound.geometry;
            clip["host_fades"]=serde_json::json!({"curve_mode":match g.fade_axes_new {Some(true)=>"reaper_new",Some(false)=>"legacy",None=>"unknown"},
                "in_curvature":g.fade_in_dir_new,"out_curvature":g.fade_out_dir_new,
                "in_s":g.fade_in_dir2_new,"out_s":g.fade_out_dir2_new});
        }
    }
    /// 无宿主getter的短事务装饰；调用者不得仍持编辑timeline锁。
    pub(crate) fn decorate_host_fades(&self,payload:&mut serde_json::Value,namespace:&str) {
        let _transaction=self.transaction.lock().unwrap();if self.is_alive() {self.decorate_host_fades_locked(payload,namespace);}
    }
    /// 普通手动/自动fade仅投影到原GUI，内核继续消费未烘焙fade的ARA时间线。
    /// 只沿已核对的唯一真实region key，不按轨名/位置猜关联。
    pub(crate) fn project_ui_fades_locked(&self,timeline:&mut TimelineState) {
        let identities=self.clip_ids.lock().unwrap().clone();
        for owner in self.renderer_owners() {
            let Some(bound)=owner.host_geometry_metadata_locked(self) else {continue;};
            let Some(id)=identities.get(&bound.region_key) else {continue;};
            let Some(clip)=timeline.clips.iter_mut().find(|clip|&clip.id==id) else {continue;};let geometry=bound.geometry;
            clip.fade_in_sec=geometry.fade_in_sec;clip.fade_out_sec=geometry.fade_out_sec;
            clip.auto_fade_in_sec=geometry.auto_fade_in_sec;clip.auto_fade_out_sec=geometry.auto_fade_out_sec;
            clip.fade_in_shape=geometry.fade_in_shape;clip.fade_out_shape=geometry.fade_out_shape;
            clip.fade_in_dir=geometry.fade_in_dir;clip.fade_out_dir=geometry.fade_out_dir;
        }
    }
    /// 无PCM复制或host getter，pending曲线也可独立更新可见宿主fade。
    pub(crate) fn ui_fade_projection(&self)->Result<(u64,TimelineState),String> {
        let _transaction=self.transaction.lock().unwrap();let mut timeline=self.workspace_timeline_locked()?;
        self.project_ui_fades_locked(&mut timeline);Ok((self.ui_geometry_revision.load(Ordering::Acquire),timeline))
    }
    /// 非实时短事务读取完整授权scope；零分配只能返回零区域，不能等同全文档。
    pub(crate) fn workspace_scope(&self)->Result<WorkspaceScope,String> {
        let _transaction=self.transaction.lock().unwrap();self.workspace_scope_locked()
    }
    /// 调用方已持transaction；真实model-ref所有权再次校验，拒绝跨文档或已销毁区域。
    pub(crate) fn workspace_scope_locked(&self)->Result<WorkspaceScope,String> {
        if !self.is_alive() {return Err("document closed".into());}
        let mut regions=BTreeSet::new();
        for owner in self.renderer_owners() {regions.extend(owner.assigned_regions().map_err(|e|e.to_string())?);}
        if !regions.is_empty() {
            let keys=regions.iter().copied().collect::<Vec<_>>();
            let (document,_)=region_owners().lock().unwrap().resolve(&keys).map_err(|e|format!("workspace region ownership: {e:?}"))?;
            if document!=self.id {return Err("workspace includes another document".into());}
        }
        Ok(WorkspaceScope {regions})
    }
    /// 原GUI的多轨快照只扩展查看/编辑范围，不更改任何播放renderer的混音归属。
    pub(crate) fn workspace_timeline(&self)->Result<TimelineState,String> {
        let _transaction=self.transaction.lock().unwrap();self.workspace_timeline_locked()
    }
    pub(crate) fn workspace_timeline_locked(&self)->Result<TimelineState,String> {
        if !self.ready.load(Ordering::Acquire) {return Err("host model not ready".into());}
        let scope=self.workspace_scope_locked()?;
        let identities=self.clip_ids.lock().unwrap();
        let clips=scope.regions.iter().map(|key|identities.get(key).cloned().ok_or("workspace clip identity missing"))
            .collect::<Result<BTreeSet<_>,_>>()?;drop(identities);
        let mut timeline=self.timeline.lock().unwrap().clone().ok_or("host timeline unavailable")?;
        timeline.clips.retain(|clip|clips.contains(&clip.id));
        let mut tracks=timeline.clips.iter().map(|clip|clip.track_id.clone()).collect::<BTreeSet<_>>();
        // 原GUI分组根需要保留，父链只沿实际宿主图，未知/循环不能创建虚构轨道。
        for id in tracks.clone() {
            let mut current=id;let mut visited=BTreeSet::new();
            while visited.insert(current.clone()) {
                let track=timeline.tracks.iter().find(|track|track.id==current).ok_or("workspace track identity missing")?;
                let Some(parent)=&track.parent_id else {break;};tracks.insert(parent.clone());current=parent.clone();
            }
            if timeline.tracks.iter().find(|track|track.id==current).is_some_and(|track|track.parent_id.is_some()) {
                return Err("workspace track parent cycle".into());
            }
        }
        timeline.tracks.retain(|track|tracks.contains(&track.id));
        timeline.params_by_root_track.retain(|id,_|tracks.contains(id));
        if !timeline.selected_track_id.as_ref().is_some_and(|id|tracks.contains(id)) {timeline.selected_track_id=timeline.tracks.first().map(|track|track.id.clone());}
        if !timeline.selected_clip_id.as_ref().is_some_and(|id|clips.contains(id)) {timeline.selected_clip_id=timeline.clips.first().map(|clip|clip.id.clone());}
        if let Some(tempo)=self.clock.tempo() {timeline.bpm=tempo;}
        Ok(timeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ara::model::ModelHandle;
    use crate::render::extension::ExtensionOwner;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::ExtensionRoles;
    use std::sync::Arc;

    fn fixture()->(ModelHandle,Vec<Arc<ExtensionOwner>>,Vec<Box<u8>>,Vec<*const ara2_bridge::sys::ARAPlugInExtensionInstance>) {
        let model=ModelHandle::new();let document=model.session();
        *document.timeline.lock().unwrap()=Some(serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"same-name","order":0},{"id":"b","name":"same-name","order":1}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":0.5,"takes":[{"id":"ta","source_path":"shared-source"}]},
                {"id":"cb","name":"B","track_id":"b","start_sec":0,"length_sec":0.5,"takes":[{"id":"tb","source_path":"shared-source"}]}]
        })).unwrap());
        let identities=vec![Box::new(0_u8),Box::new(0_u8)];let mut owners=vec![];let mut interfaces=vec![];
        for (index,id) in ["ca","cb"].into_iter().enumerate() {
            let key=(&*identities[index] as *const u8) as u64;region_owners().lock().unwrap().register(key,document.id,index).unwrap();
            document.clip_ids.lock().unwrap().insert(key,id.into());
            let owner=Arc::new(ExtensionOwner::default());
            let raw=owner.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER,None).unwrap();
            // SAFETY: 区域身份与原生extension保留到fixture销毁。
            unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
            owners.push(owner);interfaces.push(raw);
        }
        document.ready.store(true,Ordering::Release);(model,owners,identities,interfaces)
    }
    /// 同名/同源不能被合成一条轨，多个renderer也不能把同一个区域重复显示。
    #[test]
    fn shared_source_tracks_form_one_deduplicated_document_workspace() {
        let (model,owners,ids,_)=fixture();let document=model.session();
        let (second,_other_owners,_other_ids,_other_interfaces)=fixture();assert_ne!(document.id,second.session().id);
        assert!(document.workspace_scope().unwrap().regions.is_disjoint(&second.session().workspace_scope().unwrap().regions));
        let extra=Arc::new(ExtensionOwner::default());let raw=extra.bind_to_document(document.clone(),ApiGeneration::V2Final,ExtensionRoles::all(),ExtensionRoles::PLAYBACK_RENDERER,None).unwrap();
        let key=(&*ids[0] as *const u8) as u64;
        unsafe {let ext=&*raw;((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
        assert_eq!(document.workspace_scope().unwrap().regions.len(),2);
        let timeline=document.workspace_timeline().unwrap();assert_eq!(timeline.tracks.len(),2);assert_eq!(timeline.clips.len(),2);
        assert_eq!(timeline.tracks[0].id,"a");assert_eq!(timeline.tracks[1].id,"b");
        owners[0].stop_editor();extra.stop_editor();
        let left=document.workspace_timeline().unwrap();assert_eq!(left.tracks.len(),1);assert_eq!(left.tracks[0].id,"b");
    }
    /// 宿主真正移除assignment推进scope版本，空scope不能回退成全文档；关闭doc明确拒绝。
    #[test]
    fn assignments_and_component_close_revoke_workspace_scope_without_global_fallback() {
        let (model,owners,ids,interfaces)=fixture();let document=model.session();let before=document.scope_revision.load(Ordering::Acquire);
        let key=(&*ids[0] as *const u8) as u64;
        unsafe {let ext=&*interfaces[0];((*ext.playbackRendererInterface).removePlaybackRegion.unwrap())(ext.playbackRendererRef,key as *mut _);}
        assert!(document.scope_revision.load(Ordering::Acquire)>before);
        owners[1].stop_editor();let empty=document.workspace_timeline().unwrap();assert!(empty.clips.is_empty());assert!(empty.tracks.is_empty());
        document.close();assert!(document.workspace_scope().is_err());
    }
}
