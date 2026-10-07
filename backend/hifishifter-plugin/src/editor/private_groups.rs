//! 插件私有轨道参数分组：以真实轨道GUID保存关系，不读写REAPER folder、路由或轨道排序。
use hifishifter_kernel::state::{TimelineState,TrackParamsState};
use serde::{Deserialize,Serialize};
use std::collections::{BTreeMap,BTreeSet};
use super::parameter_atlas::{parameter_curves,set_curve,pad};

#[derive(Clone,Default,Debug,Serialize,Deserialize,PartialEq)]
pub(crate) struct TrackGroups {pub nodes:BTreeMap<String,GroupNode>}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub(crate) struct GroupNode {
    pub parent:Option<String>,pub order:i32,
    #[serde(default,skip_serializing_if="Option::is_none")] pub compose:Option<bool>,
    #[serde(default,skip_serializing_if="Option::is_none")] pub algo:Option<hifishifter_kernel::state::PitchAnalysisAlgo>,
}
impl TrackGroups {
    pub fn is_empty(&self)->bool {self.nodes.is_empty()}
    /// state必须有界、真实GUID形状且无环；不存在于当前窗口的轨道不被凭空创建。
    pub fn validate(&self)->Result<(),String> {
        if self.nodes.len()>10000 {return Err("private track group budget exceeded".into());}
        for (guid,node) in &self.nodes {
            if guid.len()!=38||!guid.starts_with('{')||!guid.ends_with('}')||!(0..=10000).contains(&node.order) {return Err("invalid private track group identity/order".into());}
            if node.algo.as_ref().is_some_and(|algo|matches!(algo,hifishifter_kernel::state::PitchAnalysisAlgo::Unknown|hifishifter_kernel::state::PitchAnalysisAlgo::VocalShifterVslib)) {return Err("unsupported private track group algorithm".into());}
            let mut visited=BTreeSet::new();let mut cursor=Some(guid.as_str());
            while let Some(id)=cursor {
                if !visited.insert(id) {return Err("private track group cycle".into());}
                cursor=self.nodes.get(id).ok_or("private track group parent missing")?.parent.as_deref();
            }
        }Ok(())
    }
    /// 新进来的轨道始终根级；仅应用用户在本插件存下的关系，缺失父级时自动显示为根。
    pub fn apply(&self,timeline:&mut TimelineState,aliases:&BTreeMap<String,String>) {
        let ids=timeline.tracks.iter().map(|track|track.id.clone()).collect::<BTreeSet<_>>();
        for track in &mut timeline.tracks {
            track.parent_id=None;
            if let Some(node)=aliases.iter().find(|(_,id)|**id==track.id).and_then(|(guid,_)|self.nodes.get(guid)) {
                track.parent_id=node.parent.as_ref().and_then(|guid|aliases.get(guid)).filter(|id|ids.contains(*id)).cloned();
                track.order=node.order;
                if let Some(compose)=node.compose {track.compose_enabled=compose;}
                if let Some(algo)=&node.algo {track.pitch_analysis_algo=algo.clone();}
            }
        }
        timeline.tracks.sort_by_key(|track|track.order);
    }
    /// 合成算法/开关遵循组根，但保留物理轨道根和renderer归属，不把folder变成混音路由。
    pub fn processing_settings(&self,timeline:&mut TimelineState,aliases:&BTreeMap<String,String>) {
        let mut grouped=timeline.clone();self.apply(&mut grouped,aliases);
        for track in &mut timeline.tracks {
            if let Some(root)=grouped.resolve_root_track_id(&track.id).and_then(|id|grouped.tracks.iter().find(|track|track.id==id)) {
                track.compose_enabled=root.compose_enabled;track.pitch_analysis_algo=root.pitch_analysis_algo.clone();
            }
        }
    }
    /// 从原App拖动结果捕获GUID关系；名称/会话序号绝不用于冷恢复关联。
    pub fn capture(timeline:&TimelineState,aliases:&BTreeMap<String,String>)->Result<Self,String> {
        let by_id=aliases.iter().map(|(guid,id)|(id.as_str(),guid.clone())).collect::<BTreeMap<_,_>>();
        let mut groups=Self::default();
        for track in &timeline.tracks {
            let guid=by_id.get(track.id.as_str()).ok_or("track group requires verified host track GUID")?;
            let parent=track.parent_id.as_deref().map(|id|by_id.get(id).cloned().ok_or("unknown private track group parent")).transpose()?;
            groups.nodes.insert(guid.clone(),GroupNode {parent,order:track.order,compose:Some(track.compose_enabled),algo:Some(track.pitch_analysis_algo.clone())});
        }groups.validate()?;Ok(groups)
    }
    /// 空父轨的处理设置也落在私有GUID记录中；参数提交不能顺带改变父级或排序。
    pub fn with_processing_from(&self,timeline:&TimelineState,aliases:&BTreeMap<String,String>)->Result<Self,String> {
        let mut result=self.clone();
        for (guid,id) in aliases {
            if let (Some(node),Some(track))=(result.nodes.get_mut(guid),timeline.tracks.iter().find(|track|&track.id==id)) {
                node.compose=Some(track.compose_enabled);node.algo=Some(track.pitch_analysis_algo.clone());
            }
        }result.validate()?;Ok(result)
    }
}

/// 只把分组面板相对旧投影的参数delta展开给物理轨道；未编辑的子轨曲线不被父轨覆盖。
pub(crate) fn expand_parameter_changes(client:&TimelineState,before_grouped:&TimelineState,physical:&mut TimelineState)->Result<(),String> {
    let before=physical.clone();
    for track in &mut physical.tracks {
        let root=client.resolve_root_track_id(&track.id).ok_or("unknown private parameter root")?;
        if let Some(edited)=client.tracks.iter().find(|candidate|candidate.id==track.id) {
            track.volume=edited.volume;track.muted=edited.muted;track.solo=edited.solo;
            track.compose_enabled=edited.compose_enabled;track.pitch_analysis_algo=edited.pitch_analysis_algo.clone();
        }
        if root!=track.id {
            if let (Some(next),Some(old))=(client.tracks.iter().find(|t|t.id==root),before_grouped.tracks.iter().find(|t|t.id==root)) {
                if next.compose_enabled!=old.compose_enabled {track.compose_enabled=next.compose_enabled;}
                if next.pitch_analysis_algo!=old.pitch_analysis_algo {track.pitch_analysis_algo=next.pitch_analysis_algo.clone();}
            }
        }
        let Some(next)=client.params_by_root_track.get(&root) else {continue;};
        let old=before_grouped.params_by_root_track.get(&root).cloned().unwrap_or_else(||TrackParamsState {frame_period_ms:next.frame_period_ms,..Default::default()});
        let mut own=before.params_by_root_track.get(&track.id).cloned().unwrap_or_else(||TrackParamsState {frame_period_ms:next.frame_period_ms,..Default::default()});
        if own.frame_period_ms!=next.frame_period_ms||old.frame_period_ms!=next.frame_period_ms {return Err("private group parameter frame periods differ".into());}
        let keys=parameter_curves(next).into_iter().chain(parameter_curves(&old)).map(|(key,_)|key).collect::<BTreeSet<_>>();
        for key in keys {
            let curve=|params:&TrackParamsState|parameter_curves(params).into_iter().find(|(name,_)|name==&key).map(|(_,values)|values.to_vec()).unwrap_or_default();
            let incoming=curve(next);let previous=curve(&old);
            if incoming==previous {continue;}
            if incoming.is_empty() {set_curve(&mut own,&key,Vec::new());continue;}
            let mut values=curve(&own);values.resize(values.len().max(incoming.len()).max(previous.len()),pad(&key));
            for (frame,value) in values.iter_mut().enumerate() {
                let new=incoming.get(frame).copied().unwrap_or_else(||pad(&key));
                if new!=previous.get(frame).copied().unwrap_or_else(||pad(&key)) {*value=new;}
            }set_curve(&mut own,&key,values);
        }
        for key in next.extra_params.keys().chain(old.extra_params.keys()).collect::<BTreeSet<_>>() {
            if next.extra_params.get(key)!=old.extra_params.get(key) {
                if let Some(value)=next.extra_params.get(key) {own.extra_params.insert(key.clone(),*value);} else {own.extra_params.remove(key);}
            }
        }
        if next.pitch_edit_user_modified!=old.pitch_edit_user_modified||(next.pitch_edit!=old.pitch_edit&&next.pitch_edit_user_modified) {own.pitch_edit_user_modified=next.pitch_edit_user_modified;}
        physical.params_by_root_track.insert(track.id.clone(),own);
    }Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 新会话ID变化仍沿GUID恢复；宿主原有父级不能混进私有分组，新轨始终根级。
    #[test]
    fn goal_feedback_private_group_guid_restore_and_delta_expansion() {
        let mut flat:TimelineState=serde_json::from_value(serde_json::json!({"tracks":[
            {"id":"a","name":"A","order":0},{"id":"b","name":"B","order":1},{"id":"new","name":"new","order":2,"parent_id":"a"}],"clips":[],"bpm":120,"project_sec":1})).unwrap();
        let a="{11111111-1111-1111-1111-111111111111}".to_owned();let b="{22222222-2222-2222-2222-222222222222}".to_owned();
        let aliases=BTreeMap::from([(a.clone(),"a".into()),(b.clone(),"b".into())]);
        let groups=TrackGroups {nodes:BTreeMap::from([(a.clone(),GroupNode {parent:None,order:0,compose:None,algo:None}),(b.clone(),GroupNode {parent:Some(a.clone()),order:1,compose:None,algo:None})])};
        groups.validate().unwrap();
        for (id,values) in [("a",vec![10.,11.]),("b",vec![20.,25.])] {
            let mut params=TrackParamsState {frame_period_ms:5.,..Default::default()};params.extra_curves.insert("hifigan_tension".into(),values);
            flat.params_by_root_track.insert(id.into(),params);
        }
        let mut grouped=flat.clone();groups.apply(&mut grouped,&aliases);assert_eq!(grouped.tracks[1].parent_id.as_deref(),Some("a"));assert!(grouped.tracks[2].parent_id.is_none());
        let mut edited=grouped.clone();edited.params_by_root_track.get_mut("a").unwrap().extra_curves.insert("hifigan_tension".into(),vec![65.,11.]);
        expand_parameter_changes(&edited,&grouped,&mut flat).unwrap();
        assert_eq!(flat.params_by_root_track["b"].extra_curves["hifigan_tension"],vec![65.,25.]);
        let decoded:TrackGroups=serde_json::from_slice(&serde_json::to_vec(&groups).unwrap()).unwrap();
        let mut cold=grouped.clone();cold.tracks[0].id="cold-a".into();cold.tracks[1].id="cold-b".into();
        decoded.apply(&mut cold,&BTreeMap::from([(a,"cold-a".into()),(b,"cold-b".into())]));
        assert_eq!(cold.tracks[1].parent_id.as_deref(),Some("cold-a"));
        let mut cycle=decoded;cycle.nodes.values_mut().next().unwrap().parent=Some("{22222222-2222-2222-2222-222222222222}".into());assert!(cycle.validate().is_err());
    }
}
