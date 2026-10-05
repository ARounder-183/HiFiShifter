//! 区域参数的源坐标权威；原GUI项目帧只是投影，宿主移动/拉伸不重新采样写坏原始曲线。
//! 身份来自真实ARA region→modification/source边；不从名字/文件路径/会话序号猜冷恢复归属。

use crate::ara::time_map::SourceTimeMap;
use hifishifter_kernel::state::{TimelineState,TrackParamsState};
use serde::{Serialize,Deserialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub(crate) struct RegionIdentity {
    #[serde(skip)] pub key:u64,
    pub source:String,
    pub modification:String,
}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub(crate) struct RegionGeometry {
    pub project_start:f64,pub project_duration:f64,
    pub source_start:f64,pub source_duration:f64,
}
impl RegionGeometry {
    /// JSON f64往返可能偏一ULP；只忽略数值舍入，不把真实移动/倍率变化当相同布局。
    fn equivalent(&self,other:&Self)->bool {
        [(self.project_start,other.project_start),(self.project_duration,other.project_duration),
            (self.source_start,other.source_start),(self.source_duration,other.source_duration)].into_iter().all(|(a,b)|near(a,b))
    }
    /// 几何只接受已经确认的ARA秒域；未确认raw marker不得在此成为渲染权威。
    pub fn map(&self)->Result<SourceTimeMap,String> {
        SourceTimeMap::linear(self.project_start,self.project_duration,self.source_start,self.source_duration)
    }
}
#[derive(Clone,Debug,Serialize,Deserialize)]
struct SourceCurve {
    basis:RegionGeometry,first_frame:usize,frame_ms:f64,
    #[serde(with="arc_values")] values:Arc<Vec<f32>>,
    #[serde(skip)] reservation:Option<Arc<crate::render::budget::Reservation>>,
}
mod arc_values {
    use super::*;
    pub fn serialize<S:serde::Serializer>(value:&Arc<Vec<f32>>,serializer:S)->Result<S::Ok,S::Error> {value.as_ref().serialize(serializer)}
    pub fn deserialize<'de,D:serde::Deserializer<'de>>(deserializer:D)->Result<Arc<Vec<f32>>,D::Error> {
        let values=Vec::<f32>::deserialize(deserializer)?;
        if values.len()>1_000_000||values.iter().any(|v|!v.is_finite()||v.abs()>10000.) {
            return Err(serde::de::Error::custom("invalid source parameter samples"));
        }Ok(Arc::new(values))
    }
}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub(crate) struct RegionParameters {
    pub identity:RegionIdentity,pub root:String,pub current:RegionGeometry,
    /// 历史父范围保留供恢复，但不能与最近活图的真正父候选一起制造嵌套拆分歧义。
    #[serde(default="default_live")] live:bool,
    template:TrackParamsState,curves:BTreeMap<String,SourceCurve>,
}
fn default_live()->bool {true}
#[derive(Clone,Default,Debug,Serialize,Deserialize)]
pub(crate) struct ParameterAtlas {
    pub regions:BTreeMap<String,RegionParameters>,
}
impl ParameterAtlas {
    pub fn is_empty(&self)->bool {self.regions.is_empty()}
    /// 同一源曲线可由历史/区域共享，额度按reservation身份去重，而不是按序列化值重复计数。
    pub fn accounted_curve_bytes(&self)->usize {
        let mut seen=std::collections::BTreeSet::new();
        self.regions.values().flat_map(|region|region.curves.values()).filter_map(|curve|curve.reservation.as_ref())
            .filter(|reservation|seen.insert(Arc::as_ptr(reservation) as usize)).map(|reservation|reservation.bytes()).sum()
    }
    /// 冷绑定布局完全相同时保留GUI原整轨数组（含无音频处编辑），音频仍用区域源basis。
    pub fn same_layout(&self,other:&Self)->bool {
        self.regions.len()==other.regions.len()&&self.regions.values().all(|old|other.regions.values().any(|new|
            old.identity.source==new.identity.source&&old.identity.modification==new.identity.modification&&old.root==new.root&&old.current.equivalent(&new.current)))
    }
    /// 有界state解码后为源曲线重新登记共享512MiB预算；clone共享Arc收费，不按renderer重复收费。
    pub fn reserve_restored(mut self)->Result<Self,String> {
        self.validate()?;
        for record in self.regions.values_mut() {for curve in record.curves.values_mut() {if curve.reservation.is_none() {
            curve.reservation=Some(Arc::new(crate::render::budget::global_budget().reserve(curve.values.len()*4).ok_or("source parameter memory budget exceeded")?));
        }}}Ok(self)
    }
    /// 只换活图投影与真实key，源basis保持；拆分同modification的新region可继承唯一父范围。
    pub fn follow_geometry(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>)->Result<Self,String> {
        let mut followed=self.clone();
        for record in followed.regions.values_mut() {record.live=false;}
        for clip in &timeline.clips {
            let identity=identities.get(&clip.id).ok_or("missing actual ARA parameter identity")?;
            let root=timeline.resolve_root_track_id(&clip.track_id).ok_or("unknown parameter root")?;let geometry=geometry(clip)?;
            if let Some(previous)=self.find(identity,&root,&geometry)? {
                let mut record=previous.clone();record.identity=identity.clone();record.root=root;record.current=geometry;record.live=true;followed.regions.insert(clip.id.clone(),record);
            }
        }followed.validate()?;Ok(followed)
    }
    /// 冷恢复只在本组件真实范围内绑定；重复持久身份且源窗口相同仍拒绝，不能按旧key猜。
    pub fn rebind(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>)->Result<Self,String> {
        let mut bound=Self::default();
        for record in self.regions.values() {
            let candidates=timeline.clips.iter().filter(|clip|identities.get(&clip.id).is_some_and(|id|
                id.source==record.identity.source&&id.modification==record.identity.modification)
                &&timeline.resolve_root_track_id(&clip.track_id).as_deref()==Some(record.root.as_str())).collect::<Vec<_>>();
            let matches=if candidates.len()==1 {candidates} else {candidates.into_iter().filter(|clip|geometry(clip).is_ok_and(|geometry|
                near(geometry.source_start,record.current.source_start)&&near(geometry.source_duration,record.current.source_duration))).collect()};
            let [clip]=matches.as_slice() else {return Err("ARA source parameter identity missing or ambiguous in assigned scope".into());};
            if bound.regions.contains_key(&clip.id) {return Err("ARA source parameter records collapse into one region".into());}
            let mut record=record.clone();record.identity=identities[&clip.id].clone();record.current=geometry(clip)?;record.live=true;bound.regions.insert(clip.id.clone(),record);
        }bound.validate()?;Ok(bound)
    }
    /// GUI仍用原track网格：显示投影可选重叠区，音频始终保留每region自己的参数。
    pub fn project_roots(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>)->Result<BTreeMap<String,TrackParamsState>,String> {
        let clips=self.project(timeline,identities)?;let mut roots=BTreeMap::<String,TrackParamsState>::new();
        let mut order=timeline.clips.iter().collect::<Vec<_>>();order.sort_by_key(|clip|Some(&clip.id)==timeline.selected_clip_id.as_ref());
        for clip in order {
            let Some(params)=clips.get(&clip.id) else {continue;};let root=timeline.resolve_root_track_id(&clip.track_id).ok_or("unknown parameter root")?;
            let geometry=geometry(clip)?;let (begin,end)=frame_range(&geometry,params.frame_period_ms)?;
            let entry=roots.entry(root).or_insert_with(||{let mut entry=params.clone();clear_curves(&mut entry);entry});
            if entry.frame_period_ms!=params.frame_period_ms {return Err("Conflict: source parameter frame periods differ within root".into());}
            for (key,values) in parameter_curves(params) {
                if values.is_empty() {continue;}
                let mut merged=parameter_curves(entry).into_iter().find(|(name,_)|*name==key).map(|(_,v)|v.to_vec()).unwrap_or_default();
                merged.resize(end.max(merged.len().saturating_sub(1))+1,pad(&key));
                merged[begin..=end].copy_from_slice(&values[begin..=end]);set_curve(entry,&key,merged);
            }
        }Ok(roots)
    }
    /// 接受当前投影上的真实编辑；未改动的曲线继续保留原始源坐标basis。
    pub fn capture_changes(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>,previous_view:&BTreeMap<String,TrackParamsState>)->Result<Self,String> {
        self.capture_inner(timeline,identities,Some(previous_view))
    }
    pub fn capture(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>)->Result<Self,String> {
        self.capture_inner(timeline,identities,None)
    }
    /// 只把相对旧GUI投影的delta送回源authority；重叠选区外保留该region自己的值。
    fn capture_inner(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>,previous_view:Option<&BTreeMap<String,TrackParamsState>>)->Result<Self,String> {
        let mut candidate=self.clone();
        for clip in &timeline.clips {
            let identity=identities.get(&clip.id).ok_or("missing actual ARA parameter identity")?;
            if identity.key==0||identity.source.is_empty()||identity.modification.is_empty()
                ||clip.source_path.as_deref()!=Some(identity.source.as_str()) {return Err("invalid ARA parameter identity".into());}
            let root=timeline.resolve_root_track_id(&clip.track_id).ok_or("unknown parameter root")?;
            let Some(params)=timeline.params_by_root_track.get(&root) else {candidate.regions.remove(&clip.id);continue;};
            let geometry=geometry(clip)?;let (begin,end)=frame_range(&geometry,params.frame_period_ms)?;
            let previous=self.find(identity,&root,&geometry)?;
            let mut curves=BTreeMap::new();
            for (key,values) in parameter_curves(params) {
                if values.is_empty() {continue;}
                let old=previous.and_then(|previous|previous.curves.get(&key));
                let view=previous_view.and_then(|view|view.get(&root)).filter(|view|view.frame_period_ms==params.frame_period_ms);
                let old_view=view.and_then(|view|parameter_curves(view).into_iter().find(|(name,_)|*name==key).map(|(_,values)|values));
                if let (Some(old),Some(old_view))=(old,old_view) {if values==old_view {curves.insert(key,old.clone());continue;}}
                let adjusted=if let (Some(old),Some(old_view))=(old,old_view) {
                    let mut region=old.project(&geometry,&key,params.frame_period_ms)?;
                    let selected=timeline.selected_clip_id.as_ref().and_then(|id|timeline.clips.iter().find(|selected|selected.id==*id))
                        .filter(|selected|timeline.resolve_root_track_id(&selected.track_id).as_deref()==Some(root.as_str())&&selected.id!=clip.id);
                    for frame in begin..=end {
                        let incoming=values.get(frame).copied().unwrap_or_else(||pad(&key));let prior=old_view.get(frame).copied().unwrap_or_else(||pad(&key));
                        let time=frame as f64*params.frame_period_ms/1000.;
                        let covered=selected.is_some_and(|selected|time>=selected.start_sec&&time<=selected.start_sec+selected.length_sec);
                        if incoming!=prior&&!covered {region[frame]=incoming;}
                    }Some(region)
                } else {None};
                let values=adjusted.as_deref().unwrap_or(values);
                let unchanged=old.is_some_and(|old|old.project(&geometry,&key,params.frame_period_ms)
                    .is_ok_and(|projected|(begin..=end).all(|frame|projected[frame]==values.get(frame).copied().unwrap_or_else(||pad(&key)))));
                let curve=if unchanged {old.unwrap().clone()} else {
                    if values.len()>1_000_000||values.iter().any(|v|!v.is_finite()||v.abs()>10000.) {return Err("invalid source parameter samples".into());}
                    let first=begin.saturating_sub(1);let last=(end+2).min(values.len());
                    let bytes=last.saturating_sub(first)*4;
                    let reservation=crate::render::budget::global_budget().reserve(bytes).ok_or("source parameter memory budget exceeded")?;
                    let samples=if first<last {values[first..last].to_vec()} else {Vec::new()};
                    SourceCurve {basis:geometry.clone(),first_frame:first,frame_ms:params.frame_period_ms,values:Arc::new(samples),reservation:Some(Arc::new(reservation))}
                };curves.insert(key,curve);
            }
            let mut template=params.clone();clear_curves(&mut template);
            candidate.regions.insert(clip.id.clone(),RegionParameters {identity:identity.clone(),root,current:geometry,live:true,template,curves});
        }
        candidate.validate()?;Ok(candidate)
    }
    /// 仅更新current几何，不把投影结果当新源数据；返回每个clip独立参数供worker冻结。
    pub fn project(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>)->Result<BTreeMap<String,TrackParamsState>,String> {
        self.project_in_domain(timeline,identities,false)
    }
    /// 音频直接投影到region局部零点，不经GUI项目网格再次插值；纯移动保持相同合成参数。
    pub fn project_local(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>)->Result<BTreeMap<String,TrackParamsState>,String> {
        self.project_in_domain(timeline,identities,true)
    }
    fn project_in_domain(&self,timeline:&TimelineState,identities:&BTreeMap<String,RegionIdentity>,local:bool)->Result<BTreeMap<String,TrackParamsState>,String> {
        self.validate()?;let mut projected=BTreeMap::new();
        for clip in &timeline.clips {
            let identity=identities.get(&clip.id).ok_or("missing actual ARA parameter identity")?;
            let root=timeline.resolve_root_track_id(&clip.track_id).ok_or("unknown parameter root")?;let mut geometry=geometry(clip)?;
            let Some(record)=self.find(identity,&root,&geometry)? else {continue;};
            if local {geometry.project_start=0.;}
            let mut params=record.template.clone();let frame_ms=params.frame_period_ms;
            for (key,curve) in &record.curves {set_curve(&mut params,key,curve.project(&geometry,key,frame_ms)?);}
            params.pitch_orig_key=None;params.dyn_orig_key=None;projected.insert(clip.id.clone(),params);
        }Ok(projected)
    }
    /// 原region key优先；拆分新key只能沿同modification/source与同root的唯一父源范围继承。
    fn find(&self,identity:&RegionIdentity,root:&str,geometry:&RegionGeometry)->Result<Option<&RegionParameters>,String> {
        let related=self.regions.values().filter(|record|record.identity.source==identity.source&&record.identity.modification==identity.modification);
        if identity.key!=0 {if let Some(record)=related.clone().find(|record|record.live&&record.identity.key==identity.key) {return Ok(Some(record));}}
        let candidates=related.filter(|record|record.identity.key!=0&&record.root==root
            &&geometry.source_start<record.current.source_start+record.current.source_duration
            &&geometry.source_start+geometry.source_duration>record.current.source_start).collect::<Vec<_>>();
        let live=candidates.iter().copied().filter(|record|record.live).collect::<Vec<_>>();
        let candidates=if live.is_empty() {candidates} else {live};
        match candidates.as_slice() {[]=>Ok(None),[record]=>Ok(Some(*record)),_=>Err("Conflict: ambiguous source parameter ancestry".into())}
    }
    /// 当前先保持原安全数量边界；source basis可序列化，但反序列化后也必须复验。
    pub fn validate(&self)->Result<(),String> {
        if self.regions.len()>16384 {return Err("parameter atlas region budget exceeded".into());}
        let mut bytes=0_usize;
        for record in self.regions.values() {
            if record.identity.source.is_empty()||record.identity.modification.is_empty()||record.root.is_empty()
                ||!record.template.frame_period_ms.is_finite()||record.template.frame_period_ms<0.1||record.template.frame_period_ms>1000.
                ||record.template.extra_params.values().any(|value|!value.is_finite()) {return Err("invalid source parameter record".into());}
            record.current.map()?;
            for curve in record.curves.values() {
                curve.basis.map()?;frame_range(&curve.basis,curve.frame_ms)?;
                if curve.first_frame>=1_000_000||curve.first_frame.checked_add(curve.values.len()).is_none_or(|end|end>1_000_000)
                    ||curve.values.iter().any(|v|!v.is_finite()||v.abs()>10000.) {return Err("invalid source parameter samples".into());}
                bytes=bytes.checked_add(curve.values.len()*4).ok_or("parameter atlas byte overflow")?;
                if bytes>64*1024*1024 {return Err("parameter atlas exceeds 64MiB".into());}
            }
        }Ok(())
    }
}

/// 用本体真实消费窗口而不是take媒体全长；宿主ARA已经把总倍率投影到active take。
fn geometry(clip:&hifishifter_kernel::state::Clip)->Result<RegionGeometry,String> {
    if clip.reversed {return Err("reverse parameter map unsupported".into());}
    let value=RegionGeometry {project_start:clip.start_sec,project_duration:clip.length_sec,
        source_start:clip.source_start_sec,source_duration:clip.length_sec*clip.playback_rate as f64};value.map()?;Ok(value)
}
fn near(a:f64,b:f64)->bool {a.is_finite()&&b.is_finite()&&(a-b).abs()<=8.*f64::EPSILON*a.abs().max(b.abs()).max(1.)}
fn frame_range(geometry:&RegionGeometry,frame_ms:f64)->Result<(usize,usize),String> {
    if !frame_ms.is_finite()||frame_ms<0.1||frame_ms>1000.||geometry.project_start<0. {return Err("invalid source parameter frame domain".into());}
    let first=(geometry.project_start*1000./frame_ms).floor();let last=((geometry.project_start+geometry.project_duration)*1000./frame_ms).ceil();
    if last>=1_000_000.||!last.is_finite() {return Err("source parameter projection frame budget exceeded".into());}Ok((first as usize,last as usize))
}
fn pad(key:&str)->f32 {if key=="extra:volume"||key=="extra:breath_gain" {1.} else if key=="extra:dyn" {-1.} else {0.}}
fn parameter_curves(params:&TrackParamsState)->Vec<(String,&[f32])> {
    let mut curves=vec![("pitch_orig".into(),params.pitch_orig.as_slice()),("pitch_edit".into(),params.pitch_edit.as_slice()),
        ("tension_orig".into(),params.tension_orig.as_slice()),("tension_edit".into(),params.tension_edit.as_slice()),("dyn_orig".into(),params.dyn_orig.as_slice())];
    curves.extend(params.extra_curves.iter().map(|(key,values)|(format!("extra:{key}"),values.as_slice())));curves
}
fn clear_curves(params:&mut TrackParamsState) {params.pitch_orig.clear();params.pitch_edit.clear();params.tension_orig.clear();params.tension_edit.clear();params.dyn_orig.clear();params.extra_curves.clear();}
fn set_curve(params:&mut TrackParamsState,key:&str,values:Vec<f32>) {
    match key {"pitch_orig"=>params.pitch_orig=values,"pitch_edit"=>params.pitch_edit=values,"tension_orig"=>params.tension_orig=values,
        "tension_edit"=>params.tension_edit=values,"dyn_orig"=>params.dyn_orig=values,_=>{if let Some(key)=key.strip_prefix("extra:") {params.extra_curves.insert(key.into(),values);}}}
}
impl SourceCurve {
    /// 只生成项目网格投影，原values/basis保持不可变，连续宿主变换不累积插值损失。
    fn project(&self,geometry:&RegionGeometry,key:&str,frame_ms:f64)->Result<Vec<f32>,String> {
        let (begin,end)=frame_range(geometry,frame_ms)?;let current=geometry.map()?;let basis=self.basis.map()?;
        let mut result=vec![pad(key);end+1];
        for frame in begin..=end {
            let project=frame as f64*frame_ms/1000.;
            let Some(previous)=current.previous_project_time(&basis,project) else {continue;};
            let index=previous*1000./self.frame_ms-self.first_frame as f64;
            if !index.is_finite()||index<0.||self.values.is_empty() {continue;}
            let lo=(index.floor() as usize).min(self.values.len()-1);let hi=(lo+1).min(self.values.len()-1);
            let fraction=(index-lo as f64).clamp(0.,1.) as f32;let a=self.values[lo];let b=self.values[hi];
            result[frame]=if key=="pitch_edit"&&(a<=0.||b<=0.) {a.max(b).max(0.)} else {a+(b-a)*fraction};
        }Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 连续拆分只沿最近活图的唯一父范围，保留历史父记录但不让它阻塞第二次拆分。
    #[test]
    fn nested_splits_follow_live_ancestry_without_losing_history_or_guessing_overlaps() {
        let original=edited();let atlas=ParameterAtlas::default().capture(&original,&identities()).unwrap();
        let mut first=host(1.,0.5,0.,0.5);first.clips[0].id="left".into();
        let mut right=host(1.5,0.5,0.5,0.5).clips.remove(0);right.id="right".into();first.clips.push(right);first.project_sec=2.;
        let ids=BTreeMap::from([("left".into(),RegionIdentity {key:42,source:"source".into(),modification:"mod".into()}),
            ("right".into(),RegionIdentity {key:43,source:"source".into(),modification:"mod".into()})]);
        let followed=atlas.follow_geometry(&first,&ids).unwrap();assert_eq!(followed.regions.len(),3);
        assert!(!followed.regions["clip"].live);assert!(followed.regions["left"].live);
        let mut next=host(1.,0.25,0.,0.25);next.clips[0].id="quarter-a".into();
        let mut second=host(1.25,0.25,0.25,0.25).clips.remove(0);second.id="quarter-b".into();next.clips.push(second);next.clips.push(first.clips[1].clone());next.project_sec=2.;
        let mut next_ids=ids.clone();next_ids.remove("left");
        next_ids.insert("quarter-a".into(),RegionIdentity {key:44,source:"source".into(),modification:"mod".into()});
        next_ids.insert("quarter-b".into(),RegionIdentity {key:45,source:"source".into(),modification:"mod".into()});
        let nested=followed.follow_geometry(&next,&next_ids).unwrap();let local=nested.project_local(&next,&next_ids).unwrap();
        assert_eq!(local["quarter-a"].pitch_edit,&[60.,61.]);assert_eq!(local["quarter-b"].pitch_edit,&[61.,62.]);
        assert_eq!(local["right"].pitch_edit,&[62.,63.,64.]);assert!(!nested.regions["left"].live);
        // 两条真正活且重叠的不同编辑仍有歧义，不能选最小范围/最后写入来掩盖。
        let mut ambiguous=followed.clone();ambiguous.regions.get_mut("clip").unwrap().live=true;
        assert!(ambiguous.follow_geometry(&next,&next_ids).unwrap_err().contains("ambiguous"));
    }
    fn host(start:f64,duration:f64,source_start:f64,source_duration:f64)->TimelineState {
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":start+duration,
            "clips":[{"id":"clip","track_id":"a","name":"voice","start_sec":start,"length_sec":duration,
                "takes":[{"id":"take","source_path":"source","source_start_sec":source_start,
                    "source_end_sec":source_start+source_duration,"playback_rate":source_duration/duration}]}]
        })).unwrap();timeline.clips[0].normalize_takes();timeline
    }
    fn identities()->BTreeMap<String,RegionIdentity> {BTreeMap::from([("clip".into(),RegionIdentity {key:41,source:"source".into(),modification:"mod".into()})])}
    fn edited()->TimelineState {
        let mut timeline=host(1.,1.,0.,1.);
        timeline.params_by_root_track.insert("a".into(),TrackParamsState {frame_period_ms:250.,pitch_edit_user_modified:true,
            pitch_orig:vec![0.,0.,0.,0.,57.,57.,57.,57.,57.],pitch_edit:vec![0.,0.,0.,0.,60.,61.,62.,63.,64.],
            extra_curves:std::collections::HashMap::from([("volume".into(),vec![1.,1.,1.,1.,0.5,0.6,0.7,0.8,0.9])]),..Default::default()});
        timeline
    }
    #[test]
    fn audio_local_projection_is_identical_after_non_grid_movement_and_maps_stretch_once() {
        let ids=identities();let atlas=ParameterAtlas::default().capture(&edited(),&ids).unwrap();
        let original=atlas.project_local(&edited(),&ids).unwrap();
        let moved=atlas.project_local(&host(512.013,1.,0.,1.),&ids).unwrap();
        assert_eq!(original["clip"].pitch_edit,moved["clip"].pitch_edit);
        assert_eq!(&moved["clip"].pitch_edit,&[60.,61.,62.,63.,64.]);
        let stretched=atlas.project_local(&host(512.013,2.,0.,1.),&ids).unwrap();
        assert_eq!(&stretched["clip"].pitch_edit,&[60.,60.5,61.,61.5,62.,62.5,63.,63.5,64.]);
    }
    /// 一个可见root数组不能把重叠的另一region当同一编辑；只修改可见delta对应的选区。
    #[test]
    fn overlapping_selection_edits_do_not_recapture_another_regions_visible_projection() {
        let mut initial=edited();let mut other=initial.clips[0].clone();other.id="other".into();other.start_sec=3.;initial.clips.push(other);initial.project_sec=4.;
        let params=initial.params_by_root_track.get_mut("a").unwrap();params.pitch_edit.resize(17,0.);params.pitch_orig.resize(17,0.);
        params.pitch_edit[12..17].copy_from_slice(&[67.,68.,69.,70.,71.]);params.pitch_orig[12..17].fill(57.);
        let mut ids=identities();ids.insert("other".into(),RegionIdentity {key:42,source:"source".into(),modification:"other-mod".into()});
        let atlas=ParameterAtlas::default().capture(&initial,&ids).unwrap();
        let mut overlap=initial.clone();overlap.clips[1].start_sec=1.;overlap.project_sec=2.;overlap.selected_clip_id=Some("clip".into());
        let atlas=atlas.follow_geometry(&overlap,&ids).unwrap();let before=atlas.project_roots(&overlap,&ids).unwrap();overlap.params_by_root_track=before.clone();
        let same=atlas.capture_changes(&overlap,&ids,&before).unwrap().project(&overlap,&ids).unwrap();
        assert_eq!(&same["other"].pitch_edit[4..9],&[67.,68.,69.,70.,71.],"只换GUI投影不能改另一region");
        overlap.params_by_root_track.get_mut("a").unwrap().pitch_edit[5]=72.;
        let changed=atlas.capture_changes(&overlap,&ids,&before).unwrap().project(&overlap,&ids).unwrap();
        assert_eq!(changed["clip"].pitch_edit[5],72.);assert_eq!(changed["other"].pitch_edit[5],68.);
    }
    #[test]
    fn source_curves_follow_movement_crop_and_forward_stretch_instead_of_old_project_frames() {
        let atlas=ParameterAtlas::default().capture(&edited(),&identities()).unwrap();
        let moved=atlas.project(&host(3.,1.,0.,1.),&identities()).unwrap();
        assert_eq!(&moved["clip"].pitch_edit[12..17],&[60.,61.,62.,63.,64.]);
        assert_eq!(moved["clip"].pitch_edit[4],0.);assert_eq!(moved["clip"].extra_curves["volume"][4],1.);
        let stretched=atlas.project(&host(1.,2.,0.,1.),&identities()).unwrap();
        assert_eq!(&stretched["clip"].pitch_edit[4..13],&[60.,60.5,61.,61.5,62.,62.5,63.,63.5,64.]);
        let cropped=atlas.project(&host(2.,1.,0.5,0.5),&identities()).unwrap();
        assert_eq!(&cropped["clip"].pitch_edit[8..13],&[62.,62.5,63.,63.5,64.]);
    }
    #[test]
    fn repeated_geometry_roundtrips_do_not_recapture_and_blur_unchanged_source_curves() {
        let ids=identities();let original=edited();let atlas=ParameterAtlas::default().capture(&original,&ids).unwrap();
        let mut changed=host(0.013,1.731,0.,1.);
        changed.params_by_root_track.insert("a".into(),atlas.project(&changed,&ids).unwrap().remove("clip").unwrap());
        let accepted=atlas.capture(&changed,&ids).unwrap();let back=accepted.project(&original,&ids).unwrap();
        assert_eq!(&back["clip"].pitch_edit[4..9],&[60.,61.,62.,63.,64.],"不可把已插值投影反复当新源数据");
    }
    #[test]
    fn unrelated_modification_cannot_inherit_curves_from_a_shared_source_path() {
        let atlas=ParameterAtlas::default().capture(&edited(),&identities()).unwrap();let mut ids=identities();ids.get_mut("clip").unwrap().modification="other".into();
        let projected=atlas.project(&host(1.,1.,0.,1.),&ids).unwrap();assert!(projected.is_empty());
    }
}
