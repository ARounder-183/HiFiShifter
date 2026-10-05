//! 原GUI命令的插件准入与类型适配；使用共享原参数/历史/波形实现，不接收宿主几何变更。
use super::session::EditorSession;
use hifishifter_kernel::editor::{params,history,waveform,ParamHost};
use hifishifter_kernel::state::*;
use serde::Deserialize;
use serde_json::{json,Value};
use std::sync::atomic::Ordering;
use base64::Engine as _;

pub(super) fn mutates_audio(command:&str)->bool {
    matches!(command,"set_param_frames"|"restore_param_frames"|"set_static_param"|"convert_mix_param"|
        "set_track_state"|"undo_timeline"|"redo_timeline"|"set_history_position")
}
fn value<T:serde::Serialize>(input:T)->Result<Value,String> {serde_json::to_value(input).map_err(|e|e.to_string())}
fn args<T:serde::de::DeserializeOwned>(input:Value)->Result<T,String> {serde_json::from_value(input).map_err(|e|format!("invalid editor arguments: {e}"))}
#[derive(Deserialize)]
#[serde(rename_all="camelCase")]
struct Frames {track_id:String,param:String,start_frame:u32,#[serde(default)]frame_count:u32,
    stride:Option<u32>,binary:Option<bool>,with_sentinel:Option<bool>,#[serde(default)]values:Vec<f32>,checkpoint:Option<bool>}
#[derive(Deserialize)]
#[serde(rename_all="camelCase")]
struct Static {track_id:String,param:String,value:Option<f64>,checkpoint:Option<bool>}
#[derive(Deserialize)]
#[serde(rename_all="camelCase")]
struct Mix {track_id:String,from:String,ranges:Vec<hifishifter_kernel::editor::ConvertRange>}
#[derive(Deserialize)]
#[serde(rename_all="camelCase")]
struct Segment {track_id:String,start_sec:f64,duration_sec:f64,columns:usize}
#[derive(Deserialize)]
#[serde(rename_all="camelCase")]
struct TrackPatch {track_id:String,volume:Option<f32>,muted:Option<bool>,solo:Option<bool>,
    compose_enabled:Option<bool>,pitch_analysis_algo:Option<PitchAnalysisAlgo>}

/// 构造原payload，包括真实本地dirty和历史深度，不伪造工程文件路径。
pub(super) fn payload(session:&EditorSession,lite:bool)->Result<Value,String> {
    let (position,_)=session.transport();
    let timeline=session.timeline.lock().unwrap();
    let mut payload=if lite {timeline.to_payload_lite()} else {timeline.to_payload()};
    payload.playhead_sec=position.max(0.);
    let (undo,redo)=history_depths_of(&session.history.lock().unwrap());
    payload.undo_depth=Some(undo);payload.redo_depth=Some(redo);
    let project=session.project.lock().unwrap().clone();
    payload.project=Some(hifishifter_kernel::models::ProjectMetaPayload {
        name:"REAPER / HiFiShifter".into(),path:None,
        dirty:session.generation.load(Ordering::Acquire)!=session.applied.load(Ordering::Acquire),
        recent:vec![],notes_markdown:project.notes_markdown,base_scale:project.base_scale,
        use_custom_scale:project.use_custom_scale,custom_scale:project.custom_scale,beats_per_bar:project.beats_per_bar,
        time_signature_denominator:project.time_signature_denominator,grid_size:project.grid_size,
        stretch_algorithm_override:project.stretch_algorithm_override,hifigan_mel_stretch_override:project.hifigan_mel_stretch_override,
        save_undo_history:project.save_undo_history,
    });
    drop(timeline);let mut result=value(payload)?;session.decorate_host_fades(&mut result);Ok(result)
}
fn history_state(session:&EditorSession)->Value {
    let history=session.history.lock().unwrap();let (undo,redo)=history_depths_of(&history);
    let records:Vec<_>=history.records.iter().map(|r|json!({"label":r.label,"atMs":r.at_ms})).collect();
    json!({"ok":true,"position":history.position,"undoDepth":undo,"redoDepth":redo,
        "records":if records.is_empty() {vec![json!({"label":null,"atMs":history.started_at_ms})]} else {records}})
}
fn track_exists(session:&EditorSession,track:&str)->Result<(),String> {
    if session.timeline.lock().unwrap().tracks.iter().any(|t|t.id==track) {Ok(())} else {Err("unknown host track".into())}
}
fn budget(start:u32,count:usize)->Result<(),String> {
    if count>1_000_000 || (start as usize).checked_add(count).is_none_or(|n|n>1_000_000) {return Err("parameter frame budget exceeded".into());}
    Ok(())
}
fn after_write(session:&EditorSession,result:Value)->Result<Value,String> {
    if result["ok"]==false {return Err(result["error"].as_str().unwrap_or("parameter operation rejected").into());}
    if let Some(error)=session.error.lock().unwrap().clone() {return Err(error);}
    session.emit("history_state",history_state(session));
    session.notify_timeline();
    Ok(result)
}
fn peaks(session:&EditorSession,path:&str)->Result<std::sync::Arc<hifishifter_kernel::hfspeaks_v2::HfsPeakFile>,String> {
    session.check_source(path)?;
    if let Some(found)=session.peaks.lock().unwrap().get(path).cloned() {return Ok(found);}
    let result=std::sync::Arc::new(hifishifter_kernel::hfspeaks_v2::compute_mipmap_peaks(std::path::Path::new(path))?);
    let mut cache=session.peaks.lock().unwrap();
    let total:u64=cache.values().map(|p|p.estimated_byte_size()).sum();
    if total.saturating_add(result.estimated_byte_size())>64*1024*1024 {cache.clear();}
    cache.insert(path.into(),result.clone());Ok(result)
}

/// 仅actor线程调用。宿主PCM不可用/未知命令/平台能力缺失均明确报错。
pub(super) fn dispatch(session:&EditorSession,command:&str,input:Value)->Result<Value,String> {
    match command {
        "get_ui_settings"=>return value(session.settings.lock().unwrap().clone()),
        "save_ui_settings"=>{
            let mut current=session.settings.lock().unwrap();
            let patched=hifishifter_kernel::editor::settings::merge(serde_json::to_value(&*current).map_err(|e|e.to_string())?,&input["settings"]);
            let settings:hifishifter_kernel::config::UiSettings=args(patched)?;
            *current=settings;return Ok(json!({"ok":true}));
        },
        "get_about_info"=>return Ok(json!({"ok":true,"name":"HiFiShifter","version":crate::VERSION,"host":"ARA plugin"})),
        "plugin_get_apply_state"=>return Ok(session.state()),
        "get_playback_state"=>{
            // 宿主播放态不依赖曲线载入；Unsupported/Conflict也必须还能观察播放并暂停。
            return Ok(session.playback_state());
        },
        "plugin_refresh"=>{session.ensure_loaded(input["force"].as_bool().unwrap_or(false))?;return payload(session,false);},
        "set_ui_locale"=>return Ok(json!({"ok":true,"locale":input["locale"]})),
        "consume_startup_project_path"=>return Ok(Value::Null),
        "get_processor_params"=>return value(hifishifter_kernel::editor::capabilities::get_processor_params(input["algo"].as_str().ok_or("algo missing")?.into())),
        "transliterate"=>{
            let texts:Vec<String>=args(input["texts"].clone())?;
            if texts.len()>5000 || texts.iter().map(String::len).sum::<usize>()>1024*1024 {return Err("text index budget exceeded".into());}
            let options:hifishifter_kernel::search::SearchOptions=if input["options"].is_null() {Default::default()} else {args(input["options"].clone())?};
            return value(hifishifter_kernel::search::transliterate_batch(&texts,&options));
        },
        "read_system_clipboard_object"=>{
            return match hifishifter_clipboard::read_bytes()? {
                Some(bytes)=>match String::from_utf8(bytes) {
                    Ok(payload)=>Ok(json!({"ok":true,"available":true,"payload":payload})),
                    Err(_)=>Ok(json!({"ok":true,"available":false})),
                },
                None=>Ok(json!({"ok":true,"available":false})),
            };
        },
        "write_system_clipboard_object"=>{
            let payload=input["payload"].as_str().ok_or("clipboard payload required")?;
            if payload.len()>8*1024*1024 {return Err("parameter clipboard budget exceeded".into());}
            let decoded:Value=args(serde_json::from_str(payload).map_err(|e|e.to_string())?)?;
            if decoded["kind"]!="param" {return Err("timeline clipboard geometry is controlled by host".into());}
            hifishifter_clipboard::write_bytes(payload.as_bytes(),input["textSummary"].as_str().unwrap_or("HiFiShifter parameter data copied."))?;
            return Ok(json!({"ok":true}));
        },
        "clipboard_kind"=>{
            let kind=hifishifter_clipboard::read_bytes()?.and_then(|bytes|serde_json::from_slice::<Value>(&bytes).ok())
                .filter(|p|p["kind"]=="param").map(|_|"param");
            return Ok(json!({"ok":true,"kind":kind}));
        },
        "emit_ui_event"=>{let event=input["event"].as_str().ok_or("event missing")?;session.emit(event,input["payload"].clone());return Ok(Value::Null);},
        _=>{},
    }
    session.ensure_loaded(false)?;
    match command {
        "get_timeline_state"=>payload(session,false),
        "get_timeline_state_lite"=>payload(session,true),
        "get_project_meta"=>Ok(payload(session,true)?["project"].clone()),
        "get_runtime_info"=>{
            let timeline=payload(session,true)?;
            let (_,playing)=session.transport();
            Ok(json!({"ok":true,"device":"REAPER / ARA","model_loaded":hifishifter_kernel::world_vocoder::is_available(),
                "audio_loaded":!session.timeline.lock().unwrap().clips.is_empty(),"has_synthesized":session.applied.load(Ordering::Acquire)>0,
                "is_playing":playing,"playback_target":if playing {Some("synthesized")} else {None},"gpu_backend":"","timeline":timeline}))
        },
        "get_history_state"=>Ok(history_state(session)),
        "begin_undo_group"=>{
            let timeline=session.timeline.lock().unwrap();
            history::checkpoint(&mut session.history.lock().unwrap(),&timeline,input["label"].as_str().unwrap_or("batch").into(),||None);
            session.suppress_history.store(true,Ordering::Release);drop(timeline);session.emit("history_state",history_state(session));payload(session,false)
        },
        "end_undo_group"=>{session.suppress_history.store(false,Ordering::Release);Ok(json!({"ok":true}))},
        "select_track"=>{
            let id=input["trackId"].as_str().ok_or("trackId missing")?;track_exists(session,id)?;
            session.timeline.lock().unwrap().select_track(id);session.select_source_projection()?;session.notify_timeline();payload(session,false)
        },
        "select_clip"=>{
            let id=input["clipId"].as_str().map(str::to_owned);
            if let Some(id)=&id {if !session.timeline.lock().unwrap().clips.iter().any(|c|&c.id==id) {return Err("unknown host clip".into());}}
            session.timeline.lock().unwrap().select_clip(id);session.select_source_projection()?;session.notify_timeline();payload(session,false)
        },
        "set_transport"=>{
            if input["bpm"].is_number() {return Err("tempo is controlled by REAPER".into());}
            if let Some(position)=input["playheadSec"].as_f64() {if position.is_finite() {session.timeline.lock().unwrap().playhead_sec=position.max(0.);}}
            payload(session,true)
        },
        "get_pitch_analysis_progress"=>Ok(Value::Null),
        "get_track_summary"=>{
            let timeline=session.timeline.lock().unwrap();
            Ok(json!({"ok":true,"track_id":input["trackId"].as_str().map(str::to_owned).or_else(||timeline.selected_track_id.clone()),
                "waveform_preview":[],"pitch_range":{"min":-24,"max":24}}))
        },
        "get_param_frames"=>{
            let a:Frames=args(input)?;track_exists(session,&a.track_id)?;budget(a.start_frame,a.frame_count as usize)?;
            value(params::get_param_frames(session,a.track_id,a.param,a.start_frame,a.frame_count,a.stride,a.binary,a.with_sentinel))
        },
        "set_param_frames"=>{
            let a:Frames=args(input)?;track_exists(session,&a.track_id)?;budget(a.start_frame,a.values.len())?;
            after_write(session,params::set_param_frames(session,a.track_id,a.param,a.start_frame,a.values,a.checkpoint))
        },
        "restore_param_frames"=>{
            let a:Frames=args(input)?;track_exists(session,&a.track_id)?;budget(a.start_frame,a.frame_count as usize)?;
            after_write(session,params::restore_param_frames(session,a.track_id,a.param,a.start_frame,a.frame_count,a.checkpoint))
        },
        "get_static_param"=>{let a:Static=args(input)?;track_exists(session,&a.track_id)?;value(params::get_static_param(session,a.track_id,a.param))},
        "set_static_param"=>{
            let a:Static=args(input)?;track_exists(session,&a.track_id)?;
            let number=a.value.filter(|n|n.is_finite()).ok_or("finite static parameter required")?;
            after_write(session,params::set_static_param(session,a.track_id,a.param,number,a.checkpoint))
        },
        "convert_mix_param"=>{
            let a:Mix=args(input)?;track_exists(session,&a.track_id)?;
            for range in &a.ranges {budget(range.start_frame,range.frame_count as usize)?;}
            after_write(session,params::convert_mix_param(session,a.track_id,a.from,a.ranges))
        },
        "set_track_state"=>{
            let object=input.as_object().ok_or("track arguments required")?;
            for (key,value) in object {if !matches!(key.as_str(),"trackId"|"muted"|"solo"|"volume"|"composeEnabled"|"pitchAnalysisAlgo") && !value.is_null() {return Err(format!("track property controlled by host: {key}"));}}
            let patch:TrackPatch=args(input.clone())?;
            // 工程反序列化为前向兼容保留Unknown；实时GUI命令不能把它当可执行算法。
            if patch.pitch_analysis_algo.as_ref().is_some_and(|algo|matches!(algo,PitchAnalysisAlgo::Unknown|PitchAnalysisAlgo::VocalShifterVslib)) {
                return Err("algorithm unavailable in plugin mode".into());
            }
            let id=patch.track_id.as_str();track_exists(session,id)?;
            if patch.volume.is_some_and(|n|!n.is_finite() || !(0.0..=4.0).contains(&n)) {return Err("track volume out of range".into());}
            let mut timeline=session.timeline.lock().unwrap();
            let mut candidate=timeline.clone();
            let track=candidate.tracks.iter_mut().find(|t|t.id==id).unwrap();
            if let Some(volume)=patch.volume {track.volume=volume;}
            if let Some(value)=patch.muted {track.muted=value;}
            if let Some(value)=patch.solo {track.solo=value;}
            if let Some(value)=patch.compose_enabled {track.compose_enabled=value;}
            if let Some(value)=patch.pitch_analysis_algo {track.pitch_analysis_algo=value;}
            session.checkpoint_timeline(&timeline,HistoryOp::EditTrack);
            *timeline=candidate;
            session.mark_dirty();session.publish_timeline(timeline.clone());drop(timeline);
            after_write(session,json!({"ok":true}))?;payload(session,false)
        },
        "undo_timeline"|"redo_timeline"|"set_history_position"=>{
            let mut timeline=session.timeline.lock().unwrap();let mut recorded=session.history.lock().unwrap();
            let (target,intent)=match command {
                "undo_timeline"=>(recorded.position.saturating_sub(1),HistoryJumpIntent::Undo),
                "redo_timeline"=>(recorded.position.saturating_add(1),HistoryJumpIntent::Redo),
                _=>(input["position"].as_u64().ok_or("history position missing")? as usize,HistoryJumpIntent::Jump),
            };
            if let Some((next,_,selection))=history::jump(&mut recorded,&timeline,target,intent,None) {
                *timeline=next;drop(recorded);session.mark_dirty();session.publish_timeline(timeline.clone());drop(timeline);
                session.emit("history_state",history_state(session));
                session.notify_timeline();
                let mut payload=payload(session,false)?;
                if let Some(selection)=selection {payload["param_selection_restore"]=json!(selection);}
                if let Some(error)=session.error.lock().unwrap().clone() {return Err(error);}Ok(payload)
            } else {drop(recorded);drop(timeline);payload(session,false)}
        },
        "get_waveform_mipmap_binary"=>{
            let path=input["sourcePath"].as_str().ok_or("sourcePath missing")?;let level=input["level"].as_u64().unwrap_or(2).min(2) as usize;
            Ok(Value::String(base64::engine::general_purpose::STANDARD.encode(peaks(session,path)?.to_binary_level(level))))
        },
        "preload_waveform_mipmap"=>{peaks(session,input["sourcePath"].as_str().ok_or("sourcePath missing")?)?;Ok(json!({"ok":true}))},
        "batch_get_waveform_mipmap"=>{
            let paths:Vec<String>=args(input["sourcePaths"].clone())?;let levels:Option<Vec<usize>>=args(input["levels"].clone())?;
            if paths.len()>64 {return Err("too many waveform sources".into());}
            let mut result=serde_json::Map::new();
            for path in paths {let data=peaks(session,&path)?;let encoded:Vec<_>=(0..3).map(|level| {
                if levels.as_ref().is_none_or(|l|l.is_empty()||l.contains(&level)) {base64::engine::general_purpose::STANDARD.encode(data.to_binary_level(level))} else {String::new()}
            }).collect();result.insert(path,json!(encoded));}Ok(Value::Object(result))
        },
        "get_root_mix_waveform_peaks_segment"|"get_track_mix_waveform_peaks_segment"=>{
            let a:Segment=args(input)?;track_exists(session,&a.track_id)?;
            if !a.start_sec.is_finite() || !a.duration_sec.is_finite() || a.duration_sec<=0. {return Err("invalid waveform range".into());}
            if a.duration_sec*48000.0*2.0*4.0>64.0*1024.0*1024.0 {return Err("waveform mix buffer budget exceeded".into());}
            if command=="get_root_mix_waveform_peaks_segment" {value(waveform::get_root_mix_waveform_peaks_segment(session,a.track_id,a.start_sec,a.duration_sec,a.columns))}
            else {value(waveform::get_track_mix_waveform_peaks_segment(session,a.track_id,a.start_sec,a.duration_sec,a.columns))}
        },
        _=>Err(format!("Command unavailable in ARA plugin mode: {command}")),
    }
}
