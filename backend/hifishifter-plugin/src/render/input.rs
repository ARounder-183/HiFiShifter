//! 冻结的合成输入只包含Rust值和授权PCM的Arc，不借文档锁或调用宿主ARA接口。

use super::source::SourcePcm;
use super::snapshot::{PlaybackSnapshot,mix_plain_regions};
use crate::ara::AraPlaybackRegion;
use hifishifter_kernel::state::TimelineState;
use hifishifter_kernel::mixdown::{MixdownOptions,MixdownPcm,QualityPreset,render_mixdown_with_pcm};
use std::collections::{HashMap,BTreeSet,BTreeMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool,Ordering};

pub(crate) struct RenderInput {
    pub timeline:Option<TimelineState>,
    pub regions:Vec<AraPlaybackRegion>,
    pub sources:HashMap<String,Arc<SourcePcm>>,
    /// 有atlas时为region局部零点参数，不是GUI绝对项目帧。
    pub clip_parameters:BTreeMap<String,hifishifter_kernel::state::TrackParamsState>,
}
impl RenderInput {
    /// 本函数只消费冻结值；调用前必须已释放文档transaction。新模型只能取消/拒绝发布。
    pub fn render(&self,cancel:Arc<AtomicBool>)->Result<Vec<PlaybackSnapshot>,String> {
        if cancel.load(Ordering::Acquire) {return Err("host preparation cancelled".into());}
        let Some(timeline)=&self.timeline else {
            return [44100,48000].into_iter().map(|rate| {
                if cancel.load(Ordering::Acquire) {return Err("host preparation cancelled".into());}
                mix_plain_regions(&self.regions,&self.sources,rate).map_err(|e|format!("snapshot unavailable: {e:?}"))
            }).collect();
        };
        if self.regions.is_empty() {
            return Ok([44100,48000].into_iter().map(|sample_rate|PlaybackSnapshot {
                sample_rate,origin_sample:0,left:vec![],right:vec![],_reservation:None}).collect());
        }
        // 只放行原kernel负责的真实正向时间拉伸；普通PCM mixer不放行，content-based fades另有契约。
        super::snapshot::validate_regions(&self.regions,&self.sources,44100,true).map_err(|e|format!("unsupported host region: {e:?}"))?;
        let start=timeline.clips.iter().map(|c|c.start_sec).fold(f64::INFINITY,f64::min);
        let end=timeline.clips.iter().map(|c|c.start_sec+c.length_sec).fold(0.0_f64,f64::max);
        if start<0.0 || !start.is_finite() || !end.is_finite() {return Err("unsupported host position".into());}
        let mut input=HashMap::new();
        let source_bytes=timeline.clips.iter().filter_map(|clip|clip.source_path.as_ref()).collect::<BTreeSet<_>>().into_iter()
            .try_fold(0_usize,|total,id| {let pcm=self.sources.get(id).ok_or_else(||format!("host PCM unavailable: {id}"))?;
                total.checked_add(pcm.planes.iter().map(|plane|plane.len()*4).sum::<usize>()).ok_or_else(||"host interleaved PCM size overflow".to_owned())})?;
        let _source_working=super::budget::global_budget().reserve(source_bytes).ok_or("host interleaved PCM budget exceeded")?;
        for id in timeline.clips.iter().filter_map(|clip|clip.source_path.as_ref()).collect::<BTreeSet<_>>() {
            let pcm=self.sources.get(id).ok_or_else(||format!("host PCM unavailable: {id}"))?;
            let channels=pcm.planes.len();let mut samples=Vec::with_capacity(pcm.planes[0].len()*channels);
            for frame in 0..pcm.planes[0].len() {for plane in &pcm.planes {samples.push(plane[frame]);}}
            input.insert(id.clone(),MixdownPcm {sample_rate:pcm.sample_rate,channels:channels as u16,samples:Arc::new(samples)});
        }
        let mut snapshots=Vec::new();
        let model_rate_output=timeline.tracks.iter().any(|track|track.pitch_analysis_algo==hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx);
        for sample_rate in [44100,48000] {
            if cancel.load(Ordering::Acquire) {return Err("host preparation cancelled".into());}
            let frames=((end-start)*sample_rate as f64).round() as usize;
            if frames>64*1024*1024/8 {return Err("snapshot span exceeds 64MiB".into());}
            let reservation=super::budget::global_budget().reserve(frames*8).ok_or("PCM memory budget exceeded")?;
            // HiFiGAN工作区只在模型原生44.1k合成一次，48k是便宜输出层，不重新跑HNSEP/神经网络。
            // 纯非HiFiGAN路径保持旧采样率契约；只消费已经就绪的不可变PCM，仍在worker线程。
            if model_rate_output&&sample_rate==48000 {
                let native:&PlaybackSnapshot=&snapshots[0];
                let mut left=hifishifter_kernel::mel_utils::linear_resample_mono(&native.left,44100,48000);
                let mut right=hifishifter_kernel::mel_utils::linear_resample_mono(&native.right,44100,48000);
                left.resize(frames,0.);right.resize(frames,0.);
                snapshots.push(PlaybackSnapshot {sample_rate,origin_sample:(start*sample_rate as f64).round() as i64,left,right,_reservation:Some(reservation)});
                continue;
            }
            let run=|view:&TimelineState,lo:f64,hi:f64| {
                let options=MixdownOptions {sample_rate,start_sec:lo,end_sec:Some(hi),
                    stretch:hifishifter_kernel::time_stretch::StretchAlgorithm::SoundTouchDll,apply_pitch_edit:true,
                    output:hifishifter_kernel::encode::OutputSpec::wav_32f(),quality_preset:QualityPreset::Export,
                    cancel_flag:Some(cancel.clone()),progress:None,cache_stats:None};
                render_mixdown_with_pcm(view,options,&input)
            };
            // 项目数组只负责GUI；每个区域用自己的源参数跑原kernel，再按宿主位置累加。
            // 所有track仍保留原全局solo/父链，不能因为拆region丢掉另一轨solo。
            let _working=super::budget::global_budget().reserve(frames*8).ok_or("parameter render working budget exceeded")?;
            let (channels,samples)=if self.clip_parameters.is_empty() {
                let (_,channels,_,samples)=run(timeline,start,end)?;(channels,samples)
            } else {
                let mut mixed=vec![0_f32;frames*2];
                for clip in &timeline.clips {
                    if cancel.load(Ordering::Acquire) {return Err("host preparation cancelled".into());}
                    let mut view=timeline.clone();view.clips=vec![clip.clone()];
                    let local=self.clip_parameters.get(&clip.id);
                    let (lo,hi)=if let Some(params)=local {
                        let root=view.resolve_root_track_id(&clip.track_id).ok_or("unknown source parameter root")?;
                        view.params_by_root_track.insert(root,params.clone());
                        view.clips[0].start_sec=0.;view.project_sec=clip.length_sec;(0.,clip.length_sec)
                    } else {(start,end)};
                    let part_frames=((hi-lo)*sample_rate as f64).round() as usize;
                    let _part_budget=super::budget::global_budget().reserve(part_frames*8).ok_or("parameter render working budget exceeded")?;
                    let (_,channels,_,part)=run(&view,lo,hi)?;
                    if channels!=2||part.len()!=part_frames*2 {return Err("invalid region parameter output".into());}
                    let offset=if local.is_some() {((clip.start_sec-start)*sample_rate as f64).round() as usize*2} else {0};
                    for (sum,value) in mixed.iter_mut().skip(offset).zip(part) {*sum+=value;}
                }(2,mixed)
            };
            if channels!=2 || samples.len()!=frames*2 || samples.iter().any(|v|!v.is_finite()) {return Err("invalid kernel output".into());}
            snapshots.push(PlaybackSnapshot {sample_rate,origin_sample:(start*sample_rate as f64).round() as i64,
                left:samples.iter().step_by(2).copied().collect(),right:samples.iter().skip(1).step_by(2).copied().collect(),
                _reservation:Some(reservation)});
        }
        Ok(snapshots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 双输出率从同一HiFiGAN原生结果派生，整段HNSEP也只运行一次，不以WORLD外推。
    #[test]
    #[ignore = "真实CPU插件渲染输入诊断：显式运行"]
    fn real_model_hifigan_snapshots_share_one_native_rate_synthesis() {
        hifishifter_kernel::vocoder_ort_session::set_runtime_ep_override(Some("cpu".into()));
        hifishifter_kernel::hnsep_onnx::clear_separation_cache();
        let seconds=0.5;let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"A","order":0,"pitch_analysis_algo":"nsf_hifigan_onnx","compose_enabled":true}],
            "bpm":120,"project_sec":seconds,"clips":[{"id":"clip","name":"A","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();timeline.clips[0].normalize_takes();
        timeline.params_by_root_track.insert("track".into(),hifishifter_kernel::state::TrackParamsState {
            frame_period_ms:5.,pitch_orig:vec![57.;101],pitch_edit:vec![60.;101],pitch_edit_user_modified:true,
            extra_params:HashMap::from([("breath_enabled".into(),1.)]),..Default::default()});
        let pcm=(0..22050).map(|i| {let phase=2.*std::f64::consts::PI*220.*i as f64/44100.;
            (1..=12).map(|k|0.15/k as f64*(phase*k as f64).sin()).sum::<f64>() as f32}).collect();
        let region=AraPlaybackRegion {audio_source_persistent_id:"source".into(),duration_in_modification_time:seconds,duration_in_playback_time:seconds,..Default::default()};
        let parameters=timeline.params_by_root_track["track"].clone();
        hifishifter_kernel::renderer::hifigan::clear_chunk_cache();
        hifishifter_kernel::synth_clip_cache::global_synth_clip_cache().lock().unwrap().clear();
        let mut input=RenderInput {timeline:Some(timeline),regions:vec![region],sources:HashMap::from([("source".into(),
            Arc::new(SourcePcm {sample_rate:44100,planes:vec![pcm],version:0,_reservation:None}))]),clip_parameters:BTreeMap::from([("clip".into(),parameters)])};
        let before=hifishifter_kernel::hnsep_onnx::separation_cache_stats();let began=std::time::Instant::now();
        let neural_before=hifishifter_kernel::nsf_hifigan_onnx::inference_runs();
        let output=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(output[0].left.len(),22050);assert_eq!(output[1].left.len(),24000);
        assert_eq!(output[1].left,hifishifter_kernel::mel_utils::linear_resample_mono(&output[0].left,44100,48000));
        assert_eq!(output[1].right,hifishifter_kernel::mel_utils::linear_resample_mono(&output[0].right,44100,48000));
        assert_eq!(hifishifter_kernel::hnsep_onnx::separation_cache_stats().1-before.1,1);
        assert!(output[0].left.iter().any(|v|v.abs()>0.01));
        let cold_neural=hifishifter_kernel::nsf_hifigan_onnx::inference_runs();assert!(cold_neural>neural_before);
        let cold_ms=began.elapsed().as_millis();
        // 不同owner/region名和非网格项目摆放不得改变源曲线或重推理。
        let moved_start=512.013;
        {let timeline=input.timeline.as_mut().unwrap();timeline.clips[0].id="other-owner-clip".into();timeline.clips[0].start_sec=moved_start;
            timeline.project_sec=moved_start+seconds;}
        let params=input.clip_parameters.remove("clip").unwrap();input.clip_parameters.insert("other-owner-clip".into(),params);
        input.regions[0].start_in_playback_time=moved_start;
        let moved=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(moved[0].origin_sample,(moved_start*44100.).round() as i64);
        assert_eq!(moved[0].left,output[0].left);assert_eq!(moved[1].left,output[1].left);
        assert_eq!(hifishifter_kernel::nsf_hifigan_onnx::inference_runs(),cold_neural);
        input.clip_parameters.get_mut("other-owner-clip").unwrap().pitch_edit.fill(64.);
        let changed=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert!(hifishifter_kernel::nsf_hifigan_onnx::inference_runs()>cold_neural,"真正目标F0改变必须使HiFiGAN失效");
        assert_ne!(changed[0].left,moved[0].left);assert_eq!(hifishifter_kernel::hnsep_onnx::separation_cache_stats().1-before.1,1);
        println!("PLUGIN_HIFIGAN_CONTENT cold_ms={cold_ms} neural_runs={} moved_extra_runs=0 moved_pcm_exact=true derived_48000_exact=true pitch_change_runs={} HNSEP_runs=1",
            cold_neural-neural_before,hifishifter_kernel::nsf_hifigan_onnx::inference_runs()-cold_neural);
    }
    /// 相同root的两个重叠区域保留独立源参数，不能用GUI单一数组覆盖两者。
    #[test]
    fn source_parameter_atlas_overlapping_regions_keep_independent_audio_gains() {
        let seconds=4.0/44100.;let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"A","order":0}],"bpm":120,"project_sec":seconds,
            "clips":[{"id":"a","name":"A","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"ta","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]},
                {"id":"b","name":"B","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"tb","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();for clip in &mut timeline.clips {clip.normalize_takes();}
        let region=AraPlaybackRegion {audio_source_persistent_id:"source".into(),duration_in_modification_time:seconds,duration_in_playback_time:seconds,..Default::default()};
        let params=|gain|hifishifter_kernel::state::TrackParamsState {frame_period_ms:5.,extra_curves:HashMap::from([("volume".into(),vec![gain])]),..Default::default()};
        let input=RenderInput {timeline:Some(timeline),regions:vec![region.clone(),region],sources:HashMap::from([("source".into(),
            Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.1,0.2,0.3,0.4]],version:0,_reservation:None}))]),
            clip_parameters:BTreeMap::from([("a".into(),params(0.5)),("b".into(),params(0.25))])};
        let output=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        for (actual,want) in output[0].left.iter().zip([0.075_f32,0.15,0.225,0.3]) {assert!((*actual-want).abs()<1e-6,"{actual} != {want}");}
    }
    /// 原kernel保调拉伸覆盖整段目标时间；不能只显示新长度却尾部静音或降八度重采样。
    #[test]
    fn linear_stretch_produces_full_length_audio_without_transposing_the_voice() {
        let pcm=(0..22050).map(|n|(2.0*std::f64::consts::PI*220.0*n as f64/44100.0).sin() as f32*0.2).collect();
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![pcm],version:0,_reservation:None});
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":1,
                "takes":[{"id":"ta","name":"A","source_path":"source","source_start_sec":0,"source_end_sec":0.5,"playback_rate":0.5}]}]
        })).unwrap();timeline.clips[0].normalize_takes();
        let region=AraPlaybackRegion {audio_source_persistent_id:"source".into(),duration_in_modification_time:0.5,
            duration_in_playback_time:1.0,is_timestretch_enabled:true,..Default::default()};
        let input=RenderInput {timeline:Some(timeline),regions:vec![region],sources:HashMap::from([("source".into(),source)]),clip_parameters:BTreeMap::new()};
        let output=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(output[0].left.len(),44100);assert_eq!(output[1].left.len(),48000);
        for start in [0.2,0.7] {
            let begin=(start*44100.0) as usize;let count=3528;
            let rms=(output[0].left[begin..begin+count].iter().map(|x|(*x as f64).powi(2)).sum::<f64>()/count as f64).sqrt();
            assert!(rms>0.05,"拉伸后的后半段不能只补静音");
            let mut scores=vec![0.0;442];
            for lag in 147..441 {let mut xy=0.0;let mut xx=0.0;let mut yy=0.0;
                for n in 0..count {let a=output[0].left[begin+n] as f64;let b=output[0].left[begin+n+lag] as f64;
                    xy+=a*b;xx+=a*a;yy+=b*b;}scores[lag]=xy/(xx*yy).sqrt();}
            let lag=(148..440).find(|&lag|scores[lag]>0.9&&scores[lag]>scores[lag-1]&&scores[lag]>=scores[lag+1]).unwrap();
            let hz=44100.0/lag as f64;assert!((hz-220.0).abs()<2.0,"必须保调而不是普通重采样: {hz}");
        }
    }
    /// 源表替换不能改变已捕获的输入；取消必须拒绝计算，不靠旧模型裸引用。
    #[test]
    fn frozen_pcm_is_independent_of_replacement_and_cancel_is_respected() {
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.1,0.2,0.3,0.4]],version:0,_reservation:None});
        let region=AraPlaybackRegion {audio_source_persistent_id:"fixture".into(),
            duration_in_modification_time:4.0/44100.0,duration_in_playback_time:4.0/44100.0,..Default::default()};
        let mut sources=HashMap::from([("fixture".into(),source)]);
        let input=RenderInput {timeline:None,regions:vec![region],sources:sources.clone(),clip_parameters:BTreeMap::new()};
        sources.insert("fixture".into(),Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.9;4]],version:1,_reservation:None}));
        drop(sources);
        let cancel=Arc::new(AtomicBool::new(false));
        let output=input.render(cancel.clone()).unwrap();assert_eq!(output[0].left,[0.1,0.2,0.3,0.4]);
        assert_eq!(output[0].right,output[0].left);assert_eq!(output[1].sample_rate,48000);
        cancel.store(true,Ordering::Release);assert!(input.render(cancel).is_err());
    }
}
