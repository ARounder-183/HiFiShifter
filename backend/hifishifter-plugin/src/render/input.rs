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
        self.render_with_progress(cancel,None)
    }
    pub fn render_with_progress(&self,cancel:Arc<AtomicBool>,progress:Option<hifishifter_kernel::mixdown::ProgressCallback>)->Result<Vec<PlaybackSnapshot>,String> {
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
        // 两率最终PCM、源转交错副本、当前混音/region临时输出一起预检，失败不先做重推理。
        // 此策略保留512MiB全局硬上限，不把旧30秒/64MiB探针常量直接放宽为无界分配。
        let frame_counts=[44100_u32,48000].map(|rate| {
            let frames=((end-start)*rate as f64).round();
            if !frames.is_finite()||frames<0.||frames>=usize::MAX as f64/8. {Err("snapshot size overflow".to_owned())}
            else {Ok(frames as usize)}
        }).into_iter().collect::<Result<Vec<_>,_>>()?;
        let outputs=frame_counts.iter().try_fold(0_usize,|total,frames|total.checked_add(frames*8))
            .ok_or("snapshot size overflow")?;
        let working=frame_counts.iter().copied().max().unwrap_or(0).checked_mul(8)
            .and_then(|bytes|bytes.checked_mul(if self.clip_parameters.is_empty()||timeline.clips.len()==1 {1} else {2}))
            .ok_or("render working size overflow")?;
        let required=source_bytes.checked_add(outputs).and_then(|bytes|bytes.checked_add(working))
            .ok_or("render memory size overflow")?;
        let budget=super::budget::global_budget();
        let mut batch=budget.reserve(required).ok_or_else(||format!("render PCM budget exceeded: required={required} used={} limit={}",budget.used(),budget.limit()))?;
        let _source_working=batch.split_off(source_bytes).ok_or("invalid render budget partition")?;
        let _working=batch.split_off(working).ok_or("invalid render budget partition")?;
        for id in timeline.clips.iter().filter_map(|clip|clip.source_path.as_ref()).collect::<BTreeSet<_>>() {
            let pcm=self.sources.get(id).ok_or_else(||format!("host PCM unavailable: {id}"))?;
            let channels=pcm.planes.len();let mut samples=Vec::with_capacity(pcm.planes[0].len()*channels);
            for frame in 0..pcm.planes[0].len() {for plane in &pcm.planes {samples.push(plane[frame]);}}
            input.insert(id.clone(),MixdownPcm {sample_rate:pcm.sample_rate,channels:channels as u16,samples:Arc::new(samples)});
        }
        let mut snapshots=Vec::new();
        let model_rate_output=timeline.clips.iter().filter_map(|clip|timeline.resolve_root_track_id(&clip.track_id))
            .any(|root|timeline.tracks.iter().any(|track|track.id==root&&track.pitch_analysis_algo==hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx));
        for (sample_rate,frames) in [44100,48000].into_iter().zip(frame_counts) {
            if cancel.load(Ordering::Acquire) {return Err("host preparation cancelled".into());}
            let reservation=batch.split_off(frames*8).ok_or("invalid render budget partition")?;
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
                    cancel_flag:Some(cancel.clone()),progress:progress.clone(),cache_stats:None};
                render_mixdown_with_pcm(view,options,&input)
            };
            // 项目数组只负责GUI；每个区域用自己的源参数跑原kernel，再按宿主位置累加。
            // 所有track仍保留原全局solo/父链，不能因为拆region丢掉另一轨solo。
            let (channels,samples)=if self.clip_parameters.is_empty() {
                let (_,channels,_,samples)=run(timeline,start,end)?;(channels,samples)
            } else if timeline.clips.len()==1 {
                // 单region直接保留原kernel结果，不另建全长mixed再从part复制一遍。
                // 参数仍从局部零点投影，所有track保留全局solo/父链；只省去加到零缓冲的复制。
                let clip=&timeline.clips[0];let mut view=timeline.clone();
                let (lo,hi)=if let Some(params)=self.clip_parameters.get(&clip.id) {
                    let root=view.resolve_root_track_id(&clip.track_id).ok_or("unknown source parameter root")?;
                    view.params_by_root_track.insert(root,params.clone());view.clips[0].start_sec=0.;
                    view.project_sec=clip.length_sec;(0.,clip.length_sec)
                } else {(start,end)};
                let (_,channels,_,mut samples)=run(&view,lo,hi)?;
                // 保留旧累加语义的signed-zero位形，不改变已归档PCM的零样本身份。
                for sample in &mut samples {*sample=0_f32+*sample;}(channels,samples)
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
    /// 用已核对的windows crate typed API读取真实进程高水位，不用PCM额度冒充RSS。
    #[cfg(windows)]
    fn process_memory_profile()->(usize,usize,usize) {
        use windows::Win32::System::{ProcessStatus::{K32GetProcessMemoryInfo,PROCESS_MEMORY_COUNTERS},Threading::GetCurrentProcess};
        let bytes=std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let mut counters=PROCESS_MEMORY_COUNTERS {cb:bytes,..Default::default()};
        // SAFETY: 当前进程伪句柄有效；typed结构由本调用拥有，大小与官方绑定完全匹配。
        assert!(unsafe {K32GetProcessMemoryInfo(GetCurrentProcess(),&mut counters,bytes)}.as_bool());
        (counters.WorkingSetSize,counters.PeakWorkingSetSize,counters.PeakPagefileUsage)
    }
    /// 正常三分钟完整源的HiFiGAN分块及整段HNSEP诊断；只显式运行，不冒充REAPER GUI验收。
    #[test]
    #[ignore = "真实CPU三分钟HiFiGAN/HNSEP及进程内存诊断：显式运行"]
    fn real_three_minute_hifigan_hnsep_prepares_complete_pcm_and_reuses_warm_cache() {
        hifishifter_kernel::vocoder_ort_session::set_runtime_ep_override(Some("cpu".into()));
        crate::editor::resources::initialize_models();hifishifter_kernel::hnsep_onnx::clear_separation_cache();
        hifishifter_kernel::renderer::hifigan::clear_chunk_cache();
        hifishifter_kernel::synth_clip_cache::global_synth_clip_cache().lock().unwrap().clear();
        let seconds=180.;let frames=44100*180;let budget=super::super::budget::global_budget();let baseline=budget.used();
        let reservation=budget.reserve(frames*4).unwrap();
        let samples=(0..frames).map(|index|{let phase=2.*std::f64::consts::PI*220.*index as f64/44100.;
            (1..=8).map(|harmonic|0.12/harmonic as f64*(phase*harmonic as f64).sin()).sum::<f64>() as f32}).collect();
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![samples],version:1,_reservation:Some(reservation)});
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"Long HifiGAN","order":0,"pitch_analysis_algo":"nsf_hifigan_onnx","compose_enabled":true}],
            "bpm":120,"project_sec":seconds,"clips":[{"id":"clip","name":"Long HifiGAN","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();timeline.clips[0].normalize_takes();
        let params=hifishifter_kernel::state::TrackParamsState {frame_period_ms:5.,pitch_orig:vec![57.;36001],pitch_edit:vec![60.;36001],
            pitch_edit_user_modified:true,extra_params:HashMap::from([("breath_enabled".into(),1.)]),..Default::default()};
        timeline.params_by_root_track.insert("track".into(),params.clone());
        let input=RenderInput {timeline:Some(timeline),regions:vec![AraPlaybackRegion {audio_source_persistent_id:"source".into(),
            duration_in_modification_time:seconds,duration_in_playback_time:seconds,..Default::default()}],
            sources:HashMap::from([("source".into(),source.clone())]),clip_parameters:BTreeMap::from([("clip".into(),params)])};
        let neural_before=hifishifter_kernel::nsf_hifigan_onnx::inference_runs();let separation_before=hifishifter_kernel::hnsep_onnx::separation_cache_stats().1;
        let began=std::time::Instant::now();let mut cold=input.render(Arc::new(AtomicBool::new(false))).unwrap();let cold_ms=began.elapsed().as_millis();
        let cold_neural=hifishifter_kernel::nsf_hifigan_onnx::inference_runs();let cold_separation=hifishifter_kernel::hnsep_onnx::separation_cache_stats().1;
        assert!(cold_neural-neural_before>1,"三分钟必须实际走多个神经批次，不能只返回缓存元数据");
        assert_eq!(cold_separation-separation_before,1,"HNSEP仍整段一次，不分块");
        for (snapshot,rate) in cold.iter().zip([44100,48000]) {assert_eq!(snapshot.left.len(),rate*180);assert_eq!(snapshot.right.len(),rate*180);
            assert!(snapshot.left.iter().all(|sample|sample.is_finite()));
            for start in [0,rate*90,rate*179] {let rms=(snapshot.left[start..start+rate].iter().map(|sample|(*sample as f64).powi(2)).sum::<f64>()/rate as f64).sqrt();
                assert!(rms>0.01,"源中部/末尾不能是静音占位，start={start} rms={rms}");}}
        assert_eq!(cold[1].left,hifishifter_kernel::mel_utils::linear_resample_mono(&cold[0].left,44100,48000));
        super::super::snapshot::compact_prepared(&mut cold).unwrap();
        let began=std::time::Instant::now();let mut warm=input.render(Arc::new(AtomicBool::new(false))).unwrap();let warm_ms=began.elapsed().as_millis();
        assert_eq!(hifishifter_kernel::nsf_hifigan_onnx::inference_runs(),cold_neural);assert_eq!(hifishifter_kernel::hnsep_onnx::separation_cache_stats().1,cold_separation);
        super::super::snapshot::compact_prepared(&mut warm).unwrap();
        for (cold,warm) in cold.iter().zip(&warm) {assert_eq!(cold.left,warm.left);assert_eq!(cold.right,warm.right);}
        #[cfg(windows)] {let (working,peak,commit)=process_memory_profile();
            eprintln!("LONG_HIFIGAN seconds=180 cold_ms={cold_ms} warm_ms={warm_ms} neural_runs={} HNSEP_runs=1 warm_extra_runs=0 accounted_peak={} working_set={working} peak_working_set={peak} peak_commit={commit}",cold_neural-neural_before,budget.peak());}
        drop(cold);drop(warm);drop(input);drop(source);assert_eq!(budget.used(),baseline);
    }
    /// 另一轨采用HiFiGAN不能改变当前renderer的原生率路径；其solo仍应按全工程状态参与。
    #[test]
    fn unrelated_hifigan_track_does_not_change_this_renderers_rate_path() {
        let seconds=0.05;let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"A","order":0,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":seconds,
            "clips":[{"id":"clip","name":"A","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();timeline.clips[0].normalize_takes();
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![(0..2205).map(|index|((index*113)%257) as f32/512.).collect()],version:0,_reservation:None});
        let mut input=RenderInput {timeline:Some(timeline),regions:vec![AraPlaybackRegion {audio_source_persistent_id:"source".into(),
            duration_in_modification_time:seconds,duration_in_playback_time:seconds,..Default::default()}],
            sources:HashMap::from([("source".into(),source)]),clip_parameters:BTreeMap::new()};
        let before=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        let mut other=input.timeline.as_ref().unwrap().tracks[0].clone();other.id="unrelated".into();other.order=1;
        other.pitch_analysis_algo=hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx;
        input.timeline.as_mut().unwrap().tracks.push(other);let after=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        for (before,after) in before.iter().zip(&after) {assert_eq!(before.left,after.left);assert_eq!(before.right,after.right);}
        input.timeline.as_mut().unwrap().tracks[1].solo=true;
        let solo=input.render(Arc::new(AtomicBool::new(false))).unwrap();assert!(solo.iter().all(|snapshot|snapshot.left.iter().all(|value|*value==0.)));
    }
    /// 三分钟普通人声经过授权kernel/两率快照后全尾就绪；内存按固定全局预算预检。
    #[test]
    fn three_minute_host_clip_prepares_both_rates_and_the_last_block_within_budget() {
        let frames=44100*180;let seconds=180.;let budget=super::super::budget::global_budget();let baseline=budget.used();
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.25;frames]],version:1,
            _reservation:Some(budget.reserve(frames*4).unwrap())});
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"Long voice","order":0,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":seconds,
            "clips":[{"id":"clip","name":"Long voice","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();timeline.clips[0].normalize_takes();
        let input=RenderInput {timeline:Some(timeline),regions:vec![AraPlaybackRegion {audio_source_persistent_id:"source".into(),
            duration_in_modification_time:seconds,duration_in_playback_time:seconds,..Default::default()}],
            sources:HashMap::from([("source".into(),source.clone())]),clip_parameters:BTreeMap::new()};
        let snapshots=input.render(Arc::new(AtomicBool::new(false))).unwrap();
        for (snapshot,rate) in snapshots.iter().zip([44100,48000]) {
            assert_eq!(snapshot.left.len(),rate*180);assert_eq!(snapshot.right.len(),rate*180);
            assert!(snapshot.left.iter().all(|sample|(*sample-0.25).abs()<1e-6));
        }
        let accounted=budget.used()-baseline;assert_eq!(accounted,frames*4+(44100+48000)*180*8);
        eprintln!("long-host seconds=180 source_bytes={} ready_bytes={} accounted_peak={} hard_limit={}",
            frames*4,accounted-frames*4,budget.peak(),budget.limit());
        let mut left=[0_f32;513];let mut right=[0_f32;513];let mut planes=[left.as_mut_ptr(),right.as_mut_ptr()];
        let mut bus=crate::audio_abi::AudioBusBuffers {num_channels:2,silence_flags:0,channel_buffers:planes.as_mut_ptr()};
        let publisher=super::super::snapshot::SnapshotPublisher::default();let mut snapshots=snapshots;
        publisher.publish(snapshots.pop().unwrap()).unwrap();
        assert!(unsafe {publisher.copy_block((48000*180-511) as i64,48000,&mut bus,513)});
        assert_eq!(&left[..511],&[0.25;511]);assert_eq!(&right[..511],&[0.25;511]);assert_eq!(&left[511..],&[0.;2]);
        assert!(unsafe {publisher.copy_block(0,48000,&mut bus,513)});assert_eq!(left,[0.25;513]);
        drop(publisher);drop(snapshots);drop(input);drop(source);assert_eq!(budget.used(),baseline);
    }
    /// 大跨度在任何合成前明确预算失败，不能为通过长源门扩大512MiB或静默补零。
    #[test]
    fn sparse_huge_span_is_rejected_before_any_large_render_allocation() {
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"Sparse voice","order":0,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":3601,
            "clips":[{"id":"a","name":"A","track_id":"track","start_sec":0,"length_sec":1,"takes":[{"id":"ta","source_path":"source","source_end_sec":1}]},
                {"id":"b","name":"B","track_id":"track","start_sec":3600,"length_sec":1,"takes":[{"id":"tb","source_path":"source","source_end_sec":1}]}]
        })).unwrap();for clip in &mut timeline.clips {clip.normalize_takes();}
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.25;44100]],version:1,_reservation:None});
        let region=|start|AraPlaybackRegion {audio_source_persistent_id:"source".into(),start_in_playback_time:start,
            duration_in_modification_time:1.,duration_in_playback_time:1.,..Default::default()};
        let input=RenderInput {timeline:Some(timeline),regions:vec![region(0.),region(3600.)],sources:HashMap::from([("source".into(),source)]),clip_parameters:BTreeMap::new()};
        let budget=super::super::budget::global_budget();let baseline=budget.used();
        let error=input.render(Arc::new(AtomicBool::new(false))).unwrap_err();assert!(error.contains("BudgetExceeded"));
        assert_eq!(budget.used(),baseline);
    }
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
