//! 冻结的合成输入只包含Rust值和授权PCM的Arc，不借文档锁或调用宿主ARA接口。

use super::source::SourcePcm;
use super::snapshot::{PlaybackSnapshot,mix_plain_regions};
use crate::ara::AraPlaybackRegion;
use hifishifter_kernel::state::TimelineState;
use hifishifter_kernel::mixdown::{MixdownOptions,MixdownPcm,QualityPreset,render_mixdown_with_pcm};
use std::collections::{HashMap,BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool,Ordering};

pub(crate) struct RenderInput {
    pub timeline:Option<TimelineState>,
    pub regions:Vec<AraPlaybackRegion>,
    pub sources:HashMap<String,Arc<SourcePcm>>,
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
        // 保留原普通/裁切校验，不因移到worker而放宽stretch/content fades。
        let validation=mix_plain_regions(&self.regions,&self.sources,44100).map_err(|e|format!("unsupported host region: {e:?}"))?;
        drop(validation);
        let start=timeline.clips.iter().map(|c|c.start_sec).fold(f64::INFINITY,f64::min);
        let end=timeline.clips.iter().map(|c|c.start_sec+c.length_sec).fold(0.0_f64,f64::max);
        if start<0.0 || !start.is_finite() || !end.is_finite() {return Err("unsupported host position".into());}
        let mut input=HashMap::new();
        for id in timeline.clips.iter().filter_map(|clip|clip.source_path.as_ref()).collect::<BTreeSet<_>>() {
            let pcm=self.sources.get(id).ok_or_else(||format!("host PCM unavailable: {id}"))?;
            let channels=pcm.planes.len();let mut samples=Vec::with_capacity(pcm.planes[0].len()*channels);
            for frame in 0..pcm.planes[0].len() {for plane in &pcm.planes {samples.push(plane[frame]);}}
            input.insert(id.clone(),MixdownPcm {sample_rate:pcm.sample_rate,channels:channels as u16,samples:Arc::new(samples)});
        }
        let mut snapshots=Vec::new();
        for sample_rate in [44100,48000] {
            if cancel.load(Ordering::Acquire) {return Err("host preparation cancelled".into());}
            let frames=((end-start)*sample_rate as f64).round() as usize;
            if frames>64*1024*1024/8 {return Err("snapshot span exceeds 64MiB".into());}
            let reservation=super::budget::global_budget().reserve(frames*8).ok_or("PCM memory budget exceeded")?;
            let options=MixdownOptions {sample_rate,start_sec:start,end_sec:Some(end),
                stretch:hifishifter_kernel::time_stretch::StretchAlgorithm::SoundTouchDll,apply_pitch_edit:true,
                output:hifishifter_kernel::encode::OutputSpec::wav_32f(),quality_preset:QualityPreset::Export,
                cancel_flag:Some(cancel.clone()),progress:None,cache_stats:None};
            let (_,channels,_,samples)=render_mixdown_with_pcm(timeline,options,&input)?;
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
    /// 源表替换不能改变已捕获的输入；取消必须拒绝计算，不靠旧模型裸引用。
    #[test]
    fn frozen_pcm_is_independent_of_replacement_and_cancel_is_respected() {
        let source=Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.1,0.2,0.3,0.4]],version:0,_reservation:None});
        let region=AraPlaybackRegion {audio_source_persistent_id:"fixture".into(),
            duration_in_modification_time:4.0/44100.0,duration_in_playback_time:4.0/44100.0,..Default::default()};
        let mut sources=HashMap::from([("fixture".into(),source)]);
        let input=RenderInput {timeline:None,regions:vec![region],sources:sources.clone()};
        sources.insert("fixture".into(),Arc::new(SourcePcm {sample_rate:44100,planes:vec![vec![0.9;4]],version:1,_reservation:None}));
        drop(sources);
        let cancel=Arc::new(AtomicBool::new(false));
        let output=input.render(cancel.clone()).unwrap();assert_eq!(output[0].left,[0.1,0.2,0.3,0.4]);
        assert_eq!(output[0].right,output[0].left);assert_eq!(output[1].sample_rate,48000);
        cancel.store(true,Ordering::Release);assert!(input.render(cancel).is_err());
    }
}
