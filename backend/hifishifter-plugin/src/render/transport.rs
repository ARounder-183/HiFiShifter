//! 实时回调发布的宿主播放时钟；GUI读近似最新块，process只执行原子写入。
use std::sync::atomic::{AtomicBool,AtomicU64,AtomicI64,AtomicI32,Ordering};
#[derive(Default)]
pub(crate) struct TransportClock {
    position:AtomicU64,playing:AtomicBool,tempo:AtomicU64,
    host_pose:AtomicU64,host_valid:AtomicBool,
    realtime:AtomicU64,prefetch:AtomicU64,offline:AtomicU64,switches:AtomicU64,backwards:AtomicU64,
    writer:AtomicU64,role:AtomicI32,mode:AtomicI32,sample:AtomicI64,rate:AtomicU64,observed_playing:AtomicBool,
    system_time:AtomicI64,system_valid:AtomicBool,continuous:AtomicI64,continuous_valid:AtomicBool,
    stops:AtomicU64,playback_stops:AtomicU64,editor_stops:AtomicU64,stop_writer:AtomicU64,stop_role:AtomicI32,
}
impl TransportClock {
    /// UI线程发布所属工程实际听到位置；bool占f64最低位，误差小于1ULP且元组原子一致。
    pub fn publish_host(&self,position:f64,playing:bool) {
        if !position.is_finite() {return;}
        self.host_pose.store((position.to_bits()&!1)|u64::from(playing),Ordering::Release);
        self.host_valid.store(true,Ordering::Release);
    }
    /// 只采集原子诊断，不修改播放权威/不打印日志；并发快照是近似观测，不作精确同步证明。
    pub fn observe(&self,context:&crate::audio_abi::ProcessContext,writer:u64,role:i32,mode:i32) {
        match mode {0=>{self.realtime.fetch_add(1,Ordering::Relaxed);},1=>{self.prefetch.fetch_add(1,Ordering::Relaxed);},
            2=>{self.offline.fetch_add(1,Ordering::Relaxed);},_=>return,}
        if !context.sample_rate.is_finite()||context.sample_rate<=0. {return;}
        let old_writer=self.writer.swap(writer,Ordering::Relaxed);
        if old_writer!=0 && old_writer!=writer {self.switches.fetch_add(1,Ordering::Relaxed);}
        let old_sample=self.sample.swap(context.project_time_samples,Ordering::Relaxed);
        let old_rate=self.rate.swap(context.sample_rate.to_bits(),Ordering::Relaxed);
        let playing=context.state & (1<<1)!=0;
        let old_playing=self.observed_playing.swap(playing,Ordering::Relaxed);
        if playing&&old_playing&&old_rate==context.sample_rate.to_bits()&&context.project_time_samples<old_sample {
            self.backwards.fetch_add(1,Ordering::Relaxed);
        }
        self.role.store(role,Ordering::Relaxed);self.mode.store(mode,Ordering::Relaxed);
        self.system_valid.store(context.state & (1<<8)!=0,Ordering::Relaxed);
        self.system_time.store(context.system_time,Ordering::Relaxed);
        self.continuous_valid.store(context.state & (1<<17)!=0,Ordering::Relaxed);
        self.continuous.store(context.continuous_time_samples,Ordering::Relaxed);
    }
    /// 停处理的实例来源也需记录；调用者可能是瞬时clip renderer，并不等于宿主整体停止。
    pub fn observe_stop(&self,writer:u64,role:i32) {
        self.stops.fetch_add(1,Ordering::Relaxed);
        match role {1=>{self.playback_stops.fetch_add(1,Ordering::Relaxed);},2=>{self.editor_stops.fetch_add(1,Ordering::Relaxed);},_=>{}}
        self.stop_writer.store(writer,Ordering::Relaxed);self.stop_role.store(role,Ordering::Relaxed);
    }
    /// 仅非实时actor查询，JSON分配绝不进入process/setProcessing。
    pub fn diagnostics(&self)->serde_json::Value {
        serde_json::json!({"realtime_callbacks":self.realtime.load(Ordering::Relaxed),
            "reaper_position_authority":self.host_valid.load(Ordering::Acquire),
            "prefetch_callbacks":self.prefetch.load(Ordering::Relaxed),"offline_callbacks":self.offline.load(Ordering::Relaxed),
            "writer_switches":self.switches.load(Ordering::Relaxed),"backwards_while_playing":self.backwards.load(Ordering::Relaxed),
            "last_writer":self.writer.load(Ordering::Relaxed),"last_role":self.role.load(Ordering::Relaxed),
            "last_mode":self.mode.load(Ordering::Relaxed),"last_project_samples":self.sample.load(Ordering::Relaxed),
            "last_sample_rate":f64::from_bits(self.rate.load(Ordering::Relaxed)),"last_playing":self.observed_playing.load(Ordering::Relaxed),
            "system_time":self.system_time.load(Ordering::Relaxed),"system_time_valid":self.system_valid.load(Ordering::Relaxed),
            "continuous_samples":self.continuous.load(Ordering::Relaxed),"continuous_valid":self.continuous_valid.load(Ordering::Relaxed),
            "processing_stops":self.stops.load(Ordering::Relaxed),"playback_stops":self.playback_stops.load(Ordering::Relaxed),
            "editor_stops":self.editor_stops.load(Ordering::Relaxed),"last_stop_writer":self.stop_writer.load(Ordering::Relaxed),
            "last_stop_role":self.stop_role.load(Ordering::Relaxed)})
    }
    /// VST3锁定header中kPlaying=1<<1；project_time_samples不需要音乐时间valid位。
    pub fn update(&self,context:&crate::audio_abi::ProcessContext) {
        if context.sample_rate.is_finite() && context.sample_rate>0. {
            self.position.store((context.project_time_samples as f64/context.sample_rate).to_bits(),Ordering::Release);
            self.playing.store(context.state & (1<<1)!=0,Ordering::Release);
            // 锁定ivstprocesscontext.h的kTempoValid=1<<10；不可用字段不能覆盖最后有效BPM。
            if context.state & (1<<10)!=0 && context.tempo.is_finite() && context.tempo>0. {
                self.tempo.store(context.tempo.to_bits(),Ordering::Release);
            }
        }
    }
    pub fn stopped(&self) {self.playing.store(false,Ordering::Release);}
    /// 没有有效宿主tempo时返回None，由映射初始值兜底。
    pub fn tempo(&self)->Option<f64> {let value=f64::from_bits(self.tempo.load(Ordering::Acquire));(value>0. && value.is_finite()).then_some(value)}
    /// 多个renderer可能并行发布同一文档块；展示容忍一块误差，不参与DSP寻址。
    pub fn read(&self)->(f64,bool) {
        if self.host_valid.load(Ordering::Acquire) {
            let pose=self.host_pose.load(Ordering::Acquire);return (f64::from_bits(pose&!1),pose&1!=0);
        }
        (f64::from_bits(self.position.load(Ordering::Acquire)),self.playing.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 真实工程时钟不受预取提前位置或其它clip停止覆盖，后退seek/loop仍能发布。
    #[test]
    fn actual_host_position_survives_prefetch_noise_and_accepts_backward_seeks() {
        let clock=TransportClock::default();clock.publish_host(1.25,true);
        clock.update(&crate::audio_abi::ProcessContext {state:1<<1,sample_rate:44100.,project_time_samples:441000,..Default::default()});
        clock.stopped();assert_eq!(clock.read(),(1.25,true));
        clock.publish_host(0.5,true);assert_eq!(clock.read(),(0.5,true));
        clock.publish_host(0.25,false);assert_eq!(clock.read(),(0.25,false));
        assert_eq!(clock.diagnostics()["reaper_position_authority"],true);
    }
    /// 诊断真实区分三模式/多writer/回跳/瞬时停止，但本身不能改变GUI播放时间。
    #[test]
    fn diagnostics_count_sources_without_changing_the_transport() {
        let clock=TransportClock::default();
        let mut context=crate::audio_abi::ProcessContext {state:(1<<1)|(1<<8)|(1<<17),sample_rate:44100.,project_time_samples:44100,
            system_time:123456,continuous_time_samples:44100,..Default::default()};
        clock.observe(&context,1,2,0);context.project_time_samples=88200;clock.observe(&context,2,1,1);
        context.project_time_samples=44100;clock.observe(&context,1,2,0);clock.observe(&context,3,1,2);
        clock.observe_stop(2,1);let info=clock.diagnostics();
        assert_eq!(info["realtime_callbacks"],2);assert_eq!(info["prefetch_callbacks"],1);assert_eq!(info["offline_callbacks"],1);
        assert_eq!(info["writer_switches"],3);assert_eq!(info["backwards_while_playing"],1);assert_eq!(info["playback_stops"],1);
        assert_eq!(info["system_time_valid"],true);assert_eq!(info["continuous_valid"],true);assert_eq!(clock.read(),(0.0,false));
    }
    #[test]
    fn only_valid_host_tempo_updates_bpm() {
        let clock=TransportClock::default();
        let mut context=crate::audio_abi::ProcessContext {sample_rate:44100.,tempo:150.,..Default::default()};
        clock.update(&context);assert_eq!(clock.tempo(),None);
        context.state=1<<10;clock.update(&context);assert_eq!(clock.tempo(),Some(150.));
        for tempo in [f64::NAN,0.,-1.] {context.tempo=tempo;clock.update(&context);assert_eq!(clock.tempo(),Some(150.));}
    }
}
