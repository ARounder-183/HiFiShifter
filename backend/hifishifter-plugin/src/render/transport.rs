//! 实时回调发布的宿主播放时钟；GUI读近似最新块，process只执行原子写入。
use std::sync::atomic::{AtomicBool,AtomicU64,Ordering};
#[derive(Default)]
pub(crate) struct TransportClock {position:AtomicU64,playing:AtomicBool,tempo:AtomicU64}
impl TransportClock {
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
        (f64::from_bits(self.position.load(Ordering::Acquire)),self.playing.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_valid_host_tempo_updates_bpm() {
        let clock=TransportClock::default();
        let mut context=crate::audio_abi::ProcessContext {sample_rate:44100.,tempo:150.,..Default::default()};
        clock.update(&context);assert_eq!(clock.tempo(),None);
        context.state=1<<10;clock.update(&context);assert_eq!(clock.tempo(),Some(150.));
        for tempo in [f64::NAN,0.,-1.] {context.tempo=tempo;clock.update(&context);assert_eq!(clock.tempo(),Some(150.));}
    }
}
