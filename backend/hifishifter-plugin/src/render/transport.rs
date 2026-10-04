//! 实时回调发布的宿主播放时钟；GUI读近似最新块，process只执行原子写入。
use std::sync::atomic::{AtomicBool,AtomicU64,Ordering};
#[derive(Default)]
pub(crate) struct TransportClock {position:AtomicU64,playing:AtomicBool}
impl TransportClock {
    /// VST3锁定header中kPlaying=1<<1；project_time_samples不需要音乐时间valid位。
    pub fn update(&self,context:&crate::audio_abi::ProcessContext) {
        if context.sample_rate.is_finite() && context.sample_rate>0. {
            self.position.store((context.project_time_samples as f64/context.sample_rate).to_bits(),Ordering::Release);
            self.playing.store(context.state & (1<<1)!=0,Ordering::Release);
        }
    }
    pub fn stopped(&self) {self.playing.store(false,Ordering::Release);}
    /// 多个renderer可能并行发布同一文档块；展示容忍一块误差，不参与DSP寻址。
    pub fn read(&self)->(f64,bool) {
        (f64::from_bits(self.position.load(Ordering::Acquire)),self.playing.load(Ordering::Acquire))
    }
}
