//! 把 `TimelineState` 交给本体离线内核渲染的薄封装。
//!
//! 存在的意义：插件侧与探针都不该直接构造一大坨 `MixdownOptions`，
//! 也不该各自决定用哪个拉伸算法。这里把"用本体内核渲染一段区间"收敛成一个入口，
//! 供逐样本比对与后续 ARA renderer 复用。

// 【为什么路径是分模块的而不像 app 那样有一层扁平 re-export】app 的
// `pub mod kernel` 是它给探针留的窗口；插件直接依赖内核 crate，按内核自己的
// 模块结构取用更清楚 —— 也免得插件误以为内核只有一个扁平命名空间。
use hifishifter_kernel::encode::OutputSpec;
use hifishifter_kernel::mixdown::{render_mixdown_interleaved, MixdownOptions, QualityPreset};
use hifishifter_kernel::state::TimelineState;
use hifishifter_kernel::time_stretch::StretchAlgorithm;

pub(crate) mod ownership;
pub(crate) mod extension;
pub(crate) mod document;
pub(crate) mod source;
pub(crate) mod snapshot;
pub(crate) mod budget;

/// 一段离线渲染产物。
#[derive(Debug, Clone)]
pub struct RenderedAudio {
    /// 输出采样率。
    pub sample_rate: u32,
    /// 输出声道数。
    pub channels: u16,
    /// 交织的 f32 采样。
    pub samples: Vec<f32>,
}

impl RenderedAudio {
    /// 按声道数把交织采样切成每声道一条。
    pub fn deinterleave(&self) -> Vec<Vec<f32>> {
        let channels = self.channels.max(1) as usize;
        let mut planes = vec![Vec::with_capacity(self.samples.len() / channels); channels];
        for frame in self.samples.chunks_exact(channels) {
            for (plane, sample) in planes.iter_mut().zip(frame) {
                plane.push(*sample);
            }
        }
        planes
    }
}

/// 用本体内核把 `[start_sec, end_sec)` 渲染成交织 f32。
///
/// 约定：不做音高编辑（`apply_pitch_edit = false`），拉伸算法用 Windows 上本体的
/// 默认实现。插件侧的 ARA renderer 后续会复用同一入口。
pub fn render_timeline(
    timeline: &TimelineState,
    sample_rate: u32,
    start_sec: f64,
    end_sec: f64,
) -> Result<RenderedAudio, String> {
    let options = MixdownOptions {
        sample_rate,
        start_sec,
        end_sec: Some(end_sec),
        stretch: StretchAlgorithm::SoundTouchDll,
        apply_pitch_edit: false,
        output: OutputSpec::wav_32f(),
        quality_preset: QualityPreset::Realtime,
        cancel_flag: None,
        progress: None,
        cache_stats: None,
    };
    let (sample_rate, channels, _duration_sec, samples) =
        render_mixdown_interleaved(timeline, options)?;
    Ok(RenderedAudio {
        sample_rate,
        channels,
        samples,
    })
}
