//! 基于 WORLD 声码器的渲染器实现。

use super::traits::{RenderContext, Renderer, RendererCapabilities};
use super::utils::{clip_midi_at_time, edit_midi_at_time_or_none};
use crate::state::SynthPipelineKind;
use crate::world_vocoder::PitchFrame;

/// 基于 WORLD 声码器的渲染器。
pub struct WorldRenderer;

impl Renderer for WorldRenderer {
    fn id(&self) -> &str {
        "world_vocoder"
    }

    fn display_name(&self) -> &str {
        "WORLD Vocoder"
    }

    fn kind(&self) -> SynthPipelineKind {
        SynthPipelineKind::WorldVocoder
    }

    fn is_available(&self) -> bool {
        crate::world_vocoder::is_available()
    }

    fn render(&self, ctx: &RenderContext<'_>) -> Result<Vec<f32>, String> {
        let f0_floor = 40.0;
        let f0_ceil = 1600.0;
        let fp = ctx.frame_period_ms;
        let clip_start = ctx.clip_start_sec;
        let pitch_edit = ctx.pitch_edit;
        let clip_midi = ctx.clip_midi;

        // 移调指令按**两条绝对音高**下发：源音高 = 工程里的音高线（这一帧有没有
        // 音高、包络该按哪个周期分析），目标音高 = 编辑后的音高（激励频率）。
        // 半音数在这里就换算完，声码器内部不再做"绝对 × 比率"的二次乘法 ——
        // 那条路会让输出的绝对音准取决于 WORLD 自己的估计器，而不是音高线。
        crate::world_vocoder::vocode_pitch_shift_chunked(
            ctx.mono_pcm,
            ctx.sample_rate,
            ctx.seg_start_sec,
            fp,
            f0_floor,
            f0_ceil,
            move |abs_time_sec| {
                let orig = clip_midi_at_time(fp, clip_start, clip_midi, abs_time_sec);
                if !(orig.is_finite() && orig > 0.0) {
                    return PitchFrame::UNPITCHED;
                }
                let target = match edit_midi_at_time_or_none(fp, pitch_edit, abs_time_sec) {
                    Some(v) if v.is_finite() && v > 0.0 => v,
                    _ => orig,
                };
                PitchFrame::shifted(orig, target - orig)
            },
        )
    }

    fn capabilities(&self) -> RendererCapabilities {
        RendererCapabilities {
            supports_realtime: true,
            prefers_prerender: false,
            max_pitch_shift_semitones: 24.0,
        }
    }
}
