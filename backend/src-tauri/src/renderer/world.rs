//! 基于 WORLD 声码器的渲染器实现。

use super::traits::{RenderContext, Renderer, RendererCapabilities};
use super::utils::{clip_midi_at_time, edit_midi_at_time_or_none};
use crate::state::SynthPipelineKind;

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
                    return 0.0;
                }
                let target = match edit_midi_at_time_or_none(fp, pitch_edit, abs_time_sec) {
                    Some(v) => v,
                    None => orig,
                };
                let shift = (target - orig).clamp(-24.0, 24.0);
                if shift.is_finite() {
                    shift
                } else {
                    0.0
                }
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
