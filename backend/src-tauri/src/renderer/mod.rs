//! 渲染器模块：统一的音高合成渲染接口。
//!
//! 通过 [`Renderer`] trait 将合成链路与调用方解耦，
//! 未来新增渲染器只需实现该 trait 并在此处注册，
//! 无需修改 `pitch_editing.rs` 等核心逻辑。
//!
//! `get_processor()` 返回统一的 `ClipProcessor` 实例，涵盖音高合成 +
//! 时间拉伸 + 全部声码器参数曲线。

pub(crate) mod chain;
pub(crate) mod hifigan;
mod traits;
mod utils;
pub(crate) mod world;

#[cfg(feature = "vslib")]
pub(crate) mod vslib_processor;

pub use chain::{ProcessingStage, ProcessorChain, StageContext};
pub use traits::{
    ClipProcessContext, ClipProcessor, ParamDescriptor, ParamKind, ProcessorCapabilities,
    RenderContext, Renderer, RendererCapabilities,
};
pub use utils::{clip_midi_at_time, edit_midi_at_time_or_none};

use crate::state::SynthPipelineKind;

// ─── 静态实例（Renderer，for backwards compat）────────────────────────────────

static WORLD_RENDERER: world::WorldRenderer = world::WorldRenderer;
static HIFIGAN_RENDERER: hifigan::HiFiGanRenderer = hifigan::HiFiGanRenderer;
#[cfg(feature = "vslib")]
static VSLIB_RENDERER: vslib_processor::VslibRenderer = vslib_processor::VslibRenderer;

// ─── 注册表 ────────────────────────────────────────────────────────────────────

/// 根据 [`SynthPipelineKind`] 返回对应的静态渲染器实例。
///
/// 使用静态分发（`&'static dyn Renderer`）避免堆分配，
/// 渲染器数量固定，静态分发足够高效。
pub fn get_renderer(kind: SynthPipelineKind) -> &'static dyn Renderer {
    match kind {
        SynthPipelineKind::WorldVocoder => &WORLD_RENDERER,
        SynthPipelineKind::NsfHifiganOnnx => &HIFIGAN_RENDERER,
        #[cfg(feature = "vslib")]
        SynthPipelineKind::VocalShifterVslib => &VSLIB_RENDERER,
    }
}

/// 列出所有已注册的渲染器（供前端 UI 展示或调试）。
#[allow(dead_code)]
pub fn all_renderers() -> Vec<&'static dyn Renderer> {
    vec![&WORLD_RENDERER, &HIFIGAN_RENDERER]
}

// ─── ClipProcessor 注册表 ──────────────────────────────────────────────────────

/// 根据 [`SynthPipelineKind`] 创建对应的 [`ClipProcessor`] 实例（Box 分配）。
///
/// 对于 World / HiFiGAN，返回对应的 [`ProcessorChain`]（含 Signalsmith Stretch + 声码器 Stage）。
/// 对于 vslib，返回 [`VslibProcessor`]（需 `feature = "vslib"`）。
pub fn get_processor(kind: SynthPipelineKind) -> Box<dyn ClipProcessor> {
    match kind {
        SynthPipelineKind::WorldVocoder => Box::new(chain::world_chain()),
        SynthPipelineKind::NsfHifiganOnnx => Box::new(chain::hifigan_chain()),
        #[cfg(feature = "vslib")]
        SynthPipelineKind::VocalShifterVslib => Box::new(vslib_processor::VslibProcessor),
    }
}

pub fn get_param_descriptor(kind: SynthPipelineKind, param_id: &str) -> Option<ParamDescriptor> {
    get_processor(kind)
        .param_descriptors()
        .into_iter()
        .find(|descriptor| descriptor.id == param_id)
}

pub fn automation_curve_default_value(kind: SynthPipelineKind, param_id: &str) -> Option<f32> {
    match get_param_descriptor(kind, param_id)?.kind {
        ParamKind::AutomationCurve { default_value, .. } => Some(default_value),
        _ => None,
    }
}

pub fn static_enum_default_value(kind: SynthPipelineKind, param_id: &str) -> Option<i32> {
    match get_param_descriptor(kind, param_id)?.kind {
        ParamKind::StaticEnum { default_value, .. } => Some(default_value),
        _ => None,
    }
}
