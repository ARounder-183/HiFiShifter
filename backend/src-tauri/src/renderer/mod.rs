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
pub(crate) mod external_resampler;

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

/// 根据 [`SynthPipelineKind`] 返回对应的渲染器 ID 字符串（用于缓存 key）。
///
/// 对于内置渲染器使用静态实例；对于外部 resampler 返回对应的 entry id。
pub fn get_renderer(kind: &SynthPipelineKind) -> &'static dyn Renderer {
    match kind {
        SynthPipelineKind::WorldVocoder => &WORLD_RENDERER,
        SynthPipelineKind::NsfHifiganOnnx => &HIFIGAN_RENDERER,
        #[cfg(feature = "vslib")]
        SynthPipelineKind::VocalShifterVslib => &VSLIB_RENDERER,
        // 外部 resampler 不通过 get_renderer 使用（走 get_processor），
        // 此处返回 WORLD_RENDERER 作为 fallback。
        SynthPipelineKind::ExternalResampler(_) => &WORLD_RENDERER,
    }
}

/// 返回渲染器 ID 字符串（用于缓存 key），支持外部 resampler。
pub fn renderer_id(kind: &SynthPipelineKind) -> String {
    match kind {
        SynthPipelineKind::WorldVocoder => "world_vocoder".to_string(),
        SynthPipelineKind::NsfHifiganOnnx => "nsf_hifigan_onnx".to_string(),
        #[cfg(feature = "vslib")]
        SynthPipelineKind::VocalShifterVslib => "vslib".to_string(),
        SynthPipelineKind::ExternalResampler(id) => format!("external_resampler:{}", id),
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
/// 对于外部 resampler，需要从 AppState 的 registry 中查找 entry。
pub fn get_processor(kind: SynthPipelineKind) -> Box<dyn ClipProcessor> {
    match kind {
        SynthPipelineKind::WorldVocoder => Box::new(chain::world_chain()),
        SynthPipelineKind::NsfHifiganOnnx => Box::new(chain::hifigan_chain()),
        #[cfg(feature = "vslib")]
        SynthPipelineKind::VocalShifterVslib => {
            Box::new(vslib_processor::VslibProcessor)
        }
        SynthPipelineKind::ExternalResampler(ref id) => {
            // 需要从全局 registry 获取 entry——这里使用一个 fallback 的默认 entry
            // 实际使用时由调用方通过 get_processor_with_registry() 提供。
            eprintln!("[renderer] get_processor called for external resampler '{}' without registry context", id);
            Box::new(chain::world_chain()) // fallback
        }
    }
}

/// 根据 [`SynthPipelineKind`] 创建 [`ClipProcessor`]，支持从 registry 查找外部 resampler。
pub fn get_processor_with_registry(
    kind: SynthPipelineKind,
    registry: &crate::state::ResamplerRegistry,
) -> Box<dyn ClipProcessor> {
    match kind {
        SynthPipelineKind::ExternalResampler(ref id) => {
            match registry.get(id) {
                Some(entry) => {
                    if !entry.available {
                        eprintln!(
                            "[renderer] 外部 Resampler '{}' 已注册但不可用 (exe 不存在: {})",
                            entry.display_name,
                            entry.exe_path.display(),
                        );
                    }
                    Box::new(external_resampler::ExternalResamplerProcessor::new(entry.clone()))
                }
                None => {
                    eprintln!(
                        "[renderer] 外部 Resampler '{}' 未在注册表中找到，回退到 WorldVocoder",
                        id
                    );
                    Box::new(chain::world_chain())
                }
            }
        }
        other => get_processor(other),
    }
}

pub fn get_param_descriptor(
    kind: &SynthPipelineKind,
    param_id: &str,
) -> Option<ParamDescriptor> {
    get_processor(kind.clone())
        .param_descriptors()
        .into_iter()
        .find(|descriptor| descriptor.id == param_id)
}

pub fn automation_curve_default_value(
    kind: &SynthPipelineKind,
    param_id: &str,
) -> Option<f32> {
    match get_param_descriptor(kind, param_id)?.kind {
        ParamKind::AutomationCurve { default_value, .. } => Some(default_value),
        _ => None,
    }
}

pub fn static_enum_default_value(
    kind: &SynthPipelineKind,
    param_id: &str,
) -> Option<i32> {
    match get_param_descriptor(kind, param_id)?.kind {
        ParamKind::StaticEnum { default_value, .. } => Some(default_value),
        _ => None,
    }
}
