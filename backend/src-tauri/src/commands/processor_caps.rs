//! 声码器参数能力查询命令。
//!
//! 提供 `get_processor_params(algo)` 命令，返回指定算法支持的额外参数描述符列表。
//! 前端据此动态渲染参数面板（Tab 标签 + 曲线编辑器）。

use crate::renderer::ParamKind;
use serde::Serialize;

// ─── 可序列化 DTO ─────────────────────────────────────────────────────────────

/// 参数种类（序列化给前端）。
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamKindDto {
    AutomationCurve {
        unit: &'static str,
        default_value: f32,
        min_value: f32,
        max_value: f32,
    },
    StaticEnum {
        options: Vec<(&'static str, i32)>,
        default_value: i32,
    },
}

/// 参数描述符（序列化给前端）。
#[derive(Debug, Serialize)]
pub struct ParamDescriptorDto {
    pub id: &'static str,
    pub display_name: &'static str,
    pub group: &'static str,
    pub kind: ParamKindDto,
}

// ─── 命令实现 ─────────────────────────────────────────────────────────────────

/// 查询指定算法的额外参数描述符列表。
///
/// # 参数
/// - `algo`：算法标识字符串，例 "world_dll"、"nsf_hifigan_onnx"、"vslib"、"none"。
///
/// # 返回值
/// 对应声码器链路所有 [`ParamDescriptor`] 的可序列化 DTO 列表。
/// 音高面板（pitch）不在此列表中，由前端固定显示。
pub(super) fn get_processor_params(algo: String) -> Vec<ParamDescriptorDto> {
    let kind = algo_to_kind(&algo);
    let processor = crate::renderer::get_processor(kind);
    processor
        .param_descriptors()
        .into_iter()
        .map(|d| ParamDescriptorDto {
            id: d.id,
            display_name: d.display_name,
            group: d.group,
            kind: match d.kind {
                ParamKind::AutomationCurve {
                    unit,
                    default_value,
                    min_value,
                    max_value,
                } => ParamKindDto::AutomationCurve {
                    unit,
                    default_value,
                    min_value,
                    max_value,
                },
                ParamKind::StaticEnum {
                    options,
                    default_value,
                } => ParamKindDto::StaticEnum {
                    options: options.to_vec(),
                    default_value,
                },
            },
        })
        .collect()
}

/// 将前端算法字符串映射到 `SynthPipelineKind`。
/// 未知 algo 回退到 WorldVocoder（与 `SynthPipelineKind::from_track_algo` 保持一致）。
fn algo_to_kind(algo: &str) -> crate::state::SynthPipelineKind {
    use crate::state::SynthPipelineKind;
    match algo {
        "nsf_hifigan_onnx" => SynthPipelineKind::NsfHifiganOnnx,
        #[cfg(feature = "vslib")]
        "vslib" | "vocalshifter_vslib" => SynthPipelineKind::VocalShifterVslib,
        _ => SynthPipelineKind::WorldVocoder,
    }
}

#[cfg(test)]
mod tests {
    use super::{get_processor_params, ParamKindDto};

    #[test]
    fn nsf_hifigan_exposes_breath_and_tension_params() {
        let params = get_processor_params("nsf_hifigan_onnx".to_string());

        let breath_enabled = params
            .iter()
            .find(|param| param.id == "breath_enabled")
            .expect("expected breath_enabled static param");
        match &breath_enabled.kind {
            ParamKindDto::StaticEnum {
                options,
                default_value,
            } => {
                assert_eq!(*default_value, 0);
                assert!(options
                    .iter()
                    .any(|(label, value)| *label == "Off" && *value == 0));
                assert!(options
                    .iter()
                    .any(|(label, value)| *label == "On" && *value == 1));
            }
            _ => panic!("breath_enabled should be a static enum"),
        }

        let breath_gain = params
            .iter()
            .find(|param| param.id == "breath_gain")
            .expect("expected breath_gain automation curve");
        match &breath_gain.kind {
            ParamKindDto::AutomationCurve {
                default_value,
                min_value,
                max_value,
                ..
            } => {
                assert!((*default_value - 1.0).abs() < 1e-6);
                assert!((*min_value - 0.0).abs() < 1e-6);
                assert!((*max_value - 2.0).abs() < 1e-6);
            }
            _ => panic!("breath_gain should be an automation curve"),
        }

        let tension = params
            .iter()
            .find(|param| param.id == "hifigan_tension")
            .expect("expected hifigan_tension automation curve");
        match &tension.kind {
            ParamKindDto::AutomationCurve {
                default_value,
                min_value,
                max_value,
                ..
            } => {
                assert!(default_value.abs() < 1e-6);
                assert!((*min_value + 100.0).abs() < 1e-6);
                assert!((*max_value - 100.0).abs() < 1e-6);
            }
            _ => panic!("hifigan_tension should be an automation curve"),
        }
    }
}
