//! 声码器参数能力查询命令。
//!
//! 提供 `get_processor_params(algo)` 命令，返回指定算法支持的参数描述符列表。
//! 前端据此动态渲染参数面板（Tab 标签 + 曲线编辑器）。
//!
//! 返回内容 = **共通混音级参数（volume / pan / dyn，与算法无关）** + 算法专有参数。
//! 见 `renderer::all_param_descriptors`。

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

/// 查询指定算法可编辑的参数描述符列表。
///
/// # 参数
/// - `algo`：算法标识字符串，例 "world_dll"、"nsf_hifigan_onnx"、"vslib"、"none"。
///
/// # 返回值
/// 共通混音级参数（volume / pan / dyn）+ 该算法链路所有 [`ParamDescriptor`] 的
/// 可序列化 DTO 列表。音高面板（pitch）不在此列表中，由前端固定显示。
pub(super) fn get_processor_params(algo: String) -> Vec<ParamDescriptorDto> {
    let kind = algo_to_kind(&algo);
    crate::renderer::all_param_descriptors(kind)
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
///
/// 复用 [`PitchAnalysisAlgo::from_id`] + [`SynthPipelineKind::from_track_algo`]，
/// 不再自持一份映射：此前的副本漏掉了 `world_dll`（靠 `_` 兜底才碰巧正确），
/// 又把未识别值兜成 WORLD —— 与 `from_track_algo` 的"未知 → 默认算法"不一致，
/// 于是界面显示 nsf-hifigan 的参数集、轨道头却写着别的算法。
fn algo_to_kind(algo: &str) -> crate::state::SynthPipelineKind {
    use crate::state::{PitchAnalysisAlgo, SynthPipelineKind};
    SynthPipelineKind::from_track_algo(&PitchAnalysisAlgo::from_id(algo))
}
