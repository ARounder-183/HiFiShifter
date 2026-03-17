//! 渲染器统一接口定义。
//!
//! 通过 [`Renderer`] trait 将合成链路与调用方解耦，
//! 未来新增渲染器只需实现该 trait 并在 `mod.rs` 中注册。
//!
//! 同时提供更高层的 [`ClipProcessor`] trait，统一封装全链路处理
//! （音高合成 + 时间拉伸 + 所有声码器参数曲线）。

use crate::state::SynthPipelineKind;
use std::collections::HashMap;

// ─── 上下文 ────────────────────────────────────────────────────────────────────

/// 传递给渲染器的处理上下文（借用，零拷贝）。
pub struct RenderContext<'a> {
    /// 单声道 PCM 输入（f32，已归一化）。
    pub mono_pcm: &'a [f32],
    /// 采样率（Hz）。
    pub sample_rate: u32,
    /// 当前片段在时间轴上的起始时间（秒）。
    pub seg_start_sec: f64,
    /// 当前片段在时间轴上的结束时间（秒）。
    pub seg_end_sec: f64,
    /// 所属 Clip 在时间轴上的起始时间（秒），用于 MIDI 曲线对齐。
    pub clip_start_sec: f64,
    /// 分析帧周期（毫秒）。
    pub frame_period_ms: f64,
    /// 全局 pitch_edit 曲线（绝对 MIDI；0 表示无编辑）。
    pub pitch_edit: &'a [f32],
    /// Clip 原始 MIDI 曲线（时间轴对齐）。
    pub clip_midi: &'a [f32],
    /// 所属 Clip 的唯一标识，用于 per-segment 推理缓存。
    pub clip_id: &'a str,
}

// ─── 能力描述 ──────────────────────────────────────────────────────────────────

/// 渲染器能力描述。
#[derive(Debug, Clone)]
pub struct RendererCapabilities {
    /// 是否支持实时渲染（audio callback 级低延迟）。
    pub supports_realtime: bool,
    /// 是否推荐预渲染（在后台线程执行）。
    pub prefers_prerender: bool,
    /// 最大支持的变调半音数。
    pub max_pitch_shift_semitones: f64,
}

impl Default for RendererCapabilities {
    fn default() -> Self {
        Self {
            supports_realtime: false,
            prefers_prerender: true,
            max_pitch_shift_semitones: 24.0,
        }
    }
}

// ─── Trait ──────────────────────────────────────────────────────────────────────

/// 渲染器插件接口（类似 OpenUtau 的 IRenderer）。
///
/// 实现者必须是 `Send + Sync`，以便在多线程渲染中安全使用。
pub trait Renderer: Send + Sync {
    /// 渲染器唯一标识符（如 "world_vocoder", "nsf_hifigan_onnx"）。
    fn id(&self) -> &str;

    /// 人类可读的显示名称。
    fn display_name(&self) -> &str;

    /// 返回该渲染器对应的 [`SynthPipelineKind`]。
    fn kind(&self) -> SynthPipelineKind;

    /// 检查渲染器是否可用（动态库已加载 / ONNX 模型已就绪等）。
    fn is_available(&self) -> bool;

    /// 对输入 PCM 执行音高编辑，返回处理后的单声道 PCM。
    fn render(&self, ctx: &RenderContext<'_>) -> Result<Vec<f32>, String>;

    /// 声明该渲染器支持的能力。
    fn capabilities(&self) -> RendererCapabilities {
        RendererCapabilities::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ClipProcessor：全链路合成接口（Phase 1 新增）
// ═══════════════════════════════════════════════════════════════════════════════

// ─── ClipProcessContext ────────────────────────────────────────────────────────

/// 传递给 `ClipProcessor` 的全链路处理上下文（零拷贝借用）。
pub struct ClipProcessContext<'a> {
    /// 输入单声道 PCM（f32，已归一化）。
    pub mono_pcm: &'a [f32],
    /// 采样率（Hz）。
    pub sample_rate: u32,
    /// Clip 在时间轴上的起点（秒），用于曲线对齐。
    pub clip_start_sec: f64,
    /// 本次待处理片段在时间轴上的起点（秒）。
    pub seg_start_sec: f64,
    /// 本次待处理片段在时间轴上的终点（秒）。
    pub seg_end_sec: f64,
    /// 声码器帧周期（ms），决定 pitch_edit / extra_curves 长度。
    pub frame_period_ms: f64,
    /// 每帧绝对 MIDI 音高（cent），由音高编辑层输出。
    pub pitch_edit: &'a [f32],
    /// 每帧 Clip 原始 MIDI 音高（cent），用于计算相对偏移。
    pub clip_midi: &'a [f32],
    /// 回放速率（时间拉伸比例）；1.0 = 不拉伸。
    pub playback_rate: f64,
    /// 输出帧数（应用 playback_rate 后）。
    pub out_frames: usize,
    /// 用于缓存 key 的 Clip 唯一 ID。
    pub clip_id: &'a str,
    /// 声码器专属自动化曲线（逐帧 Vec<f32>，`AutomationCurve` 类型参数）。
    /// key = `ParamDescriptor::id`；缺失 key 表示使用该参数的默认值。
    pub extra_curves: &'a HashMap<String, Vec<f32>>,
    /// 声码器专属静态参数（`StaticEnum` 类型参数，枚举整数值以 f64 存储）。
    /// key = `ParamDescriptor::id`。
    pub extra_params: &'a HashMap<String, f64>,
}

// ─── ProcessorCapabilities ────────────────────────────────────────────────────

/// `ClipProcessor` 能力描述。
#[derive(Debug, Clone, Default)]
pub struct ProcessorCapabilities {
    /// 是否原生处理 `playback_rate`（= true 时 compat 层不再调 Signalsmith Stretch）。
    pub handles_time_stretch: bool,
    /// 是否支持逐帧共振峰偏移曲线（"formant_shift_cents"）。
    pub supports_formant: bool,
    /// 是否支持逐帧气声强度曲线（"breathiness"）。
    pub supports_breathiness: bool,
}

// ─── ParamDescriptor ──────────────────────────────────────────────────────────

/// 声码器参数种类。
#[derive(Debug, Clone)]
pub enum ParamKind {
    /// 逐帧自动化曲线，显示在时间轴上，存入 `extra_curves`。
    AutomationCurve {
        /// 单位字符串，例："cents"、"×"、""。
        unit: &'static str,
        default_value: f32,
        min_value: f32,
        max_value: f32,
    },
    /// 静态枚举，前端渲染为按钮切换组，存入 `extra_params`（值为 i32 转 f64）。
    StaticEnum {
        /// 选项列表：(显示名, 整数值)。
        options: &'static [(&'static str, i32)],
        default_value: i32,
    },
}

/// 声码器参数描述符（静态生命周期，可被 `param_descriptors()` 返回 `&'static [Self]`）。
#[derive(Debug, Clone)]
pub struct ParamDescriptor {
    /// 参数唯一标识符；同时用作 `extra_curves` / `extra_params` 的 HashMap key。
    pub id: &'static str,
    /// 人类可读显示名称。
    pub display_name: &'static str,
    /// 前端面板分组标题。
    pub group: &'static str,
    /// 参数种类。
    pub kind: ParamKind,
}

// ─── ClipProcessor trait ──────────────────────────────────────────────────────

/// 全链路合成插件接口。
///
/// 一次 `process()` 调用涵盖——
/// - 音高合成（声码器内核）
/// - 时间拉伸（原生或 Signalsmith Stretch Stage）
/// - 所有声码器参数曲线（共振峰、气声等）
///
/// 实现者须 `Send + Sync`，以便在多线程场景下安全使用。
pub trait ClipProcessor: Send + Sync {
    /// 处理器唯一标识符（与 `SynthPipelineKind` 对应）。
    fn id(&self) -> &str;
    /// 人类可读显示名称。
    fn display_name(&self) -> &str;
    /// 运行时可用性检查（例：vslib 仅在 DLL 已加载时为 true）。
    fn is_available(&self) -> bool;
    /// 声明处理器支持的能力。
    fn capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities::default()
    }
    /// 静态声明支持的额外参数描述符（前端可据此动态渲染参数面板）。
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![]
    }
    /// 全链路处理：PCM → 合成 PCM（含音高 + 拉伸 + 声码器参数）。
    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String>;
}
