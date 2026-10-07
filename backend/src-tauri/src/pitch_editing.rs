use crate::state::{PitchAnalysisAlgo, SynthPipelineKind, TimelineState};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static MONO_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn pitch_edit_algo_from_env() -> Option<String> {
    std::env::var("HIFISHIFTER_PITCH_EDIT_ALGO")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchEditAlgorithm {
    WorldVocoder,
    NsfHifiganOnnx,
    #[cfg(feature = "vslib")]
    VocalShifterVslib,
    Bypass,
}

#[derive(Debug, Clone)]
pub(crate) struct PitchCurvesSnapshot<'a> {
    pub frame_period_ms: f64,
    pub pitch_orig: &'a [f32],
    pub pitch_edit: &'a [f32],
}

impl<'a> PitchCurvesSnapshot<'a> {
    #[allow(dead_code)]
    pub fn midi_at_time(&self, abs_time_sec: f64) -> f64 {
        if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
            return 0.0;
        }

        let inv_fp = 1000.0 / self.frame_period_ms.max(0.1);
        let idx_f = abs_time_sec * inv_fp;
        if !(idx_f.is_finite() && idx_f >= 0.0) {
            return 0.0;
        }
        let i0 = idx_f.floor() as isize;
        if i0 < 0 {
            return 0.0;
        }
        let i0 = i0 as usize;
        let len = self.pitch_orig.len().min(self.pitch_edit.len().max(1));
        if i0 >= len {
            return 0.0;
        }
        let i1 = (i0 + 1).min(len.saturating_sub(1));
        let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

        let orig0 = self.pitch_orig.get(i0).copied().unwrap_or(0.0) as f64;
        let orig1 = self.pitch_orig.get(i1).copied().unwrap_or(0.0) as f64;
        let edit0 = self.pitch_edit.get(i0).copied().unwrap_or(0.0) as f64;
        let edit1 = self.pitch_edit.get(i1).copied().unwrap_or(0.0) as f64;

        // For ONNX, `pitch_edit` is treated as an absolute target MIDI curve.
        // Allow it to work even when `pitch_orig` is missing (all zeros).
        let mut base0 = if edit0.is_finite() && edit0 > 0.0 {
            edit0
        } else {
            orig0
        };
        let mut base1 = if edit1.is_finite() && edit1 > 0.0 {
            edit1
        } else {
            orig1
        };

        if !(base0.is_finite() && base0 > 0.0) && (base1.is_finite() && base1 > 0.0) {
            base0 = base1;
        }
        if !(base1.is_finite() && base1 > 0.0) && (base0.is_finite() && base0 > 0.0) {
            base1 = base0;
        }
        if !(base0.is_finite() && base0 > 0.0 && base1.is_finite() && base1 > 0.0) {
            return 0.0;
        }

        let v = base0 + (base1 - base0) * frac;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    pub fn is_voiced_at_time(&self, abs_time_sec: f64) -> bool {
        let fp = self.frame_period_ms.max(0.1);
        let idx = ((abs_time_sec.max(0.0) * 1000.0) / fp).round().max(0.0) as usize;
        let orig = self.pitch_orig.get(idx).copied().unwrap_or(0.0);
        let edit = self.pitch_edit.get(idx).copied().unwrap_or(0.0);
        (orig.is_finite() && orig > 0.0) || (edit.is_finite() && edit > 0.0)
    }
}

#[allow(dead_code)]
pub(crate) fn selected_pitch_curves_snapshot<'a>(
    timeline: &'a TimelineState,
) -> Option<PitchCurvesSnapshot<'a>> {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let root = timeline.resolve_root_track_id(&selected)?;

    let entry = timeline.params_by_root_track.get(&root)?;
    Some(PitchCurvesSnapshot {
        frame_period_ms: entry.frame_period_ms.max(0.1),
        pitch_orig: &entry.pitch_orig,
        pitch_edit: &entry.pitch_edit,
    })
}

fn pitch_edit_backend_available_for_algo(algo: PitchEditAlgorithm) -> bool {
    match algo {
        PitchEditAlgorithm::WorldVocoder => crate::world_vocoder::is_available(),
        PitchEditAlgorithm::NsfHifiganOnnx => crate::nsf_hifigan_onnx::is_available(),
        #[cfg(feature = "vslib")]
        PitchEditAlgorithm::VocalShifterVslib => true,
        PitchEditAlgorithm::Bypass => true,
    }
}

fn pitch_edit_backend_available_for_track(track: &crate::state::Track) -> bool {
    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    pitch_edit_backend_available_for_algo(algo)
}

pub(crate) fn extra_param_enabled(extra_params: &HashMap<String, f64>, key: &str) -> bool {
    extra_params.get(key).copied().unwrap_or(0.0) >= 0.5
}

fn curve_differs_from_default_in_range(
    curve: Option<&[f32]>,
    frame_period_ms: f64,
    start_sec: f64,
    end_sec: f64,
    default_value: f32,
) -> bool {
    curve_differs_from_default_in_range_with_tolerance(
        curve,
        frame_period_ms,
        start_sec,
        end_sec,
        default_value,
        1e-3,
    )
}

fn curve_differs_from_default_in_range_with_tolerance(
    curve: Option<&[f32]>,
    frame_period_ms: f64,
    start_sec: f64,
    end_sec: f64,
    default_value: f32,
    tolerance: f32,
) -> bool {
    let Some(curve) = curve else {
        return false;
    };
    if curve.is_empty() {
        return false;
    }

    let fp = frame_period_ms.max(0.1);
    let start_idx = ((start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
    let end_idx = ((end_sec.max(start_sec) * 1000.0) / fp).ceil().max(0.0) as usize;
    let lo = start_idx.min(curve.len());
    let hi = end_idx.min(curve.len());
    curve[lo..hi]
        .iter()
        .any(|value| (value - default_value).abs() >= tolerance)
}

pub(crate) fn hifigan_tension_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a [f32]> {
    clip.extra_curves
        .as_ref()
        .and_then(|curves| curves.get("hifigan_tension"))
        .or_else(|| entry.extra_curves.get("hifigan_tension"))
        .map(|v| v.as_slice())
}

/// 张力曲线在该 clip 区间内是否"活跃"（**决策侧**判据）。
///
/// # 作用
/// 决定是否**预渲染**、是否跳过外部时间拉伸、以及导出能否复用渲染缓存。
/// 它不负责施加张力 —— 施加发生在 `renderer::chain::HiFiGanStage::apply_rd_tension`。
///
/// # 为什么这里要判开关
/// `breath_enabled`（UI 名「气声分离」）是张力的前提：关闭时该曲线已在
/// `ClipProcessContext` 的构造点被剥离，实际不会参与合成。若此处不判，
/// `audio/mixdown.rs` 会因本函数返回 `true` 而放弃缓存复用、做一次**结果正确
/// 但完全白费**的重渲染。
///
/// # 与合成侧的分工
/// - **决策侧**（本函数）：避免无谓的重渲染与拉伸决策偏差。
/// - **合成侧**（`ClipProcessContext` 构造点的曲线剥离）：保证真的不生效。
/// 两侧都需要，且都以 `extra_param_enabled(.., "breath_enabled")` 为同源判据。
///
/// # 参数
/// - `entry`：clip 所属 root track 的参数状态（提供轨道级曲线与帧周期）
/// - `clip`：目标 clip（其 `extra_curves` / `extra_params` 为 clip 级覆盖）
/// - `clip_start_sec`：clip 在时间轴上的起点（秒）
/// 气声分离开关闭时，需要从下发数据中剥离的曲线键。
///
/// # 为什么是这两个
/// - `breath_gain`：噪声支的混入增益。噪声支只存在于 HNSEP 分离路径。
/// - `hifigan_tension`：Rd 张力。它只重塑**谐波**支，同样依赖分离。
///
/// `formant_shift_cents` **不在**其中：共振峰走 mel 阶段的 `keyShift`，
/// 与 HNSEP 无关，关闭分离时照常可用。
///
/// # 命名
/// 键名与 `renderer::chain` 的参数描述符 id 一致；常量在此集中定义，
/// 避免两处字符串字面量漂移。
pub(crate) const HIFIGAN_SEPARATION_GATED_CURVES: [&str; 2] = ["breath_gain", "hifigan_tension"];

/// 按开关状态剥离被门禁的曲线，返回需要**下发**给处理器的曲线集合。
///
/// # 作用
/// 实现需求「开关关闭 ⇒ 气声与张力**不参与合成**」的**合成侧**落地。
///
/// # 流程
/// 1. 开关开启（含曲线本就不存在）⇒ 原样返回借用，**零克隆**（常态路径）。
/// 2. 开关关闭且确有被门禁的键 ⇒ 克隆一次并移除这些键。
///
/// # 为什么必须剥离而不是传零值
/// 曲线**缺失**是本项目既有的"未编辑"语义：`breath_gain` 缺失 ⇒ 默认增益 1.0
/// （噪声原样混回），`hifigan_tension` 缺失 ⇒ 无张力。
/// 而 `breath_gain = 0` 表示"噪声整条静音"——那是**另一种行为**，听感不同。
///
/// # 参数
/// - `curves`：已合并 clip/track 覆盖后的生效曲线
/// - `extra_params`：同源的生效静态参数（用于读取开关）
///
/// # 返回
/// 应当下发的曲线集合。返回值可能借用入参（未剥离时），
/// 也可能借用内部克隆（`Cow` 语义由调用方的局部变量承载）。
/// Compose 关闭时必须剥离的轨道级 HiFiGAN 曲线。
///
/// 需求：**Compose 关闭 ⇒ 听到未经任何改动的原音频**。Compose 影响的是
/// 音高、共振峰、气声、张力这四类"合成"参数；音高由既有判定处理
/// （不下发 pitch_edit / 不触发合成），这里覆盖其余三条曲线。
///
/// # 刻意不包含音量与声相
/// 音量/声相/动态是**混音级**参数，**刻意不受 Compose 限制** ——
/// 未开 Compose 的原始音频轨道同样支持它们（见 `pitch/pitch_clip.rs` 中
/// 关于动态（DYN）电平分析"刻意不检查 compose_enabled"的说明）。
/// 把它们塞进这里会破坏该语义。
pub(crate) const HIFIGAN_COMPOSE_GATED_CURVES: [&str; 3] =
    ["breath_gain", "hifigan_tension", "formant_shift_cents"];

/// 按 **Compose 开关**与**气声分离开关**的状态剥离曲线，返回下发给处理器的集合。
///
/// # 两道门禁
/// 1. **Compose 关闭** ⇒ 剥离 [`HIFIGAN_COMPOSE_GATED_CURVES`]（共振峰/气声/张力）。
///    这是"听到原音频"的**合成侧兜底**：即使处理器因其它原因（音高参考块、
///    子轨共振峰偏移）仍需运行，也读不到这些曲线。
/// 2. **气声分离开关关闭** ⇒ 剥离 [`HIFIGAN_SEPARATION_GATED_CURVES`]（气声/张力）。
///    注意共振峰**不**依赖分离开关，故不在其中。
///
/// # 为什么两道门禁集中在一个函数里
/// `ClipProcessContext` 全仓库只有一处构造，两道门禁都在此处落地，可一次覆盖
/// 所有消费点（`apply_rd_tension` 读张力、`process_breath` 读气声、mel 提取读
/// 共振峰），避免"每个消费点各判一次"造成的口径漂移。
///
/// # 性能
/// 常态（Compose 开启且无需剥离）返回 [`std::borrow::Cow::Borrowed`]，**零克隆**。
pub(crate) fn gate_hifigan_effect_curves<'a>(
    curves: &'a HashMap<String, Vec<f32>>,
    extra_params: &HashMap<String, f64>,
    compose_enabled: bool,
) -> std::borrow::Cow<'a, HashMap<String, Vec<f32>>> {
    let gated: &[&str] = if !compose_enabled {
        &HIFIGAN_COMPOSE_GATED_CURVES
    } else if extra_param_enabled(extra_params, crate::renderer::HIFIGAN_SEPARATION_PARAM_ID) {
        return std::borrow::Cow::Borrowed(curves);
    } else {
        &HIFIGAN_SEPARATION_GATED_CURVES
    };
    if !gated.iter().any(|k| curves.contains_key(*k)) {
        return std::borrow::Cow::Borrowed(curves);
    }
    let mut stripped = curves.clone();
    for key in gated {
        stripped.remove(*key);
    }
    std::borrow::Cow::Owned(stripped)
}

pub(crate) fn hifigan_tension_active_for_clip(
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    // 开关关闭 ⇒ 张力不参与合成，也就不存在"活跃"。
    // clip 级 extra_params 覆盖优先，与其它判定口径一致。
    let extra_params = clip.extra_params.as_ref().unwrap_or(&entry.extra_params);
    if !extra_param_enabled(extra_params, crate::renderer::HIFIGAN_SEPARATION_PARAM_ID) {
        return false;
    }

    let curve = hifigan_tension_curve_for_clip(entry, clip);
    curve_differs_from_default_in_range(
        curve,
        entry.frame_period_ms.max(0.1),
        clip_start_sec,
        clip_start_sec + clip.length_sec.max(0.0),
        0.0,
    )
}

/// 该 clip 是否会走气声（HNSEP 分离 + 独立 noise stem）渲染路径。
///
/// ★ **唯一实现**：预览渲染（`commands::playback::render_single_clip`）与导出侧的
/// **渲染缓存复用门禁**都必须经由它。两处各写一份必然漂移，而漂移的后果分两种，
/// 都不可接受：
/// - 预览侧漏判 → WORLD/vslib 白跑一次 HNSEP 推理（见 `render_single_clip` 的说明）；
/// - 导出侧漏判 → 把"谐波（`breath_gain=0`）+ 独立噪声 stem"的缓存产物当成
///   "链内已按 breath_gain 混好的成品"复用，导出音频静默变化。
///
/// 判据与预览侧逐字一致：渲染器必须是 NSF-HiGAN（切换算法**不会**清空
/// `extra_params`，残留的 `breath_enabled` 不得放行），且 `extra_params.breath_enabled`
/// 为真（clip 级覆盖优先、轨道级兜底）。
pub(crate) fn clip_breath_active(timeline: &TimelineState, clip: &crate::state::Clip) -> bool {
    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return false;
    };
    let breath_capable = timeline
        .tracks
        .iter()
        .find(|track| track.id == clip_root)
        .map(|track| {
            matches!(
                SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo),
                SynthPipelineKind::NsfHifiganOnnx
            )
        })
        .unwrap_or(false);
    if !breath_capable {
        return false;
    }
    let effective_extra_params = clip.extra_params.as_ref().or_else(|| {
        timeline
            .params_by_root_track
            .get(&clip_root)
            .map(|entry| &entry.extra_params)
    });
    effective_extra_params
        .map(|params| extra_param_enabled(params, "breath_enabled"))
        .unwrap_or(false)
}

/// 解析 clip 级覆盖优先、轨道级兜底的 extra_curve。
pub(crate) fn extra_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
    param_id: &str,
) -> Option<&'a [f32]> {
    clip.extra_curves
        .as_ref()
        .and_then(|curves| curves.get(param_id))
        .or_else(|| entry.extra_curves.get(param_id))
        .map(|v| v.as_slice())
}

/// 共通音量曲线（新 key `volume`，兼容旧 `hifigan_volume`）。
pub(crate) fn common_volume_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a [f32]> {
    extra_curve_for_clip(entry, clip, "volume")
        .or_else(|| extra_curve_for_clip(entry, clip, "hifigan_volume"))
}

/// 共通声像曲线。
pub(crate) fn common_pan_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a [f32]> {
    extra_curve_for_clip(entry, clip, "pan")
}

/// 共通动态曲线（DYN，用户绘制的**目标电平**）。
///
/// 曲线里可能出现 `DYN_FOLLOW_ORIG`（−1）哨兵帧，表示「该帧沿用原声电平」。
///
/// ⚠ 这里返回的是**存储形态**（含哨兵）。**音频路径**（实时引擎 / 离线导出）
/// 在装配 clip 时必须先经
/// [`crate::renderer::common_params::resolve_dyn_sentinels_for_audio`] 把哨兵
/// 解析成真实目标电平 —— 逐样本插值穿过负哨兵会掉到静音（咔哒）。
/// 显示路径在 `get_param_frames` 出口自行解析，两者语义一致。
pub(crate) fn common_dyn_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a [f32]> {
    extra_curve_for_clip(entry, clip, crate::renderer::common_params::DYN_PARAM_ID)
}

/// 原声电平基线（`dyn_orig`，轨道级派生数据，无 clip 级覆盖）。
///
/// 由后台响度分析写入 `TrackParamsState.dyn_orig`；分析未就绪时返回 None，
/// 此时动态增益退化为 1.0（等效「沿用原声」）。
pub(crate) fn dyn_orig_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    _clip: &'a crate::state::Clip,
) -> Option<&'a [f32]> {
    if entry.dyn_orig.is_empty() {
        return None;
    }
    Some(entry.dyn_orig.as_slice())
}

/// 该合成链路是否在 processor 内部消费 volume/pan。
///
/// **始终返回 false**：音量/声像/动态是算法无关的混音级参数，一律由音频引擎的
/// mix 阶段应用（实时播放 `audio_engine/mix.rs`、离线导出 `audio/mixdown.rs`）。
/// 任何处理器都不得再应用它们，否则会出现二次增益/声像。
///
/// 保留该函数（而非各处内联 `false`）是为了让「不存在 processor 烘焙」这一事实
/// 有唯一的、可被搜索与断言的落点。
#[allow(dead_code)]
#[inline]
pub(crate) fn processor_bakes_common_mix_curves(_kind: SynthPipelineKind) -> bool {
    false
}

#[cfg(feature = "vslib")]
fn vslib_curve_active_for_clip(
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    // 只有 vslib **专有**的参数需要触发预渲染（它们由 processor 写进控制点）。
    // volume / pan / dyn 是混音级参数，由音频引擎的 mix 阶段实时应用，
    // 改动它们**不得**触发底层重合成 —— 这正是「拖动音量/动态即时生效」的基础，
    // 也是把这三个参数从 vslib 控制点里搬出来要换取的核心收益。
    let defaults: &[(&str, f32)] = &[("formant_shift_cents", 0.0), ("breathiness", 0.0)];
    defaults.iter().any(|&(key, default)| {
        let curve = extra_curve_for_clip(entry, clip, key);
        curve_differs_from_default_in_range(
            curve,
            entry.frame_period_ms.max(0.1),
            clip_start_sec,
            clip_start_sec + clip.length_sec.max(0.0),
            default,
        )
    })
}

fn track_requests_extra_processing(
    algo: PitchEditAlgorithm,
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    // 仅 vslib 分支消费；无 vslib 构建下保留参数以维持统一调用面。
    _clip_start_sec: f64,
) -> bool {
    match algo {
        PitchEditAlgorithm::NsfHifiganOnnx => {
            let extra_params = clip.extra_params.as_ref().unwrap_or(&entry.extra_params);
            extra_param_enabled(extra_params, "breath_enabled")
        }
        #[cfg(feature = "vslib")]
        PitchEditAlgorithm::VocalShifterVslib => {
            vslib_curve_active_for_clip(entry, clip, _clip_start_sec)
        }
        _ => false,
    }
}

pub(crate) fn hifigan_formant_shift_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a [f32]> {
    clip.extra_curves
        .as_ref()
        .and_then(|curves| curves.get("formant_shift_cents"))
        .or_else(|| entry.extra_curves.get("formant_shift_cents"))
        .map(|v| v.as_slice())
}

pub(crate) fn hifigan_formant_shift_active_for_clip(
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    let curve = hifigan_formant_shift_curve_for_clip(entry, clip);
    curve_differs_from_default_in_range_with_tolerance(
        curve,
        entry.frame_period_ms.max(0.1),
        clip_start_sec,
        clip_start_sec + clip.length_sec.max(0.0),
        0.0,
        0.5,
    )
}

/// 轨道级 HiFiGAN 效果（气声 / 张力 / 共振峰）的生效标志。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HifiganEffectFlags {
    breath: bool,
    tension: bool,
    formant: bool,
}

impl HifiganEffectFlags {
    /// 是否有任一轨道级效果生效。
    fn any(self) -> bool {
        self.breath || self.tension || self.formant
    }
}

/// 计算轨道级 HiFiGAN 效果的生效标志，**已应用 Compose 门禁**。
///
/// # 为什么必须集中在这一处
/// 有三个判定点需要它：外部预拉伸决策（[`processor_should_handle_stretch`]）、
/// 逐段渲染决策（`should_process_segment`）、整 clip 预渲染决策
/// （`does_clip_need_processor_render`）。三者口径必须完全一致 —— 历史上它们
/// 各写一份，漂移的后果是"外部不拉伸、内部也不拉伸"（音频被截断/补零，
/// 或回退成源速率变调播放），或者"参数已置灰、用户却仍能听到它"。
///
/// # Compose 门禁（本函数的核心语义）
/// Compose 关闭 ⇒ 一律 `false`。需求是"Compose 关闭时听到**未经任何改动的
/// 原音频**"，因此气声/张力/共振峰既不能参与合成，也**不能触发处理器渲染**；
/// 否则处理器照样运行，用户仍能听到张力等改动。
///
/// 曲线本身的剥离另见 [`gate_hifigan_effect_curves`]（合成侧兜底）：即使处理器
/// 因其它原因（音高参考块、子轨共振峰偏移）仍要运行，也读不到这些曲线。
///
/// # 刻意不包含的项
/// 子轨共振峰偏移（`active_child_formant_offset_config`）**不在此处**：那是
/// 子轨自身的参数，与根轨 Compose 无关，由调用方按需单独叠加。
fn hifigan_effect_flags(
    algo: PitchEditAlgorithm,
    track: &crate::state::Track,
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> HifiganEffectFlags {
    if !matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx) || !track.compose_enabled {
        return HifiganEffectFlags::default();
    }
    HifiganEffectFlags {
        breath: track_requests_extra_processing(algo, entry, clip, clip_start_sec),
        tension: hifigan_tension_active_for_clip(entry, clip, clip_start_sec),
        formant: hifigan_formant_shift_active_for_clip(entry, clip, clip_start_sec),
    }
}

/// 判定当前 clip 是否应由处理器内部消费 Mel Stretch。
///
/// 调用方跳过外部拉伸前必须使用同一个判定；否则气声/张力/共振峰这类
/// 独立效果在 Compose 关闭时会出现“外部不拉伸、内部也不拉伸”的错位。
/// 注意：本判定不探测合成后端的运行时可用性（ONNX 会话加载失败等）——
/// 那类失败由 [`maybe_apply_pitch_edit_to_clip_segment`] 在渲染时回退
/// 外部拉伸兜底，此处保持纯静态判定以保证行为可预测、测试可确定。
pub(crate) fn processor_should_handle_stretch(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
) -> bool {
    let Some(root_id) = timeline.resolve_root_track_id(&clip.track_id) else {
        return false;
    };
    let Some((track, entry)) = root_pitch_edit_state(timeline, &root_id) else {
        return false;
    };
    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return false;
    }
    let fx = hifigan_effect_flags(algo, track, entry, clip, clip.start_sec.max(0.0));
    let child_formant_offset =
        active_child_formant_offset_config(timeline, &clip.track_id).is_some();
    let effect_processing =
        matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx) && (fx.any() || child_formant_offset);
    let rate = (clip.playback_rate as f64).max(1e-6);
    crate::renderer::processor_handles_time_stretch(
        algo.into_kind(),
        track.compose_enabled || entry.has_pitch_adjustment_active || effect_processing,
    ) && ((rate - 1.0).abs() > 1e-6 || effect_processing)
}

impl PitchEditAlgorithm {
    fn into_kind(self) -> SynthPipelineKind {
        match self {
            Self::WorldVocoder => SynthPipelineKind::WorldVocoder,
            Self::NsfHifiganOnnx => SynthPipelineKind::NsfHifiganOnnx,
            #[cfg(feature = "vslib")]
            Self::VocalShifterVslib => SynthPipelineKind::VocalShifterVslib,
            Self::Bypass => SynthPipelineKind::WorldVocoder,
        }
    }

    pub fn from_track_algo(algo: &PitchAnalysisAlgo) -> Self {
        if let Some(v) = pitch_edit_algo_from_env() {
            if matches!(v.as_str(), "nsf_hifigan" | "nsf_hifigan_onnx" | "onnx") {
                return Self::NsfHifiganOnnx;
            }
            if matches!(v.as_str(), "world" | "world_vocoder") {
                // fall through to track algo below
            }
        }
        // 未知算法按工程默认算法处理（见 `PitchAnalysisAlgo::effective`）。
        // 此前它与 `WorldDll` 并在一起走 WORLD：工程里存着一个本构建不认识的
        // 算法名时，引擎会静默换成 WORLD 声码器，而界面显示的是 nsf-hifigan。
        match algo.effective() {
            // `effective()` 已把 `Unknown` 归一为默认算法；此处列出只为穷举。
            PitchAnalysisAlgo::NsfHifiganOnnx | PitchAnalysisAlgo::Unknown => Self::NsfHifiganOnnx,
            PitchAnalysisAlgo::WorldDll => Self::WorldVocoder,
            #[cfg(feature = "vslib")]
            PitchAnalysisAlgo::VocalShifterVslib => Self::VocalShifterVslib,
            #[cfg(not(feature = "vslib"))]
            PitchAnalysisAlgo::VocalShifterVslib => Self::Bypass,
            PitchAnalysisAlgo::None => Self::Bypass,
        }
    }
}

#[allow(dead_code)]
pub fn selected_pitch_edit_algorithm(timeline: &TimelineState) -> PitchEditAlgorithm {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return PitchEditAlgorithm::Bypass;
    };

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return PitchEditAlgorithm::Bypass;
    };

    PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo)
}

#[cfg(test)]
mod tests {
    // 这两项本身与 `vslib` feature 无关：`processor_bakes_common_mix_curves`
    // 是无条件定义的，`SynthPipelineKind` 也只是**变体**受 feature 门控。
    // 此前它们被错误地门控在 `feature = "vslib"` 下，于是关闭 vslib 的构建
    // 整个测试模块编译不过（`cargo test --no-default-features` 报
    // "cannot find function / type"）—— 而那正是没有 vslib DLL 时的构建方式。
    use super::processor_bakes_common_mix_curves;
    use super::{
        active_child_formant_offset_config, build_clip_effective_formant_shift_curve,
        child_formant_offset_curve_key, common_pan_curve_for_clip, common_volume_curve_for_clip,
        does_clip_need_processor_render, extra_curve_for_clip, gate_hifigan_effect_curves,
        hifigan_formant_shift_active_for_clip, maybe_apply_pitch_edit_to_clip_segment,
        processor_should_handle_stretch, PitchEditAlgorithm,
    };
    use crate::state::SynthPipelineKind;
    use crate::state::{Clip, PitchAnalysisAlgo, TimelineState, TrackParamsState};
    use std::collections::HashMap;

    /// 未知算法在 pitch edit 链路里同样按工程默认算法处理。
    ///
    /// 回归对象：`Unknown` 曾与 `WorldDll` 并在一起走 WORLD，于是工程里存着
    /// 本构建不认识的算法名时，引擎静默换成 WORLD 声码器 —— 而界面显示的是
    /// nsf-hifigan。两处（渲染链路与 pitch edit 链路）必须给出同一个答案。
    #[test]
    fn unknown_track_algo_edits_with_the_default_algorithm() {
        assert_eq!(
            PitchEditAlgorithm::from_track_algo(&PitchAnalysisAlgo::Unknown),
            PitchEditAlgorithm::from_track_algo(&PitchAnalysisAlgo::default()),
            "未知算法与默认算法必须走同一条 pitch edit 链路"
        );
        assert_ne!(
            PitchEditAlgorithm::from_track_algo(&PitchAnalysisAlgo::Unknown),
            PitchEditAlgorithm::WorldVocoder,
            "未知算法不得落到 WORLD"
        );
    }

    fn make_clip() -> Clip {
        Clip {
            id: "clip-a".to_string(),
            takes: vec![],
            active_take_id: None,
            clip_playback_rate: 1.0,
            track_id: "track-a".to_string(),
            name: "Clip".to_string(),
            start_sec: 0.0,
            length_sec: 2.0,
            color: "blue".to_string(),
            source_path: Some("a.wav".to_string()),
            source_path_relative: None,
            duration_sec: Some(2.0),
            duration_frames: None,
            source_sample_rate: Some(44_100),
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            reversed: false,
            channel_mode: 0,
            source_channels: None,
            loop_enabled: false,
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
            fade_in_shape: 0.0,
            fade_out_shape: 0.0,
            fade_in_dir: 0.0,
            fade_out_dir: 0.0,
            auto_fade_in_sec: 0.0,
            auto_fade_out_sec: 0.0,
            extra_curves: None,
            extra_params: None,
            formant_morph: None,
            group_id: None,
            midi_fill_gaps: false,
            midi_note_data: None,
        }
    }

    #[test]
    fn child_formant_offset_accumulates_along_child_track() {
        let mut timeline = TimelineState::default();
        let root = timeline.add_track(Some("root".to_string()), None, None);
        let child = timeline.add_track(Some("child".to_string()), Some(root.clone()), None);

        let mut entry = TrackParamsState {
            frame_period_ms: 5.0,
            ..Default::default()
        };
        entry.extra_curves = HashMap::from([
            (
                child_formant_offset_curve_key(&child),
                vec![100.0, 250.0, -50.0],
            ),
            ("formant_shift_cents".to_string(), vec![50.0, 0.0, 0.0]),
        ]);
        timeline.params_by_root_track.insert(root, entry);

        let mut clip = make_clip();
        clip.track_id = child;

        let cfg = active_child_formant_offset_config(&timeline, &clip.track_id)
            .expect("child formant offset must be active");
        assert_eq!(cfg.layers.len(), 1);

        let effective = build_clip_effective_formant_shift_curve(
            &timeline,
            &clip,
            timeline
                .params_by_root_track
                .get(&clip_track_root(&timeline, &clip).unwrap())
                .unwrap(),
            3,
        )
        .expect("effective formant curve");
        assert_eq!(effective, vec![150.0, 250.0, -50.0]);
    }

    fn clip_track_root(timeline: &TimelineState, clip: &Clip) -> Option<String> {
        timeline.resolve_root_track_id(&clip.track_id)
    }

    /// 需求：**Compose 关闭 ⇒ 听到未经任何改动的原音频**。
    ///
    /// 因此即使气声开关为开、playback_rate ≠ 1，处理器也**不得**声明自己处理
    /// 时间拉伸 —— 一旦声明，调用方就会跳过外部拉伸，转而由处理器重新合成
    /// （用户于是仍能听到气声/张力等改动）。拉伸交由**外部**算法完成。
    #[test]
    fn compose_off_effects_do_not_claim_time_stretch() {
        crate::time_stretch::update_runtime_stretch_settings(
            crate::time_stretch::UserStretchAlgorithm::Signalsmith,
            true,
            None,
            None,
        );
        let mut timeline = TimelineState::default();
        let root = timeline.add_track(Some("root".to_string()), None, None);
        timeline.set_track_state(
            &root,
            None,
            None,
            None,
            Some(false),
            Some(crate::state::PitchAnalysisAlgo::NsfHifiganOnnx),
            None,
            None,
        );
        timeline.params_by_root_track.insert(
            root.clone(),
            TrackParamsState {
                frame_period_ms: 5.0,
                ..Default::default()
            },
        );

        let mut clip = make_clip();
        clip.track_id = root;
        clip.playback_rate = 0.5;
        clip.extra_params = Some(HashMap::from([("breath_enabled".to_string(), 1.0)]));

        assert!(
            !processor_should_handle_stretch(&timeline, &clip),
            "Compose 关闭时处理器不得声明处理时间拉伸（否则会重新合成，效果被听到）"
        );
    }

    fn breath_only_timeline() -> (TimelineState, Clip) {
        // 构造"Compose 关闭 + 气声开启 + Mel Stretch + rate=0.5"的场景。
        // 注意：新语义下这组设置**不应**触发任何处理器渲染或内部拉伸 ——
        // 气声属于"合成"参数，Compose 关闭时必须完全静默。
        crate::time_stretch::update_runtime_stretch_settings(
            crate::time_stretch::UserStretchAlgorithm::Signalsmith,
            true,
            None,
            None,
        );
        let mut timeline = TimelineState::default();
        let root = timeline.add_track(Some("root".to_string()), None, None);
        timeline.set_track_state(
            &root,
            None,
            None,
            None,
            Some(false), // compose_enabled = false
            Some(crate::state::PitchAnalysisAlgo::NsfHifiganOnnx),
            None,
            None,
        );
        timeline.params_by_root_track.insert(
            root.clone(),
            TrackParamsState {
                frame_period_ms: 5.0,
                ..Default::default()
            },
        );

        let mut clip = make_clip();
        clip.track_id = root;
        clip.playback_rate = 0.5;
        clip.extra_params = Some(HashMap::from([("breath_enabled".to_string(), 1.0)]));
        (timeline, clip)
    }

    /// Compose 关闭时，轨道级效果**不得**触发预渲染。
    ///
    /// 两个判定（外部拉伸决策 / 整 clip 预渲染决策）必须同源：这里断言二者
    /// 一致为 false。若预渲染放行而拉伸判定不放行，实时引擎会拿到一份
    /// 处理器重新合成过的 PCM —— 等于 Compose 关了却仍听得到效果。
    #[test]
    fn compose_off_effects_require_no_processor_render() {
        let (timeline, clip) = breath_only_timeline();
        assert!(
            !processor_should_handle_stretch(&timeline, &clip),
            "Compose 关闭 ⇒ 不得由效果声明内部拉伸"
        );
        assert!(
            !does_clip_need_processor_render(&timeline, &clip, 0.0),
            "Compose 关闭 ⇒ 不得由效果触发预渲染"
        );
    }

    /// Compose 关闭时，处理器**不得改写** PCM。
    ///
    /// 【契约变更说明】本测试此前断言"输出长度必须等于时间轴帧数"，因为当时
    /// 效果会在 Compose 关闭时照样进入处理器并接管内部拉伸。新需求是
    /// "Compose 关闭 ⇒ 未经任何改动的原音频"，于是拉伸责任回到**调用方**
    /// （外部预拉伸，由 `processor_should_handle_stretch == false` 决定）。
    /// 因此这里断言 PCM 逐样本不变 —— 调用方拿到的必须是原样数据。
    /// 若处理器仍改写 PCM，用户就会听到被重新合成过的音频。
    #[test]
    fn compose_off_leaves_pcm_untouched() {
        let (timeline, clip) = breath_only_timeline();

        // 0.1s @ 44.1k 立体声（短输入让可能的 ONNX 推理保持快速）。
        let frames_in = 4_410usize;
        let original = vec![0.25f32; frames_in * 2];
        let mut pcm = original.clone();
        let applied =
            maybe_apply_pitch_edit_to_clip_segment(&timeline, &clip, 0.0, 0.0, 44_100, &mut pcm)
                .expect("maybe_apply must not error");

        assert!(
            !applied,
            "Compose 关闭 ⇒ 处理器不得被应用（applied 应为 false）"
        );
        assert_eq!(
            pcm.len(),
            original.len(),
            "Compose 关闭 ⇒ PCM 长度不得被改写（拉伸由调用方负责）"
        );
        assert_eq!(
            pcm, original,
            "Compose 关闭 ⇒ PCM 必须逐样本保持原样，用户听到的是原音频"
        );
    }

    #[test]
    fn mel_stretch_with_pending_analysis_keeps_timeline_length() {
        // 回归：Compose 开启 + Mel Stretch + rate≠1，但音高分析尚未就绪
        //（pitch 曲线缓存为空）。外部拉伸已被跳过，此前 maybe_apply 在
        // build_clip_input_pitch_curve 返回 None 时直接放弃 → 音频保持源
        // 速率被截断/补零。修复后纯拉伸场景传空曲线让处理器内部回退外部
        // 算法拉伸（后端不可用时在 maybe_apply 兜底），长度必须正确。
        crate::time_stretch::update_runtime_stretch_settings(
            crate::time_stretch::UserStretchAlgorithm::Signalsmith,
            true,
            None,
            None,
        );
        let mut timeline = TimelineState::default();
        let root = timeline.add_track(Some("root".to_string()), None, None);
        timeline.set_track_state(
            &root,
            None,
            None,
            None,
            Some(true), // compose_enabled = true
            Some(crate::state::PitchAnalysisAlgo::NsfHifiganOnnx),
            None,
            None,
        );
        timeline.params_by_root_track.insert(
            root.clone(),
            TrackParamsState {
                frame_period_ms: 5.0,
                ..Default::default()
            },
        );

        let mut clip = make_clip();
        clip.track_id = root;
        clip.playback_rate = 0.5;
        // 无气声、无音高编辑 —— 纯 Mel Stretch 场景。

        let frames_in = 4_410usize;
        let mut pcm = vec![0.25f32; frames_in * 2];
        let applied =
            maybe_apply_pitch_edit_to_clip_segment(&timeline, &clip, 0.0, 0.0, 44_100, &mut pcm)
                .expect("maybe_apply must not error");

        let expected_frames = frames_in * 2;
        assert_eq!(
            pcm.len(),
            expected_frames * 2,
            "mel-stretch clip with pending analysis must keep timeline length (applied={applied})"
        );
        assert!(pcm.iter().any(|&v| v.abs() > 1e-4));
    }

    #[test]
    fn hifigan_formant_shift_ignores_near_zero_residual_values() {
        let mut entry = TrackParamsState {
            frame_period_ms: 5.0,
            ..Default::default()
        };
        entry.extra_curves = HashMap::from([("formant_shift_cents".to_string(), vec![0.1; 500])]);

        assert!(!hifigan_formant_shift_active_for_clip(
            &entry,
            &make_clip(),
            0.0
        ));
    }

    #[test]
    fn hifigan_formant_shift_detects_meaningful_offsets() {
        let mut entry = TrackParamsState {
            frame_period_ms: 5.0,
            ..Default::default()
        };
        entry.extra_curves = HashMap::from([("formant_shift_cents".to_string(), vec![1.0; 500])]);

        assert!(hifigan_formant_shift_active_for_clip(
            &entry,
            &make_clip(),
            0.0
        ));
    }

    #[test]
    fn common_volume_curve_reads_legacy_hifigan_volume_key() {
        let mut entry = TrackParamsState::default();
        entry.extra_curves = HashMap::from([("hifigan_volume".to_string(), vec![0.5, 0.75, 1.0])]);
        let clip = make_clip();
        assert_eq!(
            common_volume_curve_for_clip(&entry, &clip).map(|v| v.to_vec()),
            Some(vec![0.5, 0.75, 1.0])
        );
        assert_eq!(common_pan_curve_for_clip(&entry, &clip), None);
    }

    #[test]
    fn clip_level_common_curves_take_precedence_over_track_curves() {
        let mut entry = TrackParamsState::default();
        entry.extra_curves = HashMap::from([
            ("volume".to_string(), vec![1.0, 1.0]),
            ("pan".to_string(), vec![0.0, 0.0]),
        ]);
        let mut clip = make_clip();
        clip.extra_curves = Some(HashMap::from([("pan".to_string(), vec![0.5, -0.5])]));

        // clip 覆盖 pan；clip 没有覆盖 volume，回退到 track 级曲线。
        assert_eq!(
            extra_curve_for_clip(&entry, &clip, "pan").map(|v| v.to_vec()),
            Some(vec![0.5, -0.5])
        );
        assert_eq!(
            common_volume_curve_for_clip(&entry, &clip).map(|v| v.to_vec()),
            Some(vec![1.0, 1.0])
        );
        assert_eq!(
            common_pan_curve_for_clip(&entry, &clip).map(|v| v.to_vec()),
            Some(vec![0.5, -0.5])
        );
    }

    #[test]
    fn no_algorithm_bakes_common_mix_curves_into_processor_output() {
        // 音量/声像/动态一律由混音层应用；任何算法都不得在处理器内部再应用一次，
        // 否则会出现二次增益/声像（vslib 曾如此，已移除）。
        assert!(!processor_bakes_common_mix_curves(
            SynthPipelineKind::NsfHifiganOnnx
        ));
        assert!(!processor_bakes_common_mix_curves(
            SynthPipelineKind::WorldVocoder
        ));
        #[cfg(feature = "vslib")]
        assert!(!processor_bakes_common_mix_curves(
            SynthPipelineKind::VocalShifterVslib
        ));
    }

    #[test]
    fn vslib_exposes_no_common_mix_curves_as_processor_params() {
        // 回归护栏：共通参数只能由 `renderer::all_param_descriptors` 提供。
        // 若某个处理器的 descriptor 列表里又出现 volume/pan/dyn，
        // 说明有人把"混音级参数"重新塞回了算法内部（正是本次改造要消除的耦合）。
        for kind in [
            SynthPipelineKind::WorldVocoder,
            SynthPipelineKind::NsfHifiganOnnx,
        ] {
            let own = crate::renderer::get_processor(kind).param_descriptors();
            for d in own {
                assert!(
                    !crate::renderer::common_params::is_common_mix_param(d.id),
                    "{:?} 不应再声明共通参数 {}",
                    kind,
                    d.id
                );
            }
        }
    }

    // ── 气声分离开关的曲线门禁（需求：关闭时气声与张力不参与合成）──────────

    fn params_with_separation(on: bool) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("breath_enabled".to_string(), if on { 1.0 } else { 0.0 });
        m
    }

    /// **Compose 关闭 ⇒ 剥离全部轨道级 HiFiGAN 效果曲线。**
    ///
    /// 需求：Compose 关闭时听到的是**未经任何改动的原音频**，因此共振峰、
    /// 气声、张力三条曲线都必须收不到 —— 这是"用户仍能听到张力改动"
    /// 这一缺陷的合成侧断言。
    #[test]
    fn compose_off_strips_all_hifigan_effect_curves() {
        let mut curves = HashMap::new();
        curves.insert("breath_gain".to_string(), vec![0.5f32, 0.8]);
        curves.insert("hifigan_tension".to_string(), vec![80.0f32, -40.0]);
        curves.insert("formant_shift_cents".to_string(), vec![120.0f32]);

        // 即使分离开关是**开**的，Compose 关闭也必须全部剥离。
        let gated = gate_hifigan_effect_curves(&curves, &params_with_separation(true), false);

        for key in ["breath_gain", "hifigan_tension", "formant_shift_cents"] {
            assert!(
                !gated.contains_key(key),
                "Compose 关闭必须剥离 {key}，否则用户仍能听到该改动"
            );
        }
    }

    /// **Compose 关闭不得剥离音量 / 声相这类混音级参数。**
    ///
    /// 需求口径：Compose 只影响**音高、共振峰、气声、张力**这四类"合成"参数；
    /// 音量与声相是**混音级**参数，未开 Compose 的原始音频轨道同样要支持
    /// （项目里对动态(DYN)已有同样说明："刻意不检查 compose_enabled"）。
    /// 若把它们一并剥离，关掉 Compose 就会连音量/声相一起失效。
    #[test]
    fn compose_off_keeps_mix_level_curves() {
        let mut curves = HashMap::new();
        curves.insert("hifigan_tension".to_string(), vec![80.0f32]);
        curves.insert("volume".to_string(), vec![0.5f32, 1.5]);
        curves.insert("pan".to_string(), vec![-1.0f32, 0.25]);
        curves.insert("dyn".to_string(), vec![0.3f32]);

        let gated = gate_hifigan_effect_curves(&curves, &HashMap::new(), false);

        assert!(!gated.contains_key("hifigan_tension"), "合成参数应被剥离");
        for key in ["volume", "pan", "dyn"] {
            assert!(
                gated.contains_key(key),
                "混音级参数 {key} 不受 Compose 门禁，必须保留"
            );
        }
    }

    /// 关闭开关必须剥掉 `breath_gain` 与 `hifigan_tension`。
    ///
    /// 这是需求「不参与合成」的**合成侧**核心断言：处理器收不到这两条曲线，
    /// 无论其内部逻辑如何都不会生效，分离路径也不会被触发。
    #[test]
    fn separation_off_strips_both_gated_curves() {
        let mut curves = HashMap::new();
        curves.insert("breath_gain".to_string(), vec![0.5f32, 0.8]);
        curves.insert("hifigan_tension".to_string(), vec![80.0f32, -40.0]);
        // 对照项：共振峰与分离无关，必须保留
        curves.insert("formant_shift_cents".to_string(), vec![120.0f32]);

        let gated = gate_hifigan_effect_curves(&curves, &params_with_separation(false), true);

        assert!(
            !gated.contains_key("breath_gain"),
            "switch off must strip breath_gain so the noise gain cannot apply"
        );
        assert!(
            !gated.contains_key("hifigan_tension"),
            "switch off must strip hifigan_tension so Rd tension cannot apply"
        );
        assert_eq!(
            gated.get("formant_shift_cents").map(|v| v.as_slice()),
            Some([120.0f32].as_slice()),
            "formant shift goes through the mel stage and must NOT be gated"
        );
    }

    /// 开启开关必须**原样保留**两条曲线（不得误剥）。
    #[test]
    fn separation_on_keeps_both_curves() {
        let mut curves = HashMap::new();
        curves.insert("breath_gain".to_string(), vec![0.5f32]);
        curves.insert("hifigan_tension".to_string(), vec![80.0f32]);

        let gated = gate_hifigan_effect_curves(&curves, &params_with_separation(true), true);

        assert_eq!(
            gated.get("breath_gain").map(|v| v.as_slice()),
            Some([0.5f32].as_slice())
        );
        assert_eq!(
            gated.get("hifigan_tension").map(|v| v.as_slice()),
            Some([80.0f32].as_slice())
        );
    }

    /// 剥离必须是"键缺失"，**不能**退化成"传零值"。
    ///
    /// `breath_gain = 0` 表示"噪声整条静音"，而缺失表示"未编辑 ⇒ 默认增益 1.0
    /// （原样混回）"。两者听感不同，混淆会让关闭开关时齿音/气声消失。
    #[test]
    fn stripping_removes_the_key_rather_than_zeroing_it() {
        let mut curves = HashMap::new();
        curves.insert("breath_gain".to_string(), vec![0.5f32]);

        let gated = gate_hifigan_effect_curves(&curves, &params_with_separation(false), true);

        assert!(
            gated.get("breath_gain").is_none(),
            "must remove the key, not replace the curve with zeros"
        );
    }

    /// 常态路径（开关开启、或没有相关曲线）必须**借用**、零克隆。
    ///
    /// 关闭开关是少数情况；若为了它让每次渲染都克隆整张曲线表，
    /// 就为了一个边界情况牺牲了主路径。
    #[test]
    fn common_path_borrows_without_cloning() {
        use std::borrow::Cow;

        // 开关开启
        let mut curves = HashMap::new();
        curves.insert("breath_gain".to_string(), vec![0.5f32]);
        assert!(matches!(
            gate_hifigan_effect_curves(&curves, &params_with_separation(true), true),
            Cow::Borrowed(_)
        ));

        // 开关关闭但没有被门禁的键（无需克隆）
        let empty: HashMap<String, Vec<f32>> = HashMap::new();
        assert!(matches!(
            gate_hifigan_effect_curves(&empty, &params_with_separation(false), true),
            Cow::Borrowed(_)
        ));

        // 开关关闭且确有被门禁的键 ⇒ 此时才克隆
        assert!(matches!(
            gate_hifigan_effect_curves(&curves, &params_with_separation(false), true),
            Cow::Owned(_)
        ));
    }

    /// 开关**缺失**等同关闭（`extra_param_enabled` 的既有语义）。
    #[test]
    fn missing_switch_behaves_as_off() {
        let mut curves = HashMap::new();
        curves.insert("hifigan_tension".to_string(), vec![80.0f32]);
        let gated = gate_hifigan_effect_curves(&curves, &HashMap::new(), true);
        assert!(
            !gated.contains_key("hifigan_tension"),
            "a missing switch key means off, so curves must be stripped"
        );
    }
}

fn semitone_ratio(semitones: f64) -> f64 {
    (2.0f64).powf(semitones / 12.0)
}

#[derive(Debug, Clone, Copy)]
enum ChildPitchOffsetParamMode {
    Cents,
    Degrees,
}

#[derive(Debug, Clone, Copy)]
struct ChildPitchOffsetLayer<'a> {
    cents: f64,
    degree_steps: f64,
    cents_curve: Option<&'a [f32]>,
    degree_steps_curve: Option<&'a [f32]>,
}

#[derive(Debug, Clone)]
struct ChildPitchOffsetConfig<'a> {
    layers: Vec<ChildPitchOffsetLayer<'a>>,
}

const CHILD_PITCH_OFFSET_CENTS_PREFIX: &str = "child_pitch_offset_cents@";
const CHILD_PITCH_OFFSET_DEGREES_PREFIX: &str = "child_pitch_offset_degrees@";
const CHILD_PITCH_OFFSET_CENTS_DEFAULT: f64 = 0.0;
const CHILD_PITCH_OFFSET_DEGREES_DEFAULT: f64 = 0.0;

fn child_pitch_offset_curve_key(mode: ChildPitchOffsetParamMode, track_id: &str) -> String {
    match mode {
        ChildPitchOffsetParamMode::Cents => {
            format!("{CHILD_PITCH_OFFSET_CENTS_PREFIX}{track_id}")
        }
        ChildPitchOffsetParamMode::Degrees => {
            format!("{CHILD_PITCH_OFFSET_DEGREES_PREFIX}{track_id}")
        }
    }
}

pub(crate) fn ordered_scale_semitone_offsets(scale_notes: &[u8]) -> Vec<i32> {
    if scale_notes.is_empty() {
        return vec![0, 2, 4, 5, 7, 9, 11];
    }
    let mut normalized: Vec<i32> = scale_notes.iter().map(|v| (v % 12) as i32).collect();
    normalized.sort_unstable();
    normalized.dedup();
    if normalized.is_empty() {
        return vec![0, 2, 4, 5, 7, 9, 11];
    }

    let mut out = Vec::with_capacity(normalized.len());
    let mut prev = i32::MIN;
    for mut value in normalized {
        while value <= prev {
            value += 12;
        }
        out.push(value);
        prev = value;
    }
    out
}

pub(crate) fn scale_degree_to_midi_integer(abs_degree: i32, offsets: &[i32]) -> f64 {
    let degree_count = offsets.len() as i32;
    if degree_count <= 0 {
        return 0.0;
    }
    let oct = abs_degree.div_euclid(degree_count);
    let idx = abs_degree.rem_euclid(degree_count) as usize;
    (oct * 12 + offsets[idx]) as f64
}

pub(crate) fn scale_degree_to_midi(abs_degree: f64, offsets: &[i32]) -> f64 {
    if !abs_degree.is_finite() {
        return 0.0;
    }
    let lower_degree = abs_degree.floor() as i32;
    let frac = abs_degree - lower_degree as f64;
    let lower = scale_degree_to_midi_integer(lower_degree, offsets);
    if frac <= 1e-9 {
        return lower;
    }
    let upper = scale_degree_to_midi_integer(lower_degree + 1, offsets);
    lower + (upper - lower) * frac
}

pub(crate) fn transpose_midi_by_scale_steps(
    midi: f64,
    degree_steps: f64,
    scale_notes: &[u8],
) -> f64 {
    if !midi.is_finite() || degree_steps.abs() <= 1e-9 {
        return midi;
    }
    let offsets = ordered_scale_semitone_offsets(scale_notes);
    if offsets.is_empty() {
        return midi;
    }

    let degree_count = offsets.len() as i32;
    let base_oct = (midi / 12.0).floor() as i32;

    let mut lower: Option<(i32, f64)> = None;
    let mut upper: Option<(i32, f64)> = None;
    for oct in (base_oct - 3)..=(base_oct + 3) {
        for (idx, offset) in offsets.iter().enumerate() {
            let candidate_midi = (oct * 12 + *offset) as f64;
            let abs_degree = oct * degree_count + idx as i32;
            if candidate_midi <= midi && lower.map(|(_, v)| candidate_midi > v).unwrap_or(true) {
                lower = Some((abs_degree, candidate_midi));
            }
            if candidate_midi >= midi && upper.map(|(_, v)| candidate_midi < v).unwrap_or(true) {
                upper = Some((abs_degree, candidate_midi));
            }
        }
    }

    let (lower_degree, lower_midi) = lower.unwrap_or((0, midi));
    let (upper_degree, upper_midi) = upper.unwrap_or((lower_degree, lower_midi));
    let span = upper_midi - lower_midi;
    let ratio = if span.abs() <= 1e-9 {
        0.0
    } else {
        ((midi - lower_midi) / span).clamp(0.0, 1.0)
    };

    let target_lower = scale_degree_to_midi(lower_degree as f64 + degree_steps, &offsets);
    let target_upper = scale_degree_to_midi(upper_degree as f64 + degree_steps, &offsets);
    target_lower + (target_upper - target_lower) * ratio
}

fn active_child_pitch_offset_config<'a>(
    timeline: &'a TimelineState,
    clip_track_id: &str,
) -> Option<ChildPitchOffsetConfig<'a>> {
    let track = timeline
        .tracks
        .iter()
        .find(|track| track.id == clip_track_id)?;
    track.parent_id.as_ref()?;

    let root_track_id = timeline.resolve_root_track_id(clip_track_id)?;
    let entry = timeline.params_by_root_track.get(&root_track_id);

    let mut lineage_child_ids: Vec<&str> = Vec::new();
    let mut cursor = Some(clip_track_id);
    let mut safety = 0usize;
    while let Some(track_id) = cursor {
        let Some(node) = timeline.tracks.iter().find(|track| track.id == track_id) else {
            break;
        };
        if node.parent_id.is_none() {
            break;
        }
        lineage_child_ids.push(track_id);
        cursor = node.parent_id.as_deref();
        safety += 1;
        if safety > timeline.tracks.len() + 2 {
            break;
        }
    }

    if lineage_child_ids.is_empty() {
        return None;
    }

    lineage_child_ids.reverse();

    let frame_period_ms = entry.map(|state| state.frame_period_ms).unwrap_or(5.0);
    let mut has_effective = false;
    let mut layers: Vec<ChildPitchOffsetLayer<'a>> = Vec::with_capacity(lineage_child_ids.len());

    for track_id in lineage_child_ids {
        let cents_curve = entry
            .and_then(|state| {
                state.extra_curves.get(&child_pitch_offset_curve_key(
                    ChildPitchOffsetParamMode::Cents,
                    track_id,
                ))
            })
            .map(|v| v.as_slice());
        let degree_steps_curve = entry
            .and_then(|state| {
                state.extra_curves.get(&child_pitch_offset_curve_key(
                    ChildPitchOffsetParamMode::Degrees,
                    track_id,
                ))
            })
            .map(|v| v.as_slice());

        let static_cents = CHILD_PITCH_OFFSET_CENTS_DEFAULT;
        let static_degree_steps = CHILD_PITCH_OFFSET_DEGREES_DEFAULT;

        let has_cents_curve = curve_differs_from_default_in_range(
            cents_curve,
            frame_period_ms,
            0.0,
            f64::MAX,
            static_cents as f32,
        );
        let has_degree_curve = curve_differs_from_default_in_range(
            degree_steps_curve,
            frame_period_ms,
            0.0,
            f64::MAX,
            static_degree_steps as f32,
        );
        has_effective = has_effective || has_cents_curve || has_degree_curve;

        layers.push(ChildPitchOffsetLayer {
            cents: static_cents,
            degree_steps: static_degree_steps,
            cents_curve,
            degree_steps_curve,
        });
    }

    if !has_effective {
        return None;
    }

    Some(ChildPitchOffsetConfig { layers })
}

const CHILD_FORMANT_OFFSET_CENTS_PREFIX: &str = "child_formant_offset_cents@";
const CHILD_FORMANT_OFFSET_CENTS_DEFAULT: f32 = 0.0;
const CHILD_FORMANT_OFFSET_CENTS_RANGE: (f32, f32) = (-2400.0, 2400.0);

#[derive(Debug, Clone, Copy)]
struct ChildFormantOffsetLayer<'a> {
    curve: Option<&'a [f32]>,
}

#[derive(Debug, Clone)]
struct ChildFormantOffsetConfig<'a> {
    layers: Vec<ChildFormantOffsetLayer<'a>>,
}

fn child_formant_offset_curve_key(track_id: &str) -> String {
    format!("{CHILD_FORMANT_OFFSET_CENTS_PREFIX}{track_id}")
}

fn active_child_formant_offset_config<'a>(
    timeline: &'a TimelineState,
    clip_track_id: &str,
) -> Option<ChildFormantOffsetConfig<'a>> {
    let track = timeline
        .tracks
        .iter()
        .find(|track| track.id == clip_track_id)?;
    track.parent_id.as_ref()?;

    let root_track_id = timeline.resolve_root_track_id(clip_track_id)?;
    // 只有支持逐帧 formant_shift_cents 的声码器链路才消费子轨共振峰差。
    let root_kind = timeline
        .tracks
        .iter()
        .find(|track| track.id == root_track_id)
        .map(|track| SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(SynthPipelineKind::WorldVocoder);
    let supports_formant = matches!(root_kind, SynthPipelineKind::NsfHifiganOnnx) || {
        #[cfg(feature = "vslib")]
        {
            matches!(root_kind, SynthPipelineKind::VocalShifterVslib)
        }
        #[cfg(not(feature = "vslib"))]
        {
            false
        }
    };
    if !supports_formant {
        return None;
    }

    let entry = timeline.params_by_root_track.get(&root_track_id);

    // 从当前子轨向上收集父轨层级（应用顺序与音分差/度数差一致：根 → 父 → 子）。
    let mut lineage_child_ids: Vec<&str> = Vec::new();
    let mut cursor = Some(clip_track_id);
    let mut safety = 0usize;
    while let Some(track_id) = cursor {
        let Some(node) = timeline.tracks.iter().find(|track| track.id == track_id) else {
            break;
        };
        if node.parent_id.is_none() {
            break;
        }
        lineage_child_ids.push(track_id);
        cursor = node.parent_id.as_deref();
        safety += 1;
        if safety > timeline.tracks.len() + 2 {
            break;
        }
    }
    if lineage_child_ids.is_empty() {
        return None;
    }
    lineage_child_ids.reverse();

    let frame_period_ms = entry.map(|state| state.frame_period_ms).unwrap_or(5.0);
    let mut has_effective = false;
    let mut layers: Vec<ChildFormantOffsetLayer<'a>> = Vec::with_capacity(lineage_child_ids.len());

    for track_id in lineage_child_ids {
        let curve = entry
            .and_then(|state| {
                state
                    .extra_curves
                    .get(&child_formant_offset_curve_key(track_id))
            })
            .map(|v| v.as_slice());

        has_effective = has_effective
            || curve_differs_from_default_in_range_with_tolerance(
                curve,
                frame_period_ms,
                0.0,
                f64::MAX,
                CHILD_FORMANT_OFFSET_CENTS_DEFAULT,
                0.5,
            );
        layers.push(ChildFormantOffsetLayer { curve });
    }

    if !has_effective {
        return None;
    }
    Some(ChildFormantOffsetConfig { layers })
}

fn sample_child_formant_offset_cents(layer: &ChildFormantOffsetLayer<'_>, frame_idx: usize) -> f32 {
    layer
        .curve
        .and_then(|curve| curve.get(frame_idx).copied())
        .filter(|value| value.is_finite())
        .unwrap_or(CHILD_FORMANT_OFFSET_CENTS_DEFAULT)
}

/// 构建 clip 实际生效的共振峰偏移曲线：
/// 根轨 / clip 覆盖的 `formant_shift_cents` + 该子轨沿父轨层级累加的
/// `child_formant_offset_cents@...`。无子轨共振峰差时返回 None，调用方继续
/// 使用原始 `extra_curves`，避免任何额外分配。
pub(crate) fn build_clip_effective_formant_shift_curve(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    entry: &crate::state::TrackParamsState,
    frame_count: usize,
) -> Option<Vec<f32>> {
    let cfg = active_child_formant_offset_config(timeline, &clip.track_id)?;
    let base_curve = extra_curve_for_clip(entry, clip, "formant_shift_cents");
    let mut out = vec![0.0f32; frame_count.max(1)];

    for (frame_idx, value) in out.iter_mut().enumerate() {
        let base = base_curve
            .and_then(|curve| curve.get(frame_idx).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        let mut total = base as f64;
        for layer in &cfg.layers {
            total += sample_child_formant_offset_cents(layer, frame_idx) as f64;
        }
        *value = total.clamp(
            CHILD_FORMANT_OFFSET_CENTS_RANGE.0 as f64,
            CHILD_FORMANT_OFFSET_CENTS_RANGE.1 as f64,
        ) as f32;
    }
    Some(out)
}

fn sample_child_offset_cents(layer: &ChildPitchOffsetLayer<'_>, frame_idx: usize) -> f64 {
    layer
        .cents_curve
        .and_then(|curve| curve.get(frame_idx).copied())
        .filter(|value| value.is_finite())
        .map(|value| value as f64)
        .unwrap_or(layer.cents)
}

fn sample_child_offset_degree_steps(layer: &ChildPitchOffsetLayer<'_>, frame_idx: usize) -> f64 {
    layer
        .degree_steps_curve
        .and_then(|curve| curve.get(frame_idx).copied())
        .filter(|value| value.is_finite())
        .map(|value| value as f64)
        .unwrap_or(layer.degree_steps)
}

fn apply_child_pitch_offset_to_midi(
    midi: f64,
    cfg: &ChildPitchOffsetConfig<'_>,
    frame_idx: usize,
    scale_notes: &[u8],
) -> f64 {
    if !(midi.is_finite() && midi > 0.0) {
        return 0.0;
    }

    let mut current = midi;
    for layer in &cfg.layers {
        let steps = sample_child_offset_degree_steps(layer, frame_idx);
        let cents = sample_child_offset_cents(layer, frame_idx);

        if steps.abs() > 1e-9 {
            current = transpose_midi_by_scale_steps(current, steps, scale_notes);
        }
        if cents.abs() > 1e-9 {
            current += cents / 100.0;
        }

        if !(current.is_finite() && current > 0.0) {
            return 0.0;
        }
    }

    current
}

/// REAPER 导出用的逐帧音高偏移样本（`compute_clip_export_pitch_offsets`）。
#[derive(Debug, Clone)]
pub(crate) struct ExportPitchOffsetFrames {
    /// 逐帧半音偏移（绝对半音量，可直接写 take PITCHENV 值）；
    /// **0 = 该帧不做音高修正**（`pitch_orig ≤ 0` 或 `pitch_edit ≤ 0` 的
    /// 无声/未分析/未编辑帧，与渲染端"值 ≤ 0 不变换"语义一致）。
    /// 帧周期与根轨道 entry 一致（调用方切片同源）。
    pub offsets: Vec<f32>,
}

/// 计算 clip 时间范围内的总音高偏移（REAPER take PITCHENV 的值源）。
///
/// 与渲染语义逐帧对齐（见 `apply_child_pitch_offset_to_midi` 的消费点）：
/// - 根轨道编辑线：`pitch_edit − pitch_orig`（读取规则与 `get_param_frames`
///   / MIDI 导出一致）；
/// - **`pitch_orig ≤ 0` 或 `pitch_edit ≤ 0` 的帧一律偏移 = 0（"不做音高
///   修正"）**——渲染端对值 ≤ 0 的帧不应用编辑与子轨变换，导出必须同款
///   处理，不得用邻近帧桥接（否则无声区会被相邻有声帧的修正值污染）；
/// - 子轨音分差/度数差：作用在**编辑后**音高上的再变换（度数差先、音分差
///   后，root → 父 → 子逐层），度数差为依赖音阶的非线性换算；
/// - 总偏移 = 子轨变换后的有效音高 − 原始音高（可超过 ±24，REAPER PIT
///   包络值域无限制，不做钳制）；
/// - `pitch_orig` 为空（未分析）时回退 `pending_pitch_offset` 切片（缺失
///   帧 = 0 = 不做修正）；
/// - 两者皆无 → 返回 None（不导出音高包络）。
pub(crate) fn compute_clip_export_pitch_offsets(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
) -> Option<ExportPitchOffsetFrames> {
    let root_track_id = timeline.resolve_root_track_id(&clip.track_id)?;
    let entry = timeline.params_by_root_track.get(&root_track_id)?;
    let frame_period_ms = entry.frame_period_ms.max(0.1);

    let clip_start = clip.start_sec.max(0.0);
    let clip_length = clip.length_sec.max(0.0);
    let start_frame = ((clip_start * 1000.0) / frame_period_ms).floor().max(0.0) as usize;
    let frame_count = (((clip_length * 1000.0) / frame_period_ms).ceil().max(1.0)) as usize;
    let end_frame = start_frame + frame_count;

    let child_cfg = active_child_pitch_offset_config(timeline, &clip.track_id);
    let scale_segments = timeline.scale_segments();

    let mut offsets: Vec<f32> = Vec::with_capacity(frame_count);

    if entry.pitch_orig.is_empty() {
        // 未分析：回退 pending（REAPER 导入的待应用偏移）；缺失帧 = 0。
        let Some(pending) = entry.pending_pitch_offset.as_ref() else {
            return None;
        };
        for frame_idx in start_frame..end_frame {
            let value = pending
                .get(frame_idx)
                .copied()
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            offsets.push(value);
        }
        return Some(ExportPitchOffsetFrames { offsets });
    }

    let mut scale_cursor = ScaleSegmentsCursor::new(&scale_segments);
    for frame_idx in start_frame..end_frame {
        let orig = entry.pitch_orig.get(frame_idx).copied().unwrap_or(0.0) as f64;
        let edit_raw = entry.pitch_edit.get(frame_idx).copied().unwrap_or(0.0) as f64;
        // 原始音高或当前音高为 0 → "不做音高修正"（偏移 0）。
        // 渲染端对 ≤ 0 的帧不应用编辑与子轨变换，导出保持一致。
        if !(orig.is_finite() && orig > 0.0) || !(edit_raw.is_finite() && edit_raw > 0.0) {
            offsets.push(0.0);
            continue;
        }
        let t_sec = (frame_idx as f64) * frame_period_ms / 1000.0;
        let effective = match child_cfg.as_ref() {
            Some(cfg) => {
                let scale_notes = scale_cursor.notes_at(t_sec);
                apply_child_pitch_offset_to_midi(edit_raw, cfg, frame_idx, scale_notes)
            }
            None => edit_raw,
        };
        if !effective.is_finite() || effective <= 0.0 {
            offsets.push(0.0);
            continue;
        }
        offsets.push((effective - orig) as f32);
    }

    Some(ExportPitchOffsetFrames { offsets })
}

pub(crate) fn build_clip_input_pitch_curve(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
    frame_period_ms: f64,
    clip_playback_rate: f64,
    is_vslib: bool,
) -> Option<Vec<f32>> {
    let child_offset_cfg = active_child_pitch_offset_config(timeline, &clip.track_id);

    let timeline_midi_raw: Vec<f32> = if is_vslib {
        Vec::new()
    } else {
        let clip_root = clip_root_track_id(timeline, clip)?;
        let clip_pitch = crate::pitch_clip::get_clip_pitch_midi_global(
            timeline,
            clip,
            &clip_root,
            frame_period_ms,
        )?;

        // 非 Loop 倒放：传入真实消费窗口 [se−len·r, se]。
        let (trim_src_start, trim_src_end) = crate::state::clip_pitch_trim_window_sec(clip);
        let mut tm = crate::pitch_clip::trim_and_resample_midi(
            &clip_pitch,
            frame_period_ms,
            trim_src_start,
            trim_src_end,
            clip_playback_rate,
            clip.length_sec.max(0.0),
            clip.loop_enabled,
            // Loop 模式的回绕周期 = 完整媒体时长；倒放从 source_end 向下环绕，
            // 与 build_loop_tiled_segment 生成的 PCM 顺序逐帧对齐。
            crate::state::clip_source_media_duration_sec(clip),
            clip.reversed && clip.loop_enabled,
        );
        // 非 Loop 倒放：trim_and_resample_midi 按升序窗口映射（文档契约
        // "调用方在输出后整体翻转"）。本曲线的下游全部按**时间线帧**消费
        // （pitch_edit 覆盖、子轨偏移、处理器输入），必须先翻转 —— 否则
        // 倒放 Clip 的输入音高相对音频整体镜像（与 midi_export / schedule
        // 的 rate≈1 分支一致）。
        if clip.reversed && !clip.loop_enabled {
            tm.reverse();
        }
        if tm.is_empty() {
            return None;
        }
        tm
    };

    let timeline_midi = if let Some(ref cfg) = child_offset_cfg {
        let fp = frame_period_ms.max(0.1);
        let start_idx = ((clip_start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
        // Tempo Map 感知：按帧时刻解析生效音阶（帧时间单调递增，使用游标）。
        let scale_segments = timeline.scale_segments();
        let mut cursor = ScaleSegmentsCursor::new(&scale_segments);
        timeline_midi_raw
            .iter()
            .enumerate()
            .map(|(local_idx, &midi)| {
                let frame_idx = start_idx.saturating_add(local_idx);
                let sec = (frame_idx as f64) * fp / 1000.0;
                let scale_notes = cursor.notes_at(sec);
                apply_child_pitch_offset_to_midi(midi as f64, cfg, frame_idx, scale_notes) as f32
            })
            .collect()
    } else {
        timeline_midi_raw
    };

    Some(timeline_midi)
}

/// 单调帧时间的音阶段游标（scale_segments 按时间升序）。
pub(crate) struct ScaleSegmentsCursor<'a> {
    segments: &'a [(f64, Vec<u8>)],
    index: usize,
}

impl<'a> ScaleSegmentsCursor<'a> {
    pub(crate) fn new(segments: &'a [(f64, Vec<u8>)]) -> Self {
        Self { segments, index: 0 }
    }

    pub(crate) fn notes_at(&mut self, sec: f64) -> &'a [u8] {
        while self.index + 1 < self.segments.len() && self.segments[self.index + 1].0 <= sec + 1e-9
        {
            self.index += 1;
        }
        self.segments
            .get(self.index)
            .map(|(_, notes)| notes.as_slice())
            .unwrap_or(&[])
    }
}

fn root_pitch_edit_state<'a>(
    timeline: &'a TimelineState,
    root_track_id: &str,
) -> Option<(&'a crate::state::Track, &'a crate::state::TrackParamsState)> {
    let track = timeline
        .tracks
        .iter()
        .find(|track| track.id == root_track_id)?;
    let entry = timeline.params_by_root_track.get(root_track_id)?;
    Some((track, entry))
}

fn clip_pitch_edit_state<'a>(
    timeline: &'a TimelineState,
    clip: &crate::state::Clip,
) -> Option<(&'a crate::state::Track, &'a crate::state::TrackParamsState)> {
    let clip_root = timeline.resolve_root_track_id(&clip.track_id)?;
    root_pitch_edit_state(timeline, &clip_root)
}

fn clip_root_track_id(timeline: &TimelineState, clip: &crate::state::Clip) -> Option<String> {
    timeline.resolve_root_track_id(&clip.track_id)
}

fn edit_midi_at_time_or_none(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    abs_time_sec: f64,
) -> Option<f64> {
    if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
        return None;
    }

    let inv_fp = 1000.0 / frame_period_ms.max(0.1);
    let idx_f = abs_time_sec * inv_fp;
    if !(idx_f.is_finite() && idx_f >= 0.0) {
        return None;
    }
    let i0 = idx_f.floor() as isize;
    if i0 < 0 {
        return None;
    }
    let i0 = i0 as usize;
    if i0 >= pitch_edit.len() {
        return None;
    }
    let i1 = (i0 + 1).min(pitch_edit.len().saturating_sub(1));
    let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

    let e0 = pitch_edit.get(i0).copied().unwrap_or(0.0) as f64;
    let e1 = pitch_edit.get(i1).copied().unwrap_or(0.0) as f64;

    let e0 = if e0.is_finite() && e0 > 0.0 {
        Some(e0)
    } else {
        None
    };
    let e1 = if e1.is_finite() && e1 > 0.0 {
        Some(e1)
    } else {
        None
    };

    match (e0, e1) {
        (None, None) => None,
        (Some(v), None) => Some(v),
        (None, Some(v)) => Some(v),
        (Some(a), Some(b)) => {
            let v = a + (b - a) * frac;
            if v.is_finite() && v > 0.0 {
                Some(v)
            } else {
                None
            }
        }
    }
}

fn clip_midi_at_time(
    frame_period_ms: f64,
    clip_start_sec: f64,
    clip_midi: &[f32],
    abs_time_sec: f64,
) -> f64 {
    if !(abs_time_sec.is_finite() && abs_time_sec >= clip_start_sec) {
        return 0.0;
    }

    let local_sec = abs_time_sec - clip_start_sec;
    let inv_fp = 1000.0 / frame_period_ms.max(0.1);
    let idx_f = local_sec * inv_fp;
    if !(idx_f.is_finite() && idx_f >= 0.0) {
        return 0.0;
    }
    let i0 = idx_f.floor() as isize;
    if i0 < 0 {
        return 0.0;
    }
    let i0 = i0 as usize;
    if i0 >= clip_midi.len() {
        return 0.0;
    }
    let i1 = (i0 + 1).min(clip_midi.len().saturating_sub(1));
    let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

    let a = clip_midi.get(i0).copied().unwrap_or(0.0) as f64;
    let b = clip_midi.get(i1).copied().unwrap_or(0.0) as f64;

    let mut a = if a.is_finite() && a > 0.0 { a } else { 0.0 };
    let mut b = if b.is_finite() && b > 0.0 { b } else { 0.0 };
    if a <= 0.0 && b > 0.0 {
        a = b;
    }
    if b <= 0.0 && a > 0.0 {
        b = a;
    }
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }

    let v = a + (b - a) * frac;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn any_user_edit_in_range(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    start_sec: f64,
    end_sec: f64,
) -> bool {
    let fp = frame_period_ms.max(0.1);
    let start_f = ((start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
    let end_f = ((end_sec.max(start_sec) * 1000.0) / fp).ceil().max(0.0) as usize;
    let end_f = end_f.min(pitch_edit.len());
    if start_f >= end_f {
        return false;
    }

    // 短片段必须密集采样，否则会漏掉尾部编辑点（导致短 clip 被误判为“无编辑”）。
    let span = end_f.saturating_sub(start_f);
    let stride = if span <= 256 {
        1
    } else {
        ((20.0 / fp).round() as usize).max(1) // ~20ms for long regions
    };
    let mut i = start_f;
    let mut last_checked = start_f;
    while i < end_f {
        let v = pitch_edit.get(i).copied().unwrap_or(0.0);
        if v.is_finite() && v > 0.0 {
            return true;
        }
        last_checked = i;
        i += stride;
    }

    // Ensure tail frame is always checked even when stride skips it.
    let tail = end_f.saturating_sub(1);
    if tail != last_checked {
        let v = pitch_edit.get(tail).copied().unwrap_or(0.0);
        if v.is_finite() && v > 0.0 {
            return true;
        }
    }
    false
}

fn any_effective_pitch_change_in_range(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    clip_start_sec: f64,
    clip_midi: &[f32],
    start_sec: f64,
    end_sec: f64,
) -> bool {
    let fp = frame_period_ms.max(0.1);
    let start_f = ((start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
    let end_f = ((end_sec.max(start_sec) * 1000.0) / fp).ceil().max(0.0) as usize;
    let end_f = end_f.min(pitch_edit.len());
    if start_f >= end_f {
        return false;
    }

    // ~100ms sampling is enough to avoid wasting expensive inference.
    // Use a small epsilon to ignore tiny float noise in MIDI curves.
    let eps_semitones = 0.10f64;
    let span = end_f.saturating_sub(start_f);
    let stride = if span <= 256 {
        1
    } else {
        ((20.0 / fp).round() as usize).max(1)
    };

    let mut i = start_f;
    let mut last_checked = start_f;
    while i < end_f {
        let abs_time_sec = (i as f64) * fp / 1000.0;

        let orig = clip_midi_at_time(frame_period_ms, clip_start_sec, clip_midi, abs_time_sec);
        if !(orig.is_finite() && orig > 0.0) {
            i += stride;
            continue;
        }

        let Some(target) = edit_midi_at_time_or_none(frame_period_ms, pitch_edit, abs_time_sec)
        else {
            i += stride;
            continue;
        };

        if !(target.is_finite() && target > 0.0) {
            i += stride;
            continue;
        }

        if (target - orig).abs() > eps_semitones {
            return true;
        }

        last_checked = i;
        i += stride;
    }

    // Ensure tail frame is always checked even when stride skips it.
    let tail = end_f.saturating_sub(1);
    if tail != last_checked {
        let abs_time_sec = (tail as f64) * fp / 1000.0;
        let orig = clip_midi_at_time(frame_period_ms, clip_start_sec, clip_midi, abs_time_sec);
        if orig.is_finite() && orig > 0.0 {
            if let Some(target) =
                edit_midi_at_time_or_none(frame_period_ms, pitch_edit, abs_time_sec)
            {
                if target.is_finite() && target > 0.0 && (target - orig).abs() > eps_semitones {
                    return true;
                }
            }
        }
    }

    false
}

/// v2: Apply pitch edit to a single clip's stereo segment in-place.
///
/// Semantics:
/// - `pitch_edit[t] > 0`: target is absolute MIDI (user-set)
/// - `pitch_edit[t] == 0`: target is the clip's own original MIDI at that time (no change)
///
/// Returns whether processing was applied.
pub fn maybe_apply_pitch_edit_to_clip_segment(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
    seg_start_sec: f64,
    sample_rate: u32,
    pcm_stereo: &mut Vec<f32>,
) -> Result<bool, String> {
    if pcm_stereo.len() < 2 {
        return Ok(false);
    }

    let Some((track, entry)) = clip_pitch_edit_state(timeline, clip) else {
        return Ok(false);
    };
    let child_offset_cfg = active_child_pitch_offset_config(timeline, &clip.track_id);
    let has_child_pitch_offset = child_offset_cfg.is_some();
    let child_formant_cfg = active_child_formant_offset_config(timeline, &clip.track_id);
    let has_child_formant_offset = child_formant_cfg.is_some();

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return Ok(false);
    }

    // 三处判定同源（见 [`hifigan_effect_flags`]），且**已应用 Compose 门禁**：
    // Compose 关闭时这三个标志全为 false，因此下面的每一个"是否触发渲染"
    // 判定都不会再因为张力/气声/共振峰而放行 —— 用户听到的就是原音频。
    let fx = hifigan_effect_flags(algo, track, entry, clip, clip_start_sec);
    let extra_processing = fx.breath;
    let tension_processing = fx.tension;
    let formant_processing = fx.formant;

    // 若这类效果启用而 Mel Stretch 又被选中，外部预拉伸必须跳过，让
    // processor 使用真实 playback_rate 完成内部时间伸缩。
    let processor_effect_processing = matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && (extra_processing
            || tension_processing
            || formant_processing
            || has_child_formant_offset);

    // Compose 关闭、且没有音高参考块、也没有子轨共振峰偏移时，不进入处理器。
    // 此时调用方按 [`processor_should_handle_stretch`] 走**外部**预拉伸
    // （该判定同样已应用 Compose 门禁），因此不存在"外部不拉伸、内部也不运行"
    // 的错位 —— 两边口径由 [`hifigan_effect_flags`] 保证一致。
    if !track.compose_enabled && !entry.has_pitch_adjustment_active && !processor_effect_processing
    {
        return Ok(false);
    }

    // 当处理器声明 handles_time_stretch 且 playback_rate != 1.0 时，
    // 即使用户没有编辑音高/张力/共振峰，也需要触发处理器渲染以执行其内部拉伸。
    let needs_processor_stretch = {
        let kind = SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
        let compose_or_pitch_adjust = track.compose_enabled || entry.has_pitch_adjustment_active;
        let handles = crate::renderer::processor_handles_time_stretch(
            kind,
            compose_or_pitch_adjust || processor_effect_processing,
        );
        let rate = (clip.playback_rate as f64).max(1e-6);
        (handles && (rate - 1.0).abs() > 1e-6) || processor_effect_processing
    };

    // v2 semantics: do nothing until the user actually modified the edit curve.
    // This avoids treating auto-synced `pitch_edit` (e.g. copied from pitch_orig) as an edit.
    // 例外：needs_processor_stretch 时必须进入处理器以执行其内部拉伸。
    // 例外：has_pitch_adjustment_active 时，音高参考块提供的 MIDI 音高数据已写入 pitch_edit，
    //       即使 pitch_edit_user_modified 为 false 也应触发渲染。
    if !entry.pitch_edit_user_modified
        && !entry.has_pitch_adjustment_active
        && !extra_processing
        && !tension_processing
        && !formant_processing
        && !has_child_pitch_offset
        && !has_child_formant_offset
        && !needs_processor_stretch
    {
        return Ok(false);
    }

    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let pitch_edit = entry.pitch_edit.as_slice();

    // 预计算处理器能力：决定 seg_end_sec 的时间轴计算方式。
    // - handles_time_stretch=true（如 vslib）：
    //     输入 PCM 为源速率，输出 = 源帧数 / playback_rate（时间轴帧数）
    // - handles_time_stretch=false：输入 PCM 已由外部时间拉伸预拉伸，帧数 = 时间轴帧数
    let kind = SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
    let clip_playback_rate = (clip.playback_rate as f64).max(1e-6);
    let processor_handles_stretch = crate::renderer::processor_handles_time_stretch(
        kind,
        track.compose_enabled || entry.has_pitch_adjustment_active || processor_effect_processing,
    );

    // Quick skip when user never set a target in this segment window.
    let seg_frames = pcm_stereo.len() / 2;
    // 输出帧数（时间轴帧数）：内部拉伸时需折算，外部预拉伸时 seg_frames 已是时间轴帧数
    let expected_out_frames = if processor_handles_stretch {
        ((seg_frames as f64) / clip_playback_rate).round().max(2.0) as usize
    } else {
        seg_frames
    };
    // seg_end_sec 始终以时间轴坐标（输出帧）计，确保音高编辑范围检测与声码器上下文一致
    let seg_end_sec = seg_start_sec + (expected_out_frames as f64) / (sample_rate.max(1) as f64);
    let has_pitch_user_edit =
        any_user_edit_in_range(frame_period_ms, pitch_edit, seg_start_sec, seg_end_sec)
            || has_child_pitch_offset;
    if !has_pitch_user_edit
        && !extra_processing
        && !tension_processing
        && !formant_processing
        && !has_child_pitch_offset
        && !has_child_formant_offset
        && !needs_processor_stretch
    {
        return Ok(false);
    }

    // 合成后端不可用（如 ONNX 会话加载失败）时，处理器只会原样返回输入，
    // 不会兑现"内部拉伸"的承诺 —— 而调用方已按 processor_should_handle_stretch
    // 跳过外部拉伸。此处必须回退外部拉伸保证长度正确，否则音频以源速率
    // 被截断/补零（导出）或变调播放（实时）。
    if !pitch_edit_backend_available_for_algo(algo) {
        if processor_handles_stretch && (clip_playback_rate - 1.0).abs() > 1e-6 {
            *pcm_stereo = crate::time_stretch::time_stretch_interleaved(
                pcm_stereo,
                2,
                sample_rate,
                expected_out_frames,
                crate::time_stretch::resolved_external_stretch_algorithm(),
            );
        }
        return Ok(false);
    }

    debug_eprintln!(
        "[pitch_edit] clip_id={} algo={:?} seg=[{:.3},{:.3}) compose_enabled={} user_modified={}",
        clip.id,
        algo,
        seg_start_sec,
        seg_end_sec,
        track.compose_enabled,
        entry.pitch_edit_user_modified
    );

    // vslib 使用自身内部分析（ANALYZE_OPTION_VOCAL_SHIFTER），不依赖 WORLD 音高轮廓；
    // 向 VslibSetPitchArray 传递绝对目标音高，不需要原始 MIDI 曲线。
    // 因此对 vslib 跳过 get_or_compute_clip_pitch_midi_global 和 any_effective_pitch_change_in_range，
    // 仅凭 any_user_edit_in_range（已在上方通过）即可触发合成。
    #[cfg(feature = "vslib")]
    let is_vslib = matches!(algo, PitchEditAlgorithm::VocalShifterVslib);
    #[cfg(not(feature = "vslib"))]
    let is_vslib = false;

    // 在渲染前统一构建 clip 输入 pitch 曲线（根轨对应曲线 + 子轨偏移变换）。
    // 音高合成依赖分析曲线；纯效果/拉伸处理（如 Compose 关闭时的气声）不依赖。
    let pitch_edit_pending = has_pitch_user_edit || entry.has_pitch_adjustment_active;
    let timeline_midi: Vec<f32> = match build_clip_input_pitch_curve(
        timeline,
        clip,
        clip_start_sec,
        frame_period_ms,
        clip_playback_rate,
        is_vslib,
    ) {
        Some(v) => v,
        None => {
            // 曲线尚未就绪（分析未完成/不可用），无法做音高合成。
            if pitch_edit_pending {
                // 若外部拉伸已被跳过（承诺由处理器内部完成），必须在此回退
                // 外部拉伸，保证该窗口内音频长度正确（否则会以源速率被
                // 截断/补零）；分析完成后 ClipPitchReady 会触发重渲染。
                if processor_handles_stretch && (clip_playback_rate - 1.0).abs() > 1e-6 {
                    *pcm_stereo = crate::time_stretch::time_stretch_interleaved(
                        pcm_stereo,
                        2,
                        sample_rate,
                        expected_out_frames,
                        crate::time_stretch::resolved_external_stretch_algorithm(),
                    );
                }
                return Ok(false);
            }
            // 纯效果/拉伸处理：不依赖音高曲线，传空曲线让处理器继续运行，
            // 其内部会以外部算法回退完成时间拉伸（breath 与非 breath 路径
            // 均已实现该回退）。
            Vec::new()
        }
    };

    if has_pitch_user_edit && !has_child_pitch_offset && !is_vslib {
        // 若音高编辑值与原始音高完全一致（无实际变化），且不需要其他效果处理，则跳过。
        let has_effective_pitch_change = any_effective_pitch_change_in_range(
            frame_period_ms,
            pitch_edit,
            clip_start_sec,
            &timeline_midi,
            seg_start_sec,
            seg_end_sec,
        );
        if !has_effective_pitch_change
            && !(extra_processing
                || tension_processing
                || formant_processing
                || needs_processor_stretch)
        {
            return Ok(false);
        }
    }

    let mut effective_pitch_edit: Vec<f32> = pitch_edit.to_vec();
    if let Some(ref cfg) = child_offset_cfg {
        // Tempo Map 感知：按帧时刻解析生效音阶（帧时间单调递增，使用游标）。
        let scale_segments = timeline.scale_segments();
        let mut cursor = ScaleSegmentsCursor::new(&scale_segments);
        for (frame_idx, value) in effective_pitch_edit.iter_mut().enumerate() {
            if *value > 0.0 {
                let sec = (frame_idx as f64) * frame_period_ms / 1000.0;
                let scale_notes = cursor.notes_at(sec);
                *value =
                    apply_child_pitch_offset_to_midi(*value as f64, cfg, frame_idx, scale_notes)
                        as f32;
            }
        }

        if is_vslib && !timeline_midi.is_empty() {
            let fp = frame_period_ms.max(0.1);
            let start_idx = ((clip_start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
            for (local_idx, midi) in timeline_midi.iter().copied().enumerate() {
                if !(midi.is_finite() && midi > 0.0) {
                    continue;
                }
                let abs_idx = start_idx.saturating_add(local_idx);
                if abs_idx >= effective_pitch_edit.len() {
                    break;
                }
                if effective_pitch_edit[abs_idx] <= 0.0 {
                    effective_pitch_edit[abs_idx] = midi;
                }
            }
        }
    }
    let pitch_edit_for_ctx = effective_pitch_edit.as_slice();

    // ── 声道扇出（channel fan-out）─────────────────────────────────────────
    // 合成算法链（ClipProcessor）是单声道的；有效声道数 > 1（真立体声素材
    // 的 Normal/Swap 模式）时，L/R 各自完整走一遍处理器链（音高曲线/参数
    // 两声道相同）后交错合并 —— 双声道进、双声道出，通道差异全程保留。
    //
    // 等效单声道（mono 源、MonoMix/MonoLeft/MonoRight 条件化）时，两个平面
    // 内容相同，只处理一次后复制 —— 与旧实现（取左声道再复制）逐位一致，
    // 单声道工作流 CPU/内存零回归。
    //
    // source_channels 未知（旧工程 take 未记录）时按单声道处理：条件化后两
    // 平面相同（mono 源复制），输出与旧实现一致；若源实为立体声但未记录，
    // 退化为旧的"左声道坍缩"行为（与升级前一致，不引入新的错误）。
    //
    // ── 为什么扇出是串行的（不要改成 rayon 并行）──────────────────────────
    // 曾评估把 L/R 两路并行化以缩短真立体声的渲染耗时。实测前置条件不成立：
    // 三个 ONNX 模型的推理会话都是**单个共享 `Arc<Mutex<Session>>`**
    // （`nsf_hifigan_onnx.rs` / `hnsep_onnx.rs` / `fcpe_onnx.rs` 的
    // `SHARED_SESSION`），HiFiGAN 的分块缓存与 HNSEP 的分离缓存也是全局
    // `Mutex`。而 ONNX 推理正是本链路的耗时主体，两路并行只会在同一把会话锁
    // 上互相争用 —— 墙钟收益接近零甚至为负。
    //
    // 更关键的是风险不对称：WORLD 路径会进入 WORLD C 库，其重入性无法从本
    // 工程侧证实，一旦判断错误是**静默的音频损坏**而非报错。
    //
    // 真立体声的提速因此走另一条路：把"内容是单声道、被混流成双声道"的假
    // 立体声在导入时折叠为单声道（见 `crate::channel_policy`），使
    // `fanout_channels` 恒为 1 —— 对这类素材是无损的，且不需要任何并发。
    let frames = seg_frames;
    // kind / clip_playback_rate / processor_handles_stretch / expected_out_frames 已在函数上方计算
    let source_channels = clip.source_channels.unwrap_or(1).max(1) as usize;
    let fanout_channels =
        crate::channel_mode::effective_channels(source_channels as u16, clip.take_channel_mode())
            as usize;

    let processed_channels: Option<Vec<Vec<f32>>> =
        MONO_SCRATCH.with(|buf| -> Result<Option<Vec<Vec<f32>>>, String> {
            let mut mono = buf.borrow_mut();

            // 通过 ClipProcessor trait 调用，解耦合成链路（含音高合成）。
            let processor = crate::renderer::get_processor(kind);
            if !processor.is_available() {
                return Ok(None);
            }

            // 从 TrackParamsState 读取声码器专属曲线/参数（Phase 5 新增字段）
            let extra_curves = &entry.extra_curves;
            let extra_params = &entry.extra_params;

            // 若 Clip 有 clip 级别覆盖，优先使用；否则 fall back 到 track 级别
            let extra_curves: &std::collections::HashMap<String, Vec<f32>> =
                clip.extra_curves.as_ref().unwrap_or(extra_curves);
            let extra_params: &std::collections::HashMap<String, f64> =
                clip.extra_params.as_ref().unwrap_or(extra_params);

            // 子轨共振峰差：把沿父轨层级累加后的生效曲线注入处理器。
            // 仅存在 child_formant_offset 时克隆 HashMap，避免普通路径额外分配。
            let child_formant_curve = if has_child_formant_offset {
                build_clip_effective_formant_shift_curve(
                    timeline,
                    clip,
                    entry,
                    entry.pitch_edit.len().max(1),
                )
            } else {
                None
            };
            let effective_extra_curves_storage;
            let extra_curves_for_ctx: &std::collections::HashMap<String, Vec<f32>> =
                if let Some(curve) = child_formant_curve {
                    effective_extra_curves_storage = {
                        let mut cloned = extra_curves.clone();
                        cloned.insert("formant_shift_cents".to_string(), curve);
                        cloned
                    };
                    &effective_extra_curves_storage
                } else {
                    extra_curves
                };

            // ── 气声分离开关闭时，把被门禁的曲线从下发数据中剥离 ──────────────
            //
            // 【为什么必须在这里做】需求是「开关关闭 ⇒ 气声与张力**不参与合成**」。
            // 仅靠 UI 置灰或决策侧判定都不够 —— 真正施加张力的是
            // `renderer::chain::HiFiGanStage::apply_rd_tension`，它直接读
            // `cc.extra_curves["hifigan_tension"]`；`process_breath` 也直接读
            // `breath_gain`。两者都不看开关。若不下发这两条曲线，
            // 处理器**根本收不到**它们，无论其内部逻辑如何都不会生效，
            // 分离路径也不会被触发（即不进 HNSEP）。
            //
            // 【为什么选这个位置】`ClipProcessContext` 全仓库只有这一处构造，
            // 在这里剥离可一次覆盖张力与气声两个消费点（以及将来新增的消费点），
            // 避免"每个消费点各判一次"带来的漂移 —— 本项目已经出现过
            // "以为有单一实现、实际存在第 2 份"的教训。
            //
            // 【性能】常态（Compose 开启且无需剥离）返回借用、零克隆；
            // 详见 `gate_hifigan_effect_curves`。
            let gated_curves =
                gate_hifigan_effect_curves(extra_curves_for_ctx, extra_params, track.compose_enabled);
            let extra_curves_for_ctx: &std::collections::HashMap<String, Vec<f32>> = &gated_curves;

            // 若处理器自己处理时间拉伸（如 vslib 使用 Timing 控制点），传递实际 playback_rate；
            // 否则 PCM 已由外部时间拉伸预处理，rate=1.0。
            let ctx_playback_rate = if processor_handles_stretch { clip_playback_rate } else { 1.0 };

            let mut outs: Vec<Vec<f32>> = Vec::with_capacity(fanout_channels);
            for channel in 0..fanout_channels {
                mono.clear();
                mono.reserve(frames); // 预分配内存
                // 跨步读取目标声道平面，消除 memset 和越界检查。
                if channel == 0 {
                    mono.extend(pcm_stereo.iter().step_by(2).take(frames).copied());
                } else {
                    mono.extend(pcm_stereo.iter().skip(1).step_by(2).take(frames).copied());
                }

                let ctx = crate::renderer::ClipProcessContext {
                    mono_pcm: mono.as_slice(),
                    channel_index: channel as u16,
                    source_fingerprint: clip.source_file_fingerprint,
                    sample_rate,
                    clip_start_sec,
                    seg_start_sec,
                    seg_end_sec: seg_start_sec
                        + (expected_out_frames as f64) / (sample_rate.max(1) as f64),
                    frame_period_ms,
                    pitch_edit: pitch_edit_for_ctx,
                    clip_midi: &timeline_midi,
                    playback_rate: ctx_playback_rate,
                    out_frames: expected_out_frames,
                    clip_id: &clip.id,
                    extra_curves: extra_curves_for_ctx,
                    extra_params,
                };
                if is_vslib && channel == 0 {
                    debug_eprintln!(
                        "[pitch_edit:vslib] dispatch clip_id={} processor={} available={} handles_stretch={} in_frames={} out_frames={} channels={} seg=[{:.3},{:.3}) rate={:.3}",
                        clip.id,
                        processor.id(),
                        processor.is_available(),
                        processor_handles_stretch,
                        mono.len(),
                        expected_out_frames,
                        fanout_channels,
                        seg_start_sec,
                        ctx.seg_end_sec,
                        ctx.playback_rate,
                    );
                }
                let out = processor.process(&ctx)?;
                if is_vslib && channel == 0 {
                    let nonzero = out.iter().filter(|&&v| v.abs() > 1e-6).count();
                    let peak = out.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
                    debug_eprintln!(
                        "[pitch_edit:vslib] result clip_id={} out_frames={} nonzero={} peak={:.6}",
                        clip.id,
                        out.len(),
                        nonzero,
                        peak,
                    );
                }
                outs.push(out);
            }
            Ok(Some(outs))
        })?;

    let Some(processed_channels) = processed_channels else {
        return Ok(false);
    };

    // 若输出尺寸与输入不同，调整 Vec 大小并写入（逐声道帧数取 min，防御
    // 处理器输出不一致 —— 正常路径两声道都等于 expected_out_frames）。
    let actual_frames = processed_channels
        .iter()
        .map(|c| c.len())
        .min()
        .unwrap_or(0);
    if processed_channels
        .iter()
        .any(|c| c.len() != expected_out_frames)
    {
        log_warn_limited!(
            "pitch_edit: output length mismatch (got {:?}, expected {}), adjusting",
            processed_channels
                .iter()
                .map(|c| c.len())
                .collect::<Vec<_>>(),
            expected_out_frames
        );
    }

    let stereo_out = actual_frames * 2;
    pcm_stereo.clear();
    pcm_stereo.reserve(stereo_out);
    if processed_channels.len() <= 1 {
        // 等效单声道：复制到双声道（消除索引越界检查，批量写入）。
        if let Some(processed) = processed_channels.first() {
            for &v in processed.iter().take(actual_frames) {
                pcm_stereo.push(v);
                pcm_stereo.push(v);
            }
        }
    } else {
        // 真立体声：L/R 交错合并。
        let l = &processed_channels[0];
        let r = &processed_channels[1];
        for f in 0..actual_frames {
            pcm_stereo.push(l[f]);
            pcm_stereo.push(r[f]);
        }
    }
    // Pad or trim to expected length if needed
    while pcm_stereo.len() < expected_out_frames * 2 {
        pcm_stereo.push(0.0);
        pcm_stereo.push(0.0);
    }
    if pcm_stereo.len() > expected_out_frames * 2 {
        pcm_stereo.truncate(expected_out_frames * 2);
    }

    Ok(true)
}

#[allow(dead_code)]
pub fn is_pitch_edit_active(timeline: &TimelineState) -> bool {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return false;
    };

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return false;
    };
    if !track.compose_enabled {
        return false;
    }

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return false;
    }

    let entry = timeline.params_by_root_track.get(&root);
    let Some(entry) = entry else {
        return false;
    };

    // v2 semantics: pitch edit is considered active only after the user modifies the edit curve.
    entry.pitch_edit_user_modified
}

#[allow(dead_code)]
pub fn is_pitch_edit_backend_available(timeline: &TimelineState) -> bool {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return false;
    };

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return false;
    };

    pitch_edit_backend_available_for_track(track)
}

pub fn semitone_to_ratio(semitones: f64) -> f64 {
    semitone_ratio(semitones)
}

/// 检测指定clip是否需要pitch edit
/// 返回true表示该clip需要pitch edit处理
#[allow(dead_code)]
pub fn does_clip_need_pitch_edit(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    does_clip_need_processor_render(timeline, clip, clip_start_sec)
}

pub fn does_clip_need_processor_render(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    let has_child_pitch_offset =
        active_child_pitch_offset_config(timeline, &clip.track_id).is_some();
    let has_child_formant_offset =
        active_child_formant_offset_config(timeline, &clip.track_id).is_some();
    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return false;
    };

    let Some((track, entry)) = root_pitch_edit_state(timeline, &clip_root) else {
        return false;
    };

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return false;
    }

    // 与 [`processor_should_handle_stretch`] / `should_process_segment` 同源，
    // 且已应用 Compose 门禁（见 [`hifigan_effect_flags`]）。
    let fx = hifigan_effect_flags(algo, track, entry, clip, clip_start_sec);
    let effect_processing = matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && (fx.any() || has_child_formant_offset);

    // 当存在非静音的音高参考块时，即使 compose_enabled 为 false，
    // 也需要触发处理器预渲染，确保音高参考块的 MIDI 数据能应用到同组的音频块。
    // 同样，用户手动绘制了 pitch 曲线时也必须触发渲染。
    if !track.compose_enabled
        && !entry.has_pitch_adjustment_active
        && !entry.pitch_edit_user_modified
        && !effect_processing
    {
        return false;
    }
    // 后端不可用且用户没有音高编辑时无需预渲染 —— 例外：效果处理仍在
    // 预渲染路径中由 maybe_apply 回退外部拉伸，保证拉伸后长度正确
    // （此判定为 true 时调用方已跳过外部拉伸，若这里不放行会回到
    // varispeed 变调播放）。
    if !pitch_edit_backend_available_for_track(track)
        && !entry.pitch_edit_user_modified
        && !effect_processing
    {
        return false;
    }

    // 当处理器声明 handles_time_stretch 且 playback_rate != 1.0 时，
    // 即使用户没有编辑音高，也需要触发处理器预渲染以执行其内部拉伸。
    let needs_processor_stretch = {
        let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
        let compose_or_pitch_adjust = track.compose_enabled || entry.has_pitch_adjustment_active;
        let handles =
            crate::renderer::processor_handles_time_stretch(kind, compose_or_pitch_adjust);
        let rate = (clip.playback_rate as f64).max(1e-6);
        handles && (rate - 1.0).abs() > 1e-6
    };

    // v2 semantics: only treat pitch edit as active after the user modified the edit curve.
    // Otherwise `pitch_edit` may be auto-synced to `pitch_orig` and contain non-zero MIDI values,
    // which should NOT trigger synthesis / prerender.
    // 例外：needs_processor_stretch 时必须触发预渲染以执行其内部拉伸。
    // 例外：has_pitch_adjustment_active 时，音高参考块提供的 MIDI 音高数据已写入 pitch_edit，
    //       即使 pitch_edit_user_modified 为 false 也应触发渲染。
    debug_eprintln!("[pitch_edit] does_clip_need_processor_render: clip={} user_modified={} has_adj={} extra={} tension={} formant={} child_off={} child_formant={} stretch={}",
        clip.id, entry.pitch_edit_user_modified, entry.has_pitch_adjustment_active,
        fx.breath, fx.tension, fx.formant, has_child_pitch_offset,
        has_child_formant_offset, needs_processor_stretch);
    if !entry.pitch_edit_user_modified
        && !entry.has_pitch_adjustment_active
        && !fx.breath
        && !fx.tension
        && !fx.formant
        && !has_child_pitch_offset
        && !has_child_formant_offset
        && !needs_processor_stretch
    {
        return false;
    }

    if fx.breath
        || fx.tension
        || fx.formant
        || has_child_pitch_offset
        || has_child_formant_offset
        || needs_processor_stretch
    {
        return true;
    }

    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let pitch_edit = entry.pitch_edit.as_slice();

    // 检查clip时间范围内是否有用户设置的pitch edit
    // 注意：这里必须使用 clip 在时间线上的可见长度（length_sec），而不是源文件时长（duration_sec）。
    // 否则当 playback_rate < 1（减速拉伸）时，clip 时间线长度会变长，后半段的编辑将不会触发合成。
    let clip_end_sec = clip_start_sec + clip.length_sec.max(0.0);
    any_user_edit_in_range(frame_period_ms, pitch_edit, clip_start_sec, clip_end_sec)
}
