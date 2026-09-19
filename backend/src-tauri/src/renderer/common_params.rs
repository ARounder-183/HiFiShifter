//! 所有合成算法共通的混音级参数描述符。
//!
//! 音量（volume）、声像（pan）与动态（dyn）不依赖具体声码器内核：
//! 三者一律由音频引擎在 **mix 阶段**逐样本应用（`audio_engine/mix.rs` 实时播放、
//! `audio/mixdown.rs` 离线导出），因此任何算法（含 none / World / NSF-HiFiGAN /
//! vslib）都得到完全一致的语义，且未开启 Compose 时同样即时生效。
//!
//! 历史说明：vslib 曾把 volume/pan 写进 `VSCPINFOEX2` 控制点、由合成阶段烘焙进
//! 输出，导致「未开 Compose 不生效」「mix 阶段必须跳过」「渲染缓存键必须包含」
//! 三处特判。该路径已彻底移除 —— 相关曲线一律只在混音层消费一次，
//! 禁止任何处理器再次应用（否则会出现二次增益）。

use super::traits::{ParamDescriptor, ParamKind};

pub(crate) const VOLUME_PARAM_ID: &str = "volume";
pub(crate) const PAN_PARAM_ID: &str = "pan";
/// 动态（DYN）：逐帧目标电平（倍率域，1.0 = 0 dB）。
pub(crate) const DYN_PARAM_ID: &str = "dyn";
/// 旧 NSF-HiFiGAN 专有参数名，工程加载时迁移到 `volume`。
pub(crate) const LEGACY_HIFIGAN_VOLUME_PARAM_ID: &str = "hifigan_volume";

/// 动态参数的「未设置」哨兵：曲线该帧为负值 = 沿用原声电平（无增益变化）。
///
/// 为什么需要哨兵：dyn 的合法值域是 `0..4`（0 = 全静音，1.0 = 参考电平），
/// 因此「1.0」在语义上等于「把这一段压平到参考电平」，**不是** no-op。
/// 「不改变响度」只能由一个值域外的标记表达，取 `-1.0`。
///
/// 该哨兵只在后端存在：`get_param_frames` 在出口把它解析成真实的原声电平，
/// 前端永远看不到负值；`set_param_frames` 在入口把任何负值统一钳回哨兵。
pub(crate) const DYN_FOLLOW_ORIG: f32 = -1.0;

/// 动态增益的静音保护下限：原声电平低于此值（−26 dB）时判定为静音/噪声底，
/// **放大**请求（目标 > 原声）被拒绝（增益 1.0）；衰减/静音请求照常生效
/// —— 保护的本意是"不把噪声底抬响"，不是"噪声底不可被压静"。
pub(crate) const DYN_MIN_REF: f32 = 0.05;

/// 动态增益上限（+12 dB）。
pub(crate) const DYN_MAX_GAIN: f32 = 4.0;

/// 共通音量参数（所有算法返回同一个描述符，保证曲线在算法间切换时完全互通）。
///
/// 值域 0..2（±6 dB）：乘性增益下 >1 的提升与 <1 的衰减对称覆盖常用区间；
/// 更大的整体响度变化属于轨道增益/母带处理的职责，不该在参数线上画。
pub(crate) const VOLUME_PARAM: ParamDescriptor = ParamDescriptor {
    id: VOLUME_PARAM_ID,
    display_name: "Volume",
    group: "Mix",
    kind: ParamKind::AutomationCurve {
        unit: "×",
        default_value: 1.0,
        min_value: 0.0,
        max_value: 2.0,
    },
};

/// 共通声像参数（所有算法返回同一个描述符，-1 = 全左，1 = 全右）。
pub(crate) const PAN_PARAM: ParamDescriptor = ParamDescriptor {
    id: PAN_PARAM_ID,
    display_name: "Pan",
    group: "Mix",
    kind: ParamKind::AutomationCurve {
        unit: "",
        default_value: 0.0,
        min_value: -1.0,
        max_value: 1.0,
    },
};

/// 共通动态参数（DYN）：逐帧**目标电平**，倍率域（1.0 = 0 dB，0.5 = −6 dB）。
///
/// 与 VocalShifter 的 DYN 同量纲：曲线本身就是电平，渲染增益由
/// `目标电平 / 原声电平` 求得（见 `compute_dyn_gain`）。
///
/// 【值域为什么是 0..2】参考电平 = 轨道组最响的持续段落（99 百分位），
/// 因此 99% 的帧天然 ≤ 1.0；"画到 0 dB 以上"意味着把某段抬得比全组最响的
/// 段落还响 —— 正常编辑几乎不会发生，用户真正需要雕琢的是 0 dB 以下直到
/// 静音（−∞ dB）的整段空间。值域上限给到 +6 dB（2.0）作为极少见的提升余量，
/// 更大的提升由增益上限（`DYN_MAX_GAIN`，对安静段的提升）与音量参数接管。
pub(crate) const DYN_PARAM: ParamDescriptor = ParamDescriptor {
    id: DYN_PARAM_ID,
    display_name: "Dynamics",
    group: "Mix",
    kind: ParamKind::AutomationCurve {
        unit: "×",
        default_value: 1.0,
        min_value: 0.0,
        max_value: 2.0,
    },
};

/// 所有算法都会暴露的混音级参数（顺序即前端工具栏的「连通区域」顺序）。
pub(crate) static COMMON_MIX_PARAMS: [ParamDescriptor; 3] = [VOLUME_PARAM, PAN_PARAM, DYN_PARAM];

pub(crate) fn common_mix_params() -> &'static [ParamDescriptor] {
    &COMMON_MIX_PARAMS
}

/// 参数是否为混音阶段应用的共通音量/声像/动态（含旧 nsf 专有名）。
///
/// 该判定被渲染缓存 key 用来**排除**这些曲线：它们在混音层实时应用，
/// 改动它们不需要重新合成底层 PCM。
pub(crate) fn is_common_mix_param(param_id: &str) -> bool {
    matches!(
        param_id,
        VOLUME_PARAM_ID | PAN_PARAM_ID | DYN_PARAM_ID | LEGACY_HIFIGAN_VOLUME_PARAM_ID
    )
}

/// 区间填充/平移补位时使用的**参考值**。
///
/// 与 `default_value` 的区别只在 `dyn`：它的描述符默认值是 1.0（= 压平到参考
/// 电平），而「该帧无数据」应表达为沿用原声，即 `DYN_FOLLOW_ORIG`。若沿用
/// 描述符默认值，移动/拉伸/复制片段会给新范围凭空写入 1.0，把原本的音量包络
/// 整段压平 —— 这是静默的响度损坏。
pub(crate) fn automation_curve_pad_value(kind: crate::state::SynthPipelineKind, param_id: &str) -> f32 {
    if param_id == DYN_PARAM_ID {
        return DYN_FOLLOW_ORIG;
    }
    crate::renderer::automation_curve_default_value(kind, param_id).unwrap_or(0.0)
}

/// 采样动态（DYN）曲线在**绝对帧**处的目标电平。
///
/// 与 `sample_automation_curve`（volume/pan 用）的关键差别是**越界语义**：
/// 后者越界会持有末值（对"音量保持"是合理的），但 DYN 的语义是
/// `目标 / 原声`，把最后一个目标电平 hold 到曲线之外，等于对后续
/// 所有音频逐帧强加同一个目标电平 —— 这就是"末点污染"（历史上共振峰参数
/// 出现过同类问题，见 `renderer/chain.rs::sample_curve_at_abs_sec` 的注释）。
///
/// 因此越界回落到 `DYN_FOLLOW_ORIG` 哨兵（= 沿用原声，no-op），而不是：
/// - 持有末值 → 污染；
/// - 描述符默认值 1.0 → 那是"压平到参考电平"，同样污染。
///
/// 边界约定与共振峰修复保持一致：`[len-1, len)` 是最后一个元素自己的保持
/// 区间，仍返回末值；`idx >= len` 才算越界。
///
/// 帧索引按**绝对时间**换算，与 volume/pan 完全同源，保证三条曲线同轴。
#[inline]
pub(crate) fn sample_dyn_curve_at_frame(
    curve: Option<&[f32]>,
    abs_frame: u64,
    sample_rate: u32,
    frame_period_ms: f64,
) -> f32 {
    let Some(curve) = curve else {
        return DYN_FOLLOW_ORIG;
    };
    if curve.is_empty() {
        return DYN_FOLLOW_ORIG;
    }

    let fp = frame_period_ms.max(0.1);
    let abs_sec = abs_frame as f64 / sample_rate.max(1) as f64;
    let idx_f = (abs_sec * 1000.0) / fp;
    if !idx_f.is_finite() {
        return DYN_FOLLOW_ORIG;
    }
    let i0 = idx_f.floor().max(0.0) as usize;
    // 越界 → 哨兵。必须在插值之前判定，否则 i1 会被钳到末元素并混入 frac。
    if i0 >= curve.len() {
        return DYN_FOLLOW_ORIG;
    }
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(DYN_FOLLOW_ORIG);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 采样动态（DYN）曲线在**绝对秒**处的目标电平（导出路径）。
///
/// 与 [`sample_dyn_curve_at_frame`] 是同一份语义，只是入参单位不同。
/// 两侧必须同步修改 —— 实时监听与离线导出不一致是最难排查的一类问题。
#[inline]
pub(crate) fn sample_dyn_curve_at_sec(
    curve: Option<&[f32]>,
    abs_sec: f64,
    frame_period_ms: f64,
) -> f32 {
    let Some(curve) = curve else {
        return DYN_FOLLOW_ORIG;
    };
    if curve.is_empty() {
        return DYN_FOLLOW_ORIG;
    }

    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return DYN_FOLLOW_ORIG;
    }
    let i0 = idx_f.floor().max(0.0) as usize;
    if i0 >= curve.len() {
        return DYN_FOLLOW_ORIG;
    }
    let i1 = (i0 + 1).min(curve.len().saturating_sub(1));
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve.get(i0).copied().unwrap_or(DYN_FOLLOW_ORIG);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 由「原声电平」与「目标电平」求该帧的动态增益。
///
/// 语义（与 VocalShifter 的 DYN 一致）：
/// - 目标为哨兵（负值）→ 沿用原声 → 增益 1.0；
/// - 原声为真静音（≤ 0）→ 目标 ≤ 0 时 0.0（画静音 = 静音），否则 1.0；
/// - 原声低于静音门限（噪声底）→ **只拒绝放大请求**（目标 > 原声）：
///   增益 1.0；衰减/静音请求（目标 ≤ 原声）照常生效 —— 否则把整条曲线
///   拉到 0 后，噪声底帧（原声 0 < 值 < 门限）仍会原样发声；
/// - 其余 → `目标 / 原声`，并钳制到 `[0, DYN_MAX_GAIN]`。
#[inline]
pub(crate) fn compute_dyn_gain(target: f32, orig: f32) -> f32 {
    if !(target.is_finite() && orig.is_finite()) {
        return 1.0;
    }
    if target < 0.0 {
        return 1.0;
    }
    if orig <= 0.0 {
        // 分析电平为 0：画了静音就给 0（恒等于信号本身 ×0），提升请求给 1
        //（信号为 0，增益无实际意义，1 = 不做无谓的数值放大）。
        return if target <= 0.0 { 0.0 } else { 1.0 };
    }
    if orig < DYN_MIN_REF && target > orig {
        // 静音保护：噪声底帧只拒绝"放大"，衰减/静音照常。
        return 1.0;
    }
    (target / orig).clamp(0.0, DYN_MAX_GAIN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SynthPipelineKind;

    #[test]
    fn common_mix_params_cover_all_algorithms() {
        assert_eq!(COMMON_MIX_PARAMS.len(), 3);
        for id in [VOLUME_PARAM_ID, PAN_PARAM_ID, DYN_PARAM_ID] {
            assert!(is_common_mix_param(id), "{id} 必须是算法无关参数");
        }
        // dyn 与 volume 语义不同量纲（volume = 乘性增益，可 >1 提升；
        // dyn = 目标电平，相对最响段落归一，正常编辑几乎不需要 >1），
        // 因此值域不同（volume 0..4 / dyn 0..2），互转必须做基线补偿
        // （见 commands/params.rs::convert_mix_param），不存在"纯搬迁无损"。
        let volume = match VOLUME_PARAM.kind {
            ParamKind::AutomationCurve {
                default_value,
                min_value,
                max_value,
                ..
            } => (default_value, min_value, max_value),
            _ => panic!("volume 必须是自动化曲线"),
        };
        let dyn_kind = match DYN_PARAM.kind {
            ParamKind::AutomationCurve {
                default_value,
                min_value,
                max_value,
                ..
            } => (default_value, min_value, max_value),
            _ => panic!("dyn 必须是自动化曲线"),
        };
        assert_eq!(volume, (1.0, 0.0, 2.0));
        assert_eq!(dyn_kind, (1.0, 0.0, 2.0));
    }

    #[test]
    fn pad_value_follows_orig_for_dyn() {
        let kind = SynthPipelineKind::WorldVocoder;
        assert_eq!(automation_curve_pad_value(kind, DYN_PARAM_ID), DYN_FOLLOW_ORIG);
        // 其它参数仍用描述符默认值。
        assert_eq!(automation_curve_pad_value(kind, VOLUME_PARAM_ID), 1.0);
        assert_eq!(automation_curve_pad_value(kind, PAN_PARAM_ID), 0.0);
        // 未知参数（无描述符）回退 0.0，与旧行为一致。
        assert_eq!(automation_curve_pad_value(kind, "no_such_param"), 0.0);
    }

    #[test]
    fn dyn_gain_follows_orig_when_unset() {
        // 哨兵 → 增益 1.0（不做任何改变），无论原声多大。
        assert_eq!(compute_dyn_gain(DYN_FOLLOW_ORIG, 1.0), 1.0);
        assert_eq!(compute_dyn_gain(DYN_FOLLOW_ORIG, 0.2), 1.0);
        assert_eq!(compute_dyn_gain(-0.5, 0.2), 1.0);
    }

    #[test]
    fn dyn_gain_is_target_over_orig() {
        // 目标 0.5 / 原声 1.0 = −6 dB。
        assert!((compute_dyn_gain(0.5, 1.0) - 0.5).abs() < 1e-6);
        // 目标 1.0 / 原声 0.5 = +6 dB。
        assert!((compute_dyn_gain(1.0, 0.5) - 2.0).abs() < 1e-6);
        // 目标 0.0 = 全静音。
        assert_eq!(compute_dyn_gain(0.0, 1.0), 0.0);
    }

    #[test]
    fn dyn_gain_protects_silence_and_clamps() {
        // 原声落在噪声底：绝不放大。
        assert_eq!(compute_dyn_gain(4.0, 0.0), 1.0);
        assert_eq!(compute_dyn_gain(4.0, DYN_MIN_REF * 0.5), 1.0);
        // 提升被上限钳制到 +12 dB。
        assert_eq!(compute_dyn_gain(4.0, DYN_MIN_REF), DYN_MAX_GAIN);
        // 非有限输入不产生 NaN。
        assert_eq!(compute_dyn_gain(f32::NAN, 1.0), 1.0);
        assert_eq!(compute_dyn_gain(1.0, f32::INFINITY), 1.0);
    }

    #[test]
    fn dyn_gain_protection_never_blocks_silence_request() {
        // ★ 回归：噪声底帧（0 < 原声 < 门限）画 0（静音）必须真的静音 ——
        // 旧实现把保护无条件作用于所有请求，导致"全曲线拉到 0 后噪声底帧仍发声"。
        let quiet = DYN_MIN_REF * 0.5; // 0.025，门限之下但非零
        assert_eq!(compute_dyn_gain(0.0, quiet), 0.0);
        assert_eq!(compute_dyn_gain(0.0, DYN_MIN_REF * 0.99), 0.0);
        // 真静音（分析电平 0）：画 0 → 0。
        assert_eq!(compute_dyn_gain(0.0, 0.0), 0.0);
        // 门限之下的衰减请求照常生效（0.025 → 0.0125 = 一半）。
        let half = compute_dyn_gain(quiet * 0.5, quiet);
        assert!((half - 0.5).abs() < 1e-6, "got {half}");
        // 门限之下的放大请求仍被拒绝。
        assert_eq!(compute_dyn_gain(quiet * 2.0, quiet), 1.0);
    }
}
