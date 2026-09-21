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

/// 动态增益的**电平分母下限**：−60 dBFS。
///
/// 增益按 `目标 / max(原声, 本下限)` 求得，因此这个常数承担**两件事**：
///
/// 1. **定义"无内容"的门槛**：原声低于 −60 dBFS 时视为数字抖动/极低电平噪声，
///    分母被钳到下限 → 增益有界（`目标/下限 ≤ DYN_MAX_GAIN`），不会把噪声底
///    放大成刺耳嘶声。它必须远低于常见内容电平：真实素材的轻声、气声、尾音
///    普遍在 −34…−55 dBFS，门限若定在 −26 dBFS（历史值 0.05）会把它们整体
///    误判为噪声底而完全无法提升。
/// 2. **保证增益关于原声连续**（★ 伪影修复的关键，见下）。
///
/// 【为什么必须"钳分母"而不是"拒绝放大"】历史实现写成"原声低于门限时直接返回
/// 增益 1.0（拒绝放大）"，这使 `增益(原声)` 在门限处有一个**从 1 跳到
/// `目标/门限`（可达 ×1000）的无限阶跃**。近零原声段的逐帧基线值恰恰就在门限
/// 附近抖动，于是相邻帧的增益在 1 与 1000 之间跳变：
/// - 细缩放：每列只覆盖少数帧 → 列高在"几乎不可见"与"满高"之间随机跳；
/// - 粗缩放：一列合并多帧 → 只要有一帧过线就取满高 → 随机出现满高尖刺。
/// 两种表现都随缩放改变（缩放决定哪些帧被合并进同一列），正是用户报告的
/// "近零处拉大动态后水平缩放出现随机伪影"。
///
/// 钳分母则让增益处处连续（原声 → 0 时增益平滑趋近 `目标/下限` 这个上界，
/// 而不是阶跃），相邻帧的抖动只带来成比例的小变化，伪影不再产生。
/// 同时**语义完全保持**：`目标 ≤ 0`（画静音）在任何原声下都仍然返回 0。
pub(crate) const DYN_SILENCE_FLOOR: f32 = 0.001;

/// 动态增益上限：**从绝对静音下限兑现到满量程**所需的倍数（= `1/下限`）。
///
/// 【它不是独立的策略旋钮，而是 `DYN_SILENCE_FLOOR` 的推论】dyn 的语义是
/// **绝对目标电平**：用户画 0.582 就该听到 0.582。上限若小于"从下限兑现到
/// 值域顶端"所需的倍数，安静段就会"画了目标却达不到"—— 与门限误伤叠加时，
/// 用户看到的现象正是"怎么编辑都提不上去"（历史上限 ×4 = 仅 +12 dB，远不够
/// 把 −34 dBFS 提到 −4.7 dBFS 所需的 ×29）。
///
/// 分母被钳到下限后，增益的上界即 `目标/下限 ≤ 值域顶端/下限`。于是只需让
/// 上限不小于这个天然上界，高于下限的帧就能兑现**任何**合法目标；超过它的
/// 钳制永远不会被合法输入触发 —— 它只是数值兜底（防越界曲线造成的荒谬增益）。
///
/// 【放宽上限为何没有安全代价】`增益 = 目标/原声` 时输出峰值
/// `= 原声 × (目标/原声) = 目标 ≤ DYN_VALUE_MAX`。精确兑现的增益**不可能**
/// 制造削顶；把噪声底抬成嘶声的风险由**下限**承担（真正的"有无内容"判据），
/// 而不是由这个上限承担。收紧上限只会破坏目标语义，不换来任何保护。
pub(crate) const DYN_MAX_GAIN: f32 = DYN_VALUE_MAX / DYN_SILENCE_FLOOR;

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

/// dyn 的值域上下界（**唯一真源**）。
///
/// `DYN_PARAM` 的描述符与 `set_param_frames` 的写路径钳制都从这里取 ——
/// 两处曾各自内联字面量，导致写路径还停在旧上界、与描述符/互转/轴刻度
/// 全面脱节（同一份曲线在不同路径下含义不同）。
pub(crate) const DYN_VALUE_MIN: f32 = 0.0;
pub(crate) const DYN_VALUE_MAX: f32 = 1.0;

/// 共通动态参数（DYN）：逐帧**目标电平**，倍率域，绝对锚点
///（1.0 = 0 dBFS，0.5 = −6 dBFS，0 = 静音）。
///
/// 与 VocalShifter 的 DYN 同量纲：曲线本身就是电平，渲染增益由
/// `目标电平 / 原声电平` 求得（见 `compute_dyn_gain`）。
///
/// 【锚点是绝对的】1.0 恒等于数字满量程 —— 与 DAW 的峰值电平表同一坐标系，
/// 因此在别处测得 −4.7 dB 的一段，在这里画 0.582 就精确对应。任何"相对本轨道
/// 组最响段落归一化"的口径都会让倍率失去绝对意义，是被明确排除的设计
/// （见 `pitch_analysis/dyn_analysis.rs` 文件头）。
///
/// 【值域为什么是 0..1】锚点是数字满量程，而超过满量程的电平在播出去之前
/// 就会被削顶（clamp 到 ±1）—— 画 1.0 以上没有可兑现的物理意义，
/// 只会让曲线顶部一大片区域成为"无效编辑区"。真正可用的区间是 0 dBFS
/// 以下直到静音，因此把整个值域收成 0..1：面板高度全部留给有意义的部分。
pub(crate) const DYN_PARAM: ParamDescriptor = ParamDescriptor {
    id: DYN_PARAM_ID,
    display_name: "Dynamics",
    group: "Mix",
    kind: ParamKind::AutomationCurve {
        unit: "×",
        default_value: 1.0,
        min_value: DYN_VALUE_MIN,
        max_value: DYN_VALUE_MAX,
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
/// - 目标 ≤ 0（画静音）→ 0.0，**任何原声下都成立**（恒等于信号本身 ×0）；
/// - 其余 → `目标 / max(原声, DYN_SILENCE_FLOOR)`，钳制到 `[0, DYN_MAX_GAIN]`。
///
/// 【为什么是"钳分母"而不是"低于门限就拒绝放大"】后者会让增益在门限处出现
/// 无限阶跃（1 → 目标/门限），近零原声段的逐帧基线抖动会把相邻帧的显示高度
/// 在"不可见"与"满高"之间来回切换，随缩放合并方式表现为随机伪影。钳分母使
/// 增益关于原声**处处连续**，同时天然有界（原声 → 0 时增益 → 目标/门限 ≤ 上限）。
/// 详见 `DYN_SILENCE_FLOOR` 的说明。
#[inline]
pub(crate) fn compute_dyn_gain(target: f32, orig: f32) -> f32 {
    if !(target.is_finite() && orig.is_finite()) {
        return 1.0;
    }
    if target < 0.0 {
        return 1.0;
    }
    if target <= 0.0 {
        // 画静音：恒为 0（任何原声下都真静音，包括真静音与原声未知的帧）。
        return 0.0;
    }
    // 分母钳到下限：既界定"无内容"（增益有界，不放大噪声底），又保证增益
    // 关于原声连续（见 DYN_SILENCE_FLOOR 的伪影说明）。
    let denom = orig.max(DYN_SILENCE_FLOOR);
    (target / denom).clamp(0.0, DYN_MAX_GAIN)
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
        // dyn = 绝对目标电平，>1 会削顶、无意义 → 值域收在 0..1），
        // 因此值域不同（volume 0..2 / dyn 0..1），互转必须做基线补偿
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
        assert_eq!(dyn_kind, (1.0, 0.0, 1.0));
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
    fn dyn_gain_boosts_quiet_content_at_any_level() {
        // ★ 回归：真实素材的安静段必须被提升到**用户画的目标** —— 门限曾定在
        // −26 dBFS（0.05）而上限仅 ×4，两者叠加使轻声/气声/尾音
        // （−34…−58 dBFS）整体提不上去。
        for db in [-20.0f64, -26.0, -34.0, -40.0, -45.0, -55.0, -58.0] {
            let orig = 10f64.powf(db / 20.0) as f32;
            let target = 0.582f32; // −4.7 dBFS：用户实测的典型目标
            let gain = compute_dyn_gain(target, orig);
            // 目标必须**精确兑现**，而不是"能提升一点就行"——这正是 dyn 作为
            // 绝对目标电平的语义。
            assert!(
                (gain - target / orig).abs() < 1e-3,
                "{db} dBFS: 增益 {gain} != 目标/原声 {}",
                target / orig
            );
        }
    }

    #[test]
    fn dyn_gain_stays_bounded_below_the_floor() {
        // 下限之下：分母被钳到下限 → 增益有界（不无限放大噪声底），且**不再是 1**。
        assert_eq!(compute_dyn_gain(1.0, 0.0), DYN_MAX_GAIN);
        assert_eq!(
            compute_dyn_gain(1.0, DYN_SILENCE_FLOOR * 0.5),
            DYN_MAX_GAIN
        );
        // 提升被上限钳制（目标远超值域时也一样）。
        assert_eq!(
            compute_dyn_gain(DYN_MAX_GAIN * 2.0, DYN_SILENCE_FLOOR),
            DYN_MAX_GAIN
        );
        // 非有限输入不产生 NaN。
        assert_eq!(compute_dyn_gain(f32::NAN, 1.0), 1.0);
        assert_eq!(compute_dyn_gain(1.0, f32::INFINITY), 1.0);
    }

    /// ★ 伪影回归：增益必须关于原声**连续**。
    ///
    /// 故障形态：旧实现写成"原声低于下限就拒绝放大（返回 1.0）"，增益在下限处
    /// 有 1 → 目标/下限 的无限阶跃。近零原声段的逐帧基线值恰在下限附近抖动，
    /// 相邻帧的增益便在 1 与上限之间跳变 —— 波形列高随之在"不可见"与"满高"
    /// 之间随机切换，且随水平缩放（列合并哪些帧）改变位置，即用户报告的
    /// "近零处拉大动态 + 水平缩放 → 随机伪影"。
    ///
    /// 判据：跨越下限的密集采样点上，相邻增益的相对变化必须很小。
    #[test]
    fn dyn_gain_is_continuous_across_the_floor() {
        let target = 1.0f32;
        let mut prev: Option<f32> = None;
        let mut max_jump_ratio = 0.0f32;
        for k in 0..=2000 {
            // 从 下限×0.5 到 下限×2 密集采样（含跨越点）。
            let orig = DYN_SILENCE_FLOOR * 0.5 * (1.0 + k as f32 / 1000.0);
            let gain = compute_dyn_gain(target, orig);
            if let Some(p) = prev {
                if p > 0.0 && gain > 0.0 {
                    let ratio = (gain / p).max(p / gain);
                    max_jump_ratio = max_jump_ratio.max(ratio);
                }
            }
            prev = Some(gain);
        }
        // 相邻采样步长 0.05% 原声变化 → 增益变化不应超过 ~0.2%。
        assert!(
            max_jump_ratio < 1.002,
            "增益在原声跨过下限时必须连续，实测最大相邻跳变比 {max_jump_ratio}"
        );
    }

    #[test]
    fn dyn_gain_protection_never_blocks_silence_request() {
        // ★ 回归：任何原声下画 0（静音）都必须真的静音 —— 包括无内容帧与真静音。
        let quiet = DYN_SILENCE_FLOOR * 0.5;
        assert_eq!(compute_dyn_gain(0.0, quiet), 0.0);
        assert_eq!(compute_dyn_gain(0.0, DYN_SILENCE_FLOOR * 0.99), 0.0);
        assert_eq!(compute_dyn_gain(0.0, 0.0), 0.0);
        assert_eq!(compute_dyn_gain(0.0, 1.0), 0.0);
        // 下限之内的衰减照常生效（0.002 → 0.004 的一半）。
        let half = compute_dyn_gain(0.002, 0.004);
        assert!((half - 0.5).abs() < 1e-6, "got {half}");
        // 下限之下：分母恒被钳到下限 ⇒ 增益只由目标决定（与原声的具体值无关）。
        // 这保证了下限之下没有"相对电平"的台阶，是连续性/无伪影的直接体现。
        let below_a = compute_dyn_gain(0.0005, 0.0005);
        let below_b = compute_dyn_gain(0.0005, 0.0001);
        assert_eq!(below_a, below_b);
        assert!((below_a - 0.0005 / DYN_SILENCE_FLOOR).abs() < 1e-6);
    }
}
