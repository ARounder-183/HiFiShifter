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
/// 1. **定义"无内容"的门槛**：原声低于 −60 dBFS 时视为数字抖动/极低电平噪声。
///    分母被钳到下限 ⇒ 增益有界（`目标/下限 ≤ DYN_MAX_GAIN`）；**并且**由
///    [`no_content_fade`] 按 smoothstep 淡出到静音 —— 仅有"有界"还不够：−90 dB
///    的抖动 ×500 仍会把噪声底抬到 −36 dBFS（可闻嘶声）。它必须远低于常见内容
///    电平：真实素材的轻声、气声、尾音普遍在 −34…−55 dBFS，门限若定在 −26 dBFS
///    （历史值 0.05）会把它们整体误判为噪声底而完全无法提升。
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
/// 钳分母 + 平滑淡出则让增益处处连续：门限处既无阶跃、导数也连续（smoothstep
/// 的两端导数为 0），相邻帧的抖动只带来成比例的小变化，伪影不再产生。
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
/// - 其余 → `目标 / max(原声, 下限) × 无内容淡出(原声)`，前一项钳制到
///   `[0, DYN_MAX_GAIN]`，后一项是 [`no_content_fade`] 的 smoothstep 系数。
///
/// 【两项各自的职责】
/// - **钳分母**：让增益关于原声处处连续且**有界**（原声 → 0 时趋近 `目标/下限`
///   这个上界，而不是"拒绝放大"那样的阶跃 —— 后者会让门限附近抖动的基线把相邻
///   帧的增益在 1 与 1000 之间跳变，随缩放合并方式表现为随机伪影）。
/// - **无内容淡出**：把"有界"真正变成"不可闻"。仅钳分母时 −90 dB 的抖动仍会被
///   ×500 放大到 −36 dBFS（可闻嘶声）——下限的语义本来就是"无内容"，因此低于它
///   的素材（不论用户把目标拉多高）都应当淡出到静音。
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
    // 分母钳到下限：既界定"无内容"（增益有界），又保证增益关于原声连续
    //（见 DYN_SILENCE_FLOOR 的伪影说明）。
    let denom = orig.max(DYN_SILENCE_FLOOR);
    let level_targeting = (target / denom).clamp(0.0, DYN_MAX_GAIN);
    // ★ 无内容淡出：下限**以下**的原声只是抖动噪声底（16bit 抖动 ≈ −90 dBFS），
    // 把它按"目标电平"放大只会把噪声变成可听的嘶声 —— 实测 −90 dB 原声 + 目标
    // 0.5 时输出达 −36 dBFS（清楚可闻）。下限的语义本来就是"无内容"，故低于它
    // 时按 smoothstep 平滑淡出到静音。
    level_targeting * no_content_fade(orig)
}

/// 「原声是否算作**有内容**」的平滑度：静音下限之上恒 1，之下按 smoothstep 淡出到 0。
///
/// 【为什么不是硬门限】硬门限会在门限处产生阶跃（增益 1 → 目标/门限），近零原声
/// 段的逐帧基线抖动会让它变成随机咔哒 —— 这正是 `DYN_SILENCE_FLOOR` 当初拒绝
/// "低于门限就不再放大"的理由。smoothstep 在门限处**导数也连续**，既压掉噪声底，
/// 又不引入新的不连续。
///
/// 【为什么以原声（而非目标）为准】"有没有内容"是素材的属性，与用户画多高无关：
/// 同一段抖动噪声底，用户拉高目标时不该变成嘶声，拉低目标时也不该变成"被压的
/// 嘶声"。以原声判定即可让两种编辑都收敛到"静音"。
///
/// 【对未画帧的影响】未画帧的增益是 `原声 / max(原声, 下限)`，下限之上恰为 1；
/// 下限之下则 < 1 —— 即"无内容处淡出到静音"。这与下限自身"界定无内容"的语义
/// 一致（真实素材的轻声/气声/尾音普遍在 −34…−55 dBFS，都在下限之上，不受影响）。
#[inline]
fn no_content_fade(orig: f32) -> f32 {
    // 0 / 负 / 非有限 → 无内容（非有限在上游已兜底为 1，此处只作防御）。
    if !(orig > 0.0) {
        return 0.0;
    }
    let x = (orig / DYN_SILENCE_FLOOR).min(1.0);
    x * x * (3.0 - 2.0 * x)
}

/// 音频路径使用的 DYN 曲线对（目标 + 原声基线），已消除"非数值"与越界硬跳。
///
/// 由 [`resolve_dyn_curves_for_audio`] 产出，两条曲线帧栅格一致。
pub(crate) struct AudioDynCurves {
    /// 目标电平：哨兵已解析，长度 = 存储曲线长度 + 1（末帧留一格，见下）。
    pub(crate) target: Vec<f32>,
    /// 原声基线：**保持原值**（长度与存储基线相同）。⚠ 不得提前下钳 —— 见下。
    pub(crate) baseline: Vec<f32>,
}

/// 把存储形态的 DYN 曲线/基线转成**音频路径形态**。
///
/// @param dyn_curve 存储的 DYN 曲线（可含哨兵）。
/// @param orig_curve 原声电平基线；缺省 / 更短时按"该帧基线缺失"处理
///   （与采样器一致：数组之外持有末值；整条缺失时引擎直接退回增益 1）。
pub(crate) fn resolve_dyn_curves_for_audio(
    dyn_curve: &[f32],
    orig_curve: Option<&[f32]>,
) -> AudioDynCurves {
    // 与采样器同源的"该帧原声"：数组之外持有末值；无基线 → 0（引擎侧等价于增益 1）。
    let denominator_at = |i: usize| -> f32 {
        match orig_curve {
            Some(c) if !c.is_empty() => c
                .get(i.min(c.len() - 1))
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.0),
            _ => 0.0,
        }
    };

    let mut target = Vec::with_capacity(dyn_curve.len() + 1);
    for (i, &value) in dyn_curve.iter().enumerate() {
        // 与显示出口同一判定：`!(is_finite && >= 0)` = 未画（哨兵 / 缺失 / 非法）。
        if value.is_finite() && value >= 0.0 {
            target.push(value);
        } else {
            target.push(denominator_at(i));
        }
    }
    // 末帧留一格（见上文 2）。
    target.push(denominator_at(dyn_curve.len()));

    // 基线**原样**交给引擎 —— 淡出判据依赖它（见上文 ⚠）。
    let baseline = match orig_curve {
        Some(c) if !c.is_empty() => c
            .iter()
            .map(|v| if v.is_finite() && *v > 0.0 { *v } else { 0.0 })
            .collect(),
        _ => Vec::new(),
    };

    AudioDynCurves { target, baseline }
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
    fn dyn_gain_stays_bounded_and_fades_below_the_floor() {
        // 下限之下：分母被钳到下限 ⇒ 增益有界；同时按"无内容"淡出 ——
        // 真静音（原声 0）时增益归 0（输出本来就是 0，与旧的 ×1000 等效），
        // 抖动级原声则被显著压低（见 `no_content_fade` 与下方专门用例）。
        assert_eq!(compute_dyn_gain(1.0, 0.0), 0.0);
        let half_floor = compute_dyn_gain(1.0, DYN_SILENCE_FLOOR * 0.5);
        assert!(half_floor < DYN_MAX_GAIN && half_floor > 0.0);
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
        // 下限之下：增益**有界**（分母钳到下限）且随原声**连续、单调**（无内容淡出）。
        // 旧实现让增益在下限之下"只由目标决定"（与原声完全无关），从而把抖动噪声底
        // 也按目标电平放大（−90 dB 原声 + 目标 0.5 → 输出 −36 dBFS，可闻嘶声）；
        // 现行实现改为按 smoothstep 淡出 —— 连续性（无台阶）这一目标不变，只是
        // 增益随原声平滑下降。
        let below_a = compute_dyn_gain(0.001, 0.001);
        let below_b = compute_dyn_gain(0.001, 0.0001);
        assert!(below_b < below_a, "无内容越彻底，增益应越小");
        // 连续性：在原声轴上一阶上采样，相邻取值的差有界（没有台阶）。
        let mut prev = compute_dyn_gain(0.001, 0.0);
        for step in 1..=200 {
            let orig = DYN_SILENCE_FLOOR * (step as f32 / 200.0);
            let cur = compute_dyn_gain(0.001, orig);
            assert!(
                (cur - prev).abs() < 0.01,
                "下限附近增益出现台阶：{prev} → {cur}（原声 {orig}）"
            );
            prev = cur;
        }
    }

    /// ★ 用户报告的场景：原声基线 ≈ −90 dB（16bit 抖动量级）时把动态拉高，
    /// 不得把噪声底放大成可闻嘶声。
    #[test]
    fn dyn_gain_does_not_amplify_dither_level_content() {
        let dither = 10f32.powf(-90.0 / 20.0); // ≈ 3.16e-5
        for target in [0.25f32, 0.5, 1.0] {
            let gain = compute_dyn_gain(target, dither);
            let out = dither * gain; // 输出电平（峰值尺度）
            let out_db = 20.0 * out.max(1e-12).log10();
            assert!(
                out_db < -60.0,
                "目标 {target} / 原声 −90 dB：输出电平 {out_db:.1} dBFS 应低于 −60 dBFS（不可闻）"
            );
        }
        // 有内容处不受影响：−40 dBFS（真实素材的轻声量级）照常兑现目标。
        let quiet_content = 10f32.powf(-40.0 / 20.0);
        assert!((compute_dyn_gain(0.5, quiet_content) - 0.5 / quiet_content).abs() < 1e-3);
        // 下限之上恒等于 目标/原声（未画帧即恒 1）。
        for db in [-60.0f32, -50.0, -30.0, -6.0] {
            let orig = 10f32.powf(db / 20.0);
            assert!((compute_dyn_gain(orig, orig) - 1.0).abs() < 1e-6);
        }
    }

    // ── 音频路径形态的曲线解析（相邻点咔哒声的修复）─────────────────

    #[test]
    fn resolve_keeps_edited_frames_and_resolves_unset_ones() {
        let curve = [DYN_FOLLOW_ORIG, 0.4, DYN_FOLLOW_ORIG, 0.0];
        let orig = [0.25, 0.25, 0.25, 0.25];
        let r = resolve_dyn_curves_for_audio(&curve, Some(&orig));
        // 目标长度 = 存储长度 + 1（末帧留一格缓冲，见函数说明 ③）。
        assert_eq!(r.target.len(), curve.len() + 1);
        assert_eq!(r.baseline.len(), orig.len());
        // 已画帧逐值不变（含"画静音"的 0 —— 那是显式目标，不是哨兵）。
        assert_eq!(r.target[1], 0.4);
        assert_eq!(r.target[3], 0.0);
        // 未画帧解析成该帧基线。
        assert_eq!(r.target[0], 0.25);
        assert_eq!(r.target[2], 0.25);
        // 解析后不含负值 —— 逐样本插值不可能穿过 0。
        assert!(r.target.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn resolve_appends_one_frame_so_the_trailing_edge_ramps() {
        // 末尾留的一格取"该处分母"（基线之外按持有末值处理），于是编辑段收尾
        // 与内部交界等宽 —— 否则增益会从 T/原声 一步跳到 1（实测跳变 4.0）。
        let curve = [DYN_FOLLOW_ORIG, 0.5];
        let orig = [0.1, 0.1];
        let r = resolve_dyn_curves_for_audio(&curve, Some(&orig));
        assert_eq!(r.target.len(), 3);
        assert_eq!(r.target[2], 0.1);
        // 基线更短时，越界帧按持有末值（与采样器一致）：留的那一格取基线末值。
        let r2 = resolve_dyn_curves_for_audio(&curve, Some(&[0.2]));
        assert_eq!(r2.target[0], 0.2); // 未画帧 → 基线
        assert_eq!(r2.target[1], 0.5); // 已画帧 → 逐值不变
        assert_eq!(r2.target[2], 0.2); // 末帧留一格 → 该处分母（持有末值）
    }

    #[test]
    fn resolve_keeps_the_baseline_raw_so_the_fade_can_fire() {
        // ⚠ 分母（原声基线）必须**保持原值**：`compute_dyn_gain` 用它同时作分母与
        // "有没有内容"的判据。曾经在这里下钳到下限，结果淡出的输入被抹平
        //（`淡出(max(原声,下限))` 恒为 1）—— 音频端继续 ×1000 放大抖动噪声底，
        // 而预览端用原始基线、看起来是对的。
        let quiet = 10f32.powf(-90.0 / 20.0); // ≈ 3.16e-5
        let r = resolve_dyn_curves_for_audio(&[0.5, DYN_FOLLOW_ORIG], Some(&[quiet, quiet]));
        assert_eq!(r.baseline[0], quiet, "基线不得被下钳");
        assert_eq!(r.target[1], quiet, "未画帧解析成该帧基线原值");
        // 于是淡出真的会触发：高目标 + 抖动级原声 ⇒ 输出电平不可闻。
        let gain = compute_dyn_gain(r.target[0], r.baseline[0]);
        let out_db = 20.0 * (quiet * gain).max(1e-12).log10();
        assert!(out_db < -60.0, "输出电平 {out_db:.1} dBFS 应低于 −60 dBFS");
        // 对照：若把基线提前下钳（旧写法），淡出失效 —— 这正是那条 bug 的形状。
        let floored_gain = compute_dyn_gain(0.5, quiet.max(DYN_SILENCE_FLOOR));
        let floored_out_db = 20.0 * (quiet * floored_gain).max(1e-12).log10();
        assert!(
            floored_out_db > -40.0,
            "对照值应当是可闻的（说明下钳确实会抹平淡出），实测 {floored_out_db:.1} dBFS"
        );
    }

    #[test]
    fn resolve_yields_unity_gain_for_unset_frames() {
        // 关键性质：解析**不改变**未画帧的实际增益（恒为 1），只改变插值表现。
        for baseline in [1.0f32, 0.25, 0.0005, 0.0, f32::NAN] {
            let curve = [DYN_FOLLOW_ORIG];
            let orig = [baseline];
            let r = resolve_dyn_curves_for_audio(&curve, Some(&orig));
            let denom = if baseline.is_finite() && baseline > 0.0 {
                baseline
            } else {
                0.0 // 采样器对无数据帧的取值；`compute_dyn_gain` 会再钳到下限。
            };
            let gain = compute_dyn_gain(r.target[0], denom);
            if baseline.is_finite() && baseline >= DYN_SILENCE_FLOOR {
                assert!(
                    (gain - 1.0).abs() < 1e-6,
                    "基线 {baseline} 时未画帧增益应为 1，实测 {gain}"
                );
            } else {
                // 下限之下 = 无内容：未画帧也按"无内容"淡出（越接近真静音越彻底）。
                // 注意这是**渐弱**而非硬门限：−66 dB（下限的一半）时仍有 −6 dB 的
                // 残留增益，到 −90 dB 抖动级才基本归零（见 no_content_fade 的曲线）。
                assert!(
                    gain < 1.0,
                    "基线 {baseline}（无内容）时未画帧应淡出，实测增益 {gain}"
                );
                if baseline == 0.0 {
                    assert_eq!(gain, 0.0, "真静音处增益必须为 0");
                }
            }
        }
    }

    #[test]
    fn resolve_handles_missing_and_short_baseline() {
        let curve = [DYN_FOLLOW_ORIG, DYN_FOLLOW_ORIG, DYN_FOLLOW_ORIG];
        let r_none = resolve_dyn_curves_for_audio(&curve, None);
        // 原声未知 → 解析值 0、基线为空；引擎侧据此直接退回增益 1（既有语义）。
        assert!(r_none.target.iter().all(|v| *v == 0.0));
        assert!(r_none.baseline.is_empty());
        let r_short = resolve_dyn_curves_for_audio(&curve, Some(&[0.5]));
        assert_eq!(r_short.target[0], 0.5);
        assert_eq!(r_short.baseline, vec![0.5]);
        // 越界帧按持有末值（与采样器一致），而不是回落到下限。
        assert_eq!(r_short.target[1], 0.5);
        assert_eq!(r_short.target[2], 0.5);
        assert_eq!(r_short.target[3], 0.5);
    }

    #[test]
    fn resolve_treats_non_finite_as_unset() {
        // 非有限值在采样器里会让整段插值变成 NaN（增益恒 1）—— 与"未画"同义，
        // 故一并解析成基线，避免 NaN 污染相邻样本的斜坡。
        let curve = [f32::NAN, f32::INFINITY, 0.3];
        let orig = [0.2, 0.2, 0.2];
        let r = resolve_dyn_curves_for_audio(&curve, Some(&orig));
        assert_eq!(r.target[0], 0.2);
        assert_eq!(r.target[1], 0.2);
        assert_eq!(r.target[2], 0.3);
    }
}
