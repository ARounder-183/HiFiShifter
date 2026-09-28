//! Take 级声道模式与统一的"源条件化"（source conditioning）实现。
//!
//! 声道模式是 **Take 级**属性，数值对齐 REAPER 的 `CHANMODE` 字段
//! （ReaTeam/Doc《State Chunk Definitions》）：
//! `0 = normal, 1 = reverse stereo, 2 = mono (downmix), 3 = mono (left), 4 = mono (right)`。
//!
//! 全工程只有本模块定义声道模式的可听效果；离线混音（mixdown）、离线单
//! clip 渲染（playback）与实时引擎共享同一套语义，禁止在别处复刻折算逻辑。

/// Take 级声道模式。磁盘/IPC 上以 `i32`（`raw()` / [`TakeChannelMode::from_raw`]）传输，
/// 数值即 REAPER `CHANMODE`，导入导出免转换。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TakeChannelMode {
    /// 双声道/按源：mono 源复制为 L/R；stereo 源原样。
    Normal = 0,
    /// 交换左右（REAPER "reverse stereo"）；对 mono 源为无操作。
    Swap = 1,
    /// 混合为单声道：(L+R)×0.5 复制到双声道。
    MonoMix = 2,
    /// 仅左声道（复制到双声道）。
    MonoLeft = 3,
    /// 仅右声道（复制到双声道）。
    MonoRight = 4,
}

impl TakeChannelMode {
    pub const ALL: [TakeChannelMode; 5] = [
        TakeChannelMode::Normal,
        TakeChannelMode::Swap,
        TakeChannelMode::MonoMix,
        TakeChannelMode::MonoLeft,
        TakeChannelMode::MonoRight,
    ];

    /// 工程文件 / IPC 的原始 i32 → 模式；越界/未知值一律回落 Normal
    /// （与 `gain` 等字段的"normalize 兜底"策略一致）。
    pub fn from_raw(v: i32) -> Self {
        match v {
            1 => TakeChannelMode::Swap,
            2 => TakeChannelMode::MonoMix,
            3 => TakeChannelMode::MonoLeft,
            4 => TakeChannelMode::MonoRight,
            _ => TakeChannelMode::Normal,
        }
    }

    /// 传输用的原始 i32。
    pub fn raw(self) -> i32 {
        self as i32
    }

    /// 规范化工程文件中可能越界的原始值（加载/合并边界调用）。
    pub fn normalize_raw(v: i32) -> i32 {
        TakeChannelMode::from_raw(v).raw()
    }

    /// REAPER `CHANMODE` 原始值 → 本工程模式（导入路径专用）。
    ///
    /// ≥5 的编码是 REAPER 对多声道源的单声道/立体声对选择，本工程仅消费
    /// 前两声道，无法保真还原，按如下规则降级：
    /// - `5..=66` = Mono(ch 3..64)（ch = v − 2）→ MonoMix；
    /// - `131..=194` = Mono(ch 65..128)（ch = v − 66）→ MonoMix；
    /// - `67..=130` / `195..=257` = 立体声对 (c, c+1)，其中 (1,2) 精确对应
    ///   Normal，其余声道对同样落回 Normal（前两声道策略下最近似的选择）。
    pub fn from_reaper_chanmode(v: i32) -> Self {
        match v {
            0..=4 => TakeChannelMode::from_raw(v),
            5..=66 => TakeChannelMode::from_mono_channel(v - 2),
            131..=194 => TakeChannelMode::from_mono_channel(v - 66),
            // 立体声对段（含 (1,2) 本身）：统一取前两声道。
            67..=257 => TakeChannelMode::Normal,
            _ => TakeChannelMode::Normal,
        }
    }

    fn from_mono_channel(channel: i32) -> Self {
        match channel {
            1 => TakeChannelMode::MonoLeft,
            2 => TakeChannelMode::MonoRight,
            _ => TakeChannelMode::MonoMix,
        }
    }

    /// 本工程 `channel_mode` 原始值 → REAPER `CHANMODE`（导出路径专用）。
    /// 标准五模式一一对应；降级过的导入值无法还原为原始 ≥5 编码，属预期。
    pub fn to_reaper_chanmode(v: i32) -> i32 {
        TakeChannelMode::from_raw(v).raw()
    }
}

/// 波形/徽章/导出推断使用的"有效声道数"：模式折叠为单声道时为 1，
/// 否则取源声道数（>2 视为 2）。
pub fn effective_channels(source_channels: u16, mode: TakeChannelMode) -> u16 {
    match mode {
        TakeChannelMode::MonoMix | TakeChannelMode::MonoLeft | TakeChannelMode::MonoRight => 1,
        _ => source_channels.max(1).min(2),
    }
}

/// 把任意声道数的交错 PCM 按声道模式条件化为**固定双声道交错** PCM。
///
/// 这是全工程唯一的声道折算实现：
/// - mono 源：所有模式都复制为 L/R（Swap 等对 mono 无操作语义）；
/// - stereo/多声道源：Normal 取前两声道原样，Swap 交换前两声道，
///   MonoLeft/MonoRight 取对应单平面复制，MonoMix 取 (ch1+ch2)×0.5。
///
/// 与重采样/反转/循环平铺逐样本可交换，调用方统一放在
/// "重采样 + 反转之后、formant/拉伸/合成之前"。
pub fn condition_take_channels(
    interleaved: &[f32],
    in_channels: u16,
    mode: TakeChannelMode,
) -> Vec<f32> {
    let in_channels = in_channels.max(1) as usize;
    let frames = interleaved.len() / in_channels;
    let mut out = Vec::with_capacity(frames * 2);
    let x = interleaved;

    if in_channels == 1 {
        // mono 源：任何模式下有效内容都是同一个采样，复制为双声道。
        for &s in x {
            out.push(s);
            out.push(s);
        }
        return out;
    }

    for f in 0..frames {
        let l = x[f * in_channels];
        let r = x[f * in_channels + 1];
        match mode {
            TakeChannelMode::Normal => {
                out.push(l);
                out.push(r);
            }
            TakeChannelMode::Swap => {
                out.push(r);
                out.push(l);
            }
            TakeChannelMode::MonoMix => {
                let v = (l + r) * 0.5;
                out.push(v);
                out.push(v);
            }
            TakeChannelMode::MonoLeft => {
                out.push(l);
                out.push(l);
            }
            TakeChannelMode::MonoRight => {
                out.push(r);
                out.push(r);
            }
        }
    }
    out
}

/// 声道模式条件化后的**有效单声道**信号（音高分析等单声道消费者专用，
/// 避免先条件化成 stereo 再均值下混的重复劳动）：
/// - mono 源 → 原采样；
/// - MonoLeft / MonoRight → 对应平面；
/// - MonoMix → (ch1+ch2)×0.5；
/// - Normal / Swap（stereo）→ (ch1+ch2)×0.5（Swap 的均值与 Normal 相同）。
pub fn effective_mono(interleaved: &[f32], in_channels: u16, mode: TakeChannelMode) -> Vec<f32> {
    let in_channels = in_channels.max(1) as usize;
    let frames = interleaved.len() / in_channels;
    let mut out = Vec::with_capacity(frames);
    let x = interleaved;

    if in_channels == 1 {
        out.extend_from_slice(x);
        return out;
    }

    match mode {
        TakeChannelMode::MonoLeft => {
            for f in 0..frames {
                out.push(x[f * in_channels]);
            }
        }
        TakeChannelMode::MonoRight => {
            for f in 0..frames {
                out.push(x[f * in_channels + 1]);
            }
        }
        _ => {
            for f in 0..frames {
                out.push((x[f * in_channels] + x[f * in_channels + 1]) * 0.5);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn raw_roundtrip_and_unknown_falls_back_to_normal() {
        for mode in TakeChannelMode::ALL {
            assert_eq!(TakeChannelMode::from_raw(mode.raw()), mode);
        }
        assert_eq!(TakeChannelMode::from_raw(-7), TakeChannelMode::Normal);
        assert_eq!(TakeChannelMode::from_raw(5), TakeChannelMode::Normal);
        assert_eq!(TakeChannelMode::from_raw(999), TakeChannelMode::Normal);
        assert_eq!(TakeChannelMode::normalize_raw(3), 3);
        assert_eq!(TakeChannelMode::normalize_raw(42), 0);
    }

    #[test]
    fn reaper_chanmode_mapping() {
        // 标准五模式一一对应。
        for v in 0..=4 {
            assert_eq!(TakeChannelMode::from_reaper_chanmode(v).raw(), v);
        }
        // 多声道 mono 段降级为 MonoMix。
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(5),
            TakeChannelMode::MonoMix
        );
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(66),
            TakeChannelMode::MonoMix
        );
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(131),
            TakeChannelMode::MonoMix
        );
        // 立体声对段：(1,2) 精确 Normal，其余降级 Normal。
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(67),
            TakeChannelMode::Normal
        );
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(195),
            TakeChannelMode::Normal
        );
        // 越界。
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(-1),
            TakeChannelMode::Normal
        );
        assert_eq!(
            TakeChannelMode::from_reaper_chanmode(258),
            TakeChannelMode::Normal
        );
        // 导出侧：标准五模式原样。
        for v in 0..=4 {
            assert_eq!(TakeChannelMode::to_reaper_chanmode(v), v);
        }
    }

    #[test]
    fn effective_channels_truth_table() {
        // 立体声源。
        assert_eq!(effective_channels(2, TakeChannelMode::Normal), 2);
        assert_eq!(effective_channels(2, TakeChannelMode::Swap), 2);
        assert_eq!(effective_channels(2, TakeChannelMode::MonoMix), 1);
        assert_eq!(effective_channels(2, TakeChannelMode::MonoLeft), 1);
        assert_eq!(effective_channels(2, TakeChannelMode::MonoRight), 1);
        // 单声道源恒为 1；多声道源按 2 计。
        assert_eq!(effective_channels(1, TakeChannelMode::Normal), 1);
        assert_eq!(effective_channels(6, TakeChannelMode::Normal), 2);
        assert_eq!(effective_channels(0, TakeChannelMode::Normal), 1);
    }

    #[test]
    fn condition_stereo_matrix() {
        // 2 帧立体声：L = [0.0, 0.5], R = [1.0, -0.25]。
        let src = [0.0f32, 1.0, 0.5, -0.25];

        let out = condition_take_channels(&src, 2, TakeChannelMode::Normal);
        assert_eq!(out, vec![0.0, 1.0, 0.5, -0.25]);

        let out = condition_take_channels(&src, 2, TakeChannelMode::Swap);
        assert_eq!(out, vec![1.0, 0.0, -0.25, 0.5]);

        let out = condition_take_channels(&src, 2, TakeChannelMode::MonoMix);
        assert_eq!(out, vec![0.5, 0.5, 0.125, 0.125]);

        let out = condition_take_channels(&src, 2, TakeChannelMode::MonoLeft);
        assert_eq!(out, vec![0.0, 0.0, 0.5, 0.5]);

        let out = condition_take_channels(&src, 2, TakeChannelMode::MonoRight);
        assert_eq!(out, vec![1.0, 1.0, -0.25, -0.25]);
    }

    #[test]
    fn condition_mono_always_duplicates() {
        let src = [0.25f32, -0.5];
        for mode in TakeChannelMode::ALL {
            let out = condition_take_channels(&src, 1, mode);
            assert_eq!(out, vec![0.25, 0.25, -0.5, -0.5], "mode={:?}", mode);
        }
    }

    #[test]
    fn condition_multichannel_uses_first_two_planes() {
        // 3 声道 × 1 帧：[c1, c2, c3] = [0.1, 0.2, 0.3]。
        let src = [0.1f32, 0.2, 0.3];
        assert_eq!(
            condition_take_channels(&src, 3, TakeChannelMode::Normal),
            vec![0.1, 0.2]
        );
        assert_eq!(
            condition_take_channels(&src, 3, TakeChannelMode::Swap),
            vec![0.2, 0.1]
        );
        let mix = condition_take_channels(&src, 3, TakeChannelMode::MonoMix);
        assert!(approx(mix[0], 0.15) && approx(mix[1], 0.15));
        assert_eq!(
            condition_take_channels(&src, 3, TakeChannelMode::MonoLeft),
            vec![0.1, 0.1]
        );
        assert_eq!(
            condition_take_channels(&src, 3, TakeChannelMode::MonoRight),
            vec![0.2, 0.2]
        );
    }

    #[test]
    fn condition_empty_and_nonintegral_len() {
        assert!(condition_take_channels(&[], 2, TakeChannelMode::Normal).is_empty());
        // 长度不足一帧（尾部残缺）时按已有整帧处理。
        let out = condition_take_channels(&[0.5f32], 2, TakeChannelMode::Normal);
        assert!(out.is_empty());
    }

    #[test]
    fn effective_mono_matrix() {
        let src = [0.0f32, 1.0, 0.5, -0.25];
        let mono = effective_mono(&src, 2, TakeChannelMode::MonoLeft);
        assert_eq!(mono, vec![0.0, 0.5]);
        let mono = effective_mono(&src, 2, TakeChannelMode::MonoRight);
        assert_eq!(mono, vec![1.0, -0.25]);
        let mono = effective_mono(&src, 2, TakeChannelMode::MonoMix);
        assert!(approx(mono[0], 0.5) && approx(mono[1], 0.125));
        let mono = effective_mono(&src, 2, TakeChannelMode::Normal);
        assert!(approx(mono[0], 0.5) && approx(mono[1], 0.125));
        // mono 源：每个样本即一帧，原样返回（模式无操作语义）。
        let mono = effective_mono(&[0.25f32, -0.5], 1, TakeChannelMode::Swap);
        assert_eq!(mono, vec![0.25, -0.5]);
    }
}
