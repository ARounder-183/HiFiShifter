//! 分块渲染的接缝工具：交叉淡化窗、加权叠加、边界收尾、非有限值净化。
//!
//! # 为什么要有这个模块
//!
//! 本工程的声码器都要**分块推理**：长素材一次性送入模型会吃掉数 GB 显存，
//! 且改一个参数就要整段重算。分块本身没问题，问题在于**块与块之间怎么接**。
//!
//! 历史上每条链路各写了一套接法，且都不完整：
//! - WORLD（`world_vocoder.rs`）用等功率窗，但把重叠区"覆盖后再淡化"，
//!   同一段被渲染三次，不是叠加；
//! - HiFiGAN 主路径（`nsf_hifigan_onnx.rs::assemble_bounded_chunks`）**直接
//!   `copy_from_slice` 覆盖**，既不重叠也不淡化；
//! - HiFiGAN mel-stretch 路径（`infer_from_audio_and_midi_mel_stretch`）用了
//!   线性窗 + `wsum` 归一，是唯一正确的实现。
//!
//! 三份实现、三种行为，于是"同一个工程在预览/导出/不同算法下听感不同"这类缺陷
//! 反复出现。本模块把接缝处理收敛成**唯一实现**，三条链路都调用它。
//!
//! # 权重为什么是"等功率形状 + 按权重和归一"
//!
//! 交叉淡化有两种经典口径，二者不可兼得：
//!
//! - **功率守恒**（等功率窗、**不**归一）：`cos²+sin²=1`。两路**不相关**时总功率
//!   恒定；但两路**同相**时和值为 `cos+sin ∈ [1, √2]`，接缝中点有 **+3dB 抬升**。
//! - **幅度守恒**（按权重和归一）：等价于加权平均，两路**同相**时**精确重建**
//!   （和恒为 1）；两路**反相**时会下陷。
//!
//! 本模块取**幅度守恒**。理由是分块渲染的两路内容**本来就高度相关** ——
//! 它们由同一段 mel/f0 驱动（HiFiGAN），或由同一段频谱包络驱动（WORLD），
//! 差异只在相位基准。此时"不归一化"的纯等功率会给出每块一次的 **+3dB 电平鼓包**，
//! 而电平鼓包与本模块要消除的"约每 6s 一次杂音"是**同一类可闻缺陷**，
//! 只是表现为周期性音量起伏而非咔哒。归一化后同相内容**逐样本精确重建**，
//! 反相时的下陷是平滑渐变（跨越整个淡化区），不产生瞬态。
//!
//! 因此：窗形状取等功率（`sin/cos`，频谱上比线性三角窗更干净），
//! 但**必须**配合 [`normalize_by_wsum`] 使用。单独用 [`chunk_window`] 而不归一化，
//! 等于主动引入周期性电平调制。

/// 交叉淡化的窗形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeShape {
    /// 等功率（`sin`/`cos`）。配合 [`normalize_by_wsum`] 使用。
    EqualPower,
    /// 线性互补（`t`/`1-t`）。与等功率在**归一化后**的听感差异极小，
    /// 保留是为了与既有实现（mel-stretch 路径）逐样本兼容。
    Linear,
}

/// 生成一个分块窗口的权重，写入 `out`（长度 = `len`）。
///
/// 形状是梯形：块首 `fade_in` 个样本渐入、块尾 `fade_out` 个样本渐出、
/// 中间恒为 1。首块传 `fade_in = 0`、末块传 `fade_out = 0`，
/// 于是整条输出没有"从 0 爬升"或"落到 0"的人为包络。
///
/// `fade_in + fade_out > len` 时按比例收缩，退化为**纯交叉淡化**
/// （中段为空，全程都在渐变）—— 块长比重叠区还小时的正确行为。
///
/// 【为什么末点不取到恰好 1/0】`t` 用 `(i + 0.5) / n` 而非 `i / n`：
/// 前者关于中点对称，`fade_in == fade_out` 时相邻两块的权重和严格为 1
/// （`t` 与 `1-t` 配对）；后者会在每个重叠区的**两端**各留一个样本的偏差。
pub fn chunk_window(
    len: usize,
    fade_in: usize,
    fade_out: usize,
    shape: FadeShape,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(len, 1.0);
    if len == 0 {
        return;
    }

    // 收缩到不重叠：`fade_in + fade_out > len` 时按比例缩小，
    // 保证中段的"恒 1 区"长度非负且两块权重仍严格互补。
    let (mut fi, mut fo) = (fade_in.min(len), fade_out.min(len));
    if fi + fo > len {
        let total = fi + fo;
        fi = (fi * len).div_ceil(total).min(len);
        fo = len - fi;
    }

    for i in 0..fi {
        let t = (i as f32 + 0.5) / fi as f32;
        out[i] = ramp(t, shape, true);
    }
    for i in 0..fo {
        let t = (i as f32 + 0.5) / fo as f32;
        let idx = len - fo + i;
        out[idx] = ramp(t, shape, false);
    }
}

/// 渐入（`rising`）或渐出（`!rising`）在归一化参数 `t ∈ (0,1)` 处的权重。
#[inline]
fn ramp(t: f32, shape: FadeShape, rising: bool) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match shape {
        FadeShape::EqualPower => {
            let angle = t * std::f32::consts::FRAC_PI_2;
            if rising {
                angle.sin()
            } else {
                angle.cos()
            }
        }
        FadeShape::Linear => {
            if rising {
                t
            } else {
                1.0 - t
            }
        }
    }
}

/// 把带窗的 `seg` 按权重**叠加**到 `out` / `wsum` 的 `offset` 处。
///
/// 越界部分自动截断（末块的尾部延伸可能超过总长，这是正常调用）。
/// **不覆盖**已有内容 —— 这正是它与"直接 `copy_from_slice`"的本质区别，
/// 也是同一区间能被两个相邻块同时贡献的前提。
pub fn overlap_add(out: &mut [f32], wsum: &mut [f32], offset: usize, seg: &[f32], win: &[f32]) {
    let n = seg.len().min(win.len());
    for i in 0..n {
        let dst = offset + i;
        if dst >= out.len() {
            break;
        }
        let w = win[i];
        out[dst] += seg[i] * w;
        wsum[dst] += w;
    }
}

/// 按权重和把 `out` 归一化成加权平均。`wsum` 近零处（无任何块覆盖）置 0。
///
/// 【为什么阈值是 1e-6 而不是 0】浮点误差会让"只有一个块、权重为 1"的位置
/// 得到 0.9999999 或 1.0000001；用严格 0 判断会在极少数样本上产生巨大增益。
/// 阈值只用于识别"完全没有块覆盖"的洞，不参与正常缩放。
pub fn normalize_by_wsum(out: &mut [f32], wsum: &[f32]) {
    for (o, &w) in out.iter_mut().zip(wsum.iter()) {
        if w > 1e-6 {
            *o /= w;
        } else {
            *o = 0.0;
        }
    }
}

/// 模型 hop 换算到目标采样率后的 2 倍 —— 重建内容末端的线性收尾斜坡长度。
///
/// 覆盖两类偏差之和：mel 提取的尾部窗损失（最后不足一帧的内容没有帧覆盖）
/// 与帧数取整误差，二者合计不超过 2 帧 hop。
pub fn tail_ramp_samples(hop: usize, model_sr: u32, out_sr: u32) -> usize {
    let hop_out = (hop as u64)
        .saturating_mul(out_sr.max(1) as u64)
        .div_ceil(model_sr.max(1) as u64) as usize;
    hop_out.saturating_mul(2)
}

/// 对齐输出长度到 `target_len`，并把"内容末端 ↔ 补零/截断"的边界做成
/// 线性收尾（边界处 ≈0），避免重建内容在 fade 增益仍大时硬切。
///
/// 这是本工程**唯一**的尾部对齐实现：`nsf_hifigan_onnx` 的推理路径与
/// app 侧 `commands/playback.rs` 的渲染装配都调用它。历史上后者是裸
/// `truncate` / `resize(0.0)`，在非过零点切断，每个 clip 边界都有一次咔哒。
pub fn smooth_tail_then_align(out: &mut Vec<f32>, target_len: usize, ramp: usize) {
    if out.len() > target_len {
        // 截断前：把 [target-ramp, target) 线性压到 ≈0，截断点落在收尾内。
        let start = target_len.saturating_sub(ramp);
        let n = target_len.saturating_sub(start);
        if n >= 2 {
            for i in start..target_len {
                let k = (target_len - i) as f32 / n as f32;
                out[i] *= k;
            }
        }
        out.truncate(target_len);
    } else if out.len() < target_len {
        // 补零前：把内容末端 [len-ramp, len) 线性压到 ≈0，补零区从 ≈0 开始。
        let start = out.len().saturating_sub(ramp);
        let n = out.len().saturating_sub(start);
        if n >= 2 {
            for i in start..out.len() {
                let k = (out.len() - i) as f32 / n as f32;
                out[i] *= k;
            }
        }
        out.resize(target_len, 0.0);
    }
}

/// 把非有限值（NaN / ±Inf）就地净化为 0，并夹到 `[-1, 1]`。
///
/// 【为什么必须有这一步】[`crate::util::clamp11`] 用的是 `f32::clamp`，
/// 而 `f32::clamp` 遇 NaN 返回 NaN（不 panic，也不净化）。实时输出路径上
/// 只要有一个 NaN 泄漏到设备缓冲，就是一次满量程爆音；若该 NaN 被写进
/// PCM 缓存，则会长期存在。声码器的 STFT/ISTFT 与 ONNX 推理都存在产生
/// NaN 的路径（0 除、log(0)、非有限权重），因此**在写出前统一净化**，
/// 而不是指望每条上游路径都保证有限。
pub fn sanitize_finite_in_place(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            *v = v.clamp(-1.0, 1.0);
        }
    }
}

/// 把一条 0/1 门控序列就地平滑成定长斜坡的增益。
///
/// # 为什么需要
/// "把某段置零 / 乘 0"这类门控在**边界**必然产生台阶：门控从 1 跳到 0 的那一帧，
/// 信号从满幅瞬间变成 0。只要被门控的信号有内容（不是真静音），那就是一次咔哒。
/// 本工程至少有两处这样的门控：
/// - WORLD 的干/湿混合（浊音 ↔ 非浊音切换）；
/// - 时间拉伸的硬静音保护（把静音段置零）。
///
/// # 为什么是**居中**滑动平均
/// 居中平滑的结果只依赖门控在 `[i - ramp/2, i + ramp/2]` 内的取值，因此是
/// **位置的函数**。这一点在分块渲染里至关重要：同一个位置无论被哪个块、
/// 哪一次调用处理，都得到同一个增益。历史上 WORLD 的混合用一个跨样本的状态机
/// （`w_prev` / `ramp_left`），逐块调用时每块都从 0 重启，于是**每个块首被
/// 强行淡入一次**，每 6s 一次。
///
/// `ramp < 2` 时不做任何事（没有可平滑的空间）。
pub fn smooth_binary_gate(gate: &mut [f32], ramp: usize) {
    let n = gate.len();
    if ramp < 2 || n == 0 {
        return;
    }
    let half = ramp / 2;
    let tail = ramp - half;

    // 前缀和：把每个样本的窗口求和从 O(ramp) 降到 O(1)。
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0f32);
    let mut acc = 0.0f32;
    for &g in gate.iter() {
        acc += g;
        prefix.push(acc);
    }

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + tail).min(n);
        let count = hi - lo;
        if count > 0 {
            gate[i] = (prefix[hi] - prefix[lo]) / count as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 等功率权重在 `t` 的同一处满足 `sin²+cos²=1`。
    #[test]
    fn equal_power_weights_are_unit_norm() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let (up, down) = (
                ramp(t, FadeShape::EqualPower, true),
                ramp(t, FadeShape::EqualPower, false),
            );
            assert!((up * up + down * down - 1.0).abs() < 1e-6, "t={t}");
        }
    }

    /// 线性互补权重在 `t` 的同一处和为 1。
    #[test]
    fn linear_weights_are_complementary() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let (up, down) = (
                ramp(t, FadeShape::Linear, true),
                ramp(t, FadeShape::Linear, false),
            );
            assert!((up + down - 1.0).abs() < 1e-6, "t={t}");
        }
    }

    /// 相邻两块的窗口在重叠区**严格互补** —— 这是归一化后能精确重建的前提。
    ///
    /// 构造：块长 8、重叠 4。块 A 是 [0,8)、块 B 是 [4,12)（step = 4）。
    /// A 的渐出权重应与 B 的渐入权重在同一输出位置上逐点配对。
    #[test]
    fn adjacent_chunk_windows_are_complementary_in_overlap() {
        let (len, ov) = (8usize, 4usize);
        let mut wa = Vec::new();
        let mut wb = Vec::new();
        chunk_window(len, 0, ov, FadeShape::EqualPower, &mut wa); // 非末块：有渐出
        chunk_window(len, ov, 0, FadeShape::EqualPower, &mut wb); // 非首块：有渐入

        // A 的尾部 [len-ov, len) 对应 B 的头部 [0, ov)
        for i in 0..ov {
            let a = wa[len - ov + i];
            let b = wb[i];
            assert!(
                (a * a + b * b - 1.0).abs() < 1e-5,
                "overlap index {i}: a={a} b={b}"
            );
        }
    }

    /// 首块无渐入、末块无渐出 —— 否则整条素材会有人为的淡入/淡出包络。
    #[test]
    fn first_and_last_chunk_have_no_outer_ramp() {
        let mut w = Vec::new();
        chunk_window(16, 0, 4, FadeShape::EqualPower, &mut w);
        assert_eq!(w[0], 1.0, "首块必须在块首即为满权重");
        assert!(w[15] < 0.5, "末块尾部应收敛到 0");

        chunk_window(16, 4, 0, FadeShape::EqualPower, &mut w);
        assert!(w[0] < 0.5, "非首块块首应从 0 渐入");
        assert_eq!(w[15], 1.0, "末块必须在块尾保持满权重");
    }

    /// 重叠区超过块长时退化为纯交叉淡化，不得 panic、不得出现负长度。
    #[test]
    fn oversized_overlap_degenerates_without_panic() {
        let mut w = Vec::new();
        chunk_window(4, 10, 10, FadeShape::EqualPower, &mut w);
        assert_eq!(w.len(), 4);
        assert!(w.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    }

    /// 长度 0 不 panic。
    #[test]
    fn zero_length_window_is_empty() {
        let mut w = vec![9.0];
        chunk_window(0, 0, 0, FadeShape::Linear, &mut w);
        assert!(w.is_empty());
    }

    /// 叠加而非覆盖：同一位置被两个块贡献时应当**相加**。
    #[test]
    fn overlap_add_accumulates_instead_of_overwriting() {
        let mut out = vec![0.0f32; 4];
        let mut wsum = vec![0.0f32; 4];
        overlap_add(&mut out, &mut wsum, 0, &[1.0, 1.0], &[1.0, 1.0]);
        overlap_add(&mut out, &mut wsum, 1, &[1.0, 1.0], &[1.0, 1.0]);
        // 位置 1 被两次贡献
        assert_eq!(out, vec![1.0, 2.0, 1.0, 0.0]);
        assert_eq!(wsum, vec![1.0, 2.0, 1.0, 0.0]);
    }

    /// 越界叠加被截断，不 panic。
    #[test]
    fn overlap_add_clamps_at_buffer_end() {
        let mut out = vec![0.0f32; 2];
        let mut wsum = vec![0.0f32; 2];
        overlap_add(&mut out, &mut wsum, 1, &[5.0, 6.0, 7.0], &[1.0, 1.0, 1.0]);
        assert_eq!(out, vec![0.0, 5.0]);
    }

    /// 归一化在权重和近零处置 0 而非 NaN/Inf。
    #[test]
    fn normalize_by_wsum_handles_empty_coverage() {
        let mut out = vec![5.0f32, 4.0, 6.0];
        let wsum = vec![0.0f32, 2.0, 1e-9];
        normalize_by_wsum(&mut out, &wsum);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.0);
    }

    /// 内容末端与补零/截断的交界处必须收敛到 ≈0（两侧都测）。
    ///
    /// 【为什么是"≈0"而不是"==0"】斜坡是 `(target - i) / n`，在最后一个
    /// 保留样本处取到 `1/n`。要求恰好为 0 需要把斜坡挪到 `i+1`，那会改变
    /// 既有工程的输出波形。契约是"边界处足够小"，阈值取 1/n 的量级。
    #[test]
    fn smooth_tail_then_align_lands_on_zero() {
        // 截断：内容比目标长
        let mut out = vec![1.0f32; 250];
        smooth_tail_then_align(&mut out, 200, 50);
        assert_eq!(out.len(), 200);
        assert!(out[199].abs() < 0.03, "截断点必须落在收尾内: {}", out[199]);

        // 补零：内容比目标短
        let mut out2 = vec![1.0f32; 120];
        smooth_tail_then_align(&mut out2, 160, 50);
        assert_eq!(out2.len(), 160);
        assert!(
            out2[119].abs() < 0.03,
            "补零起点必须从 ≈0 开始: {}",
            out2[119]
        );
        assert_eq!(out2[159], 0.0);
    }

    /// ramp 过短（<2 样本）时不做收尾，但仍须完成长度对齐。
    #[test]
    fn smooth_tail_then_align_tolerates_tiny_ramp() {
        let mut out = vec![1.0f32; 5];
        smooth_tail_then_align(&mut out, 3, 0);
        assert_eq!(out.len(), 3);

        let mut out2 = vec![1.0f32; 4];
        smooth_tail_then_align(&mut out2, 6, 1);
        assert_eq!(out2.len(), 6);
    }

    /// NaN / ±Inf 被净化为 0，有限值被夹到 [-1,1]。
    #[test]
    fn sanitize_removes_non_finite_and_clamps() {
        let mut buf = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, 2.0, -3.0];
        sanitize_finite_in_place(&mut buf);
        assert_eq!(buf, vec![0.0, 0.0, 0.0, 0.5, 1.0, -1.0]);
    }

    /// tail_ramp_samples 按采样率换算，且不溢出。
    #[test]
    fn tail_ramp_scales_with_sample_rate() {
        assert_eq!(tail_ramp_samples(512, 44_100, 44_100), 1024);
        assert_eq!(tail_ramp_samples(512, 44_100, 48_000), 1116);
        // 极端值不 panic（saturating）
        assert!(tail_ramp_samples(usize::MAX, 1, u32::MAX) > 0);
    }

    /// 门控平滑：只在**过渡处**产生斜坡，远离边界保持原值，且单调。
    #[test]
    fn binary_gate_smoothing_ramps_only_at_transitions() {
        let mut gate = vec![1.0f32; 40];
        for g in gate.iter_mut().skip(20) {
            *g = 0.0;
        }
        smooth_binary_gate(&mut gate, 8);

        assert_eq!(gate[0], 1.0, "远离边界必须保持原值");
        assert_eq!(gate[39], 0.0);
        assert!(
            gate.iter().any(|v| *v > 0.0 && *v < 1.0),
            "过渡处必须出现中间值（斜坡）"
        );
        for i in 0..39 {
            assert!(gate[i] >= gate[i + 1] - 1e-6, "斜坡必须单调: i={i}");
        }
    }

    /// `ramp < 2` 时不产生任何改变（没有可平滑的空间）。
    #[test]
    fn binary_gate_smoothing_is_a_noop_for_tiny_ramps() {
        let mut gate = vec![1.0f32, 1.0, 0.0, 0.0];
        let before = gate.clone();
        smooth_binary_gate(&mut gate, 1);
        assert_eq!(gate, before);
        smooth_binary_gate(&mut gate, 0);
        assert_eq!(gate, before);
    }
}
