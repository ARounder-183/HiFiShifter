//! 参数线 ⇄ 宿主包络的**纯数学**：时间域换算、包络取值、断点简化。
//!
//! 【为什么单独一个模块】同一个换算出现三次：独立 App 导出 `.rpp`、独立 App 导入
//! `.rpp`、以及插件与 ARA 宿主实时互转。三处各写一份的直接后果是"同一段曲线在
//! 导出文件里和插件里位置不同"—— 而那是最难被察觉的一类 bug：听起来只是"偏了一点"。
//! 把时间域换算与断点简化收到这里，三处都只调这一个来源。
//!
//! 【时间域只有一个换算点】宿主包络的横轴 u 是 **take 的播放时间 × 速率**：
//!
//! <center>u = (t − item_start) × |rate|</center>
//!
//! u = 0 落在 take 的**入点**（即 `SOFFS` 所在处），沿播放方向增长；媒体文件时间
//! 是 `SOFFS + u`（倒放 take 沿反方向减少）。
//!
//! 【为什么不是媒体时间】真机渲染实测（REAPER 7.85，
//! `docs/2026-11-17-take-envelope-axis-findings.md`）：同一份 take 几何上，把点写在
//! `u = SOFFS + offset` 上**一个半音都不生效**，写在 `u = offset` 上立刻出现 +12
//! 半音的斜坡；`STARTPOS` / `MODE`（倒放）/ `LOOP` 同样不进坐标，改 `SOFFS` 也不会
//! 让宿主重定基线（`PT` 逐字保留）。所以任何调用点都**不许**再补源起点 —— 多一个
//! `SOFFS` 就是整条曲线平移，听起来是"整个片段跑调"。
//!
//! 倒放 take 也一样 —— u 是**播放顺序**的时间，方向由 SECTION 表达，不改变坐标域。
//! 因此本模块只提供 [offset_to_take_u] / [take_u_to_offset] 一对函数，任何调用点都
//! 不许就地再乘一次速率。
//!
//! 【第二个纯数学问题：未识别帧怎么填】分析产物用 `0` 当"这一帧没有音高"的哨兵，
//! 而宿主包络是**连续控制信号** —— 它没有帧，也没有"这一帧不处理"。逐帧写 0 会让
//! 移调器在几十毫秒内把比率从 +12 拉到 0 再拉回 +12。[`fill_unvoiced_offsets`]
//! 把"没有音高"从**值**里拿出来，改成"跟随邻域"，四个档位见 [`UnvoicedFill`]。
//! 它与断点简化是**两步**：先填、后简化（端点强制与同时间点合并都在简化里）。

use serde::{Deserialize, Serialize};

/// 音高断点简化的容差（半音）。
///
/// 【为什么是 0.02】逐帧 5 ms 写点，三分钟片段就是 36000 个点 —— REAPER 的包络
/// 编辑器会卡死。0.02 半音 = 2 音分，低于人耳在正常语境下的分辨阈值，同时把
/// 一条平滑曲线压到几十个点。独立 App 的导出与插件的桥接共用这一个数，
/// 否则"导出听起来对、插件里听起来不对"。
pub const SIMPLIFY_TOLERANCE_PITCH: f64 = 0.02;
/// 音量断点简化的容差（倍率）。
pub const SIMPLIFY_TOLERANCE_VOLUME: f64 = 0.005;
/// 声像断点简化的容差（−1..1）。
pub const SIMPLIFY_TOLERANCE_PAN: f64 = 0.005;

/// 宿主包络上的一个点。
///
/// `time` 是**包络自己的时间域**（take 播放时间 × 速率 u，秒；u = 0 在入点，
/// 见模块文档），`value` 是该点的值（音高包络里是半音）。两者都可能是负的：
/// u 在左延伸 item 上确实为负。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnvelopePoint {
    pub time: f64,
    pub value: f64,
}

impl EnvelopePoint {
    pub fn new(time: f64, value: f64) -> Self {
        Self { time, value }
    }
}

/// 工程时间相对 item 起点的偏移 → take 媒体时间 u。
pub fn offset_to_take_u(offset_sec: f64, rate: f64) -> f64 {
    let rate = if rate.is_finite() && rate.abs() > 1e-9 {
        rate.abs()
    } else {
        1.0
    };
    offset_sec * rate
}

/// take 媒体时间 u → 工程时间相对 item 起点的偏移。
pub fn take_u_to_offset(u: f64, rate: f64) -> f64 {
    let rate = if rate.is_finite() && rate.abs() > 1e-9 {
        rate.abs()
    } else {
        1.0
    };
    u / rate
}

/// 在包络上线性插值取值。
///
/// 【为什么区间外取端点值而不是 0】REAPER 的包络在首末点之外**保持**端点值
/// （没有点 = 默认值，有点 = 端点 hold）。按 0 处理会让"整段 +2 半音"在
/// 第一点之前变成 0，用户听到的是开头一跳。
pub fn sample_envelope(points: &[EnvelopePoint], u: f64) -> f64 {
    if points.is_empty() || !u.is_finite() {
        return 0.0;
    }
    let index = points.partition_point(|point| point.time < u);
    if index == 0 {
        return points[0].value;
    }
    if index >= points.len() {
        return points[points.len() - 1].value;
    }
    let (t0, v0) = (points[index - 1].time, points[index - 1].value);
    let (t1, v1) = (points[index].time, points[index].value);
    if (t1 - t0).abs() < 1e-12 {
        return v0;
    }
    let fraction = ((u - t0) / (t1 - t0)).clamp(0.0, 1.0);
    v0 + (v1 - v0) * fraction
}

/// 逐帧偏移 → 包络采样点（帧 0 = item 起点）。
pub fn pitch_samples(offsets: &[f32], frame_period_ms: f64) -> Vec<(f64, f64)> {
    let period = if frame_period_ms.is_finite() && frame_period_ms > 0.0 {
        frame_period_ms
    } else {
        5.0
    };
    offsets
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 * period / 1000.0, *value as f64))
        .collect()
}

/// 逐帧偏移 → 宿主 take 音高包络的断点。
///
/// 三步：采样 → 断点简化 → 换到 u 域，并**强制两端点**（u = 0 与 u = 长度 × 速率）。
/// 【为什么端点必须强制】REAPER 的包络在首个点之前/末个点之后是 hold 语义；
/// 不强制端点就会让"片段开头那一段"和"结尾那一段"继承邻居的值，跨 item 边界
/// 尤其明显（同一轨上前后两个片段会互相污染）。
pub fn build_take_pitch_points(
    offsets: &[f32],
    frame_period_ms: f64,
    item_length_sec: f64,
    rate: f64,
) -> Vec<EnvelopePoint> {
    let samples = pitch_samples(offsets, frame_period_ms);
    let range_end = item_length_sec.max(1e-6);
    let breakpoints = simplify_breakpoints(&samples, 0.0, range_end, 0.0, SIMPLIFY_TOLERANCE_PITCH);
    breakpoints
        .into_iter()
        .map(|(offset, value)| EnvelopePoint::new(offset_to_take_u(offset, rate), value))
        .collect()
}

/// 把逐帧采样压成折线断点。
///
/// 逐帧 5 ms 写点，三分钟片段就是 36000 个点 —— 包络编辑器会卡死。这里的约定：
///
/// - **全部落在容差内等于默认值** ⇒ 空结果（不写包络，保持宿主原状）；
/// - **端点强制**：恒写 `range_start` 与 `range_end` 两端点（不得依赖 hold 语义
///   跨 Item / 段边界）；
/// - **平坦非默认 → 恰好 2 点；斜线 → 仅折点**。
pub fn simplify_breakpoints(
    samples: &[(f64, f64)],
    range_start: f64,
    range_end: f64,
    default_value: f64,
    tolerance: f64,
) -> Vec<(f64, f64)> {
    if samples.is_empty() || range_end <= range_start {
        return Vec::new();
    }
    if samples
        .iter()
        .all(|(_, value)| (value - default_value).abs() <= tolerance)
    {
        return Vec::new();
    }
    if samples.len() == 1 {
        let value = samples[0].1;
        let first = samples[0].0.clamp(range_start, range_end);
        return vec![(first, value), (range_end, value)];
    }
    // O(n) 快速通道：恒平坦 / 全程线性（最常见形态）直接给两端点。
    {
        let (t0, v0) = samples[0];
        let (tn, vn) = samples[samples.len() - 1];
        let span = tn - t0;
        let slope = if span > 1e-12 { (vn - v0) / span } else { 0.0 };
        let mut flat = true;
        let mut linear = true;
        for (t, v) in samples.iter() {
            if (v - v0).abs() > tolerance {
                flat = false;
            }
            if (v - (v0 + slope * (t - t0))).abs() > tolerance {
                linear = false;
            }
            if !flat && !linear {
                break;
            }
        }
        if flat || linear {
            let first_t = t0.max(range_start);
            let last_t = range_end.max(first_t);
            return vec![
                (first_t, v0 + slope * (first_t - t0)),
                (last_t, v0 + slope * (last_t - t0)),
            ];
        }
    }

    let value_at = |t: f64| -> f64 {
        let index = samples.partition_point(|sample| sample.0 < t);
        if index == 0 {
            return samples[0].1;
        }
        if index >= samples.len() {
            return samples[samples.len() - 1].1;
        }
        let (t0, v0) = samples[index - 1];
        let (t1, v1) = samples[index];
        if (t1 - t0).abs() < 1e-12 {
            return v0;
        }
        let fraction = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        v0 + (v1 - v0) * fraction
    };

    // 【同一时间点必须合并 —— 不要为了"更连续"删掉这条】宿主包络里同一个 u 上不能
    // 有两个值：两个值意味着"这一刻有两段"，而宿主只能任选其一，选哪一个不由我们
    // 决定。这条合并也正是"写不出垂直阶跃"的原因 —— 未识别帧填充在 `ramp_frames = 0`
    // 时的阶跃会被表达成相邻两帧之间的 **5 ms 斜坡**。那是**结果**，不是缺陷：
    // 想要更长的过渡就调 `UnvoicedFillPolicy::ramp_frames`，不要改这里。
    let mut out: Vec<(f64, f64)> = Vec::new();
    let push_point = |t: f64, v: f64, out: &mut Vec<(f64, f64)>| {
        if let Some(last) = out.last_mut() {
            if (last.0 - t).abs() <= 1e-9 {
                last.1 = v;
                return;
            }
        }
        out.push((t, v));
    };

    let first_t = samples[0].0.max(range_start);
    push_point(first_t, value_at(first_t), &mut out);

    // 【弦斜率窗口】贪心"从锚点能看到多远"的判定：候选弦 (anchor → f) 成立
    // ⟺ 每个中间样本 p 都满足 |v_p - va - slope·(t_p - ta)| ≤ tolerance。
    // 对 slope 解一次不等式，这条约束就变成一个区间（窗口）；把中间样本的
    // 窗口逐个求交是 O(1)，于是整段贪心从"每个候选都从头复扫 [a+1, f)"
    // 的平方降到线性 —— 与直接求值代数等价，只是把同一约束换了次序计算。
    // 动机：长直线段后接折点（一段滑音接颤音）此前恰好落在平方路径上，
    // 3 分钟的 5 ms 帧曲线 ≈ 3.6 万样本、~10⁸ 次浮点求值，而写包络发生在
    // 编辑落定的 UI 线程上 —— 表现为可感的卡顿。
    let mut anchor = 0usize;
    loop {
        let (ta, va) = samples[anchor];
        // 弦 (anchor → f) 的可行斜率窗口：空约束集 = 全体斜率可行。
        let mut slope_lo = f64::NEG_INFINITY;
        let mut slope_hi = f64::INFINITY;
        let mut farthest = anchor + 1;
        let mut f = anchor + 1;
        while f < samples.len() {
            let (tf, vf) = samples[f];
            let span = tf - ta;
            let fits = if span <= 1e-12 {
                // 与直接求值同一条退化分支：终点与锚点几乎同时刻，弦没有斜率。
                (vf - va).abs() <= tolerance
            } else {
                let slope = (vf - va) / span;
                slope_lo <= slope && slope <= slope_hi
            };
            if !fits {
                break;
            }
            farthest = f;
            // 终点 f 自此成为下一个候选的中间样本：把它的窗口并入交集。
            // （对 f 自身该约束恒真 —— 弦正好穿过它，直接求值同样不检查它。）
            // span == 0（与锚点同刻）时偏差与斜率无关，且必在容差内 —— 上面
            // 刚按值差放行，所以对窗口无贡献，跳过（也避开 0 除的 inf/NaN）。
            let mid = vf - va;
            if span > 0.0 {
                slope_lo = slope_lo.max((mid - tolerance) / span);
                slope_hi = slope_hi.min((mid + tolerance) / span);
            } else if span < 0.0 {
                // 时间倒退（畸形输入）：除以负数翻转区间端点。
                slope_lo = slope_lo.max((mid + tolerance) / span);
                slope_hi = slope_hi.min((mid - tolerance) / span);
            }
            f += 1;
        }
        let (t, v) = samples[farthest];
        push_point(t, v, &mut out);
        anchor = farthest;
        if anchor + 1 >= samples.len() {
            break;
        }
    }

    // 末端点（强制）。末点值与端值在容差内相等时把末点延伸到 range_end，
    // 而不是追加第四点 —— 保持"线性段只有端点"的精简不变式。
    let end_v = value_at(range_end);
    if let Some(last) = out.last_mut() {
        if (last.1 - end_v).abs() <= tolerance {
            last.0 = range_end;
            last.1 = end_v;
        } else {
            push_point(range_end, end_v, &mut out);
        }
    } else {
        push_point(range_end, end_v, &mut out);
    }
    out
}

// ─── 未识别帧的填充（导出侧） ────────────────────────────────────────────────
//
// 分析产物 `pitch_orig` 用 `0` 当哨兵，同时表示"这一帧没有 F0""还没分析""超出
// 音域"三件事。而宿主的 take 音高包络是一条**连续的控制信号** —— 它没有帧，也没
// 有"这一帧不处理"。把逐帧的"不处理"写成显式 0 点，等于让宿主的移调器在 1–2 帧内
// 把比率从 +12 拉到 0 再拉回 +12：每个辅音处一下"咯噔"。
//
// 填充把"没有音高"从**值**里拿出来，改成"跟随邻域"。三个前提：
//
// 1. 它只作用于**导出边界**（`compute_clip_take_pitch_offsets`）。同一份偏移还是
//    渲染端子轨偏移的输入，而渲染端对无 F0 帧的语义就是"不变换" —— 改那里等于顺手
//    改掉渲染。
// 2. `voiced` 的口径是"至少有一侧定义了音高"，**不是**"偏移不为 0"，也**不是**
//    `orig > 0 && edit > 0`（后者的后果见 `fill_unvoiced_offsets` 的文档）。
// 3. 填充在断点简化**之前**：简化器的端点强制与同时间点合并都保持原样。

/// 未识别帧（unvoiced）的填充策略。
///
/// 【为什么不叫"平滑"】平滑会改用户画出来的曲线。这里只动"本来没有值"的帧 ——
/// 用户主动画成 0 的有声帧（`voiced = true, offset = 0`）一个字节都不许变。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnvoicedFill {
    /// 今天的行为：unvoiced 帧保持 0。
    ///
    /// **这是回归开关**：取它时 [`fill_unvoiced_offsets`] 的第一行就原样返回，
    /// 输出必须与引入填充之前逐字节一致。
    Zero,
    /// 一律 hold 前一个 voiced 值；只有整段静音才插值。
    Hold,
    /// 短间隙插值；长且可闻的间隙 hold 前值；静音段插值。**默认。**
    #[default]
    Bridge,
    /// 短间隙插值；长且可闻的间隙 hold 0（带斜坡）；静音段插值。
    ///
    /// 【为什么必须留这一档】渲染端把**音高线上没有值**的帧（齿音、气爆、静音）
    /// 原样混回（`world_vocoder::hand_back_unpitched_frames`），所以它们在插件里
    /// 本来就不被移调。这一档是"与渲染端对齐"的候选：齿音在 REAPER 里也不被移调。
    /// 哪一档更好听取决于素材与算法，只能靠真机 A/B，所以必须有一条能直接对比的路。
    Dry,
}

impl UnvoicedFill {
    /// 宽松解析。**非法值回退默认**，不报错。
    ///
    /// 配置是手可编辑的文本，一个拼写错误不该让整份设置加载失败（与
    /// `ExportSettings` 的既有惯例一致）。
    pub fn from_name(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "zero" => Self::Zero,
            "hold" => Self::Hold,
            "dry" => Self::Dry,
            _ => Self::Bridge,
        }
    }

    /// 规范名（与 [`Self::from_name`] 互逆，供配置回写与日志使用）。
    pub fn as_name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Hold => "hold",
            Self::Bridge => "bridge",
            Self::Dry => "dry",
        }
    }
}

/// 短间隙的上限（毫秒）。**以帧存储、以毫秒声明** —— 帧周期不是常量。
///
/// 【为什么是 30 ms，而不是"越长越平滑"的 120 ms】渲染端对同一件事有两处补洞，
/// 都是 15 ms（`world_vocoder::cleanup_f0_inplace` 补估计器、`close_pitch_gaps`
/// 补音高线）。120 ms 会覆盖典型齿音（80–150 ms），把渲染端**故意**保持干声的
/// 段落在 REAPER 里移调 —— 那不是在修不一致，是在制造另一种不一致。30 ms 覆盖
/// 爆破音（5–20 ms）与短气口，同时远小于最短的典型齿音。
pub const MAX_BRIDGE_MS: f64 = 30.0;

/// 允许插值的两端偏移差上限（半音）。
///
/// 同号且差值小才插值：在"前一句 +12、后一句 −5"的换气处插值会造出一个谁也没画
/// 过的中间调，而那个中间调会**穿过 0**（= 移调量为零的那一段恰好落在气口中间）。
pub const MAX_BRIDGE_SPAN_SEMITONES: f64 = 2.0;

/// 判定"静音"的峰值门（`1.0` = 0 dBFS，0.003 ≈ −50 dB）。
///
/// 【口径是**峰值**，不是 RMS】这里的输入是 `dyn_orig` 的逐帧峰值电平。
/// 渲染端那条同量级的判据（曾经的 `WORLD_DRY_SILENCE_RMS`，20 ms 窗 RMS）已经
/// 随"干/湿门改由音高线决定"一起删除 —— 导出侧不需要跟着它，因为音高线上有没有值
/// 已经是显式信息，不必再从电平反推。
pub const SILENCE_PEAK: f32 = 0.003;

/// 毫秒 → 帧。**阈值换算的唯一实现**（默认值与环境变量覆盖共用同一个）。
///
/// 与 `cleanup_f0_inplace` 同一条下限：再短的间隙也算一帧，否则 `max_bridge_frames`
/// 会退化成 0 而把所有短间隙都判成长间隙。`0` 毫秒是合法取值，表示**禁用插值**
/// （等价于"纯 hold"）—— 这是 A/B 时最需要的一档。
fn bridge_frames_for(max_bridge_ms: f64, frame_period_ms: f64) -> usize {
    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.0 {
        frame_period_ms
    } else {
        5.0
    };
    if !(max_bridge_ms.is_finite() && max_bridge_ms > 0.0) {
        return 0;
    }
    ((max_bridge_ms / fp).round() as isize).max(1) as usize
}

/// 导出侧填充未识别帧时用的一组常量。
///
/// 【为什么所有阈值都以**帧**存储】帧周期不是常量（`frame_period_ms` 是工程级
/// 设置）。以毫秒声明、在 [`Self::for_frame_period`] 一次换算，比较时不再做除法。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnvoicedFillPolicy {
    /// 换算所依据的帧周期（毫秒）。毫秒→帧的**唯一**换算点见 [`Self::for_frame_period`]。
    pub frame_period_ms: f64,
    /// 间隙 ≤ 这个长度（帧）走插值，更长走长间隙规则。
    pub max_bridge_frames: usize,
    /// 两端偏移同号且差值 ≤ 这个值（半音）才允许插值。
    pub max_bridge_span_semitones: f64,
    /// 每个 hold 段尾部留多少帧做线性过渡。
    ///
    /// `0` **不是**"没有过渡"：写不出垂直阶跃（简化器的同时间点合并会把同一时刻的
    /// 两点并成一点），所以 `0` 得到的是落在相邻两帧之间的 **5 ms 斜坡**。这个参数
    /// 因此只决定"5 ms 还是 20 ms"，由听感定，不由正确性定。
    pub ramp_frames: usize,
    /// 判定"静音"的峰值门（`1.0` = 0 dBFS）。
    pub silence_peak: f32,
    pub fill: UnvoicedFill,
}

impl UnvoicedFillPolicy {
    /// 按帧周期把毫秒常量换算成帧，其余取默认值。
    pub fn for_frame_period(fill: UnvoicedFill, frame_period_ms: f64) -> Self {
        let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.0 {
            frame_period_ms
        } else {
            5.0
        };
        Self {
            frame_period_ms: fp,
            max_bridge_frames: bridge_frames_for(MAX_BRIDGE_MS, fp),
            max_bridge_span_semitones: MAX_BRIDGE_SPAN_SEMITONES,
            ramp_frames: 0,
            silence_peak: SILENCE_PEAK,
            fill,
        }
    }

    /// 环境变量 A/B 覆盖。
    ///
    /// 【为什么必须有】这四个常量的正确取值只能靠听。有了它，A/B 时不用重新编译、
    /// 不用改配置、不用重启宿主 —— REAPER 里换一段素材就能对比。没有它，每一轮
    /// 听感实验都要走一次完整构建 + 插件重装。
    ///
    /// | 变量 | 字段 |
    /// |---|---|
    /// | `HIFISHIFTER_TAKE_PITCH_UNVOICED` | [`Self::fill`] |
    /// | `HIFISHIFTER_TAKE_PITCH_MAX_BRIDGE_MS` | [`Self::max_bridge_frames`]（换算后） |
    /// | `HIFISHIFTER_TAKE_PITCH_RAMP_FRAMES` | [`Self::ramp_frames`] |
    /// | `HIFISHIFTER_TAKE_PITCH_SILENCE_PEAK` | [`Self::silence_peak`] |
    ///
    /// 解析不了的值**保持原值**（不做部分覆盖、不报错）：一次手误不该让导出行为
    /// 悄悄落到某个"半个默认"的组合上。
    pub fn with_env_overrides(self) -> Self {
        self.with_overrides_from(|name| std::env::var(name).ok())
    }

    /// [`Self::with_env_overrides`] 的实现，取值来源可注入（因此可测）。
    ///
    /// 【为什么不直接读 `std::env`】`set_var` 在多线程测试里是不安全的，而"环境变量
    /// 钩子"恰恰是最容易写完就再没人验证的那类代码。收一个查找闭包，让"解析失败
    /// 保持原值"这件事能被真正测到。
    fn with_overrides_from(mut self, lookup: impl Fn(&str) -> Option<String>) -> Self {
        let number = |name: &str| -> Option<f64> {
            lookup(name)
                .and_then(|raw| raw.trim().parse::<f64>().ok())
                .filter(|value| value.is_finite())
        };

        if let Some(name) = lookup("HIFISHIFTER_TAKE_PITCH_UNVOICED") {
            self.fill = UnvoicedFill::from_name(&name);
        }
        if let Some(ms) = number("HIFISHIFTER_TAKE_PITCH_MAX_BRIDGE_MS") {
            self.max_bridge_frames = bridge_frames_for(ms, self.frame_period_ms);
        }
        if let Some(frames) = number("HIFISHIFTER_TAKE_PITCH_RAMP_FRAMES").filter(|v| *v >= 0.0) {
            self.ramp_frames = frames.round() as usize;
        }
        if let Some(peak) = number("HIFISHIFTER_TAKE_PITCH_SILENCE_PEAK") {
            self.silence_peak = peak.max(0.0) as f32;
        }
        self
    }
}

impl Default for UnvoicedFillPolicy {
    fn default() -> Self {
        Self::for_frame_period(UnvoicedFill::default(), 5.0)
    }
}

/// 导出侧的逐帧偏移里，把**未识别帧**填成邻域的连续值。
///
/// 参数：
/// - `offsets`：来自 `compute_clip_export_pitch_offsets` 的逐帧半音偏移；
/// - `voiced`：哪些帧**至少有一侧定义了音高**。口径是
///   `(pitch_orig > 0) || (pitch_edit > 0)`，与
///   `PitchCurvesSnapshot::is_voiced_at_time` 逐字同义。
///
///   **最容易做错的一步**：不能写 `orig > 0 && edit > 0`。整条没编辑过的有声帧
///   `edit == 0`（用户没碰过 ⇒ 不需要偏移），那样判会把**用户从未编辑过的整段**
///   判成 unvoiced 并填成邻居的值 —— 等于凭空给它们加了移调。也不能用
///   `offset != 0` 反推：用户可以把音高画回原调，也可以把一整段画成 0。
/// - `energy`：与 `offsets` **同栅格**的逐帧峰值电平（`dyn_orig`，`1.0` = 0 dBFS）。
///   `None` = **未知**，按"可闻"处理（保守 hold）。**绝不补 0** —— 补 0 等于把
///   "不知道"判成"确定的静音"，长间隙会走插值路径。
/// - `policy`：见 [`UnvoicedFillPolicy`]。
///
/// 返回与 `offsets` 等长的新表（`voiced` 比它短时，尾部的帧原样保留）。
pub fn fill_unvoiced_offsets(
    offsets: &[f32],
    voiced: &[bool],
    energy: Option<&[f32]>,
    policy: &UnvoicedFillPolicy,
) -> Vec<f32> {
    // 【回归开关必须在第一行】`Zero` 连 `voiced` 都不读 —— "与改动前逐字节一致"
    // 这件事不该需要读代码来判断。没有掩码时同理（= 旧行为）。
    if policy.fill == UnvoicedFill::Zero || voiced.is_empty() {
        return offsets.to_vec();
    }

    let mut out = offsets.to_vec();
    let n = out.len().min(voiced.len());
    let mut i = 0usize;
    while i < n {
        if voiced[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && !voiced[i] {
            i += 1;
        }
        let end = i; // [start, end) 是一段 unvoiced

        let left = if start > 0 {
            Some(out[start - 1])
        } else {
            None
        };
        let right = if end < n { Some(out[end]) } else { None };
        match (left, right) {
            // 整条没有浊音帧：没有可填的来源，保持原值。
            (None, None) => {}
            // 段首 / 段尾：只有一侧有依据，只能 hold 那一侧。
            (Some(value), None) | (None, Some(value)) => hold(&mut out, start, end, value),
            (Some(left), Some(right)) => {
                let len = end - start;
                let short = len <= policy.max_bridge_frames;
                let same_sign = left * right >= 0.0;
                let close = ((right - left) as f64).abs() <= policy.max_bridge_span_semitones;
                if short && same_sign && close {
                    lerp(&mut out, start, end, left, right);
                } else if energy
                    .is_some_and(|levels| run_is_silent(levels, start, end, policy.silence_peak))
                {
                    // 长静音：填什么都不影响听感，但**填 0 会留下一个阶跃**，而且
                    // 那个阶跃正好落在下一句的起音上。插值穿过静音既听不见又消掉
                    // 阶跃，还省断点（一条直线 2 点）。
                    lerp(&mut out, start, end, left, right);
                } else {
                    // 长且可闻：hold 一段，尾部（可选）爬到右侧值。
                    let from = if policy.fill == UnvoicedFill::Dry {
                        0.0
                    } else {
                        left
                    };
                    hold_then_ramp(&mut out, start, end, from, right, policy.ramp_frames);
                }
            }
        }
    }
    out
}

/// 这一段是否**整段都低于静音门**。
///
/// 【非有限值算"可闻"】NaN 电平意味着"不知道"。把"不知道"当静音会让长间隙走插值
/// 路径 —— 而插值正是"凭空造一个中间调"。保守方向是 hold。
/// 【电平比窗口短也算"不知道"】`dyn_orig` 部分命中时返回的是"当前可得的部分"。
fn run_is_silent(energy: &[f32], start: usize, end: usize, silence_peak: f32) -> bool {
    let Some(levels) = energy.get(start..end) else {
        return false;
    };
    !levels.is_empty()
        && levels
            .iter()
            .all(|peak| peak.is_finite() && *peak < silence_peak)
}

/// 端点不动的线性插值。
///
/// 【公式与 `world_vocoder::cleanup_f0_inplace` 逐字同形】两侧补的是同一件事
/// （"这一帧没有音高"），补出来的必须是同一条线，否则"插件里补的和 REAPER 里补的
/// 不是一个形状"。`(k + 1) / (len + 1)` 保证首末两端都不被改写。
fn lerp(out: &mut [f32], start: usize, end: usize, left: f32, right: f32) {
    let len = end - start;
    for k in 0..len {
        let t = (k + 1) as f32 / (len + 1) as f32;
        out[start + k] = left + (right - left) * t;
    }
}

fn hold(out: &mut [f32], start: usize, end: usize, value: f32) {
    out[start..end].fill(value);
}

/// 先 hold `from`，最后 `min(ramp, len)` 帧线性爬到 `to`。
///
/// `to` 是**下一帧 voiced 的实际值**（不是 `from`、也不是插值终点）：这一段结束后
/// 宿主立刻要接到那个值上，斜坡的终点必须与它重合，否则过渡末尾又多一个折角。
fn hold_then_ramp(out: &mut [f32], start: usize, end: usize, from: f32, to: f32, ramp: usize) {
    let len = end - start;
    let ramp = ramp.min(len);
    out[start..end - ramp].fill(from);
    for k in 0..ramp {
        let t = (k + 1) as f32 / ramp as f32;
        out[end - ramp + k] = from + (to - from) * t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// u ↔ 偏移必须严格互逆，且速率取绝对值（倒放的负速率不改变坐标域）。
    #[test]
    fn the_take_time_domain_round_trips() {
        for rate in [1.0, 0.5, 2.0, -1.5] {
            for offset in [0.0, 0.25, 3.5] {
                let u = offset_to_take_u(offset, rate);
                assert!((take_u_to_offset(u, rate) - offset).abs() < 1e-12);
            }
        }
        // 非法速率按 1 处理，而不是产生 NaN/0。
        assert_eq!(offset_to_take_u(2.0, 0.0), 2.0);
        assert_eq!(offset_to_take_u(2.0, f64::NAN), 2.0);
    }

    /// 采样在点之间插值，在区间外保持端点值（不是 0）。
    #[test]
    fn sampling_interpolates_and_holds_the_ends() {
        let points = [EnvelopePoint::new(1.0, 2.0), EnvelopePoint::new(3.0, 6.0)];
        assert_eq!(sample_envelope(&points, 0.5), 2.0, "首点之前 hold");
        assert_eq!(sample_envelope(&points, 2.0), 4.0);
        assert_eq!(sample_envelope(&points, 9.0), 6.0, "末点之后 hold");
        assert_eq!(sample_envelope(&[], 1.0), 0.0, "没有包络 = 不改音高");
    }

    /// 全零曲线不产生任何断点（保持宿主原状，不写空包络）。
    #[test]
    fn a_flat_default_curve_produces_no_points() {
        let offsets = vec![0.0_f32; 64];
        assert!(build_take_pitch_points(&offsets, 5.0, 1.0, 1.0).is_empty());
    }

    /// 平坦非零曲线压成恰好两个端点，且换到 u 域（端点 = 0 与 长度×速率）。
    #[test]
    fn a_flat_non_zero_curve_becomes_two_endpoints_in_take_time() {
        let offsets = vec![2.0_f32; 32];
        let points = build_take_pitch_points(&offsets, 5.0, 1.0, 2.0);
        assert_eq!(points.len(), 2, "{points:?}");
        assert!((points[0].time - 0.0).abs() < 1e-12);
        assert!((points[1].time - 2.0).abs() < 1e-12, "u_end = 1.0s × 2.0");
        assert!((points[0].value - 2.0).abs() < 1e-9);
        assert!((points[1].value - 2.0).abs() < 1e-9);
    }

    /// 斜坡只保留折点：点数远小于帧数，且首末值正确。
    #[test]
    fn a_ramp_is_reduced_to_its_corners() {
        let offsets: Vec<f32> = (0..200).map(|index| index as f32 * 0.01).collect();
        let points = build_take_pitch_points(&offsets, 5.0, 1.0, 1.0);
        assert!(points.len() <= 3, "斜坡不该逐帧写点：{}", points.len());
        assert!((points[0].value - 0.0).abs() < 0.03);
        assert!((points[points.len() - 1].value - 1.99).abs() < 0.03);
    }

    // ─── 断点精简（弦斜率窗口实现） ──────────────────────────────────────

    /// 直接求值的朴素参考实现：与弦斜率窗口版**逐点同判据**，只是每个候选都
    /// 从头复扫中间样本（平方）。它钉住"线性化不改语义"这件事 —— 窗口版与它
    /// 的差异只允许来自浮点舍入，任何构造出的曲线都不得给出不同的断点序列。
    fn simplify_breakpoints_reference(
        samples: &[(f64, f64)],
        range_start: f64,
        range_end: f64,
        default_value: f64,
        tolerance: f64,
    ) -> Vec<(f64, f64)> {
        if samples.is_empty() || range_end <= range_start {
            return Vec::new();
        }
        if samples
            .iter()
            .all(|(_, value)| (value - default_value).abs() <= tolerance)
        {
            return Vec::new();
        }
        if samples.len() == 1 {
            let value = samples[0].1;
            let first = samples[0].0.clamp(range_start, range_end);
            return vec![(first, value), (range_end, value)];
        }
        {
            let (t0, v0) = samples[0];
            let (tn, vn) = samples[samples.len() - 1];
            let span = tn - t0;
            let slope = if span > 1e-12 { (vn - v0) / span } else { 0.0 };
            let mut flat = true;
            let mut linear = true;
            for (t, v) in samples.iter() {
                if (v - v0).abs() > tolerance {
                    flat = false;
                }
                if (v - (v0 + slope * (t - t0))).abs() > tolerance {
                    linear = false;
                }
                if !flat && !linear {
                    break;
                }
            }
            if flat || linear {
                let first_t = t0.max(range_start);
                let last_t = range_end.max(first_t);
                return vec![
                    (first_t, v0 + slope * (first_t - t0)),
                    (last_t, v0 + slope * (last_t - t0)),
                ];
            }
        }
        let value_at = |t: f64| -> f64 {
            let index = samples.partition_point(|sample| sample.0 < t);
            if index == 0 {
                return samples[0].1;
            }
            if index >= samples.len() {
                return samples[samples.len() - 1].1;
            }
            let (t0, v0) = samples[index - 1];
            let (t1, v1) = samples[index];
            if (t1 - t0).abs() < 1e-12 {
                return v0;
            }
            let fraction = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            v0 + (v1 - v0) * fraction
        };
        let line_fits = |a: usize, f: usize| -> bool {
            let (ta, va) = samples[a];
            let (tf, vf) = samples[f];
            let span = tf - ta;
            if span <= 1e-12 {
                return (vf - va).abs() <= tolerance;
            }
            let slope = (vf - va) / span;
            samples[a + 1..f]
                .iter()
                .all(|(t, v)| (v - (va + slope * (t - ta))).abs() <= tolerance)
        };
        let mut out: Vec<(f64, f64)> = Vec::new();
        let push_point = |t: f64, v: f64, out: &mut Vec<(f64, f64)>| {
            if let Some(last) = out.last_mut() {
                if (last.0 - t).abs() <= 1e-9 {
                    last.1 = v;
                    return;
                }
            }
            out.push((t, v));
        };
        let first_t = samples[0].0.max(range_start);
        push_point(first_t, value_at(first_t), &mut out);
        let mut anchor = 0usize;
        loop {
            let mut f = anchor + 1;
            let mut farthest = anchor + 1;
            while f < samples.len() {
                if line_fits(anchor, f) {
                    farthest = f;
                    f += 1;
                } else {
                    break;
                }
            }
            let (t, v) = samples[farthest];
            push_point(t, v, &mut out);
            anchor = farthest;
            if anchor + 1 >= samples.len() {
                break;
            }
        }
        let end_v = value_at(range_end);
        if let Some(last) = out.last_mut() {
            if (last.1 - end_v).abs() <= tolerance {
                last.0 = range_end;
                last.1 = end_v;
            } else {
                push_point(range_end, end_v, &mut out);
            }
        } else {
            push_point(range_end, end_v, &mut out);
        }
        out
    }

    /// 确定性伪随机（无外部依赖）：足够搅动出各种折线形态，且每次运行一致。
    fn noise_sequence(length: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..length)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 11) & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64;
                (unit - 0.5) * 2.0
            })
            .collect()
    }

    /// 窗口版必须与直接求值版给出**相同**的断点序列：随机折线（各种长度/容差）、
    /// 长直线接折点（曾经的平方路径）、重复时间戳与时间倒退（畸形输入）都要一致。
    #[test]
    fn the_slope_window_simplifier_matches_direct_evaluation() {
        let cases: Vec<Vec<(f64, f64)>> = {
            let mut cases = Vec::new();
            // 随机折线：长度 × 种子 × 容差都扫一遍。
            for length in [2usize, 3, 8, 64, 257] {
                for seed in [1u64, 7, 12345, 0xDEADBEEF] {
                    let noise = noise_sequence(length * 2, seed);
                    let samples: Vec<(f64, f64)> = (0..length)
                        .map(|i| {
                            (
                                i as f64 * 0.005,
                                // 大幅值：让相邻点的差远超容差，逼出折点。
                                noise[i * 2] * 8.0,
                            )
                        })
                        .collect();
                    cases.push(samples);
                    let gentle: Vec<(f64, f64)> = (0..length)
                        .map(|i| (i as f64 * 0.005, noise[i * 2 + 1] * 0.02))
                        .collect();
                    cases.push(gentle);
                }
            }
            // 长直线段接折点（一段滑音接颤音的形态）：曾经的平方路径。
            let bent: Vec<(f64, f64)> = (0..4000usize)
                .map(|i| {
                    let t = i as f64 * 0.005;
                    let v = if t < 15.0 {
                        t / 15.0
                    } else {
                        1.0 + (t - 15.0).sin() * 0.5
                    };
                    (t, v)
                })
                .collect();
            cases.push(bent);
            // 重复时间戳（畸形）：同刻不同值、同刻同值各一组。
            cases.push(vec![
                (0.0, 1.0),
                (0.0, 1.0 + 0.02),
                (0.0, 2.0),
                (0.01, 3.0),
                (0.02, 0.5),
            ]);
            cases.push(vec![(0.0, 1.0), (0.0, 1.0), (0.005, 1.0), (0.01, 2.0)]);
            // 时间倒退（畸形）。
            cases.push(vec![(0.02, 1.0), (0.01, 2.0), (0.005, 0.5), (0.03, 1.5)]);
            cases
        };
        for samples in &cases {
            for tolerance in [0.02, 0.005, 1e-4] {
                let windowed = simplify_breakpoints(samples, 0.0, 10.0, 0.0, tolerance);
                let reference = simplify_breakpoints_reference(samples, 0.0, 10.0, 0.0, tolerance);
                assert_eq!(
                    windowed, reference,
                    "容差 {tolerance} 下与直接求值不一致：{samples:?}"
                );
            }
        }
    }

    // ─── 未识别帧填充 ───────────────────────────────────────────────────────

    fn policy(fill: UnvoicedFill, max_bridge_frames: usize) -> UnvoicedFillPolicy {
        UnvoicedFillPolicy {
            frame_period_ms: 5.0,
            max_bridge_frames,
            max_bridge_span_semitones: MAX_BRIDGE_SPAN_SEMITONES,
            ramp_frames: 0,
            silence_peak: SILENCE_PEAK,
            fill,
        }
    }

    /// 把"变量名 → 值"对当环境变量用（`with_env_overrides` 的注入版）。
    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    /// `Zero` 是回归开关：输出必须与输入逐元素相同（含 0）。
    #[test]
    fn zero_policy_reproduces_the_input_verbatim() {
        let offsets = [12.0f32, 0.0, 0.0, 7.0];
        let voiced = [true, false, false, true];
        let energy = [1.0f32, 1.0, 1.0, 1.0];
        let out = fill_unvoiced_offsets(
            &offsets,
            &voiced,
            Some(&energy),
            &policy(UnvoicedFill::Zero, 1),
        );
        assert_eq!(out, offsets.to_vec());
    }

    /// 全 voiced ⇒ 恒等（填充是恒等变换，不是"平滑"）。
    #[test]
    fn an_all_voiced_curve_is_untouched() {
        let offsets = [0.0f32, 2.0, 0.0];
        let voiced = [true; 3];
        assert_eq!(
            fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6)),
            offsets.to_vec()
        );
    }

    /// 全 unvoiced ⇒ 恒等（没有可填的来源，不能凭空造值）。
    #[test]
    fn an_all_unvoiced_curve_is_untouched() {
        let offsets = [0.0f32, 1.0, 2.0];
        let voiced = [false; 3];
        assert_eq!(
            fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6)),
            offsets.to_vec()
        );
    }

    /// 段首：只有右侧有依据 ⇒ hold 后值。
    #[test]
    fn a_leading_unvoiced_run_holds_the_first_voiced_value() {
        let offsets = [0.0f32, 0.0, 0.0, 5.0, 6.0];
        let voiced = [false, false, false, true, true];
        assert_eq!(
            fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6)),
            vec![5.0, 5.0, 5.0, 5.0, 6.0]
        );
    }

    /// 段尾：只有左侧有依据 ⇒ hold 前值。
    #[test]
    fn a_trailing_unvoiced_run_holds_the_last_voiced_value() {
        let offsets = [5.0f32, 6.0, 0.0, 0.0];
        let voiced = [true, true, false, false];
        assert_eq!(
            fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6)),
            vec![5.0, 6.0, 6.0, 6.0]
        );
    }

    /// 间隙长度**正好等于**阈值 ⇒ 插值（阈值是"≤"，不是"<"）。
    #[test]
    fn a_gap_exactly_at_the_threshold_is_interpolated() {
        let offsets = [2.0f32, 0.0, 0.0, 3.0];
        let voiced = [true, false, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 2));
        assert!((out[1] - 7.0 / 3.0).abs() < 1e-6, "{out:?}");
        assert!((out[2] - 8.0 / 3.0).abs() < 1e-6, "{out:?}");
    }

    /// 间隙比阈值长一帧 ⇒ 走长间隙规则（hold）。
    #[test]
    fn a_gap_one_frame_past_the_threshold_holds() {
        let offsets = [2.0f32, 0.0, 0.0, 0.0, 3.0];
        let voiced = [true, false, false, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 2));
        assert_eq!(&out[1..4], &[2.0, 2.0, 2.0]);
    }

    /// 插值不得改写两端的有声帧，且中间严格单调。
    #[test]
    fn interpolation_never_touches_the_bounding_frames() {
        let offsets = [2.0f32, 0.0, 0.0, 0.0, 3.0];
        let voiced = [true, false, false, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 4));
        assert_eq!(out[0], 2.0, "左端点不动");
        assert_eq!(out[4], 3.0, "右端点不动");
        assert!(out[1] < out[2] && out[2] < out[3], "{out:?}");
        assert!(out[1] > 2.0 && out[3] < 3.0, "严格夹在两端之间：{out:?}");
    }

    /// 两端异号 ⇒ hold，绝不造出穿过 0 的中间值。
    #[test]
    fn a_sign_flip_gap_holds_instead_of_crossing_zero() {
        let offsets = [12.0f32, 0.0, 0.0, -5.0];
        let voiced = [true, false, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6));
        assert_eq!(&out[1..3], &[12.0, 12.0], "{out:?}");
    }

    /// 差值超过跨度上限 ⇒ hold（哪怕间隙很短）。
    #[test]
    fn a_large_span_gap_holds() {
        let offsets = [12.0f32, 0.0, 0.0, 7.0];
        let voiced = [true, false, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6));
        assert_eq!(&out[1..3], &[12.0, 12.0], "{out:?}");
    }

    /// 长**静音**间隙插值：听不见，但消掉了落在下一句起音上的阶跃。
    #[test]
    fn a_silent_long_gap_is_interpolated() {
        let offsets = [2.0f32, 0.0, 0.0, 0.0, 3.0];
        let voiced = [true, false, false, false, true];
        let energy = [0.5f32, 0.0001, 0.0002, 0.0003, 0.5];
        let out = fill_unvoiced_offsets(
            &offsets,
            &voiced,
            Some(&energy),
            &policy(UnvoicedFill::Bridge, 1),
        );
        assert!((out[1] - 2.25).abs() < 1e-6, "{out:?}");
        assert!((out[2] - 2.5).abs() < 1e-6, "{out:?}");
        assert!((out[3] - 2.75).abs() < 1e-6, "{out:?}");
    }

    /// 长**可闻**间隙 hold：插值会让辅音被移调到中间调。
    #[test]
    fn a_loud_long_gap_holds() {
        let offsets = [2.0f32, 0.0, 0.0, 0.0, 3.0];
        let voiced = [true, false, false, false, true];
        let energy = [0.5f32, 0.4, 0.0, 0.0, 0.5];
        let out = fill_unvoiced_offsets(
            &offsets,
            &voiced,
            Some(&energy),
            &policy(UnvoicedFill::Bridge, 1),
        );
        assert_eq!(&out[1..4], &[2.0, 2.0, 2.0], "{out:?}");
    }

    /// **缺失的能量 ≠ 静音**：`None` 必须走 hold，不能走插值。
    ///
    /// 【为什么这条最容易写错】"没有电平数据"看起来像"电平是 0"，而 0 低于静音门
    /// ⇒ 判成静音 ⇒ 插值。于是"未知"被悄悄翻译成"确定的静音"，一条本该 hold 的
    /// 长间隙开始凭空造中间调。
    #[test]
    fn missing_energy_is_not_treated_as_silence() {
        let offsets = [2.0f32, 0.0, 0.0, 0.0, 3.0];
        let voiced = [true, false, false, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 1));
        assert_eq!(&out[1..4], &[2.0, 2.0, 2.0], "{out:?}");
        // 电平比窗口短同样是"不知道"。
        let short = [0.0f32, 0.0];
        let out = fill_unvoiced_offsets(
            &offsets,
            &voiced,
            Some(&short),
            &policy(UnvoicedFill::Bridge, 1),
        );
        assert_eq!(
            &out[1..4],
            &[2.0, 2.0, 2.0],
            "短电平表不得被当成静音：{out:?}"
        );
    }

    /// 用户主动画成 0 的**有声**帧必须原样保留 —— 那是用户的意思，不是"没有音高"。
    #[test]
    fn a_user_zero_frame_is_never_filled() {
        let offsets = [3.0f32, 0.0, 0.0, 2.0];
        let voiced = [true, true, false, true];
        let out = fill_unvoiced_offsets(&offsets, &voiced, None, &policy(UnvoicedFill::Bridge, 6));
        assert_eq!(out[1], 0.0, "用户画的 0 被改写：{out:?}");
        assert!((out[2] - 1.0).abs() < 1e-6, "{out:?}");
        assert_eq!(out[3], 2.0);
    }

    /// `Dry`：长可闻间隙 hold **0**，与渲染端"无 F0 有能量 ⇒ 干声"对齐。
    #[test]
    fn dry_policy_holds_zero_inside_a_long_audible_gap() {
        let offsets = [3.0f32, 0.0, 0.0, 0.0, 4.0];
        let voiced = [true, false, false, false, true];
        let energy = [0.5f32, 0.4, 0.4, 0.4, 0.5];
        let out = fill_unvoiced_offsets(
            &offsets,
            &voiced,
            Some(&energy),
            &policy(UnvoicedFill::Dry, 1),
        );
        assert_eq!(&out[1..4], &[0.0, 0.0, 0.0], "{out:?}");

        // 带斜坡时尾部爬到右侧值（`to` 是下一帧 voiced 的实际值）。
        let mut ramped = policy(UnvoicedFill::Dry, 1);
        ramped.ramp_frames = 2;
        let out = fill_unvoiced_offsets(&offsets, &voiced, Some(&energy), &ramped);
        assert_eq!(out[1], 0.0, "{out:?}");
        assert!((out[2] - 2.0).abs() < 1e-6, "{out:?}");
        assert!((out[3] - 4.0).abs() < 1e-6, "{out:?}");
    }

    /// 斜坡长度只改"过渡多长"，不改断点预算（线性段仍被压成两端点）。
    #[test]
    fn ramp_frames_does_not_change_the_point_count() {
        // frames 0..10 voiced 0；10..30 unvoiced 且可闻（hold 前值 0）；
        // 30..40 voiced 12 ⇒ 一个阶跃。ramp=4 把它磨成 4 帧线性。
        let mut offsets = vec![0.0f32; 40];
        for slot in offsets.iter_mut().skip(30) {
            *slot = 12.0;
        }
        let mut voiced = vec![true; 40];
        for slot in voiced.iter_mut().take(30).skip(10) {
            *slot = false;
        }
        let energy = vec![0.4f32; 40];
        let count = |ramp: usize| {
            let mut tuned = policy(UnvoicedFill::Bridge, 1);
            tuned.ramp_frames = ramp;
            let filled = fill_unvoiced_offsets(&offsets, &voiced, Some(&energy), &tuned);
            build_take_pitch_points(&filled, 5.0, 0.2, 1.0).len()
        };
        assert_eq!(count(0), count(4), "线性斜坡不该增加断点");
    }

    /// 四个常量的取值只能靠听 ⇒ 环境变量 A/B 钩子必须真的能改到它们。
    #[test]
    fn the_environment_can_switch_the_policy_and_its_thresholds() {
        let base = UnvoicedFillPolicy::for_frame_period(UnvoicedFill::Bridge, 5.0);
        assert_eq!(base.max_bridge_frames, 6, "30 ms ÷ 5 ms = 6 帧");
        let tuned = base.with_overrides_from(env(&[
            ("HIFISHIFTER_TAKE_PITCH_UNVOICED", "dry"),
            ("HIFISHIFTER_TAKE_PITCH_MAX_BRIDGE_MS", "120"),
            ("HIFISHIFTER_TAKE_PITCH_RAMP_FRAMES", "4"),
            ("HIFISHIFTER_TAKE_PITCH_SILENCE_PEAK", "0.01"),
        ]));
        assert_eq!(tuned.fill, UnvoicedFill::Dry);
        assert_eq!(tuned.max_bridge_frames, 24, "120 ms ÷ 5 ms = 24 帧");
        assert_eq!(tuned.ramp_frames, 4);
        assert!((tuned.silence_peak - 0.01).abs() < 1e-9);
        // 0 毫秒是合法值：禁用插值（等价于纯 hold）。
        let off = base.with_overrides_from(env(&[("HIFISHIFTER_TAKE_PITCH_MAX_BRIDGE_MS", "0")]));
        assert_eq!(off.max_bridge_frames, 0);
    }

    /// 解析不了的值**保持原值**，不做部分覆盖。
    #[test]
    fn an_unparsable_override_keeps_the_default() {
        let base = UnvoicedFillPolicy::for_frame_period(UnvoicedFill::Bridge, 5.0);
        let tuned = base.with_overrides_from(env(&[
            ("HIFISHIFTER_TAKE_PITCH_MAX_BRIDGE_MS", "not-a-number"),
            ("HIFISHIFTER_TAKE_PITCH_RAMP_FRAMES", "-3"),
            ("HIFISHIFTER_TAKE_PITCH_SILENCE_PEAK", "NaN"),
            ("HIFISHIFTER_TAKE_PITCH_UNVOICED", "smooth-ish"),
        ]));
        assert_eq!(tuned.max_bridge_frames, base.max_bridge_frames);
        assert_eq!(tuned.ramp_frames, base.ramp_frames);
        assert_eq!(tuned.silence_peak, base.silence_peak);
        assert_eq!(tuned.fill, UnvoicedFill::Bridge, "非法档位回退默认");
    }

    /// 配置里存的是名字，两个方向必须互逆（否则"存 dry、读回 bridge"）。
    #[test]
    fn the_policy_names_round_trip_through_the_loose_parser() {
        for fill in [
            UnvoicedFill::Zero,
            UnvoicedFill::Hold,
            UnvoicedFill::Bridge,
            UnvoicedFill::Dry,
        ] {
            assert_eq!(UnvoicedFill::from_name(fill.as_name()), fill);
        }
        assert_eq!(UnvoicedFill::from_name("  DRY "), UnvoicedFill::Dry);
        assert_eq!(UnvoicedFill::from_name(""), UnvoicedFill::Bridge);
        assert_eq!(UnvoicedFill::from_name("bogus"), UnvoicedFill::Bridge);
    }
}
