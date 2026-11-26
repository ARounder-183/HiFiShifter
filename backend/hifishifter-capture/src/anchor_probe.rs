//! 用素材本身回答「这一块在工程哪里」。
//!
//! ## 这个模块存在的理由
//!
//! 捕获通路有一个别的插件没有的奢侈品：**干声素材就在我们手上**。
//!
//!   * 录制期录下的就是宿主喂进来的干声（`CapturePcm`）；
//!   * 预览与 Render As 喂进来的**还是同一份干声**（同一条轨、同一段内容，只是位置不同）。
//!
//! 于是「这一块在工程哪里」是一个**可测的量**：把喂进来的这一块与素材逐点比一次，
//! 对得上的那个位置就是答案。判据是归一化互相关 —— 与 `scripts/audit/render-repeat-check.py`
//! 的 `align_to_source` 做的是同一件事，只不过它在离线脚本里比对产物，这里在音频线程上
//! 比对输入。
//!
//! ## 为什么必须在这里而不是在 `project_map` 里
//!
//! `project_map` 回答的是「喂入域第 N 帧按**位置规则**落在工程哪里」—— 它只有一个
//! 读数可用（宿主报的位置），而那个读数在 Render As 里是上一次的播放头。本模块回答的是
//! 「喂入域第 N 帧按**内容**落在工程哪里」。两者都产出同一个类型（`FeedAnchor`），
//! 但依据完全不同，所以必须是两个函数、两份日志。
//!
//! ## 坐标只说一套
//!
//! `offset` **就是工程帧**：喂入流的第 0 帧对应的工程帧。`TapeEnvelope::origin` 已经是
//! `PlaybackSnapshot::origin_sample`，所以「素材域」这个概念在本模块里不存在。
//!
//! ## 音频线程纪律
//!
//! [`locate`] 零分配、零锁、零日志：它只读一个 `Arc` 里的包络与样本平面，用定长
//! 数组攒中间结果，并把总乘加次数卡在 [`PROBE_OPS`] 之内。超预算就诚实地回答
//! `Undecided(Busy)`，由调用方退回位置规则。

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

/// 包络每多少帧一格。
///
/// 【为什么是 64】它必须**严格小于**宿主的最小块长（VEGAS 实测 848/960/1200），否则
/// 一个块折不出两格来，包络根本不存在 —— 第七轮 §6 就是用一个 1024 的 hop 把探测
/// 写成了"永远不运行"。64 帧 = 1.33 ms @ 48 kHz，够粗搜，也够给样本级精修留一个
/// 确定的窗口。
pub const HOP_FRAMES: u32 = 64;

/// 少于这么多帧不判。
///
/// 【为什么是 512】宿主块长 848/960/1200 都在这之上，而 512 帧仍然给得出 8 格包络
/// （8 格足以区分"对上了"与"没对上"）。取 4096（第七轮的值）会让条件永不成立。
pub const PROBE_MIN_FRAMES: usize = 512;

/// 块内最大格 RMS 低于它就不判（−80 dBFS）。
///
/// 【为什么静音必须单独一档】静音块在任何锚点下输出都是静音 —— 它不构成"这次机会
/// 用掉了"的证据。见 `ProbeReason::Silent`。
pub const PROBE_MIN_RMS: f32 = 1e-4;

/// 主候选的归一化相关下限。
pub const PROBE_MIN_SCORE: f32 = 0.80;

/// 主候选与次优的最小差。
pub const PROBE_MIN_MARGIN: f32 = 0.15;

/// 全局搜索的包络格数上限（≈5.5 分钟 @ hop=64）。超过就只测两个假设。
pub const PROBE_MAX_CELLS: usize = 262_144;

/// 一次定位的乘加预算（≈0.5–1 ms @ 3 GHz）。
pub const PROBE_OPS: u64 = 4_000_000;

/// 一个块最多折成多少格（64 格 × 64 帧 = 4096 帧）。定长，因为音频线程不许分配。
const MAX_BLOCK_CELLS: usize = 64;

/// 一个块参与比较的最大帧数（= `MAX_BLOCK_CELLS × HOP_FRAMES`）。
const MAX_BLOCK_FRAMES: usize = MAX_BLOCK_CELLS * HOP_FRAMES as usize;

/// 一次定位最多复核多少个候选位置。
const MAX_CANDIDATES: usize = 64;

/// 粗搜的跳步（格）。每格本身已经是 64 帧的均值，跳 4 格 = 256 帧。
const COARSE_STRIDE: usize = 4;

/// 粗搜之后精补几个候选。
const REFINE_TOP: usize = 4;

/// 样本级精修的窗口上限（帧）。
const REFINE_WINDOW: usize = 960;

/// 精修需要的**样本视图**：包络与样本必须来自同一代素材。
///
/// 【为什么是一个 trait 而不是具体类型】包络住在 `hifishifter-capture`，而样本住在格式
/// 外壳的 `PlaybackSnapshot`（它还要管读者计数与预算）。两者之间唯一的共同语言是
/// 「起点 + 两个平面 + 覆盖表」这三件事。
pub trait TapeSamples: Send + Sync {
    /// 第 0 个样本对应的工程帧。
    fn origin(&self) -> i64;
    fn left(&self) -> &[f32];
    /// **空 = 与左平面相同**（单声道素材的存储形态）。
    fn right(&self) -> &[f32];
    /// 真的属于我们的工程帧区间（升序、不重叠）。空 = 整份都算。
    fn covered(&self) -> &[Range<i64>];
}

/// 素材的**粗包络索引**：每 `hop` 帧一格，值 = 该格的 RMS（双声道下混）。
///
/// 【为什么是 RMS 而不是谱】包络对增益、相位、轻微的重采样差异都不敏感，而对「内容
/// 不同」极其敏感 —— 这正是「位置」需要的分辨率。谱会多出 20 倍的计算量，换来的只是
/// 「同一段内容被均衡过」这种根本不存在的场景。
pub struct TapeEnvelope {
    pub hop: u32,
    /// 第 0 格的**工程帧**。
    pub origin: i64,
    /// 磁带总帧数。
    pub frames: i64,
    /// `len = ceil(frames / hop)`。
    pub values: Box<[f32]>,
    /// 每一格是否**整格**都属于我们（与 `values` 同长）。
    ///
    /// 【为什么在建立期就折成布尔】洞是"没有素材"的地方，而"没有素材"绝不能被当成
    /// "内容是静音" —— 那会让探测在洞里给出一个自洽但错误的偏移。把它预计算成一位，
    /// 音频线程上就只剩一次数组索引。
    pub live: Box<[bool]>,
}

impl TapeEnvelope {
    /// 建立包络（**非音频线程**：一次 O(n) 扫描，33 s 素材 ≈ 1 ms）。
    pub fn build(samples: &dyn TapeSamples) -> Option<TapeEnvelope> {
        let origin = samples.origin();
        let left = samples.left();
        let right = samples.right();
        let frames = if right.is_empty() {
            left.len()
        } else {
            left.len().min(right.len())
        };
        if frames == 0 {
            return None;
        }
        let hop = HOP_FRAMES as usize;
        let cells = frames.div_ceil(hop);
        let covered = samples.covered();
        let mut values = vec![0.0_f32; cells];
        let mut live = vec![false; cells];
        for index in 0..cells {
            let lo = index * hop;
            let hi = (lo + hop).min(frames);
            let mut sum = 0.0_f64;
            for sample in lo..hi {
                let l = f64::from(left[sample]);
                let r = if right.is_empty() {
                    l
                } else {
                    f64::from(right[sample])
                };
                let mono = (l + r) * 0.5;
                sum += mono * mono;
            }
            values[index] = (sum / (hi - lo).max(1) as f64).sqrt() as f32;
            live[index] = covered.is_empty()
                || (lo..hi).all(|sample| {
                    let frame = origin.saturating_add(sample as i64);
                    covered.iter().any(|range| range.contains(&frame))
                });
        }
        Some(TapeEnvelope {
            hop: HOP_FRAMES,
            origin,
            frames: frames as i64,
            values: values.into_boxed_slice(),
            live: live.into_boxed_slice(),
        })
    }

    /// 工程帧 → 格号（向下取整，负数也正确）。
    fn cell_of(&self, frame: i64) -> i64 {
        (frame - self.origin).div_euclid(i64::from(self.hop))
    }
}

/// 一次喂入流的**内容定位器**：包络 + 与它同代的那份样本。
///
/// 【为什么两者必须同代】包络是从样本算出来的。分两次发布（先样本后包络，或反过来）
/// 就会出现「包络说 A、样本说 B」的窗口，而那窗口里算出的偏移是**自洽但错误**的 ——
/// 比没有偏移糟得多。所以它们装在同一个 `Arc` 里，一次换指针。
pub struct TapeProbe {
    pub envelope: Arc<TapeEnvelope>,
    pub samples: Arc<dyn TapeSamples>,
}

impl TapeProbe {
    pub fn new(samples: Arc<dyn TapeSamples>) -> Option<TapeProbe> {
        let envelope = Arc::new(TapeEnvelope::build(samples.as_ref())?);
        Some(TapeProbe { envelope, samples })
    }
}

/// 一个喂入块 + 它在喂入域的位置。
#[derive(Clone, Copy, Debug)]
pub struct ProbeBlock<'a> {
    pub left: &'a [f32],
    /// 空 = 与左平面相同。
    pub right: &'a [f32],
    /// 本块**之前**这条喂入流已经喂进来的帧数。
    pub frames_before: i64,
}

impl ProbeBlock<'_> {
    fn frames(&self) -> usize {
        if self.right.is_empty() {
            self.left.len()
        } else {
            self.left.len().min(self.right.len())
        }
    }

    /// 双声道下混的第 `index` 个样本。
    fn mono(&self, index: usize) -> f32 {
        let left = self.left[index];
        let right = if self.right.is_empty() {
            left
        } else {
            self.right[index]
        };
        (left + right) * 0.5
    }
}

/// 一次定位的结论。**没有「猜」这一档**：要么 `Content`，要么带原因的 `Undecided`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeVerdict {
    pub outcome: ProbeOutcome,
    /// 喂入流第 0 帧对应的**工程帧**（只在 `Content` 时有意义）。
    pub offset: i64,
    /// 主候选的归一化相关 [−1, 1]。
    pub score: f32,
    /// 次优候选的归一化相关。
    pub second: f32,
    /// 次优候选给出的 `offset`（同域，便于直接比较两个答案差多远）。
    pub second_offset: i64,
    /// 参与比较的帧数。
    pub frames: usize,
}

impl ProbeVerdict {
    fn undecided(reason: ProbeReason, frames: usize) -> ProbeVerdict {
        ProbeVerdict {
            outcome: ProbeOutcome::Undecided(reason),
            offset: 0,
            score: 0.0,
            second: 0.0,
            second_offset: 0,
            frames,
        }
    }

    /// 这一次为什么没按内容判（`None` = 判了）。
    pub fn reason(&self) -> Option<ProbeReason> {
        match self.outcome {
            ProbeOutcome::Undecided(reason) => Some(reason),
            ProbeOutcome::Content => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeOutcome {
    Content,
    Undecided(ProbeReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeReason {
    /// 这一块没有信号。**不消耗**这次机会 —— 静音块在任何锚点下输出都是静音。
    Silent,
    /// 块太短，折不出足够的格。
    Short,
    /// 素材在这一段没有覆盖（洞），或者比一个块还短。
    NoMaterial,
    /// 对上了，但对上了不止一处（周期性素材）。
    Ambiguous,
    /// 这一块超了乘加预算。
    Busy,
}

impl ProbeReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Silent => "silent",
            Self::Short => "short",
            Self::NoMaterial => "no-material",
            Self::Ambiguous => "ambiguous",
            Self::Busy => "busy",
        }
    }
}

/// `HFS_CAPTURE_PROBE=off`：一键退回「只按位置规则锚定」（第七轮行为）。
pub fn probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("HFS_CAPTURE_PROBE").ok().as_deref().map(str::trim),
            Some("off") | Some("0") | Some("false")
        )
    })
}

/// 主候选下限（`HFS_CAPTURE_PROBE_MIN` 可覆盖，用于现场标定）。
pub fn probe_min_score() -> f32 {
    static MIN: OnceLock<f32> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("HFS_CAPTURE_PROBE_MIN")
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
            .unwrap_or(PROBE_MIN_SCORE)
    })
}

/// 把这一块与素材逐点比一次，回答「喂入流第 0 帧在工程哪里」。
///
/// `hypotheses` 是**已经知道的两个答案**（片段起点、宿主报的位置）。它们先被算一遍：
/// 这两种形状覆盖了现场的全部历史用例，而每一个只需要一次 O(block) 的比较。
/// **先验证已经知道的两个答案，再去找第三个。**
pub fn locate(
    block: &ProbeBlock<'_>,
    envelope: &TapeEnvelope,
    tape_left: &[f32],
    tape_right: &[f32],
    hypotheses: &[i64],
) -> ProbeVerdict {
    let hop = envelope.hop as usize;
    let available = block.frames().min(MAX_BLOCK_CELLS * hop);
    if available < PROBE_MIN_FRAMES {
        return ProbeVerdict::undecided(ProbeReason::Short, available);
    }
    let cells = available / hop;
    let mut block_cells = [0.0_f32; MAX_BLOCK_CELLS];
    let mut peak = 0.0_f32;
    for index in 0..cells {
        let mut sum = 0.0_f64;
        for offset in 0..hop {
            let value = f64::from(block.mono(index * hop + offset));
            sum += value * value;
        }
        let rms = (sum / hop as f64).sqrt() as f32;
        block_cells[index] = rms;
        peak = peak.max(rms);
    }
    if !(peak >= PROBE_MIN_RMS) {
        return ProbeVerdict::undecided(ProbeReason::Silent, cells * hop);
    }
    if envelope.values.len() < cells {
        return ProbeVerdict::undecided(ProbeReason::NoMaterial, cells * hop);
    }

    let window = cells * hop;
    let mut ops = 0_u64;
    // ① 候选格：**已经知道的两个答案**先测，再让有界粗搜补上它找到的前几名。
    //
    // 【为什么先测假设】这两种形状覆盖了现场的全部历史用例（§1.4 的预览、§1.3 的
    // 第 1 次渲染），而它们各自只需要一次比较。**先验证已经知道的两个答案，再去找
    // 第三个。**
    let mut candidates = Candidates::new();
    for &hypothesis in hypotheses {
        let frame = hypothesis.saturating_add(block.frames_before);
        let cell = envelope.cell_of(frame);
        for delta in -1_i64..=1 {
            let candidate = cell + delta;
            if candidate < 0 {
                continue;
            }
            let candidate = candidate as usize;
            if candidate + cells <= envelope.values.len() {
                candidates.push(candidate);
            }
        }
    }
    if envelope.values.len() <= PROBE_MAX_CELLS {
        let mut top = [(f32::NEG_INFINITY, usize::MAX); REFINE_TOP];
        let limit = envelope.values.len() - cells;
        let mut cell = 0_usize;
        while cell <= limit {
            if ops > PROBE_OPS {
                return ProbeVerdict::undecided(ProbeReason::Busy, window);
            }
            ops = ops.saturating_add(cells as u64);
            if let Some(score) = score_at(envelope, &block_cells, cells, cell) {
                insert_top(&mut top, score, cell);
            }
            cell += COARSE_STRIDE;
        }
        // 粗搜只负责**提名**：在得分最高的几格附近按 1 格补齐，全部交给样本级复核。
        for (score, centre) in top {
            if centre == usize::MAX || !score.is_finite() {
                continue;
            }
            let lo = centre.saturating_sub(COARSE_STRIDE);
            let hi = (centre + COARSE_STRIDE).min(limit);
            for cell in lo..=hi {
                candidates.push(cell);
            }
        }
    }

    // ② **样本级复核**才是判据。
    //
    // 【为什么不能拿格级得分当判据】一个 960 帧的块只折得出 15 格 RMS，而 15 个随机
    // 数的相关系数标准差约 0.26 —— 在几千个候选里，0.80 这种「看起来很高」的分数
    // 几乎必然出现一次。实测过一次：素材里挖掉 1 s 的洞、块正好取自洞内，格级判据
    // 给出 0.85 并把它当成了结论。样本级的 960 个点把标准差压到 0.03，同一个位置
    // 只有 0.05。**格级提名，样本级判决。**
    let mut ranking = Ranking::new(cells);
    for index in 0..candidates.len {
        if ops > PROBE_OPS {
            return ProbeVerdict::undecided(ProbeReason::Busy, window);
        }
        ops = ops.saturating_add(window as u64);
        let cell = candidates.cells[index];
        if let Some(score) = sample_score(block, envelope, tape_left, tape_right, cell, window) {
            ranking.push(score, cell);
        }
    }

    let second_offset = ranking
        .second
        .map_or(0, |(_, cell)| {
            envelope
                .origin
                .saturating_add(cell as i64 * hop as i64)
        })
        .saturating_sub(block.frames_before);
    let Some((best_score, best_cell)) = ranking.best else {
        return ProbeVerdict::undecided(ProbeReason::NoMaterial, window);
    };
    let second_score = ranking.second.map_or(0.0, |(score, _)| score.max(0.0));
    // 【未决也要报出「最像的那一处」】`verdict=undecided` 那一行必须带得动现场标定：
    // 没有 offset 与分数，阈值就只能靠猜。
    let best_offset = envelope
        .origin
        .saturating_add(best_cell as i64 * hop as i64)
        .saturating_sub(block.frames_before);
    // ③ 「对不上任何一处」与「对上好几处」是两个不同的结论，必须分开报：
    //    前者说明素材在这一段没有覆盖（洞 / 换了媒体），后者说明素材是周期性的。
    if best_score < probe_min_score() {
        return ProbeVerdict {
            outcome: ProbeOutcome::Undecided(ProbeReason::NoMaterial),
            offset: best_offset,
            score: best_score,
            second: second_score,
            second_offset,
            frames: window,
        };
    }
    if best_score - second_score < PROBE_MIN_MARGIN {
        return ProbeVerdict {
            outcome: ProbeOutcome::Undecided(ProbeReason::Ambiguous),
            offset: best_offset,
            score: best_score,
            second: second_score,
            second_offset,
            frames: window,
        };
    }

    // ④ 样本级精修：把格级答案（±64 帧）压到样本精度。
    let coarse = envelope
        .origin
        .saturating_add(best_cell as i64 * hop as i64);
    let refined = refine(
        block,
        envelope,
        tape_left,
        tape_right,
        coarse,
        window.min(REFINE_WINDOW),
        hop,
        &mut ops,
    );
    ProbeVerdict {
        outcome: ProbeOutcome::Content,
        offset: refined.saturating_sub(block.frames_before),
        score: best_score,
        second: second_score,
        second_offset,
        frames: window,
    }
}

/// 候选格（定长，音频线程不分配）。
struct Candidates {
    cells: [usize; MAX_CANDIDATES],
    len: usize,
}

impl Candidates {
    fn new() -> Candidates {
        Candidates {
            cells: [usize::MAX; MAX_CANDIDATES],
            len: 0,
        }
    }

    fn push(&mut self, cell: usize) {
        if self.len >= MAX_CANDIDATES {
            return;
        }
        // 【只去重**同一个**格】±1 格是**不同**的候选：样本级评分对 64 帧的位移极其
        // 敏感（相关性直接掉到 0），所以「相邻格反正一样」这个假设在这里是错的。
        if self.cells[..self.len].iter().any(|known| *known == cell) {
            return;
        }
        self.cells[self.len] = cell;
        self.len += 1;
    }
}

/// 一个候选位置的**样本级**归一化相关；越界或落在洞里 ⇒ `None`。
fn sample_score(
    block: &ProbeBlock<'_>,
    envelope: &TapeEnvelope,
    tape_left: &[f32],
    tape_right: &[f32],
    cell: usize,
    window: usize,
) -> Option<f32> {
    let end = cell.checked_add(window.div_ceil(envelope.hop as usize))?;
    if end > envelope.values.len() {
        return None;
    }
    let first = cell.checked_mul(envelope.hop as usize)?;
    let last = first.checked_add(window)?;
    if last > tape_left.len() {
        return None;
    }
    if !envelope.live[cell..end].iter().all(|live| *live) {
        return None;
    }
    let mut block_mono = [0.0_f32; MAX_BLOCK_FRAMES];
    let mut tape_mono = [0.0_f32; MAX_BLOCK_FRAMES];
    for index in 0..window {
        block_mono[index] = block.mono(index);
        tape_mono[index] = if tape_right.is_empty() {
            tape_left[first + index]
        } else {
            (tape_left[first + index] + tape_right[first + index]) * 0.5
        };
    }
    Some(normalized(&block_mono[..window], &tape_mono[..window]))
}
/// 得分排名：主候选与次优候选**必须至少相隔一个块长**。
///
/// 【为什么】相邻格给出的是同一段内容的邻居 —— 它们是同一个答案的不同取整，不是
/// 「第二处也像」。把它们算成次优会让 margin 恒为 0，于是所有素材都判 Ambiguous。
struct Ranking {
    best: Option<(f32, usize)>,
    second: Option<(f32, usize)>,
    separation: usize,
}

impl Ranking {
    fn new(separation: usize) -> Ranking {
        Ranking {
            best: None,
            second: None,
            separation,
        }
    }

    fn push(&mut self, score: f32, cell: usize) {
        if let Some((best_score, best_cell)) = self.best {
            if cell.abs_diff(best_cell) < self.separation {
                if score > best_score {
                    self.best = Some((score, cell));
                }
                return;
            }
        }
        match self.best {
            None => self.best = Some((score, cell)),
            Some((best_score, _)) if score > best_score => {
                self.second = self.best;
                self.best = Some((score, cell));
            }
            Some(_) => {
                if self.second.is_none_or(|(second, _)| score > second) {
                    self.second = Some((score, cell));
                }
            }
        }
    }
}

/// 维护粗搜的前 `REFINE_TOP` 名（降序，定长，不分配）。
fn insert_top(top: &mut [(f32, usize); REFINE_TOP], score: f32, cell: usize) {
    for index in 0..top.len() {
        if score > top[index].0 {
            for shift in (index + 1..top.len()).rev() {
                top[shift] = top[shift - 1];
            }
            top[index] = (score, cell);
            return;
        }
    }
}

/// 一格候选的归一化相关；越界或落在洞里 ⇒ `None`。
fn score_at(envelope: &TapeEnvelope, cells: &[f32], count: usize, cell: usize) -> Option<f32> {
    let end = cell.checked_add(count)?;
    if end > envelope.values.len() {
        return None;
    }
    if !envelope.live[cell..end].iter().all(|live| *live) {
        return None;
    }
    Some(normalized(cells, &envelope.values[cell..end]))
}

/// 归一化互相关。
///
/// 【为什么减均值】素材与输入之间可能差一个直流偏置（不同宿主的输入链路），而直流
/// 偏置在 RMS 包络上是一个常数 —— 不减均值的话它会把所有候选的得分一起抬高，margin
/// 随之塌掉。
fn normalized(left: &[f32], right: &[f32]) -> f32 {
    let count = left.len().min(right.len());
    if count == 0 {
        return 0.0;
    }
    let mean_left = left[..count].iter().sum::<f32>() / count as f32;
    let mean_right = right[..count].iter().sum::<f32>() / count as f32;
    let mut dot = 0.0_f32;
    let mut var_left = 0.0_f32;
    let mut var_right = 0.0_f32;
    for index in 0..count {
        let a = left[index] - mean_left;
        let b = right[index] - mean_right;
        dot += a * b;
        var_left += a * a;
        var_right += b * b;
    }
    let denominator = (var_left * var_right).sqrt();
    if denominator <= f32::EPSILON {
        // 【为什么是 0 而不是 1】恒定的包络不说明任何位置：任何偏移都同样"吻合"。
        0.0
    } else {
        dot / denominator
    }
}

/// 样本级精修：在格级答案 ±`hop` 帧内取相关最大的那个样本位置。
#[allow(clippy::too_many_arguments)]
fn refine(
    block: &ProbeBlock<'_>,
    envelope: &TapeEnvelope,
    tape_left: &[f32],
    tape_right: &[f32],
    coarse: i64,
    window: usize,
    hop: usize,
    ops: &mut u64,
) -> i64 {
    if window == 0 {
        return coarse;
    }
    let tape_frames = if tape_right.is_empty() {
        tape_left.len()
    } else {
        tape_left.len().min(tape_right.len())
    } as i64;
    let mut best = coarse;
    let mut best_score = f32::NEG_INFINITY;
    let reach = hop as i64;
    for delta in -reach..=reach {
        let start = coarse.saturating_add(delta);
        // 逐点比较：本块的第 i 帧 ↔ 磁带第 (start - origin + i) 帧。
        let base = start.saturating_sub(envelope.origin);
        if base < 0 || base.saturating_add(window as i64) > tape_frames {
            continue;
        }
        *ops = ops.saturating_add(window as u64);
        if *ops > PROBE_OPS {
            return best;
        }
        let base = base as usize;
        let mut dot = 0.0_f32;
        let mut var_block = 0.0_f32;
        let mut var_tape = 0.0_f32;
        let mut mean_block = 0.0_f32;
        let mut mean_tape = 0.0_f32;
        for index in 0..window {
            mean_block += block.mono(index);
            mean_tape += if tape_right.is_empty() {
                tape_left[base + index]
            } else {
                (tape_left[base + index] + tape_right[base + index]) * 0.5
            };
        }
        mean_block /= window as f32;
        mean_tape /= window as f32;
        for index in 0..window {
            let a = block.mono(index) - mean_block;
            let b = (if tape_right.is_empty() {
                tape_left[base + index]
            } else {
                (tape_left[base + index] + tape_right[base + index]) * 0.5
            }) - mean_tape;
            dot += a * b;
            var_block += a * a;
            var_tape += b * b;
        }
        let denominator = (var_block * var_tape).sqrt();
        if denominator <= f32::EPSILON {
            continue;
        }
        let score = dot / denominator;
        if score > best_score {
            best_score = score;
            best = start;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: i64 = 48_000;
    const ORIGIN: i64 = 720_000;
    const BLOCK: usize = 960;

    /// 一份可控的"素材"：包络与样本都由这里给出，不碰任何宿主。
    struct TestTape {
        origin: i64,
        left: Vec<f32>,
        right: Vec<f32>,
        covered: Vec<Range<i64>>,
    }

    impl TapeSamples for TestTape {
        fn origin(&self) -> i64 {
            self.origin
        }
        fn left(&self) -> &[f32] {
            &self.left
        }
        fn right(&self) -> &[f32] {
            &self.right
        }
        fn covered(&self) -> &[Range<i64>] {
            &self.covered
        }
    }

    /// 确定性的伪随机噪声（xorshift64）：**非周期**，所以"位置"是唯一的。
    fn noise(frames: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..frames)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// 1 kHz 正弦（`48 kHz` 下周期 48 帧）—— 包络每 3 格重复一次。
    fn sine(frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|index| (std::f32::consts::TAU * index as f32 / 48.0).sin() * 0.5)
            .collect()
    }

    fn tape(left: Vec<f32>, covered: Vec<Range<i64>>) -> TestTape {
        TestTape {
            origin: ORIGIN,
            left,
            right: Vec::new(),
            covered,
        }
    }

    fn block_of(tape: &TestTape, start: usize, frames: usize) -> Vec<f32> {
        tape.left[start..start + frames].to_vec()
    }

    fn run(tape: TestTape, block: &[f32], frames_before: i64, hypotheses: &[i64]) -> ProbeVerdict {
        let probe = TapeProbe::new(Arc::new(tape)).expect("素材必须能建立包络");
        let block = ProbeBlock {
            left: block,
            right: &[],
            frames_before,
        };
        locate(
            &block,
            &probe.envelope,
            probe.samples.left(),
            probe.samples.right(),
            hypotheses,
        )
    }

    /// **不变量：锚点就是音频实际所在的位置。**
    ///
    /// 【现场原形】VEGAS `Render As`：渲染范围 `[15.000, 46.681]`，光标停在
    /// `39.396 s`（= 1 891 012）。宿主在首块报 `projectTimeSamples = 1 891 012`，
    /// 而喂进来的内容是**片段起点**（工程 720 000）。位置规则给出 `project_zero = 1 891 012`，
    /// 整条渲染右移 24.396 s；内容判定给出 720 000。
    #[test]
    fn the_anchor_is_where_the_audio_actually_is() {
        let tape = tape(noise(4 * RATE as usize, 7), Vec::new());
        let block = block_of(&tape, 0, BLOCK);
        let cursor = ORIGIN + 1_500 * 48;
        let verdict = run(tape, &block, 0, &[ORIGIN, cursor]);
        assert_eq!(verdict.outcome, ProbeOutcome::Content);
        assert_eq!(
            verdict.offset, ORIGIN,
            "内容说这一块在片段起点，宿主读数（{cursor}）只是上一次的光标"
        );
        assert!(verdict.score > 0.9, "非周期素材必须高相关：{}", verdict.score);
        assert!(
            verdict.score - verdict.second > PROBE_MIN_MARGIN,
            "次优必须明显更低：{} vs {}",
            verdict.score,
            verdict.second
        );
    }

    /// **不变量：预览从播放头起播时，内容判定与位置规则给出同一个数。**
    ///
    /// 【为什么这条必须留】它是"新判据不改变旧行为"的那一半：§1.4 的预览现场
    /// （`raw=720000 tape=23728 project=743728`）无法区分"按播放头锚"与"按片段起点锚"，
    /// 因为三者同值。内容判定必须让这一条**逐帧不变**。
    #[test]
    fn a_preview_at_the_playhead_is_confirmed_by_the_content() {
        let tape = tape(noise(4 * RATE as usize, 11), Vec::new());
        let start = 72_000;
        let block = block_of(&tape, start, BLOCK);
        let hint = ORIGIN + start as i64;
        let verdict = run(tape, &block, 0, &[ORIGIN, hint]);
        assert_eq!(verdict.outcome, ProbeOutcome::Content);
        assert_eq!(verdict.offset, hint, "播放头在片段内时它就是答案");
    }

    /// **不变量：静音头不消耗这次机会，第一个有信号的块才定稿。**
    ///
    /// 【现场原形】渲染范围从片段**之前**开始：前 5 块全零。静音块在任何锚点下输出都是
    /// 静音，所以它不能把"这次机会"用掉 —— 否则整条渲染会退回位置规则（也就是今天
    /// 那个错的答案）。
    #[test]
    fn a_silent_head_waits_for_the_first_signal_block() {
        let tape = tape(noise(4 * RATE as usize, 13), Vec::new());
        let silence = vec![0.0_f32; BLOCK];
        let probe = TapeProbe::new(Arc::new(tape)).expect("素材必须能建立包络");
        let locate_block = |block: &[f32], frames_before: i64| {
            let block = ProbeBlock {
                left: block,
                right: &[],
                frames_before,
            };
            locate(
                &block,
                &probe.envelope,
                probe.samples.left(),
                probe.samples.right(),
                &[ORIGIN],
            )
        };
        for index in 0..5_i64 {
            let verdict = locate_block(&silence, index * BLOCK as i64);
            assert_eq!(
                verdict.outcome,
                ProbeOutcome::Undecided(ProbeReason::Silent),
                "静音块必须如实回答 Silent（第 {index} 块）"
            );
        }
        let signal = probe.samples.left()[0..BLOCK].to_vec();
        let frames_before = 5 * BLOCK as i64;
        let verdict = locate_block(&signal, frames_before);
        assert_eq!(verdict.outcome, ProbeOutcome::Content);
        assert_eq!(
            verdict.offset,
            ORIGIN - frames_before,
            "喂入流第 0 帧在素材起点之前 5 个块"
        );
    }

    /// **不变量：周期性素材不许猜，必须带原因退回位置规则。**
    #[test]
    fn an_ambiguous_material_falls_back_with_a_reason() {
        let tape = tape(sine(4 * RATE as usize), Vec::new());
        let block = block_of(&tape, 0, BLOCK);
        let verdict = run(tape, &block, 0, &[ORIGIN]);
        assert_eq!(
            verdict.outcome,
            ProbeOutcome::Undecided(ProbeReason::Ambiguous)
        );
        assert_eq!(verdict.reason().map(ProbeReason::as_str), Some("ambiguous"));
        assert!(verdict.score > 0.0, "相似度读数必须如实给出：{}", verdict.score);
        assert!(verdict.second > 0.0, "次优读数必须如实给出：{}", verdict.second);
    }

    /// **不变量：洞里的内容永远不算"对上了"。**
    ///
    /// 【为什么】洞是"没有素材"的地方，而"没有素材"绝不能被当成"内容是静音" ——
    /// 那会让探测在洞里给出一个自洽但错误的偏移。
    #[test]
    fn a_tape_with_a_hole_is_never_matched_inside_the_hole() {
        let left = noise(4 * RATE as usize, 17);
        let covered = vec![
            ORIGIN..ORIGIN + RATE,
            ORIGIN + 2 * RATE..ORIGIN + 4 * RATE,
        ];
        let tape = tape(left, covered);
        let start = RATE as usize + 24_000;
        let block = block_of(&tape, start, BLOCK);
        let verdict = run(tape, &block, 0, &[ORIGIN]);
        assert_eq!(
            verdict.outcome,
            ProbeOutcome::Undecided(ProbeReason::NoMaterial)
        );
        assert_eq!(verdict.reason().map(ProbeReason::as_str), Some("no-material"));
    }

    /// **不变量：太短的块不判。**
    #[test]
    fn a_short_block_is_not_judged() {
        let tape = tape(noise(4 * RATE as usize, 19), Vec::new());
        let block = block_of(&tape, 0, 128);
        let verdict = run(tape, &block, 0, &[ORIGIN]);
        assert_eq!(verdict.outcome, ProbeOutcome::Undecided(ProbeReason::Short));
        assert_eq!(verdict.reason().map(ProbeReason::as_str), Some("short"));
    }

    /// **这条钉住第七轮那个常量错误：hop 必须小于一个宿主块。**
    ///
    /// 【为什么它值得一条用例】第七轮把 `hop` 写成 1024、把 `PROBE_MIN_FRAMES` 写成
    /// 4096，而 VEGAS 的块长是 848/960/1200 —— 两个常量各自与现场矛盾，于是"照着方案
    /// 写出来的探测一次都不会运行"。这两个数必须被断言钉住，而不是靠读代码发现。
    #[test]
    fn the_hop_is_smaller_than_a_host_block() {
        assert!(
            (HOP_FRAMES as usize) < 848,
            "hop 必须小于 VEGAS 的最小块长 848，否则一块折不出两格"
        );
        assert!(
            PROBE_MIN_FRAMES <= 848,
            "判定门槛必须能被一个 848 帧的宿主块满足"
        );
    }
}
