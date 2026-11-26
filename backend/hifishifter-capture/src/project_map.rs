//! 磁带帧 → 工程帧的映射：把"喂进来第几帧"翻译成"工程第几帧"。
//!
//! 【为什么必须有这一层】捕获通路的"位置"只有一个数，却被同时当成两种东西：
//!
//!   * **磁带走带位置**（喂入域）：这条喂入流从第一块起数到现在的帧号，永远从 0 起；
//!   * **工程时间**（工程域）：这一帧在工程里究竟落在哪一秒。
//!
//! 钉住喂入流（VEGAS 的预览与 Render As）里宿主**不报位置**，于是工程时间只能由
//! 映射给出。第四轮把"喂入域的 0"当成"工程 0"（`HFS_PINNED_ORIGIN` 的默认），
//! 第三轮把"首块 raw"当成工程起点 —— 两个都不对，因为真起点既不在宿主读数里，
//! 也不在自造的 0 上，它在**捕获片段自己身上**：
//!
//! `text
//! project_frame(t) = clip_start_frame + (t - clip_source_start_frame)
//! `
//!
//! 【为什么是分段表】一次录制可能被洞切成几片（宿主停播、位置跳跃），补录还会让
//! 同一份素材带上第二条 take。每一片各有自己的 `(clip_start, clip_source_start)`。
//! 一张"起点 + 长度"的表能同时表达全部片段，而单个 `origin` 只能表达一片。
//!
//! 【为什么读方必须无锁】音频线程每一块都要问一次"这一块在工程哪一段"。取锁就会把
//! 一次录制变成一次互斥量竞争，而竞争的另一方是 UI 读数 —— 那正是第四轮
//! `FeedTimeline` 的教训。这里用 seqlock：写方（promote / 投影重建，非音频线程）
//! 极少发生，读方（音频线程）只做有界次数的重试，失败就诚实地回答"没有映射"。

use crate::anchor_probe::{self, ProbeBlock, ProbeOutcome, ProbeVerdict, TapeProbe};
use hifishifter_plugin_core::sync::MutexExt;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// 映射表能容纳的片段数。
///
/// 【为什么是 16】一次录制的段是 8 秒定长，洞与补录各占一片；16 片覆盖了"几分钟
/// 的录制 + 若干次补录"。超出时**截断而不是扩容**：多出来的片段映射不到，读方会
/// 得到"没有映射"并退回直通 —— 一个可观测的降级，而不是一个静默的错位。
pub const PROJECT_MAP_SLOTS: usize = 16;

/// 表里的一段：磁带帧区间 `[tape_start, tape_end)` 对应工程帧
/// `project_start + (t - tape_start)`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectMapSegment {
    pub tape_start: i64,
    pub tape_end: i64,
    pub project_start: i64,
    /// 这一片属于哪一次录制（take 号）。只用于多 take 时的消歧。
    pub take: u64,
}

impl ProjectMapSegment {
    /// 本段覆盖多少帧。
    pub fn frames(&self) -> i64 {
        self.tape_end.saturating_sub(self.tape_start)
    }

    /// 工程区间 `[project_start, project_start + frames)`。
    pub fn project_end(&self) -> i64 {
        self.project_start.saturating_add(self.frames())
    }

    /// 这一帧磁带位置是否落在本段里。
    pub fn covers(&self, tape_frame: i64) -> bool {
        tape_frame >= self.tape_start && tape_frame < self.tape_end
    }
}

/// "喂入域的第 0 帧在工程哪里"的**选择策略**（环境变量 `HFS_CAPTURE_ANCHOR`）。
///
/// 【为什么这是一个可切换的判据】锚点是本轮唯一改变声音位置的决定，而三种答案在产物上
/// **完全同形**（都是"某段音频出现在错误的时间上"）：
///
///   * `Auto`（默认）：首块宿主读数落在片段的工程窗口内 ⇒ 它就是播放头（VEGAS 的预卷
///     从播放头开始喂）；否则退回片段起点（Render As 把 `projectTimeSamples` 留在上一次
///     的播放头，那个值既不是起点也不是范围起点）。
///   * `Playhead`：强制播放头（现场对照：渲染也会"跟着播放头跑"）。
///   * `Clip`：强制片段起点（= 第五轮行为，用来一键复现"预览起播读片段起点"）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnchorPolicy {
    #[default]
    Auto,
    Playhead,
    Clip,
    /// **只接受内容判定**：探测未决时宁可直接通，也不猜一个位置。
    ///
    /// 【为什么需要它】`Auto` 在探测未决时会退回位置规则 —— 那是「宁可近似也不静音」
    /// 的产品选择。但排查时需要的恰恰相反：一条**不猜**的对照，让「猜错了」与
    /// 「没猜」在产物上分开。
    Content,
}

impl AnchorPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Playhead => "playhead",
            Self::Clip => "clip",
            Self::Content => "content",
        }
    }
}

/// 锚点是从哪条证据来的。
///
/// 【为什么必须能在日志里读出来】"这一块被当成工程哪一段"是本轮唯一的判断，而判断的
/// **依据**与判断本身是两件事：读数相同、依据不同，下一次改动就会把它做错。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorSource {
    /// 宿主在喂入流首块报的位置落在片段的工程窗口里 ⇒ 它就是播放头。
    Playhead,
    /// 退回片段起点（渲染 / 宿主读数在窗口之外 / 强制）。
    Clip,
    /// **内容判定**：把这一块与素材本身逐点比过，`project_zero` 就是实测的那个位置。
    ///
    /// 【为什么它必须是一个独立的来源】它与前两个回答的是不同的问题：前两个问
    /// 「宿主说的这个数落在窗口里吗」（位置），这一个问「这段内容在素材的哪一处」
    /// （身份）。一个数的位置，不能回答它是什么。
    Content,
}

impl AnchorSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Playhead => "playhead",
            Self::Clip => "clip",
            Self::Content => "content",
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Self::Playhead => 1,
            Self::Clip => 2,
            Self::Content => 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Playhead),
            2 => Some(Self::Clip),
            3 => Some(Self::Content),
            _ => None,
        }
    }
}

/// 一条钉住喂入流的**工程锚点**：喂入域第 0 帧落在工程的哪一帧。
///
/// 【为什么必须是一个显式的值】此前"喂入域的 0 在工程哪里"这个答案藏在映射公式里
/// （隐含假设 = 片段起点），没有人能单独审阅它、断言它，也没有任何一行日志说出它。
/// 提成一个值之后，它有了三样东西：来源、日志、测试。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedAnchor {
    /// 喂入域 0 ⇒ 工程帧。
    pub project_zero: i64,
    /// 本条喂入流还能走多远（帧）。越界即"没有映射"（输出退回直通）。
    pub feed_end: i64,
    /// 这一片属于哪个 take（多 take 消歧 + 诊断）。
    pub take: u64,
    pub source: AnchorSource,
}

impl FeedAnchor {
    /// 本帧是否落在这条喂入流的覆盖范围内。
    pub fn covers(&self, feed_tape: i64) -> bool {
        feed_tape >= 0 && feed_tape < self.feed_end
    }

    /// 喂入帧 → 工程帧。越界回答 `None`（调用方据此退回直通）。
    pub fn project(&self, feed_tape: i64) -> Option<i64> {
        if !self.covers(feed_tape) {
            return None;
        }
        Some(self.project_zero.saturating_add(feed_tape))
    }
}

/// 无锁的分段映射表（seqlock）。
///
/// 【为什么不直接存一个 `FeedAnchor`】"当前是哪一片"取决于**这一块**的喂入帧号，
/// 而磁带帧号每块都在变。表是唯一能一次回答"任意磁带帧落在哪一片"的形状。
pub struct ProjectMapCell {
    /// 偶数 = 稳定，奇数 = 正在写。
    epoch: AtomicU64,
    len: AtomicUsize,
    revision: AtomicU64,
    tape_start: [AtomicI64; PROJECT_MAP_SLOTS],
    tape_end: [AtomicI64; PROJECT_MAP_SLOTS],
    project_start: [AtomicI64; PROJECT_MAP_SLOTS],
    take: [AtomicU64; PROJECT_MAP_SLOTS],
    /// 内容定位器的**叶子槽**（见 [`crate::anchor_probe`]）。
    ///
    /// 【为什么是 `Mutex` 而不是原子】它装的是一个 `Arc<TapeProbe>`，而 `Arc` 没有
    /// 安全的原子换法（`AtomicPtr` 要自己管引用计数）。纪律是**叶子锁**：锁内只做
    /// 一次 `clone()`，任何相关运算都在锁外。音频线程只在「这条流还没定稿」的块上
    /// 取一次，定稿之后一次都不取。
    probe: Mutex<Option<Arc<TapeProbe>>>,
}

impl Default for ProjectMapCell {
    fn default() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            len: AtomicUsize::new(0),
            revision: AtomicU64::new(0),
            tape_start: std::array::from_fn(|_| AtomicI64::new(0)),
            tape_end: std::array::from_fn(|_| AtomicI64::new(0)),
            project_start: std::array::from_fn(|_| AtomicI64::new(0)),
            take: std::array::from_fn(|_| AtomicU64::new(0)),
            probe: Mutex::new(None),
        }
    }
}

/// 读方最多重试几次。
///
/// 【为什么是有界重试而不是自旋】音频线程**不允许**无界自旋。写方一次发布只有十几个
/// 原子写，撞上的概率极低；撞上两次以上就说明读方在一个极窄的窗口里反复被打断 ——
/// 那时最诚实的答案是"没有映射"（输出退回直通），而不是继续等。
const READ_ATTEMPTS: usize = 4;

impl ProjectMapCell {
    /// 发布一份新表（**只允许非音频线程**：promote / 投影重建）。
    pub fn publish(&self, revision: u64, segments: &[ProjectMapSegment]) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        let len = segments.len().min(PROJECT_MAP_SLOTS);
        for (index, segment) in segments.iter().take(len).enumerate() {
            self.tape_start[index].store(segment.tape_start, Ordering::Relaxed);
            self.tape_end[index].store(segment.tape_end, Ordering::Relaxed);
            self.project_start[index].store(segment.project_start, Ordering::Relaxed);
            self.take[index].store(segment.take, Ordering::Relaxed);
        }
        self.len.store(len, Ordering::Relaxed);
        self.revision.store(revision, Ordering::Relaxed);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// 清空（清空素材 / 新建工程）。
    pub fn clear(&self) {
        self.publish(0, &[]);
    }

    /// 当前表的代次（诊断）。
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// 表里有几段。
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire).min(PROJECT_MAP_SLOTS)
    }

    /// 表是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// **本喂入流的工程锚点**：喂入域第 0 帧在工程哪里。
    ///
    /// `hint` 是宿主在**这一块**报的位置（`projectTimeSamples`）。它的可信度取决于
    /// 宿主此刻在做什么：
    ///
    ///   * 播放预览（VEGAS 的预卷）：它**就是播放头** —— 宿主从当前光标开始喂音频，
    ///     片段起点只是这段内容在工程里的落点，两者只有在"从片段起点起播"时才相等；
    ///   * Render As：它被留在**上一次的播放头**上，既不是片段起点，也不是渲染范围起点。
    ///
    /// 于是判据只能问一个可观测的问题：**它落在片段的工程窗口里吗**。落在里面 ⇒ 它是
    /// 一个合法的播放头；落在外面 ⇒ 它显然不是"这一段内容现在的位置"，退回片段起点
    /// （那是 Render As 唯一有证据的答案：渲染的范围起点通常正是片段起点）。
    ///
    /// 读方撞上写方、或表为空时返回 `None` —— 调用方据此退回直通，而不是猜一个位置。
    pub fn anchor(&self, hint: i64, policy: AnchorPolicy) -> Option<FeedAnchor> {
        self.anchor_probed(hint, policy, None).anchor
    }

    /// 与 [`Self::anchor`] 相同，但允许**用内容回答位置**（见 [`crate::anchor_probe`]）。
    ///
    /// 【为什么把结论与锚点一起交出去】调用方要写两行日志（`feed begin` 的 `probe=`
    /// 与独立的 `anchor probe:` 行），而这两行都必须能回答「为什么这次没按内容判」。
    /// 把 verdict 丢掉，那个问题就只剩「anchor=playhead」一个词 —— 那正是本轮之前的
    /// 现场。
    pub fn anchor_probed(
        &self,
        hint: i64,
        policy: AnchorPolicy,
        block: Option<ProbeBlock<'_>>,
    ) -> AnchorOutcome {
        let Some((segments, len)) = self.read_segments() else {
            return AnchorOutcome::default();
        };
        let segments = &segments[..len];
        // 【强制策略与关闭开关都走位置规则】`HFS_CAPTURE_ANCHOR=playhead|clip` 是现场
        // 对照用的强制值；`HFS_CAPTURE_PROBE=off` 是一键退回第七轮行为。两者都不该被
        // 内容判定绕过 —— 否则「一键对照」就不成立了。
        if matches!(policy, AnchorPolicy::Playhead | AnchorPolicy::Clip)
            || !anchor_probe::probe_enabled()
        {
            return AnchorOutcome {
                anchor: position_anchor(segments, hint, policy),
                probe: None,
            };
        }
        if let (Some(block), Some(probe)) = (block, self.probe_snapshot()) {
            // 【两个假设先测】片段起点与宿主读数 —— 它们覆盖了现场的全部历史用例。
            let mut hypotheses = [0_i64; 2];
            let mut count = 0_usize;
            if let Some(earliest) = earliest_segment(segments) {
                hypotheses[count] = earliest.project_start;
                count += 1;
            }
            if hint >= 0 {
                hypotheses[count] = hint;
                count += 1;
            }
            let verdict = anchor_probe::locate(
                &block,
                &probe.envelope,
                probe.samples.left(),
                probe.samples.right(),
                &hypotheses[..count],
            );
            let anchor = if verdict.outcome == ProbeOutcome::Content {
                // 【内容说得出位置，但映射表里没有那一片】这不是位置规则能补的，退回
                // 位置规则并把 verdict 一起交出去（日志里能看到 offset 与分数的矛盾）。
                anchor_at(segments, verdict.offset, block.frames_before)
            } else {
                None
            };
            return AnchorOutcome {
                anchor: anchor.or_else(|| position_anchor(segments, hint, policy)),
                probe: Some(verdict),
            };
        }
        AnchorOutcome {
            anchor: position_anchor(segments, hint, policy),
            probe: None,
        }
    }

    /// 发布（或撤销）内容定位器。**只允许非音频线程**，且必须与样本**同一次**发布。
    pub fn publish_probe(&self, probe: Option<Arc<TapeProbe>>) {
        *self.probe.lock_or_recover() = probe;
    }

    /// 取一份定位器的引用（叶子锁：锁内只有一次 `clone()`）。
    fn probe_snapshot(&self) -> Option<Arc<TapeProbe>> {
        hifishifter_plugin_core::sync::try_lock_or_recover(&self.probe)
            .and_then(|slot| slot.clone())
    }

    /// 一次 seqlock 读：把整张表拷进定长数组（**不在读区里跑探测**）。
    ///
    /// 【为什么要把读与算分开】探测要几百微秒；放在读区里会让写方（投影重建）每撞一次
    /// 就让我们重试一次、重算一遍。表只有 16 段，先拷出来再算，代价是 16 个 i64。
    fn read_segments(&self) -> Option<([ProjectMapSegment; PROJECT_MAP_SLOTS], usize)> {
        for _ in 0..READ_ATTEMPTS {
            let before = self.epoch.load(Ordering::Acquire);
            if before & 1 == 1 {
                continue;
            }
            let len = self.len.load(Ordering::Relaxed).min(PROJECT_MAP_SLOTS);
            let mut segments = [ProjectMapSegment {
                tape_start: 0,
                tape_end: 0,
                project_start: 0,
                take: 0,
            }; PROJECT_MAP_SLOTS];
            for (index, segment) in segments.iter_mut().take(len).enumerate() {
                segment.tape_start = self.tape_start[index].load(Ordering::Relaxed);
                segment.tape_end = self.tape_end[index].load(Ordering::Relaxed);
                segment.project_start = self.project_start[index].load(Ordering::Relaxed);
                segment.take = self.take[index].load(Ordering::Relaxed);
            }
            let after = self.epoch.load(Ordering::Acquire);
            if before == after {
                return Some((segments, len));
            }
        }
        None
    }
}

/// 一次锚点求解的完整产物：锚点 + 内容探测的结论。
///
/// 【为什么两者一起交出去】调用方要写两行日志，而它们回答的是两个不同的问题：
/// 「这一块被当成工程哪一段」与「凭什么」。只要锚点，第二行就只剩一个 `anchor=` 词 ——
/// 那正是本轮之前的现场。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnchorOutcome {
    /// `None` = 这一轮没有锚点（映射表为空 / 正在写 / 静音块还没定稿）。
    pub anchor: Option<FeedAnchor>,
    /// `None` = 这一轮没跑内容探测（强制策略 / 关掉开关 / 没有素材）。
    pub probe: Option<ProbeVerdict>,
}

/// 「位置规则」的锚点：宿主读数落在片段的工程窗口里 ⇒ 播放头，否则片段起点。
///
/// 这是第七轮的全部逻辑，现在只是**内容探测不可用时的回退**（见 `AnchorOutcome`）。
fn position_anchor(
    segments: &[ProjectMapSegment],
    hint: i64,
    policy: AnchorPolicy,
) -> Option<FeedAnchor> {
    let mut earliest: Option<FeedAnchor> = None;
    let mut containing: Option<FeedAnchor> = None;
    for segment in segments {
        let frames = segment.frames();
        if frames <= 0 {
            continue;
        }
        // 【工程上最早的一片】就是「渲染从片段起点开始」那一条。多 take 时它是唯一
        // 确定的退路（不能按表里的顺序挑：那是写入顺序，不是工程顺序）。
        if earliest.is_none_or(|current| {
            (segment.project_start, segment.take) < (current.project_zero, current.take)
        }) {
            earliest = Some(FeedAnchor {
                project_zero: segment.project_start,
                feed_end: frames,
                take: segment.take,
                source: AnchorSource::Clip,
            });
        }
        let project_end = segment.project_end();
        if hint >= segment.project_start && hint < project_end && containing.is_none() {
            containing = Some(FeedAnchor {
                project_zero: hint,
                feed_end: project_end.saturating_sub(hint),
                take: segment.take,
                source: AnchorSource::Playhead,
            });
        }
    }
    let playhead = matches!(policy, AnchorPolicy::Playhead) && hint >= 0;
    match policy {
        AnchorPolicy::Clip => earliest,
        // 【强制播放头】没有片段包含它时也要按播放头建锚：那是这条开关的唯一用途 ——
        // 现场对照「渲染是不是也跟着播放头跑」。
        AnchorPolicy::Playhead if playhead => containing.or_else(|| {
            earliest.map(|anchor| FeedAnchor {
                project_zero: hint,
                // 已经越过片段的起点时，剩余长度就是片段剩余长度；落在起点之前时整片
                // 都还没用到（consumed 夹到 0）。
                feed_end: anchor
                    .feed_end
                    .saturating_sub(hint.saturating_sub(anchor.project_zero).max(0)),
                take: anchor.take,
                source: AnchorSource::Playhead,
            })
        }),
        AnchorPolicy::Playhead | AnchorPolicy::Auto => containing.or(earliest),
        // 【只接受内容判定】探测未决时宁可直接通，也不猜一个位置。
        AnchorPolicy::Content => None,
    }
}

/// 工程上最早的那一片（内容探测的「片段起点」假设就用它）。
fn earliest_segment(segments: &[ProjectMapSegment]) -> Option<&ProjectMapSegment> {
    segments
        .iter()
        .filter(|segment| segment.frames() > 0)
        .min_by_key(|segment| (segment.project_start, segment.take))
}

/// 把「喂入流第 0 帧在工程第 `offset` 帧」变成一个 `FeedAnchor`。
///
/// 【`feed_end` 怎么算】它回答「这条流还能走多远」，因此必须是**这一片剩余的长度**：
/// `segment.project_end() - offset`。用片段的整段长度会让流走出片段之后仍然「有映射」，
/// 输出把片段之后的内容当成片段的一部分继续处理 —— 那比直通糟得多。
fn anchor_at(
    segments: &[ProjectMapSegment],
    offset: i64,
    frames_before: i64,
) -> Option<FeedAnchor> {
    let here = offset.saturating_add(frames_before);
    let segment = segments
        .iter()
        .find(|segment| here >= segment.project_start && here < segment.project_end())
        .or_else(|| {
            segments
                .iter()
                .find(|segment| offset >= segment.project_start && offset < segment.project_end())
        })?;
    let feed_end = segment.project_end().saturating_sub(offset);
    if feed_end <= 0 {
        return None;
    }
    Some(FeedAnchor {
        project_zero: offset,
        feed_end,
        take: segment.take,
        source: AnchorSource::Content,
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    fn cell(segments: &[ProjectMapSegment]) -> ProjectMapCell {
        let cell = ProjectMapCell::default();
        cell.publish(7, segments);
        cell
    }

    fn segment(tape_start: i64, tape_end: i64, project_start: i64, take: u64) -> ProjectMapSegment {
        ProjectMapSegment {
            tape_start,
            tape_end,
            project_start,
            take,
        }
    }

    /// **不变量：宿主读数落在片段之外时，工程位置跟着片段走。**
    ///
    /// 【现场原形】片段起点 28.000 s（= 1 344 000 帧），宿主 `projectTimeSamples` 被
    /// VEGAS 留在 893 720（18.62 s）。锚点必须答 1 344 000，而不是 893 720，也不是 0。
    #[test]
    fn project_position_follows_the_clip_when_the_playhead_is_outside_it() {
        const RATE: i64 = 48_000;
        let cell = cell(&[segment(0, 1_581_056, 28 * RATE, 1)]);
        let anchor = cell
            .anchor(893_720, AnchorPolicy::Auto)
            .expect("必须有锚点");
        assert_eq!(anchor.source, AnchorSource::Clip);
        assert_eq!(anchor.project(0), Some(28 * RATE));
        assert_eq!(anchor.project(RATE), Some(29 * RATE));
        // 宿主读数在窗口之外时只是一个无用的数：不参与换算。
        assert_eq!(
            cell.anchor(0, AnchorPolicy::Auto).map(|a| a.project(0)),
            Some(Some(28 * RATE))
        );
    }

    /// **不变量：宿主读数是片段内的播放头时，锚点跟着播放头。**
    ///
    /// 【现场原形】VEGAS 预览：片段在 15.000 s，播放头在 33.625 s（1 614 000），预卷
    /// 从播放头开始喂 9 040 帧。第五轮把喂入域 0 锚在片段起点 ⇒ 整段预卷读的是
    /// `[15.000 s, 15.188 s]`，而宿主正在播 `[33.625 s, 33.813 s]` —— 用户听到的
    /// 就是"片段起点的处理结果"，直到 kPlaying 升起、喂入流切回传输域才恢复。
    #[test]
    fn a_preview_that_starts_mid_item_anchors_on_the_playhead() {
        const RATE: i64 = 48_000;
        const CLIP_START: i64 = 15 * RATE;
        const PLAYHEAD: i64 = 1_614_000; // 33.625 s
        let cell = cell(&[segment(0, 1_568_768, CLIP_START, 1)]);
        let anchor = cell
            .anchor(PLAYHEAD, AnchorPolicy::Auto)
            .expect("必须有锚点");
        assert_eq!(anchor.source, AnchorSource::Playhead);
        assert_eq!(anchor.project(0), Some(PLAYHEAD));
        assert_eq!(anchor.project(9_040), Some(PLAYHEAD + 9_040));
        assert_eq!(
            anchor.project(1_568_768),
            None,
            "越过片段末尾的帧没有映射，必须退回直通"
        );
    }

    /// **不变量：从片段起点起播时，两条判据给出同一个数。**
    ///
    /// 【为什么这条要留】它是"新规则不改变旧行为"的那一半：片段起点 = 播放头时，
    /// 播放头锚点与片段锚点逐帧相同。第五轮的全部现场都是这个形状，所以那批用例
    /// **不可能**发现缺陷 —— 这条把"为什么"钉在测试里。
    #[test]
    fn a_preview_at_the_item_start_is_unchanged() {
        const RATE: i64 = 48_000;
        const START: i64 = 15 * RATE;
        let cell = cell(&[segment(0, 960_000, START, 1)]);
        let playhead = cell.anchor(START, AnchorPolicy::Auto).unwrap();
        let clip = cell.anchor(START, AnchorPolicy::Clip).unwrap();
        assert_eq!(playhead.source, AnchorSource::Playhead);
        assert_eq!(clip.source, AnchorSource::Clip);
        for tape in [0_i64, 1, RATE, 960_000 - 1] {
            assert_eq!(
                playhead.project(tape),
                clip.project(tape),
                "从片段起点起播时两个锚点必须同值（tape={tape}）"
            );
        }
    }

    /// 没有映射时回答 `None` —— 调用方据此退回直通。
    #[test]
    fn an_empty_table_answers_none() {
        let cell = ProjectMapCell::default();
        assert!(cell.anchor(0, AnchorPolicy::Auto).is_none());
        assert!(cell.is_empty());
    }

    /// 落在所有片段之外时回答 `None`。
    #[test]
    fn a_tape_frame_outside_every_segment_answers_none() {
        let cell = cell(&[segment(0, 960, 5_000, 1)]);
        let anchor = cell.anchor(5_000, AnchorPolicy::Auto).expect("必须有锚点");
        assert!(anchor.project(960).is_none(), "片段末尾之外没有映射");
        assert!(anchor.project(-1).is_none(), "负的喂入帧没有意义");
        let before = cell.anchor(-1, AnchorPolicy::Auto).expect("必须有锚点");
        assert_eq!(before.source, AnchorSource::Clip);
        assert!(before.project(-1).is_none());
    }

    /// 多 take：宿主读数落在哪一片，就选哪一片；落在两片之外时退回工程上最早的一片。
    #[test]
    fn the_hint_disambiguates_two_takes() {
        // 两个 take 的磁带帧号都从 0 起（各自一份源），工程位置不同。
        let cell = cell(&[segment(0, 1_000, 10_000, 1), segment(0, 1_000, 40_000, 2)]);
        let second = cell.anchor(40_500, AnchorPolicy::Auto).unwrap();
        assert_eq!(second.take, 2, "读数在第二片里，必须选第二片");
        assert_eq!(second.source, AnchorSource::Playhead);
        assert_eq!(second.project(0), Some(40_500));

        let first = cell.anchor(10_500, AnchorPolicy::Auto).unwrap();
        assert_eq!(first.take, 1);
        assert_eq!(first.project(0), Some(10_500));

        let fallback = cell.anchor(0, AnchorPolicy::Auto).unwrap();
        assert_eq!(fallback.take, 1, "读数落在两片之外时按工程顺序取最早的一片");
        assert_eq!(fallback.source, AnchorSource::Clip);
        assert_eq!(fallback.project(0), Some(10_000));
    }

    /// 重新发布之后旧读数立刻失效（代次与内容一起换）。
    #[test]
    fn republishing_replaces_the_table() {
        let cell = cell(&[segment(0, 1_000, 10_000, 1)]);
        assert_eq!(cell.revision(), 7);
        cell.publish(9, &[segment(0, 2_000, 20_000, 2)]);
        assert_eq!(cell.revision(), 9);
        assert_eq!(
            cell.anchor(20_000, AnchorPolicy::Auto)
                .map(|a| a.project(1_500)),
            Some(Some(21_500))
        );
        // 旧表覆盖磁带 [0, 1000)；新表覆盖 [0, 2000)。2 500 帧只在新表之外 ——
        // 它必须解析不到任何东西，而不是退回旧片段的起点。
        assert!(
            cell.anchor(10_000, AnchorPolicy::Auto)
                .and_then(|a| a.project(2_500))
                .is_none(),
            "旧片段必须消失"
        );
    }

    /// 表超过容量时截断，而不是越界。
    #[test]
    fn a_table_bigger_than_the_capacity_is_truncated() {
        let segments = (0..PROJECT_MAP_SLOTS as i64 + 4)
            .map(|index| segment(index * 100, index * 100 + 100, index * 1_000, 1))
            .collect::<Vec<_>>();
        let cell = cell(&segments);
        assert_eq!(cell.len(), PROJECT_MAP_SLOTS);
        assert!(cell
            .anchor(0, AnchorPolicy::Auto)
            .and_then(|a| a.project(PROJECT_MAP_SLOTS as i64 * 100 + 10))
            .is_none());
    }

    /// **开关的三种取值各自能被观测。** 现场要靠它一键对照"渲染是不是也跟着播放头跑"。
    #[test]
    fn the_policy_switches_the_anchor_source() {
        const RATE: i64 = 48_000;
        const START: i64 = 15 * RATE;
        let cell = cell(&[segment(0, 960_000, START, 1)]);
        let playhead = START + 5 * RATE;
        assert_eq!(
            cell.anchor(playhead, AnchorPolicy::Auto).unwrap().source,
            AnchorSource::Playhead
        );
        assert_eq!(
            cell.anchor(playhead, AnchorPolicy::Clip).unwrap().source,
            AnchorSource::Clip
        );
        assert_eq!(
            cell.anchor(playhead, AnchorPolicy::Clip)
                .unwrap()
                .project(0),
            Some(START)
        );
        // 宿主读数在片段之外时，强制播放头仍然按播放头建锚（那是这条开关的用途）。
        let outside = START - 100_000;
        let forced = cell.anchor(outside, AnchorPolicy::Playhead).unwrap();
        assert_eq!(forced.source, AnchorSource::Playhead);
        assert_eq!(forced.project(0), Some(outside));
    }
}
