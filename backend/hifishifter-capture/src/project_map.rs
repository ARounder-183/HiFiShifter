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

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

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

/// 解析出来的**一段**映射（读方拿到的形状）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectMapping {
    /// `tape_start` 对应的工程帧。
    pub project_origin: i64,
    /// 这份映射适用的磁带帧区间 `[tape_start, tape_end)`。
    pub tape_start: i64,
    pub tape_end: i64,
    /// 发布它的投影代次（诊断用；不同代次之间不得混用）。
    pub revision: u64,
}

impl ProjectMapping {
    /// 这一帧磁带位置是否落在本段里。
    pub fn covers(&self, tape_frame: i64) -> bool {
        tape_frame >= self.tape_start && tape_frame < self.tape_end
    }

    /// 磁带帧 → 工程帧。
    pub fn project(&self, tape_frame: i64) -> Option<i64> {
        if !self.covers(tape_frame) {
            return None;
        }
        Some(
            self.project_origin
                .saturating_add(tape_frame.saturating_sub(self.tape_start)),
        )
    }
}

/// 无锁的分段映射表（seqlock）。
///
/// 【为什么不直接存一个 `ProjectMapping`】"当前是哪一片"取决于**这一块**的磁带帧号，
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

    /// 解析 `tape_frame` 落在哪一段。
    ///
    /// `hint` 是宿主的原始位置读数（`projectTimeSamples`）。它**不可信**（VEGAS 的
    /// Render As 会把它留在最后的播放头），但它在"多 take 时该选哪一片"这一件事上
    /// 仍然是最便宜的线索：只要它落在某一片的工程区间里，就优先那一片。找不到任何
    /// 覆盖 `tape_frame` 的段，或读方撞上写方，都返回 `None` —— 调用方据此退回
    /// 直通，而不是猜一个位置。
    pub fn resolve(&self, tape_frame: i64, hint: i64) -> Option<ProjectMapping> {
        for _ in 0..READ_ATTEMPTS {
            let before = self.epoch.load(Ordering::Acquire);
            if before & 1 == 1 {
                continue;
            }
            let len = self.len.load(Ordering::Relaxed).min(PROJECT_MAP_SLOTS);
            let revision = self.revision.load(Ordering::Relaxed);
            let mut best: Option<(u8, ProjectMapping)> = None;
            for index in 0..len {
                let tape_start = self.tape_start[index].load(Ordering::Relaxed);
                let tape_end = self.tape_end[index].load(Ordering::Relaxed);
                let project_origin = self.project_start[index].load(Ordering::Relaxed);
                let mapping = ProjectMapping {
                    project_origin,
                    tape_start,
                    tape_end,
                    revision,
                };
                if !mapping.covers(tape_frame) {
                    continue;
                }
                // 2 = 宿主读数也落在这片里（多 take 时的强证据）；1 = 只按磁带帧命中。
                let project_end = project_origin.saturating_add(tape_end - tape_start);
                let score = u8::from(hint >= project_origin && hint < project_end);
                if best.is_none_or(|(best_score, _)| score > best_score) {
                    best = Some((score, mapping));
                }
            }
            let after = self.epoch.load(Ordering::Acquire);
            if before == after {
                return best.map(|(_, mapping)| mapping);
            }
        }
        None
    }
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

    /// **不变量：工程位置跟着片段走，不跟播放头走。**
    ///
    /// 【现场原形】片段起点 28.000 s（= 1 344 000 帧），宿主 `projectTimeSamples` 被
    /// VEGAS 留在 893 720（18.62 s）。映射必须答 1 344 000，而不是 893 720，也不是 0。
    #[test]
    fn project_position_follows_the_clip_not_the_playhead() {
        const RATE: i64 = 48_000;
        let cell = cell(&[segment(0, 1_581_056, 28 * RATE, 1)]);
        let mapping = cell.resolve(0, 893_720).expect("磁带第 0 帧必须能解析");
        assert_eq!(mapping.project(0), Some(28 * RATE));
        assert_eq!(mapping.project(RATE), Some(29 * RATE));
        // 宿主读数只是消歧线索，绝不参与换算。
        assert_eq!(cell.resolve(0, 0).map(|m| m.project(0)), Some(Some(28 * RATE)));
    }

    /// 没有映射时回答 `None` —— 调用方据此退回直通。
    #[test]
    fn an_empty_table_answers_none() {
        let cell = ProjectMapCell::default();
        assert!(cell.resolve(0, 0).is_none());
        assert!(cell.is_empty());
    }

    /// 落在所有片段之外时回答 `None`。
    #[test]
    fn a_tape_frame_outside_every_segment_answers_none() {
        let cell = cell(&[segment(0, 960, 5_000, 1)]);
        assert!(cell.resolve(960, 5_000).is_none());
        assert!(cell.resolve(-1, 5_000).is_none());
    }

    /// 多 take：宿主读数落在哪一片就选哪一片。
    #[test]
    fn the_hint_disambiguates_two_takes() {
        // 两个 take 的磁带帧号都从 0 起（各自一份源），工程位置不同。
        let cell = cell(&[segment(0, 1_000, 10_000, 1), segment(0, 1_000, 40_000, 2)]);
        assert_eq!(
            cell.resolve(10, 40_500).map(|m| m.project(10)),
            Some(Some(40_010))
        );
        assert_eq!(
            cell.resolve(10, 10_500).map(|m| m.project(10)),
            Some(Some(10_010))
        );
    }

    /// 重新发布之后旧读数立刻失效（代次与内容一起换）。
    #[test]
    fn republishing_replaces_the_table() {
        let cell = cell(&[segment(0, 1_000, 10_000, 1)]);
        assert_eq!(cell.revision(), 7);
        cell.publish(9, &[segment(0, 2_000, 20_000, 2)]);
        assert_eq!(cell.revision(), 9);
        assert_eq!(
            cell.resolve(1_500, 20_000).map(|m| m.project(1_500)),
            Some(Some(21_500))
        );
        // 旧表覆盖磁带 [0, 1000)；新表覆盖 [0, 2000)。2 500 帧只在新表之外 ——
        // 它必须解析不到任何东西，而不是退回旧片段的起点。
        assert!(cell.resolve(2_500, 10_000).is_none(), "旧片段必须消失");
    }

    /// 表超过容量时截断，而不是越界。
    #[test]
    fn a_table_bigger_than_the_capacity_is_truncated() {
        let segments = (0..PROJECT_MAP_SLOTS as i64 + 4)
            .map(|index| segment(index * 100, index * 100 + 100, index * 1_000, 1))
            .collect::<Vec<_>>();
        let cell = cell(&segments);
        assert_eq!(cell.len(), PROJECT_MAP_SLOTS);
        assert!(cell.resolve(PROJECT_MAP_SLOTS as i64 * 100 + 10, 0).is_none());
    }
}
