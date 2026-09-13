//! 参数编辑器「多选区 → 帧窗口」的共享描述。
//!
//! 参数编辑器（钢琴卷帘）的选区是多段的（升序、互不相交）。若干后端命令只有
//! 「单个连续时间窗」的旧接口（VocalShifter 剪贴板粘贴、MIDI 选区模式导入），
//! 这里给出统一的多段窗口语义：
//!
//! - 偏移基准 = **首段起点**（与旧单窗口一致，导入/粘贴内容整体对齐到那里）；
//! - 只有落在**任一段内**的帧可写 —— 断层保持原值，绝不填充；
//! - 无选区约束（ranges 为空）时全部可写，即旧的整体操作行为。
//!
//! 前端的对应实现是 `pianoRoll/paramSelection.beatRangesToFrameRanges`（把 beat
//! 选区换算成帧区间），语义必须保持一致。

/// 一个目标选区段（帧单位）。字段名经 `rename_all = "camelCase"` 映射到
/// 前端传入的 `startFrame` / `frameCount`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectionFrameRange {
    pub start_frame: usize,
    pub frame_count: usize,
}

/// 多选区帧窗口：偏移基准 + 可写区间集合。
#[derive(Debug, Clone, Default)]
pub struct ParamSelectionWindow {
    origin: Option<usize>,
    /// 可写区间 [start, end)，升序、互不相交、相邻已合并。
    ranges: Vec<(usize, usize)>,
}

impl ParamSelectionWindow {
    /// 由多段范围构造；未提供多段时退回旧单窗口参数（等价于一个选区段）。
    pub fn new(
        selection_ranges: Option<Vec<SelectionFrameRange>>,
        selection_start_frame: Option<usize>,
        selection_max_frames: Option<usize>,
    ) -> Self {
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        match selection_ranges {
            Some(list) => {
                for range in list {
                    let start = range.start_frame;
                    // frame_count 下钳 1：与前端 `clamp(..., 1, MAX)` 一致
                    let end = start.saturating_add(range.frame_count.max(1));
                    ranges.push((start, end));
                }
            }
            None => {
                if let Some(start) = selection_start_frame {
                    let end = match selection_max_frames {
                        Some(count) => start.saturating_add(count.max(1)),
                        None => usize::MAX,
                    };
                    ranges.push((start, end));
                }
            }
        }
        ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (start, end) in ranges {
            match merged.last_mut() {
                Some(last) if start <= last.1 => {
                    if end > last.1 {
                        last.1 = end;
                    }
                }
                _ => merged.push((start, end)),
            }
        }
        let origin = merged.first().map(|(start, _)| *start);
        Self { origin, ranges: merged }
    }

    /// 偏移基准（首段起点）；None = 不参与选区对齐。
    pub fn origin(&self) -> Option<usize> {
        self.origin
    }

    /// 是否存在选区约束。
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// 该帧是否可写（无选区约束时恒为 true）。
    pub fn allows(&self, idx: usize) -> bool {
        if self.ranges.is_empty() {
            return true;
        }
        self.ranges
            .iter()
            .any(|(start, end)| idx >= *start && idx < *end)
    }

    /// 末段末尾（含）之后的第一个帧号（写入上限）。无约束时为 None。
    pub fn end_bound(&self) -> Option<usize> {
        self.ranges.last().map(|(_, end)| *end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(list: &[(usize, usize)]) -> Option<Vec<SelectionFrameRange>> {
        Some(
            list.iter()
                .map(|(start_frame, frame_count)| SelectionFrameRange {
                    start_frame: *start_frame,
                    frame_count: *frame_count,
                })
                .collect(),
        )
    }

    #[test]
    fn no_constraint_allows_everything() {
        let window = ParamSelectionWindow::new(None, None, None);
        assert!(window.is_empty());
        assert_eq!(window.origin(), None);
        assert!(window.allows(0));
        assert!(window.allows(1_000_000));
    }

    #[test]
    fn legacy_single_window_behaves_like_one_range() {
        let window = ParamSelectionWindow::new(None, Some(100), Some(50));
        assert_eq!(window.origin(), Some(100));
        assert!(window.allows(100));
        assert!(window.allows(149));
        assert!(!window.allows(150));
        assert!(!window.allows(99));
        assert_eq!(window.end_bound(), Some(150));
    }

    #[test]
    fn multi_range_keeps_gap_unwritable() {
        let window = ParamSelectionWindow::new(ranges(&[(0, 100), (200, 100)]), None, None);
        assert_eq!(window.origin(), Some(0));
        assert!(window.allows(0));
        assert!(window.allows(99));
        // 断层：不得写入（粘贴/导入都不允许把两段连起来）
        assert!(!window.allows(100));
        assert!(!window.allows(199));
        assert!(window.allows(200));
        assert!(window.allows(299));
        assert!(!window.allows(300));
        assert_eq!(window.end_bound(), Some(300));
    }

    #[test]
    fn overlapping_and_touching_ranges_are_merged() {
        let window = ParamSelectionWindow::new(ranges(&[(200, 50), (0, 100), (100, 50)]), None, None);
        assert_eq!(window.origin(), Some(0));
        // [0,150) 与 [200,250) 各自独立；相邻的 [0,100) + [100,50) 合并
        assert!(window.allows(149));
        assert!(!window.allows(150));
        assert!(window.allows(249));
        assert_eq!(window.end_bound(), Some(250));
    }

    #[test]
    fn zero_count_range_is_clamped_to_one_frame() {
        let window = ParamSelectionWindow::new(ranges(&[(10, 0)]), None, None);
        assert!(window.allows(10));
        assert!(!window.allows(11));
    }
}
