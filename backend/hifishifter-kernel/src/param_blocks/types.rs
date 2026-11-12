//! 参数块的源域身份与用户编辑覆盖层。
//!
//! 职责：定义 Take 源域锚点、用户声明式块、以及派生视图的稳定类型。
//! 关系：`projection` 只消费本文件的源域类型；`compose` 产出 `ParamBlockView`；
//! `editing` 只改写 `ParamBlockEditSet`。维护约束：任何“块在工程上的绝对帧”
//! 都不得出现在 `UserParamBlock` 中，绝对帧只能是 `ParamBlockView` 的派生结果。

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

fn is_false(value: &bool) -> bool {
    !*value
}

/// 一个 Take 在工程中的稳定出现身份。
///
/// 【身份规则】`clip_id` 是项目内稳定的 Clip 身份，不是宿主运行时命名空间；
/// `take_id` 是内容身份。两者必须同时存在：同一个 Take 可以在多个 Clip 出现，
/// 同一个 Clip 也可以切换 Take。
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TakeKey {
    pub clip_id: String,
    pub take_id: String,
}

/// 一个源域区间，属于某个 Take 的某个循环周期。
///
/// 【坐标】`source_start_sec` / `source_end_sec` 是源媒体秒，绝不存工程秒。
/// `cycle = 0` 表示非 Loop Clip 的唯一周期；Loop 下每个周期一个具体值。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TakeSpan {
    pub clip_id: String,
    pub take_id: String,
    pub cycle: u32,
    pub source_start_sec: f64,
    pub source_end_sec: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syllable_id: Option<String>,
    pub label: BlockLabel,
}

impl TakeSpan {
    pub fn key(&self) -> TakeKey {
        TakeKey {
            clip_id: self.clip_id.clone(),
            take_id: self.take_id.clone(),
        }
    }

    pub fn width_sec(&self) -> f64 {
        (self.source_end_sec - self.source_start_sec).max(0.0)
    }

    pub fn overlaps(&self, other: &TakeSpan) -> bool {
        self.source_start_sec < other.source_end_sec - 1e-12
            && other.source_start_sec < self.source_end_sec - 1e-12
    }
}

/// 块的语义标签，只用于 UI 与调试，不参与合成优先级。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BlockLabel {
    Syllable,
    Phoneme,
    MidiNote,
    GameBaseline,
    Manual,
}

/// 没有 Take 来源的自由块。只有真正画在空白处的块才允许使用它。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RootSpan {
    pub start_sec: f64,
    pub end_sec: f64,
}

impl RootSpan {
    pub fn width_sec(&self) -> f64 {
        (self.end_sec - self.start_sec).max(0.0)
    }
}

/// 用户块的几何：Take 源域成员，或真正的空白自由块。
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum BlockGeometry {
    Take {
        members: Vec<TakeSpan>,
    },
    Root {
        span: RootSpan,
    },
    /// 一等公民的组容器：children 是唯一几何真值，组自身不再保存合并成员或快照。
    Group {
        children: Vec<UserParamBlock>,
    },
}

/// 用户块对基块的“所有权覆盖”。
///
/// - `full = true`：这些 span 命中完整基块时，隐藏整个基块；即使可见几何被
///   缩短，缩短掉的部分也保持空白；
/// - `full = false`：只覆盖这些 span 自身的投影区间（基块片段）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BaseCover {
    pub spans: Vec<TakeSpan>,
    pub full: bool,
}

/// 用户声明式块。
///
/// 【单一事实】用户只声明“哪些源域区间属于同一个块”。最终显示区间、成员投影、
/// 过渡宽度全部由 `compose` 派生，不在本类型中保存绝对帧。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", from = "UserParamBlockWire")]
pub struct UserParamBlock {
    /// 稳定用户块 id，创建后不再改变。
    pub id: String,
    pub geometry: BlockGeometry,
    /// 删除墓碑：只抑制 base，不产生显示块。
    #[serde(default, skip_serializing_if = "is_false")]
    pub deleted: bool,
    /// 手动画的块；用于 UI 菜单与 value 语义。
    #[serde(default, skip_serializing_if = "is_false")]
    pub manual: bool,
    /// 手动块/Pitch 块的逻辑中心；其他参数通常为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// 本块右边界与下一块的圆滑宽度，源域秒。
    /// 只对 Take 块有意义；Root 块用工程秒解释。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_out_sec: Option<f64>,
    /// 同一 revision 内创建多个块时的确定性排序键。
    #[serde(default)]
    pub created_seq: u64,
    /// 从基块派生时记录“我替代了哪一段基块”。普通手工块为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_cover: Option<BaseCover>,
}

impl UserParamBlock {
    pub fn members(&self) -> &[TakeSpan] {
        match &self.geometry {
            BlockGeometry::Take { members } => members,
            BlockGeometry::Root { .. } | BlockGeometry::Group { .. } => &[],
        }
    }

    pub fn is_take(&self) -> bool {
        matches!(self.geometry, BlockGeometry::Take { .. })
    }

    pub fn children(&self) -> Option<&[UserParamBlock]> {
        match &self.geometry {
            BlockGeometry::Group { children } => Some(children),
            _ => None,
        }
    }

    pub fn is_group(&self) -> bool {
        matches!(self.geometry, BlockGeometry::Group { .. })
    }

    /// 组内拍平。组创建时已禁止嵌套；这里递归只是防御旧数据。
    pub fn flatten_children(&self) -> Vec<&UserParamBlock> {
        let mut out = Vec::new();
        fn visit<'a>(block: &'a UserParamBlock, out: &mut Vec<&'a UserParamBlock>) {
            match &block.geometry {
                BlockGeometry::Group { children } => {
                    for child in children {
                        visit(child, out);
                    }
                }
                _ => out.push(block),
            }
        }
        if let BlockGeometry::Group { children } = &self.geometry {
            for child in children {
                visit(child, &mut out);
            }
        }
        out
    }

    /// Group 递归收集全部叶子成员。
    pub fn leaf_members(&self) -> Vec<TakeSpan> {
        self.flatten_children()
            .into_iter()
            .flat_map(|child| child.members().iter().cloned())
            .collect()
    }

    /// 所有 child 的 value 完全一致时才返回该值。
    pub fn group_common_value(&self) -> Option<f64> {
        let children = self.flatten_children();
        let first = children.first()?.value?;
        children
            .iter()
            .all(|child| child.value == Some(first))
            .then_some(first)
    }
}

/// 线协议 v1 -> v2 的一次性加载规整。
///
/// v1 的组是 “合并 geometry + groupedFrom 快照” 的双重表示；v2 只认
/// `BlockGeometry::Group { children }`。这里把旧快照转成容器后立即丢弃旧字段，
/// 运行期不再存在第二份几何真值。
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct UserParamBlockWire {
    id: String,
    geometry: BlockGeometryWire,
    #[serde(default)]
    deleted: bool,
    #[serde(default)]
    manual: bool,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    transition_out_sec: Option<f64>,
    #[serde(default)]
    created_seq: u64,
    #[serde(default)]
    base_cover: Option<BaseCover>,
    #[serde(default)]
    grouped_from: Option<Vec<UserParamBlockWire>>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum BlockGeometryWire {
    Take { members: Vec<TakeSpan> },
    Root { span: RootSpan },
    Group { children: Vec<UserParamBlockWire> },
}

impl From<BlockGeometryWire> for BlockGeometry {
    fn from(value: BlockGeometryWire) -> Self {
        match value {
            BlockGeometryWire::Take { members } => Self::Take { members },
            BlockGeometryWire::Root { span } => Self::Root { span },
            BlockGeometryWire::Group { children } => Self::Group {
                children: children.into_iter().map(UserParamBlock::from).collect(),
            },
        }
    }
}

impl From<UserParamBlockWire> for UserParamBlock {
    fn from(value: UserParamBlockWire) -> Self {
        let geometry = match value.grouped_from {
            Some(children) if children.len() >= 2 => BlockGeometry::Group {
                children: children.into_iter().map(UserParamBlock::from).collect(),
            },
            _ => value.geometry.into(),
        };
        Self {
            id: value.id,
            geometry,
            deleted: value.deleted,
            manual: value.manual,
            value: value.value,
            transition_out_sec: value.transition_out_sec,
            created_seq: value.created_seq,
            base_cover: value.base_cover,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ParamBlockEditSetWire {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    blocks: Vec<UserParamBlockWire>,
}

impl From<ParamBlockEditSetWire> for ParamBlockEditSet {
    fn from(value: ParamBlockEditSetWire) -> Self {
        // v1 与缺失版本都在加载时规整为当前版本；运行期只认 2。
        let _ = value.schema_version;
        Self {
            schema_version: 2,
            revision: value.revision,
            blocks: value.blocks.into_iter().map(UserParamBlock::from).collect(),
        }
    }
}

/// 一个 root 参数线的用户编辑覆盖层。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "camelCase",
    deny_unknown_fields,
    from = "ParamBlockEditSetWire"
)]
pub struct ParamBlockEditSet {
    pub schema_version: u32,
    pub revision: u64,
    #[serde(default)]
    pub blocks: Vec<UserParamBlock>,
}

impl Default for ParamBlockEditSet {
    fn default() -> Self {
        Self {
            schema_version: 2,
            revision: 0,
            blocks: Vec::new(),
        }
    }
}

impl ParamBlockEditSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        super::validate::validate_edit_set(self)
    }

    pub fn next_seq(&self) -> u64 {
        self.blocks
            .iter()
            .map(|block| block.created_seq)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    /// 整体替换用户块，revision 只增不减。
    pub fn replace(
        &self,
        expected_revision: u64,
        blocks: Vec<UserParamBlock>,
    ) -> Result<Self, String> {
        if self.revision != expected_revision {
            return Err("Conflict: parameter block edits changed".into());
        }
        let mut next = self.clone();
        next.blocks = blocks;
        next.blocks
            .sort_by(|a, b| a.created_seq.cmp(&b.created_seq).then(a.id.cmp(&b.id)));
        next.revision = self.revision.saturating_add(1);
        next.validate()?;
        Ok(next)
    }

    /// 从历史快照恢复，revision 由宿主重新分配更大的值。
    pub fn restore_snapshot(snapshot: &ParamBlockEditSet, current_revision: u64) -> Self {
        let mut restored = snapshot.clone();
        restored.revision = current_revision.saturating_add(1);
        restored
    }

    /// 清理指向已不存在 Take 的成员。返回被清理的块数；orphan 默认保留，
    /// 只有调用方明确要求“删除无效块”时才使用本函数。
    ///
    /// Group 递归清理子块；空组删除，只剩一个子块时自然退化为普通块。
    pub fn prune_missing_takes(&mut self, allowed: &BTreeSet<TakeKey>) -> usize {
        fn prune_block(
            block: &UserParamBlock,
            allowed: &BTreeSet<TakeKey>,
        ) -> Option<UserParamBlock> {
            let mut next = block.clone();
            match &next.geometry {
                BlockGeometry::Group { children } => {
                    let kept: Vec<UserParamBlock> = children
                        .iter()
                        .filter_map(|child| prune_block(child, allowed))
                        .collect();
                    if kept.is_empty() {
                        return None;
                    }
                    if kept.len() == 1 {
                        return kept.into_iter().next();
                    }
                    next.geometry = BlockGeometry::Group { children: kept };
                    Some(next)
                }
                BlockGeometry::Take { members } => {
                    let kept: Vec<TakeSpan> = members
                        .iter()
                        .filter(|member| allowed.contains(&member.key()))
                        .cloned()
                        .collect();
                    if kept.is_empty() {
                        return None;
                    }
                    next.geometry = BlockGeometry::Take { members: kept };
                    Some(next)
                }
                BlockGeometry::Root { .. } => Some(next),
            }
        }

        let before = self.blocks.len();
        self.blocks = self
            .blocks
            .iter()
            .filter_map(|block| prune_block(block, allowed))
            .collect();
        before.saturating_sub(self.blocks.len())
    }
}

/// 派生视图。它永远不是权威，App 保存前必须清空。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamBlockView {
    pub schema_version: u32,
    pub root_track_id: String,
    pub edit_revision: u64,
    pub layout_key: String,
    pub frame_period_ms: f64,
    pub blocks: Vec<ProjectedParamBlock>,
    #[serde(default)]
    pub orphan_count: usize,
    #[serde(default)]
    pub analysis_pending: bool,
}

impl ParamBlockView {
    pub fn empty(root: &str, frame_period_ms: f64) -> Self {
        Self {
            schema_version: 1,
            root_track_id: root.to_owned(),
            edit_revision: 0,
            layout_key: String::new(),
            frame_period_ms,
            blocks: Vec::new(),
            orphan_count: 0,
            analysis_pending: false,
        }
    }
}

/// 一个投影后的可见块。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectedParamBlock {
    pub id: String,
    /// 派生视图块所属的用户块 id。base 块没有；多 run 用户块的每个片段都指向父块。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_block_id: Option<String>,
    pub start_frame: u32,
    pub end_frame_exclusive: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_frames: Option<u32>,
    pub kind: ProjectedBlockKind,
    pub members: Vec<ProjectedMember>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Read-only source note for single-member blocks; never persisted in edits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_note: Option<f32>,
    /// `clipId|takeId|cycle` identity for cross-source adjacency checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    /// Visible interval starts/ends inside its source span.
    #[serde(default, skip_serializing_if = "is_false")]
    pub cut_start: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub cut_end: bool,
    /// 基块候选的所有权覆盖；用户块为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_cover: Option<BaseCover>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectedMember {
    pub clip_id: String,
    pub take_id: String,
    pub cycle: u32,
    pub source_start_sec: f64,
    pub source_end_sec: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syllable_id: Option<String>,
    /// 成员的语义标签；编辑回写时保留。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<BlockLabel>,
    pub start_frame: u32,
    pub end_frame_exclusive: u32,
    /// 该可见帧区间起点 / 终点对应的源域秒（有方向；倒放时 start > end）。
    pub frame_source_start_sec: f64,
    pub frame_source_end_sec: f64,
    /// 可见区间覆盖的源域范围（source 顺序，供 BaseCover / 片段物化使用）。
    pub visible_source_start_sec: f64,
    pub visible_source_end_sec: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub cut_start: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub cut_end: bool,
}

/// 视图块的编辑能力标签。
///
/// 【@BT@Grouped@BT@ 只有一条语义】可逆编组（@BT@UserParamBlock::grouped_from@BT@ 非空）：
/// 它的几何是"若干成员 + 一份快照"的联合，横向切开 / 伸缩没有唯一解，必须先取消
/// 编组（内核以 @BT@grouped_locked@BT@ 拒绝）。
///
/// 它与"这一 run 里有几个成员"**无关**。多成员 run（手工跨两条相邻 Clip 拖出来的
/// 块、或对 2 成员块切一刀之后的左右两半）只是几何事实，不影响编辑能力：切分有
/// 唯一解（找到包含切点的那个成员按源域一分为二），伸缩也有。把它标成 @BT@Grouped@BT@
/// 会让 UI 静默禁掉一条本来正确的路径（"播放光标确实在块内，却切不了"）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProjectedBlockKind {
    Base,
    User,
    Grouped,
    Manual,
}

/// GAME 初始分段基线：源域快照，只有用户显式刷新才替换。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameBlockBaseline {
    pub revision: u64,
    pub algorithm: String,
    pub algorithm_version: u32,
    pub source_spans_sec: Vec<crate::pitch_analysis::SyllableSpan>,
    /// Union of the source windows already represented by `source_spans_sec`.
    pub analyzed_window_sec: [f64; 2],
}

impl GameBlockBaseline {
    /// A reanalysis of a wider window extends the frozen baseline instead of
    /// discarding spans that are still valid; an algorithm change replaces it.
    pub fn merge_window(
        &mut self,
        algorithm: &str,
        algorithm_version: u32,
        window: [f64; 2],
        spans: &[crate::pitch_analysis::SyllableSpan],
    ) {
        if self.algorithm != algorithm
            || self.algorithm_version != algorithm_version
            || self.source_spans_sec.is_empty()
        {
            self.algorithm = algorithm.to_owned();
            self.algorithm_version = algorithm_version;
            self.source_spans_sec = spans.to_vec();
            self.analyzed_window_sec = window;
            return;
        }
        let prior = self.analyzed_window_sec;
        if window[0] >= prior[0] - 1e-6 && window[1] <= prior[1] + 1e-6 {
            return;
        }
        for span in spans {
            let duplicate = self.source_spans_sec.iter().any(|existing| {
                (existing.start_sec - span.start_sec).abs() <= 1e-9
                    && (existing.end_sec - span.end_sec).abs() <= 1e-9
            });
            if !duplicate {
                self.source_spans_sec.push(span.clone());
            }
        }
        self.source_spans_sec.sort_by(|a, b| {
            a.start_sec
                .total_cmp(&b.start_sec)
                .then(a.end_sec.total_cmp(&b.end_sec))
        });
        self.analyzed_window_sec = [prior[0].min(window[0]), prior[1].max(window[1])];
    }
}

/// Update the optional frozen baseline for one completed GAME analysis.
/// Kept in the kernel so App and plugin share the same window-merge policy.
pub fn merge_game_baseline(
    baseline: &mut Option<GameBlockBaseline>,
    algorithm: &str,
    algorithm_version: u32,
    window: [f64; 2],
    spans: &[crate::pitch_analysis::SyllableSpan],
) {
    let entry = baseline.get_or_insert_with(|| GameBlockBaseline {
        revision: 1,
        algorithm: algorithm.to_owned(),
        algorithm_version,
        source_spans_sec: Vec::new(),
        analyzed_window_sec: window,
    });
    entry.merge_window(algorithm, algorithm_version, window, spans);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_edit_set_always_serializes_blocks_array() {
        let value = serde_json::to_value(ParamBlockEditSet::default()).unwrap();
        assert_eq!(value["blocks"], serde_json::json!([]));
    }
}
