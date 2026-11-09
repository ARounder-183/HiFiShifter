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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum BlockGeometry {
    Take { members: Vec<TakeSpan> },
    Root { span: RootSpan },
}

/// 用户声明式块。
///
/// 【单一事实】用户只声明“哪些源域区间属于同一个块”。最终显示区间、成员投影、
/// 过渡宽度全部由 `compose` 派生，不在本类型中保存绝对帧。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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
}

impl UserParamBlock {
    pub fn members(&self) -> &[TakeSpan] {
        match &self.geometry {
            BlockGeometry::Take { members } => members,
            BlockGeometry::Root { .. } => &[],
        }
    }

    pub fn is_take(&self) -> bool {
        matches!(self.geometry, BlockGeometry::Take { .. })
    }
}

/// 一个 root 参数线的用户编辑覆盖层。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ParamBlockEditSet {
    pub schema_version: u32,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<UserParamBlock>,
}

impl Default for ParamBlockEditSet {
    fn default() -> Self {
        Self {
            schema_version: 1,
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
    pub fn prune_missing_takes(&mut self, allowed: &BTreeSet<TakeKey>) -> usize {
        let before = self.blocks.len();
        self.blocks.retain(|block| match &block.geometry {
            BlockGeometry::Take { members } => {
                members.iter().any(|member| allowed.contains(&member.key()))
            }
            BlockGeometry::Root { .. } => true,
        });
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
    pub start_frame: u32,
    pub end_frame_exclusive: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_frames: Option<u32>,
    pub kind: ProjectedBlockKind,
    pub members: Vec<ProjectedMember>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
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
    pub start_frame: u32,
    pub end_frame_exclusive: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProjectedBlockKind {
    Base,
    User,
    Merged,
    Manual,
}

/// GAME 初始分段基线：源域快照，只有用户显式刷新才替换。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameBlockBaseline {
    pub revision: u64,
    pub algorithm: String,
    pub algorithm_version: u32,
    pub source_spans_sec: Vec<crate::pitch_analysis::SyllableSpan>,
}

impl Default for GameBlockBaseline {
    fn default() -> Self {
        Self {
            revision: 0,
            algorithm: String::new(),
            algorithm_version: 0,
            source_spans_sec: Vec::new(),
        }
    }
}
