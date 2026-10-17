//! Transfer 形态的用户编辑：随 VST3 组件 state 一起存取的载荷。
//!
//! # 为什么需要一份独立载荷
//!
//! ARA 形态的编辑权威是 `DocumentSession::edits`（一份 `EditState`），而 Transfer
//! **没有文档** —— 用户的曲线直接活在本会话的 `TimelineState` 上（见
//! `SessionBacking::Transfer` 的注释）。`EditState` 的字段（`atlas` / `authorities` /
//! `copy_seeds` / `fades`）全部按 **ARA 身份**（region key、persistentID）索引，
//! Transfer 里那些身份根本不存在；硬套会让恢复走一整套"对不上就静默丢失"的路径。
//!
//! 所以这里只带**用户真正拥有的那一部分时间线**：曲线、音阶、Tempo Map、私有分组
//! 开关、以及轨道级的可编辑设置。**不带几何**（`clips` / 轨道集合）—— 几何由捕获
//! 存储重新投影：素材变了，几何就必须跟着变；把旧几何恢复回来会让片段指向不存在的音频。
//!
//! # 为什么是 JSON
//!
//! 与 Transfer 清单同一口径：它是几十 KB 的元数据，需要能被诊断工具读。
//! 二进制只换来"多写一个解析器"。

use hifishifter_kernel::state::model::{PitchAnalysisAlgo, TempoPointData, TrackParamsState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// 一条轨道里属于用户的设置。
///
/// 【为什么只挑这几个字段】其余字段（`id` / `order` / `parent_id`）是**投影**产出的
/// 几何，恢复它们等于让投影与存档打架。`compose_enabled` 与 `pitch_analysis_algo`
/// 会改变渲染结果，`muted` / `solo` / `volume` / `name` / `color` 是用户直接编辑的
/// 呈现与混音状态 —— 两类都必须跟着工程走。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TransferTrackSettings {
    #[serde(default)]
    pub compose_enabled: bool,
    #[serde(default)]
    pub pitch_analysis_algo: PitchAnalysisAlgo,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub solo: bool,
    #[serde(default = "default_volume")]
    pub volume: f32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub color: String,
}

fn default_volume() -> f32 {
    1.0
}

/// Transfer 会话里"属于用户"的那一份时间线。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct TransferEdits {
    /// 根轨道 id → 曲线（含 `pitch_edit` 与全部自动化曲线）。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params_by_root_track: BTreeMap<String, TrackParamsState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub project_scale_notes: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo_map: Option<Vec<TempoPointData>>,
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub disabled_group_ids: HashSet<String>,
    /// 轨道 id → 用户设置。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub track_settings: BTreeMap<String, TransferTrackSettings>,
}

impl TransferEdits {
    pub(crate) fn encode(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|error| format!("transfer edits encode: {error}"))
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|error| format!("transfer edits decode: {error}"))
    }
}

/// 从一条时间线抽出用户内容。
pub(crate) fn capture(timeline: &hifishifter_kernel::state::TimelineState) -> TransferEdits {
    TransferEdits {
        params_by_root_track: timeline.params_by_root_track.clone(),
        project_scale_notes: timeline.project_scale_notes.clone(),
        tempo_map: timeline.tempo_map.clone(),
        disabled_group_ids: timeline.disabled_group_ids.clone(),
        track_settings: timeline
            .tracks
            .iter()
            .map(|track| {
                (
                    track.id.clone(),
                    TransferTrackSettings {
                        compose_enabled: track.compose_enabled,
                        pitch_analysis_algo: track.pitch_analysis_algo.clone(),
                        muted: track.muted,
                        solo: track.solo,
                        volume: track.volume,
                        name: track.name.clone(),
                        color: track.color.clone(),
                    },
                )
            })
            .collect(),
    }
}

/// 把用户内容叠回一条时间线（**只覆盖**载荷里带的键）。
///
/// 【为什么是"叠加"而不是"替换"】调用点有两处：恢复（存档里的值覆盖投影的默认值）
/// 与重新加载（保留用户当前编辑）。两处都要求"载荷里没有的东西保持原样" ——
/// 替换会让一次恢复把投影刚建立的轨道集合清空。
///
/// 【为什么只有测试在用】生产路径必须让曲线经过 `ensure_params_for_root` 的规整，
/// 所以它调用 [`apply_project_state`] 并把曲线并进 `kept`（见
/// `EditorSession::load_transfer_timeline`）。这里保留完整版本，供往返测试断言
/// "两部分合起来就是全部用户内容"。
#[cfg(test)]
pub(crate) fn apply(
    timeline: &mut hifishifter_kernel::state::TimelineState,
    edits: &TransferEdits,
) {
    for (root, params) in &edits.params_by_root_track {
        timeline
            .params_by_root_track
            .insert(root.clone(), params.clone());
    }
    apply_project_state(timeline, edits);
}

/// 只叠"曲线以外"的用户内容（音阶 / Tempo Map / 私有分组 / 轨道设置）。
///
/// 【为什么要与曲线分开】`load_transfer_timeline` 必须让曲线经过
/// `ensure_params_for_root` 的规整（按工程长度取目标帧数、清掉对不上的原线缓存）。
/// 若在那里直接 `insert` 恢复的曲线，它就绕过了规整 —— 长度对不上时界面会画出一条
/// 超出工程末尾的曲线。所以那条路径把曲线并进 `kept`，只用本函数叠其余字段。
pub(crate) fn apply_project_state(
    timeline: &mut hifishifter_kernel::state::TimelineState,
    edits: &TransferEdits,
) {
    if !edits.project_scale_notes.is_empty() {
        timeline.project_scale_notes = edits.project_scale_notes.clone();
    }
    if edits.tempo_map.is_some() {
        timeline.tempo_map = edits.tempo_map.clone();
    }
    timeline
        .disabled_group_ids
        .extend(edits.disabled_group_ids.iter().cloned());
    for track in &mut timeline.tracks {
        let Some(settings) = edits.track_settings.get(&track.id) else {
            continue;
        };
        track.compose_enabled = settings.compose_enabled;
        track.pitch_analysis_algo = settings.pitch_analysis_algo.clone();
        track.muted = settings.muted;
        track.solo = settings.solo;
        track.volume = settings.volume;
        if !settings.name.is_empty() {
            track.name = settings.name.clone();
        }
        if !settings.color.is_empty() {
            track.color = settings.color.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hifishifter_kernel::state::TimelineState;

    fn timeline_with_curve() -> TimelineState {
        let mut timeline = TimelineState::default();
        timeline.tracks.clear();
        timeline.tracks.push(hifishifter_kernel::state::Track {
            id: "transfer-track".into(),
            name: "Transfer".into(),
            parent_id: None,
            order: 0,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: true,
            pitch_analysis_algo: Default::default(),
            color: String::new(),
        });
        let mut params = TrackParamsState::default();
        params.pitch_edit = vec![60.0, 62.0];
        params.pitch_edit_user_modified = true;
        timeline
            .params_by_root_track
            .insert("transfer-track".into(), params);
        timeline
    }

    /// **往返：曲线必须原样回来。**
    ///
    /// 【为什么这是回归测试】非 ARA 宿主里"保存工程后重开，参数全丢"的根因就是
    /// 组件 state 从来没序列化过这份数据。这条钉住两端。
    #[test]
    fn a_curve_round_trips_through_the_payload() {
        let timeline = timeline_with_curve();
        let bytes = capture(&timeline).encode().unwrap();
        let restored = TransferEdits::decode(&bytes).unwrap();
        let mut target = TimelineState::default();
        target.tracks = timeline.tracks.clone();
        apply(&mut target, &restored);
        assert_eq!(
            target.params_by_root_track["transfer-track"].pitch_edit,
            vec![60.0, 62.0]
        );
        assert!(target.params_by_root_track["transfer-track"].pitch_edit_user_modified);
    }

    /// **应用是叠加，不是替换。**
    ///
    /// 【为什么必须钉住】`load_transfer_timeline` 在重新投影之后应用载荷；若它把
    /// `params_by_root_track` 整体替换，投影刚建立的其它轨道曲线会被静默清掉。
    #[test]
    fn applying_a_payload_keeps_keys_it_does_not_mention() {
        let mut timeline = timeline_with_curve();
        let mut other = TrackParamsState::default();
        other.pitch_edit = vec![40.0];
        timeline
            .params_by_root_track
            .insert("another-track".into(), other);

        let mut payload_source = TimelineState::default();
        let mut only = TrackParamsState::default();
        only.pitch_edit = vec![70.0];
        payload_source
            .params_by_root_track
            .insert("transfer-track".into(), only);

        apply(&mut timeline, &capture(&payload_source));
        assert_eq!(
            timeline.params_by_root_track["transfer-track"].pitch_edit,
            vec![70.0],
            "载荷里带的键必须被覆盖"
        );
        assert_eq!(
            timeline.params_by_root_track["another-track"].pitch_edit,
            vec![40.0],
            "载荷里没带的键必须保持原样"
        );
    }

    /// 空载荷仍然可以往返：它只是一段没有键的 JSON，不是错误。
    #[test]
    fn an_empty_payload_round_trips() {
        let edits = TransferEdits::default();
        let restored = TransferEdits::decode(&edits.encode().unwrap()).unwrap();
        assert!(restored.params_by_root_track.is_empty());
        assert!(restored.track_settings.is_empty());
    }
}
