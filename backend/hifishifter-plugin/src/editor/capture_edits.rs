//! Capture 形态的用户编辑：随 VST3 组件 state 一起存取的载荷。
//!
//! # 为什么需要一份独立载荷
//!
//! ARA 形态的编辑权威是 `DocumentSession::edits`（一份 `EditState`），而 Capture
//! **没有文档** —— 用户的曲线直接活在本会话的 `TimelineState` 上（见
//! `SessionBacking::Capture` 的注释）。`EditState` 的字段（`atlas` / `authorities` /
//! `copy_seeds` / `fades`）全部按 **ARA 身份**（region key、persistentID）索引，
//! Capture 里那些身份根本不存在；硬套会让恢复走一整套"对不上就静默丢失"的路径。
//!
//! 所以这里只带**用户真正拥有的那一部分时间线**：曲线、音阶、Tempo Map、私有分组
//! 开关、以及轨道级的可编辑设置。**不带几何**（`clips` / 轨道集合）—— 几何由捕获
//! 存储重新投影：素材变了，几何就必须跟着变；把旧几何恢复回来会让片段指向不存在的音频。
//!
//! # 为什么是 JSON
//!
//! 与 Capture 清单同一口径：它是几十 KB 的元数据，需要能被诊断工具读。
//! 二进制只换来"多写一个解析器"。

use hifishifter_kernel::state::model::{PitchAnalysisAlgo, TempoPointData, TrackParamsState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Write};

/// 压缩载荷的魔数；解码时据此区分 JSON 与 deflate。
const COMPRESS_MAGIC: &[u8; 4] = b"HFSZ";
/// 超过这个大小才压缩：小载荷压缩后的头部/CPU 开销不划算。
const COMPRESS_THRESHOLD: usize = 1024 * 1024;

/// 一条轨道里属于用户的设置。
///
/// 【为什么只挑这几个字段】其余字段（`id` / `order` / `parent_id`）是**投影**产出的
/// 几何，恢复它们等于让投影与存档打架。`compose_enabled` 与 `pitch_analysis_algo`
/// 会改变渲染结果，`muted` / `solo` / `volume` / `name` / `color` 是用户直接编辑的
/// 呈现与混音状态 —— 两类都必须跟着工程走。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CaptureTrackSettings {
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

/// Capture 会话里"属于用户"的那一份时间线。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct CaptureEdits {
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
    pub track_settings: BTreeMap<String, CaptureTrackSettings>,
}

impl CaptureEdits {
    pub(crate) fn encode(&self) -> Result<Vec<u8>, String> {
        let json = serde_json::to_vec(self)
            .map_err(|error| format!("capture edits encode: {error}"))?;
        if json.len() <= COMPRESS_THRESHOLD {
            return Ok(json);
        }
        let mut encoder = flate2::write::DeflateEncoder::new(
            Vec::with_capacity(json.len() / 3),
            flate2::Compression::fast(),
        );
        encoder
            .write_all(&json)
            .map_err(|error| format!("capture edits compress: {error}"))?;
        let compressed = encoder
            .finish()
            .map_err(|error| format!("capture edits compress: {error}"))?;
        if compressed.len() >= json.len() {
            return Ok(json);
        }
        log::info!(
            "[capture] edits payload compressed {} -> {} byte(s)",
            json.len(),
            compressed.len()
        );
        let mut out = Vec::with_capacity(COMPRESS_MAGIC.len() + compressed.len());
        out.extend_from_slice(COMPRESS_MAGIC);
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let json = if bytes.starts_with(COMPRESS_MAGIC) {
            let mut decoder = flate2::read::DeflateDecoder::new(&bytes[COMPRESS_MAGIC.len()..]);
            let mut out = Vec::new();
            decoder
                .read_to_end(&mut out)
                .map_err(|error| format!("capture edits decompress: {error}"))?;
            out
        } else {
            bytes.to_vec()
        };
        let mut edits: Self = serde_json::from_slice(&json)
            .map_err(|error| format!("capture edits decode: {error}"))?;
        // 【为什么解码时就归一化】旧版本把会话命名空间写进了键。解码是唯一能
        // 同时覆盖"新会话读取旧载荷"与"旧会话读取旧载荷"的位置。
        edits.stabilize();
        Ok(edits)
    }

    /// 去掉所有会话命名空间前缀，得到与进程/会话无关的稳定键。
    pub(crate) fn stabilize(&mut self) {
        stabilize_map(&mut self.params_by_root_track);
        stabilize_map(&mut self.track_settings);
    }

    /// 把稳定键绑定到**当前会话**的命名空间。
    ///
    /// 【为什么必须做】`load_capture_timeline` 会把投影时间线的 id 改写成当前
    /// 命名空间；载荷里的键如果不跟着改，`kept.get(root)` 永远 miss，曲线被静默丢弃。
    pub(crate) fn bind_to_namespace(&mut self, namespace: &str) {
        bind_map(&mut self.params_by_root_track, namespace);
        bind_map(&mut self.track_settings, namespace);
        // `disabled_group_ids` 不是轨道 id，`rewrite_ids` 从不改写它们 —— 保持原样。
    }
}

/// 从一条时间线抽出用户内容。`namespace` 是当前会话的临时前缀，写盘前剥掉。
pub(crate) fn capture(
    timeline: &hifishifter_kernel::state::TimelineState,
    namespace: &str,
) -> CaptureEdits {
    let mut params_by_root_track = BTreeMap::new();
    for (root, params) in &timeline.params_by_root_track {
        if let Some(compact) = compact_params(params) {
            params_by_root_track.insert(root.clone(), compact);
        }
    }
    let mut edits = CaptureEdits {
        params_by_root_track,
        project_scale_notes: timeline.project_scale_notes.clone(),
        tempo_map: timeline.tempo_map.clone(),
        disabled_group_ids: timeline.disabled_group_ids.clone(),
        track_settings: timeline
            .tracks
            .iter()
            .map(|track| {
                (
                    track.id.clone(),
                    CaptureTrackSettings {
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
    };
    // 先按当前命名空间精确剥离，再用通用模式兜住历史载荷。
    strip_namespace_map(&mut edits.params_by_root_track, namespace);
    strip_namespace_map(&mut edits.track_settings, namespace);
    edits.stabilize();
    edits
}

/// 只保留**用户数据**：分析产物（`pitch_orig` / `tension_orig` / `dyn_orig`）全部丢弃，
/// 未编辑的默认曲线也丢弃。
///
/// 【为什么派生数组不能存】它们由源 PCM 重新分析得到，长工程里比用户曲线大一个数量级；
/// 组件 state 有 64 MiB 硬上限，把它们塞进去会让保存直接失败。
fn compact_params(params: &TrackParamsState) -> Option<TrackParamsState> {
    let mut compact = params.clone();
    compact.pitch_orig.clear();
    compact.tension_orig.clear();
    compact.dyn_orig.clear();
    compact.pitch_orig_key = None;
    compact.dyn_orig_key = None;
    compact.dyn_orig_source_key = None;
    compact.pending_pitch_offset = None;
    sanitize_curve(&mut compact.pitch_edit);
    sanitize_curve(&mut compact.tension_edit);
    for curve in compact.extra_curves.values_mut() {
        sanitize_curve(curve);
    }
    compact.extra_curves.retain(|_, curve| !curve.is_empty());
    for value in compact.extra_params.values_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    // `pitch_edit_user_modified` 为假 ⇒ 曲线是分析派生的默认值，恢复时会重建。
    if !params.pitch_edit_user_modified {
        compact.pitch_edit.clear();
    }
    // tension 没有 user_modified 标志：默认值是 0.0（见 `state::model` 的 resize）。
    if compact.tension_edit.iter().all(|value| *value == 0.0) {
        compact.tension_edit.clear();
    }
    (!compact.is_empty_project_data()).then_some(compact)
}

/// 非有限样本会让 `serde_json` 写成 `null`，恢复端按 `Vec<f32>` 解析失败。
/// 在写盘前清洗，把"一次分析异常"降级成"几个样本归零"。
fn sanitize_curve(curve: &mut [f32]) {
    for sample in curve.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
}

fn strip_namespace_map<V>(map: &mut BTreeMap<String, V>, namespace: &str) {
    if namespace.is_empty() {
        return;
    }
    let old = std::mem::take(map);
    for (key, value) in old {
        let stable = key.strip_prefix(namespace).unwrap_or(&key).to_owned();
        map.insert(stable, value);
    }
}

fn stabilize_map<V>(map: &mut BTreeMap<String, V>) {
    if map.is_empty() {
        return;
    }
    let old = std::mem::take(map);
    for (key, value) in old {
        map.insert(strip_session_namespace(&key).to_owned(), value);
    }
}

fn bind_map<V>(map: &mut BTreeMap<String, V>, namespace: &str) {
    if map.is_empty() || namespace.is_empty() {
        return;
    }
    let old = std::mem::take(map);
    for (key, value) in old {
        map.insert(
            format!("{namespace}{}", strip_session_namespace(&key)),
            value,
        );
    }
}

/// `hfs-ui-{pid}-{n}-{rest}` → `{rest}`；不是这个形状时原样返回。
///
/// 【为什么不用 `split('-')`】轨道 id 本身可能含 `-`；`splitn(3)` 只吃掉两个数字段，
/// 其余（含连字符）完整保留。
fn strip_session_namespace(key: &str) -> &str {
    let Some(rest) = key.strip_prefix("hfs-ui-") else {
        return key;
    };
    let mut parts = rest.splitn(3, '-');
    let pid = parts.next().unwrap_or_default();
    let seq = parts.next().unwrap_or_default();
    let Some(tail) = parts.next() else {
        return key;
    };
    let numeric = |value: &str| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit());
    if numeric(pid) && numeric(seq) {
        tail
    } else {
        key
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
/// `EditorSession::load_capture_timeline`）。这里保留完整版本，供往返测试断言
/// "两部分合起来就是全部用户内容"。
#[cfg(test)]
pub(crate) fn apply(
    timeline: &mut hifishifter_kernel::state::TimelineState,
    edits: &CaptureEdits,
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
/// 【为什么要与曲线分开】`load_capture_timeline` 必须让曲线经过
/// `ensure_params_for_root` 的规整（按工程长度取目标帧数、清掉对不上的原线缓存）。
/// 若在那里直接 `insert` 恢复的曲线，它就绕过了规整 —— 长度对不上时界面会画出一条
/// 超出工程末尾的曲线。所以那条路径把曲线并进 `kept`，只用本函数叠其余字段。
pub(crate) fn apply_project_state(
    timeline: &mut hifishifter_kernel::state::TimelineState,
    edits: &CaptureEdits,
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
            id: "capture-track".into(),
            name: "Capture".into(),
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
            .insert("capture-track".into(), params);
        timeline
    }

    /// **往返：曲线必须原样回来。**
    ///
    /// 【为什么这是回归测试】非 ARA 宿主里"保存工程后重开，参数全丢"的根因就是
    /// 组件 state 从来没序列化过这份数据。这条钉住两端。
    #[test]
    fn a_curve_round_trips_through_the_payload() {
        let timeline = timeline_with_curve();
        let bytes = capture(&timeline, "").encode().unwrap();
        let restored = CaptureEdits::decode(&bytes).unwrap();
        let mut target = TimelineState::default();
        target.tracks = timeline.tracks.clone();
        apply(&mut target, &restored);
        assert_eq!(
            target.params_by_root_track["capture-track"].pitch_edit,
            vec![60.0, 62.0]
        );
        assert!(target.params_by_root_track["capture-track"].pitch_edit_user_modified);
    }

    /// **曲线必须跨会话命名空间存活。**
    ///
    /// 【为什么这是回归测试】`CaptureEdits` 的键曾经带 `hfs-ui-{pid}-{n}-` 前缀；
    /// 重开工程后新会话的前缀不同，`load_capture_timeline` 的 `kept.get(root)`
    /// 永远 miss，曲线被静默丢弃 —— 用户看到的就是"完全未编辑"。
    #[test]
    fn curves_survive_a_new_session_namespace() {
        let mut timeline = timeline_with_curve();
        let old_key = "hfs-ui-1234-1-capture-track".to_string();
        timeline.tracks[0].id = old_key.clone();
        let params = timeline
            .params_by_root_track
            .remove("capture-track")
            .unwrap();
        timeline.params_by_root_track.insert(old_key, params);

        let bytes = capture(&timeline, "hfs-ui-1234-1-").encode().unwrap();
        let mut restored = CaptureEdits::decode(&bytes).unwrap();
        assert!(
            restored.params_by_root_track.contains_key("capture-track"),
            "载荷必须写稳定键"
        );

        restored.bind_to_namespace("hfs-ui-1234-2-");
        assert!(
            restored
                .params_by_root_track
                .contains_key("hfs-ui-1234-2-capture-track"),
            "恢复时必须绑定到当前会话的命名空间"
        );
    }

    /// **旧载荷里的命名空间前缀在解码时被剥掉。**
    #[test]
    fn legacy_namespaced_keys_are_normalized_on_decode() {
        let mut payload = CaptureEdits::default();
        payload.params_by_root_track.insert(
            "hfs-ui-9-9-capture-track".into(),
            TrackParamsState::default(),
        );
        let decoded = CaptureEdits::decode(&payload.encode().unwrap()).unwrap();
        assert!(decoded
            .params_by_root_track
            .contains_key("capture-track"));
    }

    /// **派生分析数组不落盘。**
    #[test]
    fn derived_analysis_arrays_are_not_persisted() {
        let mut timeline = timeline_with_curve();
        let params = timeline
            .params_by_root_track
            .get_mut("capture-track")
            .unwrap();
        params.pitch_orig = vec![60.0; 1024];
        params.tension_orig = vec![0.5; 1024];
        // 只有派生数据、没有用户编辑的轨道：整条记录都不该落盘。
        let mut derived_only = TrackParamsState::default();
        derived_only.pitch_orig = vec![60.0; 1024];
        timeline
            .params_by_root_track
            .insert("derived-only".into(), derived_only);

        let edits = capture(&timeline, "");
        let stored = &edits.params_by_root_track["capture-track"];
        assert!(stored.pitch_orig.is_empty(), "pitch_orig 是分析产物");
        assert!(stored.tension_orig.is_empty(), "tension_orig 是分析产物");
        assert_eq!(
            stored.pitch_edit,
            vec![60.0, 62.0],
            "用户编辑的曲线必须保留"
        );
        assert!(
            !edits.params_by_root_track.contains_key("derived-only"),
            "只有派生数据的轨道不该落盘"
        );
    }

    /// **非有限样本在写盘前被清洗。**
    #[test]
    fn non_finite_curve_samples_are_sanitized() {
        let mut timeline = timeline_with_curve();
        let params = timeline
            .params_by_root_track
            .get_mut("capture-track")
            .unwrap();
        params.pitch_edit = vec![f32::NAN, f32::INFINITY, 62.0];
        params.pitch_edit_user_modified = true;
        let edits = capture(&timeline, "");
        let stored = &edits.params_by_root_track["capture-track"].pitch_edit;
        assert!(stored.iter().all(|value| value.is_finite()));
        assert_eq!(stored[2], 62.0);
        // 清洗后的载荷必须能真正往返（NaN 写成 null 会让恢复端解析失败）。
        let bytes = edits.encode().unwrap();
        CaptureEdits::decode(&bytes).unwrap();
    }

    /// **大载荷压缩后仍能原样恢复。**
    #[test]
    fn a_large_payload_is_compressed_and_round_trips() {
        let mut timeline = timeline_with_curve();
        let params = timeline
            .params_by_root_track
            .get_mut("capture-track")
            .unwrap();
        params.pitch_edit = (0..400_000).map(|index| 60.0 + index as f32 * 0.0001).collect();
        params.pitch_edit_user_modified = true;
        let bytes = capture(&timeline, "").encode().unwrap();
        assert!(
            bytes.starts_with(COMPRESS_MAGIC),
            "超过 1 MiB 的载荷必须压缩"
        );
        let restored = CaptureEdits::decode(&bytes).unwrap();
        let curve = &restored.params_by_root_track["capture-track"].pitch_edit;
        assert_eq!(curve.len(), 400_000);
        assert!((curve[399_999] - (60.0 + 399_999.0 * 0.0001)).abs() < 1e-3);
    }

    /// **应用是叠加，不是替换。**
    ///
    /// 【为什么必须钉住】`load_capture_timeline` 在重新投影之后应用载荷；若它把
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
        only.pitch_edit_user_modified = true;
        payload_source
            .params_by_root_track
            .insert("capture-track".into(), only);

        apply(&mut timeline, &capture(&payload_source, ""));
        assert_eq!(
            timeline.params_by_root_track["capture-track"].pitch_edit,
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
        let edits = CaptureEdits::default();
        let restored = CaptureEdits::decode(&edits.encode().unwrap()).unwrap();
        assert!(restored.params_by_root_track.is_empty());
        assert!(restored.track_settings.is_empty());
    }
}
