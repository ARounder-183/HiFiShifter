//! 插件侧编辑状态；客户端只能提交参数，宿主的clip几何始终权威。
use hifishifter_kernel::state::{TimelineState, Track, TrackParamsState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// 会话轨道键只用于live关联；值完全来自宿主的modification/source persistentID。
pub(crate) type TrackBindings = BTreeMap<String, Vec<(String, String)>>;

/// 插件自有、宿主表达不到的**逐片段**属性。
///
/// 【为什么需要独立存储】clip 颜色与共振峰变形在宿主里没有对应概念（宿主只报告几何与
/// 媒体），而 `session.timeline` 每次 `ensure_loaded` 都会被整体替换。只有落在"有权威
/// 来源、且在共同投影入口重新播种"的地方，才不会被下一次重建冲掉 —— 这正是 `fades`
/// 用同一手法解决的同一个问题（见 [`EditState::fades`] 与 [`EditState::apply`]）。
///
/// 【键为什么是 clip id】`apply` 收到的时间线里 clip id 形如 `ara-item-{item GUID}`，
/// 即宿主的稳定 item 身份；与 `fades` 的键口径一致，冷重建后仍能对上。
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ClipLocalState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// 共振峰变形是**渲染输入**（内核 DSP 读它），所以必须随片段一起播种，
    /// 否则重新加载后声音会退回未变形的版本。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formant_morph: Option<hifishifter_kernel::state::ClipFormantMorph>,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub(crate) struct EditState {
    pub revision: u64,
    pub params: BTreeMap<String, TrackParamsState>,
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub bindings: TrackBindings,
    #[serde(
        default,
        skip_serializing_if = "crate::editor::parameter_atlas::ParameterAtlas::is_empty"
    )]
    pub atlas: crate::editor::parameter_atlas::ParameterAtlas,
    /// REAPER稳定item GUID对应的HFS自有形状；不写回宿主shape/c/S字段。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fades: BTreeMap<String, crate::fade::FadeStyle>,
    /// 插件自有、宿主表达不到的逐片段属性（颜色 / 共振峰变形）。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub clip_local: BTreeMap<String, ClipLocalState>,
    /// 跨组件共享的插件私有参数分组；GUID关系不参与REAPER folder或音频路由。
    #[serde(
        default,
        skip_serializing_if = "crate::editor::private_groups::TrackGroups::is_empty"
    )]
    pub groups: crate::editor::private_groups::TrackGroups,
    /// 组件恢复时延迟到完整ARA图ready后再关联，禁止逐对象创建时误套新序号。
    #[serde(skip)]
    pub needs_rebind: bool,
}

impl EditState {
    /// 形状记录有界且只允许固定GUID键，冷重开不能按会话clip序号关联。
    fn validate_fades(&self) -> Result<(), String> {
        if self.fades.len() > 10000 {
            return Err("fade state budget exceeded".into());
        }
        for (key, style) in &self.fades {
            if key.len() != 38 || !key.starts_with('{') || !key.ends_with('}') {
                return Err("invalid fade item identity".into());
            }
            style.validate()?;
        }
        Ok(())
    }
    /// 逐片段自有属性同样有界：数量与颜色长度都设上限，避免不可信 state 撑爆内存。
    fn validate_clip_local(&self) -> Result<(), String> {
        if self.clip_local.len() > 10000 {
            return Err("clip local state budget exceeded".into());
        }
        for local in self.clip_local.values() {
            if local
                .color
                .as_deref()
                .is_some_and(|color| color.len() > 64 || color.contains('\0'))
            {
                return Err("invalid clip color".into());
            }
        }
        Ok(())
    }
    /// live图已明确归属时刷新成员；restore只接受唯一完整身份集合，失败保留原state。
    pub fn reconcile(&mut self, current: &TrackBindings) -> Result<(), String> {
        let mut candidate = self.clone();
        let mut mapping = BTreeMap::new();
        let ids: BTreeSet<_> = self
            .tracks
            .iter()
            .map(|t| t.id.clone())
            .chain(self.params.keys().cloned())
            .collect();
        for old in &ids {
            let (new, identity) = if self.needs_rebind {
                let identity = self
                    .bindings
                    .get(old)
                    .ok_or("ARA edit identity missing; cannot restore by session track number")?;
                let matches: Vec<_> = current
                    .iter()
                    .filter(|(_, known)| *known == identity)
                    .collect();
                if identity.is_empty() || matches.len() != 1 {
                    return Err(
                        "ARA edit identity missing or ambiguous in rebuilt host graph".into(),
                    );
                }
                (matches[0].0.clone(), identity.clone())
            } else {
                // live轨道已由真实sequence边明确关联，不因轨内成员增删而丢掉编辑。
                let Some(identity) = current.get(old) else {
                    continue;
                };
                // mute/空轨不是身份丢失；live序列仍明确，保留此前持久成员供解除/保存。
                // 冷恢复分支仍拒绝空或歧义身份，不能用旧轨道序号猜归属。
                if identity.is_empty() {
                    mapping.insert(
                        old.clone(),
                        (
                            old.clone(),
                            self.bindings.get(old).cloned().unwrap_or_default(),
                        ),
                    );
                    continue;
                }
                (old.clone(), identity.clone())
            };
            // live归属已由宿主区域分配证明；复制轨道合法共享modification/source。
            // restore仍要求候选集中恰好一个身份匹配，不能仅按旧轨道序号猜测。
            if identity.is_empty()
                || identity
                    .iter()
                    .any(|pair| pair.0.is_empty() || pair.1.is_empty())
            {
                return Err("ARA edit identity is empty; cannot persist uniquely".into());
            }
            if mapping.values().any(|(known, _)| known == &new) {
                return Err("ARA saved edit identities are ambiguous; multiple records target one host track".into());
            }
            mapping.insert(old.clone(), (new, identity));
        }
        candidate.params = self
            .params
            .iter()
            .filter_map(|(id, p)| mapping.get(id).map(|(new, _)| (new.clone(), p.clone())))
            .collect();
        candidate.tracks = self
            .tracks
            .iter()
            .filter_map(|track| {
                mapping.get(&track.id).map(|(new, _)| {
                    let mut track = track.clone();
                    track.id = new.clone();
                    track
                })
            })
            .collect();
        candidate.bindings = mapping.values().cloned().collect();
        for record in candidate.atlas.regions.values_mut() {
            if let Some((root, _)) = mapping.get(&record.root) {
                record.root = root.clone();
            }
        }
        for record in candidate.atlas.copy_seeds.values_mut() {
            if let Some((root, _)) = mapping.get(&record.root) {
                record.root = root.clone();
            }
        }
        candidate.atlas.gaps = self
            .atlas
            .gaps
            .iter()
            .filter_map(|(root, params)| {
                mapping
                    .get(root)
                    .map(|(new, _)| (new.clone(), params.clone()))
            })
            .collect();
        candidate.needs_rebind = false;
        *self = candidate;
        Ok(())
    }

    /// 只复制用户曲线与已有轨道参数，不接受客户端移动/替换clip。
    pub fn merge(
        &self,
        host: &TimelineState,
        client: &TimelineState,
        base_revision: u64,
    ) -> Result<Self, String> {
        if base_revision != self.revision {
            return Err("Conflict: edit revision changed".into());
        }
        let host_ids: BTreeSet<_> = host.tracks.iter().map(|track| track.id.as_str()).collect();
        let client_ids: BTreeSet<_> = client
            .tracks
            .iter()
            .map(|track| track.id.as_str())
            .collect();
        if host_ids.len() != host.tracks.len()
            || client_ids.len() != client.tracks.len()
            || client_ids != host_ids
            || client.tracks.iter().any(|track| {
                !host.tracks.iter().any(|known| known.id == track.id)
                    || !track.volume.is_finite()
                    || !(0.0..=4.0).contains(&track.volume)
            })
        {
            return Err("invalid or unknown track".into());
        }
        for (id, params) in &client.params_by_root_track {
            if !host.tracks.iter().any(|track| track.id == *id)
                || !params.frame_period_ms.is_finite()
                || !(0.1..=100.0).contains(&params.frame_period_ms)
            {
                return Err("invalid curve track or frame period".into());
            }
            for curve in [
                &params.pitch_orig,
                &params.pitch_edit,
                &params.tension_orig,
                &params.tension_edit,
            ]
            .into_iter()
            .chain(params.extra_curves.values())
            {
                if curve.len() > 1_000_000
                    || curve
                        .iter()
                        .any(|value| !value.is_finite() || value.abs() > 10000.0)
                {
                    return Err("invalid parameter curve".into());
                }
            }
            if params.extra_params.values().any(|value| !value.is_finite()) {
                return Err("invalid static parameter".into());
            }
        }
        let mut merged = self.clone();
        merged.revision = self.revision.checked_add(1).ok_or("revision exhausted")?;
        merged
            .params
            .retain(|id, _| !host_ids.contains(id.as_str()));
        merged.params.extend(client.params_by_root_track.clone());
        merged
            .tracks
            .retain(|track| !host_ids.contains(track.id.as_str()));
        merged.tracks.extend(client.tracks.clone());
        Ok(merged)
    }
    /// 当前宿主几何上叠加持久化参数，旧轨道记录不会创造虚构轨道。
    pub fn apply(&self, timeline: &mut TimelineState) {
        // 未完成身份关联的恢复记录绝不能沿旧会话序号静默应用。
        if self.needs_rebind {
            return;
        }
        timeline.params_by_root_track = self
            .params
            .iter()
            .filter(|(id, _)| timeline.tracks.iter().any(|t| t.id == **id))
            .map(|(id, p)| (id.clone(), p.clone()))
            .collect();
        for track in &mut timeline.tracks {
            if let Some(edited) = self.tracks.iter().find(|t| t.id == track.id) {
                track.compose_enabled = edited.compose_enabled;
                track.pitch_analysis_algo = edited.pitch_analysis_algo.clone();
                track.volume = edited.volume;
                track.muted = edited.muted;
                track.solo = edited.solo;
            }
        }
        // 插件自有、宿主表达不到的逐片段属性：必须在**共同的投影入口**重新播种，
        // 否则会被下一次 `ensure_loaded` 的整份重建冲掉（颜色消失、共振峰变形失效）。
        for clip in &mut timeline.clips {
            let Some(local) = self.clip_local.get(&clip.id) else {
                continue;
            };
            if let Some(color) = &local.color {
                clip.color = color.clone();
            }
            if local.formant_morph.is_some() {
                clip.formant_morph = local.formant_morph.clone();
            }
        }
    }

    /// 本构建支持的**最高**状态 schema 代际。
    ///
    /// 【为什么 `encode` 不写这个常量】版本号在这里是**向后兼容**手段：只带 atlas 的
    /// 载荷写 2、带 groups 写 5……旧构建（它只认自己那一代及更低）因此仍能读。写死最高
    /// 代际会让所有旧构建一律拒绝，把"能读"变成"全丢"。这个常量只用于 `restore` 判断
    /// "由更新的插件保存"。
    ///
    /// 【为什么第 8 代必须升】第 8 代把权威曲线从"工程帧索引"改成"**源帧**索引"，
    /// 并把 `SourceCurve.basis` 从对象改成 `null`（见 `SourceCurve::basis`）。
    /// 旧构建读新存档时那个字段会解析失败 —— 升代让它得到"这份存档由更新的插件保存"
    /// 这句**可行动**的话，而不是一条看不懂的 serde 错误。反过来，新构建读第 7 代及
    /// 更早的存档仍然完全正确（`basis` 有对象 ⇒ 走旧索引分支），因此**不需要折算**。
    const CURRENT_STATE_VERSION: u64 = 8;

    /// 有版本且有界的组件state；旧空state保持默认编辑。
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.groups.validate()?;
        self.atlas.validate()?;
        self.validate_fades()?;
        self.validate_clip_local()?;
        for id in self.tracks.iter().map(|t| &t.id).chain(self.params.keys()) {
            if self.bindings.get(id).is_none_or(Vec::is_empty) {
                return Err("ARA edit identity missing; cannot save edits".into());
            }
        }
        let version = if !self.atlas.authorities.is_empty() {
            8
        } else if !self.clip_local.is_empty() {
            6
        } else if !self.groups.is_empty() {
            5
        } else if !self.fades.is_empty() {
            4
        } else if self.atlas.is_empty() {
            2
        } else {
            3
        };
        // 【为什么不直接序列化 `self`】`atlas.regions` 是**每帧派生**的投影缓存，它的
        // 权威是 `atlas.authorities`。写出派生缓存会让同一份采样在存档里出现两次
        // （曲线是 `Arc<Vec<f32>>`，内存里共享、序列化时各写一份），而它读回来之后
        // 又会被 `follow_geometry` 立刻覆盖。清掉它：存档只带权威。
        let mut trimmed = self.clone();
        trimmed.atlas.regions.clear();
        let bytes = serde_json::to_vec(&serde_json::json!({"version":version,"edits":trimmed}))
            .map_err(|e| e.to_string())?;
        if bytes.len() > hifishifter_ara_ipc::MAX_FRAME {
            return Err("state exceeds transport budget".into());
        }
        Ok(bytes)
    }
    /// 恢复使乐观并发revision前进，防止旧GUI再次覆盖宿主undo/恢复。
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        if bytes.len() > hifishifter_ara_ipc::MAX_FRAME {
            return Err("state too large".into());
        }
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        // 【为什么区分"更新版保存"与"损坏"】两者此前都报同一句
        // `unsupported state version`，用户看到"编辑全没了"却不知道原因。版本高于本
        // 构建时明确说明"由更新的插件保存"，让人知道该升级而不是以为档坏了。
        let version = value["version"].as_u64().unwrap_or(0);
        if version == 0 {
            return Err("unsupported state version".into());
        }
        if version > Self::CURRENT_STATE_VERSION {
            return Err(format!(
                "state was saved by a newer plugin version (state {version} > supported {})",
                Self::CURRENT_STATE_VERSION
            ));
        }
        let mut restored: Self =
            serde_json::from_value(value["edits"].clone()).map_err(|e| e.to_string())?;
        restored.groups.validate()?;
        restored.validate_fades()?;
        restored.atlas = restored.atlas.reserve_restored()?;
        if version == 1 && (!restored.params.is_empty() || !restored.tracks.is_empty()) {
            return Err("legacy ARA edits have no persistent identity; cannot restore by session track number".into());
        }
        if restored
            .tracks
            .iter()
            .map(|t| &t.id)
            .chain(restored.params.keys())
            .any(|id| restored.bindings.get(id).is_none_or(Vec::is_empty))
        {
            return Err("ARA edit identity missing in saved state".into());
        }
        restored.needs_rebind = !restored.tracks.is_empty() || !restored.params.is_empty();
        restored.validate_clip_local()?;
        restored.revision = self
            .revision
            .max(restored.revision)
            .checked_add(1)
            .ok_or("revision exhausted")?;
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn host() -> TimelineState {
        serde_json::from_value(serde_json::json!({"tracks":[{"id":"track","name":"host","order":0}],"clips":[],"bpm":120,"project_sec":1})).unwrap()
    }
    #[test]
    fn edits_preserve_host_geometry_and_reject_stale_writer() {
        let mut client = host();
        client.tracks[0].volume = 0.5;
        client.project_sec = 400.;
        let state = EditState::default().merge(&host(), &client, 0).unwrap();
        let mut rendered = host();
        state.apply(&mut rendered);
        assert_eq!(rendered.tracks[0].volume, 0.5);
        assert_eq!(rendered.project_sec, 1.);
        assert!(state.merge(&host(), &client, 0).is_err());
    }
    /// **不变量：插件自有逐片段属性必须在共同投影入口重新播种。**
    ///
    /// 【为什么这是回归测试】`session.timeline` 每次 `ensure_loaded` 都被整体替换；
    /// 不落在"有权威来源 + 在共同投影入口重新播种"的地方，颜色会在下一次重建后消失、
    /// 共振峰变形会失效（它还是渲染输入）。键是 `ara-item-{guid}`，与投影里的 clip id
    /// 同口径，冷重建后仍能对上。
    #[test]
    fn clip_local_properties_are_reseeded_by_apply() {
        let mut edits = EditState::default();
        edits.clip_local.insert(
            "ara-item-{11111111-1111-1111-1111-111111111111}".into(),
            ClipLocalState {
                color: Some("#abcdef".into()),
                formant_morph: Some(hifishifter_kernel::state::ClipFormantMorph {
                    enabled: true,
                    ..Default::default()
                }),
            },
        );
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"ara-item-{11111111-1111-1111-1111-111111111111}",
                      "track_id":"track","name":"c","start_sec":0,"length_sec":1,
                      "takes":[{"id":"t","source_start_sec":0,"source_end_sec":1}]}],
            "bpm":120,"project_sec":1
        }))
        .unwrap();
        let before = timeline.clips[0].color.clone();
        assert_ne!(before, "#abcdef", "前提：宿主时间线没有这条颜色");
        edits.apply(&mut timeline);
        assert_eq!(timeline.clips[0].color, "#abcdef");
        assert!(
            timeline.clips[0].formant_morph.is_some(),
            "共振峰变形是渲染输入，必须随投影播种"
        );
    }
    #[test]
    fn invalid_curve_and_unknown_track_are_rejected() {
        let mut client = host();
        client.params_by_root_track.insert(
            "track".into(),
            TrackParamsState {
                pitch_edit: vec![f32::NAN],
                ..Default::default()
            },
        );
        assert!(EditState::default().merge(&host(), &client, 0).is_err());
        client.params_by_root_track.clear();
        client.tracks[0].id = "foreign".into();
        assert!(EditState::default().merge(&host(), &client, 0).is_err());
    }
    /// live空轨/mute只暂停区域，不丢曲线或持久成员；冷恢复空身份仍拒绝。
    #[test]
    fn ui_inventory_live_empty_members_keep_parameters_and_previous_persistent_identity() {
        let mut client = host();
        client.params_by_root_track.insert(
            "track".into(),
            TrackParamsState {
                frame_period_ms: 5.,
                pitch_edit: vec![61., 63.],
                ..Default::default()
            },
        );
        let mut state = EditState::default().merge(&host(), &client, 0).unwrap();
        let members = vec![("mod".into(), "source".into())];
        state
            .reconcile(&BTreeMap::from([("track".into(), members.clone())]))
            .unwrap();
        state
            .reconcile(&BTreeMap::from([("track".into(), vec![])]))
            .unwrap();
        assert_eq!(state.params["track"].pitch_edit, [61., 63.]);
        assert_eq!(state.bindings["track"], members);
        let mut restored = EditState::default();
        restored.restore(&state.encode().unwrap()).unwrap();
        assert!(restored
            .reconcile(&BTreeMap::from([("track".into(), vec![])]))
            .is_err());
    }

    #[test]
    fn local_view_merge_preserves_other_track_curves_and_volume_after_restore() {
        let a = host();
        let mut b = host();
        b.tracks[0].id = "other".into();
        let mut edited_a = a.clone();
        edited_a.tracks[0].volume = 0.5;
        edited_a.params_by_root_track.insert(
            "track".into(),
            TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![61.0, 63.0],
                ..Default::default()
            },
        );
        let first = EditState::default().merge(&a, &edited_a, 0).unwrap();
        let mut edited_b = b.clone();
        edited_b.tracks[0].volume = 0.25;
        edited_b.params_by_root_track.insert(
            "other".into(),
            TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![70.0],
                ..Default::default()
            },
        );
        let mut second = first.merge(&b, &edited_b, 1).unwrap();
        let identities = BTreeMap::from([
            ("track".into(), vec![("mod-a".into(), "source".into())]),
            ("other".into(), vec![("mod-b".into(), "source".into())]),
        ]);
        second.reconcile(&identities).unwrap();
        assert_eq!(second.params["track"].pitch_edit, [61.0, 63.0]);
        assert_eq!(
            second
                .tracks
                .iter()
                .find(|t| t.id == "track")
                .unwrap()
                .volume,
            0.5
        );
        let mut restored = EditState::default();
        restored.restore(&second.encode().unwrap()).unwrap();
        restored.reconcile(&identities).unwrap();
        let mut applied = a.clone();
        restored.apply(&mut applied);
        assert_eq!(
            applied.params_by_root_track["track"].pitch_edit,
            [61.0, 63.0]
        );
        assert_eq!(applied.tracks[0].volume, 0.5);
    }

    #[test]
    fn duplicate_track_ids_cannot_impersonate_a_complete_authorized_view() {
        let mut known = host();
        let mut other = known.tracks[0].clone();
        other.id = "other".into();
        known.tracks.push(other);
        let mut client = known.clone();
        client.tracks[1] = client.tracks[0].clone();
        assert!(EditState::default().merge(&known, &client, 0).is_err());
    }

    #[test]
    fn restored_edits_refuse_ambiguous_empty_and_missing_host_identity_without_rebinding() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        let original = BTreeMap::from([("track".into(), vec![("mod".into(), "source".into())])]);
        state.reconcile(&original).unwrap();
        let bytes = state.encode().unwrap();
        for current in [
            BTreeMap::new(),
            BTreeMap::from([("new".into(), vec![])]),
            BTreeMap::from([
                ("a".into(), vec![("mod".into(), "source".into())]),
                ("b".into(), vec![("mod".into(), "source".into())]),
            ]),
        ] {
            let mut restored = EditState::default();
            restored.restore(&bytes).unwrap();
            assert!(restored
                .reconcile(&current)
                .unwrap_err()
                .contains("identity"));
            let mut timeline = host();
            restored.apply(&mut timeline);
            assert_eq!(timeline.tracks[0].volume, 1.0, "未解决身份不套用临时序号");
            assert!(restored.needs_rebind);
        }
        // live 空轨可能是 mute，沿已确认的身份保留；上述冷恢复仍拒绝空/歧义身份。
        state
            .reconcile(&BTreeMap::from([("track".into(), vec![])]))
            .unwrap();
        assert_eq!(state.bindings, original);
        assert_eq!(state.tracks[0].volume, 0.25);
    }

    #[test]
    fn live_membership_change_updates_saved_identity_and_legacy_nonempty_state_is_rejected() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        let mut identity =
            BTreeMap::from([("track".into(), vec![("mod-one".into(), "source".into())])]);
        state.reconcile(&identity).unwrap();
        identity
            .get_mut("track")
            .unwrap()
            .push(("mod-two".into(), "source".into()));
        state.reconcile(&identity).unwrap();
        assert_eq!(state.tracks[0].volume, 0.25);
        assert_eq!(state.bindings["track"].len(), 2);
        let mut legacy =
            serde_json::from_slice::<serde_json::Value>(&state.encode().unwrap()).unwrap();
        legacy["version"] = serde_json::json!(1);
        assert!(state
            .restore(&serde_json::to_vec(&legacy).unwrap())
            .unwrap_err()
            .contains("legacy"));
        assert_eq!(state.revision, 1, "拒绝旧state保留现有编辑");
    }

    /// 版本号是"最低够用的代际"（**向后兼容**手段），且"更新版保存的"与"损坏的"
    /// 给出可区分的报错。
    #[test]
    fn a_newer_state_version_is_reported_distinctly_from_corruption() {
        let mut state = EditState::default();
        let value: serde_json::Value = serde_json::from_slice(&state.encode().unwrap()).unwrap();
        assert_eq!(
            value["version"].as_u64(),
            Some(2),
            "空编辑只带 atlas 段 ⇒ 写最低代际 2，旧构建也能读"
        );
        // 更高版本：明确说"由更新的插件保存"，而不是含混的 unsupported。
        let mut newer = value.clone();
        newer["version"] = serde_json::json!(EditState::CURRENT_STATE_VERSION + 1);
        let error = state
            .restore(&serde_json::to_vec(&newer).unwrap())
            .unwrap_err();
        assert!(error.contains("newer plugin version"), "{error}");
        // 缺版本：仍是"不支持"。
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("version");
        assert!(state
            .restore(&serde_json::to_vec(&missing).unwrap())
            .unwrap_err()
            .contains("unsupported"));
    }

    #[test]
    fn duplicate_saved_track_identities_cannot_collapse_into_one_rebuilt_track() {
        let mut edited = host();
        edited.tracks[0].volume = 0.25;
        let mut state = EditState::default().merge(&host(), &edited, 0).unwrap();
        let identity = vec![("mod".into(), "source".into())];
        state
            .reconcile(&BTreeMap::from([("track".into(), identity.clone())]))
            .unwrap();
        let mut saved =
            serde_json::from_slice::<serde_json::Value>(&state.encode().unwrap()).unwrap();
        let mut duplicate = saved["edits"]["tracks"][0].clone();
        duplicate["id"] = serde_json::json!("other-old-track");
        saved["edits"]["tracks"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        saved["edits"]["bindings"]["other-old-track"] = serde_json::to_value(&identity).unwrap();
        let mut restored = EditState::default();
        restored
            .restore(&serde_json::to_vec(&saved).unwrap())
            .unwrap();
        assert!(
            restored
                .reconcile(&BTreeMap::from([("new-track".into(), identity)]))
                .is_err(),
            "保存数据中的重复身份不能合并串轨"
        );
        assert!(restored.needs_rebind);
    }
}
