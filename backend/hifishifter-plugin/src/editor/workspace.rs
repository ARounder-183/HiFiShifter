//! 工程编辑权限是同一活ARA文档renderer区域并集，音频输出仍按各renderer原分配隔离。
use crate::render::document::DocumentSession;
use crate::render::ownership::region_owners;
use hifishifter_kernel::state::TimelineState;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

/// gain在原GUI是active-take扁平投影；插件显示item音量，两份显示字段必须一致且不送DSP。
fn display_item_gain(clip: &mut hifishifter_kernel::state::Clip, gain: f64) {
    clip.gain = gain as f32;
    for take in &mut clip.takes {
        take.gain = gain as f32;
    }
}
/// JSON装饰与actor投影同源；host_gain额外保留真实take音量/极性用于现场排查。
fn decorate_item_gain(clip: &mut serde_json::Value, g: &crate::host::geometry::HostClipGeometry) {
    clip["gain"] = serde_json::json!(g.item_gain);
    if let Some(takes) = clip["takes"].as_array_mut() {
        for take in takes {
            take["gain"] = serde_json::json!(g.item_gain);
        }
    }
    clip["host_gain"] = serde_json::json!({"item":g.item_gain,"take":g.take_gain});
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkspaceScope {
    pub regions: BTreeSet<u64>,
}
impl DocumentSession {
    /// 参数分组只沿宿主已校验的轨道GUID关联；namespace仅用于当前WebView显示身份。
    pub(crate) fn group_aliases(
        &self,
        namespace: &str,
    ) -> std::collections::BTreeMap<String, String> {
        self.ui_tracks
            .lock()
            .unwrap()
            .values()
            .map(|track| (track.guid.clone(), format!("{namespace}{}", track.id)))
            .collect()
    }
    /// GUI清单刷新不覆盖插件私有父级和顺序；新轨未出现在私有表中时保持根级。
    pub(crate) fn present_private_groups(&self, timeline: &mut TimelineState, namespace: &str) {
        // 锁序固定为 edits → host_owned（见 `host_folder_children` 的说明）：
        // `project_private_group_view` 等路径在持 edits 时读 host_owned，反向获取会死锁。
        let edits = self.edits.lock().unwrap();
        let host_owned = self.host_folder_children.lock().unwrap().clone();
        edits
            .groups
            .apply(timeline, &self.group_aliases(namespace), &host_owned);
    }
    /// 显示参数根可包含无播放区域的空父轨；不创建任何clip或未经授权的源。
    fn add_group_view_tracks(&self, timeline: &mut TimelineState) {
        for host in self.ui_tracks.lock().unwrap().values() {
            if !timeline.tracks.iter().any(|track| track.id == host.id) {
                timeline.tracks.push(
                    serde_json::from_value(
                        serde_json::json!({"id":host.id,"name":host.name,"order":host.order,
                    "compose_enabled":true,"pitch_analysis_algo":"nsf_hifigan_onnx"}),
                    )
                    .unwrap(),
                );
            }
        }
    }
    /// 音频状态仍按物理轨道/区域保存；分组只改变GUI参数根投影。
    pub(crate) fn project_private_group_view(
        &self,
        timeline: &mut TimelineState,
        edits: &crate::state_channel::EditState,
    ) -> Result<(), String> {
        self.add_group_view_tracks(timeline);
        // 锁序 edits（调用方已持）→ host_owned，与 `present_private_groups` 一致。
        let host_owned = self.host_folder_children.lock().unwrap().clone();
        edits
            .groups
            .apply(timeline, &self.group_aliases(""), &host_owned);
        if !edits.atlas.is_empty() {
            timeline.params_by_root_track.extend(
                edits
                    .atlas
                    .project_roots(timeline, &self.parameter_identities_locked(timeline)?)?,
            );
        }
        let roots = timeline
            .tracks
            .iter()
            .filter(|track| track.parent_id.is_none())
            .map(|track| track.id.clone())
            .collect::<BTreeSet<_>>();
        timeline
            .params_by_root_track
            .retain(|root, _| roots.contains(root));
        Ok(())
    }
    /// 为一次分组编辑冻结物理旧参数与分组旧投影；复制delta而非全父轨数组，保护独立子轨曲线。
    pub(crate) fn private_parameter_views(
        &self,
        selected: Option<String>,
        groups: Option<&super::private_groups::TrackGroups>,
    ) -> Result<(TimelineState, TimelineState), String> {
        let _transaction = self.transaction.lock().unwrap();
        let mut flat = self.workspace_timeline_locked()?;
        flat.selected_clip_id = selected;
        let edits = self.edits.lock().unwrap();
        // 锁序 edits → host_owned（见 `present_private_groups` 的说明）。
        let host_owned = self.host_folder_children.lock().unwrap().clone();
        edits.apply(&mut flat);
        self.add_group_view_tracks(&mut flat);
        if !edits.atlas.is_empty() {
            flat.params_by_root_track.extend(
                edits
                    .atlas
                    .project_roots(&flat, &self.parameter_identities_locked(&flat)?)?,
            );
        }
        let mut grouped = flat.clone();
        groups
            .unwrap_or(&edits.groups)
            .apply(&mut grouped, &self.group_aliases(""), &host_owned);
        if !edits.atlas.is_empty() {
            grouped.params_by_root_track.extend(
                edits
                    .atlas
                    .project_roots(&grouped, &self.parameter_identities_locked(&grouped)?)?,
            );
        }
        let roots = grouped
            .tracks
            .iter()
            .filter(|track| track.parent_id.is_none())
            .map(|track| track.id.clone())
            .collect::<BTreeSet<_>>();
        grouped
            .params_by_root_track
            .retain(|root, _| roots.contains(root));
        Ok((flat, grouped))
    }
    /// UI可显示宿主尚未分配给ARA的静音item；这些占位不进入任何renderer或源PCM读取。
    pub(crate) fn present_host_inventory(&self, timeline: &mut TimelineState, namespace: &str) {
        let prefix = |id: &str| format!("{namespace}{id}");
        let tracks = self.ui_tracks.lock().unwrap();
        if tracks.is_empty() {
            return;
        }
        timeline.tracks.retain(|track| track.id != "track_main");
        let mut present = std::collections::BTreeSet::new();
        let mut owned = std::collections::BTreeSet::new();
        // 由宿主 folder 得到父级的轨道；它们才是私有分组必须让路的那些（见
        // `host_folder_children`）。
        let mut folder_children = std::collections::BTreeSet::new();
        for host in tracks.values() {
            let track_id = prefix(&host.id);
            owned.insert(track_id.clone());
            // 宿主 folder 父子 → 时间线参数根。
            //
            // 【为什么是参数根，而不是音频路由】REAPER 的 folder 会改变混音/路由语义，
            // 而 HFS 的父子只决定参数根（合成开关、分析算法、参数曲线归属）。这里只设
            // `parent_id`，不推断也不改动任何路由 —— 与既有私有分组同一条边界。
            //
            // 父轨必须真实存在于本次清单里才挂；否则保持根级，不凭空造一条不存在的轨。
            let parent_id = host.parent_guid.as_deref().and_then(|guid| {
                tracks
                    .get(guid)
                    .map(|parent| prefix(&parent.id))
                    .filter(|id| *id != track_id)
            });
            if parent_id.is_some() {
                folder_children.insert(track_id.clone());
            }
            if let Some(track) = timeline.tracks.iter_mut().find(|t| t.id == track_id) {
                track.name = host.name.clone();
                track.order = host.order;
                track.parent_id = parent_id;
            } else {
                let mut track: hifishifter_kernel::state::Track =
                    serde_json::from_value(serde_json::json!({"id":track_id,"name":host.name,"order":host.order,"compose_enabled":true})).unwrap();
                track.parent_id = parent_id;
                timeline.tracks.push(track);
            }
            for item in &host.items {
                let g = &item.geometry;
                let id = prefix(&format!("ara-item-{}", g.item_id));
                present.insert(id.clone());
                if !timeline.clips.iter().any(|clip| clip.id == id) {
                    let mut clip:hifishifter_kernel::state::Clip=serde_json::from_value(serde_json::json!({"id":id,"track_id":track_id,"name":item.name,
                        "start_sec":g.start_sec,"length_sec":g.duration_sec,"takes":[{"id":prefix(&g.take_id),"source_start_sec":g.source_start_sec,
                        "source_end_sec":g.source_start_sec+g.duration_sec*g.playback_rate,"playback_rate":g.playback_rate}]})).unwrap();
                    clip.normalize_takes();
                    timeline.clips.push(clip);
                }
                let clip = timeline
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == id)
                    .unwrap();
                display_item_gain(clip, g.item_gain);
                clip.track_id = track_id.clone();
                clip.name = item.name.clone();
                clip.muted = g.muted;
                clip.start_sec = g.start_sec;
                clip.length_sec = g.duration_sec;
                clip.group_id = if g.group_id == 0 {
                    None
                } else {
                    Some(format!("reaper-group-{}", g.group_id))
                };
                clip.snap_offset_sec = g.snap_offset_sec;
                clip.fade_in_sec = g.fade_in_sec;
                clip.fade_out_sec = g.fade_out_sec;
                clip.auto_fade_in_sec = g.auto_fade_in_sec;
                clip.auto_fade_out_sec = g.auto_fade_out_sec;
                clip.fade_in_shape = g.fade_in_shape;
                clip.fade_out_shape = g.fade_out_shape;
                clip.fade_in_dir = g.fade_in_dir;
                clip.fade_out_dir = g.fade_out_dir;
                timeline.project_sec = timeline.project_sec.max(g.start_sec + g.duration_sec);
            }
        }
        timeline.clips.retain(|clip| {
            !clip.id.starts_with(&prefix("ara-item-")) || present.contains(&clip.id)
        });
        timeline.clips.retain(|clip| {
            !owned.contains(&clip.track_id) || !clip.id.starts_with(&prefix("ara-clip-"))
        });
        let known = self.ui_known_tracks.lock().unwrap();
        timeline.tracks.retain(|track| {
            !known.iter().any(|id| prefix(id) == track.id) || owned.contains(&track.id)
        });
        drop(known);
        // 记下"父级由宿主 folder 决定"的轨道，供私有分组呈现时让路
        // （见 `host_folder_children`）。
        *self.host_folder_children.lock().unwrap() = folder_children;
        timeline.tracks.sort_by_key(|track| track.order);
        if timeline.selected_track_id.is_none() {
            timeline.selected_track_id = timeline.tracks.first().map(|track| track.id.clone());
        }
    }
    /// 组件保存/恢复仅操作其实际assigned clips的item GUID，防止另一个轨道的形状串入。
    pub(crate) fn fade_items_locked(
        &self,
        clips: &std::collections::BTreeSet<String>,
    ) -> std::collections::BTreeSet<String> {
        let identities = self.clip_ids.lock().unwrap();
        self.renderer_owners()
            .iter()
            .filter_map(|owner| {
                let bound = owner.host_geometry_metadata_locked(self)?;
                identities
                    .get(&bound.region_key)
                    .filter(|id| clips.contains(*id))
                    .map(|_| bound.geometry.item_id)
            })
            .collect()
    }
    /// 创建item返回的GUID只用于等待它自己的真实ARA区域，不用文件名/位置匹配新clip。
    pub(crate) fn clip_for_host_item(&self, item: &str) -> Option<String> {
        let _transaction = self.transaction.lock().unwrap();
        if !self.is_alive() {
            return None;
        }
        let identities = self.clip_ids.lock().unwrap();
        for owner in self.renderer_owners() {
            let Some(bound) = owner.host_geometry_metadata_locked(self) else {
                continue;
            };
            if bound.geometry.item_id == item {
                return identities.get(&bound.region_key).cloned();
            }
        }
        None
    }
    /// 静音分割也等待真实GUI清单中的item；没有播放区域不等同片段不存在。
    pub(crate) fn gui_clip_for_host_item(&self, item: &str) -> Option<String> {
        if let Some(id) = self.clip_for_host_item(item) {
            return Some(id);
        }
        self.ui_tracks
            .lock()
            .unwrap()
            .values()
            .any(|track| {
                track
                    .items
                    .iter()
                    .any(|entry| entry.geometry.item_id == item)
            })
            .then(|| format!("ara-item-{item}"))
    }
    /// UI专用原始曲率不写进kernel状态或参数权威；普通fade最终声音仍由宿主负责。
    pub(crate) fn decorate_host_fades_locked(
        &self,
        payload: &mut serde_json::Value,
        namespace: &str,
        fades: &std::collections::BTreeMap<String, crate::fade::FadeStyle>,
    ) {
        let identities = self.clip_ids.lock().unwrap().clone();
        let Some(clips) = payload["clips"].as_array_mut() else {
            return;
        };
        for track in self.ui_tracks.lock().unwrap().values() {
            for item in &track.items {
                if let Some(clip) = clips.iter_mut().find(|clip| {
                    clip["id"] == format!("{namespace}ara-item-{}", item.geometry.item_id)
                }) {
                    decorate_item_gain(clip, &item.geometry);
                    let g = &item.geometry;
                    clip["host_fades"] = serde_json::json!({"curve_mode":match g.fade_axes_new {Some(true)=>"reaper_new",Some(false)=>"legacy",None=>"unknown"},
                    "in_curvature":g.fade_in_dir_new,"out_curvature":g.fade_out_dir_new,"in_s":g.fade_in_dir2_new,"out_s":g.fade_out_dir2_new});
                }
            }
        }
        for owner in self.renderer_owners() {
            let Some(bound) = owner.host_geometry_metadata_locked(self) else {
                continue;
            };
            let Some(id) = identities.get(&bound.region_key) else {
                continue;
            };
            let ui_id = format!("{namespace}{id}");
            let Some(clip) = clips.iter_mut().find(|clip| clip["id"] == ui_id) else {
                continue;
            };
            let g = bound.geometry;
            decorate_item_gain(clip, &g);
            clip["snap_offset_sec"] = serde_json::json!(g.snap_offset_sec);
            let delegated = self
                .regions
                .lock()
                .unwrap()
                .get(&bound.region_key)
                .is_some_and(|region| {
                    region.has_content_based_fade_at_head || region.has_content_based_fade_at_tail
                });
            if delegated {
                let style = fades.get(&g.item_id).cloned().unwrap_or_default();
                clip["fade_in_shape"] = serde_json::json!(style.in_shape);
                clip["fade_out_shape"] = serde_json::json!(style.out_shape);
                clip["fade_in_dir"] = serde_json::json!(style.in_dir);
                clip["fade_out_dir"] = serde_json::json!(style.out_dir);
                clip["host_fades"] = serde_json::json!({"curve_mode":"hifishifter","in_curvature":style.in_dir,"out_curvature":style.out_dir,"in_s":0.,"out_s":0.});
                continue;
            }
            clip["host_fades"] = serde_json::json!({"curve_mode":match g.fade_axes_new {Some(true)=>"reaper_new",Some(false)=>"legacy",None=>"unknown"},
                "in_curvature":g.fade_in_dir_new,"out_curvature":g.fade_out_dir_new,
                "in_s":g.fade_in_dir2_new,"out_s":g.fade_out_dir2_new});
        }
    }
    /// 无宿主getter的短事务装饰；调用者不得仍持编辑timeline锁。
    pub(crate) fn decorate_host_fades(&self, payload: &mut serde_json::Value, namespace: &str) {
        let _transaction = self.transaction.lock().unwrap();
        if self.is_alive() {
            self.decorate_host_fades_locked(payload, namespace, &self.edits.lock().unwrap().fades);
        }
    }
    /// 普通手动/自动fade和吸附偏移投影到原GUI，内核继续消费未烘焙fade的ARA时间线。
    /// 只沿已核对的唯一真实region key，不按轨名/位置猜关联。
    pub(crate) fn project_ui_fades_locked(
        &self,
        timeline: &mut TimelineState,
        fades: &std::collections::BTreeMap<String, crate::fade::FadeStyle>,
    ) {
        let identities = self.clip_ids.lock().unwrap().clone();
        for owner in self.renderer_owners() {
            let Some(bound) = owner.host_geometry_metadata_locked(self) else {
                continue;
            };
            let Some(id) = identities.get(&bound.region_key) else {
                continue;
            };
            let Some(clip) = timeline.clips.iter_mut().find(|clip| &clip.id == id) else {
                continue;
            };
            let geometry = bound.geometry;
            display_item_gain(clip, geometry.item_gain);
            clip.snap_offset_sec = geometry.snap_offset_sec;
            clip.fade_in_sec = geometry.fade_in_sec;
            clip.fade_out_sec = geometry.fade_out_sec;
            clip.auto_fade_in_sec = geometry.auto_fade_in_sec;
            clip.auto_fade_out_sec = geometry.auto_fade_out_sec;
            clip.fade_in_shape = geometry.fade_in_shape;
            clip.fade_out_shape = geometry.fade_out_shape;
            clip.fade_in_dir = geometry.fade_in_dir;
            clip.fade_out_dir = geometry.fade_out_dir;
            if self
                .regions
                .lock()
                .unwrap()
                .get(&bound.region_key)
                .is_some_and(|region| {
                    region.has_content_based_fade_at_head || region.has_content_based_fade_at_tail
                })
            {
                let style = fades.get(&geometry.item_id).cloned().unwrap_or_default();
                clip.fade_in_shape = style.in_shape;
                clip.fade_out_shape = style.out_shape;
                clip.fade_in_dir = style.in_dir;
                clip.fade_out_dir = style.out_dir;
            }
        }
    }
    /// 只把宿主明确委托的一端写进原kernel；另一端仍归宿主，不能凭能力广告重复烘焙。
    pub(crate) fn project_audio_fades_locked(
        &self,
        timeline: &mut TimelineState,
        edits: &crate::state_channel::EditState,
    ) -> Result<(), String> {
        let identities = self.clip_ids.lock().unwrap().clone();
        let regions = self.regions.lock().unwrap().clone();
        for clip in &mut timeline.clips {
            let key = identities
                .iter()
                .find(|(_, id)| id.as_str() == clip.id)
                .map(|(key, _)| *key)
                .ok_or("missing fade region edge")?;
            let region = regions.get(&key).ok_or("missing fade region")?;
            if !region.has_content_based_fade_at_head && !region.has_content_based_fade_at_tail {
                continue;
            }
            let geometry = self
                .renderer_owners()
                .iter()
                .find_map(|owner| {
                    owner
                        .host_geometry_metadata_locked(self)
                        .filter(|bound| bound.region_key == key)
                })
                .ok_or("delegated fade host geometry pending")?
                .geometry;
            let style = edits
                .fades
                .get(&geometry.item_id)
                .cloned()
                .unwrap_or_default();
            if region.has_content_based_fade_at_head {
                clip.fade_in_sec = geometry.fade_in_sec;
                clip.auto_fade_in_sec = geometry.auto_fade_in_sec;
                clip.fade_in_shape = style.in_shape;
                clip.fade_in_dir = style.in_dir;
            }
            if region.has_content_based_fade_at_tail {
                clip.fade_out_sec = geometry.fade_out_sec;
                clip.auto_fade_out_sec = geometry.auto_fade_out_sec;
                clip.fade_out_shape = style.out_shape;
                clip.fade_out_dir = style.out_dir;
            }
        }
        Ok(())
    }
    /// 有效item静音按同文档唯一region身份投影；不把它存成HFS自己的可写轨道状态。
    pub(crate) fn project_host_mutes_locked(&self, timeline: &mut TimelineState) {
        let identities = self.clip_ids.lock().unwrap();
        for owner in self.renderer_owners() {
            let Some(bound) = owner.host_geometry_metadata_locked(self) else {
                continue;
            };
            let Some(id) = identities.get(&bound.region_key) else {
                continue;
            };
            if let Some(clip) = timeline.clips.iter_mut().find(|clip| &clip.id == id) {
                clip.muted = bound.geometry.muted;
            }
        }
    }
    /// 无PCM复制或host getter，pending曲线也可独立更新可见宿主fade。
    pub(crate) fn ui_fade_projection(&self) -> Result<(u64, TimelineState), String> {
        let _transaction = self.transaction.lock().unwrap();
        let mut timeline = self.workspace_timeline_locked()?;
        self.project_ui_fades_locked(&mut timeline, &self.edits.lock().unwrap().fades);
        Ok((self.ui_geometry_revision.load(Ordering::Acquire), timeline))
    }
    /// 非实时短事务读取完整授权scope；零分配只能返回零区域，不能等同全文档。
    // 非实时诊断入口，保留供授权范围核对。
    #[allow(dead_code)]
    pub(crate) fn workspace_scope(&self) -> Result<WorkspaceScope, String> {
        let _transaction = self.transaction.lock().unwrap();
        self.workspace_scope_locked()
    }
    /// 调用方已持transaction；真实model-ref所有权再次校验，拒绝跨文档或已销毁区域。
    pub(crate) fn workspace_scope_locked(&self) -> Result<WorkspaceScope, String> {
        if !self.is_alive() {
            return Err("document closed".into());
        }
        let mut regions = BTreeSet::new();
        for owner in self.renderer_owners() {
            regions.extend(owner.assigned_regions().map_err(|e| e.to_string())?);
        }
        if !regions.is_empty() {
            let keys = regions.iter().copied().collect::<Vec<_>>();
            let (document, _) = region_owners()
                .lock()
                .unwrap()
                .resolve(&keys)
                .map_err(|e| format!("workspace region ownership: {e:?}"))?;
            if document != self.id {
                return Err("workspace includes another document".into());
            }
        }
        Ok(WorkspaceScope { regions })
    }
    /// 原GUI的多轨快照只扩展查看/编辑范围，不更改任何播放renderer的混音归属。
    pub(crate) fn workspace_timeline(&self) -> Result<TimelineState, String> {
        let _transaction = self.transaction.lock().unwrap();
        self.workspace_timeline_locked()
    }
    pub(crate) fn workspace_timeline_locked(&self) -> Result<TimelineState, String> {
        let scope = self.workspace_scope_locked()?;
        let identities = self.clip_ids.lock().unwrap();
        let clips = scope
            .regions
            .iter()
            .map(|key| {
                identities
                    .get(key)
                    .cloned()
                    .ok_or("workspace clip identity missing")
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        drop(identities);
        let mut timeline = self
            .timeline
            .lock()
            .unwrap()
            .clone()
            .ok_or("host timeline unavailable")?;
        timeline.clips.retain(|clip| clips.contains(&clip.id));
        self.project_host_mutes_locked(&mut timeline);
        let mut tracks = timeline
            .clips
            .iter()
            .map(|clip| clip.track_id.clone())
            .collect::<BTreeSet<_>>();
        for track in self.ui_tracks.lock().unwrap().values() {
            if timeline.tracks.iter().any(|t| t.id == track.id) {
                tracks.insert(track.id.clone());
            }
        }
        // 原GUI分组根需要保留，父链只沿实际宿主图，未知/循环不能创建虚构轨道。
        for id in tracks.clone() {
            let mut current = id;
            let mut visited = BTreeSet::new();
            while visited.insert(current.clone()) {
                let track = timeline
                    .tracks
                    .iter()
                    .find(|track| track.id == current)
                    .ok_or("workspace track identity missing")?;
                let Some(parent) = &track.parent_id else {
                    break;
                };
                tracks.insert(parent.clone());
                current = parent.clone();
            }
            if timeline
                .tracks
                .iter()
                .find(|track| track.id == current)
                .is_some_and(|track| track.parent_id.is_some())
            {
                return Err("workspace track parent cycle".into());
            }
        }
        timeline.tracks.retain(|track| tracks.contains(&track.id));
        timeline
            .params_by_root_track
            .retain(|id, _| tracks.contains(id));
        if !timeline
            .selected_track_id
            .as_ref()
            .is_some_and(|id| tracks.contains(id))
        {
            timeline.selected_track_id = timeline.tracks.first().map(|track| track.id.clone());
        }
        if !timeline
            .selected_clip_id
            .as_ref()
            .is_some_and(|id| clips.contains(id))
        {
            timeline.selected_clip_id = timeline.clips.first().map(|clip| clip.id.clone());
        }
        if let Some(tempo) = self.clock.tempo() {
            timeline.bpm = tempo;
        }
        Ok(timeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ara::model::ModelHandle;
    use crate::render::extension::ExtensionOwner;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::ExtensionRoles;
    use std::sync::Arc;

    fn fixture() -> (
        ModelHandle,
        Vec<Arc<ExtensionOwner>>,
        Vec<Box<u8>>,
        Vec<*const ara2_bridge::sys::ARAPlugInExtensionInstance>,
    ) {
        let model = ModelHandle::new();
        let document = model.session();
        *document.timeline.lock().unwrap()=Some(serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"same-name","order":0},{"id":"b","name":"same-name","order":1}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":0.5,"takes":[{"id":"ta","source_path":"shared-source"}]},
                {"id":"cb","name":"B","track_id":"b","start_sec":0,"length_sec":0.5,"takes":[{"id":"tb","source_path":"shared-source"}]}]
        })).unwrap());
        let identities = vec![Box::new(0_u8), Box::new(0_u8)];
        let mut owners = vec![];
        let mut interfaces = vec![];
        for (index, id) in ["ca", "cb"].into_iter().enumerate() {
            let key = (&*identities[index] as *const u8) as u64;
            region_owners()
                .lock()
                .unwrap()
                .register(key, document.id, index)
                .unwrap();
            document.clip_ids.lock().unwrap().insert(key, id.into());
            let owner = Arc::new(ExtensionOwner::default());
            let raw = owner
                .bind_to_document(
                    document.clone(),
                    ApiGeneration::V2Final,
                    ExtensionRoles::all(),
                    ExtensionRoles::PLAYBACK_RENDERER,
                    None,
                )
                .unwrap();
            // SAFETY: 区域身份与原生extension保留到fixture销毁。
            unsafe {
                let ext = &*raw;
                ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                    ext.playbackRendererRef,
                    key as *mut _,
                );
            }
            owners.push(owner);
            interfaces.push(raw);
        }
        document.ready.store(true, Ordering::Release);
        (model, owners, identities, interfaces)
    }
    /// 同名/同源不能被合成一条轨，多个renderer也不能把同一个区域重复显示。
    #[test]
    fn shared_source_tracks_form_one_deduplicated_document_workspace() {
        let (model, owners, ids, _) = fixture();
        let document = model.session();
        let (second, _other_owners, _other_ids, _other_interfaces) = fixture();
        assert_ne!(document.id, second.session().id);
        assert!(document
            .workspace_scope()
            .unwrap()
            .regions
            .is_disjoint(&second.session().workspace_scope().unwrap().regions));
        let extra = Arc::new(ExtensionOwner::default());
        let raw = extra
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .unwrap();
        let key = (&*ids[0] as *const u8) as u64;
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        assert_eq!(document.workspace_scope().unwrap().regions.len(), 2);
        let timeline = document.workspace_timeline().unwrap();
        assert_eq!(timeline.tracks.len(), 2);
        assert_eq!(timeline.clips.len(), 2);
        assert_eq!(timeline.tracks[0].id, "a");
        assert_eq!(timeline.tracks[1].id, "b");
        owners[0].stop_editor();
        extra.stop_editor();
        let left = document.workspace_timeline().unwrap();
        assert_eq!(left.tracks.len(), 1);
        assert_eq!(left.tracks[0].id, "b");
    }
    /// 宿主真正移除assignment推进scope版本，空scope不能回退成全文档；关闭doc明确拒绝。
    #[test]
    fn assignments_and_component_close_revoke_workspace_scope_without_global_fallback() {
        let (model, owners, ids, interfaces) = fixture();
        let document = model.session();
        let before = document.scope_revision.load(Ordering::Acquire);
        let key = (&*ids[0] as *const u8) as u64;
        unsafe {
            let ext = &*interfaces[0];
            ((*ext.playbackRendererInterface)
                .removePlaybackRegion
                .unwrap())(ext.playbackRendererRef, key as *mut _);
        }
        assert!(document.scope_revision.load(Ordering::Acquire) > before);
        owners[1].stop_editor();
        let empty = document.workspace_timeline().unwrap();
        assert!(empty.clips.is_empty());
        assert!(empty.tracks.is_empty());
        document.close();
        assert!(document.workspace_scope().is_err());
    }
}
