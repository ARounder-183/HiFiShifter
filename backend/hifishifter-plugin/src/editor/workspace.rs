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

/// 把宿主枚举到的全部 take 投影进显示用 Clip。
///
/// 【为什么只在集合或 active 变化时重建】take 集合稳定时不动它，才能保住 slip/trim 的
/// 乐观预览（它们写在 active take 的扁平 `source_start_sec`/`source_end_sec` 上）。
/// 一旦重建就 `normalize_takes()`，把 active take 重新物化 —— 所以只在该重建的时候重建。
/// active 变化必须算作"该重建"：宿主换了当前 take，界面却还指着旧的那条 lane，是错的。
///
/// 【安全底线】**除被 ARA 授权的那一个 take 外，任何 take 都不带 `source_path`**。
/// 这份清单是显示占位，PCM 只能经 ARA 授权取得（`render/source.rs`）；非 active take
/// 更不在本实例的 ARA 范围内。授权 take 由 `authorized_take` 指名（见
/// `DocumentSession::authorized_takes`），本函数只为它保留已有的授权媒体投影。
fn sync_host_takes(
    clip: &mut hifishifter_kernel::state::Clip,
    item: &crate::host::reaper::UiItem,
    prefix: &impl Fn(&str) -> String,
    authorized_take: Option<&str>,
    // 分割出来的右半段：父段已被授权的媒体身份（见 `present_host_inventory`）。
    // 右半段没有自己的 region，所以 `authorized_take` 是 `None`；但它的源与父段是
    // 同一个文件、父段已授权 ⇒ 可以借用。这是**显示**的补位，不是新的授权。
    inherited_media: Option<&hifishifter_kernel::state::ClipTake>,
) {
    let expected: Vec<String> = item
        .takes
        .iter()
        .map(|take| prefix(&take.geometry.take_id))
        .collect();
    if expected.is_empty() {
        return;
    }
    let active = item
        .takes
        .iter()
        .find(|take| take.active)
        .map(|take| prefix(&take.geometry.take_id));
    // 循环源是 item 级属性，同一个 item 的所有 take 读到同一个值。
    let item_loop = item
        .takes
        .first()
        .is_some_and(|take| take.geometry.loop_source);
    let unchanged = clip.active_take_id == active
        && clip.takes.len() == expected.len()
        && clip
            .takes
            .iter()
            .zip(&expected)
            .all(|(take, id)| &take.id == id)
        // 【为什么循环也进这条判据】它影响渲染（内核按整份媒体回绕）与渲染缓存键。
        // 漏掉它时，用户在 REAPER 里开关"循环源"后这里会早退，界面停在旧状态。
        && clip.loop_enabled == item_loop;
    if unchanged {
        return;
    }
    // 【为什么必须把授权媒体搬到新 take 上】这份重建从宿主元数据造 take，而宿主元数据
    // **按设计**不带 `source_path`；随后 `normalize_takes()` 会把 active take 物化到扁平
    // 投影上，于是 `clip.source_path` 被清空 —— 每个 ARA 已授权片段都会变成"等待宿主
    // 音频"的斜纹占位（连波形一起消失）。此前 ARA 片段的 take id 是 `ara-clip-N-take-1`，
    // 与宿主 GUID 永不相等，所以这条重建**每次清单呈现都会发生**。
    //
    // 授权媒体只挂到 GUID 与记录相等的那一个 take 上：用户在 REAPER 里换了 active take
    // 而 ARA 尚未重新认领时，GUID 对不上 ⇒ 一个都不挂（宁可显示占位，也不把上一个 take
    // 的采样挂到新 take 上）。
    let granted = authorized_take.and_then(|guid| {
        item.takes
            .iter()
            .position(|take| take.geometry.take_id == guid)
    });
    let takes: Vec<serde_json::Value> = item
        .takes
        .iter()
        .map(|take| {
            let g = &take.geometry;
            serde_json::json!({
                "id": prefix(&g.take_id),
                "name": take.name,
                "source_start_sec": g.source_start_sec,
                "source_end_sec": g.source_start_sec + g.duration_sec * g.playback_rate,
                "playback_rate": g.playback_rate,
                "channel_mode": g.channel_mode,
                // 读不出来时留 false（不显示倒放标记），不编造"没倒放"的结论。
                "reversed": take.reversed.unwrap_or(false),
                // 循环源（item 级）。内核的 `loop_enabled` 语义与 REAPER 的循环源一致：
                // 对**整份媒体**取模回绕，而插件物化的 PCM 就是完整源。
                "loop_enabled": g.loop_source,
            })
        })
        .collect();
    let Ok(mut rebuilt) = serde_json::from_value::<Vec<hifishifter_kernel::state::ClipTake>>(
        serde_json::Value::Array(takes),
    ) else {
        return;
    };
    // 把媒体身份搬到该搬的那个 take 上。只搬**媒体身份**字段：源窗口 / 倍率 / 名字 /
    // 声道模式都是宿主几何，由上面的清单提供。
    //
    // 【搬到哪一个】
    // - 自己的 region 已认领（`granted`）→ 挂到 GUID 与授权记录相等的那一个 take；
    // - 分割出来的右半段（`inherited_media`）→ 挂到自己的 active take —— 它的源与父段
    //   是同一个文件，所以父段的媒体身份对它是成立的；
    // - 都不是 → 一个都不挂（宁可显示占位，也不把别的 take 的采样挂上来）。
    let media: Option<(usize, hifishifter_kernel::state::ClipTake)> = match granted {
        Some(index) => {
            // 授权媒体的来源取现有 active take（`normalize_takes` 保证它与扁平投影一致）；
            // 无 take 的扁平 clip 退回投影本身。
            let owned = if clip.takes.is_empty() {
                hifishifter_kernel::state::ClipTake::from_clip(clip)
            } else {
                clip.active_take().clone()
            };
            Some((index, owned))
        }
        None => inherited_media.and_then(|media| {
            item.takes
                .iter()
                .position(|take| take.active)
                .map(|index| (index, media.clone()))
        }),
    };
    if let Some((index, media)) = media {
        let target = &mut rebuilt[index];
        target.source_path = media.source_path;
        target.source_path_relative = media.source_path_relative;
        target.duration_sec = media.duration_sec;
        target.duration_frames = media.duration_frames;
        target.source_sample_rate = media.source_sample_rate;
        target.source_file_fingerprint = media.source_file_fingerprint;
        target.source_file_mtime = media.source_file_mtime;
        target.source_file_size = media.source_file_size;
        target.waveform_preview = media.waveform_preview;
        target.pitch_range = media.pitch_range;
    }
    // take 级音量在宿主侧是 item 音量（`display_item_gain` 会把两者对齐）；
    // 这里先按当前投影播种，紧接着的 display_item_gain 会统一覆盖。
    for take in &mut rebuilt {
        take.gain = clip.gain;
    }
    clip.takes = rebuilt;
    clip.active_take_id = active;
    clip.normalize_takes();
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
        // 委托 fade 的样式（宿主把边界包络交给 HFS 渲染时，shape/dir 由插件状态决定）。
        // 一次取出，避免在 track/item 双层循环里反复取锁。
        let delegated_fades = self.edits.lock().unwrap().fades.clone();
        // 哪些 item 的边界包络被委托给 HFS（按 item GUID 收集，判据与
        // `project_ui_fades_locked` 相同）。
        let delegated_items: std::collections::BTreeSet<String> = {
            let regions = self.regions.lock().unwrap();
            let items = self.region_items.lock().unwrap();
            items
                .iter()
                .filter(|(key, _)| {
                    regions.get(key).is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
                })
                .map(|(_, item)| item.clone())
                .collect()
        };
        // item GUID → 该 item 在 ARA 授权那一刻的 active take GUID。
        //
        // 一个 item 只有一个 active take，多个 region 认领同一 item 时记录的都是同一个
        // GUID，所以这里按 item 取一个即可（见 `DocumentSession::authorized_takes`）。
        let authorized_takes: std::collections::HashMap<String, String> = {
            let items = self.region_items.lock().unwrap();
            let takes = self.authorized_takes.lock().unwrap();
            items
                .iter()
                .filter_map(|(key, item)| takes.get(key).map(|take| (item.clone(), take.clone())))
                .collect()
        };
        let tracks = self.ui_tracks.lock().unwrap();
        if tracks.is_empty() {
            return;
        }
        // 分割出来的右半段：宿主还没为它分配 region，所以 `authorized_takes` 里没有它。
        // 但分割**不改变音频源**，父段已被授权 —— 于是把父段的授权媒体身份借给右半段，
        // 让它立刻有波形，而不是停在"等待 REAPER 提供音频"的占位里。
        //
        // 【为什么只借**媒体身份**】源窗口 / 倍率 / 名字都是右半段自己的宿主几何
        // （由下面的清单提供）；这里只补 `source_path` 一族的字段，与 `sync_host_takes`
        // 的嫁接同一原则：同一 `audio_source`、已在授权范围内。
        let inherited_media: std::collections::HashMap<
            String,
            hifishifter_kernel::state::ClipTake,
        > = {
            let lineage = self.split_media_from.lock().unwrap();
            lineage
                .iter()
                .filter_map(|(right, left)| {
                    let parent_id = prefix(&format!("ara-item-{left}"));
                    let parent = timeline.clips.iter().find(|clip| clip.id == parent_id)?;
                    // 父段自己也可能还没拿到音频（例如它也是被隔离的倒放片段）——
                    // 那就什么都不借，右半段照常显示为"在途"。
                    parent.source_path.as_ref()?;
                    Some((right.clone(), parent.active_take().clone()))
                })
                .collect()
        };
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
                sync_host_takes(
                    clip,
                    item,
                    &prefix,
                    authorized_takes.get(&g.item_id).map(String::as_str),
                    inherited_media.get(&g.item_id),
                );
                display_item_gain(clip, g.item_gain);
                clip.track_id = track_id.clone();
                // 【为什么空名不覆盖】宿主 take 名可能为空（`GetTakeName` 返回空串）。
                // 用空串盖掉 ARA 映射带来的名字（region / source 名）会让片段在画布与
                // 浮标上都无字可显示 —— 名字是用户识别片段的主要线索。
                if !item.name.trim().is_empty() {
                    clip.name = item.name.clone();
                }
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
                // 【为什么按轴分流】"曲率"落在哪个宿主字段取决于宿主版本（≤7.80
                // `D_FADE*DIR`，≥7.81 `D_FADE*DIR_NEW`）。这里**必须**与
                // `project_ui_fades_locked` 同口径 —— 本函数在它**之后**跑（见
                // `ensure_loaded`：先 snapshot 再 inventory），无条件取旧轴会把刚投影好
                // 的新轴曲率覆盖成一个被重映射过的旧值（实测 7.81 上新轴 0.5 读回旧轴
                // 是 0）。表现就是"拖了曲率，滑杆/曲线又跳回另一个值"。
                let (in_dir, out_dir) = match g.fade_axes_new {
                    Some(true) => (g.fade_in_dir_new, g.fade_out_dir_new),
                    _ => (g.fade_in_dir, g.fade_out_dir),
                };
                clip.fade_in_dir = in_dir;
                clip.fade_out_dir = out_dir;
                // 委托 fade 的 item：宿主把边界包络交给 HFS 渲染，shape/dir 由插件状态
                // 决定 —— 与 `project_ui_fades_locked` 的同一分支、同一判据。
                if delegated_items.contains(&g.item_id) {
                    let style = delegated_fades.get(&g.item_id).cloned().unwrap_or_default();
                    clip.fade_in_shape = style.in_shape;
                    clip.fade_out_shape = style.out_shape;
                    clip.fade_in_dir = style.in_dir;
                    clip.fade_out_dir = style.out_dir;
                }
                timeline.project_sec = timeline.project_sec.max(g.start_sec + g.duration_sec);
            }
        }
        timeline.clips.retain(|clip| {
            !clip.id.starts_with(&prefix("ara-item-")) || present.contains(&clip.id)
        });
        timeline.clips.retain(|clip| {
            !owned.contains(&clip.track_id) || !clip.id.starts_with(&prefix("ara-clip-"))
        });
        // 分割谱系回收：两段都不在宿主清单里时删掉这条边，免得它随分割次数无限增长
        // （与 `ParameterAtlas::split_parents` 的 retain 同一手法）。
        {
            let mut lineage = self.split_media_from.lock().unwrap();
            lineage.retain(|right, left| {
                present.contains(&prefix(&format!("ara-item-{right}")))
                    && present.contains(&prefix(&format!("ara-item-{left}")))
            });
        }
        {
            let known = self.ui_known_tracks.lock().unwrap();
            timeline.tracks.retain(|track| {
                !known.iter().any(|id| prefix(id) == track.id) || owned.contains(&track.id)
            });
        }
        // 【为什么在这里把 `known` 推进到本轮清单，而不是只做并集】`known` 的语义是
        // "宿主清单里的轨道"，用来剔除**宿主已不再报告**的轨道。只做并集的话，一条曾经
        // 进过清单、后来离开 folder 的轨道会被永久记住，于是每一轮都被从时间线上剔掉 ——
        // 哪怕它仍有 ARA clip、本该作为普通 ARA 轨道显示。推进到本轮之后，剔除只发生在
        // "上次呈现有、这次没有"的那一次；此后它按普通 ARA 轨道对待。
        *self.ui_known_tracks.lock().unwrap() =
            tracks.values().map(|host| host.id.clone()).collect();
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
            .flat_map(|owner| owner.host_geometries_locked(self))
            .filter_map(|bound| {
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
            for bound in owner.host_geometries_locked(self) {
                if bound.geometry.item_id == item {
                    return identities.get(&bound.region_key).cloned();
                }
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
            for bound in owner.host_geometries_locked(self) {
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
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
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
    }
    /// 无宿主getter的短事务装饰；调用者不得仍持编辑timeline锁。
    pub(crate) fn decorate_host_fades(&self, payload: &mut serde_json::Value, namespace: &str) {
        let _transaction = self.transaction.lock().unwrap();
        if self.is_alive() {
            self.decorate_host_fades_locked(payload, namespace, &self.edits.lock().unwrap().fades);
            self.decorate_host_media_locked(payload, namespace);
        }
    }
    /// 逐 clip 的宿主媒体状态（语言无关分类，文案由前端 catalog 本地化）。
    ///
    /// 【为什么需要它】前端此前用一个"没有 `source_path`"的布尔同时表示四种互不相容的
    /// 情形：刚分割/裁切后的**在途**、非 active take 的**正常**、folder 父轨的**永远
    /// 拿不到**、宿主换 take 后的**授权失效**。四者给出同一句"等待 REAPER 提供音频"，
    /// 于是"我刚分割了一下"看起来像"插件坏了"。
    ///
    /// 【为什么分类留在后端】与 `host_audio` 同一纪律：后端只给语言无关的分类名
    /// （`HostAudioState` 的先例），文案按 catalog 本地化。前端对不认识的分类名
    /// 沿用上一次已知值，不猜。
    pub(crate) fn decorate_host_media_locked(
        &self,
        payload: &mut serde_json::Value,
        namespace: &str,
    ) {
        // 被某个 region 认领的 item；不在集合里的还没拿到 ARA 音频。
        let claimed: std::collections::BTreeSet<String> = self
            .region_items
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect();
        // folder 父轨：本实例**永远**拿不到组内子轨的音频（见 `HostAudioState`）。
        let folder_parent = self.renderer_owners().into_iter().any(|owner| {
            owner.host_audio_status().state
                == crate::render::extension::HostAudioState::FolderParentWithoutRegions
        });
        let Some(clips) = payload["clips"].as_array_mut() else {
            return;
        };
        for track in self.ui_tracks.lock().unwrap().values() {
            for item in &track.items {
                let Some(clip) = clips.iter_mut().find(|clip| {
                    clip["id"] == format!("{namespace}ara-item-{}", item.geometry.item_id)
                }) else {
                    continue;
                };
                let has_source = clip["source_path"]
                    .as_str()
                    .is_some_and(|path| !path.is_empty());
                let state = if has_source {
                    "ready"
                } else if clip["reversed"].as_bool() == Some(true) {
                    // 倒放被隔离：插件渲染不出反向内容，这一条由 REAPER 处理。
                    // 与"还没拿到音频"不是一回事，文案必须分开。
                    "reversed"
                } else if folder_parent && !claimed.contains(&item.geometry.item_id) {
                    "unavailable"
                } else {
                    // 其余都是**在途**：宿主可能还在分配，不是故障。
                    "pending"
                };
                clip["host_media"] = serde_json::json!(state);
            }
        }
    }
    /// 把宿主音频读数写进 GUI 载荷（语言无关分类，文案由前端 catalog 本地化）。
    ///
    /// 【为什么取编辑器实例的读数】GUI 就挂在这个实例上（role = 2），它才是用户看到的
    /// 那个窗口；playback 实例没有 GUI。找不到编辑器实例时**不写**该字段 —— 前端沿用
    /// 上一次已知值，而不是凭空报一个"正常"。
    ///
    /// 【为什么分类留在后端】与 `ara_host_fields:` 同一原则：后端只给语言无关的分类
    /// （见 `render::extension::HostAudioState`），文案按 catalog 本地化。
    pub(crate) fn decorate_host_audio(&self, payload: &mut serde_json::Value) {
        let Some(owner) = self
            .renderer_owners()
            .into_iter()
            .find(|owner| owner.is_editor_only())
        else {
            return;
        };
        let status = owner.host_audio_status();
        payload["host_audio"] = serde_json::json!({
            "state": status.state.as_str(),
            "waiting_clips": status.waiting_items,
        });
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
            for bound in owner.host_geometries_locked(self) {
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
                // 【为什么按轴分流】"曲率"落在哪个宿主字段取决于宿主版本
                // （≤7.80 `D_FADE*DIR`，≥7.81 `D_FADE*DIR_NEW`）。取错的话滑杆
                // 显示和编辑的都是**不是权威**的那个值 —— 在 7.81+ 上表现为
                // "拖了滑杆但淡变没变"。
                let (in_dir, out_dir) = match geometry.fade_axes_new {
                    Some(true) => (geometry.fade_in_dir_new, geometry.fade_out_dir_new),
                    _ => (geometry.fade_in_dir, geometry.fade_out_dir),
                };
                clip.fade_in_dir = in_dir;
                clip.fade_out_dir = out_dir;
                if self
                    .regions
                    .lock()
                    .unwrap()
                    .get(&bound.region_key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
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
    }
    /// 只把宿主明确委托的一端写进原kernel；另一端仍归宿主，不能凭能力广告重复烘焙。
    pub(crate) fn project_audio_fades_locked(
        &self,
        timeline: &mut TimelineState,
        edits: &crate::state_channel::EditState,
    ) -> Result<(), String> {
        let identities = self.clip_ids.lock().unwrap().clone();
        let regions = self.regions.lock().unwrap().clone();
        // 先把所有 owner 的逐 region 绑定摊平成一张表：委托 fade 的 clip 可能属于
        // 多 region owner（folder 轨上的 FX），按 owner 逐个 find_map 会漏。
        let geometries = self
            .renderer_owners()
            .iter()
            .flat_map(|owner| owner.host_geometries_locked(self))
            .map(|bound| (bound.region_key, bound.geometry))
            .collect::<std::collections::HashMap<_, _>>();
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
            let geometry = geometries
                .get(&key)
                .cloned()
                .ok_or("delegated fade host geometry pending")?;
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
            for bound in owner.host_geometries_locked(self) {
                let Some(id) = identities.get(&bound.region_key) else {
                    continue;
                };
                if let Some(clip) = timeline.clips.iter_mut().find(|clip| &clip.id == id) {
                    clip.muted = bound.geometry.muted;
                }
            }
        }
    }
    /// 插件自有的音阶投影：把 `PluginMusicalContext` 播种到时间线上。
    ///
    /// 【为什么必须每次重建后做】`ara::mapping::ara_document_to_timeline` 只从
    /// tracks/clips/bpm/project_sec 重建 `TimelineState`，`project_scale_notes` 与
    /// `tempo_map` 都不在其中 —— 不播种就等于"用户选了 Gb、内核按 C 渲染"，
    /// 且渲染缓存键（`scale-signature`）也一起错。这与
    /// [`Self::project_host_mutes_locked`] 是同一条纪律：**插件自有的、会影响渲染的
    /// 状态，每次重建后重新投影，而不是指望它留在时间线上**。
    ///
    /// 【为什么用缓存】本函数在每次 `workspace_timeline_locked` 里跑（渲染与 GUI
    /// 两条路径都经过它），而 `settings()` 会克隆整份 `UiSettings`。命中缓存时
    /// 只付一次原子读 + 一次互斥。
    pub(crate) fn project_plugin_musical_context_locked(&self, timeline: &mut TimelineState) {
        let key = crate::render::document::MusicalProjectionKey::new(
            crate::settings_store::revision(),
            *self.host_meter.lock().unwrap(),
        );
        let mut cache = self.musical_projection.lock().unwrap();
        if cache.as_ref().is_none_or(|(cached, _)| *cached != key) {
            let musical = crate::settings_store::settings().plugin_musical_context;
            let meter = *self.host_meter.lock().unwrap();
            let (project_scale_notes, tempo_map) =
                hifishifter_kernel::state::plugin_musical_projection(
                    &musical,
                    meter.bpm,
                    meter.numerator,
                    meter.denominator,
                );
            *cache = Some((
                key,
                crate::render::document::PluginMusicalProjection {
                    project_scale_notes,
                    tempo_map,
                },
            ));
        }
        if let Some((_, projection)) = cache.as_ref() {
            timeline.project_scale_notes = projection.project_scale_notes.clone();
            timeline.tempo_map = projection.tempo_map.clone();
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
        self.project_plugin_musical_context_locked(&mut timeline);
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

    /// 离开宿主清单的轨道只该被剔除**一次**，不能被永久记成"宿主轨道"。
    ///
    /// 【回归】`ui_known_tracks` 旧实现只做并集：一条曾进过清单、后来离开 folder 的
    /// 轨道会被永久记住，于是每一轮呈现都把它从时间线上剔掉 —— 哪怕它仍有 ARA clip、
    /// 本该作为普通 ARA 轨道显示。
    #[test]
    fn a_track_leaving_the_host_inventory_is_dropped_once_not_forever() {
        let model = ModelHandle::new();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let api = Arc::new(host.client());
        let base = api.ui_track(&|| true).unwrap();

        let mut staying = base.clone();
        staying.id = "host-staying".into();
        staying.guid = "{66666666-6666-6666-6666-666666666666}".into();
        staying.items.clear();
        let mut leaving = base.clone();
        leaving.id = "host-leaving".into();
        leaving.guid = "{77777777-7777-7777-7777-777777777777}".into();
        leaving.items.clear();

        let mut timeline = TimelineState::default();
        timeline.tracks.clear();
        {
            let mut tracks = document.ui_tracks.lock().unwrap();
            tracks.insert(staying.guid.clone(), staying.clone());
            tracks.insert(leaving.guid.clone(), leaving.clone());
        }
        document.present_host_inventory(&mut timeline, "");
        assert!(timeline
            .tracks
            .iter()
            .any(|track| track.id == "host-staying"));
        assert!(timeline
            .tracks
            .iter()
            .any(|track| track.id == "host-leaving"));

        // 一条轨道离开清单（例如被拖出 folder），另一条还在 —— 清单非空，本轮照常呈现。
        document.ui_tracks.lock().unwrap().remove(&leaving.guid);
        document.present_host_inventory(&mut timeline, "");
        assert!(
            !timeline
                .tracks
                .iter()
                .any(|track| track.id == "host-leaving"),
            "离开清单的轨道必须被剔除一次"
        );
        assert!(
            !document
                .ui_known_tracks
                .lock()
                .unwrap()
                .contains("host-leaving"),
            "剔除之后不得永久残留，否则此后每一轮都会被误剔"
        );

        // 关键回归：此后它以普通 ARA 轨道身份出现时必须留得住。
        let mut timeline = TimelineState::default();
        timeline.tracks = vec![serde_json::from_value(
            serde_json::json!({"id":"host-leaving","name":"now an ARA track","order":0}),
        )
        .unwrap()];
        document.present_host_inventory(&mut timeline, "");
        assert!(
            timeline
                .tracks
                .iter()
                .any(|track| track.id == "host-leaving"),
            "不再属于宿主清单的轨道要按普通 ARA 轨道保留"
        );
        document.close();
    }
}
