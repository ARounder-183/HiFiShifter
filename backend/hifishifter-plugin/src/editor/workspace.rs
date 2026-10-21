//! 工程编辑权限是同一活ARA文档renderer区域并集，音频输出仍按各renderer原分配隔离。
use crate::render::document::DocumentSession;
use crate::render::ownership::region_owners;
use crate::sync::MutexExt;
use hifishifter_kernel::state::TimelineState;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

/// 一个 item 没有 ARA 音频多久之后，就从"在途"改判为"等不到了"。
///
/// 【为什么是 2 秒】一次正常的 ARA 重认领（`begin_editing` … `end_editing`）在数百
/// 毫秒内完成；2 秒足够覆盖一次分割/粘贴/切换 active take 后的回流。超过它仍无源，
/// 再显示"正在等待"就是在骗用户 —— 转成带原因的 `unavailable`，让用户知道该做什么。
///
/// 【为什么 `pub(crate)`】应用状态（`EditorSession::state` 的 `pending`）也必须按同一
/// 口径判断"是否仍在争取"：只数"缺 `source_path` 的片段"会让终态（folder 父轨 /
/// 读取失败 / 超期）把状态永久钉在"进行中"。
pub(crate) const PENDING_MEDIA_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

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
///
/// 【为什么无条件重建，而不是"未变则早退"】此前的早退判据是"active take id / take id
/// 列表 / `loop_enabled` 都没变就跳过"。那是一个**枚举型**判据：宿主能改、而判据没
/// 把一份**已授权媒体身份**搬到 `take` 上。
///
/// 【为什么只搬这些字段】源窗口 / 倍率 / 名字 / 声道模式都是宿主几何，由清单或 ARA
/// 映射提供；这里只补"这份采样是哪一个文件"这一族身份字段（与 `sync_host_takes` 的
/// 嫁接同一原则）。两条路径（清单重建、渲染派生）共用它，避免字段集再次分叉。
fn apply_take_media(
    take: &mut hifishifter_kernel::state::ClipTake,
    media: &hifishifter_kernel::state::ClipTake,
) {
    take.source_path = media.source_path.clone();
    take.source_path_relative = media.source_path_relative.clone();
    take.duration_sec = media.duration_sec;
    take.duration_frames = media.duration_frames;
    take.source_sample_rate = media.source_sample_rate;
    take.source_file_fingerprint = media.source_file_fingerprint;
    take.source_file_mtime = media.source_file_mtime;
    take.source_file_size = media.source_file_size;
    take.waveform_preview = media.waveform_preview.clone();
    take.pitch_range = media.pitch_range.clone();
}

/// 分割派生：把父段的授权媒体身份借给右半段。
///
/// 【为什么需要】宿主分割**不改变音频源** —— 两半指向同一个文件。父段已被 ARA 授权，
/// 所以右半段可以立刻显示波形，不必等宿主为它重新分配 region。缺了这条，分割后右半段
/// 会停在一段"等待 REAPER 提供音频"的占位里（用户报障过）。
///
/// 【为什么两条路径都要调】清单路径（`present_host_inventory`，经 `sync_host_takes`
/// 的 `inherited_media`）与渲染路径（`workspace_timeline_locked`）各建一份时间线。
/// 只给清单补，界面显示正常，但**渲染出来的右半段仍是无源静音**（上一篇 Plan 明写
/// "两处都要"，只做了一半）。
///
/// 【为什么只借媒体身份】源窗口 / 倍率 / 名字都是右半段自己的宿主几何；这里只补
/// `source_path` 一族字段，与 `sync_host_takes` 的嫁接同一原则：同一 `audio_source`、
/// 已在授权范围内。父段自己也可能还没拿到音频（例如它也是被隔离的倒放片段）—— 那就
/// 什么都不借，右半段照常显示为"在途"。
pub(crate) fn derive_split_media(
    timeline: &mut TimelineState,
    lineage: &std::collections::HashMap<String, String>,
    prefix: &impl Fn(&str) -> String,
) {
    for (right, left) in lineage {
        let right_id = prefix(&format!("ara-item-{right}"));
        let left_id = prefix(&format!("ara-item-{left}"));
        let Some(media) = timeline
            .clips
            .iter()
            .find(|clip| clip.id == left_id)
            .filter(|parent| parent.source_path.is_some())
            .map(|parent| parent.active_take().clone())
        else {
            continue;
        };
        let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == right_id) else {
            continue;
        };
        // ARA 已经认领过这一条 ⇒ 它有自己的授权媒体，不覆盖。
        if clip.source_path.is_some() {
            continue;
        }
        if clip.takes.is_empty() {
            let mut take = hifishifter_kernel::state::ClipTake::from_clip(clip);
            apply_take_media(&mut take, &media);
            clip.takes.push(take);
            clip.active_take_id = clip.takes.first().map(|take| take.id.clone());
        } else {
            let index = clip
                .takes
                .iter()
                .position(|take| Some(take.id.as_str()) == clip.active_take_id.as_deref())
                .unwrap_or(0);
            apply_take_media(&mut clip.takes[index], &media);
        }
        clip.normalize_takes();
    }
}

/// 覆盖的属性（`channel_mode`、`reversed`、take 名、源窗口、倍率）改了之后这里会早退，
/// 界面停在旧值 —— 用户报的"改了声道模式，闪一下又变回去"就是它（乐观更新先亮，下一份
/// 权威 payload 又发旧值）。枚举永远补不完，所以现在**每次都重建**，并把宿主不报告的
/// 插件自有字段（MIDI 音符、声道判定档案、包络、拉伸标记）从旧 take **合并**过来。
fn sync_host_takes(
    clip: &mut hifishifter_kernel::state::Clip,
    item: &crate::host::UiItem,
    prefix: &impl Fn(&str) -> String,
    authorized_take: Option<&str>,
    // ARA 授权那一刻 active take 的**源文件路径**（`authorized_sources`）。
    // 只在 GUID 对不上时作为回退身份（见下面的 `granted`）。
    authorized_source: Option<&str>,
    // 分割出来的右半段：父段已被授权的媒体身份（见 `present_host_inventory`）。
    // 右半段没有自己的 region，所以 `authorized_take` 是 `None`；但它的源与父段是
    // 同一个文件、父段已授权 ⇒ 可以借用。这是**显示**的补位，不是新的授权。
    inherited_media: Option<&hifishifter_kernel::state::ClipTake>,
    // 本实例**手里有 PCM** 的那份身份（活着的 region 指向它，且它在 `edit_sources` 里）。
    //
    // 【为什么这一环比"保留上一轮"更强】它每一轮都能**重新求值**：clip 被整份重建
    // （`ara-region-*` 升级成 `ara-item-*`）时"上一轮"已经不存在了，而 region → source →
    // PCM 这条链仍然在。这正是"重开工程后全部 Clip 都在等音频"的那条断链。
    pcm_media: Option<&crate::editor::media_identity::PcmFacts>,
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
    // 【为什么必须把授权媒体搬到新 take 上】这份重建从宿主元数据造 take，而宿主元数据
    // **按设计**不带 `source_path`；随后 `normalize_takes()` 会把 active take 物化到扁平
    // 投影上，于是 `clip.source_path` 被清空 —— 每个 ARA 已授权片段都会变成"等待宿主
    // 音频"的斜纹占位（连波形一起消失）。
    //
    // 授权媒体只挂到 GUID 与记录相等的那一个 take 上：用户在 REAPER 里换了 active take
    // 而 ARA 尚未重新认领时，GUID 对不上 ⇒ 一个都不挂（宁可显示占位，也不把上一个 take
    // 的采样挂到新 take 上）。
    let granted = authorized_take
        .and_then(|guid| {
            item.takes
                .iter()
                .position(|take| take.geometry.take_id == guid)
        })
        // 【回退：同一 item 内、源文件与授权那一刻相同的 take】用户把 active take 换成
        // **同一个文件**的另一个 take（复制 take、切换 take）时 GUID 变了，内容却没变。
        // 文件名相同即证明"同一份已授权 PCM"，可以安全地把授权媒体挂上去 —— 否则明明
        // 有音频的片段会显示占位。路径为空（读不到）时不回退，保持"宁可不挂"。
        .or_else(|| {
            let wanted = authorized_source.filter(|name| !name.is_empty())?;
            item.takes
                .iter()
                .position(|take| take.geometry.source_file_name.as_deref() == Some(wanted))
        });
    // 【为什么按 id 合并而不是整体替换】宿主元数据只描述**几何与媒体**；take 上的
    // 插件自有内容（MIDI 音符、声道判定档案、包络、拉伸标记、源声道数）宿主根本不报告。
    // 整体替换会在每次清单刷新时把它们抹掉。先按 id 建一份旧值索引，逐字段合并。
    let existing: std::collections::HashMap<String, &hifishifter_kernel::state::ClipTake> = clip
        .takes
        .iter()
        .map(|take| (take.id.clone(), take))
        .collect();
    let takes: Vec<serde_json::Value> = item
        .takes
        .iter()
        .map(|take| {
            let g = &take.geometry;
            let id = prefix(&g.take_id);
            let old = existing.get(&id);
            serde_json::json!({
                "id": id,
                "name": take.name,
                "source_start_sec": g.source_start_sec,
                "source_end_sec": g.source_start_sec + g.duration_sec * g.playback_rate,
                "playback_rate": g.playback_rate,
                "channel_mode": g.channel_mode,
                // 宿主报告的方向；读不到时 `take_reversed` 已落成 false（按正放）。
                "reversed": take.reversed,
                // 循环源（item 级）。内核的 `loop_enabled` 语义与 REAPER 的循环源一致：
                // 对**整份媒体**取模回绕，而插件物化的 PCM 就是完整源。
                "loop_enabled": g.loop_source,
                // 宿主不报告、必须从旧 take 继承的插件自有内容。
                "source_channels": old.and_then(|take| take.source_channels),
                "channel_decision": old.and_then(|take| take.channel_decision),
                "midi_note_data": old.and_then(|take| take.midi_note_data.clone()),
                "midi_fill_gaps": old.is_some_and(|take| take.midi_fill_gaps),
                "stretch_markers": old.map(|take| take.stretch_markers.clone()).unwrap_or_default(),
                "envelopes": old.and_then(|take| take.envelopes.clone()),
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
    // 【搬到哪一个】判据收敛在 `media_identity::choose_link`（单一入口，可单测）：
    // - 宿主这一次的观察与 item 对得上（`granted`）→ 挂到 GUID 相等的那一个 take；
    // - 分割出来的右半段（`inherited_media`）→ 挂到自己的 active take；
    // - **本实例手里有 PCM**（`pcm_media`）→ 挂到 active take；
    // - 前三环都缺席 → 保留上一轮已有的身份；
    // - 宿主换了 take → 一个都不挂（宁可占位，也不挂错的采样）。
    let host_switched_take = granted.is_none() && (authorized_take.is_some() || authorized_source.is_some());
    // 授权媒体的来源取现有 active take（`normalize_takes` 保证它与扁平投影一致）；
    // 无 take 的扁平 clip 退回投影本身。
    let owned_media = || {
        if clip.takes.is_empty() {
            hifishifter_kernel::state::ClipTake::from_clip(clip)
        } else {
            clip.active_take().clone()
        }
    };
    let active_index = || item.takes.iter().position(|take| take.active);
    let inherited = inherited_media.and_then(|media| active_index().map(|index| (index, media.clone())));
    // 【T-C1】"知道它是哪个源"与"这一轮有采样"分开：前者永远成立，后者看 has_pcm。
    let pcm = pcm_media.and_then(|facts| {
        active_index().map(|index| {
            let mut media = owned_media();
            facts.apply(&mut media);
            (index, media)
        })
    });
    let has_pcm = pcm_media.is_some_and(|facts| facts.has_pcm);
    // 【T22：**没有记录**不等于**没有授权**】
    //
    // `authorized_takes` / `authorized_sources` 是"某一刻观察到的授权事实"，
    // 而它只有一个写者（`remap_and_log`），且要求 `region_items` 与 `ui_tracks`
    // 在同一刻都就绪。重开工程时这个条件常常不成立（几何缓存被 `clear_renderers`
    // 的 revision 推进作废、`assignments` 尚未重发、宿主 item 几何还没稳定），
    // 于是那张表被写成**空**的。
    //
    // 此前这里把"查不到记录"直接当成"未授权"：重建出来的 take 一个都不带
    // `source_path`，`normalize_takes()` 再把扁平投影一起清掉 ⇒ **一次清单轮次
    // 剥掉全部 clip 的源**。用户看到的就是"重开工程后所有 Clip 都在等宿主音频"。
    let last_known = || {
        // 只有"已经有源"时才保留；从未有过源的 clip 仍然显示占位（宁可不挂）。
        //
        // 【为什么两处都要看】扁平投影（`clip.source_path`）是 `normalize_takes()`
        // 的产物，而清单刷新可能跑在它之前；take 列表里那一份才是真正的存储。
        let owned = clip
            .takes
            .iter()
            .find(|take| take.source_path.is_some())
            .cloned()
            .or_else(|| {
                clip.source_path
                    .as_ref()
                    .map(|_| hifishifter_kernel::state::ClipTake::from_clip(clip))
            })?;
        Some((active_index()?, owned))
    };
    // 宿主换了 take ⇒ 一个都不挂，连"记忆"也不许用（见 `choose_link` 的说明）。
    let remembered = if host_switched_take { None } else { last_known() };
    let media = match crate::editor::media_identity::choose_link(
        host_switched_take,
        granted.is_some(),
        inherited.is_some(),
        has_pcm,
        remembered.is_some() || pcm.is_some(),
    ) {
        crate::editor::media_identity::Link::Authorized => granted.map(|index| (index, owned_media())),
        crate::editor::media_identity::Link::Inherited => inherited,
        // 【为什么 LastKnown 还要再退一次到 pcm】那是"只有身份、这一轮没有采样"的
        // 事实：采样会在下一次读到时回来，而身份一旦丢掉就再也找不回来。
        crate::editor::media_identity::Link::Pcm => pcm,
        crate::editor::media_identity::Link::LastKnown => remembered.or(pcm),
        crate::editor::media_identity::Link::None => None,
    };
    if let Some((index, media)) = media {
        apply_take_media(&mut rebuilt[index], &media);
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
            .lock_or_recover()
            .values()
            .map(|track| (track.guid.clone(), format!("{namespace}{}", track.id)))
            .collect()
    }
    /// GUI清单刷新不覆盖插件私有父级和顺序；新轨未出现在私有表中时保持根级。
    pub(crate) fn present_private_groups(&self, timeline: &mut TimelineState, namespace: &str) {
        // 锁序固定为 edits → host_owned（见 `host_folder_children` 的说明）：
        // `project_private_group_view` 等路径在持 edits 时读 host_owned，反向获取会死锁。
        let edits = self.edits.lock_or_recover();
        let host_owned = self.host_folder_children.lock_or_recover().clone();
        edits
            .groups
            .apply(timeline, &self.group_aliases(namespace), &host_owned);
    }
    /// 显示参数根可包含无播放区域的空父轨；不创建任何clip或未经授权的源。
    fn add_group_view_tracks(&self, timeline: &mut TimelineState) {
        for host in self.ui_tracks.lock_or_recover().values() {
            if !timeline.tracks.iter().any(|track| track.id == host.id) {
                // 【为什么不用 unwrap】这条路径跑在**宿主 UI 线程**的清单刷新里，一次
                // panic 会带走整个 DAW 会话（与同文件 `present_host_inventory` 的注释
                // 同一条纪律）。`host.name` 是宿主给的字符串，反序列化失败时降级为
                // "跳过这条轨道"，其余轨道照常呈现。
                match serde_json::from_value::<hifishifter_kernel::state::Track>(
                    serde_json::json!({"id":host.id,"name":host.name,"order":host.order,
                    "compose_enabled":true,"pitch_analysis_algo":"nsf_hifigan_onnx"}),
                ) {
                    Ok(track) => timeline.tracks.push(track),
                    Err(error) => {
                        log::warn!("[ara] skipping empty host track {}: {error}", host.id);
                    }
                }
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
        let host_owned = self.host_folder_children.lock_or_recover().clone();
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
        let _transaction = self.transaction.lock_or_recover();
        let mut flat = self.workspace_timeline_locked()?;
        flat.selected_clip_id = selected;
        let edits = self.edits.lock_or_recover();
        // 锁序 edits → host_owned（见 `present_private_groups` 的说明）。
        let host_owned = self.host_folder_children.lock_or_recover().clone();
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
        let delegated_fades = self.edits.lock_or_recover().fades.clone();
        // 哪些 item 的边界包络被委托给 HFS（按 item GUID 收集，判据与
        // `project_ui_fades_locked` 相同）。
        let delegated_items: std::collections::BTreeSet<String> = {
            let regions = self.regions.lock_or_recover();
            let items = self.region_items.lock_or_recover();
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
            let items = self.region_items.lock_or_recover();
            let takes = self.authorized_takes.lock_or_recover();
            items
                .iter()
                .filter_map(|(key, item)| takes.get(key).map(|take| (item.clone(), take.clone())))
                .collect()
        };
        // item GUID → 授权那一刻 active take 的源文件路径（回退身份，见
        // `DocumentSession::authorized_sources`）。只在 take GUID 对不上时使用。
        let authorized_sources: std::collections::HashMap<String, String> = {
            let items = self.region_items.lock_or_recover();
            let sources = self.authorized_sources.lock_or_recover();
            items
                .iter()
                .filter_map(|(key, item)| {
                    sources
                        .get(key)
                        .map(|source| (item.clone(), source.clone()))
                })
                .collect()
        };
        // item GUID → 本实例**手里那份 PCM** 的身份（`Pcm` 环，见 `media_identity`）。
        //
        // 【为什么这一环是"重开工程后全部 Clip 都在等音频"的断点】上面两张表都来自
        // **宿主的一次观察**（`authorized_takes` / `authorized_sources`），而它们的唯一
        // 写者要求 `region_items` 与 `ui_tracks` 在同一刻就绪 —— 重开工程时这个条件
        // 常常不成立，于是两张表是空的，重建出来的 take 一个都不带 `source_path`。
        // 而音频其实一直在 `edit_sources` 里，并且"活着的 region 指向哪个源"是**文档级**
        // 事实（`regions` 由 `addPlaybackRegion` 写入，与宿主几何无关）。
        //
        // 【锁序】`region_items` → `regions` → `edit_sources`，与 `publish_readiness` 同向；
        // 前两张表先取完、立刻释放，最后才取 `edit_sources`（那里也是 `publish_source_pcm`
        // 的位置）。
        let pcm_by_item: std::collections::HashMap<
            String,
            crate::editor::media_identity::PcmFacts,
        > = {
            let pairs: Vec<(String, String)> = {
                let items = self.region_items.lock_or_recover();
                let regions = self.regions.lock_or_recover();
                items
                    .iter()
                    .filter_map(|(key, item)| {
                        let region = regions.get(key)?;
                        Some((item.clone(), region.audio_source_persistent_id.clone()))
                    })
                    .collect()
            };
            let sources = self.edit_sources.lock_or_recover();
            pairs
                .into_iter()
                .filter_map(|(item, source)| {
                    let pcm = sources.get(&source)?;
                    Some((
                        item,
                        crate::editor::media_identity::PcmFacts {
                            source_id: source,
                            sample_rate: pcm.sample_rate,
                            channels: pcm.planes.len().max(1) as u16,
                            frames: pcm.planes.first().map(Vec::len).unwrap_or(0),
                            has_pcm: true,
                        },
                    ))
                })
                .collect()
        };
        // item GUID → 本轮的媒体事实：**本轮 PCM 优先，其次历史事实**（只增不减）。
        //
        // 【为什么两层而不是只读 `edit_sources`】宿主收回样本访问时 `edit_sources` 的
        // 条目会保留，但**新文档**里它是空的；而"我们知道这条 item 的源是哪个 ARA
        // source"是文档级的既有事实，不该跟着采样一起消失。
        let media_by_item: std::collections::HashMap<
            String,
            crate::editor::media_identity::PcmFacts,
        > = {
            let mut facts: std::collections::HashMap<String, crate::editor::media_identity::PcmFacts> =
                self.media_facts.lock_or_recover().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            for (item, pcm) in &pcm_by_item {
                facts.insert(item.clone(), pcm.clone());
            }
            facts
        };
        let tracks = self.ui_tracks.lock_or_recover();
        if tracks.is_empty() {
            // 【为什么在这里整体返回，而不是只跳过两条 retain】`present` / `owned` 两个集合
            // 都由**这份**宿主清单派生，所以清单一空，下面两条 `retain` 的条件就都恒不成立
            // （"不在 present 里就删"遇上空集 = 什么都不删）。真正决定 ARA-only 上
            // "哪些片段可见"的是 `workspace_timeline_locked` 的 `retain`（按已分配 region
            // key 过滤），不在这里。所以这里返回是**正确的**，但它是"可见性由别处决定"
            // 这件事的**唯一**落点 —— 改动可见性语义时要先看这里。
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
            let lineage = self.split_media_from.lock_or_recover();
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
        // 这一轮学到的媒体事实（循环结束后一次性写回，避免在遍历中反复取锁）。
        let mut learned: Vec<(String, crate::editor::media_identity::PcmFacts)> = Vec::new();
        let mut present = std::collections::BTreeSet::new();
        let mut owned = std::collections::BTreeSet::new();
        // 由宿主 folder 得到父级的轨道；它们才是私有分组必须让路的那些（见
        // `host_folder_children`）。
        let mut folder_children = std::collections::BTreeSet::new();
        // 【T12：宿主轨的 id 只是**线索**，不是权威】
        //
        // `ui_tracks[guid].id` 是上一轮"重贴标签"的结果，而重贴标签依赖
        // `clip_ids[key]` 与已发布时间线里的 clip id **同形**。只要有一轮对不上
        // （`sync_session_derived_tables` 曾把它打回 `ara-region-` 形态），这里就会拿到
        // 一个**过期**的轨道 id：它在 `timeline.tracks` 里找不到 ⇒ 旧实现新建一条轨，
        // 并把这条宿主轨上的**所有 clip 无条件搬过去** ⇒ 用户看到"原轨空了、新轨全有"。
        //
        // 正确的顺序是**先解析、后使用**：这条宿主轨上的 item 若已被某条活 region 认领，
        // 那么真正的轨道身份来自那条 clip（模型侧），而不是宿主轨上残留的字符串。
        // 【为什么写成嵌套 `fn` 而不是闭包】闭包会不可变借用 `timeline`，而下面同一个
        // 循环要可变借用它（推轨、推 clip）。嵌套函数不捕获环境，借用冲突自然消失。
        fn resolve_model_track(
            timeline: &hifishifter_kernel::state::TimelineState,
            items: &std::collections::HashMap<u64, String>,
            ids: &std::collections::HashMap<u64, String>,
            host: &crate::host::UiTrack,
        ) -> Option<String> {
            for item in &host.items {
                let Some((key, _)) = items
                    .iter()
                    .find(|(_, guid)| **guid == item.geometry.item_id)
                else {
                    continue;
                };
                let Some(clip_id) = ids.get(key) else {
                    continue;
                };
                if let Some(clip) = timeline.clips.iter().find(|clip| &clip.id == clip_id) {
                    return Some(clip.track_id.clone());
                }
            }
            None
        }
        for host in tracks.values() {
            // 解析成功 ⇒ 用模型轨；解析不到 ⇒ 退回宿主 id（未认领的 item 走占位路径）。
            let resolved = {
                let items = self.region_items.lock_or_recover();
                let ids = self.clip_ids.lock_or_recover();
                resolve_model_track(&timeline, &items, &ids, host)
            };
            let track_id = resolved.clone().unwrap_or_else(|| prefix(&host.id));
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
                // 【为什么不用 unwrap】`host.name` 是宿主给的字符串，理论上总能量进
                // 一个字符串字段；但这里跑在**宿主 UI 线程**的清单刷新路径上，一次
                // panic 会带走整个 DAW 会话。JSON 反序列化失败时降级为"跳过这条轨道"，
                // 其余轨道照常呈现。
                match serde_json::from_value::<hifishifter_kernel::state::Track>(
                    serde_json::json!({"id":track_id,"name":host.name,"order":host.order,"compose_enabled":true}),
                ) {
                    Ok(mut track) => {
                        track.parent_id = parent_id;
                        timeline.tracks.push(track);
                    }
                    Err(error) => {
                        log::warn!("[ara] skipping host track {track_id}: {error}");
                        continue;
                    }
                }
            }
            for item in &host.items {
                let g = &item.geometry;
                // 【T3：同一个宿主 item 在一轮载荷里最多对应**一条** clip】
                //
                // clip 的身份会从 `ara-region-{key}` 升级成 `ara-item-{guid}`
                // （见 `remap_and_log`），而升级需要清单先枚举到它 —— 于是升级未完成
                // 的那一轮里，按 GUID 找不到已有 clip，旧实现就会**再造一条无源占位**：
                // 同一个 item 同时有"真 clip"与"占位 clip"。先按 region 边解析，
                // 解析得到就复用那条 clip 的身份，不再凭空造一条。
                let claimed_id = {
                    let items = self.region_items.lock_or_recover();
                    let ids = self.clip_ids.lock_or_recover();
                    items
                        .iter()
                        .find(|(_, guid)| **guid == g.item_id)
                        .and_then(|(key, _)| ids.get(key).cloned())
                };
                let id = claimed_id.unwrap_or_else(|| prefix(&format!("ara-item-{}", g.item_id)));
                present.insert(id.clone());
                if !timeline.clips.iter().any(|clip| clip.id == id) {
                    // 同上的 panic 纪律：宿主给一个非有限 `playback_rate`（离线/损坏素材）
                    // 时不能让整个清单刷新 panic。反序列化失败就跳过这个 item。
                    let built = serde_json::from_value::<hifishifter_kernel::state::Clip>(
                        serde_json::json!({"id":id,"track_id":track_id,"name":item.name,
                        "start_sec":g.start_sec,"length_sec":g.duration_sec,"takes":[{"id":prefix(&g.take_id),"source_start_sec":g.source_start_sec,
                        "source_end_sec":g.source_start_sec+g.duration_sec*g.playback_rate,"playback_rate":g.playback_rate}]}),
                    );
                    match built {
                        Ok(mut clip) => {
                            clip.normalize_takes();
                            timeline.clips.push(clip);
                        }
                        Err(error) => {
                            log::warn!("[ara] skipping host item {id}: {error}");
                            present.remove(&id);
                            continue;
                        }
                    }
                }
                let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == id) else {
                    // 上面的 `continue` 已经移除了 `present`；这里只是防御性兜底。
                    present.remove(&id);
                    continue;
                };
                sync_host_takes(
                    clip,
                    item,
                    &prefix,
                    authorized_takes.get(&g.item_id).map(String::as_str),
                    authorized_sources.get(&g.item_id).map(String::as_str),
                    inherited_media.get(&g.item_id),
                    media_by_item.get(&g.item_id),
                );
                // 【只增不减】把这一轮学到的身份记进事实表（没有 PCM 时 has_pcm=false）。
                if let Some(id) = clip.source_path.clone() {
                    learned.push((
                        g.item_id.clone(),
                        crate::editor::media_identity::PcmFacts {
                            source_id: id,
                            sample_rate: 0,
                            channels: 0,
                            frames: 0,
                            has_pcm: pcm_by_item.contains_key(&g.item_id),
                        },
                    ));
                }
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
                    Some(format!("host-group-{}", g.group_id))
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
                let (in_dir, out_dir) = match g.fade_axes_v2 {
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
        if !learned.is_empty() {
            let mut facts = self.media_facts.lock_or_recover();
            for (item, fact) in learned {
                // 【只增不减】永不用"这一轮没有 PCM"覆盖已有的真 PCM 事实。
                match facts.get(&item) {
                    Some(known) if known.has_pcm && !fact.has_pcm => {}
                    _ => {
                        facts.insert(item, fact);
                    }
                }
            }
        }
        // 【T8：clip 的去留按**实体**判定，不按 id 的字符串前缀】
        //
        // 旧实现有两条按前缀的 retain：`ara-item-*` 必须在本轮 `present` 里，
        // `ara-clip-*` 不得挂在 `owned` 轨上。而同一个 clip 的身份会从
        // `ara-region-{key}` 升级成 `ara-item-{guid}` —— 升级瞬间就换了一套过滤
        // 规则。于是"升级完成、清单尚未枚举到新 GUID"的那一轮，clip 被第一条删掉；
        // 尚未升级的反而两条都不匹配而存活。这正是用户报的"原轨 Clip 全消失、
        // 新轨 Clip 健康"。
        //
        // 正确的判据只有一条：**它代表的 region 还在不在**。`regions` 是文档级权威
        // （`destroy_playback_region` 按 key 删除），而宿主清单的职责是新增/更新，
        // 不是删除 —— 清单滞后不是"宿主删掉了它"的证据。
        {
            let live = self.regions.lock_or_recover();
            let region_items = self.region_items.lock_or_recover();
            let clip_ids = self.clip_ids.lock_or_recover();
            // 活着的 region 当前登记的 clip id（可能仍是 key 形态，也可能已升级）。
            let mut live_clip_ids: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            // 活着的 region 绑定的宿主 item GUID。
            let mut live_item_ids: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for key in live.keys() {
                if let Some(id) = clip_ids.get(key) {
                    live_clip_ids.insert(id.clone());
                }
                live_clip_ids.insert(format!("{namespace}ara-region-{key}"));
                if let Some(item) = region_items.get(key) {
                    live_item_ids.insert(item.clone());
                }
            }
            let item_prefix = prefix("ara-item-");
            let region_prefix = prefix("ara-region-");
            let clip_prefix = prefix("ara-clip-");
            timeline.clips.retain(|clip| {
                if let Some(guid) = clip.id.strip_prefix(&item_prefix) {
                    // 已升级到宿主 item GUID：本轮清单仍报告它，或它仍绑在一个
                    // 活着的 region 上，就留下。
                    return present.contains(&clip.id) || live_item_ids.contains(guid);
                }
                if clip.id.starts_with(&region_prefix) {
                    // 还没升级：region key 仍活着就留下。
                    return live_clip_ids.contains(&clip.id);
                }
                if clip.id.starts_with(&clip_prefix) {
                    // 没有宿主身份的 ARA clip：沿用旧语义 —— 挂在宿主已知轨道上时
                    // 剔除（它认领不到 item GUID，界面无法给出可操作的读数）。
                    return !owned.contains(&clip.track_id);
                }
                true
            });
        }
        // 分割谱系回收：两段都不在宿主清单里时删掉这条边，免得它随分割次数无限增长
        // （与 `ParameterAtlas::split_parents` 的 retain 同一手法）。
        {
            let mut lineage = self.split_media_from.lock_or_recover();
            lineage.retain(|right, left| {
                present.contains(&prefix(&format!("ara-item-{right}")))
                    && present.contains(&prefix(&format!("ara-item-{left}")))
            });
        }
        // 与渲染路径共用同一份派生（`derive_split_media`）：宿主清单里没有、但作为 ARA
        // clip 存在的右半段也能借到父段媒体。已有源的片段会被跳过，所以与上面
        // `sync_host_takes` 的 `inherited_media` 嫁接不冲突（同值，幂等）。
        {
            let lineage = self.split_media_from.lock_or_recover().clone();
            derive_split_media(&mut *timeline, &lineage, &prefix);
        }
        {
            let known = self.ui_known_tracks.lock_or_recover();
            timeline.tracks.retain(|track| {
                !known.iter().any(|id| prefix(id) == track.id) || owned.contains(&track.id)
            });
        }
        // 【为什么在这里把 `known` 推进到本轮清单，而不是只做并集】`known` 的语义是
        // "宿主清单里的轨道"，用来剔除**宿主已不再报告**的轨道。只做并集的话，一条曾经
        // 进过清单、后来离开 folder 的轨道会被永久记住，于是每一轮都被从时间线上剔掉 ——
        // 哪怕它仍有 ARA clip、本该作为普通 ARA 轨道显示。推进到本轮之后，剔除只发生在
        // "上次呈现有、这次没有"的那一次；此后它按普通 ARA 轨道对待。
        *self.ui_known_tracks.lock_or_recover() =
            tracks.values().map(|host| host.id.clone()).collect();
        // 记下"父级由宿主 folder 决定"的轨道，供私有分组呈现时让路
        // （见 `host_folder_children`）。
        *self.host_folder_children.lock_or_recover() = folder_children;
        // 【T9：让"多出一条轨"成为可归因事实】此前这里完全沉默 —— 删轨在渲染路径有
        // 日志，而清单侧的重复/过渡态没有任何痕迹，于是用户报的"分割后多出一条轨"
        // 只能靠肉眼复现。判据刻意取"多于一条轨且其中至少一条本轮没有任何 clip"：
        // 正常的单轨工程、以及真实的多轨工程都不会命中。
        if timeline.tracks.len() > 1 {
            let orphans: Vec<&str> = timeline
                .tracks
                .iter()
                .filter(|track| {
                    !timeline
                        .clips
                        .iter()
                        .any(|clip| clip.track_id == track.id)
                })
                .map(|track| track.id.as_str())
                .collect();
            if !orphans.is_empty() {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.inventory.orphan-track",
                    format_args!(
                        "[ara] presenting {} track(s), {} of them without any clip: {:?} \
                         (a split is likely publishing two identities for one host sequence)",
                        timeline.tracks.len(),
                        orphans.len(),
                        orphans
                    ),
                );
            }
        }
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
        let identities = self.clip_ids.lock_or_recover();
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
        let _transaction = self.transaction.lock_or_recover();
        if !self.is_alive() {
            return None;
        }
        let identities = self.clip_ids.lock_or_recover();
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
            .lock_or_recover()
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
        let identities = self.clip_ids.lock_or_recover().clone();
        let Some(clips) = payload["clips"].as_array_mut() else {
            return;
        };
        for track in self.ui_tracks.lock_or_recover().values() {
            for item in &track.items {
                if let Some(clip) = clips.iter_mut().find(|clip| {
                    clip["id"] == format!("{namespace}ara-item-{}", item.geometry.item_id)
                }) {
                    decorate_item_gain(clip, &item.geometry);
                    let g = &item.geometry;
                    clip["host_fades"] = serde_json::json!({"curve_mode":match g.fade_axes_v2 {Some(true)=>"continuous",Some(false)=>"legacy",None=>"unknown"},
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
                    .lock_or_recover()
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
                    // 委托端由 HFS 自己的曲线渲染，宿主 S 轴不参与 —— 显式归 0，
                    // 让"未变的 S"在写回去重里是一个稳定值而不是缺失字段。
                    clip["fade_in_s"] = serde_json::json!(0.0);
                    clip["fade_out_s"] = serde_json::json!(0.0);
                    clip["host_fades"] = serde_json::json!({"curve_mode":"hifishifter","in_curvature":style.in_dir,"out_curvature":style.out_dir,"in_s":0.,"out_s":0.});
                    continue;
                }
                // S 也作为**顶层字段**回流：完成门的收据按顶层字段比对，只有
                // `host_fades` 里的嵌套 S 是不够的（那会让门永远判"宿主没跟上"）。
                clip["fade_in_s"] = serde_json::json!(g.fade_in_dir2_new);
                clip["fade_out_s"] = serde_json::json!(g.fade_out_dir2_new);
                clip["host_fades"] = serde_json::json!({"curve_mode":match g.fade_axes_v2 {Some(true)=>"continuous",Some(false)=>"legacy",None=>"unknown"},
                    "in_curvature":g.fade_in_dir_new,"out_curvature":g.fade_out_dir_new,
                    "in_s":g.fade_in_dir2_new,"out_s":g.fade_out_dir2_new});
            }
        }
    }
    /// 无宿主getter的短事务装饰；调用者不得仍持编辑timeline锁。
    pub(crate) fn decorate_host_fades(&self, payload: &mut serde_json::Value, namespace: &str) {
        let _transaction = self.transaction.lock_or_recover();
        if self.is_alive() {
            self.decorate_host_fades_locked(
                payload,
                namespace,
                &self.edits.lock_or_recover().fades,
            );
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
            .lock_or_recover()
            .values()
            .cloned()
            .collect();
        // item GUID → ARA 授权那一刻的 active take GUID。用来把"换了 active take 而
        // ARA 尚未重新认领"（`take_switched`）与"从未被认领"分开 —— 两者的处理方式
        // 完全不同，不能都压成一句"正在等待"。
        let authorized_by_item: std::collections::HashMap<String, String> = {
            let items = self.region_items.lock_or_recover();
            let takes = self.authorized_takes.lock_or_recover();
            items
                .iter()
                .filter_map(|(key, item)| takes.get(key).map(|take| (item.clone(), take.clone())))
                .collect()
        };
        // item GUID → 该 item 所属 region 的 ARA 源身份。用来把"这条的源**已被授权却读
        // 不出来**"（`source_read_failed`）与"还没轮到它"分开：前者是宿主/样本侧的真故障，
        // 用户需要立刻知道，不该和"正在分配 region"共用同一句"等待中"。
        let source_by_item: std::collections::HashMap<String, String> = {
            // 【锁序】按 `document.rs` 声明的层级取锁：`regions` 先于 `region_items`。
            // 此前这里是反的，与 `document.rs::clip_for_host_item` 的正序共存，构成真实的
            // 死锁窗口（两把锁同时在手时方向相反）。
            let regions = self.regions.lock_or_recover();
            let items = self.region_items.lock_or_recover();
            items
                .iter()
                .filter_map(|(key, item)| {
                    regions
                        .get(key)
                        .map(|region| (item.clone(), region.audio_source_persistent_id.clone()))
                })
                .collect()
        };
        let failed_sources: std::collections::HashSet<String> = self
            .source_failures
            .lock_or_recover()
            .keys()
            .cloned()
            .collect();
        // 【T8】被宿主暂时收回访问的源：会自己好，但**不是**"还在分配 region"。
        let revoked_sources: std::collections::HashSet<String> = self
            .source_states
            .lock_or_recover()
            .iter()
            .filter(|(_, state)| **state == crate::render::document::SourceState::Revoked)
            .map(|(id, _)| id.clone())
            .collect();
        // folder 父轨：本实例**永远**拿不到组内子轨的音频（见 `HostAudioState`）。
        let folder_parent = self.renderer_owners().into_iter().any(|owner| {
            owner.host_audio_status().state
                == crate::render::extension::HostAudioState::FolderParentWithoutRegions
        });
        let now = std::time::Instant::now();
        let mut pending_since = self.pending_since.lock_or_recover();
        // 【T-B4：`ever_had_source` 已删除】它存在的唯一理由是"曾经有过源的 item 不走进
        // 超时终态"。既然终态本身被去掉了（见下面的等待分支），这张表就没有读者了 ——
        // 而它的两个写点用了**不同的键空间**（第一趟写 item GUID、第二趟写 clip id），
        // 保留它只会留下一个永远对不上的集合。
        let mut seen = std::collections::HashSet::new();
        // 这一趟载荷里出现过的 id（item GUID 或 clip id）。见 `retain` 处的说明。
        let mut present = std::collections::HashSet::new();
        let Some(clips) = payload["clips"].as_array_mut() else {
            return;
        };
        for track in self.ui_tracks.lock_or_recover().values() {
            for item in &track.items {
                let item_id = item.geometry.item_id.clone();
                let Some(clip) = clips
                    .iter_mut()
                    .find(|clip| clip["id"] == format!("{namespace}ara-item-{}", item_id))
                else {
                    continue;
                };
                let has_source = clip["source_path"]
                    .as_str()
                    .is_some_and(|path| !path.is_empty());
                // `waiting` 表示"这条还在等 ARA 音频"：只有它继续计时。其余状态
                // （拿到源、folder 父轨）都必须清掉起点，否则陈旧的起点会让下一次
                // 掉回"在途"时立刻被判超期。
                //
                // 【为什么没有"倒放"这一态】倒放现在由本插件渲染
                // （`ReversePolicy::Materialize` + 内核装配期翻转），不再是"交给宿主
                // 处理"。它和正放一样走 `ready` —— 方向不再是用户需要处置的状态。
                let (state, reason, waiting) = if has_source {
                    ("ready", None, false)
                } else if source_by_item
                    .get(&item_id)
                    .is_some_and(|source| failed_sources.contains(source))
                {
                    // 源已授权、读取却失败：这不是"在途"，立刻报真因，不启动计时。
                    ("unavailable", Some("source_read_failed"), false)
                } else if source_by_item
                    .get(&item_id)
                    .is_some_and(|source| revoked_sources.contains(source))
                {
                    // 【T8："暂时被收回"必须与"还在分配 region"分开】两者的处置**相反**：
                    // 这一条要做的是"等一个 tick，插件自己会重读"，后者要做的是"去宿主里
                    // 做一次编辑让 observer 重跑"。此前它们共用 `pending` → `awaiting_region`，
                    // 于是"宿主收回了一次样本访问"被说成"宿主还在分配 region" ——
                    // 三轮排查都被引到分配的时序上，而真正的动作发生在**源**这一层。
                    ("unavailable", Some("source_revoked"), false)
                } else if folder_parent && !claimed.contains(&item_id) {
                    ("unavailable", Some("folder_parent"), false)
                } else if authorized_by_item.get(&item_id).is_some_and(|authorized| {
                    // 【为什么这一条**保留**终态】它是**可证明**的事实，不是计时器的产物：
                    // 我们手里同时有"授权那一刻的 take GUID"与"宿主现在报的 take GUID"，
                    // 两者不同就说明用户换过 take 而 ARA 还没重新认领。处置明确（撤销或
                    // 再编辑一次让 observer 重跑），所以值得立刻说出来。
                    item.takes
                        .iter()
                        .find(|take| take.active)
                        .is_none_or(|take| &take.geometry.take_id != authorized)
                }) {
                    ("unavailable", Some("take_switched"), false)
                } else {
                    // 【B4：不再把"我们没观察到"升级成"宿主不会来了"】
                    //
                    // 旧实现在这里等 2 s，然后把这一条判成 `unavailable:unclaimed /
                    // `awaiting_region` —— 而这两个原因码的**真实含义**只是"这一轮没看到"。
                    // 它们不是宿主说的话，是我们自己的计时器到点了。代价是：只要展示侧的身份
                    // 链断一次（见 `media_identity` 的四环），2 s 之后**每一个** clip 都会被
                    // 盖上一个断言了错误原因的终态，而终态在界面上是"等待宿主提供音频"——
                    // 用户看到的正是"无论什么情况，所有 Clip 都在等"。
                    //
                    // 现在：一直报 `pending`（"还在等"），并把 `wait_seconds` 作为读数发出去。
                    // 等待没有上界这件事由**读数**承担（用户与诊断包都能看到等了多久），
                    // 而不是由一句错误的断言承担。
                    let since = *pending_since.entry(item_id.clone()).or_insert(now);
                    clip["wait_seconds"] = serde_json::json!(now.duration_since(since).as_secs());
                    ("pending", None, true)
                };
                clip["host_media"] = serde_json::json!(state);
                if let Some(reason) = reason {
                    clip["host_media_reason"] = serde_json::json!(reason);
                }
                if !waiting {
                    seen.insert(item_id.clone());
                }
                // 【为什么要单独一份 `present`】`seen` 是"清计时"集合，而"这一趟还在载荷里"
                // 是另一件事。TTL 只能回收**从载荷里消失**的条目；把两者混用会让"超期但仍
                // 在载荷里"的条目被 20 s 后删掉、起点重置 ⇒ 状态在 pending / unavailable
                // 之间横跳（正是上面注释要避免的）。
                present.insert(item_id);
            }
        }
        // 【H2：所有宿主共有的媒体事实】上面那趟遍历挂在 `ui_tracks` 上，而它只在
        // **有原生扩展**的宿主上被填充（`refresh_ui_inventory` 在 `native_host()` 为
        // `None` 时直接 return）。于是 Cubase / Nuendo / Studio One 上这一整趟**从不
        // 执行**：每个 clip 都拿不到 `host_media`，前端只能按默认态显示，而
        // `media_in_flight_items` 又因 clip id 不是 `ara-item-` 而**无条件**算作在途
        // ⇒ 左下角永久转圈（"编辑后一直在等"）。
        //
        // 这一趟改用**任何 ARA 宿主都有**的事实给剩下的 clip 定态：
        //   * 有 `source_path` ⇒ `ready`（PCM 已授权；映射层给每条存活 region 都写了源）；
        //   * 源在 `source_failures` 里 ⇒ `unavailable:source_read_failed`（真故障，终态）；
        //   * 否则仍在等宿主授权 ⇒ `pending`，并**按 clip id 计时**（`pending_since`），
        //     超期后转 `unavailable:awaiting_region` 这个终态。
        //
        // 【为什么必须计时】"等宿主授权"没有上界：不设 TTL 会让应用状态永久停在
        // "进行中"，用户对着一个永远转不完的圈，却不知道什么都不会再发生。
        {
            let clips = payload["clips"].as_array_mut();
            if let Some(clips) = clips {
                // 【锁序红线】`regions` 必须先于 `source_failures`（见 `document.rs` 声明的
                // 层级，`media_in_flight_items` 也是这个顺序）。若在持有
                // `source_failures` 时去取 `regions`，就与那条路径构成 ABBA。
                // 所以先把 region→source 映射**取完再释放**，然后才锁 `source_failures`。
                //
                // 【为什么不再锁 `pending_since`】它在本函数**开头**就已经被取走并持有到
                // 结束（见 `let mut pending_since = ...`），标准库的 `Mutex` 不可重入 ——
                // 在这里再锁一次会当场自锁。实现这一趟时正是这样写的第一版，实测让整份
                // 测试套件挂死在 `undo_group_diagnostic::…` 上（同一线程二次加锁）。
                // 直接复用外层那个 guard。
                let region_source: std::collections::HashMap<u64, String> = self
                    .regions
                    .lock_or_recover()
                    .iter()
                    .map(|(key, region)| (*key, region.audio_source_persistent_id.clone()))
                    .collect();
                let failures = self.source_failures.lock_or_recover();
                let now = std::time::Instant::now();
                for clip in clips.iter_mut() {
                    if clip.get("host_media").is_some() {
                        continue;
                    }
                    let Some(id) = clip["id"].as_str().map(str::to_owned) else {
                        continue;
                    };
                    let has_source = clip["source_path"]
                        .as_str()
                        .is_some_and(|path| !path.is_empty());
                    if has_source {
                        clip["host_media"] = serde_json::json!("ready");
                        // 拿到音频 ⇒ settled：清掉计时，否则下次掉回"在途"会立刻超期。
                        pending_since.remove(&id);
                        continue;
                    }
                    // 源身份：clip id 由 region key 派生，用它反查 ARA 源（映射已先取好）。
                    let failed = crate::render::document::parse_region_key_from_clip_id(&id)
                        .and_then(|key| region_source.get(&key).cloned())
                        .is_some_and(|source| failures.contains_key(&source));
                    if failed {
                        clip["host_media"] = serde_json::json!("unavailable");
                        clip["host_media_reason"] = serde_json::json!("source_read_failed");
                        pending_since.remove(&id);
                        continue;
                    }
                    // 【B4：与第一趟同一契约 —— 等待不升级成终态】
                    // 没有原生扩展的宿主（Cubase / Nuendo / Studio One）只走这一趟，
                    // 于是这里曾经是"这些宿主上所有 clip 都会在 2 s 后变成等待宿主音频"
                    // 的唯一来源。
                    let since = *pending_since.entry(id.clone()).or_insert(now);
                    clip["host_media"] = serde_json::json!("pending");
                    clip["wait_seconds"] = serde_json::json!(now.duration_since(since).as_secs());
                    // 仍在等 ⇒ 起点必须留着，所以只进 `present`（"还在载荷里"），
                    // 不进 `seen`（"清计时"）。
                    present.insert(id.clone());
                }
            }
        }
        // 【为什么要加 TTL】`seen` 只收"这一趟已经不在途"的 item，所以**从清单里彻底
        // 消失**的 GUID 永远进不了 `seen` —— 它会在表里留到文档关闭。那在长会话里是
        // 逐 GUID 的永久泄漏（用户删掉一个还没授权的 item 就留下一条）。给一个宽裕的
        // TTL：在途判定本身只用 2 s（`PENDING_MEDIA_TIMEOUT`），20 s 还没再出现就说明
        // 它确实不会再来了。
        let stale_after = PENDING_MEDIA_TIMEOUT * 10;
        pending_since.retain(|id, since| {
            // settled（拿到源 / 终态原因）⇒ 清计时。
            if seen.contains(id) {
                return false;
            }
            // 【为什么还要看 `present`】这一趟仍在载荷里的条目必须保留起点 —— 否则
            // "超期未认领"的条目会在 20 s 后被删掉、下一轮 `or_insert(now)` 重新计时，
            // 状态在 `pending` / `unavailable` 之间每 20 s 横跳一次，永不收敛
            // （用户看到占位在"转圈"和"报错"之间闪）。TTL 的职责只是回收**从载荷里
            // 消失**的条目。
            present.contains(id) || now.duration_since(*since) < stale_after
        });
        // 【T-B0-4】一行逐 clip 媒体读数：只要计数变了就打印一次。
        //
        // 【为什么是计数而不是逐条】用户报障时"所有 clip 都在等"与"两条在等"是两种
        // 完全不同的形状；计数能直接分开它们，而逐条会把日志刷成一份清单。
        {
            let (mut ready, mut pending, mut unavailable, mut unknown) = (0usize, 0usize, 0usize, 0usize);
            let mut total = 0usize;
            if let Some(clips) = payload["clips"].as_array() {
                total = clips.len();
                for clip in clips {
                    match clip["host_media"].as_str() {
                        Some("ready") => ready += 1,
                        Some("pending") => pending += 1,
                        Some("unavailable") => unavailable += 1,
                        _ => unknown += 1,
                    }
                }
            }
            let line = format!(
                "[hfs] media ready={ready} pending={pending} unavailable={unavailable} \
unknown={unknown} clips={total} facts={}",
                self.media_facts.lock_or_recover().len()
            );
            let mut last = self.last_media_diag.lock_or_recover();
            if last.as_deref() != Some(line.as_str()) {
                *last = Some(line.clone());
                log::info!("{line}");
            }
        }
        // 被跳过的 playback region（MIDI / 零长度 item…）：只 log 会让它们静默消失。
        // 语言无关地报出数量与原因码，文案由前端 catalog 本地化。
        let skipped = self.skipped_regions.lock_or_recover();
        if !skipped.is_empty() {
            payload["skipped_regions"] = serde_json::json!({
                "count": skipped.len(),
                "items": skipped
                    .iter()
                    .map(|(index, reason)| serde_json::json!({"index": index, "reason": reason}))
                    .collect::<Vec<_>>(),
            });
        }
    }
    /// 仍在"争取宿主音频"的片段数（`0` = 没有在途项，应用状态不应停在"进行中"）。
    ///
    /// 【为什么不能只数"缺 `source_path` 的片段"】`unavailable` 的三类成因都是**终态**：
    /// folder 父轨（本实例永远拿不到组内音频）、样本读取失败、超期未认领。把它们也算作
    /// "进行中"会让应用状态永久停在 pending —— 用户对着一个永远转不完的转圈，却不知道
    /// 什么都不会再发生（这正是 `EditorSession::state` 的 `inventory_pending` 的缺陷）。
    ///
    /// 口径与 [`Self::decorate_host_media_locked`] 一致：只有"已开始等待且未超期"的
    /// item 才计入；没有宿主身份的 ARA 片段（还在等认领）同样计入。
    pub(crate) fn media_in_flight_items(&self, timeline: &TimelineState) -> usize {
        let missing: Vec<&str> = timeline
            .clips
            .iter()
            .filter(|clip| !clip.muted && clip.source_path.is_none())
            .map(|clip| clip.id.as_str())
            .collect();
        if missing.is_empty() {
            return 0;
        }
        // folder 父轨：整组永远拿不到音频 ⇒ 不是"进行中"。
        if self.renderer_owners().into_iter().any(|owner| {
            owner.host_audio_status().state
                == crate::render::extension::HostAudioState::FolderParentWithoutRegions
        }) {
            return 0;
        }
        // 已授权却读取失败的 item：终态，不计入。
        let failed_items: std::collections::HashSet<String> = {
            // 【锁序】`regions` 先于 `region_items`（见 `document.rs` 的声明）。
            let regions = self.regions.lock_or_recover();
            let items = self.region_items.lock_or_recover();
            let failures = self.source_failures.lock_or_recover();
            items
                .iter()
                .filter_map(|(key, item)| {
                    let source = regions.get(key)?.audio_source_persistent_id.clone();
                    failures.contains_key(&source).then(|| item.clone())
                })
                .collect()
        };
        let since = self.pending_since.lock_or_recover();
        let now = std::time::Instant::now();
        missing
            .iter()
            .filter(|id| {
                let Some(guid) = id.strip_prefix("ara-item-") else {
                    // 还没有宿主身份的 ARA 片段：仍在等认领。
                    return true;
                };
                if failed_items.contains(guid) {
                    return false;
                }
                since
                    .get(guid)
                    .is_none_or(|at| now.duration_since(*at) < PENDING_MEDIA_TIMEOUT)
            })
            .count()
    }
    /// 把宿主音频读数写进 GUI 载荷（语言无关分类，文案由前端 catalog 本地化）。
    ///
    /// 【为什么取编辑器实例的读数】GUI 就挂在这个实例上，它才是用户看到的那个窗口；
    /// playback 实例没有 GUI。找不到编辑器实例时**不写**该字段 —— 前端沿用上一次已知值，
    /// 而不是凭空报一个"正常"。
    ///
    /// 【为什么判据是"非 playback 实例"而不是 `is_editor_only()`】后者还要求"文档上另有
    /// playback 实例"（它决定要不要透传）。而 GUI 挂在哪个实例上只取决于角色：Cubase 的
    /// 单实例形态（角色 `[editor, view]`、文档上没有任何 playback 实例）里那个实例同时是
    /// "唯一的窗口"和"接管渲染的那个"。用 `is_editor_only()` 会在这里直接找不到，
    /// 读数整段消失 —— 而那正是最需要读数的地方。
    ///
    /// 【为什么分类留在后端】与 `ara_host_fields:` 同一原则：后端只给语言无关的分类
    /// （见 `render::extension::HostAudioState`），文案按 catalog 本地化。
    pub(crate) fn decorate_host_audio(&self, payload: &mut serde_json::Value) {
        let owners = self.renderer_owners();
        // 【为什么必须比对文档身份】重开工程时常见"新旧文档各有一个 renderer"
        // （旧文档还没关）。此时 `owners` 非空，随便挑一个非 playback 实例会拿到
        // **另一份文档**的读数（通常是 `Ready`），而本会话的界面其实什么都没拿到 ——
        // 用户看到的是"没有报错、但没有片段，删掉插件再加才好"。
        let owner = owners
            .iter()
            .find(|owner| {
                owner.is_non_playback()
                    && owner.editor_document().is_ok_and(|bound| bound.id == self.id)
            })
            .cloned();
        let Some(owner) = owner else {
            // renderer 存在但都不属于本会话的文档：报一个**终态、可行动**的读数，
            // 而不是沿用另一份文档的 `Ready`。
            //
            // 【T10：不再要求 `is_non_playback()`】这里要表达的事实是"有 renderer，
            // 但都不服务于这份文档"，与它承担哪个角色无关。旧判据只看非 playback
            // 实例，于是"所有 owner 都是 playback 且都绑在别的文档上"会落到下面那条
            // 完全沉默的 `return` —— 屏幕上什么都不更新，用户看到的是"没有报错、
            // 但片段一直等着"。把角色从判据里去掉，这两种形状得到同一个可行动读数。
            let bound_elsewhere = owners
                .iter()
                .any(|owner| owner.editor_document().is_ok_and(|bound| bound.id != self.id));
            if bound_elsewhere {
                let waiting = payload["clips"].as_array().map_or(0, |clips| {
                    clips
                        .iter()
                        .filter(|clip| {
                            clip.get("source_path")
                                .and_then(|value| value.as_str())
                                .is_none_or(str::is_empty)
                        })
                        .count()
                });
                payload["host_audio"] = serde_json::json!({
                    "state": "not_bound",
                    "waiting_clips": waiting,
                    "skipped_clips": self.skipped_regions.lock_or_recover().len(),
                    "reason": "bound-to-other-document",
                });
                payload["ara_binding"] = serde_json::json!({
                    "bound": false,
                    "bound_once": self.ever_had_renderer(),
                    "renderers": owners.len(),
                    "reason": "bound-to-other-document",
                    "documents": owners
                        .iter()
                        .filter_map(|owner| owner.bound_document_id())
                        .collect::<Vec<_>>(),
                });
                return;
            }
            // 有 renderer 但没有非 playback 实例：GUI 挂在别的角色上，沿用旧行为
            // （不写读数），不能误报成"未绑定"。
            if !owners.is_empty() {
                return;
            }
            // 【为什么没有 owner 也要写读数】此前这里直接 return，前端沿用上一次
            // 已知值 —— bind 失败/文档缺失时界面永远停在"等待宿主音频"。现在把
            // "根本没有 renderer" 报成 `not_bound`（终态、可行动），并带上诊断
            // 读数回答"到底绑没绑上"。
            let waiting = payload["clips"].as_array().map_or(0, |clips| {
                clips
                    .iter()
                    .filter(|clip| {
                        clip.get("source_path")
                            .and_then(|value| value.as_str())
                            .is_none_or(str::is_empty)
                    })
                    .count()
            });
            if !self.ever_had_renderer() && waiting == 0 {
                return;
            }
            payload["host_audio"] = serde_json::json!({
                "state": "not_bound",
                "waiting_clips": waiting,
                "skipped_clips": self.skipped_regions.lock_or_recover().len(),
                "reason": "no-renderer",
            });
            payload["ara_binding"] = serde_json::json!({
                "bound": false,
                "bound_once": self.ever_had_renderer(),
                "renderers": owners.len(),
                "reason": "no-renderer",
            });
            return;
        };
        let status = owner.host_audio_status();
        payload["ara_binding"] = serde_json::json!({
            "bound": owner.is_ara_bound(),
            "bound_once": owner.was_bound_before(),
            "document_alive": owner.editor_document().is_ok(),
            "native_host": owner.native_host().is_some(),
            "bound_document_id": owner.bound_document_id(),
            "closed": owner.is_closed(),
            "renderers": owners.len(),
        });
        // 【为什么恢复读数要在这里写】恢复由**哪个实例**完成取决于宿主把载荷发给了谁；
        // 用户要回答"参数到底读回来没有"，所以看文档上全部 renderer，而不只是手里
        // 这一个。pending 不是失败：`pending_reason` 说明在等什么。
        let restore_failed = owners.iter().any(|owner| owner.restore_failed());
        let restore_pending = owners
            .iter()
            .any(|owner| owner.restore_status_name() == "pending");
        let restore_status = if restore_failed {
            "failed"
        } else if restore_pending {
            "pending"
        } else if owners
            .iter()
            .any(|owner| owner.restore_status_name() == "applied")
        {
            "applied"
        } else {
            "never"
        };
        payload["ara_restore"] = serde_json::json!({
            "status": restore_status,
            "applied_generation": owners
                .iter()
                .map(|owner| owner.restore_report().2)
                .max()
                .unwrap_or(0),
            "pending_reason": owners
                .iter()
                .find_map(|owner| owner.restore_pending_reason()),
            "error": owners
                .iter()
                .find_map(|owner| owner.restore_report().1),
        });
        // 【为什么 ARA-only 上不能直接报 `status`】没有原生扩展时
        // `refresh_ui_inventory` 早退并把 `host_audio` 复位成默认的 `Ready`（waiting 0），
        // 于是"一条 region 都没匹配上"也被报成一切正常，而左下角却在转圈
        // （`EditorSession::state` 用的是另一条 ARA 读数）—— 两个读数互相矛盾，
        // 排查时会把方向带偏。
        //
        // 所以这里用 ARA 侧的同一份在途计数（`media_in_flight_items`，与 `state()` 同源）
        // 把读数**上调**，让两个读数一致。注意是"补充"而不是"替换"：
        // `FolderParentWithoutRegions` 这类状态是宿主侧的真实分类，替换掉它会把
        // "这组永远拿不到音频"重新说成"正常"。所以只在宿主报 `Ready` 时上调。
        let mut state = status.state;
        let mut waiting = status.waiting_items;
        if state == crate::render::extension::HostAudioState::Ready {
            let timeline = self.timeline.lock_or_recover();
            let in_flight = timeline
                .as_ref()
                .map_or(0, |timeline| self.media_in_flight_items(timeline));
            if in_flight > 0 {
                state = crate::render::extension::HostAudioState::AwaitingRegions;
                waiting = in_flight;
            }
        }
        // 【为什么再分三档】"宿主还没交付 region"与"region 有了、样本访问还没开"
        // 对用户的可执行动作不同；读取失败则是终态，必须给出真因而不是继续转圈。
        // 而"插件自己拒收了 region"是第三档：它看起来和"还在等"一模一样，但再等也不
        // 会来 —— 报成 `awaiting_regions` 会让用户无限期等一个永不到达的东西。
        let failures = self.source_failures.lock_or_recover().len();
        // 拒绝真因只在"确实进了 RegionRejected 档"时才出现在载荷里；先克隆出来，
        // 避免下面按值绑走后再也取不到。
        let rejection = self.last_region_rejection();
        let rejection_reason = rejection.clone();
        if failures > 0 {
            state = crate::render::extension::HostAudioState::SourceReadFailed;
            waiting = failures;
        } else if rejection.is_some() {
            // 只在"我们确实一条 region 都没拿到"时上报：拒收之后宿主就不再交付了，
            // 这正是 `regions_assigned == 0` 的形状。已经拿到 region 的场合里，
            // 历史拒收（例如一次失败的分割）不该盖住正常工作。
            if state == crate::render::extension::HostAudioState::AwaitingRegions
                || state == crate::render::extension::HostAudioState::AwaitingSamplesAccess
            {
                state = crate::render::extension::HostAudioState::RegionRejected;
                waiting = waiting.max(1);
            }
        } else if state == crate::render::extension::HostAudioState::AwaitingRegions
            && !self.samples_access_granted.load(Ordering::Acquire)
        {
            state = crate::render::extension::HostAudioState::AwaitingSamplesAccess;
        }
        // 【为什么恢复失败要单独报】它与"等 region / 读不到样本"不是同一件事：音频通路
        // 是通的，用户丢的是参数与曲线。恢复由**哪个实例**完成取决于宿主把载荷发给了谁，
        // 所以要看文档上全部 renderer，而不是手里这一个。读取失败与拒收 region 更紧急，
        // 保持它们的优先级。
        // 【T-D6：只有"载荷还挂着"才算未解决】`restore_failed` 是一个**历史**读数：
        // 一次失败之后它就是真的，哪怕下一次已经合入成功。原来的条件只要它真就抬成
        // `RestoreUnresolved`，于是音频完全正常的片段也会被挂上"参数没读回来"的横幅。
        // 现在必须同时满足"手里确实还有未合入的载荷"——那一份才是用户真正缺的东西。
        let restore_failed = self
            .renderer_owners()
            .iter()
            .any(|owner| owner.restore_failed() && owner.has_pending_restore());
        if restore_failed
            && state != crate::render::extension::HostAudioState::SourceReadFailed
            && state != crate::render::extension::HostAudioState::RegionRejected
        {
            state = crate::render::extension::HostAudioState::RestoreUnresolved;
            waiting = waiting.max(1);
        }
        let restore_reason = if restore_failed {
            self.renderer_owners()
                .iter()
                .find_map(|owner| owner.restore_report().1)
        } else {
            None
        };
        // 被跳过的 playback region 数量随宿主读数一起给：只 log 会让 MIDI / 零长度 item
        // 在插件里静默消失（见 `DocumentSession::skipped_regions`）。放在 `host_audio`
        // 里是为了复用前端已有的读数通道，不必新增一条状态链。
        let skipped = self.skipped_regions.lock_or_recover().len();
        let reason = match state {
            crate::render::extension::HostAudioState::RegionRejected => rejection_reason,
            crate::render::extension::HostAudioState::RestoreUnresolved => restore_reason,
            _ => None,
        };
        // 【T-C6】把"是哪一环断了"的计数一起给界面：用户截图就能指认，不必再要日志。
        let debug = serde_json::json!({
            "regions": self.regions.lock_or_recover().len(),
            "items": self.region_items.lock_or_recover().len(),
            "authorized": self.authorized_takes.lock_or_recover().len(),
            "pcm": self.edit_sources.lock_or_recover().len(),
            "facts": self.media_facts.lock_or_recover().len(),
            "renderers": self.renderer_owners().len(),
        });
        payload["host_audio"] = serde_json::json!({
            "state": state.as_str(),
            "waiting_clips": waiting,
            "skipped_clips": skipped,
            "debug": debug,
            // 只在真故障档带上真因：`awaiting_regions` 是正常的中间态，
            // 给它挂一个 null 只会让前端多一条要处理的分支。
            "reason": reason,
        });
    }
    /// 普通手动/自动fade和吸附偏移投影到原GUI，内核继续消费未烘焙fade的ARA时间线。
    /// 只沿已核对的唯一真实region key，不按轨名/位置猜关联。
    pub(crate) fn project_ui_fades_locked(
        &self,
        timeline: &mut TimelineState,
        fades: &std::collections::BTreeMap<String, crate::fade::FadeStyle>,
    ) {
        let identities = self.clip_ids.lock_or_recover().clone();
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
                let (in_dir, out_dir) = match geometry.fade_axes_v2 {
                    Some(true) => (geometry.fade_in_dir_new, geometry.fade_out_dir_new),
                    _ => (geometry.fade_in_dir, geometry.fade_out_dir),
                };
                clip.fade_in_dir = in_dir;
                clip.fade_out_dir = out_dir;
                // S 是**新轴才有的第二根轴**（`D_FADE*DIR2_NEW`）。写进 clip 才能让
                // 载荷、写回去重、以及"未变的 S 不该触发写回"共用同一个权威值；
                // 旧轴宿主没有这根轴，恒为 0。
                let (in_s, out_s) = match geometry.fade_axes_v2 {
                    Some(true) => (geometry.fade_in_dir2_new, geometry.fade_out_dir2_new),
                    _ => (0.0, 0.0),
                };
                clip.fade_in_s = in_s;
                clip.fade_out_s = out_s;
                if self
                    .regions
                    .lock_or_recover()
                    .get(&bound.region_key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
                {
                    // 委托：曲线由 HFS 自己的 `FadeStyle` 渲染，宿主两轴不参与 ——
                    // S 在 `FadeStyle` 里没有位置，显式归 0（不是"漏了"）。
                    let style = fades.get(&geometry.item_id).cloned().unwrap_or_default();
                    clip.fade_in_shape = style.in_shape;
                    clip.fade_out_shape = style.out_shape;
                    clip.fade_in_dir = style.in_dir;
                    clip.fade_out_dir = style.out_dir;
                    clip.fade_in_s = 0.0;
                    clip.fade_out_s = 0.0;
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
        let identities = self.clip_ids.lock_or_recover().clone();
        let regions = self.regions.lock_or_recover().clone();
        // 先把所有 owner 的逐 region 绑定摊平成一张表：委托 fade 的 clip 可能属于
        // 多 region owner（folder 轨上的 FX），按 owner 逐个 find_map 会漏。
        let geometries = self
            .renderer_owners()
            .iter()
            .flat_map(|owner| owner.host_geometries_locked(self))
            .map(|bound| (bound.region_key, bound.geometry))
            .collect::<std::collections::HashMap<_, _>>();
        for clip in &mut timeline.clips {
            // 【为什么身份/区域缺席也只跳过】与下面缺宿主几何那条 `continue` 同一条
            // 纪律。此前这两处 `ok_or(...)?` 让"身份表这一轮还没跟上"（region 已建、
            // 派生表未同步的窗口）把整份 `project_audio_fades_locked` 打失败 ——
            // 而那经 `capture_render_input` 的后果是**整轨静音**。
            // 一条片段的淡化身份查不到，只是这一条不投影委托淡化。
            let Some(key) = identities
                .iter()
                .find(|(_, id)| id.as_str() == clip.id)
                .map(|(key, _)| *key)
            else {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.fade-region-identity",
                    format_args!(
                        "[ara] clip {} has no fade region identity this round; its delegated \
                         fades are not projected (other clips are unaffected)",
                        clip.id
                    ),
                );
                continue;
            };
            let Some(region) = regions.get(&key) else {
                continue;
            };
            if !region.has_content_based_fade_at_head && !region.has_content_based_fade_at_tail {
                continue;
            }
            let Some(geometry) = geometries.get(&key).cloned() else {
                // 【为什么降级而不是整份失败】宿主几何在 ARA-only 宿主上**恒缺**
                // （没有原生扩展就没有 `D_FADE*LEN`）。此前这里 `?` 上抛，经
                // `capture_render_input` 让整条 renderer 永远不发布快照 ——
                // Studio One / Cubase / Nuendo / Logic 上只要有一个内容淡化，
                // **整轨彻底静音**。而"某一段的淡化长度取不到"只是局部信息缺失：
                // 保留 ARA 自己给的淡化信息，其余照常渲染。
                // 与 `ara::mapping` 对坏 region"跳过而不是整份失败"是同一条纪律。
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.delegated-fade-geometry",
                    format_args!(
                        "[ara] delegated content fade for clip {} has no host geometry; \
                         keeping the ARA-declared fades",
                        clip.id
                    ),
                );
                continue;
            };
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
                // 委托端由 HFS 自己的 `FadeStyle` 渲染，宿主两轴不参与 ⇒ S 归 0。
                clip.fade_in_s = 0.0;
            }
            if region.has_content_based_fade_at_tail {
                clip.fade_out_sec = geometry.fade_out_sec;
                clip.auto_fade_out_sec = geometry.auto_fade_out_sec;
                clip.fade_out_shape = style.out_shape;
                clip.fade_out_dir = style.out_dir;
                clip.fade_out_s = 0.0;
            }
        }
        Ok(())
    }
    /// 宿主 take 事实投影：把 ARA 表达不到、但宿主能读到的渲染输入播种到时间线上。
    ///
    /// 【为什么必须每次重建后做】`ara::mapping::ara_document_to_timeline` 只能从 ARA
    /// 图重建 clip，而 ARA **没有**方向位、没有循环源、没有声道模式
    /// （见 `ara::mapping::LOST_FIELDS`）。不播种就等于：界面显示"倒放 · 由 REAPER
    /// 处理"，**渲染却在按正放合成**（`capture_render_input` 读的正是这条时间线）。
    /// 这与 [`Self::project_host_mutes_locked`] / [`Self::project_plugin_musical_context_locked`]
    /// 是同一条纪律：宿主权威的、影响渲染的状态，每次重建后重新投影。
    ///
    /// 【为什么按 item GUID 查而不是按 region】clip id 形如 `ara-item-{guid}`
    /// （`present_host_inventory` 建立的身份），与 `ui_tracks` 的 item 一一对应；
    /// region→item 的边只服务 ARA 授权，这里要的是"这个 item 现在的 active take 是什么
    /// 状态"，按 GUID 查最直接，且对没有 region 的 item（尚未授权）也成立。
    pub(crate) fn project_host_take_facts_locked(&self, timeline: &mut TimelineState) {
        // 媒体总时长：来自 ARA 的 `AudioSource.durationSeconds`（映射层已写进
        // `take.duration_sec`，见 `ara::mapping`），**与"是否已授权 PCM"无关**。
        //
        // 【为什么必须单独取一份，而不是读入参 clip 的字段】倒放的镜像坐标复原需要
        // 媒体总时长 D（`forward = [D − host_start − host_span, D − host_start]`），
        // 而 `clip.duration_sec` / `duration_frames` / `source_sample_rate` 只在
        // `sync_host_takes` 搬媒体身份时才写入 —— 分割瞬间两半都没有自己的 region，
        // 拿不到授权 ⇒ 三个字段全空 ⇒ 复原返回 `None` ⇒ 保留宿主**镜像**坐标，
        // 内核再取一次镜像 = 又变正放/偏移（用户报障：分割含倒放 Take 的 Item 后偏移）。
        // ARA 的源时长在映射阶段就存在，用它就能在认领前正确复原。
        let (media_by_item, media_by_source) = {
            let mapped = self.timeline.lock_or_recover();
            mapped
                .as_ref()
                .map(media_duration_indexes)
                .unwrap_or_default()
        };
        // item GUID → (方向, 循环源, 声道模式, 宿主源窗口)。取 **active take** 的几何：
        // 扁平投影描述的就是 active take，`normalize_takes` 之后 clip 的字段来自它。
        // 源窗口 `(start, span)` 只服务倒放的**镜像坐标复原**（见下）。
        let facts: std::collections::HashMap<String, HostItemFacts> = self
            .ui_tracks
            .lock_or_recover()
            .values()
            .flat_map(|track| &track.items)
            .map(|item| {
                let active = item.takes.iter().find(|take| take.active);
                // 读不到方向 ⇒ 正放（`take_reversed` 已落成 `bool`）。
                let reversed = active.is_some_and(|take| take.reversed);
                // 循环源与声道模式是 item 级 / active take 级；缺 active take 时退回
                // item 的权威几何。
                let geometry = active.map(|take| &take.geometry).unwrap_or(&item.geometry);
                // 源窗口解算所需的全部宿主读数，逐 item 冻结一份。媒体总长不在其中：
                // 它按 clip 查（见下面的 `media_by_item` / `media_by_source`）。
                let host = active.map(|take| HostTakeFacts {
                    start_offset_sec: take.geometry.source_start_sec,
                    item_length_sec: take.geometry.duration_sec,
                    play_rate_signed: take.geometry.play_rate_signed,
                    section: take.geometry.section,
                });
                (
                    item.geometry.item_id.clone(),
                    HostItemFacts {
                        reversed,
                        loop_source: geometry.loop_source,
                        channel_mode: geometry.channel_mode,
                        host,
                    },
                )
            })
            .collect();
        for clip in &mut timeline.clips {
            // 【为什么按后缀取 GUID】编辑器时间线的 id 带 GUI namespace 前缀，
            // `strip_prefix("ara-item-")` 在它上面恒为 `None` ⇒ 整条投影静默空转
            // （见 `host_item_guid` 的说明）。
            let Some(guid) = host_item_guid(&clip.id) else {
                continue;
            };
            let Some(facts) = facts.get(guid) else {
                continue;
            };
            clip.reversed = facts.reversed;
            clip.loop_enabled = facts.loop_source;
            clip.channel_mode = facts.channel_mode;
            // 【倒放的镜像坐标复原】ARA 报的 region 时间是**已镜像坐标**（F-4 实测，
            // 见 `probe/ara/REVERSE-FINDINGS.md`）：倒放 take 的 `D_STARTOFFS` 是
            // `源长 − 正向起点 − 跨度`，而内核的倒放消费窗口
            // （`clip_playback_window_sec`）要的是**正向**窗口的终点 —— 直接用 ARA
            // 的区间会让内核二次镜像 = 又变正放。
            //
            // 【为什么调共享解算而不是就地再写一遍】这段数学必须与独立 App 的
            // `.rpp` 导入**逐位一致**（否则同一个工程两个前端听感不同）。解算收口在
            // `hifishifter_kernel::take_geometry`，本处只负责喂宿主读数。
            // 它由宿主事实直接推导，**幂等**（不读入参 clip 的当前窗口），因此重认领/
            // 重建不会累积镜像。
            if clip.reversed {
                if let Some(host) = &facts.host {
                    // 媒体总时长：先按 **item GUID**（认领后可用），再按 **音频源 id**
                    // （映射阶段就有，与认领/授权无关）。分割瞬间两半都还没有自己的
                    // region→item 边，只有后者能命中 —— 这是"分割含倒放 Take 的 Item 后
                    // 偏移"的根因。
                    let by_source = clip
                        .active_take()
                        .source_path
                        .as_deref()
                        .and_then(|source| media_by_source.get(source).copied());
                    let media = reversed_media_duration(
                        media_by_item.get(guid).copied().or(by_source),
                        clip,
                    );
                    let geometry = host.to_geometry(media, facts.reversed, facts.loop_source);
                    let (forward_start, forward_end) = geometry.forward_source_window();
                    clip.source_start_sec = forward_start;
                    clip.source_end_sec = forward_end;
                    // 窗口只写 **active take**（或全部，当 active 身份不可解析时）：
                    // 多 take item 里各 take 的源窗口本可不同，把 active 的窗口盖到
                    // 全部 take 上会污染 inactive take 的几何 —— 用户切过去时看到的是
                    // 上一个 take 的窗口。
                    for take in active_takes_of(clip) {
                        take.source_start_sec = forward_start;
                        take.source_end_sec = forward_end;
                    }
                } else {
                    hifishifter_kernel::log_limiter::emit_limited_key(
                        log::Level::Warn,
                        "ara.reversed-window",
                        format_args!(
                            "[ara] reversed item {guid} has no active take geometry; the clip \
                             keeps the host's mirrored coordinates and will render shifted"
                        ),
                    );
                }
            }
            // 同上：让 **active take** 也带上同一份事实。`normalize_takes()` 之后 clip 的
            // 扁平字段来自 active take，只在 clip 上写会被下一次物化覆盖。
            for take in active_takes_of(clip) {
                take.reversed = facts.reversed;
                take.loop_enabled = facts.loop_source;
                take.channel_mode = facts.channel_mode;
            }
        }
    }
    /// 有效item静音按同文档唯一region身份投影；不把它存成HFS自己的可写轨道状态。
    pub(crate) fn project_host_mutes_locked(&self, timeline: &mut TimelineState) {
        let identities = self.clip_ids.lock_or_recover();
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
        let host_tempo_map = self.host_tempo_map.lock_or_recover().clone();
        let key = crate::render::document::MusicalProjectionKey::new(
            crate::settings_store::revision(),
            *self.host_meter.lock_or_recover(),
            host_tempo_map.clone(),
        );
        let mut cache = self.musical_projection.lock_or_recover();
        if cache.as_ref().is_none_or(|(cached, _)| *cached != key) {
            let musical = crate::settings_store::settings().plugin_musical_context;
            let meter = *self.host_meter.lock_or_recover();
            let (project_scale_notes, tempo_map) =
                hifishifter_kernel::state::plugin_musical_projection(
                    &musical,
                    meter.bpm,
                    meter.numerator,
                    meter.denominator,
                );
            // 【为什么要在这里覆盖】内核按**单点**宿主 bpm 给每个音阶点填速度。宿主经
            // ARA 给出多点 tempo map 时，那个单点值只在 playhead 处对 —— 每个点应当取
            // **该点位置处**的宿主速度。这里在插件侧按位置逐点改写，因此不需要改内核的
            // 投影签名（它仍然负责音阶点的生成与排序）。
            let tempo_map = match (tempo_map, host_tempo_map.as_ref()) {
                (Some(mut points), Some(host_map)) => {
                    for point in &mut points {
                        if let Some(bpm) = host_map.bpm_at(point.position_sec) {
                            if bpm.is_finite() && bpm > 0.0 {
                                point.bpm = bpm;
                            }
                        }
                        // 拍号同理：取该点位置处的宿主拍号，而不是全曲一个。
                        if let Some((numerator, denominator)) =
                            host_map.meter_at(point.position_sec)
                        {
                            point.numerator = Some(numerator);
                            point.denominator = Some(denominator);
                        }
                    }
                    Some(points)
                }
                (points, _) => points,
            };
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
        let _transaction = self.transaction.lock_or_recover();
        let mut timeline = self.workspace_timeline_locked()?;
        self.project_ui_fades_locked(&mut timeline, &self.edits.lock_or_recover().fades);
        Ok((self.ui_geometry_revision.load(Ordering::Acquire), timeline))
    }
    /// 非实时短事务读取完整授权scope；零分配只能返回零区域，不能等同全文档。
    // 非实时诊断入口，保留供授权范围核对。
    #[cfg(test)]
    pub(crate) fn workspace_scope(&self) -> Result<WorkspaceScope, String> {
        let _transaction = self.transaction.lock_or_recover();
        self.workspace_scope_locked()
    }
    /// 调用方已持transaction；真实model-ref所有权再次校验，拒绝跨文档或已销毁区域。
    pub(crate) fn workspace_scope_locked(&self) -> Result<WorkspaceScope, String> {
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        let mut regions = BTreeSet::new();
        let mut derived = false;
        for owner in self.renderer_owners() {
            // 【T1：走同一个判据，并允许文档级派生】本函数的结果是**本文档全部 owner
            // 的并集**（工作区要显示整份工程），因此"本实例还没被告知分配"在这里
            // 完全可以用文档级的 `sequence_regions` 补上 —— 补出来的并集正是它要的。
            //
            // 【为什么不再用 `assigned_regions()`】那条路径是**逐实例**语义（渲染输入
            // 只许渲染自己那一份），且此前用整批 `resolve`：一条陈旧 key 就让整份
            // 工作区为空 ⇒ `ensure_loaded` 整份 Err ⇒ 界面回落到只有宿主清单占位的
            // 降级视图（"所有 Clip 都在等宿主音频"）。
            let scope = owner
                .assigned_scope(self, None, true)
                .map_err(|e| e.to_string())?;
            derived |= scope.source == crate::render::extension::ScopeSource::DocumentDerived;
            regions.extend(scope.keys);
        }
        if derived {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Info,
                "ara.scope.document-derived",
                format_args!(
                    "[ara] no renderer assignment this round; the workspace scope was derived \
                     from the document's region sequences ({} region(s))",
                    regions.len()
                ),
            );
        }
        if !regions.is_empty() {
            // 【为什么逐条过滤而不是整批 `resolve`】`resolve` 是**整批**语义：一批里只要
            // 有一条 key 未登记（已销毁、或宿主在 `addPlaybackRegion` 里把**已销毁父段的
            // key** 一并重发），整批就返回 `Err` ⇒ 工作区时间线整份失败 ⇒ 界面一条片段都
            // 不更新。写路径（`extension.rs` 的分配观察器）早已改成逐条 `owner_of`，
            // 读路径必须同口径，否则同一次分割在两处得到不同结论。
            let owners = region_owners().lock_or_recover();
            let mut dropped = 0_usize;
            regions.retain(|key| match owners.owner_of(*key) {
                Some(document) if document == self.id => true,
                Some(_) => {
                    dropped += 1;
                    false
                }
                None => {
                    dropped += 1;
                    false
                }
            });
            drop(owners);
            if dropped > 0 {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.workspace.scope-dropped",
                    format_args!(
                        "[ara] {dropped} assigned region key(s) are not owned by this document \
                         (destroyed, or re-sent by the host after a split); the rest of the \
                         workspace scope is unaffected"
                    ),
                );
            }
        }
        Ok(WorkspaceScope { regions })
    }
    /// 原GUI的多轨快照只扩展查看/编辑范围，不更改任何播放renderer的混音归属。
    pub(crate) fn workspace_timeline(&self) -> Result<TimelineState, String> {
        let _transaction = self.transaction.lock_or_recover();
        self.workspace_timeline_locked()
    }
    pub(crate) fn workspace_timeline_locked(&self) -> Result<TimelineState, String> {
        let scope = self.workspace_scope_locked()?;
        let mut invalid_keys = 0_usize;
        let clips = scope
            .regions
            .iter()
            // 身份缺席时按 region key 派生（见 `clip_id_for`）：工作区投影跑在
            // region 分配之后、`clip_ids` 重算之前的窗口里，那不是致命错误。
            //
            // 【为什么 key==0 也只跳过】`clip_id_for` 只对 key==0 返回 `None`（那是
            // 无效键，`region_owners` 也拒它）。此前这里的 `ok_or(...)?` + `collect::<Result>()`
            // 会让**一个无效键**把整份工作区时间线打失败 ⇒ 界面一条片段都不更新。
            // 逐条过滤：无效键的那条 region 本次不进工作区，其余照常。
            .filter_map(|key| match self.clip_id_for(*key) {
                Some(id) => Some(id),
                None => {
                    invalid_keys += 1;
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        if invalid_keys > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.workspace.invalid-region-key",
                format_args!(
                    "[ara] {invalid_keys} assigned region(s) have no usable key this round; \
                     they are omitted from the workspace timeline"
                ),
            );
        }
        let mut timeline = self
            .timeline
            .lock_or_recover()
            .clone()
            .ok_or("host timeline unavailable")?;
        timeline.clips.retain(|clip| clips.contains(&clip.id));
        // 分割派生：右半段的授权媒体借自父段。**渲染路径**过去没有这一步，于是界面显示
        // 正常、渲染出来的右半段却是无源静音（见 `derive_split_media`）。
        {
            let lineage = self.split_media_from.lock_or_recover().clone();
            derive_split_media(&mut timeline, &lineage, &|id: &str| id.to_owned());
        }
        self.project_host_mutes_locked(&mut timeline);
        self.project_host_take_facts_locked(&mut timeline);
        self.project_plugin_musical_context_locked(&mut timeline);
        let mut tracks = timeline
            .clips
            .iter()
            .map(|clip| clip.track_id.clone())
            .collect::<BTreeSet<_>>();
        for track in self.ui_tracks.lock_or_recover().values() {
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
        // 【为什么留一行日志】"旧轨在第 2 次分割时才消失"曾是用户报障的一部分：
        // 一条 ARA 轨既没有 clip、也不在宿主清单里，它被丢掉本身是对的，但没有任何
        // 痕迹能回答"它是什么时候、为什么消失的"。限流日志让这条时序可归因。
        let dropped: Vec<String> = timeline
            .tracks
            .iter()
            .filter(|track| !tracks.contains(&track.id))
            .map(|track| track.id.clone())
            .collect();
        if !dropped.is_empty() {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Info,
                "ara.workspace.track-dropped",
                format_args!(
                    "[ara] dropping {} track(s) with no clip and no host entry: {}",
                    dropped.len(),
                    dropped.join(",")
                ),
            );
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

/// 从 clip id 取回宿主 item GUID（`ara-item-{guid}` 的 `{guid}` 部分）。
///
/// 【为什么不能用 `strip_prefix("ara-item-")`】同一个插件里有**两种** clip id：
///
/// - **渲染时间线**（`document.timeline`）上是原生 `ara-item-{guid}`；
/// - **编辑器时间线**（`EditorSession::timeline`）上带 GUI namespace 前缀
///   `hfs-ui-{pid}-{n}-ara-item-{guid}`（见 `EditorSession::rewrite_ids`）。
///
/// 事实投影、媒体时长索引都按这个后缀取宿主身份。只认前缀的写法在编辑器路径上
/// **静默空转**（返回 `None` ⇒ 整条 clip 被跳过），于是"界面上倒放窗口是镜像、
/// 声音却是对的"—— 这正是用户报障的"倒放 Offset 逻辑不对"的根因之一。
/// 按**后缀**取键，两种 id 都能命中。
fn host_item_guid(clip_id: &str) -> Option<&str> {
    const ITEM_TAG: &str = "ara-item-";
    clip_id
        .rfind(ITEM_TAG)
        .map(|index| &clip_id[index + ITEM_TAG.len()..])
}

/// clip 的 **active take**；active 身份解析不出来时退回**全部** take。
///
/// 【为什么需要它】多 take item 里各 take 的源窗口/方向本可不同，把 active take 的
/// 事实盖到其它 take 上会污染它们的几何。但 `normalize_takes()` 只从 active take
/// 物化扁平字段 —— 所以**必须**保证 active take 拿到事实；身份不可解析时宁可写全部，
/// 也不能漏掉它。
fn active_takes_of(
    clip: &mut hifishifter_kernel::state::Clip,
) -> impl Iterator<Item = &mut hifishifter_kernel::state::ClipTake> {
    let wanted = clip
        .active_take_id
        .clone()
        .filter(|id| clip.takes.iter().any(|take| &take.id == id));
    let all = wanted.is_none();
    clip.takes
        .iter_mut()
        .filter(move |take| all || wanted.as_ref().is_some_and(|id| &take.id == id))
}

/// 倒放镜像复原要用的**媒体总时长**（秒）。
///
/// 【为什么优先 ARA 源时长而不是 clip 自带字段】`clip.duration_sec` /
/// `duration_frames` / `source_sample_rate` 只在 `sync_host_takes` 搬媒体身份时才写入，
/// 也就是**要求该片段已被 ARA 授权 PCM**。而分割瞬间两半都没有自己的 region ——
/// 于是三个字段全空 ⇒ 复原返回 `None` ⇒ 保留宿主镜像坐标 ⇒ 内核再取一次镜像 =
/// 内容偏移/变正放（用户报障：分割含倒放 Take 的 Item 后偏移）。
/// ARA 的 `AudioSource.durationSeconds` 在映射阶段就存在，与授权无关，因此优先用它。
fn reversed_media_duration(
    ara_source_duration: Option<f64>,
    clip: &hifishifter_kernel::state::Clip,
) -> Option<f64> {
    ara_source_duration
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .or_else(|| hifishifter_kernel::state::clip_source_media_duration_sec(clip))
}

/// 从 ARA 映射产物里取"媒体总时长"的两份索引。
///
/// 返回 `(按 item GUID, 按音频源 persistent id)`。
///
/// 【为什么两份都要】item GUID 是 `present_host_inventory` 建立的身份，**只在 region
/// 被认领进 `region_items` 之后**才出现在 clip id 上；而音频源 id（`take.source_path`，
/// 即 ARA 的 `AudioSource.persistentID`）在映射阶段就写好了，与认领、授权都无关。
/// 宿主分割的瞬间两个新 region 都还没有 item GUID —— 只按 GUID 索引会让倒放的镜像
/// 复原取不到媒体总长，clip 保留宿主镜像坐标、内核再镜像一次，内容整体偏移
/// （用户报障："分割含倒放 Take 的 Item 时 Take 的偏移计算有误"）。
fn media_duration_indexes(
    mapped: &TimelineState,
) -> (
    std::collections::HashMap<String, f64>,
    std::collections::HashMap<String, f64>,
) {
    let mut by_item = std::collections::HashMap::new();
    let mut by_source = std::collections::HashMap::new();
    for clip in &mapped.clips {
        let Some(duration) = clip.active_take().duration_sec else {
            continue;
        };
        if !(duration.is_finite() && duration > 0.0) {
            continue;
        }
        if let Some(guid) = host_item_guid(&clip.id) {
            by_item.insert(guid.to_owned(), duration);
        }
        if let Some(source) = clip.active_take().source_path.as_deref() {
            if !source.is_empty() {
                by_source.entry(source.to_owned()).or_insert(duration);
            }
        }
    }
    (by_item, by_source)
}

/// 一个 item 的宿主事实快照（源窗口解算与方向投影的输入）。
///
/// 【为什么逐 item 冻结而不是每次现读宿主】`project_host_take_facts_locked` 在
/// 渲染与 GUI 两条路径上都跑，逐 clip 再调宿主 API 会把一次投影变成 N 次跨 FFI
/// 查询。清单快照本来就是"一次枚举、多处消费"。
struct HostItemFacts {
    reversed: bool,
    loop_source: bool,
    channel_mode: i32,
    /// active take 的读数；缺 active take 时为 `None`（此时无法解算窗口）。
    host: Option<HostTakeFacts>,
}

/// active take 上"解算源窗口"所需的宿主读数。
#[derive(Clone, Copy)]
struct HostTakeFacts {
    /// `D_STARTOFFS`（倒放时是**镜像**量，见 `take_geometry` 模块文档）。
    start_offset_sec: f64,
    /// `D_LENGTH`（item 长度，秒）。
    item_length_sec: f64,
    /// `D_PLAYRATE`，**保留符号**（负号是倒放的来源之一）。
    play_rate_signed: f64,
    /// `PCM_Source_GetSectionInfo` 给出的 SECTION 边界。
    section: Option<hifishifter_kernel::take_geometry::SourceSection>,
}

impl HostTakeFacts {
    fn to_geometry(
        self,
        media_duration_sec: Option<f64>,
        reversed: bool,
        loop_enabled: bool,
    ) -> hifishifter_kernel::take_geometry::TakeGeometry {
        hifishifter_kernel::take_geometry::TakeGeometry {
            start_offset_sec: self.start_offset_sec,
            item_length_sec: self.item_length_sec,
            play_rate_signed: self.play_rate_signed,
            media_duration_sec,
            section: self.section,
            reversed,
            loop_enabled,
        }
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

    /// **B4：等得再久也仍然是"在等"，不会升级成一句断言了原因的话。**
    ///
    /// 【为什么这条取代了 T14 的两个用例】T14 修的是"曾经有过源的不该走进超时"，
    /// 而超时本身才是病根：它把"我们这一轮没观察到"说成"宿主不会交付了"。那句断言
    /// 在界面上就是"等待宿主提供音频"，于是只要展示侧的身份链断一次，**每一个** clip
    /// 都会被盖上同一句话 —— 用户报的"无论什么情况全部 Clip 都在等"正是它。
    ///
    /// 现在等待没有终态：状态恒为 `pending`，等待时长作为读数（`wait_seconds`）发出。
    /// 走的是**任何宿主都会跑**的那一趟（不需要原生扩展）。
    #[test]
    fn a_clip_that_waits_forever_still_says_it_is_waiting() {
        let document = DocumentSession::new(40_002);
        let mut payload = serde_json::json!({
            "clips": [{"id": "ara-region-12", "track_id": "t",
                       "start_sec": 0.0, "length_sec": 1.0}],
        });
        // 起点早在 60 s 之前：旧实现会在这一轮判成终态。
        document.pending_since.lock_or_recover().insert(
            "ara-region-12".to_owned(),
            std::time::Instant::now() - std::time::Duration::from_secs(60),
        );
        document.decorate_host_media_locked(&mut payload, "");
        let clip = &payload["clips"][0];
        assert_eq!(clip["host_media"], "pending");
        assert!(
            clip.get("host_media_reason").is_none(),
            "等待不得附带原因码：原因码在界面上是一句断言"
        );
        assert_eq!(
            clip["wait_seconds"].as_u64(),
            Some(60),
            "等了多久必须作为读数发出，等待才有上界可言"
        );
    }



    fn fixture() -> (
        ModelHandle,
        Vec<Arc<ExtensionOwner>>,
        Vec<Box<u8>>,
        Vec<*const ara2_bridge::sys::ARAPlugInExtensionInstance>,
    ) {
        let model = ModelHandle::new();
        let document = model.session();
        *document.timeline.lock_or_recover()=Some(serde_json::from_value(serde_json::json!({
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
                .lock_or_recover()
                .register(key, document.id, index)
                .unwrap();
            document.clip_ids.lock_or_recover().insert(key, id.into());
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
        let host = crate::host::MockHost::new();
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
            let mut tracks = document.ui_tracks.lock_or_recover();
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
        document.ui_tracks.lock_or_recover().remove(&leaving.guid);
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
                .lock_or_recover()
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

    /// **T8：已升级到 `ara-item-{guid}` 的 clip，在清单尚未追上的那一轮不得被删。**
    ///
    /// 【为什么这是"分割后原轨 Clip 全消失"的回归】旧实现要求 `ara-item-*` 必须出现在
    /// 本轮的 `present` 里。分割后右半段的 GUID 是新 mint 的，而 `ui_tracks` 的枚举是
    /// 节流的 —— 升级完成、清单尚未追上的那一轮，clip 被删掉；尚未升级的
    /// （`ara-region-*` / `ara-clip-*`）反而两条 retain 都不匹配而存活。
    /// 判据必须是"它代表的 region 还在不在"，而不是 id 长什么样。
    #[test]
    fn an_upgraded_clip_survives_a_round_where_the_inventory_has_not_caught_up() {
        let model = ModelHandle::new();
        let document = model.session();
        // `present_host_inventory` 在清单为空时整体早退；这里给一条**没有任何 item**
        // 的宿主轨道，让函数真正跑到 clip 的 retain。
        let host = crate::host::MockHost::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let api = Arc::new(host.client());
        let mut track = api.ui_track(&|| true).unwrap();
        track.id = "host-only-track".into();
        track.guid = "{88888888-8888-8888-8888-888888888888}".into();
        track.items.clear();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        // 一条活着的 region：已绑定宿主 item，但本轮 `ui_tracks` 还没有枚举到它。
        document
            .regions
            .lock_or_recover()
            .insert(7, crate::ara::AraPlaybackRegion::default());
        document
            .region_items
            .lock_or_recover()
            .insert(7, "{NEW-GUID}".into());
        document
            .clip_ids
            .lock_or_recover()
            .insert(7, "ara-item-{NEW-GUID}".into());

        let mut timeline = TimelineState::default();
        timeline.tracks = vec![serde_json::from_value(
            serde_json::json!({"id":"ara-sequence-1","name":"t","order":0}),
        )
        .unwrap()];
        timeline.clips = vec![serde_json::from_value(serde_json::json!({
            "id": "ara-item-{NEW-GUID}",
            "track_id": "ara-sequence-1",
            "name": "split half",
            "start_sec": 0.0,
            "length_sec": 1.0,
        }))
        .unwrap()];

        document.present_host_inventory(&mut timeline, "");
        assert!(
            timeline
                .clips
                .iter()
                .any(|clip| clip.id == "ara-item-{NEW-GUID}"),
            "清单滞后不是删除证据：region 还活着，clip 就必须留下"
        );

        // 对照：region 真的被销毁之后，同一个 clip 必须被剔除。
        document.regions.lock_or_recover().remove(&7);
        let mut timeline = TimelineState::default();
        timeline.tracks = vec![serde_json::from_value(
            serde_json::json!({"id":"ara-sequence-1","name":"t","order":0}),
        )
        .unwrap()];
        timeline.clips = vec![serde_json::from_value(serde_json::json!({
            "id": "ara-item-{NEW-GUID}",
            "track_id": "ara-sequence-1",
            "name": "split half",
            "start_sec": 0.0,
            "length_sec": 1.0,
        }))
        .unwrap()];
        document.present_host_inventory(&mut timeline, "");
        assert!(
            timeline.clips.is_empty(),
            "region 已销毁且清单不再报告它时，clip 必须被剔除"
        );
        document.close();
    }

    /// 构造"宿主报镜像窗口 `[host_start, +host_span]`、速率 1.0"的**共享几何输入**。
    ///
    /// 【为什么在插件侧只造输入】插件不再有自带的镜像公式 —— 它已收口到
    /// `hifishifter_kernel::take_geometry`（公式本身由内核的用例覆盖）。这里测的是
    /// "喂进去的宿主读数对不对"，也就是投影层唯一的职责。
    fn mirrored_geometry(media: Option<f64>, host_start: f64, host_span: f64) -> (f64, f64) {
        hifishifter_kernel::take_geometry::TakeGeometry {
            start_offset_sec: host_start,
            item_length_sec: host_span,
            play_rate_signed: 1.0,
            media_duration_sec: media,
            section: None,
            reversed: true,
            loop_enabled: false,
        }
        .forward_source_window()
    }

    /// F-4 实测：源长 2.0、非对称裁切 `[0.25, 1.25]`、倒放后宿主报 `D_STARTOFFS=0.75`
    /// （镜像坐标）。复原必须得到正向窗口 `[0.25, 1.25]`，且**幂等**（同输入同输出），
    /// 否则重建会累积镜像 = 又变正放。
    #[test]
    fn a_reversed_host_window_is_mirrored_back_to_the_forward_window() {
        // 宿主 [0.75, 0.75+1.0]，源长 2.0 ⇒ 正向 [2.0−1.75, 2.0−0.75] = [0.25, 1.25]。
        assert_eq!(mirrored_geometry(Some(2.0), 0.75, 1.0), (0.25, 1.25));
        // 幂等：只读宿主事实，不读 clip 当前窗口，重复调用结果相同。
        assert_eq!(mirrored_geometry(Some(2.0), 0.75, 1.0), (0.25, 1.25));
    }

    /// **越界必须夹紧**：长 item / 短媒体时镜像端点会落到媒体之外，不夹紧就写出一个
    /// 越界窗口（负起点 / 超出媒体末端）。独立 App 的同类钳制同样不是静默的。
    #[test]
    fn an_out_of_range_reversed_host_window_is_clamped_into_the_media() {
        // 宿主窗口整体越过媒体末端：正向 [2.0−2.5, 2.0−1.5] = [−0.5, 0.5] ⇒ 夹到 [0, 0.5]。
        assert_eq!(mirrored_geometry(Some(2.0), 1.5, 1.0), (0.0, 0.5));
        // 窗口完全在媒体之外 ⇒ 两端夹到同一端点，退化为零长度而不是负长度。
        assert_eq!(mirrored_geometry(Some(2.0), 5.0, 1.0), (0.0, 0.0));
    }

    /// 媒体总长未知时退回**原始读数**作锚点（与独立 App 的 `else { take.s_offs }` 同口径），
    /// 而不是保留镜像坐标。两侧必须给同一个答案 —— 这正是本次收口的目的。
    #[test]
    fn an_unknown_media_duration_falls_back_to_the_raw_offset() {
        assert_eq!(mirrored_geometry(None, 0.75, 1.0), (0.0, 0.75));
    }

    /// **不变量：倒放镜像复原所需的媒体时长来自 ARA 源时长，不依赖"是否已授权 PCM"。**
    ///
    /// 【为什么这是回归测试】`clip.duration_sec` / `duration_frames` /
    /// `source_sample_rate` 只在 `sync_host_takes` 搬媒体身份时才写入 —— 也就是要求该
    /// 片段已被 ARA 授权。分割含倒放 Take 的 Item 时，两半都还没有自己的 region，
    /// 三个字段全空 ⇒ 复原拿不到 D ⇒ 保留宿主**镜像**坐标 ⇒ 内核再取一次镜像 =
    /// 内容偏移/变正放（用户报障："分割含倒放 Take 的 Item 时 Take 的偏移计算有误"，
    /// 独立 App 正常）。ARA 的源时长在映射阶段就存在，与授权无关。
    #[test]
    fn a_reversed_clip_without_authorized_media_still_restores_its_forward_window() {
        use hifishifter_kernel::state::Clip;
        // 未授权的倒放片段：没有任何"媒体身份"字段。
        let mut unclaimed: Clip = serde_json::from_value(serde_json::json!({
            "id":"ara-item-{11111111-1111-1111-1111-111111111111}","track_id":"track",
            "name":"half","start_sec":0,"length_sec":1,
            "takes":[{"id":"take","source_start_sec":0.75,"source_end_sec":1.75}]
        }))
        .unwrap();
        unclaimed.duration_sec = None;
        unclaimed.duration_frames = None;
        unclaimed.source_sample_rate = None;
        for take in &mut unclaimed.takes {
            take.duration_sec = None;
            take.duration_frames = None;
            take.source_sample_rate = None;
        }
        assert!(
            hifishifter_kernel::state::clip_source_media_duration_sec(&unclaimed).is_none(),
            "前提：未授权片段自身拿不到媒体时长（这正是缺陷的起点）"
        );

        // ARA 源时长 2.0 可用 ⇒ 复原照常成立。
        let media = reversed_media_duration(Some(2.0), &unclaimed);
        assert_eq!(media, Some(2.0));
        assert_eq!(
            mirrored_geometry(media, 0.75, 1.0),
            (0.25, 1.25),
            "宿主镜像窗口 [0.75, 1.75] + 源长 2.0 ⇒ 正向 [0.25, 1.25]"
        );

        // 连 ARA 源时长也没有时，`reversed_media_duration` 明确返回 None
        // （调用方据此留痕；共享解算会退回原始读数作锚点，而不是编造一个媒体长）。
        assert_eq!(reversed_media_duration(None, &unclaimed), None);
    }

    /// **回归：未认领的分割半段也要能拿到媒体总长。**
    ///
    /// 【为什么这条必须单独存在】上一条测试把时长**直接传进** `reversed_forward_window`，
    /// 绕过了索引 —— 于是"用 ARA 源时长"这个修复看似成立，实际在分割瞬间仍然落空：
    /// 两个新 region 还没有 item GUID，`media_by_item` 里根本没有它们（`a252a119` 的
    /// 缺陷）。这条钉住 `media_duration_indexes` 的**音频源 id 通路**：clip 仍是
    /// `ara-clip-N`（未认领，无 GUID 键），只有 `source_path` 可用。
    #[test]
    fn unclaimed_split_halves_resolve_media_duration_by_audio_source_id() {
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],
            "clips":[
                // 未认领：id 仍是 `ara-clip-N`，没有 item GUID 键。
                {"id":"ara-clip-7","track_id":"a","name":"half","start_sec":0,"length_sec":1,
                 "takes":[{"id":"take","source_path":"src-A","source_start_sec":0.75,
                           "source_end_sec":1.75,"duration_sec":2.0}]},
                // 已认领的对照项：GUID 键与源键都该有。
                {"id":"ara-item-{G}","track_id":"a","name":"claimed","start_sec":1,"length_sec":1,
                 "takes":[{"id":"take","source_path":"src-B","source_start_sec":0,
                           "source_end_sec":1,"duration_sec":3.0}]}
            ],
            "bpm":120,"project_sec":2
        }))
        .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        let (by_item, by_source) = media_duration_indexes(&timeline);
        assert!(
            by_source.get("src-A").copied() == Some(2.0),
            "未认领的半段必须能按音频源 id 取到媒体总长（这是分割偏移的根因）"
        );
        assert_eq!(by_item.get("{G}").copied(), Some(3.0));
        assert_eq!(by_source.get("src-B").copied(), Some(3.0));
    }

    /// 分割派生：只有**自己没源**的右半段才借父段媒体；已有源的片段不被覆盖。
    ///
    /// 【为什么这是回归测试】此前只有清单路径做了这件事（`sync_host_takes` 的
    /// `inherited_media`），渲染路径没有 ⇒ 界面显示正常、渲染出来的右半段却是无源静音。
    #[test]
    fn derive_split_media_lends_the_parent_media_only_to_a_sourceless_right_half() {
        let model = ModelHandle::new();
        let document = model.session();
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],
            "clips":[
                {"id":"ara-item-{P}","track_id":"a","name":"L","start_sec":0,"length_sec":0.25,
                 "takes":[{"id":"tl","source_path":"shared-source","source_end_sec":0.25}]},
                {"id":"ara-item-{R}","track_id":"a","name":"R","start_sec":0.25,"length_sec":0.25,
                 "takes":[{"id":"tr","source_end_sec":0.25}]},
                {"id":"ara-item-{O}","track_id":"a","name":"O","start_sec":0.5,"length_sec":0.25,
                 "takes":[{"id":"to","source_path":"own-source","source_end_sec":0.25}]}
            ],
            "bpm":120,"project_sec":1
        }))
        .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        let lineage = std::collections::HashMap::from([
            ("{R}".to_string(), "{P}".to_string()),
            ("{O}".to_string(), "{P}".to_string()),
        ]);
        derive_split_media(&mut timeline, &lineage, &|id: &str| id.to_owned());

        let right = timeline
            .clips
            .iter()
            .find(|clip| clip.id == "ara-item-{R}")
            .unwrap();
        assert_eq!(
            right.source_path.as_deref(),
            Some("shared-source"),
            "无源右半段必须借到父段媒体（渲染路径过去没有这一步）"
        );
        assert!(
            right
                .takes
                .iter()
                .all(|take| take.source_path.as_deref() == Some("shared-source")),
            "active take 也要带上，否则下一次 normalize_takes 会把扁平字段清空"
        );
        // 自己已经有源的片段不被覆盖。
        let own = timeline
            .clips
            .iter()
            .find(|clip| clip.id == "ara-item-{O}")
            .unwrap();
        assert_eq!(own.source_path.as_deref(), Some("own-source"));
        document.close();
    }

    /// "在途"必须排除**终态**，否则应用状态会永久停在"进行中"。
    ///
    /// 【为什么这条是回归测试】此前 `EditorSession::state` 的 `inventory_pending` 直接数
    /// "缺 `source_path` 的片段"。folder 父轨 / 样本读取失败 / 超期未认领都是终态，永远
    /// 不会再变成 ready ⇒ 状态片永远转圈。现在只有"已开始等待且未超期"的才计入。
    #[test]
    fn media_in_flight_items_excludes_terminal_states() {
        use std::time::{Duration, Instant};
        let model = ModelHandle::new();
        let document = model.session();
        let clip = |id: &str, source: Option<&str>| {
            let mut clip =
                serde_json::from_value::<hifishifter_kernel::state::Clip>(serde_json::json!({
                    "id": id, "track_id":"t", "name":"c", "start_sec":0., "length_sec":0.5,
                    "takes":[{"id":"tk","source_path":source,"source_start_sec":0.,"source_end_sec":0.5}]
                }))
                .unwrap();
            // 扁平字段来自 active take；不物化就测不到"有源"这一档。
            clip.normalize_takes();
            clip
        };
        let timeline = TimelineState {
            clips: vec![clip("ara-item-{G}", None)],
            ..Default::default()
        };
        // 无 `pending_since` 记录 ⇒ 视为刚进入在途。
        assert_eq!(document.media_in_flight_items(&timeline), 1);

        // 超期 ⇒ 终态（`unavailable`），不再计入。
        document.pending_since.lock_or_recover().insert(
            "{G}".into(),
            Instant::now() - PENDING_MEDIA_TIMEOUT - Duration::from_secs(1),
        );
        assert_eq!(document.media_in_flight_items(&timeline), 0);

        // 源读取失败 ⇒ 终态，即便还没超期也不计入。
        document.pending_since.lock_or_recover().clear();
        document
            .region_items
            .lock_or_recover()
            .insert(7, "{G}".into());
        document.regions.lock_or_recover().insert(
            7,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "ara://source".into(),
                ..Default::default()
            },
        );
        document
            .source_failures
            .lock_or_recover()
            .insert("ara://source".into(), "boom".into());
        assert_eq!(document.media_in_flight_items(&timeline), 0);

        // 有源 ⇒ 不计入；静音片段也不计入。
        document.source_failures.lock_or_recover().clear();
        let ready = TimelineState {
            clips: vec![clip("ara-item-{G}", Some("ara://source"))],
            ..Default::default()
        };
        assert_eq!(document.media_in_flight_items(&ready), 0);
        let mut muted = clip("ara-item-{G}", None);
        muted.muted = true;
        assert_eq!(
            document.media_in_flight_items(&TimelineState {
                clips: vec![muted],
                ..Default::default()
            }),
            0
        );
        document.close();
    }
    /// **回归（本次事故的根因）：倒放的镜像复原必须在编辑器（带 GUI namespace
    /// 前缀）时间线上也生效。**
    ///
    /// 【为什么这是根因】全仓只有 `project_host_take_facts_locked` 会做镜像复原，
    /// 而它按 `strip_prefix("ara-item-")` 取宿主身份 —— 编辑器时间线的 id 带
    /// `hfs-ui-{pid}-{n}-` 前缀（见 `EditorSession::rewrite_ids`），`strip_prefix`
    /// 恒为 `None` ⇒ **整条投影静默空转**。于是界面上的倒放窗口停留在宿主镜像坐标
    /// （0.75），而渲染路径（无前缀）复原成 0.25 —— 用户看到"倒放 Offset 逻辑不对、
    /// 方向与 REAPER 相反"。
    #[test]
    fn reversed_facts_project_onto_a_namespaced_editor_timeline() {
        let model = ModelHandle::new();
        let document = model.session();
        let host = crate::host::MockHost::new();
        host.enable_media();
        host.clear_markers();
        // `enable_takes` 同时给出方向读取所需的非空 `PCM_source` 哨兵。
        host.enable_takes(0);
        host.take_reversed.set(true);
        host.section_reports.set(true);
        // 宿主读数：item 长 1.0、速率 1.0、倒放镜像起点 0.75（媒体总长 2.0）。
        host.set_value("D_POSITION", 0.0);
        host.set_value("D_LENGTH", 1.0);
        host.set_value("D_STARTOFFS", 0.75);
        host.set_value("D_PLAYRATE", 1.0);

        let api = Arc::new(host.client());
        let track = api.ui_track(&|| true).expect("host inventory");
        let guid = track.items[0].geometry.item_id.clone();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);

        // 文档时间线：一条 ARA clip。媒体总长来自 ARA 源（映射阶段就有，与授权无关）。
        let mut mapped: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"t","name":"T","order":0}],
            "clips":[{"id":format!("ara-item-{guid}"),"track_id":"t","name":"c","start_sec":0,
                      "length_sec":1.0,"takes":[{"id":"take","source_path":"src",
                      "source_start_sec":0.75,"source_end_sec":1.75,"duration_sec":2.0}]}],
            "bpm":120,"project_sec":1
        }))
        .unwrap();
        for clip in &mut mapped.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock_or_recover() = Some(mapped);

        // 编辑器时间线：id 带 GUI namespace 前缀 —— 与 `rewrite_ids(to_ui=true)` 同形。
        let namespace = "hfs-ui-1-1-";
        let mut editor: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":format!("{namespace}t"),"name":"T","order":0}],
            "clips":[{"id":format!("{namespace}ara-item-{guid}"),
                      "track_id":format!("{namespace}t"),"name":"c","start_sec":0,
                      "length_sec":1.0,"takes":[{"id":format!("{namespace}take"),
                      "source_path":"src","source_start_sec":0.75,"source_end_sec":1.75,
                      "duration_sec":2.0}]}],
            "bpm":120,"project_sec":1
        }))
        .unwrap();
        for clip in &mut editor.clips {
            clip.normalize_takes();
        }

        // 【为什么把断言包进 `catch_unwind`】`UiItem` 里的 `HostClipTarget` 持有
        // `Arc<NativeHost>`；若它比夹具活得久，析构会经 vtable 回访**已释放**的夹具
        // （测试进程 STATUS_ACCESS_VIOLATION，断言失败被 AV 盖掉，看不出真实原因）。
        // 本文件其它用例插入的轨道 `items.clear()`，恰好绕开了这一点。这里先收尾、
        // 再让断言失败照常冒泡 —— 于是失败时看到的是断言消息，不是访问违例。
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            document.project_host_take_facts_locked(&mut editor);
            let clip = &editor.clips[0];
            assert!(clip.reversed, "方向必须投影到带前缀的时间线上");
            assert!(
                (clip.source_start_sec - 0.25).abs() < 1e-9,
                "镜像必须复原成正向起点，got {}",
                clip.source_start_sec
            );
            assert!(
                (clip.source_end_sec - 1.25).abs() < 1e-9,
                "镜像必须复原成正向终点，got {}",
                clip.source_end_sec
            );

            // 幂等：重复投影不得累积镜像（累积 = 又变正放）。
            document.project_host_take_facts_locked(&mut editor);
            let clip = &editor.clips[0];
            assert!((clip.source_start_sec - 0.25).abs() < 1e-9);
            assert!((clip.source_end_sec - 1.25).abs() < 1e-9);
        }));
        document.close();
        drop(model);
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }
    /// **回归：缺宿主几何的委托淡化必须降级，而不是让整份合成失败。**
    ///
    /// 【为什么这是回归测试】宿主几何在 ARA-only 宿主上**恒缺**（没有原生扩展就没有
    /// `D_FADE*LEN`）。此前 `project_audio_fades_locked` 直接 `?` 上抛，经
    /// `capture_render_input` 让整条 renderer 永远不发布快照 —— Studio One / Cubase /
    /// Nuendo / Logic 上只要有一个内容淡化，**整轨彻底静音**。而"某一段的淡化长度取不到"
    /// 只是局部信息缺失：保留 ARA 自己声明的淡化，其余照常渲染。
    #[test]
    fn a_delegated_fade_without_host_geometry_degrades_instead_of_failing() {
        use crate::ara::AraPlaybackRegion;
        let model = ModelHandle::new();
        let document = model.session();
        // 一个已认领的 clip：`clip_ids` 把 region key 映射到它的 id。
        let key = 0x5150_u64;
        document
            .clip_ids
            .lock_or_recover()
            .insert(key, "clip-a".into());
        let mut region = AraPlaybackRegion::default();
        region.audio_source_persistent_id = "ara://source".into();
        region.has_content_based_fade_at_head = true;
        region.has_content_based_fade_at_tail = true;
        document.regions.lock_or_recover().insert(key, region);

        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"t","name":"T","order":0}],
            "clips":[{"id":"clip-a","track_id":"t","name":"c","start_sec":0,"length_sec":1,
                      "takes":[{"id":"take","source_path":"ara://source"}]}],
            "bpm":120,"project_sec":1
        }))
        .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }

        // 没有任何 renderer owner ⇒ 宿主几何表为空，正是 ARA-only 的形状。
        let edits = crate::state_channel::EditState::default();
        let outcome = document.project_audio_fades_locked(&mut timeline, &edits);
        assert!(
            outcome.is_ok(),
            "缺宿主几何只能是局部降级，不能让整份合成失败：{outcome:?}"
        );
        document.close();
        drop(model);
    }
}
