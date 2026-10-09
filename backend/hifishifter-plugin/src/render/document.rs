//! 按真实 controller 身份索引文档生命周期，不从最后一个创建的文档猜测归属。

use super::extension::ExtensionOwner;
use super::ownership::DocumentId;
use ara2_bridge::companion::CompanionControllerBinding;
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::ExtensionControllerLease;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

struct RendererLease {
    lease: ExtensionControllerLease,
    _companion: Option<CompanionControllerBinding<'static>>,
    owner: Weak<ExtensionOwner>,
}

/// 内嵌GUI同进程冻结值；PCM沿授权Arc共享，不能经旧IPC/JSON复制完整长源。
pub(crate) struct WorkspaceSnapshot {
    pub timeline: hifishifter_kernel::state::TimelineState,
    pub sources: Vec<(String, Arc<super::source::SourcePcm>)>,
    pub revision: u64,
    pub model_revision: u64,
    pub audio_revision: u64,
}

#[derive(Default)]
pub(crate) struct DocumentSession {
    controller: AtomicUsize,
    alive: AtomicBool,
    renderers: Mutex<Vec<RendererLease>>,
    pub generation: Mutex<Option<ApiGeneration>>,
    pub sequence_regions: Mutex<HashMap<u64, HashSet<u64>>>,
    /// sequence 真实键 → 扁平 region sequence 下标（= 映射层轨道下标）。
    ///
    /// 只用于把"被分配了 sequence、当前却没有 clip"的轨道认出来（空 folder 轨、
    /// 只有静音/待授权 item 的子轨）—— 它们仍应是参数根。**不是**按名字/位置猜轨道：
    /// 这条边来自宿主 `createRegionSequence` 的真实键。
    pub sequence_track_index: Mutex<HashMap<u64, usize>>,
    pub regions: Mutex<HashMap<u64, crate::ara::AraPlaybackRegion>>,
    pub clip_ids: Mutex<HashMap<u64, String>>,
    pub region_items: Mutex<HashMap<u64, String>>,
    /// ARA 授权那一刻，每个 region 对应 item 的 **active take GUID**（region_key → take GUID）。
    ///
    /// 【为什么需要】ARA 只授权 active take 的 PCM（见 `probe/ara/MULTI-TAKE-FINDINGS.md`
    /// 与 plan Part 2 的授权边界）。宿主清单重建 take 列表时（`sync_host_takes`），
    /// 只有 GUID 与这里相等的那一个 take 可以带上授权媒体；其余 take 必须保持无源。
    ///
    /// 【为什么记 GUID 而不是"就是 active 那个"】用户在 REAPER 里切换 active take 后，
    /// 这条记录直到 ARA 模型重新认领才更新。这期间**不沿用**旧 PCM —— 否则会把上一个
    /// take 的采样挂到新 take 上，那是听不见的错（渲染读的是同一条 `source_path`）。
    /// 记不下来时宁可不挂：无源占位是看得见的。
    pub authorized_takes: Mutex<HashMap<u64, String>>,
    pub ui_tracks: Mutex<std::collections::BTreeMap<String, crate::host::reaper::UiTrack>>,
    pub ui_known_tracks: Mutex<HashSet<String>>,
    /// 由 REAPER folder 结构**得到父级**的轨道 id（见 `host::folder`）。
    ///
    /// 这些轨道的父子边由宿主决定，插件私有分组**不得**改写 —— 否则同一个工程会
    /// 出现两套父子关系：用户在 REAPER 里改了分组，插件却显示另一套，且无法诊断。
    ///
    /// 【为什么只记"有父级"的轨道，而不是"宿主呈现过的所有轨道"】宿主清单同时承载
    /// 轨道身份（GUID↔id 绑定）与 folder 结构两件事。宿主**没有**给出父级的轨道
    /// （folder 之外的普通轨）并不代表"宿主说它是根级"，只是没有那条边 —— 私有分组
    /// 依然可以把它编进一个参数组。把整份清单都算作权威会静默废掉用户的私有分组。
    pub host_folder_children: Mutex<std::collections::BTreeSet<String>>,
    /// 分割谱系：右半段 item GUID → 左半段（父段）item GUID。
    ///
    /// 【为什么需要】宿主分割**不改变音频源** —— 两半指向同一个文件。父段已被 ARA
    /// 授权，所以右半段可以立刻显示波形，不必等宿主为它重新分配 region。缺了这条，
    /// 分割后右半段会停在一段"等待 REAPER 提供音频"的占位里（用户报障过）。
    ///
    /// 【为什么不复用 `ParameterAtlas::split_parents`】那份谱系服务的是**参数曲线**
    /// 继承，键与生命周期都挂在参数权威上；媒体派生是显示层的事，两件事不该共用一个
    /// 记录 —— 否则一方的清理会静默影响另一方。
    pub split_media_from: Mutex<HashMap<String, String>>,
    /// 每个尚未拿到音频的 item 是**从什么时候**开始等的（item GUID → 首次进入
    /// "在途"的时刻）。
    ///
    /// 【为什么必须记时刻】"等待 REAPER 完成音频分配"此前是**吸收态**：只要一个 item
    /// 没有 `source_path`、又不是 folder 父轨、方向位也读不出来，它就永远显示"正在
    /// 等待"，无论等多久、用户做什么都出不来（典型触发：REAPER 的"倒放 Item 为新
    /// Take"换了 active take，而 ARA 不再重发模型 ⇒ `authorized_takes` 永久陈旧）。
    /// 记下时刻就能把"在途"与"等不到了"分开：超过阈值仍无源 ⇒ 转 `unavailable` 并给出
    /// 可执行的原因，而不是让用户对着一个永远转不完的占位。
    pub pending_since: Mutex<HashMap<String, std::time::Instant>>,
    pub ui_inventory_stamp: Mutex<Option<(i32, u64, u64)>>,
    pub sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    pub edit_sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    /// PCM 可以晚于结构回调到达；不借用 model/scope 版本触发 GUI 资源补齐。
    pub editor_audio_revision: AtomicU64,
    pub timeline: Mutex<Option<hifishifter_kernel::state::TimelineState>>,
    pub track_bindings: Mutex<crate::state_channel::TrackBindings>,
    pub revision: AtomicU64,
    pub render_epoch: AtomicU64,
    pub scope_revision: AtomicU64,
    /// 纯GUI宿主装饰版本，不改变编辑曲线或神经合成代次。
    pub ui_geometry_revision: AtomicU64,
    pub ready: AtomicBool,
    pub transaction: Mutex<()>,
    pub edits: Arc<Mutex<crate::state_channel::EditState>>,
    pub id: DocumentId,
    pub clock: Arc<super::transport::TransportClock>,
    pub playback: Mutex<Option<ara2_bridge::plugin::PlaybackRequestHandle>>,
    pub host_undo: crate::host::undo::HostUndo,
    /// 宿主 BPM / 拍号读数（VST3 进程上下文）。
    ///
    /// 【为什么记在文档上】Tempo Map 的音阶点需要宿主轴：0 位置点**必须**显式带拍号
    /// （前端 `effectiveTimeSignatures` 依赖它），每个点都要 BPM。而音阶投影在
    /// `workspace_timeline_locked` 里跑（`DocumentSession` 上），那里读不到
    /// per-view 的 `ProjectState`。
    pub host_meter: Mutex<HostMeter>,
    /// 插件自有音乐上下文的投影缓存，键是"输入代次"（见
    /// [`DocumentSession::project_plugin_musical_context_locked`]）。
    pub musical_projection: Mutex<Option<(MusicalProjectionKey, PluginMusicalProjection)>>,
    editor: OnceLock<Result<Arc<crate::editor::session::EditorSession>, String>>,
}

/// 宿主音乐读数的快照；默认 120 BPM 4/4（与 `TimelineState::default` 同口径）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HostMeter {
    pub bpm: f64,
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for HostMeter {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            numerator: 4,
            denominator: 4,
        }
    }
}

/// 音阶投影缓存的键：设置代次 + 宿主音乐读数。
///
/// 【为什么宿主读数也在键里】BPM/拍号变化不经过设置写入，只看设置代次会让
/// Tempo Map 的宿主轴停在旧读数上。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MusicalProjectionKey {
    settings: u64,
    bpm_bits: u64,
    numerator: u32,
    denominator: u32,
}

impl MusicalProjectionKey {
    pub(crate) fn new(settings: u64, meter: HostMeter) -> Self {
        Self {
            settings,
            bpm_bits: meter.bpm.to_bits(),
            numerator: meter.numerator,
            denominator: meter.denominator,
        }
    }
}

/// 插件自有音阶在时间线上的投影结果。
#[derive(Clone, Debug)]
pub(crate) struct PluginMusicalProjection {
    pub project_scale_notes: Vec<u8>,
    pub tempo_map: Option<Vec<hifishifter_kernel::state::TempoPointData>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    /// 真正的工厂必须在把控制器交给宿主前登记身份，空文档也必须可绑定。
    #[test]
    fn native_factory_registers_the_actual_controller_identity() {
        let factory = FactoryBuilder::new("org.hfs.identity", "org.hfs.identity.archive")
            .display("identity", "HiFiShifter", "https://example.invalid", "1")
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap();
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"empty document".as_ptr(),
        };
        // SAFETY: factory、fixture、host 和属性在控制器整个生命周期内存活。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 工厂成功返回稳定 controller instance。
        let instance = unsafe { raw.read_unaligned() };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_some());
        // SAFETY: 唯一终止回调；之后不再访问控制器。
        unsafe {
            ((*instance.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(instance.documentControllerRef)
        };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_none());
        factory.entry().uninitialize().unwrap();
    }
}

fn controllers() -> &'static Mutex<HashMap<usize, Weak<DocumentSession>>> {
    static DOCUMENTS: OnceLock<Mutex<HashMap<usize, Weak<DocumentSession>>>> = OnceLock::new();
    DOCUMENTS.get_or_init(Default::default)
}

impl DocumentSession {
    pub(crate) fn publish_source_pcm(&self, id: String, pcm: Arc<super::source::SourcePcm>) {
        let _transaction = self.transaction.lock().unwrap();
        self.edit_sources
            .lock()
            .unwrap()
            .insert(id.clone(), pcm.clone());
        self.sources.lock().unwrap().insert(id, pcm);
        self.editor_audio_revision.fetch_add(1, Ordering::AcqRel);
    }
    /// 调用方持document事务；仅从真实region边创建曲线身份，不从轨名/源路径推断。
    pub(crate) fn parameter_identities_locked(
        &self,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> Result<
        std::collections::BTreeMap<String, crate::editor::parameter_atlas::RegionIdentity>,
        String,
    > {
        let ids = self.clip_ids.lock().unwrap();
        let regions = self.regions.lock().unwrap();
        timeline
            .clips
            .iter()
            .map(|clip| {
                let (key, _) = ids
                    .iter()
                    .find(|(_, id)| **id == clip.id)
                    .ok_or("missing actual region parameter edge")?;
                let region = regions
                    .get(key)
                    .ok_or("missing actual region parameter identity")?;
                Ok((
                    clip.id.clone(),
                    crate::editor::parameter_atlas::RegionIdentity {
                        key: *key,
                        item: self.region_items.lock().unwrap().get(key).cloned(),
                        source: region.audio_source_persistent_id.clone(),
                        modification: region.audio_modification_persistent_id.clone(),
                    },
                ))
            })
            .collect()
    }
    /// 文档惰性持有唯一原编辑actor；actor只弱引用文档，组件不持第二份history。
    pub(crate) fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        let _transaction = self.transaction.lock().unwrap();
        if !self.is_alive() {
            return Err("document closed".into());
        }
        self.editor
            .get_or_init(|| crate::editor::session::EditorSession::new(self))
            .clone()
    }
    pub(crate) fn flush_editor(&self) -> Result<(), String> {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.flush()?;
        }
        Ok(())
    }
    /// 组件撤销只关闭失效入口的view，不创建actor也不停止同文档其它入口。
    pub(crate) fn revoke_editor_views(&self) {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.revoke_closed_views();
        }
    }
    /// edit/model/scope分开计数；assignment变更不能由相同model误判成未变。
    pub(crate) fn editor_versions(&self) -> Result<(u64, u64, u64), String> {
        let _transaction = self.transaction.lock().unwrap();
        if !self.is_alive() {
            return Err("document closed".into());
        }
        Ok((
            self.edits.lock().unwrap().revision,
            self.revision.load(Ordering::Acquire),
            self.scope_revision.load(Ordering::Acquire),
        ))
    }
    fn workspace_projection_locked(
        &self,
        edits: &crate::state_channel::EditState,
    ) -> Result<String, String> {
        let mut timeline = self.workspace_timeline_locked()?;
        edits.apply(&mut timeline);
        // 【为什么音阶也在指纹里】`ensure_loaded` 靠"版本没变 + 本指纹相同"早退
        // （`editor/session.rs:819-843`）。音阶与音阶变化点都是**会影响渲染结果**的
        // 插件自有状态（`scale-signature` 进渲染缓存键），却不在版本号里 —— 不进指纹
        // 就会出现"改了音阶，早退成立，于是既不重载也不重渲染"。与 `fades` 同列，
        // 理由相同。
        //
        // 【为什么可以整体哈希】音阶点的 id 是确定性的（见
        // `plugin_musical_projection`），宿主读数带 1e-6 阈值，所以"内容没变 ⇒
        // 指纹相同"成立，不会造成每轮重载。
        let bytes = serde_json::to_vec(&(
            timeline.params_by_root_track,
            timeline.tracks,
            &edits.fades,
            &edits.groups,
            &timeline.project_scale_notes,
            &timeline.tempo_map,
        ))
        .map_err(|e| e.to_string())?;
        Ok(format!(
            "{}:{}",
            self.scope_revision.load(Ordering::Acquire),
            blake3::hash(&bytes).to_hex()
        ))
    }
    /// scope版本参与投影指纹；区域退出后即便edit/model没动也不能接受旧工作区。
    pub(crate) fn workspace_projection(&self) -> Result<String, String> {
        let _transaction = self.transaction.lock().unwrap();
        self.workspace_projection_locked(&self.edits.lock().unwrap())
    }
    /// 原GUI改变素材选择只换可见source投影；不修改参数authority、history或渲染generation。
    pub(crate) fn selected_source_parameters(
        &self,
        selection: &hifishifter_kernel::state::TimelineState,
    ) -> Result<
        std::collections::BTreeMap<String, hifishifter_kernel::state::TrackParamsState>,
        String,
    > {
        let _transaction = self.transaction.lock().unwrap();
        if !self.is_alive() {
            return Err("document closed".into());
        }
        let host = self.workspace_timeline_locked()?;
        Self::validate_workspace_geometry(&host, selection)?;
        let edits = self.edits.lock().unwrap();
        let mut grouped = host;
        edits.apply(&mut grouped);
        grouped.selected_clip_id = selection.selected_clip_id.clone();
        self.project_private_group_view(&mut grouped, &edits)?;
        Ok(grouped.params_by_root_track)
    }
    /// 短事务同时冻结授权、参数、PCM和三个版本，分析副本的文件IO由actor随后执行。
    pub(crate) fn workspace_snapshot(&self) -> Result<(WorkspaceSnapshot, u64, String), String> {
        let _transaction = self.transaction.lock().unwrap();
        for owner in self.renderer_owners() {
            owner.merge_pending_restore(self)?;
        }
        let mut timeline = self.workspace_timeline_locked()?;
        let mut edits = self.edits.lock().unwrap();
        edits.reconcile(&self.track_bindings.lock().unwrap())?;
        edits.apply(&mut timeline);
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        let available = self.edit_sources.lock().unwrap();
        let ids = timeline
            .clips
            .iter()
            .flat_map(|clip| {
                std::iter::once(&clip.source_path)
                    .chain(clip.takes.iter().map(|take| &take.source_path))
            })
            .filter_map(Option::as_ref)
            .collect::<std::collections::BTreeSet<_>>();
        let sources = ids
            .into_iter()
            .map(|id| {
                let pcm = available
                    .get(id)
                    .ok_or_else(|| format!("host PCM unavailable: {id}"))?;
                Ok((id.clone(), pcm.clone()))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let projection = self.workspace_projection_locked(&edits)?;
        self.project_ui_fades_locked(&mut timeline, &edits.fades);
        self.project_private_group_view(&mut timeline, &edits)?;
        Ok((
            WorkspaceSnapshot {
                timeline,
                sources,
                revision: edits.revision,
                model_revision: self.revision.load(Ordering::Acquire),
                audio_revision: self.editor_audio_revision.load(Ordering::Acquire),
            },
            self.scope_revision.load(Ordering::Acquire),
            projection,
        ))
    }
    /// 原分析副本可补媒体元信息，除此之外clip、take及轨道结构全部必须与宿主一致。
    fn validate_workspace_geometry(
        host: &hifishifter_kernel::state::TimelineState,
        client: &hifishifter_kernel::state::TimelineState,
    ) -> Result<(), String> {
        let normalize=|timeline:&hifishifter_kernel::state::TimelineState|->Result<serde_json::Value,String> {
            let mut timeline=timeline.clone();for clip in &mut timeline.clips {clip.normalize_takes();}
            let mut value=serde_json::to_value(timeline).map_err(|e|e.to_string())?;
            let metadata=["source_path_relative","duration_sec","duration_frames","source_sample_rate","source_channels",
                "source_file_fingerprint","source_file_mtime","source_file_size","waveform_preview","pitch_range"];
            if let Some(clips)=value["clips"].as_array_mut() {for clip in clips {
                for key in metadata {clip.as_object_mut().unwrap().remove(key);}
                if let Some(takes)=clip["takes"].as_array_mut() {for take in takes {for key in metadata {take.as_object_mut().unwrap().remove(key);}}}
            }}
            if let Some(tracks)=value["tracks"].as_array_mut() {for track in tracks {for key in ["volume","muted","solo","compose_enabled","pitch_analysis_algo"] {track.as_object_mut().unwrap().remove(key);}}}
            Ok(serde_json::json!({"tracks":value["tracks"],"clips":value["clips"],"project_sec":value["project_sec"]}))
        };
        if normalize(host)? != normalize(client)? {
            return Err("host workspace geometry is read-only".into());
        }
        Ok(())
    }
    /// 全工作区在单事务校验scope/模型/曲线并接受；未知范围或几何不静默忽略。
    // 整批接受入口保留，供不分组调用方使用。
    #[allow(dead_code)]
    pub(crate) fn accept_workspace_edits(
        &self,
        base_edit: u64,
        base_model: u64,
        client: &hifishifter_kernel::state::TimelineState,
        previous: &str,
    ) -> Result<(u64, u64, String), String> {
        self.accept_workspace_edits_with_groups(base_edit, base_model, client, previous, None)
    }
    /// 组根处理设置与参数delta同一事务提交，避免空父轨设置在音频准备时退回默认值。
    pub(crate) fn accept_workspace_edits_with_groups(
        &self,
        base_edit: u64,
        base_model: u64,
        client: &hifishifter_kernel::state::TimelineState,
        previous: &str,
        groups: Option<crate::editor::private_groups::TrackGroups>,
    ) -> Result<(u64, u64, String), String> {
        let _transaction = self.transaction.lock().unwrap();
        let model = self.revision.load(Ordering::Acquire);
        if model != base_model {
            return Err("Conflict: host model changed; local curves preserved".into());
        }
        let host = self.workspace_timeline_locked()?;
        let mut edits = self.edits.lock().unwrap();
        if self.workspace_projection_locked(&edits)? != previous || base_edit > edits.revision {
            return Err(
                "Conflict: workspace scope or curves changed; local curves preserved".into(),
            );
        }
        Self::validate_workspace_geometry(&host, client)?;
        let mut candidate = edits.merge(&host, client, edits.revision)?;
        candidate.reconcile(&self.track_bindings.lock().unwrap())?;
        let identities = self.parameter_identities_locked(&host)?;
        let mut previous_view = edits.params.clone();
        if !edits.atlas.is_empty() {
            let mut selected = host.clone();
            selected.selected_clip_id = client.selected_clip_id.clone();
            previous_view.extend(edits.atlas.project_roots(&selected, &identities)?);
        }
        candidate.atlas = edits
            .atlas
            .capture_changes(client, &identities, &previous_view)?;
        if let Some(groups) = groups {
            groups.validate()?;
            candidate.groups = groups;
        }
        let projection = self.workspace_projection_locked(&candidate)?;
        *edits = candidate;
        Ok((edits.revision, model, projection))
    }
    /// 合成只在短事务外计算；最终再次核对文档、scope、assignment与编辑代次。
    // 无进度回调的合成入口保留，供简单调用方使用。
    #[allow(dead_code)]
    pub(crate) fn apply_workspace_edits(
        &self,
        base_edit: u64,
        base_model: u64,
        previous: &str,
        cancel: Arc<AtomicBool>,
        current: impl Fn() -> bool,
    ) -> Result<(), String> {
        self.apply_workspace_edits_with_progress(
            base_edit, base_model, previous, cancel, current, None,
        )
    }
    pub(crate) fn apply_workspace_edits_with_progress(
        &self,
        base_edit: u64,
        base_model: u64,
        previous: &str,
        cancel: Arc<AtomicBool>,
        current: impl Fn() -> bool,
        progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
    ) -> Result<(), String> {
        let (edit, epoch, scope, inputs) = {
            let _transaction = self.transaction.lock().unwrap();
            if !self.is_alive()
                || !self.ready.load(Ordering::Acquire)
                || self.revision.load(Ordering::Acquire) != base_model
            {
                return Err("Conflict: host model changed during automatic apply".into());
            }
            let edits = self.edits.lock().unwrap().clone();
            if edits.revision != base_edit || self.workspace_projection_locked(&edits)? != previous
            {
                return Err("Conflict: workspace changed during automatic apply".into());
            }
            let inputs = self
                .renderer_owners()
                .into_iter()
                .filter(|owner| owner.renders_playback())
                .map(|owner| {
                    let (keys, input) = owner.capture_render_input(self, &edits, true)?;
                    Ok((owner, keys, input))
                })
                .collect::<Result<Vec<_>, String>>()?;
            (
                edits.revision,
                self.render_epoch.load(Ordering::Acquire),
                self.scope_revision.load(Ordering::Acquire),
                inputs,
            )
        };
        if !current() {
            return Err("automatic apply superseded".into());
        }
        let mut prepared = Vec::new();
        for (owner, keys, input) in inputs {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
            let mut snapshots = input.render_with_progress(cancel.clone(), progress.clone())?;
            super::snapshot::compact_prepared(&mut snapshots)?;
            prepared.push((owner, keys, snapshots));
        }
        let _transaction = self.transaction.lock().unwrap();
        if !self.is_alive()
            || !self.ready.load(Ordering::Acquire)
            || self.revision.load(Ordering::Acquire) != base_model
        {
            return Err("Conflict: host model changed during automatic apply".into());
        }
        if !current()
            || self.edits.lock().unwrap().revision != edit
            || self.render_epoch.load(Ordering::Acquire) != epoch
            || self.scope_revision.load(Ordering::Acquire) != scope
        {
            return Err("automatic apply superseded".into());
        }
        for (owner, keys, snapshots) in &prepared {
            if owner.is_closed() || owner.assigned_regions().map_err(|e| e.to_string())? != *keys {
                return Err("automatic apply superseded".into());
            }
            if !owner
                .snapshots
                .iter()
                .zip(snapshots)
                .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
            {
                return Err("retired snapshot budget exhausted; reopen instance".into());
            }
        }
        for (owner, keys, snapshots) in prepared {
            for (publisher, snapshot) in owner.snapshots.iter().zip(snapshots) {
                publisher
                    .publish(snapshot)
                    .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
            }
            owner.record_prepared(base_model, edit, epoch, scope, keys);
        }
        Ok(())
    }
    /// 宿主专属API调用前核对真实文档仍存活，销毁后的view不能沿旧project指针查询。
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
    /// 会话与模型同寿，controller 地址在工厂 allocation 完成后登记。
    pub fn new(id: DocumentId) -> Arc<Self> {
        Arc::new(Self {
            id,
            alive: AtomicBool::new(true),
            ..Default::default()
        })
    }

    /// 工厂成功返回前登记真实 controllerRef，失败或销毁时撤销。
    pub fn register(self: &Arc<Self>, key: usize) {
        self.controller.store(key, Ordering::Release);
        controllers()
            .lock()
            .unwrap()
            .insert(key, Arc::downgrade(self));
    }

    /// 同步关闭文档；保持 renderer 的原生接口存储，但撤销模型操作许可。
    pub fn close(&self) {
        {
            let _transaction = self.transaction.lock().unwrap();
            if !self.alive.swap(false, Ordering::AcqRel) {
                return;
            }
            self.scope_revision.fetch_add(1, Ordering::AcqRel);
        }
        // 不持transaction join：actor可能正在收尾短事务；先停止全部编辑/分析再清宿主图。
        self.host_undo.close();
        if let Some(Ok(editor)) = self.editor.get() {
            editor.close();
        }
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.playback.lock().unwrap().take();
        self.ready.store(false, Ordering::Release);
        let leases = {
            let mut renderers = self.renderers.lock().unwrap();
            self.alive.store(false, Ordering::Release);
            std::mem::take(&mut *renderers)
        };
        let mut controllers = controllers().lock().unwrap();
        let key = self.controller.load(Ordering::Acquire);
        if controllers
            .get(&key)
            .and_then(Weak::upgrade)
            .is_some_and(|current| current.id == self.id)
        {
            controllers.remove(&key);
        }
        drop(controllers);
        for renderer in leases {
            renderer.lease.destroy();
            if let Some(owner) = renderer.owner.upgrade() {
                owner.document_closed();
            }
            // companion guard 随 renderer 释放，禁止继续借用即将销毁的 controller。
        }
        self.sequence_regions.lock().unwrap().clear();
        self.sequence_track_index.lock().unwrap().clear();
        self.regions.lock().unwrap().clear();
        self.clip_ids.lock().unwrap().clear();
        self.region_items.lock().unwrap().clear();
        self.ui_tracks.lock().unwrap().clear();
        self.ui_known_tracks.lock().unwrap().clear();
        *self.ui_inventory_stamp.lock().unwrap() = None;
        self.timeline.lock().unwrap().take();
        self.track_bindings.lock().unwrap().clear();
        *self.edits.lock().unwrap() = Default::default();
        self.sources.lock().unwrap().clear();
        self.edit_sources.lock().unwrap().clear();
    }

    /// 控制器侧独立保留 lease，companion 先释放时 raw extension 存储仍须存活。
    pub fn attach(
        &self,
        owner: &Arc<ExtensionOwner>,
        lease: ExtensionControllerLease,
        companion: Option<CompanionControllerBinding<'static>>,
    ) -> Result<(), AraError> {
        let mut renderers = self.renderers.lock().unwrap();
        if !self.alive.load(Ordering::Acquire) {
            return Err(AraError::InvalidState("document is destroyed"));
        }
        renderers.push(RendererLease {
            lease,
            _companion: companion,
            owner: Arc::downgrade(owner),
        });
        Ok(())
    }

    /// 模型线程刷新所有仍存活 renderer，任何 source/geometry/访问变化都触发重新准备。
    pub fn prepare_renderers(&self) {
        let _transaction = self.transaction.lock().unwrap();
        self.ready.store(true, Ordering::Release);
        let owners = self
            .renderers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect::<Vec<_>>();
        // 先统一恢复全部组件权威，再允许任何后台任务捕获共享revision。
        for owner in &owners {
            if let Err(error) = owner.merge_pending_restore(self) {
                log::warn!("[ara] instance state unresolved: {error}");
            }
        }
        drop(_transaction);
        for owner in owners {
            owner.refresh_reaper_state_for_model();
            owner.prepare();
        }
    }

    /// 同一ARA文档的编辑权威共享，输出仍按各renderer分配隔离。
    pub fn renderer_owners(&self) -> Vec<Arc<ExtensionOwner>> {
        self.renderers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect()
    }

    /// 播放渲染器的诊断快照（诊断包用）。
    ///
    /// 【为什么只要"准备状态"】逐输出的区间读数需要当前分配区间，而诊断包在任意
    /// 时刻被导出 —— 不值得为它触发一次完整快照。这里给的是"忙不忙、有没有错、
    /// 准备的是哪一版"，正好覆盖"渲染结果陈旧/一直没准备好"这两类报障。
    pub fn renderer_diagnostics(&self) -> serde_json::Value {
        let renderers = self
            .renderer_owners()
            .into_iter()
            .filter(|owner| owner.renders_playback())
            .map(|owner| {
                let (busy, error) = owner.local_preparation_state();
                serde_json::json!({
                    "busy": busy,
                    "error": error,
                    "prepared": owner.prepared_snapshot(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({ "renderers": renderers })
    }

    /// 在旧内容可能被改变前立即撤销发布；不回收实时读者可能仍持有的旧快照。
    pub fn clear_renderers(&self) {
        let _transaction = self.transaction.lock().unwrap();
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.revoke_snapshots();
    }

    /// 授权开关只撤销播放许可，不改变宿主内容/布局的乐观并发版本。
    pub fn revoke_renderers(&self) {
        let _transaction = self.transaction.lock().unwrap();
        self.revoke_snapshots();
    }

    fn revoke_snapshots(&self) {
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.ready.store(false, Ordering::Release);
        let owners = self
            .renderers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .collect::<Vec<_>>();
        for owner in owners {
            owner.cancel_preparation();
            owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
    }

    /// 只借出仍存活的文档，拒绝已销毁或非本插件的控制器。
    pub fn lookup(key: usize) -> Option<Arc<Self>> {
        controllers()
            .lock()
            .unwrap()
            .get(&key)
            .and_then(Weak::upgrade)
            .filter(|session| session.alive.load(Ordering::Acquire))
    }
}
