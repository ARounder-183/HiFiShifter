//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 全PCM内容参与身份，重开时同路径不同音频不能命中旧缓存。
pub(super) fn pcm_fingerprint(pcm: &super::source::SourcePcm) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(&pcm.sample_rate.to_le_bytes());
    hash.update(&(pcm.planes.len() as u64).to_le_bytes());
    for plane in &pcm.planes {
        for sample in plane {
            hash.update(&sample.to_le_bytes());
        }
    }
    hash.finalize().to_hex().to_string()
}

/// 只读元数据供非实时读者消费；UI/model采集，音频线程不访问此缓存或锁owner。
///
/// `value` 是**逐 region** 的绑定列表，不是一个实例一个几何：挂在 REAPER folder
/// 轨上的 FX 会被 ARA 分配整组 region，而 `parent(2)`（实例自己的 take）结构上只
/// 能覆盖其中一个。见 [`ExtensionOwner::reaper_geometries`]。
#[derive(Clone)]
struct CachedHostGeometry {
    model: u64,
    scope: u64,
    change: Option<i32>,
    value: Result<Vec<crate::host::geometry::BoundHostGeometry>, String>,
}

/// 两个宿主/ARA 浮点量是否相容；判据全局唯一，见 `host::geometry::host_value_compatible`
/// （绑定 / 分割规划 / 写回前置检查 / 剪贴板共用，避免"规划接受、绑定拒绝"的不对称）。
fn compatible(a: f64, b: f64) -> bool {
    crate::host::geometry::host_value_compatible(a, b)
}

/// 在候选宿主 item 里为一条 ARA region 找**唯一**匹配的几何。
///
/// 【为什么必须"唯一才采信"】ARA 侧没有 region → item 的显式边（见
/// `crate::ara::mapping` 的 `LOST_FIELDS`）：`parent(2)` 只覆盖 FX 自己的那一个
/// take，源 persistentID 会被多个 item 共享，名字/轨道序号按既有纪律不得用于猜
/// 关联。因此只能按**几何**比对；而后续所有宿主写回都沿这条边定位 item —— 猜错
/// 就是写到别的 item 上。所以 0 个匹配或多个匹配一律返回 `None`：宁可不绑定。
///
/// 四项都要相容：播放位置、播放时长、源内起点、播放速率。只比时间窗会在
/// "同一位置的两条子轨各有一个 item"这种常见布局上直接产生歧义。
///
/// 【`preferred` 是消歧而不是放宽】`preferred` 是这条 region **上一次**绑定的 item GUID。
/// 复制粘贴的安全副本、同一素材的多个 item 会给出多个几何全等的候选；纯"唯一才采信"
/// 会把它们**全部**判为歧义、两个都丢（用户看到片段凭空消失）。当且仅当候选里恰好有
/// 一个 GUID 等于上次绑定时采信它 —— 这是**稳定性**，不是猜测：上一次的绑定本身就来自
/// 一次唯一匹配，保持它不会把写回引到新的对象上。
fn unique_geometry_for_region(
    region: &crate::ara::AraPlaybackRegion,
    candidates: &[crate::host::geometry::HostClipGeometry],
    preferred: Option<&str>,
) -> Option<crate::host::geometry::HostClipGeometry> {
    let playback_rate = if region.duration_in_playback_time > 0.0 {
        region.duration_in_modification_time / region.duration_in_playback_time
    } else {
        f64::NAN
    };
    let matched: Vec<&crate::host::geometry::HostClipGeometry> = candidates
        .iter()
        .filter(|candidate| {
            compatible(candidate.start_sec, region.start_in_playback_time)
                && compatible(candidate.duration_sec, region.duration_in_playback_time)
                && compatible(
                    candidate.source_start_sec,
                    region.start_in_modification_time,
                )
                && compatible(candidate.playback_rate, playback_rate)
        })
        .collect();
    let first = *matched.first()?;
    if matched.len() == 1 {
        return Some(first.clone());
    }
    // 有歧义：只有"上次绑定的那个"能打破它 —— 且它必须在候选里。
    if let Some(preferred) = preferred {
        if let Some(found) = matched
            .iter()
            .find(|candidate| candidate.item_id == preferred)
        {
            return Some((*found).clone());
        }
    }
    None
}

#[derive(Clone, PartialEq, Eq)]
struct PreparedVersion {
    model: u64,
    edit: u64,
    epoch: u64,
    scope: u64,
    keys: Vec<u64>,
}

/// 宿主音频是否已经到达本实例 —— **语言无关**的分类，文案由前端 catalog 本地化。
///
/// 【为什么不是"有没有音频"这个布尔】用户侧有两种完全不同的处境，处置方式也不同：
/// 插件挂在了 folder 父轨上是**用法问题**，必须给出具体指引（REAPER 按轨道管理 ARA
/// 插件）；普通等待则是宿主还没分配 region，只需说明"等待中"。把两者压成一个布尔，
/// 前端就只能给出一句对谁都不准的泛泛提示 —— 而"轨道和 item 都看得见、能拖、但没
/// 内容"这种症状极具误导性，泛泛提示等于没说。
///
/// 【措辞红线】不得表述为"ARA 规范不支持跨轨"：ARA 2.0 规范**允许**一个实例服务
/// 多个 region sequence，是 **REAPER 选择按轨道管理 ARA 实例**。错的解释会把日后
/// 真正可行的改进方向带偏。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum HostAudioState {
    /// 没有等待中的 item（含空工程）—— 无可抱怨。
    #[default]
    Ready,
    /// 清单里有 item，但本实例一个 region 都没拿到：宿主尚未分配。
    AwaitingRegions,
    /// 同上，且本 FX 所在轨道是 folder 父轨：组内子轨的音频不会交给这个实例。
    FolderParentWithoutRegions,
}

impl HostAudioState {
    /// 语言无关的分类名；前端按此查 catalog（与 `ara_host_fields:` 同一原则）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            HostAudioState::Ready => "ready",
            HostAudioState::AwaitingRegions => "awaiting_regions",
            HostAudioState::FolderParentWithoutRegions => "folder_parent_without_regions",
        }
    }
}

/// 最近一次清单刷新得出的宿主音频读数。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostAudioStatus {
    pub state: HostAudioState,
    /// 清单里**尚未被任何已分配 region 认领**的 item 数（"还在等音频"的 clip 数）。
    pub waiting_items: usize,
}

/// 纯函数：由宿主侧事实判定状态。可独立单测，不需要 REAPER。
///
/// 判据只用两项可核实的事实：本 FX 轨是不是 folder 父轨、清单里有多少 item 认领
/// 到了 region。不推断、不猜 —— 读不出 folder 结构时调用方传 `false`，此时只会
/// 退化成"等待中"，不会给出一个可能指错方向的 folder 提示。
pub(crate) fn classify_host_audio(
    fx_is_folder_parent: bool,
    item_count: usize,
    claimed_items: usize,
) -> HostAudioStatus {
    let waiting_items = item_count.saturating_sub(claimed_items);
    let state = if claimed_items > 0 || item_count == 0 {
        // 有认领 = 正常；一个 item 都没有 = 没有"等待"可言（空工程不该报障）。
        HostAudioState::Ready
    } else if fx_is_folder_parent {
        HostAudioState::FolderParentWithoutRegions
    } else {
        HostAudioState::AwaitingRegions
    };
    HostAudioStatus {
        state,
        waiting_items: if state == HostAudioState::Ready {
            0
        } else {
            waiting_items
        },
    }
}

/// 原生接口与只读元数据属于真实组件，缓存不跨其文档/分配/host变更复用。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<ExtensionBinding>>,
    document: Mutex<Option<std::sync::Weak<super::document::DocumentSession>>>,
    assignments: Mutex<HashMap<i32, Vec<RegionKey>>>,
    sequences: Mutex<HashMap<i32, Vec<u64>>>,
    role: AtomicI32,
    roles: AtomicI32,
    closed: std::sync::atomic::AtomicBool,
    writer_id: AtomicU64,
    reaper: Mutex<Option<Arc<crate::host::reaper::ReaperHost>>>,
    host_geometry: Mutex<Option<CachedHostGeometry>>,
    /// 每个真实单region播放实例的有效mute；process仅读原子，不查询宿主或清修音参数。
    host_item_muted: std::sync::atomic::AtomicBool,
    prepared: Mutex<Option<PreparedVersion>>,
    pub snapshots: [super::snapshot::SnapshotPublisher; 2],
    pub(crate) edits: Arc<Mutex<crate::state_channel::EditState>>,
    pending_restore: Mutex<Option<crate::state_channel::EditState>>,
    channel: Mutex<Option<hifishifter_ara_ipc::Server>>,
    preparation: std::sync::OnceLock<Result<super::preparation::PreparationQueue, String>>,
    prepare_owner: Mutex<Option<std::sync::Weak<ExtensionOwner>>>,
    pub(crate) clock: std::sync::OnceLock<Arc<super::transport::TransportClock>>,
    /// 最近一次清单刷新得出的宿主音频读数；GUI 载荷读它（见 [`HostAudioStatus`]）。
    host_audio: Mutex<HostAudioStatus>,
    /// 上一次"重"宿主枚举（几何 + 清单）的时刻。
    ///
    /// 【为什么把时钟与枚举分开节流】UI 定时器是 20ms（50Hz）。播放光标必须跟得上
    /// （`host.sample` 很便宜），但几何/清单枚举即使有代次缓存，每 tick 仍要付
    /// `GetProjectStateChangeCount` 等若干宿主调用。50Hz 下这些调用在真实工程里会累积
    /// 成可观的 UI 线程占用（且与 actor/渲染线程争同一把 document 事务）。所以时钟保持
    /// 20ms，枚举降到 [`HOST_ENUMERATION_INTERVAL`]。宿主真的变了时，下一次节流点就会
    /// 捕捉到（≤100ms 延迟），用户不可感。
    last_enumeration: Mutex<Option<std::time::Instant>>,
}

/// 宿主几何/清单枚举的节流间隔（见 `ExtensionOwner::last_enumeration`）。
const HOST_ENUMERATION_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

#[cfg(test)]
mod bound_tests {
    use super::*;
    /// 实际原kernel PCM必须随HFS形状变化；只委托head时不能又烘焙宿主tail。
    #[test]
    fn owned_fade_shape_changes_kernel_pcm_and_state_roundtrip_and_undo_clears_it() {
        use hifishifter_kernel::editor::ParamHost;
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let key = (&*ids[0] as *const u8) as u64;
        {
            let mut timeline = document.timeline.lock().unwrap();
            let clip = &mut timeline.as_mut().unwrap().clips[0];
            clip.start_sec = 1.;
            clip.length_sec = 1.;
            clip.takes[0].source_end_sec = 1.;
            clip.normalize_takes();
        }
        {
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&key).unwrap();
            region.start_in_playback_time = 1.;
            region.duration_in_playback_time = 1.;
            region.duration_in_modification_time = 1.;
            region.has_content_based_fade_at_head = true;
        }
        let pcm = Arc::new(super::super::source::SourcePcm {
            sample_rate: 44100,
            planes: vec![vec![0.5; 44100]],
            version: 0,
            _reservation: None,
        });
        document
            .edit_sources
            .lock()
            .unwrap()
            .insert("ara://source".into(), pcm.clone());
        document
            .sources
            .lock()
            .unwrap()
            .insert("ara://source".into(), pcm);
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.clear_markers();
        for (name, value) in [
            ("D_LENGTH", 1.),
            ("D_PLAYRATE", 1.),
            ("D_FADEINLEN", 0.25),
            ("D_FADEINLEN_AUTO", 0.),
            ("D_FADEOUTLEN", 0.25),
            ("D_FADEOUTLEN_AUTO", 0.),
        ] {
            host.set_value(name, value);
        }
        unsafe {
            owners[0].bind_reaper_host(host.context());
        }
        owners[0].refresh_reaper_transport();
        let editor = owners[0].editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let before = owners[0].encode_state().unwrap();
        let render = || {
            let input = {
                let _transaction = document.transaction.lock().unwrap();
                owners[0]
                    .capture_render_input(&document, &document.edits.lock().unwrap(), true)
                    .unwrap()
                    .1
            };
            assert_eq!(
                input.timeline.as_ref().unwrap().clips[0].fade_out_sec,
                0.,
                "未委托tail仍归宿主，不能重复淡化"
            );
            input
                .render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
                .unwrap()
        };
        let original = render();
        let index = 5512;
        assert!((original[0].left[index] as f64 - 0.5 * 0.5_f64.powf(0.45)).abs() < 0.001);
        let clip = editor.timeline().lock().unwrap().clips[0].id.clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &serde_json::json!({"clipId":clip,"fadeInShape":5.,"fadeInDir":0.}),
            )
            .unwrap();
        crate::editor::host_edit::execute(&owners[0], plan, || document.is_alive()).unwrap();
        let updated = render();
        assert!((updated[0].left[index] as f64 - 0.25).abs() < 0.001);
        assert!((updated[0].left[index] - original[0].left[index]).abs() > 0.1);
        let saved = owners[0].encode_state().unwrap();
        let value: serde_json::Value = serde_json::from_slice(&saved).unwrap();
        assert_eq!(value["version"], 4);
        assert_eq!(value["edits"]["fades"].as_object().unwrap().len(), 1);
        owners[0].restore_state(&before).unwrap();
        assert!(
            document.edits.lock().unwrap().fades.is_empty(),
            "恢复旧无形状状态必须撤销HFS形状，不保留未来编辑"
        );
        owners[0].restore_state(&saved).unwrap();
        assert_eq!(
            document
                .edits
                .lock()
                .unwrap()
                .fades
                .values()
                .next()
                .unwrap()
                .in_shape,
            5.
        );
        document.close();
    }
    use crate::ara_entry::HostEntry;
    use ara2_bridge::companion::vst3::ffi::{ara2_vst3_plugin_entry_bind, ARA2_VST3_OK};
    use ara2_bridge::companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;
    /// v2首次真实恢复就建立源basis；后续移动/裁切/线性拉伸无需用户再落笔才迁移。
    #[test]
    fn legacy_v2_first_restore_promotes_source_basis_before_host_transform() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let key = (&*ids[0] as *const u8) as u64;
        {
            let _transaction = document.transaction.lock().unwrap();
            let mut timeline = document.timeline.lock().unwrap();
            let timeline = timeline.as_mut().unwrap();
            timeline.project_sec = 2.;
            let clip = &mut timeline.clips[0];
            clip.start_sec = 1.;
            clip.length_sec = 1.;
            clip.takes[0].source_end_sec = 1.;
            clip.normalize_takes();
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&key).unwrap();
            region.start_in_playback_time = 1.;
            region.duration_in_modification_time = 1.;
            region.duration_in_playback_time = 1.;
        }
        let host = owners[0].assigned_timeline(&document).unwrap();
        let mut client = host.clone();
        let mut edit = vec![0_f32; 22];
        for (frame, value) in edit.iter_mut().enumerate().take(21).skip(10) {
            *value = 60. + (frame - 10) as f32;
        }
        let params = hifishifter_kernel::state::TrackParamsState {
            frame_period_ms: 100.,
            pitch_orig: vec![57.; 22],
            pitch_edit: edit,
            pitch_edit_user_modified: true,
            ..Default::default()
        };
        client.params_by_root_track.insert("track".into(), params);
        let mut legacy = crate::state_channel::EditState::default()
            .merge(&host, &client, 0)
            .unwrap();
        legacy
            .reconcile(&std::collections::BTreeMap::from([(
                "track".into(),
                vec![("modification".into(), "ara://source".into())],
            )]))
            .unwrap();
        let bytes = legacy.encode().unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["version"],
            2
        );
        document.ready.store(false, Ordering::Release);
        owners[0].restore_state(&bytes).unwrap();
        document.prepare_renderers();
        let saved = owners[0].encode_state().unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&saved).unwrap()["version"],
            3
        );
        let atlas = document.edits.lock().unwrap().atlas.clone();
        assert_eq!(atlas.regions.len(), 1);
        let identities = {
            let _transaction = document.transaction.lock().unwrap();
            document.parameter_identities_locked(&host).unwrap()
        };
        let unchanged = atlas.project_roots(&host, &identities).unwrap();
        assert_eq!(
            &unchanged["track"].pitch_edit[10..21],
            &(60..=70).map(|note| note as f32).collect::<Vec<_>>()
        );
        let mut moved = host.clone();
        let clip = &mut moved.clips[0];
        clip.start_sec = 3.;
        clip.length_sec = 2.;
        clip.takes[0].source_start_sec = 0.25;
        clip.takes[0].source_end_sec = 0.75;
        clip.takes[0].playback_rate = 0.25;
        clip.normalize_takes();
        moved.project_sec = 5.;
        let followed = atlas.follow_geometry(&moved, &identities).unwrap();
        let roots = followed.project_roots(&moved, &identities).unwrap();
        assert_eq!(roots["track"].pitch_edit[10], 0.);
        assert_eq!(roots["track"].pitch_edit[30], 62.5);
        assert_eq!(roots["track"].pitch_edit[40], 65.);
        assert_eq!(roots["track"].pitch_edit[50], 67.5);
        let local = followed.project_local(&moved, &identities).unwrap();
        assert_eq!(local[&host.clips[0].id].pitch_edit[0], 62.5);
        let mut cold = crate::state_channel::EditState::default();
        cold.restore(&saved).unwrap();
        let rebound = cold.atlas.rebind(&moved, &identities).unwrap();
        assert_eq!(
            rebound.project_local(&moved, &identities).unwrap()[&host.clips[0].id].pitch_edit,
            local[&host.clips[0].id].pitch_edit
        );
        document.close();
    }

    /// v3源basis按组件真实区域保存，冷恢复不能借另一轨道的atlas或沿旧session key。
    #[test]
    fn source_parameter_atlas_state_is_scoped_and_rebound_without_gui() {
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let mut client = document.workspace_timeline().unwrap();
        for (root, note) in [("track", 60.), ("b", 67.)] {
            client.params_by_root_track.insert(
                root.into(),
                hifishifter_kernel::state::TrackParamsState {
                    frame_period_ms: 5.,
                    pitch_edit_user_modified: true,
                    pitch_orig: vec![57., 57.],
                    pitch_edit: vec![note, note],
                    ..Default::default()
                },
            );
        }
        document
            .accept_workspace_edits(
                0,
                document.revision.load(Ordering::Acquire),
                &client,
                &document.workspace_projection().unwrap(),
            )
            .unwrap();
        let saved = owners
            .iter()
            .map(|owner| owner.encode_state().unwrap())
            .collect::<Vec<_>>();
        let payloads = saved
            .iter()
            .map(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).unwrap())
            .collect::<Vec<_>>();
        document.close();
        for payload in &payloads {
            assert_eq!(payload["version"], 3);
            assert_eq!(
                payload["edits"]["atlas"]["regions"]
                    .as_object()
                    .unwrap()
                    .len(),
                1,
                "不能保存其它组件区域"
            );
        }
        let (cold, restored, _cold_ids) = crate::editor::session::tests::workspace_fixture();
        let cold_doc = cold.session();
        for index in 0..2 {
            restored[index].restore_state(&saved[index]).unwrap();
        }
        cold_doc.prepare_renderers();
        let atlas = cold_doc.edits.lock().unwrap().atlas.clone();
        cold_doc.close();
        assert_eq!(atlas.regions.len(), 2);
        assert!(
            atlas
                .regions
                .values()
                .all(|record| record.identity.key != 0),
            "JSON旧key不得冒充新会话身份"
        );
    }

    /// 有宿主fade缓存后，快照/活actor装饰自动变化；不推进音频generation，不再烘焙普通fade。
    #[test]
    fn host_fades_project_into_gui_and_refresh_without_reloading_or_baking_audio() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let first = (&*ids[0] as *const u8) as u64;
        {
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&first).unwrap();
            region.start_in_playback_time = 1.;
            region.duration_in_playback_time = 4.;
        }
        let host = crate::host::reaper::ReaperFixture::new();
        unsafe {
            owners[0].bind_reaper_host(host.context());
        }
        owners[0].refresh_reaper_transport();
        let (response, _, _) = document.workspace_snapshot().unwrap();
        assert_eq!(response.timeline.clips[0].fade_in_sec, 0.2);
        let actor = owners[0].editor_session().unwrap();
        let (reply, rx) = std::sync::mpsc::channel();
        let (events, _) = std::sync::mpsc::sync_channel(128);
        let sink = crate::editor::session::UiSink {
            view_id: "fade-projection".into(),
            reply,
            events,
            closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let call = |command: &str| {
            actor
                .enqueue(crate::editor::session::UiRequest {
                    id: 1,
                    command: command.into(),
                    args: serde_json::json!({}),
                    sink: sink.clone(),
                    link: None,
                })
                .unwrap();
            let response: serde_json::Value =
                rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let initial = call("get_timeline_state");
        assert_eq!(
            initial["clips"][0]["host_fades"]["curve_mode"],
            "reaper_new"
        );
        assert_eq!(initial["clips"][0]["host_fades"]["in_curvature"], -0.2);
        assert_eq!(initial["clips"][0]["host_fades"]["in_s"], -0.3);
        let generation = call("plugin_get_apply_state")["generation"].clone();
        host.set_value("D_FADEINLEN", 0.75);
        owners[0].refresh_reaper_transport();
        host.set_value("D_FADEINDIR2_NEW", 0.65);
        owners[0].refresh_reaper_transport();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            let timeline = call("get_timeline_state");
            let updated =
                (timeline["clips"][0]["fade_in_sec"].as_f64().unwrap() - 0.75).abs() < 1e-8;
            if updated {
                assert_eq!(timeline["clips"][0]["host_fades"]["in_s"], 0.65);
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let (keys, input) = {
            let _transaction = document.transaction.lock().unwrap();
            owners[0]
                .capture_render_input(&document, &document.edits.lock().unwrap(), true)
                .unwrap()
        };
        let canonical = input.timeline.unwrap().clips[0].fade_in_sec;
        let after = call("plugin_get_apply_state")["generation"].clone();
        document.close();
        assert_eq!(after, generation);
        assert_eq!(keys, vec![first]);
        assert_eq!(canonical, 0.);
    }
    /// 只打开editor-only入口时，也必须采集同文档隐藏playback owner的元数据。
    #[test]
    fn task38b_one_editor_refresh_collects_hidden_playback_metadata() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let first = (&*ids[0] as *const u8) as u64;
        {
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&first).unwrap();
            region.start_in_playback_time = 1.0;
            region.duration_in_playback_time = 4.0;
        }
        let editor = Arc::new(ExtensionOwner::default());
        let raw = editor
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: fixture保留真实region、document及extension；editor入口没有自己的take。
        unsafe {
            let ext = &*raw;
            for key in [&*ids[0] as *const u8, &*ids[1] as *const u8] {
                ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
                    ext.editorRendererRef,
                    key.cast_mut().cast(),
                );
            }
        }
        let host = crate::host::reaper::ReaperFixture::new();
        unsafe {
            owners[0].bind_reaper_host(host.context());
        }
        host.reset();
        editor.refresh_reaper_transport();
        let metadata = owners[0].host_geometry_metadata();
        let entry_metadata = editor.host_geometry_metadata();
        document.close();
        let metadata = metadata.expect("单一editor入口必须让隐藏playback的只读数据就绪");
        assert_eq!(metadata[0].region_key, first);
        assert_eq!(metadata[0].geometry.fade_in_sec, 0.2);
        assert!(entry_metadata.is_err(), "多区域editor不能冒充唯一take绑定");
    }

    /// 隐藏getter重入关闭唯一GUI入口后，不能借另一活owner继续该入口的采集批次。
    #[test]
    fn task38b_hidden_getter_revokes_closed_editor_batch_before_next_renderer() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let editor = Arc::new(ExtensionOwner::default());
        let raw = editor
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        unsafe {
            let ext = &*raw;
            ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
                ext.editorRendererRef,
                (&*ids[0] as *const u8).cast_mut().cast(),
            );
        }
        let host = crate::host::reaper::ReaperFixture::new();
        unsafe {
            owners[0].bind_reaper_host(host.context());
        }
        // assignment 回调会提前缓存“无宿主接口”的 Err；只观测这次被撤销的读取批次。
        for owner in &owners {
            owner.host_geometry.lock().unwrap().take();
        }
        host.reset();
        let weak = Arc::downgrade(&editor);
        *host.hook.borrow_mut() = Some((
            "D_LENGTH".into(),
            Box::new(move || weak.upgrade().unwrap().stop_editor()),
        ));
        editor.refresh_reaper_transport();
        let calls = host.calls();
        let sampled_next = owners[1].host_geometry.lock().unwrap().is_some();
        let sampled_first = owners[0].host_geometry.lock().unwrap().is_some();
        document.close();
        assert_eq!(calls.last().map(String::as_str), Some("D_LENGTH"));
        assert!(!sampled_first, "撤销后的第一份数据不能发布");
        assert!(!sampled_next, "入口关闭后不能继续读取另一活renderer");
    }

    /// 稳定工程只检查版本，不在每个UI tick重复读取整组markers；fade改变仍刷新。
    #[test]
    fn task38b_stable_geometry_skips_raw_fields_but_fade_change_and_model_refresh_do_not() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        {
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&first).unwrap();
            region.start_in_playback_time = 1.;
            region.duration_in_playback_time = 4.;
        }
        let host = crate::host::reaper::ReaperFixture::new();
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let first_metadata = owner.host_geometry_metadata().unwrap();
        host.reset();
        owner.refresh_reaper_transport();
        let same = owner.host_geometry_metadata().unwrap();
        let stable_calls = host.calls();
        host.set_value("D_FADEINLEN", 0.75);
        host.reset();
        owner.refresh_reaper_transport();
        let changed = owner.host_geometry_metadata().unwrap();
        let changed_calls = host.calls();
        host.reset();
        document.revision.fetch_add(1, Ordering::AcqRel);
        owner.refresh_reaper_transport();
        let model_calls = host.calls();
        document.close();
        assert_eq!(same, first_metadata);
        assert!(
            !stable_calls
                .iter()
                .any(|name| name == "count" || name == "D_FADEINLEN"),
            "稳定UI tick不能重读全量几何: {stable_calls:?}"
        );
        assert_eq!(changed[0].geometry.fade_in_sec, 0.75);
        assert!(changed_calls.iter().any(|name| name == "D_FADEINLEN"));
        assert!(
            model_calls.iter().any(|name| name == "D_FADEINLEN"),
            "相同project counter但新ARA model仍须重新核对绑定"
        );
    }

    /// 原生getter可同步调用真实doc.close/owner.stop/assignment回调，不能持任何内部锁。
    #[test]
    fn task38a_owner_gate_rechecks_document_close_owner_scope_and_model_after_transport_state() {
        for change in ["document", "owner", "scope", "model"] {
            let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
            let document = model.session();
            let owner = owners[0].clone();
            let host = crate::host::reaper::ReaperFixture::new();
            unsafe {
                owner.bind_reaper_host(host.context());
            }
            host.reset();
            let weak = Arc::downgrade(&owner);
            let doc = document.clone();
            *host.hook.borrow_mut() = Some((
                "state".into(),
                Box::new(move || match change {
                    "document" => doc.close(),
                    "owner" => weak.upgrade().unwrap().stop_editor(),
                    "scope" => {
                        doc.scope_revision.fetch_add(1, Ordering::AcqRel);
                    }
                    _ => {
                        doc.revision.fetch_add(1, Ordering::AcqRel);
                    }
                }),
            ));
            owner.refresh_reaper_transport();
            let calls = host.calls();
            let pose = document.clock.diagnostics()["reaper_position_authority"].clone();
            document.close();
            assert_eq!(
                calls,
                ["validate:ReaProject*", "state"],
                "{change}: revoked batch must not continue any getter"
            );
            assert_eq!(pose, false, "{change}: revoked result cannot publish");
        }
    }

    /// 同一owner只有唯一真实region才有typed几何，位置相等不会建立额外身份关系。
    #[test]
    fn task38a_owner_geometry_requires_unique_actual_assignment_and_compatible_playback_window() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        let second = (&*ids[1] as *const u8) as u64;
        let host = crate::host::reaper::ReaperFixture::new();
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        host.reset();
        assert!(owner
            .reaper_geometry()
            .unwrap_err()
            .contains("incompatible"));
        {
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&first).unwrap();
            region.start_in_playback_time = 1.;
            region.duration_in_playback_time = 4.;
        }
        assert_eq!(owner.reaper_geometry().unwrap().region_key, first);
        let raw = owner.binding.lock().unwrap().as_ref().unwrap().as_raw();
        // playback仍只有first，但editor另一region也属于此owner，不能忽略它伪造唯一性。
        unsafe {
            let ext = &*raw;
            ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
                ext.editorRendererRef,
                second as *mut _,
            );
        }
        host.reset();
        assert!(owner.reaper_geometry().unwrap_err().contains("exactly one"));
        assert!(host.calls().is_empty());
        unsafe {
            let ext = &*raw;
            ((*ext.editorRendererInterface).removePlaybackRegion.unwrap())(
                ext.editorRendererRef,
                second as *mut _,
            );
        }
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                second as *mut _,
            );
        }
        host.reset();
        assert!(owner.reaper_geometry().unwrap_err().contains("exactly one"));
        assert!(host.calls().is_empty());
        unsafe {
            let ext = &*raw;
            for key in [first, second] {
                ((*ext.playbackRendererInterface)
                    .removePlaybackRegion
                    .unwrap())(ext.playbackRendererRef, key as *mut _);
            }
        }
        host.reset();
        assert!(owner.reaper_geometry().unwrap_err().contains("exactly one"));
        assert!(host.calls().is_empty());
        document.close();
    }

    #[test]
    fn task38a_owner_geometry_getter_reentry_discards_closed_or_changed_scope() {
        for change in ["document", "owner", "scope", "model"] {
            let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
            let document = model.session();
            let owner = owners[0].clone();
            let host = crate::host::reaper::ReaperFixture::new();
            unsafe {
                owner.bind_reaper_host(host.context());
            }
            host.reset();
            let weak = Arc::downgrade(&owner);
            let doc = document.clone();
            *host.hook.borrow_mut() = Some((
                "D_LENGTH".into(),
                Box::new(move || match change {
                    "document" => doc.close(),
                    "owner" => weak.upgrade().unwrap().stop_editor(),
                    "scope" => {
                        doc.scope_revision.fetch_add(1, Ordering::AcqRel);
                    }
                    _ => {
                        doc.revision.fetch_add(1, Ordering::AcqRel);
                    }
                }),
            ));
            let result = owner.reaper_geometry();
            let calls = host.calls();
            document.close();
            assert!(result.unwrap_err().contains("authorization revoked"));
            assert_eq!(calls.last().unwrap(), "D_LENGTH", "{change}");
        }
    }

    /// 可消费副本只含Rust数据；普通fade变化可在UI重新采集，代次变化/关闭不返回旧值。
    #[test]
    fn task38a_owner_cached_geometry_is_read_only_and_revocable_without_host_calls() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        {
            let mut regions = document.regions.lock().unwrap();
            let region = regions.get_mut(&first).unwrap();
            region.start_in_playback_time = 1.;
            region.duration_in_playback_time = 4.;
        }
        let host = crate::host::reaper::ReaperFixture::new();
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        assert!(owner.host_geometry_metadata().is_err());
        owner.refresh_reaper_transport();
        host.reset();
        let cached = owner.host_geometry_metadata().unwrap();
        assert_eq!(cached[0].region_key, first);
        assert_eq!(cached[0].geometry.fade_in_sec, 0.2);
        assert!(host.calls().is_empty());
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    assert_eq!(owner.host_geometry_metadata().unwrap(), cached);
                    assert!(owner.reaper_geometry().is_err());
                })
                .join()
                .unwrap();
        });
        assert!(host.calls().is_empty());
        let revision = document.revision.load(Ordering::Acquire);
        host.set_value("D_FADEINLEN", 0.75);
        owner.refresh_reaper_transport();
        assert_eq!(document.revision.load(Ordering::Acquire), revision);
        assert_eq!(
            owner.host_geometry_metadata().unwrap()[0]
                .geometry
                .fade_in_sec,
            0.75
        );
        document.scope_revision.fetch_add(1, Ordering::AcqRel);
        assert!(owner.host_geometry_metadata().is_err());
        document.close();
        assert!(owner.host_geometry_metadata().is_err());
        assert!(owner.host_geometry.lock().unwrap().is_none());
    }

    /// 测试专用同步驱动：冻结阶段持锁，真正内核计算阶段不借文档事务。
    fn render_test_edits(
        owner: &ExtensionOwner,
        document: &super::super::document::DocumentSession,
        edits: &crate::state_channel::EditState,
    ) -> Result<Vec<super::super::snapshot::PlaybackSnapshot>, String> {
        let input = {
            let _transaction = document.transaction.lock().unwrap();
            owner.capture_render_input(document, edits, true)?.1
        };
        input.render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }

    /// 只读提取已提交REAPER旧归档的原始组件JSON，不调用当前encoder生成兼容证据。
    fn task34_archived_v2_states() -> Vec<Vec<u8>> {
        use base64::Engine as _;
        let mut chunks = Vec::new();
        let mut states = Vec::new();
        let mut inside = false;
        for line in include_str!("../../tests/fixtures/gui-keyboard-edited.RPP")
            .lines()
            .map(str::trim)
        {
            if line.starts_with("<VST ") {
                inside = true;
                chunks.clear();
                continue;
            }
            if !inside {
                continue;
            }
            if line == ">" {
                let start = chunks
                    .windows(b"{\"edits\":".len())
                    .position(|bytes| bytes == b"{\"edits\":")
                    .unwrap();
                let length =
                    u32::from_le_bytes(chunks[start - 4..start].try_into().unwrap()) as usize;
                states.push(chunks[start..start + length].to_vec());
                inside = false;
            } else {
                chunks.extend(
                    base64::engine::general_purpose::STANDARD
                        .decode(line)
                        .unwrap(),
                );
            }
        }
        assert_eq!(
            states.iter().map(Vec::len).collect::<Vec<_>>(),
            [481, 22516]
        );
        states
    }

    /// 两条归档记录共享完全相同的旧source/modification身份，必须由真实assignment限定恢复目标。
    #[test]
    fn task34_archived_v2_bytes_rebind_only_inside_the_component_scope() {
        let states = task34_archived_v2_states();
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let archived: serde_json::Value = serde_json::from_slice(&states[0]).unwrap();
        let identity: Vec<(String, String)> =
            serde_json::from_value(archived["edits"]["bindings"]["ara-track-0"].clone()).unwrap();
        *document.track_bindings.lock().unwrap() = std::collections::BTreeMap::from([
            ("track".into(), identity.clone()),
            ("b".into(), identity.clone()),
        ]);
        // 与宿主冷重建一致：完整图ready前暂存全部组件原始bytes，随后统一合入权威。
        document.ready.store(false, Ordering::Release);
        for (owner, bytes) in owners.iter().zip(&states) {
            owner.restore_state(bytes).unwrap();
        }
        document.prepare_renderers();
        let accepted = document.edits.lock().unwrap().clone();
        assert!(
            !accepted.params.contains_key("track"),
            "旧A无曲线，不能借旧轨序号取到B曲线"
        );
        assert_eq!(accepted.params["b"].pitch_edit.len(), 800);
        assert_eq!(&accepted.params["b"].pitch_edit[200..212], &[64.; 12]);
        assert!(accepted
            .tracks
            .iter()
            .all(|track| track.id == "track" || track.id == "b"));
        for (index, id) in ["track", "b"].into_iter().enumerate() {
            let bytes = owners[index].encode_state().unwrap();
            let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(saved["version"], if index == 0 { 2 } else { 3 });
            assert_eq!(saved["edits"]["tracks"].as_array().unwrap().len(), 1);
            assert_eq!(saved["edits"]["tracks"][0]["id"], id);
            assert_eq!(saved["edits"]["bindings"].as_object().unwrap().len(), 1);
            assert_eq!(saved["edits"]["params"].as_object().unwrap().len(), index);
        }
        // 真正扩大同一组件assignment，使两个完全相同身份同时进入候选集，必须拒绝合入。
        let key = (&*_ids[1] as *const u8) as u64;
        let raw = owners[0].binding.lock().unwrap().as_ref().unwrap().as_raw();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        let before = document.edits.lock().unwrap().revision;
        owners[0].restore_state(&states[1]).unwrap();
        assert!(owners[0].encode_state().unwrap_err().contains("ambiguous"));
        assert_eq!(document.edits.lock().unwrap().revision, before);
        assert_eq!(
            document.edits.lock().unwrap().params["b"].pitch_edit[200],
            64.
        );
        // 缺少真实身份也不能按归档旧序号恢复；既有曲线留在权威中。
        document.track_bindings.lock().unwrap().insert(
            "track".into(),
            vec![("unknown-mod".into(), "ara://source".into())],
        );
        document.track_bindings.lock().unwrap().insert(
            "b".into(),
            vec![("unknown-b".into(), "ara://source".into())],
        );
        assert!(owners[0].encode_state().unwrap_err().contains("identity"));
        assert_eq!(
            document.edits.lock().unwrap().params["b"].pitch_edit[200],
            64.
        );
        document.close();
    }

    /// 真实自动应用在计算前可接受另一组件/模型修改；旧作业不能覆盖新权威或已撤销输出。
    #[test]
    fn automatic_apply_releases_the_transaction_and_rechecks_model_and_edit_versions() {
        for model_changed in [true, false] {
            let model = crate::ara::model::ModelHandle::new();
            let document = model.session();
            *document.timeline.lock().unwrap() = Some(
                serde_json::from_value(serde_json::json!({
                "tracks":[],"clips":[],"bpm":120,"project_sec":0}))
                .unwrap(),
            );
            document.ready.store(true, Ordering::Release);
            let owner = Arc::new(ExtensionOwner::default());
            owner
                .bind_to_document(
                    document.clone(),
                    ApiGeneration::V2Final,
                    ExtensionRoles::all(),
                    ExtensionRoles::EDITOR_RENDERER,
                    None,
                )
                .unwrap();
            owner.snapshots[0]
                .publish(super::super::snapshot::PlaybackSnapshot {
                    sample_rate: 44100,
                    origin_sample: 0,
                    left: vec![0.25; 4],
                    right: vec![0.5; 4],
                    _reservation: None,
                })
                .unwrap();
            let projection = document.workspace_projection().unwrap();
            let calls = std::sync::atomic::AtomicUsize::new(0);
            let result = document.apply_workspace_edits(
                0,
                document.revision.load(Ordering::Acquire),
                &projection,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                || {
                    if calls.fetch_add(1, Ordering::AcqRel) == 0 {
                        assert!(
                            document.transaction.try_lock().is_ok(),
                            "冻结后必须先释放事务才能计算"
                        );
                        if model_changed {
                            document.clear_renderers();
                        } else {
                            let timeline = document.timeline.lock().unwrap().clone().unwrap();
                            document
                                .accept_workspace_edits(
                                    0,
                                    document.revision.load(Ordering::Acquire),
                                    &timeline,
                                    &projection,
                                )
                                .unwrap();
                        }
                    }
                    true
                },
            );
            assert!(result.is_err());
            assert!(result.unwrap_err().contains(if model_changed {
                "host model changed"
            } else {
                "superseded"
            }));
            assert_eq!(
                document.edits.lock().unwrap().revision,
                if model_changed { 0 } else { 1 }
            );
            let mut left = [9.0; 4];
            let mut right = [9.0; 4];
            let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut bus = crate::audio_abi::AudioBusBuffers {
                num_channels: 2,
                silence_flags: 0,
                channel_buffers: planes.as_mut_ptr(),
            };
            // SAFETY: 两个平面都是四帧；只检查真实publisher，不能由计算出的expected镜像掩盖覆盖。
            let available = unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) };
            assert_eq!(available, !model_changed);
            assert_eq!(left, if model_changed { [0.0; 4] } else { [0.25; 4] });
            drop(model);
        }
    }

    /// 即使其它线程暂持模型事务，真实prepare callback也必须返回，并在后台最终发布。
    #[test]
    fn prepare_callback_does_not_wait_for_a_model_transaction() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        *document.timeline.lock().unwrap() = Some(
            serde_json::from_value(serde_json::json!({
            "tracks":[],"clips":[],"bpm":120,"project_sec":0}))
            .unwrap(),
        );
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let held = document.transaction.lock().unwrap();
        let (sent, received) = std::sync::mpsc::channel();
        let requesting = owner.clone();
        let caller = std::thread::spawn(move || {
            requesting.prepare();
            sent.send(()).unwrap();
        });
        let returned = received
            .recv_timeout(std::time::Duration::from_secs(3))
            .is_ok();
        drop(held);
        caller.join().unwrap();
        assert!(returned, "宿主prepare不得等待文档事务/合成");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while owner.preparation_state().0 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(owner.preparation_state().1.is_none());
        let mut left = [9.0; 4];
        let mut right = [9.0; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 四帧平面存活；验证worker真的发布空分配快照，不接受“只是忽略prepare”。
        assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
        assert_eq!(left, [0.0; 4]);
        drop(model);
    }

    /// 冷恢复在运行任务被阻塞时也要先合入全部组件状态，不能由各worker逐轨推进revision。
    #[test]
    fn cold_restores_are_merged_document_wide_before_background_jobs_start() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let mut timeline:hifishifter_kernel::state::TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0},{"id":"b","name":"B","order":1}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":4.0/44100.0,"takes":[{"id":"ta","name":"A","source_path":"source","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"cb","name":"B","track_id":"b","start_sec":0,"length_sec":4.0/44100.0,"takes":[{"id":"tb","name":"B","source_path":"source","source_start_sec":0,"source_end_sec":4.0/44100.0}]}]
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock().unwrap() = Some(timeline);
        let pcm = Arc::new(super::super::source::SourcePcm {
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        });
        document
            .edit_sources
            .lock()
            .unwrap()
            .insert("source".into(), pcm.clone());
        document
            .sources
            .lock()
            .unwrap()
            .insert("source".into(), pcm);
        *document.track_bindings.lock().unwrap() = std::collections::BTreeMap::from([
            ("a".into(), vec![("ma".into(), "source".into())]),
            ("b".into(), vec![("mb".into(), "source".into())]),
        ]);
        let identities = [Box::new(0_u8), Box::new(0_u8)];
        let mut owners = Vec::new();
        let mut gates = Vec::new();
        for (index, clip) in ["ca", "cb"].into_iter().enumerate() {
            let key = (&*identities[index] as *const u8) as u64;
            region_owners()
                .lock()
                .unwrap()
                .register(key, document.id, index)
                .unwrap();
            document.clip_ids.lock().unwrap().insert(key, clip.into());
            document.regions.lock().unwrap().insert(
                key,
                crate::ara::AraPlaybackRegion {
                    audio_source_persistent_id: "source".into(),
                    duration_in_modification_time: 4.0 / 44100.0,
                    duration_in_playback_time: 4.0 / 44100.0,
                    ..Default::default()
                },
            );
            let owner = Arc::new(ExtensionOwner::default());
            let raw = owner
                .bind_to_document(
                    document.clone(),
                    ApiGeneration::V2Final,
                    ExtensionRoles::all(),
                    ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                    None,
                )
                .unwrap();
            // SAFETY: identities/raw extension保留到doc销毁；这是实际宿主assignment入口。
            unsafe {
                let ext = &*raw;
                ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                    ext.playbackRendererRef,
                    key as *mut _,
                );
            }
            let (release, gate) = std::sync::mpsc::channel();
            let (started, entered) = std::sync::mpsc::channel();
            owner
                .preparation
                .get()
                .unwrap()
                .as_ref()
                .unwrap()
                .request(Box::new(move |_| {
                    started.send(()).unwrap();
                    gate.recv_timeout(std::time::Duration::from_secs(3))
                        .unwrap();
                    Ok(())
                }))
                .unwrap();
            entered
                .recv_timeout(std::time::Duration::from_secs(3))
                .unwrap();
            gates.push(release);
            let host = owner.assigned_timeline(&document).unwrap();
            let mut client = host.clone();
            client.tracks[0].volume = if index == 0 { 0.5 } else { 0.25 };
            let mut saved = crate::state_channel::EditState::default()
                .merge(&host, &client, 0)
                .unwrap();
            let id = if index == 0 { "a" } else { "b" };
            saved
                .reconcile(&std::collections::BTreeMap::from([(
                    id.into(),
                    document.track_bindings.lock().unwrap()[id].clone(),
                )]))
                .unwrap();
            owner.restore_state(&saved.encode().unwrap()).unwrap();
            owners.push(owner);
        }
        document.prepare_renderers();
        let accepted = document.edits.lock().unwrap().clone();
        // 先释放测试屏障再断言，旧实现失败也不能把真实worker留在无限等待里。
        for gate in gates {
            gate.send(()).unwrap();
        }
        assert_eq!(
            accepted
                .tracks
                .iter()
                .find(|t| t.id == "a")
                .map(|t| t.volume),
            Some(0.5)
        );
        assert_eq!(
            accepted
                .tracks
                .iter()
                .find(|t| t.id == "b")
                .map(|t| t.volume),
            Some(0.25)
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        for (index, owner) in owners.iter().enumerate() {
            while owner.preparation_state().0 {
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
            assert!(owner.preparation_state().1.is_none());
            let mut left = [9.0; 4];
            let mut right = [9.0; 4];
            let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut bus = crate::audio_abi::AudioBusBuffers {
                num_channels: 2,
                silence_flags: 0,
                channel_buffers: planes.as_mut_ptr(),
            };
            // SAFETY: 两个四帧平面；保证无GUI冷恢复确实供音，而非只合入了state。
            assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
            for (actual, expected) in left.iter().zip(if index == 0 {
                [0.05, 0.1, 0.15, 0.2]
            } else {
                [0.025, 0.05, 0.075, 0.1]
            }) {
                assert!((*actual - expected).abs() < 1e-6);
            }
        }
        drop(model);
    }

    /// 实际assignment observer必须与发布使用同一事务；不能校验后移除、再发布旧区域。
    #[test]
    fn native_assignment_observer_serializes_with_snapshot_publication_transaction() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        let (ready, entered) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let blocking = document.clone();
        let holder = std::thread::spawn(move || {
            let _held = blocking.transaction.lock().unwrap();
            ready.send(()).unwrap();
            let _ = released.recv_timeout(std::time::Duration::from_millis(300));
        });
        entered
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        let began = std::time::Instant::now();
        // SAFETY: 在绑定时的真实model线程驱动API；其它线程只持Rust事务，不能绕过桥接线程检查。
        unsafe {
            let ext = &*raw;
            ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
                ext.editorRendererRef,
                key as *mut _,
            );
        }
        let elapsed = began.elapsed();
        let _ = release.send(());
        holder.join().unwrap();
        assert!(
            elapsed >= std::time::Duration::from_millis(150),
            "assignment不得绕开发布的文档事务"
        );
        assert_eq!(owner.assignments.lock().unwrap()[&2], [key]);
        drop(model);
    }

    #[test]
    fn gui_commit_changes_assigned_pcm_and_persisted_state_restores_it() {
        use crate::render::source::SourcePcm;
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let mut timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"one","track_id":"track","name":"one","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","name":"one","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"other","track_id":"track","name":"other","start_sec":1,"length_sec":4.0/44100.0,
                "takes":[{"id":"take2","name":"other","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":2
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock().unwrap() = Some(timeline);
        document.track_bindings.lock().unwrap().insert(
            "track".into(),
            vec![("host-mod".into(), "ara://pcm".into())],
        );
        document.edit_sources.lock().unwrap().insert(
            "ara://pcm".into(),
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
                version: 0,
                _reservation: None,
            }),
        );
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        document.clip_ids.lock().unwrap().insert(key, "one".into());
        document.regions.lock().unwrap().insert(
            key,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "ara://pcm".into(),
                audio_modification_persistent_id: "host-mod".into(),
                duration_in_modification_time: 4.0 / 44100.0,
                duration_in_playback_time: 4.0 / 44100.0,
                ..Default::default()
            },
        );
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        let snapshot = owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let mut client = snapshot.timeline.unwrap();
        assert_eq!(client["clips"].as_array().unwrap().len(), 1);
        client["tracks"][0]["volume"] = serde_json::json!(0.5);
        let response = owner.handle_request(hifishifter_ara_ipc::Request::Commit {
            base_revision: snapshot.revision,
            model_revision: snapshot.model_revision,
            timeline: client,
        });
        assert!(response.ok, "{:?}", response.error);
        let companion = Arc::new(ExtensionOwner::default());
        companion
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        assert_eq!(
            companion.edit_state().lock().unwrap().revision,
            response.revision
        );
        assert_eq!(companion.edit_state().lock().unwrap().tracks[0].volume, 0.5);
        let state = owner.edit_state().lock().unwrap().encode().unwrap();
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&state).unwrap();
        let output = render_test_edits(&owner, &document, &restored).unwrap();
        assert_eq!(output[0].left.len(), 4);
        for (actual, expected) in output[0].left.iter().zip([0.05_f32, 0.1, 0.15, 0.2]) {
            assert!((*actual - expected).abs() < 1e-6);
        }
        assert!(restored.restore(b"bad state").is_err());
        drop(model);
    }

    #[test]
    fn gui_request_commits_parameters_and_rejects_stale_document_revision() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let timeline: hifishifter_kernel::state::TimelineState =
            serde_json::from_value(serde_json::json!({
                "tracks":[],"clips":[],"bpm":120,"project_sec":0
            }))
            .unwrap();
        *document.timeline.lock().unwrap() = Some(timeline);
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let snapshot = owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let client = snapshot.timeline.unwrap();
        let commit = hifishifter_ara_ipc::Request::Commit {
            base_revision: snapshot.revision,
            model_revision: snapshot.model_revision,
            timeline: client,
        };
        let response = owner.handle_request(commit.clone());
        assert!(response.ok, "{:?}", response.error);
        assert_eq!(response.revision, 1);
        assert!(!owner.handle_request(commit).ok);
        document.clear_renderers();
        assert!(
            !owner
                .handle_request(hifishifter_ara_ipc::Request::Snapshot)
                .ok
        );
        drop(model);
    }

    /// 两个renderer先后提交局部视图，第二次刷新后不能抹掉第一轨PCM或持久曲线。
    #[test]
    fn two_renderer_commits_keep_both_curves_volumes_pcm_and_saved_state() {
        two_renderer_case(false);
    }
    /// REAPER复制轨道共享modification/source；状态与恢复仍必须限制在宿主实例分配内。
    #[test]
    fn copied_tracks_share_sources_but_keep_instance_state_isolated() {
        two_renderer_case(true);
    }
    fn two_renderer_case(shared_identity: bool) {
        use crate::render::source::SourcePcm;
        use hifishifter_ara_ipc::Request;
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        // 四样本只验证复制/权限/保存与逐轨gain，显式旁路；真实音高自动应用另有WORLD/模型oracle。
        let mut timeline:hifishifter_kernel::state::TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0,"pitch_analysis_algo":"none"},{"id":"b","name":"B","order":1,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":1,
            "clips":[{"id":"clip-a","track_id":"a","name":"A","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take-a","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"clip-b","track_id":"b","name":"B","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take-b","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}]
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock().unwrap() = Some(timeline);
        document.edit_sources.lock().unwrap().insert(
            "ara://pcm".into(),
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
                version: 0,
                _reservation: None,
            }),
        );
        let bindings = std::collections::BTreeMap::from([
            ("a".into(), vec![("mod-a".into(), "ara://pcm".into())]),
            (
                "b".into(),
                vec![(
                    if shared_identity { "mod-a" } else { "mod-b" }.into(),
                    "ara://pcm".into(),
                )],
            ),
        ]);
        *document.track_bindings.lock().unwrap() = bindings.clone();
        document.ready.store(true, Ordering::Release);
        let identities = [Box::new(0_u8), Box::new(0_u8)];
        let mut owners = Vec::new();
        for (index, id) in ["clip-a", "clip-b"].into_iter().enumerate() {
            let key = (&*identities[index] as *const u8) as u64;
            region_owners()
                .lock()
                .unwrap()
                .register(key, document.id, index)
                .unwrap();
            document.clip_ids.lock().unwrap().insert(key, id.into());
            document.regions.lock().unwrap().insert(
                key,
                crate::ara::AraPlaybackRegion {
                    audio_source_persistent_id: "ara://pcm".into(),
                    audio_modification_persistent_id: if index == 0 || shared_identity {
                        "mod-a"
                    } else {
                        "mod-b"
                    }
                    .into(),
                    duration_in_modification_time: 4.0 / 44100.0,
                    duration_in_playback_time: 4.0 / 44100.0,
                    ..Default::default()
                },
            );
            let owner = Arc::new(ExtensionOwner::default());
            let raw = owner
                .bind_to_document(
                    document.clone(),
                    ApiGeneration::V2Final,
                    ExtensionRoles::all(),
                    ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                    None,
                )
                .unwrap();
            // SAFETY: identities及native扩展都保留到owner/document销毁。
            unsafe {
                let ext = &*raw;
                ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                    ext.playbackRendererRef,
                    key as *mut _,
                );
            }
            owners.push(owner);
        }
        for (index, id) in ["a", "b"].into_iter().enumerate() {
            let snapshot = owners[index].handle_request(Request::Snapshot);
            assert!(snapshot.ok, "{:?}", snapshot.error);
            assert_eq!(snapshot.revision, index as u64, "B先刷新最新共享revision");
            let mut client = snapshot.timeline.unwrap();
            client["tracks"][0]["volume"] = serde_json::json!(if index == 0 { 0.5 } else { 0.25 });
            client["params_by_root_track"] = serde_json::json!({id:{"frame_period_ms":5.0,"pitch_edit":[61.0+index as f32,63.0+index as f32]}});
            let response = owners[index].handle_request(Request::Commit {
                base_revision: snapshot.revision,
                model_revision: snapshot.model_revision,
                timeline: client,
            });
            assert!(response.ok, "{:?}", response.error);
        }
        for (index, id) in ["a", "b"].into_iter().enumerate() {
            let saved: serde_json::Value =
                serde_json::from_slice(&owners[index].encode_state().unwrap()).unwrap();
            assert_eq!(
                saved["edits"]["params"].as_object().unwrap().len(),
                1,
                "单个组件不能保存整个ARA文档的其它轨道"
            );
            assert!(saved["edits"]["params"].get(id).is_some());
        }
        let mut restored = document.edits.lock().unwrap().clone();
        restored.reconcile(&bindings).unwrap();
        for (index, id) in ["a", "b"].into_iter().enumerate() {
            assert_eq!(
                restored.params[id].pitch_edit,
                [61.0 + index as f32, 63.0 + index as f32]
            );
            let output = render_test_edits(&owners[index], &document, &restored).unwrap();
            let factor = if index == 0 { 0.5 } else { 0.25 };
            for (actual, original) in output[0].left.iter().zip([0.1_f32, 0.2, 0.3, 0.4]) {
                assert!((*actual - original * factor).abs() < 1e-6);
            }
            let current = owners[index].handle_request(Request::Snapshot);
            assert!(current.ok);
            assert_eq!(
                current.timeline.unwrap()["tracks"][0]["volume"],
                serde_json::json!(factor)
            );
        }
        if shared_identity {
            let original = owners[0].encode_state().unwrap();
            let second = owners[1].encode_state().unwrap();
            // REAPER复制插件：同一源身份由另一实例的实际assignment限定到B，不能改写A。
            owners[1].restore_state(&original).unwrap();
            let a = owners[0].handle_request(Request::Snapshot);
            let b = owners[1].handle_request(Request::Snapshot);
            assert!(a.ok && b.ok, "{:?} {:?}", a.error, b.error);
            assert_eq!(a.timeline.unwrap()["tracks"][0]["volume"], 0.5);
            assert_eq!(b.timeline.unwrap()["tracks"][0]["volume"], 0.5);
            assert_eq!(
                document.edits.lock().unwrap().params["b"].pitch_edit,
                [61., 63.]
            );
            owners[1].restore_state(&second).unwrap();
            assert_eq!(
                document.edits.lock().unwrap().params["a"].pitch_edit,
                [61., 63.]
            );
            assert_eq!(
                document.edits.lock().unwrap().params["b"].pitch_edit,
                [62., 64.]
            );
            let editors: Vec<_> = owners
                .iter()
                .map(|owner| owner.editor_session().unwrap())
                .collect();
            let call = |editor: &Arc<crate::editor::session::EditorSession>,
                        command: &str,
                        args: serde_json::Value| {
                let (reply, received) = std::sync::mpsc::channel();
                let (events, _) = std::sync::mpsc::sync_channel(128);
                let sink = crate::editor::session::UiSink {
                    view_id: "dual-actor".into(),
                    reply,
                    events,
                    closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                };
                editor
                    .enqueue(crate::editor::session::UiRequest {
                        id: 1,
                        command: command.into(),
                        args,
                        sink,
                        link: None,
                    })
                    .unwrap();
                let response = received
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                assert_eq!(response["ok"], true, "{response}");
                response["value"].clone()
            };
            for editor in &editors {
                call(editor, "get_timeline_state", serde_json::json!({}));
            }
            // 同一次debounce窗口内两轨落笔；B的共享revision变化不能让A的最新作业Conflict。
            for (index, editor) in editors.iter().enumerate() {
                let timeline = call(editor, "get_timeline_state", serde_json::json!({}));
                call(
                    editor,
                    "set_track_state",
                    serde_json::json!({
                        "trackId":timeline["tracks"][index]["id"],"volume":if index==0 {0.75} else {0.125}
                    }),
                );
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            for editor in &editors {
                loop {
                    let state = call(editor, "plugin_get_apply_state", serde_json::json!({}));
                    assert!(state["error"].is_null(), "{state}");
                    if state["pending"] == false {
                        break;
                    }
                    assert!(std::time::Instant::now() < deadline, "{state}");
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
            editors[0].close(); // 现在两个真实入口共享actor，只在全部回归请求完成后关闭。
            assert_eq!(
                document
                    .edits
                    .lock()
                    .unwrap()
                    .tracks
                    .iter()
                    .find(|t| t.id == "a")
                    .unwrap()
                    .volume,
                0.75
            );
            assert_eq!(
                document
                    .edits
                    .lock()
                    .unwrap()
                    .tracks
                    .iter()
                    .find(|t| t.id == "b")
                    .unwrap()
                    .volume,
                0.125
            );
        }
    }

    /// 真实空文档绑定也必须收到销毁；不依赖 first region assignment 猜 document。
    #[test]
    fn bound_native_entry_is_tombstoned_by_actual_document_teardown() {
        let factory = Box::leak(Box::new(
            FactoryBuilder::new("org.hfs.bound", "org.hfs.bound.archive")
                .display("bound", "HiFiShifter", "https://example.invalid", "1")
                .document_controller(|| {
                    let model = crate::ara::model::ModelHandle::new();
                    let session = model.session();
                    PluginBuilder::new(model)
                        .controller_identity(move |key| session.register(key))
                        .build()
                })
                .build()
                .unwrap(),
        ));
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"bound empty".as_ptr(),
        };
        // SAFETY: test host、factory 和 properties 保持到实际 controller 终止。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 原生工厂返回完整 packed 实例。
        let controller = unsafe { raw.read_unaligned() };
        let document = crate::render::document::DocumentSession::lookup(
            controller.documentControllerRef as usize,
        )
        .unwrap();
        // SAFETY: test factory backing 已保留到进程结束。
        let association =
            unsafe { CompanionFactory::from_raw("bound", &*factory.as_raw()) }.unwrap();
        let processor =
            CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
        let probe = processor.lifetime_probe();
        let owner = Arc::new(ExtensionOwner::default());
        let entry = HostEntry::new(processor, "bound", owner.clone()).unwrap();
        let mut extension_raw = std::ptr::null();
        // SAFETY: real C++ shim 调真实产品 bind，controller 存活且已登记。
        assert_eq!(
            unsafe {
                ara2_vst3_plugin_entry_bind(
                    entry.as_raw(),
                    controller.documentControllerRef.cast(),
                    7,
                    1,
                    1,
                    &raw mut extension_raw,
                )
            },
            ARA2_VST3_OK
        );
        assert!(!extension_raw.is_null());
        assert!(probe.controller_alive());
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        // SAFETY: extension 与 controller 保留；region 仅作为不透明身份。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert_eq!(owner.assignments.lock().unwrap().get(&1).unwrap(), &[key]);
        // SAFETY: actual controller destructor invokes ModelHandle::destroy_document。
        unsafe {
            ((*controller.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(controller.documentControllerRef)
        };
        assert!(owner.assignments.lock().unwrap().is_empty());
        assert!(owner.assigned_regions().is_err());
        assert!(!probe.controller_alive());
        // SAFETY: native entry 仍保留 companion storage；销毁文档后回调必须被 tombstone 拒绝。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert!(owner.assignments.lock().unwrap().is_empty());
        drop(entry);
        drop(owner);
        factory.entry().uninitialize().unwrap();
    }

    /// sequence 预览只展开本 sequence；显式区域重复分配不能导致双倍混音。
    #[test]
    fn editor_sequence_selection_is_expanded_and_deduplicated() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let region_a = Box::new(0_u8);
        let region_b = Box::new(0_u8);
        let sequence = Box::new(0_u8);
        let a = (&*region_a as *const u8) as u64;
        let b = (&*region_b as *const u8) as u64;
        let seq = (&*sequence as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(a, document.id, 0)
            .unwrap();
        region_owners()
            .lock()
            .unwrap()
            .register(b, document.id, 1)
            .unwrap();
        document
            .sequence_regions
            .lock()
            .unwrap()
            .insert(seq, [a, b].into_iter().collect());
        // SAFETY: binding、session 和不透明身份在调用期间均存活。
        unsafe {
            let extension = &*raw;
            let api = &*extension.editorRendererInterface;
            api.addPlaybackRegion.unwrap()(extension.editorRendererRef, a as *mut _);
            api.addRegionSequence.unwrap()(extension.editorRendererRef, seq as *mut _);
        }
        let mut expected = vec![a, b];
        expected.sort_unstable();
        assert_eq!(owner.assigned_regions().unwrap(), expected);
        document
            .sequence_regions
            .lock()
            .unwrap()
            .get_mut(&seq)
            .unwrap()
            .remove(&b);
        region_owners().lock().unwrap().remove(b);
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        // SAFETY: lease 保留接口，remove sequence 不会移除显式 region。
        unsafe {
            let extension = &*raw;
            ((*extension.editorRendererInterface)
                .removeRegionSequence
                .unwrap())(extension.editorRendererRef, seq as *mut _);
        }
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        drop(model);
        assert!(owner.assigned_regions().is_err());
    }

    /// companion 先销毁时 controller lease 必须独立保留真实 raw storage。
    #[test]
    fn actual_document_retains_bound_storage_after_owner_is_dropped() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document,
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .unwrap();
        drop(owner);
        let mut identity = 0_u8;
        // SAFETY: actual model 的 session 独立持有控制器 lease，owner 已 drop 也不释放 storage。
        unsafe {
            let extension = raw.read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
            ((*extension.playbackRendererInterface)
                .removePlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
        }
        drop(model);
        // 模型终止后所有 owner 都已释放，此后不得再访问 raw。
    }
}

impl ExtensionOwner {
    /// 组件终止独立于Arc尚存与否，旧native entry不能维持可编辑授权。
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    pub(crate) fn editor_document(&self) -> Result<Arc<super::document::DocumentSession>, String> {
        if self.is_closed() {
            return Err("FX processor closed".into());
        }
        self.document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .filter(|document| document.is_alive())
            .ok_or_else(|| "document closed".into())
    }
    /// 初始化在宿主UI/model线程保存有引用的typed扩展，不在actor或process调用REAPER API。
    /// # Safety
    /// context须为宿主初始化期间活FUnknown。
    pub(crate) unsafe fn bind_reaper_host(&self, context: *mut std::ffi::c_void) {
        let bound = self.document.lock().unwrap().is_some();
        let stamp = if bound {
            match self.host_query_stamp() {
                Ok(stamp) => Some(stamp),
                Err(_) => return,
            }
        } else {
            None
        };
        let authorized = || {
            !self.is_closed()
                && match &stamp {
                    Some(stamp) => self.host_query_authorized(stamp),
                    None => self.document.lock().unwrap().is_none(),
                }
        };
        let host = unsafe { crate::host::reaper::ReaperHost::from_context(context, authorized) }
            .map(Arc::new);
        if !authorized() {
            return;
        }
        crate::log_line(&format!(
            "REAPER host extension available={}",
            host.is_some()
        ));
        let old = std::mem::replace(&mut *self.reaper.lock().unwrap(), host);
        drop(old);
        self.host_geometry.lock().unwrap().take();
    }
    /// 短事务冻结真实owner/document/model/scope；整个host调用链都不持内部锁。
    fn host_query_stamp(
        &self,
    ) -> Result<(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>), String> {
        let document = self.editor_document()?;
        let _transaction = document.transaction.lock().unwrap();
        if self.is_closed() || !document.is_alive() {
            return Err("host query document/owner closed".into());
        }
        let keys = self.host_assigned_regions(&document)?;
        let model = document.revision.load(Ordering::Acquire);
        let scope = document.scope_revision.load(Ordering::Acquire);
        drop(_transaction);
        Ok((document, model, scope, keys))
    }
    fn host_query_authorized(
        &self,
        stamp: &(Arc<super::document::DocumentSession>, u64, u64, Vec<u64>),
    ) -> bool {
        let (document, model, scope, keys) = stamp;
        let _transaction = document.transaction.lock().unwrap();
        !self.is_closed()
            && document.is_alive()
            && self
                .editor_document()
                .is_ok_and(|current| Arc::ptr_eq(&current, document))
            && document.revision.load(Ordering::Acquire) == *model
            && document.scope_revision.load(Ordering::Acquire) == *scope
            && self
                .host_assigned_regions(document)
                .is_ok_and(|current| current == *keys)
    }
    /// 几何绑定必须覆盖owner的全部已分配角色，不能只拿playback子集隐藏editor歧义。
    fn host_assigned_regions(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<Vec<u64>, String> {
        let mut keys = self
            .assignments
            .lock()
            .unwrap()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock()
            .unwrap()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let members = document.sequence_regions.lock().unwrap();
        for sequence in sequences {
            keys.extend(
                members
                    .get(&sequence)
                    .ok_or("unknown host assigned sequence")?
                    .iter()
                    .copied(),
            );
        }
        drop(members);
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty()
            && region_owners()
                .lock()
                .unwrap()
                .resolve(&keys)
                .map_err(|_| "invalid host assigned region")?
                .0
                != document.id
        {
            return Err("host assigned region belongs to another document".into());
        }
        Ok(keys)
    }
    /// 仅UI/model线程读取；唯一assignment先成立，位置/长度仅用于绑定后相容核对。
    ///
    /// 这条路径的身份来自宿主直接 parent take —— 是强证据，因此**只在恰好一个
    /// region 时**成立。多 region（folder 轨上的 FX）走
    /// [`Self::reaper_geometries`]，那里没有直接 parent 可用，只能按几何唯一匹配。
    pub(crate) fn reaper_geometry(
        &self,
    ) -> Result<crate::host::geometry::BoundHostGeometry, String> {
        let stamp = self.host_query_stamp()?;
        let [region_key] = stamp.3.as_slice() else {
            return Err("REAPER geometry requires exactly one assigned ARA region".into());
        };
        let region = stamp
            .0
            .regions
            .lock()
            .unwrap()
            .get(region_key)
            .cloned()
            .ok_or("assigned ARA region unavailable")?;
        let host = self
            .reaper
            .lock()
            .unwrap()
            .clone()
            .ok_or("REAPER host extension unavailable")?;
        let geometry = host.geometry(|| self.host_query_authorized(&stamp))?;
        if !compatible(geometry.start_sec, region.start_in_playback_time)
            || !compatible(geometry.duration_sec, region.duration_in_playback_time)
        {
            return Err(
                "direct take geometry incompatible with assigned ARA playback window".into(),
            );
        }
        if !self.host_query_authorized(&stamp) {
            return Err("host geometry authorization revoked".into());
        }
        Ok(crate::host::geometry::BoundHostGeometry {
            region_key: *region_key,
            geometry,
        })
    }

    /// 多 region owner（挂在 REAPER folder 轨上的 FX）逐 region 解析宿主几何。
    ///
    /// 【为什么不能退回"取第一个"】后续所有宿主写回都沿 region → item 这条边定位
    /// 对象。拿一个 take 的几何去冒充整组，会把 item GUID 绑到**错误的** region 上，
    /// 于是移动/切分/fade/音量全部可能落到别的 item。因此这里对每个 region 单独
    /// 求唯一匹配，匹配不到就**不绑定**，并在日志里如实报出未绑定的数量。
    ///
    /// 候选 item 与 GUI 清单同源（FX 自己的轨道 + folder 全部后代轨道），沿既有
    /// `ui_folder_tracks` 的线程/授权/预算纪律取得。
    fn reaper_geometries(&self) -> Result<Vec<crate::host::geometry::BoundHostGeometry>, String> {
        let stamp = self.host_query_stamp()?;
        if stamp.3.is_empty() {
            return Err("no assigned ARA region".into());
        }
        let host = self
            .reaper
            .lock()
            .unwrap()
            .clone()
            .ok_or("REAPER host extension unavailable")?;
        let authorized = || self.host_query_authorized(&stamp);
        let candidates = host
            .ui_folder_tracks(&authorized)?
            .into_iter()
            .flat_map(|track| track.items.into_iter().map(|item| item.geometry))
            .collect::<Vec<_>>();
        let mut bound = Vec::new();
        let mut unmatched = Vec::new();
        for key in &stamp.3 {
            let region = stamp
                .0
                .regions
                .lock()
                .unwrap()
                .get(key)
                .cloned()
                .ok_or("assigned ARA region unavailable")?;
            // 上一次绑定：给几何全等的候选（复制出来的安全副本等）一个稳定性偏好。
            let preferred = stamp.0.region_items.lock().unwrap().get(key).cloned();
            match unique_geometry_for_region(&region, &candidates, preferred.as_deref()) {
                Some(geometry) => bound.push(crate::host::geometry::BoundHostGeometry {
                    region_key: *key,
                    geometry,
                }),
                None => unmatched.push(*key),
            }
        }
        // 一个宿主 item 只能被**一个** region 认领。两条 region 绑到同一个 item 会让
        // 后续所有写回指向同一对象：移动、切分、fade、音量互相覆盖，且没有任何一侧
        // 会报错。
        //
        // 【为什么不再"全部丢弃"】旧实现把重复认领的候选**全丢**，于是"同一位置两个
        // 几何全等的 item"会两个都不绑定、两个 clip 一起消失（用户报障："片段凭空
        // 没了"）。现在按 region key 的**稳定顺序**保留第一个认领者，其余照常记账 ——
        // 至少不会一次丢掉全部；剩下的那个 region 仍走 unmatched，如实报出。
        let mut seen = std::collections::HashSet::<String>::new();
        let claimed = bound.len();
        bound.retain(|entry| seen.insert(entry.geometry.item_id.clone()));
        if bound.len() != claimed {
            log::warn!(
                "[ara] {} region(s) dropped: they claimed an item another region already owns",
                claimed - bound.len()
            );
        }
        if stamp.3.len() > 1 {
            // 用户报障时最需要的一行：这个实例是不是"一个 FX 管一组 region"。
            log::info!(
                "[ara] multi-region owner: {} assigned region(s), {} bound to a unique host \
                 item, {} candidate item(s)",
                stamp.3.len(),
                bound.len(),
                candidates.len()
            );
        }
        if !unmatched.is_empty() {
            // 认领不到的 region 会停在 `ara-clip-N`，随后被清单侧 retain 剔除 ——
            // 这正是"子轨道 Item 没有变成真 Clip"的直接读数。
            log::warn!(
                "[ara] {} of {} assigned region(s) have no unique host item (candidates={}); \
                 they stay unbound",
                unmatched.len(),
                stamp.3.len(),
                candidates.len()
            );
        }
        if bound.is_empty() {
            return Err("no assigned ARA region matched a unique host item".into());
        }
        if !self.host_query_authorized(&stamp) {
            return Err("host geometry authorization revoked".into());
        }
        Ok(bound)
    }
    /// actor/worker只可消费UI冻结的Rust值，不沿此访问器调用host；代次变化明确不可用。
    // 只读几何元数据访问器保留，供worker消费冻结值。
    #[allow(dead_code)]
    pub(crate) fn host_geometry_metadata(
        &self,
    ) -> Result<Vec<crate::host::geometry::BoundHostGeometry>, String> {
        let stamp = self.host_query_stamp()?;
        let cached = self
            .host_geometry
            .lock()
            .unwrap()
            .clone()
            .ok_or("REAPER geometry has not been sampled on model/UI thread")?;
        if cached.model != stamp.1 || cached.scope != stamp.2 || !self.host_query_authorized(&stamp)
        {
            return Err("REAPER geometry metadata superseded".into());
        }
        cached.value
    }
    /// 调用者已持所属doc事务，只消费已授权Rust缓存，不嵌套加锁或调用宿主。
    ///
    /// 返回**当前仍被分配**的 region 各自的绑定；不再要求"恰好一个" —— folder 轨上
    /// 的 FX 会被分配整组 region，要求唯一等于让整条投影链（fade / mute / item 身份）
    /// 全部静默失效。仍然只返回 region 与缓存都同意的那部分：缓存过期、region 已撤销、
    /// 或该 region 没匹配到唯一 item 的，一律不出现。
    pub(crate) fn host_geometries_locked(
        &self,
        document: &super::document::DocumentSession,
    ) -> Vec<crate::host::geometry::BoundHostGeometry> {
        if self.is_closed()
            || !document.is_alive()
            || !self
                .editor_document()
                .is_ok_and(|current| std::ptr::eq(Arc::as_ptr(&current), document))
        {
            return Vec::new();
        }
        let Some(cached) = self.host_geometry.lock().unwrap().clone() else {
            return Vec::new();
        };
        if cached.model != document.revision.load(Ordering::Acquire)
            || cached.scope != document.scope_revision.load(Ordering::Acquire)
        {
            return Vec::new();
        }
        let Ok(keys) = self.host_assigned_regions(document) else {
            return Vec::new();
        };
        match cached.value {
            Ok(bound) => bound
                .into_iter()
                .filter(|entry| keys.contains(&entry.region_key))
                .collect(),
            Err(_) => Vec::new(),
        }
    }
    /// 最近一次清单刷新得出的宿主音频读数；GUI 载荷读它。
    ///
    /// 只读快照，不触发宿主调用 —— 与 `host_geometries_locked` 同一纪律：调用方已经
    /// 在事务里，这里不得再枚举宿主。
    pub(crate) fn host_audio_status(&self) -> HostAudioStatus {
        *self.host_audio.lock().unwrap()
    }
    /// 完整刷新：时钟 + 几何 + 清单。写回路径与测试用这个（必须**同步**生效）。
    pub(crate) fn refresh_reaper_transport(&self) {
        self.refresh_reaper_transport_inner(false);
    }

    /// UI 定时器路径：时钟每次跟，几何/清单按 [`HOST_ENUMERATION_INTERVAL`] 节流。
    ///
    /// 【为什么时钟与枚举分开节流】定时器是 20ms（50Hz）。播放光标必须跟得上
    /// （`host.sample` 很便宜），但几何/清单枚举即使有代次缓存，每 tick 仍要付
    /// `GetProjectStateChangeCount` 等若干宿主调用，还会与 actor/渲染线程争同一把
    /// document 事务。宿主真变了时下一次节流点会捕捉到（≤100ms），用户不可感。
    ///
    /// 【为什么不把节流放进 `refresh_reaper_transport` 本身】写回路径（导入/分割/
    /// 剪贴板）与测试依赖它**同步**把新几何取回来；节流在那里会把"刚写完就读到新值"
    /// 变成"下次再看"。所以节流只属于定时器这一个调用方。
    pub(crate) fn refresh_reaper_transport_tick(&self) {
        self.refresh_reaper_transport_inner(true);
    }

    fn refresh_reaper_transport_inner(&self, throttled: bool) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let host = { self.reaper.lock().unwrap().clone() };
        if let Some(host) = host {
            if let Ok((position, playing)) = host.sample(|| self.host_query_authorized(&stamp)) {
                // getter可能重入关闭/改scope，发布与撤销共用最终短事务。
                let _transaction = stamp.0.transaction.lock().unwrap();
                if !self.is_closed()
                    && stamp.0.is_alive()
                    && stamp.0.revision.load(Ordering::Acquire) == stamp.1
                    && stamp.0.scope_revision.load(Ordering::Acquire) == stamp.2
                {
                    if let Some(clock) = self.clock.get() {
                        clock.publish_host(position, playing);
                    }
                }
            }
        }
        if !self.host_query_authorized(&stamp) {
            return;
        }
        if throttled {
            let mut last = self.last_enumeration.lock().unwrap();
            let now = std::time::Instant::now();
            if last.is_some_and(|last| now.duration_since(last) < HOST_ENUMERATION_INTERVAL) {
                return;
            }
            *last = Some(now);
        }
        self.refresh_reaper_geometry();
        // editor-only通常没有所属take；一个工程GUI必须驱动真正隐藏playback实例的只读采集。
        // renderer_owners仅返回本真实文档的活租约，不扫描全局实例，也不借名称/位置找take。
        for owner in stamp.0.renderer_owners() {
            if !self.host_query_authorized(&stamp) {
                break;
            }
            if owner.renders_playback() && !std::ptr::eq(self, Arc::as_ptr(&owner)) {
                owner.refresh_reaper_geometry();
            }
        }
        self.refresh_ui_inventory();
    }
    /// GUI清单按项目change/model/scope代次刷新，不每20ms枚举所有item，也不借活动工程。
    pub(crate) fn refresh_ui_inventory(&self) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let Some(host) = self.reaper.lock().unwrap().clone() else {
            return;
        };
        let allowed = || self.host_query_authorized(&stamp);
        let Ok(change) = host.geometry_revision(allowed) else {
            return;
        };
        let token = (change, stamp.1, stamp.2);
        if self.host_query_authorized(&stamp)
            && *stamp.0.ui_inventory_stamp.lock().unwrap() == Some(token)
        {
            return;
        }
        let mut tracks = std::collections::BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        // 诊断计数：多少 item 认领到了 region（没认领的只能是无源占位）。
        let mut item_count = 0usize;
        let mut claimed_items = 0usize;
        // 本实例所在轨道的 `I_FOLDERDEPTH`；读不出来保持 `None`（见 `HostAudioState`）。
        let mut fx_folder_depth: Option<i32> = None;
        for owner in stamp.0.renderer_owners() {
            if !allowed() {
                return;
            }
            let Some(host) = owner.reaper.lock().unwrap().clone() else {
                continue;
            };
            let own_allowed = || {
                allowed()
                    && !owner.is_closed()
                    && owner
                        .editor_document()
                        .is_ok_and(|doc| Arc::ptr_eq(&doc, &stamp.0))
            };
            let Ok(parent) = host.direct_track_target(&own_allowed) else {
                continue;
            };
            // 只读本实例那条轨道的单值，不枚举整个工程。放在 `seen` 去重之前：本实例
            // 可能不是本轮第一个被处理的 owner，去重会让它被跳过。
            if std::ptr::eq(self, Arc::as_ptr(&owner)) {
                fx_folder_depth = parent.folder_depth(&own_allowed).ok();
            }
            // 父轨已枚举过 ⇒ 它的全部后代也已经在 `tracks` 里，不必重走一遍。
            if seen.contains(parent.inventory_guid()) {
                continue;
            }
            // FX 挂在 folder 轨上时，folder 轨自身通常没有 item：连同后代一起枚举，
            // 否则整组在插件里"看不见"（ARA 侧没有 folder 概念，只能从宿主清单取）。
            let Ok(own_tracks) = host.ui_folder_tracks(&own_allowed) else {
                continue;
            };
            for mut track in own_tracks {
                if !seen.insert(track.guid.clone()) {
                    continue;
                }
                let ids = stamp.0.clip_ids.lock().unwrap();
                let timeline = stamp.0.timeline.lock().unwrap();
                item_count += track.items.len();
                for candidate in stamp.0.renderer_owners() {
                    // 该 owner 可能同时持有多个 region（folder 轨上的 FX）：只要本轨有
                    // 任一 item 被它的某个 region 绑定，就把轨道身份认领到那个 clip 上。
                    let matched =
                        candidate
                            .host_geometries_locked(&stamp.0)
                            .into_iter()
                            .find(|bound| {
                                track
                                    .items
                                    .iter()
                                    .any(|item| item.geometry.item_id == bound.geometry.item_id)
                            });
                    if let Some(bound) = matched {
                        claimed_items += track
                            .items
                            .iter()
                            .filter(|item| item.geometry.item_id == bound.geometry.item_id)
                            .count();
                        if let Some(id) = ids.get(&bound.region_key) {
                            if let Some(clip) = timeline
                                .as_ref()
                                .and_then(|t| t.clips.iter().find(|c| &c.id == id))
                            {
                                track.id = clip.track_id.clone();
                                break;
                            }
                        }
                    }
                }
                drop(timeline);
                drop(ids);
                if track.id.starts_with("host-track-") {
                    if let Some(old) = stamp.0.ui_tracks.lock().unwrap().get(&track.guid) {
                        track.id = old.id.clone();
                    }
                }
                tracks.insert(track.guid.clone(), track);
            }
        }
        // 分类只用两项可核实的事实（见 `classify_host_audio`）。folder 结构读不出来时
        // `None` ⇒ 退化成"等待中"，绝不给出一个可能指错方向的 folder 提示。
        let status = classify_host_audio(
            fx_folder_depth.is_some_and(|depth| depth >= 1),
            item_count,
            claimed_items,
        );
        if !allowed() || host.geometry_revision(allowed).ok() != Some(change) {
            return;
        }
        // 采样"ARA 表达不到、但会改变渲染结果"的宿主事实（方向 / 循环源 / 声道模式）。
        // 与 `project_host_take_facts_locked` 用**同一个** active-take 口径，否则"清单
        // 说倒放、渲染播种说正放"会再次分叉。
        let facts =
            crate::host::geometry::RenderFacts::from_items(tracks.values().flat_map(|track| {
                track.items.iter().map(|item| {
                    let active = item.takes.iter().find(|take| take.active);
                    let geometry = active.map(|take| &take.geometry).unwrap_or(&item.geometry);
                    (
                        item.geometry.item_id.clone(),
                        crate::host::geometry::RenderItemFacts {
                            reversed: active.is_some_and(|take| take.reversed),
                            loop_enabled: geometry.loop_source,
                            channel_mode: geometry.channel_mode,
                        },
                    )
                })
            }));
        let _transaction = stamp.0.transaction.lock().unwrap();
        if self.is_closed()
            || !stamp.0.is_alive()
            || stamp.0.revision.load(Ordering::Acquire) != stamp.1
            || stamp.0.scope_revision.load(Ordering::Acquire) != stamp.2
        {
            return;
        }
        // 【这是"宿主改了、插件立刻重渲染"的唯一通路】任何一项渲染事实变化都推进
        // `render_epoch`，于是 `PreparedVersion`（渲染缓存键）失效、`prepare_job` 不再
        // 早退。此前只有**淡变长度**有这条通路，倒放/声道/循环源都没有 —— 用户报障
        // "倒放后波形更新了、声音没变，编辑一下才对"正是这个缺口。
        let mut facts_changed = false;
        {
            let mut previous = stamp.0.render_facts.lock().unwrap();
            if *previous != facts {
                *previous = facts;
                stamp.0.render_epoch.fetch_add(1, Ordering::AcqRel);
                facts_changed = true;
            }
        }
        stamp
            .0
            .ui_known_tracks
            .lock()
            .unwrap()
            .extend(tracks.values().map(|track| track.id.clone()));
        *stamp.0.ui_tracks.lock().unwrap() = tracks;
        *stamp.0.ui_inventory_stamp.lock().unwrap() = Some(token);
        stamp.0.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        *self.host_audio.lock().unwrap() = status;
        // 用户报障时最需要的一行：宿主清单里有多少 item，其中多少被 ARA region 认领。
        // 认领不到的那些只能以无源占位出现（见方案 Task 5.1 / R4）。
        if item_count > 0 {
            log::info!(
                "[ara] host inventory: {} track(s), {} item(s), {} claimed by an assigned region",
                stamp.0.ui_tracks.lock().unwrap().len(),
                item_count,
                claimed_items
            );
        }
        if status.state != HostAudioState::Ready {
            // 直读病灶：本实例到底拿到几条 region、FX 轨是不是 folder 父轨。下一次同样
            // 的报障不必再从位掩码与清单反推（见方案 Task 3.3）。
            let assigned = self
                .host_assigned_regions(&stamp.0)
                .map(|keys| keys.len())
                .unwrap_or(0);
            let depth = match fx_folder_depth {
                Some(depth) => depth.to_string(),
                None => "unknown".to_owned(),
            };
            log::info!("[ara] fx track: folderDepth={depth}; assigned regions={assigned}");
            log::info!(
                "[ara] {} clip(s) still waiting for host audio ({})",
                status.waiting_items,
                status.state.as_str()
            );
        }
        drop(_transaction);
        if facts_changed {
            // 与 `refresh_reaper_geometry` 的淡变分支同一理由：渲染输入变了就重排准备。
            // 不持 `transaction` 调用，避免与 `prepare_job` 的事务重入。
            self.prepare();
        }
    }
    /// 元数据单独采集，不让每个隐藏实例重复写工程clock；宿主调用始终不持内部锁。
    pub(crate) fn refresh_reaper_state_for_model(&self) {
        self.refresh_reaper_geometry();
    }
    fn refresh_reaper_geometry(&self) {
        let Ok(stamp) = self.host_query_stamp() else {
            return;
        };
        let host = { self.reaper.lock().unwrap().clone() };
        let authorized = || self.host_query_authorized(&stamp);
        let before = host
            .as_ref()
            .and_then(|host| host.geometry_revision(authorized).ok());
        if !authorized() {
            return;
        }
        let unchanged = before.is_some()
            && self
                .host_geometry
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|cached| {
                    cached.model == stamp.1
                        && cached.scope == stamp.2
                        && cached.change == before
                        && cached.value.is_ok()
                });
        if unchanged {
            let after = host
                .as_ref()
                .and_then(|host| host.geometry_revision(authorized).ok());
            if before == after && authorized() {
                return;
            }
        }
        if !authorized() {
            return;
        }
        // 唯一 region 时走强身份路径（宿主直接 parent take）；多 region（folder 轨上
        // 的 FX）结构上不可能有直接 parent，只能逐 region 按几何唯一匹配。
        let mut geometry = if stamp.3.len() == 1 {
            self.reaper_geometry().map(|bound| vec![bound])
        } else {
            self.reaper_geometries()
        };
        let after = host
            .as_ref()
            .and_then(|host| host.geometry_revision(authorized).ok());
        if geometry.is_ok() && (before.is_none() || before != after) {
            geometry = Err("REAPER project changed during cached geometry refresh".into());
        }
        let _transaction = stamp.0.transaction.lock().unwrap();
        let mut prepare_fades = false;
        if !self.is_closed()
            && stamp.0.is_alive()
            && stamp.0.revision.load(Ordering::Acquire) == stamp.1
            && stamp.0.scope_revision.load(Ordering::Acquire) == stamp.2
        {
            let mut cached = self.host_geometry.lock().unwrap();
            let changed = cached.as_ref().is_none_or(|old| {
                old.model != stamp.1 || old.scope != stamp.2 || old.value != geometry
            });
            let delegated = stamp.3.iter().any(|key| {
                stamp
                    .0
                    .regions
                    .lock()
                    .unwrap()
                    .get(key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
            });
            if delegated && changed && geometry.is_ok() {
                // 比较**整组**的 fade 长度：多 region 时只看第一个会漏掉后续 region
                // 的 fade 变化，导致委托端改动不触发重渲染。
                let lengths = |value: &[crate::host::geometry::BoundHostGeometry]| {
                    value
                        .iter()
                        .flat_map(|entry| {
                            [
                                entry.geometry.fade_in_sec,
                                entry.geometry.fade_out_sec,
                                entry.geometry.auto_fade_in_sec,
                                entry.geometry.auto_fade_out_sec,
                            ]
                        })
                        .collect::<Vec<_>>()
                };
                prepare_fades = cached
                    .as_ref()
                    .and_then(|old| old.value.as_ref().ok())
                    .is_none_or(|old| lengths(old) != lengths(geometry.as_ref().unwrap()));
                if prepare_fades {
                    stamp.0.render_epoch.fetch_add(1, Ordering::AcqRel);
                }
            }
            // 门控是**实例级**的：单 region 时等同于该 item 的 mute（与既有语义一致）；
            // 多 region 时只有全部绑定 item 都静音才静音整组 —— 一个子轨静音不该让
            // 整组哑掉。逐 item 的独立门控需要渲染路径支持，见方案 Task 5.2 的已知边界。
            self.host_item_muted.store(
                geometry
                    .as_ref()
                    .is_ok_and(|bound| !bound.is_empty() && bound.iter().all(|e| e.geometry.muted)),
                Ordering::Release,
            );
            if let Ok(bound) = &geometry {
                let mut items = stamp.0.region_items.lock().unwrap();
                for entry in bound {
                    items.insert(entry.region_key, entry.geometry.item_id.clone());
                }
            }
            *cached = Some(CachedHostGeometry {
                model: stamp.1,
                scope: stamp.2,
                change: after,
                value: geometry,
            });
            drop(cached);
            if changed {
                stamp.0.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
            }
        }
        drop(_transaction);
        if prepare_fades {
            self.prepare();
        }
    }
    /// realtime只读角色原子值；只有playback角色负责替换歌曲音频。
    pub(crate) fn renders_playback(&self) -> bool {
        self.role.load(Ordering::Acquire) == 1
    }
    /// 最终输出前的有效item静音门；与编辑器内轨道mute不同，不从输入是否全零猜静音。
    pub(crate) fn host_item_muted(&self) -> bool {
        self.host_item_muted.load(Ordering::Acquire)
    }
    /// editor-only必须透传；未绑定角色不在这里虚构为editor，保留原未绑定安全行为。
    pub(crate) fn is_editor_only(&self) -> bool {
        self.role.load(Ordering::Acquire) == 2
    }
    /// process只写有界原子诊断，JSON/日志全部由actor读取，暂不把候选计数当作根因。
    pub(crate) fn observe_transport(&self, context: &crate::audio_abi::ProcessContext, mode: i32) {
        if let Some(clock) = self.clock.get() {
            clock.observe(
                context,
                self.writer_id.load(Ordering::Relaxed),
                self.roles.load(Ordering::Relaxed),
                mode,
            );
            if mode != 2 {
                clock.update(context);
            }
        }
    }
    /// SDK允许从音频线程调用；记录来源但不做文件IO。
    pub(crate) fn record_processing_stop(&self) {
        if let Some(clock) = self.clock.get() {
            clock.observe_stop(
                self.writer_id.load(Ordering::Relaxed),
                self.roles.load(Ordering::Relaxed),
            );
            clock.stopped();
        }
    }
    /// 真实组件仅作为授权入口，原GUI共享同文档唯一actor/history。
    pub(crate) fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        self.editor_document()?.editor_session()
    }
    /// GUI能力标记只回答是否有官方写API；实际每个clip仍须唯一真实take绑定。
    /// 宿主淡化轴的版本语义；`None` 表示版本读不出来（前端应保持只读）。
    ///
    /// 取**任一** owner 的读数：`fade_axes_new` 来自 `GetAppVersion`，同一进程里
    /// 所有实例必然一致，不存在"这个 owner 是新轴、那个是旧轴"的情况。
    pub(crate) fn host_fade_axes(&self) -> Option<bool> {
        self.editor_document().ok().and_then(|document| {
            document.renderer_owners().iter().find_map(|owner| {
                owner
                    .reaper
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|host| host.fade_axes_new())
            })
        })
    }

    pub(crate) fn host_clip_editing_available(&self) -> bool {
        self.editor_document().is_ok_and(|document| {
            document.renderer_owners().iter().any(|owner| {
                owner
                    .reaper
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|host| host.has_project_history())
            })
        })
    }
    /// 只返回同文档活入口的拥有引用接口；其project()仍钉在初始化时的真实所属工程。
    pub(crate) fn project_history_host(&self) -> Option<Arc<crate::host::reaper::ReaperHost>> {
        let document = self.editor_document().ok()?;
        if let Some(host) = self
            .reaper
            .lock()
            .unwrap()
            .as_ref()
            .filter(|host| host.has_project_history())
            .cloned()
        {
            return Some(host);
        }
        document.renderer_owners().iter().find_map(|owner| {
            owner
                .reaper
                .lock()
                .unwrap()
                .as_ref()
                .filter(|host| host.has_project_history())
                .cloned()
        })
    }
    /// 明确指定轨道须已有真实assigned item；空轨未指定时仅允许本FX的直接parent轨道。
    pub(crate) fn audio_import_target(
        &self,
        track: Option<&str>,
        authorized: &impl Fn() -> bool,
    ) -> Result<crate::host::reaper::HostTrackTarget, String> {
        if let Some(id) = track {
            let document = self.editor_document()?;
            let target = {
                document
                    .ui_tracks
                    .lock()
                    .unwrap()
                    .values()
                    .find(|t| t.id == id)
                    .map(|t| t.target.clone())
            };
            if let Some(target) = target {
                return Ok(target);
            }
        }
        if let Some(track) = track {
            let document = self.editor_document()?;
            let clips = {
                let timeline = document.timeline.lock().unwrap();
                timeline
                    .as_ref()
                    .ok_or("host timeline unavailable")?
                    .clips
                    .iter()
                    .filter(|clip| clip.track_id == track)
                    .map(|clip| clip.id.clone())
                    .collect::<Vec<_>>()
            };
            for clip in clips {
                if let Ok(target) = self.host_edit_target(&clip) {
                    return target.import_target(authorized);
                }
            }
            return Err("import track has no directly bound host region".into());
        }
        let host = self
            .reaper
            .lock()
            .unwrap()
            .clone()
            .ok_or("primary REAPER host interface missing")?;
        host.direct_track_target(authorized)
    }
    /// 同文档实际renderer唯一region边解析写对象，不能按名字/位置/源路径查item。
    pub(crate) fn host_edit_target(
        &self,
        clip_id: &str,
    ) -> Result<crate::host::reaper::HostClipTarget, String> {
        self.refresh_ui_inventory();
        let document = self.editor_document()?;
        if let Some(item) = clip_id.strip_prefix("ara-item-") {
            let target = document
                .ui_tracks
                .lock()
                .unwrap()
                .values()
                .flat_map(|track| &track.items)
                .find(|entry| entry.geometry.item_id == item)
                .map(|entry| entry.target.clone());
            if let Some(target) = target {
                return target.current(&|| {
                    !self.is_closed()
                        && document.is_alive()
                        && self
                            .editor_document()
                            .is_ok_and(|doc| Arc::ptr_eq(&doc, &document))
                });
            }
        }
        let key = {
            let ids = document.clip_ids.lock().unwrap();
            ids.iter()
                .find(|(_, id)| id.as_str() == clip_id)
                .map(|(key, _)| *key)
                .ok_or("unknown ARA clip identity")?
        };
        let mut target = None;
        for owner in document.renderer_owners() {
            if !owner.renders_playback()
                || !owner
                    .assigned_regions()
                    .is_ok_and(|keys| keys.as_slice() == [key])
            {
                continue;
            }
            let stamp = owner.host_query_stamp()?;
            let bound = owner.reaper_geometry()?;
            if bound.region_key != key {
                return Err("host region binding changed".into());
            }
            let host = owner
                .reaper
                .lock()
                .unwrap()
                .clone()
                .ok_or("REAPER host interface missing")?;
            let next = host.clip_target(|| owner.host_query_authorized(&stamp))?;
            if target.is_some() {
                return Err("multiple host writers for one ARA clip".into());
            }
            target = Some(next);
        }
        target.ok_or_else(|| "clip has no unique directly bound REAPER take".into())
    }
    /// 组件/文档终止时停止实例worker；不要在音频process调用。
    pub(crate) fn stop_editor(&self) {
        let document = {
            self.document
                .lock()
                .unwrap()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
        };
        if let Some(document) = document {
            {
                let _transaction = document.transaction.lock().unwrap();
                if !self.closed.swap(true, Ordering::AcqRel) {
                    document.scope_revision.fetch_add(1, Ordering::AcqRel);
                }
                self.snapshots.iter().for_each(|snapshot| snapshot.clear());
            }
            // 不持文档事务取得views锁，避免enqueue/worker授权反向等待。
            document.revoke_editor_views();
        } else {
            self.closed.store(true, Ordering::Release);
            self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
        let host = self.reaper.lock().unwrap().take();
        drop(host);
        self.host_geometry.lock().unwrap().take();
        self.cancel_preparation();
        if let Some(Ok(worker)) = self.preparation.get() {
            worker.close();
        }
    }
    /// 模型撤销时立即取消待准备作业；保持worker可供后续重新授权使用。
    pub(crate) fn cancel_preparation(&self) {
        if let Some(Ok(worker)) = self.preparation.get() {
            worker.cancel();
        }
    }
    /// 原GUI合并后台宿主准备状态，不能把冷恢复尚未准备完写成已应用。
    pub(crate) fn preparation_state(&self) -> (bool, Option<String>) {
        if self.is_editor_only() {
            if let Some(document) = self
                .document
                .lock()
                .unwrap()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
            {
                let states = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| owner.local_preparation_state())
                    .collect::<Vec<_>>();
                return (
                    states.iter().any(|state| state.0),
                    states.into_iter().find_map(|state| state.1),
                );
            }
        }
        self.local_preparation_state()
    }
    pub(crate) fn local_preparation_state(&self) -> (bool, Option<String>) {
        match self.preparation.get() {
            Some(Ok(worker)) => worker.state(),
            Some(Err(error)) => (false, Some(error.clone())),
            None => (false, None),
        }
    }
    /// 渲染器的准备状态快照（诊断包用）：模型/编辑代次、作用域与区间数。
    ///
    /// 【为什么单独抽出来】`Snapshot` 里那段诊断需要当前分配区间才能算逐输出读数；
    /// 诊断包在任意时刻被导出，不值得为它触发一次完整快照。这里只给"准备好了没有、
    /// 准备的是哪一版"，那正是排查"渲染结果陈旧"时要看的东西。
    pub(crate) fn prepared_snapshot(&self) -> Option<serde_json::Value> {
        self.prepared.lock().unwrap().as_ref().map(|version| {
            serde_json::json!({
                "model": version.model,
                "edit": version.edit,
                "epoch": version.epoch,
                "scope": version.scope,
                "regions": version.keys.len(),
            })
        })
    }
    /// native主线程取得宿主给本ARA文档的可撤销播放租约；不寻找全局REAPER窗口。
    pub(crate) fn host_playback(&self) -> Option<ara2_bridge::plugin::PlaybackRequestHandle> {
        self.document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .and_then(|document| document.playback.lock().unwrap().clone())
    }
    /// 组件释放/文档关闭前结束后台服务，避免DLL卸载后线程仍执行插件代码。
    pub fn stop_channel(&self) {
        let channel = self.channel.lock().unwrap().take();
        drop(channel);
    }

    /// 未绑定时暂存组件state；绑定后所有处理器读取同一文档的参数权威。
    // 组件state访问器保留，供未绑定路径读取。
    #[allow(dead_code)]
    pub fn edit_state(&self) -> Arc<Mutex<crate::state_channel::EditState>> {
        self.document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .map(|document| document.edits.clone())
            .unwrap_or_else(|| self.edits.clone())
    }

    /// 恢复/undo组件状态后刷新整张文档，各renderer保持原分配。
    pub fn refresh_document(&self) {
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            {
                let _transaction = document.transaction.lock().unwrap();
                let owners = document.renderer_owners();
                for owner in &owners {
                    if let Err(error) = owner.merge_pending_restore(&document) {
                        log::warn!("[ara] instance state unresolved: {error}");
                    }
                }
                for owner in owners {
                    owner.prepare();
                }
            }
        }
    }

    /// setState只暂存本组件恢复；宿主明确分配区域且模型ready后再合入共享权威。
    pub(crate) fn restore_state(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(bytes)?;
        *self.pending_restore.lock().unwrap() = Some(restored);
        self.refresh_document();
        Ok(())
    }
    /// 调用方持文档transaction；恢复候选集由宿主区域分配收窄，不按名称/旧序号猜测。
    pub(super) fn merge_pending_restore(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<(), String> {
        let mut pending = self.pending_restore.lock().unwrap();
        let Some(saved) = pending.as_ref() else {
            return Ok(());
        };
        if !document.ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let host = self.assigned_timeline(document)?;
        if host.tracks.is_empty() {
            // 空轨组件仍能恢复私有参数分组；不借零音频scope恢复其它轨的源曲线。
            if saved.params.is_empty() && saved.tracks.is_empty() && saved.atlas.is_empty() {
                let mut edits = document.edits.lock().unwrap();
                edits.groups = saved.groups.clone();
                edits.revision = edits
                    .revision
                    .checked_add(1)
                    .ok_or("edit revision exhausted")?;
                *pending = None;
            }
            return Ok(());
        }
        let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
        let bindings = document
            .track_bindings
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| allowed.contains(*id))
            .map(|(id, identity)| (id.clone(), identity.clone()))
            .collect();
        let mut restored = saved.clone();
        restored.reconcile(&bindings)?;
        restored.atlas.migrate_legacy_gaps(&restored.params)?;
        let mut changed_geometry = false;
        if !restored.atlas.is_empty() {
            let rebound = restored
                .atlas
                .rebind(&host, &document.parameter_identities_locked(&host)?)?;
            changed_geometry = !restored.atlas.same_layout(&rebound);
            restored.atlas = rebound;
        }
        let mut client = host.clone();
        restored.apply(&mut client);
        if changed_geometry {
            client.params_by_root_track.extend(
                restored
                    .atlas
                    .project_roots(&host, &document.parameter_identities_locked(&host)?)?,
            );
        } else if restored.atlas.is_empty() && !restored.params.is_empty() {
            // 旧v2只有项目帧：在首次真实绑定的宿主布局建立source basis，随后移动/拉伸沿此迁移。
            // 不猜旧会话key/名字，也不能回推旧状态未保存的、首次加载前已改变的布局。
            restored.atlas = crate::editor::parameter_atlas::ParameterAtlas::default()
                .capture(&client, &document.parameter_identities_locked(&host)?)?;
        }
        let mut edits = document.edits.lock().unwrap();
        let mut merged = edits.merge(&host, &client, edits.revision)?;
        merged.groups = restored.groups.clone();
        let clip_ids = host
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect::<BTreeSet<_>>();
        merged.atlas.regions.retain(|id, _| !clip_ids.contains(id));
        merged.atlas.regions.extend(restored.atlas.regions);
        merged
            .atlas
            .copy_seeds
            .retain(|_, seed| !allowed.contains(&seed.root));
        merged.atlas.copy_seeds.extend(restored.atlas.copy_seeds);
        merged.atlas.gaps.retain(|root, _| !allowed.contains(root));
        merged.atlas.gaps.extend(restored.atlas.gaps);
        let items = document.fade_items_locked(&clip_ids);
        if !merged.fades.is_empty() && items.is_empty() && !clip_ids.is_empty() {
            return Err("fade ownership metadata pending for restore".into());
        }
        merged.fades.retain(|item, _| !items.contains(item));
        merged.fades.extend(restored.fades);
        merged.reconcile(&document.track_bindings.lock().unwrap())?;
        *edits = merged;
        *pending = None;
        Ok(())
    }

    /// 保存前重新核对完整宿主图；歧义或中间编辑态不能伪装成可恢复的state。
    pub(crate) fn encode_state(&self) -> Result<Vec<u8>, String> {
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            document.flush_editor()?;
            let _transaction = document.transaction.lock().unwrap();
            if !document.ready.load(Ordering::Acquire) {
                return Err("host graph not ready; cannot save ARA edits".into());
            }
            self.merge_pending_restore(&document)?;
            let mut edits = document.edits.lock().unwrap();
            edits.reconcile(&document.track_bindings.lock().unwrap())?;
            let host = self.assigned_timeline(&document)?;
            let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
            let mut local = edits.clone();
            local.params.retain(|id, _| allowed.contains(id));
            local.tracks.retain(|t| allowed.contains(&t.id));
            let clips = host
                .clips
                .iter()
                .map(|clip| clip.id.clone())
                .collect::<BTreeSet<_>>();
            local.atlas.regions.retain(|id, _| clips.contains(id));
            local
                .atlas
                .copy_seeds
                .retain(|_, seed| allowed.contains(&seed.root));
            local.atlas.gaps.retain(|root, _| allowed.contains(root));
            if !local.fades.is_empty() {
                let items = document.fade_items_locked(&clips);
                if items.is_empty() && !clips.is_empty() {
                    return Err("fade ownership metadata pending for save".into());
                }
                local.fades.retain(|item, _| items.contains(item));
            }
            local.bindings.retain(|id, _| allowed.contains(id));
            local.encode()
        } else {
            self.pending_restore
                .lock()
                .unwrap()
                .as_ref()
                .unwrap_or(&self.edits.lock().unwrap())
                .encode()
        }
    }

    /// 参数/快照在短事务内校验；外部提交合成同样不能持锁阻塞宿主模型回调。
    pub(crate) fn handle_request(
        &self,
        request: hifishifter_ara_ipc::Request,
    ) -> hifishifter_ara_ipc::Response {
        use hifishifter_ara_ipc::{HostPcm, Request, Response};
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        let Some(document) = document else {
            return Response {
                error: Some("document closed".into()),
                ..Default::default()
            };
        };
        if let Request::Commit {
            base_revision,
            model_revision,
            timeline,
        } = request
        {
            return self.commit_request(&document, base_revision, model_revision, timeline);
        }
        let _transaction = document.transaction.lock().unwrap();
        if let Err(error) = self.merge_pending_restore(&document) {
            return Response {
                error: Some(error),
                ..Default::default()
            };
        }
        let model_revision = document.revision.load(Ordering::Acquire);
        let mut edits = document.edits.lock().unwrap();
        let outcome = (|| {
            if !document.ready.load(Ordering::Acquire) {
                return Err("host model not ready; refresh after editing".into());
            }
            edits.reconcile(&document.track_bindings.lock().unwrap())?;
            let mut timeline = self.assigned_timeline(&document)?;
            match request {
                Request::Snapshot => {
                    edits.apply(&mut timeline);
                    let available = document.edit_sources.lock().unwrap();
                    let mut sources = Vec::new();
                    for id in timeline
                        .clips
                        .iter()
                        .filter_map(|clip| clip.source_path.as_ref())
                        .collect::<BTreeSet<_>>()
                    {
                        let pcm = available
                            .get(id)
                            .ok_or_else(|| format!("host PCM unavailable: {id}"))?;
                        sources.push(HostPcm {
                            persistent_id: id.clone(),
                            sample_rate: pcm.sample_rate,
                            fingerprint: pcm_fingerprint(pcm),
                            planes: pcm.planes.clone(),
                        });
                    }
                    let mut clips = timeline.clips.iter().collect::<Vec<_>>();
                    clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
                    let ranges = clips
                        .iter()
                        .take(8)
                        .map(|clip| (clip.id.clone(), clip.start_sec, clip.length_sec))
                        .collect::<Vec<_>>();
                    let diagnostics=document.renderer_owners().into_iter().filter(|owner|owner.renders_playback()).map(|owner| {
                        let (busy,error)=owner.local_preparation_state();
                        let prepared=owner.prepared_snapshot();
                        serde_json::json!({"busy":busy,"error":error,"prepared":prepared,
                            "outputs":owner.snapshots.iter().map(|snapshot|snapshot.diagnose(&ranges)).collect::<Vec<_>>()})
                    }).collect::<Vec<_>>();
                    Ok(Response {
                        ok: true,
                        timeline: Some(serde_json::to_value(&timeline).map_err(|e| e.to_string())?),
                        sources,
                        diagnostics: Some(serde_json::json!({"renderers":diagnostics})),
                        ..Default::default()
                    })
                }
                Request::Commit { .. } => {
                    unreachable!("commit dispatched before snapshot transaction")
                }
            }
        })();
        let mut response = outcome.unwrap_or_else(|error| Response {
            error: Some(error),
            ..Default::default()
        });
        response.revision = edits.revision;
        response.model_revision = model_revision;
        response
    }

    /// 一期外部客户端保持同步结果契约，但捕获/计算/提交分开，失败不改变参数权威。
    fn commit_request(
        &self,
        document: &Arc<super::document::DocumentSession>,
        base_edit: u64,
        base_model: u64,
        client: serde_json::Value,
    ) -> hifishifter_ara_ipc::Response {
        let outcome = (|| {
            let (candidate, epoch, scope, inputs) = {
                let _transaction = document.transaction.lock().unwrap();
                self.merge_pending_restore(document)?;
                if !document.ready.load(Ordering::Acquire) {
                    return Err("host model not ready; refresh after editing".into());
                }
                if document.revision.load(Ordering::Acquire) != base_model {
                    return Err("Conflict: host model changed; refresh".into());
                }
                let mut edits = document.edits.lock().unwrap();
                edits.reconcile(&document.track_bindings.lock().unwrap())?;
                let client = serde_json::from_value(client).map_err(|e| e.to_string())?;
                let mut candidate =
                    edits.merge(&self.assigned_timeline(document)?, &client, base_edit)?;
                let mut curve_timeline = self.assigned_timeline(document)?;
                candidate.apply(&mut curve_timeline);
                candidate.atlas = edits.atlas.capture(
                    &curve_timeline,
                    &document.parameter_identities_locked(&curve_timeline)?,
                )?;
                candidate.reconcile(&document.track_bindings.lock().unwrap())?;
                drop(edits);
                let inputs = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| {
                        let (keys, input) =
                            owner.capture_render_input(document, &candidate, true)?;
                        Ok((owner, keys, input))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                (
                    candidate,
                    document.render_epoch.load(Ordering::Acquire),
                    document.scope_revision.load(Ordering::Acquire),
                    inputs,
                )
            };
            let mut prepared = Vec::new();
            for (owner, keys, input) in inputs {
                for publisher in &owner.snapshots {
                    publisher.collect_retired();
                }
                let mut snapshots =
                    input.render(Arc::new(std::sync::atomic::AtomicBool::new(false)))?;
                super::snapshot::compact_prepared(&mut snapshots)?;
                prepared.push((owner, keys, snapshots));
            }
            let _transaction = document.transaction.lock().unwrap();
            if !document.ready.load(Ordering::Acquire)
                || document.revision.load(Ordering::Acquire) != base_model
            {
                return Err("Conflict: host model changed during commit".into());
            }
            let mut edits = document.edits.lock().unwrap();
            if edits.revision != base_edit {
                return Err("Conflict: edit revision changed during commit".into());
            }
            if document.render_epoch.load(Ordering::Acquire) != epoch
                || document.scope_revision.load(Ordering::Acquire) != scope
            {
                return Err("host audio access/scope changed during commit; retry".into());
            }
            for (owner, keys, snapshots) in &prepared {
                if owner.assigned_regions().map_err(|e| e.to_string())? != *keys {
                    return Err("Conflict: assigned regions changed during commit".into());
                }
                if !owner
                    .snapshots
                    .iter()
                    .zip(snapshots)
                    .all(|(p, s)| p.has_capacity(s))
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
                owner.record_prepared(base_model, candidate.revision, epoch, scope, keys);
            }
            *edits = candidate;
            log::info!(
                "[ara] GUI commit ready revision={} model={base_model}",
                edits.revision
            );
            Ok::<(), String>(())
        })();
        hifishifter_ara_ipc::Response {
            ok: outcome.is_ok(),
            error: outcome.err(),
            revision: document.edits.lock().unwrap().revision,
            model_revision: document.revision.load(Ordering::Acquire),
            ..Default::default()
        }
    }

    /// 调用方在短事务内冻结几何/参数/PCM，只复制本renderer授权源，不执行合成。
    pub(super) fn capture_render_input(
        &self,
        document: &super::document::DocumentSession,
        edits: &crate::state_channel::EditState,
        edited: bool,
    ) -> Result<(Vec<u64>, super::input::RenderInput), String> {
        if !document.ready.load(Ordering::Acquire) {
            return Err("host PCM/model not ready".into());
        }
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        let mut timeline = self.assigned_timeline(document)?;
        // 所有授权轨道保留原kernel的全局solo/父链判定，clip仍只有本renderer分配区域。
        timeline.tracks = document.workspace_timeline_locked()?.tracks;
        // 【为什么这里也要播种宿主 take 事实】渲染路径**不经过** `workspace_timeline_locked`
        // 的 clip 过滤（见上一行的注释：只借了 tracks）。而 `assigned_timeline` 读的是
        // `document.timeline`（ARA 映射产物），ARA **没有**方向位/循环源/声道模式
        // （`ara::mapping::LOST_FIELDS`）—— 不播种，倒放片段会被按正放合成（用户报障：
        // "在 HiFiShifter 中，倒放被识别为正放"）。与音阶播种（`document.timeline` 的
        // `project_scale_notes`）是同一条纪律、同一个理由。
        document.project_host_take_facts_locked(&mut timeline);
        let mut resolved = edits.clone();
        resolved.reconcile(&document.track_bindings.lock().unwrap())?;
        resolved.apply(&mut timeline);
        if !resolved.groups.is_empty() {
            // 空父轨也能承载参数组算法；不加入其音频clip，更不混入其它renderer的区域。
            let mut grouped_view = timeline.clone();
            document.project_private_group_view(&mut grouped_view, &resolved)?;
            timeline.tracks = grouped_view.tracks;
            for track in &mut timeline.tracks {
                track.parent_id = None;
            }
            resolved.groups.processing_settings(
                &mut timeline,
                &document.group_aliases(""),
                &std::collections::BTreeSet::new(),
            );
        }
        let geometry = document.regions.lock().unwrap();
        let regions = keys
            .iter()
            .map(|key| {
                geometry
                    .get(key)
                    .cloned()
                    .ok_or("assigned region disappeared")
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(geometry);
        let stretch = regions.iter().any(|region| {
            (region.duration_in_modification_time - region.duration_in_playback_time).abs() > 1e-9
        });
        let owned_fade = regions.iter().any(|region| {
            region.has_content_based_fade_at_head || region.has_content_based_fade_at_tail
        });
        if owned_fade {
            document.project_audio_fades_locked(&mut timeline, &resolved)?;
        }
        // 【倒放也必须走内核】未编辑、无拉伸、无淡化委托的片段本来走
        // `mix_plain_regions` 直通（`RenderInput.timeline = None`）—— 那是"原样放 ARA
        // 授权 PCM"。但倒放片段**不能**直通：ARA 给的是正向 PCM，方向翻转只发生在内核
        // 的 `render_mixdown_internal`（`reverse_interleaved_frames`）。不把倒放片段推进
        // 内核，它就会被原样按正向放出来（实测：输出 = 源窗口正放）。
        let reversed_clip = timeline.clips.iter().any(|clip| clip.reversed);
        let kernel_render =
            edited || stretch || owned_fade || reversed_clip || !resolved.groups.is_empty();
        let clip_parameters = if resolved.atlas.is_empty() {
            Default::default()
        } else {
            resolved
                .atlas
                .project_local(&timeline, &document.parameter_identities_locked(&timeline)?)?
        };
        let available = if kernel_render {
            document.edit_sources.lock().unwrap()
        } else {
            document.sources.lock().unwrap()
        };
        let sources = regions
            .iter()
            .map(|region| region.audio_source_persistent_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| available.get(&id).map(|pcm| (id.clone(), pcm.clone())))
            .collect();
        Ok((
            keys,
            super::input::RenderInput {
                timeline: kernel_render.then_some(timeline),
                regions,
                sources,
                clip_parameters,
            },
        ))
    }

    /// 从宿主时间线筛选实际分配的区域，而不是让每个处理器混整张文档。
    fn assigned_timeline(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        let identities = document.clip_ids.lock().unwrap();
        let ids = keys
            .iter()
            .map(|key| {
                identities
                    .get(key)
                    .cloned()
                    .ok_or("missing assigned clip identity")
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut timeline = document
            .timeline
            .lock()
            .unwrap()
            .clone()
            .ok_or("host timeline unavailable")?;
        if let Some(tempo) = document.clock.tempo() {
            timeline.bpm = tempo;
        }
        // 【为什么这里也要播种音阶】渲染路径**不经过** `workspace_timeline_locked`
        // 的 clip 分支：它克隆原始 ARA 时间线后只借用后者的 `tracks`。所以只在
        // `workspace_timeline_locked` 里播种，渲染拿到的仍是默认 C 大调 ——
        // 正是"界面显示 Gb、内核按 C 渲染"那条缺陷。
        document.project_plugin_musical_context_locked(&mut timeline);
        timeline.clips.retain(|clip| ids.contains(&clip.id));
        // 被分配了 sequence、当前却没有 clip 的轨道仍然留下：空 folder 轨、只有静音
        // item 或 PCM 尚未授权的子轨都属此类，而它们本该是**参数根** —— 与
        // `add_group_view_tracks` 的既有口径一致（空父轨也能承载参数组算法）。
        //
        // 【为什么这不会泄漏无关轨道】这条边只来自宿主 `createRegionSequence` 的真实
        // 键（`sequence_track_index`），且只取**本 owner 已分配**的那些 sequence。
        // 绝不按轨名/位置/序号凭空造轨道。
        let assigned_sequences = self
            .sequences
            .lock()
            .unwrap()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let empty_roots = document
            .sequence_track_index
            .lock()
            .unwrap()
            .iter()
            .filter(|(key, _)| assigned_sequences.contains(key))
            .filter_map(|(_, index)| timeline.tracks.get(*index).map(|track| track.id.clone()))
            .collect::<BTreeSet<_>>();
        timeline.tracks.retain(|track| {
            timeline.clips.iter().any(|clip| clip.track_id == track.id)
                || empty_roots.contains(&track.id)
        });
        Ok(timeline)
    }

    /// 把租约关联到真实文档；关闭文档时由其同步撤销。
    pub fn bind_to_document(
        self: &Arc<Self>,
        document: Arc<super::document::DocumentSession>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        companion: Option<ara2_bridge::companion::CompanionControllerBinding<'static>>,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            return Err(AraError::InvalidState("extension already bound"));
        }
        let owner = Arc::downgrade(self);
        let document_id = document.id;
        let observer = Arc::new(
            move |role: ExtensionRoles, keys: &[usize], sequences: &[usize]| {
                if let Some(owner) = owner.upgrade() {
                    let document = owner
                        .document
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(std::sync::Weak::upgrade);
                    let Some(document) = document else {
                        return;
                    };
                    // assignment写入、撤销和最终发布必须共用此短事务，消除check→publish竞态。
                    let _transaction = document.transaction.lock().unwrap();
                    let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                    let valid = keys.is_empty()
                        || region_owners()
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .resolve(&keys)
                            .is_ok_and(|(owner, _)| owner == document_id);
                    let count = keys.len();
                    let next_keys = if valid { keys.clone() } else { Vec::new() };
                    let next_sequences =
                        sequences.iter().map(|key| *key as u64).collect::<Vec<_>>();
                    let changed = owner.assignments.lock().unwrap().get(&role.bits())
                        != Some(&next_keys)
                        || owner.sequences.lock().unwrap().get(&role.bits())
                            != Some(&next_sequences);
                    if changed {
                        document.scope_revision.fetch_add(1, Ordering::AcqRel);
                    }
                    owner
                        .assignments
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(role.bits(), if valid { keys } else { Vec::new() });
                    owner.sequences.lock().unwrap().insert(
                        role.bits(),
                        sequences.iter().map(|key| *key as u64).collect(),
                    );
                    if valid {
                        log::info!(
                            "[ara] renderer assignment role={} regions={count}",
                            role.bits()
                        );
                    } else {
                        log::warn!("[ara] rejected unknown or cross-document renderer assignment");
                    }
                    owner.cancel_preparation();
                    owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
                    let owners = document.renderer_owners();
                    for renderer in &owners {
                        if let Err(error) = renderer.merge_pending_restore(&document) {
                            log::warn!("[ara] instance state unresolved: {error}");
                        }
                    }
                    // 某轨首次获得分配可能恢复其state并推进共享revision；其它轨也需要最新任务。
                    drop(_transaction);
                    for renderer in owners {
                        renderer.refresh_reaper_geometry();
                        renderer.prepare();
                    }
                }
            },
        );
        let supported = ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER;
        let enabled = ExtensionRoles::resolve(known, assigned, supported)?;
        let owned = ExtensionBinding::new_with_renderer_observer(
            generation, known, assigned, supported, observer,
        )?;
        let raw = owned.0.as_raw();
        document.attach(self, owned.1, companion)?;
        *self.document.lock().unwrap() = Some(Arc::downgrade(&document));
        let weak = Arc::downgrade(self);
        if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
            let _ = self
                .preparation
                .get_or_init(super::preparation::PreparationQueue::new);
        }
        // 只登记weak供宿主回调排队；后台计算不形成owner自循环。
        *self.prepare_owner.lock().unwrap() = Some(weak);
        let _ = self.clock.set(document.clock.clone());
        static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);
        self.writer_id.store(
            NEXT_WRITER.fetch_add(1, Ordering::Relaxed),
            Ordering::Release,
        );
        *current = Some(owned.0);
        self.roles.store(enabled.bits(), Ordering::Release);
        self.role.store(
            if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                1
            } else {
                2
            },
            Ordering::Release,
        );
        drop(current);
        if enabled.contains(ExtensionRoles::EDITOR_RENDERER) {
            let weak = Arc::downgrade(self);
            match hifishifter_ara_ipc::Server::start(
                "HiFiShifter / REAPER".into(),
                move |request| {
                    weak.upgrade()
                        .map(|owner| owner.handle_request(request))
                        .unwrap_or_else(|| hifishifter_ara_ipc::Response {
                            error: Some("plugin instance closed".into()),
                            ..Default::default()
                        })
                },
            ) {
                Ok(server) => {
                    log::info!(
                        "[ara] GUI channel ready instance={}",
                        server.record().instance_id
                    );
                    *self.channel.lock().unwrap() = Some(server);
                }
                Err(error) => log::warn!("[ara] GUI channel unavailable: {error}"),
            }
        }
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        self.stop_editor();
        self.stop_channel();
        self.assignments.lock().unwrap().clear();
        self.sequences.lock().unwrap().clear();
        self.document.lock().unwrap().take();
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
    }

    /// 模型线程展开当前 renderer 的显式区域与 editor sequence，并去重，拒绝跨文档身份。
    pub fn assigned_regions(&self) -> Result<Vec<u64>, AraError> {
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or(AraError::InvalidState("document no longer available"))?;
        let role = self.role.load(Ordering::Acquire);
        let mut keys = self
            .assignments
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default();
        let members = document.sequence_regions.lock().unwrap();
        for sequence in sequences {
            keys.extend(
                members
                    .get(&sequence)
                    .ok_or(AraError::InvalidState("unknown assigned sequence"))?
                    .iter()
                    .copied(),
            );
        }
        drop(members);
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty() {
            let (owner, _) = region_owners()
                .lock()
                .unwrap()
                .resolve(&keys)
                .map_err(|_| AraError::InvalidState("invalid assigned region"))?;
            if owner != document.id {
                return Err(AraError::InvalidState(
                    "assigned region belongs to another document",
                ));
            }
        }
        Ok(keys)
    }

    /// 宿主模型callback只撤销/排最新准备任务，不在主线程执行WORLD或PCM混音。
    pub fn prepare(&self) {
        if !self.renders_playback() {
            return;
        }
        // 源/assignment撤销在文档事务内清理；此函数只排队，不能在事务外clear旧任务刚发布的音频。
        let weak = self.prepare_owner.lock().unwrap().clone();
        let Some(weak) = weak else {
            return;
        };
        match self.preparation.get() {
            Some(Ok(worker)) => {
                if let Err(error) = worker.request(Box::new(move |cancel| {
                    weak.upgrade()
                        .ok_or("renderer closed".to_owned())?
                        .prepare_job(cancel)
                })) {
                    log::warn!("[ara] preparation unavailable: {error}");
                }
            }
            Some(Err(error)) => log::warn!("[ara] preparation unavailable: {error}"),
            None => {}
        }
    }
    /// 调用者持document事务；完整两输出率发布成功后才登记，不额外持有源或快照。
    pub(crate) fn record_prepared(
        &self,
        model: u64,
        edit: u64,
        epoch: u64,
        scope: u64,
        keys: Vec<u64>,
    ) {
        *self.prepared.lock().unwrap() = Some(PreparedVersion {
            model,
            edit,
            epoch,
            scope,
            keys,
        });
        if let Some(Ok(queue)) = self.preparation.get() {
            queue.acknowledge_idle_success();
        }
    }
    /// 仅SDK规定UI线程的kOffline setup调用；不在事务锁内等待，不把旧快照当最新版本。
    pub(crate) fn prepare_offline_until(&self, deadline: std::time::Instant) -> Result<(), String> {
        if !self.renders_playback() {
            return Ok(());
        }
        // SDK离线setup在UI线程；无GUI时也刷新真实item状态，不能依赖WebView timer。
        self.refresh_reaper_geometry();
        let document = self.editor_document()?;
        loop {
            {
                let _transaction = document.transaction.lock().unwrap();
                if self.is_closed() || !document.is_alive() {
                    return Err("offline renderer closed".into());
                }
                if !document.ready.load(Ordering::Acquire) {
                    return Err("offline host model/PCM not ready".into());
                }
                let keys = self.assigned_regions().map_err(|e| e.to_string())?;
                let version = PreparedVersion {
                    model: document.revision.load(Ordering::Acquire),
                    edit: document.edits.lock().unwrap().revision,
                    epoch: document.render_epoch.load(Ordering::Acquire),
                    scope: document.scope_revision.load(Ordering::Acquire),
                    keys,
                };
                if self.prepared.lock().unwrap().as_ref() == Some(&version)
                    && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
                {
                    return Ok(());
                }
            }
            if std::time::Instant::now() >= deadline {
                return Err("offline preparation timed out".into());
            }
            let worker = self
                .preparation
                .get()
                .ok_or("offline preparation worker missing")?
                .as_ref()
                .map_err(Clone::clone)?;
            if !worker.state().0 {
                self.prepare();
            }
            worker.wait_idle_until(deadline)?;
        }
    }
    /// 后台冻结/计算/发布三阶段；撤销标记和全部版本在发布短事务内重新验证。
    fn prepare_job(&self, cancel: Arc<std::sync::atomic::AtomicBool>) -> Result<(), String> {
        let Some(document) = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return Err("document closed".into());
        };
        let (model, edit, epoch, scope, keys, input) = {
            let _transaction = document.transaction.lock().unwrap();
            if !document.ready.load(Ordering::Acquire) {
                return Ok(());
            }
            let edits = document.edits.lock().unwrap().clone();
            let (keys, input) = self.capture_render_input(&document, &edits, edits.revision > 0)?;
            // actor可能已发布本代两率结果；同版本重排队不应再争预算或留下伪失败状态。
            let version = PreparedVersion {
                model: document.revision.load(Ordering::Acquire),
                edit: edits.revision,
                epoch: document.render_epoch.load(Ordering::Acquire),
                scope: document.scope_revision.load(Ordering::Acquire),
                keys: keys.clone(),
            };
            if self.prepared.lock().unwrap().as_ref() == Some(&version)
                && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
            {
                return Ok(());
            }
            (
                document.revision.load(Ordering::Acquire),
                edits.revision,
                document.render_epoch.load(Ordering::Acquire),
                document.scope_revision.load(Ordering::Acquire),
                keys,
                input,
            )
        };
        for publisher in &self.snapshots {
            publisher.collect_retired();
        }
        let mut snapshots = input.render(cancel.clone())?;
        super::snapshot::compact_prepared(&mut snapshots)?;
        let _transaction = document.transaction.lock().unwrap();
        if cancel.load(Ordering::Acquire) || !document.ready.load(Ordering::Acquire) {
            return Ok(());
        }
        if document.revision.load(Ordering::Acquire) != model
            || document.edits.lock().unwrap().revision != edit
            || document.render_epoch.load(Ordering::Acquire) != epoch
            || document.scope_revision.load(Ordering::Acquire) != scope
            || self.assigned_regions().map_err(|e| e.to_string())? != keys
        {
            // 跨轨接受新参数不一定有新的model callback；丢弃后补排最新任务，不能永久留空且假idle。
            self.prepare();
            return Ok(());
        }
        if !self
            .snapshots
            .iter()
            .zip(&snapshots)
            .all(|(p, s)| p.has_capacity(s))
        {
            return Err("retired snapshot budget exhausted".into());
        }
        for (publisher, snapshot) in self.snapshots.iter().zip(snapshots) {
            publisher
                .publish(snapshot)
                .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
        }
        self.record_prepared(model, edit, epoch, scope, keys.clone());
        log::info!(
            "[ara] background snapshot ready role={} revision={edit} model={model} regions={}",
            self.role.load(Ordering::Relaxed),
            keys.len()
        );
        Ok(())
    }
}

/// 宿主音频分类的纯函数回归。
///
/// 【为什么必须钉死】这一段决定了 GUI 会给用户哪句话：说错方向（把"挂错轨道"说成
/// "等待中"，或反过来）比不说更坏 —— 用户会按错误的提示去改工程。所以四条分支逐一
/// 断言，并且断言"读不出 folder 结构时**不**给出 folder 结论"。
#[cfg(test)]
mod host_audio_state_tests {
    use super::*;

    #[test]
    fn a_claimed_item_means_ready() {
        let status = classify_host_audio(false, 3, 1);
        assert_eq!(status.state, HostAudioState::Ready);
        assert_eq!(status.waiting_items, 0);
    }

    #[test]
    fn an_empty_project_is_not_a_fault() {
        let status = classify_host_audio(false, 0, 0);
        assert_eq!(status.state, HostAudioState::Ready);
        assert_eq!(status.waiting_items, 0);
    }

    #[test]
    fn items_without_regions_on_a_plain_track_are_awaiting() {
        let status = classify_host_audio(false, 2, 0);
        assert_eq!(status.state, HostAudioState::AwaitingRegions);
        assert_eq!(status.waiting_items, 2);
    }

    #[test]
    fn items_without_regions_on_a_folder_parent_are_named_precisely() {
        let status = classify_host_audio(true, 2, 0);
        assert_eq!(status.state, HostAudioState::FolderParentWithoutRegions);
        assert_eq!(status.waiting_items, 2);
    }

    /// 认领数超过清单数（诊断计数口径不一致）时不得下溢成天文数字。
    #[test]
    fn an_over_counted_claim_does_not_underflow() {
        let status = classify_host_audio(false, 1, 5);
        assert_eq!(status.state, HostAudioState::Ready);
        assert_eq!(status.waiting_items, 0);
    }

    /// 分类名是前端查 catalog 的键：改名即破坏文案，必须显式钉住。
    #[test]
    fn classification_names_are_stable_catalog_keys() {
        assert_eq!(HostAudioState::Ready.as_str(), "ready");
        assert_eq!(HostAudioState::AwaitingRegions.as_str(), "awaiting_regions");
        assert_eq!(
            HostAudioState::FolderParentWithoutRegions.as_str(),
            "folder_parent_without_regions"
        );
    }

    /// GUI 载荷只带**编辑器实例**的读数：playback 实例没有 GUI，它的读数不该出现在
    /// 用户的窗口里（那会让一个正常工作的实例替另一个实例"背锅"）。
    #[test]
    fn the_payload_carries_the_editor_instances_reading() {
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let editor = Arc::new(ExtensionOwner::default());
        let _raw = editor
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER | ExtensionRoles::EDITOR_VIEW,
                None,
            )
            .unwrap();
        assert!(
            editor.is_editor_only(),
            "本夹具的第二个实例必须是编辑器实例"
        );
        // playback 实例报"正常"、编辑器实例报 folder 判据：载荷必须取后者。
        *owners[0].host_audio.lock().unwrap() = HostAudioStatus::default();
        *editor.host_audio.lock().unwrap() = HostAudioStatus {
            state: HostAudioState::FolderParentWithoutRegions,
            waiting_items: 2,
        };
        let mut payload = serde_json::json!({"ok":true});
        document.decorate_host_audio(&mut payload);
        assert_eq!(
            payload["host_audio"]["state"],
            "folder_parent_without_regions"
        );
        assert_eq!(payload["host_audio"]["waiting_clips"], 2);
        document.close();
    }

    /// 没有编辑器实例时**不写**该字段：前端沿用上一次已知值，而不是凭空报一个"正常"。
    #[test]
    fn a_document_without_an_editor_instance_writes_nothing() {
        let (model, _owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let mut payload = serde_json::json!({"ok":true});
        document.decorate_host_audio(&mut payload);
        assert!(payload.get("host_audio").is_none());
        document.close();
    }
}

/// 多 region 几何匹配的纯函数回归。
///
/// 【为什么单独测这一层】`reaper_geometries` 里真正有风险的不是宿主调用，而是
/// "哪条 region 对应哪个 item"的判定：ARA 侧没有这条边，只能按几何比对，而后续
/// 所有宿主写回都沿这条边定位对象。所以唯一性判据必须被钉死。
#[cfg(test)]
mod region_geometry_match_tests {
    use super::*;
    use crate::ara::AraPlaybackRegion;
    use crate::host::geometry::HostClipGeometry;

    fn candidate(
        item: &str,
        start_sec: f64,
        duration_sec: f64,
        source_start_sec: f64,
        playback_rate: f64,
    ) -> HostClipGeometry {
        HostClipGeometry {
            item_id: item.into(),
            take_id: format!("{item}-take"),
            start_sec,
            source_start_sec,
            duration_sec,
            snap_offset_sec: 0.,
            playback_rate,
            preserve_pitch: true,
            channel_mode: 0,
            take_pitch: 0.,
            item_timebase: 0,
            auto_stretch: false,
            muted: false,
            item_gain: 1.,
            group_id: 0,
            take_gain: 1.,
            markers: Vec::new(),
            fade_in_sec: 0.,
            fade_out_sec: 0.,
            fade_in_shape: 0.,
            fade_out_shape: 0.,
            fade_in_dir: 0.,
            fade_out_dir: 0.,
            fade_in_dir_new: 0.,
            fade_out_dir_new: 0.,
            fade_in_dir2_new: 0.,
            fade_out_dir2_new: 0.,
            fade_axes_new: None,
            auto_fade_in_sec: 0.,
            auto_fade_out_sec: 0.,
            loop_source: false,
            source_file_name: None,
        }
    }

    fn region(
        start_in_playback_time: f64,
        duration_in_playback_time: f64,
        start_in_modification_time: f64,
        duration_in_modification_time: f64,
    ) -> AraPlaybackRegion {
        AraPlaybackRegion {
            start_in_playback_time,
            duration_in_playback_time,
            start_in_modification_time,
            duration_in_modification_time,
            ..Default::default()
        }
    }

    /// 四项全相容才绑定：位置、时长、源内起点、速率。
    #[test]
    fn a_region_binds_to_the_item_matching_all_four_quantities() {
        let regions = [
            candidate("item-a", 0., 2., 0., 1.),
            candidate("item-b", 4., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(4., 2., 0., 2.), &regions, None).unwrap();
        assert_eq!(bound.item_id, "item-b");
    }

    /// 同一位置的两条子轨各有一个 item：只比时间窗会歧义，源内起点把两者分开。
    #[test]
    fn the_source_window_disambiguates_items_at_the_same_playback_position() {
        let regions = [
            candidate("trimmed-head", 0., 2., 1., 1.),
            candidate("full", 0., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(0., 2., 1., 2.), &regions, None).unwrap();
        assert_eq!(bound.item_id, "trimmed-head");
    }

    /// 速率也是判据：同位置同时长的两个 item，拉伸比不同。
    #[test]
    fn the_playback_rate_disambiguates_stretched_items() {
        let regions = [
            candidate("stretched", 0., 2., 0., 2.),
            candidate("unstretched", 0., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(0., 2., 0., 4.), &regions, None).unwrap();
        assert_eq!(bound.item_id, "stretched");
    }

    /// 四项完全相同的两个 item ⇒ 歧义 ⇒ **不绑定**（而不是取第一个）。
    #[test]
    fn ambiguous_candidates_are_refused_rather_than_guessed() {
        let regions = [
            candidate("take-one", 0., 2., 0., 1.),
            candidate("take-two", 0., 2., 0., 1.),
        ];
        assert!(unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, None).is_none());
    }

    /// 歧义时，**上一次绑定的那个**候选可以打破它（稳定性，不是猜测）。
    ///
    /// 【为什么这条必须有】复制粘贴出来的安全副本、同一素材的多个 item 会给出几何
    /// 全等的候选。旧实现把它们**全部**判为歧义、两个都丢 —— 用户看到片段凭空消失。
    /// 保持上次的绑定不会把写回引到新对象上：那次绑定本身就来自一次唯一匹配。
    #[test]
    fn a_previous_binding_disambiguates_otherwise_identical_candidates() {
        let regions = [
            candidate("take-one", 0., 2., 0., 1.),
            candidate("take-two", 0., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, Some("take-two"))
            .expect("the previously bound item must win");
        assert_eq!(bound.item_id, "take-two");
        // 上次绑定的那个已经不在候选里时，仍然不猜。
        assert!(
            unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, Some("gone")).is_none()
        );
    }

    /// 没有候选、或候选都对不上时返回 None，绝不退回"第一个 item"。
    #[test]
    fn a_region_without_a_matching_item_stays_unbound() {
        assert!(unique_geometry_for_region(&region(0., 2., 0., 2.), &[], None).is_none());
        let regions = [candidate("elsewhere", 10., 2., 0., 1.)];
        assert!(unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, None).is_none());
    }

    /// 非有限的播放时长不产生"恰好匹配 NaN"的假绑定。
    #[test]
    fn a_region_with_a_degenerate_duration_never_binds() {
        let regions = [candidate("item", 0., 2., 0., 1.)];
        assert!(unique_geometry_for_region(&region(0., 0., 0., 2.), &regions, None).is_none());
        assert!(
            unique_geometry_for_region(&region(0., f64::NAN, 0., 2.), &regions, None).is_none()
        );
    }
}

/// `assigned_timeline` 的保留规则回归。
#[cfg(test)]
mod assigned_timeline_tests {
    use super::*;

    /// 空 folder 轨 / 只有静音 item 的子轨仍须作为参数根存在。
    ///
    /// 【回归】旧实现先按 region 筛 clip，再丢掉**所有**没有 clip 的轨道 —— 于是一个
    /// 被宿主分配了 sequence、当前却没有 clip 的轨道会整条消失，用户看到"轨道没了"。
    /// 这与 `add_group_view_tracks` 的既有口径（空父轨也能承载参数组算法）直接矛盾。
    #[test]
    fn a_sequence_without_clips_still_yields_a_parameter_root() {
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        {
            let mut timeline = document.timeline.lock().unwrap();
            let timeline = timeline.as_mut().unwrap();
            timeline.tracks.push(
                serde_json::from_value(
                    serde_json::json!({"id":"empty","name":"empty folder","order":2}),
                )
                .unwrap(),
            );
            assert_eq!(timeline.tracks.len(), 3);
        }
        let sequence_key = 4242_u64;
        document
            .sequence_regions
            .lock()
            .unwrap()
            .insert(sequence_key, std::collections::HashSet::new());
        document
            .sequence_track_index
            .lock()
            .unwrap()
            .insert(sequence_key, 2);
        owner
            .sequences
            .lock()
            .unwrap()
            .insert(owner.role.load(Ordering::Acquire), vec![sequence_key]);

        let timeline = owner.assigned_timeline(&document).unwrap();
        assert!(
            timeline.tracks.iter().any(|track| track.id == "empty"),
            "被分配 sequence 的空轨道必须留下：{:?}",
            timeline.tracks.iter().map(|t| &t.id).collect::<Vec<_>>()
        );
        assert_eq!(timeline.clips.len(), 1, "空轨道不带来任何 clip");

        // 反向：**没有**被分配 sequence 的空轨道仍然要被剔除，否则等于把整张文档
        // 交给了空 renderer（旧的"零分配只能看零轨道"约束不能被这次改动放宽）。
        document
            .sequence_track_index
            .lock()
            .unwrap()
            .remove(&sequence_key);
        let timeline = owner.assigned_timeline(&document).unwrap();
        assert!(
            !timeline.tracks.iter().any(|track| track.id == "empty"),
            "未分配的空轨道不得留下"
        );
        document.close();
    }
}

/// `reaper_geometries`（多 region 入口）的端到端回归。
///
/// 单 region 走强身份路径，所以这条入口只有在"一个 FX 管一组 region"时才被用到 ——
/// 正是 folder 轨的场景。这里用真实宿主夹具驱动它，覆盖候选收集、逐 region 匹配、
/// 唯一性判据与重复认领丢弃。
#[cfg(test)]
mod reaper_geometries_tests {
    /// 把某条 region 调成与夹具 item 相容的窗口。
    ///
    /// 夹具的 item 几何是 `D_POSITION=1 / D_LENGTH=4 / D_STARTOFFS=0 / D_PLAYRATE=0.5`。
    fn align_with_fixture_item(region: &mut crate::ara::AraPlaybackRegion) {
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 4.;
        region.start_in_modification_time = 0.;
        region.duration_in_modification_time = 2.;
    }

    fn fixture_host() -> Box<crate::host::reaper::ReaperFixture> {
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        host
    }

    /// 一组 region 里能对上的那些被绑定，对不上的保持未绑定（不猜、不兜底）。
    #[test]
    fn only_the_regions_matching_a_host_item_are_bound() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        let second = (&*ids[1] as *const u8) as u64;
        // 夹具只有一个 item，先让它与 first 相容；second 留在原位（对不上）。
        {
            let mut regions = document.regions.lock().unwrap();
            align_with_fixture_item(regions.get_mut(&first).unwrap());
        }
        // 把第二条 region 也分配给同一个 owner：这才是"一个实例管一组 region"。
        let raw = owner.binding.lock().unwrap().as_ref().unwrap().as_raw();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                second as *mut _,
            );
        }
        let host = fixture_host();
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        let bound = owner.reaper_geometries().expect("至少一条 region 必须绑定");
        assert_eq!(bound.len(), 1, "对不上的 region 不得被绑定");
        assert_eq!(bound[0].region_key, first);
        assert!(bound[0].geometry.item_id.starts_with('{'));
        document.close();
    }

    /// 两条 region 都指向同一个 item ⇒ 只保留**一个**认领者，绝不留下两个指向同一
    /// 对象的绑定。
    ///
    /// 【为什么不"两个都丢"】旧实现把重复认领的候选**全部**丢弃，于是"同一位置两个
    /// 几何全等的 item"会让两个 clip 一起消失（用户报障："片段凭空没了"）。现在按
    /// region key 的稳定顺序保留第一个认领者：仍然满足"一个 item 只归一个 region"
    /// （后续写回不会互相覆盖），但不会一次丢掉全部。
    #[test]
    fn two_regions_claiming_one_item_keep_exactly_one_owner() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        let second = (&*ids[1] as *const u8) as u64;
        {
            let mut regions = document.regions.lock().unwrap();
            align_with_fixture_item(regions.get_mut(&first).unwrap());
            align_with_fixture_item(regions.get_mut(&second).unwrap());
        }
        let raw = owner.binding.lock().unwrap().as_ref().unwrap().as_raw();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                second as *mut _,
            );
        }
        let host = fixture_host();
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        let bound = owner
            .reaper_geometries()
            .expect("重复认领必须留下恰好一个绑定，不能整组丢弃");
        assert_eq!(bound.len(), 1, "一个 item 只能归一个 region");
        assert!(
            bound[0].region_key == first || bound[0].region_key == second,
            "保留的必须是其中一个真实认领者"
        );
        document.close();
    }
}
