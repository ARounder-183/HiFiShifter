//! 宿主链接的**接收侧**：打开一个工作区，把编辑写回去。
//!
//! # 它与管道时代那份「连接」的差别
//!
//! 旧实现要「连接到一个活着的实例」：`ara_list_instances` 扫心跳目录、
//! `ara_connect` 走 hello + snapshot 两次往返、会话只在内存里、断开即失去一切。
//! 现在只需要一个**目录**：它可能在宿主里刚被导出，也可能是上一个星期留下的 ——
//! 对接收侧来说没有区别。所以这里没有「在线/离线」，只有「打开」和「提交」。
//!
//! # 音频**不再复制一份**
//!
//! 旧实现把宿主 PCM 解到 `%TEMP%\hifishifter-ara-<uuid>\` 里，随会话一起删。
//! 现在工作区里的 `audio/*.wav` 本来就是物化后的形态（内容寻址、可校验），
//! 所以 `materialize` 的目标目录就是**工作区自己的**音频目录：文件名与内容都命中，
//! 于是一个字节都不用重写（见 `host_pcm::matches_authorized_pcm`）。
//! 这消掉了临时目录、`owned_dirs`、以及「提交时把私有 WAV 路径换回宿主身份」的
//! 反向表 —— 因为回写载荷里根本不再有路径。
//!
//! # 几何归属判据现在在**生产者**侧
//!
//! 消费者只把「我打开时宿主几何长什么样」原样交回去
//! （[`hifishifter_host_link::EditManifest::host_geometry`]），由生产者比对。
//! 消费者自己的活状态（后台的假立体声折叠扫描、波形/音高回填都在写它）
//! 因此再也不可能打穿判据 —— 那个用来压制后台子系统的进程级全局开关也随之删除。

use crate::state::AppState;
use hifishifter_host_link::{
    LinkManifest, SourceKind, Workspace, WorkspaceLock, EDIT_FORMAT, FORMAT_VERSION,
};
use hifishifter_kernel::editor::host_pcm::{
    materialize_with_byte_limit, PcmView, ReversePolicy, MAX_HOST_PCM_BYTES,
};
use hifishifter_kernel::editor::host_timeline::Admission;
use hifishifter_kernel::editor::host_timeline::{
    sanitize_host_timeline, Rejection, SanitizeReport,
};
use hifishifter_kernel::state::TimelineState;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use tauri::Manager;

/// 宿主链接的会话状态。
pub(crate) struct LinkBridge {
    inner: Mutex<Option<LinkSession>>,
}

impl Default for LinkBridge {
    fn default() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }
}

impl LinkBridge {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<LinkSession>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// 一次打开的工作区。
struct LinkSession {
    workspace: Workspace,
    identity: String,
    /// 打开时的清单读数：代次与能力都从这里取，不重新推导。
    manifest: LinkManifest,
    /// 打开时采集的宿主几何基线。提交时原样送回；生产者用它判断宿主有没有变过。
    host_geometry: serde_json::Value,
    /// 打开时的本地工程读数（用于判断"提交之后本地有没有又变了"）。
    project_baseline: serde_json::Value,
    /// 本次导入的逐条降级报告（"跳过了什么、为什么"）。
    report: SanitizeReport,
    /// 工作区的打开标记。持有它 = 声明"这个目录正被这个窗口编辑"；
    /// `Drop` 释放（且只释放自己写下的那一份，见 `WorkspaceLock`）。
    _mark: Option<WorkspaceLock>,
}

// ─── 打开 ─────────────────────────────────────────────────────────────────────

/// 列出可打开的工作区。
///
/// 【为什么不再需要"刷新实例"】工作区在磁盘上，不随宿主进程生灭。
/// 列表就是一次目录扫描，没有心跳、没有幽灵记录、没有"插件窗口开着吗"。
pub(crate) fn list(directory: Option<String>) -> Result<serde_json::Value, String> {
    let root = match directory {
        Some(directory) => std::path::PathBuf::from(directory),
        None => hifishifter_host_link::workspace::default_root()
            .ok_or_else(|| "no data directory to look for link workspaces in".to_string())?,
    };
    let mut items = Vec::new();
    for workspace in hifishifter_host_link::workspace::workspaces_under(&root) {
        // 单个目录读不出来不该让整个列表失败：用户会看到"有东西在那里，但读不了"，
        // 而不是"什么都没有"。
        match workspace.read_manifest() {
            Ok(manifest) => {
                let (complete, partial, external) = manifest.audio_summary();
                items.push(serde_json::json!({
                    "directory": workspace.root().display().to_string(),
                    "identity": manifest.link_id,
                    "hostLabel": manifest.host_label,
                    "sourceKind": manifest.source_kind.tag(),
                    "createdAt": manifest.created_unix_ms,
                    "clips": manifest.clip_count(),
                    "tracks": manifest.track_count(),
                    "audio": { "complete": complete, "partial": partial, "extern": external },
                    "losses": manifest.losses.len(),
                    "acceptsCommit": manifest.grants(hifishifter_host_link::Capability::Commit),
                }));
            }
            Err(error) => items.push(serde_json::json!({
                "directory": workspace.root().display().to_string(),
                "unreadable": error,
            })),
        }
    }
    Ok(serde_json::json!({ "root": root.display().to_string(), "items": items }))
}

pub(crate) fn open(
    app: &tauri::AppHandle,
    directory: String,
    force: bool,
) -> Result<serde_json::Value, String> {
    let state = app.state::<AppState>();
    guard_replace(&state, force)?;
    // 作废在途的假立体声折叠扫描：clip/take id 在会话之间会复用，旧工程的折叠结论
    // 若落进刚导入的宿主时间线，会改写 `take.channel_mode`（宿主拥有的字段）。
    crate::commands::channel_scan::bump_generation();
    let version = state.timeline_version.load(Ordering::Acquire);

    let workspace = Workspace::at(directory);
    let manifest = workspace.read_manifest()?;
    let mark = match workspace.acquire_open_mark(Some("HiFiShifter"), 12 * 60 * 60 * 1000) {
        Ok(mark) => Some(mark),
        // 已有窗口开着它：**这不是错误**，但两个窗口同时写 `edits.hfsedit` 会让
        // 后一次提交覆盖前一次。如实说出来，让用户自己决定。
        Err(error) => return Err(format!("this workspace is already open: {error}")),
    };

    // 准入策略由**来源**决定，`prepare_import` 内部按 `source_kind` 取用。
    let prepared = prepare_import(&manifest, &workspace)?;
    if !prepared.report.is_clean() {
        // 用户报障时最需要的一行事实：跳过了什么、为什么。
        log::warn!("[host-link] import degraded: {}", prepared.report.summary());
    }
    let mut timeline = prepared.timeline;
    timeline.playhead_sec = 0.0;
    timeline.selected_track_id = timeline.tracks.first().map(|track| track.id.clone());
    timeline.selected_clip_id = timeline.clips.first().map(|clip| clip.id.clone());
    // 【为什么不在这里重算】基线由生产者算好放进清单，消费者**原样回送**。
    // 自己重算会让任何浮点归一化差异变成一次假漂移（见 `LinkManifest::host_geometry`）。
    let host_geometry = manifest.host_geometry.clone();

    let bridge = app.state::<LinkBridge>();
    let mut inner = bridge.lock();
    let project_baseline;
    {
        let mut current = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        if state.timeline_version.load(Ordering::Acquire) != version || (project.dirty && !force) {
            return Err(
                "the local project changed while the workspace was being read; retry".into(),
            );
        }
        state.audio_engine.stop();
        for clip in &current.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        crate::pitch_clip::clear_pitch_analysis_state();
        *current = timeline;
        let recent = std::mem::take(&mut project.recent);
        *project = crate::state::ProjectState::default();
        project.recent = recent;
        project.name = format!(
            "{} / {}",
            prefix_for(manifest.source_kind),
            manifest.host_label
        );
        project_baseline = project_projection(&project);
        state.audio_engine.update_timeline(current.clone());
        state.bump_timeline_version();
    }
    state.clear_history();
    state.reset_notebook_assets();
    crate::render_cache::set_current_project_id(0);

    let identity = manifest.link_id.clone();
    let revisions = manifest.revisions;
    let session = LinkSession {
        workspace,
        identity,
        manifest,
        host_geometry,
        project_baseline,
        report: prepared.report,
        _mark: mark,
    };
    let session_payload_open = session_payload(&session);
    inner.replace(session);
    let roots: Vec<_> = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .tracks
        .iter()
        .filter(|track| track.parent_id.is_none())
        .map(|track| track.id.clone())
        .collect();
    for root in roots {
        crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, &root);
    }
    let mut payload = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .to_payload();
    payload.project = Some(state.project_meta_payload());
    let mut result = session_payload_open;
    result["timeline"] = serde_json::to_value(payload).map_err(|e| e.to_string())?;
    log::info!(
        "[host-link] opened {} clips / {} tracks (revisions {:?})",
        result["clips"].as_u64().unwrap_or_default(),
        result["tracks"].as_u64().unwrap_or_default(),
        revisions
    );
    Ok(result)
}

fn prefix_for(source_kind: SourceKind) -> &'static str {
    match source_kind {
        SourceKind::Ara => "ARA",
        SourceKind::Capture => "Capture",
        SourceKind::Bridge => "Host",
    }
}

/// 导入的**纯数据**部分：读音频、摘掉没有音频的片段、准入校验、物化。
///
/// 【为什么单独抽出来】它不碰 `AppState`，因此可以独立单测 —— 而这一层恰恰是最
/// 容易出错的一层（素材状态、准入策略、摘要校验都在这儿）。
fn prepare_import(
    manifest: &LinkManifest,
    workspace: &Workspace,
) -> Result<PreparedImport, String> {
    let mut timeline: TimelineState = serde_json::from_value(manifest.timeline.clone())
        .map_err(|error| format!("decode host timeline: {error}"))?;
    let mut report = SanitizeReport::default();

    // 1) 有音频的素材：读进来并**逐字节校验摘要**（工作区侧的 `read_audio` 负责）。
    //    摘要不符是明确失败，绝不"凑合读" —— 否则会把别的音频当成用户的编辑存进工程。
    let mut pending: Vec<(String, u32, Vec<Vec<f32>>)> = Vec::new();
    for entry in &manifest.sources {
        if !entry.state.requires_file() {
            continue;
        }
        let audio = workspace.read_audio(entry)?;
        pending.push((entry.key.clone(), audio.sample_rate, audio.planes));
    }
    let known: std::collections::HashSet<&str> =
        pending.iter().map(|(key, _, _)| key.as_str()).collect();

    // 2) 音频暂缺的素材 ⇒ 引用它的片段本次进不来。
    //
    // 【为什么不是静默丢弃】`SanitizeReport` 会带上 clip id 与原因，面板据此说清
    // **是什么**没进来。旧实现只报一个条数，用户无法据以行动。
    //
    // 【为什么不做成"保留几何、只空着音频槽"】那样需要渲染管线接受
    // "已声明但无音频"的片段；今天它只有"有源"与"无源占位"两种表示，
    // 而后者正是倒放隔离用的那条路。这一项记在方案的实施记录里，是**有意留下**的边界。
    let mut kept = Vec::with_capacity(timeline.clips.len());
    for clip in std::mem::take(&mut timeline.clips) {
        let references: Vec<&str> = std::iter::once(clip.source_path.as_deref())
            .chain(clip.takes.iter().map(|take| take.source_path.as_deref()))
            .flatten()
            .collect();
        let missing = references.iter().find(|key| !known.contains(**key));
        match missing {
            None => kept.push(clip),
            Some(key) => report.dropped_clips.push(Rejection {
                id: clip.id.clone(),
                reason: format!("the host has not provided {key}'s audio yet"),
            }),
        }
    }
    timeline.clips = kept;

    // 3) 准入校验（规则与插件侧同源）。
    let mut views: Vec<PcmView<'_>> = pending
        .iter()
        .map(|(key, rate, planes)| PcmView {
            persistent_id: key.as_str(),
            sample_rate: *rate,
            planes,
        })
        .collect();
    let sanitize = sanitize_host_timeline(
        &mut timeline,
        &mut views,
        MAX_HOST_PCM_BYTES,
        admission_of(manifest.source_kind),
    )?;
    merge_report(&mut report, sanitize);

    // 4) 物化**进工作区自己的音频目录**。
    //
    // 【为什么目标就是工作区】那里的文件已经是同一份音频（内容寻址、名字里带着
    // persistent_id + 采样率 + 声道 + 每个采样），`materialize` 一看就命中并跳过写出。
    // 于是"导入"不再复制任何音频，也不再需要随会话删除的临时目录。
    let (timeline, _reverse) = materialize_with_byte_limit(
        timeline,
        &views,
        &workspace.audio_dir(),
        MAX_HOST_PCM_BYTES,
        // 【为什么 ARA 也用 `Materialize` 而不是 `Reject`】倒放隔离是插件侧渲染路径的
        // 需要；接收侧拿到的是宿主已经表达过的方向，交给下游消费即可。
        ReversePolicy::Materialize,
    )?;
    Ok(PreparedImport { timeline, report })
}

#[derive(Debug)]
struct PreparedImport {
    timeline: TimelineState,
    report: SanitizeReport,
}

fn admission_of(source_kind: SourceKind) -> Admission {
    match source_kind {
        SourceKind::Ara => Admission::Strict,
        SourceKind::Capture | SourceKind::Bridge => Admission::Degrade,
    }
}

fn merge_report(into: &mut SanitizeReport, extra: SanitizeReport) {
    into.dropped_sources.extend(extra.dropped_sources);
    into.dropped_clips.extend(extra.dropped_clips);
    into.remapped_clips.extend(extra.remapped_clips);
}

// ─── 提交 ─────────────────────────────────────────────────────────────────────

/// 把当前的参数编辑写成 `edits.hfsedit`，并**尽力**当场拿到回答。
///
/// 【为什么不只是"发出去就完事"】宿主开着的时候用户期望点一下就得到反馈。
/// 所以在写完之后短暂轮询状态文件（宿主侧 actor 最多 100 ms 一轮），
/// 拿到了就如实报告；没拿到就说"已提交，等宿主接收" —— 而不是假装成功。
pub(crate) fn submit(app: &tauri::AppHandle) -> Result<serde_json::Value, String> {
    let state = app.state::<AppState>();
    let bridge = app.state::<LinkBridge>();
    //
    // 【为什么分三段持锁】旧实现把 bridge 锁横跨 write_edit + 20×100ms 状态轮询
    // （最长 2 秒）持有 —— 并发的 refresh / release 全被串到 2 秒后。锁内只取
    // 构建编辑所需字段；文件 IO 与轮询放锁外；最后短暂拿锁把回执落进会话。
    let (identity, base_revisions, host_geometry, root, project_baseline_before) = {
        let inner = bridge.lock();
        let session = inner.as_ref().ok_or("no host workspace is open")?;
        if !session
            .manifest
            .grants(hifishifter_host_link::Capability::Commit)
        {
            return Err("this host does not accept edits".into());
        }
        (
            session.identity.clone(),
            session.manifest.revisions,
            session.host_geometry.clone(),
            session.workspace.root().to_path_buf(),
            session.project_baseline.clone(),
        )
    };
    let version = state.timeline_version.load(Ordering::Acquire);
    let (params, tracks) = {
        let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        (
            serde_json::to_value(&timeline.params_by_root_track)
                .map_err(|error| error.to_string())?,
            serde_json::to_value(&timeline.tracks).map_err(|error| error.to_string())?,
        )
    };
    let edit = hifishifter_host_link::EditManifest {
        format: EDIT_FORMAT.into(),
        format_version: FORMAT_VERSION,
        link_id: identity,
        base_revisions,
        host_geometry,
        params,
        tracks,
    };
    // 锁外的文件 IO：记录提交前状态（判断"回答是不是针对这一次提交的"），
    // 写编辑，再短暂轮询宿主回执。
    let workspace = hifishifter_host_link::Workspace::at(root);
    let before_stamp = workspace
        .read_status()
        .as_ref()
        .map(|status| status.updated_unix_ms);
    workspace.write_edit(&edit)?;

    let mut status = None;
    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let current = workspace.read_status();
        let changed = current
            .as_ref()
            .is_some_and(|status| Some(status.updated_unix_ms) != before_stamp);
        if changed {
            status = current;
            break;
        }
    }
    let project_baseline = {
        let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        project_projection(&project)
    };
    let params_sent = {
        let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::to_value(&timeline.params_by_root_track).map_err(|e| e.to_string())?
    };
    let accepted = status.as_ref().is_some_and(|status| status.ok);
    if accepted
        && state.timeline_version.load(Ordering::Acquire) == version
        && project_baseline == project_baseline_before
        && params_sent == edit.params
    {
        // 只有"送出去的就是现在的"才清 dirty（与旧实现同一条纪律：
        // 不能依赖撤销版本，必须在时间轴锁内确认）。
        let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        project.dirty = false;
    }
    // 回执落账必须确认会话没有被并发的 refresh/release 换掉：identity 一致
    // 才把新代次写回（否则就是把旧会话的回执记到新会话头上）。
    let mut inner = bridge.lock();
    let session = inner.as_mut().ok_or("no host workspace is open")?;
    if session.identity != edit.link_id {
        return Err("host workspace changed while submitting".into());
    }
    if let Some(status) = status.as_ref().filter(|status| status.ok) {
        session.manifest.revisions = status.revisions;
        // 宿主接受了这次编辑：代次前进了，所以下一次提交要拿新的代次做基线。
    }
    let mut payload = session_payload(session);
    payload["applied"] = serde_json::json!(status.as_ref().is_some_and(|status| status.ok));
    payload["pending"] = serde_json::json!(status.is_none());
    payload["status"] = match status {
        Some(status) => serde_json::json!({
            "ok": status.ok,
            "code": status.code,
            "error": status.error,
            "atUnixMs": status.updated_unix_ms,
        }),
        None => serde_json::Value::Null,
    };
    Ok(payload)
}

/// 重新读一次工作区的清单（宿主里改过东西之后）。
///
/// 【为什么保留本地的参数编辑】宿主改的是几何，用户改的是曲线。重新导入几何时把
/// **仍然存在于新时间轴上的**轨道参数搬过去，用户就不用因为"在宿主里挪了一个片段"
/// 而丢掉全部曲线。搬不过去的（轨道没了）如实报出来。
pub(crate) fn refresh(app: &tauri::AppHandle, force: bool) -> Result<serde_json::Value, String> {
    let state = app.state::<AppState>();
    let bridge = app.state::<LinkBridge>();
    let directory = {
        let inner = bridge.lock();
        inner
            .as_ref()
            .ok_or("no host workspace is open")?
            .workspace
            .root()
            .display()
            .to_string()
    };
    let carried = {
        let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        (
            timeline.params_by_root_track.clone(),
            timeline.tracks.clone(),
        )
    };
    let mut result = open(app, directory, force)?;
    let (carried_params, carried_tracks) = {
        let mut timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let known: std::collections::HashSet<String> = timeline
            .tracks
            .iter()
            .map(|track| track.id.clone())
            .collect();
        let (params, tracks) = carried;
        let mut carried_params = 0_usize;
        for (id, params) in params {
            if known.contains(&id) {
                timeline.params_by_root_track.insert(id, params);
                carried_params += 1;
            }
        }
        let mut carried_tracks = 0_usize;
        for track in &mut timeline.tracks {
            if let Some(edited) = tracks.iter().find(|known| known.id == track.id) {
                track.volume = edited.volume;
                track.muted = edited.muted;
                track.solo = edited.solo;
                track.compose_enabled = edited.compose_enabled;
                track.pitch_analysis_algo = edited.pitch_analysis_algo.clone();
                carried_tracks += 1;
            }
        }
        (carried_params, carried_tracks)
    };
    state.bump_timeline_version();
    // 【为什么要如实报出"搬过去了多少"】重新导入会重建时间轴，用户最怕的是
    // "我的曲线是不是没了"。搬过去的条数是**事实**，说出来比让他自己数可靠。
    result["carriedParams"] = serde_json::json!(carried_params);
    result["carriedTracks"] = serde_json::json!(carried_tracks);
    Ok(result)
}

pub(crate) fn release(app: &tauri::AppHandle) -> serde_json::Value {
    let bridge = app.state::<LinkBridge>();
    *bridge.lock() = None;
    serde_json::json!({ "ok": true, "attached": false })
}

// ─── 载荷 ─────────────────────────────────────────────────────────────────────

fn session_payload(session: &LinkSession) -> serde_json::Value {
    let manifest = &session.manifest;
    let (complete, partial, external) = manifest.audio_summary();
    serde_json::json!({
        "attached": true,
        "directory": session.workspace.root().display().to_string(),
        "identity": session.identity,
        "hostLabel": manifest.host_label,
        "sourceKind": manifest.source_kind.tag(),
        "createdAt": manifest.created_unix_ms,
        "clips": manifest.clip_count(),
        "tracks": manifest.track_count(),
        "audio": { "complete": complete, "partial": partial, "extern": external },
        "acceptsCommit": manifest.grants(hifishifter_host_link::Capability::Commit),
        "gatesGeometry": !manifest.grants(hifishifter_host_link::Capability::GeometryWrite),
        "edits": manifest.revisions.edits,
        "model": manifest.revisions.model,
        "skipped": session.report.summary(),
        "skippedCount": session.report.dropped_sources.len()
            + session.report.dropped_clips.len()
            + session.report.remapped_clips.len(),
        "skippedDetail": session
            .report
            .dropped_clips
            .iter()
            .chain(session.report.dropped_sources.iter())
            .take(8)
            .map(|item| format!("{} ({})", item.id, item.reason))
            .collect::<Vec<_>>(),
        "losses": manifest.losses.iter().take(8).map(|loss| format!("{} ({})", loss.id, loss.reason)).collect::<Vec<_>>(),
        "unsupportedPrecision": false,
    })
}

/// 拒绝替换一个还有未保存改动的本地工程。
fn guard_replace(state: &AppState, force: bool) -> Result<(), String> {
    if force {
        return Ok(());
    }
    let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
    if project.dirty {
        return Err("dirty_project: the current project has unsaved changes".into());
    }
    Ok(())
}

/// 本地工程读数；提交时用它确认"提交之后本地有没有又变了"。
fn project_projection(project: &crate::state::ProjectState) -> serde_json::Value {
    serde_json::json!({
        "name": project.name, "path": project.path, "notes": project.notes_markdown,
        "assets": project.notebook_assets, "base_scale": project.base_scale,
        "use_custom_scale": project.use_custom_scale, "custom_scale": project.custom_scale,
        "beats_per_bar": project.beats_per_bar, "denominator": project.time_signature_denominator,
        "grid_size": project.grid_size, "stretch": project.stretch_algorithm_override,
        "mel_stretch": project.hifigan_mel_stretch_override, "save_undo": project.save_undo_history
    })
}

#[cfg(test)]
mod tests;
