//! ARA GUI 桥接：仅下载宿主授权 PCM，交由既有波形和音高管线处理。

use crate::state::{AppState, TimelineState};
use hifishifter_ara_ipc::{HostPcm, InstanceRecord, Request, Response};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Mutex},
};
use tauri::Manager;

/// 门禁拒绝时最多报告多少条字段差异；够用户定位即可，不刷屏。
const HOST_GEOMETRY_DIFF_LIMIT: usize = 8;

/// 仅在 Windows 默认临时目录权限拒绝时回退，不修改目录或系统完整性标签。
fn create_ara_temp_dir_with(
    primary: &Path,
    fallback: Option<&Path>,
    create: impl Fn(&Path) -> std::io::Result<()>,
) -> Result<PathBuf, String> {
    match create(primary) {
        Ok(()) => Ok(primary.to_path_buf()),
        Err(error) if cfg!(windows) && error.kind() == std::io::ErrorKind::PermissionDenied => {
            let fallback = fallback.ok_or_else(|| {
                format!(
                    "create ARA temporary directory {}: {error}; LocalLow unavailable",
                    primary.display()
                )
            })?;
            create(fallback).map_err(|fallback_error| format!(
                "create ARA LocalLow directory {} after temporary directory permission denied: {fallback_error}",
                fallback.display()
            ))?;
            Ok(fallback.to_path_buf())
        }
        Err(error) => Err(format!(
            "create ARA temporary directory {}: {error}",
            primary.display()
        )),
    }
}

/// Low GUI 的私有 PCM 临时空间；显式 TEMP/TMP 可写时沿用原路径。
fn create_ara_temp_dir() -> Result<PathBuf, String> {
    let id = format!("hifishifter-ara-{}", uuid::Uuid::new_v4());
    let primary = std::env::temp_dir().join(&id);
    let fallback = if cfg!(windows) {
        std::env::var_os("USERPROFILE").map(|profile| {
            PathBuf::from(profile)
                .join("AppData/LocalLow/HiFiShifter/ara")
                .join(&id)
        })
    } else {
        None
    };
    create_ara_temp_dir_with(&primary, fallback.as_deref(), |path| {
        std::fs::create_dir_all(path)
    })
}

/// 写入应用私有 WAV，先验证全部源，绝不将 persistentID 当路径打开。
fn materialize_snapshot(
    timeline: TimelineState,
    sources: &[HostPcm],
    dir: &Path,
) -> Result<(TimelineState, HashMap<String, String>), String> {
    let views: Vec<_> = sources
        .iter()
        .map(|pcm| hifishifter_kernel::editor::host_pcm::PcmView {
            persistent_id: &pcm.persistent_id,
            sample_rate: pcm.sample_rate,
            planes: &pcm.planes,
        })
        .collect();
    hifishifter_kernel::editor::host_pcm::materialize(timeline, &views, dir)
}

fn guard_replace(state: &AppState, force: bool) -> Result<(), String> {
    if state
        .project
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .dirty
        && !force
    {
        Err("dirty_project: confirm replacement of unsaved edits".into())
    } else {
        Ok(())
    }
}

struct Session {
    instance: InstanceRecord,
    revision: u64,
    model_revision: u64,
    reverse_paths: HashMap<String, String>,
    clip_ids: HashSet<String>,
    /// 连接时采集的宿主几何投影；提交时用它判断本地是否改过宿主拥有的字段。
    host_geometry_baseline: serde_json::Value,
    project_baseline: serde_json::Value,
}

/// 提交门禁的比较对象：**正向**列出宿主拥有、且本地不该改的字段。
///
/// 【为什么是正向名单，而不是"整份状态减掉几个例外"】
/// 旧判据把整个 `TimelineState` 序列化后手工挖掉十几项，等于用"整份状态相等"
/// 表达"这几个字段不许变"。但 `TimelineState` 是独立 App 的活状态：后台子系统
/// （假立体声折叠扫描、波形/音高分析、渲染缓存回填）会在用户**没有做任何编辑**
/// 时写它。只要写到的字段不在例外名单里，提交就被拒 —— 而报错文案固定指控
/// "几何/增益/名字"，与实际原因无关，用户无法据以行动。
///
/// 正向名单把判据变成"我知道宿主拥有哪些字段，只比这些"。新增任何后台子系统
/// 都不会再打穿它，而真正需要拒绝的本地几何编辑仍然会被拒绝。
///
/// 【为什么 `bpm` / `project_sec` 不在名单里】
/// 二者都由宿主单方面决定：插件每次快照都从宿主时钟重读 `bpm`
/// （`extension.rs::assigned_timeline`），并重算 `project_sec`
/// （`editor/workspace.rs`）。本地改动既提交不上去，也不代表"未保存的宿主编辑"，
/// 放进判据只会制造假失败。
fn host_geometry_projection(timeline: &TimelineState) -> Result<serde_json::Value, String> {
    let mut normalized = timeline.clone();
    // 与提交载荷走同一步：先让 Take 与扁平投影一致，再读扁平值，避免"改了 Take
    // 没改投影"或反之造成的判据抖动。
    normalized.sync_clip_takes_from_flat();
    let clips: Vec<_> = normalized
        .clips
        .iter()
        .map(|clip| {
            serde_json::json!({
                "id": clip.id,
                "track_id": clip.track_id,
                "name": clip.name,
                "start_sec": clip.start_sec,
                "length_sec": clip.length_sec,
                "source_start_sec": clip.source_start_sec,
                "source_end_sec": clip.source_end_sec,
                "playback_rate": clip.playback_rate,
                "clip_playback_rate": clip.clip_playback_rate,
                "gain": clip.gain,
                "muted": clip.muted,
                "snap_offset_sec": clip.snap_offset_sec,
                "fade_in_sec": clip.fade_in_sec,
                "fade_out_sec": clip.fade_out_sec,
                "fade_in_shape": clip.fade_in_shape,
                "fade_out_shape": clip.fade_out_shape,
                "fade_in_dir": clip.fade_in_dir,
                "fade_out_dir": clip.fade_out_dir,
                "auto_fade_in_sec": clip.auto_fade_in_sec,
                "auto_fade_out_sec": clip.auto_fade_out_sec,
            })
        })
        .collect();
    let tracks: Vec<_> = normalized
        .tracks
        .iter()
        .map(|track| {
            serde_json::json!({
                "id": track.id,
                "name": track.name,
                "order": track.order,
                "parent_id": track.parent_id,
            })
        })
        .collect();
    Ok(serde_json::json!({ "clips": clips, "tracks": tracks }))
}

/// 把 JSON 值渲染成一行短摘要；长曲线/长字符串必须截断，否则日志自己就爆了。
fn compact_json_value(value: &serde_json::Value) -> String {
    let text = value.to_string();
    if text.chars().count() <= 64 {
        return text;
    }
    let mut truncated: String = text.chars().take(61).collect();
    truncated.push_str("...");
    truncated
}

/// 逐路径收集两棵 JSON 的差异，用于把"到底哪一项漂移了"讲清楚。
///
/// 门禁拒绝时用户唯一能据以行动的信息就是这份路径列表：说"几何变了"没有用，
/// 说 `clips[2].start_sec: 0.0 -> 0.25` 才有用。
fn collect_json_differences(
    baseline: &serde_json::Value,
    current: &serde_json::Value,
    path: &str,
    limit: usize,
    out: &mut Vec<String>,
) {
    if out.len() >= limit {
        return;
    }
    match (baseline, current) {
        (serde_json::Value::Object(before), serde_json::Value::Object(after)) => {
            for (key, value) in before {
                if out.len() >= limit {
                    return;
                }
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                match after.get(key) {
                    Some(other) => collect_json_differences(value, other, &child, limit, out),
                    None => out.push(format!("{child}: removed")),
                }
            }
            for key in after.keys() {
                if out.len() >= limit {
                    return;
                }
                if !before.contains_key(key) {
                    let child = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    out.push(format!("{child}: added"));
                }
            }
        }
        (serde_json::Value::Array(before), serde_json::Value::Array(after)) => {
            if before.len() != after.len() {
                out.push(format!(
                    "{path}: length {} -> {}",
                    before.len(),
                    after.len()
                ));
                return;
            }
            for (index, (x, y)) in before.iter().zip(after.iter()).enumerate() {
                if out.len() >= limit {
                    return;
                }
                collect_json_differences(x, y, &format!("{path}[{index}]"), limit, out);
            }
        }
        _ => {
            if baseline != current {
                out.push(format!(
                    "{path}: {} -> {}",
                    compact_json_value(baseline),
                    compact_json_value(current)
                ));
            }
        }
    }
}

/// 本地工程信息没有进入插件state，备注/设置/独立保存路径仍须保护未保存提示。
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

/// 与实际IPC载荷一致的受支持投影；包含所有曲线/静态参数及被插件接受的轨道控制。
fn supported_projection(timeline: &serde_json::Value) -> Result<serde_json::Value, String> {
    let tracks = timeline
        .get("tracks")
        .and_then(serde_json::Value::as_array)
        .ok_or("missing submitted ARA tracks")?;
    let controls: Vec<_>=tracks.iter().map(|track| serde_json::json!({
        "id":track["id"], "volume":track["volume"], "muted":track["muted"], "solo":track["solo"],
        "compose_enabled":track["compose_enabled"], "pitch_analysis_algo":track["pitch_analysis_algo"]
    })).collect();
    Ok(
        serde_json::json!({"params":timeline.get("params_by_root_track").cloned().unwrap_or_else(||serde_json::json!({})),"tracks":controls}),
    )
}

/// 清dirty须在timeline锁内确认当前支持参数仍等于实际送出的参数，不能依赖undo版本。
fn clear_committed_parameter_dirty(
    state: &AppState,
    session: &Session,
    submitted_version: u64,
    submitted_parameters: &serde_json::Value,
) {
    let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    if state.timeline_version.load(Ordering::Acquire) != submitted_version
        || host_geometry_projection(&timeline).ok().as_ref()
            != Some(&session.host_geometry_baseline)
    {
        return;
    }
    let current_parameters = serde_json::to_value(&*timeline)
        .ok()
        .and_then(|timeline| supported_projection(&timeline).ok());
    if current_parameters.as_ref() != Some(submitted_parameters) {
        return;
    }
    let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
    if project_projection(&project) == session.project_baseline {
        project.dirty = false;
    }
}

/// 保留下载 WAV 到应用退出，避免断开后现有工程或分析线程失去音频。
#[derive(Default)]
pub(crate) struct AraBridge {
    inner: Mutex<BridgeInner>,
}

#[derive(Default)]
struct BridgeInner {
    session: Option<Session>,
    owned_dirs: Vec<PathBuf>,
}

impl Drop for BridgeInner {
    fn drop(&mut self) {
        for dir in &self.owned_dirs {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn check_response(response: &Response) -> Result<(), String> {
    if response.ok {
        Ok(())
    } else {
        Err(response
            .error
            .clone()
            .unwrap_or_else(|| "ARA request failed".into()))
    }
}

fn session_payload(session: &Session) -> serde_json::Value {
    serde_json::json!({ "ok": true, "instance_id": session.instance.instance_id,
        "revision": session.revision, "model_revision": session.model_revision })
}

fn apply_commit_response(session: &mut Session, response: &Response) -> Result<(), String> {
    check_response(response)?;
    session.revision = response.revision;
    session.model_revision = response.model_revision;
    Ok(())
}

/// 提交时恢复宿主源身份，拒绝把独立文件工程意外提交到仍连接的插件。
///
/// 门禁只比较宿主拥有的几何字段（见 [`host_geometry_projection`]）；参数曲线与
/// 轨道合成控制是本次提交的内容，不算"本地未保存的宿主编辑"。
fn build_commit(session: &Session, mut timeline: TimelineState) -> Result<Request, String> {
    let geometry = host_geometry_projection(&timeline)?;
    if geometry != session.host_geometry_baseline {
        let mut differences = Vec::new();
        collect_json_differences(
            &session.host_geometry_baseline,
            &geometry,
            "",
            HOST_GEOMETRY_DIFF_LIMIT,
            &mut differences,
        );
        log::warn!(
            "[ara] commit refused: host-owned geometry changed locally: {}",
            differences.join("; ")
        );
        // 语言无关的字段路径 + 分类；文案由前端按 catalog 本地化。
        return Err(format!("ara_host_fields: {}", differences.join("; ")));
    }
    let ids: HashSet<_> = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
    if ids != session.clip_ids {
        // 语言无关的分类；文案由前端按 catalog 本地化。
        return Err("ara_reconnect_required".into());
    }
    timeline.sync_clip_takes_from_flat();
    for clip in &mut timeline.clips {
        for take in &mut clip.takes {
            let path = take.source_path.as_ref().ok_or("missing ARA source")?;
            take.source_path = Some(
                session
                    .reverse_paths
                    .get(path)
                    .ok_or("source is not owned by ARA session")?
                    .clone(),
            );
            take.source_path_relative = None;
            take.source_file_fingerprint = None;
            take.waveform_preview = None;
            take.pitch_range = None;
        }
        clip.normalize_takes();
    }
    Ok(Request::Commit {
        base_revision: session.revision,
        model_revision: session.model_revision,
        timeline: serde_json::to_value(timeline).map_err(|e| e.to_string())?,
    })
}

/// 连接和刷新采用同一原子导入路径，下载期间发生本地编辑则拒绝覆盖。
///
/// 会话建立前后要动两处进程级状态：作废在途的声道扫描，并在会话存续期间抑制
/// 新的扫描请求 —— 两者都是为了不让本地后台写入改到宿主拥有的字段。
pub(crate) fn import_snapshot(
    app: &tauri::AppHandle,
    instance_id: Option<String>,
    force: bool,
) -> Result<serde_json::Value, String> {
    // 作废在途扫描：clip/take id 在会话之间会复用，旧工程的折叠结论若落进刚
    // 下载的 ARA 时间线，会改写 `take.channel_mode`，让宿主几何基线漂移、
    // 提交被门禁拒绝。理由与 `open_project` 完全一致。
    crate::commands::channel_scan::bump_generation();
    // 下载窗口内就抑制新的扫描请求（见 `channel_scan::set_ara_session_active`）。
    crate::commands::channel_scan::set_ara_session_active(true);
    let result = import_snapshot_inner(app, instance_id, force);
    // 只有真正建立了会话才保持抑制；失败或刷新失败（沿用旧会话）时如实反映。
    let connected = app
        .state::<AraBridge>()
        .inner
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .session
        .is_some();
    crate::commands::channel_scan::set_ara_session_active(connected);
    result
}

fn import_snapshot_inner(
    app: &tauri::AppHandle,
    instance_id: Option<String>,
    force: bool,
) -> Result<serde_json::Value, String> {
    let state = app.state::<AppState>();
    guard_replace(&state, force)?;
    let version = state.timeline_version.load(Ordering::Acquire);
    let bridge = app.state::<AraBridge>();
    let mut inner = bridge.inner.lock().unwrap_or_else(|e| e.into_inner());
    let instance = match instance_id {
        Some(id) => hifishifter_ara_ipc::discover()?
            .into_iter()
            .find(|record| record.instance_id == id)
            .ok_or("ARA instance unavailable")?,
        None => inner
            .session
            .as_ref()
            .ok_or("ARA is disconnected")?
            .instance
            .clone(),
    };
    let response = hifishifter_ara_ipc::exchange(&instance, &Request::Snapshot)
        .map_err(|e| format!("ARA snapshot IPC: {e}"))?;
    check_response(&response)?;
    let timeline: TimelineState =
        serde_json::from_value(response.timeline.ok_or("missing ARA timeline")?)
            .map_err(|e| format!("decode ARA snapshot timeline: {e}"))?;
    let dir = create_ara_temp_dir()?;
    let (mut timeline, reverse_paths) =
        match materialize_snapshot(timeline, &response.sources, &dir) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(format!("materialize ARA snapshot PCM: {error}"));
            }
        };
    inner.owned_dirs.push(dir);
    timeline.playhead_sec = 0.0;
    timeline.selected_track_id = timeline.tracks.first().map(|track| track.id.clone());
    timeline.selected_clip_id = timeline.clips.first().map(|clip| clip.id.clone());
    let clip_ids = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
    let host_geometry_baseline = host_geometry_projection(&timeline)?;
    let project_baseline;
    {
        let mut current = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        if state.timeline_version.load(Ordering::Acquire) != version || (project.dirty && !force) {
            return Err("local project changed during ARA download; retry".into());
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
        project.name = format!("ARA / {}", instance.name);
        project_baseline = project_projection(&project);
        state.audio_engine.update_timeline(current.clone());
        state.bump_timeline_version();
    }
    state.clear_history();
    state.reset_notebook_assets();
    crate::render_cache::set_current_project_id(0);
    inner.session = Some(Session {
        instance,
        revision: response.revision,
        model_revision: response.model_revision,
        reverse_paths,
        clip_ids,
        host_geometry_baseline,
        project_baseline,
    });
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
    let mut result = session_payload(inner.session.as_ref().unwrap());
    result["timeline"] = serde_json::to_value(payload).map_err(|e| e.to_string())?;
    // 用户报障时最需要的一行事实：连接后的规模、宿主修订号，以及本地声道判定
    // 档案的存量（后者是过去"提交莫名被拒"的头号嫌疑）。
    if let Some(session) = inner.session.as_ref() {
        let decided = {
            let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            timeline
                .clips
                .iter()
                .flat_map(|clip| clip.takes.iter())
                .filter(|take| take.channel_decision.is_some())
                .count()
        };
        log::info!(
            "[ara] connected: instance={} revision={} model_revision={} clips={} tracks={} decided_takes={decided}",
            session.instance.name,
            session.revision,
            session.model_revision,
            session.clip_ids.len(),
            session.host_geometry_baseline["tracks"]
                .as_array()
                .map(Vec::len)
                .unwrap_or(0),
        );
    }
    Ok(result)
}

/// 提交仅转换私有 WAV 引用；Conflict 保留本地时间线与已知 revision。
pub(crate) fn submit(app: &tauri::AppHandle) -> Result<serde_json::Value, String> {
    let bridge = app.state::<AraBridge>();
    let mut inner = bridge.inner.lock().unwrap_or_else(|e| e.into_inner());
    let session = inner.session.as_mut().ok_or("ARA is disconnected")?;
    let state = app.state::<AppState>();
    let (timeline, version) = {
        let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        (
            timeline.clone(),
            state.timeline_version.load(Ordering::Acquire),
        )
    };
    let request = build_commit(session, timeline)?;
    let Request::Commit {
        timeline: sent_timeline,
        ..
    } = &request
    else {
        return Err("invalid ARA commit request".into());
    };
    let submitted_parameters = supported_projection(sent_timeline)?;
    let response = hifishifter_ara_ipc::exchange(&session.instance, &request)?;
    apply_commit_response(session, &response)?;
    clear_committed_parameter_dirty(&state, session, version, &submitted_parameters);
    Ok(session_payload(session))
}

pub(crate) fn disconnect(app: &tauri::AppHandle) -> serde_json::Value {
    app.state::<AraBridge>()
        .inner
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .session = None;
    // 会话结束，恢复本地后台扫描（连接期间它被抑制，见 `import_snapshot`）。
    crate::commands::channel_scan::set_ara_session_active(false);
    serde_json::json!({"ok": true})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn denied_temp_creation_falls_back_to_real_private_directory() {
        let root = std::env::temp_dir().join(format!("ara-low-temp-test-{}", uuid::Uuid::new_v4()));
        let primary = root.join("denied-temp");
        let fallback = root.join("local-low-private");
        let chosen = create_ara_temp_dir_with(&primary, Some(&fallback), |path| {
            if path == primary {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "controlled primary denial",
                ))
            } else {
                std::fs::create_dir_all(path)
            }
        })
        .unwrap();
        std::fs::write(chosen.join("source.wav"), b"owned audio").unwrap();
        assert_eq!(chosen, fallback);
        assert!(!primary.exists());
        assert_eq!(
            std::fs::read(fallback.join("source.wav")).unwrap(),
            b"owned audio"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn writable_temp_uses_original_root_and_never_creates_fallback() {
        let root = std::env::temp_dir().join(format!("ara-low-temp-test-{}", uuid::Uuid::new_v4()));
        let primary = root.join("normal-temp");
        let fallback = root.join("unused-fallback");
        let chosen = create_ara_temp_dir_with(&primary, Some(&fallback), |path| {
            std::fs::create_dir_all(path)
        })
        .unwrap();
        std::fs::write(chosen.join("source.wav"), b"temp audio").unwrap();
        assert_eq!(chosen, primary);
        assert!(!fallback.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn non_permission_failure_does_not_redirect_to_fallback() {
        let root = std::env::temp_dir().join(format!("ara-low-temp-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let blocker = root.join("file-not-directory");
        std::fs::write(&blocker, b"leave intact").unwrap();
        let fallback = root.join("unused-fallback");
        assert!(
            create_ara_temp_dir_with(&blocker.join("child"), Some(&fallback), |path| {
                std::fs::create_dir_all(path)
            })
            .is_err()
        );
        assert_eq!(std::fs::read(blocker).unwrap(), b"leave intact");
        assert!(!fallback.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn fixture() -> TimelineState {
        serde_json::from_value(serde_json::json!({
            "tracks": [{"id":"track","name":"host","order":0}], "bpm": 120, "project_sec": 1,
            "clips": [{ "id": "clip", "track_id": "track", "name": "host", "start_sec": 0,
                "length_sec": 1, "source_path": "host-id", "source_start_sec": 0,
                "source_end_sec": 1, "gain": 1,
                "takes": [{"id": "take", "source_path": "host-id"}] }]
        }))
        .unwrap()
    }

    fn pcm() -> HostPcm {
        HostPcm {
            persistent_id: "host-id".into(),
            sample_rate: 48000,
            planes: vec![vec![0.25, -0.5], vec![0.75, -1.0]],
            fingerprint: "test".into(),
        }
    }

    #[test]
    fn downloaded_pcm_remaps_clip_and_take_to_owned_float_wav() {
        let dir = std::env::temp_dir().join(format!("ara-gui-test-{}", uuid::Uuid::new_v4()));
        let mut input = fixture();
        let mut other_take = input.clips[0].takes[0].clone();
        other_take.id = "inactive-take".into();
        other_take.source_path = Some("other-host-id".into());
        input.clips[0].takes.push(other_take);
        let mut other_pcm = pcm();
        other_pcm.persistent_id = "other-host-id".into();
        let (timeline, reverse) = materialize_snapshot(input, &[pcm(), other_pcm], &dir).unwrap();
        let path = timeline.clips[0].source_path.as_ref().unwrap();
        assert_ne!(path, "host-id");
        assert_eq!(timeline.clips[0].takes[0].source_path.as_ref(), Some(path));
        assert_eq!(reverse.get(path).map(String::as_str), Some("host-id"));
        let other_path = timeline.clips[0].takes[1].source_path.as_ref().unwrap();
        assert_ne!(other_path, path);
        assert_eq!(
            reverse.get(other_path).map(String::as_str),
            Some("other-host-id")
        );
        let mut wav = hound::WavReader::open(path).unwrap();
        assert_eq!(
            (
                wav.spec().channels,
                wav.spec().sample_rate,
                wav.spec().sample_format
            ),
            (2, 48000, hound::SampleFormat::Float)
        );
        assert_eq!(
            wav.samples::<f32>().map(Result::unwrap).collect::<Vec<_>>(),
            vec![0.25, 0.75, -0.5, -1.0]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_or_invalid_pcm_never_creates_files() {
        let dir = std::env::temp_dir().join(format!("ara-gui-test-{}", uuid::Uuid::new_v4()));
        assert!(materialize_snapshot(fixture(), &[], &dir).is_err());
        let mut bad = pcm();
        bad.planes[1].pop();
        assert!(materialize_snapshot(fixture(), &[bad], &dir).is_err());
        assert!(!dir.exists());
    }

    #[test]
    fn dirty_project_requires_explicit_replace_confirmation() {
        let state = crate::state::command_test_state_without_audio_output();
        state.project.lock().unwrap().dirty = true;
        assert!(guard_replace(&state, false).is_err());
        assert!(guard_replace(&state, true).is_ok());
        state.project.lock().unwrap().dirty = false;
        assert!(guard_replace(&state, false).is_ok());
    }

    fn session() -> Session {
        let mut baseline = fixture();
        baseline.clips[0].takes[0].source_path = Some("owned.wav".into());
        baseline.clips[0].normalize_takes();
        Session {
            instance: InstanceRecord {
                instance_id: "instance".into(),
                pid: 1,
                pipe_name: "pipe".into(),
                token: "token".into(),
                name: "host".into(),
                heartbeat_ms: 0,
                protocol: 1,
            },
            revision: 7,
            model_revision: 9,
            reverse_paths: HashMap::from([("owned.wav".into(), "host-id".into())]),
            clip_ids: HashSet::from(["clip".into()]),
            host_geometry_baseline: host_geometry_projection(&baseline).unwrap(),
            project_baseline: project_projection(&crate::state::ProjectState::default()),
        }
    }

    #[test]
    fn commit_restores_host_identity_and_sends_both_session_revisions() {
        let mut timeline = fixture();
        timeline.clips[0].takes[0].source_path = Some("owned.wav".into());
        timeline.clips[0].normalize_takes();
        match build_commit(&session(), timeline).unwrap() {
            Request::Commit {
                base_revision,
                model_revision,
                timeline,
            } => {
                assert_eq!((base_revision, model_revision), (7, 9));
                assert_eq!(timeline["clips"][0]["takes"][0]["source_path"], "host-id");
                assert!(!timeline.to_string().contains("owned.wav"));
            }
            _ => panic!("wrong operation"),
        }
    }

    #[test]
    fn commit_rejects_unowned_source_and_replaced_timeline() {
        assert!(build_commit(&session(), fixture()).is_err());
        let mut timeline = fixture();
        timeline.clips.clear();
        assert!(build_commit(&session(), timeline).is_err());
    }

    /// 未提交几何/增益/名字不能被成功响应伪装为已保存；报错必须指名漂移字段。
    #[test]
    fn commit_rejects_local_move_crop_gain_and_track_name_but_accepts_parameters() {
        let mut baseline = fixture();
        baseline.clips[0].takes[0].source_path = Some("owned.wav".into());
        baseline.clips[0].normalize_takes();
        baseline.tracks =
            serde_json::from_value(serde_json::json!([{"id":"track","name":"host","order":0}]))
                .unwrap();
        let current = session();
        for (change, expected_path) in [
            (0usize, "clips[0].start_sec"),
            (1, "clips[0].source_start_sec"),
            (2, "clips[0].gain"),
            (3, "tracks[0].name"),
        ] {
            let mut timeline = baseline.clone();
            match change {
                0 => timeline.clips[0].start_sec = 0.25,
                1 => timeline.clips[0].source_start_sec = 0.25,
                2 => timeline.clips[0].gain = 0.5,
                _ => timeline.tracks[0].name = "local rename".into(),
            }
            let error = build_commit(&current, timeline).expect_err("未支持编辑必须拒绝");
            assert!(error.starts_with("ara_host_fields:"), "{error}");
            assert!(
                error.contains(expected_path),
                "报错必须指名漂移字段 {expected_path}: {error}"
            );
        }
        baseline.tracks[0].volume = 0.5;
        baseline.params_by_root_track.insert(
            "track".into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![62.0],
                ..Default::default()
            },
        );
        assert!(build_commit(&current, baseline).is_ok());
    }

    /// 本次修复的核心不变量：后台子系统写入**非宿主**字段，不得阻塞提交。
    ///
    /// 旧门禁比较"整份状态减去例外名单"，于是声道折叠扫描写下的
    /// `channel_mode` / `channel_decision`、分析回填的波形与音高范围、工程级的
    /// Tempo Map / 分组禁用都会让提交被拒 —— 用户什么都没做，却被告知"本地还有
    /// 未保存的宿主编辑"。这些字段一律不属于宿主几何，必须被忽略。
    #[test]
    fn background_writes_to_non_host_fields_do_not_block_commit() {
        let mut baseline = fixture();
        baseline.clips[0].takes[0].source_path = Some("owned.wav".into());
        baseline.clips[0].normalize_takes();
        let current = session();

        let mut timeline = baseline.clone();
        // 假立体声折叠扫描的落点。
        timeline.clips[0].takes[0].channel_mode = 2;
        timeline.clips[0].takes[0].channel_decision =
            Some(crate::channel_decision::ChannelDecisionRecord::user(2));
        // 波形 / 音高分析的缓存回填。
        timeline.clips[0].waveform_preview = Some(vec![0.25; 16]);
        timeline.clips[0].pitch_range = Some(crate::models::PitchRange {
            min: 40.0,
            max: 80.0,
        });
        timeline.clips[0].takes[0].waveform_preview = Some(vec![0.25; 16]);
        timeline.clips[0].takes[0].pitch_range = Some(crate::models::PitchRange {
            min: 40.0,
            max: 80.0,
        });
        timeline.clips[0].takes[0].source_file_fingerprint = Some(0x1122_3344_5566_7788);
        timeline.clips[0].takes[0].source_sample_rate = Some(48_000);
        timeline.clips[0].takes[0].source_channels = Some(2);
        timeline.clips[0].takes[0].duration_frames = Some(96_000);
        timeline.clips[0].takes[0].duration_sec = Some(2.0);
        // 工程级：Tempo Map / 分组禁用 / 轨道主题色。
        timeline.tempo_map = Some(vec![serde_json::from_value(serde_json::json!({
            "id": "p0", "positionSec": 0.0, "bpm": 140.0
        }))
        .unwrap()]);
        timeline.disabled_group_ids.insert("g1".into());
        timeline.tracks[0].color = "#4f8ef7".into();
        timeline.project_scale_notes = vec![0, 2, 3, 5, 7, 8, 10];
        // 提交内容本身：参数曲线。
        timeline.params_by_root_track.insert(
            "track".into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![62.0],
                ..Default::default()
            },
        );

        assert!(
            build_commit(&current, timeline).is_ok(),
            "后台写入非宿主字段不得阻塞提交"
        );
    }

    /// 宿主单方面决定的量（`bpm` / `project_sec`）不在判据里。
    ///
    /// 插件每次快照都从宿主时钟重读 `bpm`、重算 `project_sec`；本地改动既提交
    /// 不上去，也不代表"未保存的宿主编辑"，纳入判据只会制造假失败。
    #[test]
    fn host_derived_tempo_and_length_do_not_block_commit() {
        let mut baseline = fixture();
        baseline.clips[0].takes[0].source_path = Some("owned.wav".into());
        baseline.clips[0].normalize_takes();
        let current = session();

        let mut timeline = baseline.clone();
        timeline.bpm = 143.0;
        timeline.project_sec = 512.0;
        assert!(build_commit(&current, timeline).is_ok());
    }

    /// ARA 会话期间必须抑制本地自动声道扫描（它会写宿主拥有的 take 状态）。
    #[test]
    fn ara_session_flag_round_trips() {
        // 用例之间共享进程级标记：先归位，避免相互污染。
        crate::commands::channel_scan::set_ara_session_active(false);
        assert!(!crate::commands::channel_scan::ara_session_active());
        crate::commands::channel_scan::set_ara_session_active(true);
        assert!(crate::commands::channel_scan::ara_session_active());
        crate::commands::channel_scan::set_ara_session_active(false);
        assert!(!crate::commands::channel_scan::ara_session_active());
    }

    /// 差异收集器要能定位到具体路径，而不是只说"不一样"。
    #[test]
    fn geometry_diff_names_the_drifting_path() {
        let baseline = serde_json::json!({"clips":[{"start_sec":0.0,"gain":1.0}]});
        let current = serde_json::json!({"clips":[{"start_sec":0.25,"gain":1.0}]});
        let mut out = Vec::new();
        collect_json_differences(&baseline, &current, "", 8, &mut out);
        assert_eq!(out, vec!["clips[0].start_sec: 0.0 -> 0.25".to_string()]);

        // 长度变化直接报长度，不逐项刷屏。
        let longer = serde_json::json!({"clips":[{},{}]});
        let mut out = Vec::new();
        collect_json_differences(&baseline, &longer, "", 8, &mut out);
        assert!(out[0].starts_with("clips: length 1 -> 2"), "{out:?}");
    }

    #[test]
    fn parameter_success_clears_only_submitted_dirty_and_protects_notes_local_project_and_new_edits(
    ) {
        let state = crate::state::command_test_state_without_audio_output();
        let current = session();
        let mut baseline = fixture();
        baseline.clips[0].takes[0].source_path = Some("owned.wav".into());
        baseline.clips[0].normalize_takes();
        baseline.tracks[0].volume = 0.5;
        *state.timeline.lock().unwrap() = baseline;
        let version = state.timeline_version.load(Ordering::Acquire);
        let sent_parameters =
            supported_projection(&serde_json::to_value(&*state.timeline.lock().unwrap()).unwrap())
                .unwrap();
        state.project.lock().unwrap().dirty = true;
        clear_committed_parameter_dirty(&state, &current, version, &sent_parameters);
        assert!(!state.project.lock().unwrap().dirty);
        for change in 0..4 {
            let mut project = crate::state::ProjectState::default();
            project.dirty = true;
            match change {
                0 => project.notes_markdown = "unsaved notes".into(),
                1 => project.path = Some("local.hfs".into()),
                2 => project.name = "local name".into(),
                _ => project.base_scale = "D".into(),
            }
            *state.project.lock().unwrap() = project;
            clear_committed_parameter_dirty(&state, &current, version, &sent_parameters);
            assert!(
                guard_replace(&state, false).is_err(),
                "本地工程变化仍须确认刷新"
            );
        }
        *state.project.lock().unwrap() = crate::state::ProjectState::default();
        state.project.lock().unwrap().dirty = true;
        state.bump_timeline_version();
        clear_committed_parameter_dirty(&state, &current, version, &sent_parameters);
        assert!(
            guard_replace(&state, false).is_err(),
            "IPC期间新编辑仍须确认"
        );
    }

    fn parameter_command_state() -> AppState {
        let state = crate::state::command_test_state_without_audio_output();
        let mut timeline = fixture();
        timeline.clips[0].takes[0].source_path = Some("owned.wav".into());
        timeline.clips[0].normalize_takes();
        // dirty契约不需要推理：None避免build_root_pitch_key触发FCPE后台预热。
        timeline.tracks[0].pitch_analysis_algo = hifishifter_kernel::state::PitchAnalysisAlgo::None;
        let mut pitch = vec![69.0; 201];
        pitch[0] = 62.0;
        pitch[1] = 66.0;
        timeline.params_by_root_track.insert(
            "track".into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_orig: vec![69.0; 201],
                pitch_edit: pitch,
                pitch_edit_user_modified: true,
                ..Default::default()
            },
        );
        *state.timeline.lock().unwrap() = timeline;
        state
    }

    /// 真实首块checkpoint=true，后续尾块/平滑false；不能用手动bump伪造写入。
    #[test]
    fn noncheckpoint_curve_write_during_successful_commit_keeps_unsent_parameters_dirty() {
        for smoothing in [false, true] {
            let state = parameter_command_state();
            let mut current = session();
            assert_eq!(
                crate::commands::write_param_frames_for_test(
                    &state,
                    "track".into(),
                    "pitch".into(),
                    0,
                    vec![61.0],
                    Some(true)
                )["ok"],
                true
            );
            let submitted = state.timeline.lock().unwrap().clone();
            let version = state.timeline_version.load(Ordering::Acquire);
            let Request::Commit {
                timeline: sent_timeline,
                ..
            } = build_commit(&current, submitted.clone()).unwrap()
            else {
                panic!("wrong request")
            };
            let sent_parameters = supported_projection(&sent_timeline).unwrap();
            let history = state.history_depths();
            assert_eq!(
                crate::commands::write_param_frames_for_test(
                    &state,
                    "track".into(),
                    "pitch".into(),
                    if smoothing { 0 } else { 1 },
                    vec![64.0],
                    Some(false)
                )["ok"],
                true
            );
            assert_eq!(
                state.timeline_version.load(Ordering::Acquire),
                version,
                "真实false写入没有造版本变动"
            );
            assert_eq!(state.history_depths(), history, "尾块不能制造undo");
            apply_commit_response(
                &mut current,
                &Response {
                    ok: true,
                    revision: 8,
                    model_revision: 9,
                    ..Default::default()
                },
            )
            .unwrap();
            clear_committed_parameter_dirty(&state, &current, version, &sent_parameters);
            assert!(
                guard_replace(&state, false).is_err(),
                "IPC中的未发送尾块/平滑仍须确认刷新"
            );
        }
    }

    #[test]
    fn noncheckpoint_curve_tail_after_successful_commit_marks_dirty_without_new_undo() {
        let state = parameter_command_state();
        let mut current = session();
        crate::commands::write_param_frames_for_test(
            &state,
            "track".into(),
            "pitch".into(),
            0,
            vec![61.0],
            Some(true),
        );
        let version = state.timeline_version.load(Ordering::Acquire);
        let history = state.history_depths();
        let sent_parameters =
            supported_projection(&serde_json::to_value(&*state.timeline.lock().unwrap()).unwrap())
                .unwrap();
        apply_commit_response(
            &mut current,
            &Response {
                ok: true,
                revision: 8,
                model_revision: 9,
                ..Default::default()
            },
        )
        .unwrap();
        clear_committed_parameter_dirty(&state, &current, version, &sent_parameters);
        assert!(
            !state.project.lock().unwrap().dirty,
            "已提交当前参数可清dirty"
        );
        assert_eq!(
            crate::commands::write_param_frames_for_test(
                &state,
                "track".into(),
                "pitch".into(),
                1,
                vec![64.0],
                Some(false)
            )["ok"],
            true
        );
        assert_eq!(state.timeline_version.load(Ordering::Acquire), version);
        assert_eq!(state.history_depths(), history);
        assert!(
            guard_replace(&state, false).is_err(),
            "成功后的false尾块必须重新标脏"
        );
    }

    #[test]
    fn successful_commit_requires_all_supported_track_controls_to_match_sent_payload() {
        for field in [
            "volume",
            "muted",
            "solo",
            "compose_enabled",
            "pitch_analysis_algo",
        ] {
            let state = parameter_command_state();
            let current = session();
            let submitted = state.timeline.lock().unwrap().clone();
            let version = state.timeline_version.load(Ordering::Acquire);
            let Request::Commit {
                timeline: sent_timeline,
                ..
            } = build_commit(&current, submitted).unwrap()
            else {
                panic!("wrong request")
            };
            let sent_parameters = supported_projection(&sent_timeline).unwrap();
            state.project.lock().unwrap().dirty = true;
            let mut timeline = state.timeline.lock().unwrap();
            let track = &mut timeline.tracks[0];
            match field {
                "volume" => track.volume = 0.5,
                "muted" => track.muted = true,
                "solo" => track.solo = true,
                "compose_enabled" => track.compose_enabled = true,
                _ => {
                    track.pitch_analysis_algo =
                        hifishifter_kernel::state::PitchAnalysisAlgo::WorldDll
                }
            }
            drop(timeline);
            clear_committed_parameter_dirty(&state, &current, version, &sent_parameters);
            assert!(
                guard_replace(&state, false).is_err(),
                "未发送轨道控制也不能清dirty: {field}"
            );
        }
    }

    #[test]
    fn noncheckpoint_restore_marks_clean_project_dirty_without_checkpoint() {
        let state = parameter_command_state();
        let version = state.timeline_version.load(Ordering::Acquire);
        let response = crate::commands::restore_param_frames_for_test(
            &state,
            "track".into(),
            "pitch".into(),
            0,
            1,
            Some(false),
        );
        assert_eq!(response["ok"], true);
        assert_eq!(
            state.timeline.lock().unwrap().params_by_root_track["track"].pitch_edit[0],
            69.0
        );
        assert_eq!(state.timeline_version.load(Ordering::Acquire), version);
        assert_eq!(state.history_depths(), (0, 0));
        assert!(
            guard_replace(&state, false).is_err(),
            "真实false恢复必须标脏"
        );
    }

    #[test]
    fn noncheckpoint_static_parameter_marks_clean_project_dirty_without_checkpoint() {
        let state = parameter_command_state();
        let version = state.timeline_version.load(Ordering::Acquire);
        let response = crate::commands::set_static_param_for_test(
            &state,
            "track".into(),
            "synth_mode".into(),
            1.0,
            Some(false),
        );
        assert_eq!(response["ok"], true);
        assert_eq!(
            state.timeline.lock().unwrap().params_by_root_track["track"].extra_params["synth_mode"],
            1.0
        );
        assert_eq!(state.timeline_version.load(Ordering::Acquire), version);
        assert_eq!(state.history_depths(), (0, 0));
        assert!(
            guard_replace(&state, false).is_err(),
            "真实false静态参数写入必须标脏"
        );
    }

    #[test]
    fn noncheckpoint_linked_parameter_write_marks_clean_project_dirty_without_checkpoint() {
        let state = parameter_command_state();
        let version = state.timeline_version.load(Ordering::Acquire);
        let before = state.timeline.lock().unwrap().params_by_root_track["track"]
            .pitch_edit
            .clone();
        let response = crate::commands::stretch_track_linked_params_for_test(
            &state,
            "track".into(),
            vec![crate::state::StretchLinkedRangeSec {
                old_start_sec: 0.0,
                old_length_sec: 0.01,
                new_start_sec: 0.0,
                new_length_sec: 0.02,
            }],
            Some(false),
        );
        assert_eq!(response["ok"], true);
        assert_ne!(
            state.timeline.lock().unwrap().params_by_root_track["track"].pitch_edit,
            before
        );
        assert_eq!(state.timeline_version.load(Ordering::Acquire), version);
        assert_eq!(state.history_depths(), (0, 0));
        assert!(
            guard_replace(&state, false).is_err(),
            "真实false关联参数写入必须标脏"
        );
    }

    #[test]
    fn malformed_host_geometry_is_rejected_before_materializing_pcm() {
        let mut timeline = fixture();
        timeline.clips[0].start_sec = f64::NAN;
        let dir = std::env::temp_dir().join(format!("ara-gui-test-{}", uuid::Uuid::new_v4()));
        assert!(materialize_snapshot(timeline, &[pcm()], &dir).is_err());
        assert!(!dir.exists());
    }

    #[test]
    fn conflict_retains_known_revisions_but_success_advances_them() {
        let mut current = session();
        let conflict = Response {
            ok: false,
            error: Some("Conflict".into()),
            revision: 20,
            model_revision: 21,
            ..Response::default()
        };
        assert!(apply_commit_response(&mut current, &conflict).is_err());
        assert_eq!((current.revision, current.model_revision), (7, 9));
        let accepted = Response {
            ok: true,
            revision: 8,
            model_revision: 9,
            ..Response::default()
        };
        apply_commit_response(&mut current, &accepted).unwrap();
        assert_eq!((current.revision, current.model_revision), (8, 9));
    }
}
