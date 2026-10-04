//! ARA GUI 桥接：仅下载宿主授权 PCM，交由既有波形和音高管线处理。

use crate::state::{AppState, TimelineState};
use hifishifter_ara_ipc::{HostPcm, InstanceRecord, Request, Response};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Mutex},
};
use tauri::Manager;

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
fn materialize_snapshot(timeline:TimelineState,sources:&[HostPcm],dir:&Path)
    ->Result<(TimelineState,HashMap<String,String>),String> {
    let views:Vec<_>=sources.iter().map(|pcm|hifishifter_kernel::editor::host_pcm::PcmView {
        persistent_id:&pcm.persistent_id,sample_rate:pcm.sample_rate,planes:&pcm.planes,
    }).collect();
    hifishifter_kernel::editor::host_pcm::materialize(timeline,&views,dir)
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
    unsupported_baseline: serde_json::Value,
    project_baseline: serde_json::Value,
}

/// 保存宿主参数投影之外的语义基线；播放位置、选择和分析缓存不代表未提交编辑。
fn unsupported_projection(timeline: &TimelineState) -> Result<serde_json::Value, String> {
    let mut normalized = timeline.clone();
    normalized.sync_clip_takes_from_flat();
    for clip in &mut normalized.clips {
        for take in &mut clip.takes {
            take.waveform_preview = None;
            take.pitch_range = None;
            take.source_file_fingerprint = None;
            take.source_sample_rate = None;
            take.source_channels = None;
            take.duration_frames = None;
            take.duration_sec = None;
        }
    }
    let mut value = serde_json::to_value(normalized).map_err(|e| e.to_string())?;
    let object = value.as_object_mut().ok_or("invalid ARA timeline")?;
    for key in ["params_by_root_track", "selected_track_id", "selected_clip_id", "playhead_sec", "next_track_order"] {
        object.remove(key);
    }
    for track in object.get_mut("tracks").and_then(serde_json::Value::as_array_mut).ok_or("missing ARA tracks")? {
        let track = track.as_object_mut().ok_or("invalid ARA track")?;
        for key in ["compose_enabled", "pitch_analysis_algo", "volume", "muted", "solo"] { track.remove(key); }
    }
    Ok(value)
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
    let tracks=timeline.get("tracks").and_then(serde_json::Value::as_array).ok_or("missing submitted ARA tracks")?;
    let controls: Vec<_>=tracks.iter().map(|track| serde_json::json!({
        "id":track["id"], "volume":track["volume"], "muted":track["muted"], "solo":track["solo"],
        "compose_enabled":track["compose_enabled"], "pitch_analysis_algo":track["pitch_analysis_algo"]
    })).collect();
    Ok(serde_json::json!({"params":timeline.get("params_by_root_track").cloned().unwrap_or_else(||serde_json::json!({})),"tracks":controls}))
}

/// 清dirty须在timeline锁内确认当前支持参数仍等于实际送出的参数，不能依赖undo版本。
fn clear_committed_parameter_dirty(state: &AppState, session: &Session, submitted_version: u64, submitted_parameters: &serde_json::Value) {
    let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    if state.timeline_version.load(Ordering::Acquire) != submitted_version
        || unsupported_projection(&timeline).ok().as_ref() != Some(&session.unsupported_baseline) { return; }
    let current_parameters=serde_json::to_value(&*timeline).ok().and_then(|timeline| supported_projection(&timeline).ok());
    if current_parameters.as_ref()!=Some(submitted_parameters) { return; }
    let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
    if project_projection(&project) == session.project_baseline { project.dirty = false; }
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
fn build_commit(session: &Session, mut timeline: TimelineState) -> Result<Request, String> {
    if unsupported_projection(&timeline)? != session.unsupported_baseline {
        return Err("ARA only submits parameter curves and track synthesis controls. Modify clip geometry, clip gain, names and other host fields in REAPER; local edits are still unsaved.".into());
    }
    let ids: HashSet<_> = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
    if ids != session.clip_ids {
        return Err("ARA timeline changed; reconnect to the host".into());
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
pub(crate) fn import_snapshot(
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
    let unsupported_baseline = unsupported_projection(&timeline)?;
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
        unsupported_baseline,
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
    let Request::Commit { timeline: sent_timeline, .. } = &request else { return Err("invalid ARA commit request".into()); };
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
            unsupported_baseline: unsupported_projection(&baseline).unwrap(),
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

    /// 未提交几何/增益/名字不能被成功响应伪装为已保存。
    #[test]
    fn commit_rejects_local_move_crop_gain_and_track_name_but_accepts_parameters() {
        let mut baseline = fixture();
        baseline.clips[0].takes[0].source_path = Some("owned.wav".into()); baseline.clips[0].normalize_takes();
        baseline.tracks = serde_json::from_value(serde_json::json!([{"id":"track","name":"host","order":0}])).unwrap();
        let current = session();
        for change in 0..4 {
            let mut timeline = baseline.clone();
            match change {
                0 => timeline.clips[0].start_sec = 0.25,
                1 => timeline.clips[0].source_start_sec = 0.25,
                2 => timeline.clips[0].gain = 0.5,
                _ => timeline.tracks[0].name = "local rename".into(),
            }
            let error = build_commit(&current, timeline).err().expect("未支持编辑必须拒绝");
            assert!(error.contains("REAPER"), "{error}");
        }
        baseline.tracks[0].volume = 0.5;
        baseline.params_by_root_track.insert("track".into(), hifishifter_kernel::state::TrackParamsState { frame_period_ms:5.0, pitch_edit: vec![62.0], ..Default::default() });
        assert!(build_commit(&current, baseline).is_ok());
    }

    #[test]
    fn parameter_success_clears_only_submitted_dirty_and_protects_notes_local_project_and_new_edits() {
        let state=crate::state::command_test_state_without_audio_output(); let current=session(); let mut baseline=fixture();
        baseline.clips[0].takes[0].source_path=Some("owned.wav".into()); baseline.clips[0].normalize_takes();
        baseline.tracks[0].volume=0.5; *state.timeline.lock().unwrap()=baseline;
        let version=state.timeline_version.load(Ordering::Acquire);
        let sent_parameters=supported_projection(&serde_json::to_value(&*state.timeline.lock().unwrap()).unwrap()).unwrap();
        state.project.lock().unwrap().dirty=true;
        clear_committed_parameter_dirty(&state,&current,version,&sent_parameters);
        assert!(!state.project.lock().unwrap().dirty);
        for change in 0..4 {
            let mut project=crate::state::ProjectState::default(); project.dirty=true;
            match change { 0=>project.notes_markdown="unsaved notes".into(),1=>project.path=Some("local.hfs".into()),
                2=>project.name="local name".into(),_=>project.base_scale="D".into() }
            *state.project.lock().unwrap()=project;
            clear_committed_parameter_dirty(&state,&current,version,&sent_parameters);
            assert!(guard_replace(&state,false).is_err(),"本地工程变化仍须确认刷新");
        }
        *state.project.lock().unwrap()=crate::state::ProjectState::default(); state.project.lock().unwrap().dirty=true;
        state.bump_timeline_version(); clear_committed_parameter_dirty(&state,&current,version,&sent_parameters);
        assert!(guard_replace(&state,false).is_err(),"IPC期间新编辑仍须确认");
    }

    fn parameter_command_state() -> AppState {
        let state=crate::state::command_test_state_without_audio_output(); let mut timeline=fixture();
        timeline.clips[0].takes[0].source_path=Some("owned.wav".into()); timeline.clips[0].normalize_takes();
        // dirty契约不需要推理：None避免build_root_pitch_key触发FCPE后台预热。
        timeline.tracks[0].pitch_analysis_algo=hifishifter_kernel::state::PitchAnalysisAlgo::None;
        let mut pitch=vec![69.0;201]; pitch[0]=62.0; pitch[1]=66.0;
        timeline.params_by_root_track.insert("track".into(),hifishifter_kernel::state::TrackParamsState {
            frame_period_ms:5.0,pitch_orig:vec![69.0;201],pitch_edit:pitch,pitch_edit_user_modified:true,..Default::default() });
        *state.timeline.lock().unwrap()=timeline;
        state
    }

    /// 真实首块checkpoint=true，后续尾块/平滑false；不能用手动bump伪造写入。
    #[test]
    fn noncheckpoint_curve_write_during_successful_commit_keeps_unsent_parameters_dirty() {
        for smoothing in [false,true] {
            let state=parameter_command_state(); let mut current=session();
            assert_eq!(crate::commands::write_param_frames_for_test(&state,"track".into(),"pitch".into(),0,vec![61.0],Some(true))["ok"],true);
            let submitted=state.timeline.lock().unwrap().clone(); let version=state.timeline_version.load(Ordering::Acquire);
            let Request::Commit { timeline: sent_timeline, .. }=build_commit(&current,submitted.clone()).unwrap() else { panic!("wrong request") };
            let sent_parameters=supported_projection(&sent_timeline).unwrap();
            let history=state.history_depths();
            assert_eq!(crate::commands::write_param_frames_for_test(&state,"track".into(),"pitch".into(),if smoothing {0} else {1},vec![64.0],Some(false))["ok"],true);
            assert_eq!(state.timeline_version.load(Ordering::Acquire),version,"真实false写入没有造版本变动");
            assert_eq!(state.history_depths(),history,"尾块不能制造undo");
            apply_commit_response(&mut current,&Response { ok:true,revision:8,model_revision:9,..Default::default() }).unwrap();
            clear_committed_parameter_dirty(&state,&current,version,&sent_parameters);
            assert!(guard_replace(&state,false).is_err(),"IPC中的未发送尾块/平滑仍须确认刷新");
        }
    }

    #[test]
    fn noncheckpoint_curve_tail_after_successful_commit_marks_dirty_without_new_undo() {
        let state=parameter_command_state(); let mut current=session();
        crate::commands::write_param_frames_for_test(&state,"track".into(),"pitch".into(),0,vec![61.0],Some(true));
        let version=state.timeline_version.load(Ordering::Acquire); let history=state.history_depths();
        let sent_parameters=supported_projection(&serde_json::to_value(&*state.timeline.lock().unwrap()).unwrap()).unwrap();
        apply_commit_response(&mut current,&Response { ok:true,revision:8,model_revision:9,..Default::default() }).unwrap();
        clear_committed_parameter_dirty(&state,&current,version,&sent_parameters);
        assert!(!state.project.lock().unwrap().dirty,"已提交当前参数可清dirty");
        assert_eq!(crate::commands::write_param_frames_for_test(&state,"track".into(),"pitch".into(),1,vec![64.0],Some(false))["ok"],true);
        assert_eq!(state.timeline_version.load(Ordering::Acquire),version);
        assert_eq!(state.history_depths(),history);
        assert!(guard_replace(&state,false).is_err(),"成功后的false尾块必须重新标脏");
    }

    #[test]
    fn successful_commit_requires_all_supported_track_controls_to_match_sent_payload() {
        for field in ["volume","muted","solo","compose_enabled","pitch_analysis_algo"] {
            let state=parameter_command_state(); let current=session();
            let submitted=state.timeline.lock().unwrap().clone(); let version=state.timeline_version.load(Ordering::Acquire);
            let Request::Commit { timeline:sent_timeline,.. }=build_commit(&current,submitted).unwrap() else { panic!("wrong request") };
            let sent_parameters=supported_projection(&sent_timeline).unwrap();
            state.project.lock().unwrap().dirty=true;
            let mut timeline=state.timeline.lock().unwrap(); let track=&mut timeline.tracks[0];
            match field { "volume"=>track.volume=0.5,"muted"=>track.muted=true,"solo"=>track.solo=true,
                "compose_enabled"=>track.compose_enabled=true,_=>track.pitch_analysis_algo=hifishifter_kernel::state::PitchAnalysisAlgo::WorldDll }
            drop(timeline);
            clear_committed_parameter_dirty(&state,&current,version,&sent_parameters);
            assert!(guard_replace(&state,false).is_err(),"未发送轨道控制也不能清dirty: {field}");
        }
    }

    #[test]
    fn noncheckpoint_restore_marks_clean_project_dirty_without_checkpoint() {
        let state=parameter_command_state(); let version=state.timeline_version.load(Ordering::Acquire);
        let response=crate::commands::restore_param_frames_for_test(&state,"track".into(),"pitch".into(),0,1,Some(false));
        assert_eq!(response["ok"],true); assert_eq!(state.timeline.lock().unwrap().params_by_root_track["track"].pitch_edit[0],69.0);
        assert_eq!(state.timeline_version.load(Ordering::Acquire),version); assert_eq!(state.history_depths(),(0,0));
        assert!(guard_replace(&state,false).is_err(),"真实false恢复必须标脏");
    }

    #[test]
    fn noncheckpoint_static_parameter_marks_clean_project_dirty_without_checkpoint() {
        let state=parameter_command_state(); let version=state.timeline_version.load(Ordering::Acquire);
        let response=crate::commands::set_static_param_for_test(&state,"track".into(),"synth_mode".into(),1.0,Some(false));
        assert_eq!(response["ok"],true); assert_eq!(state.timeline.lock().unwrap().params_by_root_track["track"].extra_params["synth_mode"],1.0);
        assert_eq!(state.timeline_version.load(Ordering::Acquire),version); assert_eq!(state.history_depths(),(0,0));
        assert!(guard_replace(&state,false).is_err(),"真实false静态参数写入必须标脏");
    }

    #[test]
    fn noncheckpoint_linked_parameter_write_marks_clean_project_dirty_without_checkpoint() {
        let state=parameter_command_state(); let version=state.timeline_version.load(Ordering::Acquire);
        let before=state.timeline.lock().unwrap().params_by_root_track["track"].pitch_edit.clone();
        let response=crate::commands::stretch_track_linked_params_for_test(&state,"track".into(),vec![crate::state::StretchLinkedRangeSec {
            old_start_sec:0.0,old_length_sec:0.01,new_start_sec:0.0,new_length_sec:0.02 }],Some(false));
        assert_eq!(response["ok"],true); assert_ne!(state.timeline.lock().unwrap().params_by_root_track["track"].pitch_edit,before);
        assert_eq!(state.timeline_version.load(Ordering::Acquire),version); assert_eq!(state.history_depths(),(0,0));
        assert!(guard_replace(&state,false).is_err(),"真实false关联参数写入必须标脏");
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
