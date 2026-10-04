//! ARA GUI 桥接：仅下载宿主授权 PCM，交由既有波形和音高管线处理。

use crate::state::{AppState, TimelineState};
use hifishifter_ara_ipc::{HostPcm, InstanceRecord, Request, Response};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Mutex},
};
use tauri::Manager;

/// 写入应用私有 WAV，先验证全部源，绝不将 persistentID 当路径打开。
fn materialize_snapshot(
    mut timeline: TimelineState,
    sources: &[HostPcm],
    dir: &Path,
) -> Result<(TimelineState, HashMap<String, String>), String> {
    if !timeline.bpm.is_finite()
        || timeline.bpm <= 0.0
        || !timeline.project_sec.is_finite()
        || timeline.project_sec < 0.0
    {
        return Err("invalid host timeline".into());
    }
    let mut pcm_by_id = HashMap::new();
    for pcm in sources {
        let frames = pcm.planes.first().map(Vec::len).unwrap_or(0);
        if pcm.persistent_id.is_empty()
            || !matches!(pcm.sample_rate, 44100 | 48000)
            || !(1..=2).contains(&pcm.planes.len())
            || frames == 0
            || frames > pcm.sample_rate as usize * 30
            || pcm
                .planes
                .iter()
                .any(|plane| plane.len() != frames || plane.iter().any(|v| !v.is_finite()))
            || pcm_by_id.insert(pcm.persistent_id.clone(), pcm).is_some()
        {
            return Err("invalid host PCM".into());
        }
    }
    for clip in &mut timeline.clips {
        clip.normalize_takes();
        if !clip.start_sec.is_finite()
            || !clip.length_sec.is_finite()
            || clip.length_sec <= 0.0
            || clip.takes.iter().any(|take| {
                !take.source_start_sec.is_finite()
                    || !take.source_end_sec.is_finite()
                    || !take.playback_rate.is_finite()
                    || take.playback_rate <= 0.0
            })
        {
            return Err("invalid host clip geometry".into());
        }
        if clip.reversed || clip.takes.iter().any(|take| take.reversed) {
            return Err("ARA reverse playback is unsupported".into());
        }
        for source in std::iter::once(&clip.source_path)
            .chain(clip.takes.iter().map(|take| &take.source_path))
        {
            let id = source.as_ref().ok_or("missing host PCM reference")?;
            if !pcm_by_id.contains_key(id) {
                return Err(format!("missing host PCM: {id}"));
            }
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let result = (|| {
        let mut paths = HashMap::new();
        let mut reverse = HashMap::new();
        for (index, pcm) in sources.iter().enumerate() {
            let path = dir.join(format!("source-{index}.wav"));
            let spec = hound::WavSpec {
                channels: pcm.planes.len() as u16,
                sample_rate: pcm.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut wav = hound::WavWriter::create(&path, spec).map_err(|e| e.to_string())?;
            for frame in 0..pcm.planes[0].len() {
                for plane in &pcm.planes {
                    wav.write_sample(plane[frame]).map_err(|e| e.to_string())?;
                }
            }
            wav.finalize().map_err(|e| e.to_string())?;
            let path = path.to_string_lossy().into_owned();
            paths.insert(pcm.persistent_id.clone(), path.clone());
            reverse.insert(path, pcm.persistent_id.clone());
        }
        for clip in &mut timeline.clips {
            clip.source_path = clip
                .source_path
                .as_ref()
                .and_then(|id| paths.get(id).cloned());
            clip.source_path_relative = None;
            for take in &mut clip.takes {
                let pcm = pcm_by_id[take.source_path.as_ref().ok_or("missing PCM")?];
                take.source_path = Some(paths[&pcm.persistent_id].clone());
                take.source_path_relative = None;
                take.source_sample_rate = Some(pcm.sample_rate);
                take.source_channels = Some(pcm.planes.len() as u16);
                take.duration_frames = Some(pcm.planes[0].len() as u64);
                take.duration_sec = Some(pcm.planes[0].len() as f64 / pcm.sample_rate as f64);
                take.source_file_fingerprint = None;
                take.source_file_mtime = None;
                take.source_file_size = None;
                take.waveform_preview = None;
                take.pitch_range = None;
            }
            clip.normalize_takes();
        }
        Ok((timeline, reverse))
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(dir);
    }
    result
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
    let response = hifishifter_ara_ipc::exchange(&instance, &Request::Snapshot)?;
    check_response(&response)?;
    let timeline: TimelineState =
        serde_json::from_value(response.timeline.ok_or("missing ARA timeline")?)
            .map_err(|e| e.to_string())?;
    let dir = std::env::temp_dir().join(format!("hifishifter-ara-{}", uuid::Uuid::new_v4()));
    let (mut timeline, reverse_paths) = materialize_snapshot(timeline, &response.sources, &dir)?;
    inner.owned_dirs.push(dir);
    timeline.playhead_sec = 0.0;
    timeline.selected_track_id = timeline.tracks.first().map(|track| track.id.clone());
    timeline.selected_clip_id = timeline.clips.first().map(|clip| clip.id.clone());
    let clip_ids = timeline.clips.iter().map(|clip| clip.id.clone()).collect();
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
    let response = hifishifter_ara_ipc::exchange(&session.instance, &request)?;
    apply_commit_response(session, &response)?;
    {
        let _timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        if state.timeline_version.load(Ordering::Acquire) == version {
            let mut project = state.project.lock().unwrap_or_else(|e| e.into_inner());
            if project.notes_markdown.is_empty() {
                project.dirty = false;
            }
        }
    }
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

    fn fixture() -> TimelineState {
        serde_json::from_value(serde_json::json!({
            "tracks": [], "bpm": 120, "project_sec": 1,
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
        let state = AppState::default();
        state.project.lock().unwrap().dirty = true;
        assert!(guard_replace(&state, false).is_err());
        assert!(guard_replace(&state, true).is_ok());
        state.project.lock().unwrap().dirty = false;
        assert!(guard_replace(&state, false).is_ok());
    }

    fn session() -> Session {
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
