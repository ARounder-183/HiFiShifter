//! Host-link runtime for non-ARA (Capture) editor sessions.
//!
//! ARA sessions deliberately do not participate: their geometry and audio are
//! owned by the host document. The standalone app can still open workspaces
//! produced by external bridges (see `hifishifter-host-link` examples), but
//! the plug-in only exports its own Capture timeline, where both the geometry
//! and PCM belong to the plug-in.
use super::extension::ExtensionOwner;
use crate::sync::MutexExt;
use hifishifter_host_link::{
    now_ms, AudioState, Capability, EditManifest, LinkManifest, LinkStatus, Producer, Revisions,
    SourceEntry, SourceKind, Workspace,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The current export. It is process-wide because the GUI can be closed while
/// the standalone app keeps editing the exported workspace.
enum Phase {
    Running,
    Ready {
        workspace: Workspace,
        identity: String,
        manifest: Box<LinkManifest>,
        seen_edit_stamp: Option<u64>,
        last_status: Option<LinkStatus>,
    },
    Failed {
        error: String,
    },
}

fn phase() -> &'static Mutex<Option<Phase>> {
    static PHASE: OnceLock<Mutex<Option<Phase>>> = OnceLock::new();
    PHASE.get_or_init(|| Mutex::new(None))
}

fn lock() -> std::sync::MutexGuard<'static, Option<Phase>> {
    phase()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Invalidates the result of an export that has been superseded by a newer one.
static EXPORT_EPOCH: AtomicU64 = AtomicU64::new(0);

pub(crate) fn root() -> Result<PathBuf, String> {
    hifishifter_host_link::workspace::default_root()
        .ok_or_else(|| "no writable data directory for link workspaces".to_string())
}

/// Starts a background export and returns the immediately observable status.
pub(crate) fn start_export(owner: Arc<ExtensionOwner>) -> Result<serde_json::Value, String> {
    let root = root()?;
    let epoch = EXPORT_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
    *lock() = Some(Phase::Running);
    std::thread::Builder::new()
        .name("hfs-link-export".into())
        .spawn(move || {
            let outcome = owner.export_link(&root);
            let mut guard = lock();
            if EXPORT_EPOCH.load(Ordering::Acquire) != epoch {
                return;
            }
            match outcome {
                Ok((manifest, workspace)) => {
                    log::info!(
                        "[link] exported {} clips / {} sources",
                        manifest.clip_count(),
                        manifest.sources.len()
                    );
                    *guard = Some(Phase::Ready {
                        workspace,
                        identity: manifest.link_id.clone(),
                        manifest: Box::new(manifest),
                        seen_edit_stamp: None,
                        last_status: None,
                    });
                }
                Err(error) => {
                    log::warn!("[link] export failed: {error}");
                    *guard = Some(Phase::Failed { error });
                }
            }
        })
        .map_err(|error| error.to_string())?;
    Ok(status())
}

pub(crate) fn release() -> serde_json::Value {
    *lock() = None;
    status()
}

pub(crate) fn status() -> serde_json::Value {
    let guard = lock();
    match guard.as_ref() {
        None => serde_json::json!({ "attached": false }),
        Some(Phase::Running) => serde_json::json!({ "attached": true, "phase": "running" }),
        Some(Phase::Failed { error }) => {
            serde_json::json!({ "attached": true, "phase": "failed", "error": error })
        }
        Some(Phase::Ready {
            workspace,
            identity,
            manifest,
            last_status,
            ..
        }) => {
            let (complete, partial, external) = manifest.audio_summary();
            serde_json::json!({
                "attached": true,
                "phase": "ready",
                "identity": identity,
                "hostLabel": manifest.host_label,
                "sourceKind": manifest.source_kind.tag(),
                "directory": workspace.root().display().to_string(),
                "exportedAt": manifest.created_unix_ms,
                "clips": manifest.clip_count(),
                "tracks": manifest.track_count(),
                "audio": { "complete": complete, "partial": partial, "extern": external },
                "losses": manifest.losses.len(),
                "acceptsCommit": manifest.grants(Capability::Commit),
                "diagnostics": manifest.diagnostics,
                "lastEdit": last_status.as_ref().map(|status| serde_json::json!({
                    "ok": status.ok,
                    "code": status.code,
                    "error": status.error,
                    "edits": status.revisions.edits,
                    "model": status.revisions.model,
                    "atUnixMs": status.updated_unix_ms,
                })),
            })
        }
    }
}

/// Applies a new `edits.hfsedit` from the standalone app to the live Capture session.
pub(crate) fn poll(owner: &Arc<ExtensionOwner>) {
    let (workspace, identity, previous_stamp) = {
        let guard = lock();
        let Some(Phase::Ready {
            workspace,
            identity,
            seen_edit_stamp,
            ..
        }) = guard.as_ref()
        else {
            return;
        };
        (workspace.clone(), identity.clone(), *seen_edit_stamp)
    };
    let Some(current_stamp) = workspace.edit_stamp_ms() else {
        return;
    };
    if Some(current_stamp) == previous_stamp {
        return;
    }
    let Ok(edit) = workspace.read_edit() else {
        return;
    };
    let status = owner.apply_link_edit(&identity, &edit);
    if let Err(error) = workspace.write_status(&status) {
        log::warn!("[link] cannot write the link status: {error}");
    }
    let mut guard = lock();
    if let Some(Phase::Ready {
        identity: current,
        seen_edit_stamp,
        last_status,
        ..
    }) = guard.as_mut()
    {
        if *current == identity {
            *seen_edit_stamp = Some(current_stamp);
            *last_status = Some(status);
        }
    }
}

impl ExtensionOwner {
    /// Exports the Capture timeline, its PCM and its current user curves.
    pub(crate) fn export_link(
        &self,
        root: &std::path::Path,
    ) -> Result<(LinkManifest, Workspace), String> {
        use hifishifter_host_link as link;
        if self.is_ara_bound() {
            return Err("host link is only available in Capture mode".into());
        }
        let session = self
            .capture_session_if_initialized()
            .ok_or_else(|| "the Capture editor session is not open".to_string())?;
        if !session.is_capture() || !session.is_loaded() {
            return Err("the Capture editor session is not ready".into());
        }
        if let Err(error) = session.flush_with_budget(std::time::Duration::from_millis(200)) {
            log::warn!("[link] flush before export exceeded its budget: {error}");
        }
        let state = crate::component_state::ComponentState::decode(&self.encode_state()?)?;
        let mut timeline = self.capture().session_timeline().0;
        let live = session.timeline.lock_or_recover().clone();
        let namespace = session.link_namespace();
        // Projected ids are stable; bind the live, namespaced user data back to
        // the same ids before serialising the workspace manifest.
        timeline.params_by_root_track = live
            .params_by_root_track
            .into_iter()
            .map(|(root, params)| {
                (
                    root.strip_prefix(namespace).unwrap_or(&root).to_owned(),
                    params,
                )
            })
            .collect();
        for root in timeline
            .clips
            .iter()
            .filter_map(|clip| timeline.resolve_root_track_id(&clip.track_id))
            .collect::<BTreeSet<_>>()
        {
            timeline.ensure_params_for_root(&root);
        }
        for track in &mut timeline.tracks {
            if let Some(edited) = live
                .tracks
                .iter()
                .find(|live| live.id.strip_prefix(namespace) == Some(track.id.as_str()))
            {
                track.volume = edited.volume;
                track.muted = edited.muted;
                track.solo = edited.solo;
                track.compose_enabled = edited.compose_enabled;
                track.pitch_analysis_algo = edited.pitch_analysis_algo.clone();
                track.name = edited.name.clone();
                track.color = edited.color.clone();
            }
        }
        if timeline.clips.is_empty() {
            return Err("no Capture material to export".into());
        }
        let model = self.capture().revision();
        let edits_revision = session.timeline_revision();
        let host_label = self.host_app_name().unwrap_or_else(|| "Host".to_owned());
        let doc_key = capture_doc_key(&state);
        let identity = link::link_id(SourceKind::Capture, &host_label, &doc_key);
        let workspace = Workspace::at(root.join(identity.replace(':', "-")));
        let sources = self.write_link_audio(&workspace, &timeline)?;
        let manifest = LinkManifest {
            format: link::FORMAT.into(),
            format_version: link::FORMAT_VERSION,
            link_id: identity.clone(),
            source_kind: SourceKind::Capture,
            host_label,
            created_unix_ms: now_ms(),
            producer: Producer {
                implementation: format!("hifishifter-capture {}", crate::VERSION),
                pid: Some(std::process::id()),
            },
            timeline: serde_json::to_value(&timeline).map_err(|error| error.to_string())?,
            host_geometry: hifishifter_kernel::editor::host_timeline::host_geometry_projection(
                &timeline,
            ),
            sources,
            granted: vec![Capability::Commit, Capability::Audio],
            revisions: Revisions {
                edits: edits_revision,
                model,
            },
            admission: SourceKind::Capture.admission(),
            losses: Vec::new(),
            diagnostics: None,
        };
        workspace.write_manifest(&manifest)?;
        workspace.write_status(&LinkStatus::initial(&identity, manifest.revisions))?;
        Ok((manifest, workspace))
    }

    fn write_link_audio(
        &self,
        workspace: &Workspace,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> Result<Vec<SourceEntry>, String> {
        let available = self.capture().material_sources();
        let keys: BTreeSet<String> = timeline
            .clips
            .iter()
            .filter_map(|clip| clip.source_path.clone())
            .collect();
        let mut sources = Vec::new();
        for key in keys {
            let Some(pcm) = available.get(&key) else {
                sources.push(SourceEntry {
                    key,
                    sample_rate: 0,
                    channels: 0,
                    frames: 0,
                    state: AudioState::Partial,
                    file: None,
                    blake3: None,
                    bytes: None,
                    note: Some("Capture source is no longer available".into()),
                });
                continue;
            };
            let audio = workspace.write_audio(&key, pcm.sample_rate, &pcm.planes)?;
            sources.push(SourceEntry {
                key,
                sample_rate: pcm.sample_rate,
                channels: pcm.planes.len() as u32,
                frames: pcm.planes.first().map_or(0, Vec::len) as u64,
                state: AudioState::Complete,
                file: Some(audio.file),
                blake3: Some(audio.blake3),
                bytes: Some(audio.bytes),
                note: None,
            });
        }
        Ok(sources)
    }

    /// Merges the two user-owned fields from an edit into the live Capture timeline.
    pub(crate) fn apply_link_edit(&self, identity: &str, edit: &EditManifest) -> LinkStatus {
        let mut status = LinkStatus {
            format: hifishifter_host_link::STATUS_FORMAT.into(),
            format_version: hifishifter_host_link::FORMAT_VERSION,
            link_id: identity.to_string(),
            ok: false,
            code: None,
            error: None,
            revisions: Revisions::default(),
            updated_unix_ms: now_ms(),
        };
        let outcome = (|| -> Result<(), (u32, String)> {
            if edit.link_id != identity {
                return Err((
                    hifishifter_host_link::error_code::LINK_MISMATCH,
                    format!("this edit belongs to {}, not {identity}", edit.link_id),
                ));
            }
            let expected = {
                let guard = lock();
                match guard.as_ref() {
                    Some(Phase::Ready { manifest, .. }) => manifest.revisions,
                    _ => {
                        return Err((
                            hifishifter_host_link::error_code::DOCUMENT_CLOSED,
                            "the exported workspace is no longer open".into(),
                        ))
                    }
                }
            };
            if edit.base_revisions != expected {
                return Err((
                    hifishifter_host_link::error_code::CONFLICT,
                    "the plug-in export changed after this workspace was opened".into(),
                ));
            }
            let params: std::collections::BTreeMap<
                String,
                hifishifter_kernel::state::TrackParamsState,
            > = serde_json::from_value(edit.params.clone()).map_err(|error| {
                (
                    hifishifter_host_link::error_code::INVALID_EDIT,
                    error.to_string(),
                )
            })?;
            let mut tracks: Vec<hifishifter_kernel::state::Track> =
                serde_json::from_value(edit.tracks.clone()).map_err(|error| {
                    (
                        hifishifter_host_link::error_code::INVALID_EDIT,
                        error.to_string(),
                    )
                })?;
            let session = self.capture_session_if_initialized().ok_or_else(|| {
                (
                    hifishifter_host_link::error_code::DOCUMENT_CLOSED,
                    "the Capture editor session is closed".to_string(),
                )
            })?;
            let namespace = session.link_namespace();
            let params = params
                .into_iter()
                .map(|(root, value)| (format!("{namespace}{root}"), value))
                .collect();
            for track in &mut tracks {
                track.id = format!("{namespace}{}", track.id);
                track.parent_id = track
                    .parent_id
                    .as_deref()
                    .map(|id| format!("{namespace}{id}"));
            }
            session
                .apply_link_edits(&params, &tracks)
                .map_err(|error| (hifishifter_host_link::error_code::INVALID_EDIT, error))?;
            Ok(())
        })();
        match outcome {
            Ok(()) => status.ok = true,
            Err((code, error)) => {
                log::warn!("[link] rejected an edit: {error}");
                status.code = Some(code);
                status.error = Some(error);
            }
        }
        status.revisions = {
            let guard = lock();
            match guard.as_ref() {
                Some(Phase::Ready { manifest, .. }) => manifest.revisions,
                _ => Revisions::default(),
            }
        };
        status
    }
}

fn capture_doc_key(state: &crate::component_state::ComponentState) -> String {
    let bytes = state
        .capture
        .as_deref()
        .unwrap_or_default()
        .iter()
        .copied()
        .chain(
            state
                .capture_edits
                .as_deref()
                .unwrap_or_default()
                .iter()
                .copied(),
        )
        .collect::<Vec<_>>();
    blake3::hash(&bytes).to_hex().to_string()[..16].to_string()
}
