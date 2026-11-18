use super::*;
use crate::mcp::host::HeadlessMcpHost;

fn write_sine_wav(path: &std::path::Path, freq_hz: f64, duration_sec: f64) {
    let sr = 44_100u32;
    let total = ((sr as f64) * duration_sec).round() as usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..total {
        let t = i as f64 / sr as f64;
        writer
            .write_sample(((std::f64::consts::TAU * freq_hz * t).sin() * 0.25) as f32)
            .unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn scoped_cas_allows_concurrent_human_and_agent_edits_on_separate_root_tracks() {
    let host = HeadlessMcpHost::default();
    let lead_id = {
        let mut tl = host
            .app_state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tl.tracks.clear();
        tl.clips.clear();
        let lead = tl.add_track(Some("Lead Vocal".into()), None, None);
        let harm = tl.add_track(Some("Agent Harmony".into()), None, None);
        tl.set_track_state(&lead, None, None, None, Some(true), None, None, None);
        tl.set_track_state(&harm, None, None, None, Some(true), None, None, None);
        (lead, harm)
    };
    let (lead_track, harm_track) = lead_id;

    // 1. Agent reads context before human edits Lead Vocal.
    let ctx = execute_with_host(&host, EditorRequest::Context);
    assert_eq!(ctx["ok"], true);
    let tempo_rev = ctx["scope_revisions"]["tempo_revision"]
        .as_str()
        .unwrap()
        .to_string();
    let harm_rev = ctx["scope_revisions"]["track_revisions"][&harm_track]
        .as_str()
        .unwrap()
        .to_string();
    let stale_global_rev = ctx["revision"].as_str().unwrap().to_string();

    // 2. Human concurrently edits Lead Vocal (mutates volume & bumps timeline_version).
    {
        let mut tl = host
            .app_state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tl.set_track_state(
            &lead_track,
            None,
            None,
            Some(0.72),
            None,
            None,
            None,
            Some("Lead (Human Edited)".into()),
        );
    }
    host.app_state.bump_timeline_version();

    // Whole-project CAS properly detects the change:
    let global_attempt: EditRequest = serde_json::from_value(json!({
        "expected_revision": stale_global_rev,
        "idempotency_key": "global-stale",
        "operations": [
            { "kind": "patch_track", "track_id": harm_track, "patch": { "volume": 0.5 } }
        ]
    }))
    .unwrap();
    let global_res = execute_with_host(&host, EditorRequest::Edit(global_attempt));
    assert_eq!(global_res["ok"], false);
    assert_eq!(global_res["code"], "REVISION_CONFLICT");

    // 3. Root-track scoped CAS on `harm_track` succeeds cleanly AND preserves human's Lead edit!
    let scoped_attempt: EditRequest = serde_json::from_value(json!({
        "expected_scope": {
            "mode": "root_track",
            "root_track_id": harm_track,
            "track_revision": harm_rev,
            "tempo_revision": tempo_rev
        },
        "idempotency_key": "scoped-harm-edit-1",
        "label": "Shape harmony volume",
        "operations": [
            { "kind": "patch_track", "track_id": harm_track, "patch": { "volume": 0.55 } },
            { "kind": "write_curve", "track_id": harm_track, "param": "volume", "start_frame": 0, "values": [0.5, 0.6, 0.7] }
        ]
    }))
    .unwrap();
    let scoped_res = execute_with_host(&host, EditorRequest::Edit(scoped_attempt.clone()));
    assert_eq!(scoped_res["ok"], true, "{scoped_res}");
    assert_eq!(scoped_res["committed"], true);
    assert_eq!(scoped_res["replayed"], false);

    // Idempotent retry returns cached reply with replayed=true:
    let replay_res = execute_with_host(&host, EditorRequest::Edit(scoped_attempt));
    assert_eq!(replay_res["ok"], true);
    assert_eq!(replay_res["replayed"], true);

    // IsolatedNewTracks also succeeds concurrently when creating a scratch candidate track:
    let isolated_attempt: EditRequest = serde_json::from_value(json!({
        "expected_scope": {
            "mode": "isolated_new_tracks",
            "tempo_revision": tempo_rev
        },
        "idempotency_key": "isolated-candidate-1",
        "operations": [
            { "kind": "create_track", "client_id": "cand", "name": "Agent Candidate B" },
            { "kind": "patch_track", "track_id": "cand", "patch": { "muted": true } }
        ]
    }))
    .unwrap();
    let iso_res = execute_with_host(&host, EditorRequest::Edit(isolated_attempt));
    assert_eq!(iso_res["ok"], true, "{iso_res}");

    // Verify Lead Vocal still has human's edits AND Harmony + Candidate B have Agent's edits!
    let tl = host
        .app_state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let lead = tl.tracks.iter().find(|t| t.id == lead_track).unwrap();
    assert_eq!(lead.name, "Lead (Human Edited)");
    assert!((lead.volume - 0.72).abs() < 1e-5);
    let harm = tl.tracks.iter().find(|t| t.id == harm_track).unwrap();
    assert!((harm.volume - 0.55).abs() < 1e-5);
    assert_eq!(tl.tracks.len(), 3);
}

#[test]
fn notebook_fenced_sections_preserve_human_notes_and_ledger_records_batches() {
    let host = HeadlessMcpHost::default();
    {
        let mut p = host
            .app_state
            .project
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        p.notes_markdown = "# Human Production Notes\n\nKeep chorus breathy.\n".into();
    }

    let upsert1: NotebookRequest = serde_json::from_value(json!({
        "action": {
            "kind": "upsert_agent_section",
            "section_id": "review-v1",
            "title": "Agent Review Checklist",
            "markdown_body": "- [ ] Note n04 (`ka`) @ 1.42s: cross-take consonant splice"
        }
    }))
    .unwrap();
    let r1 = execute_with_host(&host, EditorRequest::Notebook(upsert1));
    assert_eq!(r1["ok"], true, "{r1}");
    assert_eq!(r1["agent_sections"].as_array().unwrap().len(), 1);

    // Updating the same section replaces only the fenced block, leaving human notes intact.
    let upsert2: NotebookRequest = serde_json::from_value(json!({
        "action": {
            "kind": "upsert_agent_section",
            "section_id": "review-v1",
            "title": "Agent Review Checklist (Updated)",
            "markdown_body": "- [x] Note n04 verified\n- [ ] Note n09 (`shi`) OOV fallback"
        }
    }))
    .unwrap();
    let r2 = execute_with_host(&host, EditorRequest::Notebook(upsert2));
    assert_eq!(r2["ok"], true);

    let q_nb: QueryRequest = serde_json::from_value(json!({ "kind": "notebook" })).unwrap();
    let nb = execute_with_host(&host, EditorRequest::Query(q_nb));
    let md = nb["data"]["markdown"].as_str().unwrap();
    assert!(md.starts_with("# Human Production Notes\n\nKeep chorus breathy."));
    assert!(md.contains("Note n09 (`shi`) OOV fallback"));
    assert!(!md.contains("cross-take consonant splice"));

    // Remove fenced section cleanly:
    let rm: NotebookRequest = serde_json::from_value(json!({
        "action": { "kind": "remove_agent_section", "section_id": "review-v1" }
    }))
    .unwrap();
    let r3 = execute_with_host(&host, EditorRequest::Notebook(rm));
    assert_eq!(r3["ok"], true);
    assert_eq!(r3["removed"], true);
}

#[test]
fn wait_barrier_and_render_preview_deliver_closed_loop_audio_qa() {
    let dir = tempfile::tempdir().unwrap();
    let wav_in = dir.path().join("tone.wav");
    let wav_out = dir.path().join("preview.wav");
    write_sine_wav(&wav_in, 440.0, 0.5);

    let host = HeadlessMcpHost::default();
    let ctx = execute_with_host(&host, EditorRequest::Context);
    let rev = ctx["revision"].as_str().unwrap().to_string();
    let main_track = "track_main";

    let edit: EditRequest = serde_json::from_value(json!({
        "expected_revision": rev,
        "idempotency_key": "place-sine-1",
        "label": "Place test tone",
        "operations": [
            {
                "kind": "import_clip",
                "client_id": "tone",
                "track_id": main_track,
                "path": wav_in.to_string_lossy(),
                "start_sec": 0.1,
                "length_sec": 0.4
            }
        ]
    }))
    .unwrap();
    let edited = execute_with_host(&host, EditorRequest::Edit(edit));
    assert_eq!(edited["ok"], true, "{edited}");

    // Acknowledge GUI view and test hfs_wait_ready:
    let new_rev = edited["revision"].as_str().unwrap().to_string();
    *host
        .mcp_control
        .view_revision
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(new_rev);

    let wait_req: WaitRequest = serde_json::from_value(json!({
        "targets": ["clip_render", "gui_view"],
        "timeout_ms": 100
    }))
    .unwrap();
    let waited = execute_with_host(&host, EditorRequest::Wait(wait_req));
    assert_eq!(waited["ok"], true, "{waited}");
    assert_eq!(waited["ready"], true);

    // Offline render preview + acoustic QA metrics + noclobber safety:
    let preview_req: RenderPreviewRequest = serde_json::from_value(json!({
        "start_sec": 0.0,
        "end_sec": 0.6,
        "apply_pitch_edit": false,
        "output_wav_path": wav_out.to_string_lossy()
    }))
    .unwrap();
    let rendered = execute_with_host(&host, EditorRequest::RenderPreview(preview_req.clone()));
    assert_eq!(rendered["ok"], true, "{rendered}");
    assert!(wav_out.is_file());
    assert!(rendered["metrics"]["peak_dbfs"].as_f64().unwrap() > -25.0);
    assert_eq!(rendered["metrics"]["clipped_samples"], 0);
    assert_eq!(rendered["metrics"]["non_finite_samples"], 0);
    assert_eq!(
        rendered["metrics"]["boundary_checks"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // Second render to existing path without overwrite=true must refuse cleanly:
    let clobber = execute_with_host(&host, EditorRequest::RenderPreview(preview_req));
    assert_eq!(clobber["ok"], false);
    assert_eq!(clobber["code"], "SAVE_FAILED");

    // Audit ledger captured the edit batch:
    let ledger_q: QueryRequest = serde_json::from_value(json!({ "kind": "ledger" })).unwrap();
    let ledger_res = execute_with_host(&host, EditorRequest::Query(ledger_q));
    assert_eq!(ledger_res["data"]["total_retained"], 1);
    assert_eq!(
        ledger_res["data"]["entries"][0]["idempotency_key"],
        "place-sine-1"
    );
}
