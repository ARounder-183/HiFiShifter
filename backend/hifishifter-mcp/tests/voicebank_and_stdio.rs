use hifishifter_mcp::{
    discovery::{
        list_instance_records, remove_instance_record, resolve_endpoint, write_instance_record,
        InstanceRecord,
    },
    server,
    stdio::proxy_stdio_streams,
    voicebank::*,
    EditorBackend, EditorRequest,
};
use serde_json::{json, Value};
use std::{future::Future, pin::Pin, sync::Arc};
use tokio_util::sync::CancellationToken;

struct DummyBackend;
impl EditorBackend for DummyBackend {
    fn execute(&self, request: EditorRequest) -> Pin<Box<dyn Future<Output = Value> + Send>> {
        Box::pin(async move {
            match request {
                EditorRequest::Context => json!({
                    "ok": true,
                    "revision": "rev-stdio-1",
                    "scope_revisions": { "tempo_revision": "tempo:1", "track_revisions": {} }
                }),
                _ => json!({ "ok": true }),
            }
        })
    }
}

fn sample_phrase_marks() -> PhraseMarks {
    PhraseMarks {
        schema_version: VOICEBANK_SCHEMA_VERSION,
        phrase_id: "song-a/001".into(),
        wav_rel_path: "song-a/001.wav".into(),
        content_sha256: "abc123".into(),
        sample_rate: 44_100,
        duration_sec: 1.2,
        language: "ja".into(),
        lyric: Some("ka ta".into()),
        review_state: ReviewState::AutoDraft,
        phones: vec![
            PhoneSegment {
                id: "p0".into(),
                label: "k".into(),
                start_sec: 0.05,
                end_sec: 0.12,
                role: PhoneRole::Onset,
                provenance: MarkProvenance::Auto,
                confidence: Some(0.94),
            },
            PhoneSegment {
                id: "p1".into(),
                label: "a".into(),
                start_sec: 0.12,
                end_sec: 0.48,
                role: PhoneRole::Nucleus,
                provenance: MarkProvenance::Auto,
                confidence: Some(0.98),
            },
            PhoneSegment {
                id: "p2".into(),
                label: "t".into(),
                start_sec: 0.48,
                end_sec: 0.56,
                role: PhoneRole::Onset,
                provenance: MarkProvenance::Auto,
                confidence: Some(0.91),
            },
            PhoneSegment {
                id: "p3".into(),
                label: "a".into(),
                start_sec: 0.56,
                end_sec: 0.95,
                role: PhoneRole::Nucleus,
                provenance: MarkProvenance::Auto,
                confidence: Some(0.97),
            },
        ],
        syllables: vec![
            SyllableSegment {
                id: "s0".into(),
                reading: "ka".into(),
                start_sec: 0.05,
                end_sec: 0.48,
                phone_ids: vec!["p0".into(), "p1".into()],
                vowel_onset_sec: Some(0.12),
                stable_vowel_regions_sec: vec![[0.16, 0.42]],
                f0_median_midi: Some(62.5),
                review_state: ReviewState::AutoDraft,
                provenance: MarkProvenance::Auto,
            },
            SyllableSegment {
                id: "s1".into(),
                reading: "ta".into(),
                start_sec: 0.48,
                end_sec: 0.95,
                phone_ids: vec!["p2".into(), "p3".into()],
                vowel_onset_sec: Some(0.56),
                stable_vowel_regions_sec: vec![[0.60, 0.90]],
                f0_median_midi: Some(64.0),
                review_state: ReviewState::AutoDraft,
                provenance: MarkProvenance::Auto,
            },
        ],
        transitions: vec![TransitionSegment {
            id: "tr0".into(),
            reading: "at".into(),
            kind: TransitionKind::CrossSyllableVc,
            from_syllable_id: "s0".into(),
            to_syllable_id: Some("s1".into()),
            start_sec: 0.30,
            boundary_sec: 0.48,
            end_sec: 0.62,
            review_state: ReviewState::AutoDraft,
            provenance: MarkProvenance::Auto,
        }],
        tags: vec![ScopedTag {
            tag: "soft".into(),
            scope: TagScope::Phrase,
            provenance: MarkProvenance::Manual,
        }],
    }
}

#[test]
fn unified_phrase_marks_validates_spans_and_protects_human_calibrations() {
    let mut current = sample_phrase_marks();
    current.validate().expect("valid phrase marks");

    // Simulate human calibrating s0 boundary and disabling tr0.
    current.syllables[0].vowel_onset_sec = Some(0.135);
    current.syllables[0].provenance = MarkProvenance::Manual;
    current.syllables[0].review_state = ReviewState::Confirmed;
    current.transitions[0].review_state = ReviewState::Disabled;

    let mut fresh_auto = sample_phrase_marks();
    fresh_auto.syllables[0].vowel_onset_sec = Some(0.110);
    fresh_auto.syllables[1].f0_median_midi = Some(65.25);
    fresh_auto.tags.clear();

    let merged = current.merge_auto_draft(fresh_auto);
    merged.validate().expect("merged marks remain valid");
    assert_eq!(merged.syllables[0].vowel_onset_sec, Some(0.135));
    assert_eq!(merged.syllables[0].review_state, ReviewState::Confirmed);
    assert_eq!(merged.transitions[0].review_state, ReviewState::Disabled);
    assert_eq!(merged.syllables[1].f0_median_midi, Some(65.25));
    assert_eq!(merged.tags.len(), 1);
    assert_eq!(merged.tags[0].tag, "soft");

    // Out-of-bounds vowel onset is rejected.
    let mut broken = sample_phrase_marks();
    broken.syllables[0].vowel_onset_sec = Some(0.90);
    assert!(broken.validate().is_err());
}

#[test]
fn instance_discovery_registry_resolves_by_project_and_recency() {
    let dir = std::env::temp_dir().join(format!("hfs_mcp_disc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let r1 = InstanceRecord {
        instance_id: "inst-1".into(),
        pid: 101,
        endpoint: "http://127.0.0.1:39393/mcp".into(),
        port: 39393,
        project_name: "SongA".into(),
        project_path: Some("/music/SongA.hshp".into()),
        updated_at_unix_ms: 1000,
    };
    let r2 = InstanceRecord {
        instance_id: "inst-2".into(),
        pid: 102,
        endpoint: "http://127.0.0.1:40111/mcp".into(),
        port: 40111,
        project_name: "SongB".into(),
        project_path: Some("/music/SongB.hshp".into()),
        updated_at_unix_ms: 2000,
    };
    write_instance_record(&dir, &r1).unwrap();
    write_instance_record(&dir, &r2).unwrap();
    assert_eq!(list_instance_records(&dir).len(), 2);
    assert_eq!(
        resolve_endpoint(Some(&dir), None, None),
        "http://127.0.0.1:40111/mcp"
    );
    assert_eq!(
        resolve_endpoint(Some(&dir), None, Some("SongA")),
        "http://127.0.0.1:39393/mcp"
    );
    remove_instance_record(&dir, "inst-2");
    assert_eq!(
        resolve_endpoint(Some(&dir), None, None),
        "http://127.0.0.1:39393/mcp"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn stdio_bridge_proxies_handshake_and_all_twelve_tools_over_loopback_http() {
    let (listener, addr) = server::bind_local(0).unwrap();
    let endpoint = format!("http://{addr}/mcp");
    let cancel = CancellationToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        server::serve(listener, Arc::new(DummyBackend), c2)
            .await
            .unwrap();
    });

    let input_lines = [
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "stdio-test", "version": "1.0" }
            }
        })
        .to_string(),
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
        .to_string(),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        .to_string(),
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "hfs_get_context", "arguments": {} }
        })
        .to_string(),
    ]
    .join("\n")
        + "\n";

    let mut out = Vec::new();
    proxy_stdio_streams(
        &endpoint,
        std::io::Cursor::new(input_lines.into_bytes()),
        &mut out,
    )
    .await
    .unwrap();
    cancel.cancel();

    let responses: Vec<Value> = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses.len(), 3);
    let tools = responses[1]["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 12);
    for expected in [
        "hfs_get_context",
        "hfs_get_edit_capabilities",
        "hfs_query",
        "hfs_edit",
        "hfs_history",
        "hfs_transport",
        "hfs_save_project",
        "hfs_settings",
        "hfs_editor_view",
        "hfs_wait_ready",
        "hfs_render_preview",
        "hfs_notebook",
    ] {
        assert!(
            tools.iter().any(|t| t["name"] == expected),
            "missing tool {expected}"
        );
    }
    assert_eq!(
        responses[2]["result"]["structuredContent"]["revision"],
        "rev-stdio-1"
    );

    let schemas = server::export_contract_schemas();
    assert!(schemas["tools"]["hfs_render_preview"].is_object());
    assert!(schemas["tools"]["hfs_wait_ready"].is_object());
    assert!(schemas["tools"]["hfs_notebook"].is_object());
}
