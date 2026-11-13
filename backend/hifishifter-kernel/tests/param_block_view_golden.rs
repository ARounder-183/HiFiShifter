//! Compose parity fixture: Rust and the frontend preview port must agree on
//! visible intervals and read-only source identity for crossing Clips.
use hifishifter_kernel::param_blocks::{compose_param_block_view, ParamBlockEditSet};
use hifishifter_kernel::pitch_analysis::{SyllableSpan, TakeSyllableAnalysis};
use hifishifter_kernel::state::{Clip, ClipTake, TimelineState};

fn clip(
    id: &str,
    start_sec: f64,
    length_sec: f64,
    source_start_sec: f64,
    source_end_sec: f64,
    span: (f64, f64, f32),
) -> Clip {
    let mut clip: Clip = serde_json::from_value(serde_json::json!({
        "id": id,
        "track_id": "root",
        "name": id,
        "start_sec": start_sec,
        "length_sec": length_sec,
        "takes": [{
            "id": "take",
            "source_path": "a.wav",
            "source_start_sec": source_start_sec,
            "source_end_sec": source_end_sec
        }]
    }))
    .unwrap();
    clip.normalize_takes();
    let take: &mut ClipTake = clip.takes.first_mut().unwrap();
    take.syllable_analysis = Some(TakeSyllableAnalysis {
        algorithm: "test".into(),
        algorithm_version: 1,
        source_digest: "d".into(),
        source_spans_sec: vec![SyllableSpan {
            start_sec: span.0,
            end_sec: span.1,
            midi_note: Some(span.2),
            event_id: Some(hifishifter_kernel::pitch_analysis::source_event_id(
                span.0, span.1,
            )),
        }],
        analyzed_window_sec: [0.0, 8.0],
    });
    clip
}

#[test]
fn compose_matches_the_shared_view_identity_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../frontend/src/components/layout/pianoRoll/__fixtures__/paramBlockView.golden.json",
    );
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
        "tracks": [{"id":"root","name":"Root","order":0}],
        "clips": [], "project_sec": 4, "bpm": 120
    }))
    .unwrap();
    timeline.clips = vec![
        clip("A", 0.0, 2.0, 1.0, 3.0, (0.6, 1.4, 64.0)),
        clip("B", 0.5, 1.0, 1.2, 2.2, (1.18, 1.5, 62.0)),
    ];
    let view = compose_param_block_view(
        &timeline,
        "root",
        &ParamBlockEditSet::empty(),
        fixture["framePeriodMs"].as_f64().unwrap(),
    )
    .unwrap();
    let actual: Vec<serde_json::Value> = view
        .blocks
        .iter()
        .map(|block| {
            serde_json::json!({
                "startFrame": block.start_frame,
                "endFrameExclusive": block.end_frame_exclusive,
                "kind": match block.kind {
                    hifishifter_kernel::param_blocks::ProjectedBlockKind::Base => "base",
                    hifishifter_kernel::param_blocks::ProjectedBlockKind::User => "user",
                    hifishifter_kernel::param_blocks::ProjectedBlockKind::Grouped => "grouped",
                    hifishifter_kernel::param_blocks::ProjectedBlockKind::Manual => "manual",
                },
                "cutStart": block.cut_start,
                "cutEnd": block.cut_end,
                "sourceKey": block.source_key,
                "sourceNote": block.source_note,
            })
        })
        .collect();
    let expected = fixture["expected"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        for key in ["startFrame", "endFrameExclusive", "sourceNote"] {
            let left = actual[key].as_f64().unwrap();
            let right = expected[key].as_f64().unwrap();
            assert!((left - right).abs() < 1e-6, "{key}: {left} != {right}");
        }
        for key in ["kind", "cutStart", "cutEnd", "sourceKey"] {
            assert_eq!(&actual[key], &expected[key], "{key}");
        }
    }
}
