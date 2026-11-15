use super::*;
use hifishifter_mcp::{EditRequest, TrackPatch};

fn project() -> TimelineState {
    let mut tl = TimelineState::default();
    tl.tracks.clear();
    tl.clips.clear();
    tl.project_sec = 1.0;
    let id = tl.add_track(Some("Vocal".into()), None, None);
    tl.set_track_state(&id, None, None, None, Some(true), None, None, None);
    tl
}

#[test]
fn batch_failure_does_not_leak_earlier_track_or_curve_edits() {
    let tl = project();
    let before = serde_json::to_value(&tl).unwrap();
    let operations = vec![
        Operation::CreateTrack {
            client_id: "new".into(),
            name: "New".into(),
            parent_id: None,
            index: None,
        },
        Operation::PatchTrack {
            track_id: "missing".into(),
            patch: TrackPatch {
                muted: Some(true),
                ..Default::default()
            },
        },
    ];
    assert_eq!(
        apply(tl.clone(), &operations, true).unwrap_err().code,
        "TRACK_NOT_FOUND"
    );
    assert_eq!(serde_json::to_value(&tl).unwrap(), before);
}

#[test]
fn aliases_and_parameter_units_are_resolved_in_one_transaction() {
    let tl = project();
    let request:EditRequest=serde_json::from_value(json!({"expected_revision":"v1","idempotency_key":"test","operations":[
        {"kind":"create_track","client_id":"lead","name":"Lead"},
        {"kind":"patch_track","track_id":"lead","patch":{"compose_enabled":true}},
        {"kind":"write_curve","track_id":"lead","param":"volume","start_frame":0,"values":[0.5,0.8]},
        {"kind":"create_track","client_id":"child","name":"Harmony","parent_id":"lead"},
    ]})).unwrap();
    let (tl, ids) = apply(tl, &request.operations, true).unwrap();
    let lead = ids["lead"].as_str().unwrap();
    let child = ids["child"].as_str().unwrap();
    assert_eq!(
        tl.params_by_root_track[lead].extra_curves["volume"][..2],
        [0.5, 0.8]
    );
    let d = parameter(
        &tl,
        child,
        &format!("child_pitch_offset_degrees@{child}"),
        false,
    )
    .unwrap();
    assert_eq!(d["unit"], "scale_steps");
    assert_eq!(d["owner_track_id"], lead);
}

#[test]
fn rejects_unknown_parameters_invalid_values_and_disabled_processing() {
    let tl = project();
    let id = tl.tracks[0].id.clone();
    assert_eq!(
        parameter(&tl, &id, "typo", true).unwrap_err().code,
        "PARAM_NOT_SUPPORTED"
    );
    assert_eq!(
        parameter(&tl, &id, "hifigan_tension", true)
            .unwrap_err()
            .code,
        "PARAM_DISABLED"
    );
    let bad = [Operation::WriteCurve {
        track_id: id.clone(),
        param: "volume".into(),
        start_frame: 0,
        values: vec![f32::NAN],
    }];
    assert!(apply(tl.clone(), &bad, true).is_err());
    let enabled = [
        Operation::SetStatic {
            track_id: id.clone(),
            param: "breath_enabled".into(),
            value: 1.0,
        },
        Operation::WriteCurve {
            track_id: id,
            param: "hifigan_tension".into(),
            start_frame: 0,
            values: vec![10.0],
        },
    ];
    let result = apply(tl, &enabled, true);
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn cycle_is_rejected_and_input_timeline_remains_intact() {
    let mut tl = project();
    let parent = tl.tracks[0].id.clone();
    let child = tl.add_track(Some("Child".into()), Some(parent.clone()), None);
    assert!(apply(
        tl,
        &[Operation::MoveTrack {
            track_id: parent,
            parent_id: Some(child),
            index: 0
        }],
        true
    )
    .is_err());
}

#[test]
fn imports_and_splits_real_audio_with_source_coordinates_and_persisted_takes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("single note.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for i in 0..4000 {
        writer
            .write_sample(((i as f64 * 0.13).sin() * 1000.0) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
    let tl = project();
    let track = tl.tracks[0].id.clone();
    let operations = vec![
        Operation::ImportClip {
            client_id: "note".into(),
            track_id: track,
            path: path.to_string_lossy().into(),
            start_sec: 2.0,
            source_start_sec: Some(0.02),
            length_sec: Some(0.2),
            playback_rate: Some(2.0),
        },
        Operation::SplitClip {
            clip_id: "note".into(),
            at_sec: 2.1,
        },
    ];
    let (tl, _) = apply(tl, &operations, true).unwrap();
    assert_eq!(tl.clips.len(), 2);
    let mut clips = tl.clips.clone();
    clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
    assert!((clips[0].length_sec - 0.1).abs() < 1e-6);
    assert!((clips[1].source_start_sec - 0.22).abs() < 1e-6);
    assert!((clips[1].active_take().source_start_sec - clips[1].source_start_sec).abs() < 1e-6);
    assert!((clips[1].playback_rate - 2.0).abs() < 1e-6);
}

#[test]
fn rejects_gain_that_the_native_editor_would_silently_clamp() {
    let tl = project();
    let id = tl.tracks[0].id.clone();
    let operations = [Operation::PatchTrack {
        track_id: id,
        patch: TrackPatch {
            volume: Some(17.0),
            ..Default::default()
        },
    }];
    assert_eq!(
        apply(tl, &operations, true).unwrap_err().code,
        "INVALID_ARGUMENT"
    );
}
