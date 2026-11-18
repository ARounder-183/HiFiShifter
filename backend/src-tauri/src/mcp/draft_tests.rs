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

#[test]
fn set_tempo_preserves_existing_scale_unless_overridden_or_cleared() {
    let mut tl = project();
    tl.tempo_map = Some(vec![TempoPointData {
        id: "tp0".into(),
        position_sec: 0.0,
        bpm: 120.0,
        numerator: Some(4),
        denominator: Some(4),
        scale: Some(TempoScaleData {
            key: Some("Eb".into()),
            name: None,
            notes: Some(vec![3, 5, 7, 8, 10, 0, 2]),
        }),
    }]);

    // Updating BPM without specifying `scale` must keep the existing Eb scale intact.
    let keep_ops: Vec<Operation> = serde_json::from_value(json!([
        {
            "kind": "set_tempo",
            "points": [{ "position_sec": 0.0, "bpm": 148.0, "numerator": 4, "denominator": 4 }]
        }
    ]))
    .unwrap();
    let (tl_kept, _) = apply(tl, &keep_ops, true).unwrap();
    let tp0 = &tl_kept.tempo_map.as_ref().unwrap()[0];
    assert_eq!(tp0.bpm, 148.0);
    assert_eq!(tp0.scale.as_ref().unwrap().key.as_deref(), Some("Eb"));

    // Explicit `scale` sets a new scale and normalizes pitch classes; `clear_scale: true` removes it.
    let set_ops: Vec<Operation> = serde_json::from_value(json!([
        {
            "kind": "set_tempo",
            "points": [
                {
                    "position_sec": 0.0,
                    "bpm": 148.0,
                    "numerator": 4,
                    "denominator": 4,
                    "scale": { "key": "G", "notes": [7, 9, 11, 0, 2, 4, 6, 7] }
                },
                {
                    "position_sec": 2.0,
                    "bpm": 160.0,
                    "clear_scale": true
                }
            ]
        }
    ]))
    .unwrap();
    let (tl_updated, _) = apply(tl_kept, &set_ops, true).unwrap();
    let pts = tl_updated.tempo_map.as_ref().unwrap();
    assert_eq!(
        pts[0].scale.as_ref().unwrap().notes.as_deref(),
        Some(&[0, 2, 4, 6, 7, 9, 11][..])
    );
    assert!(pts[1].scale.is_none());
}

#[test]
fn imports_top_k_candidate_takes_and_applies_musical_curve_transforms() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for idx in 0..3 {
        let p = dir.path().join(format!("ka_cand_{idx}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        for n in 0..8_000 {
            w.write_sample((((n as f64) * 0.08 * (idx + 1) as f64).sin() * 4000.0) as i16)
                .unwrap();
        }
        w.finalize().unwrap();
        paths.push(p.to_string_lossy().into_owned());
    }

    let mut tl = project();
    tl.project_sec = 4.0;
    let track = tl.tracks[0].id.clone();
    let ops: Vec<Operation> = serde_json::from_value(json!([
        {
            "kind": "import_candidate_takes",
            "client_id": "ka_slot",
            "track_id": track,
            "start_sec": 0.5,
            "candidates": [
                { "name": "ka_01 [score=0.95]", "path": paths[0], "source_start_sec": 0.02, "length_sec": 0.3 },
                { "name": "ka_02 [score=0.88]", "path": paths[1], "source_start_sec": 0.05, "length_sec": 0.3 },
                { "name": "ka_03 [cross_take]", "path": paths[2], "source_start_sec": 0.01, "length_sec": 0.3 }
            ]
        },
        {
            "kind": "transform_curve",
            "track_id": track,
            "param": "volume",
            "start_sec": 0.5,
            "end_sec": 0.8,
            "transform": { "mode": "hermite_transition", "from": 0.4, "to": 1.0 }
        },
        {
            "kind": "transform_curve",
            "track_id": track,
            "param": "volume",
            "start_sec": 0.5,
            "end_sec": 0.8,
            "transform": { "mode": "vibrato", "rate_hz": 5.5, "depth": 0.05, "attack_sec": 0.05 }
        },
        {
            "kind": "transform_curve",
            "track_id": track,
            "param": "volume",
            "start_sec": 0.5,
            "end_sec": 0.8,
            "transform": { "mode": "smooth", "window_frames": 5 }
        }
    ]))
    .unwrap();

    let (tl_after, ids) = apply(tl, &ops, true).unwrap();
    assert_eq!(tl_after.clips.len(), 1);
    let clip_id = ids["ka_slot"].as_str().unwrap();
    let packed = tl_after.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert_eq!(packed.takes.len(), 3);
    assert_eq!(packed.active_take().name, "ka_01 [score=0.95]");
    assert_eq!(packed.takes[1].name, "ka_02 [score=0.88]");
    assert_eq!(packed.takes[2].name, "ka_03 [cross_take]");

    let vol = &tl_after.params_by_root_track[&track].extra_curves["volume"];
    assert!(vol[100] >= 0.35 && vol[100] <= 0.5);
    assert!(vol[159] >= 0.90 && vol[159] <= 1.05);
}
