//! 投影 golden fixture：Rust 与前端预览端口必须逐项相等。
use hifishifter_kernel::param_blocks::projection::{project_span, ClipProjection};
use hifishifter_kernel::param_blocks::{BlockLabel, TakeSpan};
use hifishifter_kernel::state::Clip;

#[test]
fn projection_golden_matches_the_shared_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../frontend/src/components/layout/pianoRoll/__fixtures__/paramBlockProjection.golden.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let frame_period_ms = fixture["framePeriodMs"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let clip = &case["clip"];
        let mut clip: Clip = serde_json::from_value(serde_json::json!({
            "id": clip["id"],
            "track_id": "root",
            "name": "clip",
            "start_sec": clip["startSec"],
            "length_sec": clip["lengthSec"],
            "loop_period_sec": clip["loopPeriodSec"].as_f64().or_else(|| clip["contentDurationSec"].as_f64()),
            "clip_playback_rate": clip["clipPlaybackRate"].as_f64().unwrap_or(1.0),
            "takes": [{
                "id": "take",
                "source_path": "a.wav",
                "source_start_sec": clip["sourceStartSec"],
                "source_end_sec": clip["sourceEndSec"],
                "playback_rate": clip["playbackRate"].as_f64().unwrap_or(1.0),
                "reversed": clip["reversed"],
                "loop_enabled": clip["loopEnabled"],
                "duration_sec": clip["contentDurationSec"],
            }]
        }))
        .unwrap();
        clip.normalize_takes();
        let projection = ClipProjection::from_clip(&clip).unwrap();
        let span = TakeSpan {
            clip_id: "c".into(),
            take_id: "take".into(),
            cycle: 0,
            source_start_sec: case["span"]["sourceStartSec"].as_f64().unwrap(),
            source_end_sec: case["span"]["sourceEndSec"].as_f64().unwrap(),
            syllable_id: None,
            label: BlockLabel::Syllable,
        };
        let got = project_span(&projection, &span, frame_period_ms);
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(got.len(), expected.len(), "{}", case["name"]);
        for (actual, expected) in got.iter().zip(expected) {
            assert_eq!(actual.cycle, expected["cycle"].as_u64().unwrap() as u32);
            assert_eq!(
                actual.start_frame,
                expected["startFrame"].as_u64().unwrap() as u32
            );
            assert_eq!(
                actual.end_frame_exclusive,
                expected["endFrameExclusive"].as_u64().unwrap() as u32
            );
            assert_eq!(
                actual.cut_start,
                expected["cutStart"].as_bool().unwrap_or(false),
                "{} cut_start",
                case["name"]
            );
            assert_eq!(
                actual.cut_end,
                expected["cutEnd"].as_bool().unwrap_or(false),
                "{} cut_end",
                case["name"]
            );
        }
    }
}
