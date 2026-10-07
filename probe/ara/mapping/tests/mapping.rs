//! HiFiShifter ARA 探针 / Task 3 测试 —— ARA → TimelineState 映射的无损性与降级。
//!
//! 判据由 plan 的 Task 3 固定：
//! - 渲染所需的每个字段都能在真实 `TimelineState` 里找到对应且值一致；
//! - ARA 不提供的字段**显式降级**，并用测试钉住，而不是悄悄填默认值。
//!
//! 一次性产物。

use ara_mapping_probe::{
    ara_document_from_json, ara_document_to_timeline, has_observed_time_stretch,
    source_sample_rates, LostField, LOST_FIELDS,
};
use std::path::PathBuf;

const CLEAN_FIXTURE: &str = "ara-model.reaper.json";
const AWKWARD_FIXTURE: &str = "ara-model.awkward.json";

fn fixture_json(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("captures")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

/// Task 1 的干净样本：一个源、一处摆放。
fn clean_doc() -> ara_mapping_probe::AraDocument {
    ara_document_from_json(&fixture_json(CLEAN_FIXTURE)).expect("parse clean fixture")
}

/// Task 1 的 awkward 样本：同源四处摆放 + 一个 48k 源。
fn awkward_doc() -> ara_mapping_probe::AraDocument {
    ara_document_from_json(&fixture_json(AWKWARD_FIXTURE)).expect("parse awkward fixture")
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1：无损性
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ara_regions_map_to_clips_without_losing_render_inputs() {
    let doc = clean_doc();
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");

    // 每个 playbackRegion 对应一个 Clip。
    assert_eq!(timeline.clips.len(), doc.playback_regions.len());
    assert_eq!(timeline.tracks.len(), 1);
    assert_eq!(timeline.tracks[0].name, "probe-44k");

    for (clip, region) in timeline.clips.iter().zip(doc.playback_regions.iter()) {
        // 位置与长度（时间线秒）。
        assert_close(clip.start_sec, region.start_in_playback_time);
        assert_close(clip.length_sec, region.duration_in_playback_time);
        // 源身份：ARA 的 persistentID 就是绝对路径，直接落成 source_path。
        assert_eq!(
            clip.source_path.as_deref(),
            Some(region.audio_source_persistent_id.as_str())
        );
        // 源内偏移与速率。
        assert_close(clip.source_start_sec, region.start_in_modification_time);
        let expected_rate = region.duration_in_modification_time / region.duration_in_playback_time;
        assert_close(clip.playback_rate as f64, expected_rate);
        // 逐源保真的采样率。
        assert_eq!(clip.source_sample_rate, Some(44_100));
    }
}

#[test]
fn duplicate_placements_share_one_source_but_stay_separate_clips() {
    let doc = awkward_doc();
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert_eq!(timeline.clips.len(), 5);

    let tone44k: Vec<_> = timeline
        .clips
        .iter()
        .filter(|clip| {
            clip.source_path
                .as_deref()
                .is_some_and(|path| path.ends_with("tone44100.wav"))
        })
        .collect();
    assert_eq!(tone44k.len(), 4, "四处摆放应各自成为独立 Clip");

    // 同一个源，四个不同位置。
    let mut starts: Vec<f64> = tone44k.iter().map(|clip| clip.start_sec).collect();
    starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(starts, vec![0.0, 3.0, 6.0, 9.0]);

    // 每个 Clip 有独立 id。
    let ids: std::collections::HashSet<&str> =
        timeline.clips.iter().map(|clip| clip.id.as_str()).collect();
    assert_eq!(ids.len(), timeline.clips.len());
}

#[test]
fn track_names_and_ownership_come_from_region_sequences() {
    let doc = awkward_doc();
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert_eq!(timeline.tracks.len(), 2);
    assert_eq!(timeline.tracks[0].name, "awkward-44k");
    assert_eq!(timeline.tracks[1].name, "awkward-48k");

    // 前四条 region 属于 44k 序列，第五条属于 48k 序列。
    for clip in timeline.clips.iter().take(4) {
        assert_eq!(clip.track_id, timeline.tracks[0].id);
    }
    assert_eq!(timeline.clips[4].track_id, timeline.tracks[1].id);
}

#[test]
fn source_sample_rate_is_per_source_not_per_project() {
    let doc = awkward_doc();
    let mut rates = source_sample_rates(&doc);
    rates.sort_unstable();
    assert_eq!(rates, vec![44_100, 48_000]);

    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    let clip_rates: std::collections::HashSet<Option<u32>> =
        timeline.clips.iter().map(|clip| clip.source_sample_rate).collect();
    assert!(clip_rates.contains(&Some(44_100)));
    assert!(clip_rates.contains(&Some(48_000)));
}

#[test]
fn empty_document_maps_to_an_empty_timeline() {
    let doc = ara_document_from_json("{}").expect("empty document parses");
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert!(timeline.clips.is_empty());
    assert!(timeline.tracks.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// 拉伸：Task 1 的结论需要更正
// ─────────────────────────────────────────────────────────────────────────────

/// Task 1 的 ledger 断言"拉伸 = 时长差"，并以 awkward 样本为证。
/// 但那份样本里**每个 region 的两个时长都相等** —— 也就是说它其实没有携带任何拉伸信号。
/// 这条测试把事实钉住，避免后续实现者沿用错误的样本去验证拉伸。
#[test]
fn task1_awkward_fixture_does_not_actually_carry_a_time_stretch() {
    let doc = awkward_doc();
    assert!(
        !has_observed_time_stretch(&doc),
        "Task 1 的 awkward 样本里所有 region 的两个时长都相等"
    );
    for region in &doc.playback_regions {
        assert_close(
            region.duration_in_modification_time,
            region.duration_in_playback_time,
        );
        assert!(!region.is_timestretch_enabled);
    }
}

/// 拉伸公式本身（时长比 → playback_rate）用合成样本单独验证，
/// 不依赖宿主是否真的施加了拉伸。
#[test]
fn synthetic_duration_ratio_becomes_playback_rate() {
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [
            {"name": null, "orderIndex": 0,
             "regionSequences": [{"name": "synthetic", "orderIndex": 0, "playbackRegionCount": 1}]}
          ],
          "audioSources": [
            {"persistentID": "C:/probe/tone.wav", "name": "tone.wav", "sampleRate": 44100,
             "sampleCount": 88200, "durationSeconds": 2, "channelCount": 1,
             "sampleAccessEnabled": true}
          ],
          "audioModifications": [
            {"persistentID": "C:/probe/tone.wav", "audioSourcePersistentID": "C:/probe/tone.wav"}
          ],
          "playbackRegions": [
            {"name": "tone.wav", "audioSourcePersistentID": "C:/probe/tone.wav",
             "audioModificationPersistentID": "C:/probe/tone.wav",
             "startInModificationTime": 0.0, "durationInModificationTime": 2.0,
             "startInPlaybackTime": 4.0, "durationInPlaybackTime": 1.0,
             "isTimestretchEnabled": true}
          ]
        }"#,
    )
    .expect("synthetic document parses");

    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert_eq!(timeline.clips.len(), 1);
    let clip = &timeline.clips[0];
    // 2 秒的源内容播 1 秒 → 2 倍速。
    assert_close(clip.playback_rate as f64, 2.0);
    assert_close(clip.length_sec, 1.0);
    assert_close(clip.source_start_sec, 0.0);
    assert_close(clip.source_end_sec, 2.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5：丢失字段必须显式降级
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lost_fields_are_explicitly_degraded_not_silently_guessed() {
    let doc = awkward_doc();
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    let clip = &timeline.clips[0];

    // 倒放：ARA 没有方向位 → 显式 false。
    assert!(!clip.reversed);
    // 淡化：ARA 只有两个布尔，且本插件不声明支持 → 长度与形状显式归零。
    assert_close(clip.fade_in_sec, 0.0);
    assert_close(clip.fade_out_sec, 0.0);
    assert_close(clip.fade_in_shape, 0.0);
    assert_close(clip.fade_in_dir, 0.0);
    // Loop：ARA 无该语义 → 显式关闭。
    assert!(!clip.loop_enabled);
    // Tempo Map：ARA 对象模型不带 tempo → 显式无 Tempo Map + 默认 BPM。
    assert!(timeline.tempo_map.is_none());
    assert_close(timeline.bpm, 120.0);
    // 内容指纹：ARA 只给路径 → 显式 None（产品的缓存键必须另行处理这一项）。
    assert!(clip.source_file_fingerprint.is_none());
}

/// 丢失字段清单本身也要被钉住：它从"一句担心"变成可核对的契约。
#[test]
fn lost_field_checklist_is_pinned() {
    assert_eq!(
        LOST_FIELDS,
        &[
            LostField::Reversed,
            LostField::FadeShapeAndCurvature,
            LostField::FadeLength,
            LostField::LoopSource,
            LostField::TempoMap,
            LostField::ItemGain,
            LostField::SourceFileFingerprint,
        ]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 边界与错误路径
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zero_length_region_is_rejected_rather_than_producing_non_finite_rate() {
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [
            {"regionSequences": [{"name": "s", "playbackRegionCount": 1}]}
          ],
          "audioSources": [
            {"persistentID": "p", "sampleRate": 44100, "sampleCount": 0, "durationSeconds": 0}
          ],
          "audioModifications": [{"persistentID": "p", "audioSourcePersistentID": "p"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInModificationTime": 0, "durationInModificationTime": 0,
             "startInPlaybackTime": 0, "durationInPlaybackTime": 0}
          ]
        }"#,
    )
    .expect("document parses");
    let error = ara_document_to_timeline(&doc).expect_err("zero-length region must be rejected");
    assert!(
        matches!(
            error,
            ara_mapping_probe::MappingError::NonPositivePlaybackDuration
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn unknown_source_is_an_error() {
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [{"regionSequences": [{"name": "s", "playbackRegionCount": 1}]}],
          "audioSources": [],
          "audioModifications": [{"persistentID": "m", "audioSourcePersistentID": "missing"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "missing", "audioModificationPersistentID": "m",
             "startInModificationTime": 0, "durationInModificationTime": 1,
             "startInPlaybackTime": 0, "durationInPlaybackTime": 1}
          ]
        }"#,
    )
    .expect("document parses");
    assert!(matches!(
        ara_document_to_timeline(&doc),
        Err(ara_mapping_probe::MappingError::UnknownSource(_))
    ));
}
