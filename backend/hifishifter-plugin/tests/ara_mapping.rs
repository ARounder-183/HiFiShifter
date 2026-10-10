//! ARA 映射测试 + 逐样本渲染比对。
//!
//! 前半部分是从探针 Task 3 搬过来的字段级无损/降级测试；后半部分是探针里**没能做**的
//! Step 6：把映射出的 `TimelineState` 真的交给本体内核渲染，与"手工拼出来的同内容
//! 时间线"逐样本比较。两者现在都能做，是因为 `hifishifter_kernel` 已经把
//! `render_mixdown_interleaved` 暴露出来了。

use hifishifter_plugin::ara::{
    ara_document_from_json, ara_document_to_timeline, ara_document_to_timeline_reporting,
    clip_starts_line, has_observed_time_stretch, source_sample_rates, summary_line, AraDocument,
    LostField, SkipReason, LOST_FIELDS,
};
use hifishifter_plugin::render::render_timeline;
use serde_json::json;
use std::path::{Path, PathBuf};

const CLEAN_FIXTURE: &str = "ara-model.reaper.json";
const AWKWARD_FIXTURE: &str = "ara-model.awkward.json";

/// 测试夹具目录（随 crate 一起提交）。
///
/// 【为什么不放在 `probe/` 下】这些夹具原在 `probe/ara/{captures,fixtures}`，而
/// `probe/` 被当作可丢弃的开发树整体删掉了（连同 SDK 缓存）—— 于是本测试在 CI 上
/// **静默全红**：它读的是已经不存在的路径。夹具现在住在 crate 自己的 `tests/fixtures`
/// 里，测试不再依赖任何仓库外的目录。
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn fixture_json(name: &str) -> String {
    let path = fixtures_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

/// 样本里的素材路径是采集机器的绝对路径；把每条源重指到本工作树的同名素材，
/// 这样渲染比对在任何 checkout 上都能跑。
fn localize(doc: &mut AraDocument) {
    let dir = fixtures_dir();
    let remap = |path: &str| -> String {
        match Path::new(path).file_name() {
            Some(name) => dir.join(name).to_string_lossy().into_owned(),
            None => path.to_owned(),
        }
    };
    for source in &mut doc.audio_sources {
        source.persistent_id = remap(&source.persistent_id);
    }
    for modification in &mut doc.audio_modifications {
        modification.persistent_id = remap(&modification.persistent_id);
        modification.audio_source_persistent_id = remap(&modification.audio_source_persistent_id);
    }
    for region in &mut doc.playback_regions {
        region.audio_source_persistent_id = remap(&region.audio_source_persistent_id);
        region.audio_modification_persistent_id = remap(&region.audio_modification_persistent_id);
    }
}

fn load(name: &str) -> AraDocument {
    let mut doc = ara_document_from_json(&fixture_json(name)).expect("fixture parses");
    localize(&mut doc);
    doc
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 字段级：无损
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ara_regions_map_to_clips_without_losing_render_inputs() {
    let doc = load(CLEAN_FIXTURE);
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");

    assert_eq!(timeline.clips.len(), doc.playback_regions.len());
    assert_eq!(timeline.tracks.len(), 1);
    assert_eq!(timeline.tracks[0].name, "probe-44k");

    for (clip, region) in timeline.clips.iter().zip(doc.playback_regions.iter()) {
        assert_close(clip.start_sec, region.start_in_playback_time);
        assert_close(clip.length_sec, region.duration_in_playback_time);
        assert_eq!(
            clip.source_path.as_deref(),
            Some(region.audio_source_persistent_id.as_str())
        );
        assert_close(clip.source_start_sec, region.start_in_modification_time);
        assert_close(
            clip.playback_rate as f64,
            region.duration_in_modification_time / region.duration_in_playback_time,
        );
        assert_eq!(clip.source_sample_rate, Some(44_100));
    }
}

#[test]
fn duplicate_placements_share_one_source_but_stay_separate_clips() {
    let doc = load(AWKWARD_FIXTURE);
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert_eq!(timeline.clips.len(), 5);

    let mut starts: Vec<f64> = timeline
        .clips
        .iter()
        .filter(|clip| {
            clip.source_path
                .as_deref()
                .is_some_and(|path| path.ends_with("tone44100.wav"))
        })
        .map(|clip| clip.start_sec)
        .collect();
    starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(starts, vec![0.0, 3.0, 6.0, 9.0]);
}

#[test]
fn source_sample_rate_is_per_source_not_per_project() {
    let doc = load(AWKWARD_FIXTURE);
    let mut rates = source_sample_rates(&doc);
    rates.sort_unstable();
    assert_eq!(rates, vec![44_100, 48_000]);
}

#[test]
fn empty_document_maps_to_an_empty_timeline() {
    let doc = ara_document_from_json("{}").expect("empty document parses");
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert!(timeline.clips.is_empty());
    assert!(timeline.tracks.is_empty());
}

#[test]
fn zero_length_region_is_skipped_not_fatal() {
    // 【为什么改判据】零长度 region 是**局部**问题（在边界处分割的产物）。此前它让整份
    // 映射 `Err`，进而把 `DocumentSession::ready` 打成 `false`、整个实例永久砖化。
    // 现在逐 region 跳过：坏的那条不产生 clip，其余照常。
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [{"regionSequences": [{"name": "s", "playbackRegionCount": 2}]}],
          "audioSources": [{"persistentID": "p", "sampleRate": 44100, "sampleCount": 0}],
          "audioModifications": [{"persistentID": "p", "audioSourcePersistentID": "p"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInModificationTime": 0, "durationInModificationTime": 0,
             "startInPlaybackTime": 0, "durationInPlaybackTime": 0},
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInModificationTime": 0, "durationInModificationTime": 1,
             "startInPlaybackTime": 1, "durationInPlaybackTime": 1}
          ]
        }"#,
    )
    .expect("document parses");
    let outcome =
        ara_document_to_timeline_reporting(&doc).expect("a bad region must not fail the mapping");
    assert_eq!(outcome.timeline.clips.len(), 1, "only the good region maps");
    assert_eq!(
        outcome.clip_regions,
        vec![1],
        "the good region keeps its index"
    );
    assert_eq!(outcome.skipped.len(), 1);
    assert_eq!(outcome.skipped[0].index, 0);
    assert_eq!(
        outcome.skipped[0].reason,
        SkipReason::NonPositivePlaybackDuration
    );
    assert_eq!(outcome.skipped[0].reason.as_str(), "non_positive_duration");
}

#[test]
fn unknown_source_region_is_skipped_and_keeps_the_rest() {
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [{"regionSequences": [{"name": "s", "playbackRegionCount": 2}]}],
          "audioSources": [{"persistentID": "p", "sampleRate": 44100, "sampleCount": 44100}],
          "audioModifications": [{"persistentID": "p", "audioSourcePersistentID": "p"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "ghost", "audioModificationPersistentID": "p",
             "durationInModificationTime": 1, "durationInPlaybackTime": 1},
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInPlaybackTime": 1, "durationInModificationTime": 1, "durationInPlaybackTime": 1}
          ]
        }"#,
    )
    .expect("document parses");
    let outcome = ara_document_to_timeline_reporting(&doc).expect("mapping survives");
    assert_eq!(outcome.timeline.clips.len(), 1);
    assert_eq!(outcome.clip_regions, vec![1]);
    assert_eq!(outcome.skipped[0].reason, SkipReason::UnknownSource);
}

#[test]
fn clip_identity_stays_aligned_when_an_earlier_region_is_skipped() {
    // 【为什么这条必须有】`remap_and_log` 要把 clip 身份挂回 region key。若调用方按
    // "文档顺序 zip 存活 region"对齐，一旦前面有 region 被跳过就会整体错位 —— 把身份
    // 挂到错误的 clip 上。`clip_regions` 就是为此带回的对齐信息。
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [{"regionSequences": [{"name": "s", "playbackRegionCount": 3}]}],
          "audioSources": [{"persistentID": "p", "sampleRate": 44100, "sampleCount": 44100}],
          "audioModifications": [{"persistentID": "p", "audioSourcePersistentID": "p"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "durationInModificationTime": 0, "durationInPlaybackTime": 0},
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInPlaybackTime": 1, "durationInModificationTime": 1, "durationInPlaybackTime": 1},
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInPlaybackTime": 2, "durationInModificationTime": 1, "durationInPlaybackTime": 1}
          ]
        }"#,
    )
    .expect("document parses");
    let outcome = ara_document_to_timeline_reporting(&doc).expect("mapping survives");
    assert_eq!(outcome.clip_regions, vec![1, 2]);
    // 第 2、3 条 region 的 clip id 用的是**原文档下标**（2、3），不因跳过而前移。
    assert_eq!(outcome.timeline.clips[0].id, "ara-clip-2");
    assert_eq!(outcome.timeline.clips[1].id, "ara-clip-3");
}

#[test]
fn explicit_region_sequence_index_wins_over_count_partition() {
    // 真实 ARA 自带归属边；这里让两条 region 都指向第 2 个序列，
    // 即使序列声明的数量（1/1）暗示它们分属两条。
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [
            {"regionSequences": [
              {"name": "first", "playbackRegionCount": 1},
              {"name": "second", "playbackRegionCount": 1}
            ]}
          ],
          "audioSources": [{"persistentID": "p", "sampleRate": 44100, "sampleCount": 44100}],
          "audioModifications": [{"persistentID": "p", "audioSourcePersistentID": "p"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "regionSequenceIndex": 1, "durationInModificationTime": 1, "durationInPlaybackTime": 1},
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "regionSequenceIndex": 1, "startInPlaybackTime": 1,
             "durationInModificationTime": 1, "durationInPlaybackTime": 1}
          ]
        }"#,
    )
    .expect("document parses");
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    assert_eq!(timeline.clips.len(), 2);
    for clip in &timeline.clips {
        assert_eq!(clip.track_id, timeline.tracks[1].id);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 拉伸：Task 1 的样本没有携带拉伸
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn task1_awkward_fixture_does_not_actually_carry_a_time_stretch() {
    let doc = load(AWKWARD_FIXTURE);
    assert!(!has_observed_time_stretch(&doc));
    for region in &doc.playback_regions {
        assert_close(
            region.duration_in_modification_time,
            region.duration_in_playback_time,
        );
    }
}

#[test]
fn synthetic_duration_ratio_becomes_playback_rate() {
    let doc = ara_document_from_json(
        r#"{
          "musicalContexts": [{"regionSequences": [{"name": "s", "playbackRegionCount": 1}]}],
          "audioSources": [{"persistentID": "p", "sampleRate": 44100, "sampleCount": 88200}],
          "audioModifications": [{"persistentID": "p", "audioSourcePersistentID": "p"}],
          "playbackRegions": [
            {"audioSourcePersistentID": "p", "audioModificationPersistentID": "p",
             "startInModificationTime": 0.0, "durationInModificationTime": 2.0,
             "startInPlaybackTime": 4.0, "durationInPlaybackTime": 1.0}
          ]
        }"#,
    )
    .expect("document parses");
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    let clip = &timeline.clips[0];
    assert_close(clip.playback_rate as f64, 2.0);
    assert_close(clip.length_sec, 1.0);
    assert_close(clip.source_end_sec, 2.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 丢失字段：显式降级
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lost_fields_are_explicitly_degraded_not_silently_guessed() {
    let doc = load(AWKWARD_FIXTURE);
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");
    let clip = &timeline.clips[0];
    assert!(!clip.reversed);
    assert_close(clip.fade_in_sec, 0.0);
    assert_close(clip.fade_out_sec, 0.0);
    assert_close(clip.fade_in_shape, 0.0);
    assert!(!clip.loop_enabled);
    assert!(timeline.tempo_map.is_none());
    assert_close(timeline.bpm, 120.0);
    assert!(clip.source_file_fingerprint.is_none());
}

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
// 逐样本渲染比对（探针 Step 6）
// ─────────────────────────────────────────────────────────────────────────────

/// 参考时间线：用 **扁平投影** 路径手工拼出同样的内容（`takes` 留空，
/// 由 `normalize_takes()` 从投影生成 take），与 ARA 路径的"显式写 take"互为独立构造。
fn reference_timeline(doc: &AraDocument) -> hifishifter_kernel::state::TimelineState {
    let mut tracks = Vec::new();
    let mut track_ids = Vec::new();
    let mut declared = Vec::new();
    let mut order = 0;
    for context in &doc.musical_contexts {
        for sequence in &context.region_sequences {
            let id = format!("ref-track-{}", track_ids.len());
            tracks.push(json!({
                "id": id,
                "name": sequence.name.clone().unwrap_or_default(),
                "order": order,
            }));
            track_ids.push(id);
            declared.push(sequence.playback_region_count.max(0) as usize);
            order += 1;
        }
    }

    let mut ownership = Vec::new();
    for (index, count) in declared.iter().enumerate() {
        for _ in 0..*count {
            ownership.push(index);
        }
    }

    let mut clips = Vec::new();
    for (index, region) in doc.playback_regions.iter().enumerate() {
        let sequence_index = ownership.get(index).copied().unwrap_or(0);
        let track_id = track_ids.get(sequence_index).cloned().unwrap_or_default();
        let source = doc
            .audio_sources
            .iter()
            .find(|source| source.persistent_id == region.audio_source_persistent_id)
            .expect("reference source exists");
        let name = region
            .name
            .clone()
            .or_else(|| source.name.clone())
            .unwrap_or_else(|| format!("ref-clip-{}", index + 1));
        clips.push(json!({
            "id": format!("ref-clip-{}", index + 1),
            "track_id": track_id,
            "name": name,
            "start_sec": region.start_in_playback_time,
            "length_sec": region.duration_in_playback_time,
            // 扁平投影路径：不写 takes。
            "source_path": source.persistent_id,
            "source_start_sec": region.start_in_modification_time,
            "source_end_sec":
                region.start_in_modification_time + region.duration_in_modification_time,
            "playback_rate":
                region.duration_in_modification_time / region.duration_in_playback_time,
            "source_sample_rate": source.sample_rate as u32,
            "source_channels": source.channel_count.max(0) as u16,
            "duration_sec": source.duration_seconds,
        }));
    }

    let project_sec = clips
        .iter()
        .map(|clip| {
            clip.get("start_sec")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                + clip
                    .get("length_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
        })
        .fold(0.0_f64, f64::max);

    let mut timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(json!({
        "tracks": tracks,
        "clips": clips,
        "bpm": 120.0,
        "project_sec": project_sec,
    }))
    .expect("reference timeline deserializes");
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    timeline
}

#[test]
fn ara_mapped_timeline_renders_identically_to_a_hand_built_reference() {
    let doc = load(AWKWARD_FIXTURE);
    let mapped = ara_document_to_timeline(&doc).expect("mapping must succeed");
    let reference = reference_timeline(&doc);

    // 结构先对上：同样的 clip 数、同样的位置。
    assert_eq!(mapped.clips.len(), reference.clips.len());
    let mut mapped_starts: Vec<f64> = mapped.clips.iter().map(|c| c.start_sec).collect();
    let mut reference_starts: Vec<f64> = reference.clips.iter().map(|c| c.start_sec).collect();
    mapped_starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    reference_starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(mapped_starts, reference_starts);

    let sample_rate = 44_100;
    let end_sec = 11.0;
    let ara_audio =
        render_timeline(&mapped, sample_rate, 0.0, end_sec).expect("render ARA mapping");
    let reference_audio =
        render_timeline(&reference, sample_rate, 0.0, end_sec).expect("render reference");

    assert_eq!(ara_audio.sample_rate, reference_audio.sample_rate);
    assert_eq!(ara_audio.channels, reference_audio.channels);
    assert_eq!(
        ara_audio.samples.len(),
        reference_audio.samples.len(),
        "两条路径的渲染长度必须一致"
    );

    let max_abs_diff = ara_audio
        .samples
        .iter()
        .zip(reference_audio.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-6,
        "逐样本最大绝对差 {max_abs_diff} 超过 1e-6"
    );
}

#[test]
fn summary_line_has_a_stable_shape() {
    let doc = load(CLEAN_FIXTURE);
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");

    assert_eq!(
        summary_line(&doc, &timeline),
        "ara: sources=1 modifications=1 regionSequences=1 playbackRegions=1 clips=1"
    );
}

#[test]
fn clip_starts_line_has_a_stable_shape() {
    let doc = load(CLEAN_FIXTURE);
    let timeline = ara_document_to_timeline(&doc).expect("mapping must succeed");

    assert_eq!(clip_starts_line(&timeline), "ara: clipStartsSec=[0.000000]");
}
