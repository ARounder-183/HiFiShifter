//! 工程文件策略与通道扫描的集成测试。
//!
//! 【为什么它们住在 app 而不是内核】这几条测试断言的是"打开工程时的通道扫描"——
//! 而扫描的实现 `commands::channel_scan` 属于 app 层（IPC 包装一侧）。
//! 它们原先写在 `project.rs` 的 `#[cfg(test)] mod tests` 里，`project.rs` 搬进内核后
//! 那层引用就断了。搬到 app 来是**顺理成章**的：测试跨的正是 app 与内核的边界，
//! 所以它本来就该住在能同时看见两边的那一侧。
//!
//! 其余不依赖扫描的工程文件测试（序列化、迁移、紧凑 JSON 等）留在内核的
//! `project.rs` 里 —— 它们只测内核自己的行为。

use crate::project::*;
use crate::state::{PitchAnalysisAlgo, SynthPipelineKind, TimelineState};
use std::path::Path;

fn project_file_with_clip(tl: TimelineState) -> ProjectFile {
    ProjectFile::new(
        "test".to_string(),
        tl,
        "D".to_string(),
        3,
        8,
        "1/8".to_string(),
    )
}

/// 写一个临时 WAV；`identical` 为 true 时 L == R（假立体声）。
fn write_test_wav(name: &str, identical: bool) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("hifishifter_project_policy_test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(name);
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&path, spec).expect("wav");
    for i in 0..44_100 {
        let v = ((i as f32) * 0.01).sin() * 0.4;
        w.write_sample(v).expect("l");
        w.write_sample(if identical { v } else { -v }).expect("r");
    }
    w.finalize().expect("finalize");
    path
}

/// 造一个"v4 形态"的工程：Take 存在但没有 channel_mode 字段。
fn timeline_with_legacy_take(source: &std::path::Path) -> TimelineState {
    let mut tl = TimelineState::default();
    let root = tl.tracks[0].id.clone();
    let clip_id = tl.add_clip(
        Some(root),
        Some("V".to_string()),
        Some(0.0),
        Some(1.0),
        Some(source.to_string_lossy().to_string()),
    );
    let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).expect("clip");
    clip.sync_take_from_flat();
    for take in &mut clip.takes {
        take.channel_mode = 0;
        take.source_channels = None;
    }
    tl
}

#[test]
fn legacy_project_takes_stay_eligible_for_the_resumable_scan() {
    let _guard = crate::config::channel_policy_test_guard();
    crate::config::set_channel_import_policy(&crate::config::ChannelImportPolicy::default());
    let path = write_test_wav("legacy_fake.wav", true);
    let tl = timeline_with_legacy_take(&path);

    let (finalized, _missing) = finalize_timeline_for_session(tl, Path::new("C:/proj/t.hshp"), 4);

    // v4 的 Take **不再在打开时同步折叠**（那会让大工程冻结分钟级，且任何
    // 一次读不到都会因工程随即存成 v5 而永久漏判）。打开只负责把它标成
    // "从未判定"，折叠交给可恢复扫描。
    assert_eq!(
        finalized.clips[0].takes[0].channel_mode, 0,
        "打开阶段不得改写模式"
    );
    assert_eq!(
        finalized.clips[0].takes[0].channel_decision, None,
        "v4 的 Take 必须保持'从未判定'，否则不会被扫描纳入候选"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn legacy_take_is_actually_folded_by_the_resumable_scan() {
    // 端到端：打开（不折叠）→ 可恢复扫描（折叠）。这是"漏判不再永久化"
    // 的主链路。
    let _guard = crate::config::channel_policy_test_guard();
    crate::config::set_channel_import_policy(&crate::config::ChannelImportPolicy::default());
    let path = write_test_wav("legacy_fold_e2e.wav", true);
    let tl = timeline_with_legacy_take(&path);

    let (mut finalized, _missing) =
        finalize_timeline_for_session(tl, Path::new("C:/proj/t.hshp"), 4);

    let policy = crate::config::channel_import_policy();
    let targets =
        crate::commands::channel_scan::collect_targets(&finalized, None, &policy, false, true);
    assert_eq!(targets.targets.len(), 1, "v4 的 Take 必须在扫描候选里");
    let planned = crate::commands::channel_scan::plan(targets.targets, &policy, false);
    assert_eq!(
        planned[0].outcome,
        crate::channel_policy::ChannelScanOutcome::FakeStereo
    );
    let resolution = planned[0].resolution;
    let applied =
        crate::channel_policy::apply_resolution(&mut finalized.clips[0].takes[0], resolution);
    assert!(applied.mode_changed, "假立体声应被折叠");
    assert_eq!(finalized.clips[0].takes[0].channel_mode, 2);
    // 折叠后档案权威 ⇒ 下一次打开不必再判（零解码）。
    let ctx = crate::channel_policy::DecisionContext::for_take(
        &finalized.clips[0].takes[0],
        &policy.detect_options(),
    );
    assert!(
        finalized.clips[0].takes[0]
            .channel_decision
            .expect("record")
            .is_authoritative_for(
                finalized.clips[0].takes[0].source_file_fingerprint,
                ctx.policy_sig,
                ctx.region_q
            ),
        "折叠结论必须落成权威档案，否则每次打开都会重判"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn legacy_project_keeps_true_stereo_takes() {
    let _guard = crate::config::channel_policy_test_guard();
    crate::config::set_channel_import_policy(&crate::config::ChannelImportPolicy::default());
    let path = write_test_wav("legacy_true.wav", false);
    let tl = timeline_with_legacy_take(&path);

    let (finalized, _missing) = finalize_timeline_for_session(tl, Path::new("C:/proj/t.hshp"), 4);

    assert_eq!(
        finalized.clips[0].takes[0].channel_mode, 0,
        "真立体声不得被折叠"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_trusted_user_seal_survives_the_load_boundary() {
    // 用户在 v5 工程里显式把某 Take 设为 Normal(0)（= 明确不要折叠）：
    // 档案里记着"选了什么"，加载边界必须原样保留，扫描也不得碰它。
    let _guard = crate::config::channel_policy_test_guard();
    crate::config::set_channel_import_policy(&crate::config::ChannelImportPolicy::default());
    let path = write_test_wav("v5_explicit.wav", true);
    let mut tl = timeline_with_legacy_take(&path);
    tl.clips[0].takes[0].channel_decision =
        Some(crate::channel_decision::ChannelDecisionRecord::user(0));

    let (finalized, _missing) = finalize_timeline_for_session(tl, Path::new("C:/proj/t.hshp"), 5);

    assert_eq!(
        finalized.clips[0].takes[0].channel_mode, 0,
        "用户显式选择的模式不得被改写"
    );
    let record = finalized.clips[0].takes[0]
        .channel_decision
        .expect("可信的用户封印必须保留");
    assert!(record.is_trusted_user_seal());
    assert_eq!(record.chosen_mode, Some(0), "封印必须记得用户选了什么");
    let policy = crate::config::channel_import_policy();
    assert!(
        crate::commands::channel_scan::collect_targets(&finalized, None, &policy, false, true)
            .targets
            .is_empty(),
        "用户真实选择过的 Take 不得进入自动扫描候选"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_fabricated_user_seal_is_cleared_so_the_scan_can_fold_again() {
    // 回归：历史上加载边界把"没有档案"批量伪造成"用户决定"，而本程序写出的
    // 每个工程都是 v5 —— 于是保存过一次的工程里**所有** Take 都带着伪造封印，
    // 折叠功能（含右键"扫描假立体声并转换"）对所有工程彻底失效。
    // 这类档案没有 `chosen_mode`（没记用户选了什么），必须清回"未判定"。
    let _guard = crate::config::channel_policy_test_guard();
    crate::config::set_channel_import_policy(&crate::config::ChannelImportPolicy::default());
    let path = write_test_wav("v5_fabricated.wav", true);
    let mut tl = timeline_with_legacy_take(&path);
    // 伪造封印：标着 ORIGIN_USER，但没有任何"选了啥"的记录。
    tl.clips[0].takes[0].channel_decision = Some(crate::channel_decision::ChannelDecisionRecord {
        origin: crate::channel_decision::ORIGIN_USER,
        verdict: crate::channel_decision::VERDICT_USER,
        chosen_mode: None,
        fingerprint: None,
        policy_sig: 0,
        region_q: None,
    });

    let (mut finalized, _missing) =
        finalize_timeline_for_session(tl, Path::new("C:/proj/t.hshp"), 5);

    assert_eq!(
        finalized.clips[0].takes[0].channel_decision, None,
        "伪造的用户封印必须被清除（否则该 Take 永久免疫于折叠）"
    );

    // 清除之后，同一文件里的假立体声必须能被重新判定并折叠 —— 这就是
    // "右键菜单什么都没转换"的修复点。
    let policy = crate::config::channel_import_policy();
    let targets =
        crate::commands::channel_scan::collect_targets(&finalized, None, &policy, true, true);
    assert_eq!(targets.targets.len(), 1, "清掉伪造封印后 Take 必须重回候选");
    let planned = crate::commands::channel_scan::plan(targets.targets, &policy, false);
    assert_eq!(
        planned[0].outcome,
        crate::channel_policy::ChannelScanOutcome::FakeStereo
    );
    let applied = crate::channel_policy::apply_resolution(
        &mut finalized.clips[0].takes[0],
        planned[0].resolution,
    );
    assert!(applied.mode_changed, "假立体声必须被折叠");
    assert_eq!(finalized.clips[0].takes[0].channel_mode, 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_v5_take_with_no_record_stays_eligible_for_the_scan() {
    // 回归：v5 工程里"没有档案"就是"从未判定"，绝不是"用户决定"。
    // 曾经版本号 ≥ 5 被当作"用户已决定"，把整个功能锁死。
    let _guard = crate::config::channel_policy_test_guard();
    crate::config::set_channel_import_policy(&crate::config::ChannelImportPolicy::default());
    let path = write_test_wav("v5_undecided.wav", true);
    let tl = timeline_with_legacy_take(&path);

    let (finalized, _missing) = finalize_timeline_for_session(tl, Path::new("C:/proj/t.hshp"), 5);

    assert_eq!(
        finalized.clips[0].takes[0].channel_decision, None,
        "v5 工程的空档案必须保持'未判定'，不得被伪造为'用户决定'"
    );
    let policy = crate::config::channel_import_policy();
    assert_eq!(
        crate::commands::channel_scan::collect_targets(&finalized, None, &policy, false, true)
            .targets
            .len(),
        1,
        "未判定的 Take 必须在自动扫描候选里"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn probe_v4_flat_project_targets() {
    let dir = std::env::temp_dir().join("hifishifter_probe_v4");
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("probe.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&wav, spec).unwrap();
    for i in 0..44_100 {
        let v = ((i as f32) * 0.01).sin() * 0.4;
        w.write_sample(v).unwrap();
        w.write_sample(v).unwrap();
    }
    w.finalize().unwrap();
    let src = wav.to_string_lossy().replace('\\', "/");

    let value = serde_json::json!({
        "version": 4,
        "name": "legacy",
        "timeline": {
            "tracks": [{
                "id": "track_1", "name": "Track", "order": 0,
                "muted": false, "solo": false, "volume": 1.0,
                "compose_enabled": false,
                "pitch_analysis_algo": "nsf_hifigan_onnx",
                "color": "#4a8fd1"
            }],
            "clips": [{
                "id": "clip_1", "track_id": "track_1", "name": "Legacy Clip",
                "start_sec": 0.0, "length_sec": 1.0, "color": "blue",
                "source_path": src,
                "duration_sec": 1.0, "duration_frames": 44100,
                "source_sample_rate": 44100,
                "gain": 1.0, "muted": false,
                "source_start_sec": 0.0, "source_end_sec": 1.0,
                "playback_rate": 1.0, "reversed": false,
                "channel_mode": 0,
                "fade_in_sec": 0.0, "fade_out_sec": 0.0,
                "fade_in_curve": "sine", "fade_out_curve": "sine"
            }],
            "bpm": 120.0, "playhead_sec": 0.0, "project_sec": 1.0,
            "next_track_order": 1
        }
    });
    let bytes = serde_json::to_vec(&value).unwrap();
    let loaded = match crate::project::load_project_file(&bytes) {
        Ok(v) => v,
        Err(e) => {
            println!("PARSE ERR: {e}");
            return;
        }
    };
    let (fin, _m) = crate::project::finalize_timeline_for_session(
        loaded.timeline,
        std::path::Path::new(&wav),
        4,
    );
    let clip = &fin.clips[0];
    println!(
        "V4FLAT: takes={} clip.src={:?} take0.src={:?} take0.region={:?}",
        clip.takes.len(),
        clip.source_path,
        clip.takes.first().and_then(|t| t.source_path.clone()),
        clip.takes
            .first()
            .map(|t| (t.source_start_sec, t.source_end_sec))
    );
    let policy = crate::config::channel_import_policy().for_explicit_scan();
    let filter: std::collections::HashSet<String> = [clip.id.clone()].into_iter().collect();
    println!(
        "V4FLAT TARGETS(filtered)={} TARGETS(all)={}",
        crate::commands::channel_scan::collect_targets(&fin, Some(&filter), &policy, true, true)
            .targets
            .len(),
        crate::commands::channel_scan::collect_targets(&fin, None, &policy, true, true)
            .targets
            .len()
    );
    let _ = std::fs::remove_file(&wav);
}

#[test]
fn channel_decision_record_survives_a_project_roundtrip() {
    // 判定档案是"漏判不再永久化"的载体：它必须随工程持久化，否则每次
    // 打开都会退回"从未判定"，重判成本（解码）永远付不完。
    let mut tl = timeline_with_clip_and_zero_curves();
    let record = crate::channel_decision::ChannelDecisionRecord::auto(
        crate::channel_decision::VERDICT_FAKE_STEREO,
        Some(0xABCD_1234),
        0x55AA,
        Some((0, 5_000)),
    );
    {
        let clip = &mut tl.clips[0];
        clip.sync_take_from_flat();
        clip.takes[0].channel_decision = Some(record);
        clip.takes[0].channel_mode = 2;
        // 与生产一致：Take 是权威，改完必须物化回 Clip 投影，否则下一次
        // sync 会用旧投影把 Take 覆盖回去。
        let take = clip.takes[0].clone();
        take.apply_to_clip(clip);
    }

    let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
    let pf = project_file_with_clip(prepared);
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    let loaded = load_project_file(&bytes).expect("roundtrip");

    let take = &loaded.timeline.clips[0].takes[0];
    assert_eq!(take.channel_decision, Some(record), "判定档案必须持久化");
    assert_eq!(take.channel_mode, 2);
}

#[test]
fn saving_does_not_mint_a_decision_where_there_was_none() {
    // 反方向同样重要：没有档案的 Take 存盘后必须**仍然**没有档案，
    // 否则一次保存就会把"从未判定"伪装成"已定论"。
    let mut tl = timeline_with_clip_and_zero_curves();
    {
        let clip = &mut tl.clips[0];
        clip.sync_take_from_flat();
        clip.takes[0].channel_decision = None;
    }
    let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
    let pf = project_file_with_clip(prepared);
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    let loaded = load_project_file(&bytes).expect("roundtrip");
    assert_eq!(loaded.timeline.clips[0].takes[0].channel_decision, None);
}

fn timeline_with_clip_and_zero_curves() -> TimelineState {
    let mut tl = TimelineState::default();
    let root = tl.tracks[0].id.clone();
    tl.ensure_params_for_root(&root);
    let clip_id = tl.add_clip(
        Some(root.clone()),
        Some("Vocal".to_string()),
        Some(0.0),
        Some(3.0),
        Some("C:/audio/Vocal.wav".to_string()),
    );
    if let Some(clip) = tl.clips.iter_mut().find(|c| c.id == clip_id) {
        clip.waveform_preview = Some(vec![0.25f32; 4096]);
    }
    if let Some(params) = tl.params_by_root_track.get_mut(&root) {
        params.pitch_orig = vec![0.0f32; 6400];
        params.pitch_edit = vec![0.0f32; 6400];
        params.tension_orig = vec![0.0f32; 6400];
        params.tension_edit = vec![0.0f32; 6400];
    }
    tl
}

#[test]
fn finalize_restores_clip_audio_metadata_dropped_by_serialization() {
    // 复现「读取 -UNDO 后撤销 → Clip 里的音频内容被清空」的根因：
    // `Clip` 的媒体字段只是 active take 的内存投影，序列化时被
    // `skip_serializing` 省略；权威数据只在 `takes` 里。任何反序列化后
    // 直接使用（不做 normalize）的恢复路径都会得到「有位置没音频」的 Clip。
    let tl = timeline_with_clip_and_zero_curves();
    let round_tripped: TimelineState =
        serde_json::from_slice(&serde_json::to_vec(&tl).expect("serialize")).expect("deserialize");
    assert_eq!(
        round_tripped.clips[0].takes.len(),
        1,
        "take 是媒体字段的权威来源"
    );
    assert!(
        round_tripped.clips[0].source_path.is_none(),
        "磁盘形态不应携带 active take 投影"
    );

    let (finalized, _missing) =
        finalize_timeline_for_session(round_tripped, Path::new("C:/proj/test.hshp"), 4);
    let clip = &finalized.clips[0];
    assert!(
        clip.source_path.is_some(),
        "finalize 必须把音频源路径物化回 Clip"
    );
    assert_eq!(clip.source_path, clip.takes[0].source_path);
}

#[test]
fn json_project_output_is_compact() {
    let pf = project_file_with_clip(TimelineState::default());
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(!text.contains('\n'), "JSON project should be compact");
    // 用常量而非字面量：版本号每 bump 一次都改测试是纯噪音，
    // 这条测试要钉的是"输出是紧凑 JSON 且带 version 字段"。
    assert!(text.contains(&format!("\"version\":{}", CURRENT_PROJECT_FILE_VERSION)));
}

#[test]
fn project_file_version_can_be_read_without_full_timeline_parse() {
    let pf = project_file_with_clip(TimelineState::default());
    let json_bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    assert_eq!(
        read_project_file_version(&json_bytes),
        Some(CURRENT_PROJECT_FILE_VERSION)
    );

    let msgpack_bytes = serialize_project_file_for_path(&pf, Path::new("test.hshp")).unwrap();
    assert_eq!(
        read_project_file_version(&msgpack_bytes),
        Some(CURRENT_PROJECT_FILE_VERSION)
    );
}

#[test]
fn prepare_project_save_strips_waveform_and_zero_curves() {
    let tl = timeline_with_clip_and_zero_curves();
    let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));

    let clip = &prepared.clips[0];
    assert!(
        clip.waveform_preview.is_none(),
        "waveform cache must not be saved"
    );
    assert!(clip.source_path_relative.is_some());

    assert!(
        prepared.params_by_root_track.is_empty(),
        "all-default root params should be omitted entirely"
    );
    assert!(prepared.project_scale_notes.is_empty());
}

#[test]
fn compact_json_keeps_required_and_core_fields() {
    // 即使全部处于默认值，工程核心参数与旧版本必需的基础字段也必须始终出现。
    let tl = prepare_timeline_for_project_save(
        timeline_with_clip_and_zero_curves(),
        Path::new("test.hshp"),
    );
    let pf = project_file_with_clip(tl);
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    for key in [
        "\"base_scale\"",
        "\"beats_per_bar\"",
        "\"time_signature_denominator\"",
        "\"grid_size\"",
        "\"use_custom_scale\"",
        "\"version\"",
    ] {
        assert!(
            text.contains(key),
            "core project parameter {key} must always be serialized"
        );
    }

    // Track / Clip 容器 / TimelineState 的基础语义字段必须始终出现。
    for key in [
        "\"parent_id\"",
        "\"muted\"",
        "\"solo\"",
        "\"volume\"",
        "\"compose_enabled\"",
        "\"pitch_analysis_algo\"",
        "\"fade_in_sec\"",
        "\"fade_out_sec\"",
        "\"fade_in_shape\"",
        "\"fade_out_shape\"",
        "\"fade_in_dir\"",
        "\"fade_out_dir\"",
        "\"selected_track_id\"",
        "\"selected_clip_id\"",
        "\"playhead_sec\"",
    ] {
        assert!(text.contains(key), "field {key} must always be serialized");
    }

    // v4 起媒体字段位于 ClipTake 中；Clip 顶层不再平铺源媒体字段。
    for key in [
        "\"takes\"",
        "\"active_take_id\"",
        "\"source_path\"",
        "\"gain\"",
        "\"source_start_sec\"",
        "\"source_end_sec\"",
        "\"playback_rate\"",
        "\"reversed\"",
        "\"loop_enabled\"",
    ] {
        assert!(
            text.contains(key),
            "take field {key} must be serialized inside takes"
        );
    }
}

#[test]
fn zero_curve_project_serializes_tiny() {
    let tl = timeline_with_clip_and_zero_curves();
    let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
    let pf = project_file_with_clip(prepared);
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    assert!(
        bytes.len() < 4096,
        "cache/default-only project should serialize to <4KB, got {}",
        bytes.len()
    );
}

#[test]
fn prepare_project_save_keeps_user_edited_and_unmodified_pitch_data() {
    let mut tl = timeline_with_clip_and_zero_curves();
    let root = tl.tracks[0].id.clone();
    let params = tl.params_by_root_track.get_mut(&root).unwrap();
    params.pitch_orig = vec![60.0f32; 10];
    params.pitch_edit = vec![62.0f32; 10];
    params.pitch_edit_user_modified = true;
    params.tension_orig = vec![1.0f32; 10];
    params.tension_edit = vec![2.0f32; 10];
    params
        .extra_curves
        .insert("volume".to_string(), vec![0.5f32; 10]);

    let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
    let params = prepared
        .params_by_root_track
        .get(&root)
        .expect("root params should exist");
    assert_eq!(
        params.pitch_orig.len(),
        10,
        "edited pitch keeps orig baseline"
    );
    assert_eq!(params.pitch_edit.len(), 10);
    assert!(
        params.tension_orig.is_empty(),
        "tension_orig is a legacy cache"
    );
    assert_eq!(params.tension_edit.len(), 10);
    assert_eq!(params.extra_curves.get("volume").map(Vec::len), Some(10));

    // Unmodified track: the edit copy is redundant with orig.
    let mut unmodified = TimelineState::default();
    let unmodified_root = unmodified.tracks[0].id.clone();
    unmodified.ensure_params_for_root(&unmodified_root);
    let params = unmodified
        .params_by_root_track
        .get_mut(&unmodified_root)
        .unwrap();
    params.pitch_orig = vec![57.0f32; 8];
    params.pitch_edit = params.pitch_orig.clone();
    let prepared =
        prepare_timeline_for_project_save(unmodified, Path::new("C:/proj/unmodified.hshp"));
    let params = prepared
        .params_by_root_track
        .get(&unmodified_root)
        .expect("root params should exist");
    assert_eq!(params.pitch_orig.len(), 8);
    assert!(params.pitch_edit.is_empty());
}

#[test]
fn compact_json_roundtrip_preserves_non_default_fields() {
    let mut tl = timeline_with_clip_and_zero_curves();
    let root = tl.tracks[0].id.clone();
    {
        let track = &mut tl.tracks[0];
        track.muted = true;
        track.volume = 0.5;
        track.compose_enabled = true;
        track.pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
        track.color = "#112233".to_string();
    }
    {
        let clip = &mut tl.clips[0];
        clip.gain = 0.75;
        clip.muted = true;
        clip.source_start_sec = 0.25;
        clip.playback_rate = 1.5;
        clip.reversed = true;
        clip.loop_enabled = true;
        clip.fade_in_sec = 0.1;
        clip.fade_out_sec = 0.2;
        clip.fade_in_shape = 3.0;
        clip.fade_out_shape = 4.0;
        clip.fade_in_dir = 0.35;
        clip.fade_out_dir = -0.5;
        clip.color = "blue".to_string();
        clip.source_file_fingerprint = Some(0x1122334455667788);
    }
    {
        let params = tl.params_by_root_track.get_mut(&root).unwrap();
        params.pitch_orig = vec![60.0f32; 12];
        params.pitch_edit = vec![63.0f32; 12];
        params.pitch_edit_user_modified = true;
        params.tension_edit = vec![1.5f32; 12];
    }

    let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
    let mut pf = project_file_with_clip(prepared);
    pf.use_custom_scale = true;
    pf.custom_scale = Some(CustomScale {
        id: "custom".to_string(),
        name: "Custom".to_string(),
        notes: vec![0, 2, 3, 5, 7, 9, 10],
    });
    pf.notes_markdown = "# Notes".to_string();
    pf.synth_config.default_pipeline = Some(SynthPipelineKind::WorldVocoder);

    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    let loaded = load_project_file(&bytes).expect("compact JSON must roundtrip");

    assert_eq!(loaded.name, "test");
    assert_eq!(loaded.base_scale, "D");
    assert_eq!(loaded.beats_per_bar, 3);
    assert_eq!(loaded.time_signature_denominator, 8);
    assert_eq!(loaded.grid_size, "1/8");
    assert_eq!(loaded.notes_markdown, "# Notes");
    assert!(loaded.use_custom_scale);
    assert_eq!(
        loaded.custom_scale.as_ref().map(|s| s.notes.clone()),
        Some(vec![0, 2, 3, 5, 7, 9, 10])
    );

    let track = &loaded.timeline.tracks[0];
    assert!(track.muted);
    assert!((track.volume - 0.5).abs() < f32::EPSILON);
    assert!(track.compose_enabled);
    assert_eq!(track.pitch_analysis_algo, PitchAnalysisAlgo::WorldDll);
    assert_eq!(track.color, "#112233");

    let clip = &loaded.timeline.clips[0];
    assert!((clip.gain - 0.75).abs() < f32::EPSILON);
    assert!(clip.muted);
    assert!((clip.source_start_sec - 0.25).abs() < 1e-12);
    assert!((clip.playback_rate - 1.5).abs() < f32::EPSILON);
    assert!(clip.reversed);
    assert!(clip.loop_enabled, "loop flag must roundtrip");
    assert!((clip.fade_in_sec - 0.1).abs() < 1e-12);
    assert!((clip.fade_out_sec - 0.2).abs() < 1e-12);
    assert_eq!(clip.fade_in_shape, 3.0);
    assert_eq!(clip.fade_out_shape, 4.0);
    assert_eq!(clip.fade_in_dir, 0.35);
    assert_eq!(clip.fade_out_dir, -0.5);
    assert_eq!(
        clip.source_file_fingerprint,
        Some(0x1122334455667788),
        "source fingerprint must be persisted for later hash matching"
    );
    assert!(
        clip.waveform_preview.is_none(),
        "waveform preview is stripped"
    );

    let params = loaded.timeline.params_by_root_track.get(&root).unwrap();
    assert_eq!(params.pitch_orig.len(), 12);
    assert_eq!(params.pitch_edit.len(), 12);
    assert!(params.pitch_edit_user_modified);
    assert_eq!(params.tension_edit.len(), 12);
    assert_eq!(
        loaded.synth_config.default_pipeline,
        Some(SynthPipelineKind::WorldVocoder)
    );

    let take = &clip.takes[0];
    assert_eq!(take.id, clip.active_take_id.as_deref().unwrap());
    assert!((take.gain - 0.75).abs() < f32::EPSILON);
    assert_eq!(take.source_start_sec, 0.25);
    assert_eq!(take.source_file_fingerprint, Some(0x1122334455667788));
}

#[test]
fn legacy_flat_clip_json_migrates_to_single_take() {
    let value = serde_json::json!({
        "version": 3,
        "name": "legacy",
        "timeline": {
            "tracks": [{
                "id": "track_1",
                "name": "Track",
                "order": 0,
                "muted": false,
                "solo": false,
                "volume": 1.0,
                "compose_enabled": false,
                "pitch_analysis_algo": "nsf_hifigan_onnx",
                "color": "#4a8fd1"
            }],
            "clips": [{
                "id": "clip_1",
                "track_id": "track_1",
                "name": "Legacy Clip",
                "start_sec": 0.0,
                "length_sec": 2.0,
                "color": "blue",
                "source_path": "C:/audio/a.wav",
                "duration_sec": 10.0,
                "duration_frames": 480000,
                "source_sample_rate": 48000,
                "waveform_preview": null,
                "pitch_range": null,
                "gain": 0.5,
                "muted": false,
                "source_start_sec": 1.0,
                "source_end_sec": 0.0,
                "playback_rate": 1.0,
                "reversed": false,
                "fade_in_sec": 0.0,
                "fade_out_sec": 0.0,
                "fade_in_curve": "sine",
                "fade_out_curve": "sine"
            }],
            "bpm": 120.0,
            "playhead_sec": 0.0,
            "project_sec": 32.0,
            "next_track_order": 1
        }
    });
    let bytes = serde_json::to_vec(&value).unwrap();
    let pf = load_project_file(&bytes).expect("legacy JSON should load");

    let clip = &pf.timeline.clips[0];
    assert_eq!(clip.takes.len(), 1);
    let take = &clip.takes[0];
    assert_eq!(clip.active_take_id.as_deref(), Some(take.id.as_str()));
    assert_eq!(take.source_path.as_deref(), Some("C:/audio/a.wav"));
    assert!((take.gain - 0.5).abs() < f32::EPSILON);
    assert_eq!(take.source_start_sec, 1.0);
    assert_eq!(
        take.source_end_sec, 0.0,
        "v3 哨兵保留给 open_project 版本迁移"
    );
    assert_eq!(clip.source_path.as_deref(), Some("C:/audio/a.wav"));
    assert!((clip.gain - 0.5).abs() < f32::EPSILON);
    // 旧命名曲线在加载时换算为 REAPER 形状/曲率模型（sine → 轻微 S）。
    assert_eq!(clip.fade_in_shape, 5.0);
    assert_eq!(clip.fade_out_shape, 5.0);
    assert_eq!(clip.fade_in_dir, 0.0);
    assert_eq!(clip.fade_out_dir, 0.0);
}

#[test]
fn legacy_named_curves_migrate_to_reaper_shapes() {
    let value = serde_json::json!({
        "version": 3,
        "name": "legacy-fades",
        "timeline": {
            "tracks": [{
                "id": "track_1",
                "name": "Track",
                "order": 0,
                "muted": false,
                "solo": false,
                "volume": 1.0,
                "compose_enabled": false,
                "pitch_analysis_algo": "nsf_hifigan_onnx",
                "color": "#4a8fd1"
            }],
            "clips": [{
                "id": "clip_1",
                "track_id": "track_1",
                "name": "Fades",
                "start_sec": 0.0,
                "length_sec": 2.0,
                "color": "blue",
                "fade_in_sec": 0.25,
                "fade_out_sec": 0.4,
                "fade_in_curve": "exponential",
                "fade_out_curve": "logarithmic"
            }],
            "bpm": 120.0,
            "playhead_sec": 0.0,
            "project_sec": 32.0
        }
    });
    let bytes = serde_json::to_vec(&value).unwrap();
    let pf = load_project_file(&bytes).expect("legacy JSON should load");
    let clip = &pf.timeline.clips[0];
    // exponential（晚起）→ 形状 2；logarithmic（早起）→ 形状 1。
    assert_eq!(clip.fade_in_shape, 2.0);
    assert_eq!(clip.fade_out_shape, 1.0);
    // 迁移后旧字符串被清空，新保存不再写出。
    let serialized = serde_json::to_string(&pf).expect("project must serialize");
    assert!(
        !serialized.contains("\"fade_in_curve\""),
        "legacy curve strings must not be serialized after migration"
    );
}

#[test]
fn reaper_fade_fields_roundtrip_through_project_file() {
    let mut tl = timeline_with_clip_and_zero_curves();
    {
        let clip = &mut tl.clips[0];
        clip.fade_in_shape = 1.1; // REAPER 小数变体原样透传
        clip.fade_in_dir = 0.25;
        clip.fade_out_shape = 6.0;
        clip.fade_out_dir = -1.0;
    }
    let pf = project_file_with_clip(tl);
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.hshp")).unwrap();
    let loaded = load_project_file(&bytes).expect("msgpack roundtrip must load");
    let clip = &loaded.timeline.clips[0];
    assert_eq!(clip.fade_in_shape, 1.1);
    assert_eq!(clip.fade_in_dir, 0.25);
    assert_eq!(clip.fade_out_shape, 6.0);
    assert_eq!(clip.fade_out_dir, -1.0);
}

#[test]
fn v4_flat_projection_is_not_serialized_at_clip_level() {
    let tl = prepare_timeline_for_project_save(
        timeline_with_clip_and_zero_curves(),
        Path::new("C:/proj/test.hshp"),
    );
    let pf = project_file_with_clip(tl);
    let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let clip = &value["timeline"]["clips"][0];
    assert!(clip.get("takes").is_some(), "takes must be serialized");
    for flat in [
        "source_path",
        "duration_sec",
        "gain",
        "source_start_sec",
        "source_end_sec",
        "playback_rate",
        "reversed",
        "loop_enabled",
        "midi_note_data",
    ] {
        assert!(
            clip.get(flat).is_none(),
            "Clip 顶层不应再序列化 {flat}（{text}）"
        );
    }
}
