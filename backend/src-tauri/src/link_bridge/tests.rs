//! 接收侧的可单测部分：导入的**纯数据**转换。
//!
//! 这里不构造 `AppState`：真正容易出错的是「素材状态怎么读」「缺音频的片段怎么办」
//! 「摘要不符怎么办」「导入有没有偷偷复制一份音频」—— 全都在 [`super::prepare_import`]
//! 这一层里，而它不碰任何应用状态。

use super::*;
use hifishifter_host_link::{
    link_id, AudioRef, AudioState, Capability, LinkManifest, Producer, Revisions, SourceEntry,
    FORMAT, FORMAT_VERSION,
};

fn temp_root(tag: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "hfs-app-link-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// 一份工作区：一条轨、一个片段、一份素材。
fn fixture(tag: &str, with_audio: bool) -> (Workspace, LinkManifest, Option<AudioRef>) {
    let workspace = Workspace::at(temp_root(tag));
    let key = "ara://source";
    let audio = with_audio.then(|| {
        let planes = vec![vec![0.1_f32, 0.2, 0.3, 0.4], vec![-0.1, -0.2, -0.3, -0.4]];
        workspace.write_audio(key, 48_000, &planes).unwrap()
    });
    let source = match &audio {
        Some(audio) => SourceEntry {
            key: key.into(),
            sample_rate: 48_000,
            channels: 2,
            frames: 4,
            state: AudioState::Complete,
            file: Some(audio.file.clone()),
            blake3: Some(audio.blake3.clone()),
            bytes: Some(audio.bytes),
            note: None,
        },
        None => SourceEntry {
            key: key.into(),
            sample_rate: 48_000,
            channels: 2,
            frames: 4,
            // 几何有效、音频暂缺：宿主还没授权这份素材。
            state: AudioState::Partial,
            file: None,
            blake3: None,
            bytes: None,
            note: Some("host has not authorised this source".into()),
        },
    };
    let manifest = LinkManifest {
        format: FORMAT.into(),
        format_version: FORMAT_VERSION,
        link_id: link_id(SourceKind::Ara, "REAPER", "fixture"),
        source_kind: SourceKind::Ara,
        host_label: "REAPER".into(),
        created_unix_ms: 1,
        producer: Producer {
            implementation: "test".into(),
            pid: None,
        },
        timeline: serde_json::json!({
            "bpm": 120.0,
            "project_sec": 1.0,
            "tracks": [{"id": "t1", "name": "Track", "order": 0}],
            "clips": [{
                "id": "c1", "track_id": "t1", "name": "Clip",
                "start_sec": 0.0, "length_sec": 4.0 / 48_000.0,
                "source_path": key,
                "source_start_sec": 0.0, "source_end_sec": 4.0 / 48_000.0,
                "playback_rate": 1.0,
            }],
        }),
        // 基线由生产者算；夹具直接给一份与时间轴同形的投影。
        host_geometry: serde_json::json!({"clips": [], "tracks": []}),
        sources: vec![source],
        granted: vec![Capability::Commit, Capability::Audio],
        revisions: Revisions { edits: 0, model: 1 },
        admission: hifishifter_host_link::Admission::Strict,
        losses: Vec::new(),
        diagnostics: None,
    };
    (workspace, manifest, audio)
}

/// 导入**不复制音频**：素材直接指向工作区里那份内容寻址的文件。
#[test]
fn importing_adopts_the_workspace_audio_instead_of_copying_it() {
    let (workspace, manifest, audio) = fixture("adopt", true);
    let audio = audio.unwrap();
    let prepared = prepare_import(&manifest, &workspace).unwrap();
    assert!(prepared.report.is_clean(), "{:?}", prepared.report);
    assert_eq!(prepared.timeline.clips.len(), 1);

    let adopted = prepared.timeline.clips[0]
        .source_path
        .as_deref()
        .expect("clip must keep a source");
    // 按**文件**比较：Windows 下同一个路径会因分隔符写成两种字符串。
    let expected = workspace.resolve(&audio.file);
    assert_eq!(
        std::fs::canonicalize(adopted).unwrap(),
        std::fs::canonicalize(&expected).unwrap(),
        "{adopted}"
    );

    // 目录里应当**只有一个**文件：物化命中已有文件并跳过了写出。
    let files = std::fs::read_dir(workspace.audio_dir()).unwrap().count();
    assert_eq!(files, 1, "导入不得在音频目录里多写一份");
    let _ = std::fs::remove_dir_all(workspace.root());
}

/// 音频暂缺的素材 ⇒ 引用它的片段本次进不来，但**指名字段地**报出来。
///
/// 【为什么这条重要】旧实现只报一个条数（"跳过了 1 项"），用户无法据此行动。
/// 这条测试钉住「是什么」和「为什么」都要在报告里。
#[test]
fn a_clip_whose_audio_is_missing_is_dropped_with_a_named_reason() {
    let (workspace, manifest, _) = fixture("partial", false);
    let prepared = prepare_import(&manifest, &workspace).unwrap();
    assert!(prepared.timeline.clips.is_empty());
    assert_eq!(prepared.report.dropped_clips.len(), 1);
    let rejection = &prepared.report.dropped_clips[0];
    assert_eq!(rejection.id, "c1");
    assert!(
        rejection.reason.contains("ara://source"),
        "原因必须指名是哪个素材：{}",
        rejection.reason
    );
    let summary = prepared.report.summary();
    assert!(summary.contains("c1"), "{summary}");
    let _ = std::fs::remove_dir_all(workspace.root());
}

/// 被换过的音频必须让导入**失败**，绝不"凑合读"。
#[test]
fn tampered_audio_fails_the_import() {
    let (workspace, manifest, audio) = fixture("tamper", true);
    let audio = audio.unwrap();
    let path = workspace.resolve(&audio.file);
    // 换成**同样的头、同样的帧数**、不同的采样内容：只有摘要能发现它。
    let fake = {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut buffer,
            hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for _ in 0..4 {
            writer.write_sample(0.9_f32).unwrap();
            writer.write_sample(-0.9_f32).unwrap();
        }
        writer.finalize().unwrap();
        buffer.into_inner()
    };
    std::fs::write(&path, fake).unwrap();
    let error = prepare_import(&manifest, &workspace).unwrap_err();
    assert!(error.contains("does not match its digest"), "{error}");
    let _ = std::fs::remove_dir_all(workspace.root());
}

/// 准入策略由**来源**决定：处置方向相反，不能由能力位推导。
#[test]
fn admission_follows_the_source_kind() {
    assert_eq!(admission_of(SourceKind::Ara), Admission::Strict);
    assert_eq!(admission_of(SourceKind::Capture), Admission::Degrade);
    assert_eq!(admission_of(SourceKind::Bridge), Admission::Degrade);
    assert_eq!(prefix_for(SourceKind::Ara), "ARA");
    assert_eq!(prefix_for(SourceKind::Capture), "Capture");
    assert_eq!(prefix_for(SourceKind::Bridge), "Host");
}

/// 一个普通目录不是工作区 —— 用户拖错东西时必须说清楚。
#[test]
fn a_plain_directory_is_not_openable() {
    let root = temp_root("plain");
    let workspace = Workspace::at(root.clone());
    let error = workspace.read_manifest().unwrap_err();
    assert!(error.contains("cannot read"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
}

/// 清单的 `link_id` 会进入每一个提交载荷；空身份必须被拒。
#[test]
fn a_manifest_without_an_identity_is_rejected() {
    let (workspace, mut manifest, _) = fixture("identity", false);
    manifest.link_id = "   ".into();
    assert!(manifest.validate().is_err());
    let _ = std::fs::remove_dir_all(workspace.root());
}
