//! 参数分组编辑与组件状态回归；验证 actor/state，不代替真实宿主 GUI 验收。
use super::*;
use crate::sync::MutexExt;
use std::sync::{atomic::AtomicBool, mpsc, Arc};

#[test]
fn param_block_edits_are_independent_of_clip_members_and_survive_component_save() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    editor.ensure_loaded(false).unwrap();
    let root = editor.timeline.lock_or_recover().tracks[0].id.clone();
    let generation = editor.generation.load(Ordering::Acquire);
    let reply = dispatch(
        &editor,
        "set_param_block_edits",
        json!({
            "rootTrackId": root,
            "expectedEditRevision": 0,
            "edits": [{
                "id": "ordinary",
                "geometry": {"kind": "root", "span": {"startSec": 0.05, "endSec": 0.4}},
                "manual": true,
                "createdSeq": 1
            }],
            "checkpoint": true
        }),
    )
    .unwrap();
    assert_eq!(reply["blocks"][0]["id"], "ordinary");
    assert_eq!(
        editor.generation.load(Ordering::Acquire),
        generation,
        "boundary-only edits must not invalidate audio"
    );
    editor.ensure_loaded(true).unwrap();
    let after = dispatch(
        &editor,
        "get_param_block_view",
        json!({"rootTrackId": root}),
    )
    .unwrap();
    assert_eq!(after["blocks"][0]["id"], "ordinary");
    let encoded = owner.encode_state().unwrap();
    let (cold_model, cold_owner, _cold_identity) = crate::editor::session::tests::fixture();
    cold_owner.restore_state(&encoded).unwrap();
    let cold_editor = cold_owner.editor_session().unwrap();
    cold_editor.ensure_loaded(false).unwrap();
    let cold_root = cold_editor.timeline.lock_or_recover().tracks[0].id.clone();
    let restored = dispatch(
        &cold_editor,
        "get_param_block_view",
        json!({"rootTrackId": cold_root}),
    )
    .unwrap();
    assert_eq!(restored["blocks"][0]["id"], "ordinary");
    cold_model.session().close();
    model.session().close();
}

#[test]
fn param_block_edit_and_curve_commit_or_reject_together() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    editor.ensure_loaded(false).unwrap();
    let root = editor.timeline.lock_or_recover().tracks[0].id.clone();
    let read = dispatch(
        &editor,
        "get_param_frames",
        json!({"trackId": root, "param": "volume", "startFrame": 0, "frameCount": 4, "binary": false}),
    )
    .unwrap();
    let request = json!({
        "rootTrackId": root,
        "expectedEditRevision": 0,
        "edits": [{
            "id": "created",
            "geometry": {"kind": "root", "span": {"startSec": 0.0, "endSec": 0.02}},
            "manual": true,
            "createdSeq": 1
        }],
        "curvePatches": [{"param": "volume", "startFrame": 0, "values": [0.5, 0.6, 0.7, 0.8]}],
        "expectedCurveRevision": read["authority_revision"],
        "expectedGeometrySignature": read["geometry_signature"],
        "checkpoint": true
    });
    let result = dispatch(&editor, "set_param_block_edits", request.clone()).unwrap();
    assert_eq!(result["blocks"][0]["id"], "created");
    let curve = dispatch(
        &editor,
        "get_param_frames",
        json!({"trackId": root, "param": "volume", "startFrame": 0, "frameCount": 4, "binary": false}),
    )
    .unwrap();
    assert!((curve["edit"][0].as_f64().unwrap() - 0.5).abs() < 1e-5);
    let history_before = dispatch(&editor, "get_history_state", json!({})).unwrap();
    let mut stale = request;
    stale["expectedEditRevision"] = result["editRevision"].clone();
    stale["edits"][0]["id"] = json!("must-not-appear");
    assert!(dispatch(&editor, "set_param_block_edits", stale).is_err());
    let after = dispatch(
        &editor,
        "get_param_block_view",
        json!({"rootTrackId": root}),
    )
    .unwrap();
    assert_eq!(after["blocks"][0]["id"], "created");
    assert_eq!(
        dispatch(&editor, "get_history_state", json!({})).unwrap(),
        history_before,
        "rejected edits must not add history"
    );
    model.session().close();
}

#[test]
fn param_block_boundary_undo_keeps_audio_generation_and_empty_layout() {
    let (model, owner, _) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    editor.ensure_loaded(false).unwrap();
    let root = editor.timeline.lock_or_recover().tracks[0].id.clone();
    let initial = dispatch(
        &editor,
        "set_param_block_edits",
        json!({"rootTrackId": root, "expectedEditRevision": 0, "edits": [], "checkpoint": false}),
    )
    .unwrap();
    let generation = editor.generation.load(Ordering::Acquire);
    let added = dispatch(
        &editor,
        "set_param_block_edits",
        json!({
            "rootTrackId": root,
            "expectedEditRevision": initial["editRevision"],
            "edits": [{
                "id": "manual",
                "geometry": {"kind": "root", "span": {"startSec": 0.05, "endSec": 0.15}},
                "manual": true,
                "createdSeq": 1
            }],
            "checkpoint": true
        }),
    )
    .unwrap();
    dispatch(&editor, "undo_parameter_edit", json!({})).unwrap();
    let undone = dispatch(
        &editor,
        "get_param_block_view",
        json!({"rootTrackId": root}),
    )
    .unwrap();
    assert_eq!(undone["blocks"], json!([]));
    assert!(undone["editRevision"].as_u64().unwrap() > added["editRevision"].as_u64().unwrap());
    assert_eq!(
        editor.generation.load(Ordering::Acquire),
        generation,
        "boundary undo must not regenerate audio"
    );
    dispatch(&editor, "redo_parameter_edit", json!({})).unwrap();
    assert_eq!(
        dispatch(
            &editor,
            "get_param_block_view",
            json!({"rootTrackId": root})
        )
        .unwrap()["blocks"][0]["id"],
        "manual"
    );
    model.session().close();
}

#[test]
fn timeline_payload_omits_redundant_waveforms_and_preserves_all_block_sources() {
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    editor.ensure_loaded(false).unwrap();
    // This test exercises serialization of an isolated snapshot, not the ARA inventory pump.
    let snapshot = editor.timeline.lock_or_recover().clone();
    editor.close();
    *editor.timeline.lock_or_recover() = snapshot;
    let original_bytes = {
        let mut timeline = editor.timeline.lock_or_recover();
        let mut clip = timeline.clips[0].clone();
        clip.waveform_preview = Some(vec![0.1234567; 4096]);
        let notes = vec![hifishifter_kernel::midi_import::MidiNoteEvent {
            start_sec: 0.0,
            end_sec: 0.1,
            note: 60.0,
            velocity: 100,
            channel: 0,
        }];
        clip.midi_note_data = Some(notes.clone());
        clip.takes[0].midi_note_data = Some(notes);
        clip.takes[0].syllable_analysis =
            Some(hifishifter_kernel::pitch_analysis::TakeSyllableAnalysis {
                algorithm: hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM.into(),
                algorithm_version: hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION,
                source_digest: "payload-test".into(),
                source_spans_sec: vec![hifishifter_kernel::pitch_analysis::SyllableSpan {
                    start_sec: 0.0,
                    end_sec: 0.1,
                    midi_note: None,
                }],
                analyzed_window_sec: [0.0, 0.0],
            });
        timeline.clips = (0..256)
            .map(|index| {
                let mut clip = clip.clone();
                clip.id = format!("payload-{index}");
                clip
            })
            .collect();
        serde_json::to_vec(&timeline.to_payload()).unwrap().len()
    };
    let full = payload(&editor, false).unwrap();
    assert_eq!(
        full["clips"].as_array().unwrap().len(),
        256,
        "host refresh must not replace the fixture"
    );
    assert!(serde_json::to_vec(&full).unwrap().len() < original_bytes / 4);
    for clip in full["clips"].as_array().unwrap() {
        assert!(clip["waveform_preview"].is_null());
        assert_eq!(clip["midi_note_data"].as_array().unwrap().len(), 1);
        assert_eq!(
            clip["takes"][0]["midi_note_data"].as_array().unwrap().len(),
            1
        );
        assert_eq!(clip["takes"][0]["syllable_spans_sec"], json!([[0.0, 0.1]]));
    }
    let lite = payload(&editor, true).unwrap();
    assert_eq!(
        lite["clips"][0]["takes"][0]["syllable_spans_sec"],
        json!([[0.0, 0.1]])
    );
    model.session().close();
}

/// 同一真实actor分派中，分组编辑与非分组编辑的本地历史深度对比。
#[test]
fn grouped_edits_preserve_local_undo_checkpoint() {
    for grouped in [false, true] {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(32);
        let sink = crate::editor::session::UiSink {
            view_id: "undo-diagnostic".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let call = |id, command: &str, args| {
            editor
                .enqueue(crate::editor::session::UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sink.clone(),
                    link: None,
                })
                .unwrap();
            let response: Value = rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let timeline = call(1, "get_timeline_state", json!({}));
        if grouped {
            call(2, "begin_undo_group", json!({"label":"diagnostic"}));
        }
        call(
            3,
            "set_track_state",
            json!({"trackId":timeline["tracks"][0]["id"],"volume":0.5}),
        );
        if grouped {
            call(4, "end_undo_group", json!({}));
        }
        let history = call(5, "get_history_state", json!({}));
        let depth = history["undoDepth"].as_u64().unwrap();
        println!("grouped={grouped} local_undo_depth={depth}");
        if grouped {
            assert!(depth > 0, "分组编辑必须创建一个本地检查点");
        } else {
            assert!(depth > 0, "非分组编辑的本地历史作为对照应存在");
        }
        document.close();
    }
}

/// 隔离宿主录入问题：直接调用组件保存/恢复使用的入口，核对分组编辑立即入state且恢复能回流GUI。
#[test]
fn diagnostic_grouped_state_round_trip_without_host_history() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    let editor = owner.editor_session().unwrap();
    // 模拟原生Undo块尚未收尾：延后自动DSP，不延后参数写入与组件getState屏障。
    document.host_undo.pending.store(true, Ordering::Release);
    let (reply, received) = mpsc::channel();
    let (events, _) = mpsc::sync_channel(64);
    let sink = crate::editor::session::UiSink {
        view_id: "state-round-trip-diagnostic".into(),
        reply,
        events,
        closed: Arc::new(AtomicBool::new(false)),
    };
    let call = |id, command: &str, args| {
        editor
            .enqueue(crate::editor::session::UiRequest {
                id,
                command: command.into(),
                args,
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let response: Value = received
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        assert_eq!(response["ok"], true, "{response}");
        response["value"].clone()
    };
    let initial = call(1, "get_timeline_state", json!({}));
    let track = initial["tracks"][0]["id"].clone();
    let initial_volume = initial["tracks"][0]["volume"].clone();
    assert_ne!(initial_volume, json!(0.5));
    let before = owner.encode_state().unwrap();
    call(2, "begin_undo_group", json!({"label": "diagnostic"}));
    call(
        3,
        "set_track_state",
        json!({"trackId": track, "volume": 0.5}),
    );
    call(4, "end_undo_group", json!({}));
    let after = owner.encode_state().unwrap();
    let saved: Value = serde_json::from_slice(&after).unwrap();
    assert_eq!(saved["edits"]["tracks"][0]["volume"], json!(0.5));
    assert_ne!(before, after, "未等待音频渲染，组件state已包含分组编辑");
    owner.restore_state(&before).unwrap();
    let restored = call(5, "get_timeline_state", json!({}));
    assert_eq!(restored["tracks"][0]["volume"], initial_volume);
    owner.restore_state(&after).unwrap();
    let reapplied = call(6, "get_timeline_state", json!({}));
    assert_eq!(reapplied["tracks"][0]["volume"], json!(0.5));
    println!(
        "state contains grouped edit before DSP; restore before/after reaches actor GUI timeline"
    );
    document.close();
}

/// 音阶是 HiFiShifter 自有的设置，插件里必须能改、且必须落盘。
///
/// 【为什么这两件事一起测】REAPER 没有工程调号概念，所以音阶只能由 HiFiShifter
/// 自己拥有；而插件的 `ProjectState` 是 per-ARA-document 且从不写的 —— 只改它就等于
/// 用户选的音阶在换工程/重启后归零，而渲染缓存键与级数渲染都锚定它。
#[test]
fn the_project_scale_can_be_set_and_survives_a_restart() {
    crate::settings_store::test_support::reset();
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let editor = owner.editor_session().unwrap();
    let (reply, rx) = mpsc::channel();
    let (events, _) = mpsc::sync_channel(32);
    let sink = crate::editor::session::UiSink {
        view_id: "scale-diagnostic".into(),
        reply,
        events,
        closed: Arc::new(AtomicBool::new(false)),
    };
    let call = |id, command: &str, args| {
        editor
            .enqueue(crate::editor::session::UiRequest {
                id,
                command: command.into(),
                args,
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let response: Value = rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(response["ok"], true, "{response}");
        response["value"].clone()
    };

    // 白名单外的取值回落 C，而不是被静默接受。
    let payload = call(
        1,
        "set_project_base_scale",
        json!({"baseScale": "not-a-key"}),
    );
    assert_eq!(payload["project"]["base_scale"], "C");
    let payload = call(2, "set_project_base_scale", json!({"baseScale": "Gb"}));
    assert_eq!(payload["project"]["base_scale"], "Gb");
    assert_eq!(payload["project"]["use_custom_scale"], false);

    // 【为什么这两条是本测试的重点】`project.base_scale` 只是 GUI 显示值；内核的
    // 渲染锚点是 `timeline.project_scale_notes`（`scale_segments()` →
    // `render_scale_signature()` → 渲染缓存键）。只改前者等于"显示成 Gb、内核按 C 渲染"。
    let expected = hifishifter_kernel::state::scale_notes_for_key("Gb").unwrap();
    assert_eq!(
        editor.timeline.lock_or_recover().project_scale_notes,
        expected,
        "音阶必须真的进内核，而不是只改了 GUI 显示值"
    );
    // ★ 关键：插件的 `TimelineState` 每次 ARA 重新认领都会被整体重建
    // （`ara::mapping::ara_document_to_timeline` 只带 tracks/clips/bpm/project_sec），
    // 所以必须验"重建之后仍是 Gb" —— 只测"改完立刻变"测的是被冲掉之前的状态。
    document
        .revision
        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let payload = call(3, "get_timeline_state", json!({}));
    assert_eq!(payload["project"]["base_scale"], "Gb");
    assert_eq!(
        editor.timeline.lock_or_recover().project_scale_notes,
        expected,
        "音阶必须在每次 ARA 重建后重新播种，否则会被冲回 C 大调"
    );

    // 落盘：丢掉内存状态再读回来（等价于宿主重启）。
    crate::settings_store::test_support::simulate_restart();
    assert_eq!(
        crate::settings_store::settings()
            .plugin_musical_context
            .base_scale,
        "Gb"
    );

    document.close();
}

/// 插件里的 Tempo Map **只保存音阶轴**：BPM / 拍号是宿主权威，不落盘。
///
/// 【为什么这条值得单测】Tempo Map 就是"随时间变化的音阶"的存储，而插件此前对它整组
/// 回 `Command unavailable` —— 于是"音阶功能完整保留"这句话有一半是假的。反过来，
/// 若连 BPM 一起收下，又会造出第二个 BPM 真相（宿主才是那个真相）。
#[test]
fn the_tempo_map_keeps_only_the_scale_axis() {
    crate::settings_store::test_support::reset();
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let editor = owner.editor_session().unwrap();
    let (reply, rx) = mpsc::channel();
    let (events, _) = mpsc::sync_channel(32);
    let sink = crate::editor::session::UiSink {
        view_id: "tempo-map-diagnostic".into(),
        reply,
        events,
        closed: Arc::new(AtomicBool::new(false)),
    };
    let call = |id, command: &str, args| {
        editor
            .enqueue(crate::editor::session::UiRequest {
                id,
                command: command.into(),
                args,
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let response: Value = rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(response["ok"], true, "{response}");
        response["value"].clone()
    };
    let point = |position: f64, key: &str| {
        json!({
            "id": format!("p{position}"),
            "positionSec": position,
            "bpm": 120.0,
            "numerator": 4,
            "denominator": 4,
            "scale": {"key": key, "name": null, "notes": null},
        })
    };

    // 前端形态：`{ tempoMap: [...] }`。
    let payload = call(
        1,
        "set_timeline_tempo_map",
        json!({"tempoMap": [point(0.0, "C"), point(4.0, "Gb")]}),
    );
    // 落盘的是**音阶轴**（两条），且没有 BPM 字段。
    let stored = crate::settings_store::settings()
        .plugin_musical_context
        .scale_points;
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[1].key.as_deref(), Some("Gb"));
    assert_eq!(stored[1].position_sec, 4.0);
    // 响应本身就要反映新值：前端把响应当权威回声照单应用，陈旧回声会抹掉用户的编辑。
    assert_eq!(payload["tempo_map"].as_array().map(Vec::len), Some(2));
    // 生效：内核的时间线真的有两段音阶（而不只是设置里存了两条）。
    assert_eq!(editor.timeline.lock_or_recover().scale_segments().len(), 2);

    // 宿主权威字段以宿主为准：载荷带一个不同的 BPM/拍号时，音阶仍然保存、BPM 不保存。
    let _ = call(
        2,
        "set_timeline_tempo_map",
        json!({"tempoMap": [{
            "id": "p0", "positionSec": 0.0,
            "bpm": 999.0, "numerator": 7, "denominator": 8,
            "scale": {"key": "D", "name": null, "notes": null},
        }]}),
    );
    let stored = crate::settings_store::settings()
        .plugin_musical_context
        .scale_points;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].key.as_deref(), Some("D"));

    // 清空 = 取消 Tempo Map，回到"无时变音阶"。
    let payload = call(3, "set_timeline_tempo_map", json!({"tempoMap": null}));
    assert!(crate::settings_store::settings()
        .plugin_musical_context
        .scale_points
        .is_empty());
    assert!(payload["tempo_map"].is_null());

    document.close();
}
///
/// 【为什么值得单测】这是插件里少数几条会**写用户磁盘**的命令之一，路径校验就是它
/// 唯一的边界 —— 两个调用点（时间线菜单、钢琴卷帘）都不会再校验一次。收口漏了，
/// 命令就会替前端决定往哪写。
#[test]
fn export_pitch_to_midi_validates_the_output_path() {
    let (_model, owner, _id) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    let request = |path: &str| {
        json!({"outputPath":path,"tracks":[],"bpm":120,"beatsPerBar":4,
            "baseScale":"C","projectScaleNotes":[]})
    };
    // 相对路径：宿主进程的工作目录不可预测，不能作为基准。
    assert!(dispatch(&editor, "export_pitch_to_midi", request("export.mid")).is_err());
    // 扩展名不是 MIDI。
    let wrong_ext = std::env::temp_dir().join("hfs-midi-export.wav");
    assert!(dispatch(
        &editor,
        "export_pitch_to_midi",
        request(&wrong_ext.to_string_lossy())
    )
    .is_err());
    // 目录不存在：宁可不写，也不要凭空建目录树。
    let missing = std::env::temp_dir()
        .join("hfs-midi-export-missing")
        .join("out.mid");
    assert!(dispatch(
        &editor,
        "export_pitch_to_midi",
        request(&missing.to_string_lossy())
    )
    .is_err());

    // 合法路径：命令真的走到了内核（夹具时间线里没有音高数据，内核如实回报）。
    let dir = std::env::temp_dir().join(format!("hfs-midi-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("out.mid");
    let result = dispatch(
        &editor,
        "export_pitch_to_midi",
        request(&out.to_string_lossy()),
    )
    .unwrap();
    assert_eq!(result["ok"], false);
    assert_eq!(result["error"], "no_pitch_data");

    std::fs::remove_dir_all(&dir).ok();
    editor.close();
}

/// 记事本「把暂存载荷写回剪贴板」：只接受插件自己的原生载荷。
///
/// 【为什么必须在写剪贴板**之前**拒绝】写回去的字节随后会被粘贴链路解码。放行
/// 任意字节等于让一个"看起来像暂存块"的东西变成一次必然失败的粘贴 —— 而剪贴板
/// 已经被覆盖，用户原来的内容也回不来了。这里只测拒绝路径：正例要真的写系统剪贴板，
/// 那是本地环境的事，不是单元测试该碰的。
#[test]
fn notebook_write_clipboard_payload_rejects_foreign_bytes() {
    let (_model, owner, _id) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    assert!(dispatch(
        &editor,
        "notebook_write_clipboard_payload",
        json!({"payloadBase64":"not base64!!"})
    )
    .is_err());
    // 合法 base64，但不是本插件的剪贴板格式（缺 format/version）。
    let foreign = base64::engine::general_purpose::STANDARD.encode(br#"{"kind":"clips"}"#);
    assert!(dispatch(
        &editor,
        "notebook_write_clipboard_payload",
        json!({"payloadBase64":foreign})
    )
    .is_err());
    // 完全不是 JSON。
    let junk = base64::engine::general_purpose::STANDARD.encode(b"\x00\x01\x02");
    assert!(dispatch(
        &editor,
        "notebook_write_clipboard_payload",
        json!({"payloadBase64":junk})
    )
    .is_err());
    editor.close();
}

/// 最小合法 SMF：格式 0、单轨、480 ticks/四分音符，C4 从 0 弹到半秒。
fn minimal_midi() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MThd");
    bytes.extend_from_slice(&6u32.to_be_bytes());
    bytes.extend_from_slice(&0u16.to_be_bytes());
    bytes.extend_from_slice(&1u16.to_be_bytes());
    bytes.extend_from_slice(&480u16.to_be_bytes());
    let mut track = Vec::new();
    track.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
    track.extend_from_slice(&[0x83, 0x60, 0x80, 0x3C, 0x40]);
    track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
    bytes.extend_from_slice(b"MTrk");
    bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
    bytes.extend_from_slice(&track);
    bytes
}

/// MIDI 导入：路径判据、共享编排真的落地到曲线，而"建成片段"仍然是明确拒绝。
///
/// 【为什么"建成片段"必须是拒绝而不是静默】插件的时间线是宿主清单的投影
/// （`workspace_timeline_locked` 只保留已分配 region 的 clip）。若这里放行，命令会
/// 成功返回、片段在下一次宿主同步时消失 —— 那是比报错更坏的结果。这条断言把
/// 前端闸门（`canImportMidiAsClip`）背后的理由钉在后端行为上。
#[test]
fn midi_import_writes_the_curve_but_never_a_local_clip() {
    let (_model, owner, _id) = crate::editor::session::tests::fixture();
    let editor = owner.editor_session().unwrap();
    let timeline = dispatch(&editor, "get_timeline_state", json!({})).unwrap();
    let track = timeline["tracks"][0]["id"].as_str().unwrap().to_string();

    // 相对路径：宿主进程的工作目录不可预测，不能作为基准。
    assert!(dispatch(&editor, "get_midi_tracks", json!({"midiPath":"note.mid"})).is_err());
    assert!(dispatch(
        &editor,
        "import_midi_to_pitch",
        json!({"midiPath":"note.mid","trackIndices":[0]})
    )
    .is_err());

    let gated = dispatch(
        &editor,
        "import_midi_as_clip",
        json!({"midiPath":"note.mid","startSec":0.0}),
    )
    .unwrap_err();
    assert!(gated.contains("unavailable"), "{gated}");

    let dir = std::env::temp_dir().join(format!("hfs-plugin-midi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("note.mid");
    std::fs::write(&path, minimal_midi()).unwrap();
    let path = path.to_string_lossy().into_owned();

    let tracks = dispatch(&editor, "get_midi_tracks", json!({"midiPath":path})).unwrap();
    assert_eq!(tracks["ok"], true, "{tracks}");
    assert_eq!(tracks["tracks"][0]["note_count"], 1);

    // 夹具轨默认既没开合成、也没有音高算法，而写曲线要求目标轨真的会去读它 ——
    // 先按 GUI 的正常路径把它打开，再导入。
    dispatch(
        &editor,
        "set_track_state",
        json!({"trackId":track,"composeEnabled":true,"pitchAnalysisAlgo":"nsf_hifigan_onnx"}),
    )
    .unwrap();
    let result = dispatch(
        &editor,
        "import_midi_to_pitch",
        json!({"midiPath":path,"trackIndices":[0]}),
    )
    .unwrap();
    assert_eq!(result["ok"], true, "{result}");
    assert!(result["frames_touched"].as_u64().unwrap() > 0);

    // 【为什么不再问 `get_param_frames`】那个读入口要把**工程帧**投影回源域，而投影
    // 需要分析基线；本夹具没有分析，于是它恒报 `analysis_pending` 并回零 —— 那测的是
    // 分析闸门，不是"MIDI 导入把曲线写进去了"。写没写进去只有权威说得准。
    // 【权威按宿主根轨 id 索引，而 `get_timeline_state` 给的是本视图命名空间下的 id】
    // 直接拿后者查权威永远查不到 —— 这条曾经让"导入确实写进去了"看起来像产品 bug。
    let root = track
        .strip_prefix(editor.namespace.as_str())
        .unwrap_or(track.as_str())
        .to_owned();
    let curve = {
        let document = editor.document().unwrap();
        let state = document.edits.lock_or_recover();
        state
            .authority
            .root(&root)
            .and_then(|authority| authority.runs.get("pitch_edit"))
            .map(|runs| {
                runs.iter()
                    .flat_map(|run| run.values.iter().copied())
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default()
    };
    assert!(
        curve.iter().any(|value| (*value - 60.).abs() < 1e-3),
        "MIDI 导入必须把 60 写进权威：{curve:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
    editor.close();
}

/// **B 组命令在插件里必须真的可用，而不是落到 `Command unavailable`。**
///
/// 【为什么这几条要单独钉】它们**不需要任何宿主 API**（只读 PCM / 只改插件自己的
/// 参数 / 只写用户选定的文件），因此 ARA-only 宿主上与独立 App 完全同义。此前它们
/// 一律落到那句兜底错误，于是前端的缓存清理、音高参考、参数剪贴板整块不可用 ——
/// 用户看到的是"插件功能不全"，而根因只是没有派发。
#[test]
fn host_api_free_commands_are_dispatched_in_the_plugin() {
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let editor = owner.editor_session().unwrap();
    let (reply, rx) = mpsc::channel();
    let (events, _) = mpsc::sync_channel(32);
    let sink = crate::editor::session::UiSink {
        view_id: "host-api-free".into(),
        reply,
        events,
        closed: Arc::new(AtomicBool::new(false)),
    };
    let call = |id, command: &str, args| {
        editor
            .enqueue(crate::editor::session::UiRequest {
                id,
                command: command.into(),
                args,
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap()
    };
    // 缓存清理：插件没有持久缓存 ⇒ 显式成功的 no-op（不是"命令不存在"）。
    for (id, command) in [(1, "clear_render_cache"), (2, "clear_waveform_cache")] {
        let response = call(id, command, json!({}));
        assert_eq!(response["ok"], true, "{command}: {response}");
        assert_eq!(response["value"]["cleared"], 0, "{command}: {response}");
    }
    // 未知片段：明确回 `clip_not_found`，而不是"命令不可用"。
    let response = call(3, "get_clip_linked_params", json!({"clipId": "nope"}));
    assert_eq!(response["ok"], true, "{response}");
    assert_eq!(response["value"]["ok"], false, "{response}");
    assert_eq!(response["value"]["error"], "clip_not_found", "{response}");
    // 会话确实接受这条命令（而不是落到那句兜底错误）：它在时间线上查片段，
    // 找不到就回 `clip_not_found` —— 两种结果都证明派发已通，区别只在数据。
    // 夹具没有 renderer 分配，时间线是空的，因此这里必然落到 `clip_not_found`。
    // 没有选中片段时快速导出必须**明确拒绝**，而不是导出一个空文件。
    let response = call(
        5,
        "quick_export_selected_clips",
        json!({"clipIds": [], "outputDir": "C:/tmp", "fileName": "x"}),
    );
    assert_eq!(response["ok"], false, "{response}");
    let error = response["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("none of the selected clips"),
        "必须说明原因：{response}"
    );
    drop(document);
    editor.close();
}

#[test]
fn startup_editor_lookup_does_not_wait_for_pcm_or_geometry_transaction() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    let expected = document.editor_session().unwrap();
    let transaction = document.transaction.lock_or_recover();
    let (tx, rx) = mpsc::channel();
    let concurrent = document.clone();
    let reader = std::thread::spawn(move || {
        let _ = tx.send(concurrent.editor_session());
    });
    let result = rx.recv_timeout(std::time::Duration::from_secs(1));
    drop(transaction);
    reader.join().unwrap();
    let actual = result
        .expect("UI editor lookup waited for the PCM transaction")
        .unwrap();
    assert!(Arc::ptr_eq(&expected, &actual));
    drop(owner);
    document.close();
}

#[test]
fn source_preparation_returns_inventory_before_files_and_preserves_history() {
    let (model, owner, _) = crate::editor::session::tests::fixture();
    let document = model.session();
    document.publish_source_pcm(
        "ara://source".into(),
        Arc::new(crate::render::source::SourcePcm {
            sample_rate: 44100,
            planes: vec![vec![0.1; 100000]],
            version: 0,
            covered: vec![],
            _reservation: None,
        }),
    );
    let editor = owner.editor_session().unwrap();
    let initial = dispatch(&editor, "get_timeline_state", json!({})).unwrap();
    assert_eq!(initial["tracks"].as_array().unwrap().len(), 1);
    let root = initial["tracks"][0]["id"].as_str().unwrap();
    dispatch(&editor,"set_param_block_edits",json!({"rootTrackId":root,"expectedEditRevision":0,"edits":[{"id":"early","geometry":{"kind":"root","span":{"startSec":0.0,"endSec":0.02}},"manual":true,"createdSeq":1}],"checkpoint":true})).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if editor.ensure_loaded(false).is_ok() {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        dispatch(&editor, "get_history_state", json!({})).unwrap()["undoDepth"],
        1
    );
    assert!(editor.timeline.lock_or_recover().clips[0]
        .source_path
        .as_ref()
        .is_some_and(|p| std::path::Path::new(p).is_file()));
    assert_eq!(
        dispatch(&editor, "get_param_block_view", json!({"rootTrackId":root})).unwrap()["blocks"]
            [0]["id"],
        "early"
    );
    document.close();
}
