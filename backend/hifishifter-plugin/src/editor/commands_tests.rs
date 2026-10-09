//! 参数分组编辑与组件状态回归；验证 actor/state，不代替真实宿主 GUI 验收。
use super::*;
use std::sync::{atomic::AtomicBool, mpsc, Arc};

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
        editor.timeline.lock().unwrap().project_scale_notes,
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
        editor.timeline.lock().unwrap().project_scale_notes,
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
    assert_eq!(editor.timeline.lock().unwrap().scale_segments().len(), 2);

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
/// 那是本机环境的事，不是单元测试该碰的。
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

    let frames = dispatch(
        &editor,
        "get_param_frames",
        json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":4,"binary":false}),
    )
    .unwrap();
    assert_eq!(frames["ok"], true, "{frames}");
    assert_eq!(frames["edit"][0], 60.0, "{frames}");
    assert_eq!(frames["pitch_edit_user_modified"], true, "{frames}");

    std::fs::remove_dir_all(&dir).ok();
    editor.close();
}
