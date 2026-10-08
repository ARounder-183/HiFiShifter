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

/// 导出 MIDI 的收口：目标路径必须绝对、以 `.mid` 结尾，且目录已存在。
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
