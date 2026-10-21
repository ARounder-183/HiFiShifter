//! `render/extension.rs` 的 `bound_tests` 测试（从内联模块纯搬到这里，`use super::*` 解析不变）。

use super::*;
/// 实际原kernel PCM必须随HFS形状变化；只委托head时不能又烘焙宿主tail。
#[test]
fn owned_fade_shape_changes_kernel_pcm_and_state_roundtrip_and_undo_clears_it() {
    use hifishifter_kernel::editor::ParamHost;
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let key = (&*ids[0] as *const u8) as u64;
    {
        let mut timeline = document.timeline.lock_or_recover();
        let clip = &mut timeline.as_mut().unwrap().clips[0];
        clip.start_sec = 1.;
        clip.length_sec = 1.;
        clip.takes[0].source_end_sec = 1.;
        clip.normalize_takes();
    }
    {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&key).unwrap();
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 1.;
        region.duration_in_modification_time = 1.;
        region.has_content_based_fade_at_head = true;
    }
    let pcm = Arc::new(super::super::source::SourcePcm {
        covered: Vec::new(),
        sample_rate: 44100,
        planes: vec![vec![0.5; 44100]],
        version: 0,
        _reservation: None,
    });
    document
        .edit_sources
        .lock_or_recover()
        .insert("ara://source".into(), pcm.clone());
    document
        .sources
        .lock_or_recover()
        .insert("ara://source".into(), pcm);
    let host = crate::host::MockHost::new();
    host.enable_writer();
    host.clear_markers();
    for (name, value) in [
        ("D_LENGTH", 1.),
        ("D_PLAYRATE", 1.),
        ("D_FADEINLEN", 0.25),
        ("D_FADEINLEN_AUTO", 0.),
        ("D_FADEOUTLEN", 0.25),
        ("D_FADEOUTLEN_AUTO", 0.),
    ] {
        host.set_value(name, value);
    }
    unsafe {
        owners[0].bind_host_adapter(host.context());
    }
    owners[0].refresh_host_transport();
    let editor = owners[0].editor_session().unwrap();
    editor.ensure_loaded(false).unwrap();
    let before = owners[0].encode_state().unwrap();
    let render = || {
        let input = {
            let _transaction = document.transaction.lock_or_recover();
            owners[0]
                .capture_render_input(&document, &document.edits.lock_or_recover(), true)
                .unwrap()
                .1
        };
        assert_eq!(
            input.timeline.as_ref().unwrap().clips[0].fade_out_sec,
            0.,
            "未委托tail仍归宿主，不能重复淡化"
        );
        input
            .render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
            .unwrap()
    };
    let original = render();
    let index = 5512;
    assert!((original[0].left[index] as f64 - 0.5 * 0.5_f64.powf(0.45)).abs() < 0.001);
    let clip = editor.timeline().lock_or_recover().clips[0].id.clone();
    let plan = editor
        .plan_host_edit(
            "set_clip_state",
            &serde_json::json!({"clipId":clip,"fadeInShape":5.,"fadeInDir":0.}),
        )
        .unwrap();
    crate::host::host_edit::execute(&owners[0], plan, || document.is_alive()).unwrap();
    let updated = render();
    assert!((updated[0].left[index] as f64 - 0.25).abs() < 0.001);
    assert!((updated[0].left[index] - original[0].left[index]).abs() > 0.1);
    let saved = owners[0].encode_state().unwrap();
    let value: serde_json::Value = serde_json::from_slice(&saved).unwrap();
    assert_eq!(value["version"], 4);
    assert_eq!(value["edits"]["fades"].as_object().unwrap().len(), 1);
    owners[0].restore_state(&before).unwrap();
    assert!(
        document.edits.lock_or_recover().fades.is_empty(),
        "恢复旧无形状状态必须撤销HFS形状，不保留未来编辑"
    );
    owners[0].restore_state(&saved).unwrap();
    assert_eq!(
        document
            .edits
            .lock_or_recover()
            .fades
            .values()
            .next()
            .unwrap()
            .in_shape,
        5.
    );
    document.close();
}
use crate::ara_entry::HostEntry;
use ara2_bridge::companion::vst3::ffi::{ara2_vst3_plugin_entry_bind, ARA2_VST3_OK};
use ara2_bridge::companion::{CompanionFactory, CompanionRoles};
use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
use ara2_bridge::sys::*;
/// v2首次真实恢复就建立源basis；后续移动/裁切/线性拉伸无需用户再落笔才迁移。
#[test]
fn legacy_v2_first_restore_promotes_source_basis_before_host_transform() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let key = (&*ids[0] as *const u8) as u64;
    {
        let _transaction = document.transaction.lock_or_recover();
        let mut timeline = document.timeline.lock_or_recover();
        let timeline = timeline.as_mut().unwrap();
        timeline.project_sec = 2.;
        let clip = &mut timeline.clips[0];
        clip.start_sec = 1.;
        clip.length_sec = 1.;
        clip.takes[0].source_end_sec = 1.;
        clip.normalize_takes();
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&key).unwrap();
        region.start_in_playback_time = 1.;
        region.duration_in_modification_time = 1.;
        region.duration_in_playback_time = 1.;
    }
    let host = owners[0].assigned_timeline(&document).unwrap();
    let mut client = host.clone();
    let mut edit = vec![0_f32; 22];
    for (frame, value) in edit.iter_mut().enumerate().take(21).skip(10) {
        *value = 60. + (frame - 10) as f32;
    }
    let params = hifishifter_kernel::state::TrackParamsState {
        frame_period_ms: 100.,
        pitch_orig: vec![57.; 22],
        pitch_edit: edit,
        pitch_edit_user_modified: true,
        ..Default::default()
    };
    client.params_by_root_track.insert("track".into(), params);
    let mut legacy = crate::state_channel::EditState::default()
        .merge(&host, &client, 0)
        .unwrap();
    legacy
        .reconcile(&std::collections::BTreeMap::from([(
            "track".into(),
            vec![("modification".into(), "ara://source".into())],
        )]))
        .unwrap();
    let bytes = legacy.encode().unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["version"],
        2
    );
    document.ready.store(false, Ordering::Release);
    owners[0].restore_state(&bytes).unwrap();
    document.prepare_renderers();
    let saved = owners[0].encode_state().unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&saved).unwrap()["version"],
        8
    );
    let atlas = document.edits.lock_or_recover().atlas.clone();
    assert_eq!(atlas.regions.len(), 1);
    let identities = {
        let _transaction = document.transaction.lock_or_recover();
        document.parameter_identities_locked(&host).unwrap()
    };
    let unchanged = atlas.project_roots(&host, &identities).unwrap();
    assert_eq!(
        &unchanged["track"].pitch_edit[10..21],
        &(60..=70).map(|note| note as f32).collect::<Vec<_>>()
    );
    let mut moved = host.clone();
    let clip = &mut moved.clips[0];
    clip.start_sec = 3.;
    clip.length_sec = 2.;
    clip.takes[0].source_start_sec = 0.25;
    clip.takes[0].source_end_sec = 0.75;
    clip.takes[0].playback_rate = 0.25;
    clip.normalize_takes();
    moved.project_sec = 5.;
    let followed = atlas.follow_geometry(&moved, &identities).unwrap();
    let roots = followed.project_roots(&moved, &identities).unwrap();
    assert_eq!(roots["track"].pitch_edit[10], 0.);
    assert_eq!(roots["track"].pitch_edit[30], 62.5);
    assert_eq!(roots["track"].pitch_edit[40], 65.);
    assert_eq!(roots["track"].pitch_edit[50], 67.5);
    let local = followed.project_local(&moved, &identities).unwrap();
    assert_eq!(local[&host.clips[0].id].pitch_edit[0], 62.5);
    let mut cold = crate::state_channel::EditState::default();
    cold.restore(&saved).unwrap();
    let rebound = cold.atlas.rebind(&moved, &identities).unwrap();
    assert_eq!(
        rebound.project_local(&moved, &identities).unwrap()[&host.clips[0].id].pitch_edit,
        local[&host.clips[0].id].pitch_edit
    );
    document.close();
}

/// v3源basis按组件真实区域保存，冷恢复不能借另一轨道的atlas或沿旧session key。
#[test]
fn source_parameter_atlas_state_is_scoped_and_rebound_without_gui() {
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let mut client = document.workspace_timeline().unwrap();
    for (root, note) in [("track", 60.), ("b", 67.)] {
        client.params_by_root_track.insert(
            root.into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.,
                pitch_edit_user_modified: true,
                pitch_orig: vec![57., 57.],
                pitch_edit: vec![note, note],
                ..Default::default()
            },
        );
    }
    document
        .accept_workspace_edits(
            0,
            document.revision.load(Ordering::Acquire),
            &client,
            &document.workspace_projection().unwrap(),
        )
        .unwrap();
    let saved = owners
        .iter()
        .map(|owner| owner.encode_state().unwrap())
        .collect::<Vec<_>>();
    let payloads = saved
        .iter()
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).unwrap())
        .collect::<Vec<_>>();
    document.close();
    for payload in &payloads {
        assert_eq!(payload["version"], 8);
        // 权威参数是**唯一**被序列化的参数存储（`regions` 是每帧派生的投影缓存，
        // 不进存档）—— 因此"只保存本组件范围"这条保证现在落在 `authorities` 上。
        assert_eq!(
            payload["edits"]["atlas"]["authorities"]
                .as_object()
                .unwrap()
                .len(),
            1,
            "不能保存其它组件的参数权威"
        );
    }
    let (cold, restored, _cold_ids) = crate::editor::session::tests::workspace_fixture();
    let cold_doc = cold.session();
    for index in 0..2 {
        restored[index].restore_state(&saved[index]).unwrap();
    }
    cold_doc.prepare_renderers();
    let atlas = cold_doc.edits.lock_or_recover().atlas.clone();
    cold_doc.close();
    assert_eq!(atlas.regions.len(), 2);
    assert!(
        atlas
            .regions
            .values()
            .all(|record| record.identity.key != 0),
        "JSON旧key不得冒充新会话身份"
    );
}

/// 有宿主fade缓存后，快照/活actor装饰自动变化；不推进音频generation，不再烘焙普通fade。
#[test]
fn host_fades_project_into_gui_and_refresh_without_reloading_or_baking_audio() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let first = (&*ids[0] as *const u8) as u64;
    {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&first).unwrap();
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 4.;
    }
    let host = crate::host::MockHost::new();
    unsafe {
        owners[0].bind_host_adapter(host.context());
    }
    owners[0].refresh_host_transport();
    let (response, _, _) = document.workspace_snapshot().unwrap();
    assert_eq!(response.timeline.clips[0].fade_in_sec, 0.2);
    let actor = owners[0].editor_session().unwrap();
    let (reply, rx) = std::sync::mpsc::channel();
    let (events, _) = std::sync::mpsc::sync_channel(128);
    let sink = crate::editor::session::UiSink {
        view_id: "fade-projection".into(),
        reply,
        events,
        closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let call = |command: &str| {
        actor
            .enqueue(crate::editor::session::UiRequest {
                id: 1,
                command: command.into(),
                args: serde_json::json!({}),
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let response: serde_json::Value =
            rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(response["ok"], true, "{response}");
        response["value"].clone()
    };
    let initial = call("get_timeline_state");
    assert_eq!(
        initial["clips"][0]["host_fades"]["curve_mode"],
        "continuous"
    );
    assert_eq!(initial["clips"][0]["host_fades"]["in_curvature"], -0.2);
    assert_eq!(initial["clips"][0]["host_fades"]["in_s"], -0.3);
    let generation = call("plugin_get_apply_state")["generation"].clone();
    host.set_value("D_FADEINLEN", 0.75);
    owners[0].refresh_host_transport();
    host.set_value("D_FADEINDIR2_NEW", 0.65);
    owners[0].refresh_host_transport();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        let timeline = call("get_timeline_state");
        let updated = (timeline["clips"][0]["fade_in_sec"].as_f64().unwrap() - 0.75).abs() < 1e-8;
        if updated {
            assert_eq!(timeline["clips"][0]["host_fades"]["in_s"], 0.65);
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let (keys, input) = {
        let _transaction = document.transaction.lock_or_recover();
        owners[0]
            .capture_render_input(&document, &document.edits.lock_or_recover(), true)
            .unwrap()
    };
    let canonical = input.timeline.unwrap().clips[0].fade_in_sec;
    let after = call("plugin_get_apply_state")["generation"].clone();
    document.close();
    assert_eq!(after, generation);
    assert_eq!(keys, vec![first]);
    assert_eq!(canonical, 0.);
}
/// 只打开editor-only入口时，也必须采集同文档隐藏playback owner的元数据。
#[test]
fn task38b_one_editor_refresh_collects_hidden_playback_metadata() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let first = (&*ids[0] as *const u8) as u64;
    {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&first).unwrap();
        region.start_in_playback_time = 1.0;
        region.duration_in_playback_time = 4.0;
    }
    let editor = Arc::new(ExtensionOwner::default());
    let raw = editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    // SAFETY: fixture保留真实region、document及extension；editor入口没有自己的take。
    unsafe {
        let ext = &*raw;
        for key in [&*ids[0] as *const u8, &*ids[1] as *const u8] {
            ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
                ext.editorRendererRef,
                key.cast_mut().cast(),
            );
        }
    }
    let host = crate::host::MockHost::new();
    unsafe {
        owners[0].bind_host_adapter(host.context());
    }
    host.reset();
    editor.refresh_host_transport();
    let metadata = owners[0].host_geometry_metadata();
    let entry_metadata = editor.host_geometry_metadata();
    document.close();
    let metadata = metadata.expect("单一editor入口必须让隐藏playback的只读数据就绪");
    assert_eq!(metadata[0].region_key, first);
    assert_eq!(metadata[0].geometry.fade_in_sec, 0.2);
    assert!(entry_metadata.is_err(), "多区域editor不能冒充唯一take绑定");
}

/// 重复 `bindDocumentToExtension` 的幂等复用需要"已绑定且未关闭"的只读实例指针；
/// 关闭后必须为 `None` —— 绝不能把一个已终止的实例交回宿主。
#[test]
fn existing_binding_raw_disappears_once_the_owner_is_closed() {
    let (model, _owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let editor = Arc::new(ExtensionOwner::default());
    assert!(
        editor.existing_binding_raw().is_none(),
        "未绑定时不得有实例指针"
    );
    editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    assert!(
        editor.existing_binding_raw().is_some(),
        "绑定后应可复用实例指针"
    );
    document.close();
    assert!(
        editor.existing_binding_raw().is_none(),
        "关闭后不得把死实例交回宿主"
    );
}

/// **文档关闭后同一实例必须能重绑到新文档。**
///
/// 【为什么这是问题二的回归】REAPER 重开工程时对同一个 FX 槽**复用同一个实例**。
/// `document_closed()` 已把上一份文档的记号线撤掉，若 `bind_to_document` 一律拒绝
/// "already bound"，`ara_entry::bind` 只能把 null 交给宿主 —— 插件加载了却永远不绑，
/// 界面停在"等待宿主音频"，而删掉插件再添加就好（那会建一个全新实例）。
#[test]
fn an_instance_can_rebind_to_a_new_document_after_its_first_closed() {
    let (first_model, _owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let first_document = first_model.session();
    let owner = Arc::new(ExtensionOwner::default());
    owner
        .bind_to_document(
            first_document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    assert_eq!(owner.rebind_generation_count(), 0, "首次绑定不算重绑");

    first_document.close();
    assert!(owner.is_closed(), "关闭后实例应当处于关闭态");

    let (second_model, _owners2, _ids2) = crate::editor::session::tests::workspace_fixture();
    let second_document = second_model.session();
    owner
        .bind_to_document(
            second_document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .expect("文档关闭后同一实例必须能重绑到新文档");

    assert_eq!(owner.rebind_generation_count(), 1, "必须记录一次重绑");
    assert!(
        !owner.is_closed(),
        "重绑成功必须解除关闭态，否则编辑器永久不可用"
    );
    assert!(
        owner.existing_binding_raw().is_some(),
        "重绑后宿主应当拿到有效的扩展实例"
    );
    assert!(owner.is_ara_bound(), "重绑后实例应当真的挂在新文档上");
}

/// 隐藏getter重入关闭唯一GUI入口后，不能借另一活owner继续该入口的采集批次。
#[test]
fn task38b_hidden_getter_revokes_closed_editor_batch_before_next_renderer() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let editor = Arc::new(ExtensionOwner::default());
    let raw = editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    unsafe {
        let ext = &*raw;
        ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
            ext.editorRendererRef,
            (&*ids[0] as *const u8).cast_mut().cast(),
        );
    }
    let host = crate::host::MockHost::new();
    unsafe {
        owners[0].bind_host_adapter(host.context());
    }
    // assignment 回调会提前缓存“无宿主接口”的 Err；只观测这次被撤销的读取批次。
    for owner in &owners {
        owner.host_geometry.lock_or_recover().take();
    }
    host.reset();
    let weak = Arc::downgrade(&editor);
    *host.hook.borrow_mut() = Some((
        "D_LENGTH".into(),
        Box::new(move || weak.upgrade().unwrap().stop_editor()),
    ));
    editor.refresh_host_transport();
    let calls = host.calls();
    let sampled_next = owners[1].host_geometry.lock_or_recover().is_some();
    let sampled_first = owners[0].host_geometry.lock_or_recover().is_some();
    document.close();
    assert_eq!(calls.last().map(String::as_str), Some("D_LENGTH"));
    assert!(!sampled_first, "撤销后的第一份数据不能发布");
    assert!(!sampled_next, "入口关闭后不能继续读取另一活renderer");
}

/// 稳定工程只检查版本，不在每个UI tick重复读取整组markers；fade改变仍刷新。
#[test]
fn task38b_stable_geometry_skips_raw_fields_but_fade_change_and_model_refresh_do_not() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = &owners[0];
    let first = (&*ids[0] as *const u8) as u64;
    {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&first).unwrap();
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 4.;
    }
    let host = crate::host::MockHost::new();
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    owner.refresh_host_transport();
    let first_metadata = owner.host_geometry_metadata().unwrap();
    host.reset();
    owner.refresh_host_transport();
    let same = owner.host_geometry_metadata().unwrap();
    let stable_calls = host.calls();
    host.set_value("D_FADEINLEN", 0.75);
    host.reset();
    owner.refresh_host_transport();
    let changed = owner.host_geometry_metadata().unwrap();
    let changed_calls = host.calls();
    host.reset();
    document.revision.fetch_add(1, Ordering::AcqRel);
    owner.refresh_host_transport();
    let model_calls = host.calls();
    document.close();
    assert_eq!(same, first_metadata);
    assert!(
        !stable_calls
            .iter()
            .any(|name| name == "count" || name == "D_FADEINLEN"),
        "稳定UI tick不能重读全量几何: {stable_calls:?}"
    );
    assert_eq!(changed[0].geometry.fade_in_sec, 0.75);
    assert!(changed_calls.iter().any(|name| name == "D_FADEINLEN"));
    assert!(
        model_calls.iter().any(|name| name == "D_FADEINLEN"),
        "相同project counter但新ARA model仍须重新核对绑定"
    );
}

/// 原生getter可同步调用真实doc.close/owner.stop/assignment回调，不能持任何内部锁。
#[test]
fn task38a_owner_gate_rechecks_document_close_owner_scope_and_model_after_transport_state() {
    for change in ["document", "owner", "scope", "model"] {
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = owners[0].clone();
        let host = crate::host::MockHost::new();
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        host.reset();
        let weak = Arc::downgrade(&owner);
        let doc = document.clone();
        *host.hook.borrow_mut() = Some((
            "state".into(),
            Box::new(move || match change {
                "document" => doc.close(),
                "owner" => weak.upgrade().unwrap().stop_editor(),
                "scope" => {
                    doc.scope_revision.fetch_add(1, Ordering::AcqRel);
                }
                _ => {
                    doc.revision.fetch_add(1, Ordering::AcqRel);
                }
            }),
        ));
        owner.refresh_host_transport();
        let calls = host.calls();
        let pose = document.clock.diagnostics()["host_position_authority"].clone();
        document.close();
        assert_eq!(
            calls,
            ["validate:ReaProject*", "state"],
            "{change}: revoked batch must not continue any getter"
        );
        assert_eq!(pose, false, "{change}: revoked result cannot publish");
    }
}

/// 同一owner只有唯一真实region才有typed几何，位置相等不会建立额外身份关系。
#[test]
fn task38a_owner_geometry_requires_unique_actual_assignment_and_compatible_playback_window() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = &owners[0];
    let first = (&*ids[0] as *const u8) as u64;
    let second = (&*ids[1] as *const u8) as u64;
    let host = crate::host::MockHost::new();
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    host.reset();
    assert!(owner.host_geometry().unwrap_err().contains("incompatible"));
    {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&first).unwrap();
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 4.;
    }
    assert_eq!(owner.host_geometry().unwrap().region_key, first);
    let raw = owner.binding.lock_or_recover().as_ref().unwrap().as_raw();
    // playback仍只有first，但editor另一region也属于此owner，不能忽略它伪造唯一性。
    unsafe {
        let ext = &*raw;
        ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
            ext.editorRendererRef,
            second as *mut _,
        );
    }
    host.reset();
    assert!(owner.host_geometry().unwrap_err().contains("exactly one"));
    assert!(host.calls().is_empty());
    unsafe {
        let ext = &*raw;
        ((*ext.editorRendererInterface).removePlaybackRegion.unwrap())(
            ext.editorRendererRef,
            second as *mut _,
        );
    }
    unsafe {
        let ext = &*raw;
        ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
            ext.playbackRendererRef,
            second as *mut _,
        );
    }
    host.reset();
    assert!(owner.host_geometry().unwrap_err().contains("exactly one"));
    assert!(host.calls().is_empty());
    unsafe {
        let ext = &*raw;
        for key in [first, second] {
            ((*ext.playbackRendererInterface)
                .removePlaybackRegion
                .unwrap())(ext.playbackRendererRef, key as *mut _);
        }
    }
    host.reset();
    assert!(owner.host_geometry().unwrap_err().contains("exactly one"));
    assert!(host.calls().is_empty());
    document.close();
}

#[test]
fn task38a_owner_geometry_getter_reentry_discards_closed_or_changed_scope() {
    for change in ["document", "owner", "scope", "model"] {
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = owners[0].clone();
        let host = crate::host::MockHost::new();
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        host.reset();
        let weak = Arc::downgrade(&owner);
        let doc = document.clone();
        *host.hook.borrow_mut() = Some((
            "D_LENGTH".into(),
            Box::new(move || match change {
                "document" => doc.close(),
                "owner" => weak.upgrade().unwrap().stop_editor(),
                "scope" => {
                    doc.scope_revision.fetch_add(1, Ordering::AcqRel);
                }
                _ => {
                    doc.revision.fetch_add(1, Ordering::AcqRel);
                }
            }),
        ));
        let result = owner.host_geometry();
        let calls = host.calls();
        document.close();
        assert!(result.unwrap_err().contains("authorization revoked"));
        assert_eq!(calls.last().unwrap(), "D_LENGTH", "{change}");
    }
}

/// 可消费副本只含Rust数据；普通fade变化可在UI重新采集，代次变化/关闭不返回旧值。
#[test]
fn task38a_owner_cached_geometry_is_read_only_and_revocable_without_host_calls() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = &owners[0];
    let first = (&*ids[0] as *const u8) as u64;
    {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&first).unwrap();
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 4.;
    }
    let host = crate::host::MockHost::new();
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    assert!(owner.host_geometry_metadata().is_err());
    owner.refresh_host_transport();
    host.reset();
    let cached = owner.host_geometry_metadata().unwrap();
    assert_eq!(cached[0].region_key, first);
    assert_eq!(cached[0].geometry.fade_in_sec, 0.2);
    assert!(host.calls().is_empty());
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert_eq!(owner.host_geometry_metadata().unwrap(), cached);
                assert!(owner.host_geometry().is_err());
            })
            .join()
            .unwrap();
    });
    assert!(host.calls().is_empty());
    let revision = document.revision.load(Ordering::Acquire);
    host.set_value("D_FADEINLEN", 0.75);
    owner.refresh_host_transport();
    assert_eq!(document.revision.load(Ordering::Acquire), revision);
    assert_eq!(
        owner.host_geometry_metadata().unwrap()[0]
            .geometry
            .fade_in_sec,
        0.75
    );
    document.scope_revision.fetch_add(1, Ordering::AcqRel);
    assert!(owner.host_geometry_metadata().is_err());
    document.close();
    assert!(owner.host_geometry_metadata().is_err());
    assert!(owner.host_geometry.lock_or_recover().is_none());
}

/// 测试专用同步驱动：冻结阶段持锁，真正内核计算阶段不借文档事务。
fn render_test_edits(
    owner: &ExtensionOwner,
    document: &super::super::document::DocumentSession,
    edits: &crate::state_channel::EditState,
) -> Result<Vec<super::super::snapshot::PlaybackSnapshot>, String> {
    let input = {
        let _transaction = document.transaction.lock_or_recover();
        owner.capture_render_input(document, edits, true)?.1
    };
    input.render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
}

// ─── 交接（host-link）测试脚手架 ─────────────────────────────────────────────
//
// 管道时代的 `Request::Snapshot` 返回的是**本实例被分配**的那一窄份视图，而
// `ExtensionOwner::export_link` 导出的是**整个文档**（见那里的说明：导出的东西是
// 「这个工程」，不是「这个插件窗口」）。因此这里的脚手架把两者分开：
//   * [`LinkExport`] 是一次真实导出（临时目录 + 清单 + 清单里的文档时间线），
//     供断言代次/来源/几何，并据此构造回写载荷；
//   * [`assigned_view`] 保留旧快照的**实例视角**（`assigned_timeline` 叠加已存编辑），
//     用于那些本来就在验证「本实例只能看见/提交自己分配到的轨道」的断言。

/// 指向一个导出好的工作区的临时根目录；`Drop` 清掉整棵树。
struct LinkExport {
    root: std::path::PathBuf,
    manifest: hifishifter_host_link::LinkManifest,
    /// 清单里的**整份文档**时间线（导出返回的那一份）。
    timeline: hifishifter_kernel::state::TimelineState,
}

/// 一个不重复的临时根目录（与 `hifishifter-host-link/src/workspace.rs` 的测试同形）。
fn link_temp_root(tag: &str) -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "hfs-link-{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::AcqRel),
    ))
}

impl LinkExport {
    fn take(owner: &ExtensionOwner, tag: &str) -> Self {
        let root = link_temp_root(tag);
        let (manifest, _workspace) = owner.export_link(&root).expect("export_link");
        let timeline = serde_json::from_value(manifest.timeline.clone())
            .expect("manifest timeline is a TimelineState");
        Self {
            root,
            manifest,
            timeline,
        }
    }

    /// 回写载荷：`params`/`tracks` 是用户想合并的值，其余（身份、代次、宿主几何）
    /// 取本次导出时的读数 —— 这正是消费者会送去的那一份。
    fn edit(
        &self,
        params: serde_json::Value,
        tracks: serde_json::Value,
    ) -> hifishifter_host_link::EditManifest {
        hifishifter_host_link::EditManifest {
            format: hifishifter_host_link::EDIT_FORMAT.into(),
            format_version: hifishifter_host_link::FORMAT_VERSION,
            link_id: self.manifest.link_id.clone(),
            base_revisions: self.manifest.revisions,
            // 基线**原样回送**清单里那一份（见 `LinkManifest::host_geometry`）：
            // 自己重算会把 JSON 往返里的最后一位浮点差变成一次假漂移。
            host_geometry: self.manifest.host_geometry.clone(),
            params,
            tracks,
        }
    }

    /// 把一份时间线的用户可写字段（曲线 + 逐轨 5 个字段）打包成回写载荷。
    fn edit_from_timeline(
        &self,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> hifishifter_host_link::EditManifest {
        self.edit(
            serde_json::to_value(&timeline.params_by_root_track).unwrap(),
            serde_json::to_value(&timeline.tracks).unwrap(),
        )
    }
}

impl Drop for LinkExport {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// 旧 `Request::Snapshot` 的等价物：本实例被分配的视图，叠加已持久化的编辑。
fn assigned_view(
    owner: &ExtensionOwner,
    document: &super::super::document::DocumentSession,
) -> hifishifter_kernel::state::TimelineState {
    let mut timeline = owner.assigned_timeline(document).unwrap();
    document.edits.lock_or_recover().apply(&mut timeline);
    timeline
}

/// 最小可导出的文档：一条轨、一个片段（引用 `ara://pcm`）、一个已分配 region，
/// 以及（可选）一份授权 PCM。返回值里的 `Box` 让 region 的不透明身份活到调用方结束。
fn linkable_document(
    pcm: Option<Vec<Vec<f32>>>,
) -> (
    crate::ara::model::ModelHandle,
    Arc<super::super::document::DocumentSession>,
    Arc<ExtensionOwner>,
    Box<u8>,
) {
    use crate::render::source::SourcePcm;
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let mut timeline: hifishifter_kernel::state::TimelineState =
        serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"one","track_id":"track","name":"one","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","name":"one","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":2
        }))
        .unwrap();
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    *document.timeline.lock_or_recover() = Some(timeline);
    document.track_bindings.lock_or_recover().insert(
        "track".into(),
        vec![("host-mod".into(), "ara://pcm".into())],
    );
    if let Some(planes) = pcm {
        document.edit_sources.lock_or_recover().insert(
            "ara://pcm".into(),
            Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes,
                version: 0,
                _reservation: None,
            }),
        );
    }
    let identity = Box::new(0_u8);
    let key = (&*identity as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(key, document.id, 0)
        .unwrap();
    document
        .clip_ids
        .lock_or_recover()
        .insert(key, "one".into());
    document.regions.lock_or_recover().insert(
        key,
        crate::ara::AraPlaybackRegion {
            audio_source_persistent_id: "ara://pcm".into(),
            audio_modification_persistent_id: "host-mod".into(),
            duration_in_modification_time: 4.0 / 44100.0,
            duration_in_playback_time: 4.0 / 44100.0,
            ..Default::default()
        },
    );
    document.ready.store(true, Ordering::Release);
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    // SAFETY: owner 与 document 都保留到调用方结束。
    unsafe {
        let ext = &*raw;
        ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
            ext.playbackRendererRef,
            key as *mut _,
        );
    }
    (model, document, owner, identity)
}

/// 一次真实导出必须落成一份**磁盘上的工作区**：清单与音频都能被独立读者读回。
#[test]
fn an_export_lands_as_a_readable_workspace_directory() {
    let (_model, _document, owner, _identity) =
        linkable_document(Some(vec![vec![0.1, 0.2, 0.3, 0.4]]));
    let root = link_temp_root("workspace-roundtrip");
    let (manifest, workspace) = owner.export_link(&root).unwrap();
    assert!(workspace.root().starts_with(&root), "目录名由身份派生");
    assert!(workspace.looks_like_workspace());

    let reread = workspace.read_manifest().unwrap();
    assert_eq!(reread.link_id, manifest.link_id);
    assert_eq!(reread.clip_count(), 1);
    assert_eq!(manifest.audio_summary(), (1, 0, 0));

    // complete 素材的音频确实写在了工作区里，且能被摘要校验读回。
    let entry = manifest
        .sources
        .iter()
        .find(|entry| entry.key == "ara://pcm")
        .unwrap();
    assert!(workspace.resolve(entry.file.as_ref().unwrap()).is_file());
    assert_eq!(
        workspace.read_audio(entry).unwrap().planes,
        vec![vec![0.1, 0.2, 0.3, 0.4]]
    );

    // 导出同时写下一份「还没发生过编辑」的初值状态。
    let status = workspace.read_status().unwrap();
    assert!(status.is_current(&manifest.link_id));
    assert!(status.ok);
    // 临时目录不留半个文件。
    let leftovers: Vec<_> = std::fs::read_dir(workspace.root().join(".tmp"))
        .unwrap()
        .flatten()
        .collect();
    assert!(leftovers.is_empty(), "临时文件必须被改名走");
    let _ = std::fs::remove_dir_all(&root);
}

/// 没有授权 PCM 的素材以 `partial` 登记：几何保留、音频槽留空，**整份导出不失败**。
#[test]
fn a_source_without_audio_exports_as_partial() {
    let (_model, _document, owner, _identity) = linkable_document(None);
    let export = LinkExport::take(&owner, "partial-audio");
    assert_eq!(export.manifest.clip_count(), 1, "几何必须保留");
    assert_eq!(export.manifest.audio_summary(), (0, 1, 0));
    let entry = &export.manifest.sources[0];
    assert_eq!(entry.state, hifishifter_host_link::AudioState::Partial);
    assert!(entry.file.is_none());
    assert!(entry.note.is_some(), "必须说明为什么没有音频");
    export.manifest.validate().unwrap();
}

/// 一份不属于本清单的编辑必须被身份判据拒收（`LINK_MISMATCH`）。
#[test]
fn an_edit_for_another_link_is_rejected_by_identity() {
    let (_model, _document, owner, _identity) =
        linkable_document(Some(vec![vec![0.1, 0.2, 0.3, 0.4]]));
    let export = LinkExport::take(&owner, "identity-mismatch");
    let mut edit = export.edit_from_timeline(&export.timeline);
    edit.link_id = "ara:SomeOtherHost:deadbeef".into();
    let status = owner.apply_link_edit(&export.manifest.link_id, &edit);
    assert!(!status.ok, "{:?}", status.error);
    assert_eq!(
        status.code,
        Some(hifishifter_host_link::error_code::LINK_MISMATCH)
    );
    assert!(status.error.unwrap().contains("this edit belongs to"));
}

/// **宿主几何在导出之后变了**（宿主侧的本地编辑），回写必须被拒绝并**指名字段**。
///
/// 【为什么判据在生产侧】消费者只有打开时的一份几何拷贝，而它自己的活状态
/// （波形回填、假立体声折叠扫描）一直在写本地时间线。把判据放在生产者侧之后，
/// 只有**宿主真的改了**才会被拒 —— 这正是旧实现里「用户编辑几何就被拒」的反面。
#[test]
fn a_host_geometry_change_since_export_is_rejected_and_named() {
    let (_model, document, owner, _identity) =
        linkable_document(Some(vec![vec![0.1, 0.2, 0.3, 0.4]]));
    let export = LinkExport::take(&owner, "geometry-drift");
    let edit = export.edit_from_timeline(&export.timeline);
    // 宿主侧（几何的真正权威）挪动了片段起点。
    document.timeline.lock_or_recover().as_mut().unwrap().clips[0].start_sec = 0.25;
    let status = owner.apply_link_edit(&export.manifest.link_id, &edit);
    assert!(!status.ok, "{:?}", status.error);
    assert_eq!(
        status.code,
        Some(hifishifter_host_link::error_code::GEOMETRY_CHANGED)
    );
    let error = status.error.unwrap();
    assert!(error.contains("ara_host_fields:"), "{error}");
    assert!(error.contains("clips[0].start_sec"), "{error}");
}

/// export → 写 `edits.hfsedit` → 读回 → 应用：走一遍真实的工作区文件往返。
#[test]
fn an_edit_round_trips_through_the_workspace_file() {
    let (_model, _document, owner, _identity) =
        linkable_document(Some(vec![vec![0.1, 0.2, 0.3, 0.4]]));
    let root = link_temp_root("edit-roundtrip");
    let (manifest, workspace) = owner.export_link(&root).unwrap();
    let mut client: hifishifter_kernel::state::TimelineState =
        serde_json::from_value(manifest.timeline.clone()).unwrap();
    client.tracks[0].volume = 0.5;
    let edit = hifishifter_host_link::EditManifest {
        format: hifishifter_host_link::EDIT_FORMAT.into(),
        format_version: hifishifter_host_link::FORMAT_VERSION,
        link_id: manifest.link_id.clone(),
        base_revisions: manifest.revisions,
        host_geometry: manifest.host_geometry.clone(),
        params: serde_json::to_value(&client.params_by_root_track).unwrap(),
        tracks: serde_json::to_value(&client.tracks).unwrap(),
    };
    workspace.write_edit(&edit).unwrap();
    let reread = workspace.read_edit().unwrap();
    assert_eq!(reread.link_id, edit.link_id);
    assert_eq!(reread.tracks, edit.tracks);

    let status = owner.apply_link_edit(&reread.link_id, &reread);
    assert!(status.ok, "{:?}", status.error);
    assert_eq!(status.revisions.edits, 1);
    // 生产者把回答落盘：成功与被拒**同一份可见**。
    workspace.write_status(&status).unwrap();
    let answer = workspace.read_status().unwrap();
    assert!(answer.ok);
    assert_eq!(answer.revisions.edits, 1);
    let _ = std::fs::remove_dir_all(&root);
}

/// 只读提取已提交REAPER旧归档的原始组件JSON，不调用当前encoder生成兼容证据。
fn task34_archived_v2_states() -> Vec<Vec<u8>> {
    use base64::Engine as _;
    let mut chunks = Vec::new();
    let mut states = Vec::new();
    let mut inside = false;
    for line in include_str!("../../../tests/fixtures/gui-keyboard-edited.RPP")
        .lines()
        .map(str::trim)
    {
        if line.starts_with("<VST ") {
            inside = true;
            chunks.clear();
            continue;
        }
        if !inside {
            continue;
        }
        if line == ">" {
            let start = chunks
                .windows(b"{\"edits\":".len())
                .position(|bytes| bytes == b"{\"edits\":")
                .unwrap();
            let length = u32::from_le_bytes(chunks[start - 4..start].try_into().unwrap()) as usize;
            states.push(chunks[start..start + length].to_vec());
            inside = false;
        } else {
            chunks.extend(
                base64::engine::general_purpose::STANDARD
                    .decode(line)
                    .unwrap(),
            );
        }
    }
    assert_eq!(
        states.iter().map(Vec::len).collect::<Vec<_>>(),
        [481, 22516]
    );
    states
}

/// 两条归档记录共享完全相同的旧source/modification身份，必须由真实assignment限定恢复目标。
#[test]
fn task34_archived_v2_bytes_rebind_only_inside_the_component_scope() {
    let states = task34_archived_v2_states();
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let archived: serde_json::Value = serde_json::from_slice(&states[0]).unwrap();
    let identity: Vec<(String, String)> =
        serde_json::from_value(archived["edits"]["bindings"]["ara-track-0"].clone()).unwrap();
    *document.track_bindings.lock_or_recover() = std::collections::BTreeMap::from([
        ("track".into(), identity.clone()),
        ("b".into(), identity.clone()),
    ]);
    // 与宿主冷重建一致：完整图ready前暂存全部组件原始bytes，随后统一合入权威。
    document.ready.store(false, Ordering::Release);
    for (owner, bytes) in owners.iter().zip(&states) {
        owner.restore_state(bytes).unwrap();
    }
    document.prepare_renderers();
    let accepted = document.edits.lock_or_recover().clone();
    assert!(
        !accepted.params.contains_key("track"),
        "旧A无曲线，不能借旧轨序号取到B曲线"
    );
    assert_eq!(accepted.params["b"].pitch_edit.len(), 800);
    assert_eq!(&accepted.params["b"].pitch_edit[200..212], &[64.; 12]);
    assert!(accepted
        .tracks
        .iter()
        .all(|track| track.id == "track" || track.id == "b"));
    for (index, id) in ["track", "b"].into_iter().enumerate() {
        let bytes = owners[index].encode_state().unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(saved["version"], if index == 0 { 2 } else { 8 });
        assert_eq!(saved["edits"]["tracks"].as_array().unwrap().len(), 1);
        assert_eq!(saved["edits"]["tracks"][0]["id"], id);
        assert_eq!(saved["edits"]["bindings"].as_object().unwrap().len(), 1);
        assert_eq!(saved["edits"]["params"].as_object().unwrap().len(), index);
    }
    // 真正扩大同一组件assignment，使两个完全相同身份同时进入候选集，必须拒绝合入。
    let key = (&*_ids[1] as *const u8) as u64;
    let raw = owners[0]
        .binding
        .lock_or_recover()
        .as_ref()
        .unwrap()
        .as_raw();
    unsafe {
        let ext = &*raw;
        ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
            ext.playbackRendererRef,
            key as *mut _,
        );
    }
    let before = document.edits.lock_or_recover().revision;
    owners[0].restore_state(&states[1]).unwrap();
    // 【为什么不再要求失败】身份歧义只该让这一份恢复不套用，不该让宿主保存失败；
    // 权威仍保留原曲线（下面两条断言）。
    assert!(owners[0].encode_state().is_ok(), "身份歧义不能让保存失败");
    assert_eq!(document.edits.lock_or_recover().revision, before);
    assert_eq!(
        document.edits.lock_or_recover().params["b"].pitch_edit[200],
        64.
    );
    // 缺少真实身份也不能按归档旧序号恢复；既有曲线留在权威中。
    document.track_bindings.lock_or_recover().insert(
        "track".into(),
        vec![("unknown-mod".into(), "ara://source".into())],
    );
    document.track_bindings.lock_or_recover().insert(
        "b".into(),
        vec![("unknown-b".into(), "ara://source".into())],
    );
    // 【为什么不再要求失败】缺身份只该让这次恢复不套用；保存仍要成功，
    // 权威保留原曲线（下面一条断言）。
    assert!(owners[0].encode_state().is_ok(), "缺身份不能让保存失败");
    assert_eq!(
        document.edits.lock_or_recover().params["b"].pitch_edit[200],
        64.
    );
    document.close();
}

/// 真实自动应用在计算前可接受另一组件/模型修改；旧作业不能覆盖新权威或已撤销输出。
#[test]
fn automatic_apply_releases_the_transaction_and_rechecks_model_and_edit_versions() {
    for model_changed in [true, false] {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        *document.timeline.lock_or_recover() = Some(
            serde_json::from_value(serde_json::json!({
                "tracks":[],"clips":[],"bpm":120,"project_sec":0}))
            .unwrap(),
        );
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        owner.snapshots[0]
            .publish(super::super::snapshot::PlaybackSnapshot {
                sample_rate: 44100,
                origin_sample: 0,
                left: vec![0.25; 4],
                right: vec![0.5; 4],
                covered: Vec::new(),
                _reservation: None,
            })
            .unwrap();
        let projection = document.workspace_projection().unwrap();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let result = document.apply_workspace_edits(
            0,
            document.revision.load(Ordering::Acquire),
            &projection,
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            || {
                if calls.fetch_add(1, Ordering::AcqRel) == 0 {
                    assert!(
                        document.transaction.try_lock().is_ok(),
                        "冻结后必须先释放事务才能计算"
                    );
                    if model_changed {
                        document.clear_renderers();
                    } else {
                        let timeline = document.timeline.lock_or_recover().clone().unwrap();
                        document
                            .accept_workspace_edits(
                                0,
                                document.revision.load(Ordering::Acquire),
                                &timeline,
                                &projection,
                            )
                            .unwrap();
                    }
                }
                true
            },
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains(if model_changed {
            "host model changed"
        } else {
            "superseded"
        }));
        assert_eq!(
            document.edits.lock_or_recover().revision,
            if model_changed { 0 } else { 1 }
        );
        let mut left = [9.0; 4];
        let mut right = [9.0; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 两个平面都是四帧；只检查真实publisher，不能由计算出的expected镜像掩盖覆盖。
        //
        // 【为什么宿主模型变化后快照仍然可读】`clear_renderers()` 的本职是"撤销渲染
        // 许可并让代次失效"，不再是"撤掉已发布的快照"。撤掉会让 bound 实例的每一个
        // 音频块写出纯零（`layered == false`，没有兜底）—— 那正是用户报的间歇性
        // 完全无声。上一份快照继续可听，新快照算完后由 `publish` 原子替换。
        let available = unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) };
        assert!(available);
        assert_eq!(left, [0.25; 4]);
        assert_eq!(right, [0.5; 4]);
        // 真正"这个实例不再渲染"的路径仍然清空：那里由文档关闭负责。
        document.revoke_renderers();
        let available = unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) };
        assert!(!available);
        drop(model);
    }
}

/// 即使其它线程暂持模型事务，真实prepare callback也必须返回，并在后台最终发布。
#[test]
fn prepare_callback_does_not_wait_for_a_model_transaction() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    *document.timeline.lock_or_recover() = Some(
        serde_json::from_value(serde_json::json!({
            "tracks":[],"clips":[],"bpm":120,"project_sec":0}))
        .unwrap(),
    );
    document.ready.store(true, Ordering::Release);
    let owner = Arc::new(ExtensionOwner::default());
    owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let held = document.transaction.lock_or_recover();
    let (sent, received) = std::sync::mpsc::channel();
    let requesting = owner.clone();
    let caller = std::thread::spawn(move || {
        requesting.prepare();
        sent.send(()).unwrap();
    });
    let returned = received
        .recv_timeout(std::time::Duration::from_secs(3))
        .is_ok();
    drop(held);
    caller.join().unwrap();
    assert!(returned, "宿主prepare不得等待文档事务/合成");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while owner.preparation_state().0 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(owner.preparation_state().1.is_none());
    let mut left = [9.0; 4];
    let mut right = [9.0; 4];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    // SAFETY: 四帧平面存活；验证worker真的发布空分配快照，不接受“只是忽略prepare”。
    assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
    assert_eq!(left, [0.0; 4]);
    drop(model);
}

/// 冷恢复在运行任务被阻塞时也要先合入全部组件状态，不能由各worker逐轨推进revision。
#[test]
fn cold_restores_are_merged_document_wide_before_background_jobs_start() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let mut timeline:hifishifter_kernel::state::TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0},{"id":"b","name":"B","order":1}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":4.0/44100.0,"takes":[{"id":"ta","name":"A","source_path":"source","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"cb","name":"B","track_id":"b","start_sec":0,"length_sec":4.0/44100.0,"takes":[{"id":"tb","name":"B","source_path":"source","source_start_sec":0,"source_end_sec":4.0/44100.0}]}]
        })).unwrap();
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    *document.timeline.lock_or_recover() = Some(timeline);
    let pcm = Arc::new(super::super::source::SourcePcm {
        covered: Vec::new(),
        sample_rate: 44100,
        planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
        version: 0,
        _reservation: None,
    });
    document
        .edit_sources
        .lock_or_recover()
        .insert("source".into(), pcm.clone());
    document
        .sources
        .lock_or_recover()
        .insert("source".into(), pcm);
    *document.track_bindings.lock_or_recover() = std::collections::BTreeMap::from([
        ("a".into(), vec![("ma".into(), "source".into())]),
        ("b".into(), vec![("mb".into(), "source".into())]),
    ]);
    let identities = [Box::new(0_u8), Box::new(0_u8)];
    let mut owners = Vec::new();
    let mut gates = Vec::new();
    for (index, clip) in ["ca", "cb"].into_iter().enumerate() {
        let key = (&*identities[index] as *const u8) as u64;
        region_owners()
            .lock_or_recover()
            .register(key, document.id, index)
            .unwrap();
        document.clip_ids.lock_or_recover().insert(key, clip.into());
        document.regions.lock_or_recover().insert(
            key,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "source".into(),
                duration_in_modification_time: 4.0 / 44100.0,
                duration_in_playback_time: 4.0 / 44100.0,
                ..Default::default()
            },
        );
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: identities/raw extension保留到doc销毁；这是实际宿主assignment入口。
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        let (release, gate) = std::sync::mpsc::channel();
        let (started, entered) = std::sync::mpsc::channel();
        owner
            .preparation()
            .unwrap()
            .request(Box::new(move |_| {
                started.send(()).unwrap();
                gate.recv_timeout(std::time::Duration::from_secs(3))
                    .unwrap();
                Ok(())
            }))
            .unwrap();
        entered
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        gates.push(release);
        let host = owner.assigned_timeline(&document).unwrap();
        let mut client = host.clone();
        client.tracks[0].volume = if index == 0 { 0.5 } else { 0.25 };
        let mut saved = crate::state_channel::EditState::default()
            .merge(&host, &client, 0)
            .unwrap();
        let id = if index == 0 { "a" } else { "b" };
        saved
            .reconcile(&std::collections::BTreeMap::from([(
                id.into(),
                document.track_bindings.lock_or_recover()[id].clone(),
            )]))
            .unwrap();
        owner.restore_state(&saved.encode().unwrap()).unwrap();
        owners.push(owner);
    }
    document.prepare_renderers();
    let accepted = document.edits.lock_or_recover().clone();
    // 先释放测试屏障再断言，旧实现失败也不能把真实worker留在无限等待里。
    for gate in gates {
        gate.send(()).unwrap();
    }
    assert_eq!(
        accepted
            .tracks
            .iter()
            .find(|t| t.id == "a")
            .map(|t| t.volume),
        Some(0.5)
    );
    assert_eq!(
        accepted
            .tracks
            .iter()
            .find(|t| t.id == "b")
            .map(|t| t.volume),
        Some(0.25)
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    for (index, owner) in owners.iter().enumerate() {
        while owner.preparation_state().0 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(owner.preparation_state().1.is_none());
        let mut left = [9.0; 4];
        let mut right = [9.0; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 两个四帧平面；保证无GUI冷恢复确实供音，而非只合入了state。
        assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
        for (actual, expected) in left.iter().zip(if index == 0 {
            [0.05, 0.1, 0.15, 0.2]
        } else {
            [0.025, 0.05, 0.075, 0.1]
        }) {
            assert!((*actual - expected).abs() < 1e-6);
        }
    }
    drop(model);
}

/// 实际assignment observer必须与发布使用同一事务；不能校验后移除、再发布旧区域。
#[test]
fn native_assignment_observer_serializes_with_snapshot_publication_transaction() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let identity = Box::new(0_u8);
    let key = (&*identity as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(key, document.id, 0)
        .unwrap();
    let (ready, entered) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let blocking = document.clone();
    let holder = std::thread::spawn(move || {
        let _held = blocking.transaction.lock_or_recover();
        ready.send(()).unwrap();
        let _ = released.recv_timeout(std::time::Duration::from_millis(300));
    });
    entered
        .recv_timeout(std::time::Duration::from_secs(3))
        .unwrap();
    let began = std::time::Instant::now();
    // SAFETY: 在绑定时的真实model线程驱动API；其它线程只持Rust事务，不能绕过桥接线程检查。
    unsafe {
        let ext = &*raw;
        ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(
            ext.editorRendererRef,
            key as *mut _,
        );
    }
    let elapsed = began.elapsed();
    let _ = release.send(());
    holder.join().unwrap();
    assert!(
        elapsed >= std::time::Duration::from_millis(150),
        "assignment不得绕开发布的文档事务"
    );
    assert_eq!(owner.assignments.lock_or_recover()[&2], [key]);
    drop(model);
}

#[test]
fn gui_commit_changes_assigned_pcm_and_persisted_state_restores_it() {
    use crate::render::source::SourcePcm;
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let mut timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"one","track_id":"track","name":"one","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","name":"one","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"other","track_id":"track","name":"other","start_sec":1,"length_sec":4.0/44100.0,
                "takes":[{"id":"take2","name":"other","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":2
        })).unwrap();
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    *document.timeline.lock_or_recover() = Some(timeline);
    document.track_bindings.lock_or_recover().insert(
        "track".into(),
        vec![("host-mod".into(), "ara://pcm".into())],
    );
    document.edit_sources.lock_or_recover().insert(
        "ara://pcm".into(),
        Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        }),
    );
    let identity = Box::new(0_u8);
    let key = (&*identity as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(key, document.id, 0)
        .unwrap();
    document
        .clip_ids
        .lock_or_recover()
        .insert(key, "one".into());
    document.regions.lock_or_recover().insert(
        key,
        crate::ara::AraPlaybackRegion {
            audio_source_persistent_id: "ara://pcm".into(),
            audio_modification_persistent_id: "host-mod".into(),
            duration_in_modification_time: 4.0 / 44100.0,
            duration_in_playback_time: 4.0 / 44100.0,
            ..Default::default()
        },
    );
    document.ready.store(true, Ordering::Release);
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    unsafe {
        let ext = &*raw;
        ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
            ext.playbackRendererRef,
            key as *mut _,
        );
    }
    let export = LinkExport::take(&owner, "gui-commit");
    // 导出的是**整个文档**（两个片段），与管道时代的实例视图不同。
    assert_eq!(
        export.manifest.timeline["clips"].as_array().unwrap().len(),
        2
    );
    assert_eq!(export.manifest.sources.len(), 1);
    assert_eq!(
        export.manifest.sources[0].state,
        hifishifter_host_link::AudioState::Complete
    );
    // 而提交的权威仍是本实例的分配视图 —— 只有它自己那一个片段。
    let mut client = owner.assigned_timeline(&document).unwrap();
    assert_eq!(client.clips.len(), 1);
    client.tracks[0].volume = 0.5;
    let edit = export.edit_from_timeline(&client);
    let status = owner.apply_link_edit(&export.manifest.link_id, &edit);
    assert!(status.ok, "{:?}", status.error);
    let companion = Arc::new(ExtensionOwner::default());
    companion
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    assert_eq!(
        companion.edit_state().lock_or_recover().revision,
        status.revisions.edits
    );
    assert_eq!(
        companion.edit_state().lock_or_recover().tracks[0].volume,
        0.5
    );
    let state = owner.edit_state().lock_or_recover().encode().unwrap();
    let mut restored = crate::state_channel::EditState::default();
    restored.restore(&state).unwrap();
    let output = render_test_edits(&owner, &document, &restored).unwrap();
    assert_eq!(output[0].left.len(), 4);
    for (actual, expected) in output[0].left.iter().zip([0.05_f32, 0.1, 0.15, 0.2]) {
        assert!((*actual - expected).abs() < 1e-6);
    }
    assert!(restored.restore(b"bad state").is_err());
    drop(model);
}

#[test]
fn gui_request_commits_parameters_and_rejects_stale_document_revision() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let timeline: hifishifter_kernel::state::TimelineState =
        serde_json::from_value(serde_json::json!({
            "tracks":[],"clips":[],"bpm":120,"project_sec":0
        }))
        .unwrap();
    *document.timeline.lock_or_recover() = Some(timeline);
    document.ready.store(true, Ordering::Release);
    let owner = Arc::new(ExtensionOwner::default());
    owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let export = LinkExport::take(&owner, "stale-revision");
    // 空的文档导出：没有 clip、没有素材，但身份/代次齐备。
    assert_eq!(export.manifest.clip_count(), 0);
    let edit = export.edit_from_timeline(&export.timeline);
    let status = owner.apply_link_edit(&export.manifest.link_id, &edit);
    assert!(status.ok, "{:?}", status.error);
    assert_eq!(status.revisions.edits, 1);
    // 同一份载荷再交一次：它的基准**编辑**代次已经过期，必须被拒。
    let stale = owner.apply_link_edit(&export.manifest.link_id, &edit);
    assert!(!stale.ok, "{:?}", stale.error);
    // 针对过期**宿主模型**代次的载荷被明确判为冲突（`CONFLICT`），并指名字段。
    let mut model_stale = edit.clone();
    model_stale.base_revisions.model = edit.base_revisions.model + 1;
    let conflict = owner.apply_link_edit(&export.manifest.link_id, &model_stale);
    assert!(!conflict.ok, "{:?}", conflict.error);
    assert_eq!(
        conflict.code,
        Some(hifishifter_host_link::error_code::CONFLICT)
    );
    assert!(conflict.error.unwrap().contains("host model is"));
    // 渲染器被撤销后文档不再 ready ⇒ 导出失败（旧实现里快照同样失败）。
    document.clear_renderers();
    let failed_root = link_temp_root("stale-revision-fail");
    assert!(owner.export_link(&failed_root).is_err());
    let _ = std::fs::remove_dir_all(&failed_root);
    drop(model);
}

/// 两个renderer先后提交局部视图，第二次刷新后不能抹掉第一轨PCM或持久曲线。
#[test]
fn two_renderer_commits_keep_both_curves_volumes_pcm_and_saved_state() {
    two_renderer_case(false);
}
/// REAPER复制轨道共享modification/source；状态与恢复仍必须限制在宿主实例分配内。
#[test]
fn copied_tracks_share_sources_but_keep_instance_state_isolated() {
    two_renderer_case(true);
}
fn two_renderer_case(shared_identity: bool) {
    use crate::render::source::SourcePcm;
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    // 四样本只验证复制/权限/保存与逐轨gain，显式旁路；真实音高自动应用另有WORLD/模型oracle。
    let mut timeline:hifishifter_kernel::state::TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0,"pitch_analysis_algo":"none"},{"id":"b","name":"B","order":1,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":1,
            "clips":[{"id":"clip-a","track_id":"a","name":"A","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take-a","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"clip-b","track_id":"b","name":"B","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take-b","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}]
        })).unwrap();
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    *document.timeline.lock_or_recover() = Some(timeline);
    document.edit_sources.lock_or_recover().insert(
        "ara://pcm".into(),
        Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        }),
    );
    let bindings = std::collections::BTreeMap::from([
        ("a".into(), vec![("mod-a".into(), "ara://pcm".into())]),
        (
            "b".into(),
            vec![(
                if shared_identity { "mod-a" } else { "mod-b" }.into(),
                "ara://pcm".into(),
            )],
        ),
    ]);
    *document.track_bindings.lock_or_recover() = bindings.clone();
    document.ready.store(true, Ordering::Release);
    let identities = [Box::new(0_u8), Box::new(0_u8)];
    let mut owners = Vec::new();
    for (index, id) in ["clip-a", "clip-b"].into_iter().enumerate() {
        let key = (&*identities[index] as *const u8) as u64;
        region_owners()
            .lock_or_recover()
            .register(key, document.id, index)
            .unwrap();
        document.clip_ids.lock_or_recover().insert(key, id.into());
        document.regions.lock_or_recover().insert(
            key,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "ara://pcm".into(),
                audio_modification_persistent_id: if index == 0 || shared_identity {
                    "mod-a"
                } else {
                    "mod-b"
                }
                .into(),
                duration_in_modification_time: 4.0 / 44100.0,
                duration_in_playback_time: 4.0 / 44100.0,
                ..Default::default()
            },
        );
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: identities及native扩展都保留到owner/document销毁。
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        owners.push(owner);
    }
    for (index, id) in ["a", "b"].into_iter().enumerate() {
        let export = LinkExport::take(&owners[index], &format!("two-renderer-{index}"));
        assert_eq!(
            export.manifest.revisions.edits, index as u64,
            "B先刷新最新共享revision"
        );
        // 每个 renderer 只能提交它自己分配到的轨道（旧快照的实例视角）。
        let mut client = owners[index].assigned_timeline(&document).unwrap();
        assert_eq!(client.tracks.len(), 1, "分配视图只含本实例的轨道");
        client.tracks[0].volume = if index == 0 { 0.5 } else { 0.25 };
        client.params_by_root_track.clear();
        client.params_by_root_track.insert(
            id.into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.0,
                pitch_edit: vec![61.0 + index as f32, 63.0 + index as f32],
                ..Default::default()
            },
        );
        let edit = export.edit_from_timeline(&client);
        let status = owners[index].apply_link_edit(&export.manifest.link_id, &edit);
        assert!(status.ok, "{:?}", status.error);
    }
    for (index, id) in ["a", "b"].into_iter().enumerate() {
        let saved: serde_json::Value =
            serde_json::from_slice(&owners[index].encode_state().unwrap()).unwrap();
        assert_eq!(
            saved["edits"]["params"].as_object().unwrap().len(),
            1,
            "单个组件不能保存整个ARA文档的其它轨道"
        );
        assert!(saved["edits"]["params"].get(id).is_some());
    }
    let mut restored = document.edits.lock_or_recover().clone();
    restored.reconcile(&bindings).unwrap();
    for (index, id) in ["a", "b"].into_iter().enumerate() {
        assert_eq!(
            restored.params[id].pitch_edit,
            [61.0 + index as f32, 63.0 + index as f32]
        );
        let output = render_test_edits(&owners[index], &document, &restored).unwrap();
        let factor = if index == 0 { 0.5 } else { 0.25 };
        for (actual, original) in output[0].left.iter().zip([0.1_f32, 0.2, 0.3, 0.4]) {
            assert!((*actual - original * factor).abs() < 1e-6);
        }
        let current = assigned_view(&owners[index], &document);
        assert_eq!(current.clips.len(), 1);
        assert_eq!(current.tracks[0].volume, factor);
    }
    if shared_identity {
        let original = owners[0].encode_state().unwrap();
        let second = owners[1].encode_state().unwrap();
        // REAPER复制插件：同一源身份由另一实例的实际assignment限定到B，不能改写A。
        owners[1].restore_state(&original).unwrap();
        let a = assigned_view(&owners[0], &document);
        let b = assigned_view(&owners[1], &document);
        assert_eq!(a.tracks[0].volume, 0.5);
        assert_eq!(b.tracks[0].volume, 0.5);
        assert_eq!(
            document.edits.lock_or_recover().params["b"].pitch_edit,
            [61., 63.]
        );
        owners[1].restore_state(&second).unwrap();
        assert_eq!(
            document.edits.lock_or_recover().params["a"].pitch_edit,
            [61., 63.]
        );
        assert_eq!(
            document.edits.lock_or_recover().params["b"].pitch_edit,
            [62., 64.]
        );
        let editors: Vec<_> = owners
            .iter()
            .map(|owner| owner.editor_session().unwrap())
            .collect();
        let call = |editor: &Arc<crate::editor::session::EditorSession>,
                    command: &str,
                    args: serde_json::Value| {
            let (reply, received) = std::sync::mpsc::channel();
            let (events, _) = std::sync::mpsc::sync_channel(128);
            let sink = crate::editor::session::UiSink {
                view_id: "dual-actor".into(),
                reply,
                events,
                closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            };
            editor
                .enqueue(crate::editor::session::UiRequest {
                    id: 1,
                    command: command.into(),
                    args,
                    sink,
                    link: None,
                })
                .unwrap();
            let response = received
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        for editor in &editors {
            call(editor, "get_timeline_state", serde_json::json!({}));
        }
        // 同一次debounce窗口内两轨落笔；B的共享revision变化不能让A的最新作业Conflict。
        for (index, editor) in editors.iter().enumerate() {
            let timeline = call(editor, "get_timeline_state", serde_json::json!({}));
            call(
                editor,
                "set_track_state",
                serde_json::json!({
                    "trackId":timeline["tracks"][index]["id"],"volume":if index==0 {0.75} else {0.125}
                }),
            );
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        for editor in &editors {
            loop {
                let state = call(editor, "plugin_get_apply_state", serde_json::json!({}));
                assert!(state["error"].is_null(), "{state}");
                if state["pending"] == false {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "{state}");
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        editors[0].close(); // 现在两个真实入口共享actor，只在全部回归请求完成后关闭。
        assert_eq!(
            document
                .edits
                .lock_or_recover()
                .tracks
                .iter()
                .find(|t| t.id == "a")
                .unwrap()
                .volume,
            0.75
        );
        assert_eq!(
            document
                .edits
                .lock_or_recover()
                .tracks
                .iter()
                .find(|t| t.id == "b")
                .unwrap()
                .volume,
            0.125
        );
    }
}

/// 真实空文档绑定也必须收到销毁；不依赖 first region assignment 猜 document。
#[test]
fn bound_native_entry_is_tombstoned_by_actual_document_teardown() {
    let factory = Box::leak(Box::new(
        FactoryBuilder::new("org.hfs.bound", "org.hfs.bound.archive")
            .display(
                "bound",
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                "1",
            )
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap(),
    ));
    factory
        .entry()
        .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
        .unwrap();
    let mut fixture = crate::test_host::HostFixture::new(vec![]);
    let host = fixture.instance();
    let properties = ARADocumentProperties {
        structSize: std::mem::size_of::<ARADocumentProperties>(),
        name: c"bound empty".as_ptr(),
    };
    // SAFETY: test host、factory 和 properties 保持到实际 controller 终止。
    let raw = unsafe {
        (factory
            .raw_copy()
            .createDocumentControllerWithDocument
            .unwrap())(&host, &properties)
    };
    assert!(!raw.is_null());
    // SAFETY: 原生工厂返回完整 packed 实例。
    let controller = unsafe { raw.read_unaligned() };
    let document =
        crate::render::document::DocumentSession::lookup(controller.documentControllerRef as usize)
            .unwrap();
    // SAFETY: test factory backing 已保留到进程结束。
    let association = unsafe { CompanionFactory::from_raw("bound", &*factory.as_raw()) }.unwrap();
    let owner = Arc::new(ExtensionOwner::default());
    // `HostEntry` 现在自己建 companion（重开工程时要能重建），所以传原始 factory。
    let entry = HostEntry::new(association, CompanionRoles::all(), "bound", owner.clone()).unwrap();
    // 探针必须指向**真正被绑定**的那一份 (HostEntry 内部的绑定)。
    let probe = entry
        .lifetime_probe()
        .expect("HostEntry 必须持有初始绑定探针");
    let mut extension_raw = std::ptr::null();
    // SAFETY: real C++ shim 调真实产品 bind，controller 存活且已登记。
    assert_eq!(
        unsafe {
            ara2_vst3_plugin_entry_bind(
                entry.as_raw(),
                controller.documentControllerRef.cast(),
                7,
                1,
                1,
                &raw mut extension_raw,
            )
        },
        ARA2_VST3_OK
    );
    assert!(!extension_raw.is_null());
    assert!(probe.controller_alive());
    let identity = Box::new(0_u8);
    let key = (&*identity as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(key, document.id, 0)
        .unwrap();
    // SAFETY: extension 与 controller 保留；region 仅作为不透明身份。
    unsafe {
        let extension = extension_raw
            .cast::<ARAPlugInExtensionInstance>()
            .read_unaligned();
        ((*extension.playbackRendererInterface)
            .addPlaybackRegion
            .unwrap())(extension.playbackRendererRef, key as *mut _);
    }
    assert_eq!(owner.assignments.lock_or_recover().get(&1).unwrap(), &[key]);
    // SAFETY: actual controller destructor invokes ModelHandle::destroy_document。
    unsafe {
        ((*controller.documentControllerInterface)
            .destroyDocumentController
            .unwrap())(controller.documentControllerRef)
    };
    assert!(owner.assignments.lock_or_recover().is_empty());
    assert!(owner.assigned_regions().is_err());
    assert!(!probe.controller_alive());
    // SAFETY: native entry 仍保留 companion storage；销毁文档后回调必须被 tombstone 拒绝。
    unsafe {
        let extension = extension_raw
            .cast::<ARAPlugInExtensionInstance>()
            .read_unaligned();
        ((*extension.playbackRendererInterface)
            .addPlaybackRegion
            .unwrap())(extension.playbackRendererRef, key as *mut _);
    }
    assert!(owner.assignments.lock_or_recover().is_empty());
    drop(entry);
    drop(owner);
    factory.entry().uninitialize().unwrap();
}

/// sequence 预览只展开本 sequence；显式区域重复分配不能导致双倍混音。
#[test]
fn editor_sequence_selection_is_expanded_and_deduplicated() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let region_a = Box::new(0_u8);
    let region_b = Box::new(0_u8);
    let sequence = Box::new(0_u8);
    let a = (&*region_a as *const u8) as u64;
    let b = (&*region_b as *const u8) as u64;
    let seq = (&*sequence as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(a, document.id, 0)
        .unwrap();
    region_owners()
        .lock_or_recover()
        .register(b, document.id, 1)
        .unwrap();
    document
        .sequence_regions
        .lock_or_recover()
        .insert(seq, [a, b].into_iter().collect());
    // SAFETY: binding、session 和不透明身份在调用期间均存活。
    unsafe {
        let extension = &*raw;
        let api = &*extension.editorRendererInterface;
        api.addPlaybackRegion.unwrap()(extension.editorRendererRef, a as *mut _);
        api.addRegionSequence.unwrap()(extension.editorRendererRef, seq as *mut _);
    }
    let mut expected = vec![a, b];
    expected.sort_unstable();
    assert_eq!(owner.assigned_regions().unwrap(), expected);
    document
        .sequence_regions
        .lock_or_recover()
        .get_mut(&seq)
        .unwrap()
        .remove(&b);
    region_owners().lock_or_recover().remove(b);
    assert_eq!(owner.assigned_regions().unwrap(), [a]);
    // SAFETY: lease 保留接口，remove sequence 不会移除显式 region。
    unsafe {
        let extension = &*raw;
        ((*extension.editorRendererInterface)
            .removeRegionSequence
            .unwrap())(extension.editorRendererRef, seq as *mut _);
    }
    assert_eq!(owner.assigned_regions().unwrap(), [a]);
    drop(model);
    assert!(owner.assigned_regions().is_err());
}

/// **T1：一条陈旧 key 不得把整份 scope 打成空。**
///
/// 【为什么这是"重开工程后所有 Clip 都在等宿主音频"的回归】宿主在一次分割里会把
/// **已销毁父段的 key** 一并重发（`workspace_scope_locked` 的注释早已记录这个形状）。
/// 旧实现用整批 `resolve`：只要有一条 key 未登记，整个 `assigned_regions` 就返回 Err
/// ⇒ `workspace_scope_locked` 返回 Err ⇒ `ensure_loaded` 整份 Err ⇒ 界面回落到
/// "只由宿主清单拼出来的占位"（永远没有源、也不重试）。
///
/// 写路径（assignment observer）早已改成逐条 `owner_of`；读路径必须同口径。
#[test]
fn a_stale_region_key_does_not_empty_the_scope() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let live = Box::new(0_u8);
    let stale = Box::new(0_u8);
    let a = (&*live as *const u8) as u64;
    let dead = (&*stale as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(a, document.id, 0)
        .unwrap();
    // `dead` **从未登记**（等价于宿主重发了一条已销毁父段的 key）。
    // SAFETY: binding 与不透明身份在调用期间存活。
    unsafe {
        let extension = &*raw;
        let api = &*extension.playbackRendererInterface;
        api.addPlaybackRegion.unwrap()(extension.playbackRendererRef, a as *mut _);
        api.addPlaybackRegion.unwrap()(extension.playbackRendererRef, dead as *mut _);
    }
    let scope = owner
        .assigned_scope(&document, None, false)
        .expect("一条陈旧 key 不得让整份 scope 失败");
    assert_eq!(scope.keys, vec![a], "存活的那条 key 必须留下");
    assert_eq!(owner.assigned_regions().unwrap(), vec![a]);
    document.close();
    drop(model);
}

/// **T1：本实例还没被告知分配时，并集语义的调用方可以回退文档级。**
///
/// 【为什么只给并集语义的调用方】工作区要显示整份工程，它的 scope 本来就是"本文档
/// 全部 owner 的并集"；而渲染输入是逐实例的（REAPER 一个文档上挂着多条轨道的实例），
/// 派生会让同一段音频被渲染两次。
#[test]
fn a_rebound_owner_derives_the_scope_without_the_assignment_observer() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let region = Box::new(0_u8);
    let sequence = Box::new(0_u8);
    let key = (&*region as *const u8) as u64;
    let seq = (&*sequence as *const u8) as u64;
    region_owners()
        .lock_or_recover()
        .register(key, document.id, 0)
        .unwrap();
    document
        .sequence_regions
        .lock_or_recover()
        .insert(seq, [key].into_iter().collect());
    // 刻意**不**调用 addPlaybackRegion / addRegionSequence：这正是重开工程时
    // "宿主复用了 ARA 图却不再重发分配"的形状。
    let derived = owner
        .assigned_scope(&document, None, true)
        .expect("派生不得失败");
    assert_eq!(derived.keys, vec![key]);
    assert_eq!(
        derived.source,
        crate::render::extension::ScopeSource::DocumentDerived,
        "必须如实报出「这份 scope 是派生的」，界面才能给出正确的可行动提示"
    );
    // 渲染路径不允许派生：逐实例语义，派生会让同一段音频被渲染两次。
    let audio = owner
        .assigned_scope(&document, Some(1), false)
        .expect("逐实例路径不得失败");
    assert!(audio.keys.is_empty(), "渲染输入不派生");
    assert_eq!(audio.source, crate::render::extension::ScopeSource::Empty);
    document.close();
    drop(model);
    let _ = raw;
}

/// companion 先销毁时 controller lease 必须独立保留真实 raw storage。
#[test]
fn actual_document_retains_bound_storage_after_owner_is_dropped() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document,
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    drop(owner);
    let mut identity = 0_u8;
    // SAFETY: actual model 的 session 独立持有控制器 lease，owner 已 drop 也不释放 storage。
    unsafe {
        let extension = raw.read_unaligned();
        ((*extension.playbackRendererInterface)
            .addPlaybackRegion
            .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
        ((*extension.playbackRendererInterface)
            .removePlaybackRegion
            .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
    }
    drop(model);
    // 模型终止后所有 owner 都已释放，此后不得再访问 raw。
}

/// **回归：region key 缺席时按"源 + 播放几何"反查身份，而不是判"无身份"。**
///
/// 【为什么这条测试是必须的】派生表（`regions` / `clip_ids`）与映射出的 clip id 之间
/// 存在一个"还没跟上"的窗口（宿主在收口前重投模型、或 region 刚被重建）。旧实现
/// 在这个窗口里直接判"无身份"，于是那一条 clip 本轮**不投影任何曲线** —— 用户看到的
/// 正是"编辑之后这一段参数被重置"（真机日志：`N of M clip(s) have no region identity`）。
/// 新实现按 clip 自带的源与播放几何反查派生表，把它接回同一份参数权威。
#[test]
fn a_clip_without_a_live_region_key_still_resolves_by_source_and_geometry() {
    let (model, _owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let mut timeline = document
        .timeline
        .lock_or_recover()
        .as_ref()
        .unwrap()
        .clone();
    // 取派生表里一条真实 region 作为"应当被反查到的"目标，并给它一份**唯一**的
    // 源与播放几何 —— 夹具里两条 region 是克隆关系，不这样做反查会命中两条而
    // 判为歧义（那是正确行为，但不是这条测试要验证的）。
    let target_key = *document
        .regions
        .lock_or_recover()
        .keys()
        .next()
        .expect("夹具必须有一条 region");
    let (target_source, target_start, target_length, target_modification) = {
        let mut regions = document.regions.lock_or_recover();
        let region = regions.get_mut(&target_key).expect("目标 region 必须存在");
        region.audio_source_persistent_id = "ara://unique-source".into();
        region.start_in_playback_time = 7.0;
        region.duration_in_playback_time = 3.0;
        (
            region.audio_source_persistent_id.clone(),
            region.start_in_playback_time,
            region.duration_in_playback_time,
            region.audio_modification_persistent_id.clone(),
        )
    };

    let unknown_key = target_key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    assert!(
        !document
            .regions
            .lock_or_recover()
            .contains_key(&unknown_key),
        "构造的 key 必须真的不在派生表里，否则这条测试什么都没验证"
    );

    let clip_id = {
        let clip = timeline.clips.first_mut().expect("夹具必须有一条 clip");
        // clip id 指向一个派生表里不存在的 key，但源与播放几何仍指向那条真实 region。
        clip.id = format!("ara-region-{unknown_key}");
        clip.source_path = Some(target_source.clone());
        clip.start_sec = target_start;
        clip.length_sec = target_length;
        clip.id.clone()
    };

    let identities = {
        let _transaction = document.transaction.lock_or_recover();
        document.parameter_identities_locked(&timeline).unwrap()
    };
    let identity = identities
        .get(&clip_id)
        .expect("身份必须由源 + 播放几何反查出来，而不是判无身份");
    assert_eq!(identity.source, target_source);
    assert_eq!(
        identity.modification, target_modification,
        "反查出的身份必须指向同一条 modification，否则参数权威仍然接不上"
    );
    document.close();
}

/// 形态判定：**激活之前不能落定**，激活之后过宽限期才落定成 Transfer。
///
/// 【为什么这两条一起**】此前判定只发生在 UI 定时器里，而 bootstrap（`hostKind`）与编辑器
/// 会话都要在那之前读形态 —— 于是窗口打开的那一刻永远读到 `Undetermined`，
/// `host_backend()` 退成 ARA-only。判定改成可**按需求值**（`resolve_binding_mode`）之后，
/// 这两条边界必须钉住：不能早（绑定窗口还开着），也不能不落定。
#[test]
fn the_binding_mode_resolves_only_after_the_grace_period() {
    let owner = Arc::new(ExtensionOwner::default());
    // 宿主还没激活：绑定窗口仍然开着，任何结论都是早的。
    owner.resolve_binding_mode();
    assert_eq!(owner.mode(), InstanceMode::Undetermined);

    // 激活但还在宽限期内：同样不落定。
    owner.note_component_activated();
    owner.resolve_binding_mode();
    assert_eq!(owner.mode(), InstanceMode::Undetermined);

    // 过了宽限期仍未绑定 ⇒ Transfer。
    *owner.set_active_at.lock_or_recover() = std::time::Instant::now()
        .checked_sub(ARA_BINDING_GRACE + std::time::Duration::from_secs(1));
    owner.resolve_binding_mode();
    assert_eq!(owner.mode(), InstanceMode::Transfer);
    assert_eq!(
        owner.host_backend().kind(),
        hifishifter_kernel::editor::host::HostKind::Transfer,
        "形态落定之后后端选择必须跟着翻"
    );
}
/// 绑定过又关闭的实例**不能**悄悄变成 Transfer：那才是真的"会话已结束"。
///
/// 【为什么这条与"从未绑定过就给 Transfer 会话"成对】判据是 `bound_once`（单调历史）
/// 而不是"此刻有没有文档"。只看后者的话，一个 ARA 会话结束之后界面会静默变成 Transfer
/// 实例 —— 用户看到的是"宿主的内容没了、只剩一条空轨"，而没有任何地方说会话已经结束。
#[test]
fn an_instance_whose_ara_session_ended_does_not_fall_back_to_transfer() {
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let session = owner.editor_session().unwrap();
    document.close();
    let error = match owner.editor_session() {
        Ok(_) => panic!("绑定过又关闭必须如实报错，而不是换成 Transfer 会话"),
        Err(error) => error,
    };
    assert_eq!(
        crate::editor::errors::classify_error(&error),
        "terminal",
        "会话结束是终态：{error}"
    );
    assert!(
        owner.transfer_session.get().is_none(),
        "不该顺手造一份 Transfer 会话"
    );
    session.close();
}

/// **重开工程：上一份文档关闭后，同一实例必须能重新绑定。**
///
/// 【为什么这是问题二的主回归】companion 的 `bound` 是一次性的：`Drop` 都不复位它。
/// 宿主重开工程复用同一个处理器实例，于是第二次 `bind` 必然被 "already bound" 拒绝。
/// 上一版把这条误当成"宿主重复绑定"，把 null 扩展实例交给宿主 —— 插件加载了却永远
/// 不绑，既有 ARA 数据与新导入音频都不加载，只有删掉插件再添加才好。
///
/// 本用例走真实 C++ shim：第一次绑定成功 → 关闭文档（`closed = true`，binding 保留）
/// → 再次绑定必须**复位 companion 后成功**，而不是返回 null。
#[test]
fn a_reopened_document_rebinds_through_a_reset_companion() {
    use ara2_bridge::companion::CompanionFactory;
    use ara2_bridge::plugin::PluginBuilder;

    let factory = Box::leak(Box::new(
        FactoryBuilder::new("org.hfs.rebind", "org.hfs.rebind.archive")
            .display(
                "rebind",
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                "1",
            )
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap(),
    ));
    factory
        .entry()
        .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
        .unwrap();
    let mut fixture = crate::test_host::HostFixture::new(vec![]);
    let host = fixture.instance();
    let properties = ARADocumentProperties {
        structSize: std::mem::size_of::<ARADocumentProperties>(),
        name: c"rebind".as_ptr(),
    };
    // SAFETY: test host、factory 与 properties 保持到 controller 终止。
    let raw = unsafe {
        (factory
            .raw_copy()
            .createDocumentControllerWithDocument
            .unwrap())(&host, &properties)
    };
    assert!(!raw.is_null());
    // SAFETY: 原生工厂返回完整 packed 实例。
    let controller = unsafe { raw.read_unaligned() };
    // SAFETY: test factory backing 保留到进程结束。
    let association = unsafe { CompanionFactory::from_raw("rebind", &*factory.as_raw()) }.unwrap();
    let owner = Arc::new(ExtensionOwner::default());
    let entry =
        HostEntry::new(association, CompanionRoles::all(), "rebind", owner.clone()).unwrap();

    let mut first = std::ptr::null();
    // SAFETY: controller 存活且已登记。
    assert_eq!(
        unsafe {
            ara2_vst3_plugin_entry_bind(
                entry.as_raw(),
                controller.documentControllerRef.cast(),
                7,
                1,
                1,
                &raw mut first,
            )
        },
        ARA2_VST3_OK
    );
    assert!(!first.is_null(), "第一次绑定必须成功");
    assert_eq!(owner.companion_reset_count(), 0);

    // 工程关闭：文档撤销 lease，实例被关闭（binding 仍保留给宿主最后几次回调）。
    owner.document_closed();
    assert!(owner.was_bound_before(), "历史事实不回落");
    assert!(
        owner.existing_binding_raw().is_none(),
        "关闭后不再复用旧实例指针"
    );

    let mut second = std::ptr::null();
    // SAFETY: 同一 controller，宿主重开工程会这样调用。
    assert_eq!(
        unsafe {
            ara2_vst3_plugin_entry_bind(
                entry.as_raw(),
                controller.documentControllerRef.cast(),
                7,
                1,
                1,
                &raw mut second,
            )
        },
        ARA2_VST3_OK
    );
    assert!(
        !second.is_null(),
        "重开工程必须重新拿到非空扩展实例，而不是 null"
    );
    assert_eq!(owner.companion_reset_count(), 1, "必须复位一次 companion");
    assert_eq!(owner.rebind_generation_count(), 1, "必须走重绑路径");

    // SAFETY: 真实 controller 析构会撤销文档。
    unsafe {
        ((*controller.documentControllerInterface)
            .destroyDocumentController
            .unwrap())(controller.documentControllerRef)
    };
    drop(entry);
    drop(owner);
    factory.entry().uninitialize().unwrap();
}

/// **关闭后的准备队列必须被替换，重绑的实例才排得出快照。**
///
/// 【为什么是问题二 2b】`stop_editor()` 会 `close()` 队列（永久），而旧实现用
/// `OnceLock::get_or_init` —— 重绑时拿回的是同一个死队列，`request()` 恒失败。
#[test]
fn a_closed_preparation_queue_is_replaced_on_rebind() {
    let owner = ExtensionOwner::default();
    owner.ensure_preparation();
    let first = owner.preparation().expect("第一次 ensure 必须建出队列");
    first.close();
    assert!(first.is_closed(), "前提：队列已被关闭");
    owner.ensure_preparation();
    let second = owner.preparation().expect("关闭后必须换一条新队列");
    assert!(!second.is_closed(), "新队列必须是可用的");
    assert!(
        !Arc::ptr_eq(&first, &second),
        "必须真的换了对象，而不是复用已关闭的那条"
    );
}

/// **文档关闭不能丢掉原生宿主适配器；只有组件销毁才释放它。**
///
/// 【为什么这是问题二/三的主回归】`stop_editor()` 同时服务"文档结束"与"组件结束"。
/// 合并在一起会在一份文档关闭时把 REAPER 扩展适配器永久丢掉；宿主复用处理器
/// 重开工程后 `native_host()` 永远是 None，不枚举 item、不填 `region_items`，
/// 界面停在"等待宿主音频"。
#[test]
fn a_document_close_keeps_the_native_host_adapter() {
    let owner = ExtensionOwner::default();
    let host = crate::host::reaper::MockHost::new();
    // SAFETY: MockHost 的 context 在测试期间存活。
    unsafe { owner.bind_host_adapter(host.context()) };
    assert!(owner.native_host().is_some(), "前提：宿主适配器已绑定");

    owner.document_closed();
    assert!(
        owner.native_host().is_some(),
        "文档关闭不是组件结束：宿主适配器必须保留给下一份文档"
    );

    owner.stop_editor();
    assert!(owner.native_host().is_none(), "组件销毁必须释放宿主适配器");
}

/// **旧文档还活着时，实例仍必须能迁移到新文档。**
///
/// 【为什么不能只允许 `closed == true` 后重绑】宿主顺序不可控：REAPER 可能在旧
/// 文档还没关闭时就为新工程发出 bind。此时若拒绝，新文档永远拿不到 renderer。
#[test]
fn a_live_document_does_not_block_rebinding_to_another_document() {
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::ExtensionRoles;

    let owner = Arc::new(ExtensionOwner::default());
    let first = crate::render::document::DocumentSession::new(101);
    let first_raw = owner
        .bind_to_document(
            first.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::all(),
            None,
        )
        .expect("第一份文档必须能绑定");
    assert!(!first_raw.is_null());

    let second = crate::render::document::DocumentSession::new(102);
    let second_raw = owner
        .bind_to_document(
            second.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::all(),
            None,
        )
        .expect("旧文档仍活着时也必须能迁移到新文档");
    assert!(!second_raw.is_null());
    assert_eq!(owner.bound_document_id(), Some(102));
    assert!(
        first.renderer_owners().is_empty(),
        "旧文档不得继续持有本实例的租约"
    );
    assert_eq!(second.renderer_owners().len(), 1);
    first.close();
    second.close();
}
