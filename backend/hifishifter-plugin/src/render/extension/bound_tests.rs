//! `render/extension.rs` 的 `bound_tests` 测试（从内联模块纯搬到这里，`use super::*` 解析不变）。

use super::*;

#[test]
fn an_unassigned_renderer_does_not_start_preparation_work() {
    let (model, _owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    owner.prepare();
    assert!(owner.preparation().is_none());
    let key = (&*ids[0] as *const u8) as u64;
    unsafe {
        let extension = &*raw;
        ((*extension.playbackRendererInterface)
            .addPlaybackRegion
            .unwrap())(extension.playbackRendererRef, key as *mut _);
    }
    owner.prepare();
    assert!(owner.preparation().is_some());
    owner.stop_editor();
    owner.ensure_preparation();
    assert!(owner.preparation().is_none());
    document.close();
}

#[test]
fn offline_empty_renderer_prepares_silence_without_a_periodic_worker() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    *document.timeline.lock_or_recover() = Some(
        serde_json::from_value(serde_json::json!({
            "tracks": [], "clips": [], "bpm": 120, "project_sec": 0
        }))
        .unwrap(),
    );
    document.ready.store(true, Ordering::Release);
    let owner = Arc::new(ExtensionOwner::default());
    owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    owner.prepare();
    assert!(owner.preparation().is_none());
    owner
        .prepare_offline_blocking(std::time::Instant::now() + std::time::Duration::from_secs(3))
        .unwrap();
    let mut left = [9.0; 4];
    let mut right = [9.0; 4];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    // SAFETY: 输出平面各有四帧。
    assert!(unsafe {
        owner.snapshots[0].copy_block(0, 44100, &mut crate::audio_abi::output_bus(&mut bus), 4)
    });
    assert_eq!(left, [0.0; 4]);
    assert_eq!(right, left);
    document.close();
}

#[test]
fn closing_preparation_releases_owner_slot_before_waiting_for_worker() {
    let owner = Arc::new(ExtensionOwner::default());
    owner.ensure_preparation();
    let queue = owner.preparation().unwrap();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    queue
        .request(Box::new(move |_| {
            entered_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(3))
                .unwrap();
            Ok(())
        }))
        .unwrap();
    entered_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    let stopping = owner.clone();
    let closer = std::thread::spawn(move || stopping.stop_editor());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while !queue.is_closed() && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    let closed = queue.is_closed();
    let slot_released = owner.preparation.try_lock().is_ok();
    release_tx.send(()).unwrap();
    closer.join().unwrap();
    assert!(closed, "preparation close did not start");
    assert!(
        slot_released,
        "worker exit must not wait behind its owner slot"
    );
}

/// REAPER asks getState while inserting an instance, before opening its editor.
/// Encoding already owns the document transaction and must not reacquire it
/// through the public workspace-timeline accessor.
#[test]
fn component_state_encoding_after_ready_graph_does_not_relock_transaction() {
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        document
            .write_authority_range("track", "pitch", 100, &[64.; 4], None, false)
            .unwrap();
        document.ready.store(true, Ordering::Release);
        started_tx.send(()).unwrap();
        let result = owners[0].encode_state();
        let _ = result_tx.send(result);
        document.close();
    });
    started_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("component-state fixture must be ready before timing the save");
    let bytes = result_rx
        .recv_timeout(std::time::Duration::from_secs(3))
        .expect("getState must return without relocking its own document transaction")
        .expect("encoding the bound component state must succeed");
    let mut state = crate::state_channel::EditState::default();
    state.restore(&bytes).unwrap();
    assert!(authority_curve(&state, "track", "pitch_edit").contains(&64.));
}

#[test]
fn host_tick_defers_instead_of_waiting_for_a_busy_document_transaction() {
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let stamp = owner.host_query_stamp().unwrap();
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let locked_document = document.clone();
    let holder = std::thread::spawn(move || {
        let _transaction = locked_document.transaction.lock_or_recover();
        held_tx.send(()).unwrap();
        let _ = release_rx.recv();
    });
    held_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let query_owner = owner.clone();
    let query = std::thread::spawn(move || {
        query_owner.refresh_host_transport_tick();
        let _query_scope = super::document::HostQueryScope::nonblocking();
        let deferred = query_owner.host_query_stamp().is_err();
        let authorized = query_owner.host_query_authorized(&stamp);
        done_tx.send((deferred, authorized)).unwrap();
    });
    let before_release = done_rx.recv_timeout(std::time::Duration::from_millis(200));
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    query.join().unwrap();
    assert_eq!(
        before_release.expect("host UI tick must not wait for a background document transaction"),
        (true, false)
    );
    let resumed = owner.host_query_stamp().unwrap();
    assert!(owner.host_query_authorized(&resumed));
    document.close();
}

/// 读某根轨在权威里的整条曲线（跨 run 平铺）。
///
/// 【为什么要这个助手】T2 之后用户曲线**只**住在 `ParamAuthority` 的稀疏 run 里，
/// 显示数组 `params[...]` 不再承载它们 —— 断言必须问权威。
fn authority_curve(state: &crate::state_channel::EditState, root: &str, key: &str) -> Vec<f32> {
    state
        .authority
        .root(root)
        .and_then(|authority| authority.runs.get(key))
        .map(|runs| {
            runs.iter()
                .flat_map(|run| run.values.iter().copied())
                .collect()
        })
        .unwrap_or_default()
}
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
    crate::editor::session::tests::wait_for_source_files(&editor);
    let before = owners[0].encode_state().unwrap();
    let render = || {
        let input = {
            let _transaction = document.transaction.lock_or_recover();
            // 与生产路径同一写法：在短事务里冻结一份编辑快照再交出去。把
            // `lock_or_recover()` 的守卫当参数直接传下去，等于在整个调用期间持有
            // 这把不可重入的锁 —— 任何一次内部回读都会变成自锁。
            let edits = document.edits.lock_or_recover().clone();
            owners[0]
                .ara_render_input(&document, &edits, true)
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
            .into_snapshots()
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
    // 【为什么断言常量而不是字面量】与 `state_channel` 自己的用例同一口径：
    // 字面量会在下一次提升代际时悄悄变成"测试要求旧格式"。
    assert_eq!(
        value["version"].as_u64(),
        Some(crate::state_channel::EditState::STATE_VERSION)
    );
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
        // 同上：冻结一份快照，不在调用期间持有不可重入的编辑锁。
        let edits = document.edits.lock_or_recover().clone();
        owners[0].ara_render_input(&document, &edits, true).unwrap()
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
    assert!(owner
        .host_geometry()
        .unwrap_err()
        .to_string()
        .contains("incompatible"));
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
    assert!(owner
        .host_geometry()
        .unwrap_err()
        .to_string()
        .contains("exactly one"));
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
    assert!(owner
        .host_geometry()
        .unwrap_err()
        .to_string()
        .contains("exactly one"));
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
    assert!(owner
        .host_geometry()
        .unwrap_err()
        .to_string()
        .contains("exactly one"));
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("authorization revoked"));
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

/// 两条组件记录共享完全相同的旧 source/modification 身份时，恢复目标只能由**真实
/// assignment**（本实例被分配的 region）限定。
///
/// 【为什么不再喂归档里的旧字节】那份载荷的曲线只存在**显示数组**（`edits.params`）里，
/// 而 `merge_restore_locked` 收尾一律用**权威**重新合成显示 —— 于是合入的最后一步把它
/// 抹掉。这不是丢了一个用例，而是 round-4 §4.7.3 的结论：插件从未发布过，旧存档迁移
/// **不需要**，唯一保留的形状是"当前编码器写出的载荷能被正确限定"。所以这里用
/// `encode_state()` 现造两份组件载荷，把同一条契约钉在仍然存在的形状上。
#[test]
fn task34_component_state_restore_is_scoped_by_the_real_assignment() {
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    // 【为什么两条根用**同一个**身份】这正是"只能靠 assignment 区分"的场景：两份载荷
    // 认不出谁是谁，唯一可靠的判据是本实例被分配了哪些轨。
    let identity: Vec<(String, String)> =
        vec![("shared-mod".to_string(), "ara://source".to_string())];
    *document.track_bindings.lock_or_recover() = std::collections::BTreeMap::from([
        ("track".into(), identity.clone()),
        ("b".into(), identity.clone()),
    ]);
    // 两份载荷各写一条**不同**的曲线：恢复时若串了轨，值会立刻对不上。
    document
        .write_authority_range("track", "pitch", 100, &[32.; 4], None, false)
        .unwrap();
    document
        .write_authority_range("b", "pitch", 200, &[64.; 12], None, false)
        .unwrap();
    let states = owners
        .iter()
        .map(|owner| owner.encode_state().unwrap())
        .collect::<Vec<_>>();

    // 与宿主冷重建一致：完整图 ready 前暂存全部组件原始 bytes，随后统一合入权威。
    let (cold_model, cold_owners, cold_ids) = crate::editor::session::tests::workspace_fixture();
    let cold = cold_model.session();
    *cold.track_bindings.lock_or_recover() = std::collections::BTreeMap::from([
        ("track".into(), identity.clone()),
        ("b".into(), identity.clone()),
    ]);
    cold.ready.store(false, Ordering::Release);
    for (owner, bytes) in cold_owners.iter().zip(&states) {
        owner.restore_state(bytes).unwrap();
    }
    cold.prepare_renderers();
    let accepted = cold.edits.lock_or_recover().clone();
    assert_eq!(
        authority_curve(&accepted, "track", "pitch_edit"),
        vec![32.; 4],
        "组件 A 的载荷只能落回它被分配的那条根"
    );
    assert_eq!(
        authority_curve(&accepted, "b", "pitch_edit"),
        vec![64.; 12],
        "组件 B 的曲线既不能被写进 A 的根，也不能被 A 覆盖"
    );
    assert!(accepted
        .tracks
        .iter()
        .all(|track| track.id == "track" || track.id == "b"));
    for (index, id) in ["track", "b"].into_iter().enumerate() {
        let bytes = cold_owners[index].encode_state().unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // 【单一版本】组件状态只有一个代际（`EditState::STATE_VERSION`），内容不再
        // 决定版本号 —— 此前那个"没有曲线 ⇒ 2 / 有曲线 ⇒ 3"的阶梯与全部迁移一起删掉了。
        assert_eq!(
            saved["version"].as_u64(),
            Some(crate::state_channel::EditState::STATE_VERSION)
        );
        assert_eq!(saved["edits"]["tracks"].as_array().unwrap().len(), 1);
        assert_eq!(saved["edits"]["tracks"][0]["id"], id);
        assert_eq!(saved["edits"]["bindings"].as_object().unwrap().len(), 1);
        // 【载荷只带自己那一条根】两个组件共享同一身份，裁剪判据只能是 assignment。
        let params = saved["edits"]["params"].as_object().unwrap();
        assert_eq!(params.len(), 1, "{saved}");
        assert!(params.contains_key(id), "{saved}");
    }
    // 真正扩大同一组件 assignment，使两个完全相同身份同时进入候选集：只能不套用。
    let key = (&*cold_ids[1] as *const u8) as u64;
    let raw = cold_owners[0]
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
    let before = cold.edits.lock_or_recover().revision;
    cold_owners[0].restore_state(&states[1]).unwrap();
    // 【为什么不再要求失败】身份歧义只该让这一份恢复不套用，不该让宿主保存失败；
    // 权威仍保留原曲线（下面两条断言）。
    assert!(
        cold_owners[0].encode_state().is_ok(),
        "身份歧义不能让保存失败"
    );
    assert_eq!(cold.edits.lock_or_recover().revision, before);
    assert_eq!(
        cold.edits.lock_or_recover().params["b"].pitch_edit[200],
        64.
    );
    // 缺少真实身份也不能按旧序号恢复；既有曲线留在权威中。
    cold.track_bindings.lock_or_recover().insert(
        "track".into(),
        vec![("unknown-mod".into(), "ara://source".into())],
    );
    cold.track_bindings.lock_or_recover().insert(
        "b".into(),
        vec![("unknown-b".into(), "ara://source".into())],
    );
    cold_owners[0].restore_state(&states[1]).unwrap();
    assert!(
        cold_owners[0].encode_state().is_ok(),
        "缺身份不能让保存失败"
    );
    assert_eq!(
        cold.edits.lock_or_recover().params["b"].pitch_edit[200],
        64.
    );
    cold.close();
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
        // 【为什么先等后台排空】`bind_to_document` 尾部会把 prepare_job 排上后台线程；
        // 该 job 从空时间线渲染出 0 帧空快照并 `publish` 原子替换 `current`，同时把
        // 下面手工发布的那份（无读者、已非 current）`reclaim` 掉 —— 若它晚于手工
        // publish 落地，`copy_block` 读到的就是空快照（available=true、内容全零），
        // 本用例就间歇性失败。先等它退场再手工发布，用例与后台 worker 不再共享时序。
        owner.await_preparation_idle(std::time::Duration::from_secs(3));
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
        let baseline = document.workspace_view().unwrap();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let result = document.apply_workspace_edits(
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
                                &baseline,
                            )
                            .unwrap();
                    }
                }
                true
            },
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(if model_changed {
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
        let available = unsafe {
            owner.snapshots[0].copy_block(0, 44100, &mut crate::audio_abi::output_bus(&mut bus), 4)
        };
        assert!(available);
        assert_eq!(left, [0.25; 4]);
        assert_eq!(right, [0.5; 4]);
        // 真正"这个实例不再渲染"的路径仍然清空：那里由文档关闭负责。
        document.revoke_renderers();
        let available = unsafe {
            owner.snapshots[0].copy_block(0, 44100, &mut crate::audio_abi::output_bus(&mut bus), 4)
        };
        assert!(!available);
        drop(model);
    }
}

/// 即使其它线程暂持模型事务，真实prepare callback也必须返回，并在后台最终发布。
#[test]
fn prepare_callback_does_not_wait_for_a_model_transaction() {
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let sources = document.edit_sources.lock_or_recover().clone();
    *document.sources.lock_or_recover() = sources;
    owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
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
    assert!(
        owner.preparation_state().1.is_none(),
        "{:?}",
        owner.preparation_state()
    );
    let mut left = [9.0; 4];
    let mut right = [9.0; 4];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    // SAFETY: 输出平面各有四帧。
    assert!(unsafe {
        owner.snapshots[0].copy_block(0, 44100, &mut crate::audio_abi::output_bus(&mut bus), 4)
    });
    assert_eq!(left, [0.1, 0.2, 0.3, 0.4]);
    assert_eq!(right, left);
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
        assert!(unsafe {
            owner.snapshots[0].copy_block(0, 44100, &mut crate::audio_abi::output_bus(&mut bus), 4)
        });
        for (actual, expected) in left.iter().zip(if index == 0 {
            [0.1, 0.2, 0.3, 0.4]
        } else {
            [0.1, 0.2, 0.3, 0.4]
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

/// 形态判定：**激活之前不能落定**，激活之后过宽限期才落定成 Capture。
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

    // 过了宽限期仍未绑定 ⇒ Capture。
    *owner.set_active_at.lock_or_recover() = std::time::Instant::now()
        .checked_sub(ARA_BINDING_GRACE + std::time::Duration::from_secs(1));
    owner.resolve_binding_mode();
    assert_eq!(owner.mode(), InstanceMode::Capture);
    assert_eq!(
        owner.host_backend().kind(),
        hifishifter_kernel::editor::host::HostKind::Capture,
        "形态落定之后后端选择必须跟着翻"
    );
}
/// 绑定过又关闭的实例**不能**悄悄变成 Capture：那才是真的"会话已结束"。
///
/// 【为什么这条与"从未绑定过就给 Capture 会话"成对】判据是 `bound_once`（单调历史）
/// 而不是"此刻有没有文档"。只看后者的话，一个 ARA 会话结束之后界面会静默变成 Capture
/// 实例 —— 用户看到的是"宿主的内容没了、只剩一条空轨"，而没有任何地方说会话已经结束。
#[test]
fn an_instance_whose_ara_session_ended_does_not_fall_back_to_capture() {
    let (model, owner, _id) = crate::editor::session::tests::fixture();
    let document = model.session();
    let session = owner.editor_session().unwrap();
    document.close();
    let error = match owner.editor_session() {
        Ok(_) => panic!("绑定过又关闭必须如实报错，而不是换成 Capture 会话"),
        Err(error) => error,
    };
    assert_eq!(
        crate::editor::errors::classify_error(&error),
        "terminal",
        "会话结束是终态：{error}"
    );
    assert!(
        owner.capture_session.get().is_none(),
        "不该顺手造一份 Capture 会话"
    );
    session.close();
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

/// **同步发布第一份快照的前提：源 PCM 必须已经在手上。**
///
/// 【为什么这条是"每隔一小会儿一个静音断点"的一半】**宿主不会等我们的后台队列**：
/// 实例建好之后 REAPER 立刻按 region 一块块渲染（实测 21 个块 ≈225 ms 里我们还没有
/// 快照，于是逐样本写零，而宿主把那些零烘进自己的 ARA 缓存）。所以第一份快照必须在
/// 宿主问第一块之前就发布 —— 这也意味着"什么时候可以同步发布"必须判准：源 PCM 由
/// 宿主异步授权，素材没到时同步渲染必然 `MissingSource`，白烧掉那次机会。
///
/// 本用例钉的就是这个前提：没有 PCM 时不许认为"可以算了"，PCM 到手之后必须是。
#[test]
fn the_inline_first_snapshot_sees_the_host_pcm_and_publishes() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    assert!(owner.is_bound(), "夹具的实例绑定在 ARA 文档上");
    // 夹具只塞了 `edit_sources`（内核渲染读的那一份）；直通通路读的是 `sources`。
    document.sources.lock_or_recover().insert(
        "ara://source".into(),
        Arc::new(crate::render::source::SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        }),
    );
    assert!(
        owner.has_pending_render_sources(),
        "PCM 到手之后必须认为可以渲染，否则第一份快照永远发不出去"
    );
    // 并证明这条前提确实通向"能发布"：素材在就是能算出来。
    owner
        .prepare_job(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
            false,
        )))
        .expect("素材在手时第一份快照必须能算出并发布");
    assert!(
        owner.snapshots.iter().all(|snapshot| snapshot.is_ready()),
        "算完就必须有可播的快照"
    );
    document.close();
}

/// **ARA 回退的渲染事实必须与原生写者同身份、同口径（合并，不许整表替换）。**
///
/// 现场形状（2026-11-22 02:06:22）：REAPER 按轨道管理 ARA 实例——持有原生适配器的
/// 编辑实例整表重建 render_facts（item GUID 键、`B_LOOPSRC=true`），而**没有**原生
/// 适配器的播放实例走 `refresh_ara_render_facts`。此前那里用 `ara-region-*` 键 +
/// 原生字段硬编码 false 整表替换 —— 两个写者逐轮翻转 `loops=1↔0`，render_epoch 每
/// 轮推进，`prepare_job` 的版本闸永远失配 ⇒ 一次光标移动连出 4 轮全量 mixdown
/// （状态栏"渲染中"随清单刷新闪烁）。
#[test]
fn the_ara_render_fact_fallback_merges_with_native_facts_instead_of_replacing_them() {
    use crate::host::geometry::{RenderFacts, RenderItemFacts};

    let owner = ExtensionOwner::default();
    let document = crate::render::document::DocumentSession::new(202);
    // 原生写者的产物：item GUID 键 + 原生真值（循环源开、声道模式 2）。
    document.render_facts.lock_or_recover().items.insert(
        "item-A".to_string(),
        RenderItemFacts {
            reversed: false,
            loop_enabled: true,
            channel_mode: 2,
            timestretch: false,
            content_fade_head: false,
            content_fade_tail: false,
        },
    );
    // ARA 侧：region 1 已认领到 item-A，时间拉伸开启。
    let identity = Box::new(0_u8);
    let key = (&*identity as *const u8) as u64;
    document.regions.lock_or_recover().insert(
        key,
        crate::ara::AraPlaybackRegion {
            audio_source_persistent_id: "source".into(),
            is_timestretch_enabled: true,
            ..Default::default()
        },
    );
    document
        .region_items
        .lock_or_recover()
        .insert(key, "item-A".to_string());

    let stamp = (
        document.clone(),
        document.revision.load(Ordering::Acquire),
        document.scope_revision.load(Ordering::Acquire),
        Vec::new(),
    );
    owner.refresh_ara_render_facts(&stamp);

    let facts: &RenderFacts = &document.render_facts.lock_or_recover();
    assert_eq!(
        facts.items.len(),
        1,
        "不许出现 ara-region-* 的并行身份：键必须与原生写者一致"
    );
    let entry = facts
        .items
        .get("item-A")
        .expect("item-A 必须仍在（同一身份被更新，而不是被默认表覆盖）");
    assert!(
        entry.loop_enabled && entry.channel_mode == 2,
        "读不到的原生字段必须保留上一轮的真值，不许被安全默认覆盖"
    );
    assert!(entry.timestretch, "ARA 能给出的字段必须照实更新");
}

/// **不变量：宿主回调链上的首快照只排队，绝不等待 worker。**
///
/// 【为什么钉死】现场 REAPER 7.8.5 的 `AppHangB1` 就是这条：`endEditing` 里等了 2 s
/// 的 `wait_idle_until`，Windows 判主线程"不再与窗口交互"并关掉它。上界不是零 ——
/// 只要宿主在必须立刻返回的回调里等，它就已经在挂起的路上了。
#[test]
fn a_host_callback_never_asks_the_worker_to_wait() {
    let budget = std::time::Duration::from_millis(250);
    // 守卫外：预算照用（离线/测试这类非宿主线程可以等一个有界的时间）。
    assert_eq!(
        ExtensionOwner::first_snapshot_budget(Some(budget)),
        Some(budget)
    );
    assert_eq!(ExtensionOwner::first_snapshot_budget(None), None);
    {
        let _call = crate::host::HostCallGuard::enter("test");
        assert_eq!(
            ExtensionOwner::first_snapshot_budget(Some(budget)),
            None,
            "宿主调用里必须降级成\"只排队\""
        );
    }
    // 守卫退出后立刻恢复：一次宿主调用不能永久关掉这条能力。
    assert_eq!(
        ExtensionOwner::first_snapshot_budget(Some(budget)),
        Some(budget)
    );
}

/// **不变量：`prepare_renderers` 走的是"只排队"那条。**
///
/// 【为什么单独一条】`DocumentSession::prepare_renderers` 的主要调用者是 ARA 的
/// `end_editing`（`remap_and_log` 的尾巴）与存档回调 —— 两者都在宿主线程上。
/// 它一旦传预算，`first_snapshot_budget` 就只剩 `in_host_call()` 一道保险；
/// 而宿主**自己**触发的编辑（用户在 REAPER 里拖 item）根本不经过我们的守卫。
#[test]
fn the_document_prepare_never_waits_for_the_worker() {
    let (model, _owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    // 走的是与 `end_editing` 完全相同的入口。
    document.prepare_renderers();
    let waits = owner.preparation().map_or(0, |queue| queue.waits());
    assert_eq!(waits, 0, "文档收口路径不得等准备 worker");
    document.close();
}
