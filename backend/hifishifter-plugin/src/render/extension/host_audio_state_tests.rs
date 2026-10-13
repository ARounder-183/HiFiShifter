//! 宿主音频分类的纯函数回归。

use super::*;

#[test]
fn a_claimed_item_means_ready() {
    let status = classify_host_audio(false, 3, 1, 0, 0);
    assert_eq!(status.state, HostAudioState::Ready);
    assert_eq!(status.waiting_items, 0);
}

#[test]
fn an_empty_project_is_not_a_fault() {
    let status = classify_host_audio(false, 0, 0, 0, 0);
    assert_eq!(status.state, HostAudioState::Ready);
    assert_eq!(status.waiting_items, 0);
}

#[test]
fn items_without_regions_on_a_plain_track_are_awaiting() {
    let status = classify_host_audio(false, 2, 0, 0, 0);
    assert_eq!(status.state, HostAudioState::AwaitingRegions);
    assert_eq!(status.waiting_items, 2);
}

#[test]
fn items_without_regions_on_a_folder_parent_are_named_precisely() {
    let status = classify_host_audio(true, 2, 0, 0, 0);
    assert_eq!(status.state, HostAudioState::FolderParentWithoutRegions);
    assert_eq!(status.waiting_items, 2);
}

/// 认领数超过清单数（诊断计数口径不一致）时不得下溢成天文数字。
#[test]
fn an_over_counted_claim_does_not_underflow() {
    let status = classify_host_audio(false, 1, 5, 0, 0);
    assert_eq!(status.state, HostAudioState::Ready);
    assert_eq!(status.waiting_items, 0);
}

/// **"拿到了 region 但一条都绑不上"必须与"还没拿到"分开报。**
///
/// 【为什么这是回归测试】这正是用户报障的场景：FX 挂在 folder 父轨上，宿主分配了
/// 整组 region（实测 2 条），而候选 item 只有 1 个 ⇒ 按几何唯一匹配一条都对不上。
/// 此前这个读数**只存在于刷屏日志里**（见 `BindingReport`），界面上没有任何对应物。
/// 判据一旦退化成"等待中"，用户就又回到"时间线是空的、没有任何说明"。
#[test]
fn assigned_but_unbound_regions_are_reported_as_such() {
    let status = classify_host_audio(false, 1, 0, 2, 2);
    assert_eq!(status.state, HostAudioState::UnmatchedRegions);
    assert_eq!(status.waiting_items, 1);
    // 优先于 folder 判定：它更具体（folder 判定说的是"一个都没拿到"）。
    let status = classify_host_audio(true, 1, 0, 2, 2);
    assert_eq!(
        status.state,
        HostAudioState::UnmatchedRegions,
        "已分配但未绑定比 folder 父轨更具体，必须优先报出"
    );
}

/// 部分绑定（1/2）不算本态：有片段能显示，就不是"片段凭空消失"那种症状。
///
/// 【为什么钉住】把部分绑定也报成"未绑定"会让一个**大部分正常**的工程挂上一条
/// 误导性的警告。本态专指"一条都没绑上"。
#[test]
fn partial_binding_is_not_reported_as_unmatched() {
    let status = classify_host_audio(false, 2, 1, 2, 1);
    assert_eq!(status.state, HostAudioState::Ready);
}

/// 读不出绑定读数（还没求解过）时传 0 ⇒ 退回既有判定，不得凭空报"未绑定"。
#[test]
fn an_unknown_binding_reading_falls_back_to_the_previous_classification() {
    let status = classify_host_audio(false, 2, 0, 0, 0);
    assert_eq!(status.state, HostAudioState::AwaitingRegions);
    let status = classify_host_audio(true, 2, 0, 0, 0);
    assert_eq!(status.state, HostAudioState::FolderParentWithoutRegions);
}

/// `process` 选快照必须按 `setupProcessing` 的**权威率**，不能按宿主上下文率拒绝。
///
/// 【为什么这是回归测试】此前 `process` 用 `ProcessContext::sample_rate` 且只认
/// 44.1/48k，其余 `kResultFalse` —— 而 `setupProcessing` 已接受 44.1..192k。两者
/// 不一致时（96k 宿主）音频线程持续返回错误码：静音、甚至被踢出链路。
#[test]
fn the_snapshot_index_follows_the_declared_setup_rate() {
    let owner = ExtensionOwner::default();
    // 未声明 ⇒ 按传入率就近。
    assert_eq!(owner.snapshot_index_for_rate(44100.0), 0);
    assert_eq!(owner.snapshot_index_for_rate(48000.0), 1);
    // 非原生率：整数倍优先（88.2k→44.1k，176.4k→44.1k），96k/192k→48k。
    assert_eq!(owner.snapshot_index_for_rate(88200.0), 0);
    assert_eq!(owner.snapshot_index_for_rate(176400.0), 0);
    assert_eq!(owner.snapshot_index_for_rate(96000.0), 1);
    assert_eq!(owner.snapshot_index_for_rate(192000.0), 1);
    // 非整数倍（44.1k 的奇数倍之外）退到最近的原生率。
    assert_eq!(owner.snapshot_index_for_rate(50000.0), 1);
    assert_eq!(owner.snapshot_index_for_rate(44101.0), 0);
    // `setupProcessing` 的声明优先：96k 工程里即便 context 报 44.1 也按 96k 选。
    owner.set_setup_sample_rate(96000.0);
    assert_eq!(owner.setup_sample_rate(), 96000.0);
    assert_eq!(owner.snapshot_index_for_rate(44100.0), 1);
    // 非法声明被忽略（保持上一次的有效值）。
    owner.set_setup_sample_rate(f64::NAN);
    assert_eq!(owner.setup_sample_rate(), 96000.0);
    owner.set_setup_sample_rate(0.0);
    assert_eq!(owner.setup_sample_rate(), 96000.0);
}

/// 分类名是前端查 catalog 的键：改名即破坏文案，必须显式钉住。
///
/// 【为什么只剩这几态】宿主支持类提示（"本宿主尚不支持" / "此宿主只提供 ARA"）已整体
/// 删除：前者从来没有产出者（`bind_host_adapter` 只存 `Some(Native)` 或 `None`，而
/// `host_backend()` 把 `None` 映射成 `AraOnly`），后者对每一个能打开编辑器的宿主都是
/// 假的 —— ARA-only 宿主本来就正常工作，只读的只有片段几何写回，而那由能力位门控。
#[test]
fn classification_names_are_stable_catalog_keys() {
    assert_eq!(HostAudioState::Ready.as_str(), "ready");
    assert_eq!(HostAudioState::AwaitingRegions.as_str(), "awaiting_regions");
    assert_eq!(
        HostAudioState::FolderParentWithoutRegions.as_str(),
        "folder_parent_without_regions"
    );
    // 【为什么单独钉住这一条】它是"拿到了 region 但绑不上"的唯一分类名。与
    // `folder_parent_without_regions` 合并会让"宿主分配了但匹配不上"被说成
    // "宿主没分配"，而两者的成因与处置不同。
    assert_eq!(
        HostAudioState::UnmatchedRegions.as_str(),
        "unmatched_regions"
    );
    assert_ne!(
        HostAudioState::UnmatchedRegions.as_str(),
        HostAudioState::FolderParentWithoutRegions.as_str(),
        "「分配了但绑不上」不是「没拿到 region」"
    );
}

/// **不变量：渲染输入失效必须通知本文档的**全部** owner，而不只是发现变化的那个实例。**
///
/// 【为什么这是回归测试】REAPER 按轨道管理 ARA 实例：20 ms 定时器与清单采样跑在
/// 编辑/视图实例上（`role == 2`），而快照被播放渲染实例持有（`role == 1`）。
/// 此前 `RenderFacts` 变化时调用的是 `self.prepare()` —— 编辑实例的 `prepare()`
/// 首行就因 `!renders_playback()` 直接返回，播放实例**从未被通知**。用户报障：
/// "在宿主里倒放 Take，波形更新了、声音没变，编辑一下才对"（编辑路径恰好遍历了
/// 全部 owner，所以"碰巧"恢复）。单实例夹具无法区分"通知全部"与"只通知自己"，
/// 因此必须让编辑实例触发、断言播放实例被通知。
#[test]
fn render_input_invalidation_reaches_the_playback_instance() {
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let playback = owners[0].clone();
    assert!(
        playback.renders_playback(),
        "夹具第一个实例必须是播放渲染器"
    );
    let editor = Arc::new(ExtensionOwner::default());
    let _raw = editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    assert!(
        editor.is_editor_only(),
        "本夹具的第二个实例必须是编辑器实例"
    );
    // 编辑实例自己的 prepare() 是合法 no-op（它不持有快照）；关键是播放实例必须被通知。
    let before = playback.prepare_requests.load(Ordering::Acquire);
    let epoch_before = document.render_epoch.load(Ordering::Acquire);
    // 持事务调用：`prepare_job` 会在事务处阻塞，后台不会真的去渲染合成夹具文档
    // （与 `vst3.rs` 的离线 setup 用例同一手法）。`invalidate_render_input` 刻意
    // **不**取事务，所以这里不会自锁。
    let held = document.transaction.lock_or_recover();
    document.invalidate_render_input();
    assert!(
        document.render_epoch.load(Ordering::Acquire) > epoch_before,
        "渲染输入失效必须推进 render_epoch（PreparedVersion 才会失配）"
    );
    assert_eq!(
        playback.prepare_requests.load(Ordering::Acquire),
        before + 1,
        "播放实例必须被通知重排，而不是只通知发现变化的实例"
    );
    drop(held);
    // 放行后等后台排空，避免用例结束时与仍在运行的任务抢生命周期。
    for owner in [&playback, &owners[1]] {
        if let Some(Ok(worker)) = owner.preparation.get() {
            let _ = worker
                .wait_idle_until(std::time::Instant::now() + std::time::Duration::from_secs(3));
        }
    }
}

/// **ARA-only 后端的渲染事实必须推进 `render_epoch`。**
///
/// 【为什么这是回归测试】原生扩展缺席时 `refresh_ui_inventory` 此前直接 `return`，
/// `render_facts` 永不更新、`render_epoch` 永不推进 ⇒ 宿主在 ARA 侧改了时间拉伸 /
/// 内容淡化，插件继续用旧配置渲染（静默的正确性 bug）。本用例不绑任何原生宿主，
/// 只翻 ARA region 的 `is_timestretch_enabled`，断言事实被采到且渲染被失效。
#[test]
fn ara_only_backend_still_advances_the_render_epoch() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    assert!(
        owner.native_host().is_none(),
        "夹具必须是没有原生扩展的 ARA-only 后端"
    );
    // 第一轮：时间拉伸关 ⇒ 事实从"空"变为"一个 region" ⇒ epoch 前进。
    owner.refresh_ui_inventory();
    let epoch = document.render_epoch.load(Ordering::Acquire);
    let facts = document.render_facts.lock_or_recover().clone();
    assert_eq!(facts.items.len(), 1, "ARA region 必须进渲染事实");
    assert!(
        facts.items.values().all(|f| !f.timestretch),
        "时间拉伸关时事实为假"
    );
    // 事实未变 ⇒ 幂等，不推进 epoch（否则空闲工程会被搅得反复重绘）。
    owner.refresh_ui_inventory();
    assert_eq!(
        document.render_epoch.load(Ordering::Acquire),
        epoch,
        "事实未变不得推进 render_epoch"
    );
    // 宿主在 ARA 侧开启时间拉伸 ⇒ 必须重新失效渲染。
    for region in document.regions.lock_or_recover().values_mut() {
        region.is_timestretch_enabled = true;
    }
    owner.refresh_ui_inventory();
    assert!(
        document.render_epoch.load(Ordering::Acquire) > epoch,
        "ARA 侧时间拉伸变化必须推进 render_epoch"
    );
}

/// **路由选择：有宿主撤销栈走宿主、没有走本地。**
///
/// 【为什么这是回归测试】`webview.rs` 用 `history_authority().backend() == "host"` 决定
/// `undo_timeline` / `redo_timeline` / `set_history_position` 走宿主异步路径还是本地队列。
/// 判别式一旦漂移（曾把 `"host"` 写成 `"reaper"`），REAPER 里宿主撤销实际会走本地栈，
/// `history_jump*` 整条路径不可达。这里把两条栈的路由都钉死。
#[test]
fn history_routing_follows_the_authority_backend() {
    // 无原生扩展 ⇒ 本地权威栈。
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    assert_eq!(
        owner.history_authority().backend(),
        "local",
        "没有宿主撤销栈时必须走本地队列"
    );
    model.session().close();

    // 绑定原生宿主（含 Undo_* 与写接口）⇒ 宿主权威栈。
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let host = crate::host::MockHost::new();
    host.enable_writer();
    // SAFETY: 夹具与 owner 在整个用例期间存活。
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    assert!(owner.native_host().is_some(), "绑定后必须拿到原生适配器");
    assert_eq!(
        owner.history_authority().backend(),
        "host",
        "有宿主撤销栈时必须走宿主异步路径"
    );
    model.session().close();
}

/// **不变量：take 级事实变化（倒放 / 声道模式）不能被 `change count` 早退挡住。**
///
/// 【为什么这是回归测试】`ui_inventory_stamp` 早退以 `GetProjectStateChangeCount`
/// 为主，而方向是 **take 级**属性 —— 宿主未必为它推进项目 change count。只靠令牌
/// 早退会让"倒放 Take"连采样这一步都被跳过：`render_facts` 不更新 ⇒ 渲染输入不失效
/// ⇒ 波形变了、声音没变。夹具把 change count 钉住不动，只翻 `take_reversed`，断言
/// 事实仍被采到、渲染仍被失效（`FACTS_REFRESH_INTERVAL` 到期即强制重采）。
#[test]
fn a_take_level_fact_change_survives_the_change_count_early_return() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    let host = crate::host::MockHost::new();
    host.enable_takes(0);
    host.enable_media();
    host.clear_markers();
    // SAFETY: 夹具与 owner 在整个用例期间存活。
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    // 第一轮完整采集：方向按正放。
    owner.refresh_host_transport();
    assert!(
        document
            .render_facts
            .lock_or_recover()
            .items
            .values()
            .all(|item| !item.reversed),
        "首轮必须采到正放，否则这条测试对缺陷不敏感"
    );
    let epoch_before = document.render_epoch.load(Ordering::Acquire);
    let change = host.change_count();

    // 宿主倒放该 take，但 **不** 推进 change count（这正是要兜住的情况）。
    host.section_reports.set(true);
    host.take_reversed.set(true);
    assert_eq!(
        host.change_count(),
        change,
        "夹具必须把 change count 钉住不动，否则测不到兜底路径"
    );
    // 让强制重采兜底路径生效：把上次采集时刻倒推到间隔之前。
    *document.ui_inventory_sampled_at.lock_or_recover() = Some(
        std::time::Instant::now() - FACTS_REFRESH_INTERVAL - std::time::Duration::from_secs(1),
    );

    owner.refresh_ui_inventory();

    assert!(
        document
            .render_facts
            .lock_or_recover()
            .items
            .values()
            .all(|item| item.reversed),
        "take 级方向变化必须穿透 change-count 早退被采到"
    );
    assert!(
        document.render_epoch.load(Ordering::Acquire) > epoch_before,
        "方向变化必须失效渲染输入，否则波形变了、声音不变"
    );
    document.close();
}

/// GUI 载荷只带**编辑器实例**的读数：playback 实例没有 GUI，它的读数不该出现在
/// 用户的窗口里（那会让一个正常工作的实例替另一个实例"背锅"）。
#[test]
fn the_payload_carries_the_editor_instances_reading() {
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let editor = Arc::new(ExtensionOwner::default());
    let _raw = editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER | ExtensionRoles::EDITOR_VIEW,
            None,
        )
        .unwrap();
    assert!(
        editor.is_editor_only(),
        "本夹具的第二个实例必须是编辑器实例"
    );
    // playback 实例报"正常"、编辑器实例报 folder 判据：载荷必须取后者。
    *owners[0].host_audio.lock_or_recover() = HostAudioStatus::default();
    *editor.host_audio.lock_or_recover() = HostAudioStatus {
        state: HostAudioState::FolderParentWithoutRegions,
        waiting_items: 2,
    };
    let mut payload = serde_json::json!({"ok":true});
    document.decorate_host_audio(&mut payload);
    assert_eq!(
        payload["host_audio"]["state"],
        "folder_parent_without_regions"
    );
    assert_eq!(payload["host_audio"]["waiting_clips"], 2);
    document.close();
}

/// 没有编辑器实例时**不写**该字段：前端沿用上一次已知值，而不是凭空报一个"正常"。
#[test]
fn a_document_without_an_editor_instance_writes_nothing() {
    let (model, _owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let mut payload = serde_json::json!({"ok":true});
    document.decorate_host_audio(&mut payload);
    assert!(payload.get("host_audio").is_none());
    document.close();
}

/// **只拿到 EDITOR_RENDERER 的实例必须保持透传，即使文档里没有 playback 实例。**
///
/// 【为什么这条是"不许改"的契约】"宿主一个 playback 角色都不分配"与"本实例只是
/// 编辑实例"在文档级**完全同形**（两者都是 known=all, assigned=editor），而 ARA 没有
/// "文档级哪个实例是播放渲染器"的查询。曾试过让 editor 实例在"没有 playback 实例"
/// 时接管渲染 —— 它会让这个用例（以及 `vst3/audio_boundary_tests` 的整组透传用例）
/// 失败：编辑实例开始渲染，宿主音频被清掉。
///
/// 所以行为**不变**，只把事实做成读数
/// （`DocumentSession::report_playback_ownership`，见下一个用例）。
#[test]
fn an_editor_only_instance_stays_passthrough_even_without_a_playback_instance() {
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let editor = Arc::new(ExtensionOwner::default());
    let _raw = editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER | ExtensionRoles::EDITOR_VIEW,
            None,
        )
        .unwrap();
    assert!(
        !editor.has_playback_role(),
        "本用例的实例不能拿到宿主分配的 playback 角色"
    );
    assert!(
        !editor.renders_playback(),
        "没有 playback 角色就不渲染 —— 这条契约是刻意的，见本用例的说明"
    );
    assert!(
        editor.is_editor_only(),
        "editor-only 必须透传，否则会清掉宿主已经处理好的音频"
    );
    document.close();
}

/// **"有实例但没有任何一个持 playback 角色"必须是可见的读数。**
///
/// 这是替代"自动接管"的取证手段：Cubase 上"编辑不改变声音"究竟是不是角色分配造成的，
/// 靠这一行日志判定，而不是靠改行为去猜。有 playback 实例时则**不报**。
#[test]
fn a_document_with_no_playback_instance_reports_it_instead_of_taking_over() {
    // 只有 editor 实例 ⇒ 应当报警（读数可见）。
    let model = crate::ara::model::ModelHandle::new();
    let document = model.session();
    let editor = Arc::new(ExtensionOwner::default());
    let _raw = editor
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    assert_eq!(
        document
            .renderer_owners()
            .iter()
            .filter(|owner| owner.has_playback_role())
            .count(),
        0,
        "本用例故意只绑定 editor 实例"
    );
    // 只断言读数函数可安全调用（它是限流日志，不断言文本）。
    document.report_playback_ownership();
    assert!(editor.is_editor_only(), "报读数**不得**顺带改变行为");
    document.close();

    // 有 playback 实例 ⇒ 读数不报（此断言通过计数表达：playback 存在）。
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    assert!(
        owners.iter().any(|owner| owner.has_playback_role()),
        "夹具必须有一个宿主分配的 playback 实例"
    );
    document.report_playback_ownership();
    document.close();
}
