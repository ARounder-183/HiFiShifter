//! 健康探活（`plugin_health` / `get_runtime_info`）的**无锁**不变量。
//!
//! 【为什么单独一份】这两条测的不是某个功能的输出，而是"GUI 线程上的命令绝不去取
//! 那些会被 actor 长时间持有的锁"。它们与 `bound_tests`（绑定/重绑行为）判的是两件
//! 事，混在一起时一旦失败，读的人分不清是绑定语义变了还是锁序变了。

use super::*;

/// **不变量：健康探活不碰时间线锁，也不创建会话。**
///
/// 【为什么要紧】`plugin_health` / `get_runtime_info` 是状态栏 4 Hz 的轮询命令，跑在
/// GUI 线程上。它们一旦去取 `timeline` 锁，就会与 actor 撞上同一个 ABBA：actor 持锁
/// 的同时往 WebView 投递，而投递需要 GUI 线程泵消息 —— 现场表现就是状态栏红条
/// `BackendInvokeError`。这条测试在另一条线程里**长时间**持锁，然后要求探活仍然立刻
/// 回答：它一旦开始取那把锁，这里就会等到锁释放（远超预算）。
#[test]
fn the_health_snapshot_answers_while_the_timeline_is_locked() {
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
    let holder = Arc::clone(&document);
    let (locked_tx, locked_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let guard = std::thread::spawn(move || {
        let _timeline = holder.timeline.lock_or_recover();
        locked_tx.send(()).unwrap();
        // 一直持锁，直到主线程量完这次探活。
        let _ = release_rx.recv_timeout(std::time::Duration::from_secs(5));
    });
    locked_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("测试线程必须真的拿到时间线锁");
    let started = std::time::Instant::now();
    let snapshot = owner.health_snapshot();
    let elapsed = started.elapsed();
    let _ = release_tx.send(());
    guard.join().unwrap();
    assert_eq!(snapshot["ok"], serde_json::json!(true));
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "探活不许等时间线锁，实际等了 {elapsed:?}"
    );
    document.close();
}

/// **不变量：探活不创建编辑器会话。**
///
/// 【为什么】创建会话会拉起 actor 线程、扫临时目录、初始化模型资源 —— 那正是
/// `get_runtime_info` 曾经 30 s 超时的来源。探活只需要回答"会话还在不在"，所以
/// 它必须用 `existing_*` 那一类只读入口。
#[test]
fn the_health_snapshot_never_creates_an_editor_session() {
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
    assert!(!document.editor_initialized(), "前提：会话还没创建");
    let snapshot = owner.health_snapshot();
    assert_eq!(snapshot["session_initialized"], serde_json::json!(false));
    assert!(
        !document.editor_initialized(),
        "探活一旦创建会话，启动探活就又会排在模型加载后面"
    );
    document.close();
}
