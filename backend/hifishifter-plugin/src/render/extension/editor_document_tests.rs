//! `editor_document` 的终态措辞回归。
//!
//! 【为什么单独一组】用户报障：Cubase 里插件窗口能打开，但每一个命令都回
//! "document closed"，界面据此提示"本次编辑会话已结束。请关闭并重新打开插件窗口
//! 后继续"。日志显示 Cubase **从未查询** `IPlugInEntryPoint`、从未创建文档控制器、
//! `setActive` 时仍未绑定 —— 也就是"从未绑定"，而不是"已关闭"。用户照提示重开窗口
//! 只会走完全相同的路径、拿到完全相同的错误。
//!
//! 这两条用例把"从未绑定"与"绑定过又关闭"钉成两个可区分的读数。

use super::*;

/// 从未绑定过 ⇒ `ARA_NOT_BOUND`（"宿主尚未把本插件接入 ARA"）。
#[test]
fn an_instance_that_was_never_bound_says_so() {
    let owner = ExtensionOwner::default();
    // `Arc<DocumentSession>` 没有 `Debug`，因此取 `err()` 而不是 `expect_err()`。
    let error = owner.editor_document().err().expect("未绑定必须报错");
    assert_eq!(
        error,
        crate::editor::errors::ARA_NOT_BOUND.error(),
        "从未绑定不得被表述成「会话已结束」"
    );
    assert_eq!(
        crate::editor::errors::classify_error(&error),
        "terminal",
        "它是终态：宿主不会再给这个组件机会"
    );
    assert_eq!(
        crate::editor::errors::terminal_error_code(&error),
        Some(crate::editor::errors::ARA_NOT_BOUND.code)
    );
}

/// 绑定过、随后文档关闭 ⇒ 仍是"会话已结束"类终态，**不得**退化成 `ARA_NOT_BOUND`。
///
/// 【为什么必须钉住反向】只测"从未绑定"会让一个把两种情形都报成 `ARA_NOT_BOUND`
/// 的实现通过 —— 那对真正需要关窗重开的用户是错的（重开确实能修好）。
///
/// 【为什么不断言具体是哪一条终态】文档关闭会连带 `stop_editor()`（宿主拆除插件的
/// 真实顺序），于是先命中 `FX_PROCESSOR_CLOSED`；只有"文档单独销毁、owner 未被关闭"
/// 才会走到 `DOCUMENT_CLOSED`。两者都是"会话已结束"，对用户是同一句指引 ——
/// 这里钉的是**不该是哪一个**，而不是恰好是哪一个。
#[test]
fn a_bound_then_closed_instance_is_not_reported_as_never_bound() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    assert!(owner.editor_document().is_ok(), "夹具必须已绑定");
    document.close();
    let error = owner.editor_document().err().expect("关闭后必须报错");
    assert_ne!(
        error,
        crate::editor::errors::ARA_NOT_BOUND.error(),
        "绑定过又关闭是真的关闭，措辞不得退化成「从未绑定」"
    );
    assert_eq!(
        crate::editor::errors::classify_error(&error),
        "terminal",
        "关闭后的错误必须是终态，否则前端会继续重试"
    );
}

/// `document_closed()` 不得清除 `bound_once`。
///
/// 【为什么这是回归测试】宿主在拆除插件时会调 `document_closed()`。如果它顺手把
/// `bound_once` 也清了，之后任何一次查询都会变成 `ARA_NOT_BOUND` —— 一条**假的**
/// 诊断（"宿主从未绑定"），而宿主明明绑定过。`editor_document()` 的措辞靠这个单调
/// 标志分流，因此它必须是单调的。
#[test]
fn document_closed_keeps_the_never_bound_flag_monotonic() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    owner.document_closed();
    let error = owner.editor_document().err().expect("关闭后必须报错");
    assert_ne!(
        error,
        crate::editor::errors::ARA_NOT_BOUND.error(),
        "`document_closed()` 之后仍是「已关闭」，不能变成「从未绑定」"
    );
    assert_eq!(crate::editor::errors::classify_error(&error), "terminal");
    document.close();
}

/// 造一个最小的 UI 请求（只用于断言"是否入队"，不会真的执行）。
fn request(id: u64) -> crate::editor::session::UiRequest {
    let (reply, _rx) = std::sync::mpsc::channel();
    let (events, _ev) = std::sync::mpsc::sync_channel(1);
    crate::editor::session::UiRequest {
        id,
        command: "get_timeline_state".into(),
        args: serde_json::json!({}),
        sink: crate::editor::session::UiSink {
            view_id: "terminal-guard".into(),
            reply,
            events,
            closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        },
        link: None,
    }
}

/// **不变量：终态会话不得把请求排进队列。**
///
/// 【为什么这是回归测试】前端闸门（`services/sessionGate.ts`）省掉了 IPC，但后端仍要
/// 自己拒绝：将来任何一个新接入的调用点、或绕过 `invoke()` 的直连路径，都不该把请求
/// 排进一个已经终结的会话 —— 那会占住 32 任务预算、唤醒 actor，并把同一个终态错误
/// 再答一遍（用户报障的"同一错误答 19,056 次"正是这样形成的）。
///
/// 【为什么在 `editor_session()` 层面断言】`webview` 的命令分发是
/// `link.owner().and_then(|owner| owner.editor_session()?.enqueue(..))` —— 终态在这里
/// **先**被 `editor_session()` 挡下，`enqueue` 根本不会被调用。所以真正的守卫点是
/// `editor_session()`，断言它必须落在这一层。
#[test]
fn a_terminal_session_never_reaches_the_queue() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    let editor = owner.editor_session().expect("夹具必须已绑定");

    // 未关闭时入队成功（否则下面的断言对缺陷不敏感）。
    editor.enqueue(request(1)).expect("活会话必须接受入队");

    document.close();

    // 关闭后：`editor_session()` 必须直接失败 —— 分发路径在构造 `UiRequest` 之前就停住。
    let error = owner
        .editor_session()
        .err()
        .expect("终态会话必须拒绝新的编辑会话入口");
    assert_ne!(
        error,
        crate::editor::errors::ARA_NOT_BOUND.error(),
        "绑定过又关闭的实例必须报「会话已结束」，而不是「从未绑定」"
    );
    assert_eq!(crate::editor::errors::classify_error(&error), "terminal");

    // 即便有人绕过上一步直接入队，也必须被拒（`closed` / 文档存活双重守卫）。
    let direct = editor.enqueue(request(2));
    assert!(
        direct.is_err(),
        "终态会话的 enqueue 不得静默成功：{direct:?}"
    );
    assert_eq!(
        crate::editor::errors::classify_error(&direct.err().unwrap()),
        "terminal",
        "拒绝入队的错误必须被分类为终态，否则前端会继续重试"
    );
}
