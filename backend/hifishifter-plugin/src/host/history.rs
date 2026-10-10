//! 撤销历史的**能力**抽象。
//!
//! 【为什么需要 trait】此前"宿主历史"在类型上就是 `Arc<ReaperHost>`（具体类型），
//! 于是"宿主不提供撤销栈"只能表达为 `None`，调用点用 `is_some()` 做 match guard 跳过
//! —— 跳过意味着 `undo_timeline` / `redo_timeline` / `set_history_position` 在
//! Cubase / Logic / Studio One 这类不暴露撤销栈的宿主上**静默 no-op**，用户按 Ctrl+Z
//! 什么都不发生，也没有任何提示。
//!
//! 把"没有宿主历史"变成一个**实现**（[`LocalHistory`]）而不是一条被跳过的分支，
//! 三个命令就都有了确定行为：宿主给栈就用宿主的（保持与宿主 Ctrl+Z 一致），
//! 不给就用插件自管的那份（[`crate::editor::session::EditorSession`] 的
//! `TimelineHistory`，参数/几何检查点一直都在记）。
//!
//! 【为什么不把 jump 也做成纯同步 trait 方法】REAPER 的撤销必须**等在途写入落定**
//! 才能跳（否则跳到一半的编辑状态）。那条路径要持有 document / view / 异步 barrier，
//! 不是一次函数调用能表达的。所以 trait 负责**判别式与快照**，跳转按 `backend()`
//! 分流：`"reaper"` 走既有的异步路径，其余走本地队列（本地命令天然排在已入队写入
//! 之后，不需要额外的 barrier）。

use super::reaper::ReaperHost;
use std::sync::{Arc, Weak};

/// 宿主项目历史的只读快照 + 判别式。
pub(crate) trait HostHistory: Send + Sync {
    /// 判别式：`"reaper"` = 宿主权威栈；`"local"` = 插件自管栈。
    ///
    /// 【为什么要这个字符串而不是 bool】它是**跨进程契约**的一部分：前端按它决定
    /// `label` 是 op key（要查 `history_op_*`）还是宿主自己的可读字符串（原样显示）。
    /// 见 `UndoHistoryPanel::labelOf`。
    fn backend(&self) -> &'static str;

    /// 读取完整的历史载荷（`position` / 深度 / 记录）。
    fn snapshot(&self, authorized: &dyn Fn() -> bool) -> Result<serde_json::Value, String>;
}

/// REAPER 原生撤销栈。
pub(crate) struct ReaperHistory {
    pub host: Arc<ReaperHost>,
}

impl HostHistory for ReaperHistory {
    fn backend(&self) -> &'static str {
        "reaper"
    }

    fn snapshot(&self, authorized: &dyn Fn() -> bool) -> Result<serde_json::Value, String> {
        // `project_history` 要 `&impl Fn`，而 trait 方法是对象安全的（只收 `&dyn Fn`）——
        // 就地包一层具体闭包把它接过去。
        let authorized = || authorized();
        self.host.project_history(&authorized)
    }
}

/// 插件自管的撤销栈（宿主不提供撤销栈时的兜底）。
///
/// 【为什么持 `Weak`】会话与文档同寿，而权威对象由 owner 按需构造 —— 强引用会形成
/// `owner → authority → editor → document → owner` 的环。
pub(crate) struct LocalHistory {
    pub editor: Weak<crate::editor::session::EditorSession>,
}

impl HostHistory for LocalHistory {
    fn backend(&self) -> &'static str {
        "local"
    }

    fn snapshot(&self, _authorized: &dyn Fn() -> bool) -> Result<serde_json::Value, String> {
        let editor = self
            .editor
            .upgrade()
            .ok_or("editor session closed before history snapshot")?;
        Ok(crate::editor::commands::history_state(&editor))
    }
}

/// 连编辑会话都还没有时的空历史（只读、深度 0）。
///
/// 【为什么不是 `None`】调用点不应该因为"此刻还没有会话"而跳过命令 —— 跳过就是
/// 静默失效。给一个诚实的空栈，界面显示"无历史"，用户看到的是事实。
pub(crate) struct NullHistory;

impl HostHistory for NullHistory {
    fn backend(&self) -> &'static str {
        "local"
    }

    fn snapshot(&self, _authorized: &dyn Fn() -> bool) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({
            "ok": true,
            "backend": "local",
            "position": 0,
            "undoDepth": 0,
            "redoDepth": 0,
            "records": [{"label": null, "atMs": null}],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 没有宿主撤销栈时，权威必须是**本地**实现而不是 `None`。
    ///
    /// 【为什么这是回归测试】此前 `get_history_state` / `undo_timeline` / `redo_timeline`
    /// / `set_history_position` 都用 `project_history_host().is_some()` 做 match guard ——
    /// 宿主不提供撤销栈时这几条命令落到通用分支被静默丢弃，用户按 Ctrl+Z 毫无反应。
    /// 现在"没有宿主撤销栈"是 `LocalHistory` 这个实现，快照带 `backend: "local"`，
    /// 前端据此按 op key 本地化。
    ///
    /// 【为什么必须 close 文档】`fixture()` 会把 region 身份登记进**进程级**的
    /// `region_owners()`，并起一个 actor 线程。不关掉它，后面那些重活（真实 ONNX
    /// 渲染、3 分钟素材）会与这个残留会话争资源 —— 表现为无关的 UI 命令 5 秒超时。
    #[test]
    fn the_local_authority_answers_when_the_host_has_no_undo_stack() {
        let (model, owner, identity) = crate::editor::session::tests::fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let authority = owner.history_authority();
        assert_eq!(
            authority.backend(),
            "local",
            "无宿主撤销栈时必须退到本地权威"
        );

        let snapshot = authority.snapshot(&|| true).unwrap();
        assert_eq!(snapshot["backend"], "local");
        assert!(snapshot["records"].is_array(), "{snapshot}");
        assert_eq!(snapshot["position"], 0);
        assert_eq!(snapshot["ok"], true);

        editor.close();
        document.close();
        drop(identity);
    }

    /// 连会话都还没有时给诚实的空栈，而不是 `None`（`None` 会被调用点当成"跳过"）。
    #[test]
    fn the_null_authority_reports_an_honest_empty_stack() {
        let authority = NullHistory;
        assert_eq!(authority.backend(), "local");
        let snapshot = authority.snapshot(&|| true).unwrap();
        assert_eq!(snapshot["undoDepth"], 0);
        assert_eq!(snapshot["redoDepth"], 0);
        assert_eq!(snapshot["backend"], "local");
    }
}
