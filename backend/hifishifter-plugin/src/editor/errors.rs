//! 编辑会话错误的**单一事实来源**：措辞常量 + 协议码。
//!
//! 【为什么常量要共享】终态错误的**产生点**（`session` / `routing` /
//! `render::extension` / `render::document`）与**分类点**（[`classify_error`]）
//! 分处不同模块。此前两边各写字面量、靠子串匹配对齐 —— 任何一处改措辞都会把
//! 终态判成 transient（或反之），且无编译期信号（"19,056 次轮询"那类缺陷的
//! 温床）。现在产生点引用这里的常量，分类点匹配同一常量：改措辞只改一处。
//!
//! 协议码随回执下发（回执 JSON 的 `errorCode` / `hifishifter-ara-ipc`
//! `Response.code`），客户端按码分类；字符串只做展示。**码的取值是协议承诺：
//! 只增不改**。分类判定看码，码不存在（旧产生点/旧服务端）才退回字符串匹配。

use hifishifter_ara_ipc::error_code;

/// 一条错误的稳定措辞与协议码。
pub(crate) struct TerminalError {
    /// 协议码（只增不改）。
    pub code: u32,
    /// 稳定措辞；产生点经 [`TerminalError::error`] 引用，分类点按它匹配。
    pub message: &'static str,
}

impl TerminalError {
    /// 产生点用它构造 `Err(String)` —— 与既有 `Result<_, String>` 链兼容。
    pub fn error(&self) -> String {
        self.message.to_string()
    }
}

/// 编辑文档已关闭（`alive` 原子量一旦 false 永不再活）。
pub(crate) const DOCUMENT_CLOSED: TerminalError = TerminalError {
    code: error_code::DOCUMENT_CLOSED,
    message: "document closed",
};
/// FX 处理器（组件）已销毁。
pub(crate) const FX_PROCESSOR_CLOSED: TerminalError = TerminalError {
    code: error_code::FX_PROCESSOR_CLOSED,
    message: "FX processor closed",
};
/// 编辑器视图不存在或身份未知。
pub(crate) const VIEW_CLOSED_OR_UNKNOWN: TerminalError = TerminalError {
    code: error_code::VIEW_CLOSED_OR_UNKNOWN,
    message: "editor view closed or unknown",
};
/// 视图身份与会话不匹配。
pub(crate) const VIEW_IDENTITY_MISMATCH: TerminalError = TerminalError {
    code: error_code::VIEW_IDENTITY_MISMATCH,
    message: "editor view identity mismatch",
};
/// 请求排队期间编辑路由已改变。
pub(crate) const ROUTE_CHANGED_WHILE_QUEUED: TerminalError = TerminalError {
    code: error_code::ROUTE_CHANGED_WHILE_QUEUED,
    message: "editor route changed while request queued",
};
/// 编辑路由属于另一份文档。
pub(crate) const ROUTE_OTHER_DOCUMENT: TerminalError = TerminalError {
    code: error_code::ROUTE_OTHER_DOCUMENT,
    message: "editor route belongs to another document",
};
/// 编辑租约已被撤销。
pub(crate) const LEASE_REVOKED: TerminalError = TerminalError {
    code: error_code::LEASE_REVOKED,
    message: "editor lease revoked",
};
/// 插件实例已关闭。
pub(crate) const INSTANCE_CLOSED: TerminalError = TerminalError {
    code: error_code::INSTANCE_CLOSED,
    message: "plugin instance closed",
};
/// 宿主从未把本实例绑定到 ARA 文档。
///
/// 【为什么必须与 `DOCUMENT_CLOSED` 分开】两者此前共用一句 "document closed"，
/// 于是"宿主根本没走 ARA 路径"（Cubase 实测：从不查询 `IPlugInEntryPoint`、从不
/// 创建文档控制器、`setActive` 时仍未绑定）被表述成"会话已结束"。前端据此提示
/// "关闭并重新打开插件窗口"—— 而重开会走完全相同的路径、得到完全相同的错误。
/// 那条提示把用户引向一个**必然失败**的补救动作。
///
/// 【为什么它是终态】ARA 规定 `bindToDocumentControllerWithRoles` 只能在
/// `setActive` / `setState` / GUI 创建之前调用一次（`ARAVST3.h` 的
/// `IPlugInEntryPoint2` 契约）；宿主既然已经 `setActive`，就不会再给这个组件机会。
/// 对这个编辑器窗口而言，确实永久。
///
/// 【措辞红线】不得包含 "closed" —— [`classify_error`] 用**子串**匹配（见其文档），
/// 一旦新措辞里出现该词，它会被 `DOCUMENT_CLOSED` 抢先命中，两种终态重新混为一谈。
pub(crate) const ARA_NOT_BOUND: TerminalError = TerminalError {
    code: error_code::ARA_NOT_BOUND,
    message: "plug-in instance is not bound to an ARA document",
};

/// panic 毒化后的编辑权威错误。
///
/// 【为什么它是终态】编辑权威处于未知状态时，重放同一次编辑只会重复失败 ——
/// 必须由用户显式刷新（或关闭重开窗口）才能重新建立权威。这与"宿主状态变了一次、
/// 重放就会收敛"的暂时冲突**性质不同**，所以单列出来，并让
/// [`is_retryable_host_conflict`] 把它排除在可重试之外。
pub(crate) const EDIT_AUTHORITY_POISONED: &str = "Conflict: edit authority is in an unknown state";

/// 终态表：分类点遍历它做子串匹配（措辞本身来自上面这些常量）。
const TERMINAL: &[&str] = &[
    DOCUMENT_CLOSED.message,
    FX_PROCESSOR_CLOSED.message,
    VIEW_CLOSED_OR_UNKNOWN.message,
    VIEW_IDENTITY_MISMATCH.message,
    ROUTE_CHANGED_WHILE_QUEUED.message,
    ROUTE_OTHER_DOCUMENT.message,
    LEASE_REVOKED.message,
    INSTANCE_CLOSED.message,
    ARA_NOT_BOUND.message,
    EDIT_AUTHORITY_POISONED,
];

/// "宿主模型冲突"前缀：可重试（宿主把模型带到了真实状态后会收敛）。
pub(crate) const CONFLICT_PREFIX: &str = "Conflict: host";

/// 这一条错误是不是"宿主状态在我们脚下变了一次"造成的**暂时**冲突。
///
/// 【为什么不能只认 `CONFLICT_PREFIX`】`Conflict:` 家族里除了 `Conflict: host …`
/// 还有若干同源措辞：工作区 scope 变了、编辑代次变了、分配变了、没有活动 renderer。
/// 它们同样是"重放一次就会收敛"的暂时状态，但此前只有 `Conflict: host` 被判为可
/// 重试，其余会被写进 `EditorSession::error` 并**永久**挡住自动渲染 —— 用户必须
/// 手动刷新。这与"宿主里做了操作，插件毫无反应"是同一个观感。
///
/// 【为什么排除毒化】见 [`EDIT_AUTHORITY_POISONED`]：那是真终态，重放只会重复失败。
pub(crate) fn is_retryable_host_conflict(error: &str) -> bool {
    error.starts_with("Conflict:") && !error.contains(EDIT_AUTHORITY_POISONED)
}

/// 宿主 PCM 尚未就绪（可重试的暂时失败源）。
pub(crate) const HOST_PCM_UNAVAILABLE: &str = "host PCM unavailable";
/// 宿主模型尚未就绪（可重试的暂时失败源）。
pub(crate) const HOST_MODEL_NOT_READY: &str = "host model not ready";

/// 会话错误的**机器可读分类**：`terminal`（永久停止轮询）/ `transient`（可重试）。
///
/// 【为什么用"包含"而不是全等】措辞可能带上下文前缀（如 `host {LEASE_REVOKED}`
/// 变成 "host editor lease revoked"），常量保证的是**子串出现**而非整串相等。
pub(crate) fn classify_error(error: &str) -> &'static str {
    if TERMINAL.iter().any(|needle| error.contains(needle)) {
        "terminal"
    } else {
        "transient"
    }
}

/// 错误的协议码；非终态或未知措辞返回 `None`（客户端退回字符串分类）。
pub(crate) fn terminal_error_code(error: &str) -> Option<u32> {
    const TABLE: &[(&str, u32)] = &[
        (DOCUMENT_CLOSED.message, DOCUMENT_CLOSED.code),
        (FX_PROCESSOR_CLOSED.message, FX_PROCESSOR_CLOSED.code),
        (VIEW_CLOSED_OR_UNKNOWN.message, VIEW_CLOSED_OR_UNKNOWN.code),
        (VIEW_IDENTITY_MISMATCH.message, VIEW_IDENTITY_MISMATCH.code),
        (
            ROUTE_CHANGED_WHILE_QUEUED.message,
            ROUTE_CHANGED_WHILE_QUEUED.code,
        ),
        (ROUTE_OTHER_DOCUMENT.message, ROUTE_OTHER_DOCUMENT.code),
        (LEASE_REVOKED.message, LEASE_REVOKED.code),
        (INSTANCE_CLOSED.message, INSTANCE_CLOSED.code),
        (ARA_NOT_BOUND.message, ARA_NOT_BOUND.code),
    ];
    TABLE
        .iter()
        .find(|(needle, _)| error.contains(needle))
        .map(|(_, code)| *code)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 措辞与码一一对应，且码只增不改 —— 协议承诺。
    #[test]
    fn terminal_codes_are_stable_and_unique() {
        let table = [
            (DOCUMENT_CLOSED, 1),
            (FX_PROCESSOR_CLOSED, 2),
            (VIEW_CLOSED_OR_UNKNOWN, 3),
            (VIEW_IDENTITY_MISMATCH, 4),
            (ROUTE_CHANGED_WHILE_QUEUED, 5),
            (ROUTE_OTHER_DOCUMENT, 6),
            (LEASE_REVOKED, 7),
            (INSTANCE_CLOSED, 8),
            (ARA_NOT_BOUND, 9),
        ];
        for (entry, code) in table {
            assert_eq!(
                entry.code, code,
                "{} must keep its protocol code",
                entry.message
            );
        }
    }

    /// 带上下文前缀的产生点措辞仍能按常量命中（"包含"而非全等）。
    #[test]
    fn prefixed_producers_still_classify_terminal() {
        assert_eq!(
            classify_error(&format!("host split: {}", LEASE_REVOKED.message)),
            "terminal"
        );
        assert_eq!(
            terminal_error_code(&format!("host split: {}", LEASE_REVOKED.message)),
            Some(LEASE_REVOKED.code)
        );
    }

    /// **"从未绑定"必须与"已关闭"分得开。**
    ///
    /// 【为什么这是回归测试】`classify_error` 与 `terminal_error_code` 都用**子串**
    /// 匹配，且 `DOCUMENT_CLOSED` 在表里排在前面。新措辞一旦包含 "closed"，它会被
    /// `DOCUMENT_CLOSED` 抢先命中 —— 两种终态重新合并，前端又会对"宿主从未绑定 ARA"
    /// 给出"请关闭并重新打开插件窗口"这条**必然失败**的指引。
    #[test]
    fn a_never_bound_instance_is_not_reported_as_closed() {
        assert!(
            !ARA_NOT_BOUND.message.contains("closed"),
            "措辞含 'closed' 会被 DOCUMENT_CLOSED 抢先命中：{}",
            ARA_NOT_BOUND.message
        );
        assert_eq!(classify_error(ARA_NOT_BOUND.message), "terminal");
        assert_eq!(
            terminal_error_code(ARA_NOT_BOUND.message),
            Some(ARA_NOT_BOUND.code)
        );
        // 反向：真正的关闭仍归 DOCUMENT_CLOSED，不能被新常量抢走。
        assert_eq!(
            terminal_error_code(DOCUMENT_CLOSED.message),
            Some(DOCUMENT_CLOSED.code)
        );
    }
}
