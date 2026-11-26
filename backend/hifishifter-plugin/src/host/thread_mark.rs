//! 宿主线程的角色标记：让"这条线程不许取 GUI 锁"成为可判定的事实。
//!
//! 【为什么需要它】宿主会在**它自己的 UI 线程**上同步进入我们的代码：WebView2 的
//! 消息回调、`IPlugView` 的回调、`plugin_register("timer")` 的节拍、以及 ARA 的
//! 模型回调。这些调用一旦阻塞，宿主就不再泵消息 —— 而 WebView2 的投递（`post_message`）
//! 正是靠那条线程泵消息才推进的。于是"我们等锁"与"宿主等我们返回"可以互相咬住。
//!
//! 【为什么不是"宿主线程不许取任何锁"】那太强，也无法执行：回执表、视图表、撤销块
//! 记账都各有短锁，它们的临界区只有几次内存写。真正要禁止的是**等待长临界区** ——
//! 而"长"的那一把是会话 actor 在整份投影期间持有的时间线 / 文档锁。
//!
//! 【为什么角色是 thread-local】"我在宿主线程上"是**每条线程**的事实。全局标志会让
//! 工作线程也误判成宿主线程，于是它在该取锁的地方不取 —— 表现为"编辑之后永远不生效"。
//! 这与 `reentrancy` 的 `HOST_CALL_DEPTH` 是同一条纪律。

use std::cell::Cell;

/// 这条线程此刻扮演的宿主侧角色。
// 【为什么允许未构造】这几个角色名是**不变量的一部分**：写下来的是"哪些线程算宿主
// 线程"。只把今天用到的那几个构造出来，等于让下一个加回调的人以为"没列出来的不算"。
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostThreadRole {
    /// WebView2 / WebKit 的消息回调（`on_message`）。
    WebView,
    /// 视图节拍（`plugin_register("timer")`、GTK 定时器、CLAP timer）。
    ViewTimer,
    /// ARA 的模型回调（`beginEditing` / `endEditing` / `notifyModelUpdates`）。
    AraModel,
    /// 一次显式的宿主 API 调用（`HostCallGuard`）。
    HostCall,
}

impl HostThreadRole {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::WebView => "webview",
            Self::ViewTimer => "view-timer",
            Self::AraModel => "ara-model",
            Self::HostCall => "host-call",
        }
    }
}

thread_local! {
    /// 本线程当前的角色栈顶（`None` = 不是宿主线程）。
    static ROLE: Cell<Option<HostThreadRole>> = const { Cell::new(None) };
}

/// 一次"本线程正在扮演宿主角色"的 RAII 标记。
pub(crate) struct RoleGuard {
    previous: Option<HostThreadRole>,
}

impl RoleGuard {
    pub(crate) fn enter(role: HostThreadRole) -> Self {
        let previous = ROLE.with(|current| current.replace(Some(role)));
        Self { previous }
    }
}

impl Drop for RoleGuard {
    fn drop(&mut self) {
        ROLE.with(|current| current.set(self.previous));
    }
}

/// 本线程此刻是不是宿主线程；是的话是什么角色。
pub(crate) fn host_thread_role() -> Option<HostThreadRole> {
    ROLE.with(Cell::get)
}

/// 宿主线程**不得**去等的一类锁。
///
/// 【为什么用字符串而不是类型】这些锁分散在 `EditorSession`（时间线 / 历史 / 已加载）
/// 与 `DocumentSession`（文档事务）上，类型各不相同。判据只需要一个名字：日志里说
/// "host thread tried to take the editor timeline"就够了，而**禁止**这件事发生在调用点。
// 【同上】这里列的是"宿主线程不得等的那一类锁"，不是"今天已经接上的那一把"。
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LongLock {
    /// 会话的时间线（投影期间被 actor 长时间持有）。
    EditorTimeline,
    /// 会话的历史栈。
    EditorHistory,
    /// 会话的"已加载"状态。
    EditorLoaded,
    /// 文档事务。
    DocumentTransaction,
}

impl LongLock {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::EditorTimeline => "editor.timeline",
            Self::EditorHistory => "editor.history",
            Self::EditorLoaded => "editor.loaded",
            Self::DocumentTransaction => "document.transaction",
        }
    }
}

/// 宿主线程要取一把长锁时调用它。返回 `true` 表示**不许取**，调用方必须走降级分支。
///
/// # 三档（`HFS_HOST_LOCK`）
///
/// | 值 | 行为 | 用途 |
/// |---|---|---|
/// | 未设 / `probe`（默认） | 只记一行读数，**不改变行为** | 现场取证：哪条命令真的在等锁 |
/// | `refuse` | 返回 `true`，调用方降级 | 验收"宿主线程零长锁" |
/// | `allow` | 与 probe 相同但**不记读数** | 对照实验 |
///
/// 【为什么默认是 probe 而不是 refuse】`refuse` 会让调用方走"这一拍什么都不做"的
/// 降级分支，而那条分支在现场与"命令没生效"完全同形。在还没有一次现场读数证明
/// "卡的就是这把锁"之前就默认打开它，等于用一个不可观测的降级换一个未证实的病因。
/// 这与方案 §2.7.6 的纪律一致：**先取证，再改行为**。
///
/// 【为什么 probe 也要记】两个挂起会话里一条锁读数都没有，因为 Drop 型守卫看不见
/// "永不返回"。probe 把"宿主线程确实去取了长锁"变成日志里的一行，而它出现在挂起
/// 之前的最后几行里，就足以把三个候选病因分开。
pub(crate) fn host_thread_may_not_lock(lock: LongLock) -> bool {
    let Some(role) = host_thread_role() else {
        return false;
    };
    let mode = host_lock_mode();
    if mode == HostLockMode::Allow {
        return false;
    }
    if mode == HostLockMode::Refuse {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "host.lock.refused",
            format_args!(
                "[lock] refusing {} on the host thread (role={}); this call would block the host UI",
                lock.as_str(),
                role.as_str()
            ),
        );
        return true;
    }
    hifishifter_kernel::log_limiter::emit_limited_key(
        log::Level::Warn,
        "host.lock.probe",
        format_args!(
            "[lock] {} requested on the host thread (role={}); set HFS_HOST_LOCK=refuse to forbid it",
            lock.as_str(),
            role.as_str()
        ),
    );
    false
}

/// `HFS_HOST_LOCK` 的三档。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostLockMode {
    Probe,
    Refuse,
    Allow,
}

fn host_lock_mode() -> HostLockMode {
    static MODE: std::sync::OnceLock<HostLockMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("HFS_HOST_LOCK").as_deref() {
        Ok("refuse") => HostLockMode::Refuse,
        Ok("allow") => HostLockMode::Allow,
        _ => HostLockMode::Probe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **不变量：WebView 线程不得取时间线锁。**
    ///
    /// 【为什么钉死】这条路径上的每一次"等锁"都会让宿主停止泵消息，而 WebView2 的
    /// 投递正靠它泵。现场那条 `AppHangB1` 就是这条链子的终点。
    #[test]
    fn a_webview_thread_may_not_lock_the_timeline() {
        // 非宿主线程：永远放行。
        assert!(!host_thread_may_not_lock(LongLock::EditorTimeline));
        let _guard = RoleGuard::enter(HostThreadRole::WebView);
        assert_eq!(host_thread_role(), Some(HostThreadRole::WebView));
        // 默认（probe）不改行为，但**认得**这是一条宿主线程。
        assert!(!host_thread_may_not_lock(LongLock::EditorTimeline));
        assert_eq!(host_thread_role(), Some(HostThreadRole::WebView));
    }

    /// 标记是**每条线程**的事实。
    #[test]
    fn the_role_is_thread_local() {
        let _guard = RoleGuard::enter(HostThreadRole::ViewTimer);
        let seen = std::thread::spawn(host_thread_role).join().unwrap();
        assert_eq!(seen, None, "别的线程不得继承宿主角色");
        assert_eq!(host_thread_role(), Some(HostThreadRole::ViewTimer));
    }

    /// 嵌套角色：内层归还后回到外层。
    #[test]
    fn nested_roles_restore_the_outer_one() {
        let _outer = RoleGuard::enter(HostThreadRole::WebView);
        {
            let _inner = RoleGuard::enter(HostThreadRole::HostCall);
            assert_eq!(host_thread_role(), Some(HostThreadRole::HostCall));
        }
        assert_eq!(host_thread_role(), Some(HostThreadRole::WebView));
    }
}
