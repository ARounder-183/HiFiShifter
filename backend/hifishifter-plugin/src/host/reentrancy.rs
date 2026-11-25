//! 宿主调用的**重入守卫**与看门狗。
//!
//! 【为什么需要它】宿主 API 会在**同一条线程上同步**回调插件：REAPER 的
//! `UpdateItem` / `UpdateArrange` 立刻触发 ARA 的 `beginEditing` / `endEditing`。
//! 于是"一次编辑"在那条线程上是**嵌套**的 —— 宿主调用里跑着我们的 ARA 回调，回调里
//! 又可能回到宿主。没有这一层记账时，回调无法判断自己正处在一次宿主调用里，于是它
//! 会走那条"准备一次完整合成并等 worker"的重路径。
//!
//! 现场后果（2026-11-25，REAPER 7.8.5）：`文件 → 导入媒体` 之后主线程在
//! `endEditing` 里等 `wait_idle_until(2 s)`，Windows 记录 `AppHangB1`
//! ——"stopped interacting with Windows and was closed"。这不是崩溃，是**主线程不泵
//! 消息**。上界（2 s）不是零：只要宿主在一个必须立刻返回的回调里等，它就已经在挂起
//! 的路上了。
//!
//! 【为什么是线程局部的深度计数而不是全局标志】"在宿主调用里"是**每条线程**的事实：
//! 编辑器命令线程把工作投递到宿主线程时，投递方自己并没有在宿主调用里。全局标志会
//! 让投递方也误判成"在重入"，于是它在该等的地方不等 —— 首快照就永远发不出去。
//!
//! 【为什么守卫是 RAII 而不是成对的 enter/leave】宿主调用的失败路径极多（`?` 满地
//! 都是），成对调用迟早会漏一个 `leave`，而漏掉的后果是**永久**认为自己在重入：
//! 此后所有准备都只排队、不执行，表现是"编辑之后永远不生效"。

use super::stall_watch::{mark, StallTicket};
use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    /// 本线程当前嵌了几层宿主调用（0 = 不在宿主调用里）。
    static HOST_CALL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// 一次宿主调用超过这么久就记一行 —— 宿主的 UI 线程被占住 2 s，用户已经在感觉"卡"。
const HOST_CALL_WARN: Duration = Duration::from_millis(2000);

/// 一次 ARA 回调超过这么久就记一行。
///
/// 【为什么比宿主调用更严】ARA 回调是宿主**直接**进入我们的代码（不经过命令队列），
/// 它占住的正是宿主主线程；1 s 已经足够让 Windows 把窗口画成"未响应"。
const ARA_CALLBACK_WARN: Duration = Duration::from_millis(1000);

/// 看门狗开关（环境变量 `HFS_HOST_WATCHDOG=0` 关闭）。
///
/// 【为什么要开关】看门狗只在**事后**写一行（它不改行为），但现场排障时"日志里多了
/// 一类行"本身就是需要解释的变化。留一个一键关掉的开关，与 `HFS_PINNED_FEED` 同一
/// 条纪律：新读数可以先关掉再对比。
///
/// 【为什么用 OnceLock】它在音频线程与宿主线程上都会被读到；getenv 只在进程内第一次
/// 发生，之后是一次原子读。
pub(crate) fn watchdog_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("HFS_HOST_WATCHDOG").is_ok_and(|value| value == "0" || value == "false")
    })
}

/// 一次宿主调用的 RAII 守卫。
///
/// 它只做两件事：把"本线程正在宿主调用里"变成可读的事实，以及在这一层退出时量一次
/// 耗时。它**不**禁止重入 —— 宿主本来就会重入，我们要做的是让回调**知道**自己在重入。
pub(crate) struct HostCallGuard {
    kind: &'static str,
    started: Instant,
    /// 【为什么守卫要顺带登记到停滞表里】`Drop` 只看得见"返回了但很慢"，看不见
    /// "永远不返回"。而后者才是宿主失去响应的那一种。登记与归还都在这里，两件事
    /// 共享同一个生命周期 —— 分成两处写必然有一天只改了一处。
    _stall: StallTicket,
}

impl HostCallGuard {
    /// 进入一次宿主调用。`kind` 同时是日志里的名字与看门狗的限流键。
    pub(crate) fn enter(kind: &'static str) -> Self {
        HOST_CALL_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self {
            kind,
            started: Instant::now(),
            _stall: StallTicket::enter(kind),
        }
    }

    /// 本调用现在走到哪个阶段（写进停滞看门狗的表）。
    ///
    /// 【为什么是方法而不是自由函数】读方永远希望"这个阶段属于**这一次**调用"。
    /// 写成方法，调用点就必须先拿到守卫，于是"没有守卫却在写标记"变得不可能。
    #[allow(dead_code)]
    pub(crate) fn mark(&self, stage: &'static str) {
        mark(stage);
    }
}

impl Drop for HostCallGuard {
    fn drop(&mut self) {
        HOST_CALL_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        // 【为什么只在最外层计时】内层守卫的耗时已经包含在外层里；每层都报会让一次
        // 慢调用写成十几行。而"慢"这件事本来就是整条调用链的属性。
        if HOST_CALL_DEPTH.with(Cell::get) > 0 {
            return;
        }
        let elapsed = self.started.elapsed();
        if elapsed < HOST_CALL_WARN || !watchdog_enabled() {
            return;
        }
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "host.call.exceeded",
            format_args!(
                "[host] call {} took {}ms on thread {:?}; the host UI thread was blocked for that long",
                self.kind,
                elapsed.as_millis(),
                std::thread::current().id()
            ),
        );
    }
}

/// 本线程此刻是不是正处在一次宿主调用里。
///
/// 读者只有一个方向：**是 ⇒ 不许等**。任何会阻塞在另一个线程上的动作
/// （`wait_idle_until`、`on_host`、锁序未知的跨线程等待）都必须在这条判据上让路。
pub(crate) fn in_host_call() -> bool {
    HOST_CALL_DEPTH.with(Cell::get) > 0
}

/// 一次 ARA 回调的 RAII 守卫（与 HostCallGuard 同一形状，独立阈值与限流键）。
///
/// 【为什么与宿主调用分开】两者的成因不同：宿主调用慢 = 宿主 API 慢；ARA 回调慢 =
/// **我们**在宿主直接进入的代码里做了重活。混成一个键之后，"到底是我们慢还是宿主慢"
/// 又变成需要读源码才能回答的问题。
pub(crate) struct AraCallbackGuard {
    name: &'static str,
    started: Instant,
    /// 同 `HostCallGuard` 的那一格：ARA 回调才是宿主主线程被我们占住的那条路。
    _stall: StallTicket,
}

impl AraCallbackGuard {
    pub(crate) fn enter(name: &'static str) -> Self {
        Self {
            name,
            started: Instant::now(),
            _stall: StallTicket::enter(name),
        }
    }

    /// 本回调现在走到哪个阶段（写进停滞看门狗的表）。
    #[allow(dead_code)]
    pub(crate) fn mark(&self, stage: &'static str) {
        mark(stage);
    }
}

impl Drop for AraCallbackGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        if elapsed < ARA_CALLBACK_WARN || !watchdog_enabled() {
            return;
        }
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.callback.exceeded",
            format_args!(
                "[ara] callback {} took {}ms on thread {:?}; the host cannot pump while it runs",
                self.name,
                elapsed.as_millis(),
                std::thread::current().id()
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **不变量：守卫内 `in_host_call()` 为真，离开后为假。**
    #[test]
    fn a_guard_marks_the_thread_and_clears_on_drop() {
        assert!(!in_host_call());
        {
            let _guard = HostCallGuard::enter("test");
            assert!(in_host_call());
            {
                let _nested = HostCallGuard::enter("nested");
                assert!(in_host_call());
            }
            assert!(in_host_call(), "内层退出不得清掉外层的标记");
        }
        assert!(!in_host_call());
    }

    /// **不变量：panic 展开也会把标记清掉。**
    ///
    /// 【为什么钉死】漏掉这一次复位，本线程此后**永远**认为自己在宿主调用里 ——
    /// 所有准备都只排队不执行，用户看到的是"编辑之后永远不生效"，而日志里没有一行
    /// 异常。RAII 的全部意义就是这条。
    #[test]
    fn a_panicking_host_call_still_clears_the_marker() {
        let outcome = std::panic::catch_unwind(|| {
            let _guard = HostCallGuard::enter("test");
            panic!("boom");
        });
        assert!(outcome.is_err());
        assert!(!in_host_call());
    }

    /// 标记是**每条线程**的事实：另一条线程不受影响。
    #[test]
    fn the_marker_is_thread_local() {
        let _guard = HostCallGuard::enter("test");
        let seen = std::thread::spawn(in_host_call).join().unwrap();
        assert!(!seen, "别的线程不得继承宿主调用标记");
        assert!(in_host_call());
    }

    /// ARA 回调守卫不改变宿主调用标记（两者是不同的事实）。
    #[test]
    fn an_ara_callback_guard_does_not_mark_a_host_call() {
        let _callback = AraCallbackGuard::enter("test");
        assert!(!in_host_call());
    }
}
