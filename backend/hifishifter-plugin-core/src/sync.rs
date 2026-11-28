//! 锁毒化恢复：把"某线程 panic 过一次"从**永久挂死**降级为**继续运行**。
//!
//! 【为什么必须有这一层】本插件跑在宿主进程里，任何一处 `Mutex::lock().unwrap()`
//! 在某线程 panic 之后都会**永久毒化**那把锁：之后每一个 `.lock_or_recover()` 都在
//! 宿主里 panic。`catch_unwind` 边界（音频回调、`window_proc`、IPC）能挡住单次 panic
//! 越过 FFI 边界，却挡不住毒化 —— 表现是"插件 UI 挂死"，比崩溃更难诊断，也更难让
//! 用户恢复。
//!
//! 【为什么恢复而不是继续 panic】宿主是 DAW：宁可让某一帧/某次读取拿到可能不完整的
//! 数据并继续出声，也不要让整个宿主界面卡住。这里恢复的是"锁的可用性"，不是"数据
//! 一定正确" —— 真正的正确性由各调用点在拿到 guard 后重新校验版本来保证（渲染路径
//! 已有 `PreparedVersion` 这层校验）。
//!
//! 【为什么不逐个改 `unwrap_or_else(|e| e.into_inner())`】那是同一件事的手写形式，
//! 661 处手写必然漏改、也读不出意图。统一成一个方法名之后，"哪些锁允许毒化恢复"
//! 一眼可查。

use std::sync::{Mutex, MutexGuard};

/// `Mutex::lock` 的毒化恢复版本。
pub trait MutexExt<T> {
    /// 取锁；锁已被毒化时**恢复**内部数据而不是 panic。
    fn lock_or_recover(&self) -> MutexGuard<'_, T>;

    /// 取锁并**报告**是否从毒化中恢复（`true` = 数据可能半写）。
    ///
    /// 【为什么需要"报告"这一版】`lock_or_recover` 恢复的是"锁的可用性"，不是"数据一定
    /// 正确"（见模块文档）。纯读路径可以照常跑；但**读改写并影响宿主**的路径（把编辑器
    /// 时间线写回 DAW）不能拿一份半写状态去改宿主 —— 那会把一次局部 panic 升级成工程级
    /// 损坏。这类调用点在返回 `true` 时必须**放弃本次操作**并留痕，而不是继续。
    fn lock_or_recover_checked(&self) -> (MutexGuard<'_, T>, bool);
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_or_recover(&self) -> MutexGuard<'_, T> {
        self.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn lock_or_recover_checked(&self) -> (MutexGuard<'_, T>, bool) {
        match self.lock() {
            Ok(guard) => (guard, false),
            Err(poisoned) => (poisoned.into_inner(), true),
        }
    }
}

/// 取锁并记录"本线程此刻持有它"；同一线程**再次**取同一把锁时点名 panic。
///
/// 【为什么需要它】自锁的现场形状是"进程安静地冻住、日志里一个字都没有"——它和
/// "什么都没发生"完全同形，事后只能靠通读代码找那一处嵌套。探针把它变成一句话：
/// **哪一行先拿的、哪一行又拿了一次**。本仓已经踩过两次（`present_host_inventory`
/// 在自己的 `ui_tracks` 作用域里调对账；`bound_tests` 的两条 ABBA 用例），两次都是
/// "读代码看不出来、跑起来全冻住"。
///
/// 【为什么只在 `debug_assertions` 下记账】发布构建里它是 `lock_or_recover` 的逐字
/// 等价实现：一次 `lock`、一个 `Drop`、零查表。热路径（音频回调）本来就不经过这里
/// ——它们用的是 `try_lock_or_recover`。
///
/// 【代价】每次取锁多一次 thread-local 查表；只在调试构建里。用它换"自锁会自己报名字"。
#[track_caller]
pub fn lock_tracked<T>(mutex: &Mutex<T>) -> TrackedGuard<'_, T> {
    let key = mutex as *const Mutex<T> as usize;
    #[cfg(debug_assertions)]
    {
        let site = std::panic::Location::caller();
        if let Some(first) = tracker::enter(key, site) {
            panic!(
                "reentrant lock: this thread has taken it at {}:{} and is taking it again at {}:{}",
                first.file(),
                first.line(),
                site.file(),
                site.line()
            );
        }
    }
    TrackedGuard {
        inner: mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        key,
    }
}

/// `lock_tracked` 的 guard：除了 `Deref` 之外，唯一职责是在 `Drop` 时销账。
pub struct TrackedGuard<'a, T> {
    inner: MutexGuard<'a, T>,
    /// 记账键（互斥量自身的地址）。
    key: usize,
}

impl<T> std::ops::Deref for TrackedGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for TrackedGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T> Drop for TrackedGuard<'_, T> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        tracker::leave(self.key);
        let _ = self.key;
    }
}

#[cfg(debug_assertions)]
mod tracker {
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        /// 本线程此刻持有的锁 → 第一次取它的位置。
        static HELD: RefCell<HashMap<usize, &'static std::panic::Location<'static>>> =
            RefCell::new(HashMap::new());
    }

    /// 登记；已经有记录时把**第一次**的位置原样返回（调用方据此报名字）。
    pub(super) fn enter(
        key: usize,
        site: &'static std::panic::Location<'static>,
    ) -> Option<&'static std::panic::Location<'static>> {
        HELD.with(|held| held.borrow_mut().insert(key, site))
    }

    pub(super) fn leave(key: usize) {
        HELD.with(|held| {
            held.borrow_mut().remove(&key);
        });
    }
}

/// `Mutex::try_lock` 的毒化恢复版本：锁被其它线程持有时返回 `None`。
///
/// 【为什么音频线程要用它而不是 lock_or_recover】阻塞 `lock` 在音频回调里是
/// 优先级反转窗口：非音频侧（诊断读数等）持锁瞬间被抢占，音频线程就被挂起。
/// 竞争窗口里宁可走"本块丢弃 / 本次不入池"的既有计数路径，也不冒把回调
/// 挂起的风险；毒化仍按全仓策略恢复（互斥语义在毒化后仍然成立）。
pub fn try_lock_or_recover<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{lock_tracked, MutexExt};
    use std::sync::{Arc, Mutex};

    /// 毒化之后必须仍能取到锁 —— 这是"UI 不再挂死"的唯一判据。
    #[test]
    fn a_poisoned_mutex_is_recovered_instead_of_panicking() {
        let mutex = Arc::new(Mutex::new(7_u32));
        let clone = Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = clone.lock_or_recover();
            panic!("poison the mutex");
        })
        .join();
        assert!(mutex.is_poisoned(), "前提：锁确实被毒化了");
        assert_eq!(*mutex.lock_or_recover(), 7, "恢复后仍能读到内部数据");
    }

    /// **不变量：重入会被点名，而不是静默冻住。**
    ///
    /// 【为什么它值得一条用例】自锁是本仓唯一一种"编译通过、测试全绿、现场必挂"的
    /// 缺陷形态。探针如果哪天被改成静默（比如有人嫌 panic 太吵），这条会立刻红。
    #[test]
    #[should_panic(expected = "reentrant lock")]
    fn a_reentrant_lock_names_itself() {
        let mutex = Mutex::new(1_u32);
        let _first = lock_tracked(&mutex);
        let _second = lock_tracked(&mutex);
    }

    /// 不同线程先后取同一把锁不是重入。
    #[test]
    fn a_lock_handed_between_threads_is_not_reentrant() {
        let mutex = Arc::new(Mutex::new(0_u32));
        let clone = Arc::clone(&mutex);
        let handle = std::thread::spawn(move || {
            *lock_tracked(&clone) += 1;
        });
        handle.join().unwrap();
        *lock_tracked(&mutex) += 1;
        assert_eq!(*mutex.lock_or_recover(), 2);
    }

    /// 影响宿主的读改写路径必须能**知道**这次取锁是从毒化恢复来的 —— 否则它会把
    /// 一份可能半写的状态写回 DAW。
    #[test]
    fn checked_recovery_reports_poison_to_host_facing_callers() {
        let mutex = Arc::new(Mutex::new(7_u32));
        let (_guard, poisoned) = mutex.lock_or_recover_checked();
        assert!(!poisoned, "未毒化时必须是 false（正常路径不受影响）");
        drop(_guard);
        let clone = Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = clone.lock_or_recover();
            panic!("poison the mutex");
        })
        .join();
        let (guard, poisoned) = mutex.lock_or_recover_checked();
        assert!(poisoned, "毒化恢复必须报告 true");
        assert_eq!(*guard, 7);
    }
}
