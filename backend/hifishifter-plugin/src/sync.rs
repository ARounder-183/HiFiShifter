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
pub(crate) trait MutexExt<T> {
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

#[cfg(test)]
mod tests {
    use super::MutexExt;
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
