//! Allocation-free head/tail callback adapter with model-thread snapshot replacement.

use ara2_bridge_core::{
    AraError, HeadTailEntry, RealtimeFailureCode, RealtimeFailureQueue, RealtimeHeadTailView,
};
use ara2_bridge_sys::{ARAPlaybackRegionRef, ARATimeDuration};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// An installed view plus **its own** reader count.
///
/// 【为什么读者计数必须逐快照】`query` 是"载入当前指针 → 读它"两步。安装新快照时若
/// 立刻释放旧的那份，正读着它的 `query` 就会读到已释放内存。而把每一份都永远留着
/// （本文件此前就是这样）会随编辑次数无界增长：每次 `install` 都推一份进 Vec 且从不
/// 移除，长会话里每次重算都留下一份。逐快照计数把"读当前"与"读退役"分开 —— 退役的
/// 那份一旦没人读就能立刻回收。
struct RetainedView {
    readers: AtomicUsize,
    view: RealtimeHeadTailView,
}

/// 回收自旋上限：超过就放弃本次回收，把工作留给下一次 `install`。
///
/// 【为什么必须有上限】`install` 跑在 ARA 模型线程上，而模型线程还要处理宿主的其他
/// 回调。被抢占的读者只会在 `query` 的几条指令之间停留，正常一两次自旋即可；万一
/// 真的卡住，宁可这次不回收（内存稍晚一点释放），也不能让模型线程停在这里。
const RECLAIM_SPIN_LIMIT: u32 = 4096;

/// Owns immutable snapshots while realtime queries use one atomically selected view.
pub struct RealtimeHeadTailAdapter {
    snapshots: Mutex<Vec<Box<RetainedView>>>,
    current: AtomicPtr<RetainedView>,
    /// 正在"载入当前指针并登记到该快照"的读者数；封住 `query` 的两步窗口。
    registering: AtomicUsize,
    capacity: usize,
    failures: RealtimeFailureQueue,
}

#[allow(dead_code)] // Constructed by the controller adapter added after capability composition.
pub(crate) struct HeadTailDispatch {
    pub(crate) adapter: Arc<RealtimeHeadTailAdapter>,
}

impl crate::ffi::generated_callbacks::ControllerDelegate for HeadTailDispatch {
    fn get_playback_region_head_and_tail_time(
        &mut self,
        playback_region: ARAPlaybackRegionRef,
        head_time: *mut ARATimeDuration,
        tail_time: *mut ARATimeDuration,
    ) {
        if playback_region.is_null() || head_time.is_null() || tail_time.is_null() {
            let _ = self
                .adapter
                .failures
                .report(RealtimeFailureCode::InvalidState);
            return;
        }
        let key = playback_region as usize as u64;
        let Some((head, tail)) = self.adapter.query(key) else {
            return;
        };
        // SAFETY: ARA supplies both output pointers writable for one duration during this callback.
        unsafe {
            head_time.write(head);
            tail_time.write(tail);
        }
    }
}

impl RealtimeHeadTailAdapter {
    /// Creates an adapter and its preallocated deferred-failure queue.
    pub fn new(capacity: usize) -> Result<Self, AraError> {
        if capacity == 0 {
            return Err(AraError::InvalidArgument(
                "head/tail snapshot capacity must be nonzero",
            ));
        }
        let initial = Box::new(RetainedView {
            readers: AtomicUsize::new(0),
            view: RealtimeHeadTailView::new([], capacity)?,
        });
        // Box 的堆分配在移动 Box 本身时不动，所以先把地址取出来再把它放进 Vec 是安全的。
        let pointer = &*initial as *const RetainedView as *mut RetainedView;
        Ok(Self {
            snapshots: Mutex::new(vec![initial]),
            current: AtomicPtr::new(pointer),
            registering: AtomicUsize::new(0),
            capacity,
            failures: RealtimeFailureQueue::new(capacity)?,
        })
    }

    /// Builds and atomically installs a validated immutable snapshot off the realtime path.
    pub fn install(
        &self,
        entries: impl IntoIterator<Item = HeadTailEntry>,
    ) -> Result<(), AraError> {
        let view = RealtimeHeadTailView::new(entries, self.capacity)?;
        let mut retained = self
            .snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entry = Box::new(RetainedView {
            readers: AtomicUsize::new(0),
            view,
        });
        let pointer = &mut *entry as *mut RetainedView;
        retained.push(entry);
        // 【顺序】先换指针、再回收。反过来的话，回收会释放**当前**那一份，而指针还没换 ——
        // 这中间进入 `query` 的读者就会拿到已释放的地址。回收本身也保留 `current` 指向的
        // 那一份（见 `reclaim`），所以换完指针之后旧的那份才可能被回收。
        self.current.store(pointer, Ordering::Release);
        self.reclaim(&mut retained);
        Ok(())
    }

    /// Performs one allocation-free and blocking-lock-free head/tail lookup.
    pub fn query(&self, region_key: u64) -> Option<(f64, f64)> {
        self.registering.fetch_add(1, Ordering::SeqCst);
        let pointer = self.current.load(Ordering::SeqCst);
        if !pointer.is_null() {
            // SAFETY: 指针来自 `retained` 中存活的 Box；`registering` 非零期间不会被回收。
            unsafe { (*pointer).readers.fetch_add(1, Ordering::SeqCst) };
        }
        self.registering.fetch_sub(1, Ordering::SeqCst);
        // SAFETY: 本读区登记的视图在读者计数归零前不会被回收。
        let result = if pointer.is_null() {
            None
        } else {
            unsafe { (*pointer).view.query(region_key) }
        };
        if !pointer.is_null() {
            // SAFETY: `enter` 登记过该视图；这里恰好析构一次。
            unsafe { (*pointer).readers.fetch_sub(1, Ordering::SeqCst) };
        }
        if result.is_none() {
            let _ = self.failures.report(RealtimeFailureCode::MissingRegion);
        }
        result
    }

    /// Pops one deferred fixed-size realtime failure for model-thread diagnostics.
    pub fn pop_failure(&self) -> Option<RealtimeFailureCode> {
        self.failures.pop()
    }

    /// Returns the number of snapshots retained to protect concurrent readers.
    ///
    /// 【为什么这是个有界数】只保留"当前那一份"与"仍有读者在用的退役份"。没有并发读者
    /// 时它恒为 1；此前它等于 `install` 的调用次数，也就是随编辑次数无界增长。
    pub fn retained_snapshot_count(&self) -> usize {
        self.snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// 释放既不是当前、也没有读者的退役视图。必须持 `snapshots` 锁且仅在模型线程调用。
    fn reclaim(&self, retained: &mut Vec<Box<RetainedView>>) {
        let current = self.current.load(Ordering::SeqCst);
        let mut spins = 0u32;
        while self.registering.load(Ordering::SeqCst) != 0 {
            if spins >= RECLAIM_SPIN_LIMIT {
                return;
            }
            spins += 1;
            std::hint::spin_loop();
        }
        retained.retain(|entry| {
            std::ptr::eq(&**entry, current) || entry.readers.load(Ordering::SeqCst) != 0
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{callbacks, generated_callbacks};

    #[test]
    fn query_does_not_acquire_snapshot_storage_lock() {
        let adapter = RealtimeHeadTailAdapter::new(2).unwrap();
        adapter
            .install([HeadTailEntry::new(1, 0.1, 0.2).unwrap()])
            .unwrap();
        let storage = adapter.snapshots.lock().unwrap();
        assert_eq!(adapter.query(1), Some((0.1, 0.2)));
        drop(storage);
    }

    #[test]
    fn abi_head_tail_shell_uses_snapshot_while_storage_lock_is_held() {
        let adapter = Arc::new(RealtimeHeadTailAdapter::new(2).unwrap());
        let identity = Box::new(0_u8);
        let region: ARAPlaybackRegionRef = std::ptr::from_ref(identity.as_ref()).cast_mut().cast();
        adapter
            .install([HeadTailEntry::new(region as usize as u64, 0.25, 0.5).unwrap()])
            .unwrap();
        let controller = callbacks::controller_ref(Box::new(HeadTailDispatch {
            adapter: adapter.clone(),
        }));
        let storage = adapter.snapshots.lock().unwrap();
        let mut head = 0.0;
        let mut tail = 0.0;
        // SAFETY: all opaque identities, output pointers, and controller storage are live.
        unsafe {
            generated_callbacks::get_playback_region_head_and_tail_time(
                controller,
                region,
                &raw mut head,
                &raw mut tail,
            )
        };
        assert_eq!((head, tail), (0.25, 0.5));
        drop(storage);
        // SAFETY: this uniquely consumes the controller allocation after its callback completes.
        unsafe { callbacks::destroy_controller_ref(controller) };
    }

    /// **回归：反复安装不得让保留的视图数无界增长。**
    ///
    /// 【为什么这是回归测试】此前每次 `install` 都 `push` 一份进 Vec 且从不移除，于是
    /// 保留数等于安装次数 —— 插件每次重算都安装一次，长会话里内存随编辑次数线性增长。
    /// 现在只保留"当前那一份"（没有并发读者时）。
    #[test]
    fn repeated_installs_do_not_grow_the_retained_set() {
        let adapter = RealtimeHeadTailAdapter::new(8).unwrap();
        for key in 0..1000_u64 {
            adapter
                .install([HeadTailEntry::new(key, 0.0, 0.0).unwrap()])
                .unwrap();
            assert_eq!(
                adapter.retained_snapshot_count(),
                1,
                "无并发读者时只应保留当前那一份"
            );
        }
        assert_eq!(adapter.query(999), Some((0.0, 0.0)));
    }

    /// 活跃读者期间退役视图必须保留；读者退出后下一次安装才回收它。
    #[test]
    fn an_active_reader_keeps_the_retired_view_alive_until_it_leaves() {
        let adapter = RealtimeHeadTailAdapter::new(8).unwrap();
        adapter
            .install([HeadTailEntry::new(1, 0.5, 0.25).unwrap()])
            .unwrap();
        // 手动登记一个读者（`query` 内部就是这两步），模拟"读区横跨一次安装"。
        adapter.registering.fetch_add(1, Ordering::SeqCst);
        let pointer = adapter.current.load(Ordering::SeqCst);
        unsafe { (*pointer).readers.fetch_add(1, Ordering::SeqCst) };
        adapter.registering.fetch_sub(1, Ordering::SeqCst);

        adapter
            .install([HeadTailEntry::new(2, 0.0, 0.0).unwrap()])
            .unwrap();
        assert_eq!(
            adapter.retained_snapshot_count(),
            2,
            "被读的退役视图必须保留"
        );

        unsafe { (*pointer).readers.fetch_sub(1, Ordering::SeqCst) };
        adapter
            .install([HeadTailEntry::new(3, 0.0, 0.0).unwrap()])
            .unwrap();
        assert_eq!(
            adapter.retained_snapshot_count(),
            1,
            "读者退出后退役视图必须被回收"
        );
    }
}
