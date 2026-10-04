//! 源 PCM 与全部 renderer 退役快照共用的进程级 512 MiB 硬预算。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
pub(crate) struct MemoryBudget {
    used: AtomicUsize,
    limit: usize,
}
#[derive(Debug)]
pub(crate) struct Reservation {
    budget: Arc<MemoryBudget>,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
impl MemoryBudget {
    /// 先收费再分配，失败时不创建大缓冲；RAII 回收在非实时 owner drop 中进行。
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .ok()?;
        Some(Reservation {
            budget: self.clone(),
            bytes,
        })
    }
}
pub(crate) fn global_budget() -> &'static Arc<MemoryBudget> {
    static BUDGET: OnceLock<Arc<MemoryBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| {
        Arc::new(MemoryBudget {
            used: AtomicUsize::new(0),
            limit: 512 * 1024 * 1024,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 两类持有者共享上限，释放只回收自己的额度。
    #[test]
    fn shared_source_and_snapshot_reservations_cannot_exceed_the_budget() {
        let budget = Arc::new(MemoryBudget {
            used: AtomicUsize::new(0),
            limit: 16,
        });
        let source = budget.reserve(8).unwrap();
        let snapshot = budget.reserve(8).unwrap();
        assert!(budget.reserve(1).is_none());
        drop(source);
        assert!(budget.reserve(9).is_none());
        assert!(budget.reserve(8).is_some());
        drop(snapshot);
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
    }
}
