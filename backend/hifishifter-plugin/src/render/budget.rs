//! 源 PCM 与全部 renderer 退役快照共用的进程级 512 MiB 硬预算。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
pub(crate) struct MemoryBudget {
    used: AtomicUsize,
    peak: AtomicUsize,
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
impl Reservation {
    /// 整批预检额度分给实际持有者，不重复收费；余额随临时工作域离开而退还。
    pub fn split_off(&mut self,bytes:usize)->Option<Self> {
        self.bytes=self.bytes.checked_sub(bytes)?;Some(Self {budget:self.budget.clone(),bytes})
    }
}
impl MemoryBudget {
    /// PCM显式额度固定上限，不代表模型/GPU或系统实际驻留内存。
    pub fn limit(&self)->usize {self.limit}
    /// 当前由源、播放快照与准备工作域持有的显式额度。
    pub fn used(&self)->usize {self.used.load(Ordering::Acquire)}
    /// 额度高水位用于诊断，不能冒充进程WorkingSet峰值。
    pub fn peak(&self)->usize {self.peak.load(Ordering::Acquire)}
    /// 先收费再分配，失败时不创建大缓冲；RAII 回收在非实时 owner drop 中进行。
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        let previous=self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .ok()?;
        self.peak.fetch_max(previous+bytes,Ordering::AcqRel);
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
            peak:AtomicUsize::new(0),
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
            peak:AtomicUsize::new(0),
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
    /// 预检额度转交最终PCM后只保留实际份额，拆分失败不改变余额/全局收费。
    #[test]
    fn batch_partition_preserves_peak_and_releases_temporary_working_bytes() {
        let budget=Arc::new(MemoryBudget {used:AtomicUsize::new(0),peak:AtomicUsize::new(0),limit:16});
        let mut batch=budget.reserve(16).unwrap();let pcm=batch.split_off(8).unwrap();
        assert!(batch.split_off(9).is_none());assert_eq!(budget.used(),16);assert_eq!(budget.peak(),16);
        drop(batch);assert_eq!(budget.used(),8);assert!(budget.reserve(9).is_none());drop(pcm);assert_eq!(budget.used(),0);
    }
}
