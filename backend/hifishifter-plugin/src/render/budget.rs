//! 源 PCM 与全部 renderer 退役快照共用的进程级 PCM 预算。
//!
//! 【为什么不再是写死的 512 MiB】那个常数是"先导版"的探针上限，而 PCM 需求与
//! **素材长度**成正比：一段 165 s 的立体声在旧实现里要 248 MB（两档采样率 + 工作区 +
//! 源交错副本），512 MiB 只够两分半。用户报的 `render PCM budget exceeded` 就是
//! 这条硬墙 —— 而且它是**永久失败**：跨度没变，重试多少次都一样。
//!
//! 现在上限按物理内存给（1/4，夹在 512 MiB..4 GiB 之间），环境变量
//! `HIFISHIFTER_VST_PCM_BUDGET_MB` 可覆盖。下限仍是 512 MiB —— 绝不比今天更小。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// 预算下限（= 旧实现的固定上限）。
const MIN_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// 预算上限（防止在超大内存机器上抬到荒唐的值）。
const MAX_LIMIT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// 物理内存总量（字节）。取不到时返回 `None`，调用方退回下限。
fn physical_memory_bytes() -> Option<usize> {
    #[cfg(windows)]
    {
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: 结构体按 SDK 定义清零后由系统填写；只读 `ullTotalPhys`。
        unsafe {
            let mut status = MEMORYSTATUSEX {
                dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
                ..Default::default()
            };
            if GlobalMemoryStatusEx(&mut status).is_err() {
                return None;
            }
            usize::try_from(status.ullTotalPhys).ok()
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // SAFETY: `sysconf` 只读系统常数，无副作用。
        unsafe {
            let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
            let page_size = libc::sysconf(libc::_SC_PAGESIZE);
            if pages <= 0 || page_size <= 0 {
                return None;
            }
            usize::try_from(pages).ok()?.checked_mul(usize::try_from(page_size).ok()?)
        }
    }
    #[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
    {
        None
    }
}

/// 本次进程的 PCM 预算上限。
fn adaptive_limit() -> usize {
    if let Some(mb) = std::env::var("HIFISHIFTER_VST_PCM_BUDGET_MB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
    {
        return usize::try_from(mb)
            .ok()
            .and_then(|mb| mb.checked_mul(1024 * 1024))
            .unwrap_or(MAX_LIMIT_BYTES);
    }
    let Some(physical) = physical_memory_bytes() else {
        return MIN_LIMIT_BYTES;
    };
    (physical / 4).clamp(MIN_LIMIT_BYTES, MAX_LIMIT_BYTES)
}

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
    /// 已转交该持有者的实际额度，用于非实时诊断；不是重新读取全局used。
    // 非实时诊断读取入口，保留。
    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// 整批预检额度分给实际持有者，不重复收费；余额随临时工作域离开而退还。
    pub fn split_off(&mut self, bytes: usize) -> Option<Self> {
        self.bytes = self.bytes.checked_sub(bytes)?;
        Some(Self {
            budget: self.budget.clone(),
            bytes,
        })
    }
}
impl MemoryBudget {
    /// PCM显式额度固定上限，不代表模型/GPU或系统实际驻留内存。
    pub fn limit(&self) -> usize {
        self.limit
    }
    /// 当前由源、播放快照与准备工作域持有的显式额度。
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
    /// 额度高水位用于诊断，不能冒充进程WorkingSet峰值。
    // 高水位诊断读取入口，保留。
    #[cfg(test)]
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
    /// 先收费再分配，失败时不创建大缓冲；RAII 回收在非实时 owner drop 中进行。
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        let previous = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .ok()?;
        self.peak.fetch_max(previous + bytes, Ordering::AcqRel);
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
            peak: AtomicUsize::new(0),
            limit: adaptive_limit(),
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
            peak: AtomicUsize::new(0),
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
        let budget = Arc::new(MemoryBudget {
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: 16,
        });
        let mut batch = budget.reserve(16).unwrap();
        let pcm = batch.split_off(8).unwrap();
        assert!(batch.split_off(9).is_none());
        assert_eq!(budget.used(), 16);
        assert_eq!(budget.peak(), 16);
        drop(batch);
        assert_eq!(budget.used(), 8);
        assert!(budget.reserve(9).is_none());
        drop(pcm);
        assert_eq!(budget.used(), 0);
    }
}
