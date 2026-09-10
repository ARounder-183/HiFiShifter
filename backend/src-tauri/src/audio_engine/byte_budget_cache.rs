//! Byte-budgeted LRU cache for PCM data.
//!
//! Wraps `lru::LruCache` with byte-weight tracking. When the total estimated
//! bytes exceeds the budget, the least-recently-used entries are evicted.
//!
//! Default budget: 1 GB. Overridable by the user setting
//! `UiSettings::audio_cache_budget_mb` (applied at runtime via
//! [`apply_cache_budget`]) or, for support/diagnostics, by
//! `HIFISHIFTER_PCM_CACHE_BUDGET_MB`.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Default byte budget for all PCM caches combined (1 GB).
const DEFAULT_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

/// 用户可设预算的允许范围（MB）。
pub const MIN_BUDGET_MB: u64 = 256;
pub const MAX_BUDGET_MB: u64 = 4096;

/// 运行时可调预算（字节）。0 = 未设置，回落到环境变量 / 默认值。
///
/// 之所以用 0 作哨兵：预算是"总量"而非"单缓存量"，0 字节预算没有意义。
static RUNTIME_BUDGET_BYTES: AtomicU64 = AtomicU64::new(0);

/// Read `HIFISHIFTER_PCM_CACHE_BUDGET_MB` or return default budget in bytes.
pub fn env_cache_budget_bytes() -> u64 {
    let mb = std::env::var("HIFISHIFTER_PCM_CACHE_BUDGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_BUDGET_BYTES / (1024 * 1024));
    mb.saturating_mul(1024 * 1024)
}

/// 当前**实际生效**的缓存总预算（字节）。
///
/// 优先级：用户设置 > 环境变量 > 默认值。
pub fn cache_budget_bytes() -> u64 {
    match RUNTIME_BUDGET_BYTES.load(Ordering::Relaxed) {
        0 => env_cache_budget_bytes(),
        v => v,
    }
}

/// 设置运行时预算（字节；0 = 清除覆盖，回落到环境变量/默认值）。
///
/// 仅写入本原子值。要把新预算**推送**到各缓存实例（并触发缩容回收），请调用
/// `cache_registry::apply_cache_budget`。两者分开是因为缓存实例化路径无法访问
/// 注册表（构造期就可能被调用）。
pub fn set_cache_budget_bytes(bytes: u64) {
    RUNTIME_BUDGET_BYTES.store(bytes, Ordering::Relaxed);
}

/// 把 MB 值钳制到允许范围。
pub fn clamp_budget_mb(mb: u64) -> u64 {
    mb.clamp(MIN_BUDGET_MB, MAX_BUDGET_MB)
}

/// 默认预算（MB），供 `UiSettings` 的 serde 默认值使用。
pub fn default_budget_mb() -> u64 {
    DEFAULT_BUDGET_BYTES / (1024 * 1024)
}

/// A byte-budgeted LRU cache.
///
/// Each entry has an associated byte weight. When inserting causes total bytes
/// to exceed `budget_bytes`, LRU entries are evicted until under budget.
pub struct ByteBudgetCache<K: Eq + std::hash::Hash + Clone, V> {
    inner: LruCache<K, (V, u64)>,
    total_bytes: u64,
    budget_bytes: u64,
}

impl<K: Eq + std::hash::Hash + Clone, V> ByteBudgetCache<K, V> {
    /// Create a new cache with the given entry capacity and byte budget.
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            inner: LruCache::new(cap),
            total_bytes: 0,
            budget_bytes: budget_bytes.max(1),
        }
    }

    /// Create a cache with capacity from env or default, and budget from env.
    // 泛型 API 面：部分 (K, V) 实例化组合暂未用到这两个入口，
    // dead_code 分析按实例化逐个报告，这里统一标注保留。
    #[allow(dead_code)]
    pub fn from_env(capacity: usize) -> Self {
        Self::new(capacity, env_cache_budget_bytes())
    }

    /// Get a reference to an entry, promoting it in LRU order.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.inner.get(key).map(|(v, _)| v)
    }

    /// Get a mutable reference to an entry, promoting it in LRU order.
    #[allow(dead_code)]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.inner.get_mut(key).map(|(v, _)| v)
    }

    /// Insert an entry with its byte weight.
    ///
    /// If the entry already exists, it is updated (old weight is subtracted).
    /// After insertion, if total bytes exceeds budget, LRU entries are evicted.
    pub fn insert(&mut self, key: K, value: V, weight_bytes: u64) {
        // `push`（而非 `put`）会返回被顶掉的条目：key 已存在时是旧条目，
        // 触及条目容量上限时是被 LRU 逐出的其他条目。两种情况的权重都必须
        // 从累计值中扣除 —— `put` 会静默丢弃该条目，导致"幽灵字节"不断
        // 累积，最终预算检查把整个缓存清空（历史 bug，见回归测试）。
        let displaced = self.inner.push(key, (value, weight_bytes));
        if let Some((_, (_, displaced_weight))) = displaced {
            self.total_bytes = self.total_bytes.saturating_sub(displaced_weight);
        }
        self.total_bytes = self.total_bytes.saturating_add(weight_bytes);

        // Evict LRU entries until under budget.
        while self.total_bytes > self.budget_bytes {
            if let Some((_, (_, weight))) = self.inner.pop_lru() {
                self.total_bytes = self.total_bytes.saturating_sub(weight);
            } else {
                break;
            }
        }
    }

    /// Remove an entry by key, returning its value and byte weight.
    // 以下多个入口同 from_env / get_mut：泛型 API 面，按实例化组合
    // 逐个报告 dead_code，这里统一标注保留。
    #[allow(dead_code)]
    pub fn pop(&mut self, key: &K) -> Option<(V, u64)> {
        if let Some((value, weight)) = self.inner.pop(key) {
            self.total_bytes = self.total_bytes.saturating_sub(weight);
            Some((value, weight))
        } else {
            None
        }
    }

    /// Invalidate all entries matching a predicate.
    pub fn invalidate_where(&mut self, mut predicate: impl FnMut(&K) -> bool) {
        let keys_to_remove: Vec<K> = self
            .inner
            .iter()
            .filter(|(k, _)| predicate(k))
            .map(|(k, _)| k.clone())
            .collect();

        for key in &keys_to_remove {
            if let Some((_, weight)) = self.inner.pop(key) {
                self.total_bytes = self.total_bytes.saturating_sub(weight);
            }
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.total_bytes = 0;
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cache is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if a key exists without promoting it.
    #[allow(dead_code)]
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    /// Total estimated bytes currently held.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Budget in bytes.
    #[allow(dead_code)]
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// 运行时调整字节预算（见 P1-7：用户可在设置里改"音频缓存预算"）。
    ///
    /// 缩容时**按 LRU 立即驱逐**到新预算以下，而不是等下次插入才回收 —— 否则
    /// "调小预算"在长时间不插入的情况下等于没生效，用户会以为设置无效。
    /// 扩容不预分配任何内存。
    pub fn set_budget(&mut self, budget_bytes: u64) {
        self.budget_bytes = budget_bytes.max(1);
        while self.total_bytes > self.budget_bytes {
            if let Some((_, (_, weight))) = self.inner.pop_lru() {
                self.total_bytes = self.total_bytes.saturating_sub(weight);
            } else {
                break;
            }
        }
    }

    /// Iterate over entries in LRU order (most recent first).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter().map(|(k, (v, _))| (k, v))
    }

    /// Ensure capacity is at least `min_capacity` (does not shrink).
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        let new_cap = NonZeroUsize::new(min_capacity.max(1)).unwrap();
        if new_cap.get() > self.inner.cap().get() {
            self.inner.resize(new_cap);
        }
    }

    /// Resize the entry capacity (may cause eviction of LRU entries).
    #[allow(dead_code)]
    pub fn resize(&mut self, new_capacity: usize) {
        let new_cap = NonZeroUsize::new(new_capacity.max(1)).unwrap();
        self.inner.resize(new_cap);
        // Recalculate total_bytes from remaining entries.
        self.total_bytes = self.inner.iter().map(|(_, (_, w))| *w).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_capacity_eviction_keeps_total_bytes_accurate() {
        let mut cache: ByteBudgetCache<u64, u64> = ByteBudgetCache::new(2, u64::MAX);

        cache.insert(1, 1, 100);
        cache.insert(2, 2, 100);
        assert_eq!(cache.total_bytes(), 200);

        // Capacity is 2: inserting key 3 must evict key 1 and drop its weight.
        cache.insert(3, 3, 100);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), 200);

        // The byte budget is enforced after every insert, so a fill sequence
        // never exceeds it mid-way: each insert beyond the budget immediately
        // evicts the LRU entry down to the budget.
        let mut budgeted: ByteBudgetCache<u64, u64> = ByteBudgetCache::new(8, 250);
        for i in 0..8 {
            budgeted.insert(i, i, 100);
        }
        assert_eq!(budgeted.total_bytes(), 200);
        assert_eq!(budgeted.len(), 2);

        // Capacity-pressure evictions (entry bound) must also keep the total
        // accurate: inserting at capacity displaces the LRU weight, not adds
        // a ghost copy of it.
        budgeted.insert(100, 100, 100);
        assert_eq!(budgeted.total_bytes(), 200);
        assert_eq!(budgeted.len(), 2);

        // Overwriting an existing key must not double-count its old weight:
        // {7:100, 100:100} + overwrite 100 with weight 50 → 100 + 50.
        budgeted.insert(100, 100, 50);
        assert_eq!(budgeted.total_bytes(), 150);
    }

    #[test]
    fn shrinking_the_budget_reclaims_immediately() {
        // 用户把"音频缓存预算"调小时，回收必须立即发生：若只更新数值、等下次插入
        // 才驱逐，在长时间不插入的情况下等于设置无效（见 P1-7）。
        let mut cache: ByteBudgetCache<u64, u64> = ByteBudgetCache::new(16, u64::MAX);

        for k in 0..6u64 {
            cache.insert(k, k, 100);
        }
        assert_eq!(cache.len(), 6);
        assert_eq!(cache.total_bytes(), 600);

        // 缩到 250 → 必须驱逐到 250 以下，且从 LRU（最旧的 0）开始。
        cache.set_budget(250);
        assert!(cache.total_bytes() <= 250, "got {}", cache.total_bytes());
        assert_eq!(cache.total_bytes(), 200);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&0).is_none(), "oldest entry must be evicted first");
        assert!(cache.get(&5).is_some(), "newest entry must survive");
    }

    #[test]
    fn growing_the_budget_keeps_existing_entries() {
        let mut cache: ByteBudgetCache<u64, u64> = ByteBudgetCache::new(16, 250);
        for k in 0..4u64 {
            cache.insert(k, k, 100);
        }
        let before_len = cache.len();
        let before_bytes = cache.total_bytes();

        cache.set_budget(10_000);

        assert_eq!(cache.len(), before_len, "growing must not evict anything");
        assert_eq!(cache.total_bytes(), before_bytes);
        assert_eq!(cache.budget_bytes(), 10_000);
    }

    #[test]
    fn zero_budget_is_clamped_to_one_byte() {
        // 0 预算会让每次插入立刻被驱逐（缓存等于关闭且无报错），因此下限钳到 1。
        let mut cache: ByteBudgetCache<u64, u64> = ByteBudgetCache::new(4, 1000);
        cache.insert(1, 1, 100);
        cache.set_budget(0);
        assert_eq!(cache.budget_bytes(), 1);
        assert_eq!(cache.len(), 0, "everything exceeds a 1-byte budget");
    }

    #[test]
    fn runtime_budget_override_wins_over_env_default() {
        // 用户设置优先于环境变量/默认值；0 是"未设置"的哨兵，故清除后应回落。
        let original = cache_budget_bytes();
        set_cache_budget_bytes(3 * 1024 * 1024 * 1024);
        assert_eq!(cache_budget_bytes(), 3 * 1024 * 1024 * 1024);
        set_cache_budget_bytes(0);
        assert_eq!(
            cache_budget_bytes(),
            env_cache_budget_bytes(),
            "clearing the override must fall back to env/default"
        );
        set_cache_budget_bytes(original.max(1));
        set_cache_budget_bytes(0);
    }

    #[test]
    fn budget_mb_is_clamped_to_the_supported_range() {
        assert_eq!(clamp_budget_mb(0), MIN_BUDGET_MB);
        assert_eq!(clamp_budget_mb(1), MIN_BUDGET_MB);
        assert_eq!(clamp_budget_mb(1024), 1024);
        assert_eq!(clamp_budget_mb(u64::MAX), MAX_BUDGET_MB);
    }
}
