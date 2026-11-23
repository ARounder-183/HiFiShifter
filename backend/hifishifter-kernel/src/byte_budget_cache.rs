//! Byte-budgeted LRU cache for PCM data.
//!
//! Wraps `lru::LruCache` with byte-weight tracking. When the total estimated
//! bytes exceeds the budget, the least-recently-used entries are evicted.
//!
//! Default budget: 1 GB on desktop, 160 MB on mobile (both overridable via
//! `HIFISHIFTER_PCM_CACHE_BUDGET_MB`).

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

/// 桌面形态的默认预算（1 GB）。
pub const DESKTOP_DEFAULT_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

/// 移动形态的默认预算（160 MB）。
///
/// 【为什么是 160 MB 而不是"1 GB 除以几"】这个数字来自移动端的**硬约束**，不是比例：
/// iOS 给单个应用的常驻内存上限大致在 1~1.5 GB（机型相关），超过即被 jetsam 直接杀掉
/// —— 没有异常、没有日志，用户看到的是"应用闪退"。而 PCM 缓存只是进程里的一份**派生
/// 数据**，它旁边还要放：ORT 会话（百 MB 级）、WebGL 纹理（几十 MB）、解码后的工程
/// 音频。把派生数据压在 160 MB 以内，剩下的预算才留给"不缓存就会算错"的东西。
///
/// 缓存变小不会改变正确性：`ByteBudgetCache` 只是更早地淘汰，淘汰后按需重算。
pub const MOBILE_DEFAULT_BUDGET_BYTES: u64 = 160 * 1024 * 1024;

/// 进程级默认预算。
///
/// 【为什么是运行期设置而不是 `cfg(target_os)`】内核是"设备无关"的（见 crate 文档），
/// 里面出现平台判断会让它在插件形态下也要猜自己在哪。宿主在启动时用
/// [`set_default_budget_bytes`] 声明一次，内核只管读 —— 于是"移动端用多少内存"这个
/// 决策留在收敛层（`src-tauri/src/platform/`）里，只有一处。
static DEFAULT_BUDGET_BYTES: AtomicU64 = AtomicU64::new(DESKTOP_DEFAULT_BUDGET_BYTES);

/// 由宿主在启动时声明默认预算；传 0 或异常值被忽略。
pub fn set_default_budget_bytes(bytes: u64) {
    if bytes > 0 {
        DEFAULT_BUDGET_BYTES.store(bytes, Ordering::Relaxed);
    }
}

/// 当前生效的默认预算（字节）。
pub fn default_budget_bytes() -> u64 {
    DEFAULT_BUDGET_BYTES.load(Ordering::Relaxed)
}

/// Read `HIFISHIFTER_PCM_CACHE_BUDGET_MB` or return default budget in bytes.
pub fn env_cache_budget_bytes() -> u64 {
    let fallback_mb = (default_budget_bytes() / (1024 * 1024)).max(1);
    let mb = std::env::var("HIFISHIFTER_PCM_CACHE_BUDGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(fallback_mb);
    mb.saturating_mul(1024 * 1024)
}

/// A byte-budgeted LRU cache.
///
/// Each entry has an associated byte weight. When inserting causes total bytes
/// to exceed `budget_bytes`, LRU entries are evicted until under budget.
pub struct ByteBudgetCache<K: Eq + std::hash::Hash + Clone, V> {
    inner: LruCache<K, (V, u64)>,
    total_bytes: u64,
    budget_bytes: u64,
    /// Optional "in use" predicate: entries for which it returns true are
    /// skipped by eviction (pinned). See [`Self::with_pin`].
    is_pinned: Option<fn(&K, &V) -> bool>,
}

impl<K: Eq + std::hash::Hash + Clone, V> ByteBudgetCache<K, V> {
    /// Create a new cache with the given entry capacity and byte budget.
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            inner: LruCache::new(cap),
            total_bytes: 0,
            budget_bytes: budget_bytes.max(1),
            is_pinned: None,
        }
    }

    /// Create a cache with capacity from env or default, and budget from env.
    // 泛型 API 面：部分 (K, V) 实例化组合暂未用到这两个入口，
    // dead_code 分析按实例化逐个报告，这里统一标注保留。
    pub fn from_env(capacity: usize) -> Self {
        Self::new(capacity, env_cache_budget_bytes())
    }

    /// Set the "in use" (pinned) predicate: entries it flags are never evicted.
    ///
    /// 【为什么需要】缓存条目与引擎快照共享同一个 `Arc`（见
    /// `ResampledStereo::pcm`）。若把仍在播放的 PCM 逐出，缓存账面字节下降但内存
    /// 并未释放；下一次快照重建还会**再解码一份**，同一素材两份常驻。用
    /// `Arc::strong_count > 1` 判定"在用"即可让逐出只针对真正无人引用的条目 ——
    /// 与 `WaveformPeakCache` 的 pin 语义一致。
    pub fn with_pin(mut self, is_pinned: fn(&K, &V) -> bool) -> Self {
        self.is_pinned = Some(is_pinned);
        self
    }

    /// Get a reference to an entry, promoting it in LRU order.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.inner.get(key).map(|(v, _)| v)
    }

    /// Get a mutable reference to an entry, promoting it in LRU order.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.inner.get_mut(key).map(|(v, _)| v)
    }

    /// Insert an entry with its byte weight.
    ///
    /// If the entry already exists, it is updated (old weight is subtracted).
    /// After insertion, if total bytes exceeds budget, LRU entries are evicted —
    /// but never the entry that was just inserted, and never a pinned entry.
    pub fn insert(&mut self, key: K, value: V, weight_bytes: u64) {
        // `push`（而非 `put`）会返回被顶掉的条目：key 已存在时是旧条目，
        // 触及条目容量上限时是被 LRU 逐出的其他条目。两种情况的权重都必须
        // 从累计值中扣除 —— `put` 会静默丢弃该条目，导致"幽灵字节"不断
        // 累积，最终预算检查把整个缓存清空（历史 bug，见回归测试）。
        // 记住刚插入的键：`push` 把它置于 MRU 端，下面从 LRU 端找牺牲品时，
        // 若所有更老的条目都被 pin 住，未 pin 的**新条目**会成为唯一候选并被
        // 立即逐出 —— 与"绝不逐出刚插入的条目"这条不变量直接冲突（超长 PCM
        // 因此每次 update_timeline 都全量重解码）。缓存是软上限，宁可超预算。
        let inserted_key = key.clone();
        let displaced = self.inner.push(key, (value, weight_bytes));
        if let Some((_, (_, displaced_weight))) = displaced {
            self.total_bytes = self.total_bytes.saturating_sub(displaced_weight);
        }
        self.total_bytes = self.total_bytes.saturating_add(weight_bytes);

        // Evict LRU entries until under budget.
        //
        // 【两条必须的例外】
        // 1. **绝不逐出刚插入的条目**：新条目是 MRU，仅当它是唯一剩余条目时才会
        //    被 `pop_lru` 选中。若没有这道 `len() > 1` 闸门，一个比整个预算还大的
        //    条目（超长音频的整文件 PCM，例如 1 小时立体声 ≈ 1.18GiB > 1GB 预算）
        //    会"插入即自逐出"，永远留在不了缓存里：每次 `update_timeline` 都重新
        //    全量解码（解码风暴，瞬时内存翻倍），快照还会因 miss 静默跳过该 clip。
        // 2. **绝不逐出在用（pin）条目**：见 `with_pin`。没有它，仍在播放的 PCM 被
        //    逐出后账面字节下降但内存未释放，下一次快照重建会再解码一份。
        // 因此预算在此处是**软**上限：为保住唯一/在用的巨型条目，可以短暂超出。
        while self.total_bytes > self.budget_bytes && self.inner.len() > 1 {
            let victim = {
                let pinned = &self.is_pinned;
                self.inner
                    .iter()
                    .rev() // 从 LRU 端开始找
                    .find(|(k, (v, _))| {
                        // 跳过刚插入的条目（见上方说明）与所有在用条目。
                        **k != inserted_key && !matches!(pinned, Some(is_pinned) if is_pinned(k, v))
                    })
                    .map(|(k, _)| k.clone())
            };
            let Some(victim) = victim else {
                // 所有条目都在用：保留，宁可超预算也不制造重复解码。
                break;
            };
            if let Some((_, weight)) = self.inner.pop(&victim) {
                self.total_bytes = self.total_bytes.saturating_sub(weight);
            } else {
                break;
            }
        }
    }

    /// Remove an entry by key, returning its value and byte weight.
    // 以下多个入口同 from_env / get_mut：泛型 API 面，按实例化组合
    // 逐个报告 dead_code，这里统一标注保留。
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
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if a key exists without promoting it.
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    /// Total estimated bytes currently held.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Budget in bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
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
    fn oversized_entry_survives_insert() {
        let mut cache: ByteBudgetCache<u64, u64> = ByteBudgetCache::new(8, 250);
        // 1000 字节远超 250 预算：旧实现会"插入即自逐出"，之后永远取不回。
        // 超长音频的整文件 PCM 正是这种条目。
        cache.insert(1, 1, 1000);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&1), Some(&1));
        assert_eq!(cache.total_bytes(), 1000);
    }

    #[test]
    fn just_inserted_entry_survives_when_all_others_are_pinned() {
        struct Item {
            pinned: bool,
        }
        // 老条目全被 pin 住；新条目未 pin 且超预算。旧实现从 LRU 端找牺牲品时
        // 会一路跳过所有老条目，最后选中刚插入的 MRU 键并 pop 掉 —— 插入即自逐出。
        // 缓存是软上限，正确行为是保留新条目（短暂超预算）。
        let mut cache: ByteBudgetCache<u64, Item> =
            ByteBudgetCache::new(8, 250).with_pin(|_, v| v.pinned);
        cache.insert(1, Item { pinned: true }, 100);
        cache.insert(2, Item { pinned: true }, 100);
        cache.insert(3, Item { pinned: false }, 100); // 300 > 250

        assert!(cache.get(&3).is_some(), "刚插入的条目不得被逐出");
        assert!(cache.get(&1).is_some(), "被 pin 的老条目仍在");
        assert!(cache.get(&2).is_some(), "被 pin 的老条目仍在");
        assert_eq!(cache.total_bytes(), 300);
    }

    #[test]
    fn pinned_entries_are_skipped_by_eviction() {
        struct Item {
            pinned: bool,
        }
        // 预算 250；条目都重 100。key=1 与 key=3 被 pin，只有 key=2 可被逐出。
        let mut cache: ByteBudgetCache<u64, Item> =
            ByteBudgetCache::new(8, 250).with_pin(|_, v| v.pinned);
        cache.insert(1, Item { pinned: true }, 100);
        cache.insert(2, Item { pinned: false }, 100);
        cache.insert(3, Item { pinned: true }, 100); // 300 > 250 → 需要逐出

        assert!(cache.get(&1).is_some(), "被 pin 的条目不得被逐出");
        assert!(cache.get(&3).is_some(), "被 pin 的条目不得被逐出");
        assert!(cache.get(&2).is_none(), "唯一未 pin 的条目应被逐出");
        assert_eq!(cache.total_bytes(), 200);
    }
}
