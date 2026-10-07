//! 让**发布线程**而不是音频回调承担旧值的析构。
//!
//! 【要解决的问题】音频回调与命令/渲染线程通过 `ArcSwap` / `ArcSwapOption` 共享
//! 快照（`EngineSnapshot`）与节拍器响点表。发布方换入新值时，回调手里那份旧
//! `Arc` 要等它自己发现"世代变了"才释放 —— 而那一刻它往往已经是**最后一个
//! 持有者**：
//!
//! ```text
//! worker: swap(new) ─→ drop(old)        // 此时回调仍持有 old，引用计数 > 0，不释放
//! rt:     发现换代 ──→ drop(old)        // 引用计数归零 → 在实时线程上析构
//! ```
//!
//! `EngineSnapshot` 里挂着 `Arc<Vec<EngineClip>>`、每个 clip 的 `String` 与若干
//! `Arc<Vec<f32>>` PCM；响点表可以有数万条。这些析构包含多次原子递减与堆释放，
//! 成本不确定 —— 放在实时线程上就是爆音/掉帧的来源。实时线程不允许有非确定性
//! 成本的释放操作。
//!
//! 【为什么不是无锁退休队列】那需要新的原语、溢出策略，以及"队列满时怎么办"
//! 的答案（而队列满时唯一能做的就是就地析构 —— 正是要避免的事）。这里改用
//! 最小手段：**发布方自己多留一小段**。回调在换代后一个音频块内就会释放它那份
//! 引用，因此只要发布方留得比"下一次发布"更久，回调就永远不可能是最后一个
//! 持有者，析构自然落在发布线程上。
//!
//! 【代价】最多多留 `keep` 个已退休的值（有界）。对快照而言这可能意味着多留
//! 几份渲染 PCM，所以 `keep` 取小值。这是"避免实时线程析构"与"多留一点内存"
//! 之间的取舍，取前者。
//!
//! 【为什么不改音频回调】本模块**完全不触碰实时路径** —— 回调里一个字符都不用
//! 改。修复只发生在发布侧，因此不存在"修 RT 问题却引入新 RT 问题"的风险。

/// 暂存刚被换下的引用，让发布线程承担析构（见模块文档）。
pub struct RetirementKeeper<T> {
    retired: Vec<T>,
    keep: usize,
}

impl<T> RetirementKeeper<T> {
    /// `keep` = 除当前值外最多保留几个已退休的值（至少为 1）。
    pub fn new(keep: usize) -> Self {
        let keep = keep.max(1);
        Self {
            // 预留 `keep + 1`：稳态下 `retire` 永不触发重新分配。
            retired: Vec::with_capacity(keep + 1),
            keep,
        }
    }

    /// 记下一个刚被换下的值；超出 `keep` 的最旧值在**调用本方法的线程**上析构。
    pub fn retire(&mut self, value: T) {
        if self.retired.len() >= self.keep {
            self.retired.remove(0);
        }
        self.retired.push(value);
    }

    /// 当前暂存数量（诊断与测试用）。
    pub fn len(&self) -> usize {
        self.retired.len()
    }

    pub fn is_empty(&self) -> bool {
        self.retired.is_empty()
    }
}

impl<T> Default for RetirementKeeper<T> {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::RetirementKeeper;
    use std::sync::Arc;

    /// 退休的值在**下一次发布之前**不能被释放 —— 否则回调仍可能成为最后持有者。
    #[test]
    fn retired_values_outlive_the_next_publish() {
        let mut keeper: RetirementKeeper<Arc<u32>> = RetirementKeeper::new(2);
        let first = Arc::new(1);
        let second = Arc::new(2);

        keeper.retire(first.clone());
        assert_eq!(Arc::strong_count(&first), 2, "刚退休的值必须仍被暂存区持有");

        // 再来一次仍不释放第一个（keep = 2）。
        keeper.retire(second.clone());
        assert_eq!(Arc::strong_count(&first), 2);
        assert_eq!(Arc::strong_count(&second), 2);
    }

    /// 超出 `keep` 的最旧值在**本线程**释放（这才是修复的意义所在）。
    #[test]
    fn eviction_releases_on_the_calling_thread() {
        let mut keeper: RetirementKeeper<Arc<u32>> = RetirementKeeper::new(2);
        let first = Arc::new(1);
        keeper.retire(first.clone());
        keeper.retire(Arc::new(2));
        keeper.retire(Arc::new(3)); // 触发淘汰最旧的 first

        assert_eq!(
            Arc::strong_count(&first),
            1,
            "被淘汰的旧值应当已在调用 retire 的线程上释放"
        );
        assert_eq!(keeper.len(), 2, "暂存数量恒不超过 keep");
    }

    /// `keep` 下界为 1：传 0 不能让暂存区变成"立刻释放"。
    #[test]
    fn keep_is_clamped_to_at_least_one() {
        let mut keeper: RetirementKeeper<Arc<u32>> = RetirementKeeper::new(0);
        let value = Arc::new(1);
        keeper.retire(value.clone());
        assert_eq!(Arc::strong_count(&value), 2);
    }
}
