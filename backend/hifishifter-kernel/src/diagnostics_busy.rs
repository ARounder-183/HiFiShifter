//! 「诊断正在进行」的进程级标志：**它存在时音频通路必须静音**。
//!
//! # 它解决的是什么
//!
//! 推理设备基准测试与启动预热都会在后台建 ONNX 会话、跑若干轮真实推理。它们与实时
//! 音频线程抢同一批核，而 ORT 的 CPU EP 默认自旋等待 —— 用户听到的就是断续、爆音、
//! 杂音。事后调线程数只能**降低**概率，给不出"一定不出声"的保证。
//!
//! 所以这里换一条判据：诊断期间**直接静音**，让"会不会有杂音"这个问题在结构上不存在。
//!
//! # 为什么静音而不是停止时钟
//!
//! 只填零、照常推进位置。停钟会让播放头冻住，用户以为程序卡死；而"播放头在走、没有
//! 声音"是任何 DAW 用户都见过、也能立刻理解的形态（导出、离线渲染时就是这样）。
//!
//! # 为什么是计数而不是布尔
//!
//! 预热与基准测试可以重叠（用户在预热还没结束时点了"运行基准测试"）。布尔会被后者
//! 覆盖、被先结束的那个提前清掉，于是静音在半途解除 —— 而那时另一个诊断还在跑。
//! 计数 + RAII guard 让"最后一个离开的人关灯"。
//!
//! # 为什么放在 kernel
//!
//! 写方在 kernel（基准测试、预热），读方在 src-tauri 的音频回调
//! （`audio_engine/mix.rs`，输出静音）与 hifishifter-plugin 的实时输出阶段
//! （`render/extension/playback.rs` 的 `capture_stage`，只压实时流、不动离线渲染）。
//! 两边都依赖 kernel，住在这里才不需要反向依赖。

use std::sync::atomic::{AtomicUsize, Ordering};

/// 当前有多少个诊断正在运行。
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// 是否有诊断正在运行（**实时通路的唯一判据**）。
///
/// 刻意做成无锁读：它会在音频回调里被调用，而那里的每一次取锁都可能变成优先级反转。
pub fn active() -> bool {
    ACTIVE.load(Ordering::Acquire) > 0
}

/// 当前并发诊断数（诊断输出用）。
pub fn depth() -> usize {
    ACTIVE.load(Ordering::Acquire)
}

/// 一次诊断的存活期。Drop 时自动递减，因此不会因为提前 return 或 `?` 而漏掉。
pub struct BusyGuard {
    label: &'static str,
}

impl BusyGuard {
    /// 进入诊断状态。返回的 guard 活多久，静音就持续多久。
    pub fn new(label: &'static str) -> Self {
        let depth = ACTIVE.fetch_add(1, Ordering::AcqRel) + 1;
        log::warn!(
            "[diagnostics] audio muted while '{label}' runs (depth={depth}) — playback continues without sound"
        );
        Self { label }
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        let depth = ACTIVE.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
        log::warn!(
            "[diagnostics] '{}' finished (depth={depth}) — audio output restored",
            self.label
        );
    }
}

/// 进入诊断状态（见 [`BusyGuard`]）。
pub fn enter(label: &'static str) -> BusyGuard {
    BusyGuard::new(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 计数是**可重入**的：两个诊断重叠时，先结束的那个不得解除静音。
    #[test]
    fn overlapping_diagnostics_keep_the_audio_muted_until_the_last_one_ends() {
        assert!(!active());
        let first = enter("prewarm");
        assert!(active());
        let second = enter("benchmark");
        assert_eq!(depth(), 2);
        drop(first);
        assert!(active(), "第二个诊断还在跑，静音不能解除");
        drop(second);
        assert!(!active());
        assert_eq!(depth(), 0);
    }

    /// guard 被 `?` 或提前 return 丢弃时同样会递减（Drop 保证）。
    #[test]
    fn a_dropped_guard_always_releases() {
        {
            let _guard = enter("scoped");
            assert!(active());
        }
        assert!(!active());
    }
}
