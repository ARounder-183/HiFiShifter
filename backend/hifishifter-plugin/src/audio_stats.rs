//! 音频回调的**跨格式**读数：panic 计数、慢块计量与写满审计开关。
//!
//! 【为什么这些必须是共享的】它们此前全部长在 `vst3.rs` 里：计数器是 VST3 的私有
//! static，开关是 VST3 的私有函数，诊断包（`editor/gui/client.rs`）直接读
//! `crate::vst3::audio_panic_count()`。于是 CLAP 会话里这三项**永远是 0** ——
//! 不是"CLAP 上没发生过 panic"，而是"CLAP 从来没有计数的人"。诊断包在两种格式下
//! 长得一样、含义却不一样，这比缺一项读数更糟。
//!
//! 【为什么不是 `ExtensionOwner` 的字段】慢块与 panic 是**进程级**事实：它们描述的是
//! 本模块在宿主音频线程上的行为，与"哪个实例、哪份文档"无关。放进 owner 会让每个
//! 实例各记一份，而诊断包只能读到其中一个。`audio_panic_count()` 的调用方
//! （`client.rs` 的 bootstrap 载荷）也没有 owner 上下文。
//!
//! 【音频线程纪律】这里只有原子读写与一次 `OnceLock` 读，**不分配、不加锁、不写文件**。
//! 格式化与落盘由非实时线程完成（见 `ExtensionOwner::log_buffer_audit` 与诊断包）。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// 超过该时长的音频块记一次"慢块"。
///
/// 【为什么是 50 ms】`process` 一秒几百次，不可能逐次记日志；但**它一旦阻塞**，
/// 宿主在关闭窗口或改变尺寸时会等音频线程，表现为整个 UI 冻住 —— 而日志里什么都
/// 没有。50 ms（约两个 1024 帧 @44.1k 的块）足以把"宿主在等音频线程"与正常抖动分开。
pub const AUDIO_SLOW_BLOCK: Duration = Duration::from_millis(50);

/// 音频回调里被 `catch_unwind` 拦下的 panic 次数。
static AUDIO_PANIC_COUNT: AtomicU64 = AtomicU64::new(0);

/// 慢块累计次数。
static AUDIO_SLOW_BLOCK_COUNT: AtomicU64 = AtomicU64::new(0);

/// 慢块的峰值时长（纳秒）。
static AUDIO_SLOW_BLOCK_PEAK_NS: AtomicU64 = AtomicU64::new(0);

/// 记一次被拦下的音频回调 panic（**音频线程**：只做一次原子加）。
///
/// 【为什么是计数而不是日志】音频线程不能写文件、不能抢日志锁（见
/// `realtime_callbacks_do_not_enter_the_file_logger`），而 panic 现场又必须留下痕迹。
/// 计一次数，由下一次 UI 轮询 / 诊断包落盘。
pub fn note_panic() {
    AUDIO_PANIC_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// 音频回调被拦下的 panic 次数（诊断用）。
pub fn panic_count() -> u64 {
    AUDIO_PANIC_COUNT.load(Ordering::Relaxed)
}

/// 音频回调里被拒绝的块数（状态非法 / 并发调用），按原因分开记。
///
/// 【为什么是计数而不是日志】宿主在错误状态下会**逐块**反复调 process；每个块
/// 打一条 `log::error!` 就会在音频线程上抢文件日志锁（见
/// `realtime_callbacks_do_not_enter_the_file_logger`）。VST3 的同等路径也是零日志、
/// 只写原子量，两种格式因此共用这里的进程级读数。
static AUDIO_INVALID_STATE_COUNT: AtomicU64 = AtomicU64::new(0);
static AUDIO_CONCURRENT_PROCESS_COUNT: AtomicU64 = AtomicU64::new(0);

/// 记一次"实例未处于 active/processing 状态"的拒绝（**音频线程**：一次原子加）。
pub fn note_invalid_state_rejection() {
    AUDIO_INVALID_STATE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// 记一次"同一实例并发 process"的拒绝（**音频线程**：一次原子加）。
pub fn note_concurrent_process_rejection() {
    AUDIO_CONCURRENT_PROCESS_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// 被拒绝的块数：`(状态非法, 并发调用)`（诊断用）。
pub fn rejection_stats() -> (u64, u64) {
    (
        AUDIO_INVALID_STATE_COUNT.load(Ordering::Relaxed),
        AUDIO_CONCURRENT_PROCESS_COUNT.load(Ordering::Relaxed),
    )
}

/// 记一次块耗时：超阈值才碰原子量。
pub fn note_block_elapsed(elapsed: Duration) {
    if elapsed < AUDIO_SLOW_BLOCK {
        return;
    }
    AUDIO_SLOW_BLOCK_COUNT.fetch_add(1, Ordering::Relaxed);
    AUDIO_SLOW_BLOCK_PEAK_NS.fetch_max(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

/// 慢块统计：`(次数, 最慢一块的毫秒数)`。
pub fn slow_block_stats() -> (u64, u64) {
    (
        AUDIO_SLOW_BLOCK_COUNT.load(Ordering::Relaxed),
        AUDIO_SLOW_BLOCK_PEAK_NS.load(Ordering::Relaxed) / 1_000_000,
    )
}

/// 输出写满审计是否开启（`HIFISHIFTER_BUFFER_AUDIT=1`）。
///
/// 【为什么用环境变量而不是设置项】它是排查工具，不是产品行为；用户永远不会打开它，
/// 而我们要能在不改构建的情况下让用户跑一遍。读一次就缓存。
///
/// 【为什么两种格式共用这一份】审计的对象是**输出平面有没有被写满**，而输出平面
/// 由共享的 `render_block` 写入 —— 外壳不同，漏写的分支却是同一批。
pub fn buffer_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HIFISHIFTER_BUFFER_AUDIT")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 慢块只在超过阈值时计数，峰值取最大值。
    #[test]
    fn only_blocks_over_the_threshold_are_counted() {
        let before = slow_block_stats().0;
        note_block_elapsed(Duration::from_millis(1));
        assert_eq!(slow_block_stats().0, before, "1 ms 的块不是慢块");
        note_block_elapsed(AUDIO_SLOW_BLOCK + Duration::from_millis(10));
        let (count, peak_ms) = slow_block_stats();
        assert!(count > before);
        assert!(peak_ms >= AUDIO_SLOW_BLOCK.as_millis() as u64);
    }

    /// panic 计数只增不减：它是"发生过多少次"的事实，不是瞬时状态。
    #[test]
    fn panics_accumulate() {
        let before = panic_count();
        note_panic();
        note_panic();
        assert_eq!(panic_count(), before + 2);
    }

    /// 两类拒绝计数互相独立、只增不减：音频线程的拒绝不能靠日志核对。
    #[test]
    fn rejections_are_counted_per_reason() {
        let (invalid_before, concurrent_before) = rejection_stats();
        note_invalid_state_rejection();
        note_invalid_state_rejection();
        note_concurrent_process_rejection();
        let (invalid, concurrent) = rejection_stats();
        assert_eq!(invalid, invalid_before + 2);
        assert_eq!(concurrent, concurrent_before + 1);
    }
}
