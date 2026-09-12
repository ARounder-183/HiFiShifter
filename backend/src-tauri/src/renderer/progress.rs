//! 渲染进度追踪器：一轮渲染 pass 的跨处理器共享进度通道。
//!
//! 本模块把进度拆成两个正交来源，对所有渲染器一视同仁：
//! - **clip 级**（权威）：渲染循环每完成一个 clip（命中缓存或新渲染入库）
//!   调用 [`advance_clip`]，进度 = 已完成 clip 数 / 本轮 clip 总数；
//! - **clip 内**（细化）：处理器在单个 clip 的渲染过程中调用
//!   [`report_clip_progress`]（local ∈ [0,1]），进度 = (已完成 + local) / 总数。
//!   NSF-HiGAN 按 mel chunk、WORLD 按 6s 合成块各自上报。

/// 进度回调：参数为整体进度（0.0 ~ 1.0）。
pub type ProgressCallback = Box<dyn Fn(f64) + Send + Sync>;

static CALLBACK: OnceLock<Mutex<Option<ProgressCallback>>> = OnceLock::new();
/// 本轮渲染 pass 的 clip 总数（含命中缓存的 clip）。
static TOTAL_CLIPS: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();
/// 已完成（处理出结果）的 clip 数：命中缓存或新渲染入库各计 1。
static DONE_CLIPS: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();

use std::sync::{Mutex, OnceLock};

fn callback_slot() -> &'static Mutex<Option<ProgressCallback>> {
    CALLBACK.get_or_init(|| Mutex::new(None))
}

fn total_slot() -> &'static std::sync::atomic::AtomicUsize {
    TOTAL_CLIPS.get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
}

fn done_slot() -> &'static std::sync::atomic::AtomicUsize {
    DONE_CLIPS.get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
}

/// 设置/清除进度回调。每轮渲染 pass 启动时注册，收尾时清除。
pub fn set_callback(cb: Option<ProgressCallback>) {
    *callback_slot()
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = cb;
}

/// 开始新一轮 pass：记录 clip 总数并把已完成计数归零。
pub fn reset(total_clips: usize) {
    total_slot().store(total_clips, std::sync::atomic::Ordering::Relaxed);
    done_slot().store(0, std::sync::atomic::Ordering::Relaxed);
}

/// 渲染循环在每个 clip 边界调用：已完成计数 +1。
///
/// 只推进计数，不发射事件 —— 是否可见（进度条是否已点亮）由渲染循环的
/// `rendering_started` 门控决定，缓存全命中、无需展示进度的 pass 不应闪进度条。
pub fn advance_clip() {
    done_slot().fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// 当前进度（已完成 clip 数 / 总数）；总数未知时返回 0。
pub fn current_fraction() -> f64 {
    let total = total_slot().load(std::sync::atomic::Ordering::Relaxed);
    if total == 0 {
        return 0.0;
    }
    let done = done_slot().load(std::sync::atomic::Ordering::Relaxed);
    (done as f64 / total as f64).clamp(0.0, 1.0)
}

/// 处理器在单个 clip 渲染过程中上报 clip 内进度（local ∈ [0,1]）。
///
/// 整体进度 = (已完成 clip 数 + local) / 总数。无回调时（导出路径、非渲染
/// pass 上下文）本调用是空操作，开销只有一次锁前的是否存在检查。
pub fn report_clip_progress(local: f64) {
    let Some(slot) = CALLBACK.get() else {
        return;
    };
    let guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    let Some(cb) = guard.as_ref() else {
        return;
    };
    let total = total_slot().load(std::sync::atomic::Ordering::Relaxed);
    let done = done_slot().load(std::sync::atomic::Ordering::Relaxed);
    let progress = if total > 0 {
        ((done as f64 + local.clamp(0.0, 1.0)) / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    cb(progress);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    /// 全局进度状态的用例必须串行执行，避免并行用例互相污染计数器。
    static GLOBAL_LOCK: StdMutex<()> = StdMutex::new(());

    fn reset_for_test(total: usize) {
        set_callback(None);
        reset(total);
    }

    #[test]
    fn clip_progress_is_fraction_of_total() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(4);

        assert_eq!(current_fraction(), 0.0);
        advance_clip();
        assert_eq!(current_fraction(), 0.25);
        advance_clip();
        advance_clip();
        assert_eq!(current_fraction(), 0.75);

        set_callback(None);
    }

    #[test]
    fn local_progress_combines_done_and_local() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(5);

        let seen: Arc<StdMutex<Vec<f64>>> = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        set_callback(Some(Box::new(move |p| sink.lock().unwrap().push(p))));

        advance_clip(); // 第 1 个 clip 完成（done=1）
        report_clip_progress(0.0); // 第 2 个 clip 刚开始 → (1+0)/5
        report_clip_progress(0.5);
        report_clip_progress(1.0); // 第 2 个 clip 收尾 → (1+1)/5

        let events = seen.lock().unwrap().clone();
        assert_eq!(events, vec![0.2, 0.3, 0.4]);

        set_callback(None);
    }

    #[test]
    fn local_progress_clamps_out_of_range_input() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(2);

        let seen: Arc<StdMutex<Vec<f64>>> = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        set_callback(Some(Box::new(move |p| sink.lock().unwrap().push(p))));

        report_clip_progress(-1.0); // 钳到 0 → (0+0)/2
        report_clip_progress(2.0); // 钳到 1 → (0+1)/2
        let events = seen.lock().unwrap().clone();
        assert_eq!(events, vec![0.0, 0.5]);

        set_callback(None);
    }

    #[test]
    fn without_callback_report_is_noop_and_zero_total_is_safe() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(0);

        // total=0：不 panic，进度恒为 0。
        report_clip_progress(0.5);
        assert_eq!(current_fraction(), 0.0);

        // 无回调：report 不 panic（导出路径没有注册回调）。
        reset_for_test(3);
        report_clip_progress(0.5);
        set_callback(None);
    }
}
