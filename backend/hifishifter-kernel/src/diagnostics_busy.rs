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
//! 写方在 kernel（基准测试、预热），读方在 src-tauri（App 实时回调）与
//! hifishifter-plugin（插件 process）。两边都依赖 kernel，住在这里才不需要反向依赖。

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 当前有多少个诊断正在运行。
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// 本线程自己进入了多少层诊断。
    ///
    /// 【为什么需要线程内的读数】`ACTIVE` 是进程级的，它回答不了"**这个**诊断是不是我"。
    /// 而两者必须区分：预热自己就是一个诊断，它要建的正是 ONNX 会话 —— 用进程级读数
    /// 去挡会话构建，预热会拒绝自己。见 [`blocks_session_build`]。
    static LOCAL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

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

/// 本线程自己是否正处在诊断里。
pub fn in_current_thread() -> bool {
    LOCAL_DEPTH.with(|depth| depth.get() > 0)
}

/// "**别人的**诊断正在跑吗" —— 渲染侧据此让出会话构建。
///
/// 【为什么不是直接用 `active()`】预热是一个诊断，而它要做的正是建会话。用 `active()`
/// 判据会让预热拒绝自己（见 [`in_current_thread`]）。真正该让路的情形只有一个：
/// 另一个线程正在跑诊断，而渲染线程要去建 DML 会话 —— 那两种并发都会在驱动层挂起
/// （见 `vocoder_ort_session` 的 `BUILD_LEASE`）。
///
/// 【为什么让路是 fail-soft】渲染侧对构建失败早有既定语义：该 clip 失败、pass 继续
/// 推进、稍后重试。见 `vocoder_ort_session::defer_to_running_diagnostic`。
pub fn blocks_session_build() -> bool {
    active() && !in_current_thread()
}

/// 一次诊断的存活期。Drop 时自动递减，因此不会因为提前 return 或 `?` 而漏掉。
pub struct BusyGuard {
    label: &'static str,
}

impl BusyGuard {
    /// 进入诊断状态。返回的 guard 活多久，静音就持续多久。
    pub fn new(label: &'static str) -> Self {
        let depth = ACTIVE.fetch_add(1, Ordering::AcqRel) + 1;
        LOCAL_DEPTH.with(|local| local.set(local.get().saturating_add(1)));
        log::warn!(
            "[diagnostics] audio muted while '{label}' runs (depth={depth}) — playback continues without sound"
        );
        Self { label }
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        LOCAL_DEPTH.with(|local| local.set(local.get().saturating_sub(1)));
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

    /// 进程级计数与线程内计数在测试间共享：涉及 [`blocks_session_build`] 的用例必须
    /// 串行，否则一个用例的 guard 会让另一个用例读到"别人在诊断"。
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// 计数是**可重入**的：两个诊断重叠时，先结束的那个不得解除静音。
    #[test]
    fn overlapping_diagnostics_keep_the_audio_muted_until_the_last_one_ends() {
        let _serial = serial();
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
        let _serial = serial();
        {
            let _guard = enter("scoped");
            assert!(active());
        }
        assert!(!active());
    }

    /// **不变量：让路只针对"别人的"诊断。**
    ///
    /// 【为什么这是回归测试】判据若写成 `active()`，预热（它自己就是一个诊断，且它要建的
    /// 正是 ONNX 会话）会拒绝自己 —— 表现为"启动后模型永远不可用"。而判据若写成恒 false，
    /// 并发建 DML 会话会在驱动层挂起。两个错法都很贵，所以这条同时钉住两侧。
    #[test]
    fn a_diagnostic_only_makes_room_for_other_threads() {
        let _serial = serial();
        assert!(!blocks_session_build());

        let (ready, wait) = std::sync::mpsc::channel::<()>();
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let _busy = enter("other-thread");
            ready.send(()).expect("signal");
            finished.recv().expect("wait");
        });
        wait.recv().expect("other thread entered");
        assert!(active(), "另一个线程在诊断");
        assert!(
            blocks_session_build(),
            "别的线程在诊断 ⇒ 渲染侧必须让路（否则会与 DML 建会话并发）"
        );
        done.send(()).expect("release");
        handle.join().expect("join");
        assert!(!blocks_session_build(), "诊断结束后必须恢复");
    }

    /// 本线程自己的诊断不算"别人的"：预热据此才能建出它要的会话。
    #[test]
    fn a_diagnostic_on_this_thread_does_not_block_itself() {
        let _serial = serial();
        let _busy = enter("self");
        assert!(active());
        assert!(in_current_thread());
        assert!(!blocks_session_build(), "预热自己就是诊断，不能被自己挡住");
        assert!(
            crate::vocoder_ort_session::defer_to_running_diagnostic().is_ok(),
            "本线程的诊断不得触发让路"
        );
    }
}
