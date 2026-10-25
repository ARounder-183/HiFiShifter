//! 宿主线程的**身份表**：谁在哪条线程上。
//!
//! 【为什么需要第二个角色】REAPER 的原生扩展 API 只认它自己的 model/UI 线程
//! （`host/reaper` 的五处守卫 + `host::marshal`）。而 ARA 的播放控制请求还有
//! **第二道同类守卫**，钉在 ARA 的 model 线程上：
//!
//! ```text
//! [cmd] stop_audio failed: invalid state: host playback request off model thread
//! ```
//!
//! 那道守卫在 vendored 的 `ara2-bridge-plugin` 里（三重校验、tree 一脏就拒绝编译），
//! 改不得。唯一正确解是从**我们这一侧**把调用投递回它要求的线程 —— 与原生扩展
//! 那条路完全同一个做法，只是目标线程不同。
//!
//! 【为什么 model 线程要自己捕获】`ara2_bridge_core::ModelThread` 只暴露
//! `is_current()`，不给出 `ThreadId`。而 model 线程就是宿主调用我们 ARA 回调的
//! 那条线程，在我们的 `document_controller` 闭包里就能拿到 —— 自己记一份即可，
//! 不必改第三方代码。

use std::thread::ThreadId;

static ARA_MODEL: std::sync::OnceLock<ThreadId> = std::sync::OnceLock::new();

/// 记下 ARA model 线程。由 `runtime::factory` 的 `document_controller` 闭包调用 ——
/// 那是宿主第一次、也是在 model 线程上回调我们的地方。重复调用无副作用。
pub(crate) fn note_ara_model_thread() {
    ARA_MODEL.get_or_init(|| std::thread::current().id());
}

/// ARA model 线程；尚未捕获时为 `None`。
pub(crate) fn ara_model() -> Option<ThreadId> {
    ARA_MODEL.get().copied()
}

/// 在 ARA model 线程上执行 `job`，并让投递方原地等它完成。
///
/// 【为什么未知时直接失败，而不是就地执行】就地执行正是今天的行为，而它已经被
/// 现场日志证明会失败（`host playback request off model thread`）。宁可返回一句
/// 可读的原因，也不要制造一次注定失败的调用。
pub(crate) fn on_ara_model<R: Send>(job: impl FnOnce() -> R) -> Result<R, String> {
    let target = ara_model().ok_or("the ARA model thread is not known yet")?;
    crate::host::marshal::scoped(target, job).map_err(|error| error.message())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 未捕获时必须是**可读的失败**，而不是就地执行。
    #[test]
    fn an_unknown_model_thread_fails_instead_of_running_inline() {
        // 这个测试不依赖全局状态：`on_ara_model` 只在 ARA_MODEL 未初始化时走到这一支，
        // 而测试进程里没有任何 ARA 回调，所以它必然未初始化。
        if ara_model().is_some() {
            return;
        }
        let error = on_ara_model(|| 7u32).unwrap_err();
        assert!(error.contains("model thread"), "unexpected error: {error}");
    }
}
