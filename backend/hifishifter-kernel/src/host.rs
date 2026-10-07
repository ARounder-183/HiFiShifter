//! 内核 worker 回头找宿主的出口。
//!
//! 【为什么需要独立出口】内核不认识 Tauri，也不认识 ARA 宿主。它需要"往外说几件事"：
//! ① 发 UI 事件（已有 [`crate::events`]）；② 触发/查询宿主侧的后台渲染；
//! ③ 将来还要投递引擎命令。后两者由本模块承载。
//!
//! 【为什么这件事是硬前置，而不是"顺手做"】实测：内核闭包里唯一一处**生产代码**越界，
//! 就是 `pitch_analysis/schedule.rs` 直接调用
//! `crate::commands::playback::{request_background_render, AUTO_BG_RENDER_ENABLED,
//! BG_RENDER_PITCH_PENDING}`。而 `commands` 一旦被拉进来，它自己那 196 处 `tauri::`
//! 与 `recording` / `search` / `system_clipboard` / `linux_clipboard` 就跟着全进来 ——
//! 46 个模块里有 6 个是这么来的。**改掉这一条边，内核闭包才回到 39 个模块。**
//!
//! 【为什么是"具体类型 + 固有方法"而不是裸 trait 对象】调用点的改写量最小：
//! `crate::host::host().request_background_render()` 与原调用形状几乎一样。

use std::sync::{Arc, OnceLock};

use crate::engine_command::EngineCommand;

/// 宿主必须能提供的回调。
///
/// 【设计原则】这里的方法刻意**贴合调用点的语义**，而不是暴露宿主的内部结构。
/// 例如 `take_pitch_pending_flag` 把"读并清零"合成一个方法 —— 因为那本来就是一次
/// 原子操作，拆成 get + clear 会让两个消费者同时看到 `true`（补触发两次）。
pub trait HostCallbacks: Send + Sync + 'static {
    /// 把一条引擎命令投递给设备层。
    ///
    /// 插件侧可以丢弃它，或转成 ARA renderer 的请求 —— 实现方自己决定。
    fn send_engine_command(&self, command: EngineCommand);

    /// 宿主的"自动后台预渲染"开关当前是否启用。
    fn auto_background_render_enabled(&self) -> bool;

    /// 消费"有音高相关的待办"标志：返回它先前的值并清零。
    ///
    /// 【为什么必须原子】调用点的语义是"如果之前置过位，就补触发一次渲染"。
    /// 拆成读 + 清会让并发消费者重复触发。
    fn take_pitch_pending_flag(&self) -> bool;

    /// 请求宿主启动/继续后台渲染。
    ///
    /// 实现方必须自行容忍"宿主不在线"：没有宿主时什么都不做，而不是 panic。
    fn request_background_render(&self);
}

/// 可跨线程持有的宿主回调。
pub type SharedHostCallbacks = Arc<dyn HostCallbacks>;

/// 内核侧的宿主出口。由宿主在装配时 `install` 一次；未安装时所有调用都是静默降级。
pub struct HostServices {
    inner: OnceLock<SharedHostCallbacks>,
}

impl HostServices {
    /// 构造一个空的出口（`const` 以便放进 `static`）。
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// 安装宿主实现。返回 `false` 表示已经装过（**不替换**）。
    pub fn install(&self, callbacks: SharedHostCallbacks) -> bool {
        self.inner.set(callbacks).is_ok()
    }

    /// 宿主是否已装配。
    pub fn is_installed(&self) -> bool {
        self.inner.get().is_some()
    }

    /// 投递一条引擎命令；无宿主时什么都不做。
    pub fn send_engine_command(&self, command: EngineCommand) {
        if let Some(host) = self.inner.get() {
            host.send_engine_command(command);
        }
    }

    /// 自动后台预渲染是否启用；无宿主时按"未启用"处理（静默降级）。
    pub fn auto_background_render_enabled(&self) -> bool {
        self.inner
            .get()
            .is_some_and(|host| host.auto_background_render_enabled())
    }

    /// 消费音高待办标志；无宿主时返回 `false`。
    pub fn take_pitch_pending_flag(&self) -> bool {
        self.inner
            .get()
            .is_some_and(|host| host.take_pitch_pending_flag())
    }

    /// 请求后台渲染；无宿主时什么都不做。
    pub fn request_background_render(&self) {
        if let Some(host) = self.inner.get() {
            host.request_background_render();
        }
    }
}

impl Default for HostServices {
    fn default() -> Self {
        Self::new()
    }
}

/// 进程级的宿主出口。
///
/// 【为什么用进程级单例】内核模块散落在调用栈深处（音高分析的后台线程、
/// 渲染线程的回调），把入口一路当参数传下去会污染几十个签名。宿主在一个进程里
/// 只有一个（要么是 Tauri app 的进程，要么是 DAW 里插件的进程），所以进程级
/// 单例不是偷懒而是贴合事实 —— 与 `app_events` 里的 `AppHandle` 出口同一做法。
pub fn host() -> &'static HostServices {
    static HOST: HostServices = HostServices::new();
    &HOST
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// 记录型的假宿主：把内核发来的调用原样存下来供断言。
    #[derive(Default)]
    struct RecordingHost {
        auto_enabled: AtomicBool,
        pitch_pending: AtomicBool,
        renders_requested: AtomicUsize,
        commands_received: AtomicUsize,
    }

    impl HostCallbacks for RecordingHost {
        fn send_engine_command(&self, _command: EngineCommand) {
            self.commands_received.fetch_add(1, Ordering::Relaxed);
        }
        fn auto_background_render_enabled(&self) -> bool {
            self.auto_enabled.load(Ordering::Relaxed)
        }
        fn take_pitch_pending_flag(&self) -> bool {
            self.pitch_pending.swap(false, Ordering::AcqRel)
        }
        fn request_background_render(&self) {
            self.renders_requested.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 没有宿主时必须是**静默降级**，不是 panic：内核会被单测与插件直接使用，
    /// 那时根本没有宿主。
    #[test]
    fn an_uninstalled_outlet_is_a_silent_no_op() {
        let host = HostServices::new();
        assert!(!host.is_installed());
        assert!(!host.auto_background_render_enabled());
        assert!(!host.take_pitch_pending_flag());
        host.request_background_render();
        host.send_engine_command(EngineCommand::Stop);
    }

    /// 装了宿主就必须原样转发。
    #[test]
    fn an_installed_outlet_forwards_every_call() {
        let host = HostServices::new();
        let recorder = Arc::new(RecordingHost::default());
        recorder.auto_enabled.store(true, Ordering::Relaxed);
        recorder.pitch_pending.store(true, Ordering::Relaxed);
        assert!(host.install(recorder.clone()));

        assert!(host.auto_background_render_enabled());
        assert!(host.take_pitch_pending_flag());
        host.request_background_render();
        host.send_engine_command(EngineCommand::Stop);

        assert_eq!(recorder.renders_requested.load(Ordering::Relaxed), 1);
        assert_eq!(recorder.commands_received.load(Ordering::Relaxed), 1);
    }

    /// 「消费标志」必须只成功一次：第二次读到的必须是 `false`。
    ///
    /// 【为什么单独钉住】调用点用它来决定"是否补触发一次渲染"。若这里退化成
    /// 普通读，两个消费者会各自补触发一次 —— 表现为打开工程时渲染被重复启动。
    #[test]
    fn taking_the_pitch_pending_flag_consumes_it() {
        let host = HostServices::new();
        let recorder = Arc::new(RecordingHost::default());
        recorder.pitch_pending.store(true, Ordering::Relaxed);
        assert!(host.install(recorder.clone()));

        assert!(host.take_pitch_pending_flag(), "第一次应当拿到 true");
        assert!(!host.take_pitch_pending_flag(), "第二次必须已经是 false");
    }

    /// 二次安装必须失败且**不替换**已有宿主。
    #[test]
    fn installing_twice_keeps_the_first_host() {
        let host = HostServices::new();
        let first = Arc::new(RecordingHost::default());
        let second = Arc::new(RecordingHost::default());

        assert!(host.install(first.clone()));
        assert!(!host.install(second.clone()));

        first.auto_enabled.store(true, Ordering::Relaxed);
        assert!(host.auto_background_render_enabled());
        assert!(!second.auto_enabled.load(Ordering::Relaxed));
    }
}
