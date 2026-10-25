//! 把「宿主侧心跳」挂到宿主主线程的节拍上：投递泵 + 周期工作。
//!
//! 【为什么必须借宿主的节拍】`IPlugView::getSize` / `canResize` / `onSize` 只在宿主
//! 重排窗口时来一次 —— 编辑器**空闲**时一次都不来，而编辑器命令随时会需要宿主线程。
//! 实测：只靠视图回调，投递十次有九次在 250 ms 后超时。
//!
//! 【为什么周期工作也挂在这里】独立进程形态（Linux 的 `hfs-gui`）下，GTK 定时器
//! 跑在插件自己的线程上，碰不得宿主 API；而 `LinuxHost::host_tick` 是刻意的空实现
//! （在宿主的 `onSize`/`setFrame` 回调栈里回调宿主会把宿主冻住）。于是"宿主时钟采样 /
//! 宿主 Undo 收口 / setDirty 冲刷"这一整类工作此前**没有任何执行者**。宿主节拍是
//! 唯一既能拿到宿主线程、又不在重排回调栈里的入口。
//!
//! REAPER 的 `plugin_register("timer", p)` 正是为此存在的：它把一个 `void(*)()`
//! 挂到主线程上约 30 Hz 的节拍。这比任何自建线程都正确 —— 回调本来就在我们要的
//! 那条线程上。
//!
//! 【为什么是引用计数】多个插件实例（一条轨一个）共用同一个节拍回调；谁先谁后
//! 销毁不确定，注册/注销必须成对。计数归零才注销，避免第一个实例退场时把别人的
//! 泵也拆掉。

use super::PluginRegister;
use crate::sync::MutexExt;
use std::ffi::c_void;

struct TimerState {
    register: Option<PluginRegister>,
    live: usize,
}

static TIMERS: std::sync::Mutex<TimerState> = std::sync::Mutex::new(TimerState {
    register: None,
    live: 0,
});

/// 宿主节拍的回调体。
///
/// 【为什么外面还要 catch_unwind】它从 `extern "C"` 进、由宿主调用 —— panic 越过
/// 那个边界是 abort，也就是 REAPER 整个进程退出。单个任务自己的 panic 已经在
/// `marshal::Ticket::run` 里被拦成一次"任务失败"，这里再兜一层是为了**泵本身**。
extern "C" fn host_timer_tick() {
    let _ = std::panic::catch_unwind(|| {
        // 1) 先排空投递队列：编辑器命令要回到这条线程上执行。
        crate::host::marshal::pump();
        // 2) 再跑宿主侧周期工作。它与情形 A 的 GTK 定时器**共用同一份实现**，
        //    于是"内嵌同线程"与"独立进程"两种形态的宿主侧行为逐字相同。
        crate::editor::gui::host_heartbeat();
    });
}

/// 注册节拍（引用计数 +1）。重复注册同一实例不会重复挂载。
pub(super) fn acquire(register: PluginRegister) {
    let mut state = TIMERS.lock_or_recover();
    state.register = Some(register);
    if state.live == 0 {
        // SAFETY: `plugin_register` 来自核对过的官方头；"timer" 的第二个参数按
        // 约定是一个 `void(*)()`。注册失败时只是没有节拍，泵仍由视图回调驱动。
        let ok = unsafe { (register)(c"timer".as_ptr(), host_timer_tick as *mut c_void) };
        if ok == 0 {
            crate::log_line(
                "[reaper-host] the host refused the timer registration; \
                 the pump stays on the view callbacks only",
            );
            return;
        }
        crate::log_line("[reaper-host] host-thread pump attached to the host timer");
    }
    state.live += 1;
}

/// 注销节拍（引用计数 -1）。计数归零才真正从宿主那里摘掉。
pub(super) fn release() {
    let mut state = TIMERS.lock_or_recover();
    if state.live == 0 {
        return;
    }
    state.live -= 1;
    if state.live > 0 {
        return;
    }
    let Some(register) = state.register.take() else {
        return;
    };
    // SAFETY: 与 `acquire` 同一条约定；注销的必须是**同一个**函数指针。
    unsafe { (register)(c"-timer".as_ptr(), host_timer_tick as *mut c_void) };
}
