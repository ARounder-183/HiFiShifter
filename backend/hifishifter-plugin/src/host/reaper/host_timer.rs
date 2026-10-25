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
/// 心跳计数：用来证明"注册成功"之外，节拍**真的在跑**。
static TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 最近一次心跳遍历到的 owner 数（由 `host_heartbeat` 每拍写入）。
static OWNERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 记录本拍遍历到的 owner 数；在第 200 拍把它并进那一行诊断。
///
/// 【为什么要数它】"心跳在跑"与"心跳采到了样"是两件事。上一版只证明了前者，
/// 而采不到样的心跳和没有心跳的症状完全一样 —— 这一行把两者分开。
pub(crate) fn host_heartbeat_report(owners: usize) {
    OWNERS.store(owners, std::sync::atomic::Ordering::Relaxed);
}

extern "C" fn host_timer_tick() {
    let _ = std::panic::catch_unwind(|| {
        // 【为什么要数节拍并按条打点】`plugin_register("timer")` 返回非零只说明
        // "注册被接受"，不说明宿主真的在调度它。而这条链上任何一环断掉，症状都是
        // 同一个"光标不跟手"。第 1 次与第 200 次（约 6.6 s）各留一行，成本是每次
        // 会话两行，换来的是"节拍到底跑没跑"这个二值答案。
        let ticks = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if ticks == 1 || ticks == 200 {
            let owners = OWNERS.load(std::sync::atomic::Ordering::Relaxed);
            log::warn!(
                "[reaper-host] host timer tick #{ticks} (read by the host's main loop, \
                 {owners} owner(s) sampled)"
            );
        }
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
            // 【为什么是 WARN】`log_line` 是 `log::info!`，而默认级别是 Warn ——
            // 用它记"心跳失败"，用户在日志里只会看到"什么都没有"，与"心跳正常"同形。
            log::warn!(
                "[reaper-host] the host refused the timer registration; \
                 the pump stays on the view callbacks only (the playhead will not follow)"
            );
            return;
        }
        log::warn!("[reaper-host] host-thread pump attached to the host timer");
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
    log::warn!("[reaper-host] host-thread pump detached (last instance released)");
}
