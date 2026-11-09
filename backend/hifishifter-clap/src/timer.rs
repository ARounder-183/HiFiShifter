//! `clap.timer-support`：把宿主侧心跳挂到宿主主线程的节拍上。
//!
//! 【为什么必须有它】编辑器命令要从页面线程回到宿主主线程执行（`host::marshal`），
//! 宿主时钟采样 / Undo 收口 / `setDirty` 冲刷也要在**空闲时**继续跑 —— 而 `clap.gui`
//! 的尺寸/焦点回调只在宿主重排窗口时来一次。没有节拍时，一个从不重排的编辑器窗口
//! 会让这些工作全部停摆（表现：播放头不跟随、工程不标脏、宿主写回超时）。
//!
//! 【为什么与 VST3 的 REAPER 定时器是同一份】每次节拍要做的事只有
//! `hifishifter_plugin::editor::host_main_thread_tick()` 这一份实现；这里只回答
//! "CLAP 的节拍从哪里来"。
//!
//! 【宿主没有这个扩展时】如实降级：不发节拍，也不假装成功。视图自己的 20 ms 泵
//! （`gui/windows.rs` 的 WM_TIMER）仍然工作，所以界面可用；只是"没有视图时"的
//! 周期工作不会发生。

use crate::abi::{clap_host_t, clap_plugin_t};
use std::sync::atomic::Ordering;

/// `clap_host_timer_support_t`（clap/timer-support.h）。
#[repr(C)]
pub struct HostTimerSupport {
    pub register_timer: Option<unsafe extern "C" fn(*const clap_host_t, u32, *mut u32) -> bool>,
    pub unregister_timer: Option<unsafe extern "C" fn(*const clap_host_t, u32) -> bool>,
}

/// `clap.timer-support` 扩展 ID。
pub const CLAP_EXT_TIMER_SUPPORT: &[u8] = b"clap.timer-support\0";

/// 节拍周期（毫秒）。与 REAPER 的 `plugin_register("timer")` 约 30 Hz 同频：
/// 再快只是把 UI 线程变成 JSON 工厂，再慢会让播放头明显滞后。
const PERIOD_MS: u32 = 30;

#[repr(C)]
pub struct TimerExt {
    pub on_timer: Option<unsafe extern "C" fn(*const clap_plugin_t, u32)>,
}
impl Default for TimerExt {
    fn default() -> Self {
        Self::new()
    }
}
impl TimerExt {
    pub const fn new() -> Self {
        Self {
            on_timer: Some(on_timer),
        }
    }
}

/// 取宿主的 timer-support 扩展；宿主没提供时返回 `None`。
fn support(host: *const clap_host_t) -> Option<&'static HostTimerSupport> {
    if host.is_null() {
        return None;
    }
    let get_extension = unsafe { (*host).get_extension }?;
    let table = unsafe { get_extension(host, CLAP_EXT_TIMER_SUPPORT.as_ptr().cast()) };
    if table.is_null() {
        return None;
    }
    Some(unsafe { &*table.cast::<HostTimerSupport>() })
}

/// 注册本实例的节拍（幂等）。宿主拒绝时只记一行，不影响激活。
///
/// 【调用点】`plugin.init` —— 不是 `activate`。编辑器可以在未激活的实例上打开，
/// 而投递泵必须在那之前就存在（VST3 的 REAPER 定时器同样挂在宿主适配器建立时）。
/// 注销点是 `plugin.destroy`：失活只是暂时状态，窗口可能还开着。
///
/// 返回 `true` = 宿主接受了注册（节拍已在跑）；`false` = 没有 timer-support 或宿主
/// 拒绝 —— 调用方据此改走 `request_callback` 自挂（见 [`request_main_thread`]）。
///
/// # Safety
/// `plugin` 必须是本模块创建、尚未 destroy 的插件记录（或空指针）。
pub unsafe fn acquire(plugin: *const clap_plugin_t) -> bool {
    let Some(instance) = (unsafe { crate::instance(plugin) }) else {
        return false;
    };
    if instance.timer_id.load(Ordering::Acquire) != u32::MAX {
        return true;
    }
    let Some(support) = support(instance.host) else {
        // 【为什么是 WARN】默认日志级别是 Warn；用 info 记"没有节拍"会让现场与"一切正常"
        // 长得一样，而这正是"播放头不跟随"最难自查的形态。
        log::warn!(
            "[clap] host has no clap.timer-support; falling back to clap_host.request_callback"
        );
        return false;
    };
    let Some(register) = support.register_timer else {
        return false;
    };
    let mut id = u32::MAX;
    // SAFETY: `instance.host` 是本实例存活期内的宿主记录；回调按 CLAP 契约在调用期间有效。
    let ok = unsafe { register(instance.host, PERIOD_MS, &mut id) };
    if !ok || id == u32::MAX {
        log::warn!(
            "[clap] the host refused the timer registration; falling back to request_callback"
        );
        return false;
    }
    instance.timer_id.store(id, Ordering::Release);
    log::warn!("[clap] host-thread pump attached to clap.timer-support (id={id}, {PERIOD_MS} ms)");
    true
}

/// 注销本实例的节拍（幂等）。
///
/// # Safety
/// `plugin` 必须是本模块创建、尚未 destroy 的插件记录（或空指针）。
pub unsafe fn release(plugin: *const clap_plugin_t) {
    let Some(instance) = (unsafe { crate::instance(plugin) }) else {
        return;
    };
    let id = instance.timer_id.swap(u32::MAX, Ordering::AcqRel);
    if id == u32::MAX {
        return;
    }
    if let Some(support) = support(instance.host) {
        if let Some(unregister) = support.unregister_timer {
            // SAFETY: 与 acquire 同一条契约；id 是本次注册拿到的那个。
            unsafe { unregister(instance.host, id) };
        }
    }
    log::info!("[clap] host-thread pump detached (id={id})");
}

/// 请宿主尽快在主线程上回调我们（`clap_host.request_callback`）。
///
/// 【为什么需要它】`clap_plugin.on_main_thread` **只有被请求之后**才会被宿主调用。
/// 宿主没有 `clap.timer-support` 时，这是唯一的周期来源：`on_main_thread` 跑完一次
/// 节拍后自己再要一次，就形成"宿主主线程上的自挂节拍"。有 timer-support 时不用它 ——
/// 那条路更稳定（宿主按固定周期回调，不需要每次重新请求）。
///
/// 【线程】`request_callback` 标了 `[thread-safe]`；本函数因此可以从任意线程调用。
///
/// # Safety
/// `host` 必须是本实例存活期内有效的 `clap_host_t`（或空指针）。
pub unsafe fn request_main_thread(host: *const clap_host_t) {
    if host.is_null() {
        return;
    }
    let Some(callback) = (unsafe { (*host).request_callback }) else {
        return;
    };
    // SAFETY: 见函数契约；`request_callback` 是 [thread-safe] 的。
    unsafe { callback(host) };
}
unsafe extern "C" fn on_timer(plugin: *const clap_plugin_t, timer_id: u32) {
    hifishifter_plugin_core::guard_ffi((), || {
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return;
        };
        if instance.timer_id.load(Ordering::Acquire) != timer_id {
            return;
        }
        // 节拍内容只有一份：投递泵 + 宿主侧周期工作。
        hifishifter_plugin::editor::host_main_thread_tick();
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for TimerExt {}
unsafe impl Send for TimerExt {}
// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for HostTimerSupport {}
unsafe impl Send for HostTimerSupport {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 扩展 ID 必须是 CLAP 头文件里那个字节串（少了 NUL 宿主就查不到）。
    #[test]
    fn the_timer_extension_id_is_nul_terminated() {
        assert_eq!(CLAP_EXT_TIMER_SUPPORT, b"clap.timer-support\0");
        assert!(TimerExt::new().on_timer.is_some());
    }

    /// 空实例上的 acquire/release 必须是无操作，不能崩。
    #[test]
    fn a_null_instance_never_registers_anything() {
        unsafe {
            assert!(!acquire(std::ptr::null()), "空实例不可能挂上节拍");
            release(std::ptr::null());
            request_main_thread(std::ptr::null());
        }
    }
}
