//! `clap.timer-support`：主线程心跳入口（当前不注册定时器，避免无宿主回调时空转）。
use crate::abi::clap_plugin_t;
#[repr(C)]
pub struct TimerExt {
    on_timer: Option<unsafe extern "C" fn(*const clap_plugin_t, u32)>,
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
unsafe extern "C" fn on_timer(plugin: *const clap_plugin_t, _timer_id: u32) {
    hifishifter_plugin_core::guard_ffi((), || {
        if let Some(instance) = unsafe { crate::instance(plugin) } {
            let _ = instance;
        }
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for TimerExt {}
unsafe impl Send for TimerExt {}
