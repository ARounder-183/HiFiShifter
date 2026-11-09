//! `clap.render`：只把宿主明确给出的 OFFLINE 模式当作离线证据。
use crate::abi::{clap_plugin_t, CLAP_RENDER_OFFLINE, CLAP_RENDER_REALTIME};
#[repr(C)]
pub struct RenderExt {
    pub has_hard_realtime_requirement: Option<unsafe extern "C" fn(*const clap_plugin_t) -> bool>,
    pub set: Option<unsafe extern "C" fn(*const clap_plugin_t, i32) -> bool>,
}
impl Default for RenderExt {
    fn default() -> Self {
        Self::new()
    }
}
impl RenderExt {
    pub const fn new() -> Self {
        Self {
            has_hard_realtime_requirement: Some(hard),
            set: Some(set),
        }
    }
}
unsafe extern "C" fn hard(_p: *const clap_plugin_t) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || false)
}
unsafe extern "C" fn set(plugin: *const clap_plugin_t, mode: i32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if mode != CLAP_RENDER_REALTIME && mode != CLAP_RENDER_OFFLINE {
            return false;
        }
        // 【为什么按实例】一个进程里可能有实时与离线两个实例同时存在；
        // 全局标记会让其中一个读到另一个的模式。
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return false;
        };
        if !instance.audio.set_render_mode(mode) {
            return false;
        }
        // 【为什么这里要重跑 setup 序列】`clap.render.set` 是 CLAP 里"宿主声明这一段
        // 要离线渲染"的唯一时刻，等价于 VST3 的 `setupProcessing(mode=2)`。离线导出
        // 只有**一次**机会：不在第一块之前把带用户曲线的结果准备好，产物里就是静音
        // （VST3 的 setupProcessing 为此同步等待准备完成）。此前这里只记了一个布尔量，
        // 于是 CLAP 的 bounce 从头到尾没有准备过。
        let rate = instance.audio.sample_rate();
        if rate > 0.0 {
            if let Err(error) = instance
                .owner
                .setup_processing(rate, instance.audio.is_offline())
            {
                log::warn!("[clap] offline preparation at render.set failed: {error}");
            }
        }
        true
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for RenderExt {}
unsafe impl Send for RenderExt {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offline_is_not_inferred_from_realtime() {
        let state = crate::audio::AudioState::default();
        assert!(state.set_render_mode(CLAP_RENDER_REALTIME));
        assert!(!state.is_offline());
        assert!(
            !state.set_render_mode(99),
            "unknown render modes must be rejected"
        );
        assert!(!state.is_offline(), "被拒绝的模式不得改动状态");
        assert!(state.set_render_mode(CLAP_RENDER_OFFLINE));
        assert!(state.is_offline());
    }
}
