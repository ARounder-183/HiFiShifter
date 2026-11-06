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
        instance.audio.set_render_mode(mode)
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
        let process = crate::abi::clap_process_t {
            steady_time: 0,
            frames_count: 0,
            transport: std::ptr::null(),
            audio_inputs: std::ptr::null(),
            audio_outputs: std::ptr::null_mut(),
            audio_inputs_count: 0,
            audio_outputs_count: 0,
            in_events: std::ptr::null(),
            out_events: std::ptr::null(),
        };
        assert!(state.set_render_mode(CLAP_RENDER_REALTIME));
        assert!(!state.to_transport_frame(&process).offline);
        assert!(
            !state.set_render_mode(99),
            "unknown render modes must be rejected"
        );
        assert!(!state.to_transport_frame(&process).offline);
    }
}
