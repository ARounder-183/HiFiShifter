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
unsafe extern "C" fn set(_p: *const clap_plugin_t, mode: i32) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if mode != CLAP_RENDER_REALTIME && mode != CLAP_RENDER_OFFLINE {
            return false;
        }
        crate::audio::set_render_mode(mode);
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
        assert!(unsafe { set(std::ptr::null(), CLAP_RENDER_REALTIME) });
        assert!(
            !crate::audio::to_transport_frame(&crate::abi::clap_process_t {
                steady_time: 0,
                frames_count: 0,
                transport: std::ptr::null(),
                audio_inputs: std::ptr::null(),
                audio_outputs: std::ptr::null_mut(),
                audio_inputs_count: 0,
                audio_outputs_count: 0,
                in_events: std::ptr::null(),
                out_events: std::ptr::null()
            })
            .offline
        );
    }
}
