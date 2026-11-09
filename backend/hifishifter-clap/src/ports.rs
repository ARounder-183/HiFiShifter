//! `clap.audio-ports`：声明一条主立体声输入与输出，允许 in-place。
use crate::abi::clap_plugin_t;
use std::ffi::c_char;
/// `clap_audio_port_info_t` 的必填布局。
#[repr(C)]
pub struct AudioPortInfo {
    pub id: u32,
    pub name: [c_char; 256],
    pub flags: u32,
    pub channel_count: u32,
    pub port_type: *const c_char,
    pub in_place_pair: u32,
}
#[repr(C)]
pub struct PortsExt {
    pub count: Option<unsafe extern "C" fn(*const clap_plugin_t, bool) -> u32>,
    pub get:
        Option<unsafe extern "C" fn(*const clap_plugin_t, u32, bool, *mut AudioPortInfo) -> bool>,
}
impl Default for PortsExt {
    fn default() -> Self {
        Self::new()
    }
}
impl PortsExt {
    pub const fn new() -> Self {
        Self {
            count: Some(count),
            get: Some(get),
        }
    }
}
unsafe extern "C" fn count(_p: *const clap_plugin_t, _is_input: bool) -> u32 {
    hifishifter_plugin_core::guard_ffi(0, || 1)
}
unsafe extern "C" fn get(
    _p: *const clap_plugin_t,
    index: u32,
    _is_input: bool,
    info: *mut AudioPortInfo,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        if index != 0 || info.is_null() {
            return false;
        }
        let info = unsafe { &mut *info };
        info.id = 0;
        info.name.fill(0);
        // 【为什么不是空串】宿主的路由界面把端口名直接显示给用户；空串在那里就是一条
        // 没有名字的端口，而 VST3 报的是同一个名字（共享常量 `MAIN_BUS_NAME`）。
        for (slot, byte) in info
            .name
            .iter_mut()
            .zip(hifishifter_plugin_core::MAIN_BUS_NAME.bytes())
        {
            *slot = byte as c_char;
        }
        info.flags = 1;
        info.channel_count = 2;
        info.port_type = c"stereo".as_ptr().cast();
        info.in_place_pair = 0;
        true
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for PortsExt {}
unsafe impl Send for PortsExt {}
// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for AudioPortInfo {}
unsafe impl Send for AudioPortInfo {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_one_stereo_port_is_reported() {
        assert_eq!(unsafe { count(std::ptr::null(), true) }, 1);
        let mut info = AudioPortInfo {
            id: 9,
            name: [0; 256],
            flags: 0,
            channel_count: 0,
            port_type: std::ptr::null(),
            in_place_pair: 9,
        };
        assert!(unsafe { get(std::ptr::null(), 0, true, &mut info) });
        assert_eq!(info.channel_count, 2);
        assert!(!unsafe { get(std::ptr::null(), 1, true, &mut info) });
    }
}
