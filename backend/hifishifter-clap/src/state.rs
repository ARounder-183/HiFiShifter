//! CLAP state：把宿主的不透明字节块交给 core 的 `ComponentState` 容器。
//!
//! 【为什么 core 认识字节格式】`ComponentState` 是产品存档格式，与插件格式无关；
//! CLAP 与 VST3 共用同一份魔数/版本/三段载荷，跨格式复制粘贴实例仍然可读。

use crate::abi::clap_plugin_t;
use hifishifter_plugin_core::component_state::ComponentState;
use hifishifter_plugin_core::format::FormatState;
use std::ffi::c_void;
use std::sync::Mutex;

/// 一个 CLAP 实例的持久状态。
#[derive(Default)]
pub struct InstanceState(Mutex<Vec<u8>>);

impl FormatState for InstanceState {
    fn load(&self, bytes: &[u8]) -> Result<(), String> {
        // 验证容器可读，再保存原始字节：宿主拿到的与存进来的逐字相同。
        ComponentState::decode(bytes)?;
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = bytes.to_vec();
        Ok(())
    }
    fn save(&self) -> Result<Vec<u8>, String> {
        Ok(self.0.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }
}

/// `clap_ostream_t`。
#[repr(C)]
pub struct clap_ostream_t {
    pub ctx: *mut c_void,
    pub write: Option<unsafe extern "C" fn(*const clap_ostream_t, *const c_void, u64) -> i64>,
}
/// `clap_istream_t`。
#[repr(C)]
pub struct clap_istream_t {
    pub ctx: *mut c_void,
    pub read: Option<unsafe extern "C" fn(*const clap_istream_t, *mut c_void, u64) -> i64>,
}
/// `clap_plugin_state_t`。
#[repr(C)]
pub struct clap_plugin_state_t {
    pub save: Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_ostream_t) -> bool>,
    pub load: Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_istream_t) -> bool>,
}

/// 读取宿主流的全部字节；流返回 0 表示 EOF。
unsafe fn read_all(stream: *const clap_istream_t) -> Vec<u8> {
    let Some(stream) = (unsafe { stream.as_ref() }) else {
        return Vec::new();
    };
    let Some(read) = stream.read else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let count = unsafe { read(stream, chunk.as_mut_ptr().cast(), chunk.len() as u64) };
        if count <= 0 {
            break;
        }
        out.extend_from_slice(&chunk[..(count as usize).min(chunk.len())]);
        if (count as usize) < chunk.len() {
            break;
        }
    }
    out
}

/// 写回流；返回值 -1 表示失败。
unsafe fn write_all(stream: *const clap_ostream_t, bytes: &[u8]) -> bool {
    let Some(stream) = (unsafe { stream.as_ref() }) else {
        return false;
    };
    let Some(write) = stream.write else {
        return false;
    };
    let count = unsafe { write(stream, bytes.as_ptr().cast(), bytes.len() as u64) };
    count == bytes.len() as i64
}

pub static STATE_EXT: clap_plugin_state_t = clap_plugin_state_t {
    save: Some(state_save),
    load: Some(state_load),
};

unsafe extern "C" fn state_save(
    plugin: *const clap_plugin_t,
    stream: *const clap_ostream_t,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return false;
        };
        let Ok(bytes) = instance.state.save() else {
            return false;
        };
        unsafe { write_all(stream, &bytes) }
    })
}

unsafe extern "C" fn state_load(
    plugin: *const clap_plugin_t,
    stream: *const clap_istream_t,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(instance) = crate::instance(plugin) else {
            return false;
        };
        let bytes = unsafe { read_all(stream) };
        instance.state.load(&bytes).is_ok()
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for clap_plugin_state_t {}
unsafe impl Send for clap_plugin_state_t {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn component_state_round_trips_through_the_clap_state_contract() {
        let container = ComponentState {
            edits: vec![1, 2, 3],
            capture: Some(b"{\"takes\":[]}".to_vec()),
            capture_edits: None,
        };
        let bytes = container.encode();
        let state = InstanceState::default();
        assert!(state.load(&bytes).is_ok());
        assert_eq!(state.save().unwrap(), bytes);
    }
    #[test]
    fn a_bare_legacy_blob_is_still_accepted() {
        let state = InstanceState::default();
        assert!(state.load(b"legacy edit state").is_ok());
        assert_eq!(state.save().unwrap(), b"legacy edit state");
    }
}
