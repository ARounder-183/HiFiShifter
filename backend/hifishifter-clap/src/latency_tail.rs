//! `clap.latency` 与 `clap.tail`：如实声明本外壳的延迟与尾音。
//!
//! 【为什么必须声明】宿主按这两个扩展决定要不要为插件补偿延迟、以及停止播放后
//! 还要继续送多少帧。不声明时某些宿主按"未知"处理，可能给直通效果器留出一段
//! 额外的静音尾。当前 `process` 是逐帧直通、不缓存任何样本，因此**真实值就是 0**；
//! 这不是降级，是唯一正确的答案。阶段 2 接上渲染器后必须回来改这里。
use crate::abi::clap_plugin_t;

/// `clap.latency` 扩展 ID（`clap/ext/latency.h`）。
pub const CLAP_EXT_LATENCY: &[u8] = b"clap.latency\0";
/// `clap.tail` 扩展 ID（`clap/ext/tail.h`）。
pub const CLAP_EXT_TAIL: &[u8] = b"clap.tail\0";

/// `clap_plugin_latency_t`：只有一个 `get`。
#[repr(C)]
pub struct LatencyExt {
    pub get: Option<unsafe extern "C" fn(*const clap_plugin_t) -> u32>,
}
/// `clap_plugin_tail_t`：只有一个 `get`。
#[repr(C)]
pub struct TailExt {
    pub get: Option<unsafe extern "C" fn(*const clap_plugin_t) -> u32>,
}

unsafe extern "C" fn latency_get(_p: *const clap_plugin_t) -> u32 {
    hifishifter_plugin_core::guard_ffi(u32::MAX, || 0)
}
unsafe extern "C" fn tail_get(_p: *const clap_plugin_t) -> u32 {
    hifishifter_plugin_core::guard_ffi(u32::MAX, || 0)
}

/// 直通外壳的延迟表。
pub static LATENCY_EXT: LatencyExt = LatencyExt {
    get: Some(latency_get),
};
/// 直通外壳的尾音表。
pub static TAIL_EXT: TailExt = TailExt {
    get: Some(tail_get),
};

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for LatencyExt {}
unsafe impl Send for LatencyExt {}
// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for TailExt {}
unsafe impl Send for TailExt {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_pass_through_shell_reports_zero_latency_and_tail() {
        assert_eq!(unsafe { latency_get(std::ptr::null()) }, 0);
        assert_eq!(unsafe { tail_get(std::ptr::null()) }, 0);
    }
}
