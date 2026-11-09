//! `clap.latency` 与 `clap.tail`：如实声明本外壳的延迟与尾音。
//!
//! 【为什么必须声明】宿主按这两个扩展决定要不要为插件补偿延迟、以及停止播放后
//! 还要继续送多少帧。不声明时某些宿主按"未知"处理，可能给直通效果器留出一段
//! 额外的静音尾。
//!
//! 【为什么真实值就是 0（接上渲染器之后仍然如此）】输出通路按**工程位置**从已发布的
//! 快照里取样本（`render/snapshot.rs` 的 `copy_block_at`），既不缓存输入样本，也不
//! 提前/延后一个块 —— 本块第 n 帧对应工程位置 `position + n`。VST3 的
//! `getLatencySamples` / `getTailSamples` 回的是同一个 0（`vst3.rs`），两种格式
//! 因此给出同一个补偿值。
//!
//! 【什么时候必须回来改这里】一旦输出通路开始引入块级缓冲（预读、重采样滤波器组、
//! 或跨块交叉淡化改变了首帧对齐），这里就必须与 VST3 的 `getLatencySamples` 一起改 ——
//! 两个格式报不同的延迟会让同一个工程在两种外壳里听起来错位。
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
