//! vslib 可用性查询。
//!
//! 与 `onnx_status` 同构：把「这个后端在当前构建里到底能不能用」回答给前端，
//! 供算法列表按能力过滤 —— 而不是让前端拿着一份硬编码列表去赌。
//!
//! vslib 是**加载期**链接（导入库，见 `vocoder/vslib.rs` 的 `#[link]`），
//! 所以「不可用」只有两种来源：
//! 1. 编译期没满足条件（未开启 `vslib` feature，或不是 Windows x86_64）；
//! 2. 符号探测失败（`VslibGetVersion` 未返回有效版本号）。

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VslibStatusPayload {
    /// 是否把 vslib 编译进了本构建。
    pub compiled: bool,
    /// 是否可用（compiled 且探测成功）。
    pub available: bool,
    /// 库版本号（探测成功时）。
    pub version: Option<i32>,
    /// 不可用原因，可用时为 null。
    pub error: Option<String>,
}

pub(super) fn get_vslib_status() -> VslibStatusPayload {
    // 条件必须与 `lib.rs` 的 `mod vslib` 完全一致，且带上 `target_arch`：
    // `build.rs` 只对 x86_64-Windows 链接 `vslib_x64`，而 `#[link]` 无条件
    // 存在 —— Windows ARM64 上若把模块编进来，得到的是链接失败，而不是
    // 「不可用」。
    #[cfg(all(feature = "vslib", target_os = "windows", target_arch = "x86_64"))]
    {
        let version = crate::vslib::probe();
        VslibStatusPayload {
            compiled: true,
            available: version.is_some(),
            version,
            error: if version.is_some() {
                None
            } else {
                Some("vslib: VslibGetVersion 未返回有效版本号".to_string())
            },
        }
    }

    #[cfg(not(all(feature = "vslib", target_os = "windows", target_arch = "x86_64")))]
    {
        let reason = if cfg!(feature = "vslib") {
            "vslib: 仅在 Windows x86_64 上可用"
        } else {
            "vslib: 未启用 vslib feature"
        };
        VslibStatusPayload {
            compiled: false,
            available: false,
            version: None,
            error: Some(reason.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 未开启 vslib feature 的构建（或非 Windows x86_64）必须报告不可用，
    /// 且给出原因 —— 前端据此把 vslib 从算法列表里过滤掉。
    #[test]
    #[cfg(not(all(feature = "vslib", target_os = "windows", target_arch = "x86_64")))]
    fn unavailable_without_vslib() {
        let status = get_vslib_status();
        assert!(!status.compiled);
        assert!(!status.available);
        assert!(status.version.is_none());
        assert!(status.error.is_some());
    }
}
