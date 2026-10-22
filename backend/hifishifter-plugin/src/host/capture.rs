//! Capture 编辑后端：宿主既没给原生扩展，也没把本实例绑定到 ARA。
//!
//! 【为什么它必须是一等后端】它是**受支持**的形态，不是失败态。宿主把播放中的音频
//! 从 VST3 输入总线流经插件，插件按 `ProcessContext::projectTimeSamples` 把音频对齐
//! 到自己的时间轴上（见 `docs/plans/2026-10-15-vst-capture-mode.md`）。此后一切
//! 照常：调参、渲染、导出。
//!
//! 【它与 [`super::ara_only::AraOnlyHost`] 的本质差别只有一条】ARA 宿主交给我们
//! region 几何（位置 / 长度 / 源窗），我们只是不能改；Capture 宿主**一个片段都不
//! 告诉我们** —— 时间轴上的片段是录制时按连续性自己切出来的。因此
//! `clip_geometry` 在这一侧是 `false`（见 `HostCapabilities::capture_backend`）。
//!
//! 【与 ARA 的关系】两者都实现 [`EditorHost`]，都报告"能听 / 能渲染 / 能调参 / 能导出"。
//! 差别集中在"片段几何从哪来"这一件事上，所以能力位是唯一需要分叉的地方。

use hifishifter_kernel::editor::host::{EditorHost, HostCapabilities, HostKind};

/// Capture 后端。
///
/// 【为什么是无状态单元结构】与 ARA-only 后端同一个理由：它需要的一切都在别处 ——
/// 素材在 `crate::capture::CaptureStore` 里，播放位置在 `ProcessContext` 里，
/// 工程率在 `setupProcessing` 里。保持无状态意味着它没有生命周期，也就没有
/// "后端失效"这一类 bug。
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CaptureHost;

impl CaptureHost {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl EditorHost for CaptureHost {
    fn kind(&self) -> HostKind {
        HostKind::Capture
    }

    fn capabilities(&self) -> HostCapabilities {
        HostCapabilities::capture_backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 这是"Capture 形态下时间线几何没有宿主来源"的唯一出处。
    ///
    /// 【为什么钉死 `!clip_geometry`】照抄 `ara_only()` 就会写成 `true`，
    /// 而后果不是崩溃：前端会去读一份从来不存在于 Capture 形态的宿主几何，
    /// 表现为"时间线空着但没有任何解释"。这条断言让那次照抄失败在测试期。
    #[test]
    fn the_capture_backend_owns_no_host_clip_geometry() {
        let host = CaptureHost::new();
        assert_eq!(host.kind(), HostKind::Capture);
        let caps = host.capabilities();
        assert!(caps.capture, "这一位就是本后端存在的理由");
        assert!(caps.transport, "工程时间是录制对齐的前提");
        assert!(!caps.clip_geometry, "Capture 宿主不给片段几何");
        assert!(!caps.clip_write && !caps.split && !caps.item_clipboard);
        assert!(!caps.undo_stack, "Capture 走插件自管的本地栈");
        assert!(caps.export_audio && caps.export_data);
        assert!(host.app_name().is_none() && host.app_version().is_none());
    }
}
