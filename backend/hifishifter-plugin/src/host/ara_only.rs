//! ARA-only 编辑后端：宿主只提供 ARA 2.0，没有原生扩展。
//!
//! 【为什么必须是一等后端，而不是"全 `None` 的失败态"】ARA 本身已经给出这条身份通路：
//! `RegionSequence` + region 直接给片段枚举与几何，`read_source_pcm` 直接给音频，
//! VST3 `ProcessContext` 直接给播放位置。缺的只有"写回片段几何"——而 ARA 2.0 的契约
//! **本来就不允许**插件改 region 几何。所以这个后端要做的是**诚实地只读**，而不是
//! 让时间线空白。
//!
//! 【与 REAPER 适配器的关系】两者都实现 [`EditorHost`]。REAPER 适配器在此之上多了
//! item GUID、写回、分割、导入等能力；本后端只报告 ARA 能提供的那部分。

use hifishifter_kernel::editor::host::{EditorHost, HostCapabilities, HostKind};

/// ARA-only 后端。
///
/// 【为什么是无状态单元结构】它能提供的一切（枚举 / 几何 / 音频 / BPM）都已经由
/// ARA 文档与 VST3 上下文给出，插件侧不需要再缓存任何宿主句柄。保持无状态意味着
/// 它没有生命周期，也就没有"后端失效"这一类 bug。
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct AraOnlyHost;

impl AraOnlyHost {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl EditorHost for AraOnlyHost {
    fn kind(&self) -> HostKind {
        HostKind::AraOnly
    }

    fn capabilities(&self) -> HostCapabilities {
        HostCapabilities::ara_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 这是"非 REAPER 宿主上编辑只读"的唯一出处；一旦身份或能力被改坏，
    /// 前端会给出必然失败的写入口。
    #[test]
    fn the_ara_only_backend_is_an_honest_read_only_peer() {
        let host = AraOnlyHost::new();
        assert_eq!(host.kind(), HostKind::AraOnly);
        let caps = host.capabilities();
        assert!(caps.transport && caps.clip_geometry);
        assert!(!caps.clip_write && !caps.split && !caps.item_clipboard);
        assert!(!caps.undo_stack, "ARA-only 走插件自管的本地栈");
        assert!(host.app_name().is_none() && host.app_version().is_none());
    }
}
