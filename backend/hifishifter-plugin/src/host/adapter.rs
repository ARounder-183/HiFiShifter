//! 编辑后端选择：宿主给了原生扩展就用它，否则退回 ARA-only。
//!
//! 【为什么需要一个显式的选择器】此前"没有 REAPER"是一个裸 `None`，调用点到处
//! `is_some_and(...)`，于是"后端是谁"这个问题没有一个统一的答案。这个枚举把
//! "当前后端"收敛成一个值：插件启动时按"宿主有没有给出原生扩展"二选一，此后所有
//! 能力/身份判断都问它。

use hifishifter_kernel::editor::host::{EditorHost, HostCapabilities, HostKind};
use std::sync::Arc;

use super::ara_only::AraOnlyHost;
use super::reaper::ReaperHost;

/// 当前生效的编辑后端。
#[derive(Clone)]
pub(crate) enum HostBackend {
    /// 宿主给出了原生 REAPER 扩展。
    Reaper(Arc<ReaperHost>),
    /// 宿主只提供 ARA 2.0。
    AraOnly(AraOnlyHost),
}

impl HostBackend {
    /// 按"宿主有没有给出原生扩展"选择后端。
    pub(crate) fn from_native(reaper: Option<Arc<ReaperHost>>) -> Self {
        match reaper {
            Some(host) => HostBackend::Reaper(host),
            None => HostBackend::AraOnly(AraOnlyHost::new()),
        }
    }

    pub(crate) fn kind(&self) -> HostKind {
        match self {
            HostBackend::Reaper(_) => HostKind::Reaper,
            HostBackend::AraOnly(host) => host.kind(),
        }
    }

    pub(crate) fn capabilities(&self) -> HostCapabilities {
        match self {
            HostBackend::Reaper(host) => host.capabilities(),
            HostBackend::AraOnly(host) => host.capabilities(),
        }
    }

    /// 宿主应用名（诊断用；**不用于分支**）。
    pub(crate) fn app_name(&self) -> Option<&str> {
        match self {
            HostBackend::Reaper(host) => host.app_name(),
            HostBackend::AraOnly(host) => host.app_name(),
        }
    }

    pub(crate) fn app_version(&self) -> Option<&str> {
        match self {
            HostBackend::Reaper(host) => host.app_version(),
            HostBackend::AraOnly(host) => host.app_version(),
        }
    }

    /// 底层 REAPER 适配器；ARA-only 后端返回 `None`。
    ///
    /// 【为什么还留这条逃生口】宿主专有的写回 / 分割 / 导入路径需要具体类型，
    /// 无法只靠能力位表达。调用点必须先查能力位再用它，但类型上只能这样给出。
    pub(crate) fn reaper(&self) -> Option<&Arc<ReaperHost>> {
        match self {
            HostBackend::Reaper(host) => Some(host),
            HostBackend::AraOnly(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 没有原生扩展时必须落到 ARA-only，而不是"空后端"。
    #[test]
    fn a_host_without_a_native_extension_falls_back_to_ara_only() {
        let backend = HostBackend::from_native(None);
        assert_eq!(backend.kind(), HostKind::AraOnly);
        assert!(backend.reaper().is_none());
        let caps = backend.capabilities();
        assert!(caps.clip_geometry && !caps.clip_write);
    }
}
