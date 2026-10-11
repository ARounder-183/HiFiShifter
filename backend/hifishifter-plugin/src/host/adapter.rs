//! 编辑后端选择：宿主给了原生扩展就用它，否则退回 ARA-only。
//!
//! 【为什么需要一个显式的选择器】此前"没有原生扩展"是一个裸 `None`，调用点到处
//! `is_some_and(...)`，于是"后端是谁"这个问题没有一个统一的答案。这个枚举把
//! "当前后端"收敛成一个值：插件启动时按"宿主有没有给出原生扩展"二选一，此后所有
//! 能力/身份判断都问它。
//!
//! 【为什么这里不出现 REAPER 名字】本模块属于**通用层**（`host/reaper/` 之外）。它
//! 只描述"原生扩展 vs ARA-only"两种后端，具体是哪一家宿主由 [`NativeHost`] 的实现
//! 决定。REAPER 专有的类型 / API / 键名全部封在 `host/reaper/` 内，通用层只认
//! [`HostKind`] 与能力位。

use hifishifter_kernel::editor::host::{EditorHost, HostCapabilities, HostKind};
use std::sync::Arc;

use super::ara_only::AraOnlyHost;
use super::NativeHost;

/// 当前生效的编辑后端。
#[derive(Clone)]
pub(crate) enum HostBackend {
    /// 宿主给出了原生编辑扩展（当前实现是 REAPER）。
    Native(Arc<NativeHost>),
    /// 宿主只提供 ARA 2.0。
    AraOnly(AraOnlyHost),
}

impl HostBackend {
    /// 按"宿主有没有给出原生扩展"选择后端。
    pub(crate) fn from_native(native: Option<Arc<NativeHost>>) -> Self {
        match native {
            Some(host) => HostBackend::Native(host),
            None => HostBackend::AraOnly(AraOnlyHost::new()),
        }
    }

    pub(crate) fn kind(&self) -> HostKind {
        match self {
            HostBackend::Native(_) => HostKind::Reaper,
            HostBackend::AraOnly(host) => host.kind(),
        }
    }

    pub(crate) fn capabilities(&self) -> HostCapabilities {
        match self {
            HostBackend::Native(host) => host.capabilities(),
            HostBackend::AraOnly(host) => host.capabilities(),
        }
    }

    /// 宿主应用名（诊断用；**不用于分支**）。
    pub(crate) fn app_name(&self) -> Option<&str> {
        match self {
            HostBackend::Native(host) => host.app_name(),
            HostBackend::AraOnly(host) => host.app_name(),
        }
    }

    pub(crate) fn app_version(&self) -> Option<&str> {
        match self {
            HostBackend::Native(host) => host.app_version(),
            HostBackend::AraOnly(host) => host.app_version(),
        }
    }

    /// 底层原生扩展适配器；ARA-only 后端返回 `None`。
    ///
    /// 【为什么还留这条逃生口】宿主专有的写回 / 分割 / 导入路径需要具体类型，
    /// 无法只靠能力位表达。调用点必须先查能力位再用它，但类型上只能这样给出。
    pub(crate) fn native(&self) -> Option<&Arc<NativeHost>> {
        match self {
            HostBackend::Native(host) => Some(host),
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
        assert!(backend.native().is_none());
        let caps = backend.capabilities();
        assert!(caps.clip_geometry && !caps.clip_write);
    }
}
