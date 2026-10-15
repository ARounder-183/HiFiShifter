//! 编辑后端选择：宿主给了原生扩展就用它，否则按"有没有绑定 ARA"二选一。
//!
//! 【为什么需要一个显式的选择器】此前"没有原生扩展"是一个裸 `None`，调用点到处
//! `is_some_and(...)`，于是"后端是谁"这个问题没有一个统一的答案。这个枚举把
//! "当前后端"收敛成一个值，此后所有能力/身份判断都问它。
//!
//! 【为什么是两个正交的维度】"宿主给了原生扩展吗"与"宿主把本实例绑到 ARA 了吗"
//! 是两个不同的问题，答案组合出三种形态：
//!
//! | 原生扩展 | ARA 绑定 | 后端 |
//! |---|---|---|
//! | 有 | （无所谓） | [`HostBackend::Native`] |
//! | 无 | 有 | [`HostBackend::AraOnly`] |
//! | 无 | 无 | [`HostBackend::Transfer`] |
//!
//! 第二、三行的**唯一**差别是"片段几何从哪来"：ARA 交给我们 region 几何，
//! Transfer 一个片段都不给。混成一个"没有原生扩展"会让这两种形态无法区分，
//! 而它们的用户可见行为完全不同（一个有完整时间线，一个要先录制）。
//!
//! 【为什么这里不出现 REAPER 名字】本模块属于**通用层**（`host/reaper/` 之外）。它
//! 只描述后端种类，具体是哪一家宿主由 [`NativeHost`] 的实现决定。REAPER 专有的
//! 类型 / API / 键名全部封在 `host/reaper/` 内，通用层只认 [`HostKind`] 与能力位。

use hifishifter_kernel::editor::host::{EditorHost, HostCapabilities, HostKind};
use std::sync::Arc;

use super::ara_only::AraOnlyHost;
use super::transfer::TransferHost;
use super::NativeHost;

/// 当前生效的编辑后端。
#[derive(Clone)]
pub(crate) enum HostBackend {
    /// 宿主给出了原生编辑扩展（当前实现是 REAPER）。
    Native(Arc<NativeHost>),
    /// 宿主只提供 ARA 2.0。
    AraOnly(AraOnlyHost),
    /// 宿主既没有原生扩展、也没绑定 ARA：素材靠 Transfer 从输入总线捕获。
    Transfer(TransferHost),
}

impl HostBackend {
    /// 按"宿主有没有给出原生扩展"选择后端。
    ///
    /// 【为什么这里仍退到 `AraOnly` 而不是 `Transfer`】本函数在 `initialize` 期间被调用，
    /// 而那时 ARA 绑定窗口**还开着**（绑定允许发生在 `setActive` 之前）。此刻断言
    /// "没有 ARA"是早的 —— 多数支持 ARA 的宿主正是在这段时间里完成绑定。
    /// 真正的 Transfer 判定由 `ExtensionOwner::host_backend` 在窗口关闭后给出。
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
            HostBackend::Transfer(host) => host.kind(),
        }
    }

    pub(crate) fn capabilities(&self) -> HostCapabilities {
        match self {
            HostBackend::Native(host) => host.capabilities(),
            HostBackend::AraOnly(host) => host.capabilities(),
            HostBackend::Transfer(host) => host.capabilities(),
        }
    }

    /// 宿主应用名（诊断用；**不用于分支**）。
    pub(crate) fn app_name(&self) -> Option<&str> {
        match self {
            HostBackend::Native(host) => host.app_name(),
            HostBackend::AraOnly(host) => host.app_name(),
            // Transfer 后端不知道宿主名：宿主没给原生扩展，也就没有"宿主自称什么"
            // 这一读数。返回 `None` 而不是猜一个 —— 猜出来的名字会进日志与诊断包。
            HostBackend::Transfer(_) => None,
        }
    }

    pub(crate) fn app_version(&self) -> Option<&str> {
        match self {
            HostBackend::Native(host) => host.app_version(),
            HostBackend::AraOnly(host) => host.app_version(),
            HostBackend::Transfer(_) => None,
        }
    }

    /// 底层原生扩展适配器；非原生后端返回 `None`。
    ///
    /// 【为什么还留这条逃生口】宿主专有的写回 / 分割 / 导入路径需要具体类型，
    /// 无法只靠能力位表达。调用点必须先查能力位再用它，但类型上只能这样给出。
    pub(crate) fn native(&self) -> Option<&Arc<NativeHost>> {
        match self {
            HostBackend::Native(host) => Some(host),
            HostBackend::AraOnly(_) | HostBackend::Transfer(_) => None,
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

    /// 三种后端的能力必须**互不相同**，否则"后端是谁"这个问题没有实际含义。
    ///
    /// 【为什么值得一条用例】最容易犯的错是让 Transfer 与 ARA-only 报同一套能力
    /// （例如从 `ara_only()` 复制一份），那样 `HostKind` 的区分就只是装饰：
    /// 前端读到 `clipGeometry: true` 就会去渲染一份不存在的宿主几何。
    #[test]
    fn the_transfer_backend_is_distinguishable_from_ara_only() {
        let ara_only = HostBackend::AraOnly(AraOnlyHost::new());
        let transfer = HostBackend::Transfer(TransferHost::new());
        assert_ne!(ara_only.kind(), transfer.kind());
        assert_ne!(
            ara_only.capabilities(),
            transfer.capabilities(),
            "两者唯一的差别是片段几何来源，但这一差别必须在能力位里可见"
        );
        assert!(ara_only.capabilities().clip_geometry);
        assert!(!transfer.capabilities().clip_geometry);
        assert!(transfer.capabilities().transfer);
        assert!(!ara_only.capabilities().transfer);
    }
}
