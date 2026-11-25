//! 宿主专属扩展薄适配；音频权威仍是ARA，不读取RPP或脚本冒充宿主源。
pub(crate) mod adapter;
pub(crate) mod ara_only;
pub(crate) mod capture;
pub(crate) mod geometry;
pub(crate) mod history;
/// 宿主线程编组（有界等待）。见模块文档：命令跑在插件线程上，而宿主 API
/// 只认它自己的 model/UI 线程。
pub(crate) mod marshal;
pub(crate) mod reaper;
/// 宿主调用的重入守卫与看门狗：让"本线程正在一次宿主调用里"成为可读的事实。
/// 见模块文档 —— ARA 回调据此决定"能不能等"。
pub(crate) mod reentrancy;
/// 采样型停滞看门狗：给"永不返回的宿主调用"一个名字。
///
/// 【为什么与 `reentrancy` 分开】那一层回答"我在不在宿主调用里"，这一层回答
/// "这一次调用已经多久没归还"。前者的证据在 `Drop` 里，而后者的**全部意义**
/// 是那条永远不会走到 `Drop` 的路径。
pub(crate) mod stall_watch;

// 中立门面：通用层（`editor/`、`render/`）只经这里访问宿主原生执行层，不直接命名
// `reaper`。REAPER 专有的 API / 类型字符串 / 键名全部封在 `reaper/` 内。
#[cfg(any(test, feature = "test-support"))]
pub use reaper::MockHost;
pub(crate) use reaper::{
    clipboard, host_edit, host_split, undo, HostClipTarget, HostTrackTarget, NativeHost, UiItem,
    UiTrack,
};
pub(crate) use reentrancy::{in_host_call, AraCallbackGuard, HostCallGuard};
/// 记录"本线程这次宿主调用现在走到哪个阶段"（见 `stall_watch`）。
pub(crate) use stall_watch::mark as mark_progress;
pub(crate) mod work_budget;
