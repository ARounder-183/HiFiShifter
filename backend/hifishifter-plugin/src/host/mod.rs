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

// 中立门面：通用层（`editor/`、`render/`）只经这里访问宿主原生执行层，不直接命名
// `reaper`。REAPER 专有的 API / 类型字符串 / 键名全部封在 `reaper/` 内。
#[cfg(test)]
pub(crate) use reaper::MockHost;
pub(crate) use reaper::{
    clipboard, host_edit, host_split, undo, HostClipTarget, HostTrackTarget, NativeHost, UiItem,
    UiTrack,
};
