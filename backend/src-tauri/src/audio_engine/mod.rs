// 已迁到 `hifishifter-kernel`。再导出，`crate::audio_engine::byte_budget_cache::…`
// 与模块内 `super::byte_budget_cache::…` 的路径都保持不变。
pub(crate) use hifishifter_kernel::byte_budget_cache;
mod engine;
mod io;
// `pub(crate)`：`renderer::chain` 的一致性测试需要跨模块调用
// `sample_automation_curve`，以钉住"预览与导出的曲线越界语义一致"。
pub(crate) mod mix;
// 节拍器已迁到 `hifishifter-kernel`（`EngineCommand::SetMetronome` 的载荷类型在那里）。
// 再导出，`crate::audio_engine::metronome::…` 与 `super::metronome::…` 的路径都保持不变。
pub(crate) use hifishifter_kernel::metronome;
mod resource_manager;
pub(crate) mod snapshot;
pub(crate) mod types;
pub(crate) use hifishifter_kernel::util;

pub use engine::AudioEngine;
#[allow(unused_imports)]
pub use types::AudioEngineStateSnapshot;
