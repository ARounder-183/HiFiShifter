pub(crate) mod byte_budget_cache;
mod engine;
mod io;
// `pub(crate)`：`renderer::chain` 的一致性测试需要跨模块调用
// `sample_automation_curve`，以钉住"预览与导出的曲线越界语义一致"。
pub(crate) mod mix;
pub(crate) mod metronome;
mod resource_manager;
pub(crate) mod snapshot;
pub(crate) mod types;
mod util;

pub use engine::AudioEngine;
#[allow(unused_imports)]
pub use types::AudioEngineStateSnapshot;
