//! 进程级 PCM 预算的转发入口。
//!
//! 【为什么这里只剩一句 re-export】预算的持有者已经不只有渲染：捕获运行时要按同样的
//! 规则为源 PCM 与"原始磁带"记账，而它和 VST3 外壳是两个 crate。实现搬到了
//! `hifishifter_capture::budget`（同一份上限、同一份高水位），这里保留既有路径
//! `crate::render::budget::*`，让渲染侧的调用点一行都不用改。
pub(crate) use hifishifter_capture::budget::*;
