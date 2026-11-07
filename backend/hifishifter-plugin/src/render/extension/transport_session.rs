//! 传输会话状态机的转发入口。
//!
//! 【为什么这里只剩一句 re-export】"这是不是同一次播放、何时冻结"是**宿主无关**的
//! 判定（输入只有 playing/rising/host_reset/offline/frames），因此实现搬到了
//! `hifishifter_capture::transport`，VST3 与 CLAP 两个外壳共用同一份。本文件保留
//! 既有路径 `super::transport_session::…`，渲染侧的调用点一行都不用改。
pub(crate) use hifishifter_capture::transport::*;
