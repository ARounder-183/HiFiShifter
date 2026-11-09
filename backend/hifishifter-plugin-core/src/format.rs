//! 格式外壳与共享语义层之间的中性契约。
//!
//! 【为什么每个字段都带 Option】除位置与采样率外，其余时间信息在 CLAP / AU 上都是
//! 「宿主可选给」的：`Some(true)` = 明确在播，`Some(false)` = 宿主给了 transport 但
//! 当前不在播，`None` = 宿主根本没有 transport 概念。三者绝不能合并。

#[cfg(test)]
mod tests;

/// 一个音频块的时间上下文。**刻意不含任何格式 ABI 类型**。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransportFrame {
    /// 工程时间轴上的样本位置（VST3 = projectTimeSamples）。
    pub project_time_samples: i64,
    /// 本块的采样率（Hz）。
    pub sample_rate: f64,
    /// 宿主是否在播放；`None` = 宿主没有 transport 概念。
    pub playing: Option<bool>,
    /// 宿主是否在录音；`None` = 未给该状态。
    pub recording: Option<bool>,
    /// 循环是否激活；`None` = 未给该状态。
    pub cycle_active: Option<bool>,
    /// 速度（BPM）；`None` = 宿主未给有效速度。
    pub tempo: Option<f64>,
    /// 拍号分子；`None` = 宿主未给。
    pub time_sig_num: Option<i32>,
    /// 拍号分母；`None` = 宿主未给。
    pub time_sig_den: Option<i32>,
    /// 稳态/连续时钟；CLAP steady_time 与 VST3 工程位置语义不同，单列。
    pub steady_time: Option<i64>,
    /// 本块是否属于离线 bounce。
    pub offline: bool,
    /// 位置读数的可信度：宿主没给位置时为 false，Capture 必须拒绝录制。
    pub position_trusted: bool,
}

impl TransportFrame {
    /// 明确的播放判定。`Some(false)` 不是「宿主没给」。
    pub fn is_playing(&self) -> bool {
        self.playing == Some(true)
    }

    /// 明确的录音判定。
    pub fn is_recording(&self) -> bool {
        self.recording == Some(true)
    }

    /// 宿主是否表达过 transport 概念。
    pub fn has_transport(&self) -> bool {
        self.playing.is_some()
    }
}

/// 本块的时间基与位置是否足以作为录制坐标。
///
/// 【为什么单独一个纯函数】VST3 边界、CLAP 事件解析与离线 render 都问同一个问题；
/// 只有一处判据才不会出现「一种格式拒绝、另一种格式悄悄按 0 处理」。
pub fn frame_is_trustworthy(frame: &TransportFrame) -> bool {
    frame.position_trusted
        && frame.project_time_samples >= 0
        && frame.sample_rate.is_finite()
        && frame.sample_rate > 0.0
}

// 【为什么这里只有 `TransportFrame`】本模块此前还声明过三个 trait（`FormatState` /
// `GuiAttach` / `AraBinding`），它们的形状来自"每个格式各写一份实现"的时代：GUI 与 ARA
// 绑定各自由外壳实现，state 由外壳缓存字节。现在三件事都收敛到了共享层（`EditorView`、
// `AraBinder`、`ExtensionOwner::encode_state/restore_state`），没有任何类型实现这三个
// trait —— 留着它们只会让下一个人以为"应该在这里再实现一份"，而那正是本轮要消除的分叉。
// 真正中性的东西只剩时间事实：`TransportFrame` 与 `frame_is_trustworthy`。
