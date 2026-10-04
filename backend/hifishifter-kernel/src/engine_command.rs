//! 内核与设备层之间的命令词汇表。
//!
//! 【为什么在内核里】内核 worker（音高分析、渲染调度）需要向设备层投递命令，
//! 而设备层（cpal 流、`audio_engine/engine.rs`）是宿主相关的、不进内核。
//! 命令本身是**纯数据**，所以它属于内核：这样内核既不需要认识 cpal，也不需要认识 Tauri。
//!
//! 【为什么没有 `SetAppHandle`】那个变体曾把 `tauri::AppHandle` 塞进命令通道，
//! 使整个枚举无法离开 app 层。现在 engine worker 需要的句柄改从
//! `app_events::app_handle()` 取（它已经是进程级出口），命令通道回到纯数据。

use crate::metronome::{MetronomeClick, MetronomeConfig};
use crate::state::TimelineState;
use crate::time_stretch::UserStretchAlgorithm;
use std::path::PathBuf;
use std::sync::Arc;

/// 解码缓存键：源路径 + 目标采样率。
pub type AudioKey = (PathBuf, u32);

/// 拉伸任务键。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StretchKey {
    pub path: PathBuf,
    pub out_rate: u32,
    pub algorithm: UserStretchAlgorithm,
    /// 保留字段以兼容 `Hash`，固定为 0。
    pub bpm_q: u32,
    pub trim_start_q: i64,
    pub trim_end_q: i64,
    pub playback_rate_q: u32,
}

/// 内核与设备层之间的命令。**纯数据**，不含任何宿主句柄。
#[allow(dead_code)]
pub enum EngineCommand {
    UpdateTimeline(TimelineState),
    SeekSec {
        sec: f64,
    },
    SetPlaying {
        playing: bool,
        target: Option<String>,
    },
    PlayFile {
        path: PathBuf,
        offset_sec: f64,
        target: String,
    },
    StretchReady {
        key: StretchKey,
    },
    AudioReady {
        #[allow(dead_code)]
        key: AudioKey,
    },
    /// clip pitch MIDI 异步预计算完成，触发 snapshot rebuild。
    ClipPitchReady {
        clip_id: String,
    },
    /// 请求 worker 侧为「动态（DYN）」提交后台分析任务。
    ///
    /// 为什么需要它：`schedule_clip_pitch_jobs` 需要 worker 持有的 sender，
    /// 命令层拿不到；而动态是**混音级**参数，未开启合成的轨道同样需要它，
    /// 因此不能挂在 pitch（受 compose_enabled 门控）的调度上。
    ScheduleDynLevelAnalysis,
    /// 使指定源路径的解码缓存和拉伸缓存失效（源文件被替换时调用）。
    EvictSourcePath {
        path: String,
    },
    /// 更新节拍器配置（开关 / 音量 / 细分模式 / 重音 / 音色）。
    SetMetronome {
        config: MetronomeConfig,
    },
    /// 换入节拍器响点表（命令层按工程 Tempo Map + 网格预展开）。
    SetMetronomeSchedule {
        clicks: Arc<Vec<MetronomeClick>>,
    },
    /// 渲染结果已变更（**发送方是渲染线程**）：每处理完一个 Clip（写入缓存
    /// 或命中缓存并注册 key）后立即发送，worker 据此按当前 last_timeline
    /// 重建快照。
    ///
    /// 这是"原地等待渲染"解除的**唯一**机制，取代了此前依赖
    /// RT 上报 → 观察线程轮询比对 → 触发重建的被动链（任一环节漏掉都会让
    /// 等待永久悬空）。推送模型下：产出者发布 → worker 换入新快照 → RT
    /// 回调下一块自动重新判定（就绪即前进），不存在观测窗口与时序竞态。
    RenderedClipsChanged,
    Stop,
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 命令必须能跨线程投递给 device worker。
    ///
    /// 【为什么钉住】`EngineCommand` 是内核 → 设备层的唯一通道，而设备层在另一条线程上。
    /// 一旦某个变体塞进 `Rc` / 裸指针这类非 `Send` 的东西，整个通道就断了。
    #[test]
    fn commands_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<EngineCommand>();
    }
}
