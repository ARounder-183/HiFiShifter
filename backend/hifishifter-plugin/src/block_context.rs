//! 与宿主 ABI 无关的"一个音频块"描述。
//!
//! 【为什么需要它】捕获观察（[`crate::render::extension::ExtensionOwner::capture_observe`]）
//! 要读的是**同一批事实**：这一块的采样率、工程位置、播放/录音位、连续时间、音乐时间、
//! 速度与循环区间。它们此前以 VST3 的 `ProcessContext` 形态直接出现在签名里，于是
//! 任何别的格式想复用同一条逻辑，都必须先伪造一个 VST3 结构 —— 那是把 ABI 当中间语言，
//! 一旦 VST3 SDK 结构变了，CLAP 侧的适配会**静默**跟着错。
//!
//! 现在这一层只描述事实：VST3 侧在 [`BlockContext::from_vst3`] 里读一次位，
//! CLAP 侧从 `clap_event_transport_t` 构造同一份结构，两边之后走的是同一段代码。
//!
//! 【为什么缓冲不在这里】音频平面用 [`hifishifter_capture::recorder::InputBlock`]：
//! 它本来就是格式无关的（`planes: [Option<&[f32]>; 2]` + 起点/帧数/采样率/播放位），
//! 而且捕获逻辑早已只面对它。再造一个平行的块类型只会多一层翻译。

/// 一个音频块的**时间事实**（不含样本）。
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockContext {
    /// 本块的采样率（Hz）。0 表示宿主没给 —— 调用方必须用自己声明的率兜底。
    pub sample_rate: f64,
    /// 工程时间轴上的位置（采样帧）。
    pub project_time_samples: i64,
    /// 宿主处于播放状态。
    pub playing: bool,
    /// 宿主处于录音状态。
    pub recording: bool,
    /// 连续时间轴位置（采样帧）；宿主没声明有效时为 `None`。
    pub continuous: Option<i64>,
    /// 音乐时间轴位置（四分音符）；宿主没声明有效时为 `None`。
    pub music: Option<f64>,
    /// 工程速度（BPM）；宿主没声明有效或值非法时为 `None`。
    pub tempo: Option<f64>,
    /// 循环区间是否有效。
    pub cycle_valid: bool,
    /// 循环此刻是否在走。
    pub cycle_active: bool,
    /// 循环起点（四分音符）。
    pub cycle_start_music: f64,
    /// 循环终点（四分音符）。
    pub cycle_end_music: f64,
}

impl From<&BlockContext> for BlockContext {
    fn from(context: &BlockContext) -> Self {
        *context
    }
}

impl From<&crate::audio_abi::ProcessContext> for BlockContext {
    fn from(context: &crate::audio_abi::ProcessContext) -> Self {
        Self::from_vst3(context)
    }
}

impl BlockContext {
    /// 诊断用：把事实还原成 VST3 的 `state` 位图。
    ///
    /// 【为什么保留这个形状】诊断环里那一列 `state` 是现场日志与既有用例读的字段，
    /// 换成布尔会改变诊断包的格式。位图只在这里重建一次，其它消费方仍只看事实。
    pub fn state_bits(&self) -> u32 {
        use crate::audio_abi::{
            PROCESS_CONTEXT_CONTINUOUS_VALID, PROCESS_CONTEXT_CYCLE_ACTIVE,
            PROCESS_CONTEXT_CYCLE_VALID, PROCESS_CONTEXT_PLAYING,
            PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID, PROCESS_CONTEXT_RECORDING,
            PROCESS_CONTEXT_TEMPO_VALID,
        };
        let mut bits = 0;
        if self.playing {
            bits |= PROCESS_CONTEXT_PLAYING;
        }
        if self.recording {
            bits |= PROCESS_CONTEXT_RECORDING;
        }
        if self.continuous.is_some() {
            bits |= PROCESS_CONTEXT_CONTINUOUS_VALID;
        }
        if self.music.is_some() {
            bits |= PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID;
        }
        if self.tempo.is_some() {
            bits |= PROCESS_CONTEXT_TEMPO_VALID;
        }
        if self.cycle_valid {
            bits |= PROCESS_CONTEXT_CYCLE_VALID;
        }
        if self.cycle_active {
            bits |= PROCESS_CONTEXT_CYCLE_ACTIVE;
        }
        bits
    }

    /// VST3 `ProcessContext` → 中立描述。
    ///
    /// 【为什么集中在这里】VST3 的 `state` 位是本仓库里**唯一**需要按 SDK 语义解码的
    /// 一处；解码之后所有消费方只看布尔与 `Option`，不再各自去按位与。
    pub(crate) fn from_vst3(context: &crate::audio_abi::ProcessContext) -> Self {
        use crate::audio_abi::{
            PROCESS_CONTEXT_CONTINUOUS_VALID, PROCESS_CONTEXT_CYCLE_ACTIVE,
            PROCESS_CONTEXT_CYCLE_VALID, PROCESS_CONTEXT_PLAYING,
            PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID, PROCESS_CONTEXT_RECORDING,
            PROCESS_CONTEXT_TEMPO_VALID,
        };
        let state = context.state;
        let tempo_ok = state & PROCESS_CONTEXT_TEMPO_VALID != 0;
        Self {
            sample_rate: context.sample_rate,
            project_time_samples: context.project_time_samples,
            playing: state & PROCESS_CONTEXT_PLAYING != 0,
            recording: state & PROCESS_CONTEXT_RECORDING != 0,
            continuous: (state & PROCESS_CONTEXT_CONTINUOUS_VALID != 0)
                .then_some(context.continuous_time_samples),
            music: (state & PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID != 0)
                .then_some(context.project_time_music),
            tempo: (tempo_ok && context.tempo.is_finite() && context.tempo > 0.0)
                .then_some(context.tempo),
            cycle_valid: state & PROCESS_CONTEXT_CYCLE_VALID != 0,
            cycle_active: state & PROCESS_CONTEXT_CYCLE_ACTIVE != 0,
            cycle_start_music: context.cycle_start_music,
            cycle_end_music: context.cycle_end_music,
        }
    }
}
