use std::path::PathBuf;
use std::sync::Arc;

use crate::state::TimelineState;
use crate::time_stretch::{StretchAlgorithm, UserStretchAlgorithm};

pub(crate) type AudioKey = (PathBuf, u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct StretchKey {
    pub(crate) path: PathBuf,
    pub(crate) out_rate: u32,
    pub(crate) algorithm: UserStretchAlgorithm,
    pub(crate) bpm_q: u32, // 保留字段以兼容 Hash，固定为 0
    pub(crate) trim_start_q: i64,
    pub(crate) trim_end_q: i64,
    pub(crate) playback_rate_q: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct StretchJob {
    pub(crate) key: StretchKey,
    pub(crate) algorithm: StretchAlgorithm,
    pub(crate) source_start_sec: f64,
    pub(crate) source_end_sec: f64,
    pub(crate) playback_rate: f64,
    /// clip 名称，用于向前端推送拉伸进度信息
    pub(crate) clip_name: String,
    /// Tauri app handle，用于 emit 事件
    pub(crate) app_handle: Option<Arc<tauri::AppHandle>>,
}

#[derive(Debug, Clone)]
pub struct AudioEngineStateSnapshot {
    pub is_playing: bool,
    /// "传输层原地等待渲染"：is_playing=true 但位置冻结（等待后台渲染完成
    /// 后自动开始/继续播放）。前端轮询据此跳过时延外推、冻结播放光标。
    pub waiting_for_render: bool,
    pub target: Option<String>,
    pub base_sec: f64,
    pub position_sec: f64,
    pub duration_sec: f64,
    #[allow(dead_code)]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TrackMeterValue {
    pub(crate) peak_linear: f32,
    pub(crate) max_peak_linear: f32,
    pub(crate) clipped: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ResampledStereo {
    pub(crate) sample_rate: u32,
    pub(crate) frames: usize,
    // interleaved stereo f32 in [-1, 1]
    pub(crate) pcm: Arc<Vec<f32>>,
}

impl ResampledStereo {
    /// Approximate heap footprint of the interleaved PCM buffer.
    pub(crate) fn pcm_bytes(&self) -> u64 {
        self.pcm.len() as u64 * std::mem::size_of::<f32>() as u64
    }
}

#[derive(Debug, Clone)]
pub(crate) struct EngineClip {
    pub(crate) clip_id: String,
    #[allow(dead_code)]
    pub(crate) track_id: String,

    pub(crate) start_frame: u64,
    pub(crate) length_frames: u64,

    // Source PCM is always stereo and resampled to engine rate.
    pub(crate) src: ResampledStereo,

    // Source loop bounds in frames (end is exclusive).
    // For timeline clips we repeat within [src_start_frame, src_end_frame).
    // For file playback we do not repeat and treat src_end_frame as a hard end.
    pub(crate) src_start_frame: u64,
    pub(crate) src_end_frame: u64,
    pub(crate) reversed: bool,
    pub(crate) playback_rate: f64,

    // Local (timeline) frame offset applied before sampling the source.
    // Negative values mean leading silence (i.e. slip-edit past the source start).
    pub(crate) local_src_offset_frames: i64,

    pub(crate) repeat: bool,

    /// Loop（循环源）模式：对**整个 src 缓冲**（完整媒体文件）做模运算回绕。
    ///
    /// `Some(anchor)` 时，消费帧数 `src_frame` 的采样位置为
    /// `floor_mod(anchor ± src_frame, src.frames)`（正放 +、倒放 −）。
    /// 锚点是 Clip 进入媒体的起点：正放 = `source_start_sec`，
    /// 倒放 = `source_end_sec`（与既有非 Loop 倒放"从末端向下播放"的约定一致，
    /// 使启用 Loop 的瞬间可见内容保持连续）。此时
    /// `src_start_frame / src_end_frame` 不参与回绕数学。
    ///
    /// `None` 保持旧语义：越界静音（`repeat` 仅作为兼容开关保留）。
    pub(crate) loop_anchor_frame: Option<i64>,

    pub(crate) fade_in_frames: u64,
    pub(crate) fade_out_frames: u64,
    /// 形状化淡化的增益查表（含两端点，FADE_LUT_SIZE+1 项）。
    /// None 时退化为旧线性渐变（长度为 0 的淡化不会进入混音分支）。
    pub(crate) fade_in_lut: Option<Arc<Vec<f32>>>,
    pub(crate) fade_out_lut: Option<Arc<Vec<f32>>>,
    pub(crate) gain: f32,

    /// 预渲染后的 stereo interleaved PCM（优先级最高）。
    /// 当有 pitch edit 时，由后台线程预渲染并填充。
    /// 长度 = clip_length_frames * 2（stereo），采样从 local frame 0 开始。
    pub(crate) rendered_pcm: Option<Arc<Vec<f32>>>,

    /// 可选的独立气声 stem；存在时在 audio callback 中按当前曲线实时混音。
    pub(crate) breath_noise_pcm: Option<Arc<Vec<f32>>>,
    pub(crate) breath_curve: Option<Arc<Vec<f32>>>,
    pub(crate) breath_curve_frame_period_ms: f64,

    /// 可选的 volume 曲线；存在时在 audio callback / mixdown 中逐帧乘到最终输出上。
    pub(crate) volume_curve: Option<Arc<Vec<f32>>>,
    pub(crate) volume_curve_frame_period_ms: f64,

    /// 可选的 pan 曲线；存在时在 audio callback / mixdown 中逐帧应用到左右声道。
    pub(crate) pan_curve: Option<Arc<Vec<f32>>>,
    pub(crate) pan_curve_frame_period_ms: f64,

    /// 该 clip 是否需要 pitch 合成。
    /// - true：需要合成；若 rendered_pcm 为 None，则静音等待渲染完成。
    /// - false：无需合成；直接回退到源 PCM 播放。
    pub(crate) needs_synthesis: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct EngineSnapshot {
    pub(crate) bpm: f64,
    pub(crate) sample_rate: u32,
    pub(crate) duration_frames: u64,
    pub(crate) track_ids: Arc<Vec<String>>,
    pub(crate) clips: Arc<Vec<EngineClip>>,
}

impl EngineSnapshot {
    pub(crate) fn empty(sample_rate: u32) -> Self {
        Self {
            bpm: 120.0,
            sample_rate,
            duration_frames: 0,
            track_ids: Arc::new(vec![]),
            clips: Arc::new(vec![]),
        }
    }
}

#[allow(dead_code)]
pub(crate) enum EngineCommand {
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
    /// 设置 Tauri app handle，使 engine worker 能向前端推送事件。
    SetAppHandle {
        handle: tauri::AppHandle,
    },
    /// 使指定源路径的解码缓存和拉伸缓存失效（源文件被替换时调用）。
    EvictSourcePath {
        path: String,
    },
    /// 更新节拍器配置（开关 / 音量 / 细分模式 / 重音 / 音色）。
    SetMetronome {
        config: crate::audio_engine::metronome::MetronomeConfig,
    },
    /// 换入节拍器响点表（命令层按工程 Tempo Map + 网格预展开）。
    SetMetronomeSchedule {
        clicks: Arc<Vec<crate::audio_engine::metronome::MetronomeClick>>,
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
    /// 用户发起了新的播放请求（play_original 入口处发送）。
    ///
    /// 复位 `stopped_since_play`：完成命令的停止意图判定以"自**本次播放
    /// 请求**以来"为准，而不是"自上一次 set_playing(true) 以来"。前台
    /// 预渲染路径中 set_playing(true) 只发生在渲染完成时 —— 若请求本身
    /// 不复位标志，用户此前任何一次停止（哪怕只是按了一次空格暂停）都会
    /// 让所有后续预渲染完成被永久拒绝：标志唯一清除点是完成命令自身，而
    /// 完成命令恰被它把关（意图死锁，表现为渲染完毕后完全无法播放）。
    /// 命令按序处理：请求先入队、渲染窗口内的用户停止后入队，完成时
    /// 标志仍能正确反映"渲染期间用户是否停止过"。
    BeginPlayIntent,
    /// 前台预渲染线程完成后的"应用时间线并开始播放"。
    ///
    /// 与"先 update_timeline 再 set_playing"两条独立命令的区别：本命令在
    /// worker 内**原子地**检查"用户是否在渲染期间按过停止"——若是则只重建
    /// 快照、不进入播放。两条独立命令存在竞态：停止命令插在渲染线程的
    /// update_timeline 与 set_playing 之间时（Tauri 命令线程池并发 + 渲染
    /// 线程独立推进），set_playing 仍会执行，播放会在用户明确停止后"复活"，
    /// 播放光标随之跳回播放起点（后台预渲染负载下渲染窗口可达数秒，为
    /// 反复播放/暂停/停止时跳变的根源之一）。
    CompletePrerenderAndPlay {
        timeline: TimelineState,
    },
    Stop,
    Shutdown,
}
