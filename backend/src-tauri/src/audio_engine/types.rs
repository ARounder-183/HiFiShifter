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
    Stop,
    Shutdown,
}
