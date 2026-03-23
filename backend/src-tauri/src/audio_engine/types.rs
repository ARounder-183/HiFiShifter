// EngineSnapshot / EngineClip 等音频引擎核心数据类型定义。
// 用于实时播放（mix.rs）和快照构建（snapshot.rs）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::state::TimelineState;

pub(crate) type AudioKey = (PathBuf, u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct StretchKey {
    pub(crate) path: PathBuf,
    pub(crate) out_rate: u32,
    pub(crate) bpm_q: u32, // 保留字段以兼容 Hash，固定为 0
    pub(crate) trim_start_q: i64,
    pub(crate) trim_end_q: i64,
    pub(crate) playback_rate_q: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct StretchJob {
    pub(crate) key: StretchKey,
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
    pub(crate) playback_rate: f64,

    // Local (timeline) frame offset applied before sampling the source.
    // Negative values mean leading silence (i.e. slip-edit past the source start).
    pub(crate) local_src_offset_frames: i64,

    pub(crate) repeat: bool,

    pub(crate) fade_in_frames: u64,
    pub(crate) fade_out_frames: u64,
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

    /// 该 clip 是否需要 pitch 合成。
    /// - true：需要合成；若 rendered_pcm 为 None，则静音等待渲染完成。
    /// - false：无需合成；直接回退到源 PCM 播放。
    pub(crate) needs_synthesis: bool,
}

// ─── per-track VST FX stages（实时路径用） ────────────────────────────────────

/// 单条轨道的 VST 插件实例列表（已加载、按信号流顺序排列）。
/// 在 snapshot 构建阶段创建，在实时 audio callback 中使用 `try_lock` 非阻塞处理。
#[cfg(feature = "vst")]
pub(crate) struct VstTrackStages {
    /// 按信号流顺序排列的 VST 插件实例。
    pub(crate) instances: Vec<Arc<std::sync::Mutex<crate::vst_host::plugin_instance::VstPluginInstance>>>,
}

/// 所有轨道的 VST stages 映射 (track_id → stages)。
#[cfg(feature = "vst")]
pub(crate) type VstStagesMap = HashMap<String, VstTrackStages>;

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct EngineSnapshot {
    pub(crate) bpm: f64,
    pub(crate) sample_rate: u32,
    pub(crate) duration_frames: u64,
    pub(crate) clips: Arc<Vec<EngineClip>>,

    /// Per-track VST FX stages（实时处理用）。
    /// 仅在 `vst` feature 启用时有值。
    #[cfg(feature = "vst")]
    pub(crate) vst_stages: Arc<VstStagesMap>,
}

// 手动实现 Debug（VstTrackStages 不 derive Debug）
impl std::fmt::Debug for EngineSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineSnapshot")
            .field("bpm", &self.bpm)
            .field("sample_rate", &self.sample_rate)
            .field("duration_frames", &self.duration_frames)
            .field("clips_count", &self.clips.len())
            .finish()
    }
}

impl EngineSnapshot {
    pub(crate) fn empty(sample_rate: u32) -> Self {
        Self {
            bpm: 120.0,
            sample_rate,
            duration_frames: 0,
            clips: Arc::new(vec![]),

            #[cfg(feature = "vst")]
            vst_stages: Arc::new(HashMap::new()),
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
    Stop,
    Shutdown,
}
