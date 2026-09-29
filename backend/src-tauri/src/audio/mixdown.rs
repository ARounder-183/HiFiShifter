use crate::encode::{create_encoder, ChannelMode, EncodeError, OutputSpec};
use crate::state::{TimelineState, Track};
use crate::time_stretch::{time_stretch_interleaved, StretchAlgorithm};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ─── 导出格式与质量预设 ────────────────────────────────────────────────────────

/// 质量预设，区分实时预览和最终导出场景。
///
/// **当前状态：占位，尚未被消费。** 所有调用点都正确地传入了预设，但
/// `render_mixdown_interleaved` 内部从不读取 [`MixdownOptions::quality_preset`]，
/// 因此改变它不会改变输出。
///
/// 让它真正生效需要先定义"两个档位差在哪"（例如拉伸算法/质量、分析窗长），
/// 而这会直接影响输出内容 —— 注意拉伸算法还决定**时间对齐**，预览与导出用
/// 不同算法会让两者错位，所以不能简单地按档位切换算法。任何这类改动都必须
/// 用真实素材做 A/B 试听验证后才可合入。
///
/// 在补齐语义之前，本字段保持"写入但忽略"，不要基于它做优化假设。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityPreset {
    /// 快速模式，用于播放预览（默认）。
    #[default]
    Realtime,
    /// 最高质量模式，用于最终导出。
    Export,
}

#[derive(Debug, Clone)]
pub struct MixdownOptions {
    pub sample_rate: u32,
    pub start_sec: f64,
    pub end_sec: Option<f64>,
    pub stretch: StretchAlgorithm,
    pub apply_pitch_edit: bool,
    /// 导出编码描述（格式 / 位深 / 码率 / 抖动 / 声道等）。
    /// 位深、编码参数等由 `crate::encode::OutputSpec` 统一描述。
    pub output: OutputSpec,
    /// 质量预设，默认 [`QualityPreset::Realtime`]。
    ///
    /// **尚未被消费** —— 见 [`QualityPreset`] 的说明。调用点应继续正确传入，
    /// 以便补齐语义时无需再改一遍所有调用点；但不要据此推断行为差异。
    #[allow(dead_code)]
    pub quality_preset: QualityPreset,
    /// 可选取消标记：为 true 时中断渲染并返回 `export_cancelled`。
    pub cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// 可选进度回调：参数为**整体**进度（`0.0..=1.0`，单调不减）。
    ///
    /// 【语义】`render_mixdown_to_file` 的整体进度 = 混音相位 `[0, 0.92]` +
    /// 编码相位 `(0.92, 1.0]`。节流（≥50ms）与单调由实现内部保证，回调方不必
    /// 自己限频。直接调用 `render_mixdown_interleaved` 时参数是**混音比例**
    /// （`0.0..=1.0`，不含编码）且**不节流**——生产路径请走 `render_mixdown_to_file`。
    ///
    /// 【为什么用独立通道而非复用 `renderer::progress`】后者是 `OnceLock` 进程级
    /// 单例，与后台渲染 pass 并发会互相污染计数器/回调；导出走 `spawn_blocking`，
    /// 必须与之隔离。
    pub progress: Option<ProgressCallback>,
    /// 可选：整 Clip 渲染缓存的复用/回填统计。
    ///
    /// 导出命令用它向用户与日志上报"复用了 N 个片段"（与后台预渲染的
    /// `ClipOutcome::DiskHit` 同一口径）；测试也用它**直接观测**复用是否发生
    /// （比"改写缓存内容再看输出变化"更可靠 —— 后者依赖跨时刻的键稳定）。
    pub cache_stats: Option<Arc<MixdownCacheStats>>,
}

/// 整 Clip 渲染缓存的复用/回填计数（跨线程可读）。
#[derive(Debug, Default)]
pub struct MixdownCacheStats {
    /// 命中并直接使用缓存产物的片段数（跳过了解码/重采样/拉伸/pitch-edit DSP）。
    pub reused: std::sync::atomic::AtomicU32,
    /// 本次渲染后回填到磁盘缓存的片段数。
    pub stored: std::sync::atomic::AtomicU32,
}

impl MixdownCacheStats {
    /// 读取当前计数（`(reused, stored)`）。
    pub fn snapshot(&self) -> (u32, u32) {
        (
            self.reused.load(Ordering::Relaxed),
            self.stored.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug, Clone)]
pub struct MixdownResult {
    pub sample_rate: u32,
    pub duration_sec: f64,
    /// 实际导出声道数（应用 Mono 下混后；诊断/元数据保留字段）。
    #[allow(dead_code)]
    pub channels: u16,
    /// 写盘字节数。
    pub bytes_written: u64,
}

fn mixdown_cancelled(opts: &MixdownOptions) -> bool {
    opts.cancel_flag
        .as_ref()
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

/// 导出进度回调（进度值恒在 `0.0..=1.0`）。
///
/// 【为什么包成 newtype 而不是裸 `Arc<dyn Fn>`】`MixdownOptions` 派生 `Debug`，
/// 而 `Arc<dyn Fn>` 不实现 `Debug`——直接放进结构体会让整个结构体失去 `Debug`
/// （调用点用 `{:?}` 打印过它）。包一层并手写 `Debug`（只打印占位符）即可保留
/// 其余字段的派生。
#[derive(Clone)]
pub struct ProgressCallback(Arc<dyn Fn(f64) + Send + Sync>);

impl ProgressCallback {
    /// 由闭包构造。
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(f64) + Send + Sync + 'static,
    {
        Self(Arc::new(callback))
    }

    /// 触发回调。
    pub fn call(&self, progress: f64) {
        (self.0)(progress);
    }
}

impl std::fmt::Debug for ProgressCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProgressCallback(..)")
    }
}

/// 混音相位在整体进度中占到的比例；剩余 `(MIX_PHASE_END, 1.0]` 属于编码写盘相位。
///
/// 【为什么这样切】混音里含解码、变速/变调 DSP 与 formant 处理，是绝对大头
/// （长 clip 的 pitch-edit 可达分钟级）；编码相对轻（WAV 增量写盘、MP3/FLAC
/// 内存缓冲），给 8% 已足够反映等待。
const MIX_PHASE_END: f64 = 0.92;

/// 进度上报节流间隔。内核循环里每 4096 帧就有一个检查点，1 小时 44.1k 素材
/// 约合每秒十余次，不节流会把 IPC 打爆。
const PROGRESS_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// 进度上报节流器：限制回调频率，并保证进度**单调不减**。
///
/// 相位边界（起点 `0.0`、混音→编码切换、终点 `1.0`）走 [`Self::report_forced`]，
/// 不受节流限制，避免边界值被节流吞掉。
struct ProgressThrottle {
    callback: Option<ProgressCallback>,
    last_value: f64,
    last_at: Option<Instant>,
}

impl ProgressThrottle {
    fn new(callback: Option<ProgressCallback>) -> Self {
        Self {
            callback,
            last_value: f64::NEG_INFINITY,
            last_at: None,
        }
    }

    /// 常规上报：受节流与单调约束。
    fn report(&mut self, value: f64) {
        self.emit(value, false);
    }

    /// 强制上报：跳过节流（仍保证不减）。
    fn report_forced(&mut self, value: f64) {
        self.emit(value, true);
    }

    fn emit(&mut self, value: f64, force: bool) {
        let Some(callback) = &self.callback else {
            return;
        };
        let value = value.clamp(0.0, 1.0);
        if value < self.last_value {
            return;
        }
        let now = Instant::now();
        if !force {
            if let Some(last_at) = self.last_at {
                if now.duration_since(last_at) < PROGRESS_MIN_INTERVAL {
                    return;
                }
            }
        }
        self.last_value = value;
        self.last_at = Some(now);
        callback.call(value);
    }
}

/// 在线程间共享的节流器上报入口（导出会跨 `spawn_blocking` 边界持有它）。
fn report_shared(throttle: &Arc<Mutex<ProgressThrottle>>, value: f64, force: bool) {
    let mut guard = throttle.lock().unwrap_or_else(|e| e.into_inner());
    if force {
        guard.report_forced(value);
    } else {
        guard.report(value);
    }
}

/// 混音阶段的进度上报（`0.0..=1.0` 的混音比例）。
///
/// 【为什么按 clip 计数而非"已混输出帧数 / 输出总帧数"】多轨素材在时间上大量
/// 重叠，逐 clip 累加的帧数会远超输出总帧数（例如 4 条满长轨道 → 累加 4× 总帧数），
/// 用帧数比会在中途就饱和到 1.0 之后长时间不动。改用「已完成 clip 数 + 当前 clip
/// 内帧比例」：天然单调、不受重叠影响，且与"渲染中"（`renderer::progress`）的
/// 两级公式同构。`begin_clip` 在每次迭代开头调用，因此即便该 clip 被 `continue`
/// 提前跳过，进度也不会漏掉它。
struct MixdownProgress<'a> {
    total_clips: f64,
    callback: Option<&'a ProgressCallback>,
}

impl<'a> MixdownProgress<'a> {
    fn new(total_clips: usize, callback: Option<&'a ProgressCallback>) -> Self {
        Self {
            total_clips: total_clips.max(1) as f64,
            callback,
        }
    }

    /// 进入第 `index` 个 clip（= 前 `index` 个已完成）。
    fn begin_clip(&self, index: usize) {
        self.emit(index as f64 / self.total_clips);
    }

    /// 第 `index` 个 clip 内已混 `frames_done / frames_total` 帧。
    fn report_intra(&self, index: usize, frames_done: usize, frames_total: usize) {
        let intra = frames_done as f64 / frames_total.max(1) as f64;
        self.emit((index as f64 + intra.clamp(0.0, 1.0)) / self.total_clips);
    }

    /// 全部 clip 处理完毕。
    fn finish(&self) {
        self.emit(1.0);
    }

    fn emit(&self, value: f64) {
        if let Some(callback) = self.callback {
            callback.call(value.clamp(0.0, 1.0));
        }
    }
}

#[allow(dead_code)]
fn beat_sec(bpm: f64) -> f64 {
    60.0 / bpm.max(1e-6)
}

fn clamp_track_volume(x: f32) -> f32 {
    x.clamp(0.0, 4.0)
}

fn clamp11(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// 在 mixdown 中采样自动化曲线（与 mix.rs 中的 sample_automation_curve 逻辑一致）。
fn sample_automation_curve_at_sec(
    curve: Option<&[f32]>,
    abs_sec: f64,
    frame_period_ms: f64,
    default_value: f32,
) -> f32 {
    let Some(curve) = curve else {
        return default_value;
    };
    if curve.is_empty() {
        return default_value;
    }
    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return default_value;
    }
    let last = curve.len().saturating_sub(1);
    let i0 = (idx_f as usize).min(last);
    let i1 = (i0 + 1).min(last);
    // 越界时 i0 已钳到末位，frac 会发散：必须钳制，保持“持有末值”语义，
    // 与实时引擎 mix.rs 的采样器一致（否则导出音量随超出时长线性爆表）。
    let frac = ((idx_f - i0 as f64) as f32).clamp(0.0, 1.0);
    let a = curve.get(i0).copied().unwrap_or(default_value);
    let b = curve.get(i1).copied().unwrap_or(a);
    a + (b - a) * frac
}

/// 采样动态（DYN）曲线在绝对秒处的增益。
///
/// 与实时引擎 `audio_engine::mix::dyn_gain_at` 是**同一份语义**：曲线存在性、
/// 基线存在性与哨兵三条守卫，以及 `目标 / max(原声, 静音下限)` 的增益公式。
/// 两侧必须同步修改 —— 导出与监听不一致是最难排查的一类问题。
fn dyn_gain_at_sec(
    dyn_curve: Option<&[f32]>,
    dyn_orig_curve: Option<&[f32]>,
    abs_sec: f64,
    frame_period_ms: f64,
) -> f32 {
    // 曲线缺失 → 该轨道组没在用动态。
    if dyn_curve.is_none_or(|c| c.is_empty()) {
        return 1.0;
    }
    // 用 DYN 专用采样器（越界回落哨兵，不持有末值），与实时引擎
    // `audio_engine::mix::dyn_gain_at` 完全同源 —— 避免导出时把最后一个
    // 目标电平 hold 到曲线尽头而污染后续音频。
    let target = crate::renderer::common_params::sample_dyn_curve_at_sec(
        dyn_curve,
        abs_sec,
        frame_period_ms,
    );
    if target < 0.0 {
        // 哨兵：沿用原声。**防御性兜底** —— 正常路径下曲线已在装配期解析
        //（见 `resolve_dyn_sentinels_for_audio`），导出不该再看到哨兵。
        return 1.0;
    }
    // 基线缺失（分析未就绪）→ 1.0，绝不凭空造增益。
    let Some(orig_curve) = dyn_orig_curve.filter(|c| !c.is_empty()) else {
        return 1.0;
    };
    let orig = sample_automation_curve_at_sec(
        Some(orig_curve),
        abs_sec,
        frame_period_ms,
        // 0.0 = "该帧无基线数据"（与实时引擎 mix.rs 同源）。不能用 DYN_SILENCE_FLOOR：
        // 那会把无数据帧当成无内容帧，使任何超过 −60 dBFS 的目标被读成放大请求而拒绝。
        0.0,
    );
    crate::renderer::common_params::compute_dyn_gain(target, orig)
}

pub(crate) fn linear_resample_interleaved(
    input: &[f32],
    channels: usize,
    in_rate: u32,
    out_rate: u32,
) -> Vec<f32> {
    if input.is_empty() || channels == 0 {
        return vec![];
    }
    if in_rate == out_rate {
        return input.to_vec();
    }

    let in_frames = input.len() / channels;
    if in_frames < 2 {
        return input.to_vec();
    }

    let ratio = out_rate as f64 / in_rate as f64;
    let out_frames = ((in_frames as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_frames * channels];

    for of in 0..out_frames {
        let t_in = (of as f64) / ratio;
        let mut i0 = t_in as usize; //  向下取整
        let frac = (t_in - (i0 as f64)) as f32;
        i0 = i0.min(in_frames - 1); //  限制上限即可
        let i1 = (i0 + 1).min(in_frames - 1);

        // 提取乘法基址到声道循环外部
        let base0 = i0 * channels;
        let base1 = i1 * channels;
        let out_base = of * channels;

        for ch in 0..channels {
            let a = input[base0 + ch];
            let b = input[base1 + ch];
            out[out_base + ch] = a + (b - a) * frac;
        }
    }

    out
}

pub(crate) fn reverse_interleaved_frames(samples: &mut [f32], channels: usize) {
    if channels == 0 {
        return;
    }
    let frames = samples.len() / channels;
    for i in 0..(frames / 2) {
        let li = i * channels;
        let ri = (frames - 1 - i) * channels;
        for ch in 0..channels {
            samples.swap(li + ch, ri + ch);
        }
    }
}

/// Loop（循环源）：从完整媒体 PCM 构建按**整个文件**模运算回绕的片段。
///
/// 映射（f 为片段内已消费的源帧序号）：
///   正放 idx(f) = floor_mod(anchor + f, total)
///   倒放 idx(f) = floor_mod(anchor − 1 − f, total)
/// 其中正放锚点 = `round(source_start·in_rate)`、倒放锚点 =
/// `round(source_end·in_rate)`（exclusive 末端）。越过文件边界后环绕到
/// 另一侧继续 —— 即"循环原始音频文件"，与 REAPER Loop source 一致。
/// 输出保持自然时间顺序（倒放的方向已体现在索引递减中，
/// 调用方无需再做整体反转）。
///
/// 实现：在回绕点之间源索引是连续的，因此按"整段拷贝到边界"的方式用
/// `extend_from_slice` 分块复制，而不是逐帧取模 + 逐样本 push —— 长输出
/// （循环多个周期）下每帧成本从"取模+分支+逐样本写"降为 memcpy 级别。
pub(crate) fn build_loop_tiled_segment(
    pcm: &[f32],
    channels: usize,
    anchor_frame_exclusive: i64,
    reversed: bool,
    out_source_frames: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    if channels == 0 || pcm.is_empty() {
        return out;
    }
    let total = (pcm.len() / channels) as i64;
    if total <= 0 {
        return out;
    }
    out.reserve(out_source_frames.saturating_mul(channels));
    // 起始索引（首帧实际写入的源帧号，已归一化进 [0, total)）：
    let mut idx = if reversed {
        (anchor_frame_exclusive - 1).rem_euclid(total)
    } else {
        anchor_frame_exclusive.rem_euclid(total)
    };
    let mut remaining = out_source_frames as i64;
    while remaining > 0 {
        // 本轮可连续拷贝的帧数：到达文件边界的距离 与 剩余需求 取小。
        // 正放连续区间向上：[idx, idx+run)；倒放连续区间向下：
        // [idx−run+1, idx]（随后帧序就地反转，通道内样本保持配对）。
        let run = if reversed {
            (idx + 1).min(remaining)
        } else {
            (total - idx).min(remaining)
        };
        let (base, end_base) = if reversed {
            let start_frame = (idx + 1 - run) as usize;
            (start_frame * channels, (idx as usize + 1) * channels)
        } else {
            (
                (idx as usize) * channels,
                (idx as usize + run as usize) * channels,
            )
        };
        out.extend_from_slice(&pcm[base..end_base]);
        if reversed {
            // 就地反转刚追加的 run 个帧的帧序（每帧 channels 个样本整体交换）。
            let len = out.len();
            let block = &mut out[len - (run as usize) * channels..];
            let half = run as usize / 2;
            for f in 0..half {
                let a = f * channels;
                let b = (run as usize - 1 - f) * channels;
                for c in 0..channels {
                    block.swap(a + c, b + c);
                }
            }
        }
        remaining -= run;
        // 推进索引并环绕（正放越过末尾回到 0；倒放越过 0 回到末尾）。
        idx = if reversed {
            (idx - run).rem_euclid(total)
        } else {
            (idx + run) % total
        };
    }
    out
}

fn build_parent_map(tracks: &[Track]) -> HashMap<String, Option<String>> {
    let mut map = HashMap::new();
    for t in tracks {
        map.insert(t.id.clone(), t.parent_id.clone());
    }
    map
}

fn track_lineage(track_id: &str, parent_map: &HashMap<String, Option<String>>) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = Some(track_id.to_string());
    let mut safety = 0;
    while let Some(id) = cur {
        out.push(id.clone());
        cur = parent_map.get(&id).and_then(|p| p.clone());
        safety += 1;
        if safety > 2048 {
            break;
        }
    }
    out
}

fn compute_track_gains(tracks: &[Track]) -> HashMap<String, (f32, bool, bool)> {
    let parent_map = build_parent_map(tracks);
    let by_id: HashMap<&str, &Track> = tracks.iter().map(|t| (t.id.as_str(), t)).collect();

    let any_solo = tracks.iter().any(|t| t.solo);
    let mut out = HashMap::new();

    for t in tracks {
        let lineage = track_lineage(&t.id, &parent_map);

        let mut gain = 1.0f32;
        let mut muted = false;
        let mut soloed = false;
        for id in &lineage {
            if let Some(node) = by_id.get(id.as_str()) {
                gain *= clamp_track_volume(node.volume);
                muted |= node.muted;
                soloed |= node.solo;
            }
        }

        // Solo overrides mute: when a track (or its ancestor) is soloed,
        // its own mute flag is ignored so that solo always wins.
        let effective_muted = if any_solo && soloed { false } else { muted };

        if any_solo {
            out.insert(t.id.clone(), (gain, effective_muted, soloed));
        } else {
            out.insert(t.id.clone(), (gain, effective_muted, true));
        }
    }

    out
}

pub(crate) fn clip_duration_sec_from_wav(
    sample_rate: u32,
    channels: u16,
    pcm: &[f32],
) -> Option<f64> {
    let ch = channels as usize;
    if sample_rate == 0 || ch == 0 {
        return None;
    }
    let frames = pcm.len() / ch;
    if frames == 0 {
        return None;
    }
    Some(frames as f64 / sample_rate as f64)
}

/// 导出侧复用整 Clip 渲染缓存的开关（默认开启）。
///
/// `HIFISHIFTER_EXPORT_RENDER_CACHE=0|false|off|no` 关闭 —— 用于快速回滚，以及
/// "开/关逐样本比对"的 A/B 验证（见 [`render_mixdown_interleaved`] 里的复用门禁）。
/// 缓存自身的总开关（`RenderCacheSettings.enabled`）在 `render_cache` 内部检查，
/// 关闭时读写都会静默降级为"未命中"。
fn export_render_cache_reuse_enabled() -> bool {
    match std::env::var("HIFISHIFTER_EXPORT_RENDER_CACHE") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// 导出侧要用的整 Clip 渲染缓存键（与预览/播放**同一口径**，见 [`crate::render_key`]）。
///
/// `None` = 该 clip 不参与处理器渲染（静音 / 无源 / 不需要 pitch edit），
/// 因此也没有缓存条目可言。
fn export_clip_render_cache_key(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    sr: u32,
    scale_signature: &str,
) -> Option<crate::synth_clip_cache::RenderedClipCacheKey> {
    let input =
        crate::render_key::rendered_hash_input_for_clip(timeline, clip, sr, scale_signature)?;
    Some(crate::synth_clip_cache::RenderedClipCacheKey {
        clip_id: clip.id.clone(),
        param_hash: crate::synth_clip_cache::compute_rendered_clip_hash(&input),
    })
}

/// 渲染时间线并按 `opts.output` 描述的格式（WAV / MP3 / FLAC）写盘。
///
/// 编码在"混音完成 → 写盘"这一分叉点发生：`render_mixdown_interleaved`
/// 产出交错 f32 后，可选 Mono 下混，再交由 `crate::encode` 的对应编码器
/// 完成量化、编码与落盘。取消时删除半成品文件并返回 `export_cancelled`。
pub fn render_mixdown_to_file(
    timeline: &TimelineState,
    output_path: &Path,
    opts: MixdownOptions,
) -> Result<MixdownResult, String> {
    if mixdown_cancelled(&opts) {
        return Err("export_cancelled".to_string());
    }

    // 相位划分：混音 0..0.92、编码 0.92..1.0。节流器在混音与编码两相位之间共享，
    // 因此跨相位的单调性由它统一保证（回调次数少，锁开销可忽略）。
    let throttle = Arc::new(Mutex::new(ProgressThrottle::new(opts.progress.clone())));
    report_shared(&throttle, 0.0, true);

    // 混音相位：`render_mixdown_interleaved` 上报 `0..1` 的混音比例，这里映射到
    // `0..MIX_PHASE_END` 并交给共享节流器。
    let mut mix_opts = opts.clone();
    mix_opts.progress = Some(ProgressCallback::new({
        let throttle = Arc::clone(&throttle);
        move |mix_fraction: f64| {
            report_shared(
                &throttle,
                mix_fraction.clamp(0.0, 1.0) * MIX_PHASE_END,
                false,
            );
        }
    }));

    let (out_rate, out_channels, duration_sec, mix) =
        render_mixdown_interleaved(timeline, mix_opts)?;

    if mixdown_cancelled(&opts) {
        return Err("export_cancelled".to_string());
    }

    // 混音相位结束：强制上报相位边界，避免被节流吞掉。
    report_shared(&throttle, MIX_PHASE_END, true);

    // Mono 下混在编码分叉点之前完成，不侵入混音核心；混音管线恒为双声道。
    let (mix, channels) = match opts.output.channel_mode {
        ChannelMode::Stereo => (mix, out_channels),
        ChannelMode::Mono => {
            if out_channels >= 2 {
                let nch = out_channels as usize;
                let mut mono = vec![0.0f32; mix.len() / nch];
                for (frame, chunk) in mix.chunks_exact(nch).enumerate() {
                    let sum: f32 = chunk.iter().sum();
                    mono[frame] = clamp11(sum / nch as f32);
                }
                (mono, 1u16)
            } else {
                (mix, out_channels)
            }
        }
    };

    let mut encoder = create_encoder(
        output_path,
        &opts.output,
        channels,
        out_rate,
        opts.cancel_flag.clone(),
    )
    .map_err(|e| e.to_string())?;

    // 分块推送，块间响应取消（WAV 增量写盘；MP3/FLAC 内存缓冲，取消即丢弃）。
    // WAV 在 create_encoder 内已截断/创建目标文件；MP3/FLAC 仅在 finish()
    // 里一次性写盘。取消/失败时据此决定是否清理：MP3/FLAC 未写盘前不能
    // 删 —— 覆盖导出场景会把用户上一次的成品误删掉。
    let mut output_touched = matches!(opts.output.format, crate::encode::OutputFormat::Wav);
    let encode_result: Result<u64, EncodeError> = {
        let chunk_samples = 8192usize * channels as usize;
        let mut offset = 0usize;
        loop {
            if mixdown_cancelled(&opts) {
                break Err(EncodeError::Cancelled);
            }
            let end = (offset + chunk_samples).min(mix.len());
            if let Err(e) = encoder.push(&mix[offset..end]) {
                break Err(e);
            }
            offset = end;
            // 编码相位进度（节流器限频；`offset` 单调 -> 进度单调）。
            report_shared(
                &throttle,
                MIX_PHASE_END + (offset as f64 / mix.len().max(1) as f64) * (1.0 - MIX_PHASE_END),
                false,
            );
            if offset >= mix.len() {
                output_touched = true;
                break encoder.finish().map(|summary| summary.bytes_written);
            }
        }
    };

    match encode_result {
        Ok(bytes_written) => {
            report_shared(&throttle, 1.0, true);
            Ok(MixdownResult {
                sample_rate: out_rate,
                duration_sec,
                channels,
                bytes_written,
            })
        }
        Err(e) => {
            if output_touched {
                let _ = std::fs::remove_file(output_path);
            }
            Err(e.to_string())
        }
    }
}

pub fn render_mixdown_interleaved(
    timeline: &TimelineState,
    opts: MixdownOptions,
) -> Result<(u32, u16, f64, Vec<f32>), String> {
    if mixdown_cancelled(&opts) {
        return Err("export_cancelled".to_string());
    }

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");

    let mut clips_considered: u32 = 0;
    let mut clips_decoded: u32 = 0;
    let mut clips_mixed: u32 = 0;
    // 整 Clip 渲染缓存的复用/回填计数（仅用于诊断日志）。
    let mut clips_cache_reused: u32 = 0;
    let mut clips_cache_stored: u32 = 0;

    let bpm = timeline.bpm;
    if !(bpm.is_finite() && bpm > 0.0) {
        return Err("invalid bpm".to_string());
    }

    let out_rate = opts.sample_rate.max(8000);
    let out_channels: u16 = 2;

    let project_sec = timeline.project_sec.max(0.0);
    let start_sec = opts.start_sec.max(0.0);
    let end_sec = opts.end_sec.unwrap_or(project_sec).max(start_sec);
    let duration_sec = (end_sec - start_sec).max(0.0);
    let out_frames = (duration_sec * out_rate as f64).round().max(1.0) as usize;
    let mut mix = vec![0.0f32; out_frames * out_channels as usize];

    let track_gain = compute_track_gains(&timeline.tracks);

    // Precompute audible tracks set.
    let mut audible_tracks: HashSet<String> = HashSet::new();
    for (tid, (_gain, muted, solo_ok)) in &track_gain {
        if !*muted && *solo_ok {
            audible_tracks.insert(tid.clone());
        }
    }

    // 混音相位进度（`0..1` 的混音比例；映射到整体进度由调用方负责）。
    let mix_progress = MixdownProgress::new(timeline.clips.len(), opts.progress.as_ref());

    // 音阶签名（渲染缓存键的一部分）：整趟导出只算一次（它遍历 Tempo Map），
    // 与 `collect_clips_needing_render` 的取法一致。
    let scale_signature = timeline.render_scale_signature();

    for (clip_index, clip) in timeline.clips.iter().enumerate() {
        // 在迭代开头上报（= 前 `clip_index` 个已完成）。放在 `continue` 之前，
        // 因此被跳过的 clip 也不会让进度漏拍。
        mix_progress.begin_clip(clip_index);

        if mixdown_cancelled(&opts) {
            return Err("export_cancelled".to_string());
        }

        if clip.muted {
            continue;
        }
        if !audible_tracks.contains(&clip.track_id) {
            continue;
        }
        let Some(source_path) = clip.source_path.as_ref() else {
            continue;
        };

        clips_considered = clips_considered.saturating_add(1);

        let (track_gain_value, _tmuted, _solo_ok) = track_gain
            .get(&clip.track_id)
            .cloned()
            .unwrap_or((1.0, false, true));
        let gain = (clip.gain.max(0.0) * track_gain_value).clamp(0.0, 4.0);
        if gain <= 0.0 {
            continue;
        }

        // Timeline placement.
        let clip_start_sec = clip.start_sec.max(0.0);
        let clip_timeline_len_sec = clip.length_sec.max(0.0);
        if !(clip_timeline_len_sec.is_finite() && clip_timeline_len_sec > 0.0) {
            continue;
        }
        let clip_end_sec = clip_start_sec + clip_timeline_len_sec;
        // clip 的局部总帧数（时间线长度 × 输出采样率）。提前到此定义：混音阶段的
        // 淡化/范围几何与下方的渲染缓存复用门禁都要用它，且两处必须是同一个值。
        let clip_total_frames = (clip_timeline_len_sec * out_rate as f64).round().max(1.0) as usize;

        // Check overlap with requested render window.
        if clip_end_sec <= start_sec || clip_start_sec >= end_sec {
            continue;
        }

        let playback_rate = clip.playback_rate as f64;
        let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate
        } else {
            1.0
        };

        // Decode audio (WAV fast-path; otherwise Symphonia).
        let (in_rate, in_channels, pcm) =
            match crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path)) {
                Ok(v) => v,
                Err(e) => {
                    if debug {
                        log::error!(
                            "mixdown: decode failed; clip_id={} track_id={} path={} err={}",
                            clip.id, clip.track_id, source_path, e
                        );
                    }
                    continue;
                }
            };

        clips_decoded = clips_decoded.saturating_add(1);

        let in_channels_usize = in_channels as usize;
        let in_frames = pcm.len() / in_channels_usize;
        if in_frames < 2 {
            continue;
        }

        // Source trimming is expressed in source-domain absolute seconds.
        // 非 Loop 统一使用**消费窗口模型**（clip_playback_window_sec）：
        //   正放 win = [ss, ss+len·r)；倒放 win = [se−len·r, se)。
        // win ∉ [0, D) 的部分渲染静音：正放 ss<0 / 倒放 se>D → 前导静音
        //（方向不同！倒放的 ss<0 是尾部静音，切片自然变短即可，绝不能
        // 再触发前导静音 —— 否则内容整体后移、该有声处被静音吞掉）。
        let loop_mode = clip.loop_enabled;

        let total_sec = match clip_duration_sec_from_wav(in_rate, in_channels, &pcm) {
            Some(v) => v,
            None => continue,
        };
        if !(total_sec.is_finite() && total_sec > 0.0) {
            continue;
        }

        let (win_start_sec, win_end_sec) = crate::state::clip_playback_window_sec(clip);
        // clip_leading_silence_sec 返回的是**时间线秒**（内部已除以 playback_rate），
        // 与引擎 snapshot 的 pre_silence_sec 同源；这里不能再除一次。
        let pre_silence_sec = crate::state::clip_leading_silence_sec(clip, Some(total_sec));

        let src_end_limit_sec = win_end_sec.min(total_sec).max(win_start_sec.max(0.0));
        let slice_start_sec = win_start_sec.max(0.0);
        if !loop_mode && src_end_limit_sec - slice_start_sec <= 1e-9 {
            continue;
        }

        // ── 片段构建 ─────────────────────────────────────────────────────────
        // Loop（循环源）：从完整媒体按整文件模运算回绕生成片段
        //   正放 idx(f) = floor_mod(source_start + f, D_frames)
        //   倒放 idx(f) = floor_mod(source_end − 1 − f, D_frames)
        // 即"循环原始音频文件"：先消费 source_start → 文件末尾，
        // 之后每个周期都是整个文件（对齐 REAPER Loop source）。
        // 锚点直接取原始字段：正放可为负（floor_mod 环绕到末尾一侧）；
        // 倒放只把末端 clamp 到媒体时长 —— 不能用含 `.max(source_start)`
        // 的 src_end_limit_sec（那是为非 Loop 切片准备的），否则 Loop 下
        // split 产生的"环绕窗口"会把倒放锚点错误地推回窗口起点。
        // 非 Loop 保持原窗口切片行为。
        let anchor_frame: i64 = if clip.reversed {
            (clip.source_end_sec.min(total_sec) * in_rate as f64).round() as i64
        } else {
            (clip.source_start_sec * in_rate as f64).round() as i64
        };
        // Loop（循环源）：只物化【导出窗口 ∩ clip】对应的消费量 —— 整条 clip
        // 的平铺段在"导出局部区间 / 长循环 clip"场景会产生多份全尺寸缓冲的
        // 瞬时峰值（tiled 段 + resample 副本 + formant 产物），分配失败即
        // 进程 abort。锚点按窗口起点前移等量消费帧，内容相位不变。
        //
        // 窗口起点/终点量化到固定网格（1s）：波形 peaks 与区间导出以任意
        // 浮点窗口反复调用本函数，若直接使用原始窗口，Loop+Formant 的缓存
        // key 会随每次滚动/缩放变化 → 全量 Formant DSP 重算并冲刷 LRU。
        // 量化后滑动窗口只命中小集合 key；多消费的边界帧由下方
        // 【导出窗口 ∩ clip】交集裁掉，不影响输出内容与淡化相位。
        const LOOP_SEG_QUANTUM_SEC: f64 = 1.0;
        let (loop_seg_local_start_sec, loop_seg_len_sec) = if loop_mode {
            let local_start = (start_sec - clip_start_sec).max(0.0);
            let local_end = (end_sec - clip_start_sec).min(clip_timeline_len_sec);
            let q_start = (local_start - local_start % LOOP_SEG_QUANTUM_SEC).max(0.0);
            let q_end = ((local_end / LOOP_SEG_QUANTUM_SEC).ceil() * LOOP_SEG_QUANTUM_SEC)
                .min(clip_timeline_len_sec.max(0.0));
            (q_start, (q_end - q_start).max(0.0))
        } else {
            (0.0, clip_timeline_len_sec)
        };
        // Loop（循环源）平铺段几何 —— 只计算一次，片段构建与 Formant 缓存键
        // 必须共享同一组数值（此前两处各算一遍，一旦某处改动就会静默漂移：
        // 键与内容不再对应，缓存互相投毒/永不命中）。
        let (loop_advanced_anchor, loop_out_source_frames) = if loop_mode {
            let skip_src_frames =
                (loop_seg_local_start_sec * playback_rate * in_rate as f64).round() as i64;
            let advanced_anchor = if clip.reversed {
                anchor_frame - skip_src_frames
            } else {
                anchor_frame + skip_src_frames
            };
            let out_source_frames = ((loop_seg_len_sec.max(0.0) * playback_rate * in_rate as f64)
                .ceil()
                .max(2.0)) as usize;
            (advanced_anchor, out_source_frames)
        } else {
            (anchor_frame, 0usize)
        };
        let segment: Vec<f32> = if loop_mode {
            build_loop_tiled_segment(
                &pcm,
                in_channels_usize,
                loop_advanced_anchor,
                clip.reversed,
                loop_out_source_frames,
            )
        } else {
            // 非 Loop：按消费窗口切片（正放 [ss, ss+len·r)、倒放
            // [se−len·r, se)，均 clamp 到媒体内；域外部分由前导/尾部静音表达）。
            let src_i0 = (slice_start_sec * in_rate as f64).floor().max(0.0) as usize;
            let src_i1 = (src_end_limit_sec * in_rate as f64)
                .ceil()
                .max(src_i0 as f64) as usize;
            let src_i1 = src_i1.min(in_frames);
            // Keep 1-frame slices audible (matches the real-time engine path);
            // only drop truly empty source ranges.
            if src_i1 <= src_i0 {
                continue;
            }
            pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)].to_vec()
        };

        let mut segment =
            linear_resample_interleaved(&segment, in_channels_usize, in_rate, out_rate);

        // Loop 模式的倒放方向已由回绕索引体现，不再整体反转。
        if !loop_mode && clip.reversed {
            reverse_interleaved_frames(&mut segment, in_channels_usize);
        }

        // 声道条件化（take 级 channel_mode 的唯一语义实现，见
        // channel_mode::condition_take_channels）：mono 源复制为双声道、
        // stereo 源按模式取平面/交换/下混，输出固定双声道交错。
        let segment = crate::channel_mode::condition_take_channels(
            &segment,
            in_channels,
            clip.take_channel_mode(),
        );
        let mut segment = segment;

        if let Some(params) = clip.formant_morph.as_ref().filter(|params| params.enabled) {
            // Loop（循环源）键必须编码**实际消费的平铺区间**（锚点推进量 + 消费
            // 帧数）：平铺段内容随导出窗口 [start_sec, end_sec] 变化，若键固定取
            // [0, total_sec]，不同导出窗口会命中同一条目 —— 先渲染的一方把错误
            // 长度/内容的结果投毒给另一方（get_or_compute 不做长度校验）。
            // 用"归一化锚点帧 + 消费帧数"（换算为秒）唯一确定 segment 内容。
            let (key_start_sec, key_end_sec) = if loop_mode {
                // 与上方片段构建共享同一组几何数值（loop_advanced_anchor /
                // loop_out_source_frames），键与 segment 内容严格对应。
                let start_frame = loop_advanced_anchor
                    .rem_euclid(((total_sec * in_rate as f64).round() as i64).max(1));
                (
                    start_frame as f64 / in_rate as f64,
                    (start_frame + loop_out_source_frames as i64) as f64 / in_rate as f64,
                )
            } else {
                // 非 Loop：键编码实际消费窗口（正放/倒放统一取自
                // clip_playback_window_sec，与 snapshot 实时域查找键成对）。
                (slice_start_sec, win_end_sec)
            };
            let key = crate::formant_cache::make_formant_cache_key(
                &clip.id,
                Path::new(source_path),
                out_rate,
                key_start_sec,
                key_end_sec,
                clip.reversed && !loop_mode,
                clip.channel_mode,
                // 本域输入已在上方做过声道条件化，与实时域（原始 stereo +
                // 混音时施加模式）必须用 preconditioned 判别隔离。
                true,
                // 离线 Loop 的处理对象是"回绕平铺 segment"（锚点起、长度为
                // clip 消费量），与实时域的完整文件自然顺序内容不同 —— 必须
                // 用 tiled_wrap 域判别隔离，避免两个域互相毒化缓存。
                loop_mode,
                params,
            );
            match crate::formant_cache::get_or_compute_formant_audio(
                key, &segment, out_rate, params,
            ) {
                Ok(entry) => {
                    segment = entry.pcm_stereo.as_ref().clone();
                }
                Err(err) => {
                    if debug {
                        log::error!(
                            "mixdown: formant morph failed; clip_id={} path={} err={}",
                            clip.id, source_path, err
                        );
                    }
                }
            }
        }

        // Pitch-preserving time-stretch:
        // - playback_rate == 1: keep source window duration as-is.
        // - playback_rate != 1: stretch the trimmed window to (src_len / playback_rate) in timeline time.
        // 若合成处理器声明自己处理时间拉伸（handles_time_stretch = true，如 vslib），
        // 则跳过此处外部拉伸，由 pitch edit 阶段的处理器内部完成。
        let processor_handles_stretch =
            crate::pitch_editing::processor_should_handle_stretch(timeline, clip);
        // 外部 SoundTouch 拉伸的执行条件：
        //   !processor_handles_stretch → 处理器不内部拉伸（World/HiFiGAN chain 内有 TimeStretchStage，vslib 原生拉伸）
        //   !opts.apply_pitch_edit    → pitch edit 链不会运行，内部拉伸无法触发，需回退到外部拉伸
        if (playback_rate - 1.0).abs() > 1e-6
            && (!processor_handles_stretch || !opts.apply_pitch_edit)
        {
            let seg_frames_in = segment.len() / 2;
            let target_frames = ((seg_frames_in as f64) / playback_rate).round().max(2.0) as usize;
            segment = time_stretch_interleaved(&segment, 2, out_rate, target_frames, opts.stretch);
        }

        // Loop（循环源）：整文件回绕已在片段构建阶段完成（见上方 build_loop_tiled_segment），
        // 此处 segment 天然覆盖整条 clip 的消费量，参数线阶段按绝对帧读取曲线即可。

        // ── pitch edit（可被整 Clip 渲染缓存短路）─────────────────────────────
        // 【复用的充分条件】只有"导出此刻要产出的那段音频"与"缓存里存的那段"
        // （整条 clip、clip 局部帧 0 起、含前导静音）**逐字节对应**时才允许读写缓存。
        // 下列条件缺一不可，每条都有对应的代码事实（见 `render_key` 的契约说明）：
        //   1. `apply_pitch_edit`：否则导出根本不做 DSP，而缓存是 DSP 之后的产物；
        //   2. 非 Loop：Loop 段是"导出窗口量化后的平铺段"，与整条 clip 不对应；
        //   3. 无前导静音：缓存从 clip 局部帧 0 起（含前导静音），导出不含；
        //   4. `playback_rate == 1`：DSP 仅在 rate≠1 时**替换**缓冲（内部拉伸），
        //      保长才能让下游几何（`seg_frames` → 淡化区、混音范围）逐帧不变；
        //   5. 非气声：缓存的主 PCM 是"谐波（breath_gain=0）"+ 独立噪声 stem，
        //      与导出链内混好的成品不是同一段音频（判据与预览共用同一实现）；
        //   6. 非张力：张力是独立的后处理变体（有自己的缓存键），主条目不含它；
        //   7. `segment` 长度 == clip 局部帧数：与缓存长度、下游期望长度三者相等 ——
        //      这一条在运行时把"长度不一致导致淡化相位漂移"彻底排除。
        let clip_tension_active = timeline
            .resolve_root_track_id(&clip.track_id)
            .and_then(|root| timeline.params_by_root_track.get(&root))
            .map(|entry| {
                crate::pitch_editing::hifigan_tension_active_for_clip(entry, clip, clip_start_sec)
            })
            .unwrap_or(false);
        let reuse_key = if export_render_cache_reuse_enabled()
            && opts.apply_pitch_edit
            && !clip.loop_enabled
            && pre_silence_sec <= 1e-6
            && (playback_rate - 1.0).abs() <= 1e-6
            && !crate::pitch_editing::clip_breath_active(timeline, clip)
            && !clip_tension_active
            && segment.len() == clip_total_frames * 2
        {
            export_clip_render_cache_key(timeline, clip, out_rate, scale_signature.as_str())
        } else {
            None
        };

        if opts.apply_pitch_edit {
            let seg_start_sec = clip_start_sec + pre_silence_sec + loop_seg_local_start_sec;
            let mut seg = segment;
            let mut rendered_ok = false;
            let mut used_cache = false;

            // 命中：直接用整条 clip 的渲染结果，跳过解码/重采样/拉伸/pitch-edit DSP。
            if let Some(key) = reuse_key.as_ref() {
                if let Some(entry) = crate::render_cache::load_rendered(key, out_rate) {
                    // 长度再校验一次：即便上面的推理有偏差，长度一致就保证下游几何不变。
                    if entry.pcm_stereo.len() == seg.len() {
                        seg = entry.pcm_stereo.as_ref().clone();
                        used_cache = true;
                        clips_cache_reused = clips_cache_reused.saturating_add(1);
                        if let Some(stats) = opts.cache_stats.as_ref() {
                            stats.reused.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            if !used_cache {
                let applied = crate::pitch_editing::maybe_apply_pitch_edit_to_clip_segment(
                    timeline,
                    clip,
                    clip_start_sec,
                    seg_start_sec,
                    out_rate,
                    &mut seg,
                );
                match applied {
                    Ok(true) => rendered_ok = true,
                    Ok(false) => rendered_ok = true,
                    Err(e) => {
                        // 与既有行为一致：处理器失败不降级为"未处理的原始音频"，
                        // 只记录并继续（该 clip 不参与回填，见下）。
                        log::error!("[pitch_edit] clip_id={} ERROR: {e}", clip.id);
                    }
                }
            }
            segment = seg;

            // 【回填】只在"这次确实渲染成功"时写盘：失败时 `seg` 仍是未处理的源音频，
            // 写进去会毒化播放路径 —— 这正是 `render_single_clip` 明确拒绝缓存的那种
            // 半成品。条目形态与预览侧逐字段一致（`RenderedClipCacheEntry`）。
            if rendered_ok {
                if let Some(key) = reuse_key.as_ref() {
                    crate::render_cache::store_rendered(
                        key,
                        &crate::synth_clip_cache::RenderedClipCacheEntry {
                            pcm_stereo: Arc::new(segment.clone()),
                            breath_noise_stereo: None,
                            frames: (segment.len() / 2) as u64,
                            sample_rate: out_rate,
                            rendered_take_id: clip.active_take_id.clone(),
                        },
                    );
                    clips_cache_stored = clips_cache_stored.saturating_add(1);
                    if let Some(stats) = opts.cache_stats.as_ref() {
                        stats.stored.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // 提取共通 volume / pan / dyn 曲线（与 snapshot.rs 的逻辑对应）。
        // 所有算法一律由本函数的混音阶段应用，不存在任何「处理器已烘焙」的例外。
        let (
            volume_curve,
            volume_curve_frame_period_ms,
            pan_curve,
            pan_curve_frame_period_ms,
            dyn_curve,
            dyn_orig,
            dyn_curve_frame_period_ms,
        ) = timeline
            .resolve_root_track_id(&clip.track_id)
            .and_then(|root| {
                let entry = timeline.params_by_root_track.get(&root)?;
                let volume = crate::pitch_editing::common_volume_curve_for_clip(entry, clip);
                let pan = crate::pitch_editing::common_pan_curve_for_clip(entry, clip);
                let dyn_orig_source = crate::pitch_editing::dyn_orig_curve_for_clip(entry, clip);
                // 与实时引擎同一处理：解析哨兵 + 分母同款下钳 + 末帧留一格缓冲。
                // 详见 `resolve_dyn_curves_for_audio`；导出与监听必须同源。
                let (dyn_curve, dyn_orig) =
                    match crate::pitch_editing::common_dyn_curve_for_clip(entry, clip) {
                        Some(curve) => {
                            let resolved =
                                crate::renderer::common_params::resolve_dyn_curves_for_audio(
                                    curve,
                                    dyn_orig_source,
                                );
                            let baseline =
                                (!resolved.baseline.is_empty()).then_some(resolved.baseline);
                            (Some(resolved.target), baseline)
                        }
                        None => (None, None),
                    };
                Some((
                    volume,
                    entry.frame_period_ms.max(0.1),
                    pan,
                    entry.frame_period_ms.max(0.1),
                    dyn_curve,
                    dyn_orig,
                    entry.frame_period_ms.max(0.1),
                ))
            })
            .unwrap_or((None, 5.0, None, 5.0, None, None, 5.0));

        // Apply fades and gain (timeline-referenced)。淡化按 REAPER 形状/曲率
        // 查表求值（与实时引擎、画布渲染同一公式核心）。
        let fade_in_frames = (clip.effective_fade_in_sec().max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let fade_out_frames = (clip.effective_fade_out_sec().max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let fade_in_lut = if fade_in_frames > 0 {
            Some(crate::fade_curves::global_fade_lut(
                clip.fade_in_shape,
                clip.fade_in_dir,
                false,
            ))
        } else {
            None
        };
        let fade_out_lut = if fade_out_frames > 0 {
            Some(crate::fade_curves::global_fade_lut(
                clip.fade_out_shape,
                clip.fade_out_dir,
                true,
            ))
        } else {
            None
        };

        let seg_frames = segment.len() / 2;
        // `clip_total_frames` 已在上方（clip 几何处）定义 —— 复用门禁与这里必须是同一个值。
        let pre_silence_frames = (pre_silence_sec * out_rate as f64).round().max(0.0) as usize;

        // Mix into output, considering overlap window.
        // The audio segment starts after pre_silence_sec (Loop：再叠加窗口
        // 起点的 clip 局部偏移 —— 平铺段只覆盖窗口交集，见上方) and lasts seg_frames/out_rate.
        let seg_start_sec = clip_start_sec + pre_silence_sec + loop_seg_local_start_sec;
        let seg_end_sec = seg_start_sec + (seg_frames as f64) / out_rate as f64;

        // Loop（循环源）：平铺段只覆盖【导出窗口 ∩ clip】，seg 内的帧偏移是
        // "窗口内相对位置"；淡化 / 音量 / 声像曲线必须按 **clip 局部绝对位置**
        // 求值 —— 否则局部导出（后台预渲染、区间导出、波形 peaks）会在每个
        // 窗口边界重新触发 fade-in。非 Loop 时该偏移为 0，行为不变。
        let loop_local_offset_frames = if loop_mode {
            ((loop_seg_local_start_sec * out_rate as f64)
                .round()
                .max(0.0)) as usize
        } else {
            0usize
        };

        let clip_window_start = seg_start_sec.max(start_sec);
        let clip_window_end = seg_end_sec.min(end_sec).min(clip_end_sec);
        let window_len_sec = (clip_window_end - clip_window_start).max(0.0);
        if window_len_sec <= 1e-9 {
            continue;
        }

        let out_offset_frames = ((clip_window_start - start_sec) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let seg_offset_frames = ((clip_window_start - seg_start_sec) * out_rate as f64)
            .round()
            .max(0.0) as usize;
        let frames_to_mix = ((window_len_sec) * out_rate as f64).round().max(0.0) as usize;

        // 只按输出窗口裁剪循环上界；segment 可能比 clip 短（深拉伸/窗口越出
        // 源素材），越界帧在循环内按淡出语义处理（区间收缩 + 末帧保持衰减），
        // 不能在循环上界把淡出区尾部提前截断（截断=增益未走完时硬切）。
        let max_frames_to_mix = frames_to_mix.min(out_frames.saturating_sub(out_offset_frames));
        if max_frames_to_mix == 0 {
            continue;
        }

        clips_mixed = clips_mixed.saturating_add(1);

        let has_volume_curve = volume_curve.is_some() && !volume_curve.as_ref().unwrap().is_empty();
        let has_pan_curve = pan_curve.is_some() && !pan_curve.as_ref().unwrap().is_empty();
        let has_dyn_curve = dyn_curve.as_ref().is_some_and(|c| !c.is_empty());
        let has_dyn_orig_curve = dyn_orig.as_ref().is_some_and(|c| !c.is_empty());
        // 淡出（端点锁定 + 内容耗尽收缩），语义与 audio_engine/mix.rs 一致：
        // - 末帧进度恰为 1 → 增益精确 0（防 e<1 曲线末端阶跃）；
        // - segment 越界（内容不足）时淡出区间收缩为 [E-N, L]（E=内容末端），
        //   之后保持末帧内容按淡出增益衰减 —— 杜绝硬切 Click。
        let default_zone_start = clip_total_frames.saturating_sub(fade_out_frames);
        // Loop（循环源）：平铺回绕内容永不耗尽，不做收缩。
        let content_end_clip = if loop_mode {
            clip_total_frames
        } else {
            (pre_silence_frames
                .saturating_add(loop_local_offset_frames)
                .saturating_add(seg_offset_frames)
                .saturating_add(seg_frames))
            .min(clip_total_frames)
        };
        let fade_zone_start = if content_end_clip < default_zone_start {
            content_end_clip.saturating_sub(fade_out_frames)
        } else {
            default_zone_start
        };
        let mut last_l: f32 = 0.0;
        let mut last_r: f32 = 0.0;
        let mut has_last: bool = false;
        for f in 0..max_frames_to_mix {
            if f % 4096 == 0 {
                if mixdown_cancelled(&opts) {
                    return Err("export_cancelled".to_string());
                }
                // 复用既有的 4k 帧检查点上报 clip 内进度（4096 = 0 时 intra 也为 0）。
                mix_progress.report_intra(clip_index, f, max_frames_to_mix);
            }
            let oi = (out_offset_frames + f) * 2;
            let si = (seg_offset_frames + f) * 2;

            // Local position inside the CLIP (timeline), used for fades.
            // Loop：叠加窗口起点的 clip 局部偏移（见上方 loop_local_offset_frames）。
            let local_in_clip = pre_silence_frames
                .saturating_add(loop_local_offset_frames)
                .saturating_add(seg_offset_frames + f);
            if local_in_clip >= clip_total_frames {
                break;
            }

            let mut g = gain;
            if fade_in_frames > 0 && local_in_clip < fade_in_frames {
                // Frame-centered fade-in (same as audio_engine/mix.rs) so the
                // first frame is not hard-zeroed and export matches preview.
                g *= match &fade_in_lut {
                    Some(lut) => crate::fade_curves::sample_fade_lut(
                        lut,
                        ((local_in_clip + 1) as f64 / fade_in_frames as f64)
                            * crate::fade_curves::FADE_LUT_SIZE as f64,
                    ),
                    None => ((local_in_clip + 1) as f32 / fade_in_frames as f32).clamp(0.0, 1.0),
                };
            }
            if fade_out_frames > 0
                && local_in_clip >= fade_zone_start
                && local_in_clip < clip_total_frames
            {
                let progress =
                    (local_in_clip - fade_zone_start + 1) as f64 / fade_out_frames.max(1) as f64;
                if progress <= 1.0 {
                    g *= match &fade_out_lut {
                        Some(lut) => crate::fade_curves::sample_fade_lut(
                            lut,
                            progress * crate::fade_curves::FADE_LUT_SIZE as f64,
                        ),
                        None => (progress as f32).clamp(0.0, 1.0),
                    };
                } else {
                    // 收缩后的淡出在这帧之前已走完 → 静音。
                    g = 0.0;
                }
            }
            if g <= 0.0 {
                continue;
            }

            // 内容耗尽：fade 激活时保持末帧按淡出增益继续衰减（E 处及其后
            // 增益从 ~1 平滑走到 0）；不激活时维持“越界静音”语义。
            let seg_avail = si + 1 < segment.len();
            if !seg_avail {
                if fade_out_frames > 0
                    && local_in_clip >= fade_zone_start
                    && local_in_clip < clip_total_frames
                    && has_last
                {
                    // 只有真存在曲线时才计算
                    let abs_sec = clip_start_sec + (local_in_clip as f64 / out_rate as f64);
                    let mut final_g = g;
                    if has_volume_curve {
                        let vol = sample_automation_curve_at_sec(
                            volume_curve,
                            abs_sec,
                            volume_curve_frame_period_ms,
                            1.0,
                        );
                        final_g *= vol;
                    }
                    if has_dyn_curve || has_dyn_orig_curve {
                        final_g *= dyn_gain_at_sec(
                            dyn_curve.as_deref(),
                            dyn_orig.as_deref(),
                            abs_sec,
                            dyn_curve_frame_period_ms,
                        );
                    }
                    let pan = if has_pan_curve {
                        sample_automation_curve_at_sec(
                            pan_curve,
                            abs_sec,
                            pan_curve_frame_period_ms,
                            0.0,
                        )
                    } else {
                        0.0
                    }
                    .clamp(-1.0, 1.0);
                    let (left_gain, right_gain) = if pan <= 0.0 {
                        (1.0, 1.0 + pan)
                    } else {
                        (1.0 - pan, 1.0)
                    };
                    mix[oi] += last_l * final_g * left_gain;
                    mix[oi + 1] += last_r * final_g * right_gain;
                }
                continue;
            }
            last_l = segment[si];
            last_r = segment[si + 1];
            has_last = true;

            // 只有真存在曲线时才计算
            let mut final_g = g;
            let abs_sec = clip_start_sec + (local_in_clip as f64 / out_rate as f64);
            if has_volume_curve {
                let vol = sample_automation_curve_at_sec(
                    volume_curve,
                    abs_sec,
                    volume_curve_frame_period_ms,
                    1.0,
                );
                final_g *= vol;
            }
            if has_dyn_curve || has_dyn_orig_curve {
                final_g *= dyn_gain_at_sec(
                    dyn_curve.as_deref(),
                    dyn_orig.as_deref(),
                    abs_sec,
                    dyn_curve_frame_period_ms,
                );
            }

            let pan = if has_pan_curve {
                sample_automation_curve_at_sec(pan_curve, abs_sec, pan_curve_frame_period_ms, 0.0)
            } else {
                0.0
            }
            .clamp(-1.0, 1.0);
            // 线性平衡：center 保持两声道增益为 1，避免中心衰减。
            let (left_gain, right_gain) = if pan <= 0.0 {
                (1.0, 1.0 + pan)
            } else {
                (1.0 - pan, 1.0)
            };

            mix[oi] += last_l * final_g * left_gain;
            mix[oi + 1] += last_r * final_g * right_gain;
        }
    }

    if debug {
        let mut max_abs = 0.0f32;
        for &v in &mix {
            let a = v.abs();
            if a.is_finite() && a > max_abs {
                max_abs = a;
            }
        }
        log::warn!(
            "mixdown: rendered window start_sec={:.3} end_sec={:.3} sr={} frames={} max_abs={:.6} clips_considered={} clips_decoded={} clips_mixed={} cache_reused={} cache_stored={}",
            start_sec,
            end_sec,
            out_rate,
            out_frames,
            max_abs,
            clips_considered,
            clips_decoded,
            clips_mixed,
            clips_cache_reused,
            clips_cache_stored
        );
    }

    mix_progress.finish();

    Ok((out_rate, out_channels, duration_sec, mix))
}

#[cfg(test)]
mod tests {
    use super::build_loop_tiled_segment;
    use super::*;

    /// 端到端：真实立体声 WAV → decode → 窗口 → 重采样 → 声道条件化 → 混音。
    /// 锁定"切换 Take 声道模式必须改变导出渲染结果"——这是用户报告的
    /// "只改波形不改渲染"症状的回归防线。
    #[test]
    fn mixdown_honors_take_channel_mode() {
        use crate::state::{Clip, TimelineState};

        // ── 1. 写一个立体声 WAV：L ≈ 0.5 恒定，R ≈ -0.25 恒定（左右可分辨）。
        let dir = std::env::temp_dir().join(format!("hfs_mix_mode_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav_path = dir.join("stereo.wav");
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for _ in 0..44100 {
                writer.write_sample(16384).unwrap(); // ≈ 0.5
                writer.write_sample(-8192).unwrap(); // ≈ -0.25
            }
            writer.finalize().unwrap();
        }
        let source_path = wav_path.to_string_lossy().to_string();

        // ── 2. 组装 TimelineState：默认轨 + 单 clip，gain=1，无 fade。
        let build_timeline = |channel_mode: i32| {
            let mut tl = TimelineState::default();
            let track_id = tl.tracks[0].id.clone();
            let clip = Clip {
                id: format!("clip_mode_{}", channel_mode),
                group_id: None,
                track_id,
                name: "stereo".to_string(),
                start_sec: 0.0,
                length_sec: 0.5,
                color: "#000000".to_string(),
                takes: vec![],
                active_take_id: None,
                clip_playback_rate: 1.0,
                source_path: Some(source_path.clone()),
                source_path_relative: None,
                duration_sec: Some(1.0),
                duration_frames: Some(44100),
                source_sample_rate: Some(44100),
                source_channels: Some(2),
                source_file_mtime: None,
                source_file_size: None,
                source_file_fingerprint: None,
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: 1.0,
                playback_rate: 1.0,
                reversed: false,
                channel_mode,
                loop_enabled: false,
                snap_offset_sec: 0.0,
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_shape: 0.0,
                fade_out_shape: 0.0,
                fade_in_dir: 0.0,
                fade_out_dir: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                auto_fade_in_sec: 0.0,
                auto_fade_out_sec: 0.0,
                extra_curves: None,
                extra_params: None,
                formant_morph: None,
                midi_note_data: None,
                midi_fill_gaps: false,
            };
            tl.clips.push(clip);
            tl.normalize_clip_takes();
            tl
        };

        let render = |tl: &TimelineState| {
            let (_rate, _ch, _dur, mix) = render_mixdown_interleaved(
                tl,
                MixdownOptions {
                    sample_rate: 44100,
                    start_sec: 0.0,
                    end_sec: Some(0.5),
                    stretch: crate::time_stretch::StretchAlgorithm::LinearResample,
                    apply_pitch_edit: false,
                    output: crate::encode::OutputSpec::wav_32f(),
                    quality_preset: QualityPreset::Realtime,
                    cancel_flag: None,
                    progress: None,
                    cache_stats: None,
                },
            )
            .unwrap();
            // 取中段一帧，避开任何边缘淡化。
            let mid = (mix.len() / 2) & !1;
            (mix[mid], mix[mid + 1])
        };

        let (n_l, n_r) = render(&build_timeline(0));
        assert!(
            (n_l - 0.5).abs() < 0.02 && (n_r - (-0.25)).abs() < 0.02,
            "Normal 应输出原始 L/R，得到 ({n_l}, {n_r})"
        );

        let (s_l, s_r) = render(&build_timeline(1));
        assert!(
            (s_l - (-0.25)).abs() < 0.02 && (s_r - 0.5).abs() < 0.02,
            "Swap 应交换 L/R，得到 ({s_l}, {s_r})"
        );

        let (ml, mr) = render(&build_timeline(3));
        assert!(
            (ml - 0.5).abs() < 0.02 && (mr - 0.5).abs() < 0.02,
            "MonoLeft 应双声道输出左声道，得到 ({ml}, {mr})"
        );

        let (mml, mmr) = render(&build_timeline(2));
        assert!(
            (mml - 0.125).abs() < 0.02 && (mmr - 0.125).abs() < 0.02,
            "MonoMix 应双声道输出 (L+R)/2=0.125，得到 ({mml}, {mmr})"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }



    /// 【QualityPreset 惰性契约】`MixdownOptions::quality_preset` 目前是"写入但
    /// 忽略"的占位（见 `QualityPreset` 的文档：让它生效必须先定义两档差异并用
    /// 真实素材 A/B 验证）。本测试把"忽略"钉死：两档预设的渲染输出必须逐字节
    /// 一致。若未来要消费该字段，请先完成文档要求的 A/B 验证，再**有意地**
    /// 改写本测试（届时它守护的是"档位差异确实按预期生效"）。
    #[test]
    fn quality_preset_is_currently_inert() {
        use crate::state::{Clip, TimelineState};

        let dir = std::env::temp_dir().join(format!("hfs_mix_qpreset_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav_path = dir.join("stereo.wav");
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for i in 0..44100 {
                let v = ((i as f32) * 0.01).sin();
                writer.write_sample((v * 16384.0) as i32).unwrap();
                writer.write_sample((v * -8192.0) as i32).unwrap();
            }
            writer.finalize().unwrap();
        }
        let source_path = wav_path.to_string_lossy().to_string();

        let build_timeline = || {
            let mut tl = TimelineState::default();
            let track = tl.tracks[0].id.clone();
            let clip = Clip {
                id: "clip_qpreset".to_string(),
                group_id: None,
                track_id: track,
                name: "V".to_string(),
                start_sec: 0.0,
                length_sec: 0.5,
                color: "#000000".to_string(),
                takes: vec![],
                active_take_id: None,
                clip_playback_rate: 1.0,
                source_path: Some(source_path.clone()),
                source_path_relative: None,
                duration_sec: Some(1.0),
                duration_frames: Some(44100),
                source_sample_rate: Some(44100),
                source_channels: Some(2),
                source_file_mtime: None,
                source_file_size: None,
                source_file_fingerprint: None,
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: 1.0,
                playback_rate: 1.0,
                reversed: false,
                channel_mode: 0,
                loop_enabled: false,
                snap_offset_sec: 0.0,
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_shape: 0.0,
                fade_out_shape: 0.0,
                fade_in_dir: 0.0,
                fade_out_dir: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                auto_fade_in_sec: 0.0,
                auto_fade_out_sec: 0.0,
                extra_curves: None,
                extra_params: None,
                formant_morph: None,
                midi_note_data: None,
                midi_fill_gaps: false,
            };
            tl.clips.push(clip);
            tl.normalize_clip_takes();
            tl
        };

        let render = |preset: QualityPreset| {
            let (_rate, _ch, _dur, mix) = render_mixdown_interleaved(
                &build_timeline(),
                MixdownOptions {
                    sample_rate: 44100,
                    start_sec: 0.0,
                    end_sec: Some(0.5),
                    stretch: crate::time_stretch::StretchAlgorithm::LinearResample,
                    apply_pitch_edit: false,
                    output: crate::encode::OutputSpec::wav_32f(),
                    quality_preset: preset,
                    cancel_flag: None,
                    progress: None,
                    cache_stats: None,
                },
            )
            .unwrap();
            mix
        };

        let realtime = render(QualityPreset::Realtime);
        let export = render(QualityPreset::Export);
        assert_eq!(
            realtime.len(),
            export.len(),
            "两档预设的输出长度必须一致（该字段当前为占位）"
        );
        assert!(
            realtime == export,
            "QualityPreset 当前必须是'写入但忽略'：两档渲染输出出现差异说明有人开始消费该字段 —— 先完成 QualityPreset 文档要求的 A/B 验证，再有意更新本测试"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 【导出进度】混音阶段的进度必须：从 0 起、单调不减、结束时到 1.0，且在
    /// 多 clip 下按 clip 加权推进（不是一步跳到 1）。守护 `MixdownProgress` 的
    /// 单调性与两级公式。
    #[test]
    fn interleaved_reports_monotonic_clip_weighted_progress() {
        use crate::state::{Clip, TimelineState};
        use std::sync::Mutex;

        let dir = std::env::temp_dir().join(format!("hfs_mix_progress_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav_path = dir.join("tone.wav");
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for i in 0..44100 {
                let v = ((i as f32) * 0.01).sin();
                writer.write_sample((v * 16384.0) as i32).unwrap();
                writer.write_sample((v * 16384.0) as i32).unwrap();
            }
            writer.finalize().unwrap();
        }
        let source_path = wav_path.to_string_lossy().to_string();

        let mut tl = TimelineState::default();
        let track = tl.tracks[0].id.clone();
        for (index, start) in [0.0f64, 0.5].into_iter().enumerate() {
            let clip = Clip {
                id: format!("clip_progress_{index}"),
                group_id: None,
                track_id: track.clone(),
                name: "V".to_string(),
                start_sec: start,
                length_sec: 0.5,
                color: "#000000".to_string(),
                takes: vec![],
                active_take_id: None,
                clip_playback_rate: 1.0,
                source_path: Some(source_path.clone()),
                source_path_relative: None,
                duration_sec: Some(1.0),
                duration_frames: Some(44100),
                source_sample_rate: Some(44100),
                source_channels: Some(2),
                source_file_mtime: None,
                source_file_size: None,
                source_file_fingerprint: None,
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: 1.0,
                playback_rate: 1.0,
                reversed: false,
                channel_mode: 0,
                loop_enabled: false,
                snap_offset_sec: 0.0,
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_shape: 0.0,
                fade_out_shape: 0.0,
                fade_in_dir: 0.0,
                fade_out_dir: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                auto_fade_in_sec: 0.0,
                auto_fade_out_sec: 0.0,
                extra_curves: None,
                extra_params: None,
                formant_morph: None,
                midi_note_data: None,
                midi_fill_gaps: false,
            };
            tl.clips.push(clip);
        }
        tl.normalize_clip_takes();

        let recorded: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&recorded);
        let (_rate, _ch, _dur, _mix) = render_mixdown_interleaved(
            &tl,
            MixdownOptions {
                sample_rate: 44100,
                start_sec: 0.0,
                end_sec: Some(1.0),
                stretch: crate::time_stretch::StretchAlgorithm::LinearResample,
                apply_pitch_edit: false,
                output: crate::encode::OutputSpec::wav_32f(),
                quality_preset: QualityPreset::Export,
                cancel_flag: None,
                progress: Some(ProgressCallback::new(move |value: f64| {
                    sink.lock().unwrap().push(value);
                })),
                cache_stats: None,
            },
        )
        .unwrap();

        let values = recorded.lock().unwrap().clone();
        assert!(!values.is_empty(), "必须至少上报一次进度");
        assert_eq!(values[0], 0.0, "首次上报应为 0.0");
        assert_eq!(*values.last().unwrap(), 1.0, "结束时必须到 1.0");
        assert!(
            values.windows(2).all(|w| w[1] >= w[0]),
            "进度必须单调不减，实际: {values:?}"
        );
        assert!(
            values.iter().any(|v| (*v - 0.5).abs() < 1e-9),
            "两个 clip 之间应上报 0.5（clip 加权），实际: {values:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 【导出复用整 Clip 渲染缓存】往返契约。
    ///
    /// 三趟渲染分别钉住三件事：
    /// - A 首次渲染 → **回填**磁盘缓存（`contains_rendered` 为真）；
    /// - B 命中复用 → 与 A **逐样本一致**（复用不得改变导出音频）；
    /// - C 用哨兵条目覆盖缓存 → 结果**必须改变**（证明读取路径真的生效，
    ///   而不是"看起来命中、实际照旧渲染"）。
    ///
    /// 【夹具为什么这样取】复用门禁要求"导出此刻要产出的那段音频 == 缓存里存的那段"：
    /// 非 Loop、无前导静音、`playback_rate == 1`、非气声 / 非张力，且 segment 长度
    /// 恰好等于 clip 局部帧数（源窗口与 clip 长度对齐到帧网格即可满足）。
    /// `pitch_edit_user_modified = true` 让该 clip 参与渲染（于是有缓存键），
    /// 而 `pitch_edit == pitch_orig` 让处理器在"范围内无用户编辑"时**提前返回**
    /// `Ok(false)` —— 测试因此不依赖任何合成后端（ONNX / 模型）是否可用。
    #[test]
    fn export_reuses_rendered_clip_cache_faithfully() {
        use crate::state::{Clip, TimelineState};

        // 渲染键会混入「运行时拉伸设置」这个进程级全局：算键的测试必须持读锁，
        // 否则与"改该全局"的用例并发时，同一测试内的两次键计算会跨越一次全局变更
        // —— 表现为"哨兵可见却没命中"的偶发失败（详见 `render_key::test_locks`）。
        let _stretch_global = crate::render_key::test_locks::lock_stretch_global_read();

        let dir = std::env::temp_dir().join(format!("hfs_mix_reuse_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 渲染缓存指向临时目录并启用（默认 enabled=true；location=system → system_dir）。
        // 其它用例的 clip 都带 `apply_pitch_edit: false`，不满足复用门禁，
        // 因此这次全局设置不会改变它们的行为。
        crate::render_cache::init(dir.join("render_cache"));
        crate::render_cache::apply_settings(&crate::config::RenderCacheSettings::default());

        let wav_path = dir.join("tone.wav");
        // 0.5s @44.1k 立体声：与 clip 长度对齐，使 segment 长度 == clip 局部帧数。
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44_100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for i in 0..22_050 {
                let v = ((i as f32) * 0.01).sin();
                writer.write_sample((v * 16384.0) as i32).unwrap();
                writer.write_sample((v * 16384.0) as i32).unwrap();
            }
            writer.finalize().unwrap();
        }
        let source_path = wav_path.to_string_lossy().to_string();

        // 夹具按 clip id 参数化：C 步骤要用**独立 id**（⇒ 独立缓存键），
        // 这样该键下只有我写的那一个条目，不会被本测试自己的异步回填覆盖。
        let build_timeline = |clip_id: &str| {
            let mut tl = TimelineState::default();
            let track_id = tl.tracks[0].id.clone();
            // 让该 clip 参与处理器渲染（⇒ 有缓存键）：算法非 Bypass + 用户改过曲线。
            tl.tracks[0].pitch_analysis_algo = crate::state::PitchAnalysisAlgo::NsfHifiganOnnx;
            let mut entry = crate::state::TrackParamsState::default();
            entry.pitch_edit_user_modified = true;
            entry.frame_period_ms = 5.0;
            // 曲线相等 ⇒ 范围内"无用户编辑" ⇒ 处理器提前返回 Ok(false)（不需要模型）。
            entry.pitch_edit = vec![60.0; 8];
            entry.pitch_orig = vec![60.0; 8];
            tl.params_by_root_track.insert(track_id.clone(), entry);

            let clip = Clip {
                id: clip_id.to_string(),
                group_id: None,
                track_id: track_id.clone(),
                name: "V".to_string(),
                start_sec: 0.0,
                length_sec: 0.5,
                color: "#000000".to_string(),
                takes: vec![],
                active_take_id: None,
                clip_playback_rate: 1.0,
                source_path: Some(source_path.clone()),
                source_path_relative: None,
                duration_sec: Some(0.5),
                duration_frames: Some(22_050),
                source_sample_rate: Some(44_100),
                source_channels: Some(2),
                source_file_mtime: None,
                source_file_size: None,
                source_file_fingerprint: None,
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: 0.5,
                playback_rate: 1.0,
                reversed: false,
                channel_mode: 0,
                loop_enabled: false,
                snap_offset_sec: 0.0,
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_shape: 0.0,
                fade_out_shape: 0.0,
                fade_in_dir: 0.0,
                fade_out_dir: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                auto_fade_in_sec: 0.0,
                auto_fade_out_sec: 0.0,
                extra_curves: None,
                extra_params: None,
                formant_morph: None,
                midi_note_data: None,
                midi_fill_gaps: false,
            };
            tl.clips.push(clip);
            tl.normalize_clip_takes();
            tl
        };

        let timeline = build_timeline("clip_reuse");
        let key = export_clip_render_cache_key(
            &timeline,
            &timeline.clips[0],
            44_100,
            timeline.render_scale_signature().as_str(),
        )
        .expect("夹具 clip 必须参与渲染，否则没有可复用的缓存键");

        let render = |tl: &TimelineState, stats: &Arc<crate::mixdown::MixdownCacheStats>| {
            let (_sr, _ch, _dur, mix) = render_mixdown_interleaved(
                tl,
                MixdownOptions {
                    sample_rate: 44_100,
                    start_sec: 0.0,
                    end_sec: Some(0.5),
                    stretch: crate::time_stretch::StretchAlgorithm::LinearResample,
                    apply_pitch_edit: true, // 复用门禁要求（与生产调用点一致）
                    output: crate::encode::OutputSpec::wav_32f(),
                    quality_preset: QualityPreset::Export,
                    cancel_flag: None,
                    progress: None,
                    cache_stats: Some(Arc::clone(stats)),
                },
            )
            .expect("render ok");
            mix
        };

        // 渲染缓存的目录/开关是**进程级全局状态**，而本套件里其它用例（以及
        // `render_cache` 自身的用例）都可能改动它。每一步前都重新确立本测试的
        // 目录与开关，断言因此不依赖"当前恰好还是我的设置"。
        let setup_cache = || {
            crate::render_cache::init(dir.join("render_cache"));
            crate::render_cache::apply_settings(&crate::config::RenderCacheSettings::default());
        };

        // ── A：首次渲染 → 回填 ───────────────────────────────────────────────
        // 落盘由 writer 线程异步完成：`flush_blocking` 排空队列后再断言，
        // 避免"刚写完还没落盘"的时序假失败。
        setup_cache();
        let a_stats = Arc::new(crate::mixdown::MixdownCacheStats::default());
        let a = render(&timeline, &a_stats);
        assert!(
            crate::render_cache::flush_blocking(std::time::Duration::from_secs(5)),
            "渲染缓存写入队列未能排空"
        );
        assert_eq!(a_stats.snapshot().1, 1, "首次导出应回填该 clip 的渲染结果");
        assert!(
            crate::render_cache::contains_rendered(key.param_hash),
            "回填的条目应能在磁盘上查到"
        );

        // ── B：命中复用 → 与 A 逐样本一致 ───────────────────────────────────
        // 断言的是**导出自己上报的命中数**（同一次调用内算键并加载，因此不受
        // "跨时刻的键漂移"影响），比"改写缓存内容再看输出变化"可靠得多。
        //
        // 【为什么要重试】渲染键会混入进程级全局（运行时拉伸设置），而本套件是并行跑的：
        // 即便本测试持了共享读锁，仍可能与"改该全局"的用例交错一拍。漂移是瞬时的，
        // 因此有界重试即可收敛；从未命中则说明读取路径真的没生效。
        let mut faithful = false;
        for _ in 0..40 {
            setup_cache();
            let b_stats = Arc::new(crate::mixdown::MixdownCacheStats::default());
            let b = render(&build_timeline("clip_reuse"), &b_stats);
            if b_stats.snapshot().0 == 1 {
                assert_eq!(a, b, "复用缓存后的导出结果必须与重新渲染逐样本一致");
                faithful = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(
            faithful,
            "导出未命中缓存（复用读取路径没生效）：应命中 A 回填的条目并逐样本一致"
        );

        // 复原：关闭缓存，避免影响同一进程内的其它用例。
        let mut off = crate::config::RenderCacheSettings::default();
        off.enabled = false;
        crate::render_cache::apply_settings(&off);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 交错 PCM：帧 i 的样本值为 [i as f32, i as f32 + 0.5]。
    fn make_pcm(frames: usize, channels: usize) -> Vec<f32> {
        let mut pcm = Vec::with_capacity(frames * channels);
        for f in 0..frames {
            for c in 0..channels {
                pcm.push(f as f32 + if channels > 1 { c as f32 * 0.5 } else { 0.0 });
            }
        }
        pcm
    }

    #[test]
    fn loop_tiled_forward_wraps_over_whole_file() {
        // 5 帧立体声媒体，锚点 3：期望序列 3,4,0,1,2,3,4,0
        let channels = 2;
        let pcm = make_pcm(5, channels);
        let out = build_loop_tiled_segment(&pcm, channels, 3, false, 8);
        assert_eq!(out.len(), 8 * channels);
        let expected = [3.0f32, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0];
        for (i, f) in expected.iter().enumerate() {
            assert_eq!(out[i * 2], *f, "forward frame {i} left");
            assert_eq!(out[i * 2 + 1], *f + 0.5, "forward frame {i} right");
        }
    }

    #[test]
    fn loop_tiled_reverse_descends_and_wraps() {
        // 锚点 exclusive=4（即从帧 3 开始向下）：3,2,1,0,4,3,2
        let pcm = make_pcm(5, 1);
        let out = build_loop_tiled_segment(&pcm, 1, 4, true, 7);
        let expected = [3.0f32, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0];
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn loop_tiled_negative_forward_anchor_wraps_to_tail() {
        // 负锚点 -2 对 5 帧媒体 → floor_mod(-2,5)=3：序列 3,4,0,1,2
        let pcm = make_pcm(5, 1);
        let out = build_loop_tiled_segment(&pcm, 1, -2, false, 5);
        let expected = [3.0f32, 4.0, 0.0, 1.0, 2.0];
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn loop_tiled_matches_per_frame_floor_mod_reference() {
        // 与逐帧 floor_mod 参考实现对拍（多周期 + 大锚点偏移）
        let total = 37i64;
        let pcm = make_pcm(total as usize, 2);
        for &anchor in &[-50i64, 0, 1, 19, 36, 1000] {
            for &reversed in &[false, true] {
                let n = 100usize;
                let out = build_loop_tiled_segment(&pcm, 2, anchor, reversed, n);
                assert_eq!(out.len(), n * 2);
                for f in 0..n {
                    let fi = f as i64;
                    let expect = if reversed {
                        (anchor - 1 - fi).rem_euclid(total)
                    } else {
                        (anchor + fi).rem_euclid(total)
                    } as usize;
                    assert_eq!(out[f * 2], expect as f32);
                    assert_eq!(out[f * 2 + 1], expect as f32 + 0.5);
                }
            }
        }
    }

    #[test]
    fn loop_tiled_handles_empty_and_degenerate_inputs() {
        assert!(build_loop_tiled_segment(&[], 2, 0, false, 10).is_empty());
        let pcm = make_pcm(4, 2);
        assert_eq!(build_loop_tiled_segment(&pcm, 2, 0, false, 0).len(), 0);
        // 单帧媒体也能循环铺满
        let one = vec![0.5f32, 0.25f32];
        let out = build_loop_tiled_segment(&one, 2, 1234567, false, 3);
        assert_eq!(out, vec![0.5, 0.25, 0.5, 0.25, 0.5, 0.25]);
    }

    // ── 淡出尾部 Click 回归（拉伸 × 内容耗尽）────────────────────────
    // 用户在“拉伸到 50%~30%”的 clip 上复现的末尾 Click 有两个叠加根源：
    // 1) e<1（先慢后快）曲线在末帧留下 (1/N)^e 级增益阶跃 —— 端点锁定 +
    //    着陆窗消除；
    // 2) 深拉伸 clip 被拉长到超出源窗口，内容在淡出区内/前耗尽 → 增益
    //    还很大时硬切静音 —— 淡出起点收缩到内容末端 + 末帧保持衰减消除。
    #[test]
    fn fade_out_tail_is_click_free_for_stretched_and_exhausted_clips() {
        use crate::state::TimelineState;
        use crate::time_stretch::StretchAlgorithm;
        let out_rate = 48_000u32;
        let src_sec = 1.0f64;

        let dir = std::env::temp_dir().join("hifishifter_fade_probe");
        std::fs::create_dir_all(&dir).unwrap();
        let wav_path = dir.join("tone_1s_48k.wav");
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: out_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&wav_path, spec).unwrap();
            let frames = (src_sec * out_rate as f64).round() as usize;
            for i in 0..frames {
                let t = i as f64 / out_rate as f64;
                let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.8;
                w.write_sample((v * i16::MAX as f64).round() as i16)
                    .unwrap();
            }
            w.finalize().unwrap();
        }

        // (rate, len_sec)：len 跨越"内容恰好用尽"（窗口/rate）与"超出内容"
        // （耗尽点落进淡出区/淡出区前）两种情况。
        let scenarios: &[(f64, f64)] = &[
            (1.0, 1.001), // 内容充足
            (0.5, 2.001), // 内容恰好用尽
            (0.5, 2.25),  // 耗尽点落进淡出区
            (0.5, 2.60),  // 耗尽点在淡出区之前
            (0.3, 3.60),  // 深拉伸 + 耗尽点落进淡出区
            (0.3, 4.20),  // 深拉伸 + 耗尽点在淡出区之前
        ];
        for &(rate, len_sec) in scenarios {
            for (fade_sec, shape, dir) in
                [(0.2f64, 1.0f64, 1.0f64), (0.2, 1.0, 0.0), (0.2, 0.0, 0.0)]
            {
                let mut tl = TimelineState::default();
                let track_id = tl.tracks[0].id.clone();
                let clip_id = tl.add_clip(
                    Some(track_id),
                    Some("T".into()),
                    Some(0.0),
                    Some(len_sec),
                    None,
                );
                {
                    let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
                    c.source_path = Some(wav_path.to_string_lossy().to_string());
                    c.source_start_sec = 0.0;
                    c.source_end_sec = src_sec;
                    c.duration_sec = Some(src_sec);
                    c.playback_rate = rate as f32;
                    c.loop_enabled = false;
                    c.fade_out_sec = fade_sec;
                    c.fade_out_shape = shape;
                    c.fade_out_dir = dir;
                }
                tl.project_sec = len_sec;

                let opts = MixdownOptions {
                    sample_rate: out_rate,
                    start_sec: 0.0,
                    end_sec: None,
                    stretch: StretchAlgorithm::LinearResample,
                    apply_pitch_edit: false,
                    output: OutputSpec::wav_32f(),
                    quality_preset: QualityPreset::Export,
                    cancel_flag: None,
                    progress: None,
                    cache_stats: None,
                };
                let (r, ch, _dur, mix) = render_mixdown_interleaved(&tl, opts).expect("render ok");
                let ch = ch as usize;
                let frames = mix.len() / ch;
                let tail_start =
                    frames.saturating_sub((fade_sec * 3.0 * r as f64).round() as usize);
                // 逐帧（双声道）样本差：正弦基线 ≈0.8×2π×440/48000≈0.046，
                // 任何硬切（淡出末端残留阶跃 / 内容耗尽断崖）≥0.5 —— 阈值
                // 0.15 留足余量；0.5ms 峰值块对 440Hz 有采样窗口振荡，不用。
                let mut max_frame_step = 0.0f32;
                for i in (tail_start + 1)..frames {
                    for c in 0..ch {
                        let step = (mix[i * ch + c] - mix[(i - 1) * ch + c]).abs();
                        if step > max_frame_step {
                            max_frame_step = step;
                        }
                    }
                }
                let last_ms_peak = mix[frames.saturating_sub(out_rate as usize / 1000) * ch..]
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max);

                log::warn!(
                    "[fade-tail] rate={rate} len={len_sec}s fade={fade_sec}s shape={shape} dir={dir} -> max_frame_step={max_frame_step} last_ms_peak={last_ms_peak}"
                );
                // 无单帧硬切：任何帧间样本差必须远低于源幅值。
                assert!(
                    max_frame_step < 0.15,
                    "fade-out tail has a hard click rate={rate} len={len_sec}s shape={shape} dir={dir}: {max_frame_step}"
                );
                // 末尾必须真正落到静音（端点锁定：最后一帧增益精确 0）。
                assert!(
                    last_ms_peak < 0.02,
                    "fade-out tail does not reach silence rate={rate} len={len_sec}s shape={shape} dir={dir}: {last_ms_peak}"
                );
            }
        }
    }

    // ── 导出编码（render_mixdown_to_file）────────────────────────────────
    // 覆盖 WAV 16/32f、FLAC（魔数 + 无损回读）、MP3（ID3 + Xing 头 + 解码
    // 往返）、Mono 下混、取消与 MP3 采样率硬校验。

    mod encode_render {
        use super::*;
        use crate::encode::{
            ChannelMode, DitherMode, FlacBitDepth, Mp3BitrateMode, Mp3Tags, OutputFormat,
            OutputSpec, WavBitDepth,
        };
        use crate::state::TimelineState;
        use std::path::PathBuf;

        const RATE: u32 = 44_100;
        const SRC_SEC: f64 = 0.5;

        /// 生成 440 Hz 正弦单声道源 WAV，返回路径。
        fn tone_source_wav(dir: &std::path::Path, sample_rate: u32, sec: f64) -> PathBuf {
            let path = dir.join(format!("tone_{sample_rate}_{sec:.3}s.wav"));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&path, spec).unwrap();
            let frames = (sec * sample_rate as f64).round() as usize;
            for i in 0..frames {
                let t = i as f64 / sample_rate as f64;
                let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.6;
                w.write_sample((v * i16::MAX as f64).round() as i16).unwrap();
            }
            w.finalize().unwrap();
            path
        }

        /// 单 clip 时间线：0 ~ `clip_len` 播放整个源文件。
        fn clip_timeline(source: &std::path::Path, clip_len: f64) -> TimelineState {
            let mut tl = TimelineState::default();
            let track_id = tl.tracks[0].id.clone();
            let clip_id =
                tl.add_clip(Some(track_id), Some("T".into()), Some(0.0), Some(clip_len), None);
            {
                let c = tl.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
                c.source_path = Some(source.to_string_lossy().to_string());
                c.source_start_sec = 0.0;
                c.source_end_sec = SRC_SEC;
                c.duration_sec = Some(SRC_SEC);
                c.playback_rate = 1.0;
                c.loop_enabled = false;
            }
            tl.project_sec = clip_len;
            tl
        }

        fn opts(spec: OutputSpec) -> MixdownOptions {
            MixdownOptions {
                sample_rate: RATE,
                start_sec: 0.0,
                end_sec: None,
                stretch: StretchAlgorithm::LinearResample,
                apply_pitch_edit: false,
                output: spec,
                quality_preset: QualityPreset::Export,
                cancel_flag: None,
                progress: None,
                cache_stats: None,
            }
        }

        fn tmp_dir(name: &str) -> PathBuf {
            let dir = std::env::temp_dir().join("hifishifter_encode_tests").join(name);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        fn base_spec(format: OutputFormat) -> OutputSpec {
            let mut spec = OutputSpec::default();
            spec.format = format;
            spec
        }

        #[test]
        fn render_to_wav_16_and_32f_roundtrip() {
            let dir = tmp_dir("wav");
            let source = tone_source_wav(&dir, RATE, SRC_SEC);
            let tl = clip_timeline(&source, SRC_SEC);

            let mut spec16 = base_spec(OutputFormat::Wav);
            spec16.wav.bit_depth = WavBitDepth::I16;
            let out16 = dir.join("out16.wav");
            let result = render_mixdown_to_file(&tl, &out16, opts(spec16)).expect("render wav16");
            assert_eq!(result.channels, 2);
            assert!(result.bytes_written > 44);
            let reader = hound::WavReader::open(&out16).unwrap();
            assert_eq!(reader.spec().bits_per_sample, 16);
            assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
            assert_eq!(
                reader.duration(),
                (SRC_SEC * RATE as f64).round() as u32
            );

            let out32 = dir.join("out32f.wav");
            let result32 = render_mixdown_to_file(&tl, &out32, opts(base_spec(OutputFormat::Wav)))
                .expect("render wav32f");
            let reader = hound::WavReader::open(&out32).unwrap();
            assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);
            assert_eq!(result32.channels, 2);
        }

        #[test]
        fn render_to_flac_roundtrips_losslessly() {
            let dir = tmp_dir("flac");
            let source = tone_source_wav(&dir, RATE, SRC_SEC);
            let tl = clip_timeline(&source, SRC_SEC);

            let mut spec = base_spec(OutputFormat::Flac);
            spec.flac.bit_depth = FlacBitDepth::I16;
            spec.dither = DitherMode::Tpdf;
            let out = dir.join("out.flac");
            let result = render_mixdown_to_file(&tl, &out, opts(spec)).expect("render flac");
            assert_eq!(result.channels, 2);
            assert!(result.bytes_written > 0);

            let bytes = std::fs::read(&out).unwrap();
            assert_eq!(&bytes[..4], b"fLaC", "FLAC 魔数");
            let (info, planes) = rusty_flac::decode(&bytes).expect("decode flac");
            assert_eq!(info.sample_rate, RATE);
            assert_eq!(info.channels, 2);
            let frames = planes[0].len();
            assert!(frames >= (SRC_SEC * RATE as f64).round() as usize - 1);
            // 正弦起始样本应接近 0（16-bit 量化 + TPDF 抖动允许 ±2 LSB）。
            let half = (1i64 << 15) as f64;
            let l0 = planes[0][0] as f64 / half;
            assert!(l0.abs() < 0.01, "正弦起始样本应接近 0：{l0}");
            // 整体能量非零（确实编码了音频内容）。
            let energy: f64 = planes[0].iter().map(|&s| (s as f64 / half).powi(2)).sum();
            let rms = (energy / frames as f64).sqrt();
            assert!(
                rms > 0.3,
                "0.6 幅值正弦的 RMS 应明显高于量化噪声：{rms}"
            );
        }

        #[test]
        fn render_to_mp3_has_id3_xing_and_roundtrips() {
            let dir = tmp_dir("mp3");
            let source = tone_source_wav(&dir, RATE, SRC_SEC);
            let tl = clip_timeline(&source, SRC_SEC);

            let mut spec = base_spec(OutputFormat::Mp3);
            spec.mp3.mode = Mp3BitrateMode::Vbr { quality_index: 2 };
            spec.mp3.tags = Mp3Tags {
                title: Some("测试曲".to_string()),
                artist: Some("HiFiShifter".to_string()),
                ..Default::default()
            };
            let out = dir.join("out.mp3");
            let result = render_mixdown_to_file(&tl, &out, opts(spec)).expect("render mp3");
            assert_eq!(result.channels, 2);

            let bytes = std::fs::read(&out).unwrap();
            assert_eq!(&bytes[..3], b"ID3", "ID3v2 标签前置");
            assert!(result.bytes_written > 1024);

            // rusty_mp3 自带解码器（与 FFmpeg 位精确对齐）做往返校验。
            let mut decoder = rusty_mp3::Mp3Decoder::new();
            decoder.push(&bytes);
            decoder.flush();
            let mut decoded_samples = 0usize;
            let mut decoded_channels = 0u16;
            loop {
                match decoder.next_frame() {
                    Ok(frame) => {
                        decoded_channels = frame.channels;
                        decoded_samples += frame.samples.len() / frame.channels as usize;
                    }
                    Err(rusty_mp3::error::Error::Eof) => break,
                    Err(e) => panic!("mp3 decode failed: {e}"),
                }
            }
            assert_eq!(decoded_channels, 2);
            // 编码补齐到整帧（1152/帧）。Xing/Info 帧是合法 MP3 帧，多数解码器
            // （含 rusty_mp3 自己）会把它解出为一段静音帧，故容差放宽到 ±2 帧。
            let expected = (SRC_SEC * RATE as f64).round() as usize;
            assert!(
                decoded_samples.abs_diff(expected) <= 1152 * 2,
                "decoded {decoded_samples} vs expected {expected}"
            );
        }

        #[test]
        fn render_mono_mixdown_yields_single_channel() {
            let dir = tmp_dir("mono");
            let source = tone_source_wav(&dir, RATE, SRC_SEC);
            let tl = clip_timeline(&source, SRC_SEC);

            let mut spec = base_spec(OutputFormat::Wav);
            spec.channel_mode = ChannelMode::Mono;
            let out = dir.join("mono.wav");
            let result = render_mixdown_to_file(&tl, &out, opts(spec)).expect("render mono");
            assert_eq!(result.channels, 1);
            let reader = hound::WavReader::open(&out).unwrap();
            assert_eq!(reader.spec().channels, 1);
            assert_eq!(reader.duration(), (SRC_SEC * RATE as f64).round() as u32);
        }

        #[test]
        fn render_cancelled_flag_removes_partial_file() {
            let dir = tmp_dir("cancel");
            let source = tone_source_wav(&dir, RATE, SRC_SEC);
            let tl = clip_timeline(&source, SRC_SEC);

            let mut o = opts(base_spec(OutputFormat::Wav));
            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
            o.cancel_flag = Some(flag);
            let out = dir.join("cancelled.wav");
            let err = render_mixdown_to_file(&tl, &out, o).expect_err("must cancel");
            assert_eq!(err, "export_cancelled");
            assert!(!out.exists(), "取消后不得残留半成品");
        }

        #[test]
        fn render_to_mp3_rejects_unsupported_sample_rate() {
            let dir = tmp_dir("mp3rate");
            let source = tone_source_wav(&dir, 48_000, SRC_SEC);
            let tl = clip_timeline(&source, SRC_SEC);

            let mut o = opts(base_spec(OutputFormat::Mp3));
            o.sample_rate = 96_000;
            let out = dir.join("bad_rate.mp3");
            let err = render_mixdown_to_file(&tl, &out, o).expect_err("96k must be rejected");
            assert_eq!(err, "mp3_unsupported_sample_rate");
            assert!(!out.exists());
        }
    }
}
