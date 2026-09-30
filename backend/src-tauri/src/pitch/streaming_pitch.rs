//! 超长素材的流式音高分析。
//!
//! ## 为什么需要它
//!
//! 直筒实现（一次解码整份 → 整份单声道 → 整份去直流副本 → 整份 FCPE mel）会同时
//! 持有数份与素材等长的缓冲：1 小时 44.1 kHz 立体声素材的峰值约 1.9 GB，且**与
//! 素材长度线性相关** —— 导入超长音频时内存爆炸正是这条路径。
//!
//! 本模块把同一套分析改成「顺序流式解码 + 定长分块分析」：工作集只与块长（默认
//! 30 s）有关，与文件长度无关。
//!
//! 分块参数复用 `pitch_config` 里既有的 `chunk_sec` / `chunk_ctx_sec`
//! （含 `HIFISHIFTER_PITCH_CHUNK_SEC` / `HIFISHIFTER_PITCH_CHUNK_CTX_SEC` 环境
//! 覆盖）—— 那套设计本来就在仓库里，只是一直没有接线。
//!
//! ## 块边界为什么不产生接缝
//!
//! 分析帧栅格由实数 hop 定义：`hop_an = analysis_rate · fp / 1000` 个样本/帧
//! （44.1 kHz + 5 ms → 220.5）。块边界取在**源帧坐标**上 `step` 的整数倍处，其中
//! `step` 是使 `step · analysis_rate / in_rate` 恰为 `hop_an` 整数倍的最小源帧数
//! （44.1 kHz + 5 ms → 441 源帧 = 2 个分析帧）。于是每块切片起点都精确落在全局帧
//! 栅格上，相邻块保留的帧区间严格首尾相接、不重不漏 —— 拼接结果就是全局帧序列。
//!
//! 每块额外向两侧各取 `ctx_sec` 的**真实**音频作为上下文喂给音高检测器，再把
//! 上下文的帧裁掉。这样块边缘的 reflect padding 落在被丢弃的区间里，检测器在边界
//! 处看到的是真实邻域而非被折叠的假信号。
//!
//! ## 已知的、被刻意接受的近似
//!
//! - **DC 与归一化系数取自源采样率下的降混信号**（一趟只累加标量的预扫描），
//!   而非重采样后的信号。两者只差一个全局常量偏移与增益：重采样是插值，均值几乎
//!   不变、峰值只会更低，因此该差异不改变音高检测的结构，也不会造成逐块台阶
//!   （系数是全局的，不是每块各算一份）。
//! - 块内部 `resample_f0_linear` 按块跨度做线性映射，与整份一次重采样相比存在
//!   亚帧级的相位差。因为每块的**绝对**时间位置由块序号决定，误差不累积。

use std::path::{Path, PathBuf};

// ── 数据源 ───────────────────────────────────────────────────────────────────

/// 顺序产出源采样率音频块的抽象。
///
/// 抽出这一层是为了让分块逻辑脱离文件系统做单元测试（见测试模块的
/// `MemorySource`），同时生产路径仍然走 `media::visit_media_audio_frames` 的
/// 流式解码 —— 它不累积整份文件。
pub(crate) trait SampleSource {
    /// 顺序遍历源音频块。`on_block` 收到的是**交错**样本、以及该块的采样率与声道数。
    fn for_each_block(
        &mut self,
        on_block: &mut dyn FnMut(&[f32], u32, u16) -> Result<(), String>,
    ) -> Result<(), String>;
}

/// 生产路径：直接顺序解码媒体文件。
pub(crate) struct MediaFileSource {
    pub path: PathBuf,
}

impl MediaFileSource {
    pub(crate) fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl SampleSource for MediaFileSource {
    fn for_each_block(
        &mut self,
        on_block: &mut dyn FnMut(&[f32], u32, u16) -> Result<(), String>,
    ) -> Result<(), String> {
        crate::media::visit_media_audio_frames(&self.path, None, |frame, rate, channels| {
            on_block(frame, rate, channels)
        })
        .map(|_| ())
    }
}

// ── 音高检测器 ───────────────────────────────────────────────────────────────

/// 逐帧音高检测。抽成 trait 是为了让分块/拼接逻辑可以在没有 ONNX 模型的环境下
/// 被完整测试（测试用自相关检测器即可覆盖边界对齐这一最关键的性质）。
pub(crate) trait PitchEstimator {
    /// 输入分析采样率的单声道信号，返回逐分析帧的 f0（Hz，0 = 无声）。
    fn estimate(
        &self,
        mono: &[f32],
        sample_rate: u32,
        frame_period_ms: f64,
    ) -> Result<Vec<f64>, String>;
}

/// 生产实现：FCPE ONNX 模型。
pub(crate) struct FcpeEstimator;

impl PitchEstimator for FcpeEstimator {
    fn estimate(
        &self,
        mono: &[f32],
        sample_rate: u32,
        frame_period_ms: f64,
    ) -> Result<Vec<f64>, String> {
        crate::fcpe_onnx::infer_f0_hz_f32(
            mono,
            sample_rate,
            frame_period_ms,
            crate::fcpe_onnx::FCPE_F0_MIN_HZ,
            crate::fcpe_onnx::FCPE_F0_MAX_HZ,
        )
    }
}

// ── 分块参数 ─────────────────────────────────────────────────────────────────

/// 分块参数。默认取 `pitch_config` 的全局值，测试可直接构造小值以便在短信号上
/// 覆盖多个块边界。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Chunking {
    pub chunk_sec: f64,
    pub ctx_sec: f64,
}

impl Chunking {
    pub(crate) fn from_config() -> Self {
        let cfg = crate::pitch_config::PitchAnalysisConfig::global();
        Self {
            chunk_sec: cfg.chunk_sec,
            ctx_sec: cfg.chunk_ctx_sec,
        }
    }
}

/// 流式分析请求。
pub(crate) struct StreamParams<'a> {
    pub analysis_rate: u32,
    pub frame_period_ms: f64,
    /// 是否需要音高。`false` 时跳过检测器（只产出电平，DYN 的原声基线不依赖声码器）。
    pub want_pitch: bool,
    pub chunking: Chunking,
    pub estimator: &'a dyn PitchEstimator,
    /// 每块检查一次：命中后立即停止并把 `StreamedPitch::cancelled` 置位。
    pub cancelled: &'a dyn Fn() -> bool,
}

/// 流式分析结果。
#[derive(Debug)]
pub(crate) struct StreamedPitch {
    /// 逐分析帧的 f0（Hz，0 = 无声）。`want_pitch == false` 时为空。
    pub f0_hz: Vec<f64>,
    /// 逐分析帧的原声电平（linear，与 f0 同帧率）。
    pub level: Vec<f32>,
    /// 分析期间检测到取消（调用方应丢弃结果）。
    pub cancelled: bool,
}

// ── 分析 ─────────────────────────────────────────────────────────────────────

/// 源帧/分析帧（例如 44.1 kHz + 5 ms → 220.5）。
fn source_frames_per_analysis_frame(in_rate: u32, frame_period_ms: f64) -> f64 {
    (in_rate.max(1) as f64) * (frame_period_ms.max(0.1) / 1000.0)
}

/// 使 `j · frame_src` 为整数的最小正整数 `j`，返回 `j · frame_src`。
///
/// 这个值就是源帧坐标上的对齐步长：只有落在它整数倍处的切片起点，其对应的分析
/// 采样位置才恰好是 `hop_an` 的整数倍 —— 也就是全局帧栅格上的帧边界。
fn alignment_step(frame_src: f64) -> u64 {
    if !(frame_src.is_finite() && frame_src > 0.0) {
        return 1;
    }
    for j in 1..=8192i64 {
        let v = frame_src * j as f64;
        if (v - v.round()).abs() < 1e-6 {
            return (v.round() as u64).max(1);
        }
    }
    (frame_src.round() as u64).max(1)
}

fn round_up_to_multiple(value: f64, step: u64) -> u64 {
    let step = step.max(1);
    let n = (value / step as f64).ceil().max(1.0);
    let out = n * step as f64;
    if out.is_finite() && out > 0.0 {
        out.round() as u64
    } else {
        step
    }
}

/// 一趟预扫描：求源采样率下降混信号的均值与峰值。
///
/// 只累加两个标量，不保留任何音频 —— 这是「先知道全局 DC 与归一化系数、再逐块
/// 分析」所付出的代价（多一趟顺序解码），换来的是每块不必各算一份系数而产生台阶。
fn scan_source_stats<S: SampleSource>(source: &mut S) -> Result<(f64, f64), String> {
    let mut sum = 0.0f64;
    let mut count: u64 = 0;
    let mut peak = 0.0f64;
    let mut channels_seen = 0usize;

    source.for_each_block(&mut |frame: &[f32], _rate: u32, channels: u16| {
        let ch = (channels.max(1)) as usize;
        if channels_seen != 0 && channels_seen != ch {
            // 声道数在容器内变化：无法与已累加的样本保持同一口径，停止累加。
            return Ok(());
        }
        channels_seen = ch;
        let frames = frame.len() / ch;
        for f in 0..frames {
            let base = f * ch;
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += frame[base + c];
            }
            let v = (acc / ch as f32) as f64;
            sum += v;
            count += 1;
            let a = v.abs();
            if a.is_finite() && a > peak {
                peak = a;
            }
        }
        Ok(())
    })?;

    if count == 0 {
        return Ok((0.0, 0.0));
    }
    Ok((sum / count as f64, peak))
}

/// 分块流式分析。
pub(crate) fn analyze_streaming<S: SampleSource>(
    source: &mut S,
    params: &StreamParams<'_>,
) -> Result<StreamedPitch, String> {
    let fp = params.frame_period_ms.max(0.1);

    let (mean, peak) = scan_source_stats(source)?;
    let max_abs = (peak - mean).abs().max(peak.abs());
    let scale = if max_abs.is_finite() && max_abs > 1.0 {
        (1.0 / max_abs).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let mut analyzer = ChunkAnalyzer {
        analysis_rate: params.analysis_rate.max(1),
        fp,
        mean,
        scale,
        want_pitch: params.want_pitch,
        chunking: params.chunking,
        estimator: params.estimator,
        in_rate: 0,
        in_channels: 0,
        frame_src: 0.0,
        step: 1,
        chunk_src: 1,
        ctx_src: 0,
        frames_per_chunk: 1,
        staged: Vec::new(),
        staged_start: 0,
        next_chunk: 0,
        f0_hz: Vec::new(),
        level: Vec::new(),
        cancelled: false,
    };

    source
        .for_each_block(&mut |frame: &[f32], rate: u32, channels: u16| {
            if analyzer.cancelled {
                return Err(CANCEL_SENTINEL.to_string());
            }
            analyzer.push(frame, rate, channels, params.cancelled)
        })
        .or_else(|e| {
            if e == CANCEL_SENTINEL {
                Ok(())
            } else {
                Err(e)
            }
        })?;

    analyzer.finish(params.cancelled);

    Ok(StreamedPitch {
        f0_hz: analyzer.f0_hz,
        level: analyzer.level,
        cancelled: analyzer.cancelled,
    })
}

/// 主动停止流式解码用的内部哨兵（不会外泄给调用方）。
const CANCEL_SENTINEL: &str = "__hifi_pitch_stream_cancelled__";

struct ChunkAnalyzer<'a> {
    analysis_rate: u32,
    fp: f64,
    mean: f64,
    scale: f64,
    want_pitch: bool,
    chunking: Chunking,
    estimator: &'a dyn PitchEstimator,

    in_rate: u32,
    in_channels: usize,
    frame_src: f64,
    step: u64,
    /// 每块推进的源帧数（`step` 的整数倍）。
    chunk_src: u64,
    /// 每侧上下文的源帧数（`step` 的整数倍）。
    ctx_src: u64,
    /// 每块保留的分析帧数；由 `chunk_src / frame_src` 精确导出。
    frames_per_chunk: usize,

    /// 已解码、尚未消费的源帧（交错 @ `in_rate`），覆盖
    /// `[staged_start, staged_start + staged.len()/in_channels)`。
    staged: Vec<f32>,
    staged_start: u64,
    next_chunk: u64,

    f0_hz: Vec<f64>,
    level: Vec<f32>,
    cancelled: bool,
}

impl ChunkAnalyzer<'_> {
    fn staged_frames(&self) -> u64 {
        if self.in_channels == 0 {
            0
        } else {
            (self.staged.len() / self.in_channels) as u64
        }
    }

    fn staged_end(&self) -> u64 {
        self.staged_start + self.staged_frames()
    }

    fn init(&mut self, rate: u32, channels: u16) {
        self.in_rate = rate.max(1);
        self.in_channels = (channels.max(1)) as usize;
        self.frame_src = source_frames_per_analysis_frame(self.in_rate, self.fp);
        self.step = alignment_step(self.frame_src);

        let chunk_frames = ((self.chunking.chunk_sec * 1000.0) / self.fp).round().max(1.0);
        let ctx_frames = ((self.chunking.ctx_sec * 1000.0) / self.fp).round().max(0.0);
        self.chunk_src = round_up_to_multiple(chunk_frames * self.frame_src, self.step);
        self.ctx_src = round_up_to_multiple(ctx_frames * self.frame_src, self.step);
        // `chunk_src` 是 `step` 的整数倍、`step` 是 `frame_src` 的整数倍，因此
        // 这个除法的结果是整数 —— 块的帧数与源坐标严格一致，不会逐块漂移。
        self.frames_per_chunk = ((self.chunk_src as f64) / self.frame_src).round().max(1.0) as usize;
    }

    fn push(
        &mut self,
        frame: &[f32],
        rate: u32,
        channels: u16,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), String> {
        if self.in_rate == 0 {
            self.init(rate, channels);
        }
        if rate.max(1) != self.in_rate || (channels.max(1)) as usize != self.in_channels {
            return Err(format!(
                "streaming pitch analysis: source parameters changed mid-stream (rate {rate}, channels {channels}); expected rate {}, channels {}",
                self.in_rate, self.in_channels
            ));
        }

        self.staged.extend_from_slice(frame);

        // 只要已解码的内容覆盖了某块的完整扩展窗口，就处理它并丢掉落在其前的数据。
        loop {
            let k = self.next_chunk;
            let want_end = (k + 1)
                .saturating_mul(self.chunk_src)
                .saturating_add(self.ctx_src);
            if self.staged_end() < want_end {
                break;
            }
            if cancelled() {
                self.cancelled = true;
                return Err(CANCEL_SENTINEL.to_string());
            }
            self.process_chunk(k, want_end);
            // 下一块还需要 `(k+1)·chunk_src − ctx_src` 起的数据，之前的可以丢。
            let keep_from = (k + 1)
                .saturating_mul(self.chunk_src)
                .saturating_sub(self.ctx_src)
                .max(self.staged_start);
            self.drop_prefix_before(keep_from);
            self.next_chunk = k + 1;
        }
        Ok(())
    }

    fn finish(&mut self, cancelled: &dyn Fn() -> bool) {
        if self.in_rate == 0 {
            return;
        }
        loop {
            let k = self.next_chunk;
            if k.saturating_mul(self.chunk_src) >= self.staged_end() {
                break;
            }
            if cancelled() {
                self.cancelled = true;
                return;
            }
            self.process_chunk(k, self.staged_end());
            self.next_chunk = k + 1;
        }
    }

    fn drop_prefix_before(&mut self, keep_from: u64) {
        if keep_from <= self.staged_start {
            return;
        }
        let drop_frames = (keep_from - self.staged_start).min(self.staged_frames());
        if drop_frames == 0 {
            return;
        }
        let drop_samples = drop_frames as usize * self.in_channels;
        self.staged.drain(..drop_samples);
        self.staged_start += drop_frames;
    }

    /// 处理第 `k` 块。`win_end` 是窗口右端（正常流程下等于
    /// `(k+1)·chunk_src + ctx_src`；收尾时被截到实际解码长度）。
    fn process_chunk(&mut self, k: u64, win_end: u64) {
        let win_start = k
            .saturating_mul(self.chunk_src)
            .saturating_sub(self.ctx_src)
            .max(self.staged_start);
        let win_end = win_end.min(self.staged_end()).max(win_start);
        if win_end <= win_start {
            return;
        }

        let from = ((win_start - self.staged_start) as usize) * self.in_channels;
        let to = (((win_end - self.staged_start) as usize) * self.in_channels).min(self.staged.len());
        if to <= from {
            return;
        }

        // 源采样率 → 分析采样率（同率时该函数直接复制，无重采样开销）。
        let resampled = crate::mixdown::linear_resample_interleaved(
            &self.staged[from..to],
            self.in_channels,
            self.in_rate,
            self.analysis_rate,
        );
        let ch = self.in_channels;
        let frames = resampled.len() / ch;
        if frames < 2 {
            return;
        }

        // ── 降混为单声道（f32，分析采样率）────────────────────────────────
        let mut mono: Vec<f32> = Vec::with_capacity(frames);
        for f in 0..frames {
            let base = f * ch;
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += resampled[base + c];
            }
            mono.push(acc / ch as f32);
        }
        drop(resampled);

        // 本块窗口的起点落在全局帧栅格的第 `frames_before` 个帧边界上；块 k 保留的
        // 是全局帧 `[k·F, (k+1)·F)`，因此窗口内应保留的帧从 `k·F − frames_before` 起。
        let frames_before = (win_start as f64 / self.frame_src).round();
        let keep_from_global = (k as f64) * (self.frames_per_chunk as f64);
        let local_start = (keep_from_global - frames_before).round().max(0.0) as usize;
        let local_end = local_start + self.frames_per_chunk;

        // ── 逐帧电平（DYN 的原声基线）──────────────────────────────────────
        // 用**去直流但未归一化**的信号：归一化是为音高检测准备的动态范围拉伸，
        // 若把它算进电平，响度就会随"这一片段有多响"被反复改写。
        {
            let dc_removed: Vec<f32> =
                mono.iter().map(|&v| (v as f64 - self.mean) as f32).collect();
            let lv =
                crate::pitch_clip::compute_frame_levels(&dc_removed, self.analysis_rate, self.fp);
            let end = local_end.min(lv.len());
            if local_start < end {
                self.level.extend_from_slice(&lv[local_start..end]);
            }
        }

        // ── 音高 ─────────────────────────────────────────────────────────
        if self.want_pitch {
            for v in mono.iter_mut() {
                *v = (((*v as f64 - self.mean) * self.scale) as f32).clamp(-1.0, 1.0);
            }
            match self.estimator.estimate(&mono, self.analysis_rate, self.fp) {
                Ok(f0) => {
                    let end = local_end.min(f0.len());
                    if local_start < end {
                        self.f0_hz.extend_from_slice(&f0[local_start..end]);
                    }
                }
                Err(e) => {
                    log::error!("[pitch] pitch estimation failed on chunk {k}: {e}");
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// 内存信号源：按固定块长喂数据，让分块逻辑脱离文件系统可测。
    struct MemorySource {
        rate: u32,
        channels: u16,
        samples: Vec<f32>,
        block_frames: usize,
    }

    impl SampleSource for MemorySource {
        fn for_each_block(
            &mut self,
            on_block: &mut dyn FnMut(&[f32], u32, u16) -> Result<(), String>,
        ) -> Result<(), String> {
            let ch = (self.channels.max(1)) as usize;
            let block = self.block_frames.max(1) * ch;
            let mut start = 0usize;
            while start < self.samples.len() {
                let end = (start + block).min(self.samples.len());
                on_block(&self.samples[start..end], self.rate, self.channels)?;
                start = end;
            }
            Ok(())
        }
    }

    /// 过零率检测器：与 ONNX 无关、确定性、每帧 O(窗长)。
    ///
    /// 用它而不是 FCPE，是为了让测试在没有模型的环境下也能跑，并把「分块/拼接是否
    /// 走样」与「模型推理是否精确」两件事分开 —— 前者才是这里要验证的。
    struct ZeroCrossEstimator;

    impl PitchEstimator for ZeroCrossEstimator {
        fn estimate(
            &self,
            mono: &[f32],
            sample_rate: u32,
            frame_period_ms: f64,
        ) -> Result<Vec<f64>, String> {
            let sr = sample_rate.max(1) as f64;
            let hop_f = (frame_period_ms.max(0.1) / 1000.0) * sr;
            // 与 `compute_frame_levels` 同一帧栅格定义，便于两者对位比较。
            let total = (((mono.len() as f64) / hop_f) - 1e-9).ceil().max(1.0) as usize;
            let win = ((0.02 * sr).round() as usize).max(2);
            let mut out = Vec::with_capacity(total);
            for j in 0..total {
                let center = (j as f64) * hop_f + hop_f * 0.5;
                let start = (center - win as f64 * 0.5).max(0.0) as usize;
                let end = ((center + win as f64 * 0.5) as usize).min(mono.len());
                if end <= start + 2 {
                    out.push(0.0);
                    continue;
                }
                let seg = &mono[start..end];
                let mut crossings = 0usize;
                for i in 1..seg.len() {
                    if (seg[i - 1] < 0.0) != (seg[i] < 0.0) {
                        crossings += 1;
                    }
                }
                let secs = seg.len() as f64 / sr;
                out.push(if crossings >= 2 {
                    crossings as f64 / (2.0 * secs)
                } else {
                    0.0
                });
            }
            Ok(out)
        }
    }

    /// 线性扫频正弦（`f0` → `f1` Hz），幅度留足余量避免削顶。
    fn chirp(rate: u32, secs: f64, f0: f64, f1: f64) -> Vec<f32> {
        let n = (rate as f64 * secs) as usize;
        let mut phase = 0.0f64;
        (0..n)
            .map(|i| {
                let t = i as f64 / rate as f64;
                let f = f0 + (f1 - f0) * (t / secs);
                phase += 2.0 * std::f64::consts::PI * f / rate as f64;
                (0.7 * phase.sin()) as f32
            })
            .collect()
    }

    /// 左右相同 → 降混结果与输入 mono 一致，便于直接比对。
    fn stereo(mono: &[f32]) -> Vec<f32> {
        mono.iter().flat_map(|&v| [v, v]).collect()
    }

    fn run_chunked(
        samples: &[f32],
        rate: u32,
        channels: u16,
        chunk_sec: f64,
        ctx_sec: f64,
        want_pitch: bool,
    ) -> StreamedPitch {
        let mut source = MemorySource {
            rate,
            channels,
            samples: samples.to_vec(),
            block_frames: 4096,
        };
        let estimator = ZeroCrossEstimator;
        let params = StreamParams {
            analysis_rate: rate,
            frame_period_ms: 5.0,
            want_pitch,
            chunking: Chunking {
                chunk_sec,
                ctx_sec,
            },
            estimator: &estimator,
            cancelled: &|| false,
        };
        analyze_streaming(&mut source, &params).expect("streaming analysis failed")
    }

    /// 整份参考：与分块路径同一套降混 / 去直流 / 电平口径，只是不切块。
    fn reference_level_and_f0(samples: &[f32], rate: u32, channels: u16) -> (Vec<f32>, Vec<f64>) {
        let ch = channels.max(1) as usize;
        let frames = samples.len() / ch;
        let mut mono: Vec<f32> = Vec::with_capacity(frames);
        for f in 0..frames {
            let base = f * ch;
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += samples[base + c];
            }
            mono.push(acc / ch as f32);
        }
        let mean = mono.iter().map(|&v| v as f64).sum::<f64>() / mono.len().max(1) as f64;
        let peak = mono.iter().map(|&v| (v as f64).abs()).fold(0.0f64, f64::max);
        let max_abs = (peak - mean).abs().max(peak.abs());
        let scale = if max_abs > 1.0 { 1.0 / max_abs } else { 1.0 };

        let dc_removed: Vec<f32> = mono.iter().map(|&v| (v as f64 - mean) as f32).collect();
        let level = crate::pitch_clip::compute_frame_levels(&dc_removed, rate, 5.0);

        let normalized: Vec<f32> = mono
            .iter()
            .map(|&v| (((v as f64 - mean) * scale) as f32).clamp(-1.0, 1.0))
            .collect();
        let f0 = ZeroCrossEstimator
            .estimate(&normalized, rate, 5.0)
            .expect("reference estimate failed");
        (level, f0)
    }

    #[test]
    fn alignment_step_matches_known_rates() {
        // 44.1 kHz + 5 ms：220.5 源帧/分析帧 → 最小公共整数步长 441（= 2 帧）。
        assert_eq!(alignment_step(220.5), 441);
        // 48 kHz + 5 ms：240 源帧/分析帧，本身就是整数。
        assert_eq!(alignment_step(240.0), 240);
        // 22.05 kHz + 5 ms：110.25 → 441（= 4 帧）。
        assert_eq!(alignment_step(110.25), 441);
    }

    /// 电平曲线是逐帧从信号算出的确定量（20 ms 窗峰值）。若分块的帧区间发生
    /// 重叠、遗漏或整体错位，窗口覆盖的样本就变了，数值必然对不上 —— 因此这是
    /// 对「块边界对齐」最直接的逐位验证。
    #[test]
    fn chunked_level_is_bit_identical_to_whole_file() {
        let rate = 44_100u32;
        let mono = chirp(rate, 3.0, 200.0, 500.0);
        let samples = stereo(&mono);

        let (ref_level, _) = reference_level_and_f0(&samples, rate, 2);
        // 0.5 s 块 + 0.05 s 上下文 → 3 秒信号上跨越多个块边界。
        let streamed = run_chunked(&samples, rate, 2, 0.5, 0.05, false);

        assert_eq!(
            streamed.level.len(),
            ref_level.len(),
            "chunked level frame count differs from whole-file"
        );
        assert_eq!(
            streamed.level, ref_level,
            "chunked level diverged from whole-file: frame alignment is broken"
        );
    }

    /// 更极端的切分（0.2 s 块 + 0.3 s 上下文：上下文比块还长，相邻窗口大幅重叠）
    /// 同样必须与整份结果一致。
    #[test]
    fn overlapping_context_still_matches_whole_file() {
        let rate = 44_100u32;
        let mono = chirp(rate, 2.0, 220.0, 440.0);
        let samples = stereo(&mono);

        let (ref_level, _) = reference_level_and_f0(&samples, rate, 2);
        let streamed = run_chunked(&samples, rate, 2, 0.2, 0.3, false);

        assert_eq!(streamed.level.len(), ref_level.len());
        assert_eq!(streamed.level, ref_level);
    }

    /// 音高曲线在**块边界**处必须与整份结果一致。
    ///
    /// 电平的逐位相等已经证明帧区间不重不漏，这里进一步验证音高检测在边界附近
    /// 看到的邻域是真实音频：检测窗口横跨块边界时，若上下文缺失（例如用零填充或
    /// 只用块内数据），边界处的 f0 会明显偏离整份结果。
    #[test]
    fn chunked_pitch_matches_whole_file_at_chunk_boundaries() {
        let rate = 44_100u32;
        let mono = chirp(rate, 3.0, 400.0, 800.0);
        let samples = stereo(&mono);

        let (_, ref_f0) = reference_level_and_f0(&samples, rate, 2);
        let streamed = run_chunked(&samples, rate, 2, 0.5, 0.05, true);

        assert_eq!(streamed.f0_hz.len(), ref_f0.len());

        let mut worst = 0.0f64;
        let mut sum = 0.0f64;
        let mut counted = 0usize;
        for (a, b) in streamed.f0_hz.iter().zip(ref_f0.iter()) {
            if *b <= 1.0 || *a <= 1.0 {
                continue;
            }
            let rel = ((a - b) / b).abs();
            worst = worst.max(rel);
            sum += rel;
            counted += 1;
        }
        assert!(counted > 100, "too few voiced frames to compare");
        let mean = sum / counted as f64;
        assert!(mean < 0.01, "mean relative f0 error {mean} too large");
        assert!(worst < 0.05, "worst relative f0 error {worst} too large");

        // 0.5 s 块 @ 5 ms 帧 → 每块 100 帧，边界落在 100 的整数倍处。
        // 边界两侧各看 ±4 帧：这些帧的检测窗口横跨块边界，最依赖上下文。
        let frames_per_chunk = 100usize;
        let mut boundary_checked = 0usize;
        for k in 1..(streamed.f0_hz.len() / frames_per_chunk) {
            let center = k * frames_per_chunk;
            for j in center.saturating_sub(4)..(center + 4).min(ref_f0.len()) {
                let (a, b) = (streamed.f0_hz[j], ref_f0[j]);
                if b <= 1.0 || a <= 1.0 {
                    continue;
                }
                boundary_checked += 1;
                let rel = ((a - b) / b).abs();
                assert!(
                    rel < 0.05,
                    "f0 diverges by {rel:.4} at chunk boundary frame {j} (chunked {a:.1} vs whole {b:.1})"
                );
            }
        }
        assert!(
            boundary_checked > 20,
            "boundary check did not actually cover enough frames ({boundary_checked})"
        );
    }

    /// 分块路径与整份路径的帧数必须一致（不重不漏），覆盖多种长度以跨越不同的
    /// 尾块形态。
    #[test]
    fn chunked_and_whole_agree_on_frame_count() {
        let rate = 44_100u32;
        for secs in [0.4f64, 1.0, 2.5, 7.0] {
            let mono = chirp(rate, secs, 200.0, 400.0);
            let samples = stereo(&mono);
            let (ref_level, _) = reference_level_and_f0(&samples, rate, 2);
            let streamed = run_chunked(&samples, rate, 2, 0.5, 0.05, false);
            assert_eq!(
                streamed.level.len(),
                ref_level.len(),
                "frame count mismatch at {secs}s"
            );
        }
    }

    /// 取消应在块间生效：返回已收集的部分并置位 `cancelled`，而不是跑完整份。
    #[test]
    fn cancellation_stops_early_and_flags_result() {
        let rate = 44_100u32;
        let mono = chirp(rate, 4.0, 200.0, 400.0);
        let samples = stereo(&mono);

        let mut source = MemorySource {
            rate,
            channels: 2,
            samples: samples.clone(),
            block_frames: 4096,
        };
        let estimator = ZeroCrossEstimator;
        // 第二次查询开始返回"已取消"。
        let calls = std::cell::Cell::new(0usize);
        let cancelled = || {
            let n = calls.get() + 1;
            calls.set(n);
            n > 2
        };
        let params = StreamParams {
            analysis_rate: rate,
            frame_period_ms: 5.0,
            want_pitch: false,
            chunking: Chunking {
                chunk_sec: 0.5,
                ctx_sec: 0.05,
            },
            estimator: &estimator,
            cancelled: &cancelled,
        };
        let result = analyze_streaming(&mut source, &params).expect("should not error on cancel");

        assert!(result.cancelled, "cancellation flag not set");
        let full = run_chunked(&samples, rate, 2, 0.5, 0.05, false);
        assert!(
            result.level.len() < full.level.len(),
            "cancelled run produced the full curve ({} vs {})",
            result.level.len(),
            full.level.len()
        );
    }

    /// 源参数中途改变（容器内采样率变化）必须报错，而不是静默产出错位数据。
    #[test]
    fn mid_stream_parameter_change_is_an_error() {
        struct ChangingSource;
        impl SampleSource for ChangingSource {
            fn for_each_block(
                &mut self,
                on_block: &mut dyn FnMut(&[f32], u32, u16) -> Result<(), String>,
            ) -> Result<(), String> {
                on_block(&[0.0f32; 2048], 44_100, 2)?;
                on_block(&[0.0f32; 2048], 48_000, 2)?;
                Ok(())
            }
        }
        let estimator = ZeroCrossEstimator;
        let params = StreamParams {
            analysis_rate: 44_100,
            frame_period_ms: 5.0,
            want_pitch: false,
            chunking: Chunking {
                chunk_sec: 0.5,
                ctx_sec: 0.05,
            },
            estimator: &estimator,
            cancelled: &|| false,
        };
        let err = analyze_streaming(&mut ChangingSource, &params)
            .expect_err("rate change should be rejected");
        assert!(err.contains("changed mid-stream"), "unexpected error: {err}");
    }

    // ── Phase 4：峰值工作集必须与素材长度解耦 ──────────────────────────────

    /// 廉价方波：内存测试只关心"跑了多少分配"，不关心检测精度，因此避免
    /// 逐样本 `sin()` 的开销（数千万样本下那会拖慢基准本身）。
    fn square(rate: u32, secs: f64, period_hz: f64) -> Vec<f32> {
        let n = (rate as f64 * secs) as usize;
        let half = ((rate as f64 / period_hz) * 0.5).max(1.0) as usize;
        (0..n)
            .map(|i| if (i / half) % 2 == 0 { 0.6 } else { -0.6 })
            .collect()
    }

    /// 借用式信号源：基准里的大素材不应被再克隆一份（那会污染测量）。
    struct BorrowedSource<'a> {
        rate: u32,
        channels: u16,
        samples: &'a [f32],
        block_frames: usize,
    }

    impl SampleSource for BorrowedSource<'_> {
        fn for_each_block(
            &mut self,
            on_block: &mut dyn FnMut(&[f32], u32, u16) -> Result<(), String>,
        ) -> Result<(), String> {
            let ch = (self.channels.max(1)) as usize;
            let block = self.block_frames.max(1) * ch;
            let mut start = 0usize;
            while start < self.samples.len() {
                let end = (start + block).min(self.samples.len());
                on_block(&self.samples[start..end], self.rate, self.channels)?;
                start = end;
            }
            Ok(())
        }
    }

    fn measure_stream_peak(samples: &[f32], rate: u32, channels: u16) -> usize {
        let estimator = ZeroCrossEstimator;
        let params = StreamParams {
            analysis_rate: rate,
            frame_period_ms: 5.0,
            want_pitch: true,
            // 生产默认：30 s 块 + 0.3 s 上下文。
            chunking: Chunking {
                chunk_sec: 30.0,
                ctx_sec: 0.3,
            },
            estimator: &estimator,
            cancelled: &|| false,
        };
        let mut source = BorrowedSource {
            rate,
            channels,
            samples,
            block_frames: 8192,
        };
        let (result, peak) =
            crate::alloc_probe::measure_peak_alloc(|| analyze_streaming(&mut source, &params));
        result.expect("analysis failed");
        peak
    }

    /// 峰值工作集必须由**块长**决定，而不是素材长度：这是本模块存在的全部理由。
    ///
    /// `#[ignore]`：分配计数器是进程级的，必须独占运行。手动执行：
    ///   cargo test --lib -- --ignored --test-threads=1 pitch_memory
    #[test]
    #[ignore]
    fn pitch_memory_peak_is_bounded_and_independent_of_source_length() {
        let rate = 44_100u32;
        let channels = 2u16;

        // 60 s 与 240 s（4×）：若峰值仍与长度线性相关，后者会接近 4 倍。
        let short = stereo(&square(rate, 60.0, 300.0));
        let long = stereo(&square(rate, 240.0, 300.0));
        let long_source_bytes = long.len() * std::mem::size_of::<f32>();

        let short_peak = measure_stream_peak(&short, rate, channels);
        let long_peak = measure_stream_peak(&long, rate, channels);

        println!(
            "chunked peak: 60s = {} MiB, 240s = {} MiB (source 240s = {} MiB)",
            short_peak / (1024 * 1024),
            long_peak / (1024 * 1024),
            long_source_bytes / (1024 * 1024),
        );

        // (1) 峰值必须由块长决定：一块的工作集 = staged（源采样率交错）+ 重采样副本
        //     + 单声道 + 去直流副本 ≈ 3×块字节数。给 4 倍余量以容纳输出累积器与
        //     分配器碎片，但绝不允许它随素材长度走。
        let chunk_bytes = (30.0 * rate as f64) as usize * channels as usize * 4;
        assert!(
            long_peak < chunk_bytes * 4,
            "peak {long_peak} B exceeds the chunk-derived bound {} B",
            chunk_bytes * 4
        );

        // (2) 峰值必须**远小于**素材样本量：素材比一块长 8 倍，因此若没有流式化，
        //     峰值至少要与样本量同量级。
        assert!(
            long_peak * 2 < long_source_bytes,
            "peak {long_peak} B is not well below the {long_source_bytes} B of source samples"
        );

        // (3) 4 倍素材长度不应带来成比例的峰值增长。输出曲线随长度增长（那是必须
        //     留下的结果，240 s ≈ 0.5 MB），所以给 1.6 倍余量；线性增长会是 ~4 倍。
        assert!(
            long_peak < short_peak * 16 / 10,
            "peak grew with source length: {short_peak} B -> {long_peak} B"
        );
    }
}
