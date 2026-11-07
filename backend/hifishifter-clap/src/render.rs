//! CLAP 外壳的**音频输出**：把捕获时间线渲染成 PCM，并在 `process` 里替换输出。
//!
//! 【为什么需要这一层】在此之前 CLAP 外壳是纯直通（`process_block` 把输入抄到输出）。
//! 捕获、参数编辑、音高基线全都在工作，但**用户听到的永远是宿主原声** —— 界面上画满
//! 曲线，声音一点没变。这个模块把那条断掉的输出接上。
//!
//! 【为什么是"离线渲染 + 播放"而不是逐块合成】VST3 那条路是逐块喂处理器（快照 +
//! 双率发布 + 预算），它依赖整个 `render` 子系统（约 2 万行）。CLAP 阶段只需要
//! "改完能听到"，于是走内核已有的整段渲染入口
//! （`mixdown::render_mixdown_with_pcm`，与 App 导出同一份实现）：后台线程把整条
//! 时间线渲染成 PCM，音频线程只做一次内存拷贝。代价是编辑后要等一次重渲染，
//! 收益是**没有第二条 DSP 通路**，听到的与导出的是同一份结果。
//!
//! 【为什么编辑要防抖】参数曲线是拖拽出来的：一次拖动会发几十条 `set_param_frames`。
//! 每条都重渲染等于把 CPU 打满，而且用户还在拖的时候没有意义。所以请求只置脏，
//! 后台线程等 `DEBOUNCE_MS` 没新请求才真正开跑。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arc_swap::ArcSwapOption;
use hifishifter_capture::types::CapturePcm;
use hifishifter_kernel::encode::OutputSpec;
use hifishifter_kernel::mixdown::{render_mixdown_with_pcm, MixdownOptions, MixdownPcm};
use hifishifter_kernel::state::TimelineState;
use hifishifter_kernel::time_stretch::StretchAlgorithm;

/// 编辑停止多久之后才真正开渲染。
const DEBOUNCE_MS: u64 = 600;

/// 一段渲染好的输出。
pub struct RenderTape {
    pub sample_rate: u32,
    /// 声道数（渲染入口固定立体声）。
    pub channels: usize,
    /// 交错样本。
    pub interleaved: Vec<f32>,
    /// 帧数（整条工程）。
    pub frames: usize,
    /// 真正**属于素材**的工程帧区间（左闭右开）：
    /// 之外的帧宿主原声直通（与 Capture 的 `outside_passthrough` 同义）。
    pub start_frame: i64,
    pub end_frame: i64,
}

impl RenderTape {
    /// 取工程帧 `frame` 处的两个声道；越界返回 `None`（调用方按"直通"处理）。
    pub fn sample(&self, frame: i64) -> Option<[f32; 2]> {
        if frame < self.start_frame || frame >= self.end_frame {
            return None;
        }
        let index = usize::try_from(frame).ok()?;
        if index >= self.frames {
            return None;
        }
        let base = index * self.channels;
        let right = if self.channels > 1 { base + 1 } else { base };
        Some([self.interleaved[base], self.interleaved[right]])
    }
}

/// 一次待渲染的作业（时间线 + 源 PCM 的只读快照）。
struct RenderJob {
    timeline: TimelineState,
    sources: HashMap<String, Arc<CapturePcm>>,
    sample_rate: u32,
}

/// 渲染引擎：请求、防抖、后台渲染、无锁发布。
pub struct RenderEngine {
    current: ArcSwapOption<RenderTape>,
    pending: Mutex<Option<RenderJob>>,
    /// 有工作线程在跑（避免每条命令都起一个线程）。
    worker: AtomicBool,
    /// 已发布结果的代数（诊断用）。
    published: AtomicU64,
    /// 最近一次渲染失败的原因（诊断用；不让它变成静默的"没声音"）。
    last_error: Mutex<Option<String>>,
}

impl Default for RenderEngine {
    fn default() -> Self {
        Self {
            current: ArcSwapOption::from(None),
            pending: Mutex::new(None),
            worker: AtomicBool::new(false),
            published: AtomicU64::new(0),
            last_error: Mutex::new(None),
        }
    }
}

impl RenderEngine {
    /// 当前可播放的输出（音频线程调用：无锁、无分配）。
    pub fn current(&self) -> Option<Arc<RenderTape>> {
        self.current.load_full()
    }

    /// 丢掉当前输出（没有素材了 ⇒ 回到直通）。
    pub fn clear(&self) {
        self.current.store(None);
    }

    /// 已发布结果的代数（每成功渲染一次 +1）。
    pub fn generation(&self) -> u64 {
        self.published.load(Ordering::Acquire)
    }

    /// 最近一次失败原因（诊断）。
    pub fn last_error(&self) -> Option<String> {
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 请求一次渲染（防抖后执行）。
    ///
    /// 【为什么接收者是 `&Arc<Self>`】工作线程要活过这次调用，只能持有引擎的
    /// 一份引用计数；裸地址会在"渲染没跑完用户就关掉窗口"时变成 use-after-free。
    pub fn request(
        self: &Arc<Self>,
        timeline: TimelineState,
        sources: HashMap<String, Arc<CapturePcm>>,
        sample_rate: u32,
    ) {
        if sample_rate == 0 {
            return;
        }
        *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = Some(RenderJob {
            timeline,
            sources,
            sample_rate,
        });
        if self.worker.swap(true, Ordering::AcqRel) {
            return;
        }
        let engine = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("hfs-clap-render".into())
            .spawn(move || engine.run());
        if spawned.is_err() {
            self.worker.store(false, Ordering::Release);
        }
    }

    /// 工作线程：等防抖窗口、取作业、渲染、发布，直到没有待办。
    fn run(&self) {
        loop {
            std::thread::sleep(Duration::from_millis(DEBOUNCE_MS));
            let job = {
                let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
                match pending.take() {
                    Some(job) => job,
                    None => {
                        self.worker.store(false, Ordering::Release);
                        return;
                    }
                }
            };
            match render(&job) {
                Ok(tape) => {
                    let frames = tape.frames;
                    self.current.store(Some(Arc::new(tape)));
                    *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    let generation = self.published.fetch_add(1, Ordering::AcqRel) + 1;
                    log::info!(
                        "[clap][render] published {frames} frame(s) at {} Hz (generation {generation})",
                        job.sample_rate
                    );
                }
                Err(error) => {
                    log::warn!("[clap][render] render failed: {error}");
                    *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                }
            }
        }
    }
}

/// 把时间线渲染成交错立体声。
fn render(job: &RenderJob) -> Result<RenderTape, String> {
    let end_sec = job.timeline.project_sec.max(0.0);
    if end_sec <= 0.0 || job.timeline.clips.is_empty() {
        return Err("nothing to render".into());
    }
    let sources: HashMap<String, MixdownPcm> = job
        .sources
        .iter()
        .map(|(id, pcm)| {
            let channels = pcm.planes.len().clamp(1, 2);
            let frames = pcm.planes.first().map(Vec::len).unwrap_or(0);
            let mut interleaved = Vec::with_capacity(frames * channels);
            for index in 0..frames {
                for plane in pcm.planes.iter().take(channels) {
                    interleaved.push(plane[index]);
                }
            }
            (
                id.clone(),
                MixdownPcm {
                    sample_rate: pcm.sample_rate,
                    channels: channels as u16,
                    samples: Arc::new(interleaved),
                },
            )
        })
        .collect();
    let options = MixdownOptions {
        sample_rate: job.sample_rate,
        start_sec: 0.0,
        end_sec: Some(end_sec),
        stretch: StretchAlgorithm::SignalsmithStretch,
        // 【为什么恒为 true】参数曲线就是 Capture 形态下"处理"的全部内容；
        // 关掉它等于渲染一份没人改过的原声。
        apply_pitch_edit: true,
        // 这条路径只取内存里的交错样本，不写盘；编码描述因此只是一个占位。
        output: OutputSpec::wav_32f(),
        quality_preset: Default::default(),
        cancel_flag: None,
        progress: None,
        cache_stats: None,
    };
    // 【素材的真实跨度】混音从 0 秒开始（时间线的工程坐标），但素材可能从第 30 秒
    // 才开始 —— 之前那些帧必须留给宿主原声，否则"插上插件前 30 秒全静音"。
    let start_sec = job
        .timeline
        .clips
        .iter()
        .map(|clip| clip.start_sec.max(0.0))
        .fold(f64::INFINITY, f64::min);
    let end_sec_clips = job
        .timeline
        .clips
        .iter()
        .map(|clip| clip.start_sec.max(0.0) + clip.length_sec.max(0.0))
        .fold(f64::NEG_INFINITY, f64::max);
    let (sample_rate, channels, _duration, samples) =
        render_mixdown_with_pcm(&job.timeline, options, &sources)?;
    let channels = channels.max(1) as usize;
    if samples.is_empty() {
        return Err("the render produced no samples".into());
    }
    let rate = f64::from(sample_rate.max(1));
    Ok(RenderTape {
        sample_rate,
        channels,
        frames: samples.len() / channels,
        start_frame: (start_sec.max(0.0) * rate).round() as i64,
        end_frame: (end_sec_clips.max(0.0) * rate).round() as i64,
        interleaved: samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tape(frames: usize) -> RenderTape {
        RenderTape {
            sample_rate: 48_000,
            channels: 2,
            interleaved: (0..frames * 2).map(|value| value as f32).collect(),
            frames,
            start_frame: 0,
            end_frame: frames as i64,
        }
    }

    /// 采样按帧对齐：第 N 帧的两声道是 2N / 2N+1。
    #[test]
    fn samples_are_fetched_per_frame() {
        let tape = tape(4);
        assert_eq!(tape.sample(0), Some([0.0, 1.0]));
        assert_eq!(tape.sample(3), Some([6.0, 7.0]));
        assert_eq!(
            tape.sample(4),
            None,
            "past the end falls back to passthrough"
        );
        assert_eq!(tape.sample(-1), None, "negative project time is not ours");
    }
}
