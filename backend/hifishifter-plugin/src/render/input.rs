//! 冻结的合成输入只包含Rust值和授权PCM的Arc，不借文档锁或调用宿主ARA接口。
//!
//! 【快照率为什么固定 44.1/48k】内部合成在两档各出一份快照；宿主用别的率时由
//! `PlaybackSnapshot::copy_block` 线性重采样到宿主率（`vst3.rs::setupProcessing`
//! 记录权威率，`ExtensionOwner::snapshot_index_for_rate` 选最合适的原生档：整数倍
//! 优先，否则取最近）。这是**有意的降级**：能听见、略有染色，远好于"没有快照 ⇒
//! 整条插件静默"。真正的"按宿主率渲染内部快照"（消除这次重采样）要把合成/缓存/预算
//! 全部按宿主率参数化，是独立一期工作 —— 在那之前，本文件的两档固定率就是权威设计，
//! 不随 `setupProcessing` 变化。

use super::snapshot::{mix_plain_regions, PlaybackSnapshot};
use super::source::SourcePcm;
use crate::ara::AraPlaybackRegion;
use hifishifter_kernel::mixdown::{
    render_mixdown_with_pcm, MixdownOptions, MixdownPcm, QualityPreset,
};
use hifishifter_kernel::state::TimelineState;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// 两条固定率上的空快照。
///
/// 【为什么是一个具名函数】"没有可渲染的片段"有三种来源 —— 没有 region、没有 clip、
/// clip 全部几何非法 —— 它们都应当发布**同一份空快照**，而不是各自报错或各自拼一份。
/// 此前只有第一种走了空快照，后两种（尤其"空列表"）撞上了 `unsupported host position`
/// 这个**整份失败**的守卫。合并成一个具名函数后，这个语义只有一处定义。
fn empty_snapshots() -> Vec<PlaybackSnapshot> {
    [44100, 48000]
        .into_iter()
        .map(|sample_rate| PlaybackSnapshot {
            sample_rate,
            origin_sample: 0,
            left: vec![],
            right: vec![],
            covered: Vec::new(),
            _reservation: None,
        })
        .collect()
}

pub(crate) struct RenderInput {
    pub timeline: Option<TimelineState>,
    pub regions: Vec<AraPlaybackRegion>,
    pub sources: HashMap<String, Arc<SourcePcm>>,
    /// 宿主通报的工程采样率（Hz）；`None` = 宿主没给，直通路径按宿主原始单位读。
    ///
    /// 【为什么渲染输入需要它】`AraPlaybackRegion` 的修改时长单位由宿主决定：素材率与
    /// 工程率不同时宿主按工程速率折算它。没有内核渲染时（`timeline == None`），
    /// `mix_plain_regions` 直接拿它定位源样本，因此必须知道该除以哪个速率。
    pub project_sample_rate: Option<f64>,
    /// 有atlas时为region局部零点参数，不是GUI绝对项目帧。
    pub clip_parameters: BTreeMap<String, hifishifter_kernel::state::TrackParamsState>,
}
/// 输出帧域的"这些帧真的属于我们"覆盖区间（升序、不重叠，已夹进快照窗口）。
///
/// 【为什么这份计算必须存在】捕获期有若干种"我们没拿到样本"的形状 —— 起播窗口的
/// 桥接、掉块、循环回卷的裁缝。它们在素材里是数字零，在 `Segment::coverage` 里是
/// **不存在**。`raw_tape` 一直如实发布这份覆盖，而**处理结果**从来没有：它读的是
/// `Tape::concat_planes()`，覆盖在那里丢了。于是播放侧把"我们没录到"当成"我们录到的
/// 静音"按满幅输出，既不会回退宿主原声、也不会在边界淡化 —— 洞的两侧各是一次硬切。
/// 用户听到的就是"起播 0.4~0.7 s 必然一次咔"（起播窗口就是最大的那个洞）以及播放
/// 途中的随机咔（录制期每一次 resync/掉块留下的小洞）。
///
/// 【为什么不在这里做任何"修补"】洞是"我们没录到"，正确语义是"这一段交给宿主原声"，
/// 不是"编一段出来"。本函数只如实描述事实，让播放侧既有的回退与淡化机制接管。
///
/// 【空 = 整份都算】没有覆盖信息的源（ARA 通路：宿主授权的就是整份素材）与
/// 窗口被整段覆盖的情形都返回空，与旧语义逐字相同。
fn snapshot_covered(
    timeline: &TimelineState,
    sources: &HashMap<String, Arc<SourcePcm>>,
    start_sec: f64,
    end_sec: f64,
    sample_rate: u32,
) -> Vec<std::ops::Range<i64>> {
    let rate = sample_rate as f64;
    let origin = (start_sec * rate).round() as i64;
    let limit = (end_sec * rate).round() as i64;
    let mut covered: Vec<std::ops::Range<i64>> = Vec::new();
    for clip in &timeline.clips {
        let Some(id) = clip.source_path.as_ref() else {
            // 没有源就没有可播放的内容：这一段交给宿主原声（不宣称覆盖）。
            continue;
        };
        let Some(pcm) = sources.get(id) else {
            continue;
        };
        let clip_start = clip.start_sec;
        let clip_end = clip.start_sec + clip.length_sec;
        if !(clip_start.is_finite() && clip_end.is_finite()) || clip_end <= clip_start {
            continue;
        }
        // 【为什么取 active take 的倍率】它是**源窗**的权威：扁平投影上的
        // `clip.playback_rate` 只是 `normalize_takes()` 的产物，在反序列化边界之前可以是 0。
        // 用错倍率会把洞画到错误的帧上 —— 那比"没有覆盖"更糟。
        let take = clip
            .takes
            .get(clip.active_take_index())
            .or_else(|| clip.takes.first());
        let rate_ratio = take
            .map(|take| f64::from(take.playback_rate))
            .unwrap_or(f64::from(clip.playback_rate));
        // 【为什么倒放/循环源退回"整段都算"】倒放把源窗口镜像、循环源对整份媒体取模
        // 回绕，两种映射都比这里能表达的复杂。它们**不是**本缺陷的现场（Capture 的
        // 片段恒为正向、不循环），而为它们猜一个近似映射会在真正的洞上制造假的边界。
        // 退回整段 = 与改动前逐字相同，不引入新的错误方向。
        if clip.reversed || take.is_some_and(|take| take.loop_enabled) {
            push_covered(&mut covered, origin..limit);
            continue;
        }
        if !rate_ratio.is_finite() || rate_ratio <= 0.0 {
            push_covered(&mut covered, origin..limit);
            continue;
        }
        if pcm.covered.is_empty() {
            push_covered(&mut covered, origin..limit);
            continue;
        }
        let source_rate = f64::from(pcm.sample_rate.max(1));
        let source_start = clip.source_start_sec;
        // 源帧 → 播放秒 → 输出帧。倍率把源时长折算成播放时长（`playback_rate` 大
        // 表示同样的源跨度在时间轴上更短）。
        let map = |frame: usize| -> i64 {
            let source_sec = frame as f64 / source_rate;
            let play_sec = clip_start + (source_sec - source_start) / rate_ratio;
            (play_sec * rate).round() as i64
        };
        for range in &pcm.covered {
            let from = map(range.start).max(origin).max((clip_start * rate).round() as i64);
            let to = map(range.end).min(limit).min((clip_end * rate).round() as i64);
            if from < to {
                push_covered(&mut covered, from..to);
            }
        }
    }
    // 【为什么这里要排序再合并】片段之间可以重叠、顺序也不保证按时间排 —— 直接按
    // "与上一条合并"累积会给出一个**非升序**的列表，而 `is_covered` 对它做二分。
    // 非升序的覆盖表会让"有源"与"没源"随机互换，那是比原来的缺陷更坏的错法。
    normalize_covered(&mut covered);
    // 整段窗口都被覆盖 ⇒ 返回空（"全算"，与旧语义同一件事，且不给 `edge_gain`
    // 多一条等价的边界）。
    if covered.len() == 1 && covered[0].start <= origin && covered[0].end >= limit {
        return Vec::new();
    }
    covered
}

/// 把一堆可能乱序、可能重叠的区间整理成**升序、不重叠**的列表。
fn normalize_covered(covered: &mut Vec<std::ops::Range<i64>>) {
    covered.retain(|range| range.start < range.end);
    if covered.len() < 2 {
        return;
    }
    covered.sort_by_key(|range| range.start);
    let mut merged: Vec<std::ops::Range<i64>> = Vec::with_capacity(covered.len());
    for range in covered.drain(..) {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    *covered = merged;
}

/// 记下一个覆盖区间（不合并；由 `normalize_covered` 统一整理）。
fn push_covered(covered: &mut Vec<std::ops::Range<i64>>, range: std::ops::Range<i64>) {
    if range.start < range.end {
        covered.push(range);
    }
}

/// 一段混音请求（播放快照与导出共用）。
///
/// 【为什么打包】这七个值在两条路径上逐一对应，分开传参数会让
/// "导出路径少传一个拉伸算法"这类差异看不出来。打包之后两边共用
/// 同一个口径，漂移会直接变成编译错误。
struct MixRequest<'a> {
    timeline: &'a TimelineState,
    sources: &'a HashMap<String, MixdownPcm>,
    start_sec: f64,
    end_sec: f64,
    sample_rate: u32,
    cancel: &'a Arc<AtomicBool>,
    progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
}

/// 一次导出请求：范围 + 编码目标。
pub(crate) struct ExportRequest<'a> {
    pub start_sec: f64,
    pub end_sec: f64,
    pub output_path: &'a std::path::Path,
    pub spec: &'a hifishifter_kernel::encode::OutputSpec,
    pub sample_rate: u32,
    pub cancel: Arc<AtomicBool>,
    pub progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
}

impl RenderInput {
    /// 本函数只消费冻结值；调用前必须已释放文档transaction。新模型只能取消/拒绝发布。
    pub fn render(&self, cancel: Arc<AtomicBool>) -> Result<Vec<PlaybackSnapshot>, String> {
        self.render_with_progress(cancel, None)
    }
    pub fn render_with_progress(
        &self,
        cancel: Arc<AtomicBool>,
        progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
    ) -> Result<Vec<PlaybackSnapshot>, String> {
        if cancel.load(Ordering::Acquire) {
            return Err("host preparation cancelled".into());
        }
        let Some(timeline) = &self.timeline else {
            return [44100, 48000]
                .into_iter()
                .map(|rate| {
                    if cancel.load(Ordering::Acquire) {
                        return Err("host preparation cancelled".into());
                    }
                    mix_plain_regions(&self.regions, &self.sources, rate, self.project_sample_rate)
                        .map_err(|e| format!("snapshot unavailable: {e:?}"))
                })
                .collect();
        };
        if self.regions.is_empty() {
            return Ok(empty_snapshots());
        }
        // 只放行原kernel负责的真实正向时间拉伸；普通PCM mixer不放行，content-based fades另有契约。
        //
        // 【为什么用宿主工程率而不是写死的 44100】这个参数只把**播放域**秒换算成输出帧，
        // 96 kHz / 192 kHz 工程下写死 44.1k 会让 origin/end 算错，进而是整份快照错位。
        // `setupProcessing` 给出的就是权威工程率；缺席时退回 44100（既有默认档）。
        // 取整是因为这里只用它做秒→帧换算，与输出帧号同为整数域。
        let project_rate = self.project_sample_rate.unwrap_or(44100.0);
        let project_rate = if project_rate.is_finite() && project_rate > 0.0 {
            project_rate.round().clamp(1.0, u32::MAX as f64) as u32
        } else {
            44100
        };
        super::snapshot::validate_regions_with_fades(
            &self.regions,
            &self.sources,
            project_rate,
            true,
            true,
        )
        .map_err(|e| format!("unsupported host region: {e:?}"))?;
        // 【为什么不能整份报错】此前这里对全部 clip 求 min/max，然后
        //     `if start < 0.0 || !start.is_finite() || !end.is_finite() { return Err("unsupported host position") }`
        // 一条坏 clip 就让整份渲染失败 —— 而调用方（自动应用）在 8 次重试后置 `settled`，
        // **永久停止自动渲染**，用户看到的是"声音停在旧版本、界面再也不更新"。
        // 更糟的是空列表也会命中：`fold(INFINITY, min)` 在空列表上得 `INFINITY`，
        // 于是"没有可渲染的片段"被报成了"宿主位置不支持"，日志指向完全错误的方向。
        //
        // 现在逐条降级：非有限几何的 clip 不进渲染（映射层已先按 `SkipReason` 跳过它们，
        // 这里只是兜底）；窗口起点夹到非负 —— 0 之前的部分本就发布不出去，夹掉它是保内容对齐的。
        // 一条片段都不剩时发布**空快照**，不是错误。
        let sanitized = timeline
            .clips
            .iter()
            .any(|clip| !clip.start_sec.is_finite() || !clip.length_sec.is_finite())
            .then(|| {
                let mut view = timeline.clone();
                view.clips
                    .retain(|clip| clip.start_sec.is_finite() && clip.length_sec.is_finite());
                view
            });
        let timeline = sanitized.as_ref().unwrap_or(timeline);
        if let Some(view) = &sanitized {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.render.nonfinite-clip",
                format_args!(
                    "[ara] dropped {} clip(s) with non-finite geometry from this render; \
                     the rest of the timeline is published",
                    self.timeline
                        .as_ref()
                        .map_or(0, |original| original.clips.len() - view.clips.len())
                ),
            );
        }
        let mut start = f64::INFINITY;
        let mut end = f64::NEG_INFINITY;
        for clip in &timeline.clips {
            start = start.min(clip.start_sec);
            end = end.max(clip.start_sec + clip.length_sec);
        }
        if !(start.is_finite() && end.is_finite()) || end <= start.max(0.0) {
            return Ok(empty_snapshots());
        }
        let start = start.max(0.0);
        let mut input = HashMap::new();
        let source_bytes = timeline
            .clips
            .iter()
            .filter_map(|clip| clip.source_path.as_ref())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .try_fold(0_usize, |total, id| {
                let pcm = self.sources.get(id).ok_or_else(|| {
                    format!("{}: {id}", crate::editor::errors::HOST_PCM_UNAVAILABLE)
                })?;
                total
                    .checked_add(
                        pcm.planes
                            .iter()
                            .map(|plane| plane.len() * 4)
                            .sum::<usize>(),
                    )
                    .ok_or_else(|| "host interleaved PCM size overflow".to_owned())
            })?;
        // 两率最终PCM、源转交错副本、当前混音/region临时输出一起预检，失败不先做重推理。
        // 此策略保留512MiB全局硬上限，不把旧30秒/64MiB探针常量直接放宽为无界分配。
        let frame_counts = [44100_u32, 48000]
            .map(|rate| {
                let frames = ((end - start) * rate as f64).round();
                if !frames.is_finite() || frames < 0. || frames >= usize::MAX as f64 / 8. {
                    Err("snapshot size overflow".to_owned())
                } else {
                    Ok(frames as usize)
                }
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = frame_counts
            .iter()
            .try_fold(0_usize, |total, frames| total.checked_add(frames * 8))
            .ok_or("snapshot size overflow")?;
        let working = frame_counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_mul(8)
            .and_then(|bytes| {
                bytes.checked_mul(
                    if self.clip_parameters.is_empty() || timeline.clips.len() == 1 {
                        1
                    } else {
                        2
                    },
                )
            })
            .ok_or("render working size overflow")?;
        let required = source_bytes
            .checked_add(outputs)
            .and_then(|bytes| bytes.checked_add(working))
            .ok_or("render memory size overflow")?;
        let budget = super::budget::global_budget();
        let mut batch = budget.reserve(required).ok_or_else(|| {
            format!(
                "render PCM budget exceeded: required={required} used={} limit={}",
                budget.used(),
                budget.limit()
            )
        })?;
        let _source_working = batch
            .split_off(source_bytes)
            .ok_or("invalid render budget partition")?;
        let _working = batch
            .split_off(working)
            .ok_or("invalid render budget partition")?;
        for id in timeline
            .clips
            .iter()
            .filter_map(|clip| clip.source_path.as_ref())
            .collect::<BTreeSet<_>>()
        {
            let pcm = self
                .sources
                .get(id)
                .ok_or_else(|| format!("{}: {id}", crate::editor::errors::HOST_PCM_UNAVAILABLE))?;
            let channels = pcm.planes.len();
            let mut samples = Vec::with_capacity(pcm.planes[0].len() * channels);
            for frame in 0..pcm.planes[0].len() {
                for plane in &pcm.planes {
                    samples.push(plane[frame]);
                }
            }
            input.insert(
                id.clone(),
                MixdownPcm {
                    sample_rate: pcm.sample_rate,
                    channels: channels as u16,
                    samples: Arc::new(samples),
                },
            );
        }
        let mut snapshots = Vec::new();
        let model_rate_output = timeline
            .clips
            .iter()
            .filter_map(|clip| timeline.resolve_root_track_id(&clip.track_id))
            .any(|root| {
                timeline.tracks.iter().any(|track| {
                    track.id == root
                        && track.pitch_analysis_algo
                            == hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx
                })
            });
        for (sample_rate, frames) in [44100, 48000].into_iter().zip(frame_counts) {
            if cancel.load(Ordering::Acquire) {
                return Err("host preparation cancelled".into());
            }
            let reservation = batch
                .split_off(frames * 8)
                .ok_or("invalid render budget partition")?;
            // HiFiGAN工作区只在模型原生44.1k合成一次，48k是便宜输出层，不重新跑HNSEP/神经网络。
            // 纯非HiFiGAN路径保持旧采样率契约；只消费已经就绪的不可变PCM，仍在worker线程。
            if model_rate_output && sample_rate == 48000 {
                let native: &PlaybackSnapshot = &snapshots[0];
                let mut left =
                    hifishifter_kernel::mel_utils::linear_resample_mono(&native.left, 44100, 48000);
                let mut right = hifishifter_kernel::mel_utils::linear_resample_mono(
                    &native.right,
                    44100,
                    48000,
                );
                left.resize(frames, 0.);
                right.resize(frames, 0.);
                snapshots.push(PlaybackSnapshot {
                    sample_rate,
                    origin_sample: (start * sample_rate as f64).round() as i64,
                    left,
                    right,
                    // 48 k 这一份是 44.1 k 的重采样，但**覆盖**是按时间算的：直接按
                    // 本档率重算一次，比按比例缩放区间少一整类舍入错误。
                    covered: snapshot_covered(timeline, &self.sources, start, end, sample_rate),
                    _reservation: Some(reservation),
                });
                continue;
            }
            let samples = self.mix_interleaved(MixRequest {
                timeline,
                sources: &input,
                start_sec: start,
                end_sec: end,
                sample_rate,
                cancel: &cancel,
                progress: progress.clone(),
            })?;
            debug_assert_eq!(samples.len(), frames * 2);
            snapshots.push(PlaybackSnapshot {
                sample_rate,
                origin_sample: (start * sample_rate as f64).round() as i64,
                left: samples.iter().step_by(2).copied().collect(),
                right: samples.iter().skip(1).step_by(2).copied().collect(),
                covered: snapshot_covered(timeline, &self.sources, start, end, sample_rate),
                _reservation: Some(reservation),
            });
        }
        Ok(snapshots)
    }

    /// 授权源 PCM → 混音器要的交错副本。
    fn interleaved_sources(
        &self,
        timeline: &TimelineState,
    ) -> Result<HashMap<String, MixdownPcm>, String> {
        let mut sources = HashMap::new();
        for id in timeline
            .clips
            .iter()
            .filter_map(|clip| clip.source_path.as_ref())
            .collect::<BTreeSet<_>>()
        {
            let pcm = self
                .sources
                .get(id)
                .ok_or_else(|| format!("{}: {id}", crate::editor::errors::HOST_PCM_UNAVAILABLE))?;
            let channels = pcm.planes.len();
            let mut samples = Vec::with_capacity(pcm.planes[0].len() * channels);
            for frame in 0..pcm.planes[0].len() {
                for plane in &pcm.planes {
                    samples.push(plane[frame]);
                }
            }
            sources.insert(
                id.clone(),
                MixdownPcm {
                    sample_rate: pcm.sample_rate,
                    channels: channels as u16,
                    samples: Arc::new(samples),
                },
            );
        }
        Ok(sources)
    }

    /// 把一段工程范围混成**交错立体声**样本。
    ///
    /// 【为什么播放与导出必须共用这一条】逐 clip 用各自的源域参数渲染、再按宿主位置
    /// 累加，是"同一条轨道上有多个片段（含重叠）"唯一正确的算法。导出若另写一份，两份
    /// 迟早漂移 —— 症状是"导出的音频和听到的不一样"，那比不提供导出更糟。
    ///
    /// 【为什么读内核的解析器而不是写死 SoundTouch】拉伸算法是**用户设置**
    /// （插件通过 `update_project_stretch_overrides` 收下它，与独立 App 共用同一份
    /// 运行时设置）。此前这里硬编码 `SoundTouchDll`，于是用户在插件里改拉伸算法菜单
    /// **毫无效果** —— 一个看起来生效、实际被忽略的设置，比一个明确禁用掉的菜单更糟。
    fn mix_interleaved(&self, request: MixRequest<'_>) -> Result<Vec<f32>, String> {
        let MixRequest {
            timeline,
            sources,
            start_sec: start,
            end_sec: end,
            sample_rate,
            cancel,
            progress,
        } = request;
        let frames = ((end - start) * sample_rate as f64).round() as usize;
        let run = |view: &TimelineState, lo: f64, hi: f64| {
            let options = MixdownOptions {
                sample_rate,
                start_sec: lo,
                end_sec: Some(hi),
                stretch: hifishifter_kernel::time_stretch::resolved_external_stretch_algorithm(),
                apply_pitch_edit: true,
                output: hifishifter_kernel::encode::OutputSpec::wav_32f(),
                quality_preset: QualityPreset::Export,
                cancel_flag: Some(cancel.clone()),
                progress: progress.clone(),
                cache_stats: None,
            };
            render_mixdown_with_pcm(view, options, sources)
        };
        // 项目数组只负责GUI；每个区域用自己的源参数跑原kernel，再按宿主位置累加。
        // 所有track仍保留原全局solo/父链，不能因为拆region丢掉另一轨solo。
        let (channels, samples) = if self.clip_parameters.is_empty() {
            let (_, channels, _, samples) = run(timeline, start, end)?;
            (channels, samples)
        } else if timeline.clips.len() == 1 {
            // 单region直接保留原kernel结果，不另建全长mixed再从part复制一遍。
            // 参数仍从局部零点投影，所有track保留全局solo/父链；只省去加到零缓冲的复制。
            let clip = &timeline.clips[0];
            let mut view = timeline.clone();
            let (lo, hi) = if let Some(params) = self.clip_parameters.get(&clip.id) {
                let root = view
                    .resolve_root_track_id(&clip.track_id)
                    .ok_or("unknown source parameter root")?;
                view.params_by_root_track.insert(root, params.clone());
                view.clips[0].start_sec = 0.;
                view.project_sec = clip.length_sec;
                (0., clip.length_sec)
            } else {
                (start, end)
            };
            let (_, channels, _, mut samples) = run(&view, lo, hi)?;
            // 保留旧累加语义的signed-zero位形，不改变已归档PCM的零样本身份。
            for sample in &mut samples {
                *sample += 0_f32;
            }
            (channels, samples)
        } else {
            let mut mixed = vec![0_f32; frames * 2];
            for clip in &timeline.clips {
                if cancel.load(Ordering::Acquire) {
                    return Err("host preparation cancelled".into());
                }
                let mut view = timeline.clone();
                view.clips = vec![clip.clone()];
                let local = self.clip_parameters.get(&clip.id);
                let (lo, hi) = if let Some(params) = local {
                    let root = view
                        .resolve_root_track_id(&clip.track_id)
                        .ok_or("unknown source parameter root")?;
                    view.params_by_root_track.insert(root, params.clone());
                    view.clips[0].start_sec = 0.;
                    view.project_sec = clip.length_sec;
                    (0., clip.length_sec)
                } else {
                    (start, end)
                };
                let part_frames = ((hi - lo) * sample_rate as f64).round() as usize;
                let (_, channels, _, part) = run(&view, lo, hi)?;
                if channels != 2 || part.len() != part_frames * 2 {
                    return Err("invalid region parameter output".into());
                }
                let offset = if local.is_some() {
                    ((clip.start_sec - start) * sample_rate as f64).round() as usize * 2
                } else {
                    0
                };
                for (sum, value) in mixed.iter_mut().skip(offset).zip(part) {
                    *sum += value;
                }
            }
            (2, mixed)
        };
        if channels != 2 || samples.len() != frames * 2 || samples.iter().any(|v| !v.is_finite()) {
            return Err("invalid kernel output".into());
        }
        Ok(samples)
    }

    /// 把一段工程范围渲染并**编码到文件**。
    ///
    /// 【为什么插件里可行】ARA 对音频读取**没有角色限制**（`ARAInterface.h:1784-1789`），
    /// 授权 PCM 就在 `self.sources` 里；编码器在内核里（`encode::create_encoder`）。
    /// 此前 ARA 版**完全没有**这条路径 —— 用户在 Cubase 里做完处理却拿不出文件，
    /// 这是"ARA 版功能不全"里最大的一项。
    pub fn export_range_to_file(
        &self,
        request: ExportRequest<'_>,
    ) -> Result<hifishifter_kernel::encode::EncodeSummary, String> {
        let ExportRequest {
            start_sec,
            end_sec,
            output_path,
            spec,
            sample_rate,
            cancel,
            progress,
        } = request;
        let Some(timeline) = &self.timeline else {
            return Err("export requires a mapped host timeline".into());
        };
        if !(start_sec.is_finite() && end_sec.is_finite()) || end_sec <= start_sec {
            return Err("invalid export range".into());
        }
        if !matches!(sample_rate, 44_100 | 48_000) {
            return Err("unsupported export sample rate".into());
        }
        let sources = self.interleaved_sources(timeline)?;
        let samples = self.mix_interleaved(MixRequest {
            timeline,
            sources: &sources,
            start_sec,
            end_sec,
            sample_rate,
            cancel: &cancel,
            progress,
        })?;
        let mut encoder = hifishifter_kernel::encode::create_encoder(
            output_path,
            spec,
            2,
            sample_rate,
            Some(cancel),
        )
        .map_err(|error| error.to_string())?;
        // 分块推送：编码器内部按块检查取消标志，也避免一次拷一份全长缓冲。
        for chunk in samples.chunks(8192) {
            encoder.push(chunk).map_err(|error| error.to_string())?;
        }
        encoder.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::MutexExt;

    /// 一份只有覆盖信息不同的源（其余字段无关）。
    fn source_with_coverage(covered: Vec<std::ops::Range<usize>>) -> Arc<SourcePcm> {
        Arc::new(SourcePcm {
            sample_rate: 48_000,
            planes: vec![vec![0.5; 96_000], vec![0.5; 96_000]],
            version: 1,
            covered,
            _reservation: None,
        })
    }

    /// 一条正向、不循环、单源的 clip。
    fn plain_clip() -> hifishifter_kernel::state::TimelineState {
        let mut timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks": [{"id": "t", "name": "t", "order": 0}],
            "clips": [{
                "id": "c", "track_id": "t", "name": "c",
                "start_sec": 0.0, "length_sec": 2.0,
                "takes": [{
                    "id": "k", "source_path": "src",
                    "source_start_sec": 0.0, "source_end_sec": 2.0, "playback_rate": 1.0
                }]
            }],
            "bpm": 120.0, "project_sec": 2.0
        }))
        .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        timeline
    }

    /// **处理结果的快照必须如实标注"我们没录到"的洞。**
    ///
    /// 【为什么这是"咔"的根因级回归】`raw_tape` 一直带覆盖，而**处理结果**从来没有：
    /// 它读的是 `Tape::concat_planes()`，洞在那里就是数字零。播放侧于是把"我们没录到"
    /// 当成"我们录到的静音"按满幅输出，既不回退宿主原声也不淡化 —— 洞的两侧各是一次硬切。
    #[test]
    fn the_published_snapshot_marks_the_hole() {
        let timeline = plain_clip();
        let mut sources: HashMap<String, Arc<SourcePcm>> = HashMap::new();
        // 源里 48000..50000 这 2000 帧是桥接出来的洞。
        sources.insert(
            "src".into(),
            source_with_coverage(vec![0..48_000, 50_000..96_000]),
        );
        let covered = snapshot_covered(&timeline, &sources, 0.0, 2.0, 48_000);
        assert_eq!(
            covered,
            vec![0..48_000, 50_000..96_000],
            "the hole must survive into the published coverage"
        );
    }

    /// 没有覆盖信息的源（ARA 通路）与"整段都覆盖"的情形都退化成空表。
    ///
    /// 【为什么要钉】空表 = "整份都是我们的"，与改动前逐字相同。若这里返回一张满表，
    /// `edge_gain` 会凭空多出一条边界，ARA 通路的输出就会出现一次本不该有的淡化。
    #[test]
    fn a_source_without_coverage_stays_fully_ours() {
        let timeline = plain_clip();
        let mut sources: HashMap<String, Arc<SourcePcm>> = HashMap::new();
        sources.insert("src".into(), source_with_coverage(Vec::new()));
        assert!(snapshot_covered(&timeline, &sources, 0.0, 2.0, 48_000).is_empty());
    }

    /// 覆盖区间必须按时间排序并合并：片段之间可以重叠、顺序也不保证。
    ///
    /// 【为什么这条最要紧】`is_covered` 对覆盖表做**二分**。一张非升序的表会让
    /// "有源"与"没源"随机互换 —— 那是比原来的缺陷更坏的错法。
    #[test]
    fn overlapping_clips_yield_one_sorted_coverage_table() {
        let mut timeline = plain_clip();
        // 第二条 clip 与前一条重叠，且排在后面（时间上更早）。
        let mut second = timeline.clips[0].clone();
        second.id = "c2".into();
        second.start_sec = 1.0;
        second.length_sec = 0.5;
        second.takes[0].source_start_sec = 0.0;
        second.takes[0].source_end_sec = 0.5;
        second.normalize_takes();
        timeline.clips.push(second);
        let mut sources: HashMap<String, Arc<SourcePcm>> = HashMap::new();
        sources.insert(
            "src".into(),
            source_with_coverage(vec![0..24_000, 26_000..48_000]),
        );
        let covered = snapshot_covered(&timeline, &sources, 0.0, 2.0, 48_000);
        assert!(
            covered.windows(2).all(|pair| pair[0].end <= pair[1].start),
            "the table must stay sorted and disjoint: {covered:?}"
        );
    }
    /// 用已核对的windows crate typed API读取真实进程高水位，不用PCM额度冒充RSS。
    #[cfg(windows)]
    fn process_memory_profile() -> (usize, usize, usize) {
        use windows::Win32::System::{
            ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::GetCurrentProcess,
        };
        let bytes = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: bytes,
            ..Default::default()
        };
        // SAFETY: 当前进程伪句柄有效；typed结构由本调用拥有，大小与官方绑定完全匹配。
        assert!(
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, bytes) }.as_bool()
        );
        (
            counters.WorkingSetSize,
            counters.PeakWorkingSetSize,
            counters.PeakPagefileUsage,
        )
    }
    /// 正常三分钟完整源的HiFiGAN分块及整段HNSEP诊断；只显式运行，不冒充REAPER GUI验收。
    #[test]
    #[ignore = "真实CPU三分钟HiFiGAN/HNSEP及进程内存诊断：显式运行"]
    fn real_three_minute_hifigan_hnsep_prepares_complete_pcm_and_reuses_warm_cache() {
        hifishifter_kernel::vocoder_ort_session::set_runtime_ep_override(Some("cpu".into()));
        crate::editor::resources::initialize_models();
        hifishifter_kernel::hnsep_onnx::clear_separation_cache();
        hifishifter_kernel::renderer::hifigan::clear_chunk_cache();
        hifishifter_kernel::synth_clip_cache::global_synth_clip_cache()
            .lock_or_recover()
            .clear();
        let seconds = 180.;
        let frames = 44100 * 180;
        let budget = super::super::budget::global_budget();
        let baseline = budget.used();
        let reservation = budget.reserve(frames * 4).unwrap();
        let samples = (0..frames)
            .map(|index| {
                let phase = 2. * std::f64::consts::PI * 220. * index as f64 / 44100.;
                (1..=8)
                    .map(|harmonic| 0.12 / harmonic as f64 * (phase * harmonic as f64).sin())
                    .sum::<f64>() as f32
            })
            .collect();
        let source = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![samples],
            version: 1,
            _reservation: Some(reservation),
        });
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"Long HifiGAN","order":0,"pitch_analysis_algo":"nsf_hifigan_onnx","compose_enabled":true}],
            "bpm":120,"project_sec":seconds,"clips":[{"id":"clip","name":"Long HifiGAN","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();
        timeline.clips[0].normalize_takes();
        let params = hifishifter_kernel::state::TrackParamsState {
            frame_period_ms: 5.,
            pitch_orig: vec![57.; 36001],
            pitch_edit: vec![60.; 36001],
            pitch_edit_user_modified: true,
            extra_params: HashMap::from([("breath_enabled".into(), 1.)]),
            ..Default::default()
        };
        timeline
            .params_by_root_track
            .insert("track".into(), params.clone());
        let input = RenderInput {
            timeline: Some(timeline),
            regions: vec![AraPlaybackRegion {
                audio_source_persistent_id: "source".into(),
                duration_in_modification_time: seconds,
                duration_in_playback_time: seconds,
                ..Default::default()
            }],
            sources: HashMap::from([("source".into(), source.clone())]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::from([("clip".into(), params)]),
        };
        let neural_before = hifishifter_kernel::nsf_hifigan_onnx::inference_runs();
        let separation_before = hifishifter_kernel::hnsep_onnx::separation_cache_stats().1;
        let began = std::time::Instant::now();
        let mut cold = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        let cold_ms = began.elapsed().as_millis();
        let cold_neural = hifishifter_kernel::nsf_hifigan_onnx::inference_runs();
        let cold_separation = hifishifter_kernel::hnsep_onnx::separation_cache_stats().1;
        assert!(
            cold_neural - neural_before > 1,
            "三分钟必须实际走多个神经批次，不能只返回缓存元数据"
        );
        assert_eq!(
            cold_separation - separation_before,
            1,
            "HNSEP仍整段一次，不分块"
        );
        for (snapshot, rate) in cold.iter().zip([44100, 48000]) {
            assert_eq!(snapshot.left.len(), rate * 180);
            assert_eq!(snapshot.right.len(), rate * 180);
            assert!(snapshot.left.iter().all(|sample| sample.is_finite()));
            for start in [0, rate * 90, rate * 179] {
                let rms = (snapshot.left[start..start + rate]
                    .iter()
                    .map(|sample| (*sample as f64).powi(2))
                    .sum::<f64>()
                    / rate as f64)
                    .sqrt();
                assert!(
                    rms > 0.01,
                    "源中部/末尾不能是静音占位，start={start} rms={rms}"
                );
            }
        }
        assert_eq!(
            cold[1].left,
            hifishifter_kernel::mel_utils::linear_resample_mono(&cold[0].left, 44100, 48000)
        );
        super::super::snapshot::compact_prepared(&mut cold).unwrap();
        let began = std::time::Instant::now();
        let mut warm = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        let warm_ms = began.elapsed().as_millis();
        assert_eq!(
            hifishifter_kernel::nsf_hifigan_onnx::inference_runs(),
            cold_neural
        );
        assert_eq!(
            hifishifter_kernel::hnsep_onnx::separation_cache_stats().1,
            cold_separation
        );
        super::super::snapshot::compact_prepared(&mut warm).unwrap();
        for (cold, warm) in cold.iter().zip(&warm) {
            assert_eq!(cold.left, warm.left);
            assert_eq!(cold.right, warm.right);
        }
        #[cfg(windows)]
        {
            let (working, peak, commit) = process_memory_profile();
            eprintln!("LONG_HIFIGAN seconds=180 cold_ms={cold_ms} warm_ms={warm_ms} neural_runs={} HNSEP_runs=1 warm_extra_runs=0 accounted_peak={} working_set={working} peak_working_set={peak} peak_commit={commit}",cold_neural-neural_before,budget.peak());
        }
        drop(cold);
        drop(warm);
        drop(input);
        drop(source);
        assert_eq!(budget.used(), baseline);
    }
    /// 另一轨采用HiFiGAN不能改变当前renderer的原生率路径；其solo仍应按全工程状态参与。
    #[test]
    fn unrelated_hifigan_track_does_not_change_this_renderers_rate_path() {
        let seconds = 0.05;
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"A","order":0,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":seconds,
            "clips":[{"id":"clip","name":"A","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();
        timeline.clips[0].normalize_takes();
        let source = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![(0..2205)
                .map(|index| ((index * 113) % 257) as f32 / 512.)
                .collect()],
            version: 0,
            _reservation: None,
        });
        let mut input = RenderInput {
            timeline: Some(timeline),
            regions: vec![AraPlaybackRegion {
                audio_source_persistent_id: "source".into(),
                duration_in_modification_time: seconds,
                duration_in_playback_time: seconds,
                ..Default::default()
            }],
            sources: HashMap::from([("source".into(), source)]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::new(),
        };
        let before = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        let mut other = input.timeline.as_ref().unwrap().tracks[0].clone();
        other.id = "unrelated".into();
        other.order = 1;
        other.pitch_analysis_algo = hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx;
        input.timeline.as_mut().unwrap().tracks.push(other);
        let after = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        for (before, after) in before.iter().zip(&after) {
            assert_eq!(before.left, after.left);
            assert_eq!(before.right, after.right);
        }
        input.timeline.as_mut().unwrap().tracks[1].solo = true;
        let solo = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert!(solo
            .iter()
            .all(|snapshot| snapshot.left.iter().all(|value| *value == 0.)));
    }
    /// 三分钟普通人声经过授权kernel/两率快照后全尾就绪；内存按固定全局预算预检。
    #[test]
    fn three_minute_host_clip_prepares_both_rates_and_the_last_block_within_budget() {
        let frames = 44100 * 180;
        let seconds = 180.;
        let budget = super::super::budget::global_budget();
        let baseline = budget.used();
        let source = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.25; frames]],
            version: 1,
            _reservation: Some(budget.reserve(frames * 4).unwrap()),
        });
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"Long voice","order":0,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":seconds,
            "clips":[{"id":"clip","name":"Long voice","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();
        timeline.clips[0].normalize_takes();
        let input = RenderInput {
            timeline: Some(timeline),
            regions: vec![AraPlaybackRegion {
                audio_source_persistent_id: "source".into(),
                duration_in_modification_time: seconds,
                duration_in_playback_time: seconds,
                ..Default::default()
            }],
            sources: HashMap::from([("source".into(), source.clone())]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::new(),
        };
        let snapshots = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        for (snapshot, rate) in snapshots.iter().zip([44100, 48000]) {
            assert_eq!(snapshot.left.len(), rate * 180);
            assert_eq!(snapshot.right.len(), rate * 180);
            assert!(snapshot
                .left
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1e-6));
        }
        let accounted = budget.used() - baseline;
        assert_eq!(accounted, frames * 4 + (44100 + 48000) * 180 * 8);
        eprintln!(
            "long-host seconds=180 source_bytes={} ready_bytes={} accounted_peak={} hard_limit={}",
            frames * 4,
            accounted - frames * 4,
            budget.peak(),
            budget.limit()
        );
        let mut left = [0_f32; 513];
        let mut right = [0_f32; 513];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        let publisher = super::super::snapshot::SnapshotPublisher::default();
        let mut snapshots = snapshots;
        publisher.publish(snapshots.pop().unwrap()).unwrap();
        assert!(unsafe { publisher.copy_block((48000 * 180 - 511) as i64, 48000, &mut bus, 513) });
        assert_eq!(&left[..511], &[0.25; 511]);
        assert_eq!(&right[..511], &[0.25; 511]);
        assert_eq!(&left[511..], &[0.; 2]);
        assert!(unsafe { publisher.copy_block(0, 48000, &mut bus, 513) });
        assert_eq!(left, [0.25; 513]);
        drop(publisher);
        drop(snapshots);
        drop(input);
        drop(source);
        assert_eq!(budget.used(), baseline);
    }
    /// 大跨度在任何合成前明确预算失败，不能为通过长源门扩大512MiB或静默补零。
    #[test]
    fn sparse_huge_span_is_rejected_before_any_large_render_allocation() {
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"Sparse voice","order":0,"pitch_analysis_algo":"none"}],"bpm":120,"project_sec":3601,
            "clips":[{"id":"a","name":"A","track_id":"track","start_sec":0,"length_sec":1,"takes":[{"id":"ta","source_path":"source","source_end_sec":1}]},
                {"id":"b","name":"B","track_id":"track","start_sec":3600,"length_sec":1,"takes":[{"id":"tb","source_path":"source","source_end_sec":1}]}]
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        let source = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.25; 44100]],
            version: 1,
            _reservation: None,
        });
        let region = |start| AraPlaybackRegion {
            audio_source_persistent_id: "source".into(),
            start_in_playback_time: start,
            duration_in_modification_time: 1.,
            duration_in_playback_time: 1.,
            ..Default::default()
        };
        let input = RenderInput {
            timeline: Some(timeline),
            regions: vec![region(0.), region(3600.)],
            sources: HashMap::from([("source".into(), source)]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::new(),
        };
        let budget = super::super::budget::global_budget();
        let baseline = budget.used();
        let error = input.render(Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(
            error.contains("BudgetExceeded"),
            "unexpected error: {error}"
        );
        assert_eq!(budget.used(), baseline);
    }
    /// 双输出率从同一HiFiGAN原生结果派生，整段HNSEP也只运行一次，不以WORLD外推。
    #[test]
    #[ignore = "真实CPU插件渲染输入诊断：显式运行"]
    fn real_model_hifigan_snapshots_share_one_native_rate_synthesis() {
        hifishifter_kernel::vocoder_ort_session::set_runtime_ep_override(Some("cpu".into()));
        hifishifter_kernel::hnsep_onnx::clear_separation_cache();
        let seconds = 0.5;
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"A","order":0,"pitch_analysis_algo":"nsf_hifigan_onnx","compose_enabled":true}],
            "bpm":120,"project_sec":seconds,"clips":[{"id":"clip","name":"A","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"take","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();
        timeline.clips[0].normalize_takes();
        timeline.params_by_root_track.insert(
            "track".into(),
            hifishifter_kernel::state::TrackParamsState {
                frame_period_ms: 5.,
                pitch_orig: vec![57.; 101],
                pitch_edit: vec![60.; 101],
                pitch_edit_user_modified: true,
                extra_params: HashMap::from([("breath_enabled".into(), 1.)]),
                ..Default::default()
            },
        );
        let pcm = (0..22050)
            .map(|i| {
                let phase = 2. * std::f64::consts::PI * 220. * i as f64 / 44100.;
                (1..=12)
                    .map(|k| 0.15 / k as f64 * (phase * k as f64).sin())
                    .sum::<f64>() as f32
            })
            .collect();
        let region = AraPlaybackRegion {
            audio_source_persistent_id: "source".into(),
            duration_in_modification_time: seconds,
            duration_in_playback_time: seconds,
            ..Default::default()
        };
        let parameters = timeline.params_by_root_track["track"].clone();
        hifishifter_kernel::renderer::hifigan::clear_chunk_cache();
        hifishifter_kernel::synth_clip_cache::global_synth_clip_cache()
            .lock_or_recover()
            .clear();
        let mut input = RenderInput {
            timeline: Some(timeline),
            regions: vec![region],
            sources: HashMap::from([(
                "source".into(),
                Arc::new(SourcePcm {
                    covered: Vec::new(),
                    sample_rate: 44100,
                    planes: vec![pcm],
                    version: 0,
                    _reservation: None,
                }),
            )]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::from([("clip".into(), parameters)]),
        };
        let before = hifishifter_kernel::hnsep_onnx::separation_cache_stats();
        let began = std::time::Instant::now();
        let neural_before = hifishifter_kernel::nsf_hifigan_onnx::inference_runs();
        let output = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(output[0].left.len(), 22050);
        assert_eq!(output[1].left.len(), 24000);
        assert_eq!(
            output[1].left,
            hifishifter_kernel::mel_utils::linear_resample_mono(&output[0].left, 44100, 48000)
        );
        assert_eq!(
            output[1].right,
            hifishifter_kernel::mel_utils::linear_resample_mono(&output[0].right, 44100, 48000)
        );
        assert_eq!(
            hifishifter_kernel::hnsep_onnx::separation_cache_stats().1 - before.1,
            1
        );
        assert!(output[0].left.iter().any(|v| v.abs() > 0.01));
        let cold_neural = hifishifter_kernel::nsf_hifigan_onnx::inference_runs();
        assert!(cold_neural > neural_before);
        let cold_ms = began.elapsed().as_millis();
        // 不同owner/region名和非网格项目摆放不得改变源曲线或重推理。
        let moved_start = 512.013;
        {
            let timeline = input.timeline.as_mut().unwrap();
            timeline.clips[0].id = "other-owner-clip".into();
            timeline.clips[0].start_sec = moved_start;
            timeline.project_sec = moved_start + seconds;
        }
        let params = input.clip_parameters.remove("clip").unwrap();
        input
            .clip_parameters
            .insert("other-owner-clip".into(), params);
        input.regions[0].start_in_playback_time = moved_start;
        let moved = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(
            moved[0].origin_sample,
            (moved_start * 44100.).round() as i64
        );
        assert_eq!(moved[0].left, output[0].left);
        assert_eq!(moved[1].left, output[1].left);
        assert_eq!(
            hifishifter_kernel::nsf_hifigan_onnx::inference_runs(),
            cold_neural
        );
        input
            .clip_parameters
            .get_mut("other-owner-clip")
            .unwrap()
            .pitch_edit
            .fill(64.);
        let changed = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert!(
            hifishifter_kernel::nsf_hifigan_onnx::inference_runs() > cold_neural,
            "真正目标F0改变必须使HiFiGAN失效"
        );
        assert_ne!(changed[0].left, moved[0].left);
        assert_eq!(
            hifishifter_kernel::hnsep_onnx::separation_cache_stats().1 - before.1,
            1
        );
        println!("PLUGIN_HIFIGAN_CONTENT cold_ms={cold_ms} neural_runs={} moved_extra_runs=0 moved_pcm_exact=true derived_48000_exact=true pitch_change_runs={} HNSEP_runs=1",
            cold_neural-neural_before,hifishifter_kernel::nsf_hifigan_onnx::inference_runs()-cold_neural);
    }
    /// 相同root的两个重叠区域保留独立源参数，不能用GUI单一数组覆盖两者。
    #[test]
    fn source_parameter_atlas_overlapping_regions_keep_independent_audio_gains() {
        let seconds = 4.0 / 44100.;
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"A","order":0}],"bpm":120,"project_sec":seconds,
            "clips":[{"id":"a","name":"A","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"ta","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]},
                {"id":"b","name":"B","track_id":"track","start_sec":0,"length_sec":seconds,
                "takes":[{"id":"tb","source_path":"source","source_start_sec":0,"source_end_sec":seconds}]}]
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        let region = AraPlaybackRegion {
            audio_source_persistent_id: "source".into(),
            duration_in_modification_time: seconds,
            duration_in_playback_time: seconds,
            ..Default::default()
        };
        let params = |gain| hifishifter_kernel::state::TrackParamsState {
            frame_period_ms: 5.,
            extra_curves: HashMap::from([("volume".into(), vec![gain])]),
            ..Default::default()
        };
        let input = RenderInput {
            timeline: Some(timeline),
            regions: vec![region.clone(), region],
            sources: HashMap::from([(
                "source".into(),
                Arc::new(SourcePcm {
                    covered: Vec::new(),
                    sample_rate: 44100,
                    planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
                    version: 0,
                    _reservation: None,
                }),
            )]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::from([
                ("a".into(), params(0.5)),
                ("b".into(), params(0.25)),
            ]),
        };
        let output = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        for (actual, want) in output[0].left.iter().zip([0.075_f32, 0.15, 0.225, 0.3]) {
            assert!((*actual - want).abs() < 1e-6, "{actual} != {want}");
        }
    }
    /// 原kernel保调拉伸覆盖整段目标时间；不能只显示新长度却尾部静音或降八度重采样。
    #[test]
    fn linear_stretch_produces_full_length_audio_without_transposing_the_voice() {
        let pcm = (0..22050)
            .map(|n| (2.0 * std::f64::consts::PI * 220.0 * n as f64 / 44100.0).sin() as f32 * 0.2)
            .collect();
        let source = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![pcm],
            version: 0,
            _reservation: None,
        });
        let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"a","name":"A","order":0}],"bpm":120,"project_sec":1,
            "clips":[{"id":"ca","name":"A","track_id":"a","start_sec":0,"length_sec":1,
                "takes":[{"id":"ta","name":"A","source_path":"source","source_start_sec":0,"source_end_sec":0.5,"playback_rate":0.5}]}]
        })).unwrap();
        timeline.clips[0].normalize_takes();
        let region = AraPlaybackRegion {
            audio_source_persistent_id: "source".into(),
            duration_in_modification_time: 0.5,
            duration_in_playback_time: 1.0,
            is_timestretch_enabled: true,
            ..Default::default()
        };
        let input = RenderInput {
            timeline: Some(timeline),
            regions: vec![region],
            sources: HashMap::from([("source".into(), source)]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::new(),
        };
        let output = input.render(Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(output[0].left.len(), 44100);
        assert_eq!(output[1].left.len(), 48000);
        for start in [0.2, 0.7] {
            let begin = (start * 44100.0) as usize;
            let count = 3528;
            let rms = (output[0].left[begin..begin + count]
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                / count as f64)
                .sqrt();
            assert!(rms > 0.05, "拉伸后的后半段不能只补静音");
            let mut scores = vec![0.0; 442];
            for lag in 147..441 {
                let mut xy = 0.0;
                let mut xx = 0.0;
                let mut yy = 0.0;
                for n in 0..count {
                    let a = output[0].left[begin + n] as f64;
                    let b = output[0].left[begin + n + lag] as f64;
                    xy += a * b;
                    xx += a * a;
                    yy += b * b;
                }
                scores[lag] = xy / (xx * yy).sqrt();
            }
            let lag = (148..440)
                .find(|&lag| {
                    scores[lag] > 0.9
                        && scores[lag] > scores[lag - 1]
                        && scores[lag] >= scores[lag + 1]
                })
                .unwrap();
            let hz = 44100.0 / lag as f64;
            assert!((hz - 220.0).abs() < 2.0, "必须保调而不是普通重采样: {hz}");
        }
    }
    /// 现场ka/n几何的源尾比真实PCM多一帧：REAPER秒域取整不得把整段元音拒绝为无声。
    #[test]
    fn source_eof_one_frame_rounding_keeps_ka_and_n_regions_audible() {
        for (name, source_frames, start, source_start, duration, rate) in [
            (
                "ka",
                8073,
                3.27272727272727,
                0.03853961089064,
                0.20538156546032998,
                0.703784167766571,
            ),
            (
                "n",
                8084,
                4.63636363635305,
                0.,
                0.22229856611387003,
                0.8247166872024536,
            ),
        ] {
            let pcm = (0..source_frames)
                .map(|frame| (std::f32::consts::TAU * 220. * frame as f32 / 44100.).sin() * 0.2)
                .collect();
            let source = Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![pcm],
                version: 0,
                _reservation: None,
            });
            let source_duration = duration * rate;
            let mut timeline:TimelineState=serde_json::from_value(serde_json::json!({
                "tracks":[{"id":"a","name":"A","order":0,"compose_enabled":false}],"bpm":120,"project_sec":start+duration,
                "clips":[{"id":name,"name":name,"track_id":"a","start_sec":start,"length_sec":duration,
                    "takes":[{"id":"take","source_path":"source","source_start_sec":source_start,
                        "source_end_sec":source_start+source_duration,"playback_rate":rate}]}]
            })).unwrap();
            timeline.clips[0].normalize_takes();
            let region = AraPlaybackRegion {
                audio_source_persistent_id: "source".into(),
                start_in_modification_time: source_start,
                duration_in_modification_time: source_duration,
                start_in_playback_time: start,
                duration_in_playback_time: duration,
                is_timestretch_enabled: true,
                ..Default::default()
            };
            let input = RenderInput {
                timeline: Some(timeline),
                regions: vec![region],
                sources: HashMap::from([("source".into(), source)]),
                project_sample_rate: None,
                clip_parameters: BTreeMap::new(),
            };
            let output = input
                .render(Arc::new(AtomicBool::new(false)))
                .expect("一帧源尾取整不得拒绝整个renderer");
            for snapshot in output {
                assert!(
                    snapshot.left.iter().any(|sample| sample.abs() > 0.01),
                    "{name}必须有实际播放PCM"
                );
            }
        }
    }
    /// 源表替换不能改变已捕获的输入；取消必须拒绝计算，不靠旧模型裸引用。
    #[test]
    fn frozen_pcm_is_independent_of_replacement_and_cancel_is_respected() {
        let source = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        });
        let region = AraPlaybackRegion {
            audio_source_persistent_id: "fixture".into(),
            duration_in_modification_time: 4.0 / 44100.0,
            duration_in_playback_time: 4.0 / 44100.0,
            ..Default::default()
        };
        let mut sources = HashMap::from([("fixture".into(), source)]);
        let input = RenderInput {
            timeline: None,
            regions: vec![region],
            sources: sources.clone(),
            project_sample_rate: None,
            clip_parameters: BTreeMap::new(),
        };
        sources.insert(
            "fixture".into(),
            Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![vec![0.9; 4]],
                version: 1,
                _reservation: None,
            }),
        );
        drop(sources);
        let cancel = Arc::new(AtomicBool::new(false));
        let output = input.render(cancel.clone()).unwrap();
        assert_eq!(output[0].left, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(output[0].right, output[0].left);
        assert_eq!(output[1].sample_rate, 48000);
        cancel.store(true, Ordering::Release);
        assert!(input.render(cancel).is_err());
    }
    /// **导出到文件必须与播放路径产出同一份音频。**
    ///
    /// 【为什么这条是核心】导出若另写一份混合逻辑，两份迟早漂移 —— 症状是"导出的音频
    /// 和听到的不一样"，那比不提供导出更糟。这里直接比对"导出文件解码回来的样本"与
    /// "播放路径 `mix_interleaved` 的样本"。
    #[test]
    fn exporting_a_range_writes_the_same_audio_the_playback_path_produces() {
        let frames = 8_000_usize;
        let seconds = frames as f64 / 44_100.;
        let mono = (0..frames)
            .map(|n| ((n as f32) * 0.01).sin() * 0.5)
            .collect::<Vec<f32>>();
        let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks": [{"id": "a", "name": "A", "order": 0}],
            "bpm": 120,
            "project_sec": 1.0,
            "clips": [{
                "id": "clip",
                "track_id": "a",
                "name": "voice",
                "start_sec": 0.0,
                "length_sec": seconds,
                "takes": [{
                    "id": "take",
                    "source_path": "source",
                    "source_start_sec": 0.0,
                    "source_end_sec": seconds,
                    "playback_rate": 1.0
                }]
            }]
        }))
        .unwrap();
        timeline.clips[0].normalize_takes();
        let input = RenderInput {
            timeline: Some(timeline.clone()),
            regions: Vec::new(),
            sources: HashMap::from([(
                "source".into(),
                Arc::new(SourcePcm {
                    covered: Vec::new(),
                    sample_rate: 44_100,
                    planes: vec![mono.clone(), mono],
                    version: 1,
                    _reservation: None,
                }),
            )]),
            project_sample_rate: None,
            clip_parameters: BTreeMap::new(),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let sources = input.interleaved_sources(&timeline).unwrap();
        let expected = input
            .mix_interleaved(MixRequest {
                timeline: &timeline,
                sources: &sources,
                start_sec: 0.,
                end_sec: seconds,
                sample_rate: 44_100,
                cancel: &cancel,
                progress: None,
            })
            .unwrap();
        assert!(
            expected.iter().any(|sample| sample.abs() > 1e-4),
            "夹具必须产出实际音频，否则这条测试什么也没证明"
        );

        let dir = std::env::temp_dir().join(format!("hfs-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("range.wav");
        let _ = std::fs::remove_file(&path);
        let summary = input
            .export_range_to_file(ExportRequest {
                start_sec: 0.,
                end_sec: seconds,
                output_path: &path,
                spec: &hifishifter_kernel::encode::OutputSpec::wav_32f(),
                sample_rate: 44_100,
                cancel,
                progress: None,
            })
            .unwrap();
        assert!(summary.bytes_written > 0, "必须真的写出数据");

        let (rate, channels, written) =
            hifishifter_kernel::audio_utils::decode_audio_f32_interleaved(&path).unwrap();
        assert_eq!((rate, channels), (44_100, 2));
        assert_eq!(
            written.len(),
            expected.len(),
            "写出的样本数必须与播放路径一致"
        );
        for (index, (a, b)) in written.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "第 {index} 个样本与播放路径不一致：{a} vs {b}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}
