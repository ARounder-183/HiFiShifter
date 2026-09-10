//! 静音检测（Silence Detection）：识别 Clip 活跃 Take 消费窗口内的静音段。
//!
//! 设计要点：
//! - 在**时间线域**按引擎播放语义展开虚拟信号后分析（正向 / 倒放锚定窗 /
//!   Loop 整文件回卷 / slip 前导静音，均与 `audio_engine::mix::sample_clip_pcm`
//!   同款取数规则），保证"检测到的静音 = 实际听到的静音"；
//! - 分析基于**源音频本身**，忽略 Clip 增益 / 淡化 / 轨道音量（静音是内容
//!   属性，不应受混音设置影响）；
//! - 粗分层级 5 ms hop（与参数线 `frame_period_ms` 对齐），边界在 ±25 ms 内
//!   以 0.25 ms 步进精修，随后内缩保留余量（padding）。

use std::path::Path;

/// 静音检测选项（与 models::SilenceDetectOptionsPayload 一一对应）。
#[derive(Debug, Clone)]
pub(crate) struct SilenceDetectOptions {
    /// true = 按帧内最大绝对值（Peak）判定；false = 按 RMS（推荐）。
    pub(crate) use_peak: bool,
    /// 阈值 dBFS（−96…−6）。
    pub(crate) threshold_db: f64,
    /// 自适应阈值：忽略 `threshold_db`，按全片电平分布估计噪底 + 6 dB。
    pub(crate) adaptive: bool,
    /// 最短静音时长（ms）：短于此的安静段不切。
    pub(crate) min_silence_ms: f64,
    /// 最短发声时长（ms）：被静音包夹、短于此的发声段视为杂音并入静音（0 = 关）。
    pub(crate) min_sound_ms: f64,
    /// 保留余量（ms）：切口向静音内侧各保留的时长。
    pub(crate) padding_ms: f64,
}

impl Default for SilenceDetectOptions {
    fn default() -> Self {
        Self {
            use_peak: false,
            threshold_db: -50.0,
            adaptive: false,
            min_silence_ms: 120.0,
            min_sound_ms: 0.0,
            padding_ms: 10.0,
        }
    }
}

const HOP_SEC: f64 = 0.005;
const WINDOW_SEC: f64 = 0.02;
const REFINE_WINDOW_SEC: f64 = 0.025;
const REFINE_STEP_SEC: f64 = 0.00025;

/// 单次分析的最大 hop 数（5ms hop ≈ 5.5 小时）。防御性上限：损坏工程数据
/// 里的异常巨大 `clip_length_sec` 会先触发分配失败把进程 abort。
const MAX_ANALYSIS_HOPS: usize = 4_000_000;

/// 对单个 Take 的消费窗口做静音分析。
///
/// 返回**时间线绝对秒**锚定的静音区间（升序、已合并、已内缩 padding）。
/// `source_start_sec / source_end_sec` 为该 Take 的原始窗口字段（可为负 /
/// 超出媒体时长，语义与引擎一致）；`playback_rate` = clip_rate × take_rate。
#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_take_silence(
    source_path: &str,
    source_start_sec: f64,
    source_end_sec: f64,
    playback_rate: f64,
    reversed: bool,
    loop_enabled: bool,
    clip_start_sec: f64,
    clip_length_sec: f64,
    options: &SilenceDetectOptions,
) -> Result<Vec<(f64, f64)>, String> {
    if !clip_length_sec.is_finite() || clip_length_sec <= 1e-6 {
        return Ok(Vec::new());
    }
    let rate = if playback_rate.is_finite() && playback_rate > 1e-6 {
        playback_rate
    } else {
        1.0
    };

    // 走进程级解码缓存（见 P1-3）。
    let decoded = crate::audio_utils::decode_audio_cached_interleaved(Path::new(source_path))?;
    let sample_rate = decoded.sample_rate;
    let channels = decoded.channels;
    let pcm = decoded.pcm.clone();
    let channels = channels.max(1) as usize;
    let frames = pcm.len() / channels;
    if frames == 0 || sample_rate == 0 {
        return Ok(Vec::new());
    }
    let media_dur_sec = frames as f64 / sample_rate as f64;
    let sr = sample_rate as f64;

    let ss = if source_start_sec.is_finite() { source_start_sec } else { 0.0 };
    let se = if source_end_sec.is_finite() { source_end_sec } else { media_dur_sec };

    // ── 时间线 local 秒 → 源帧索引（None = 该位置为静音）──────────────────
    // 与 sample_clip_pcm 的取数语义一致（Loop 整文件回绕 / 非循环越界静音 /
    // 方向性前导静音）。
    let (win_start, win_end, leading) = if loop_enabled {
        (ss, se, 0.0f64)
    } else if reversed {
        let se_c = se.min(media_dur_sec).max(0.0);
        let ss_c = ss.max(0.0);
        let leading = ((se - media_dur_sec).max(0.0)) / rate;
        (ss_c, se_c, leading)
    } else {
        let ss_c = ss.max(0.0);
        let se_c = se.min(media_dur_sec).max(ss_c);
        let leading = (-ss).max(0.0) / rate;
        (ss_c, se_c, leading)
    };
    let range_sec = if loop_enabled {
        media_dur_sec
    } else {
        (win_end - win_start).max(0.0)
    };

    let sample_at = move |local_sec: f64| -> Option<f32> {
        if local_sec < 0.0 || local_sec > clip_length_sec {
            return None;
        }
        let consumed = (local_sec - leading) * rate;
        // Loop 语义与引擎 / mixdown 一致（帧域）：
        //   正放 idx(f) = floor_mod(round(ss·sr) + f, D_frames)
        //   倒放 idx(f) = floor_mod(round(se·sr) − 1 − f, D_frames)
        // 倒放锚点同样 clamp 到媒体时长（source_end 越界时与引擎对齐）。
        let frame_f = if loop_enabled {
            let consumed_frames = consumed * sr;
            if reversed {
                let anchor = (se.min(media_dur_sec) * sr).round() - 1.0;
                (anchor - consumed_frames).rem_euclid(frames as f64)
            } else {
                (ss * sr + consumed_frames).rem_euclid(frames as f64)
            }
        } else {
            if consumed < 0.0 {
                return None;
            }
            if consumed >= range_sec {
                return None;
            }
            let src_sec = if reversed {
                win_end - consumed
            } else {
                win_start + consumed
            };
            src_sec * sr
        };
        if !(frame_f >= 0.0) {
            return None;
        }
        let frame = frame_f.floor().clamp(0.0, (frames - 1) as f64) as usize;
        let base = frame * channels;
        // 峰值选择下混：取各声道中绝对值最大的样本。若取声道平均，
        // 反相立体声（L=−R）会整体抵消成"静音"，与导出 / 监听表现不符。
        let mut best = pcm[base];
        let mut best_abs = best.abs();
        for ch in 1..channels {
            let v = pcm[base + ch];
            let a = v.abs();
            if a > best_abs {
                best_abs = a;
                best = v;
            }
        }
        Some(best)
    };

    // ── 逐 hop 电平（RMS / Peak）──────────────────────────────────────────
    let total_hops = ((clip_length_sec / HOP_SEC).ceil() as usize).max(1);
    if total_hops > MAX_ANALYSIS_HOPS {
        return Err(format!(
            "clip too long for silence analysis ({clip_length_sec:.0}s)"
        ));
    }
    let refine_step = REFINE_STEP_SEC;
    let mut levels_db: Vec<f64> = Vec::with_capacity(total_hops);
    for hop in 0..total_hops {
        let center = (hop as f64 + 0.5) * HOP_SEC;
        let win_start_t = (center - WINDOW_SEC / 2.0).max(0.0);
        let win_end_t = (center + WINDOW_SEC / 2.0).min(clip_length_sec);
        let mut sum_sq = 0.0f64;
        let mut peak = 0.0f64;
        let mut n = 0usize;
        let mut t = win_start_t;
        while t <= win_end_t {
            if let Some(v) = sample_at(t) {
                let a = v.abs() as f64;
                sum_sq += a * a;
                peak = peak.max(a);
                n += 1;
            }
            t += refine_step;
        }
        if n == 0 {
            // hop 完全落在消费域之外（越界 / 前导静音）：电平记 -inf。
            levels_db.push(f64::NEG_INFINITY);
            continue;
        }
        let level = if options.use_peak { peak } else { (sum_sq / n as f64).sqrt() };
        levels_db.push(20.0 * level.max(1e-10).log10());
    }

    // ── 阈值（固定 / 自适应噪底）──────────────────────────────────────────
    let threshold_db = if options.adaptive {
        let mut finite: Vec<f64> = levels_db.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            // 整段无内容 → 全静音，交给下方分段逻辑输出整段区间。
            f64::NEG_INFINITY
        } else {
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((finite.len() as f64) * 0.10).ceil() as usize;
            let floor_db = finite[idx.saturating_sub(1).min(finite.len() - 1)];
            (floor_db + 6.0).clamp(-96.0, -6.0)
        }
    } else {
        options.threshold_db.clamp(-96.0, -6.0)
    };

    // ── 分段：hop 级静音标记 → 连续段（应用最短发声合并）→ 最短静音过滤 ──
    let silent_flags: Vec<bool> = levels_db
        .iter()
        .map(|v| !v.is_finite() || *v < threshold_db)
        .collect();

    let runs = |flags: &[bool]| -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for (i, s) in flags.iter().enumerate() {
            match (start, s) {
                (None, true) => start = Some(i),
                (Some(_), false) => {
                    out.push((start.take().unwrap(), i));
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            out.push((s, flags.len()));
        }
        out
    };

    let mut silent_runs = runs(&silent_flags);
    // 最短发声时长：被静音包夹、短于阈值的发声段并入静音（0 = 关）。
    if options.min_sound_ms > 0.0 && silent_runs.len() >= 2 {
        let min_sound_hops = (options.min_sound_ms / 1000.0 / HOP_SEC).ceil() as usize;
        let mut merged_flags = silent_flags.clone();
        for w in silent_runs.windows(2) {
            let gap_start = w[0].1;
            let gap_end = w[1].0;
            if gap_end - gap_start < min_sound_hops && gap_end > gap_start {
                for f in merged_flags.iter_mut().take(gap_end).skip(gap_start) {
                    *f = true;
                }
            }
        }
        silent_runs = runs(&merged_flags);
    }
    // 最短静音时长：短于此的安静段不切。
    let min_silence_hops = (options.min_silence_ms / 1000.0 / HOP_SEC).ceil() as usize;
    silent_runs.retain(|(s, e)| e - s >= min_silence_hops.max(1));

    if silent_runs.is_empty() {
        return Ok(Vec::new());
    }

    // ── hop 段 → 秒区间 + 边界精修 + padding 内缩 ──────────────────────────
    let thr_lin = 10f64.powf(threshold_db / 20.0);
    let mut regions: Vec<(f64, f64)> = Vec::with_capacity(silent_runs.len());
    for (hs, he) in silent_runs {
        let mut start_t = hs as f64 * HOP_SEC;
        let mut end_t = (he as f64 * HOP_SEC).min(clip_length_sec);

        // 精修起点：在 ±窗口内找"最后一次超过阈值"的位置 → 区间从其后开始。
        let scan_start = (start_t - REFINE_WINDOW_SEC).max(0.0);
        let scan_end = (start_t + REFINE_WINDOW_SEC).min(clip_length_sec);
        let mut last_sound: Option<f64> = None;
        let mut t = scan_start;
        while t <= scan_end {
            if let Some(v) = sample_at(t) {
                if (v.abs() as f64) > thr_lin {
                    last_sound = Some(t);
                }
            }
            t += refine_step;
        }
        if let Some(p) = last_sound {
            start_t = (p + refine_step).min(clip_length_sec);
        }

        // 精修终点：在 ±窗口内找"下一次超过阈值"的位置 → 区间到其前为止。
        let scan_start = (end_t - REFINE_WINDOW_SEC).max(0.0);
        let scan_end = (end_t + REFINE_WINDOW_SEC).min(clip_length_sec);
        let mut next_sound: Option<f64> = None;
        let mut t = scan_end;
        while t >= scan_start {
            if let Some(v) = sample_at(t) {
                if (v.abs() as f64) > thr_lin {
                    next_sound = Some(t);
                }
            }
            t -= refine_step;
        }
        if let Some(p) = next_sound {
            end_t = p.max(0.0);
        }

        // padding 内缩（保留余量在静音内侧）。
        let pad = (options.padding_ms / 1000.0).max(0.0);
        start_t += pad;
        end_t -= pad;
        if end_t - start_t <= 1e-4 {
            continue;
        }
        regions.push((
            (clip_start_sec + start_t).max(0.0),
            clip_start_sec + end_t,
        ));
    }

    // 合并相邻（精修 / padding 后可能产生的重叠或贴近段）。
    regions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(regions.len());
    for (s, e) in regions {
        match merged.last_mut() {
            Some(last) if s <= last.1 + 1e-4 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 写一段 16-bit 单声道 WAV：`amplitude_at(f) >= 0` 给出逐帧幅度（负值即终止）。
    fn write_test_wav(
        dir: &Path,
        name: &str,
        sample_rate: u32,
        frames: usize,
        mut gen: impl FnMut(usize) -> f32,
    ) -> String {
        use hound::{SampleFormat, WavSpec};
        let path = dir.join(name);
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..frames {
            let v = gen(i).clamp(-1.0, 1.0);
            writer.write_sample((v * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
        path.to_string_lossy().to_string()
    }

    /// 音型素材：tone(0..1s) + 静音(1..2s) + tone(2..3s)，44.1kHz 单声道。
    fn write_tone_gap_wav(dir: &Path, name: &str) -> String {
        let sr = 44100.0f64;
        write_test_wav(dir, name, 44100, (3.0 * sr) as usize, move |i| {
            let t = i as f64 / sr;
            if (t < 1.0 || (2.0..3.0).contains(&t)) {
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t) as f32)
            } else {
                0.0
            }
        })
    }

    fn opts() -> SilenceDetectOptions {
        SilenceDetectOptions {
            use_peak: false,
            threshold_db: -50.0,
            adaptive: false,
            min_silence_ms: 100.0,
            min_sound_ms: 0.0,
            padding_ms: 0.0,
        }
    }

    #[test]
    fn detects_silence_between_tones() {
        let dir = std::env::temp_dir();
        let path = write_tone_gap_wav(&dir, "hsd_tone_gap.wav");
        let regions =
            analyze_take_silence(&path, 0.0, 3.0, 1.0, false, false, 0.0, 3.0, &opts()).unwrap();
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let (s, e) = regions[0];
        assert!((s - 1.0).abs() < 0.05, "start {s}");
        assert!((e - 2.0).abs() < 0.05, "end {e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn min_silence_filters_short_gaps() {
        let dir = std::env::temp_dir();
        let sr = 44100.0f64;
        // 30ms 的短气口（低于 100ms 最短静音）不应被切除。
        let path = write_test_wav(&dir, "hsd_short_gap.wav", 44100, (3.0 * sr) as usize, |i| {
            let t = i as f64 / sr;
            if (0.0..1.0).contains(&t) || t >= 1.03 {
                0.5 * (2.0 * std::f64::consts::PI * 440.0 * t) as f32
            } else {
                0.0
            }
        });
        let regions =
            analyze_take_silence(&path, 0.0, 3.0, 1.0, false, false, 0.0, 3.0, &opts()).unwrap();
        assert!(regions.is_empty(), "regions: {regions:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn padding_shrinks_regions() {
        let dir = std::env::temp_dir();
        let path = write_tone_gap_wav(&dir, "hsd_padding.wav");
        let mut o = opts();
        o.padding_ms = 50.0;
        let regions =
            analyze_take_silence(&path, 0.0, 3.0, 1.0, false, false, 0.0, 3.0, &o).unwrap();
        assert_eq!(regions.len(), 1);
        let (s, e) = regions[0];
        // padding 保留在静音内侧：起点后移、终点前移约 50ms。
        assert!(s > 1.0 && (s - 1.0) <= 0.1, "start {s}");
        assert!(e < 2.0 && (2.0 - e) <= 0.1, "end {e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn clip_offset_maps_regions_to_timeline_time() {
        let dir = std::env::temp_dir();
        let path = write_tone_gap_wav(&dir, "hsd_offset.wav");
        // 源窗口 [0.5, 2.5]（rate=1）：clip 从时间线 10s 开始、长 2s；
        // 源 [1.0, 2.0] 的静音映射到时间线 [10.5, 11.5]。
        let regions =
            analyze_take_silence(&path, 0.5, 2.5, 1.0, false, false, 10.0, 2.0, &opts()).unwrap();
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let (s, e) = regions[0];
        assert!((s - 10.5).abs() < 0.05, "start {s}");
        assert!((e - 11.5).abs() < 0.05, "end {e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn playback_rate_stretches_silence() {
        let dir = std::env::temp_dir();
        let path = write_tone_gap_wav(&dir, "hsd_rate.wav");
        // rate = 0.5：源 1s 静音在时间线上占 2s（clip 长度 = 3/0.5 = 6s）。
        let regions =
            analyze_take_silence(&path, 0.0, 3.0, 0.5, false, false, 0.0, 6.0, &opts()).unwrap();
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let (s, e) = regions[0];
        assert!((s - 2.0).abs() < 0.1, "start {s}");
        assert!((e - 4.0).abs() < 0.1, "end {e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reversed_take_maps_via_anchor_window() {
        let dir = std::env::temp_dir();
        let path = write_tone_gap_wav(&dir, "hsd_reversed.wav");
        // 倒放窗口 [0,3]：时间线 t 播放源 3−t；源静音 [1,2] → 时间线 [1,2]。
        let regions =
            analyze_take_silence(&path, 0.0, 3.0, 1.0, true, false, 0.0, 3.0, &opts()).unwrap();
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let (s, e) = regions[0];
        assert!((s - 1.0).abs() < 0.05, "start {s}");
        assert!((e - 2.0).abs() < 0.05, "end {e}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fully_silent_clip_returns_single_region() {
        let dir = std::env::temp_dir();
        let sr = 44100.0f64;
        let path = write_test_wav(&dir, "hsd_silent.wav", 44100, (2.0 * sr) as usize, |_| 0.0f32);
        let regions =
            analyze_take_silence(&path, 0.0, 2.0, 1.0, false, false, 0.0, 2.0, &opts()).unwrap();
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let (s, e) = regions[0];
        assert!(s <= 0.05 && e >= 1.95, "({s}, {e})");
        let _ = std::fs::remove_file(&path);
    }
}
