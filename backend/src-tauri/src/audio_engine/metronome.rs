//! 节拍器（Metronome）：跟随网格标尺与 Tempo Map 的点击声。
//!
//! 设计要点：
//! - **响点 = 当前网格线；与小节首重合的响点用重音**。网格步长镜像前端
//!   `gridStepBeats`（含附点 ×1.5、三连音 ×2/3）；
//! - 响点表（[`MetronomeClick`] 升序列表）在命令层按当前工程
//!   （BPM / Tempo Map / 工程长度）与 UI 设置（网格、细分模式）预展开，
//!   经 `EngineCommand::SetMetronomeSchedule` 换入 [`MetronomeRt::schedule`]；
//! - RT 回调只做二分查找 + 少量叠加合成（正弦 × 指数衰减），全部原子 /
//!   ArcSwap 读取，无锁无分配；跨块尾音由 RT 本地的 [`MetronomeVoices`] 承接；
//! - 混音导出（离线 mixdown）不含节拍器 —— 本模块仅存在于实时回调路径。

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;

/// 单个响点：`frame` 相对工程 0 点（输出设备采样率域，升序），`accent` =
/// 该响点是否落在小节首。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetronomeClick {
    pub(crate) frame: u64,
    pub(crate) accent: bool,
}

/// 细分模式：跟随网格 / 仅每拍 / 仅小节首。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MetronomeMode {
    #[default]
    Grid,
    Beat,
    Bar,
}

/// 音色（程序化合成，无资源文件）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MetronomeSound {
    #[default]
    Click,
    Woodblock,
    Beep,
}

impl MetronomeSound {
    pub(crate) fn to_u8(self) -> u8 {
        match self {
            MetronomeSound::Click => 0,
            MetronomeSound::Woodblock => 1,
            MetronomeSound::Beep => 2,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            1 => MetronomeSound::Woodblock,
            2 => MetronomeSound::Beep,
            _ => MetronomeSound::Click,
        }
    }
}

/// 节拍器配置（命令线程构造，RT 侧经原子读取）。
///
/// 细分模式不在此处：响点表由命令层按模式预展开，RT 只消费响点表。
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetronomeConfig {
    pub(crate) enabled: bool,
    /// 0..1（映射自前端 0..100%）。
    pub(crate) gain: f32,
    /// 关闭时全部响点使用非重音音色。
    pub(crate) accent_enabled: bool,
    pub(crate) sound: MetronomeSound,
}

/// RT 共享状态：worker（命令处理）写，音频回调读。所有字段无锁。
pub(crate) struct MetronomeRt {
    enabled: AtomicBool,
    gain_bits: AtomicU32,
    accent_enabled: AtomicBool,
    sound: AtomicU8,
    /// 响点表世代：schedule / config 每次 `store_*` 换入即 +1。RT 侧据此
    /// 丢弃跨换表的残留尾音——旧响点表的声音数据被**整体、立即**清理，
    /// 不会与换表后的新响点叠加。
    generation: AtomicU64,
    /// 响点表（升序）。None = 尚未构建 / 工程为空。
    schedule: ArcSwapOption<Vec<MetronomeClick>>,
}

impl MetronomeRt {
    pub(crate) fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            gain_bits: AtomicU32::new(0.5f32.to_bits()),
            accent_enabled: AtomicBool::new(true),
            sound: AtomicU8::new(MetronomeSound::default().to_u8()),
            generation: AtomicU64::new(0),
            schedule: ArcSwapOption::from_pointee(Vec::new()),
        }
    }

    pub(crate) fn store_config(&self, config: &MetronomeConfig) {
        // 配置变化同样推进世代：换音色 / 换增益时旧的尾音一并作废。
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.enabled
            .store(config.enabled, std::sync::atomic::Ordering::Relaxed);
        self.gain_bits
            .store(config.gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.accent_enabled
            .store(config.accent_enabled, std::sync::atomic::Ordering::Relaxed);
        self.sound
            .store(config.sound.to_u8(), std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn store_schedule(&self, clicks: Arc<Vec<MetronomeClick>>) {
        // 推进世代：RT 在下一个块把旧表的残留尾音整体清空。
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.schedule.store(Some(clicks));
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    fn gain(&self) -> f32 {
        f32::from_bits(self.gain_bits.load(Ordering::Relaxed))
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn accent_enabled(&self) -> bool {
        self.accent_enabled.load(Ordering::Relaxed)
    }

    fn sound(&self) -> MetronomeSound {
        MetronomeSound::from_u8(self.sound.load(Ordering::Relaxed))
    }
}

// ─── RT 合成 ─────────────────────────────────────────────────────────────────

/// 单个音色的合成参数：基频 + 二次谐波比例 + 衰减时间常数。
#[derive(Debug, Clone, Copy)]
struct ClickTimbre {
    freq: f32,
    tau: f32,
    harm2: f32,
}

fn click_timbre(sound: MetronomeSound, accent: bool) -> ClickTimbre {
    match (sound, accent) {
        // 短促点击：重音高、非重音低。
        (MetronomeSound::Click, true) => ClickTimbre { freq: 1800.0, tau: 0.012, harm2: 0.0 },
        (MetronomeSound::Click, false) => ClickTimbre { freq: 1200.0, tau: 0.012, harm2: 0.0 },
        // 木鱼：更短促、带少量二次谐波增加“木质感”。
        (MetronomeSound::Woodblock, true) => ClickTimbre { freq: 1900.0, tau: 0.008, harm2: 0.35 },
        (MetronomeSound::Woodblock, false) => ClickTimbre { freq: 1250.0, tau: 0.008, harm2: 0.35 },
        // 蜂鸣：长尾、同音双音高。
        (MetronomeSound::Beep, true) => ClickTimbre { freq: 880.0, tau: 0.06, harm2: 0.2 },
        (MetronomeSound::Beep, false) => ClickTimbre { freq: 660.0, tau: 0.06, harm2: 0.2 },
    }
}

/// 包络衰减到 1e-4 的尾长上限（秒）。9.2·τ 已足够安静，再封顶防极端参数。
const CLICK_MAX_TAIL_SEC: f32 = 0.25;

fn click_tail_frames(timbre: &ClickTimbre, sample_rate: u32) -> i64 {
    let tail = (timbre.tau * 9.21).clamp(0.001, CLICK_MAX_TAIL_SEC);
    (tail * sample_rate as f32).ceil() as i64
}

/// 峰值幅度系数：留出叠加余量，最终输出仍经 `clamp11`。
const CLICK_PEAK: f32 = 0.8;

/// 同时存在的响点尾音上限。极限情形（1/64 网格 @ 960 BPM ≈ 3.9 ms 步长，
/// beep 尾长 250 ms）约 64 个；超限时丢弃最旧的尾音。
const MAX_METRONOME_VOICES: usize = 64;

struct ActiveClick {
    start_frame: i64,
    end_frame: i64,
    timbre: ClickTimbre,
}

/// RT 本地（每音频回调线程一份）的响点尾音池：承接跨块边界的尾音。
pub(crate) struct MetronomeVoices {
    /// 上次混音时的响点表世代；失配即清空残留尾音（见 [`Self::mix`]）。
    generation: u64,
    /// 活跃尾音，按 start_frame 升序（生成顺序即升序）。
    active: Vec<ActiveClick>,
}

impl Default for MetronomeVoices {
    fn default() -> Self {
        Self {
            generation: 0,
            active: Vec::with_capacity(8),
        }
    }
}

impl MetronomeVoices {
    /// 将 `[pos0, pos1)` 块内需要发声的响点叠加进 stereo interleaved scratch。
    ///
    /// 调用约定与 `mix_into_scratch_stereo` 一致：仅在 `is_playing` 且快照
    /// 可渲染时调用（自动暂停等静音路径不响节拍器）。
    pub(crate) fn mix(
        &mut self,
        scratch: &mut [f32],
        metro: &MetronomeRt,
        pos0: u64,
        pos1: u64,
        sample_rate: u32,
    ) {
        // ① 世代失配（换响点表 / 换配置）→ 旧表的残留尾音整体丢弃，
        //    保证换表后绝无“修改前 + 修改后”的响点叠加。
        let generation = metro.generation();
        if generation != self.generation {
            self.active.clear();
            self.generation = generation;
        }

        // ② 无条件剪枝：尾部已越过高水位线的尾音一律移除（不依赖首元素
        //    是否结束——位置回退 / 重播后 active 的顺序不再保证单调）。
        let pos0_i64 = pos0 as i64;
        self.active.retain(|v| v.end_frame > pos0_i64);

        if !metro.enabled() {
            return;
        }
        let gain = metro.gain();
        if gain <= 1e-4 {
            return;
        }
        let Some(schedule) = metro.schedule.load_full() else {
            return;
        };
        let schedule: &[MetronomeClick] = &schedule;

        // 生成：块内新起的响点（schedule 升序，二分定位起点）。
        let accent_enabled = metro.accent_enabled();
        let sound = metro.sound();
        let mut idx = schedule.partition_point(|c| c.frame < pos0);
        while idx < schedule.len() && schedule[idx].frame < pos1 {
            let click = &schedule[idx];
            let start = click.frame as i64;
            // ③ 起振去重：该帧的尾音已在池中（位置回退 / 播放头回跳后
            //    重新覆盖同一帧区间）则跳过，绝不叠加第二次起振。
            if !self.active.iter().any(|v| v.start_frame == start) {
                let accent = click.accent && accent_enabled;
                let timbre = click_timbre(sound, accent);
                let end = start + click_tail_frames(&timbre, sample_rate);
                if end > pos0_i64 {
                    if self.active.len() >= MAX_METRONOME_VOICES {
                        self.active.remove(0);
                    }
                    self.active.push(ActiveClick {
                        start_frame: start,
                        end_frame: end,
                        timbre,
                    });
                }
            }
            idx += 1;
        }
        if self.active.is_empty() {
            return;
        }

        // 叠加合成：闭式公式直接按绝对相位取样，无需跨块相位状态。
        let sr = sample_rate.max(1) as f32;
        let two_pi = std::f32::consts::TAU;
        let frames = (pos1 - pos0).min((scratch.len() / 2) as u64) as usize;
        for voice in &self.active {
            let from = voice.start_frame.max(pos0_i64);
            let to = voice.end_frame.min(pos1 as i64);
            for f in from..to {
                let off = (f - pos0_i64) as usize;
                if off >= frames {
                    break;
                }
                let t = (f - voice.start_frame) as f32 / sr;
                let env = (-t / voice.timbre.tau).exp();
                let w = two_pi * voice.timbre.freq * t;
                let s = w.sin() + voice.timbre.harm2 * (2.0 * w).sin();
                let amp = gain * CLICK_PEAK * env * s;
                let i = off * 2;
                scratch[i] += amp;
                scratch[i + 1] += amp;
            }
        }
    }
}

// ─── 响点表构建（命令线程） ──────────────────────────────────────────────────

/// Tempo Map 分段（拍域积分缓存，[`build_tempo_segments`] 产物）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetroSegment {
    /// 段起始时间（秒）。
    pub(crate) start_sec: f64,
    /// 段起始拍（全工程累计拍，0 = 工程开头）。
    pub(crate) start_beat: f64,
    /// 每拍秒数（60 / BPM）。
    pub(crate) sec_per_beat: f64,
    /// 每小节拍数（分子 × 4 ÷ 分母；允许小数，如 7/8 = 3.5）。
    pub(crate) beats_per_bar: f64,
    /// 小节对齐锚（拍域）：拍号变化点重置（REAPER 语义），
    /// 重音判定 = `(beat - bar_anchor_beat) mod beats_per_bar == 0`。
    pub(crate) bar_anchor_beat: f64,
}

/// 工程拍号 → 每小节拍数。
fn project_bar_beats(beats_per_bar: u32, denominator: u32) -> f64 {
    let num = if beats_per_bar >= 1 { beats_per_bar as f64 } else { 4.0 };
    let den = if denominator >= 1 { denominator as f64 } else { 4.0 };
    num * 4.0 / den
}

/// 由工程 BPM / Tempo Map（拍号变化点重置小节对齐）构建拍域积分分段。
///
/// 无 Tempo Map 时返回单段（工程 BPM + 工程拍号）。
pub(crate) fn build_tempo_segments(
    bpm: f64,
    tempo_map: Option<&[crate::state::TempoPointData]>,
    project_beats_per_bar: u32,
    project_denominator: u32,
) -> Vec<MetroSegment> {
    let fallback_bpb = project_bar_beats(project_beats_per_bar, project_denominator);
    let points: Vec<&crate::state::TempoPointData> = match tempo_map {
        Some(points) if !points.is_empty() => points.iter().collect(),
        _ => {
            let spb = if bpm.is_finite() && bpm > 0.0 { 60.0 / bpm } else { 0.5 };
            return vec![MetroSegment {
                start_sec: 0.0,
                start_beat: 0.0,
                sec_per_beat: spb,
                beats_per_bar: fallback_bpb,
                bar_anchor_beat: 0.0,
            }];
        }
    };

    let mut segments: Vec<MetroSegment> = Vec::with_capacity(points.len() + 1);
    let mut beat_cursor = 0.0f64;
    let mut prev_pos = 0.0f64;
    let mut prev_spb = if bpm.is_finite() && bpm > 0.0 { 60.0 / bpm } else { 0.5 };
    let mut prev_bpb = fallback_bpb;
    let mut bar_anchor = 0.0f64;
    let mut started = false;
    for point in points {
        let pos = if point.position_sec.is_finite() && point.position_sec >= 0.0 {
            point.position_sec
        } else {
            0.0
        };
        let point_bpm = if point.bpm.is_finite() && point.bpm > 0.0 { point.bpm } else { 120.0 };
        let bpb = match (point.numerator, point.denominator) {
            (Some(n), Some(d)) if n >= 1 && d >= 1 => (n as f64) * 4.0 / (d as f64),
            _ => prev_bpb,
        };
        if !started {
            // 首点不在 0 时先补一段工程基准段，保证从工程开头有节奏定义。
            if pos > 1e-9 {
                segments.push(MetroSegment {
                    start_sec: 0.0,
                    start_beat: 0.0,
                    sec_per_beat: prev_spb,
                    beats_per_bar: fallback_bpb,
                    bar_anchor_beat: 0.0,
                });
                beat_cursor = pos / prev_spb;
            }
            started = true;
        } else if pos <= prev_pos + 1e-9 {
            // 重复 / 回退位置：忽略（与规范化后的 Tempo Map 不一致时兜底）。
            continue;
        } else {
            beat_cursor += (pos - prev_pos) / prev_spb;
        }
        // 仅拍号变化时重置小节对齐（REAPER 语义）；纯变速段保持原对齐。
        let signature_changed = segments
            .last()
            .is_some_and(|prev| (bpb - prev.beats_per_bar).abs() > 1e-9);
        if segments.is_empty() || signature_changed {
            bar_anchor = beat_cursor;
        }
        let spb = 60.0 / point_bpm;
        segments.push(MetroSegment {
            start_sec: pos,
            start_beat: beat_cursor,
            sec_per_beat: spb,
            beats_per_bar: bpb,
            bar_anchor_beat: bar_anchor,
        });
        prev_pos = pos;
        prev_spb = spb;
        prev_bpb = bpb;
    }
    segments
}

/// 镜像前端 `gridStepBeats`（frontend/src/components/layout/timeline/grid.ts）：
/// 基础值按拍（1 拍 = 四分音符）计，附点 ×1.5、三连音 ×2/3。
pub(crate) fn grid_step_beats(grid: &str) -> Option<f64> {
    let (base_part, suffix) = match grid.rfind(['d', 't']) {
        Some(idx) if idx > 1 => (&grid[..idx], &grid[idx..]),
        _ => (grid, ""),
    };
    let base = match base_part {
        "1/1" => 4.0,
        "1/2" => 2.0,
        "1/4" => 1.0,
        "1/8" => 0.5,
        "1/16" => 0.25,
        "1/32" => 0.125,
        "1/64" => 0.0625,
        _ => return None,
    };
    let mult = match suffix {
        "d" => 1.5,
        "t" => 2.0 / 3.0,
        _ => 1.0,
    };
    let step: f64 = base * mult;
    (step.is_finite() && step > 0.0).then_some(step)
}

/// 由拍域分段 + 步长展开响点表（升序，含重音标记）。
///
/// `step_beats` = 每个响点间隔的拍数；`0` 表示“仅小节首”模式（按各段的
/// 拍号锚步进小节首）。工程为空（`duration_sec <= 0`）时返回空表。
pub(crate) fn build_click_schedule(
    segments: &[MetroSegment],
    step_beats: f64,
    sample_rate: u32,
    duration_sec: f64,
) -> Vec<MetronomeClick> {
    let mut clicks: Vec<MetronomeClick> = Vec::new();
    if !(duration_sec.is_finite()) || duration_sec <= 0.0 || sample_rate == 0 {
        return clicks;
    }
    let sr = sample_rate as f64;
    let eps = 1e-6;
    for (i, seg) in segments.iter().enumerate() {
        let seg_end_beat = segments.get(i + 1).map(|s| s.start_beat);
        let step = if step_beats > eps { step_beats } else { seg.beats_per_bar };
        if !step.is_finite() || step <= eps {
            continue;
        }
        // 本段内第一个响点拍：全局网格对齐到步长倍数；“仅小节首”模式从
        // 小节锚（推进到段内）开始。
        let mut beat = if step_beats > eps {
            (seg.start_beat / step).ceil() * step
        } else {
            let n = ((seg.start_beat - seg.bar_anchor_beat) / step).ceil().max(0.0);
            seg.bar_anchor_beat + n * step
        };
        while beat < seg_end_beat.unwrap_or(f64::INFINITY) - eps {
            let sec = seg.start_sec + (beat - seg.start_beat) * seg.sec_per_beat;
            if sec > duration_sec + eps {
                break;
            }
            if sec >= -eps {
                let frame = (sec * sr).round().max(0.0) as u64;
                let within_bar = (beat - seg.bar_anchor_beat) % seg.beats_per_bar;
                let accent = within_bar.abs() < 1e-4
                    || (seg.beats_per_bar - within_bar).abs() < 1e-4;
                clicks.push(MetronomeClick { frame, accent });
            }
            beat += step;
        }
        if clicks.len() >= 2_000_000 {
            break;
        }
    }
    // 兜底：段边界浮点取整可能产生同帧重复响点，只保留第一个。
    clicks.dedup_by(|a, b| a.frame == b.frame);
    clicks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TempoPointData;

    fn point(pos: f64, bpm: f64, num: Option<u32>, den: Option<u32>) -> TempoPointData {
        TempoPointData {
            id: format!("p{pos}"),
            position_sec: pos,
            bpm,
            numerator: num,
            denominator: den,
            scale: None,
        }
    }

    #[test]
    fn grid_step_beats_mirrors_frontend_table() {
        let cases: &[(&str, f64)] = &[
            ("1/1", 4.0),
            ("1/2", 2.0),
            ("1/4", 1.0),
            ("1/8", 0.5),
            ("1/16", 0.25),
            ("1/32", 0.125),
            ("1/64", 0.0625),
            ("1/2d", 3.0),
            ("1/4d", 1.5),
            ("1/8d", 0.75),
            ("1/4t", 2.0 / 3.0),
            ("1/8t", 1.0 / 3.0),
        ];
        for (grid, expect) in cases {
            let got = grid_step_beats(grid).unwrap_or_else(|| panic!("{grid} should parse"));
            assert!((got - expect).abs() < 1e-9, "{grid}: {got} != {expect}");
        }
        assert!(grid_step_beats("bogus").is_none());
    }

    #[test]
    fn constant_tempo_schedule_clicks_every_beat_with_bar_accents() {
        // 120 BPM、4/4：每拍 0.5s 一个响点，44100 Hz 下重音落在每小节首。
        let segs = build_tempo_segments(120.0, None, 4, 4);
        let clicks = build_click_schedule(&segs, 1.0, 44100, 3.0);
        assert_eq!(clicks.len(), 7); // 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        let accents: Vec<usize> = clicks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.accent)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(accents, vec![0, 4]);
        assert_eq!(clicks[0].frame, 0);
        assert_eq!(clicks[1].frame, 22050);
    }

    #[test]
    fn grid_subdivision_accent_only_at_bar_starts() {
        // 1/8 网格：每半拍一响；4/4 下重音仅与小节首重合（每 8 响一次）。
        let segs = build_tempo_segments(120.0, None, 4, 4);
        let clicks = build_click_schedule(&segs, 0.5, 44100, 4.0);
        assert_eq!(clicks.len(), 17); // 拍 0, 0.5 … 8.0（含终点 4.0s 处的小节首）
        let accents: Vec<usize> = clicks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.accent)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(accents, vec![0, 8, 16]); // 拍 0、4、8
    }

    #[test]
    fn triplet_grid_bar_start_coincides_every_six_clicks() {
        // 1/4t（每 2/3 拍一响）+ 4/4：每小节 6 响，第 1、7 响为重音。
        let segs = build_tempo_segments(120.0, None, 4, 4);
        let clicks = build_click_schedule(&segs, 2.0 / 3.0, 44100, 5.0);
        let accents: Vec<usize> = clicks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.accent)
            .map(|(i, _)| i)
            .collect();
        // 5s 覆盖三个 4/4 小节（0s/2s/4s）→ 第 0、6、12 响为重音。
        assert_eq!(accents, vec![0, 6, 12]);
    }

    #[test]
    fn tempo_map_integration_and_time_signature_change_realigns_bars() {
        // 0-2s: 120 BPM（4 拍）；2s 起 60 BPM（3/4 → 3 拍/小节）。
        // 拍域：2s 处 = 4 拍；2-5s 再积 3 拍 → 共 7 拍。
        let map = vec![
            point(0.0, 120.0, Some(4), Some(4)),
            point(2.0, 60.0, Some(3), Some(4)),
        ];
        let segs = build_tempo_segments(120.0, Some(&map), 4, 4);
        assert_eq!(segs.len(), 2);
        assert!((segs[1].start_beat - 4.0).abs() < 1e-9);
        assert!((segs[1].sec_per_beat - 1.0).abs() < 1e-9);
        // 拍号变化重置小节锚：4/4 域重音拍 0（0s）；3/4 域锚 4 拍，
        // 重音拍 4、7（2s、5s）。
        let clicks = build_click_schedule(&segs, 1.0, 1000, 6.0);
        let accent_beats: Vec<f64> = clicks
            .iter()
            .filter(|c| c.accent)
            .map(|c| c.frame as f64 / 1000.0)
            .collect();
        assert_eq!(accent_beats, vec![0.0, 2.0, 5.0]);
    }

    #[test]
    fn schedule_swap_clears_stale_voices_no_overlap() {
        // 用户报告的场景：播放中（或停止后）更换响点表（拍号/BPM/Tempo Map
        // 变化 → store_schedule），随后从回退的位置继续/再次播放——
        // 旧表的残留尾音必须被整体清空，绝不能与新表的响点叠加。
        let metro = MetronomeRt::new();
        metro.store_config(&MetronomeConfig {
            enabled: true,
            gain: 0.5,
            accent_enabled: true,
            sound: MetronomeSound::Click,
        });
        let mut voices = MetronomeVoices::default();
        let segs = build_tempo_segments(120.0, None, 4, 4);

        // 表 A：120 BPM 4/4，每拍一响（0.5s = 22050 帧 @44.1kHz）。
        metro.store_schedule(Arc::new(build_click_schedule(&segs, 1.0, 44100, 60.0)));
        let gen_a = metro.generation();

        // 播放推进到 4000 帧：块 [0,1024) 起振 frame 0 的 click（尾音到 ~4876），
        // 后续块无新起振。
        let mut scratch = vec![0.0f32; 2048];
        voices.mix(&mut scratch, &metro, 0, 1024, 44100);
        assert!(scratch.iter().any(|v| v.abs() > 0.0), "block 0 should click");
        assert_eq!(voices.active.len(), 1);
        let stale_end = voices.active[0].end_frame;
        assert!(stale_end > 4000, "voice must still be ringing at 4000");

        // 前进到 [3072, 4096)：旧尾音仍在池中、仍在发声。
        for start in [1024u64, 2048, 3072] {
            voices.mix(&mut scratch, &metro, start, start + 1024, 44100);
        }
        assert_eq!(voices.active.len(), 1, "tail still ringing");

        // 换表（拍号/BPM 变化 → store_schedule，generation +1）+ 位置回退到
        // [3072, 4096) 重播：旧尾音必须被清空，该块没有任何 click 能量
        // （新表的下一个响点在 22050 帧，远在本块之外）。
        metro.store_schedule(Arc::new(build_click_schedule(&segs, 1.0, 44100, 60.0)));
        assert_eq!(metro.generation(), gen_a + 1);
        scratch.fill(0.0);
        voices.mix(&mut scratch, &metro, 3072, 4096, 44100);
        assert!(
            scratch.iter().all(|v| v.abs() == 0.0),
            "stale voice must be cleared on schedule swap (no old+new overlap)"
        );
        assert!(voices.active.is_empty());
    }

    #[test]
    fn rewind_recovering_same_click_spawns_only_once() {
        // 播放头回跳（seek 回退 / 循环回跳）后同一帧区间被重新覆盖：
        // 同一响点帧只允许起振一次，绝不叠加第二份。
        let metro = MetronomeRt::new();
        metro.store_config(&MetronomeConfig {
            enabled: true,
            gain: 0.5,
            accent_enabled: true,
            sound: MetronomeSound::Click,
        });
        let mut voices = MetronomeVoices::default();
        let segs = build_tempo_segments(120.0, None, 4, 4);
        metro.store_schedule(Arc::new(build_click_schedule(&segs, 1.0, 44100, 60.0)));

        let mut scratch = vec![0.0f32; 4096];
        // 块 [21000, 25000)：起振 22050 的 click。
        voices.mix(&mut scratch, &metro, 21000, 25000, 44100);
        let clicks_after_first = voices.active.len();
        assert_eq!(clicks_after_first, 1);

        // 回跳重播同一区间（位置不前进）：不得再起振第二个 voice。
        voices.mix(&mut scratch, &metro, 21000, 25000, 44100);
        assert_eq!(
            voices.active.len(),
            1,
            "same click frame must not spawn a second overlapping voice"
        );
    }

    #[test]
    fn bar_only_mode_clicks_on_bar_starts() {
        let segs = build_tempo_segments(120.0, None, 4, 4);
        let clicks = build_click_schedule(&segs, 0.0, 44100, 3.0);
        assert_eq!(clicks.len(), 2); // 0s、2s
        assert!(clicks.iter().all(|c| c.accent));
    }

    #[test]
    fn leading_offset_clip_window_is_representable() {
        // 工程拍号 7/8（bpb = 3.5）无 Tempo Map：小节锚 0，3.5 拍/小节。
        let segs = build_tempo_segments(120.0, None, 7, 8);
        assert!((segs[0].beats_per_bar - 3.5).abs() < 1e-9);
        let clicks = build_click_schedule(&segs, 1.0, 1000, 4.0);
        let accent_frames: Vec<u64> = clicks
            .iter()
            .filter(|c| c.accent)
            .map(|c| c.frame)
            .collect();
        // 3.5 拍/小节 → 小节首在拍 0、3.5、7；1/4 网格只落在整数拍上，
        // 故仅拍 0 与拍 7（0s、3.5s）是“与小节首重合的网格线”→ 重音。
        assert_eq!(accent_frames, vec![0, 3500]);
    }
}
