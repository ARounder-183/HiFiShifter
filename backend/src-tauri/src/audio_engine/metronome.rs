//! 节拍器（Metronome）：跟随网格标尺与 Tempo Map 的点击声。
//!
//! 设计要点（与前端 `utils/tempoMap.ts` 的 Tempo Map 语义**逐段完全一致**）：
//! - **响点 = 时间标尺/背景网格画出的网格线（弱网格线 ∪ 小节线）**，
//!   与小节线重合的响点用重音。网格步长镜像前端 `gridStepBeats`（含附点
//!   ×1.5、三连音 ×2/3）；小节线独立于弱网格参与展开 —— 时间标尺同样会
//!   单独画出不落在弱网格上的小节线（如 7/8 段内 3.5 拍处），节拍器若
//!   只跟弱网格会丢失强拍；
//! - **每个 Tempo 变化点处重新对齐小节/节拍**（`buildTempoGridLines` 的
//!   "逐段局部对齐"规则）：弱网格线 = 段起点 + k×step×每拍秒数；小节线 =
//!   段起点 + k×beatsPerBar×每拍秒数。BPM 变化点同样重新对齐 —— 段起点本身
//!   就是一条小节线（段末尾不足一小节的余拍计为"不完整小节"，
//!   与 `beatToBarBeat` 一致）；
//! - **Swing 只作用于弱网格线的奇数格**（与 `swingAt` / `swingOffsetSec`
//!   同一公式），小节线永不偏移；
//! - 拍号 carry 语义镜像 `effectiveTimeSignatures`：null 跟随前点，种子为
//!   4/4（与 `DEFAULT_TIME_SIGNATURE` / `effective_time_signature_at` 一致）；
//! - 响点表（[`MetronomeClick`] 升序列表）在命令层预展开，经
//!   `EngineCommand::SetMetronomeSchedule` 换入 [`MetronomeRt::schedule`]；
//!   RT 回调只做二分查找 + 少量叠加合成，跨块尾音由 [`MetronomeVoices`]
//!   承接（换表时按世代整体清理，绝无新旧叠加）；
//! - 混音导出（离线 mixdown）不含节拍器 —— 本模块仅存在于实时回调路径。

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;

/// 单个响点：`frame` 相对工程 0 点（输出设备采样率域，升序），`accent` =
/// 该响点是否落在小节线（强网格）上。
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
        // 木鱼：更短促、带少量二次谐波增加"木质感"。
        (MetronomeSound::Woodblock, true) => ClickTimbre { freq: 1900.0, tau: 0.008, harm2: 0.35 },
        (MetronomeSound::Woodblock, false) => ClickTimbre { freq: 1250.0, tau: 0.008, harm2: 0.35 },
        // 蜂鸣：长尾、双音高。
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
    /// 活跃尾音（生成顺序按 start_frame 升序；位置回退后不保证单调）。
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
        //    保证换表后绝无"修改前 + 修改后"的响点叠加。
        let generation = metro.generation();
        if generation != self.generation {
            self.active.clear();
            self.generation = generation;
        }

        // ② 无条件剪枝：尾部已越过高水位线的尾音一律移除。
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

// ─── 响点表构建（命令线程；逐段镜像前端 buildTempoGridLines 语义）───────────

/// Tempo Map 分段（镜像前端 `tempoMapSegments` 的节奏相关字段）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetroSegment {
    /// 段起始时间（秒）= 变化点位置。
    pub(crate) start_sec: f64,
    /// 段结束时间（秒）= 下一变化点位置（末段 = 展开地平线）。
    pub(crate) end_sec: f64,
    /// 每拍秒数（60 / BPM，BPM 钳制 10..960）。
    pub(crate) sec_per_beat: f64,
    /// 每小节拍数（分子 × 4 ÷ 分母；允许小数，如 7/8 = 3.5）。
    pub(crate) beats_per_bar: f64,
}

/// 工程拍号 → 每小节拍数（分子 × 4 ÷ 分母，镜像前端 `beatsPerBarOf`）。
fn project_bar_beats(beats_per_bar: u32, denominator: u32) -> f64 {
    let num = if beats_per_bar >= 1 { beats_per_bar as f64 } else { 4.0 };
    let den = if denominator >= 1 { denominator as f64 } else { 4.0 };
    num * 4.0 / den
}

/// 由工程 BPM / Tempo Map 构建展开分段。
///
/// - 无 Tempo Map：单段（工程 BPM + 工程拍号），网格从工程 0 点全局对齐
///   （与 `buildTempoGridLines` 无 Map 分支一致）；
/// - 有 Tempo Map：**每个变化点一段**，拍号按 `effectiveTimeSignatures`
///   carry（null 跟随前点，种子 4/4）。`normalize_tempo_map` 已保证点升序、
///   去重、首点位于 0 且显式携带拍号；此处仍做防御性规范化。
pub(crate) fn build_tempo_segments(
    bpm: f64,
    tempo_map: Option<&[crate::state::TempoPointData]>,
    project_beats_per_bar: u32,
    project_denominator: u32,
    end_sec: f64,
) -> Vec<MetroSegment> {
    let end_sec = if end_sec.is_finite() && end_sec > 0.0 { end_sec } else { 0.0 };
    let fallback_spb = 60.0 / bpm.clamp(10.0, 960.0).max(1.0);
    let fallback_bpb = project_bar_beats(project_beats_per_bar, project_denominator);

    let Some(points) = tempo_map else {
        if end_sec <= 0.0 {
            return Vec::new();
        }
        return vec![MetroSegment {
            start_sec: 0.0,
            end_sec,
            sec_per_beat: fallback_spb,
            beats_per_bar: fallback_bpb,
        }];
    };

    // 防御性规范化（与 normalize_tempo_map / normalizeTempoMap 契约一致）：
    // 升序、去重（< 1e-6）、首点钉在 0。
    let mut points: Vec<crate::state::TempoPointData> = points.to_vec();
    points.sort_by(|a, b| {
        a.position_sec
            .partial_cmp(&b.position_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    points.dedup_by(|a, b| (a.position_sec - b.position_sec).abs() < 1e-6);
    if points.is_empty() {
        return Vec::new();
    }
    if points[0].position_sec > 1e-9 {
        // 首点不在 0：按 normalize 契约补一个工程基准点（BPM = 工程 BPM，
        // 拍号 4/4 —— 与 DEFAULT_TIME_SIGNATURE 的 carry 种子一致）。
        points.insert(
            0,
            crate::state::TempoPointData {
                id: String::new(),
                position_sec: 0.0,
                bpm: bpm.clamp(10.0, 960.0),
                numerator: Some(4),
                denominator: Some(4),
                scale: None,
            },
        );
    }

    // 逐段展开：拍号 carry（null 跟随前点，种子 4/4，镜像
    // `effectiveTimeSignatures` / `effective_time_signature_at`）。
    let mut segments: Vec<MetroSegment> = Vec::with_capacity(points.len());
    let mut carry: (u32, u32) = (4, 4);
    for (i, point) in points.iter().enumerate() {
        if let (Some(n), Some(d)) = (point.numerator, point.denominator) {
            carry = (n.clamp(1, 32), d);
        }
        let start_sec = point.position_sec.max(0.0);
        if start_sec > end_sec + 1e-9 {
            break;
        }
        let seg_end = match points.get(i + 1).map(|p| p.position_sec) {
            Some(next) if next.is_finite() => next.min(end_sec),
            _ => end_sec,
        };
        let seg_end = seg_end.max(start_sec);
        let point_bpm = if point.bpm.is_finite() { point.bpm.clamp(10.0, 960.0) } else { 120.0 };
        let bpb = carry.0 as f64 * 4.0 / carry.1.max(1) as f64;
        segments.push(MetroSegment {
            start_sec,
            end_sec: seg_end,
            sec_per_beat: 60.0 / point_bpm,
            beats_per_bar: bpb.max(1.0),
        });
        if start_sec >= end_sec {
            break;
        }
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

/// 由分段 + 步长展开响点表（升序，含重音标记）。
///
/// 逐段**局部对齐**，响点 = 时间标尺画出的网格线（镜像 `buildTempoGridLines`
/// 的 Tempo Map 分支，弱线与强线取并集）：
/// - 弱响点（`step_beats > 0`）：`段起点 + k×step×每拍秒数`，k = 0,1,2…
///   （变化点本身就是网格线）；奇数 k 施加 Swing 偏移
///   （`(swing/100) × 0.5 × step × 每拍秒数`，仅弱线，镜像 `swingAt`）；
/// - 强响点（小节线）：`段起点 + k×beatsPerBar×每拍秒数`，全部重音、
///   不受 Swing 影响。小节线**独立于弱网格**参与展开（如 7/8 段内 3.5 拍
///   处不在 1/4 网格上，但时间标尺同样画出它）——否则强拍会丢失；
/// - 弱线与段内小节线重合时同样标记重音（`k×step` 是 `beats_per_bar`
///   的整数倍）；同帧响点在末尾合并（段边界 = 前段末线 + 后段 k=0）。
pub(crate) fn build_click_schedule(
    segments: &[MetroSegment],
    step_beats: f64,
    sample_rate: u32,
    duration_sec: f64,
    swing_percent: f64,
) -> Vec<MetronomeClick> {
    let mut clicks: Vec<MetronomeClick> = Vec::new();
    if !duration_sec.is_finite() || duration_sec <= 0.0 || sample_rate == 0 {
        return clicks;
    }
    let sr = sample_rate as f64;
    let eps = 1e-9;
    let swing = swing_percent.clamp(0.0, 100.0) / 100.0;
    let bar_only = !(step_beats.is_finite() && step_beats > eps);
    let step = if bar_only { 0.0 } else { step_beats };

    for seg in segments {
        let seg_len = seg.end_sec - seg.start_sec;
        if seg_len <= eps {
            continue;
        }
        let bpb = seg.beats_per_bar.max(1.0);
        // 弱网格线（bar_only 模式跳过）。
        if !bar_only {
            let line_sec = step * seg.sec_per_beat;
            let line_count = (seg_len / line_sec + eps).floor().max(0.0) as u64;
            for k in 0..=line_count {
                // Swing：仅弱网格线的奇数格（镜像前端 swingAt：k % 2 == 1）。
                let swing_sec = if k % 2 == 1 {
                    swing * 0.5 * step * seg.sec_per_beat
                } else {
                    0.0
                };
                let t = seg.start_sec + k as f64 * line_sec + swing_sec;
                if t > duration_sec + 1e-6 {
                    break;
                }
                // 重音 = 该弱线与段内小节线重合（k×step 是 bpb 的整数倍）。
                let r = (k as f64 * step) % bpb;
                let accent = r < 1e-4 || (bpb - r).abs() < 1e-4;
                let frame = (t * sr).round().max(0.0) as u64;
                clicks.push(MetronomeClick { frame, accent });
            }
        }
        // 强网格线（小节线，全部重音；永不偏移）。
        let bar_sec = bpb * seg.sec_per_beat;
        let bar_count = (seg_len / bar_sec + eps).floor().max(0.0) as u64;
        for k in 0..=bar_count {
            let t = seg.start_sec + k as f64 * bar_sec;
            if t > duration_sec + 1e-6 {
                break;
            }
            let frame = (t * sr).round().max(0.0) as u64;
            clicks.push(MetronomeClick { frame, accent: true });
        }
        if clicks.len() >= 2_000_000 {
            break;
        }
    }

    // 段边界（下一变化点位置）同时是前段末线与后段 k=0 小节线：同帧合并，
    // 重音取并集（变化点本身总是小节线，与 beatToBarBeat 一致）。
    clicks.sort_by_key(|c| c.frame);
    let mut merged: Vec<MetronomeClick> = Vec::with_capacity(clicks.len());
    for click in clicks {
        match merged.last_mut() {
            Some(last) if last.frame == click.frame => last.accent |= click.accent,
            _ => merged.push(click),
        }
    }
    merged
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
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);
        let clicks = build_click_schedule(&segs, 1.0, 44100, 3.0, 0.0);
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
        // 1/8 网格：每半拍一响；4/4 下重音仅与小节首重合。
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);
        let clicks = build_click_schedule(&segs, 0.5, 44100, 4.0, 0.0);
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
        // 1/4t（每 2/3 拍一响）+ 4/4：5s 覆盖三个小节 → 第 0、6、12 响为重音。
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);
        let clicks = build_click_schedule(&segs, 2.0 / 3.0, 44100, 5.0, 0.0);
        let accents: Vec<usize> = clicks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.accent)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(accents, vec![0, 6, 12]);
    }

    #[test]
    fn tempo_map_integration_and_time_signature_change_realigns_bars() {
        // 0-2s: 120 BPM（4/4）；2s 起 60 BPM（3/4）。
        // 段起点（2s）本身是小节线（3/4 域锚）。
        let map = vec![
            point(0.0, 120.0, Some(4), Some(4)),
            point(2.0, 60.0, Some(3), Some(4)),
        ];
        let segs = build_tempo_segments(120.0, Some(&map), 4, 4, 6.0);
        assert_eq!(segs.len(), 2);
        assert!((segs[1].start_sec - 2.0).abs() < 1e-9);
        assert!((segs[1].sec_per_beat - 1.0).abs() < 1e-9);
        let clicks = build_click_schedule(&segs, 1.0, 1000, 6.0, 0.0);
        let accent_times: Vec<f64> = clicks
            .iter()
            .filter(|c| c.accent)
            .map(|c| c.frame as f64 / 1000.0)
            .collect();
        // 4/4 域重音拍 0（0s）；3/4 域锚 2s → 重音 2s、5s。
        assert_eq!(accent_times, vec![0.0, 2.0, 5.0]);
    }

    #[test]
    fn bpm_only_change_point_realigns_clicks_and_is_a_bar_line() {
        // 仅变速、拍号不变的变化点同样**逐段局部重对齐**：
        // - 变化点 1.3s 本身就是一条小节线（段起点）→ 重音；
        // - 之后网格从 1.3s 起按新 BPM 局部等距，而不是继续全局拍域格点。
        let map = vec![
            point(0.0, 120.0, Some(4), Some(4)),
            point(1.3, 60.0, Some(4), Some(4)),
        ];
        let segs = build_tempo_segments(120.0, Some(&map), 4, 4, 6.0);
        assert_eq!(segs.len(), 2);
        let clicks = build_click_schedule(&segs, 1.0, 1000, 4.0, 0.0);
        let times: Vec<f64> = clicks.iter().map(|c| c.frame as f64 / 1000.0).collect();
        // 段 1：0、0.5、1.0；段 2：从 1.3s 起每秒一响（局部对齐）。
        assert_eq!(times, vec![0.0, 0.5, 1.0, 1.3, 2.3, 3.3]);
        let accent_times: Vec<f64> = clicks
            .iter()
            .filter(|c| c.accent)
            .map(|c| c.frame as f64 / 1000.0)
            .collect();
        // 重音：0s（段1 拍0）、1.3s（段起点 = 小节线；段2 下一小节线在
        // 1.3 + 4×1.0 = 5.3s，超出本例时长）。
        assert_eq!(accent_times, vec![0.0, 1.3]);
    }

    #[test]
    fn time_signature_carry_from_previous_point() {
        // 7/8（bpb 3.5）之后的点拍号为 null → 跟随前点（carry）。
        let map = vec![
            point(0.0, 120.0, Some(7), Some(8)),
            point(1.0, 120.0, None, None),
        ];
        let segs = build_tempo_segments(120.0, Some(&map), 4, 4, 5.0);
        assert!((segs[1].beats_per_bar - 3.5).abs() < 1e-9);
        // 1/8 网格（0.5 拍）下与 3.5 拍小节线重合的格点：
        // 0s、1.0s（段起点）、2.75s（段2 第二小节线：1.0 + 3.5×0.5）。
        let clicks = build_click_schedule(&segs, 0.5, 1000, 4.0, 0.0);
        let accent_times: Vec<f64> = clicks
            .iter()
            .filter(|c| c.accent)
            .map(|c| c.frame as f64 / 1000.0)
            .collect();
        assert_eq!(accent_times, vec![0.0, 1.0, 2.75]);
    }

    #[test]
    fn swing_offsets_weak_lines_only() {
        // Swing 50%：弱网格奇数格右移 (50/100)×0.5×step×spb = 125ms
        // （120 BPM、step 1 拍）；小节线（重音）永不偏移。
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);
        let clicks = build_click_schedule(&segs, 1.0, 44100, 2.0, 50.0);
        let frames: Vec<i64> = clicks.iter().map(|c| c.frame as i64).collect();
        // k=0: 0；k=1: (0.5+0.125)×44100 = 27562.5 → 27563；
        // k=2: 44100；k=3: 1.625×44100 = 71662.5 → 71663；k=4: 88200（重音）。
        assert_eq!(frames[0], 0);
        assert_eq!(frames[1], 27563);
        assert_eq!(frames[2], 44100);
        assert_eq!(frames[3], 71663);
        assert_eq!(frames[4], 88200);
        assert!(clicks[0].accent);
        assert!(clicks[4].accent);
        assert!(!clicks[1].accent);
    }

    #[test]
    fn bar_only_mode_clicks_on_bar_starts() {
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);
        let clicks = build_click_schedule(&segs, 0.0, 44100, 3.0, 0.0);
        assert_eq!(clicks.len(), 2); // 0s、2s
        assert!(clicks.iter().all(|c| c.accent));
    }

    #[test]
    fn bar_only_mode_realigns_at_every_change_point() {
        // 仅小节首模式：BPM-only 变化点（1.3s）同样是一条小节线。
        let map = vec![
            point(0.0, 120.0, Some(4), Some(4)),
            point(1.3, 60.0, Some(4), Some(4)),
        ];
        let segs = build_tempo_segments(120.0, Some(&map), 4, 4, 6.0);
        let clicks = build_click_schedule(&segs, 0.0, 1000, 6.0, 0.0);
        let times: Vec<f64> = clicks.iter().map(|c| c.frame as f64 / 1000.0).collect();
        // 段1 长 1.3s、不足一个 2s 小节 → 只有段起点 0s；段2 小节线从
        // 1.3s 起每 4s 一条：1.3、5.3（重对齐后不再落在全局 4.0/6.0 格点）。
        assert_eq!(times, vec![0.0, 1.3, 5.3]);
        assert!(clicks.iter().all(|c| c.accent));
    }

    #[test]
    fn bar_lines_off_the_weak_grid_are_still_clicked() {
        // 工程拍号 7/8（bpb = 3.5）无 Tempo Map：小节锚 0，3.5 拍/小节。
        // 1/4 网格只落在整数拍上；1.75s（3.5 拍）不在其上，但它是时间
        // 标尺画出的小节线 → 节拍器必须在该处发声并标记重音。
        let segs = build_tempo_segments(120.0, None, 7, 8, 6.0);
        assert!((segs[0].beats_per_bar - 3.5).abs() < 1e-9);
        let clicks = build_click_schedule(&segs, 1.0, 1000, 4.0, 0.0);
        let accent_frames: Vec<u64> = clicks
            .iter()
            .filter(|c| c.accent)
            .map(|c| c.frame)
            .collect();
        // 小节线：0、1.75s、3.5s。
        assert_eq!(accent_frames, vec![0, 1750, 3500]);
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
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);

        // 表 A：120 BPM 4/4，每拍一响（0.5s = 22050 帧 @44.1kHz）。
        metro.store_schedule(Arc::new(build_click_schedule(&segs, 1.0, 44100, 60.0, 0.0)));
        let gen_a = metro.generation();

        // 播放推进到 4000 帧：块 [0,1024) 起振 frame 0 的 click（尾音到 ~4876）。
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
        metro.store_schedule(Arc::new(build_click_schedule(&segs, 1.0, 44100, 60.0, 0.0)));
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
        let segs = build_tempo_segments(120.0, None, 4, 4, 6.0);
        metro.store_schedule(Arc::new(build_click_schedule(&segs, 1.0, 44100, 60.0, 0.0)));

        let mut scratch = vec![0.0f32; 4096];
        // 块 [21000, 25000)：起振 22050 的 click。
        voices.mix(&mut scratch, &metro, 21000, 25000, 44100);
        assert_eq!(voices.active.len(), 1);

        // 回跳重播同一区间（位置不前进）：不得再起振第二个 voice。
        voices.mix(&mut scratch, &metro, 21000, 25000, 44100);
        assert_eq!(
            voices.active.len(),
            1,
            "same click frame must not spawn a second overlapping voice"
        );
    }
}
