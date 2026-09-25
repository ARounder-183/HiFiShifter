//! 导入声道策略 → Take `channel_mode` 的统一入口。
//!
//! # 生效边界（改动前请先读）
//!
//! 判定规则只有一条：**该 Take 的 `channel_mode` 是否已有权威来源**。
//! 有权威来源就尊重，没有才走策略。
//!
//! | 路径 | 是否走策略 | 理由 |
//! | --- | --- | --- |
//! | 媒体导入（`import_audio_item` / `import_media_files_as_takes` / `add_clip_take_from_media`） | 是 | 文件不携带声道语义 |
//! | `add_clip`（粘贴 / 新建 / 重复） | 是 | 同上 |
//! | VocalShifter 工程 / 剪贴板导入 | 是 | VS 格式无声道字段 |
//! | v4 及更早工程升级 | 是 | Take 无 `channel_mode` 字段 |
//! | **REAPER 导入 / 剪贴板 / 导出** | **否** | REAPER 自带 `CHANMODE` 权威字段 |
//! | v5+ 工程加载 | 否 | 已持久化的用户决定 |
//!
//! # 锁外约定
//!
//! [`precompute_decision`] **会解码音频**，调用方必须保证不在持有 timeline
//! 全局锁时调用（慢盘/网络盘上可能耗时数百毫秒，而该锁是所有命令与 UI
//! 轮询的串行点）。锁内只允许调用零成本的 [`apply_decision`]。

use std::path::Path;

use crate::config::ChannelImportPolicy;
use crate::state::ClipTake;
use crate::stereo_detect::{self, ChannelVerdict};

/// 锁外判定的产物：随后在锁内零成本应用。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelDecision {
    /// 保持原样（策略关闭 / 单声道源 / 真立体声 / 判定失败 / 无可读源）。
    Keep,
    /// 折叠为单声道（携带目标模式 2..=4）。
    FoldToMono(i32),
}

impl ChannelDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelDecision::Keep => "keep",
            ChannelDecision::FoldToMono(_) => "foldToMono",
        }
    }

    pub fn target_mode(self) -> Option<i32> {
        match self {
            ChannelDecision::Keep => None,
            ChannelDecision::FoldToMono(mode) => Some(mode),
        }
    }
}

/// 判定结果（含可展示的原因）。
///
/// 与 [`ChannelDecision`] 分开是因为两者回答的问题不同：本枚举回答"为什么"，
/// 用于扫描报告与日志；`ChannelDecision` 回答"做什么"，用于写回 Take。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelScanOutcome {
    /// 策略关闭：未做任何判定。
    PolicyOff,
    /// 单声道源（`channels < 2`）：本就单声道，无需折叠。
    MonoSource,
    /// 策略为"全部转换"：不判定内容，直接折叠。
    ForcedMono,
    /// 假立体声（L/R 在容差内一致）：折叠。
    FakeStereo,
    /// 真立体声：保持。
    TrueStereo,
    /// 无法判定（源缺失 / 不可解码 / 无有效采样）：保持 —— 宁可不优化，
    /// 也不能把无法确认的素材误折叠。
    Unknown,
}

impl ChannelScanOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelScanOutcome::PolicyOff => "policyOff",
            ChannelScanOutcome::MonoSource => "mono",
            ChannelScanOutcome::ForcedMono => "forcedMono",
            ChannelScanOutcome::FakeStereo => "fakeStereo",
            ChannelScanOutcome::TrueStereo => "trueStereo",
            ChannelScanOutcome::Unknown => "unknown",
        }
    }

    /// 由判定结果导出可执行的决定。
    ///
    /// 先看策略总开关：策略关闭时任何判定结果都不折叠（`scan_source` 本就不会
    /// 在关闭时给出 `FakeStereo`，但本函数不该依赖调用方顺序才正确）。
    pub fn decision(self, policy: &ChannelImportPolicy) -> ChannelDecision {
        let policy = policy.normalized();
        if policy.is_off() {
            return ChannelDecision::Keep;
        }
        match self {
            ChannelScanOutcome::ForcedMono | ChannelScanOutcome::FakeStereo => {
                ChannelDecision::FoldToMono(policy.mono_target_mode)
            }
            ChannelScanOutcome::PolicyOff
            | ChannelScanOutcome::MonoSource
            | ChannelScanOutcome::TrueStereo
            | ChannelScanOutcome::Unknown => ChannelDecision::Keep,
        }
    }
}

/// 锁外：按策略判定一个源文件。
///
/// `source_channels` 为已探测到的源声道数（`None` / `0` 时本函数自行做一次
/// O(1) 的 header 探测）。`region` 是源域秒区间，判定只对该区间负责；
/// `None` 表示整个文件。
pub fn scan_source(
    source_path: Option<&Path>,
    source_channels: Option<u16>,
    region: Option<(f64, f64)>,
    policy: &ChannelImportPolicy,
) -> ChannelScanOutcome {
    let policy = policy.normalized();
    if policy.is_off() {
        return ChannelScanOutcome::PolicyOff;
    }

    // 声道数未知时探测一次（WAV 读头 / 容器探测，均为 O(1)）。
    // 探测失败按单声道处理 —— "不确定"时不转换。
    let channels = match source_channels {
        Some(c) if c > 0 => c,
        _ => match source_path.and_then(crate::audio_utils::try_read_audio_header_only) {
            Some(info) if info.channels > 0 => info.channels,
            _ => 1,
        },
    };
    if channels < 2 {
        return ChannelScanOutcome::MonoSource;
    }

    if !policy.is_smart() {
        // alwaysMono：无需解码，直接折叠。
        return ChannelScanOutcome::ForcedMono;
    }

    let Some(path) = source_path else {
        return ChannelScanOutcome::Unknown;
    };
    match stereo_detect::verdict_for_file(path, region, &policy.detect_options()) {
        ChannelVerdict::FakeStereo => ChannelScanOutcome::FakeStereo,
        ChannelVerdict::Mono => ChannelScanOutcome::MonoSource,
        ChannelVerdict::TrueStereo => ChannelScanOutcome::TrueStereo,
        ChannelVerdict::Unknown => ChannelScanOutcome::Unknown,
    }
}

/// 锁外：按策略为一个源文件决定是否折叠为单声道。
pub fn precompute_decision(
    source_path: Option<&Path>,
    source_channels: Option<u16>,
    region: Option<(f64, f64)>,
    policy: &ChannelImportPolicy,
) -> ChannelDecision {
    scan_source(source_path, source_channels, region, policy).decision(policy)
}

/// 锁内：把一个已算好的决定应用到 Take；返回是否发生了改变。
///
/// 零解码、零 IO，可安全在持锁状态下调用。
pub fn apply_decision(take: &mut ClipTake, decision: ChannelDecision) -> bool {
    match decision {
        ChannelDecision::Keep => false,
        ChannelDecision::FoldToMono(mode) => {
            let mode = crate::channel_mode::TakeChannelMode::from_raw(mode).raw();
            if take.channel_mode == mode {
                return false;
            }
            take.channel_mode = mode;
            true
        }
    }
}

/// 为一个 Take 锁外预判定（区间取自该 Take 的消费窗口）。
pub fn precompute_take_decision(
    take: &ClipTake,
    policy: &ChannelImportPolicy,
) -> ChannelDecision {
    scan_take(take, policy).decision(policy)
}

/// 为一个 Take 锁外判定并返回可展示的原因（区间取自其消费窗口）。
pub fn scan_take(take: &ClipTake, policy: &ChannelImportPolicy) -> ChannelScanOutcome {
    scan_source(
        take.source_path.as_deref().map(Path::new),
        take.source_channels,
        take_consumption_region(take),
        policy,
    )
}

/// Take 的源域消费区间；无效时返回 `None`（= 整文件）。
///
/// 判定只覆盖实际会被渲染/听到的那一段：同一文件被切成多个 Take 时，
/// 某个 Take 可能只消费到"尚未分叉"的前半段，那一段折叠为单声道是无损的。
pub fn take_consumption_region(take: &ClipTake) -> Option<(f64, f64)> {
    let start = take.source_start_sec;
    let end = take.source_end_sec;
    if start.is_finite() && end.is_finite() && end > start {
        Some((start, end))
    } else {
        None
    }
}

/// 对一个 Clip 的全部 Take 应用（**会解码**，必须在锁外调用）。
///
/// 返回发生改变的数量。用于 VocalShifter 导入等"整批 Take 都无权威声道
/// 信息"的场景。
pub fn resolve_clip_takes_channel_mode(
    clip: &mut crate::state::Clip,
    policy: &ChannelImportPolicy,
) -> usize {
    let decisions: Vec<ChannelDecision> = clip
        .takes
        .iter()
        .map(|take| precompute_take_decision(take, policy))
        .collect();
    let mut changed = 0usize;
    for (take, decision) in clip.takes.iter_mut().zip(decisions) {
        if apply_decision(take, decision) {
            changed += 1;
        }
    }
    changed
}

/// 对一批**刚导入、尚无声道权威信息**的 Clip 套用当前导入策略。
///
/// **会解码音频**，必须在锁外调用。返回被改写的 Take 总数。
///
/// 供 VocalShifter 工程 / 剪贴板导入使用。REAPER 导入自带 `CHANMODE`
/// 权威字段，**不得**调用本函数。
pub fn apply_policy_to_clips(clips: &mut [crate::state::Clip]) -> usize {
    let policy = crate::config::channel_import_policy();
    if policy.is_off() {
        return 0;
    }
    let mut changed = 0usize;
    for clip in clips.iter_mut() {
        changed += resolve_clip_takes_channel_mode(clip, &policy);
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ChannelImportPolicy;

    /// 测试便捷：走一遍生产的"锁外判定 → 锁内应用"两步。
    fn resolve(take: &mut ClipTake, policy: &ChannelImportPolicy) -> bool {
        let decision = precompute_take_decision(take, policy);
        apply_decision(take, decision)
    }

    /// 造一个空白 Take（走 add_clip 的正式构造路径，避免手写字面量漂移）。
    fn blank_take() -> ClipTake {
        let mut tl = crate::state::TimelineState::default();
        let track = tl.tracks.first().map(|t| t.id.clone());
        let id = tl.add_clip(track, Some("t".into()), Some(0.0), Some(1.0), None);
        let clip = tl.clips.iter_mut().find(|c| c.id == id).expect("clip");
        clip.sync_take_from_flat();
        clip.takes[0].clone()
    }

    fn take_with(path: Option<&str>, channels: Option<u16>) -> ClipTake {
        let mut take = blank_take();
        take.source_path = path.map(|p| p.to_string());
        take.source_channels = channels;
        take.source_start_sec = 0.0;
        take.source_end_sec = 10.0;
        take
    }

    #[test]
    fn off_mode_never_converts() {
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        let policy = ChannelImportPolicy {
            mode: "off".into(),
            ..Default::default()
        };
        assert!(!resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 0);
    }

    #[test]
    fn always_mono_converts_without_decoding() {
        // 源路径不存在：alwaysMono 不需要解码，仍必须折叠。
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        let policy = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            ..Default::default()
        };
        assert!(resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 2);
    }

    #[test]
    fn always_mono_honors_target_mode() {
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        let policy = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            mono_target_mode: 3,
            ..Default::default()
        };
        assert!(resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 3);
    }

    #[test]
    fn mono_source_is_left_alone_in_smart_mode() {
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(1));
        let policy = ChannelImportPolicy::default(); // smart
        assert!(!resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 0);
    }

    #[test]
    fn missing_file_yields_no_conversion_in_smart_mode() {
        // Unknown（文件不可读）→ 不动，避免把无法判定的素材误折叠。
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        let policy = ChannelImportPolicy::default();
        assert!(!resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 0);
    }

    #[test]
    fn no_source_path_is_left_alone() {
        let mut take = take_with(None, Some(2));
        let policy = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            ..Default::default()
        };
        // alwaysMono 仍会折叠：它不依赖源文件内容。
        assert!(resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 2);
    }

    #[test]
    fn smart_mode_folds_a_real_fake_stereo_file() {
        // 构造一个 L == R 的临时 WAV，走完整判定链路。
        let dir = std::env::temp_dir().join("hifishifter_channel_policy_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fake.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..44_100 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();

        let mut take = take_with(Some(path.to_str().unwrap()), Some(2));
        let policy = ChannelImportPolicy::default();
        assert!(resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn smart_mode_keeps_a_true_stereo_file() {
        let dir = std::env::temp_dir().join("hifishifter_channel_policy_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("true.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..44_100 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(-v).unwrap();
        }
        w.finalize().unwrap();

        let mut take = take_with(Some(path.to_str().unwrap()), Some(2));
        let policy = ChannelImportPolicy::default();
        assert!(!resolve(&mut take, &policy));
        assert_eq!(take.channel_mode, 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_decision_is_idempotent() {
        let mut take = take_with(None, Some(2));
        assert!(apply_decision(&mut take, ChannelDecision::FoldToMono(2)));
        assert_eq!(take.channel_mode, 2);
        // 已经是目标模式 → 第二次不再报告改变。
        assert!(!apply_decision(&mut take, ChannelDecision::FoldToMono(2)));
    }

    #[test]
    fn apply_decision_keep_is_a_noop() {
        let mut take = take_with(None, Some(2));
        take.channel_mode = 1;
        assert!(!apply_decision(&mut take, ChannelDecision::Keep));
        assert_eq!(take.channel_mode, 1);
    }

    #[test]
    fn apply_decision_normalizes_out_of_range_mode() {
        // 越界目标模式经 from_raw 回落 Normal(0)；绝不能让非法值写进 Take。
        let mut take = take_with(None, Some(2));
        take.channel_mode = 1;
        assert!(apply_decision(&mut take, ChannelDecision::FoldToMono(99)));
        assert_eq!(take.channel_mode, 0, "越界模式回落 Normal");

        // 目标已等于当前值时不报告改变（幂等）。
        assert!(!apply_decision(&mut take, ChannelDecision::FoldToMono(99)));
    }

    #[test]
    fn consumption_region_ignores_degenerate_windows() {
        let mut take = take_with(None, Some(2));
        take.source_start_sec = 0.0;
        take.source_end_sec = 0.0;
        assert_eq!(take_consumption_region(&take), None, "零长度窗口按整文件");
        take.source_start_sec = 5.0;
        take.source_end_sec = 2.0;
        assert_eq!(take_consumption_region(&take), None, "反向窗口按整文件");
        take.source_end_sec = 6.0;
        assert_eq!(take_consumption_region(&take), Some((5.0, 6.0)));
    }

    #[test]
    fn resolve_clip_takes_counts_changes() {
        let mut tl = crate::state::TimelineState::default();
        let track = tl.tracks.first().map(|t| t.id.clone());
        let id = tl.add_clip(track, Some("t".into()), Some(0.0), Some(1.0), None);
        let clip = tl.clips.iter_mut().find(|c| c.id == id).expect("clip");
        clip.sync_take_from_flat();

        let mut a = take_with(None, Some(2));
        a.id = "a".into();
        let mut b = take_with(None, Some(1));
        b.id = "b".into();
        clip.takes = vec![a, b];
        clip.active_take_id = Some("a".into());

        let policy = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            ..Default::default()
        };
        // 只有双声道源的那条被折叠。
        assert_eq!(resolve_clip_takes_channel_mode(clip, &policy), 1);
        assert_eq!(clip.takes[0].channel_mode, 2);
        assert_eq!(clip.takes[1].channel_mode, 0);
    }

    #[test]
    fn scan_outcome_reports_the_reason_and_derives_the_decision() {
        let smart = ChannelImportPolicy::default();
        let forced = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            mono_target_mode: 4,
            ..Default::default()
        };

        // 只有"假立体声"与"强制转换"会折叠。
        assert_eq!(
            ChannelScanOutcome::FakeStereo.decision(&smart),
            ChannelDecision::FoldToMono(2)
        );
        assert_eq!(
            ChannelScanOutcome::ForcedMono.decision(&forced),
            ChannelDecision::FoldToMono(4),
            "目标模式取自策略"
        );
        for keep in [
            ChannelScanOutcome::PolicyOff,
            ChannelScanOutcome::MonoSource,
            ChannelScanOutcome::TrueStereo,
            ChannelScanOutcome::Unknown,
        ] {
            assert_eq!(keep.decision(&smart), ChannelDecision::Keep, "{keep:?}");
        }
        // 即便判定结果是"假立体声"，策略关闭时也不得折叠。
        assert_eq!(
            ChannelScanOutcome::FakeStereo.decision(&ChannelImportPolicy {
                mode: "off".into(),
                ..Default::default()
            }),
            ChannelDecision::Keep
        );
        // 原因字符串是 IPC 契约的一部分。
        assert_eq!(ChannelScanOutcome::PolicyOff.as_str(), "policyOff");
        assert_eq!(ChannelScanOutcome::FakeStereo.as_str(), "fakeStereo");
        assert_eq!(ChannelScanOutcome::TrueStereo.as_str(), "trueStereo");
        assert_eq!(ChannelScanOutcome::MonoSource.as_str(), "mono");
        assert_eq!(ChannelScanOutcome::ForcedMono.as_str(), "forcedMono");
        assert_eq!(ChannelScanOutcome::Unknown.as_str(), "unknown");
    }

    #[test]
    fn scan_source_distinguishes_the_reasons() {
        // 策略关闭：不做判定。
        assert_eq!(
            scan_source(None, Some(2), None, &ChannelImportPolicy { mode: "off".into(), ..Default::default() }),
            ChannelScanOutcome::PolicyOff
        );
        // 单声道源：无需判定。
        assert_eq!(
            scan_source(None, Some(1), None, &ChannelImportPolicy::default()),
            ChannelScanOutcome::MonoSource
        );
        // 强制转换：不判定内容。
        assert_eq!(
            scan_source(None, Some(2), None, &ChannelImportPolicy { mode: "alwaysMono".into(), ..Default::default() }),
            ChannelScanOutcome::ForcedMono
        );
        // 智能模式但源不可读 → Unknown（不折叠）。
        assert_eq!(
            scan_source(
                Some(std::path::Path::new("C:/definitely/missing.wav")),
                Some(2),
                None,
                &ChannelImportPolicy::default()
            ),
            ChannelScanOutcome::Unknown
        );
    }

    #[test]
    fn scan_source_classifies_real_files() {
        let dir = std::env::temp_dir().join("hifishifter_channel_policy_scan");
        std::fs::create_dir_all(&dir).unwrap();
        let policy = ChannelImportPolicy::default();

        let fake = dir.join("scan_fake.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&fake, spec).unwrap();
        for i in 0..44_100 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        assert_eq!(
            scan_source(Some(&fake), Some(2), None, &policy),
            ChannelScanOutcome::FakeStereo
        );

        let real = dir.join("scan_true.wav");
        let mut w = hound::WavWriter::create(&real, spec).unwrap();
        for i in 0..44_100 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(-v).unwrap();
        }
        w.finalize().unwrap();
        assert_eq!(
            scan_source(Some(&real), Some(2), None, &policy),
            ChannelScanOutcome::TrueStereo
        );

        let _ = std::fs::remove_file(&fake);
        let _ = std::fs::remove_file(&real);
    }
}
