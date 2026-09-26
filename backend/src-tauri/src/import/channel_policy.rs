//! 导入声道策略 → Take `channel_mode` 的统一入口。
//!
//! # 生效边界（改动前请先读）
//!
//! 判定规则只有一条：**该 Take 的 `channel_mode` 是否已有权威来源**。
//! 权威来源由 [`crate::channel_decision::ChannelDecisionRecord`] 显式记录：
//! 用户显式设置过（[`crate::channel_decision::ORIGIN_USER`]）就尊重，没有
//! 权威档案就走策略。
//!
//! | 路径 | 是否走策略 | 理由 |
//! | --- | --- | --- |
//! | 媒体导入（`import_audio_item` / `import_media_files_as_takes` / `add_clip_take_from_media`） | 是 | 文件不携带声道语义 |
//! | `add_clip`（粘贴 / 新建 / 重复） | 是 | 同上 |
//! | VocalShifter 工程 / 剪贴板导入 | 是 | VS 格式无声道字段 |
//! | v4 及更早工程升级 | 是 | Take 无判定档案 |
//! | **REAPER 导入 / 剪贴板 / 导出** | **否** | REAPER 自带 `CHANMODE` 权威字段 |
//! | 已定论且上下文未变的 Take | 否 | 档案权威 ⇒ 零解码直接采用 |
//!
//! # 为什么不再用工程版本号
//!
//! 旧实现以 `project_file_version < 5` 判断"这个 Take 有没有权威来源"。那是
//! **一次性**的：打开瞬间没判成的 Take 停在 `channel_mode = 0`，工程一保存成
//! v5 就再没有第二次机会，而 0 会被当成"用户显式决定"永久尊重。于是"第一次
//! 打开时漏判的，永远漏"。
//!
//! 现在判定结论连同**上下文**（内容指纹 / 策略签名 / 消费区间）一起落进
//! [`crate::channel_decision::ChannelDecisionRecord`]，因此判定是幂等、可恢复、
//! 可重跑的：读不到的 Take 记 [`ChannelScanOutcome::Pending`]，下次打开自动重试。
//!
//! # 锁外约定
//!
//! [`precompute_decision`] / [`precompute_take_decision`] **会解码音频**，调用方
//! 必须保证不在持有 timeline 全局锁时调用（慢盘/网络盘上可能耗时数百毫秒，而
//! 该锁是所有命令与 UI 轮询的串行点）。锁内只允许调用零成本的
//! [`apply_resolution`]。

use std::path::Path;

use crate::channel_decision::{
    ChannelDecisionRecord, VERDICT_FAKE_STEREO, VERDICT_FORCED_MONO, VERDICT_MONO,
    VERDICT_TRUE_STEREO,
};
use crate::config::ChannelImportPolicy;
use crate::state::ClipTake;
use crate::stereo_detect::{self, ChannelVerdict};

/// 一次判定的**上下文快照**：与判定档案里三个可失效字段一一对应。
///
/// 「结论还有效吗」只由这三项决定 —— 源文件被换掉（指纹）、用户改了抽样策略
/// （签名）、Take 的消费区间被 trim 改变（区间）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionContext {
    /// 源文件内容指纹（取 Take 上持久化的那份，跨会话稳定）。
    pub fingerprint: Option<u64>,
    /// 判定时的策略签名。
    pub policy_sig: u64,
    /// 判定的源域区间（毫秒量化）；`None` = 整个文件。
    pub region_q: Option<(i64, i64)>,
}

impl DecisionContext {
    /// 由「Take 的持久化指纹 + 消费区间 + 判定参数」构造。
    ///
    /// 签名取自 `opts` 而非 policy：容器解码预算也是签名的一部分，短预算
    /// （导入路径）与长预算（后台扫描）必须产出不同的上下文，否则短预算的
    /// 结论会被当成权威结论而跳过重判。
    pub fn for_take(
        take: &ClipTake,
        opts: &stereo_detect::DetectOptions,
    ) -> Self {
        Self {
            fingerprint: take.source_file_fingerprint,
            policy_sig: opts.signature(),
            region_q: stereo_detect::quantize_region(take_consumption_region(take)),
        }
    }

    /// 由显式给出的区间构造（导入路径：区间即整文件）。
    pub fn for_region(
        fingerprint: Option<u64>,
        region: Option<(f64, f64)>,
        opts: &stereo_detect::DetectOptions,
    ) -> Self {
        Self {
            fingerprint,
            policy_sig: opts.signature(),
            region_q: stereo_detect::quantize_region(region),
        }
    }
}

/// 一次判定在 Take 上的**完整落点**：写什么模式 + 记什么档案。
///
/// 取代原先只表达"写什么模式"的 `ChannelDecision`：判定档案是「漏判不再永久化」
/// 的载体，必须与模式一起落盘，否则读不到的 Take 会被当成"已定论"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelResolution {
    /// 要写入的声道模式；`None` = 不改写（保持既有 / 用户的选择）。
    pub mode: Option<i32>,
    /// 要写入的判定档案；`None` = 什么都不记（策略关闭 / 无源 Take）。
    pub record: Option<ChannelDecisionRecord>,
}

impl ChannelResolution {
    /// 只记账、不改模式（已定论但无需动作的结论）。
    pub fn record_only(record: ChannelDecisionRecord) -> Self {
        Self {
            mode: None,
            record: Some(record),
        }
    }

    /// 什么都不做（策略关闭 / 无源 Take）。
    pub const NOOP: Self = Self {
        mode: None,
        record: None,
    };

    /// 该落点是否会**改变听感**（模式变了才需要失效渲染缓存）。
    pub fn changes_mode(self) -> bool {
        self.mode.is_some()
    }
}

/// 判定结果（含可展示的原因）。
///
/// 与 [`ChannelResolution`] 分开是因为两者回答的问题不同：本枚举回答"为什么"，
/// 用于扫描报告与日志；`ChannelResolution` 回答"做什么"，用于写回 Take。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelScanOutcome {
    /// 策略关闭：未做任何判定。
    PolicyOff,
    /// 该 Take 没有音频源（MIDI / 空白 Clip）：不属于声道折叠的适用范围。
    NoSource,
    /// 单声道源（`channels < 2`）：本就单声道，无需折叠。
    MonoSource,
    /// 策略为"全部转换"：不判定内容，直接折叠。
    ForcedMono,
    /// 假立体声（L/R 在容差内一致）：折叠。
    FakeStereo,
    /// 真立体声：保持。
    TrueStereo,
    /// **本次读不到**（源缺失 / 不可解码 / 探测不出声道数 / 覆盖不完整）：
    /// 保持，并记下"待重试"。
    ///
    /// 与 [`ChannelScanOutcome::MonoSource`] 严格区分 —— 把"读不了"记成
    /// "单声道"会让漏判永久静默（报告里显示成"无事可做"而不是"我没读到"）。
    Pending,
}

impl ChannelScanOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelScanOutcome::PolicyOff => "policyOff",
            ChannelScanOutcome::NoSource => "noSource",
            ChannelScanOutcome::MonoSource => "mono",
            ChannelScanOutcome::ForcedMono => "forcedMono",
            ChannelScanOutcome::FakeStereo => "fakeStereo",
            ChannelScanOutcome::TrueStereo => "trueStereo",
            ChannelScanOutcome::Pending => "pending",
        }
    }

    /// 是否值得在报告里提醒用户"这次没读到"。
    pub fn is_pending(self) -> bool {
        self == ChannelScanOutcome::Pending
    }

    /// 由判定结果导出可执行落点。
    ///
    /// 先看策略总开关：策略关闭时任何判定结果都不折叠（`scan_source` 本就不会
    /// 在关闭时给出 `FakeStereo`，但本函数不该依赖调用方顺序才正确）。
    pub fn resolution(
        self,
        policy: &ChannelImportPolicy,
        ctx: DecisionContext,
    ) -> ChannelResolution {
        let policy = policy.normalized();
        if policy.is_off() {
            // 策略关闭：不折叠，也**不记档案** —— 用户关掉了这个功能，我们不该
            // 在他的工程里留下"我们看过了"的痕迹，否则将来重新打开功能时，
            // 这些 Take 会被当成"已定论"而跳过。
            return ChannelResolution::NOOP;
        }
        let target = policy.mono_target_mode;
        match self {
            ChannelScanOutcome::PolicyOff | ChannelScanOutcome::NoSource => ChannelResolution::NOOP,
            ChannelScanOutcome::MonoSource => ChannelResolution::record_only(
                ChannelDecisionRecord::auto(VERDICT_MONO, ctx.fingerprint, ctx.policy_sig, ctx.region_q),
            ),
            ChannelScanOutcome::ForcedMono => ChannelResolution {
                mode: Some(target),
                record: Some(ChannelDecisionRecord::auto(
                    VERDICT_FORCED_MONO,
                    ctx.fingerprint,
                    ctx.policy_sig,
                    ctx.region_q,
                )),
            },
            ChannelScanOutcome::FakeStereo => ChannelResolution {
                mode: Some(target),
                record: Some(ChannelDecisionRecord::auto(
                    VERDICT_FAKE_STEREO,
                    ctx.fingerprint,
                    ctx.policy_sig,
                    ctx.region_q,
                )),
            },
            ChannelScanOutcome::TrueStereo => ChannelResolution::record_only(
                ChannelDecisionRecord::auto(
                    VERDICT_TRUE_STEREO,
                    ctx.fingerprint,
                    ctx.policy_sig,
                    ctx.region_q,
                ),
            ),
            ChannelScanOutcome::Pending => ChannelResolution::record_only(
                ChannelDecisionRecord::pending(ctx.fingerprint, ctx.policy_sig, ctx.region_q),
            ),
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
    scan_source_with_opts(
        source_path,
        source_channels,
        region,
        policy,
        &policy.detect_options(),
    )
}

/// [`scan_source`] 的显式判定参数版本（导入路径用它收窄容器解码预算）。
pub fn scan_source_with_opts(
    source_path: Option<&Path>,
    source_channels: Option<u16>,
    region: Option<(f64, f64)>,
    policy: &ChannelImportPolicy,
    opts: &stereo_detect::DetectOptions,
) -> ChannelScanOutcome {
    let policy = policy.normalized();
    if policy.is_off() {
        return ChannelScanOutcome::PolicyOff;
    }
    let Some(path) = source_path else {
        return ChannelScanOutcome::NoSource;
    };

    // 声道数未知时探测一次（WAV 读头 / 容器探测，均为 O(1)）。
    // **探测失败 ⇒ Pending，而不是"按单声道处理"**：把读不到当成单声道会让
    // 假立体声素材被静默跳过、永不解码、永不重试，而报告里看起来"无事可做"。
    let channels = match source_channels {
        Some(c) if c > 0 => c,
        _ => match crate::audio_utils::try_read_audio_header_only(path) {
            Some(info) if info.channels > 0 => info.channels,
            _ => return ChannelScanOutcome::Pending,
        },
    };
    if channels < 2 {
        return ChannelScanOutcome::MonoSource;
    }

    if !policy.is_smart() {
        // alwaysMono：无需解码，直接折叠。
        return ChannelScanOutcome::ForcedMono;
    }

    match stereo_detect::verdict_for_file(path, region, opts) {
        ChannelVerdict::FakeStereo => ChannelScanOutcome::FakeStereo,
        ChannelVerdict::Mono => ChannelScanOutcome::MonoSource,
        ChannelVerdict::TrueStereo => ChannelScanOutcome::TrueStereo,
        // 覆盖不完整 / 解码失败 / 文件被占用：一律"本次读不到"。
        ChannelVerdict::Unknown => ChannelScanOutcome::Pending,
    }
}

/// 锁外：按策略为一个源文件算出落点。
pub fn precompute_decision(
    source_path: Option<&Path>,
    source_channels: Option<u16>,
    region: Option<(f64, f64)>,
    policy: &ChannelImportPolicy,
) -> ChannelResolution {
    precompute_decision_with_opts(
        source_path,
        source_channels,
        region,
        policy,
        &policy.detect_options(),
    )
}

/// [`precompute_decision`] 的显式判定参数版本。
///
/// 导入路径用短容器预算换低延迟：写下的档案带着短预算的签名，与后台扫描的
/// 长预算签名不同，因此**不会被误认为权威结论** —— 后台会带着完整预算重判，
/// 用户看到的结果不变，只是晚几百毫秒。
pub fn precompute_decision_with_opts(
    source_path: Option<&Path>,
    source_channels: Option<u16>,
    region: Option<(f64, f64)>,
    policy: &ChannelImportPolicy,
    opts: &stereo_detect::DetectOptions,
) -> ChannelResolution {
    let ctx = DecisionContext::for_region(None, region, opts);
    scan_source_with_opts(source_path, source_channels, region, policy, opts).resolution(policy, ctx)
}

/// 锁外：为一个 Take 算出落点（区间取自该 Take 的消费窗口，上下文取自该 Take
/// 持久化的指纹 —— 跨会话稳定，正是判定档案要比对的那一份）。
pub fn precompute_take_decision(take: &ClipTake, policy: &ChannelImportPolicy) -> ChannelResolution {
    let opts = policy.detect_options();
    let ctx = DecisionContext::for_take(take, &opts);
    scan_take(take, policy).resolution(policy, ctx)
}

/// 为一个 Take 判定并返回可展示的原因（区间取自其消费窗口）。
pub fn scan_take(take: &ClipTake, policy: &ChannelImportPolicy) -> ChannelScanOutcome {
    scan_source(
        take.source_path.as_deref().map(Path::new),
        take.source_channels,
        take_consumption_region(take),
        policy,
    )
}

/// 批量判定的请求条目：[`scan_source`] 三参的打包形（便于按文件分组）。
pub struct ScanRequest<'a> {
    pub source_path: Option<&'a Path>,
    pub source_channels: Option<u16>,
    /// 源域秒区间；`None` = 整个文件。
    pub region: Option<(f64, f64)>,
}

/// 锁外：按「源文件」分组批量判定。
///
/// 同一源文件被多个 Take/Clip 以不同（甚至相同）消费区间引用是常态——人声
/// 切片、多轨引用同一条伴奏。逐 Take 独立判定会对同一文件反复解码：一次整
/// 工程扫描 / 一次 v4→v5 迁移的解码量随**引用数**线性放大。分组后解码与
/// 头部/指纹探测都只随**文件数**增长；判定语义与逐条调用 [`scan_source`]
/// 完全一致（逐区间独立判定，结果照样进出进程级判定缓存）。
///
/// 返回值与 `requests` 一一对应。同样**会解码音频**，调用方必须保证不在
/// 持有 timeline 全局锁时调用。
pub fn scan_sources_grouped(
    requests: &[ScanRequest<'_>],
    policy: &ChannelImportPolicy,
) -> Vec<ChannelScanOutcome> {
    let policy = policy.normalized();
    let mut outcomes = vec![ChannelScanOutcome::PolicyOff; requests.len()];
    if policy.is_off() {
        return outcomes;
    }
    if requests.is_empty() {
        return outcomes;
    }

    // ── 无 I/O 的快路径逐条分类；需要解码的按路径分组 ──
    // 分组键的声道数先经一次 O(1) 头部探测定死（与 scan_source 相同的
    // "未知即探测、**探测失败即 Pending**" 口径），同组共享同一份判定 I/O。
    // `None` = 交给文件组批量判定；`Some` = 已就地定论（无需任何 I/O）。
    let mut resolved: Vec<Option<ChannelScanOutcome>> = Vec::with_capacity(requests.len());
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();

    for (index, request) in requests.iter().enumerate() {
        if !policy.is_smart() {
            // alwaysMono：不需要知道声道数，也不需要解码，直接折叠。
            // 无源 Take 除外（MIDI / 空白 Clip 不在折叠范围内）。
            resolved.push(Some(if request.source_path.is_some() {
                ChannelScanOutcome::ForcedMono
            } else {
                ChannelScanOutcome::NoSource
            }));
            continue;
        }
        let Some(path) = request.source_path else {
            resolved.push(Some(ChannelScanOutcome::NoSource));
            continue;
        };
        let channels = match request.source_channels {
            Some(c) if c > 0 => Some(c),
            _ => crate::audio_utils::try_read_audio_header_only(path)
                .map(|info| info.channels)
                .filter(|channels| *channels > 0),
        };
        match channels {
            Some(channels) if channels >= 2 => {}
            // 真的单声道源：定论，不需要解码。
            Some(_) => {
                resolved.push(Some(ChannelScanOutcome::MonoSource));
                continue;
            }
            // 探测不出声道数 ⇒ Pending（不是"按单声道"）：留待下次重试。
            None => {
                resolved.push(Some(ChannelScanOutcome::Pending));
                continue;
            }
        }
        let key = path.to_string_lossy().to_string();
        let group_index = match groups.iter().position(|(existing, _)| *existing == key) {
            Some(existing) => existing,
            None => {
                groups.push((key, Vec::new()));
                groups.len() - 1
            }
        };
        groups[group_index].1.push(index);
        resolved.push(None);
    }

    // ── 逐文件组判定：同组内相同区间只判一次（区间语义不变，纯去重）。 ──
    for (_, (_, member_indices)) in groups.into_iter().enumerate() {
        // 组内成员的路径一致（分组键即路径）。
        let path = requests[member_indices[0]]
            .source_path
            .unwrap_or_else(|| Path::new(""));
        // 组内唯一区间表（位模式做键，避免 f64 哈希歧义；语义上同键 = 同区间）。
        let mut unique_regions: Vec<Option<(f64, f64)>> = Vec::new();
        let mut region_slots: std::collections::HashMap<(u64, u64), usize> =
            std::collections::HashMap::new();
        let mut member_slot: Vec<usize> = Vec::with_capacity(member_indices.len());
        for &member in &member_indices {
            let region = requests[member].region;
            let key = (
                region.map(|(s, _)| s.to_bits()).unwrap_or(0),
                region.map(|(_, e)| e.to_bits()).unwrap_or(0),
            );
            let slot = *region_slots.entry(key).or_insert_with(|| {
                unique_regions.push(region);
                unique_regions.len() - 1
            });
            member_slot.push(slot);
        }

        let verdicts = stereo_detect::verdict_for_regions(path, &unique_regions, &policy.detect_options());

        for (position, &member) in member_indices.iter().enumerate() {
            outcomes[member] = match verdicts[member_slot[position]] {
                ChannelVerdict::FakeStereo => ChannelScanOutcome::FakeStereo,
                ChannelVerdict::Mono => ChannelScanOutcome::MonoSource,
                ChannelVerdict::TrueStereo => ChannelScanOutcome::TrueStereo,
                ChannelVerdict::Unknown => ChannelScanOutcome::Pending,
            };
        }
    }

    // 就地定论的条目最后落位（分组覆盖不到它们）。
    for (index, slot) in resolved.into_iter().enumerate() {
        if let Some(outcome) = slot {
            outcomes[index] = outcome;
        }
    }
    outcomes
}

/// 锁内：把一个已算好的落点应用到 Take。
///
/// 零解码、零 IO，可安全在持锁状态下调用。返回**分别**报告"模式变了"与
/// "档案变了" —— 两者后果不同：模式变化改变听感（必须失效渲染/formant 缓存），
/// 档案变化只是记账（不该触发任何失效，也不该产生撤销步）。
pub fn apply_resolution(take: &mut ClipTake, resolution: ChannelResolution) -> AppliedResolution {
    let mut applied = AppliedResolution::default();
    if let Some(mode) = resolution.mode {
        let mode = crate::channel_mode::TakeChannelMode::from_raw(mode).raw();
        if take.channel_mode != mode {
            take.channel_mode = mode;
            applied.mode_changed = true;
        }
    }
    if let Some(record) = resolution.record {
        if take.channel_decision != Some(record) {
            take.channel_decision = Some(record);
            applied.record_changed = true;
        }
    }
    applied
}

/// 该落点若应用到 `take`，是否会改变它的**声道模式**。
///
/// 与"是否发生任何写入"不同：判定档案是内部记账，它从无到有不构成"用户的一次
/// 编辑"，因此**不该**产生撤销步 —— 否则用户按下撤销会发现"什么都没变"。
/// 只有模式真的变了（听感变了）才值得留一个撤销点。
///
/// 与 [`apply_resolution`] 用同一套模式判据，但**不改动** Take。
pub fn resolution_changes_mode(take: &ClipTake, resolution: ChannelResolution) -> bool {
    resolution
        .mode
        .map(|mode| crate::channel_mode::TakeChannelMode::from_raw(mode).raw() != take.channel_mode)
        .unwrap_or(false)
}

/// [`apply_resolution`] 的结果：两个变化维度分开报告。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppliedResolution {
    /// 声道模式是否真的变了（听感变化 → 需要失效缓存 / 重调度）。
    pub mode_changed: bool,
    /// 判定档案是否变了（纯记账 → 不需要失效任何东西）。
    pub record_changed: bool,
}

impl AppliedResolution {
    /// 是否发生了任何写入（用于判断要不要留撤销步）。
    pub fn any(self) -> bool {
        self.mode_changed || self.record_changed
    }
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
/// 返回声道模式真正发生变化的 Take 数。用于 VocalShifter 导入等"整批 Take
/// 都无权威声道信息"的场景。
pub fn resolve_clip_takes_channel_mode(
    clip: &mut crate::state::Clip,
    policy: &ChannelImportPolicy,
) -> usize {
    let resolutions: Vec<ChannelResolution> = clip
        .takes
        .iter()
        .map(|take| precompute_take_decision(take, policy))
        .collect();
    let mut changed = 0usize;
    for (take, resolution) in clip.takes.iter_mut().zip(resolutions) {
        if apply_resolution(take, resolution).mode_changed {
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
    ///
    /// 返回声道模式是否真的变了（与生产语义一致：只有模式变化才算"折叠"）。
    fn resolve(take: &mut ClipTake, policy: &ChannelImportPolicy) -> bool {
        let resolution = precompute_take_decision(take, policy);
        apply_resolution(take, resolution).mode_changed
    }

    /// 同上，但返回完整落点（用于断言判定档案）。
    fn resolve_full(
        take: &mut ClipTake,
        policy: &ChannelImportPolicy,
    ) -> AppliedResolution {
        let resolution = precompute_take_decision(take, policy);
        apply_resolution(take, resolution)
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
        // 无音频源的 Take（MIDI / 空白 Clip）不属于声道折叠的适用范围：
        // 连 alwaysMono 也不碰它，也不留下判定档案（否则它会在扫描报告里
        // 冒充"读不到"而反复重试）。
        let mut take = take_with(None, Some(2));
        let policy = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            ..Default::default()
        };
        let applied = resolve_full(&mut take, &policy);
        assert!(!applied.any());
        assert_eq!(take.channel_mode, 0);
        assert_eq!(take.channel_decision, None);
    }

    #[test]
    fn pending_records_are_marked_for_retry() {
        // 文件不存在 ⇒ Pending（不是"单声道"）：档案必须记成待重试，
        // 否则这个 Take 的漏判会被永久静默。
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        let policy = ChannelImportPolicy::default();
        let applied = resolve_full(&mut take, &policy);
        assert!(applied.record_changed, "必须落下判定档案");
        assert!(!applied.mode_changed, "读不到时不得改模式");
        assert!(
            take.channel_decision.expect("record").is_pending(),
            "读不到必须记成 pending"
        );
    }

    #[test]
    fn probe_failure_is_pending_not_mono() {
        // 关键回归：探测不出声道数（文件不可读）曾被当成"单声道源"静默跳过，
        // 于是假立体声素材永不解码、永不重试，报告里还显示成"无事可做"。
        let outcome = scan_source(
            Some(std::path::Path::new("C:/definitely/missing.wav")),
            None,
            None,
            &ChannelImportPolicy::default(),
        );
        assert_eq!(outcome, ChannelScanOutcome::Pending);
        assert_ne!(outcome, ChannelScanOutcome::MonoSource);
    }

    #[test]
    fn off_policy_leaves_no_record() {
        // 策略关闭时不该留下"我们看过了"的痕迹：否则用户重新打开功能后，
        // 这些 Take 会被当成"已定论"而跳过。
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        let policy = ChannelImportPolicy {
            mode: "off".into(),
            ..Default::default()
        };
        let applied = resolve_full(&mut take, &policy);
        assert!(!applied.any());
        assert_eq!(take.channel_decision, None);
    }

    #[test]
    fn user_seal_is_never_touched_by_the_policy() {
        // 用户显式设置过的 Take：自动扫描必须永远放手。
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        take.channel_decision = Some(ChannelDecisionRecord::user());
        take.channel_mode = 0;
        let policy = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            ..Default::default()
        };
        // `precompute_take_decision` 本身不做候选筛选（那是 collect_targets 的
        // 职责），所以这里直接断言候选筛选的判据。
        let ctx = DecisionContext::for_take(&take, &policy.detect_options());
        assert!(!crate::channel_decision::needs_auto_scan(
            take.channel_decision,
            take.source_file_fingerprint,
            ctx.policy_sig,
            ctx.region_q,
        ));
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
    fn apply_resolution_is_idempotent() {
        let mut take = take_with(None, Some(2));
        let ctx = DecisionContext::for_region(None, None, &crate::stereo_detect::DetectOptions::default());
        let fold = ChannelScanOutcome::FakeStereo.resolution(&ChannelImportPolicy::default(), ctx);
        assert!(apply_resolution(&mut take, fold).mode_changed);
        assert_eq!(take.channel_mode, 2);
        // 已经是目标模式、且档案已落 → 第二次不再报告任何改变。
        assert!(!apply_resolution(&mut take, fold).any());
    }

    #[test]
    fn apply_resolution_noop_leaves_everything_alone() {
        let mut take = take_with(None, Some(2));
        take.channel_mode = 1;
        assert!(!apply_resolution(&mut take, ChannelResolution::NOOP).any());
        assert_eq!(take.channel_mode, 1);
        assert_eq!(take.channel_decision, None);
    }

    #[test]
    fn apply_resolution_normalizes_out_of_range_mode() {
        // 越界目标模式经 from_raw 回落 Normal(0)；绝不能让非法值写进 Take。
        let mut take = take_with(None, Some(2));
        take.channel_mode = 1;
        let resolution = ChannelResolution {
            mode: Some(99),
            record: None,
        };
        assert!(apply_resolution(&mut take, resolution).mode_changed);
        assert_eq!(take.channel_mode, 0, "越界模式回落 Normal");
        assert!(!apply_resolution(&mut take, resolution).any(), "幂等");
    }

    #[test]
    fn record_only_resolution_does_not_touch_the_mode() {
        // "已定论但无需动作"（真立体声）：只记账，不改模式。
        let mut take = take_with(Some("C:/definitely/missing.wav"), Some(2));
        take.channel_mode = 1;
        let applied = resolve_full(&mut take, &ChannelImportPolicy::default());
        assert!(!applied.mode_changed, "读不到时不得改模式");
        assert_eq!(take.channel_mode, 1, "既有模式必须原样保留");
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

        let mut a = take_with(Some("C:/a.wav"), Some(2));
        a.id = "a".into();
        let mut b = take_with(Some("C:/b.wav"), Some(1));
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
    fn scan_outcome_reports_the_reason_and_derives_the_resolution() {
        let smart = ChannelImportPolicy::default();
        let forced = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            mono_target_mode: 4,
            ..Default::default()
        };
        let ctx = DecisionContext::for_region(None, None, &smart.detect_options());

        // 只有"假立体声"与"强制转换"会改模式。
        assert_eq!(
            ChannelScanOutcome::FakeStereo.resolution(&smart, ctx).mode,
            Some(2)
        );
        assert_eq!(
            ChannelScanOutcome::ForcedMono.resolution(&forced, ctx).mode,
            Some(4),
            "目标模式取自策略"
        );
        // 其余结论一律不改模式，但**仍然要记账**（这样"读不到"才有重试机会）。
        for keep in [
            ChannelScanOutcome::MonoSource,
            ChannelScanOutcome::TrueStereo,
            ChannelScanOutcome::Pending,
        ] {
            let resolution = keep.resolution(&smart, ctx);
            assert_eq!(resolution.mode, None, "{keep:?}");
            assert!(resolution.record.is_some(), "{keep:?} 必须落下档案");
        }
        // 不适用 / 策略关闭：既不改模式，也不留档案。
        for noop in [ChannelScanOutcome::PolicyOff, ChannelScanOutcome::NoSource] {
            let resolution = noop.resolution(&smart, ctx);
            assert_eq!(resolution.mode, None, "{noop:?}");
            assert!(resolution.record.is_none(), "{noop:?} 不该留下痕迹");
        }
        // 即便判定结果是"假立体声"，策略关闭时也不得折叠、不留档案。
        assert_eq!(
            ChannelScanOutcome::FakeStereo.resolution(
                &ChannelImportPolicy {
                    mode: "off".into(),
                    ..Default::default()
                },
                ctx
            ),
            ChannelResolution::NOOP
        );
        // 原因字符串是 IPC 契约的一部分。
        assert_eq!(ChannelScanOutcome::PolicyOff.as_str(), "policyOff");
        assert_eq!(ChannelScanOutcome::NoSource.as_str(), "noSource");
        assert_eq!(ChannelScanOutcome::FakeStereo.as_str(), "fakeStereo");
        assert_eq!(ChannelScanOutcome::TrueStereo.as_str(), "trueStereo");
        assert_eq!(ChannelScanOutcome::MonoSource.as_str(), "mono");
        assert_eq!(ChannelScanOutcome::ForcedMono.as_str(), "forcedMono");
        assert_eq!(ChannelScanOutcome::Pending.as_str(), "pending");
    }

    #[test]
    fn scan_source_distinguishes_the_reasons() {
        let missing = std::path::Path::new("C:/definitely/missing.wav");
        // 策略关闭：不做判定。
        assert_eq!(
            scan_source(Some(missing), Some(2), None, &ChannelImportPolicy { mode: "off".into(), ..Default::default() }),
            ChannelScanOutcome::PolicyOff
        );
        // 无源（MIDI / 空白 Clip）：不适用，且**不是**"读不到"。
        assert_eq!(
            scan_source(None, Some(1), None, &ChannelImportPolicy::default()),
            ChannelScanOutcome::NoSource
        );
        // 单声道源：无需判定。
        assert_eq!(
            scan_source(Some(missing), Some(1), None, &ChannelImportPolicy::default()),
            ChannelScanOutcome::MonoSource
        );
        // 强制转换：不判定内容。
        assert_eq!(
            scan_source(Some(missing), Some(2), None, &ChannelImportPolicy { mode: "alwaysMono".into(), ..Default::default() }),
            ChannelScanOutcome::ForcedMono
        );
        // 智能模式但源不可读 → Pending（不折叠，且下次重试）。
        assert_eq!(
            scan_source(Some(missing), Some(2), None, &ChannelImportPolicy::default()),
            ChannelScanOutcome::Pending
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

    /// 写 2 秒 WAV；`identical` 时 L == R，否则 R = −L。
    fn write_scan_wav(path: &std::path::Path, identical: bool) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..44_100 * 2 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(if identical { v } else { -v }).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn slices_of_one_source_agree_whatever_the_batch() {
        // 用户可见性质：同一源文件的多个切片被放在**同一批**判定时，结论必须与
        // 逐条判定时一致 —— 不能因为同批里有个读不到的切片就把其余的都降级。
        let dir = std::env::temp_dir().join("hifishifter_policy_batch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fake.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..44_100 * 3 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        let policy = ChannelImportPolicy::default();

        let slice = |region: Option<(f64, f64)>| ScanRequest {
            source_path: Some(path.as_path()),
            source_channels: Some(2),
            region,
        };

        let solo = scan_sources_grouped(&[slice(Some((0.0, 1.0)))], &policy);
        // 第二个切片越出文件末尾 ⇒ 读不到，但它不该影响第一个切片。
        let mixed = scan_sources_grouped(
            &[slice(Some((0.0, 1.0))), slice(Some((90.0, 120.0)))],
            &policy,
        );
        let pair = scan_sources_grouped(
            &[slice(Some((0.0, 1.0))), slice(Some((1.0, 2.0)))],
            &policy,
        );
        assert_eq!(solo[0], ChannelScanOutcome::FakeStereo);
        assert_eq!(
            mixed[0], solo[0],
            "同批里有个读不到的切片时，其余切片结论不得改变"
        );
        assert_eq!(pair[0], solo[0]);
        assert_eq!(pair[1], solo[0], "同一源的两个切片结论必须一致");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn grouped_scan_matches_per_take_semantics() {
        // 同一文件的多个 Take 以不同消费区间引用 —— 分组判定的结论必须与
        // 逐 Take 独立判定逐条一致（区间语义不变，去重的只是 I/O）。
        let dir = std::env::temp_dir().join("hifishifter_channel_policy_grouped");
        std::fs::create_dir_all(&dir).unwrap();
        let policy = ChannelImportPolicy::default();

        // 前 1 秒假立体声、后 1 秒真立体声的拼接文件。
        let mixed = dir.join("grouped_mixed.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&mixed, spec).unwrap();
        for i in 0..44_100 {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        for i in 0..44_100 {
            let v = ((i as f32) * 0.02).sin() * 0.4;
            w.write_sample(v).unwrap();
            w.write_sample(-v).unwrap();
        }
        w.finalize().unwrap();

        let fake_part = dir.join("grouped_fake.wav");
        let true_part = dir.join("grouped_true.wav");
        write_scan_wav(&fake_part, true);
        write_scan_wav(&true_part, false);

        let requests = vec![
            // 同一假立体声文件被三个 Take 引用：两个相同区间 + 一个不同区间。
            ScanRequest {
                source_path: Some(fake_part.as_path()),
                source_channels: Some(2),
                region: Some((0.0, 0.5)),
            },
            ScanRequest {
                source_path: Some(fake_part.as_path()),
                source_channels: Some(2),
                region: Some((0.0, 0.5)),
            },
            ScanRequest {
                source_path: Some(fake_part.as_path()),
                source_channels: Some(2),
                region: Some((0.5, 1.5)),
            },
            // 同一"前假后真"文件：整文件判 TrueStereo（保守），窄前段判 FakeStereo。
            ScanRequest {
                source_path: Some(mixed.as_path()),
                source_channels: Some(2),
                region: None,
            },
            ScanRequest {
                source_path: Some(mixed.as_path()),
                source_channels: Some(2),
                region: Some((0.0, 1.0)),
            },
            // 快路径：单声道源 / 策略强制 / 缺源。
            ScanRequest {
                source_path: Some(fake_part.as_path()),
                source_channels: Some(1),
                region: None,
            },
            ScanRequest {
                source_path: None,
                source_channels: Some(2),
                region: None,
            },
        ];
        let outcomes = scan_sources_grouped(&requests, &policy);
        assert_eq!(outcomes.len(), requests.len());
        assert_eq!(outcomes[0], ChannelScanOutcome::FakeStereo);
        assert_eq!(outcomes[1], ChannelScanOutcome::FakeStereo, "相同区间去重共享判定");
        assert_eq!(outcomes[2], ChannelScanOutcome::FakeStereo);
        assert_eq!(
            outcomes[3],
            ChannelScanOutcome::TrueStereo,
            "整文件区间覆盖到后半真立体声段，不得判假"
        );
        assert_eq!(outcomes[4], ChannelScanOutcome::FakeStereo);
        assert_eq!(outcomes[5], ChannelScanOutcome::MonoSource);
        assert_eq!(
            outcomes[6],
            ChannelScanOutcome::NoSource,
            "无源 Take 不属于声道折叠的适用范围"
        );

        // 与逐条 scan_source 逐条对拍（同样输入、独立实现路径）。
        for (request, expected) in requests.iter().zip(&outcomes) {
            assert_eq!(
                scan_source(request.source_path, request.source_channels, request.region, &policy),
                *expected,
                "分组判定必须与逐条判定一致"
            );
        }

        let _ = std::fs::remove_file(&mixed);
        let _ = std::fs::remove_file(&fake_part);
        let _ = std::fs::remove_file(&true_part);
    }
}
