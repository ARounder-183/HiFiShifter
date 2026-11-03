//! 每条进程流的块几何、活动窗口与输出接缝状态（音频线程读写，无锁）。
use super::timebase::TimebaseKind;
use super::*;

/// 一条进程流自己的块几何（音频线程写，无锁）。
///
/// # 位置记账只有一只钟
///
/// 这里曾经有**两只**：宿主报的 `projectTimeSamples`，和插件按块长数出来的累积时钟
/// （`virtual_position` + `host_clock_offset` + `pinned_frames` + 4 个窗口原子）。两者
/// 之间的差用一个 1.5 s 的容差仲裁，容差之内一律相信累积时钟。
///
/// 那个设计的后果是：**用户回到起点再播（0.3–1.2 s 的循环）落在容差之内**，于是被
/// 判成"宿主读数滞后"，位置被强制成"上一遍末尾 + 块长"，整遍内容偏移恰好一个循环
/// 长度。这不是一个可以靠调容差修好的参数问题 —— 只要存在两只钟，就必须有一个仲裁，
/// 而任何仲裁都会在某一种真实操作上判错。
///
/// 现在只有三条规则（见 `note_stream_geometry` 的 A/B/C）：
/// **宿主每报一次位置就以它为准；两次报告之间按收到的帧数前进（无上限）；
/// 跨 pass 一律归零。**
#[derive(Default)]
pub(super) struct StreamGeometry {
    pub(super) seen: AtomicBool,
    pub(super) last_start: AtomicI64,
    pub(super) last_frames: AtomicU32,
    pub(super) last_end: AtomicI64,
    /// 同一个块首连续出现的次数。
    pub(super) same_start: AtomicU32,
    /// 本块首上是否已经允许过一次“显著更大”的扩展块。
    pub(super) extension_used: AtomicBool,

    // ── PassClock ──────────────────────────────────────────────────────
    /// 本 pass 的锚点：宿主**最后一次**报告的位置。
    ///
    /// 【为什么只需要一个 i64】`R1`（新 pass）与 `R3`（宿主报了新位置）都会把它覆盖成
    /// 宿主刚给的值；`R2`（宿主没动）只读不写。于是"锚点是否有效"由 `seen` 与
    /// `pass` 共同回答，不需要一个独立的 `anchor_valid`。
    pub(super) anchor: AtomicI64,
    /// 自 `anchor` 以来**已经送来**的帧数（不含当前块）。
    ///
    /// 【为什么是"已送来的帧数"而不是"上一位置 + 本块长"】钉住的起点不一定落在第一个
    /// 块上；累加"已送来的帧数"让位置从一开始就对齐，而不必依赖 latch 在哪一块打开。
    pub(super) frames_since_anchor: AtomicU64,
    /// 本 pass 里"位置由**计数器**给出"的连续帧数（宿主没有报新位置）。
    ///
    /// 【它是**事实**，不是额度】旧实现这里有**两笔预算**（`advance_limit` 与
    /// `activity_budget`，都等于 `PRE_ROLL_CAP_SEC` = 1 秒）。预算不是判据：它既不能
    /// 证明"时间在走"（超过 1 秒的真实播放会被切断 —— 非实时 FX 预览恰好卡在
    /// 1.025 秒），也不能否证"时间在走"（宿主真的停播时它只是恰好用完）。
    ///
    /// 现在它只回答三个**可核对**的问题：
    ///
    ///   1. `start_window`：本 pass 的计数器窗口还开着吗（静音也要放行的那一段）；
    ///   2. `resync_limit_frames`：宿主下次报位置时，差值落在"我们可能错过多少帧"
    ///      之内吗（之内 = 记账纠正，之外 = 时间基切换 / 真 seek）；
    ///   3. `new_pass`：位置倒退时，本 pass 里计数器有没有权威（有 ⇒ 那是宿主换了
    ///      **时间基**，不是 seek）。
    ///
    /// 【为什么它必须单调无上限】连续音频被切成两截的唯一原因就是给这个数设了上限。
    pub(super) counted_run: AtomicU64,
    /// 本 pass 的**位置域偏移**：宿主读数与音频真实连续性之间的固定差（帧）。
    ///
    /// 【为什么需要它】VEGAS 的 Event 级 FX 在起播窗口里报**绝对工程位置**，窗口结束后
    /// 改报**事件相对位置**（实测 `raw` 从 31.091 s 跳到 0）。音频是连续的，所以这不是
    /// seek —— 但两个读数之间差了整整一个事件偏移。偏移量就是"计数器在本 pass 里
    /// 权威了多久"：宿主的新域从**本 pass 的起点**重新起算。
    ///
    /// 于是 `position = raw_start + domain_offset`，整条 take 落在新域里（Event 起点
    /// = 工程起点），而音频一帧不丢。`new_pass` 时清零（新一次播放，宿主的域是权威）。
    pub(super) domain_offset: AtomicI64,
    /// 上一块的 `kPlaying`（"停播很久之后的上升沿"需要它）。
    pub(super) last_playing: AtomicBool,
    /// 传输会话的相位（见 `transport_session::SessionPhase`）。
    pub(super) session_phase: AtomicU8,
    /// 传输会话序号。每次新的播放 / 离线 pass 递增。
    pub(super) session_epoch: AtomicU64,
    /// 本会话已经收到的帧数（用于 2 秒域切换窗口与 1 秒起点窗口）。
    pub(super) session_frames: AtomicU64,
    /// 当前宿主时基（见 `timebase::TimebaseKind`）。
    pub(super) timebase_kind: AtomicU8,
    /// 本地域偏移：local = raw + offset。
    pub(super) timebase_offset: AtomicI64,
    /// 最近一次整条 take 平移的帧数（诊断；正 = 向右）。
    pub(super) last_rebase: AtomicI64,
    /// 整条 take 平移次数。
    pub(super) rebase_count: AtomicU64,
    /// 本流的 pass 序号（R1 递增）。`0` = 还没有过任何一块。
    pub(super) pass: AtomicU64,
    /// 锚点那一刻 `continuousTimeSamples` / `projectTimeMusic` 的读数。
    ///
    /// 【为什么存锚而不是累加增量】两者都是**绝对**读数（相对工程起点）。把"本块与
    /// 上一块的差"再叠到计数器上，会把同一段时间算两遍 —— 实测就是"位置钉住 +
    /// continuous 前进"时位置翻倍。存锚之后位置恒为
    /// `anchor + (continuous − continuous_anchor)`，与计数器互不叠加。
    pub(super) continuous_anchor: AtomicI64,
    pub(super) music_anchor: AtomicU64,

    pub(super) last_raw_start: AtomicI64,
    pub(super) last_continuous: AtomicI64,
    pub(super) last_music_bits: AtomicU64,
    pub(super) last_position: AtomicI64,
    pub(super) last_source: AtomicU8,

    // ── 读数（I-3：位置是否逐块续接）─────────────────────────────────
    /// 位置**倒退**的块数。稳态下恒为 0。
    pub(super) clock_rewinds: AtomicU64,
    /// 位置增量绝对值的最小/最大（帧）。稳态下两者都等于块长。
    pub(super) clock_delta_min: AtomicU64,
    pub(super) clock_delta_max: AtomicU64,
    /// 已经量过增量的块数（`delta_min` 的惰性初始化用）。
    pub(super) clock_deltas: AtomicU64,
    /// `R3` 里"宿主报的位置与计数器不同"的次数。
    pub(super) jumps: AtomicU64,
    /// 最近一次 jump 的带符号幅度（帧）。正 = 计数器落后于宿主。
    pub(super) last_jump: AtomicI64,
    /// 位置由**计数器**推进的块数 / 帧数（R2 且没有更好的宿主时钟）。
    ///
    /// 【为什么它必须与 `position_source` 分开】"位置由我们数"与"位置来自哪条宿主
    /// 时钟"是两个正交的事实。混进一个枚举，就得到了 round 24 那个粘住整场会话的
    /// `Carried` 标签 —— 它让"宿主钉住过位置"这件事在读数上无法与"宿主一直在报位置"
    /// 区分。
    pub(super) counted_blocks: AtomicU64,
    pub(super) counted_frames: AtomicU64,

    pub(super) blocks: AtomicU64,
    pub(super) rolling_blocks: AtomicU64,
    pub(super) resting_blocks: AtomicU64,
    pub(super) rising_edges: AtomicU64,
    pub(super) continuous_used: AtomicU64,
    pub(super) music_used: AtomicU64,
    pub(super) position_resyncs: AtomicU64,
    pub(super) cycle_wraps: AtomicU64,
    pub(super) cycle_wrap_unknown: AtomicU64,

    /// 最近 8 个块里"输入内容与上一块不同"的位图。
    ///
    /// 【为什么是"内容变化"而不是"有信号"】停播时宿主也可能持续送同一份非零内容
    /// （FL Studio 就是如此）。"有信号"会把停播判成播放；"内容在变"才是音频**正在
    /// 流动**的证据 —— 同一段内容反复出现说明时间没有前进。
    pub(super) activity_changed: AtomicU32,
    /// 已经观测过的块数（饱和）。
    pub(super) activity_seen: AtomicU32,
    /// 上一块的内容指纹。
    pub(super) activity_hash: AtomicU64,
    pub(super) activity_hash_valid: AtomicBool,
    /// 连续"内容没变"的块数（有变化就清零）。
    pub(super) unchanged_run: AtomicU32,
    /// 本块处理**之前**是否处在"内容连续不变"的静止段。
    ///
    /// 【为什么要单独记】`unchanged_run` 在内容一变就清零，而相位判定发生在那之后；
    /// 想回答"这一块是不是刚从静止转入流动"就必须把结论留到那时。
    pub(super) was_stationary: AtomicBool,
    /// 本流在位置钉住期间**紧接在本块之前**的连续静音帧数。
    ///
    /// 【为什么需要】宿主可能先送 0.4 s 静音再送内容（VEGAS 起播窗口）。这段静音
    /// 没有被写入捕获时，快照会从内容起点才开始覆盖，播放时就在 0.4 s 发生一次
    /// 块内源切换（"咔"）。捕获侧只需要帧数，不需要 PCM —— 它可以在第一段开头
    /// 合成等长静音，把段起点前移到播放起点。
    pub(super) trailing_silent_frames: AtomicU64,
    /// 本块是否全零。
    ///
    /// 【为什么还需要它】它不参与位置推进，只作为诊断读数与"停播重复投递"的
    /// 判据：窗口内有多少块是静音，用来判断宿主起播窗口的形状。
    pub(super) last_block_silent: AtomicBool,
}

/// "音频在流动"的观测窗口（块数）。
///
/// 【为什么是 8】窗口同时决定两件事：判据的灵敏度（至少 1 块有变化才成立）与
/// **停播后的尾巴**（最多 8 块 ≈ 170 ms @ 1024/48k）。再大就会让"停下之后"继续出声；
/// 再小则一次气口（静音块）就能把"在流动"判成"停了"。
pub(super) const ACTIVITY_WINDOW_BLOCKS: u32 = 8;
/// 窗口内至少这么多块的内容有变化，才认为音频在流动。
///
/// 【为什么是 1】停播时宿主重复投递的是**同一份**内容（逐块相同），而真正在播放
/// 时相邻块几乎不可能完全相同。要求 2 块会让起播第一块被判成停播而丢失（那正是
/// 用户报的"开头一小段录不到"）。单块误判的代价只是多出一块（约 21 ms），
/// 与丢一块的代价同量级，而"宁可多录"是本轮的既定方向。
pub(super) const ACTIVITY_MIN_CHANGES: u32 = 1;
/// 一个 pass 里允许累积的**前导静音**上限（秒）。
///
/// 【它约束什么，不约束什么】只约束 `trailing_silent_frames` 的累积与 `start_window`
/// 的宽度（两者都是"这次播放刚开始"的证据）；**不约束位置推进** —— 这条纪律此前被
/// 代码违反（同一个常数被当成两笔位置预算），非实时 FX 预览因此卡在恰好 1.000 秒。
/// VEGAS 的起播窗口约 0.66 s，1 秒足够覆盖；超过 1 秒的静音由捕获侧的记账纠正补齐。
pub(super) const PRE_ROLL_CAP_SEC: f64 = 1.0;
impl StreamGeometry {
    /// 记录本块的内容指纹，回答"这一块与上一块是不是同一份内容"。
    pub(super) fn note_activity(&self, hash: u64) -> bool {
        let previous_valid = self.activity_hash_valid.swap(true, Ordering::AcqRel);
        let previous = self.activity_hash.swap(hash, Ordering::AcqRel);
        let changed = previous_valid && previous != hash;
        self.was_stationary.store(
            self.unchanged_run.load(Ordering::Acquire) > 0,
            Ordering::Release,
        );
        if changed {
            self.unchanged_run.store(0, Ordering::Release);
        } else if previous_valid {
            self.unchanged_run.fetch_add(1, Ordering::Relaxed);
        }
        // 只保留最近 8 位；`changed` 占最低位。
        let window = self.activity_changed.load(Ordering::Acquire);
        let window = (window << 1) | u32::from(changed);
        self.activity_changed.store(
            window & ((1 << ACTIVITY_WINDOW_BLOCKS) - 1),
            Ordering::Release,
        );
        self.activity_seen
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |seen| {
                Some(seen.saturating_add(1))
            })
            .ok();
        changed
    }

    /// 记录本块是否全零（由 `capture_observe` 在几何判定之前调用）。
    pub(super) fn note_block_silence(&self, silent: bool) {
        self.last_block_silent.store(silent, Ordering::Release);
    }

    /// 最近窗口里是否有足够多的"内容变化"块。
    pub(super) fn activity_moving(&self) -> bool {
        self.activity_changed.load(Ordering::Acquire).count_ones() >= ACTIVITY_MIN_CHANGES
    }

    /// 本块是否"音频在流动"（= 宿主确实在送时间上相邻的样本）。
    ///
    /// 【它是**唯一**的计数器判据】此前计数器要过两笔预算（`advance_limit` /
    /// `activity_budget`）才肯推进，而预算把连续音频切在恰好 1.000 秒。判据换回事实
    /// 之后，位置推进不再有任何与宿主无关的上限。
    ///
    /// 【为什么静音也算流动】真正的静音是宿主在送**全零**，那是一段休止（真实时间）；
    /// 而"宿主重复投递同一段有内容的缓冲"是卡住。`unchanged_run` 已经能分开它们 ——
    /// 它要求"逐块完全相同**且非零**"。
    ///
    /// 【为什么"卡住"必须单独判】FL Studio 停播时 `kPlaying` 仍为 1、内容逐块相同。
    /// 只看"有信号"会把停播判成播放，计数器一路前进（实测 3.4 s）。
    pub(super) fn flowing(&self) -> bool {
        let silent = self.last_block_silent.load(Ordering::Acquire);
        let stalled = self.unchanged_run.load(Ordering::Acquire) > 0 && !silent;
        (self.activity_moving() || silent) && !stalled
    }
}

/// 内容指纹一次扫描覆盖的帧数（块首固定长度）。
///
/// 【为什么是固定长度】指纹回答的是"这一块是不是上一块的**重复投递**"，而重复投递
/// 的形状是整块相同（含块首）。取固定长度前缀让**块长交替**不再影响判据 ——
/// VEGAS 的块长在 1200/848 之间交替（日志实测），把块长拌进指纹会让同一份内容
/// 被判成"在变"，重复抑制彻底失效。
///
/// 【为什么是 512 而不是抽样】旧实现每 8 个采样取一个点。对"内容真的相同"这个判据
/// 来说抽样是**不安全**的：一个周期恰好整除抽样步长的信号（持续长音最常见的形状）
/// 会被误判成静止。512 个**连续**采样要求信号在块边界上逐位重复，那对任何真实音频
/// 都不成立。
pub(super) const ACTIVITY_FINGERPRINT_FRAMES: usize = 512;

/// 一次扫描同时给出"内容指纹"与"是否全零"。
///
/// 【为什么合并】两者都要遍历整块；分开写就是两遍。合并之后净成本比旧实现更低
/// （旧实现也是两遍：一遍抽样算指纹、一遍判静音）。
pub(super) fn block_fingerprint(block: &crate::capture::recorder::InputBlock<'_>) -> (u64, bool) {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut silent = true;
    for plane in block.planes.iter().flatten() {
        for (index, sample) in plane.iter().enumerate() {
            if *sample != 0.0 {
                silent = false;
            }
            if index < ACTIVITY_FINGERPRINT_FRAMES {
                hash ^= u64::from(sample.to_bits());
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    (hash, silent)
}

/// 本块是否全零（锁存推进只对静音块生效）。
///
/// 【为什么只剩测试在用】生产路径现在一次扫描同时拿到指纹与静音
/// （见 [`block_fingerprint`]），再单独扫一遍只为判静音就是白跑一趟。
#[cfg(test)]
pub(super) fn block_is_silent(block: &crate::capture::recorder::InputBlock<'_>) -> bool {
    block_fingerprint(block).1
}

/// 一条流在本块上的位置/相位结论。
#[derive(Clone, Copy, Debug)]
pub(super) struct StreamBlock {
    /// 本块的有效工程起点（可能来自 continuous/music 回退）。
    pub(super) position: i64,
    /// 宿主原始 `projectTimeSamples`。
    pub(super) raw_start: i64,
    pub(super) phase: TransportPhase,
    pub(super) had_previous: bool,
    pub(super) cycle_wrap: bool,
    /// 本块开始了一个新的 pass（循环回卷或位置后退）。
    pub(super) new_pass: bool,
    /// 本流最近是否观测到"输入内容在变化"（音频在流动）。
    pub(super) activity_moving: bool,
    /// 本块是宿主**纠正了我们数出来的位置**（见 `note_stream_geometry` 的 R3）。
    pub(super) resync: bool,
    /// 允许当作"记账纠正"的最大帧差（钉住期间送来的帧数 + 本块）。
    pub(super) resync_limit_frames: u64,
    /// 本块之前连续静音的帧数（见 `StreamGeometry::trailing_silent_frames`）。
    pub(super) preceding_silent_frames: u64,
    /// 本块位置由**计数器**推进（宿主没有报新位置）。
    ///
    /// 【为什么与 `position_source` 分开】"位置由我们数"与"位置来自哪条宿主时钟"
    /// 是两个正交的事实：计数器推进时标签仍是 `Project`（宿主给的那个起点确实是
    /// `Project`），只是这一块没有新的宿主报告。混进一个枚举就得到了 round 24 那个
    /// 粘住整场会话的 `Carried`。
    pub(super) counted: bool,
    /// 本块是停播很久之后的上升沿：此前累积的静音必须从 take 前导里扣掉。
    pub(super) discard_pre_roll: bool,
    /// 起播/录音上升沿打开的窗口仍开着（门控据此放行窗口内静音）。
    pub(super) start_window: bool,
    /// 本块与上一块逐块完全相同且非零（FL 停播重复投递的形状）。
    pub(super) repeated_nonzero: bool,
    /// 本块有效位置来自哪条**宿主时钟**（见 `PositionSource`）。
    ///
    /// 【它不再表示"位置可不可信"】位置只有一只钟（见 `note_stream_geometry` 的
    /// R1–R4），所以这个字段只回答"宿主用哪个字段给的"。它**不**影响覆盖 ——
    /// 覆盖只回答"这一段样本我们拷过吗"。
    pub(super) position_source: PositionSource,
    /// 本块所属传输 epoch。捕获侧用它判断是否跨会话。
    pub(super) epoch: u64,
    /// 本块是否有权推进位置/写捕获（kPlaying、离线 pass 或有界起点窗口）。
    pub(super) transport_active: bool,
    /// 本块开始的传输会话是否是新 epoch。
    pub(super) new_epoch: bool,
    /// 域切换时整条 take 需要平移的帧数；0 = 不是域切换。
    pub(super) rebase_frames: i64,
    /// 本块生效的宿主时基（诊断）。
    pub(super) timebase: TimebaseKind,
}

/// 本块可用的宿主时间字段（全部可选；缺失时退回工程位置）。
///
/// 【为什么是 `pub(crate)`】它出现在 `ExtensionOwner::note_block_trace` 的参数里，
/// 而那个方法是 `pub(crate)`（`vst3` 的音频回调要调它）。私有类型泄进更宽可见度的
/// 签名是 `private_interfaces` 告警的来源；把类型提到同一档即可，不需要放宽方法。
#[derive(Clone, Copy, Default)]
pub(crate) struct StreamTime {
    pub(super) continuous: Option<i64>,
    pub(super) music: Option<f64>,
    pub(super) tempo: Option<f64>,
    pub(super) cycle_active: bool,
    pub(super) cycle_valid: bool,
    pub(super) cycle_start_music: f64,
    pub(super) cycle_end_music: f64,
}

impl StreamTime {
    pub(super) fn from_context(context: &crate::audio_abi::ProcessContext) -> Self {
        let state = context.state;
        let continuous_ok = state & crate::audio_abi::PROCESS_CONTEXT_CONTINUOUS_VALID != 0;
        let music_ok = state & crate::audio_abi::PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID != 0;
        let tempo_ok = state & crate::audio_abi::PROCESS_CONTEXT_TEMPO_VALID != 0;
        let cycle_ok = state & crate::audio_abi::PROCESS_CONTEXT_CYCLE_VALID != 0;
        Self {
            continuous: continuous_ok.then_some(context.continuous_time_samples),
            music: music_ok.then_some(context.project_time_music),
            tempo: (tempo_ok && context.tempo.is_finite() && context.tempo > 0.0)
                .then_some(context.tempo),
            cycle_active: state & crate::audio_abi::PROCESS_CONTEXT_CYCLE_ACTIVE != 0,
            cycle_valid: cycle_ok,
            cycle_start_music: context.cycle_start_music,
            cycle_end_music: context.cycle_end_music,
        }
    }
}

/// 输出源切换的去咔长度（帧）。48 kHz 下约 1.3 ms，足以消除阶跃。
///
/// 【它现在是**默认**宽度，不再是唯一宽度】异源切换（透传 ↔ 处理结果）只能淡入，
/// 1.3 ms 足够躲开阶跃；而**同一段素材的两种处理**（原始磁带 ↔ 处理结果）内容高度
/// 相关，正确处置是 20 ms 等功率交叉 —— 见 [`OUTPUT_WIDE_CROSSFADE_FRAMES`]。
pub(super) const OUTPUT_DECLICK_FRAMES: usize = 64;

/// 接缝状态保留的历史长度（帧）。48 kHz 下 20 ms。
///
/// 【为什么缓冲区要比默认淡化宽】要交叉淡化 20 ms，就必须手里有上一块输出的
/// 20 ms 尾巴。缓冲区按最大宽度开，具体一次淡化用多宽由 `fade_len` 决定 ——
/// 于是"64 帧淡入"与"960 帧交叉"共用同一份状态，音频线程仍然只写原子、不分配。
pub(super) const DECLICK_HISTORY_FRAMES: usize = 960;

/// `Raw ↔ Processed` 的交叉淡化宽度（帧）。48 kHz 下 20 ms。
///
/// 【为什么这一对要宽得多】两者是**同一段素材的两种处理**：原始磁带是捕获原样，
/// 处理结果是声码器输出。它们高度相关，所以等功率交叉**不会**叠出双声；而 1.3 ms
/// 只够躲开阶跃，躲不开"音色突变"的听感 —— 用户描述的"播放中途随机咔"里有一份
/// 正是渲染恰好在起播后 0.4~0.7 s 完成、把 Raw 换成 Processed 的那一下。
pub(super) const OUTPUT_WIDE_CROSSFADE_FRAMES: usize = 960;
/// 本流**第一块**的淡入宽度（帧）。48 kHz 下 5 ms。
///
/// 【为什么比 `OUTPUT_DECLICK_FRAMES` 宽】接缝淡化有"上一块输出的尾巴"作为旧侧，
/// 1.3 ms 就够把两条连续信号接上；首块**没有任何旧侧**，1.3 ms 在听感上仍是一次
/// 突变。5 ms 是"听不出渐强、但足以消掉阶跃"的那一档。
///
/// 【为什么不是 20 ms】首块之后马上就是真实内容；20 ms 的起播淡入会被听成
/// "开头有点闷"。
pub(super) const OUTPUT_FIRST_BLOCK_FADE_FRAMES: usize = 240;

/// 输出源的**内容家族**（接缝判据的输入，见 `ExtensionOwner::family`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ContentFamily {
    /// 宿主原声（透传 / 待合成 / ARA 未接管）。
    Passthrough,
    /// 原始捕获磁带（未处理）。
    Raw,
    /// 算出来的处理结果。
    Processed,
}

/// 一次接缝该怎么处理。
///
/// 【为什么是一个枚举而不是布尔】此前只有"淡入 / 交叉淡化"两档，于是"同一段素材的
/// 两种处理"被迫归到淡入 —— 那会在 1.3 ms 里把处理结果从零拉满，听感仍是一次突变。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Seam {
    /// 异源（任一侧是透传）：对当前结果线性淡入，宽度 `OUTPUT_DECLICK_FRAMES`。
    FadeIn,
    /// 同一族内的版本切换：交叉淡化，宽度 `OUTPUT_DECLICK_FRAMES`。
    Crossfade,
    /// 同一段素材的两种处理（Raw ↔ Processed）：等功率交叉，宽度 20 ms。
    WideCrossfade,
    /// 本流的**第一块**：没有"上一块输出的尾巴"，按零起点淡入。
    ///
    /// 【为什么需要它】"真首块原样输出"这条旧契约的理由是"从零淡入会改变宿主直通
    /// 的样本"。那只对**透传**成立（透传的样本就是宿主正在播的东西，它自己已经
    /// 连续）。当第一块是**我们算出来的内容**（Raw / Processed）时，它从哪一相位
    /// 开始是任意的 —— 从零直接跳到块内波形就是一次标准的起播咔。
    FirstBlock,
}

impl Seam {
    /// 诊断编码（进块账本）。**只增不改**：日志里已经出现过的数字必须保持含义。
    pub(super) fn code(self) -> crate::render::click::SeamCode {
        match self {
            Self::FadeIn => crate::render::click::SeamCode::FadeIn,
            Self::Crossfade => crate::render::click::SeamCode::Crossfade,
            Self::WideCrossfade => crate::render::click::SeamCode::WideCrossfade,
            Self::FirstBlock => crate::render::click::SeamCode::FadeIn,
        }
    }

    /// 这一次淡化是否用等功率权重（见 `arm_seam` 的说明）。
    pub(super) fn uses_equal_power(self) -> bool {
        matches!(self, Self::FadeIn | Self::WideCrossfade | Self::FirstBlock)
    }
}

/// 一条进程流的输出接缝状态（音频线程写，无锁）。
///
/// 【为什么按流分开】prefetch 与 realtime 的块会交替到达；共用一份 tail 会让
/// 一条流把另一条流的"上一块"当成自己的接缝。
pub(super) struct OutputDeclickState {
    /// 上一块输出的最后 `DECLICK_HISTORY_FRAMES` 帧（f32 位模式）。
    pub(super) tail: [Vec<AtomicU32>; 2],
    /// 源切换那一刻抓下来的旧信号，供跨块淡入淡出（前 `fade_len` 项有效）。
    pub(super) fade_from: [Vec<AtomicU32>; 2],
    pub(super) fade_pos: AtomicU32,
    pub(super) fade_active: AtomicBool,
    /// 本次淡化用多宽（帧）。0 = 还没设过（按 `OUTPUT_DECLICK_FRAMES` 处理）。
    pub(super) fade_len: AtomicU32,
    /// 本次淡化用哪条权重（`1` = 等功率，`0` = 线性）。见 `Seam::uses_equal_power`。
    ///
    /// 【为什么权重也要进状态】淡化可以跨块继续，而"两条不同源"与"同一份内容的
    /// 两次渲染"需要**不同**的数学。把结论存下来，跨块的每一帧才用同一条曲线。
    pub(super) fade_curve: AtomicU32,
    pub(super) last_token: AtomicU64,
    /// 本流是否已经输出过**至少一块**。
    ///
    /// 【为什么不能用 `primed_kind == 0` 表达"没见过"】`CaptureOutput::Untouched` 的
    /// 编码值**就是 0**，而 ARA 实例的输出仲裁恒为它。于是"从没见过"与"一直见到 0"
    /// 在旧判据下无法区分：ARA 的每一块都会被当成"真首块"。
    pub(super) primed: AtomicBool,
    /// 本流上一次输出源的**种类**（`primed` 为假时无意义）。
    ///
    /// 【为什么记种类而不是一个 bool】真首块没有"上一块输出的尾巴"，从零淡入会改变
    /// 宿主直通/录制监听的样本（测试与真实行为都要求第一块原样输出）。但"上一轮播放
    /// 收在 `Raw`、这一轮首块是 `Processed`"仍然是一次源切换 —— 那种场合必须去咔，
    /// 否则用户听到的正是"咔声位置不固定"。种类相同才允许首块原样通过。
    pub(super) primed_kind: AtomicU32,
}

impl Default for OutputDeclickState {
    fn default() -> Self {
        let zeros = || {
            (0..DECLICK_HISTORY_FRAMES)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>()
        };
        Self {
            tail: [zeros(), zeros()],
            fade_from: [zeros(), zeros()],
            fade_pos: AtomicU32::new(0),
            fade_active: AtomicBool::new(false),
            fade_len: AtomicU32::new(0),
            fade_curve: AtomicU32::new(0),
            last_token: AtomicU64::new(0),
            primed: AtomicBool::new(false),
            primed_kind: AtomicU32::new(0),
        }
    }
}

/// 一块分层拷贝的**覆盖读数**（音频线程写、同块读走）。
///
/// 【为什么用原子而不是局部变量】写它的地方在 `snapshot.rs`（只知道"哪一帧属于
/// 快照"），读它的地方在 `apply_output_declick`（只知道"这一块输出了什么"）。
/// 两者之间隔着一次跨模块调用，而那块缓冲的生命周期不允许我们把闭包传出去 ——
/// 一个每流一格、每块重置的原子单元是最小耦合。
#[derive(Default)]
pub(super) struct LayeredProbeCell {
    /// 本块里**不属于我们**的帧数。
    pub(super) uncovered: AtomicU32,
    /// 那些帧上宿主输入的峰值（`NaN` 位模式 = 本块没有未覆盖帧）。
    pub(super) input_peak: AtomicU32,
    /// 覆盖边（覆盖 ↔ 未覆盖的翻转点）是否落在本块内。
    pub(super) edge: AtomicBool,
    /// 本块写过这份读数（否则读走的是上一块的值）。
    pub(super) valid: AtomicBool,
}
