//! 传输会话（TransportSession）：一次“播放”从开始到停止的显式生命周期。
//!
//! 旧代码用四个互相独立的信号推断“在不在播”：kPlaying、位置有没有动、输入内容
//! 有没有变、prefetch 播放位。停播后宿主仍送块时，这四个信号可以同时给出“在走”的
//! 假象。这里把它们收敛成一个状态机：只有 Starting（有界起播窗口）或 Rolling
//! 允许时间前进；Paused 与 Stopped 一律冻结。
//!
//! 【本轮的职责收缩】它只回答“这是不是同一次播放、何时冻结”。坐标域由
//! `HostClock` 回答，离线/冻结预览由 `ProcessingPass` 回答。三者顺序固定：
//! ProcessingPass -> HostClock -> TransportSession。
//!
//! 【为什么 Starting -> Rolling 不递增 epoch】VEGAS Event FX 的域切换恰好发生在
//! kPlaying 上升沿：pre-roll 的绝对域块先打开 Starting 窗口，下一块 kPlaying 置位
//! 并切到相对域。若这里把上升沿判成新会话，域切换就永远没有机会执行 —— 现场
//! 那 0.68 秒绝对域首段正是这样留下的。

/// 传输会话的相位。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SessionPhase {
    #[default]
    Idle,
    /// 已经收到块，但宿主的 kPlaying 还没置位；只允许一个有界的起点窗口。
    Starting,
    /// 宿主明确在播放。
    Rolling,
    /// kPlaying 已落，但捕获仍 armed：可以续录，位置必须冻结。
    Paused,
    /// 用户停止 / 离线 pass 结束。
    Stopped,
    /// 离线 pass 的本地游标（由 ProcessingPass 驱动，保留编码兼容）。
    OfflineRolling,
}

impl SessionPhase {
    pub fn code(self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Starting => 1,
            Self::Rolling => 2,
            Self::Paused => 3,
            Self::Stopped => 4,
            Self::OfflineRolling => 5,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Idle),
            1 => Some(Self::Starting),
            2 => Some(Self::Rolling),
            3 => Some(Self::Paused),
            4 => Some(Self::Stopped),
            5 => Some(Self::OfflineRolling),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn is_advancing(self) -> bool {
        matches!(self, Self::Starting | Self::Rolling | Self::OfflineRolling)
    }

    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Rolling => "rolling",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::OfflineRolling => "offline",
        }
    }
}

/// 宿主喂给我们的东西属于哪一类时间轴。
///
/// 【为什么要分两类】"宿主没报传输"有两种完全不同的成因：
///
///   * **起播前预热**：真的还没开始，窗口一关就该冻结；
///   * **钉住的时间轴**：宿主在渲染/离线出图，它不报 kPlaying、也不动位置，却持续
///     把彼此不同的音频按块喂进来。那不是预热，是一条**时间轴** —— 该一直推进到它
///     改口为止。
///
/// 把后者当前者，窗口就每秒重开一次，每次重开都伴随一个新 epoch，而新 epoch 会把
/// 位置重锚回 `raw_start`（= 0）：现场那 31 次"每 1.024 s 重来一遍"正是这样来的。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FeedKind {
    /// 宿主报传输（kPlaying / offline），或它开始报新位置：一切以宿主为准。
    #[default]
    Transport,
    /// 宿主不报传输、却持续喂彼此不同的音频：这是一条钉住的时间轴。
    Pinned,
}

impl FeedKind {
    pub fn code(self) -> u8 {
        match self {
            Self::Transport => 0,
            Self::Pinned => 1,
        }
    }

    pub fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Pinned,
            _ => Self::Transport,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::Pinned => "pinned",
        }
    }
}

/// "第一块 raw 未知"的哨兵。
///
/// 【为什么不用 0】工程 0 是合法起点；用 0 当哨兵会让「首块在工程 0」被读成「未知」。
pub const NO_RAW: i64 = i64::MIN;

/// 工程位置的来源（环境变量 `HFS_CAPTURE_MAP`）。
///
/// 【为什么这是一个可切换的判据】"喂入域的第几帧对应工程第几帧"是这一轮唯一改变
/// 声音位置的决定。它有三种历史答案，而它们在产物上**完全同形**（都是"某段音频
/// 出现在错误的时间上"）：
///
///   * `clip`（默认，本轮）：由捕获片段在工程里的位置决定 —— 唯一有证据的答案；
///   * `origin`（第四轮）：喂入域的 0 就是工程 0；
///   * `raw`（第三轮）：喂入域的 0 是首块 `projectTimeSamples`。
///
/// 留一个开关，现场就能**一键对照**三种映射，不必重新出包 —— 与 `HFS_PINNED_FEED` /
/// `HFS_RENDER_RUN` / `HFS_HOST_WAIT_MS` 同一条纪律。判读方法：把同一段素材渲染三次，
/// 只有正确的那个映射会给出零偏移。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CaptureMapMode {
    /// 工程位置由捕获片段的位置决定（`project = clip_start + (tape - clip_source_start)`）。
    #[default]
    Clip,
    /// 喂入域的 0 = 工程 0（第四轮行为）。
    Origin,
    /// 喂入域的 0 = 首块 `projectTimeSamples`（第三轮行为）。
    Raw,
}

impl CaptureMapMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clip => "clip",
            Self::Origin => "origin",
            Self::Raw => "raw",
        }
    }
}

/// 当前的工程位置来源。
///
/// 【为什么 `HFS_PINNED_ORIGIN` 还留着】它是第三轮那个开关的名字。现场脚本与排查
/// 笔记里都写着它，直接删掉会让"照着旧笔记复现"变成一次静默的无效操作。这里把它
/// 读成 `HFS_CAPTURE_MAP=raw` 的别名，并在日志里记一条 deprecation。
pub fn capture_map_mode() -> CaptureMapMode {
    static MODE: std::sync::OnceLock<CaptureMapMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let explicit = std::env::var("HFS_CAPTURE_MAP").ok();
        let mode = match explicit.as_deref().map(str::trim) {
            Some("origin") => CaptureMapMode::Origin,
            Some("raw") => CaptureMapMode::Raw,
            Some("clip") => CaptureMapMode::Clip,
            Some(other) => {
                log::warn!(
                    "[capture] unknown HFS_CAPTURE_MAP={other}; using clip (the mapping follows the clip)"
                );
                CaptureMapMode::Clip
            }
            None => {
                if std::env::var("HFS_PINNED_ORIGIN").is_ok_and(|value| value == "raw") {
                    log::warn!(
                        "[capture] HFS_PINNED_ORIGIN is deprecated; it now means HFS_CAPTURE_MAP=raw"
                    );
                    CaptureMapMode::Raw
                } else {
                    CaptureMapMode::Clip
                }
            }
        };
        log::info!("[capture] project mapping mode: {}", mode.as_str());
        mode
    })
}

/// 一条喂入流**自己的**走带位置。只有一个域。
///
/// 【为什么它必须与工程位置分开】此前同一个 `i64` 同时被当成"喂入流的第几帧"与
/// "工程第几帧"，于是"钉住流的 0 是不是工程 0"这个问题只能靠一个环境变量回答
/// （见 `CaptureMapMode`）。答案是两个都不是：喂入域的 0 是**这条流**的
/// 第一帧，它在工程里的位置由 `project_map::ProjectMapping` 给出。
///
/// 【为什么不在这里存 origin】存了它，下一个读代码的人就会拿它当工程起点 ——
/// 那正是上一轮的错误。本类型只回答"这条流已经喂进来多少帧"，一个纯粹的计数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapeCursor {
    /// 本遍已经喂进来的帧数（从 0 起，**提级不重置、不复用**）。
    pub frames: u64,
    /// 首块宿主读数（只作诊断；`NO_RAW` = 未知）。
    pub first_raw: i64,
}

impl Default for TapeCursor {
    /// 【为什么手写而不是 derive】`derive(Default)` 会给 `first_raw` 一个 0，而 0 是
    /// **合法的宿主读数**（工程 0）。用 0 当"未知"会让"首块在工程 0"被读成"未知"，
    /// 于是 `note_first_raw` 永远记不下第一块 —— 诊断值静默变成假的。
    fn default() -> Self {
        Self {
            frames: 0,
            first_raw: NO_RAW,
        }
    }
}

impl TapeCursor {
    /// 归零（传输事件 / 复位）。
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 记下首块的宿主读数（只诊断，不参与换算）。
    pub fn note_first_raw(&mut self, raw: i64) {
        if self.first_raw == NO_RAW {
            self.first_raw = raw;
        }
    }

    /// 本块之后的走带位置（喂入域）。
    pub fn advance(&mut self, frames: u32) -> u64 {
        self.frames = self.frames.saturating_add(u64::from(frames));
        self.frames
    }
}

/// 一个传输会话的纯状态。
#[derive(Clone, Copy, Debug)]
pub struct TransportSession {
    pub epoch: u64,
    pub phase: SessionPhase,
    pub session_frames: u64,
    pub last_playing: bool,
    pub max_start_window_frames: u64,
    /// 这一块属于哪一类时间轴。
    pub feed: FeedKind,
    /// 起点窗口是否**已经用掉**（一次性）。只有真正的传输事件能让它复原。
    pub window_spent: bool,
    /// "raw 被钉住 + 没有 kPlaying"期间收到的、彼此不同的有信号帧数。
    /// 到 [`FEED_PROMOTE_SEC`] 就把窗口提级成 [`FeedKind::Pinned`]。
    pub pinned_signal_frames: u64,
    /// 这条喂入流自己的走带位置（见 [`TapeCursor`]）。
    ///
    /// 【为什么它必须跨块存活】本结构体每块都是临时重建的（见 playback.rs 的构造），
    /// 所以它像 feed / window_spent / pinned_signal_frames 一样持久化在流几何里，
    /// 在构造时读入、observe 之后写回。
    pub tape: TapeCursor,
}

impl Default for TransportSession {
    fn default() -> Self {
        Self {
            epoch: 0,
            phase: SessionPhase::Idle,
            session_frames: 0,
            last_playing: false,
            max_start_window_frames: 48_000,
            feed: FeedKind::Transport,
            window_spent: false,
            pinned_signal_frames: 0,
            tape: TapeCursor::default(),
        }
    }
}

/// 一只钉住的喂入流要跑多久才被认出来（秒）。
///
/// 【为什么是 0.5】足以排除"预卷的第一块"（它通常只有一块），又远小于任何一次真实
/// 渲染。它与起点窗口（1 秒）的关系是"窗口的一半"：识别必须在窗口用掉**之前**发生，
/// 否则位置已经冻结过一段，再推进就出现了洞。
pub const FEED_PROMOTE_SEC: f64 = 0.5;

/// 钉住喂入流的**行为**开关（环境变量 `HFS_PINNED_FEED=0` 关闭）。
///
/// 【为什么要有开关】这条判据改变的是"位置由谁决定"：开着时我们不再把位置拉回锚点，
/// 输出随宿主喂入的顺序走。判据本身有三道（pinned + 有信号 + 未布防），但"某个宿主
/// 的块序列恰好长得像钉住"这种误判在离线环境里无法穷举。留一个开关，现场就能一键退回
/// 旧行为，不必重新出包 —— 与 `HFS_RENDER_RUN` 同一条纪律。
///
/// 【为什么用 OnceLock 而不是每次读环境变量】它在音频线程上被调用；getenv 只在进程内
/// 第一次发生，之后是一次原子读。
pub fn pinned_feed_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("HFS_PINNED_FEED").is_ok_and(|value| value == "0" || value == "false")
    })
}

/// 本块看到的传输事实。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionInput {
    pub playing: bool,
    pub playing_rising: bool,
    pub recording_rising: bool,
    pub host_reset: bool,
    pub offline: bool,
    pub frames: u32,
    /// 宿主这一块报的位置与上一块相同（"钉住"）。
    pub pinned: bool,
    /// 这一块有**彼此不同**的非静音内容（不是同一份缓冲的重复投递）。
    pub signal: bool,
    /// 捕获已武装：录制中绝不发明时间（录制的时间只能来自宿主）。
    pub armed: bool,
    /// 宿主这一块报的原始位置（帧）。只用于给 [`TapeCursor`] 记首块诊断值。
    pub raw_start: i64,
    /// 本块之前这条流已经有块了（`!had` 的反面）。
    ///
    /// 【为什么要显式传】"首块"是喂入流起点的唯一可用证据：首块上 `pinned` 天然为假
    /// （没有上一块可比），而漏掉它会让整条时间轴晚一块才开始。
    pub had_previous: bool,
}

/// 状态机结论。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionDecision {
    pub phase: SessionPhase,
    pub new_epoch: bool,
    /// 本块是否有权推进位置 / 写捕获。
    pub transport_active: bool,
    /// 起点窗口还开着：窗口内的块可以按块长前进，即使 kPlaying 还是 0。
    pub start_window_open: bool,
    /// 本块之后的喂入流类别（用于诊断与 `playback` 侧的决策）。
    pub feed: FeedKind,
    /// 本块刚刚完成了"窗口 → 钉住的喂入流"的提级。
    pub promoted: bool,
    /// 本块属于「钉住候选」，但还没到 0.5 s 提级。
    ///
    /// 【为什么候选也要算进喂入域】候选块已经按块长推进位置了（起点窗口）。若只有
    /// 提级之后的块走 TapeCursor，前 0.5 s 就会留在 raw 域里，接缝处出现一个洞。
    pub candidate: bool,
    /// 本块**之前**已经喂进来的帧数（喂入域，含起点窗口里的块）。
    ///
    /// 【为什么不再附带一个 origin】工程位置**不在这里**：它由
    /// `project_map::ProjectMapCell` 从磁带帧号算出来。把 origin 放在这里，下一个读
    /// 代码的人就会拿它当工程起点 —— 那正是上一轮的错误。
    pub tape_frames_before: u64,
}

impl TransportSession {
    #[cfg(test)]
    pub fn with_rate(rate: f64) -> Self {
        let mut session = Self::default();
        if rate.is_finite() && rate > 0.0 {
            session.max_start_window_frames = (rate * 1.0).round().max(1.0) as u64;
        }
        session
    }

    /// 起点窗口对应多少帧（= 1 秒 × 采样率；见 `with_rate`）。
    fn window_frames(&self) -> u64 {
        self.max_start_window_frames.max(1)
    }

    /// 提级成"钉住的喂入流"需要多少帧有信号内容。
    ///
    /// 【为什么取窗口的一半】见 [`FEED_PROMOTE_SEC`]：识别必须在窗口用掉之前完成。
    fn promote_frames(&self) -> u64 {
        ((self.window_frames() as f64) * FEED_PROMOTE_SEC)
            .round()
            .max(1.0) as u64
    }

    /// 【规则 1】真正的传输事件：宿主在报播放 / 离线，或它开始报新位置。
    ///
    /// 它同时复原起点窗口与提级计数 —— "钉住的时间轴"到这一刻结束了。
    fn note_transport(&mut self) {
        self.feed = FeedKind::Transport;
        self.window_spent = false;
        self.pinned_signal_frames = 0;
        // 【为什么连时间轴一起归零】规则 1 的语义是「钉住流到此结束」。留着 frames 会
        // 让下一条钉住流从上一条的末尾接着数 —— 一次渲染就凭空多出几十秒的偏移。
        self.tape.reset();
    }

    /// 前进一步。host_reset = 宿主位置倒退或跳变；它是新会话的强证据之一。
    pub fn observe(&mut self, input: SessionInput) -> SessionDecision {
        // 【本块之前已经喂进来多少帧】必须在任何分支之前读：下面 `note_transport` 会
        // 在传输事件上把它清零。
        let tape_frames_before = self.tape.frames;
        let mut candidate = false;
        let mut new_epoch = false;
        let mut transport_active = false;
        let mut start_window_open = false;
        let mut promoted = false;

        // 【规则 1：传输优先】宿主报播放、报离线，或开始报**新位置**，都说明它在
        // 自己的时间轴上走 —— 钉住的喂入流到此结束。
        if input.playing || input.offline || !input.pinned {
            self.note_transport();
        }

        if input.offline {
            if self.phase != SessionPhase::OfflineRolling {
                self.epoch = self.epoch.saturating_add(1);
                self.phase = SessionPhase::OfflineRolling;
                self.session_frames = 0;
                new_epoch = true;
            }
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
            self.last_playing = true;
            return SessionDecision {
                phase: self.phase,
                new_epoch,
                transport_active: true,
                start_window_open: false,
                feed: self.feed,
                promoted,
                candidate: false,
                tape_frames_before,
            };
        }

        if input.playing {
            match self.phase {
                // 真正的新起播：从 Idle / Paused / Stopped 进入 Rolling。
                SessionPhase::Idle | SessionPhase::Paused | SessionPhase::Stopped => {
                    self.epoch = self.epoch.saturating_add(1);
                    self.session_frames = 0;
                    self.phase = SessionPhase::Rolling;
                    new_epoch = true;
                }
                // 【关键】pre-roll 与正文是同一次演奏：不新开 epoch。
                SessionPhase::Starting => {
                    self.phase = SessionPhase::Rolling;
                }
                SessionPhase::Rolling | SessionPhase::OfflineRolling => {
                    self.phase = SessionPhase::Rolling;
                }
            }
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
            transport_active = true;
        } else if self.feed == FeedKind::Pinned {
            // 【规则 2：钉住的喂入流一直走到宿主改口】
            //
            // 不新开 epoch、不重开窗口、不动锚点：位置由调用方按块长继续往下数。
            // 这就是"一次渲染只有一个 pass"的全部秘密 —— 位置重锚只可能由新 epoch
            // 触发，而这条分支一个都不产生。
            self.phase = SessionPhase::Rolling;
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
            transport_active = true;
        } else if input.playing_rising || input.recording_rising {
            if self.phase != SessionPhase::Starting {
                self.epoch = self.epoch.saturating_add(1);
                self.session_frames = 0;
                self.phase = SessionPhase::Starting;
                new_epoch = true;
            }
            // 【规则 4：窗口是一次性的】用掉之后只有规则 1 能让它复原。
            start_window_open = !self.window_spent && self.session_frames < self.window_frames();
            transport_active = start_window_open;
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
        } else if self.last_playing || self.phase == SessionPhase::Rolling {
            // 【停播优先于窗口】kPlaying 从 1 落到 0：立即冻结。绝不再用内容变化/静音
            // 重新解释，也不再给起点窗口续命（那不是"还没开始"，是"已经停了"）。
            self.phase = SessionPhase::Paused;
        } else if !self.window_spent
            && self.phase == SessionPhase::Starting
            && self.session_frames < self.window_frames()
        {
            // 起播窗口：宿主还来不及置 kPlaying，但窗口有严格上限。
            start_window_open = true;
            transport_active = true;
            self.session_frames = self.session_frames.saturating_add(u64::from(input.frames));
        } else if self.phase != SessionPhase::Idle {
            self.phase = SessionPhase::Paused;
        }

        // 【规则 3：窗口 → 钉住的喂入流】
        //
        // 【判据为什么是这三条】`pinned` 说明宿主没在报位置；`signal` 说明它喂的确实是
        // 彼此不同的音频（排除 FL Studio 式的"停播反复送同一份缓冲"）；`!armed` 说明
        // 用户没有在录（录制的时间只能来自宿主，插件绝不发明）。三条同时成立，才承认
        // "这是一条钉住的时间轴"。
        // 【首块也是这条喂入流的第一帧】`pinned` 在首块上天然为假（没有上一块可比），
        // 但"宿主没在播放 + 块里有内容 + 没在录"已经足以说明它是一条喂入流的开头：
        // 真实播放的 kPlaying 会同时到达，那时 `start_window_open` 根本不成立。
        // 少算这一块，整条时间轴就从第 1 块之后才开始 —— 位置在提级处跳回 0。
        if start_window_open
            && (input.pinned || !input.had_previous)
            && input.signal
            && !input.armed
        {
            candidate = true;
            self.pinned_signal_frames = self
                .pinned_signal_frames
                .saturating_add(u64::from(input.frames));
            if self.pinned_signal_frames >= self.promote_frames() {
                self.feed = FeedKind::Pinned;
                self.window_spent = true;
                self.phase = SessionPhase::Rolling;
                promoted = true;
                transport_active = true;
                start_window_open = false;
            }
        }
        // 【喂入域的推进】候选块与钉住块都算 —— 它们喂进来的就是这条时间轴的连续内容。
        // 提级本身不改变 origin：一次渲染只有一个起点，提级只是「认出它是时间轴」。
        // 【喂入域的推进】候选块与钉住块都算 —— 它们喂进来的就是这条时间轴的连续内容。
        // 提级本身不改变走带位置：一次渲染只有一个起点，提级只是「认出它是时间轴」。
        // 计数从**第一块**就开始，于是提级那一刻位置是连续的（此前候选窗口的 0.489 s
        // 会被算成一次硬跳变）。
        if candidate || self.feed == FeedKind::Pinned {
            self.tape.note_first_raw(input.raw_start);
            self.tape.advance(input.frames);
        }

        // 【窗口用掉】跑到上限就是"用掉了"：此后不再重开，直到规则 1 复原它。
        if self.session_frames >= self.window_frames() && self.feed == FeedKind::Transport {
            self.window_spent = true;
        }

        if input.host_reset && self.phase == SessionPhase::Rolling {
            self.epoch = self.epoch.saturating_add(1);
            self.session_frames = 0;
            self.window_spent = true;
            new_epoch = true;
        }
        self.last_playing = input.playing;
        SessionDecision {
            phase: self.phase,
            new_epoch,
            transport_active,
            start_window_open,
            feed: self.feed,
            promoted,
            candidate,
            tape_frames_before,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一块"宿主报了播放"的普通块。
    fn input(playing: bool, rising: bool, frames: u32) -> SessionInput {
        SessionInput {
            playing,
            playing_rising: rising,
            recording_rising: false,
            host_reset: false,
            offline: false,
            frames,
            // 普通块：位置在动（不钉住）、有内容、没在录。
            pinned: false,
            signal: true,
            armed: false,
            raw_start: 0,
            had_previous: true,
        }
    }

    /// 一块"宿主不报传输、位置钉住、但喂来彼此不同的音频"的块。
    fn pinned_block(frames: u32) -> SessionInput {
        SessionInput {
            pinned: true,
            ..input(false, false, frames)
        }
    }

    #[test]
    fn k_playing_zero_freezes_even_when_input_keeps_arriving() {
        let mut session = TransportSession::with_rate(48_000.0);
        session.observe(input(true, false, 512));
        let decision = session.observe(input(false, false, 512));
        assert_eq!(decision.phase, SessionPhase::Paused);
        assert!(!decision.transport_active);
        assert!(!decision.start_window_open);
    }

    #[test]
    fn a_new_playing_edge_from_paused_is_a_new_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(true, true, 512));
        assert!(first.new_epoch);
        session.observe(input(false, false, 512));
        let second = session.observe(input(true, true, 512));
        assert!(second.new_epoch, "每次重新起播都必须开新 epoch");
        assert_ne!(second.phase, SessionPhase::Paused);
    }

    /// 问题 1 的直接回归：pre-roll 打开 Starting，kPlaying 上升沿进入 Rolling
    /// **不得**递增 epoch，否则 HostClock 会把它当成新会话而跳过域切换。
    #[test]
    fn starting_to_rolling_keeps_the_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(false, true, 848));
        assert_eq!(first.phase, SessionPhase::Starting);
        assert!(first.new_epoch);
        let second = session.observe(input(true, true, 848));
        assert_eq!(second.phase, SessionPhase::Rolling);
        assert!(!second.new_epoch);
        assert!(second.transport_active);
    }

    #[test]
    fn the_start_window_is_bounded_to_one_second() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(false, true, 480));
        assert!(first.start_window_open);
        let mut last = first;
        for _ in 0..120 {
            last = session.observe(input(false, false, 480));
        }
        assert!(!last.transport_active, "超过 1 秒的 kPlaying=0 必须冻结");
    }

    #[test]
    fn offline_blocks_always_advance_and_create_one_epoch() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut offline = input(false, false, 512);
        offline.offline = true;
        let first = session.observe(offline);
        assert!(first.new_epoch);
        assert!(first.transport_active);
        let second = session.observe(offline);
        assert!(!second.new_epoch);
        assert_eq!(second.phase, SessionPhase::OfflineRolling);
    }

    // ── 钉住的喂入流（VEGAS 渲染复读的回归）──────────────────────────

    /// **回归：一条钉住的时间轴只被认出一次，而且从此不再重锚。**
    ///
    /// 【现场原形】VEGAS `Render As` 的产物是"前一秒复读 31 遍"。根因不是渲染 run，
    /// 而是起点窗口每秒重开一次 ⇒ 每块都开新 epoch ⇒ 位置每 1.024 s 被拉回 raw_start。
    #[test]
    fn a_pinned_feed_is_promoted_once_and_never_re_anchors() {
        let mut session = TransportSession::with_rate(48_000.0);
        // 4 秒、48 kHz、块长 960（= 20 ms），与 VEGAS 现场同量级。
        let mut decisions = Vec::new();
        decisions.push(session.observe(pinned_block_with_rising(960)));
        for _ in 0..199 {
            decisions.push(session.observe(pinned_block(960)));
        }
        let promoted = decisions.iter().filter(|d| d.promoted).count();
        assert_eq!(promoted, 1, "提级只许发生一次");
        let epochs = decisions.iter().filter(|d| d.new_epoch).count();
        assert_eq!(epochs, 1, "整段喂入只许有一个 epoch");
        assert!(
            decisions.iter().skip(1).all(|d| d.transport_active),
            "提级之后每一块都必须有权推进位置，否则渲染会出现洞"
        );
        assert_eq!(session.feed, FeedKind::Pinned);
        assert_eq!(session.session_frames, 200 * 960);
    }

    /// **回归：宿主重新报传输 ⇒ 回到 Transport。**
    #[test]
    fn a_pinned_feed_ends_when_the_host_reports_transport_again() {
        let mut session = TransportSession::with_rate(48_000.0);
        session.observe(pinned_block_with_rising(960));
        for _ in 0..40 {
            session.observe(pinned_block(960));
        }
        assert_eq!(session.feed, FeedKind::Pinned);
        let decision = session.observe(input(true, true, 960));
        assert_eq!(
            decision.feed,
            FeedKind::Transport,
            "宿主报播放就必须以它为准"
        );
        assert!(!decision.promoted);
    }

    /// **回归：起点窗口是一次性的。**
    #[test]
    fn the_start_window_is_spent_once() {
        let mut session = TransportSession::with_rate(48_000.0);
        let first = session.observe(input(false, true, 960));
        assert!(first.start_window_open);
        // 2 秒的 kPlaying=0：信号为假（静音），所以不会被提级成喂入流。
        let mut silent = pinned_block(960);
        silent.signal = false;
        // 前 1 秒是**同一个**窗口（50 块 × 960 = 48 000 帧），之后必须彻底关上。
        for _ in 0..49 {
            assert!(session.observe(silent).start_window_open);
        }
        assert!(session.window_spent, "跑满 1 秒就是用掉了");
        for _ in 0..51 {
            assert!(!session.observe(silent).start_window_open);
        }
        assert!(session.window_spent);
        assert_eq!(session.epoch, 1, "整段只许有一个 epoch");
    }

    /// **回归：输入监听不算时间轴。**
    ///
    /// 【为什么】宿主没播、位置钉住、内容全静音时，那块音频不是"播放时间" —— 它是
    /// 输入监听的回路。把它当成时间轴就等于凭空发明时间。
    #[test]
    fn input_monitoring_never_promotes() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut silent = pinned_block(960);
        silent.signal = false;
        session.observe(pinned_block_with_rising(960));
        for _ in 0..200 {
            let decision = session.observe(silent);
            assert!(!decision.promoted);
        }
        assert_eq!(session.feed, FeedKind::Transport);
    }

    /// **回归：录制中绝不提级。**
    ///
    /// 【为什么】录制的时间只能来自宿主。提级会让"宿主没播"的那一段变成会走动的时间，
    /// 用户录到的就是一段凭空的静音。
    #[test]
    fn a_recording_never_promotes() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut armed = pinned_block(960);
        armed.armed = true;
        let mut first = armed;
        first.playing_rising = true;
        session.observe(first);
        let mut advanced_frames = 0;
        for _ in 0..200 {
            let decision = session.observe(armed);
            assert!(!decision.promoted);
            if decision.transport_active {
                advanced_frames += 960;
            }
        }
        assert_eq!(session.feed, FeedKind::Transport);
        assert!(
            advanced_frames <= 48_000,
            "武装时只许走一个 1 秒窗口，之后必须冻结"
        );
    }

    /// **回归：停播反复送同一份内容不是时间轴（FL Studio 形状）。**
    #[test]
    fn repeated_content_is_not_a_feed() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut repeated = pinned_block(960);
        repeated.signal = false;
        session.observe(pinned_block_with_rising(960));
        for _ in 0..200 {
            assert!(!session.observe(repeated).promoted);
        }
        assert_eq!(session.feed, FeedKind::Transport);
    }

    /// 第一块"钉住但有信号"，并且合成一个起播上升沿（与 `playback` 侧一致）。
    fn pinned_block_with_rising(frames: u32) -> SessionInput {
        SessionInput {
            playing_rising: true,
            ..pinned_block(frames)
        }
    }

    /// **不变量：钉住喂入流的走带位置从 0 起，宿主播放头只作诊断。**
    ///
    /// 【为什么钉死】VEGAS Render As 把 `projectTimeSamples` 留在**最后播放/停止**的
    /// 位置（现场 r3 = 893720 = 18.62 s）。它既不是工程起点，也不参与换算 —— 工程
    /// 位置由 `project_map` 从片段位置给出（见 `project_map::tests`）。
    #[test]
    fn a_pinned_feed_starts_at_tape_zero_regardless_of_the_host_playhead() {
        const RAW: i64 = 893_720;
        let mut session = TransportSession::with_rate(48_000.0);
        let mut first = pinned_block(1200);
        // 起点窗口由"kPlaying 还来不及置位、但音频已经在流动"打开（见 playback.rs）。
        first.playing_rising = true;
        first.raw_start = RAW;
        let decision = session.observe(first);
        assert!(decision.candidate, "第一块必须被认成钉住候选");
        assert_eq!(decision.tape_frames_before, 0);
        assert_eq!(decision.feed, FeedKind::Transport, "0.5 s 之内还没提级");

        let mut promoted = None;
        for index in 1..=40u64 {
            let mut next = pinned_block(1200);
            next.raw_start = RAW;
            let decision = session.observe(next);
            if decision.promoted {
                promoted = Some((index, decision));
                break;
            }
        }
        let (index, decision) = promoted.expect("钉住喂入流必须在 0.5 s 内提级");
        assert_eq!(decision.feed, FeedKind::Pinned);
        assert_eq!(decision.tape_frames_before, index * 1200);
        assert_eq!(
            session.tape.frames,
            (index + 1) * 1200,
            "提级之后走带位置就是已喂帧数"
        );
        assert_eq!(session.tape.first_raw, RAW);
    }

    /// **不变量：提级那一刻走带位置是连续的。**
    ///
    /// 【为什么这条是"起播 1 s 内错位"的接线端】此前候选块只计进 `frames` 而不被
    /// 认成喂入域，于是第一次 `feed begin` 报出的位置是 0.489 s（23 488 帧）而不是 0
    /// —— 提级那一刻位置从宿主域跳到自造域的 0.489 s，一次 0.5 s 的硬跳变。
    #[test]
    fn tape_cursor_starts_at_zero_after_promotion() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut first = pinned_block(960);
        first.playing_rising = true;
        let mut previous = session.observe(first);
        assert_eq!(previous.tape_frames_before, 0);
        let mut advanced = 960_u64;
        for _ in 0..200 {
            let decision = session.observe(pinned_block(960));
            // 相邻两块的位置差恒等于块长 —— 提级那一块也不例外。
            assert_eq!(
                decision.tape_frames_before,
                advanced,
                "走带位置必须逐块连续，提级不得跳变"
            );
            advanced += 960;
            previous = decision;
        }
        assert_eq!(previous.feed, FeedKind::Pinned);
        assert_eq!(session.tape.frames, 201 * 960);
    }

    /// **不变量：宿主一报传输，喂入时间轴立刻结束并归零。**
    #[test]
    fn a_pinned_feed_ends_when_the_host_reports_transport() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut first = pinned_block(1200);
        first.playing_rising = true;
        first.raw_start = 5_000;
        session.observe(first);
        for _ in 0..30 {
            let mut block = pinned_block(1200);
            block.raw_start = 5_000;
            session.observe(block);
        }
        assert_eq!(session.feed, FeedKind::Pinned);
        assert!(session.tape.frames > 0);

        let mut playing = input(true, true, 1200);
        playing.raw_start = 999_000;
        let decision = session.observe(playing);
        assert_eq!(decision.feed, FeedKind::Transport);
        assert_eq!(session.tape.frames, 0, "传输事件必须把走带位置归零");
        assert_eq!(session.tape.first_raw, NO_RAW);
    }

    /// **不变量：候选期间 raw 变了，就不是同一条钉住流。**
    ///
    /// 【为什么它必须重开】宿主在预卷之后把播放头移到渲染起点，是正常形状。把两段
    /// 接成一条，位置就从预卷的位置接着数 —— 那正是"偏移一个预卷长度"的来源。
    #[test]
    fn a_pre_roll_that_changes_raw_restarts_the_tape_cursor() {
        let mut session = TransportSession::with_rate(48_000.0);
        let mut first = pinned_block(1200);
        first.playing_rising = true;
        first.raw_start = 100;
        session.observe(first);
        let mut next = pinned_block(1200);
        next.raw_start = 100;
        session.observe(next);
        assert_eq!(session.tape.frames, 2_400);

        let mut moved = pinned_block(1200);
        moved.raw_start = 48_000;
        // 真实判据是「本块 raw 与上一块相同」；raw 变了就不是钉住块。
        moved.pinned = false;
        let decision = session.observe(moved);
        assert!(!decision.candidate, "raw 变了就不是同一条钉住流");
        assert_eq!(session.tape.frames, 0);
        assert_eq!(
            decision.tape_frames_before, 2_400,
            "本块之前的帧数是读数，不因归零而变"
        );
    }

    /// `TapeCursor` 只回答"已经喂进来多少帧"，没有第二个域。
    #[test]
    fn the_tape_cursor_counts_frames_and_nothing_else() {
        let mut cursor = TapeCursor::default();
        assert_eq!(cursor.frames, 0);
        assert_eq!(cursor.first_raw, NO_RAW);
        cursor.note_first_raw(893_720);
        cursor.note_first_raw(1_000);
        assert_eq!(cursor.first_raw, 893_720, "首块读数只记一次");
        assert_eq!(cursor.advance(960), 960);
        assert_eq!(cursor.advance(960), 1_920);
        cursor.reset();
        assert_eq!(cursor.frames, 0);
        assert_eq!(cursor.first_raw, NO_RAW);
    }
}
