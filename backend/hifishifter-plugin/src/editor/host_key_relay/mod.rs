//! 宿主专属快捷键的让路策略（Host Key Relay）。
//!
//! 【要解决的问题】在不支持 ARA 的宿主里，插件拿不到 `ARAPlaybackControllerInterface`，
//! 于是 `playback.toggle` / `playback.stop` 必然被能力矩阵拒绝，只弹一句
//! 「当前宿主没有提供这个接口」。可用户按 Space 的本意就是**让宿主播放** —— 而键盘
//! 消息不向父窗口冒泡，宿主根本收不到这个键。结果是死角：插件没能力做，宿主没收到键。
//!
//! 【本模块只负责"怎么让、让成了没有"】"这一次按键该不该让"由前端判（能力位、键位、
//! 焦点域都在那边，且能穷举测试）。两边各答一半，接口只有三个字段，判据不重复。
//!
//! 【三层：Policy / Primitive / Fidelity】
//!
//! - **Policy**（本文件）：让哪一族、先验证还是直接让、投几次、什么时候说话。
//!   **不许出现 `#[cfg]`** —— 换平台只换 Primitive，Policy 一行不改。
//! - **Primitive**（`windows.rs` / 未来的 `macos.rs`、`linux.rs`）：真正把键交回宿主的那一步。
//! - **Fidelity**（原语自报）：直接决定 Policy 要不要探针、要不要提示。
//!
//! | Fidelity | 需要探针吗 | 需要观察窗吗 | 失败要提示吗 | 哪个平台 |
//! |---|---|---|---|---|
//! | `PassThrough` | 不需要 | 不需要 | 不需要 | macOS（响应链让路） |
//! | `Synthetic` | 需要（防双触发） | 需要（判断成没成） | 需要 | Windows、Linux/X11 |
//! | `Unavailable` | —— | —— | 立刻一句 | Wayland / 没有编辑器窗口的平台 |
//!
//! 【为什么 `Unavailable` 不是"什么都不做"】做不到就该**当场**说清楚，而不是让用户
//! 反复按同一个键、每次都得到沉默。
//!
//! 【铁律：只转交按下的那个键，绝不翻译意图】不查宿主的动作表（那不可枚举），
//! 不做"按下播放键就发某个动作 ID"的映射。用户按 Space，宿主收到的就是 Space。

mod strokes;

#[cfg(windows)]
mod windows;

mod unavailable;

pub(crate) use strokes::{KeyToken, Stroke, StrokeSet};

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// 探针窗口：等"宿主自己会不会动"。
///
/// 宿主动作 → 音频块 `kPlaying`（最大 1024 帧 @44.1k ≈ 23 ms）→ 20 ms 的 UI tick，
/// 90 ms 是 3–4 个 tick：足够看清，又不牺牲手感。
const PROBE: Duration = Duration::from_millis(90);

/// 观察窗：让路之后等"宿主真的动了吗"。
///
/// 覆盖 4096 帧 @44.1k（≈93 ms）的宿主缓冲若干倍，仍远小于"用户以为卡死"。
const OBSERVE: Duration = Duration::from_millis(400);

/// 两次让路之间的硬间隔：连打两下 Space 不该变成投递风暴。
const MIN_ESCAPE_GAP: Duration = Duration::from_millis(60);

/// 限流窗口与窗口内的上限。
///
/// 【为什么还要这一道】协议层已经堵了"宿主把键弹回插件"（我们只对 c/x/v/z/y 返回
/// `K_RESULT_OK`），但那是**我们**的约定；宿主自己的钩子仍可能把键弹回来。
const RATE_WINDOW: Duration = Duration::from_secs(10);
const ESCAPES_PER_WINDOW: usize = 20;

/// 让路的保真度：原语**自报**，Policy 只据此分支。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fidelity {
    /// 键根本没进页面（事件沿响应链继续走）。不存在双触发，也就不需要探针。
    ///
    /// 【为什么允许"本仓没人构造它"】今天只有 Windows 原语（`Synthetic`）；这一档由
    /// macOS 的响应链让路构造。保留它不是预埋，而是 Policy 平台无关的前提 —— 少了它，
    /// `Arbiter::new` 就得按平台分支，而那正是本模块存在的理由。
    #[allow(dead_code)]
    PassThrough,
    /// 页面先吃掉了键，我们再补一次。**必须**先证明宿主没自己处理过。
    Synthetic,
    /// 本平台没有这条路。第一次按就说清楚。
    ///
    /// 【同上】由 `unavailable.rs` 的原语构造；Windows 上那个原语不会被实例化。
    #[allow(dead_code)]
    Unavailable,
}

impl Fidelity {
    /// 诊断/日志用的稳定短名。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Fidelity::PassThrough => "pass-through",
            Fidelity::Synthetic => "synthetic",
            Fidelity::Unavailable => "unavailable",
        }
    }
}

/// 宿主播放态的一个快照。
///
/// 【为什么只看 `playing`】播放中位置一直在动，拿位置判"宿主动没动"没有意义
/// （暂停时它也会被 seek 改）。播放/暂停的翻转才是"宿主收到了这个键"的唯一硬证据。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Reading {
    pub(crate) playing: bool,
}

/// 让路失败的结构化原因（**不解析文本**，与 `docs/thread-model.md` §1.3 同一条）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EscapeError {
    /// 找不到宿主窗口（桥接进程 / 窗口已销毁 / 不可见）。
    NoHostWindow,
    /// 宿主不在前台 —— 此时投递，键可能落到别的应用身上。
    NotForeground,
    /// 这个键不在本平台的键表里。
    UnsupportedKey,
    /// 投递本身失败（窗口在投递瞬间被销毁等）。
    PostFailed,
}

impl EscapeError {
    /// 回执/日志用的稳定短名。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            EscapeError::NoHostWindow => "no-host-window",
            EscapeError::NotForeground => "not-foreground",
            EscapeError::UnsupportedKey => "unsupported-key",
            EscapeError::PostFailed => "post-failed",
        }
    }
}

/// 平台原语：把一次按键交回宿主。
///
/// 【为什么不拆成"能不能让"+"让"两段】两段式中间那段时间里宿主可能又变了（切前台、
/// 关窗）。一次调用一个结论：能就做，不能就给出结构化原因。
///
/// 【为什么 `window` 是参数而不是字段】窗口句柄在 `WM_NCCREATE` 之前并不存在，
/// 而持有本原语的 `BrowserState` 早于它构造。把它当参数传，构造顺序问题就消失了，
/// 也顺手让原语保持无状态（可测、可在任意线程重建）。
pub(crate) trait HostKeyEscape {
    fn fidelity(&self) -> Fidelity;

    /// `window` = 本编辑器窗口的平台句柄（HWND / NSView / XID），由调用方给出。
    fn escape(&mut self, window: usize, stroke: &Stroke) -> Result<(), EscapeError>;

    /// 把键盘焦点**交还宿主**。
    ///
    /// 【为什么它是最后手段】焦点真的走了 —— 用户想在插件里继续编辑必须先点回来。
    /// 但它是**确定**的那条路：宿主自己的键位表一定会生效，不像投递那样要看宿主脸色。
    /// 所以它不做默认，只做"投递不灵时的下一步"。
    ///
    /// 【为什么与 `escape` 同一个 trait】两者都是"把键盘还给宿主"，只是彻底程度不同；
    /// 分开成两个 trait 会让"这个平台支持到哪一步"变成两个各自演化的答案。
    /// `previous` = 焦点离开本编辑器之前**最后一个拿到焦点的宿主窗口**（见
    /// `EditorClient::previous_focus`）。原语优先还给它 —— "回到原处"比"跳到某个
    /// 祖先窗口"更符合用户对"交还"的预期，也更容易让宿主的快捷键表命中。
    fn focus_host(&mut self, window: usize, previous: Option<usize>) -> Result<(), EscapeError>;
}

/// 本平台的原语。
#[cfg(windows)]
pub(crate) fn escape_for_platform() -> Box<dyn HostKeyEscape> {
    Box::new(windows::WindowsEscape)
}

/// 本平台的原语（无编辑器窗口的平台）。
#[cfg(not(windows))]
pub(crate) fn escape_for_platform() -> Box<dyn HostKeyEscape> {
    Box::new(unavailable::UnavailableEscape)
}

/// 状态机必须让调用方去做的事。
///
/// 【为什么没有"告诉用户"这一种】让路的成功与失败都是**静默**的：成功时宿主自己动
/// 起来了，那就是反馈；失败时用户的下一步本来就是"点一下宿主窗口"，一句被状态栏
/// 裁掉一半的警告并不能让他做得更好。留下的证据是诊断日志与诊断包读数 —— 它们回答
/// 的是"这个功能到底有没有在工作"，而那是支持场景的问题，不是操作场景的问题。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Effect {
    /// 把这一次按键交回宿主。
    Escape { action: String, stroke: Stroke },
    /// 只记一行诊断日志，不改变任何状态。
    ///
    /// 【为什么把日志做成 effect】状态机因此保持纯函数：它只**声明**发生了什么，
    /// 谁去写文件由调用方决定。测试可以直接断言这条轨迹，不必去截日志。
    Trace(&'static str),
}

/// 不接手的原因。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RejectReason {
    /// 宿主自己处理了这个键（探针学到的会话级事实）。**不该提示**。
    HostNative,
    /// 上一拍（探针或观察窗）还没走完。**不该提示**。
    InFlight,
    /// 连按太快 / 限流。**不该提示**。
    RateLimited,
    /// 这个键不在键表里 —— 让不成立，调用方应回落旧文案。
    UnsupportedKey,
    /// 带修饰键 —— 让不成立，调用方应回落旧文案。
    ModifiedKey,
}

impl RejectReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RejectReason::HostNative => "host-native",
            RejectReason::InFlight => "in-flight",
            RejectReason::RateLimited => "rate-limited",
            RejectReason::UnsupportedKey => "unsupported-key",
            RejectReason::ModifiedKey => "modified-key",
        }
    }
}

/// 一次让路判定的结论。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KeyVerdict {
    /// 已接手（前端什么都不用做）。
    Accepted,
    Rejected(RejectReason),
}

impl KeyVerdict {
    pub(crate) fn reason(self) -> Option<&'static str> {
        match self {
            KeyVerdict::Accepted => None,
            KeyVerdict::Rejected(reason) => Some(reason.as_str()),
        }
    }
}

/// `on_key` 的完整结论：判决 + 立即产生的副作用。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct KeyOutcome {
    pub(crate) verdict: KeyVerdict,
    pub(crate) effects: Vec<Effect>,
}

impl KeyOutcome {
    fn accepted() -> Self {
        Self {
            verdict: KeyVerdict::Accepted,
            effects: Vec::new(),
        }
    }

    fn rejected(reason: RejectReason) -> Self {
        Self {
            verdict: KeyVerdict::Rejected(reason),
            effects: Vec::new(),
        }
    }

    fn accepted_with(effects: Vec<Effect>) -> Self {
        Self {
            verdict: KeyVerdict::Accepted,
            effects,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// 等一等：宿主会不会自己处理这个键。
    Probe,
    /// 已经证明宿主不处理：以后立即让路。
    Relay,
    /// 宿主自己处理这个键（会话闩锁）。以后永不投递，也不提示。
    HostNative,
    /// 本平台没有这条路：每次按都如实说一句。
    Unavailable,
}

/// 在途的窗口（探针或观察窗，二者互斥）。
#[derive(Clone, Debug)]
struct Pending {
    /// `true` = 探针（等宿主自己动），`false` = 观察窗（等让路之后动）。
    probe: bool,
    deadline: Instant,
    /// 判定"动没动"的基线；`None` = 还没拿到过宿主读数。
    baseline: Option<Reading>,
    /// 这次窗口能不能得出"宿主没接"的结论。没有读数时不能 —— 把"我没看见"
    /// 说成"它没做"是另一种撒谎。
    verifiable: bool,
    action: String,
    stroke: Stroke,
}

/// 让路状态机。纯逻辑：不碰窗口、不碰 COM、不碰会话，只认注入的 `Instant` 与读数。
pub(crate) struct Arbiter {
    fidelity: Fidelity,
    mode: Mode,
    pending: Option<Pending>,
    last_escape: Option<Instant>,
    escapes: VecDeque<Instant>,
}

impl Arbiter {
    pub(crate) fn new(fidelity: Fidelity) -> Self {
        Self {
            fidelity,
            mode: match fidelity {
                // 键根本没进页面 ⇒ 不存在双触发 ⇒ 探针纯属浪费一次手感。
                Fidelity::PassThrough => Mode::Relay,
                Fidelity::Synthetic => Mode::Probe,
                Fidelity::Unavailable => Mode::Unavailable,
            },
            pending: None,
            last_escape: None,
            escapes: VecDeque::new(),
        }
    }

    pub(crate) fn fidelity(&self) -> Fidelity {
        self.fidelity
    }

    /// 收到一次"可让"的按键（前端已经判定过）。返回判决 + 立即产生的副作用。
    pub(crate) fn on_key(
        &mut self,
        now: Instant,
        action: &str,
        stroke: Stroke,
        reading: Option<Reading>,
    ) -> KeyOutcome {
        match self.mode {
            Mode::Unavailable => {
                // 【为什么"接手"却什么也不做】前端已经判定"这个键插件用不上"。若我们
                // 回"不接手"，前端只会回落那句"当前宿主没有提供这个接口" —— 那是把
                // 用户往死路上指。这里接手，把结论留在日志里：本平台没有让路原语。
                return KeyOutcome::accepted_with(vec![Effect::Trace("unavailable")]);
            }
            Mode::HostNative => return KeyOutcome::rejected(RejectReason::HostNative),
            Mode::Probe | Mode::Relay => {}
        }
        if !stroke.is_bare() {
            return KeyOutcome::rejected(RejectReason::ModifiedKey);
        }
        // 单飞**只针对探针**：探针靠"基线有没有翻转"下结论，中途插进来的按键会污染基线。
        //
        // 【为什么观察窗不单飞】观察窗只回答"上一次让路成没成"，它是记账、不是闸门。
        // 让它挡住第二下，用户连按两下 Space（播 / 停）就只会生效一次 —— 而他按的
        // 是两次不同的意图。被替换掉的观察窗只是"这一次成没成"没人记账，没有代价。
        if self.pending.as_ref().is_some_and(|pending| pending.probe) {
            return KeyOutcome::rejected(RejectReason::InFlight);
        }
        if self.throttled(now) {
            return KeyOutcome::rejected(RejectReason::RateLimited);
        }
        match self.mode {
            Mode::Probe => match reading {
                Some(baseline) => {
                    self.pending = Some(Pending {
                        probe: true,
                        deadline: now + PROBE,
                        baseline: Some(baseline),
                        verifiable: true,
                        action: action.to_owned(),
                        stroke,
                    });
                    KeyOutcome::accepted()
                }
                // 没有读数就无从验证：跳过探针直接让一次，观察窗记为"不可验证"。
                // 把它当成"宿主没动"会让探针每次都误判，于是每次都多等 90 ms。
                None => {
                    self.mode = Mode::Relay;
                    self.record_escape(now);
                    self.pending = Some(Pending {
                        probe: false,
                        deadline: now + OBSERVE,
                        baseline: None,
                        verifiable: false,
                        action: action.to_owned(),
                        stroke: stroke.clone(),
                    });
                    KeyOutcome::accepted_with(vec![
                        Effect::Trace("relayed"),
                        Effect::Escape {
                            action: action.to_owned(),
                            stroke,
                        },
                    ])
                }
            },
            Mode::Relay => {
                self.record_escape(now);
                self.pending = Some(Pending {
                    probe: false,
                    deadline: now + OBSERVE,
                    baseline: reading,
                    verifiable: reading.is_some(),
                    action: action.to_owned(),
                    stroke: stroke.clone(),
                });
                KeyOutcome::accepted_with(vec![
                    Effect::Trace("relayed"),
                    Effect::Escape {
                        action: action.to_owned(),
                        stroke,
                    },
                ])
            }
            Mode::HostNative => KeyOutcome::rejected(RejectReason::HostNative),
            Mode::Unavailable => KeyOutcome::rejected(RejectReason::HostNative),
        }
    }

    /// 20 ms tick：推进探针/观察窗。
    pub(crate) fn tick(&mut self, now: Instant, reading: Option<Reading>) -> Vec<Effect> {
        let Some(mut pending) = self.pending.take() else {
            return Vec::new();
        };

        // 读数迟到：观察窗就地升级为可验证，基线取第一次拿到的读数。
        // 【为什么不立刻判成功】刚建立的基线还没被"让路后的宿主"影响过。
        if pending.baseline.is_none() {
            match reading {
                Some(reading) => {
                    pending.baseline = Some(reading);
                    pending.verifiable = true;
                    self.pending = Some(pending);
                }
                None => {
                    if now < pending.deadline {
                        self.pending = Some(pending);
                    }
                    // 到期仍无读数：无可奉告（verifiable=false ⇒ 不发提示）。
                }
            }
            return Vec::new();
        }

        if reading.is_some_and(|reading| Some(reading) != pending.baseline) {
            if pending.probe {
                // 宿主自己动了 ⇒ 这个宿主本来就会处理这个键。会话闩锁，永不投递。
                self.mode = Mode::HostNative;
                return vec![Effect::Trace("host-native")];
            }
            return vec![Effect::Trace("accepted")];
        }

        if now < pending.deadline {
            self.pending = Some(pending);
            return Vec::new();
        }

        if pending.probe {
            // 探针结论：宿主没动 ⇒ 让它一次，转入观察窗。
            self.mode = Mode::Relay;
            self.record_escape(now);
            let stroke = pending.stroke.clone();
            let action = pending.action.clone();
            self.pending = Some(Pending {
                probe: false,
                deadline: now + OBSERVE,
                baseline: pending.baseline,
                verifiable: true,
                action: pending.action,
                stroke: pending.stroke,
            });
            return vec![Effect::Trace("relayed"), Effect::Escape { action, stroke }];
        }

        // 观察窗到期、宿主仍没动。静默：用户的下一个动作本来就是去点宿主窗口。
        if pending.verifiable {
            return vec![Effect::Trace("no-response")];
        }
        Vec::new()
    }

    fn throttled(&self, now: Instant) -> bool {
        if self
            .last_escape
            .is_some_and(|at| now.duration_since(at) < MIN_ESCAPE_GAP)
        {
            return true;
        }
        self.escapes
            .iter()
            .filter(|at| now.duration_since(**at) < RATE_WINDOW)
            .count()
            >= ESCAPES_PER_WINDOW
    }

    fn record_escape(&mut self, now: Instant) {
        self.last_escape = Some(now);
        self.escapes.push_back(now);
        while self
            .escapes
            .front()
            .is_some_and(|at| now.duration_since(*at) >= RATE_WINDOW)
        {
            self.escapes.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare(token: KeyToken) -> Stroke {
        Stroke {
            token,
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }
    }

    fn playing(playing: bool) -> Option<Reading> {
        Some(Reading { playing })
    }

    /// 探针窗口 + 观察窗的推进器：把"20 ms 一次 tick"手写成显式时间，测试因此完全确定。
    fn tick_for(
        arbiter: &mut Arbiter,
        from: Instant,
        millis: u64,
        reading: Option<Reading>,
    ) -> Vec<Effect> {
        let mut effects = Vec::new();
        let mut step = 20;
        while step <= millis {
            effects.extend(arbiter.tick(from + Duration::from_millis(step), reading));
            step += 20;
        }
        effects
    }

    fn escaped(effects: &[Effect]) -> usize {
        effects
            .iter()
            .filter(|effect| matches!(effect, Effect::Escape { .. }))
            .count()
    }

    fn traces(effects: &[Effect]) -> Vec<&'static str> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Trace(trace) => Some(*trace),
                _ => None,
            })
            .collect()
    }

    /// **不变量：宿主自己动了 ⇒ 会话闩锁，永不投递。**
    ///
    /// 【为什么这是最重要的一条】有的宿主本来就会把按键交给插件视图处理。若我们也投一次，
    /// 用户按一下 Space 会被处理两次（播放→又暂停），表现是"按了完全没反应" —— 比现状更坏。
    #[test]
    fn a_host_that_handles_the_key_itself_is_never_posted_to() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        let outcome = arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(outcome.verdict, KeyVerdict::Accepted);
        assert_eq!(escaped(&outcome.effects), 0, "探针期间不得投递");

        // 宿主自己开始播放：探针看到翻转。
        let effects = tick_for(&mut arbiter, start, 100, playing(true));
        assert_eq!(escaped(&effects), 0, "宿主已处理 ⇒ 不得投递");
        assert_eq!(traces(&effects), vec!["host-native"], "正常情况只记一行日志");

        // 之后所有按键都不再走这条路，也不提示。
        let later = start + Duration::from_secs(1);
        let outcome = arbiter.on_key(
            later,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(true),
        );
        assert_eq!(
            outcome.verdict,
            KeyVerdict::Rejected(RejectReason::HostNative)
        );
        assert!(outcome.effects.is_empty());
    }

    /// **不变量：宿主没动 ⇒ 投递一次，并转入"以后立即投"。**
    #[test]
    fn a_host_that_ignores_the_key_gets_exactly_one_post() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );

        // 90 ms 探针内不得投递。
        let early = tick_for(&mut arbiter, start, 80, playing(false));
        assert_eq!(escaped(&early), 0, "探针未到期就投递 = 双触发的来源");

        let effects = tick_for(&mut arbiter, start, 100, playing(false));
        assert_eq!(escaped(&effects), 1, "探针到期后必须且只能投一次");

        // 第二下：模式已是 Relay ⇒ 立即投，不再等 90 ms。
        let second = start + Duration::from_millis(200);
        let outcome = arbiter.on_key(
            second,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(outcome.verdict, KeyVerdict::Accepted);
        assert_eq!(escaped(&outcome.effects), 1, "Relay 模式下必须立即投递");
    }

    /// **不变量：成功与超时都只留一行日志，绝不对用户说话。**
    ///
    /// 【为什么超时也不说话】"让路没成"的下一步永远是同一个：点一下宿主窗口。而状态栏
    /// 那个位置放不下一句完整的解释（状态片是 `shrink-0`，父容器 `truncate` 会把它
    /// 从中间裁掉）。与其给一句被裁掉一半的警告，不如把事实留在日志与诊断包里。
    #[test]
    fn a_silent_success_and_a_silent_timeout() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        tick_for(&mut arbiter, start, 100, playing(false));

        // 宿主动起来了：观察窗判成功。
        let success = tick_for(
            &mut arbiter,
            start + Duration::from_millis(100),
            60,
            playing(true),
        );
        assert_eq!(traces(&success), vec!["accepted"]);

        // 再来一次，这次宿主一直不动：观察窗到期后提示一次。
        let second = start + Duration::from_secs(1);
        arbiter.on_key(
            second,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(true),
        );
        let timeout = tick_for(&mut arbiter, second, 460, playing(true));
        assert_eq!(traces(&timeout), vec!["no-response"]);
        assert_eq!(
            timeout.iter().filter(|e| matches!(e, Effect::Escape { .. })).count(),
            0,
            "观察窗只记账，不再投递"
        );

        // 同一个动作再失败一次：照常记账，仍然不说话。
        let third = start + Duration::from_secs(3);
        arbiter.on_key(
            third,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(true),
        );
        let again = tick_for(&mut arbiter, third, 460, playing(true));
        assert_eq!(traces(&again), vec!["no-response"]);
    }

    /// **不变量：探针期间不接第二下；观察窗期间照接（连按 = 两次意图）。**
    ///
    /// 【为什么分开】探针靠"基线有没有翻转"下结论，插进来的按键会污染基线；而观察窗
    /// 只是记账。让它挡住第二下，用户连按两下 Space（播 / 停）就只会生效一次 ——
    /// 可他按的是两次不同的意图。
    #[test]
    fn a_second_key_during_the_probe_is_dropped_but_not_during_observation() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );

        let during_probe = arbiter.on_key(
            start + Duration::from_millis(40),
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(
            during_probe.verdict,
            KeyVerdict::Rejected(RejectReason::InFlight)
        );
        assert!(during_probe.effects.is_empty());

        // 探针到期 ⇒ 投递一次，转入观察窗。
        let posted = tick_for(&mut arbiter, start, 100, playing(false));
        assert_eq!(escaped(&posted), 1);

        // 观察窗期间的第二次按键：照接、立即投递（连按两下 = 播 / 停）。
        let during_observe = arbiter.on_key(
            start + Duration::from_millis(200),
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(during_observe.verdict, KeyVerdict::Accepted);
        assert_eq!(escaped(&during_observe.effects), 1);
    }

    /// **不变量：连按与高频都被限流，且限流期间不提示。**
    #[test]
    fn bursts_are_rate_limited_without_nagging() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        tick_for(&mut arbiter, start, 100, playing(false));
        tick_for(&mut arbiter, start, 500, playing(false)); // 让观察窗走完

        // 距上次投递不到 60 ms：限流。
        let outcome = arbiter.on_key(
            start + Duration::from_millis(120),
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(
            outcome.verdict,
            KeyVerdict::Rejected(RejectReason::RateLimited)
        );
        assert!(outcome.effects.is_empty(), "限流不该顺带提示用户");
    }

    /// **不变量：没有读数时跳过探针，直接让一次，并且绝不因此提示。**
    ///
    /// 【为什么】"还没收到宿主读数"不等于"宿主没动"。把它当成"没动"会让探针每次都误判、
    /// 每次都白等 90 ms；而据此发提示更是把"我没看见"说成"它没做"。
    #[test]
    fn a_missing_reading_skips_the_probe_and_stays_silent() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        let outcome = arbiter.on_key(start, "playback.toggle", bare(KeyToken::Space), None);
        assert_eq!(outcome.verdict, KeyVerdict::Accepted);
        assert_eq!(escaped(&outcome.effects), 1, "没有读数时直接让一次");

        let effects = tick_for(&mut arbiter, start, 460, None);
        assert_eq!(traces(&effects), Vec::<&str>::new(), "不可验证就无可奉告");
    }

    /// **不变量：读数迟到时观察窗就地升级为可验证，基线取第一次拿到的读数。**
    #[test]
    fn a_late_reading_upgrades_the_observation_window() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        arbiter.on_key(start, "playback.toggle", bare(KeyToken::Space), None);
        // 100 ms 时读数出现（此时基线 = false）；之后宿主动了 ⇒ 成功、不提示。
        let effects = tick_for(&mut arbiter, start, 100, playing(false));
        assert_eq!(traces(&effects), Vec::<&str>::new());
        let effects = tick_for(
            &mut arbiter,
            start + Duration::from_millis(100),
            200,
            playing(true),
        );
        assert_eq!(traces(&effects), vec!["accepted"]);
    }

    /// **不变量：PassThrough 平台不付探针的成本。**
    ///
    /// 【为什么】键根本没进页面，宿主当场就动了 —— 再等 90 ms 只是白白拖慢手感。
    #[test]
    fn pass_through_platforms_relay_immediately() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::PassThrough);
        let outcome = arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(escaped(&outcome.effects), 1, "PassThrough 必须立即让路");
    }

    /// **不变量：Unavailable 平台接手但什么都不做，只留一行日志。**
    ///
    /// 【为什么是"接手"】前端已经判定"这个键插件用不上"。回"不接手"只会让它回落那句
    /// "当前宿主没有提供这个接口" —— 那是把用户往死路上指。
    #[test]
    fn unavailable_platforms_accept_without_posting_or_speaking() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Unavailable);
        let outcome = arbiter.on_key(
            start,
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(outcome.verdict, KeyVerdict::Accepted);
        assert_eq!(escaped(&outcome.effects), 0, "本平台没有原语，不得假装投递");
        assert_eq!(traces(&outcome.effects), vec!["unavailable"]);

        let again = arbiter.on_key(
            start + Duration::from_secs(5),
            "playback.toggle",
            bare(KeyToken::Space),
            playing(false),
        );
        assert_eq!(again.verdict, KeyVerdict::Accepted);
    }

    /// **不变量：带修饰的键一律不让。**
    #[test]
    fn modified_strokes_are_never_relayed() {
        let start = Instant::now();
        let mut arbiter = Arbiter::new(Fidelity::Synthetic);
        let stroke = Stroke {
            token: KeyToken::Space,
            ctrl: true,
            shift: false,
            alt: false,
            meta: false,
        };
        let outcome = arbiter.on_key(start, "playback.toggle", stroke, playing(false));
        assert_eq!(
            outcome.verdict,
            KeyVerdict::Rejected(RejectReason::ModifiedKey)
        );
        assert!(outcome.effects.is_empty());
    }
}
