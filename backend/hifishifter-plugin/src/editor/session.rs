//! 真实ARA文档共享原编辑命令actor：UI只排队，分析/文件/离线渲染均不在音频或UI回调执行。
use crate::render::document::DocumentSession;
use crate::sync::MutexExt;
use hifishifter_kernel::editor::{
    history,
    host_pcm::{materialize_with_byte_limit, PcmView},
    midi_import::MidiImportHost,
    ParamHost,
};
use hifishifter_kernel::state::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// 自动渲染连续失败多少次后**停止重试**，把错误稳定地交给状态栏。
///
/// 【为什么必须有上限】此前失败路径只设一个 150 ms / 1 s 的 `deadline`，下一轮又
/// `start_render` —— 而 `start_render` 会先发 `active:true`。于是一个**永不收敛**的
/// 失败会以约 1 Hz 无限回潮，用户看到左下角永远在"渲染中"。有上限后，连续失败到预算
/// 就转成稳定的错误读数（前端 `PluginApplyStatus` 的"尚未应用：{error}"）。
///
/// 【为什么只统计**真错误**】`"pitch analysis pending"` 是正常在途状态，不是失败：
/// 一段几分钟素材的分析本来就可能跑很久，按次数计数会把"慢"误判成"不收敛"而提前
/// 放弃渲染。它由 [`RetryState`] 的 `analysis_pending_since` 单独按**时间**约束。
///
/// 【为什么耗尽预算只是**退避**而不是永久停止】此前预算耗尽即 `settled = true`，
/// 自动渲染**再也不会**启动，直到用户下一次操作。可 8 次重试的间隔是 150 ms ——
/// 它实际衡量的是"这 1.2 秒里宿主一直在变"，而不是"这件事永远不会成功"。实测 Cubase 15
/// 上一次 `unsupported host position` 就是 8 次全失败后把界面**永久**冻结在旧版本
/// （日志 22:22:35 `gave up after 8 attempts`），而那个错误本身是瞬时的。
/// 改成退到 [`RENDER_RETRY_BACKOFF`] 后，宿主一旦静止就会自然收敛；同时错误读数仍然
/// 显示给用户（`render_error` 照写），不再有"看起来在工作、其实再也不动了"的状态。
const RENDER_RETRY_BUDGET: u32 = 8;

/// 重试预算耗尽后的退避间隔（0.2 Hz）。
///
/// 仍然重试（不是一个永久终态），但足以让宿主把编辑事务收口。终态错误
/// （panic 毒化 / 文档关闭 / 租约撤销）在 actor 循环里另有判据，不受本常量影响。
const RENDER_RETRY_BACKOFF: Duration = Duration::from_secs(5);

/// 预算内的快速重试间隔。
const RENDER_RETRY_INTERVAL: Duration = Duration::from_millis(150);

/// 分析 pending 的**绝对**上限：超过它就**不再等这条原线**，本次渲染直接放行。
///
/// 【为什么只留一个上限，不再有"2 秒没有 worker 就算不收敛"】那条判据问的是本会话
/// 自己的 worker 表，而"表里没有"在两种完全正常的情形下都成立（任务刚排出去、或同一份
/// 内容的分析由别的会话在跑），于是它把正常等待判成死局并永久停掉自动渲染。现在
/// "要不要等"由内核门禁回答（见 `pitch_analysis_can_run`），等待永远有理由 ——
/// 只剩一个上限兜住**真的卡死**（后端初始化失败、worker 永久阻塞）。
///
/// 【为什么到顶是"放弃等它"而不是"放弃渲染"】把渲染一起停掉是最坏的选择：用户既听不到
/// 处理结果，也看不到任何进展。到顶后那根被记进 `analysis_abandoned`，渲染立刻放行，
/// 内核走它在缺少原线时的既定回退 —— 降级输出永远好过一条无法被满足的等待。
const ANALYSIS_PENDING_CEILING: Duration = Duration::from_secs(60);

/// "pitch analysis pending" 的重试间隔。
///
/// 【为什么不是 150 ms】`start_render` 会先跑 `prepare_apply`，而它内部要
/// `complete_cached_analysis`（按根重组装原线，不便宜）。150 ms 的热重试在慢分析期间
/// 会以每秒 6–7 次的速度反复做这件事，把单线程 actor 饿死 —— 表现是**无关的 UI 命令
/// 5 秒超时**。分析本来就慢，用 1 s 轮询足够；而分析一有进展，
/// `refresh_analysis()` 会立刻重新武装成 150 ms，不会拖慢正常收敛。
const ANALYSIS_PENDING_RETRY: Duration = Duration::from_secs(1);

/// 自动渲染重试的本地状态（只属于 actor 循环，不跨线程）。
struct RetryState {
    /// 上一轮见过的 `submitted`：用户的新操作会推进它，据此清零预算、重新武装。
    submitted: u64,
    /// 连续**真错误**的次数。
    streak: u32,
    /// "pitch analysis pending" 从什么时候开始（有进展即清零）。
    analysis_pending_since: Option<Instant>,
}

/// 一次渲染尝试失败后的处置。
///
/// 【为什么只有"稍后重试"】这个枚举此前还有一个 `Settled`（"不再重排、不再发
/// `active:true`"），而它唯一的来源是"分析不收敛"的判定 —— 那条判定本身是错的
/// （它问的是**本会话自己的** worker 表，见 [`RetryState::classify`]）。真正的失败
/// 现在一律是退避重试：错误读数照写（前端显示"尚未应用：{error}"），宿主的瞬时冲突
/// 一旦收口就自然收敛；而**终态**错误（panic 毒化 / 文档关闭 / 租约撤销）在 actor
/// 循环里另有判据，不受这里影响。于是"自动渲染永久停摆"这个状态不存在了。
enum RenderAction {
    /// 稍后重试。
    Retry(Duration),
}

impl RetryState {
    fn new(submitted: u64) -> Self {
        Self {
            submitted,
            streak: 0,
            analysis_pending_since: None,
        }
    }

    /// 用户做了新操作 / 分析有新进展 ⇒ 清零预算并重新武装。
    fn rearm(&mut self) {
        self.streak = 0;
        self.analysis_pending_since = None;
    }

    /// 分类一次失败。
    fn classify(&mut self, session: &EditorSession, error: String) -> RenderAction {
        if error == "pitch analysis pending" {
            // 正常在途：不计入次数预算，只按时间约束。
            //
            // 【为什么这里不再判"没有 worker 在跑"】这条判据此前是"分析不收敛"的
            // 主要来源，而它问的是**本会话自己的** worker 表 —— 于是两种完全正常的
            // 情形都被它判死：分析任务刚被排出去（表里还没有），以及同一个源文件的
            // 分析由**别的会话**在跑（`GLOBAL_CLIP_PITCH_INFLIGHT` 去重，结果会进
            // 全局缓存，本会话的组装照样能取到）。判死的后果是 `settled`：自动渲染
            // 从此不再启动，状态栏永久停在"正在自动应用"，而用户画的那条线一次都没
            // 生效。现在"要不要等"由内核的门禁统一回答（它只在**真的有任务会产出**
            // 时为真，见 `pitch_analysis_can_run`），所以等待本身永远是有理由的 ——
            // 这里只剩一个绝对上限，用来兜住"任务卡死"。
            let since = *self.analysis_pending_since.get_or_insert_with(Instant::now);
            if since.elapsed() > ANALYSIS_PENDING_CEILING {
                // 【为什么是"放弃等它"而不是"放弃渲染"】分析卡死（例如声码器/分析
                // 后端初始化失败）时，把渲染一起停掉是最坏的选择：用户既听不到处理
                // 结果，也看不到任何进展，只能对着一个永远转不完的状态。这里改为
                // **放行**这次渲染（内核在缺少原线时有既定回退语义），并把这个根记进
                // `analysis_abandoned`，于是后续渲染不再等它。
                for root in session.pending_analysis_roots() {
                    session.analysis_abandoned.lock_or_recover().insert(root);
                }
                log::warn!(
                    "[ara] pitch analysis did not produce a line after {:?}; rendering without it \
                     instead of freezing (workers in flight: {})",
                    since.elapsed(),
                    session.analysis_in_flight()
                );
                self.analysis_pending_since = None;
                return RenderAction::Retry(RENDER_RETRY_INTERVAL);
            }
            return RenderAction::Retry(ANALYSIS_PENDING_RETRY);
        }
        self.analysis_pending_since = None;
        self.streak = self.streak.saturating_add(1);
        // 预算内：快速重试（宿主的编辑事务通常会在这 1.2 秒内收口）。
        if self.streak < RENDER_RETRY_BUDGET {
            if error != "automatic apply superseded" {
                *session.render_error.lock_or_recover() = Some(error);
            }
        } else {
            // 预算耗尽：**退避**，不是放弃（见 `RENDER_RETRY_BACKOFF`）。错误读数仍然写出去，
            // 前端照常显示"尚未应用：{error}"，但自动渲染会在退避后继续尝试。
            if self.streak == RENDER_RETRY_BUDGET {
                log::warn!(
                    "[ara] automatic render is backing off after {} attempts (every {:?}): {error}",
                    self.streak,
                    RENDER_RETRY_BACKOFF
                );
            }
            *session.render_error.lock_or_recover() = Some(error);
        }
        render_retry_action(self.streak)
    }
}

/// 一次**真错误**之后的处置：预算内快速重试，预算外退避。
///
/// 【为什么抽成纯函数】`classify` 需要一整份 `EditorSession`，而 `RetryState` 的其余分支
/// 又依赖分析 worker 的在途状态 —— 直接在测试里断言"预算耗尽的间隔"很贵。这条判据本身
/// 是纯的，抽出来就能精确钉住"耗尽预算只是退避、不是永久停止"（见 `RENDER_RETRY_BUDGET`）。
fn render_retry_action(streak: u32) -> RenderAction {
    if streak < RENDER_RETRY_BUDGET {
        RenderAction::Retry(RENDER_RETRY_INTERVAL)
    } else {
        RenderAction::Retry(RENDER_RETRY_BACKOFF)
    }
}

#[derive(Clone)]
pub(crate) struct UiSink {
    pub view_id: String,
    pub reply: mpsc::Sender<Value>,
    pub events: mpsc::SyncSender<Value>,
    pub closed: Arc<AtomicBool>,
}
impl UiSink {
    fn response(&self, value: Value) {
        if !self.closed.load(Ordering::Acquire) {
            let _ = self.reply.send(value);
        }
    }
    fn event(&self, value: Value) {
        if !self.closed.load(Ordering::Acquire) {
            let _ = self.events.try_send(value);
        }
    }
}
pub(crate) struct UiRequest {
    pub id: u64,
    pub command: String,
    pub args: Value,
    pub sink: UiSink,
    pub link: Option<Arc<super::routing::EditorLink>>,
}
enum Job {
    Request(UiRequest, Option<u64>),
    Barrier(mpsc::Sender<()>),
    /// 只唤醒 actor 回到循环顶部，不携带命令、不推进任何票据。
    ///
    /// 【为什么需要它】`request_reload` 要"立刻"被看到，但它不是用户编辑。
    /// 此前它借用 `submitted` 票据来唤醒 actor，结果把账本推进到 `processed`
    /// 永远追不上的位置（见 `request_reload` 的说明）。专用唤醒把"叫醒"与
    /// "记账"彻底分开。
    Wake,
    Close,
}
/// 单个可取消DSP任务；编辑actor只收结果，计算线程不改GUI历史或参数权威。
struct RenderTask {
    ticket: u64,
    generation: u64,
    cancel: Arc<AtomicBool>,
    reply: mpsc::Receiver<Result<(), String>>,
    worker: std::thread::JoinHandle<()>,
}
impl RenderTask {
    fn spawn(
        ticket: u64,
        generation: u64,
        cancel: Arc<AtomicBool>,
        work: impl FnOnce() -> Result<(), String> + Send + 'static,
    ) -> Result<Self, String> {
        let (sender, reply) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("hfs-editor-dsp".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .unwrap_or_else(|_| Err("automatic render panicked".into()));
                let _ = sender.send(result);
            })
            .map_err(|error| format!("create editor DSP worker: {error}"))?;
        Ok(Self {
            ticket,
            generation,
            cancel,
            reply,
            worker,
        })
    }
}
struct RegisteredView {
    sink: UiSink,
    route: Option<(Arc<super::routing::EditorLink>, u64)>,
}
#[derive(Default)]
struct Loaded {
    initialized: bool,
    edit: u64,
    model: u64,
    scope: u64,
    audio: u64,
    projection: String,
    reverse_paths: HashMap<String, String>,
}
/// 本会话的**时间线/参数从哪里来**。
///
/// 【为什么需要一个显式来源，而不是"有没有文档"这个隐含判据】`EditorSession` 此前
/// 只服务 ARA：文档既是时间线的来源，也是几何/撤销的权威。Capture 形态（宿主没给
/// ARA，音频从 VST3 输入总线录进来）**永远没有文档**，但它的编辑界面必须能用 ——
/// 否则"录进来由插件处理"这件事根本无从发生（用户改不了音高）。
///
/// 两者的差别被收敛成两件事，其余（历史、参数、分析、波形、命令分发）完全共用：
///   * 时间线从哪来：ARA 是宿主投影；Capture 是捕获存储。
///   * 谁是权威：ARA 的几何/撤销属于宿主；Capture 的一切都属于本会话。
#[derive(Clone)]
pub(crate) enum SessionBacking {
    /// ARA 文档：时间线是宿主投影，几何/撤销由宿主拥有。
    Ara(Weak<DocumentSession>),
    /// Capture：没有 ARA 文档；时间线由捕获存储构建，之后完全由本会话拥有。
    Capture(Weak<crate::render::extension::ExtensionOwner>),
}

pub(crate) struct EditorSession {
    /// 时间线/参数来源（[`SessionBacking`]）。
    ///
    /// 【为什么不拆成两个字段】两者互斥，而且几乎每个使用点都要先回答"我有没有文档"。
    /// 一个枚举让"Capture 没有文档"成为**编译期可见**的事实，而不是一个可能被忽略的
    /// `Option`。取值一律通过 [`Self::document`] / [`Self::require_document`]。
    backing: SessionBacking,
    pub(crate) timeline: Mutex<TimelineState>,
    /// 组件 state 恢复出来的 Capture 用户编辑，等待下一次时间线加载时叠上去。
    ///
    /// 【为什么放在会话上而不是 owner 上】它是"要应用进本会话时间线的数据"，
    /// 与 `load_capture_timeline` 的 `kept` 机制是同一件事。owner 只负责在会话
    /// **还不存在**时把字节块暂存下来（见 `ExtensionOwner::pending_capture_edits`）。
    pub(super) pending_capture_edits: Mutex<Option<super::capture_edits::CaptureEdits>>,
    pub(super) history: Mutex<TimelineHistory>,
    pub(super) project: Mutex<ProjectState>,
    // 设置**不在**这里：它属于用户，不属于某一个 ARA 文档。此前它是本结构体的字段，
    // 于是每个工程一份、初始化为出厂默认，换工程即丢。现在由进程级的
    // `crate::settings_store` 持有（与独立 App 共用同一份配置文件）。
    loaded: Mutex<Loaded>,
    pub(crate) namespace: String,
    pub(super) browser_roots: Mutex<Vec<PathBuf>>,
    // 剪贴板 MIDI 载荷（"Standard MIDI File"）：前端先探测一次拿到 GUID，导入命令再
    // 按 GUID 取字节。与 App 侧同形状、同上限，语义见内核 `editor::midi_import`。
    pub(super) clipboard_midi: Mutex<std::collections::VecDeque<(String, Vec<u8>)>>,
    // 只保留已获ARA授权后生成的GUI媒体元信息/路径，不保留或复用实时播放PCM。
    display_waveforms: Mutex<HashMap<String, (String, hifishifter_kernel::state::Clip)>>,
    pcm_dir: PathBuf,
    pub(super) peaks: Mutex<HashMap<String, Arc<hifishifter_kernel::hfspeaks_v2::HfsPeakFile>>>,
    queue: mpsc::SyncSender<Job>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    analysis_sender: mpsc::Sender<hifishifter_kernel::engine_command::EngineCommand>,
    analysis_updates: Mutex<mpsc::Receiver<hifishifter_kernel::engine_command::EngineCommand>>,
    analysis_cancel: Arc<AtomicBool>,
    analysis_workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    views: Mutex<HashMap<String, RegisteredView>>,
    pub(super) generation: AtomicU64,
    /// 时间线代次：`mark_dirty`（编辑）与 `publish_timeline`（发布）**任一**都推进它。
    ///
    /// 【为什么必须是两者的并集】Capture 的渲染版本闸以它为据，而它要回答的是
    /// "渲染读的东西变了没有"。两个来源各有一半：
    ///
    /// - `mark_dirty` 覆盖"编辑"这一层语义，但它**不是**曲线改动的必要条件 ——
    ///   `convert_mix_param` 把音量曲线换算成 dyn 曲线并发布，却不标脏。只认它会让
    ///   那种改动不推进版本 ⇒ 渲染被跳过 ⇒ 曲线永远不生效（用户报障：
    ///   "画完参数线不生效"）。
    /// - `publish_timeline` 覆盖"时间线被发布"，也就是渲染实际读的那份数据。
    ///
    /// 取并集是**故意的**：多推进一次最多多排一次渲染（幂等、且 actor 一次只跑一个），
    /// 而漏推进就是一次静默的"改了不生效"。错法不对称，所以选安全的那一侧。
    timeline_revision: AtomicU64,
    /// 【B4 归因】渲染版本是**谁**推进的，分成三档。
    ///
    /// 【为什么需要它】`edit` 只告诉我们"版本动了"，不告诉我们"谁动的"。现场
    /// 排查时这两种情形在日志里完全同形：用户在编辑（该重渲染），与某条内部路径
    /// 每轮都在推进版本（不该重渲染）。上一轮就是因为无法区分，把 24 次版本推进
    /// 归因到了音高分析上 —— 而分析其实只跑了 3 次。
    revision_bumps_dirty: AtomicU64,
    revision_bumps_publish: AtomicU64,
    revision_bumps_analysis: AtomicU64,
    /// 上一次打过的三档计数；只在变化时打印（20 ms 定时器驱动，不去重就是每秒 50 行）。
    last_revision_report: Mutex<Option<(u64, u64, u64)>>,
    /// 已发布时间线的片段数（健康探活用，避免取 `timeline` 锁）。
    ///
    /// 【为什么不在 `plugin_health` 里直接读 `timeline.lock()`】探活发生在宿主 UI
    /// 线程上，而 actor 可能正在做长任务并持有时间线锁；一个只回答"几段"的状态查询
    /// 不该排在那后面。写入点只有 `publish_timeline` / `load_capture_timeline` /
    /// `ensure_loaded` 三处。
    clip_count: AtomicU64,
    /// 已经**放弃等待**原线分析的根轨道（见 [`RetryState::classify`] 的上限分支）。
    ///
    /// 【为什么需要这份状态】等待本身是对的（原线是渲染的输入之一），但等待必须**有界**：
    /// 一个卡死的分析任务会让渲染永远不跑，用户既听不到处理结果也看不到进展。超时之后
    /// 把它记在这里，于是后续渲染不再等它 —— 内核在缺少原线时有既定回退语义，降级输出
    /// 永远好过一条无法被满足的等待。
    ///
    /// 【什么时候清】素材变化时（重新加载时间线把 `pitch_orig_key` 全部清空的那一刻）：
    /// 那是"该为新素材重新分析"的唯一语义点，也是一次新的、值得等待的机会。
    analysis_abandoned: Mutex<std::collections::BTreeSet<String>>,
    submitted: AtomicU64,
    processed: AtomicU64,
    pub(super) applied: AtomicU64,
    host_version: AtomicU64,
    ui_geometry_version: AtomicU64,
    project_duration: AtomicU64,
    pub(super) error: Mutex<Option<String>>,
    render_error: Mutex<Option<String>>,
    render_requested: AtomicBool,
    render_task: Mutex<Option<RenderTask>>,
    closed: AtomicBool,
    transport_probe: bool,
    transport_probe_next: Mutex<Instant>,
    pub(super) suppress_history: AtomicBool,
    /// **上一次采纳过的**宿主编辑器视图选择（A5），用于与宿主的新选择比对。
    ///
    /// 【为什么要"上次采纳的"而不是"当前选择"】`notifySelection` 是**通知**不是
    /// **设定**：SDK 明说插件随时可以用自己的 UI 改内部选择
    /// （`ARAInterface.h:3646-3655`）。记上次采纳的那一条，宿主重复发同一个选择时
    /// 就不会把用户在插件里刚点的那条抢回去；只有宿主的**新**选择才覆盖。
    host_selection: Mutex<Option<String>>,
    /// "下一轮 tick 重新加载时间线"的一次性请求。
    ///
    /// 【为什么需要它】ARA 形态下宿主变化由文档版本推动（`refresh_host`）。Capture
    /// 形态**没有任何宿主回调**：录制结束、切换 take、清空 take 都发生在视图层
    /// （`capture::commands`），而加载时间线要做文件 I/O —— 不能在那条 UI 线程上做。
    /// 于是那里只置一个标志，由 actor 线程在下一轮自己加载。
    reload_requested: AtomicBool,
    /// 首次加载失败后的下一次重试时刻（见 `refresh_host` 的慢速重试）。
    ///
    /// 【为什么需要它】`refresh_host` 此前在 `!initialized` 时**直接返回**。而
    /// `initialized` 只在 `ensure_loaded` **成功**的末尾置真 —— 于是"第一次加载失败"
    /// 就变成"此后这条路径永远什么都不做"。窗口里只剩 UI 的 `get_timeline_state`
    /// 还在重试；窗口一关（REAPER 会销毁 view），就再也没有驱动者，工程永久停在
    /// 一份读不出来的状态上。
    ///
    /// 【为什么是 1 s】`ensure_loaded` 会做实打实的 materialize（写 WAV）；没有模型的
    /// 会话里 1 Hz 是安全上界，也远快于用户感知。
    initial_retry_at: Mutex<Option<Instant>>,
}

impl Drop for EditorSession {
    /// 会话结束时清掉自己的 PCM 暂存目录。
    ///
    /// 【为什么必须有】这个目录是按命名空间（含 pid 与序号）建的，宿主每次启动都会
    /// 得到新目录；没有回收点就会在磁盘上无限累积渲染用的 PCM。清理失败只留日志，
    /// 不影响关闭流程。
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.pcm_dir) {
            // 目录可能从未被创建过（没有渲染发生）—— 那不算错误。
            if error.kind() != std::io::ErrorKind::NotFound {
                crate::log_line(&format!("plugin PCM scratch cleanup failed: {error}"));
            }
        }
    }
}

impl EditorSession {
    /// 导入目标只接受当前原GUI已授权轨道；未指定时由native直接parent决定空轨首次导入。
    pub(crate) fn host_track_id(&self, id: &str) -> Result<String, String> {
        if !self
            .timeline
            .lock_or_recover()
            .tracks
            .iter()
            .any(|track| track.id == id)
        {
            return Err("unknown editor import track".into());
        }
        id.strip_prefix(&self.namespace)
            .map(str::to_owned)
            .ok_or_else(|| "import track belongs to another editor".into())
    }
    pub(super) fn ui_clip_id(&self, native: &str) -> String {
        format!("{}{native}", self.namespace)
    }
    /// 原App轨道拖动只形成插件参数分组；一次事务保存GUID关系，不调用任何宿主轨道setter。
    pub(super) fn move_private_track(
        &self,
        id: &str,
        index: usize,
        parent: Option<String>,
    ) -> Result<(), String> {
        let before = self.timeline.lock_or_recover().clone();
        if !before.tracks.iter().any(|track| track.id == id) || index > 10000 {
            return Err("unknown private track or invalid index".into());
        }
        let mut candidate = before.clone();
        candidate.move_track(id, index, parent.clone());
        if candidate
            .tracks
            .iter()
            .find(|track| track.id == id)
            .unwrap()
            .parent_id
            != parent
        {
            return Err("private track parent is missing or cyclic".into());
        }
        // Capture：轨道是本会话自己的，父子边**就是**权威（没有宿主会覆盖它）。
        // 只记一次历史并通知，不走宿主分组通道 —— 那条通道写的是宿主撤销栈。
        let Some(document) = self.document() else {
            self.checkpoint_timeline(&before, HistoryOp::MoveTrack);
            *self.timeline.lock_or_recover() = candidate;
            self.mark_dirty();
            self.render_requested.store(true, Ordering::Release);
            self.schedule_analysis();
            self.notify_timeline();
            return Ok(());
        };
        let groups = super::private_groups::TrackGroups::capture(
            &candidate,
            &document.group_aliases(&self.namespace),
        )?;
        let versions = document.editor_versions()?;
        let selected = candidate
            .selected_clip_id
            .as_deref()
            .map(|id| id.strip_prefix(&self.namespace).unwrap_or(id).to_owned());
        let (_, grouped) = document.private_parameter_views(selected, Some(&groups))?;
        let revision = {
            let _transaction = document.transaction.lock_or_recover();
            let mut edits = document.edits.lock_or_recover();
            if (
                edits.revision,
                document.revision.load(Ordering::Acquire),
                document.scope_revision.load(Ordering::Acquire),
            ) != versions
            {
                return Err("Conflict: host changed during private track grouping".into());
            }
            let revision = edits
                .revision
                .checked_add(1)
                .ok_or("edit revision exhausted")?;
            edits.groups = groups;
            edits.revision = revision;
            revision
        };
        candidate.params_by_root_track = grouped
            .params_by_root_track
            .into_iter()
            .map(|(root, params)| (format!("{}{root}", self.namespace), params))
            .collect();
        self.checkpoint_timeline(&before, HistoryOp::MoveTrack);
        *self.timeline.lock_or_recover() = candidate;
        {
            let mut loaded = self.loaded.lock_or_recover();
            loaded.edit = revision;
            loaded.projection = document.workspace_projection()?;
        }
        self.mark_dirty();
        self.render_requested.store(true, Ordering::Release);
        self.schedule_analysis();
        self.notify_timeline();
        Ok(())
    }
    /// 每个真实文档唯一actor；worker与会话均用weak，不形成document→actor→document循环。
    pub(crate) fn new(document: &Arc<DocumentSession>) -> Result<Arc<Self>, String> {
        Self::new_with(SessionBacking::Ara(Arc::downgrade(document)))
    }

    /// Capture 形态的会话：宿主没给 ARA，时间线来自捕获存储。
    ///
    /// 【为什么可以没有文档】Capture 的全部内容都是本插件自己录进来的：没有宿主
    /// 几何、没有宿主撤销栈、参数就是权威。文档在这里唯一的作用是"时间线从哪来"，
    /// 而那件事由 [`SessionBacking`] 回答。
    pub(crate) fn new_capture(
        owner: &Arc<crate::render::extension::ExtensionOwner>,
    ) -> Result<Arc<Self>, String> {
        Self::new_with(SessionBacking::Capture(Arc::downgrade(owner)))
    }

    fn new_with(backing: SessionBacking) -> Result<Arc<Self>, String> {
        // 【为什么这里不再做 temp sweep】`remove_dir_all` 在 Windows 上可能耗时数秒；
        // 本函数跑在宿主 UI 线程上，而它的 ARA 调用点还持有 `DocumentSession::transaction`。
        // 清扫改由会话 actor 的空闲 tick 分片执行（见 `temp_sweep::sweep_step`）。
        super::resources::initialize_models();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let namespace = format!(
            "hfs-ui-{}-{}-",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let (queue, receiver) = mpsc::sync_channel(32);
        let (analysis_sender, analysis_updates) = mpsc::channel();
        // 【为什么从设置里播种网格】网格是 HiFiShifter 自有的编辑设置（宿主没有对应
        // 概念），存在进程级的 `settings_store` 里。`ProjectState::default()` 只带出厂
        // 值，不播种的话用户在插件里设过的网格会在换工程/重启后归零 —— 而设置本该
        // 属于用户、不属于某一个工程。
        let initial_project = {
            let settings = crate::settings_store::settings();
            let mut project = ProjectState::default();
            project.grid_size =
                hifishifter_kernel::config::TimelineSnapSettings::normalize_grid_size(
                    &settings.grid_size,
                );
            // 音阶是 HiFiShifter 自有的设置（宿主没有对应概念），必须从用户设置里
            // 播种 —— 否则用户在插件里选过的音阶会在换工程/重启后归零，而渲染缓存键
            // 与级数渲染都锚定它。
            let musical = settings.plugin_musical_context;
            project.base_scale =
                hifishifter_kernel::state::model::normalize_scale_key(&musical.base_scale);
            project.use_custom_scale = musical.use_custom_scale && musical.custom_scale.is_some();
            project.custom_scale = musical.custom_scale.map(|scale| scale.normalized());
            // 记事本正文同理：插件没有工程文件，正文存在插件自己的数据目录里
            // （见 `editor/notebook.rs`）。不在这里播种的话，`get_project_meta`
            // 会回报空正文，前端面板一片空白 —— 而磁盘上其实有内容。
            project.notes_markdown = super::notebook::store().notes();
            project
        };
        let session = Arc::new(Self {
            backing,
            timeline: Mutex::new(TimelineState::default()),
            pending_capture_edits: Mutex::new(None),
            history: Mutex::new(Default::default()),
            project: Mutex::new(initial_project),
            loaded: Mutex::new(Default::default()),
            // 渲染用的 PCM 暂存同样搬出 `%TEMP%`：磁盘清理在会话进行中删掉这些
            // 文件会让正在进行的渲染失败。命名空间含 pid 与序号，多个宿主进程
            // 之间不会互相覆盖。
            pcm_dir: hifishifter_kernel::config_location::local_data_subdir("pcm").join(&namespace),
            namespace,
            browser_roots: Mutex::new(Vec::new()),
            clipboard_midi: Mutex::new(std::collections::VecDeque::new()),
            display_waveforms: Mutex::new(HashMap::new()),
            peaks: Mutex::new(HashMap::new()),
            queue,
            worker: Mutex::new(None),
            analysis_sender,
            analysis_updates: Mutex::new(analysis_updates),
            views: Mutex::new(HashMap::new()),
            analysis_cancel: Arc::new(AtomicBool::new(false)),
            analysis_workers: Mutex::new(Vec::new()),
            generation: AtomicU64::new(0),
            timeline_revision: AtomicU64::new(0),
            revision_bumps_dirty: AtomicU64::new(0),
            revision_bumps_publish: AtomicU64::new(0),
            revision_bumps_analysis: AtomicU64::new(0),
            last_revision_report: Mutex::new(None),
            clip_count: AtomicU64::new(0),
            analysis_abandoned: Mutex::new(std::collections::BTreeSet::new()),
            submitted: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            applied: AtomicU64::new(0),
            host_version: AtomicU64::new(0),
            ui_geometry_version: AtomicU64::new(0),
            project_duration: AtomicU64::new(0),
            error: Mutex::new(None),
            render_error: Mutex::new(None),
            render_requested: AtomicBool::new(false),
            render_task: Mutex::new(None),
            closed: AtomicBool::new(false),
            suppress_history: AtomicBool::new(false),
            transport_probe: std::env::var_os("HIFISHIFTER_ARA_TRANSPORT_PROBE").is_some(),
            transport_probe_next: Mutex::new(Instant::now()),
            host_selection: Mutex::new(None),
            reload_requested: AtomicBool::new(false),
            initial_retry_at: Mutex::new(None),
        });
        let weak = Arc::downgrade(&session);
        let worker = std::thread::Builder::new()
            .name("hfs-embedded-editor".into())
            .spawn(move || Self::run(weak, receiver))
            .map_err(|e| format!("create editor command worker: {e}"))?;
        *session.worker.lock_or_recover() = Some(worker);
        super::events::register(&session);
        Ok(session)
    }

    /// 本会话是否服务 Capture 形态（没有 ARA 文档）。
    pub(crate) fn is_capture(&self) -> bool {
        matches!(self.backing, SessionBacking::Capture(_))
    }

    /// 活着的 ARA 文档；Capture 形态恒为 `None`。
    pub(super) fn document(&self) -> Option<Arc<DocumentSession>> {
        match &self.backing {
            SessionBacking::Ara(document) => document.upgrade(),
            SessionBacking::Capture(_) => None,
        }
    }

    /// 需要文档的路径用这个取；Capture 形态给出与"文档已关闭"同一句错误。
    fn require_document(&self) -> Result<Arc<DocumentSession>, String> {
        self.document()
            .ok_or(crate::editor::errors::DOCUMENT_CLOSED.error())
    }

    /// Capture 形态的 owner；ARA 形态恒为 `None`。
    pub(super) fn capture_owner(&self) -> Option<Arc<crate::render::extension::ExtensionOwner>> {
        match &self.backing {
            SessionBacking::Capture(owner) => owner.upgrade(),
            SessionBacking::Ara(_) => None,
        }
    }

    /// 时间线代次（`mark_dirty` 与 `publish_timeline` 任一都推进）。
    ///
    /// 【用途】Capture 渲染的版本闸。它比 `generation` 更贴近"渲染读的东西变没变"：
    /// 任何改了时间线的路径都会推进它，包括不标脏的那些（见字段说明）。
    /// `generation` 本身不再对外暴露 —— 它留在 `pending`/`applied` 那套进度读数里，
    /// 而那套读数问的是"编辑应用了没有"，与"渲染读的东西变没变"是两件事。
    pub(crate) fn timeline_revision(&self) -> u64 {
        self.timeline_revision.load(Ordering::Acquire)
    }

    /// 【B4】把"渲染版本是被谁推进的"打成一行（非音频线程，20 ms 定时器驱动）。
    ///
    /// 【判读方式】`dirty` 是用户/命令编辑，`analysis` 是原线补齐，
    /// `publish` 是全部发布（含 analysis）。健康状态是：播放期间三者都**不涨**；
    /// 用户在画线时 `dirty` 与 `publish` 同步涨、`analysis` 不涨。
    ///
    /// 若播放期间 `analysis` 与 `publish` 同步上涨而用户没碰任何东西，
    /// 那就是"内部路径每轮都在推进版本"—— 它会经版本闸变成周期性重渲染。
    pub(crate) fn log_revision_bumps(&self) {
        let triple = (
            self.revision_bumps_dirty.load(Ordering::Relaxed),
            self.revision_bumps_publish.load(Ordering::Relaxed),
            self.revision_bumps_analysis.load(Ordering::Relaxed),
        );
        let total = self.timeline_revision.load(Ordering::Acquire);
        let mut last = self.last_revision_report.lock_or_recover();
        if *last == Some(triple) {
            return;
        }
        *last = Some(triple);
        log::warn!(
            "[capture] revision: total={total} dirty={} publish={} analysis={}",
            triple.0,
            triple.1,
            triple.2,
        );
    }

    /// 时间线是否已经加载过（Capture 的渲染要等这一份就绪才用得上用户曲线）。
    pub(crate) fn is_loaded(&self) -> bool {
        self.loaded.lock_or_recover().initialized
    }

    /// 已发布片段数；健康探活/状态栏用，不取时间线锁。
    pub(crate) fn clip_count(&self) -> u64 {
        self.clip_count.load(Ordering::Acquire)
    }

    /// 时间线上是否有**可保存的用户内容**（不要求 `is_loaded`）。
    ///
    /// 【为什么不能只看 `is_loaded`】`setState` 恢复后、actor 第一次加载前，时间线
    /// 可能已经被测试/恢复路径直接写入；此时 live 编辑就是权威，不能因为"还没走
    /// 加载流程"而回落到空载荷。
    pub(crate) fn has_live_capture_edits(&self) -> bool {
        let timeline = self.timeline.lock_or_recover();
        !timeline.params_by_root_track.is_empty()
            || !timeline.tracks.is_empty()
            || !timeline.project_scale_notes.is_empty()
            || timeline.tempo_map.is_some()
            || !timeline.disabled_group_ids.is_empty()
    }

    /// 把界面时间线还原成**音频权威**形态，供 Capture 渲染使用。
    ///
    /// 【为什么必须还原】界面上的 `source_path` 是本会话的私有分析副本路径（波形要从
    /// 磁盘读），而 `RenderInput.sources` 按**源 id** 索引 —— 不做这一步，渲染会因为
    /// "找不到素材"失败。参数曲线不受影响：它就是时间线本身。
    pub(crate) fn render_timeline(&self) -> Result<TimelineState, String> {
        self.native_timeline(self.timeline.lock_or_recover().clone())
    }

    /// 把本会话里**属于用户**的那一部分时间线序列化出来（Capture 存档用）。
    ///
    /// 【为什么只带用户内容】几何（片段、轨道集合）由捕获存储重新投影 —— 素材变了，
    /// 几何就必须跟着变。把旧几何存回去会让片段指向不存在的音频。详见
    /// `editor::capture_edits`。
    ///
    /// 【为什么 ARA 形态也允许调用】调用方已经按形态分支；这里不做形态判断，
    /// 是为了让"存的是不是用户内容"这一条只有一个定义。
    pub(crate) fn serialize_capture_edits(&self) -> Result<Vec<u8>, String> {
        super::capture_edits::capture(&self.timeline.lock_or_recover(), &self.namespace).encode()
    }

    /// 会话尚未加载时间线时，序列化**恢复载荷**。
    ///
    /// 【为什么需要】`setState` 恢复出来的曲线先暂存在这里；用户重开工程后如果
    /// 没打开插件窗口就再次保存，`serialize_capture_edits()` 会读到一份空的
    /// `self.timeline`，把恢复出来的曲线覆盖成默认值。
    pub(crate) fn serialize_pending_capture_edits(&self) -> Option<Vec<u8>> {
        self.pending_capture_edits
            .lock_or_recover()
            .as_ref()
            .and_then(|edits| edits.encode().ok())
    }

    /// 暂存从组件 state 恢复出来的用户内容，等下一次时间线加载时叠上去。
    ///
    /// 【为什么只暂存、不立刻应用】`load_capture_timeline` 会用投影重建整份时间线；
    /// 在它之前写进 `self.timeline` 的内容要么被重建覆盖，要么与 `kept` 机制打架。
    /// 暂存下来、由加载路径统一叠加，是唯一不会丢失的时序。
    pub(crate) fn stash_serialized_capture_edits(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let edits = super::capture_edits::CaptureEdits::decode(bytes)?;
        *self.pending_capture_edits.lock_or_recover() = Some(edits);
        Ok(())
    }

    /// 暂存里的根轨道 id（测试用）。
    #[cfg(test)]
    pub(crate) fn staged_capture_edit_roots(&self) -> Vec<String> {
        self.pending_capture_edits
            .lock_or_recover()
            .as_ref()
            .map(|edits| edits.params_by_root_track.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// 宿主撤销栈是否在途（"宿主正在写，别插队"）。
    ///
    /// 【为什么 Capture 恒为 false】宿主撤销栈是有原生扩展 / ARA 写口的后端才有的
    /// 概念，Capture 的每一次写入都是本会话自己的。文档缺席同理不是理由：ARA 下
    /// "文档消失"意味着会话已结束，而 Capture 根本没有文档。
    fn host_write_pending(&self) -> bool {
        match self.document() {
            Some(document) => document.host_undo.pending.load(Ordering::Acquire),
            None => !self.is_capture(),
        }
    }

    /// 请求"下一轮 tick 重新加载时间线"。
    ///
    /// 由视图层的 `capture_*` 命令在录制结束 / 切换 take / 清空之后调用。加载要读
    /// 捕获存储并物化分析副本（文件 I/O），不能在那条 UI 线程上做。
    pub(crate) fn request_reload(&self) {
        self.reload_requested.store(true, Ordering::Release);
        // 【为什么不再推进 `submitted`】`submitted` / `processed` 是"**用户编辑**"的账本：
        // `processed` 只在 `Job::Request` 到达且 `mutates_audio` 为真时兑付（:984），
        // 而重载既不产生 `Job::Request`、也不在 `mutates_audio` 里
        // （`commands.rs:11-31`）。推进它会让 `processed < submitted`（:912）恒真 ⇒
        // actor 每 1ms 空转一次、`start_render` 永远走不到 ⇒ 一次合成都排不出去 ⇒
        // 用户永远只听到直通（用户报障）。
        //
        // 重载的即时性改由一个**专用唤醒**保证：它只让 actor 立刻回到循环顶部，
        // 不占用任何票据。队列满时（有界 32）`try_send` 失败也没关系 ——
        // `reload_requested` 已经置位，actor 最迟在空闲的 100ms 轮询里消费它。
        let _ = self.queue.try_send(Job::Wake);
    }
    /// 有界32任务FIFO；关view取消回信，真实组件入口撤销则拒绝排队请求执行。
    pub fn enqueue(&self, request: UiRequest) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Err(crate::editor::errors::FX_PROCESSOR_CLOSED.error());
        }
        // 【为什么 Capture 允许没有文档】ARA 形态下 route 租约回答的是"这个视图属于
        // **这份文档**吗"（`EditorLink::authorize` 最后比的正是文档身份）。Capture
        // 没有文档，视图的归属由 `EditorLink` 自己的 owner 决定 —— 那条校验在
        // `webview` 建链时已经做过一次，这里不必（也无法）再比文档。
        let document = self.document();
        let lease = match (document.as_ref(), request.link.as_ref()) {
            (Some(document), Some(link)) => Some(link.authorize(document)?),
            _ => None,
        };
        if request.sink.view_id.is_empty()
            || (request.link.is_some() && request.sink.closed.load(Ordering::Acquire))
        {
            return Err(crate::editor::errors::VIEW_CLOSED_OR_UNKNOWN.error());
        }
        let mut views = self.views.lock_or_recover();
        if self.closed.load(Ordering::Acquire) {
            return Err("editor session closed".into());
        }
        if request.link.is_some()
            && views
                .get(&request.sink.view_id)
                .is_some_and(|view| !Arc::ptr_eq(&view.sink.closed, &request.sink.closed))
        {
            return Err(crate::editor::errors::VIEW_IDENTITY_MISMATCH.error());
        }
        let sink = request.sink.clone();
        let route = request.link.clone().zip(lease);
        // 播放观察不排到神经合成后面。这里只读原子缓存，不执行actor命令/host API/IO。
        // 原始view身份及同文档route授权仍与普通队列入口一致。
        if request.command == "get_playback_state" {
            if document
                .as_ref()
                .is_some_and(|document| !document.is_alive())
            {
                return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
            }
            let value = self.playback_state();
            if let (Some(document), Some(link)) = (document.as_ref(), request.link.as_ref()) {
                if Some(link.authorize(document)?) != lease {
                    return Err("editor route changed".into());
                }
            }
            views.insert(
                sink.view_id.clone(),
                RegisteredView {
                    sink: sink.clone(),
                    route,
                },
            );
            sink.response(
                json!({"version":1,"viewId":sink.view_id,"id":request.id,"ok":true,"value":value}),
            );
            return Ok(());
        }
        let mutates = super::commands::mutates_audio(&request.command);
        // 与worker的检查共用views锁；只有成功入队才登记view和推进合并票据。
        self.queue
            .try_send(Job::Request(request, lease))
            .map_err(|e| format!("editor queue unavailable: {e}"))?;
        views.insert(sink.view_id.clone(), RegisteredView { sink, route });
        if mutates {
            self.submitted.fetch_add(1, Ordering::AcqRel);
            self.cancel_render();
        }
        Ok(())
    }
    /// getState的非实时屏障：先排完已收到的尾块，不需要等音频快照才持久化曲线。
    ///
    /// 【为什么超时返回 `Err` 而不是静默 `Ok`】"flush 成功"是调用方的**前置条件**：
    /// `encode_state` 依赖已排队的编辑都已落进 timeline，撤销/恢复的测试也依赖这一点。
    /// 把超时吞掉会让调用方在**未排空**的状态上继续工作 —— 那比一次可诊断的失败糟得多。
    /// 上限从 30 s 收到 10 s：仍足以排空正常积压，又不会在异常情况下把宿主 UI 线程
    /// 卡到用户以为程序死了。
    pub fn flush(&self) -> Result<(), String> {
        self.flush_with_budget(Duration::from_secs(10))
    }

    /// 有预算的 flush：**不阻塞宿主保存线程**。
    ///
    /// 【为什么用 `try_send`】队列是有界 32 的同步通道；`send` 在队列满时会阻塞，
    /// 而调用点是宿主的保存线程。保存预算内排不进去就应该立即返回"没排空"，
    /// 让调用方用最后一份已提交状态继续，而不是把宿主 UI 冻住。
    pub(crate) fn flush_with_budget(&self, budget: Duration) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let (sender, receiver) = mpsc::channel();
        match self.queue.try_send(Job::Barrier(sender)) {
            Ok(()) => receiver
                .recv_timeout(budget)
                .map_err(|e| format!("editor flush: {e}")),
            Err(mpsc::TrySendError::Full(_)) => Err("editor queue busy".into()),
            Err(mpsc::TrySendError::Disconnected(_)) => Err("editor queue closed".into()),
        }
    }
    /// 在组件stop返回前撤销其回信/订阅；排队任务仍会独立核对原始route代次。
    pub(crate) fn revoke_closed_views(&self) {
        let document = self.document();
        self.views.lock_or_recover().retain(|_, view| {
            let valid = view.route.as_ref().is_none_or(|(link, lease)| {
                document.as_ref().is_some_and(|document| {
                    link.authorize(document)
                        .is_ok_and(|current| current == *lease)
                })
            });
            if !valid {
                view.sink.closed.store(true, Ordering::Release);
            }
            valid
        });
    }
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        {
            let mut views = self.views.lock_or_recover();
            for view in views.values() {
                view.sink.closed.store(true, Ordering::Release);
            }
            views.clear();
        }
        self.analysis_cancel.store(true, Ordering::Release);
        self.cancel_render();
        self.submitted.fetch_add(1, Ordering::AcqRel);
        let _ = self.queue.send(Job::Close);
        let worker = self.worker.lock_or_recover().take();
        if let Some(worker) = worker {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
        let render = self.render_task.lock_or_recover().take();
        if let Some(render) = render {
            render.cancel.store(true, Ordering::Release);
            if render.worker.thread().id() != std::thread::current().id() {
                let _ = render.worker.join();
            }
        }
        for worker in self.analysis_workers.lock_or_recover().drain(..) {
            let _ = worker.join();
        }
        // 旧view可继续持有已关闭actor的Arc；join之后释放分析投影/历史/波形缓存。
        *self.timeline.lock_or_recover() = Default::default();
        *self.history.lock_or_recover() = Default::default();
        *self.loaded.lock_or_recover() = Default::default();
        self.peaks.lock_or_recover().clear();
        // MIDI 剪贴板此前只增不减、连 close 都不清 —— 每次"读 MIDI 剪贴板"都往
        // 队列尾部追加一份字节，长会话里就是一条无界增长。
        self.clipboard_midi.lock_or_recover().clear();
    }
    fn run(weak: Weak<Self>, receiver: mpsc::Receiver<Job>) {
        let mut deadline: Option<Instant> = None;
        // 自动渲染的**重试预算**状态（见 `RENDER_RETRY_BUDGET`）。`submitted` 是"用户
        // 做了新操作"的权威信号（`enqueue` 对 `mutates_audio` 命令自增它），因此用它
        // 做清零键：上限只挡住"没人操作时的空转"，不挡住正常重试。
        let retry_submitted = weak
            .upgrade()
            .map(|session| session.submitted.load(Ordering::Acquire))
            .unwrap_or(0);
        let mut retry = RetryState::new(retry_submitted);
        loop {
            if let Some(session) = weak.upgrade() {
                if session.closed.load(Ordering::Acquire) {
                    break;
                }
                let submitted_now = session.submitted.load(Ordering::Acquire);
                if submitted_now != retry.submitted {
                    retry.submitted = submitted_now;
                    retry.rearm();
                }
                session.refresh_host();
                // 【为什么补给线与自动 promote 挂在会话 actor 上】它是**周期性**维护，
                // 且必须在音频线程取用备用缓冲之前完成。此前它挂在 WebView 的 20ms
                // 定时器上 —— 于是关掉插件窗口就断录（2 粒 × 8 秒 = 24 秒后 `NoSpare`）。
                // actor 由会话创建，与窗口开没开无关；它空闲时 100ms 一轮（见循环末尾的
                // `recv_timeout`），相对 8 秒一粒的备用缓冲有两个数量级的余量。
                //
                // 【为什么还要在这里 promote】捕获会**自己**停下（到上限、备用缓冲用尽、
                // 采样率变化），而那些时刻没有任何用户动作。没有这一条，已经录好的段会
                // 一直留在捕获里：界面显示"尚未录制"、素材却在内存里，而下一次
                // `start()` 会把它们覆盖掉 —— "到点自动停"于是变成"到点丢掉"。
                if let Some(owner) = session.capture_owner() {
                    owner.capture().service();
                    // 【为什么原始磁带也要在这里补】它此前只在**用户动作**上发布
                    // （`capture_stop` / 清空 / 删 take 都走 `request_capture_reload`）。
                    // 而 take 还有第二条来源：宿主重开工程时从组件 state 的清单恢复
                    // （`restore_manifest`）。那种会话里没有任何用户动作，原始磁带就
                    // 一份都没有 —— 于是输出仲裁落到"透传宿主原声"：用户播放时听到的
                    // 永远是直通，而素材明明在时间轴上（用户报障）。挂在 actor 心跳上
                    // 之后，"有素材就有声音兜底"不再取决于谁碰过按钮。
                    // 幂等：素材代次没变时它只做一次原子比较。
                    owner.sync_raw_tape();
                    // 一次性诊断：宿主给了位置却从来没给过输入（见该方法的说明）。
                    owner.capture().report_missing_input_once();
                    // 宿主链接的回写监视。与补缓冲同一个理由挂在会话 actor 上：
                    // actor 由**会话**创建，与"插件窗口开没开"无关（见 `render::link::poll`）。
                    crate::render::link::poll(&owner);
                    // 【为什么这里补一次调度】音频线程发现"有素材但处理结果不是当前代次"
                    // 时只置一个原子（`ExtensionOwner::capture_stage` 的 `render_wanted`）。
                    // 真正的合成仍由本 actor 的渲染任务执行 —— 它自带在途检查、取消与
                    // 重试预算。此前合成只由用户动作触发，一次票据事故就能让它无限期
                    // 停摆，而界面照样显示素材已就绪（用户报障："永远是直通"）。
                    if owner.take_render_wanted()
                        && !session.render_requested.load(Ordering::Acquire)
                    {
                        session.render_requested.store(true, Ordering::Release);
                        deadline = Some(Instant::now());
                    }
                    // 一次性诊断：素材在、结果不在，就把这件事写进日志（按代次去重）。
                    owner.report_pending_synthesis_once();
                    // 起播窗口的块级 trace：只在音频线程标记了可疑窗口时落一次。
                    // 一次即可回答“0.7 s 是一个大块还是很多 kPlaying=0 的小块”。
                    if owner.take_trace_wanted() {
                        log::info!(
                            "[capture] suspicious start-window block trace {}",
                            owner.block_diagnostics_json(64)
                        );
                    }
                    // 【为什么这些读数挂在这里】`[capture] pass` / `[capture] clock` /
                    // `[capture] ctx` 此前只由 `ExtensionOwner::prepare()` 驱动，而
                    // VST3 宿主**在播放期间根本不会调 prepare()** —— 40 分钟的现场日志里
                    // 这三行一行都没有，于是每一轮排查都只能靠"用户再听一遍"。
                    //
                    // 会话 actor 是唯一"与窗口开没开无关、又不在音频线程上"的周期点：
                    // 它在录制结束时就存在（`arm_capture_rendering` 建它），并且每轮
                    // 都会跑。音频线程只写原子，格式化留在这里。
                    owner.log_click_report();
                    owner.log_capture_context();
                    owner.log_capture_report();
                    owner.log_buffer_audit();
                    // 【B4】版本推进的归因。上面几行回答"输出了什么"，这一行回答
                    // "谁让它重算的" —— 后者此前只能靠猜。
                    session.log_revision_bumps();
                }
                // Capture：录制结束 / 切换 take / 清空由视图层置位（那里是 UI 线程，
                // 不能做文件 I/O）。加载在这里消费。
                if session.reload_requested.swap(false, Ordering::AcqRel) {
                    if let Err(error) = session.ensure_loaded(true) {
                        *session.error.lock_or_recover() = Some(error);
                    }
                    session.emit_state();
                }
                if let Some(render) = session.render_task.lock_or_recover().as_ref() {
                    if render.ticket != session.submitted.load(Ordering::Acquire)
                        || render.generation != session.generation.load(Ordering::Acquire)
                        || session.host_write_pending()
                    {
                        render.cancel.store(true, Ordering::Release);
                    }
                }
                if let Some((generation, ticket, cancelled, result)) = session.complete_render() {
                    session.emit("playback_rendering_state",json!({"active":false,"progress":if result.is_ok()&&!cancelled {Some(1.0)} else {None::<f64>},"target":"background"}));
                    let current = !cancelled
                        && ticket == session.submitted.load(Ordering::Acquire)
                        && generation == session.generation.load(Ordering::Acquire);
                    if current {
                        match result {
                            Ok(()) => {
                                session.applied.store(generation, Ordering::Release);
                                session.render_requested.store(false, Ordering::Release);
                                *session.render_error.lock_or_recover() = None;
                                retry.rearm();
                            }
                            Err(error) => {
                                let RenderAction::Retry(delay) = retry.classify(&session, error);
                                deadline = Some(Instant::now() + delay);
                            }
                        }
                    } else {
                        deadline = Some(Instant::now() + Duration::from_millis(150));
                    }
                    session.emit_state();
                }
                session.report_transport_probe();
                if session.refresh_analysis() {
                    // 分析有新进展 ⇒ 之前失败的前提可能已消失，重新武装重试。
                    retry.rearm();
                    deadline = Some(Instant::now() + Duration::from_millis(150));
                }
                if deadline.is_none()
                    && session.render_requested.load(Ordering::Acquire)
                    // 【为什么终态错误才停自动渲染】此前只要 `error` 非空就停 —— 而
                    // `Conflict: workspace scope or curves changed` 这类"宿主状态在我们
                    // 脚下变了一次"的**暂时**冲突也会被写成 error ⇒ 自动渲染永久停摆，
                    // 用户必须手动刷新。终态（panic 毒化 / 文档关闭 / 租约撤销）仍然停：
                    // 那些重放只会重复失败。
                    && session
                        .error
                        .lock_or_recover()
                        .as_deref()
                        .is_none_or(|error| super::classify_error(error) != "terminal")
                {
                    deadline = Some(Instant::now() + Duration::from_millis(150));
                }
                // 到期应用先于下一条只读轮询；持续get_playback_state不能令合成饥饿。
                if deadline.is_some_and(|d| d <= Instant::now()) {
                    let ticket = session.submitted.load(Ordering::Acquire);
                    if session.render_task.lock_or_recover().is_some()
                        || session
                            .document()
                            .is_some_and(|doc| doc.host_undo.pending.load(Ordering::Acquire))
                    {
                        deadline = Some(Instant::now() + Duration::from_millis(50));
                    } else if session.processed.load(Ordering::Acquire) < ticket {
                        // 【为什么是 `<` 而不是 `!=`】`submitted` 在 `try_send` **之后**自增，
                        // 而 worker 可能在自增前就取走并处理了该请求（`processed` 先涨）。
                        // 用 `!=` 时这个瞬时反超会被读成"还有待处理"，于是每 1ms 重试一次；
                        // 稳态下就是渲染请求风暴（或反之饿死）。语义只有"队列里还有未处理的
                        // 变更"—— 那就是 `processed < submitted`。
                        deadline = Some(Instant::now() + Duration::from_millis(1));
                    } else {
                        deadline = None;
                        if let Err(error) = session.start_render(ticket) {
                            session.emit("playback_rendering_state",json!({"active":false,"progress":None::<f64>,"target":"background"}));
                            let RenderAction::Retry(delay) = retry.classify(&session, error);
                            deadline = Some(Instant::now() + delay);
                            session.emit_state();
                        }
                    }
                }
            } else {
                break;
            }
            let wait = deadline
                .map(|d| d.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::from_millis(100));
            match receiver.recv_timeout(wait) {
                Ok(Job::Close) => break,
                Ok(Job::Barrier(sender)) => {
                    let _ = sender.send(());
                }
                // 一次重载是"用户确实做了个动作"，因此把重试预算也重新武装：
                // 否则此前失败过的合成会因预算耗尽而拒绝这一次的新请求。
                Ok(Job::Wake) => retry.rearm(),
                Ok(Job::Request(request, lease)) => {
                    let Some(session) = weak.upgrade() else {
                        break;
                    };
                    if session.closed.load(Ordering::Acquire) {
                        break;
                    }
                    let before = session.generation.load(Ordering::Acquire);
                    let mutates = super::commands::mutates_audio(&request.command);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if let Some(link) = &request.link {
                            // 【为什么 Capture 跳过租约复核】租约是"视图属于这份文档"的
                            // 凭据（`authorize` 比的是文档身份）；Capture 没有文档，
                            // 视图身份由下面的 view 表比对回答。
                            if let Some(document) = session.document() {
                                if Some(link.authorize(&document)?) != lease {
                                    return Err(
                                        crate::editor::errors::ROUTE_CHANGED_WHILE_QUEUED.error()
                                    );
                                }
                            } else if lease.is_some() {
                                return Err(
                                    crate::editor::errors::ROUTE_CHANGED_WHILE_QUEUED.error()
                                );
                            }
                            let views = session.views.lock_or_recover();
                            // view关闭仅取消回信；已入队尾笔仍持enqueue验证的身份与route代次。
                            // emit可能已移除closed view，但同名新view不能替代这次原始请求。
                            match views.get(&request.sink.view_id) {
                                Some(view)
                                    if !Arc::ptr_eq(&view.sink.closed, &request.sink.closed) =>
                                {
                                    return Err("editor view identity mismatch".into())
                                }
                                None if !request.sink.closed.load(Ordering::Acquire) => {
                                    return Err("editor view closed or unknown".into())
                                }
                                _ => {}
                            }
                        }
                        super::commands::dispatch(&session, &request.command, request.args)
                    }))
                    .unwrap_or_else(|_| Err("editor command panicked".into()));
                    if mutates {
                        session.processed.fetch_add(1, Ordering::AcqRel);
                    }
                    session.schedule_analysis();
                    if session.generation.load(Ordering::Acquire) != before {
                        deadline = Some(Instant::now() + Duration::from_millis(150));
                    }
                    let response = match result {
                        Ok(value) => {
                            json!({"version":1,"viewId":request.sink.view_id,"id":request.id,"ok":true,"value":value})
                        }
                        Err(error) => {
                            // 与 `webview.rs` 的同一条协议：终态错误必须带 `errorKind`，
                            // 否则前端无法区分"稍后重试"与"会话已结束"。
                            json!({"version":1,"viewId":request.sink.view_id,"id":request.id,"ok":false,
                                "errorKind":super::classify_error(&error),
                                "errorCode":super::errors::terminal_error_code(&error),"error":error})
                        }
                    };
                    request.sink.response(response);
                    session.emit_state();
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // 【为什么在这里清扫】只有队列空闲时才删一个陈旧临时目录。一次只删
                    // 一个，命令队列在下一个 tick 就能被处理 —— 不再出现"重开工程时
                    // 同步删 838 MB，把 get_runtime_info 拖到 30 s 超时"。
                    //
                    // 【为什么测试里不跑】测试进程会扫描/删除真实用户目录，既慢又危险
                    // （删一个大的 WebView2 profile 会让测试退出卡住）。
                    #[cfg(not(test))]
                    super::temp_sweep::sweep_step();
                    // 回到顶部统一调度，不要求命令队列出现空闲间隙。
                }
            }
        }
    }
    /// 新写入只设取消标记，不join或等待DSP，保证FIFO参数读写/flush始终可运行。
    fn cancel_render(&self) {
        if let Some(render) = self.render_task.lock_or_recover().as_ref() {
            render.cancel.store(true, Ordering::Release);
        }
    }
    fn complete_render(&self) -> Option<(u64, u64, bool, Result<(), String>)> {
        let mut task = self.render_task.lock_or_recover();
        let current = task.as_ref()?;
        let result = match current.reply.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Err("DSP worker disconnected".into()),
        };
        let current = task.take().unwrap();
        drop(task);
        let cancelled = current.cancel.load(Ordering::Acquire);
        let _ = current.worker.join();
        Some((current.generation, current.ticket, cancelled, result))
    }
    /// 原线缓存组装保留在actor；只把已确认的版本票据交给DSP线程，不并发修改timeline。
    fn prepare_apply(&self) -> Result<(Arc<DocumentSession>, u64, u64, String), String> {
        if let Some(error) = self.error.lock_or_recover().clone() {
            return Err(error);
        }
        self.complete_cached_analysis();
        if let Some(error) = self.error.lock_or_recover().clone() {
            return Err(error);
        }
        // 全零占位原线并非已完成的清音分析；收敛前保留旧快照及dirty代次。
        if self.requires_analysis(&self.timeline.lock_or_recover()) {
            return Err("pitch analysis pending".into());
        }
        let (edit, model, projection) = {
            let loaded = self.loaded.lock_or_recover();
            (loaded.edit, loaded.model, loaded.projection.clone())
        };
        let document = self.require_document()?;
        Ok((document, edit, model, projection))
    }
    fn start_render(self: &Arc<Self>, ticket: u64) -> Result<(), String> {
        if self.is_capture() {
            return self.start_capture_render(ticket);
        }
        let (document, edit, model, projection) = self.prepare_apply()?;
        let generation = self.generation.load(Ordering::Acquire);
        let cancel = Arc::new(AtomicBool::new(false));
        let current = self.clone();
        let job_cancel = cancel.clone();
        let progress_session = self.clone();
        let progress_cancel = cancel.clone();
        let progress_last = AtomicU64::new(0);
        let progress = hifishifter_kernel::mixdown::ProgressCallback::new(move |value| {
            if progress_cancel.load(Ordering::Acquire)
                || progress_session.closed.load(Ordering::Acquire)
                || progress_session.submitted.load(Ordering::Acquire) != ticket
                || progress_session.generation.load(Ordering::Acquire) != generation
            {
                return;
            }
            let value = value.clamp(0.0, 1.0);
            let previous = f64::from_bits(progress_last.load(Ordering::Acquire));
            if value + 1e-6 < previous {
                return;
            }
            progress_last.store(value.to_bits(), Ordering::Release);
            progress_session.emit(
                "playback_rendering_state",
                json!({"active":true,"progress":value,"target":"background"}),
            );
        });
        self.emit(
            "playback_rendering_state",
            json!({"active":true,"progress":0.0,"target":"background"}),
        );
        let worker = RenderTask::spawn(ticket, generation, cancel, move || {
            document.apply_workspace_edits_with_progress(
                edit,
                model,
                &projection,
                job_cancel.clone(),
                || {
                    !job_cancel.load(Ordering::Acquire)
                        && !current.closed.load(Ordering::Acquire)
                        && current.submitted.load(Ordering::Acquire) == ticket
                        && current.generation.load(Ordering::Acquire) == generation
                        && !document.host_undo.pending.load(Ordering::Acquire)
                },
                Some(progress),
            )
        })?;
        *self.render_task.lock_or_recover() = Some(worker);
        Ok(())
    }

    /// Capture 形态的渲染：把**带用户曲线的时间线**交给 owner 的 Capture 渲染准备。
    ///
    /// 【为什么参数在这里能生效】Capture 的时间线**就是**参数权威（没有宿主会重建它），
    /// 于是 `RenderInput.timeline` 里带着 `params_by_root_track`；`clip_parameters` 为空
    /// 时内核直接按时间线合成（见 `render/input.rs::mix_interleaved` 的第一支）。
    /// ARA 之所以需要参数图谱（`atlas.project_local`），是因为它的时间线每次都被宿主
    /// 重建、clip 身份由 region key 派生 —— Capture 没有那个问题。
    ///
    /// 【为什么要票据复核】渲染期间用户又改了参数 ⇒ 本次结果作废。与 ARA 的
    /// `apply_workspace_edits_with_progress` 的收尾复核同一条语义。
    fn start_capture_render(self: &Arc<Self>, ticket: u64) -> Result<(), String> {
        if let Some(error) = self.error.lock_or_recover().clone() {
            return Err(error);
        }
        self.complete_cached_analysis();
        if let Some(error) = self.error.lock_or_recover().clone() {
            return Err(error);
        }
        if self.requires_analysis(&self.timeline.lock_or_recover()) {
            return Err("pitch analysis pending".into());
        }
        let Some(owner) = self.capture_owner() else {
            return Err("capture host is gone".into());
        };
        let generation = self.generation.load(Ordering::Acquire);
        let cancel = Arc::new(AtomicBool::new(false));
        let current = self.clone();
        let job_cancel = cancel.clone();
        self.emit(
            "playback_rendering_state",
            json!({"active":true,"progress":0.0,"target":"background"}),
        );
        let worker = RenderTask::spawn(ticket, generation, cancel, move || {
            owner.capture_prepare_job(job_cancel.clone())?;
            if current.closed.load(Ordering::Acquire) {
                return Ok(());
            }
            if current.submitted.load(Ordering::Acquire) != ticket
                || current.generation.load(Ordering::Acquire) != generation
            {
                // 有更新的编辑在等着，本次结果已被后来者取代：让 actor 重新排一次。
                current.render_requested.store(true, Ordering::Release);
                return Err("capture render superseded".into());
            }
            Ok(())
        })?;
        *self.render_task.lock_or_recover() = Some(worker);
        Ok(())
    }

    #[cfg(test)]
    fn apply(
        &self,
        ticket: u64,
        progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
    ) -> Result<(), String> {
        let (document, edit, model, projection) = self.prepare_apply()?;
        document.apply_workspace_edits_with_progress(
            edit,
            model,
            &projection,
            self.analysis_cancel.clone(),
            || {
                !self.closed.load(Ordering::Acquire)
                    && self.submitted.load(Ordering::Acquire) == ticket
            },
            progress,
        )
    }
    /// 音高和气声等处理器效果都依赖完成的原线；纯混音/禁用算法不受此门禁阻塞。
    ///
    /// 【两个条件都必须来自内核】"这一根还没收敛"用
    /// `root_pitch_assembly_pending`，"有 clip 真的需要处理器"用
    /// `does_clip_need_processor_render`。两者此前各写各的：内核的分析门禁不看手绘
    /// 曲线，而这里看 —— 关掉 `合成` 但留着手绘曲线时，门禁在等一个永远不会被调度的
    /// 分析任务，`prepare_apply` 因此永远返回 `"pitch analysis pending"`（用户报障：
    /// 关掉合成后左下角一直"渲染中"）。现在分析门禁**覆盖**渲染门禁（见
    /// `assembly_gate_open` 的不变量注释），所以"pending"一定意味着"真的有人会产出它"。
    /// 【为什么还要减去 `analysis_abandoned`】"门禁在等"只说明**有理由等**，不说明
    /// "值得无限等"。一个卡死的分析任务（后端初始化失败、worker 永久阻塞）会让渲染
    /// 永远不跑；超时之后那根被记进 `analysis_abandoned`，这里就不再为它挡住渲染 ——
    /// 降级输出（内核在缺少原线时有既定回退）永远好过一条无法被满足的等待。
    fn requires_analysis(&self, timeline: &TimelineState) -> bool {
        let abandoned = self.analysis_abandoned.lock_or_recover();
        timeline.params_by_root_track.keys().any(|root| {
            !abandoned.contains(root)
                && hifishifter_kernel::pitch_analysis::analysis::root_pitch_assembly_pending(
                    timeline, root,
                )
                && timeline.clips.iter().any(|clip| {
                    timeline.resolve_root_track_id(&clip.track_id).as_deref() == Some(root.as_str())
                        && hifishifter_kernel::pitch_editing::does_clip_need_processor_render(
                            timeline,
                            clip,
                            clip.start_sec,
                        )
                })
        })
    }

    /// 此刻**正在等原线**的那些根（`RetryState` 的超时分支据此记账）。
    ///
    /// 【为什么取锁要分开】调用方（actor 循环）随后要写 `analysis_abandoned`，而
    /// `requires_analysis` 会在**持时间线锁**的情况下读它。两把锁的获取顺序必须处处
    /// 一致（先时间线、后 abandoned），所以这里先算完、把时间线锁放掉再交给调用方。
    fn pending_analysis_roots(&self) -> Vec<String> {
        let timeline = self.timeline.lock_or_recover();
        timeline
            .params_by_root_track
            .keys()
            .filter(|root| {
                hifishifter_kernel::pitch_analysis::analysis::root_pitch_assembly_pending(
                    &timeline, root,
                )
            })
            .cloned()
            .collect()
    }
    /// 投影撤销项目缓存键时，源分析可能早已全量命中；由actor组装，不等GUI读取/新通知。
    /// 只消费clip分析缓存，不做神经推理；尚未齐全仍pending，用户目标曲线保持。
    fn complete_cached_analysis(&self) {
        let roots = {
            let mut timeline = self.timeline.lock_or_recover();
            // 首次载入尚未get_param_frames时也必须有根状态，否则完成通知找不到接收原线的条目。
            let active_roots = timeline
                .clips
                .iter()
                .filter_map(|clip| timeline.resolve_root_track_id(&clip.track_id))
                .collect::<std::collections::BTreeSet<_>>();
            for root in active_roots {
                timeline.ensure_params_for_root(&root);
            }
            timeline
                .params_by_root_track
                .iter()
                .filter(|(_, params)| {
                    params.pitch_orig_key.is_none() || params.dyn_orig_key.is_none()
                })
                .map(|(root, _)| root.clone())
                .collect::<Vec<_>>()
        };
        if roots.is_empty() {
            return;
        }
        let analysis = Mutex::new(self.analysis_timeline());
        {
            let mut timeline = analysis.lock_or_recover();
            if let Some(selected) = timeline.selected_clip_id.clone() {
                if let Some(index) = timeline.clips.iter().position(|clip| clip.id == selected) {
                    let clip = timeline.clips.remove(index);
                    timeline.clips.push(clip);
                }
            }
        }
        for root in &roots {
            hifishifter_kernel::pitch_analysis::maybe_schedule_pitch_orig(&analysis, root);
            hifishifter_kernel::pitch_analysis::maybe_schedule_dyn_orig(&analysis, root);
        }
        let params = analysis.into_inner().unwrap().params_by_root_track;
        let mut changed = false;
        {
            let mut timeline = self.timeline.lock_or_recover();
            for root in roots {
                let Some(next) = params.get(&root) else {
                    continue;
                };
                let Some(current) = timeline.params_by_root_track.get_mut(&root) else {
                    continue;
                };
                // 【原线补齐不等于内容变了】`pitch_orig_key` 是**缓存失效标记**，
                // 不是内容本身：清掉它之后重新组装出的数组可能逐值相同（同一份
                // 素材、同一份分析结果）。那时渲染结果一个字都不会变，就不该推进
                // 渲染版本。
                //
                // 旧实现无条件 `changed = true`，于是"分析完成"这个本该一次性的
                // 事件变成了每轮都推进版本：版本闸触发重渲染 → 快照换版 →
                // 输出 token 变 → 装填一次接缝淡化（用户听到的咔）。
                //
                // 【为什么键仍然要写】它是失效标记：不写回去，下一轮 `roots` 仍然
                // 包含它，组装会无限重复。写键、但不推进版本 —— 这两件事必须分开。
                if current.pitch_orig_key.is_none() && next.pitch_orig_key.is_some() {
                    let edit_differs = !current.pitch_edit_user_modified
                        && current.pitch_edit != next.pitch_edit;
                    changed |= current.pitch_orig != next.pitch_orig || edit_differs;
                    current.pitch_orig = next.pitch_orig.clone();
                    if !current.pitch_edit_user_modified {
                        current.pitch_edit = next.pitch_edit.clone();
                    }
                    current.pitch_orig_key = next.pitch_orig_key.clone();
                    current.has_pitch_adjustment_active = next.has_pitch_adjustment_active;
                }
                if current.dyn_orig_key.is_none() && next.dyn_orig_key.is_some() {
                    changed |= current.dyn_orig != next.dyn_orig;
                    current.dyn_orig = next.dyn_orig.clone();
                    current.dyn_orig_key = next.dyn_orig_key.clone();
                }
            }
        }
        if changed {
            self.revision_bumps_analysis.fetch_add(1, Ordering::Relaxed);
            let updated = self.timeline.lock_or_recover().clone();
            self.publish_timeline(updated);
            self.render_requested.store(true, Ordering::Release);
        }
    }
    /// 把宿主的编辑器视图选择搬进时间线（A5：`notifySelection` → 插件选区）。
    ///
    /// 【为什么只在"宿主的选择变了"时才覆盖】`notifySelection` 的契约是"通知插件用户
    /// 在宿主里做了什么交互，插件按最合适的方式采纳"（`ARAInterface.h:3646-3655`），
    /// **不是**"这是唯一正确的选择" —— SDK 明说插件随时可以用自己的 UI 改内部选择。
    /// 因此这里与**上一次采纳过的**宿主选择比对：相同就不动，于是宿主重复发同一个
    /// 选择不会把用户在插件里刚点的那条抢回去。
    ///
    /// 【为什么取第一条】SDK：对象列表**按重要性排序，第一条是 focused**
    /// （`ARAInterface.h:3685-3689`）。单片段选择的插件只需第一条。
    ///
    /// 【为什么解析不出来就什么都不做】被选中的片段可能**已经销毁**（SDK 明说选中
    /// 不影响对象生命周期，`:3693-3694`），此时正确的反应是保持插件自己的选择，而不是清空。
    fn apply_host_selection(&self, timeline: &mut TimelineState) {
        let Ok(document) = self.document().ok_or(()) else {
            return;
        };
        let focused = document.renderer_owners().into_iter().find_map(|owner| {
            let selection = owner.host_view_selection()?;
            selection
                .playback_regions()
                .iter()
                .find_map(|key| document.clip_id_for(*key as u64))
        });
        self.adopt_host_selection(timeline, focused);
    }

    /// 采纳判据本身（纯逻辑，便于测试）：只有与上次不同的选择才覆盖。
    fn adopt_host_selection(&self, timeline: &mut TimelineState, observed: Option<String>) {
        let mut applied = self.host_selection.lock_or_recover();
        if *applied == observed {
            return;
        }
        *applied = observed.clone();
        if let Some(clip_id) = observed {
            log::debug!(
                "[ara] host editor selection focused clip {clip_id}; adopting it as the                  plugin selection"
            );
            timeline.selected_clip_id = Some(clip_id);
        }
    }

    pub(crate) fn ensure_loaded(&self, force: bool) -> Result<(), String> {
        if self.is_capture() {
            return self.load_capture_timeline(force);
        }
        let document = self.require_document()?;
        // GUI清单有独立代次：粘贴/删除可以先于ARA模型回调，不能被模型缓存早退吞掉。
        if self.loaded.lock_or_recover().initialized
            && document.renderer_owners().is_empty()
            && !self.timeline.lock_or_recover().clips.is_empty()
        {
            return Err("Conflict: no active renderer; local curves preserved".into());
        }
        if self.loaded.lock_or_recover().initialized {
            self.refresh_ui_geometry(&document);
        }
        let versions = document.editor_versions()?;
        {
            let mut loaded = self.loaded.lock_or_recover();
            let audio_current =
                loaded.audio == document.editor_audio_revision.load(Ordering::Acquire);
            if loaded.initialized
                && (loaded.edit, loaded.model, loaded.scope) == versions
                && audio_current
                && !force
            {
                return Ok(());
            }
            if loaded.initialized
                && loaded.model == versions.1
                && loaded.scope == versions.2
                && audio_current
                && !force
                && document.workspace_projection()? == loaded.projection
            {
                loaded.edit = versions.0;
                return Ok(());
            }
        }
        // generation 未应用不等于宿主冲突：workspace_snapshot 会把本地曲线
        // 叠加到最新宿主几何上。只有 ARA 还没有重新交付输入时才暂缓重载。
        //
        // 【为什么"会话中未就绪"不是冲突】`ready` 的语义是"ARA 是否已重新交付可用输入"，
        // 会被会话内的**真实内容变化**（几何、增删 region、源内容）按回 `false`。
        // 但那时宿主只是还没发 `endEditing` —— 这是必然收敛的**暂缓**，不是需要用户
        // 处置的冲突。此前二者共用一个 `ready`，于是用户每做一次编辑都撞一串
        // "尚未应用：Conflict: host changed"（自伤：插件自己的写回触发了它）。
        // 现在：会话中保留已有时间线，等 `endEditing` 自己收敛；只有**不在会话中**
        // 却拿不到输入，才是真正的宿主冲突。从未加载过时仍必须继续往下走，
        // 否则界面永远空白。
        if !force && !document.ready.load(Ordering::Acquire) {
            if document.editing_session.load(Ordering::Acquire) {
                if self.loaded.lock_or_recover().initialized {
                    return Ok(());
                }
            } else if !document.workspace_timeline()?.clips.is_empty() {
                // 【为什么不再一律返回 Conflict】宿主改完内容后可能既不发
                // `endEditing` 也不发 `notifyModelUpdates`（后台恢复、工程加载的
                // 中段）。那不是用户能处置的冲突 —— 报错只会让窗口永久停在旧读数上，
                // 而每一次后续操作都重复同一个错误。只要**我们确实观察到了**这次
                // 内容变化（`last_content_change` 有记录），就继续用最后一份已发布的
                // 时间线刷新界面（本地曲线仍叠加在上面），等宿主收口后由
                // `remap_and_log` 自然覆盖。
                //
                // 没有观察记录却拿到 `ready == false` 才是真冲突：那说明宿主声称
                // 输入不可用，而我们没有任何可归因的变化 —— 报出来，不要静默。
                // 【T2：先问"输入到底卡在哪一步"，再决定要不要报冲突】
                // `ready == false` 有三种完全不同的成因，此前共用一条 Conflict：
                //   * 宿主暂时收回了样本访问 / 还没读出 PCM（`AwaitingSamples`）——
                //     **不是冲突**，等一个 tick 就会好，报错只会让窗口停在旧读数上；
                //   * 宿主还没交付任何音频源（`AwaitingRegions`）—— 同上；
                //   * 授权在却读不出来（`Degraded`）—— 真故障，但它是**可行动**的，
                //     该由 `host_audio` 的分类说清楚，而不是一句"host changed"。
                // 只有"既没有读数、也没有观察记录"才保留原来的 Conflict 语义。
                let verdict = document.input_verdict();
                let known_cause = matches!(
                    verdict,
                    Some(
                        crate::render::document::InputVerdict::AwaitingSamples
                            | crate::render::document::InputVerdict::AwaitingRegions
                            | crate::render::document::InputVerdict::Degraded
                    )
                );
                let observed_change = document.last_content_change.lock_or_recover().is_some();
                if !observed_change && !known_cause {
                    return Err(
                        "Conflict: host changed; local curves preserved, host is not ready".into(),
                    );
                }
                let stale = document
                    .last_content_change
                    .lock_or_recover()
                    .is_none_or(|at| at.elapsed() >= crate::render::document::EDIT_SETTLE_GRACE);
                if stale {
                    hifishifter_kernel::log_limiter::emit_limited_key(
                        log::Level::Warn,
                        "ara.ready.stale",
                        format_args!(
                            "[ara] the host changed the model but never closed the edit \
                             session; publishing the last timeline until it does"
                        ),
                    );
                }
            }
        }
        let (snapshot, scope, projection) = document.workspace_snapshot()?;
        let mut timeline = snapshot.timeline;
        // 快照冻结take权威；先重建扁平投影，再判断真实的倒放/组合倍率。
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        // 正向倍率交给原kernel保调处理，不能再把全部拉伸挡在GUI外。
        //
        // 【倒放为什么不再是整份失败，也不再隔离】倒放是**宿主**的既有状态（REAPER 的
        // `SOURCE SECTION MODE` / 负 `PLAYRATE`），用户没做错任何事；而此前这里一句
        // `any(reversed)` 就 `Err`，于是"一个片段倒放"升级成"整个插件打不开"。
        // 隔离（`ReversePolicy::Isolate`）只解决了"不打死会话"，代价是倒放片段永远
        // 渲染不出来。F-4 已判明 ARA 报的是**镜像坐标**，播种层
        // （`project_host_take_facts_locked`）据此把窗口翻回正向，内核的倒放数学
        // 自会反向取窗 —— 所以现在**照常物化**（`Materialize`），倒放片段真的会渲染。
        if timeline.target_param_frames(timeline.frame_period_ms()) > 1_000_000 {
            return Err("ARA editor parameter frame budget exceeded for project span".into());
        }
        let views: Vec<_> = snapshot
            .sources
            .iter()
            .map(|(id, pcm)| PcmView {
                persistent_id: id,
                sample_rate: pcm.sample_rate,
                planes: &pcm.planes,
            })
            .collect();
        // 内容hash决定私有分析路径；model/edit代次和项目位置不应制造新的F0缓存身份。
        let dir = self.pcm_dir.join("sources");
        let (mut timeline, reverse_paths) = materialize_with_byte_limit(
            timeline,
            &views,
            &dir,
            crate::render::budget::global_budget().limit(),
            hifishifter_kernel::editor::host_pcm::ReversePolicy::Materialize,
        )?;
        // 旧状态或同URI换音频的原线不能假报就绪；已有完整clip cache由actor重新组装。
        //
        // 【为什么这里要撤销"放弃等待"】这一刻是"该为新素材重新分析"的唯一语义点：
        // 上一轮素材上卡死的分析结论不该跟着新素材走，否则新录的东西会被当成
        // "算不出来"，直接降级渲染（见 `analysis_abandoned` 的字段说明）。
        self.analysis_abandoned.lock_or_recover().clear();
        for params in timeline.params_by_root_track.values_mut() {
            params.pitch_orig_key = None;
            params.dyn_orig_key = None;
        }
        self.rewrite_ids(&mut timeline, true);
        self.retain_display_waveforms(&mut timeline, false);
        document.present_host_inventory(&mut timeline, &self.namespace);
        document.present_private_groups(&mut timeline, &self.namespace);
        // 【必须成对】清单写的是宿主**镜像**坐标，这一步把倒放窗口翻回正向 ——
        // 少了它，界面上的倒放窗口就与真实声音不一致（见 `project_host_facts`）。
        self.project_host_facts(&mut timeline);
        self.retain_display_waveforms(&mut timeline, true);
        // 音频迟到只能补齐资源，不能把用户已选中的新粘贴片段跳回首个clip。
        {
            let previous = self.timeline.lock_or_recover();
            if previous
                .selected_track_id
                .as_ref()
                .is_some_and(|id| timeline.tracks.iter().any(|track| &track.id == id))
            {
                timeline.selected_track_id = previous.selected_track_id.clone();
            }
            if previous
                .selected_clip_id
                .as_ref()
                .is_some_and(|id| timeline.clips.iter().any(|clip| &clip.id == id))
            {
                timeline.selected_clip_id = previous.selected_clip_id.clone();
            }
        }
        if timeline.selected_track_id.is_none() {
            timeline.selected_track_id = timeline.tracks.first().map(|t| t.id.clone());
        }
        if timeline.selected_clip_id.is_none() {
            timeline.selected_clip_id = timeline.clips.first().map(|c| c.id.clone());
        }
        self.apply_host_selection(&mut timeline);
        self.project_duration
            .store(timeline.project_sec.to_bits(), Ordering::Release);
        self.clip_count
            .store(timeline.clips.len() as u64, Ordering::Release);
        *self.timeline.lock_or_recover() = timeline;
        {
            let loaded = self.loaded.lock_or_recover();
            if force
                || !loaded.initialized
                || loaded.model != snapshot.model_revision
                || loaded.scope != scope
            {
                *self.history.lock_or_recover() = Default::default();
            }
        }
        *self.loaded.lock_or_recover() = Loaded {
            initialized: true,
            edit: snapshot.revision,
            model: snapshot.model_revision,
            scope,
            audio: snapshot.audio_revision,
            reverse_paths,
            projection,
        };
        *self.error.lock_or_recover() = None;
        *self.render_error.lock_or_recover() = None;
        self.applied
            .store(self.generation.load(Ordering::Acquire), Ordering::Release);
        self.peaks.lock_or_recover().clear();
        // snapshot返回后UI缓存可能再次更新，强制下一轮按短事务版本同步，不能吞掉该变化。
        self.ui_geometry_version.store(0, Ordering::Release);
        self.complete_cached_analysis();
        self.schedule_analysis();
        self.render_requested.store(true, Ordering::Release);
        self.host_version.fetch_add(1, Ordering::AcqRel);
        self.emit(
            "plugin_host_changed",
            json!({"version":self.host_version.load(Ordering::Acquire)}),
        );
        Ok(())
    }

    /// Capture 形态的加载：几何来自捕获存储，参数保留本会话已编辑的曲线。
    ///
    /// 【为什么每次都重建几何】捕获存储只有"活动 take"这一个版本量，没有 ARA 那样的
    /// 多级代次。重建的代价被 `materialize` 的内容寻址摊平（同一份 PCM 直接复用已有的
    /// 分析副本，见 `host_pcm::matches_authorized_pcm`），收益是"录制结束 / 切换 take /
    /// 清空之后界面必然跟着变"—— 不需要再维护一条失效通知通道。
    ///
    /// 【为什么参数要**保留**】`params_by_root_track` 是用户在这个会话里做的编辑，而捕获
    /// 存储产出的时间线只有几何：整份替换会把曲线清掉（那正是"改了音高，一录就又没了"
    /// 那类不可见错误）。轨道身份（`capture-track`）跨 take 稳定，因此按键保留是安全的。
    fn load_capture_timeline(&self, force: bool) -> Result<(), String> {
        let owner = self.capture_owner().ok_or("capture host is gone")?;
        // 【为什么版本是"素材代次"而不是 take 号】投影跨全部 take（后录的胜出），
        // 删掉一个较早的 take 同样会改变时间线 —— 而那一刻 take 号没变。
        let version = owner.capture().revision();
        {
            let loaded = self.loaded.lock_or_recover();
            if loaded.initialized && !force && loaded.model == version {
                return Ok(());
            }
        }
        let (timeline, sources) = owner.capture().session_timeline();
        // 【为什么恢复载荷要并进 `kept`】它必须与"本会话已编辑的曲线"走同一条规整
        // （`ensure_params_for_root`）。直接 insert 会绕过那一步，曲线长度对不上工程时
        // 界面会画到末尾之外。存档只在工程打开时到达，此时 `kept` 为空，因此让 `kept`
        // 覆盖存档是安全的：真同时存在时，用户当前的编辑更权威。
        let mut restored = self.pending_capture_edits.lock_or_recover().take();
        // 【为什么在这里绑定命名空间】载荷里的键是稳定的（不带 `hfs-ui-...-`），而
        // 下面 `rewrite_ids` 会把投影时间线改写成**当前会话**的命名空间。不绑定，
        // `kept.get(root)` 永远 miss —— 那正是"重开工程参数全丢"的直接原因。
        if let Some(edits) = restored.as_mut() {
            edits.bind_to_namespace(&self.namespace);
        }
        let mut kept: HashMap<String, TrackParamsState> = restored
            .as_ref()
            .map(|edits| {
                edits
                    .params_by_root_track
                    .iter()
                    .map(|(root, params)| (root.clone(), params.clone()))
                    .collect()
            })
            .unwrap_or_default();
        kept.extend(
            self.timeline
                .lock_or_recover()
                .params_by_root_track
                .iter()
                .map(|(root, params)| (root.clone(), params.clone())),
        );
        let views: Vec<PcmView<'_>> = sources
            .iter()
            .map(|(id, pcm)| PcmView {
                persistent_id: id,
                sample_rate: pcm.sample_rate,
                planes: &pcm.planes,
            })
            .collect();
        let dir = self.pcm_dir.join("sources");
        let (mut timeline, reverse_paths) = materialize_with_byte_limit(
            timeline,
            &views,
            &dir,
            crate::render::budget::global_budget().limit(),
            hifishifter_kernel::editor::host_pcm::ReversePolicy::Materialize,
        )?;
        // 先命名空间化，再按键对曲线 —— 保留的那份键已经是命名空间化的。
        self.rewrite_ids(&mut timeline, true);
        let roots = timeline
            .clips
            .iter()
            .filter_map(|clip| timeline.resolve_root_track_id(&clip.track_id))
            .collect::<std::collections::BTreeSet<_>>();
        // 只有"当前时间线上还存在的根"的旧曲线会被带过来；被删掉的那条不留残渣。
        timeline.params_by_root_track = roots
            .iter()
            .filter_map(|root| kept.get(root).map(|params| (root.clone(), params.clone())))
            .collect();
        // 无论有没有旧曲线都要走这一步：它把曲线规整到当前工程长度的目标帧数，并按
        // 新的源清掉对不上的原线缓存（保留用户显式编辑过的那部分）。
        for root in roots {
            timeline.ensure_params_for_root(&root);
        }
        // 曲线已随 `kept` 过完规整；其余用户内容（音阶 / Tempo Map / 私有分组 /
        // 轨道设置）只在这里叠一次 —— 它们在投影里没有对应来源。
        if let Some(edits) = restored.as_ref() {
            super::capture_edits::apply_project_state(&mut timeline, edits);
        }
        // 新 take 的源内容不同 ⇒ 缓存的原线/动态谱必须失效（与 ARA 加载同一条纪律）。
        // 撤销"放弃等待"同理：素材换了，就再给分析一次机会。
        self.analysis_abandoned.lock_or_recover().clear();
        for params in timeline.params_by_root_track.values_mut() {
            params.pitch_orig_key = None;
            params.dyn_orig_key = None;
        }
        // 宿主音乐读数是**只读**的宿主权威：Capture 没有文档，读 owner 自己的时钟。
        if let Some(clock) = owner.clock.get() {
            if let Some(tempo) = clock.tempo() {
                if tempo.is_finite() && tempo > 0.0 {
                    timeline.bpm = tempo;
                }
            }
            if let Some((numerator, denominator)) = clock.time_signature() {
                let mut project = self.project.lock_or_recover();
                project.beats_per_bar = (numerator as u32).clamp(1, 32);
                project.time_signature_denominator = (denominator as u32).clamp(1, 32);
            }
        }
        {
            let previous = self.timeline.lock_or_recover();
            if previous
                .selected_track_id
                .as_ref()
                .is_some_and(|id| timeline.tracks.iter().any(|track| &track.id == id))
            {
                timeline.selected_track_id = previous.selected_track_id.clone();
            }
            if previous
                .selected_clip_id
                .as_ref()
                .is_some_and(|id| timeline.clips.iter().any(|clip| &clip.id == id))
            {
                timeline.selected_clip_id = previous.selected_clip_id.clone();
            }
            // 【为什么要带过轨道自身的设置】投影每次都重建轨道，于是"用户在轨道头上
            // 关掉了合成 / 换了分析算法"会在下一次录制或删 take 之后被静默重置 ——
            // 而那两件事都会触发一次重新加载。轨道级设置与选择同性质：它们是**用户的
            // 选择**，不属于"素材的几何"。只在同名轨道确实存在过时才带过，因此
            // "第一次创建"仍然拿到投影给的默认值（合成开启，见 `capture/timeline.rs`）。
            for track in &mut timeline.tracks {
                let Some(old) = previous.tracks.iter().find(|t| t.id == track.id) else {
                    continue;
                };
                track.compose_enabled = old.compose_enabled;
                track.pitch_analysis_algo = old.pitch_analysis_algo.clone();
                track.volume = old.volume;
                track.muted = old.muted;
                track.solo = old.solo;
            }
        }
        if timeline.selected_track_id.is_none() {
            timeline.selected_track_id = timeline.tracks.first().map(|t| t.id.clone());
        }
        if timeline.selected_clip_id.is_none() {
            timeline.selected_clip_id = timeline.clips.first().map(|c| c.id.clone());
        }
        self.project_duration
            .store(timeline.project_sec.to_bits(), Ordering::Release);
        self.clip_count
            .store(timeline.clips.len() as u64, Ordering::Release);
        *self.timeline.lock_or_recover() = timeline;
        *self.loaded.lock_or_recover() = Loaded {
            initialized: true,
            edit: 0,
            model: version,
            scope: version,
            audio: version,
            reverse_paths,
            projection: format!("capture:{version}"),
        };
        *self.error.lock_or_recover() = None;
        *self.render_error.lock_or_recover() = None;
        self.peaks.lock_or_recover().clear();
        self.complete_cached_analysis();
        self.schedule_analysis();
        self.render_requested.store(true, Ordering::Release);
        self.host_version.fetch_add(1, Ordering::AcqRel);
        self.emit(
            "plugin_host_changed",
            json!({"version":self.host_version.load(Ordering::Acquire)}),
        );
        Ok(())
    }

    /// 清单结构与普通fade独立同步；ARA未ready时也通知新增/删除，不清空曲线、选择或历史。
    pub(super) fn refresh_ui_geometry(&self, document: &DocumentSession) {
        let version = document.ui_geometry_revision.load(Ordering::Acquire);
        if version == self.ui_geometry_version.load(Ordering::Acquire) {
            return;
        }
        // 先冻结可用fade，再合并清单；只确认开始时的代次，期间的新变化留到下一轮。
        let fades = document.ui_fade_projection().ok();
        {
            let mut timeline = self.timeline.lock_or_recover();
            self.retain_display_waveforms(&mut timeline, false);
            document.present_host_inventory(&mut timeline, &self.namespace);
            document.present_private_groups(&mut timeline, &self.namespace);
            // 与 `ensure_loaded` 同一条不变量：清单之后必须补一次事实投影，
            // 否则增量刷新会把已经复原的正向窗口重新打回镜像。
            document.project_host_take_facts_locked(&mut timeline);
            self.retain_display_waveforms(&mut timeline, true);
            if let Some((_, host)) = fades {
                for clip in &mut timeline.clips {
                    if let Some(original) = host
                        .clips
                        .iter()
                        .find(|original| format!("{}{}", self.namespace, original.id) == clip.id)
                    {
                        clip.muted = original.muted;
                        clip.gain = original.gain;
                        for take in &mut clip.takes {
                            take.gain = original.gain;
                        }
                        clip.snap_offset_sec = original.snap_offset_sec;
                        clip.fade_in_sec = original.fade_in_sec;
                        clip.fade_out_sec = original.fade_out_sec;
                        clip.auto_fade_in_sec = original.auto_fade_in_sec;
                        clip.auto_fade_out_sec = original.auto_fade_out_sec;
                        clip.fade_in_shape = original.fade_in_shape;
                        clip.fade_out_shape = original.fade_out_shape;
                        clip.fade_in_dir = original.fade_in_dir;
                        clip.fade_out_dir = original.fade_out_dir;
                    }
                }
            }
            self.project_duration
                .store(timeline.project_sec.to_bits(), Ordering::Release);
        }
        self.ui_geometry_version.store(version, Ordering::Release);
        self.notify_timeline();
    }
    /// 稳定模型变化在后台自动读取；pending曲线遇到真实冲突保留，不静默强制重载。
    fn refresh_host(&self) {
        let Some(document) = self.document() else {
            return;
        };
        let (initialized, edit, model, scope, audio) = {
            let loaded = self.loaded.lock_or_recover();
            (
                loaded.initialized,
                loaded.edit,
                loaded.model,
                loaded.scope,
                loaded.audio,
            )
        };
        if !initialized {
            // 【T4：失败一次不得变成永久失败】`initialized` 只在 `ensure_loaded` **成功**
            // 的末尾置真。此前这里直接返回 ⇒ 冷启动第一次加载失败之后，这条路径再也
            // 什么都不做，只剩 UI 命令还在重试；窗口一关就彻底没有驱动者。
            //
            // 慢速重试（1 s）把"失败一次"降级成"暂时失败"：成功即恢复正常节奏，
            // 成功路径的行为一个字都没变。
            let due = {
                let mut next = self.initial_retry_at.lock_or_recover();
                match *next {
                    Some(at) if at > Instant::now() => false,
                    _ => {
                        *next = Some(Instant::now() + Duration::from_secs(1));
                        true
                    }
                }
            };
            if due {
                if let Err(error) = self.ensure_loaded(false) {
                    if error.starts_with("Conflict") || error.starts_with("Unsupported") {
                        let changed = {
                            let mut current = self.error.lock_or_recover();
                            let changed = current.as_deref() != Some(error.as_str());
                            *current = Some(error);
                            changed
                        };
                        if changed {
                            self.emit_state();
                        }
                    }
                }
            }
            return;
        }
        *self.initial_retry_at.lock_or_recover() = None;
        self.refresh_ui_geometry(&document);
        // 原生媒体写入的短Undo窗口先回执GUI结构；音频物化留给写入收尾后的后台加载。
        if !document.host_undo.pending.load(Ordering::Acquire)
            && (document.editor_audio_revision.load(Ordering::Acquire) != audio
                || document
                    .editor_versions()
                    .is_ok_and(|versions| versions != (edit, model, scope)))
        {
            if let Err(error) = self.ensure_loaded(false) {
                if error.starts_with("Conflict") || error.starts_with("Unsupported") {
                    let changed = {
                        let mut current = self.error.lock_or_recover();
                        let changed = current.as_deref() != Some(error.as_str());
                        *current = Some(error);
                        changed
                    };
                    if changed {
                        self.emit_state();
                    }
                }
            }
        }
        // 宿主音乐上下文：BPM 与拍号都来自 VST3 进程上下文，同为**只读**的宿主权威
        // 读数。两者各自取锁（不嵌套），最后合成一次变更通知 —— 一次回调触发两次
        // 刷新会让 GUI 做两遍工作。
        let tempo_changed = match document.clock.tempo() {
            Some(tempo) => {
                let mut timeline = self.timeline.lock_or_recover();
                let changed = (timeline.bpm - tempo).abs() > 1e-6;
                timeline.bpm = tempo;
                changed
            }
            None => false,
        };
        // 【为什么拍号也写进来】此前只有 BPM 被下发，拍号虽然早在 `audio_abi` 的
        // 进程上下文里解析好了，却从来没人读 —— 于是插件里的拍号永远是工程默认值，
        // 而用户改不了它（那是宿主的东西）。读出来至少让界面说的是实话。
        let meter_changed = match document.clock.time_signature() {
            Some((numerator, denominator)) => {
                let mut project = self.project.lock_or_recover();
                // 与 `normalize_tempo_map` 同口径地夹取，不让宿主读数的越界值
                // 进入工程状态。
                let numerator = (numerator as u32).clamp(1, 32);
                let denominator = (denominator as u32).clamp(1, 32);
                let changed = project.beats_per_bar != numerator
                    || project.time_signature_denominator != denominator;
                project.beats_per_bar = numerator;
                project.time_signature_denominator = denominator;
                changed
            }
            None => false,
        };
        if tempo_changed || meter_changed {
            self.host_version.fetch_add(1, Ordering::AcqRel);
            self.emit(
                "plugin_host_changed",
                json!({"version":self.host_version.load(Ordering::Acquire)}),
            );
        }
        // 文档级宿主音乐读数：Tempo Map 的音阶点需要它当**宿主轴**
        // （0 位置点必须显式带拍号，每个点都要 BPM；见 `DocumentSession::host_meter`）。
        // per-view 的 `timeline.bpm` / `project.beats_per_bar` 只服务 GUI 显示，
        // 投影读的是这一份。
        //
        // 【为什么带阈值】读数每次回调都会来；不加阈值时浮点噪声会让投影缓存的键
        // 每轮都变，进而让 `workspace_projection` 变化、`ensure_loaded` 早退失效 ——
        // 变成"每次回调都整体重载"。
        {
            let mut meter = document.host_meter.lock_or_recover();
            if let Some(tempo) = document.clock.tempo() {
                if tempo.is_finite() && tempo > 0. && (meter.bpm - tempo).abs() > 1e-6 {
                    meter.bpm = tempo;
                }
            }
            if let Some((numerator, denominator)) = document.clock.time_signature() {
                let numerator = (numerator as u32).clamp(1, 32);
                let denominator = (denominator as u32).clamp(1, 32);
                // 与 `normalize_tempo_map` 同口径地夹取分母，不让宿主读数的越界值
                // 进入 Tempo Map。
                let denominator = if matches!(denominator, 1 | 2 | 4 | 8 | 16 | 32) {
                    denominator
                } else {
                    4
                };
                if meter.numerator != numerator || meter.denominator != denominator {
                    meter.numerator = numerator;
                    meter.denominator = denominator;
                }
            }
        }
        // 【T6】心跳末尾自检：卡住时必须留下一份可归因的读数。
        self.report_stuck_if_needed();
    }
    /// **T6：把"卡住了"写成一份可归因的读数。**
    ///
    /// 【为什么必须存在】这四类症状反复复发的真正原因不是某个 bug，而是**失败不可
    /// 归因**：十几条路径都能静默早退，而它们留下的现场几乎同形。这一行把每一条早退
    /// 都变成一个具名的数（verdict / region / 源 / 授权 / 发布 / 失败 / 会话 / 脏期），
    /// 下一轮不必再从位掩码与调用顺序反推。
    ///
    /// 【为什么在这里、且限流】`refresh_host` 是 actor 的心跳，与窗口开没开无关；
    /// 限流由 `emit_limited_key` 负责（同 key 只打一次），因此不会刷屏。
    fn report_stuck_if_needed(&self) {
        let Some(document) = self.document() else {
            return;
        };
        let Some(verdict) = document.input_verdict() else {
            return;
        };
        if verdict == crate::render::document::InputVerdict::Ready {
            return;
        }
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.stuck",
            format_args!("[ara] stuck: {}", document.stuck_report()),
        );
    }

    /// 波形入口只接受本会话从宿主PCM生成的路径，不能让JS任意读取本机文件。
    pub(super) fn check_source(&self, path: &str) -> Result<(), String> {
        if self
            .loaded
            .lock_or_recover()
            .reverse_paths
            .contains_key(path)
            || self
                .display_waveforms
                .lock_or_recover()
                .values()
                .any(|(_, clip)| clip.source_path.as_deref() == Some(path))
        {
            Ok(())
        } else {
            Err("source is not authorized by this ARA session".into())
        }
    }
    /// mute只撤销播放分配，不撤销已经生成的显示波形；take更换/真实删除不沿旧缓存猜。
    fn retain_display_waveforms(&self, timeline: &mut TimelineState, restore: bool) {
        let Some(document) = self.document() else {
            return;
        };
        let tracks = document.ui_tracks.lock_or_recover();
        let mut cache = self.display_waveforms.lock_or_recover();
        let actual = tracks
            .values()
            .flat_map(|track| &track.items)
            .map(|item| {
                (
                    format!("{}ara-item-{}", self.namespace, item.geometry.item_id),
                    item.geometry.take_id.clone(),
                )
            })
            .collect::<HashMap<_, _>>();
        cache.retain(|id, (take, _)| actual.get(id) == Some(take));
        for clip in &mut timeline.clips {
            let Some(take) = actual.get(&clip.id) else {
                continue;
            };
            if !restore && clip.source_path.is_some() {
                cache.insert(clip.id.clone(), (take.clone(), clip.clone()));
            }
            // 【T-C3：任何"这一轮丢了源"的 clip 都复用上一份显示波形，不再只限 mute】
            //
            // 【为什么】占位（"等待宿主提供音频"）与"波形稍旧"是两个量级完全不同的
            // 体验：前者让用户以为片段坏了，后者只是一次短暂的内容过期。缓存键含
            // take GUID（见上面的 `retain`），所以换了 take 绝不会复用旧波形。
            // 物化出来的文件是**内容哈希**命名的，旧路径指向的是上一版内容，不会错位。
            if restore && clip.source_path.is_none() {
                if let Some((_, old)) = cache.get(&clip.id) {
                    clip.source_path = old.source_path.clone();
                    clip.duration_sec = old.duration_sec;
                    clip.duration_frames = old.duration_frames;
                    clip.source_sample_rate = old.source_sample_rate;
                    clip.source_channels = old.source_channels;
                    clip.waveform_preview = old.waveform_preview.clone();
                    for current in &mut clip.takes {
                        current.source_path = old.source_path.clone();
                        current.duration_sec = old.duration_sec;
                        current.duration_frames = old.duration_frames;
                        current.source_sample_rate = old.source_sample_rate;
                        current.source_channels = old.source_channels;
                        current.waveform_preview = old.waveform_preview.clone();
                    }
                }
            }
        }
    }
    /// 分析快照：与 `session.timeline` 的**唯一**差别是只保留本实例认领的 clip。
    ///
    /// 【为什么不再强制打开 `compose_enabled`】此前这里对手绘曲线所在的根轨强制
    /// `compose_enabled = true`，以便"关掉 Compose 也产出原线"。但
    /// `build_root_pitch_key` 把 `compose_enabled` 计入缓存键 ⇒ 分析产出的是
    /// compose=true 的键，而会话时间线算出的是 compose=false 的键，两者不符 ⇒
    /// `pitch_orig_key` 被反复清空 ⇒ 渲染门禁永远 `pending`（用户报障：关掉合成后
    /// 左下角一直"渲染中"）。现在"手绘曲线也算需要原线"这条规则落在**内核门禁**
    /// （`pitch_analysis::analysis::assembly_gate_open`）里，两个视图用同一份事实、
    /// 算出同一个键，因此不再需要这个会造成键漂移的绕过。
    fn analysis_timeline(&self) -> TimelineState {
        let mut timeline = self.timeline.lock_or_recover().clone();
        if let Some(document) = self.document() {
            let ids = document.clip_ids.lock_or_recover();
            timeline.clips.retain(|clip| {
                ids.values()
                    .any(|id| format!("{}{id}", self.namespace) == clip.id)
            });
        }
        // 倒放 clip **重新纳入分析**：分析读的是**正向完整源**（缓存键含源路径 + 文件
        // 签名 + 帧周期，不含消费窗口），倒放只发生在**装配期** ——
        // `pitch_analysis/schedule.rs` 与 `pitch_editing.rs` 在把曲线写入时间线帧时
        // 整体翻转。所以让倒放进分析是安全的：它读的还是同一份正向缓存，只是取窗口时
        // 镜像。此前把它排除是因为倒放被隔离（没有源可分析）；现在照常物化
        // （`ReversePolicy::Materialize`），排除反而会让它的 f0 全零 ⇒ 只剩气声。
        timeline
    }
    /// 原分析实现可取消/join；每会话持有自己的worker，不推进全局generation。
    fn schedule_analysis(&self) {
        if self.closed.load(Ordering::Acquire) || !self.loaded.lock_or_recover().initialized {
            return;
        }
        let mut workers = self.analysis_workers.lock_or_recover();
        let mut pending = Vec::new();
        for worker in workers.drain(..) {
            if worker.is_finished() {
                let _ = worker.join();
            } else {
                pending.push(worker);
            }
        }
        pending.extend(
            hifishifter_kernel::pitch_clip::schedule_clip_pitch_jobs_scoped(
                &self.analysis_timeline(),
                &self.analysis_sender,
                self.analysis_cancel.clone(),
            ),
        );
        *workers = pending;
    }
    /// 是否有分析 worker 仍在跑。
    ///
    /// 【用途】把"分析在途（等就是了）"与"原线永远不会产出（再等也没用）"分开。
    /// 只看时间会把几分钟素材的**正常**慢分析误判成不收敛；只看这个布尔又会把
    /// "worker 刚结束、结果还在通道里"的瞬间误判成死局。两者结合才既不过早放弃、
    /// 也不无限空转。
    fn analysis_in_flight(&self) -> bool {
        self.analysis_workers
            .lock_or_recover()
            .iter()
            .any(|worker| !worker.is_finished())
    }
    /// 原设备worker负责的ClipPitchReady现在由本实例actor接收，不创建cpal设备。
    fn refresh_analysis(&self) -> bool {
        let updates = self
            .analysis_updates
            .lock_or_recover()
            .try_iter()
            .collect::<Vec<_>>();
        let mut roots = std::collections::BTreeSet::new();
        {
            let mut timeline = self.timeline.lock_or_recover();
            for update in updates {
                if let hifishifter_kernel::engine_command::EngineCommand::ClipPitchReady {
                    clip_id,
                } = update
                {
                    if let Some(clip) = timeline.clips.iter().find(|c| c.id == clip_id) {
                        if let Some(root) = timeline.resolve_root_track_id(&clip.track_id) {
                            roots.insert(root);
                        }
                    }
                }
            }
            for root in &roots {
                if let Some(params) = timeline.params_by_root_track.get_mut(root) {
                    params.pitch_orig_key = None;
                    params.dyn_orig_key = None;
                }
            }
        }
        if roots.is_empty() {
            return false;
        }
        self.complete_cached_analysis();
        // 保存冷恢复没有本地generation，原线完成仍须重应用，不能永久沿旧源F0快照。
        true
    }
    /// 本会话的宿主时钟：ARA 形态是文档的，Capture 形态是 owner 自己的。
    ///
    /// 【为什么必须有这一条】`transport()` 与 `playback_state()` 此前只读**文档**时钟，
    /// 而 Capture 形态没有文档 —— 于是两者恒定返回 `(0, false)`：插件 UI 的播放光标
    /// 钉在 0，且永远认为宿主没在播放。而宿主位置其实一直写着（`observe_transport` 在
    /// **每个音频块**上写 owner 的时钟），只是没有任何一条命令去读它。
    ///
    /// 【为什么 owner 的时钟在这里就一定有值】它由 `setupProcessing` 装载
    /// （见 `ExtensionOwner::arm_capture_rendering` 的调用点），而那是宿主在任何
    /// `process` 之前的确定性时刻 —— 与窗口开没开无关。
    fn host_clock(&self) -> Option<Arc<crate::render::transport::TransportClock>> {
        match self.document() {
            Some(document) => Some(document.clock.clone()),
            None => self.capture_owner()?.clock.get().cloned(),
        }
    }

    /// GUI读取真实宿主时钟，不驱动独立设备；未收到process上下文时保留本地查看游标。
    pub(super) fn transport(&self) -> (f64, bool) {
        self.host_clock()
            .map(|clock| clock.read())
            .unwrap_or_else(|| (self.timeline.lock_or_recover().playhead_sec, false))
    }
    /// 新淡化轴只装饰发给原GUI的JSON，不进入算法状态、撤销历史或音频缓存键。
    pub(super) fn decorate_host_fades(&self, payload: &mut Value) {
        if let Some(document) = self.document() {
            document.decorate_host_fades(payload, &self.namespace);
        }
    }
    /// 宿主音频读数：与 `decorate_host_fades` 同一路径，只多一个语言无关的分类字段。
    pub(super) fn decorate_host_audio(&self, payload: &mut Value) {
        if let Some(document) = self.document() {
            document.decorate_host_audio(payload);
        }
    }
    /// 只读宿主原子时钟；重合成阻塞actor时UI仍取得新鲜播放态，不借编辑timeline锁。
    pub(super) fn playback_state(&self) -> Value {
        let clock = self.host_clock();
        let (position, playing) = clock
            .as_ref()
            .map(|clock| clock.read())
            .unwrap_or((0., false));
        json!({"ok":true,"is_playing":playing,"waiting_for_render":false,
            "target":if playing {Some("synthesized")} else {None},"base_sec":0.,
            "position_sec":position,"duration_sec":f64::from_bits(self.project_duration.load(Ordering::Acquire)),
            // 【为什么如实报而不是恒真】这个字段的语义是"上面那个位置来自宿主"。
            // Capture 形态下时钟来自 owner（同样是宿主写入的），因此也为真；
            // 两条都没有时它是假 —— 前端据此知道那个 0 不是"宿主在 0"。
            "host_authoritative":clock.is_some()})
    }
    /// 一次性时钟探针最多每秒写一条；只在命令actor线程调用，不在音频callback写文件。
    pub(super) fn report_transport_probe(&self) {
        if !self.transport_probe {
            return;
        }
        let now = Instant::now();
        let mut next = self.transport_probe_next.lock_or_recover();
        if now < *next {
            return;
        }
        *next = now + Duration::from_secs(1);
        drop(next);
        if let Some(clock) = self.document().map(|document| document.clock.clone()) {
            log::info!("[ara] transport diagnostic {}", clock.diagnostics());
        }
    }
    fn rewrite_ids(&self, timeline: &mut TimelineState, to_ui: bool) {
        let id = |value: &str| {
            if to_ui {
                format!("{}{value}", self.namespace)
            } else {
                value
                    .strip_prefix(&self.namespace)
                    .unwrap_or(value)
                    .to_owned()
            }
        };
        for track in &mut timeline.tracks {
            track.id = id(&track.id);
            track.parent_id = track.parent_id.as_deref().map(id);
        }
        for clip in &mut timeline.clips {
            clip.id = id(&clip.id);
            clip.track_id = id(&clip.track_id);
        }
        timeline.selected_track_id = timeline.selected_track_id.as_deref().map(id);
        timeline.selected_clip_id = timeline.selected_clip_id.as_deref().map(id);
        timeline.params_by_root_track = std::mem::take(&mut timeline.params_by_root_track)
            .into_iter()
            .map(|(key, value)| (id(&key), value))
            .collect();
    }

    /// 把**宿主事实**（方向 / 循环源 / 声道模式 / 正向源窗口）投影到编辑器时间线。
    ///
    /// 【为什么必须单独一步，且必须在 `present_host_inventory` 之后】
    /// `present_host_inventory` 写的是宿主 `D_STARTOFFS` 的**原值**（倒放时是镜像
    /// 坐标），而 `project_host_take_facts_locked` 是全仓**唯一**的镜像复原点。
    /// 此前编辑器路径只做了前者、漏了后者 —— 于是界面里的倒放窗口是镜像的、声音
    /// 却是对的（渲染路径做了复原）。用户看到的正是"倒放 Offset 逻辑不对"。
    ///
    /// 【为什么不做成"顺手在 present_host_inventory 里加一句"】那条函数还会被
    /// 别的调用点使用（增量刷新等），把两条职责焊死会让"只想刷清单"的调用方
    /// 意外触发一次事实投影。分开命名、由调用方显式成对调用，边界才清楚。
    fn project_host_facts(&self, timeline: &mut TimelineState) {
        if let Some(document) = self.document() {
            document.project_host_take_facts_locked(timeline);
        }
    }
    /// 只将已授权私有分析路径还原到ARA身份，未知路径明确失败。
    fn native_timeline(&self, mut timeline: TimelineState) -> Result<TimelineState, String> {
        self.rewrite_ids(&mut timeline, false);
        if let Some(document) = self.document() {
            if !document.ui_tracks.lock_or_recover().is_empty() {
                // GUI占位、名字/排序及可见总时长不进入音频权威；只取参数/轨道可编辑值。
                let mut host = document.workspace_timeline()?;
                if !document.edits.lock_or_recover().groups.is_empty() {
                    let ids = host
                        .tracks
                        .iter()
                        .map(|track| track.id.clone())
                        .collect::<std::collections::BTreeSet<_>>();
                    let (mut flat, grouped) = document
                        .private_parameter_views(timeline.selected_clip_id.clone(), None)?;
                    super::private_groups::expand_parameter_changes(
                        &timeline, &grouped, &mut flat,
                    )?;
                    flat.tracks.retain(|track| ids.contains(&track.id));
                    flat.params_by_root_track.retain(|id, _| ids.contains(id));
                    flat.selected_track_id =
                        timeline.selected_track_id.filter(|id| ids.contains(id));
                    return Ok(flat);
                }
                for track in &mut host.tracks {
                    if let Some(client) = timeline.tracks.iter().find(|t| t.id == track.id) {
                        track.volume = client.volume;
                        track.muted = client.muted;
                        track.solo = client.solo;
                        track.compose_enabled = client.compose_enabled;
                        track.pitch_analysis_algo = client.pitch_analysis_algo.clone();
                    }
                }
                host.params_by_root_track = timeline
                    .params_by_root_track
                    .into_iter()
                    .filter(|(id, _)| host.tracks.iter().any(|t| &t.id == id))
                    .collect();
                host.selected_track_id = timeline
                    .selected_track_id
                    .filter(|id| host.tracks.iter().any(|t| &t.id == id));
                host.selected_clip_id = timeline
                    .selected_clip_id
                    .filter(|id| host.clips.iter().any(|clip| &clip.id == id));
                return Ok(host);
            }
        }
        // 宿主显示占位没有ARA音频授权，只提交真实活动图，不能把占位送入分析/合成。
        if let Some(document) = self.document() {
            let ids = document.clip_ids.lock_or_recover();
            timeline
                .clips
                .retain(|clip| ids.values().any(|id| id == &clip.id));
            drop(ids);
            let host = document.timeline.lock_or_recover();
            if let Some(host) = host.as_ref() {
                timeline
                    .tracks
                    .retain(|t| host.tracks.iter().any(|h| h.id == t.id));
                timeline
                    .params_by_root_track
                    .retain(|id, _| host.tracks.iter().any(|t| &t.id == id));
            }
        }
        let loaded = self.loaded.lock_or_recover();
        for clip in &mut timeline.clips {
            // 原GUI普通宿主fade仅装饰，参数提交还原到ARA内核域，避免二次淡化。
            clip.fade_in_sec = 0.;
            clip.fade_out_sec = 0.;
            clip.auto_fade_in_sec = 0.;
            clip.auto_fade_out_sec = 0.;
            clip.fade_in_shape = 0.;
            clip.fade_out_shape = 0.;
            clip.fade_in_dir = 0.;
            clip.fade_out_dir = 0.;
            // 宿主item音量只在GUI显示，提交参数前还原原始ARA源域，不能重复烘焙。
            clip.gain = 1.;
            for take in &mut clip.takes {
                take.gain = 1.;
            }
            for path in std::iter::once(&mut clip.source_path)
                .chain(clip.takes.iter_mut().map(|t| &mut t.source_path))
            {
                let local = path.as_ref().ok_or("clip analysis source missing")?;
                *path = Some(
                    loaded
                        .reverse_paths
                        .get(local)
                        .ok_or("unknown analysis path")?
                        .clone(),
                );
            }
            clip.normalize_takes();
        }
        Ok(timeline)
    }
    /// 选region后沿当前真实doc取只读投影，防止重叠区仍显示前一素材的曲线。
    pub(super) fn select_source_projection(&self) -> Result<(), String> {
        if self
            .error
            .lock_or_recover()
            .as_ref()
            .is_some_and(|error| error.starts_with("Conflict"))
        {
            return Ok(());
        }
        // Capture：没有"宿主的 region 参数"可投影 —— 时间线里的曲线本来就是权威，
        // 不存在"选中的那条素材另有一份曲线"这件事。
        let Some(document) = self.document() else {
            return Ok(());
        };
        let current = self.native_timeline(self.timeline.lock_or_recover().clone())?;
        let roots = document.selected_source_parameters(&current)?;
        let mut timeline = self.timeline.lock_or_recover();
        timeline.params_by_root_track.extend(
            roots
                .into_iter()
                .map(|(root, params)| (format!("{}{root}", self.namespace), params)),
        );
        Ok(())
    }
    pub(super) fn emit(&self, event: &str, payload: Value) {
        let document = self.document();
        let views = {
            let mut views = self.views.lock_or_recover();
            views.retain(|_, view| {
                !view.sink.closed.load(Ordering::Acquire)
                    && view.route.as_ref().is_none_or(|(link, lease)| {
                        document.as_ref().is_some_and(|document| {
                            link.authorize(document)
                                .is_ok_and(|current| current == *lease)
                        })
                    })
            });
            views
                .values()
                .map(|view| view.sink.clone())
                .collect::<Vec<_>>()
        };
        for sink in views {
            sink.event(json!({"version":1,"viewId":sink.view_id,"event":event,"payload":payload}));
        }
    }
    /// 复用原GUI已订阅的刷新事件；共享选轨/曲线变化让所有视图重取同一个权威payload。
    pub(super) fn notify_timeline(&self) {
        self.host_version.fetch_add(1, Ordering::AcqRel);
        self.emit(
            "plugin_host_changed",
            json!({"version":self.host_version.load(Ordering::Acquire)}),
        );
    }
    pub(super) fn state(&self) -> Value {
        let generation = self.generation.load(Ordering::Acquire);
        let applied = self.applied.load(Ordering::Acquire);
        let states = self
            .document()
            .map(|document| {
                document
                    .renderer_owners()
                    .into_iter()
                    .map(|owner| owner.preparation_state())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let host_pending = states.iter().any(|state| state.0);
        let host_error = states.into_iter().find_map(|state| state.1);
        let analysis_pending = self.requires_analysis(&self.timeline.lock_or_recover());
        let rendering = self.render_task.lock_or_recover().is_some();
        // 【为什么不是"有片段缺 source_path 就算在途"】`unavailable` 的三类成因都是终态
        // （folder 父轨 / 样本读取失败 / 超期未认领）。把它们也算作"进行中"会让应用状态
        // 永久停在 pending —— 用户对着一个永远转不完的转圈。口径交给文档侧的唯一判据
        // （`media_in_flight_items`，与 `decorate_host_media_locked` 同一套分类）。
        let inventory_pending = {
            let timeline = self.timeline.lock_or_recover();
            self.document()
                .is_some_and(|document| document.media_in_flight_items(&timeline) > 0)
        };
        json!({
            // 【为什么把形态放进状态快照】bootstrap 是**窗口创建那一刻**算的，而形态判定
            // 要等 `setActive` 之后 5 秒宽限期才落定 —— 于是窗口打开时读到的一律是
            // `ara_only`，前端按形态分流的地方（录制面板的自动浮出）拿不到真实答案。
            // 状态快照每 250ms 轮询一次，是现成的、不依赖时序的通道。
            "mode":if self.is_capture(){"capture"}else{"ara"},
            "generation":generation,"applied_generation":applied,"pending":generation!=applied||host_pending||inventory_pending||analysis_pending||rendering||self.render_requested.load(Ordering::Acquire),
            // 【为什么把"在等分析"单独报出来】它是**唯一会持续很久**的一步（几分钟素材的
            // 神经分析本来就慢），而它此前被 `pending` 一句话盖住 —— 用户无法区分
            // "卡住了"与"在正常跑"。前端据此把状态栏片分成"在分析 / 在合成 / 在应用"。
            "analysis_pending":analysis_pending,
            // `rendering_active` 是**派生状态**，前端据此点亮/熄灭"渲染中"片，而不是
            // 只靠 `playback_rendering_state` 事件流。事件用 `try_send`（队列满即丢），
            // 丢一个 `active:false` 就会让片永久亮着；快照每 250 ms 重读一次，自愈。
            "rendering_active":rendering,
            "host_version":self.host_version.load(Ordering::Acquire),
            "error":self.error.lock_or_recover().clone().or_else(||self.render_error.lock_or_recover().clone()).or(host_error),"connected":!self.closed.load(Ordering::Acquire),
            "ready":self.loaded.lock_or_recover().initialized})
    }
    fn emit_state(&self) {
        self.emit("plugin_apply_state", self.state());
    }
}
impl ParamHost for EditorSession {
    fn timeline(&self) -> &Mutex<TimelineState> {
        &self.timeline
    }
    fn checkpoint_timeline(&self, timeline: &TimelineState, operation: HistoryOp) {
        if self.suppress_history.load(Ordering::Acquire) {
            return;
        }
        history::checkpoint(
            &mut self.history.lock_or_recover(),
            timeline,
            operation.key().into(),
            || None,
        );
    }
    fn mark_dirty(&self) {
        self.revision_bumps_dirty.fetch_add(1, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::AcqRel);
        // 见 `timeline_revision` 的字段说明：它是"编辑 ∪ 发布"的并集。
        self.timeline_revision.fetch_add(1, Ordering::AcqRel);
        let views = self.views.lock_or_recover();
        for view in views.values() {
            if let Some((link, _)) = &view.route {
                link.mark_dirty();
            }
        }
    }
    fn publish_timeline(&self, timeline: TimelineState) {
        // 【为什么在分支之前】这是"时间线变了"的**唯一**统一信号：所有曲线写入
        // （set_param_frames / restore / convert_mix_param / set_static_param /
        // stretch_track_linked_params）与撤销重做都会走到这里。Capture 的渲染版本闸
        // 以它为据，因此不标脏的改动也不会被跳过（见字段说明）。
        self.revision_bumps_publish.fetch_add(1, Ordering::Relaxed);
        self.timeline_revision.fetch_add(1, Ordering::AcqRel);
        self.clip_count
            .store(timeline.clips.len() as u64, Ordering::Release);
        // Capture：没有宿主可写回 —— 参数就是权威，落在本会话的时间线上。
        // 只需要排一次重渲染（owner 的 Capture 渲染准备会读这份时间线）。
        if self.is_capture() {
            self.render_requested.store(true, Ordering::Release);
            return;
        }
        let result = (|| {
            let document = self.require_document()?;
            let groups = document.edits.lock_or_recover().groups.clone();
            let groups =
                if groups.is_empty() {
                    None
                } else {
                    Some(groups.with_processing_from(
                        &timeline,
                        &document.group_aliases(&self.namespace),
                    )?)
                };
            let timeline = self.native_timeline(timeline)?;
            let (edit, model, projection) = {
                let loaded = self.loaded.lock_or_recover();
                (loaded.edit, loaded.model, loaded.projection.clone())
            };
            let versions = document.accept_workspace_edits_with_groups(
                edit,
                model,
                &timeline,
                &projection,
                groups,
            )?;
            let mut loaded = self.loaded.lock_or_recover();
            loaded.edit = versions.0;
            loaded.model = versions.1;
            loaded.projection = versions.2;
            *self.error.lock_or_recover() = None;
            Ok::<_, String>(())
        })();
        if let Err(error) = result {
            *self.error.lock_or_recover() = Some(error);
        }
    }
}

/// MIDI 导入在插件里与 App 共用同一份编排（`kernel::editor::midi_import`）：写的是
/// 插件自己的音高曲线，不碰宿主几何。唯一差异是剪贴板载荷缓存归本会话持有。
impl MidiImportHost for EditorSession {
    fn clipboard_midi(&self) -> &Mutex<std::collections::VecDeque<(String, Vec<u8>)>> {
        &self.clipboard_midi
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ara::model::ModelHandle;
    use crate::render::extension::ExtensionOwner;
    use crate::render::ownership::region_owners;
    use crate::render::source::SourcePcm;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::ExtensionRoles;

    /// **预算耗尽只是退避，不是永久停止自动渲染。**
    ///
    /// 【为什么必须钉住】此前 `streak >= RENDER_RETRY_BUDGET` 即 `settled = true` ——
    /// 自动渲染**再也不会**启动，直到用户下一次操作。而 8 次重试的间隔是 150 ms：
    /// 它衡量的是"这 1.2 秒里宿主一直在变"，不是"这件事永远不会成功"。实测 Cubase 15
    /// 上一次 `unsupported host position` 就是 8 次全失败后把界面**永久**冻结在旧版本
    /// （日志 22:22:35 `gave up after 8 attempts`），而那个错误本身是瞬时的。
    #[test]
    fn exhausting_the_retry_budget_backs_off_instead_of_stopping() {
        for streak in 1..RENDER_RETRY_BUDGET {
            let RenderAction::Retry(delay) = render_retry_action(streak);
            assert_eq!(
                delay, RENDER_RETRY_INTERVAL,
                "预算内必须是快速重试（streak={streak}）"
            );
        }
        for streak in [RENDER_RETRY_BUDGET, RENDER_RETRY_BUDGET + 1, 1_000] {
            let RenderAction::Retry(delay) = render_retry_action(streak);
            assert_eq!(
                delay, RENDER_RETRY_BACKOFF,
                "预算耗尽必须退避而不是停止（streak={streak}）"
            );
        }
        assert!(
            RENDER_RETRY_BACKOFF > RENDER_RETRY_INTERVAL,
            "退避间隔必须真的更疏，否则这条改动等于没做"
        );
    }

    /// 参数专用Undo/Redo确实走actor历史，而不是宿主几何；同一分组中的多块曲线只撤一次。
    #[test]
    fn parameter_history_commands_restore_grouped_curve_and_redo_without_geometry_changes() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(64);
        let sink = UiSink {
            view_id: "parameter-history".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let before = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let track = before["tracks"][0]["id"].clone();
        call(
            &editor,
            &sink,
            &rx,
            2,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        call(
            &editor,
            &sink,
            &rx,
            3,
            "begin_undo_group",
            json!({"label":"curve"}),
        );
        call(
            &editor,
            &sink,
            &rx,
            4,
            "set_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"values":[35.,35.],"checkpoint":false}),
        );
        call(
            &editor,
            &sink,
            &rx,
            5,
            "set_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":2,"values":[45.,45.],"checkpoint":false}),
        );
        call(&editor, &sink, &rx, 6, "end_undo_group", json!({}));
        let undo = call(&editor, &sink, &rx, 7, "undo_parameter_edit", json!({}));
        assert_eq!(
            undo["clips"][0]["start_sec"],
            before["clips"][0]["start_sec"]
        );
        let cleared = call(
            &editor,
            &sink,
            &rx,
            8,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        assert!(cleared["edit"]
            .as_array()
            .unwrap()
            .iter()
            .all(|value| value.as_f64() == Some(0.)));
        call(&editor, &sink, &rx, 9, "redo_parameter_edit", json!({}));
        let restored = call(
            &editor,
            &sink,
            &rx,
            10,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        assert_eq!(restored["edit"], json!([35., 35., 45., 45.]));
        document.close();
    }

    struct ReleaseDsp(Option<mpsc::Sender<()>>);
    impl Drop for ReleaseDsp {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }
    /// 实际actor在DSP线程受控阻塞时仍处理读参、写参和getState屏障；不是神经模型速度测试。
    #[test]
    fn async_dsp_does_not_block_parameter_commands_or_state_barriers() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let track = editor.timeline.lock_or_recover().tracks[0].id.clone();
        let (release, wait) = mpsc::channel();
        let mut release = ReleaseDsp(Some(release));
        let (started, ready) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = RenderTask::spawn(
            editor.submitted.load(Ordering::Acquire),
            editor.generation.load(Ordering::Acquire),
            cancel.clone(),
            move || {
                started.send(()).unwrap();
                wait.recv_timeout(Duration::from_secs(3))
                    .map_err(|error| error.to_string())?;
                Ok(())
            },
        )
        .unwrap();
        ready.recv_timeout(Duration::from_secs(1)).unwrap();
        {
            let mut active = editor.render_task.lock_or_recover();
            *active = Some(job);
            document.host_undo.pending.store(false, Ordering::Release);
        }
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(64);
        let sink = UiSink {
            view_id: "async-dsp-diagnostic".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let started = Instant::now();
        let frames = call(
            &editor,
            &sink,
            &rx,
            1,
            "get_param_frames",
            json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"frameCount":4,"binary":false}),
        );
        assert_eq!(frames["ok"], true);
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":track,"volume":0.5}),
        );
        editor.flush().unwrap();
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "编辑命令不应等待受控DSP任务: {:?}",
            started.elapsed()
        );
        assert!(
            cancel.load(Ordering::Acquire),
            "成功入队的新编辑必须取消旧DSP任务"
        );
        release.0.take().unwrap().send(()).unwrap();
        document.close();
        assert!(
            editor.render_task.lock_or_recover().is_none(),
            "关闭完成前必须回收DSP任务"
        );
    }

    /// 完成结果与计算线程独立，取消位始终保留，不能把被新编辑取代的成功计算标成已应用。
    #[test]
    fn async_dsp_completion_retains_ticket_generation_and_cancellation() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let job = RenderTask::spawn(41, 17, cancel, || Ok(())).unwrap();
        {
            let mut active = editor.render_task.lock_or_recover();
            assert!(active.is_none());
            *active = Some(job);
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some((generation, ticket, cancelled, result)) = editor.complete_render() {
                assert_eq!((generation, ticket), (17, 41));
                assert!(cancelled);
                assert!(result.is_ok());
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        document.close();
    }

    /// 宿主音量只影响GUI；参数提交不能把item增益再交给内核烘焙。
    #[test]
    fn host_clip_gain_display_is_removed_before_parameter_commit() {
        let (_model, owner, _id) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let mut display = editor.timeline.lock_or_recover().clone();
        display.clips[0].gain = 0.5;
        display.clips[0].takes[0].gain = 0.5;
        let native = editor.native_timeline(display).unwrap();
        assert_eq!(native.clips[0].gain, 1.);
        assert_eq!(native.clips[0].takes[0].gain, 1.);
        editor.close();
    }

    /// **从未绑定过 ARA 的实例必须能拿到编辑器会话 —— 哪怕形态还没落定。**
    ///
    /// 【为什么这条是那个 bug 的回归】形态要等 `setActive` 之后 `ARA_BINDING_GRACE`
    /// （5 秒）才落定，而**窗口打开的那一刻必然早于它**。此前 `editor_session()` 要求
    /// `mode() == Capture`，于是探活拿到 `ARA_NOT_BOUND`（终态）⇒ 整页替换成
    /// "插件会话不可用"，且不可恢复 —— 非 ARA 宿主永远打不开界面。
    #[test]
    fn an_instance_that_never_bound_ara_is_served_by_the_capture_session() {
        let owner = Arc::new(ExtensionOwner::default());
        assert_eq!(
            owner.mode(),
            crate::render::extension::InstanceMode::Undetermined,
            "没有 setActive 也没有绑定 ⇒ 形态未定（这正是窗口刚打开时的样子）"
        );
        let first = owner
            .editor_session()
            .expect("未绑定的实例必须有可用的编辑器会话");
        assert!(first.is_capture());
        assert!(
            Arc::ptr_eq(&first, &owner.editor_session().unwrap()),
            "同一实例必须是同一份会话"
        );
        first.close();
    }

    /// **Capture 形态的宿主时钟来自 owner，不是文档。**
    ///
    /// 【为什么这是回归测试】`transport()` / `playback_state()` 此前只读**文档**时钟，
    /// 而 Capture 形态没有文档 —— 于是两者恒定返回 `(0, false)`：插件 UI 的播放光标
    /// 钉在 0，且永远认为宿主没在播放。而宿主位置其实一直在写（`observe_transport` 在
    /// **每个音频块**上写 owner 的时钟），只是没有任何一条命令去读它。
    #[test]
    fn a_capture_session_reads_the_host_clock_instead_of_a_document() {
        let owner = Arc::new(ExtensionOwner::default());
        owner.set_setup_sample_rate(48_000.0);
        owner.note_binding_window_closed_without_ara();
        // 时钟的装载点与 `setupProcessing` 里调的是同一个函数。
        owner.arm_capture_rendering();
        let session = owner.capture_session().unwrap();
        assert!(session.is_capture());
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_eq!(session.transport(), (0.0, false), "还没收到宿主上下文");
            // 时钟**已经装载**（`setupProcessing` 会做这件事），只是还没收到任何上下文：
            // 位置是 0、没在播放，而"位置来自宿主"这件事为真 —— 与 ARA 形态同一口径
            // （那里也恒为真）。前端据此知道那个 0 是"宿主在 0"而不是"读不到"。
            assert_eq!(session.playback_state()["host_authoritative"], true);
            // 宿主在播放，位置 1.5 s。
            owner.observe_transport(
                &crate::audio_abi::ProcessContext {
                    state: (1 << 1) | (1 << 10),
                    sample_rate: 48_000.0,
                    project_time_samples: 72_000,
                    ..Default::default()
                },
                0,
            );
            let (position, playing) = session.transport();
            assert!(playing, "宿主在播放必须如实报出");
            assert!(
                (position - 1.5).abs() < 1e-9,
                "位置必须来自宿主时钟，实际 {position}"
            );
            let state = session.playback_state();
            assert_eq!(state["is_playing"], true);
            assert_eq!(state["host_authoritative"], true);
            assert!((state["position_sec"].as_f64().unwrap() - 1.5).abs() < 1e-9);
            // 宿主停止 ⇒ 如实回落（面板据此说"宿主没在播放，录不到内容"）。
            owner.observe_transport(
                &crate::audio_abi::ProcessContext {
                    state: 0,
                    sample_rate: 48_000.0,
                    project_time_samples: 72_000,
                    ..Default::default()
                },
                0,
            );
            assert!(!session.transport().1, "停止播放必须如实回落");
        }));
        session.close();
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    /// 建一个**真的录过一段**的 Capture 实例：没有 ARA 文档，素材来自输入总线。
    fn capture_owner_with_a_take() -> Arc<ExtensionOwner> {
        let owner = Arc::new(ExtensionOwner::default());
        owner.set_setup_sample_rate(48_000.0);
        // 绑定窗口关闭且没有绑定 ⇒ 形态是 Capture（宿主没给 ARA）。
        owner.note_binding_window_closed_without_ara();
        assert_eq!(
            owner.mode(),
            crate::render::extension::InstanceMode::Capture,
            "这个夹具的全部意义就是宿主没给 ARA"
        );
        let capture = owner.capture();
        capture.note_project_rate(48_000.0);
        capture.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 4096].into_boxed_slice());
        for index in 0..4_i64 {
            capture.on_block(crate::capture::InputBlock {
                start: index * 4096,
                frames: 4096,
                sample_rate: 48_000.0,
                planes: [Some(samples), Some(samples)],
                playing: true,
            });
        }
        capture.stop().unwrap();
        owner
    }

    /// **Capture 形态没有 ARA 文档，但界面必须能用。**
    ///
    /// 【为什么这条必须有】`EditorSession` 此前只能从 `DocumentSession` 创建，于是非 ARA
    /// 宿主里每一条会话级命令都返回 `ARA_NOT_BOUND` —— 界面整页报"会话不可用"，用户既录
    /// 不了也改不了音高。这条钉住四件事：会话存在、时间线来自捕获存储、用户曲线在重投影
    /// 之后仍然在、渲染用的时间线回到源 id 域。
    #[test]
    fn a_capture_session_edits_without_any_ara_document() {
        let owner = capture_owner_with_a_take();
        let session = owner.editor_session().expect("Capture 形态必须有会话");
        // 【为什么要包 catch_unwind】断言失败会 unwind，而会话的 worker 线程只在
        // `close()` 里 join —— 不兜住就会留下一个活着的线程与进程，整套测试挂住。
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            capture_session_body(&session);
        }));
        session.close();
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    /// Capture 的渲染输入：带**用户曲线**的时间线 + 捕获存储的 PCM，且与导出共用。
    ///
    /// 【为什么这条必须存在】播放（`capture_prepare_job`）与导出
    /// （`export_audio_to_file`）都取这一份。若其中一条退回"只有几何的捕获存储时间线"，
    /// 用户就会听到处理后的声音、导出未处理的（或反过来）—— 而那是最难排查的一类不一致。
    #[test]
    fn capture_render_input_carries_the_user_curves_for_both_playback_and_export() {
        let owner = capture_owner_with_a_take();
        let session = owner.editor_session().unwrap();
        session.ensure_loaded(true).unwrap();
        let (take, edit, with_curves, input) = owner.capture_render_input().unwrap();
        assert!(take > 0, "渲染输入必须绑定到活动 take");
        assert!(
            with_curves,
            "会话已加载 ⇒ 这份时间线必须带着用户曲线（否则画完线不会生效）"
        );
        // 【为什么比的是"发布代次"】它是渲染版本闸的据：任何改了时间线并发布的路径
        // 都会推进它，包括不标脏的那些（见 `timeline_revision` 的字段说明）。
        assert_eq!(edit, session.timeline_revision());
        let timeline = input.timeline.as_ref().unwrap();
        assert_eq!(timeline.clips.len(), 1);
        assert!(
            timeline.clips[0]
                .source_path
                .as_deref()
                .unwrap()
                .starts_with("capture://"),
            "渲染时间线必须在源 id 域（`RenderInput.sources` 按 id 索引）"
        );
        assert!(!input.sources.is_empty(), "素材必须随输入一起给出");
        assert!(!input.regions.is_empty(), "区域必须由同一条时间线派生");
        // 用户改一次参数 ⇒ 编辑代次前进 ⇒ 渲染版本变化。不变的话"改了音高却不重算"
        // 就是一个静默的失效：界面变了，声音还是旧的。
        let before = edit;
        session.mark_dirty();
        assert_ne!(
            owner.capture_render_input().unwrap().1,
            before,
            "参数写入必须推进渲染版本"
        );
        session.close();
    }

    /// **诊断**：素材长度 → "画一条线"到"输出是处理结果"要多久。
    ///
    /// 【为什么值得留着】它回答的是 `ANALYSIS_PENDING_CEILING` 那个 60 秒到底合不合理：
    /// 上限太低会把**正常**的慢分析判成卡死，太高则让真卡死多冻用户一分钟。本机实测
    /// （48 kHz，含首次加载 ONNX 会话）：2 秒素材 ≈ 9.9 s、10 秒 ≈ 5.3 s、30 秒 ≈ 5.7 s
    /// —— 时间**几乎全在一次性模型加载**上，不随素材长度增长。所以 60 秒对常规素材有
    /// 两个数量级的余量，而真正需要调的是"卡死"那一侧（现在到顶只放弃**等待**，
    /// 不再放弃渲染）。
    ///
    /// 默认忽略：它要跑一次真分析 + 真合成（数十秒），不属于每次都要跑的回归。
    #[test]
    #[ignore]
    fn measure_convergence_against_material_length() {
        for seconds in [2_u32, 10, 30] {
            let owner = Arc::new(ExtensionOwner::default());
            owner.set_setup_sample_rate(48_000.0);
            owner.note_binding_window_closed_without_ara();
            let capture = owner.capture();
            capture.note_project_rate(48_000.0);
            capture.start().unwrap();
            let frames = 48_000_usize;
            let samples: &'static [f32] = Box::leak(vec![0.4_f32; frames].into_boxed_slice());
            for index in 0..i64::from(seconds) {
                capture.on_block(crate::capture::InputBlock {
                    start: index * frames as i64,
                    frames,
                    sample_rate: 48_000.0,
                    planes: [Some(samples), Some(samples)],
                    playing: true,
                });
            }
            capture.stop().unwrap();
            let session = owner.editor_session().unwrap();
            session.ensure_loaded(true).unwrap();
            let root = session
                .timeline
                .lock_or_recover()
                .params_by_root_track
                .keys()
                .next()
                .unwrap()
                .clone();
            let (reply, reply_rx) = mpsc::channel();
            let (events, _events_rx) = mpsc::sync_channel(4096);
            session
                .enqueue(UiRequest {
                    id: 1,
                    command: "set_param_frames".into(),
                    args: json!({
                        "trackId": root,
                        "param": "pitch",
                        "startFrame": 0,
                        "values": vec![7.0_f32; 16],
                        "checkpoint": true,
                    }),
                    sink: UiSink {
                        view_id: "measure".into(),
                        reply,
                        events,
                        closed: Arc::new(AtomicBool::new(false)),
                    },
                    link: None,
                })
                .unwrap();
            let _ = reply_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            let started = Instant::now();
            let deadline = Instant::now() + Duration::from_secs(600);
            while Instant::now() < deadline && session.state()["pending"] == true {
                std::thread::sleep(Duration::from_millis(100));
            }
            println!(
                "MEASURE seconds={seconds} pending={} elapsed={:?}",
                session.state()["pending"],
                started.elapsed()
            );
            session.close();
        }
    }

    /// **画完参数线之后，播放必须真的听到处理结果 —— 由 actor 自己收敛。**
    ///
    /// 【为什么这条必须存在】前面两条只钉到"渲染输入里带着曲线"，而用户的判据是
    /// **听到**。中间还有三步会各自静默失效：分析门禁（`start_capture_render` 早退）、
    /// 版本闸（跳过这次合成）、输出仲裁（发布出来的不是处理结果）。
    ///
    /// 【为什么必须让**真 actor** 跑】自动应用的重试与放弃全部发生在 actor 循环里
    /// （`RetryState::classify` 的 2 秒判定、60 秒上限、`settled`）。手工一步步调
    /// `start_capture_render` 会把那一整套绕过去 —— 而用户报障的正是"卡在正在自动
    /// 应用"：那恰恰是 `settled` 之后的状态。所以这里只投一条真实命令，剩下交给 actor。
    #[test]
    fn a_drawn_curve_reaches_the_processed_output() {
        let owner = capture_owner_with_a_take();
        let session = owner.editor_session().unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            session.ensure_loaded(true).unwrap();
            let root = session
                .timeline
                .lock_or_recover()
                .params_by_root_track
                .keys()
                .next()
                .unwrap()
                .clone();
            assert!(owner.capture().has_material(), "夹具必须真的录到了素材");
            // 与前端完全同一条路：排队 → actor 分发 → `params::set_param_frames`。
            // 这就是"用户画了一条音高线"，而 actor 会因此自己排一次自动渲染。
            let (reply, reply_rx) = mpsc::channel();
            let (events, _events_rx) = mpsc::sync_channel(4096);
            session
                .enqueue(UiRequest {
                    id: 1,
                    command: "set_param_frames".into(),
                    args: json!({
                        "trackId": root,
                        "param": "pitch",
                        "startFrame": 0,
                        "values": vec![7.0_f32; 16],
                        "checkpoint": true,
                    }),
                    sink: UiSink {
                        view_id: "curve-test".into(),
                        reply,
                        events,
                        closed: Arc::new(AtomicBool::new(false)),
                    },
                    link: None,
                })
                .unwrap();
            let reply = reply_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("画线命令必须有回信");
            assert_eq!(reply["ok"], true, "画一条音高线必须成功：{reply}");
            // 现在**什么都不做**，只等 actor 自己收敛。
            let deadline = Instant::now() + Duration::from_secs(180);
            while Instant::now() < deadline
                && (session.state()["pending"] == true
                    || owner.capture_output()
                        != crate::render::extension::CaptureOutput::Processed)
            {
                std::thread::sleep(Duration::from_millis(50));
            }
            assert_eq!(
                owner.capture_output(),
                crate::render::extension::CaptureOutput::Processed,
                "画完参数线之后必须听到处理结果，而不是直通/原始磁带"
            );
            assert_eq!(
                session.state()["pending"],
                false,
                "状态栏必须收敛到「已应用」，而不是永远停在「正在自动应用」：{}",
                session.state()
            );
        }));
        session.close();
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    /// **只发布、不标脏的曲线改动也必须推进渲染版本。**
    ///
    /// 【为什么这条必须存在】`convert_mix_param`（前端"换算"菜单）把音量曲线换算成
    /// dyn 曲线并 `publish_timeline`，但**不调 `mark_dirty`**。版本闸若只认编辑代次，
    /// 那种改动不会推进版本 ⇒ 渲染被跳过 ⇒ 换算完的曲线永远不生效 —— 用户看到的就是
    /// "画完参数线不生效"。这里按同一形状喂（只发布），把这条钉住。
    #[test]
    fn a_published_timeline_change_advances_the_render_version_without_marking_dirty() {
        let owner = capture_owner_with_a_take();
        let session = owner.editor_session().unwrap();
        session.ensure_loaded(true).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let before = owner.capture_render_input().unwrap().1;
            // 模拟 `convert_mix_param` 的收尾：发布，但不标脏。
            let snapshot = session.timeline.lock_or_recover().clone();
            session.publish_timeline(snapshot);
            assert_ne!(
                owner.capture_render_input().unwrap().1,
                before,
                "发布过的时间线必须推进渲染版本，否则这类改动会被版本闸静默跳过"
            );
        }));
        session.close();
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    /// **重载不得占用"用户编辑"的票据。**
    ///
    /// 【为什么这是"永远是直通"的回归测试】`request_reload` 此前自增 `submitted`，
    /// 而 `processed` 只在 `Job::Request` 到达且 `mutates_audio` 为真时兑付
    /// （`capture_*` 不在那个名单里，它也不产生 `Job::Request`）。于是
    /// `processed < submitted` 恒真 ⇒ actor 每 1ms 空转一次、`start_render` 永远
    /// 走不到 ⇒ 一次合成都排不出去 ⇒ 用户听到的永远是直通。
    #[test]
    fn a_reload_does_not_consume_the_edit_ticket() {
        // 【为什么用没有素材的会话】本用例只钉票据账本，不该付一次真合成的代价
        // （debug 构建下模型加载可达分钟级）。空时间线同样会走 `request_reload`
        // 的完整路径：置标志 → 唤醒 actor → 消费重载。
        let owner = Arc::new(ExtensionOwner::default());
        owner.set_setup_sample_rate(48_000.0);
        owner.note_binding_window_closed_without_ara();
        let session = owner.editor_session().unwrap();
        let submitted = session.submitted.load(Ordering::Acquire);
        let processed = session.processed.load(Ordering::Acquire);
        session.request_reload();
        assert_eq!(
            session.submitted.load(Ordering::Acquire),
            submitted,
            "重载不是用户编辑，不得推进票据"
        );
        assert_eq!(
            session.processed.load(Ordering::Acquire),
            processed,
            "重载也不得伪造一次『已处理』"
        );
        assert!(
            session.processed.load(Ordering::Acquire) >= session.submitted.load(Ordering::Acquire),
            "票据必须保持可兑付；否则 actor 永远停在『还有未处理变更』那一支，\
             渲染调度永远走不到（用户报障：永远只听到直通）"
        );
        // 专用唤醒必须真的到达 actor，并且重载标志最终被消费。
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline && session.reload_requested.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !session.reload_requested.load(Ordering::Acquire),
            "actor 必须在唤醒后消费重载标志"
        );
        session.close();
    }

    fn capture_session_body(session: &Arc<EditorSession>) {
        session.ensure_loaded(true).unwrap();

        let (root, path) = {
            let timeline = session.timeline.lock_or_recover();
            assert_eq!(timeline.tracks.len(), 1, "Capture 只有一条自建轨道");
            assert_eq!(timeline.clips.len(), 1, "录进来的音频必须出现在时间线上");
            assert_eq!(
                timeline.params_by_root_track.len(),
                1,
                "根轨道必须有参数状态，否则音高编辑器没有可改的东西"
            );
            let path = timeline.clips[0].source_path.clone().unwrap();
            assert!(
                std::path::Path::new(&path).exists(),
                "界面上的源必须是本会话物化出来的私有副本：{path}"
            );
            (
                timeline.params_by_root_track.keys().next().unwrap().clone(),
                path,
            )
        };

        // 用户改曲线 ⇒ 重新投影不能把它清掉。
        //
        // 【为什么必须置 `pitch_edit_user_modified`】内核的既有规则是"未显式编辑过的
        // 曲线在源变化时清空（它是从旧原线推导出来的），显式编辑过的保留"。真实的参数
        // 写入会置这一位（`ParamHost` 的写入路径），测试要按同一形状喂。
        {
            let mut timeline = session.timeline.lock_or_recover();
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.pitch_edit = vec![62.0; 8];
            params.pitch_edit_user_modified = true;
        }
        session.ensure_loaded(true).unwrap();
        {
            let timeline = session.timeline.lock_or_recover();
            let params = timeline.params_by_root_track.get(&root).unwrap();
            // 【为什么不断言长度】`ensure_params_for_root` 会把曲线规整到"当前工程长度的
            // 目标帧数"（这里比 8 长），那是它的既定职责。要看的是**用户写下的那几个值
            // 还在不在** —— 丢了就等于"改完音高又白改"，而且界面上看不出来。
            assert!(
                params.pitch_edit[..8].iter().all(|value| *value == 62.0),
                "重投影清掉用户曲线（前 8 个值应仍是 62.0）：{:?}",
                &params.pitch_edit[..8]
            );
            assert!(
                params.pitch_edit_user_modified,
                "用户显式编辑过这一位必须保留，否则下一次源变化会把曲线清掉"
            );
        }

        // 渲染要的是源 id 域（`RenderInput.sources` 按 id 索引），不是私有副本路径。
        let render = session.render_timeline().unwrap();
        let id = render.clips[0].source_path.clone().unwrap();
        assert!(
            id.starts_with("capture://"),
            "渲染时间线的源引用必须是源 id：{id}"
        );
        assert_ne!(id, path);

        // 没有文档时播放读数、传输读数、发布都仍然可用（界面每 250ms 问一次）。
        assert_eq!(session.playback_state()["ok"], true);
        assert!(!session.transport().1, "Capture 没有宿主播放控制");
        let current = session.timeline.lock_or_recover().clone();
        session.publish_timeline(current);
        assert!(
            session.error.lock_or_recover().is_none(),
            "Capture 的发布不得因为没有文档而失败：{:?}",
            session.error.lock_or_recover()
        );

        // 命令分发走的是同一条会话（前端启动扇出全是它）。
        let payload = crate::editor::commands::dispatch(
            &session,
            "get_timeline_state",
            serde_json::json!({}),
        )
        .expect("Capture 形态下时间线命令必须可用");
        assert!(
            payload["clips"]
                .as_array()
                .is_some_and(|clips| clips.len() == 1),
            "时间线载荷必须带着录进来的那条片段：{payload}"
        );
    }

    /// 只替代真实DAW的建图/授权边界；使用真正owner、native扩展、actor与state编码。
    pub(crate) fn fixture() -> (ModelHandle, Arc<ExtensionOwner>, Box<u8>) {
        let model = ModelHandle::new();
        let document = model.session();
        let mut timeline:TimelineState=serde_json::from_value(json!({
            "tracks":[{"id":"track","name":"host","order":0,"pitch_analysis_algo":"none"}],
            "clips":[{"id":"clip","track_id":"track","name":"source","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","source_path":"ara://source","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":1
        })).unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock_or_recover() = Some(timeline);
        document.track_bindings.lock_or_recover().insert(
            "track".into(),
            vec![("modification".into(), "ara://source".into())],
        );
        document.edit_sources.lock_or_recover().insert(
            "ara://source".into(),
            Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
                version: 0,
                _reservation: None,
            }),
        );
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock_or_recover()
            .register(key, document.id, 0)
            .unwrap();
        document
            .clip_ids
            .lock_or_recover()
            .insert(key, "clip".into());
        document.regions.lock_or_recover().insert(
            key,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "ara://source".into(),
                audio_modification_persistent_id: "modification".into(),
                duration_in_modification_time: 4.0 / 44100.0,
                duration_in_playback_time: 4.0 / 44100.0,
                ..Default::default()
            },
        );
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: identity和扩展owner在所有测试调用期间存活。
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        document.ready.store(true, Ordering::Release);
        (model, owner, identity)
    }
    fn call(
        editor: &EditorSession,
        sink: &UiSink,
        receiver: &mpsc::Receiver<Value>,
        id: u64,
        command: &str,
        args: Value,
    ) -> Value {
        editor
            .enqueue(UiRequest {
                id,
                command: command.into(),
                args,
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let result = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(result["id"], id);
        assert_eq!(result["ok"], true, "{result}");
        result["value"].clone()
    }
    /// 两真实组件使用同源的不同区域，测试原actor而不是拼接UI快照。
    pub(crate) fn workspace_fixture() -> (ModelHandle, Vec<Arc<ExtensionOwner>>, Vec<Box<u8>>) {
        let (model, a, first) = fixture();
        let document = model.session();
        {
            let mut host = document.timeline.lock_or_recover();
            let host = host.as_mut().unwrap();
            let mut track = host.tracks[0].clone();
            track.id = "b".into();
            track.order = 1;
            host.tracks.push(track);
            let mut clip = host.clips[0].clone();
            clip.id = "cb".into();
            clip.track_id = "b".into();
            host.clips.push(clip);
        }
        document.track_bindings.lock_or_recover().insert(
            "b".into(),
            vec![("modification-b".into(), "ara://source".into())],
        );
        let second = Box::new(0_u8);
        let key = (&*second as *const u8) as u64;
        region_owners()
            .lock_or_recover()
            .register(key, document.id, 1)
            .unwrap();
        document.clip_ids.lock_or_recover().insert(key, "cb".into());
        let mut region = document
            .regions
            .lock_or_recover()
            .values()
            .next()
            .unwrap()
            .clone();
        region.audio_modification_persistent_id = "modification-b".into();
        document.regions.lock_or_recover().insert(key, region);
        let b = Arc::new(ExtensionOwner::default());
        let raw = b
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        // SAFETY: 模型、两owner和真实region身份由返回值保留。
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                key as *mut _,
            );
        }
        (model, vec![a, b], vec![first, second])
    }
    /// 两真实组件统一根编辑后仍按物理区域保存参数；GUI分组/拆组不改宿主轨道图。
    #[test]
    fn goal_feedback_private_group_actor_edit_and_state_roundtrip() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let host = crate::host::MockHost::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let api = Arc::new(host.client());
        let base = api.ui_track(&|| true).unwrap();
        for (id, guid, order) in [
            ("track", "{11111111-1111-1111-1111-111111111111}", 0),
            ("b", "{22222222-2222-2222-2222-222222222222}", 1),
        ] {
            let mut track = base.clone();
            track.id = id.into();
            track.guid = guid.into();
            track.order = order;
            track.items.clear();
            document
                .ui_tracks
                .lock_or_recover()
                .insert(guid.into(), track);
        }
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "private-groups".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let initial = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let a = initial["tracks"][0]["id"].as_str().unwrap().to_owned();
        let b = initial["tracks"][1]["id"].as_str().unwrap().to_owned();
        for (id, track, value) in [(2, &a, 10.), (3, &b, 20.)] {
            call(
                &editor,
                &sink,
                &rx,
                id,
                "set_param_frames",
                json!({"trackId":track,"param":"hifigan_tension","startFrame":0,"values":[value],"checkpoint":true}),
            );
        }
        let grouped = call(
            &editor,
            &sink,
            &rx,
            4,
            "move_track",
            json!({"trackId":b,"targetIndex":0,"parentTrackId":a}),
        );
        assert_eq!(grouped["tracks"][1]["parent_id"], a);
        assert!(
            document
                .timeline
                .lock_or_recover()
                .as_ref()
                .unwrap()
                .tracks
                .iter()
                .all(|track| track.parent_id.is_none()),
            "REAPER图必须保持原样"
        );
        assert_eq!(
            document.edits.lock_or_recover().params["b"].extra_curves["hifigan_tension"][0],
            20.,
            "仅分组不覆盖子轨曲线"
        );
        call(
            &editor,
            &sink,
            &rx,
            5,
            "set_param_frames",
            json!({"trackId":a,"param":"hifigan_tension","startFrame":0,"values":[65.],"checkpoint":true}),
        );
        for id in ["track", "b"] {
            assert_eq!(
                document.edits.lock_or_recover().params[id].extra_curves["hifigan_tension"][0],
                65.
            );
        }
        let encoded = owners[1].encode_state().unwrap();
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&encoded).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded).unwrap()["version"],
            8
        );
        assert_eq!(
            restored.groups,
            document.edits.lock_or_recover().groups,
            "组件保存必须保留共享GUID分组"
        );
        let parent_state = owners[0].encode_state().unwrap();
        let (cold_model, cold_owners, _cold_ids) = workspace_fixture();
        let cold_document = cold_model.session();
        cold_document
            .host_undo
            .pending
            .store(true, Ordering::Release);
        *cold_document.ui_tracks.lock_or_recover() = document.ui_tracks.lock_or_recover().clone();
        cold_owners[0].restore_state(&parent_state).unwrap();
        cold_owners[1].restore_state(&encoded).unwrap();
        let cold_editor = cold_owners[0].editor_session().unwrap();
        cold_editor.ensure_loaded(false).unwrap();
        let cold = cold_editor.timeline.lock_or_recover().clone();
        let cold_b = cold
            .tracks
            .iter()
            .find(|track| track.id.ends_with("-b"))
            .unwrap();
        assert!(
            cold_b
                .parent_id
                .as_ref()
                .is_some_and(|id| id.ends_with("-track")),
            "冷恢复后GUID分组必须重投影到新GUI namespace"
        );
        assert_eq!(
            cold_document.edits.lock_or_recover().params["b"].extra_curves["hifigan_tension"][0],
            65.
        );
        cold_document.close();
        let ungrouped = call(
            &editor,
            &sink,
            &rx,
            6,
            "move_track",
            json!({"trackId":b,"targetIndex":1,"parentTrackId":null}),
        );
        assert!(ungrouped["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|track| track["parent_id"].is_null()));
        document.close();
        assert!(
            !host
                .calls()
                .iter()
                .any(|call| call.starts_with("set-") || call.starts_with("undo-")),
            "私有分组不能调用宿主setter"
        );
    }
    /// 与已有WORLD oracle相同的谐波源，双轨同源但持久modification身份不同。
    fn task34_world_fixture() -> (ModelHandle, Vec<Arc<ExtensionOwner>>, Vec<Box<u8>>) {
        let (model, owners, ids) = workspace_fixture();
        let document = model.session();
        {
            let mut known = document.timeline.lock_or_recover();
            let timeline = known.as_mut().unwrap();
            timeline.project_sec = 2.;
            for track in &mut timeline.tracks {
                track.compose_enabled = true;
                track.pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            }
            for clip in &mut timeline.clips {
                clip.length_sec = 2.;
                clip.takes[0].source_end_sec = 2.;
                clip.normalize_takes();
            }
        }
        for region in document.regions.lock_or_recover().values_mut() {
            region.duration_in_modification_time = 2.;
            region.duration_in_playback_time = 2.;
        }
        let samples = (0..88200)
            .map(|n| {
                let phase = 2. * std::f64::consts::PI * 220. * n as f64 / 44100.;
                (1..=16)
                    .map(|harmonic| (phase * harmonic as f64).sin() * 0.2 / harmonic as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        let pcm = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![samples],
            version: 0,
            _reservation: None,
        });
        document
            .edit_sources
            .lock_or_recover()
            .insert("ara://source".into(), pcm.clone());
        document
            .sources
            .lock_or_recover()
            .insert("ara://source".into(), pcm);
        (model, owners, ids)
    }
    /// 生产route尾块在view关闭后仍先入权威；render失败也能保存，共享undo保存后无GUI恢复供音。
    #[test]
    fn task34_tail_and_shared_undo_save_restore_both_world_tracks_without_gui() {
        let (model, owners, _ids) = task34_world_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let leases = owners
            .iter()
            .map(super::super::routing::RouteLease::new)
            .collect::<Vec<_>>();
        let links = leases
            .iter()
            .map(|lease| {
                let link = Arc::new(super::super::routing::EditorLink::default());
                link.bind(std::process::id() as i64, lease.token()).unwrap();
                link
            })
            .collect::<Vec<_>>();
        let mut sinks = Vec::new();
        let mut receivers = Vec::new();
        for index in 0..2 {
            let (reply, rx) = mpsc::channel();
            let (events, _) = mpsc::sync_channel(128);
            sinks.push(UiSink {
                view_id: format!("task34-world-{index}"),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            });
            receivers.push(rx);
        }
        let request = |index: usize, id: u64, command: &str, args: Value| {
            editor
                .enqueue(UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sinks[index].clone(),
                    link: Some(links[index].clone()),
                })
                .unwrap();
            let response = receivers[index]
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let timeline = request(0, 1, "get_timeline_state", json!({}));
        let ta = timeline["tracks"][0]["id"].as_str().unwrap();
        let tb = timeline["tracks"][1]["id"].as_str().unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        for track in [ta, tb] {
            loop {
                let frames = request(
                    0,
                    2,
                    "get_param_frames",
                    json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}),
                );
                if frames["orig"]
                    .as_array()
                    .is_some_and(|orig| orig.iter().filter_map(Value::as_f64).any(|p| p > 30.))
                {
                    break;
                }
                assert!(Instant::now() < deadline, "WORLD原线未完成: {frames}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        request(
            0,
            3,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":false}),
        );
        request(
            0,
            4,
            "set_param_frames",
            json!({"trackId":ta,"param":"pitch","startFrame":0,"values":vec![60.;400],"checkpoint":true}),
        );
        request(
            1,
            5,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":vec![67.;400],"checkpoint":true}),
        );
        let held = editor.timeline.lock_or_recover();
        editor
            .enqueue(UiRequest {
                id: 6,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sinks[1].clone(),
                link: Some(links[1].clone()),
            })
            .unwrap();
        editor.enqueue(UiRequest {id:7,command:"set_param_frames".into(),args:json!({"trackId":tb,"param":"pitch","startFrame":399,"values":[69.],"checkpoint":false}),sink:sinks[1].clone(),link:Some(links[1].clone())}).unwrap();
        sinks[1].closed.store(true, Ordering::Release);
        drop(held);
        let saved = [
            owners[0].encode_state().unwrap(),
            owners[1].encode_state().unwrap(),
        ];
        // 【为什么用"宿主未就绪"当失败触发】本用例要验的是"合成失败也不能丢权威尾笔"。
        // 此前用"委托淡化缺宿主几何"，但那条路径已经改成**降级**（保留 ARA 自己声明的
        // 淡化，不整份失败）—— 它在 ARA-only 宿主上曾让整轨静音，见
        // `project_audio_fades_locked`。`ready=false` 会让 `apply_workspace_edits` 以
        // "host model changed during automatic apply" 拒绝，触发条件等价而不再依赖
        // 那个已被改掉的行为。恢复 `ready` 以免影响后续的撤销/恢复断言。
        document.ready.store(false, Ordering::Release);
        let applied = editor.apply(editor.submitted.load(Ordering::Acquire), None);
        document.ready.store(true, Ordering::Release);
        assert!(applied.is_err(), "合成必须失败，才能验证尾笔不依赖它");
        let b: Value = serde_json::from_slice(&saved[1]).unwrap();
        assert_eq!(b["edits"]["params"]["b"]["pitch_edit"][399], 69.);
        assert_eq!(
            editor.history.lock_or_recover().position,
            2,
            "尾笔不新增undo步"
        );
        assert!(editor
            .enqueue(UiRequest {
                id: 8,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sinks[1].clone(),
                link: Some(links[1].clone())
            })
            .is_err());
        request(0, 9, "undo_timeline", json!({}));
        let undone = [
            owners[0].encode_state().unwrap(),
            owners[1].encode_state().unwrap(),
        ];
        for (states, b_midi) in [(saved, 67.), (undone, 64.)] {
            let (cold, cold_owners, _cold_ids) = task34_world_fixture();
            let cold_document = cold.session();
            // 不创建editor_session，不轮询GUI命令，直接驱动原setState和真实publisher供音。
            cold_document.ready.store(false, Ordering::Release);
            for (owner, bytes) in cold_owners.iter().zip(&states) {
                owner.restore_state(bytes).unwrap();
            }
            cold_document.prepare_renderers();
            // 【为什么是 60s 而不是 15s】这里等的是**真实的神经 DSP 预热**
            // （`PreparationQueue` 跑的是 HiFiGAN / HNSEP 推理），不是被断言的那件事。
            // 断言的本意是"准备**会**结束"，而 15s 只保证"在空载机器上结束" ——
            // 实测在同时跑宿主（CPU 被占）时同一段要 ~35s，于是这条成了假阴性。
            // 本文件对同类 DSP 等待本来就用 30s（见下面的 `analysis` 与 render 等待），
            // 这里取 60s 给最坏情况留余量。
            let deadline = Instant::now() + Duration::from_secs(60);
            for (index, owner) in cold_owners.iter().enumerate() {
                while owner.preparation_state().0 {
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(10));
                }
                assert!(
                    owner.preparation_state().1.is_none(),
                    "{:?}",
                    owner.preparation_state()
                );
                let mut left = vec![0.; 8820];
                let mut right = vec![0.; 8820];
                let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut bus = crate::audio_abi::AudioBusBuffers {
                    num_channels: 2,
                    silence_flags: 0,
                    channel_buffers: planes.as_mut_ptr(),
                };
                // SAFETY: 两个8820帧平面；oracle读取实际后台准备发布的PCM。
                assert!(unsafe { owner.snapshots[0].copy_block(22050, 44100, &mut bus, 8820) });
                assert!(left.iter().all(|sample| sample.is_finite()));
                assert_eq!(left, right);
                let correlation = |lag: usize| {
                    let mut xy = 0.;
                    let mut xx = 0.;
                    let mut yy = 0.;
                    for n in 0..left.len() - lag {
                        let a = left[n] as f64;
                        let b = left[n + lag] as f64;
                        xy += a * b;
                        xx += a * a;
                        yy += b * b;
                    }
                    xy / (xx * yy).sqrt().max(1e-20)
                };
                let target =
                    440. * 2_f64.powf(((if index == 0 { 60. } else { b_midi }) - 69.) / 12.);
                let expected_lag = 44100. / target;
                let first = (expected_lag * 0.8) as usize;
                let last = (expected_lag * 1.2) as usize;
                let lag = (first..=last)
                    .max_by(|a, b| correlation(*a).total_cmp(&correlation(*b)))
                    .unwrap();
                let hz = 44100. / lag as f64;
                eprintln!("Task34 cold track={index} B_midi={b_midi} frequency={hz:.3}Hz target={target:.3}Hz");
                assert!(
                    (hz - target).abs() < 8.,
                    "cold WORLD pitch mismatch {hz} vs {target}"
                );
                let local: Value = serde_json::from_slice(&owner.encode_state().unwrap()).unwrap();
                assert_eq!(local["edits"]["tracks"].as_array().unwrap().len(), 1);
                assert_eq!(local["edits"]["params"].as_object().unwrap().len(), 1);
                assert_eq!(
                    local["edits"]["params"][if index == 0 { "track" } else { "b" }]["pitch_edit"]
                        [100],
                    if index == 0 { 60. } else { b_midi }
                );
            }
            cold_document.close();
        }
        // 模型变化必须拒绝旧actor投影，不能把曲线保存误报成音频已经同步。
        document.clear_renderers();
        editor
            .enqueue(UiRequest {
                id: 10,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sinks[0].clone(),
                link: Some(links[0].clone()),
            })
            .unwrap();
        let rejected = receivers[0].recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(rejected["ok"], false);
        assert!(rejected["error"]
            .as_str()
            .unwrap()
            .contains("local curves preserved"));
        let local = editor.timeline.lock_or_recover();
        assert_eq!(local.params_by_root_track[ta].pitch_edit[100], 60.);
        assert_eq!(local.params_by_root_track[tb].pitch_edit[100], 64.);
        drop(local);
        assert_eq!(editor.state()["pending"], true);
        // 【为什么不再要求失败】宿主保存线程不能因为"图还没就绪"而拿不到状态；
        // 保存一份当前可用的载荷，恢复端再逐条 reconcile。
        assert!(
            owners[0].encode_state().is_ok(),
            "图未就绪时保存也必须成功"
        );
        document.close();
    }
    /// 最后一个renderer入口撤销后，actor屏障/空范围应用/文档join都不等待不存在的renderer。
    #[test]
    fn task34_no_renderer_does_not_retry_or_fall_back_to_the_whole_document() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "task34-no-renderer".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":timeline["tracks"][0]["id"],"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),
        );
        for owner in &owners {
            owner.stop_editor();
        }
        let projection = document.workspace_projection().unwrap();
        let workspace = document.workspace_timeline().unwrap();
        assert!(workspace.tracks.is_empty());
        assert!(workspace.clips.is_empty());
        let began = Instant::now();
        editor.flush().unwrap();
        let edit = document.edits.lock_or_recover().revision;
        document
            .apply_workspace_edits(
                edit,
                document.revision.load(Ordering::Acquire),
                &projection,
                Arc::new(AtomicBool::new(false)),
                || true,
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while editor.error.lock_or_recover().is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(editor
            .error
            .lock_or_recover()
            .as_ref()
            .unwrap()
            .contains("local curves preserved"));
        assert_eq!(
            editor.state()["pending"],
            true,
            "失去全部renderer不能把保存曲线伪报成已渲染"
        );
        assert!(
            !editor.closed.load(Ordering::Acquire),
            "组件入口都撤销也不能误关文档actor"
        );
        document.close();
        assert!(began.elapsed() < Duration::from_secs(3));
        assert!(editor.worker.lock_or_recover().is_none());
    }
    /// 旧view仍持actor Arc时，文档close也必须在join之后释放分析投影/历史和持久权威缓存。
    #[test]
    fn task34_document_close_releases_state_even_when_the_closed_actor_is_retained() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "task34-retained-actor".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":timeline["tracks"][0]["id"],"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),
        );
        let authority = document.edits.clone();
        document.close();
        assert!(editor.worker.lock_or_recover().is_none());
        assert!(editor.analysis_workers.lock_or_recover().is_empty());
        assert!(
            editor.timeline.lock_or_recover().clips.is_empty(),
            "join后旧actor不得继续持有分析投影"
        );
        assert!(editor.history.lock_or_recover().records.is_empty());
        assert!(editor.peaks.lock_or_recover().is_empty());
        assert!(!editor.loaded.lock_or_recover().initialized);
        assert!(document.track_bindings.lock_or_recover().is_empty());
        assert!(authority.lock_or_recover().params.is_empty());
        assert!(authority.lock_or_recover().tracks.is_empty());
    }
    #[test]
    fn task34_component_stop_revokes_its_views_and_output_before_returning() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let leases = owners
            .iter()
            .map(super::super::routing::RouteLease::new)
            .collect::<Vec<_>>();
        let mut sinks = Vec::new();
        let mut receivers = Vec::new();
        let mut links = Vec::new();
        for (index, lease) in leases.iter().enumerate() {
            let link = Arc::new(super::super::routing::EditorLink::default());
            link.bind(std::process::id() as i64, lease.token()).unwrap();
            let (reply, rx) = mpsc::channel();
            let (events, _) = mpsc::sync_channel(8);
            let sink = UiSink {
                view_id: format!("task34-stop-{index}"),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            };
            editor
                .enqueue(UiRequest {
                    id: 1,
                    command: "get_ui_settings".into(),
                    args: json!({}),
                    sink: sink.clone(),
                    link: Some(link.clone()),
                })
                .unwrap();
            assert_eq!(rx.recv_timeout(Duration::from_secs(3)).unwrap()["ok"], true);
            owners[index].snapshots[0]
                .publish(crate::render::snapshot::PlaybackSnapshot {
                    sample_rate: 44100,
                    origin_sample: 0,
                    left: vec![0.25; 4],
                    right: vec![0.5; 4],
                    covered: Vec::new(),
                    _reservation: None,
                })
                .unwrap();
            sinks.push(sink);
            receivers.push(rx);
            links.push(link);
        }
        owners[0].stop_editor();
        // 失败也先join文档worker，避免把RED断言变成线程退出问题。
        let revoked = sinks[0].closed.load(Ordering::Acquire);
        let other_live = !sinks[1].closed.load(Ordering::Acquire);
        let output = owners
            .iter()
            .map(|owner| {
                let mut left = [9.; 4];
                let mut right = [9.; 4];
                let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut bus = crate::audio_abi::AudioBusBuffers {
                    num_channels: 2,
                    silence_flags: 0,
                    channel_buffers: planes.as_mut_ptr(),
                };
                // SAFETY: 两个平面各四帧；观察真实发布输出是否已经撤销。
                unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) }
            })
            .collect::<Vec<_>>();
        assert!(editor
            .enqueue(UiRequest {
                id: 2,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sinks[0].clone(),
                link: Some(links[0].clone())
            })
            .is_err());
        editor
            .enqueue(UiRequest {
                id: 3,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sinks[1].clone(),
                link: Some(links[1].clone()),
            })
            .unwrap();
        assert_eq!(
            receivers[1].recv_timeout(Duration::from_secs(3)).unwrap()["ok"],
            true
        );
        document.close();
        assert!(revoked, "组件stop必须同步撤销自己的view");
        assert!(other_live, "另一组件view仍有租约");
        assert_eq!(
            output,
            [false, true],
            "组件stop必须撤销自己的输出，不能清除另一renderer"
        );
    }
    /// 原actor选择重叠region只换其参数投影；不能生成编辑/history或覆盖另一素材。
    #[test]
    fn atlas_selection_commands_project_overlaps_without_audio_or_history_changes() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let mut original = document
            .timeline
            .lock_or_recover()
            .as_ref()
            .unwrap()
            .clone();
        for (index, track) in original.tracks.iter().enumerate() {
            original.params_by_root_track.insert(
                track.id.clone(),
                hifishifter_kernel::state::TrackParamsState {
                    frame_period_ms: 5.,
                    pitch_edit: vec![if index == 0 { 60. } else { 67. }; 2],
                    ..Default::default()
                },
            );
        }
        let identities = document.parameter_identities_locked(&original).unwrap();
        let atlas = super::super::parameter_atlas::ParameterAtlas::default()
            .capture(&original, &identities)
            .unwrap();
        let root = original.tracks[0].id.clone();
        let ca = original.clips[0].id.clone();
        let cb = original.clips[1].id.clone();
        original.clips[1].track_id = root.clone();
        original.tracks.truncate(1);
        original.selected_clip_id = Some(ca.clone());
        let atlas = atlas.follow_geometry(&original, &identities).unwrap();
        original.params_by_root_track = atlas.project_roots(&original, &identities).unwrap();
        *document.timeline.lock_or_recover() = Some(original.clone());
        document.track_bindings.lock_or_recover().insert(
            root.clone(),
            vec![
                ("modification".into(), "ara://source".into()),
                ("modification-b".into(), "ara://source".into()),
            ],
        );
        {
            let mut edits = document.edits.lock_or_recover();
            edits.params = original.params_by_root_track;
            edits.atlas = atlas;
        }
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "atlas-select".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let root = loaded["tracks"][0]["id"].as_str().unwrap();
        let ca = format!("{}{ca}", editor.namespace);
        let cb = format!("{}{cb}", editor.namespace);
        let before = (
            document.edits.lock_or_recover().revision,
            editor.generation.load(Ordering::Acquire),
        );
        call(&editor, &sink, &rx, 2, "select_clip", json!({"clipId":cb}));
        let b = editor.timeline.lock_or_recover().params_by_root_track[root].pitch_edit[0];
        call(&editor, &sink, &rx, 3, "select_clip", json!({"clipId":ca}));
        let a = editor.timeline.lock_or_recover().params_by_root_track[root].pitch_edit[0];
        let after = (
            document.edits.lock_or_recover().revision,
            editor.generation.load(Ordering::Acquire),
        );
        let history = editor.history.lock_or_recover().records.len();
        document.close();
        assert_eq!((a, b), (60., 67.));
        assert_eq!(before, after);
        assert_eq!(history, 0);
    }
    #[test]
    fn workspace_components_share_the_original_actor_and_cross_track_history() {
        let (_model, owners, _ids) = workspace_fixture();
        let a = owners[0].editor_session().unwrap();
        let b = owners[1].editor_session().unwrap();
        assert!(Arc::ptr_eq(&a, &b), "同文档必须只有一个actor/history权威");
        let (_other, owner, _id) = fixture();
        assert!(!Arc::ptr_eq(&a, &owner.editor_session().unwrap()));
        let (reply, rx) = mpsc::channel();
        let (events, event_rx) = mpsc::sync_channel(128);
        let sa = UiSink {
            view_id: "workspace-a".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let (reply, rb) = mpsc::channel();
        let (events, eb) = mpsc::sync_channel(128);
        let sb = UiSink {
            view_id: "workspace-b".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let leases = owners
            .iter()
            .map(super::super::routing::RouteLease::new)
            .collect::<Vec<_>>();
        let links = leases
            .iter()
            .map(|lease| {
                let link = Arc::new(super::super::routing::EditorLink::default());
                link.bind(std::process::id() as i64, lease.token()).unwrap();
                link
            })
            .collect::<Vec<_>>();
        let call = |editor: &EditorSession,
                    sink: &UiSink,
                    receiver: &mpsc::Receiver<Value>,
                    id: u64,
                    command: &str,
                    args: Value| {
            let link = links[usize::from(sink.view_id == "workspace-b")].clone();
            editor
                .enqueue(UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sink.clone(),
                    link: Some(link),
                })
                .unwrap();
            let result = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(result["id"], id);
            assert_eq!(result["ok"], true, "{result}");
            result["value"].clone()
        };
        let loaded = call(&a, &sa, &rx, 1, "get_timeline_state", json!({}));
        assert_eq!(loaded["tracks"].as_array().unwrap().len(), 2);
        let ta = loaded["tracks"][0]["id"].as_str().unwrap();
        let tb = loaded["tracks"][1]["id"].as_str().unwrap();
        call(
            &a,
            &sa,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":[64.],"checkpoint":false}),
        );
        call(&a, &sa, &rx, 3, "select_track", json!({"trackId":tb}));
        call(
            &a,
            &sa,
            &rx,
            4,
            "set_param_frames",
            json!({"trackId":ta,"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),
        );
        let second = call(&b, &sb, &rb, 5, "get_timeline_state", json!({}));
        assert_eq!(second["selected_track_id"], tb);
        assert_eq!(second["undo_depth"], 1);
        event_rx.try_iter().for_each(drop);
        eb.try_iter().for_each(drop);
        call(
            &b,
            &sb,
            &rb,
            6,
            "set_param_frames",
            json!({"trackId":tb,"param":"pitch","startFrame":0,"values":[67.],"checkpoint":true}),
        );
        assert!(
            event_rx
                .try_iter()
                .any(|e| e["event"] == "plugin_host_changed"),
            "其它view需要收到原GUI时间线刷新事件"
        );
        assert!(eb.try_iter().any(|e| e["event"] == "plugin_host_changed"));
        call(&a, &sa, &rx, 7, "undo_timeline", json!({}));
        assert_eq!(
            a.timeline.lock_or_recover().params_by_root_track[ta].pitch_edit[0],
            60.
        );
        assert_eq!(
            a.timeline.lock_or_recover().params_by_root_track[tb].pitch_edit[0],
            64.
        );
        call(&b, &sb, &rb, 8, "redo_timeline", json!({}));
        assert_eq!(
            a.timeline.lock_or_recover().params_by_root_track[tb].pitch_edit[0],
            67.
        );
        assert!(event_rx.try_iter().any(|e| e["event"] == "history_state"));
        assert!(eb.try_iter().any(|e| e["event"] == "history_state"));
        owners[0].stop_editor();
        assert!(
            !a.closed.load(Ordering::Acquire),
            "关闭组件不能停止共享actor"
        );
        assert!(call(&b, &sb, &rb, 9, "get_ui_settings", json!({})).is_object());
        a.close();
    }
    #[test]
    fn workspace_queued_requests_revalidate_original_routes_and_reject_unknown_views() {
        use super::super::routing::{EditorLink, RouteLease};
        let (model, owners, _ids) = workspace_fixture();
        let editor = owners[0].editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let ra = RouteLease::new(&owners[0]);
        let rb = RouteLease::new(&owners[1]);
        let link = Arc::new(EditorLink::default());
        link.bind(std::process::id() as i64, ra.token()).unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-route".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        // 真实命令在timeline锁上停住，随后命令一定排队；clear/bind不需持document事务。
        let held = editor.timeline.lock_or_recover();
        editor
            .enqueue(UiRequest {
                id: 1,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link.clone()),
            })
            .unwrap();
        let track = held.tracks[0].id.clone();
        editor.enqueue(UiRequest {id:2,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":0,"values":[60.],"checkpoint":true}),sink:sink.clone(),link:Some(link.clone())}).unwrap();
        link.clear();
        link.bind(std::process::id() as i64, rb.token()).unwrap();
        drop(held);
        let mut results = [
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
        ];
        results.sort_by_key(|r| r["id"].as_u64().unwrap());
        assert_eq!(
            results[1]["ok"], false,
            "排队入口不能重新绑定别的组件继续写入"
        );
        assert_eq!(model.session().edits.lock_or_recover().revision, 0);
        drop(rb);
        assert!(
            editor
                .enqueue(UiRequest {
                    id: 3,
                    command: "get_ui_settings".into(),
                    args: json!({}),
                    sink: sink.clone(),
                    link: Some(link.clone())
                })
                .is_err(),
            "过期route拒绝入队"
        );
        let unknown = Arc::new(EditorLink::default());
        assert!(editor
            .enqueue(UiRequest {
                id: 4,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(unknown)
            })
            .is_err());
        link.bind(std::process::id() as i64, ra.token()).unwrap();
        editor
            .enqueue(UiRequest {
                id: 5,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link.clone()),
            })
            .unwrap();
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(5)).unwrap()["ok"],
            true
        );
        let (reply, _) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(8);
        let forged = UiSink {
            view_id: sink.view_id.clone(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        assert!(editor
            .enqueue(UiRequest {
                id: 6,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: forged,
                link: Some(link.clone())
            })
            .is_err());
        sink.closed.store(true, Ordering::Release);
        assert!(editor
            .enqueue(UiRequest {
                id: 7,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink,
                link: Some(link)
            })
            .is_err());
        editor.close();
    }
    #[test]
    fn workspace_queued_origin_close_cannot_borrow_the_other_live_component() {
        let (_model, owners, _ids) = workspace_fixture();
        let editor = owners[0].editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let lease = super::super::routing::RouteLease::new(&owners[0]);
        let link = Arc::new(super::super::routing::EditorLink::default());
        link.bind(std::process::id() as i64, lease.token()).unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, event_rx) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-origin-close".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let held = editor.timeline.lock_or_recover();
        editor
            .enqueue(UiRequest {
                id: 1,
                command: "get_timeline_state".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link.clone()),
            })
            .unwrap();
        editor
            .enqueue(UiRequest {
                id: 2,
                command: "get_ui_settings".into(),
                args: json!({}),
                sink: sink.clone(),
                link: Some(link),
            })
            .unwrap();
        owners[0].stop_editor();
        drop(held);
        editor.flush().unwrap();
        assert!(
            sink.closed.load(Ordering::Acquire),
            "组件关闭同步撤销旧view回信"
        );
        assert!(receiver.try_iter().next().is_none());
        assert!(!editor.closed.load(Ordering::Acquire));
        assert!(Arc::ptr_eq(&editor, &owners[1].editor_session().unwrap()));
        assert!(
            event_rx.try_iter().next().is_none(),
            "入口撤销后不能通过共享actor继续订阅事件"
        );
        editor.close();
    }
    #[test]
    fn workspace_document_close_joins_actor_revokes_views_and_releases_model_pcm() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let weak = Arc::downgrade(&document);
        let editor = owners[0].editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-lifecycle".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let started = Instant::now();
        document.close();
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "document close应join而不持事务等待actor"
        );
        assert!(editor.worker.lock_or_recover().is_none());
        assert!(editor.analysis_workers.lock_or_recover().is_empty());
        assert!(document.timeline.lock_or_recover().is_none());
        assert!(document.edit_sources.lock_or_recover().is_empty());
        assert!(
            sink.closed.load(Ordering::Acquire),
            "文档关闭撤销全部view回信/事件"
        );
        assert!(owners[1].editor_session().is_err());
        assert!(document.workspace_projection().is_err());
        drop(document);
        drop(model);
        assert!(weak.upgrade().is_none(), "actor不能强持document");
    }
    #[test]
    fn workspace_scope_and_geometry_checks_reject_partial_or_stale_transactions() {
        let (model, owners, _ids) = workspace_fixture();
        let document = model.session();
        let host = document.workspace_timeline().unwrap();
        let previous = document.workspace_projection().unwrap();
        let version = document.revision.load(Ordering::Acquire);
        for changed in 0..5 {
            let mut client = host.clone();
            match changed {
                0 => {
                    client.tracks.pop();
                }
                1 => client.tracks[1] = client.tracks[0].clone(),
                2 => {
                    client.clips.pop();
                }
                3 => client.clips[0].takes[0].source_path = Some("C:/private.wav".into()),
                _ => client.clips[0].takes[0].playback_rate = 2.,
            }
            assert!(document
                .accept_workspace_edits(0, version, &client, &previous)
                .is_err());
            assert_eq!(
                document.edits.lock_or_recover().revision,
                0,
                "拒绝前不能部分提交"
            );
        }
        owners[0].stop_editor();
        assert_eq!(document.revision.load(Ordering::Acquire), version);
        assert!(
            document
                .accept_workspace_edits(0, version, &host, &previous)
                .unwrap_err()
                .starts_with("Conflict"),
            "scope必须独立于model校验"
        );
        assert_eq!(document.edits.lock_or_recover().revision, 0);
    }
    #[test]
    fn workspace_accept_rejects_forged_geometry_before_changing_document_edits() {
        let (model, _owners, _ids) = workspace_fixture();
        let document = model.session();
        let projection = document.workspace_projection().unwrap();
        let mut client = document.workspace_timeline().unwrap();
        client.clips[0].start_sec = 42.;
        assert!(
            document
                .accept_workspace_edits(
                    0,
                    document.revision.load(Ordering::Acquire),
                    &client,
                    &projection
                )
                .is_err(),
            "宿主几何必须拒绝而不是忽略"
        );
        assert_eq!(document.edits.lock_or_recover().revision, 0);
    }
    #[test]
    fn workspace_global_solo_keeps_each_renderer_output_independent() {
        let (_model, owners, _ids) = workspace_fixture();
        let editor = owners[0].editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "workspace-solo".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_track_state",
            json!({"trackId":track,"solo":true,"volume":0.5}),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while editor.applied.load(Ordering::Acquire) < 1 {
            assert!(
                Instant::now() < deadline,
                "{:?}",
                editor.error.lock_or_recover()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let outputs = owners
            .iter()
            .map(|owner| {
                let mut left = [9.; 4];
                let mut right = [9.; 4];
                let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut bus = crate::audio_abi::AudioBusBuffers {
                    num_channels: 2,
                    silence_flags: 0,
                    channel_buffers: planes.as_mut_ptr(),
                };
                // SAFETY: 每个publisher仅写本次提供的两个四帧平面。
                assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
                left
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], [0.05, 0.1, 0.15, 0.2]);
        assert_eq!(outputs[1], [0.; 4], "其它轨solo不能在本renderer投影后丢失");
        editor.close();
    }
    #[test]
    fn actor_accepts_original_curve_and_save_flush_keeps_tail_after_ui_closed() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "test-view".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        assert!(track.starts_with(&editor.namespace));
        assert_ne!(timeline["clips"][0]["source_path"], "ara://source");
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_param_frames",
            json!({"trackId":track,"param":"pitch","startFrame":0,"values":[60.0],"checkpoint":true}),
        );
        sink.closed.store(true, Ordering::Release);
        editor.enqueue(UiRequest {id:3,command:"set_param_frames".into(),args:json!({"trackId":track,"param":"pitch","startFrame":1,"values":[64.0],"checkpoint":false}),sink,link:None}).unwrap();
        let encoded = owner.encode_state().unwrap(); // 真getState路径会flush，不只测手工barrier。
        let saved: Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            saved["version"], 8,
            "权威参数按稳定宿主身份存档（v8：曲线按源帧索引）；旧 v2/v3 解码与范围回归另行保留"
        );
        assert_eq!(
            &saved["edits"]["params"]["track"]["pitch_edit"]
                .as_array()
                .unwrap()[..2],
            &[json!(60.), json!(64.)]
        );
        assert_eq!(
            editor.history.lock_or_recover().position,
            1,
            "尾块不得增加undo步"
        );
        editor.close();
    }
    #[test]
    fn invalid_track_patch_cannot_partially_mutate_volume_or_create_history() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let track = editor.timeline.lock_or_recover().tracks[0].id.clone();
        let result = super::super::commands::dispatch(
            &editor,
            "set_track_state",
            json!({"trackId":track,"volume":0.5,"pitchAnalysisAlgo":"not-an-algorithm"}),
        );
        assert!(result.is_err());
        assert_eq!(editor.timeline.lock_or_recover().tracks[0].volume, 1.);
        assert!(editor.history.lock_or_recover().records.is_empty());
        assert_eq!(editor.generation.load(Ordering::Acquire), 0);
        assert!(editor.check_source("C:/Users/user/private.wav").is_err());
        editor.close();
    }
    /// 原前端以base+position消费；停播seek也必须得到宿主绝对位置。
    #[test]
    fn host_cursor_is_absolute_once_and_tracks_stopped_seeks() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        // 时钟投影与分析无关；标记夹具已加载，避免为四帧源启动异步ONNX预热。
        editor.loaded.lock_or_recover().initialized = true;
        owner
            .clock
            .get()
            .unwrap()
            .update(&crate::audio_abi::ProcessContext {
                state: 0,
                sample_rate: 44100.,
                project_time_samples: 88200,
                ..Default::default()
            });
        let playback =
            super::super::commands::dispatch(&editor, "get_playback_state", json!({})).unwrap();
        let timeline =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        editor.close();
        assert_eq!(
            playback["base_sec"].as_f64().unwrap() + playback["position_sec"].as_f64().unwrap(),
            2.
        );
        assert_eq!(timeline["playhead_sec"], 2.);
    }
    /// 宿主tempo/geometry通知必须在无人点击重新载入时进入原GUI。
    #[test]
    fn host_changes_and_bpm_sync_automatically_without_reloading() {
        let (model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "host-sync".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        owner
            .clock
            .get()
            .unwrap()
            .update(&crate::audio_abi::ProcessContext {
                state: 1 << 10,
                sample_rate: 44100.,
                tempo: 150.,
                ..Default::default()
            });
        let document = model.session();
        document.timeline.lock_or_recover().as_mut().unwrap().clips[0].start_sec = 2.;
        document.revision.fetch_add(1, Ordering::AcqRel);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let synced = editor.timeline.lock_or_recover().clone();
            if synced.clips[0].start_sec == 2. && synced.bpm == 150. {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "宿主改动没有自动同步: start={} bpm={}",
                synced.clips[0].start_sec,
                synced.bpm
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        editor.close();
    }
    /// 倒放是**宿主**的既有状态，现在**照常物化**（不再隔离），且绝不让整份会话失败。
    ///
    /// 【为什么这条是本测试的重点】此前 `ensure_loaded` 里一句
    /// `any(|clip| clip.reversed)` 就 `Err`，于是"一个片段倒放"升级成"整个插件打不开"。
    /// 中间态是 `ReversePolicy::Isolate`：不物化、界面按"由 REAPER 处理"显示 —— 它只
    /// 解决"不打死会话"，代价是倒放片段**永远渲染不出来**。F-4 判明 ARA 报镜像坐标后，
    /// 播种层（`project_host_take_facts_locked`）把窗口翻回正向，内核的倒放数学自会反向
    /// 取窗，所以现在 `Materialize`：倒放 clip 照常带源。
    #[test]
    fn a_reversed_clip_is_materialized_instead_of_failing_the_session() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let mut timeline = document.timeline.lock_or_recover();
        let clip = &mut timeline.as_mut().unwrap().clips[0];
        clip.reversed = true;
        clip.takes[0].reversed = true;
        drop(timeline);
        let editor = owner.editor_session().unwrap();
        // 会话照常加载 —— 既不整份失败，也不再隔离。
        editor.ensure_loaded(false).unwrap();
        let loaded = editor.timeline.lock_or_recover().clone();
        let clip = &loaded.clips[0];
        assert!(clip.reversed, "宿主事实要保留下来，界面才能显示倒放标记");
        assert!(
            clip.source_path.is_some(),
            "倒放 clip 现在照常物化：内核按反向窗口消费这份正向 PCM"
        );
        assert!(
            clip.takes.iter().all(|take| take.source_path.is_some()),
            "take 层同样带源引用（否则 normalize_takes 会把扁平投影清空）"
        );
        // 倒放 clip 也必须纳入分析：分析读正向完整源，装配期再整体翻转。
        // 排除它会让 f0 全零 ⇒ 只剩气声（用户报障的另一半）。
        assert!(
            editor.analysis_timeline().clips.iter().any(|c| c.reversed),
            "倒放 clip 必须出现在分析时间线上"
        );
        // 播放观察照常工作（此前那条路径被整份失败连带打断）。
        owner
            .clock
            .get()
            .unwrap()
            .update(&crate::audio_abi::ProcessContext {
                state: 1 << 1,
                sample_rate: 44100.,
                project_time_samples: 44100,
                ..Default::default()
            });
        let playing =
            super::super::commands::dispatch(&editor, "get_playback_state", json!({})).unwrap();
        assert_eq!(playing["is_playing"], true);
        owner.clock.get().unwrap().stopped();
        let state = editor.state();
        editor.close();
        assert_eq!(state["ready"], true, "会话必须处于就绪态，而不是报错态");
    }
    /// 方向 / 循环源 / 声道模式是 ARA 表达不到的渲染输入，**必须**参与工作区指纹。
    ///
    /// 【为什么这是回归测试】宿主倒放一条 take 后，`prepare_job` 只看 `PreparedVersion`
    /// （model/edit/epoch/scope/keys），而 `ensure_loaded` 早退只看工作区投影。此前
    /// 方向既不进 `render_epoch`（只有淡变长度有写者）也不进投影指纹（只哈希
    /// `params_by_root_track` / `tracks`，不含 `clips`）—— 于是倒放后既不重载也不重渲染，
    /// 用户听到的还是旧的正向音频，直到手动编辑一次才被 `edit` 分量带出来。
    #[test]
    fn render_facts_participate_in_the_workspace_projection() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let before = document.workspace_projection().unwrap();

        document.render_facts.lock_or_recover().items.insert(
            "item-1".into(),
            crate::host::geometry::RenderItemFacts {
                reversed: true,
                loop_enabled: false,
                channel_mode: 0,
                timestretch: false,
                content_fade_head: false,
                content_fade_tail: false,
            },
        );
        let after = document.workspace_projection().unwrap();
        assert_ne!(
            before, after,
            "宿主把一条 take 倒放必须改变工作区指纹，否则 ensure_loaded 会带着旧方向早退"
        );

        // 幂等：同样的方向再投影一次不得继续变化（否则每轮清单刷新都会重载）。
        let again = document.workspace_projection().unwrap();
        assert_eq!(after, again, "同一份渲染事实必须产生同一个指纹");
        owner.editor_session().unwrap().close();
    }

    /// **不变量：编辑会话中的"未就绪"是暂缓，不是宿主冲突。**
    ///
    /// 【为什么这是回归测试】`ready` 的语义是"ARA 是否已重新交付可用输入"，会被会话内
    /// 的**真实内容变化**（几何、增删 region、源内容）按回 `false`。但 `ensure_loaded`
    /// 此前把"未就绪"一律判成宿主冲突，而插件自己的写回同样会触发这条通路 —— 于是用户
    /// 每做一次编辑都撞一串"尚未应用：Conflict: host changed"（自伤正反馈，与宿主是否
    /// 真的变了无关）。会话状态必须被单独识别：会话中 → 暂缓；非会话中 → 才是真冲突。
    #[test]
    fn a_mid_session_not_ready_window_is_deferred_instead_of_reported_as_a_conflict() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();

        // 会话内的一次真实内容变化：`ready` 被按回 false，而宿主尚未发 `endEditing`。
        document.ready.store(false, Ordering::Release);
        document.editing_session.store(true, Ordering::Release);
        document.revision.fetch_add(1, Ordering::AcqRel);
        let deferred = editor.ensure_loaded(false);
        assert!(
            deferred.is_ok(),
            "会话中的未就绪必须暂缓（等 endEditing 收敛），不得报成宿主冲突：{deferred:?}"
        );

        // 同一状态、但**不在**会话中 ⇒ 宿主真的没交付输入，仍是冲突，不能被暂缓语义吞掉。
        document.editing_session.store(false, Ordering::Release);
        let conflict = editor.ensure_loaded(false);
        assert!(
            conflict.is_err(),
            "非会话状态下的未就绪仍是真冲突，否则真正的宿主故障会被静默"
        );
        editor.close();
    }
    /// **宿主改完内容却不再收口时，窗口必须继续可用，而不是永久 Conflict。**
    ///
    /// 【为什么这是问题二/三的主回归】`clear_renderers` 把 `ready` 按回 false，
    /// 而它只在宿主发 `endEditing` / `notifyModelUpdates` 时才被 `remap_and_log`
    /// 置回真。后台恢复与工程加载的中段常常两者都不发 —— 旧行为会让之后每一次
    /// `ensure_loaded` 都返回 Conflict，窗口永久停在旧读数上。
    #[test]
    fn an_observed_change_without_end_editing_keeps_publishing() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();

        // 宿主内容变化（记录时刻），随后不发收口。
        document.clear_renderers();
        document.editing_session.store(false, Ordering::Release);
        document.revision.fetch_add(1, Ordering::AcqRel);
        let result = editor.ensure_loaded(false);
        assert!(
            result.is_ok(),
            "已观察到的内容变化必须继续发布最后一份时间线：{result:?}"
        );
        editor.close();
    }
    /// 同一已打开会话只改变GUI清单代次，读取仍采纳新增/删除；不依赖重开或定时器抢跑。
    #[test]
    fn inventory_only_read_refreshes_existing_editor_without_resetting_parameter_history() {
        let (model, owner, _) = fixture();
        let document = model.session();
        document.host_undo.pending.store(true, Ordering::Release);
        let editor = owner.editor_session().unwrap();
        // 暂停idle pump，使测试准确检查命令自身的缓存早退，而非碰巧被后台刷新。
        editor.queue.send(Job::Close).unwrap();
        editor
            .worker
            .lock_or_recover()
            .take()
            .unwrap()
            .join()
            .unwrap();
        let loaded =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        let root = loaded["tracks"][0]["id"].as_str().unwrap().to_owned();
        super::super::commands::dispatch(
            &editor,
            "set_param_frames",
            json!({"trackId":root,"param":"hifigan_tension",
            "startFrame":0,"values":[35.,45.],"checkpoint":true}),
        )
        .unwrap();
        let versions = document.editor_versions().unwrap();
        let history =
            super::super::commands::dispatch(&editor, "get_history_state", json!({})).unwrap();
        let selected = editor.timeline.lock_or_recover().selected_clip_id.clone();
        let host = crate::host::MockHost::new();
        host.enable_writer();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let host_api = Arc::new(host.client());
        let mut track = host_api.ui_track(&|| true).unwrap();
        track.id = "track".into();
        let item = track.items[0].clone();
        track.items.clear();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track.clone());
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        track.items.push(item.clone());
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track.clone());
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        let added =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        let id = format!("{}ara-item-{}", editor.namespace, item.geometry.item_id);
        assert!(
            added["clips"]
                .as_array()
                .unwrap()
                .iter()
                .any(|clip| clip["id"] == id),
            "当前UI应立刻看见新item"
        );
        assert_eq!(
            document.editor_versions().unwrap(),
            versions,
            "清单不伪造ARA/edit/scope代次"
        );
        assert_eq!(editor.timeline.lock_or_recover().selected_clip_id, selected);
        track.items.clear();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        document.ui_geometry_revision.fetch_add(1, Ordering::AcqRel);
        let removed =
            super::super::commands::dispatch(&editor, "get_timeline_state", json!({})).unwrap();
        assert!(!removed["clips"]
            .as_array()
            .unwrap()
            .iter()
            .any(|clip| clip["id"] == id));
        assert_eq!(
            super::super::commands::dispatch(&editor, "get_history_state", json!({})).unwrap(),
            history
        );
        let params = super::super::commands::dispatch(
            &editor,
            "get_param_frames",
            json!({"trackId":root,"param":"hifigan_tension",
            "startFrame":0,"frameCount":2,"binary":false}),
        )
        .unwrap();
        assert_eq!(params["edit"], json!([35., 45.]));
        document.close();
    }
    /// 已授权生成的波形在mute显示占位期间保留；take更换后不错误沿用旧图。
    #[test]
    fn muted_display_keeps_authorized_waveform_without_reentering_audio_assignment() {
        let (model, owner, _) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let host = crate::host::MockHost::new();
        host.enable_writer();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let host_api = Arc::new(host.client());
        let mut track = host_api.ui_track(&|| true).unwrap();
        let mut audio = editor.timeline.lock_or_recover().clips[0].clone();
        let path = audio.source_path.clone().unwrap();
        audio.id = format!(
            "{}ara-item-{}",
            editor.namespace, track.items[0].geometry.item_id
        );
        audio.track_id = format!("{}{}", editor.namespace, track.id);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track.clone());
        let mut visible = TimelineState::default();
        visible.tracks.clear();
        visible.clips = vec![audio];
        editor.retain_display_waveforms(&mut visible, false);
        track.items[0].geometry.muted = true;
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track.clone());
        visible.clips.clear();
        document.present_host_inventory(&mut visible, &editor.namespace);
        editor.retain_display_waveforms(&mut visible, true);
        assert!(visible.clips[0].muted);
        assert_eq!(visible.clips[0].source_path.as_ref(), Some(&path));
        assert!(editor.check_source(&path).is_ok());
        track.items[0].geometry.take_id = "different-take".into();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        visible.clips.clear();
        document.present_host_inventory(&mut visible, &editor.namespace);
        editor.retain_display_waveforms(&mut visible, true);
        assert!(visible.clips[0].source_path.is_none());
        document.close();
    }
    /// **ARA-only 全链路**：宿主只提供 ARA、没有原生扩展时，加载仍必须给出带媒体的时间线，
    /// 后端如实报告为只读的 `AraOnly`。
    ///
    /// 【为什么这是解耦的核心验收】此前"没有 REAPER"与"有 REAPER 但缺 API"无法区分，
    /// 退化形态一律是空时间线。这条测试用一个**没有 `ui_tracks`** 的文档（即 ARA-only
    /// 宿主的形状）驱动真实会话：加载 → 出时间线 → 片段带媒体身份（不是占位）→ 授权
    /// PCM 已物化（波形与渲染输入可用）。后端身份与能力位也必须如实反映"片段几何只读"。
    #[test]
    fn an_ara_only_host_loads_a_timeline_with_media_and_reports_read_only() {
        use hifishifter_kernel::editor::host::HostKind;
        let (model, owner, _identity) = fixture();
        let document = model.session();
        // 前提：ARA-only 宿主的形状就是"没有任何宿主清单"。
        assert!(document.ui_tracks.lock_or_recover().is_empty());
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();

        // 后端身份：没有原生适配器 ⇒ ARA-only，且片段几何只读。
        let backend = owner.host_backend();
        assert_eq!(backend.kind(), HostKind::AraOnly);
        assert!(backend.native().is_none());
        let caps = backend.capabilities();
        assert!(caps.clip_geometry, "ARA 直接给出 region 几何");
        assert!(!caps.clip_write, "ARA-only 后端不得宣称可写片段几何");

        // 时间线：clip 带媒体身份，不是"等待宿主提供音频"的占位。
        let timeline = editor.timeline.lock_or_recover().clone();
        assert_eq!(timeline.clips.len(), 1, "ARA-only 也必须出时间线");
        let source = timeline.clips[0]
            .source_path
            .clone()
            .expect("ARA 授权 PCM 必须给出媒体路径，而不是占位");
        // 插件把 ARA 授权 PCM 物化成会话本地的 wav，`source_path` 指向它。
        assert!(
            source.ends_with(".wav"),
            "媒体路径应指向物化后的本地 PCM：{source}"
        );
        // 波形与渲染输入共用同一份授权 PCM：能取到源即证明两者都可用。
        editor
            .check_source(&source)
            .expect("授权 PCM 必须已物化，波形/渲染输入才可用");
        document.close();
    }
    /// 真实宿主正向倍率不再被GUI挡住；take的源窗口和项目时长保留各自坐标。
    #[test]
    fn forward_time_stretch_loads_the_real_host_geometry_in_original_gui_state() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        {
            let mut timeline = document.timeline.lock_or_recover();
            let clip = &mut timeline.as_mut().unwrap().clips[0];
            clip.length_sec = 8.0 / 44100.0;
            clip.takes[0].playback_rate = 0.5;
            clip.normalize_takes();
        }
        {
            let mut regions = document.regions.lock_or_recover();
            for region in regions.values_mut() {
                region.duration_in_playback_time = 8.0 / 44100.0;
                region.is_timestretch_enabled = true;
            }
        }
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let timeline = editor.timeline.lock_or_recover();
        assert_eq!(timeline.clips[0].playback_rate, 0.5);
        assert_eq!(timeline.clips[0].length_sec, 8.0 / 44100.0);
        drop(timeline);
        editor.close();
    }
    /// 两根的源分析已缓存后切回另一素材，再改隐藏根；不调用reload或原线getter推动应用。
    #[test]
    fn cached_other_track_pitch_applies_after_selection_without_reload_or_parameter_polling() {
        let (model, owners, _ids) = task34_world_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "cached-other-track".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let roots = loaded["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|track| track["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let ca = loaded["clips"][0]["id"].as_str().unwrap();
        let began = Instant::now();
        loop {
            for root in &roots {
                call(
                    &editor,
                    &sink,
                    &rx,
                    2,
                    "get_param_frames",
                    json!({"trackId":root,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}),
                );
            }
            if roots.iter().all(|root| {
                editor.timeline.lock_or_recover().params_by_root_track[root]
                    .pitch_orig_key
                    .is_some()
            }) {
                break;
            }
            assert!(began.elapsed() < Duration::from_secs(20));
            std::thread::sleep(Duration::from_millis(10));
        }
        let wait = |target| {
            let began = Instant::now();
            while editor.applied.load(Ordering::Acquire) < target {
                assert!(
                    began.elapsed() < Duration::from_secs(10),
                    "{:?}",
                    editor.error.lock_or_recover()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        call(
            &editor,
            &sink,
            &rx,
            3,
            "set_param_frames",
            json!({"trackId":roots[1],"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":true}),
        );
        wait(1);
        call(&editor, &sink, &rx, 4, "select_clip", json!({"clipId":ca}));
        call(
            &editor,
            &sink,
            &rx,
            5,
            "set_param_frames",
            json!({"trackId":roots[1],"param":"pitch","startFrame":0,"values":vec![67.;400],"checkpoint":true}),
        );
        wait(2);
        let restored = owners[1].encode_state().unwrap();
        let accepted = document.edits.lock_or_recover().params["b"].pitch_edit[100];
        document.close();
        assert_eq!(accepted, 67.);
        assert!(!restored.is_empty());
    }
    /// GUI保持连续只读请求时，150ms到期的自动应用仍应先行，不能等FIFO空闲才渲染。
    #[test]
    fn automatic_apply_is_not_starved_by_continuous_read_only_requests() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "read-flood".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":loaded["tracks"][0]["id"],"volume":0.5}),
        );
        let running = Arc::new(AtomicBool::new(true));
        let flag = running.clone();
        let client = editor.clone();
        let view = sink.clone();
        let poller = std::thread::spawn(move || {
            while flag.load(Ordering::Acquire) {
                let _ = client.enqueue(UiRequest {
                    id: 99,
                    command: "get_ui_settings".into(),
                    args: json!({}),
                    sink: view.clone(),
                    link: None,
                });
                std::thread::yield_now();
            }
        });
        let began = Instant::now();
        while editor.applied.load(Ordering::Acquire) < 1 && began.elapsed() < Duration::from_secs(5)
        {
            rx.try_iter().for_each(drop);
            std::thread::sleep(Duration::from_millis(5));
        }
        let applied = editor.applied.load(Ordering::Acquire);
        running.store(false, Ordering::Release);
        poller.join().unwrap();
        document.close();
        assert!(applied >= 1, "不断只读轮询不能饿死已经到期的应用");
    }
    /// 实际合成尚占actor时，播放态查询不得等待编辑timeline锁；参数权威仍走原队列。
    #[test]
    fn playback_observation_does_not_wait_for_actor_timeline_or_render() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        document.clock.publish_host(3.125, true);
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(4);
        let sink = UiSink {
            view_id: "fast-transport".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let held = editor.timeline.lock_or_recover();
        editor
            .enqueue(UiRequest {
                id: 1,
                command: "get_playback_state".into(),
                args: json!({}),
                sink: sink.clone(),
                link: None,
            })
            .unwrap();
        let response = rx.recv_timeout(Duration::from_millis(200)).unwrap();
        drop(held);
        assert_eq!(response["value"]["position_sec"], 3.125);
        assert_eq!(response["value"]["is_playing"], true);
        document.close();
        assert!(editor
            .enqueue(UiRequest {
                id: 2,
                command: "get_playback_state".into(),
                args: json!({}),
                sink,
                link: None
            })
            .is_err());
    }
    /// 暂时渲染失败只影响就绪状态；宿主恢复后自动重试，不能要求用户手工reload。
    ///
    /// 【为什么不再用"内容淡化缺宿主几何"当失败源】那条路径已按 Task 4.1 改为**逐 clip
    /// 降级**（ARA-only 宿主上淡化不再让整条 renderer 失败）—— 它不再产生渲染错误，
    /// 因此这里改用真正**暂时**的失败源：宿主尚未就绪（`host PCM/model not ready`）。
    /// 这与本用例要钉的性质完全一致：失败只影响就绪读数，恢复后自动重试即可。
    #[test]
    fn automatic_apply_recovers_from_render_failure_without_reload() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "render-retry".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        // 宿主暂时未就绪 ⇒ 渲染暂时失败（局部/暂时，不是文档级故障）。
        document.ready.store(false, Ordering::Release);
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":loaded["tracks"][0]["id"],"volume":0.5}),
        );
        let began = Instant::now();
        while editor.render_error.lock_or_recover().is_none() {
            assert!(began.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(editor.error.lock_or_recover().is_none());
        assert_eq!(editor.applied.load(Ordering::Acquire), 0);
        // 宿主恢复就绪 ⇒ 自动重试必须自己回到可用状态，不需要 reload。
        document.ready.store(true, Ordering::Release);
        let began = Instant::now();
        while editor.applied.load(Ordering::Acquire) < 1 {
            assert!(
                began.elapsed() < Duration::from_secs(5),
                "{:?}",
                editor.state()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(editor.render_error.lock_or_recover().is_none());
        document.close();
    }
    /// 没有手绘pitch的HiFiGAN气声/张力/共振峰也**要求**原线；None与纯混音不要求。
    ///
    /// 【为什么断言的是渲染门禁而不是 `requires_analysis`】"这条配置要不要原线"是**静态**
    /// 事实（`does_clip_need_processor_render`），而"现在要不要等"是**动态**问题 ——
    /// 后者还要减去"已经证明产不出"的那些内容（见 `clip_pitch_analysis_can_be_produced`）。
    /// 用一个会随调度时机变化的读数去钉静态事实，测出来的只会是时序：夹具的 4 帧素材
    /// 本来就分析不出东西，它的内容键随时可能被标成"产不出"。所以这条按内核的静态判据
    /// 断言，动态那一半由 `a_root_whose_line_can_never_be_produced_stops_blocking_the_render`
    /// 在**内核侧**钉住。
    #[test]
    fn effect_only_analysis_gate_does_not_block_none_or_plain_mix() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let mut timeline = editor.timeline.lock_or_recover().clone();
        let root = timeline.tracks[0].id.clone();
        timeline.ensure_params_for_root(&root);
        timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::NsfHifiganOnnx;
        timeline.tracks[0].compose_enabled = true;
        let needs_render = |timeline: &TimelineState| {
            timeline.clips.iter().any(|clip| {
                hifishifter_kernel::pitch_editing::does_clip_need_processor_render(
                    timeline,
                    clip,
                    clip.start_sec,
                )
            })
        };
        for effect in ["breath_enabled", "hifigan_tension", "formant_shift_cents"] {
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.extra_params.clear();
            params.extra_curves.clear();
            if effect == "breath_enabled" {
                params.extra_params.insert(effect.into(), 1.);
            } else {
                params.extra_curves.insert(effect.into(), vec![100.; 200]);
                if effect == "hifigan_tension" {
                    params.extra_params.insert("breath_enabled".into(), 1.);
                }
            }
            params.pitch_orig_key = None;
            assert!(!params.pitch_edit_user_modified);
            assert!(
                needs_render(&timeline),
                "{effect}缺原线必须要求处理器渲染（否则效果会被静默丢掉）"
            );
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::None;
            assert!(
                !needs_render(&timeline),
                "算法为 None 时纯混音不该要求处理器"
            );
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::NsfHifiganOnnx;
        }
        let params = timeline.params_by_root_track.get_mut(&root).unwrap();
        params.extra_params.clear();
        params.extra_curves.clear();
        assert!(
            !needs_render(&timeline),
            "没有曲线也没有效果 ⇒ 不要求处理器"
        );
        assert!(
            !editor.requires_analysis(&timeline),
            "没有任何东西要求原线 ⇒ 绝不能把渲染判成 pending"
        );
        editor.close();
    }
    /// 三分钟真实授权PCM同进程Arc冻结，原GUI流式分析副本不受旧30秒门禁或整源复制影响。
    #[test]
    fn long_host_source_loads_original_gui_through_shared_pcm_without_ipc_copy() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let seconds = 180.;
        let frames = 44100 * 180;
        let budget = crate::render::budget::global_budget();
        let baseline = budget.used();
        let pcm = Arc::new(SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.125; frames]],
            version: 1,
            _reservation: Some(budget.reserve(frames * 4).unwrap()),
        });
        document
            .sources
            .lock_or_recover()
            .insert("ara://source".into(), pcm.clone());
        document
            .edit_sources
            .lock_or_recover()
            .insert("ara://source".into(), pcm.clone());
        {
            let mut timeline = document.timeline.lock_or_recover();
            let timeline = timeline.as_mut().unwrap();
            timeline.project_sec = seconds;
            for clip in &mut timeline.clips {
                clip.length_sec = seconds;
                clip.takes[0].source_end_sec = seconds;
                clip.normalize_takes();
            }
        }
        for region in document.regions.lock_or_recover().values_mut() {
            region.duration_in_modification_time = seconds;
            region.duration_in_playback_time = seconds;
        }
        let (snapshot, _, _) = document.workspace_snapshot().unwrap();
        assert!(Arc::ptr_eq(&snapshot.sources[0].1, &pcm));
        assert_eq!(
            budget.used(),
            baseline + frames * 4,
            "冻结不能复制一份整源PCM"
        );
        drop(snapshot);
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let timeline = editor.timeline.lock_or_recover();
        assert_eq!(timeline.clips[0].length_sec, seconds);
        assert_eq!(timeline.clips[0].duration_frames, Some(frames as u64));
        let path = timeline.clips[0].source_path.clone().unwrap();
        drop(timeline);
        use std::io::{Read, Seek};
        let mut wav = std::fs::File::open(path).unwrap();
        assert!(wav.metadata().unwrap().len() >= frames as u64 * 4);
        wav.seek(std::io::SeekFrom::End(-4)).unwrap();
        let mut tail = [0_u8; 4];
        wav.read_exact(&mut tail).unwrap();
        assert_eq!(f32::from_le_bytes(tail), 0.125);
        document.close();
        drop(pcm);
        assert_eq!(budget.used(), baseline);
    }
    /// 双轨各三分钟保留旧就绪快照时，修改另一轨仍能原子更新；双侧输出不能相加/串轨。
    #[test]
    fn two_long_tracks_update_atomically_with_old_ready_audio_inside_pcm_budget() {
        let (model, owners, ids) = workspace_fixture();
        let document = model.session();
        document.ready.store(false, Ordering::Release);
        let budget = crate::render::budget::global_budget();
        let baseline = budget.used();
        let frames = 44100 * 180;
        let seconds = 180.;
        let source = |value| {
            Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![vec![value; frames]],
                version: 1,
                _reservation: Some(budget.reserve(frames * 4).unwrap()),
            })
        };
        let a = source(0.25);
        let b = source(0.5);
        {
            let _transaction = document.transaction.lock_or_recover();
            let mut timeline = document.timeline.lock_or_recover();
            let timeline = timeline.as_mut().unwrap();
            timeline.project_sec = seconds;
            for (index, clip) in timeline.clips.iter_mut().enumerate() {
                clip.length_sec = seconds;
                clip.takes[0].source_end_sec = seconds;
                clip.takes[0].source_path = Some(
                    if index == 0 {
                        "ara://source"
                    } else {
                        "ara://source-b"
                    }
                    .into(),
                );
                clip.normalize_takes();
            }
            for (index, id) in ids.iter().enumerate() {
                let key = (&**id as *const u8) as u64;
                let mut regions = document.regions.lock_or_recover();
                let region = regions.get_mut(&key).unwrap();
                region.duration_in_modification_time = seconds;
                region.duration_in_playback_time = seconds;
                if index == 1 {
                    region.audio_source_persistent_id = "ara://source-b".into();
                }
            }
            document.track_bindings.lock_or_recover().insert(
                "b".into(),
                vec![("modification-b".into(), "ara://source-b".into())],
            );
            for sources in [&document.sources, &document.edit_sources] {
                let mut sources = sources.lock_or_recover();
                sources.insert("ara://source".into(), a.clone());
                sources.insert("ara://source-b".into(), b.clone());
            }
            document.revision.fetch_add(1, Ordering::AcqRel);
            document.render_epoch.fetch_add(1, Ordering::AcqRel);
            document.ready.store(true, Ordering::Release);
        }
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "long-multitrack".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let wait = |generation| {
            let began = Instant::now();
            loop {
                let pending = editor.state()["pending"].as_bool().unwrap();
                if !pending && editor.applied.load(Ordering::Acquire) >= generation {
                    break;
                }
                assert!(
                    began.elapsed() < Duration::from_secs(30),
                    "{:?}",
                    editor.state()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        wait(0);
        let check = |index: usize, rate: usize, value: f32| {
            let mut left = [9_f32; 513];
            let mut right = [9_f32; 513];
            let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut bus = crate::audio_abi::AudioBusBuffers {
                num_channels: 2,
                silence_flags: 0,
                channel_buffers: planes.as_mut_ptr(),
            };
            // SAFETY: 两个513帧平面存活；逐轨读取尾部511帧，剩余2帧必须补零。
            assert!(unsafe {
                owners[index].snapshots[usize::from(rate == 48000)].copy_block(
                    (rate * 180 - 511) as i64,
                    rate as u32,
                    &mut bus,
                    513,
                )
            });
            assert_eq!(&left[..511], &[value; 511]);
            assert_eq!(right, left);
            assert_eq!(&left[511..], &[0.; 2]);
        };
        for rate in [44100, 48000] {
            check(0, rate, 0.25);
            check(1, rate, 0.5);
        }
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_track_state",
            json!({"trackId":loaded["tracks"][1]["id"],"volume":0.5}),
        );
        wait(1);
        for rate in [44100, 48000] {
            check(0, rate, 0.25);
            check(1, rate, 0.25);
        }
        for owner in &owners {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
        }
        let ready_bytes = (44100 + 48000) * 180 * 4 * 2;
        let curve_bytes = document
            .edits
            .lock_or_recover()
            .atlas
            .accounted_curve_bytes();
        assert_eq!(
            budget.used() - baseline,
            frames * 4 * 2 + ready_bytes + curve_bytes
        );
        assert_eq!(editor.state()["error"], Value::Null);
        eprintln!("two-long-tracks seconds=180 tracks=2 ready_bytes={ready_bytes} curve_bytes={curve_bytes} accounted_peak={} hard_limit={} automatic_other_track=true",budget.peak(),budget.limit());
        document.close();
        drop(a);
        drop(b);
        for owner in &owners {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
        }
        assert_eq!(budget.used(), baseline);
    }
    /// 无本地落笔的宿主移动也要重新准备；cache全命中不会发ClipPitchReady，不能静默遗漏。
    #[test]
    fn host_move_without_local_generation_reapplies_and_reuses_analysis_file() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let before_path = editor.timeline.lock_or_recover().clips[0]
            .source_path
            .clone()
            .unwrap();
        let modified = std::fs::metadata(&before_path).unwrap().modified().unwrap();
        {
            let _transaction = document.transaction.lock_or_recover();
            let mut timeline = document.timeline.lock_or_recover();
            let timeline = timeline.as_mut().unwrap();
            timeline.clips[0].start_sec = 1.;
            timeline.project_sec = 2.;
            for region in document.regions.lock_or_recover().values_mut() {
                region.start_in_playback_time = 1.;
            }
            document.revision.fetch_add(1, Ordering::AcqRel);
            document.render_epoch.fetch_add(1, Ordering::AcqRel);
        }
        let mut left = [0_f32; 4];
        let mut right = [0_f32; 4];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        let began = Instant::now();
        loop {
            unsafe {
                owner.snapshots[0].copy_block(44100, 44100, &mut bus, 4);
            }
            if left == [0.1, 0.2, 0.3, 0.4] {
                break;
            }
            assert!(
                began.elapsed() < Duration::from_secs(5),
                "{:?}",
                editor.state()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let after_path = editor.timeline.lock_or_recover().clips[0]
            .source_path
            .clone()
            .unwrap();
        assert_eq!(after_path, before_path);
        assert_eq!(
            std::fs::metadata(after_path).unwrap().modified().unwrap(),
            modified
        );
        assert_eq!(editor.generation.load(Ordering::Acquire), 0);
        document.close();
    }
    /// PCM 晚到而 edit/model/scope 不变时，当前窗口仍补波形，保留选择和参数历史。
    #[test]
    fn late_host_pcm_refreshes_existing_editor_without_model_change() {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let root = editor.timeline.lock_or_recover().tracks[0].id.clone();
        super::super::commands::dispatch(
            &editor,
            "set_track_state",
            json!({"trackId":root,"volume":0.5}),
        )
        .unwrap();
        let versions = document.editor_versions().unwrap();
        let selected = editor.timeline.lock_or_recover().selected_clip_id.clone();
        let history = editor.history.lock_or_recover().position;
        assert!(history > 0);
        {
            let mut timeline = editor.timeline.lock_or_recover();
            timeline.clips[0].source_path = None;
            for take in &mut timeline.clips[0].takes {
                take.source_path = None;
            }
        }
        document.publish_source_pcm(
            "ara://source".into(),
            Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![vec![0.4, 0.3, 0.2, 0.1]],
                version: 0,
                _reservation: None,
            }),
        );
        assert_eq!(document.editor_versions().unwrap(), versions);
        editor.refresh_host();
        let timeline = editor.timeline.lock_or_recover();
        let clip = &timeline.clips[0];
        let path = clip
            .source_path
            .as_ref()
            .expect("late PCM must fill the visible clip");
        assert!(std::path::Path::new(path).is_file());
        assert_eq!(clip.takes[0].source_path.as_ref(), Some(path));
        assert_eq!(timeline.selected_clip_id, selected);
        assert_eq!(timeline.tracks[0].volume, 0.5);
        drop(timeline);
        assert_eq!(editor.history.lock_or_recover().position, history);
        document.close();
    }

    /// 同一ARA源ID换为不同PCM后，原线从57重建为69，保留目标60；不靠读取原线推动分析。
    #[test]
    fn same_host_source_identity_changed_pcm_invalidates_original_pitch_without_reload() {
        let (model, owners, _ids) = task34_world_fixture();
        let document = model.session();
        let editor = owners[0].editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "same-uri-change".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let loaded = call(&editor, &sink, &rx, 1, "get_timeline_state", json!({}));
        let root = loaded["tracks"][0]["id"].as_str().unwrap();
        let wait = |predicate: &dyn Fn() -> bool| {
            let began = Instant::now();
            while !predicate() {
                assert!(
                    began.elapsed() < Duration::from_secs(25),
                    "{:?}",
                    editor.state()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        wait(&|| {
            editor
                .timeline
                .lock_or_recover()
                .params_by_root_track
                .get(root)
                .is_some_and(|params| params.pitch_orig_key.is_some())
        });
        let old_path = editor.timeline.lock_or_recover().clips[0]
            .source_path
            .clone()
            .unwrap();
        assert!(
            (editor.timeline.lock_or_recover().params_by_root_track[root].pitch_orig[100] - 57.)
                .abs()
                < 1.
        );
        call(
            &editor,
            &sink,
            &rx,
            2,
            "set_param_frames",
            json!({"trackId":root,"param":"pitch","startFrame":0,"values":vec![60.;400],"checkpoint":true}),
        );
        wait(&|| editor.applied.load(Ordering::Acquire) >= 1);
        let samples = (0..88200)
            .map(|n| {
                let phase = 2. * std::f64::consts::PI * 440. * n as f64 / 44100.;
                (1..=16)
                    .map(|harmonic| (phase * harmonic as f64).sin() * 0.2 / harmonic as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        {
            let _transaction = document.transaction.lock_or_recover();
            let pcm = Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![samples],
                version: 1,
                _reservation: None,
            });
            document
                .edit_sources
                .lock_or_recover()
                .insert("ara://source".into(), pcm.clone());
            document
                .sources
                .lock_or_recover()
                .insert("ara://source".into(), pcm);
            document.revision.fetch_add(1, Ordering::AcqRel);
            document.render_epoch.fetch_add(1, Ordering::AcqRel);
        }
        wait(&|| {
            let timeline = editor.timeline.lock_or_recover();
            timeline
                .params_by_root_track
                .get(root)
                .is_some_and(|params| {
                    params.pitch_orig_key.is_some()
                        && params
                            .pitch_orig
                            .get(100)
                            .is_some_and(|note| (*note - 69.).abs() < 1.)
                })
        });
        wait(&|| {
            document
                .edits
                .lock_or_recover()
                .params
                .get("track")
                .is_some_and(|params| {
                    params
                        .pitch_orig
                        .get(100)
                        .is_some_and(|note| (*note - 69.).abs() < 1.)
                })
        });
        let timeline = editor.timeline.lock_or_recover();
        assert_ne!(
            timeline.clips[0].source_path.as_deref(),
            Some(old_path.as_str())
        );
        assert_eq!(timeline.params_by_root_track[root].pitch_edit[100], 60.);
        drop(timeline);
        document.close();
    }
    /// 真actor的自动调度消费快速写入/undo/redo，不调用手工render函数冒充自动应用。
    #[test]
    fn automatic_audio_uses_latest_edit_and_follows_undo_redo_without_ui() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "audio-view".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_track_state",
            json!({"trackId":track,"volume":0.25}),
        );
        call(
            &editor,
            &sink,
            &receiver,
            3,
            "set_track_state",
            json!({"trackId":track,"volume":0.5}),
        );
        let wait = |expected: u64| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while editor.applied.load(Ordering::Acquire) < expected {
                assert!(
                    Instant::now() < deadline,
                    "automatic apply failed: {:?}",
                    editor.error.lock_or_recover()
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let output = || {
            let mut left = [9_f32; 4];
            let mut right = [9_f32; 4];
            let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
            let mut bus = crate::audio_abi::AudioBusBuffers {
                num_channels: 2,
                silence_flags: 0,
                channel_buffers: planes.as_mut_ptr(),
            };
            // SAFETY: 两个四帧可写平面完整覆盖copy_block，owner与snapshot仍存活。
            assert!(unsafe { owner.snapshots[0].copy_block(0, 44100, &mut bus, 4) });
            (left, right)
        };
        wait(2);
        for plane in [output().0, output().1] {
            for (actual, want) in plane.into_iter().zip([0.05, 0.1, 0.15, 0.2]) {
                assert!((actual - want).abs() < 1e-6);
            }
        }
        call(&editor, &sink, &receiver, 4, "undo_timeline", json!({}));
        wait(3);
        for (actual, want) in output().0.into_iter().zip([0.025, 0.05, 0.075, 0.1]) {
            assert!((actual - want).abs() < 1e-6);
        }
        call(&editor, &sink, &receiver, 5, "redo_timeline", json!({}));
        wait(4);
        for (actual, want) in output().0.into_iter().zip([0.05, 0.1, 0.15, 0.2]) {
            assert!((actual - want).abs() < 1e-6);
        }
        sink.closed.store(true, Ordering::Release);
        editor
            .enqueue(UiRequest {
                id: 6,
                command: "set_track_state".into(),
                args: json!({"trackId":track,"muted":true}),
                sink,
                link: None,
            })
            .unwrap();
        wait(5);
        assert_eq!(
            output().0,
            [0.; 4],
            "关UI取消响应，不能取消已排参数或后台供音"
        );
        editor.close();
    }
    /// WORLD自动作业实际改变音高；频率从输出PCM独立自相关估算，不读参数分析结果。
    #[test]
    fn automatic_world_pitch_changes_real_pcm_without_manual_submit() {
        world_pitch_case(true);
    }
    /// 分析前落笔也需自动收敛，不依赖GUI不停请求get_param_frames才能供修音音频。
    #[test]
    fn early_pitch_waits_for_analysis_and_applies_without_ui_polling() {
        world_pitch_case(false);
    }
    fn world_pitch_case(wait_for_analysis: bool) {
        let (model, owner, _identity) = fixture();
        let document = model.session();
        {
            let mut known = document.timeline.lock_or_recover();
            let timeline = known.as_mut().unwrap();
            timeline.project_sec = 2.;
            timeline.tracks[0].compose_enabled = true;
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            timeline.clips[0].length_sec = 2.;
            timeline.clips[0].takes[0].source_end_sec = 2.;
            timeline.clips[0].normalize_takes();
        }
        for region in document.regions.lock_or_recover().values_mut() {
            region.duration_in_modification_time = 2.;
            region.duration_in_playback_time = 2.;
        }
        let samples: Vec<f32> = (0..88200)
            .map(|n| {
                let phase = 2. * std::f64::consts::PI * 220. * n as f64 / 44100.;
                // 与既有WORLD oracle相同的有声谐波结构；三谐波被Harvest判为清音。
                (1..=16)
                    .map(|harmonic| (phase * harmonic as f64).sin() * 0.2 / harmonic as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        document.edit_sources.lock_or_recover().insert(
            "ara://source".into(),
            Arc::new(SourcePcm {
                covered: Vec::new(),
                sample_rate: 44100,
                planes: vec![samples],
                version: 0,
                _reservation: None,
            }),
        );
        let editor = owner.editor_session().unwrap();
        let (reply, receiver) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "world-view".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let timeline = call(
            &editor,
            &sink,
            &receiver,
            1,
            "get_timeline_state",
            json!({}),
        );
        assert_eq!(timeline["clips"].as_array().unwrap().len(), 1);
        {
            let live = editor.timeline.lock_or_recover();
            assert!(live.tracks[0].compose_enabled);
            assert!(!live.clips[0].muted);
            assert_eq!(live.clips[0].track_id, live.tracks[0].id);
            assert!(std::path::Path::new(live.clips[0].source_path.as_ref().unwrap()).is_file());
        }
        let track = timeline["tracks"][0]["id"].as_str().unwrap();
        let analysis_deadline = Instant::now() + Duration::from_secs(15);
        while wait_for_analysis {
            let frames = call(
                &editor,
                &sink,
                &receiver,
                10,
                "get_param_frames",
                json!({"trackId":track,"param":"pitch","startFrame":0,"frameCount":400,"binary":false}),
            );
            if frames["orig"]
                .as_array()
                .is_some_and(|orig| orig.iter().filter_map(Value::as_f64).any(|p| p > 30.))
            {
                break;
            }
            assert!(
                Instant::now() < analysis_deadline,
                "原GUI真实get_param_frames分析未完成: {frames}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        call(
            &editor,
            &sink,
            &receiver,
            2,
            "set_param_frames",
            json!({"trackId":track,"param":"pitch","startFrame":0,"values":vec![64.;400],"checkpoint":true}),
        );
        {
            let accepted = owner.edit_state();
            let accepted = accepted.lock_or_recover();
            assert!(
                accepted.params["track"].pitch_edit_user_modified,
                "参数写入必须保留用户修改标记"
            );
            assert!(
                matches!(
                    accepted.tracks[0].pitch_analysis_algo,
                    PitchAnalysisAlgo::WorldDll
                ),
                "接受后的算法必须WORLD"
            );
            assert_eq!(accepted.params["track"].pitch_edit[100], 64.);
            if wait_for_analysis {
                assert!(
                    accepted.params["track"].pitch_orig[100] > 30.,
                    "自动渲染的原线必须分析完成"
                );
            }
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        while editor.applied.load(Ordering::Acquire) < 1 {
            assert!(
                Instant::now() < deadline,
                "WORLD automatic apply failed: {:?}",
                editor.error.lock_or_recover()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut left = vec![0_f32; 8820];
        let mut right = vec![0_f32; 8820];
        let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut bus = crate::audio_abi::AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        // SAFETY: 两个8820帧平面存活，读取实际已经由actor发布的0.5秒起输出。
        assert!(unsafe { owner.snapshots[0].copy_block(22050, 44100, &mut bus, 8820) });
        let correlation = |lag: usize| {
            let mut cross = 0_f64;
            let mut first = 0_f64;
            let mut second = 0_f64;
            for i in 0..left.len() - lag {
                let a = left[i] as f64;
                let b = left[i + lag] as f64;
                cross += a * b;
                first += a * a;
                second += b * b;
            }
            cross / (first * second).sqrt().max(1e-20)
        };
        let lag = (88..=245)
            .max_by(|a, b| correlation(*a).total_cmp(&correlation(*b)))
            .unwrap();
        let frequency = 44100. / lag as f64;
        eprintln!("automatic WORLD output frequency={frequency:.3}Hz target=329.63Hz");
        assert!(
            (frequency - 329.63).abs() < 8.,
            "MIDI64应约329.63Hz，实际{frequency}Hz；不能只改变gain"
        );
        editor.close();
    }
    /// 缺少原声分析时不能把用户的pitch标成已应用；曲线权威仍可保存。
    #[test]
    fn pending_pitch_analysis_does_not_publish_raw_audio_as_applied() {
        let (_model, owner, _identity) = fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        {
            let mut timeline = editor.timeline.lock_or_recover();
            timeline.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            timeline.tracks[0].compose_enabled = true;
            let root = timeline.tracks[0].id.clone();
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.pitch_edit_user_modified = true;
            params.pitch_edit = vec![64.; 200];
            params.pitch_orig = vec![0.; 200];
            params.pitch_orig_key = None;
            editor.publish_timeline(timeline.clone());
        }
        let outcome = editor.apply(0, None);
        assert!(outcome.is_err(), "未分析的全零原线不能渲染原声并假报成功");
        // 【为什么把消息打进断言】"是错误"与"是**哪一种**错误"是两件事：前者可能来自
        // 任何一条守卫，只有后者才说明"渲染门禁在等分析"这条契约仍然成立。
        let error = outcome.unwrap_err();
        assert!(error.contains("analysis"), "unexpected error: {error}");
        let bytes = owner.encode_state().unwrap();
        let saved: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(saved["edits"]["params"]["track"]["pitch_edit"][0], 64.);
        editor.close();
    }
}

#[cfg(test)]
mod host_selection_tests {
    use super::*;

    /// **宿主的选择只在"确实变了"时覆盖插件选区。**
    ///
    /// 【为什么这条最要紧】`notifySelection` 是**通知**不是**设定**
    /// （`ARAInterface.h:3646-3655`）：宿主会把同一个选择反复发（重开窗口、切换视图），
    /// 若无条件采纳，用户在插件里刚点的那条每次都会被抢回去 —— 插件自己的 UI 就废了。
    /// 反之，宿主的**新**选择必须被采纳：那正是"在 Cubase 里选中一条 clip，插件跟着选中"
    /// 这条功能的全部内容。
    #[test]
    fn only_a_changed_host_selection_overrides_the_plugin_selection() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        let mut timeline = TimelineState::default();

        // 宿主第一次报 A ⇒ 采纳。
        editor.adopt_host_selection(&mut timeline, Some("A".into()));
        assert_eq!(timeline.selected_clip_id.as_deref(), Some("A"));

        // 用户在插件里改选了 B。
        timeline.selected_clip_id = Some("B".into());

        // 宿主**重复**报 A ⇒ 不得抢回（还是 B）。
        editor.adopt_host_selection(&mut timeline, Some("A".into()));
        assert_eq!(
            timeline.selected_clip_id.as_deref(),
            Some("B"),
            "宿主重复同一个选择不得覆盖用户在插件里的选择"
        );

        // 宿主报一个新的选择 C ⇒ 采纳。
        editor.adopt_host_selection(&mut timeline, Some("C".into()));
        assert_eq!(timeline.selected_clip_id.as_deref(), Some("C"));

        // 宿主报"没有任何选择"（空选择合法）⇒ 不覆盖，保持插件自己的选择。
        timeline.selected_clip_id = Some("D".into());
        editor.adopt_host_selection(&mut timeline, None);
        assert_eq!(
            timeline.selected_clip_id.as_deref(),
            Some("D"),
            "宿主清空选择不得连带清空插件里的选择"
        );

        drop(model);
        editor.close();
    }
}
