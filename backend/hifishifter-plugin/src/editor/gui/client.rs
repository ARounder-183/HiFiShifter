//! **Policy 层**：编辑器视图的全部判定，一行平台代码都没有。
//!
//! 这里原来是 `editor/webview.rs` 里除了建窗口之外的所有东西：在途预算、慢车道、
//! 撤销门、几何回执、宿主回复重投、让路状态机、分组 id、`close_track_gaps` /
//! `rename_clip_take` 改写、以及那张 600 行的命令分发表。三份拷贝必然漂移，
//! 所以它只有一份，三个平台共用。
//!
//! 【为什么是「自由函数 + `Rc<RefCell<EditorClient>>`」而不是「`&mut self` 方法」】
//! 因为**宿主回调会重入**：`deliver` 里的 `history_jump` 是宿主回调，它可能同步回到
//! 本模块改 `host_replies`。原实现用 `Rc<RefCell<..>>` + "调用宿主前先释放借用" 的
//! 纪律处理这件事，那段纪律是**二十多处的显式 `drop`/作用域**（每处都有注释说明为什么）。
//! 改成 `&mut self` 会把它们全部推翻重写 —— 那是纯风险，没有收益。
//!
//! 【为什么 `port` 是 `Rc<dyn EditorPort>` 而不是引用】端口在窗口创建之后才存在
//! （WebView2 的控制器是异步的），而 `EditorClient` 早于它构造。

use super::port::{EditorPort, NativeWindow};
use super::protocol;
use crate::editor::host_key_relay;
use crate::sync::MutexExt;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

/// 在途请求上限。
const PENDING_BUDGET: usize = 32;
/// 在途请求的存活时间。超过它仍未收到回复的请求视为"永远不会回"。
///
/// 【为什么是 30 秒】与各处 `HostReply::until` 的等待窗口一致：正常的宿主回流
/// （一次 `get_timeline_state` + ARA 重认领）在数百毫秒内完成；30 秒仍无回复只可能是
/// 请求已被丢弃。超时后剔除该条目，避免永久占用预算。
const PENDING_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// 慢车道请求的存活预算。
///
/// 【为什么不是 `PENDING_TTL`】那 30 秒回答的是"一次正常的宿主回流多久没回就该判丢"。
/// 慢车道的任务以分钟计（基准测试实测 31–134 秒），用同一个数字只会把"正在跑"判成
/// "回执丢了"，并打出那条误导性的 `expired without a reply`。
///
/// 【为什么是 30 分钟而不是"永不过期"】它只是**绊线**：前端 15 分钟就放弃了，这里留
/// 双倍余量，只为不让一个真正卡住的条目永久占着 map。永远不过期等于把"卡死"变成
/// "静默占位"，那是更难查的一类缺陷。
const SLOW_PENDING_TTL: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// 单个编辑请求最多重投多少次（等待宿主几何/媒体追上）。
///
/// 【为什么必须有上限】回复泵由 20 ms 定时器驱动。没有上限时，一个**永不收敛**的
/// 等待会以 50 Hz 空转到 `until`（30 s ≈ 1500 次 `get_timeline_state`），每次都争
/// 同一把文档事务。有了预算，失败在几秒内转成明确的终态错误，用户可以据此处置。
const HOST_REPLY_RETRY_BUDGET: u32 = 24;

/// 第 `attempt` 次重投前的等待：20 ms 起指数退避，上限 500 ms。
///
/// 【为什么起点仍是 20 ms】宿主追上几何通常只要一两帧；第一档保持与定时器同频，
/// 不牺牲正常情况的响应速度，只把"长期不收敛"从 50 Hz 压到 2 Hz。
fn host_reply_retry_backoff(attempt: u32) -> std::time::Duration {
    // attempt 从 1 起算（第一次重投前），所以第 1 档就是 20 ms。
    let shift = attempt.saturating_sub(1).min(5);
    let millis = 20_u64.saturating_mul(1_u64 << shift).min(500);
    std::time::Duration::from_millis(millis)
}

/// 把前端上报的错误文本折成一个**稳定、有界**的限流键片段。
///
/// 【为什么截断】键里带的是运行期数据（方法名、错误摘要、栈）。不截断就等于把
/// 整条栈当成键 —— 而栈里往往含地址/行号这类每次都变的内容，会让"同一类错误"
/// 每次都落进新桶，限流器形同虚设。64 字符足以区分 invoke 方法名与错误分类，
/// 又不会让键本身变成新的日志体量。
fn truncate_key(text: &str) -> String {
    const MAX: usize = 64;
    if text.len() <= MAX {
        return text.to_owned();
    }
    // 按字符边界截断，避免在多字节 UTF-8 中间切开造成 panic。
    let mut end = MAX;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

/// 一组 clip id → 宿主 `I_GROUPID`（`1..=host_edit::HOST_GROUP_ID_MAX`）。
///
/// 【为什么是哈希而不是"显式序号"】序号要从当前编组集合取 `max+1`，而本处理器在
/// UI 线程读到的 `editor.timeline` 未必已含刚提交的上一次编组 ⇒ 两次快速编组会取到
/// 同一个序号，反而把两组并成一组。哈希只依赖**本次选择**：同一选择恒等（幂等），
/// 不受提交时序影响。
///
/// 【为什么必须折进区间】旧实现 `raw & 0x7fff_ffff` 会给出到 `2^31-1` 的值，而宿主
/// 只收 `0..=HOST_GROUP_ID_MAX`：约 7% 的编组会撞上 "invalid host item group id"，
/// 整条 `set_clips_state_bulk` 被拒。取模后用到全部 32 位熵，同时保证落在区间内。
fn host_group_id_for(clip_ids: &[String]) -> i32 {
    let joined = clip_ids.join("\n");
    let digest = blake3::hash(joined.as_bytes());
    let mut bytes = [0_u8; 4];
    bytes.copy_from_slice(&digest.as_bytes()[..4]);
    (u32::from_le_bytes(bytes) % crate::host::host_edit::HOST_GROUP_ID_MAX as u32 + 1) as i32
}

/// 一次模态交互（文件对话框）前后的视图身份。
///
/// 两个分量，缺一不可：
/// * **owner 指针** —— 每个组件一个 owner。它回答"对话框开着的时候用户换了实例 / 关了窗吗"。
/// * **文档租约代次**（只有 ARA 形态有）—— 它回答"还属于同一份文档吗"。同一 owner 上换了
///   ARA 文档时 owner 指针不变，光看它会让对话框选出的路径落到新文档上。
///
/// 【为什么不能只用 `authorize`】它需要一份文档，而 Capture 形态**永远没有**文档 ——
/// 那些对话框（文件夹选择、另存为、打开 MIDI、导出布局/主题/记事本）与"哪个文档"无关，
/// 却会因为一个与它们无关的前置条件整块不可用。
type InteractionIdentity = (usize, Option<u64>);

fn interaction_identity(
    link: &crate::editor::routing::EditorLink,
) -> Result<InteractionIdentity, String> {
    let owner = link.owner()?;
    let lease = match owner.editor_document() {
        Ok(document) => Some(link.authorize(&document)?),
        // Capture：没有文档可租；身份退化成 owner 本身。
        Err(_) => None,
    };
    Ok((Arc::as_ptr(&owner) as usize, lease))
}

/// 与 [`interaction_identity`] 配对：身份没变即这次交互仍然有效。
fn interaction_unchanged(
    link: &crate::editor::routing::EditorLink,
    before: &InteractionIdentity,
) -> bool {
    interaction_identity(link).is_ok_and(|now| now == *before)
}
/// 一条视图的全部判定状态。
///
/// 【与旧 `BrowserState` 的差别只有四处】`hwnd` 变成 `window`（`usize`，跨平台句柄）、
/// `controller`/`error_window` 移进平台端口、多了就绪门（`ready` + `queue`）、
/// 多了"投递连续失败 ⇒ 自毁"的计数器（`post_failures` / `dead`）。其余字段逐字未改。
pub(crate) struct EditorClient {
    /// 本视图身份。回执与事件都带它，前端据此丢弃属于别的实例的消息。
    pub(crate) view_id: String,
    /// 宿主路由（实例 / 文档租约 / 私有状态）。
    pub(crate) link: Arc<crate::editor::routing::EditorLink>,
    /// 回执与事件的唯一出口（actor 与慢车道往这里写）。
    pub(crate) sink: crate::editor::session::UiSink,
    replies: mpsc::Receiver<serde_json::Value>,
    events: mpsc::Receiver<serde_json::Value>,
    /// 视图是否已关闭。关闭后一切入站消息与出站投递都是无操作。
    pub(crate) closed: bool,
    /// 本视图的平台句柄，原样交给文件对话框与让路原语。
    pub(crate) window: NativeWindow,
    /// 出站端口。`None` = 平台还没把原生视图接上。
    port: Option<Rc<dyn EditorPort>>,
    /// 页面是否已就绪（导航完成）。未就绪时出站消息只入队。
    ready: bool,
    /// 视图建立时刻，只用来算 `[gui] page ready after {n}ms`。
    opened_at: std::time::Instant,
    /// 就绪前的待发队列（见 `4.3`）。
    queue: std::collections::VecDeque<String>,
    /// 连续投递失败次数；页面进程死掉后它只增不减，是"视图该自毁"的唯一判据。
    post_failures: u32,
    /// 已经判定为死视图（投递连续失败或平台报告进程崩溃）。
    dead: bool,
    /// 是否已经就"页面一直没就绪"报过一次。只为不把同一条错误重复刷进界面。
    reported_not_ready: bool,
    /// 在途请求 id → 它的插入时刻与预算。
    ///
    /// 【为什么带时刻】`pending` 只在**收到回复**时移除（`pump` 的 `retain_mut`）。
    /// 若某个请求永远不回（文档关闭、投递失败、actor 丢弃），条目会永久残留；累积到
    /// 32（各处的预算上限）后，本视图会**拒绝一切后续命令**。
    /// 带时刻就能在定时器里超时剔除，把"永久失效"降级成"这一次超时"。
    ///
    /// 【为什么预算跟着请求走】见 `SLOW_PENDING_TTL`：慢车道的任务是分钟级的，
    /// 用 30 秒去判它"没有回复"只会把"正在跑"误判成"丢了"。
    pending: std::collections::HashMap<u64, Pending>,
    /// 已按下未松开的共享编辑键（去重用）。
    ///
    /// 【为什么 Linux 上是死字段】那里键盘事件直达 WebView，宿主根本不会把 Ctrl+C
    /// 交给 §IPlugView::keyDown§，于是 §forward_edit_key§ 永远不会被调用。
    #[cfg_attr(all(unix, not(target_os = "macos")), allow(dead_code))]
    forwarded_keys: HashSet<char>,
    host_replies: std::collections::HashMap<u64, HostReply>,
    undo_requests:
        std::collections::HashMap<u64, std::sync::Weak<crate::render::document::DocumentSession>>,
    undo_document: Option<std::sync::Weak<crate::render::document::DocumentSession>>,
    history_requests: HashSet<u64>,
    // 首位是连接绑定标识（route token + 重绑代数，见 `EditorLink::route_identity`），
    // 不是 `Arc::as_ptr`：地址回收复用后旧指针可能命中陈旧缓存（撤销历史来自
    // 上一份文档）。
    history_cache: Option<((String, u64, i32, i32), serde_json::Value)>,
    last_history: Option<serde_json::Value>,
    /// 宿主专属快捷键的让路状态机（见 `editor::host_key_relay`）。
    relay: host_key_relay::Arbiter,
    /// 本平台的"把键交回宿主"原语。自报 `Fidelity`，状态机据此决定要不要探针/提示。
    escape: Box<dyn host_key_relay::HostKeyEscape>,
    /// 前端声明的"属于宿主"的按键集合（`host_key_escape_set` 的产物）。
    escape_set: host_key_relay::StrokeSet,
    /// 待发给前端的**非回执**事件（让路没成的回报）。
    ///
    /// 【为什么与 `replies` 分开】回执按 id 对应一次 `invoke`；这里的事件没有 id，
    /// 是"后台发生了一件事"。共用一条通道会逼前端去猜"这条到底对应哪个 promise"。
    view_events: std::collections::VecDeque<serde_json::Value>,
}

/// 出站消息的字节上限（单条）。
///
/// 【为什么与入站的 1 MiB 不同】入站是"页面能发多大"，出站是"我们可能发多大" ——
/// 时间线快照是整条工程的 JSON，比任何请求都大。超过它就只能如实回一个错误，
/// 而不是让宿主 UI 线程去序列化一个几十兆的对象。
const RESPONSE_BUDGET: usize = 8 * 1024 * 1024;

/// 就绪前最多攒多少条出站消息。
///
/// 【为什么是 256】正常启动会在就绪前攒下十几条（bootstrap、历史、时间线、若干能力
/// 探活）；256 是它的十几倍，超过就说明页面根本没起来 —— 那时保留最旧的反而更糟，
/// 因为用户最需要看的是**最近**的状态。
const QUEUE_BUDGET: usize = 256;

/// 连续投递失败多少次就认定视图已死。
///
/// 【为什么必须有这条】WebKit 的 web process 崩溃之后，所有 `evaluateJavaScript` 都会
/// 失败，而界面此时已经是一张死图；继续投递只会让 `pending` 涨到预算上限、把后续
/// 命令全部拒掉 —— 那是"看起来能用、其实全废"的形态。
const POST_FAILURE_LIMIT: u32 = 8;

/// 在途预算判据（纯函数：`EditorClient` 要 COM 才建得起来，判据本身必须可测）。
fn budget_exhausted(in_flight: usize, id_present: bool, slow: bool) -> bool {
    !slow && (in_flight >= PENDING_BUDGET || id_present)
}

/// 一个在途请求的存活预算。
struct Pending {
    since: std::time::Instant,
    /// 是否跑在慢车道上（见 [`crate::editor::slow_lane`]）。
    slow: bool,
}

impl Pending {
    fn ttl(&self) -> std::time::Duration {
        if self.slow {
            SLOW_PENDING_TTL
        } else {
            PENDING_TTL
        }
    }
}

impl EditorClient {
    /// 记录一个在途请求。
    fn mark_pending(&mut self, id: u64, slow: bool) {
        self.pending.insert(
            id,
            Pending {
                since: std::time::Instant::now(),
                slow,
            },
        );
    }

    /// 在途请求是否已满（或该 id 已在途）。
    ///
    /// 先剔除超时条目，再判预算 —— 否则一个永不回复的请求会永久占用一个名额。
    ///
    /// 【为什么慢车道不占名额】那 32 个名额护的是"宿主写入的回执没回来"：每条都持有
    /// 一份宿主 Undo 块。慢任务不持有任何宿主写入，而它跑着的时候编辑器还在轮询 ——
    /// 让轮询把它挤出预算，正是要修的那个故障。
    fn pending_full(&mut self, id: u64, slow: bool) -> bool {
        self.expire_pending();
        budget_exhausted(self.pending.len(), self.pending.contains_key(&id), slow)
    }

    /// 剔除超过 [`PENDING_TTL`] 仍无回复的请求，并同步释放它占用的关联状态。
    ///
    /// 只删 `pending` 条目还不够：`host_replies` / `undo_requests` / `history_requests`
    /// 都以同一 id 为键，留着会泄漏内存，也会让宿主 Undo 块一直开着（`finish_request`
    /// 收不到）。所以这里把它们一并清掉。
    fn expire_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let now = std::time::Instant::now();
        let expired: Vec<(u64, bool)> = self
            .pending
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.since) >= entry.ttl())
            .map(|(id, entry)| (*id, entry.slow))
            .collect();
        if expired.is_empty() {
            return;
        }
        let view_id = self.view_id.clone();
        for (id, slow) in &expired {
            self.pending.remove(id);
            let reply = self.host_replies.remove(id);
            let document = reply
                .as_ref()
                .and_then(|reply| reply.document.upgrade())
                .or_else(|| self.undo_document.as_ref().and_then(|weak| weak.upgrade()));
            let undo = self
                .undo_requests
                .remove(id)
                .and_then(|weak| weak.upgrade());
            let history = self.history_requests.remove(id);
            if let Some(document) = undo {
                document.host_undo.finish_request(&view_id, *id);
            }
            if history {
                if let Some(document) = &document {
                    document.host_undo.finish_history(&view_id, *id);
                }
            }
            if *slow {
                // 慢车道的条目只在"真的卡死"时才该被清掉：前端 15 分钟就放弃了，
                // 这条绊线只为不让一个卡住的条目永久占着 map。措辞必须与"丢了回执"
                // 分开，否则排查时会把"还在跑"读成"回执没到"。
                log::warn!(
                    "[ara] slow request {id} still running after {:?} — the long task is stuck",
                    SLOW_PENDING_TTL
                );
            } else {
                log::warn!("[ara] editor request {id} expired without a reply");
            }
        }
    }
}
#[derive(Clone, Copy)]
enum HistoryJump {
    Undo,
    Redo,
    Position(i32),
}
#[derive(Clone)]
struct MediaAction {
    command: String,
    input: serde_json::Value,
}
/// 宿主写完后只重试读取ARA回流，不重复setter；保存最初的document/route租约。
struct HostReply {
    editor: Arc<crate::editor::session::EditorSession>,
    document: std::sync::Weak<crate::render::document::DocumentSession>,
    lease: u64,
    until: std::time::Instant,
    /// 已重投次数与下一次允许重投的时刻。
    ///
    /// 【为什么需要退避】回复泵由 20 ms `WM_TIMER` 驱动，此前只要判据仍成立就**每一 tick
    /// 重投一次**（50 Hz，直到 30 s 的 `until`）。宿主追上几何通常只需一两帧，这种
    /// 线性热重投让"等待收敛"变成 1500 次无意义的 `get_timeline_state`，与渲染线程争
    /// 同一把文档事务。改为指数退避（20→40→80…≤500 ms）并设总次数上限，超限即走明确的
    /// 终态错误，而不是继续空转。
    attempts: u32,
    next_attempt: std::time::Instant,
    jump: Option<HistoryJump>,
    imported: Option<String>,
    geometry: Option<crate::host::host_edit::HostEditReceipt>,
    split: Option<crate::host::host_split::SplitReceipt>,
    media: Option<crate::host::clipboard::MediaReceipt>,
    action: Option<MediaAction>,
}

impl EditorClient {
    /// 建一个视图的判定状态，并把它交给调用方的 `Rc`。
    ///
    /// 【为什么在这里建 sink 与通道】`view_id` 是"回执属于谁"的唯一凭据，它必须在
    /// 通道建立的那一刻就定下来；让平台各建一次，就是三份 `format!("view-{}-{token}")`。
    pub(crate) fn open(link: Arc<crate::editor::routing::EditorLink>) -> Rc<RefCell<Self>> {
        static NEXT_VIEW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let token = NEXT_VIEW.fetch_add(1, Ordering::Relaxed);
        let view_id = format!("view-{}-{token}", std::process::id());
        let (reply, replies) = mpsc::channel();
        let (event_sender, events) = mpsc::sync_channel(128);
        let sink = crate::editor::session::UiSink {
            view_id: view_id.clone(),
            reply,
            events: event_sender,
            closed: Arc::new(AtomicBool::new(false)),
        };
        // 原语先建：状态机要知道本平台能做什么（`Fidelity`）才能决定要不要探针。
        let escape = host_key_relay::escape_for_platform();
        let relay = host_key_relay::Arbiter::new(escape.fidelity());
        Rc::new(RefCell::new(Self {
            view_id,
            link,
            sink,
            replies,
            events,
            closed: false,
            window: 0,
            port: None,
            ready: false,
            opened_at: std::time::Instant::now(),
            queue: std::collections::VecDeque::new(),
            post_failures: 0,
            dead: false,
            reported_not_ready: false,
            pending: std::collections::HashMap::new(),
            forwarded_keys: HashSet::new(),
            host_replies: std::collections::HashMap::new(),
            undo_requests: std::collections::HashMap::new(),
            undo_document: None,
            history_requests: HashSet::new(),
            history_cache: None,
            last_history: None,
            relay,
            escape,
            escape_set: host_key_relay::StrokeSet::default(),
            view_events: std::collections::VecDeque::new(),
        }))
    }

    /// 接上平台端口。窗口建好、控制器就绪之后由平台调用。
    pub(crate) fn set_port(&mut self, port: Rc<dyn EditorPort>) {
        self.port = Some(port);
    }

    /// 平台句柄（文件对话框与让路原语要用）。
    pub(crate) fn set_window(&mut self, window: NativeWindow) {
        self.window = window;
    }

    pub(crate) fn is_dead(&self) -> bool {
        self.dead
    }

    /// 平台报告"页面已经没了"（WebKit 的 `web_process_terminated` / `load_failed`）。
    ///
    /// 【为什么平台可以直接判死】WebKit 的两个平台**没有**同步投递返回值，靠"连续
    /// 失败计数"永远等不到结论；而它们各自有一条明确的信号。让平台把结论交进来，
    /// 比让 Policy 去猜更准，也比三份判据各写一遍更省。
    // 【为什么 Windows 上允许未使用】Windows 的 WebView2 有同步投递返回值，靠
    // "连续失败计数"就能得出结论；这条入口是给两个 WebKit 平台的信号用的。
    #[cfg_attr(windows, allow(dead_code))]
    pub(crate) fn mark_dead(&mut self, reason: &str) {
        if self.dead {
            return;
        }
        self.dead = true;
        crate::gui_log_line(&format!("the embedded page is gone: {reason}"));
    }

    /// 页面就绪：把就绪前攒下的消息一次刷出。
    ///
    /// 【为什么两处都会调它】平台在"导航完成"时调一次（那时页面脚本已经装好监听），
    /// 页面自己在脚本末尾再发一条 `plugin_ready` 调一次。两者都幂等 —— 谁先到都算数，
    /// 后到的那个只多打一行日志。宁可多一条日志，也不要赌某个平台的时序。
    pub(crate) fn on_page_ready(&mut self) {
        if self.ready || self.closed {
            return;
        }
        self.ready = true;
        let queued = self.queue.len();
        crate::log_line(&format!(
            "[gui] page ready after {}ms ({queued} queued)",
            self.opened_at.elapsed().as_millis()
        ));
        while let Some(text) = self.queue.pop_front() {
            if !self.deliver_text(&text) {
                // 一条投不出去，后面的也排在同一条通道上；留着队列只会在下次
                // `on_page_ready` 里重复失败。
                self.queue.clear();
                break;
            }
        }
    }

    /// 把一条 JSON 交给页面。返回是否**已经**投递（入队也算 `true`：它没有丢）。
    ///
    /// 【为什么入队返回 true】调用方用返回值判断"要不要停止这一批" —— 队列是正常的
    /// 启动状态，不该被当成失败而截断后续消息。
    pub(crate) fn post(&mut self, value: serde_json::Value) -> bool {
        if self.closed {
            return false;
        }
        let text = value.to_string();
        if !self.ready {
            if self.queue.len() >= QUEUE_BUDGET {
                self.queue.pop_front();
                let waited = self.opened_at.elapsed().as_millis();
                crate::log_line(&format!(
                    "[gui] dropping the oldest queued message: the page has not become ready after {waited}ms"
                ));
                // 队列满了 = 页面根本没起来。这时候**必须**让用户看见一句话，
                // 否则界面是一块空面板，而日志在宿主的日志目录里没人会去翻。
                if !self.reported_not_ready {
                    self.reported_not_ready = true;
                    if let Some(port) = &self.port {
                        port.show_error("the editor page did not load; see the plug-in log");
                    }
                }
            }
            self.queue.push_back(text);
            return true;
        }
        self.deliver_text(&text)
    }

    /// 就绪之后的实际投递，顺带维护"连续失败"计数。
    fn deliver_text(&mut self, text: &str) -> bool {
        let Some(port) = self.port.clone() else {
            // 端口还没接上：与"未就绪"同义，但不要再攒进队列（队列已经刷过了）。
            return false;
        };
        if port.post_message(text) {
            self.post_failures = 0;
            return true;
        }
        self.post_failures = self.post_failures.saturating_add(1);
        if self.post_failures >= POST_FAILURE_LIMIT && !self.dead {
            self.dead = true;
            crate::log_line(&format!(
                "[gui] {} consecutive deliveries failed — the embedded page is gone",
                self.post_failures
            ));
        }
        false
    }

    /// 宿主投递的共享编辑键（Ctrl+C/X/V/Z/Y）。
    ///
    /// 【为什么去重在这里】按住不放的自动重复会被宿主反复投递；页面侧的
    /// `keydown` 处理按 `repeat` 字段区分。判据是"这个键是否已经按下未松开"，
    /// 属于 Policy —— 平台只管把键送进来。
    #[cfg_attr(all(unix, not(target_os = "macos")), allow(dead_code))]
    pub(crate) fn forward_edit_key(&mut self, key: char, mods: i16, down: bool) -> bool {
        if self.closed {
            return false;
        }
        let repeat = if down {
            !self.forwarded_keys.insert(key)
        } else {
            if !self.forwarded_keys.contains(&key) {
                return false;
            }
            self.forwarded_keys.remove(&key);
            false
        };
        let value = serde_json::json!({"version":1,"viewId":self.view_id,"event":"plugin_keyboard",
            "payload":{"type":if down {"keydown"} else {"keyup"},"key":key.to_string(),
                "ctrlKey":true,"shiftKey":mods&1!=0,"altKey":mods&2!=0,"metaKey":mods&8!=0,"repeat":repeat}});
        self.post(value)
    }

    /// 关闭视图：撤销所有"本视图还开着宿主 Undo 块"的记账，并停止出站投递。
    ///
    /// 【为什么先记账后关端口】`release_view` 之后宿主才不会等一个永远不会来的收尾；
    /// 端口随后由平台自己销毁（它才是原生视图的所有者）。
    pub(crate) fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.sink.closed.store(true, Ordering::Release);
        let documents = self
            .undo_document
            .iter()
            .cloned()
            .chain(self.undo_requests.values().cloned())
            .chain(
                self.host_replies
                    .values()
                    .map(|reply| reply.document.clone()),
            )
            .collect::<Vec<_>>();
        let view_id = self.view_id.clone();
        for document in documents.into_iter().filter_map(|doc| doc.upgrade()) {
            document.host_undo.release_view(&view_id);
        }
        self.port = None;
    }
}

/// 宿主侧的 20 ms 补做：私有状态通知与宿主 Undo 收尾。
///
/// 【为什么单独成函数】它必须发生在**宿主 UI 线程**上。情形 1/2 由视图定时器调用；
/// 情形 3（GTK 跑在私有线程上）里定时器在另一条线程，够不着宿主 —— 那时改由宿主
/// 下一次 `IPlugView` 回调补做（见 `view.rs` 的 `host_tick` 调用点），合并粒度变粗。
pub(crate) fn host_tick(link: &Arc<crate::editor::routing::EditorLink>) {
    link.flush_dirty();
    if let Ok(owner) = link.owner() {
        if let Ok(document) = owner.editor_document() {
            document.host_undo.tick();
        }
    }
}

/// 20 ms 定时器：先把宿主侧的事做完，再泵回执。
///
/// 【顺序不能换】私有状态通知必须先于可能收尾的 Undo 块；反过来会在宿主已记录状态后
/// 才标脏。原实现把这两步放在 `deliver` 的头部，这里只是把它们提到 `pump` 之外 ——
/// 因为情形 3 里 `pump` 跑在另一条线程上。
pub(crate) fn tick(state: &Rc<RefCell<EditorClient>>) {
    let link = state.borrow().link.clone();
    if let Ok(owner) = link.owner() {
        owner.refresh_host_transport_tick();
        // 【为什么在视图定时器上做这件事】宿主激活后仍无 ARA 绑定时要给出
        // 一次性诊断，而它必须走一条"未绑定也照跑"的路径：
        // `refresh_host_transport_tick` 在未绑定时会因 `host_query_stamp`
        // 失败而立即返回，诊断挂在它内部永远不会执行。视图定时器与视图
        // 同寿，且已经在每 20ms 跑一次。
        owner.note_editor_tick();
        // 【为什么在定时器上补发】窗口可能先于 ARA 绑定打开；绑定后
        // `hostKind` 与能力位都变了，而导航前注入的 bootstrap 是静态快照。
        // 只在已有会话时发事件 —— 这里绝不创建会话。
        if owner.take_bootstrap_changed() {
            if let Some(session) = owner.existing_editor_session() {
                let view_id = state.borrow().view_id.clone();
                session.emit("plugin_bootstrap_changed", bootstrap_json(&link, &view_id));
            }
        }
        // 让路状态机：与宿主时钟同一个节拍（20 ms）。
        //
        // 【为什么用视图定时器而不是自己的计时器】探针/观察窗要判的是"宿主动没动"，
        // 而宿主的播放态只有 `TransportClock` 知道；这个定时器已经在读它，
        // 而且它是"宿主不再回调时唯一还会跑的节拍"（见 `refresh_host_transport_tick`）。
        let reading = owner.transport_reading();
        let effects = state
            .borrow_mut()
            .relay
            .tick(std::time::Instant::now(), reading);
        apply_relay_effects(state, effects);
    }
    host_tick(&link);
    if !state.borrow().closed {
        pump(state);
    }
}

pub(crate) fn pump(state: &Rc<RefCell<EditorClient>>) {
    let link = state.borrow().link.clone();
    let (mut replies, events) = {
        let state = state.borrow();
        if state.closed {
            return;
        }
        let replies = state.replies.try_iter().take(32).collect::<Vec<_>>();
        let events = state.events.try_iter().take(64).collect::<Vec<_>>();
        (replies, events)
    };
    // 在宿主模型/PCM尚未ready的过渡期，保留原Promise。只排actor读取，不在UI做文件/推理。
    replies.retain_mut(|reply| {
        let Some(id)=reply["id"].as_u64() else {return true;};
        // 媒体复制需要先越过actor写入屏障；API仍在本UI线程，调用期间不借用EditorClient。
        // 【为什么用单次 `get` 而不是 `get` 之后再 `[&id]`】后者是 `Index` panic 形状：一旦
        // 将来这段改成跨借用边界，缺失的条目会直接 panic 在宿主进程里。这里一次取到
        // 需要的字段再克隆出来。
        let action={let state=state.borrow();state.host_replies.get(&id).and_then(|host|{
            host.action.clone().map(|action|(action,host.document.clone(),host.lease,host.editor.clone(),state.sink.clone(),state.view_id.clone()))})};
        if let Some((action,weak,lease,editor,sink,view_id))=action.filter(|_|reply["ok"]==true) {
            let outcome=(||->Result<bool,String>{
                let document=weak.upgrade().ok_or("media document closed")?;
                if state.borrow().host_replies.get(&id).is_some_and(|host|std::time::Instant::now()>=host.until) {return Err("media request timed out".into());}
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                if !allowed() {return Err("media editor lease changed".into());}
                state.borrow_mut().host_replies.get_mut(&id).ok_or("media request closed")?.action=None;
                let owner=link.owner()?;
                if action.command=="copy_timeline_clips" {reply["value"]=crate::host::clipboard::copy(&owner,&editor,&action.input,&allowed)?;return Ok(true);}
                let host=owner.project_history_host().filter(|host|host.can_clipboard_items()).ok_or("host media clipboard API unavailable")?;
                let duplicated=action.command=="duplicate_clips_bulk";
                let pasted=action.command=="paste_timeline_clipboard"||duplicated;
                // 计划只读预检；native Undo开始后才创建/删除，失败也必须结束请求。
                let paste=if duplicated {Some(crate::host::clipboard::plan_duplicate(&owner,&editor,&action.input,&allowed)?)} else if pasted {Some(crate::host::clipboard::plan_paste(&owner,&editor,&action.input,&allowed)?)} else {None};
                let delete=if pasted {None} else {Some(crate::host::clipboard::plan_delete(&owner,&editor,&action.input,action.command=="remove_clip",&allowed)?)};
                document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                state.borrow_mut().undo_requests.insert(id,Arc::downgrade(&document));
                let receipt=if let Some(plan)=paste {crate::host::clipboard::execute_paste(&owner,plan,&allowed)?}
                    else {crate::host::clipboard::execute_delete(&owner,delete.unwrap(),&allowed)?};
                // 所有native写入已结束；回执只读等待不能撑住宿主Undo块/自动准备，形成相互等待。
                // 失败仍由下方统一收尾；成功回执继续核对真实GUID，不重复创建item。
                state.borrow_mut().undo_requests.remove(&id);
                document.host_undo.finish_request(&view_id,id);
                state.borrow_mut().host_replies.get_mut(&id).ok_or("media view closed during write")?.media=Some(receipt);
                editor.enqueue(crate::editor::session::UiRequest {id,command:"get_timeline_inventory".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(false)
            })();
            match outcome {Ok(false)=>return false,Ok(true)=>{},Err(error)=>{reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!(error);}}
        }
        // Undo/Redo的actor屏障到达后才调用宿主；该外部调用期间不持EditorClient借用。
        let jump={let state=state.borrow();state.host_replies.get(&id).and_then(|host|{
            host.jump.map(|redo|(redo,host.document.clone(),host.lease,host.editor.clone(),state.sink.clone()))})};
        if let Some((jump,weak,lease,editor,sink))=jump {
            let result=(||->Result<bool,String>{
                let document=weak.upgrade().ok_or("undo document closed")?;
                if state.borrow().host_replies.get(&id).is_some_and(|host|std::time::Instant::now()>=host.until) {return Err("host history request timed out".into());}
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                if !allowed() {return Err("undo editor lease changed".into());}
                if !document.host_undo.finish_for_history() {
                    editor.enqueue(crate::editor::session::UiRequest {id,command:"plugin_history_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                    return Ok(false);
                }
                let host=link.owner()?.project_history_host().ok_or("host history unavailable")?;
                match jump {HistoryJump::Undo=>{host.history_jump(false,&allowed)?;},HistoryJump::Redo=>{host.history_jump(true,&allowed)?;},HistoryJump::Position(index)=>{host.history_jump_to(index,&allowed)?;}}
                // 【为什么用 `get_mut` 而不是 `[&id]`/`unwrap`】上面的 `history_jump` 是
                // **宿主回调**，其间不持 EditorClient 借用（见本段的注释），因此它可能
                // 重入并改动 `host_replies`。用索引会 panic，而这里跑在宿主进程里。
                if let Some(entry)=state.borrow_mut().host_replies.get_mut(&id) {entry.jump=None;}
                editor.enqueue(crate::editor::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(false)
            })();
            match result {Ok(false)=>return false,Ok(true)=>{},Err(error)=>{reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!(error);}}
        }
        let mut state=state.borrow_mut();
        if let Some(host)=state.host_replies.get(&id) {
            let error=reply["error"].as_str().unwrap_or("").to_owned();
            let imported=host.imported.as_ref().and_then(|item|host.document.upgrade().and_then(|doc|doc.clip_for_host_item(item))).map(|clip|host.editor.ui_clip_id(&clip));
            let waiting=host.imported.is_some()&&(imported.is_none()||!reply["value"]["clips"].as_array().is_some_and(|clips|clips.iter().any(|clip|Some(clip["id"].as_str().unwrap_or(""))==imported.as_deref())));
            let geometry_waiting=reply["ok"]==true&&host.geometry.as_ref().is_some_and(|geometry|!geometry.matches(&reply["value"]));
            let split_waiting=reply["ok"]==true&&host.split.as_ref().is_some_and(|split|!host.document.upgrade().is_some_and(|doc|split.matches(&doc,&host.editor.namespace,&mut reply["value"])));
            let media_waiting=reply["ok"]==true&&host.media.as_ref().is_some_and(|media|!host.document.upgrade().is_some_and(|doc|media.matches(&doc,&host.editor.namespace,&mut reply["value"])));
            let now = std::time::Instant::now();
            let retryable = waiting
                || geometry_waiting
                || split_waiting
                || media_waiting
                || reply["ok"] == false
                    && (crate::editor::errors::is_retryable_host_conflict(&error)
                        || error.contains(crate::editor::errors::HOST_PCM_UNAVAILABLE)
                        || error.contains(crate::editor::errors::HOST_MODEL_NOT_READY));
            // 退避：只在到点且未超预算时重投。预算与 `until` 双保险 —— 前者挡住
            // "永不收敛的失败"（否则 50 Hz 空转到 30 s），后者挡住"慢宿主"。
            let can_retry = host.attempts < HOST_REPLY_RETRY_BUDGET && now >= host.next_attempt;
            if retryable && now < host.until && can_retry {
                let allowed = host.document.upgrade().is_some_and(|doc| {
                    state
                        .link
                        .authorize(&doc)
                        .is_ok_and(|lease| lease == host.lease)
                });
                let command = if host.media.is_some() {
                    "get_timeline_inventory"
                } else {
                    "get_timeline_state"
                };
                let editor = host.editor.clone();
                let sink = state.sink.clone();
                let link = state.link.clone();
                let enqueued = allowed
                    && editor
                        .enqueue(crate::editor::session::UiRequest {
                            id,
                            command: command.into(),
                            args: serde_json::json!({}),
                            sink,
                            link: Some(link),
                        })
                        .is_ok();
                if enqueued {
                    // 单独取可变借用推进退避状态（上面的只读借用到此结束）。
                    if let Some(host) = state.host_replies.get_mut(&id) {
                        host.attempts = host.attempts.saturating_add(1);
                        host.next_attempt = now + host_reply_retry_backoff(host.attempts);
                    }
                    return false;
                }
            }
            if waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("item was created in the host but its ARA audio is not ready; use host Undo if canceling");}
            if geometry_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("clip was edited in the host but its ARA geometry has not caught up; host Undo remains available");}
            if split_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("item was split in the host but both clips have not caught up; host Undo remains available");}
            if media_waiting {reply["ok"]=serde_json::json!(false);reply["error"]=serde_json::json!("media was edited in the host but its GUI/ARA identities have not caught up; host Undo remains available");}
            if !waiting&&host.imported.is_some() {reply["value"]["imported_clip_id"]=serde_json::json!(imported);}
        }
        let done_history=state.history_requests.remove(&id);let view_id=state.view_id.clone();
        let finished_undo=state.undo_requests.remove(&id).and_then(|document|document.upgrade());
        let document=state.host_replies.remove(&id).and_then(|host|host.document.upgrade());state.pending.remove(&id);drop(state);
        if let Some(document)=finished_undo {document.host_undo.finish_request(&view_id,id);}
        if done_history {if let Some(document)=document {document.host_undo.finish_history(&view_id,id);}}
        true
    });
    // 让路事件与回执同一条出口（同一把"只在本线程发"的伞）。
    //
    // 每次只取 16 条：一次 tick 最多产生一两条，但宿主被按住不放时可能积压；
    // 分批发不会让一个卡住的 `PostWebMessageAsJson` 把整批都吞掉。
    {
        let mut state = state.borrow_mut();
        let mut taken = 0;
        while taken < 16 {
            match state.view_events.pop_front() {
                Some(event) => {
                    replies.push(event);
                    taken += 1;
                }
                None => break,
            }
        }
    }
    replies.extend(events);
    let history = link.owner().ok().and_then(|owner| {
        let document = owner.editor_document().ok()?;
        let lease = link.authorize(&document).ok()?;
        let host = owner.project_history_host()?;
        let allowed = || {
            link.authorize(&document)
                .is_ok_and(|current| current == lease)
        };
        let token = host.history_token(&allowed).ok()?;
        let identity = link.route_identity()?;
        let key = (identity.0, identity.1, token.0, token.1);
        if let Some((_, cached)) = state
            .borrow()
            .history_cache
            .as_ref()
            .filter(|(known, _)| *known == key)
        {
            return Some(cached.clone());
        }
        let value = host.project_history(&allowed).ok()?;
        state.borrow_mut().history_cache = Some((key, value.clone()));
        Some(value)
    });
    if let Some(history) = &history {
        for response in &mut replies {
            if response["event"] == "history_state" {
                response["payload"] = history.clone();
            }
            if response["value"]["clips"].is_array() {
                response["value"]["undo_depth"] = history["undoDepth"].clone();
                response["value"]["redo_depth"] = history["redoDepth"].clone();
            }
        }
        let changed = state.borrow().last_history.as_ref() != Some(history);
        if changed {
            state.borrow_mut().last_history = Some(history.clone());
            replies.push(serde_json::json!({"version":1,"viewId":state.borrow().view_id,"event":"history_state","payload":history}));
        }
    }
    for response in replies {
        let oversized = response.to_string().len() > RESPONSE_BUDGET;
        let response = if oversized {
            serde_json::json!({"version":1,"viewId":response["viewId"],"id":response["id"],
                "ok":false,"error":"native response budget exceeded"})
        } else {
            response
        };
        // 投递失败就停手：后面的回执排在同一条通道上，硬塞只会让它们一起丢。
        if !state.borrow_mut().post(response) {
            break;
        }
    }
}
/// 执行让路状态机声明的副作用。
///
/// 【为什么副作用不在状态机里做】`host_key_relay::Arbiter` 要保持纯逻辑（可单测、
/// 不碰窗口/COM/日志）。这里只做三件机械的事：投递、记账、写一行诊断。
fn apply_relay_effects(state: &Rc<RefCell<EditorClient>>, effects: Vec<host_key_relay::Effect>) {
    for effect in effects {
        match effect {
            host_key_relay::Effect::Trace(trace) => {
                let fidelity = state.borrow().relay.fidelity();
                log_relay(&format!("fidelity={} {}", fidelity.as_str(), trace));
            }
            host_key_relay::Effect::Notice { action, reason } => {
                log_relay(&format!("action={action} notice={reason}"));
                push_relay_notice(state, &action, reason);
            }
            host_key_relay::Effect::Escape { action, stroke } => {
                // 【一致性闸】页面只能让"它自己声明过"的键。集合为空 = 前端还没发布
                // （旧前端 / 启动瞬间），此时按"前端已经判定过"处理 —— 与引入集合之前的
                // 行为一致；集合非空时它就是一份白名单。
                let declared = {
                    let state = state.borrow();
                    state.escape_set.is_empty() || state.escape_set.contains(&stroke)
                };
                if !declared {
                    refuse_relay(state, &action, "unsupported-key");
                    continue;
                }
                let (fidelity, result) = {
                    let mut state = state.borrow_mut();
                    let fidelity = state.escape.fidelity();
                    // 句柄在 `WM_NCCREATE` 之后才有效；在此之前按下按键是不可能的
                    // （窗口还没显示），因此这里不做额外判空。
                    let window = state.window;
                    let result = state.escape.escape(window, &stroke);
                    (fidelity, result)
                };
                match result {
                    Ok(()) => log_relay(&format!(
                        "fidelity={} key={:?} relayed",
                        fidelity.as_str(),
                        stroke.token
                    )),
                    // 没投成：结构化原因进日志，并且**回报用户一次**（每个动作一次）。
                    Err(error) => refuse_relay(state, &action, error.as_str()),
                }
            }
        }
    }
}

/// 让路失败：记一行日志，并按"每个动作每次会话一次"回报用户。
///
/// 【为什么记账在状态机里】"每个动作只说一次"是判定的一部分，不是 UI 的偏好；
/// 放在状态机里，前端看到的与日志里记的是同一个事实。
fn refuse_relay(state: &Rc<RefCell<EditorClient>>, action: &str, reason: &str) {
    log_relay(&format!("action={action} refused reason={reason}"));
    let newly = state.borrow_mut().relay.note_refusal(action);
    if newly {
        push_relay_notice(state, action, "refused");
    }
}

fn push_relay_notice(state: &Rc<RefCell<EditorClient>>, action: &str, reason: &str) {
    let view_id = state.borrow().view_id.clone();
    state.borrow_mut().view_events.push_back(serde_json::json!({
        "version": 1,
        "viewId": view_id,
        "event": "host_key_relay_notice",
        "payload": { "action": action, "reason": reason },
    }));
}

/// 让路的诊断日志。
///
/// 【为什么不需要额外限流】投递本身已被状态机的 60 ms 硬间隔与"10 秒 20 次"限流，
/// 稳态最多 2 行/秒 —— 比给这里再加一个节流器更简单，也更不会掩盖真实节奏。
fn log_relay(text: &str) {
    crate::log_line(&format!("[relay] {text}"));
}
/// WebView 启动时注入的能力标志（`window.__HFS_PLUGIN_BOOTSTRAP__`）。
///
/// 【为什么单独成类型】前端按这些标志决定开放哪些操作（`services/hostCapabilities.ts`）。
/// 内联在一个 300 行的浏览器配置函数里时，**没人能对它写测试** —— 而这里恰好出过一次
/// 错（`trackGrouping` 被写成复用 `audioImport`，见 [`BootstrapFlags::to_json`]）。
///
/// 【为什么宿主写能力不再各占一个布尔】`clip_editing` / `clip_splitting` /
/// `clip_clipboard` / `audio_import` / `track_creation` 曾经与 `hostCapabilities` 并存，
/// 由**两处**各自计算（一处按本实例的原生扩展、一处按 `project_history_host()`）——
/// 两份判据必然漂移。现在它们只存在于 `host_capabilities`（由
/// [`crate::render::extension::ExtensionOwner::effective_host_capabilities`] 统一算出），
/// 前端只读一处。
#[derive(Clone, Copy, Debug)]
struct BootstrapFlags {
    /// 宿主提供 ARA 播放请求（`ARAPlaybackControllerInterface`）。
    ///
    /// 【为什么**不**放进 `hostCapabilities`】它描述的是**插件**能否向宿主发播放请求，
    /// 不是宿主能否提供 transport 读数（后者是 `HostCapabilities::transport`）。两者是
    /// 不同的东西：ARA 播放控制器可用不代表宿主把播放位置喂给了插件，反之亦然。
    transport_control: bool,
    /// 宿主用哪一套淡化轴：`Some("legacy")`（≤7.80，`C_FADE*SHAPE` 决定形状）
    /// 或 `Some("continuous")`（≥7.81，curvature/S 两轴决定形状）。
    ///
    /// 【为什么是枚举而不是布尔】"能不能编辑"与"编辑的是哪套轴"是两件事：新轴宿主
    /// 上没有形状号可写，界面得把预设翻译成 `(curvature, S)` 一对分量。`None` =
    /// 版本读不出来 → 前端整块保持只读（不猜）。
    ///
    /// 这套翻译的映射是实测的（`hifishifter_kernel::fade_axes`，证据
    /// `docs/2026-10-09-fade-axis-findings.md`），所以新轴宿主上**也**摆七个预设按钮；
    /// 此前这里的注释写着"预设映射尚未校准，所以只给连续滑杆"，那条限制已随实测消失。
    fade_axes: Option<&'static str>,
    /// 撤销历史由谁承载：`"host"`（宿主权威栈）或 `"local"`（插件自管栈）。
    ///
    /// 【为什么前端需要它】两条栈共用同一个「操作记录」面板，但 `label` 的含义不同：
    /// 本地栈是 op key（按 `history_op_*` 本地化），宿主栈是宿主自己的可读字符串
    /// （原样显示）。缺了这个判别式，前端只能无差别加前缀 —— 那正是
    /// `history_op_Move items` 这类原始键名被摊给用户的原因。
    history_backend: &'static str,
    /// 当前后端身份：`"reaper"` 或 `"ara_only"`。
    ///
    /// 【为什么前端需要它】非 REAPER 宿主上片段几何**只读**（ARA 2.0 不允许插件改
    /// region 几何）。前端据此显示只读横幅，而不是让用户点了写操作再撞一个错误。
    host_kind: &'static str,
    /// **生效的**能力位快照：写片段 / 分割 / 剪贴板 / 导入 / 建轨 / 撤销栈的唯一来源。
    host_capabilities: hifishifter_kernel::editor::host::HostCapabilities,
    /// Capture 通路的观察位；非 Capture 后端时为 `None`。
    ///
    /// 【为什么是"观察位"而不是能力位】它们描述宿主**实际**做了什么（给过位置吗、
    /// 给过音频吗、走过离线吗），而宿主不会声明这些。前端据此决定显示哪一条引导语：
    /// 见过位置就能录；没见过位置就该说"这个宿主没有把播放位置交给插件"，
    /// 而不是让用户对着一个点得动、永远不进度的按钮猜。
    capture_caps: Option<CaptureBootstrap>,
}

/// Capture 的观察位（`hostKind == "capture"` 时才有值）。
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureBootstrap {
    /// 宿主给过可用的 `ProcessContext` —— **录制的唯一硬前提**。
    saw_context: bool,
    /// 宿主在输入总线上给过音频。
    saw_input: bool,
    /// 宿主走过 `kOffline`（bounce / 导出）—— Capture 体验最好的那条通路。
    saw_offline: bool,
    /// 现在能不能录（等价于 `sawContext`，单独给出是为了让前端读一个字段就够）。
    can_capture: bool,
}

impl BootstrapFlags {
    fn to_json(self, view_id: &str) -> serde_json::Value {
        serde_json::json!({
            "version": 1,
            "viewId": view_id,
            "transportControl": self.transport_control,
            "fadeAxes": self.fade_axes,
            "historyBackend": self.history_backend,
            // 后端身份与能力：前端据此显示只读横幅、禁用不可用的入口。
            // 【这是写能力的唯一来源】clipWrite / split / itemClipboard / mediaImport /
            // trackCreate / undoStack 都从这里读，不再有并存的顶层布尔。
            "hostKind": self.host_kind,
            "hostCapabilities": self.host_capabilities,
            // Capture 的观察位（非 Capture 后端为 null）：前端据此选择引导语。
            "captureCaps": self.capture_caps,
            // 【为什么是常量 true，而不是复用 mediaImport】它守卫的是
            // `move_private_track`（`editor/session.rs`）—— 那条路径只改插件自己的
            // timeline 与私有参数分组，注释里明写"不调用任何宿主轨道 setter"，因此
            // **不需要任何宿主写能力**。此前它写成 `"trackGrouping": audio_import`，
            // 于是一个纯插件的参数分组功能被"能不能导入音频"门住：宿主缺少媒体 API
            // 时，插件里的轨道拖动会无声失效，而它本来根本不需要那个 API。
            "trackGrouping": true,
        })
    }
}

/// 当前 bootstrap 快照。导航前注入与绑定后的补发共用同一份事实。
pub(crate) fn bootstrap_json(
    link: &crate::editor::routing::EditorLink,
    view_id: &str,
) -> serde_json::Value {
    let owner = link.owner().ok();
    // 【为什么先落定形态】窗口打开时可能还在 ARA 绑定宽限期内；不按需求值，
    // `hostKind` 会停在 `ara_only`，非 ARA 宿主会被当成 ARA-only 对待。
    if let Some(owner) = &owner {
        owner.resolve_binding_mode();
    }
    BootstrapFlags {
        transport_control: owner
            .as_ref()
            .is_some_and(|owner| owner.host_playback().is_some()),
        fade_axes: owner
            .as_ref()
            .and_then(|owner| owner.host_fade_axes())
            .map(|axes_new| if axes_new { "continuous" } else { "legacy" }),
        history_backend: owner
            .as_ref()
            .map_or("local", |owner| owner.history_backend()),
        host_kind: owner
            .as_ref()
            .map(|owner| owner.host_backend().kind().as_str())
            .unwrap_or("ara_only"),
        capture_caps: owner
            .as_ref()
            .filter(|owner| {
                owner.host_backend().kind() == hifishifter_kernel::editor::host::HostKind::Capture
            })
            .map(|owner| {
                let status = owner.capture().status();
                CaptureBootstrap {
                    saw_context: status.saw_context,
                    saw_input: status.saw_input,
                    saw_offline: status.saw_offline,
                    can_capture: status.saw_context,
                }
            }),
        host_capabilities: owner
            .as_ref()
            .map(|owner| owner.effective_host_capabilities())
            .unwrap_or_else(hifishifter_kernel::editor::host::HostCapabilities::none),
    }
    .to_json(view_id)
}

/// 一条页面消息：校验 → 就地改写少数命令 → 分流。
///
/// 【为什么"就地改写"留在 Policy 里】`group_clips` / `close_track_gaps` /
/// `rename_clip_take` 三条命令的改写只依赖时间线与路由（不依赖任何平台 API），
/// 而它们改的是**同一条**前端命令的语义。放到平台侧就等于三份拷贝。
pub(crate) fn on_message(state: &Rc<RefCell<EditorClient>>, text: &str) {
    let view_id = {
        let state = state.borrow();
        if state.closed {
            return;
        }
        state.view_id.clone()
    };
    let Some(mut request) = protocol::parse_request(text, &view_id) else {
        return;
    };
    let Some(id) = request["id"].as_u64() else {
        return;
    };
    // 页面的第一句话：不进命令表，就地消费（见 `protocol::PAGE_READY`）。
    if request["command"] == protocol::PAGE_READY {
        // 【为什么要回一条回执】前端用 §invoke§ 发它（那是唯一现成的请求-响应通道），
        // 不回就等于让那个 promise 挂到 30 秒超时 —— 一条"证明通道能用"的消息
        // 自己先超时，是最糟的第一印象。
        let mut state = state.borrow_mut();
        state.on_page_ready();
        state.post(
            serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":true,
            "value":serde_json::Value::Null}),
        );
        return;
    }

    // 编组/解组仍使用同一宿主item setter与Undo块；把UI字符串groupId映射为有界REAPER I_GROUPID。
    if matches!(
        request["command"].as_str(),
        Some("group_clips") | Some("ungroup_clips")
    ) {
        let ids = request["args"]["clipIds"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if ids.is_empty() || ids.len() > 512 || ids.iter().any(|id| id.as_str().is_none()) {
            return;
        }
        let group = if request["command"] == "ungroup_clips" {
            0_i32
        } else {
            let names = ids
                .iter()
                .filter_map(|id| id.as_str())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            host_group_id_for(&names)
        };
        request["command"] = serde_json::json!("set_clips_state_bulk");
        request["args"] = serde_json::json!({"updates":ids.into_iter().map(|id|serde_json::json!({"clipId":id,"hostGroupId":group})).collect::<Vec<_>>(),"checkpoint":true});
    }
    if request["command"] == "close_track_gaps" {
        let track = request["args"]["trackId"].as_str().unwrap_or("").to_owned();
        let from = request["args"]["fromSec"].as_f64().unwrap_or(f64::NAN);
        let link = state.borrow().link.clone();
        if let Ok(owner) = link.owner() {
            if let Ok(editor) = owner.editor_session() {
                let moves = editor
                    .timeline
                    .lock_or_recover()
                    .close_track_gaps_moves(&track, from);
                request["command"] = if moves.is_empty() {
                    serde_json::json!("get_timeline_state")
                } else {
                    serde_json::json!("move_clips")
                };
                request["args"] = if moves.is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::json!({"moves":moves,"moveLinkedParams":true})
                };
            }
        }
    }
    if request["command"] == "rename_clip_take" {
        let clip_id = request["args"]["clipId"].as_str().unwrap_or("").to_owned();
        let take_id = request["args"]["takeId"].as_str().unwrap_or("").to_owned();
        let name = request["args"]["name"].as_str().unwrap_or("").to_owned();
        let link = state.borrow().link.clone();
        if let Ok(owner) = link.owner() {
            if let Ok(editor) = owner.editor_session() {
                let active = editor
                    .timeline
                    .lock_or_recover()
                    .clips
                    .iter()
                    .find(|clip| clip.id == clip_id)
                    .and_then(|clip| clip.active_take_id.clone());
                if active.as_deref() == Some(take_id.as_str()) {
                    // 只有 active take 的名字能落到宿主：`set_clip_state{name}` 写的是
                    // 宿主 item 的当前 take 名。非 active take 的重命名没有宿主对应
                    // （`SetTakeName` 只作用于 take 指针，而 ARA 侧只认 active take）。
                    request["command"] = serde_json::json!("set_clip_state");
                    request["args"] = serde_json::json!({"clipId":clip_id,"name":name,"checkpoint":request["args"]["checkpoint"]});
                } else {
                    // 【为什么给明确错误而不是让它掉进 catch-all】此前这条命令原样落到
                    // `Command unavailable in plug-in mode: rename_clip_take` ——
                    // 用户看到的是"命令不可用"，而不是"只有当前 Take 能改名"。后者才是
                    // 可操作的。
                    state.borrow_mut().post(serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,
                    "error":"only the active take can be renamed in the plug-in; switch to it in the host first"}));
                    return;
                }
            }
        }
    }
    // 拖入的真实磁盘文件由**平台**在进入本函数之前解析成路径（见 `gui/windows.rs`）：
    // WebKit 的 `File` 没有磁盘路径（沙箱），macOS/Linux 走原生拖放事件 + `import_audio_item`。
    // 普通命令只排队；UI回调不得分析、推理、读文件或等待worker。
    let result = match request["command"].as_str() {
        // 【为什么必须内联】启动探活发生在宿主重开工程的那一刻，而 `editor_session()`
        // 可能正在做模型加载 / actor 创建。探活只需要 owner 的原子读数，不能排在
        // 会话队列后面 —— 那正是 `get_runtime_info` 30 s 超时的来源。
        Some("plugin_health") => {
            let link = state.borrow().link.clone();
            link.owner().map(|owner| owner.health_snapshot())
        }
        // ARA 绑定后补发 bootstrap：前端拿到新快照后更新 hostKind / 能力位。
        Some("get_plugin_bootstrap") => {
            let state = state.borrow();
            Ok(bootstrap_json(&state.link, &state.view_id))
        }
        // 【为什么也内联】状态栏在启动扇出里会问一次。它会创建会话、走 actor 队列；
        // 重开工程时那正是最慢的一段。这里只读 owner 的原子读数，返回同样的形状。
        Some("get_runtime_info") => {
            let link = state.borrow().link.clone();
            link.owner().map(|owner| {
                let health = owner.health_snapshot();
                let clip_count = health["clip_count"].as_u64().unwrap_or(0);
                let capture = health["host_kind"].as_str() == Some("capture");
                serde_json::json!({
                    "ok": true,
                    "device": if capture { "Host / Capture" } else { "Host / ARA" },
                    "model_loaded": health["model_available"].as_bool().unwrap_or(false),
                    "audio_loaded": clip_count > 0,
                    "has_synthesized": false,
                    "is_playing": false,
                    "playback_target": serde_json::Value::Null,
                    "gpu_backend": "",
                    "clip_count": clip_count,
                })
            })
        }
        Some("ping") => Ok(serde_json::json!({"ok":true,"mode":"plugin","view_id":view_id,
            "connected":state.borrow().link.owner().is_ok()})),
        Some("log_frontend_error") => {
            // 【为什么按 (message, detail) 分桶限流，而不是原样落盘】
            // 这个入口是**运行期数据驱动**的：每个失败的方法名是一类新错误。
            // 此前每条都原样写进日志，实测把一个终态错误（组件已关闭而页面仍在
            // 轮询）放大成 50 分钟 19,056 行、轮转文件合计约 32 MB 纯噪声 ——
            // 用户报障时真正有用的线索全被冲掉。
            // 现在每类错误各自 10 秒最多一条，且窗口内被抑制的条数会在下一条
            // 输出前以 `[throttled]` 汇总行补记（不丢信息，只丢重复）。
            let args = &request["args"];
            let message = args["message"].as_str().unwrap_or("");
            let detail = args["detail"].as_str().unwrap_or("");
            let key = format!(
                "frontend:{}:{}",
                truncate_key(message),
                truncate_key(detail)
            );
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Info,
                &key,
                format_args!("Embedded frontend: {}", args),
            );
            Ok(serde_json::Value::Null)
        }
        Some(command @ ("open_audio_dialog"|"open_audio_dialog_multi"))=>{
            let link=state.borrow().link.clone();( ||->Result<serde_json::Value,String>{
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                let host=owner.project_history_host().filter(|host|host.can_import_audio()).ok_or("host audio import is unavailable")?;
                let paths=host.pick_audio_paths(command=="open_audio_dialog_multi",&||link.authorize(&document).is_ok_and(|current|current==lease))?;
                Ok(if command=="open_audio_dialog_multi" {serde_json::json!({"ok":true,"canceled":paths.is_empty(),"paths":paths})}
                    else {serde_json::json!({"ok":true,"canceled":paths.is_empty(),"path":paths.first()})})
            })()
        },
        Some("pick_directory")=>{
            let (link,window)={let state=state.borrow();(state.link.clone(),state.window)};
            (||->Result<serde_json::Value,String>{
                let owner=link.owner()?;
                let identity=interaction_identity(&link)?;
                let editor=owner.editor_session()?;
                let path=crate::editor::browser_files::pick_directory(window)?;
                if !interaction_unchanged(&link,&identity) {return Err("folder picker editor lease changed".into());}
                match path {Some(path)=>Ok(serde_json::json!({"ok":true,"path":editor.grant_browser_directory(&path)?})),
                    None=>Ok(serde_json::json!({"ok":true,"canceled":true}))}
            })()
        },
        Some("import_audio_item")=>{
            let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id,false))};
            let outcome=(||->Result<HostReply,String>{
                if full {return Err("native import request budget exceeded".into());}
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;let editor=owner.editor_session()?;
                let input=&request["args"];let path=input["audioPath"].as_str().ok_or("audio path missing")?;
                if input["mediaAudioStreamIndex"].as_u64().is_some_and(|index|index!=0) {return Err("specific container audio stream import is not supported yet".into());}
                let track=input["trackId"].as_str().map(|id|editor.host_track_id(id)).transpose()?;
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                let host=owner.project_history_host().ok_or("import project history missing")?;
                let new_track=input["trackId"].is_null()&&input.as_object().is_some_and(|object|object.contains_key("trackId"));
                let existing=if new_track {None} else {Some(owner.audio_import_target(track.as_deref(),&allowed)?)};
                state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                let result=(||->Result<String,String>{
                    let mut created=if new_track {Some(host.create_audio_track(std::path::Path::new(path).file_stem().and_then(|name|name.to_str()).unwrap_or("Imported audio"),None,&allowed)?)} else {None};
                    let target=existing.as_ref().or_else(||created.as_ref().map(|track|&track.target)).ok_or("import target missing")?;
                    let id=target.import_audio(path,input["startSec"].as_f64().unwrap_or_else(||owner.clock.get().map(|clock|clock.read().0).unwrap_or(0.)),&allowed)?;
                    if let Some(track)=&mut created {track.commit();}Ok(id)
                })();
                document.host_undo.finish_request(&view_id,id);let imported=result?;
                owner.refresh_host_transport();editor.notify_timeline();
                editor.enqueue(crate::editor::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:Some(imported),geometry:None,split:None,media:None,action:None})
            })();
            match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id,false);state.host_replies.insert(id,reply);return;},Err(error)=>Err(error)}
        },
        // 独立的"添加宿主轨道"入口。
        //
        // 【为什么不是 `add_track`】`add_track` 建的是**插件自己的**轨道，且被
        // `pluginAllowsAction` 的 `track.*` 规则一刀切掉；这条建的是**宿主**轨道。
        // 两者语义不同，不能共用一个名字。
        //
        // 【为什么插入位置要锚定】用户说的"最下方"是**本实例能看到的最下方**，
        // 不是整个工程的最底下 —— 后者会把新轨道丢到别的 FX 实例的轨道下面。
        Some("create_host_track")=>{
            let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id,false))};
            let outcome=(||->Result<HostReply,String>{
                if full {return Err("native track request budget exceeded".into());}
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                let editor=owner.editor_session()?;
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                let host=owner.project_history_host().ok_or("host project history missing")?;
                if !host.can_create_audio_track() {return Err("host track creation is unavailable".into());}
                let anchor=host.folder_anchor_order(&allowed);
                state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                let result=(||->Result<(),String>{
                    let mut created=host.create_audio_track("HiFiShifter",anchor,&allowed)?;
                    created.commit();
                    Ok(())
                })();
                document.host_undo.finish_request(&view_id,id);result?;
                owner.refresh_host_transport();editor.notify_timeline();
                editor.enqueue(crate::editor::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:None,geometry:None,split:None,media:None,action:None})
            })();
            match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id,false);state.host_replies.insert(id,reply);return;},Err(error)=>Err(error)}
        },
        // 精简诊断导出（Help 菜单）。命令名与独立 App **相同** —— 前端不需要按
        // 模式分支，也不需要一个"插件能不能导出"的能力标志。
        //
        // 【为什么是精简版】App 的完整包会跑推理设备基准测试，而那条路径在
        // GPU/驱动异常时可能硬崩；在 DAW 进程里崩会带走用户整个会话。基准测试
        // 在插件里**保持拒绝**（见 `commands.rs`），这里只做不会崩的部分。
        Some("pick_diagnostics_output_path")=>{
            let (link,window)={let state=state.borrow();(state.link.clone(),state.window)};
            (||->Result<serde_json::Value,String>{
                let identity=interaction_identity(&link)?;
                let name=crate::editor::plugin_diagnostics::default_file_name();
                let path=crate::editor::browser_files::pick_save_path(window,&name,"zip")?;
                // 对话框期间用户可能关窗/切实例；租约变了就不再交付这个路径。
                if !interaction_unchanged(&link,&identity) {return Err("diagnostics picker editor lease changed".into());}
                match path {Some(path)=>Ok(serde_json::json!({"ok":true,"path":path.to_string_lossy()})),
                    None=>Ok(serde_json::json!({"ok":false,"canceled":true}))}
            })()
        },
        // 「导出 MIDI」的另存为对话框。命令名与独立 App **相同**，前端不必按模式
        // 分支；真正的导出（读曲线 → 写文件）走 actor 线程的 `export_pitch_to_midi`。
        //
        // 【为什么对话框必须在 UI 线程】`IFileSaveDialog` 需要一个父 HWND，而
        // HWND 只存在于 UI 线程（与诊断导出、记事本另存为同一条理由）。
        Some("pick_midi_output_path")=>{
            let (link,window)={let state=state.borrow();(state.link.clone(),state.window)};
            (||->Result<serde_json::Value,String>{
                let identity=interaction_identity(&link)?;
                let path=crate::editor::browser_files::pick_save_path(window,"export.mid","mid")?;
                // 对话框期间用户可能关窗/切实例；租约变了就不再交付这个路径。
                if !interaction_unchanged(&link,&identity) {return Err("MIDI picker editor lease changed".into());}
                match path {
                    Some(path)=>Ok(serde_json::json!({"ok":true,"canceled":false,"path":path.to_string_lossy()})),
                    None=>Ok(serde_json::json!({"ok":true,"canceled":true})),
                }
            })()
        },
        // 「导入 MIDI」的打开对话框。命令名与独立 App **相同**，前端不必按模式
        // 分支；真正的解析/写入走 actor 线程的 `get_midi_tracks` /
        // `import_midi_to_pitch`。
        //
        // 【为什么对话框必须在 UI 线程】`IFileOpenDialog` 需要一个父 HWND，而
        // HWND 只存在于 UI 线程（与文件夹选择器、另存为对话框同一条理由）。
        Some("open_midi_dialog")=>{
            let (link,window)={let state=state.borrow();(state.link.clone(),state.window)};
            (||->Result<serde_json::Value,String>{
                let identity=interaction_identity(&link)?;
                let path=crate::editor::browser_files::pick_open_path(window,"MIDI","mid")?;
                // 对话框期间用户可能关窗/切实例；租约变了就不再交付这个路径。
                if !interaction_unchanged(&link,&identity) {return Err("MIDI picker editor lease changed".into());}
                match path {
                    Some(path)=>Ok(serde_json::json!({"ok":true,"canceled":false,"path":path.to_string_lossy()})),
                    None=>Ok(serde_json::json!({"ok":true,"canceled":true})),
                }
            })()
        },
        Some("export_diagnostics")=>{
            // 【为什么结果在慢车道上回执】打包要读最多两份 32 MB 日志并 deflate，
            // 而这里是宿主 UI 线程 —— 同步写完，编辑器在整个导出期间不响应（前端给
            // 它的预算是 5 分钟）。下面那些读（宿主版本、时钟、渲染器诊断、设置）
            // 都是纯读 + JSON，且只有 UI 线程拿得到，所以留在原地：
            // 这就是"**先在 UI 线程取快照，再把慢 I/O 下放**"的样板。
            let (link,sink)={let state=state.borrow();(state.link.clone(),state.sink.clone())};
            let outcome=(||->Result<(),String>{
                let output=request["args"]["outputPath"].as_str().ok_or("diagnostics output path missing")?;
                let out=std::path::PathBuf::from(output);
                if !out.is_absolute() {return Err("diagnostics output path must be absolute".into());}
                let owner=link.owner()?;
                // Capture 形态没有文档：时钟退化到 owner 自己的那份，渲染者列表为空
                // （没有 ARA renderer 租约）。诊断包在两种形态下都必须能导出 ——
                // 用户报障时它正是第一件要拿到的东西。
                let document=owner.editor_document().ok();
                let backend=owner.host_backend();
                let system=serde_json::json!({
                    "product":"HiFiShifter VST3 / ARA plugin",
                    "about":hifishifter_kernel::build_info::about_payload(crate::plugin_form_label(Some(owner.as_ref())),crate::VERSION),
                    "os":std::env::consts::OS,"arch":std::env::consts::ARCH,
                    "generatedAt":chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
                    // 用户报障时"你是哪个宿主"是第一个要问的问题。宿主名来自
                    // `IHostApplication::getName`（只诊断用，绝不参与分支）。
                    "host":{"appName":backend.app_name().map(str::to_owned)
                            .or_else(crate::editor::connection::host_application_name),
                        "appVersion":backend.app_version()},
                    // 当前编辑后端与能力位：报障时能直接读出"是不是 ARA-only 只读后端"。
                    "backend":{"kind":backend.kind().as_str(),
                        "nativeAdapter":backend.native().is_some(),
                        "capabilities":backend.capabilities()},
                    "transport":document.as_ref().map(|document|document.clock.diagnostics())
                        .or_else(||owner.clock.get().map(|clock|clock.diagnostics())),
                    // 音频线程被 catch_unwind 拦下的 panic 次数。音频线程不能写日志，
                    // 只能计数；这里（UI 线程）把它落进诊断包，否则"崩过但没痕迹"。
                    "audioPanicCount":crate::vst3::audio_panic_count(),
                    "renderers":document.as_ref().map(|document|document.renderer_diagnostics())
                        .unwrap_or_else(||serde_json::json!([])),
                    // 慢车道状态：报障时"有一条长任务卡在这里"是最难从日志猜出来的
                    // 事实，让它随诊断包一起走。
                    "slowLane":crate::editor::slow_lane::snapshot(),
                });
                let settings=serde_json::json!({
                    "plugin":crate::settings_store::settings(),
                    "frontendPrefs":crate::settings_store::frontend_prefs(),
                    "frontend":request["args"]["frontendSettings"].clone(),
                });
                crate::editor::slow_lane::submit(crate::editor::slow_lane::SlowTask {
                    id,
                    command:"export_diagnostics".into(),
                    sink,
                    work:Box::new(move||{
                        crate::editor::plugin_diagnostics::write_package(&out,&system,&settings)?;
                        crate::log_line(&format!("[diagnostics] plugin package exported: {}",out.display()));
                        Ok(serde_json::json!({"ok":true,"path":out.to_string_lossy()}))
                    }),
                })?;
                Ok(())
            })();
            match outcome {
                // 回执由慢车道发出：这里**不能**再走尾部的通用回执，否则同一个 id 会收到两条。
                Ok(())=>{state.borrow_mut().mark_pending(id,true);return;}
                Err(error)=>Err(error),
            }
        },
        // 记事本导出 / 附件另存为：需要系统保存对话框（HWND 只有 UI 线程有），
        // 因此不走 actor dispatch。
        // ── 纯数据导出（布局 / 主题 / 颤音预设）──
        //
        // 【为什么插件必须支持它们】三者都是"把前端已经算好的 JSON 存成文件"：
        // 不碰宿主、不碰音频、不碰时间线 —— 在 ARA-only 宿主上与独立 App 完全同义。
        // 此前它们落到 `Command unavailable`，于是对应菜单项点了没有任何反应。
        //
        // 【为什么对话框必须在 UI 线程】与诊断导出、MIDI 另存为同一条理由：
        // `IFileSaveDialog` 需要一个父 HWND，而 HWND 只存在于 UI 线程。
        Some(command @ ("export_layout_json"|"export_theme_json"|"export_vibrato_presets_json"))=>{
            let (link,window)={let state=state.borrow();(state.link.clone(),state.window)};
            let json=request["json"].as_str().unwrap_or_default().to_owned();
            let fallback=match command{
                "export_layout_json"=>"hifishifter-layout.json",
                "export_theme_json"=>"hifishifter-theme.json",
                _=>"hifishifter-vibrato-preset.json",
            };
            let requested=request["defaultFileName"].as_str().unwrap_or_default().trim().to_owned();
            let name=if requested.is_empty(){fallback.to_owned()}else{requested};
            (||->Result<serde_json::Value,String>{
                // 对话框是模态的：期间用户可能关窗或切实例，所以前后各核对一次租约。
                let identity=interaction_identity(&link)?;
                let path=crate::editor::browser_files::pick_save_path(window,&name,"json")?;
                if !interaction_unchanged(&link,&identity) {return Err("json export dialog editor lease changed".into());}
                let Some(path)=path else {return Ok(serde_json::json!({"ok":false,"canceled":true}))};
                std::fs::write(&path,json.as_bytes()).map_err(|e|e.to_string())?;
                Ok(serde_json::json!({"ok":true,"canceled":false,"path":path.to_string_lossy()}))
            })()
        },
        Some(command @ ("notebook_export_document"|"notebook_save_asset_as"))=>{
            let (link,window)={let state=state.borrow();(state.link.clone(),state.window)};
            (||->Result<serde_json::Value,String>{
                // 对话框是模态的：期间用户可能关窗或切实例，所以前后各核对一次租约。
                let identity=interaction_identity(&link)?;
                let result=crate::editor::notebook::export_with_dialog(window,command,&request["args"])?;
                if !interaction_unchanged(&link,&identity) {return Err("notebook dialog editor lease changed".into());}
                Ok(result)
            })()
        },
        Some(command @ ("play_original"|"play_synthesized"|"stop_audio"))=>{
            // host callback可能同步重入；调用期间不持EditorClient/会话锁。
            let link=state.borrow().link.clone();
            (||->Result<serde_json::Value,String> {
                let owner=link.owner()?;
                let playback=owner.host_playback().ok_or("host does not provide ARA playback control")?;
                let position=owner.clock.get().map(|clock|clock.read().0).unwrap_or(0.);
                if command=="stop_audio" {playback.stop().map_err(|e|e.to_string())?;
                    // ARA只有Stop请求；暂停后定位到实际停止点，Stop按钮的回锚由原thunk另发seek。
                    playback.set_position(position).map_err(|e|e.to_string())?;
                    Ok(serde_json::json!({"ok":true,"stopped_at_sec":position}))
                } else {playback.start().map_err(|e|e.to_string())?;
                    Ok(serde_json::json!({"ok":true,"host_request":true,"anchorSec":position,"start_sec":position}))}
            })()
        },
        Some(command @ ("split_clip"|"split_clips_at"))=>{
            let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id,false))};
            let outcome=(||->Result<HostReply,String> {
                if full {return Err("native split request budget exceeded".into());}
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                let editor=owner.editor_session()?;let input=request.get("args").cloned().unwrap_or_else(||serde_json::json!({}));
                let plan=editor.plan_host_split(command,&input)?;
                let host=owner.project_history_host().ok_or("host project history missing")?;
                if !host.can_split_clips() {return Err("host split capability unavailable".into());}
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                document.host_undo.begin_request(&view_id,id,true,true,&host,&allowed)?;
                let outcome=crate::host::host_split::execute(&owner,plan,&allowed);
                document.host_undo.finish_request(&view_id,id);let split=Some(outcome?);
                editor.enqueue(crate::editor::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:None,geometry:None,split,media:None,action:None})
            })();
            match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id,false);state.host_replies.insert(id,reply);return;},Err(error)=>Err(error)}
        },
        Some(command) if crate::host::host_edit::is_clip_edit(command)=>{
            let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id,false))};
            let outcome=(||->Result<HostReply,String> {
                if full {return Err("native host edit request budget exceeded".into());}
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                let editor=owner.editor_session()?;let input=request.get("args").cloned().unwrap_or_else(||serde_json::json!({}));
                let plan=editor.plan_host_edit(command,&input)?;
                // 完成门需要知道宿主的淡化轴版本：形状预设在新轴/旧轴上落盘的字段不同，
                // 期望值也就不同（见 `HostEditPlan::receipt`）。
                let fade_axes_v2=owner.host_fade_axes();
                let geometry=Some(plan.receipt(&editor.namespace,fade_axes_v2));
                state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                let host=owner.project_history_host().ok_or("host project history missing")?;
                let allowed=||link.authorize(&document).is_ok_and(|current|current==lease);
                // checkpoint 由调用方声明（流式提交的末笔才收尾批次），此前这里写死
                // `true`，任何按帧分块提交的调用方会把宿主撤销栈打爆（对照通用命令
                // 分支对 `args.checkpoint` 的读法）。缺省仍是 true：整笔提交是常态。
                let checkpoint=input["checkpoint"].as_bool().unwrap_or(true);
                document.host_undo.begin_request(&view_id,id,checkpoint,true,&host,&allowed)?;
                let outcome=crate::host::host_edit::execute_managed(&owner,plan,allowed,true);
                document.host_undo.finish_request(&view_id,id);outcome?;
                editor.enqueue(crate::editor::session::UiRequest {id,command:"get_timeline_state".into(),args:serde_json::json!({}),sink,link:Some(link.clone())})?;
                Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,imported:None,geometry,split:None,media:None,action:None})
            })();
            match outcome {
                Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id,false);state.host_replies.insert(id,reply);return;},
                Err(error)=>Err(error),
            }
        },
        Some("has_timeline_clipboard")=>crate::host::clipboard::available(),
        Some(command @ ("copy_timeline_clips"|"paste_timeline_clipboard"|"duplicate_clips_bulk"|"remove_clip"|"remove_clips"))=>{
            let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id,false))};
            let outcome=(||->Result<HostReply,String>{
                if full {return Err("native media request budget exceeded".into());}
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;
                if !owner.project_history_host().is_some_and(|host|host.can_clipboard_items()) {return Err("host media clipboard API unavailable".into());}
                let editor=owner.editor_session()?;
                document.host_undo.begin_history(&view_id,id)?;
                if let Err(error)=editor.enqueue(crate::editor::session::UiRequest {id,command:"plugin_editor_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())}) {
                    document.host_undo.finish_history(&view_id,id);return Err(error);
                }
                Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:None,
                    imported:None,geometry:None,split:None,media:None,action:Some(MediaAction {command:command.into(),input:request.get("args").cloned().unwrap_or_else(||serde_json::json!({}))})})
            })();
            match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.undo_document=Some(reply.document.clone());state.mark_pending(id,false);state.history_requests.insert(id);state.host_replies.insert(id,reply);return;},Err(error)=>Err(error)}
        },
        // 【无条件处理，不再用 `is_some()` 做 guard】此前这两条 arm 只在宿主提供撤销栈
        // 时才匹配，非 REAPER 宿主会落到下面的 `Some(command)` 分支并被静默丢弃 ——
        // 用户按 Ctrl+Z 什么都不发生。现在"没有宿主撤销栈"是 `LocalHistory` 这个
        // **实现**，`history_authority()` 永远返回一个权威（见 `host::history`）。
        Some("get_history_state")=>{
            let link=state.borrow().link.clone();( ||->Result<serde_json::Value,String>{
                // 【为什么不再先取文档】历史权威有两套实现：宿主栈（需要文档）与本地栈
                // （不需要）。此前这里无条件先取文档，于是 Capture 形态 —— 它**只有**
                // 本地栈 —— 连"读一次历史状态"都失败。
                let owner=link.owner()?;
                let identity=interaction_identity(&link)?;
                owner.history_authority().snapshot(&||interaction_unchanged(&link,&identity))
            })()
        },
        Some(command @ ("undo_timeline"|"redo_timeline"|"set_history_position"))=>{
            let (link,sink)={let state=state.borrow();(state.link.clone(),state.sink.clone())};
            // 宿主权威栈 ⇒ 走异步路径（必须等在途写入落定再跳，否则会跳到一半的状态）。
            //
            // 【为什么比 `"host"` 而不是 `"reaper"`】`ProjectHistory::backend()` 返回
            // `"host"`（`host::history`，随后端去品牌化改名）。此前这里比的是旧值
            // `"reaper"`，于是 `uses_host` 恒为 `false` —— REAPER 里宿主撤销/重做
            // 实际走了本地栈，`history_jump*` 整条路径不可达。
            let uses_host = link
                .owner()
                .is_ok_and(|owner| owner.history_authority().backend() == "host");
            if !uses_host {
                // 本地权威栈 ⇒ 与普通命令同一条队列：本地命令天然排在已入队写入之后，
                // 不需要额外的 barrier，也不会被 `host_undo.pending` 的门控挡住。
                let full=state.borrow_mut().pending_full(id,false);
                let outcome=if full {Err("pending request budget exceeded".into())} else {
                    link.owner().and_then(|owner|owner.editor_session()?.enqueue(crate::editor::session::UiRequest {
                        id,command:command.into(),args:request.get("args").cloned().unwrap_or_else(||serde_json::json!({})),sink,link:Some(link.clone()),
                    }))
                };
                match outcome {Ok(())=>{state.borrow_mut().mark_pending(id,false);Ok(serde_json::Value::Null)},Err(error)=>Err(error)}
            } else {
            let outcome=(||->Result<HostReply,String>{
                if state.borrow_mut().pending_full(id,false) {return Err("native history request budget exceeded".into());}
                let owner=link.owner()?;let document=owner.editor_document()?;let lease=link.authorize(&document)?;let editor=owner.editor_session()?;
                let jump=match command {"undo_timeline"=>HistoryJump::Undo,"redo_timeline"=>HistoryJump::Redo,_=>HistoryJump::Position(request["args"]["position"].as_i64().filter(|v|(0..10000).contains(v)).ok_or("invalid host history position")? as i32)};
                state.borrow_mut().undo_document=Some(Arc::downgrade(&document));
                document.host_undo.begin_history(&view_id,id)?;
                if let Err(error)=editor.enqueue(crate::editor::session::UiRequest {id,command:"plugin_history_barrier".into(),args:serde_json::json!({}),sink,link:Some(link.clone())}) {
                    document.host_undo.finish_history(&view_id,id);return Err(error);
                }
                Ok(HostReply {editor,document:Arc::downgrade(&document),lease,until:std::time::Instant::now()+std::time::Duration::from_secs(30),attempts:0,next_attempt:std::time::Instant::now(),jump:Some(jump),imported:None,geometry:None,split:None,media:None,action:None})
            })();
            match outcome {Ok(reply)=>{let mut state=state.borrow_mut();state.mark_pending(id,false);state.history_requests.insert(id);state.host_replies.insert(id,reply);return;},Err(error)=>Err(error)}
            }
        },
        // ── 宿主专属快捷键的让路（Host Key Relay）────────────────────────────
        //
        // 【为什么在视图层内联】全部状态都在 EditorClient 上（状态机、平台原语、
        // 待发事件），而会话分发只拿得到 EditorSession —— 与 `set_transport` 取宿主
        // 播放控制同一条路。
        //
        // 【为什么立即回执】探针（90 ms）与观察窗（400 ms）是**之后**的事；前端此刻
        // 要的只是"这一下接不接手"。把回执拖到探针之后，会让连按的第二下永远撞上
        // "in-flight"。
        Some("host_key_relay")=>{
            let action=request["args"]["action"].as_str().unwrap_or_default().to_owned();
            // 【为什么不在这里用 `?`】本闭包返回的是 `windows::core::Result`（COM 调用），
            // 而命令失败要变成 `Err(String)` 交给下面的统一回执分支。两种错误类型混用
            // 会让 `?` 去找一个不存在的 `From<&str>` —— 这里显式分支即可。
            match host_key_relay::Stroke::from_json(&request["args"]) {
                Some(stroke)=>{
                    let reading=state.borrow().link.owner().ok().and_then(|owner|owner.transport_reading());
                    let outcome=state.borrow_mut().relay.on_key(std::time::Instant::now(),&action,stroke,reading);
                    apply_relay_effects(&state,outcome.effects);
                    Ok(serde_json::json!({
                        "accepted":matches!(outcome.verdict,host_key_relay::KeyVerdict::Accepted),
                        "reason":outcome.verdict.reason(),
                    }))
                }
                // 键不在表里 / 载荷不合法：**让不成立**。回执里说清楚，让前端回落
                // 旧文案；而不是让整条命令失败 —— 那会连带触发前端错误上报，噪声更大。
                None=>Ok(serde_json::json!({
                    "accepted":false,
                    "reason":host_key_relay::RejectReason::UnsupportedKey.as_str(),
                })),
            }
        }
        // 把键盘焦点交还宿主：让路的**确定**那条路（不投递任何键，只换焦点）。
        //
        // 【为什么要有它】投递窗口消息要看宿主脸色（有的宿主只看自己的前台焦点
        // 窗口）。当投递不灵时，用户至少需要一条一定能生效的路 —— 代价是焦点真的
        // 走了，之后要点击插件才能回来。所以它不做默认，只做显式菜单项。
        Some("host_key_focus")=>{
            let result={let mut state=state.borrow_mut();let window=state.window;state.escape.focus_host(window)};
            Ok(match result {
                Ok(())=>{log_relay("fidelity=focus handed the keyboard back to the host");serde_json::json!({"ok":true})},
                Err(error)=>serde_json::json!({"ok":false,"reason":error.as_str()}),
            })
        }
        Some("host_key_escape_set")=>{
            let set=host_key_relay::StrokeSet::from_json(&request["args"]["strokes"]);
            let count=set.len();
            state.borrow_mut().escape_set=set;
            Ok(serde_json::json!({"ok":true,"count":count}))
        }
        // 拖入的真实磁盘文件：Windows 在**进入本函数之前**就把它改写成
        // `import_audio_item`（见 `gui/windows.rs` 的 `native_file_request`）；
        // WebKit 的 `File` 没有磁盘路径（沙箱），两个 WebKit 平台走原生拖放事件。
        // 落到这里说明页面走了 WebView2 那条 API，如实说清楚该怎么导入。
        Some("import_native_audio_file")=>Err(
            "this window cannot read the disk path of a dropped file; drop it on the window itself or use the File menu".into()),
        Some(command)=>{
            // 【为什么 Capture 命令在视图层拦下】它的全部状态都在
            // `ExtensionOwner` 上（捕获存储、take、观察位），而会话分发只拿得到
            // `EditorSession`（它到不了 owner）。这里已经有 `link.owner()`，
            // 与 `set_transport` 取宿主播放控制是同一条路。
            if crate::capture::commands::is_capture_command(command) {
                let link=state.borrow().link.clone();
                let outcome=(||->Result<serde_json::Value,String>{
                    let owner=link.owner()?;
                    crate::capture::commands::handle(&owner,command,&request["args"])
                })();
                let reply=match outcome {
                    Ok(value)=>serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":true,"value":value}),
                    Err(error)=>serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}),
                };
                state.borrow_mut().post(reply);
                return;
            }
            if command=="set_transport" && request["args"]["playheadSec"].is_number() {
                let link=state.borrow().link.clone();
                let result=(||->Result<(),String> {
                    let position=request["args"]["playheadSec"].as_f64().ok_or("invalid seek position")?;
                    let owner=link.owner()?;
                    owner.host_playback().ok_or("host does not provide ARA playback control")?
                        .set_position(position).map_err(|e|e.to_string())?;
                    // seek回执前同步真实宿主光标，避免继续发布上次UI tick的旧位置。
                    owner.refresh_host_transport();Ok(())
                })();
                if let Err(error)=result {
                    state.borrow_mut().post(serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}));return;
                }
            }
            // 慢车道的命令不占那 32 个在途名额，也不按 30 秒判"没有回复"（见
            // `SLOW_PENDING_TTL`）。判据只有一处来源：`commands::lane`。
            let slow=crate::editor::commands::is_slow(command);
            let (link,sink,full)={let mut state=state.borrow_mut();(state.link.clone(),state.sink.clone(),state.pending_full(id,slow))};
            let history_context=link.owner().ok().and_then(|owner|{
                let document=owner.editor_document().ok()?;let host=owner.project_history_host()?;
                let lease=link.authorize(&document).ok()?;Some((document,host,lease))
            });
            let mut undo_started=false;
            if let Some((document,host,lease))=&history_context {
                state.borrow_mut().undo_document=Some(Arc::downgrade(document));
                if command=="begin_undo_group" {document.host_undo.begin_group(&view_id);}
                if command=="end_undo_group" {document.host_undo.end_group(&view_id);}
                if crate::editor::commands::mutates_audio(command) {
                    let checkpoint=request["args"]["checkpoint"].as_bool().unwrap_or(true);
                    match document.host_undo.begin_request(&view_id,id,checkpoint,false,host,&||link.authorize(document).is_ok_and(|current|current==*lease)) {
                        Ok(())=>undo_started=true,
                        Err(error)=>{state.borrow_mut().post(serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,"error":error}));return;},
                    }
                }
            }
            let outcome=if full {Err("native pending request budget exceeded".into())} else {
                link.owner().and_then(|owner|owner.editor_session()?.enqueue(crate::editor::session::UiRequest {
                    id,command:command.into(),args:request.get("args").cloned().unwrap_or_else(||serde_json::json!({})),sink,link:Some(link.clone()),
                }))
            };
            match outcome {
                Ok(())=>{let mut state=state.borrow_mut();state.mark_pending(id,slow);if undo_started {if let Some((document,_,_))=&history_context {state.undo_requests.insert(id,Arc::downgrade(document));}}return;},
                Err(error)=>{
                    if let Some((document,_,_))=&history_context {
                        if undo_started {document.host_undo.finish_request(&view_id,id);}
                        if command=="begin_undo_group" {document.host_undo.end_group(&view_id);}
                    }Err(error)
                },
            }
        },
        None=>Err(String::from("native command missing")),
    };
    let response = match result {
        Ok(value) => {
            serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":true,"value":value})
        }
        // `errorKind` 让前端能区分"暂时失败"与"会话已终结"。后者必须让前端
        // **停止轮询**并如实显示，而不是每 250 ms 重问一个永远不会好的问题
        // （见 `editor::classify_error` 的说明）。
        Err(error) => serde_json::json!({"version":1,"viewId":view_id,"id":id,"ok":false,
            "errorKind":crate::editor::classify_error(&error),
            "errorCode":crate::editor::errors::terminal_error_code(&error),"error":error}),
    };
    state.borrow_mut().post(response);
}
#[cfg(test)]
mod pending_budget_tests {
    use super::*;

    /// **不变量：慢车道不占那 32 个在途名额。**
    ///
    /// 【为什么这是回归测试】那 32 个名额护的是"宿主写入的回执没回来"（每条都持有
    /// 一份宿主 Undo 块）。慢任务不持有任何宿主写入，而它跑着的时候编辑器还在轮询 ——
    /// 让轮询把它挤出预算，正是要修的那个故障。
    #[test]
    fn a_slow_request_does_not_consume_the_fast_budget() {
        assert!(
            budget_exhausted(PENDING_BUDGET, false, false),
            "快车道满了就该拒"
        );
        assert!(
            !budget_exhausted(PENDING_BUDGET, false, true),
            "慢车道不占那 32 个名额"
        );
        assert!(
            budget_exhausted(PENDING_BUDGET - 1, true, false),
            "同一个 id 重复在途必须拒（否则回执会串号）"
        );
        assert!(!budget_exhausted(0, false, false));
    }

    /// **不变量：慢请求拿到分钟级预算，而不是 30 秒。**
    ///
    /// 【为什么】`PENDING_TTL` 回答的是"正常的宿主回流多久没回就该判丢"。用同一个数字
    /// 去量一次 31–134 秒的基准测试，只会把"正在跑"判成"回执丢了"，并打出那条误导性的
    /// `expired without a reply`。
    #[test]
    fn a_slow_request_gets_the_long_budget() {
        let fast = Pending {
            since: std::time::Instant::now(),
            slow: false,
        };
        let slow = Pending {
            since: std::time::Instant::now(),
            slow: true,
        };
        assert_eq!(fast.ttl(), PENDING_TTL);
        assert_eq!(slow.ttl(), SLOW_PENDING_TTL);
        assert!(
            SLOW_PENDING_TTL >= PENDING_TTL * 10,
            "慢车道的预算必须是分钟级，否则等于没改"
        );
    }
}

#[cfg(test)]
mod backoff_tests {
    use super::*;

    /// **不变量：重投必须退避且封顶。**
    ///
    /// 【为什么这是回归测试】回复泵由 20 ms 定时器驱动，此前只要等待判据仍成立就
    /// **每一 tick 重投一次**（50 Hz，直到 30 s 的 `until`）。一个永不收敛的等待会变成
    /// ~1500 次 `get_timeline_state`，每次都与渲染线程争同一把文档事务。退避必须
    /// 单调不减、首档与定时器同频（不牺牲正常情况的响应），并封顶。
    #[test]
    fn host_reply_retry_backoff_grows_and_is_capped() {
        let first = host_reply_retry_backoff(1);
        assert_eq!(
            first,
            std::time::Duration::from_millis(20),
            "第一档保持与 20 ms 定时器同频，正常情况不受影响"
        );
        let mut previous = std::time::Duration::ZERO;
        for attempt in 1..=HOST_REPLY_RETRY_BUDGET {
            let delay = host_reply_retry_backoff(attempt);
            assert!(delay >= previous, "退避不得回退（attempt={attempt}）");
            assert!(
                delay <= std::time::Duration::from_millis(500),
                "退避必须封顶，否则收敛会慢到不可接受（attempt={attempt}）"
            );
            previous = delay;
        }
    }
}

#[cfg(test)]
mod bootstrap_tests {
    use super::BootstrapFlags;

    /// `trackGrouping` 不随任何宿主能力变化 —— 它守卫的是纯插件的参数分组。
    ///
    /// 【回归】它此前被写成复用 `audioImport`，于是宿主缺少媒体 API 时，插件里
    /// 本不需要任何宿主写能力的轨道拖动会无声失效。
    #[test]
    fn track_grouping_does_not_depend_on_host_media_apis() {
        for media_import in [true, false] {
            let flags = BootstrapFlags {
                capture_caps: None,
                transport_control: false,
                fade_axes: None,
                history_backend: "local",
                host_kind: "ara_only",
                host_capabilities: hifishifter_kernel::editor::host::HostCapabilities {
                    media_import,
                    ..hifishifter_kernel::editor::host::HostCapabilities::ara_only()
                },
            };
            assert_eq!(
                flags.to_json("view-7")["trackGrouping"],
                serde_json::json!(true),
                "mediaImport={media_import} 时 trackGrouping 被错误地关掉了"
            );
        }
    }

    /// 宿主能力标志必须**原样**透传：前端按它们决定开放哪些操作。
    #[test]
    fn host_capabilities_pass_through_unchanged() {
        let flags = BootstrapFlags {
            capture_caps: None,
            transport_control: true,
            fade_axes: Some("legacy"),
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities {
                split: true,
                media_import: true,
                track_create: true,
                ..hifishifter_kernel::editor::host::HostCapabilities::ara_only()
            },
        };
        let json = flags.to_json("view-42");
        assert_eq!(json["version"], serde_json::json!(1));
        assert_eq!(json["viewId"], serde_json::json!("view-42"));
        assert_eq!(json["transportControl"], serde_json::json!(true));
        assert_eq!(json["fadeAxes"], serde_json::json!("legacy"));
        // 后端身份与能力位必须原样透传：前端据此显示只读横幅。
        assert_eq!(json["hostKind"], serde_json::json!("ara_only"));
        assert_eq!(
            json["hostCapabilities"]["clipWrite"],
            serde_json::json!(false)
        );
        assert_eq!(
            json["hostCapabilities"]["clipGeometry"],
            serde_json::json!(true)
        );
        assert_eq!(json["hostCapabilities"]["split"], serde_json::json!(true));
        assert_eq!(
            json["hostCapabilities"]["mediaImport"],
            serde_json::json!(true)
        );
        assert_eq!(
            json["hostCapabilities"]["trackCreate"],
            serde_json::json!(true)
        );
    }

    /// 写能力只有**一个**来源：顶层不得再出现 `clipEditing` / `audioImport` 这类副本。
    ///
    /// 【回归】两处并存时（一处按本实例原生扩展、一处按 `project_history_host()`）
    /// 必然漂移，而"哪一份算数"没人说得清。
    #[test]
    fn host_write_capabilities_have_a_single_source() {
        let flags = BootstrapFlags {
            capture_caps: None,
            transport_control: false,
            fade_axes: None,
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities::ara_only(),
        };
        let json = flags.to_json("v");
        for stale in [
            "clipEditing",
            "clipSplitting",
            "clipClipboard",
            "audioImport",
            "trackCreation",
        ] {
            assert!(
                json.get(stale).is_none(),
                "顶层不应再出现 `{stale}`：写能力只由 `hostCapabilities` 提供"
            );
        }
    }

    /// `trackCreate` 与 `mediaImport` 是独立能力：导入音频蕴含建轨，建轨不蕴含导入。
    ///
    /// 【为什么单独钉住】合成一个标志会让"添加轨道"随"能不能导入音频"一起开关 ——
    /// 而这两件事的门槛不同（建轨还需要 `InsertTrackInProject` / FX 接口）。
    #[test]
    fn track_creation_is_independent_of_audio_import() {
        let flags = BootstrapFlags {
            capture_caps: None,
            transport_control: false,
            fade_axes: None,
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities {
                media_import: false,
                track_create: true,
                ..hifishifter_kernel::editor::host::HostCapabilities::ara_only()
            },
        };
        let json = flags.to_json("v");
        assert_eq!(
            json["hostCapabilities"]["mediaImport"],
            serde_json::json!(false)
        );
        assert_eq!(
            json["hostCapabilities"]["trackCreate"],
            serde_json::json!(true)
        );
    }

    /// 未知轴语义必须如实报 `null`，前端据此保持只读（不猜）。
    #[test]
    fn an_unknown_fade_axis_set_is_reported_as_null() {
        let flags = BootstrapFlags {
            capture_caps: None,
            transport_control: false,
            fade_axes: None,
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities::ara_only(),
        };
        assert_eq!(flags.to_json("v")["fadeAxes"], serde_json::Value::Null);
    }

    /// 新轴宿主如实报 `continuous`：前端据此只给连续滑杆、不摆预设按钮。
    #[test]
    fn a_continuous_fade_axis_host_is_reported_as_such() {
        let flags = BootstrapFlags {
            capture_caps: None,
            transport_control: false,
            fade_axes: Some("continuous"),
            history_backend: "local",
            host_kind: "ara_only",
            host_capabilities: hifishifter_kernel::editor::host::HostCapabilities::ara_only(),
        };
        assert_eq!(
            flags.to_json("v")["fadeAxes"],
            serde_json::json!("continuous")
        );
    }
}

#[cfg(test)]
mod group_id_tests {
    use super::host_group_id_for;
    use crate::host::host_edit::HOST_GROUP_ID_MAX;

    fn raw_first_four(key: &str) -> u32 {
        let digest = blake3::hash(key.as_bytes());
        u32::from_le_bytes(digest.as_bytes()[..4].try_into().unwrap())
    }

    /// 编组 id 必须落在宿主接受区间内 —— 旧实现 `& 0x7fff_ffff` 会给出到 `2^31-1`
    /// 的值，约 7% 的编组被 `host_edit` 的 "invalid host item group id" 整条拒绝。
    #[test]
    fn host_group_ids_stay_inside_the_host_accepted_range() {
        let mut highest = 0_i32;
        for i in 0..20_000 {
            let id = host_group_id_for(&[format!("clip-{i}")]);
            assert!((1..=HOST_GROUP_ID_MAX).contains(&id), "越界 id: {id}");
            highest = highest.max(id);
        }
        // 分布应覆盖到区间上半，否则本测试没证明折进区间后仍用满了熵。
        assert!(
            highest > HOST_GROUP_ID_MAX / 2,
            "取值上界偏低，折进区间可能压掉了熵：{highest}"
        );
    }

    /// 直接钉住回归：样本里存在旧实现会拒绝的取值，新实现必须把它们收进区间。
    #[test]
    fn values_the_old_mask_would_have_rejected_are_now_in_range() {
        let mut rejected_by_old = 0;
        for i in 0..20_000 {
            let key = format!("clip-{i}");
            if raw_first_four(&key) & 0x7fff_ffff > HOST_GROUP_ID_MAX as u32 {
                rejected_by_old += 1;
                let id = host_group_id_for(&[key]);
                assert!((1..=HOST_GROUP_ID_MAX).contains(&id), "仍越界: {id}");
            }
        }
        assert!(
            rejected_by_old > 0,
            "样本里应存在旧实现会拒绝的取值，否则本测试没覆盖到回归"
        );
    }

    /// 同一选择必须恒等映射：重复编组不得不断改号。
    #[test]
    fn the_same_selection_always_maps_to_the_same_group() {
        let first = host_group_id_for(&["b".into(), "a".into()]);
        let second = host_group_id_for(&["b".into(), "a".into()]);
        assert_eq!(first, second);
    }
}
