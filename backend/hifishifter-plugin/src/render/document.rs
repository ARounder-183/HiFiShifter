//! 按真实 controller 身份索引文档生命周期，不从最后一个创建的文档猜测归属。

use super::extension::ExtensionOwner;
use super::ownership::DocumentId;
use crate::sync::{try_lock_or_recover, MutexExt};
use ara2_bridge::companion::CompanionControllerBinding;
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::ExtensionControllerLease;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

thread_local! {
    static NONBLOCKING_HOST_QUERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Timer-driven readback may skip a busy frame; explicit synchronous commands
/// retain their existing transaction semantics outside this scope.
pub(crate) struct HostQueryScope {
    previous: bool,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl HostQueryScope {
    pub(crate) fn nonblocking() -> Self {
        Self {
            previous: NONBLOCKING_HOST_QUERY.with(|flag| flag.replace(true)),
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for HostQueryScope {
    fn drop(&mut self) {
        NONBLOCKING_HOST_QUERY.with(|flag| flag.set(self.previous));
    }
}

struct RendererLease {
    lease: ExtensionControllerLease,
    _companion: Option<CompanionControllerBinding<'static>>,
    owner: Weak<ExtensionOwner>,
}

/// 内嵌GUI同进程冻结值；PCM沿授权Arc共享，不能经旧IPC/JSON复制完整长源。
pub(crate) struct WorkspaceSnapshot {
    pub timeline: hifishifter_kernel::state::TimelineState,
    pub sources: Vec<(String, Arc<super::source::SourcePcm>)>,
    pub revision: u64,
    pub model_revision: u64,
    pub audio_revision: u64,
}

/// 编辑器**读到过**的那一份工作区视图：指纹 + 逐根轨参数。
///
/// 【为什么是值而不是键】它是"编辑器看到的是什么"的唯一陈述。旧实现把它做成一张
/// 按指纹索引的表（`last_workspace_views`），于是"这次编辑基于哪份视图"退化成一次
/// 可能落空的查表 —— 而落空的代价是用户这一次编辑被整条丢弃。值不可能落空。
///
/// 【为什么 params 用 Arc】一份根轨数组可能有几十万帧；编辑器每次读都会克隆一份视图，
/// 而提交闸只读不写。Arc 让"记录视图"与"提交时读基线"共享同一份内存。
#[derive(Debug)]
pub(crate) struct WorkspaceView {
    /// 与 `workspace_projection_locked` 同一口径的指纹：只用于相等判定与日志。
    pub projection: String,
    /// 逐根轨参数：`edits.apply` 之后、编辑器真正拿到的那一份（文档 id 空间）。
    pub params:
        Arc<std::collections::BTreeMap<String, hifishifter_kernel::state::TrackParamsState>>,
    /// 这份视图对应的 UI 几何代次。提交时与当前代次不一致就走重基/重试。
    pub geometry_revision: u64,
}

/// ARA 文档存档通道的**读数**。
///
/// 【为什么需要它】这条通道的四个回调此前只有 `log::info!`，而 release 的默认级别是
/// `Warn` —— 用户机器上一条都不落盘。于是"宿主到底有没有走存档通道"这个问题在真机上
/// **无法回答**：修法完全取决于它（存档通道 ≠ VST3 组件状态通道），而证据链在第一步
/// 就断了。这里把同一批事实记成读数，让它同时出现在日志（限流 WARN）与诊断包里。
///
/// 【为什么在文档上】`storeDocumentToArchive` / `beginRestoringDocumentFromArchive`
/// 是**文档控制器**的回调，它们的读数天然是文档级的。
#[derive(Clone, Debug, Default)]
pub(crate) struct ArchiveReading {
    /// 宿主最近一次交回来的字节数。
    restored_bytes: usize,
    /// 我们最近一次交给宿主的字节数。
    stored_bytes: usize,
    /// 最近一次保存是否走了「交回旧载荷」那条分支（I3），以及累计次数。
    unmerged_returns: u32,
    /// 最近一次"存档已合入"的单调毫秒（0 = 还没合入过）。
    merged_at_ms: u64,
}

impl ArchiveReading {
    /// 诊断包 / 状态栏读的 JSON 形状。
    pub(crate) fn json(&self, pending: bool, pending_reason: Option<String>) -> serde_json::Value {
        serde_json::json!({
            "restoredBytes": self.restored_bytes,
            "storedBytes": self.stored_bytes,
            "unmergedReturns": self.unmerged_returns,
            "mergedAtMs": self.merged_at_ms,
            "pending": pending,
            "pendingReason": pending_reason,
        })
    }
}

/// "你的这次编辑有没有被采纳"的**读数**。
///
/// 【为什么必须有】§1.3 列出的 11 条回退路径里，9 条**连字符串都没有**：写回被拒只留
/// 一行限流日志，静默丢轨连日志都没有。用户唯一能做的事是"再画一次试试"，而这恰好
/// 与"数据被抹平"最难区分。这里把同一批事实记成读数，让它进状态快照与诊断包。
///
/// 【为什么不再有"捕获读数"】那一栏统计的是图集 `capture()` 拒绝写回的曲线条数。
/// 曲线现在只由显式写入（`set_param_frames`）产生，写入失败会直接变成一条命令
/// 错误 —— 不再存在"悄悄没写进去"的中间态，因此这一栏没有可统计的对象。
#[derive(Clone, Debug, Default)]
pub(crate) struct EditHealth {
    /// 累计"这次编辑没有被采纳"的次数（重放预算耗尽）。
    dropped_edits: u64,
    /// 最近一次被丢弃的编辑的可读原因。
    last_drop: Option<String>,
}

impl EditHealth {
    /// 记一次"这次编辑没有被采纳"（重放预算耗尽）。
    fn drop_edit(&mut self, reason: String) {
        self.dropped_edits = self.dropped_edits.saturating_add(1);
        self.last_drop = Some(reason);
    }

    /// 状态快照 / 诊断包的 JSON 形状。
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "droppedEdits": self.dropped_edits,
            "lastDrop": self.last_drop,
        })
    }
}

/// 一次存档合入的结论。
///
/// 【为什么要把"还在等"与"失败"分开】宿主的图在重开工程时是**逐块**长出来的：
/// 存档到达时 region 往往还没交付。那时身份对不上不是失败，而是"还没长成"。
/// 两种结论的处置相反：一个必须留着载荷再试，一个必须报出来。
pub(crate) enum RestoreMerge {
    /// 已合入（`edits` 已更新）。
    Merged,
    /// 还缺东西（宿主图未重建 / fade 元数据在途），载荷必须留着下次再试。
    Waiting(String),
}

/// 一个 ARA 文档的共享状态。
///
/// # 锁序（必须遵守）
///
/// 多把锁同时持有时，一律按下面的顺序获取；**反向获取会死锁**：
///
/// 1. `transaction` —— 最外层：几何/编辑的短事务
/// 2. `timeline`
/// 3. `edits`
/// 4. 宿主事实：`ui_tracks` → `host_folder_children` / `clip_ids` / `regions` /
///    `region_items` / `authorized_takes` / `authorized_sources` / `render_facts`
/// 5. `region_owners()`（进程级全局表）—— 最内层叶子锁
///
/// 例外与约束：
/// - `region_owners()` 是**叶子**：持有它时不得再取上面任何一把锁。它出现在
///   `ara/model.rs` 的回调里（不持文档锁）与 `render/extension.rs` 的查询里
///   （在 `transaction` 之内）—— 两者都满足"最后获取"。
/// - `RenderFacts` 采样（`refresh_ui_inventory`）在 `transaction` 之内、且在其它
///   文档锁之外，因此它与 `workspace_projection_locked` 里对 `render_facts` 的读取
///   不会互相阻塞到死锁。
/// - 单把锁内的长操作（宿主 API 调用、神经推理）必须先把数据 `clone` 出来再解锁 ——
///   见 `render/extension.rs` 的"宿主调用始终不持内部锁"。
///
/// 宿主内容变化后等待其收口的宽限期。
///
/// 与 bridge 的 `DEFERRED_EDITING_GRACE` 同量级：一次编辑从内容回调到
/// `endEditing` 通常在毫秒级；超过这个窗口仍未收口，就按"宿主不再收口"处理 ——
/// 保留最后一份已发布的时间线继续刷新界面，而不是把窗口永久锁在冲突状态上。
pub(crate) const EDIT_SETTLE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// 插件侧强制收口一次**从未关闭**的编辑会话的宽限期。
///
/// 【为什么比 `EDIT_SETTLE_GRACE` 长得多】那一条判的是"宿主已经发过 `endEditing`，
/// 只是被 reader 挡住了"——收敛是必然的，短一点没有风险。这一条判的是"宿主**根本
/// 没打算收口**"，而合法的编辑会话可以持续很久：宿主通常在鼠标按下时
/// `beginEditing`、抬起时 `endEditing`，一次慢速拖拽就能超过 2 秒。用 2 秒去强制
/// 收口，等于在用户拖动到一半时把模型提交一次 —— 曲线会跳、撤销栈会碎。
///
/// 取 8 秒：足够覆盖任何正常交互，又短到"宿主真的挂了"时用户不必等太久。
pub(crate) const FORCED_EDIT_CLOSE_GRACE: std::time::Duration = std::time::Duration::from_secs(8);

/// 一个 ARA 音频源相对本插件的**可读性**。
///
/// 【为什么是三态而不是一个布尔】三种"没有 PCM"的处置完全不同：
/// * `Published` —— 有 PCM，正常；
/// * `Revoked` —— 宿主暂时收回了样本访问（或把源放进了 undo 历史）。**会自己好**：
///   下一次授权 / 插件侧重试都能恢复，界面上不该说成"等待分配"；
/// * `Failed` —— 授权在，但读取失败（真故障，需要留真因）。
///
/// 把它们压成一个布尔，正是"后台切回来之后 Clip 一直等宿主音频"看起来像
/// "宿主还在分配 region"的原因。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceState {
    /// 已读到并发布了 PCM。
    Published,
    /// 宿主暂时不可读（样本访问被收回 / 源被停用），但**已发布的 PCM 仍然保留**。
    Revoked,
    /// 授权在、读取失败。
    Failed,
}

impl SourceState {}

/// 输入可用性的**综合判定**。
///
/// 【为什么不是一个布尔】`ready` 只回答"渲染许可恢复过没有"，它不回答"PCM 到底在不在"。
/// 于是"宿主授权了 0 个源"与"一切正常"在插件自己看来是同一种状态，界面也只能说一句
/// 含糊的"等待宿主音频"。这个判据把四种处境分开，每一处都对应一个不同的下一步。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputVerdict {
    /// 有 region、有授权、有已发布的 PCM。
    Ready,
    /// 宿主还没交付任何音频源（还没分配 region / 还没建 source）。
    AwaitingRegions,
    /// 源有了，但样本访问没开、或还没读出来。
    AwaitingSamples,
    /// 授权在却读不出来，或部分源失败 —— 可行动的真故障。
    Degraded,
}

impl InputVerdict {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::AwaitingRegions => "awaiting_regions",
            Self::AwaitingSamples => "awaiting_samples",
            Self::Degraded => "degraded",
        }
    }

    /// 由四个计数判定"输入到底可不可用"。
    ///
    /// 【为什么抽成纯函数】这条判据是"参数线画完回弹 / 操作像没生效"的总开关：
    /// 编辑器把 `verdict != Ready` 当成"素材还没到"，于是所有依赖素材的动作都被
    /// 按"等一个 tick 就会好"处理。它必须能被单测直接钉住，而不是只能起一个宿主
    /// 去观察。
    ///
    /// 【为什么 `usable` 排在 `authorized` 前面】两个字段回答的是不同的问题：
    ///   * `authorized` —— 宿主**此刻允不允许**读 ARA 样本；
    ///   * `usable`    —— 我们手里**有没有能出声的 PCM**（`edit_sources`）。
    ///
    /// REAPER-Linux 的 `[editor, view]` 形态下前者可以是恒 false，而渲染侧照样
    /// 拿着 `edit_sources` 出声。用前者判定，编辑器就被永久钉在
    /// `awaiting_samples` 上，而同一个文档在渲染侧看是 `pcm=1` —— 两份真相。
    ///
    /// 判据的顺序因此是：先问"能不能出声"，再问"卡在哪一步"。
    pub(crate) fn decide(sources: usize, published: usize, failed: usize, usable: usize) -> Self {
        if usable > 0 || published > 0 {
            // 有能出声的 PCM 就算可用；部分源失败只降级，不把整条链判成"没素材"。
            return if failed > 0 {
                Self::Degraded
            } else {
                Self::Ready
            };
        }
        if failed > 0 {
            return Self::Degraded;
        }
        if sources == 0 {
            return Self::AwaitingRegions;
        }
        Self::AwaitingSamples
    }
}

/// 一次输入可用性的完整读数（见 [`InputVerdict`]）。
///
/// 【为什么带这么多计数】用户报障时最需要回答的是"卡在哪一步"。只给一个 verdict，
/// 下一轮仍然只能靠猜；把"几条 region、几个源、几个授权、几个已发布、几个失败"
/// 一起给出来，一行日志就能指认。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InputReadiness {
    pub regions: usize,
    pub regions_with_source: usize,
    pub sources: usize,
    pub sources_authorized: usize,
    pub sources_published: usize,
    pub sources_failed: usize,
    pub model_published: bool,
    pub verdict: InputVerdict,
}

impl InputReadiness {
    /// 一行可读的读数（诊断包与 `stuck_report` 共用）。
    pub(crate) fn summary(&self) -> String {
        format!(
            "verdict={} regions={}/{} sources={} authorized={} published={} failed={} model={}",
            self.verdict.as_str(),
            self.regions_with_source,
            self.regions,
            self.sources,
            self.sources_authorized,
            self.sources_published,
            self.sources_failed,
            self.model_published,
        )
    }
}

/// 宿主交给本文档的播放控制租约。
///
/// 【为什么线程要**跟着租约**存，而不是进程级记一份】`PlaybackRequestHandle` 的
/// 守卫比较的是「创建这个句柄的那条线程」：
///
/// ```text
/// if endpoint.thread != current().id() { return Err(InvalidState(...)) }
/// ```
///
/// 而一条进程里可以有多个 ARA 文档（多工程 / 多实例），每个文档控制器都可能在
/// **不同**的线程上创建。此前那份 `OnceLock<ThreadId>` 只记住第一次见到的那条，
/// 于是第二个文档的播放请求会被投到一条与它无关的线程上 —— 永远不会被执行。
/// 让线程随租约走，这两个事实就不可能再分叉。
#[derive(Clone)]
pub struct PlaybackEndpoint {
    /// 宿主创建这条租约时所在的线程（= 句柄守卫要求的那条）。
    pub thread: std::thread::ThreadId,
    /// 租约本体（`Clone`：每次调用取一份副本）。
    pub handle: ara2_bridge::plugin::PlaybackRequestHandle,
    /// 本端点的传输请求**票据域**（`editor::host_threads` 的取代判定只在本域内
    /// 生效，且**传输与定位分票**：SetPosition 不得作废排队中的 Start/StopAtPosition）。
    ///
    /// 【为什么随端点而不是进程】进程级票号下，任何文档的任意一次投递都会把
    /// **别的**文档排队中的请求判成「已被取代」而静默丢弃（多工程 tab / 双实例
    /// 时「按了播放没反应」）。票据域随端点走，取代语义才与用户意图同域；
    /// 域内传输/定位分票的完整规则见
    /// [`editor::host_threads::SupersessionDomain`]。
    pub supersession: std::sync::Arc<crate::editor::host_threads::SupersessionDomain>,
}

/// 因"两份输入不同代"而推迟发布的上界。
///
/// 【为什么是 2 s】拖动一个 item 期间，清单每 100 ms 就可能发布一代新的，于是
/// "输入动过 ⇒ 再等等"会一直成立。2 s 足够覆盖任何一次真实的两份输入错位
/// （现场日志里是 40~120 ms），又远短于用户能接受"我拖完了它还没更新"的时间。
/// 预算耗尽后照常发布（不能饿死界面），但会把两个代次写进 WARN。
const ASSEMBLY_RETRY_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);

/// "这一轮的 `unbound` 该不该推迟发布" —— 纯判据（可单测）。
///
/// 参数是四个**已经读到**的事实，因此它不碰锁、不碰宿主、不看时间以外的任何状态：
///
/// * `unbound` —— 这一轮没挂上任何 item 的 region 条数；
/// * `consumed` —— 上一次装配尝试消费的那一对输入代次（`None` = 还没有过上一次）；
/// * `inputs` —— 这一轮读到的代次；
/// * `deferred_since` —— 当前这串推迟的起点（`None` = 没有在推迟）；
/// * `now` —— 由调用方给出，测试因此不必睡觉。
///
/// 【三条判据，每条都对应一次真实的误判】
///
/// 1. `unbound == 0` ⇒ 不推迟。这一轮就是答案，推迟只会白白晚一帧；
/// 2. `consumed == Some(inputs)` ⇒ 不推迟。两份输入与上一次尝试**逐位相同**，再试一次
///    只会得到同一个结论 —— 此时推迟是纯损失（界面停在旧时间线上）；
/// 3. `consumed == None` ⇒ 不推迟。没有上一次可比，"输入动过"无从谈起；而第一轮装配
///    必须产出点东西（否则冷启动永远空白）。
///
/// 剩下的唯一情形 —— `unbound > 0` 且输入动过 —— 才是"两份输入可能还没对齐"。
/// 预算（[`ASSEMBLY_RETRY_BUDGET`]）封住上界：拖动 item 期间清单每一代都在变，
/// 没有预算就会一直推迟到用户松手。
pub(crate) fn should_defer_assembly(
    unbound: usize,
    consumed: Option<(u64, u64)>,
    inputs: (u64, u64),
    deferred_since: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    if unbound == 0 {
        return false;
    }
    let Some(consumed) = consumed else {
        return false;
    };
    if consumed == inputs {
        return false;
    }
    deferred_since.is_none_or(|since| now.duration_since(since) < ASSEMBLY_RETRY_BUDGET)
}

/// 一次装配尝试的结局。
///
/// 【为什么不是一个 `bool`】"推迟发布"与"发布了"必须被**区别对待**：推迟时
/// `assembly_dirty` 要留着（下一个安全点再来），发布时它要清掉。用布尔表达这件事，
/// 唯一可能的写法是"调用方再去读一次标记" —— 而两次读写之间正好是产生新竞态的地方。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AssemblyOutcome {
    /// 这一轮的输入被消费了：时间线已发布（或确认无需改动）。
    Published,
    /// 两份输入还没对齐：**保留上一轮已发布的时间线**，下一个安全点重试。
    Deferred,
}

/// 一条"item 已在宿主里建成、但 ARA 还没认领"的导入。
///
/// 【为什么需要一张独立的账】导入命令能观测到的事实只有一条：宿主给了我们一个 item
/// GUID。它什么时候被 ARA 图认领，由**宿主的**编辑会话决定 —— 现场实测 125~299 ms，
/// 而命令的回执只有几十毫秒的生命周期。把两件事绑在同一个超时上，唯一的结局是把
/// 宿主的调度延迟翻译成我们的失败率（用户看到的就是"导入成功了一半"）。
///
/// 这张表是"谁还没到"的唯一账本：导入时登记，任何一轮 ARA 图发布都来对一次账，
/// 对上了就转正并划掉。
pub(crate) struct PendingImport {
    /// 宿主 item GUID。
    pub item: String,
    /// 导入时写下的源文件路径（波形授权判定要用，见 `pending_import_for_source`）。
    pub source_path: String,
    /// 落点轨道（新建轨时为采纳后的清单 guid）。
    pub landed_track: Option<String>,
    /// 登记时刻：`after_ms` 与绊线都从它算。
    pub requested_at: std::time::Instant,
    /// 经过了多少轮发布。宿主拖得久时这个数会大 —— 它是唯一能回答"等了多久"的读数。
    pub rounds: u32,
    /// 已经报过 `stalled`（每一条只报一次）。
    pub stall_reported: bool,
}

/// 一次转正：待定导入终于被某一轮 ARA 图（或宿主清单）认领。
pub(crate) struct ResolvedImport {
    pub item: String,
    /// 导入时写下的源文件路径（日志与前端都可能要用；它来自导入命令，不是宿主）。
    pub source_path: String,
    /// **未加命名空间**的 clip id（`ara-item-{guid}`）；调用方按自己的 namespace 加前缀。
    pub clip_id: String,
    pub landed_track: Option<String>,
    pub rounds: u32,
    pub elapsed: std::time::Duration,
}

/// 两个源路径是不是同一个文件。
///
/// 【为什么要一个比较函数】路径来自两条不同的通路：用户选中的路径（导入命令）与宿主
/// 回报的 persistentID（ARA / 清单）。在 Windows 上两者的大小写与分隔符都可能不同，
/// 而这条判据只决定"这次波形请求该不该等"，放宽一点是安全的 —— 它只能放宽到
/// **我们自己登记过的那几个文件**。
fn same_source_path(left: &str, right: &str) -> bool {
    fn normalised(value: &str) -> String {
        value.replace('/', "\\").to_ascii_lowercase()
    }
    !left.is_empty() && !right.is_empty() && normalised(left) == normalised(right)
}

/// 一条 region 连续多少轮未认领才值得 WARN。
///
/// 【为什么是 3】ARA 的认领是分轮的，现场实测一轮的间隔是几十毫秒、正常中间态在
/// 213 ms 内自愈；3 轮已经覆盖"宿主慢一拍"，又不至于把真正挂不上的 region 拖到
/// 用户看不见。低于它的一律 DEBUG。
const UNBOUND_WARN_STREAK: u32 = 3;

/// 对账所需的**事实快照**：由调用方在自己取锁之前/之后算好，作为参数传进来。
///
/// 【为什么必须是数据而不是方法】上一版的对账会去取 `region_items` / `clip_ids` /
/// `ui_tracks` 三把锁，而它的调用方 `present_host_inventory` 从头到尾持着 `ui_tracks`。
/// `std::sync::Mutex` 不可重入，于是"对账"这件小事把整个宿主卡死 —— 现场表现是
/// REAPER 在导入成功之后整体未响应（`AppHangB1`），而状态栏那条
/// `plugin_get_apply_state` 只是宿主线程停止泵消息 30 s 之后才浮现的读数。
///
/// 把输入变成数据之后，对账器**结构上不可能**再取锁：它手里没有表，只有一份快照。
/// 这类缺陷不该靠"记得别这么写"来防。
pub(crate) struct ImportClaims {
    /// 本轮宿主清单里的 item GUID（`present_host_inventory` 手上那份 `ui_tracks` 的投影）。
    pub present_items: std::collections::BTreeSet<String>,
    /// region key → 文档 clip id（`PresentedIds` 的同一份快照）。
    pub clip_ids: std::collections::HashMap<u64, String>,
    /// region key → item GUID（`region_items` 的只读克隆）。
    pub region_items: std::collections::HashMap<u64, String>,
}

impl ImportClaims {
    /// item GUID → 当前时间线里的 clip id（**不带命名空间**，命名空间由调用方加）。
    ///
    /// 【两条判据、一个顺序】活着的 region 认领它 ⇒ 有源；否则退到宿主清单 ⇒ 至少
    /// 画得出占位。这与 `clip_for_host_item` / `gui_clip_for_host_item` 是同一条边。
    fn clip_for_item(&self, item: &str) -> Option<String> {
        if let Some((key, _)) = self
            .region_items
            .iter()
            .find(|(_, bound)| bound.as_str() == item)
        {
            if let Some(clip) = self.clip_ids.get(key) {
                return Some(clip.clone());
            }
        }
        self.present_items
            .contains(item)
            .then(|| format!("ara-item-{item}"))
    }
}

/// 一次自动应用（`apply_workspace_edits_with_progress`）的结论。
///
/// 【为什么要多一个值】"发布了"与"这一轮没有可渲染内容"必须分开：后者既不能写
/// `applied`（写了下一轮就不再重试，宿主下一次发布真模型时没人接），也不能算错误
/// （那正是状态栏闪红字的来源）。见 [`crate::render::input::RenderOutcome`]。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApplyOutcome {
    /// 至少发布了一份快照。
    Published,
    /// 这一轮没有可渲染的 clip：**什么都不发布**，下一轮再来。
    ///
    /// `report` 非空 = 连续为空刚刚跨过预算，这一次值得说给用户听（一次性）。
    Nothing { report: Option<EmptyStreak> },
}

/// 连续多少个"空轮"值得说给用户听（≈ 10 s @ 4 Hz，也覆盖一次 Click 的最坏节奏）。
const EMPTY_ROUNDS_MAX: u32 = 40;

/// 连续为空的墙钟上限：到点就说一次，不再等轮数。
const EMPTY_BUDGET_MS: u64 = 10_000;

/// 一次"空轮"的累积读数（由 [`DocumentSession::note_empty_round`] 给出，**只在跨过预算
/// 的那一轮返回一次**）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmptyStreak {
    pub rounds: u32,
    pub elapsed_ms: u64,
}

/// 墙钟毫秒：只服务"连续为空多久了"这条预算，不参与任何音视频时序。
fn wall_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Default)]
pub struct DocumentSession {
    controller: AtomicUsize,
    alive: AtomicBool,
    /// 本文档是否曾经成功 attach 过 renderer（重绑/关闭后用于区分"还没绑"与"绑过又没了"）。
    ever_had_renderer: AtomicBool,
    renderers: Mutex<Vec<RendererLease>>,
    pub generation: Mutex<Option<ApiGeneration>>,
    pub sequence_regions: Mutex<HashMap<u64, HashSet<u64>>>,
    /// sequence 真实键 → 扁平 region sequence 下标（= 映射层轨道下标）。
    ///
    /// 只用于把"被分配了 sequence、当前却没有 clip"的轨道认出来（空 folder 轨、
    /// 只有静音/待授权 item 的子轨）—— 它们仍应是参数根。**不是**按名字/位置猜轨道：
    /// 这条边来自宿主 `createRegionSequence` 的真实键。
    pub sequence_track_index: Mutex<HashMap<u64, usize>>,
    pub regions: Mutex<HashMap<u64, crate::ara::AraPlaybackRegion>>,
    /// region key → clip id 的**缓存**投影。
    ///
    /// 【为什么是缓存而不是权威】权威身份是 region key 本身，clip id 由它派生
    /// （`ara-region-{key}`，见 `ara::mapping::MappingIdentities`）。这张表只是把
    /// "key → 当前时间线里的 id"记下来供反查，**缺席不是错误** —— [`Self::clip_id_for`]
    /// 会按同一个算式补出来。此前三处调用点把缺席当致命错误（`missing assigned clip
    /// identity` 等），而 `clip_ids` 只在 `end_editing` 时整体重写，于是"region 已分配、
    /// identity 还没重算"这个窗口里的每次渲染都必然失败。
    pub clip_ids: Mutex<HashMap<u64, String>>,
    pub region_items: Mutex<HashMap<u64, String>>,
    /// ARA 授权那一刻，每个 region 对应 item 的 **active take GUID**（region_key → take GUID）。
    ///
    /// 【为什么需要】ARA 只授权 active take 的 PCM（真机实测）。宿主清单重建 take 列表时（`sync_host_takes`），
    /// 只有 GUID 与这里相等的那一个 take 可以带上授权媒体；其余 take 必须保持无源。
    ///
    /// 【为什么记 GUID 而不是"就是 active 那个"】用户在 REAPER 里切换 active take 后，
    /// 这条记录直到 ARA 模型重新认领才更新。这期间**不沿用**旧 PCM —— 否则会把上一个
    /// take 的采样挂到新 take 上，那是听不见的错（渲染读的是同一条 `source_path`）。
    /// 记不下来时宁可不挂：无源占位是看得见的。
    pub authorized_takes: Mutex<HashMap<u64, String>>,
    /// ARA 授权那一刻，每个 region 对应 item 的 active take 的**源文件路径**
    /// （region_key → 文件路径）。与 [`Self::authorized_takes`] 同处写入。
    ///
    /// 【为什么需要】GUID 相等是**最强**的身份判据，但用户在 REAPER 里把 active take
    /// 换成**同一个文件**的另一个 take（复制 take、切换 take）时 GUID 就变了，而内容
    /// 并未改变。此时文件路径相同即可证明"同一份已授权 PCM"，`sync_host_takes` 可以
    /// 安全地把它挂上去，而不是让明明有音频的片段显示占位。
    ///
    /// 【边界不可松动】这只回退到**已被 ARA 授权的那份 PCM**；路径为空或读不到时
    /// 仍然"宁可不挂"（无源占位是看得见的，挂错采样是听不见的错）。
    pub authorized_sources: Mutex<HashMap<u64, String>>,
    /// 每个 ARA 音频源（persistentID → 状态）的**可读性**，供界面与自愈判定。
    ///
    /// 【为什么需要它】此前"源暂时不可读"与"源从来没读到过"在实现里是同一件事：
    /// `refresh_source_pcm` 无论什么原因都先把已发布的 PCM 删掉，再"若可读则重读"。
    /// 于是宿主收回一次样本访问（后台、undo 历史、宿主自己整理缓存）就等于**永久**
    /// 丢掉音频 —— 恢复只依赖"宿主记得再发一次 enable"，而那条回调丢了就没了。
    ///
    /// 把状态记下来之后：已发布的 PCM 只在"源被销毁 / 内容版本换了且新版本就绪 /
    /// 文档关闭"这三种情况下被替换，其余一切暂时不可读都只是状态变化。
    pub source_states: Mutex<HashMap<String, SourceState>>,
    /// 最近一次由**模型侧**算出的输入可用性读数（`None` = 还没算过，按旧行为处理）。
    ///
    /// 【为什么由模型侧写、文档侧读】`audio_sources` 是 `ModelHandle` 的私有数据，
    /// 文档看不到；而判定结果要给 `ensure_loaded` / `ara_render_input` / 诊断包用。
    /// 让模型在每次源变化之后写一份读数，是两者之间唯一不需要反向引用的通道。
    ///
    /// 【`None` 为什么等于旧行为】没有跑过任何源回调的会话（测试夹具、纯几何会话）
    /// 不该因为"没算过"被判成未就绪 —— 那会把未知变成否定，正是本轮要修的那类错误。
    /// 本地补丁（T8）：本份文档控制器的图账本。
    ///
    /// 【为什么文档持有它】桥把账本记在 `ControllerAdapter` 上，而诊断路径在文档侧。
    /// `document_controller` 闭包每建一份模型就运行一次，模型在那里接管句柄并转存到
    /// 这里 —— 一条不需要反向引用的通道。缺席时降级为空账本（账本是纯观测）。
    pub graph_ledger: Mutex<Option<std::sync::Arc<ara2_bridge::plugin::GraphLedger>>>,
    pub readiness: Mutex<Option<InputReadiness>>,
    pub ui_tracks: Mutex<std::collections::BTreeMap<String, crate::host::UiTrack>>,
    pub ui_known_tracks: Mutex<HashSet<String>>,
    /// 由 REAPER folder 结构**得到父级**的轨道 id（见 `host::folder`）。
    ///
    /// 这些轨道的父子边由宿主决定，插件私有分组**不得**改写 —— 否则同一个工程会
    /// 出现两套父子关系：用户在 REAPER 里改了分组，插件却显示另一套，且无法诊断。
    ///
    /// 【为什么只记"有父级"的轨道，而不是"宿主呈现过的所有轨道"】宿主清单同时承载
    /// 轨道身份（GUID↔id 绑定）与 folder 结构两件事。宿主**没有**给出父级的轨道
    /// （folder 之外的普通轨）并不代表"宿主说它是根级"，只是没有那条边 —— 私有分组
    /// 依然可以把它编进一个参数组。把整份清单都算作权威会静默废掉用户的私有分组。
    pub host_folder_children: Mutex<std::collections::BTreeSet<String>>,
    /// 分割谱系：右半段 item GUID → 左半段（父段）item GUID。
    ///
    /// 【为什么需要】宿主分割**不改变音频源** —— 两半指向同一个文件。父段已被 ARA
    /// 授权，所以右半段可以立刻显示波形，不必等宿主为它重新分配 region。缺了这条，
    /// 分割后右半段会停在一段"等待 REAPER 提供音频"的占位里（用户报障过）。
    ///
    /// 【为什么不复用图集时代的 `split_parents`】那份谱系服务的是**参数曲线**
    /// 继承，键与生命周期都挂在参数权威上；媒体派生是显示层的事，两件事不该共用一个
    /// 记录 —— 否则一方的清理会静默影响另一方。
    pub split_media_from: Mutex<HashMap<String, String>>,
    /// 每个尚未拿到音频的 item 是**从什么时候**开始等的（item GUID → 首次进入
    /// "在途"的时刻）。
    ///
    /// 【为什么曾经需要它】"等待宿主完成音频分配"此前是**吸收态**：一个 item 没有
    /// `source_path`、又不是 folder 父轨，就永远显示"正在等待"，用户做什么都出不来。
    /// 记下时刻是为了"超过 2 s 就转 `unavailable` 并给出原因"。
    ///
    /// 【为什么它现在**不**决定状态】那个终态把"我们这一轮没观察到"说成了"宿主不会
    /// 交付了"，而后者是一句我们无法证明的断言 —— 只要展示侧的身份链断一次，所有片段
    /// 都会被盖上同一句话（用户看到的正是"无论什么情况，所有 Clip 都在等"）。
    /// 现在它只回答"等了多久"（`wait_seconds`），状态恒为 `pending`。
    pub pending_since: Mutex<HashMap<String, std::time::Instant>>,
    /// 最近一次映射里被**跳过**的 playback region（下标 + 语言无关原因码）。
    ///
    /// 【为什么要进 GUI 载荷】此前只 `log::warn`，于是 MIDI / 零长度 item 在插件里
    /// **静默消失**：用户看到"我明明有这一段"，界面却什么都没有，也没有任何提示。
    /// 记在这里，载荷给一句"N 个片段未显示（宿主数据无法映射）"，用户至少知道该查什么。
    pub skipped_regions: Mutex<Vec<(usize, String)>>,
    /// 上一次打印的诊断行（去重用；见 `log_diag_if_changed`）。
    last_diag: Mutex<Option<String>>,
    /// 上一次打印的逐 clip 媒体读数（去重用；见 `decorate_host_media_locked`）。
    pub(crate) last_media_diag: Mutex<Option<String>>,
    /// 上一轮呈现里被**收敛掉**的重复 clip 数（见 `present_host_inventory` 的 T-S2-2）。
    ///
    /// 【为什么是一个文档级读数】"同一个宿主 item 被画成两条 clip"是**呈现层**的事实，
    /// 而渲染时间线（`timeline`）里始终只有一条 —— 此前所有诊断读数都读渲染时间线，
    /// 于是这个问题在日志里完全隐形，四轮排查都在"为什么真 clip 没有源"上打转。
    /// 把这个数放进 `diag_line` 之后，"多余的一条"在日志里就是可数的。
    pub(crate) presentation_dupes: std::sync::atomic::AtomicUsize,
    /// 最近一次**完整**宿主清单采集的代次令牌 `(change_count, model, scope)`。
    ///
    /// 【为什么是文档级、不是实例级】`ui_tracks` / `render_facts` 都是文档级共享状态，
    /// 采集结果写进 `document`，任何实例调用 [`crate::render::extension::ExtensionOwner::refresh_ui_inventory`]
    /// 都命中同一份缓存 —— 于是"谁采的"与"谁要用的"不再分离（多实例下只有第一个带
    /// 新令牌的实例真正枚举，其余早退）。
    pub ui_inventory_stamp: Mutex<Option<(i32, u64, u64)>>,
    /// 最近一次**完整**采集的时刻。
    ///
    /// 【为什么必须与令牌并存】令牌以 `GetProjectStateChangeCount` 为主。方向（倒放）是
    /// **take 级**属性，宿主未必为它推进项目 change count；只靠令牌早退会让"倒放 Take"
    /// 连采样这一步都被跳过（`render_facts` 不更新 ⇒ 渲染输入不失效 ⇒ 波形变了声音没变）。
    /// 超过 [`crate::render::extension::FACTS_REFRESH_INTERVAL`] 就**强制**重采一轮，
    /// 让任何 take 级事实变化都必然穿透这道早退 —— 代价被钉在低频（见该常量）。
    pub ui_inventory_sampled_at: Mutex<Option<std::time::Instant>>,
    /// 上一轮宿主清单里"ARA 表达不到、但会改变渲染结果"的事实快照。
    ///
    /// 【为什么必须有它】`render_epoch` 此前只有**淡变长度**一个写者，于是宿主把一条
    /// take 倒放（或改声道模式、开循环源）之后，`PreparedVersion` 五分量全不变 ⇒
    /// `prepare_job` / `prepare_offline_until` 判定"已是最新"直接早退，继续播旧的正向
    /// 音频。用户表现为"波形更新了、声音没变，编辑一下才对"（编辑是唯一能推动
    /// `PreparedVersion.edit` 的手动途径）。见 `host::geometry::RenderFacts`。
    pub render_facts: Mutex<crate::host::geometry::RenderFacts>,
    /// 连续"这一轮没有可渲染内容"的轮数 / 起点（Unix 毫秒）/ 是否已经说过一次。
    ///
    /// 【为什么需要计数】`RenderOutcome::Nothing` 是**常态**：宿主在拖动一条 clip 的
    /// 那几秒里，每几百毫秒就会发布一次空模型（`item map: attached=2 unbound=1`、
    /// `renderer assignment regions=0`）。所以"闪一下"永远不该留痕迹；而**持续**为空
    /// 不是常态，值得说一次。三个量就是这条分界线，且它只在这里定义一次。
    empty_rounds: AtomicU32,
    empty_since_ms: AtomicU64,
    empty_reported: AtomicBool,
    pub sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    pub edit_sources: Mutex<HashMap<String, Arc<super::source::SourcePcm>>>,
    /// 每个宿主 item 最近一次观察到的**媒体事实**（item GUID → 源身份）。**只增不减**。
    ///
    /// 【为什么需要它】显示身份此前完全依赖"这一轮我们有没有 PCM"。而宿主随时可以收回
    /// 样本访问（ARA 的访问默认关闭、由宿主开关），那一刻 `edit_sources` 仍有副本，
    /// 但**新文档**的 `edit_sources` 一开始是空的 —— 于是"这一轮有没有 PCM"直接决定
    /// "这一轮有没有身份"，一次收回就让**所有** clip 一起变占位。
    ///
    /// 这张表把"我们知道它是哪个源"与"这一轮有没有采样"分开：事实只增不减，
    /// 只在 region 集合与文档关闭时按 key 回收。
    pub media_facts:
        Mutex<std::collections::BTreeMap<String, crate::editor::media_identity::PcmFacts>>,
    /// 已授权但**读取失败**的 ARA 音频源（`audioSource.persistentID` → 原因）。
    ///
    /// 【为什么必须与 `ready` 分开】`ready` 的语义是"ARA 是否已重新交付输入"，它还
    /// 参与 `ensure_loaded` 的宿主冲突判据（见 `editing_session`）。把"某个源读不出来"
    /// 折进 `ready=false` 会让一次样本读取失败被报成"宿主冲突"，正是用户报障的
    /// "尚未应用：Conflict: host changed" 的来源。所以这里单记一份：读取失败时条目
    /// 出现、成功时消失，`decorate_host_media_locked` 据此把 clip 的
    /// `host_media_reason` 报成真因（`source_read_failed`），而不是含混的
    /// `awaiting_region`。
    pub source_failures: Mutex<HashMap<String, String>>,
    /// 已经建成、但 ARA 还没认领的导入（见 `PendingImport`）。
    pub(crate) pending_imports: Mutex<Vec<PendingImport>>,
    /// 未认领 region 的**连续轮次**（region key → `(轮次, 首次未认领时刻)`）。
    ///
    /// 【为什么需要"连续多少轮"】region → item 的认领是**分轮**完成的：一轮里还没轮到
    /// 的那条 region，下一轮就认领了（现场实测 213 ms 自愈）。把这种正常中间态记成
    /// WARN，等于让每份日志都背着一堆假告警，真告警被压在里面。只有连续
    /// [`UNBOUND_WARN_STREAK`] 轮仍未认领才值得 WARN，并带上 streak 与首次出现时刻。
    pub(crate) unbound_streaks: Mutex<HashMap<u64, (u32, std::time::Instant)>>,
    /// 本插件拒绝创建 region 的原因（有界环形缓冲，最新的在尾部）。
    ///
    /// 【为什么要单独一张表】"等待宿主音频"是用户可见的唯一状态，而它把两种截然
    /// 不同的事实混成一句话：宿主**还没**交付，和插件**拒收**了交付。后者是
    /// 可行动的故障 —— `HostAudioState::RegionRejected` 就是为它加的。
    ///
    /// 【为什么有界】它只在故障路径上写；不设上限的话一个反复失败的宿主会把内存
    /// 吃光。32 条足够覆盖"前因 + 若干次重复"。
    pub region_rejections: Mutex<std::collections::VecDeque<String>>,
    /// PCM 可以晚于结构回调到达；不借用 model/scope 版本触发 GUI 资源补齐。
    pub editor_audio_revision: AtomicU64,
    pub timeline: Mutex<Option<hifishifter_kernel::state::TimelineState>>,
    /// 装配器的**输入 A**：ARA 侧的 region 表（一条 playback region 一行）。
    ///
    /// 【为什么不是 clip】region 只回答"这段时间播放素材的哪一段"。clip 由
    /// [`Self::rebuild_timeline_locked`] 唯一产生 —— 这是"一个循环延伸的 item
    /// 不会再被画成 N 条片段"的结构性保证（见 `hifishifter_kernel::editor::assembly`）。
    pub region_rows: Mutex<Vec<hifishifter_kernel::editor::assembly::RegionRow>>,
    /// 装配器的**输入 A** 的轨道侧：region sequence → 模型轨道。
    pub region_tracks: Mutex<Vec<hifishifter_kernel::editor::assembly::TrackRow>>,
    /// 最近一次装配的 `spans`：clip id → 落在它里面的 region 行下标。
    ///
    /// 【为什么留一份】PCM 授权、媒体时长索引、`ara.item-map` 读数都要回答
    /// "这条 item 的音频来自哪个 ARA 源"，而答案就在这张边上。它**不是**第二份
    /// clip 集合：只有行下标，没有 clip。
    pub assembly_spans: Mutex<std::collections::BTreeMap<String, Vec<usize>>>,
    /// 最近一次装配里**没挂上任何 item** 的 region 行（不产生 clip，只进读数）。
    pub assembly_unbound: Mutex<Vec<usize>>,
    /// 装配的**输入 A**（`region_rows`）写了几次。
    ///
    /// 【为什么要一个计数而不是"读一下内容变没变"】两份输入是**两张表**，逐字段比较
    /// 的代价与"它们是不是同一时刻的事实"这个问题根本不对等：一个代次就够回答
    /// "这一轮装配读到的和上一轮是不是同一对"。见 [`Self::assembly_consumed`]。
    pub(crate) region_rows_stamp: AtomicU64,
    /// 装配的**输入 B**（`ui_tracks`）发布了几次。
    pub(crate) ui_tracks_stamp: AtomicU64,
    /// 上一次装配尝试消费的那一对代次 `(ui_tracks_stamp, region_rows_stamp)`。
    ///
    /// 【它回答什么】"这一轮读到的两份输入，与上一轮相比**动过**吗"。动了就说明
    /// 其中至少一份可能还没跟上另一份 —— 那正是 `unbound` 最可能的成因（item 已经
    /// 挪到新位置、region 表还是旧的，或反过来）。没动则说明再试一次也只会得到同一个
    /// 答案：那就是结论，该发布（并报警）。
    pub(crate) assembly_consumed: Mutex<Option<(u64, u64)>>,
    /// 当前这一串"推迟发布"的起点（`None` = 没有在推迟）。
    ///
    /// 【为什么必须有上界】清单在**拖动 item** 期间每 100 ms 就可能发布一代新的，
    /// 于是"输入动过 ⇒ 再等等"会一直成立。没有预算的话界面会停在上一份时间线上直到
    /// 用户松手 —— 那比"少一条片段"糟得多。
    pub(crate) assembly_deferred_since: Mutex<Option<std::time::Instant>>,
    /// 装配的输入变过、还没被消费。
    ///
    /// 【为什么必须有这个标记】装配有两份输入：ARA 侧的 region 表（模型回调写）与宿主
    /// 侧的 item 清单（20 ms 定时器写）。此前只有一个调用点（模型收口），于是**清单自己
    /// 的变化永远不会触发装配** —— 而在 REAPER 里复制/粘贴一个 item 时，清单先看到新
    /// item、模型收口那一轮又可能因为宿主工程正在变化而拿不到新清单。结果是新 item 只能
    /// 以"无源占位"出现：没有波形、没有原始音高线、永远显示"正在等待宿主完成音频分配"，
    /// 直到下一次重新加载（那次模型收口带上了完整清单）才正确。用户报障的正是这一条。
    ///
    /// 【为什么是"标记 + 单一冲刷点"而不是就地重装】装配要写 `clip_ids`，而清单刷新会
    /// 被 `host_geometries_locked` / `clip_for_host_item` 调到（那两处正持有 `clip_ids`，
    /// `std::sync::Mutex` 不可重入），也可能被持有 `transaction` 的路径调到。就地重装
    /// 实测会让两个 bound 用例永久挂起。标记把"该重装"与"现在能重装"分开：任何写输入的
    /// 地方只置位，真正重装只发生在**不持任何相关锁**的定时器/收口点。
    pub(crate) assembly_dirty: AtomicBool,
    /// ARA 模型是否**已经交付过**至少一次。
    ///
    /// 【为什么需要】`rebuild_timeline_locked` 会在每次清单刷新后跑。而清单可能先于
    /// ARA 模型到达（宿主加载工程、测试夹具手工塞 `ui_tracks`）—— 那时 region 表是空
    /// 的，装配结果只会是一堆没有源窗口的占位。**用它当"这份时间线归不归我管"的判据**：
    /// 模型从未交付过时，不动已经发布的时间线（否则"宿主里一切都在、插件里什么都没有"）。
    pub ara_model_seen: std::sync::atomic::AtomicBool,
    pub track_bindings: Mutex<crate::state_channel::TrackBindings>,
    pub revision: AtomicU64,
    pub render_epoch: AtomicU64,
    pub scope_revision: AtomicU64,
    /// 纯GUI宿主装饰版本，不改变编辑曲线或神经合成代次。
    pub ui_geometry_revision: AtomicU64,
    /// 编辑器**可见内容**的代次：装配发布、源状态变化、宿主写回后的投影变化都推进它。
    ///
    /// 【为什么不能复用 `ui_geometry_revision`】那个读数的语义是"宿主清单的几何变了"，
    /// 而本轮用户报障的三种情形**都不改宿主几何**：
    ///   * 粘贴后装配发布了新 clip（几何早就在清单里了，变的是媒体身份）；
    ///   * 宿主重新授权样本访问，源从 Revoked 回到 Published（同代重读，PCM 一模一样）；
    ///   * 宿主写回成功、fade 需要重新投影（fade 长度不推进 `GetProjectStateChangeCount`）。
    ///     复用几何代次会让"几何没变"变成"不用重画"，而真正该重画的正是那些情形 ——
    ///     用户看到的就是"新粘贴的 Clip 一直等宿主音频，重开工程才恢复"。
    ///
    /// 【为什么是独立原子量而不是布尔】`refresh_ui_geometry` 是**读-比较-投影-写回**
    /// 的循环：它必须在投影成功后记住"我投影的是哪一代"。布尔只能表达"脏了"，
    /// 无法回答"我投影的这一份是不是最新的" —— 投影期间到来的新一代会被静默吞掉。
    pub presentation_revision: AtomicU64,
    /// 最近一次把图集重投影进 `edits.params` 时的 `ui_geometry_revision`。
    ///
    /// 【为什么需要它】视图重建只应发生在**几何真的变了**之后。没有这个读数时，
    /// 任何一次 `workspace_view()`（诊断、私有分组、清单刷新）都会重写参数数组 ——
    /// 而"读一次工作区"不应改变一个待提交编辑所依赖的指纹（并发的 automatic apply
    /// 会因此判成 workspace changed）。
    resynced_geometry: AtomicU64,
    pub ready: AtomicBool,
    /// 宿主是否至少给过一个源样本访问授权。
    ///
    /// 【用途】把"宿主还没交付 region"与"region 有了、但样本访问还没开"分开显示；
    /// 两者对用户的可执行动作不同。
    pub samples_access_granted: AtomicBool,
    /// 宿主是否处在一次 `beginEditing` / `endEditing` 会话中。
    ///
    /// 【为什么与 `ready` 分开】`ready` 的语义是"ARA 是否已重新交付可用输入"，
    /// 会被**任何**内容变化按回 `false`；而"宿主正在编辑"是另一件事。此前二者共用
    /// `ready`，于是会话内每一次真实变化都被 `ensure_loaded` 判成"宿主冲突"
    /// （用户报障：每次编辑都闪"尚未应用：Conflict: host changed"）。
    /// 有了这个标志，会话中的未就绪窗口被识别为**暂缓**而不是冲突 —— 它会随
    /// `endEditing` 自然收敛，不需要用户干预，也不该报错。
    pub editing_session: AtomicBool,
    /// 编辑会话开始的**单调毫秒**（0 = 不在会话中）。
    ///
    /// 【为什么与 `editing_session` 并存】那个布尔回答"是不是在会话中"，这个数回答
    /// "在会话中多久了"。而"多久"正是插件侧唯一能自主判断"宿主不会再来收口"的依据
    /// （见 [`Self::request_settle_if_overdue`]）。两者分开是因为布尔在音频线程上读，
    /// 而这个数只在非音频线程上读 —— 语义不同，读者也不同。
    pub editing_since_ms: std::sync::atomic::AtomicU64,
    /// 模型自上次成功发布以来**第一次变脏**的单调毫秒（0 = 干净）。
    ///
    /// 【为什么要它】`ModelHandle::model_dirty` 是模型线程私有的布尔，插件侧看不到；
    /// 而"脏了多久还没被 flush"正是第二条死路的判据（宿主既不 `endEditing` 也不
    /// `notifyModelUpdates` 时，唯一的 flush 入口永远到不了）。
    pub model_dirty_since_ms: std::sync::atomic::AtomicU64,
    /// 插件侧请求"强制收口一次编辑会话"（由非 ARA 线程置位，由 ARA 线程消费）。
    ///
    /// 【为什么必须是**请求**而不是直接调用】桥的 `PluginRuntime` 由文档控制器独占，
    /// 不是 `Send`/`Sync`；从视图定时器直接调用它是未定义行为。这个原子量是两者之间
    /// 唯一的通道：谁都可以置位，控制器在**下一次任何回调**的入口消费它。
    pub forced_edit_close: AtomicBool,
    /// 插件侧请求"把积压的模型变化 flush 一次"（同上的通道）。
    pub forced_model_flush: AtomicBool,
    /// 最近一次因宿主内容变化而撤销渲染许可的时刻（见 [`Self::clear_renderers`]）。
    ///
    /// 【为什么需要它】`ready == false` 有两种完全不同的来源：宿主正在一次编辑
    /// 会话中（`editing_session` 已表达，必然收敛），与宿主改完内容后**不再收口**
    /// （不发 `endEditing`、也不发 `notifyModelUpdates`）。后者此前被
    /// `ensure_loaded` 当成"宿主冲突"永久拒绝，用户每做一次操作就看到一条
    /// `Conflict: host changed`，而窗口再也刷不出来。用这个时刻把"刚刚才变过、
    /// 正在等宿主收口"与"确实没有可用输入"分开。
    pub last_content_change: Mutex<Option<std::time::Instant>>,
    pub transaction: Mutex<()>,
    pub edits: Arc<Mutex<crate::state_channel::EditState>>,
    pub id: DocumentId,
    pub clock: Arc<super::transport::TransportClock>,
    /// 宿主给本**文档**的播放控制租约，连同它被创建时所在的那条线程。
    pub playback: Mutex<Option<PlaybackEndpoint>>,
    pub host_undo: crate::host::undo::HostUndo,

    /// 渲染输入失效的"一秒钟里发生了几次"（诊断用，见 `invalidate_render_input`）。
    ///
    /// 【为什么需要它】"渲染停不下来"这个症状在日志里此前**完全不可见**：每次失效都只
    /// 推进一个原子计数，而重算发生在另一条线程上。于是"某个判据把每 tick 都当成变化"
    /// 这种缺陷只能靠读代码发现。这里只记一个窗口内的次数，超阈值发一条 WARN。
    render_epoch_window: Mutex<(Option<std::time::Instant>, u32)>,
    /// 宿主 BPM / 拍号读数（VST3 进程上下文）。
    ///
    /// 【为什么记在文档上】Tempo Map 的音阶点需要宿主轴：0 位置点**必须**显式带拍号
    /// （前端 `effectiveTimeSignatures` 依赖它），每个点都要 BPM。而音阶投影在
    /// `workspace_timeline_locked` 里跑（`DocumentSession` 上），那里读不到
    /// per-view 的 `ProjectState`。
    pub host_meter: Mutex<HostMeter>,
    /// 宿主经 ARA 音乐上下文给出的真实 tempo map 与拍号。
    ///
    /// 【为什么在文档上】它是**文档级**音乐事实（整个工程一份），不是某个 renderer 的
    /// 属性；而投影缓存（`musical_projection`）与 `host_meter` 也在文档上 —— 三者必须
    /// 同源，否则缓存键会漏掉 tempo map 的变化。
    ///
    /// `None` = 宿主没提供（或读取失败），投影退回 VST3 的单点 tempo。
    pub host_tempo_map: Mutex<Option<crate::ara::model::HostTempoMap>>,
    /// 插件自有音乐上下文的投影缓存，键是"输入代次"（见
    /// [`DocumentSession::project_plugin_musical_context_locked`]）。
    pub musical_projection: Mutex<Option<(MusicalProjectionKey, PluginMusicalProjection)>>,
    /// ARA 存档恢复出来的编辑，等待模型图就绪后合入。
    ///
    /// 【为什么放在文档上而不是 owner 上】`beginRestoringDocumentFromArchive` 在
    /// **模型图重建之前**到达（ARA 的恢复顺序：先给存档，宿主再重新交付模型）。那一刻
    /// 可能一个 renderer 都还没绑定，`renderer_owners()` 是空的 —— 所以载荷必须存在
    /// 文档级，等 `prepare_renderers` 收口时再交给 owner 的既有合并路径。
    ///
    /// 【为什么复用 owner 的 `pending_restore`】合并语义（reconcile、身份重绑、冲突
    /// 判定）已经完整实现并测试过（`ExtensionOwner::merge_pending_restore`）。存档通道
    /// 只是**另一个来源**，不该有第二份实现 —— 两份实现迟早会漂移。
    pub pending_archive_restore: Mutex<Option<crate::state_channel::EditState>>,
    /// 下一次允许尝试合入文档级存档的时刻（退避）。
    ///
    /// 【为什么需要】合入会在工作区快照上被重试（33–400 ms 一次），而失败原因是
    /// "宿主图还没长成"，那要等几百毫秒到几秒。没有退避就会在等待窗口里每帧跑一次
    /// rebind（O(曲线) 的工作）。
    archive_merge_next_attempt: Mutex<Option<std::time::Instant>>,
    /// 本实例**主动创建**的宿主轨道 GUID（`create_host_track` / 粘贴 / 导入建轨）。
    ///
    /// 【为什么必须有这张表】清单的轨道范围是「本轨 + folder 后代」（`ui_folder_tracks`），
    /// 而 `create_host_track` 把新轨插在锚点**之后** —— 那是本轨的**兄弟**。
    /// 兄弟不在「本轨 + 后代」里，于是新轨在宿主里真实存在，插件却**构造上**永远看不到。
    /// 这不是刷新时机的问题：加多少次 `refresh_host_transport` 都不会让它出现。
    ///
    /// 【为什么登记而不是放宽枚举范围】放宽会让每个实例看到整个工程，与"每实例只服务
    /// 自己那批 region"的既有纪律冲突。登记表表达的是"谁建的谁负责"，这是唯一的正确语义。
    ///
    /// 【为什么不做自动裁剪】用户删掉轨道之后，残留的 GUID 在树里查不到，每轮只是一次
    /// 线性扫描的开销；而"枚举这一轮恰好失败"与"用户真的删了"在插件侧无法可靠区分，
    /// 裁错一次就会让一条活着的轨道永久消失。所以只设上限（最旧先丢），不自动删。
    pub adopted_tracks: Mutex<Vec<String>>,
    /// 【G5】此刻是否处在一次 ARA 编辑会话里（`beginEditing` … `endEditing`）。
    ///
    /// 【为什么它是判据】SDK 明确说明**还原会话在概念上就是一次编辑会话**
    /// （`ARAInterface.h:2938-2943`）。于是"载荷到达时我们在编辑会话里"就是
    /// "这是一次撤销/重做"的权威证据，那种场合**必须**让它覆盖在线状态。
    in_edit_session: std::sync::atomic::AtomicBool,
    /// ARA 2 **部分存档**的暂存载荷（`restoreObjectsFromArchive` 的 `Selected` 分支）。
    ///
    /// 【为什么不复用 `pending_archive_restore`】那条路径经 `apply` + `merge` 走"整份
    /// 文档状态替换"，而部分载荷**故意**只带一部分（例如只带一条 modification 的参数），
    /// 走那条路会用空值覆盖掉其它状态。这里只做**按键合并**：载荷带了什么就覆盖什么。
    pub pending_partial_restore: Mutex<Vec<crate::state_channel::EditState>>,
    /// 最后一次成功编码的完整存档。
    ///
    /// 【为什么需要】宿主可以在模型图尚未 ready 时调用 `storeDocumentToArchive`
    /// （加载中的自动保存、工程关闭）。此前 `encode_archive` 直接返回
    /// `host graph not ready`，宿主据此认为插件存档失败，重开时编辑全丢。
    /// 现在 `ready == false` 时返回这份快照；恢复到达时也会先写一份，保证
    /// "恢复后还没合入就保存"不会把编辑覆盖成空。
    pub archive_cache: Mutex<Option<Vec<u8>>>,
    /// ARA 文档存档通道的读数（见 [`ArchiveReading`]）。
    ///
    /// 【为什么与 `archive_cache` 分开】那个字段是**数据**（要原样交回宿主），这个
    /// 字段是**证据**（通道有没有被调用、交回了多少字节、合入发生在什么时候）。
    /// 把两者混在一起，就得在读数据时顺带改证据。
    pub archive_reading: Mutex<ArchiveReading>,
    /// 用户编辑的采纳读数（见 [`EditHealth`]）。
    pub edit_health: Mutex<EditHealth>,
    /// 最近一次把「锁定参数线」应用到既有权威时的设置代次。
    ///
    /// 【为什么是「设置代次」而不是布尔】切换一次就要重新落一次；而"当前值是什么"
    /// 由 `settings_store::lock_param_lines()` 回答。用代次做缓存戳，热路径只付一次
    /// 原子读。初值取 `u64::MAX`：进程启动后的第一次调用必须真的落一次。
    /// 运行时几何日志：锁定关闭时只在这里记账；锁定开启时才是参数事务的输入。
    pub(crate) geometry_journal: Mutex<hifishifter_kernel::editor::param_geometry::GeometryJournal>,
    /// 宿主清单枚举**正在失败/中途放弃**（默认 false = 清单可用）。
    ///
    /// 【为什么用反向语义】`DocumentSession` 用 `..Default::default()` 构造，
    /// 而"清单可用"才是常态 —— 用一个默认 false 的"不完整"标记，就不必为它写
    /// 手写 `Default`。枚举方在入口置 true、成功走到末尾置 false。
    pub(crate) inventory_incomplete: std::sync::atomic::AtomicBool,
    /// clip id → 连续多少轮**完整**清单没见到它（幽灵 clip 的宽限期计数）。
    pub(crate) orphan_streak: Mutex<std::collections::HashMap<String, u32>>,
    /// 最近一次几何事务的读数（诊断与状态栏）。
    pub(crate) last_geometry_report:
        Mutex<Option<hifishifter_kernel::editor::param_geometry::TransformReport>>,
    /// 逐根轨的**分析基线**：派生原始值的唯一权威（见 `RootBaseline`）。
    ///
    /// 【为什么单独一层】派生值此前住在 `edits.params` / `timeline.params_by_root_track`
    /// 里，于是"分析完成"与"编辑器回传"会互相覆盖 —— 用户看到的就是参数线随机回退。
    /// 这里只有一个写入口（[publish_baseline]），显示永远由它派生。
    pub(crate) baseline: Mutex<
        std::collections::BTreeMap<
            String,
            hifishifter_kernel::editor::param_authority::RootBaseline,
        >,
    >,
    /// ARA 变更通知队列，由 `runtime.rs` 在构建插件时注入。
    ///
    /// 【为什么必须有】宿主只保存"它知道变过"的部分。插件从不通知，只存增量的宿主
    /// 就会丢数据 —— ARA 在 `ARAModelUpdateControllerInterface` 的三处文档里反复重申
    /// "必须可靠发送"（`ARAInterface.h:2098-2099` / `:2116-2117` / `:2144-2145`）。
    /// 此前插件**从未**调用过它（`mark_*` 在生产代码里出现次数为 0）。
    update_emitter: Mutex<Option<ara2_bridge::plugin::UpdateEmitter>>,
    /// modification persistentID → 宿主句柄身份 `(index, session)`。
    ///
    /// 通知要按**宿主句柄**投递（桥接层用它查 `ARAAudioModificationHostRef`），
    /// 而插件侧的状态是按 persistentID 索引的。`createAudioModification` /
    /// `cloneAudioModification` 是唯一拿得到句柄的时刻。
    ///
    /// 【为什么两个方向各存一份】通知要 persistentID → 句柄，而部分存档的写出侧
    /// （`StoreFilter` 只给句柄）要句柄 → persistentID。
    modification_handles: Mutex<HashMap<String, ara2_bridge::core::RawHandle>>,
    modification_ids: Mutex<HashMap<(std::num::NonZeroU32, std::num::NonZeroU64), String>>,
    editor: OnceLock<Result<Arc<crate::editor::session::EditorSession>, String>>,
}

/// 从 clip id 反解 region key。
///
/// 【为什么需要反解】clip id 由 region key 派生（`ara-region-{key}`，见
/// `ara::mapping::MappingIdentities`），所以它是自描述的。`clip_ids` 这张缓存表
/// 由 `remap_and_log` 整体重写，而 region 分配发生在它之前；在两者之间的窗口里
/// 反查缓存会落空。此时直接从 id 解析出 key，与缓存命中给出同一个值。
///
/// 位置派生的旧 id（`ara-clip-{N}`）**不可反解** —— 那正是它被替换掉的原因。
pub(crate) fn parse_region_key_from_clip_id(clip_id: &str) -> Option<u64> {
    clip_id
        .strip_prefix("ara-region-")?
        .parse::<u64>()
        .ok()
        .filter(|key| *key != 0)
}

/// 宿主音乐读数的快照；默认 120 BPM 4/4（与 `TimelineState::default` 同口径）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HostMeter {
    pub bpm: f64,
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for HostMeter {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            numerator: 4,
            denominator: 4,
        }
    }
}

/// 音阶投影缓存的键：设置代次 + 宿主音乐读数。
///
/// 【为什么宿主读数也在键里】BPM/拍号变化不经过设置写入，只看设置代次会让
/// Tempo Map 的宿主轴停在旧读数上。
///
/// 【为什么 tempo map 也在键里】宿主经 ARA 给出的多点 tempo map 是**独立于**单点
/// `host_meter` 的另一条读数（`host_meter` 来自 VST3 `ProcessContext`）。它变了而键
/// 不变的话，投影会一直用旧速度 —— 用户改工程速度后插件里的小节网格纹丝不动。
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MusicalProjectionKey {
    settings: u64,
    bpm_bits: u64,
    numerator: u32,
    denominator: u32,
    tempo_map: Option<crate::ara::model::HostTempoMap>,
}

impl MusicalProjectionKey {
    pub(crate) fn new(
        settings: u64,
        meter: HostMeter,
        tempo_map: Option<crate::ara::model::HostTempoMap>,
    ) -> Self {
        Self {
            settings,
            bpm_bits: meter.bpm.to_bits(),
            numerator: meter.numerator,
            denominator: meter.denominator,
            tempo_map,
        }
    }
}

/// 插件自有音阶在时间线上的投影结果。
#[derive(Clone, Debug)]
pub(crate) struct PluginMusicalProjection {
    pub project_scale_notes: Vec<u8>,
    pub tempo_map: Option<Vec<hifishifter_kernel::state::TempoPointData>>,
}

#[cfg(test)]
mod baseline_tests {
    use super::*;

    /// **基线是值：它描述的那份数组与它一起交付，靠 Arc 共享而不是复制。**
    ///
    /// 【为什么这条取代了旧的三条用例】旧实现按指纹存取基线，于是"编辑器读到的那份"
    /// 与"提交时查回来的那份"可以不是同一份（缓存里根本没有 / 被下一次读挤出 /
    /// 来自另一个会话）。现在基线随视图一起交付，不可能落空。
    ///
    /// 【为什么不在这里开一个 DocumentSession】`workspace_view` 要求宿主时间线已经
    /// 交付（文档级 fixture）；这条用例钉的是**类型语义**，不需要那套装置。真正的
    /// 端到端行为由 `editor/session` 与 `bound_tests` 覆盖。
    #[test]
    fn a_view_carries_the_arrays_it_describes() {
        let params = Arc::new(std::collections::BTreeMap::from([(
            "root".to_owned(),
            hifishifter_kernel::state::TrackParamsState::default(),
        )]));
        let view = WorkspaceView {
            projection: "scope:abc".to_owned(),
            params: Arc::clone(&params),
            geometry_revision: 0,
        };
        let shared = Arc::new(view);
        assert!(
            Arc::ptr_eq(&params, &shared.params),
            "基线按 Arc 共享，不复制数组"
        );
        assert_eq!(shared.projection, "scope:abc");
        assert_eq!(shared.params.len(), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::core::ApiGeneration;
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    /// **每一个 `ready = false` 的写点都必须留下"宿主刚改过内容"的时刻。**
    ///
    /// 【为什么这是重开工程的回归】宿主授权 PCM 走的是
    /// `refresh_source_pcm` → `revoke_renderers()` → `revoke_snapshots()`。
    /// 那条路径此前**不写** `last_content_change`，于是 `ensure_loaded` 把它当成
    /// "宿主声称输入不可用、而我们没有观察到任何变化" —— 直接返回
    /// `Conflict: host changed; local curves preserved, host is not ready`，
    /// 时间线再也刷不出来（用户看到的就是"所有 Clip 只显示等待宿主音频"）。
    #[test]
    fn revoking_renderers_records_a_content_change() {
        let document = Arc::new(DocumentSession::new(1));
        *document.last_content_change.lock_or_recover() = None;
        document.revoke_renderers();
        assert!(
            document.last_content_change.lock_or_recover().is_some(),
            "撤销渲染许可必须记下观察时刻，否则会被误判成宿主冲突"
        );
    }

    fn document_with_one_empty_track(project_sec: f64) -> Arc<DocumentSession> {
        let document = DocumentSession::new(9_001);
        *document.timeline.lock_or_recover() = Some(
            serde_json::from_value(serde_json::json!({
                "tracks": [{"id": "track", "name": "t", "order": 0}],
                "clips": [],
                "bpm": 120.0,
                "project_sec": project_sec,
            }))
            .unwrap(),
        );
        document
    }

    /// 一个只有一条**循环源** clip 的时间线（媒体 10 s、item 10 s、倍率 1.0）。
    fn document_with_a_loop_clip() -> Arc<DocumentSession> {
        let document = DocumentSession::new(9_002);
        let mut timeline: hifishifter_kernel::state::TimelineState =
            serde_json::from_value(serde_json::json!({
                "tracks": [{"id": "track", "name": "t", "order": 0}],
                "clips": [{
                    "id": "ara-item-{A}", "track_id": "track", "name": "vocal",
                    "start_sec": 0.0, "length_sec": 10.0,
                    "takes": [{
                        "id": "take", "name": "vocal", "source_path": "src",
                        "source_start_sec": 0.0, "source_end_sec": 10.0,
                        "playback_rate": 1.0, "duration_sec": 10.0, "loop_enabled": true
                    }]
                }],
                "bpm": 120.0,
                "project_sec": 20.0,
            }))
            .unwrap();
        for clip in &mut timeline.clips {
            clip.normalize_takes();
        }
        *document.timeline.lock_or_recover() = Some(timeline);
        document
    }

    /// 跑一次**完整**的几何事务（与 `reconcile_parameter_geometry` 同一条链路）。
    fn reconcile_geometry_once(
        document: &DocumentSession,
    ) -> hifishifter_kernel::editor::param_geometry::TransformReport {
        let host = document
            .timeline
            .lock_or_recover()
            .clone()
            .expect("the fixture installs a timeline");
        let frame = document.geometry_frame(&host, true);
        let mut edits = document.edits.lock_or_recover();
        document
            .reconcile_parameter_geometry_locked(&frame, &mut edits)
            .expect("a well-formed frame must reconcile")
    }

    /// **装配的两份输入里，任何一份变了都必须在安全点上重装。**
    ///
    /// 【为什么这条是用户报障的可执行形式】REAPER 复制/粘贴一个 item 时，宿主清单先看到
    /// 新 item，而模型收口那一轮的清单刷新可能因为"宿主工程正在变化"被放弃。此前装配
    /// 只在模型收口跑，于是新 item 永远进不了装配：它只能以无源占位出现 —— 没有波形、
    /// 没有原始音高线、一直显示"正在等待宿主完成音频分配"，直到下一次重新加载。
    ///
    /// 这里钉住三件事：
    /// 1. 没有输入变化时，安全点什么都不做（不允许每 20 ms 白跑一次装配）；
    /// 2. 标记置位后，安全点必须产出 region → clip 的身份边；
    /// 3. clip 集合真的变了就必须推进 model 版本 —— 否则编辑器与渲染都不会重读，
    ///    装配发生了却等于没发生。
    #[test]
    fn a_changed_assembly_input_is_rebuilt_at_the_next_safe_point() {
        use hifishifter_kernel::editor::assembly::{RegionRow, TrackRow};
        use std::sync::atomic::Ordering;
        let document = document_with_one_empty_track(0.0);
        document.ara_model_seen.store(true, Ordering::Release);
        *document.region_tracks.lock_or_recover() = vec![TrackRow {
            id: "track".into(),
            name: "track".into(),
            order: 0,
            color: String::new(),
        }];
        *document.region_rows.lock_or_recover() = vec![RegionRow {
            key: 7,
            clip_id: "ara-region-7".into(),
            track_id: "track".into(),
            start_sec: 0.0,
            duration_sec: 2.0,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            name: "vocal".into(),
            source_id: "src".into(),
            modification_id: "mod".into(),
            source_sample_rate: 48_000.0,
            source_duration_sec: 2.0,
            source_channels: 1,
        }];
        // ① 没有任何输入变化 ⇒ 一个安全点都不该动它。
        document.flush_pending_assembly();
        assert!(
            document.clip_ids.lock_or_recover().is_empty(),
            "no input changed, so nothing may be assembled"
        );
        let revision = document.revision.load(Ordering::Acquire);
        // ② 输入变了（清单发布 / region 表发布）⇒ 下一个安全点必须把装配跑掉。
        document.note_assembly_input_changed();
        document.flush_pending_assembly();
        assert_eq!(
            document
                .clip_ids
                .lock_or_recover()
                .get(&7)
                .map(String::as_str),
            Some("ara-region-7"),
            "the new input must reach the clip set at the next safe point"
        );
        // ③ clip 集合变了 ⇒ model 版本前进（读者据此重读）。
        assert_eq!(
            document.revision.load(Ordering::Acquire),
            revision + 1,
            "a changed clip set must advance the model revision"
        );
        // ④ 标记已被消费：再冲一次不会重复推进版本。
        document.flush_pending_assembly();
        assert_eq!(
            document.revision.load(Ordering::Acquire),
            revision + 1,
            "the flag is consumed by one flush"
        );
    }

    /// **回归：正常的"这一轮还没认领"不许记成 WARN。**
    ///
    /// 【现场】20:04:15.431 一轮里 `attached=3 unbound=1`，15.647（213 ms 后）
    /// `attached=4 unbound=0`。这是 ARA 分轮认领的正常中间态，却以 WARN 出现，
    /// 而且不带"已经持续了多久"这个量 —— 与"把还没轮到写成出错了"是同一个病根。
    /// 现在只有连续 [`UNBOUND_WARN_STREAK`] 轮仍未认领才 WARN。
    #[test]
    fn an_unbound_region_only_warns_after_repeated_rounds() {
        use hifishifter_kernel::editor::assembly::{Assembly, RegionRow};
        let document = DocumentSession::new(900123);
        let rows = vec![RegionRow {
            key: 7,
            clip_id: "ara-region-7".into(),
            track_id: "track".into(),
            start_sec: 0.0,
            duration_sec: 2.0,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            name: "vocal".into(),
            source_id: "src".into(),
            modification_id: "mod".into(),
            source_sample_rate: 48_000.0,
            source_duration_sec: 2.0,
            source_channels: 1,
        }];
        let mut assembly = Assembly::default();
        assembly.unbound = vec![0];
        // 前两轮是中间态（DEBUG 档），第三轮才够 WARN。
        assert_eq!(
            document.track_unbound_streaks(&assembly, &rows, false)[0].1,
            1
        );
        assert_eq!(
            document.track_unbound_streaks(&assembly, &rows, false)[0].1,
            2
        );
        assert_eq!(
            document.track_unbound_streaks(&assembly, &rows, false)[0].1,
            UNBOUND_WARN_STREAK,
            "第三轮才够 WARN 档"
        );
        // 认领上了 ⇒ 账必须清掉；否则"失败 → 成功 → 又失败"会继承旧轮次直接报 WARN。
        assert!(document
            .track_unbound_streaks(&Assembly::default(), &rows, false)
            .is_empty());
        assert!(document.unbound_streaks.lock_or_recover().is_empty());
        assert_eq!(
            document.track_unbound_streaks(&assembly, &rows, false)[0].1,
            1,
            "清账之后重新从第一轮数起"
        );
        // 推迟的那一轮没有真的判定过：不计数，也不清账。
        assert!(document
            .track_unbound_streaks(&assembly, &rows, true)
            .is_empty());
        assert_eq!(document.unbound_streaks.lock_or_recover()[&7].0, 1);
    }

    /// **`unbound` 什么时候该推迟发布、什么时候就是结论。**
    ///
    /// 【为什么这条判据值得单测】它决定"一条本来有源的 clip 会不会被写成永久无源"。
    /// 判错的方向有两个，代价完全不同：
    ///   * 该推迟而不推迟 ⇒ 一次 100 ms 的竞态被写成终局（用户看到"一直在等宿主音频"，
    ///     重开工程才好）；
    ///   * 不该推迟而推迟 ⇒ 界面晚一帧（20 ms），且预算封住上界。
    ///
    /// 所以只有"输入确实动过"这一种情形才推迟，其余一律发布。
    #[test]
    fn assembly_defers_only_while_the_two_inputs_are_moving() {
        let now = std::time::Instant::now();
        let previous = Some((3_u64, 4_u64));
        // 输入动过 + 有 unbound ⇒ 推迟（这是"item 挪了、region 表还没跟上"的形状）。
        assert!(should_defer_assembly(1, previous, (4, 4), None, now));
        assert!(should_defer_assembly(1, previous, (3, 5), None, now));
        // 没有 unbound ⇒ 这一轮就是答案。
        assert!(!should_defer_assembly(0, previous, (4, 5), None, now));
        // 输入与上一次尝试逐位相同 ⇒ 再试一百次也是同一个结论。
        assert!(!should_defer_assembly(1, previous, (3, 4), None, now));
        // 还没有过上一次尝试 ⇒ 无从判断，先把这一轮发布出去（冷启动不能永远空白）。
        assert!(!should_defer_assembly(1, None, (0, 0), None, now));
    }

    /// 拖动 item 期间清单每一代都在变：预算必须封住上界，不能饿死界面。
    #[test]
    fn assembly_publishes_after_the_retry_budget() {
        let now = std::time::Instant::now();
        let started = now - ASSEMBLY_RETRY_BUDGET - std::time::Duration::from_millis(1);
        // 预算内：继续推迟。
        assert!(should_defer_assembly(
            2,
            Some((1, 1)),
            (2, 2),
            Some(now - std::time::Duration::from_millis(10)),
            now
        ));
        // 预算耗尽：照常发布（`unbound` 会连同两个代次进 WARN）。
        assert!(!should_defer_assembly(
            2,
            Some((1, 1)),
            (2, 2),
            Some(started),
            now
        ));
    }

    /// 文档级的旁通结论是**派生**的：没有实例 ⇒ 未知；实例的读数怎么合成由
    /// `fx_state::combine` 负责（它的用例钉住"只有全部确定被旁通才算旁通"）。
    ///
    /// 【为什么这里不再存一格】`DocumentSession` 是**每个 ARA 文档（工程）一份**，
    /// 而旁通是**每个 FX** 的事实。存一格等于让不同轨道的结论互相覆盖 —— 那会让读数
    /// 每轮在 `active` / `bypassed` 之间跳，每次跳都让渲染输入失效 ⇒ 每 100 ms 重算
    /// 一整次神经合成（"无限渲染"），而且一条轨被旁通会把另一条轨一起静音。
    #[test]
    fn the_document_bypass_verdict_is_derived_from_the_instances() {
        use crate::host::reaper::FxBypass;
        let document = DocumentSession::new(9_003);
        assert_eq!(
            document.host_bypass(),
            FxBypass::Unknown,
            "没有任何渲染实例时无从判断（必须等于改动前的行为：照常渲染）"
        );
    }

    /// **一条轨被旁通，不许把另一条轨一起静音。**
    ///
    /// 【为什么这是本轮最重要的一条回归】`DocumentSession` 每**文档**一份，而旁通是每
    /// **FX** 的事实。把结论存进文档那一格，等于让两条轨道的读数互相覆盖：
    ///   * 谁最后采样谁说了算 ⇒ 轨 2 会在轨 1 被旁通时一起静音（用户听到"整轨没声音"）；
    ///   * 两个结论每 100 ms 互相覆盖一次 ⇒ 每次覆盖都让渲染输入失效 ⇒ 每个实例每
    ///     tick 重算一整次神经合成（用户看到"无限渲染"）。
    ///
    /// 所以结论必须**派生**：只有全部播放实例都确定被旁通，才算这份工程不会出声。
    #[test]
    fn one_bypassed_track_never_silences_another() {
        use crate::host::reaper::FxBypass;
        let document = DocumentSession::new(9_004);
        let bypassed = Arc::new(ExtensionOwner::default());
        let other = Arc::new(ExtensionOwner::default());
        for owner in [&bypassed, &other] {
            owner
                .bind_to_document(
                    document.clone(),
                    ApiGeneration::V2Final,
                    ara2_bridge::plugin::ExtensionRoles::all(),
                    ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER
                        | ara2_bridge::plugin::ExtensionRoles::EDITOR_RENDERER,
                    None,
                )
                .expect("两个实例都必须能绑到同一个文档");
        }
        bypassed.set_host_bypass_for_tests(FxBypass::Bypassed);
        other.set_host_bypass_for_tests(FxBypass::Active);
        assert_eq!(
            document.host_bypass(),
            FxBypass::Active,
            "一条轨被旁通时另一条轨必须照常渲染"
        );
        // 全部旁通才是"这份工程不会出声"。
        other.set_host_bypass_for_tests(FxBypass::Bypassed);
        assert_eq!(document.host_bypass(), FxBypass::Bypassed);
        // 【没有读数 = 弃权，不是否决】REAPER 的播放实例是宿主按 item 现建的：它既没有
        // 视图定时器也没有模型收口，用户按下旁通的那一刻它很可能一次都没被采样过。
        // 把它当否决票，门就永远关着 —— 用户报的"旁通了还在渲染"。
        other.set_host_bypass_for_tests(FxBypass::Unknown);
        assert_eq!(
            document.host_bypass(),
            FxBypass::Bypassed,
            "从没读到过读数的实例不得否决别人已经确定的旁通结论"
        );
    }

    /// **播放实例从没报过读数时，编辑实例的那一票必须算数（REAPER 的真实形态）。**
    ///
    /// 【现场证据】2026-11-19 13:14 的 REAPER 会话里，用户按下旁通之后日志只有一行
    /// `writer 1 host bypass: active -> bypassed` —— 那是**编辑**实例（视图定时器每
    /// 100 ms 采一次）。播放实例（同一文档、同一 FX，由宿主按需现建）从头到尾**一行
    /// 都没有**：它没有视图定时器，而模型收口只在宿主改模型时发生。
    ///
    /// 【为什么会变成"旁通了还在渲染"】上一版的门只让 `renders_playback()` 的实例投票，
    /// 而唯一的播放实例读数永远是初始的 `Unknown` ⇒ 文档结论永远是 `Unknown` ⇒ 门
    /// 永远开着。这条用例把那个形态原样钉住。
    #[test]
    fn an_editor_bypass_reading_counts_when_the_playback_instance_never_reported() {
        use crate::host::reaper::FxBypass;
        let document = DocumentSession::new(9_006);
        let editor = Arc::new(ExtensionOwner::default());
        let playback = Arc::new(ExtensionOwner::default());
        editor
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ara2_bridge::plugin::ExtensionRoles::all(),
                ara2_bridge::plugin::ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .expect("编辑实例必须能绑到文档");
        playback
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ara2_bridge::plugin::ExtensionRoles::all(),
                ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .expect("播放实例必须能绑到同一个文档");
        // 让"编辑实例是 editor-only、播放实例才是渲染者"这件事成立（与宿主一致）。
        document.report_playback_ownership();
        assert!(!editor.renders_playback(), "编辑实例不应被当作渲染者");
        assert!(playback.renders_playback(), "播放实例才是渲染者");
        // 编辑实例读到了旁通；播放实例一次都没被采样过（缓存仍是初始的 `Unknown`）。
        editor.set_host_bypass_for_tests(FxBypass::Bypassed);
        assert_eq!(playback.host_bypass(), FxBypass::Unknown);
        assert_eq!(
            document.host_bypass(),
            FxBypass::Bypassed,
            "唯一能说话的那一票必须算数，否则门永远关不上"
        );
        // 播放实例真的读到 `active` 时，仍然必须照常渲染。
        playback.set_host_bypass_for_tests(FxBypass::Active);
        assert_eq!(document.host_bypass(), FxBypass::Active);
    }

    /// **过期读数不得再投票。**
    ///
    /// 【为什么这条与上一条不矛盾】"没有读数"与"读数过期"都必须回到"照常渲染"：
    /// 过期读数若继续阻止渲染，用户解除旁通之后就没有声音；若要它继续**允许**渲染，
    /// 就是"旁通了还在渲染"。所以它一律弃权 —— 而弃权在门上的表现就是 `Unknown`。
    #[test]
    fn an_expired_reading_stops_counting_as_a_bypass() {
        use crate::host::reaper::FxBypass;
        let document = DocumentSession::new(9_005);
        let owner = Arc::new(ExtensionOwner::default());
        owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ara2_bridge::plugin::ExtensionRoles::all(),
                ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .expect("实例必须能绑到文档");
        owner.set_host_bypass_for_tests(FxBypass::Bypassed);
        assert_eq!(document.host_bypass(), FxBypass::Bypassed);
        // 做旧到远超有效期：这一票作废，门回到"照常渲染"。
        owner.age_host_bypass_for_tests(std::time::Duration::from_secs(60));
        assert_eq!(
            document.host_bypass(),
            FxBypass::Unknown,
            "过期读数必须弃权（`Unknown` = 照常渲染），而不是继续挡住渲染"
        );
        // 重新采到 ⇒ 立刻恢复投票权。
        owner.set_host_bypass_for_tests(FxBypass::Bypassed);
        assert_eq!(document.host_bypass(), FxBypass::Bypassed);
    }

    /// **分割一个循环源 item，参数线一根都不许动（插件侧粘合层）。**
    ///
    /// 【为什么内核已有分类用例，这里还要一条】分类器只回答"这一帧算什么"；用户
    /// 看到的是"事务跑完之后权威里还剩什么"。这条把 `geometry_frame` →
    /// `reconcile_parameter_geometry_locked` → `apply_geometry_transaction` 串起来，
    /// 钉住"分割前后参数权威逐字节相等"——用户报的正是这条链断在最后一步：
    /// 分割被当成拉伸，整条曲线被压进半段。
    ///
    /// 【为什么 fixture 必须是循环源】REAPER 新建 item 默认 `B_LOOPSRC=1`，源窗口是
    /// `[锚点, 媒体末)` 而不是 `[锚点, 锚点+长度×倍率)`。少了这个前提，用例就绕开
    /// 了真正的缺陷（非循环源的分割本来就分类正确）。
    #[test]
    fn a_loop_source_split_leaves_the_authority_byte_identical() {
        let _serial = crate::settings_store::test_support::exclusive();
        let _ = crate::settings_store::save_settings_patch(&serde_json::json!({
            "lockParamLines": true
        }));
        let document = document_with_a_loop_clip();
        document
            .write_authority_range("track", "pitch", 0, &[61.0; 200], Some(&[true; 200]), false)
            .unwrap();
        let before = document.edits.lock_or_recover().authority.content_hash();
        assert!(
            reconcile_geometry_once(&document).is_empty(),
            "首次事务只记账（日志为空 ⇒ 没有 delta）"
        );
        // 宿主在 5 s 处分割：item 变短，循环源的源窗口不变（仍是 [0, 10)）。
        document.timeline.lock_or_recover().as_mut().unwrap().clips[0].length_sec = 5.0;
        let report = reconcile_geometry_once(&document);
        assert!(report.is_empty(), "分割不是拉伸：{report:?}");
        assert_eq!(
            document.edits.lock_or_recover().authority.content_hash(),
            before,
            "锁定参数线开启时，分割前后参数权威必须逐字节相等"
        );
    }

    /// **锁定关闭 ⇒ 宿主怎么改，参数线都一个字节不动。**
    ///
    /// 【为什么三种编辑一起打】用户的原话是"对 Clip 的编辑操作前后，参数线需要保持
    /// 完全不变"。移动、拉伸、分割在分类器里走的是三条不同的分支，逐条测只能证明
    /// "某一条没动"；只有同一帧里同时出现三种变化，才证明**整条判定被短路**了 ——
    /// 而这正是关锁该有的形状：不是"每个 delta 都不生效"，是"根本不算 delta"。
    #[test]
    fn host_edits_leave_the_authority_alone_when_the_lock_is_off() {
        let _serial = crate::settings_store::test_support::exclusive();
        let _ = crate::settings_store::save_settings_patch(&serde_json::json!({
            "lockParamLines": false
        }));
        let document = document_with_a_loop_clip();
        document
            .write_authority_range("track", "pitch", 0, &[61.0; 200], Some(&[true; 200]), false)
            .unwrap();
        let before = document.edits.lock_or_recover().authority.content_hash();
        assert!(reconcile_geometry_once(&document).is_empty());
        {
            let mut slot = document.timeline.lock_or_recover();
            let clip = &mut slot.as_mut().unwrap().clips[0];
            clip.start_sec = 3.0; // 移动
            clip.length_sec = 5.0; // 分割 / 裁剪
            clip.playback_rate = 0.5; // 拉伸
        }
        let report = reconcile_geometry_once(&document);
        assert!(
            report.is_empty(),
            "锁关闭时任何几何变化都不搬参数：{report:?}"
        );
        assert_eq!(
            document.edits.lock_or_recover().authority.content_hash(),
            before,
            "锁关闭时，宿主编辑前后参数权威必须逐字节相等"
        );
        // 恢复默认值，避免同进程的其他用例读到关锁状态。
        let _ = crate::settings_store::save_settings_patch(&serde_json::json!({
            "lockParamLines": true
        }));
    }

    /// **锁定开启 ⇒ 移动片段必须把该片段范围内的参数线整段搬走。**
    ///
    /// 【为什么这条与内核的 `move_only_changes_placement` 不重复】内核那条证明的是
    /// "给定 delta，事务会搬"；这条证明的是"宿主真的动了一个 item 之后，插件算出了
    /// 那条 delta 并落到权威上"。中间隔着 `geometry_frame` 与日志配对 —— 用户报的
    /// "锁开了线却没动"就死在这一段。
    #[test]
    fn moving_a_clip_carries_its_parameter_line_when_the_lock_is_on() {
        let _serial = crate::settings_store::test_support::exclusive();
        let _ = crate::settings_store::save_settings_patch(&serde_json::json!({
            "lockParamLines": true
        }));
        let document = document_with_a_loop_clip();
        document
            .write_authority_range("track", "pitch", 0, &[61.0; 200], Some(&[true; 200]), false)
            .unwrap();
        let period_ms = document.edits.lock_or_recover().authority.frame_period_ms;
        let frames_per_second = (1000.0 / period_ms).round() as u32;
        assert!(reconcile_geometry_once(&document).is_empty());
        document.timeline.lock_or_recover().as_mut().unwrap().clips[0].start_sec = 1.0;
        let report = reconcile_geometry_once(&document);
        assert_eq!(
            report.affected_runs, 1,
            "移动必须搬走那一段曲线：{report:?}"
        );
        assert!(
            !document.authority_has_runs_in("track", "pitch", 0, frames_per_second),
            "原位置不得留下残影"
        );
        assert!(
            document.authority_has_runs_in("track", "pitch", frames_per_second, 10),
            "曲线必须落在新位置"
        );
    }

    /// **【第三轮 P1】clip 之外的参数线必须能被擦掉。**
    ///
    /// 【为什么这是用户报的"右键拖拽回弹"】旧实现把重置区间夹到
    /// `target_param_frames`（= 最后一个 clip 的终点）。整条轨没有 clip 时它就是 0，
    /// 于是"在空白处画的线"永远 `count == 0` —— 返回 `ok:true` 却什么都没改，
    /// 前端下一次取数就把线取回来。
    #[test]
    fn restore_authority_range_erases_a_line_outside_all_clips() {
        let document = document_with_one_empty_track(2.0);
        document
            .write_authority_range(
                "track",
                "pitch",
                2_000,
                &[61.0; 10],
                Some(&[true; 10]),
                false,
            )
            .unwrap();
        assert!(document.authority_has_runs_in("track", "pitch", 2_000, 10));
        let value = document
            .restore_authority_range("track", "pitch", 2_000, 10, false)
            .unwrap();
        assert_eq!(value["ok"], serde_json::json!(true));
        assert_eq!(
            value["changed"],
            serde_json::json!(true),
            "擦掉真实存在的 run 必须报告 changed"
        );
        assert!(!document.authority_has_runs_in("track", "pitch", 2_000, 10));
        assert_eq!(document.authority_revision(), value["authority_revision"]);
    }

    /// **【第三轮 P2】越界到"工程尾之后"的区间同样是合法擦除域。**
    #[test]
    fn restore_authority_range_beyond_project_sec_still_erases() {
        let document = document_with_one_empty_track(1.0);
        document
            .write_authority_range("track", "pitch", 5_000, &[70.0], Some(&[true]), false)
            .unwrap();
        let value = document
            .restore_authority_range("track", "pitch", 5_000, 1, false)
            .unwrap();
        assert_eq!(value["changed"], serde_json::json!(true));
        assert!(document.edits.lock_or_recover().authority.is_pristine());
    }

    /// **【第三轮 P4】重复擦除必须报 `changed == false`，且不推进 revision。**
    #[test]
    fn restore_reports_changed_false_when_nothing_was_edited() {
        let document = document_with_one_empty_track(1.0);
        let before = document.authority_revision();
        let value = document
            .restore_authority_range("track", "pitch", 0, 100, false)
            .unwrap();
        assert_eq!(value["changed"], serde_json::json!(false));
        assert_eq!(value["authority_revision"], serde_json::json!(before));
        assert_eq!(document.authority_revision(), before);
    }

    /// **T7/T8：源暂时不可读不得删掉已发布的编辑侧 PCM。**
    ///
    /// 【为什么这是"后台切回前台之后 Clip 一直等音频"的回归】旧实现无论什么原因都先
    /// 把 PCM 删掉、再"若可读则重读"。宿主收回一次样本访问（后台、undo 历史、宿主
    /// 自己整理缓存）就等于永久丢掉音频，而恢复只依赖"宿主记得再发一次 enable" ——
    /// 那条回调丢了就再也不会来。
    ///
    /// 钉住三件事：① 收回时**渲染权威表**撤掉（不能继续播旧采样）、**编辑侧副本**保留
    /// （波形不消失）；② 状态记为 `Revoked`（界面据此说"暂时收回"而不是"还在分配"）；
    /// ③ 重新授权后状态回到 `Published`。
    /// **B0：诊断行必须指名断在哪一环，且同样的读数只打一次。**
    ///
    /// 【为什么钉这两件事】19 轮排查没有拿到过真机日志，一半原因是读数不成形、
    /// 另一半是日志被同一条重复行刷满。这一行是"下一份报告可读"的全部基础。
    #[test]
    fn the_diag_line_names_the_ring_that_broke_and_prints_once() {
        let document = Arc::new(DocumentSession::new(41));
        // 宿主交付了 item，但授权表是空的 —— 重开工程最典型的断点形状。
        document
            .region_items
            .lock_or_recover()
            .insert(7, "item-a".to_owned());
        let line = document.diag_line();
        assert!(line.starts_with("[hfs] diag v1"), "{line}");
        assert!(line.contains("items=1"), "{line}");
        assert!(line.contains("authorized=0"), "{line}");
        assert!(line.contains("pcm=0"), "{line}");
        assert!(line.contains("clips_src=0"), "{line}");
        // 授权表补上之后，读数必须跟着变（否则它回答不了"是不是授权没写"）。
        document
            .authorized_takes
            .lock_or_recover()
            .insert(7, "take-a".to_owned());
        assert!(document.diag_line().contains("authorized=1"));
        // 同样的读数不重复打印。
        document.log_diag_if_changed();
        let first = document.last_diag.lock_or_recover().clone();
        assert!(first.is_some());
        document.log_diag_if_changed();
        assert_eq!(document.last_diag.lock_or_recover().clone(), first);
    }

    /// **I3-1：解不开的存档也必须原样交回宿主。**
    ///
    /// 【为什么这是"参数全丢"的最后一环】解码失败时 `restore_archive_with_remap` 只
    /// 记一行日志就 `Ok(())` —— 宿主以为恢复成功。此后宿主任何一次保存都会问
    /// `store_document`，而旧实现用空 `edits` 重编码并**覆盖** `archive_cache`：
    /// 磁盘上从此只剩默认值，没有任何一步报错。
    #[test]
    fn an_undecodable_archive_is_still_handed_back() {
        let document = Arc::new(DocumentSession::new(31));
        let payload = b"{ this is not a valid archive".to_vec();
        document
            .restore_archive(&payload)
            .expect("host restore continues");
        assert_eq!(
            document.archive_cache.lock_or_recover().as_ref(),
            Some(&payload),
            "解码失败不得丢掉宿主交给我们的字节"
        );
        assert_eq!(document.encode_archive().unwrap(), payload);
    }

    /// **I3-2：载荷还没合入时，存档必须交回它，而不是用内存态覆盖。**
    ///
    /// 【为什么连续保存三次】第一次覆盖之后用户数据就不在磁盘上了；只测一次会漏掉
    /// "第一次侥幸还留着、第二次被固化"的那种退化。
    #[test]
    fn a_pending_restore_is_never_overwritten_by_the_in_memory_state() {
        let document = Arc::new(DocumentSession::new(32));
        let mut saved = crate::state_channel::EditState::default();
        saved.revision = 7;
        let bytes = saved.encode().unwrap();
        document.restore_archive(&bytes).expect("archive restores");
        for round in 0..3 {
            assert_eq!(
                document.encode_archive().unwrap(),
                bytes,
                "第 {round} 次保存改写了未合入的载荷"
            );
        }
    }

    #[test]
    fn a_revoked_source_keeps_its_editor_copy_and_reports_revoked() {
        use crate::render::document::SourceState;
        let document = Arc::new(DocumentSession::new(21));
        let id = "ara://source".to_string();
        let pcm = Arc::new(crate::render::source::SourcePcm {
            covered: Vec::new(),
            sample_rate: 44100,
            planes: vec![vec![0.1, 0.2, 0.3, 0.4]],
            version: 0,
            _reservation: None,
        });
        document.publish_source_pcm(id.clone(), pcm);
        document
            .source_states
            .lock_or_recover()
            .insert(id.clone(), SourceState::Published);
        // 宿主收回访问：渲染权威表撤掉，编辑侧副本保留。
        document.sources.lock_or_recover().remove(&id);
        document
            .source_states
            .lock_or_recover()
            .insert(id.clone(), SourceState::Revoked);
        assert!(
            document.edit_sources.lock_or_recover().contains_key(&id),
            "收回样本访问不得让界面丢掉已经物化的副本（那正是'整条轨突然空了'）"
        );
        assert!(
            !document.sources.lock_or_recover().contains_key(&id),
            "渲染权威表必须撤掉：宿主已经收回访问，不能继续播旧采样"
        );
        assert_eq!(
            document.source_states.lock_or_recover().get(&id),
            Some(&SourceState::Revoked),
            "状态必须如实记为 Revoked，界面才能说'暂时收回'而不是'还在分配'"
        );
        document.close();
        assert!(
            document.source_states.lock_or_recover().is_empty(),
            "文档关闭必须清空源状态表"
        );
    }

    /// **T23：宿主不再收口时，插件侧必须能把这件事变成一次可执行的请求。**
    ///
    /// 【为什么这是"删掉插件再加就好"的回归】`ready` 回到真只依赖宿主发起的四个
    /// 入口（`endEditing`、`notifyModelUpdates`、样本访问变更、存档恢复）。宿主一旦
    /// `beginEditing` 之后再也不收口，四个入口**一个都不会来** —— 插件永久停在未就绪，
    /// 而用户唯一能做的就是重装插件。
    ///
    /// 这里钉住三件事：① 会话开始的时刻被记下；② 超过宽限期后请求被置位；
    /// ③ 请求**取走即清**（否则每次回调都会重跑一次全量重映射）。
    #[test]
    fn an_overdue_edit_session_becomes_a_settle_request() {
        let document = Arc::new(DocumentSession::new(11));
        assert!(document.editing_elapsed().is_none(), "初始不在会话中");
        document.note_editing_started();
        assert!(document.editing_session.load(Ordering::Acquire));
        // 刚开始：还没到宽限期，不该请求。
        document.request_settle_if_overdue();
        assert!(!document.forced_edit_close.load(Ordering::Acquire));
        // 把起点推到宽限期之外（模拟"宿主再也不发 endEditing"）。
        document.backdate_settle();
        document.request_settle_if_overdue();
        assert!(
            document.forced_edit_close.load(Ordering::Acquire),
            "超过宽限期的会话必须变成一次可执行的收口请求"
        );
        // 取走即清：同一次请求只消费一次。
        assert!(document.forced_edit_close.swap(false, Ordering::AcqRel));
        assert!(!document.forced_edit_close.swap(false, Ordering::AcqRel));
        // 收口之后截止期归零，不再反复请求。
        document.note_editing_finished();
        document.request_settle_if_overdue();
        assert!(!document.forced_edit_close.load(Ordering::Acquire));
        document.close();
    }

    /// **T23：积压的模型变化同样有截止期。**
    ///
    /// 【为什么要单独一条】它与"会话没关"是两件不同的事：宿主可能根本不开编辑会话，
    /// 却在后台重建了图。那时唯一的 flush 入口是 `notifyModelUpdates`，宿主不调用就
    /// 永远不 flush —— 界面停在旧时间线上，而 `ready` 可能是真的（更隐蔽）。
    #[test]
    fn an_overdue_model_change_becomes_a_flush_request() {
        let document = Arc::new(DocumentSession::new(12));
        assert!(document.model_dirty_elapsed().is_none());
        document.note_model_dirty();
        document.note_model_dirty(); // 重复置脏不得推迟截止期
        let first = document.model_dirty_since_ms.load(Ordering::Acquire);
        assert_eq!(
            document.model_dirty_since_ms.load(Ordering::Acquire),
            first,
            "截止期从**第一次**变脏算起"
        );
        document.request_settle_if_overdue();
        assert!(!document.forced_model_flush.load(Ordering::Acquire));
        document.backdate_settle();
        document.request_settle_if_overdue();
        assert!(document.forced_model_flush.load(Ordering::Acquire));
        // 消费掉这一次请求，再看"干净之后不再请求"。
        assert!(document.forced_model_flush.swap(false, Ordering::AcqRel));
        document.note_model_clean();
        document.request_settle_if_overdue();
        assert!(
            !document.forced_model_flush.load(Ordering::Acquire),
            "已发布的模型不得再触发 flush 请求"
        );
        document.close();
    }

    /// **持着 `transaction` 刷新文档不得自锁。**
    ///
    /// 【为什么必须有这条】用户报的"重新打开含本插件的 ARA 工程，宿主直接未响应"
    /// 就是一次同线程自锁：`prepare_renderers` 持着 `transaction` 调了
    /// `adopt_archive_restore` → `refresh_document`，后者又要取同一把不可重入的锁。
    /// 这里把契约钉在 `refresh_document_locked` 上：它**不得**再取 `transaction`。
    ///
    /// 【为什么用线程 + 超时而不是直接调用】自锁的失败形态是"永远不返回"，直接调用
    /// 会把测试套件一起挂死；有超时才是一个可判定的断言。
    #[test]
    fn refreshing_while_the_transaction_is_held_does_not_deadlock() {
        use std::time::Duration;
        let document = Arc::new(DocumentSession::new(1));
        let owner = Arc::new(ExtensionOwner::default());
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = {
            let document = Arc::clone(&document);
            let owner = Arc::clone(&owner);
            std::thread::spawn(move || {
                let _transaction = document.transaction.lock_or_recover();
                owner.refresh_document_locked(&document);
                let _ = sender.send(());
            })
        };
        assert!(
            receiver.recv_timeout(Duration::from_secs(5)).is_ok(),
            "持锁刷新必须返回：`refresh_document_locked` 不得再取 `transaction`"
        );
        worker.join().expect("worker must not panic");
    }

    /// 真正的工厂必须在把控制器交给宿主前登记身份，空文档也必须可绑定。
    #[test]
    fn native_factory_registers_the_actual_controller_identity() {
        let factory = FactoryBuilder::new("org.hfs.identity", "org.hfs.identity.archive")
            .display(
                "identity",
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                "1",
            )
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap();
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"empty document".as_ptr(),
        };
        // SAFETY: factory、fixture、host 和属性在控制器整个生命周期内存活。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 工厂成功返回稳定 controller instance。
        let instance = unsafe { raw.read_unaligned() };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_some());
        // SAFETY: 唯一终止回调；之后不再访问控制器。
        unsafe {
            ((*instance.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(instance.documentControllerRef)
        };
        assert!(DocumentSession::lookup(instance.documentControllerRef as usize).is_none());
        factory.entry().uninitialize().unwrap();
    }
}

fn controllers() -> &'static Mutex<HashMap<usize, Weak<DocumentSession>>> {
    static DOCUMENTS: OnceLock<Mutex<HashMap<usize, Weak<DocumentSession>>>> = OnceLock::new();
    DOCUMENTS.get_or_init(Default::default)
}

/// 一次渲染的守卫票据：渲染**入口**处取得，收尾处复核。
///
/// 【为什么是四个计数器而不是投影指纹】渲染输入由 `ara_render_input` 从**文档自身**
/// 构造，与"编辑器读到什么"无关；而任何会让输入失效的变化都必然推进这四个数之一：
///   * 参数写入 / 撤销 ⇒ `edit`
///   * 宿主模型收口   ⇒ `model`
///   * 分配范围变化   ⇒ `scope`
///   * 几何 / 内容失效 ⇒ `epoch`
///
/// 旧实现把"编辑器上次读到的票据"（`loaded.edit` / `loaded.view.projection`）当守卫，
/// 于是**每一次参数写入都会让下一次渲染必然 Conflict** —— 而写参数正是请求渲染的那件
/// 事。票据必须由渲染入口自己取。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RenderTicket {
    pub edit: u64,
    pub model: u64,
    pub scope: u64,
    pub epoch: u64,
}

impl DocumentSession {
    /// Periodic host/UI queries must defer while a background transaction owns
    /// the document; waiting here can prevent the host from pumping that work.
    pub(crate) fn try_transaction(&self) -> Option<std::sync::MutexGuard<'_, ()>> {
        match self.transaction.try_lock() {
            Ok(guard) => Some(guard),
            Err(std::sync::TryLockError::Poisoned(error)) => Some(error.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }

    pub(crate) fn host_query_transaction(&self) -> Option<std::sync::MutexGuard<'_, ()>> {
        if NONBLOCKING_HOST_QUERY.with(std::cell::Cell::get) {
            self.try_transaction()
        } else {
            Some(self.transaction.lock_or_recover())
        }
    }

    pub(crate) fn publish_source_read_result(
        &self,
        id: String,
        result: Result<super::source::SourcePcm, String>,
        cancelled: &AtomicBool,
    ) {
        {
            let _transaction = self.transaction.lock_or_recover();
            if !self.is_alive() || cancelled.load(Ordering::Acquire) {
                return;
            }
            match result {
                Ok(pcm) => {
                    let pcm = Arc::new(pcm);
                    // 【同一代内容不重复发布】宿主每次重新授予样本访问（起播、前台恢复、
                    // 重同步）都会让插件把源**重读一遍**；读回来的是同一代内容
                    // （`version` 只在宿主通知内容变更时才前进），却不加区别地推进
                    // `editor_audio_revision` ⇒ 指纹必变 ⇒ 排一轮输出完全相同的重渲染
                    // ⇒ 状态栏"渲染中"随每次播放开始闪一下（REAPER 23:40:34 实测）。
                    // 同代重读是幂等的：留着手上这份，只把读数状态修好。
                    let superseded = match self.edit_sources.lock_or_recover().get(&id) {
                        Some(existing) if Self::same_pcm_generation(existing, &pcm) => {
                            log::debug!(
                                "[ara] source {id} re-read the same content generation {}; keeping the published PCM",
                                pcm.version
                            );
                            true
                        }
                        _ => false,
                    };
                    if !superseded {
                        self.edit_sources
                            .lock_or_recover()
                            .insert(id.clone(), pcm.clone());
                        self.sources.lock_or_recover().insert(id.clone(), pcm);
                        self.editor_audio_revision.fetch_add(1, Ordering::AcqRel);
                    }
                    self.source_states
                        .lock_or_recover()
                        .insert(id.clone(), SourceState::Published);
                    self.source_failures.lock_or_recover().remove(&id);
                    // 【同代重读也必须推进呈现代次】宿主每次重新授予样本访问都会重读一遍，
                    // 读回的是同一代内容（`superseded == true`）—— 于是
                    // `editor_audio_revision` 不动。但**界面**要的恰恰是这一次：
                    // 源从 Revoked 回到 Published，占位应当立刻变成波形。少了这一句，
                    // 日志里就是"re-read the same content generation 0"之后仍然 pending。
                    self.note_presentation_changed();
                }
                Err(error) => {
                    self.source_states
                        .lock_or_recover()
                        .insert(id.clone(), SourceState::Failed);
                    self.source_failures.lock_or_recover().insert(id, error);
                    // 真因同样要出现在界面上（`source_read_failed`），否则它要等到
                    // 下一次几何刷新才被看见 —— 而那时用户早已把"还在等"当成结论。
                    self.note_presentation_changed();
                }
            }
        }
        if cancelled.load(Ordering::Acquire) || !self.is_alive() {
            return;
        }
        let usable = self.edit_sources.lock_or_recover().len();
        let (published, failed) = {
            let states = self.source_states.lock_or_recover();
            (
                states
                    .values()
                    .filter(|state| **state == SourceState::Published)
                    .count(),
                states
                    .values()
                    .filter(|state| **state == SourceState::Failed)
                    .count(),
            )
        };
        if let Some(mut readiness) = self.readiness() {
            readiness.sources_published = published;
            readiness.sources_failed = failed;
            readiness.verdict = InputVerdict::decide(readiness.sources, published, failed, usable);
            self.note_readiness(readiness);
        }
        if let Some(editor) = self.existing_editor_session() {
            editor.wake_for_source();
        }
        if self.ready.load(Ordering::Acquire) {
            for owner in self.renderer_owners() {
                owner.prepare();
            }
        }
    }

    pub(crate) fn publish_source_pcm(&self, id: String, pcm: Arc<super::source::SourcePcm>) {
        let _transaction = self.transaction.lock_or_recover();
        // 【同一代内容不重复发布】见 `publish_source_read_result` 的说明：重读幂等。
        if let Some(existing) = self.edit_sources.lock_or_recover().get(&id) {
            if Self::same_pcm_generation(existing, &pcm) {
                // 内容没变，但"这一轮有 PCM 了"是界面要的事实（见
                // `publish_source_read_result` 的同一条说明）。
                self.note_presentation_changed();
                return;
            }
        }
        self.edit_sources
            .lock_or_recover()
            .insert(id.clone(), pcm.clone());
        self.sources.lock_or_recover().insert(id, pcm);
        self.editor_audio_revision.fetch_add(1, Ordering::AcqRel);
    }

    /// 两份源 PCM 是否同一代内容（同代重读 ⇒ 幂等，见发布点的说明）。
    ///
    /// 【为什么用内容代次指纹判】`version` 契约是"宿主改了内容必然通知"，但系统
    /// 既有口径（`refresh_source_fingerprints`）本来就承认"代次没变但内容被就地
    /// 改写"这一兜底形状 —— 去重只按 version+几何判会把它吞掉（late-PCM 测试
    /// `late_host_pcm_refreshes_existing_editor_without_model_change` 的形状：同
    /// version、内容不同，必须照常发布）。指纹与键刷新同源
    /// （`render::source::source_content_stamp`），两处口径永不漂移。
    fn same_pcm_generation(
        existing: &super::source::SourcePcm,
        incoming: &super::source::SourcePcm,
    ) -> bool {
        super::source::source_content_stamp(existing)
            == super::source::source_content_stamp(incoming)
            && existing.sample_rate == incoming.sample_rate
            && existing.planes.len() == incoming.planes.len()
            && existing
                .planes
                .iter()
                .zip(&incoming.planes)
                .all(|(a, b)| a.len() == b.len())
    }

    /// 宿主清单 → 装配器的**输入 B**（一条 item 一行）。
    ///
    /// 【为什么从 `ui_tracks` 现算而不是缓存一份】清单本身就是缓存
    /// （`ui_inventory_stamp` 管它的代次），再存一份只会多出一个"两份缓存不同步"的
    /// 失败模式。这里每次都从同一份清单派生，装配结果因此永远与"用户此刻在宿主里
    /// 看到的东西"一致。
    pub(crate) fn collect_item_rows(&self) -> Vec<hifishifter_kernel::editor::assembly::ItemRow> {
        let tracks = self.ui_tracks_guard();
        tracks
            .values()
            .flat_map(|track| {
                track
                    .items
                    .iter()
                    .map(|item| crate::host::geometry::item_row_from_host(item, &track.id))
            })
            .collect()
    }

    /// 记一次"装配的输入变了"（见 [Self::assembly_dirty]）。
    ///
    /// 唯一职责是置位；重装由 [Self::flush_pending_assembly] 在安全点上完成。
    pub(crate) fn note_assembly_input_changed(&self) {
        self.assembly_dirty.store(true, Ordering::Release);
    }

    /// 记一次"编辑器可见内容变了"（见 [Self::presentation_revision]）。
    ///
    /// 【为什么装配发布也要调】装配产出的 clip 集合/媒体身份是编辑器呈现的第一来源；
    /// 它变了而 `ui_geometry_revision` 不变（清单早就枚举到这个 item 了）时，旧实现
    /// 让 `refresh_ui_geometry` 早退 —— 已经画在画布上的无源占位永远不会被升级成
    /// 有源 clip。这正是"粘贴后一直显示正在等待宿主完成音频分配"的那条断链。
    pub(crate) fn note_presentation_changed(&self) {
        self.presentation_revision.fetch_add(1, Ordering::AcqRel);
    }

    /// 在安全点上把挂起的装配跑掉（不持任何相关锁时调用）。
    ///
    /// 【为什么用 `try_transaction` 而不是阻塞】调用方是 20 ms 的 UI 定时器与宿主写回
    /// 路径；后台 actor 正持事务时阻塞会把它要等的那次宿主泵堵死（见
    /// [Self::try_transaction] 的说明）。拿不到就留到下一个 tick —— 标记还在。
    ///
    /// 【为什么拿不到事务也要留标记】"这一轮没装上"不是"不需要装"：清掉标记会让一次
    /// 真正的输入变化永久丢失，而那正是本函数要消灭的那类缺陷。
    pub(crate) fn flush_pending_assembly(&self) {
        if !self.assembly_dirty.load(Ordering::Acquire) {
            return;
        }
        let Some(_transaction) = self.try_transaction() else {
            return;
        };
        if !self.is_alive() {
            return;
        }
        let before = self.clip_ids.lock_or_recover().clone();
        match self.rebuild_timeline_locked() {
            Ok(AssemblyOutcome::Published) => {}
            // 【推迟不是失败，也不是"做完了"】标记必须留着：它表达的正是"这份输入
            // 还没被消费"。清掉它等于把一次真正的输入变化永久丢失 —— 而那正是本函数
            // 要消灭的那类缺陷。
            Ok(AssemblyOutcome::Deferred) => return,
            Err(error) => {
                log::warn!("[ara] deferred timeline assembly failed: {error}");
                return;
            }
        }
        // 【clip 集合真的变了才算一次模型变化】`clip_ids` 是装配产物里唯一的身份投影
        // （region key → clip id）；它一变，读者必须重读：编辑器按 `revision` 判"要不要
        // 重新加载"，渲染按 `revision` 判"已发布的快照还对不对"。少了这一句，界面会
        // 一直显示上一次装配的片段 —— 而装配本身已经发生，症状与"没重装"一模一样。
        //
        // 【为什么在这里直接加而不是调 `clear_renderers`】本函数持有 `transaction`，
        // 而那个入口要取同一把不可重入的锁。这里只需要"版本前进一步"这一半语义：
        // 已发布的快照继续可听，下一次 `prepare_job` 会因为版本不匹配而重算。
        if *self.clip_ids.lock_or_recover() != before {
            self.revision.fetch_add(1, Ordering::AcqRel);
        }
        // 【为什么无条件推进】装配发布就是"编辑器该看的那份 clip 集合/媒体身份换了"。
        // 只在 `clip_ids` 变化时推进会漏掉最要命的一类：**clip 集合没变、媒体身份变了**
        // （占位 → 有源）。那正是粘贴后卡在"正在等待宿主完成音频分配"的形状 ——
        // id 一直是 `ara-item-{guid}`，变的是它背后的 `source_path`。
        self.note_presentation_changed();
    }

    /// **唯一**的时间线重建入口 —— clip 集合只在这里产生。
    ///
    /// 【为什么必须只有一个入口】此前 clip 有两个生产者：`ara/mapping.rs` 按 playback
    /// region 造一条，`present_host_inventory` 又按宿主 item 造一条。两者的关联判据是
    /// "四项几何全等且唯一"，而 REAPER 对一个**循环延伸**的 item 按循环节给出多条
    /// region —— 与整段 item 永远不全等。于是这些 region 既拿不到 item 身份、也不会被
    /// 任何归并收走：界面上多出 N 条片段，而且一条都拖不动（写回报
    /// `no unique directly bound host take`）。
    ///
    /// 现在：装配器决定"有哪些 clip"，本函数是它唯一的调用点；清单与 ARA 两侧都只写
    /// 各自的**输入表**（`ui_tracks` / `region_rows`）。
    ///
    /// 【锁序】调用方必须已持有 `transaction`。本函数先取两张输入表（宿主事实层级），
    /// 立刻释放，再取 `timeline` / `clip_ids` —— 与 `transaction → timeline → edits →
    /// host facts` 同向。
    pub(crate) fn rebuild_timeline_locked(&self) -> Result<AssemblyOutcome, String> {
        use hifishifter_kernel::editor::assembly::{
            assemble, attach, inherit_media, inherit_presentation,
        };
        // 【模型没交付过 ⇒ 不动已发布的时间线】见 `ara_model_seen` 的说明。装配器的
        // 输入只有两份，缺了 ARA 那一份时它只能给出"没有源窗口的占位" —— 用那份占位
        // 盖掉一份真实的图，比不刷新糟得多。
        if !self
            .ara_model_seen
            .load(std::sync::atomic::Ordering::Acquire)
            && self
                .timeline
                .lock_or_recover()
                .as_ref()
                .is_some_and(|timeline| !timeline.clips.is_empty())
        {
            // 这一轮什么也没做：清掉标记，等模型真正交付时由收口路径重装。
            self.assembly_dirty.store(false, Ordering::Release);
            return Ok(AssemblyOutcome::Published);
        }
        // 【代次必须在读表**之前**取】两个写者各自把计数 +1 再写表，所以"先取后读"
        // 只可能让记下的那一对比读到的**更旧** —— 那种情况下下一轮会被判成"输入动过"
        // 而多推迟一次，代价是一轮 20 ms；反过来（先读后取）会把"读表期间到达的新一代"
        // 记成已消费，于是那份输入永远等不到装配。
        let inputs = (
            self.ui_tracks_stamp.load(Ordering::Acquire),
            self.region_rows_stamp.load(Ordering::Acquire),
        );
        let rows = self.region_rows.lock_or_recover().clone();
        let tracks = self.region_tracks.lock_or_recover().clone();
        let mut items = self.collect_item_rows();
        // 【为什么要挂一次再装配】媒体总时长是**装配必需**的输入（倒放镜像轴、循环
        // 回绕周期都要它），而它只有两个来源：宿主 `GetMediaSourceLength`（清单层）
        // 或 ARA 的 `AudioSource.durationSeconds`（region 层，与授权无关、映射阶段就有）。
        // item 侧不知道自己的 region，region 侧不知道自己的 item —— 所以先挂一次把
        // 这份读数补上，再真正装配。`attach` 是纯函数，多跑一遍没有副作用。
        let attached = attach(&items, &rows);
        for (index, item) in items.iter_mut().enumerate() {
            let media = attached
                .iter()
                .enumerate()
                .find(|(_, owner)| **owner == Some(index))
                .and_then(|(row_index, _)| rows.get(row_index))
                .map(|row| row.source_duration_sec)
                .filter(|duration| duration.is_finite() && *duration > 0.0);
            let Some(media) = media else {
                continue;
            };
            for take in &mut item.takes {
                if take.geometry.media_duration_sec.is_none() {
                    take.geometry.media_duration_sec = Some(media);
                }
            }
        }
        let mut assembly = assemble(&items, &rows);
        // 【"有源"不许被一次竞态降级成"无源"】
        //
        // 上面那次 `attach` 用的是**这一轮**的两份输入。它们不是同一时刻的事实：
        // item 已经挪到新位置、region 表还是旧的（或反过来）时，一条本来挂得好好的
        // region 会落在 item 的**旧**矩形之外 ⇒ `unbound` ⇒ 这个 item 的 clip 被
        // `clip_for_item(item, track, None)` 写成无源 ⇒ 界面上波形与原始音高线一起
        // 消失，只剩一句"正在等待宿主完成音频分配"。
        //
        // 但"这条 item 用哪个源"是**文档级事实**：它不会因为 item 挪了 0.39 s 而改变。
        // 所以这里把上一轮已发布的媒体身份借回来（几何仍然取 item，判据见
        // `assembly::inherit_media`）。借不到的那些仍然是无源 —— 那是真的没有。
        let live_sources: std::collections::BTreeSet<String> = rows
            .iter()
            .map(|row| row.source_id.clone())
            .filter(|id| !id.is_empty())
            .collect();
        let previous_clips = self
            .timeline
            .lock_or_recover()
            .as_ref()
            .map(|timeline| timeline.clips.clone())
            .unwrap_or_default();
        let inherited = inherit_media(&mut assembly.clips, &previous_clips, &live_sources);
        // 【淡变与吸附偏移在同一处继承】见 `assembly::inherit_presentation`。装配的输入
        // 里根本没有这些字段（`ItemRow` 只有几何与媒体身份），所以「发布一条 clip」
        // 这个动作必须把推导不出来的东西一起带走 —— 否则每一次清单刷新都会把它们抹成
        // 0，再由 `project_ui_fades_locked` 补回来，而那一步只在宿主几何缓存这一轮
        // 有效时才说话，缓存恰好被这次发布本身推进的 `revision` 作废。
        let presented = inherit_presentation(&mut assembly.clips, &previous_clips);
        // 【`unbound` 不是结论，是"两份输入还没对齐"】
        //
        // 判据只有一条：**这一轮读到的两份输入，与上一次尝试相比动过吗**。
        //   * 动过 ⇒ 其中至少一份可能还没跟上另一份，值得在下一个安全点再看一眼；
        //   * 没动 ⇒ 再试一百次也是同一个答案：那就是结论，该发布（并报 WARN）。
        //
        // 推迟的代价是"界面晚 20~120 ms 才看到新片段"，收益是"不会把一次 100 ms 的
        // 竞态写成一条永久无源的 clip"。预算（[`ASSEMBLY_RETRY_BUDGET`]）保证拖动
        // item 期间（清单每一代都在变）不会无限推迟。
        {
            let now = std::time::Instant::now();
            let consumed = *self.assembly_consumed.lock_or_recover();
            let mut since = self.assembly_deferred_since.lock_or_recover();
            let defer =
                should_defer_assembly(assembly.unbound.len(), consumed, inputs, *since, now);
            if defer {
                if since.is_none() {
                    *since = Some(now);
                }
                drop(since);
                // 【"输入动过"这一轮也算被消费了】否则同一个代次会被反复判成"动过" ——
                // 下一次尝试就会带着同样的输入再推迟一次，预算被白白烧掉。
                *self.assembly_consumed.lock_or_recover() = Some(inputs);
                self.log_item_map(&assembly, &rows, inputs, true);
                return Ok(AssemblyOutcome::Deferred);
            }
        }
        // region key → clip id：由 `spans` 反算。**这是唯一写 `clip_ids` 的地方** ——
        // 此前它由映射层与清单层各写一次，于是"身份升级"窗口里两边各说一套，
        // `workspace_timeline_locked` 按 id 过滤会把全部 clip 滤掉。
        let mut ids = std::collections::HashMap::new();
        for (clip_id, row_indices) in &assembly.spans {
            for index in row_indices {
                if let Some(row) = rows.get(*index) {
                    if row.key != 0 {
                        ids.insert(row.key, clip_id.clone());
                    }
                }
            }
        }
        let mut timeline = assembly.timeline(&tracks, crate::ara::mapping::default_bpm());
        self.log_item_map(&assembly, &rows, inputs, false);
        // 这一轮真的发布了 ⇒ 推迟窗口结束（下一次竞态重新计时）。
        *self.assembly_consumed.lock_or_recover() = Some(inputs);
        *self.assembly_deferred_since.lock_or_recover() = None;
        if inherited > 0 {
            // 解释性跟踪：拖动 item 期间它每一轮都可能 > 0 —— 那正是这条继承在起作用，
            // 属于竞态的自愈，用户不需要为此做任何事，所以不是 WARN。
            log::debug!("[ara] inherited the media identity of {inherited} clip(s) this round");
        }
        if presented > 0 {
            // 与上面同一档：它是「这一轮有多少条 clip 的淡变靠继承活了下来」。
            // 拖动期间每轮都 > 0 是**正常**的（装配每轮重跑一次），不是故障。
            log::debug!("[ara] carried the published fades of {presented} clip(s) across this rebuild");
        }
        // 这一轮已经把**当前**两份输入都消费掉了（含中途到达的那一份）。
        self.assembly_dirty.store(false, Ordering::Release);
        *self.clip_ids.lock_or_recover() = ids;
        *self.assembly_spans.lock_or_recover() = assembly.spans.clone();
        *self.assembly_unbound.lock_or_recover() = assembly.unbound.clone();
        // 【默认合成开关：只对**没有持久化编辑记录**的轨道播种，不覆写用户的选择】
        //
        // 这段逻辑此前在 `remap_and_log` 里，只在 ARA 收口时跑一次。现在装配会在
        // **每次清单刷新**后重跑 —— 留在那里就等于"每次重跑都丢一次"。挪到这里，
        // 它才与"轨道从哪来"这件事同源。
        //
        // 【轨道级设置从**上一轮时间线**继承，不读 `edits`】
        //
        // 装配只产出几何；合成开关 / 分析算法 / 音量 / 静音 / 独奏 / 私有分组父级都是
        // **用户的设置**，重装一次不能把它们抹回默认值。
        //
        // 【为什么不读 `edits`】那会构成 "transaction → edits" 的反向边：工作区快照
        // 走的是 "edits → transaction"，两条路一交叉就是 ABBA 死锁（实测两个
        // `bound_tests` 用例因此永久挂起）。上一轮时间线里的值本来就是
        // `edits.apply` 的产物，继承它等价，而且不碰第二把锁。
        {
            let mut slot = self.timeline.lock_or_recover();
            let previous = slot.clone();
            for track in &mut timeline.tracks {
                let Some(old) = previous
                    .as_ref()
                    .and_then(|previous| previous.tracks.iter().find(|old| old.id == track.id))
                else {
                    // 新出现的轨道才播种插件默认值（开箱即用）。
                    track.compose_enabled = true;
                    track.pitch_analysis_algo =
                        hifishifter_kernel::state::PitchAnalysisAlgo::NsfHifiganOnnx;
                    continue;
                };
                track.compose_enabled = old.compose_enabled;
                track.pitch_analysis_algo = old.pitch_analysis_algo.clone();
                track.volume = old.volume;
                track.muted = old.muted;
                track.solo = old.solo;
                track.parent_id = old.parent_id.clone();
            }
            *slot = Some(timeline);
        }
        Ok(AssemblyOutcome::Published)
    }

    /// `[ara] item map` —— **真读数**。
    ///
    /// 【为什么旧的那一行回答不了它被加进来要回答的问题】旧实现从 `region_items` 算
    /// `max_regions_per_item`，而那张表的写入方自己去重（一个 item 只留一条 region）——
    /// 于是这个字段**结构上恒为 1**，20 次读数无一例外。现在从装配器的 `spans` 算，
    /// 它才是"一个 item 到底吃掉了几条 region"的真值；`unbound` 也是真的。
    fn log_item_map(
        &self,
        assembly: &hifishifter_kernel::editor::assembly::Assembly,
        rows: &[hifishifter_kernel::editor::assembly::RegionRow],
        inputs: (u64, u64),
        deferred: bool,
    ) {
        let items = assembly.clips.len();
        let attached: usize = assembly.spans.values().map(Vec::len).sum();
        let multi = assembly
            .spans
            .values()
            .filter(|rows| rows.len() > 1)
            .count();
        let max = assembly.spans.values().map(Vec::len).max().unwrap_or(0);
        let loops = assembly
            .clips
            .iter()
            .filter(|clip| clip.loop_enabled)
            .count();
        // 【连续未认领的轮次】见 `unbound_streaks`：只有"这一轮还没轮到"与"一直轮不到"
        // 分开，中间态才不会被记成故障。推迟的那一轮不计数（它没有真的判定过）。
        let unbound = self.track_unbound_streaks(assembly, rows, deferred);
        let max_streak = unbound
            .iter()
            .map(|(_, streak, _)| *streak)
            .max()
            .unwrap_or(0);
        log::info!(
            "[ara] item map: items={items} regions={} attached={attached} unbound={}                  multi={multi} max_regions_per_item={max} loops={loops} unmatched_max_streak={max_streak}",
            rows.len(),
            assembly.unbound.len()
        );
        // 【为什么这条读数必须出现在默认档】`verbosity=low` ⇒ `LevelFilter::Warn`
        // （`kernel/log_policy.rs`），所以上面那行 `info!` 在真机上**一行都没有**。
        // 而"一个 item 到底吃掉了几条 region、有几条挂不上"正是"粘贴出来的片段一直
        // 等宿主音频"这个问题的第一现场 —— 查了三轮才拿到实锤，就是因为这个。
        //
        // 只在**不干净**时发：稳态（每条 region 都挂上、每条 item 只吃一条）下这行
        // 没有任何信息量，而它会随每次清单刷新重跑。带上两个代次，是为了让"哪一份
        // 输入是旧的"从此不必再推。
        if !assembly.unbound.is_empty() || multi > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.item-map",
                format_args!(
                    "[ara] item map: items={items} regions={} unbound={} multi={multi}                      items_stamp={} regions_stamp={} deferred={deferred} unmatched_max_streak={max_streak}",
                    rows.len(),
                    assembly.unbound.len(),
                    inputs.0,
                    inputs.1,
                ),
            );
        }
        // 【为什么循环要单独一行】周期未知时内核会退化成"完全不循环、后半段静音"，
        // 而那在插件里无法与"宿主没给音频"区分 —— 用户只会以为素材坏了。
        for clip in assembly.clips.iter().filter(|clip| clip.loop_enabled) {
            log::info!(
                "[ara] loop clip {}: length={:.6} period={:?} media={:?} section={}",
                clip.id,
                clip.length_sec,
                clip.loop_period_sec,
                clip.takes.first().and_then(|take| take.duration_sec),
                clip.loop_period_sec.is_some(),
            );
        }
        if !assembly.unknown_loop_period.is_empty() {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.loop-period-unknown",
                format_args!(
                    "[ara] {} loop clip(s) have no resolvable wrap period ({}); they will not                      loop — the host reported neither a source section nor a source length",
                    assembly.unknown_loop_period.len(),
                    assembly.unknown_loop_period.join(",")
                ),
            );
        }
        // 【挂不上的 region 必须能归因】它们不画、不渲染，只进读数；不报出来就等于
        // "宿主给了、我们丢了"，而用户看到的只是"某一段不见了"。
        // 【推迟的那一轮不逐条报】下一轮要么已经挂上（这些 region 就不该再报），要么在
        // 发布时统一报一次（上面的汇总行）。逐条报会让同一条 region 每 20 ms 刷一行，
        // 而它恰好是"清单每一代都在变"的拖动期间。
        for (index, streak, age_ms) in unbound {
            let Some(row) = rows.get(index) else {
                continue;
            };
            // 【分档：中间态 DEBUG，连续多轮才 WARN】见 `unbound_streaks`。
            if streak < UNBOUND_WARN_STREAK {
                log::debug!(
                    "[ara] playback region {} (start={:.6} duration={:.6} source={}) is not inside any host item this round (streak={streak} age_ms={age_ms}); it produces no clip",
                    row.key,
                    row.start_sec,
                    row.duration_sec,
                    row.source_id
                );
                continue;
            }
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.region.unbound",
                format_args!(
                    "[ara] playback region {} (start={:.6} duration={:.6} source={}) is still not inside any host item after {streak} rounds (first seen {age_ms} ms ago); it produces no clip",
                    row.key, row.start_sec, row.duration_sec, row.source_id
                ),
            );
        }
    }

    /// 更新每个未认领 region 的连续轮次，返回 `(region 行下标, 轮次, 已持续毫秒)`。
    ///
    /// 【为什么把计数与输出分开】计数是纯状态判定，必须能脱离日志后端被断言 ——
    /// 与 `binding_report_is_new` 同一手法。它决定"正常中间态"会不会又被写成告警。
    fn track_unbound_streaks(
        &self,
        assembly: &hifishifter_kernel::editor::assembly::Assembly,
        rows: &[hifishifter_kernel::editor::assembly::RegionRow],
        deferred: bool,
    ) -> Vec<(usize, u32, u128)> {
        // 推迟的那一轮没有真的判定过（清单还在变），不计数、也不清账。
        if deferred {
            return Vec::new();
        }
        let mut streaks = self.unbound_streaks.lock_or_recover();
        let mut live = std::collections::HashSet::new();
        let mut out = Vec::new();
        for index in &assembly.unbound {
            let Some(row) = rows.get(*index) else {
                continue;
            };
            let entry = streaks
                .entry(row.key)
                .or_insert_with(|| (0, std::time::Instant::now()));
            entry.0 = entry.0.saturating_add(1);
            live.insert(row.key);
            out.push((*index, entry.0, entry.1.elapsed().as_millis()));
        }
        // 【认领上了就必须清账】否则"失败 → 成功 → 又失败"会继承旧轮次，把一次新的
        // 中间态直接报成 WARN，而那条 region 其实刚刚好过。
        streaks.retain(|key, _| live.contains(key));
        out
    }

    /// region key → clip id；`clip_ids` 缺席时按映射层同一算式补出来。
    ///
    /// 【为什么必须补而不是报错】`clip_ids` 由 `remap_and_log` 整体重写，而 region 的
    /// 分配（`addPlaybackRegion`）发生在**那之前**。两者之间任何一次渲染、提交或投影
    /// 都会查到缺席。把这个窗口从"致命错误"降成"按 key 派生"—— 派生的 id 与映射层
    /// 给的完全一致（`ara-region-{key}`），所以下一次重算会把它原样认领回去，
    /// 会话内身份保持单调稳定。
    ///
    /// `key == 0` 是无效键（`region_owners` 也拒它），此时无身份可派生，返回 `None`。
    pub(crate) fn clip_id_for(&self, key: u64) -> Option<String> {
        if key == 0 {
            return None;
        }
        Some(
            self.clip_ids
                .lock_or_recover()
                .get(&key)
                .cloned()
                .unwrap_or_else(|| format!("ara-region-{key}")),
        )
    }
    /// 记录宿主经 ARA 给出的 tempo map；`None` 表示宿主没提供。
    ///
    /// 【为什么要比较再写】投影缓存以它为键的一部分，而宿主可能在每次
    /// `updateMusicalContextContent` 上重复给同一份内容 —— 无条件写会让缓存永远失效、
    /// 每帧重建 tempo 点。
    pub(crate) fn set_host_tempo_map(&self, map: Option<crate::ara::model::HostTempoMap>) {
        let mut slot = self.host_tempo_map.lock_or_recover();
        if *slot == map {
            return;
        }
        *slot = map;
        drop(slot);
        // tempo map 变了 ⇒ 音乐投影必须重建（缓存键含它，这里只是立刻让读数可见）。
        log::info!("[ara] host tempo map updated");
    }

    /// 编码 ARA 存档载荷（文档级）。
    ///
    /// 【为什么文档级而不是 per-renderer】`storeDocumentToArchive` 是**文档控制器**的
    /// 回调，载荷应当覆盖整份文档的编辑；而 `ExtensionOwner::encode_state` 会按
    /// "本 renderer 被分配的 region"收窄 —— 那是 VST3 组件状态的语义（一个组件一份
    /// 状态），不是存档的语义。
    ///
    /// 【为什么先 reconcile】`TrackBindings` 是编辑归属的权威边；存档必须只带当前
    /// 模型图里真实存在的轨道/片段，否则重载后会出现指向不存在对象的编辑。
    pub(crate) fn encode_archive(&self) -> Result<Vec<u8>, String> {
        let _transaction = self.transaction.lock_or_recover();
        // 【I3：只要还有一份**没合入**的载荷，交回宿主的就是那一份】
        //
        // 这是"永不回退"最直接的一条实现，也是本轮把"修不好"与"丢数据"解耦的地方：
        // 内存里的 `edits` 此时**不是**用户的数据（载荷还没合进去），用它重编码再
        // 覆盖 `archive_cache`，等于在磁盘上把用户的参数换成默认值 —— 而宿主此后
        // 每一次保存都会把这个结果固化下去。
        //
        // 判据取"有没有未合入的载荷"，不取"合并成不成功"：合并成功时载荷已被取走，
        // 走下面的正常路径；合并不成功时载荷仍在，这里就把宿主当初给的那份原样还回去。
        if self.has_unmerged_restore() {
            let cached = self.archive_cache.lock_or_recover().clone();
            let pending = self
                .renderer_owners()
                .iter()
                .find_map(|owner| owner.pending_restore_bytes());
            if let Some(bytes) = cached.or(pending) {
                self.note_archive_unmerged(bytes.len());
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.archive.unmerged-returned",
                    format_args!(
                        "[ara] archive store returns the unmerged payload ({} byte(s)); the saved \
                         parameters are preserved instead of being replaced by the in-memory state",
                        bytes.len()
                    ),
                );
                return Ok(bytes);
            }
        }
        // 【为什么不再硬失败】`storeDocumentToArchive` 返回错误时宿主会认为"插件存档
        // 失败"，有些宿主会据此跳过这份工程状态。图未就绪（加载中的自动保存、关闭时
        // 保存）不是"没有编辑可存"，而是"这次还没合并完"；返回最后一份成功快照
        // 永远好过让宿主丢掉整块状态。
        if !self.ready.load(Ordering::Acquire) {
            if let Some(cached) = self.archive_cache.lock_or_recover().clone() {
                log::warn!(
                    "[ara] archive store uses the last good snapshot ({} byte(s)); the host \
                     graph is not ready",
                    cached.len()
                );
                return Ok(cached);
            }
            let edits = self.edits.lock_or_recover();
            return match edits.encode() {
                Ok(bytes) => Ok(bytes),
                Err(error) => {
                    // 【为什么退回"最后一份成功快照"而不是空块】空块是**可解析的空编辑**：
                    // 宿主会把它当成"这个工程没有任何用户参数"写进文件，而恢复端读到的
                    // 是一份合法的、全默认的状态 —— 没有任何一处会报错。那正是用户看到
                    // "重开工程后参数全丢"的最后一环。
                    log::error!("[ara] archive store degraded to the last good snapshot: {error}");
                    Ok(self
                        .archive_cache
                        .lock_or_recover()
                        .clone()
                        .unwrap_or_default())
                }
            };
        }
        let mut edits = self.edits.lock_or_recover();
        if let Err(error) = edits.reconcile(&self.track_bindings.lock_or_recover()) {
            // 身份暂时对不上不该让整份存档失败；保留未裁剪状态，恢复端会逐条 reconcile。
            log::warn!("[ara] archive store reconcile failed: {error}");
        }
        match edits.encode() {
            Ok(bytes) => {
                *self.archive_cache.lock_or_recover() = Some(bytes.clone());
                self.note_archive_stored(bytes.len());
                Ok(bytes)
            }
            Err(error) => {
                // 同上：能交回上一份成功载荷就绝不交空块。
                log::error!("[ara] archive store degraded to the last good snapshot: {error}");
                Ok(self
                    .archive_cache
                    .lock_or_recover()
                    .clone()
                    .unwrap_or_default())
            }
        }
    }

    /// 接收 ARA 存档载荷，等模型图就绪后合入。
    ///
    /// 【为什么只暂存】恢复到达时模型图通常还没重建（ARA 先给存档、宿主再重新交付
    /// 模型），此刻没有轨道可绑、也没有 renderer 可合并。暂存到文档级，
    /// `prepare_renderers` 收口时再交给 owner 的既有合并路径。
    pub(crate) fn restore_archive(&self, bytes: &[u8]) -> Result<(), String> {
        // 【为什么整份存档走空映射】ARA 只在**带筛选**的恢复里给出
        // `audioModificationArchiveIDs → audioModificationCurrentIDs`
        // （`ARARestoreObjectsFilter`，`ARAInterface.h:2322-2333`）；整份文档恢复
        // 没有这张表，因为 persistentID 本就是持久的。这里仍走同一条带映射的路径，
        // 于是"将来宿主给得出映射"时只有调用点需要改，两条路不会各自漂移。
        self.restore_archive_with_remap(bytes, &[])
    }

    /// 接收存档载荷并按 `archive → current` 重映射权威键后暂存（两条恢复路径共用）。
    fn restore_archive_with_remap(
        &self,
        bytes: &[u8],
        remap: &[(String, String)],
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        // 【I3-2：先存后解】在尝试解码**之前**就把宿主给的字节留在 `archive_cache` 里。
        //
        // 解码失败时下面的 `return Ok(())` 会让宿主以为恢复成功，而内存里什么都没有；
        // 此时若宿主再保存一次，旧实现会用空 `edits` 重编码并覆盖 cache —— 磁盘上
        // 从此只剩默认值。先把原始字节存下来，`encode_archive` 才有东西可以还回去。
        *self.archive_cache.lock_or_recover() = Some(bytes.to_vec());
        self.note_archive_restored(bytes.len());
        let mut restored = crate::state_channel::EditState::default();
        // 【为什么解码失败也只记日志】宿主把 `beginRestoringDocumentFromArchive`
        // 的失败当成"整份文档恢复失败"：region 不再交付、界面永远停在
        // `awaiting_regions`。丢一份编辑是局部损失，让宿主中止恢复是整份工程不可用。
        if let Err(error) = restored.restore(bytes) {
            log::error!(
                "[ara] archive restore ignored: {error}; the host document restore continues \
                 without these edits"
            );
            return Ok(());
        }
        // 【为什么必须重映射】persistentID 只保证**文档内**唯一，跨文档导入时宿主
        // 要求把存档 ID 映射到当前图里的 ID。不映射会把参数挂到一个不存在的键上 ——
        // 静默丢失，症状正是"重载工程后参数被重置"。
        // 【曲线不按 modification 重映射】权威的键是 (root, param, 工程秒)；跨文档
        // 导入要重映射的是**轨道身份**（由 `reconcile` 按 bindings 处理），与曲线无关。
        // 图集时代这里必须把 `mod:{旧ID}` 下的曲线搬到新键，因为曲线按 modification
        // 索引 —— 那份表示已经删除。
        let _ = &remap;
        // 【G5】记下"它到达时我们在不在一次编辑会话里"：撤销/重做的还原会话在概念上
        // 就是一次编辑会话（`ARAInterface.h:2938-2943`），那是它**必须**覆盖在线
        // 状态的唯一权威证据。
        restored.arrived_in_edit_session = self.in_edit_session();
        restored.from_archive = true;
        // 【为什么不再把 `restored` 重编码后写回 cache】恢复后到合入前，宿主可能再次
        // 保存（甚至还没等模型图 ready），需要一份"能交回去的东西" —— 那已经在函数
        // 开头存好了：**宿主给的原始字节**。再写一份重编码的副本有两个害处：
        // ① 它经过 `restore` 的 revision 前进，与宿主手里的那份不再逐字相同，
        //    "原样还回去"这条不变量就断了；
        // ② 它是"我们的"字节，而宿主下一次保存会把它当成权威 —— 一旦合并逻辑还有
        //    缺陷，被固化的就是我们的误解，而不是用户的数据。
        *self.pending_archive_restore.lock_or_recover() = Some(restored);
        // 【触发点①：存档到达】此前合并只有一个入口（`prepare_renderers`），而它在
        // "宿主不再收口"时不会再来 —— 于是载荷永远躺在队列里。这里排一次尝试：
        // 图已经就绪的场合（撤销、切屏）当场就能合入。
        if let Err(error) = self.merge_archive_restore() {
            log::warn!("[ara] archive merge on arrival failed: {error}");
        }
        Ok(())
    }
    /// 文档惰性持有唯一原编辑actor；actor只弱引用文档，组件不持第二份history。
    pub fn editor_session(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::editor::session::EditorSession>, String> {
        // UI command routing must not acquire the PCM/geometry transaction.
        // Session construction stores a Weak document and handles a concurrent
        // document close independently; the atomic liveness check is sufficient.
        // Liveness is atomic. Taking the PCM/geometry transaction here makes
        // every UI command wait behind a large document snapshot or analysis.
        // This lock never protected construction: it was released before
        // get_or_init, and the session already handles a concurrent close.
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        // 初始化期间文档可能被关闭：会话只弱引用文档，后续命令会自然返回
        // `DOCUMENT_CLOSED`，不需要在这里把构造失败当成终态。
        self.editor
            .get_or_init(|| crate::editor::session::EditorSession::new(self))
            .clone()
    }
    /// 编辑器会话是否已经创建（不创建、不取事务）。
    pub(crate) fn editor_initialized(&self) -> bool {
        self.editor
            .get()
            .is_some_and(|result| result.as_ref().is_ok())
    }

    /// 已发布片段数（健康探活用；不创建会话）。
    pub(crate) fn editor_clip_count(&self) -> u64 {
        self.editor
            .get()
            .and_then(|result| result.as_ref().ok())
            .map_or(0, |session| session.clip_count())
    }

    /// 已经创建的编辑器会话（**不创建**）。
    pub(crate) fn existing_editor_session(
        &self,
    ) -> Option<Arc<crate::editor::session::EditorSession>> {
        self.editor
            .get()
            .and_then(|result| result.as_ref().ok())
            .cloned()
    }

    /// 有预算的编辑器 flush；用于宿主保存线程，不能等 10 秒。
    pub(crate) fn flush_editor_with_budget(
        &self,
        budget: std::time::Duration,
    ) -> Result<(), String> {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.flush_with_budget(budget)?;
        }
        Ok(())
    }
    /// 组件撤销只关闭失效入口的view，不创建actor也不停止同文档其它入口。
    pub(crate) fn revoke_editor_views(&self) {
        if let Some(Ok(editor)) = self.editor.get() {
            editor.revoke_closed_views();
        }
    }
    /// edit/model/scope分开计数；assignment变更不能由相同model误判成未变。
    pub(crate) fn editor_versions(&self) -> Result<(u64, u64, u64), String> {
        let _transaction = self.transaction.lock_or_recover();
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        Ok((
            self.edits.lock_or_recover().revision,
            self.revision.load(Ordering::Acquire),
            self.scope_revision.load(Ordering::Acquire),
        ))
    }
    fn workspace_projection_locked(
        &self,
        edits: &crate::state_channel::EditState,
    ) -> Result<String, String> {
        let mut timeline = self.workspace_timeline_locked()?;
        edits.apply(&mut timeline);
        // 指纹必须反映权威合成后的显示；否则用户一笔写入后投影不变，
        // `workspace_view`/`accept_workspace_edits` 会拿旧水位判冲突。
        if !edits.authority.is_pristine() {
            let roots = timeline
                .tracks
                .iter()
                .filter(|track| track.parent_id.is_none())
                .map(|track| track.id.clone())
                .collect::<Vec<_>>();
            for root in roots {
                let view = self.compose_param_view_with(&timeline, edits, &root);
                timeline.params_by_root_track.insert(root, view.params);
            }
        }
        // 【为什么音阶也在指纹里】`ensure_loaded` 靠"版本没变 + 本指纹相同"早退
        // （`editor/session.rs:819-843`）。音阶与音阶变化点都是**会影响渲染结果**的
        // 插件自有状态（`scale-signature` 进渲染缓存键），却不在版本号里 —— 不进指纹
        // 就会出现"改了音阶，早退成立，于是既不重载也不重渲染"。与 `fades` 同列，
        // 理由相同。
        //
        // 【为什么可以整体哈希】音阶点的 id 是确定性的（见
        // `plugin_musical_projection`），宿主读数带 1e-6 阈值，所以"内容没变 ⇒
        // 指纹相同"成立，不会造成每轮重载。
        let mut audio_params = timeline.params_by_root_track;
        for params in audio_params.values_mut() {
            params.param_block_edits = Default::default();
            params.param_block_view = None;
        }
        audio_params.retain(|_, params| {
            !params.is_empty_project_data()
                || !params.pitch_orig.is_empty()
                || !params.dyn_orig.is_empty()
                || !params.tension_orig.is_empty()
        });
        let bytes = serde_json::to_vec(&(
            audio_params,
            timeline.tracks,
            &edits.fades,
            &edits.groups,
            // Syllable boundaries have a separate metadata event. They must
            // not invalidate audio projections or in-flight curve baselines.
            &timeline.project_scale_notes,
            &timeline.tempo_map,
            // 【为什么渲染事实也进指纹】`ensure_loaded` 靠"版本没变 + 本指纹相同"早退。
            // 方向 / 循环源 / 声道模式由 `project_host_take_facts_locked` 播种进**clips**
            // —— 而 clips 不在上面的哈希里。不进指纹就会出现"宿主倒放了，早退成立，
            // 于是既不重载也不重渲染"，与音阶进指纹是同一条理由。
            &*self.render_facts.lock_or_recover(),
        ))
        .map_err(|e| e.to_string())?;
        Ok(format!(
            "{}:{}",
            self.scope_revision.load(Ordering::Acquire),
            blake3::hash(&bytes).to_hex()
        ))
    }
    /// 【G5】当前是否处在一次 ARA 编辑会话里。
    pub(crate) fn in_edit_session(&self) -> bool {
        self.in_edit_session.load(Ordering::Acquire)
    }

    /// 【G5】此刻到达的载荷**是否应当覆盖**在线状态。
    ///
    /// 两种情形是"撤销/重做"，它们必须覆盖：
    ///   * ARA 的还原会话（`beginEditing` … `endEditing`）—— SDK 明说它概念上就是
    ///     一次编辑会话（`ARAInterface.h:2938-2943`）；
    ///   * 宿主正在一个**原生 undo 组**里 —— 组件 `setState` 在那种时刻就是 undo 的
    ///     回放，而不是"宿主把刚才那份还给我"。
    ///
    /// 其余时刻到达的载荷按"补缺"处理：它可能比在线状态**旧**，覆盖就是回退。
    pub(crate) fn authoritative_restore(&self) -> bool {
        self.in_edit_session.load(Ordering::Acquire)
            || self.host_undo.pending.load(Ordering::Acquire)
    }

    /// 【G5】编辑会话开关（`ModelHandle::begin_editing` / `end_editing` 各调一次）。
    pub(crate) fn set_in_edit_session(&self, editing: bool) {
        self.in_edit_session.store(editing, Ordering::Release);
    }

    /// 登记一条"本实例自己建出来的宿主轨道"。
    ///
    /// 重复登记是幂等的；超过上限时丢最旧的（见字段注释里"为什么不自动裁剪"）。
    pub(crate) fn adopt_track(&self, guid: &str) {
        /// 一张文档最多登记多少条。一个会话里用户手点几十次建轨已经很少见，
        /// 64 足够；它同时是这张表的硬上界，防止长时间运行后无界增长。
        const MAX_ADOPTED: usize = 64;
        if guid.is_empty() {
            return;
        }
        let mut adopted = self.adopted_tracks.lock_or_recover();
        if adopted.iter().any(|known| known == guid) {
            return;
        }
        adopted.push(guid.to_owned());
        if adopted.len() > MAX_ADOPTED {
            let excess = adopted.len() - MAX_ADOPTED;
            adopted.drain(..excess);
        }
    }

    /// 撤销一次登记（轨道没建成 / 提交失败 / 被显式删除）。
    ///
    /// 【为什么需要显式撤销，而不是"等下一轮枚举自然丢掉"】采纳表是**枚举范围**的输入：
    /// 一条指向已不存在轨道的 GUID 只是多一次线性扫描，无害；但一次"提交失败却留在表里"
    /// 的登记会在中途那一轮把一条**半成品轨道**枚举进来，时间线上就凭空多出一行。
    /// 显式撤销让"撤销"这件事与创建在同一次请求里完成，中间不留窗口。
    pub(crate) fn unadopt_track(&self, guid: &str) {
        if guid.is_empty() {
            return;
        }
        self.adopted_tracks
            .lock_or_recover()
            .retain(|known| known != guid);
    }

    /// 当前登记表快照（清单枚举用；返回副本以免持锁回调宿主）。
    pub(crate) fn adopted_track_ids(&self) -> Vec<String> {
        self.adopted_tracks.lock_or_recover().clone()
    }

    /// 登记一条待定导入（导入命令在宿主返回 item GUID **之后**调用）。
    ///
    /// 【为什么判重】回执可能被重投（媒体文档重开、视图重连）；同一个 item 登记两次
    /// 只会让 `rounds` 翻倍，读起来像是两件事。
    ///
    /// 【为什么这里可以阻塞取锁】这是**写**：登记一旦丢掉，这条导入就再也不会转正，
    /// 用户会在 60 s 后收到一条 `import_stalled`，而 item 明明在宿主里。所以宁可等。
    /// 等得起的前提是这把锁是**叶子**：它的持有者只做纯计算（见
    /// `take_resolved_imports`），不取别的锁、不做 I/O、不向 WebView 投递。
    pub(crate) fn begin_pending_import(&self, pending: PendingImport) {
        let mut imports = self.pending_imports.lock_or_recover();
        if imports.iter().any(|entry| entry.item == pending.item) {
            return;
        }
        imports.push(pending);
    }

    /// 对账一次，并把转正的每一条记一行、发一条 `import_ready`。
    ///
    /// 【`claims` 是参数不是自取】见 [`ImportClaims`]。这条约束是**结构性**的：本函数
    /// 不再取任何 `DocumentSession` 的表锁，于是"在持锁的作用域里调它"不可能变成自锁。
    pub(crate) fn publish_pending_imports(&self, claims: &ImportClaims) {
        let Some(resolved) = self.take_resolved_imports(claims) else {
            return;
        };
        if resolved.is_empty() {
            return;
        }
        for done in &resolved {
            // 【为什么要这一行】`[import] pending` 说"登记了"，这一行说"转正了，
            // 用了多久"。两者之间没有任何别的观测点，现场只能靠这两行的差判断宿主节奏。
            log::info!(
                "[import] ready item={} clip={} rounds={} after_ms={} source={}",
                done.item,
                done.clip_id,
                done.rounds,
                done.elapsed.as_millis(),
                done.source_path
            );
        }
        if let Some(editor) = self.existing_editor_session() {
            editor.announce_import_ready(&resolved);
        }
    }

    /// 对账一次：把已经被 ARA 图（或宿主清单）认领的待定导入转正，返回本轮转正的条目。
    ///
    /// 给测试用；生产路径只有 `publish_pending_imports`（它要顺手把事件播报出去）。
    #[cfg(test)]
    pub(crate) fn resolve_pending_imports(&self, claims: &ImportClaims) -> Vec<ResolvedImport> {
        self.take_resolved_imports(claims).unwrap_or_default()
    }

    /// 对账的实现：**只取 `pending_imports` 这一把锁**，并在同一次持锁里完成
    /// "读候选 → 判定 → 划掉"。
    ///
    /// 【为什么三步必须在一次持锁里】"报了却没划掉"会让同一条导入在下一轮再发一次
    /// `import_ready`，前端于是重复做一次规格化（用户勾了"导入后规格化"会得到一串增益）。
    /// 返回 `None` = 账本此刻被别人持有，这一轮什么都没做，下一次发布再来。
    fn take_resolved_imports(&self, claims: &ImportClaims) -> Option<Vec<ResolvedImport>> {
        // 【为什么拿不到就走】对账是**机会性**的：每一次 ARA 发布、每一次清单刷新都会
        // 再来一次。拿不到账本只有两种可能：① 有别的线程正在写它（下一个 tick 就好）；
        // ② 我们自己已经持有它（那是缺陷，但绝不能以"挂死宿主"的形式表现）。
        // 两种情况都只能得出同一个结论：这一轮不动，留一行可归因的字。
        let Some(mut imports) = try_lock_or_recover(&self.pending_imports) else {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "import.reconcile.busy",
                format_args!(
                    "[import] reconcile skipped: the pending ledger is busy \
                     (retried on the next publish)"
                ),
            );
            return None;
        };
        if imports.is_empty() || !self.is_alive() {
            return Some(Vec::new());
        }
        let mut resolved = Vec::new();
        for entry in imports.iter_mut() {
            entry.rounds = entry.rounds.saturating_add(1);
            let Some(clip_id) = claims.clip_for_item(&entry.item) else {
                continue;
            };
            resolved.push(ResolvedImport {
                item: entry.item.clone(),
                source_path: entry.source_path.clone(),
                clip_id,
                landed_track: entry.landed_track.clone(),
                rounds: entry.rounds,
                elapsed: entry.requested_at.elapsed(),
            });
        }
        if !resolved.is_empty() {
            imports.retain(|entry| !resolved.iter().any(|done| done.item == entry.item));
        }
        Some(resolved)
    }

    /// 取宿主清单（`ui_tracks`）。**本仓对这张表的所有访问都走这里**。
    ///
    /// 【为什么要包一层】它是唯一一张"被长时间持有、而且持有期间还要调用别的函数"的
    /// 表：`present_host_inventory` 拿着它跑完整个投影。一次重入就是宿主级挂死
    /// （REAPER 现场：导入成功之后整机未响应），而挂死的样子与"什么都没发生"完全同形。
    /// `lock_tracked` 在调试构建里让重入**当场点名**（见 `hifishifter_plugin_core::sync`），
    /// 发布构建里与直接取锁逐字等价。
    #[track_caller]
    pub(crate) fn ui_tracks_guard(
        &self,
    ) -> crate::sync::TrackedGuard<'_, std::collections::BTreeMap<String, crate::host::UiTrack>>
    {
        crate::sync::lock_tracked(&self.ui_tracks)
    }

    /// 一份对账输入（三张表的快照）。
    ///
    /// 【调用方不得持有任何 `DocumentSession` 锁】它自己会取三把；而在**已经持有**
    /// `ui_tracks` 的作用域里调用它就是自锁。需要"边算边用"的路径（`present_host_inventory`）
    /// 应当自己按 [`ImportClaims`] 的形状就地组一份。
    pub(crate) fn import_claims(&self) -> ImportClaims {
        ImportClaims {
            present_items: self
                .ui_tracks
                .lock_or_recover()
                .values()
                .flat_map(|track| track.items.iter().map(|item| item.geometry.item_id.clone()))
                .collect(),
            clip_ids: self.clip_ids.lock_or_recover().clone(),
            region_items: self.region_items.lock_or_recover().clone(),
        }
    }

    /// 这条路径是不是一条**还没到**的导入所写下的源？
    ///
    /// 【它为什么不是放松安全边界】它只承认"我们自己刚为某个宿主 item 写下的路径"，
    /// 而那条记录是导入命令在宿主**已经建成 item 之后**才写的 —— 证据来自宿主。
    ///
    /// 【读不到账本时回答"不是"】这是**安全边界**的一侧：认不出这条路径时只能拒绝。
    /// 账本被持有是微秒级的事，而拒绝的后果只是"这一拍波形读不到" —— 前端对
    /// `source is not authorized` 本来就有"未就绪 + 有界重试"的兜底。
    /// 反过来猜"是"会把一条与本次会话无关的路径放进来。
    pub(crate) fn pending_import_for_source(&self, path: &str) -> bool {
        try_lock_or_recover(&self.pending_imports).is_some_and(|imports| {
            imports
                .iter()
                .any(|entry| same_source_path(&entry.source_path, path))
        })
    }

    /// 取走"等了太久、还没转正"的导入（每一条只报一次，**不改回执、不删占位**）。
    pub(crate) fn take_stalled_imports(
        &self,
        after: std::time::Duration,
    ) -> Vec<(String, std::time::Duration)> {
        // 【为什么这里也不能阻塞】调用方是宿主 UI 线程上的 20 ms 视图定时器。账本被
        // 别人持有的那一拍直接跳过 —— 下一条 `ui_tick` 会再来，而"报告超期导入"
        // 本来就不需要精确到某一块。
        let Some(mut imports) = try_lock_or_recover(&self.pending_imports) else {
            return Vec::new();
        };
        let mut stalled = Vec::new();
        for entry in imports.iter_mut() {
            if entry.stall_reported {
                continue;
            }
            let elapsed = entry.requested_at.elapsed();
            if elapsed < after {
                continue;
            }
            entry.stall_reported = true;
            stalled.push((entry.item.clone(), elapsed));
        }
        stalled
    }

    /// scope版本参与投影指纹；区域退出后即便edit/model没动也不能接受旧工作区。
    pub(crate) fn workspace_projection(&self) -> Result<String, String> {
        let _transaction = self.transaction.lock_or_recover();
        self.workspace_projection_locked(&self.edits.lock_or_recover())
    }

    /// 一份**自洽**的工作区视图：指纹与逐根轨参数在同一事务里读出。
    ///
    /// 【为什么有它】基线必须与它描述的那份数组同代。分两次读（先指纹、再数组）会
    /// 在两次之间被宿主推进，得到的视图谁都不认识。这个入口只给"就地补丁后换基线"
    /// 之类的调用方用；正常的读路径走 [`Self::workspace_snapshot`]。
    pub(crate) fn workspace_view(&self) -> Result<Arc<WorkspaceView>, String> {
        let _transaction = self.transaction.lock_or_recover();
        let mut edits = self.edits.lock_or_recover();
        let mut timeline = self.workspace_timeline_locked()?;
        // 【锁序】`edits → timeline → baseline → host facts`（文档声明的顺序）。
        edits.apply(&mut timeline);
        let geometry = self.ui_geometry_revision.load(Ordering::Acquire);
        self.compose_root_view_locked(&mut edits, &mut timeline)?;
        self.resynced_geometry.store(geometry, Ordering::Release);
        let projection = self.workspace_projection_locked(&edits)?;
        Ok(Arc::new(WorkspaceView {
            projection,
            params: Arc::new(timeline.params_by_root_track),
            geometry_revision: geometry,
        }))
    }
    /// 原GUI改变素材选择只换可见source投影；不修改参数authority、history或渲染generation。
    pub(crate) fn selected_source_parameters(
        &self,
        selection: &hifishifter_kernel::state::TimelineState,
    ) -> Result<
        std::collections::BTreeMap<String, hifishifter_kernel::state::TrackParamsState>,
        String,
    > {
        let _transaction = self.transaction.lock_or_recover();
        if !self.is_alive() {
            return Err(crate::editor::errors::DOCUMENT_CLOSED.error());
        }
        let host = self.workspace_timeline_locked()?;
        Self::validate_workspace_geometry(&host, selection)?;
        let edits = self.edits.lock_or_recover();
        let mut grouped = host;
        edits.apply(&mut grouped);
        grouped.selected_clip_id = selection.selected_clip_id.clone();
        self.project_private_group_view(&mut grouped, &edits)?;
        Ok(grouped.params_by_root_track)
    }
    /// 把「锁定参数线」的当前设置落到**既有**权威上（纯元数据、幂等）。
    ///
    /// 【为什么它在投影之前跑】设置变了 ⇒ 曲线的**可见位置**必须立刻改变，而这不需要
    /// 一次模型收口、不需要烘焙、也不需要用户先分割一次。旧实现只在
    /// `remap_and_log`（模型收口）里读这个设置，于是"移动 item 不跟随、分割之后才对"。
    ///
    /// 【为什么按设置代次早退】本函数在每次读工作区时都被调用（33–400 ms 一次）。
    /// 代次没变时它只付一次原子读的代价。
    ///
    /// 【L5：代次只在**真的落地**之后推进】旧实现无论 `set_placement` 有没有改动都写
    /// stamp，于是"这一轮没有活几何"会让开关被永久吞掉。现在只要还有权威在等几何，
    /// 就不推进代次 —— 下一轮还会再试。
    /// 从图集 + 基线**合成**这一轮每根轨的显示数组，并装进时间线。
    ///
    /// 【为什么必须与读视图在同一个事务里】`timeline.params_by_root_track` 既是编辑器
    /// 看到的数组，也是 `workspace_projection_locked` 指纹的一部分。只要几何变了
    /// （方向 / Loop / 裁剪 / 分割）而这一层没重建，指纹就与"编辑器真正看到的"脱钩 ——
    /// 那正是"参数线停在旧方向"与"提交闸在错误的时间点判定冲突"的同一个根。
    ///
    /// 【为什么是"整根替换"而不是"按帧合并"】合成的输入已经包含基线层：未编辑帧就是
    /// 这一代分析结果。因此输出**就是**完整显示，不需要再猜"这一轮哪些帧有信息"。
    pub(crate) fn compose_root_view_locked(
        &self,
        edits: &mut crate::state_channel::EditState,
        timeline: &mut hifishifter_kernel::state::TimelineState,
    ) -> Result<(), String> {
        self.materialize_views_locked(edits, timeline);
        Ok(())
    }

    /// 把基线层装进时间线的派生字段（显示层的唯一来源）。
    pub(crate) fn install_baseline_locked(
        &self,
        timeline: &mut hifishifter_kernel::state::TimelineState,
    ) {
        let baseline = self.baseline.lock_or_recover();
        for (root, base) in baseline.iter() {
            if !base.is_ready() {
                continue;
            }
            let params = timeline
                .params_by_root_track
                .entry(root.clone())
                .or_default();
            params.frame_period_ms = base.frame_period_ms;
            params.pitch_orig = base.pitch_orig.clone();
            params.dyn_orig = base.dyn_orig.clone();
            params.tension_orig = base.tension_orig.clone();
            params.pitch_orig_key = base.pitch_key.clone();
            params.dyn_orig_key = base.dyn_key.clone();
        }
    }

    /// 这一代基线的快照（随视图交付给提交闸）。
    pub(crate) fn baseline_snapshot(
        &self,
    ) -> std::collections::BTreeMap<String, hifishifter_kernel::editor::param_authority::RootBaseline>
    {
        self.baseline.lock_or_recover().clone()
    }

    /// 发布一条根轨的新基线（分析流水线的**唯一**写入口）。
    ///
    /// 【为什么不在这里发通知】它可能跑在 ARA 模型线程上；回调宿主会重入。
    /// 通知由调用方在事务外发（见 `EditorSession::complete_cached_analysis`）。
    pub(crate) fn publish_baseline(
        &self,
        root: &str,
        baseline: hifishifter_kernel::editor::param_authority::RootBaseline,
    ) -> bool {
        let mut map = self.baseline.lock_or_recover();
        let changed = map.get(root).is_none_or(|old| {
            old.pitch_orig != baseline.pitch_orig
                || old.dyn_orig != baseline.dyn_orig
                || old.tension_orig != baseline.tension_orig
                || old.frame_period_ms != baseline.frame_period_ms
                || old.pitch_key != baseline.pitch_key
                || old.dyn_key != baseline.dyn_key
                // 【布局键必须参与比较】数组完全相同时（例如只移动了 clip 但内容
                // 恰好一致），键变了也意味着"这份基线描述的是旧布局"，必须重发。
                || old.layout_key != baseline.layout_key
                || old.content_key != baseline.content_key
        });
        if changed {
            let mut baseline = baseline;
            baseline.revision = map
                .get(root)
                .map(|old| old.revision.saturating_add(1))
                .unwrap_or(1);
            map.insert(root.to_owned(), baseline);
        }
        changed
    }

    /// 取一份**当前布局**的基线；必要时从内容缓存同步重组装并发布。
    ///
    /// 【为什么是纯函数式】输入只有 `timeline`（几何 + 轨道设置）与内容缓存。
    /// 绝不读 `edits`、绝不读显示数组 —— 那正是"用户编辑被固化成原始值"的入口。
    ///
    /// 【为什么可以同步做】`assemble_*_from_cache` 只是按当前 clip 窗口把 per-clip
    /// 缓存曲线搬进根数组（memcpy 级），不跑任何推理。代价与"取一次参数数组"同级。
    pub(crate) fn baseline_for(
        &self,
        root: &str,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> hifishifter_kernel::editor::param_authority::RootBaseline {
        use hifishifter_kernel::editor::param_authority::RootBaseline;
        use hifishifter_kernel::pitch_analysis::schedule::assemble_pitch_orig_from_cache;
        use hifishifter_kernel::pitch_analysis::{
            build_root_pitch_key, build_root_pitch_layout_key,
        };
        let layout = build_root_pitch_layout_key(timeline, root);
        let stored = self.baseline.lock_or_recover().get(root).cloned();
        if let Some(base) = &stored {
            if base.is_ready() && base.layout_key.as_deref() == Some(layout.as_str()) {
                return base.clone();
            }
            // 【已有基线但布局变了】不要在这里反复尝试文档侧组装：ARA 文档时间线的
            // `source_path` 是 ARA source 的 persistent id（不是文件路径），内容缓存
            // 按真实路径建键，组装必然全 0；每帧重试等于每帧分配一条整长数组。
            // 正确的做法是等分析侧（它有真实路径）发布带新布局键的基线。
            if base.is_ready() && base.layout_key.is_some() {
                let mut stale = base.clone();
                stale.analysis_pending = true;
                return stale;
            }
        }
        let mut rebuilt = RootBaseline {
            frame_period_ms: timeline.frame_period_ms(),
            layout_key: Some(layout),
            content_key: stored.as_ref().and_then(|base| base.content_key.clone()),
            ..Default::default()
        };
        // 【全 0 不算就绪】`assemble_pitch_orig_from_cache` 在没有 clip 时也返回一列 0；
        // 把它当 ready 会让 `mixdown` 的 `is_empty` 守卫失效（"看起来有源曲线、
        // 实际全是 0" ⇒ 静音 / 纯气声）。必须有非零样本才算真的组装出来了。
        if let Some((pitch, _, _)) = assemble_pitch_orig_from_cache(timeline, root) {
            if pitch.iter().any(|value| *value > 0.0) {
                rebuilt.pitch_orig = pitch;
                rebuilt.content_key = Some(build_root_pitch_key(timeline, root));
                rebuilt.pitch_key = rebuilt.content_key.clone();
                rebuilt.ready = true;
            }
        }
        let (dyn_curve, _) =
            hifishifter_kernel::pitch_analysis::dyn_analysis::assemble_dyn_orig_from_cache(
                timeline, root,
            );
        if dyn_curve.iter().any(|value| *value > 0.0) {
            rebuilt.dyn_orig = dyn_curve;
            rebuilt.dyn_key = Some(build_root_pitch_key(timeline, root));
            rebuilt.ready = true;
        }
        if !rebuilt.ready {
            // 内容缓存还没建立：保留上一份（**必须是真的就绪的那一份**）并标记
            // pending，由分析完成补齐。未就绪的 stored 不能回退给它 —— 它的
            // pitch_orig 可能只是"占位数组"，拿它合成会把占位值当成原始曲线。
            if let Some(mut stale) = stored {
                stale.analysis_pending = true;
                return stale;
            }
            return rebuilt;
        }
        rebuilt.revision = 1;
        self.publish_baseline(root, rebuilt.clone());
        rebuilt
    }

    /// 这一根是否已经有一份**当前几何**的基线。
    pub(crate) fn baseline_is_ready(&self, root: &str) -> bool {
        self.baseline
            .lock_or_recover()
            .get(root)
            .is_some_and(|baseline| baseline.is_ready())
    }

    /// 宿主清单是否可用（呈现层据此决定"缺席"能不能当"删除"）。
    pub(crate) fn inventory_is_complete(&self) -> bool {
        !self.inventory_incomplete.load(Ordering::Acquire)
    }

    /// 一条"带宿主身份、但本轮清单里没有"的 clip 要不要继续留在画布上。
    ///
    /// 【判据是"见过没有"，不是"连续几轮没见过"】
    /// * **见过**（上一轮还在清单里）⇒ 现在没了就是**删除**，立即撤下；
    /// * **从没见过** ⇒ 这是"清单还没枚举到它"（身份刚升级、枚举滞后），保留；
    /// * 清单本轮**不完整** ⇒ 一律保留（枚举失败不能当删除证据）。
    ///
    /// 这条判据同时满足两个相反的真实需求：升级滞后必须留下（否则"原轨 clip
    /// 全消失"），删除必须立刻消失（否则"幽灵 clip"）。权威一律不动 —— 撤销删除
    /// 之后 clip 会带着曲线原样回来。
    pub(crate) fn orphan_within_grace(&self, clip_id: &str, seen: bool, complete: bool) -> bool {
        /// "从未见过"的哨兵：与"见过（0）"区分开。
        const NEVER_SEEN: u32 = u32::MAX;
        let mut streaks = self.orphan_streak.lock_or_recover();
        if seen {
            streaks.insert(clip_id.to_owned(), 0);
            return true;
        }
        if !complete {
            return true;
        }
        match streaks.get(clip_id).copied() {
            Some(0) => false,
            Some(_) => true,
            None => {
                streaks.insert(clip_id.to_owned(), NEVER_SEEN);
                true
            }
        }
    }

    /// 从图集重新派生"视图"这一层（`timeline.params_by_root_track`）。
    pub(crate) fn resync_parameter_view_locked(
        &self,
        edits: &mut crate::state_channel::EditState,
        timeline: &mut hifishifter_kernel::state::TimelineState,
    ) -> Result<(), String> {
        if edits.needs_rebind {
            return Ok(());
        }
        self.compose_root_view_locked(edits, timeline)
    }

    /// 短事务同时冻结授权、参数、PCM和三个版本，分析副本的文件IO由actor随后执行。
    pub(crate) fn workspace_snapshot(
        &self,
    ) -> Result<(WorkspaceSnapshot, u64, Arc<WorkspaceView>), String> {
        let _startup_span = crate::startup::Span::new("document.snapshot", self.id, 0);
        let _transaction = self.transaction.lock_or_recover();
        // 【T-D7：恢复失败不能连坐显示】这里以前用 `?` 把恢复的**致命**错误传播出去，
        // 于是一份坏存档会让整份工作区快照失败 ⇒ 界面停在空时间线（用户看到的不是
        // "参数没读回来"，而是"什么都没有"）。恢复的载荷与显示的载荷是两件事：
        // 前者失败只该改读数，后者必须照常产出。
        // 【触发点④：兜底心跳】编辑器每 33–400 ms 读一次工作区。合入失败的唯一成因是
        // "宿主图还没长成"，而那一刻一定会过去 —— 让这条既有心跳把它带过去，比另开
        // 一个定时器简单，也不会在宿主静止时白跑（内部有 1 s 退避）。
        if let Err(error) = self.merge_archive_restore_locked() {
            log::warn!("[ara] archive merge on snapshot failed: {error}");
        }
        for owner in self.renderer_owners() {
            if let Err(error) = owner.merge_pending_restore(self) {
                log::warn!("[ara] instance state unresolved: {error}");
            }
        }
        // 【为什么先取 `edits` 再取时间线】文档声明的锁序是
        // `transaction → timeline → edits → host facts`（见 `DocumentSession` 的说明），
        // 而 `workspace_timeline_locked` 内部要取 `clip_ids` / `ui_tracks`（host facts）。
        // 先取时间线就是 `host facts → edits`，与 `accept_workspace_edits_with_groups`
        // 的 `edits → host facts`（它持 `edits` 时调 `parameter_identities_locked`）
        // 构成 ABBA —— 两条路径分别跑在编辑器 actor 与 ARA 模型线程上，是真死锁候选。
        // `workspace_timeline_locked` 本身不碰 `edits`，所以把顺序倒过来是安全的。
        let mut edits = self.edits.lock_or_recover();
        let mut timeline = self.workspace_timeline_locked()?;
        edits.reconcile(&self.track_bindings.lock_or_recover())?;
        // 【为什么这里**不**重投影视图】`workspace_snapshot` 是**读**路径：它随时可能被
        // 编辑器 actor 的后台刷新调用，而一次读不能改变"待提交编辑"所依赖的指纹
        // （那会让并发的 apply 判成 workspace changed）。视图的重建放在两个真正的
        // "几何变了"的写点上：`remap_and_log` 与 `refresh_ui_geometry`。
        edits.apply(&mut timeline);
        // 显示是 `(authority, baseline, geometry)` 的纯函数；读路径只物化派生缓存。
        self.materialize_views_locked(&mut edits, &mut timeline);
        // 【为什么先只取一份 id 集合、立刻释放锁】`source_failures` 与 `edit_sources`
        // 此前从不同时持有；`media_in_flight_items` 的锁序是
        // `regions → region_items → source_failures`。在这里引入 `edit_sources →
        // source_failures` 会给锁序图加一条新边。id 集合很小（每条 clip 一个），
        // 先取出来再释放，既省锁也避免新的 ABBA 候选。

        // 【T-C2：不再提前清源】
        //
        // 旧实现在这里把"这一轮不在 `edit_sources` 里"的 take 源**全部清掉**，因为
        // 下游 `materialize_with_byte_limit` 见到一个未知 id 就会让整份快照失败。
        // 代价是"这一轮有没有 PCM"直接决定"这一轮有没有身份"：宿主一收回样本访问
        // （ARA 默认关闭、由宿主开关，停播/切后台都会发生），所有 clip 一起变占位，
        // 而 `last_known` 读的正是刚被清空的自己 —— 永远回不来。
        //
        // 现在物化侧**逐条降级**：缺 PCM 的那几条只在这份快照里没有源，其余照常；
        // 身份留在文档侧（`media_facts` / ARA 映射），下一次读到采样就恢复。
        let available = self.edit_sources.lock_or_recover();
        let ids = timeline
            .clips
            .iter()
            .flat_map(|clip| {
                std::iter::once(&clip.source_path)
                    .chain(clip.takes.iter().map(|take| &take.source_path))
            })
            .filter_map(Option::as_ref)
            .collect::<std::collections::BTreeSet<_>>();
        // 到这里 `ids` 只含已授权的源；仍用 `filter_map` 兜住"授权表在两步之间变化"
        // 的竞态 —— 那种情况下也只是少一条 PCM，不该赔上整份快照。
        let sources = ids
            .into_iter()
            .filter_map(|id| available.get(id).map(|pcm| (id.clone(), pcm.clone())))
            .collect::<Vec<_>>();
        let projection = self.workspace_projection_locked(&edits)?;
        self.project_ui_fades_locked(&mut timeline, &edits.fades);
        self.project_private_group_view(&mut timeline, &edits)?;
        // 【G1】这是编辑器**真正会看到**的那份参数视图 —— 连同它的指纹一起交出去。
        // 冲突重基的唯一基线就是它（见 `accept_workspace_edits_with_groups`）；
        // 它随快照一起返回，而不是按指纹存进某张表里等提交时查回来。
        // 读路径不重投影，但仍把与当前数组同代的根投影交出去；adopt 只做
        // 按帧合并，不会用一份默认值覆盖整根。
        // 快照是读路径，不负责重投影；param_roots 只在 refresh_ui_geometry 的
        // workspace_view 里生成，避免读一次工作区也去锁 host facts。
        let view = Arc::new(WorkspaceView {
            projection,
            params: Arc::new(timeline.params_by_root_track.clone()),
            geometry_revision: self.ui_geometry_revision.load(Ordering::Acquire),
        });
        Ok((
            WorkspaceSnapshot {
                timeline,
                sources,
                revision: edits.revision,
                model_revision: self.revision.load(Ordering::Acquire),
                audio_revision: self.editor_audio_revision.load(Ordering::Acquire),
            },
            self.scope_revision.load(Ordering::Acquire),
            view,
        ))
    }
    /// 原分析副本可补媒体元信息，除此之外clip、take及轨道结构全部必须与宿主一致。
    fn validate_workspace_geometry(
        host: &hifishifter_kernel::state::TimelineState,
        client: &hifishifter_kernel::state::TimelineState,
    ) -> Result<(), String> {
        let normalize=|timeline:&hifishifter_kernel::state::TimelineState|->Result<serde_json::Value,String> {
            let mut timeline=timeline.clone();for clip in &mut timeline.clips {clip.normalize_takes();}
            let mut value=serde_json::to_value(timeline).map_err(|e|e.to_string())?;
            let metadata=["source_path_relative","duration_sec","duration_frames","source_sample_rate","source_channels",
                "source_file_fingerprint","source_file_mtime","source_file_size","waveform_preview","pitch_range"];
            if let Some(clips)=value["clips"].as_array_mut() {for clip in clips {
                for key in metadata {clip.as_object_mut().unwrap().remove(key);}
                if let Some(takes)=clip["takes"].as_array_mut() {for take in takes {for key in metadata {take.as_object_mut().unwrap().remove(key);}}}
            }}
            if let Some(tracks)=value["tracks"].as_array_mut() {for track in tracks {for key in ["volume","muted","solo","compose_enabled","pitch_analysis_algo"] {track.as_object_mut().unwrap().remove(key);}}}
            Ok(serde_json::json!({"tracks":value["tracks"],"clips":value["clips"],"project_sec":value["project_sec"]}))
        };
        if normalize(host)? != normalize(client)? {
            // 这是**冲突**而不是普通错误：客户端手里的 clip/轨布局已经不匹配宿主了。
            // 前缀让前端按既有分类走"刷新后重试"，而不是把它当成一次内部失败。
            return Err(format!(
                "{} workspace geometry is read-only",
                crate::editor::errors::CONFLICT_PREFIX
            ));
        }
        Ok(())
    }
    /// 提交载荷的**身份**校验：只问"这是不是同一批 clip / 轨 / 素材"。
    ///
    /// 【为什么不是整份时间线相等】位置、长度、源窗、淡变、增益都是宿主拥有的字段；
    /// 用户画线期间宿主完全可能移动 / 拉伸 item。把这些字段纳入判据，等于把一次正常
    /// 的并发几何变化判成冲突（真机 15 分钟里 11 次 `Conflict: host geometry changed`）。
    /// 身份不同才是真的冲突：那说明载荷来自另一份工程状态。
    fn validate_workspace_identity(
        host: &hifishifter_kernel::state::TimelineState,
        client: &hifishifter_kernel::state::TimelineState,
    ) -> Result<(), String> {
        fn conflict(what: &str) -> Result<(), String> {
            Err(format!(
                "{} workspace {what} is read-only",
                crate::editor::errors::CONFLICT_PREFIX
            ))
        }
        let tracks = |timeline: &hifishifter_kernel::state::TimelineState| {
            timeline
                .tracks
                .iter()
                .map(|track| track.id.clone())
                .collect::<std::collections::BTreeSet<_>>()
        };
        if tracks(host) != tracks(client) {
            return conflict("track set");
        }
        let clips = |timeline: &hifishifter_kernel::state::TimelineState| {
            timeline
                .clips
                .iter()
                .map(|clip| clip.id.clone())
                .collect::<std::collections::BTreeSet<_>>()
        };
        if clips(host) != clips(client) {
            return conflict("clip set");
        }
        for clip in &client.clips {
            let Some(original) = host.clips.iter().find(|known| known.id == clip.id) else {
                return conflict("clip set");
            };
            let same_takes = original.takes.len() == clip.takes.len()
                && original
                    .takes
                    .iter()
                    .zip(clip.takes.iter())
                    .all(|(a, b)| a.source_path == b.source_path);
            if original.track_id != clip.track_id
                || original.active_take_id != clip.active_take_id
                || !same_takes
            {
                return conflict("clip identity");
            }
        }
        Ok(())
    }
    /// 全工作区在单事务校验scope/模型/曲线并接受；未知范围或几何不静默忽略。
    // 整批接受入口保留，供不分组调用方使用。
    #[cfg(test)]
    pub(crate) fn accept_workspace_edits(
        &self,
        base_edit: u64,
        base_model: u64,
        client: &hifishifter_kernel::state::TimelineState,
        baseline: &WorkspaceView,
    ) -> Result<(u64, u64, Arc<WorkspaceView>), String> {
        self.accept_workspace_edits_with_groups(base_edit, base_model, client, baseline, None)
    }
    /// 组根处理设置与参数delta同一事务提交，避免空父轨设置在音频准备时退回默认值。
    pub(crate) fn accept_workspace_edits_with_groups(
        &self,
        // 【为什么这个参数没有被读】它原本用来做"编辑代次是否仍然一致"的判据，但那个
        // 检查从未实现 —— 真正挡住陈旧写回的是下面的 `base_model`（模型代次）与
        // `previous`（编辑器读到的那份投影的指纹）。保留形参是为了不动六个调用点
        // （其中四个在测试里），同时用下划线把"它确实没被读"写在签名上，而不是让它
        // 悄悄看起来像一个生效的守卫。
        _base_edit: u64,
        base_model: u64,
        client: &hifishifter_kernel::state::TimelineState,
        baseline: &WorkspaceView,
        groups: Option<crate::editor::private_groups::TrackGroups>,
    ) -> Result<(u64, u64, Arc<WorkspaceView>), String> {
        let _transaction = self.transaction.lock_or_recover();
        let model = self.revision.load(Ordering::Acquire);
        if model != base_model {
            return Err(format!(
                "{} {}",
                crate::editor::errors::CONFLICT_PREFIX,
                crate::editor::errors::MODEL_CHANGED
            ));
        }
        // 【为什么不再按几何代次拒绝】差分域是**编辑器看到的那份几何**：`capture` 的
        // 第一个参数就是 `client`，它带着用户画线时的 clip 窗口。宿主在这期间移动 /
        // 拉伸 item 只影响"这一笔在工程上看起来在哪"，那是**投影**的职责，下一次
        // `compose_root_views` 用新几何重算即可。旧实现在这里硬拒，于是用户每在宿主
        // 里动一下 item、回到插件画一笔就撞一次 `Conflict: host geometry changed`
        // （真机 15 分钟里 11 次）。可重基的并发几何变化不是冲突。
        let host = self.workspace_timeline_locked()?;
        // 【为什么这里用**检查版**取锁】这是"读改写并影响宿主"的路径：`candidate` 由
        // `edits` 推导，最后写回 `*edits`。若上一次持锁者在 `merge`/`capture_changes`
        // 中途 panic，`edits` 可能停在半写状态 —— 拿它推导再写回，会把一次局部 panic
        // 升级成工程级的编辑损坏（而它随后会经 VST3 `getState` 落进宿主工程）。
        // 宁可放弃这一次编辑并留痕，也不要拿半写状态去改宿主。
        let (mut edits, poisoned) = self.edits.lock_or_recover_checked();
        if poisoned {
            log::error!(
                "[ara] workspace edit refused: the edit authority was left half-written by an                  earlier panic; this edit is dropped instead of being derived from it"
            );
            return Err(
                "Conflict: edit authority is in an unknown state after a panic; the edit was                  not applied"
                    .into(),
            );
        }
        // 【G1：提交闸从"拒绝"改成"重基"】
        //
        // 旧实现只要"宿主在我编辑期间动过"就整条拒绝。而那条冲突是**可重基**的：
        // 用户的意图是"这条曲线的这几个帧改成这些值"，与"宿主在这期间移动了 item"
        // 在语义上并不冲突 —— 只要把 delta 落到**编辑器看到的那份基线**上。
        //
        // 【为什么基线必须是"编辑器读到的"而不是"现在算出来的"】`capture_changes`
        // 只记录 `client` 与 `previous_view` 的**差异**。拿现在算出来的投影当基线时，
        // 宿主移动造成的几何差异会被当成"用户改了参数"写回源曲线（抹平）；而拿
        // 编辑器读到的那一份当基线，用户没碰的片段两侧**逐字相同** ⇒ 一条都不写回，
        // 权威（源域）原样保留，几何变化由投影吸收。后者才是正确的那一半。
        let current_projection = self.workspace_projection_locked(&edits)?;
        // 【G1：判定"要不要重基"只需要一次比较；基线本身恒等于"编辑器读到的那份"】
        //
        // 旧实现把基线做成一张按指纹索引的表，于是"要不要重基"与"基线是谁"共用一次
        // 可能落空的查表 —— 而落空时只能猜或拒绝，两个都错（猜 = 把曲线写到错误的帧
        // 上；拒绝 = 用户这次编辑整条丢弃）。现在基线随视图一起交付，这里只剩读数：
        // 投影动过没有，只决定日志里的 rebase=yes/no。
        let rebased = current_projection != baseline.projection;
        // 几何代次只作**读数**：它变了不代表冲突（见上），但值得留在日志里 ——
        // "宿主在画线期间动了几何"是排查参数线错位时最常需要的现场。
        let geometry_moved =
            baseline.geometry_revision != self.ui_geometry_revision.load(Ordering::Acquire);
        // 【为什么只校验身份】位置 / 长度 / 源窗 / 淡变都是宿主拥有的字段，客户端
        // 原样回传即可；把它们纳入"逐字相等"会让每一次宿主几何刷新都变成提交失败。
        // 真正必须守住的是**身份**：不许借这份载荷新增 / 删除 clip 或轨，也不许把
        // 一条 clip 换成另一段素材。
        Self::validate_workspace_identity(&host, client)?;
        let mut candidate = edits.merge(&host, client, edits.revision)?;
        candidate.reconcile(&self.track_bindings.lock_or_recover())?;
        // 【不再有"per-clip 身份"这一层】它唯一的用途是把曲线按 clip 归属索引
        // （`RegionIdentity`）。权威按根轨索引，曲线与 clip 的对应关系由**工程时间**
        // 决定 —— 不需要一张身份表。`parameter_identities_locked` 随图集一起删除。

        // 【U-4：基线**只能**来自"编辑器真正读到的那一份"】
        //
        // `capture_changes` 只记录 `client` 与这一份的**差异**。基线一旦不是编辑器
        // 看到的那份，判据就变成"当前视图与某个别的视图不同" —— 而"不同"里既有用户的
        // 编辑，也有宿主移动造成的几何差异、以及投影范围的变化。把它当成用户意图写回
        // 源曲线，就是用户看到的"编辑一下 clip，参数线被重置"。
        // 【客户端 = 编辑器这一轮交上来的数组】`baseline.params` 是**上一轮**的视图，
        // 拿它当客户端等于"自己和自己比" —— 差分恒空，用户的笔画永远进不了权威。

        // 【曲线只由显式写入产生】`set_param_frames` 是唯一写入口（带 `present` 位图），
        // 这里只合并轨道设置 / 分组等非曲线字段。绝不再把客户端整条显示数组做差分 ——
        // 那会把几何差异与未编辑原值一起固化成编辑。
        if rebased {
            log::info!(
                "[ara] workspace rebase: baseline={} current={} roots={} geometry_moved={geometry_moved}",
                &baseline.projection[..baseline.projection.len().min(8)],
                &current_projection[..current_projection.len().min(8)],
                baseline.params.len()
            );
        }
        // 【显示层唯一写者】权威 + 基线 + 当前几何重新合成。客户端数组**不是**输入 ——
        // 它只是"用户此刻看到的东西"，与权威的差由 `set_param_frames` 显式提交。
        {
            let mut composed = std::collections::BTreeMap::new();
            let roots = host
                .tracks
                .iter()
                .filter(|track| track.parent_id.is_none())
                .map(|track| track.id.clone())
                .collect::<Vec<_>>();
            for root in roots {
                let view = self.compose_param_view_with(&host, &candidate, &root);
                composed.insert(root, view.params);
            }
            if !composed.is_empty() {
                candidate.params = composed;
            }
        }
        if let Some(groups) = groups {
            groups.validate()?;
            candidate.groups = groups;
        }
        let projection = self.workspace_projection_locked(&candidate)?;
        // 【下一份基线就在这一步产生】`merge_params` 逐字复制了 client 的 params
        // （仅限宿主轨道），所以 candidate.params 就是编辑器此刻显示的那份数组；
        // 指纹与它同代（同一份 candidate 算出）。调用方必须把它存回 `Loaded` ——
        // 类型上它不能"忘记"（返回值的唯一用途就是那个）。
        let view = Arc::new(WorkspaceView {
            projection,
            params: Arc::new(candidate.params.clone()),
            geometry_revision: self.ui_geometry_revision.load(Ordering::Acquire),
        });
        *edits = candidate;
        // 用户的参数编辑落盘了 ⇒ 告诉宿主"这些 modification 的持久状态变了"。
        // 宿主只保存它知道变过的部分；不通知就会在它做增量保存时丢掉这次编辑。
        self.notify_parameters_changed();
        Ok((edits.revision, model, view))
    }
    /// 合成只在短事务外计算；最终再次核对文档、scope、assignment与编辑代次。
    // 无进度回调的合成入口保留，供简单调用方使用。
    #[cfg(test)]
    pub(crate) fn apply_workspace_edits(
        &self,
        cancel: Arc<AtomicBool>,
        current: impl Fn() -> bool,
    ) -> Result<(), String> {
        self.apply_workspace_edits_with_progress(cancel, current, None)
            .map(|_| ())
    }
    /// 这一轮该渲整段还是只渲播放头附近的一段。
    ///
    /// 【为什么需要】整段物化的内存与跨度成正比。跨度装得下就整段渲（与今天逐字节
    /// 相同）；装不下就只渲播放头附近 —— 跨度于是只决定"渲几段"，不再决定"能不能渲"。
    /// 窗口之外的帧在快照里不声明覆盖，播放侧按既有规则退回宿主原声。
    ///
    /// 【为什么锚在播放头】用户听到的是当前位置；把有限的内存花在他正在听的地方，
    /// 比"从 0 开始渲一段然后永远听不到后面"有用得多。
    fn plan_render_window(
        &self,
        input: &super::input::RenderInput,
        owner: &ExtensionOwner,
    ) -> Option<(f64, f64)> {
        let timeline = input.timeline.as_ref()?;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for clip in &timeline.clips {
            if !(clip.start_sec.is_finite() && clip.length_sec.is_finite()) {
                continue;
            }
            lo = lo.min(clip.start_sec);
            hi = hi.max(clip.start_sec + clip.length_sec);
        }
        if !(lo.is_finite() && hi.is_finite()) {
            return None;
        }
        let lo = lo.max(0.0);
        if hi <= lo {
            return None;
        }
        let budget = super::budget::global_budget();
        let available = budget.limit().saturating_sub(budget.used());
        if input.estimate_bytes(hi - lo) <= available {
            // 整段装得下：保持今天的行为，逐字节不变。
            return None;
        }
        let per_sec = input.estimate_bytes(1.0).max(1);
        let window_sec = (available / per_sec).clamp(8, 120) as f64;
        let rate = owner.setup_sample_rate().max(1.0);
        let anchor = owner
            .snapshots
            .first()
            .and_then(|publisher| publisher.last_sample_seconds(rate))
            .unwrap_or(lo);
        let start = (anchor - 2.0).max(lo);
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.render.window",
            format_args!(
                "[render] span={:.1}s does not fit in the PCM budget (available={available});                  rendering a {window_sec:.0}s window from {start:.1}s instead",
                hi - lo
            ),
        );
        Some((start, start + window_sec))
    }

    /// 记一轮"没有可渲染内容"；**只在刚刚跨过预算的那一轮**返回读数。
    ///
    /// 【为什么要"报一次"而不是"每轮都报"】空轮是常态：宿主拖动一条 clip 时每几百毫秒
    /// 一次。每轮都报会把日志刷满、也会让状态栏一直红着；一次都不报则"真的空了"没有声音。
    fn note_empty_round(&self) -> Option<EmptyStreak> {
        let rounds = self.empty_rounds.fetch_add(1, Ordering::AcqRel) + 1;
        let now = wall_millis();
        let since =
            match self
                .empty_since_ms
                .compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => now,
                Err(known) => known,
            };
        let elapsed_ms = now.saturating_sub(since);
        if rounds < EMPTY_ROUNDS_MAX && elapsed_ms < EMPTY_BUDGET_MS {
            return None;
        }
        if self.empty_reported.swap(true, Ordering::AcqRel) {
            return None;
        }
        Some(EmptyStreak { rounds, elapsed_ms })
    }

    /// 发布成功：三个量一起归零（下一次空轮从零开始计）。
    fn clear_empty_rounds(&self) {
        self.empty_rounds.store(0, Ordering::Release);
        self.empty_since_ms.store(0, Ordering::Release);
        self.empty_reported.store(false, Ordering::Release);
    }

    pub(crate) fn apply_workspace_edits_with_progress(
        &self,
        cancel: Arc<AtomicBool>,
        current: impl Fn() -> bool,
        progress: Option<hifishifter_kernel::mixdown::ProgressCallback>,
    ) -> Result<ApplyOutcome, String> {
        let (ticket, inputs) = {
            let _transaction = self.transaction.lock_or_recover();
            if !self.is_alive() || !self.ready.load(Ordering::Acquire) {
                return Err(format!(
                    "{} model changed during automatic apply",
                    crate::editor::errors::CONFLICT_PREFIX
                ));
            }
            // 【入口自取】这一刻之后发生的变化才叫"并发"；之前的变化（例如用户刚画完
            // 那一笔）是这次渲染的**输入**，不是冲突。
            let ticket = self.render_ticket_locked();
            let edits = self.edits.lock_or_recover().clone();
            let inputs = self
                .renderer_owners()
                .into_iter()
                .filter(|owner| owner.renders_playback())
                .map(|owner| {
                    let (keys, input) = owner.ara_render_input(self, &edits, true)?;
                    Ok((owner, keys, input))
                })
                .collect::<Result<Vec<_>, String>>()?;
            (ticket, inputs)
        };
        if !current() {
            return Err("automatic apply superseded".into());
        }
        let mut prepared = Vec::new();
        // 这一轮里有没有任何 owner 明确回答"没有可渲染的 clip"。
        let mut empty = false;
        // 【一个作业要渲染多个 owner 时，进度必须**分段聚合**】
        //
        // 每个 owner 的 `render_with_progress` 都从 0 走到 1。此前它们共用一个回调，
        // 而回调侧有一条"不得回退"的判据（那是对的：回退对用户是纯粹的故障信号）——
        // 于是第二个 owner 的 0 被挡掉、随后每个中间值也被挡掉，显示**一直停在 100%**
        // 直到作业结束。把第 i 个 owner 的 0..1 折算到 `[i/n, (i+1)/n)` 之后，整份作业
        // 天然单调，且最后一个 owner 的 1.0 恰好等于整份作业的 1.0。
        let owners = inputs.len().max(1);
        for (index, (owner, keys, mut input)) in inputs.into_iter().enumerate() {
            for publisher in &owner.snapshots {
                publisher.collect_retired();
            }
            let window = self.plan_render_window(&input, owner.as_ref());
            input.window = window;
            let slice = progress.clone().map(|callback| {
                hifishifter_kernel::mixdown::ProgressCallback::new(move |value: f64| {
                    let value = value.clamp(0.0, 1.0);
                    callback.call((index as f64 + value) / owners as f64);
                })
            });
            match input.render_with_progress(cancel.clone(), slice)? {
                crate::render::input::RenderOutcome::Snapshots(snapshots) => {
                    prepared.push((owner, keys, snapshots))
                }
                crate::render::input::RenderOutcome::Nothing => {
                    // 【为什么不 push 一份空快照】空快照是一个**真实的发布动作**：播放侧
                    // 随即逐样本读 0，用户听到一次"处理消失"。见 `RenderOutcome`。
                    empty = true;
                }
            }
        }
        let _transaction = self.transaction.lock_or_recover();
        // 复核只看"这份已算好的东西还能不能用"：活着、仍就绪，且票据没有变。
        if !self.is_alive() || !self.ready.load(Ordering::Acquire) {
            return Err(format!(
                "{} model changed during automatic apply",
                crate::editor::errors::CONFLICT_PREFIX
            ));
        }
        if !current() || self.render_ticket_locked() != ticket {
            return Err("automatic apply superseded".into());
        }
        for (owner, keys, snapshots) in &prepared {
            if owner.is_closed() || owner.assigned_regions().map_err(|e| e.to_string())? != *keys {
                return Err("automatic apply superseded".into());
            }
            if !owner
                .snapshots
                .iter()
                .zip(snapshots)
                .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
            {
                return Err("retired snapshot budget exhausted; reopen instance".into());
            }
        }
        // 【空轮按"这一轮"记，不按"某条 owner"】一次作业要么发布了内容，要么没有；
        // 逐 owner 计数会让"两个 renderer 里只有一个空"变成两套时间线。
        let report = if empty { self.note_empty_round() } else { None };
        let published = !prepared.is_empty();
        for (owner, keys, snapshots) in prepared {
            for (publisher, snapshot) in owner.snapshots.iter().zip(snapshots) {
                publisher
                    .publish_shared(snapshot)
                    .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
            }
            owner.record_prepared(ticket.model, ticket.edit, ticket.epoch, ticket.scope, keys);
        }
        if published {
            // 只要这一轮真的发布了东西，连续为空的账就一笔勾销。
            self.clear_empty_rounds();
        }
        if empty {
            // 【不写 `applied`】下一轮 150 ms 后还会再来一次：宿主把模型补回来时，
            // 这一次的空轮就是"没人接"的那一半。
            return Ok(ApplyOutcome::Nothing { report });
        }
        Ok(ApplyOutcome::Published)
    }

    /// 宿主专属API调用前核对真实文档仍存活，销毁后的view不能沿旧project指针查询。
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
    /// 会话与模型同寿，controller 地址在工厂 allocation 完成后登记。
    pub fn new(id: DocumentId) -> Arc<Self> {
        Arc::new(Self {
            id,
            alive: AtomicBool::new(true),
            // `resynced_geometry` 与 `ui_geometry_revision` 同为 0 起步是**有意**的：
            // 让首轮 `workspace_view()`（读路径）不做重投影 —— "读一次工作区"不得
            // 改变待提交编辑依赖的指纹，否则并发的 automatic apply 会判成
            // workspace changed（该字段注释与 bound_tests 钉死这一契约）。
            // 冷恢复后首轮跳过的窗口由 restore 路径自己的 inventory 推进兜住。
            ..Default::default()
        })
    }

    /// 宿主几何刷新后调用：锁定开启时变换 run，关闭时只记账。
    ///
    /// 【为什么是 partial 帧】这里的 `workspace_timeline_locked` 已经按本实例的
    /// assignment 过滤过；把它当完整帧会让"这一轮没分配到的 clip"从日志里消失。
    /// 模型收口（`remap_and_log`）那条路径才提供完整帧。
    pub(crate) fn reconcile_parameter_geometry(
        &self,
    ) -> Result<hifishifter_kernel::editor::param_geometry::TransformReport, String> {
        let _transaction = self.transaction.lock_or_recover();
        let host = self.workspace_timeline_locked()?;
        let frame = self.geometry_frame(&host, false);
        let mut edits = self.edits.lock_or_recover();
        self.reconcile_parameter_geometry_locked(&frame, &mut edits)
    }

    /// 这个区间里有没有任何 run（"这次重置会不会真的改东西"）。
    ///
    /// 只取 `edits`，不取时间线 —— 不给锁序图加新边。
    pub(crate) fn authority_has_runs_in(
        &self,
        track_id: &str,
        param: &str,
        start_frame: u32,
        frame_count: u32,
    ) -> bool {
        let Ok(host) = self.workspace_timeline_locked() else {
            return false;
        };
        let Some(root) = host.resolve_root_track_id(track_id) else {
            return false;
        };
        let edits = self.edits.lock_or_recover();
        edits
            .authority
            .has_runs_in(&root, param, start_frame as usize, frame_count as usize)
    }

    /// 所有 clip 的渲染局部参数：从根视图切窗，不按源窗口重建曲线。
    pub(crate) fn compose_clip_params_locked(
        &self,
        timeline: &hifishifter_kernel::state::TimelineState,
        edits: &crate::state_channel::EditState,
    ) -> Result<
        std::collections::BTreeMap<String, hifishifter_kernel::state::TrackParamsState>,
        String,
    > {
        let mut views = std::collections::BTreeMap::new();
        let roots = timeline
            .clips
            .iter()
            .filter_map(|clip| timeline.resolve_root_track_id(&clip.track_id))
            .collect::<std::collections::BTreeSet<_>>();
        for root in roots {
            views.insert(
                root.clone(),
                self.compose_param_view_with(timeline, edits, &root),
            );
        }
        Ok(timeline
            .clips
            .iter()
            .filter_map(|clip| {
                let root = timeline.resolve_root_track_id(&clip.track_id)?;
                let view = views.get(&root)?;
                Some((
                    clip.id.clone(),
                    hifishifter_kernel::editor::param_authority::compose_clip_params(
                        view,
                        clip,
                        view.params.frame_period_ms,
                    ),
                ))
            })
            .collect())
    }

    /// 取一份权威快照（undo 步骤用）。
    pub(crate) fn authority_snapshot(
        &self,
    ) -> hifishifter_kernel::editor::param_authority::ParamAuthority {
        self.edits.lock_or_recover().authority.clone()
    }

    /// 用历史快照整体替换权威，然后失效派生视图。
    pub(crate) fn replace_authority_with_edits(
        &self,
        mut authority: hifishifter_kernel::editor::param_authority::ParamAuthority,
        block_edits: Option<
            std::collections::BTreeMap<String, hifishifter_kernel::param_blocks::ParamBlockEditSet>,
        >,
    ) -> Result<bool, String> {
        let _transaction = self.transaction.lock_or_recover();
        let mut host = self.workspace_timeline_locked()?;
        let mut edits = self.edits.lock_or_recover();
        authority.sanitize();
        let changed = authority.content_hash() != edits.authority.content_hash();
        if changed {
            authority.revision = edits.authority.revision.saturating_add(1);
            edits.authority = authority;
            edits.revision = edits.revision.saturating_add(1);
            self.render_epoch.fetch_add(1, Ordering::AcqRel);
        }
        if let Some(mut block_edits) = block_edits {
            for (root, current) in &edits.param_block_edits {
                let target = block_edits.entry(root.clone()).or_insert_with(|| {
                    hifishifter_kernel::param_blocks::ParamBlockEditSet {
                        revision: current.revision,
                        ..Default::default()
                    }
                });
                target.revision = current.revision.saturating_add(1);
            }
            for value in block_edits.values() {
                value.validate()?;
            }
            edits.param_block_edits = block_edits;
        }
        self.materialize_views_locked(&mut edits, &mut host);
        drop(edits);
        self.archive_cache.lock_or_recover().take();
        if changed {
            self.notify_parameters_changed();
        }
        Ok(changed)
    }

    /// 渲染守卫票据（调用方必须已持 `transaction`）。
    ///
    /// 只碰 `edits` 与四个原子量，**不取 timeline** —— 因此不会给锁序图
    /// （`transaction → edits → timeline → clip_ids`）加新边。
    pub(crate) fn render_ticket_locked(&self) -> RenderTicket {
        RenderTicket {
            edit: self.edits.lock_or_recover().revision,
            model: self.revision.load(Ordering::Acquire),
            scope: self.scope_revision.load(Ordering::Acquire),
            epoch: self.render_epoch.load(Ordering::Acquire),
        }
    }

    /// 当前参数权威 revision（前端水位；只读）。
    pub(crate) fn authority_revision(&self) -> u64 {
        self.edits.lock_or_recover().authority.revision
    }

    /// 把"权威 + 基线"合成进时间线（只读 `edits`，不写派生缓存）。
    ///
    /// 【为什么需要它】`materialize_views_locked` 还要把结果写回 `edits.params`，而
    /// 分组视图 / 选中源这几条路径只借到 `&EditState`。语义与它完全相同：每个根轨
    /// 一次 `compose_param_view_with`，整根替换。**这是唯一允许的第二处合成入口**，
    /// 且它与 `materialize_views_locked` 调的是同一个纯函数 —— 不存在两套算法。
    pub(crate) fn materialize_into(
        &self,
        timeline: &mut hifishifter_kernel::state::TimelineState,
        edits: &crate::state_channel::EditState,
    ) {
        let roots = timeline
            .tracks
            .iter()
            .filter(|track| track.parent_id.is_none())
            .map(|track| track.id.clone())
            .collect::<Vec<_>>();
        let composed = roots
            .into_iter()
            .map(|root| {
                let params = self.compose_param_view_with(timeline, edits, &root).params;
                (root, params)
            })
            .collect::<Vec<_>>();
        for (root, params) in composed {
            timeline.params_by_root_track.insert(root, params);
        }
    }

    /// 为一条根轨合成参数视图；调用方已持有 `edits` 时使用这个版本。
    pub(crate) fn compose_param_view_with(
        &self,
        timeline: &hifishifter_kernel::state::TimelineState,
        edits: &crate::state_channel::EditState,
        root: &str,
    ) -> hifishifter_kernel::editor::param_authority::RootParamView {
        let fp = if edits.authority.frame_period_ms.is_finite()
            && edits.authority.frame_period_ms > 0.0
        {
            edits.authority.frame_period_ms
        } else {
            timeline.frame_period_ms()
        };
        // 【基线的唯一入口】按当前布局取；布局变了就地重组装（纯数组搬运）。
        // 旧实现在这里读"上一次发布的基线"，于是移动 clip 之后仍然描述旧布局：
        // 旧位置留着原曲线、新位置 `pitch_orig = 0`（渲染器判定"无源 F0"而直通原音）。
        let mut baseline = self.baseline_for(root, timeline);
        // 分析尚未给出数组时，权威仍必须能显示：用目标帧数撑出基线，未编辑帧是默认值。
        let run_frames = edits
            .authority
            .root(root)
            .map(|authority| {
                authority
                    .runs
                    .values()
                    .flat_map(|runs| runs.iter())
                    // run 覆盖的是**格**：最后一格铺到 `span_end_sec`。按末样本算长度
                    // 会少一格，拉伸后的那一帧便无处可写（显示回落成原始值）。
                    .map(|run| (run.span_end_sec() * 1000.0 / fp).ceil() as usize)
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        // 【硬上限】`run_frames` 由存档/宿主给出的 run 位置推导；一条越界的 run
        // 就能把"需要的帧数"推到十亿级，随后 `resize` 直接申请数十 GB。
        // 与内核的"参数帧预算"同量级取 1M 帧：正常工程（约 1.4 小时 @5ms）够用，
        // 越界数据只会被截断显示，不会把进程打死。
        const MAX_VIEW_FRAMES: usize = 1_000_000;
        let needed = timeline
            .target_param_frames(fp)
            .max(run_frames)
            .max(baseline.frame_count())
            .min(MAX_VIEW_FRAMES);
        if baseline.frame_period_ms <= 0.0 || !baseline.frame_period_ms.is_finite() {
            baseline.frame_period_ms = fp;
        }
        // 【空曲线保持为空】空数组是“这一层没有基线”的明确语义，不是“一列零”。
        // 尤其 `dyn_orig`：把它补成 0 会让 -1 哨兵被解析成 0 dB 目标，整条 clip
        // 的混音增益变成 0（冷恢复后“有编辑、没声音”）。只有真实存在的曲线才按
        // 需要的帧数补零，避免越界采样；缺席的层继续缺席，由渲染器的默认分支处理。
        if !baseline.pitch_orig.is_empty() && baseline.pitch_orig.len() < needed {
            baseline.pitch_orig.resize(needed, 0.0);
        }
        if !baseline.dyn_orig.is_empty() && baseline.dyn_orig.len() < needed {
            baseline.dyn_orig.resize(needed, 0.0);
        }
        if !baseline.tension_orig.is_empty() && baseline.tension_orig.len() < needed {
            baseline.tension_orig.resize(needed, 0.0);
        }
        let mut view = hifishifter_kernel::editor::param_authority::compose_root_view(
            root,
            &edits.authority,
            &baseline,
            self.ui_geometry_revision.load(Ordering::Acquire),
        );
        view.params.param_block_edits = edits
            .param_block_edits
            .get(root)
            .cloned()
            .unwrap_or_default();
        view.params.param_block_view = hifishifter_kernel::param_blocks::compose_param_block_view(
            timeline,
            root,
            &view.params.param_block_edits,
            fp,
        )
        .ok();
        view
    }

    /// 唯一把 `authority + baseline` 写成显示数组的地方。只写派生缓存。
    pub(crate) fn materialize_views_locked(
        &self,
        edits: &mut crate::state_channel::EditState,
        timeline: &mut hifishifter_kernel::state::TimelineState,
    ) -> std::collections::BTreeMap<
        String,
        hifishifter_kernel::editor::param_authority::RootParamView,
    > {
        let roots = timeline
            .tracks
            .iter()
            .filter(|track| track.parent_id.is_none())
            .map(|track| track.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let mut views = std::collections::BTreeMap::new();
        timeline
            .params_by_root_track
            .retain(|root, _| roots.contains(root) || edits.authority.roots.contains_key(root));
        for root in roots {
            let view = self.compose_param_view_with(timeline, edits, &root);
            timeline
                .params_by_root_track
                .insert(root.clone(), view.params.clone());
            views.insert(root, view);
        }
        edits.params = timeline.params_by_root_track.clone();
        views
    }

    fn authority_param_default(
        timeline: &hifishifter_kernel::state::TimelineState,
        root: &str,
        param: &str,
    ) -> f32 {
        use hifishifter_kernel::editor::param_authority::{
            canonical_param_key, default_value_for_key, EXTRA_DYN, EXTRA_VOLUME,
        };
        let key = canonical_param_key(param);
        if key == EXTRA_DYN {
            return -1.0;
        }
        if key == EXTRA_VOLUME
            || key == hifishifter_kernel::editor::param_authority::EXTRA_BREATH_GAIN
        {
            return 1.0;
        }
        let kind = timeline
            .tracks
            .iter()
            .find(|track| track.id == root)
            .map(|track| {
                hifishifter_kernel::state::SynthPipelineKind::from_track_algo(
                    &track.pitch_analysis_algo,
                )
            })
            .unwrap_or(hifishifter_kernel::state::SynthPipelineKind::WorldVocoder);
        hifishifter_kernel::renderer::common_params::automation_curve_pad_value(
            kind,
            key.strip_prefix("extra:").unwrap_or(param),
        )
        .max(default_value_for_key(&key))
        .min(1.0)
    }

    fn normalize_authority_value(param: &str, value: f32, default: f32) -> f32 {
        use hifishifter_kernel::editor::param_authority::canonical_param_key;
        let key = canonical_param_key(param);
        if !value.is_finite() {
            return default;
        }
        match key.as_str() {
            "pitch_edit" => {
                if value == 0.0 {
                    0.0
                } else {
                    value.clamp(1.0, 127.0)
                }
            }
            "tension_edit" => value.clamp(-100.0, 100.0),
            "extra:dyn" => {
                if value < 0.0 {
                    -1.0
                } else {
                    value.clamp(0.0, 1.0)
                }
            }
            "extra:volume" | "extra:breath_gain" => value.clamp(0.0, 2.0),
            _ => value,
        }
    }

    /// 权威读：GUI 参数编辑器唯一入口。返回 orig/edit/edited 三平面。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_authority_frames(
        &self,
        track_id: &str,
        param: &str,
        start_frame: u32,
        frame_count: u32,
        stride: Option<u32>,
        with_sentinel: Option<bool>,
        binary: Option<bool>,
    ) -> Result<hifishifter_kernel::models::ParamFramesPayload, String> {
        let _transaction = self.transaction.lock_or_recover();
        let host = self.workspace_timeline_locked()?;
        let root = host
            .resolve_root_track_id(track_id)
            .ok_or_else(|| "unknown parameter track".to_owned())?;
        let edits = self.edits.lock_or_recover();
        let fp = if edits.authority.frame_period_ms.is_finite()
            && edits.authority.frame_period_ms > 0.0
        {
            edits.authority.frame_period_ms
        } else {
            host.frame_period_ms()
        };
        let view = self.compose_param_view_with(&host, &edits, &root);
        let target = host.target_param_frames(fp);
        let count = (frame_count as usize)
            .max(1)
            .min(target.saturating_add(1 << 20));
        let step = stride.unwrap_or(1).max(1) as usize;
        let key = hifishifter_kernel::editor::param_authority::canonical_param_key(param);
        let mask = view.edited_mask(param);
        let default = Self::authority_param_default(&host, &root, param);
        let mut orig = Vec::with_capacity(count);
        let mut edit = Vec::with_capacity(count);
        let mut edited = Vec::with_capacity(count);
        for i in 0..count {
            let idx = (start_frame as usize).saturating_add(i.saturating_mul(step));
            let (o, e) = match key.as_str() {
                "pitch_edit" => {
                    let o = view.params.pitch_orig.get(idx).copied().unwrap_or(0.0);
                    let e = view.params.pitch_edit.get(idx).copied().unwrap_or(o);
                    (o, if e == 0.0 && o != 0.0 { o } else { e })
                }
                "tension_edit" => (
                    view.params.tension_orig.get(idx).copied().unwrap_or(0.0),
                    view.params.tension_edit.get(idx).copied().unwrap_or(0.0),
                ),
                "extra:dyn" => {
                    // 没有 clip 覆盖的帧没有声音 ⇒ 原始值映射到 0（唯一规则见
                    // `common_params::dyn_orig_level_at`，与 App 的 `get_param_frames`
                    // 同源）。
                    let o = hifishifter_kernel::renderer::common_params::dyn_orig_level_at(
                        &view.params.dyn_orig,
                        idx,
                    );
                    let raw = view
                        .params
                        .extra_curves
                        .get("dyn")
                        .and_then(|curve| curve.get(idx))
                        .copied()
                        .unwrap_or(-1.0);
                    (o, if raw < 0.0 { o } else { raw })
                }
                _ => {
                    let name = key.strip_prefix("extra:").unwrap_or(param);
                    (
                        default,
                        view.params
                            .extra_curves
                            .get(name)
                            .and_then(|curve| curve.get(idx))
                            .copied()
                            .unwrap_or(default),
                    )
                }
            };
            orig.push(o);
            edit.push(e);
            edited.push(mask.get(idx).copied().unwrap_or(false));
        }
        let edit_sentinel = if param == "dyn" && with_sentinel.unwrap_or(false) {
            Some(edited.iter().map(|flag| !*flag).collect::<Vec<bool>>())
        } else {
            None
        };
        let mut payload = hifishifter_kernel::models::ParamFramesPayload {
            ok: true,
            root_track_id: root.clone(),
            param: param.to_owned(),
            frame_period_ms: fp,
            start_frame,
            orig,
            edit,
            edited,
            authority_revision: view.authority_revision,
            baseline_revision: view.baseline_revision,
            geometry_revision: view.geometry_revision,
            geometry_signature: Some(
                hifishifter_kernel::pitch_analysis::build_root_pitch_layout_key(&host, &root),
            ),
            binary: None,
            reference_kind: if matches!(param, "pitch" | "dyn") {
                hifishifter_kernel::models::ParamReferenceKind::SourceCurve
            } else {
                hifishifter_kernel::models::ParamReferenceKind::DefaultValue
            },
            analysis_pending: Some(!self.baseline_is_ready(&root)),
            analysis_progress: None,
            pitch_edit_user_modified: (param == "pitch")
                .then_some(view.params.pitch_edit_user_modified),
            pitch_edit_backend_available: None,
            edit_sentinel,
            dyn_orig_key: (param == "dyn")
                .then(|| {
                    host.params_by_root_track
                        .get(&root)?
                        .dyn_orig_source_key
                        .clone()
                })
                .flatten(),
        };
        if binary.unwrap_or(false) {
            let encoded = hifishifter_kernel::editor::params::encode_pfb2(
                &payload.orig,
                &payload.edit,
                &payload.edited,
            );
            payload.binary = Some(encoded);
            payload.orig.clear();
            payload.edit.clear();
            payload.edited.clear();
            // sentinel 已在构造前按显式编辑位图算好，不依赖已被清空的三平面。
        }
        Ok(payload)
    }

    /// 私有分组里 `root` 的**全部后代根轨**（子、孙……）。
    ///
    /// 【为什么写路径必须展开】分组共享一条参数线（用户手册：「轨道组共享一套参数，
    /// 子轨从根轨继承」）：GUI 只给分组根一条线，用户在它上面落笔时 `set_param_frames`
    /// 报的是**根**的 id。而渲染与存档都按**物理轨**取参数 —— `workspace_timeline_locked`
    /// 不带私有分组边（分组只存在于 `edits.groups` 的 GUID 树里）。子轨的权威不跟着写，
    /// 用户看到的就是"画了只有根轨有声音，子轨没反应"。
    ///
    /// 【为什么只对分组根生效】子轨自己的偏移参数（cents / degree / formant）写的是
    /// 子轨自己，那条路上没有后代；`root` 解析用的是宿主物理图，所以子轨永远解析成
    /// 它自己。
    ///
    /// 【为什么要已访问集】分组树是用户手拖出来的，坏存档里可能有环。这里宁可少写
    /// 一条轨，也不能让一次落笔把命令线程转死。
    fn private_group_descendants(
        &self,
        edits: &crate::state_channel::EditState,
        root: &str,
    ) -> Vec<String> {
        let aliases = self.group_aliases("");
        // 两张方向相反的表：分组树按 GUID 存，宿主轨按 id 存，展开要来回换。
        let guid_of = aliases
            .iter()
            .map(|(guid, id)| (id.as_str(), guid.as_str()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let id_of = aliases
            .iter()
            .map(|(guid, id)| (guid.as_str(), id.as_str()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let Some(&root_guid) = guid_of.get(root) else {
            return Vec::new();
        };
        let mut visited = std::collections::BTreeSet::from([root_guid]);
        let mut frontier = vec![root_guid];
        let mut found = Vec::new();
        while let Some(parent) = frontier.pop() {
            for (guid, node) in &edits.groups.nodes {
                if node.parent.as_deref() != Some(parent) || !visited.insert(guid.as_str()) {
                    continue;
                }
                if let Some(id) = id_of.get(guid.as_str()) {
                    found.push((*id).to_owned());
                }
                frontier.push(guid.as_str());
            }
        }
        found
    }

    /// 权威写：块编辑集替换 + 显式曲线补丁。绝不读取/写回整条显示数组。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_param_block_edits(
        &self,
        root: &str,
        expected: u64,
        blocks: Vec<hifishifter_kernel::param_blocks::UserParamBlock>,
        patches: &[hifishifter_kernel::param_blocks::ParamBlockCurvePatch],
        expected_curve: Option<u64>,
        expected_geometry: Option<u64>,
        expected_signature: Option<&str>,
        before_commit: impl FnOnce(),
    ) -> Result<u64, String> {
        hifishifter_kernel::param_blocks::validate_curve_patches(patches)?;
        let _transaction = self.transaction.lock_or_recover();
        let host = self.workspace_timeline_locked()?;
        if !host.tracks.iter().any(|track| track.id == root) {
            return Err("unknown root parameter line".into());
        }
        let edits = self.edits.lock_or_recover();
        let current = edits
            .param_block_edits
            .get(root)
            .cloned()
            .unwrap_or_default();
        let next_edits = current.replace(expected, blocks)?;
        if !patches.is_empty() && expected_curve != Some(edits.authority.revision) {
            return Err("Conflict: parameter curve changed".into());
        }
        if !patches.is_empty()
            && expected_signature.is_some_and(|signature| {
                signature
                    != hifishifter_kernel::pitch_analysis::build_root_pitch_layout_key(&host, root)
            })
        {
            return Err("Conflict: parameter geometry changed".into());
        }
        if !patches.is_empty()
            && expected_signature.is_none()
            && expected_geometry.is_some_and(|revision| {
                revision != self.ui_geometry_revision.load(Ordering::Acquire)
            })
        {
            return Err("Conflict: parameter geometry changed".into());
        }
        let mut next = edits.clone();
        let children = self.private_group_descendants(&next, root);
        for patch in patches {
            for target in std::iter::once(root).chain(children.iter().map(String::as_str)) {
                let default = Self::authority_param_default(&host, target, &patch.param);
                let values = patch
                    .values
                    .iter()
                    .map(|value| Self::normalize_authority_value(&patch.param, *value, default))
                    .collect::<Vec<_>>();
                next.authority.write_range(
                    target,
                    &patch.param,
                    patch.start_frame as usize,
                    &values,
                    &vec![true; values.len()],
                    default,
                )?;
            }
        }
        let changed = next.authority.revision != edits.authority.revision;
        next.param_block_edits.insert(root.to_owned(), next_edits);
        next.revision = next.revision.saturating_add(1);
        if next.tracks.is_empty() {
            next.tracks = host.tracks.clone();
        }
        for target in std::iter::once(root).chain(children.iter().map(String::as_str)) {
            let view = self.compose_param_view_with(&host, &next, target);
            next.params.insert(target.to_owned(), view.params);
        }
        let revision = next.authority.revision;
        drop(edits);
        before_commit();
        *self.edits.lock_or_recover() = next;
        self.archive_cache.lock_or_recover().take();
        if changed {
            self.notify_parameters_changed();
        }
        Ok(revision)
    }

    pub(crate) fn write_authority_range(
        &self,
        track_id: &str,
        param: &str,
        start_frame: u32,
        values: &[f32],
        present: Option<&[bool]>,
        checkpoint: bool,
    ) -> Result<serde_json::Value, String> {
        if present.is_some_and(|present| present.len() != values.len()) {
            return Err("values/present length mismatch".into());
        }
        let _transaction = self.transaction.lock_or_recover();
        let host = self.workspace_timeline_locked()?;
        let root = host
            .resolve_root_track_id(track_id)
            .ok_or_else(|| "unknown parameter track".to_owned())?;
        let default = Self::authority_param_default(&host, &root, param);
        let normalized = values
            .iter()
            .map(|value| Self::normalize_authority_value(param, *value, default))
            .collect::<Vec<_>>();
        let present = present
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| vec![true; normalized.len()]);
        self.checkpoint_authority(checkpoint)?;
        let mut edits = self.edits.lock_or_recover();
        // `EditState.tracks` 是轨道级设置的持久化容器；新权威写路径也必须
        // 在第一次写入时把当前宿主轨道登记进去，否则存档/诊断会看到空轨道表。
        if edits.tracks.is_empty() {
            edits.tracks = host.tracks.clone();
        }
        let before = edits.authority.revision;
        let runs_before = edits.authority.run_count();
        edits.authority.write_range(
            &root,
            param,
            start_frame as usize,
            &normalized,
            &present,
            default,
        )?;
        // 【分组根落笔 = 整组落笔】理由见 `private_group_descendants`。
        let group_roots = self.private_group_descendants(&edits, &root);
        for child in &group_roots {
            let child_default = Self::authority_param_default(&host, child, param);
            let child_values = values
                .iter()
                .map(|value| Self::normalize_authority_value(param, *value, child_default))
                .collect::<Vec<_>>();
            if let Err(error) = edits.authority.write_range(
                child,
                param,
                start_frame as usize,
                &child_values,
                &present,
                child_default,
            ) {
                // 【一条子轨写不进去不该让整笔编辑失败】根已经写下了；把它降级成一行
                // 日志，比让用户"画完了却收到一个错误、曲线又回弹"好。
                log::warn!("[param] private group write to {child} failed: {error}");
            }
        }
        let revision = edits.authority.revision;
        let changed = revision != before;
        if changed {
            edits.revision = edits.revision.saturating_add(1);
            // 显示缓存与权威同代更新；GUI 的下一帧不能读到上一代数组。
            for target in std::iter::once(&root).chain(group_roots.iter()) {
                let view = self.compose_param_view_with(&host, &edits, target);
                edits.params.insert(target.clone(), view.params);
            }
            // 【`param.mutate` 是唯一的曲线改动读数】它把"谁改的"写进 `cause`：
            // 用户操作（write）、撤销还原（restore）、几何事务（geometry）走的是三条
            // 不同的路径，而此前它们在日志里完全同形 —— 于是一次"参数线自己变了"
            // 根本无从归因。`runs=A->B` 给出这条根上的 run 条数变化，它是"多出一条
            // 重叠的 run"这类内部错误的唯一廉价判据。
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "param.mutate",
                format_args!(
                    "[param] mutate cause=write root={root} param={param} range=[{start_frame}, {}) \
                     present={} group_roots={} runs={runs_before}->{} rev={revision}",
                    start_frame as usize + normalized.len(),
                    present.iter().filter(|value| **value).count(),
                    group_roots.len(),
                    edits.authority.run_count()
                ),
            );
        }
        drop(edits);
        if changed {
            self.notify_parameters_changed();
        }
        Ok(serde_json::json!({"ok": true, "authority_revision": revision}))
    }

    /// 一次重置请求最多擦多少帧。
    ///
    /// 【为什么不是 `project_sec`】run 活在工程时间、与 clip 无关；用户可以在最后一个
    /// clip 之后再画一条线（写入路径一直允许），那么他也必须能擦掉它。旧实现把区间夹到
    /// `target_param_frames`（= 最后一个 clip 的终点），于是"clip 之外"的重置永远
    /// `count == 0` —— 返回 `ok:true` 却什么都没改，前端下一次取数就把线取回来了
    /// （用户报的"回弹"）。
    ///
    /// 现在唯一的上限是"一次请求最多处理多少帧"，与 `write_range` 的
    /// `MAX_RUN_VALUES` 同量级：再大也只是遍历现有 run，不分配内存。
    const MAX_RESTORE_FRAMES: usize = 1_000_000;

    pub(crate) fn restore_authority_range(
        &self,
        track_id: &str,
        param: &str,
        start_frame: u32,
        frame_count: u32,
        checkpoint: bool,
    ) -> Result<serde_json::Value, String> {
        let _transaction = self.transaction.lock_or_recover();
        let host = self.workspace_timeline_locked()?;
        let root = host
            .resolve_root_track_id(track_id)
            .ok_or_else(|| "unknown parameter track".to_owned())?;
        let start = start_frame as usize;
        let count = (frame_count as usize).min(Self::MAX_RESTORE_FRAMES);
        if count == 0 {
            return Ok(serde_json::json!({
                "ok": true,
                "changed": false,
                "authority_revision": self.authority_revision(),
            }));
        }
        self.checkpoint_authority(checkpoint)?;
        let mut edits = self.edits.lock_or_recover();
        let (mut revision, mut changed) = edits.authority.restore_range(&root, param, start, count);
        // 【分组根重置 = 整组重置】与写入同一条理由（见 `private_group_descendants`）：
        // 右键"重置"作用的是分组共享的那一条线，子轨必须一起回到基线。
        let group_roots = self.private_group_descendants(&edits, &root);
        for child in &group_roots {
            let (child_revision, child_changed) =
                edits.authority.restore_range(child, param, start, count);
            revision = revision.max(child_revision);
            changed |= child_changed;
        }
        if changed {
            edits.revision = edits.revision.saturating_add(1);
            // 【显示必须与权威同代】旧实现只推进 revision，显示要等下一次
            // workspace_view() —— 而右键重置之后紧接着的一次快照可能来自别的路径
            // （渲染、诊断），用户看到的就是"松手又回来"。这里与写入路径同口径。
            for target in std::iter::once(&root).chain(group_roots.iter()) {
                let view = self.compose_param_view_with(&host, &edits, target);
                edits.params.insert(target.clone(), view.params);
            }
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.param.restore",
                format_args!(
                    "[param] restore root={root} param={param} range=[{start}, {}) rev={revision}",
                    start + count
                ),
            );
        }
        drop(edits);
        if changed {
            self.notify_parameters_changed();
        }
        Ok(serde_json::json!({
            "ok": true,
            "changed": changed,
            "authority_revision": revision,
        }))
    }

    /// 插件本地参数撤回占位：前端状态已由 history 记录驱动；这里只保证写入事务的
    /// revision 前进，几何/轨道历史仍由既有 TimelineHistory 负责。
    fn checkpoint_authority(&self, _checkpoint: bool) -> Result<(), String> {
        // Undo 快照由 EditorSession 的 authority_history 负责；文档 revision 不是
        // 参数版本，不能在这里推进（否则一次画线会被误判成宿主模型变化）。
        Ok(())
    }

    /// 一条 clip 的几何身份：item GUID 优先，其次 region key，最后才是 clip id。
    ///
    /// 【为什么不能直接用 `clip.id`】它会在 `ara-region-{key}` 与 `ara-item-{guid}`
    /// 之间迁移；几何日志一旦按 id 建键，迁移就会让日志"少一条、多一条"，下一次移动
    /// 因此算不出 Move（用户看到的正是"锁开了跟没开一样"）。
    fn geometry_key_for(
        clip: &hifishifter_kernel::state::Clip,
        region_items: &std::collections::HashMap<u64, String>,
    ) -> hifishifter_kernel::editor::param_geometry::GeometryKey {
        use hifishifter_kernel::editor::param_geometry::GeometryKey;
        if let Some(guid) = crate::editor::workspace::host_item_guid(&clip.id) {
            return GeometryKey::item(guid);
        }
        if let Some(key) = parse_region_key_from_clip_id(&clip.id) {
            if let Some(guid) = region_items.get(&key) {
                return GeometryKey::item(guid);
            }
            return GeometryKey::region(key);
        }
        GeometryKey::clip(&clip.id)
    }

    /// 这一轮时间线的几何事实。
    ///
    /// `complete` 只能由"枚举覆盖了本文档全部 region"的调用方声明（模型收口）；呈现层
    /// 过滤后的视图必须传 `false` —— 否则"这一轮没看到"会被当成"被删了"，几何日志被
    /// 空帧清空。
    pub(crate) fn geometry_frame(
        &self,
        timeline: &hifishifter_kernel::state::TimelineState,
        complete: bool,
    ) -> hifishifter_kernel::editor::param_geometry::GeometryFrame {
        use hifishifter_kernel::editor::param_geometry::{ClipGeometry, GeometryFrame};
        let region_items = self.region_items.lock_or_recover();
        let clips = timeline
            .clips
            .iter()
            .filter_map(|clip| {
                let root = timeline.resolve_root_track_id(&clip.track_id)?;
                let key = Self::geometry_key_for(clip, &region_items);
                let mut geometry = ClipGeometry::from_clip(clip, &root, key);
                if geometry.source_id.is_empty() {
                    geometry.source_id = clip
                        .source_path
                        .clone()
                        .unwrap_or_else(|| format!("identity:{}", clip.id));
                }
                Some(geometry)
            })
            .collect();
        GeometryFrame::new(clips, complete)
    }

    /// 把这一帧宿主几何与日志做一次事务；锁定关闭只更新日志。
    ///
    /// 【为什么收 `GeometryFrame` 而不是裸数组】`complete` 是"能不能剪枝"的唯一判据。
    /// 旧签名（`&[ClipGeometry]`）让"空帧"与"全部删除"长得一模一样，日志因此会被
    /// 呈现层的任何一次过滤清空。
    pub(crate) fn reconcile_parameter_geometry_locked(
        &self,
        frame: &hifishifter_kernel::editor::param_geometry::GeometryFrame,
        edits: &mut crate::state_channel::EditState,
    ) -> Result<hifishifter_kernel::editor::param_geometry::TransformReport, String> {
        use hifishifter_kernel::editor::param_geometry::apply_geometry_transaction;
        let locked = crate::settings_store::lock_param_lines();
        let mut journal = self.geometry_journal.lock_or_recover();
        let deltas = journal.diff(frame);
        // 【诊断】几何是"锁定失效"类问题的第一现场：这条读数回答"这一帧有几条
        // clip、日志里有几条、算出了哪些 delta"。没有它，用户报"锁没生效"时只能
        // 从代码反推哪一步断了（见 docs/superpowers/plans/…-round2.md §1.5）。
        if !deltas.is_empty() || frame.clips.is_empty() {
            // 【为什么每个 delta 都要带两个窗口】"判成了拉伸"本身不是答案：
            // 真正的答案在"哪一项让它这么判"。几何分类只看三个宿主事实（工程窗口、
            // 源窗口、播放倍率），而它们全都曾经只存在于内存里 —— 于是"分割一个
            // item 却把参数线压进半段"只能靠反推代码才找得到（循环源的源窗口不随
            // 长度收缩 ⇒ 除出来的倍率翻倍 ⇒ 冒充拉伸）。把这两个窗口打进同一行，
            // 下一次现场日志就能直接指认。
            let ops = deltas
                .iter()
                .map(|delta| {
                    use hifishifter_kernel::editor::param_geometry::GeometryOp;
                    let op = match delta.op {
                        GeometryOp::Move { .. } => "move",
                        GeometryOp::Stretch { .. } => "stretch",
                        GeometryOp::Copy { .. } => "copy",
                        GeometryOp::TrimOrNone => "trim",
                        GeometryOp::Replace => "replace",
                    };
                    let old = journal.entries().get(&delta.key);
                    let new = frame.clips.iter().find(|clip| clip.key == delta.key);
                    match (old, new) {
                        (Some(old), Some(new)) => format!(
                            "{}:{op}{{{} -> {}}}",
                            delta.key,
                            old.trace_facts(),
                            new.trace_facts()
                        ),
                        _ => format!("{}:{op}", delta.key),
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.geom.trace",
                format_args!(
                    "[geom] clips={} complete={} journal={} locked={locked} deltas=[{ops}]",
                    frame.clips.len(),
                    frame.complete,
                    journal.entries().len(),
                ),
            );
        }
        if !locked {
            journal.observe_only(frame);
            return Ok(Default::default());
        }
        let report = match apply_geometry_transaction(&mut edits.authority, &deltas) {
            Ok(report) => report,
            Err(error) => {
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Error,
                    "ara.param.geometry-failed",
                    format_args!("[param] geometry transaction deferred: {error}"),
                );
                return Err(error);
            }
        };
        if report.affected_runs > 0 {
            // 只推进编辑代次。派生显示由调用方在自己的时间线上重新合成
            // （`materialize_views_locked` 会整份替换 `edits.params`，而这里拿到的
            // `timeline` 可能是另一份宿主视图 —— 用它覆盖会把在线权威的根 id
            // 写成宿主新 id，下一次 reconcile 会把这条轨判成"认不出"）。
            edits.revision = edits.revision.saturating_add(1);
        }
        if !report.is_empty() {
            // 【为什么几何事务也要走 `param.mutate`】此前它**一条读数都没有**：
            // 曲线被搬走了，日志里却只有一行 `[geom] ... deltas=[...]`（那是分类器
            // 的输入，不是"权威真的变了"）。于是"锁开着但线没动"与"锁关着线却动了"
            // 这两种相反的报障，在日志里长得一模一样 —— 都只有 geom 那一行。
            // 与用户画线共用同一个键，是为了让"这条曲线什么时候被谁改过"只有一条流。
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "param.mutate",
                format_args!(
                    "[param] mutate cause=geometry affected={} split={} cleared={} dropped={} \
                     deltas={} rev={}",
                    report.affected_runs,
                    report.split_runs,
                    report.cleared_runs,
                    report.dropped_frames,
                    report.applied_deltas,
                    edits.revision
                ),
            );
        }
        let epoch = journal.epoch();
        journal.commit(frame, epoch);
        *self.last_geometry_report.lock_or_recover() = Some(report.clone());
        Ok(report)
    }

    /// 按 ARA 2 的部分存档筛选产出载荷。
    ///
    /// 【为什么需要它】整份存档在"只改了一条 modification"时也要重写全部状态，而
    /// `storeObjectsToArchive` 的筛选正是为了按对象粒度保存（`ARAInterface.h:2979-2988`）。
    /// 不支持它时宿主会拿到 `kARAFalse`，于是它做增量保存时**丢失**这次编辑。
    ///
    /// `modifications == None` 表示"全部选中"；`document_data` 表示是否带上文档级状态
    /// （参数表 / 分组 / 空白编辑 / 逐片段私有属性）。
    pub(crate) fn encode_partial(
        &self,
        modifications: Option<&std::collections::BTreeSet<String>>,
        document_data: bool,
    ) -> Result<Vec<u8>, String> {
        let _transaction = self.transaction.lock_or_recover();
        if !self.ready.load(Ordering::Acquire) {
            // 部分存档没有"最后一份同筛选快照"可复用；返回完整缓存是安全的超集，
            // 宿主恢复时会按文档状态合入。没有缓存才返回空载荷（而不是失败）。
            if let Some(cached) = self.archive_cache.lock_or_recover().clone() {
                log::warn!(
                    "[ara] partial archive store uses the last full snapshot ({} byte(s)); \
                     the host graph is not ready",
                    cached.len()
                );
                return Ok(cached);
            }
            log::warn!(
                "[ara] partial archive store degraded to an empty payload; the host graph is \
                 not ready"
            );
            return Ok(Vec::new());
        }
        let mut edits = self.edits.lock_or_recover();
        if let Err(error) = edits.reconcile(&self.track_bindings.lock_or_recover()) {
            log::warn!("[ara] partial archive reconcile failed: {error}");
        }
        // 【为什么先整份克隆再裁剪】从 `EditState::default()` 起手会把**所有**状态
        // 一起丢掉 —— 包括本次真正要保存的那几条 modification 参数。裁剪必须是
        // "从完整状态里挑出被选中的部分"，而不是"从空状态里填"。
        let mut partial = edits.clone();
        // 【部分存档的裁剪口径：根轨】权威是 root 作用域的，而 ARA 的"选中部分"是按
        // modification 选的。两者之间没有直接边，于是这里退一步：只要本实例此刻服务的
        // 根轨上还有被选中的 modification 的 region，就保留那个根的全部曲线。
        // 宁可多带一条轨的曲线，也不要按猜出来的边把用户数据裁掉。
        let keep = self
            .track_bindings
            .lock_or_recover()
            .iter()
            .filter(|(_, members)| {
                members.iter().any(|(modification, _)| {
                    modifications.is_none_or(|wanted| wanted.contains(modification))
                })
            })
            .map(|(root, _)| root.clone())
            .collect::<std::collections::BTreeSet<_>>();
        partial.authority.retain_roots(&keep);
        partial
            .param_block_edits
            .retain(|root, _| keep.contains(root));
        if !document_data {
            // 文档级状态没被选中 ⇒ 载荷里不带它，恢复端因此不会覆盖它。
            partial.params.clear();
            partial.param_block_edits.clear();
            partial.tracks.clear();
            partial.bindings.clear();
            partial.fades.clear();
            partial.clip_local.clear();
            partial.groups = Default::default();
        }
        match partial.encode() {
            Ok(bytes) => Ok(bytes),
            Err(error) => {
                log::error!("[ara] partial archive store degraded to an empty payload: {error}");
                Ok(Vec::new())
            }
        }
    }

    /// 接收 ARA 2 的部分存档载荷，按 `archive → current` 重映射后暂存。
    pub(crate) fn restore_archive_partial(
        &self,
        bytes: &[u8],
        remap: &[(String, String)],
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut restored = crate::state_channel::EditState::default();
        // 与整份恢复同一条纪律：解码失败不能让宿主中止恢复。
        if let Err(error) = restored.restore(bytes) {
            log::error!(
                "[ara] partial archive restore ignored: {error}; the host document restore \
                 continues without these edits"
            );
            return Ok(());
        }
        // 【曲线不按 modification 重映射】权威的键是 (root, param, 工程秒)；跨文档
        // 导入要重映射的是**轨道身份**（由 `reconcile` 按 bindings 处理），与曲线无关。
        // 图集时代这里必须把 `mod:{旧ID}` 下的曲线搬到新键，因为曲线按 modification
        // 索引 —— 那份表示已经删除。
        let _ = &remap;
        self.pending_partial_restore
            .lock_or_recover()
            .push(restored);
        Ok(())
    }
    /// 把暂存的部分载荷**按键合并**进编辑权威（`prepare_renderers` 收口时调用）。
    ///
    /// 【为什么是合并而不是替换】部分载荷只带它被选中的那一部分；替换会把未选中的
    /// 状态（其它 modification 的参数、文档级参数表）清空。
    fn merge_partial_restores(&self) {
        let pending = self
            .pending_partial_restore
            .lock_or_recover()
            .drain(..)
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return;
        }
        let mut edits = self.edits.lock_or_recover();
        for partial in pending {
            for (root, authority) in partial.authority.roots {
                edits.authority.roots.insert(root, authority);
            }
            edits.params.extend(partial.params);
            if !partial.groups.is_empty() {
                edits.groups = partial.groups;
            }
            edits.fades.extend(partial.fades);
            edits.clip_local.extend(partial.clip_local);
            edits.revision = edits.revision.saturating_add(1);
        }
    }

    /// 本地补丁（T8）：接管本份文档控制器的图账本。
    ///
    /// 【为什么在文档上而不是各处传】诊断路径（`stuck_report` / 证据包）只需要
    /// "这一份文档的图走到哪一步"。记在这里就不必让每个调用方持有桥的类型。
    /// 【幂等】宿主每建一个控制器就接管一次；重绑时换成新账本，旧读数随之失效 ——
    /// 那正是我们要的（新文档从零开始记账）。
    pub(crate) fn set_graph_ledger(
        &self,
        ledger: std::sync::Arc<ara2_bridge::plugin::GraphLedger>,
    ) {
        *self.graph_ledger.lock_or_recover() = Some(ledger);
    }

    /// 图账本（没有接管过时为 `None`）。
    pub(crate) fn graph_ledger(&self) -> Option<std::sync::Arc<ara2_bridge::plugin::GraphLedger>> {
        self.graph_ledger.lock_or_recover().clone()
    }
    /// 注入 ARA 变更通知队列（`runtime.rs` 构建插件时调用一次）。
    pub(crate) fn set_update_emitter(&self, emitter: ara2_bridge::plugin::UpdateEmitter) {
        *self.update_emitter.lock_or_recover() = Some(emitter);
    }

    /// 登记一个 modification 的宿主句柄（`createAudioModification` / clone 时）。
    pub(crate) fn register_modification_handle(
        &self,
        persistent_id: &str,
        handle: ara2_bridge::core::RawHandle,
    ) {
        if persistent_id.is_empty() {
            return;
        }
        self.modification_ids
            .lock_or_recover()
            .insert(handle.identity(), persistent_id.to_owned());
        self.modification_handles
            .lock_or_recover()
            .insert(persistent_id.to_owned(), handle);
    }

    /// 宿主销毁 / 换掉一个 modification 的 persistentID 时同步。
    pub(crate) fn forget_modification_handle(&self, persistent_id: &str) {
        if let Some(handle) = self
            .modification_handles
            .lock_or_recover()
            .remove(persistent_id)
        {
            self.modification_ids
                .lock_or_recover()
                .remove(&handle.identity());
        }
    }

    /// 宿主句柄 → persistentID（部分存档的写出侧筛选）。
    pub(crate) fn modification_id_for_handle(
        &self,
        index: std::num::NonZeroU32,
        session: std::num::NonZeroU64,
    ) -> Option<String> {
        self.modification_ids
            .lock_or_recover()
            .get(&(index, session))
            .cloned()
    }

    /// **参数变了 ⇒ 告诉宿主这些 modification 的持久状态变了。**
    ///
    /// 【为什么宁可多报】通知是幂等且会被合并的（桥接层的 `UpdateTracker`），而漏报
    /// 会让"只存增量"的宿主丢数据。因此这里按当前登记的全部句柄投递，不做精确差分。
    ///
    /// 【为什么信号位为空】音高 / 共振峰 / 气声一变，信号本身就变了，因此
    /// `ContentUpdateScopes` 不得置 `SIGNAL_REMAINS_UNCHANGED`。
    pub(crate) fn notify_parameters_changed(&self) {
        let emitter = self.update_emitter.lock_or_recover().clone();
        let Some(emitter) = emitter else {
            return;
        };
        let handles = self
            .modification_handles
            .lock_or_recover()
            .values()
            .copied()
            .collect::<Vec<_>>();
        let mut refused = 0_usize;
        for handle in handles {
            if emitter
                .mark_modification(
                    handle,
                    None,
                    ara2_bridge::core::ContentUpdateScopes::empty(),
                    ara2_bridge::plugin::UpdateOrigin::Application,
                )
                .is_err()
            {
                refused += 1;
            }
        }
        if refused > 0 {
            // 队列是进程内的，失败只可能是"这个句柄不属于当前会话"（宿主换了图）。
            // 限流留痕：静默吞掉会让"宿主没保存"无从复盘。
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.update.refused",
                format_args!(
                    "[ara] {refused} modification update notification(s) were refused by the                      queue; the host will not be told those objects changed"
                ),
            );
        }
        // 文档级私有状态（分组 / 空白编辑 / 参数权威本身）也在同一次编辑里变了。
        emitter.mark_document(ara2_bridge::plugin::UpdateOrigin::Application);
    }

    /// 工厂成功返回前登记真实 controllerRef，失败或销毁时撤销。
    ///
    /// 【为什么在这里清扫】region 身份表是**进程级**的，而上一次文档可能没有走完
    /// `destroy_document`（宿主不按 leaf-first 销毁）就留下了无主 key。新文档注册时
    /// 主动扫掉"归属文档已死"的登记，让"重开工程"不依赖后续创建顺序就能摆脱上一次
    /// 的残留 —— 否则 `create_playback_region` 会在同一批地址上撞车，
    /// 宿主从此不再交付任何 region，界面永远停在"等待宿主音频"。
    ///
    /// 【为什么逐个问存活】同一进程里可以同时有多份文档（多个工程窗口）。只清死
    /// 文档的登记，绝不能全表清空。
    pub fn register(self: &Arc<Self>, key: usize) {
        self.controller.store(key, Ordering::Release);
        let mut table = controllers().lock_or_recover();
        let dead: Vec<usize> = table
            .iter()
            .filter(|(_, weak)| {
                weak.upgrade()
                    .is_none_or(|session| !session.alive.load(Ordering::Acquire))
            })
            .map(|(controller, _)| *controller)
            .collect();
        for controller in &dead {
            table.remove(controller);
        }
        if !dead.is_empty() {
            log::info!(
                "[ara] swept {} controller entrie(s) of dead document(s) before registering \
                 document {}",
                dead.len(),
                self.id
            );
        }
        // 同一份存活判定扫 region 表：controller 表活着但文档已 close 的也算死
        // （`close()` 已经把它从表里摘掉，因此这里看到的一定是活的）。
        let alive_ids: Vec<u64> = table
            .values()
            .filter_map(|weak| weak.upgrade())
            .filter(|session| session.alive.load(Ordering::Acquire))
            .map(|session| session.id)
            .collect();
        drop(table);
        crate::render::ownership::region_owners()
            .lock_or_recover()
            .retain_alive_documents(&|document| alive_ids.contains(&document));
        controllers()
            .lock_or_recover()
            .insert(key, Arc::downgrade(self));
    }

    /// 同步关闭文档；保持 renderer 的原生接口存储，但撤销模型操作许可。
    pub fn close(&self) {
        {
            let _transaction = self.transaction.lock_or_recover();
            if !self.alive.swap(false, Ordering::AcqRel) {
                return;
            }
            self.scope_revision.fetch_add(1, Ordering::AcqRel);
        }
        // 不持transaction join：actor可能正在收尾短事务；先停止全部编辑/分析再清宿主图。
        self.host_undo.close();
        if let Some(Ok(editor)) = self.editor.get() {
            editor.close();
        }
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.playback.lock_or_recover().take();
        self.ready.store(false, Ordering::Release);
        // 【T1】文档关闭也是"渲染许可失效"的一种。对这份文档本身已无意义，
        // 但让三个置假点语义一致是显式不变量，避免后来者只改其中两个。
        self.note_content_change();
        // 文档关闭时不存在"进行中的宿主会话"；留着会让下一次 ensure_loaded 误判为暂缓。
        self.editing_session.store(false, Ordering::Release);
        let leases = {
            let mut renderers = self.renderers.lock_or_recover();
            self.alive.store(false, Ordering::Release);
            std::mem::take(&mut *renderers)
        };
        let mut controllers = controllers().lock_or_recover();
        let key = self.controller.load(Ordering::Acquire);
        if controllers
            .get(&key)
            .and_then(Weak::upgrade)
            .is_some_and(|current| current.id == self.id)
        {
            controllers.remove(&key);
        }
        drop(controllers);
        for renderer in leases {
            renderer.lease.destroy();
            if let Some(owner) = renderer.owner.upgrade() {
                owner.document_closed();
            }
            // companion guard 随 renderer 释放，禁止继续借用即将销毁的 controller。
        }
        self.sequence_regions.lock_or_recover().clear();
        self.sequence_track_index.lock_or_recover().clear();
        self.regions.lock_or_recover().clear();
        // 【为什么先取本文档的 clip id 再清表】共振峰代次表在内核里是**进程级**的，
        // 同一进程可以同时开着多份工程（多个 ARA 文档）。整表清会把另一份仍在编辑的
        // 工程的活跃代次一并抹掉 —— 那边的后台重建随后会以为代次过期而静默放弃，
        // 表现为"另一个工程里的波形/共振峰不再更新"。只清本文档持有的那些。
        let owned_clip_ids = self
            .clip_ids
            .lock_or_recover()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        self.clip_ids.lock_or_recover().clear();
        self.region_items.lock_or_recover().clear();
        // 授权/事实映射随文档一起释放：此前 `region_items` / `authorized_takes` /
        // `authorized_sources` / `split_media_from` 只在文档销毁时清，而 region 销毁
        // 回调并不回收它们 —— 长会话里每次分割/移动/粘贴都会留下永不释放的条目。
        self.authorized_takes.lock_or_recover().clear();
        self.authorized_sources.lock_or_recover().clear();
        self.split_media_from.lock_or_recover().clear();
        self.source_failures.lock_or_recover().clear();
        self.source_states.lock_or_recover().clear();
        self.region_rejections.lock_or_recover().clear();
        self.pending_since.lock_or_recover().clear();
        self.skipped_regions.lock_or_recover().clear();
        self.ui_tracks_guard().clear();
        self.ui_known_tracks.lock_or_recover().clear();
        *self.ui_inventory_stamp.lock_or_recover() = None;
        *self.ui_inventory_sampled_at.lock_or_recover() = None;
        *self.render_facts.lock_or_recover() = Default::default();
        self.timeline.lock_or_recover().take();
        self.track_bindings.lock_or_recover().clear();
        *self.edits.lock_or_recover() = Default::default();
        self.sources.lock_or_recover().clear();
        self.edit_sources.lock_or_recover().clear();
        self.media_facts.lock_or_recover().clear();
        *self.host_tempo_map.lock_or_recover() = None;
        *self.pending_archive_restore.lock_or_recover() = None;
        // 【为什么文档关闭时要清共振峰代次表】那张表在**内核里是进程级**的
        // （`FORMANT_REBUILD_GENERATIONS`），按 clip_id 索引且只增不减：条目仅在同一个
        // clip 被再次重建时覆写，而工程切换后旧 clip_id 不会再出现 —— 它的清理函数
        // 此前**没有任何调用者**。单条很小，但反复导入工程会一直涨。
        // 只清本文档的 id，别动同进程里的另一份工程。
        hifishifter_kernel::formant_cache::forget_formant_rebuild_generations(
            owned_clip_ids.iter().map(String::as_str),
        );
    }

    /// 控制器侧独立保留 lease，companion 先释放时 raw extension 存储仍须存活。
    pub fn attach(
        &self,
        owner: &Arc<ExtensionOwner>,
        lease: ExtensionControllerLease,
        companion: Option<CompanionControllerBinding<'static>>,
    ) -> Result<(), AraError> {
        let mut renderers = self.renderers.lock_or_recover();
        if !self.alive.load(Ordering::Acquire) {
            return Err(AraError::InvalidState("document is destroyed"));
        }
        renderers.push(RendererLease {
            lease,
            _companion: companion,
            owner: Arc::downgrade(owner),
        });
        self.ever_had_renderer.store(true, Ordering::Release);
        Ok(())
    }

    /// 本文档是否曾经有过 renderer（用于区分"还没绑定"与"绑定过又丢了"）。
    pub(crate) fn ever_had_renderer(&self) -> bool {
        self.ever_had_renderer.load(Ordering::Acquire)
    }

    /// 把某个 owner 的 lease 从本文档摘掉（文档本身仍存活）。
    ///
    /// 【为什么需要它】宿主复用同一个处理器、把实例从旧文档迁到新文档时，旧文档
    /// 可能仍然活着。此时必须只摘掉**这一个** owner 的租约，不能关闭整份文档，
    /// 也不能打断同文档里其他实例（多工程窗口/多实例）。
    pub(crate) fn detach_owner(&self, owner: &Arc<ExtensionOwner>) {
        let mut renderers = self.renderers.lock_or_recover();
        renderers.retain(|lease| {
            !lease
                .owner
                .upgrade()
                .is_some_and(|current| Arc::ptr_eq(&current, owner))
        });
    }

    /// 模型线程刷新所有仍存活 renderer，任何 source/geometry/访问变化都触发重新准备。
    pub fn prepare_renderers(&self) {
        let _startup_span = crate::startup::Span::new("ara.prepare_renderers", self.id, 0);
        let _transaction = self.transaction.lock_or_recover();
        // 【T2：`ready` 的语义收窄为"模型已收口"，不再假装"输入可用"】
        //
        // 它此前同时承担两件事："我们跑过一次 settle"与"宿主交付了可用输入"。
        // 后者根本无从判断（宿主可以授权 0 个源而一切"成功"），于是重开工程后
        // 界面一片"等待宿主音频"、而日志里没有任何异常。
        //
        // 【为什么**不**在这里按 `input_verdict` 置假】那样会把"宿主还没交音频"
        // 变成"渲染许可失效"，让 `ara_render_input` 在每一次正常的加载窗口里
        // 都失败一次 —— 而失败的下游是"快照停在上一版"，比静音更难排查。
        // 判据的真正消费者是**呈现**：`ensure_loaded` 据此区分"等音频"与"真冲突"，
        // `decorate_host_audio` 据此给出可行动的分类（见 `input_verdict` 的读者）。
        self.ready.store(true, Ordering::Release);
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect::<Vec<_>>();
        // 【ARA 存档的落点：由**文档**合并，不再交给某个 owner】
        //
        // 载荷是**文档级**的（`storeDocumentToArchive` 是文档控制器的回调），而合入
        // 判据此前是**逐实例**的（只认本实例被分配的 region）。工程里只要有两条轨有
        // 参数编辑，"不属于本实例的那条轨"就会 reconcile 成 Unobserved ⇒ 载荷永久
        // pending ⇒ 而 I3 守卫又让此后所有新编辑都不落盘。用户的原话是"编辑了也白忙活"。
        //
        // 现在合入只有一个入口：`merge_archive_restore_locked`（候选集是整份
        // `track_bindings`）。部分存档先合入（按键合并，不会覆盖整份状态），再走整份。
        self.merge_partial_restores();
        // 【为什么这里是 `_locked`】本函数已经持有 `transaction`，而公开入口要取同一把
        // 不可重入的锁 —— 那会当场自锁（用户报的"重开 ARA 工程宿主未响应"）。
        match self.merge_archive_restore_locked() {
            Ok(true) => log::info!("[ara] document archive merged into the edit authority"),
            Ok(false) => {}
            Err(error) => log::warn!("[ara] document archive merge failed: {error}"),
        }

        for owner in &owners {
            if let Err(error) = owner.merge_pending_restore(self) {
                log::warn!("[ara] instance state unresolved: {error}");
            }
        }
        drop(_transaction);
        // 【B0】收口之后打一行读数（同样的读数不重复）。放在释放事务之后：
        // `diag_line` 要取多个文档锁，而本函数的锁序已经足够长，没必要再加一层。
        self.log_diag_if_changed();
        for owner in owners {
            owner.refresh_host_state_for_model();
            // 【这里**绝不**能等】本函数的主要调用者是 ARA 的 `end_editing`（`remap_and_log`
            // 的尾巴）与存档回调 —— 两者都在宿主线程上，且都是宿主**同步**进入我们的代码。
            // 传 `None` 让首快照只排队：等它就是把宿主的消息泵扣在这里，现场 REAPER 因此
            // 被判 `AppHangB1`（`reaper.exe stopped interacting with Windows`）。
            owner.prepare_with_budget(None);
        }
    }

    /// 记一次"宿主交回了存档"（证据，不是数据）。
    pub(crate) fn note_archive_restored(&self, bytes: usize) {
        self.archive_reading.lock_or_recover().restored_bytes = bytes;
    }

    /// 记一次"我们交给宿主一份存档"。
    pub(crate) fn note_archive_stored(&self, bytes: usize) {
        self.archive_reading.lock_or_recover().stored_bytes = bytes;
    }

    /// 记一次"这次保存交回的是**旧载荷**"（I3 分支）。
    ///
    /// 【为什么必须计数而不是只打日志】它是"编辑落不了盘"的唯一直接证据。用户报的
    /// "编辑了也白忙活"就是它连续发生；一个累计数让"偶发"与"一直"在诊断包里可分。
    pub(crate) fn note_archive_unmerged(&self, bytes: usize) {
        let mut reading = self.archive_reading.lock_or_recover();
        reading.unmerged_returns = reading.unmerged_returns.saturating_add(1);
        reading.stored_bytes = bytes;
    }

    /// 记一次"这次编辑没有被采纳"。
    pub(crate) fn note_edit_dropped(&self, reason: impl Into<String>) {
        let reason = reason.into();
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Error,
            "ara.edit.dropped",
            format_args!("[ara] edit not applied: {reason}"),
        );
        self.edit_health.lock_or_recover().drop_edit(reason);
    }

    /// 编辑采纳读数（状态快照与诊断包共用）。
    pub(crate) fn edit_health_json(&self) -> serde_json::Value {
        self.edit_health.lock_or_recover().json()
    }

    /// 记一次"存档已合入"。
    pub(crate) fn note_archive_merged(&self) {
        self.archive_reading.lock_or_recover().merged_at_ms = Self::monotonic_ms();
    }

    /// 存档通道的读数（诊断包与状态栏共用）。
    pub(crate) fn archive_reading(&self) -> serde_json::Value {
        let pending = self.pending_archive_restore.lock_or_recover().is_some()
            || self
                .renderer_owners()
                .iter()
                .any(|owner| owner.has_pending_restore());
        let reason = self
            .pending_archive_restore
            .lock_or_recover()
            .as_ref()
            .map(|_| "document-payload-not-merged".to_owned())
            .or_else(|| {
                self.renderer_owners()
                    .iter()
                    .find_map(|owner| owner.restore_pending_reason())
            });
        self.archive_reading.lock_or_recover().json(pending, reason)
    }

    /// 把一份存档载荷合入**文档级**编辑权威。
    ///
    /// 【为什么在文档上，而不是在某个实例上】`storeDocumentToArchive` 是**文档控制器**
    /// 的回调，载荷覆盖整份文档；而合入判据此前是**逐实例**的（只认本实例被分配的
    /// region）。工程里只要有两条轨有参数编辑，"不属于本实例的那条轨"就会 reconcile
    /// 成 Unobserved ⇒ 载荷永久 pending ⇒ 而"有未合入载荷就不许覆盖存档"的守卫又让
    /// 此后**所有新编辑都不落盘**。这正是"编辑了也白忙活"的机理。
    ///
    /// 【为什么候选集是整份 `track_bindings`】`track_bindings` 由真实 sequence 边写入，
    /// 是**文档级**的归属事实。按实例分配范围收窄它是 VST3 组件状态的语义，不是存档的。
    ///
    /// 【调用方必须持 `transaction`】本函数会取 `edits` 与宿主事实，锁序是
    /// `transaction → timeline → edits → host facts`。
    pub(crate) fn merge_restore_locked(
        &self,
        saved: &crate::state_channel::EditState,
        host: &hifishifter_kernel::state::TimelineState,
        bindings: &crate::state_channel::TrackBindings,
    ) -> Result<RestoreMerge, String> {
        let mut restored = saved.clone();
        // 【T18：身份对不上是"还在等"，不是"失败"】`reconcile` 的契约是严格的：
        // 任何一条旧轨在新图里找不到唯一身份就 `Err`（猜测归属会把参数挂到别人的
        // 轨道上）。但"对不上"有两种成因，处置相反：存档坏了 ⇒ 真失败；宿主**这一次**
        // 的图还没长成 ⇒ 该继续等。
        if let Err(error) = restored.reconcile(bindings) {
            match crate::render::extension::ExtensionOwner::classify_restore_identity_error(&error)
            {
                crate::render::extension::RestoreIdentityVerdict::AwaitingHostGraph => {
                    return Ok(RestoreMerge::Waiting(format!(
                        "host-graph-not-rebuilt: {error}"
                    )))
                }
                crate::render::extension::RestoreIdentityVerdict::Fatal => return Err(error),
            }
        }
        // 【第三轮：恢复不再走图集】参数编辑只有一份表示（`authority`），显示由
        // `compose_param_view_with` 在合并后统一合成（见本函数末尾）。旧实现这里还有
        // 三段图集逻辑（迁移 gaps / 从图集导出权威 / 重新 capture），它们正是"恢复之后
        // 一部分编辑看不见"的来源：图集与权威各说一套，而显示只认其中一套。
        // 存档里的基线缓存不是编辑真相，却是冷恢复第一次渲染的“原始值”来源。
        // 不在合入时装回文档基线，下面的 compose 只能看到空宿主时间线，生成的
        // 显示数组会缺 pitch_orig；渲染器于是把用户曲线当成“没有源基频”而直通原音。
        // 只补文档里还没有的根：在线分析已经发布的基线永远优先。
        for (root, params) in &restored.params {
            if self.baseline_is_ready(root) {
                continue;
            }
            let mut baseline =
                hifishifter_kernel::editor::param_authority::RootBaseline::from_params(params);
            // 【布局键必须一起播种】不写它，`baseline_needs_rebuild` 会恒为真，
            // `baseline_for` 每一帧都会尝试从（此刻还冷的）内容缓存重建，然后把
            // 这份存档基线当成"过期"再退回 —— 冷恢复首帧的可听性全靠这份基线，
            // 不能让它停在"每次都要重算一次"的状态上。
            baseline.layout_key =
                Some(hifishifter_kernel::pitch_analysis::build_root_pitch_layout_key(host, root));
            baseline.content_key = params.pitch_orig_key.clone();
            if baseline.is_ready() {
                self.publish_baseline(root, baseline);
            }
        }
        // 【客户端 = 恢复后的时间线】`merge` 用它取轨道/片段设置与显示缓存；
        // 曲线真相在下面的 `authority` 合并里（显示由 `compose_param_view_with` 重算）。
        let mut client = host.clone();
        restored.apply(&mut client);
        let mut edits = self.edits.lock_or_recover();
        let mut merged = edits.merge(host, &client, edits.revision)?;
        merged.groups = restored.groups.clone();
        let clip_ids = host
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        // 【G5：权威侧合并分两种，不能一律覆盖】
        //   * 在线状态是空的（冷启动）⇒ 整份装上，没有别的数据源；
        //   * 载荷到达时我们在一次编辑会话里（撤销/重做）⇒ 覆盖，undo 必须生效；
        //   * 其余（宿主交回一份旧载荷）⇒ 按键补缺，只补在线状态里没有的键。
        let authoritative = !restored.from_archive
            || edits.is_pristine()
            || restored.arrived_in_edit_session
            || self.authoritative_restore();
        // 【唯一权威合并】冷启动/undo 才覆盖；其余只补在线不存在的 root。
        let mut kept_live_roots = 0_usize;
        for (root, authority) in restored.authority.roots.clone() {
            if authoritative || !merged.authority.roots.contains_key(&root) {
                merged.authority.roots.insert(root, authority);
            } else {
                kept_live_roots += 1;
            }
        }
        // 音节缓存不是编辑权威，而是一份**可重建的派生缓存**：冷恢复/undo 才整份
        // 继承；普通旧载荷只补在线缺失的 Clip/Take，绝不抹掉刚算出的结果。
        if authoritative {
            merged.take_syllables = restored.take_syllables.clone();
        } else {
            for (key, analysis) in &restored.take_syllables {
                merged
                    .take_syllables
                    .entry(key.clone())
                    .or_insert_with(|| analysis.clone());
            }
        }
        merged.validate_take_syllables()?;
        for (root, edits) in &restored.param_block_edits {
            if authoritative || !merged.param_block_edits.contains_key(root) {
                merged.param_block_edits.insert(root.clone(), edits.clone());
            }
        }
        if kept_live_roots > 0 {
            log::info!(
                "[param] restore kept {kept_live_roots} live root(s); stale payload only filled missing roots"
            );
        }
        let items = self.fade_items_locked(&clip_ids);
        if !merged.fades.is_empty() && items.is_empty() && !clip_ids.is_empty() {
            // fade 归属元数据要等宿主几何刷新后才能算出来；这是一次可重试的暂缓。
            return Ok(RestoreMerge::Waiting("fade-metadata-pending".into()));
        }
        merged.fades.retain(|item, _| !items.contains(item));
        merged.fades.extend(restored.fades);
        merged.reconcile(&self.track_bindings.lock_or_recover())?;
        merged.authority.sanitize();
        // 恢复后的显示必须立刻由 authority 合成，不能沿用载荷里空的 params。
        for root in host
            .tracks
            .iter()
            .filter(|track| track.parent_id.is_none())
            .map(|track| track.id.clone())
            .collect::<Vec<_>>()
        {
            let view = self.compose_param_view_with(host, &merged, &root);
            merged.params.insert(root, view.params);
        }
        *edits = merged;
        // 恢复换了参数权威 ⇒ 旧渲染快照不再代表用户数据；必须让准备队列重算。
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        Ok(RestoreMerge::Merged)
    }

    /// 尝试把**文档级** ARA 存档载荷合入（幂等，自带事务）。
    ///
    /// 【为什么要自带事务】它有四个触发点（存档到达 / 实例绑定 / 清单推进 / 模型收口），
    /// 其中两个在文档事务之外。带上事务让调用点不必各自记住这件事。
    pub(crate) fn merge_archive_restore(&self) -> Result<bool, String> {
        let _transaction = self.transaction.lock_or_recover();
        self.merge_archive_restore_locked()
    }

    /// 调用方持 `transaction` 的版本（见 [`Self::merge_archive_restore`]）。
    pub(crate) fn merge_archive_restore_locked(&self) -> Result<bool, String> {
        // 【为什么要退避】它在工作区快照（33–400 ms 一次）上被重试。没有退避就会在
        // "宿主图还没长成"的几秒里每帧跑一次 rebind（那是 O(曲线) 的工作）。
        const RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
        if !self.ready.load(Ordering::Acquire) {
            return Ok(false);
        }
        if self.pending_archive_restore.lock_or_recover().is_none() {
            return Ok(false);
        }
        {
            let now = std::time::Instant::now();
            let mut next = self.archive_merge_next_attempt.lock_or_recover();
            if next.is_some_and(|at| now < at) {
                return Ok(false);
            }
            *next = Some(now + RETRY_INTERVAL);
        }
        let Some(saved) = self.pending_archive_restore.lock_or_recover().take() else {
            return Ok(false);
        };
        let Some(host) = self.timeline.lock_or_recover().clone() else {
            *self.pending_archive_restore.lock_or_recover() = Some(saved);
            return Ok(false);
        };
        let bindings = self.track_bindings.lock_or_recover().clone();
        let outcome = self.merge_restore_locked(&saved, &host, &bindings);
        match outcome {
            Ok(RestoreMerge::Merged) => {
                self.note_archive_merged();
                // 【合入之后必须重投影一次】合入是"权威刚换了"，而视图重建只在"几何
                // 代次变了"时发生。不补这一步，重开工程后第一帧仍然看不见曲线。
                let mut edits = self.edits.lock_or_recover();
                let mut host = host;
                if let Err(error) = self.resync_parameter_view_locked(&mut edits, &mut host) {
                    log::warn!("[ara] parameter view resync after archive merge failed: {error}");
                }
                drop(edits);
                *self.archive_merge_next_attempt.lock_or_recover() = None;
                Ok(true)
            }
            Ok(RestoreMerge::Waiting(reason)) => {
                *self.pending_archive_restore.lock_or_recover() = Some(saved);
                hifishifter_kernel::log_limiter::emit_limited_key(
                    log::Level::Warn,
                    "ara.archive.merge-waiting",
                    format_args!("[ara] archive payload is held: {reason}"),
                );
                Ok(false)
            }
            Err(error) => {
                *self.pending_archive_restore.lock_or_recover() = Some(saved);
                Err(error)
            }
        }
    }

    /// 是否还有一份**没合入**的持久化载荷（文档级 ARA 存档或任一实例的组件状态）。
    ///
    /// 【为什么存档路径要问它】`store_document` 唯一正确的回答是"把宿主交给我们的
    /// 那一份还回去"。只要还有载荷挂着，内存里的 `edits` 就**不是**用户的数据；
    /// 用它在覆盖 `archive_cache`，等于在磁盘上把用户参数换成默认值，而宿主此后
    /// 每一次保存都会把这个结果固化下去。
    pub(crate) fn has_unmerged_restore(&self) -> bool {
        if self.pending_archive_restore.lock_or_recover().is_some() {
            return true;
        }
        self.renderer_owners()
            .iter()
            .any(|owner| owner.has_pending_restore())
    }
    /// 同一ARA文档的编辑权威共享，输出仍按各renderer分配隔离。
    pub fn renderer_owners(&self) -> Vec<Arc<ExtensionOwner>> {
        self.renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect()
    }

    /// 把"本档有没有实例持 playback 角色"这一文档级事实推给每个实例，并报告读数。
    ///
    /// 【为什么必须推送而不是让每个实例自己查】`is_editor_only()` 在**音频线程**上求值
    /// （`vst3.rs::audio_process`），只能读本实例的一个原子量；而"文档上还有没有别的
    /// playback 实例"是文档级事实。这里在每次绑定/分配变化时算一次并推过去 ——
    /// 变化频率是"用户增删插件实例"，不是音频帧率。
    ///
    /// 【为什么这不再破坏透传契约】此前只能比"本实例是不是 editor"，而 REAPER 的正常
    /// 编辑实例与"Cubase 没分配 playback"在**实例级**确实同形。改比**文档级**事实之后就
    /// 区分开了：REAPER 的编辑实例另有 playback 兄弟 ⇒ 仍透传；Cubase 的单实例
    /// 没有任何 playback 兄弟 ⇒ 接管渲染（实测日志：`playback ownership: 1 live
    /// renderer instance(s) but NONE holds the playback role`，用户报的"编辑不改变声音"）。
    ///
    /// 调用点：绑定完成时、分配变化时（角色集合唯一可能改变的两个时刻）。
    pub(crate) fn report_playback_ownership(&self) {
        let owners = self.renderer_owners();
        if owners.is_empty() {
            return;
        }
        let covered = owners.iter().any(|owner| owner.has_playback_role());
        if covered {
            for owner in &owners {
                owner.set_playback_covered(true);
            }
            return;
        }
        // 没有任何实例带 playback 角色：把**已经准备好快照**的 editor 实例提为渲染者。
        //
        // 【为什么必须提升】`set_playback_covered(false)` 会让编辑实例的
        // `is_editor_only()` 翻真，于是它从"透传宿主原声"改走快照通路；而它很可能
        // 从未 `prepare()` 过（`prepare()` 在 `!renders_playback()` 时直接返回），
        // 快照是空的 —— 此时每一个音频块都是纯零。反过来，若播放实例先关闭而编辑
        // 实例没被提升，同样没人出声。两侧都指向同一个动作：挑一个能出声的实例。
        //
        // 【为什么要求"已经准备好"】没准备过的实例没有可发布的快照，提升它只会把
        // 静音从"一个实例"扩大成"另一个实例"。挑不到的场合保持 `covered = false`，
        // 让所有实例都透传宿主原声 —— 至少不是静音。
        let promoted = owners
            .iter()
            .find(|owner| owner.has_prepared_snapshot())
            .or_else(|| owners.first());
        for owner in &owners {
            owner.set_playback_covered(false);
        }
        if let Some(promoted) = promoted {
            promoted.set_playback_covered(false);
            promoted.promote_to_renderer();
        }
        // 限流：绑定与分配都可能连续触发，这条读数不需要每次都打。
        //
        // 【为什么把角色位一起打出来】`bind document` 那一行是 `INFO`，而现场日志可能被
        // 滤成 `WARN`（实测 Cubase 会话整段零行 INFO）。角色分配恰恰是"编辑听不出来"
        // 的头号线索，所以这一条必须自带全部信息：每个实例的 writer_id 与角色位。
        // 1 = playback、2 = editor、4 = view、3 = playback|editor。
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "ara.playback.no-renderer",
            format_args!(
                "[ara] playback ownership: {} live renderer instance(s) but NONE holds the \
                 playback role (writers/roles: {:?}); an instance takes over rendering so that \
                 edits are audible. If this host meant to provide a playback renderer, its role \
                 assignment is the thing to check",
                owners.len(),
                owners
                    .iter()
                    .map(|owner| (owner.writer_id(), owner.roles_bits()))
                    .collect::<Vec<_>>()
            ),
        );
    }

    /// 播放渲染器的诊断快照（诊断包用）。
    ///
    /// 【为什么只要"准备状态"】逐输出的区间读数需要当前分配区间，而诊断包在任意
    /// 时刻被导出 —— 不值得为它触发一次完整快照。这里给的是"忙不忙、有没有错、
    /// 准备的是哪一版"，正好覆盖"渲染结果陈旧/一直没准备好"这两类报障。
    pub fn renderer_diagnostics(&self) -> serde_json::Value {
        let renderers = self
            .renderer_owners()
            .into_iter()
            .filter(|owner| owner.renders_playback())
            .map(|owner| {
                let (busy, error) = owner.local_preparation_state();
                serde_json::json!({
                    "busy": busy,
                    "error": error,
                    "prepared": owner.prepared_snapshot(),
                    // 宿主是否仍在读我们的输出。它是"编辑完没有声音 / 算了但没人要"这类
                    // 报障里最先要排除的一格：bypassed 说明渲染被有意跳过（见 prepare）。
                    "host_bypass": owner.host_bypass().as_str(),
                    // 【读数有多旧】`Unknown` 与"很久没采过"是两件不同的事：前者是
                    // 宿主不提供这一族 API（非 REAPER），后者是本插件没找到采样时机。
                    // 报障时这一格能直接分开它们（"旁通了还在渲染"最常见的原因就是
                    // 这一格一直是 `None`）。
                    "host_bypass_age_ms": owner.host_bypass_age().map(|age| age.as_millis() as u64),
                    // 【确定读数有多旧】`None` = 从来没读到过。它与上面那一格合起来能
                    // 分开"没采过"（两格都是 None）与"采了但没读到"（上面有值、这里 None）。
                    "host_bypass_verdict_age_ms": owner
                        .host_bypass_verdict_age()
                        .map(|age| age.as_millis() as u64),
                    // 这一票到底算不算数（确定 + 新鲜）。渲染门的判据就是它。
                    "host_bypass_vote": owner.fresh_host_bypass().map(|verdict| verdict.as_str()),
                    // 【我挂在哪条链上】`track-fx` / `take-fx` / `unbound`。旁通读的
                    // 是哪条链、清单枚举了哪一段、能力位关掉了哪几项，全由它派生 ——
                    // "旁通了还在渲染"的第一格要排除的就是它是不是被读错了位置。
                    "host_placement": owner.host_placement().as_str(),
                })
            })
            .collect::<Vec<_>>();
        let placements = self
            .renderer_owners()
            .into_iter()
            .map(|owner| owner.host_placement().as_str())
            .collect::<Vec<_>>();
        serde_json::json!({
            "renderers": renderers,
            // 每个活实例的放置位置（顺序与 `renderers` 无关，仅供总览）。
            "placements": placements,
            // 【文档级的门此刻是怎么算出来的】渲染门只看这一格。它等于
            // `combine(renderers[].host_bypass_vote)`，把两个数组放在一起就能一眼看出
            // "为什么门没关上"：是没人投票，还是有人投了 `active`。
            "bypass_verdict": self.host_bypass().as_str(),
            // 【装配的两份输入各写到第几代】见 `assembly_consumed`：两者不同步时
            // `unbound` 会先被推迟一轮，而这一对数字就是"谁还没跟上"的答案。
            "assembly_inputs": {
                "ui_tracks_stamp": self.ui_tracks_stamp.load(Ordering::Acquire),
                "region_rows_stamp": self.region_rows_stamp.load(Ordering::Acquire),
                "consumed": self.assembly_consumed.lock_or_recover().map(|(items, regions)| {
                    serde_json::json!({ "ui_tracks_stamp": items, "region_rows_stamp": regions })
                }),
                "deferred": self.assembly_deferred_since.lock_or_recover().is_some(),
                "unbound": self.assembly_unbound.lock_or_recover().len(),
            },
        })
    }

    /// 记一次"宿主侧发生了变化，渲染许可随之失效"。
    ///
    /// 【为什么必须共用一个 helper】`ready` 有三个置假点（`close` / `clear_renderers` /
    /// `revoke_snapshots`），而 `ensure_loaded` 靠这个时刻把"宿主正在收敛"与
    /// "宿主永不收敛"分开（见字段说明）。漏写任何一个，就把那一类变化误报成
    /// 用户级冲突 —— 正是重开工程后仍然出现的那条 `Conflict: host changed`：
    /// 宿主授权 PCM 走的是 `refresh_source_pcm` → `revoke_renderers()` →
    /// `revoke_snapshots()`，而它此前**没有**落这个时刻。
    fn note_content_change(&self) {
        *self.last_content_change.lock_or_recover() = Some(std::time::Instant::now());
    }

    /// 进程内单调毫秒。0 保留给"没有事件"。
    ///
    /// 【为什么不用 `Instant` + `Mutex`】这几个时刻要在**音频线程**上读
    /// （`request_settle_if_overdue` 由 `process` 顺带调用），而 `Mutex` 在实时线程上
    /// 是不可接受的。整数毫秒既无锁又足够（宽限期是秒级）。
    fn monotonic_ms() -> u64 {
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let epoch = EPOCH.get_or_init(std::time::Instant::now);
        // +1 保证"刚发生"也不等于 0（0 是哨兵）。
        (epoch.elapsed().as_millis() as u64).saturating_add(1)
    }

    /// 记下编辑会话的开始（`begin_editing`）。
    pub(crate) fn note_editing_started(&self) {
        self.editing_session.store(true, Ordering::Release);
        self.editing_since_ms
            .store(Self::monotonic_ms(), Ordering::Release);
    }

    /// 记下编辑会话的结束（`end_editing`）。
    pub(crate) fn note_editing_finished(&self) {
        self.editing_session.store(false, Ordering::Release);
        self.editing_since_ms.store(0, Ordering::Release);
    }

    /// 记下"模型变脏了"（首次才落时刻：后续脏不推迟截止期）。
    pub(crate) fn note_model_dirty(&self) {
        let now = Self::monotonic_ms();
        let _ =
            self.model_dirty_since_ms
                .compare_exchange(0, now, Ordering::AcqRel, Ordering::Relaxed);
    }

    /// 记下"模型已成功发布"。
    pub(crate) fn note_model_clean(&self) {
        self.model_dirty_since_ms.store(0, Ordering::Release);
    }

    /// 距模型变脏已经过去多久（`None` = 干净）。
    pub(crate) fn model_dirty_elapsed(&self) -> Option<std::time::Duration> {
        let since = self.model_dirty_since_ms.load(Ordering::Acquire);
        (since != 0).then(|| Self::elapsed_since(since))
    }

    /// 距编辑会话开始已经过去多久（`None` = 不在会话中）。
    pub(crate) fn editing_elapsed(&self) -> Option<std::time::Duration> {
        let since = self.editing_since_ms.load(Ordering::Acquire);
        (since != 0).then(|| Self::elapsed_since(since))
    }

    /// **T23：给"宿主不再回调"的三条死路一个插件侧出口。**
    ///
    /// 【为什么必须有它】`ready` 回到真只依赖四个入口：`endEditing`、
    /// `notifyModelUpdates`（→ flush）、`refresh_source_pcm`（要求 `!editing`）、
    /// ARA 存档恢复。**四个全在宿主手里。** 宿主一旦 `beginEditing` 之后再也不收口，
    /// 或者积压了模型变化却不再调 `notifyModelUpdates`，插件就永久停在未就绪 ——
    /// 用户唯一的出路是删掉插件再加。这不是"某个 bug"，是**没有出口**。
    ///
    /// 【为什么是"置位请求"而不是直接收口】桥的 `PluginRuntime` 由文档控制器独占、
    /// 不是 `Send`/`Sync`，从视图定时器直接调它是未定义行为。这里只置两个原子量，
    /// 由控制器在**下一次任何回调**的入口消费（`runtime_mut` 与 `notifyModelUpdates`）。
    ///
    /// 【为什么可以重复调用】纯原子读 + 幂等置位，音频线程也可以调。
    ///
    /// 【边界】宿主连一次回调都不再发时，这里也只能等到下一次回调。那一段的兜底是
    /// VST3 侧的 `getState` / `setActive`（宿主存盘与激活）—— 本函数不假装能覆盖它。
    /// 单调毫秒差的哨兵：`u64::MAX` 表示"起点被回拨到无穷远"（仅测试使用）。
    const BACKDATED: u64 = u64::MAX;

    /// 由起点算已过去多久；`BACKDATED` 视为远超任何宽限期。
    fn elapsed_since(since: u64) -> std::time::Duration {
        if since == Self::BACKDATED {
            // 比任何宽限期都大，但仍是有限值 —— 调用方可能对它做加法。
            return std::time::Duration::from_secs(86_400);
        }
        std::time::Duration::from_millis(Self::monotonic_ms().saturating_sub(since))
    }

    /// 发布一份由模型侧算出的输入可用性读数（见 [`InputReadiness`]）。
    pub(crate) fn note_readiness(&self, readiness: InputReadiness) {
        *self.readiness.lock_or_recover() = Some(readiness);
    }

    /// 当前读数（`None` = 还没算过）。
    pub(crate) fn readiness(&self) -> Option<InputReadiness> {
        *self.readiness.lock_or_recover()
    }

    /// 当前判据；`None` 表示"未知"，调用方按旧行为处理。
    pub(crate) fn input_verdict(&self) -> Option<InputVerdict> {
        self.readiness().map(|readiness| readiness.verdict)
    }

    /// **T6：把"卡住了"变成一份自我描述的读数。**
    ///
    /// 【为什么必须存在】这四类症状反复复发的真正原因不是某个 bug，而是**失败不可
    /// 归因**：十几条路径都能静默早退，而它们留下的现场几乎同形。这一行把每一条早退
    /// 都变成一个具名的数，下一轮不必再从位掩码与调用顺序反推。
    ///
    /// 只在 `verdict != Ready` 时打印，并由调用方限流。
    pub(crate) fn stuck_report(&self) -> String {
        let readiness = self.readiness();
        let scope_regions = self.sequence_regions.lock_or_recover().len();
        let assignments = self.renderer_owners().len();
        let ui_tracks = self.ui_tracks_guard().len();
        let region_items = self.region_items.lock_or_recover().len();
        let editing = self.editing_elapsed().map(|elapsed| elapsed.as_millis());
        // 【T11：图账本进这一行】没有它，"宿主没发图"与"我们拒了图"在这里完全同形 ——
        // 而那正是四类症状反复复发的共同形状：两条路径共用一句读数。
        let graph = self
            .graph_ledger
            .lock_or_recover()
            .as_ref()
            .map(|ledger| ledger.summary());
        // 【T-B0-3】读游标接缝读数：块级，不进音频线程（这里是诊断线程）。
        let seam = self
            .renderer_owners()
            .first()
            .map(|owner| owner.seam_summary())
            .unwrap_or_else(|| "seam=none".to_owned());
        format!(
            "doc={} {} ready={} editing={:?} dirty={:?} sequences={} renderers={} \
             ui_tracks={} region_items={} {} {}",
            self.id,
            readiness.map_or_else(|| "verdict=unknown".to_owned(), |r| r.summary()),
            self.ready.load(Ordering::Acquire),
            editing,
            self.model_dirty_elapsed()
                .map(|elapsed| elapsed.as_millis()),
            scope_regions,
            assignments,
            ui_tracks,
            region_items,
            graph.unwrap_or_else(|| "graph=unbound".to_owned()),
            seam,
        ) + "\n"
            + &self.diag_line()
    }

    /// 一行可复制的诊断读数：把"这条 clip 为什么没有音频"拆成五个可数的整数。
    ///
    /// 【为什么必须存在这一行】19 轮排查里没有拿到过一次真机日志 —— 用户不知道去哪
    /// 找，日志里也没有一个读数能把"哪一环断了"回答清楚。整条链是
    /// regions → items → authorized → pcm → clips_src，每一环都能用一个整数回答，
    /// 于是"重开工程后全部 Clip 都在等音频"不再需要猜：
    ///   * items=0 且 regions>0 ⇒ 宿主几何/清单没通（原生扩展那一环）；
    ///   * authorized=0 而 items>0 ⇒ 授权表没写（`authorized_takes` 的写者没跑）；
    ///   * authorized>0 而 pcm=0 ⇒ PCM 没读出来（源那一层）；
    ///   * pcm>0 而 clips_src=0 ⇒ **身份链断了**（媒体身份没挂到 clip 上）。
    ///
    /// 【为什么只在这些字段上】它们都是**计数**，不随宿主型号或工程内容变化，
    /// 因此可以直接对比两次运行。
    pub(crate) fn diag_line(&self) -> String {
        let regions = self.regions.lock_or_recover().len();
        let items = self.region_items.lock_or_recover().len();
        let authorized = self.authorized_takes.lock_or_recover().len();
        let pcm = self.edit_sources.lock_or_recover().len();
        let facts = self.media_facts.lock_or_recover().len();
        let (published, failed) = {
            let states = self.source_states.lock_or_recover();
            (
                states
                    .values()
                    .filter(|state| **state == SourceState::Published)
                    .count(),
                states
                    .values()
                    .filter(|state| **state == SourceState::Failed)
                    .count(),
            )
        };
        let (clips, clips_src) = {
            let timeline = self.timeline.lock_or_recover();
            let clips = timeline.as_ref().map_or(0, |timeline| timeline.clips.len());
            let with_source = timeline.as_ref().map_or(0, |timeline| {
                timeline
                    .clips
                    .iter()
                    .filter(|clip| clip.source_path.is_some())
                    .count()
            });
            (clips, with_source)
        };
        // 【T-S2-7】`dupes` 是**呈现层**的读数：上一轮 `present_host_inventory` 收敛掉了
        // 几条重复 clip。渲染时间线里永远只有一条，所以这个数此前完全不可观测 ——
        // 而"同一个 item 被画成两条 clip"正是用户报的那条症状。
        let dupes = self.presentation_dupes.load(Ordering::Acquire);
        format!(
            "[hfs] diag v1 ara{{regions={regions} items={items} authorized={authorized} \
pcm={pcm} facts={facts} published={published} failed={failed} clips={clips} \
clips_src={clips_src} dupes={dupes}}}"
        )
    }

    /// 诊断读数变化时打一行（同样的读数不重复）。
    ///
    /// 【为什么去重】它在每次收口都会跑（宿主每做一次编辑就一次）。不去重会把日志
    /// 刷成一条恒定的重复行，而那正是"没人看得下去"的原因。
    pub(crate) fn log_diag_if_changed(&self) {
        let line = self.diag_line();
        let mut last = self.last_diag.lock_or_recover();
        if last.as_deref() == Some(line.as_str()) {
            return;
        }
        *last = Some(line.clone());
        log::info!("{line}");
    }

    /// 测试专用：把两个截止期的起点回拨到"早已超期"。
    ///
    /// 【为什么需要】真实场景里"宿主卡了 30 秒"无法在单测中等待；把起点回拨等价，
    /// 而且不会引入 `sleep`（那会让整个套件变慢且不稳定）。
    #[cfg(test)]
    pub(crate) fn backdate_settle(&self) {
        for slot in [&self.editing_since_ms, &self.model_dirty_since_ms] {
            if slot.load(Ordering::Acquire) != 0 {
                slot.store(Self::BACKDATED, Ordering::Release);
            }
        }
    }

    pub fn request_settle_if_overdue(&self) {
        if self
            .editing_elapsed()
            .is_some_and(|elapsed| elapsed >= FORCED_EDIT_CLOSE_GRACE)
        {
            self.forced_edit_close.store(true, Ordering::Release);
        }
        if self
            .model_dirty_elapsed()
            .is_some_and(|elapsed| elapsed >= EDIT_SETTLE_GRACE)
        {
            self.forced_model_flush.store(true, Ordering::Release);
        }
    }

    /// 在旧内容可能被改变前立即撤销**渲染许可与代次**；不撤已发布快照本身。
    ///
    /// 名字是历史遗留：本函数**不会**清空 renderers 列表，只做失效标记；
    /// 真正清空列表的是 [`Self::close`] / [`Self::detach_owner`]。
    ///
    /// 【为什么不再清空快照】`clear_renderers()` 被一次宿主编辑触发**多次**
    /// （`ara/model.rs` 里 8 处调用）。清空 `current` 之后，`copy_block_at` 的读闭包
    /// 逐样本返回 `0.0`，而 bound 实例固定 `layered == false`，于是**每个音频块都是
    /// 纯零** —— 用户听到的就是"有概率完全没声音"。清空与后台合成还要再叠一层
    /// `prepare_job` 的版本重检丢弃（`extension.rs`），空窗最长可达一次神经合成的时长。
    ///
    /// 保留上一份快照永远好过静音：它对应上一次稳定状态，`publish` 会在新快照算完后
    /// **原子**替换 `current`，音频线程的 `enter()` 要么拿到旧的、要么拿到新的，
    /// 不会拿到空。真正该清空的是"这个实例从此不再渲染"——那由 [`Self::close`]
    /// 与 `ExtensionOwner::stop_editor` 负责，它们会调 [`Self::revoke_snapshots`]。
    ///
    /// 【为什么仍然按回落 `ready`】`ready` 回答的是"现在能不能重排并发布"，宿主编辑
    /// 期间答案确实是否；把它留在 true 会让"还没重排完"被误判成"已经就绪"。
    /// 它与"已发布的快照还在不在"是两个问题，本轮只改后者。
    pub fn clear_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
        self.revision.fetch_add(1, Ordering::AcqRel);
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.ready.store(false, Ordering::Release);
        // 记下"宿主刚刚改过内容"：`ensure_loaded` 据此把紧随其后的未就绪窗口
        // 当成"等宿主收口"而不是"宿主冲突"（见字段说明）。
        self.note_content_change();
        for owner in self.renderer_owners() {
            owner.cancel_preparation();
            // 让 revision 门（`processed_ready` / `raw_tape_ready`）与输出代次失效，
            // 但不动 `current` —— 上一份快照继续可听。
            owner
                .snapshots
                .iter()
                .for_each(|snapshot| snapshot.invalidate_generation());
            owner.raw_tape.invalidate_generation();
        }
    }

    /// 授权开关只撤销播放许可，不改变宿主内容/布局的乐观并发版本。
    pub fn revoke_renderers(&self) {
        let _transaction = self.transaction.lock_or_recover();
        self.revoke_snapshots();
    }

    /// 渲染输入失效：推进 `render_epoch` 并**通知本文档全部 owner** 重排。
    ///
    /// 【为什么必须是文档级、必须遍历全部 owner】`render_epoch` 是本文档的共享计数器，
    /// 但"谁发现了变化"与"谁持有快照"在多实例下**不是同一个实例**。REAPER 按轨道管理
    /// ARA 实例：持有 webview 与 20 ms 定时器的是编辑/视图实例（`role == 2`），真正被
    /// 音频回调读取快照的是播放渲染实例（`role == 1`）。若发现变化的一侧只
    /// `self.prepare()`，编辑实例会因 `prepare()` 首行的 `!renders_playback()` 直接返回
    /// （`extension.rs`），播放实例则**从未被通知** —— 于是波形更新、声音不变，直到
    /// 用户做一次编辑（编辑路径恰好遍历了全部 owner）才"碰巧"恢复。
    ///
    /// 【为什么不在这里清快照】失效只需要让 `PreparedVersion` 不再匹配（它含 `epoch`），
    /// `prepare_job` 便不会早退；清快照会制造一段静音空窗，而重排是异步的。
    ///
    /// 【为什么**不**取 `transaction`】这是一次通知，不是与文档状态必须原子提交的变更：
    /// 只推进一个原子计数并枚举 owner。`prepare_job` 会在自己的事务里重新校验全部版本，
    /// 因此这里取事务没有正确性收益，却会在这个 20 ms 定时器路径上多抢一次文档锁，
    /// 并让调用方无法"持事务阻止后台渲染"（测试与离线 setup 都依赖这个手法）。
    /// 锁序也因此更简单：本函数只取 `renderers`，不参与"事务 → renderers"的嵌套。
    /// 只推进渲染代次，**不**通知 owner（见 `refresh_ui_inventory` 的说明）。
    ///
    /// 【为什么需要它】纯几何变化（宿主移动 / 裁剪 item）会让已发布的快照不再对应
    /// 当前时间线，但重排必须由定时器驱动 —— 在清单刷新的调用栈上直接 `prepare()`
    /// 会自锁（那条栈可能就在会话 actor 上，而 `prepare` 要投递回同一个 actor）。
    pub(crate) fn note_render_geometry_changed(&self) {
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
    }

    pub fn invalidate_render_input(&self) {
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.note_render_epoch_pressure();
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .filter(|owner| !owner.is_closed())
            .collect::<Vec<_>>();
        for owner in owners {
            // 【先刷新旁通缓存，再通知重排】重排是否真的发生取决于各 owner 自己缓存里的
            // 旁通读数；缓存还是旧的时，刚被解除旁通的那个实例会以"我被旁通了"为由
            // 直接返回，用户解除旁通后仍然没有声音。
            owner.refresh_host_bypass_cache();
            owner.prepare();
        }
    }

    /// **这份工程**此刻还会不会出声（三态，见 `FxBypass`）。
    ///
    /// 【为什么是"派生"而不是"存一格"】旁通是**每个 FX** 的属性，而一个 ARA 文档里可以
    /// 有多条轨道、每轨一个实例（`ModelHandle` 每文档一份，所有实例绑到同一个文档）。
    /// 把结论写进共享格子，等于让"轨 1 被旁通"与"轨 2 没被旁通"互相覆盖：读数每轮都在
    /// 两个值之间跳，而每次跳都让渲染输入失效 ⇒ 每个实例每 100 ms 重算一整次神经合成
    /// （用户看到的就是"无限渲染"），而且轨 2 会跟着轨 1 一起静音。
    ///
    /// 所以这里**只读各实例自己的读数**，用 `fx_state::combine` 合成：只有全部**有读数**
    /// 的实例都确定被旁通，才算这份工程不会出声。
    ///
    /// 【为什么读数的"有没有"也算一票】上一版把"从没采到过"的实例也算成否决票
    /// （`Unknown` ⇒ 照常渲染）。而 REAPER 的播放实例是宿主按 item 现建的，它们既没有
    /// 视图定时器，也不会在用户按下旁通时被模型收口扫到 —— 于是一份**所有实例都被
    /// 旁通**的工程里，只要有一个刚建好的播放实例还没被采样，结论就永远是 `Unknown`，
    /// 门永远开着。用户看到的就是"旁通了还在渲染"。现在"没有读数"= 弃权
    /// （见 [`crate::render::extension::ExtensionOwner::fresh_host_bypass`]），
    /// 而不是否决。
    pub(crate) fn host_bypass(&self) -> crate::host::reaper::FxBypass {
        self.bypass_sample().0
    }

    /// 结论 + 每一票的来源：`(writer, 是否播放实例, 最后一次确定读数, 读数年龄 ms)`。
    ///
    /// 【为什么要把票源带出来】"旁通了还在渲染"只有这张表能回答：到底是**没人投出旁通**
    /// （实例没被采样）、**有人投了反票**（读数是 `active`）、还是**票过期了**（年龄超过
    /// `BYPASS_VERDICT_TTL`）。三种成因的治疗完全不同，而在旧日志里它们长得一模一样：
    /// 同一个限流键、同一行 `[throttled]`，什么都看不出来。
    ///
    /// 【表格是给日志/诊断包的，不是判据】判据只有 [`Self::host_bypass`]。
    pub(crate) fn bypass_sample(
        &self,
    ) -> (
        crate::host::reaper::FxBypass,
        Vec<(u64, bool, crate::host::reaper::FxBypass, Option<u64>)>,
    ) {
        let owners = self.renderer_owners();
        let mut table = Vec::with_capacity(owners.len());
        let mut votes = Vec::with_capacity(owners.len());
        for owner in &owners {
            let verdict = owner.host_bypass();
            let age = owner
                .host_bypass_verdict_age()
                .map(|age| age.as_millis() as u64);
            table.push((owner.writer_id(), owner.renders_playback(), verdict, age));
            if let Some(fresh) = owner.fresh_host_bypass() {
                votes.push(fresh);
            }
        }
        // 【为什么不分"播放实例 / 编辑实例"】旁通是**这个 FX** 的属性，而一个 FX 的编辑
        // 实例与播放实例读的是同一条 TrackFX 记录。按角色筛掉编辑实例，等于在"播放实例
        // 还没被采样"时把唯一能说话的那一票丢掉 —— 那正是上一版的门失效的地方。
        // 多轨工程里"轨 1 旁通、轨 2 正常"仍然成立：轨 2 的实例会说 `active`。
        (crate::host::reaper::combine_fx_bypass(&votes), table)
    }

    /// 会话心跳：把**最该重采**的那一个实例采一次。
    ///
    /// 【为什么必须有这条通道】三个采样点里有两个只覆盖**可见**实例：视图定时器
    /// （`poll_host_bypass`，要开窗口）与模型收口（`refresh_host_state_for_model`，
    /// 要宿主改模型）。REAPER 的播放实例两样都没有 —— 用户按下旁通之后，它的缓存可以
    /// 一直停在"上一次采样时的 active"，而渲染门正是拿它做的判据（用户报的"旁通了
    /// 还在渲染"）。
    ///
    /// 【为什么一轮只采一个】实例数由宿主决定，而每次采样都是一次宿主线程往返（最坏
    /// 250 ms）。一轮一个 + 750 ms 的间隔，几个实例就在几秒内轮一遍，而音频线程与 UI
    /// 线程都不会被它拖住。轮不到的那些由有效期兜底：读数过期 = 弃权（照常渲染），
    /// 绝不因为"排队没轮到"而把一条轨静音。
    pub(crate) fn service_host_bypass_samples(&self) {
        /// 有读数的实例：按 750 ms 保鲜（有效期 5 s，留足六次机会）。
        const INTERVAL_MS: u64 = 750;
        /// 从没读到过读数的实例：退到 3 s 一次。
        ///
        /// 【为什么要退】这类实例要么是宿主根本没给这一族 API（每次都白花一次宿主
        /// 往返），要么这条轨上找不到本 FX。而心跳跑在会话 actor 上 —— 每 750 ms 付
        /// 一次最坏 250 ms 的阻塞，就足以让 UI 命令排队。真正需要"随时知道"的是**已经
        /// 读到过**的实例（旁通的跃迁）；从没读到过的那些，慢一点没有代价。
        const NEVER_READ_MS: u64 = 3_000;
        let owners = self.renderer_owners();
        let Some(owner) = owners
            .iter()
            .filter(|owner| owner.renders_playback())
            .max_by_key(|owner| owner.host_bypass_sample_age_ms())
        else {
            return;
        };
        let interval = if owner.host_bypass_verdict_age().is_none() {
            NEVER_READ_MS
        } else {
            INTERVAL_MS
        };
        if owner.host_bypass_sample_age_ms() < interval {
            return;
        }
        // 【为什么用 `sample_host_bypass` 而不是 `refresh_host_bypass_cache`】旁通跃迁的
        // 两个动作（进入旁通丢队列、离开旁通重排）必须与采样在同一处发生，否则"解除
        // 旁通之后没有声音"会以另一种形式回来。
        owner.sample_host_bypass();
    }

    /// 一秒钟内失效太多次 ⇒ 报一次（"渲染停不下来"的唯一可查痕迹）。
    ///
    /// 【阈值为什么是 8】正常编辑一轮最多两三次（源物化 / 事实变化 / 旁通跃迁）；
    /// 8 次/秒已经说明有个判据把"没变"读成了"变了"，而每一次都意味着一次完整重算。
    fn note_render_epoch_pressure(&self) {
        const WINDOW: std::time::Duration = std::time::Duration::from_secs(1);
        const THRESHOLD: u32 = 8;
        let mut window = self.render_epoch_window.lock_or_recover();
        let now = std::time::Instant::now();
        let (since, count) = &mut *window;
        if since.is_none_or(|at| now.duration_since(at) >= WINDOW) {
            *since = Some(now);
            *count = 0;
        }
        *count += 1;
        if *count == THRESHOLD {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.render.epoch-storm",
                format_args!(
                    "[render] the render input was invalidated {THRESHOLD} times within one second;                      something re-invalidates it every tick, so the published snapshot can never be                      reused — this is what a render that never stops looks like from the inside"
                ),
            );
        }
    }

    fn revoke_snapshots(&self) {
        self.render_epoch.fetch_add(1, Ordering::AcqRel);
        self.ready.store(false, Ordering::Release);
        // 【T1】这里同样是一次"宿主侧变化"（源权限撤销 / 源内容变了），必须与
        // `clear_renderers()` 一样留下观察时刻。漏掉它会让 `ensure_loaded` 把
        // `refresh_source_pcm` 造成的未就绪窗口判成真实冲突。
        self.note_content_change();
        let owners = self
            .renderers
            .lock_or_recover()
            .iter()
            .filter_map(|lease| lease.owner.upgrade())
            .collect::<Vec<_>>();
        for owner in owners {
            owner.cancel_preparation();
            // 【为什么这里是**硬清**而不是只推进代次】`revoke_renderers()` 的语义是
            // "宿主撤销了源权限 / 源内容变了"：继续播旧快照等于播一份已经失效的
            // 渲染结果。真正被频繁调用、且**不该**造成静音的是 `clear_renderers()`
            // （一次宿主编辑触发多次）—— 那一条已经改成只 `invalidate_generation()`。
            owner.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
    }

    /// 是否至少有一个实例正在实时播放（音频线程在推进）。
    ///
    /// 【用途】内容更新要不要**立刻**撤渲染许可，取决于这一刻有没有人在听。播放中撤掉
    /// 快照等于给用户一个空洞；而"旧快照"对应的是上一版内容，与 `StaleProcessed`
    /// 是同一条既有纪律。
    pub(crate) fn realtime_playing(&self) -> bool {
        self.renderer_owners()
            .iter()
            .any(|owner| owner.realtime_flow_active())
    }

    /// 记录一次"本插件拒绝了 region 创建"。
    ///
    /// 【为什么必须留下真因】`create_playback_region` 失败对宿主只表现为一个 null
    /// region，插件侧唯一的痕迹是一条 error 日志。把它接进状态链之后，
    /// `decorate_host_audio` 能把 `awaiting_regions` 换成 `region_rejected`，
    /// 用户看到的是"插件拒收了"而不是"还在等"。
    pub fn note_region_rejection(&self, reason: impl Into<String>) {
        const REGION_REJECTION_LIMIT: usize = 32;
        let mut rejections = self.region_rejections.lock_or_recover();
        if rejections.len() == REGION_REJECTION_LIMIT {
            rejections.pop_front();
        }
        rejections.push_back(reason.into());
    }

    /// 最近一条 region 拒绝原因（没有则为 `None`）。
    pub fn last_region_rejection(&self) -> Option<String> {
        self.region_rejections.lock_or_recover().back().cloned()
    }

    /// 只借出仍存活的文档，拒绝已销毁或非本插件的控制器。
    pub fn lookup(key: usize) -> Option<Arc<Self>> {
        controllers()
            .lock_or_recover()
            .get(&key)
            .and_then(Weak::upgrade)
            .filter(|session| session.alive.load(Ordering::Acquire))
    }

    /// 当前进程里仍存活的控制器键（诊断；有界）。
    ///
    /// 【为什么需要它】`bind` 拿到一个不在表里的 controller ref 时，必须能一眼看出
    /// "是表里根本没有新文档"还是"新文档用了另一个 ref"。只返回前若干条，避免
    /// 多工程/异常宿主把日志撑爆。
    pub(crate) fn live_controller_keys() -> Vec<usize> {
        const MAX_KEYS: usize = 16;
        controllers()
            .lock_or_recover()
            .iter()
            .filter(|(_, weak)| {
                weak.upgrade()
                    .is_some_and(|session| session.alive.load(Ordering::Acquire))
            })
            .map(|(key, _)| *key)
            .take(MAX_KEYS)
            .collect()
    }
}

#[cfg(test)]
mod input_verdict_tests {
    use super::InputVerdict;

    /// 渲染侧有 PCM、宿主却没授权 ARA 样本 —— REAPER-Linux 的实际形状。
    ///
    /// 【为什么这条最要紧】它就是"两份真相"的那条缝：`authorized=0 published=0`
    /// 而 `pcm=1`。旧判据在这种输入下给出 `AwaitingSamples`，编辑器于是永远认为
    /// 素材没到；新判据必须以能出声的 PCM 为准。
    #[test]
    fn usable_pcm_outranks_a_missing_sample_access_grant() {
        assert_eq!(
            InputVerdict::decide(1, 0, 0, 1),
            InputVerdict::Ready,
            "a published PCM must count as usable even when the host never granted sample access"
        );
    }

    /// 什么都没有的时候仍然是"等 region"或"等样本"，不能被新判据吞掉。
    #[test]
    fn an_empty_document_still_reports_what_it_is_waiting_for() {
        assert_eq!(
            InputVerdict::decide(0, 0, 0, 0),
            InputVerdict::AwaitingRegions
        );
        assert_eq!(
            InputVerdict::decide(2, 0, 0, 0),
            InputVerdict::AwaitingSamples
        );
        assert_eq!(InputVerdict::decide(2, 1, 0, 0), InputVerdict::Ready);
    }

    /// 有 PCM 但部分源失败：可用，但必须报成 `Degraded` 而不是装作一切正常。
    #[test]
    fn a_partial_failure_degrades_a_usable_document() {
        assert_eq!(InputVerdict::decide(3, 2, 1, 2), InputVerdict::Degraded);
        assert_eq!(InputVerdict::decide(1, 0, 1, 0), InputVerdict::Degraded);
    }
}

#[cfg(test)]
mod presentation_revision_tests {
    use super::*;

    fn pcm(version: u64) -> crate::render::source::SourcePcm {
        crate::render::source::SourcePcm {
            covered: Vec::new(),
            sample_rate: 44_100,
            planes: vec![vec![0.25, -0.25, 0.5, -0.5]],
            version,
            _reservation: None,
        }
    }

    /// **回归：同代重读也必须推进呈现代次。**
    ///
    /// 【为什么这条是用户报障的原形】宿主每次重新授予样本访问（起播、前台恢复、
    /// 重同步）都会让插件把源重读一遍；读回来的是同一代内容，于是
    /// `editor_audio_revision` 刻意**不**推进（否则每次播放开始都会排一轮输出完全
    /// 相同的重渲染）。但界面要的恰恰是这一次：源从 `Revoked` 回到 `Published`，
    /// 已经画成占位的 clip 应当立刻拿到波形。
    ///
    /// 旧实现两者都不推进 ⇒ 日志里就是 "re-read the same content generation 0"
    /// 之后仍然 `media ready=3 pending=2`，直到用户重新加载工程。
    #[test]
    fn a_same_generation_reread_still_refreshes_the_presentation() {
        let document = DocumentSession::new(50_001);
        let cancelled = AtomicBool::new(false);
        let id = "ara://source".to_owned();
        document.publish_source_read_result(id.clone(), Ok(pcm(0)), &cancelled);
        let after_first = document.presentation_revision.load(Ordering::Acquire);
        assert!(after_first > 0, "首次发布必须推进呈现代次");
        document.publish_source_read_result(id.clone(), Ok(pcm(0)), &cancelled);
        assert!(
            document.presentation_revision.load(Ordering::Acquire) > after_first,
            "同代重读不得让界面停在旧投影上"
        );
        document.close();
    }

    /// **失败同样要上屏。**
    ///
    /// 源读不出来时界面必须立刻报 `source_read_failed`，而不是等下一次几何刷新 ——
    /// 那段等待正是"所有 Clip 都在等宿主音频"的一部分。
    #[test]
    fn a_failed_read_also_refreshes_the_presentation() {
        let document = DocumentSession::new(50_002);
        let cancelled = AtomicBool::new(false);
        document.publish_source_read_result(
            "ara://broken".to_owned(),
            Err("sample access denied".to_owned()),
            &cancelled,
        );
        assert!(
            document.presentation_revision.load(Ordering::Acquire) > 0,
            "读取失败必须推进呈现代次，界面才能报出真因"
        );
        document.close();
    }

    /// **装配发布 = 编辑器该看的内容变了。**
    ///
    /// 这里钉的是"代次只在 `clip_ids` 变化时才推进"会漏掉的那一类：clip 集合没变、
    /// 媒体身份变了（占位 → 有源）。它正是粘贴后卡在"正在等待宿主完成音频分配"的形状。
    fn document_with_one_empty_track(project_sec: f64) -> Arc<DocumentSession> {
        let document = DocumentSession::new(50_003);
        *document.timeline.lock_or_recover() = Some(
            serde_json::from_value(serde_json::json!({
                "tracks": [{"id": "track", "name": "t", "order": 0}],
                "clips": [],
                "bpm": 120.0,
                "project_sec": project_sec,
            }))
            .unwrap(),
        );
        document
    }

    #[test]
    fn publishing_an_assembly_always_advances_the_presentation() {
        let document = document_with_one_empty_track(1.0);
        // 没有 ARA 模型 ⇒ 装配器按既有纪律不动已发布的时间线，本函数直接返回。
        let before = document.presentation_revision.load(Ordering::Acquire);
        document.flush_pending_assembly();
        assert_eq!(
            document.presentation_revision.load(Ordering::Acquire),
            before,
            "没有脏标记时不得白跑一次呈现"
        );
        // 模型交付过之后，同样的清单刷新就是一次真正的发布。
        document.ara_model_seen.store(true, Ordering::Release);
        document.note_assembly_input_changed();
        document.flush_pending_assembly();
        assert!(
            document.presentation_revision.load(Ordering::Acquire) > before,
            "装配发布必须推进呈现代次（clip 集合没变、媒体身份变了的情形也要推进）"
        );
    }

    /// 【回归】连续为空**跨过预算**才说一次；说过之后不再重复。
    ///
    /// 空轮是常态（拖动 clip 时每几百毫秒一次），所以一个都不该留痕迹；但持续为空
    /// 不是常态，10 s 之后值得让用户看见一条可读的话 —— 而且只看见一次。
    #[test]
    fn a_long_empty_streak_surfaces_once() {
        let document = DocumentSession::new(50_004);
        let reports = (0..EMPTY_ROUNDS_MAX)
            .map(|_| document.note_empty_round())
            .collect::<Vec<_>>();
        assert_eq!(
            reports.iter().filter(|report| report.is_some()).count(),
            1,
            "整段空轮里只该有**一次**播报"
        );
        let streak = reports
            .last()
            .unwrap()
            .expect("最后一轮就是跨过预算的那一轮");
        assert_eq!(streak.rounds, EMPTY_ROUNDS_MAX);
        assert!(
            document.note_empty_round().is_none(),
            "说过一次之后不许刷屏"
        );
        document.close();
    }

    /// 【不回归】一旦发布了内容，连续为空的账就一笔勾销（下一次重新从零开始）。
    #[test]
    fn a_published_snapshot_clears_the_empty_streak() {
        let document = DocumentSession::new(50_005);
        for _ in 0..10 {
            assert!(document.note_empty_round().is_none());
        }
        document.clear_empty_rounds();
        assert_eq!(document.empty_rounds.load(Ordering::Acquire), 0);
        assert_eq!(document.empty_since_ms.load(Ordering::Acquire), 0);
        // 清零之后重新开始计：单轮空不再是"跨过预算"。
        assert!(document.note_empty_round().is_none());
        document.close();
    }
}
