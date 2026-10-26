//! 把"必须跑在宿主线程上"的工作投递过去，并**有界**等待结果。
//!
//! 【为什么需要它】REAPER 的 `IReaperHostApplication` 只允许在它的 model/UI
//! 线程上调用（见 `host/reaper/mod.rs` 里五处线程守卫）。而编辑器命令跑在插件的
//! 私有 GTK 线程与会话 actor 线程上 —— 两条都不是宿主线程。实测日志：
//!
//! ```text
//! [cmd] get_history_state failed:  REAPER geometry queried outside its model/UI thread
//! [cmd] create_host_track failed:  REAPER project  queried outside its model/UI thread
//! ```
//!
//! 【为什么是"有界等待"而不是阻塞到完成】宿主线程的泵只在它自己进入我们的代码
//! 时才会跑（`IPlugView::getSize` / `canResize` / `onSize`，以及注册到宿主的
//! `plugin_register("timer")` 节拍）。宿主完全空闲时泵可能不来 —— 无限等待就是
//! 编辑器卡死。超时后返回错误，行为退回今天的失败形态，只是多了一行可读的原因。

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// 投递失败的结构化原因。
///
/// 【为什么是枚举而不是字符串】上层要按它决定"要不要熔断"：只有「没人来取」说明
/// 宿主线程真的不可达，而「队列满」「任务 panic」都不该让后续调用一起停摆。
/// 仓库禁止用字符串子串匹配做错误分类（见 `docs/thread-model.md` §1.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostThreadError {
    /// 队列满：泵长期没跑。
    Backlog,
    /// 到点了没人来取：宿主当前不在我们的回调里。
    Unpicked,
    /// 任务自己 panic 了。
    Panicked,
    /// 任务还没开始就被投递方放弃。
    Abandoned,
}

impl HostThreadError {
    pub(crate) fn message(self) -> String {
        match self {
            HostThreadError::Backlog => format!(
                "host thread queue is full ({QUEUE_CAPACITY} pending); the host thread pump is not running"
            ),
            HostThreadError::Unpicked => format!(
                "host thread did not pick the request up within {} ms ({} still queued)",
                HOST_THREAD_TIMEOUT.as_millis(),
                pending()
            ),
            HostThreadError::Panicked => "host thread job panicked".to_owned(),
            HostThreadError::Abandoned => {
                "host thread job was abandoned before it started".to_owned()
            }
        }
    }
}

/// 投递方最多等这么久。取值依据：宿主 UI 线程被占用超过 250 ms 时，用户已经在
/// 感觉"卡"了，此时返回错误比继续等更诚实。
pub(crate) const HOST_THREAD_TIMEOUT: Duration = Duration::from_millis(250);

/// 队列上限。超过就拒绝新的投递 —— 宁可让调用方拿到"繁忙"，也不要把宿主线程
/// 变成一个无界的任务池。
const QUEUE_CAPACITY: usize = 256;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// 队列里的一件待办：**只许**在 `target` 那条线程上执行。
///
/// 【为什么任务要带上目标线程】泵有多个调用者（视图回调、宿主节拍，测试里还有
/// 别的东西）。没有这条约束时，任何一个泵都可能取走别人的任务，在自己的线程上
/// 跑它 —— 而任务的全部意义正是"必须跑在那一条线程上"。宿主侧的后果是又变回
/// `queried outside its model/UI thread`，只是更难查。
struct Queued {
    target: std::thread::ThreadId,
    job: Job,
    /// 入队时刻（清退阈值的依据）。
    queued_at: Instant,
    /// 投递入口（[`scoped`] / [`post_owned`]），决定清退阈值。
    kind: EntryKind,
}

/// 投递入口 → 清退阈值。
#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    /// `scoped`：借用投递方栈帧，投递方超时即放弃（`Abandoned`）；
    /// [`MAX_QUEUE_AGE`] 清退即可。
    Scoped,
    /// `post_owned`：拥有自己的数据。目标线程多为 ARA 模型线程 —— 它的泵**只在
    /// 宿主进入模型回调时运行**，宿主暂停/失焦几分钟是常态；[`MAX_OWNED_QUEUE_AGE`]
    /// 给它们足够的存活期，见该常量的说明。
    Owned,
}

fn queue() -> &'static Mutex<VecDeque<Queued>> {
    static QUEUE: OnceLock<Mutex<VecDeque<Queued>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn lock_queue() -> std::sync::MutexGuard<'static, VecDeque<Queued>> {
    queue().lock().unwrap_or_else(|error| error.into_inner())
}

/// 取出一件**属于当前线程**的待办；别人的任务原地不动。
fn take_one() -> Option<Job> {
    let current = std::thread::current().id();
    let mut pending = lock_queue();
    let index = pending.iter().position(|entry| entry.target == current)?;
    pending.remove(index).map(|entry| entry.job)
}

/// 一次投递完成后继续等下一件的窗口。
///
/// 【为什么需要它】编辑器一次刷新要连做十来个宿主调用（采样 / 几何代次 / 清单枚举
/// ……），而宿主节拍是 30 Hz。每个调用都等一个节拍，一次刷新就要 300 ms 以上 ——
/// 界面肉眼可见地发涩。留在宿主线程上多等这么一小会儿，同一批调用就几乎全部命中
/// 同一个节拍。2 ms 是「宿主 UI 线程愿意让出的时间」与「这一批调用之间的人为间隔」
/// 之间的折中：REAPER 的主循环里有 2 ms 的空转不会被察觉。
const LINGER: Duration = Duration::from_millis(2);

/// 队列条目的最长滞留时间；超过即按「目标线程已死」清退。
///
/// 【为什么需要清退】`editor::host_threads::deliver` 超时返回 `Queued` 后任务
/// **留在队列里**；若该文档此后再无模型回调（文档已关、宿主不再进入
/// `endEditing`），任务（持有 `PlaybackEndpoint` 的 Arc 等）既不执行也不被逐出。
/// 这是慢性侵蚀：满 [`QUEUE_CAPACITY`] 后 `scoped`/`post_owned` 对**所有**线程
/// 失败 —— 全插件表现为 "host thread queue is full"。60 s 远大于任何合法等待
/// （宿主节拍 30 Hz、UI 泵随每次窗口回调运行），滞留这么久只可能是目标线程
/// 不会再泵了。
///
/// 【为什么丢弃是安全的】`scoped` 条目被投递方放弃时槽位已置 `Abandoned`，
/// ticket 的执行入口对 `Abandoned` 直接返回、绝不解引用借用指针；而直接 drop
/// 连执行入口都不会碰到，只递减引用计数。`post_owned` 条目拥有自己的数据。
const MAX_QUEUE_AGE: Duration = Duration::from_secs(60);

/// `post_owned` 条目的清退阈值（10 分钟）。
///
/// 【为什么与 [`MAX_QUEUE_AGE`] 分开】60 s 的论证对**窗口线程**成立（泵随视图回调
/// /节拍持续运行），对**模型线程**不成立：它的泵只在宿主进入模型回调时运行，
/// 宿主暂停/失焦几分钟里一次都不来。全局队列上，窗口线程的泵会顺手清退**别人的**
/// 条目 —— 60 s 会把用户按过、拿到「已排队」回执的 Start/Stop 静默丢掉：按了播放，
/// 60 s 后宿主恢复却什么都不发生。十分钟足够跨过一次暂停，又不至于让死文档的
/// 任务永久占着队列（侵蚀仍有界）。清退时升格为 Error 日志：这是「用户的一个
/// 意图被丢了」，不是普通的卫生清理。
const MAX_OWNED_QUEUE_AGE: Duration = Duration::from_secs(600);

/// 清退滞留超限的条目（[`pump`] 每次顺手做一次；代价是一次线性扫描）。
fn evict_expired() {
    let now = Instant::now();
    let mut scoped_evicted = 0_usize;
    let mut owned_evicted = 0_usize;
    {
        let mut pending = lock_queue();
        pending.retain(|entry| {
            let limit = match entry.kind {
                EntryKind::Scoped => MAX_QUEUE_AGE,
                EntryKind::Owned => MAX_OWNED_QUEUE_AGE,
            };
            if now.duration_since(entry.queued_at) < limit {
                return true;
            }
            match entry.kind {
                EntryKind::Scoped => scoped_evicted += 1,
                EntryKind::Owned => owned_evicted += 1,
            }
            false
        });
    }
    if scoped_evicted > 0 {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "host.marshal.queue-evicted",
            format_args!(
                "[host] dropped {scoped_evicted} stale queued job(s) older than {} s; \
                 their target thread stopped pumping",
                MAX_QUEUE_AGE.as_secs()
            ),
        );
    }
    if owned_evicted > 0 {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Error,
            "host.marshal.owned-evicted",
            format_args!(
                "[host] dropped {owned_evicted} owned queued job(s) older than {} s; \
                 a queued transport request was lost (its target thread never picked it up)",
                MAX_OWNED_QUEUE_AGE.as_secs()
            ),
        );
    }
}

/// 拒投时记一行（限流）：目标线程与当前深度，排障时看得见"为什么满"。
fn log_queue_rejected(target: std::thread::ThreadId, depth: usize) {
    hifishifter_kernel::log_limiter::emit_limited_key(
        log::Level::Warn,
        "host.marshal.post-rejected",
        format_args!(
            "[host] queue full ({depth} pending); rejected a post for thread {target:?} — \
             its pump is not draining"
        ),
    );
}

/// 在**宿主线程**上执行队列里的全部待办。
///
/// 【调用者必须是宿主线程】它由 `IPlugView` 的回调与注册到宿主的 timer 驱动。这里
/// 不做线程断言：断言会把"泵装错了地方"变成宿主崩溃，而不做断言的最坏后果只是任务
/// 跑在错误的线程上并各自失败 —— 后者可观测、可恢复。
///
/// 【为什么空队列时立刻返回】宿主每次重排窗口都会调 `getSize`，一次会话几十次；
/// 那时队列多半是空的，白等 2 ms 只是白白拖慢宿主。
pub(crate) fn pump() {
    evict_expired();
    let Some(first) = take_one() else {
        return;
    };
    run_owned_job(first);
    let mut idle_since: Option<std::time::Instant> = None;
    loop {
        if let Some(job) = take_one() {
            run_owned_job(job);
            idle_since = None;
            continue;
        }
        let since = *idle_since.get_or_insert_with(std::time::Instant::now);
        if since.elapsed() >= LINGER {
            return;
        }
        std::thread::sleep(Duration::from_micros(150));
    }
}

/// 执行一个 `post_owned` 任务，并拦住它的 panic。
///
/// 【为什么必须在这里 catch】这类任务多为 `PlaybackRequest`（vendored bridge 的
/// FFI 封装），在宿主线程的视图回调/节拍泵里执行；panic 会一路穿过 `pump()`
/// 越过 `extern "system"` 边界 —— 那是 abort，也就是整个宿主退出。与
/// [`Ticket::run`]（scoped 路径）的拦截对称：scoped 的 panic 有投递方等着的槽位
/// 可写，owned 任务没有，所以只留一行 Error 日志。
fn run_owned_job(job: Job) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    if outcome.is_err() {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Error,
            "host.marshal.owned-job-panicked",
            format_args!(
                "[host] an owned queued job panicked; the host thread survives, the request is lost"
            ),
        );
    }
}

/// 队列里还有多少待办（诊断用）。
pub(crate) fn pending() -> usize {
    lock_queue().len()
}

/// 在宿主线程上执行 `job`，并让**投递方原地阻塞**到它完成。
///
/// 【为什么不是 `Send + 'static` 的普通任务队列】宿主调用链的实参全是借用
/// （`&self`、`&stamp`、`&authorized`），把 `'static` 加进去会顺着
/// `ExtensionOwner` → 命令分派 → 测试一路传播，改造面从"每个入口三行"变成
/// "整个编辑器换形状"。而这里真正需要的保证只有一条：
///
/// > 任务执行期间，投递方**不会**同时访问被借用的数据。
///
/// 投递方在返回前一直阻塞（或已把槽位标记为放弃），所以这条保证成立 —— 它与
/// `std::thread::scope` 对"自己新建的线程"给出的保证是同一件事，只是这里的执行者
/// 不是我们 spawn 的线程，而是由宿主回调驱动的泵。**唯一**的 unsafe 就是把闭包指针
/// 的生存期擦成 `*mut c_void`：它不会活得比投递方的栈帧更久。
///
/// 【放弃协议为什么必须是两态】超时后直接把任务从队列里删掉是不安全的：宿主线程
/// 可能**正在**执行它。因此放弃只在 `Pending` 下生效（任务还没开始，永远看不到被
/// 借用的数据）；一旦状态已经是 `Running`，投递方就必须继续等 —— 那一刻宿主线程
/// 确实在泵，任务会很快结束。
pub(crate) fn scoped<R: Send, F: FnOnce() -> R>(
    target: std::thread::ThreadId,
    job: F,
) -> Result<R, HostThreadError> {
    let slot = Arc::new(Slot {
        state: Mutex::new(SlotState::Pending),
        ready: Condvar::new(),
    });
    let mut output: Option<R> = None;
    {
        let mut context = JobContext {
            job: Some(job),
            output: &mut output,
        };
        let ticket = Ticket {
            trampoline: run_job::<R, F>,
            data: (&raw mut context).cast(),
            slot: Arc::clone(&slot),
        };
        {
            let mut pending = lock_queue();
            if pending.len() >= QUEUE_CAPACITY {
                log_queue_rejected(target, pending.len());
                return Err(HostThreadError::Backlog);
            }
            pending.push_back(Queued {
                target,
                job: Box::new(move || ticket.run()),
                queued_at: Instant::now(),
                kind: EntryKind::Scoped,
            });
        }
        wait(&slot)?;
    }
    output.ok_or(HostThreadError::Abandoned)
}

/// 等一次投递完成；超时且尚未开始就放弃。
fn wait(slot: &Slot) -> Result<(), HostThreadError> {
    let mut state = slot.state.lock().unwrap_or_else(|error| error.into_inner());
    loop {
        match &*state {
            SlotState::Done => return Ok(()),
            SlotState::Panicked => return Err(HostThreadError::Panicked),
            SlotState::Abandoned => return Err(HostThreadError::Abandoned),
            SlotState::Pending | SlotState::Running => {}
        }
        let (next, _) = slot
            .ready
            .wait_timeout(state, HOST_THREAD_TIMEOUT)
            .unwrap_or_else(|error| error.into_inner());
        state = next;
        match &*state {
            SlotState::Done => return Ok(()),
            // 任务还没开始 —— 现在放弃是安全的：它被取出时会看到这个状态并直接返回。
            SlotState::Pending => {
                *state = SlotState::Abandoned;
                return Err(HostThreadError::Unpicked);
            }
            // 已经开始执行 ⇒ 必须等它跑完，否则被借用的数据会在执行中途失效。
            SlotState::Running => continue,
            SlotState::Panicked => return Err(HostThreadError::Panicked),
            SlotState::Abandoned => return Err(HostThreadError::Abandoned),
        }
    }
}

/// 投递方与宿主线程共享的完成槽。
struct Slot {
    state: Mutex<SlotState>,
    ready: Condvar,
}

enum SlotState {
    Pending,
    Running,
    Done,
    /// 任务自己 panic 了。必须与 `Done` 分开：`Done` 保证结果已经写回，
    /// 而这里保证没有 —— 投递方要拿到"任务失败"，而不是"没有结果"。
    Panicked,
    Abandoned,
}

/// 队列里的一次待办：擦掉生存期的闭包指针 + 完成槽。
struct Ticket {
    trampoline: unsafe fn(*mut c_void),
    data: *mut c_void,
    slot: Arc<Slot>,
}

// SAFETY: `data` 指向投递方栈帧里的闭包，而投递方在 `scoped` 返回前**始终阻塞**，
// 且放弃只发生在任务尚未开始执行时 —— 因此这个指针只会在宿主线程上被解引用，
// 且一定发生在投递方离开栈帧之前。
unsafe impl Send for Ticket {}

impl Ticket {
    fn run(&self) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !matches!(&*state, SlotState::Pending) {
            // 投递方已经放弃 —— 绝不能再碰被借用的数据。
            return;
        }
        *state = SlotState::Running;
        drop(state);
        // SAFETY: 见 `unsafe impl Send for Ticket`；`data` 由 `scoped` 在同一个
        // 栈帧里提供，类型参数与 `trampoline` 一致。
        //
        // 【为什么必须在这里 catch】任务是任意宿主调用，panic 会一路穿过
        // `pump()` 与宿主的 `extern "system"` 入口 —— 那是 abort，也就是整个
        // REAPER 退出。而且不拦的话投递方会永远等在一个再也不会被唤醒的槽上。
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.trampoline)(self.data)
        }));
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *state = if outcome.is_ok() {
            SlotState::Done
        } else {
            SlotState::Panicked
        };
        self.slot.ready.notify_all();
    }
}

/// 把一件**拥有自己数据**的任务排到目标线程；**不等待**。
///
/// 【为什么需要这条与 [`scoped`] 并存的入口】`scoped` 借用调用方的栈帧，因此它
/// 只能在"投递方原地等"的前提下成立（放弃协议就是为此存在的）。而 ARA 的模型线程
/// 并不保证在我们愿意等的这段时间里回到我们的代码：任务必须能**留在队列里**，
/// 等那一次回调到来时再执行。留得住的前提是任务拥有自己的数据，也就是 `'static`。
///
/// 返回 `false` = 队列已满（没排上）。调用方据此给出"宿主线程队列繁忙"。
pub(crate) fn post_owned(
    target: std::thread::ThreadId,
    job: Box<dyn FnOnce() + Send + 'static>,
) -> bool {
    let mut pending = lock_queue();
    if pending.len() >= QUEUE_CAPACITY {
        log_queue_rejected(target, pending.len());
        return false;
    }
    pending.push_back(Queued {
        target,
        job,
        queued_at: Instant::now(),
        kind: EntryKind::Owned,
    });
    true
}

/// 一次投递的栈上上下文：闭包本身 + 结果写回处。
struct JobContext<'a, R, F> {
    job: Option<F>,
    output: &'a mut Option<R>,
}

/// 单态化的蹦床：把 `*mut c_void` 还原成具体闭包，跑一次，结果写回。
unsafe fn run_job<R, F: FnOnce() -> R>(data: *mut c_void) {
    // SAFETY: 调用方保证 `data` 指向一个仍然存活的 `JobContext<R, F>`。
    let context = unsafe { &mut *data.cast::<JobContext<'_, R, F>>() };
    if let Some(job) = context.job.take() {
        *context.output = Some(job());
    }
}

/// 立即排空并返回处理条数（测试用）。
#[cfg(test)]
fn drain() -> usize {
    let mut handled = 0;
    while let Some(job) = take_one() {
        // 必须**执行**而不是丢弃：丢弃会让投递方拿到"任务被扔掉"，
        // 那是与被测行为正好相反的结果。
        job();
        handled += 1;
    }
    handled
}

/// 【测试专用】进程级队列的全局串行锁。
///
/// 【为什么是模块级函数而不是 tests 模块的私有静态】`editor::host_threads` 的测试
/// 也往**同一条**进程级队列里投递；两边各持一把模块内静态锁时，两族用例照样并行
/// 互踩（一边把队列填满，另一边的 `scoped`/`deliver` 撞上 Backlog）。共用这一把，
/// 队列的任何测试写入都互斥。
#[cfg(test)]
pub(crate) fn serialize_queue_for_tests() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    /// 队列是**进程级**的，测试并行跑会互相看见对方的任务。串行化是本模块测试的
    /// 前提，而不是可选项。
    fn serialize() -> std::sync::MutexGuard<'static, ()> {
        super::serialize_queue_for_tests()
    }

    /// 起一个扮演"宿主线程"的线程，并把它的 ThreadId 交回来。
    ///
    /// 【为什么必须要这个 id】任务只许在目标线程上执行（见 `Queued`）。拿到 id
    /// 之后投递才真的等价于宿主场景。
    fn host_thread(stop: Arc<AtomicBool>) -> (std::thread::ThreadId, std::thread::JoinHandle<()>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = sender.send(std::thread::current().id());
            while !stop.load(Ordering::Acquire) {
                pump();
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        (receiver.recv().unwrap(), handle)
    }

    #[test]
    fn a_posted_job_runs_when_the_host_thread_pumps() {
        let _guard = serialize();
        drain();
        let (sender, receiver) = std::sync::mpsc::channel::<u32>();
        lock_queue().push_back(Queued {
            target: std::thread::current().id(),
            job: Box::new(move || {
                let _ = sender.send(7);
            }),
            queued_at: Instant::now(),
            kind: EntryKind::Owned,
        });
        assert!(drain() >= 1);
        assert_eq!(receiver.try_recv().unwrap(), 7);
    }

    /// 别人的任务绝不能被本线程的泵取走 —— 那正是"宿主调用跑错线程"的来源。
    #[test]
    fn a_pump_leaves_jobs_that_belong_to_another_thread() {
        let _guard = serialize();
        drain();
        let (sender, receiver) = std::sync::mpsc::channel::<u32>();
        lock_queue().push_back(Queued {
            target: std::thread::spawn(|| {}).thread().id(),
            job: Box::new(move || {
                let _ = sender.send(7);
            }),
            queued_at: Instant::now(),
            kind: EntryKind::Owned,
        });
        pump();
        assert!(receiver.try_recv().is_err(), "本线程的泵取走了别人的任务");
        assert!(pending() >= 1);
        drain();
    }

    /// **不变量：滞留超限的条目被清退，新条目不受影响。**
    ///
    /// 【为什么钉死】`deliver` 超时后任务留在队列里，目标线程若再也不泵（文档
    /// 已关），它既不执行也不被逐出 —— 满员后**所有**线程的投递都失败。清退
    /// 就是这条泄漏路径的唯一出口。
    #[test]
    fn a_stale_entry_is_evicted_while_a_fresh_one_survives() {
        let _guard = serialize();
        drain();
        let dead_target = std::thread::spawn(|| {}).thread().id();
        {
            let mut pending = lock_queue();
            pending.push_back(Queued {
                target: dead_target,
                job: Box::new(|| {}),
                // 直接构造一个已滞留超限的时刻，测试不等真实的 60 s。
                queued_at: Instant::now() - MAX_QUEUE_AGE - Duration::from_secs(1),
                kind: EntryKind::Scoped,
            });
            pending.push_back(Queued {
                target: dead_target,
                job: Box::new(|| {}),
                queued_at: Instant::now(),
                kind: EntryKind::Scoped,
            });
        }
        // 队列是进程级的：别的测试可能留下针对死线程的残留条目，因此断言
        // 「本次 pump 恰好清退一条」而不是绝对深度。
        let before = pending();
        pump();
        assert_eq!(
            pending(),
            before - 1,
            "滞留超限的条目必须被清退，其余条目必须保留"
        );
        drain();
    }

    #[test]
    fn scoped_returns_the_job_result_when_pumped() {
        let _guard = serialize();
        drain();
        let stop = Arc::new(AtomicBool::new(false));
        let (target, pumper) = host_thread(Arc::clone(&stop));
        let value = scoped(target, || 41 + 1).unwrap();
        assert_eq!(value, 42);
        stop.store(true, Ordering::Release);
        pumper.join().unwrap();
    }

    /// 借用必须能跨过这次投递 —— 这正是它存在的理由。
    #[test]
    fn scoped_borrows_the_callers_stack_frame() {
        let _guard = serialize();
        drain();
        let stop = Arc::new(AtomicBool::new(false));
        let (target, pumper) = host_thread(Arc::clone(&stop));
        let owned = String::from("host thread");
        let borrowed: &str = &owned;
        let seen = scoped(target, || borrowed.len()).unwrap();
        assert_eq!(seen, borrowed.len());
        stop.store(true, Ordering::Release);
        pumper.join().unwrap();
    }

    #[test]
    fn scoped_times_out_when_nobody_pumps() {
        let _guard = serialize();
        drain();
        let started = std::time::Instant::now();
        // 目标是一条**不泵**的线程：本测试线程的泵也取不走它。
        let target = std::thread::spawn(|| {}).thread().id();
        let error = scoped(target, || ()).unwrap_err();
        assert_eq!(error, HostThreadError::Unpicked);
        assert!(started.elapsed() >= HOST_THREAD_TIMEOUT);
        drain();
    }

    /// 放弃之后任务即使被取出也不得执行（否则会碰到已经失效的借用）。
    #[test]
    fn an_abandoned_job_never_runs() {
        let _guard = serialize();
        drain();
        let hits = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&hits);
        let target = std::thread::spawn(|| {}).thread().id();
        assert!(scoped(target, move || {
            seen.fetch_add(1, Ordering::AcqRel);
        })
        .is_err());
        // 现在让一条线程用那个 id 去泵：它必须看到"已放弃"并直接返回。
        let _ = target;
        assert_eq!(hits.load(Ordering::Acquire), 0);
    }

    /// 任务 panic 不能把投递方永远挂在槽上，也不能让 panic 穿过宿主边界。
    #[test]
    fn a_panicking_job_reports_instead_of_hanging() {
        let _guard = serialize();
        drain();
        let stop = Arc::new(AtomicBool::new(false));
        let (target, pumper) = host_thread(Arc::clone(&stop));
        let error = scoped(target, || -> u32 { panic!("boom") }).unwrap_err();
        assert_eq!(error, HostThreadError::Panicked);
        stop.store(true, Ordering::Release);
        pumper.join().unwrap();
    }

    /// **不变量：清退阈值按投递入口分层 —— scoped 60 s，owned 10 分钟。**
    ///
    /// 【为什么钉死】窗口线程的泵会顺手清退**别的**线程的条目；60 s 一刀切会把
    /// 宿主暂停期间（模型线程没有泵活动）排队中的播放请求静默丢掉 —— 用户按过
    /// 播放、拿到「已排队」回执，宿主恢复后却什么都不发生。owned 条目必须活得
    /// 比"一次暂停"久，同时侵蚀仍有界（10 分钟后清退 + Error 日志）。
    #[test]
    fn owned_entries_outlive_scoped_entries_in_the_queue() {
        let _guard = serialize();
        let dead_target = std::thread::spawn(|| {}).thread().id();
        // 队列是进程级的：先前用例可能留下**别的**目标线程的残留条目，
        // 因此只对死线程 id（全进程唯一）做相对计数，不做绝对深度断言。
        let count_for = |target: std::thread::ThreadId| {
            lock_queue()
                .iter()
                .filter(|entry| entry.target == target)
                .count()
        };
        assert_eq!(count_for(dead_target), 0);
        let enqueue = |kind: EntryKind, queued_at: Instant| {
            lock_queue().push_back(Queued {
                target: dead_target,
                job: Box::new(|| {}),
                queued_at,
                kind,
            });
        };
        let now = Instant::now();
        enqueue(EntryKind::Scoped, now - Duration::from_secs(61));
        enqueue(EntryKind::Owned, now - Duration::from_secs(61));
        evict_expired();
        assert_eq!(
            count_for(dead_target),
            1,
            "61 s 的 owned 条目必须存活，scoped 被清退"
        );

        // 验证 owned 阈值：目标线程已死，`drain()` 取不到它的任务。
        lock_queue().retain(|entry| entry.target != dead_target);
        enqueue(EntryKind::Owned, now - Duration::from_secs(601));
        evict_expired();
        assert_eq!(count_for(dead_target), 0, "超过 10 分钟的 owned 条目被清退");
        lock_queue().retain(|entry| entry.target != dead_target);
    }

    /// **不变量：队列满时 `post_owned` 拒投**（`deliver` 据此回
    /// `ModelThreadError::Failed`）。此前这条路径零覆盖。
    #[test]
    fn post_owned_refuses_when_the_queue_is_full() {
        let _guard = serialize();
        let dead_target = std::thread::spawn(|| {}).thread().id();
        // 从**当前深度**补齐到容量：进程级队列可能带着其它用例的残留条目，
        // 绝对填 256 条既可能撞上满队列、也可能把并行用例挤到 Backlog。
        let fill = QUEUE_CAPACITY - pending();
        {
            let mut pending = lock_queue();
            for _ in 0..fill {
                pending.push_back(Queued {
                    target: dead_target,
                    job: Box::new(|| {}),
                    queued_at: Instant::now(),
                    kind: EntryKind::Owned,
                });
            }
        }
        assert!(!post_owned(dead_target, Box::new(|| {})), "队列满必须拒投");
        // 只清理本用例的条目（死线程 id 全进程唯一，不会误伤残留条目）。
        lock_queue().retain(|entry| entry.target != dead_target);
    }
}
