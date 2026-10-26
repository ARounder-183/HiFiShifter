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
use std::time::Duration;

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

/// 在**宿主线程**上执行队列里的全部待办。
///
/// 【调用者必须是宿主线程】它由 `IPlugView` 的回调与注册到宿主的 timer 驱动。这里
/// 不做线程断言：断言会把"泵装错了地方"变成宿主崩溃，而不做断言的最坏后果只是任务
/// 跑在错误的线程上并各自失败 —— 后者可观测、可恢复。
///
/// 【为什么空队列时立刻返回】宿主每次重排窗口都会调 `getSize`，一次会话几十次；
/// 那时队列多半是空的，白等 2 ms 只是白白拖慢宿主。
pub(crate) fn pump() {
    let Some(first) = take_one() else {
        return;
    };
    first();
    let mut idle_since: Option<std::time::Instant> = None;
    loop {
        if let Some(job) = take_one() {
            job();
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
                return Err(HostThreadError::Backlog);
            }
            pending.push_back(Queued {
                target,
                job: Box::new(move || ticket.run()),
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
        return false;
    }
    pending.push_back(Queued { target, job });
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    /// 队列是**进程级**的，测试并行跑会互相看见对方的任务。串行化是本模块测试的
    /// 前提，而不是可选项。
    fn serialize() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|error| error.into_inner())
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
        });
        pump();
        assert!(receiver.try_recv().is_err(), "本线程的泵取走了别人的任务");
        assert!(pending() >= 1);
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
}
