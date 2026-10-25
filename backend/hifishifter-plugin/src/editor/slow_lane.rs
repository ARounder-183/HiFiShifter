//! 长任务的独立执行车道：一次一条、按请求回执、绝不阻塞 actor 与 UI 线程。
//!
//! # 它解决的是什么
//!
//! 插件有两条执行线：宿主 UI 线程（WebMessageReceived，内联处理需要 HWND/COM 的命令）
//! 与会话 actor（EditorSession::run，单线程 FIFO，同步调 commands::dispatch）。
//! 一条 31 秒的推理命令放在任何一条上，都会让另一件事停摆：actor 停摆 = 每一次编辑、
//! 每一次轮询、以及宿主保存时的 flush 全部排队到期 —— 用户看到的就是一串
//! `[ara] editor request N expired without a reply`，以及一个不响应的编辑器。
//!
//! 所以长任务需要**第三条线**。它必须满足两件事：
//!
//! 1. **不阻塞任何人**：任务在独立线程上跑，actor 与 UI 线程照常服务其它请求。
//! 2. **按请求回执**：结果是一个**返回值**（基准测试结果、诊断包路径），而前端每个
//!    invoke 都已经有一条 promise 在等它。让它轮询一个 status 命令，等于逼每个新功能
//!    再造一套状态机（对比 render::link::start_export：那个结果是**状态**，轮询才对）。
//!
//! # 铁律：worker 绝不触碰 UI
//!
//! BrowserState 是 Rc<RefCell<..>>，COM 接口只属于宿主 UI 线程，EditorSession 的状态
//! 只属于 actor。worker 对这三者**一个都不许碰**。它唯一能做的是把 JSON 塞进
//! UiSink::response 的 mpsc —— 那条通道的另一端是视图的 20 ms WM_TIMER，由 deliver()
//! 统一 PostWebMessageAsJson。这就是为什么并发在这里不需要任何新的线程亲和性论证：
//! 回执路径与 actor 完全共用。
//!
//! # 为什么是进程级一条车道
//!
//! 慢车道上的活都是进程级资源（ORT 会话、GPU 设备、设置文件）。两个视图各跑一份基准
//! 测试没有意义，而且它们最终会在内核的会话构建租约上排队 —— 那不如在这里就排队，
//! 并把「已经有一条在跑」如实告诉第二个调用者。

use super::session::UiSink;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Mutex, OnceLock};

/// 队列槽位数。在途上限 = 队列 4 + 正在执行 1 = **5**。
///
/// 【为什么这么小】长任务以分钟计。允许排 10 条等于承诺一个 5 分钟的等待，而调用方拿到
/// 的回执会早已超时。宁可立刻说「已经有一条在跑」。
const SLOW_CAPACITY: usize = 4;

/// 车道忙时的错误文本（唯一来源：测试与日志都按它比较）。
///
/// 【为什么必须是一句**可操作**的话】它会被 classify_error 判成 transient（值得重试），
/// 而用户看到的正是这句。他要的是「等它跑完」，不是「出错了」。
pub(crate) const SLOW_BUSY: &str = "a long-running task is already running; wait for it to finish";

/// 一次慢任务。
pub(crate) struct SlowTask {
    pub id: u64,
    /// 仅用于日志与诊断，**不参与**分发决策（分发决策在 commands::lane）。
    pub command: String,
    pub sink: UiSink,
    /// 真正的活。
    ///
    /// 【必须只捕获 owned 数据】Value / String / PathBuf / Arc<DocumentSession> 都可以；
    /// 绝不能捕获 &EditorSession、Rc 或 BrowserState —— 见模块头的铁律。Send 约束挡不住
    /// 这种误用，所以它是纪律 + 代码评审的一部分。
    pub work: Box<dyn FnOnce() -> Result<Value, String> + Send>,
}

/// 车道与它的线程共享的记账。
///
/// 【为什么单独一个结构】worker 必须更新"正在跑哪条 / 还有几条在途"，而它不能持有
/// `Lane` 本身（`Lane` 拥有队列的发送端，worker 持有接收端 —— 让它再拿一份发送端
/// 就成了自己喂自己）。`Arc` 共享记账是唯一干净的形状。
#[derive(Default)]
struct Shared {
    /// 正在执行的那条（诊断输出用）。
    running: Mutex<Option<String>>,
    /// 在途条数（排队 + 正在跑）。
    in_flight: AtomicUsize,
}

impl Shared {
    fn begin(&self, command: &str) {
        if let Ok(mut running) = self.running.lock() {
            *running = Some(command.to_owned());
        }
    }

    fn end(&self) {
        if let Ok(mut running) = self.running.lock() {
            *running = None;
        }
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    fn snapshot(&self) -> Value {
        json!({
            "running": self.running.lock().ok().and_then(|running| running.clone()),
            "inFlight": self.in_flight.load(Ordering::Acquire),
            "capacity": SLOW_CAPACITY,
        })
    }
}

/// 车道本身：一条常驻线程 + 有界队列。
pub(crate) struct Lane {
    queue: SyncSender<SlowTask>,
    shared: std::sync::Arc<Shared>,
}

impl Lane {
    fn new() -> Result<Self, String> {
        let shared = std::sync::Arc::new(Shared::default());
        let (queue, receiver) = sync_channel(SLOW_CAPACITY);
        let worker_shared = std::sync::Arc::clone(&shared);
        // 【为什么不是 expect】首条慢任务可能在宿主 UI 线程（wndproc 的 WM_TIMER →
        // deliver → enqueue）上触发车道初始化，那条路径没有 catch_unwind 包裹 ——
        // panic 会穿透窗口过程边界带走宿主进程。失败降级成「车道不可用」的
        // 可操作错误，由 submit 的调用方按既有错误回执路径交付。
        std::thread::Builder::new()
            .name("hfs-slow-lane".into())
            .spawn(move || worker(receiver, worker_shared))
            .map(|_handle| Self { queue, shared })
            .map_err(|error| format!("the slow lane worker failed to start: {error}"))
    }

    fn submit(&self, task: SlowTask) -> Result<(), String> {
        let (id, command) = (task.id, task.command.clone());
        // 先记账再入队：worker 侧的 fetch_sub 若插在「入队成功」与「记账」之间，
        // 计数会瞬态下溢回绕（诊断快照读出巨数）。先加后发最多把计数多报 1
        // （入队失败立刻减回），绝不会回绕。
        self.shared.in_flight.fetch_add(1, Ordering::AcqRel);
        match self.queue.try_send(task) {
            Ok(()) => {
                log::info!("[ara] slow lane: queued {command} (id={id})");
                Ok(())
            }
            Err(_) => {
                self.shared.in_flight.fetch_sub(1, Ordering::AcqRel);
                log::warn!(
                    "[ara] slow lane: rejected {command} (id={id}) — another long task is running"
                );
                Err(SLOW_BUSY.to_string())
            }
        }
    }

    fn snapshot(&self) -> Value {
        self.shared.snapshot()
    }
}

/// 进程级车道的惰性句柄。
///
/// 【为什么缓存 Result】spawn 失败（OOM 级）之后，下一次 submit 重试一次初始化是
/// 合理的；但 OnceLock 只能存一个值，于是把 Result 整个存下来 —— 成功路径照常，
/// 失败路径把同一句可操作的话交给每一次后续 submit。
fn lane() -> Result<&'static Lane, String> {
    static LANE: OnceLock<Result<Lane, String>> = OnceLock::new();
    LANE.get_or_init(Lane::new).as_ref().map_err(String::clone)
}

/// 提交一条慢任务。立即返回；回执由车道线程发出。
pub(crate) fn submit(task: SlowTask) -> Result<(), String> {
    lane()?.submit(task)
}

/// 卸载前等慢车道清空。返回是否在限时内归零（车道不可用视为空闲）。
///
/// 【为什么是限时等待而不是 join】任务以分钟计，`ExitDll` 不能冻住宿主；
/// 超时就放弃 —— 模块已被 `pin_module` 钉住，线程安全地活到进程退出。
pub(crate) fn wait_until_idle(timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let busy = match lane() {
            Ok(lane) => lane.shared.in_flight.load(Ordering::Acquire) > 0,
            Err(_) => false,
        };
        if !busy {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// 车道状态（诊断包用）。
pub(crate) fn snapshot() -> Value {
    match lane() {
        Ok(lane) => lane.snapshot(),
        Err(error) => json!({
            "running": null,
            "inFlight": 0,
            "capacity": SLOW_CAPACITY,
            "available": false,
            "error": error,
        }),
    }
}

/// 车道线程：一条一条地跑，每条都**必然**得到一条回执。
fn worker(receiver: Receiver<SlowTask>, shared: std::sync::Arc<Shared>) {
    for task in receiver {
        let SlowTask {
            id,
            command,
            sink,
            work,
        } = task;
        let started = std::time::Instant::now();
        shared.begin(&command);
        // 任务真正开始执行的那一刻把模块钉住：任务以分钟计，宿主在此窗口内
        // ExitDll → FreeLibrary 会让本线程落到已卸载的代码页（第 5 轮审查 §3）。
        super::pin_module();
        log::info!("[ara] slow lane: started {command} (id={id})");
        // 长任务里的 panic 既不能带走宿主进程，也不能让调用方永远等不到回执。
        let outcome = run_task(work);
        let ok = outcome.is_ok();
        let response = match outcome {
            Ok(value) => {
                json!({"version":1,"viewId":sink.view_id,"id":id,"ok":true,"value":value})
            }
            // 与 actor / 视图层同一条协议：终态错误必须带 errorKind，否则前端无法区分
            // 「稍后重试」与「会话已结束」。
            Err(error) => json!({"version":1,"viewId":sink.view_id,"id":id,"ok":false,
                "errorKind":super::classify_error(&error),
                "errorCode":super::errors::terminal_error_code(&error),"error":error}),
        };
        log::info!(
            "[ara] slow lane: finished {command} (id={id}) in {}ms ok={ok}",
            started.elapsed().as_millis()
        );
        // 唯一回执出口。视图已关闭时 response 自己丢弃（见 UiSink）。
        sink.response(response);
        shared.end();
    }
}

/// 执行一条任务，panic 折成错误。
fn run_task(work: Box<dyn FnOnce() -> Result<Value, String> + Send>) -> Result<Value, String> {
    // 【为什么测试里不真的跑】唯一的生产慢命令是基准测试，它要连建三个 ONNX 会话并各跑
    // 五轮推理。测试要验证的是**路由与回执**，不该为此付出三十秒与一块 GPU。
    #[cfg(test)]
    if tests::passthrough() {
        return Ok(json!({"ok":true,"dry_run":true}));
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
        .unwrap_or_else(|_| Err("long-running command panicked".to_string()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// 测试期间让车道「回执但不干活」。
    static PASSTHROUGH: AtomicBool = AtomicBool::new(false);

    pub(crate) fn passthrough() -> bool {
        PASSTHROUGH.load(Ordering::Acquire)
    }

    /// 开启透传并返回守卫：用例结束（含 panic）即恢复。
    pub(crate) struct Passthrough;

    impl Passthrough {
        pub(crate) fn enable() -> Self {
            PASSTHROUGH.store(true, Ordering::Release);
            Self
        }
    }

    impl Drop for Passthrough {
        fn drop(&mut self) {
            PASSTHROUGH.store(false, Ordering::Release);
        }
    }

    fn fake_sink() -> (UiSink, std::sync::mpsc::Receiver<Value>) {
        let (reply, receiver) = std::sync::mpsc::channel();
        let (events, event_receiver) = std::sync::mpsc::sync_channel(64);
        // 事件通道在测试里不用，但要保住接收端，否则 sender 一 send 就失败。
        std::mem::forget(event_receiver);
        (
            UiSink {
                view_id: "test-view".into(),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        )
    }

    /// **不变量：一条任务恰好一条回执，形状与 actor 完全一致。**
    #[test]
    fn a_task_replies_exactly_once_with_the_actor_protocol() {
        let lane = Lane::new().expect("车道线程必须能启动");
        let (sink, replies) = fake_sink();
        lane.submit(SlowTask {
            id: 7,
            command: "run_vocoder_benchmark".into(),
            sink,
            work: Box::new(|| Ok(json!({"cpuMedianMs": 1.0}))),
        })
        .expect("submit");
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reply");
        assert_eq!(reply["version"], 1);
        assert_eq!(reply["viewId"], "test-view");
        assert_eq!(reply["id"], 7);
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["value"]["cpuMedianMs"], 1.0);
        assert!(
            replies
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "一条任务只能有一条回执"
        );
    }

    /// **不变量：失败也要回执，且带分类。**
    ///
    /// 【为什么】调用方在等一条 promise。没有回执时它只能等超时，而超时无法区分
    /// 「任务失败」与「插件卡死」。
    #[test]
    fn a_failing_task_replies_with_a_classified_error() {
        let lane = Lane::new().expect("车道线程必须能启动");
        let (sink, replies) = fake_sink();
        lane.submit(SlowTask {
            id: 8,
            command: "run_vocoder_benchmark".into(),
            sink,
            work: Box::new(|| Err("model file missing".to_string())),
        })
        .expect("submit");
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reply");
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error"], "model file missing");
        assert_eq!(reply["errorKind"], "transient");
        assert!(reply["errorCode"].is_null(), "非终态错误不带协议码");
    }

    /// **不变量：panic 不带走宿主进程，也不吞掉回执。**
    #[test]
    fn a_panicking_task_still_replies_and_the_lane_survives() {
        let lane = Lane::new().expect("车道线程必须能启动");
        let (sink, replies) = fake_sink();
        lane.submit(SlowTask {
            id: 9,
            command: "run_vocoder_benchmark".into(),
            sink,
            work: Box::new(|| panic!("driver exploded")),
        })
        .expect("submit");
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reply");
        assert_eq!(reply["ok"], false);
        assert!(reply["error"]
            .as_str()
            .unwrap_or_default()
            .contains("panicked"));

        let (sink, replies) = fake_sink();
        lane.submit(SlowTask {
            id: 10,
            command: "run_vocoder_benchmark".into(),
            sink,
            work: Box::new(|| Ok(json!({"ok":true}))),
        })
        .expect("submit after panic");
        let reply = replies
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reply");
        assert_eq!(reply["id"], 10);
        assert_eq!(reply["ok"], true);
    }

    /// **不变量：一次只跑一条。**
    ///
    /// 【为什么】慢任务占的是进程级资源（GPU / ORT 会话）。并发跑两条不会更快，只会让
    /// 两条都变慢，并且在内核的构建租约上互相挡。
    #[test]
    fn the_lane_runs_one_task_at_a_time() {
        let lane = Arc::new(Lane::new().expect("车道线程必须能启动"));
        let peak = Arc::new(AtomicUsize::new(0));
        let live = Arc::new(AtomicUsize::new(0));
        let mut receivers = Vec::new();
        for id in 0..3u64 {
            let (sink, replies) = fake_sink();
            receivers.push(replies);
            let peak = peak.clone();
            let live = live.clone();
            lane.submit(SlowTask {
                id,
                command: "run_vocoder_benchmark".into(),
                sink,
                work: Box::new(move || {
                    let now = live.fetch_add(1, Ordering::AcqRel) + 1;
                    peak.fetch_max(now, Ordering::AcqRel);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    live.fetch_sub(1, Ordering::AcqRel);
                    Ok(json!({"id":id}))
                }),
            })
            .expect("submit");
        }
        for replies in &receivers {
            replies
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("reply");
        }
        assert_eq!(peak.load(Ordering::Acquire), 1, "车道必须串行执行");
    }

    /// **不变量：队列满时立刻给出可操作的话，而不是让调用方等到超时。**
    #[test]
    fn a_full_lane_is_refused_with_an_actionable_error() {
        let lane = Lane::new().expect("车道线程必须能启动");
        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate_rx));
        // 第一条：占住车道，直到测试放行。
        let (sink, first) = fake_sink();
        let held = gate.clone();
        lane.submit(SlowTask {
            id: 1,
            command: "run_vocoder_benchmark".into(),
            sink,
            work: Box::new(move || {
                let _ = held.lock().expect("gate").recv();
                Ok(json!({}))
            }),
        })
        .expect("first submit");
        // 一直填到被拒：worker 可能已经取走第一条，所以"还差几条"取决于时序，
        // 能钉住的是**必然**会满，且满的时候给的是那句可操作的话。
        let mut blockers = Vec::new();
        let mut refused = None;
        for id in 2..=(SLOW_CAPACITY as u64 + 8) {
            let (sink, replies) = fake_sink();
            match lane.submit(SlowTask {
                id,
                command: "run_vocoder_benchmark".into(),
                sink,
                work: Box::new(|| Ok(json!({}))),
            }) {
                Ok(()) => blockers.push(replies),
                Err(error) => {
                    refused = Some(error);
                    break;
                }
            }
        }
        assert_eq!(
            refused,
            Some(SLOW_BUSY.to_string()),
            "队列满时必须给出可操作的话，而不是让调用方等到超时"
        );
        // 放行，让车道把积压跑完（否则这些回执一直悬着）。
        gate_tx.send(()).expect("release");
        let _ = first.recv_timeout(std::time::Duration::from_secs(10));
        for replies in blockers {
            let _ = replies.recv_timeout(std::time::Duration::from_secs(10));
        }
    }

    /// **不变量：视图已关闭时回执被丢弃，车道继续工作。**
    #[test]
    fn a_closed_view_swallows_the_reply() {
        let lane = Lane::new().expect("车道线程必须能启动");
        let (sink, replies) = fake_sink();
        sink.closed.store(true, Ordering::Release);
        lane.submit(SlowTask {
            id: 11,
            command: "run_vocoder_benchmark".into(),
            sink,
            work: Box::new(|| Ok(json!({"ok":true}))),
        })
        .expect("submit");
        assert!(
            replies
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "关闭的视图不该收到回执"
        );
        assert_eq!(lane.snapshot()["inFlight"], 0, "回执后必须清零在途计数");
    }
}
