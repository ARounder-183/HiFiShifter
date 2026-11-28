//! 采样型停滞看门狗：给"永不返回的宿主调用"一个名字。
//!
//! 【为什么必须存在】`HostCallGuard` / `AraCallbackGuard` 都在 `Drop` 里量耗时。
//! 一次**永不返回**的调用永远不会 `Drop` —— 于是它留下的证据是"日志里一条告警都没有"。
//! 现场两个挂起会话（REAPER `8e1c452b0d7daae8`、`3860ac3e2b6544c7`）里
//! `[host] call ... exceeded` 与 `[ara] callback ... exceeded` 都是 **0 条**：
//! 那不是"没问题"，是"看不见"。
//!
//! 【为什么是采样而不是超时定时器】定时器要挂在**被测量的那条线程**上才有机会触发，
//! 而那条线程正是卡住的那条。只有**另一条**线程定期读表，才能观察到"这一项已经很久
//! 没有归还"。
//!
//! 【为什么要 `progress_marker`】release 构建下的 backtrace 只有符号名，而且我们
//! **无法**从另一条线程安全地取到别人的栈。退一步：让被测量的线程在每个阶段入口写一个
//! 静态字符串（一次 TLS 写，纳秒级），看门狗把它打出来。于是日志能直接说
//! "卡在 remap_and_log -> prepare_renderers"，而不是只说"卡了 12 秒"。
//!
//! 【为什么表是固定槽位而不是 `Mutex<Vec<_>>`】登记发生在**宿主 UI 线程**上（每一次
//! 宿主调用都会登记）。一次 `Mutex` 争用就会把"测量"变成"新的阻塞源"，而那正是我们
//! 要消除的东西。固定槽位 + 原子量：登记是一次 CAS，读取是一次顺序扫描。

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// 同时在飞的宿主调用上限。
///
/// 【为什么是 64】宿主调用会嵌套（宿主回调里再调宿主），但实测嵌套深度个位数。
/// 64 个槽位 ≈ 一次 2 KB 的顺序扫描，1 Hz 采样下代价可以忽略。
pub(crate) const STALL_SLOTS: usize = 64;

/// 默认的停滞阈值（毫秒）。环境变量 `HFS_STALL_MS` 覆盖（`0` = 关掉看门狗）。
///
/// 【为什么是 2000】Windows 把窗口画成"未响应"的门槛量级就是 2 s；而正常的宿主调用
/// （一次 `get_timeline_state`、一次 `SetItemStateChunk`）都在几十毫秒内。
pub(crate) const DEFAULT_STALL_MS: u64 = 2000;

/// 看门狗的采样间隔。
const SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

/// 一个槽位：一次在飞的调用。
struct Slot {
    active: AtomicBool,
    /// 名字是 `&'static str`：指针 + 长度两个原子量。
    name_ptr: AtomicUsize,
    name_len: AtomicUsize,
    /// 登记的单调时刻（相对 `EPOCH` 的微秒）。
    started_us: AtomicU64,
    /// 登记线程的序号（见 `thread_serial`）。
    thread: AtomicU64,
    /// 调用方最近一次写的阶段名（同样是 `&'static str`）。
    marker_ptr: AtomicUsize,
    marker_len: AtomicUsize,
    /// 这一条已经报过没有（同一次调用只报一行）。
    reported: AtomicBool,
}

impl Slot {
    const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            name_ptr: AtomicUsize::new(0),
            name_len: AtomicUsize::new(0),
            started_us: AtomicU64::new(0),
            thread: AtomicU64::new(0),
            marker_ptr: AtomicUsize::new(0),
            marker_len: AtomicUsize::new(0),
            reported: AtomicBool::new(false),
        }
    }
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_SLOT: Slot = Slot::new();

static SLOTS: [Slot; STALL_SLOTS] = [EMPTY_SLOT; STALL_SLOTS];

/// 进程启动时刻：所有时间读数都是相对它的微秒。
static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// "阈值还没初始化"的哨兵。
///
/// 【为什么不能用 0】0 是**合法值**（= 关掉看门狗）。用它当"没读过环境变量"，
/// 会让"关掉"在下一次查询时被重新读成默认值 —— 开关看上去完全不起作用。
const THRESHOLD_UNSET: u64 = u64::MAX;

/// 停滞阈值（微秒，0 = 关）。
static THRESHOLD_US: AtomicU64 = AtomicU64::new(THRESHOLD_UNSET);

fn epoch() -> Instant {
    *EPOCH.get_or_init(Instant::now)
}

/// 停滞阈值。第一次调用时从 `HFS_STALL_MS` 读，之后是一次原子读。
pub(crate) fn stall_threshold() -> Option<Duration> {
    let micros = THRESHOLD_US.load(Ordering::Acquire);
    let micros = if micros == THRESHOLD_UNSET {
        let millis = std::env::var("HFS_STALL_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_STALL_MS);
        let micros = millis.saturating_mul(1000);
        THRESHOLD_US.store(micros, Ordering::Release);
        micros
    } else {
        micros
    };
    (micros > 0).then(|| Duration::from_micros(micros))
}

/// 测试专用：直接设定阈值（`Duration::ZERO` = 关）。
#[cfg(test)]
pub(crate) fn set_stall_threshold_for_test(threshold: Duration) {
    THRESHOLD_US.store(threshold.as_micros() as u64, Ordering::Release);
}

thread_local! {
    /// 本线程最内层在飞调用的槽位下标（`usize::MAX` = 没有）。
    static CURRENT: Cell<usize> = const { Cell::new(usize::MAX) };
    /// 本线程的稳定序号（与 `std::thread::ThreadId` 同寿命，但可写进原子量）。
    static SERIAL: u64 = next_serial();
}

static NEXT_SERIAL: AtomicU64 = AtomicU64::new(1);

fn next_serial() -> u64 {
    NEXT_SERIAL.fetch_add(1, Ordering::AcqRel)
}

/// 本线程的序号（只用于日志里区分"哪条线程"）。
pub(crate) fn thread_serial() -> u64 {
    SERIAL.with(|serial| *serial)
}

/// 把 `&'static str` 拆成两个原子量能装下的形状。
fn split(text: &'static str) -> (usize, usize) {
    (text.as_ptr() as usize, text.len())
}

/// 从两个原子量还原 `&'static str`。
///
/// # Safety
/// 指针与长度只由 `split` 写入，而它只接受 `&'static str` —— 因此还原出的切片
/// 指向的字节在整个进程生命周期内都有效。
unsafe fn join(ptr: usize, len: usize) -> &'static str {
    if ptr == 0 {
        return "";
    }
    // SAFETY: 见函数文档。
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr as *const u8, len)) }
}

/// 一次在飞调用被看门狗观测到的形状。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StalledCall {
    pub name: &'static str,
    pub marker: &'static str,
    pub thread: u64,
    pub elapsed: Duration,
}

/// 一次在飞的宿主调用。`Drop` 时归还槽位。
///
/// 【为什么与 `HostCallGuard` 分开】守卫回答"我在不在宿主调用里"，本类型回答
/// "这一次调用跑了多久"。两者生命周期相同，但一个要能被另一条线程看见，
/// 另一个只需要 thread-local。合在一起会让"可见"这件事被 `Drop` 语义绑死。
pub(crate) struct StallTicket {
    slot: usize,
    previous: usize,
    enabled: bool,
}

impl StallTicket {
    /// 登记一次在飞调用。看门狗关掉时它只做一次原子读。
    pub(crate) fn enter(name: &'static str) -> Self {
        let previous = CURRENT.with(|current| current.get());
        if stall_threshold().is_none() {
            return Self {
                slot: usize::MAX,
                previous,
                enabled: false,
            };
        }
        let mut chosen = usize::MAX;
        for (index, slot) in SLOTS.iter().enumerate() {
            if slot
                .active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let (ptr, len) = split(name);
                slot.name_ptr.store(ptr, Ordering::Release);
                slot.name_len.store(len, Ordering::Release);
                slot.marker_ptr.store(0, Ordering::Release);
                slot.marker_len.store(0, Ordering::Release);
                slot.thread.store(thread_serial(), Ordering::Release);
                slot.reported.store(false, Ordering::Release);
                let started = epoch().elapsed().as_micros() as u64;
                slot.started_us.store(started, Ordering::Release);
                chosen = index;
                break;
            }
        }
        if chosen != usize::MAX {
            CURRENT.with(|current| current.set(chosen));
            ensure_watchdog();
        }
        Self {
            slot: chosen,
            previous,
            enabled: chosen != usize::MAX,
        }
    }

    /// 本线程此刻是否有一个可写的槽位（`mark` 的快速判据）。
    fn current_slot() -> Option<usize> {
        let index = CURRENT.with(|current| current.get());
        (index != usize::MAX).then_some(index)
    }
}

impl Drop for StallTicket {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.previous));
        if !self.enabled {
            return;
        }
        SLOTS[self.slot].active.store(false, Ordering::Release);
    }
}

/// 记录"本线程这次宿主调用现在走到哪个阶段"。
///
/// 【代价】一次 TLS 读 + 两次原子写（只在有槽位时）。没有槽位时是一次 TLS 读。
/// 它被允许出现在任何路径上 —— 包括将来可能出现的音频线程。
pub(crate) fn mark(stage: &'static str) {
    let Some(index) = StallTicket::current_slot() else {
        return;
    };
    let (ptr, len) = split(stage);
    let slot = &SLOTS[index];
    slot.marker_ptr.store(ptr, Ordering::Release);
    slot.marker_len.store(len, Ordering::Release);
}

/// 扫一遍表，返回**第一条**已经超过阈值的在飞调用。
///
/// 【为什么是纯读而不是"事件"】看门狗线程与测试都要它；把"判定"与"写日志"分开，
/// 测试就能在没有日志后端的情况下钉住判据。
pub(crate) fn stalled(threshold: Duration) -> Option<StalledCall> {
    scan(threshold).next()
}

/// 扫一遍表，按槽位顺序产出**所有**超过阈值的在飞调用。
fn scan(threshold: Duration) -> impl Iterator<Item = StalledCall> {
    let now = epoch().elapsed().as_micros() as u64;
    SLOTS.iter().filter_map(move |slot| {
        if !slot.active.load(Ordering::Acquire) {
            return None;
        }
        let started = slot.started_us.load(Ordering::Acquire);
        let elapsed_us = now.saturating_sub(started);
        if elapsed_us < threshold.as_micros() as u64 {
            return None;
        }
        // SAFETY: 两个指针只由 `split` 写入（`&'static str`）。
        let (name, marker) = unsafe {
            (
                join(
                    slot.name_ptr.load(Ordering::Acquire),
                    slot.name_len.load(Ordering::Acquire),
                ),
                join(
                    slot.marker_ptr.load(Ordering::Acquire),
                    slot.marker_len.load(Ordering::Acquire),
                ),
            )
        };
        Some(StalledCall {
            name,
            marker: if marker.is_empty() { "?" } else { marker },
            thread: slot.thread.load(Ordering::Acquire),
            elapsed: Duration::from_micros(elapsed_us),
        })
    })
}

/// 测试专用：按名字找一条在飞调用。
///
/// 【为什么要按名字找】测试跑在同一个进程里、默认并行；只看"第一条停滞"会被
/// 相邻测试的槽位干扰。按名字过滤让每条测试只断言自己登记的那一条。
#[cfg(test)]
pub(crate) fn stalled_named(name: &str, threshold: Duration) -> Option<StalledCall> {
    scan(threshold).find(|call| call.name == name)
}

/// 把"这一条已经报过"记下来（同一次调用只报一行）。
fn note_reported(call: StalledCall) -> bool {
    for slot in SLOTS.iter() {
        if !slot.active.load(Ordering::Acquire) {
            continue;
        }
        let (ptr, len) = split(call.name);
        if slot.name_ptr.load(Ordering::Acquire) == ptr
            && slot.name_len.load(Ordering::Acquire) == len
            && slot.thread.load(Ordering::Acquire) == call.thread
        {
            return !slot.reported.swap(true, Ordering::AcqRel);
        }
    }
    false
}

/// 起一次看门狗线程（只起一次）。
fn ensure_watchdog() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("hfs-stall-watch".to_owned())
            .spawn(|| loop {
                std::thread::sleep(SAMPLE_INTERVAL);
                let Some(threshold) = stall_threshold() else {
                    continue;
                };
                while let Some(call) = stalled(threshold) {
                    if !note_reported(call) {
                        break;
                    }
                    // 【为什么是 Error】这是一条"进程正在失去响应"的判据，不是提示。
                    log::error!(
                        "[host] STALL {} for {}ms on thread {} at marker={}",
                        call.name,
                        call.elapsed.as_millis(),
                        call.thread,
                        call.marker
                    );
                    // 【为什么必须落盘】挂起现场里最贵的就是"最后那一段"，而 64 KiB 的
                    // 缓冲正好把它吃掉（两次 AppHangB1 的日志尾都是一行看起来很正常
                    // 的启动行）。这条判据既然知道"进程正在失去响应"，就必须让已经
                    // 写下的每一行先落到磁盘上 —— 证据比带宽重要。
                    crate::diagnostics::flush_writer_soon();
                }
            });
        if let Err(error) = spawned {
            log::warn!("[host] the stall watchdog could not start: {error}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **不变量：一次不归还的调用会被看门狗点名，并带上它最后走到的阶段。**
    ///
    /// 【为什么这条测试是本轮的核心】两个挂起会话里一条告警都没有，因为 `Drop` 型
    /// 看门狗看不见"永不返回"。这条测试把"看不见"变成"看得见"。
    #[test]
    fn stall_watchdog_names_the_site() {
        set_stall_threshold_for_test(Duration::from_millis(30));
        let ticket = StallTicket::enter("test.stall");
        mark("prepare_renderers");
        std::thread::sleep(Duration::from_millis(80));
        let call =
            stalled_named("test.stall", Duration::from_millis(30)).expect("这次调用必须被判成停滞");
        assert_eq!(call.name, "test.stall");
        assert_eq!(call.marker, "prepare_renderers");
        assert!(call.elapsed >= Duration::from_millis(30));
        drop(ticket);
        // 归还之后就不再是"在飞"的调用。
        assert!(
            stalled_named("test.stall", Duration::from_millis(30)).is_none(),
            "归还的调用不得再被点名"
        );
    }

    /// 关掉看门狗时它不登记、不判定。
    #[test]
    fn a_disabled_watchdog_registers_nothing() {
        set_stall_threshold_for_test(Duration::ZERO);
        let _ticket = StallTicket::enter("test.disabled");
        std::thread::sleep(Duration::from_millis(20));
        assert!(stalled_named("test.disabled", Duration::from_millis(1)).is_none());
    }

    /// `mark` 在没有槽位时是安全的空操作（音频线程也会走到它）。
    #[test]
    fn mark_without_a_ticket_is_a_no_op() {
        set_stall_threshold_for_test(Duration::from_millis(10));
        mark("nowhere");
        assert!(stalled_named("nowhere", Duration::from_millis(1)).is_none());
    }

    /// 嵌套登记：内层归还后，外层的槽位仍然是"在飞"的。
    #[test]
    fn nested_tickets_keep_the_outer_call_visible() {
        set_stall_threshold_for_test(Duration::from_millis(10));
        let outer = StallTicket::enter("test.outer");
        mark("outer.stage");
        {
            let _inner = StallTicket::enter("test.inner");
            mark("inner.stage");
        }
        std::thread::sleep(Duration::from_millis(40));
        let call = stalled_named("test.outer", Duration::from_millis(10)).expect("外层仍在飞");
        assert_eq!(call.marker, "outer.stage", "内层归还后标记必须回到外层");
        drop(outer);
    }
}
