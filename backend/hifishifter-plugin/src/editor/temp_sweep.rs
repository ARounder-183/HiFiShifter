//! 本机数据目录里临时子目录的**分片**清扫。
//!
//! 【为什么需要】`pcm/{namespace}` 与 `webview/{profile}` 都按会话/进程新建目录，只在
//! `Drop` 时清理。宿主被强杀、插件被卸载、进程崩溃时 `Drop` 不跑，这些目录就永久留下
//! —— 实测本机攒到 `webview` 147 个目录约 7.1 GiB，全仓没有任何回收点。启动时按
//! "属主进程是否还活着"扫一遍，把已退出进程的目录删掉。
//!
//! 【为什么必须分片】`remove_dir_all` 在 Windows 上不是常数时间：WebView2 profile 是
//! 大量小文件，删一个目录就可能耗时数百毫秒到数秒。此前它在 `EditorSession::new` 里
//! 同步执行，而那个调用点又持有 `DocumentSession::transaction`、跑在宿主 UI 线程上 ——
//! 一次重开工程就可能把 `get_runtime_info` 的 30 s IPC 超时直接触发。
//!
//! 现在只做两件事：
//!
//! 1. [`sweep_step`] 每次只处理**一个**目录，由会话 actor 的空闲 tick 调用；
//! 2. 先 `rename` 成 `.deleting-{pid}`（毫秒级），再 `remove_dir_all`。即使删除中途
//!    失败或进程退出，下一轮也能凭这个标记认出"已经决定要删的目录"。
//!
//! 【为什么必须**周期**重扫，而不是"扫一次就完"】曾经的实现是队列空了就置 `done`，
//! 此后再也不扫。于是"第一次扫的时候属主还活着"这一个瞬时事实，会变成整场会话的
//! 结论 —— 用户在会话中途退出又重开 REAPER，上一次的 profile 就永远留在磁盘上。
//! 现在每 [`RESCAN_EVERY`] 重新扫一次；代价只是两次目录读。
//!
//! 【怎么判"属主已退出"】名字里就带着属主 PID（`view-{pid}` / `hfs-ui-{pid}-{n}-`）。
//! 三种情形：
//!
//! | 情形 | 判据 | 处置 |
//! | --- | --- | --- |
//! | PID 不存在 | `OpenProcess` 报 `ERROR_INVALID_PARAMETER` | 陈旧 |
//! | PID 存在且**创建时间与目录同龄** | 进程创建时刻 ≈ 目录创建时刻 | 活着，保留 |
//! | PID 存在但**创建时间晚于目录** | 进程比目录新 | PID 复用 ⇒ 陈旧 |
//!
//! 第三种是 PID 复用的**精确**判据：Windows 的 PID 会复用，于是"PID 1234 还活着"
//! 完全可能是另一个进程。旧实现只能用"目录超过 7 天"兜底，而 7 天既太长（一个
//! 大 profile 会占着磁盘一周）又只对**认不出 PID** 的名字生效。改成比较创建时刻之后，
//! 已知 PID 的目录也能被判陈旧，且不需要任何时间阈值。
//!
//! 【为什么访问被拒时判"活着"】`OpenProcess` 对高完整性级别的进程会报 `ACCESS_DENIED`，
//! 那**不**等于进程不存在。把"查不到"当成"已退出"会删掉另一个还开着的宿主进程的
//! profile —— 那种损坏无法回滚。宁可不删。
//!
//! 【为什么删不掉不算错误】目录可能正被另一个进程的 WebView2 占用。删除失败就留着
//! —— 下一轮（下一次扫描）会再试，而不是让插件启动失败。

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// 名字里认不出属主进程时，超过这个年龄就删。
const STALE_AGE: Duration = Duration::from_secs(7 * 24 * 3600);

/// 已决定删除、但可能还没删完的目录标记。
///
/// 【为什么是后缀而不是前缀】它必须保留原目录名里的属主 PID 信息；同时
/// [`pid_after_prefix`] 只认前缀，所以带标记的目录会走"无属主"分支 —— 而我们在
/// [`is_marked_for_deletion`] 里把它直接判成可删，不必等 7 天。
const DELETING_MARK: &str = ".deleting-";

/// 多久重新扫一次根目录。
///
/// 【为什么是 60 s 而不是每 tick】扫描是两次 `read_dir` 加每个条目一次进程查询，
/// 147 个目录约 1 ms —— 60 s 一次在噪声以下，而 60 s 的滞后对"删一个已经退出的
/// 会话的 profile"毫无影响。
const RESCAN_EVERY: Duration = Duration::from_secs(60);

/// 判 PID 复用时允许的时钟误差。
///
/// 目录创建时刻与进程创建时刻都来自系统时钟，但取的是两次不同的系统调用（且 NTFS
/// 的时间戳在更新时有惰性）。2 s 的松弛足以吸收它，又远小于"两个不同进程复用一个
/// PID"所需的间隔。
const PID_REUSE_SLACK: Duration = Duration::from_secs(2);

/// 一次扫描的读数。
#[derive(Default, Clone, Copy)]
struct ScanStats {
    /// 根目录下见到的子目录总数（无论死活）。
    dirs: usize,
    /// 判为陈旧、已进队的目录数。
    stale: usize,
}

/// 一次进程只建一份清扫队列。
struct SweepState {
    /// 本轮待删目录。空 = 要么还没扫过，要么扫完了。
    pending: VecDeque<PathBuf>,
    /// 是否扫过至少一次（第一次扫描不受 [`RESCAN_EVERY`] 限制）。
    scanned: bool,
    /// 上一次扫描的时刻。
    last_scan: Instant,
    /// 上一次扫描的读数（用来只在**变化时**打日志）。
    stats: ScanStats,
    /// 累计已回收目录数与失败次数（诊断用）。
    reclaimed: u64,
    failed: u64,
    /// 一次性磁盘占用报告是否已经派发。
    size_report_spawned: bool,
}

static SWEEP: OnceLock<Mutex<SweepState>> = OnceLock::new();

fn state() -> &'static Mutex<SweepState> {
    SWEEP.get_or_init(|| {
        Mutex::new(SweepState {
            pending: VecDeque::new(),
            scanned: false,
            last_scan: Instant::now(),
            stats: ScanStats::default(),
            reclaimed: 0,
            failed: 0,
            size_report_spawned: false,
        })
    })
}

/// 处理一个待删目录。返回是否真的做了工作。
///
/// 【调用点】`EditorSession::run` 的 `recv_timeout` 超时分支 —— 只有队列空闲时才跑。
/// 一次只删一个目录，把"删 7 GiB"摊到多个 100 ms tick 上，命令队列始终有机会先被
/// 处理。
pub(crate) fn sweep_step() -> bool {
    let path = {
        let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
        let now = Instant::now();
        let due = !state.scanned
            || (state.pending.is_empty() && now.duration_since(state.last_scan) >= RESCAN_EVERY);
        if due {
            let (pending, stats) = collect_stale_dirs();
            // 【只在变化时打】稳态下每 60 s 都会扫到同样的读数，逐次打日志会把
            // 诊断文件刷满。首次扫描必须打（那是"这台机器上到底攒了多少"的唯一读数）。
            let changed = !state.scanned || stats.dirs != state.stats.dirs || stats.stale > 0;
            state.pending = pending;
            state.stats = stats;
            state.scanned = true;
            state.last_scan = now;
            if changed {
                crate::gui_log_line(&format!(
                    "webview profiles: dirs={} stale={} reclaimed={} failed={}",
                    stats.dirs, stats.stale, state.reclaimed, state.failed
                ));
            }
            // 【为什么另起线程】把 7 GiB 走一遍要几秒。这条读数只服务诊断，绝不能
            // 让它在会话 actor 上跑 —— 那正是本模块存在的理由。
            if !state.size_report_spawned && stats.dirs > 0 {
                state.size_report_spawned = true;
                spawn_size_report();
            }
        }
        match state.pending.pop_front() {
            Some(path) => path,
            None => return false,
        }
    };
    let deleted = remove_stale_dir(&path);
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    if deleted {
        state.reclaimed += 1;
    } else {
        state.failed += 1;
    }
    if state.pending.is_empty() && (state.reclaimed > 0 || state.failed > 0) {
        // 队列排空：把这一轮的成果落一行，然后等下一次重扫。
        crate::gui_log_line(&format!(
            "webview profiles swept: reclaimed={} failed={} (the rest is kept for the next scan)",
            state.reclaimed, state.failed
        ));
        state.reclaimed = 0;
        state.failed = 0;
    }
    true
}

/// 扫描两个根目录，返回（待删队列，读数）。
fn collect_stale_dirs() -> (VecDeque<PathBuf>, ScanStats) {
    let mut pending = VecDeque::new();
    let mut stats = ScanStats::default();
    for (subdir, prefix) in [("webview", "view-"), ("pcm", "hfs-ui-")] {
        collect_stale_in(
            &hifishifter_kernel::config_location::local_data_subdir(subdir),
            prefix,
            &mut pending,
            &mut stats,
        );
    }
    (pending, stats)
}

fn collect_stale_in(
    root: &Path,
    prefix: &str,
    pending: &mut VecDeque<PathBuf>,
    stats: &mut ScanStats,
) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let now = SystemTime::now();
    let own = std::process::id();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        stats.dirs += 1;
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_marked_for_deletion(&name) {
            pending.push_back(path);
            stats.stale += 1;
            continue;
        }
        let removable = match pid_after_prefix(&name, prefix) {
            // 自己的目录当然要留（可能正被本次会话使用）。
            Some(pid) if pid == own => false,
            Some(pid) => owner_is_gone(pid, &path),
            None => is_older_than(&path, now, STALE_AGE),
        };
        if removable {
            pending.push_back(path);
            stats.stale += 1;
        }
    }
}

/// 属主进程是否已经不在了（或者这个 PID 现在是别的进程）。
fn owner_is_gone(pid: u32, dir: &Path) -> bool {
    match process_start_time(pid) {
        // 进程不存在。
        ProcessProbe::Missing => true,
        // 查不到（权限不足等）：**不能**当成已退出。
        ProcessProbe::Unknown => false,
        ProcessProbe::Started(start) => {
            // 目录的创建时刻就是属主进程创建它的时刻。若现在这个 PID 对应的进程
            // 比目录**还新**，那它不可能是属主 —— 这是 PID 复用的精确判据。
            match dir.metadata().ok().and_then(|meta| meta.created().ok()) {
                Some(created) => start > created + PID_REUSE_SLACK,
                // 取不到创建时刻：退化为"活着就保留"。宁可漏删。
                None => false,
            }
        }
    }
}

/// 进程存活探测的结果。
///
/// 【为什么要三态】"查不到"与"不存在"必须分开：前者删目录会造成不可回滚的损坏，
/// 后者不删只是多占磁盘。
enum ProcessProbe {
    Missing,
    Started(SystemTime),
    Unknown,
}

/// 先改名再删除：改名是毫秒级的，删除可以慢。返回是否真的删掉了。
fn remove_stale_dir(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    if is_marked_for_deletion(&name) {
        // 上一轮已经改过名：直接删。
        return std::fs::remove_dir_all(path).is_ok();
    }
    let Some(parent) = path.parent() else {
        return false;
    };
    let marked = parent.join(format!("{name}{DELETING_MARK}{}", std::process::id()));
    // 目标已存在时先清掉它（上一次删除失败留下的）。
    if marked.exists() {
        let _ = std::fs::remove_dir_all(&marked);
    }
    if std::fs::rename(path, &marked).is_err() {
        // 【为什么不再退化为原地删除】改名失败几乎只有两种原因：目录被占用、
        // 或权限不足。原地删除在第一种情形下会**部分删掉**一个还活着的 profile
        // （WebView2 之后会以各种方式失败），在第二种情形下同样失败。两种都不该做，
        // 留给下一次扫描重试。
        return false;
    }
    std::fs::remove_dir_all(&marked).is_ok()
}

fn is_marked_for_deletion(name: &str) -> bool {
    name.contains(DELETING_MARK)
}

/// `{prefix}{pid}-…` → pid。
fn pid_after_prefix(name: &str, prefix: &str) -> Option<u32> {
    name.strip_prefix(prefix)?.split('-').next()?.parse().ok()
}

fn is_older_than(path: &Path, now: SystemTime, age: Duration) -> bool {
    path.metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|elapsed| elapsed > age)
}

/// 一次性报告 `webview` 根目录的磁盘占用（诊断读数，不参与判定）。
fn spawn_size_report() {
    let root = hifishifter_kernel::config_location::local_data_subdir("webview");
    let spawned = std::thread::Builder::new()
        .name("hfs-profile-size".into())
        .spawn(move || {
            let (dirs, bytes) = dir_usage(&root);
            crate::gui_log_line(&format!(
                "webview profiles on disk: dirs={dirs} bytes={bytes} mib={}",
                bytes / (1024 * 1024)
            ));
        });
    if let Err(error) = spawned {
        // 起不了线程只丢一条读数，不影响清扫。
        crate::gui_log_line(&format!("profile size report skipped: {error}"));
    }
}

/// 递归统计目录数与字节数（只读，容忍任何错误）。
fn dir_usage(root: &Path) -> (usize, u64) {
    let mut dirs = 0;
    let mut bytes = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => {
                    dirs += 1;
                    stack.push(entry.path());
                }
                Ok(kind) if kind.is_file() => {
                    if let Ok(meta) = entry.metadata() {
                        bytes = bytes.saturating_add(meta.len());
                    }
                }
                _ => {}
            }
        }
    }
    (dirs, bytes)
}

/// 属主进程的创建时刻。
///
/// 【为什么用 `PROCESS_QUERY_LIMITED_INFORMATION`】`SYNCHRONIZE` 在跨完整性级别时
/// 会被拒；而本函数要回答的是"这个 PID 现在是谁"，`QUERY_LIMITED_INFORMATION` 是
/// 为此设计的最小权限。
#[cfg(windows)]
fn process_start_time(pid: u32) -> ProcessProbe {
    use windows::Win32::Foundation::{CloseHandle, FILETIME};
    use windows::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    // ERROR_INVALID_PARAMETER：PID 不存在。其它错误一律按"查不到"处理。
    const ERROR_INVALID_PARAMETER: i32 = 87;
    // SAFETY: 只查询内核对象；返回的句柄立即关闭。
    let handle = match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
        Ok(handle) => handle,
        Err(error) => {
            let code = error.code().0 & 0xFFFF;
            return if code == ERROR_INVALID_PARAMETER {
                ProcessProbe::Missing
            } else {
                ProcessProbe::Unknown
            };
        }
    };
    let mut created = FILETIME::default();
    let mut exited = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: 四个输出指针都指向本栈上的有效 FILETIME。
    let queried =
        unsafe { GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user) };
    let _ = unsafe { CloseHandle(handle) };
    if queried.is_err() {
        return ProcessProbe::Unknown;
    }
    match filetime_to_system_time(created) {
        Some(start) => ProcessProbe::Started(start),
        None => ProcessProbe::Unknown,
    }
}

/// `FILETIME`（自 1601-01-01 起的 100 ns）→ `SystemTime`。
#[cfg(windows)]
fn filetime_to_system_time(time: windows::Win32::Foundation::FILETIME) -> Option<SystemTime> {
    // 1601-01-01 到 1970-01-01 的秒数。
    const EPOCH_DIFFERENCE_SECS: u64 = 11_644_473_600;
    let ticks = ((time.dwHighDateTime as u64) << 32) | time.dwLowDateTime as u64;
    let secs = ticks / 10_000_000;
    let nanos = ((ticks % 10_000_000) * 100) as u32;
    let since_unix = secs.checked_sub(EPOCH_DIFFERENCE_SECS)?;
    Some(SystemTime::UNIX_EPOCH + Duration::new(since_unix, nanos))
}

/// 非 Windows 上没有 WebView2 形态；保留"属主一定活着"的安全默认，不误删。
#[cfg(not(windows))]
fn process_start_time(_pid: u32) -> ProcessProbe {
    ProcessProbe::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_prefix_parsing_matches_the_directory_naming() {
        assert_eq!(pid_after_prefix("view-1234", "view-"), Some(1234));
        assert_eq!(pid_after_prefix("hfs-ui-4321-2-", "hfs-ui-"), Some(4321));
        // 名字不符合格式 ⇒ 无属主，交给年龄兜底。
        assert_eq!(pid_after_prefix("webview", "view-"), None);
        assert_eq!(pid_after_prefix("view-notanumber-1", "view-"), None);
        assert_eq!(pid_after_prefix("hfs-ui--1-", "hfs-ui-"), None);
    }

    /// 带删除标记的目录必须被优先回收，不受年龄限制。
    #[test]
    fn a_marked_directory_is_collected_regardless_of_age() {
        let root = scratch("mark");
        let marked = root.join(format!("view-1-1{DELETING_MARK}{}", std::process::id()));
        std::fs::create_dir_all(&marked).unwrap();
        let mut pending = VecDeque::new();
        let mut stats = ScanStats::default();
        collect_stale_in(&root, "view-", &mut pending, &mut stats);
        assert!(
            pending.iter().any(|path| path == &marked),
            "带删除标记的目录必须进入待删队列"
        );
        assert_eq!(stats.dirs, 1);
        assert_eq!(stats.stale, 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 自己的目录必须留下；属主已消失的目录必须进队；非目录条目必须被跳过。
    #[test]
    fn only_directories_whose_owner_is_gone_are_collected() {
        let root = scratch("collect");
        let own = root.join(format!("view-{}", std::process::id()));
        // 一个几乎不可能存在的 PID（Windows 进程表上限远低于它）。
        let gone = root.join("view-4294967290");
        let plain_file = root.join("view-4294967291");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::create_dir_all(&gone).unwrap();
        std::fs::write(&plain_file, b"x").unwrap();

        let mut pending = VecDeque::new();
        let mut stats = ScanStats::default();
        collect_stale_in(&root, "view-", &mut pending, &mut stats);

        assert!(own.exists(), "本次进程自己的目录必须保留");
        assert!(
            pending.iter().any(|path| path == &gone),
            "属主已退出的目录必须进队"
        );
        assert!(
            !pending.iter().any(|path| path == &own),
            "自己的目录不得进队"
        );
        assert!(plain_file.exists(), "非目录条目不得被删");
        assert_eq!(stats.dirs, 2, "只统计目录");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 改名成功即删除：`remove_stale_dir` 成功之后原路径必须消失，且不留标记目录。
    #[test]
    fn removal_renames_then_deletes() {
        let root = scratch("remove");
        let target = root.join("view-4294967290");
        std::fs::create_dir_all(target.join("Cache")).unwrap();
        std::fs::write(target.join("Cache").join("blob"), b"data").unwrap();
        assert!(remove_stale_dir(&target), "改名 + 删除必须成功");
        assert!(!target.exists(), "原目录必须消失");
        let leftovers = std::fs::read_dir(&root).unwrap().flatten().count();
        assert_eq!(leftovers, 0, "不得留下删除标记目录");
        let _ = std::fs::remove_dir_all(&root);
    }

    fn scratch(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hfs-sweep-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}
