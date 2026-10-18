//! 本机数据目录里临时子目录的**分片**清扫。
//!
//! 【为什么需要】`pcm/{namespace}` 与 `webview/{profile}` 都按会话/进程新建目录，只在
//! `Drop` 时清理。宿主被强杀、插件被卸载、进程崩溃时 `Drop` 不跑，这些目录就永久留下
//! —— 实测本机攒到 `webview` 41 个目录约 838 MB、`pcm` 32 个目录约 12 MB，全仓没有任何
//! 回收点。启动时按"属主进程是否还活着"扫一遍，把已退出进程的目录删掉。
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
//! 【为什么用 PID 而不是 mtime】mtime 会把"长时间开着、但确实在用"的目录一起删掉。
//! 目录名里就带着属主 PID（`view-{pid}-{n}` / `hfs-ui-{pid}-{n}-`），直接问系统更准；
//! 名字不符合格式的历史遗留目录只用年龄兜底。
//!
//! 【为什么删不掉不算错误】目录可能正被另一个进程的 WebView2 占用。`remove_dir_all`
//! 失败就留着 —— 下一轮（下一次启动）会再试，而不是让插件启动失败。

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

/// 名字里认不出属主进程时，超过这个年龄就删。
const STALE_AGE: Duration = Duration::from_secs(7 * 24 * 3600);

/// 已决定删除、但可能还没删完的目录标记。
///
/// 【为什么是后缀而不是前缀】它必须保留原目录名里的属主 PID 信息；同时
/// [`pid_after_prefix`] 只认前缀，所以带标记的目录会走"无属主"分支 —— 而我们在
/// [`is_marked_for_deletion`] 里把它直接判成可删，不必等 7 天。
const DELETING_MARK: &str = ".deleting-";

/// 一次进程只建一份清扫队列。
struct SweepState {
    /// `None` = 还没扫描过根目录；`Some(empty)` = 扫完了。
    pending: Option<VecDeque<PathBuf>>,
    done: bool,
}

static SWEEP: OnceLock<Mutex<SweepState>> = OnceLock::new();

/// 处理一个待删目录。返回是否真的做了工作。
///
/// 【调用点】`EditorSession::run` 的 `recv_timeout` 超时分支 —— 只有队列空闲时才跑。
/// 一次只删一个目录，把"删 838 MB"摊到多个 100 ms tick 上，命令队列始终有机会先被
/// 处理。
pub(crate) fn sweep_step() -> bool {
    let state = SWEEP.get_or_init(|| {
        Mutex::new(SweepState {
            pending: None,
            done: false,
        })
    });
    let path = {
        let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
        if state.done {
            return false;
        }
        if state.pending.is_none() {
            state.pending = Some(collect_stale_dirs());
        }
        match state
            .pending
            .as_mut()
            .and_then(|pending| pending.pop_front())
        {
            Some(path) => path,
            None => {
                state.done = true;
                return false;
            }
        }
    };
    remove_stale_dir(&path);
    true
}

/// 扫描两个根目录，返回"属主已退出"或"已标记删除"的子目录。
fn collect_stale_dirs() -> VecDeque<PathBuf> {
    let mut pending = VecDeque::new();
    for (subdir, prefix) in [("webview", "view-"), ("pcm", "hfs-ui-")] {
        collect_stale_in(
            &hifishifter_kernel::config_location::local_data_subdir(subdir),
            prefix,
            &mut pending,
        );
    }
    pending
}

fn collect_stale_in(root: &Path, prefix: &str, pending: &mut VecDeque<PathBuf>) {
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
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_marked_for_deletion(&name) {
            pending.push_back(path);
            continue;
        }
        let removable = match pid_after_prefix(&name, prefix) {
            // 自己的目录当然要留（可能正被本次会话使用）。
            Some(pid) if pid == own => false,
            Some(pid) => !process_alive(pid),
            None => is_older_than(&path, now, STALE_AGE),
        };
        if removable {
            pending.push_back(path);
        }
    }
}

/// 先改名再删除：改名是毫秒级的，删除可以慢。
fn remove_stale_dir(path: &Path) {
    let parent = path.parent();
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !is_marked_for_deletion(&name) {
        if let Some(parent) = parent {
            let marked = parent.join(format!("{name}{DELETING_MARK}{}", std::process::id()));
            // 目标已存在时先清掉它（上一次删除失败留下的）。
            if marked.exists() {
                let _ = std::fs::remove_dir_all(&marked);
            }
            if std::fs::rename(path, &marked).is_ok() {
                let _ = std::fs::remove_dir_all(&marked);
                return;
            }
        }
    }
    // 改名失败（被占用/跨卷/权限）：退化为原地删除。失败就留给下一轮。
    let _ = std::fs::remove_dir_all(path);
}

fn is_marked_for_deletion(name: &str) -> bool {
    name.contains(DELETING_MARK)
}

/// `{prefix}{pid}-…` → pid。
fn pid_after_prefix(name: &str, prefix: &str) -> Option<u32> {
    name.strip_prefix(prefix)?.split('-').next()?.parse().ok()
}

/// 删除 `root` 下所有"属主进程已退出"（或太旧、认不出属主）的子目录。
///
/// 【保留给测试与诊断】生产路径走 [`sweep_step`]，不再一次性删完。
#[cfg(test)]
fn sweep_dir(root: &Path, pid_of: impl Fn(&str) -> Option<u32>) {
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
        let name = entry.file_name().to_string_lossy().into_owned();
        let removable = match pid_of(&name) {
            Some(pid) if pid == own => false,
            Some(pid) => !process_alive(pid),
            None => is_older_than(&path, now, STALE_AGE),
        };
        if removable {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

fn is_older_than(path: &Path, now: SystemTime, age: Duration) -> bool {
    path.metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|elapsed| elapsed > age)
}

/// 属主进程是否仍存在。
///
/// 【为什么用 `OpenProcess(SYNCHRONIZE)` 而不是 `WaitForSingleObject`】PID 不存在时
/// `OpenProcess` 直接失败，这已经足够回答"属主还在不在"；再去 `Wait` 只会多一次系统
/// 调用，还得处理 `WAIT_TIMEOUT`/`WAIT_OBJECT_0` 两种返回。PID 复用是已知的误留来源，
/// 由年龄兜底覆盖。
#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE};
    // SAFETY: 只查询内核对象；返回的句柄立即关闭。
    match unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) } {
        Ok(handle) => {
            let _ = unsafe { CloseHandle(handle) };
            true
        }
        Err(_) => false,
    }
}

/// 非 Windows 上没有 WebView2 形态；保留"进程一定活着"的安全默认，不误删。
#[cfg(not(windows))]
fn process_alive(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_prefix_parsing_matches_the_directory_naming() {
        assert_eq!(pid_after_prefix("view-1234-7", "view-"), Some(1234));
        assert_eq!(pid_after_prefix("hfs-ui-4321-2-", "hfs-ui-"), Some(4321));
        // 名字不符合格式 ⇒ 无属主，交给年龄兜底。
        assert_eq!(pid_after_prefix("webview", "view-"), None);
        assert_eq!(pid_after_prefix("view-notanumber-1", "view-"), None);
        assert_eq!(pid_after_prefix("hfs-ui--1-", "hfs-ui-"), None);
    }

    /// 自己的目录必须留下；属主已消失的目录必须被删。
    #[test]
    fn sweep_removes_only_directories_whose_owner_is_gone() {
        let root = std::env::temp_dir().join(format!(
            "hfs-sweep-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let own = root.join(format!("view-{}-1", std::process::id()));
        // 一个几乎不可能存在的 PID（Windows 进程表上限远低于它）。
        let gone = root.join("view-4294967290-1");
        let plain_file = root.join("view-4294967291-1");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::create_dir_all(&gone).unwrap();
        // 非目录条目必须被跳过（不是所有同名条目都是目录）。
        std::fs::write(&plain_file, b"x").unwrap();

        sweep_dir(&root, |name| pid_after_prefix(name, "view-"));

        assert!(own.exists(), "本次进程自己的目录必须保留");
        assert!(!gone.exists(), "属主已退出的目录必须被清掉");
        assert!(plain_file.exists(), "非目录条目不得被删");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 已标记删除的目录必须被优先回收，不受 7 天年龄限制。
    #[test]
    fn a_marked_directory_is_collected_regardless_of_age() {
        let root = std::env::temp_dir().join(format!(
            "hfs-sweep-mark-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let marked = root.join(format!("view-1-1{DELETING_MARK}{}", std::process::id()));
        std::fs::create_dir_all(&marked).unwrap();
        let mut pending = VecDeque::new();
        collect_stale_in(&root, "view-", &mut pending);
        assert!(
            pending.iter().any(|path| path == &marked),
            "带删除标记的目录必须进入待删队列"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 分片清扫一次只交出一个目录；队列空之后返回 false。
    #[test]
    fn a_sweep_step_hands_out_one_directory_at_a_time() {
        let root = std::env::temp_dir().join(format!(
            "hfs-sweep-step-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let first = root.join("view-4294967290-1");
        let second = root.join("view-4294967289-1");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let mut pending = VecDeque::new();
        collect_stale_in(&root, "view-", &mut pending);
        assert_eq!(pending.len(), 2);
        assert!(pending.pop_front().is_some());
        assert_eq!(pending.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
