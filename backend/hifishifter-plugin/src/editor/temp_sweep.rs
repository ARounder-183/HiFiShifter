//! 本机数据目录里临时子目录的启动清扫。
//!
//! 【为什么需要】`pcm/{namespace}` 与 `webview/{profile}` 都按会话/进程新建目录，只在
//! `Drop` 时清理。宿主被强杀、插件被卸载、进程崩溃时 `Drop` 不跑，这些目录就永久留下
//! —— 实测本机攒到 `webview` 41 个目录约 838 MB、`pcm` 32 个目录约 12 MB，全仓没有任何
//! 回收点。启动时按"属主进程是否还活着"扫一遍，把已退出进程的目录删掉。
//!
//! 【为什么用 PID 而不是 mtime】mtime 会把"长时间开着、但确实在用"的目录一起删掉。
//! 目录名里就带着属主 PID（`view-{pid}-{n}` / `hfs-ui-{pid}-{n}-`），直接问系统更准；
//! 名字不符合格式的历史遗留目录只用年龄兜底。
//!
//! 【为什么删不掉不算错误】目录可能正被另一个进程的 WebView2 占用。`remove_dir_all`
//! 失败就留着 —— 下一轮（下一次启动）会再试，而不是让插件启动失败。

use std::path::Path;
use std::time::{Duration, SystemTime};

/// 名字里认不出属主进程时，超过这个年龄就删。
const STALE_AGE: Duration = Duration::from_secs(7 * 24 * 3600);

/// 进程内只扫一次：清扫会遍历目录并删除文件，放在每个会话创建时跑没有必要。
pub(crate) fn sweep_stale_temp_dirs() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        for (subdir, prefix) in [("webview", "view-"), ("pcm", "hfs-ui-")] {
            sweep_dir(
                &hifishifter_kernel::config_location::local_data_subdir(subdir),
                |name| pid_after_prefix(name, prefix),
            );
        }
    });
}

/// `{prefix}{pid}-…` → pid。
fn pid_after_prefix(name: &str, prefix: &str) -> Option<u32> {
    name.strip_prefix(prefix)?.split('-').next()?.parse().ok()
}

/// 删除 `root` 下所有"属主进程已退出"（或太旧、认不出属主）的子目录。
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
            // 自己的目录当然要留（可能正被本次会话使用）。
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
}
