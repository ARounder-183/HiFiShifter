//! 清扫失败的**归因**与**计数**：一个可被普通单元测试守住的纯函数模块。
//!
//! 【为什么单独一个文件】`editor/temp_sweep.rs` 用 `#[cfg(not(test))]` 编译 —— 测试进程
//! 绝不该去删真实的用户目录。代价是那个文件里的任何判据都不会出现在 `cargo test` 里，
//! 于是"失败原因全被记成同一个"这种缺陷不会有任何测试变红（同一取舍与
//! `editor::TEMP_SWEEP_TARGETS` 的注释所述一致）。所以"怎么归因、怎么去重计数"这层
//! **判定**放在这里：它不碰文件系统，只做字符串与数字，可以在测试里直接调用。
//!
//! 【它回答什么问题】现场读到的是 `webview profiles swept: reclaimed=0 failed=234`，
//! 而磁盘上只有 61 个目录。一个既没有归因、又数着"尝试次数"的计数器，下一轮只能靠猜：
//! 是目录被占用？权限不足？还是路径太长？三者的处置完全不同 —— 只有"占用"会随着
//! 进程退出自己好起来。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 一次删除尝试的结局。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Removal {
    /// 目录已经不在磁盘上了（删掉了，或本来就没有）。
    Reclaimed,
    /// 目录被占用（WebView2 还在里面开着文件）。下一轮重试。
    Busy,
    /// 权限不足。重试不会变好。
    Denied,
    /// 路径过长。
    PathTooLong,
    /// 其它 I/O 错误，带原始错误码。
    Other(i32),
}

impl Removal {
    /// 日志与汇总里的稳定短名。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Reclaimed => "reclaimed",
            Self::Busy => "busy",
            Self::Denied => "denied",
            Self::PathTooLong => "path-too-long",
            Self::Other(_) => "other",
        }
    }

    /// 括号里的细节：`busy` / `other (os error 1234)`。
    pub(crate) fn detail(self) -> String {
        match self {
            Self::Other(code) => format!("other (os error {code})"),
            other => other.as_str().to_string(),
        }
    }
}

/// "目录被占用"的操作系统错误码。
///
/// 【为什么按平台分开】Windows 是 `ERROR_SHARING_VIOLATION`(32) /
/// `ERROR_LOCK_VIOLATION`(33)，Unix 是 `EBUSY`(16) / `ETXTBSY`(26)。名字判错只会让诊断行
/// 多一个 `other`，不会改变任何删除行为 —— 判据始终是"删不掉就留着"。
#[cfg(windows)]
const OS_BUSY: [i32; 2] = [32, 33];
#[cfg(not(windows))]
const OS_BUSY: [i32; 2] = [16, 26];

/// "路径过长"的操作系统错误码。
#[cfg(windows)]
const OS_PATH_TOO_LONG: i32 = 206;
#[cfg(target_os = "macos")]
const OS_PATH_TOO_LONG: i32 = 63;
#[cfg(all(unix, not(target_os = "macos")))]
const OS_PATH_TOO_LONG: i32 = 36;

/// 把一次 `io::Error` 翻译成一个可归因的结局。
pub(crate) fn classify_io(error: &std::io::Error) -> Removal {
    // 【"找不到"算成功】目标已经不在磁盘上了 —— 清扫的目的已经达到。把它记成失败
    // 会让下一轮继续为它排队。
    if error.kind() == std::io::ErrorKind::NotFound {
        return Removal::Reclaimed;
    }
    if error.kind() == std::io::ErrorKind::PermissionDenied {
        return Removal::Denied;
    }
    if let Some(code) = error.raw_os_error() {
        if OS_BUSY.contains(&code) {
            return Removal::Busy;
        }
        if code == OS_PATH_TOO_LONG {
            return Removal::PathTooLong;
        }
        return Removal::Other(code);
    }
    Removal::Other(0)
}

/// 这一轮"删不掉"的目录：**按路径去重**，每条带上原因。
#[derive(Default)]
pub(crate) struct FailureTally {
    by_path: BTreeMap<PathBuf, Removal>,
}

impl FailureTally {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_path.is_empty()
    }

    /// 有多少个目录没删掉（不是"试了多少次"）。
    pub(crate) fn len(&self) -> usize {
        self.by_path.len()
    }

    pub(crate) fn clear(&mut self) {
        self.by_path.clear();
    }

    /// 记账一次失败；返回"是否要为它落一行带路径与原因的日志"。
    ///
    /// 【为什么按路径去重】同一个目录在一轮里会失败很多次（每次重扫都把它重新排进
    /// 队列），于是计数会涨到远超磁盘上的目录数 —— 现场 61 个目录读成 `failed=234`。
    /// 计数应当回答"有多少个目录没删掉"，不是"我们试了多少次"。
    ///
    /// 【为什么每种原因只落一行】61 个目录被同一个进程占着时，61 行完全相同的日志
    /// 只会把诊断文件刷满；"第一个例子"已经足够让人认出是哪一个进程。
    pub(crate) fn record(&mut self, path: &Path, removal: Removal) -> Option<String> {
        debug_assert_ne!(
            removal,
            Removal::Reclaimed,
            "reclaimed 不该进失败表 —— 调用方先判掉它"
        );
        if self.by_path.contains_key(path) {
            return None;
        }
        let first_of_its_kind = !self.by_path.values().any(|seen| *seen == removal);
        self.by_path.insert(path.to_path_buf(), removal);
        first_of_its_kind.then(|| {
            format!(
                "[gui] webview sweep: cannot remove {}: {}",
                path.display(),
                removal.detail()
            )
        })
    }

    /// `busy=60, denied=1` —— 按原因汇总这一轮删不掉的目录。
    pub(crate) fn summary(&self) -> String {
        if self.by_path.is_empty() {
            return "no failures".to_string();
        }
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for removal in self.by_path.values() {
            *counts.entry(removal.as_str()).or_default() += 1;
        }
        counts
            .into_iter()
            .map(|(reason, count)| format!("{reason}={count}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 失败必须带得出原因：占用 / 权限 / 路径过长 / 其它错误码 / 已经不在。
    #[cfg(windows)]
    #[test]
    fn a_failed_removal_names_its_reason() {
        use std::io::{Error, ErrorKind};
        assert_eq!(classify_io(&Error::from_raw_os_error(32)), Removal::Busy);
        assert_eq!(classify_io(&Error::from_raw_os_error(33)), Removal::Busy);
        // Windows 的 ERROR_ACCESS_DENIED 走 `PermissionDenied`，不是"其它"。
        assert_eq!(classify_io(&Error::from_raw_os_error(5)), Removal::Denied);
        assert_eq!(
            classify_io(&Error::from_raw_os_error(206)),
            Removal::PathTooLong
        );
        assert_eq!(
            classify_io(&Error::from_raw_os_error(1_234)),
            Removal::Other(1_234)
        );
        // 目标已经不在磁盘上：目的达到了，不该记成一次失败。
        assert_eq!(
            classify_io(&Error::from(ErrorKind::NotFound)),
            Removal::Reclaimed
        );
    }

    /// 非 Windows 上至少要能认出"权限不足"与"已经不在"这两条跨平台判据。
    #[cfg(not(windows))]
    #[test]
    fn the_portable_reasons_are_recognized() {
        use std::io::{Error, ErrorKind};
        assert_eq!(
            classify_io(&Error::from(ErrorKind::PermissionDenied)),
            Removal::Denied
        );
        assert_eq!(
            classify_io(&Error::from(ErrorKind::NotFound)),
            Removal::Reclaimed
        );
    }

    /// **不变量：同一个目录反复失败只算一个目录，每种原因只落一行日志。**
    #[test]
    fn a_repeated_failure_counts_the_directory_once() {
        let mut tally = FailureTally::new();
        let busy = Path::new("view-1");
        let denied = Path::new("view-2");

        assert_eq!(
            tally.record(busy, Removal::Busy).as_deref(),
            Some("[gui] webview sweep: cannot remove view-1: busy"),
            "第一种原因的第一个实例必须落一行带路径的日志"
        );
        // 同一轮里的第二次重试：既不重复计数，也不再打一行。
        assert_eq!(tally.record(busy, Removal::Busy), None);
        // 第二种原因仍然要落一行 —— 它的处置与"占用"完全不同。
        assert_eq!(
            tally.record(denied, Removal::Denied).as_deref(),
            Some("[gui] webview sweep: cannot remove view-2: denied")
        );
        assert_eq!(tally.len(), 2, "两个目录，不是三次尝试");
        assert_eq!(tally.summary(), "busy=1, denied=1");

        tally.clear();
        assert!(tally.is_empty());
        assert_eq!(tally.summary(), "no failures");
    }

    /// 不同的原始错误码各算一条 —— 归因要能指出"是哪个错误"。
    #[test]
    fn distinct_os_errors_stay_distinct() {
        let mut tally = FailureTally::new();
        assert!(tally.record(Path::new("a"), Removal::Other(1)).is_some());
        assert!(tally.record(Path::new("b"), Removal::Other(2)).is_some());
        assert_eq!(tally.len(), 2);
        // 汇总把它们并成一类（都是"其它"），但计数分开不了 —— 明细在各自的日志行里。
        assert_eq!(tally.summary(), "other=2");
    }
}
