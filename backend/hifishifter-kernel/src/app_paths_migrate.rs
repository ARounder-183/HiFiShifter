//! 历史布局 → v2 布局的一次性迁移。
//!
//! 【为什么迁移必须自己做、不能"就地兼容"】旧布局把数据放在
//! `%APPDATA%\com.arounder.hifishifter\HiFiShifter»（设置）与
//! `%LOCALAPPDATA%\com.arounder.hifishifter\hifishifter»（缓存 + 本机数据）下。
//! 永远兼容旧路径意味着**每个新用户**也要为那段历史付代价：目录里永远有一串反向
//! 域名。搬一次、留个标记，之后所有代码只需要认识一套路径。
//!
//! 【为什么"绝不合并"】把两个目录的同类文件合成一个的结果是无法预测的：
//! 谁覆盖谁、某份设置是新还是旧、两边的 PCM 谁是孤儿 —— 这些问题都没有正确答案，
//! 而**猜错就是丢用户数据**。因此只要新旧同时存在，就原样不动并记一条 warn：
//! 用户看到的仍然是"设置回到了默认"，但至少没有任何东西被悄悄丢掉。
//!
//! 【为什么先 rename】同卷 rename 是原子的、瞬时完成的（152 MB 的 PCM 缓存也是 0 秒）。
//! 逐文件 copy 只作为跨卷 / 被占用的退化路径。
//!
//! 【为什么要原子锁】App 与插件可能**同时**启动（用户在 DAW 里打开插件，而 App 正
//! 好在跑）。两个进程同时搬同一个目录，轻则半途失败，重则两份都不完整。锁用
//! `create_dir» 的原子性，抢不到的一方等对方写完标记后按"已完成"处理。

use crate::app_paths::{self, LegacyRoot};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// 迁移锁目录名（放在系统临时目录里：**不能**放在新根下，否则会提前把新根建出来，
/// 从而触发"新旧并存"的判据）。
const LOCK_DIR_NAME: &str = "hifishifter-layout-v2.migrating";

/// 抢不到锁时等对方完成的最长时间。
const LOCK_WAIT: Duration = Duration::from_secs(2);

/// 残留锁的失效时间（进程崩溃留下的锁不能让后续启动永远按"别人在迁移"处理）。
const LOCK_STALE_AFTER: Duration = Duration::from_secs(30);

/// 一次迁移的结局。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MigrationOutcome {
    /// 标记已存在：之前搬过了（绝大多数启动都是这个）。
    #[default]
    AlreadyDone,
    /// 全新用户：旧布局不存在，无事可做。
    NothingToMigrate,
    /// 全部搬完并写下了标记。
    Migrated,
    /// 搬了一部分；有旧根因"新旧并存"或失败而没搬（下次启动还会重试）。
    PartiallyMigrated,
    /// 新旧根同时存在：一份都没搬（**绝不合并**）。
    SkippedBothExist,
    /// 另一个宿主正在迁移且迟迟没写完标记：本次直接放过，不阻塞启动。
    SkippedLocked,
    /// 连本地数据根都解析不出来（极简环境）：迁移不可能进行。
    Unavailable,
}

impl MigrationOutcome {
    /// 是否需要让调用方知道（全新用户与已完成是静默的）。
    pub fn is_noteworthy(self) -> bool {
        !matches!(self, Self::AlreadyDone | Self::NothingToMigrate)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AlreadyDone => "already-done",
            Self::NothingToMigrate => "nothing-to-migrate",
            Self::Migrated => "migrated",
            Self::PartiallyMigrated => "partially-migrated",
            Self::SkippedBothExist => "skipped-both-exist",
            Self::SkippedLocked => "skipped-locked",
            Self::Unavailable => "unavailable",
        }
    }
}

/// 迁移结果明细（诊断包与日志用；也让单测能断言"到底搬了什么"）。
#[derive(Debug, Default)]
pub struct MigrationReport {
    pub outcome: MigrationOutcome,
    /// 成功搬走的旧根（旧 → 新）。
    pub moved: Vec<(PathBuf, PathBuf)>,
    /// 因"新旧并存"而没搬的旧根。
    pub skipped_both_exist: Vec<PathBuf>,
    /// 搬移失败的旧根与原因。
    pub failed: Vec<(PathBuf, String)>,
}

impl MigrationReport {
    fn new(outcome: MigrationOutcome) -> Self {
        Self {
            outcome,
            ..Self::default()
        }
    }

    /// 把结局写进日志（调用方在日志初始化之后调一次）。
    pub fn log_summary(&self, host: &str) {
        if !self.outcome.is_noteworthy() {
            log::debug!(
                "[layout] {host}: {} (moved={}, skipped={}, failed={})",
                self.outcome.as_str(),
                self.moved.len(),
                self.skipped_both_exist.len(),
                self.failed.len()
            );
            return;
        }
        log::info!(
            "[layout] {host}: {} (moved={}, skipped={}, failed={})",
            self.outcome.as_str(),
            self.moved.len(),
            self.skipped_both_exist.len(),
            self.failed.len()
        );
        for (from, to) in &self.moved {
            log::info!("[layout] moved {} -> {}", from.display(), to.display());
        }
        for path in &self.skipped_both_exist {
            log::warn!(
                "[layout] both the legacy and the current directory exist, left untouched: {}",
                path.display()
            );
        }
        for (path, reason) in &self.failed {
            log::warn!("[layout] failed to migrate {}: {reason}", path.display());
        }
    }
}

/// 执行（或确认已执行）v2 布局迁移。**幂等**，可以在每次启动时无条件调用。
///
/// 调用点必须是两个宿主中**最早**的一处（早于日志初始化、早于任何设置读写）：
/// 日志初始化会创建 `HiFiShifter\logs»，一旦它先跑，旧日志目录就会被判成"新旧并存"。
pub fn ensure_layout_v2() -> MigrationReport {
    let Some(local_root) = app_paths::local_data_dir() else {
        return MigrationReport::new(MigrationOutcome::Unavailable);
    };
    let marker = local_root.join(app_paths::LAYOUT_MARKER);
    let lock = std::env::temp_dir().join(LOCK_DIR_NAME);
    run_migration(&app_paths::legacy_roots(), &marker, &lock)
}

/// 迁移的**纯**实现：根、标记与锁都由调用方给定（单测因此可以在临时目录里
/// 构造任意新旧布局，而不必碰真实用户目录）。
fn run_migration(roots: &[LegacyRoot], marker: &Path, lock: &Path) -> MigrationReport {
    if marker.exists() {
        return MigrationReport::new(MigrationOutcome::AlreadyDone);
    }

    let Some(_guard) = LockGuard::acquire(lock) else {
        // 另一个宿主正在搬：等它把标记写出来。等不到就放过 —— 启动不能被它拖住。
        let outcome = if wait_for_marker(marker, LOCK_WAIT) {
            MigrationOutcome::AlreadyDone
        } else {
            MigrationOutcome::SkippedLocked
        };
        return MigrationReport::new(outcome);
    };

    // 抢到锁之后再确认一次：上一个持锁者可能刚好写完了标记。
    if marker.exists() {
        return MigrationReport::new(MigrationOutcome::AlreadyDone);
    }

    let mut report = MigrationReport::default();
    let pending: Vec<&LegacyRoot> = roots
        .iter()
        .filter(|root| has_content(&root.from))
        .collect();

    if pending.is_empty() {
        // 全新用户：写标记，省掉以后每次启动的探测。
        write_marker(marker);
        report.outcome = MigrationOutcome::NothingToMigrate;
        return report;
    }

    for root in pending {
        match migrate_root(root) {
            RootOutcome::Moved => report.moved.push((root.from.clone(), root.to.clone())),
            RootOutcome::BothExist => report.skipped_both_exist.push(root.from.clone()),
            RootOutcome::Failed(reason) => report.failed.push((root.from.clone(), reason)),
        }
    }

    report.outcome = if report.failed.is_empty() && report.skipped_both_exist.is_empty() {
        // 全部成功（或本来就没有可搬的）：写下标记，此后每次启动都是 AlreadyDone。
        write_marker(marker);
        MigrationOutcome::Migrated
    } else if report.moved.is_empty() && report.failed.is_empty() {
        MigrationOutcome::SkippedBothExist
    } else {
        MigrationOutcome::PartiallyMigrated
    };
    report
}

enum RootOutcome {
    Moved,
    BothExist,
    Failed(String),
}

fn migrate_root(root: &LegacyRoot) -> RootOutcome {
    // 新根已经有内容 → 绝不合并。
    if has_content(&root.to) {
        return RootOutcome::BothExist;
    }
    // 新根存在但**是空的**（例如日志初始化刚建出空目录）：可以直接被 rename 覆盖。
    if root.to.exists() && std::fs::remove_dir(&root.to).is_err() {
        return RootOutcome::BothExist;
    }

    match move_tree(&root.from, &root.to) {
        Ok(()) => {
            apply_redirects(root);
            RootOutcome::Moved
        }
        Err(reason) => RootOutcome::Failed(reason),
    }
}

/// 根内子项的重定向（缓存拆分）。
///
/// 【为什么失败就直接删】这些子项是**派生物**（波形峰值、整段渲染 PCM）：删掉只会
/// 让下次播放慢一点，而"为了搬缓存而让启动失败"是不可接受的取舍。
fn apply_redirects(root: &LegacyRoot) {
    for (name, dest) in &root.redirects {
        let source = root.to.join(name);
        if !source.exists() {
            continue;
        }
        if dest.exists() {
            // 目标缓存已存在：旧的一律丢弃（保留哪一半都不影响正确性，但混在一起会）。
            let _ = std::fs::remove_dir_all(&source);
            continue;
        }
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::rename(&source, dest).is_err() {
            log::warn!(
                "[layout] could not move cache {} -> {}; discarding it (it is rebuilt on demand)",
                source.display(),
                dest.display()
            );
            let _ = std::fs::remove_dir_all(&source);
        }
    }
}

/// 把 `from» 整棵树搬到 `to»。
///
/// 优先同卷 rename（原子、瞬时）；失败时退化为"逐文件拷贝 → 校验 → 删旧"。
fn move_tree(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            copy_tree(from, to).map_err(|copy_error| {
                format!("rename failed ({rename_error}); copy fallback failed: {copy_error}")
            })?;
            remove_tree(from)
        }
    }
}

/// 逐层拷贝（**先拷后删**：任何一步失败都保留旧树，用户数据不会凭空消失）。
fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("create {}: {e}", to.display()))?;
    for entry in std::fs::read_dir(from).map_err(|e| format!("read {}: {e}", from.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let source = entry.path();
        let destination = to.join(entry.file_name());
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        if file_type.is_dir() {
            copy_tree(&source, &destination)?;
        } else {
            std::fs::copy(&source, &destination).map_err(|e| {
                format!(
                    "copy {} -> {}: {e}",
                    source.display(),
                    destination.display()
                )
            })?;
        }
    }
    Ok(())
}

/// 删除一棵树（拷贝成功、校验通过之后才调用）。
fn remove_tree(path: &Path) -> Result<(), String> {
    match std::fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        // 旧树里可能有正在被别的进程打开的文件：迁走的副本已经完整，残留不影响
        // 正确性 —— 只记日志，不让启动失败。
        Err(error) => {
            log::warn!(
                "[layout] copied {} but could not remove the old tree: {error}",
                path.display()
            );
            Ok(())
        }
    }
}

/// 目录是否存在且**非空**。
///
/// 【为什么要判"非空"】日志初始化、Tauri 的目录创建都会提前建出空的新根。把"空目录"
/// 当成"已经有一份数据"，用户就会永远停在"新旧并存"，迁移再也不会发生。
fn has_content(path: &Path) -> bool {
    match std::fs::read_dir(path) {
        Ok(mut entries) => entries.next().is_some(),
        Err(_) => false,
    }
}

fn write_marker(marker: &Path) {
    if let Some(parent) = marker.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let payload = serde_json::json!({
        "from": app_paths::LEGACY_IDENTIFIER,
        "at": chrono::Local::now().to_rfc3339(),
    });
    if let Err(error) = std::fs::write(marker, payload.to_string()) {
        log::warn!(
            "[layout] could not write the migration marker {}: {error}",
            marker.display()
        );
    }
}

fn wait_for_marker(marker: &Path, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if marker.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    marker.exists()
}

/// 原子锁：`create_dir» 成功即持有，Drop 时删除。
struct LockGuard(PathBuf);

impl LockGuard {
    fn acquire(path: &Path) -> Option<Self> {
        for _ in 0..2 {
            match std::fs::create_dir(path) {
                Ok(()) => return Some(Self(path.to_path_buf())),
                Err(_) => {
                    // 崩溃残留：超过 LOCK_STALE_AFTER 视为失效，删掉重试一次。
                    let stale = std::fs::metadata(path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .map(|age| age > LOCK_STALE_AFTER)
                        .unwrap_or(false);
                    if !stale {
                        return None;
                    }
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }
        None
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个测试一个独立沙箱（进程 id + 名字），析构时清理。
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("hfs-layout-migrate-{}-{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn marker(&self) -> PathBuf {
            self.0.join("new").join(app_paths::LAYOUT_MARKER)
        }

        fn lock(&self) -> PathBuf {
            self.0.join("lock")
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// 设置搬过去、缓存被**拆到**缓存根、其余本机数据留在本地根。
    #[test]
    fn migrates_settings_and_splits_cache_from_local_data() {
        let scratch = Scratch::new("split");
        // 【为什么分成两个父目录】真实布局里设置住在 Roaming、本机数据住在 Local；
        // 两者的叶子同名不同大小写（HiFiShifter / hifishifter），放在同一个父目录下
        // 在 Windows 上就是**同一个目录**（文件系统大小写不敏感），测试会测到别的东西。
        let legacy_config = scratch
            .path()
            .join("old-roaming/com.arounder.hifishifter/HiFiShifter");
        let legacy_local = scratch
            .path()
            .join("old-local/com.arounder.hifishifter/hifishifter");
        let new_config = scratch.path().join("new/roaming/HiFiShifter");
        let new_local = scratch.path().join("new/local/HiFiShifter");
        let new_cache = scratch.path().join("new/cache/HiFiShifter");

        write(&legacy_config.join("app_config.json"), b"settings");
        write(&legacy_local.join("pcm/a.raw"), b"pcm");
        write(&legacy_local.join("waveform_peaks_cache/a.hfs"), b"peaks");
        write(&legacy_local.join("render_cache/a.hfr"), b"render");

        let roots = vec![
            LegacyRoot {
                from: legacy_config.clone(),
                to: new_config.clone(),
                redirects: Vec::new(),
            },
            LegacyRoot {
                from: legacy_local.clone(),
                to: new_local.clone(),
                redirects: vec![
                    (
                        "waveform_peaks_cache".to_string(),
                        new_cache.join("waveform_peaks_cache"),
                    ),
                    ("render_cache".to_string(), new_cache.join("render_cache")),
                ],
            },
        ];

        let report = run_migration(&roots, &scratch.marker(), &scratch.lock());
        assert_eq!(report.outcome, MigrationOutcome::Migrated, "{report:?}");
        assert!(new_config.join("app_config.json").is_file(), "设置没搬过去");
        assert!(new_local.join("pcm/a.raw").is_file(), "本机数据没搬过去");
        assert!(
            new_cache.join("waveform_peaks_cache/a.hfs").is_file(),
            "波形缓存应被拆到缓存根"
        );
        assert!(
            new_cache.join("render_cache/a.hfr").is_file(),
            "渲染缓存应被拆到缓存根"
        );
        assert!(!new_local.join("waveform_peaks_cache").exists());
        assert!(
            !legacy_config.exists() && !legacy_local.exists(),
            "旧根应已腾空"
        );
        assert!(scratch.marker().is_file(), "标记没有落下");
    }

    /// 新旧同时存在时**一份都不搬**，且不改动任何一边。
    #[test]
    fn does_not_merge_when_both_roots_exist() {
        let scratch = Scratch::new("both");
        let legacy = scratch.path().join("old/HiFiShifter");
        let current = scratch.path().join("new/HiFiShifter");
        write(&legacy.join("app_config.json"), b"old-settings");
        write(&current.join("app_config.json"), b"new-settings");

        let roots = vec![LegacyRoot {
            from: legacy.clone(),
            to: current.clone(),
            redirects: Vec::new(),
        }];
        let report = run_migration(&roots, &scratch.marker(), &scratch.lock());

        assert_eq!(
            report.outcome,
            MigrationOutcome::SkippedBothExist,
            "{report:?}"
        );
        assert_eq!(
            std::fs::read(legacy.join("app_config.json")).unwrap(),
            b"old-settings"
        );
        assert_eq!(
            std::fs::read(current.join("app_config.json")).unwrap(),
            b"new-settings"
        );
        assert!(!scratch.marker().exists(), "跳过时不该写标记");
    }

    /// 第二次调用必须直接返回 AlreadyDone，且不再动任何文件。
    #[test]
    fn is_idempotent_via_marker() {
        let scratch = Scratch::new("idempotent");
        let legacy = scratch.path().join("old/HiFiShifter");
        let current = scratch.path().join("new/HiFiShifter");
        write(&legacy.join("a.json"), b"x");
        let roots = vec![LegacyRoot {
            from: legacy.clone(),
            to: current.clone(),
            redirects: Vec::new(),
        }];

        assert_eq!(
            run_migration(&roots, &scratch.marker(), &scratch.lock()).outcome,
            MigrationOutcome::Migrated
        );
        // 旧位置又出现一份（老版本重装留下的）：标记在，就不再搬。
        write(&legacy.join("a.json"), b"y");
        assert_eq!(
            run_migration(&roots, &scratch.marker(), &scratch.lock()).outcome,
            MigrationOutcome::AlreadyDone
        );
        assert_eq!(std::fs::read(current.join("a.json")).unwrap(), b"x");
        assert_eq!(std::fs::read(legacy.join("a.json")).unwrap(), b"y");
    }

    /// 局部失败（某项搬不动）不能中断其余项，也不写标记（下次还会重试）。
    #[test]
    fn survives_partial_local_move() {
        let scratch = Scratch::new("partial");
        let good_legacy = scratch.path().join("old/HiFiShifter");
        let good_current = scratch.path().join("new/HiFiShifter");
        write(&good_legacy.join("a.json"), b"x");

        // 「坏的」旧根指向一个不存在也不可读的位置：rename 与 copy 都会失败。
        let broken = scratch.path().join("old/missing");
        let broken_current = scratch.path().join("new/never");

        let roots = vec![
            LegacyRoot {
                from: broken.clone(),
                to: broken_current,
                redirects: Vec::new(),
            },
            LegacyRoot {
                from: good_legacy.clone(),
                to: good_current.clone(),
                redirects: Vec::new(),
            },
        ];
        let report = run_migration(&roots, &scratch.marker(), &scratch.lock());

        // 坏的那条根本不进 pending（目录不存在），好的那条照常搬完。
        assert_eq!(report.outcome, MigrationOutcome::Migrated, "{report:?}");
        assert!(good_current.join("a.json").is_file());
    }

    /// 两个并发调用者：一个搬，另一个要么等标记、要么按各自结局收场，
    /// 但结果必须是"文件完整、没有半成品"。
    #[test]
    fn two_concurrent_callers_do_not_corrupt() {
        let scratch = Scratch::new("concurrent");
        let legacy_config = scratch.path().join("old/config/HiFiShifter");
        let new_config = scratch.path().join("new/roaming/HiFiShifter");
        let legacy_local = scratch.path().join("old/local-data/hifishifter");
        let new_local = scratch.path().join("new/local/HiFiShifter");
        write(&legacy_config.join("app_config.json"), b"settings");
        for index in 0..50 {
            write(&legacy_local.join(format!("pcm/{index}.raw")), b"pcm");
        }

        let roots = vec![
            LegacyRoot {
                from: legacy_config,
                to: new_config.clone(),
                redirects: Vec::new(),
            },
            LegacyRoot {
                from: legacy_local,
                to: new_local.clone(),
                redirects: Vec::new(),
            },
        ];

        let marker = scratch.marker();
        let lock = scratch.lock();
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let roots = roots.clone();
                let marker = marker.clone();
                let lock = lock.clone();
                std::thread::spawn(move || run_migration(&roots, &marker, &lock).outcome)
            })
            .collect();
        let outcomes: Vec<MigrationOutcome> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert!(
            outcomes.iter().any(|outcome| matches!(
                outcome,
                MigrationOutcome::Migrated | MigrationOutcome::AlreadyDone
            )),
            "{outcomes:?}"
        );
        assert!(new_config.join("app_config.json").is_file());
        for index in 0..50 {
            assert!(
                new_local.join(format!("pcm/{index}.raw")).is_file(),
                "第 {index} 个文件丢了"
            );
        }
        assert!(marker.is_file(), "标记没有落下");
    }
}
