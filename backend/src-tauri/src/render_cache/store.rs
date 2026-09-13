//! 渲染缓存的磁盘存储层：目录布局、原子写、扫描与清理。
//!
//! 目录布局（`v1` 为格式世代，破坏性变更时切世代整体弃用旧目录）：
//! ```text
//! <root>/v1/rendered/<hash 前 2 位>/<hash>.hsrc
//! <root>/v1/tension/<hash 前 2 位>/<hash>.hsrc
//! <root>/v1/noise/<hash 前 2 位>/<hash>.hsrc
//! ```
//!
//! 分片是为了把单目录条目数控制在千级以内（避免文件系统遍历退化）；
//! 每类变体独立目录是因为失效条件不同（tension 依赖 base + tension 曲线、
//! noise 与 formant 无关），必须能各自独立命中/回收。

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::format::{self, EntryKind, LoadedEntry};

/// 目录世代。
pub const DIR_VERSION: &str = "v1";

const ENTRY_SUFFIX: &str = ".hsrc";
const TMP_SUFFIX: &str = ".hsrc.tmp";

/// 磁盘上的一条缓存条目（扫描结果，只含头部元数据）。
#[derive(Debug, Clone)]
pub struct ScannedEntry {
    pub path: PathBuf,
    pub kind: EntryKind,
    pub bytes: u64,
    pub modified: Option<SystemTime>,
    pub project_id: u64,
    pub sample_rate: u32,
}

impl ScannedEntry {
    /// 文件龄（秒）；无法读取修改时间时返回 `None`。
    pub fn age_secs(&self) -> Option<u64> {
        self.modified
            .and_then(|m| SystemTime::now().duration_since(m).ok())
            .map(|d| d.as_secs())
    }
}

/// 目录扫描统计。
#[derive(Debug, Default, Clone)]
pub struct ScanReport {
    pub entries: Vec<ScannedEntry>,
    /// 无法解析（损坏/半写/旧世代）而被清理的文件数。
    pub dropped_corrupt: u64,
    /// 清理 `*.tmp` 残留的文件数。
    pub dropped_tmp: u64,
}

impl ScanReport {
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }
}

/// 一个世代目录的存储句柄（无状态，可随时构造）。
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// `base` 为缓存根目录（例如 `<app_cache_dir>/hifishifter/render_cache`）。
    pub fn new(base: PathBuf) -> Self {
        Self {
            root: base.join(DIR_VERSION),
        }
    }

    /// 确保三类目录存在。
    pub fn ensure_dirs(&self) -> io::Result<()> {
        for kind in [
            EntryKind::Rendered,
            EntryKind::Tension,
            EntryKind::Noise,
        ] {
            fs::create_dir_all(self.kind_dir(kind))?;
        }
        Ok(())
    }

    fn kind_dir(&self, kind: EntryKind) -> PathBuf {
        self.root.join(kind.dir_name())
    }

    /// 条目文件路径。
    pub fn path_for(&self, kind: EntryKind, hash: u64) -> PathBuf {
        self.kind_dir(kind)
            .join(format!("{:02x}", (hash >> 56) & 0xFF))
            .join(format!("{hash:016x}{ENTRY_SUFFIX}"))
    }

    /// 尝试读取一条条目。
    ///
    /// 任何失败（不存在、截断、校验失败、哈希/采样率错配）都返回 `None`，并且
    /// 把损坏文件删除——缓存的自愈语义是"当作 miss 重渲染"，绝不能把异常
    /// 抛给渲染链路或播放链路。
    pub fn load(
        &self,
        kind: EntryKind,
        hash: u64,
        expected_sample_rate: u32,
        verify_checksum: bool,
    ) -> Option<LoadedEntry> {
        let path = self.path_for(kind, hash);
        let file = File::open(&path).ok()?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        match format::read_entry(
            &mut reader,
            hash,
            expected_sample_rate,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            verify_checksum,
        ) {
            Ok(loaded) => {
                if loaded.header.kind != kind {
                    log::warn!(
                        "[render_cache] kind mismatch at {} (expected {:?}, got {:?}); dropping",
                        path.display(),
                        kind,
                        loaded.header.kind
                    );
                    let _ = fs::remove_file(&path);
                    return None;
                }
                Some(loaded)
            }
            Err(e) => {
                log::warn!(
                    "[render_cache] dropping unreadable entry {}: {e}",
                    path.display()
                );
                let _ = fs::remove_file(&path);
                None
            }
        }
    }

    /// 原子写入一条条目：`<hash>.hsrc.tmp` → `<hash>.hsrc`。
    ///
    /// 返回写入的字节数。崩溃最多留下 `.tmp` 残留，由扫描/启动清理回收。
    pub fn write(
        &self,
        kind: EntryKind,
        hash: u64,
        sample_rate: u32,
        project_id: u64,
        take_id: Option<&str>,
        primary: &[f32],
        secondary: Option<&[f32]>,
    ) -> io::Result<u64> {
        let path = self.path_for(kind, hash);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut tmp_os = path.clone().into_os_string();
        tmp_os.push(TMP_SUFFIX);
        let tmp = PathBuf::from(tmp_os);

        {
            let file = File::create(&tmp)?;
            let mut writer = BufWriter::with_capacity(1024 * 1024, file);
            format::write_entry(
                &mut writer,
                kind,
                sample_rate,
                hash,
                project_id,
                crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
                take_id,
                primary,
                secondary,
            )?;
            writer.flush()?;
        }
        let bytes_written = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

        // Windows 的 rename 不能覆盖已存在目标：同 hash 重写（参数回退等场景）
        // 需要先删旧文件。竞态窗口内旧文件短暂缺失，只会退化为一次 miss。
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        fs::rename(&tmp, &path)?;
        Ok(bytes_written)
    }

    /// 扫描全部条目（读取头部元数据），并顺手清理损坏文件与 `.tmp` 残留。
    ///
    /// 只在统计/容量回收/清理这些低频路径调用；命中热路径只做定点读取。
    pub fn scan(&self) -> ScanReport {
        let mut report = ScanReport::default();
        for kind in [
            EntryKind::Rendered,
            EntryKind::Tension,
            EntryKind::Noise,
        ] {
            self.scan_kind_dir(kind, &mut report);
        }
        report
    }

    fn scan_kind_dir(&self, kind: EntryKind, report: &mut ScanReport) {
        let dir = self.kind_dir(kind);
        let Ok(shards) = fs::read_dir(&dir) else {
            return;
        };
        for shard in shards.flatten() {
            let shard_path = shard.path();
            if !shard_path.is_dir() {
                continue;
            }
            let Ok(files) = fs::read_dir(&shard_path) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if !path.is_file() {
                    continue;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.ends_with(TMP_SUFFIX) {
                    if fs::remove_file(&path).is_ok() {
                        report.dropped_tmp += 1;
                    }
                    continue;
                }
                if !name.ends_with(ENTRY_SUFFIX) {
                    continue;
                }
                let bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
                let modified = file.metadata().ok().and_then(|m| m.modified().ok());
                match File::open(&path)
                    .map_err(io::Error::from)
                    .and_then(|f| format::read_header_only(&mut BufReader::new(f)))
                {
                    Ok((_kind, sample_rate, _frames, _hash, project_id, _pipeline)) => {
                        report.entries.push(ScannedEntry {
                            path,
                            kind,
                            bytes,
                            modified,
                            project_id,
                            sample_rate,
                        });
                    }
                    Err(e) => {
                        log::warn!(
                            "[render_cache] dropping corrupt entry {}: {e}",
                            path.display()
                        );
                        if fs::remove_file(&path).is_ok() {
                            report.dropped_corrupt += 1;
                        }
                    }
                }
            }
        }
    }

    /// 目录当前占用字节数（不读头部，仅元数据；用于写盘线程的水位维护）。
    pub fn total_bytes(&self) -> u64 {
        let mut total = 0u64;
        for kind in [
            EntryKind::Rendered,
            EntryKind::Tension,
            EntryKind::Noise,
        ] {
            total += dir_bytes(&self.kind_dir(kind));
        }
        total
    }

    /// 清空全部条目（含 `.tmp` 残留），返回 `(文件数, 字节数)`。
    pub fn clear_all(&self) -> (u64, u64) {
        let mut files = 0u64;
        let mut bytes = 0u64;
        for kind in [
            EntryKind::Rendered,
            EntryKind::Tension,
            EntryKind::Noise,
        ] {
            let (f, b) = clear_dir(&self.kind_dir(kind));
            files += f;
            bytes += b;
        }
        (files, bytes)
    }

    /// 按谓词删除条目，返回 `(文件数, 字节数)`。
    pub fn clear_matching(
        &self,
        mut predicate: impl FnMut(&ScannedEntry) -> bool,
    ) -> (u64, u64) {
        let report = self.scan();
        let mut files = 0u64;
        let mut bytes = 0u64;
        for entry in report.entries {
            if !predicate(&entry) {
                continue;
            }
            match fs::remove_file(&entry.path) {
                Ok(()) => {
                    files += 1;
                    bytes += entry.bytes;
                }
                Err(e) => {
                    log::warn!(
                        "[render_cache] failed to remove {}: {e}",
                        entry.path.display()
                    );
                }
            }
        }
        (files, bytes)
    }

    /// 按"最旧优先"淘汰，直到总占用回落到 `target_bytes` 以内。
    ///
    /// 返回 `(删除文件数, 释放字节数)`。容量回收是低优先级行为：候选按
    /// 修改时间升序（近似 LRU —— 读取命中会触碰 mtime），只删到水位即止。
    pub fn evict_to(&self, target_bytes: u64) -> (u64, u64) {
        let report = self.scan();
        let mut total = report.total_bytes();
        if total <= target_bytes {
            return (0, 0);
        }
        let mut entries = report.entries;
        entries.sort_by_key(|e| e.modified.unwrap_or(SystemTime::UNIX_EPOCH));

        let mut files = 0u64;
        let mut bytes = 0u64;
        for entry in entries {
            if total <= target_bytes {
                break;
            }
            if fs::remove_file(&entry.path).is_ok() {
                files += 1;
                bytes += entry.bytes;
                total = total.saturating_sub(entry.bytes);
            }
        }
        (files, bytes)
    }

    /// 删除超过 `max_age_secs` 未写入的条目（0 表示不限龄）。
    pub fn prune_older_than(&self, max_age_secs: u64) -> (u64, u64) {
        if max_age_secs == 0 {
            return (0, 0);
        }
        self.clear_matching(|entry| {
            entry
                .age_secs()
                .map(|age| age > max_age_secs)
                .unwrap_or(false)
        })
    }
}

fn dir_bytes(dir: &Path) -> u64 {
    let Ok(shards) = fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0u64;
    for shard in shards.flatten() {
        let path = shard.path();
        if path.is_dir() {
            if let Ok(files) = fs::read_dir(&path) {
                for file in files.flatten() {
                    total += file.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        } else {
            total += shard.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    total
}

fn clear_dir(dir: &Path) -> (u64, u64) {
    let Ok(shards) = fs::read_dir(dir) else {
        return (0, 0);
    };
    let mut files = 0u64;
    let mut bytes = 0u64;
    for shard in shards.flatten() {
        let path = shard.path();
        if path.is_dir() {
            if let Ok(inner) = fs::read_dir(&path) {
                for file in inner.flatten() {
                    let file_path = file.path();
                    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
                    if file_path.is_file() && fs::remove_file(&file_path).is_ok() {
                        files += 1;
                        bytes += size;
                    }
                }
            }
            let _ = fs::remove_dir(&path);
        } else {
            let size = shard.metadata().map(|m| m.len()).unwrap_or(0);
            if fs::remove_file(&path).is_ok() {
                files += 1;
                bytes += size;
            }
        }
    }
    (files, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hifishifter_render_cache_test_{}_{}_{}",
            tag,
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn write_load_round_trip_and_layout() {
        let base = temp_root("roundtrip");
        let store = Store::new(base.clone());
        store.ensure_dirs().expect("dirs");

        let hash = 0xAB12_3456_7890_CDEFu64;
        let primary: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
        let secondary: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.25).collect();

        store
            .write(
                EntryKind::Rendered,
                hash,
                48_000,
                7,
                Some("take-a"),
                &primary,
                Some(&secondary),
            )
            .expect("write");

        // 分片目录名 = hash 高 8 位。
        let path = store.path_for(EntryKind::Rendered, hash);
        assert!(path.to_string_lossy().contains("ab"), "{path:?}");
        assert!(path.exists());

        let loaded = store.load(EntryKind::Rendered, hash, 48_000, true).expect("load");
        assert_eq!(loaded.primary, primary);
        assert_eq!(loaded.secondary.as_deref(), Some(secondary.as_slice()));
        assert_eq!(loaded.header.take_id.as_deref(), Some("take-a"));

        // 工程归属 / 采样率等元数据经"只读头部"的扫描读取（清理与统计用）。
        let report = store.scan();
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].project_id, 7);
        assert_eq!(report.entries[0].sample_rate, 48_000);
        assert!(report.dropped_corrupt == 0 && report.dropped_tmp == 0);

        // 采样率错配 → miss（不会跨采样率误用）。
        assert!(store.load(EntryKind::Rendered, hash, 44_100, true).is_none());
        // 类别错配 → miss 并自愈删除。
        assert!(store.load(EntryKind::Tension, hash, 48_000, true).is_none());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn corrupt_entry_is_dropped_and_reported_as_miss() {
        let base = temp_root("corrupt");
        let store = Store::new(base.clone());
        store.ensure_dirs().expect("dirs");
        let hash = 0x1111_2222_3333_4444u64;
        let primary = vec![0.1f32; 16];
        store
            .write(EntryKind::Rendered, hash, 48_000, 0, None, &primary, None)
            .expect("write");

        let path = store.path_for(EntryKind::Rendered, hash);
        fs::write(&path, b"garbage-not-hsrc").expect("corrupt");

        assert!(store.load(EntryKind::Rendered, hash, 48_000, true).is_none());
        assert!(!path.exists(), "corrupt entry must self-heal (removed)");

        let report = store.scan();
        assert!(report.entries.is_empty());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn evict_to_removes_oldest_first() {
        let base = temp_root("evict");
        let store = Store::new(base.clone());
        store.ensure_dirs().expect("dirs");

        for (i, hash) in [0xAA00u64, 0xBB00, 0xCC00].iter().enumerate() {
            let primary = vec![i as f32; 1024];
            store
                .write(EntryKind::Rendered, *hash, 48_000, 0, None, &primary, None)
                .expect("write");
            // 拉开 mtime，确保淘汰顺序确定。
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        let before = store.total_bytes();
        assert!(before > 0);
        let (files, bytes) = store.evict_to(before / 2);
        assert!(files >= 1);
        assert!(bytes > 0);

        let report = store.scan();
        assert!(report.total_bytes() <= before / 2);
        assert!(report.entries.len() < 3);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn clear_matching_filters_by_project_and_age() {
        let base = temp_root("clear");
        let store = Store::new(base.clone());
        store.ensure_dirs().expect("dirs");

        let primary = vec![0.0f32; 8];
        store
            .write(EntryKind::Rendered, 1, 48_000, 100, None, &primary, None)
            .expect("write");
        store
            .write(EntryKind::Rendered, 2, 48_000, 200, None, &primary, None)
            .expect("write");

        let (files, _) = store.clear_matching(|entry| entry.project_id == 100);
        assert_eq!(files, 1);

        let report = store.scan();
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].project_id, 200);

        // 超龄清理：刚写入的条目不应被删。
        let (files, _) = store.prune_older_than(3_600);
        assert_eq!(files, 0);
        let (files, _) = store.clear_all();
        assert_eq!(files, 1);
        let _ = fs::remove_dir_all(&base);
    }
}
