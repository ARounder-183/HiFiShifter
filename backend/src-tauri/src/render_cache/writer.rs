//! 渲染缓存的写盘线程。
//!
//! # 设计
//! - 渲染线程只做 `send`（`std::sync::mpsc` 为无界队列），**绝不因磁盘 I/O
//!   阻塞渲染**；写盘失败只计入统计并静默降级为"仅内存缓存"。
//! - 单线程串行写：避免多线程随机写造成的磁盘抖动，也让容量水位维护无需锁。
//! - 容量回收与超龄清理在同一线程执行：它们本身就是低频 I/O 行为，串行化后
//!   不会与写入互相踩踏。

use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use super::format::EntryKind;
use super::store::Store;

/// 一个待落盘条目（PCM 已常驻内存，仅等待 I/O）。
pub struct PendingEntry {
    pub kind: EntryKind,
    pub hash: u64,
    pub sample_rate: u32,
    pub project_id: u64,
    pub take_id: Option<String>,
    pub primary: Arc<Vec<f32>>,
    pub secondary: Option<Arc<Vec<f32>>>,
    pub bytes: u64,
}

enum Job {
    /// 写一个条目。
    Write(PendingEntry),
    /// 按"最旧优先"回收至水位以内。
    Evict,
    /// 删除超过给定年龄的条目（0 = 不限龄）。
    Prune { max_age_secs: u64 },
    /// 排空确认（退出前 flush 与清理前同步用）。
    Flush(Sender<()>),
}

/// 写盘线程的作业通道。
///
/// 用 `Mutex` 包裹而非直接静态持有 `Sender`：`Sender` 的 `Sync` 实现随标准库
/// 版本演进，加锁是各版本都成立的稳妥写法（投递频率为"每个 clip 一次"，
/// 锁竞争可忽略）。
static SENDER: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();

fn sender() -> &'static Mutex<Sender<Job>> {
    SENDER.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("render-cache-writer".to_string())
            .spawn(move || writer_loop(rx))
            .expect("spawn render-cache writer thread");
        Mutex::new(tx)
    })
}

/// 投递一个待写条目（不阻塞；线程异常退出时静默丢弃）。
pub fn enqueue(entry: PendingEntry) {
    let tx = sender().lock().unwrap_or_else(|e| e.into_inner());
    if tx.send(Job::Write(entry)).is_err() {
        log::warn!("[render_cache] writer thread is gone; dropping pending entry");
    }
}

/// 请求一次容量回收。
pub fn enqueue_evict() {
    let tx = sender().lock().unwrap_or_else(|e| e.into_inner());
    let _ = tx.send(Job::Evict);
}

/// 请求一次超龄清理。
pub fn enqueue_prune(max_age_secs: u64) {
    let tx = sender().lock().unwrap_or_else(|e| e.into_inner());
    let _ = tx.send(Job::Prune { max_age_secs });
}

/// 等待队列排空（带超时）。返回是否在超时前确认排空。
pub fn flush_blocking(timeout: Duration) -> bool {
    let (ack_tx, ack_rx) = channel::<()>();
    {
        let tx = sender().lock().unwrap_or_else(|e| e.into_inner());
        if tx.send(Job::Flush(ack_tx)).is_err() {
            return false;
        }
    }
    matches!(ack_rx.recv_timeout(timeout), Ok(()))
}

fn writer_loop(rx: std::sync::mpsc::Receiver<Job>) {
    // 延迟初始化：第一次写盘时统计目录占用，之后增量维护。
    let mut total_bytes: Option<u64> = None;

    while let Ok(job) = rx.recv() {
        match job {
            Job::Write(entry) => {
                let snapshot = super::runtime_snapshot();
                if !snapshot.enabled {
                    continue;
                }
                let store = Store::new(snapshot.base_dir.clone());
                let entry_bytes = entry.bytes;
                match store.write(
                    entry.kind,
                    entry.hash,
                    entry.sample_rate,
                    entry.project_id,
                    entry.take_id.as_deref(),
                    entry.primary.as_slice(),
                    entry.secondary.as_deref().map(|v| v.as_slice()),
                ) {
                    Ok(written) => {
                        super::note_stored();
                        let total = total_bytes.get_or_insert_with(|| store.total_bytes());
                        *total = total.saturating_add(written.max(entry_bytes));
                        if snapshot.max_size_bytes > 0 && *total > snapshot.max_size_bytes {
                            // 回收到 90% 水位：避免"每写一条就淘汰一次"的抖动。
                            let target = snapshot.max_size_bytes / 10 * 9;
                            let (files, freed) = store.evict_to(target);
                            if files > 0 {
                                log::warn!(
                                    "[render_cache] evicted {files} entries ({:.1} MB) to respect the {} MB budget",
                                    freed as f64 / (1024.0 * 1024.0),
                                    snapshot.max_size_bytes / (1024 * 1024)
                                );
                            }
                            *total = store.total_bytes();
                        }
                    }
                    Err(e) => {
                        super::note_write_error();
                        log::warn!("[render_cache] write failed ({:?}): {e}", entry.kind);
                        // 写失败（只读目录、磁盘满、权限…）→ 降级为仅内存缓存。
                        // 下次投递仍会尝试，但不再视为致命错误。
                        total_bytes = None;
                    }
                }
            }
            Job::Evict => {
                let snapshot = super::runtime_snapshot();
                let store = Store::new(snapshot.base_dir.clone());
                if snapshot.max_size_bytes > 0 {
                    let (files, freed) = store.evict_to(snapshot.max_size_bytes / 10 * 9);
                    if files > 0 {
                        log::warn!(
                            "[render_cache] eviction removed {files} entries ({:.1} MB)",
                            freed as f64 / (1024.0 * 1024.0)
                        );
                    }
                }
                total_bytes = Some(store.total_bytes());
            }
            Job::Prune { max_age_secs } => {
                let snapshot = super::runtime_snapshot();
                let store = Store::new(snapshot.base_dir.clone());
                let (files, freed) = store.prune_older_than(max_age_secs);
                if files > 0 {
                    log::warn!(
                        "[render_cache] pruned {files} stale entries ({:.1} MB) older than {} days",
                        freed as f64 / (1024.0 * 1024.0),
                        max_age_secs / 86_400
                    );
                }
                total_bytes = Some(store.total_bytes());
            }
            Job::Flush(ack) => {
                let _ = ack.send(());
            }
        }
    }
}
