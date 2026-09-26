//! 渲染缓存持久化（Render Cache Persistence）。
//!
//! 把进程内的整 Clip 渲染结果（[`RenderedClipCacheEntry`] / 张力变体 /
//! 气声噪声 stem）以内容哈希为键落盘，使"重新打开工程"不再需要重新合成
//! 未变更的片段。
//!
//! # 三条铁律（改动本模块前请先读）
//! 1. **音频回调不读盘**：磁盘回退只允许发生在渲染循环里（`commands/playback.rs`
//!    的 `render_background_pass`）。`build_snapshot` 在锁内查内存缓存，任何
//!    在那里引入 I/O 的改动都会卡住音频线程。
//! 2. **内存失效 ≠ 磁盘失效**：[`crate::synth_clip_cache::invalidate_clip_all_caches`]
//!    只清内存。磁盘条目的失效靠**键变化**自然发生（旧文件由容量/超龄回收），
//!    打开工程时的全量失效因此不会摧毁本功能。
//! 3. **写盘不阻塞渲染**：渲染线程只投递（见 [`writer`]），失败静默降级。
//!
//! # 正确性
//! 键由 [`crate::synth_clip_cache::compute_rendered_clip_hash`] 统一产出
//! （源身份 / 渲染器 / 管线指纹 / 时间与源窗口 / 曲线 / formant / Take…），
//! 文件头另存 `param_hash` 与采样率做二次校验：宁可判损坏重渲染，绝不误用。

pub mod format;
mod store;
mod writer;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use format::EntryKind;
use store::Store;

use crate::synth_clip_cache::{
    BreathNoiseCacheEntry, BreathNoiseCacheKey, RenderedClipCacheEntry, RenderedClipCacheKey,
    TensionRenderedClipCacheEntry, TensionRenderedClipCacheKey,
};

/// 缓存写入模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// 渲染完成即异步落盘（默认，最稳健）。
    Immediate,
    /// 退出应用时批量落盘（省 I/O；崩溃会丢未落盘条目）。
    OnExit,
    /// 仅在保存工程时落盘（对 I/O 最敏感的用户的兜底选项）。
    Manual,
}

impl WriteMode {
    fn from_setting(value: &str) -> Self {
        match value {
            "onExit" => Self::OnExit,
            "manual" => Self::Manual,
            _ => Self::Immediate,
        }
    }
}

/// 运行时设置的不可变快照（跨线程传递用）。
#[derive(Debug, Clone)]
pub struct RuntimeSettingsSnapshot {
    pub enabled: bool,
    pub max_size_bytes: u64,
    pub max_age_secs: u64,
    pub min_clip_secs: f64,
    pub min_entry_bytes: u64,
    pub max_entry_bytes: u64,
    pub write_mode: WriteMode,
    pub verify_checksum: bool,
    pub min_free_disk_bytes: u64,
    /// 生效的缓存根目录（已解析 custom / system）。
    pub base_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct Runtime {
    snapshot: RuntimeSettingsSnapshot,
    /// 系统缓存根目录（`app_cache_dir()/hifishifter/render_cache`）。
    system_dir: PathBuf,
}

/// 默认（未初始化/未应用设置时）的保守配置：开启、无上限、系统临时目录。
fn default_runtime() -> Runtime {
    let system_dir = std::env::temp_dir()
        .join("hifishifter")
        .join("render_cache");
    Runtime {
        snapshot: RuntimeSettingsSnapshot {
            enabled: true,
            max_size_bytes: 0,
            max_age_secs: 0,
            min_clip_secs: 0.0,
            min_entry_bytes: 4 * 1024,
            max_entry_bytes: 512 * 1024 * 1024,
            write_mode: WriteMode::Immediate,
            verify_checksum: true,
            min_free_disk_bytes: 0,
            base_dir: system_dir.clone(),
        },
        system_dir,
    }
}

static RUNTIME: OnceLock<RwLock<Runtime>> = OnceLock::new();

fn runtime() -> &'static RwLock<Runtime> {
    RUNTIME.get_or_init(|| RwLock::new(default_runtime()))
}

// ─── 会话统计（进程内，不持久化）───────────────────────────────────────────────

static SESSION_HITS: AtomicU64 = AtomicU64::new(0);
static SESSION_MISSES: AtomicU64 = AtomicU64::new(0);
static SESSION_STORED: AtomicU64 = AtomicU64::new(0);
static SESSION_WRITE_ERRORS: AtomicU64 = AtomicU64::new(0);
static PRUNED_ONCE: AtomicBool = AtomicBool::new(false);
static CURRENT_PROJECT_ID: AtomicU64 = AtomicU64::new(0);

fn note_stored() {
    SESSION_STORED.fetch_add(1, Ordering::Relaxed);
}

fn note_write_error() {
    SESSION_WRITE_ERRORS.fetch_add(1, Ordering::Relaxed);
}

fn note_hit() {
    SESSION_HITS.fetch_add(1, Ordering::Relaxed);
}

fn note_miss() {
    SESSION_MISSES.fetch_add(1, Ordering::Relaxed);
}

// ─── 落盘准入统计 ──────────────────────────────────────────────────────────────

/// 一条渲染产物被拒绝落盘的原因。
///
/// 这类拒绝**曾经完全静默**：产物照常进内存缓存、播放一切正常，只是永远不落盘，
/// 于是每次重开工程都要重新合成一遍 —— 实测一个 465 片段的工程有 37.6% 的片段
/// 落在这个状态里，而日志与 UI 都没有任何线索。把它变成一等公民后，同类问题
/// 可以在一次 pass 的日志里自证。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// 持久化缓存总开关关闭。
    Disabled,
    /// 产物为空（0 帧 / 0 采样率）。
    Empty,
    /// 短于用户设置的时长下限。
    TooShort,
    /// 小于单条大小下限。
    TooSmall,
    /// 超过单条大小上限。
    TooLarge,
    /// 磁盘可用空间低于保留值。
    LowDiskSpace,
}

impl SkipReason {
    /// 全部原因，按固定顺序（统计输出稳定，便于比对与测试）。
    pub const ALL: [SkipReason; 6] = [
        SkipReason::Disabled,
        SkipReason::Empty,
        SkipReason::TooShort,
        SkipReason::TooSmall,
        SkipReason::TooLarge,
        SkipReason::LowDiskSpace,
    ];

    /// 稳定标识（日志 / JSON 键，勿随文案改动）。
    pub fn id(self) -> &'static str {
        match self {
            SkipReason::Disabled => "disabled",
            SkipReason::Empty => "empty",
            SkipReason::TooShort => "tooShort",
            SkipReason::TooSmall => "tooSmall",
            SkipReason::TooLarge => "tooLarge",
            SkipReason::LowDiskSpace => "lowDiskSpace",
        }
    }

    /// 在 [`AdmissionCounters::skipped_by_reason`] 中的下标。
    pub fn index(self) -> usize {
        match self {
            SkipReason::Disabled => 0,
            SkipReason::Empty => 1,
            SkipReason::TooShort => 2,
            SkipReason::TooSmall => 3,
            SkipReason::TooLarge => 4,
            SkipReason::LowDiskSpace => 5,
        }
    }
}

/// 会话级准入计数快照。
///
/// 计数只增不减，调用方用"前后差值"取某一轮的增量 —— 后台渲染会反复重启，
/// 全局累计值直接当每轮结果会互相污染。
#[derive(Debug, Clone, Copy, Default)]
pub struct AdmissionCounters {
    /// 通过准入、已投递给写盘线程的条目数（不代表已成功落盘，见 `session_stored`）。
    pub accepted: u64,
    /// 被拒绝的条目总数。
    pub skipped: u64,
    /// 按 [`SkipReason::ALL`] 顺序的拒绝数。
    pub skipped_by_reason: [u64; SkipReason::ALL.len()],
}

impl AdmissionCounters {
    /// 两快照之差（`self` 为较早的一份）。
    pub fn since(&self, earlier: &AdmissionCounters) -> AdmissionCounters {
        let mut skipped_by_reason = [0u64; SkipReason::ALL.len()];
        for (slot, (now, before)) in skipped_by_reason
            .iter_mut()
            .zip(self.skipped_by_reason.iter().zip(earlier.skipped_by_reason.iter()))
        {
            *slot = now.saturating_sub(*before);
        }
        AdmissionCounters {
            accepted: self.accepted.saturating_sub(earlier.accepted),
            skipped: self.skipped.saturating_sub(earlier.skipped),
            skipped_by_reason,
        }
    }

    /// 形如 `tooShort=175` 的原因分解（仅列出非零项；全零时为空串）。
    pub fn reason_summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for reason in SkipReason::ALL {
            let count = self.skipped_by_reason[reason.index()];
            if count > 0 {
                parts.push(format!("{}={}", reason.id(), count));
            }
        }
        parts.join(" ")
    }
}

static SESSION_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static SESSION_SKIPPED: [AtomicU64; 6] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn note_accepted() {
    SESSION_ACCEPTED.fetch_add(1, Ordering::Relaxed);
}

fn note_skipped(reason: SkipReason) {
    SESSION_SKIPPED[reason.index()].fetch_add(1, Ordering::Relaxed);
}

/// 会话级落盘准入计数（供每轮 pass 取增量、供统计面板展示）。
pub fn admission_counters() -> AdmissionCounters {
    let mut skipped_by_reason = [0u64; SkipReason::ALL.len()];
    let mut skipped = 0u64;
    for (slot, counter) in skipped_by_reason.iter_mut().zip(SESSION_SKIPPED.iter()) {
        let value = counter.load(Ordering::Relaxed);
        *slot = value;
        skipped = skipped.saturating_add(value);
    }
    AdmissionCounters {
        accepted: SESSION_ACCEPTED.load(Ordering::Relaxed),
        skipped,
        skipped_by_reason,
    }
}

/// 已成功落盘的条目数（写盘线程异步累加，与 [`admission_counters`] 的
/// `accepted` 存在天然延迟差）。
pub fn session_stored() -> u64 {
    SESSION_STORED.load(Ordering::Relaxed)
}

// ─── 初始化与设置 ──────────────────────────────────────────────────────────────

/// 注入系统缓存根目录（`lib.rs` 启动时调用；未调用时退化为临时目录）。
pub fn init(system_dir: PathBuf) {
    let mut rt = runtime().write().unwrap_or_else(|e| e.into_inner());
    rt.system_dir = system_dir.clone();
    if rt.snapshot.base_dir != system_dir {
        // 仅在用户尚未自定义目录（即仍是默认值）时跟随系统目录。
        let is_default = rt.snapshot.base_dir == default_runtime().system_dir;
        if is_default {
            rt.snapshot.base_dir = system_dir;
        }
    }
}

/// 应用 UI 设置（`get_ui_settings` / `save_ui_settings` 时调用）。
pub fn apply_settings(settings: &crate::config::RenderCacheSettings) {
    let normalized = settings.normalized();
    let (dir_changed, max_age_secs) = {
        let mut rt = runtime().write().unwrap_or_else(|e| e.into_inner());
        let resolved = match normalized.location.as_str() {
            "custom" => normalized
                .custom_dir
                .as_ref()
                .map(PathBuf::from)
                .filter(|dir| ensure_writable(dir))
                .unwrap_or_else(|| rt.system_dir.clone()),
            _ => rt.system_dir.clone(),
        };
        let dir_changed = resolved != rt.snapshot.base_dir;
        rt.snapshot = RuntimeSettingsSnapshot {
            enabled: normalized.enabled,
            max_size_bytes: normalized.max_size_bytes(),
            max_age_secs: normalized.max_age_secs(),
            min_clip_secs: normalized.min_clip_secs,
            min_entry_bytes: normalized.min_entry_bytes(),
            max_entry_bytes: normalized.max_entry_bytes(),
            write_mode: WriteMode::from_setting(&normalized.write_mode),
            verify_checksum: normalized.verify_checksum,
            min_free_disk_bytes: normalized.min_free_disk_bytes(),
            base_dir: resolved,
        };
        (dir_changed, rt.snapshot.max_age_secs)
    };

    if dir_changed {
        let _ = Store::new(current_dir()).ensure_dirs();
    }
    // 启动后第一次应用设置时做一轮超龄清理（异步，不阻塞 UI）。
    if !PRUNED_ONCE.swap(true, Ordering::AcqRel) {
        writer::enqueue_prune(max_age_secs);
    }
    if normalized.enabled {
        writer::enqueue_evict();
    }
}

/// 是否启用持久化缓存。
pub fn enabled() -> bool {
    runtime_snapshot().enabled
}

/// 生效的缓存根目录。
pub fn current_dir() -> PathBuf {
    runtime_snapshot().base_dir
}

/// 当前运行时快照。
pub fn runtime_snapshot() -> RuntimeSettingsSnapshot {
    runtime()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .snapshot
        .clone()
}

/// 记录当前工程身份（打开/新建/另存工程时调用），用于"清理当前工程"。
pub fn set_current_project_id(id: u64) {
    CURRENT_PROJECT_ID.store(id, Ordering::Relaxed);
}

pub fn current_project_id() -> u64 {
    CURRENT_PROJECT_ID.load(Ordering::Relaxed)
}

/// 由工程路径推导工程身份哈希（0 表示未知）。
///
/// 使用规范化小写路径（分隔符统一）：同一工程在不同大小写/分隔符写法下
/// 命中同一身份；工程移动或重命名后会被视为新工程（可接受——缓存只是
/// 多占一份容量，不会误用）。
pub fn project_id_for_path(path: Option<&str>) -> u64 {
    let Some(path) = path else {
        return 0;
    };
    let normalized: String = path
        .trim()
        .replace('\\', "/")
        .to_lowercase()
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("/");
    if normalized.is_empty() {
        return 0;
    }
    // FNV-1a 64，最高位置 1 以避开"0 = 未知"。
    let mut h: u64 = 14695981039346656037;
    for b in normalized.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h | 1
}

// ─── 读取（仅在渲染循环中调用）─────────────────────────────────────────────────

/// 读取整 Clip 渲染条目。
pub fn load_rendered(
    key: &RenderedClipCacheKey,
    expected_sample_rate: u32,
) -> Option<RenderedClipCacheEntry> {
    let rt = runtime_snapshot();
    if !rt.enabled {
        return None;
    }
    let loaded = Store::new(rt.base_dir.clone()).load(
        EntryKind::Rendered,
        key.param_hash,
        expected_sample_rate,
        rt.verify_checksum,
    );
    let Some(loaded) = loaded else {
        note_miss();
        return None;
    };
    let expected_frames = loaded.header.frames as usize * 2;
    if loaded.primary.len() != expected_frames {
        note_miss();
        return None;
    }
    note_hit();
    Some(RenderedClipCacheEntry {
        pcm_stereo: Arc::new(loaded.primary),
        breath_noise_stereo: loaded.secondary.map(Arc::new),
        frames: loaded.header.frames as u64,
        sample_rate: loaded.header.sample_rate,
        rendered_take_id: loaded.header.take_id,
    })
}

/// 读取 HiFiGAN tension 变体。
pub fn load_tension(
    key: &TensionRenderedClipCacheKey,
    expected_sample_rate: u32,
) -> Option<TensionRenderedClipCacheEntry> {
    let rt = runtime_snapshot();
    if !rt.enabled {
        return None;
    }
    let loaded = Store::new(rt.base_dir.clone()).load(
        EntryKind::Tension,
        key.tension_hash,
        expected_sample_rate,
        rt.verify_checksum,
    );
    let Some(loaded) = loaded else {
        note_miss();
        return None;
    };
    if loaded.primary.len() != loaded.header.frames as usize * 2 {
        note_miss();
        return None;
    }
    note_hit();
    Some(TensionRenderedClipCacheEntry {
        pcm_stereo: Arc::new(loaded.primary),
        frames: loaded.header.frames as u64,
        sample_rate: loaded.header.sample_rate,
        rendered_take_id: loaded.header.take_id,
    })
}

/// 读取独立的气声噪声 stem。
pub fn load_noise(
    key: &BreathNoiseCacheKey,
    expected_sample_rate: u32,
) -> Option<BreathNoiseCacheEntry> {
    let rt = runtime_snapshot();
    if !rt.enabled {
        return None;
    }
    let loaded = Store::new(rt.base_dir.clone()).load(
        EntryKind::Noise,
        key.param_hash,
        expected_sample_rate,
        rt.verify_checksum,
    );
    let Some(loaded) = loaded else {
        note_miss();
        return None;
    };
    if loaded.primary.len() != loaded.header.frames as usize * 2 {
        note_miss();
        return None;
    }
    note_hit();
    Some(BreathNoiseCacheEntry {
        noise_stereo: Arc::new(loaded.primary),
        frames: loaded.header.frames as u64,
        sample_rate: loaded.header.sample_rate,
    })
}

// ─── 写入（异步投递）───────────────────────────────────────────────────────────

/// 投递整 Clip 渲染条目。
pub fn store_rendered(key: &RenderedClipCacheKey, entry: &RenderedClipCacheEntry) {
    let payload_bytes = entry.pcm_stereo.len() as u64 * 4
        + entry
            .breath_noise_stereo
            .as_ref()
            .map(|n| n.len() as u64 * 4)
            .unwrap_or(0);
    let rt = runtime_snapshot();
    if let Err(reason) = admit(&rt, entry.frames, entry.sample_rate, payload_bytes) {
        note_skipped(reason);
        return;
    }
    note_accepted();
    dispatch(
        writer::PendingEntry {
            kind: EntryKind::Rendered,
            hash: key.param_hash,
            sample_rate: entry.sample_rate,
            project_id: current_project_id(),
            take_id: entry.rendered_take_id.clone(),
            primary: entry.pcm_stereo.clone(),
            secondary: entry.breath_noise_stereo.clone(),
            bytes: payload_bytes,
        },
        &rt,
    );
}

/// 投递 HiFiGAN tension 变体。
pub fn store_tension(key: &TensionRenderedClipCacheKey, entry: &TensionRenderedClipCacheEntry) {
    let payload_bytes = entry.pcm_stereo.len() as u64 * 4;
    let rt = runtime_snapshot();
    if let Err(reason) = admit(&rt, entry.frames, entry.sample_rate, payload_bytes) {
        note_skipped(reason);
        return;
    }
    note_accepted();
    dispatch(
        writer::PendingEntry {
            kind: EntryKind::Tension,
            hash: key.tension_hash,
            sample_rate: entry.sample_rate,
            project_id: current_project_id(),
            take_id: entry.rendered_take_id.clone(),
            primary: entry.pcm_stereo.clone(),
            secondary: None,
            bytes: payload_bytes,
        },
        &rt,
    );
}

/// 投递独立的气声噪声 stem。
pub fn store_noise(key: &BreathNoiseCacheKey, entry: &BreathNoiseCacheEntry) {
    let payload_bytes = entry.noise_stereo.len() as u64 * 4;
    let rt = runtime_snapshot();
    if let Err(reason) = admit(&rt, entry.frames, entry.sample_rate, payload_bytes) {
        note_skipped(reason);
        return;
    }
    note_accepted();
    dispatch(
        writer::PendingEntry {
            kind: EntryKind::Noise,
            hash: key.param_hash,
            sample_rate: entry.sample_rate,
            project_id: current_project_id(),
            take_id: None,
            primary: entry.noise_stereo.clone(),
            secondary: None,
            bytes: payload_bytes,
        },
        &rt,
    );
}

/// 落盘准入判定（纯函数，便于测试）。
///
/// # 判据
/// 1. **开关**：关闭即全部拒绝。
/// 2. **空产物**：0 帧 / 0 采样率无意义（文件头也拒绝）。
/// 3. **大小下限**（[`RuntimeSettingsSnapshot::min_entry_bytes`]）：这是准入的
///    主判据。它直接对应"这份产物值不值得占一个文件"，出厂 4 KB。
/// 4. **时长下限**（[`RuntimeSettingsSnapshot::min_clip_secs`]）：**默认 0（关闭）**。
///    时长与渲染成本弱相关、与存储成本强相关，用它当闸门会剔除性价比最高的
///    短片段（见 `config::RenderCacheSettings::min_clip_secs` 的实测数据）。
///    保留此判据只为兼容用户显式的旧偏好。
/// 5. **大小上限 / 磁盘保留空间**：容量保护。
///
/// 注意判据顺序：先便宜且确定的（开关 / 空），再大小，最后才做磁盘探测
/// （`available_disk_bytes` 是一次系统调用）。这样绝大多数拒绝都不付 I/O 成本。
fn admit(
    rt: &RuntimeSettingsSnapshot,
    frames: u64,
    sample_rate: u32,
    payload_bytes: u64,
) -> Result<(), SkipReason> {
    if !rt.enabled {
        return Err(SkipReason::Disabled);
    }
    if sample_rate == 0 || frames == 0 {
        return Err(SkipReason::Empty);
    }
    if rt.min_entry_bytes > 0 && payload_bytes < rt.min_entry_bytes {
        return Err(SkipReason::TooSmall);
    }
    if rt.min_clip_secs > 0.0 && (frames as f64 / sample_rate as f64) < rt.min_clip_secs {
        return Err(SkipReason::TooShort);
    }
    if rt.max_entry_bytes > 0 && payload_bytes > rt.max_entry_bytes {
        return Err(SkipReason::TooLarge);
    }
    if rt.min_free_disk_bytes > 0 {
        if let Some(free) = available_disk_bytes(&rt.base_dir) {
            if free < rt.min_free_disk_bytes {
                return Err(SkipReason::LowDiskSpace);
            }
        }
    }
    Ok(())
}

fn dispatch(entry: writer::PendingEntry, rt: &RuntimeSettingsSnapshot) {
    match rt.write_mode {
        WriteMode::Immediate => writer::enqueue(entry),
        WriteMode::OnExit | WriteMode::Manual => {
            let mut pending = pending_queue().lock().unwrap_or_else(|e| e.into_inner());
            pending.bytes = pending.bytes.saturating_add(entry.bytes);
            pending.entries.push_back(entry);
            // 缓冲上限：避免"退出时批量写"模式在超长会话里吃光内存。
            const MAX_PENDING_BYTES: u64 = 512 * 1024 * 1024;
            while pending.bytes > MAX_PENDING_BYTES {
                match pending.entries.pop_front() {
                    Some(dropped) => pending.bytes = pending.bytes.saturating_sub(dropped.bytes),
                    None => break,
                }
            }
        }
    }
}

struct PendingQueue {
    entries: VecDeque<writer::PendingEntry>,
    bytes: u64,
}

static PENDING: OnceLock<Mutex<PendingQueue>> = OnceLock::new();

fn pending_queue() -> &'static Mutex<PendingQueue> {
    PENDING.get_or_init(|| {
        Mutex::new(PendingQueue {
            entries: VecDeque::new(),
            bytes: 0,
        })
    })
}

/// 把缓冲队列（`onExit` / `manual` 模式）投递给写盘线程。
pub fn flush_pending() {
    let drained: Vec<writer::PendingEntry> = {
        let mut pending = pending_queue().lock().unwrap_or_else(|e| e.into_inner());
        let entries: Vec<_> = pending.entries.drain(..).collect();
        pending.bytes = 0;
        entries
    };
    for entry in drained {
        writer::enqueue(entry);
    }
}

/// 等待写盘队列排空（退出前调用；先投递缓冲队列）。
pub fn flush_blocking(timeout: Duration) -> bool {
    flush_pending();
    writer::flush_blocking(timeout)
}

// ─── 统计与清理 ────────────────────────────────────────────────────────────────

/// 分类统计。
#[derive(Debug, Clone)]
pub struct KindStats {
    pub kind: &'static str,
    pub entries: u64,
    pub bytes: u64,
}

/// 按原因分类的拒绝统计。
#[derive(Debug, Clone)]
pub struct SkipStats {
    /// 稳定标识（见 [`SkipReason::id`]）。
    pub reason: &'static str,
    pub count: u64,
}

/// 缓存统计（面板展示 + 命中率）。
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub enabled: bool,
    pub dir: String,
    pub writable: bool,
    pub total_bytes: u64,
    pub entries: u64,
    pub by_kind: Vec<KindStats>,
    pub session_hits: u64,
    pub session_misses: u64,
    pub session_stored: u64,
    pub session_write_errors: u64,
    /// 本会话通过准入、已投递写盘的条目数。
    pub session_accepted: u64,
    /// 本会话被拒绝落盘的条目总数。
    pub session_skipped: u64,
    /// 拒绝原因分解（只含非零项）。
    pub session_skipped_by_reason: Vec<SkipStats>,
    pub max_size_bytes: u64,
    pub max_age_days: u64,
}

/// 清理作用域。
#[derive(Debug, Clone)]
pub enum ClearScope {
    /// 全部条目。
    All,
    /// 仅当前工程。
    CurrentProject,
    /// 超过 `days` 天未写入的条目。
    OlderThan(u64),
    /// 与给定采样率不同的条目（切换音频设备后的回收）。
    OtherSampleRates(u32),
}

/// 清理结果。
#[derive(Debug, Clone, Default)]
pub struct ClearReport {
    pub files: u64,
    pub bytes: u64,
}

/// 采集统计（会扫描目录读取文件头，低频调用）。
pub fn stats() -> CacheStats {
    let rt = runtime_snapshot();
    let store = Store::new(rt.base_dir.clone());
    let report = store.scan();
    let admission = admission_counters();

    let mut rendered = (0u64, 0u64);
    let mut tension = (0u64, 0u64);
    let mut noise = (0u64, 0u64);
    for entry in &report.entries {
        let slot = match entry.kind {
            EntryKind::Rendered => &mut rendered,
            EntryKind::Tension => &mut tension,
            EntryKind::Noise => &mut noise,
        };
        slot.0 += 1;
        slot.1 += entry.bytes;
    }

    CacheStats {
        enabled: rt.enabled,
        dir: rt.base_dir.display().to_string(),
        writable: ensure_writable(&rt.base_dir),
        total_bytes: report.total_bytes(),
        entries: report.entries.len() as u64,
        by_kind: vec![
            KindStats {
                kind: EntryKind::Rendered.display_name(),
                entries: rendered.0,
                bytes: rendered.1,
            },
            KindStats {
                kind: EntryKind::Tension.display_name(),
                entries: tension.0,
                bytes: tension.1,
            },
            KindStats {
                kind: EntryKind::Noise.display_name(),
                entries: noise.0,
                bytes: noise.1,
            },
        ],
        session_hits: SESSION_HITS.load(Ordering::Relaxed),
        session_misses: SESSION_MISSES.load(Ordering::Relaxed),
        session_stored: SESSION_STORED.load(Ordering::Relaxed),
        session_write_errors: SESSION_WRITE_ERRORS.load(Ordering::Relaxed),
        session_accepted: admission.accepted,
        session_skipped: admission.skipped,
        session_skipped_by_reason: SkipReason::ALL
            .iter()
            .filter_map(|reason| {
                let count = admission.skipped_by_reason[reason.index()];
                (count > 0).then_some(SkipStats {
                    reason: reason.id(),
                    count,
                })
            })
            .collect(),
        max_size_bytes: rt.max_size_bytes,
        max_age_days: rt.max_age_secs / 86_400,
    }
}

/// 执行清理。
pub fn clear(scope: ClearScope) -> ClearReport {
    let rt = runtime_snapshot();
    let store = Store::new(rt.base_dir.clone());
    let (files, bytes) = match scope {
        ClearScope::All => store.clear_all(),
        ClearScope::CurrentProject => {
            let id = current_project_id();
            if id == 0 {
                (0, 0)
            } else {
                store.clear_matching(|entry| entry.project_id == id)
            }
        }
        ClearScope::OlderThan(days) => store.prune_older_than(days.saturating_mul(86_400)),
        ClearScope::OtherSampleRates(sample_rate) => {
            store.clear_matching(|entry| entry.sample_rate != sample_rate)
        }
    };
    // 让写盘线程的水位统计与磁盘实际状态重新对齐。
    writer::enqueue_evict();
    ClearReport { files, bytes }
}

// ─── 磁盘空间 ──────────────────────────────────────────────────────────────────

/// 探测 `path` 所在卷的可用字节数（探测失败返回 `None`，调用方按"未知"处理）。
#[cfg(target_os = "windows")]
fn available_disk_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut free: u64 = 0;
    let ok = unsafe { GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), Some(&mut free), None, None) };
    ok.ok()?;
    Some(free)
}

#[cfg(unix)]
fn available_disk_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_bsize as u64)
}

#[cfg(not(any(target_os = "windows", unix)))]
fn available_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

/// 目录是否可写（不存在则尝试创建）。
fn ensure_writable(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".hsrc-probe");
    match std::fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_id_is_stable_and_case_insensitive() {
        let a = project_id_for_path(Some(r"D:\Projects\Song.hshp"));
        let b = project_id_for_path(Some("d:/projects/song.hshp"));
        assert_eq!(a, b);
        assert_ne!(a, 0);
        assert_eq!(project_id_for_path(None), 0);
        assert_eq!(project_id_for_path(Some("   ")), 0);
        assert_ne!(a, project_id_for_path(Some("d:/projects/other.hshp")));
    }

    /// 准入测试用的快照：默认按"出厂值"（时长下限关闭、字节下限 4 KB、不限容量）。
    fn snapshot() -> RuntimeSettingsSnapshot {
        RuntimeSettingsSnapshot {
            enabled: true,
            max_size_bytes: 0,
            max_age_secs: 0,
            min_clip_secs: 0.0,
            min_entry_bytes: 4 * 1024,
            max_entry_bytes: 0,
            write_mode: WriteMode::Immediate,
            verify_checksum: true,
            min_free_disk_bytes: 0,
            base_dir: std::env::temp_dir(),
        }
    }

    /// 参考工程里最短的片段（0.023 s @ 48k 立体声 f32 ≈ 8.8 KB）必须落盘。
    ///
    /// 这条断言是本模块最重要的一条：它的失败意味着"按秒数过滤"回归了 ——
    /// 那正是让一个 465 片段的工程有 175 个片段永远无法落盘的原因。
    #[test]
    fn short_clips_are_persisted_by_default() {
        let rt = snapshot();
        // 0.023 s @ 48k → 1104 帧 → 8.8 KB，远在 4 KB 之上。
        assert_eq!(admit(&rt, 1_104, 48_000, 8_832), Ok(()));
        // 0.25 s @ 48k → 96 KB。
        assert_eq!(admit(&rt, 12_000, 48_000, 96_000), Ok(()));
        // 1 s @ 48k → 384 KB。
        assert_eq!(admit(&rt, 48_000, 48_000, 384_000), Ok(()));
    }

    #[test]
    fn degenerate_entries_are_rejected_as_too_small() {
        let rt = snapshot();
        // 4 KB 下限之下：单帧 / 极短产物不占文件。
        assert_eq!(admit(&rt, 1, 48_000, 8), Err(SkipReason::TooSmall));
        assert_eq!(
            admit(&rt, 100, 48_000, 4 * 1024 - 1),
            Err(SkipReason::TooSmall)
        );
        // 恰好等于下限 → 放行（判据是严格小于）。
        assert_eq!(admit(&rt, 100, 48_000, 4 * 1024), Ok(()));
    }

    #[test]
    fn duration_floor_is_opt_in_only() {
        let mut rt = snapshot();
        rt.min_clip_secs = 0.5;
        // 用户显式设了时长下限时才生效。
        assert_eq!(admit(&rt, 12_000, 48_000, 96_000), Err(SkipReason::TooShort));
        assert_eq!(admit(&rt, 48_000, 48_000, 384_000), Ok(()));
        // 关掉（默认）即不再拒绝短片段。
        rt.min_clip_secs = 0.0;
        assert_eq!(admit(&rt, 12_000, 48_000, 96_000), Ok(()));
    }

    #[test]
    fn entry_size_limit_is_respected() {
        let mut rt = snapshot();
        rt.max_entry_bytes = 1_000;
        rt.min_entry_bytes = 0;
        assert_eq!(admit(&rt, 48_000, 48_000, 384_000), Err(SkipReason::TooLarge));
        assert_eq!(admit(&rt, 48_000, 48_000, 999), Ok(()));
    }

    #[test]
    fn disabled_cache_never_admits() {
        let mut rt = snapshot();
        rt.enabled = false;
        assert_eq!(admit(&rt, 48_000, 48_000, 384_000), Err(SkipReason::Disabled));
    }

    #[test]
    fn empty_payload_is_rejected_before_size_checks() {
        let rt = snapshot();
        assert_eq!(admit(&rt, 0, 48_000, 0), Err(SkipReason::Empty));
        assert_eq!(admit(&rt, 1_000, 0, 8_000), Err(SkipReason::Empty));
    }

    #[test]
    fn admission_counters_diff_and_summarise() {
        let earlier = AdmissionCounters {
            accepted: 10,
            skipped: 3,
            skipped_by_reason: [0, 0, 3, 0, 0, 0],
        };
        let later = AdmissionCounters {
            accepted: 12,
            skipped: 180,
            skipped_by_reason: [0, 0, 175, 5, 0, 0],
        };
        let delta = later.since(&earlier);
        assert_eq!(delta.accepted, 2);
        assert_eq!(delta.skipped, 177);
        assert_eq!(delta.skipped_by_reason[SkipReason::TooShort.index()], 172);
        assert_eq!(delta.skipped_by_reason[SkipReason::TooSmall.index()], 5);
        // 分解只列非零项，且顺序稳定。
        assert_eq!(delta.reason_summary(), "tooShort=172 tooSmall=5");
        // 无跳过时不产出任何噪声。
        assert_eq!(AdmissionCounters::default().reason_summary(), "");
    }
}
