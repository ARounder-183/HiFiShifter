//! Transfer：在没有 ARA 的宿主里，把宿主播放中的音频录进插件自己的时间轴。
//!
//! # 这个模块解决什么
//!
//! VST3 里**没有**"把音频交给插件"的接口（`pluginterfaces/` 全库检索
//! `capture` / `record` / `freeze` / `transfer` 无结果）。唯一的通路是 `process()`
//! 的输入总线，而工程位置的唯一来源是 `ProcessContext::projectTimeSamples`
//! （SDK 唯一标注 always valid 的时间字段）。所以 Transfer 的全部机制就是：
//!
//! > 在音频回调里把输入总线的样本，按块首的工程位置写进插件自己的时间轴。
//!
//! 用户必须先让宿主播放 —— 插件没有任何办法请求宿主播放（`IComponentHandler`
//! 三个版本里都没有播放请求）。
//!
//! # 分层
//!
//! | 层 | 文件 | 职责 |
//! |---|---|---|
//! | 捕获 | [`capture`] | 音频线程：按工程位置写定长段，零分配、零阻塞 |
//! | 投影 | [`timeline`] | 非音频线程：解决覆盖关系、生成 `TimelineState` |
//! | 存储 | 本文件 | 会话状态、take 管理、渲染输入、诊断读数 |
//!
//! # 为什么素材不是 `EditState`
//!
//! `EditState`（`state_channel.rs`）的键是**宿主 item GUID** 空间的东西，而 Transfer
//! 没有宿主 item。所以素材与 take 由本模块自己持有。参数曲线仍然用 `EditState` ——
//! 它只关心"键是否稳定"，不关心键是谁给的，而 Transfer 的键（`transfer-track` /
//! `transfer-clip-{take}-{n}`）是稳定的。

pub(crate) mod capture;
pub(crate) mod commands;
pub(crate) mod sidecar;
pub(crate) mod timeline;

use crate::render::source::SourcePcm;
use crate::sync::MutexExt;
use capture::{Capture, CaptureOutcome, ClosedReason, InputBlock, Segment};
use hifishifter_kernel::state::TimelineState;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 单段容量（秒）。
///
/// 【为什么是 8 而不是宿主单块上限 6】6 秒是**宿主单块**的上限，段是**容器**。
/// 留 2 秒余量保证任何宿主在任何块大小下都不会刚填满一个段就立刻要切下一个。
const SEGMENT_SECONDS: f64 = 8.0;

/// 常驻备用缓冲个数。
///
/// 【为什么不是 1】音频线程在"封口 + 取下一个"之间需要至少一个可用缓冲；留两个让
/// 非音频线程的补充有一次容错。取不到时停止捕获并记原因，绝不阻塞、绝不分配。
const SPARE_BUFFERS: usize = 2;

/// 单次录制的最长工程时长（秒）的默认值。
///
/// 【为什么需要上限】捕获的 PCM 会一直驻留到 take 被丢弃。4 分钟立体声 96 kHz 已是
/// 184 MB；没有上限时一次误按 Transfer 就能吃掉整个进程的预算。
const DEFAULT_MAX_DURATION_SEC: f64 = 600.0;

/// Transfer 的选项。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TransferOptions {
    /// 单次录制的最长工程时长（秒）。
    pub max_duration_sec: f64,
    /// 单段容量（秒）。
    pub segment_sec: f64,
    /// 录制时是否直通监听（`false` = 输出静音，只听处理结果）。
    pub monitor: bool,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            max_duration_sec: DEFAULT_MAX_DURATION_SEC,
            segment_sec: SEGMENT_SECONDS,
            monitor: true,
        }
    }
}

/// 一次录制的结果。
pub(crate) struct TransferTake {
    pub id: u64,
    pub sample_rate: u32,
    pub channels: u8,
    pub segments: Vec<Segment>,
    pub timeline: TimelineState,
    pub summary: timeline::TimelineSummary,
    /// 段 → 源 PCM，键是 [`timeline::source_id`]。
    pub sources: HashMap<String, Arc<SourcePcm>>,
    /// PCM 的落盘位置（`None` = 没写成，重启后不可恢复）。
    pub sidecar: Option<PathBuf>,
}

/// 录制中的读数（给状态栏）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RecordingStatus {
    /// 已捕获的工程帧数。
    pub captured_frames: u64,
    /// 已封口的段数。
    pub sealed_segments: usize,
    /// 当前工程位置（帧）。
    pub position: i64,
    /// 峰值（左右）。
    pub peak: [f32; 2],
    /// 跳跃次数。
    pub gaps: u64,
    /// 备用缓冲剩余个数（`0` 表示即将停止）。
    pub spares: usize,
}

/// 对外状态。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TransferStatus {
    /// `"empty"` / `"recording"` / `"ready"`。
    pub state: &'static str,
    pub recording: Option<RecordingStatus>,
    pub options: TransferOptions,
    pub take_count: usize,
    pub active_take: Option<u64>,
    /// 宿主给过可用的 `ProcessContext`（没有它就无法对齐时间轴）。
    pub saw_context: bool,
    /// 宿主在输入总线上给过音频。
    pub saw_input: bool,
    /// 宿主走过 `kOffline`（bounce / 导出）。
    pub saw_offline: bool,
}

/// 每个插件实例一份的 Transfer 存储。
///
/// 【为什么是实例级而不是进程级】同一文档上可能有两个实例，一个在录、一个在放。
/// 共享状态会让它们互相覆盖。
pub(crate) struct TransferStore {
    /// 音频线程只读这一个原子就能早退。
    armed: AtomicBool,
    /// 捕获状态。音频线程只 `try_lock` —— 拿不到就跳过这一块并计数，
    /// **绝不阻塞**。非音频线程持有它做补缓冲/收段/promote。
    capture: Mutex<Capture>,
    takes: Mutex<Vec<TransferTake>>,
    active_take: AtomicU64,
    next_take_id: AtomicU64,
    options: Mutex<TransferOptions>,
    /// 宿主在 `setupProcessing` 声明的工程率（`0` = 尚未声明）。
    project_rate: AtomicI32,
    /// 观察位：这三项描述宿主**实际**在做什么，前端据此决定引导语。
    saw_context: AtomicBool,
    saw_input: AtomicBool,
    saw_offline: AtomicBool,
}

impl Default for TransferStore {
    fn default() -> Self {
        Self {
            armed: AtomicBool::new(false),
            capture: Mutex::new(Capture::default()),
            takes: Mutex::new(Vec::new()),
            active_take: AtomicU64::new(0),
            next_take_id: AtomicU64::new(1),
            options: Mutex::new(TransferOptions::default()),
            project_rate: AtomicI32::new(0),
            saw_context: AtomicBool::new(false),
            saw_input: AtomicBool::new(false),
            saw_offline: AtomicBool::new(false),
        }
    }
}

impl TransferStore {
    /// 音频线程早退用的廉价判据。
    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// 活动 take 号（`None` = 还没有 take）。
    ///
    /// 【为什么单独一个访问器】渲染路径用它做版本闸：`prepare()` 由 UI 定时器每 20ms
    /// 驱动一次，而一次合成要好几秒。这个读数必须**不取 takes 锁** —— 那把锁在
    /// 合成期间会被占住，而版本闸恰恰要在那之前问一次。
    pub(crate) fn active_take_id(&self) -> Option<u64> {
        let active = self.active_take.load(Ordering::Acquire);
        (active > 0).then_some(active)
    }

    /// 有没有可渲染的素材（决定 [`SignalSource`](crate::render::extension::SignalSource)）。
    pub(crate) fn has_material(&self) -> bool {
        self.takes.lock_or_recover().iter().any(|take| {
            !take.timeline.clips.is_empty() && take.id == self.active_take.load(Ordering::Acquire)
        })
    }

    /// 记录宿主声明的工程率（`setupProcessing`）。
    pub(crate) fn note_project_rate(&self, rate: f64) {
        if rate.is_finite() && rate > 0.0 {
            self.project_rate
                .store(rate.round() as i32, Ordering::Release);
        }
    }

    /// 记录"宿主走的是离线处理"（bounce / 导出）。
    pub(crate) fn note_offline(&self) {
        self.saw_offline.store(true, Ordering::Release);
    }

    pub(crate) fn options(&self) -> TransferOptions {
        *self.options.lock_or_recover()
    }

    pub(crate) fn set_options(&self, options: TransferOptions) {
        let mut current = self.options.lock_or_recover();
        // 【为什么在这里夹取值而不是信任调用方】上限决定内存占用，段长决定延迟；
        // 两者都不该由前端的一个笔误变成 0 或 1e9。
        *current = TransferOptions {
            max_duration_sec: options.max_duration_sec.clamp(1.0, 3600.0),
            segment_sec: options.segment_sec.clamp(1.0, 60.0),
            monitor: options.monitor,
        };
    }

    /// 开始录制。返回失败原因（宿主没给工程率时无法确定段容量）。
    pub(crate) fn start(&self) -> Result<(), String> {
        if self.is_armed() {
            return Ok(());
        }
        let rate = self.project_rate.load(Ordering::Acquire);
        if rate <= 0 {
            return Err(
                "the host has not declared a sample rate yet; start playback once so \
                 setupProcessing runs"
                    .into(),
            );
        }
        let options = self.options();
        let capacity = ((options.segment_sec * rate as f64).round() as usize).max(1);
        let channels = 2_u8;
        let mut capture = self.capture.lock_or_recover();
        let spares = (0..SPARE_BUFFERS)
            .map(|_| allocate_spare(capacity, channels))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "not enough PCM budget for the transfer buffers".to_owned())?;
        capture.start(channels, capacity, spares);
        drop(capture);
        self.armed.store(true, Ordering::Release);
        Ok(())
    }

    /// 停止录制并 promote 成一个 take。
    pub(crate) fn stop(&self) -> Result<u64, String> {
        self.armed.store(false, Ordering::Release);
        let segments = self.capture.lock_or_recover().finish(ClosedReason::Stopped);
        self.promote(segments)
    }

    /// 取消录制：丢弃本次捕获，不留 take。
    pub(crate) fn cancel(&self) {
        self.armed.store(false, Ordering::Release);
        let mut capture = self.capture.lock_or_recover();
        let _ = capture.finish(ClosedReason::Stopped);
        *capture = Capture::default();
    }

    /// 丢弃全部 take（含落盘副本）。
    pub(crate) fn clear(&self) {
        self.armed.store(false, Ordering::Release);
        *self.capture.lock_or_recover() = Capture::default();
        for take in self.takes.lock_or_recover().drain(..) {
            // 移入回收站而不是直接删：录了几分钟的东西被误删是不可接受的。
            if let Some(path) = take.sidecar {
                sidecar::remove(&path);
            }
        }
        self.active_take.store(0, Ordering::Release);
    }

    /// 组件 state 里的 Transfer 清单。
    ///
    /// 【为什么只存路径而不存 PCM】`MAX_FRAME` 是 64 MiB，而 4 分钟立体声 96 kHz
    /// 已是 184 MB；何况组件 state 会进宿主的每一次保存与撤销快照。清单是几 KB。
    pub(crate) fn manifest(&self) -> serde_json::Value {
        let active = self.active_take.load(Ordering::Acquire);
        let takes = self.takes.lock_or_recover();
        let entries: Vec<_> = takes
            .iter()
            .map(|take| {
                serde_json::json!({
                    "id": take.id,
                    "sidecar": take.sidecar.as_ref().map(|path| path.to_string_lossy()),
                    "active": take.id == active,
                })
            })
            .collect();
        serde_json::json!({ "version": 1, "takes": entries })
    }

    /// 从清单恢复 take。
    ///
    /// 【为什么失败的条目只跳过】一个 sidecar 被用户删了、或在别的机器上不存在，
    /// 不该让**整份**清单失败 —— 其余 take 仍然是好的。逐条降级，与渲染路径同一纪律。
    pub(crate) fn restore_manifest(&self, manifest: &serde_json::Value) -> Result<(), String> {
        let Some(entries) = manifest["takes"].as_array() else {
            return Ok(());
        };
        let mut highest = 0_u64;
        for entry in entries {
            let Some(id) = entry["id"].as_u64() else {
                continue;
            };
            highest = highest.max(id);
            let Some(path) = entry["sidecar"].as_str().map(PathBuf::from) else {
                continue;
            };
            if self.takes.lock_or_recover().iter().any(|take| take.id == id) {
                continue;
            }
            let segments = match sidecar::read(&path) {
                Ok(segments) => segments,
                Err(error) => {
                    log::warn!("[transfer] take {id} could not be restored: {error}");
                    continue;
                }
            };
            let sample_rate = segments[0].sample_rate;
            let channels = segments[0].planes.len().clamp(1, 2) as u8;
            let (timeline, summary) = timeline::build(&segments, id, 120.0);
            let sources = build_sources(id, &segments);
            self.takes.lock_or_recover().push(TransferTake {
                id,
                sample_rate,
                channels,
                segments,
                timeline,
                summary,
                sources,
                sidecar: Some(path),
            });
            if entry["active"].as_bool() == Some(true) {
                self.active_take.store(id, Ordering::Release);
            }
        }
        // take 号分配器必须跳过已恢复的号，否则下一次录制会覆盖一个已存在的文件。
        if highest > 0 {
            self.next_take_id.fetch_max(highest + 1, Ordering::AcqRel);
        }
        if self.active_take.load(Ordering::Acquire) == 0 {
            if let Some(first) = self.takes.lock_or_recover().first().map(|take| take.id) {
                self.active_take.store(first, Ordering::Release);
            }
        }
        Ok(())
    }

    /// 选择一个 take 作为活动 take。
    pub(crate) fn select_take(&self, id: u64) -> Result<(), String> {
        if !self.takes.lock_or_recover().iter().any(|take| take.id == id) {
            return Err(format!("unknown transfer take {id}"));
        }
        self.active_take.store(id, Ordering::Release);
        Ok(())
    }

    /// 音频线程入口：本实例是否正在录制，以及本块要不要直通。
    ///
    /// 【为什么整段逻辑在 `try_lock` 之内】捕获状态的写入必须与"封口/收段"互斥。
    /// 拿不到锁只意味着**非音频线程正持有它**（补缓冲、收段、promote），而那一刻
    /// 我们唯一正确的选择是跳过这一块 —— 阻塞会把这变成 DAW 的音频线程卡顿。
    pub(crate) fn on_block(&self, block: InputBlock<'_>) -> TransferBlockOutcome {
        if !self.is_armed() {
            return TransferBlockOutcome::Idle;
        }
        if block.sample_rate.is_finite() && block.sample_rate > 0.0 {
            self.saw_context.store(true, Ordering::Release);
        }
        if block.has_audio() {
            self.saw_input.store(true, Ordering::Release);
        }
        let Ok(mut capture) = self.capture.try_lock() else {
            // 记一次"没跟上"，但**不改**任何捕获状态。
            return TransferBlockOutcome::Busy;
        };
        if capture.is_stopped() {
            // 捕获已经结束（备用缓冲用尽 / 采样率变化）；等非音频线程 promote。
            self.armed.store(false, Ordering::Release);
            return TransferBlockOutcome::Stopped;
        }
        match capture.on_block(block) {
            CaptureOutcome::Captured | CaptureOutcome::SilentInput => {
                TransferBlockOutcome::Capturing
            }
            CaptureOutcome::NoPosition => TransferBlockOutcome::NoPosition,
            CaptureOutcome::RateChanged | CaptureOutcome::OutOfSpares => {
                self.armed.store(false, Ordering::Release);
                TransferBlockOutcome::Stopped
            }
        }
    }

    /// 非音频线程：补充备用缓冲并收走已封口的段。
    ///
    /// 【为什么由调用方周期性驱动】音频线程不能分配，所以"下一个段用哪个缓冲"必须由
    /// 非音频线程提前备好。这个函数就是那条供给线。它在没有窗口打开时不会被调用 ——
    /// 而录制本来就必须由窗口里的用户动作发起。
    pub(crate) fn top_up(&self) {
        let options = self.options();
        let rate = self.project_rate.load(Ordering::Acquire);
        if rate <= 0 {
            return;
        }
        let capacity = ((options.segment_sec * rate as f64).round() as usize).max(1);
        let mut capture = self.capture.lock_or_recover();
        // 捕获已结束（用尽备用缓冲 / 采样率变化 / 用户停止）：不再补，等 promote。
        if capture.is_stopped() {
            return;
        }
        while capture.spare_count() < SPARE_BUFFERS {
            let Some(spare) = allocate_spare(capacity, 2) else {
                // 预算不足：不静默失败，交给 `status()` 的读数与段结束原因表达。
                return;
            };
            capture.push_spare(spare);
        }
    }

    /// 录制中读数。
    pub(crate) fn recording_status(&self) -> Option<RecordingStatus> {
        if !self.is_armed() {
            return None;
        }
        let capture = self.capture.lock_or_recover();
        let sealed = capture.sealed_len();
        Some(RecordingStatus {
            captured_frames: capture.captured_frames(),
            sealed_segments: sealed,
            position: capture.last_position().unwrap_or(0),
            peak: capture.peak(),
            gaps: capture.gaps,
            spares: capture.spare_count(),
        })
    }

    /// 对外状态。
    pub(crate) fn status(&self) -> TransferStatus {
        let armed = self.is_armed();
        let take_count = self.takes.lock_or_recover().len();
        let active = self.active_take.load(Ordering::Acquire);
        TransferStatus {
            state: if armed {
                "recording"
            } else if take_count > 0 {
                "ready"
            } else {
                "empty"
            },
            recording: self.recording_status(),
            options: self.options(),
            take_count,
            active_take: (active > 0).then_some(active),
            saw_context: self.saw_context.load(Ordering::Acquire),
            saw_input: self.saw_input.load(Ordering::Acquire),
            saw_offline: self.saw_offline.load(Ordering::Acquire),
        }
    }

    /// 活动 take 的渲染输入。
    ///
    /// 返回 `(timeline, regions, sources)` —— 与 ARA 路径**同一组类型**，
    /// 因此下游（渲染、导出、快照）完全共用，不需要为 Transfer 分叉。
    pub(crate) fn render_input(
        &self,
    ) -> Option<(
        TimelineState,
        Vec<crate::ara::AraPlaybackRegion>,
        HashMap<String, Arc<SourcePcm>>,
    )> {
        let active = self.active_take.load(Ordering::Acquire);
        let takes = self.takes.lock_or_recover();
        let take = takes.iter().find(|take| take.id == active)?;
        if take.timeline.clips.is_empty() {
            return None;
        }
        Some((
            take.timeline.clone(),
            regions_for(&take.timeline),
            take.sources.clone(),
        ))
    }

    /// 把封口的段提升成一个 take。
    fn promote(&self, segments: Vec<Segment>) -> Result<u64, String> {
        let mut capture = self.capture.lock_or_recover();
        let mut all = segments;
        all.extend(capture.drain_sealed());
        *capture = Capture::default();
        drop(capture);
        if all.is_empty() {
            return Err("nothing was captured".into());
        }
        let id = self.next_take_id.fetch_add(1, Ordering::AcqRel);
        let sample_rate = all[0].sample_rate;
        let channels = all[0].planes.len().clamp(1, 2) as u8;
        let (timeline, summary) = timeline::build(&all, id, 120.0);
        let sources = build_sources(id, &all);
        // 【为什么在 promote 时就落盘，而不是等 `getState`】落盘是**一次**几十 MB 的
        // 写；放在 `getState` 里就等于在宿主每次自动保存时都写一遍。而 promote 是
        // 用户刚做完的动作，等它有明确的语义（"我刚录完"），且只发生一次。
        let sidecar = sidecar::default_dir().and_then(|dir| {
            match sidecar::write(&dir, id, &all) {
                Ok(path) => Some(path),
                Err(error) => {
                    // 落盘失败不该丢掉这次录制：内存里那份仍然可用，只是重启后不在。
                    log::warn!("[transfer] take {id} was not persisted: {error}");
                    None
                }
            }
        });
        let take = TransferTake {
            id,
            sample_rate,
            channels,
            segments: all,
            timeline,
            summary,
            sources,
            sidecar,
        };
        let mut takes = self.takes.lock_or_recover();
        // 【为什么保留旧 take 而不是覆盖】用户要能在两次录制之间比较。内存由
        // `max_duration_sec` 与全局预算共同约束；显式清理由 `clear()` 承担。
        takes.push(take);
        drop(takes);
        self.active_take.store(id, Ordering::Release);
        Ok(id)
    }
}

/// 本块在 Transfer 通路上的结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferBlockOutcome {
    /// 没在录制。
    Idle,
    /// 已写入。
    Capturing,
    /// 捕获状态被非音频线程占着：本块未写入。
    Busy,
    /// 宿主没给可用的工程位置。
    NoPosition,
    /// 捕获已结束（等 promote）。
    Stopped,
}

/// 分配一个空的段缓冲，并计入全局 PCM 预算。
///
/// 【为什么预留失败就返回 `None` 而不是分配】预算耗尽时分配会让"内存超限"变成
/// 一次 OOM 崩溃，而不是一次可解释的"录制停止"。
fn allocate_spare(capacity: usize, channels: u8) -> Option<Segment> {
    let bytes = capacity.checked_mul(channels as usize)?.checked_mul(4)?;
    let reservation = crate::render::budget::global_budget().reserve(bytes)?;
    Some(Segment {
        id: 0,
        origin: 0,
        sample_rate: 0,
        planes: vec![vec![0.0; capacity]; channels as usize],
        filled: 0,
        loss: capture::SegmentLoss::First,
        peak: [0.0; 2],
        closed: ClosedReason::Full,
        _reservation: Some(reservation),
    })
}

/// 段 → 源 PCM。一个段一份源，键是 [`timeline::source_id`]。
fn build_sources(
    take_id: u64,
    segments: &[Segment],
) -> HashMap<String, Arc<SourcePcm>> {
    let mut sources = HashMap::new();
    for segment in segments {
        let id = timeline::source_id(take_id, segment.id);
        // 【为什么按段而不是按 take 建源】段的采样率与内容各不相同，且时间线里的
        // clip 源窗是**段内偏移** —— 一份源对应一个段，窗口语义才自洽。
        sources.insert(
            id,
            Arc::new(SourcePcm {
                sample_rate: segment.sample_rate,
                planes: segment.planes.clone(),
                version: segment.id,
                _reservation: None,
            }),
        );
    }
    sources
}

/// 由时间线合成 ARA 形状的播放区间。
///
/// 【为什么复用 `AraPlaybackRegion`】渲染入口（`RenderInput`）的几何校验与帧数计算都
/// 以它为准。Transfer 的片段与它是 1:1 的（1:1 映射、无拉伸、无淡化），因此合成出来
/// 的区间就是精确描述 —— 于是**整条下游渲染路径一行都不用改**。
fn regions_for(timeline: &TimelineState) -> Vec<crate::ara::AraPlaybackRegion> {
    timeline
        .clips
        .iter()
        .filter_map(|clip| {
            let take = clip.active_take();
            let source = take.source_path.clone()?;
            Some(crate::ara::AraPlaybackRegion {
                name: Some(clip.name.clone()),
                audio_source_persistent_id: source,
                audio_modification_persistent_id: clip.id.clone(),
                region_sequence_index: Some(0),
                start_in_modification_time: take.source_start_sec,
                duration_in_modification_time: take.source_end_sec - take.source_start_sec,
                start_in_playback_time: clip.start_sec,
                duration_in_playback_time: clip.length_sec,
                is_timestretch_enabled: false,
                has_content_based_fade_at_head: false,
                has_content_based_fade_at_tail: false,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> TransferStore {
        let store = TransferStore::default();
        store.note_project_rate(48_000.0);
        store
    }

    /// 没声明工程率时不能开始录制：段容量无从确定。
    #[test]
    fn arming_requires_a_declared_project_rate() {
        let store = TransferStore::default();
        assert!(store.start().is_err());
        store.note_project_rate(48_000.0);
        assert!(store.start().is_ok());
    }

    /// 录制 → 停止 → 产生一个带片段的 take。
    #[test]
    fn a_capture_round_trip_produces_a_take_with_clips() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 4096].into_boxed_slice());
        for index in 0..4_i64 {
            store.on_block(InputBlock {
                start: index * 4096,
                frames: 4096,
                sample_rate: 48_000.0,
                planes: [Some(samples), Some(samples)],
            });
        }
        let id = store.stop().unwrap();
        let status = store.status();
        assert_eq!(status.state, "ready");
        assert_eq!(status.active_take, Some(id));
        assert_eq!(status.take_count, 1);
        let (timeline, regions, sources) = store.render_input().expect("有素材");
        assert_eq!(timeline.clips.len(), 1, "连续块应合成一个片段");
        assert_eq!(regions.len(), 1);
        assert_eq!(sources.len(), 1);
        assert!(store.has_material());
    }

    /// 取消录制不留 take。
    #[test]
    fn cancelling_leaves_no_take() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.4_f32; 1024].into_boxed_slice());
        store.on_block(InputBlock {
            start: 0,
            frames: 1024,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
        });
        store.cancel();
        assert_eq!(store.status().take_count, 0);
        assert!(!store.has_material());
    }

    /// 录制前的观察位如实反映"宿主给过什么"。
    #[test]
    fn observation_bits_record_what_the_host_actually_did() {
        let store = store();
        let status = store.status();
        assert!(!status.saw_context && !status.saw_input && !status.saw_offline);
        store.note_offline();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 256].into_boxed_slice());
        store.on_block(InputBlock {
            start: 0,
            frames: 256,
            sample_rate: 48_000.0,
            planes: [Some(samples), None],
        });
        let status = store.status();
        assert!(status.saw_context && status.saw_input && status.saw_offline);
    }

    /// 未录制时 `on_block` 完全不动状态（音频线程上的零成本早退）。
    #[test]
    fn an_unarmed_store_does_nothing_on_a_block() {
        let store = store();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 256].into_boxed_slice());
        assert_eq!(
            store.on_block(InputBlock {
                start: 0,
                frames: 256,
                sample_rate: 48_000.0,
                planes: [Some(samples), None],
            }),
            TransferBlockOutcome::Idle
        );
        assert_eq!(store.status().take_count, 0);
    }

    /// 选项被夹到合法区间：一个笔误不该变成 0 秒或 1e9 秒。
    #[test]
    fn options_are_clamped_to_a_usable_range() {
        let store = TransferStore::default();
        store.set_options(TransferOptions {
            max_duration_sec: -5.0,
            segment_sec: 1e9,
            monitor: false,
        });
        let options = store.options();
        assert!(options.max_duration_sec >= 1.0);
        assert!(options.segment_sec <= 60.0);
        assert!(!options.monitor);
    }

    /// 临时目录（每个用例一个）。
    fn temp_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hfs-xfer-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn captured_segment() -> Segment {
        Segment {
            id: 1,
            origin: 0,
            sample_rate: 48_000,
            planes: vec![vec![0.4_f32; 512], vec![0.4_f32; 512]],
            filled: 512,
            loss: capture::SegmentLoss::First,
            peak: [0.4; 2],
            closed: ClosedReason::Stopped,
            _reservation: None,
        }
    }

    /// **重启后录音仍在**：清单指向 sidecar，恢复后能直接渲染。
    ///
    /// 【为什么这是持久化的唯一验收】落盘的价值全在"下次打开还在"。只测"写了文件"
    /// 是不够的 —— 恢复路径要能把段读回来、重建成时间线、并让 `has_material` 为真，
    /// 否则用户看到的是一个空面板加一个存在于磁盘上的文件。
    #[test]
    fn a_manifest_restores_takes_from_their_sidecars() {
        let dir = temp_dir();
        let path = sidecar::write(&dir, 5, &[captured_segment()]).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "takes": [{ "id": 5, "sidecar": path.to_string_lossy(), "active": true }],
        });
        let store = TransferStore::default();
        store.note_project_rate(48_000.0);
        store.restore_manifest(&manifest).unwrap();
        assert_eq!(store.status().take_count, 1);
        assert_eq!(store.active_take_id(), Some(5));
        assert!(store.has_material(), "恢复之后必须立刻可渲染");
        let (timeline, regions, sources) = store.render_input().expect("恢复的 take 应当可用");
        assert_eq!(timeline.clips.len(), 1);
        assert_eq!(regions.len(), 1);
        assert_eq!(sources.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 一个 sidecar 不见了**只跳过它**，其余 take 照常恢复。
    ///
    /// 【为什么逐条降级】用户可能清了缓存、或把工程拷到了另一台机器。一条缺失不该
    /// 让整份清单失败 —— 那会把"少了一条录音"变成"所有录音都没了"。
    #[test]
    fn a_missing_sidecar_is_skipped_without_losing_the_other_takes() {
        let dir = temp_dir();
        let present = sidecar::write(&dir, 1, &[captured_segment()]).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "takes": [
                { "id": 1, "sidecar": present.to_string_lossy(), "active": true },
                { "id": 2, "sidecar": dir.join("gone.hfx").to_string_lossy(), "active": false },
            ],
        });
        let store = TransferStore::default();
        store.restore_manifest(&manifest).unwrap();
        assert_eq!(store.status().take_count, 1, "只有存在的那个 take 被恢复");
        assert_eq!(store.active_take_id(), Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 恢复后新录的 take 不能复用已存在的号（否则会覆盖一个已有的 sidecar 文件）。
    #[test]
    fn restoring_bumps_the_take_id_allocator_past_the_restored_ids() {
        let dir = temp_dir();
        let path = sidecar::write(&dir, 9, &[captured_segment()]).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "takes": [{ "id": 9, "sidecar": path.to_string_lossy(), "active": true }],
        });
        let store = TransferStore::default();
        store.note_project_rate(48_000.0);
        store.restore_manifest(&manifest).unwrap();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.3_f32; 256].into_boxed_slice());
        store.on_block(InputBlock {
            start: 0,
            frames: 256,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
        });
        let id = store.stop().unwrap();
        assert!(id > 9, "新 take 号必须跳过已恢复的 9，实际 {id}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 清单为空或形状不对时不报错（老工程里根本没有这一段）。
    #[test]
    fn an_empty_or_absent_manifest_is_not_an_error() {
        let store = TransferStore::default();
        assert!(store.restore_manifest(&serde_json::json!({})).is_ok());
        assert!(store
            .restore_manifest(&serde_json::json!({ "version": 1, "takes": [] }))
            .is_ok());
        assert_eq!(store.status().take_count, 0);
    }

    /// 合成的播放区间必须与片段的源窗自洽（否则下游几何校验会拒绝整份）。
    #[test]
    fn synthesized_regions_describe_the_clips_exactly() {
        let store = store();
        store.start().unwrap();
        let samples: &'static [f32] = Box::leak(vec![0.3_f32; 2048].into_boxed_slice());
        for index in 0..3_i64 {
            store.on_block(InputBlock {
                start: index * 2048,
                frames: 2048,
                sample_rate: 48_000.0,
                planes: [Some(samples), Some(samples)],
            });
        }
        store.stop().unwrap();
        let (timeline, regions, sources) = store.render_input().unwrap();
        assert_eq!(regions.len(), timeline.clips.len());
        for (region, clip) in regions.iter().zip(&timeline.clips) {
            assert!((region.start_in_playback_time - clip.start_sec).abs() < 1e-9);
            assert!((region.duration_in_playback_time - clip.length_sec).abs() < 1e-9);
            // 1:1：修改域长度 == 播放域长度。
            assert!(
                (region.duration_in_modification_time - region.duration_in_playback_time).abs()
                    < 1e-9
            );
            assert!(
                sources.contains_key(&region.audio_source_persistent_id),
                "区间引用的源必须存在，否则渲染整份失败"
            );
        }
    }
}
