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

use crate::render::snapshot::PlaybackSnapshot;
use crate::render::source::SourcePcm;
use crate::sync::MutexExt;
use capture::{Capture, CaptureOutcome, ClosedReason, Segment};
use timeline::ProjectionTake;
// 【为什么单独 re-export】编辑器会话的测试要按真实形状喂音频块（`on_block`），
// 而 `capture` 是私有子模块。
pub(crate) use capture::InputBlock;
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
    /// 素材覆盖范围**之外**是否直通宿主原声（`false` = 区间外静音）。
    ///
    /// 【为什么默认直通】只录了 10 秒的素材不该把整条轨变成静音。直通让"录过的那一段
    /// 被替换"这件事在听感上可分辨，而不是让用户以为插件把整条轨吃掉了。
    pub outside_passthrough: bool,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            max_duration_sec: DEFAULT_MAX_DURATION_SEC,
            segment_sec: SEGMENT_SECONDS,
            monitor: true,
            outside_passthrough: true,
        }
    }
}

/// 一次录制的结果。
pub(crate) struct TransferTake {
    pub id: u64,
    pub sample_rate: u32,
    pub channels: u8,
    pub segments: Vec<Segment>,
    /// 本次录制自己的摘要（面板的 take 列表用它）。
    ///
    /// 【为什么不留一份时间线】投影是**跨全部 take** 的，单 take 的时间线没有任何
    /// 消费者；而它和 `sources`（每段 PCM 的副本）一起，会让一次录制在内存里存两份
    /// PCM。投影与源表都在读取时按需构造（`with_projection` + `sources_for`）。
    pub summary: timeline::TimelineSummary,
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

/// 录制中的工程几何（给时间轴上的实时录制带）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CaptureSpan {
    /// 已经覆盖到的工程区间（并集）。
    pub start_frame: i64,
    pub end_frame: i64,
    /// 当前块首（= 实时右边缘）。
    pub position_frame: i64,
    pub sample_rate: u32,
}

/// 活动素材的摘要。
///
/// 【为什么需要它】"录到了但时间轴是空的"此前是一句无法归因的现象：可能是宿主没给
/// 位置、可能是输入总线是静音、也可能是宿主压根没播放。这几个数把它们分开：
/// `captured_frames > 0 && clips == 0` 就只可能是"全静音"。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaterialSummary {
    /// 投影出的片段数（全静音区间不产生片段）。
    pub clips: usize,
    /// 捕获到的工程帧数（含静音块）。
    pub captured_frames: u64,
    /// 片段覆盖的工程帧数（不含静音空洞）。
    pub covered_frames: u64,
    /// 整条素材跨越的工程帧数（含空洞）。
    pub spanned_frames: u64,
    pub sample_rate: u32,
}

impl MaterialSummary {
    /// 录到了东西，但没有任何有声内容。
    pub(crate) fn is_silent(&self) -> bool {
        self.captured_frames > 0 && self.clips == 0
    }
}

/// 一个 take 的只读清单条目。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TakeSummary {
    pub id: u64,
    pub active: bool,
    pub clips: usize,
    /// 工程秒跨度（`None` = 没有任何片段）。
    pub span_sec: Option<(f64, f64)>,
}

/// 已 promote 素材的工程跨度（给"录制落在哪"的自动定位）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaterialSpan {
    pub start_frame: i64,
    pub end_frame: i64,
    pub sample_rate: u32,
}

/// 对外状态。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TransferStatus {
    /// `"empty"` / `"recording"` / `"ready"`。
    pub state: &'static str,
    pub recording: Option<RecordingStatus>,
    /// 录制中正在长出来的区间（`None` = 没在录）。
    pub capture: Option<CaptureSpan>,
    /// 活动素材的摘要（`None` = 没有 take）。
    pub material: Option<MaterialSummary>,
    /// 有没有可渲染的素材（与 [`TransferStore::has_material`] 同一口径）。
    pub has_material: bool,
    /// 活动素材的工程跨度（`None` = 没有片段）。
    pub span: Option<MaterialSpan>,
    /// 全部 take 的清单。
    pub takes: Vec<TakeSummary>,
    pub options: TransferOptions,
    pub take_count: usize,
    pub active_take: Option<u64>,
    /// 宿主给过可用的 `ProcessContext`（没有它就无法对齐时间轴）。
    pub saw_context: bool,
    /// 宿主在输入总线上给过音频。
    pub saw_input: bool,
    /// 宿主走过 `kOffline`（bounce / 导出）。
    pub saw_offline: bool,
    /// 宿主**此刻**是否在播放（实时，不是"曾经播过"）。
    pub host_playing: bool,
    /// 宿主在用非 32-bit 浮点跑本实例（插件按设计静音，见字段说明）。
    pub unsupported_precision: bool,
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
    next_take_id: AtomicU64,
    /// "手上有可渲染的素材"的 O(1) 读数。
    ///
    /// 【为什么不能现场投影】`transfer_output()` 在**音频线程**上问这个问题（它决定本块
    /// 该不该出音）。投影要取 takes 锁并遍历全部段 —— 那两件事都不能发生在音频回调里。
    /// 因此这里维护一个由**变更点**（promote / 删除 / 清空 / 恢复）负责的原子读数。
    material_present: AtomicBool,
    /// 素材代次：**任何** take 的变化（新增 / 删除 / 清空 / 恢复）都会推进它。
    ///
    /// 【为什么不能再用"活动 take 号"当版本】投影现在是**跨全部 take** 的
    /// （后录的胜出），于是"删掉一个较早的 take"同样会改变时间线 —— 而 take 号本身
    /// 在那一刻没变。用一个只增的代次表达"素材变过"，渲染版本闸与会话重载才不会漏。
    revision: AtomicU64,
    options: Mutex<TransferOptions>,
    /// 宿主在 `setupProcessing` 声明的工程率（`0` = 尚未声明）。
    project_rate: AtomicI32,
    /// 观察位：这三项描述宿主**实际**在做什么，前端据此决定引导语。
    saw_context: AtomicBool,
    saw_input: AtomicBool,
    saw_offline: AtomicBool,
    /// 宿主**此刻**是否在播放（实时读数，不是"曾经播过"）。
    ///
    /// 【为什么是实时的】按了录制却没在播放，是这个功能最容易踩的坑（Transfer 的用法是
    /// "先按录制、再按播放"）。实时读数让面板能直接说"宿主没有在播放"，而不是让用户对着
    /// 一个 0.00 秒的读数猜。
    host_playing: AtomicBool,
    /// 宿主用**非 32-bit 浮点**（`kSample64`）在跑本实例。
    ///
    /// 【为什么要单独一个读数】那种情况下插件的平面是 64 位的（联合体同一地址），
    /// 我们既不能捕获也不能直通，输出**按设计**是静音。此前这件事只存在于一行日志里，
    /// 而它表现为"插上插件就没声音、也录不到东西" —— 与"宿主没给输入"完全同形。
    /// 它有一个用户可执行的补救：把宿主的处理精度切回 32-bit 浮点。
    unsupported_precision: AtomicBool,
    /// "宿主给了位置但从来没给过输入"这条诊断是否已经报过（一次性）。
    ///
    /// 【为什么要它】这是"插件没声音、也录不到东西"最可能的原因，而它此前**只**表现为
    /// 一条条静音的读数。音频线程不能写日志（文件 IO），所以由非音频的会话 actor
    /// 在心跳里检查并只报一次（见 [`TransferStore::report_missing_input_once`]）。
    missing_input_reported: AtomicBool,
}

impl Default for TransferStore {
    fn default() -> Self {
        Self {
            armed: AtomicBool::new(false),
            capture: Mutex::new(Capture::default()),
            takes: Mutex::new(Vec::new()),
            next_take_id: AtomicU64::new(1),
            material_present: AtomicBool::new(false),
            revision: AtomicU64::new(0),
            options: Mutex::new(TransferOptions::default()),
            project_rate: AtomicI32::new(0),
            saw_context: AtomicBool::new(false),
            saw_input: AtomicBool::new(false),
            saw_offline: AtomicBool::new(false),
            host_playing: AtomicBool::new(false),
            unsupported_precision: AtomicBool::new(false),
            missing_input_reported: AtomicBool::new(false),
        }
    }
}

impl TransferStore {
    /// 音频线程早退用的廉价判据。
    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// 素材代次（`0` = 从来没有过任何 take）。
    ///
    /// 【为什么单独一个访问器】渲染路径用它做版本闸：`prepare()` 由 UI 定时器每 20ms
    /// 驱动一次，而一次合成要好几秒。这个读数必须**不取 takes 锁** —— 那把锁在
    /// 合成期间会被占住，而版本闸恰恰要在那之前问一次。
    pub(crate) fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// 最近一次录制的 take 号（`None` = 还没有 take）。**派生**读数，只给界面用。
    ///
    /// 【为什么是派生的】"活动 take"曾是一个可被用户切换的存储字段，而那时它决定
    /// 渲染什么 —— 于是补录一句会把整条时间轴换掉。现在投影取全部 take，"最新一次录制"
    /// 就只是 take 号里最大的那个，不需要（也不该）有一个独立的可变状态。
    pub(crate) fn latest_take(&self) -> Option<u64> {
        self.takes.lock_or_recover().iter().map(|take| take.id).max()
    }

    /// 有没有可渲染的素材（决定 [`SignalSource`](crate::render::extension::SignalSource)）。
    ///
    /// 【为什么是"任意一个 take"】投影跨全部 take，所以任何一次录制都可能贡献内容。
    pub(crate) fn has_material(&self) -> bool {
        self.material_present.load(Ordering::Acquire)
    }

    /// 重算 [`Self::has_material`]（**只允许非音频线程**：它要取 takes 锁）。
    fn refresh_material_present(&self) {
        let present = self
            .takes
            .lock_or_recover()
            .iter()
            .any(|take| take.summary.clips > 0);
        self.material_present.store(present, Ordering::Release);
    }

    /// 在持锁期间投影：`f` 拿到跨全部 take 的投影输入。
    ///
    /// 【为什么把闭包而不是 `Vec<ProjectionTake>` 交出去】段是几十 MB 的 PCM 缓冲，
    /// 投影不该克隆它们；而"借用"要求 takes 锁在投影期间活着。闭包是唯一能同时表达
    /// 这两件事的形状：输入的生命周期被关在调用内部，逃不出去。
    pub(crate) fn with_projection<T>(&self, f: impl FnOnce(&[ProjectionTake<'_>]) -> T) -> T {
        let takes = self.takes.lock_or_recover();
        let input: Vec<ProjectionTake<'_>> = takes
            .iter()
            .map(|take| ProjectionTake {
                id: take.id,
                segments: &take.segments,
            })
            .collect();
        f(&input)
    }

    /// 记录本块的样本精度（`ProcessData::symbolic_sample_size`）。
    pub(crate) fn note_precision(&self, symbolic_sample_size: i32) {
        self.unsupported_precision
            .store(symbolic_sample_size != 0, Ordering::Release);
    }

    /// 宿主是否在用非 32-bit 浮点（那种情况下本插件按设计静音，见字段说明）。
    pub(crate) fn precision_unsupported(&self) -> bool {
        self.unsupported_precision.load(Ordering::Acquire)
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
            outside_passthrough: options.outside_passthrough,
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
        self.material_present.store(false, Ordering::Release);
        self.bump_revision();
    }

    /// 组件 state 里的 Transfer 清单。
    ///
    /// 【为什么只存路径而不存 PCM】组件 state 的预算是 64 MiB，而 4 分钟立体声 96 kHz
    /// 已是 184 MB；何况组件 state 会进宿主的每一次保存与撤销快照。清单是几 KB。
    pub(crate) fn manifest(&self) -> serde_json::Value {
        let takes = self.takes.lock_or_recover();
        let latest = takes.iter().map(|take| take.id).max();
        let entries: Vec<_> = takes
            .iter()
            .map(|take| {
                serde_json::json!({
                    "id": take.id,
                    "sidecar": take.sidecar.as_ref().map(|path| path.to_string_lossy()),
                    // 【为什么保留这个字段】它是清单格式的一部分（老工程里有），
                    // 语义是"最近一次录制"。读回时只用于显示，不再决定渲染什么。
                    "active": Some(take.id) == latest,
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
        let mut restored = 0_usize;
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
            let (_, summary) = timeline::build(&segments, id, 120.0);
            self.takes.lock_or_recover().push(TransferTake {
                id,
                sample_rate,
                channels,
                segments,
                summary,
                sidecar: Some(path),
            });
            restored += 1;
        }
        // take 号分配器必须跳过已恢复的号，否则下一次录制会覆盖一个已存在的文件。
        if highest > 0 {
            self.next_take_id.fetch_max(highest + 1, Ordering::AcqRel);
        }
        if restored > 0 {
            // 【为什么要推进代次】恢复改变了素材 ⇒ 时间线与渲染都必须重来。不推进的话
            // 一个刚打开的工程会停在"没有素材"的状态上，直到用户做一次别的动作。
            self.refresh_material_present();
            self.bump_revision();
        }
        Ok(())
    }

    /// 删掉一个 take。
    ///
    /// 【为什么落盘副本进回收站而不是直接删】录了几分钟的东西被误删是不可接受的，
    /// 与 [`Self::clear`] 同一处置。
    ///
    /// 【为什么没有"选择 take"】投影跨全部 take（后录的胜出），所以"哪一个是活动的"
    /// 不再决定任何事 —— 保留一个不决定行为的开关只会让用户以为它有用。要换一条录音
    /// 的听感，就删掉盖住它的那一次。
    pub(crate) fn delete_take(&self, id: u64) -> Result<(), String> {
        let mut takes = self.takes.lock_or_recover();
        let Some(index) = takes.iter().position(|take| take.id == id) else {
            return Err(format!("unknown transfer take {id}"));
        };
        let take = takes.remove(index);
        drop(takes);
        if let Some(path) = take.sidecar {
            sidecar::remove(&path);
        }
        self.refresh_material_present();
        self.bump_revision();
        Ok(())
    }

    /// 推进素材代次（任何改变 take 集合的动作都要调它）。
    fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    /// 音频线程入口：本实例是否正在录制，以及本块要不要直通。
    ///
    /// 【为什么整段逻辑在 `try_lock` 之内】捕获状态的写入必须与"封口/收段"互斥。
    /// 拿不到锁只意味着**非音频线程正持有它**（补缓冲、收段、promote），而那一刻
    /// 我们唯一正确的选择是跳过这一块 —— 阻塞会把这变成 DAW 的音频线程卡顿。
    /// 观察位：宿主**实际**做了什么。与是否武装**无关**。
    ///
    /// 【为什么必须与武装解耦 —— 这是 Transfer 曾经在所有宿主上都用不了的原因】
    /// 面板的录制按钮由 `canCapture`（= `saw_context`）门控，而 `saw_context` 此前只在
    /// **武装之后**的 `on_block` 里更新 —— 于是"必须先录一次才能录"：按钮永远是灰的，
    /// 用户根本没有办法开始。观察的代价只是几个原子写 + 一次缓冲形状检查
    /// （`capture_view` 不分配、不读样本），可以在每个音频块上做。
    ///
    /// 【为什么播放状态是实时的】见 `host_playing` 字段。
    pub(crate) fn observe(&self, block: &InputBlock<'_>, playing: bool) {
        if block.sample_rate.is_finite() && block.sample_rate > 0.0 {
            self.saw_context.store(true, Ordering::Release);
        }
        if block.has_audio() {
            self.saw_input.store(true, Ordering::Release);
        }
        self.host_playing.store(playing, Ordering::Release);
    }
    /// 只报一次"宿主给了位置、却从来没有给过输入"。
    ///
    /// 【为什么由非音频线程调用】它要写日志（文件 IO），音频线程不允许。
    /// 【为什么值得单独一条日志】这条现象此前在界面上只是"什么都不动"：录制读数停在
    /// 0.0 秒、时间轴空白、播放无声。把结论写进日志，用户（或我们）不必再去猜
    /// 是"没位置"、"没输入"还是"没播放"。
    pub(crate) fn report_missing_input_once(&self) {
        if self.saw_input.load(Ordering::Acquire) || !self.saw_context.load(Ordering::Acquire) {
            return;
        }
        if self.missing_input_reported.swap(true, Ordering::AcqRel) {
            return;
        }
        log::warn!(
            "[transfer] the host supplies a playhead but this instance has never received an              audio input bus; Transfer cannot record anything and the plug-in has nothing to              pass through. Put the plug-in on the track whose audio you want, and check that              the host routes that track into it."
        );
    }

    pub(crate) fn on_block(&self, block: InputBlock<'_>) -> TransferBlockOutcome {
        if !self.is_armed() {
            return TransferBlockOutcome::Idle;
        }
        // 观察不在这里做：它必须在**未武装**时也发生（见 `observe`），所以由每个音频块的
        // 唯一入口 `transfer_stage` 负责。这里只管捕获。
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
    /// 非音频线程提前备好。这个函数就是那条供给线。
    ///
    /// 【为什么未武装时直接返回】补缓冲只服务于"正在录制"这件事：`start()` 已经同步
    /// 备好了第一批。没有这道早退，一个空闲的 Transfer 实例会在每一轮心跳里分配
    /// `SPARE_BUFFERS` 粒缓冲（2 × 8 秒立体声 ≈ 6 MB）并永久持有 —— 而它什么都没在录。
    pub(crate) fn top_up(&self) {
        if !self.is_armed() {
            return;
        }
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

    /// 录制中的工程几何（给时间轴上的实时录制带）。
    ///
    /// 【为什么单独一个访问器】它是"录制正在长出来"这件事的唯一读数来源。没有它，
    /// 用户在录制过程中看不到任何东西 —— 而"不知道自己录了什么"正是这条通路最常见的
    /// 失败感（见 `docs/plans/2026-10-15-vst-transfer-monitor-and-ux.md`）。
    pub(crate) fn capture_span(&self) -> Option<CaptureSpan> {
        if !self.is_armed() {
            return None;
        }
        let capture = self.capture.lock_or_recover();
        let span = capture.span()?;
        let sample_rate = capture.sample_rate();
        Some(CaptureSpan {
            start_frame: span.start,
            end_frame: span.end,
            position_frame: capture.last_position().unwrap_or(span.end),
            sample_rate,
        })
    }

    /// 素材的摘要（跨全部 take 的**投影**结果）。
    ///
    /// 【为什么片段数取投影而不是逐 take 相加】用户看到的是投影后的时间线：被补录盖住
    /// 的部分已经不在那里了。逐 take 相加会报出一个用户看不见的数字。
    pub(crate) fn material_summary(&self) -> Option<MaterialSummary> {
        self.with_projection(|input| {
            if input.iter().all(|take| take.segments.is_empty()) {
                return None;
            }
            let captured_frames: u64 = input
                .iter()
                .flat_map(|take| take.segments.iter())
                .map(|segment| segment.filled as u64)
                .sum();
            let sample_rate = input
                .iter()
                .flat_map(|take| take.segments.iter())
                .map(|segment| segment.sample_rate)
                .max()
                .unwrap_or(0);
            let (_, summary) = timeline::project(input, 120.0);
            Some(MaterialSummary {
                clips: summary.clips,
                captured_frames,
                covered_frames: summary.covered_frames,
                spanned_frames: summary.spanned_frames,
                sample_rate,
            })
        })
    }

    /// 全部 take 的只读清单（面板的 take 列表与"录制在哪"的定位都用它）。
    pub(crate) fn take_summaries(&self) -> Vec<TakeSummary> {
        let takes = self.takes.lock_or_recover();
        let latest = takes.iter().map(|take| take.id).max();
        takes
            .iter()
            .map(|take| {
                let rate = take.sample_rate.max(1) as f64;
                let span = take.summary.spanned_frames;
                TakeSummary {
                    id: take.id,
                    active: Some(take.id) == latest,
                    clips: take.summary.clips,
                    span_sec: (span > 0).then(|| {
                        let start = take
                            .segments
                            .iter()
                            .map(|segment| segment.origin)
                            .min()
                            .unwrap_or(0);
                        (start as f64 / rate, (start + span as i64) as f64 / rate)
                    }),
                }
            })
            .collect()
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

    /// 素材的工程跨度（用于"录完把视口移到录制区间"）。
    ///
    /// 【为什么是全部 take 的并集】用户问的是"我刚录的那段在哪" —— 每一次录制都算。
    /// 只报最近一次会让补录一句之后的定位跳到那个小句子上，而用户想看的是整段成果。
    pub(crate) fn material_span(&self) -> Option<MaterialSpan> {
        self.with_projection(|input| {
            let mut start = i64::MAX;
            let mut end = i64::MIN;
            let mut sample_rate = 0_u32;
            for segment in input.iter().flat_map(|take| take.segments.iter()) {
                if segment.span().is_empty() || segment.has_nonfinite() {
                    continue;
                }
                start = start.min(segment.origin);
                end = end.max(segment.origin + segment.filled as i64);
                sample_rate = sample_rate.max(segment.sample_rate);
            }
            (start < end).then_some(MaterialSpan {
                start_frame: start,
                end_frame: end,
                sample_rate,
            })
        })
    }

    /// 对外状态。
    pub(crate) fn status(&self) -> TransferStatus {
        let armed = self.is_armed();
        let take_count = self.takes.lock_or_recover().len();
        TransferStatus {
            state: if armed {
                "recording"
            } else if take_count > 0 {
                "ready"
            } else {
                "empty"
            },
            recording: self.recording_status(),
            capture: self.capture_span(),
            material: self.material_summary(),
            has_material: self.has_material(),
            span: self.material_span(),
            takes: self.take_summaries(),
            options: self.options(),
            take_count,
            active_take: self.latest_take(),
            saw_context: self.saw_context.load(Ordering::Acquire),
            saw_input: self.saw_input.load(Ordering::Acquire),
            saw_offline: self.saw_offline.load(Ordering::Acquire),
            host_playing: self.host_playing.load(Ordering::Acquire),
            unsupported_precision: self.precision_unsupported(),
        }
    }

    /// 全部素材的渲染输入。
    ///
    /// 返回 `(timeline, regions, sources)` —— 与 ARA 路径**同一组类型**，
    /// 因此下游（渲染、导出、快照）完全共用，不需要为 Transfer 分叉。
    ///
    /// 【为什么没有可渲染内容时返回 `None`】渲染那条路以"有没有东西可渲染"为判据
    /// （见 `renders()` 与 `transfer_prepare_job` 的早退）。空时间线在那里是"还没录"，
    /// 不是"有一条空轨"。
    pub(crate) fn render_input(
        &self,
    ) -> Option<(
        TimelineState,
        Vec<crate::ara::AraPlaybackRegion>,
        HashMap<String, Arc<SourcePcm>>,
    )> {
        self.with_projection(|input| {
            let (timeline, _) = timeline::project(input, 120.0);
            if timeline.clips.is_empty() {
                return None;
            }
            Some((
                timeline.clone(),
                regions_for(&timeline),
                sources_for(input),
            ))
        })
    }

    /// 会话用的时间线 + 素材（**不**构造渲染区域）。
    ///
    /// 【为什么与 `render_input` 分开】渲染那条路要的是 `AraPlaybackRegion`；编辑器要的
    /// 是"界面该显示什么"。还没有素材（用户还没录）时编辑器必须拿到一条**空的**
    /// 时间线，而不是 `None` —— 否则界面停在"没有时间线"，而不是"还没有内容，请录制"。
    pub(crate) fn session_timeline(&self) -> (TimelineState, HashMap<String, Arc<SourcePcm>>) {
        self.with_projection(|input| {
            let (timeline, _) = timeline::project(input, 120.0);
            (timeline, sources_for(input))
        })
    }

    /// 用**给定的**时间线构造渲染区域。
    ///
    /// 【为什么要能换时间线】编辑器会话持有的那份带着用户曲线（参数就是 Transfer 形态下
    /// "处理"的全部内容），而捕获存储那份只有几何。渲染要用前者的曲线、后者的 PCM。
    pub(crate) fn regions_for_timeline(
        &self,
        timeline: &TimelineState,
    ) -> Vec<crate::ara::AraPlaybackRegion> {
        regions_for(timeline)
    }

    /// 全部素材的源 PCM 表（按源 id 索引）。
    pub(crate) fn material_sources(&self) -> HashMap<String, Arc<SourcePcm>> {
        self.with_projection(sources_for)
    }

    /// 素材的**未处理** PCM，做成一份可直接发布的快照（"原始磁带"）。
    ///
    /// 【为什么用 `PlaybackSnapshot` 的形状】处理结果与原始磁带同形，于是音频线程只有
    /// "选哪个发布器"一个分支：不需要第二条读取通路、第二套预算、第二套回收。
    ///
    /// 【为什么铺片这件事在 `timeline` 里】它必须与投影共用同一套覆盖规则（后来的胜出、
    /// 静音片不出现），否则"听到的原始磁带"与"时间轴上看到的"会分叉。
    ///
    /// 【为什么在发布方分配而不是让音频线程给缓冲】发布只在素材变化时发生一次，
    /// 而发布方（UI 线程 / 会话 actor）本来就不是实时线程。
    pub(crate) fn raw_tape(&self) -> Option<PlaybackSnapshot> {
        self.with_projection(timeline::raw_tape)
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
        let (_, summary) = timeline::build(&all, id, 120.0);
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
            summary,
            sidecar,
        };
        let mut takes = self.takes.lock_or_recover();
        // 【为什么保留旧 take 而不是覆盖】用户要能在两次录制之间比较。内存由
        // `max_duration_sec` 与全局预算共同约束；显式清理由 `clear()` 承担。
        takes.push(take);
        drop(takes);
        // 【为什么要推进代次】新增素材 ⇒ 时间线、原始磁带与合成结果全都要重来。
        self.refresh_material_present();
        self.bump_revision();
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
fn sources_for(takes: &[ProjectionTake<'_>]) -> HashMap<String, Arc<SourcePcm>> {
    let mut sources = HashMap::new();
    for take in takes {
        for segment in take.segments {
            // 【为什么按段而不是按 take 建源】段的采样率与内容各不相同，且时间线里的
            // clip 源窗是**段内偏移** —— 一份源对应一个段，窗口语义才自洽。
            sources.insert(
                timeline::source_id(take.id, segment.id),
                Arc::new(SourcePcm {
                    sample_rate: segment.sample_rate,
                    planes: segment.planes.clone(),
                    version: segment.id,
                    _reservation: None,
                }),
            );
        }
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

    /// 观察位如实反映"宿主给过什么"，而且**不需要先武装**。
    ///
    /// 【为什么"不需要武装"是这条测试的重点】面板的录制按钮由 `canCapture`
    /// （= `saw_context`）门控，而观察此前发生在武装之后 —— 于是"必须先录一次才能录"，
    /// 按钮永远是灰的，Transfer 在**所有**宿主上都用不了。观察必须与武装无关。
    #[test]
    fn observation_bits_record_what_the_host_did_without_being_armed() {
        let store = store();
        let status = store.status();
        assert!(!status.saw_context && !status.saw_input && !status.saw_offline);
        assert!(!status.host_playing);
        store.note_offline();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 256].into_boxed_slice());
        // 注意：**没有** `store.start()` —— 这正是回归点。
        store.observe(
            &InputBlock {
                start: 0,
                frames: 256,
                sample_rate: 48_000.0,
                planes: [Some(samples), None],
            },
            true,
        );
        let status = store.status();
        assert!(
            status.saw_context,
            "见过 ProcessContext 就必须记下来，否则录制按钮永远点不动"
        );
        assert!(status.saw_input && status.saw_offline);
        assert!(status.host_playing, "播放状态是实时读数");
        assert!(!store.is_armed(), "观察不该顺手把录制武装起来");
    }

    /// 播放状态是**实时**读数：停止之后必须回落，否则面板会一直说"宿主在播放"。
    #[test]
    fn the_playing_reading_follows_the_host_instead_of_latching() {
        let store = store();
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 16].into_boxed_slice());
        let block = InputBlock {
            start: 0,
            frames: 16,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
        };
        store.observe(&block, true);
        assert!(store.status().host_playing);
        store.observe(&block, false);
        assert!(!store.status().host_playing, "停止播放必须如实回落");
        // 而"见过上下文"是**单调**的：它描述历史，不描述此刻。
        assert!(store.status().saw_context);
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
            outside_passthrough: false,
        });
        let options = store.options();
        assert!(options.max_duration_sec >= 1.0);
        assert!(options.segment_sec <= 60.0);
        assert!(!options.monitor);
        assert!(!options.outside_passthrough);
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
        assert_eq!(store.latest_take(), Some(5));
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
        assert_eq!(store.latest_take(), Some(1));
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
