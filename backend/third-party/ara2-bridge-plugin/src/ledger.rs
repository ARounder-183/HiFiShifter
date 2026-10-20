//! 宿主的 ARA 图走到哪一步的账本（T8）。
//!
//! 【为什么需要它】"重开工程后插件一片等待宿主音频"在日志里与"宿主根本没发图回调"
//! 完全同形：前者是拒绝（我们收到了但拒了），后者是缺席（我们什么都没收到）。
//! 前八轮每一轮都在猜这两者中的哪一个，因为**没有任何计数**区分它们。
//!
//! 纪律：账本只记"到达 / 成功 / 拒绝"三个数与最后一条拒绝原因，**不**参与任何控制流 ——
//! 它存在的唯一目的是让下一次复现能自述。

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// 图上的一步。下标即 `GraphLedger` 各数组的索引。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphStage {
    /// `createMusicalContext`
    MusicalContext,
    /// `createRegionSequence`
    RegionSequence,
    /// `createAudioSource`
    AudioSource,
    /// `createAudioModification`
    AudioModification,
    /// `createPlaybackRegion`
    PlaybackRegion,
}

impl GraphStage {
    /// 固定数量的步（数组长度）。
    pub const COUNT: usize = 5;

    /// 该步在账本数组里的下标。
    pub const fn index(self) -> usize {
        match self {
            Self::MusicalContext => 0,
            Self::RegionSequence => 1,
            Self::AudioSource => 2,
            Self::AudioModification => 3,
            Self::PlaybackRegion => 4,
        }
    }

    /// 读数里用的短名（与 `[ara] graph:` 一行的字段一致）。
    pub const fn key(self) -> &'static str {
        match self {
            Self::MusicalContext => "ctx",
            Self::RegionSequence => "seq",
            Self::AudioSource => "src",
            Self::AudioModification => "mod",
            Self::PlaybackRegion => "region",
        }
    }
}

/// 一次被闸门绕过的类别（不再视为硬门，但要留痕）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    /// content reader 仍存活（不再阻断模型操作）。
    ContentReader,
    /// 角色未分配给本实例（宿主声明但未分配）。
    Role,
    /// 调用线程与绑定线程不同（不再阻断记录，只阻断生命周期操作）。
    Thread,
    /// 同一角色重复分配（幂等接受）。
    Duplicate,
}

impl GateKind {
    /// 固定数量的类别。
    pub const COUNT: usize = 4;

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::ContentReader => 0,
            Self::Role => 1,
            Self::Thread => 2,
            Self::Duplicate => 3,
        }
    }

    const fn key(self) -> &'static str {
        match self {
            Self::ContentReader => "reader",
            Self::Role => "role",
            Self::Thread => "thread",
            Self::Duplicate => "duplicate",
        }
    }
}

/// 一张图账本：只增计数，永远不参与控制流。
///
/// 【线程模型】计数全部是原子量：ARA 模型线程写，插件自己的非实时线程与诊断线程读。
/// 不用 `Mutex` —— 诊断读数不应为了"看一眼"而阻塞模型的回调路径；只有"最后一条拒绝"
/// 需要写字符串，那用一把小锁，且只在真拒绝时取。
#[derive(Debug, Default)]
pub struct GraphLedger {
    creates: [AtomicU32; GraphStage::COUNT],
    rejects: [AtomicU32; GraphStage::COUNT],
    last_reject_reason: std::sync::Mutex<Option<(GraphStage, String)>>,
    enable_access_true: AtomicU32,
    enable_access_false: AtomicU32,
    begin_editing: AtomicU32,
    end_editing: AtomicU32,
    restore_archive: AtomicU32,
    store_archive: AtomicU32,
    notify_model_updates: AtomicU32,
    gates_bypassed: [AtomicU32; GateKind::COUNT],
    bound: AtomicBool,
    assignment_rejected: AtomicU32,
}

impl GraphLedger {
    /// 记一次"宿主要创建这个对象，我们收到了"。
    pub fn note_create(&self, stage: GraphStage) {
        self.creates[stage.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次"我们拒了这个对象"（连同原因）。
    pub fn note_reject(&self, stage: GraphStage, reason: impl Into<String>) {
        self.rejects[stage.index()].fetch_add(1, Ordering::Relaxed);
        let reason = reason.into();
        *self.last_reject_reason.lock().unwrap_or_else(|p| p.into_inner()) = Some((stage, reason));
    }

    /// 记一次闸门被绕过（不再拒绝，但要留痕）。
    pub fn note_gate_bypassed(&self, gate: GateKind) {
        self.gates_bypassed[gate.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次样本访问开关。
    pub fn note_enable_access(&self, enable: bool) {
        let counter = if enable {
            &self.enable_access_true
        } else {
            &self.enable_access_false
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次编辑会话开始。
    pub fn note_begin_editing(&self) {
        self.begin_editing.fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次编辑会话结束。
    pub fn note_end_editing(&self) {
        self.end_editing.fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次存档恢复到达。
    pub fn note_restore_archive(&self) {
        self.restore_archive.fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次宿主请求存档。
    pub fn note_store_archive(&self) {
        self.store_archive.fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次 `notifyModelUpdates` 到达。
    pub fn note_notify_model_updates(&self) {
        self.notify_model_updates.fetch_add(1, Ordering::Relaxed);
    }

    /// 记一次扩展绑定成功。
    pub fn note_bound(&self) {
        self.bound.store(true, Ordering::Release);
    }

    /// 记一次 renderer 分配被拒绝（含线程/角色/重复三类）。
    pub fn note_assignment_rejected(&self) {
        self.assignment_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// 各步的到达次数。
    pub fn creates(&self, stage: GraphStage) -> u32 {
        self.creates[stage.index()].load(Ordering::Relaxed)
    }

    /// 各步的拒绝次数。
    pub fn rejects(&self, stage: GraphStage) -> u32 {
        self.rejects[stage.index()].load(Ordering::Relaxed)
    }

    /// 最后一条拒绝原因（`stage` + 原文）。
    pub fn last_reject(&self) -> Option<(GraphStage, String)> {
        self.last_reject_reason
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// 各闸门被绕过的次数。
    pub fn gates_bypassed(&self, gate: GateKind) -> u32 {
        self.gates_bypassed[gate_index(gate)].load(Ordering::Relaxed)
    }

    /// 样本访问的开关次数。
    pub fn enable_access(&self) -> (u32, u32) {
        (
            self.enable_access_true.load(Ordering::Relaxed),
            self.enable_access_false.load(Ordering::Relaxed),
        )
    }

    /// 编辑会话起止、存档、空闲入口的计数（顺序固定，供序列化）。
    pub fn sessions(&self) -> (u32, u32, u32, u32, u32) {
        (
            self.begin_editing.load(Ordering::Relaxed),
            self.end_editing.load(Ordering::Relaxed),
            self.restore_archive.load(Ordering::Relaxed),
            self.store_archive.load(Ordering::Relaxed),
            self.notify_model_updates.load(Ordering::Relaxed),
        )
    }

    /// 是否绑定过扩展。
    pub fn bound(&self) -> bool {
        self.bound.load(Ordering::Acquire)
    }

    /// 分配被拒绝的总次数。
    pub fn assignment_rejected(&self) -> u32 {
        self.assignment_rejected.load(Ordering::Relaxed)
    }

    /// 一行读数：`ctx=1/0 seq=1/0 src=2/0 mod=2/0 region=2/1 ...`。
    ///
    /// 【为什么是"到达/拒绝"而不是只有到达】两者的区别正是每次复现要回答的问题：
    /// `region=0/0` 是宿主没发，`region=2/1` 是我们拒了一条。
    pub fn summary(&self) -> String {
        const STAGES: [GraphStage; GraphStage::COUNT] = [
            GraphStage::MusicalContext,
            GraphStage::RegionSequence,
            GraphStage::AudioSource,
            GraphStage::AudioModification,
            GraphStage::PlaybackRegion,
        ];
        const GATES: [GateKind; GateKind::COUNT] = [
            GateKind::ContentReader,
            GateKind::Role,
            GateKind::Thread,
            GateKind::Duplicate,
        ];
        let mut stages = String::new();
        for (index, stage) in STAGES.into_iter().enumerate() {
            if index > 0 {
                stages.push(' ');
            }
            stages.push_str(&format!("{}={}/{}", stage.key(), self.creates(stage), self.rejects(stage)));
        }
        let mut gates = String::new();
        for (index, gate) in GATES.into_iter().enumerate() {
            if index > 0 {
                gates.push(',');
            }
            gates.push_str(&format!("{}={}", gate.key(), self.gates_bypassed(gate)));
        }
        let (access_on, access_off) = self.enable_access();
        let (begin, end, restore, store, notify) = self.sessions();
        let mut line = format!(
            "[ara] graph: {stages} gates({gates}) access={access_on}/{access_off} \
             edit={begin}/{end} archive={restore}/{store} notify={notify} bound={}",
            self.bound()
        );
        if let Some((stage, reason)) = self.last_reject() {
            line.push_str(&format!(" last={}:{reason}", stage.key()));
        }
        line
    }
}

/// `GateKind::index` 的对外可调用形态（保持私有实现与公共读法之间没有两份逻辑）。
fn gate_index(gate: GateKind) -> usize {
    match gate {
        GateKind::ContentReader => 0,
        GateKind::Role => 1,
        GateKind::Thread => 2,
        GateKind::Duplicate => 3,
    }
}