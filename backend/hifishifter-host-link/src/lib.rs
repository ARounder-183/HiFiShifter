//! 宿主链接：宿主工程与独立 App 之间的**工作区交接**。
//!
//! # 为什么是文件，不是管道
//!
//! 本 crate 的前身是一条 Windows 命名管道（插件做服务端、独立 App 做客户端、JSON 帧、
//! 心跳发现目录、令牌）。它把「把工程交给另一个程序编辑」做成了一次**实时远程过程调用**，
//! 于是并发、容量、进程生命周期、平台这四件事全都被迫由**协议**回答。
//!
//! 而这四件里没有一件真的需要协议：
//!
//! * **容量**：ARA 通路只发内嵌 f32 平面，撞在 64 MiB 帧上限上 —— 长素材是**整份导入失败**。
//! * **生命周期**：服务端随插件实例生灭，于是「关掉插件窗口」就等于「链接消失」。
//! * **平台**：管道是 `#[cfg(windows)]`，而独立 App 的发布矩阵是 macOS / Linux / Windows。
//! * **并发**：真正需要回答的只有「宿主几何变了吗」，而那个问题**只有宿主侧能诚实回答**。
//!
//! 文件系统对这些问题天然免疫。所以现在的传输层就是文件系统本身：
//!
//! ```text
//! <workspace>/
//!   manifest.hfslink   宿主 → App：清单（时间轴 + 素材登记 + 版本 + 降级报告）
//!   audio/<hash>.wav   宿主 → App：素材音频，内容寻址
//!   edits.hfsedit      App → 宿主：用户编辑（只含 HiFiShifter 自有字段）
//!   status.hfslink     宿主 → App：上一次编辑是否被接受（成功与被拒都写）
//!   .open              App 侧打开标记（同一工作区不开两个编辑窗口）
//! ```
//!
//! # 为什么保留 JSON
//!
//! `manifest` / `edits` 是**对外契约**：第三方宿主扩展要实现它。
//! 人可读被第三方实现者反复验证过是有价值的（管道时代那份契约的读者就靠它）。
//! 工程文件 `.hshp` 走 msgpack 是**内部**格式，两者不必统一。
//! 音频则相反：它是纯数据、体积最大、没有任何理由被读 —— 走二进制 WAV（见 [`workspace`]）。
//!
//! # 术语
//!
//! * **生产者**：写 `manifest.hfslink` 的一侧。ARA 插件、Transfer 录制、第三方桥接都是。
//! * **消费者**：写 `edits.hfsedit` 的一侧。今天只有独立 App。
//!
//! 两侧**各写各的文件**，因此不需要锁、不需要协商、不需要令牌。

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod workspace;

pub use workspace::{AudioRef, LoadedAudio, Workspace, WorkspaceLock};

/// 清单魔数。
pub const FORMAT: &str = "hifishifter-link";
/// 编辑载荷魔数。
pub const EDIT_FORMAT: &str = "hifishifter-edit";
/// 状态载荷魔数。
pub const STATUS_FORMAT: &str = "hifishifter-status";

/// 载荷结构版本。**只有一个版本轴。**
///
/// 【为什么不是三个】管道时代有三个互不隶属的版本号（`PROTOCOL` /
/// `TIMELINE_SCHEMA_VERSION` / `EditState::CURRENT_STATE_VERSION`），其中只有
/// `TIMELINE_SCHEMA_VERSION` 真正守着载荷语义。交接是**一次性的、文件里的**，
/// 双方都能读到完整内容，一个版本号足够；而每多一个版本轴，就多一处可以忘记递增的地方。
///
/// 语义变更（新增必填字段、改变已有字段含义）必须递增本值。纯增量的可选字段不需要。
pub const FORMAT_VERSION: u32 = 1;

/// 工作区里的固定文件名。
pub const MANIFEST_FILE: &str = "manifest.hfslink";
pub const EDIT_FILE: &str = "edits.hfsedit";
pub const STATUS_FILE: &str = "status.hfslink";
pub const AUDIO_DIR: &str = "audio";
pub const TMP_DIR: &str = ".tmp";
pub const OPEN_MARK: &str = ".open";

/// VST3 **组件 state** 的预算。
///
/// 【为什么与交接无关】组件 state 是宿主存档（保存 / 撤销快照 / 模板）的一部分，
/// 必须保持小；而工作区音频是磁盘文件，没有理由服从这个数字。
/// 管道时代这个常量叫 `MAX_FRAME` 并同时约束了互不相干的三件事 ——
/// 一个协议常量在充当两件不同事情的上限。现在它只约束组件 state，名字也照实叫。
pub const MAX_COMPONENT_STATE: usize = 64 * 1024 * 1024;

/// 工作区里**素材音频**的三种状态。
///
/// 【为什么需要三态而不是「有/没有」】管道时代只有「内嵌」与「文件引用」两种形态，
/// 于是任何一处缺音频都只能把**整份导入**判失败。三态把这件事拆开：
/// 几何有效、音频暂时没有，是一个**可以显示给用户**的状态，不是一个错误。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioState {
    /// 音频已完整写入工作区，摘要可校验。自包含 —— 工作区拷到哪都能用。
    #[default]
    Complete,
    /// **几何有效、音频暂缺**（宿主尚未授权 / 仍在捕获 / 磁盘不足）。
    ///
    /// 消费者必须保留该素材的 clip，把音频槽留空，并把这件事显示出来 ——
    /// 而不是把整份导入判失败，也不是静默丢弃。
    Partial,
    /// 只引用宿主或用户自己的文件，不复制进工作区。
    ///
    /// 【为什么值得有】4 分钟立体声 96 kHz 是 184 MB。复制一份既慢又占盘，
    /// 而那个文件通常就在本机、就在用户的工程目录里。代价是工作区不再自包含，
    /// 所以消费者必须把它**标成外部引用**并提示移动后会失效。
    Extern,
}

impl AudioState {
    /// 该状态下 [`SourceEntry::file`] 是否必须存在。
    pub fn requires_file(self) -> bool {
        matches!(self, AudioState::Complete | AudioState::Extern)
    }
}

/// 这个工作区是**从哪来的**。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// ARA 插件实例。
    #[default]
    Ara,
    /// 非 ARA 的 VST3 实时录制通路。
    Transfer,
    /// 宿主侧扩展桥接（第三方）。
    Bridge,
}

impl SourceKind {
    /// 消费者该用多严的准入策略。
    ///
    /// 【为什么由 `source_kind` 决定而不是由能力位决定】ARA 的几何由宿主拥有，
    /// 任何一条读不懂的条目都说明**我们对宿主的理解错了**，静默降级会把错误藏起来；
    /// 第三方桥接的时间轴可能是扩展自己的 bug，把整份导入判失败会让用户什么都拿不到。
    /// 两种处置的正确方向相反，所以必须是两种策略。
    pub fn admission(self) -> Admission {
        match self {
            SourceKind::Ara => Admission::Strict,
            SourceKind::Transfer | SourceKind::Bridge => Admission::Degrade,
        }
    }

    /// 稳定的短名，用于 `link_id` 与 UI。
    pub fn tag(self) -> &'static str {
        match self {
            SourceKind::Ara => "ara",
            SourceKind::Transfer => "transfer",
            SourceKind::Bridge => "bridge",
        }
    }
}

/// 生产者声明**真正会做**的事。
///
/// 【为什么比管道时代少了两个】管道时代的 `Transport` / `TempoMap` 从未被 ARA 插件授予，
/// 客户端请求了也没用 —— `Response.transport` 被生产出来后在 App 侧**没有任何消费者**。
/// 一个从不说真话的能力位比没有更糟：它让「这个宿主只给了时间轴」这类事实无法被表达。
/// 实时跟随宿主播放**明确不做**（交接模型下它是纯加法，见方案 §11.1）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// 接受 `edits.hfsedit`（写回参数曲线与轨道混音）。
    Commit,
    /// 接受几何写回（今日只有第三方桥接可能声明）。
    GeometryWrite,
    /// 工作区携带素材音频（否则只有几何，用户听不到东西）。
    Audio,
}

/// 准入策略。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Admission {
    /// 任何不合格条目 ⇒ 整份拒绝。ARA 用。
    #[default]
    Strict,
    /// 逐条降级并报告。Transfer / Bridge 用。
    Degrade,
}

/// 生产这份工作区的是谁 —— 仅用于诊断。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Producer {
    pub implementation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
}

/// 两个**互不隶属**的代次计数器。
///
/// 【为什么必须是两个】`edits` 是 HiFiShifter 自己的编辑代次（参数曲线改了多少次），
/// `model` 是宿主模型的代次（宿主里动了几次工程）。宿主改工程不该影响我们的编辑代次，
/// 我们的编辑也不该让宿主的一次编译被当成「过期」。合并成一个数字就再也分不开了。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revisions {
    pub edits: u64,
    pub model: u64,
}

/// 一条**素材**的登记。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceEntry {
    /// 稳定素材键；等于时间轴里 clip 的 `source_path`。宿主提供
    /// （ARA 是 audio source 的 persistentID）。
    pub key: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
    pub state: AudioState,
    /// 相对工作区根目录的路径；`complete` / `extern` 必有。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// 音频内容摘要（见 [`Workspace::write_audio`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// `partial` 的原因；显示给用户。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// 一条**没能进来**的东西。
///
/// 【为什么要结构化，而不是一句字符串】管道时代被跳过的 clip 会**消失**
/// （`docs/host-timeline-channel.md` 原文：「跳过的 clip 会消失，不是变成『空 clip』」），
/// 而用户看到的只有一句「跳过了 N 项」。把 id 与原因留下来，面板才能说清**是什么**没进来。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossEntry {
    /// 被跳过的 clip id / 素材键。
    pub id: String,
    /// 机器可读的原因（`unknown_source` / `non_positive_duration` …）。
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// 宿主 → App 的清单。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkManifest {
    pub format: String,
    pub format_version: u32,
    /// 这份工作区的身份：同一次交接的多次导出必须一致。见 [`link_id`]。
    pub link_id: String,
    pub source_kind: SourceKind,
    /// 人类可读的宿主身份（`"REAPER"`），用于 UI。
    pub host_label: String,
    pub created_unix_ms: u64,
    pub producer: Producer,
    /// `TimelineState`。字段契约见 `docs/host-timeline-channel.md`。
    ///
    /// 【为什么仍是自由 JSON】生产者**可以是第三方**的宿主扩展，
    /// 而 `TimelineState` 的类型住在内核里。协议 crate 不依赖内核 —— 那份契约的价值
    /// 恰恰在于它不要求实现者用 Rust。结构由 [`FORMAT_VERSION`] 守着。
    pub timeline: Value,
    /// 宿主拥有的几何在**导出时**的投影（正向名单）；回写时原样送回给生产者比对。
    ///
    /// 【为什么由生产者算出、消费者原样回送】几何归属判据要回答的是
    /// 「宿主的时间轴变了吗」。消费者只有一份拷贝，`bpm`/`project_sec` 之外还有
    /// 一大堆浮点字段 —— 让它自己重算一次，任何一处归一化顺序或精度差异都会
    /// 变成一次**假漂移**（实测：同一份 `TimelineState` 经 JSON 往返后，`length_sec`
    /// 会差最后一位，`source_end_sec` 甚至能从 `0.0` 变成 `9.07e-5`）。
    /// 生产者算、消费者原样回送，比较的两边就出自同一次计算。
    ///
    /// 【为什么它是不透明的】它的语义是「生产者会拿什么来比」，与消费者无关。
    /// 消费者不得解读、不得修改、不得重算。
    pub host_geometry: Value,
    #[serde(default)]
    pub sources: Vec<SourceEntry>,
    #[serde(default)]
    pub granted: Vec<Capability>,
    pub revisions: Revisions,
    pub admission: Admission,
    /// 逐条降级报告。**空表示没有丢掉任何东西**（而不是「没检查」）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub losses: Vec<LossEntry>,
    /// 只读现场诊断（渲染器就绪度、片段覆盖与峰值）。
    ///
    /// 【为什么与管道时代的同名字段命运不同】它在管道时代被生产出来后在 App 侧
    /// 没有任何消费者 —— 是一条只写不读的线。放进清单它就有地方显示了（见方案 §10）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Value>,
}

impl LinkManifest {
    pub fn grants(&self, capability: Capability) -> bool {
        self.granted.contains(&capability)
    }

    /// 结构自检；消费者在读入后、用之前调用。
    pub fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT {
            return Err(format!("not a link manifest: {}", self.format));
        }
        if self.format_version != FORMAT_VERSION {
            return Err(format!(
                "link format {} does not match this build's {FORMAT_VERSION}",
                self.format_version
            ));
        }
        if self.link_id.trim().is_empty() {
            return Err("link manifest has no identity".into());
        }
        for source in &self.sources {
            if source.key.trim().is_empty() {
                return Err("source entry has no key".into());
            }
            if source.state.requires_file() && source.file.is_none() {
                return Err(format!(
                    "source {} carries audio but names no file",
                    source.key
                ));
            }
        }
        Ok(())
    }

    /// 素材状态计数，供 UI 直接显示。
    pub fn audio_summary(&self) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for source in &self.sources {
            match source.state {
                AudioState::Complete => counts.0 += 1,
                AudioState::Partial => counts.1 += 1,
                AudioState::Extern => counts.2 += 1,
            }
        }
        counts
    }

    /// 时间轴里的 clip 数（结构不对时为 0）。
    pub fn clip_count(&self) -> usize {
        self.timeline
            .get("clips")
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    }

    /// 时间轴里的轨道数。
    pub fn track_count(&self) -> usize {
        self.timeline
            .get("tracks")
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    }
}

/// App → 宿主的编辑载荷。
///
/// 【为什么只装这些字段】管道时代的 `Commit` 传了一整份 `TimelineState`，而接收端的
/// `EditState::merge` **只读其中两项**：`params_by_root_track` 与每条 `Track` 的
/// 5 个字段（`volume` / `muted` / `solo` / `compose_enabled` / `pitch_analysis_algo`）。
/// 其余（clip 几何、`bpm`、`project_sec`、`tempo_map`）全被丢弃。
/// **发送一个你明知会被丢弃的东西、还要在接收侧校验它，是纯粹的浪费和纯粹的误解来源。**
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditManifest {
    pub format: String,
    pub format_version: u32,
    pub link_id: String,
    /// 消费者提交时**以为**的生产者代次。与生产者当前的比对 ⇒ 乐观并发。
    pub base_revisions: Revisions,
    /// 宿主拥有的几何在消费者打开时的投影（正向名单）。
    ///
    /// 【为什么由消费者把它送回来】几何归属判据需要「宿主现在是什么」。
    /// 消费者只有一份拷贝，生产者才有权威 —— 所以判据在生产者侧算，
    /// 消费者只需交上它当时的读数。这样消费者侧的**活状态**（后台的假立体声折叠、
    /// 波形回填、音高分析都在写它）再也不可能打穿判据。
    pub host_geometry: Value,
    /// `params_by_root_track`：曲线与静态参数，按 root track id 键控。
    pub params: Value,
    /// 逐轨的 5 个可写字段。
    pub tracks: Value,
}

impl EditManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != EDIT_FORMAT {
            return Err(format!("not an edit manifest: {}", self.format));
        }
        if self.format_version != FORMAT_VERSION {
            return Err(format!(
                "edit format {} does not match this build's {FORMAT_VERSION}",
                self.format_version
            ));
        }
        if self.link_id.trim().is_empty() {
            return Err("edit manifest has no link identity".into());
        }
        Ok(())
    }
}

/// 生产者对**上一次编辑**的回答。
///
/// 【为什么成功也要写】管道时代「提交成功」只存在于那个连接的回包里：用户关掉 App
/// 就再也查不到自己到底提交成功了没有。落盘之后，成功与被拒**同一份可见**，
/// 而且可以复查。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkStatus {
    pub format: String,
    pub format_version: u32,
    pub link_id: String,
    pub ok: bool,
    /// 协议级错误码（[`error_code`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<u32>,
    /// 人类可读原因。失败时必须**指名字段**（见 [`error_code::GEOMETRY_CHANGED`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// 回答之后的代次；消费者据此决定是否要重新导出。
    pub revisions: Revisions,
    pub updated_unix_ms: u64,
}

impl LinkStatus {
    /// 一份「还没发生过任何编辑」的初值；生产者写出清单时同时写下它。
    pub fn initial(link_id: &str, revisions: Revisions) -> Self {
        Self {
            format: STATUS_FORMAT.into(),
            format_version: FORMAT_VERSION,
            link_id: link_id.to_string(),
            ok: true,
            code: None,
            error: None,
            revisions,
            updated_unix_ms: crate::now_ms(),
        }
    }

    pub fn is_current(&self, link_id: &str) -> bool {
        self.format == STATUS_FORMAT
            && self.format_version == FORMAT_VERSION
            && self.link_id == link_id
    }
}

/// `LinkStatus.code` 的稳定取值。
///
/// 【为什么在协议 crate 里】码是**协议承诺**（只增不改）：生产者的产生点与消费者的
/// 分类点都从这里取，任何一侧改措辞都不会漂移。文本只做展示。
pub mod error_code {
    /// 编辑文档已关闭（终态：重试不会成功）。
    pub const DOCUMENT_CLOSED: u32 = 1;
    /// FX 处理器（组件）已销毁。
    pub const FX_PROCESSOR_CLOSED: u32 = 2;
    /// 编辑器视图不存在或身份未知。
    pub const VIEW_CLOSED_OR_UNKNOWN: u32 = 3;
    /// 视图身份与会话不匹配。
    pub const VIEW_IDENTITY_MISMATCH: u32 = 4;
    /// 请求排队期间编辑路由已改变。
    pub const ROUTE_CHANGED_WHILE_QUEUED: u32 = 5;
    /// 编辑路由属于另一份文档。
    pub const ROUTE_OTHER_DOCUMENT: u32 = 6;
    /// 编辑租约已被撤销。
    pub const LEASE_REVOKED: u32 = 7;
    /// 插件实例已关闭。
    pub const INSTANCE_CLOSED: u32 = 8;
    /// 宿主从未把本实例绑定到 ARA 文档（终态）。
    ///
    /// 【为什么必须与 `DOCUMENT_CLOSED` 分开】两者此前共用一句 "document closed"，
    /// 于是「宿主根本没走 ARA 路径」被表述成「会话已结束」。前端据此提示
    /// 「关闭并重新打开插件窗口」—— 而重开会走完全相同的路径、得到完全相同的错误。
    /// 那条提示把用户引向一个**必然失败**的补救动作。
    pub const ARA_NOT_BOUND: u32 = 9;

    // ── 回写（edits.hfsedit）的回答 ──────────────────────────────────────────

    /// 编辑载荷结构版本不认识。消费者该升级。
    pub const UNSUPPORTED_FORMAT: u32 = 10;
    /// 宿主模型尚未就绪（可重试）。
    pub const MODEL_NOT_READY: u32 = 11;
    /// 代次冲突（宿主或编辑在合并期间变了）。
    pub const CONFLICT: u32 = 12;
    /// 消费者动过宿主拥有的几何。`error` 里带**指名字段**的差异列表。
    pub const GEOMETRY_CHANGED: u32 = 13;
    /// 编辑载荷本身不合格（曲线非有限、轨道未知、音量越界……）。
    pub const INVALID_EDIT: u32 = 14;
    /// 链接身份对不上：这份编辑不属于这份清单。
    pub const LINK_MISMATCH: u32 = 15;
}

/// 交接身份：`<source_kind>:<host_label>:<doc_key>`。
///
/// 【为什么需要它】消费者要把编辑挂到「某一次交接」上。宿主侧的 clip id / track id
/// 在一个会话内稳定，但**跨会话不一定**（`ara/mapping.rs` 记录过位置派生的身份
/// 在宿主删除后会整体前移）。用 `link_id` 把「这是哪一次交接」记下来，
/// 消费者重开工程时才能判断「我这份编辑还是不是对着同一个宿主工程」。
///
/// `doc_key` 由生产者给：ARA 侧取 `TrackBindings` 的签名，Transfer 侧取首个 take，
/// 桥接侧自报。它只回答「是不是同一个」，不承担 clip 身份的职责。
pub fn link_id(source_kind: SourceKind, host_label: &str, doc_key: &str) -> String {
    format!(
        "{}:{}:{}",
        source_kind.tag(),
        sanitize(host_label),
        sanitize(doc_key)
    )
}

/// 把任意文本塞进 `link_id` 的合法片段（可打印、无分隔符、有长度上限）。
fn sanitize(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed.chars().take(96).collect()
    }
}

/// 当前 unix 毫秒。
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> LinkManifest {
        LinkManifest {
            format: FORMAT.into(),
            format_version: FORMAT_VERSION,
            link_id: link_id(SourceKind::Ara, "REAPER", "a3f1"),
            source_kind: SourceKind::Ara,
            host_label: "REAPER".into(),
            created_unix_ms: 1_770_000_000_000,
            producer: Producer {
                implementation: "hifishifter-ara 0.9.0".into(),
                pid: Some(4212),
            },
            timeline: serde_json::json!({
                "clips": [{"id": "ara-region-1"}],
                "tracks": [{"id": "t"}]
            }),
            host_geometry: serde_json::json!({"clips": [], "tracks": []}),
            sources: vec![SourceEntry {
                key: "ara://vox".into(),
                sample_rate: 48_000,
                channels: 2,
                frames: 1024,
                state: AudioState::Complete,
                file: Some("audio/ab.wav".into()),
                blake3: Some("ab".into()),
                bytes: Some(8192),
                note: None,
            }],
            granted: vec![Capability::Commit, Capability::Audio],
            revisions: Revisions { edits: 7, model: 42 },
            admission: Admission::Strict,
            losses: Vec::new(),
            diagnostics: None,
        }
    }

    #[test]
    fn a_well_formed_manifest_validates() {
        manifest().validate().unwrap();
        let m = manifest();
        assert_eq!(m.clip_count(), 1);
        assert_eq!(m.track_count(), 1);
        assert_eq!(m.audio_summary(), (1, 0, 0));
        assert!(m.grants(Capability::Commit));
        assert!(!m.grants(Capability::GeometryWrite));
    }

    /// 结构版本不一致必须被拒 —— 静默接受一份读不懂的清单比拒绝更糟。
    #[test]
    fn a_future_format_version_is_rejected() {
        let mut m = manifest();
        m.format_version = FORMAT_VERSION + 1;
        assert!(m.validate().unwrap_err().contains("does not match"));
    }

    #[test]
    fn a_foreign_document_is_not_a_link_manifest() {
        let mut m = manifest();
        m.format = "something-else".into();
        assert!(m.validate().unwrap_err().contains("not a link manifest"));
    }

    /// `partial` 没有音频文件是**合法**的；`complete` 没有则是清单自相矛盾。
    #[test]
    fn only_states_that_carry_audio_require_a_file() {
        let mut m = manifest();
        m.sources[0].state = AudioState::Partial;
        m.sources[0].file = None;
        m.sources[0].blake3 = None;
        m.sources[0].bytes = None;
        m.sources[0].note = Some("host has not authorised this source yet".into());
        m.validate().unwrap();
        assert_eq!(m.audio_summary(), (0, 1, 0));

        m.sources[0].state = AudioState::Complete;
        assert!(m.validate().unwrap_err().contains("carries audio"));
    }

    /// 三个状态必须能各自往返，且 `extern` 与 `complete` 都带文件。
    #[test]
    fn audio_states_round_trip_and_drive_the_summary() {
        let mut m = manifest();
        let base = m.sources[0].clone();
        m.sources = vec![
            SourceEntry {
                key: "a".into(),
                state: AudioState::Complete,
                file: Some("audio/a.wav".into()),
                ..base.clone()
            },
            SourceEntry {
                key: "b".into(),
                state: AudioState::Extern,
                file: Some("D:/Media/b.wav".into()),
                ..base.clone()
            },
            SourceEntry {
                key: "c".into(),
                state: AudioState::Partial,
                file: None,
                ..base
            },
        ];
        m.validate().unwrap();
        assert_eq!(m.audio_summary(), (1, 1, 1));
        let text = serde_json::to_string(&m).unwrap();
        assert!(text.contains(r#""state":"extern""#), "{text}");
        let back: LinkManifest = serde_json::from_str(&text).unwrap();
        assert_eq!(back.sources[1].state, AudioState::Extern);
        assert_eq!(back.sources[2].state, AudioState::Partial);
    }

    /// 缺省字段必须能被旧读者容忍：`sources` / `granted` / `losses` / `diagnostics`
    /// 都是可选的，否则一份只有几何的清单读不出来。
    #[test]
    fn optional_sections_may_be_absent() {
        let minimal = serde_json::json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "link_id": "ara:REAPER:a3f1",
            "source_kind": "ara",
            "host_label": "REAPER",
            "created_unix_ms": 1,
            "producer": {"implementation": "test"},
            "timeline": {"clips": []},
            "host_geometry": {"clips": [], "tracks": []},
            "revisions": {"edits": 0, "model": 0},
            "admission": "strict",
        });
        let m: LinkManifest = serde_json::from_value(minimal).unwrap();
        m.validate().unwrap();
        assert!(m.sources.is_empty());
        assert_eq!(m.clip_count(), 0);
        assert!(!m.grants(Capability::Audio));
    }

    /// ARA 是严格准入，其余逐条降级 —— 处置方向相反，所以由来源决定。
    #[test]
    fn admission_follows_the_source_kind() {
        assert_eq!(SourceKind::Ara.admission(), Admission::Strict);
        assert_eq!(SourceKind::Transfer.admission(), Admission::Degrade);
        assert_eq!(SourceKind::Bridge.admission(), Admission::Degrade);
    }

    /// `link_id` 必须稳定、可读，且不能因为宿主名里有路径分隔符就长得不像一个身份。
    #[test]
    fn link_ids_are_stable_and_sanitised() {
        let a = link_id(SourceKind::Ara, "REAPER", "C:/Projects/My Song");
        let b = link_id(SourceKind::Ara, "REAPER", "C:/Projects/My Song");
        assert_eq!(a, b);
        assert!(a.starts_with("ara:REAPER:"), "{a}");
        assert!(!a.contains('/'), "{a}");
        assert!(!a.contains('\\'), "{a}");
        assert_eq!(link_id(SourceKind::Bridge, "  ", ""), "bridge:unnamed:unnamed");
        assert_ne!(
            link_id(SourceKind::Ara, "REAPER", "x"),
            link_id(SourceKind::Transfer, "REAPER", "x")
        );
    }

    /// 编辑载荷只校验自己的结构；`params` / `tracks` 的语义归内核。
    #[test]
    fn an_edit_manifest_validates_its_own_envelope() {
        let edit = EditManifest {
            format: EDIT_FORMAT.into(),
            format_version: FORMAT_VERSION,
            link_id: "ara:REAPER:a3f1".into(),
            base_revisions: Revisions { edits: 7, model: 42 },
            host_geometry: serde_json::json!({"clips": []}),
            params: serde_json::json!({}),
            tracks: serde_json::json!([]),
        };
        edit.validate().unwrap();
        let mut wrong = edit.clone();
        wrong.format_version = 99;
        assert!(wrong.validate().is_err());
        let mut unowned = edit;
        unowned.link_id = "  ".into();
        assert!(unowned.validate().is_err());
    }

    /// 状态文件必须认得出「这不是给这份清单的」。
    #[test]
    fn a_status_only_belongs_to_its_own_link() {
        let status = LinkStatus::initial("ara:REAPER:a3f1", Revisions { edits: 0, model: 3 });
        assert!(status.is_current("ara:REAPER:a3f1"));
        assert!(!status.is_current("ara:REAPER:other"));
    }

    /// 两个计数器互不隶属：宿主模型动了不该影响编辑代次。
    #[test]
    fn revisions_are_independent() {
        let a = Revisions { edits: 3, model: 1 };
        let b = Revisions { edits: 3, model: 9 };
        assert_ne!(a, b);
        assert_eq!(a.edits, b.edits);
    }
}
