//! 宿主时间轴通道：ARA 插件实例与宿主侧扩展（桥接）共用的有界协议。
//!
//! 只传宿主 PCM、时间轴 JSON 与用户参数，不信任客户端几何。协议**字节**对 ARA
//! 通路保持不变（`PROTOCOL` / 帧格式 / token / 心跳 / `Snapshot` / `Commit`）；
//! 本版新增四处**纯增量**扩展：能力协商（`Hello`）、实例来源标识
//! （`source_kind`）、PCM 文件引用（`HostPcm::file`）与传输状态
//! （`Response::transport`）。对外契约见 `docs/host-timeline-channel.md`。

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{Read, Write};
use std::path::PathBuf;

#[cfg(windows)]
mod transport;
#[cfg(windows)]
pub use transport::Server;

pub const PROTOCOL: u32 = 1;
pub const MAX_FRAME: usize = 64 * 1024 * 1024;

/// 这个实例是**哪种**时间轴源。
///
/// 【为什么需要它】发现列表要能区分"REAPER 的 ARA 实例"与"Vegas 的桥接"。
/// `name` 是自由字符串，前端无法据它决定 UI 措辞与可用操作。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// ARA 插件实例（既有行为）。
    #[default]
    Ara,
    /// 宿主侧扩展桥接（Vegas 扩展等）。
    Bridge,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceRecord {
    pub instance_id: String,
    pub pid: u32,
    pub pipe_name: String,
    pub token: String,
    pub name: String,
    pub heartbeat_ms: u64,
    pub protocol: u32,
    /// 实例来源。旧记录没有该字段 → `Ara`（历史行为），因此缺省值必须与
    /// 既有 ARA 实现一致。
    #[serde(default)]
    pub source_kind: SourceKind,
}

/// 客户端想要、服务端声明真正会做的可选能力。
///
/// 【为什么要有协商】桥接服务端是**第三方**代码。没有协商，客户端只能猜对方支持
/// 什么，猜错了就是静默降级。协商让"这个宿主只给了时间轴、没给 PCM"变成一个
/// **明示的事实**。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// 接受 `Commit`（写回时间轴参数）。
    Commit,
    /// 在 `Snapshot` 里提供宿主 PCM 平面。
    HostPcm,
    /// 提供宿主播放位置 / 播放状态。
    Transport,
    /// 提供宿主 tempo map。
    TempoMap,
    /// 宿主几何可写（能接受写回的 clip 位置/长度）。
    GeometryWrite,
}

/// `Hello` 的响应。
///
/// 【为什么响应帧直接是它，而不是 `Response`】`Hello` 不是时间轴操作：它不该
/// 借用 `Snapshot` 的 `revision` / `timeline` 字段。混在一起会让"能力变了"和
/// "时间轴变了"共用一条路径。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HelloResponse {
    pub source_kind: SourceKind,
    /// 服务端支持的载荷结构版本。与客户端不一致 → 客户端拒连并说明要升级谁。
    pub timeline_schema: u32,
    /// 服务端**真正会做**的能力子集（必须是请求 `wants` 的子集）。
    pub granted: Vec<Capability>,
    /// 人类可读的宿主身份（"Vegas Pro 22"），用于 UI。
    pub host_label: String,
    /// 本实现的协议实现版本，用于诊断。
    pub implementation: String,
}

impl HelloResponse {
    /// 服务端是否真的会做某项能力。**声明了就必须做到**，所以这里是唯一判据。
    pub fn grants(&self, capability: Capability) -> bool {
        self.granted.contains(&capability)
    }
}

/// 宿主播放位置 / 状态；仅当 `Capability::Transport` 被授予时填充。
///
/// 【为什么值得要】桥接是**拉取式**的，天生没有播放位置。加上它，独立 App 就能
/// 做"跟随宿主播放"，把桥接从"导入/导出"升级为"半联动"。
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TransportState {
    pub position_sec: f64,
    pub playing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_signature: Option<(i32, i32)>,
}

/// 宿主已把 PCM 落成文件时的引用形态。
///
/// 【为什么必须带 `blake3`】路径是**可变的**：宿主写完、接收端读之前，那个文件
/// 可能被别的东西覆盖（临时目录清理、用户操作）。没有摘要校验，"读到别的音频"
/// 是静默的，而且会被当成用户的编辑存进工程 —— 那是最难排查的一类故障。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcmFile {
    pub path: PathBuf,
    /// 文件内容的 blake3 摘要（hex）；接收端校验。
    pub blake3: String,
    /// 该文件是否是宿主私有临时文件（接收端可在使用后删除）。
    #[serde(default)]
    pub ephemeral: bool,
}

/// 一份宿主 PCM。
///
/// 【为什么保留 `planes` 而不改成 `kind` 标签枚举】`planes` 是 ARA 通路**现有**的
/// 线上形态。把它换成标签枚举会让 ARA 的字节也变（并让 serde 在反序列化时多一次
/// 全量缓冲），违反"ARA 通路逐位不变"。两种形态互斥：`file` 是引用形态，
/// `planes` 是内嵌形态；同时给出或都不给出都是歧义的，由准入校验拒绝。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostPcm {
    pub persistent_id: String,
    pub sample_rate: u32,
    pub fingerprint: String,
    /// 内嵌平面（小素材；ARA 通路的既有形态）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub planes: Vec<Vec<f32>>,
    /// 文件引用（大素材；让传输量从 O(音频) 降到 O(1)，绕开 64 MiB 帧上限）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<PcmFile>,
}

/// PCM 载荷的穷尽视图：使用点匹配它，线格式保持增量兼容。
pub enum PcmPayload<'a> {
    Inline(&'a [Vec<f32>]),
    File(&'a PcmFile),
}

impl HostPcm {
    /// 解析出唯一形态；两种都给或都不给返回 `None`（歧义，应由准入校验拒绝）。
    pub fn payload(&self) -> Option<PcmPayload<'_>> {
        match (&self.file, self.planes.is_empty()) {
            (Some(file), true) => Some(PcmPayload::File(file)),
            (None, false) => Some(PcmPayload::Inline(&self.planes)),
            _ => None,
        }
    }
}

/// `Commit.timeline` 载荷的结构版本。
///
/// 【为什么在载荷里而不是信封】`PROTOCOL` 守卫的是帧格式；`Commit` 里的
/// `timeline` 是一个**自由 JSON**，它自己的字段结构变更（新增/改义）不会动到
/// 信封。接收端按它拒绝"结构对不上"的提交，而不是静默把看不懂的 JSON 合并进
/// 编辑会话。历史客户端不带该字段 → 反序列化为 0 → 接收端按 pre-versioning
/// 接受；从本版起，timeline 语义变更必须递增本值。
pub const TIMELINE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// 能力协商。**必须是任何新客户端的第一个请求。**
    Hello {
        /// 客户端能理解的载荷结构版本。
        timeline_schema: u32,
        /// 客户端想要的可选能力。
        #[serde(default)]
        wants: Vec<Capability>,
    },
    Snapshot,
    Commit {
        base_revision: u64,
        model_revision: u64,
        /// timeline JSON 的结构版本；0 = 版本化之前的历史客户端。
        #[serde(default)]
        schema_version: u32,
        timeline: Value,
    },
}

/// `Response.code` 的稳定取值。
///
/// 【为什么在协议 crate 里】码是**协议承诺**（只增不改）：插件侧的产生点与
/// GUI 侧的消费点都从这里取，任何一侧改措辞都不会漂移。客户端分类
/// （如"终态 => 永久停止轮询"）先看 `code`；`code` 缺失（旧服务端）才退回
/// 对 `error` 文本的子串匹配 —— 文本只做展示。
pub mod error_code {
    /// 编辑文档已关闭（终态：再重试也不会成功）。
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
    /// 于是"宿主根本没走 ARA 路径"（Cubase 实测：从不查询 `IPlugInEntryPoint`、
    /// 从不创建文档控制器）被表述成"会话已结束"。前端据此提示"关闭并重新打开插件
    /// 窗口"—— 而重开会走完全相同的路径、得到完全相同的错误。那条提示把用户引向
    /// 一个**必然失败**的补救动作。
    pub const ARA_NOT_BOUND: u32 = 9;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    pub error: Option<String>,
    /// 协议级错误码（[`error_code`]）。可选字段 —— 旧客户端忽略，
    /// 不改变 Snapshot/Commit 协议语义。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<u32>,
    pub revision: u64,
    pub model_revision: u64,
    pub timeline: Option<Value>,
    #[serde(default)]
    pub sources: Vec<HostPcm>,
    /// 只读现场诊断；旧客户端忽略该可选字段，不改变Snapshot/Commit协议语义。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Value>,
    /// 宿主播放位置/状态；仅当授予 `Capability::Transport` 时填充。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<TransportState>,
}

/// 服务端响应帧：`Hello` 的响应是**裸** `HelloResponse`，其余是 `Response`。
///
/// 【为什么不是 `Response` 加一个 `hello` 字段】那会让 `Hello` 的回包也带上
/// `revision` / `timeline` 这些与它无关的字段，客户端无法从形状上区分两种回包。
/// `untagged` 只作用于序列化（服务端产生），客户端按自己发出的请求类型解码。
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Reply {
    Hello(HelloResponse),
    Response(Response),
}

impl Reply {
    /// 供日志使用的成功判据；`Hello` 永远算成功（能回就说明实例活着）。
    pub fn is_ok(&self) -> bool {
        match self {
            Reply::Hello(_) => true,
            Reply::Response(response) => response.ok,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    protocol: u32,
    token: String,
    request: Request,
}

/// 编码有界小端长度帧，拒绝无法序列化的内容。
pub fn write_frame<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if bytes.is_empty() || bytes.len() > MAX_FRAME {
        return Err("frame too large".into());
    }
    writer
        .write_all(&(bytes.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer.write_all(&bytes).map_err(|e| e.to_string())
}

/// 只在检查长度后分配；短帧不能成为合法请求。
pub fn read_frame<T: for<'a> Deserialize<'a>>(reader: &mut impl Read) -> Result<T, String> {
    let mut prefix = [0; 4];
    reader.read_exact(&mut prefix).map_err(|e| e.to_string())?;
    let size = u32::from_le_bytes(prefix) as usize;
    if size == 0 || size > MAX_FRAME {
        return Err("invalid frame length".into());
    }
    let mut bytes = vec![0; size];
    reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

/// GUI只发现仍有心跳的本机实例，不删除其他会话记录。
pub fn discover() -> Result<Vec<InstanceRecord>, String> {
    let root = instance_dir()?;
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut instances = Vec::new();
    for entry in std::fs::read_dir(root)
        .map_err(|e| e.to_string())?
        .flatten()
    {
        if entry.path().extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        if entry.metadata().is_ok_and(|m| m.len() > 8192) {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        let Ok(instance) = serde_json::from_slice::<InstanceRecord>(&bytes) else {
            continue;
        };
        if validate_record(&instance).is_ok()
            && now_ms().saturating_sub(instance.heartbeat_ms) < 15000
        {
            instances.push(instance);
        }
    }
    instances.sort_by(|a, b| a.instance_id.cmp(&b.instance_id));
    Ok(instances)
}

/// 有界同步请求；GUI必须在阻塞工作线程调用。
pub fn exchange(instance: &InstanceRecord, request: &Request) -> Result<Response, String> {
    validate_record(instance)?;
    #[cfg(windows)]
    {
        transport::exchange(instance, request)
    }
    #[cfg(not(windows))]
    {
        let _ = request;
        Err("host timeline channel requires Windows".into())
    }
}

/// 能力协商；**任何新客户端的第一个请求**。
///
/// 【为什么把 schema 守卫放在这里而不是让调用方自己比】"版本对不上"的处置是
/// 协议级承诺：必须**拒连**并说明要升级哪一侧，而不是让每个调用点各自判断。
pub fn hello(instance: &InstanceRecord, wants: &[Capability]) -> Result<HelloResponse, String> {
    validate_record(instance)?;
    let request = Request::Hello {
        timeline_schema: TIMELINE_SCHEMA_VERSION,
        wants: wants.to_vec(),
    };
    #[cfg(windows)]
    let response: HelloResponse = transport::exchange(instance, &request)?;
    #[cfg(not(windows))]
    let response: HelloResponse = {
        let _ = &request;
        return Err("host timeline channel requires Windows".into());
    };
    if response.timeline_schema != TIMELINE_SCHEMA_VERSION {
        return Err(format!(
            "host timeline schema {} does not match this build's {}; upgrade {}",
            response.timeline_schema,
            TIMELINE_SCHEMA_VERSION,
            if response.timeline_schema > TIMELINE_SCHEMA_VERSION {
                "HiFiShifter"
            } else {
                "the host extension"
            }
        ));
    }
    Ok(response)
}

/// 发现目录可在验收时定向到worktree，独立app与插件须使用同一个目录。
pub fn instance_dir() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("HIFISHIFTER_ARA_INSTANCE_DIR") {
        return Ok(root.into());
    }
    std::env::var_os("LOCALAPPDATA")
        .map(|root| PathBuf::from(root).join("HiFiShifter/ara-instances"))
        .ok_or_else(|| "LOCALAPPDATA unavailable".into())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn validate_record(record: &InstanceRecord) -> Result<(), String> {
    let expected = format!(
        r"\\.\pipe\HiFiShifter-ARA-{}-{}",
        record.pid, record.instance_id
    );
    if record.protocol != PROTOCOL
        || uuid::Uuid::parse_str(&record.instance_id).is_err()
        || uuid::Uuid::parse_str(&record.token).is_err()
        || record.pipe_name != expected
    {
        return Err("invalid host timeline instance record".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_frame_round_trip_preserves_edit_revision() {
        let request = Request::Commit {
            base_revision: 7,
            model_revision: 9,
            schema_version: TIMELINE_SCHEMA_VERSION,
            timeline: serde_json::json!({"pitch": [60, 62]}),
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &request).unwrap();
        let actual: Request = read_frame(&mut bytes.as_slice()).unwrap();
        match actual {
            Request::Commit {
                base_revision,
                model_revision,
                schema_version,
                timeline,
            } => {
                assert_eq!(schema_version, TIMELINE_SCHEMA_VERSION);
                assert_eq!((base_revision, model_revision), (7, 9));
                assert_eq!(timeline["pitch"], serde_json::json!([60, 62]));
            }
            _ => panic!("lost edit request"),
        }
    }

    /// `Hello` 必须能在信封里往返，且 `wants` 缺省为空。
    #[test]
    fn hello_request_round_trips_with_and_without_wants() {
        let request = Request::Hello {
            timeline_schema: TIMELINE_SCHEMA_VERSION,
            wants: vec![Capability::Commit, Capability::GeometryWrite],
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &request).unwrap();
        let text = String::from_utf8(bytes[4..].to_vec()).unwrap();
        assert!(text.contains(r#""op":"hello""#), "{text}");
        assert!(text.contains(r#""geometry_write""#), "{text}");
        match read_frame::<Request>(&mut bytes.as_slice()).unwrap() {
            Request::Hello {
                timeline_schema,
                wants,
            } => {
                assert_eq!(timeline_schema, TIMELINE_SCHEMA_VERSION);
                assert_eq!(wants, vec![Capability::Commit, Capability::GeometryWrite]);
            }
            _ => panic!("lost hello request"),
        }
        let bare = Request::Hello {
            timeline_schema: 1,
            wants: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &bare).unwrap();
        assert!(matches!(
            read_frame::<Request>(&mut bytes.as_slice()).unwrap(),
            Request::Hello { wants, .. } if wants.is_empty()
        ));
    }

    /// 响应帧的形状：`Hello` 是裸 `HelloResponse`，其余是 `Response`。
    #[test]
    fn reply_serializes_hello_flat_and_snapshot_as_response() {
        let hello = Reply::Hello(HelloResponse {
            source_kind: SourceKind::Bridge,
            timeline_schema: TIMELINE_SCHEMA_VERSION,
            granted: vec![Capability::HostPcm],
            host_label: "Vegas Pro 22".into(),
            implementation: "vegas-bridge 0.1.0".into(),
        });
        let value = serde_json::to_value(&hello).unwrap();
        assert_eq!(value["source_kind"], "bridge");
        assert_eq!(value["granted"][0], "host_pcm");
        assert_eq!(value["host_label"], "Vegas Pro 22");
        assert!(value.get("ok").is_none(), "Hello 回包不该带 Response 字段");
        assert!(value.get("revision").is_none());

        let snapshot = Reply::Response(Response {
            ok: true,
            revision: 4,
            ..Default::default()
        });
        let value = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(value["ok"], true);
        assert_eq!(value["revision"], 4);
        assert!(value.get("granted").is_none());
    }

    /// 历史记录（无 `source_kind`）必须读成 `Ara`，否则既有 ARA 实例会全部消失。
    #[test]
    fn legacy_instance_records_default_to_the_ara_source_kind() {
        let legacy = serde_json::json!({
            "instance_id": "3f2a1b9c-7d5e-4a8b-9c0d-1e2f3a4b5c6d",
            "pid": 12345,
            "pipe_name": "\\\\.\\pipe\\HiFiShifter-ARA-12345-3f2a1b9c-7d5e-4a8b-9c0d-1e2f3a4b5c6d",
            "token": "9e8d7c6b-5a49-4837-a6b5-c4d3e2f1a0b9",
            "name": "HiFiShifter / REAPER",
            "heartbeat_ms": 1760000000000_u64,
            "protocol": 1,
        });
        let record: InstanceRecord = serde_json::from_value(legacy).unwrap();
        assert_eq!(record.source_kind, SourceKind::Ara);
        assert_eq!(
            serde_json::from_value::<SourceKind>(serde_json::json!("bridge")).unwrap(),
            SourceKind::Bridge
        );
    }

    /// 内嵌平面是既有形态；`file` 形态与它互斥；同时给出或都不给出都是歧义。
    #[test]
    fn host_pcm_payload_is_inline_or_file_but_never_ambiguous() {
        let inline = HostPcm {
            persistent_id: "ara://a".into(),
            sample_rate: 44100,
            fingerprint: "f".into(),
            planes: vec![vec![0.5; 4]],
            file: None,
        };
        assert!(matches!(inline.payload(), Some(PcmPayload::Inline(p)) if p.len() == 1));
        let text = serde_json::to_string(&inline).unwrap();
        assert!(
            !text.contains("\"file\""),
            "内嵌形态不该写空 file 字段: {text}"
        );
        assert!(
            !text.contains("\"kind\""),
            "内嵌形态字节必须与既有实现一致: {text}"
        );

        let file = HostPcm {
            persistent_id: "ara://a".into(),
            sample_rate: 44100,
            fingerprint: "f".into(),
            planes: Vec::new(),
            file: Some(PcmFile {
                path: PathBuf::from("C:/tmp/a.wav"),
                blake3: "9f2c".into(),
                ephemeral: true,
            }),
        };
        assert!(matches!(file.payload(), Some(PcmPayload::File(f)) if f.ephemeral));
        let text = serde_json::to_string(&file).unwrap();
        assert!(!text.contains("\"planes\""), "引用形态不该写空 planes: {text}");

        let ambiguous = HostPcm {
            planes: vec![vec![0.0; 2]],
            file: Some(PcmFile {
                path: PathBuf::from("a.wav"),
                blake3: "x".into(),
                ephemeral: false,
            }),
            ..file.clone()
        };
        assert!(ambiguous.payload().is_none(), "两种形态同时给出是歧义");
        let empty = HostPcm {
            planes: Vec::new(),
            file: None,
            ..file
        };
        assert!(empty.payload().is_none(), "两种形态都不给是歧义");
    }

    #[test]
    fn oversized_and_truncated_frames_are_rejected_before_parsing() {
        let oversized = ((MAX_FRAME + 1) as u32).to_le_bytes();
        assert!(read_frame::<Request>(&mut oversized.as_slice()).is_err());
        let short = [10, 0, 0, 0, b'{'];
        assert!(read_frame::<Request>(&mut short.as_slice()).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn real_pipe_preserves_revision_and_rejects_wrong_token() {
        let root = std::env::temp_dir().join(format!("hfs-ipc-test-{}", uuid::Uuid::new_v4()));
        let server = Server::start_at("IPC test".into(), root.clone(), |request| {
            Reply::Response(Response {
                ok: matches!(request, Request::Snapshot),
                revision: 17,
                ..Default::default()
            })
        })
        .unwrap();
        let response = exchange(server.record(), &Request::Snapshot).unwrap();
        assert!(response.ok);
        assert_eq!(response.revision, 17);
        let mut invalid = server.record().clone();
        invalid.token = uuid::Uuid::new_v4().to_string();
        assert!(!exchange(&invalid, &Request::Snapshot).unwrap().ok);
        let path = root.join(format!("{}.json", server.record().instance_id));
        assert!(path.exists());
        drop(server);
        assert!(!path.exists());
    }

    /// 协商走的是同一根管道，但回包是裸 `HelloResponse`。
    #[cfg(windows)]
    #[test]
    fn real_pipe_negotiates_capabilities_and_source_kind() {
        let root = std::env::temp_dir().join(format!("hfs-ipc-hello-{}", uuid::Uuid::new_v4()));
        let server = Server::start_at("bridge".into(), root, |request| match request {
            Request::Hello { .. } => Reply::Hello(HelloResponse {
                source_kind: SourceKind::Bridge,
                timeline_schema: TIMELINE_SCHEMA_VERSION,
                granted: vec![Capability::HostPcm, Capability::Transport],
                host_label: "Vegas Pro 22".into(),
                implementation: "test".into(),
            }),
            _ => Reply::Response(Response {
                error: Some("unexpected".into()),
                ..Default::default()
            }),
        })
        .unwrap();
        let hello = hello(server.record(), &[Capability::HostPcm]).unwrap();
        assert_eq!(hello.source_kind, SourceKind::Bridge);
        assert_eq!(hello.host_label, "Vegas Pro 22");
        assert!(hello.grants(Capability::HostPcm));
        assert!(!hello.grants(Capability::Commit));
    }

    /// schema 不一致必须拒连，且要说清该升级哪一侧。
    #[cfg(windows)]
    #[test]
    fn hello_rejects_a_schema_mismatch_and_names_the_side_to_upgrade() {
        let root = std::env::temp_dir().join(format!("hfs-ipc-schema-{}", uuid::Uuid::new_v4()));
        let server = Server::start_at("old".into(), root, |_| {
            Reply::Hello(HelloResponse {
                source_kind: SourceKind::Bridge,
                timeline_schema: TIMELINE_SCHEMA_VERSION + 1,
                granted: Vec::new(),
                host_label: "future".into(),
                implementation: "test".into(),
            })
        })
        .unwrap();
        let error = hello(server.record(), &[]).unwrap_err();
        assert!(error.contains("HiFiShifter"), "{error}");
    }

    #[cfg(windows)]
    #[test]
    fn real_pipe_transfers_pcm_larger_than_the_kernel_pipe_buffer() {
        let root = std::env::temp_dir().join(format!("hfs-ipc-large-{}", uuid::Uuid::new_v4()));
        let server = Server::start_at("large PCM".into(), root, |_| {
            Reply::Response(Response {
                ok: true,
                sources: vec![HostPcm {
                    persistent_id: "ara://large".into(),
                    sample_rate: 44100,
                    planes: vec![vec![0.12345; 88200]],
                    fingerprint: "test".into(),
                    file: None,
                }],
                ..Default::default()
            })
        })
        .unwrap();
        let response = exchange(server.record(), &Request::Snapshot).unwrap();
        assert_eq!(response.sources[0].planes[0].len(), 88200);
        assert_eq!(response.sources[0].planes[0][88199], 0.12345);
    }
}
