//! ARA 插件与独立 GUI 的有界协议；只传宿主 PCM 和用户参数，不信任客户端几何。

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceRecord {
    pub instance_id: String,
    pub pid: u32,
    pub pipe_name: String,
    pub token: String,
    pub name: String,
    pub heartbeat_ms: u64,
    pub protocol: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostPcm {
    pub persistent_id: String,
    pub sample_rate: u32,
    pub planes: Vec<Vec<f32>>,
    pub fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Snapshot,
    Commit { base_revision: u64, model_revision: u64, timeline: Value },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    pub error: Option<String>,
    pub revision: u64,
    pub model_revision: u64,
    pub timeline: Option<Value>,
    #[serde(default)]
    pub sources: Vec<HostPcm>,
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
    if bytes.is_empty() || bytes.len() > MAX_FRAME { return Err("frame too large".into()); }
    writer.write_all(&(bytes.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    writer.write_all(&bytes).map_err(|e| e.to_string())
}

/// 只在检查长度后分配；短帧不能成为合法请求。
pub fn read_frame<T: for<'a> Deserialize<'a>>(reader: &mut impl Read) -> Result<T, String> {
    let mut prefix = [0; 4];
    reader.read_exact(&mut prefix).map_err(|e| e.to_string())?;
    let size = u32::from_le_bytes(prefix) as usize;
    if size == 0 || size > MAX_FRAME { return Err("invalid frame length".into()); }
    let mut bytes = vec![0; size];
    reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

/// GUI只发现仍有心跳的本机实例，不删除其他会话记录。
pub fn discover() -> Result<Vec<InstanceRecord>, String> {
    let root = instance_dir()?;
    if !root.exists() { return Ok(Vec::new()); }
    let mut instances = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|e| e.to_string())?.flatten() {
        if entry.path().extension().is_none_or(|ext| ext != "json") { continue; }
        if entry.metadata().is_ok_and(|m| m.len() > 8192) { continue; }
        let Ok(bytes) = std::fs::read(entry.path()) else { continue; };
        let Ok(instance) = serde_json::from_slice::<InstanceRecord>(&bytes) else { continue; };
        if validate_record(&instance).is_ok() && now_ms().saturating_sub(instance.heartbeat_ms) < 15000 {
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
    { transport::exchange(instance, request) }
    #[cfg(not(windows))]
    { let _ = request; Err("ARA channel requires Windows".into()) }
}

/// 发现目录可在验收时定向到worktree，独立app与插件须使用同一个目录。
pub fn instance_dir() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("HIFISHIFTER_ARA_INSTANCE_DIR") { return Ok(root.into()); }
    std::env::var_os("LOCALAPPDATA").map(|root| PathBuf::from(root).join("HiFiShifter/ara-instances"))
        .ok_or_else(|| "LOCALAPPDATA unavailable".into())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

fn validate_record(record: &InstanceRecord) -> Result<(), String> {
    let expected = format!(r"\\.\pipe\HiFiShifter-ARA-{}-{}", record.pid, record.instance_id);
    if record.protocol != PROTOCOL || uuid::Uuid::parse_str(&record.instance_id).is_err()
        || uuid::Uuid::parse_str(&record.token).is_err() || record.pipe_name != expected {
        return Err("invalid ARA instance record".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_frame_round_trip_preserves_edit_revision() {
        let request = Request::Commit { base_revision: 7, model_revision: 9, timeline: serde_json::json!({"pitch": [60, 62]}) };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &request).unwrap();
        let actual: Request = read_frame(&mut bytes.as_slice()).unwrap();
        match actual {
            Request::Commit { base_revision, model_revision, timeline } => {
                assert_eq!((base_revision, model_revision), (7, 9));
                assert_eq!(timeline["pitch"], serde_json::json!([60, 62]));
            }
            _ => panic!("lost edit request"),
        }
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
        let server = Server::start_at("IPC test".into(), root.clone(), |request| Response {
            ok: matches!(request, Request::Snapshot), revision: 17, ..Default::default()
        }).unwrap();
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

    #[cfg(windows)]
    #[test]
    fn real_pipe_transfers_pcm_larger_than_the_kernel_pipe_buffer() {
        let root = std::env::temp_dir().join(format!("hfs-ipc-large-{}", uuid::Uuid::new_v4()));
        let server = Server::start_at("large PCM".into(), root, |_| Response {
            ok: true, sources: vec![HostPcm { persistent_id:"ara://large".into(), sample_rate:44100,
                planes:vec![vec![0.12345;88200]],fingerprint:"test".into() }], ..Default::default()
        }).unwrap();
        let response = exchange(server.record(), &Request::Snapshot).unwrap();
        assert_eq!(response.sources[0].planes[0].len(),88200);
        assert_eq!(response.sources[0].planes[0][88199],0.12345);
    }
}
