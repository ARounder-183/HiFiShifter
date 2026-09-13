//! `.hsrc` 渲染缓存文件格式（编码 / 解码 / 校验）。
//!
//! # 为什么是自定义容器而不是 WAV
//! - 一条渲染条目可能包含两条 stem（主 PCM + 气声噪声），WAV 需要两个文件，
//!   命中主条目却要再开一次气声文件，读写原子性也更差；
//! - 缓存需要携带元数据（键哈希、工程身份、管线指纹、Take id）用于二级校验，
//!   这些塞进 WAV 的自定义 chunk 反而更绕；
//! - 格式固定为 f32 LE，读写都是纯内存搬运，没有编码/解码成本。
//!
//! # 布局（小端）
//! ```text
//! 偏移  长度  字段
//! 0     4     magic          = b"HSRC"
//! 4     2     format_version
//! 6     1     kind           (0=rendered, 1=tension, 2=noise)
//! 7     1     flags          (bit0=有第二 stem, bit1=有 take id)
//! 8     4     sample_rate
//! 12    4     frames         (单声道帧数)
//! 16    1     stems          (1..=2)
//! 17    7     reserved
//! 24    8     param_hash     (键哈希原文，防错配)
//! 32    8     project_id     (工程身份哈希，0=未知)
//! 40    4     pipeline_version
//! 44    4     header_len     (含 take id，便于前向兼容)
//! 48    8     payload_bytes
//! 56    4     checksum       (payload 的 blake3 截断)
//! 60    2     take_id_len
//! 62    2     reserved2
//! 64    ..    take_id (UTF-8)
//! ..    ..    payload: stems × frames × 2 × f32 LE
//! ```

use std::io::{self, Read, Write};

/// 文件魔数。
pub const MAGIC: [u8; 4] = *b"HSRC";
/// 格式版本（与 `render_cache/v{N}` 目录世代同步）。
pub const FORMAT_VERSION: u16 = 1;
/// 固定头长度（take id 之前的字节数）。
pub const FIXED_HEADER_LEN: usize = 64;

const FLAG_SECONDARY: u8 = 0b0000_0001;
const FLAG_TAKE_ID: u8 = 0b0000_0010;

/// 单次转换的样本数（32 KB 缓冲）。
const CHUNK_FRAMES: usize = 8 * 1024;
const CHUNK_BYTES: usize = CHUNK_FRAMES * 4;

/// 缓存条目类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// 整 clip 渲染结果（可含气声 stem）。
    Rendered = 0,
    /// HiFiGAN tension 后处理变体。
    Tension = 1,
    /// 独立的气声噪声 stem（formant 变化时可复用）。
    Noise = 2,
}

impl EntryKind {
    /// 磁盘子目录名。
    pub fn dir_name(self) -> &'static str {
        match self {
            EntryKind::Rendered => "rendered",
            EntryKind::Tension => "tension",
            EntryKind::Noise => "noise",
        }
    }

    /// 中文显示名（统计面板用）。
    pub fn display_name(self) -> &'static str {
        match self {
            EntryKind::Rendered => "合成渲染",
            EntryKind::Tension => "张力变体",
            EntryKind::Noise => "气声噪声",
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(EntryKind::Rendered),
            1 => Some(EntryKind::Tension),
            2 => Some(EntryKind::Noise),
            _ => None,
        }
    }
}

/// 头部元数据（调用方实际需要的字段）。
///
/// 其余字段（键哈希 / 工程身份 / 管线指纹 / 长度 / 校验和）在解析阶段即完成
/// 校验，校验通过后不再向上保留 —— 避免出现"看起来可用但从未被读"的死字段。
#[derive(Debug, Clone)]
pub struct EntryHeader {
    pub kind: EntryKind,
    pub sample_rate: u32,
    pub frames: u32,
    pub take_id: Option<String>,
}

/// 从磁盘读出的完整条目。
#[derive(Debug, Clone)]
pub struct LoadedEntry {
    pub header: EntryHeader,
    /// 主 stem（交错立体声）。
    pub primary: Vec<f32>,
    /// 第二 stem（仅 rendered 的气声噪声）。
    pub secondary: Option<Vec<f32>>,
}

/// 计算 payload 校验和（两条 stem 依序混入）。
pub fn checksum_of(primary: &[f32], secondary: Option<&[f32]>) -> u32 {
    let mut hasher = blake3::Hasher::new();
    update_hasher(&mut hasher, primary);
    if let Some(secondary) = secondary {
        update_hasher(&mut hasher, secondary);
    }
    truncate_hash(&hasher.finalize())
}

fn truncate_hash(hash: &blake3::Hash) -> u32 {
    let bytes = hash.as_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn update_hasher(hasher: &mut blake3::Hasher, data: &[f32]) {
    let mut buf = [0u8; CHUNK_BYTES];
    for chunk in data.chunks(CHUNK_FRAMES) {
        for (i, value) in chunk.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        hasher.update(&buf[..chunk.len() * 4]);
    }
}

fn write_samples<W: Write>(writer: &mut W, data: &[f32]) -> io::Result<()> {
    let mut buf = [0u8; CHUNK_BYTES];
    for chunk in data.chunks(CHUNK_FRAMES) {
        for (i, value) in chunk.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&buf[..chunk.len() * 4])?;
    }
    Ok(())
}

/// 读取 `frames * 2` 个 f32，并同步把原始字节混入校验和。
fn read_samples<R: Read>(
    reader: &mut R,
    frames: u32,
    hasher: &mut blake3::Hasher,
) -> io::Result<Vec<f32>> {
    let total = frames as usize * 2;
    let mut out: Vec<f32> = Vec::with_capacity(total);
    let mut buf = [0u8; CHUNK_BYTES];
    let mut remaining = total;

    while remaining > 0 {
        let want = remaining.min(CHUNK_FRAMES) * 4;
        reader.read_exact(&mut buf[..want])?;
        hasher.update(&buf[..want]);
        for i in 0..remaining.min(CHUNK_FRAMES) {
            let b = &buf[i * 4..i * 4 + 4];
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        remaining -= remaining.min(CHUNK_FRAMES);
    }

    Ok(out)
}

/// 写入一条完整条目（头部 + payload）。
///
/// `primary` / `secondary` 交错立体声；校验和由本函数按 [`checksum_of`] 的口径
/// 计算后写入头部（调用方不必自行计算，避免口径漂移）。
pub fn write_entry<W: Write>(
    writer: &mut W,
    kind: EntryKind,
    sample_rate: u32,
    param_hash: u64,
    project_id: u64,
    pipeline_version: u32,
    take_id: Option<&str>,
    primary: &[f32],
    secondary: Option<&[f32]>,
) -> io::Result<()> {
    let frames = (primary.len() / 2) as u32;
    let stems: u8 = if secondary.is_some() { 2 } else { 1 };
    let payload_bytes = ((primary.len() + secondary.map(|s| s.len()).unwrap_or(0)) * 4) as u64;
    let checksum = checksum_of(primary, secondary);
    let take_bytes = take_id.map(|s| s.as_bytes()).unwrap_or(&[]);

    let mut flags: u8 = 0;
    if secondary.is_some() {
        flags |= FLAG_SECONDARY;
    }
    if take_id.is_some() {
        flags |= FLAG_TAKE_ID;
    }

    let header_len = (FIXED_HEADER_LEN + take_bytes.len()) as u32;
    let mut head = [0u8; FIXED_HEADER_LEN];
    head[0..4].copy_from_slice(&MAGIC);
    head[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    head[6] = kind as u8;
    head[7] = flags;
    head[8..12].copy_from_slice(&sample_rate.to_le_bytes());
    head[12..16].copy_from_slice(&frames.to_le_bytes());
    head[16] = stems;
    head[24..32].copy_from_slice(&param_hash.to_le_bytes());
    head[32..40].copy_from_slice(&project_id.to_le_bytes());
    head[40..44].copy_from_slice(&pipeline_version.to_le_bytes());
    head[44..48].copy_from_slice(&header_len.to_le_bytes());
    head[48..56].copy_from_slice(&payload_bytes.to_le_bytes());
    head[56..60].copy_from_slice(&checksum.to_le_bytes());
    head[60..62].copy_from_slice(&(take_bytes.len() as u16).to_le_bytes());

    writer.write_all(&head)?;
    if !take_bytes.is_empty() {
        writer.write_all(take_bytes)?;
    }
    write_samples(writer, primary)?;
    if let Some(secondary) = secondary {
        write_samples(writer, secondary)?;
    }
    Ok(())
}

/// 读取并校验一条条目。
///
/// # 参数
/// - `expected_param_hash`：调用方期望的键哈希。与文件头不一致说明文件名与内容
///   错配（哈希碰撞或人为改名），按损坏处理 —— 宁可重渲染也不播放错误音频。
/// - `expected_sample_rate`：0 表示不校验。
/// - `expected_pipeline_version`：渲染管线指纹（实现变更后旧条目必须判废，
///   与"指纹混入哈希"形成双重保险）。
/// - `verify_checksum`：是否做 payload 级校验（长度与头部自检始终执行）。
pub fn read_entry<R: Read>(
    reader: &mut R,
    expected_param_hash: u64,
    expected_sample_rate: u32,
    expected_pipeline_version: u32,
    verify_checksum: bool,
) -> io::Result<LoadedEntry> {
    let mut head = [0u8; FIXED_HEADER_LEN];
    reader.read_exact(&mut head)?;

    if head[0..4] != MAGIC {
        return Err(invalid("magic mismatch"));
    }
    let format_version = u16::from_le_bytes([head[4], head[5]]);
    if format_version != FORMAT_VERSION {
        return Err(invalid("format version mismatch"));
    }
    let kind = EntryKind::from_u8(head[6]).ok_or_else(|| invalid("unknown entry kind"))?;
    let flags = head[7];
    let sample_rate = u32::from_le_bytes([head[8], head[9], head[10], head[11]]);
    let frames = u32::from_le_bytes([head[12], head[13], head[14], head[15]]);
    let stems = head[16];
    let param_hash = u64::from_le_bytes(head[24..32].try_into().unwrap());
    let pipeline_version = u32::from_le_bytes(head[40..44].try_into().unwrap());
    let header_len = u32::from_le_bytes(head[44..48].try_into().unwrap());
    let payload_bytes = u64::from_le_bytes(head[48..56].try_into().unwrap());
    let checksum = u32::from_le_bytes(head[56..60].try_into().unwrap());
    let take_id_len = u16::from_le_bytes([head[60], head[61]]) as usize;

    if param_hash != expected_param_hash {
        return Err(invalid("param hash mismatch"));
    }
    if expected_sample_rate != 0 && sample_rate != expected_sample_rate {
        return Err(invalid("sample rate mismatch"));
    }
    if pipeline_version != expected_pipeline_version {
        return Err(invalid("pipeline version mismatch"));
    }
    if frames == 0 || stems == 0 || stems > 2 {
        return Err(invalid("invalid frame/stem count"));
    }
    if (flags & FLAG_SECONDARY != 0) != (stems == 2) {
        return Err(invalid("stem flag/count mismatch"));
    }
    let expected_payload = frames as u64 * 2 * stems as u64 * 4;
    if payload_bytes != expected_payload {
        return Err(invalid("payload length mismatch"));
    }
    if header_len as usize != FIXED_HEADER_LEN + take_id_len {
        return Err(invalid("header length mismatch"));
    }

    let take_id = if flags & FLAG_TAKE_ID != 0 && take_id_len > 0 {
        let mut buf = vec![0u8; take_id_len];
        reader.read_exact(&mut buf)?;
        Some(String::from_utf8(buf).map_err(|_| invalid("take id is not utf-8"))?)
    } else {
        None
    };

    let mut hasher = blake3::Hasher::new();
    let primary = read_samples(reader, frames, &mut hasher)?;
    let secondary = if stems == 2 {
        Some(read_samples(reader, frames, &mut hasher)?)
    } else {
        None
    };

    if verify_checksum {
        let actual = truncate_hash(&hasher.finalize());
        if actual != checksum {
            return Err(invalid("checksum mismatch"));
        }
    }

    Ok(LoadedEntry {
        header: EntryHeader {
            kind,
            sample_rate,
            frames,
            take_id,
        },
        primary,
        secondary,
    })
}

/// 只读取头部（扫描/统计用，不加载 payload）。
///
/// 返回 `(kind, sample_rate, frames, param_hash, project_id, pipeline_version)`。
pub fn read_header_only<R: Read>(
    reader: &mut R,
) -> io::Result<(EntryKind, u32, u32, u64, u64, u32)> {
    let mut head = [0u8; FIXED_HEADER_LEN];
    reader.read_exact(&mut head)?;
    if head[0..4] != MAGIC {
        return Err(invalid("magic mismatch"));
    }
    if u16::from_le_bytes([head[4], head[5]]) != FORMAT_VERSION {
        return Err(invalid("format version mismatch"));
    }
    let kind = EntryKind::from_u8(head[6]).ok_or_else(|| invalid("unknown entry kind"))?;
    let sample_rate = u32::from_le_bytes([head[8], head[9], head[10], head[11]]);
    let frames = u32::from_le_bytes([head[12], head[13], head[14], head[15]]);
    let param_hash = u64::from_le_bytes(head[24..32].try_into().unwrap());
    let project_id = u64::from_le_bytes(head[32..40].try_into().unwrap());
    let pipeline_version = u32::from_le_bytes(head[40..44].try_into().unwrap());
    Ok((kind, sample_rate, frames, param_hash, project_id, pipeline_version))
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("hsrc: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload(frames: u32) -> Vec<f32> {
        (0..frames as usize * 2)
            .map(|i| (i as f32 * 0.001).sin())
            .collect()
    }

    /// 生产路径口径的读取（带管线指纹与校验和）。
    fn read<R: Read>(reader: &mut R, hash: u64, sample_rate: u32) -> io::Result<LoadedEntry> {
        read_entry(
            reader,
            hash,
            sample_rate,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            true,
        )
    }

    /// 关闭 payload 校验的读取（验证"关校验仍执行长度自检"）。
    fn read_without_checksum<R: Read>(
        reader: &mut R,
        hash: u64,
        sample_rate: u32,
    ) -> io::Result<LoadedEntry> {
        read_entry(
            reader,
            hash,
            sample_rate,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            false,
        )
    }

    #[test]
    fn entry_round_trips_with_two_stems_and_take_id() {
        let primary = sample_payload(64);
        let secondary = sample_payload(64);
        let mut buf: Vec<u8> = Vec::new();
        write_entry(
            &mut buf,
            EntryKind::Rendered,
            48_000,
            0xDEAD_BEEF_0000_0001,
            42,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            Some("take-1"),
            &primary,
            Some(&secondary),
        )
        .expect("write");

        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = read(&mut cursor, 0xDEAD_BEEF_0000_0001, 48_000).expect("read");
        assert_eq!(loaded.header.kind, EntryKind::Rendered);
        assert_eq!(loaded.header.frames, 64);
        assert_eq!(loaded.header.take_id.as_deref(), Some("take-1"));
        assert_eq!(loaded.primary, primary);
        assert_eq!(loaded.secondary.as_deref(), Some(secondary.as_slice()));
    }

    #[test]
    fn entry_round_trips_without_secondary_stem() {
        let primary = sample_payload(16);
        let mut buf: Vec<u8> = Vec::new();
        write_entry(
            &mut buf,
            EntryKind::Noise,
            44_100,
            7,
            0,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            None,
            &primary,
            None,
        )
        .expect("write");

        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = read(&mut cursor, 7, 44_100).expect("read");
        assert_eq!(loaded.header.kind, EntryKind::Noise);
        assert!(loaded.secondary.is_none());
        assert_eq!(loaded.primary, primary);
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let primary = sample_payload(64);
        let mut buf: Vec<u8> = Vec::new();
        write_entry(
            &mut buf,
            EntryKind::Rendered,
            48_000,
            9,
            0,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            None,
            &primary,
            None,
        )
        .expect("write");

        // 截断尾部 → 读取必须失败（长度自检或校验和任一先触发）。
        buf.truncate(buf.len() - 16);
        let mut cursor = std::io::Cursor::new(&buf);
        assert!(read(&mut cursor, 9, 48_000).is_err());
    }

    #[test]
    fn corrupted_payload_fails_checksum() {
        let primary = sample_payload(32);
        let mut buf: Vec<u8> = Vec::new();
        write_entry(
            &mut buf,
            EntryKind::Rendered,
            48_000,
            11,
            0,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            None,
            &primary,
            None,
        )
        .expect("write");
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;

        let mut cursor = std::io::Cursor::new(&buf);
        assert!(read(&mut cursor, 11, 48_000).is_err());
        // 关闭校验和时不校验内容（长度仍自检）——仅用于极端性能取舍。
        let mut cursor = std::io::Cursor::new(&buf);
        assert!(read_without_checksum(&mut cursor, 11, 48_000).is_ok());
    }

    #[test]
    fn mismatched_param_hash_or_sample_rate_is_rejected() {
        let primary = sample_payload(8);
        let mut buf: Vec<u8> = Vec::new();
        write_entry(
            &mut buf,
            EntryKind::Rendered,
            48_000,
            13,
            0,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION,
            None,
            &primary,
            None,
        )
        .expect("write");

        let mut cursor = std::io::Cursor::new(&buf);
        assert!(read(&mut cursor, 14, 48_000).is_err());
        let mut cursor = std::io::Cursor::new(&buf);
        assert!(read(&mut cursor, 13, 44_100).is_err());
    }

    #[test]
    fn pipeline_version_mismatch_is_rejected() {
        // 管线指纹不匹配 = 旧实现产物：即使键哈希碰巧一致也必须判废。
        let primary = sample_payload(8);
        let mut buf: Vec<u8> = Vec::new();
        write_entry(
            &mut buf,
            EntryKind::Rendered,
            48_000,
            17,
            0,
            crate::synth_clip_cache::RENDER_PIPELINE_VERSION + 1,
            None,
            &primary,
            None,
        )
        .expect("write");

        let mut cursor = std::io::Cursor::new(&buf);
        assert!(read(&mut cursor, 17, 48_000).is_err());
    }
}
