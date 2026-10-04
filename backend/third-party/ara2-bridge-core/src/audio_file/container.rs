//! Streaming RIFF/FORM container inspection and iXML rewriting.

use super::{AraChunkSet, ChunkError, ChunkLimits};
use std::io::{Read, Seek, SeekFrom, Write};

const WAVE64_RIFF_GUID: [u8; 16] = [
    b'r', b'i', b'f', b'f', 0x2e, 0x91, 0xcf, 0x11, 0xa5, 0xd6, 0x28, 0xdb, 0x04, 0xc1, 0, 0,
];
const MAX_CONTAINER_CHUNKS: usize = 65_536;
const MAX_DS64_TABLE_ENTRIES: u32 = 65_536;

/// Supported audio-container family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioFileKind {
    /// 32-bit RIFF/WAVE.
    Wave,
    /// RF64 with a required `ds64` chunk.
    Rf64,
    /// BW64 with a required `ds64` chunk.
    Bw64,
    /// Big-endian AIFF.
    Aiff,
    /// Big-endian compressed AIFF.
    Aifc,
}

/// Typed audio-container and embedded-chunk failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AudioFileError {
    /// Underlying stream operation failed.
    #[error("audio-file I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// The container is recognized but intentionally unsupported.
    #[error("unsupported audio container: {0}")]
    Unsupported(&'static str),
    /// Container structure or declared sizes are invalid.
    #[error("invalid audio container: {0}")]
    Invalid(&'static str),
    /// More than one iXML chunk makes selection ambiguous.
    #[error("multiple iXML chunks are ambiguous")]
    AmbiguousIxml,
    /// A rewritten 32-bit size cannot be represented.
    #[error("rewritten audio container exceeds its size representation")]
    SizeOverflow,
    /// A bounded parser limit was exceeded.
    #[error("audio-file limit exceeded: {0}")]
    Limit(&'static str),
    /// The embedded ARA XML is invalid.
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    /// Atomic path replacement refuses symbolic links by default.
    #[error("refusing to replace a symbolic link")]
    SymlinkRefused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Endian {
    Little,
    Big,
}

#[derive(Clone, Debug)]
struct ChunkInfo {
    id: [u8; 4],
    start: u64,
    data_start: u64,
    data_size: u64,
    end: u64,
    large_table_index: Option<usize>,
}

#[derive(Clone, Debug)]
struct Layout {
    kind: AudioFileKind,
    endian: Endian,
    signature: [u8; 4],
    form: [u8; 4],
    file_len: u64,
    chunks: Vec<ChunkInfo>,
    ixml: Option<usize>,
}

impl Layout {
    fn is_large_riff(&self) -> bool {
        matches!(self.kind, AudioFileKind::Rf64 | AudioFileKind::Bw64)
    }
}

/// Reads the sole iXML payload without interpreting its XML.
pub fn read_ixml<R: Read + Seek>(input: &mut R) -> Result<Option<Vec<u8>>, AudioFileError> {
    read_ixml_with_limit(input, ChunkLimits::default().max_xml_bytes)
}

/// Reads the sole iXML payload with an explicit allocation limit.
pub fn read_ixml_with_limit<R: Read + Seek>(
    input: &mut R,
    max_xml_bytes: usize,
) -> Result<Option<Vec<u8>>, AudioFileError> {
    let layout = inspect(input)?;
    let Some(index) = layout.ixml else {
        return Ok(None);
    };
    let chunk = &layout.chunks[index];
    let length = usize::try_from(chunk.data_size).map_err(|_| AudioFileError::SizeOverflow)?;
    if length > max_xml_bytes {
        return Err(AudioFileError::Limit("iXML payload"));
    }
    let mut data = vec![0_u8; length];
    input.seek(SeekFrom::Start(chunk.data_start))?;
    input.read_exact(&mut data)?;
    Ok(Some(data))
}

/// Streams a copy of an audio file while replacing, inserting, or removing its iXML chunk.
///
/// `input` is only read and sought. `output` should be a fresh or already-truncated stream because
/// the generic `Write + Seek` contract cannot truncate an existing destination after a shrink.
pub fn rewrite_ixml<R: Read + Seek, W: Write + Seek>(
    input: &mut R,
    output: &mut W,
    replacement: Option<&[u8]>,
) -> Result<(), AudioFileError> {
    let layout = inspect(input)?;
    let replacement_total = replacement.map(chunk_extent).transpose()?;
    let old_total = layout
        .ixml
        .map(|index| layout.chunks[index].end - layout.chunks[index].start)
        .unwrap_or(0);
    let new_total = layout
        .file_len
        .checked_sub(old_total)
        .and_then(|length| length.checked_add(replacement_total.unwrap_or(0)))
        .ok_or(AudioFileError::SizeOverflow)?;
    let outer_size = new_total.checked_sub(8).ok_or(AudioFileError::Invalid(
        "container is shorter than its header",
    ))?;
    if !layout.is_large_riff() && outer_size > u64::from(u32::MAX) {
        return Err(AudioFileError::SizeOverflow);
    }

    output.seek(SeekFrom::Start(0))?;
    output.write_all(&layout.signature)?;
    if layout.is_large_riff() {
        output.write_all(&u32::MAX.to_le_bytes())?;
    } else {
        write_u32(
            output,
            u32::try_from(outer_size).map_err(|_| AudioFileError::SizeOverflow)?,
            layout.endian,
        )?;
    }
    output.write_all(&layout.form)?;

    let mut replaced = false;
    for (index, chunk) in layout.chunks.iter().enumerate() {
        if Some(index) == layout.ixml {
            if let Some(xml) = replacement {
                write_chunk(
                    output,
                    *b"iXML",
                    xml,
                    layout.endian,
                    chunk.large_table_index.is_some(),
                )?;
            }
            replaced = true;
            continue;
        }
        let output_start = output.stream_position()?;
        copy_range(input, output, chunk.start, chunk.end - chunk.start)?;
        if layout.is_large_riff() && chunk.id == *b"ds64" {
            if chunk.data_size < 28 {
                return Err(AudioFileError::Invalid("ds64 payload is too short"));
            }
            let resume = output.stream_position()?;
            output.seek(SeekFrom::Start(output_start + 8))?;
            output.write_all(&outer_size.to_le_bytes())?;
            if let (Some(table_index), Some(xml)) = (
                layout
                    .ixml
                    .and_then(|index| layout.chunks[index].large_table_index),
                replacement,
            ) {
                let table_offset = u64::try_from(table_index)
                    .map_err(|_| AudioFileError::SizeOverflow)?
                    .checked_mul(12)
                    .and_then(|offset| output_start.checked_add(8 + 28 + 4 + offset))
                    .ok_or(AudioFileError::SizeOverflow)?;
                output.seek(SeekFrom::Start(table_offset))?;
                output.write_all(
                    &u64::try_from(xml.len())
                        .map_err(|_| AudioFileError::SizeOverflow)?
                        .to_le_bytes(),
                )?;
            }
            output.seek(SeekFrom::Start(resume))?;
        }
    }
    if !replaced {
        if let Some(xml) = replacement {
            write_chunk(output, *b"iXML", xml, layout.endian, false)?;
        }
    }
    output.flush()?;
    Ok(())
}

impl AraChunkSet {
    /// Parses the ARA dictionary from the sole iXML chunk in an audio container.
    pub fn from_audio(input: &[u8]) -> Result<Option<Self>, AudioFileError> {
        Self::from_audio_with_limits(input, ChunkLimits::default())
    }

    /// Parses the ARA dictionary with explicit XML and decoded-archive limits.
    pub fn from_audio_with_limits(
        input: &[u8],
        limits: ChunkLimits,
    ) -> Result<Option<Self>, AudioFileError> {
        Self::from_audio_reader(&mut std::io::Cursor::new(input), limits)
    }

    pub(crate) fn from_audio_reader<R: Read + Seek>(
        input: &mut R,
        limits: ChunkLimits,
    ) -> Result<Option<Self>, AudioFileError> {
        let Some(xml) = read_ixml_with_limit(input, limits.max_xml_bytes)? else {
            return Ok(None);
        };
        Ok(Some(Self::parse_with_limits(&xml, limits)?))
    }
}

fn inspect<R: Read + Seek>(input: &mut R) -> Result<Layout, AudioFileError> {
    let file_len = input.seek(SeekFrom::End(0))?;
    input.seek(SeekFrom::Start(0))?;
    let mut prefix = [0_u8; 16];
    let prefix_len = usize::try_from(file_len.min(16)).expect("at most sixteen bytes");
    input.read_exact(&mut prefix[..prefix_len])?;
    if prefix_len == 16 && prefix == WAVE64_RIFF_GUID {
        return Err(AudioFileError::Unsupported("Wave64"));
    }
    if file_len < 12 {
        return Err(AudioFileError::Invalid(
            "file is shorter than a container header",
        ));
    }
    let signature: [u8; 4] = prefix[..4].try_into().expect("four-byte prefix");
    let form: [u8; 4] = prefix[8..12].try_into().expect("twelve-byte prefix");
    let (kind, endian) = match (signature, form) {
        (id, wave) if id == *b"RIFF" && wave == *b"WAVE" => (AudioFileKind::Wave, Endian::Little),
        (id, wave) if id == *b"RF64" && wave == *b"WAVE" => (AudioFileKind::Rf64, Endian::Little),
        (id, wave) if id == *b"BW64" && wave == *b"WAVE" => (AudioFileKind::Bw64, Endian::Little),
        (id, aiff) if id == *b"FORM" && aiff == *b"AIFF" => (AudioFileKind::Aiff, Endian::Big),
        (id, aifc) if id == *b"FORM" && aifc == *b"AIFC" => (AudioFileKind::Aifc, Endian::Big),
        _ => return Err(AudioFileError::Unsupported("unknown")),
    };
    let raw_outer = decode_u32(prefix[4..8].try_into().expect("size bytes"), endian);
    let large = matches!(kind, AudioFileKind::Rf64 | AudioFileKind::Bw64);
    if large && raw_outer != u32::MAX {
        return Err(AudioFileError::Invalid(
            "RF64/BW64 outer size is not 0xFFFFFFFF",
        ));
    }
    let (declared_end, large_sizes) = if large {
        read_ds64(input, file_len)?
    } else {
        (u64::from(raw_outer) + 8, LargeSizes::default())
    };
    if declared_end != file_len {
        return Err(AudioFileError::Invalid(
            "declared container size does not match the stream",
        ));
    }

    let mut chunks = Vec::new();
    let mut cursor = 12_u64;
    let mut ixml = None;
    let mut table_use = vec![false; large_sizes.table.len()];
    while cursor < declared_end {
        if chunks.len() >= MAX_CONTAINER_CHUNKS {
            return Err(AudioFileError::Limit("container chunk count"));
        }
        if declared_end - cursor < 8 {
            return Err(AudioFileError::Invalid("truncated chunk header"));
        }
        input.seek(SeekFrom::Start(cursor))?;
        let mut header = [0_u8; 8];
        input.read_exact(&mut header)?;
        let id: [u8; 4] = header[..4].try_into().expect("four-byte chunk ID");
        let raw_size = decode_u32(header[4..].try_into().expect("size bytes"), endian);
        let (data_size, large_table_index) = if raw_size == u32::MAX {
            if !large {
                return Err(AudioFileError::Invalid(
                    "0xFFFFFFFF chunk size outside RF64/BW64",
                ));
            }
            resolve_large_size(id, &large_sizes, &mut table_use)?
        } else {
            (u64::from(raw_size), None)
        };
        let data_start = cursor + 8;
        let end = data_start
            .checked_add(data_size)
            .and_then(|position| position.checked_add(data_size & 1))
            .ok_or(AudioFileError::SizeOverflow)?;
        if end > declared_end {
            return Err(AudioFileError::Invalid(
                "chunk extends beyond the container",
            ));
        }
        if id == *b"iXML" && ixml.replace(chunks.len()).is_some() {
            return Err(AudioFileError::AmbiguousIxml);
        }
        chunks.push(ChunkInfo {
            id,
            start: cursor,
            data_start,
            data_size,
            end,
            large_table_index,
        });
        cursor = end;
    }
    Ok(Layout {
        kind,
        endian,
        signature,
        form,
        file_len,
        chunks,
        ixml,
    })
}

#[derive(Default)]
struct LargeSizes {
    data: u64,
    table: Vec<([u8; 4], u64)>,
}

fn read_ds64<R: Read + Seek>(
    input: &mut R,
    file_len: u64,
) -> Result<(u64, LargeSizes), AudioFileError> {
    input.seek(SeekFrom::Start(12))?;
    let mut header = [0_u8; 8];
    input.read_exact(&mut header)?;
    if header[..4] != *b"ds64" {
        return Err(AudioFileError::Invalid("RF64/BW64 must begin with ds64"));
    }
    let size = u32::from_le_bytes(header[4..].try_into().expect("size bytes"));
    if size < 28 {
        return Err(AudioFileError::Invalid("ds64 payload is too short"));
    }
    let end = 20_u64
        .checked_add(u64::from(size))
        .ok_or(AudioFileError::SizeOverflow)?;
    if end > file_len {
        return Err(AudioFileError::Invalid("truncated ds64 payload"));
    }
    let mut head = [0_u8; 28];
    input.read_exact(&mut head)?;
    let riff_size = u64::from_le_bytes(head[..8].try_into().expect("riffSize bytes"));
    let data_size = u64::from_le_bytes(head[8..16].try_into().expect("dataSize bytes"));
    let table_len = u32::from_le_bytes(head[24..28].try_into().expect("table length bytes"));
    if table_len > MAX_DS64_TABLE_ENTRIES {
        return Err(AudioFileError::Limit("ds64 size-table entries"));
    }
    let table_bytes = usize::try_from(table_len)
        .map_err(|_| AudioFileError::SizeOverflow)?
        .checked_mul(12)
        .ok_or(AudioFileError::SizeOverflow)?;
    let expected = 28_usize
        .checked_add(table_bytes)
        .ok_or(AudioFileError::SizeOverflow)?;
    if expected > usize::try_from(size).map_err(|_| AudioFileError::SizeOverflow)? {
        return Err(AudioFileError::Invalid("truncated ds64 size table"));
    }
    let mut table = Vec::with_capacity(table_len as usize);
    for _ in 0..table_len {
        let mut row = [0_u8; 12];
        input.read_exact(&mut row)?;
        table.push((
            row[..4].try_into().expect("chunk ID bytes"),
            u64::from_le_bytes(row[4..].try_into().expect("chunk size bytes")),
        ));
    }
    let declared_end = riff_size
        .checked_add(8)
        .ok_or(AudioFileError::SizeOverflow)?;
    Ok((
        declared_end,
        LargeSizes {
            data: data_size,
            table,
        },
    ))
}

fn resolve_large_size(
    id: [u8; 4],
    sizes: &LargeSizes,
    used: &mut [bool],
) -> Result<(u64, Option<usize>), AudioFileError> {
    if id == *b"data" {
        return Ok((sizes.data, None));
    }
    for (index, (table_id, size)) in sizes.table.iter().enumerate() {
        if !used[index] && *table_id == id {
            used[index] = true;
            return Ok((*size, Some(index)));
        }
    }
    Err(AudioFileError::Invalid(
        "large chunk has no ds64 size-table entry",
    ))
}

fn chunk_extent(data: &[u8]) -> Result<u64, AudioFileError> {
    let length = u64::try_from(data.len()).map_err(|_| AudioFileError::SizeOverflow)?;
    if length > u64::from(u32::MAX) {
        return Err(AudioFileError::SizeOverflow);
    }
    8_u64
        .checked_add(length)
        .and_then(|extent| extent.checked_add(length & 1))
        .ok_or(AudioFileError::SizeOverflow)
}

fn write_chunk<W: Write>(
    output: &mut W,
    id: [u8; 4],
    data: &[u8],
    endian: Endian,
    use_large_size: bool,
) -> Result<(), AudioFileError> {
    output.write_all(&id)?;
    let encoded_size = if use_large_size {
        u32::MAX
    } else {
        u32::try_from(data.len()).map_err(|_| AudioFileError::SizeOverflow)?
    };
    write_u32(output, encoded_size, endian)?;
    output.write_all(data)?;
    if data.len() & 1 != 0 {
        output.write_all(&[0])?;
    }
    Ok(())
}

fn copy_range<R: Read + Seek, W: Write>(
    input: &mut R,
    output: &mut W,
    start: u64,
    length: u64,
) -> Result<(), AudioFileError> {
    input.seek(SeekFrom::Start(start))?;
    let copied = std::io::copy(&mut input.take(length), output)?;
    if copied != length {
        return Err(AudioFileError::Invalid("stream ended during chunk copy"));
    }
    Ok(())
}

fn decode_u32(bytes: [u8; 4], endian: Endian) -> u32 {
    match endian {
        Endian::Little => u32::from_le_bytes(bytes),
        Endian::Big => u32::from_be_bytes(bytes),
    }
}

fn write_u32<W: Write>(output: &mut W, value: u32, endian: Endian) -> Result<(), std::io::Error> {
    match endian {
        Endian::Little => output.write_all(&value.to_le_bytes()),
        Endian::Big => output.write_all(&value.to_be_bytes()),
    }
}
