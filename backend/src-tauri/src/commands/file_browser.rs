use std::path::Path;

/// 目录条目
#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: Option<u64>,
    pub extension: Option<String>,
    pub modified_time: Option<f64>,
}

/// 音频文件元信息
#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AudioFileInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_sec: f64,
    pub total_frames: u64,
}

/// 预览 PCM 数据
#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AudioPreviewData {
    pub sample_rate: u32,
    pub channels: u16,
    pub pcm_base64: String,
}

/// 支持的音频扩展名（用于前端标记）
#[allow(dead_code)]
const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "aac", "aif", "aiff", "m4a"];

fn _is_audio_extension(ext: &str) -> bool {
    AUDIO_EXTENSIONS
        .iter()
        .any(|&e| e.eq_ignore_ascii_case(ext))
}

/// 列出指定目录下的文件和子目录
pub(crate) fn list_directory(dir_path: String) -> Result<Vec<FileEntry>, String> {
    let path = Path::new(&dir_path);
    if !path.is_dir() {
        return Err(format!("Not a directory: {}", dir_path));
    }

    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(path).map_err(|e| e.to_string())?;

    for entry in read_dir {
        let entry = entry.map_err(|e| e.to_string())?;
        let metadata = entry.metadata().map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();

        // 跳过隐藏文件（以 . 开头）
        if name.starts_with('.') {
            continue;
        }

        let is_dir = metadata.is_dir();
        let size = if is_dir { None } else { Some(metadata.len()) };
        let extension = if is_dir {
            None
        } else {
            entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
        };
        let modified_time = metadata.modified().ok().and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs_f64())
        });

        entries.push(FileEntry {
            name,
            path: entry.path().to_string_lossy().into_owned(),
            is_dir,
            size,
            extension,
            modified_time,
        });
    }

    // 目录在前，文件在后；各自按名称排序（不区分大小写）
    entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });

    Ok(entries)
}

/// 在指定目录下递归搜索文件（中间匹配，不区分大小写，忽略隐藏文件和目录）。
/// 最多返回 500 条结果，按文件名排序。
pub(crate) fn search_files_recursive(
    dir_path: String,
    query: String,
) -> Result<Vec<FileEntry>, String> {
    let path = Path::new(&dir_path);
    if !path.is_dir() {
        return Err(format!("Not a directory: {}", dir_path));
    }

    let query_lower = query.to_lowercase();
    let mut results = Vec::new();
    collect_matching_files(path, &query_lower, &mut results, 500);

    results.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(results)
}

fn collect_matching_files(dir: &Path, query: &str, results: &mut Vec<FileEntry>, max: usize) {
    if results.len() >= max {
        return;
    }
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        if results.len() >= max {
            break;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if metadata.is_dir() {
            collect_matching_files(&entry.path(), query, results, max);
        } else {
            // 匹配文件名的 stem（不包含后缀），中间匹配、不区分大小写 → 忽略扩展名
            let path = entry.path();
            let stem_lower = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if stem_lower.contains(query) {
                let extension = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase());
                let modified_time = metadata.modified().ok().and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_secs_f64())
                });
                results.push(FileEntry {
                    name,
                    path: path.to_string_lossy().into_owned(),
                    is_dir: false,
                    size: Some(metadata.len()),
                    extension,
                    modified_time,
                });
            }
        }
    }
}

/// 获取音频文件元信息（时长、采样率、声道数、总帧数）
pub(crate) fn get_audio_file_info(file_path: String) -> Result<AudioFileInfo, String> {
    let path = Path::new(&file_path);
    if !path.is_file() {
        return Err(format!("Not a file: {}", file_path));
    }

    // 使用 decode_audio_f32_interleaved 获取精确的元信息
    // 这里只需要采样率和声道数，但为了简单起见复用现有函数
    // 先尝试 try_read_wav_info 获取快速元信息
    if let Some(info) = crate::audio_utils::try_read_wav_info(path, 0) {
        // try_read_wav_info 不返回声道数，需要通过 decode 获取
        // 对于快速路径，使用 hound 直接读取 header
        let channels = read_channel_count(path).unwrap_or(2);
        return Ok(AudioFileInfo {
            sample_rate: info.sample_rate,
            channels,
            duration_sec: info.duration_sec,
            total_frames: info.total_frames,
        });
    }

    Err(format!("Failed to read audio info: {}", file_path))
}

/// 快速读取音频文件的声道数
fn read_channel_count(path: &Path) -> Option<u16> {
    // WAV: 直接用 hound 读 header
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);

    if is_wav {
        if let Ok(reader) = hound::WavReader::open(path) {
            return Some(reader.spec().channels);
        }
    }

    // 非 WAV: 用 symphonia probe 读取 codec params
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let track = probed.format.default_track()?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(2);
    Some(channels)
}

/// 读取音频预览 PCM 数据（f32 LE interleaved → base64）
/// max_frames 限制最大帧数，默认 480000（~10 秒 @48kHz）
pub(crate) fn read_audio_preview(
    file_path: String,
    max_frames: Option<u32>,
) -> Result<AudioPreviewData, String> {
    let path = Path::new(&file_path);
    if !path.is_file() {
        return Err(format!("Not a file: {}", file_path));
    }

    let max = max_frames.unwrap_or(480_000) as usize;

    let (sample_rate, channels, samples) = crate::audio_utils::decode_audio_f32_interleaved(path)?;

    let total_frames = samples.len() / channels.max(1) as usize;
    let frames_to_use = total_frames.min(max);
    let samples_to_use = frames_to_use * channels.max(1) as usize;

    // 将 f32 PCM 转为 bytes 再编码为 base64
    let bytes: Vec<u8> = samples[..samples_to_use]
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    use base64::Engine as _;
    let pcm_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    Ok(AudioPreviewData {
        sample_rate,
        channels,
        pcm_base64,
    })
}
