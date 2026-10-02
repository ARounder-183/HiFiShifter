use crate::search::{
    match_translit, translit, MatchInfo, MatchOptions, Query, SearchMode, SearchOptions,
    TranslitOptions,
};
use std::cmp::Ordering;
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
    /// 搜索命中说明（仅搜索路径产出；目录列表为 `None`）。
    ///
    /// 【为什么要回传】转写匹配最大的风险不是误命中，而是「用户不知道为什么这条
    /// 会出来」—— 打 `zge` 冒出「主歌.wav」时，界面需要能说明「匹配拼音 zhuge」。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_info: Option<MatchInfo>,
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

/// 「计算机」虚拟路径：前端用它表示盘符根的上一级（Windows 的「此电脑」层）。
///
/// 必须与前端 `fileBrowserSlice.ts` 的 `FILE_BROWSER_COMPUTER_PATH` 一字不差 ——
/// 这是跨进程约定的哨兵值，不是磁盘上真实存在的路径。
pub(crate) const COMPUTER_VIRTUAL_PATH: &str = "computer://";

/// `list_directory` 的选项。
#[derive(serde::Deserialize, Default, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ListDirectoryOptions {
    /// 是否列出隐藏项。默认否（与系统文件管理器一致）。
    #[serde(default)]
    pub include_hidden: bool,
}

/// 该条目是否应被当作"隐藏"而不列出。
///
/// 【Windows 为什么不能只看点开头】Windows 的隐藏是一个文件属性
/// （`FILE_ATTRIBUTE_HIDDEN`），`desktop.ini` / `Thumbs.db` 这类都不以点开头。
/// 只判前缀等于对 Windows 用户完全没生效。
///
/// 【系统文件为什么不随开关显示】`System Volume Information` / `$RECYCLE.BIN` 带
/// `FILE_ATTRIBUTE_SYSTEM`，即使用户打开"显示隐藏文件"也不该看到 —— 资源管理器
/// 为此另有一个默认关闭的"隐藏受保护的操作系统文件"选项。与其再加一个开关，
/// 这里直接始终隐藏：显示它们只会让人误删。
#[cfg(windows)]
fn is_hidden_entry(metadata: &std::fs::Metadata, name: &str) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
    const FILE_ATTRIBUTE_SYSTEM: u32 = 0x4;
    let attributes = metadata.file_attributes();
    if attributes & FILE_ATTRIBUTE_SYSTEM != 0 {
        return true;
    }
    name.starts_with('.') || attributes & FILE_ATTRIBUTE_HIDDEN != 0
}

/// 非 Windows：点开头即隐藏（macOS / Linux 的惯例）。
#[cfg(not(windows))]
fn is_hidden_entry(_metadata: &std::fs::Metadata, name: &str) -> bool {
    name.starts_with('.')
}

/// 列出指定目录下的文件和子目录
pub(crate) fn list_directory(
    dir_path: String,
    options: Option<ListDirectoryOptions>,
) -> Result<Vec<FileEntry>, String> {
    if dir_path == COMPUTER_VIRTUAL_PATH {
        return list_logical_drives();
    }
    let include_hidden = options.unwrap_or_default().include_hidden;
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

        if !include_hidden && is_hidden_entry(&metadata, &name) {
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
            match_info: None,
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

// ===================== 写操作（新建 / 重命名 / 删除） =====================

/// 校验一个用户输入的文件/目录名。
///
/// 【为什么要显式校验而不是交给文件系统】`fs::rename` 对非法名会给出平台相关的
/// 错误串（Windows 是"文件名、目录名或卷标语法不正确。"），直接弹给用户没有信息量；
/// 而"含路径分隔符"这种情况更危险 —— 允许 `..\..\x` 就等于把重命名变成移动。
/// 在调用点之前拦下来，错误才是可读且可控的。
fn validate_entry_name(name: &str) -> Result<&str, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("empty_name".to_string());
    }
    if trimmed == "." || trimmed == ".." {
        return Err("invalid_name".to_string());
    }
    // 路径分隔符与 Windows 非法字符。`/` 与 `\` 必须挡：否则重命名等价于移动。
    if trimmed.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) || trimmed.contains('\0') {
        return Err("invalid_name".to_string());
    }
    // Windows 保留设备名（含带扩展名的形态，如 `CON.txt`）。
    let stem = trimmed.split('.').next().unwrap_or(trimmed);
    let upper = stem.to_ascii_uppercase();
    const RESERVED: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    if RESERVED.contains(&upper.as_str()) {
        return Err("invalid_name".to_string());
    }
    // 尾随点/空格在 Windows 上会被静默吞掉，导致"改完名字不对"。
    if trimmed.ends_with('.') || trimmed.ends_with(' ') {
        return Err("invalid_name".to_string());
    }
    Ok(trimmed)
}

/// 应用自身拥有的目录（配置 / 日志 / 渲染缓存 / 临时）。
///
/// 【为什么需要】这些目录对用户没有意义，删掉只会让应用以奇怪的方式坏掉；而它们
/// 又常常正好出现在用户浏览的路径附近（便携版的 exe 同级）。
fn app_owned_dirs(config_dir: Option<&Path>) -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Some(dir) = config_dir {
        dirs.push(dir.to_path_buf());
    }
    if let Some(dir) = crate::logging::log_dir() {
        dirs.push(dir.to_path_buf());
    }
    dirs.push(crate::render_cache::current_dir());
    dirs.push(std::env::temp_dir().join("hifishifter"));
    dirs
}

/// 目标路径是否落在应用自身目录之内（含其本身）。
fn is_inside_app_dirs(path: &Path, config_dir: Option<&Path>) -> bool {
    let target = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    for candidate in app_owned_dirs(config_dir) {
        let dir = candidate.canonicalize().unwrap_or(candidate);
        if target.starts_with(&dir) {
            return true;
        }
    }
    false
}

/// 在 `parent_dir` 下新建目录，返回新目录的绝对路径。
pub(crate) fn create_directory(
    parent_dir: String,
    name: String,
    config_dir: Option<&Path>,
) -> Result<String, String> {
    if parent_dir == COMPUTER_VIRTUAL_PATH {
        return Err("virtual_path".to_string());
    }
    let name = validate_entry_name(&name)?;
    let parent = Path::new(&parent_dir);
    if !parent.is_dir() {
        return Err(format!("Not a directory: {}", parent_dir));
    }
    if is_inside_app_dirs(parent, config_dir) {
        return Err("protected_path".to_string());
    }
    let target = parent.join(name);
    if target.exists() {
        return Err("name_exists".to_string());
    }
    std::fs::create_dir(&target).map_err(|e| e.to_string())?;
    Ok(target.to_string_lossy().into_owned())
}

/// 把 `path` 重命名为同目录下的 `new_name`，返回新路径。
pub(crate) fn rename_path(
    path: String,
    new_name: String,
    config_dir: Option<&Path>,
) -> Result<String, String> {
    let source = Path::new(&path);
    if !source.exists() {
        return Err("not_found".to_string());
    }
    if is_inside_app_dirs(source, config_dir) {
        return Err("protected_path".to_string());
    }
    let new_name = validate_entry_name(&new_name)?;
    let parent = source
        .parent()
        .ok_or_else(|| "cannot rename a root path".to_string())?;
    let target = parent.join(new_name);
    if target == source {
        // 名字没变：当作成功（用户可能只改了大小写，Windows 上这两者相同）。
        return Ok(path);
    }
    if target.exists() {
        return Err("name_exists".to_string());
    }
    std::fs::rename(source, &target).map_err(|e| e.to_string())?;
    Ok(target.to_string_lossy().into_owned())
}

/// 把一批路径移入系统回收站 / 废纸篓。
///
/// 【为什么默认是回收站而不是永久删除】文件浏览器里删掉的是**用户素材**，不在本
/// 应用的撤销栈里 —— 永久删除不可逆。要永久删除需显式传 `permanent: true`
/// （前端对应 Shift+Delete）。
///
/// 【为什么用 `trash` crate】回收站是各平台差异极大的系统集成：Windows 是
/// `IFileOperation`、macOS 是 `NSFileManager`、Linux 是 XDG trash 规范（含跨挂载点
/// 的 `.Trash-$uid`）。自己实现等于把三个平台各写一遍且难以验证。
pub(crate) fn delete_paths(
    paths: Vec<String>,
    permanent: bool,
    config_dir: Option<&Path>,
) -> serde_json::Value {
    let mut deleted = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed == COMPUTER_VIRTUAL_PATH {
            continue;
        }
        let path = Path::new(trimmed);
        if !path.exists() {
            errors.push(format!("not found: {trimmed}"));
            continue;
        }
        if is_inside_app_dirs(path, config_dir) {
            errors.push(format!("protected path: {trimmed}"));
            continue;
        }
        let result = if permanent {
            if path.is_dir() {
                std::fs::remove_dir_all(path)
            } else {
                std::fs::remove_file(path)
            }
        } else {
            trash::delete(path).map_err(|e| std::io::Error::other(e.to_string()))
        };
        match result {
            Ok(()) => deleted += 1,
            Err(e) => errors.push(format!("{trimmed}: {e}")),
        }
    }

    if errors.is_empty() {
        serde_json::json!({ "ok": true, "deleted": deleted })
    } else {
        serde_json::json!({ "ok": false, "deleted": deleted, "error": errors.join("; ") })
    }
}

/// 列出全部逻辑盘符（「计算机」虚拟层的内容）。
///
/// Windows 用 `GetLogicalDriveStringsW` 枚举，它**只读系统卷信息、不触碰磁盘**：
/// 断开的网络映射盘也会被列出（与资源管理器一致），逐个探测反而会让命令卡在
/// 已失效的网络路径上数秒。非 Windows 没有「计算机」层 —— `/` 已是文件系统
/// 顶端，前端永远不会导航到这里；返回空列表兜底。
#[cfg(windows)]
fn list_logical_drives() -> Result<Vec<FileEntry>, String> {
    use windows::Win32::Storage::FileSystem::GetLogicalDriveStringsW;

    // 每个盘符形如 `C:\`（4 个 u16 + 分隔 NUL），26 个字母加结尾双 NUL 足够。
    let mut buffer = [0u16; 26 * 4 + 1];
    // SAFETY: buffer 以可写切片传入；函数只填缓冲、不保留指针，返回写入长度。
    let len = unsafe { GetLogicalDriveStringsW(Some(&mut buffer)) } as usize;
    if len == 0 {
        return Err("Failed to enumerate logical drives".to_string());
    }
    let len = len.min(buffer.len());

    let mut entries = Vec::new();
    let mut start = 0;
    while start < len && buffer[start] != 0 {
        let end = buffer[start..len]
            .iter()
            .position(|&c| c == 0)
            .map(|p| start + p)
            .unwrap_or(len);
        let root: String = String::from_utf16_lossy(&buffer[start..end]);
        // 行展示走 `{name}{isDir ? "/" : ""}`，用 `C:` 作名字得到「C:/」；
        // path 必须带反斜杠，`Path::is_dir()` 对裸 `C:` 的解释依赖进程当前目录。
        let path = root;
        entries.push(FileEntry {
            name: path.trim_end_matches('\\').to_string(),
            path,
            is_dir: true,
            size: None,
            extension: None,
            modified_time: None,
            match_info: None,
        });
        start = end + 1;
    }

    Ok(entries)
}

#[cfg(not(windows))]
fn list_logical_drives() -> Result<Vec<FileEntry>, String> {
    Ok(Vec::new())
}

/// 命中候选：档位分值 + 排序键 + 条目。
struct Scored {
    score: u8,
    name_lower: String,
    entry: FileEntry,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 「更大」= 更该排在前面：分数高的在前；同分按名称字典序（不区分大小写）；
/// 再同则按路径 —— 路径在同一棵目录树里唯一，因此这是全序，排序结果可复现。
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.name_lower.cmp(&self.name_lower))
            .then_with(|| other.entry.path.cmp(&self.entry.path))
    }
}

/// 一条文件名与查询串的匹配器。
///
/// 【为什么分成三态而不是一个函数】「关闭转写」承诺与旧行为一致，就应当走旧代码
/// 路径本身（`to_lowercase().contains()`），而不是「转写实现恰好退化成的样子」——
/// 后者会在某次转写改动里悄悄偏离承诺。
enum NameMatcher {
    /// 空查询：不过滤。前端正则模式就是这样调用的（过滤在前端做）。
    All,
    /// `mode = off`：与转写功能上线前逐字节一致的小写子串匹配。
    Legacy { query_lower: String },
    /// 转写 + 分档匹配。
    Translit {
        query: Query,
        mode: SearchMode,
        opts: TranslitOptions,
    },
}

impl NameMatcher {
    fn new(raw_query: &str, options: &MatchOptions) -> Self {
        let trimmed = raw_query.trim();
        if trimmed.is_empty() {
            return Self::All;
        }
        if options.mode == SearchMode::Off {
            return Self::Legacy {
                query_lower: raw_query.to_lowercase(),
            };
        }
        Self::Translit {
            query: Query::new(trimmed, options),
            mode: options.mode,
            opts: options.translit,
        }
    }

    /// 命中则返回 `(档位分值, 命中说明)`。
    fn score(&self, stem: &str) -> Option<(u8, Option<MatchInfo>)> {
        match self {
            Self::All => Some((0, None)),
            Self::Legacy { query_lower } => stem
                .to_lowercase()
                .contains(query_lower.as_str())
                .then_some((crate::search::SCORE_LITERAL, None)),
            Self::Translit { query, mode, opts } => {
                let forms = translit(stem, opts);
                match_translit(&forms, query, *mode).map(|info| (info.score, Some(info)))
            }
        }
    }
}

/// 命中数的超采样倍数：见 `search_files_recursive` 的注释。
const SEARCH_OVERSCAN: usize = 4;

/// 在指定目录下递归搜索文件。
///
/// 匹配文件名 stem（忽略扩展名）、跳过隐藏项；`options` 缺省或 `mode = off` 时与
/// 转写功能上线前的行为一致（小写子串、按名称排序）。
pub(crate) fn search_files_recursive(
    dir_path: String,
    query: String,
    options: Option<SearchOptions>,
) -> Result<Vec<FileEntry>, String> {
    let path = Path::new(&dir_path);
    if !path.is_dir() {
        return Err(format!("Not a directory: {}", dir_path));
    }

    let options = options.unwrap_or_default();
    let match_options = options.match_options();
    let max = options.max_results();
    let matcher = NameMatcher::new(&query, &match_options);

    // 【为什么要「多收几倍再截断」】旧实现遇到第 500 个命中就停止遍历，于是返回
    // 哪 500 条取决于目录遍历顺序 —— 真正最像的那条可能排在后面。这里把停止阈值
    // 放宽到 4 倍（遍历成本仍有界：命中数达到 4×上限即停），再按相关度取前 N。
    let stop_at = max.saturating_mul(SEARCH_OVERSCAN);
    let mut hits: Vec<Scored> = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_matching_files(path, &matcher, &mut hits, stop_at, &mut visited, 0);

    hits.sort_by(|a, b| b.cmp(a));
    hits.truncate(max);
    Ok(hits.into_iter().map(|scored| scored.entry).collect())
}

/// 递归深度上限：防止在极深目录树上无界遍历。
const MAX_SEARCH_DEPTH: usize = 32;

fn collect_matching_files(
    dir: &Path,
    matcher: &NameMatcher,
    hits: &mut Vec<Scored>,
    stop_at: usize,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
    depth: usize,
) {
    if hits.len() >= stop_at || depth >= MAX_SEARCH_DEPTH {
        return;
    }
    // 环防护：junction/符号链接目录环会让无防护的递归栈溢出崩溃。
    // canonicalize 把不同写法/链接指向同一物理目录的路径归一。
    let Ok(dir_key) = dir.canonicalize() else {
        return;
    };
    if !visited.insert(dir_key) {
        return;
    }
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        if hits.len() >= stop_at {
            break;
        }
        // file_type() 不跟随符号链接/junction，避免把链接目录当作真实目录深入。
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if file_type.is_dir() {
            collect_matching_files(&entry.path(), matcher, hits, stop_at, visited, depth + 1);
        } else {
            // 匹配文件名的 stem（不包含后缀）→ 忽略扩展名，与旧实现一致。
            let path = entry.path();
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let Some((score, match_info)) = matcher.score(&stem) else {
                continue;
            };
            let extension = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            let modified_time = entry.metadata().ok().and_then(|m| {
                m.modified().ok().and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_secs_f64())
                })
            });
            let size = entry.metadata().ok().map(|m| m.len());
            hits.push(Scored {
                score,
                name_lower: name.to_lowercase(),
                entry: FileEntry {
                    name,
                    path: path.to_string_lossy().into_owned(),
                    is_dir: false,
                    size,
                    extension,
                    modified_time,
                    match_info,
                },
            });
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

    // 非 WAV（音频与视频容器）统一通过 Symphonia 读取音轨参数。
    crate::media::probe_media(path, 0, None)
        .map(|info| info.channels)
        .or(Some(2))
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

    // 试听只需前 `max` 帧：一律走**限界前缀解码**。视频分支一直如此，音频分支
    // 此前是整文件解码（`decode_audio_f32_interleaved`）——一条 1 小时音轨会因此
    // 把 1.27GB PCM 读进内存，只为取前 10 秒。
    let (sample_rate, channels, samples) =
        crate::media::decode_media_audio_prefix_f32(path, None, max)?;

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

/// 列出媒体文件（尤其是视频）中的全部音轨。
pub(crate) fn list_media_audio_streams(
    file_path: String,
) -> Result<Vec<crate::media::MediaAudioStream>, String> {
    let path = Path::new(&file_path);
    if !path.is_file() {
        return Err(format!("Not a file: {}", file_path));
    }
    crate::media::list_audio_streams(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{SearchMode, SearchOptions};

    /// 建一个带固定文件的临时目录。文件名刻意混合中 / 日 / 韩 / 拉丁，
    /// 因为本模块的职责正是「遍历 + 匹配」，只测纯拉丁会漏掉整条转写路径。
    fn fixture() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hifishifter_search_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(dir.join("子目录")).expect("temp dir");
        for name in [
            "主歌_vocal01.wav",
            "主歌_vocal02.wav",
            "副歌.wav",
            "ボーカル.wav",
            "한국어.wav",
            "vocal_take.wav",
            "readme.txt",
        ] {
            std::fs::write(dir.join(name), b"x").expect("write file");
        }
        // 隐藏文件必须被跳过（与改动前一致）。
        std::fs::write(dir.join(".hidden.wav"), b"x").expect("write hidden");
        std::fs::write(dir.join("子目录").join("深处的主歌.wav"), b"x").expect("write nested");
        dir
    }

    fn names(entries: &[FileEntry]) -> Vec<&str> {
        entries.iter().map(|entry| entry.name.as_str()).collect()
    }

    fn search(
        dir: &std::path::Path,
        query: &str,
        options: Option<SearchOptions>,
    ) -> Vec<FileEntry> {
        search_files_recursive(
            dir.to_string_lossy().into_owned(),
            query.to_string(),
            options,
        )
        .expect("search")
    }

    #[test]
    fn default_options_match_pinyin() {
        let dir = fixture();
        let hits = search(&dir, "zhuge", None);
        // 主形态命中优先：两条「主歌」都在，嵌套的那条也算。
        assert!(
            names(&hits).contains(&"主歌_vocal01.wav"),
            "命中: {:?}",
            names(&hits)
        );
        assert!(
            names(&hits).contains(&"深处的主歌.wav"),
            "命中: {:?}",
            names(&hits)
        );
        // 不含「主歌」读音的文件不得混进来。
        assert!(!names(&hits).contains(&"readme.txt"));
    }

    #[test]
    fn initials_and_romaji_and_choseong() {
        let dir = fixture();
        assert!(names(&search(&dir, "zg", None)).contains(&"主歌_vocal01.wav"));
        assert!(names(&search(&dir, "bokaru", None)).contains(&"ボーカル.wav"));
        assert!(names(&search(&dir, "hg", None)).contains(&"한국어.wav"));
    }

    #[test]
    fn literal_still_wins_and_hidden_files_are_skipped() {
        let dir = fixture();
        let hits = search(&dir, "vocal", None);
        assert!(names(&hits).contains(&"vocal_take.wav"));
        assert!(names(&hits).contains(&"主歌_vocal01.wav"));
        assert!(!names(&hits).contains(&".hidden.wav"));
        // 匹配的是 stem：扩展名不参与。
        assert!(!names(&search(&dir, "txt", None)).contains(&"readme.txt"));
    }

    #[test]
    fn match_info_is_reported_for_explainability() {
        let dir = fixture();
        let hits = search(&dir, "zhuge", None);
        let entry = hits
            .iter()
            .find(|entry| entry.name == "主歌_vocal01.wav")
            .expect("命中");
        let info = entry.match_info.as_ref().expect("转写命中应带说明");
        assert_eq!(info.kind, crate::search::MatchKind::Pinyin);
        assert_eq!(info.form, "zhuge");
    }

    #[test]
    fn off_mode_is_literal_only() {
        let dir = fixture();
        let options = Some(SearchOptions {
            mode: Some(SearchMode::Off),
            ..SearchOptions::default()
        });
        assert!(search(&dir, "zhuge", options.clone()).is_empty());
        assert!(!search(&dir, "vocal", options).is_empty());
    }

    #[test]
    fn empty_query_returns_files_only() {
        let dir = fixture();
        let hits = search(&dir, "", None);
        assert!(names(&hits).contains(&"readme.txt"));
        // 递归搜索只返回**文件**：目录只被用来深入遍历，不进结果集
        // （与改动前一致，也是「搜到的东西都能拖进时间轴」的前提）。
        assert!(!names(&hits).contains(&"子目录"));
        assert!(names(&hits).contains(&"深处的主歌.wav"));
    }

    #[test]
    fn relevance_order_puts_the_prefix_match_first() {
        let dir = fixture();
        let hits = search(&dir, "zhuge", None);
        // 「主歌…」开头的两条走全拼前缀档；嵌套的「深处的主歌」只走子串档，
        // 因此必须排在其后。
        let first_nested = hits.iter().position(|entry| entry.name == "深处的主歌.wav");
        let last_prefix = hits
            .iter()
            .rposition(|entry| entry.name.starts_with("主歌"));
        assert!(last_prefix < first_nested, "命中顺序: {:?}", names(&hits));
    }

    #[test]
    fn max_results_is_respected() {
        let dir = fixture();
        let options = Some(SearchOptions {
            max_results: Some(2),
            ..SearchOptions::default()
        });
        assert_eq!(search(&dir, "zhuge", options).len(), 2);
    }

    #[test]
    fn computer_virtual_path_lists_drives() {
        // 「计算机」哨兵不落在真实文件系统上，list_directory 必须拦截并返回盘符。
        let entries =
            list_directory(COMPUTER_VIRTUAL_PATH.to_string(), None).expect("computer level");
        let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
        #[cfg(windows)]
        assert!(
            !names.is_empty(),
            "Windows 必须至少列出一个盘符，实际: {names:?}"
        );
        for entry in &entries {
            assert!(entry.is_dir, "盘符应表现为目录: {}", entry.name);
            // path 必须能被再次 list_directory（从「计算机」进入盘符）。
            assert!(
                Path::new(&entry.path).is_dir(),
                "盘符路径应有效: {}",
                entry.path
            );
        }
    }

    #[test]
    fn hidden_files_are_listed_only_when_asked() {
        let dir = fixture();
        let visible = list_directory(dir.to_string_lossy().into_owned(), None).expect("list");
        assert!(
            !names(&visible).contains(&".hidden.wav"),
            "默认不列出隐藏文件"
        );

        let all = list_directory(
            dir.to_string_lossy().into_owned(),
            Some(ListDirectoryOptions {
                include_hidden: true,
            }),
        )
        .expect("list with hidden");
        assert!(names(&all).contains(&".hidden.wav"), "开启后应列出隐藏文件");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entry_name_validation_rejects_paths_and_reserved_names() {
        assert_eq!(
            validate_entry_name("  take 01.wav  ").unwrap(),
            "take 01.wav"
        );
        // 首尾空格是**修掉**而不是拒绝：用户从别处粘贴过来的名字常带尾随空格，
        // 而 Windows 本来就会把尾随空格吞掉 —— 拒绝它只会让人困惑。
        assert_eq!(validate_entry_name("trailing ").unwrap(), "trailing");
        for bad in [
            "",
            "   ",
            ".",
            "..",
            "a/b.wav",
            "a\\b.wav",
            "a:b.wav",
            "a*b.wav",
            "a?b.wav",
            "a|b.wav",
            "trailing.",
            "CON",
            "con.txt",
            "LPT9",
        ] {
            assert!(validate_entry_name(bad).is_err(), "应拒绝非法名: {bad:?}");
        }
    }

    #[test]
    fn create_and_rename_and_delete_round_trip() {
        let dir = fixture();
        let root = dir.to_string_lossy().into_owned();

        // 新建：成功，且拒绝重名。
        let created = create_directory(root.clone(), "新目录".to_string(), None).expect("create");
        assert!(Path::new(&created).is_dir());
        assert_eq!(
            create_directory(root.clone(), "新目录".to_string(), None),
            Err("name_exists".to_string())
        );

        // 重命名：文件换名后旧路径消失、新路径存在。
        let source = dir.join("readme.txt").to_string_lossy().into_owned();
        let renamed = rename_path(source.clone(), "README.md".to_string(), None).expect("rename");
        assert!(Path::new(&renamed).is_file());
        assert!(!Path::new(&source).exists());

        // 重命名到已存在的名字必须被拒（否则会静默覆盖）。
        let other = dir.join("副歌.wav").to_string_lossy().into_owned();
        assert_eq!(
            rename_path(other, "README.md".to_string(), None),
            Err("name_exists".to_string())
        );

        // 删除（永久，避免测试往用户回收站里塞东西）。
        let result = delete_paths(vec![renamed.clone()], true, None);
        assert_eq!(result["ok"], serde_json::json!(true));
        assert_eq!(result["deleted"], serde_json::json!(1));
        assert!(!Path::new(&renamed).exists());

        // 不存在的路径计入失败，但其余项仍然照常删除。
        let result = delete_paths(
            vec![
                dir.join("副歌.wav").to_string_lossy().into_owned(),
                dir.join("does-not-exist.wav")
                    .to_string_lossy()
                    .into_owned(),
            ],
            true,
            None,
        );
        assert_eq!(result["ok"], serde_json::json!(false));
        assert_eq!(result["deleted"], serde_json::json!(1));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_operations_refuse_the_computer_sentinel() {
        assert!(
            create_directory(COMPUTER_VIRTUAL_PATH.to_string(), "x".to_string(), None).is_err()
        );
        assert!(rename_path(COMPUTER_VIRTUAL_PATH.to_string(), "x".to_string(), None).is_err());
        // 哨兵不是真实路径，删除时直接跳过、不报成功也不报失败。
        let result = delete_paths(vec![COMPUTER_VIRTUAL_PATH.to_string()], true, None);
        assert_eq!(result["ok"], serde_json::json!(true));
        assert_eq!(result["deleted"], serde_json::json!(0));
    }
}
