use crate::search::{
    match_translit, translit, MatchInfo, MatchOptions, Query, SearchMode, SearchOptions,
    TranslitOptions, MAX_DIR_RESULTS,
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

/// 一个路径的存在性与类型（`stat_paths` 的产出）。
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PathStat {
    pub path: String,
    pub exists: bool,
    pub is_dir: bool,
}

/// `collect_folder_media` 的选项。
#[derive(serde::Deserialize, Default, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CollectFolderMediaOptions {
    /// 是否递归下钻子目录。默认否（与 REAPER 的默认一致）。
    #[serde(default)]
    pub recursive: bool,
    /// 是否包含隐藏项。默认否（与 `list_directory` 同口径）。
    #[serde(default)]
    pub include_hidden: bool,
    /// 媒体文件总数上限；缺省取 [`DEFAULT_MAX_COLLECT_FILES`]。
    #[serde(default)]
    pub max_files: Option<usize>,
}

/// 单次目录导入的媒体文件数上限。
///
/// 【为什么是两万】与用户实测的"一个目录两万多个文件"这一已知量级对齐：它既能让
/// 正常的大目录一次导入完，又能挡住"误拖 `C:\Users`"这类把应用拖死的情形。
/// 触顶时**不静默截断**，而是回传 `truncated` 让前端确认。
pub(crate) const DEFAULT_MAX_COLLECT_FILES: usize = 20_000;

/// 一个目录（及其直属媒体文件）在导入时的分组。
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct FolderMediaGroup {
    /// 该组的源目录绝对路径。
    pub dir: String,
    /// 建议的轨道名：顶层目录 = 目录名；子目录 = 相对路径（`Takes/Sub`）。
    pub label: String,
    /// 该目录**直属**的媒体文件绝对路径。
    pub paths: Vec<String>,
    /// 该目录是否含有子目录。
    ///
    /// 【为什么在非递归时也要给】前端用它决定"递归导入"这个选项要不要展示 ——
    /// 展示与否不能取决于"这次有没有真的递归"。
    pub has_subdirs: bool,
}

/// 一次目录扫描的结果。
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct FolderMediaScan {
    pub groups: Vec<FolderMediaGroup>,
    pub total_files: usize,
    /// 是否因上限提前收手。为真时前端必须让用户确认，而不是当作完整结果导入。
    pub truncated: bool,
    /// 被拒绝的顶层路径及原因（`drive_root` / `not_found` / `not_a_directory` /
    /// `virtual_path`）。
    pub rejected: Vec<RejectedPath>,
}

/// 一条被拒绝的顶层路径。
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RejectedPath {
    pub path: String,
    pub reason: String,
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
        // `name` 去掉尾随反斜杠得到 `C:`，行上直接显示为「C:」（与资源管理器的
        // 「此电脑」一致）；`path` 必须保留反斜杠 —— `Path::is_dir()` 对裸 `C:`
        // 的解释依赖进程当前目录，去掉就解析不到盘根。
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

/// 在指定目录下递归搜索文件（可选包含目录）。
///
/// 匹配文件名 stem（忽略扩展名）；`options` 缺省或 `mode = off` 时与转写功能上线前
/// 的行为一致（小写子串、按名称排序）。
///
/// 目录结果与文件结果**各占一条独立通道**：目录不参与文件的超采样与 `max_results`
/// 截断（见 [`MAX_DIR_RESULTS`]），否则一个含数百个子目录的树会把文件结果整个挤掉，
/// 而用户搜的是文件。
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
    let mut collector = SearchCollector {
        matcher: &matcher,
        files: Vec::new(),
        dirs: Vec::new(),
        file_stop_at: max.saturating_mul(SEARCH_OVERSCAN),
        dir_stop_at: MAX_DIR_RESULTS,
        // 遍历预算 = 文件停止阈值的 4 倍（默认 500 上限 → 8000 条目）。
        //
        // 【为什么必须有】文件通道收满后，若目录通道还没收满，遍历会继续走 ——
        // 这是为了让"搜一个目录名"不被海量文件命中提前打断。但若不设上限，
        // 在一个巨大目录树里搜一个几乎不存在的目录名就会退化成**全树遍历**，
        // 正是本项目已经修过一次的那类卡顿。预算与结果规模成比例，既够用又有界。
        visit_budget: max
            .saturating_mul(SEARCH_OVERSCAN)
            .saturating_mul(SEARCH_OVERSCAN),
        visits: 0,
        include_dirs: options.include_dirs(),
        include_hidden: options.include_hidden(),
    };
    let mut visited = std::collections::HashSet::new();
    collect_matching_files(path, &mut collector, &mut visited, 0);

    collector.files.sort_by(|a, b| b.cmp(a));
    collector.files.truncate(max);
    collector.dirs.sort_by(|a, b| b.cmp(a));
    collector.dirs.truncate(MAX_DIR_RESULTS);

    // 目录排在文件之前：与文件浏览器 `foldersFirst` 的默认一致，也让"跳到某个
    // 子目录"这类动作落在结果顶部。
    let mut out: Vec<FileEntry> = Vec::with_capacity(collector.dirs.len() + collector.files.len());
    out.extend(collector.dirs.into_iter().map(|scored| scored.entry));
    out.extend(collector.files.into_iter().map(|scored| scored.entry));
    Ok(out)
}

/// 递归深度上限：防止在极深目录树上无界遍历。
const MAX_SEARCH_DEPTH: usize = 32;

/// 搜索遍历的收集器：文件与目录各一条通道。
///
/// 【为什么两条通道共用一个结构而不是走两趟】截断阈值不同（文件走超采样、目录走
/// [`MAX_DIR_RESULTS`]），但**遍历必须是同一趟** —— 拆成两次会让 junction 环防护
/// 各做一份 `visited`，同一棵物理目录被走两遍，`visited` 也就白设了。
struct SearchCollector<'a> {
    matcher: &'a NameMatcher,
    files: Vec<Scored>,
    dirs: Vec<Scored>,
    file_stop_at: usize,
    dir_stop_at: usize,
    /// 已检视的目录条目总数上限（见 `search_files_recursive` 的注释）。
    visit_budget: usize,
    visits: usize,
    include_dirs: bool,
    include_hidden: bool,
}

impl SearchCollector<'_> {
    /// 该停了：要么预算用尽，要么两条通道都收满。
    ///
    /// 【注意这是"上界"而不是"下界"】`include_dirs` 为假时（快速搜索等既有调用方），
    /// 文件通道一满即停，预算不会让遍历多走一步。
    fn saturated(&self) -> bool {
        if self.visits >= self.visit_budget {
            return true;
        }
        self.files.len() >= self.file_stop_at
            && (!self.include_dirs || self.dirs.len() >= self.dir_stop_at)
    }
}

fn collect_matching_files(
    dir: &Path,
    collector: &mut SearchCollector<'_>,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
    depth: usize,
) {
    if collector.saturated() || depth >= MAX_SEARCH_DEPTH {
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
    // 复制出引用（`&NameMatcher` 是 Copy），后面的 `collector.dirs.push` 才能拿到
    // `&mut collector` 而不与之冲突。
    let matcher = collector.matcher;
    for entry in read_dir.flatten() {
        if collector.saturated() {
            break;
        }
        collector.visits += 1;
        // file_type() 不跟随符号链接/junction，避免把链接目录当作真实目录深入。
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        // 元数据只取一次：隐藏判定（Windows 要看文件属性）、大小、修改时间都要它。
        let metadata = entry.metadata().ok();
        if !collector.include_hidden {
            let hidden = match metadata.as_ref() {
                Some(meta) => is_hidden_entry(meta, &name),
                // 取不到元数据时退回点开头规则（非 Windows 的 is_hidden_entry 即如此）。
                None => name.starts_with('.'),
            };
            if hidden {
                continue;
            }
        }
        let modified_time = metadata.as_ref().and_then(|m| {
            m.modified().ok().and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| d.as_secs_f64())
            })
        });
        let path = entry.path();
        if file_type.is_dir() {
            // 目录命中匹配**全名**而不是 stem：目录没有"扩展名"这个概念，
            // `项目.备份` 里的 `.备份` 是名字的一部分。文件侧继续用 stem 是为了让
            // `vocal` 命中 `vocal.wav`。这条不一致是有意的。
            if collector.include_dirs && collector.dirs.len() < collector.dir_stop_at {
                if let Some((score, match_info)) = matcher.score(&name) {
                    collector.dirs.push(Scored {
                        score,
                        name_lower: name.to_lowercase(),
                        entry: FileEntry {
                            name,
                            path: path.to_string_lossy().into_owned(),
                            is_dir: true,
                            size: None,
                            extension: None,
                            modified_time,
                            match_info,
                        },
                    });
                }
            }
            collect_matching_files(&path, collector, visited, depth + 1);
        } else {
            // 匹配文件名的 stem（不包含后缀）→ 忽略扩展名，与旧实现一致。
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
            collector.files.push(Scored {
                score,
                name_lower: name.to_lowercase(),
                entry: FileEntry {
                    name,
                    path: path.to_string_lossy().into_owned(),
                    is_dir: false,
                    size: metadata.as_ref().map(|m| m.len()),
                    extension,
                    modified_time,
                    match_info,
                },
            });
        }
    }
}

// ===================== 目录导入：枚举与探测 =====================

/// 批量查询路径的存在性与类型。
///
/// 【为什么需要它】拖放时前端只拿到**路径字符串**：Tauri 的原生拖放事件不带类型，
/// HTML5 的 `dataTransfer` 更不带。判断"用户拖进来的是不是文件夹"必须问文件系统，
/// 而逐个路径各发一次 IPC 在拖入 20 项时就是 20 次往返 —— 所以这里一次问完。
///
/// 【为什么不跟随符号链接】与 `list_directory` 同口径（它用
/// `DirEntry::metadata()`，对符号链接不跟随）。同一个路径在列表里和拖放判定里
/// 必须是同一个答案，否则会出现"列表里看着是目录、拖进去却按文件处理"。
pub(crate) fn stat_paths(paths: Vec<String>) -> Vec<PathStat> {
    paths
        .into_iter()
        .map(|raw| {
            let trimmed = raw.trim().to_string();
            match std::fs::symlink_metadata(&trimmed) {
                Ok(metadata) => PathStat {
                    path: trimmed,
                    exists: true,
                    is_dir: metadata.is_dir(),
                },
                Err(_) => PathStat {
                    path: trimmed,
                    exists: false,
                    is_dir: false,
                },
            }
        })
        .collect()
}

/// 目录导入可枚举的媒体扩展名（音频 + 视频容器，视频按音轨导入）。
///
/// 【为什么这里必须有一份】`import_audio_item` 的可解码性靠**内容嗅探**
/// （`try_read_audio_header_only`），但枚举一个目录时不能逐个解码 —— 两万个文件会
/// 把"导入前的等待"变成卡死。只能按扩展名先筛。
///
/// 【为什么必须与前端一致】筛出来的路径随后交给 `importAudioItem`。这份名单比前端
/// 宽，用户就会"导入一个不认识的格式然后失败"；比前端窄，文件浏览器里能拖的文件、
/// 拖文件夹时却被漏掉。`media_extensions_match_frontend` 测试直接读 `fileKinds.ts`
/// 的源码比对，漂移会当场失败。
///
/// MIDI 不在其中：它走独立的导入对话框（`IMPORT_MIDI_PATH_EVENT`），塞进同一条
/// 批量管线会引入半配置状态。见设计文档 §8.6。
pub(crate) const MEDIA_EXTENSIONS: &[&str] = &[
    // 音频
    "wav", "mp3", "flac", "ogg", "oga", "opus", "aac", "m4a", "aif", "aiff", "wma", "ac3", "eac3",
    "ape", "wv", "mp2", "mpa", "dts", "amr", // 视频容器（按音轨导入）
    "mp4", "m4v", "mov", "mkv", "webm", "avi", "flv", "wmv", "ts", "mts", "m2ts", "vob", "mpg",
    "mpeg", "3gp", "3g2", "ogv", "rm", "rmvb",
];

fn is_media_extension(extension: &str) -> bool {
    let lower = extension.to_ascii_lowercase();
    MEDIA_EXTENSIONS.contains(&lower.as_str())
}

/// 单次扫描检视的目录条目总数上限（防病态目录树，与文件数上限是两道不同的闸）。
const MAX_COLLECT_ENTRIES: usize = 200_000;
/// 递归深度上限。与搜索遍历同一个数量级：轨道树再深也没有意义。
const MAX_COLLECT_DEPTH: usize = 32;

/// 顶层路径为什么不能被扫描。
///
/// 【为什么必须挡盘符根】`list_logical_drives()` 返回的盘符条目 `is_dir` 为真
/// （`file_browser.rs` 的 `list_logical_drives`），于是"目录可拖"会让 `C:\` 变成
/// 可拖对象 —— 递归开启时等于全盘扫描。
///
/// 【为什么用"父目录为空"而不是匹配盘符形状】`C:` 只是恰好像盘符的路径；UNC
/// （`\\server\share`）与其它平台的根写法会让形状匹配漏判。结构化判据更稳：
/// 任何根都没有父目录。
fn reject_reason(path: &Path) -> Option<&'static str> {
    if path.to_string_lossy() == COMPUTER_VIRTUAL_PATH {
        return Some("virtual_path");
    }
    if !path.exists() {
        return Some("not_found");
    }
    if !path.is_dir() {
        return Some("not_a_directory");
    }
    if path.parent().is_none() {
        return Some("drive_root");
    }
    None
}

/// 目录扫描的累积状态。
struct FolderScanState {
    groups: Vec<FolderMediaGroup>,
    total_files: usize,
    truncated: bool,
    max_files: usize,
    include_hidden: bool,
    recursive: bool,
    entries_visited: usize,
    visited: std::collections::HashSet<std::path::PathBuf>,
}

impl FolderScanState {
    /// 是否已经收满（文件数或条目预算任一用尽）。
    fn full(&self) -> bool {
        self.total_files >= self.max_files || self.entries_visited >= MAX_COLLECT_ENTRIES
    }
}

/// 递归枚举一个目录，按目录分组产出媒体文件。
///
/// 【为什么不排序文件】排序用前端的 `compareFileNames`（手写的资源管理器序）。
/// 在后端再写一份必然与前端漂移，而"同一份目录在两个界面给出两种顺序"正是本项目
/// 已经踩过的坑。这里只保证**遍历确定**（子目录按小写名排序），组内文件的顺序
/// 交给前端。
fn collect_folder_group(dir: &Path, label: String, depth: usize, state: &mut FolderScanState) {
    if state.full() {
        state.truncated = true;
        return;
    }
    if depth >= MAX_COLLECT_DEPTH {
        state.truncated = true;
        return;
    }
    // 环防护：junction / 符号链接目录环会让无防护递归栈溢出崩溃。
    let Ok(key) = dir.canonicalize() else {
        return;
    };
    if !state.visited.insert(key) {
        return;
    }
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };

    let mut paths: Vec<String> = Vec::new();
    let mut subdirs: Vec<(String, std::path::PathBuf)> = Vec::new();
    for entry in read_dir.flatten() {
        if state.full() {
            state.truncated = true;
            break;
        }
        state.entries_visited += 1;
        // 不跟随符号链接：与 `list_directory` 同口径。
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if !state.include_hidden {
            let metadata = entry.metadata().ok();
            let hidden = match metadata.as_ref() {
                Some(meta) => is_hidden_entry(meta, &name),
                None => name.starts_with('.'),
            };
            if hidden {
                continue;
            }
        }
        if file_type.is_dir() {
            subdirs.push((name, entry.path()));
        } else if let Some(extension) = entry.path().extension().and_then(|e| e.to_str()) {
            if is_media_extension(extension) {
                paths.push(entry.path().to_string_lossy().into_owned());
            }
        }
    }

    // 文件数上限要在**这里**收口，而不是等所有组收完 —— 否则截断的粒度会是"整组"，
    // 一个含 3 万文件的目录会先把 total 顶穿再被丢弃。
    let remaining = state.max_files.saturating_sub(state.total_files);
    if paths.len() > remaining {
        paths.truncate(remaining);
        state.truncated = true;
    }
    state.total_files += paths.len();

    // 子目录按小写名排序：`read_dir` 的顺序由文件系统决定，不排序会让"哪些组先
    // 建轨道"随机器而变，导入结果不可复现。
    subdirs.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));

    let has_subdirs = !subdirs.is_empty();
    state.groups.push(FolderMediaGroup {
        dir: dir.to_string_lossy().into_owned(),
        label: label.clone(),
        paths,
        has_subdirs,
    });

    if !state.recursive {
        return;
    }
    for (name, path) in subdirs {
        // 子目录的 label 是**相对路径**（`Takes/Sub`）：前端按它建轨道树，
        // 名称里也就自带"这个目录从哪来"的全部信息。
        collect_folder_group(&path, format!("{label}/{name}"), depth + 1, state);
    }
}

/// 把一个或一批目录展开成"按目录分组的媒体文件清单"。
///
/// 【为什么它不是 `search_files_recursive` 的一个选项】后者是"找东西"：带相关性
/// 排序与 `max_results` 截断 —— 截断在搜索里是对的（用户看前 N 条），在导入里是
/// **数据丢失**（用户以为全进来了）。本命令只做枚举与分组，不排序、不按相关度截断，
/// 唯一的截断是总量上限且必须回传 `truncated` 让前端确认。
pub(crate) fn collect_folder_media(
    dirs: Vec<String>,
    options: Option<CollectFolderMediaOptions>,
) -> FolderMediaScan {
    let options = options.unwrap_or_default();
    let max_files = options
        .max_files
        .unwrap_or(DEFAULT_MAX_COLLECT_FILES)
        .clamp(1, MAX_COLLECT_ENTRIES);
    let mut state = FolderScanState {
        groups: Vec::new(),
        total_files: 0,
        truncated: false,
        max_files,
        include_hidden: options.include_hidden,
        recursive: options.recursive,
        entries_visited: 0,
        visited: std::collections::HashSet::new(),
    };
    let mut rejected: Vec<RejectedPath> = Vec::new();

    for raw in dirs {
        let trimmed = raw.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        let path = std::path::PathBuf::from(&trimmed);
        if let Some(reason) = reject_reason(&path) {
            rejected.push(RejectedPath {
                path: trimmed,
                reason: reason.to_string(),
            });
            continue;
        }
        // 顶层组的 label 就是目录名（根轨道名）；子目录用相对路径，前端据此建树。
        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| trimmed.clone());
        // 每个被拖入的目录各自一份 visited：同一个物理目录被两个不同的拖入项覆盖时
        // 不该互相吞掉（用户明确拖了两个，就该看到两组）。
        state.visited.clear();
        collect_folder_group(&path, label, 0, &mut state);
    }

    FolderMediaScan {
        groups: state.groups,
        total_files: state.total_files,
        truncated: state.truncated,
        rejected,
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
        // 两个名字含 "vocal" 的子目录：用来验证"目录结果不挤占文件结果名额"。
        std::fs::create_dir_all(dir.join("vocal_takes")).expect("temp dir");
        std::fs::create_dir_all(dir.join("vocal_alt")).expect("temp dir");
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
    fn dirs_are_returned_only_when_asked() {
        let dir = fixture();
        // 缺省（含所有既有调用方）：结果里没有目录 —— 既有行为不得回退。
        assert!(!names(&search(&dir, "vocal", None)).contains(&"vocal_takes"));

        let with_dirs = search(
            &dir,
            "vocal",
            Some(SearchOptions {
                include_dirs: Some(true),
                ..SearchOptions::default()
            }),
        );
        assert!(
            names(&with_dirs).contains(&"vocal_takes"),
            "开启后应出现目录: {:?}",
            names(&with_dirs)
        );
        assert!(names(&with_dirs).contains(&"vocal_alt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_results_carry_dir_shaped_metadata() {
        let dir = fixture();
        let hits = search(
            &dir,
            "vocal",
            Some(SearchOptions {
                include_dirs: Some(true),
                ..SearchOptions::default()
            }),
        );
        let folder = hits
            .iter()
            .find(|entry| entry.name == "vocal_takes")
            .expect("目录命中");
        assert!(folder.is_dir);
        // 与 list_directory 对目录的产出逐字段一致：没有大小、没有扩展名。
        assert!(folder.size.is_none());
        assert!(folder.extension.is_none());
        assert!(Path::new(&folder.path).is_dir());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_results_do_not_consume_the_file_budget() {
        let dir = fixture();
        let hits = search(
            &dir,
            "vocal",
            Some(SearchOptions {
                max_results: Some(2),
                include_dirs: Some(true),
                ..SearchOptions::default()
            }),
        );
        // 文件仍拿满 2 个名额；目录另占一条通道（上限 MAX_DIR_RESULTS）。
        assert_eq!(hits.iter().filter(|entry| !entry.is_dir).count(), 2);
        assert_eq!(hits.iter().filter(|entry| entry.is_dir).count(), 2);
        // 目录排在文件之前（与 foldersFirst 的默认一致）。
        assert!(hits[0].is_dir, "目录应排在结果前面: {:?}", names(&hits));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn include_hidden_aligns_search_with_directory_listing() {
        let dir = fixture();
        // 默认与 list_directory 同口径：点开头的隐藏项不出现在结果里。
        assert!(!names(&search(&dir, "hidden", None)).contains(&".hidden.wav"));
        let all = search(
            &dir,
            "hidden",
            Some(SearchOptions {
                include_hidden: Some(true),
                ..SearchOptions::default()
            }),
        );
        assert!(
            names(&all).contains(&".hidden.wav"),
            "开启后应能搜到隐藏项: {:?}",
            names(&all)
        );
        let _ = std::fs::remove_dir_all(&dir);
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

    // ── 目录导入：枚举 ──────────────────────────────────────────────────

    /// 目录枚举用的临时树：
    /// ```text
    /// root/
    ///   a.wav  b.mp3  notes.txt  .hidden.wav
    ///   Sub/  c.flac  Deep/  d.wav
    ///   Empty/
    /// ```
    fn folder_fixture() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hifishifter_collect_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(dir.join("Sub").join("Deep")).expect("temp dir");
        std::fs::create_dir_all(dir.join("Empty")).expect("temp dir");
        for name in ["a.wav", "b.mp3", "notes.txt", ".hidden.wav"] {
            std::fs::write(dir.join(name), b"x").expect("write");
        }
        std::fs::write(dir.join("Sub").join("c.flac"), b"x").expect("write");
        std::fs::write(dir.join("Sub").join("Deep").join("d.wav"), b"x").expect("write");
        dir
    }

    fn scan(dirs: Vec<String>, options: CollectFolderMediaOptions) -> FolderMediaScan {
        collect_folder_media(dirs, Some(options))
    }

    fn labels(scan: &FolderMediaScan) -> Vec<&str> {
        scan.groups
            .iter()
            .map(|group| group.label.as_str())
            .collect()
    }

    #[test]
    fn non_recursive_scan_takes_only_direct_media() {
        let dir = folder_fixture();
        let result = scan(
            vec![dir.to_string_lossy().into_owned()],
            CollectFolderMediaOptions::default(),
        );
        assert_eq!(result.groups.len(), 1, "非递归只产出顶层一组");
        let group = &result.groups[0];
        assert_eq!(group.label, dir.file_name().unwrap().to_string_lossy());
        let mut found: Vec<&str> = group
            .paths
            .iter()
            .map(|p| Path::new(p).file_name().unwrap().to_str().unwrap())
            .collect();
        found.sort();
        // 只收媒体文件：`notes.txt` 与点开头的隐藏项都不在内。
        assert_eq!(found, vec!["a.wav", "b.mp3"]);
        assert_eq!(result.total_files, 2);
        // has_subdirs 在**非递归**时也要给出 —— 前端据此决定要不要展示递归选项。
        assert!(group.has_subdirs);
        assert!(!result.truncated);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recursive_scan_nests_labels_by_relative_path() {
        let dir = folder_fixture();
        let root_name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let result = scan(
            vec![dir.to_string_lossy().into_owned()],
            CollectFolderMediaOptions {
                recursive: true,
                ..Default::default()
            },
        );
        let found = labels(&result);
        // 子目录用相对路径作 label（前端据此建轨道树），并保持 DFS 顺序。
        assert!(found.contains(&root_name.as_str()), "顶层组: {found:?}");
        assert!(
            found.contains(&format!("{root_name}/Sub").as_str()),
            "子目录组: {found:?}"
        );
        assert!(
            found.contains(&format!("{root_name}/Sub/Deep").as_str()),
            "更深一层用完整相对路径: {found:?}"
        );
        // 空目录也成组（它自己有 has_subdirs=false，但仍是"这个文件夹存在"的事实）。
        assert!(found.contains(&format!("{root_name}/Empty").as_str()));
        assert_eq!(result.total_files, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_respects_max_files_and_reports_truncation() {
        let dir = folder_fixture();
        let result = scan(
            vec![dir.to_string_lossy().into_owned()],
            CollectFolderMediaOptions {
                recursive: true,
                max_files: Some(1),
                include_hidden: false,
            },
        );
        assert_eq!(result.total_files, 1);
        // 截断必须显式回传：静默截断在导入场景里等于数据丢失。
        assert!(result.truncated);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_includes_hidden_only_when_asked() {
        let dir = folder_fixture();
        let root = dir.to_string_lossy().into_owned();
        let hidden = |recursive: bool| {
            scan(
                vec![root.clone()],
                CollectFolderMediaOptions {
                    recursive,
                    include_hidden: true,
                    max_files: None,
                },
            )
            .total_files
        };
        assert_eq!(hidden(false), 3, "开启后应多出 .hidden.wav");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_rejects_drive_roots_and_non_directories() {
        let dir = folder_fixture();
        let file = dir.join("a.wav").to_string_lossy().into_owned();
        let missing = dir.join("nope").to_string_lossy().into_owned();
        #[cfg(windows)]
        let roots = vec!["C:\\".to_string()];
        #[cfg(not(windows))]
        let roots = vec!["/".to_string()];
        let result = scan(
            vec![
                file.clone(),
                missing.clone(),
                COMPUTER_VIRTUAL_PATH.to_string(),
            ]
            .into_iter()
            .chain(roots)
            .collect(),
            CollectFolderMediaOptions::default(),
        );
        assert!(result.groups.is_empty(), "全部应被拒绝");
        let reasons: Vec<(&str, &str)> = result
            .rejected
            .iter()
            .map(|entry| (entry.path.as_str(), entry.reason.as_str()))
            .collect();
        assert!(reasons.contains(&(file.as_str(), "not_a_directory")));
        assert!(reasons.contains(&(missing.as_str(), "not_found")));
        assert!(reasons.contains(&(COMPUTER_VIRTUAL_PATH, "virtual_path")));
        // 盘符根：递归开启时扫整个盘，必须挡在最前面。
        assert!(
            reasons.iter().any(|(_, reason)| *reason == "drive_root"),
            "盘符根应被拒: {reasons:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stat_paths_reports_type_and_existence() {
        let dir = folder_fixture();
        let dir_path = dir.to_string_lossy().into_owned();
        let file_path = dir.join("a.wav").to_string_lossy().into_owned();
        let missing_path = dir.join("nope").to_string_lossy().into_owned();
        let stats = stat_paths(vec![
            dir_path.clone(),
            file_path.clone(),
            missing_path.clone(),
        ]);
        assert_eq!(stats.len(), 3);
        assert!(stats[0].exists && stats[0].is_dir, "目录");
        assert!(stats[1].exists && !stats[1].is_dir, "文件");
        assert!(!stats[2].exists, "不存在的路径");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 扩展名白名单：三处对齐 ──────────────────────────────────────────

    /// 从 `export const NAME = new Set([ ... ]);` 里取出字符串字面量。
    fn extract_string_set(source: &str, marker: &str) -> Vec<String> {
        let start = source
            .find(marker)
            .unwrap_or_else(|| panic!("找不到 {marker}"));
        let rest = &source[start..];
        let open = rest.find('[').expect("缺少 [");
        let close = rest[open..].find(']').expect("缺少 ]");
        rest[open + 1..open + close]
            .split(',')
            .filter_map(|piece| {
                let piece = piece.trim().strip_prefix('"')?;
                Some(piece[..piece.find('"')?].to_string())
            })
            .collect()
    }

    /// 从 `const NAME =\n    /\.(a|b|c)$/i;` 里取出竖线分隔的扩展名。
    fn extract_regex_alternation(source: &str, marker: &str) -> Vec<String> {
        let start = source
            .find(marker)
            .unwrap_or_else(|| panic!("找不到 {marker}"));
        let rest = &source[start..];
        let open = rest.find(r"/\.(").expect("缺少正则前缀") + 4;
        let close = rest[open..].find(')').expect("缺少正则后缀");
        rest[open..open + close]
            .split('|')
            .map(|piece| piece.trim().to_string())
            .collect()
    }

    fn sorted(mut values: Vec<String>) -> Vec<String> {
        values.sort();
        values.dedup();
        values
    }

    /// Rust 的媒体扩展名表必须与前端**唯一来源**一致。
    ///
    /// 【为什么值得一个跨语言测试】枚举出的路径随后交给前端的导入流程。这份名单比
    /// 前端宽，用户就会"导入一个不认识的格式然后失败"；比前端窄，文件浏览器里能拖的
    /// 文件、拖文件夹时却被漏掉。直接读 TS 源码比对是唯一能挡住漂移的办法。
    #[test]
    fn media_extensions_match_frontend() {
        let source = include_str!("../../../../frontend/src/features/fileBrowser/fileKinds.ts");
        let expected = sorted(
            extract_string_set(source, "export const AUDIO_EXTENSIONS")
                .into_iter()
                .chain(extract_string_set(source, "export const VIDEO_EXTENSIONS"))
                .collect(),
        );
        let actual = sorted(MEDIA_EXTENSIONS.iter().map(|s| s.to_string()).collect());
        assert_eq!(
            actual, expected,
            "Rust MEDIA_EXTENSIONS 与 fileKinds.ts 的 AUDIO_EXTENSIONS ∪ VIDEO_EXTENSIONS 漂移了"
        );
    }

    /// `fileKinds.ts` 是唯一来源，但 `timeline/dnd.ts` 另抄了一份正则。
    /// 这条断言挡住那份副本悄悄漂移。
    #[test]
    fn frontend_kind_lists_agree_with_drop_admission() {
        let kinds = include_str!("../../../../frontend/src/features/fileBrowser/fileKinds.ts");
        let dnd = include_str!("../../../../frontend/src/components/layout/timeline/dnd.ts");
        let from_kinds = sorted(
            extract_string_set(kinds, "export const AUDIO_EXTENSIONS")
                .into_iter()
                .chain(extract_string_set(kinds, "export const VIDEO_EXTENSIONS"))
                .collect(),
        );
        let from_dnd = sorted(
            extract_regex_alternation(dnd, "const AUDIO_FILE_RE")
                .into_iter()
                .chain(extract_regex_alternation(dnd, "const VIDEO_FILE_RE"))
                .collect(),
        );
        assert_eq!(
            from_kinds, from_dnd,
            "fileKinds.ts 与 timeline/dnd.ts 的媒体扩展名白名单漂移了"
        );
    }
}
