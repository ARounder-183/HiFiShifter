//! 「导入文件夹」的目录扫描：把一个或一批目录展开成**按目录分组**的媒体文件清单。
//!
//! # 为什么这一层在内核
//! 它只做文件系统枚举与扩展名筛选，不碰任何宿主 API —— 于是独立 App 与 ARA 插件
//! 能用**同一份**实现。此前它只住在 app 的 `commands/file_browser.rs` 里，插件够不到，
//! 于是插件里"导入文件夹"是死的（扫描命令不存在）。复制一份到插件则是另一条死路：
//! 媒体扩展名名单一旦有两份，两边就会各自漂移（见 [`crate::media`] 与前端
//! `fileKinds.ts` 的既有对齐测试）。
//!
//! # 它不是 `search_files_recursive` 的一个选项
//! 后者是"找东西"：带相关性排序与 `max_results` 截断 —— 截断在搜索里是对的（用户
//! 看前 N 条），在导入里是**数据丢失**（用户以为全进来了）。本模块只做枚举与分组，
//! 不排序、不按相关度截断，唯一的截断是总量上限且必须回传 `truncated` 让前端确认。
use std::path::{Path, PathBuf};

/// 文件浏览器里代表「此电脑」的虚拟路径。它不是真实目录，任何扫描都必须拒绝它。
pub const VIRTUAL_COMPUTER_PATH: &str = "computer://";

/// 单次目录导入的媒体文件数上限。
///
/// 【为什么是两万】与用户实测的"一个目录两万多个文件"这一已知量级对齐：它既能让
/// 正常的大目录一次导入完，又能挡住"误拖 `C:\Users`"这类把应用拖死的情形。
/// 触顶时**不静默截断**，而是回传 `truncated` 让前端确认。
pub const DEFAULT_MAX_COLLECT_FILES: usize = 20_000;

/// 单次扫描检视的目录条目总数上限（防病态目录树，与文件数上限是两道不同的闸）。
pub const MAX_COLLECT_ENTRIES: usize = 200_000;

/// 递归深度上限。与搜索遍历同一个数量级：轨道树再深也没有意义。
pub const MAX_COLLECT_DEPTH: usize = 32;

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
pub fn is_hidden_entry(metadata: &std::fs::Metadata, name: &str) -> bool {
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
pub fn is_hidden_entry(_metadata: &std::fs::Metadata, name: &str) -> bool {
    name.starts_with('.')
}

/// 顶层路径为什么不能被扫描。
///
/// 【为什么必须挡盘符根】文件浏览器把盘符条目也显示成"目录"，于是"目录可拖"会让
/// `C:\` 变成可拖对象 —— 递归开启时等于全盘扫描。
///
/// 【为什么用"父目录为空"而不是匹配盘符形状】`C:` 只是恰好像盘符的路径；UNC
/// （`\\server\share`）与其它平台的根写法会让形状匹配漏判。结构化判据更稳：
/// 任何根都没有父目录。
fn reject_reason(path: &Path) -> Option<&'static str> {
    if path.to_string_lossy() == VIRTUAL_COMPUTER_PATH {
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
    visited: std::collections::HashSet<PathBuf>,
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
/// 在这里再写一份必然与前端漂移，而"同一份目录在两个界面给出两种顺序"正是本项目
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
    let mut subdirs: Vec<(String, PathBuf)> = Vec::new();
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
        } else {
            let path = entry.path();
            if crate::media::is_media_extension(&path) {
                paths.push(path.to_string_lossy().into_owned());
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
    subdirs.sort_by_key(|a| a.0.to_lowercase());

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
/// 调用方负责**路径准入**（App 侧是它自己的文件系统，插件侧必须先过授权根校验）；
/// 本函数只做枚举与分组，不读宿主、不写磁盘。
pub fn collect_folder_media(
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
        let path = PathBuf::from(&trimmed);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hfs-folder-scan-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 顶层媒体进本组、子目录成组、非媒体与隐藏项被筛掉。
    #[test]
    fn scan_groups_by_directory_and_filters_media() {
        let root = temp_dir("basic");
        std::fs::write(root.join("主歌.wav"), b"x").unwrap();
        std::fs::write(root.join("说明.txt"), b"x").unwrap();
        std::fs::write(root.join(".hidden.wav"), b"x").unwrap();
        std::fs::create_dir_all(root.join("Takes")).unwrap();
        std::fs::write(root.join("Takes").join("take2.flac"), b"x").unwrap();

        let flat = collect_folder_media(vec![root.to_string_lossy().into_owned()], None);
        assert_eq!(flat.total_files, 1, "非递归只收本层媒体");
        assert_eq!(flat.groups.len(), 1);
        assert_eq!(flat.groups[0].paths.len(), 1);
        assert!(flat.groups[0].paths[0].ends_with("主歌.wav"));
        assert!(flat.groups[0].has_subdirs, "子目录的存在必须被回报");
        assert!(!flat.truncated);
        assert!(flat.rejected.is_empty());

        let deep = collect_folder_media(
            vec![root.to_string_lossy().into_owned()],
            Some(CollectFolderMediaOptions {
                recursive: true,
                include_hidden: false,
                max_files: None,
            }),
        );
        assert_eq!(deep.total_files, 2);
        // 顶层组在前，子目录组在后；子目录的 label 是**相对路径**（前端据此建轨道树）。
        assert_eq!(deep.groups.len(), 2);
        assert!(deep.groups[1].label.ends_with("/Takes"));
        assert!(deep.groups[1].paths[0].ends_with("take2.flac"));

        let _ = std::fs::remove_dir_all(&root);
    }

    /// 盘符根与不存在的路径被拒绝，且**不**产生空组。
    #[test]
    fn scan_rejects_drive_roots_and_missing_paths() {
        let missing = temp_dir("reject").join("gone");
        let result = collect_folder_media(
            vec![
                r"C:\".to_string(),
                missing.to_string_lossy().into_owned(),
                VIRTUAL_COMPUTER_PATH.to_string(),
                "   ".to_string(),
            ],
            None,
        );
        assert!(result.groups.is_empty());
        assert_eq!(
            result.rejected.len(),
            3,
            "空白项被静默跳过，其余三条都要有理由"
        );
        let reasons = result
            .rejected
            .iter()
            .map(|entry| entry.reason.as_str())
            .collect::<Vec<_>>();
        assert!(reasons.contains(&"drive_root"));
        assert!(reasons.contains(&"not_found"));
        assert!(reasons.contains(&"virtual_path"));
    }

    /// 触顶必须回传 `truncated`，而不是静默给半份结果。
    #[test]
    fn scan_reports_truncation_instead_of_silently_cutting() {
        let root = temp_dir("truncate");
        for index in 0..5 {
            std::fs::write(root.join(format!("take{index}.wav")), b"x").unwrap();
        }
        let scan = collect_folder_media(
            vec![root.to_string_lossy().into_owned()],
            Some(CollectFolderMediaOptions {
                recursive: false,
                include_hidden: false,
                max_files: Some(2),
            }),
        );
        assert_eq!(scan.total_files, 2);
        assert!(scan.truncated);
        let _ = std::fs::remove_dir_all(&root);
    }
}
