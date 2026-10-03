import { invoke } from "../invoke";
import type { SearchOptionsPayload } from "../../features/search/searchSettings";

/** 搜索命中说明（与后端 `search::MatchInfo` 对齐）。 */
export interface FileMatchInfo {
    kind: "literal" | "pinyin" | "romaji" | "choseong" | "fuzzy";
    /** 档位分值，用于排序。 */
    score: number;
    /** 命中的形态（`zhuge` / `cx`），用于显示「匹配拼音 zhuge」。 */
    form: string;
}

export interface FileEntry {
    name: string;
    path: string;
    isDir: boolean;
    size: number | null;
    extension: string | null;
    modifiedTime: number | null;
    /** 仅搜索路径产出；目录列表没有这一项。 */
    matchInfo?: FileMatchInfo;
}

export interface AudioFileInfo {
    sampleRate: number;
    channels: number;
    durationSec: number;
    totalFrames: number;
}

/**
 * 一个路径的存在性与类型（`stat_paths` 的产出）。
 *
 * 【为什么需要】拖放时前端只拿到路径字符串 —— Tauri 的原生拖放事件不带类型。
 * 判断"拖进来的是不是目录"必须问文件系统，而逐个路径各发一次 IPC 在拖入二十项
 * 时就是二十次往返，所以后端提供批量版本。
 */
export interface PathStat {
    path: string;
    exists: boolean;
    isDir: boolean;
}

/** 目录导入时一个目录（及其直属媒体文件）的分组。 */
export interface FolderMediaGroup {
    /** 该组的源目录绝对路径。 */
    dir: string;
    /** 建议的轨道名：顶层 = 目录名，子目录 = 相对路径（`Takes/Sub`）。 */
    label: string;
    /** 该目录**直属**的媒体文件绝对路径（顺序未定，由调用方排序）。 */
    paths: string[];
    /** 该目录是否含子目录（决定"递归导入"选项是否展示）。 */
    hasSubdirs: boolean;
}

/** 顶层路径为什么没被扫描。 */
export type FolderScanRejectReason =
    | "drive_root"
    | "not_found"
    | "not_a_directory"
    | "virtual_path";

export interface FolderMediaScan {
    groups: FolderMediaGroup[];
    totalFiles: number;
    /** 是否因上限提前收手 —— 为真时必须让用户确认，不能当作完整结果导入。 */
    truncated: boolean;
    rejected: { path: string; reason: FolderScanRejectReason }[];
}

export interface CollectFolderMediaOptions {
    recursive?: boolean;
    includeHidden?: boolean;
    maxFiles?: number;
}

export interface AudioPreviewData {
    sampleRate: number;
    channels: number;
    pcmBase64: string;
}

export interface MediaAudioStream {
    index: number;
    title: string | null;
    language: string | null;
    codec: string;
    sampleRate: number;
    channels: number;
    durationSec: number;
}

export const fileBrowserApi = {
    /**
     * 列出目录内容。
     *
     * @param options `includeHidden` 为真时列出隐藏项（Windows 含
     *   `FILE_ATTRIBUTE_HIDDEN`；带 `FILE_ATTRIBUTE_SYSTEM` 的始终隐藏）。
     */
    listDirectory: (dirPath: string, options?: { includeHidden: boolean }) =>
        invoke<FileEntry[]>("list_directory", dirPath, options),

    /** 在 `parentDir` 下新建目录，返回新目录的绝对路径。 */
    createDirectory: (parentDir: string, name: string) =>
        invoke<string>("create_directory", parentDir, name),

    /** 把 `path` 重命名为同目录下的 `newName`，返回新路径。 */
    renamePath: (path: string, newName: string) => invoke<string>("rename_path", path, newName),

    /** 把一批路径移入回收站（`permanent` 为真时永久删除）。 */
    deletePaths: (paths: string[], permanent = false) =>
        invoke<{ ok: boolean; deleted?: number; error?: string }>("delete_paths", paths, permanent),

    searchFilesRecursive: (dirPath: string, query: string, options?: SearchOptionsPayload) =>
        invoke<FileEntry[]>("search_files_recursive", dirPath, query, options),

    /** 批量查询路径的存在性与类型（拖放时判断"拖进来的是不是目录"）。 */
    statPaths: (paths: string[]) => invoke<PathStat[]>("stat_paths", paths),

    /**
     * 把一个或一批目录展开成"按目录分组的媒体文件清单"。
     *
     * 【与 `searchFilesRecursive` 的分工】后者是"找东西"，带相关性排序与结果截断；
     * 本接口是**导入枚举**，不做相关性截断（截断在导入里等于数据丢失），唯一的截断
     * 是总量上限，且会通过 `truncated` 显式回传。
     */
    collectFolderMedia: (dirs: string[], options?: CollectFolderMediaOptions) =>
        invoke<FolderMediaScan>("collect_folder_media", dirs, options),

    getAudioFileInfo: (filePath: string) => invoke<AudioFileInfo>("get_audio_file_info", filePath),

    getMediaAudioStreams: (filePath: string) =>
        invoke<MediaAudioStream[]>("get_media_audio_streams", filePath),

    readAudioPreview: (filePath: string, maxFrames?: number) =>
        invoke<AudioPreviewData>("read_audio_preview", filePath, maxFrames),

    pickDirectory: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("pick_directory"),

    /**
     * 在系统文件管理器中定位一批路径：存在的文件被多选高亮，全是目录时打开第一个。
     *
     * 【为什么走 Rust 命令而不是前端 `@tauri-apps/plugin-opener`】`opener:default`
     * 实际只授予 `open-url` / `reveal-item-in-dir` / `default-urls`，**不含
     * `open_path`**；且 `open_path` 还要过 `Scope::is_path_allowed`，没有 scope 配置
     * 时任何绝对路径都会被拒。Rust 侧的 `OpenerExt` 不过 ACL —— 本仓既有做法。
     */
    revealPaths: (paths: string[]) =>
        invoke<{ ok: boolean; count?: number; path?: string; error?: string }>(
            "reveal_paths_in_file_manager",
            paths,
        ),

    /** 用系统默认程序打开一个路径（文件或目录）。 */
    openPathWithDefaultApp: (path: string) =>
        invoke<{ ok: boolean; error?: string }>("open_path_with_default_app", path),
};
