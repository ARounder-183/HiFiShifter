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
