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
    listDirectory: (dirPath: string) => invoke<FileEntry[]>("list_directory", dirPath),

    searchFilesRecursive: (dirPath: string, query: string, options?: SearchOptionsPayload) =>
        invoke<FileEntry[]>("search_files_recursive", dirPath, query, options),

    getAudioFileInfo: (filePath: string) => invoke<AudioFileInfo>("get_audio_file_info", filePath),

    getMediaAudioStreams: (filePath: string) =>
        invoke<MediaAudioStream[]>("get_media_audio_streams", filePath),

    readAudioPreview: (filePath: string, maxFrames?: number) =>
        invoke<AudioPreviewData>("read_audio_preview", filePath, maxFrames),

    pickDirectory: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("pick_directory"),
};
