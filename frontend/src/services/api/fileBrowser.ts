import { invoke } from "../invoke";

export interface FileEntry {
    name: string;
    path: string;
    isDir: boolean;
    size: number | null;
    extension: string | null;
    modifiedTime: number | null;
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

    searchFilesRecursive: (dirPath: string, query: string) =>
        invoke<FileEntry[]>("search_files_recursive", dirPath, query),

    getAudioFileInfo: (filePath: string) => invoke<AudioFileInfo>("get_audio_file_info", filePath),

    getMediaAudioStreams: (filePath: string) =>
        invoke<MediaAudioStream[]>("get_media_audio_streams", filePath),

    readAudioPreview: (filePath: string, maxFrames?: number) =>
        invoke<AudioPreviewData>("read_audio_preview", filePath, maxFrames),

    pickDirectory: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("pick_directory"),
};
