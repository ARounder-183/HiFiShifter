/**
 * 文件类型判定 —— 扩展名白名单与"这是什么文件"的唯一来源。
 *
 * 【为什么单独成模块】这些白名单此前只存在于 `FileBrowserPanel.tsx`，而
 * `QuickSearchPopup.tsx` 又抄了一份 `AUDIO_EXTENSIONS`。两份名单一旦漂移，
 * 同一份文件在"文件浏览器里能拖"和"快速搜索里搜不到"之间就会给出两种答案 ——
 * 这正是本项目在 `.hshp-bak` 上踩过的坑（拖放准入与面板白名单不一致）。
 *
 * 现在名单只有一处：面板、快速搜索、右键菜单都从这里取。
 *
 * 【与后端的对齐】`MIDI_EXTENSIONS` 必须与拖放准入（`timeline/dnd.ts`）和后端
 * `SUPPORTED_MIDI_EXTS` 一致；`PROJECT_EXTENSIONS` 必须与拖放准入含同一批
 * `-bak` 备份后缀。新增格式时三处一起改。
 */

import type { FileEntry } from "../../services/api/fileBrowser";

/** 支持的音频与视频媒体扩展名（视频按音轨导入）。 */
export const AUDIO_EXTENSIONS = new Set([
    "wav",
    "mp3",
    "flac",
    "ogg",
    "oga",
    "opus",
    "aac",
    "m4a",
    "aif",
    "aiff",
    "wma",
    "ac3",
    "eac3",
    "ape",
    "wv",
    "mp2",
    "mpa",
    "dts",
    "amr",
    "mp4",
    "m4v",
    "mov",
    "mkv",
    "webm",
    "avi",
    "flv",
    "wmv",
    "ts",
    "mts",
    "m2ts",
    "vob",
    "mpg",
    "mpeg",
    "3gp",
    "3g2",
    "ogv",
    "rm",
    "rmvb",
]);

/** 容器扩展名里哪些是"视频"（用于换图标；它们与音频走同一条导入路径）。 */
export const VIDEO_EXTENSIONS = new Set([
    "mp4",
    "m4v",
    "mov",
    "mkv",
    "webm",
    "avi",
    "flv",
    "wmv",
    "ts",
    "mts",
    "m2ts",
    "vob",
    "mpg",
    "mpeg",
    "3gp",
    "3g2",
    "ogv",
    "rm",
    "rmvb",
]);

/**
 * 支持的 MIDI 文件扩展名（可拖拽导入到时间轴或参数编辑器）。
 *
 * 与拖放准入（`timeline/dnd`）和后端 `SUPPORTED_MIDI_EXTS` 保持一致：`smf`
 * （Standard MIDI File）一并支持，否则它能"被放进来"却"不能从文件浏览器拖出去"。
 */
export const MIDI_EXTENSIONS = new Set(["mid", "midi", "smf"]);

/**
 * 支持的工程文件扩展名（可拖拽导入）。
 *
 * 与拖放准入（`timeline/dnd`）保持**同一份白名单**：含 `-bak` 备份后缀
 * （`.hshp-bak` / `.hsp-bak` / `.rpp-bak`）。此前这里不含备份后缀，于是同一个
 * `.hshp-bak` 文件"拖进时间轴会被接受、从文件浏览器里却拖不动"——同一应用对同一
 * 文件给出两种答案。
 *
 * 注意 FileEntry.extension 是最后一个点后的完整后缀，因此 "proj.hshp-bak" 的
 * extension 为 "hshp-bak"、而不是 "bak"，天然不会与正本混淆。
 */
export const PROJECT_EXTENSIONS = new Set([
    "hshp",
    "hsp",
    "hshp-bak",
    "hsp-bak",
    "rpp",
    "rpp-bak",
    "vshp",
    "vsp",
]);

/** Reaper 工程（`.rpp` / `.rpp-bak`）—— 与 HiFiShifter 工程走不同的导入通道。 */
export const REAPER_EXTENSIONS = new Set(["rpp", "rpp-bak"]);

/** VocalShifter 工程（`.vshp` / `.vsp`）。 */
export const VOCALSHIFTER_EXTENSIONS = new Set(["vshp", "vsp"]);

function hasExtension(entry: FileEntry, allowed: ReadonlySet<string>): boolean {
    return !entry.isDir && !!entry.extension && allowed.has(entry.extension);
}

/** 音频或视频文件（视频按音轨导入，与音频同一条导入路径）。 */
export function isAudioFile(entry: FileEntry): boolean {
    return hasExtension(entry, AUDIO_EXTENSIONS);
}

/** 视频容器（`isAudioFile` 的子集，仅用于图标区分）。 */
export function isVideoFile(entry: FileEntry): boolean {
    return hasExtension(entry, VIDEO_EXTENSIONS);
}

/** MIDI 文件。 */
export function isMidiFile(entry: FileEntry): boolean {
    return hasExtension(entry, MIDI_EXTENSIONS);
}

/** 工程文件（HiFiShifter / Reaper / VocalShifter 工程，含备份后缀）。 */
export function isProjectFile(entry: FileEntry): boolean {
    return hasExtension(entry, PROJECT_EXTENSIONS);
}

/** Reaper 工程文件。 */
export function isReaperFile(entry: FileEntry): boolean {
    return hasExtension(entry, REAPER_EXTENSIONS);
}

/** VocalShifter 工程文件。 */
export function isVocalShifterFile(entry: FileEntry): boolean {
    return hasExtension(entry, VOCALSHIFTER_EXTENSIONS);
}

/**
 * 可拖拽的文件：音频/视频（拖入时间轴）+ MIDI（拖入时间轴或参数编辑器）
 * + 工程文件（拖入时间轴弹出 打开/导入 操作）。
 */
export function isDraggableFile(entry: FileEntry): boolean {
    return isAudioFile(entry) || isMidiFile(entry) || isProjectFile(entry);
}

/**
 * 媒体文件：音频/视频 + MIDI。
 *
 * 【为什么要单独一个判据】文件浏览器的「仅显示媒体文件」与快速搜索的候选列表
 * 必须是**同一件事** —— 两边各写一遍 `isAudioFile || isMidiFile`，迟早会分叉成
 * "文件浏览器里看得到、快速搜索里搜不到"。
 */
export function isMediaFile(entry: FileEntry): boolean {
    return isAudioFile(entry) || isMidiFile(entry);
}

/**
 * 可"插入到时间轴"的文件：音频/视频走 `importAudioAtPosition`，
 * MIDI 走导入对话框，工程文件走"打开/导入工程"。
 */
export function isInsertableFile(entry: FileEntry): boolean {
    return isAudioFile(entry) || isMidiFile(entry) || isProjectFile(entry);
}
