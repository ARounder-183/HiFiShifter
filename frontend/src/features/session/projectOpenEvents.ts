// 外部文件动作事件定义。
//
// 用于在组件间传递外部文件触发的动作请求（打开工程/导入工程/导入音频/导入 MIDI）。

import { detectExternalPathAction } from "../../components/layout/timeline/dnd";
import { emitWindowEvent } from "../dock/windowEventBridge";

export const OPEN_PROJECT_PATH_EVENT = "hifi:open-project-path";

/** 「打开导入工程对话框」请求（`App.tsx` 监听并弹出 `ImportProjectDialog`）。 */
export const IMPORT_PROJECT_PICK_EVENT = "hifi:importProjectPick";

/**
 * 「导入 MIDI」请求。
 *
 * 【为什么是事件而不是 Redux 字段】MIDI 导入对话框的全部选项状态（十余项）由
 * `App.tsx` 持有 —— 只有它能在任何面板布局下渲染那个对话框。文件浏览器发一条
 * 请求、`App.tsx` 接住并用自己的默认值打开对话框，与 `OPEN_PROJECT_PATH_EVENT`
 * 同一条通道；比让文件浏览器持有一份对话框状态副本更不容易分叉。
 */
export const IMPORT_MIDI_PATH_EVENT = "hifi:import-midi-path";

export type ExternalFileActionKind =
    "openProject" | "importVocalShifter" | "importReaper" | "importAudio";

export type ExternalFileActionDetail = {
    kind: ExternalFileActionKind;
    path: string;
};

export type ImportMidiRequestDetail = {
    path: string;
    /** 落点（秒）。省略表示用对话框自己的默认值。 */
    startSec?: number;
    /** 目标轨道 id；`null` 表示新建轨道。省略表示用当前选中轨道。 */
    trackId?: string | null;
};

export function emitOpenProjectPath(path: string) {
    emitExternalFileAction("openProject", path);
}

export function emitExternalFileAction(kind: ExternalFileActionKind, path: string) {
    const normalized = String(path ?? "").trim();
    if (!normalized) return;
    emitWindowEvent("main", OPEN_PROJECT_PATH_EVENT, {
        kind,
        path: normalized,
    } satisfies ExternalFileActionDetail);
}

/** 请求打开「导入工程」对话框（携带路径）。 */
export function emitImportProjectPick(path: string) {
    const normalized = String(path ?? "").trim();
    if (!normalized) return;
    emitWindowEvent("main", IMPORT_PROJECT_PICK_EVENT, { path: normalized });
}

/** 请求打开「导入 MIDI」对话框（携带路径与可选落点 / 目标轨道）。 */
export function emitImportMidiRequest(detail: ImportMidiRequestDetail) {
    const normalized = String(detail.path ?? "").trim();
    if (!normalized) return;
    emitWindowEvent("main", IMPORT_MIDI_PATH_EVENT, {
        ...detail,
        path: normalized,
    } satisfies ImportMidiRequestDetail);
}

/**
 * 判定路径对应的外部文件动作种类。
 *
 * 与拖放共用 detectExternalPathAction：准入判据只有一份，MIDI / 文件夹路径
 * 在这一通道没有对应动作，显式排除（它们有自己的对话框/导入流程）。
 */
export function detectExternalActionKindFromPath(path: string): ExternalFileActionKind | null {
    const kind = detectExternalPathAction(path);
    if (kind === null || kind === "importMidi" || kind === "importFolder") return null;
    return kind;
}
