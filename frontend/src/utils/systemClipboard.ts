/*
 * 系统剪贴板对象读写工具。
 *
 * 实际传输由 Rust 后端负责：后端写入平台原生自定义格式，并附带
 * base64 文本信封作为回退。这样两个 HiFiShifter 进程之间复制粘贴
 * 不再依赖 WebView 的剪贴板权限，在 Windows / macOS / Linux 上均可工作。
 */

import type { ClipTemplate } from "../features/session/sessionTypes";
import type { ParamName } from "../components/layout/pianoRoll/types";
import { invoke } from "../services/invoke";

type ClipboardKind = "clip" | "param";

export interface ClipClipboardObject {
    version: 1;
    kind: "clip";
    templates: ClipTemplate[];
    groupIds?: string[];
}

export interface ParamClipboardObject {
    version: 1;
    kind: "param";
    param: ParamName;
    framePeriodMs: number;
    values: number[];
}

function parseClipboardObject(raw: string): ClipClipboardObject | ParamClipboardObject | null {
    try {
        const parsed = JSON.parse(raw) as ClipClipboardObject | ParamClipboardObject;
        if (parsed?.version !== 1 || (parsed?.kind !== "clip" && parsed?.kind !== "param")) {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function clipboardSummary(payload: ClipClipboardObject | ParamClipboardObject): string {
    if (payload.kind === "param") {
        return `HiFiShifter: ${payload.values.length} parameter frame(s) copied. Paste in HiFiShifter Parameter Editor.`;
    }
    return `HiFiShifter: ${payload.templates.length} clip(s) copied. Paste in HiFiShifter timeline.`;
}

export async function writeSystemClipboardObject(
    payload: ClipClipboardObject | ParamClipboardObject,
): Promise<void> {
    const result = await invoke<{ ok: boolean; error?: string }>(
        "write_system_clipboard_object",
        JSON.stringify(payload),
        clipboardSummary(payload),
    );
    if (!result.ok) {
        throw new Error(result.error ?? "clipboard_write_failed");
    }
}

export async function readSystemClipboardObject(
    kind: ClipboardKind,
): Promise<ClipClipboardObject | ParamClipboardObject | null> {
    const result = await invoke<{
        ok: boolean;
        available?: boolean;
        payload?: string;
        error?: string;
    }>("read_system_clipboard_object");
    if (!result.ok || !result.available || typeof result.payload !== "string") {
        return null;
    }
    const parsed = parseClipboardObject(result.payload);
    if (parsed?.kind === kind) {
        return parsed;
    }
    return null;
}
