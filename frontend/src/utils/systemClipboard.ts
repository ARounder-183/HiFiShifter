/*
 * 系统剪贴板对象读写工具。
 * 使用自定义 MIME 存储结构化对象，避免把完整业务数据直接暴露到明文文本剪贴板。
 */

import type { ClipTemplate } from "../features/session/sessionTypes";
import type { ParamName } from "../components/layout/pianoRoll/types";

const CLIP_MIME = "application/x-hifishifter-clip+json";
const PARAM_MIME = "application/x-hifishifter-param+json";
const TEXT_PREFIX = "hifishifter_clipboard_v1:";

type ClipboardKind = "clip" | "param";

export interface ClipClipboardObject {
    version: 1;
    kind: "clip";
    templates: ClipTemplate[];
}

export interface ParamClipboardObject {
    version: 1;
    kind: "param";
    param: ParamName;
    framePeriodMs: number;
    values: number[];
}

function hasClipboardReadWrite(): boolean {
    return (
        typeof navigator !== "undefined" &&
        !!navigator.clipboard &&
        typeof navigator.clipboard.read === "function" &&
        typeof navigator.clipboard.write === "function" &&
        typeof ClipboardItem !== "undefined"
    );
}

function hasClipboardTextReadWrite(): boolean {
    return (
        typeof navigator !== "undefined" &&
        !!navigator.clipboard &&
        typeof navigator.clipboard.readText === "function" &&
        typeof navigator.clipboard.writeText === "function"
    );
}

function encodeClipboardEnvelope(payload: ClipClipboardObject | ParamClipboardObject): string {
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    let binary = "";
    for (const b of bytes) {
        binary += String.fromCharCode(b);
    }
    return `${TEXT_PREFIX}${btoa(binary)}`;
}

function decodeClipboardEnvelope(raw: string): ClipClipboardObject | ParamClipboardObject | null {
    if (!raw.startsWith(TEXT_PREFIX)) return null;
    const body = raw.slice(TEXT_PREFIX.length);
    try {
        const binary = atob(body);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
            bytes[i] = binary.charCodeAt(i);
        }
        const text = new TextDecoder().decode(bytes);
        const parsed = JSON.parse(text) as ClipClipboardObject | ParamClipboardObject;
        if (parsed?.version !== 1 || (parsed?.kind !== "clip" && parsed?.kind !== "param")) {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function mimeFor(kind: ClipboardKind): string {
    return kind === "clip" ? CLIP_MIME : PARAM_MIME;
}

export async function writeSystemClipboardObject(
    payload: ClipClipboardObject | ParamClipboardObject,
): Promise<void> {
    if (hasClipboardReadWrite()) {
        const mime = mimeFor(payload.kind);
        const body = JSON.stringify(payload);
        const item = new ClipboardItem({
            [mime]: new Blob([body], { type: mime }),
            "text/plain": new Blob([encodeClipboardEnvelope(payload)], {
                type: "text/plain",
            }),
        });
        await navigator.clipboard.write([item]);
        return;
    }

    if (hasClipboardTextReadWrite()) {
        await navigator.clipboard.writeText(encodeClipboardEnvelope(payload));
    }
}

export async function readSystemClipboardObject(
    kind: ClipboardKind,
): Promise<ClipClipboardObject | ParamClipboardObject | null> {
    if (hasClipboardReadWrite()) {
        const mime = mimeFor(kind);
        const items = await navigator.clipboard.read();
        for (const item of items) {
            if (!item.types.includes(mime)) continue;
            const blob = await item.getType(mime);
            const text = await blob.text();
            try {
                const parsed = JSON.parse(text) as ClipClipboardObject | ParamClipboardObject;
                if (parsed?.version !== 1 || parsed?.kind !== kind) continue;
                return parsed;
            } catch {
                continue;
            }
        }
    }

    if (hasClipboardTextReadWrite()) {
        const raw = await navigator.clipboard.readText();
        const parsed = decodeClipboardEnvelope(raw);
        if (parsed?.kind === kind) {
            return parsed;
        }
    }

    return null;
}
