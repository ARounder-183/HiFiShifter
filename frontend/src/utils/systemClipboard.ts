/*
 * 系统剪贴板对象读写工具。
 *
 * 实际传输由 Rust 后端负责：写入平台原生自定义格式，并附带 base64 文本
 * 信封作为回退。这样两个 HiFiShifter 进程之间复制粘贴不依赖 WebView 的
 * 剪贴板权限，在 Windows / macOS / Linux 上均可工作。
 *
 * 单剪贴板纪律：时间轴 Clip 与参数线载荷共享同一个原生格式槽位、互相
 * 覆盖（最后复制的获胜）；读取方按自身格式解析，解析失败即视为"没有
 * 可粘贴的本类数据"（时间轴侧统一报 timeline_clipboard_empty，由前端
 * 映射为"剪贴板中没有可粘贴的内容"提示）。
 *
 * 参数线载荷当前版本为 v2（`segments` + 相对偏移，见
 * pianoRoll/paramClipboardMapping）。v1（单段 `values`）仍可读入，因此
 * 槽位里由旧版本进程写入的数据不会让粘贴失效。
 */

import type { ClipTemplate } from "../features/session/sessionTypes";
import {
    clipboardTotalFrames,
    parseParamClipboardPayload,
    type ParamClipboardData,
    type ParamClipboardPayload,
} from "../components/layout/pianoRoll/paramClipboardMapping";
import { invoke } from "../services/invoke";

type ClipboardKind = "clip" | "param";

/** 解析结果：Clip 载荷保留 wire 形状，参数线载荷已归一为领域模型。 */
type ParsedSystemClipboardObject = ClipClipboardObject | ParamClipboardData;

export interface ClipClipboardObject {
    version: 1;
    kind: "clip";
    templates: ClipTemplate[];
    groupIds?: string[];
}

export type SystemClipboardObject = ClipClipboardObject | ParamClipboardPayload;

function parseClipboardObject(raw: string): ClipClipboardObject | ParamClipboardData | null {
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        return null;
    }
    if (!parsed || typeof parsed !== "object") return null;
    const record = parsed as { kind?: unknown; templates?: unknown };
    if (record.kind === "param") {
        return parseParamClipboardPayload(parsed);
    }
    if (record.kind === "clip" && Array.isArray(record.templates)) {
        return parsed as ClipClipboardObject;
    }
    return null;
}

function clipboardSummary(payload: SystemClipboardObject): string {
    if (payload.kind === "param") {
        // v1（单段）/ v2（多段）都经同一解析器归一，摘要只关心总帧数与段数。
        const data = parseParamClipboardPayload(payload);
        const frames = clipboardTotalFrames(data);
        const rangeCount = data?.segments.length ?? 0;
        const rangeText = rangeCount > 1 ? ` in ${rangeCount} range(s)` : "";
        return `HiFiShifter: ${frames} parameter frame(s)${rangeText} copied. Paste in HiFiShifter Parameter Editor.`;
    }
    return `HiFiShifter: ${payload.templates.length} clip(s) copied. Paste in HiFiShifter timeline.`;
}

export async function writeSystemClipboardObject(
    payload: SystemClipboardObject,
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

export function readSystemClipboardObject(kind: "clip"): Promise<ClipClipboardObject | null>;
export function readSystemClipboardObject(kind: "param"): Promise<ParamClipboardData | null>;
export async function readSystemClipboardObject(
    kind: ClipboardKind,
): Promise<ParsedSystemClipboardObject | null> {
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
    if (!parsed) return null;
    // 参数线载荷解析后已无 `kind` 字段（领域模型），按判别字段分派。
    const parsedKind: ClipboardKind = "kind" in parsed ? "clip" : "param";
    return parsedKind === kind ? parsed : null;
}
