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

export function parseClipboardObject(raw: string): ClipClipboardObject | ParamClipboardData | null {
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
    if (record.kind === "clip" && isClipClipboardTemplates(record.templates)) {
        // 槽位里可能躺着旧版本/外部进程写入的残缺载荷：templates 元素缺关键字段
        // 就按"没有可粘贴的本类数据"处理（与文件头的解析纪律一致），而不是让
        // 粘贴路径拿到 NaN 坐标/无名 clip。空数组同样不是可粘贴载荷。
        // 只保留本类载荷的字段，不把外部对象上的任意键原样透传进程序状态。
        const groupIds = (parsed as { groupIds?: unknown }).groupIds;
        return {
            version: 1,
            kind: "clip",
            templates: record.templates.map(sanitizeClipTemplate),
            ...(Array.isArray(groupIds) && groupIds.every((id) => typeof id === "string")
                ? { groupIds: groupIds as string[] }
                : {}),
        };
    }
    return null;
}

/**
 * Clip 模板元素的最小形状校验：粘贴路径必用的标量字段必须存在且为有限数
 * （`trackId`/`name`/`startSec`/`lengthSec`，见 `ClipTemplate` 的必填面）。
 * 可选数值字段同样按类型过滤：外部载荷里的 NaN / Infinity / 字符串会把
 * 几何、速率与声道模式带进 reducer，表现为坏 clip 而没有任何解析错误。
 */
const CLIP_TEMPLATE_OPTIONAL_NUMBER_KEYS = [
    "gain",
    "clip_gain",
    "durationSec",
    "durationFrames",
    "sourceSampleRate",
    "sourceChannels",
    "sourceStartSec",
    "sourceEndSec",
    "playbackRate",
    "clipPlaybackRate",
    "midiNoteCount",
    "channelMode",
    "loopPeriodSec",
    "snapOffsetSec",
    "fadeInSec",
    "fadeOutSec",
    "fadeInShape",
    "fadeOutShape",
    "fadeInDir",
    "fadeOutDir",
    "fadeInS",
    "fadeOutS",
    "autoFadeInSec",
    "autoFadeOutSec",
] as const satisfies readonly (keyof ClipTemplate)[];

/** 布尔 / 字符串字段的允许形态；外来载荷里的错型值一律按缺省处理。 */
const CLIP_TEMPLATE_OPTIONAL_BOOLEAN_KEYS = [
    "muted",
    "reversed",
    "loopEnabled",
    "midiFillGaps",
] as const satisfies readonly (keyof ClipTemplate)[];

const CLIP_TEMPLATE_OPTIONAL_STRING_KEYS = [
    "activeTakeId",
    "sourcePath",
    "sourcePathRelative",
    "sourceClipId",
    "hostMedia",
    "hostMediaReason",
] as const satisfies readonly (keyof ClipTemplate)[];

/** 复合字段：只放行纯对象 / 数组，且限制规模，避免外来载荷带进巨型结构。 */
const CLIP_TEMPLATE_OPTIONAL_OBJECT_KEYS = [
    "takes",
    "waveformPreview",
    "linkedParams",
    "hostFades",
    "formantMorph",
    "midiNoteData",
    "syllableSpansSec",
    "syllableNotes",
    "syllableEventIds",
    "syllableAnalysisWindowSec",
    "gameBlockBaseline",
] as const satisfies readonly (keyof ClipTemplate)[];

const CLIP_COLORS = new Set(["blue", "violet", "emerald", "amber", "cyan"]);

/**
 * 只把**已知且形态正确**的字段搬进程序状态。
 *
 * 【为什么不直接 cast 原对象】剪贴板槽位是跨进程共享的外部输入：未列入白名单
 * 的键（含 `__proto__` 之类的注入）、错型的波形预览 / 曲线、超长字符串都会
 * 被粘贴路径原样带进 reducer 与渲染层。逐字段挑出已知面，解析纪律才完整。
 */
function sanitizeClipTemplate(value: ClipTemplate): ClipTemplate {
    const source = value as Record<string, unknown>;
    const out: Record<string, unknown> = {
        trackId: source.trackId,
        name: source.name,
        startSec: source.startSec,
        lengthSec: source.lengthSec,
    };
    for (const key of CLIP_TEMPLATE_OPTIONAL_NUMBER_KEYS) {
        if (typeof source[key] === "number" && Number.isFinite(source[key])) {
            out[key] = source[key];
        }
    }
    for (const key of CLIP_TEMPLATE_OPTIONAL_BOOLEAN_KEYS) {
        if (typeof source[key] === "boolean") out[key] = source[key];
    }
    for (const key of CLIP_TEMPLATE_OPTIONAL_STRING_KEYS) {
        if (typeof source[key] === "string" && source[key].length <= 4096) {
            out[key] = source[key];
        }
    }
    if (typeof source.color === "string" && CLIP_COLORS.has(source.color)) {
        out.color = source.color;
    }
    if (source.loopPeriodSec === null) out.loopPeriodSec = null;
    for (const key of CLIP_TEMPLATE_OPTIONAL_OBJECT_KEYS) {
        const value = source[key];
        if (value === null || typeof value !== "object") continue;
        // JSON.parse 产出的一定是普通对象 / 数组；这里再挡一次 null 原型与
        // 超大载荷（剪贴板是跨进程输入，不该决定本进程的内存分配）。
        if (Array.isArray(value)) {
            if (value.length > 100_000) continue;
            out[key] = value;
        } else if (Object.getPrototypeOf(value) === Object.prototype) {
            if (Object.keys(value as object).length > 10_000) continue;
            out[key] = value;
        }
    }
    return out as ClipTemplate;
}

function isClipClipboardTemplates(value: unknown): value is ClipTemplate[] {
    // 空 templates 不是"可粘贴的 clip"：返回它只会让粘贴路径空转一次。
    if (!Array.isArray(value) || value.length === 0) return false;
    return value.every((template): template is ClipTemplate => {
        if (!template || typeof template !== "object" || Array.isArray(template)) return false;
        const candidate = template as Record<string, unknown>;
        if (
            typeof candidate.trackId !== "string" ||
            typeof candidate.name !== "string" ||
            typeof candidate.startSec !== "number" ||
            !Number.isFinite(candidate.startSec) ||
            typeof candidate.lengthSec !== "number" ||
            !Number.isFinite(candidate.lengthSec) ||
            !(candidate.lengthSec > 0)
        ) {
            return false;
        }
        for (const key of CLIP_TEMPLATE_OPTIONAL_NUMBER_KEYS) {
            const optional = candidate[key];
            // loopPeriodSec 允许为 null（宿主未给周期时按媒体时长推断）。
            if (key === "loopPeriodSec" && optional === null) continue;
            if (
                optional !== undefined &&
                (typeof optional !== "number" || !Number.isFinite(optional))
            ) {
                return false;
            }
        }
        return true;
    });
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

export async function writeSystemClipboardObject(payload: SystemClipboardObject): Promise<void> {
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
