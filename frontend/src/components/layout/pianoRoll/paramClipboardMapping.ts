/**
 * paramClipboardMapping.ts — 参数线剪贴板的载荷格式与「剪贴板 → 目标选区」映射。
 *
 * 为什么剪贴板要带偏移
 * --------------------
 * 多选区复制时，被复制的是**若干互不相连的段**（如 0~1s、2~3s）。若像旧实现
 * 那样只存一维 `values[]`，1~2s 的断层信息就丢失了 —— 粘贴时会把两段首尾
 * 相接压成 0~2s。因此载荷升级为 `segments: [{startFrame, values}]`，
 * `startFrame` 是**相对复制起点**的帧偏移，断层以偏移空洞的形式被完整保留。
 *
 * 唯一的映射规则（预览 / 粘贴 / 剪切后映射共用，见 mapClipboardToTargetRanges）
 * ---------------------------------------------------------------------------
 * 设目标选区（多段）的帧区间集合为 T，O = T 中第一段的起点帧（**选区整体
 * 起点**，不是每段各自起点）；剪贴板第 j 段占据偏移区间 C_j。
 * 对每个目标帧 f ∈ T，令 off = f − O：
 *   - off 落在某个 C_j 内 → 该帧写入 values[off − C_j.startFrame]；
 *   - 否则（落在剪贴板空洞里，或超出剪贴板范围）→ **不写**。
 *
 * 这条规则同时满足需求里的两个方向：
 *   - 复制 0~1s、2~3s → 目标 0~3s：只写 0~1s 与 2~3s，1~2s 保持原值（不合并）；
 *   - 复制 0~3s → 目标 0~1s、2~3s：第一段取剪贴板开头 1s，第二段取剪贴板
 *     最后 1s（因为该段相对选区起点偏移 2s），1~2s 既无预览也无写入。
 * 单段 ↔ 单段时退化为改造前的行为（对齐到选区起点 + 超出选区长度即截断）。
 *
 * 帧率不做重采样（与改造前一致）：帧号即写入位置的下标，`framePeriodMs` 只
 * 用于诊断与预览的时长换算。
 */

import type { ParamName } from "./types";
import type { FrameRange } from "./paramSelection";

/** 剪贴板中的一段：`startFrame` 为相对复制起点的帧偏移。 */
export interface ParamClipboardSegment {
    startFrame: number;
    values: number[];
}

/** 剪贴板领域模型（内部缓存与系统载荷解析后共用）。 */
export interface ParamClipboardData {
    param: ParamName;
    framePeriodMs: number;
    segments: ParamClipboardSegment[];
}

/** 规划出的一段落盘写入：绝对帧号 + 逐帧值。 */
export interface ClipboardWrite {
    startFrame: number;
    values: number[];
}

/** 当前系统剪贴板载荷版本。v1（无 segments）仍可读入，见 parseParamClipboardPayload。 */
export const PARAM_CLIPBOARD_VERSION = 2;

export interface ParamClipboardPayloadV2 {
    version: 2;
    kind: "param";
    param: ParamName;
    framePeriodMs: number;
    segments: ParamClipboardSegment[];
}

/** v1 载荷：单段、无起始偏移（仅在读取旧进程/旧版本写入的槽位时出现）。 */
export interface ParamClipboardPayloadV1 {
    version: 1;
    kind: "param";
    param: ParamName;
    framePeriodMs: number;
    values: number[];
}

export type ParamClipboardPayload = ParamClipboardPayloadV2 | ParamClipboardPayloadV1;

function sanitizeValues(raw: unknown): number[] {
    if (!Array.isArray(raw)) return [];
    const out = new Array<number>(raw.length);
    for (let i = 0; i < raw.length; i += 1) {
        out[i] = Number(raw[i]) || 0;
    }
    return out;
}

/** 剪贴板覆盖的总帧数（各段值长度之和，不含断层）。 */
export function clipboardTotalFrames(clipboard: ParamClipboardData | null): number {
    if (!clipboard) return 0;
    let total = 0;
    for (const segment of clipboard.segments) total += segment.values.length;
    return total;
}

/** 剪贴板末尾偏移（最后一段相对复制起点的结束帧，独占）；空载荷为 0。 */
export function clipboardSpanFrames(clipboard: ParamClipboardData | null): number {
    if (!clipboard) return 0;
    let end = 0;
    for (const segment of clipboard.segments) {
        end = Math.max(end, segment.startFrame + segment.values.length);
    }
    return end;
}

/**
 * 归一化剪贴板：段按偏移升序、丢弃空段、值去除非有限。
 * 我们自己的 copy 产出的段天然互不相交；外来载荷若存在重叠段，后写覆盖
 * 先写（mapClipboardToTargetRanges 的输出按起点升序，落盘顺序即此顺序）。
 */
export function normalizeClipboardData(
    data: ParamClipboardData | null | undefined,
): ParamClipboardData | null {
    if (!data) return null;
    const segments: ParamClipboardSegment[] = [];
    for (const segment of data.segments ?? []) {
        const values = sanitizeValues(segment.values);
        if (values.length === 0) continue;
        const startFrame = Math.max(0, Math.round(Number(segment.startFrame) || 0));
        segments.push({ startFrame, values });
    }
    if (segments.length === 0) return null;
    segments.sort((a, b) => a.startFrame - b.startFrame);
    return {
        param: data.param,
        framePeriodMs: Math.max(1e-6, Number(data.framePeriodMs) || 5),
        segments,
    };
}

/** 领域模型 → v2 系统载荷（写入系统剪贴板前调用）。 */
export function toParamClipboardPayload(data: ParamClipboardData): ParamClipboardPayloadV2 {
    const normalized = normalizeClipboardData(data);
    return {
        version: PARAM_CLIPBOARD_VERSION,
        kind: "param",
        param: data.param,
        framePeriodMs: normalized?.framePeriodMs ?? Math.max(1e-6, Number(data.framePeriodMs) || 5),
        segments: normalized?.segments ?? [],
    };
}

/**
 * 解析系统剪贴板载荷（v2 优先，v1 兼容为单段）。
 * 解析失败 / 空载荷 → null，由调用方视为「剪贴板里没有参数线数据」。
 */
export function parseParamClipboardPayload(value: unknown): ParamClipboardData | null {
    if (!value || typeof value !== "object") return null;
    const raw = value as Record<string, unknown>;
    if (raw.kind !== "param") return null;
    const param = typeof raw.param === "string" ? raw.param : "";
    if (!param) return null;
    const framePeriodMs = Math.max(1e-6, Number(raw.framePeriodMs) || 5);

    if (raw.version === 1 && Array.isArray(raw.values)) {
        const values = sanitizeValues(raw.values);
        if (values.length === 0) return null;
        return { param, framePeriodMs, segments: [{ startFrame: 0, values }] };
    }
    if (raw.version !== PARAM_CLIPBOARD_VERSION) return null;
    const segmentsRaw = Array.isArray(raw.segments) ? raw.segments : [];
    const segments: ParamClipboardSegment[] = [];
    for (const entry of segmentsRaw) {
        if (!entry || typeof entry !== "object") continue;
        const seg = entry as Record<string, unknown>;
        segments.push({
            startFrame: Math.max(0, Math.round(Number(seg.startFrame) || 0)),
            values: sanitizeValues(seg.values),
        });
    }
    return normalizeClipboardData({ param, framePeriodMs, segments });
}

/**
 * 把剪贴板映射到目标选区的帧区间上（预览与粘贴的唯一实现）。
 *
 * @param targetRanges 目标选区的帧区间（升序、互不相交；见 beatRangesToFrameRanges）
 * @returns 按起点升序的写入片段；空数组表示「按规则没有任何帧会被写入」
 *          （过去用于「不显示预览、不执行粘贴」）。
 */
export function mapClipboardToTargetRanges(args: {
    targetRanges: readonly FrameRange[];
    clipboard: ParamClipboardData | null | undefined;
}): ClipboardWrite[] {
    const clipboard = normalizeClipboardData(args.clipboard);
    const targetRanges = args.targetRanges;
    if (!clipboard || targetRanges.length === 0) return [];

    // 偏移基准：目标选区**整体**起点（第一段起点）。用「每段各自起点」会让
    // 第二段错拿剪贴板开头（需求明确要求第二段取剪贴板的对应偏移段）。
    const origin = targetRanges[0].startFrame;

    const writes: ClipboardWrite[] = [];
    for (const target of targetRanges) {
        const frameCount = Math.max(0, Math.floor(target.frameCount));
        if (frameCount === 0) continue;
        const windowStart = target.startFrame - origin; // 目标区间在剪贴板坐标系中的偏移
        const windowEnd = windowStart + frameCount; // 独占
        for (const segment of clipboard.segments) {
            const from = Math.max(windowStart, segment.startFrame);
            const to = Math.min(windowEnd, segment.startFrame + segment.values.length);
            if (to <= from) continue;
            const values = segment.values.slice(
                from - segment.startFrame,
                to - segment.startFrame,
            );
            if (values.length === 0) continue;
            writes.push({ startFrame: origin + from, values });
        }
    }
    writes.sort((a, b) => a.startFrame - b.startFrame);
    return writes;
}

/**
 * 剪贴板在「目标选区坐标系」中的预览片段（秒区间 + 逐帧值）。
 *
 * 与 mapClipboardToTargetRanges 同源：预览画的就是粘贴会落下的数据，
 * 因此断层两侧的截断完全一致。时间换算用**目标帧周期**（粘贴是按帧号
 * 落盘的，用剪贴板帧周期换算会让预览与结果错位）。
 */
export function clipboardPreviewSpans(args: {
    targetRanges: readonly FrameRange[];
    clipboard: ParamClipboardData | null | undefined;
    targetFramePeriodMs: number;
}): Array<{ startSec: number; framePeriodMs: number; values: number[] }> {
    const fp = Math.max(1e-6, Number(args.targetFramePeriodMs) || 5);
    return mapClipboardToTargetRanges({
        targetRanges: args.targetRanges,
        clipboard: args.clipboard,
    }).map((write) => ({
        startSec: (write.startFrame * fp) / 1000,
        framePeriodMs: fp,
        values: write.values,
    }));
}
