/**
 * Clip/Take 源域 → 工程帧的纯函数投影（Rust `param_blocks::projection` 的预览端口）。
 *
 * 【权威在 Rust】本文件只服务交互预览；任何公式改动必须同时更新
 * `param_blocks/projection.rs` 与 golden fixture。
 */

import type { ClipInfo } from "../../../features/session/sessionTypes";
import type { TakeSpan } from "./paramBlockTypes";

export interface ParamBlockSourceClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
    readonly playbackRate?: number;
    readonly reversed?: boolean;
    readonly loopEnabled?: boolean;
    /** 源媒体总时长（秒）；Loop 回绕周期优先取它。 */
    readonly contentDurationSec?: number | null;
    readonly activeTakeId?: string;
    readonly takes?: ReadonlyArray<{
        readonly id: string;
        readonly sourceStartSec: number;
        readonly sourceEndSec: number;
        readonly playbackRate: number;
        readonly reversed: boolean;
        readonly loopEnabled: boolean;
        readonly durationSec?: number;
        readonly durationFrames?: number;
        readonly sourceSampleRate?: number;
        readonly syllableSpansSec?: ReadonlyArray<{
            readonly startSec: number;
            readonly endSec: number;
        }>;
        readonly midiNoteData?: ReadonlyArray<{
            readonly startSec: number;
            readonly endSec: number;
            readonly note: number;
        }>;
    }>;
    readonly syllableSpansSec?: ReadonlyArray<{
        readonly startSec: number;
        readonly endSec: number;
    }>;
    readonly midiNoteData?: ReadonlyArray<{
        readonly startSec: number;
        readonly endSec: number;
        readonly note: number;
    }>;
}

export interface ProjectedInterval {
    readonly cycle: number;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

const EPS = 1e-9;
export const MAX_FRAME = 1_000_000;

export function toFrame(sec: number, framePeriodMs: number): number {
    const value = Math.round((sec * 1000) / Math.max(0.1, framePeriodMs));
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(MAX_FRAME, value));
}

export function toSec(frame: number, framePeriodMs: number): number {
    return (frame * Math.max(0.1, framePeriodMs)) / 1000;
}

function finite(value: number | undefined, fallback: number): number {
    return Number.isFinite(value) ? (value as number) : fallback;
}

function activeTake(clip: ParamBlockSourceClip) {
    return clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
}

export interface ClipProjection {
    readonly clipId: string;
    readonly takeId: string;
    readonly projectStartSec: number;
    readonly projectLengthSec: number;
    readonly sourceWindowStartSec: number;
    readonly sourceWindowEndSec: number;
    readonly mediaDurationSec: number | null;
    readonly rate: number;
    readonly reversed: boolean;
    readonly loopEnabled: boolean;
    readonly loopCycleSec: number | null;
}

export function projectionForClip(clip: ParamBlockSourceClip): ClipProjection {
    const take = activeTake(clip);
    const rate = Math.max(1e-9, finite(clip.playbackRate, 1));
    const span = Math.max(0, finite(clip.lengthSec, 0) * rate);
    const takeStart = finite(take?.sourceStartSec ?? clip.sourceStartSec, 0);
    const takeEnd = finite(take?.sourceEndSec ?? clip.sourceEndSec, takeStart + span);
    const reversed = take?.reversed ?? clip.reversed === true;
    const loopEnabled = take?.loopEnabled ?? clip.loopEnabled === true;
    const windowStart = loopEnabled ? takeStart : reversed ? takeEnd - span : takeStart;
    const windowEnd = loopEnabled ? takeEnd : reversed ? takeEnd : takeStart + span;
    const mediaDuration =
        clip.contentDurationSec ??
        take?.durationSec ??
        (take?.durationFrames && take.sourceSampleRate
            ? take.durationFrames / take.sourceSampleRate
            : null) ??
        (take?.midiNoteData?.length
            ? Math.max(...take.midiNoteData.map((note) => note.endSec))
            : null) ??
        (clip.midiNoteData?.length
            ? Math.max(...clip.midiNoteData.map((note) => note.endSec))
            : null);
    const loopCycle = loopEnabled
        ? Math.max(1e-9, mediaDuration ?? Math.abs(takeEnd - takeStart))
        : null;
    return {
        clipId: clip.id,
        takeId: take?.id ?? clip.activeTakeId ?? `${clip.id}_take_1`,
        projectStartSec: finite(clip.startSec, 0),
        projectLengthSec: Math.max(0, finite(clip.lengthSec, 0)),
        sourceWindowStartSec: windowStart,
        sourceWindowEndSec: windowEnd,
        mediaDurationSec: mediaDuration,
        rate,
        reversed,
        loopEnabled: loopEnabled && loopCycle !== null,
        loopCycleSec: loopCycle,
    };
}

function loopAnchor(projection: ClipProjection): number {
    if (!projection.reversed) return projection.sourceWindowStartSec;
    const duration = projection.mediaDurationSec;
    return duration === null
        ? projection.sourceWindowEndSec
        : Math.min(projection.sourceWindowEndSec, duration);
}

export function projectTakeSpan(
    projection: ClipProjection,
    span: Pick<TakeSpan, "sourceStartSec" | "sourceEndSec" | "cycle">,
    framePeriodMs: number,
): ProjectedInterval[] {
    const fp = Math.max(0.1, framePeriodMs);
    const visibleStart = toFrame(projection.projectStartSec, fp);
    const visibleEnd = toFrame(projection.projectStartSec + projection.projectLengthSec, fp);
    if (visibleEnd <= visibleStart) return [];
    const rate = Math.max(1e-9, projection.rate);
    const out: ProjectedInterval[] = [];
    if (projection.loopEnabled && projection.loopCycleSec !== null) {
        const cycle = projection.loopCycleSec;
        const anchor = loopAnchor(projection);
        const phase = projection.reversed
            ? modEuclid(anchor - span.sourceEndSec, cycle)
            : modEuclid(span.sourceStartSec - anchor, cycle);
        const lengthSec = Math.min((span.sourceEndSec - span.sourceStartSec) / rate, cycle / rate);
        const cycleTimeline = cycle / rate;
        if (cycleTimeline <= EPS) return [];
        for (
            let startLocal = phase / rate, index = 0;
            startLocal < projection.projectLengthSec - EPS && index < 20000;
            startLocal += cycleTimeline, index += 1
        ) {
            const startSec = projection.projectStartSec + startLocal;
            const endSec = Math.min(
                startSec + lengthSec,
                projection.projectStartSec + projection.projectLengthSec,
            );
            const start = Math.max(visibleStart, toFrame(startSec, fp));
            const end = Math.min(visibleEnd, toFrame(endSec, fp));
            if (end > start) out.push({ cycle: index, startFrame: start, endFrameExclusive: end });
        }
        return out;
    }
    const windowStart = projection.reversed
        ? projection.sourceWindowEndSec - projection.projectLengthSec * rate
        : projection.sourceWindowStartSec;
    const windowEnd = projection.reversed
        ? projection.sourceWindowEndSec
        : projection.sourceWindowStartSec + projection.projectLengthSec * rate;
    const startSec = Math.max(span.sourceStartSec, windowStart);
    const endSec = Math.min(span.sourceEndSec, windowEnd);
    if (endSec <= startSec + EPS) return [];
    const project = (source: number) =>
        projection.reversed
            ? projection.projectStartSec + (projection.sourceWindowEndSec - source) / rate
            : projection.projectStartSec + (source - projection.sourceWindowStartSec) / rate;
    const a = project(startSec);
    const b = project(endSec);
    const start = Math.max(visibleStart, toFrame(Math.min(a, b), fp));
    const end = Math.min(visibleEnd, toFrame(Math.max(a, b), fp));
    return end > start ? [{ cycle: 0, startFrame: start, endFrameExclusive: end }] : [];
}

export function inverseProject(
    projection: ClipProjection,
    frame: number,
    cycle: number,
    framePeriodMs: number,
): number | null {
    const fp = Math.max(0.1, framePeriodMs);
    const projectSec = toSec(frame, fp);
    const localSec = projectSec - projection.projectStartSec;
    if (localSec < -EPS || localSec > projection.projectLengthSec + EPS) return null;
    const rate = Math.max(1e-9, projection.rate);
    let source: number;
    if (projection.loopEnabled && projection.loopCycleSec !== null) {
        const cycleSec = projection.loopCycleSec;
        const phase = (localSec - cycle * (cycleSec / rate)) * rate;
        const anchor = loopAnchor(projection);
        source = projection.reversed
            ? anchor - modEuclid(phase, cycleSec)
            : anchor + modEuclid(phase, cycleSec);
    } else {
        source = projection.reversed
            ? projection.sourceWindowEndSec - localSec * rate
            : projection.sourceWindowStartSec + localSec * rate;
    }
    if (projection.mediaDurationSec !== null) {
        source = Math.max(0, Math.min(projection.mediaDurationSec, source));
    }
    return Number.isFinite(source) ? source : null;
}

export function modEuclid(value: number, modulus: number): number {
    return ((value % modulus) + modulus) % modulus;
}

/**
 * 会话里的 `ClipInfo` → 源域投影输入。
 *
 * 它描述的是"源域模型需要 Clip 的哪几个字段"，与持久化无关，所以住在投影模块
 * 而不是某个 hook 里：任何要合成视图的调用方都必须用**同一份**映射，否则
 * "合成用的 clips"与"编辑用的 clips"会悄悄分叉。
 */
export function toParamBlockSourceClip(clip: ClipInfo): ParamBlockSourceClip {
    return {
        id: clip.id,
        trackId: clip.trackId,
        startSec: clip.startSec,
        lengthSec: clip.lengthSec,
        sourceStartSec: clip.sourceStartSec,
        sourceEndSec: clip.sourceEndSec,
        playbackRate: clip.playbackRate,
        reversed: clip.reversed,
        loopEnabled: clip.loopEnabled,
        contentDurationSec: clip.durationSec ?? null,
        activeTakeId: clip.activeTakeId,
        takes: clip.takes?.map((take) => ({
            id: take.id,
            sourceStartSec: take.sourceStartSec,
            sourceEndSec: take.sourceEndSec,
            playbackRate: take.playbackRate,
            reversed: take.reversed,
            loopEnabled: take.loopEnabled,
            durationSec: take.durationSec,
            durationFrames: take.durationFrames,
            sourceSampleRate: take.sourceSampleRate,
            syllableSpansSec: take.syllableSpansSec,
            midiNoteData: take.midiNoteData,
        })),
        syllableSpansSec: clip.syllableSpansSec,
        midiNoteData: clip.midiNoteData,
    };
}

/** 源域区间 → 工程秒区间（供旧 UI 命中/矩形继续工作）。 */
export function projectTakeSpanToSeconds(
    clip: ParamBlockSourceClip,
    span: Pick<TakeSpan, "sourceStartSec" | "sourceEndSec" | "cycle">,
    framePeriodMs: number,
): Array<{ startSec: number; endSec: number; cycle: number }> {
    const projection = projectionForClip(clip);
    return projectTakeSpan(projection, span, framePeriodMs).map((interval) => ({
        startSec: toSec(interval.startFrame, framePeriodMs),
        endSec: toSec(interval.endFrameExclusive, framePeriodMs),
        cycle: interval.cycle,
    }));
}
