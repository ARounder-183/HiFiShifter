/**
 * Clip/Take 源域 → 工程帧的纯函数投影（Rust `param_blocks::projection` 的预览端口）。
 *
 * 【权威在 Rust】本文件只服务交互预览；任何公式改动必须同时更新
 * `param_blocks/projection.rs` 与 golden fixture。
 */

import type { ClipInfo } from "../../../features/session/sessionTypes";
import type { TakeSpan } from "./paramBlockTypes";

export interface ParamBlockSourceNote {
    readonly startSec: number;
    readonly endSec: number;
    readonly note?: number | null;
    readonly eventId?: string;
    readonly source: "syllable" | "gameBaseline" | "midiNote";
}

export function sourceNoteEventId(span: {
    readonly startSec: number;
    readonly endSec: number;
    readonly eventId?: string;
}): string {
    return span.eventId ?? `${span.startSec.toFixed(9)}:${span.endSec.toFixed(9)}`;
}

export interface ParamBlockSourceClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
    readonly playbackRate?: number;
    /** Clip 级播放倍率；与 take.playbackRate 相乘得到有效速率。 */
    readonly clipPlaybackRate?: number;
    readonly reversed?: boolean;
    readonly loopEnabled?: boolean;
    /** 宿主给出的 Loop 周期（秒）；优先于媒体时长。 */
    readonly loopPeriodSec?: number | null;
    /** 源媒体总时长（秒）；Loop 回绕周期的兜底。 */
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
            readonly note?: number | null;
            readonly eventId?: string;
        }>;
        readonly syllableNotes?: ReadonlyArray<number | null>;
        readonly syllableEventIds?: ReadonlyArray<string>;
        readonly syllableAnalysisWindowSec?: readonly [number, number];
        readonly gameBlockBaseline?: {
            readonly revision?: number;
            readonly algorithm?: string;
            readonly algorithmVersion?: number;
            readonly sourceSpansSec?: ReadonlyArray<{
                readonly startSec: number;
                readonly endSec: number;
                readonly note?: number | null;
                readonly eventId?: string;
            }>;
            readonly analyzedWindowSec?: readonly [number, number];
        };
        readonly midiNoteData?: ReadonlyArray<{
            readonly startSec: number;
            readonly endSec: number;
            readonly note: number;
        }>;
    }>;
    readonly syllableSpansSec?: ReadonlyArray<{
        readonly startSec: number;
        readonly endSec: number;
        readonly note?: number | null;
        readonly eventId?: string;
    }>;
    readonly syllableNotes?: ReadonlyArray<number | null>;
    readonly syllableEventIds?: ReadonlyArray<string>;
    readonly syllableAnalysisWindowSec?: readonly [number, number];
    readonly gameBlockBaseline?: {
        readonly sourceSpansSec?: ReadonlyArray<{
            readonly startSec: number;
            readonly endSec: number;
            readonly note?: number | null;
        }>;
        readonly analyzedWindowSec?: readonly [number, number];
    };
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
    /** The visible interval starts inside the source span (window/overlap cut). */
    readonly cutStart: boolean;
    /** The visible interval ends inside the source span (window/overlap cut). */
    readonly cutEnd: boolean;
}

/** Minimal source geometry accepted by the projection port. */
export type ProjectableTakeSpan = Pick<TakeSpan, "sourceStartSec" | "sourceEndSec" | "cycle">;

/**
 * A source-domain fragment that belongs to a single Loop cycle.
 *
 * cutStart / cutEnd are timeline-oriented: they describe the left and right
 * visible edges. Reverse playback swaps which source edge maps to which flag.
 */
export interface LoopSpanPart<T extends ProjectableTakeSpan = TakeSpan> {
    readonly span: T;
    readonly cutStart: boolean;
    readonly cutEnd: boolean;
}

const EPS = 1e-9;
/** Shared tolerance for positive source-domain intersections. */
const SOURCE_EPS = 1e-12;
const MAX_FRAME = 1_000_000;

/** A positive source/window intersection; never silently drops a left-cut note. */
export function windowIntersection(
    span: readonly [number, number],
    window: readonly [number, number],
): readonly [number, number] | null {
    const start = Math.max(span[0], window[0]);
    const end = Math.min(span[1], window[1]);
    return end > start + SOURCE_EPS ? [start, end] : null;
}

/** A positive continuous intersection rounded to one frame is still visible. The fabricated frame must stay inside the clip's visible range — a sliver at the very end rounds past `visibleEnd`, and emitting `[visibleEnd, visibleEnd + 1)` would collide with the next clip's first block (mirrors `projection.rs`). */
export function positiveIntervalToFrames(
    start: number,
    end: number,
    visibleEnd: number,
): readonly [number, number] | null {
    if (end > start) return [start, end];
    return start < MAX_FRAME && start < visibleEnd ? [start, start + 1] : null;
}

/** Cut flags for a non-loop source/window intersection. */
export function cutFlagsForIntersection(
    span: readonly [number, number],
    window: readonly [number, number],
    reversed: boolean,
): readonly [boolean, boolean] {
    return reversed
        ? [span[1] > window[1] + EPS, span[0] < window[0] - EPS]
        : [span[0] < window[0] - EPS, span[1] > window[1] + EPS];
}

/** Analysis status has one definition for reanalysis, diagnostics and compose. */
export function paramBlockAnalysisRequestKey(args: {
    readonly rootTrackId: string;
    readonly clipId: string;
    readonly activeTakeId?: string;
    readonly sourceDigest?: string;
    readonly algorithm?: string;
    readonly algorithmVersion?: number;
    readonly audible: { readonly startSec: number; readonly endSec: number };
    readonly framePeriodMs: number;
}): string {
    const round = (value: number) => value.toFixed(6);
    return [
        args.rootTrackId,
        args.clipId,
        args.activeTakeId ?? "",
        args.sourceDigest ?? "",
        args.algorithm ?? "",
        String(args.algorithmVersion ?? 0),
        round(args.audible.startSec),
        round(args.audible.endSec),
        round(args.framePeriodMs),
    ].join("|");
}

export type AnalysisWindowStatus = "missing" | "stale" | "current";
export function analysisWindowStatus(
    audible: { readonly startSec: number; readonly endSec: number },
    analyzed: readonly [number, number] | null | undefined,
    spans: readonly unknown[] | null | undefined,
): AnalysisWindowStatus {
    if (!spans || spans.length === 0) return "missing";
    return windowContained(audible, analyzed) ? "current" : "stale";
}

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

function nonEmpty<T>(value: readonly T[] | undefined | null): value is readonly T[] {
    return Array.isArray(value) && value.length > 0;
}

function firstSpanList<T>(
    ...candidates: ReadonlyArray<readonly T[] | undefined | null>
): readonly T[] | null {
    for (const candidate of candidates) {
        if (nonEmpty(candidate)) return candidate;
    }
    return null;
}

export function projectionForClip(clip: ParamBlockSourceClip): ClipProjection {
    const take = activeTake(clip);
    /*
     * 【与 Rust `ClipProjection::from_clip_take` 同源】有效速率 = Clip 级倍率 ×
     * Take 级倍率。ClipInfo.playbackRate 通常是两者相乘后的有效值，但插件/乐观
     * 更新可能只写其中一个；两个来源都读，才能保证前后端投影不漂移。
     */
    const takeRate = Math.max(1e-9, finite(take?.playbackRate, 1));
    const clipRate = finite(clip.clipPlaybackRate, Number.NaN);
    const rate = Number.isFinite(clipRate)
        ? Math.max(1e-9, clipRate * takeRate)
        : Math.max(1e-9, finite(clip.playbackRate, 1));
    const span = Math.max(0, finite(clip.lengthSec, 0) * rate);
    const takeStart = finite(take?.sourceStartSec ?? clip.sourceStartSec, 0);
    const takeEnd = finite(take?.sourceEndSec ?? clip.sourceEndSec, takeStart + span);
    const reversed = take?.reversed ?? clip.reversed === true;
    const loopEnabled = take?.loopEnabled ?? clip.loopEnabled === true;
    const windowStart = loopEnabled ? takeStart : reversed ? takeEnd - span : takeStart;
    const windowEnd = loopEnabled ? takeEnd : reversed ? takeEnd : takeStart + span;
    const durationFromFrames =
        take?.durationFrames && take?.sourceSampleRate && take.sourceSampleRate > 0
            ? take.durationFrames / take.sourceSampleRate
            : null;
    const mediaDuration =
        (durationFromFrames !== null && durationFromFrames > 0 ? durationFromFrames : null) ??
        (take?.durationSec !== undefined && take.durationSec > 0 ? take.durationSec : null) ??
        (clip.contentDurationSec !== undefined &&
        clip.contentDurationSec !== null &&
        clip.contentDurationSec > 0
            ? clip.contentDurationSec
            : null) ??
        (take?.midiNoteData?.length
            ? Math.max(...take.midiNoteData.map((note) => note.endSec))
            : null) ??
        (clip.midiNoteData?.length
            ? Math.max(...clip.midiNoteData.map((note) => note.endSec))
            : null);
    /*
     * Loop 周期优先级与后端逐项一致：宿主读数 → 精确帧数/采样率 → durationSec
     * → 媒体时长 → 源窗口宽度。空数组/0 都必须继续回退，不能当成“已给出”。
     */
    const loopCycle = loopEnabled
        ? Math.max(
              1e-9,
              (clip.loopPeriodSec !== undefined &&
              clip.loopPeriodSec !== null &&
              clip.loopPeriodSec > 1e-9
                  ? clip.loopPeriodSec
                  : null) ??
                  (durationFromFrames !== null && durationFromFrames > 1e-9
                      ? durationFromFrames
                      : null) ??
                  (take?.durationSec !== undefined && take.durationSec > 1e-9
                      ? take.durationSec
                      : null) ??
                  (mediaDuration !== null && mediaDuration > 1e-9 ? mediaDuration : null) ??
                  Math.abs(takeEnd - takeStart),
          )
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

/** 基块分析区间的统一回退链：音节分析 → GAME baseline → MIDI。 */
export function baseSourceSpansForClip(clip: ParamBlockSourceClip): {
    readonly spans: readonly ParamBlockSourceNote[];
    readonly label: "syllable" | "gameBaseline" | "midiNote";
} | null {
    const take = activeTake(clip);
    const syllable = firstSpanList(take?.syllableSpansSec, clip.syllableSpansSec);
    if (syllable !== null) {
        const notes = take?.syllableEventIds ?? clip.syllableEventIds;
        return {
            spans: syllable.map((span, index) => ({
                ...span,
                eventId: span.eventId ?? notes?.[index],
                source: "syllable",
            })),
            label: "syllable",
        };
    }
    const baseline = firstSpanList(
        take?.gameBlockBaseline?.sourceSpansSec,
        clip.gameBlockBaseline?.sourceSpansSec,
    );
    if (baseline !== null) {
        return {
            spans: baseline.map((span) => ({ ...span, source: "gameBaseline" })),
            label: "gameBaseline",
        };
    }
    const midi = firstSpanList(take?.midiNoteData, clip.midiNoteData);
    if (midi !== null) {
        return {
            spans: midi.map((note) => ({
                startSec: note.startSec,
                endSec: note.endSec,
                note: note.note,
                source: "midiNote",
            })),
            label: "midiNote",
        };
    }
    return null;
}

/** Frontend mirror of Rust `ClipProjection::audible_source_window`. A reversed Take's ring extends **downward** from its anchor (`sourceAtProjectedFrame` / audio wrap all agree), so its section window is `[anchor - cycle, anchor]`. */
export function audibleSourceWindow(projection: ClipProjection): {
    readonly startSec: number;
    readonly endSec: number;
} {
    const span = Math.max(0, projection.projectLengthSec * projection.rate);
    let start: number;
    let end: number;
    if (projection.loopEnabled && projection.loopCycleSec !== null) {
        const cycle = projection.loopCycleSec;
        const anchor = loopAnchor(projection);
        const media = projection.mediaDurationSec;
        // The ring wraps once the cycle reaches beyond one side of the media:
        // forward past the end, reversed below zero.
        const wraps = (duration: number) =>
            projection.reversed
                ? anchor - cycle < -EPS || anchor > duration + EPS
                : anchor + cycle > duration + EPS || anchor < -EPS;
        const wholeRing = media !== null && media > EPS && (cycle >= media - EPS || wraps(media));
        if (wholeRing) {
            start = 0;
            end = media as number;
        } else if (cycle > EPS && Number.isFinite(anchor) && anchor >= -EPS) {
            if (projection.reversed) {
                start = anchor - cycle;
                end = anchor;
            } else {
                start = anchor;
                end = anchor + cycle;
            }
        } else {
            start = projection.sourceWindowStartSec;
            end = projection.sourceWindowEndSec;
        }
    } else if (projection.reversed) {
        start = projection.sourceWindowEndSec - span;
        end = projection.sourceWindowEndSec;
    } else {
        start = projection.sourceWindowStartSec;
        end = projection.sourceWindowStartSec + span;
    }
    if (!Number.isFinite(start) || !Number.isFinite(end)) return { startSec: 0, endSec: 0 };
    return { startSec: Math.min(start, end), endSec: Math.max(start, end) };
}

/** An analysis is current only when its window contains the audible window. */
export function windowContained(
    audible: { readonly startSec: number; readonly endSec: number },
    analyzed: readonly [number, number] | null | undefined,
): boolean {
    return (
        Array.isArray(analyzed) &&
        analyzed.length === 2 &&
        analyzed.every((value) => Number.isFinite(value)) &&
        analyzed[1] >= analyzed[0] &&
        audible.startSec >= analyzed[0] - 1e-6 &&
        audible.endSec <= analyzed[1] + 1e-6
    );
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
    span: ProjectableTakeSpan,
    framePeriodMs: number,
    fromCycle = span.cycle,
): ProjectedInterval[] {
    const parts = splitLoopTakeSpanParts(projection, span);
    if (parts.length === 0) return [];
    const out: ProjectedInterval[] = [];
    for (const part of parts) {
        out.push(...projectTakeSpanPart(projection, part, framePeriodMs, fromCycle));
    }
    if (projection.loopEnabled) {
        out.sort(
            (a, b) =>
                a.startFrame - b.startFrame ||
                a.endFrameExclusive - b.endFrameExclusive ||
                a.cycle - b.cycle,
        );
    }
    return out;
}

export function projectTakeSpanPart<T extends ProjectableTakeSpan>(
    projection: ClipProjection,
    part: LoopSpanPart<T>,
    framePeriodMs: number,
    fromCycle = part.span.cycle,
): ProjectedInterval[] {
    const fp = Math.max(0.1, framePeriodMs);
    const visibleStart = toFrame(projection.projectStartSec, fp);
    const visibleEnd = toFrame(projection.projectStartSec + projection.projectLengthSec, fp);
    if (visibleEnd <= visibleStart) return [];
    const rate = Math.max(1e-9, projection.rate);
    if (projection.loopEnabled && projection.loopCycleSec !== null) {
        const cycleSec = projection.loopCycleSec;
        const anchor = loopAnchor(projection);
        const phase = projection.reversed
            ? modEuclid(anchor - part.span.sourceEndSec, cycleSec)
            : modEuclid(part.span.sourceStartSec - anchor, cycleSec);
        const widthSec = Math.max(0, part.span.sourceEndSec - part.span.sourceStartSec);
        const lengthSec = Math.min(widthSec / rate, Math.max(0, (cycleSec - phase) / rate));
        if (!(lengthSec > SOURCE_EPS)) return [];
        const cycleTimeline = cycleSec / rate;
        if (cycleTimeline <= EPS) return [];
        const clipStart = projection.projectStartSec;
        const clipEnd = clipStart + projection.projectLengthSec;
        const startCycle = Number.isFinite(fromCycle) ? Math.max(0, Math.floor(fromCycle)) : 0;
        const out: ProjectedInterval[] = [];
        let cycle = startCycle;
        let startLocal = phase / rate + startCycle * cycleTimeline;
        while (startLocal < projection.projectLengthSec - EPS && out.length < 20_000) {
            const rawStartSec = clipStart + startLocal;
            const rawEndSec = rawStartSec + lengthSec;
            const start = Math.max(visibleStart, toFrame(rawStartSec, fp));
            let end = Math.min(visibleEnd, toFrame(rawEndSec, fp));
            // Rounding may collapse a positive source interval to zero frames.
            // A visible block still deserves one frame of pointer target.
            if (
                end <= start &&
                rawEndSec > rawStartSec + SOURCE_EPS &&
                rawStartSec < clipEnd - EPS &&
                rawEndSec > clipStart + EPS &&
                start < visibleEnd
            ) {
                end = Math.min(visibleEnd, start + 1);
            }
            if (end > start) {
                out.push({
                    cycle,
                    startFrame: start,
                    endFrameExclusive: end,
                    cutStart: part.cutStart || rawStartSec < clipStart - EPS,
                    cutEnd: part.cutEnd || rawEndSec > clipEnd + EPS,
                });
            }
            startLocal += cycleTimeline;
            cycle += 1;
        }
        return out;
    }
    return projectNonLoopSpanPart(projection, part, fp, visibleStart, visibleEnd);
}

function projectNonLoopSpanPart<T extends ProjectableTakeSpan>(
    projection: ClipProjection,
    part: LoopSpanPart<T>,
    framePeriodMs: number,
    visibleStart: number,
    visibleEnd: number,
): ProjectedInterval[] {
    const { startSec: windowStart, endSec: windowEnd } = audibleSourceWindow(projection);
    const intersection = windowIntersection(
        [part.span.sourceStartSec, part.span.sourceEndSec],
        [windowStart, windowEnd],
    );
    if (intersection === null) return [];
    const [startSec, endSec] = intersection;
    const project = (source: number) =>
        projection.reversed
            ? projection.projectStartSec +
              (projection.sourceWindowEndSec - source) / projection.rate
            : projection.projectStartSec +
              (source - projection.sourceWindowStartSec) / projection.rate;
    const a = project(startSec);
    const b = project(endSec);
    const start = Math.max(visibleStart, toFrame(Math.min(a, b), framePeriodMs));
    const end = Math.min(visibleEnd, toFrame(Math.max(a, b), framePeriodMs));
    const interval = positiveIntervalToFrames(start, end, visibleEnd);
    if (interval === null) return [];
    const [cutStart, cutEnd] = cutFlagsForIntersection(
        [part.span.sourceStartSec, part.span.sourceEndSec],
        [windowStart, windowEnd],
        projection.reversed,
    );
    return [
        {
            cycle: 0,
            startFrame: interval[0],
            endFrameExclusive: interval[1],
            cutStart: part.cutStart || cutStart,
            cutEnd: part.cutEnd || cutEnd,
        },
    ];
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
        /*
         * 【不能先 clamp 再取模】旧实现把 anchor + phase 直接截到媒体末尾，导致
         * anchor 之前的回绕成员被映射到媒体尾部。正确顺序是：先得到未归一化的
         * 源坐标，再在媒体域做一次归一化（取模），最后才允许等于 mediaDuration
         * 的端点保持原值。
         */
        source = projection.reversed ? anchor - phase : anchor + phase;
        if (projection.mediaDurationSec !== null) {
            const duration = projection.mediaDurationSec;
            if (source < -EPS || source > duration + EPS) {
                source = modEuclid(source, duration);
            } else if (Math.abs(source - duration) <= EPS) {
                source = duration;
            }
        }
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
 * 工程帧 → 源域秒的**局部仿射**映射（不取模、不 clamp）。
 *
 * `cycle` 必须是该投影区间自己的周期号。与 `inverseProject` 的区别：
 * - 不把 anchor + phase 再截到媒体末尾（Loop 回绕会因此跳到媒体尾部）；
 * - 不在周期边界把 endFrameExclusive 折回 anchor（否则“结束于边界”变成“开始”）；
 * - 倒放时直接按 `anchor - offset` 计算，方向由成员自己的 frameSource* 表达。
 */
export function sourceAtProjectedFrame(
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
    if (projection.loopEnabled && projection.loopCycleSec !== null) {
        const cycleSec = projection.loopCycleSec;
        const cycleTimeline = cycleSec / rate;
        const offsetSec = localSec - cycle * cycleTimeline;
        const anchor = loopAnchor(projection);
        let source = projection.reversed ? anchor - offsetSec * rate : anchor + offsetSec * rate;
        /*
         * Loop 的源域是 [0, mediaDuration) 的环。anchor 之前的成员在局部仿射
         * 映射里会得到 anchor+phase（大于媒体末尾），必须归一化回环内；否则
         * R1/R3 仍会读到媒体末尾。端点恰好等于 mediaDuration 时保留原值，
         * 因为那是“独占结束”而不是回绕到 0。
         */
        const media = projection.mediaDurationSec;
        if (media !== null && media > EPS) {
            if (source < -EPS || source > media + EPS) {
                source = modEuclid(source, media);
            } else if (Math.abs(source - media) <= EPS) {
                source = media;
            }
        }
        return source;
    }
    return projection.reversed
        ? projection.sourceWindowEndSec - localSec * rate
        : projection.sourceWindowStartSec + localSec * rate;
}

/**
 * 把一个可能跨 Loop 周期边界的源域区间切成“每段都属于单一周期”的成员。
 *
 * 这是 Loop 拆分的唯一实现：normalize 只负责持久化，projectTakeSpan 自己
 * 也要调用它，因此未 normalize 的 span 同样能正确跨 anchor。
 */
export function normalizeLoopTakeSpans(projection: ClipProjection, span: TakeSpan): TakeSpan[] {
    return splitLoopTakeSpanParts(projection, span).map((part) => part.span);
}

interface MutableLoopPart<T extends ProjectableTakeSpan> {
    span: T;
    cutStart: boolean;
    cutEnd: boolean;
}

function markLowEdgeCut<T extends ProjectableTakeSpan>(
    part: MutableLoopPart<T>,
    reversed: boolean,
): void {
    if (reversed) part.cutEnd = true;
    else part.cutStart = true;
}

function markHighEdgeCut<T extends ProjectableTakeSpan>(
    part: MutableLoopPart<T>,
    reversed: boolean,
): void {
    if (reversed) part.cutStart = true;
    else part.cutEnd = true;
}

/**
 * Split any TakeSpan into fragments that each live inside one Loop cycle.
 *
 * This handles the anchor seam (head and tail belong to the same cycle), media
 * wrap-around, and pathological spans wider than one cycle. cutStart/cutEnd
 * are timeline-oriented and therefore swap source edges under reverse playback.
 */
export function splitLoopTakeSpanParts<T extends ProjectableTakeSpan>(
    projection: ClipProjection,
    span: T,
): LoopSpanPart<T>[] {
    if (!projection.loopEnabled || projection.loopCycleSec === null) {
        return [{ span, cutStart: false, cutEnd: false }];
    }
    if (!Number.isFinite(span.sourceStartSec) || !Number.isFinite(span.sourceEndSec)) {
        return [];
    }
    const cycleSec = projection.loopCycleSec;
    const width = span.sourceEndSec - span.sourceStartSec;
    if (!(width > SOURCE_EPS)) return [];
    const media =
        projection.mediaDurationSec !== null && projection.mediaDurationSec > EPS
            ? projection.mediaDurationSec
            : cycleSec;
    const anchor = loopAnchor(projection);
    const out: LoopSpanPart<T>[] = [];
    if (width > cycleSec + EPS) {
        let start = span.sourceStartSec;
        let cycle = span.cycle;
        let index = 0;
        while (start < span.sourceEndSec - SOURCE_EPS && index < 20_000) {
            const end = Math.min(start + cycleSec, span.sourceEndSec);
            const last = end >= span.sourceEndSec - SOURCE_EPS;
            const [cutStart, cutEnd] = projection.reversed
                ? [!last, index > 0]
                : [index > 0, !last];
            const chunk = {
                ...span,
                sourceStartSec: start,
                sourceEndSec: end,
                cycle,
            } as T;
            out.push(
                ...splitLoopPiece(projection, { span: chunk, cutStart, cutEnd }, media, anchor),
            );
            start = end;
            cycle += 1;
            index += 1;
        }
    } else {
        out.push(
            ...splitLoopPiece(projection, { span, cutStart: false, cutEnd: false }, media, anchor),
        );
    }
    out.sort(
        (a, b) =>
            a.span.sourceStartSec - b.span.sourceStartSec ||
            a.span.sourceEndSec - b.span.sourceEndSec ||
            a.span.cycle - b.span.cycle,
    );
    return out;
}

function splitLoopPiece<T extends ProjectableTakeSpan>(
    projection: ClipProjection,
    part: MutableLoopPart<T>,
    media: number,
    anchor: number,
): LoopSpanPart<T>[] {
    const reversed = projection.reversed;
    const queue: MutableLoopPart<T>[] = [part];
    const wrapped: MutableLoopPart<T>[] = [];
    let guard = 0;
    while (queue.length > 0) {
        guard += 1;
        if (guard > 20_000) break;
        const current = queue.pop()!;
        const lo = current.span.sourceStartSec;
        const hi = current.span.sourceEndSec;
        if (hi <= lo + SOURCE_EPS) continue;
        if (lo < -EPS) {
            if (hi <= EPS) {
                queue.push({
                    ...current,
                    span: {
                        ...current.span,
                        sourceStartSec: media + lo,
                        sourceEndSec: media + hi,
                    } as T,
                });
            } else {
                const tail: MutableLoopPart<T> = {
                    ...current,
                    span: { ...current.span, sourceStartSec: media + lo, sourceEndSec: media } as T,
                };
                markHighEdgeCut(tail, reversed);
                const head: MutableLoopPart<T> = {
                    ...current,
                    span: { ...current.span, sourceStartSec: 0, sourceEndSec: hi } as T,
                };
                markLowEdgeCut(head, reversed);
                queue.push(tail, head);
            }
        } else if (lo >= media - EPS) {
            queue.push({
                ...current,
                span: {
                    ...current.span,
                    sourceStartSec: lo - media,
                    sourceEndSec: hi - media,
                } as T,
            });
        } else if (hi > media + EPS) {
            const tail: MutableLoopPart<T> = {
                ...current,
                span: { ...current.span, sourceEndSec: media } as T,
            };
            markHighEdgeCut(tail, reversed);
            const head: MutableLoopPart<T> = {
                ...current,
                span: { ...current.span, sourceStartSec: 0, sourceEndSec: hi - media } as T,
            };
            markLowEdgeCut(head, reversed);
            queue.push(tail, head);
        } else {
            wrapped.push(current);
        }
    }
    const out: LoopSpanPart<T>[] = [];
    for (const piece of wrapped) {
        const lo = piece.span.sourceStartSec;
        const hi = piece.span.sourceEndSec;
        if (lo < anchor - EPS && hi > anchor + EPS) {
            // Forward: [lo, anchor] is the tail at the end of this very cycle,
            // [anchor, hi] is the head at its start. Reverse mirrors the flags
            // but keeps both fragments in the same cycle.
            const left: MutableLoopPart<T> = {
                ...piece,
                span: { ...piece.span, sourceEndSec: anchor } as T,
            };
            markHighEdgeCut(left, reversed);
            const right: MutableLoopPart<T> = {
                ...piece,
                span: { ...piece.span, sourceStartSec: anchor } as T,
            };
            markLowEdgeCut(right, reversed);
            out.push(left, right);
        } else {
            out.push(piece);
        }
    }
    return out;
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
        clipPlaybackRate: clip.clipPlaybackRate,
        reversed: clip.reversed,
        loopEnabled: clip.loopEnabled,
        loopPeriodSec: clip.loopPeriodSec ?? null,
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
            syllableNotes: take.syllableNotes,
            syllableEventIds: take.syllableEventIds,
            syllableAnalysisWindowSec: take.syllableAnalysisWindowSec,
            gameBlockBaseline: take.gameBlockBaseline,
            midiNoteData: take.midiNoteData,
        })),
        syllableSpansSec: clip.syllableSpansSec,
        syllableNotes: clip.syllableNotes,
        syllableEventIds: clip.syllableEventIds,
        syllableAnalysisWindowSec: clip.syllableAnalysisWindowSec,
        gameBlockBaseline: clip.gameBlockBaseline,
        midiNoteData: clip.midiNoteData,
    };
}
