/**
 * 时间轴吸附引擎（对标 REAPER Snap/Grid Settings 的行为矩阵）。
 *
 * 设计目标：
 * - 网格吸附与网格显示使用同一套 Swing/间距规则，保证“看到哪里、吸到哪里”。
 * - 候选目标可组合：网格、媒体项（Clip）边缘、选区边缘、播放光标、采样率边界。
 * - 所有候选按像素距离选择最近者；`snapToGridAnyDistance` 会让网格无条件胜出。
 * - `snapRelativeToGrid` 保留拖动起点相对网格的偏移（REAPER 语义）。
 */
import type {
    ClipInfo,
    GridSize,
    TimelineSnapSettings,
} from "../features/session/sessionTypes";
import type { TrackInfo } from "../features/session/sessionTypes";
import type { TempoMap } from "./tempoMap";
import {
    pointIndexAtSec,
    snapSecToTempoGrid,
} from "./tempoMap";
import { gridStepBeats } from "../components/layout/timeline/grid";

export type SnapObjectKind = "mediaItem" | "selection" | "cursor";
export type SnapCandidateKind =
    | "grid"
    | "mediaStart"
    | "mediaEnd"
    | "snapOffset"
    | "sourceStart"
    | "sourceEnd"
    | "selection"
    | "cursor"
    | "sampleRate";

export interface SnapCandidate {
    sec: number;
    kind: SnapCandidateKind;
    /** 候选优先级，数值越小越优先（tie-break 用）。 */
    priority: number;
    clipId?: string;
    trackId?: string;
}

export interface TimelineSnapContext {
    settings: TimelineSnapSettings;
    /** 当前显示网格（工程网格设置）。 */
    grid: GridSize;
    bpm: number;
    beatsPerBar: number;
    tempoMap: TempoMap | null;
    pxPerSec: number;
    clips: readonly ClipInfo[];
    tracks: readonly TrackInfo[];
    selectedClipIds: readonly string[];
    playheadSec: number;
    /** 正在拖动的对象类型。 */
    object: SnapObjectKind;
    /** 拖动起点（用于 snapRelativeToGrid 与 selection 倍数）。 */
    originSec?: number;
    /** 拖动锚点所在轨道。 */
    anchorTrackId?: string | null;
    /** 从候选集中排除的 clip id（通常是正在移动的 clip 自身）。 */
    excludeClipIds?: ReadonlySet<string>;
    /** 工程采样率；snapToProjectSampleRate 时使用，默认 48000。 */
    projectSampleRate?: number;
}

export interface SnapResult {
    sec: number;
    candidate: SnapCandidate | null;
    distancePx: number;
    snapped: boolean;
}

export const DEFAULT_PROJECT_SAMPLE_RATE = 48000;

function clampSec(sec: number): number {
    return Number.isFinite(sec) ? Math.max(0, sec) : 0;
}

function snapSpacingGrid(settings: TimelineSnapSettings, grid: GridSize): GridSize {
    return settings.useIndependentSnapSpacing ? settings.snapSpacing : grid;
}

/** 当前生效的网格吸附步长（拍）。 */
export function snapStepBeats(settings: TimelineSnapSettings, grid: GridSize): number {
    return Math.max(1e-9, gridStepBeats(snapSpacingGrid(settings, grid)));
}

/** 考虑当前缩放后的最小像素间距，返回最终吸附步长（拍）。 */
export function snapStepBeatsForZoom(
    settings: TimelineSnapSettings,
    grid: GridSize,
    bpm: number,
    pxPerSec: number,
    tempoMap: TempoMap | null = null,
    rawSec = 0,
): number {
    let step = snapStepBeats(settings, grid);
    if (!settings.useIndependentSnapSpacing) return step;
    const minPx = Math.max(2, settings.snapSpacingMinPx);
    const effectiveBpm =
        tempoMap && tempoMap.points.length > 0
            ? Math.max(1, tempoMap.points[pointIndexAtSec(tempoMap, rawSec)].bpm)
            : Math.max(1, bpm);
    const secPerBeat = 60 / effectiveBpm;
    let guard = 0;
    while (step * secPerBeat * Math.max(1e-9, pxPerSec) < minPx - 1e-9 && guard < 16) {
        step *= 2;
        guard += 1;
    }
    return step;
}

function swingOffsetSec(bpm: number, stepBeats: number, swingPercent: number): number {
    if (!(swingPercent > 0)) return 0;
    return (swingPercent / 100) * 0.5 * stepBeats * (60 / Math.max(1, bpm));
}

/**
 * 生成 rawSec 附近的 Swing 网格候选（偶数拍在格上，奇数拍按 Swing 偏移）。
 * 返回升序且去重的候选位置。
 */
function swingGridCandidatesAround(
    rawSec: number,
    tempoMap: TempoMap | null,
    stepBeats: number,
    fallbackBpm: number,
    swingPercent: number,
): number[] {
    const safeRaw = clampSec(rawSec);
    const safeStep = Math.max(1e-9, stepBeats);
    const out: number[] = [];

    if (!tempoMap || tempoMap.points.length === 0) {
        const bpm = Math.max(1, fallbackBpm || 120);
        const stepSec = (safeStep * 60) / bpm;
        const center = Math.round(safeRaw / stepSec);
        const swing = swingOffsetSec(bpm, safeStep, swingPercent);
        for (let idx = center - 3; idx <= center + 3; idx += 1) {
            if (idx < 0) continue;
            const sec = idx * stepSec + (idx % 2 === 0 ? 0 : swing);
            out.push(Math.max(0, sec));
        }
    } else {
        const idx = pointIndexAtSec(tempoMap, safeRaw);
        const currentPoint = tempoMap.points[idx];
        const currentBpm = Math.max(1, currentPoint.bpm);
        const currentStepSec = (safeStep * 60) / currentBpm;
        const currentSegStart = currentPoint.positionSec;
        // 接近变化点边界时把上一段最后几个候选也纳入，避免边界处漏掉更近的网格线。
        const segmentIndices =
            idx > 0 && safeRaw - currentSegStart < currentStepSec * 3
                ? [idx - 1, idx]
                : [idx];
        for (const segmentIndex of segmentIndices) {
            const point = tempoMap.points[segmentIndex];
            const bpm = Math.max(1, point.bpm);
            const stepSec = (safeStep * 60) / bpm;
            const swing = swingOffsetSec(bpm, safeStep, swingPercent);
            const segStart = point.positionSec;
            const nextSec =
                segmentIndex + 1 < tempoMap.points.length
                    ? tempoMap.points[segmentIndex + 1].positionSec
                    : Number.POSITIVE_INFINITY;
            const localStep = Math.round((safeRaw - segStart) / stepSec);
            for (let k = localStep - 3; k <= localStep + 3; k += 1) {
                if (k < 0) continue;
                const sec = segStart + k * stepSec + (k % 2 === 0 ? 0 : swing);
                if (sec > nextSec + 1e-6) break;
                out.push(Math.max(0, sec));
            }
            // 段起点与下一段起点永远都是合法候选。
            out.push(segStart);
            if (Number.isFinite(nextSec)) out.push(nextSec);
        }
    }

    out.sort((a, b) => a - b);
    const deduped: number[] = [];
    for (const sec of out) {
        if (deduped.length === 0 || Math.abs(deduped[deduped.length - 1] - sec) > 1e-9) {
            deduped.push(sec);
        }
    }
    return deduped;
}

/** 吸附到（可选 Swing）网格；关闭 Swing 时等价于原有网格吸附。 */
export function snapToConfiguredGrid(
    rawSec: number,
    tempoMap: TempoMap | null,
    stepBeats: number,
    fallbackBpm: number,
    settings: Pick<TimelineSnapSettings, "swingEnabled" | "swingPercent">,
): number {
    if (!settings.swingEnabled || settings.swingPercent <= 0) {
        return snapSecToTempoGrid(rawSec, tempoMap, stepBeats, fallbackBpm);
    }
    const candidates = swingGridCandidatesAround(
        rawSec,
        tempoMap,
        stepBeats,
        fallbackBpm,
        settings.swingPercent,
    );
    let best = candidates[0] ?? Math.max(0, rawSec);
    let bestDelta = Math.abs(best - rawSec);
    for (const sec of candidates) {
        const delta = Math.abs(sec - rawSec);
        if (delta < bestDelta - 1e-9) {
            best = sec;
            bestDelta = delta;
        }
    }
    return best;
}

/**
 * 网格候选（含 Swing）。`rawSec` 附近 ±3 个周期。
 */
function collectGridCandidates(ctx: TimelineSnapContext, rawSec: number): SnapCandidate[] {
    const { settings } = ctx;
    if (!settings.gridVisible && settings.gridSnapFollowsGridVisibility) return [];
    const step = snapStepBeatsForZoom(
        settings,
        ctx.grid,
        ctx.bpm,
        ctx.pxPerSec,
        ctx.tempoMap,
        rawSec,
    );
    const positions = swingGridCandidatesAround(
        rawSec,
        ctx.tempoMap,
        step,
        ctx.bpm,
        settings.swingEnabled ? settings.swingPercent : 0,
    );
    const originOffset = settings.snapRelativeToGrid
        ? ctx.originSec != null
            ? ctx.originSec -
              snapToConfiguredGrid(ctx.originSec, ctx.tempoMap, step, ctx.bpm, settings)
            : 0
        : 0;
    const out: SnapCandidate[] = [];
    for (const sec of positions) {
        const shifted = clampSec(sec + originOffset);
        if (out.some((c) => Math.abs(c.sec - shifted) < 1e-9)) continue;
        out.push({ sec: shifted, kind: "grid", priority: 10 });
    }
    return out;
}

/** clip 内容起点（近似 REAPER snap offset）。 */
export function clipSnapOffsetSec(clip: ClipInfo): number {
    const rate = Math.max(1e-6, Number(clip.playbackRate) || 1);
    const offset = Math.max(0, Number(clip.sourceStartSec) || 0) / rate;
    return clampSec(Number(clip.startSec) + offset);
}

function clipTrackDistance(ctx: TimelineSnapContext, clip: ClipInfo): number {
    if (ctx.anchorTrackId == null) return 0;
    const indexOf = (id: string) => ctx.tracks.findIndex((t) => t.id === id);
    const a = indexOf(ctx.anchorTrackId);
    const b = indexOf(clip.trackId);
    if (a < 0 || b < 0) return Number.POSITIVE_INFINITY;
    return Math.abs(a - b);
}

function addClipCandidates(
    ctx: TimelineSnapContext,
    out: SnapCandidate[],
    opts: {
        includeSelection?: boolean;
        excludeSelected?: boolean;
    },
) {
    const { settings } = ctx;
    const excluded = ctx.excludeClipIds ?? new Set<string>();
    const selectedSet = new Set(ctx.selectedClipIds);
    for (const clip of ctx.clips) {
        if (excluded.has(clip.id)) continue;
        if (!settings.snapAcrossTracks && clip.trackId !== (ctx.anchorTrackId ?? clip.trackId)) {
            continue;
        }
        if (settings.snapAcrossTracks) {
            const distance = clipTrackDistance(ctx, clip);
            if (distance > settings.snapTrackDistance) continue;
        }
        const isSelected = selectedSet.has(clip.id);
        if (opts.excludeSelected && isSelected) continue;
        if (opts.includeSelection && !isSelected) {
            out.push({ sec: clampSec(clip.startSec), kind: "selection", priority: 30, clipId: clip.id, trackId: clip.trackId });
        }
        if (settings.snapItemStart) {
            out.push({ sec: clampSec(clip.startSec), kind: "mediaStart", priority: 20, clipId: clip.id, trackId: clip.trackId });
            out.push({
                sec: clampSec(clip.startSec + Math.max(0, clip.lengthSec)),
                kind: "mediaEnd",
                priority: 21,
                clipId: clip.id,
                trackId: clip.trackId,
            });
        }
        if (settings.snapItemSnapOffset) {
            out.push({
                sec: clipSnapOffsetSec(clip),
                kind: "snapOffset",
                priority: 22,
                clipId: clip.id,
                trackId: clip.trackId,
            });
        }
        if (settings.snapMediaEdgesToSource) {
            const rate = Math.max(1e-6, Number(clip.playbackRate) || 1);
            const sourceStartSec = Number(clip.sourceStartSec) || 0;
            const sourceEndSec = Number(clip.sourceEndSec);
            const durationSec = Number(clip.durationSec);
            const sourceDuration = Number.isFinite(durationSec) && durationSec != null && durationSec > 0
                ? durationSec
                : Number.isFinite(sourceEndSec) && sourceEndSec > sourceStartSec
                  ? sourceEndSec - sourceStartSec
                  : Number(clip.lengthSec) || 0;
            out.push({
                sec: clampSec(Number(clip.startSec) - sourceStartSec / rate),
                kind: "sourceStart",
                priority: 24,
                clipId: clip.id,
                trackId: clip.trackId,
            });
            out.push({
                sec: clampSec(Number(clip.startSec) + (sourceDuration - sourceStartSec) / rate),
                kind: "sourceEnd",
                priority: 24,
                clipId: clip.id,
                trackId: clip.trackId,
            });
        }
    }
}

function collectSampleRateCandidates(ctx: TimelineSnapContext, rawSec: number): SnapCandidate[] {
    if (!ctx.settings.snapToProjectSampleRate) return [];
    const rate = Math.max(1, ctx.projectSampleRate ?? DEFAULT_PROJECT_SAMPLE_RATE);
    const step = 1 / rate;
    const lo = Math.floor(rawSec / step) * step;
    const hi = Math.ceil(rawSec / step) * step;
    return [
        { sec: clampSec(lo), kind: "sampleRate", priority: 40 },
        { sec: clampSec(hi), kind: "sampleRate", priority: 40 },
    ];
}

/**
 * 计算 rawSec 的吸附结果。
 */
export function snapTimelinePosition(ctx: TimelineSnapContext, rawSec: number): SnapResult {
    const settings = ctx.settings;
    const safeRaw = clampSec(rawSec);
    if (!settings.enabled) {
        return { sec: safeRaw, candidate: null, distancePx: 0, snapped: false };
    }

    const candidates: SnapCandidate[] = [];

    // ── 网格候选 ──
    const wantsGrid =
        (ctx.object === "mediaItem" && settings.snapMediaItemsToGrid) ||
        (ctx.object === "selection" && settings.snapSelectionToGrid) ||
        (ctx.object === "cursor" && settings.snapCursorToGrid);
    const gridCandidates = wantsGrid ? collectGridCandidates(ctx, safeRaw) : [];
    candidates.push(...gridCandidates);

    // ── 选择 / 标记 / 光标 ──
    const wantsSelMarkerCursor =
        (ctx.object === "mediaItem" && settings.snapMediaItemsToSelectionMarkersCursor) ||
        (ctx.object === "selection" && settings.snapSelectionToSelectionMarkersCursor) ||
        (ctx.object === "cursor" && settings.snapCursorToSelectionMarkersCursor);
    if (wantsSelMarkerCursor) {
        addClipCandidates(ctx, candidates, {
            includeSelection: ctx.object !== "selection",
            excludeSelected: ctx.object === "selection",
        });
        if (ctx.object !== "cursor") {
            candidates.push({ sec: clampSec(ctx.playheadSec), kind: "cursor", priority: 5 });
        }
    }

    candidates.push(...collectSampleRateCandidates(ctx, safeRaw));

    if (ctx.object === "selection" && settings.forceSelectionsToMultiples) {
        const multiple = gridStepBeats(settings.selectionMultiple);
        const origin = ctx.originSec ?? 0;
        const originGrid = snapToConfiguredGrid(origin, ctx.tempoMap, multiple, ctx.bpm, {
            swingEnabled: false,
            swingPercent: 0,
        });
        const offset = origin - originGrid;
        const snapped =
            snapToConfiguredGrid(safeRaw - offset, ctx.tempoMap, multiple, ctx.bpm, {
                swingEnabled: false,
                swingPercent: 0,
            }) + offset;
        candidates.push({ sec: clampSec(snapped), kind: "grid", priority: 15 });
    }

    if (candidates.length === 0) {
        return { sec: safeRaw, candidate: null, distancePx: 0, snapped: false };
    }

    // 激进模式：任意距离吸附网格。
    if (settings.snapToGridAnyDistance && gridCandidates.length > 0) {
        let best = gridCandidates[0];
        for (const candidate of gridCandidates) {
            if (Math.abs(candidate.sec - safeRaw) < Math.abs(best.sec - safeRaw) - 1e-9) {
                best = candidate;
            }
        }
        return {
            sec: best.sec,
            candidate: best,
            distancePx: Math.abs(best.sec - safeRaw) * Math.max(1e-9, ctx.pxPerSec),
            snapped: true,
        };
    }

    const thresholdSec = settings.snapDistancePx / Math.max(1e-9, ctx.pxPerSec);
    let best: SnapCandidate | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const candidate of candidates) {
        const distance = Math.abs(candidate.sec - safeRaw);
        if (distance > thresholdSec + 1e-12) continue;
        if (distance < bestDistance - 1e-12 || (Math.abs(distance - bestDistance) <= 1e-12 && candidate.priority < (best?.priority ?? Infinity))) {
            best = candidate;
            bestDistance = distance;
        }
    }

    if (!best) {
        return { sec: safeRaw, candidate: null, distancePx: thresholdSec * Math.max(1e-9, ctx.pxPerSec), snapped: false };
    }

    return {
        sec: best.sec,
        candidate: best,
        distancePx: bestDistance * Math.max(1e-9, ctx.pxPerSec),
        snapped: true,
    };
}

/**
 * 将全部 clips 对齐到当前 Swing 网格（“Adjust all items when changing swing”）。
 * 返回 clipId → startSec 更新表。
 */
export function alignClipsToSwingGrid(args: {
    clips: readonly ClipInfo[];
    settings: TimelineSnapSettings;
    grid: GridSize;
    tempoMap: TempoMap | null;
    bpm: number;
}): Record<string, number> {
    const step = snapStepBeats(args.settings, args.grid);
    const updates: Record<string, number> = {};
    for (const clip of args.clips) {
        const next = snapToConfiguredGrid(clip.startSec, args.tempoMap, step, args.bpm, args.settings);
        if (Math.abs(next - clip.startSec) > 1e-9) {
            updates[clip.id] = next;
        }
    }
    return updates;
}
