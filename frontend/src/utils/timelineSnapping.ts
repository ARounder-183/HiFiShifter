/**
 * 时间轴吸附引擎（对标 REAPER Snap/Grid Settings 的行为矩阵）。
 *
 * 设计目标：
 * - 吸附规则与网格显示使用同一套 Swing/间距规则，保证“看到哪里、吸到哪里”。
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
import {
    modEuclid,
    resolveClipContentDurationSec,
    resolveLeadingSilenceSec,
    resolvePlaybackWindowSec,
} from "./loopRender";

export type SnapObjectKind = "clip" | "selection" | "cursor";
export type SnapCandidateKind =
    | "grid"
    | "clipStart"
    | "clipEnd"
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

/** 当前生效的吸附步长（拍）。 */
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

/** 吸附到（可选 Swing）网格；关闭 Swing 时等价于普通网格对齐。 */
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
    if (!settings.gridVisible && settings.snapFollowsGridVisibility) return [];
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

/**
 * clip 内容起点（首个可听采样）的投影（近似 REAPER snap offset）。
 *
 * 与后端 clip_leading_silence_sec / 前端 resolveLeadingSilenceSec 同一模型：
 * 正放看窗口起点越过媒体起点、倒放看窗口终点越过媒体末端 —— 越过部分为
 * 静音，内容真正开始于前导静音之后。Loop 的负锚点是环绕相位（无静音），
 * 恒返回 clip 起点。
 */
export function clipSnapOffsetSec(clip: ClipInfo): number {
    const rate = Number(clip.playbackRate);
    const safeRate = Number.isFinite(rate) && rate > 1e-6 ? rate : 1;
    // 内容时长与后端 clip_source_media_duration_sec 同一取值链
    //（frames/采样率 → durationSec → 音符内容最大结束）。
    const contentDurSec = resolveClipContentDurationSec({
        sourcePath: clip.sourcePath,
        midiNoteData: clip.midiNoteData ?? null,
        durationFrames: clip.durationFrames,
        sourceSampleRate: clip.sourceSampleRate,
        durationSec: clip.durationSec,
    });
    const leadingSilenceSec = resolveLeadingSilenceSec(
        {
            loopEnabled: Boolean(clip.loopEnabled),
            reversed: Boolean(clip.reversed),
            sourceStartSec: Number(clip.sourceStartSec) || 0,
            playbackRate: safeRate,
            lengthSec: Math.max(0, Number(clip.lengthSec) || 0),
            sourceEndSec: Number(clip.sourceEndSec) || 0,
        },
        contentDurSec,
    );
    return clampSec(Number(clip.startSec) + leadingSilenceSec);
}

function clipTrackDistance(ctx: TimelineSnapContext, clip: ClipInfo): number {
    if (ctx.anchorTrackId == null) return 0;
    const indexOf = (id: string) => ctx.tracks.findIndex((t) => t.id === id);
    const a = indexOf(ctx.anchorTrackId);
    const b = indexOf(clip.trackId);
    if (a < 0 || b < 0) return Number.POSITIVE_INFINITY;
    return Math.abs(a - b);
}

/** 轨道距离过滤后的可见 clip 列表（供各候选组共用）。 */
function visibleClipsForSnap(ctx: TimelineSnapContext): ClipInfo[] {
    const excluded = ctx.excludeClipIds ?? new Set<string>();
    const out: ClipInfo[] = [];
    for (const clip of ctx.clips) {
        if (excluded.has(clip.id)) continue;
        if (!settings_gateAcrossTracks(ctx, clip)) continue;
        out.push(clip);
    }
    return out;
}

function settings_gateAcrossTracks(ctx: TimelineSnapContext, clip: ClipInfo): boolean {
    if (ctx.settings.snapAcrossTracks) {
        const distance = clipTrackDistance(ctx, clip);
        return distance <= ctx.settings.snapTrackDistance;
    }
    return clip.trackId === (ctx.anchorTrackId ?? clip.trackId);
}

/** "吸附到选择/标记/光标"族：其他（或已选）Clip 的起点候选。 */
function addSelectionCandidates(
    ctx: TimelineSnapContext,
    out: SnapCandidate[],
    opts: { includeSelection?: boolean; excludeSelected?: boolean },
) {
    const selectedSet = new Set(ctx.selectedClipIds);
    for (const clip of visibleClipsForSnap(ctx)) {
        const isSelected = selectedSet.has(clip.id);
        if (opts.excludeSelected && isSelected) continue;
        if (opts.includeSelection && !isSelected) {
            out.push({ sec: clampSec(clip.startSec), kind: "selection", priority: 30, clipId: clip.id, trackId: clip.trackId });
        }
    }
}

/**
 * "Clip 边缘 / 内容起点 / 源素材首尾"三组独立目标候选。
 *
 * 源素材首尾的投影按**消费方向**取窗口模型（与 loopRender /
 * WaveformTrackCanvas 的边界标记同一套公式）：
 *   正放 t(source=b) = startSec + (b − winStart) / rate
 *   倒放 t(source=b) = startSec + (winEnd  − b) / rate
 * Loop Clip 的媒体边界呈 mod-D 等差回绕族 —— 取 clip 内的前两个回绕点。
 * 投影落在 clip 可见范围之外的候选不生成（避免幻影目标）。
 */
function addClipEdgeCandidates(ctx: TimelineSnapContext, out: SnapCandidate[]) {
    const { settings } = ctx;
    for (const clip of visibleClipsForSnap(ctx)) {
        if (settings.snapClipEdges) {
            out.push({ sec: clampSec(clip.startSec), kind: "clipStart", priority: 20, clipId: clip.id, trackId: clip.trackId });
            out.push({
                sec: clampSec(clip.startSec + Math.max(0, clip.lengthSec)),
                kind: "clipEnd",
                priority: 21,
                clipId: clip.id,
                trackId: clip.trackId,
            });
        }
        if (settings.snapClipSnapOffset) {
            out.push({
                sec: clipSnapOffsetSec(clip),
                kind: "snapOffset",
                priority: 22,
                clipId: clip.id,
                trackId: clip.trackId,
            });
        }
        if (!settings.snapClipsToSourceMedia) continue;

        const rateRaw = Number(clip.playbackRate);
        const rate = Number.isFinite(rateRaw) && rateRaw > 1e-6 ? rateRaw : 1;
        const startSec = Number(clip.startSec) || 0;
        const lengthSec = Math.max(0, Number(clip.lengthSec) || 0);
        const windowArgs = {
            loopEnabled: Boolean(clip.loopEnabled),
            reversed: Boolean(clip.reversed),
            sourceStartSec: Number(clip.sourceStartSec) || 0,
            playbackRate: rate,
            lengthSec,
            sourceEndSec: Number(clip.sourceEndSec ?? clip.durationSec ?? 0) || 0,
        };
        const { winStartSec, winEndSec } = resolvePlaybackWindowSec(windowArgs);
        // 内容时长：frames/采样率精确链优先，durationSec 兜底，纯 MIDI
        // 回退音符内容范围（与后端 clip_source_media_duration_sec 一致）。
        // 未知时正放/倒放只保留 s=0 候选，Loop 无法确定周期则整体跳过。
        const mediaDur = resolveClipContentDurationSec({
            sourcePath: clip.sourcePath,
            midiNoteData: clip.midiNoteData ?? null,
            durationFrames: clip.durationFrames,
            sourceSampleRate: clip.sourceSampleRate,
            durationSec: clip.durationSec,
        });

        /** 把源坐标 b 投影为时间线秒；仅保留落在 clip 范围内的目标。 */
        const pushProjected = (b: number, kind: SnapCandidateKind) => {
            const sec = clip.reversed
                ? startSec + (winEndSec - b) / rate
                : startSec + (b - winStartSec) / rate;
            if (!Number.isFinite(sec)) return;
            if (sec < startSec - 1e-6 || sec > startSec + lengthSec + 1e-6) return;
            out.push({ sec: clampSec(sec), kind, priority: 24, clipId: clip.id, trackId: clip.trackId });
        };

        if (!clip.loopEnabled) {
            pushProjected(0, "sourceStart");
            if (mediaDur != null && mediaDur > 1e-9) pushProjected(mediaDur, "sourceEnd");
        } else {
            if (mediaDur == null || !(mediaDur > 1e-9)) continue;
            // 回绕点相位：正放锚点 ss（t·r ≡ −ss mod D）、倒放锚点
            // min(se, D)（t·r ≡ +φ mod D，与引擎/波形标记的锚点约定一致）；
            // b∈{0,D} 两边界投影到同一 mod-D 相位族。
            const anchorSrc = clip.reversed ? Math.min(winEndSec, mediaDur) : winStartSec;
            const firstWrapLocal =
                (clip.reversed
                    ? modEuclid(anchorSrc, mediaDur)
                    : modEuclid(-anchorSrc, mediaDur)) / rate;
            for (let k = 0; k <= 1; k += 1) {
                const sec = startSec + firstWrapLocal + k * (mediaDur / rate);
                if (sec < startSec - 1e-6 || sec > startSec + lengthSec + 1e-6) continue;
                out.push({ sec, kind: k === 0 ? "sourceStart" : "sourceEnd", priority: 24, clipId: clip.id, trackId: clip.trackId });
            }
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
        (ctx.object === "clip" && settings.snapClipsToGrid) ||
        (ctx.object === "selection" && settings.snapSelectionToGrid) ||
        (ctx.object === "cursor" && settings.snapCursorToGrid);
    const gridCandidates = wantsGrid ? collectGridCandidates(ctx, safeRaw) : [];
    candidates.push(...gridCandidates);

    // ── 选择 / 标记 / 光标 ──
    const wantsSelMarkerCursor =
        (ctx.object === "clip" && settings.snapClipsToSelectionMarkersCursor) ||
        (ctx.object === "selection" && settings.snapSelectionToSelectionMarkersCursor) ||
        (ctx.object === "cursor" && settings.snapCursorToSelectionMarkersCursor);
    if (wantsSelMarkerCursor) {
        addSelectionCandidates(ctx, candidates, {
            includeSelection: ctx.object !== "selection",
            excludeSelected: ctx.object === "selection",
        });
        if (ctx.object !== "cursor") {
            candidates.push({ sec: clampSec(ctx.playheadSec), kind: "cursor", priority: 5 });
        }
    }

    // ── Clip 边缘 / 内容起点 / 源素材首尾 ──
    // 三组各自独立的目标开关，不隶属于"选择/标记/光标"族：
    // 任意拖动对象在对应开关开启时都可吸附到这些候选。
    addClipEdgeCandidates(ctx, candidates);

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

// ── "拖动时切换吸附"（modifier.clipNoSnap）────────────────────────────

/**
 * "拖动时切换吸附"语义：修饰键按住时把吸附总开关临时取反。
 *   总开关开 + 修饰键 → 不吸附；总开关关 + 修饰键 → 吸附。
 */
export function computeEffectiveSnap(
    snapEnabled: boolean,
    toggleModifierActive: boolean,
): boolean {
    return toggleModifierActive ? !snapEnabled : snapEnabled;
}

// ── 吸附手势登记 ──
// 拖拽期间工具栏"吸附"按钮需要临时视觉切换。通过轻量发布/订阅解耦：
// 各拖拽 hook 在手势起止处登记深度，ActionBar 订阅后计算有效视觉状态。

let snapGestureDepth = 0;
const snapGestureListeners = new Set<() => void>();

function emitSnapGestureChange(): void {
    for (const listener of snapGestureListeners) listener();
}

export function beginSnapGesture(): void {
    snapGestureDepth += 1;
    emitSnapGestureChange();
}

export function endSnapGesture(): void {
    snapGestureDepth = Math.max(0, snapGestureDepth - 1);
    emitSnapGestureChange();
}

/** 当前是否有吸附感知的拖拽手势进行中。 */
export function isSnapGestureActive(): boolean {
    return snapGestureDepth > 0;
}

/** 订阅手势状态变化；返回取消订阅函数。 */
export function subscribeSnapGesture(listener: () => void): () => void {
    snapGestureListeners.add(listener);
    return () => {
        snapGestureListeners.delete(listener);
    };
}
