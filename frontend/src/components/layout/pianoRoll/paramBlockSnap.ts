/**
 * 参数块吸附适配器。
 *
 * 纵向复用参数线绘制使用的 \`getDrawPreviewValue\`（半音 / 音阶 / 容差），
 * 横向复用时间轴 Clip 吸附引擎（网格 / Swing / Tempo Map / Clip 边缘 / 光标）。
 * 两者都遵循 \`modifier.clipNoSnap\` 的 XOR 语义，不再维护第二套默认吸附。
 */
import type {
    ClipInfo,
    GridSize,
    TimelineSnapSettings,
    TrackInfo,
} from "../../../features/session/sessionTypes";
import type { TempoMap } from "../../../utils/tempoMap";
import {
    computeEffectiveSnap,
    snapTimelinePosition,
    type SnapResult,
    type TimelineSnapContext,
} from "../../../utils/timelineSnapping";
import {
    buildValueGuideEntry,
    clearSnapHighlights,
    publishSingleTargetSnapHighlight,
    publishSnapHighlights,
} from "../../../utils/snapHighlight";
import type { ScaleLike } from "../../../utils/musicalScales";
import { getDrawPreviewValue } from "./paramValuePreviewLogic";

export const PARAM_BLOCK_SNAP_HIGHLIGHT_GROUP = "paramBlock";
export const PARAM_BLOCK_SPLIT_SNAP_HIGHLIGHT_GROUP = "paramBlockSplit";
export const PARAM_BLOCK_PITCH_SNAP_HIGHLIGHT_GROUP = "paramBlockPitch";

export interface ParamBlockTimeSnapInput {
    readonly rawSec: number;
    readonly settings: TimelineSnapSettings;
    readonly grid: GridSize;
    readonly bpm: number;
    readonly beatsPerBar: number;
    readonly tempoMap: TempoMap | null;
    readonly pxPerSec: number;
    readonly clips: readonly ClipInfo[];
    readonly tracks: readonly TrackInfo[];
    readonly selectedClipIds: readonly string[];
    readonly playheadSec: number;
    readonly originSec?: number;
    readonly toggleModifierActive: boolean;
}

export function resolveParamBlockTimeSnapResult(input: ParamBlockTimeSnapInput): SnapResult {
    const enabled = computeEffectiveSnap(input.settings.enabled, input.toggleModifierActive);
    if (!enabled) {
        return { sec: input.rawSec, candidate: null, distancePx: 0, snapped: false };
    }
    const ctx: TimelineSnapContext = {
        settings: { ...input.settings, enabled: true },
        grid: input.grid,
        bpm: input.bpm,
        beatsPerBar: input.beatsPerBar,
        tempoMap: input.tempoMap,
        pxPerSec: input.pxPerSec,
        clips: input.clips,
        tracks: input.tracks,
        selectedClipIds: input.selectedClipIds,
        playheadSec: input.playheadSec,
        object: "clip",
        originSec: input.originSec,
    };
    return snapTimelinePosition(ctx, input.rawSec);
}

export function resolveParamBlockTimeSnap(input: ParamBlockTimeSnapInput): number {
    return resolveParamBlockTimeSnapResult(input).sec;
}

export function resolveParamBlockPitchSnapDelta(args: {
    readonly startValue: number;
    readonly rawDelta: number;
    readonly pitchSnapEnabled: boolean;
    readonly pitchSnapUnit: "semitone" | "scale";
    readonly pitchSnapToleranceCents: number;
    readonly projectScale: ScaleLike | null;
    readonly scaleAtSec?: (sec: number) => ScaleLike | null;
    readonly anchorSec: number;
    readonly toggleModifierActive: boolean;
}): number {
    const effective = args.toggleModifierActive ? !args.pitchSnapEnabled : args.pitchSnapEnabled;
    if (!effective) return args.rawDelta;
    const target = getDrawPreviewValue({
        editParam: "pitch",
        rawValue: args.startValue + args.rawDelta,
        effectiveSnap: true,
        pitchSnapUnit: args.pitchSnapUnit,
        projectScale:
            args.pitchSnapUnit === "scale"
                ? (args.scaleAtSec?.(args.anchorSec) ?? args.projectScale ?? undefined)
                : undefined,
        pitchSnapToleranceCents: args.pitchSnapToleranceCents,
    });
    return target - args.startValue;
}

/** 发布横向吸附竖线；参数块没有“另一个对象边缘”，只发布吸附目标。 */
export function publishParamBlockSnapHighlight(result: SnapResult): void {
    publishSingleTargetSnapHighlight({
        group: PARAM_BLOCK_SNAP_HIGHLIGHT_GROUP,
        result,
        id: "param-block",
        surface: "pianoRoll",
    });
}

/** Split uses the same target-only time guide; the line lives until the click settles. */
export function publishParamBlockSplitSnapHighlight(result: SnapResult): void {
    publishSingleTargetSnapHighlight({
        group: PARAM_BLOCK_SPLIT_SNAP_HIGHLIGHT_GROUP,
        result,
        id: "param-block-split",
        surface: "pianoRoll",
    });
}

/**
 * Horizontal pitch guide. Only called when a snapped target actually differs
 * from the raw value; null clears immediately on pointerup/cancel.
 */
export function publishParamBlockPitchSnapHighlight(targetValue: number | null): void {
    if (targetValue === null || !Number.isFinite(targetValue)) {
        clearSnapHighlights(PARAM_BLOCK_PITCH_SNAP_HIGHLIGHT_GROUP);
        return;
    }
    publishSnapHighlights(
        PARAM_BLOCK_PITCH_SNAP_HIGHLIGHT_GROUP,
        [
            buildValueGuideEntry({
                id: "param-block-pitch",
                kind: "grid",
                value: targetValue,
            }),
        ],
        "pianoRoll",
    );
}

export function clearParamBlockSplitSnapHighlight(): void {
    clearSnapHighlights(PARAM_BLOCK_SPLIT_SNAP_HIGHLIGHT_GROUP);
}

/** Clear every parameter-block guide (time and value). */
export function clearParamBlockSnapHighlight(): void {
    clearSnapHighlights(PARAM_BLOCK_SNAP_HIGHLIGHT_GROUP);
    clearSnapHighlights(PARAM_BLOCK_SPLIT_SNAP_HIGHLIGHT_GROUP);
    clearSnapHighlights(PARAM_BLOCK_PITCH_SNAP_HIGHLIGHT_GROUP);
}
