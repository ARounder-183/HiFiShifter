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
    clearSnapHighlights,
    publishSingleTargetSnapHighlight,
} from "../../../utils/snapHighlight";
import type { ScaleLike } from "../../../utils/musicalScales";
import { getDrawPreviewValue } from "./paramValuePreviewLogic";

export const PARAM_BLOCK_SNAP_HIGHLIGHT_GROUP = "paramBlock";

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
    });
}

export function clearParamBlockSnapHighlight(): void {
    clearSnapHighlights(PARAM_BLOCK_SNAP_HIGHLIGHT_GROUP);
}
