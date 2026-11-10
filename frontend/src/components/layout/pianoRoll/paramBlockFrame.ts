import {
    averageParamValueInSpan,
    modeCenterOfValuesInSamples,
    projectParamBlockDragDisplayValue,
    type ParamBlock,
} from "./paramBlocks";
import type { ParamViewSegment } from "./types";
import { reportParamBlockDiagnostic } from "./paramBlockDiagnostics";

type LiveEdit = {
    readonly key: string;
    readonly edit: readonly number[];
};

/** Block boundaries are shared, but a saved MIDI center belongs only to pitch. */
export function resolveParamBlockPanelValues(args: {
    readonly blocks: readonly ParamBlock[];
    readonly param: string;
    readonly paramView: ParamViewSegment | null;
    readonly fallbackValue: number;
}): ParamBlock[] {
    return args.blocks.map((block) => {
        const manualValue = args.param === "pitch" ? (block.manualValue ?? null) : null;
        const pv = args.paramView;
        const sampled =
            pv === null
                ? null
                : averageParamValueInSpan({
                      startSec: block.startSec,
                      endSec: block.endSec,
                      values: pv.edit,
                      startFrame: pv.startFrame,
                      stride: pv.stride,
                      framePeriodMs: pv.framePeriodMs,
                      isPitch: args.param === "pitch",
                      param: args.param,
                  });
        return {
            ...block,
            manualValue,
            averageValue:
                manualValue ??
                sampled ??
                (pv === null && args.param === "pitch" ? block.averageValue : null) ??
                args.fallbackValue,
        };
    });
}

/**
 * One visual frame can expose several caches for the same parameter window.
 * The block gesture owns the frame while it exists; after it ends, an
 * uncommitted live edit is the next local response; the authoritative window is
 * the final fallback.
 */
export function resolveParamBlockFrameView(args: {
    readonly interaction: ParamViewSegment | null;
    readonly live: LiveEdit | null;
    readonly paramView: ParamViewSegment | null;
    readonly param?: string;
}): ParamViewSegment | null {
    const scope = (key: string) =>
        key.startsWith("v2|") ? key.split("|").slice(0, 3).join("|") : key;
    const interaction = args.interaction;
    if (
        interaction !== null &&
        (!args.param ||
            !interaction.key.startsWith("v2|") ||
            interaction.key.split("|")[2] === args.param) &&
        (!args.paramView || scope(interaction.key) === scope(args.paramView.key))
    )
        return interaction;
    if (args.live !== null && args.live.key === args.paramView?.key) {
        return { ...args.paramView, edit: args.live.edit } as ParamViewSegment;
    }
    return args.paramView;
}

/**
 * Gesture centers and explicit user placements survive asynchronous resampling.
 * Automatic blocks without an explicit placement still follow their curve.
 */
export function resolveParamBlockFrameValue(args: {
    readonly curveOwned?: boolean;
    readonly param?: string;
    readonly block: {
        readonly startSec: number;
        readonly endSec: number;
        readonly averageValue?: number | null;
        readonly logicalCenterValue?: number | null;
        readonly displayCenterValue?: number | null;
        /** Explicit center policy; locked previews must not fall back to curve sampling. */
        readonly centerPolicy?: "live" | "locked";
        /** Center produced by the active block gesture, in the same frame as its curve. */
        readonly frameCenterValue?: number | null;
    };
    readonly samples: {
        readonly values: readonly number[];
        readonly startFrame: number;
        readonly stride: number;
        readonly framePeriodMs: number;
        readonly isPitch: boolean;
        readonly param: string;
    } | null;
}): number | null {
    if (
        args.block.frameCenterValue !== null &&
        args.block.frameCenterValue !== undefined &&
        Number.isFinite(args.block.frameCenterValue)
    ) {
        return args.block.frameCenterValue;
    }
    if (args.block.centerPolicy === "locked") {
        if (
            args.block.displayCenterValue !== null &&
            args.block.displayCenterValue !== undefined &&
            Number.isFinite(args.block.displayCenterValue)
        ) {
            return args.block.displayCenterValue;
        }
        if (
            (args.param ?? args.samples?.param ?? "pitch") === "pitch" &&
            args.block.logicalCenterValue != null &&
            Number.isFinite(args.block.logicalCenterValue)
        ) {
            return args.block.logicalCenterValue;
        }
        if (args.block.averageValue == null) {
            reportParamBlockDiagnostic({
                code: "drag_center_missing",
                gesture: "drag",
                rootTrackId: null,
                blockId: null,
                frame: null,
            });
        }
        return args.block.averageValue ?? null;
    }
    if (
        !args.curveOwned &&
        (args.param ?? args.samples?.param ?? "pitch") === "pitch" &&
        args.block.logicalCenterValue != null &&
        Number.isFinite(args.block.logicalCenterValue)
    ) {
        return args.block.logicalCenterValue;
    }
    if (
        !args.curveOwned &&
        args.block.displayCenterValue != null &&
        Number.isFinite(args.block.displayCenterValue)
    ) {
        return args.block.displayCenterValue;
    }
    const sampled =
        args.samples === null
            ? null
            : modeCenterOfValuesInSamples(args.block.startSec, args.block.endSec, args.samples);
    return sampled ?? args.block.averageValue ?? null;
}

/**
 * Projects all selected block centers for one gesture frame. The caller applies
 * the same value delta to the curve, so the handles cannot lag behind the line
 * while a sampled parameter window is being rebuilt.
 */
export function buildParamBlockFrameCenters(args: {
    readonly blocks: readonly {
        readonly id: string;
        readonly anchorValue: number | null;
    }[];
    readonly valueDelta: number;
    readonly param: string;
    readonly snapToSemitone: boolean;
}): Map<string, number> {
    const centers = new Map<string, number>();
    for (const block of args.blocks) {
        const center = projectParamBlockDragDisplayValue({
            startValue: block.anchorValue,
            valueDelta: args.valueDelta,
            param: args.param,
            snapToSemitone: args.snapToSemitone,
        });
        if (center !== null && Number.isFinite(center)) {
            centers.set(block.id, center);
        }
    }
    return centers;
}
