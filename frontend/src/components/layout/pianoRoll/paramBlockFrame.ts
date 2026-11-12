import {
    logicalBlockKey,
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

/** Block boundaries are shared; every non-auto block may carry an explicit center. */
export function resolveParamBlockPanelValues(args: {
    readonly blocks: readonly ParamBlock[];
    readonly param: string;
    readonly paramView: ParamViewSegment | null;
    /** Viewport midpoint; only used by previews that explicitly opt in. */
    readonly fallbackValue: number;
    /** Last known value per run, used while a refresh is in flight. */
    readonly previousValues?: ReadonlyMap<
        string,
        { readonly value: number; readonly startSec: number; readonly endSec: number }
    >;
    /** Canonical blocks must never guess a center from the viewport. */
    readonly allowFallback?: boolean;
    /** Parameter default; the last display-only fallback for unvalued blocks. */
    readonly currentParamDefaultValue?: number | null;
}): ParamBlock[] {
    const nearestValue = (index: number): number | null => {
        const block = args.blocks[index];
        const center = (block.startSec + block.endSec) / 2;
        let best: { distance: number; value: number } | null = null;
        for (let other = 0; other < args.blocks.length; other += 1) {
            if (other === index) continue;
            const candidate = args.blocks[other];
            if (candidate.sourceKey !== block.sourceKey || candidate.sourceKey === undefined)
                continue;
            const value = candidate.averageValue;
            if (value === null || !Number.isFinite(value)) continue;
            const distance = Math.abs((candidate.startSec + candidate.endSec) / 2 - center);
            if (distance > 2) continue;
            if (best === null || distance < best.distance) best = { distance, value };
        }
        return best?.value ?? null;
    };
    return args.blocks.map((block, index) => {
        // volume / dyn 的拖动落点同样写在 `value` 里；只对 pitch 生效会让块高
        // 在下一次曲线刷新时跳回主簇。auto 块没有持久化中心，保持 null。
        const manualValue =
            block.kind === "auto" && args.param !== "pitch" ? null : (block.manualValue ?? null);
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
        const key = logicalBlockKey(block);
        const cached = args.previousValues?.get(key) ?? args.previousValues?.get(block.id);
        // Geometry changes (trim/slip) are not a new logical block: reuse the
        // last known value instead of hiding the block during a refresh.
        const previous =
            cached !== undefined && Number.isFinite(cached.value) ? cached.value : null;
        const previewAverage =
            pv === null && args.param === "pitch" && Number.isFinite(block.averageValue)
                ? block.averageValue
                : null;
        const sourceNote =
            args.param === "pitch" && block.members.length === 1 && block.sourceNote != null
                ? block.sourceNote
                : null;
        const neighbor = nearestValue(index);
        const defaultValue =
            typeof args.currentParamDefaultValue === "number" &&
            Number.isFinite(args.currentParamDefaultValue)
                ? args.currentParamDefaultValue
                : null;
        const derived =
            manualValue ??
            sampled ??
            previous ??
            previewAverage ??
            sourceNote ??
            neighbor ??
            defaultValue;
        const usedFallback =
            manualValue === null &&
            sampled === null &&
            previous === null &&
            previewAverage === null &&
            derived !== null;
        return {
            ...block,
            manualValue,
            averageValue:
                manualValue ??
                sampled ??
                previous ??
                previewAverage ??
                (args.allowFallback === true ? args.fallbackValue : derived),
            ...(usedFallback ? { fallbackValue: derived, centerPolicy: "fallback" as const } : {}),
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
    const sameScope =
        interaction !== null &&
        args.paramView !== null &&
        scope(interaction.key) === scope(args.paramView.key);
    if (
        interaction !== null &&
        sameScope &&
        (!args.param ||
            !interaction.key.startsWith("v2|") ||
            interaction.key.split("|")[2] === args.param)
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
        readonly centerPolicy?: "live" | "locked" | "fallback";
        readonly sourceNote?: number | null;
        readonly fallbackValue?: number | null;
        readonly members?: readonly unknown[];
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
    /** Fallback chains used by the host when no sampled value exists. */
    readonly fallbackValue?: number | null;
    readonly currentParamDefaultValue?: number | null;
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
    const param = args.param ?? args.samples?.param ?? "pitch";
    const sourceNote =
        param === "pitch" && args.block.members?.length === 1 && args.block.sourceNote != null
            ? args.block.sourceNote
            : null;
    const fallback = args.block.fallbackValue ?? sourceNote ?? args.fallbackValue ?? null;
    const defaultValue =
        typeof args.currentParamDefaultValue === "number" &&
        Number.isFinite(args.currentParamDefaultValue)
            ? args.currentParamDefaultValue
            : null;
    return sampled ?? args.block.averageValue ?? fallback ?? defaultValue;
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
