import { describe, expect, it } from "vitest";
import { createTimelineAxis } from "../../renderKernel/timelineAxis";
import type { ParamViewSegment } from "../types";
import { buildCurveLayers } from "./sceneBuilders";

Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { devicePixelRatio: 1 },
});

const canonical: ParamViewSegment = {
    key: "v2|track|pitch|0|4|1",
    framePeriodMs: 5,
    startFrame: 0,
    stride: 1,
    referenceKind: "source_curve",
    orig: [60, 60, 60, 60],
    edit: [60, 60, 60, 60],
    edited: [],
};

function layersWith(interaction: ParamViewSegment | null, frameView?: ParamViewSegment | null) {
    return buildCurveLayers(
        {
            clipboardRef: { current: null },
            clipPitchOverlays: [],
            editParam: "pitch",
            liveEditOverrideRef: { current: null },
            paramView: canonical,
            pitchEnabled: true,
            referencePitchOverlays: [],
            secSpanClipRect: () => null,
            secondaryParamViews: {},
            selectionClipRect: () => null,
            selectionRef: { current: null },
            themeMode: "dark",
            valueToY: (_param, value) => 200 - value,
            viewSizeRef: { current: { w: 800, h: 400 } },
            visibleSecondaryParamIds: [],
            interactionParamViewRef: { current: interaction },
        },
        createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 800 }),
        frameView,
    );
}

describe("block interaction curve frame", () => {
    it("uses the interaction curve as the only edited curve source", () => {
        const preview = { ...canonical, edit: [64, 64, 64, 64] };
        const layers = layersWith(preview);
        const edited = layers.filter((layer) => layer.lineWidthPx === 2.6);
        expect(edited).toHaveLength(1);
        expect(edited[0]?.values).toEqual(preview.edit);
        expect(edited[0]?.values).not.toEqual(canonical.edit);
    });

    it("falls back to the canonical window after the transaction ends", () => {
        const layers = layersWith(null);
        const edited = layers.filter((layer) => layer.lineWidthPx === 2.6);
        expect(edited[0]?.values).toEqual(canonical.edit);
    });
});

it("an atomically published frame wins over a late live window", () => {
    const late = { ...canonical, startFrame: 100, stride: 4, edit: [90, 90, 90, 90] };
    const frame = { ...canonical, edit: [64, 64, 60, 60] };
    const layer = layersWith(late, frame).find((layer) => layer.lineWidthPx === 2.6)!;
    expect(layer.values).toBe(frame.edit);
    expect(layer.startFrame).toBe(0);
    expect(layer.stride).toBe(1);
});
