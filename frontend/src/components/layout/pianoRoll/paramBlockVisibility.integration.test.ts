import { describe, expect, it } from "vitest";
import { composeParamBlockView } from "./paramBlockComposition";
import { materializeParamBlocks, modeCenterOfValuesInSamples } from "./paramBlocks";
import { resolveParamBlockPanelValues } from "./paramBlockFrame";
import { buildParamBlockRenderSpec } from "./paramBlockRenderSpec";
import { buildParamBlockScene } from "./kernel/scene/blockInstances";
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import type { ParamBlockEditSet } from "./paramBlockTypes";

const EMPTY: ParamBlockEditSet = { schemaVersion: 2, revision: 0, blocks: [] };
const COLORS = {
    borderRgba: [1, 1, 1, 1] as const,
    selectedRgba: [1, 1, 0, 1] as const,
    groupedRgba: [0, 1, 1, 1] as const,
    fillRgba: [0.2, 0.2, 0.2, 0.6] as const,
};

function clip(overrides: Partial<ParamBlockSourceClip> = {}): ParamBlockSourceClip {
    return {
        id: "A",
        trackId: "root",
        startSec: 0,
        lengthSec: 2,
        sourceStartSec: 1,
        sourceEndSec: 3,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        syllableSpansSec: [{ startSec: 0.6, endSec: 1.4, note: 64 }],
        syllableAnalysisWindowSec: [0.6, 3],
        ...overrides,
    } as ParamBlockSourceClip;
}

function sceneFor(clips: readonly ParamBlockSourceClip[]) {
    const view = composeParamBlockView("root", clips, EMPTY, 5);
    const blocks = materializeParamBlocks({ rootTrackId: "root", view });
    const values = resolveParamBlockPanelValues({
        blocks,
        param: "pitch",
        paramView: null,
        fallbackValue: 0,
        allowFallback: false,
        currentParamDefaultValue: 60,
    });
    const spec = buildParamBlockRenderSpec({
        blocks: values.map((block) => ({
            id: block.id,
            startSec: block.startSec,
            endSec: block.endSec,
            averageValue: block.averageValue,
            fallbackValue: block.fallbackValue,
            sourceNote: block.sourceNote,
            cutStart: block.cutStart,
            cutEnd: block.cutEnd,
            sourceKey: block.sourceKey,
            centerPolicy: block.centerPolicy,
            selected: false,
            grouped: false,
        })),
        param: "pitch",
        valueMin: 24,
        valueMax: 96,
        valueToY: (value) => (96 - value) * 10,
        colors: COLORS,
    });
    const scene = buildParamBlockScene({
        axis: {
            scrollLeftPx: 0,
            pxPerSec: 100,
            viewportWidthPx: 1000,
            dpr: 1,
        } as never,
        viewportHeightPx: 600,
        blocks: spec.blocks.map((block) => ({
            ...block,
            topPx: 300,
            heightPx: 18,
            fallback: block.centerPolicy === "fallback",
        })),
        borderRgba: COLORS.borderRgba,
        selectedRgba: COLORS.selectedRgba,
        groupedRgba: COLORS.groupedRgba,
        fillRgba: COLORS.fillRgba,
    });
    return { view, values, spec, scene };
}

describe("left-cut parameter block visibility", () => {
    it("draws a single Clip's left-cut fallback block", () => {
        const { view, values, scene } = sceneFor([clip()]);
        expect(view.blocks).toHaveLength(1);
        expect(view.blocks[0]?.startFrame).toBe(0);
        expect(view.blocks[0]?.cutStart).toBe(true);
        expect(values[0]?.averageValue).toBe(64);
        expect(values[0]?.centerPolicy).toBe("fallback");
        expect(scene.fills.length).toBeGreaterThan(0);
        expect(scene.items.length).toBeGreaterThan(0);
    });

    it("draws a one-frame sub-frame intersection", () => {
        const { view, scene } = sceneFor([
            clip({ syllableSpansSec: [{ startSec: 0.5, endSec: 1.001, note: 64 }] }),
        ]);
        expect(view.blocks[0]?.endFrameExclusive).toBe(1);
        expect(scene.fills.length).toBeGreaterThan(0);
    });

    it("keeps crossing sources separate while drawing every actual interval", () => {
        const { view, scene } = sceneFor([
            clip({ id: "A" }),
            clip({
                id: "B",
                startSec: 0.5,
                lengthSec: 1,
                sourceStartSec: 1.2,
                sourceEndSec: 2.2,
                syllableSpansSec: [{ startSec: 1.18, endSec: 1.5, note: 62 }],
                syllableAnalysisWindowSec: [1.18, 2.2],
            }),
        ]);
        expect(view.blocks.length).toBeGreaterThanOrEqual(2);
        const keys = view.blocks.map((block) => block.sourceKey);
        expect(keys.every((key) => key !== undefined)).toBe(true);
        expect(new Set(keys).size).toBe(2);
        expect(scene.fills.length).toBe(view.blocks.length);
    });

    it("samples a one-frame block that falls between strided frames", () => {
        expect(
            modeCenterOfValuesInSamples(0.05, 0.055, {
                values: [64],
                startFrame: 10,
                stride: 1,
                framePeriodMs: 5,
                isPitch: true,
                param: "pitch",
            }),
        ).toBe(64);
    });
});
