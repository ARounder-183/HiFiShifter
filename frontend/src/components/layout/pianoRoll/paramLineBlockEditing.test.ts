import { expect, test } from "vitest";
import {
    EMPTY_PARAM_BLOCK_STATE,
    materializeParamBlocks,
    splitParamBlock,
    resizeParamBlock,
    buildParamBlockDragSegments,
} from "./paramBlocks";
import { createLineBlock, mergeLineBlocks } from "./paramLineBlockEditing";
import { paramLineBlockValue } from "./paramLineBlockValue";
import { paramLineBlockIdsInSelection, paramLineRestoreRanges } from "./paramLineBlockSelection";
import { rebaseParamLineBlocks } from "./paramLineBlockPatch";
const state = {
    ...EMPTY_PARAM_BLOCK_STATE,
    lineRootTrackId: "root",
    lineLayout: {
        schemaVersion: 1 as const,
        revision: 1,
        framePeriodMs: 5,
        blocks: [
            { id: "a", startFrame: 0, endFrameExclusive: 20 },
            { id: "b", startFrame: 20, endFrameExclusive: 40 },
        ],
        gameBaseline: { revision: 0, segments: [] },
    },
};
test("manual create, split and resize need no Clip or Take", () => {
    const result = createLineBlock(state, 0.025, 0.075, "manual");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const blocks = materializeParamBlocks({ rootTrackId: "root", clips: [], state: result.state });
    expect(blocks.map((b) => b.members)).toEqual([[], [], [], []]);
    const block = blocks.find((b) => b.id === "manual")!;
    expect(splitParamBlock({ state: result.state, block, sec: 0.05 }).ok).toBe(true);
    expect(
        resizeParamBlock({ state: result.state, block, edge: "right", targetSec: 0.09 }).ok,
    ).toBe(true);
    expect(JSON.parse(JSON.stringify(result.state.lineLayout))).toEqual(result.state.lineLayout);
});
test("merge rejects a nonadjacent choice instead of consuming intervening blocks", () => {
    const separated = {
        ...state,
        lineLayout: {
            ...state.lineLayout,
            blocks: [
                ...state.lineLayout.blocks,
                { id: "c", startFrame: 40, endFrameExclusive: 60 },
            ],
        },
    };
    expect(mergeLineBlocks(separated, ["a", "c"])).toEqual({ ok: false, reason: "not_adjacent" });
});
test("time selection includes blocks independently of vertical visibility and restores empty regions", () => {
    const selection = [{ startFrame: 10, frameCount: 20 }];
    expect(paramLineBlockIdsInSelection(state.lineLayout.blocks, selection)).toEqual(["a", "b"]);
    expect(paramLineRestoreRanges([], [], selection)).toEqual(selection);
});
test("parameter representative translates exactly and zero is valid outside pitch", () => {
    const samples = {
        values: [0.13, 0.24, 0.41, 0.5],
        startFrame: 0,
        stride: 1,
        framePeriodMs: 5,
        isPitch: false,
    };
    const value = paramLineBlockValue(0, 0.02, samples)!;
    expect(
        paramLineBlockValue(0, 0.02, { ...samples, values: samples.values.map((v) => v + 0.13) }),
    ).toBeCloseTo(value + 0.13, 10);
    expect(paramLineBlockValue(0, 0.02, { ...samples, values: [0, 0, 0, 0] })).toBe(0);
});
test("empty pitch can be created by a deliberate block drag, voiced gaps remain gaps", () => {
    const args = {
        blocks: [
            {
                members: [
                    {
                        clipId: "",
                        takeId: "",
                        startSec: 0,
                        endSec: 0.01,
                        sourceStartSec: 0,
                        sourceEndSec: 0.01,
                    },
                ],
            },
        ],
        framePeriodMs: 5,
        param: "pitch",
        anchorValue: 60,
        valueDelta: 2,
        fillEmptyPitch: true,
    };
    expect(buildParamBlockDragSegments({ ...args, sourceAt: () => 0 })[0].values).toEqual([62, 62]);
    expect(
        buildParamBlockDragSegments({ ...args, sourceAt: (i) => (i ? 0 : 60) })[0].values,
    ).toEqual([62, 0]);
});
test("disjoint GAME updates rebase, a changed geometry rejects a stale gesture", () => {
    const basis = state.lineLayout.blocks;
    const edited = [{ ...basis[0], endFrameExclusive: 15 }, basis[1]];
    expect(
        rebaseParamLineBlocks(
            [...basis, { id: "new", startFrame: 50, endFrameExclusive: 60 }],
            basis,
            edited,
        ),
    ).toHaveLength(3);
    expect(() =>
        rebaseParamLineBlocks([{ ...basis[0], startFrame: 5 }, basis[1]], basis, edited),
    ).toThrow("Conflict:");
});
