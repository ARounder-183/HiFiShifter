import { expect, test } from "vitest";
import {
    mapParamLineBlocks,
    placeParamLineBlock,
    resizeParamLineBlock,
    restoreGameParamLineBlocks,
    normalizeParamLineBlocks,
    splitParamLineBlock,
    type ParamLineBlock,
} from "./paramLineBlocks";

const basis: ParamLineBlock[] = [
    { id: "left", startFrame: 0, endFrameExclusive: 1 },
    { id: "middle", startFrame: 1, endFrameExclusive: 2 },
    { id: "right", startFrame: 2, endFrameExclusive: 3 },
];
const spans = (blocks: readonly ParamLineBlock[]) =>
    blocks.map((block) => [block.startFrame, block.endFrameExclusive]);

test("creation preserves both uncovered sides of another block", () => {
    expect(
        spans(
            placeParamLineBlock([{ id: "old", startFrame: 0, endFrameExclusive: 3 }], {
                id: "new",
                startFrame: 1,
                endFrameExclusive: 2,
            }),
        ),
    ).toEqual([
        [0, 1],
        [1, 2],
        [2, 3],
    ]);
});

test("extension shrinks a partly overlapped neighbour instead of removing it", () => {
    expect(
        spans(
            resizeParamLineBlock(
                [
                    { id: "a", startFrame: 0, endFrameExclusive: 10 },
                    { id: "b", startFrame: 10, endFrameExclusive: 20 },
                ],
                "a",
                0,
                15,
            ),
        ),
    ).toEqual([
        [0, 15],
        [15, 20],
    ]);
});

test("pulling an overshot preview back restores all neighbours from the gesture basis", () => {
    expect(spans(resizeParamLineBlock(basis, "middle", 0, 3))).toEqual([[0, 3]]);
    expect(resizeParamLineBlock(basis, "middle", 1, 2)).toEqual(basis);
});

test("preview and persisted data are the same normalized interval list", () => {
    const preview = placeParamLineBlock(basis, {
        id: "created",
        startFrame: 1,
        endFrameExclusive: 2,
    });
    expect(JSON.parse(JSON.stringify(preview))).toEqual(preview);
});

test("Clip move and stretch affect the same parameter interval and preserve unrelated blocks", () => {
    expect(
        spans(
            mapParamLineBlocks(basis, {
                oldStartFrame: 1,
                oldEndFrameExclusive: 2,
                newStartFrame: 3,
                newEndFrameExclusive: 5,
            }),
        ),
    ).toEqual([
        [0, 1],
        [2, 3],
        [3, 5],
    ]);
});

test("Clip crop removes only its affected parameter segment", () => {
    expect(
        spans(
            mapParamLineBlocks([{ id: "whole", startFrame: 0, endFrameExclusive: 3 }], {
                oldStartFrame: 1,
                oldEndFrameExclusive: 2,
                newStartFrame: 1,
                newEndFrameExclusive: 1,
            }),
        ),
    ).toEqual([
        [0, 1],
        [2, 3],
    ]);
});

test("split creates two ordinary blocks without retaining a parent record", () => {
    expect(
        splitParamLineBlock([{ id: "a", startFrame: 0, endFrameExclusive: 2 }], "a", 1, "b"),
    ).toEqual([
        { id: "a", startFrame: 0, endFrameExclusive: 1, transitionFrames: 0 },
        { id: "b", startFrame: 1, endFrameExclusive: 2 },
    ]);
});

test("GAME restore replaces only segmentation inside the selected range", () => {
    const edited = [{ id: "left", startFrame: 0, endFrameExclusive: 30 }];
    const baseline = [
        { id: "left", startFrame: 0, endFrameExclusive: 10 },
        { id: "b", startFrame: 10, endFrameExclusive: 20 },
        { id: "c", startFrame: 20, endFrameExclusive: 30 },
    ];
    expect(spans(restoreGameParamLineBlocks(edited, baseline, 5, 25))).toEqual([
        [0, 5],
        [5, 10],
        [10, 20],
        [20, 25],
        [25, 30],
    ]);
});

test("invalid persisted data is rejected instead of silently losing blocks", () => {
    expect(() =>
        normalizeParamLineBlocks([{ id: "a", startFrame: 0.5, endFrameExclusive: 2 }]),
    ).toThrow();
    expect(() =>
        normalizeParamLineBlocks([
            { id: "a", startFrame: 0, endFrameExclusive: 2 },
            { id: "b", startFrame: 1, endFrameExclusive: 3 },
        ]),
    ).toThrow();
});
