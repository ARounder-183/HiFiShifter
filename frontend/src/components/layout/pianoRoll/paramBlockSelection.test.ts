import { describe, expect, it } from "vitest";
import {
    expandParamBlockSelection,
    paramBlockIdsInSelection,
    paramBlockLogicalFrameRanges,
    paramBlockRestoreRanges,
} from "./paramBlockSelection";

const blocks = [
    { id: "a:0", userBlockId: "a", startFrame: 0, endFrameExclusive: 20 },
    { id: "a:1", userBlockId: "a", startFrame: 100, endFrameExclusive: 120 },
    { id: "b", startFrame: 40, endFrameExclusive: 60 },
];

describe("paramBlockSelection", () => {
    it("expands any touched fragment to the whole logical block", () => {
        expect(expandParamBlockSelection(blocks, [{ startFrame: 0, frameCount: 5 }])).toEqual([
            { startFrame: 0, frameCount: 20 },
            { startFrame: 100, frameCount: 20 },
        ]);
    });

    it("clicking one run selects every visible run of the logical block", () => {
        expect(paramBlockLogicalFrameRanges(blocks, "a:0")).toEqual([
            { startFrame: 0, frameCount: 20 },
            { startFrame: 100, frameCount: 20 },
        ]);
        expect(
            paramBlockIdsInSelection(blocks, paramBlockLogicalFrameRanges(blocks, "a:0")),
        ).toEqual(["a"]);
    });

    it("requires full coverage before a block is selected", () => {
        expect(paramBlockIdsInSelection(blocks, [{ startFrame: 0, frameCount: 10 }])).toEqual([]);
        expect(paramBlockIdsInSelection(blocks, [{ startFrame: 40, frameCount: 20 }])).toEqual([
            "b",
        ]);
    });

    it("zero-length runs do not become ghost blocks", () => {
        const withZero = [...blocks, { id: "zero", startFrame: 80, endFrameExclusive: 80 }];
        expect(paramBlockLogicalFrameRanges(withZero, "zero")).toEqual([]);
        expect(paramBlockIdsInSelection(withZero, [{ startFrame: 79, frameCount: 2 }])).toEqual([]);
        expect(
            paramBlockIdsInSelection(withZero, [
                { startFrame: 0, frameCount: 20 },
                { startFrame: 100, frameCount: 20 },
            ]),
        ).toEqual(["a"]);
    });

    it("restore scope uses whole logical blocks, not just the clicked run", () => {
        expect(paramBlockRestoreRanges(blocks, ["a:0"], null)).toEqual([
            { startFrame: 0, frameCount: 20 },
            { startFrame: 100, frameCount: 20 },
        ]);
    });
});
