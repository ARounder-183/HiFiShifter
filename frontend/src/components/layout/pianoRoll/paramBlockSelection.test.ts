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
        ).toEqual(["a:0", "a:1"]);
    });

    it("restore scope uses whole logical blocks, not just the clicked run", () => {
        expect(paramBlockRestoreRanges(blocks, ["a:0"], null)).toEqual([
            { startFrame: 0, frameCount: 20 },
            { startFrame: 100, frameCount: 20 },
        ]);
    });
});
