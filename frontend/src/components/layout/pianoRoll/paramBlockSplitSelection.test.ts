import { describe, expect, it } from "vitest";
import { resolveParamBlockSplitSelection } from "./paramBlockSplitSelection";

const block = (
    id: string,
    startFrame: number,
    endFrameExclusive: number,
    userBlockId?: string,
) => ({
    id,
    ...(userBlockId === undefined ? {} : { userBlockId }),
    startFrame,
    endFrameExclusive,
});

describe("resolveParamBlockSplitSelection", () => {
    it("selects only the right result for a single split", () => {
        const result = resolveParamBlockSplitSelection({
            beforeBlocks: [block("b", 10, 20)],
            beforeSelection: [{ startFrame: 10, frameCount: 10 }],
            splitFrame: 15,
            targetLogicalId: "b",
            leftBlockId: "b-left",
            rightBlockId: "b-right",
            viewAfter: {
                blocks: [block("b-left", 10, 15), block("b-right", 15, 20)],
            },
            rightSpan: { startFrame: 15, endFrameExclusive: 20 },
        });
        expect(result.selectedLogicalIds).toEqual(["b-right"]);
        expect(result.selection).toEqual([{ startFrame: 15, frameCount: 5 }]);
        expect(result.anchor).toEqual({ blockId: "b-right", frame: 15 });
    });

    it("drops left-side selected blocks and keeps right-side ones", () => {
        const result = resolveParamBlockSplitSelection({
            beforeBlocks: [block("a", 0, 10), block("b", 10, 20), block("c", 20, 30)],
            beforeSelection: [
                { startFrame: 0, frameCount: 10 },
                { startFrame: 10, frameCount: 10 },
                { startFrame: 20, frameCount: 10 },
            ],
            splitFrame: 15,
            targetLogicalId: "b",
            leftBlockId: "b-left",
            rightBlockId: "b-right",
            viewAfter: {
                blocks: [
                    block("a", 0, 10),
                    block("b-left", 10, 15),
                    block("b-right", 15, 20),
                    block("c", 20, 30),
                ],
            },
            rightSpan: { startFrame: 15, endFrameExclusive: 20 },
        });
        expect(result.selectedLogicalIds).toEqual(["c", "b-right"]);
        expect(result.selection).toEqual([{ startFrame: 15, frameCount: 15 }]);
    });

    it("expands a multi-run right result to every run", () => {
        const result = resolveParamBlockSplitSelection({
            beforeBlocks: [block("p:0", 10, 20, "p"), block("p:1", 30, 40, "p")],
            beforeSelection: [
                { startFrame: 10, frameCount: 10 },
                { startFrame: 30, frameCount: 10 },
            ],
            splitFrame: 15,
            targetLogicalId: "p",
            rightBlockId: "p",
            viewAfter: {
                blocks: [block("p:0", 15, 20, "p"), block("p:1", 30, 40, "p")],
            },
            rightSpan: { startFrame: 15, endFrameExclusive: 20 },
        });
        expect(result.selection).toEqual([
            { startFrame: 15, frameCount: 5 },
            { startFrame: 30, frameCount: 10 },
        ]);
    });

    it("leaves selection untouched when selectResult is false (caller responsibility)", () => {
        const selection = [{ startFrame: 0, frameCount: 5 }];
        const result = resolveParamBlockSplitSelection({
            beforeBlocks: [block("b", 10, 20)],
            beforeSelection: selection,
            splitFrame: 15,
            targetLogicalId: "b",
            rightBlockId: "b-right",
            viewAfter: { blocks: [block("b-right", 15, 20)] },
        });
        // The pure function still describes the select-result policy; the
        // caller must not invoke it when selectResult === false.
        expect(result.selectedLogicalIds).toEqual(["b-right"]);
    });
});
