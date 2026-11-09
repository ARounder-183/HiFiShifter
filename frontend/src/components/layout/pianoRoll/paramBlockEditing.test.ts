import { describe, expect, it } from "vitest";
import { createUserBlock, mergeProjectedBlocks, splitProjectedBlock } from "./paramBlockEditing";
import { composeParamBlockView } from "./paramBlockComposition";
import type { ParamBlockSourceClip } from "./paramBlockProjection";

const clip: ParamBlockSourceClip = {
    id: "clip",
    trackId: "root",
    startSec: 0,
    lengthSec: 2,
    sourceStartSec: 0,
    sourceEndSec: 2,
    playbackRate: 1,
    reversed: false,
    loopEnabled: false,
    syllableSpansSec: [
        { startSec: 0, endSec: 0.5 },
        { startSec: 0.5, endSec: 1 },
    ],
};

describe("paramBlockEditing", () => {
    it("creates a Take-anchored block instead of an absolute interval", () => {
        const edits = createUserBlock(
            [clip],
            { schemaVersion: 1, revision: 0, blocks: [] },
            20,
            80,
            61,
            5,
        );
        const block = edits.blocks[0];
        expect(block?.geometry.kind).toBe("take");
        if (block?.geometry.kind !== "take") throw new Error("expected take");
        expect(block.geometry.members[0]?.sourceStartSec).toBeCloseTo(0.1);
        expect(block.geometry.members[0]?.sourceEndSec).toBeCloseTo(0.4);
    });

    it("merges adjacent projected blocks and splits them at a frame", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 1, revision: 0, blocks: [] },
            5,
        );
        const merged = mergeProjectedBlocks(
            { schemaVersion: 1, revision: 0, blocks: [] },
            base.blocks,
            5,
        );
        const mergedView = composeParamBlockView("root", [clip], merged, 5);
        expect(mergedView.blocks).toHaveLength(1);
        const split = splitProjectedBlock([clip], merged, mergedView.blocks[0]!, 50, 5);
        const splitView = composeParamBlockView("root", [clip], split, 5);
        expect(
            splitView.blocks.map((block) => [block.startFrame, block.endFrameExclusive]),
        ).toEqual([
            [0, 50],
            [50, 200],
        ]);
    });
});
