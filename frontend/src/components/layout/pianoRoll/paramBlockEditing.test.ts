import { describe, expect, it } from "vitest";
import {
    createUserBlock,
    ensureEditableUserBlock,
    groupProjectedBlocks,
    splitProjectedBlock,
    ungroupUserBlock,
} from "./paramBlockEditing";
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

    it("materializes a multi-run user block before editing one run", () => {
        const edits = {
            schemaVersion: 1 as const,
            revision: 1,
            blocks: [
                {
                    id: "multi",
                    geometry: {
                        kind: "take" as const,
                        members: [
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0,
                                sourceEndSec: 0.2,
                                label: "manual" as const,
                            },
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0.6,
                                sourceEndSec: 0.8,
                                label: "manual" as const,
                            },
                        ],
                    },
                    deleted: false,
                    manual: true,
                    value: 60,
                    createdSeq: 1,
                },
            ],
        };
        const view = composeParamBlockView("root", [clip], edits, 5);
        // 基块衬底：两个用户 run 之间的空隙会露出基块余段，因此视图不止两块。
        const userBlocks = view.blocks.filter((block) => block.kind !== "base");
        expect(userBlocks).toHaveLength(2);
        const ensured = ensureEditableUserBlock([clip], edits, userBlocks[0]!, 5);
        expect(ensured.user.id).toBe(userBlocks[0]!.id);
        expect(ensured.user.geometry.kind).toBe("take");
        expect(ensured.edits.blocks.some((block) => block.id === "multi")).toBe(true);
    });

    it("groups adjacent projected blocks reversibly and splits at a frame", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 1, revision: 0, blocks: [] },
            5,
        );
        const grouped = groupProjectedBlocks(
            [clip],
            { schemaVersion: 1, revision: 0, blocks: [] },
            base.blocks,
            5,
        );
        const groupedView = composeParamBlockView("root", [clip], grouped, 5);
        expect(groupedView.blocks).toHaveLength(1);
        expect(groupedView.blocks[0]?.kind).toBe("grouped");
        const restored = ungroupUserBlock(grouped, groupedView.blocks[0]!.id);
        const restoredView = composeParamBlockView("root", [clip], restored, 5);
        expect(restoredView.blocks).toHaveLength(2);
        const split = splitProjectedBlock([clip], restored, restoredView.blocks[0]!, 50, 5);
        const splitView = composeParamBlockView("root", [clip], split, 5);
        expect(
            splitView.blocks.map((block) => [block.startFrame, block.endFrameExclusive]),
        ).toEqual([
            [0, 50],
            [50, 100],
            [100, 200],
        ]);
    });
});
