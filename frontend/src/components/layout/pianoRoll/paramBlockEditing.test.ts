import { describe, expect, it } from "vitest";
import {
    createUserBlock,
    deleteProjectedBlocks,
    ensureEditableUserBlock,
    groupProjectedBlocks,
    planResizeProjectedBlock,
    planSplitProjectedBlock,
    restoreDeletedBlocks,
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
            { schemaVersion: 2, revision: 0, blocks: [] },
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
            schemaVersion: 2 as const,
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

    it("delete group creates child tombstones and restores the whole group", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const grouped = groupProjectedBlocks(
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            base.blocks,
            5,
        );
        const view = composeParamBlockView("root", [clip], grouped, 5);
        const deleted = deleteProjectedBlocks([clip], grouped, view.blocks, 5);
        expect(deleted.deleted).toHaveLength(1);
        expect(deleted.deleted[0]?.childTombstones?.length ?? 0).toBeGreaterThanOrEqual(2);
        expect(
            composeParamBlockView("root", [clip], deleted.edits, 5).blocks.filter(
                (block) => block.kind !== "base",
            ),
        ).toHaveLength(0);
        const restored = restoreDeletedBlocks(deleted.edits, deleted.deleted);
        const restoredView = composeParamBlockView("root", [clip], restored, 5);
        const user = restoredView.blocks.filter((block) => block.kind !== "base");
        expect(user).toHaveLength(1);
        expect(user[0]?.kind).toBe("grouped");
    });

    it("splits a group inside a child and can ungroup the edited children", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const grouped = groupProjectedBlocks(
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            base.blocks,
            5,
        );
        const view = composeParamBlockView("root", [clip], grouped, 5);
        const plan = planSplitProjectedBlock([clip], grouped, view.blocks[0]!, 50, 5);
        const after = composeParamBlockView("root", [clip], plan.edits, 5);
        const userBlocks = after.blocks.filter((block) => block.kind !== "base");
        expect(userBlocks).toHaveLength(2);
        expect([plan.leftSpan.startFrame, plan.leftSpan.endFrameExclusive]).toEqual([0, 50]);
        expect([plan.rightSpan.startFrame, plan.rightSpan.endFrameExclusive]).toEqual([50, 200]);
        const rightGroup = plan.edits.blocks.find((block) => block.id === plan.rightBlockId);
        expect(rightGroup?.geometry.kind).toBe("group");
        if (rightGroup?.geometry.kind !== "group") throw new Error("expected right group");
        expect(rightGroup.geometry.children).toHaveLength(2);
        const restored = ungroupUserBlock(plan.edits, rightGroup.id);
        const restoredUser = composeParamBlockView("root", [clip], restored, 5).blocks.filter(
            (block) => block.kind !== "base",
        );
        expect(restoredUser).toHaveLength(3);
        expect(restoredUser[2]?.endFrameExclusive).toBe(200);
    });

    it("splits a group exactly at a child boundary into plain blocks", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const grouped = groupProjectedBlocks(
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            base.blocks,
            5,
        );
        const view = composeParamBlockView("root", [clip], grouped, 5);
        const plan = planSplitProjectedBlock([clip], grouped, view.blocks[0]!, 100, 5);
        expect(plan.edits.blocks.map((block) => block.geometry.kind)).toEqual(["take", "take"]);
    });

    it("resizes group edges and degrades to a child when only one remains", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const grouped = groupProjectedBlocks(
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            base.blocks,
            5,
        );
        const view = composeParamBlockView("root", [clip], grouped, 5);
        const shortened = planResizeProjectedBlock(
            [clip],
            grouped,
            view.blocks[0]!,
            "right",
            50,
            5,
        );
        expect(shortened.targetSpan).toEqual({ startFrame: 0, endFrameExclusive: 50 });
        expect(shortened.edits.blocks).toHaveLength(1);
        expect(shortened.edits.blocks[0]?.id).not.toBe(view.blocks[0]?.id);
        const extended = planResizeProjectedBlock(
            [clip],
            shortened.edits,
            (() => {
                const block = composeParamBlockView("root", [clip], shortened.edits, 5).blocks[0];
                if (block === undefined) throw new Error("missing shortened block");
                return block;
            })(),
            "right",
            150,
            5,
        );
        expect(extended.targetSpan?.endFrameExclusive).toBe(150);
    });

    it("groups adjacent projected blocks reversibly and splits at a frame", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const grouped = groupProjectedBlocks(
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
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
