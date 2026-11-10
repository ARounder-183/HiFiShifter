import { describe, expect, it } from "vitest";
import {
    createUserBlock,
    placeUserBlock,
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
        // 被缩掉的 base 派生 child 变成墓碑；组退化为剩下的 child。
        expect(shortened.edits.blocks).toHaveLength(2);
        expect(shortened.edits.blocks.some((block) => block.deleted === true)).toBe(true);
        expect(shortened.edits.blocks.some((block) => block.deleted !== true)).toBe(true);
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

    it("does not resurrect base blocks when a group is shortened", () => {
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
            150,
            5,
        );
        const after = composeParamBlockView("root", [clip], shortened.edits, 5);
        expect(after.blocks.filter((block) => block.kind === "base")).toHaveLength(0);
        expect(
            shortened.edits.blocks.some((block) => block.id === view.blocks[0]?.userBlockId),
        ).toBe(true);
        const tighter = planResizeProjectedBlock([clip], grouped, view.blocks[0]!, "right", 50, 5);
        const afterTight = composeParamBlockView("root", [clip], tighter.edits, 5);
        expect(afterTight.blocks.filter((block) => block.kind === "base")).toHaveLength(0);
    });

    it("splits a multi-run block as one logical block", () => {
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
        const first = view.blocks.find((block) => block.userBlockId === "multi");
        if (first === undefined) throw new Error("multi run missing");
        const plan = planSplitProjectedBlock([clip], edits, first, 20, 5);
        const after = composeParamBlockView("root", [clip], plan.edits, 5);
        const left = after.blocks.filter((block) => block.userBlockId === plan.leftBlockId);
        const right = after.blocks.filter((block) => block.userBlockId === plan.rightBlockId);
        expect(left).toHaveLength(1);
        expect(right).toHaveLength(2);
        expect(left[0]?.startFrame).toBe(0);
        expect(right[0]?.startFrame).toBe(20);
        expect(right[1]?.startFrame).toBe(120);
    });

    it("right-expands the last member of a multi-member run, not the first", () => {
        const edits = {
            schemaVersion: 2 as const,
            revision: 1,
            blocks: [
                {
                    id: "multi-member",
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
                                sourceStartSec: 0.2,
                                sourceEndSec: 0.4,
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
        const block = view.blocks.find((item) => item.userBlockId === "multi-member");
        if (block === undefined) throw new Error("multi-member run missing");
        const resized = planResizeProjectedBlock([clip], edits, block, "right", 120, 5);
        const result = resized.edits.blocks.find((item) => item.id === "multi-member");
        if (result?.geometry.kind !== "take") throw new Error("expected take");
        expect(result.geometry.members[0]?.sourceEndSec).toBeCloseTo(0.2, 6);
        expect(result.geometry.members[1]?.sourceEndSec).toBeCloseTo(0.6, 6);
    });

    it("splitting a base-derived block partitions its baseCover so deleting one half restores the base", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const ensured = ensureEditableUserBlock(
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            base.blocks[0]!,
            5,
        );
        const userView = composeParamBlockView("root", [clip], ensured.edits, 5).blocks.find(
            (block) => block.userBlockId === ensured.user.id,
        );
        if (userView === undefined) throw new Error("materialized block missing");
        const split = planSplitProjectedBlock([clip], ensured.edits, userView, 50, 5);
        const right = split.viewAfter.blocks.find((block) => block.id === split.rightBlockId);
        if (right === undefined) throw new Error("right split missing");
        const deleted = deleteProjectedBlocks([clip], split.edits, [right], 5);
        const after = composeParamBlockView("root", [clip], deleted.edits, 5);
        expect(
            after.blocks.some(
                (block) =>
                    block.kind === "base" &&
                    block.startFrame === 50 &&
                    block.endFrameExclusive === 100,
            ),
        ).toBe(true);
    });

    it("resizes only the grabbed run of a multi-run block", () => {
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
        const first = view.blocks.find((block) => block.userBlockId === "multi");
        if (first === undefined) throw new Error("multi run missing");
        const plan = planResizeProjectedBlock([clip], edits, first, "right", 20, 5);
        const after = composeParamBlockView("root", [clip], plan.edits, 5);
        const runs = after.blocks.filter((block) => block.userBlockId === "multi");
        expect(runs.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [0, 20],
            [120, 160],
        ]);
    });

    it("trims a group recursively when a new block covers part of it", () => {
        const group: import("./paramBlockTypes").UserParamBlock = {
            id: "group",
            geometry: {
                kind: "group",
                children: [
                    {
                        id: "child-a",
                        geometry: {
                            kind: "take",
                            members: [
                                {
                                    clipId: "clip",
                                    takeId: "clip_take_1",
                                    cycle: 0,
                                    sourceStartSec: 0,
                                    sourceEndSec: 0.5,
                                    label: "manual",
                                },
                            ],
                        },
                        manual: true,
                        createdSeq: 1,
                    },
                    {
                        id: "child-b",
                        geometry: {
                            kind: "take",
                            members: [
                                {
                                    clipId: "clip",
                                    takeId: "clip_take_1",
                                    cycle: 0,
                                    sourceStartSec: 0.5,
                                    sourceEndSec: 1,
                                    label: "manual",
                                },
                            ],
                        },
                        manual: true,
                        createdSeq: 2,
                    },
                ],
            },
            manual: true,
            createdSeq: 3,
        };
        const target: import("./paramBlockTypes").UserParamBlock = {
            id: "target",
            geometry: {
                kind: "take",
                members: [
                    {
                        clipId: "clip",
                        takeId: "clip_take_1",
                        cycle: 0,
                        sourceStartSec: 0.2,
                        sourceEndSec: 0.4,
                        label: "manual",
                    },
                ],
            },
            manual: true,
            createdSeq: 4,
        };
        const next = placeUserBlock({ schemaVersion: 2, revision: 0, blocks: [group] }, target);
        const groupAfter = next.blocks.find((block) => block.id === "group");
        expect(groupAfter?.geometry.kind).toBe("group");
        if (groupAfter?.geometry.kind !== "group") return;
        const members = groupAfter.geometry.children.flatMap((child) =>
            child.geometry.kind === "take" ? child.geometry.members : [],
        );
        expect(
            members.some((member) => member.sourceStartSec < 0.4 && member.sourceEndSec > 0.2),
        ).toBe(false);
    });
    it("groups non-adjacent projected blocks without filling the gap", () => {
        const edits = {
            schemaVersion: 2 as const,
            revision: 1,
            blocks: [
                {
                    id: "left",
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
                        ],
                    },
                    deleted: false,
                    manual: true,
                    value: 60,
                    createdSeq: 1,
                },
                {
                    id: "right",
                    geometry: {
                        kind: "take" as const,
                        members: [
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
                    createdSeq: 2,
                },
            ],
        };
        const view = composeParamBlockView("root", [clip], edits, 5);
        const selected = view.blocks.filter((block) => block.kind !== "base");
        expect(selected).toHaveLength(2);
        const grouped = groupProjectedBlocks([clip], edits, selected, 5);
        const groupedView = composeParamBlockView("root", [clip], grouped, 5);
        const runs = groupedView.blocks.filter((block) => block.kind !== "base");
        expect(runs.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [0, 40],
            [120, 160],
        ]);
        expect(runs.every((block) => block.userBlockId !== undefined)).toBe(true);
        expect(new Set(runs.map((block) => block.userBlockId)).size).toBe(1);
        // 组只覆盖两个 child 的真实 run，80..120 的空白不能被组填满。
        expect(runs.some((block) => block.startFrame < 80 && block.endFrameExclusive > 80)).toBe(
            false,
        );
        const restored = ungroupUserBlock(grouped, runs[0]!.userBlockId!);
        const restoredView = composeParamBlockView("root", [clip], restored, 5);
        expect(
            restoredView.blocks
                .filter((block) => block.kind !== "base")
                .map((block) => block.id)
                .sort(),
        ).toEqual(["left", "right"]);
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
