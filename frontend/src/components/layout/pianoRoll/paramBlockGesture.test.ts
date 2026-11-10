import { describe, expect, it } from "vitest";
import { composeParamBlockView } from "./paramBlockComposition";
import { planParamBlockGesture } from "./paramBlockGesture";
import { createUserBlock, deleteProjectedBlocks, restoreDeletedBlocks } from "./paramBlockEditing";
import { EMPTY_PARAM_BLOCK_STATE, createParamBlock } from "./paramBlocks";
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import type { ParamBlockEditSet } from "./paramBlockTypes";

const EMPTY: ParamBlockEditSet = { schemaVersion: 1, revision: 0, blocks: [] };
const FRAME_MS = 5;

function loopClip(): ParamBlockSourceClip {
    return {
        id: "clip",
        trackId: "root",
        startSec: 0,
        lengthSec: 6,
        sourceStartSec: 0.266,
        sourceEndSec: 2.27,
        playbackRate: 1,
        reversed: false,
        loopEnabled: true,
        loopPeriodSec: 2.27,
        contentDurationSec: 2.27,
        syllableSpansSec: [{ startSec: 0.046, endSec: 0.252 }],
    };
}

function reverseClip(): ParamBlockSourceClip {
    return {
        id: "clip",
        trackId: "root",
        startSec: 0,
        lengthSec: 2,
        sourceStartSec: 0,
        sourceEndSec: 2,
        playbackRate: 1,
        reversed: true,
        loopEnabled: false,
        syllableSpansSec: [{ startSec: 0.2, endSec: 0.6 }],
    };
}

function crossClips(): ParamBlockSourceClip[] {
    return [
        {
            id: "a",
            trackId: "root",
            startSec: 0,
            lengthSec: 4,
            sourceStartSec: 10,
            sourceEndSec: 14,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
        },
        {
            id: "b",
            trackId: "root",
            startSec: 4,
            lengthSec: 4,
            sourceStartSec: 0,
            sourceEndSec: 4,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
        },
    ];
}

function crossEdits(): ParamBlockEditSet {
    return {
        schemaVersion: 1,
        revision: 1,
        blocks: [
            {
                id: "cross",
                geometry: {
                    kind: "take",
                    members: [
                        {
                            clipId: "a",
                            takeId: "a_take_1",
                            cycle: 0,
                            sourceStartSec: 10,
                            sourceEndSec: 14,
                            label: "manual",
                        },
                        {
                            clipId: "b",
                            takeId: "b_take_1",
                            cycle: 0,
                            sourceStartSec: 0,
                            sourceEndSec: 4,
                            label: "manual",
                        },
                    ],
                },
                manual: true,
                value: 60,
                createdSeq: 1,
            },
        ],
    };
}

function twoMemberEdits(): ParamBlockEditSet {
    return {
        schemaVersion: 1,
        revision: 1,
        blocks: [
            {
                id: "user",
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
                value: 60,
                createdSeq: 1,
            },
        ],
    };
}

const plainClip: ParamBlockSourceClip = {
    id: "clip",
    trackId: "root",
    startSec: 0,
    lengthSec: 2,
    sourceStartSec: 0,
    sourceEndSec: 2,
    playbackRate: 1,
    reversed: false,
    loopEnabled: false,
};

describe("paramBlockGesture — loop / reverse", () => {
    it("R1: splits a loop-wrap block at every interior frame", () => {
        const clips = [loopClip()];
        const view = composeParamBlockView("root", clips, EMPTY, FRAME_MS);
        expect(view.blocks).toHaveLength(2);
        const block = view.blocks[0]!;
        expect(block.startFrame).toBe(410);
        expect(block.endFrameExclusive).toBe(451);
        for (const frame of [420, 430, 440]) {
            const result = planParamBlockGesture({
                clips,
                edits: EMPTY,
                framePeriodMs: FRAME_MS,
                gesture: { kind: "split", blockId: block.id, frame },
            });
            expect(result.ok, `frame ${frame}`).toBe(true);
            if (!result.ok) continue;
            expect(result.plan.leftSpan).toEqual({
                startFrame: 410,
                endFrameExclusive: frame,
            });
            expect(result.plan.rightSpan).toEqual({
                startFrame: frame,
                endFrameExclusive: 451,
            });
        }
    });

    it("R3: resizing a loop-wrap block by 2 frames does not swallow following blocks", () => {
        const clips = [loopClip()];
        const view = composeParamBlockView("root", clips, EMPTY, FRAME_MS);
        const block = view.blocks[0]!;
        const result = planParamBlockGesture({
            clips,
            edits: EMPTY,
            framePeriodMs: FRAME_MS,
            gesture: {
                kind: "resize",
                blockId: block.id,
                edge: "right",
                targetFrame: block.endFrameExclusive + 2,
            },
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.plan.targetSpan).toEqual({
            startFrame: 410,
            endFrameExclusive: 453,
        });
        expect(result.plan.viewAfter.blocks).toHaveLength(2);
        expect(result.plan.viewAfter.blocks[1]!.kind).toBe("base");
    });

    it("R2: splits a user block that crosses the loop cycle boundary", () => {
        const clips = [
            {
                ...loopClip(),
                lengthSec: 4,
                syllableSpansSec: [],
            },
        ];
        const edits: ParamBlockEditSet = {
            schemaVersion: 1,
            revision: 1,
            blocks: [
                {
                    id: "u",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0.2,
                                sourceEndSec: 0.3,
                                label: "manual",
                            },
                        ],
                    },
                    manual: true,
                    value: 60,
                    createdSeq: 1,
                },
            ],
        };
        const view = composeParamBlockView("root", clips, edits, FRAME_MS);
        expect(view.blocks).toHaveLength(1);
        expect(view.blocks[0]!.members).toHaveLength(2);
        for (const frame of [452, 454, 456]) {
            const result = planParamBlockGesture({
                clips,
                edits,
                framePeriodMs: FRAME_MS,
                gesture: { kind: "split", blockId: "u", frame },
            });
            expect(result.ok, `frame ${frame}`).toBe(true);
        }
    });

    it("R8: reverse resize changes the correct source edge and never drops the block", () => {
        const clips = [reverseClip()];
        const view = composeParamBlockView("root", clips, EMPTY, FRAME_MS);
        const block = view.blocks[0]!;
        expect([block.startFrame, block.endFrameExclusive]).toEqual([280, 360]);
        for (const [edge, target] of [
            ["right", 340],
            ["right", 380],
            ["left", 260],
            ["left", 240],
        ] as const) {
            const result = planParamBlockGesture({
                clips,
                edits: EMPTY,
                framePeriodMs: FRAME_MS,
                gesture: { kind: "resize", blockId: block.id, edge, targetFrame: target },
            });
            expect(result.ok, `${edge} ${target}`).toBe(true);
            if (!result.ok) continue;
            expect(result.plan.targetSpan).not.toBeNull();
            if (edge === "right") {
                expect(result.plan.targetSpan?.endFrameExclusive).toBe(target);
            } else {
                expect(result.plan.targetSpan?.startFrame).toBe(target);
            }
        }
    });
});

describe("paramBlockGesture — multi clip / multi member", () => {
    it("R4: cross-clip resize works on both edges", () => {
        const clips = crossClips();
        const edits = crossEdits();
        const view = composeParamBlockView("root", clips, edits, FRAME_MS);
        expect(view.blocks[0]!.startFrame).toBe(0);
        expect(view.blocks[0]!.endFrameExclusive).toBe(1600);
        for (const [edge, target] of [
            ["right", 1580],
            ["left", 20],
        ] as const) {
            const result = planParamBlockGesture({
                clips,
                edits,
                framePeriodMs: FRAME_MS,
                gesture: { kind: "resize", blockId: "cross", edge, targetFrame: target },
            });
            expect(result.ok, `${edge} ${target}`).toBe(true);
            if (!result.ok) continue;
            if (edge === "right") {
                expect(result.plan.targetSpan?.endFrameExclusive).toBe(target);
            } else {
                expect(result.plan.targetSpan?.startFrame).toBe(target);
            }
        }
    });

    it("R6: cross-clip split keeps projected member order", () => {
        const clips = crossClips();
        const edits = crossEdits();
        const result = planParamBlockGesture({
            clips,
            edits,
            framePeriodMs: FRAME_MS,
            gesture: { kind: "split", blockId: "cross", frame: 400 },
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        const users = result.plan.viewAfter.blocks.filter((block) => block.kind !== "base");
        expect(users).toHaveLength(2);
        expect(users[0]!.members).toHaveLength(1);
        expect(users[0]!.members[0]!.clipId).toBe("a");
        expect(users[1]!.members).toHaveLength(2);
        expect(users[1]!.members.some((member) => member.clipId === "b")).toBe(true);
    });

    it("R5: splitting at a member boundary produces two user blocks, not a 1-frame sliver", () => {
        const clips = [plainClip];
        const edits = twoMemberEdits();
        const result = planParamBlockGesture({
            clips,
            edits,
            framePeriodMs: FRAME_MS,
            gesture: { kind: "split", blockId: "user", frame: 100 },
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        const users = result.plan.viewAfter.blocks.filter((block) => block.kind !== "base");
        expect(users).toHaveLength(2);
        expect(users[0]!.startFrame).toBe(0);
        expect(users[1]!.endFrameExclusive).toBe(200);
    });

    it("R7: multi-member shrink stops exactly at the target frame", () => {
        const clips = [plainClip];
        const edits = twoMemberEdits();
        for (const [edge, target] of [
            ["right", 50],
            ["right", 75],
            ["left", 150],
            ["left", 125],
        ] as const) {
            const result = planParamBlockGesture({
                clips,
                edits,
                framePeriodMs: FRAME_MS,
                gesture: { kind: "resize", blockId: "user", edge, targetFrame: target },
            });
            expect(result.ok, `${edge} ${target}`).toBe(true);
            if (!result.ok) continue;
            if (edge === "right") {
                expect(result.plan.targetSpan?.endFrameExclusive).toBe(target);
            } else {
                expect(result.plan.targetSpan?.startFrame).toBe(target);
            }
        }
    });
});

describe("paramBlockGesture — base underlay / identity", () => {
    const baseClip: ParamBlockSourceClip = {
        ...plainClip,
        syllableSpansSec: [{ startSec: 0, endSec: 1 }],
    };

    it("R9/R10: manual block leaves base remainders; deleting it restores the base", () => {
        const base = composeParamBlockView("root", [baseClip], EMPTY, FRAME_MS);
        expect(base.blocks).toHaveLength(1);
        const edits: ParamBlockEditSet = {
            schemaVersion: 1,
            revision: 1,
            blocks: [
                {
                    id: "manual",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0.4,
                                sourceEndSec: 0.6,
                                label: "manual",
                            },
                        ],
                    },
                    manual: true,
                    value: 60,
                    createdSeq: 1,
                },
            ],
        };
        const view = composeParamBlockView("root", [baseClip], edits, FRAME_MS);
        expect(view.blocks.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [0, 80],
            [80, 120],
            [120, 200],
        ]);
        const target = view.blocks.find((block) => block.id === "manual")!;
        const deleted = deleteProjectedBlocks([baseClip], edits, [target], FRAME_MS);
        expect(deleted.edits.blocks.filter((block) => block.deleted)).toHaveLength(0);
        const after = composeParamBlockView("root", [baseClip], deleted.edits, FRAME_MS);
        expect(after.blocks).toHaveLength(1);
        expect([after.blocks[0]!.startFrame, after.blocks[0]!.endFrameExclusive]).toEqual([0, 200]);
    });

    it("base-derived block shrinks to blank, deletes to blank, restores", () => {
        const base = composeParamBlockView("root", [baseClip], EMPTY, FRAME_MS);
        const block = base.blocks[0]!;
        const resized = planParamBlockGesture({
            clips: [baseClip],
            edits: EMPTY,
            framePeriodMs: FRAME_MS,
            gesture: { kind: "resize", blockId: block.id, edge: "right", targetFrame: 50 },
        });
        expect(resized.ok).toBe(true);
        if (!resized.ok) return;
        const view = composeParamBlockView("root", [baseClip], resized.plan.edits, FRAME_MS);
        expect(view.blocks).toHaveLength(1);
        expect(view.blocks[0]!.kind).toBe("user");
        expect([view.blocks[0]!.startFrame, view.blocks[0]!.endFrameExclusive]).toEqual([0, 50]);

        const target = view.blocks[0]!;
        const deleted = deleteProjectedBlocks([baseClip], resized.plan.edits, [target], FRAME_MS);
        const after = composeParamBlockView("root", [baseClip], deleted.edits, FRAME_MS);
        expect(after.blocks).toHaveLength(0);
        const restored = restoreDeletedBlocks(deleted.edits, deleted.deleted);
        const restoredView = composeParamBlockView("root", [baseClip], restored, FRAME_MS);
        expect(restoredView.blocks).toHaveLength(1);
        expect(restoredView.blocks[0]!.kind).toBe("user");
    });

    it("R11: create uses the caller block id", () => {
        const created = createUserBlock([plainClip], EMPTY, 20, 80, 61, FRAME_MS, "want");
        expect(created.blocks.map((block) => block.id)).toEqual(["want"]);
        const result = createParamBlock({
            state: { ...EMPTY_PARAM_BLOCK_STATE, editSet: EMPTY, framePeriodMs: FRAME_MS },
            clips: [plainClip],
            startSec: 0.1,
            endSec: 0.4,
            value: 61,
            blockId: "want2",
        });
        expect(result.ok).toBe(true);
        if (result.ok) expect(result.blockId).toBe("want2");
    });

    it("R12: empty syllable spans fall back to MIDI base blocks", () => {
        const clip: ParamBlockSourceClip = {
            ...plainClip,
            syllableSpansSec: [],
            midiNoteData: [{ startSec: 0.2, endSec: 0.6, note: 60 }],
        };
        const view = composeParamBlockView("root", [clip], EMPTY, FRAME_MS);
        expect(view.blocks).toHaveLength(1);
        expect(view.blocks[0]!.kind).toBe("base");
        expect(view.blocks[0]!.members[0]!.label).toBe("midiNote");
    });
});
