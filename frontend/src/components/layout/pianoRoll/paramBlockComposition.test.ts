import { describe, expect, it } from "vitest";
import { composeParamBlockView } from "./paramBlockComposition";
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import type { ParamBlockEditSet, UserParamBlock } from "./paramBlockTypes";

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

function mergedEdits(): ParamBlockEditSet {
    const block: UserParamBlock = {
        id: "grouped",
        geometry: {
            kind: "take",
            members: [
                {
                    clipId: "clip",
                    takeId: "clip_take_1",
                    cycle: 0,
                    sourceStartSec: 0,
                    sourceEndSec: 0.5,
                    syllableId: "span:clip_take_1:0:0",
                    label: "syllable",
                },
                {
                    clipId: "clip",
                    takeId: "clip_take_1",
                    cycle: 0,
                    sourceStartSec: 0.5,
                    sourceEndSec: 1,
                    syllableId: "span:clip_take_1:1:500",
                    label: "syllable",
                },
            ],
        },
        createdSeq: 1,
    };
    return { schemaVersion: 2, revision: 1, blocks: [block] };
}

describe("paramBlockComposition", () => {
    it("derives base blocks from Take spans and suppresses them with a merged user block", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        expect(base.blocks.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [0, 100],
            [100, 200],
        ]);
        const merged = composeParamBlockView("root", [clip], mergedEdits(), 5);
        expect(merged.blocks).toHaveLength(1);
        /*
         * 两个相邻成员合成一个 run 是**几何事实**，不是可逆编组：它照样可以切分 /
         * 伸缩（内核只拒绝 `grouped_from.is_some()`）。标成 `"grouped"` 会让菜单置灰、
         * 手柄消失 —— "播放光标确实在块内，却切不了"。
         */
        expect(merged.blocks[0]?.kind).toBe("user");
        expect(merged.blocks[0]?.startFrame).toBe(0);
        expect(merged.blocks[0]?.endFrameExclusive).toBe(200);
    });

    it("hashes clip-level analysis and duration, but not the edit revision", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const durationChanged = composeParamBlockView(
            "root",
            [{ ...clip, contentDurationSec: 3 }],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const analysisChanged = composeParamBlockView(
            "root",
            [
                {
                    ...clip,
                    syllableSpansSec: [
                        { startSec: 0, endSec: 0.25 },
                        { startSec: 0.25, endSec: 1 },
                    ],
                },
            ],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const midiChanged = composeParamBlockView(
            "root",
            [{ ...clip, midiNoteData: [{ startSec: 0, endSec: 0.5, note: 60 }] }],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const revisionChanged = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 9, blocks: [] },
            5,
        );
        expect(durationChanged.layoutKey).not.toBe(base.layoutKey);
        expect(analysisChanged.layoutKey).not.toBe(base.layoutKey);
        expect(midiChanged.layoutKey).not.toBe(base.layoutKey);
        expect(revisionChanged.layoutKey).toBe(base.layoutKey);
    });

    it("组容器产生一个 grouped 块，且不复制合并成员", () => {
        const edits = mergedEdits();
        const block = edits.blocks[0];
        if (block === undefined || block.geometry.kind !== "take") {
            throw new Error("expected a take block");
        }
        const [first, second] = block.geometry.members;
        if (first === undefined || second === undefined) throw new Error("expected two members");
        const child = (id: string, member: typeof first): UserParamBlock => ({
            ...block,
            id,
            geometry: { kind: "take", members: [member] },
        });
        const grouped: UserParamBlock = {
            ...block,
            id: "group",
            geometry: {
                kind: "group",
                children: [child("child-a", first), child("child-b", second)],
            },
        };
        const view = composeParamBlockView("root", [clip], { ...edits, blocks: [grouped] }, 5);
        expect(view.blocks).toHaveLength(1);
        expect(view.blocks[0]?.kind).toBe("grouped");
        expect(view.blocks[0]?.startFrame).toBe(0);
        expect(view.blocks[0]?.endFrameExclusive).toBe(200);
    });

    it("projects a root group as one run per child instead of a bounding box", () => {
        const edits: ParamBlockEditSet = {
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "root-group",
                    geometry: {
                        kind: "group",
                        children: [
                            {
                                id: "root-a",
                                geometry: { kind: "root", span: { startSec: 0.1, endSec: 0.3 } },
                                createdSeq: 1,
                            },
                            {
                                id: "root-b",
                                geometry: { kind: "root", span: { startSec: 0.5, endSec: 0.7 } },
                                createdSeq: 2,
                            },
                        ],
                    },
                    createdSeq: 3,
                },
            ],
        };
        const view = composeParamBlockView("root", [], edits, 5);
        expect(view.blocks).toHaveLength(2);
        expect(view.blocks.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [20, 60],
            [100, 140],
        ]);
        expect(view.blocks.every((block) => block.userBlockId === "root-group")).toBe(true);
    });
    it("layoutKey changes when the active take changes", () => {
        const take = (id: string) => ({
            id,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
        });
        const base = composeParamBlockView(
            "root",
            [{ ...clip, activeTakeId: "a", takes: [take("a"), take("b")] }],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const switched = composeParamBlockView(
            "root",
            [{ ...clip, activeTakeId: "b", takes: [take("a"), take("b")] }],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        expect(base.layoutKey).not.toBe(switched.layoutKey);
    });

    it("keeps cycle>0 members when a loop span crosses the anchor", () => {
        const loopClip: ParamBlockSourceClip = {
            id: "loop",
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0.2,
            sourceEndSec: 0.8,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 0.6,
        };
        const edits: ParamBlockEditSet = {
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "cross-anchor",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "loop",
                                takeId: "loop_take_1",
                                cycle: 0,
                                sourceStartSec: 0.1,
                                sourceEndSec: 0.3,
                                label: "manual",
                            },
                        ],
                    },
                    createdSeq: 1,
                },
            ],
        };
        const view = composeParamBlockView("root", [loopClip], edits, 5);
        const user = view.blocks.filter((block) => block.userBlockId === "cross-anchor");
        expect(user).toHaveLength(1);
        expect(user[0]?.members.map((member) => member.cycle)).toEqual([0, 1]);
        expect(view.orphanCount).toBe(0);
    });

    it("moves a merged block when the Take source window slips", () => {
        const slipped = { ...clip, startSec: 1, sourceStartSec: 0.5, sourceEndSec: 2.5 };
        const edits = mergedEdits();
        const block = edits.blocks[0];
        if (block?.geometry.kind !== "take") throw new Error("expected take block");
        const shifted = {
            ...edits,
            blocks: [
                {
                    ...block,
                    geometry: {
                        kind: "take" as const,
                        members: block.geometry.members.map((member) => ({
                            ...member,
                            sourceStartSec: member.sourceStartSec + 0.5,
                            sourceEndSec: member.sourceEndSec + 0.5,
                        })),
                    },
                },
            ],
        };
        const view = composeParamBlockView("root", [slipped], shifted, 5);
        expect(view.blocks).toHaveLength(1);
        expect(view.blocks[0]?.startFrame).toBe(200);
        expect(view.blocks[0]?.endFrameExclusive).toBe(400);
    });
});
