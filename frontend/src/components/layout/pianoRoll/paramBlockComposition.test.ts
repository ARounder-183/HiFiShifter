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

    it("orders crossing clips by start time and keeps the later clip's left-cut block", () => {
        const a: ParamBlockSourceClip = {
            id: "A",
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
        const b: ParamBlockSourceClip = {
            id: "B",
            trackId: "root",
            startSec: 0.2,
            lengthSec: 1,
            sourceStartSec: 0,
            sourceEndSec: 1,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            syllableSpansSec: [{ startSec: 0, endSec: 0.5 }],
        };
        const edits: ParamBlockEditSet = { schemaVersion: 2, revision: 0, blocks: [] };
        const ab = composeParamBlockView("root", [a, b], edits, 5);
        const ba = composeParamBlockView("root", [b, a], edits, 5);
        const ranges = (view: typeof ab) =>
            view.blocks.map((block) => [block.startFrame, block.endFrameExclusive]);
        expect(ranges(ab)).toEqual([
            [0, 40],
            [40, 140],
            [140, 200],
        ]);
        expect(ranges(ba)).toEqual(ranges(ab));
        expect(ab.layoutKey).toBe(ba.layoutKey);
        expect(
            ab.blocks.some((block) => block.startFrame === 40 && block.endFrameExclusive === 140),
        ).toBe(true);
        for (let i = 1; i < ab.blocks.length; i += 1) {
            expect(ab.blocks[i - 1]!.endFrameExclusive).toBeLessThanOrEqual(
                ab.blocks[i]!.startFrame,
            );
        }
    });

    it("uses id as the tie-break for clips with the same start", () => {
        const base = (id: string): ParamBlockSourceClip => ({
            id,
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            syllableSpansSec: [{ startSec: 0, endSec: 1 }],
        });
        const edits: ParamBlockEditSet = { schemaVersion: 2, revision: 0, blocks: [] };
        const ab = composeParamBlockView("root", [base("A"), base("B")], edits, 5);
        const ba = composeParamBlockView("root", [base("B"), base("A")], edits, 5);
        expect(ab.layoutKey).toBe(ba.layoutKey);
        expect(ab.blocks.map((block) => block.members[0]?.clipId)).toEqual(
            ba.blocks.map((block) => block.members[0]?.clipId),
        );
    });

    it("keeps all three blocks non-overlapping under a three-way crossing", () => {
        const mk = (id: string, startSec: number, lengthSec: number): ParamBlockSourceClip => ({
            id,
            trackId: "root",
            startSec,
            lengthSec,
            sourceStartSec: 0,
            sourceEndSec: lengthSec,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            syllableSpansSec: [{ startSec: 0, endSec: lengthSec }],
        });
        const view = composeParamBlockView(
            "root",
            [mk("A", 0, 2), mk("B", 0.2, 1), mk("C", 0.4, 0.4)],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        for (let i = 1; i < view.blocks.length; i += 1) {
            expect(view.blocks[i - 1]!.endFrameExclusive).toBeLessThanOrEqual(
                view.blocks[i]!.startFrame,
            );
        }
        expect(
            view.blocks.some((block) => block.startFrame === 80 && block.endFrameExclusive === 160),
        ).toBe(true);
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

    it("narrows a shared base cover to each loop run instead of copying it wholesale", () => {
        const loopClip: ParamBlockSourceClip = {
            id: "loop",
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 0.5,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 0.5,
            syllableSpansSec: [{ startSec: 0, endSec: 0.5 }],
        };
        const edits: ParamBlockEditSet = {
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "multi-run",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "loop",
                                takeId: "loop_take_1",
                                cycle: 0,
                                sourceStartSec: 0.1,
                                sourceEndSec: 0.2,
                                label: "manual",
                            },
                            {
                                clipId: "loop",
                                takeId: "loop_take_1",
                                cycle: 1,
                                sourceStartSec: 0.1,
                                sourceEndSec: 0.2,
                                label: "manual",
                            },
                        ],
                    },
                    createdSeq: 1,
                    baseCover: {
                        full: true,
                        spans: [
                            {
                                clipId: "loop",
                                takeId: "loop_take_1",
                                cycle: 0,
                                sourceStartSec: 0,
                                sourceEndSec: 0.5,
                                label: "syllable",
                            },
                            {
                                clipId: "loop",
                                takeId: "loop_take_1",
                                cycle: 1,
                                sourceStartSec: 0,
                                sourceEndSec: 0.5,
                                label: "syllable",
                            },
                        ],
                    },
                },
            ],
        };
        const runs = composeParamBlockView("root", [loopClip], edits, 5).blocks.filter(
            (block) => block.userBlockId === "multi-run",
        );
        expect(runs).toHaveLength(2);
        for (const run of runs) {
            expect(run.baseCover?.full).toBe(false);
            expect(run.baseCover?.spans).toHaveLength(1);
            expect(run.baseCover?.spans[0]?.sourceStartSec).toBeCloseTo(0.1, 9);
            expect(run.baseCover?.spans[0]?.sourceEndSec).toBeCloseTo(0.2, 9);
        }
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

    it("keeps a left-cut note at the Clip edge with its source note", () => {
        const trimmed: ParamBlockSourceClip = {
            id: "clip",
            trackId: "root",
            startSec: 0,
            lengthSec: 1.4,
            sourceStartSec: 0.3,
            sourceEndSec: 1.7,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            syllableSpansSec: [
                { startSec: 0.2, endSec: 0.8, note: 60 },
                { startSec: 0.9, endSec: 1.4, note: 62 },
            ],
            syllableAnalysisWindowSec: [0.2, 1.7],
        };
        const view = composeParamBlockView(
            "root",
            [trimmed],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        expect(view.analysisPending).toBe(false);
        expect(view.blocks[0]?.startFrame).toBe(0);
        expect(view.blocks[0]?.endFrameExclusive).toBe(100);
        expect(view.blocks[0]?.cutStart).toBe(true);
        expect(view.blocks[0]?.sourceNote).toBe(60);
    });

    it("trims overlapping spans instead of dropping the earlier left-cut span", () => {
        const overlapped: ParamBlockSourceClip = {
            ...clip,
            syllableSpansSec: [
                { startSec: 0, endSec: 0.6, note: 60 },
                { startSec: 0.4, endSec: 0.9, note: 62 },
            ],
        };
        const view = composeParamBlockView(
            "root",
            [overlapped],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        expect(view.blocks.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [0, 120],
            [120, 180],
        ]);
    });

    it("full cover only suppresses the base range it still geometrically covers", () => {
        const base: ParamBlockSourceClip = {
            ...clip,
            syllableSpansSec: [{ startSec: 0.9, endSec: 1.6 }],
        };
        const edits: ParamBlockEditSet = {
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "partial",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0.5,
                                sourceEndSec: 1.2,
                                label: "syllable",
                            },
                        ],
                    },
                    createdSeq: 1,
                    baseCover: {
                        full: true,
                        spans: [
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0.5,
                                sourceEndSec: 1.2,
                                label: "syllable",
                            },
                        ],
                    },
                },
            ],
        };
        const view = composeParamBlockView("root", [base], edits, 5);
        expect(view.blocks.some((block) => block.kind === "base" && block.startFrame === 240)).toBe(
            true,
        );
    });

    it("exports a stable source key for every derived block", () => {
        const single = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        for (const block of single.blocks) {
            expect(block.sourceKey).toMatch(/^clip\|/);
        }
        const crossing = composeParamBlockView(
            "root",
            [
                { ...clip, id: "A" },
                { ...clip, id: "B", startSec: 0.5, lengthSec: 1 },
            ],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        expect(crossing.blocks.every((block) => block.sourceKey !== undefined)).toBe(true);
        expect(new Set(crossing.blocks.map((block) => block.sourceKey)).size).toBeGreaterThan(1);
    });

    it("keeps a one-frame base remainder with a positive edit window", () => {
        const oneFrame: ParamBlockSourceClip = {
            ...clip,
            syllableSpansSec: [{ startSec: 0, endSec: 0.4, note: 64 }],
        };
        const edits: ParamBlockEditSet = {
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "cover",
                    geometry: { kind: "root", span: { startSec: 0, endSec: 0.396 } },
                    createdSeq: 1,
                },
            ],
        };
        const view = composeParamBlockView("root", [oneFrame], edits, 5);
        expect(
            view.blocks.some(
                (block) =>
                    block.kind === "base" &&
                    block.endFrameExclusive - block.startFrame === 1 &&
                    block.members[0]!.sourceEndSec > block.members[0]!.sourceStartSec,
            ),
        ).toBe(true);
    });

    it("crossing clips keep the upper block and expose lower remainders without holes", () => {
        const lower: ParamBlockSourceClip = {
            id: "A",
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            syllableSpansSec: [{ startSec: 0, endSec: 1 }],
            syllableAnalysisWindowSec: [0, 2],
        };
        const upper: ParamBlockSourceClip = {
            id: "B",
            trackId: "root",
            startSec: 0.2,
            lengthSec: 0.6,
            sourceStartSec: 0.7,
            sourceEndSec: 1.3,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            syllableSpansSec: [{ startSec: 0.4, endSec: 1.0 }],
            syllableAnalysisWindowSec: [0.4, 1.3],
        };
        const view = composeParamBlockView(
            "root",
            [lower, upper],
            { schemaVersion: 2, revision: 0, blocks: [] },
            5,
        );
        const ranges = view.blocks.map((block) => [block.startFrame, block.endFrameExclusive]);
        expect(ranges).toEqual([
            [0, 40],
            [40, 100],
            [100, 200],
        ]);
        expect(view.blocks[1]?.cutStart).toBe(true);
    });
});
