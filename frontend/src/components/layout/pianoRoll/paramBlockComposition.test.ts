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
    return { schemaVersion: 1, revision: 1, blocks: [block] };
}

describe("paramBlockComposition", () => {
    it("derives base blocks from Take spans and suppresses them with a merged user block", () => {
        const base = composeParamBlockView(
            "root",
            [clip],
            { schemaVersion: 1, revision: 0, blocks: [] },
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

    it("可逆编组才是 grouped，且与成员数无关", () => {
        const edits = mergedEdits();
        const block = edits.blocks[0];
        if (block === undefined) throw new Error("expected a block");
        const grouped = composeParamBlockView(
            "root",
            [clip],
            { ...edits, blocks: [{ ...block, groupedFrom: [] }] },
            5,
        );
        expect(grouped.blocks).toHaveLength(1);
        expect(grouped.blocks[0]?.kind).toBe("grouped");
        // 单成员的可逆编组同样是 grouped（判据是快照，不是成员数）。
        if (block.geometry.kind !== "take") throw new Error("expected a take block");
        const first = block.geometry.members[0];
        if (first === undefined) throw new Error("expected a take member");
        const single: UserParamBlock = {
            ...block,
            geometry: { kind: "take", members: [first] },
            groupedFrom: [block],
        };
        const singleView = composeParamBlockView("root", [clip], { ...edits, blocks: [single] }, 5);
        expect(singleView.blocks[0]?.kind).toBe("grouped");
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
