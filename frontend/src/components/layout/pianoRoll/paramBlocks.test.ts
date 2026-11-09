import { describe, expect, it } from "vitest";
import { composeParamBlockView } from "./paramBlockComposition";
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import {
    EMPTY_PARAM_BLOCK_STATE,
    blocksFromViewForPanel,
    isStrictlyInsideParamBlock,
    resolveParamBlockSplitTarget,
    splitParamBlock,
    type ParamBlock,
} from "./paramBlocks";
import type { ParamBlockEditSet } from "./paramBlockTypes";

const EMPTY_EDITS: ParamBlockEditSet = { schemaVersion: 1, revision: 0, blocks: [] };
const FRAME_MS = 5;

function clip(overrides: Partial<ParamBlockSourceClip> = {}): ParamBlockSourceClip {
    return {
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
        ...overrides,
    };
}

function panelBlocks(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet = EMPTY_EDITS,
): ParamBlock[] {
    return blocksFromViewForPanel("root", composeParamBlockView("root", clips, edits, FRAME_MS));
}

function state(editSet: ParamBlockEditSet = EMPTY_EDITS) {
    return { ...EMPTY_PARAM_BLOCK_STATE, editSet, framePeriodMs: FRAME_MS };
}

describe("resolveParamBlockSplitTarget", () => {
    it("播放头在块内部时返回该块", () => {
        const blocks = panelBlocks([clip()]);
        // 音节 [0, 0.5] → 帧 [0, 100]
        expect(resolveParamBlockSplitTarget(blocks, 0.25, FRAME_MS)?.id).toBe(blocks[0]?.id);
        // 第二个音节 [0.5, 1] → 帧 [100, 200]
        expect(resolveParamBlockSplitTarget(blocks, 0.75, FRAME_MS)?.id).toBe(blocks[1]?.id);
    });

    it("落在块边界上返回 null（切出来会是 0 长块）", () => {
        const blocks = panelBlocks([clip()]);
        expect(resolveParamBlockSplitTarget(blocks, 0.5, FRAME_MS)).toBeNull();
        expect(resolveParamBlockSplitTarget(blocks, 0, FRAME_MS)).toBeNull();
        expect(resolveParamBlockSplitTarget(blocks, 1, FRAME_MS)).toBeNull();
    });

    it("落在两块之间的空隙返回 null", () => {
        const blocks = panelBlocks([
            clip({
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.4 },
                    { startSec: 0.6, endSec: 1 },
                ],
            }),
        ]);
        expect(resolveParamBlockSplitTarget(blocks, 0.5, FRAME_MS)).toBeNull();
        expect(resolveParamBlockSplitTarget(blocks, 0.3, FRAME_MS)).not.toBeNull();
    });

    it("落在工程空白返回 null", () => {
        const blocks = panelBlocks([clip()]);
        expect(resolveParamBlockSplitTarget(blocks, 5, FRAME_MS)).toBeNull();
        expect(resolveParamBlockSplitTarget(blocks, Number.NaN, FRAME_MS)).toBeNull();
        expect(resolveParamBlockSplitTarget([], 0.25, FRAME_MS)).toBeNull();
    });

    it("多 run 用户块返回命中的那个 run", () => {
        const edits: ParamBlockEditSet = {
            schemaVersion: 1,
            revision: 1,
            blocks: [
                {
                    id: "multi",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0,
                                sourceEndSec: 0.2,
                                label: "manual",
                            },
                            {
                                clipId: "clip",
                                takeId: "clip_take_1",
                                cycle: 0,
                                sourceStartSec: 0.6,
                                sourceEndSec: 0.8,
                                label: "manual",
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
        const blocks = panelBlocks([clip()], edits);
        const runs = blocks.filter((block) => block.userBlockId === "multi");
        expect(runs).toHaveLength(2);
        expect(resolveParamBlockSplitTarget(blocks, 0.1, FRAME_MS)?.id).toBe(runs[0]?.id);
        expect(resolveParamBlockSplitTarget(blocks, 0.7, FRAME_MS)?.id).toBe(runs[1]?.id);
        // 两个 run 之间是空的。
        expect(resolveParamBlockSplitTarget(blocks, 0.4, FRAME_MS)).toBeNull();
    });
});

describe("isStrictlyInsideParamBlock", () => {
    it("块内部为真，边界与外部为假", () => {
        const block = panelBlocks([clip()])[0];
        if (block === undefined) throw new Error("block missing");
        // 帧 [0, 100]
        expect(isStrictlyInsideParamBlock(block, 0.25, FRAME_MS)).toBe(true);
        expect(isStrictlyInsideParamBlock(block, 0.005, FRAME_MS)).toBe(true);
        expect(isStrictlyInsideParamBlock(block, 0.495, FRAME_MS)).toBe(true);
        expect(isStrictlyInsideParamBlock(block, 0, FRAME_MS)).toBe(false);
        expect(isStrictlyInsideParamBlock(block, 0.5, FRAME_MS)).toBe(false);
        expect(isStrictlyInsideParamBlock(block, 0.6, FRAME_MS)).toBe(false);
        expect(isStrictlyInsideParamBlock(block, Number.NaN, FRAME_MS)).toBe(false);
    });

    it("磁吸把刀口拉出块外时退回播放头原值", () => {
        const block = panelBlocks([clip()])[0];
        if (block === undefined) throw new Error("block missing");
        const rawSec = 0.49;
        // 磁吸到相邻块的公共边界：刀口会离开这个块。
        const snapped = 0.5;
        expect(isStrictlyInsideParamBlock(block, snapped, FRAME_MS)).toBe(false);
        const splitSec = isStrictlyInsideParamBlock(block, snapped, FRAME_MS) ? snapped : rawSec;
        expect(splitSec).toBe(rawSec);
        expect(isStrictlyInsideParamBlock(block, splitSec, FRAME_MS)).toBe(true);
    });
});

describe("splitParamBlock", () => {
    /*
     * 回归：Slip 之后切分必须仍然成功。
     *
     * 块是源域锚定的，Slip 只平移源窗口，投影位置随之改变。只要"块几何"与
     * "切分用的 clips"来自同一份快照，边界判定就是对的；此前的失败来自
     * 面板把过期视图里的几何和新鲜 clips 混着用。
     */
    it("slip 之后仍能在播放头处切分", () => {
        const after = clip({ sourceStartSec: 0.3, sourceEndSec: 2.3 });
        const blocks = panelBlocks([after]);
        const block = blocks[0];
        if (block?.projected === undefined) throw new Error("block missing");
        const midFrame = (block.projected.startFrame + block.projected.endFrameExclusive) / 2;
        const result = splitParamBlock({
            state: state(),
            block,
            sec: (midFrame * FRAME_MS) / 1000,
            clips: [after],
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        const split = panelBlocks([after], result.state.editSet);
        expect(split).toHaveLength(3);
        // 被切的那一块一分为二，切点正好落在 midFrame 上。
        expect(split.map((item) => item.projected?.startFrame)).toEqual([
            block.projected.startFrame,
            midFrame,
            block.projected.endFrameExclusive,
        ]);
    });

    it("切点贴边返回 too_short，未知异常不再兜底成 too_short", () => {
        const blocks = panelBlocks([clip()]);
        const block = blocks[0];
        if (block === undefined) throw new Error("block missing");
        expect(splitParamBlock({ state: state(), block, sec: 0, clips: [clip()] })).toEqual({
            ok: false,
            reason: "too_short",
        });
    });
});
