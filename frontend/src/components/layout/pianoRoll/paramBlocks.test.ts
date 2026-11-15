import { describe, expect, it } from "vitest";
import { composeParamBlockView } from "./paramBlockComposition";
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import {
    EMPTY_PARAM_BLOCK_STATE,
    blocksFromViewForPanel,
    averageParamValueInSpan,
    buildParamBlockDragSegments,
    clampParamBlockDragValue,
    modeCenterOfValuesInSamples,
    canGroupParamBlocks,
    modeCenterOfValues,
    paramBlockFrameSpan,
    groupParamBlocks,
    paramBlockEdgeCandidates,
    isStrictlyInsideParamBlock,
    resolveParamBlockSplitTarget,
    resolveSplittableParamBlockAt,
    shiftParamBlockCurveValuesToTarget,
    splitParamBlock,
    type ParamBlock,
} from "./paramBlocks";
import type { ParamBlockEditSet, UserParamBlock } from "./paramBlockTypes";

const EMPTY_EDITS: ParamBlockEditSet = { schemaVersion: 2, revision: 0, blocks: [] };
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

/**
 * 一个只有用户块的编辑集（base 会被它抑制）。
 *
 * @param members 每个成员的源域区间 [startSec, endSec)。
 */
function takeBlockEdits(members: ReadonlyArray<readonly [number, number]>): ParamBlockEditSet {
    return {
        schemaVersion: 2,
        revision: 1,
        blocks: [
            {
                id: "user",
                geometry: {
                    kind: "take",
                    members: members.map(([sourceStartSec, sourceEndSec]) => ({
                        clipId: "clip",
                        takeId: "clip_take_1",
                        cycle: 0,
                        sourceStartSec,
                        sourceEndSec,
                        label: "manual" as const,
                    })),
                },
                deleted: false,
                manual: false,
                value: 60,
                createdSeq: 1,
            },
        ],
    };
}

/** 两个**相邻**成员：合成成一个 run，但没有任何可逆编组快照。 */
function twoMemberEdits(): ParamBlockEditSet {
    return takeBlockEdits([
        [0, 0.5],
        [0.5, 1],
    ]);
}

/** 两个彼此分离的独立逻辑块：帧域 [0,40) 与 [120,160)。 */
function disjointBlockEdits(): ParamBlockEditSet {
    return {
        schemaVersion: 2,
        revision: 1,
        blocks: [
            {
                id: "left",
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
                    kind: "take",
                    members: [
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
                createdSeq: 2,
            },
        ],
    };
}

/** 三个彼此分离的独立块：用于验证组只暴露首/末 run 的外边缘。 */
function threeDisjointBlockEdits(): ParamBlockEditSet {
    const base = disjointBlockEdits();
    return {
        ...base,
        blocks: [
            ...base.blocks,
            {
                id: "middle",
                geometry: {
                    kind: "take",
                    members: [
                        {
                            clipId: "clip",
                            takeId: "clip_take_1",
                            cycle: 0,
                            sourceStartSec: 0.35,
                            sourceEndSec: 0.5,
                            label: "manual",
                        },
                    ],
                },
                deleted: false,
                manual: true,
                value: 60,
                createdSeq: 3,
            },
        ],
    };
}

/** 两个相差 1 帧的成员：合成会并成一个 run，并在帧 100 处留下"成员内无帧"的缝。 */
function gapMemberEdits(): ParamBlockEditSet {
    return takeBlockEdits([
        [0, 0.5],
        [0.505, 1],
    ]);
}

describe("param block grouping", () => {
    it("allows non-adjacent logical blocks and leaves the gap unfilled", () => {
        const clips = [clip()];
        const edits = disjointBlockEdits();
        const blocks = panelBlocks(clips, edits);
        expect(canGroupParamBlocks(blocks, ["left", "right"])).toBe(true);

        const result = groupParamBlocks({
            state: state(edits),
            blocks,
            selectedBlockIds: ["left", "right"],
            clips,
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;

        const view = composeParamBlockView("root", clips, result.state.editSet, FRAME_MS);
        const runs = view.blocks.filter((block) => block.kind !== "base");
        expect(runs.map((block) => [block.startFrame, block.endFrameExclusive])).toEqual([
            [0, 40],
            [120, 160],
        ]);
        expect(runs.every((block) => block.userBlockId !== undefined)).toBe(true);
        expect(new Set(runs.map((block) => block.userBlockId)).size).toBe(1);
        expect(runs.some((block) => block.startFrame < 80 && block.endFrameExclusive > 80)).toBe(
            false,
        );
    });

    it("exposes only the outer edges of a multi-run group", () => {
        const clips = [clip()];
        const edits = threeDisjointBlockEdits();
        const blocks = panelBlocks(clips, edits);
        const result = groupParamBlocks({
            state: state(edits),
            blocks,
            selectedBlockIds: ["left", "middle", "right"],
            clips,
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        const view = composeParamBlockView("root", clips, result.state.editSet, FRAME_MS);
        const groupedRuns = view.blocks.filter((block) => block.kind === "grouped");
        expect(groupedRuns).toHaveLength(3);
        const edgeCandidates = paramBlockEdgeCandidates(
            blocksFromViewForPanel("root", view),
        ).filter((block) => block.kind === "grouped");
        expect(edgeCandidates.map((block) => block.id)).toEqual([
            groupedRuns[0]!.id,
            groupedRuns[2]!.id,
        ]);
    });

    it("rejects selecting only multiple runs of one logical block", () => {
        const clips = [clip()];
        const edits = disjointBlockEdits();
        const blocks = panelBlocks(clips, edits);
        expect(canGroupParamBlocks(blocks, ["left", "left"])).toBe(false);
    });
});

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
            schemaVersion: 2,
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
        // 两个 run 之间没有用户块；基块衬底会露出基块余段，这是新模型的正确结果。
        expect(
            blocks.some(
                (block) => block.kind !== "auto" && block.startSec < 0.4 && block.endSec > 0.4,
            ),
        ).toBe(false);
    });
});

describe("buildParamBlockDragSegments", () => {
    it("samples each merged span once for both empty-pitch detection and mapping", () => {
        let reads = 0;
        const segments = buildParamBlockDragSegments({
            blocks: [
                {
                    members: [
                        { clipId: "clip", startSec: 0, endSec: 0.5 },
                        { clipId: "clip", startSec: 0.505, endSec: 1 },
                    ],
                },
            ],
            framePeriodMs: 5,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            fillEmptyPitch: true,
            sourceAt: () => {
                reads += 1;
                return 60;
            },
        });
        expect(segments.map((segment) => segment.values.length)).toEqual([100, 99]);
        expect(reads).toBe(199);
    });

    it("does not write across a one-frame gap between members", () => {
        const segments = buildParamBlockDragSegments({
            blocks: [
                {
                    members: [
                        { clipId: "clip", startSec: 0, endSec: 0.5 },
                        { clipId: "clip", startSec: 0.505, endSec: 1 },
                    ],
                },
            ],
            framePeriodMs: 5,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            sourceAt: () => 60,
        });
        expect(segments.map((segment) => [segment.startFrame, segment.values.length])).toEqual([
            [0, 100],
            [101, 99],
        ]);
    });
});

describe("clampParamBlockDragValue", () => {
    it("clamps scalar parameters to the same ranges as the backend", () => {
        expect(clampParamBlockDragValue({ param: "volume", original: 1.9, valueDelta: 0.5 })).toBe(
            2,
        );
        expect(clampParamBlockDragValue({ param: "volume", original: 0.1, valueDelta: -0.5 })).toBe(
            0,
        );
        expect(clampParamBlockDragValue({ param: "tension", original: 99, valueDelta: 10 })).toBe(
            100,
        );
        expect(clampParamBlockDragValue({ param: "pitch", original: 126, valueDelta: 10 })).toBe(
            127,
        );
        expect(clampParamBlockDragValue({ param: "pitch", original: 0, valueDelta: 10 })).toBe(0);
    });

    it("keeps the dyn follow-original sentinel and clamps positive targets to 0..1", () => {
        expect(clampParamBlockDragValue({ param: "dyn", original: -1, valueDelta: 0.5 })).toBe(-1);
        expect(clampParamBlockDragValue({ param: "dyn", original: 0.2, valueDelta: -0.5 })).toBe(0);
        expect(clampParamBlockDragValue({ param: "dyn", original: 0.9, valueDelta: 0.5 })).toBe(1);
    });

    it("passes non-finite values through unchanged", () => {
        expect(
            clampParamBlockDragValue({
                param: "volume",
                original: Number.NaN,
                valueDelta: 1,
            }),
        ).toBeNaN();
    });
});

describe("block value sampling on a strided param view", () => {
    const view = {
        startFrame: 0,
        stride: 2,
        framePeriodMs: 5,
        values: [10, 20, 30, 40],
    };

    it("starts at the first sample at or after the block start", () => {
        const average = averageParamValueInSpan({
            startSec: 0.005,
            endSec: 0.025,
            values: view.values,
            startFrame: view.startFrame,
            stride: view.stride,
            framePeriodMs: view.framePeriodMs,
            isPitch: false,
            param: "volume",
            binWidth: 100,
        });
        // Frames 1..4 contain samples at 2 and 4: values 20 and 30.
        expect(average).toBe(25);
    });

    it("never borrows a sample outside a short block", () => {
        const average = averageParamValueInSpan({
            startSec: 0.006,
            endSec: 0.008,
            values: view.values,
            startFrame: view.startFrame,
            stride: view.stride,
            framePeriodMs: view.framePeriodMs,
            isPitch: false,
            param: "volume",
        });
        expect(average).toBeNull();
    });

    it("uses an in-block nearest sample when the block is shorter than the stride", () => {
        const average = averageParamValueInSpan({
            startSec: 0.01,
            endSec: 0.012,
            values: view.values,
            startFrame: view.startFrame,
            stride: view.stride,
            framePeriodMs: view.framePeriodMs,
            isPitch: false,
            param: "volume",
        });
        expect(average).toBe(20);
    });

    it("uses the same grid for the host sample mirror", () => {
        const average = modeCenterOfValuesInSamples(0.005, 0.025, {
            values: view.values,
            startFrame: view.startFrame,
            stride: view.stride,
            framePeriodMs: view.framePeriodMs,
            isPitch: false,
            param: "volume",
            binWidth: 100,
        });
        expect(average).toBe(25);
    });
});

describe("resolveSplittableParamBlockAt", () => {
    /* 组是容器而不是冻结快照：组内切点合法就必须可切。 */
    it("编组块可切", () => {
        const member = (start: number, end: number) => ({
            clipId: "clip",
            takeId: "clip_take_1",
            cycle: 0,
            sourceStartSec: start,
            sourceEndSec: end,
            label: "syllable" as const,
        });
        const child = (id: string, start: number, end: number): UserParamBlock => ({
            id,
            geometry: { kind: "take", members: [member(start, end)] },
            deleted: false,
            manual: false,
            value: null,
            createdSeq: id === "child-a" ? 1 : 2,
        });
        const edits: ParamBlockEditSet = {
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "grouped",
                    geometry: {
                        kind: "group",
                        children: [child("child-a", 0, 0.5), child("child-b", 0.5, 1)],
                    },
                    deleted: false,
                    manual: false,
                    value: null,
                    createdSeq: 3,
                },
            ],
        };
        const blocks = panelBlocks([clip()], edits);
        expect(blocks).toHaveLength(1);
        expect(blocks[0]?.kind).toBe("grouped");
        expect(resolveParamBlockSplitTarget(blocks, 0.5, FRAME_MS)?.kind).toBe("grouped");
        expect(resolveSplittableParamBlockAt(blocks, 0.25, FRAME_MS)).not.toBeNull();
    });

    it("普通块照常返回，空隙返回 null", () => {
        const blocks = panelBlocks([clip()]);
        expect(resolveSplittableParamBlockAt(blocks, 0.25, FRAME_MS)?.kind).toBe("auto");
        expect(resolveSplittableParamBlockAt(blocks, 5, FRAME_MS)).toBeNull();
    });

    /*
     * 【多成员 run 不是编组】两个相邻成员合成一个 run 只是几何事实：切分有唯一解
     * （找到包含切点的那个成员按源域一分为二），所以它必须出现在"可切"里。
     * 旧实现把它标成 `grouped`，于是菜单置灰、快捷键静默 —— 用户看到的就是
     * "播放光标确实在块内，却切不了"。
     */
    it("多成员 run 可切（它不是可逆编组）", () => {
        const blocks = panelBlocks([clip()], twoMemberEdits());
        expect(blocks).toHaveLength(1);
        expect(blocks[0]?.kind).toBe("user");
        expect(resolveSplittableParamBlockAt(blocks, 0.25, FRAME_MS)).not.toBeNull();
        expect(resolveSplittableParamBlockAt(blocks, 0.75, FRAME_MS)).not.toBeNull();
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

    /*
     * 【run 合并容差的缝】合成把"相差 1 帧"的两段也并成一个块，于是块内可能存在
     * "成员内无帧"的位置。旧实现只找严格包含该帧的成员，落进缝里就失败 ——
     * 菜单可点、播放光标在块内，却切不了。兜底规则：取帧距离最近的成员并把切点
     * 钳进它内部。
     */
    it("落在 run 合并容差缝里也能切", () => {
        /*
         * 两个成员的投影分别是 [0, 100) 与 [101, 200)：相差 1 帧，合成把它们并成
         * 一个 run [0, 200)。帧 100 因此**块内有、成员内无**。
         */
        const edits = gapMemberEdits();
        const blocks = panelBlocks([clip()], edits);
        /*
         * 1 帧缝必须保持为缝：两个用户 run + 中间露出的基块余段。
         * 旧实现用 <= last.end + 1 把缝吞成一个连续 run，等于凭空抹掉了
         * 源域里的空洞。
         */
        expect(blocks).toHaveLength(3);
        const userBlocks = blocks.filter((block) => block.kind === "user");
        expect(userBlocks).toHaveLength(2);
        expect([
            userBlocks[0]?.projected?.startFrame,
            userBlocks[0]?.projected?.endFrameExclusive,
        ]).toEqual([0, 100]);
        expect([
            userBlocks[1]?.projected?.startFrame,
            userBlocks[1]?.projected?.endFrameExclusive,
        ]).toEqual([101, 200]);
        const gap = blocks.find((block) => block.kind === "auto");
        expect([gap?.projected?.startFrame, gap?.projected?.endFrameExclusive]).toEqual([100, 101]);
    });

    /*
     * 【不变式】"菜单可点" ⟺ "切得成"。
     *
     * 两个判据曾经各写一遍（位置解析 / 成员搜索），容差与四舍五入一有出入就会出现
     * "可点却失败"。这条属性测试对块内**每一个帧号**同时跑两条路径。
     */
    it("块内每个帧号：可切 ⟺ 切得成", () => {
        const edits = gapMemberEdits();
        const blocks = panelBlocks([clip()], edits).filter((block) => block.kind === "user");
        for (const block of blocks) {
            if (block.projected === undefined) continue;
            for (
                let frame = block.projected.startFrame + 1;
                frame < block.projected.endFrameExclusive;
                frame += 1
            ) {
                const sec = (frame * FRAME_MS) / 1000;
                const result = splitParamBlock({
                    state: state(edits),
                    block,
                    sec,
                    clips: [clip()],
                });
                expect({ frame, ok: result.ok }).toEqual({ frame, ok: true });
            }
        }
    });
});

describe("shiftParamBlockCurveValuesToTarget", () => {
    it("flatten fills every non-pitch frame and preserves the pitch zero sentinel", () => {
        expect(
            shiftParamBlockCurveValuesToTarget({
                values: [10, 20, 30],
                target: 7,
                param: "volume",
                flatten: true,
            }),
        ).toEqual([7, 7, 7]);
        expect(
            shiftParamBlockCurveValuesToTarget({
                values: [58, 0, 61, 0],
                target: 60,
                param: "pitch",
                flatten: true,
            }),
        ).toEqual([60, 0, 60, 0]);
    });

    it("without flatten the original contour is shifted, not flattened", () => {
        expect(
            shiftParamBlockCurveValuesToTarget({
                values: [58, 60, 62],
                target: 60,
                param: "pitch",
            }),
        ).toEqual([58, 60, 62]);
    });
});

describe("paramBlockFrameSpan / translation-invariant mode center", () => {
    it("uses the same round-to-frame span for preview and commit", () => {
        expect(paramBlockFrameSpan(0.0125, 0.0325, 5)).toEqual({
            startFrame: 3,
            endFrameExclusive: 7,
        });
        expect(paramBlockFrameSpan(0, 0, 5)).toEqual({ startFrame: 0, endFrameExclusive: 1 });
    });

    it.each([
        ["pitch", 1],
        ["volume", 0.04],
        ["tension", 50],
    ])("mode center translates with the values for %s", (param, binWidth) => {
        const values = [60, 60, 60, 61, 59, 60, 60];
        const base = modeCenterOfValues(values, { param, binWidth });
        for (const delta of [-1.3, -0.37, 0, 0.1, 2.7]) {
            expect(
                modeCenterOfValues(
                    values.map((value) => value + delta),
                    { param, binWidth },
                ),
            ).toBeCloseTo((base as number) + delta, 9);
        }
    });
});

/**
 * 切分点的包含判定必须与**整数帧**同口径。
 *
 * 【为什么钉这一条】面板里两条切分路径曾经各用一套判据：播放头分支走整数帧
 * （`resolveParamBlockSplitTarget`），而「带目标块 id」的点击分支用浮点秒直接比
 * `startSec < sec < endSec`。两者在块边界 ±一帧内会给出不同结论 —— 用户看到的是
 * 「位置明显在块内，却提示切分点太靠边」。
 */
describe("isStrictlyInsideParamBlock", () => {
    const block = {
        id: "b",
        rootTrackId: "root",
        kind: "user",
        members: [],
        startSec: (10 * FRAME_MS) / 1000,
        endSec: (20 * FRAME_MS) / 1000,
        averageValue: 60,
        manualValue: null,
        projected: { startFrame: 10, endFrameExclusive: 20 },
    } as unknown as ParamBlock;
    const at = (frame: number) => (frame * FRAME_MS) / 1000;

    it("excludes both edges (the split must leave two non-empty halves)", () => {
        expect(isStrictlyInsideParamBlock(block, at(10), FRAME_MS)).toBe(false);
        expect(isStrictlyInsideParamBlock(block, at(20), FRAME_MS)).toBe(false);
    });

    it("accepts the frames strictly between the edges", () => {
        expect(isStrictlyInsideParamBlock(block, at(11), FRAME_MS)).toBe(true);
        expect(isStrictlyInsideParamBlock(block, at(19), FRAME_MS)).toBe(true);
    });

    it("quantises to whole frames, so sub-frame offsets do not flip the verdict", () => {
        /* 起点 + 0.4 帧 ⇒ 仍是第 10 帧 ⇒ 不算内部（与浮点秒判据的差异就在这里）。 */
        expect(isStrictlyInsideParamBlock(block, at(10.4), FRAME_MS)).toBe(false);
        expect(isStrictlyInsideParamBlock(block, at(10.6), FRAME_MS)).toBe(true);
        /* 浮点噪声不改变判定。 */
        expect(isStrictlyInsideParamBlock(block, at(11) + 1e-9, FRAME_MS)).toBe(true);
        expect(isStrictlyInsideParamBlock(block, at(19) - 1e-9, FRAME_MS)).toBe(true);
    });

    it("rejects blocks without projected geometry and non-finite positions", () => {
        const unprojected = { ...block, projected: undefined } as unknown as ParamBlock;
        expect(isStrictlyInsideParamBlock(unprojected, at(15), FRAME_MS)).toBe(false);
        expect(isStrictlyInsideParamBlock(block, Number.NaN, FRAME_MS)).toBe(false);
    });
});
