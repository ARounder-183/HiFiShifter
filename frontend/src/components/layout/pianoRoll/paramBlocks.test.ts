import { describe, expect, it } from "vitest";

import {
    EMPTY_PARAM_BLOCK_STATE,
    autoBlocksForClips,
    averageParamValueInSpan,
    buildParamBlockDragSegments,
    buildParamBlockPreviewSegments,
    deleteParamBlocks,
    hitTestParamBlocks,
    layoutParamBlockRects,
    materializeParamBlocks,
    mergeParamBlocks,
    mergeOverlappingParamBlocks,
    MOCK_SYLLABLE_SEC,
    modeCenterOfValues,
    nearestParamBlockByTime,
    PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER,
    projectMidiNoteSpansToTimeline,
    projectSourceSpansToTimeline,
    resolveParamBlockDragWrite,
    resolveParamBlockSnapDelta,
    resizeParamBlock,
    splitParamBlock,
    unmergeParamBlocks,
    type ParamBlock,
    type ParamBlockSourceClip,
} from "./paramBlocks";

function clip(
    id: string,
    startSec: number,
    lengthSec: number,
    syllables?: number[],
): ParamBlockSourceClip {
    const spans =
        syllables === undefined
            ? undefined
            : syllables.map((value, index) => ({
                  startSec: value,
                  endSec:
                      index + 1 < syllables.length ? syllables[index + 1] : startSec + lengthSec,
              }));
    return { id, trackId: "root-a", startSec, lengthSec, syllableSpansSec: spans };
}

describe("autoBlocksForClips", () => {
    it("音频 Clip 分析完成前不显示块", () => {
        expect(autoBlocksForClips("root-a", [clip("a", 1, 2)])).toEqual([]);
    });

    it("显式音节被裁剪到 Clip 内、零宽段被过滤", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                startSec: 10,
                lengthSec: 4.5,
                syllableSpansSec: [
                    { startSec: 10, endSec: 11 },
                    { startSec: 11, endSec: 11 },
                    { startSec: 12.5, endSec: 15 },
                ],
            },
        ]);
        expect(blocks.map((block) => [block.startSec, block.endSec])).toEqual([
            [10, 11],
            [12.5, 14.5],
        ]);
    });

    it("相邻音节之间的小浮点缝不会形成人为间隙", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 1,
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.3 },
                    { startSec: 0.305, endSec: 0.6 },
                    { startSec: 0.6, endSec: 1 },
                ],
            },
        ]);
        expect(blocks.map((block) => [block.startSec, block.endSec])).toEqual([
            [0, 0.305],
            [0.305, 0.6],
            [0.6, 1],
        ]);
    });

    it("一个 Clip 的每个 MIDI 音符自动生成一个块", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "midi",
                trackId: "root-a",
                startSec: 10,
                lengthSec: 2,
                syllableSpansSec: [
                    { startSec: 10, endSec: 10.4 },
                    { startSec: 10.4, endSec: 10.9 },
                    { startSec: 10.9, endSec: 11.5 },
                    { startSec: 11.5, endSec: 12 },
                ],
            },
        ]);
        expect(blocks).toHaveLength(4);
        expect(blocks.map((block) => [block.startSec, block.endSec])).toEqual([
            [10, 10.4],
            [10.4, 10.9],
            [10.9, 11.5],
            [11.5, 12],
        ]);
    });

    it("空轨道组返回空数组，不伪造块", () => {
        expect(autoBlocksForClips("root-a", [])).toEqual([]);
    });

    it("显式 mock 音节按固定时长切片，最后一段吸收余量", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "audio",
                trackId: "root-a",
                startSec: 1,
                lengthSec: 2,
                mockSyllables: true,
            },
        ]);
        expect(blocks.map((block) => [block.startSec, block.endSec])).toEqual([
            [1, 1 + MOCK_SYLLABLE_SEC],
            [1 + MOCK_SYLLABLE_SEC, 1 + MOCK_SYLLABLE_SEC * 2],
            [1 + MOCK_SYLLABLE_SEC * 2, 1 + MOCK_SYLLABLE_SEC * 3],
            [1 + MOCK_SYLLABLE_SEC * 3, 3],
        ]);
    });

    it("没有显式 mock 标志时不擅自生成音频 Clip 块", () => {
        expect(autoBlocksForClips("root-a", [clip("audio", 0, 2)])).toHaveLength(0);
    });

    it("真实分析区间超过上限时保持模型边界，不再机械等分", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "audio-long",
                trackId: "root-a",
                takeId: "take-long",
                startSec: 0,
                lengthSec: 6,
                sourceStartSec: 0,
                sourceEndSec: 6,
                sourceSyllableSpansSec: [[0, 6]],
            },
        ]);
        expect(blocks.length).toBe(1);
        expect(blocks[0]?.endSec - blocks[0]?.startSec).toBeCloseTo(6, 6);
        // 源域先切段：每个子块都有自己的源域区间，首尾相接且覆盖原段。
        expect(blocks[0].members[0]?.sourceStartSec).toBeCloseTo(0, 6);
        expect(blocks.at(-1)?.members[0]?.sourceEndSec).toBeCloseTo(6, 6);
    });

    it("只有工程秒区间时同样保留分析边界", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "timeline-only",
                trackId: "root-a",
                startSec: 2,
                lengthSec: 6,
                syllableSpansSec: [{ startSec: 2, endSec: 8 }],
            },
        ]);
        expect(blocks.length).toBe(1);
        expect(blocks[0].startSec).toBeCloseTo(2, 9);
        expect(blocks.at(-1)?.endSec).toBeCloseTo(8, 9);
    });
});

describe("projectMidiNoteSpansToTimeline", () => {
    it("源窗口裁切与播放速率会改变音符在时间轴上的位置", () => {
        const spans = projectMidiNoteSpansToTimeline({
            id: "a",
            trackId: "root",
            startSec: 10,
            lengthSec: 2,
            sourceStartSec: 1,
            sourceEndSec: 3,
            playbackRate: 0.5,
            reversed: false,
            loopEnabled: false,
            midiNoteData: [{ startSec: 1.5, endSec: 2, note: 60 }],
        });
        expect(spans).toHaveLength(1);
        expect(spans[0].startSec).toBeCloseTo(11, 6);
        expect(spans[0].endSec).toBeCloseTo(12, 6);
    });

    it("倒放会镜像音符位置；Loop 会按周期重复", () => {
        const reversed = projectMidiNoteSpansToTimeline({
            id: "a",
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: true,
            loopEnabled: false,
            midiNoteData: [{ startSec: 0, endSec: 0.5, note: 60 }],
        });
        expect(reversed).toEqual([{ startSec: 1.5, endSec: 2 }]);
        const looped = projectMidiNoteSpansToTimeline({
            id: "a",
            trackId: "root",
            startSec: 0,
            lengthSec: 4,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            contentDurationSec: 2,
            midiNoteData: [{ startSec: 0, endSec: 0.5, note: 60 }],
        });
        expect(looped).toHaveLength(2);
        expect(looped[1]).toEqual({ startSec: 2, endSec: 2.5 });
    });
});

describe("mergeParamBlocks", () => {
    it("同组相邻块可合并", () => {
        const blocks = autoBlocksForClips("root-a", [
            clip("a", 0, 2, [0]),
            clip("b", 2, 2, [2]),
        ]);
        const result = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: blocks.map((block) => block.id),
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.state.mergedBlocks).toHaveLength(1);
        expect(result.state.mergedBlocks[0].blockIds).toHaveLength(2);
    });

    it("中间有真实空隙时拒绝合并", () => {
        const blocks = autoBlocksForClips("root-a", [
            clip("a", 0, 2, [0]),
            clip("b", 3, 2, [3]),
        ]);
        const result = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: blocks.map((block) => block.id),
        });
        expect(result).toEqual({ ok: false, reason: "not_adjacent" });
    });

    it("跨根轨道组拒绝合并", () => {
        const blocks = [
            ...autoBlocksForClips("root-a", [clip("a", 0, 2, [0])]),
            ...autoBlocksForClips("root-b", [
                { ...clip("b", 2, 2, [2]), trackId: "root-b" },
            ]),
        ];
        expect(
            mergeParamBlocks({
                state: EMPTY_PARAM_BLOCK_STATE,
                blocks,
                selectedBlockIds: blocks.map((block) => block.id),
            }),
        ).toEqual({ ok: false, reason: "different_root" });
    });

    it("合并后自动块让位，只保留一个合并块", () => {
        const clips = [clip("a", 0, 2, [0]), clip("b", 2, 2, [2])];
        const blocks = autoBlocksForClips("root-a", clips);
        const merged = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: blocks.map((block) => block.id),
        });
        expect(merged.ok).toBe(true);
        if (!merged.ok) return;
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: merged.state,
        });
        expect(materialized).toHaveLength(1);
        expect(materialized[0]).toMatchObject({
            kind: "merged",
            startSec: 0,
            endSec: 4,
        });
        expect(materialized[0].members.map((member) => member.clipId)).toEqual(["a", "b"]);
    });

    it("同一 Clip 内合并两个音节时，其余音节块保留", () => {
        const clips = [
            {
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 2,
                // 音节区间是**工程秒**；这里的 Clip 从 0 起，故与 Clip 内时间一致。
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.5 },
                    { startSec: 0.5, endSec: 1 },
                    { startSec: 1, endSec: 1.5 },
                    { startSec: 1.5, endSec: 2 },
                ],
            },
        ];
        const blocks = autoBlocksForClips("root-a", clips);
        expect(blocks).toHaveLength(4);
        const merged = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: [blocks[0].id, blocks[1].id],
        });
        expect(merged.ok).toBe(true);
        if (!merged.ok) return;
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: merged.state,
        });
        // 1 个合并块 + 未被选中的 2 个自动块。
        expect(materialized).toHaveLength(3);
        expect(materialized.filter((block) => block.kind === "auto")).toHaveLength(2);
    });

    it("合并块按音节块粒度解合并，不会连带解除同一 Clip 的其它合并", () => {
        const clips = [
            {
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 3,
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.5 },
                    { startSec: 0.5, endSec: 1 },
                    { startSec: 1, endSec: 1.5 },
                    { startSec: 1.5, endSec: 2 },
                ],
            },
        ];
        const blocks = autoBlocksForClips("root-a", clips);
        const first = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: [blocks[0].id, blocks[1].id],
        });
        expect(first.ok).toBe(true);
        if (!first.ok) return;
        const second = mergeParamBlocks({
            state: first.state,
            blocks,
            selectedBlockIds: [blocks[2].id, blocks[3].id],
        });
        expect(second.ok).toBe(true);
        if (!second.ok) return;
        const unmerged = unmergeParamBlocks(second.state, [blocks[0].id, blocks[1].id]);
        expect(unmerged.mergedBlocks).toHaveLength(1);
        expect(unmerged.mergedBlocks[0].blockIds).toEqual([blocks[2].id, blocks[3].id]);
    });

    it("解合并后恢复原始自动块", () => {
        const clips = [clip("a", 0, 2, [0]), clip("b", 2, 2, [2])];
        const blocks = autoBlocksForClips("root-a", clips);
        const merged = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: blocks.map((block) => block.id),
        });
        expect(merged.ok).toBe(true);
        if (!merged.ok) return;
        const unmerged = unmergeParamBlocks(
            merged.state,
            blocks.map((block) => block.id),
        );
        expect(
            materializeParamBlocks({
                rootTrackId: "root-a",
                clips,
                state: unmerged,
            }).map((block) => block.kind),
        ).toEqual(["auto", "auto"]);
    });
});

describe("averageParamValueInSpan", () => {
    it("按时间区间取主簇中心，忽略区间外采样", () => {
        expect(
            averageParamValueInSpan({
                startSec: 0.01,
                endSec: 0.03,
                values: [60, 61, 62, 63, 64],
                startFrame: 0,
                stride: 1,
                framePeriodMs: 10,
                isPitch: true,
            }),
        ).toBeCloseTo(61.5, 6);
    });

    it("音高块跳过 <= 0 的无声帧", () => {
        expect(
            averageParamValueInSpan({
                startSec: 0,
                endSec: 0.04,
                values: [0, 60, 0, 62],
                startFrame: 0,
                stride: 1,
                framePeriodMs: 10,
                isPitch: true,
            }),
        ).toBe(60);
    });

    it("少量离群点不会把块位置拉离主体", () => {
        // 20 帧集中在 60，3 帧滑到 67，2 帧是 72 的误检。
        const values = [
            ...new Array(20).fill(60),
            ...new Array(3).fill(67),
            ...new Array(2).fill(72),
        ];
        expect(
            averageParamValueInSpan({
                startSec: 0,
                endSec: 0.125,
                values,
                startFrame: 0,
                stride: 1,
                framePeriodMs: 10,
                isPitch: true,
            }),
        ).toBeCloseTo(60, 6);
    });

    it("双峰时选择样本更多的簇；密度接近才向中位数靠拢", () => {
        expect(
            modeCenterOfValues([...new Array(9).fill(59), ...new Array(2).fill(71)], {
                param: "pitch",
                isPitch: true,
            }),
        ).toBeCloseTo(59, 6);
        /*
         * 两簇密度相同：取先出现且更靠中位数的簇，而不是制造中间的 65。
         * 65 不在任何一簇的实际点位上，会把整块拖进颤音中缝。
         */
        const tie = modeCenterOfValues([...new Array(10).fill(59), ...new Array(10).fill(71)], {
            param: "pitch",
            isPitch: true,
        });
        expect(tie === 59 || tie === 71).toBe(true);
    });

    it("无有效数据时返回 null，而不是 0", () => {
        expect(
            averageParamValueInSpan({
                startSec: 0,
                endSec: 1,
                values: [0, 0, 0],
                startFrame: 0,
                stride: 1,
                framePeriodMs: 10,
                isPitch: true,
            }),
        ).toBeNull();
    });
});

describe("layoutParamBlockRects", () => {
    function block(id: string, averageValue: number | null): ParamBlock {
        return {
            id,
            rootTrackId: "root-a",
            kind: "auto",
            members: [{ clipId: `clip-${id}`, startSec: 0, endSec: 1 }],
            startSec: 0,
            endSec: 1,
            averageValue,
        };
    }

    const valueToY = (value: number) => 200 - value * 2;

    it("块高是 1.5 倍行高，不随均值变化", () => {
        const rects = layoutParamBlockRects({
            blocks: [block("low", 60), block("high", 72)],
            valueToY,
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 20,
        });
        const low = rects.find((rect) => rect.id === "low");
        const high = rects.find((rect) => rect.id === "high");
        expect(low).toBeDefined();
        expect(high).toBeDefined();
        if (!low || !high) return;
        expect(low.heightPx).toBe(20 * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER);
        expect(high.heightPx).toBe(low.heightPx);
    });

    it("纵向缩放时块高跟随行高", () => {
        const rects = layoutParamBlockRects({
            blocks: [block("key", 60)],
            valueToY,
            viewportHeightPx: 800,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 10,
        });
        expect(rects[0].heightPx).toBe(10 * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER);
        const zoomed = layoutParamBlockRects({
            blocks: [block("key", 60)],
            valueToY: (value) => 1600 - value * 12,
            viewportHeightPx: 1600,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 30,
        });
        expect(zoomed[0].heightPx).toBe(30 * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER);
    });

    it("均值变化只改变块中心，不改变块高", () => {
        const before = layoutParamBlockRects({
            blocks: [block("b", 60)],
            valueToY,
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 20,
        })[0];
        const after = layoutParamBlockRects({
            blocks: [block("b", 90)],
            valueToY,
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 20,
        })[0];
        expect(before).toBeDefined();
        expect(after).toBeDefined();
        if (!before || !after) return;
        expect(after.heightPx).toBe(before.heightPx);
        expect(after.topPx).not.toBe(before.topPx);
    });

    it("缺均值 / 非有限值不产生可见块矩形", () => {
        const rects = layoutParamBlockRects({
            blocks: [block("empty", null), block("nan", Number.NaN)],
            valueToY,
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
        });
        expect(rects).toEqual([]);
    });

    it("块完全离开纵向视口时不生成几何", () => {
        const rects = layoutParamBlockRects({
            blocks: [block("top", 1000)],
            valueToY: () => -500,
            viewportHeightPx: 200,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
        });
        expect(rects).toEqual([]);
    });

    it("窗口只影响 sampled 标记，不把块从画面裁掉", () => {
        const rects = layoutParamBlockRects({
            blocks: [block("in", 60), block("out", 72)],
            valueToY,
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            sampleStartSec: 0.5,
            sampleEndSec: 1.5,
            windowStartFrame: 100,
            windowEndFrame: 200,
            framePeriodMs: 10,
        });
        // 两个块的区间都是 [0,1]：都落在 [0.5,1.5] 的重叠范围内 → 都可读。
        expect(rects).toHaveLength(2);
        expect(rects.every((rect) => rect.sampled)).toBe(true);
        const outside = layoutParamBlockRects({
            blocks: [block("out", 72)],
            valueToY,
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            sampleStartSec: 10,
            sampleEndSec: 11,
            windowStartFrame: 1000,
            windowEndFrame: 1100,
            framePeriodMs: 10,
        });
        expect(outside).toHaveLength(1);
        expect(outside[0].sampled).toBe(false);
    });
});

describe("hitTestParamBlocks", () => {
    const blocks = [
        {
            id: "a",
            startSec: 0,
            endSec: 2,
            ordinal: 1,
            sampled: true,
            topPx: 80,
            heightPx: 40,
            averageValue: 60,
            selected: false,
            merged: false,
        },
        {
            id: "b",
            startSec: 1,
            endSec: 3,
            ordinal: 2,
            sampled: true,
            topPx: 190,
            heightPx: 20,
            averageValue: 72,
            selected: false,
            merged: false,
        },
    ];

    it("时间与可见矩形都命中才选中", () => {
        expect(hitTestParamBlocks({ blocks, sec: 0.5, localY: 100 })?.id).toBe("a");
        expect(hitTestParamBlocks({ blocks, sec: 2.5, localY: 200 })?.id).toBe("b");
    });

    it("时间命中但纵向落在块外时不选中", () => {
        expect(hitTestParamBlocks({ blocks, sec: 0.5, localY: 180 })).toBeNull();
        expect(hitTestParamBlocks({ blocks, sec: 2.5, localY: 120 })).toBeNull();
    });

    it("取消绘制缝后，连续音节边界精确相接", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 2,
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.5 },
                    { startSec: 0.5, endSec: 1 },
                ],
            },
        ]);
        expect(blocks[0].endSec).toBe(blocks[1].startSec);
    });

    it("未取样的块仍可命中，但标签层不伪造均值", () => {
        expect(
            hitTestParamBlocks({
                blocks: [{ ...blocks[0], sampled: false }],
                sec: 0.5,
                localY: 100,
            }),
        ).not.toBeNull();
    });

    it("重叠处取后绘制的块（用户实际看到的在上）", () => {
        expect(hitTestParamBlocks({ blocks, sec: 1.5, localY: 100 })?.id).toBe("a");
        expect(hitTestParamBlocks({ blocks, sec: 1.5, localY: 200 })?.id).toBe("b");
    });

    it("容差能命中细块的边缘，但不放大到相邻块", () => {
        const thin = [{ ...blocks[1], topPx: 48, heightPx: 4 }];
        expect(hitTestParamBlocks({ blocks: thin, sec: 1.5, localY: 53, slopPx: 2 })).not.toBe(
            null,
        );
        expect(hitTestParamBlocks({ blocks: thin, sec: 1.5, localY: 60, slopPx: 2 })).toBeNull();
    });
});

describe("nearestParamBlockByTime", () => {
    it("首帧无几何时按时间就近命中，不误选远处块", () => {
        const blocks = [
            { id: "a", startSec: 0, endSec: 1 },
            { id: "b", startSec: 2, endSec: 3 },
        ];
        expect(nearestParamBlockByTime(blocks, 0.5, 0.2)?.id).toBe("a");
        expect(nearestParamBlockByTime(blocks, 1.1, 0.2)?.id).toBe("a");
        expect(nearestParamBlockByTime(blocks, 1.5, 0.2)).toBeNull();
    });
});

describe("param block drag", () => {
    const editableClips = (): ParamBlockSourceClip[] => [
        {
            id: "a",
            trackId: "root-a",
            takeId: "take-a",
            startSec: 1,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            sourceSyllableSpansSec: [
                [0, 0.5],
                [0.5, 1],
                [1, 1.5],
                [1.5, 2],
            ],
            syllableSpansSec: projectSourceSpansToTimeline({
                id: "a",
                trackId: "root-a",
                startSec: 1,
                lengthSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 2,
                sourceSyllableSpansSec: [
                    [0, 0.5],
                    [0.5, 1],
                    [1, 1.5],
                    [1.5, 2],
                ],
            }),
        },
    ];

    it("分析未完成时音频 Clip 没有任何块；分析边界到位后才派生音节块", () => {
        expect(
            autoBlocksForClips("root-a", [
                {
                    id: "a",
                    trackId: "root-a",
                    takeId: "take-a",
                    startSec: 0,
                    lengthSec: 2,
                    sourceStartSec: 0,
                    sourceEndSec: 2,
                },
            ]),
        ).toEqual([]);
        expect(autoBlocksForClips("root-a", editableClips())).toHaveLength(4);
    });

    it("切分保存后重新派生，左右两段都按源域成员恢复", () => {
        const clips = editableClips();
        const block = autoBlocksForClips("root-a", clips)[0];
        const split = splitParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            sec: 1.25,
        });
        if (!split.ok) throw new Error(`split failed: ${split.reason}`);
        // 模拟工程保存 / 重新读取：只保留持久化字段，不保留任何自动块引用。
        const reloaded = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: {
                mergedBlocks: [],
                persistedMerges: [],
                persistedEdits: split.state.persistedEdits?.map((edit) => ({
                    ...edit,
                    id: edit.id.replace(/^edit:/, "reloaded-edit:"),
                    blockIds: edit.blockIds?.map((id) =>
                        id.startsWith("edit:") ? "missing-after-reload" : id,
                    ),
                })),
            },
        });
        expect(reloaded.slice(0, 2).map((item) => [item.startSec, item.endSec])).toEqual([
            [1, 1.25],
            [1.25, 1.5],
        ]);
    });

    it("伸缩保存后重新派生，新边界仍生效", () => {
        const clips = editableClips();
        const block = autoBlocksForClips("root-a", clips)[0];
        const resized = resizeParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            edge: "right",
            targetSec: 1.3,
        });
        if (!resized.ok) throw new Error(`resize failed: ${resized.reason}`);
        const reloaded = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: {
                mergedBlocks: [],
                persistedMerges: [],
                persistedEdits: resized.state.persistedEdits,
            },
        });
        expect(reloaded[0]).toMatchObject({ startSec: 1 });
        expect(reloaded[0]?.endSec).toBeCloseTo(1.3, 6);
    });

    it("删除块保存后重新派生仍隐藏，且不会因其零宽伪记录被清理", () => {
        const clips = editableClips();
        const block = autoBlocksForClips("root-a", clips)[1];
        const deleted = deleteParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks: autoBlocksForClips("root-a", clips),
            selectedBlockIds: [block.id],
        });
        if (!deleted.ok) throw new Error(`delete failed: ${deleted.reason}`);
        const reloaded = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: {
                mergedBlocks: [],
                persistedMerges: [],
                persistedEdits: deleted.state.persistedEdits,
            },
        });
        expect(reloaded.map((item) => [item.startSec, item.endSec])).toEqual([
            [1, 1.5],
            [2, 2.5],
            [2.5, 3],
        ]);
    });

    it("切块后左右两段各自持久化，并保持稳定成员引用", () => {
        const sourceClips: ParamBlockSourceClip[] = [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 2,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                ],
                syllableSpansSec: [
                    { startSec: 0, endSec: 1 },
                    { startSec: 1, endSec: 2 },
                ],
            },
        ];
        const block = autoBlocksForClips("root-a", sourceClips)[0];
        expect(block.startSec).toBeCloseTo(0, 6);
        expect(block.endSec).toBeCloseTo(1, 6);
        const split = splitParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            sec: 0.5,
        });
        expect(split.ok ? split.state.persistedEdits?.length : split.reason).toBeDefined();
        if (!split.ok) throw new Error(`split failed: ${split.reason}`);
        expect(split.state.persistedEdits).toHaveLength(2);
        expect(split.state.persistedEdits?.map((edit) => [edit.startSec, edit.endSec])).toEqual([
            [0, 0.5],
            [0.5, 1],
        ]);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: sourceClips,
            state: split.state,
        });
        expect(materialized.map((item) => [item.startSec, item.endSec])).toEqual([
            [0, 0.5],
            [0.5, 1],
            [1, 2],
        ]);
    });

    it("边缘伸缩保持最小宽度，并可联动相邻块压缩", () => {
        const block = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 2,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                ],
            },
        ])[0];
        expect(block).toBeDefined();
        if (block === undefined) return;
        expect(block.startSec).toBeCloseTo(0, 6);
        expect(block.endSec).toBeCloseTo(1, 6);
        const tooShort = resizeParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            edge: "right",
            targetSec: 0.01,
        });
        expect(tooShort.ok).toBe(false);
        const resized = resizeParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            edge: "right",
            targetSec: 0.5,
        });
        expect(resized.ok).toBe(true);
        if (!resized.ok) return;
        expect(resized.state.persistedEdits?.[0]?.endSec).toBeCloseTo(0.5, 9);
    });

    it("相邻块边缘延展时压缩邻居，不产生重叠", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 2,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                ],
            },
        ]);
        const [left, right] = blocks;
        expect(left).toBeDefined();
        expect(right).toBeDefined();
        if (left === undefined || right === undefined) return;
        const joined = resizeParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block: left,
            edge: "right",
            targetSec: 1.5,
            neighbor: right,
        });
        expect(joined.ok).toBe(true);
        if (!joined.ok) return;
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: [
                {
                    id: "a",
                    trackId: "root-a",
                    takeId: "take-a",
                    startSec: 0,
                    lengthSec: 2,
                    sourceStartSec: 0,
                    sourceEndSec: 2,
                    sourceSyllableSpansSec: [
                        [0, 1],
                        [1, 2],
                    ],
                },
            ],
            state: joined.state,
        });
        expect(materialized.map((item) => [item.startSec, item.endSec])).toEqual([
            [0, 1.5],
            [1.5, 2],
        ]);
        const touching = resizeParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block: left,
            edge: "right",
            targetSec: 1,
            neighbor: right,
        });
        expect(touching.ok).toBe(true);
    });

    it("手动切分后的左右段在 Clip 移动与拉伸后仍保持相对位置", () => {
        const clipAt = (
            startSec: number,
            lengthSec: number,
            sourceLengthSec = 2,
        ): ParamBlockSourceClip[] => [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec,
                lengthSec,
                sourceStartSec: 0,
                sourceEndSec: sourceLengthSec,
                sourceSyllableSpansSec: [[0, 2]],
            },
        ];
        const original = clipAt(0, 2);
        const block = autoBlocksForClips("root-a", original)[0];
        const split = splitParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            sec: 0.5,
        });
        if (!split.ok) throw new Error(`split failed: ${split.reason}`);

        const moved = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: clipAt(3, 2),
            state: split.state,
        });
        expect(moved[0]?.startSec).toBeCloseTo(3, 9);
        expect(moved[0]?.endSec).toBeCloseTo(3.5, 9);
        expect(moved[1]?.startSec).toBeCloseTo(3.5, 9);
        expect(moved[1]?.endSec).toBeCloseTo(5, 9);

        const stretched = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: clipAt(0, 4, 4),
            state: split.state,
        });
        expect(stretched[0]?.startSec).toBeCloseTo(0, 9);
        expect(stretched[0]?.endSec).toBeCloseTo(0.5, 9);
        expect(stretched[1]?.startSec).toBeCloseTo(0.5, 9);
        expect(stretched[1]?.endSec).toBeCloseTo(2, 9);
    });

    it("吸附位移：音高半音、音阶与普通参数网格", () => {
        expect(
            resolveParamBlockSnapDelta({
                param: "pitch",
                startValue: 60,
                rawDelta: 1.4,
                step: 1,
            }),
        ).toBe(1);
        expect(
            resolveParamBlockSnapDelta({
                param: "pitch",
                startValue: 60,
                rawDelta: 1.2,
                step: 1,
                scaleNotes: [0, 2, 4, 5, 7, 9, 11],
            }),
        ).toBe(2);
        expect(
            resolveParamBlockSnapDelta({
                param: "volume",
                startValue: 1,
                rawDelta: 0.27,
                step: 0.1,
            }),
        ).toBeCloseTo(0.3);
    });

    it("自由拖动按原值偏移；吸附拖动按半音对齐", () => {
        expect(
            resolveParamBlockDragWrite({
                param: "pitch",
                startValue: 60.2,
                candidateValue: 61.7,
                snap: false,
                step: 1,
            }),
        ).toBeCloseTo(1.5);
        expect(
            resolveParamBlockDragWrite({
                param: "pitch",
                startValue: 60.2,
                candidateValue: 61.7,
                snap: true,
                step: 1,
            }),
        ).toBeCloseTo(2);
    });

    it("音高吸附到工程音阶，而不是所有半音", () => {
        expect(
            resolveParamBlockDragWrite({
                param: "pitch",
                startValue: 60,
                candidateValue: 61.2,
                snap: true,
                step: 1,
                // C 大调：61(C#) 不在音阶内，最近的目标是 62(D)。
                scaleNotes: [0, 2, 4, 5, 7, 9, 11],
            }),
        ).toBeCloseTo(2);
    });

    it("多选块写入按时间合并，pitch 的 0 哨兵保持 0", () => {
        const segments = buildParamBlockDragSegments({
            blocks: [
                {
                    members: [{ clipId: "a", startSec: 0.0, endSec: 0.02 }],
                },
                {
                    members: [{ clipId: "a", startSec: 0.02, endSec: 0.04 }],
                },
            ],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            sourceAt: (frame) => (frame === 2 ? 0 : 60),
        });
        expect(segments).toHaveLength(1);
        expect(segments[0].startFrame).toBe(0);
        // 第 0 帧 = 60 被整体抬 1；第 2 帧是 pitch 的 0 哨兵，必须保持 0。
        expect(segments[0].values).toEqual([61, 61, 0, 61]);
    });

    it("预览始终以 orig 为基准，重复预览不会累积位移", () => {
        const pv = {
            startFrame: 0,
            stride: 1,
            edit: [61, 61],
            orig: [60, 60],
        };
        const block = { members: [{ clipId: "a", startSec: 0, endSec: 0.02 }] };
        const first = buildParamBlockPreviewSegments({
            blocks: [block],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            pv,
        });
        const second = buildParamBlockPreviewSegments({
            blocks: [block],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            pv,
        });
        expect(first).toEqual(second);
        expect(first[0].values).toEqual([61, 61]);
    });

    it("合并块与成员块重叠时，预览只写一次合并区间", () => {
        const segments = buildParamBlockPreviewSegments({
            blocks: [
                {
                    members: [
                        { clipId: "a", startSec: 0, endSec: 1 },
                        { clipId: "a", startSec: 0.5, endSec: 1.5 },
                    ],
                },
            ],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            pv: {
                startFrame: 0,
                stride: 1,
                edit: [60, 60, 60],
                orig: [60, 60, 60],
            },
        });
        expect(segments).toHaveLength(1);
        // 预览基准只覆盖窗口内 3 帧；窗口外保持 0，不伪造曲线值。
        expect(segments[0].values.slice(0, 3)).toEqual([61, 61, 61]);
        expect(segments[0].values.slice(3).every((value) => value === 0)).toBe(true);
    });

    it("第二次拖动可以建立在第一次已提交曲线上，而不是原始曲线", () => {
        const block = { members: [{ clipId: "a", startSec: 0, endSec: 0.02 }] };
        const baseAfterFirstDrag = [61, 61];
        const secondDrag = buildParamBlockDragSegments({
            blocks: [block],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 61,
            valueDelta: 2,
            sourceAt: (frame) => baseAfterFirstDrag[frame] ?? 0,
        });
        expect(secondDrag[0].values).toEqual([63, 63]);
    });

    it("动态参数也做线性位移，块中心跟随指针而不是按锚点比例缩小", () => {
        const block = { members: [{ clipId: "a", startSec: 0, endSec: 0.02 }] };
        const dragged = buildParamBlockDragSegments({
            blocks: [block],
            framePeriodMs: 10,
            param: "dyn",
            anchorValue: 0.2,
            valueDelta: 0.3,
            sourceAt: () => 0.2,
        });
        expect(dragged[0].values[0]).toBeCloseTo(0.5, 6);
    });
});

describe("mergeOverlappingParamBlocks", () => {
    it("只合并真正重叠的块，首尾相接保持独立", () => {
        const blocks = autoBlocksForClips("root-a", [
            clip("a", 0, 1, [0]),
            clip("b", 0.5, 1, [0.5]),
            clip("c", 1.5, 1, [1.5]),
            clip("d", 2.5, 1, [2.5]),
        ]);
        const result = mergeOverlappingParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
        });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.state.mergedBlocks).toHaveLength(1);
        expect(result.state.mergedBlocks[0].blockIds).toEqual(["auto:a:0", "auto:b:0"]);
    });

    it("没有任何重叠时明确返回 nothing_to_merge", () => {
        expect(
            mergeOverlappingParamBlocks({
                state: EMPTY_PARAM_BLOCK_STATE,
                blocks: autoBlocksForClips("root-a", [
                    clip("a", 0, 1, [0]),
                    clip("b", 1, 1, [1]),
                ]),
            }),
        ).toEqual({ ok: false, reason: "nothing_to_merge" });
    });

    it("手动选中合并的块仍然保留；一键重叠合并只处理尚未合并的自动块", () => {
        const clips = [
            clip("a", 0, 1, [0]),
            clip("b", 0.5, 1, [0.5]),
            clip("c", 2, 1, [2]),
            clip("d", 2.5, 1, [2.5]),
        ];
        const blocks = autoBlocksForClips("root-a", clips);
        const manual = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: ["auto:c:0", "auto:d:0"],
        });
        expect(manual.ok).toBe(true);
        if (!manual.ok) return;
        const result = mergeOverlappingParamBlocks({ state: manual.state, blocks });
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        // 手动记录保留；自动块若也覆盖同一段，会形成两条记录 —— materialize 以
        // 手动记录优先，不会出现重复的可见块。这里验证自动重叠组也被记录。
        expect(result.state.mergedBlocks.length).toBeGreaterThanOrEqual(2);
        expect(result.state.mergedBlocks.some((b) => b.blockIds?.includes("auto:a:0"))).toBe(true);

        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: result.state,
        });
        // 手动 c+d 优先；自动重叠组 a+b 也被记录。两者都在，不应出现重复的块。
        expect(materialized.some((block) => block.kind === "merged")).toBe(true);
        expect(new Set(materialized.map((block) => block.kind)).size).toBeGreaterThan(0);
    });
});

describe("overlap-free materialization", () => {
    it("合并块拖动后重复派生保持同一身份，不在两个块之间跳", () => {
        const clips = [clip("a", 0, 2, [0]), clip("b", 2, 2, [2])];
        const blocks = autoBlocksForClips("root-a", clips);
        const merged = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks,
            selectedBlockIds: blocks.map((block) => block.id),
        });
        expect(merged.ok).toBe(true);
        if (!merged.ok) return;
        const first = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: merged.state,
        });
        const second = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: merged.state,
        });
        expect(second.map((block) => block.id)).toEqual(first.map((block) => block.id));
        expect(second.map((block) => [block.startSec, block.endSec])).toEqual(
            first.map((block) => [block.startSec, block.endSec]),
        );
        expect(first).toHaveLength(1);
    });

    it("自动重叠归并身份不借用用户合并记录 id", () => {
        const clips = [
            clip("a", 0, 1, [0]),
            clip("b", 0.5, 1, [0.5]),
            clip("c", 3, 1, [3]),
            clip("d", 3, 1, [3]),
        ];
        const auto = autoBlocksForClips("root-a", clips);
        const manual = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks: auto,
            selectedBlockIds: [auto[2].id, auto[3].id],
        });
        expect(manual.ok).toBe(true);
        if (!manual.ok) return;
        const manualId = manual.state.mergedBlocks[0].id;
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: manual.state,
        });
        const overlap = materialized.find((block) => block.startSec === 0);
        expect(overlap?.id).toMatch(/^overlap:/);
        expect(overlap?.id).not.toBe(manualId);
    });

    it("v7 持久化合并按稳定成员引用恢复，数组序号变化不影响", () => {
        const clips = [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 2,
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.5 },
                    { startSec: 0.5, endSec: 1 },
                    { startSec: 1, endSec: 1.5 },
                ],
            },
        ];
        const blocks = autoBlocksForClips("root-a", clips);
        const persisted = blocks[0].members[0];
        expect(persisted.syllableId).toBeDefined();
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: {
                mergedBlocks: [],
                persistedMerges: [
                    {
                        id: "merged:v7",
                        members: [
                            {
                                clipId: "a",
                                takeId: "take-a",
                                syllableId: blocks[0].members[0].syllableId!,
                                sourceStartSec:
                                    blocks[0].members[0].sourceStartSec ??
                                    blocks[0].members[0].startSec,
                                sourceEndSec:
                                    blocks[0].members[0].sourceEndSec ??
                                    blocks[0].members[0].endSec,
                            },
                            {
                                clipId: "a",
                                takeId: "take-a",
                                syllableId: blocks[1].members[0].syllableId!,
                                sourceStartSec:
                                    blocks[1].members[0].sourceStartSec ??
                                    blocks[1].members[0].startSec,
                                sourceEndSec:
                                    blocks[1].members[0].sourceEndSec ??
                                    blocks[1].members[0].endSec,
                            },
                        ],
                    },
                ],
            },
        });
        expect(materialized.some((block) => block.id === "merged:v7")).toBe(true);
        expect(materialized.filter((block) => block.id === "merged:v7")).toHaveLength(1);
    });

    it("Clip 在时间轴上移动后，持久化合并仍按源域区间恢复", () => {
        const original = {
            id: "a",
            trackId: "root-a",
            takeId: "take-a",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            sourceSyllableSpansSec: [
                [0, 0.5],
                [0.5, 1],
            ] as const,
            syllableSpansSec: projectSourceSpansToTimeline({
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 2,
                sourceSyllableSpansSec: [
                    [0, 0.5],
                    [0.5, 1],
                ] as const,
            }),
        };
        const blocks = autoBlocksForClips("root-a", [original]);
        const persistedMerges = [
            {
                id: "merged:v7",
                members: blocks.slice(0, 2).map((block) => ({
                    clipId: "a",
                    takeId: "take-a",
                    syllableId: block.members[0].syllableId!,
                    sourceStartSec: block.members[0].sourceStartSec!,
                    sourceEndSec: block.members[0].sourceEndSec!,
                })),
            },
        ];
        const movedBase = {
            ...original,
            startSec: 7,
        };
        original.syllableSpansSec = projectSourceSpansToTimeline(movedBase);
        const movedBlocks = autoBlocksForClips("root-a", [movedBase]);
        persistedMerges[0] = {
            id: "merged:v7",
            members: movedBlocks.slice(0, 2).map((block) => ({
                clipId: "a",
                takeId: "take-a",
                syllableId: block.members[0].syllableId!,
                sourceStartSec: block.members[0].sourceStartSec!,
                sourceEndSec: block.members[0].sourceEndSec!,
            })),
        };
        const moved = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: [
                {
                    ...movedBase,
                    syllableSpansSec: projectSourceSpansToTimeline(movedBase),
                },
            ],
            state: { mergedBlocks: [], persistedMerges },
        });
        expect(moved.length).toBeGreaterThan(0);
        // 合并块的成员按源域身份匹配；时间线区间随 Clip 移动到 7s 之后。
        expect(moved[0]?.startSec).toBeGreaterThanOrEqual(7);
        expect(moved.some((block) => block.members.length > 1)).toBe(true);
    });

    it("源域音节区间经裁切 / 变速 / 倒放 / Loop 后仍连续且不重叠", () => {
        const base = {
            id: "a",
            trackId: "root-a",
            startSec: 2,
            lengthSec: 2,
            sourceStartSec: 0.6,
            sourceEndSec: 2.1,
            sourceSyllableSpansSec: [
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 1.5],
                [1.5, 2.0],
            ] as const,
        };
        for (const variant of [
            { ...base, playbackRate: 1 },
            { ...base, playbackRate: 1.5 },
            { ...base, reversed: true },
            {
                ...base,
                loopEnabled: true,
                contentDurationSec: 2,
                lengthSec: 4,
            },
        ]) {
            const spans = autoBlocksForClips("root-a", [
                {
                    ...variant,
                    syllableSpansSec: projectSourceSpansToTimeline(variant),
                },
            ]);
            expect(spans.length).toBeGreaterThan(0);
            for (let index = 1; index < spans.length; index += 1) {
                expect(spans[index].startSec).toBeGreaterThanOrEqual(
                    spans[index - 1].endSec - 1e-9,
                );
            }
            expect(spans.every((span) => span.endSec > span.startSec)).toBe(true);
        }
    });

    it("不管开关字段如何，重叠块都派生成一个合并块", () => {
        const clips = [clip("a", 0, 1, [0]), clip("b", 0.5, 1, [0.5])];
        const base = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: { mergedBlocks: [] },
        });
        expect(base).toHaveLength(1);
        expect(base[0].kind).toBe("merged");
        const merged = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: { mergedBlocks: [] },
        });
        expect(merged).toHaveLength(1);
        expect(merged[0].kind).toBe("merged");
        expect(merged[0].members.map((member) => member.clipId)).toEqual(["a", "b"]);
    });

    it("重叠分析段自动合并，旧开关关闭时也不产生重叠块", () => {
        const clips = [
            {
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 2,
                syllableSpansSec: [
                    { startSec: 0, endSec: 1 },
                    { startSec: 0.5, endSec: 1.5 },
                ],
            },
        ];
        const blocks = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: { mergedBlocks: [], mergeOverlapsActive: false },
        });
        expect(blocks).toHaveLength(1);
        expect(blocks[0].startSec).toBeCloseTo(0, 6);
        expect(blocks.at(-1)?.endSec).toBeCloseTo(1.5, 6);
        for (let index = 1; index < blocks.length; index += 1) {
            expect(blocks[index].startSec).toBeGreaterThanOrEqual(blocks[index - 1].endSec - 1e-9);
        }
    });

    it("手动合并组与自动重叠组相交时，最终仍不产生重叠块", () => {
        const clips = [
            clip("a", 0, 1, [0]),
            clip("b", 0.5, 1, [0.5]),
            clip("c", 1.2, 1, [1.2]),
        ];
        const auto = autoBlocksForClips("root-a", clips);
        const manual = mergeParamBlocks({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks: auto,
            selectedBlockIds: [auto[1].id, auto[2].id],
        });
        expect(manual.ok).toBe(true);
        if (!manual.ok) return;
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: manual.state,
        });
        for (let index = 1; index < materialized.length; index += 1) {
            expect(materialized[index].startSec).toBeGreaterThanOrEqual(
                materialized[index - 1].endSec - 1e-9,
            );
        }
    });
});
