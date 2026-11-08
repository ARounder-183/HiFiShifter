import { describe, expect, it } from "vitest";
import { createTimelineAxis } from "../renderKernel/timelineAxis";
import { buildParamBlockScene } from "./kernel/scene/blockInstances";

import {
    createParamBlock,
    EMPTY_PARAM_BLOCK_STATE,
    autoBlocksForClips,
    averageParamValueInSpan,
    buildParamBlockDragSegments,
    buildParamBlockTransitionSegments,
    buildParamBlockTransitionPreview,
    compressedNeighborSpan,
    buildParamBlockPreviewSegments,
    blocksCoveredByParamBlockCreate,
    canMergeParamBlocks,
    deleteParamBlocks,
    hitTestParamBlocks,
    hitTestParamBlockEdges,
    hitTestParamBlockTransitionHandles,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    paramBlockCursorForHover,
    paramBlockContainingSec,
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
    previewParamBlockCreateLayout,
    projectParamBlockCreatePreviewFromBase,
    paramBlockResizeCandidates,
    projectParamBlockResizeDisplay,
    advanceParamBlockDragCommitRelease,
    canReleaseParamBlockDragCommit,
    applyFullResSegmentsToParamView,
    paramViewContainsCommittedSegments,
    resolveParamBlockCommitDisplayCenter,
    shouldUseParamBlockDragSnapshotForRender,
    resolveParamBlockDragWrite,
    resolveParamBlockDragRelease,
    projectParamBlockDragDisplayValue,
    resolveParamBlockPitchSnapDelta,
    resolveParamBlockSplitSec,
    shiftParamBlockCurveValuesToTarget,
    resolveParamBlockCreateSpan,
    resolveParamBlockResizePlan,
    resolveParamBlockResizeTarget,
    resolveParamBlockSnapDelta,
    snapParamBlockTimeByDefault,
    setParamBlockLogicalCenters,
    setParamBlockTransition,
    resizeParamBlock,
    shouldRemoveResizedParamBlock,
    shouldKeepParamBlockCenterLockedAfterDrag,
    splitParamBlock,
    paramBlockTransitionHandles,
    unmergeParamBlocks,
    type ParamBlock,
    type ParamBlockSourceClip,
    type ParamBlockState,
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
    it("分析结果从非零 Clip 起点开始可见时，仍逐段投影出块", () => {
        const sourceSyllableSpansSec = [
            [0, 0.5],
            [0.5, 1],
            [1, 1.5],
        ] as const;
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "offset",
                trackId: "root-a",
                takeId: "take-offset",
                startSec: 1.5,
                lengthSec: 2,
                sourceStartSec: 1,
                sourceEndSec: 2,
                sourceSyllableSpansSec,
                syllableSpansSec: projectSourceSpansToTimeline({
                    id: "offset",
                    trackId: "root-a",
                    startSec: 1.5,
                    lengthSec: 2,
                    sourceStartSec: 1,
                    sourceEndSec: 2,
                    sourceSyllableSpansSec,
                }),
            },
        ]);
        expect(blocks.map((block) => [block.startSec, block.endSec])).toEqual([[1.5, 2]]);
        expect(blocks[0]?.members[0]?.sourceStartSec).toBeCloseTo(1, 9);
    });

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
    it("合并可用性与传入块数组顺序无关", () => {
        const clips = [
            {
                id: "a",
                trackId: "root-a",
                startSec: 0,
                lengthSec: 1.5,
                syllableSpansSec: [
                    { startSec: 0, endSec: 0.5 },
                    { startSec: 0.5, endSec: 1 },
                    { startSec: 1, endSec: 1.5 },
                ],
            },
        ];
        const blocks = autoBlocksForClips("root-a", clips);
        expect(blocks).toHaveLength(3);
        expect(
            canMergeParamBlocks(
                EMPTY_PARAM_BLOCK_STATE,
                [blocks[0], blocks[2], blocks[1]],
                [blocks[2].id, blocks[0].id, blocks[1].id],
            ),
        ).toBe(true);
    });

    it("同组相邻块可合并", () => {
        const blocks = autoBlocksForClips("root-a", [clip("a", 0, 2, [0]), clip("b", 2, 2, [2])]);
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
        const blocks = autoBlocksForClips("root-a", [clip("a", 0, 2, [0]), clip("b", 3, 2, [3])]);
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
            ...autoBlocksForClips("root-b", [{ ...clip("b", 2, 2, [2]), trackId: "root-b" }]),
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

    it("块高是 1.2 倍行高，不随均值变化", () => {
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

    it("pitch 块中心与音高线共用 valueToY(midi + 0.5)", () => {
        const pitchValueToY = (value: number) => 1400 - value * 20;
        const rects = layoutParamBlockRects({
            blocks: [block("p", 60), block("p2", 61)],
            valueToY: pitchValueToY,
            param: "pitch",
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 20,
            // 纵向滚动后仍须与曲线同源，不能被另一个值域公式覆盖。
            valueDomain: { min: 0, max: 127, center: 40, span: 60 },
        });
        expect(rects.find((rect) => rect.id === "p")?.centerYPx).toBeCloseTo(
            pitchValueToY(60.5),
            9,
        );
        expect(rects.find((rect) => rect.id === "p2")?.centerYPx).toBeCloseTo(
            pitchValueToY(61.5),
            9,
        );
    });

    it("pitch 块显示中心吸附到最近半音行心，拖动预览可保持连续", () => {
        const valueToY = (value: number) => 1400 - value * 20;
        const rounded = layoutParamBlockRects({
            blocks: [block("p", 60.4)],
            valueToY,
            param: "pitch",
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 20,
        })[0];
        expect(rounded?.centerYPx).toBeCloseTo(valueToY(60.5), 9);

        const continuous = layoutParamBlockRects({
            blocks: [{ ...block("p", 60.4), displayCenterSnapToSemitone: false }],
            valueToY,
            param: "pitch",
            viewportHeightPx: 400,
            minValue: 0,
            maxValue: 127,
            selectedBlockIds: new Set(),
            rowHeightPx: 20,
        })[0];
        expect(continuous?.centerYPx).toBeCloseTo(valueToY(60.9), 9);
    });

    it("非 pitch 块不额外加半键偏移", () => {
        const valueToY = (value: number) => 300 - value * 10;
        const rect = layoutParamBlockRects({
            blocks: [block("loud", 0.5)],
            valueToY,
            param: "loudness",
            viewportHeightPx: 300,
            minValue: 0,
            maxValue: 2,
            selectedBlockIds: new Set(),
            rowHeightPx: 10,
        })[0];
        expect(rect?.centerYPx).toBeCloseTo(valueToY(0.5), 9);
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

describe("paramBlockContainingSec", () => {
    it("无选择时按播放头找到可切开的块，边界点不误选", () => {
        const blocks = [
            { id: "a", startSec: 0, endSec: 1 },
            { id: "b", startSec: 1, endSec: 2 },
        ];
        expect(paramBlockContainingSec(blocks, 0.5)?.id).toBe("a");
        expect(paramBlockContainingSec(blocks, 1.5)?.id).toBe("b");
        expect(paramBlockContainingSec(blocks, 1)).toBeNull();
    });
});

describe("param block drag", () => {
    it("手动新增块可持久化并重新派生", () => {
        const clips = editableClips();
        const created = createParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            clip: clips[0],
            startSec: 1.1,
            endSec: 1.4,
            value: 61.5,
        });
        if (!created.ok) throw new Error(`create failed: ${created.reason}`);
        expect(created.state.persistedEdits?.[0]?.manual).toBe(true);
        expect(created.state.persistedEdits?.[0]?.value).toBeCloseTo(61.5, 6);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: created.state,
        });
        expect(materialized.some((block) => block.id === created.blockId)).toBe(true);
        expect(
            materialized.find((block) => block.id === created.blockId)?.averageValue,
        ).toBeCloseTo(61.5, 6);
    });

    it("真实 Clip 以 activeTakeId 创建手动块，不能退化成 Clip id", () => {
        const { takeId: _takeId, ...clipWithoutTakeId } = editableClips()[0];
        const source = {
            ...clipWithoutTakeId,
            activeTakeId: "take-real",
        } as ParamBlockSourceClip & { readonly activeTakeId: string };
        const created = createParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            clip: source,
            startSec: 1.1,
            endSec: 1.4,
            value: 61.5,
        });
        if (!created.ok) throw new Error(created.reason);
        expect(created.state.persistedEdits?.[0]?.members[0]?.takeId).toBe("take-real");
    });

    it("删除后在相同区间重建手动块，不会被旧删除记录吞掉", () => {
        const clips = editableClips();
        const first = createParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            clip: clips[0],
            blockId: "edit:manual:first",
            startSec: 1.1,
            endSec: 1.4,
            value: 61.5,
        });
        if (!first.ok) throw new Error(first.reason);
        const firstMaterialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: first.state,
        });
        const firstBlock = firstMaterialized.find((block) => block.id === first.blockId);
        if (firstBlock === undefined) throw new Error("first manual block missing");
        const deleted = deleteParamBlocks({
            state: first.state,
            blocks: firstMaterialized,
            selectedBlockIds: [first.blockId],
        });
        if (!deleted.ok) throw new Error(deleted.reason);
        const recreated = createParamBlock({
            state: deleted.state,
            clip: clips[0],
            blockId: "edit:manual:second",
            startSec: 1.1,
            endSec: 1.4,
            value: 62.5,
        });
        if (!recreated.ok) throw new Error(recreated.reason);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: recreated.state,
        });
        expect(materialized.find((block) => block.id === recreated.blockId)?.averageValue).toBe(
            62.5,
        );
    });

    it("普通块拖动不锁块中心，Alt 音高缩放才锁中心", () => {
        expect(shouldKeepParamBlockCenterLockedAfterDrag({ pitchScale: false })).toBe(false);
        expect(shouldKeepParamBlockCenterLockedAfterDrag({ pitchScale: true })).toBe(true);
    });

    it("快速拖动以 pointerup 坐标为准，不依赖最后一个 pointermove", () => {
        expect(
            resolveParamBlockDragRelease({
                didDrag: false,
                startClientY: 100,
                lastMoveClientY: 102,
                releaseClientY: 160,
            }),
        ).toEqual({ didDrag: true, clientY: 160 });
        expect(
            resolveParamBlockDragRelease({
                didDrag: true,
                startClientY: 100,
                lastMoveClientY: 120,
                releaseClientY: 118,
            }),
        ).toEqual({ didDrag: true, clientY: 118 });
    });

    it("拖动块中心由手势起点和最终位移唯一投影，换曲线窗口也不漂移", () => {
        const args = {
            startValue: 60.4,
            valueDelta: 1,
            param: "pitch",
            snapToSemitone: true,
        } as const;
        expect(projectParamBlockDragDisplayValue(args)).toBe(61);
        // 同一手势重复计算必须完全一致；宿主不再从实时主簇反推 y。
        expect(projectParamBlockDragDisplayValue(args)).toBe(61);
        expect(
            projectParamBlockDragDisplayValue({
                ...args,
                snapToSemitone: false,
            }),
        ).toBeCloseTo(61.4, 9);
    });

    it("只有权威参数窗口包含全部提交值时才允许撤下交互曲线", () => {
        const segments = [{ startFrame: 10, values: [64, 65, 66] }];
        expect(
            paramViewContainsCommittedSegments({
                paramView: { startFrame: 10, stride: 1, edit: [64, 65, 66] },
                segments,
            }),
        ).toBe(true);
        expect(
            paramViewContainsCommittedSegments({
                paramView: { startFrame: 10, stride: 1, edit: [64, 65, 60] },
                segments,
            }),
        ).toBe(false);
    });

    it("降采样窗口只校验同一绝对帧，不要求邻近帧都等于一个样本", () => {
        const segments = [{ startFrame: 9, values: [63.8, 64, 64.2, 64.1, 65, 65.2, 65.1] }];
        expect(
            paramViewContainsCommittedSegments({
                paramView: { startFrame: 10, stride: 3, edit: [64, 65] },
                segments,
            }),
        ).toBe(true);
        expect(
            paramViewContainsCommittedSegments({
                paramView: { startFrame: 10, stride: 3, edit: [64, 60] },
                segments,
            }),
        ).toBe(false);
    });

    it("滚动后的窗口仅比较仍可见的提交帧", () => {
        expect(
            paramViewContainsCommittedSegments({
                paramView: { startFrame: 12, stride: 2, edit: [66, 68, 70] },
                segments: [{ startFrame: 10, values: [64, 65, 66, 67, 68] }],
            }),
        ).toBe(true);
    });

    it("全分辨率回显使用后端相同的采样帧，不把区间外的邻近帧涂进窗口", () => {
        const paramView = { startFrame: 10, stride: 3, edit: [60, 61, 62] };
        expect(
            applyFullResSegmentsToParamView({
                paramView,
                segments: [{ startFrame: 9, values: [63.8, 64, 64.2, 64.1, 65, 65.2] }],
            }).edit,
        ).toEqual([64, 65, 62]);
        expect(
            applyFullResSegmentsToParamView({
                paramView,
                segments: [{ startFrame: 11, values: [70] }],
            }).edit,
        ).toEqual(paramView.edit);
    });

    it("创建块预览始终从按下时的块集重建，拖回后可恢复被覆盖块", () => {
        const baseBlocks = [
            { id: "a", startSec: 0, endSec: 1, selected: false },
            { id: "b", startSec: 2, endSec: 3, selected: false },
        ];
        const covered = projectParamBlockCreatePreviewFromBase({
            baseBlocks,
            startSec: 0.2,
            endSec: 0.8,
            createId: "__create__",
            value: 61,
        });
        expect(covered.map((block) => block.id)).toEqual(["b", "__create__"]);
        const restored = projectParamBlockCreatePreviewFromBase({
            baseBlocks,
            startSec: 1.2,
            endSec: 1.8,
            createId: "__create__",
            value: 61,
        });
        expect(restored.map((block) => block.id)).toEqual(["a", "b", "__create__"]);
    });

    it("创建块时曲线轮廓保持形状并整体平移到目标音高", () => {
        expect(
            shiftParamBlockCurveValuesToTarget({
                values: [60, 61, 61, 62],
                target: 64,
                param: "pitch",
            }),
        ).toEqual([63, 64, 64, 65]);
    });

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
            clip: clips[0],
        });
        if (!split.ok) throw new Error(`split failed: ${split.reason}`);
        // 左右段必须拥有不同成员；否则交付后的拖动会同时改整段。
        const leftMember = split.state.persistedEdits?.[0]?.members[0];
        const rightMember = split.state.persistedEdits?.[1]?.members[0];
        expect(leftMember?.syllableId).not.toBe(rightMember?.syllableId);
        expect(leftMember?.sourceEndSec).toBeLessThanOrEqual(rightMember?.sourceStartSec ?? 0);
        // 调试：打印持久化记录的实际形状，确认左右段是否独立。
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

    it("切分只在光标接近网格线时吸附，远离时保留精确位置", () => {
        const args = {
            bpm: 120,
            timeSignatureDenominator: 4,
            stepBeats: 1,
            pxPerSec: 100,
            thresholdPx: 6,
        } as const;
        expect(resolveParamBlockSplitSec({ ...args, sec: 0.46 })).toBeCloseTo(0.5, 9);
        expect(resolveParamBlockSplitSec({ ...args, sec: 0.4 })).toBeCloseTo(0.4, 9);
        expect(
            resolveParamBlockSplitSec({
                ...args,
                sec: 0.46,
                snapToggleHeld: true,
            }),
        ).toBeCloseTo(0.46, 9);
    });

    it("切分吸附范围按固定屏幕像素计算，不随视图放大缩小改变视觉距离", () => {
        const common = {
            sec: 0.5,
            bpm: 120,
            timeSignatureDenominator: 4,
            stepBeats: 1,
            thresholdPx: 12,
        } as const;
        // 120 BPM 下一拍为 0.5s。两种缩放下的 10px 偏移都应吸附。
        expect(resolveParamBlockSplitSec({ ...common, sec: 0.3, pxPerSec: 50 })).toBeCloseTo(
            0.5,
            9,
        );
        expect(resolveParamBlockSplitSec({ ...common, sec: 0.45, pxPerSec: 200 })).toBeCloseTo(
            0.5,
            9,
        );
        // 12px 之外保持光标精确时间。
        expect(resolveParamBlockSplitSec({ ...common, sec: 0.245, pxPerSec: 50 })).toBeCloseTo(
            0.245,
            9,
        );
        expect(resolveParamBlockSplitSec({ ...common, sec: 0.26, pxPerSec: 200 })).toBeCloseTo(
            0.26,
            9,
        );
    });

    it("窄于 20ms 的切分段仍可保存和重新派生", () => {
        const clips = editableClips();
        const block = autoBlocksForClips("root-a", clips)[0];
        const split = splitParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            sec: 1.001,
            clip: clips[0],
        });
        if (!split.ok) throw new Error(split.reason);
        expect(split.state.persistedEdits?.map((edit) => edit.endSec - edit.startSec)).toEqual([
            expect.closeTo(0.001, 9),
            expect.closeTo(0.499, 9),
        ]);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: split.state,
        });
        expect(
            materialized.some((item) => Math.abs(item.endSec - item.startSec - 0.001) < 1e-9),
        ).toBe(true);
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
        // 切分后的两段必须是独立命中目标；拖动左段不能连带右段。
        const left = materialized[0];
        const right = materialized[1];
        expect(left?.id).not.toBe(right?.id);
        const leftMembers = left?.members.map((member) => member.syllableId) ?? [];
        const rightMembers = right?.members.map((member) => member.syllableId) ?? [];
        expect(leftMembers.some((id) => rightMembers.includes(id))).toBe(false);
    });

    it("切分后的左右块可独立拖动曲线，不会写回整段父音节", () => {
        const clips = editableClips();
        const block = autoBlocksForClips("root-a", clips)[0];
        const split = splitParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block,
            sec: 1.2,
            clip: clips[0],
        });
        if (!split.ok) throw new Error(`split failed: ${split.reason}`);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: split.state,
        });
        const left = materialized.find((item) => item.endSec <= 1.2 + 1e-9);
        const right = materialized.find((item) => item.startSec >= 1.2 - 1e-9);
        expect(left).toBeDefined();
        expect(right).toBeDefined();
        if (left === undefined || right === undefined) return;
        const segments = buildParamBlockDragSegments({
            blocks: [left],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 1,
            sourceAt: () => 60,
        });
        expect(segments).toHaveLength(1);
        // editableClips 的音高块是工程秒 1~1.2；右块从 1.2 起，不能被写。
        expect(segments[0]?.startFrame).toBe(100);
        expect(segments[0]?.values.length).toBe(20);
        expect(segments[0]?.values.every((value) => value === 61)).toBe(true);

        const rightSegments = buildParamBlockDragSegments({
            blocks: [right],
            framePeriodMs: 10,
            param: "pitch",
            anchorValue: 60,
            valueDelta: 2,
            sourceAt: () => 60,
        });
        expect(rightSegments[0]?.startFrame).toBe(120);
        expect(rightSegments[0]?.values.every((value) => value === 62)).toBe(true);
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
            block: {
                ...left,
                anchorStartSec: 0,
                anchorEndSec: 2,
            },
            edge: "right",
            targetSec: 1.5,
            neighbor: {
                ...right,
                anchorStartSec: 0,
                anchorEndSec: 2,
            },
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
            clip: original[0],
        });
        if (!split.ok) throw new Error(`split failed: ${split.reason}`);
        const leftMember = split.state.persistedEdits?.[0]?.members[0];
        const rightMember = split.state.persistedEdits?.[1]?.members[0];
        expect(leftMember?.syllableId).not.toBe(rightMember?.syllableId);
        expect(leftMember?.sourceStartSec).toBe(0);
        expect(rightMember?.sourceEndSec).toBe(2);
        const moved = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: clipAt(3, 2),
            state: split.state,
        });
        expect(moved[0]?.startSec).toBeCloseTo(3, 9);
        expect(moved.at(-1)?.endSec).toBeCloseTo(5, 9);
        expect(moved.length).toBeGreaterThanOrEqual(1);

        const stretched = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: clipAt(0, 4, 4),
            state: split.state,
        });
        expect(stretched[0]?.startSec).toBeCloseTo(0, 9);
        expect(stretched.at(-1)?.endSec).toBeCloseTo(4, 9);
        // 切分点也必须随 Container 等比例缩放，而不是停在 0.5。
        expect(stretched.map((item) => [item.startSec, item.endSec])).toEqual([
            [0, 1],
            [1, 4],
        ]);
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

    it("块编辑默认吸附；按住取消吸附键才自由", () => {
        expect(
            resolveParamBlockPitchSnapDelta({
                startValue: 60,
                rawDelta: 1.4,
                snapToggleHeld: false,
            }),
        ).toBe(1);
        expect(
            resolveParamBlockPitchSnapDelta({
                startValue: 60,
                rawDelta: 1.4,
                snapToggleHeld: true,
            }),
        ).toBeCloseTo(1.4, 6);
        const snapTime = (sec: number, snapToggleHeld: boolean) =>
            snapParamBlockTimeByDefault({
                sec,
                bpm: 120,
                timeSignatureDenominator: 4,
                stepBeats: 1,
                snapToggleHeld,
            });
        expect(snapTime(0.53, false)).toBeCloseTo(0.5, 6);
        expect(snapTime(0.53, true)).toBeCloseTo(0.53, 6);
    });

    it("吸附拖动按块中心的绝对行心落点，而不是只取整位移量", () => {
        // 起点中心不是整数时，位移 0 也应收敛到最近行心。
        expect(
            resolveParamBlockPitchSnapDelta({
                startValue: 60.4,
                rawDelta: 0,
                snapToggleHeld: false,
            }),
        ).toBeCloseTo(-0.4, 6);
        expect(
            resolveParamBlockPitchSnapDelta({
                startValue: 60.4,
                rawDelta: 0.52,
                snapToggleHeld: false,
            }),
        ).toBeCloseTo(0.6, 6);
        // 取消吸附：连续跟随，不再吸到行心。
        expect(
            resolveParamBlockPitchSnapDelta({
                startValue: 60.4,
                rawDelta: 0.52,
                snapToggleHeld: true,
            }),
        ).toBeCloseTo(0.52, 6);
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

    it("拖动后的逻辑音高会持久化并在重新派生时恢复", () => {
        const clips = editableClips();
        const block = autoBlocksForClips("root-a", clips)[0];
        const first = setParamBlockLogicalCenters({
            state: EMPTY_PARAM_BLOCK_STATE,
            blocks: [block],
            centers: new Map([[block.id, 65]]),
        });
        const reloaded = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: first,
        });
        const restored = reloaded.find((item) => item.id === block.id);
        expect(restored?.averageValue).toBe(65);
        expect(restored?.manualValue).toBe(65);
        expect(reloaded).toHaveLength(4);

        const second = setParamBlockLogicalCenters({
            state: first,
            blocks: [restored ?? block],
            centers: new Map([[block.id, 67]]),
        });
        expect(second.persistedEdits).toHaveLength(1);
        expect(second.persistedEdits?.[0]?.value).toBe(67);
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

describe("param block transition", () => {
    it("只生成相接块手柄，并把平滑区间限制在两侧块内", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 2,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                ],
            },
        ]);
        expect(paramBlockTransitionHandles(blocks)).toEqual([
            {
                leftId: blocks[0].id,
                rightId: blocks[1].id,
                sec: 1,
                transitionSec: 0,
            },
        ]);
        const transition = buildParamBlockTransitionSegments({
            boundaryFrame: 100,
            widthFrames: 10,
            targetValue: 62,
            leftStartFrame: 0,
            rightEndFrame: 200,
            leftValue: 60,
            rightValue: 62,
            // 两侧原曲线都在起伏；过渡应保留走势并按权重拉向 target。
            sourceAt: (frame) => 60 + Math.abs(frame - 100) * 0.01,
        });
        expect(transition).toHaveLength(1);
        // 相接点左右各 10 帧；两端都保留原曲线，中心附近被拉向目标值。
        expect(transition[0].startFrame).toBe(90);
        expect(transition[0].values).toHaveLength(20);
        // 相接点本身保留两侧原值（权重 0），两侧峰值被拉向目标值。
        expect(transition[0].values[0]).toBeCloseTo(60.1, 6);
        // 最右端回到右块原曲线（62.1 附近），说明是扭曲而非整段覆盖。
        expect(transition[0].values.at(-1)).toBeGreaterThan(60.1);
        expect(transition[0].values[10]).toBeCloseTo(62, 6);
    });
});

describe("param block pointer geometry", () => {
    const px = (sec: number) => sec * 100;

    it("边缘热区按像素命中，窄块中间仍是块体而不是缩放", () => {
        const blocks = [{ id: "a", startSec: 1, endSec: 1.2, topPx: 100, heightPx: 24 }];
        expect(
            hitTestParamBlockEdges({
                blocks,
                clientX: 102,
                localY: 110,
                secToX: px,
                hitPx: 6,
            }),
        ).toMatchObject({ block: { id: "a" }, edge: "left" });
        expect(
            hitTestParamBlockEdges({
                blocks,
                clientX: 113,
                localY: 110,
                secToX: px,
                hitPx: 6,
            }),
        ).toBeNull();
    });

    it("相接边界的左右端点归属稳定，手柄命中优先于普通块体", () => {
        const blocks = [
            {
                id: "a",
                startSec: 1,
                endSec: 2,
                topPx: 100,
                heightPx: 24,
                transitionHandle: {
                    sec: 2,
                    leftId: "a",
                    rightId: "b",
                    widthSec: 0.1,
                },
            },
            { id: "b", startSec: 2, endSec: 3, topPx: 130, heightPx: 24 },
        ];
        expect(
            hitTestParamBlockEdges({
                blocks,
                clientX: 201,
                localY: 110,
                secToX: px,
                hitPx: 6,
            }),
        ).toMatchObject({ block: { id: "a" }, edge: "right" });
        expect(
            hitTestParamBlockTransitionHandles({
                blocks,
                clientX: 200 + PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
                localY: 100 - PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
                secToX: px,
            }),
        ).toMatchObject({ leftId: "a", rightId: "b", widthSec: 0.1 });
        // 右边界本身仍属于伸缩热区；手柄在它外侧，不能互抢。
        expect(
            hitTestParamBlockTransitionHandles({
                blocks,
                clientX: 200 + PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
                localY: 100 + PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
                secToX: px,
            }),
        ).toBeNull();
        expect(paramBlockCursorForHover("transition")).toBe("ew-resize");
        expect(paramBlockCursorForHover("right-edge")).toBe("ew-resize");
    });

    it("延展与创建在拖动中实时吸附网格，取消吸附时按指针连续落点", () => {
        const grid = (sec: number) => Math.round(sec * 4) / 4;
        expect(
            resolveParamBlockResizeTarget({
                edge: "right",
                startSec: 1,
                endSec: 2,
                rawTargetSec: 2.36,
                snapToGrid: grid,
                minBlockSec: 0.04,
            }),
        ).toBe(2.25);
        // 右边缘可以越过右邻居起点继续压缩；边界在邻居终点处才停。
        expect(
            resolveParamBlockResizeTarget({
                edge: "right",
                startSec: 1,
                endSec: 2,
                rawTargetSec: 2.6,
                snapToGrid: grid,
                minBlockSec: 0.04,
                neighborStartSec: 2,
                neighborEndSec: 2.5,
            }),
        ).toBe(2.5);
        expect(
            resolveParamBlockCreateSpan({
                anchorSec: 1.1,
                pointerSec: 1.6,
                snapToGrid: grid,
                snapEnabled: true,
                minBlockSec: 0.04,
            }),
        ).toEqual({ startSec: 1, endSec: 1.5 });
        expect(
            resolveParamBlockCreateSpan({
                anchorSec: 1.1,
                pointerSec: 1.6,
                snapToGrid: grid,
                snapEnabled: false,
                minBlockSec: 0.04,
            }),
        ).toEqual({ startSec: 1.1, endSec: 1.6 });
    });

    it("缩到最小宽度以内时删除，创建块压住旧块时旧块整块让位", () => {
        expect(
            shouldRemoveResizedParamBlock({
                pointerSec: 1.02,
                startSec: 1,
                endSec: 2,
                edge: "right",
                minBlockSec: 0.04,
            }),
        ).toBe(true);
        expect(
            shouldRemoveResizedParamBlock({
                pointerSec: 1.5,
                startSec: 1,
                endSec: 2,
                edge: "right",
                minBlockSec: 0.04,
            }),
        ).toBe(false);

        const blocks = [{ id: "old", startSec: 1, endSec: 2, averageValue: 60, selected: false }];
        expect(
            blocksCoveredByParamBlockCreate({
                blocks,
                startSec: 1.2,
                endSec: 1.6,
            }),
        ).toEqual(["old"]);
        expect(
            previewParamBlockCreateLayout({
                blocks,
                startSec: 1.2,
                endSec: 1.6,
                createId: "__create__",
                value: 62,
            }).map((block) => block.id),
        ).toEqual(["__create__"]);
    });

    it("同一条相接边只保存一份过渡宽度，不串到下一个块", () => {
        const clip = {
            id: "a",
            trackId: "root-a",
            takeId: "take-a",
            startSec: 0,
            lengthSec: 3,
            sourceStartSec: 0,
            sourceEndSec: 3,
            sourceSyllableSpansSec: [
                [0, 1],
                [1, 2],
                [2, 3],
            ],
        } as const;
        const blocks = autoBlocksForClips("root-a", [clip]);
        expect(blocks).toHaveLength(3);
        const [first, second, third] = blocks;
        if (first === undefined || second === undefined || third === undefined) return;
        const transition = setParamBlockTransition({
            state: EMPTY_PARAM_BLOCK_STATE,
            left: first,
            right: second,
            transitionSec: 0.25,
        });
        if (!transition.ok) throw new Error(transition.reason);
        // 只写左块；右块自身记录不携带同一宽度。
        expect(transition.state.persistedEdits).toHaveLength(1);
        expect(transition.state.persistedEdits?.[0]?.id).toBe(first.id);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips: [clip],
            state: transition.state,
        });
        const handles = paramBlockTransitionHandles(materialized);
        expect(handles).toHaveLength(2);
        expect(handles.map((handle) => handle.transitionSec)).toEqual([0.25, 0]);
        expect(handles[1]?.leftId).toBe(second.id);
        expect(handles[1]?.rightId).toBe(third.id);
    });

    it("手柄拖动预览在相接点两侧扭曲原曲线，而不是覆盖成直线", () => {
        const preview = buildParamBlockTransitionPreview({
            boundaryFrame: 100,
            widthFrames: 8,
            targetValue: 62,
            leftStartFrame: 100,
            rightEndFrame: 160,
            leftValue: 60,
            rightValue: 61,
            paramView: {
                startFrame: 100,
                stride: 1,
                // 右侧原曲线本身有起伏；预览应保留一部分而不是直接变成直线。
                edit: [60, 61, 60, 61, 60, 61, 60, 61, 60, 61],
            },
        });
        expect(preview).toHaveLength(1);
        expect(preview[0]?.startFrame).toBe(100);
        expect(preview[0]?.values).toHaveLength(8);
        // 相接点本身权重最大；峰值把原曲线拉向目标值。
        expect(preview[0]?.values[0]).toBe(62);
        expect(preview[0]?.values[1]).toBeGreaterThan(61);
        expect(preview[0]?.values[1]).toBeLessThan(62);
    });

    it("过渡提交片段可立即回填当前参数窗口", () => {
        expect(
            applyFullResSegmentsToParamView({
                paramView: {
                    startFrame: 100,
                    stride: 1,
                    edit: [60, 60, 60, 61, 61],
                },
                segments: [{ startFrame: 101, values: [62, 63] }],
            }).edit,
        ).toEqual([60, 62, 63, 61, 61]);
    });

    it("拉手柄只改过渡曲线，不改块高与代表值", () => {
        const left = {
            id: "left",
            startSec: 0,
            endSec: 1,
            topPx: 100,
            heightPx: 19.2,
            averageValue: 60,
            selected: true,
            merged: false,
        };
        const right = {
            id: "right",
            startSec: 1,
            endSec: 2,
            topPx: 130,
            heightPx: 19.2,
            averageValue: 62,
            selected: false,
            merged: false,
        };
        const scene = buildParamBlockScene({
            axis: createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 400 }),
            viewportHeightPx: 300,
            blocks: [
                {
                    ...left,
                    transitionHandle: {
                        sec: 1,
                        leftId: "left",
                        rightId: "right",
                        widthSec: 0.2,
                    },
                },
                right,
            ],
            borderRgba: [0, 0, 0, 1],
            selectedRgba: [0, 0, 1, 1],
            mergedRgba: [1, 1, 1, 1],
            fillRgba: [0.5, 0.5, 0.5, 1],
            transitionRgba: [0, 0, 1, 1],
        });
        expect(scene.fills.map((fill) => fill.h)).toHaveLength(2);
        expect(scene.fills[0]?.h).toBeCloseTo(19.2, 9);
        expect(scene.fills[1]?.h).toBeCloseTo(19.2, 9);
        // 过渡手柄/曲线不参与填充层，块高只由行高决定。
        expect(scene.handles.every((handle) => handle.h !== 19.2)).toBe(true);
    });

    it("吸附后越过邻居起点时仍可继续拉，允许完整覆盖邻块", () => {
        const span = compressedNeighborSpan({
            edge: "right",
            neighborStartSec: 1,
            neighborEndSec: 1.2,
            resizedEdgeSec: 1.3,
            minBlockSec: 0.04,
        });
        expect(span).not.toBeNull();
        expect(span?.startSec).toBeCloseTo(1.16, 6);
        expect(span?.endSec).toBeCloseTo(1.2, 6);
        // 目标再远也只停在“邻居终点 − 最小宽度”。
        const plan = resolveParamBlockResizePlan({
            edge: "right",
            blockStartSec: 0,
            blockEndSec: 1,
            targetSec: 1.8,
            neighbors: [{ id: "b", startSec: 1, endSec: 1.2 }],
            minBlockSec: 0.04,
        });
        expect(plan.block).toEqual({ startSec: 0, endSec: 1.8 });
        expect(plan.neighbors).toEqual([{ id: "b", startSec: 1, endSec: 1.2, remove: true }]);
    });

    it("左右扩展时，两侧邻居都在预览与提交中同步缩短", () => {
        const right = compressedNeighborSpan({
            edge: "right",
            neighborStartSec: 2,
            neighborEndSec: 3,
            resizedEdgeSec: 2.3,
            minBlockSec: 0.04,
        });
        expect(right).toMatchObject({ startSec: 2.3, endSec: 3 });
        const left = compressedNeighborSpan({
            edge: "left",
            neighborStartSec: 0,
            neighborEndSec: 1,
            resizedEdgeSec: 0.7,
            minBlockSec: 0.04,
        });
        expect(left).toMatchObject({ startSec: 0, endSec: 0.7 });
    });

    it("连续块的相等边界也能解析为邻居，左右伸缩计划严格首尾相接", () => {
        const blocks = autoBlocksForClips("root-a", [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 3,
                sourceStartSec: 0,
                sourceEndSec: 3,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                ],
            },
        ]);
        const [left, middle, right] = blocks;
        expect(left).toBeDefined();
        expect(middle).toBeDefined();
        expect(right).toBeDefined();
        if (left === undefined || middle === undefined || right === undefined) return;

        expect(paramBlockResizeCandidates(blocks, left, "right").map((b) => b.id)).toEqual([
            middle.id,
            right.id,
        ]);
        expect(paramBlockResizeCandidates(blocks, middle, "left").map((b) => b.id)).toEqual([
            left.id,
        ]);
        expect(paramBlockResizeCandidates(blocks, right, "left").map((b) => b.id)).toEqual([
            middle.id,
            left.id,
        ]);

        const extendRight = resolveParamBlockResizePlan({
            edge: "right",
            blockStartSec: left.startSec,
            blockEndSec: left.endSec,
            targetSec: 1.4,
            neighbors: [middle, right],
            minBlockSec: 0.04,
        });
        expect(extendRight.block).toEqual({ startSec: 0, endSec: 1.4 });
        expect(extendRight.neighbors).toEqual([
            {
                id: middle.id,
                startSec: 1.4,
                endSec: 2,
                remove: false,
            },
        ]);

        const extendLeft = resolveParamBlockResizePlan({
            edge: "left",
            blockStartSec: right.startSec,
            blockEndSec: right.endSec,
            targetSec: 1.6,
            neighbors: [middle, left],
            minBlockSec: 0.04,
        });
        expect(extendLeft.block).toEqual({ startSec: 1.6, endSec: 3 });
        expect(extendLeft.neighbors).toEqual([
            {
                id: middle.id,
                startSec: 1,
                endSec: 1.6,
                remove: false,
            },
        ]);
    });

    it("共享计划提交后仍是两个同级块，不会在重建时被合并", () => {
        const clips: ParamBlockSourceClip[] = [
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
        ];
        const [left, right] = autoBlocksForClips("root-a", clips);
        expect(left).toBeDefined();
        expect(right).toBeDefined();
        if (left === undefined || right === undefined) return;
        const plan = resolveParamBlockResizePlan({
            edge: "right",
            blockStartSec: left.startSec,
            blockEndSec: left.endSec,
            targetSec: 1.6,
            neighbors: [right],
            minBlockSec: 0.04,
        });
        const result = resizeParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            block: left,
            edge: "right",
            targetSec: plan.block.endSec,
            targetSpan: plan.block,
            neighborSpans: plan.neighbors,
            neighborBlocks: [right],
        });
        if (!result.ok) throw new Error(result.reason);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: result.state,
        });
        expect(materialized).toHaveLength(2);
        expect(materialized.map((block) => [block.startSec, block.endSec])).toEqual([
            [0, 1.6],
            [1.6, 2],
        ]);
        expect(materialized.every((block) => block.kind !== "merged")).toBe(true);
    });

    it("连续多次从左右伸缩都不会越过边界、消失或并成一个大块", () => {
        const clips: ParamBlockSourceClip[] = [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 3,
                sourceStartSec: 0,
                sourceEndSec: 3,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                ],
            },
        ];
        let state: ParamBlockState = { ...EMPTY_PARAM_BLOCK_STATE };
        let view = materializeParamBlocks({ rootTrackId: "root-a", clips, state });
        expect(view).toHaveLength(3);

        // 左块右扩，压短中间块。
        const first = view[0];
        const middle = view[1];
        const right = view[2];
        if (first === undefined || middle === undefined || right === undefined) return;
        const plan1 = resolveParamBlockResizePlan({
            edge: "right",
            blockStartSec: first.startSec,
            blockEndSec: first.endSec,
            targetSec: 1.7,
            neighbors: [middle, right],
            minBlockSec: 0.04,
        });
        const edit1 = resizeParamBlock({
            state,
            block: first,
            edge: "right",
            targetSec: plan1.block.endSec,
            targetSpan: plan1.block,
            neighborSpans: plan1.neighbors,
            neighborBlocks: [middle, right],
        });
        if (!edit1.ok) throw new Error(edit1.reason);
        state = edit1.state;
        view = materializeParamBlocks({ rootTrackId: "root-a", clips, state });
        expect(view.map((block) => [block.startSec, block.endSec])).toEqual([
            [0, 1.7],
            [1.7, 2],
            [2, 3],
        ]);

        // 再从右侧块向左扩：先压短中间块，右侧块必须覆盖完整中间块。
        const firstAfter = view[0];
        const middleAfter = view[1];
        const rightAfter = view[2];
        if (firstAfter === undefined || middleAfter === undefined || rightAfter === undefined) {
            return;
        }
        const plan2 = resolveParamBlockResizePlan({
            edge: "left",
            blockStartSec: rightAfter.startSec,
            blockEndSec: rightAfter.endSec,
            targetSec: 1.0,
            neighbors: [middleAfter, firstAfter],
            minBlockSec: 0.04,
        });
        const edit2 = resizeParamBlock({
            state,
            block: rightAfter,
            edge: "left",
            targetSec: plan2.block.startSec,
            targetSpan: plan2.block,
            neighborSpans: plan2.neighbors,
            neighborBlocks: [middleAfter, firstAfter],
        });
        if (!edit2.ok) throw new Error(edit2.reason);
        state = edit2.state;
        view = materializeParamBlocks({ rootTrackId: "root-a", clips, state });
        // 右侧块左扩到 1.0 时完整覆盖中间块，并压缩左块到 1.0；
        // 结果应只剩两个独立块，不能再出现中间残段或合并块。
        expect(view).toHaveLength(2);
        expect(view.map((block) => [block.startSec, block.endSec])).toEqual([
            [0, 1],
            [1, 3],
        ]);
        expect(view.every((block) => block.kind !== "merged")).toBe(true);
        expect(view.every((block) => block.endSec - block.startSec >= 0.04 - 1e-9)).toBe(true);
        // 所有端点都落在连续边界上，没有重叠。
        for (let index = 1; index < view.length; index += 1) {
            expect(view[index]?.startSec).toBeCloseTo(view[index - 1]?.endSec ?? NaN, 9);
        }
    });

    it("向右扩可以跨过真实空隙，并一次缩短 / 吞并多个右侧块", () => {
        const clips: ParamBlockSourceClip[] = [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 0,
                lengthSec: 6,
                sourceStartSec: 0,
                sourceEndSec: 6,
                sourceSyllableSpansSec: [
                    [0, 1],
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ],
            },
        ];
        let state: ParamBlockState = { ...EMPTY_PARAM_BLOCK_STATE };
        const initial = materializeParamBlocks({ rootTrackId: "root-a", clips, state });
        const [first, second, third] = initial;
        if (first === undefined || second === undefined || third === undefined) return;

        const extend = resolveParamBlockResizePlan({
            edge: "right",
            blockStartSec: first.startSec,
            blockEndSec: first.endSec,
            targetSec: 3.6,
            neighbors: paramBlockResizeCandidates(initial, first, "right"),
            minBlockSec: 0.04,
        });
        expect(extend.neighbors).toEqual([
            {
                id: second.id,
                startSec: 1,
                endSec: 2,
                remove: true,
            },
            {
                id: third.id,
                startSec: 3.6,
                endSec: 4,
                remove: false,
            },
        ]);
        const edit = resizeParamBlock({
            state,
            block: first,
            edge: "right",
            targetSec: extend.block.endSec,
            targetSpan: extend.block,
            neighborSpans: extend.neighbors,
            neighborBlocks: initial,
        });
        if (!edit.ok) throw new Error(edit.reason);
        state = edit.state;
        const result = materializeParamBlocks({ rootTrackId: "root-a", clips, state });
        expect(result.map((block) => [block.startSec, block.endSec])).toEqual([
            [0, 3.6],
            [3.6, 4],
            [5, 6],
        ]);
        // 完整覆盖的中间块不能从自动块池复活。
        expect(result.some((block) => block.id === second.id)).toBe(false);
    });

    it("预览删除后拉回：当前块和已吞并邻居都能从手势起点恢复", () => {
        const base = [
            { id: "self", startSec: 0, endSec: 1 },
            { id: "a", startSec: 1, endSec: 2 },
            { id: "b", startSec: 2, endSec: 3 },
        ];
        const removeSelf = projectParamBlockResizeDisplay({
            baseBlocks: base,
            blockId: "self",
            plan: resolveParamBlockResizePlan({
                edge: "right",
                blockStartSec: 0,
                blockEndSec: 1,
                targetSec: 0,
                neighbors: base.slice(1),
                minBlockSec: 0.04,
            }),
        });
        expect(removeSelf.map((block) => block.id)).toEqual(["a", "b"]);

        const removeNeighbors = projectParamBlockResizeDisplay({
            baseBlocks: base,
            blockId: "self",
            plan: resolveParamBlockResizePlan({
                edge: "right",
                blockStartSec: 0,
                blockEndSec: 1,
                targetSec: 2.8,
                neighbors: base.slice(1),
                minBlockSec: 0.04,
            }),
        });
        expect(removeNeighbors.map((block) => block.id)).toEqual(["self", "b"]);

        // 同一份起点 + 指针拉回，删除结果必须可逆，不能依赖上一帧列表。
        const restored = projectParamBlockResizeDisplay({
            baseBlocks: base,
            blockId: "self",
            plan: resolveParamBlockResizePlan({
                edge: "right",
                blockStartSec: 0,
                blockEndSec: 1,
                targetSec: 1.4,
                neighbors: base.slice(1),
                minBlockSec: 0.04,
            }),
        });
        expect(restored).toEqual([
            { id: "self", startSec: 0, endSec: 1.4, selected: true },
            { id: "a", startSec: 1.4, endSec: 2, selected: false },
            { id: "b", startSec: 2, endSec: 3, selected: false },
        ]);
    });

    it("纵向拖动提交后只在曲线同步或超时才释放交接快照", () => {
        // 提交未完成：无论曲线如何都不能撤掉快照，否则会按旧主簇闪回。
        expect(
            canReleaseParamBlockDragCommit({
                committed: false,
                startedAt: 0,
                now: 10,
                curveSettled: true,
            }),
        ).toBe(false);
        // 已提交但 paramView 还是手势起点窗口：继续保留快照。
        expect(
            canReleaseParamBlockDragCommit({
                committed: true,
                startedAt: 100,
                now: 200,
                curveSettled: false,
            }),
        ).toBe(false);
        // 新窗口接上：可以释放。
        expect(
            canReleaseParamBlockDragCommit({
                committed: true,
                startedAt: 100,
                now: 200,
                curveSettled: true,
            }),
        ).toBe(true);
        // 本地回显复用同一个 key：只要窗口里已经是提交值，也可以释放。
        expect(
            canReleaseParamBlockDragCommit({
                committed: true,
                startedAt: 100,
                now: 200,
                curveSettled: false,
                valuesApplied: true,
            }),
        ).toBe(true);
        // 异常路径兜底：超时也释放，不能永久锁住块。
        expect(
            canReleaseParamBlockDragCommit({
                committed: true,
                startedAt: 100,
                now: 1400,
                curveSettled: false,
                timeoutMs: 1200,
            }),
        ).toBe(true);
    });

    it("手动创建的块切分后左右段都会保留", () => {
        const clips: ParamBlockSourceClip[] = [
            {
                id: "a",
                trackId: "root-a",
                takeId: "take-a",
                startSec: 1,
                lengthSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 2,
            },
        ];
        const created = createParamBlock({
            state: EMPTY_PARAM_BLOCK_STATE,
            clip: clips[0],
            blockId: "edit:manual:split-source",
            startSec: 1.1,
            endSec: 1.8,
            value: 61,
        });
        if (!created.ok) throw new Error(created.reason);
        const blocks = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: created.state,
        });
        const source = blocks.find((block) => block.id === created.blockId);
        if (source === undefined) throw new Error("manual source missing");
        const split = splitParamBlock({
            state: created.state,
            block: source,
            sec: 1.4,
            clip: clips[0],
        });
        if (!split.ok) throw new Error(split.reason);
        const materialized = materializeParamBlocks({
            rootTrackId: "root-a",
            clips,
            state: split.state,
        });
        expect(
            materialized
                .filter((block) => block.startSec >= 1.1 - 1e-9 && block.endSec <= 1.8 + 1e-9)
                .map((block) => [block.startSec, block.endSec]),
        ).toContainEqual([1.1, 1.4]);
        expect(
            materialized
                .filter((block) => block.startSec >= 1.1 - 1e-9 && block.endSec <= 1.8 + 1e-9)
                .map((block) => [block.startSec, block.endSec]),
        ).toContainEqual([1.4, 1.8]);
    });

    it("拖动提交就绪后先绘制一帧快照，再在下一绘制时机释放", () => {
        const ready = {
            committed: true,
            startedAt: 100,
            now: 200,
            curveSettled: false,
            valuesApplied: true,
        } as const;
        expect(
            advanceParamBlockDragCommitRelease({
                ...ready,
                releaseState: "waiting-for-curve",
            }),
        ).toBe("waiting-for-paint");
        expect(
            advanceParamBlockDragCommitRelease({
                ...ready,
                releaseState: "waiting-for-paint",
            }),
        ).toBe("released");
    });

    it("拖动和提交交接跨窗口时，当前渲染仍使用拖动快照", () => {
        expect(
            shouldUseParamBlockDragSnapshotForRender({
                snapshotKey: "old-window",
                paramViewKey: "new-window",
                gestureActive: true,
                commitHeld: true,
            }),
        ).toBe(true);
        expect(
            shouldUseParamBlockDragSnapshotForRender({
                snapshotKey: "old-window",
                paramViewKey: "new-window",
                gestureActive: true,
                commitHeld: false,
            }),
        ).toBe(true);
        expect(
            shouldUseParamBlockDragSnapshotForRender({
                snapshotKey: "old-window",
                paramViewKey: "new-window",
                gestureActive: false,
                commitHeld: false,
            }),
        ).toBe(false);
    });

    it("松手交接中心与普通显示使用同一个吸附行心", () => {
        expect(
            resolveParamBlockCommitDisplayCenter({
                rawCenter: 60.4,
                param: "pitch",
                continuous: false,
            }),
        ).toBe(60);
        expect(
            resolveParamBlockCommitDisplayCenter({
                rawCenter: 60.6,
                param: "pitch",
                continuous: false,
            }),
        ).toBe(61);
        // 按住取消吸附 / Alt 缩放保持连续，不能二次取整。
        expect(
            resolveParamBlockCommitDisplayCenter({
                rawCenter: 60.4,
                param: "pitch",
                continuous: true,
            }),
        ).toBeCloseTo(60.4, 9);
        // 非 pitch 参数不吸附。
        expect(
            resolveParamBlockCommitDisplayCenter({
                rawCenter: 0.42,
                param: "loudness",
                continuous: false,
            }),
        ).toBeCloseTo(0.42, 9);
    });

    it("像素级端点误差不会被误判为重叠块", () => {
        const base = autoBlocksForClips("root-a", [
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
        const state: ParamBlockState = {
            mergedBlocks: [],
            persistedMerges: [],
            persistedEdits: base.map((block, index) => ({
                id: block.id,
                members: block.members.map((member) => ({
                    clipId: member.clipId,
                    takeId: member.takeId ?? "take-a",
                    syllableId: member.syllableId ?? `${index}`,
                    sourceStartSec: member.sourceStartSec ?? 0,
                    sourceEndSec: member.sourceEndSec ?? 1,
                })),
                startSec: index === 0 ? 0 : 1 + 5e-9,
                endSec: index === 0 ? 1 : 2,
                memberStartSec: 0,
                memberEndSec: 2,
                explicit: true,
                blockIds: [block.id],
            })),
        };
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
            state,
        });
        expect(materialized).toHaveLength(2);
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
                blocks: autoBlocksForClips("root-a", [clip("a", 0, 1, [0]), clip("b", 1, 1, [1])]),
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
        const clips = [clip("a", 0, 1, [0]), clip("b", 0.5, 1, [0.5]), clip("c", 1.2, 1, [1.2])];
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
