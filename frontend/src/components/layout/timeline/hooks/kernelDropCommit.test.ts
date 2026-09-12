/**
 * 内核拖拽落点解析（./kernelDropCommit）行为自检。
 *
 * 【主要内容】
 * 1. 哨兵落点：拖到全部轨道之下 → 新建轨道标记；
 * 2. 已有轨道落点：同轨 / 向下跨轨 / 向上跨轨（负偏移）；
 * 3. 异常输入：落点不在轨道列表、锚点下标非法 → 一律不跨轨。
 *
 * 【作用】#1 的根因是提交分支把 `dropToNewTrack` / `trackOffset` 写死为
 * `false` / `0`，而预览分支用 `args.targetTrackId` 算出了真实落点；两处各写一份
 * 必然分叉（这正是缺陷本身）。这些断言是「幽灵预览能到新轨道/其他轨道、落库却
 * 留在原轨」的回归护栏——一旦解析退回写死值，下游提交会静默只留在原轨。
 *
 * 【与其他模块的关系】覆盖 `kernelDropCommit.ts`；不依赖 React / Redux / DOM。
 */
import { describe, expect, it } from "vitest";

import { resolveKernelDropTarget } from "./kernelDropCommit";

const NEW_TRACK_SENTINEL = "__new_track__";

describe("resolveKernelDropTarget", () => {
    it("落点为新轨哨兵：dropToNewTrack 为真、trackOffset 为 0", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: NEW_TRACK_SENTINEL,
                trackIds: ["t1", "t2"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: true, trackOffset: 0, targetTrackIndex: -1 });
    });

    it("落到其他已有轨道：trackOffset 为下标差（缺陷 1 的核心）", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t3",
                trackIds: ["t1", "t2", "t3"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 1, targetTrackIndex: 2 });
    });

    it("落到原轨：trackOffset 为 0", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t2",
                trackIds: ["t1", "t2", "t3"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 0, targetTrackIndex: 1 });
    });

    it("向上跨轨得到负偏移", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t1",
                trackIds: ["t1", "t2", "t3"],
                anchorTrackIndex: 2,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: -2, targetTrackIndex: 0 });
    });

    it("落点不在轨道列表里且不是哨兵：回落原轨（不跨轨）", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "ghost-id",
                trackIds: ["t1", "t2"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 0, targetTrackIndex: -1 });
    });

    it("锚点下标非法（-1）：不跨轨", () => {
        // 全对象断言（而非只断言 trackOffset）：本分支刻意让 `targetTrackIndex`
        // 保留落点的**真实下标**，只有 `trackOffset` 归零。消费方若去读
        // `targetTrackIndex` 就会跨轨——把它锁在测试里，逼后续改动显式面对这个选择。
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t2",
                trackIds: ["t1", "t2"],
                anchorTrackIndex: -1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 0, targetTrackIndex: 1 });
    });
});
