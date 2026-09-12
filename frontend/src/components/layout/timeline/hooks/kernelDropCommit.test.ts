/**
 * 内核拖拽落点解析单测。
 *
 * 【为什么必须有这一层】#1 的根因是提交分支把 `dropToNewTrack` / `trackOffset`
 * 写死为 `false` / `0`，而预览分支用 `args.targetTrackId` 算出了真实落点。
 * 两处各写一份必然分叉（这正是缺陷本身），因此把解析收敛到本模块并在此锁定行为。
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
            }).trackOffset,
        ).toBe(-2);
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
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t2",
                trackIds: ["t1", "t2"],
                anchorTrackIndex: -1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }).trackOffset,
        ).toBe(0);
    });
});
