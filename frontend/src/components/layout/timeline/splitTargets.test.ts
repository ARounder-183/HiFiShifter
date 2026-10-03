import { describe, expect, it } from "vitest";

import {
    SPLIT_EDGE_EPSILON_SEC,
    isClipSplittableAtSec,
    resolveSplitTargetsAtSec,
    resolveSplitTargetsWithSnap,
} from "./splitTargets";

type TestClip = { id: string; startSec: number; lengthSec: number };

function clip(id: string, startSec: number, lengthSec: number): TestClip {
    return { id, startSec, lengthSec };
}

/** 三个 Clip：A[0,4) B[2,6) C[10,14) —— 播放头 3 落在 A 与 B 内。 */
const CLIPS: TestClip[] = [clip("a", 0, 4), clip("b", 2, 6), clip("c", 10, 4)];

describe("isClipSplittableAtSec", () => {
    it("accepts a position strictly inside the clip", () => {
        expect(isClipSplittableAtSec(clip("a", 0, 4), 2)).toBe(true);
    });

    it("rejects the exact start and end edges", () => {
        const c = clip("a", 0, 4);
        expect(isClipSplittableAtSec(c, 0)).toBe(false);
        expect(isClipSplittableAtSec(c, 4)).toBe(false);
    });

    it("rejects positions within the backend's epsilon of an edge", () => {
        // 与后端 `split_clip` 的 `1e-6` 留白一致：贴边的分割在后端会被丢弃，
        // 前端必须提前认为"不可切"，否则菜单会亮着却点不动。
        const c = clip("a", 0, 4);
        expect(isClipSplittableAtSec(c, SPLIT_EDGE_EPSILON_SEC)).toBe(false);
        expect(isClipSplittableAtSec(c, SPLIT_EDGE_EPSILON_SEC / 2)).toBe(false);
        expect(isClipSplittableAtSec(c, 4 - SPLIT_EDGE_EPSILON_SEC)).toBe(false);
        expect(isClipSplittableAtSec(c, 2)).toBe(true);
    });

    it("rejects positions outside the clip", () => {
        const c = clip("a", 0, 4);
        expect(isClipSplittableAtSec(c, -1)).toBe(false);
        expect(isClipSplittableAtSec(c, 9)).toBe(false);
    });
});

describe("resolveSplitTargetsAtSec", () => {
    it("prefers the selection when there is one", () => {
        const result = resolveSplitTargetsAtSec({
            clips: CLIPS,
            splitSec: 3,
            selectedIds: ["c"],
        });
        expect(result).toEqual({ ids: ["c"], source: "selection" });
    });

    it("returns every clip under the playhead when nothing is selected", () => {
        const result = resolveSplitTargetsAtSec({ clips: CLIPS, splitSec: 3, selectedIds: [] });
        expect(result.source).toBe("playhead");
        expect(result.ids).toEqual(["a", "b"]);
    });

    it("returns nothing when the playhead is in empty space", () => {
        const result = resolveSplitTargetsAtSec({ clips: CLIPS, splitSec: 8, selectedIds: [] });
        expect(result).toEqual({ ids: [], source: "playhead" });
    });

    it("drops selected ids that no longer exist", () => {
        // 失效选区（删除 / 胶合 / 拆分替换 id 后的残留）不得把死 id 传给后端。
        const result = resolveSplitTargetsAtSec({
            clips: CLIPS,
            splitSec: 3,
            selectedIds: ["ghost", "b"],
        });
        expect(result).toEqual({ ids: ["b"], source: "selection" });
    });

    it("falls back to the playhead operand when the whole selection is stale", () => {
        const result = resolveSplitTargetsAtSec({
            clips: CLIPS,
            splitSec: 3,
            selectedIds: ["ghost"],
        });
        expect(result.source).toBe("playhead");
        expect(result.ids).toEqual(["a", "b"]);
    });
});

describe("resolveSplitTargetsWithSnap — 落点吸附后的两趟解析", () => {
    it("无吸附时与直接解析等价", () => {
        const result = resolveSplitTargetsWithSnap({
            clips: CLIPS,
            playheadSec: 3,
            selectedIds: [],
            snap: (contextIds) => {
                // 吸附器收到的是第一趟候选（作为吸附上下文）。
                expect(contextIds).toEqual(["a", "b"]);
                return 3;
            },
        });
        expect(result.splitSec).toBe(3);
        expect(result.ids).toEqual(["a", "b"]);
    });

    it("吸附把落点挪进另一个 Clip 时，操作数跟着落点走", () => {
        // 相邻两段 A[0,4) / B[4,9)。播放头 4.1 只落在 B 内；网格把落点吸到
        // 3.9 —— 此刻剃刀实际画在 A 上，操作数必须是 A（否则会出现"剃刀线
        // 画在 A 上却没切它"）。
        const clips = [clip("a", 0, 4), clip("b", 4, 5)];
        const snapped = resolveSplitTargetsWithSnap({
            clips,
            playheadSec: 4.1,
            selectedIds: [],
            snap: () => 3.9,
        });
        expect(snapped.splitSec).toBe(3.9);
        expect(snapped.ids).toEqual(["a"]);
    });

    it("反方向同理：吸附把落点挪到下一段时，改切下一段", () => {
        const clips = [clip("a", 0, 4), clip("b", 4, 5)];
        const snapped = resolveSplitTargetsWithSnap({
            clips,
            playheadSec: 3.9,
            selectedIds: [],
            snap: () => 4.1,
        });
        expect(snapped.splitSec).toBe(4.1);
        expect(snapped.ids).toEqual(["b"]);
    });

    it("吸附落点同时落在两段内时，两段都切", () => {
        const clips = [clip("a", 0, 4), clip("b", 2, 6)];
        const snapped = resolveSplitTargetsWithSnap({
            clips,
            playheadSec: 3.98,
            selectedIds: [],
            snap: () => 3.0,
        });
        expect(snapped.ids).toEqual(["a", "b"]);
    });

    it("吸附把落点挪出某个 Clip 时，那个 Clip 不再被切", () => {
        // 播放头 3.98 落在 A[0,4) 内；吸附到 4.0 后 A 不再包含落点（边界不算）。
        const clips = [clip("a", 0, 4), clip("b", 4, 4)];
        const result = resolveSplitTargetsWithSnap({
            clips,
            playheadSec: 3.98,
            selectedIds: [],
            snap: () => 4.0,
        });
        expect(result.ids).toEqual([]);
    });

    it("第一趟就没有候选时不调用吸附器（空操作不惊动吸附）", () => {
        let snapCalls = 0;
        const result = resolveSplitTargetsWithSnap({
            clips: CLIPS,
            playheadSec: 8,
            selectedIds: [],
            snap: () => {
                snapCalls += 1;
                return 8;
            },
        });
        expect(result.ids).toEqual([]);
        expect(snapCalls).toBe(0);
    });

    it("有选区时第二趟仍返回选区（吸附不改变操作数）", () => {
        const result = resolveSplitTargetsWithSnap({
            clips: CLIPS,
            playheadSec: 3,
            selectedIds: ["c"],
            snap: () => 3.5,
        });
        expect(result.source).toBe("selection");
        expect(result.ids).toEqual(["c"]);
    });
});
