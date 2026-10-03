import { describe, expect, it } from "vitest";

import {
    SPLIT_EDGE_EPSILON_SEC,
    isClipSplittableAtSec,
    resolveSplitTargetsAtSec,
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
