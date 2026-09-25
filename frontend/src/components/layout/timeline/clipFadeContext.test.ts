import { describe, expect, it } from "vitest";

import { sharedFadeShape } from "./clipFadeContext";
import type { ClipInfo } from "../../../features/session/sessionTypes";

/** 造一个只关心淡变形状的最小 Clip。 */
function clip(id: string, fadeInShape: number, fadeOutShape: number): ClipInfo {
    return {
        id,
        trackId: "t1",
        name: id,
        startSec: 0,
        lengthSec: 1,
        fadeInSec: 0.1,
        fadeOutSec: 0.1,
        fadeInShape,
        fadeOutShape,
    } as ClipInfo;
}

/**
 * 多选淡变形状行的"当前值"。
 *
 * 【为什么需要】多选时这一行不再是"某个 Clip 的形状"，而是整组的共同状态。
 * 若在形状不一致时仍然高亮第一项，用户会以为点击是"保持不变"，实际却把
 * 整组都改成了它 —— 静默改错一批数据。`null`（不高亮任何一项）才是诚实表示。
 */
describe("sharedFadeShape", () => {
    it("returns the common shape when every clip agrees", () => {
        const clips = [clip("a", 2, 1), clip("b", 2, 1)];
        expect(sharedFadeShape(clips, "in")).toBe(2);
        expect(sharedFadeShape(clips, "out")).toBe(1);
    });

    it("returns null when the shapes differ", () => {
        const clips = [clip("a", 2, 1), clip("b", 3, 1)];
        expect(sharedFadeShape(clips, "in")).toBeNull();
        // 另一侧一致时仍应给出该侧的形状 —— 两侧互不影响。
        expect(sharedFadeShape(clips, "out")).toBe(1);
    });

    it("returns null for a single clip that differs from nothing", () => {
        // 单个 Clip 没有"不一致"可言：它就是自己的形状。
        expect(sharedFadeShape([clip("a", 4, 0)], "in")).toBe(4);
    });

    it("returns null for an empty list", () => {
        expect(sharedFadeShape([], "in")).toBeNull();
    });

    it("compares decimal variants by their base family", () => {
        // 1.1 与 1 属同一族（REAPER 语义）：不应被判成不一致。
        const clips = [clip("a", 1, 0), clip("b", 1.1, 0)];
        expect(sharedFadeShape(clips, "in")).toBe(1);
        // 2 与 1 是不同族。
        expect(sharedFadeShape([clip("a", 1, 0), clip("b", 2, 0)], "in")).toBeNull();
    });

    it("treats a non-finite shape as the default family", () => {
        const clips = [clip("a", 0, 0), clip("b", Number.NaN, 0)];
        expect(sharedFadeShape(clips, "in")).toBe(0);
    });
});
