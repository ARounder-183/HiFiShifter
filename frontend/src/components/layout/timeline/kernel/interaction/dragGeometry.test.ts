import { describe, expect, it } from "vitest";

import { resolveDragDelta, resolveTargetTrackIndex } from "./dragGeometry";

describe("resolveDragDelta", () => {
    it("按 pxPerSec 把水平位移换算为秒", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 150,
            pxPerSec: 150,
            startSec: 10,
            lengthSec: 4,
            projectSec: 60,
        });
        expect(out.deltaSec).toBeCloseTo(1, 6);
        expect(out.startSec).toBeCloseTo(11, 6);
    });

    it("左移越界时钳制到 0，且实际位移随之收缩", () => {
        const out = resolveDragDelta({
            deltaContentXPx: -3000,
            pxPerSec: 150,
            startSec: 1,
            lengthSec: 4,
            projectSec: 60,
        });
        expect(out.startSec).toBe(0);
        expect(out.deltaSec).toBe(-1);
    });

    it("右移越界时钳制到工程末端（clip 不越出工程长度）", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 100000,
            pxPerSec: 150,
            startSec: 10,
            lengthSec: 4,
            projectSec: 60,
        });
        expect(out.startSec).toBe(56);
    });

    it("pxPerSec 非法时不产生 NaN（退化为不位移）", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 100,
            pxPerSec: 0,
            startSec: 5,
            lengthSec: 2,
            projectSec: 30,
        });
        expect(Number.isFinite(out.startSec)).toBe(true);
        expect(out.startSec).toBe(5);
        expect(out.deltaSec).toBe(0);
    });

    it("clip 比工程还长时右边界退化为 0", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 500,
            pxPerSec: 100,
            startSec: 0,
            lengthSec: 120,
            projectSec: 60,
        });
        expect(out.startSec).toBe(0);
    });
});

describe("resolveTargetTrackIndex", () => {
    it("按行高取整", () => {
        expect(resolveTargetTrackIndex(0, 96, 6)).toBe(0);
        expect(resolveTargetTrackIndex(95, 96, 6)).toBe(0);
        expect(resolveTargetTrackIndex(96, 96, 6)).toBe(1);
        expect(resolveTargetTrackIndex(300, 96, 6)).toBe(3);
    });

    it("越界时钳制到两端", () => {
        expect(resolveTargetTrackIndex(-50, 96, 6)).toBe(0);
        expect(resolveTargetTrackIndex(99999, 96, 6)).toBe(5);
    });

    it("无轨道时返回 -1", () => {
        expect(resolveTargetTrackIndex(100, 96, 0)).toBe(-1);
    });

    it("行高非法时退化为 1px 行高而不是除零", () => {
        expect(Number.isFinite(resolveTargetTrackIndex(100, 0, 3))).toBe(true);
    });
});
