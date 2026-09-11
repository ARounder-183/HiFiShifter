import { describe, expect, it } from "vitest";

import { clipIntersectsBox, resolveBoxBounds } from "./boxSelection";

describe("resolveBoxBounds", () => {
    it("从左往右、从上往下拖：边界即起止点", () => {
        expect(resolveBoxBounds(10, 20, 100, 200)).toEqual({
            left: 10,
            top: 20,
            right: 100,
            bottom: 200,
        });
    });

    it("反向拖拽（右下 → 左上）被规范化", () => {
        expect(resolveBoxBounds(100, 200, 10, 20)).toEqual({
            left: 10,
            top: 20,
            right: 100,
            bottom: 200,
        });
    });

    it("非法值退化为 0 而不是 NaN", () => {
        const out = resolveBoxBounds(Number.NaN, 10, 50, Number.NaN);
        expect(Number.isFinite(out.left)).toBe(true);
        expect(Number.isFinite(out.bottom)).toBe(true);
        expect(out.left).toBe(0);
        expect(out.bottom).toBe(10);
    });
});

describe("clipIntersectsBox", () => {
    // 100 px/s、行高 80：轨道 0 的 clip [1,3) → 内容矩形 [100,300] × [0,80]
    const base = { clipStartSec: 1, clipLengthSec: 2, trackIndex: 0, pxPerSec: 100, rowHeight: 80 };

    it("框完全覆盖 clip → 命中", () => {
        expect(
            clipIntersectsBox({ ...base, box: { left: 0, top: 0, right: 500, bottom: 500 } }),
        ).toBe(true);
    });

    it("框与 clip 部分重叠 → 命中", () => {
        expect(
            clipIntersectsBox({ ...base, box: { left: 250, top: 0, right: 400, bottom: 80 } }),
        ).toBe(true);
    });

    it("框在 clip 右侧之外 → 不命中", () => {
        expect(
            clipIntersectsBox({ ...base, box: { left: 301, top: 0, right: 400, bottom: 80 } }),
        ).toBe(false);
    });

    it("框在 clip 下方（另一条轨道）→ 不命中", () => {
        expect(
            clipIntersectsBox({ ...base, box: { left: 0, top: 81, right: 500, bottom: 200 } }),
        ).toBe(false);
    });

    it("框与 clip 边缘相切 → 命中（闭区间）", () => {
        expect(
            clipIntersectsBox({ ...base, box: { left: 300, top: 0, right: 400, bottom: 80 } }),
        ).toBe(true);
    });

    it("pxPerSec 非法 → 不命中而不是误判", () => {
        expect(
            clipIntersectsBox({
                ...base,
                pxPerSec: 0,
                box: { left: 0, top: 0, right: 500, bottom: 500 },
            }),
        ).toBe(false);
    });
});
