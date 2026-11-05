import { describe, expect, it } from "vitest";

import {
    isClipHeaderControlVisible,
    isClipHeaderRightLabelVisible,
    resolveClipHeaderLayout,
} from "./clipHeaderLayout";

describe("resolveClipHeaderLayout", () => {
    it("无重叠时名称从控件区右缘开始", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 300,
            leadingOverlapPx: 0,
            leadingControlsWidth: 80,
            trailingReservePx: 60,
        });
        expect(layout.visibleLeftPx).toBe(0);
        expect(layout.nameLeftPx).toBe(80);
        expect(layout.nameRightPx).toBe(240);
        expect(layout.nameVisible).toBe(true);
    });

    it("重叠段越过控件区时名称从重叠边界之后开始", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 300,
            leadingOverlapPx: 120,
            leadingControlsWidth: 80,
            trailingReservePx: 60,
        });
        expect(layout.visibleLeftPx).toBe(120);
        expect(layout.nameLeftPx).toBe(120);
        expect(layout.nameWidthPx).toBe(120);
        expect(layout.nameVisible).toBe(true);
    });

    it("重叠几乎覆盖整个 header 时名称不可见，但保留真实边界", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 140,
            leadingOverlapPx: 136,
            leadingControlsWidth: 80,
            trailingReservePx: 60,
        });
        expect(layout.visibleLeftPx).toBe(136);
        expect(layout.nameLeftPx).toBe(136);
        expect(layout.nameRightPx).toBe(80);
        expect(layout.nameWidthPx).toBe(0);
        expect(layout.nameVisible).toBe(false);
    });

    it("重叠宽度超过自身宽度时收敛到 Clip 宽度", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 100,
            leadingOverlapPx: 999,
            leadingControlsWidth: 80,
            trailingReservePx: 60,
        });
        expect(layout.coveredWidthPx).toBe(100);
        expect(layout.visibleLeftPx).toBe(100);
        expect(layout.nameVisible).toBe(false);
    });

    it("非有限输入按 0 处理，不产生 NaN 几何", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: Number.NaN,
            leadingOverlapPx: Number.NaN,
            leadingControlsWidth: Number.NaN,
            trailingReservePx: Number.NaN,
        });
        expect(layout.visibleLeftPx).toBe(0);
        expect(Number.isFinite(layout.nameLeftPx)).toBe(true);
        expect(Number.isFinite(layout.nameRightPx)).toBe(true);
        expect(Number.isFinite(layout.nameWidthPx)).toBe(true);
    });
});

describe("控件可见性", () => {
    const overlapLayout = resolveClipHeaderLayout({
        clipWidthPx: 300,
        leadingOverlapPx: 100,
        leadingControlsWidth: 80,
        trailingReservePx: 60,
    });

    it("完全被重叠段盖住的左旋钮 / 徽标不可见", () => {
        expect(isClipHeaderControlVisible({ leftPx: 20, widthPx: 14 }, overlapLayout)).toBe(false);
        expect(isClipHeaderControlVisible({ leftPx: 28, widthPx: 20 }, overlapLayout)).toBe(false);
    });

    it("跨在重叠边界上的控件也不可见（不允许半个控件可点）", () => {
        expect(isClipHeaderControlVisible({ leftPx: 92, widthPx: 20 }, overlapLayout)).toBe(false);
    });

    it("完全落在可见 header 区的控件可见", () => {
        expect(isClipHeaderControlVisible({ leftPx: 104, widthPx: 20 }, overlapLayout)).toBe(true);
    });

    it("右侧标签按左缘是否越过重叠边界判断", () => {
        expect(isClipHeaderRightLabelVisible({ leftPx: 60 }, overlapLayout)).toBe(false);
        expect(isClipHeaderRightLabelVisible({ leftPx: 110 }, overlapLayout)).toBe(true);
    });
});
