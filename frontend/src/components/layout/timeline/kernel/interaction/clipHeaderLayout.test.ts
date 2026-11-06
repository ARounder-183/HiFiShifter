import { describe, expect, it } from "vitest";

import { isClipHeaderControlVisible, resolveClipHeaderLayout } from "./clipHeaderLayout";

describe("resolveClipHeaderLayout", () => {
    it("无重叠时名称与右侧标签保持原位置", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 300,
            trailingOverlapPx: 0,
            leadingControlsWidth: 80,
            showGainLabel: true,
            showPlaybackRate: true,
            gainLabelWidth: 45,
            rateLabelWidth: 22,
        });
        expect(layout.visibleRightPx).toBe(300);
        expect(layout.gainLabelRightPx).toBe(294);
        expect(layout.gainLabelLeftPx).toBe(249);
        expect(layout.rateLabelLeftPx).toBe(219);
        expect(layout.nameLeftPx).toBe(80);
        expect(layout.nameRightPx).toBe(215);
        expect(layout.nameVisible).toBe(true);
    });

    it("后序 Clip 覆盖尾部时名称与 dB/倍率整体左移", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 300,
            trailingOverlapPx: 100,
            leadingControlsWidth: 80,
            showGainLabel: true,
            showPlaybackRate: true,
            gainLabelWidth: 45,
            rateLabelWidth: 22,
        });
        expect(layout.visibleRightPx).toBe(200);
        expect(layout.gainLabelRightPx).toBe(194);
        expect(layout.gainLabelLeftPx).toBe(149);
        expect(layout.rateLabelLeftPx).toBe(119);
        expect(layout.nameRightPx).toBe(115);
        expect(layout.nameWidthPx).toBe(35);
        expect(layout.nameVisible).toBe(true);
    });

    it("尾部重叠过宽时先隐藏倍率、再隐藏 dB，不裁半个标签", () => {
        const rateHidden = resolveClipHeaderLayout({
            clipWidthPx: 300,
            trailingOverlapPx: 165,
            leadingControlsWidth: 80,
            showGainLabel: true,
            showPlaybackRate: true,
            gainLabelWidth: 45,
            rateLabelWidth: 22,
        });
        expect(rateHidden.gainLabelVisible).toBe(true);
        expect(rateHidden.rateLabelVisible).toBe(false);

        const bothHidden = resolveClipHeaderLayout({
            clipWidthPx: 300,
            trailingOverlapPx: 260,
            leadingControlsWidth: 80,
            showGainLabel: true,
            showPlaybackRate: true,
            gainLabelWidth: 45,
            rateLabelWidth: 22,
        });
        expect(bothHidden.gainLabelVisible).toBe(false);
        expect(bothHidden.rateLabelVisible).toBe(false);
        expect(bothHidden.nameVisible).toBe(false);
    });

    it("尾部遮挡宽度超过自身宽度时收敛到 Clip 宽度", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: 100,
            trailingOverlapPx: 999,
            leadingControlsWidth: 80,
            showGainLabel: true,
            showPlaybackRate: true,
            gainLabelWidth: 45,
            rateLabelWidth: 22,
        });
        expect(layout.trailingOverlapPx).toBe(100);
        expect(layout.visibleRightPx).toBe(0);
        expect(layout.nameVisible).toBe(false);
        expect(layout.gainLabelVisible).toBe(false);
        expect(layout.rateLabelVisible).toBe(false);
    });

    it("非有限输入按 0 处理，不产生 NaN 几何", () => {
        const layout = resolveClipHeaderLayout({
            clipWidthPx: Number.NaN,
            trailingOverlapPx: Number.NaN,
            leadingControlsWidth: Number.NaN,
            showGainLabel: true,
            showPlaybackRate: true,
            gainLabelWidth: Number.NaN,
            rateLabelWidth: Number.NaN,
        });
        expect(layout.visibleRightPx).toBe(1);
        expect(Number.isFinite(layout.nameLeftPx)).toBe(true);
        expect(Number.isFinite(layout.nameRightPx)).toBe(true);
        expect(Number.isFinite(layout.gainLabelLeftPx)).toBe(true);
    });
});

describe("控件可见性", () => {
    const layout = resolveClipHeaderLayout({
        clipWidthPx: 300,
        trailingOverlapPx: 100,
        leadingControlsWidth: 80,
        showGainLabel: true,
        showPlaybackRate: true,
        gainLabelWidth: 45,
        rateLabelWidth: 22,
    });

    it("完全落在可见 header 区的左旋钮 / 徽标可见", () => {
        expect(isClipHeaderControlVisible({ leftPx: 4, widthPx: 22 }, layout)).toBe(true);
        expect(isClipHeaderControlVisible({ leftPx: 28, widthPx: 20 }, layout)).toBe(true);
    });

    it("跨过尾部重叠边界或完全被覆盖的控件不可见", () => {
        expect(isClipHeaderControlVisible({ leftPx: 192, widthPx: 20 }, layout)).toBe(false);
        expect(isClipHeaderControlVisible({ leftPx: 210, widthPx: 20 }, layout)).toBe(false);
    });
});
