import { describe, expect, it } from "vitest";

import { hitClipHeaderControl, type ClipHeaderControlStyle } from "./clipHeaderControls";

/**
 * 构造一个「全控件可见」的样式，偏移规则与 `buildTimelineClipVisualStyle` 一致：
 * 旋钮圆心 (15,10) r=5；链 x=28；静音 x=50；共振峰 x=72；徽标均 20×14 @ y=3。
 */
function makeStyle(overrides: Partial<ClipHeaderControlStyle> = {}): ClipHeaderControlStyle {
    return {
        showMuteBadge: true,
        showChainBadge: true,
        showFormantBadge: true,
        showGainKnob: true,
        showGainLabel: true,
        showPlaybackRate: true,
        showName: true,
        muteBadgeWidth: 20,
        muteBadgeHeight: 14,
        muteBadgeOffsetX: 50,
        muteBadgeOffsetY: 3,
        chainBadgeWidth: 20,
        chainBadgeHeight: 14,
        chainBadgeOffsetX: 28,
        chainBadgeOffsetY: 3,
        formantBadgeWidth: 20,
        formantBadgeHeight: 14,
        formantBadgeOffsetX: 72,
        formantBadgeOffsetY: 3,
        gainKnobCenterOffsetX: 15,
        gainKnobCenterOffsetY: 10,
        gainKnobRadius: 5,
        leadingControlsWidth: 102,
        trailingReservePx: 60,
        gainLabelWidth: 40,
        rateLabelWidth: 16,
        ...overrides,
    };
}

describe("hitClipHeaderControl", () => {
    const CLIP_WIDTH = 300;

    it("命中静音徽标", () => {
        expect(
            hitClipHeaderControl({
                localX: 60,
                localY: 10,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
            }),
        ).toBe("mute");
    });

    it("命中共振峰徽标", () => {
        expect(
            hitClipHeaderControl({
                localX: 80,
                localY: 10,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
            }),
        ).toBe("formant");
    });

    it("命中链徽标", () => {
        expect(
            hitClipHeaderControl({
                localX: 38,
                localY: 10,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
            }),
        ).toBe("chain");
    });

    it("命中增益旋钮（圆形判定，圆外不算）", () => {
        const style = makeStyle();
        expect(
            hitClipHeaderControl({ localX: 15, localY: 10, clipWidthPx: CLIP_WIDTH, style }),
        ).toBe("gain-knob");
        // 圆心右侧 9px：超出半径 5 + 容差 2 → 不命中旋钮
        expect(
            hitClipHeaderControl({ localX: 24, localY: 10, clipWidthPx: CLIP_WIDTH, style }),
        ).not.toBe("gain-knob");
    });

    it("命中增益标签（右对齐：gainX = 宽度 − 标签宽 − 6）", () => {
        // gainX = 300 − 40 − 6 = 254，标签占据 [254, 294]
        expect(
            hitClipHeaderControl({
                localX: 274,
                localY: 9,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
            }),
        ).toBe("gain-label");
    });

    it("命中速率标签（在增益标签左侧：rateX = gainX − 速率宽 − 8）", () => {
        // rateX = 254 − 16 − 8 = 230，标签占据 [230, 246]
        expect(
            hitClipHeaderControl({
                localX: 238,
                localY: 9,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
            }),
        ).toBe("rate-label");
    });

    it("命中名称区（左侧控件与右侧标签之间）", () => {
        expect(
            hitClipHeaderControl({
                localX: 150,
                localY: 9,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
            }),
        ).toBe("name");
    });

    it("控件不可见时不命中", () => {
        const style = makeStyle({
            showMuteBadge: false,
            showGainLabel: false,
            showPlaybackRate: false,
        });
        expect(
            hitClipHeaderControl({ localX: 60, localY: 10, clipWidthPx: CLIP_WIDTH, style }),
        ).not.toBe("mute");
        expect(
            hitClipHeaderControl({ localX: 274, localY: 9, clipWidthPx: CLIP_WIDTH, style }),
        ).not.toBe("gain-label");
    });

    it("clip 太窄时名称区退化为 null（不误判为名称）", () => {
        const style = makeStyle({ showGainLabel: false, showPlaybackRate: false });
        // nameLeft = 102 但 nameRight = 80 − 60 + 4 = 24 → 名称区无效
        expect(hitClipHeaderControl({ localX: 100, localY: 9, clipWidthPx: 80, style })).toBe(null);
    });

    it("纵向超出 header 高度时不命中任何控件", () => {
        expect(
            hitClipHeaderControl({
                localX: 60,
                localY: 40,
                clipWidthPx: CLIP_WIDTH,
                style: makeStyle(),
                headerHeightPx: 16,
            }),
        ).toBe(null);
    });
});
