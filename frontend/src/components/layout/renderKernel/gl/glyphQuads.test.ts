/**
 * 字形四边形构建（./glyphQuads）行为自检。
 *
 * 【主要内容】
 * 1. 位置 = 文本起点 + 字形 x 偏移；尺寸取布局宽度与统一高度；
 * 2. uv 由槽位坐标除以页边长得到；
 * 3. 槽位缺失时跳过该字形（不产出占位四边形）；
 * 4. 空输入 / 非法页边长安全。
 *
 * 【作用】uv 换算是"字形贴图错位"的唯一风险点；跳过缺失槽位则是防止采样到
 * 邻居字形（乱码）的守卫。
 *
 * 【与其他模块的关系】覆盖 `glyphQuads.ts`；不依赖 DOM / WebGL。
 */

import { describe, expect, it } from "vitest";

import type { AtlasSlot } from "../glyph/glyphAtlas";
import type { LayoutGlyph } from "../glyph/glyphLayout";
import { buildGlyphQuads } from "./glyphQuads";
import type { Rgba } from "../instanceTypes";

const WHITE: Rgba = [1, 1, 1, 1];

function glyph(char: string, x: number, width: number): LayoutGlyph {
    return { char, x, width };
}

function slot(x: number, y: number, w: number, h: number): AtlasSlot {
    return { page: 0, x, y, w, h };
}

function build(
    glyphs: LayoutGlyph[],
    resolveSlot: (char: string) => AtlasSlot | null,
    overrides: Partial<Parameters<typeof buildGlyphQuads>[0]> = {},
) {
    return buildGlyphQuads({
        glyphs,
        originX: 100,
        originY: 200,
        heightPx: 12,
        atlasPageSizePx: 512,
        resolveSlot,
        rgba: WHITE,
        ...overrides,
    });
}

describe("buildGlyphQuads", () => {
    it("位置 = 文本起点 + 字形偏移，尺寸取布局宽度与统一高度", () => {
        const quads = build([glyph("a", 0, 7), glyph("b", 7, 8)], () => slot(0, 0, 7, 12));
        expect(quads).toHaveLength(2);
        expect(quads[0].x).toBe(100);
        expect(quads[0].y).toBe(200);
        expect(quads[0].w).toBe(7);
        expect(quads[0].h).toBe(12);
        expect(quads[1].x).toBe(107);
    });

    it("uv 由槽位坐标除以页边长得到", () => {
        const quads = build([glyph("a", 0, 7)], () => slot(64, 128, 7, 12));
        expect(quads[0].u0).toBeCloseTo(64 / 512, 10);
        expect(quads[0].v0).toBeCloseTo(128 / 512, 10);
        expect(quads[0].u1).toBeCloseTo(71 / 512, 10);
        expect(quads[0].v1).toBeCloseTo(140 / 512, 10);
    });

    it("槽位缺失时跳过该字形", () => {
        const quads = build([glyph("a", 0, 7), glyph("b", 7, 8)], (char) =>
            char === "a" ? slot(0, 0, 7, 12) : null,
        );
        expect(quads).toHaveLength(1);
        expect(quads[0].x).toBe(100);
    });

    it("空输入返回空数组", () => {
        expect(build([], () => slot(0, 0, 1, 1))).toEqual([]);
    });

    it("非法页边长按 1 处理（不产生 NaN uv）", () => {
        const quads = build([glyph("a", 0, 7)], () => slot(0, 0, 7, 12), {
            atlasPageSizePx: 0,
        });
        expect(Number.isFinite(quads[0].u0)).toBe(true);
        expect(quads[0].u0).toBe(0);
    });
});
