/**
 * 字形图集分配器（./glyphAtlas）行为自检。
 *
 * 【主要内容】
 * 1. 同页货架内横向排列且互不重叠（含 padding 间隔）；
 * 2. 货架高度不足时起新货架，纵向不重叠；
 * 3. 单页放满后溢出到新页；页数达上限后返回 null；
 * 4. 尺寸超过单页边长、非法尺寸一律拒绝。
 *
 * 【作用】图集装箱错误会直接表现为「字形串色 / 覆盖」；本模块是纯逻辑，
 * 在这里锁死「不重叠」「不越界」「页数受限」三条不变量，渲染层才敢直接用槽位。
 *
 * 【与其他模块的关系】覆盖 `glyphAtlas.ts`；不依赖 DOM / WebGL / React。
 */

import { describe, expect, it } from "vitest";

import { createGlyphAtlas, type AtlasSlot } from "./glyphAtlas";

/** 判断两个槽位是否不重叠（允许边贴边）。 */
function disjoint(a: AtlasSlot, b: AtlasSlot): boolean {
    return a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
}

describe("glyphAtlas", () => {
    it("同页货架内横向排列且互不重叠", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 1, maxPages: 1 });
        const a = atlas.allocate(20, 12);
        const b = atlas.allocate(20, 12);
        expect(a).not.toBeNull();
        expect(b).not.toBeNull();
        expect(disjoint(a as AtlasSlot, b as AtlasSlot)).toBe(true);
        expect((a as AtlasSlot).page).toBe(0);
        expect((b as AtlasSlot).page).toBe(0);
    });

    it("槽位之间保留 padding 间隔", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 4, maxPages: 1 });
        const a = atlas.allocate(10, 10) as AtlasSlot;
        const b = atlas.allocate(10, 10) as AtlasSlot;
        expect(b.x - (a.x + a.w)).toBeGreaterThanOrEqual(4);
    });

    it("货架高度不足时起新货架，纵向不重叠", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 0, maxPages: 1 });
        const a = atlas.allocate(10, 20) as AtlasSlot;
        const b = atlas.allocate(10, 30) as AtlasSlot;
        expect(a.y).toBe(0);
        expect(b.y).toBe(20);
        expect(disjoint(a, b)).toBe(true);
        // 新货架内继续横向排列。
        const c = atlas.allocate(10, 30) as AtlasSlot;
        expect(c.y).toBe(b.y);
        expect(c.x).toBe(b.x + b.w);
    });

    it("单页放满后溢出到新页", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 32, paddingPx: 0, maxPages: 3 });
        const first = atlas.allocate(30, 30) as AtlasSlot;
        const second = atlas.allocate(30, 30) as AtlasSlot;
        expect(first.page).toBe(0);
        expect(second.page).toBe(1);
        expect(atlas.pageCount()).toBe(2);
    });

    it("页数达上限后返回 null", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 16, paddingPx: 0, maxPages: 1 });
        expect(atlas.allocate(16, 16)).not.toBeNull();
        expect(atlas.allocate(16, 16)).toBeNull();
    });

    it("尺寸超过单页边长直接拒绝（不浪费开页）", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 0, maxPages: 4 });
        expect(atlas.allocate(65, 10)).toBeNull();
        expect(atlas.allocate(10, 65)).toBeNull();
        expect(atlas.pageCount()).toBe(0);
    });

    it("非法尺寸返回 null", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 0, maxPages: 1 });
        expect(atlas.allocate(0, 10)).toBeNull();
        expect(atlas.allocate(10, -1)).toBeNull();
        expect(atlas.allocate(Number.NaN, 10)).toBeNull();
    });

    it("起新货架时保留纵向 padding", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 4, maxPages: 1 });
        const a = atlas.allocate(10, 20) as AtlasSlot;
        const b = atlas.allocate(10, 30) as AtlasSlot;
        // 若 nextShelfY 漏加 padding，这里会得到 20（而不是 >= 24）。
        expect(b.y - (a.y + a.h)).toBeGreaterThanOrEqual(4);
    });

    it("货架 x 溢出时换到新货架（而非新页）", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 24, paddingPx: 0, maxPages: 1 });
        const a = atlas.allocate(20, 10) as AtlasSlot;
        const b = atlas.allocate(20, 10) as AtlasSlot;
        expect(a.page).toBe(0);
        expect(b.page).toBe(0);
        expect(b.y).toBeGreaterThan(a.y);
        expect(disjoint(a, b)).toBe(true);
    });

    it("页数达上限后仍可复用已有页的空位", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 32, paddingPx: 0, maxPages: 1 });
        const first = atlas.allocate(10, 10) as AtlasSlot;
        const second = atlas.allocate(10, 10) as AtlasSlot;
        // 若实现改成「pages.length >= maxPages 就立刻返回 null」，本用例会拿到 null。
        expect(second.page).toBe(first.page);
        expect(atlas.pageCount()).toBe(1);
    });

    it("小数尺寸向上取整到整数像素", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 0, maxPages: 1 });
        const slot = atlas.allocate(10.5, 10.2) as AtlasSlot;
        expect(slot.w).toBe(11);
        expect(slot.h).toBe(11);
    });

    it("非法构造参数回退到最小可用值（不抛错）", () => {
        const atlas = createGlyphAtlas({ pageSizePx: 0, paddingPx: -1, maxPages: 0 });
        // pageSize / maxPages 回退为 1，padding 回退为 0：1×1 图集只放得下一个 1×1。
        expect(atlas.allocate(1, 1)).not.toBeNull();
        expect(atlas.allocate(1, 1)).toBeNull();
    });
});
