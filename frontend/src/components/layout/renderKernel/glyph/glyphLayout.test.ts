/**
 * 字形布局（./glyphLayout）行为自检。
 *
 * 【主要内容】
 * 1. 按字符切分并累加 x 偏移；
 * 2. 超宽截断并追加省略号，结果宽度恒不超过上限；
 * 3. 上限小于省略号宽度时返回空序列（不越界绘制）；
 * 4. 测量结果按 `(字符, 字体)` 缓存，不同字体不共用；
 * 5. 非法测量值按 0 归一化，不产生 NaN 偏移。
 *
 * 【作用】文字由字形四边形拼出后，布局层是「与 Canvas2D 排版等价」的唯一保证；
 * 缓存与归一化失效会让每帧重复测量或产生 NaN 实例数据（整批 draw call 失效）。
 *
 * 【与其他模块的关系】覆盖 `glyphLayout.ts`；不依赖 DOM / Canvas / React。
 */

import { describe, expect, it, vi } from "vitest";

import { createGlyphLayout, ELLIPSIS_CHAR } from "./glyphLayout";

/** 造一个「每字符固定宽度」的测量桩，便于手算期望值。 */
function measureStub(charWidthPx: number): (text: string) => number {
    return (text) => text.length * charWidthPx;
}

describe("glyphLayout", () => {
    it("按字符切分为字形序列并累加 x 偏移", () => {
        const layout = createGlyphLayout(measureStub(10));
        const run = layout.layout("abc", "12px sans", 100);
        expect(run.glyphs.map((g) => g.char)).toEqual(["a", "b", "c"]);
        expect(run.glyphs.map((g) => g.x)).toEqual([0, 10, 20]);
        expect(run.glyphs.map((g) => g.width)).toEqual([10, 10, 10]);
        expect(run.width).toBe(30);
        expect(run.truncated).toBe(false);
    });

    it("超宽时截断并追加省略号，且宽度不超过上限", () => {
        const layout = createGlyphLayout(measureStub(10));
        // limit 35：a/b/c 依次放下（x=30），d 放不下 → 截断；省略号 10px 需要回退一个字形。
        const run = layout.layout("abcdef", "12px sans", 35);
        expect(run.truncated).toBe(true);
        expect(run.glyphs.map((g) => g.char)).toEqual(["a", "b", ELLIPSIS_CHAR]);
        expect(run.glyphs[2].x).toBe(20);
        expect(run.width).toBe(30);
        expect(run.width).toBeLessThanOrEqual(35);
    });

    it("上限小于省略号宽度时返回空序列", () => {
        const layout = createGlyphLayout(measureStub(10));
        const run = layout.layout("abc", "12px sans", 0);
        expect(run.glyphs).toEqual([]);
        expect(run.width).toBe(0);
        expect(run.truncated).toBe(true);
    });

    it("空文本返回空序列且不截断", () => {
        const layout = createGlyphLayout(measureStub(10));
        const run = layout.layout("", "12px sans", 100);
        expect(run.glyphs).toEqual([]);
        expect(run.width).toBe(0);
        expect(run.truncated).toBe(false);
    });

    it("测量结果按 (字符, 字体) 缓存，重复布局不重复测量", () => {
        const measure = vi.fn((text: string) => text.length * 10);
        const layout = createGlyphLayout(measure);
        layout.layout("aaa", "12px sans", 100);
        layout.layout("aa", "12px sans", 100);
        expect(measure.mock.calls.filter(([text]) => text === "a")).toHaveLength(1);
    });

    it("不同字体不共用缓存", () => {
        const measure = vi.fn(
            (text: string, fontKey: string) => text.length * (fontKey === "big" ? 20 : 10),
        );
        const layout = createGlyphLayout(measure);
        expect(layout.layout("a", "small", 100).width).toBe(10);
        expect(layout.layout("a", "big", 100).width).toBe(20);
        expect(layout.measureCacheSize()).toBe(2);
    });

    it("非法测量值按 0 处理（不产生 NaN 偏移）", () => {
        const layout = createGlyphLayout(() => Number.NaN);
        const run = layout.layout("abc", "12px sans", 100);
        expect(run.glyphs.map((g) => g.x)).toEqual([0, 0, 0]);
        expect(run.width).toBe(0);
        expect(run.truncated).toBe(false);
    });

    it("按 Unicode 码点切分（代理对不拆开）", () => {
        const layout = createGlyphLayout(measureStub(10));
        const run = layout.layout("😀a", "12px sans", 100);
        // 码元切分实现会得到 3 个字形（代理对被拆开），码点切分得到 2 个。
        expect(run.glyphs.map((g) => g.char)).toEqual(["😀", "a"]);
        // 桩按 text.length 计宽，代理对长度为 2 → 20px。
        expect(run.glyphs[0].width).toBe(20);
    });

    it("宽度恰好等于上限时不截断（off-by-one 边界）", () => {
        const layout = createGlyphLayout(measureStub(10));
        const run = layout.layout("abc", "12px sans", 30);
        // 若截断判定写成 `>=`，这里会退化成 ["a", "…"]。
        expect(run.glyphs.map((g) => g.char)).toEqual(["a", "b", "c"]);
        expect(run.width).toBe(30);
        expect(run.truncated).toBe(false);
    });

    it("Infinity 上限视为不限制宽度", () => {
        const layout = createGlyphLayout(measureStub(10));
        const run = layout.layout("abcdef", "12px sans", Number.POSITIVE_INFINITY);
        expect(run.glyphs).toHaveLength(6);
        expect(run.width).toBe(60);
        expect(run.truncated).toBe(false);
    });
});
