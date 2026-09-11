/**
 * 字形光栅化（./glyphRasterizer）行为自检。
 *
 * 【主要内容】
 * 1. 参数归一化（纯函数）：页边长 / 页数 / dpr 的兜底规则，尤其是 NaN 页边长
 *    必须回退而不是原样传递（NaN 会让 canvas 尺寸非法、上下文创建失败）；
 * 2. 无 DOM 环境（node 单测）构造返回 null 而不是抛错——调用方据此跳过文字渲染。
 *
 * 【作用】光栅化依赖离屏 Canvas2D，无法在 node 下完整验证；这里守住两条契约：
 * 参数兜底（真机传脏值时不静默失效）与环境降级（无 DOM 时不崩溃）。
 *
 * 【与其他模块的关系】覆盖 `glyphRasterizer.ts`；不依赖 DOM。
 */

import { describe, expect, it } from "vitest";

import { createGlyphRasterizer, resolveGlyphRasterizerParams } from "./glyphRasterizer";

describe("resolveGlyphRasterizerParams", () => {
    it("正常参数原样保留（页边长取整）", () => {
        expect(resolveGlyphRasterizerParams({ pageSizePx: 2048.7, maxPages: 3, dpr: 2 })).toEqual({
            pageSizePx: 2048,
            maxPages: 3,
            dpr: 2,
        });
    });

    it("NaN / 过小的页边长回退 16", () => {
        expect(
            resolveGlyphRasterizerParams({ pageSizePx: Number.NaN, maxPages: 1, dpr: 1 })
                .pageSizePx,
        ).toBe(16);
        expect(
            resolveGlyphRasterizerParams({ pageSizePx: 4, maxPages: 1, dpr: 1 }).pageSizePx,
        ).toBe(16);
    });

    it("非法页数回退 1，非法 dpr 回退 1", () => {
        const params = resolveGlyphRasterizerParams({
            pageSizePx: 512,
            maxPages: 0,
            dpr: -2,
        });
        expect(params.maxPages).toBe(1);
        expect(params.dpr).toBe(1);
    });
});

describe("createGlyphRasterizer", () => {
    it("无 DOM 环境返回 null（不抛错）", () => {
        expect(createGlyphRasterizer({ pageSizePx: 256, maxPages: 1, dpr: 2 })).toBeNull();
    });
});
