/**
 * 字形光栅化（./glyphRasterizer）行为自检。
 *
 * 【主要内容】
 * 1. 参数归一化（纯函数）：页边长 / 页数 / dpr 的兜底规则，尤其是 NaN 页边长
 *    必须回退而不是原样传递（NaN 会让 canvas 尺寸非法、上下文创建失败）；
 * 2. 无 DOM 环境（node 单测）构造返回 null 而不是抛错——调用方据此跳过文字渲染；
 * 3. CSS 字体简写解析：字号提取与 dpr 放大必须支持**带前缀**的简写
 *    （`bold 9px …`），见文件末尾 describe 的说明。
 *
 * 【作用】光栅化依赖离屏 Canvas2D，无法在 node 下完整验证；这里守住两条契约：
 * 参数兜底（真机传脏值时不静默失效）与环境降级（无 DOM 时不崩溃）。
 *
 * 【与其他模块的关系】覆盖 `glyphRasterizer.ts`；不依赖 DOM。
 */

import { describe, expect, it } from "vitest";

import {
    createGlyphRasterizer,
    parseFontSizePx,
    resolveGlyphRasterizerParams,
    scaleFontKey,
} from "./glyphRasterizer";

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

/**
 * CSS 字体简写解析：必须支持 `bold 9px …` 这类**带前缀**的简写。
 *
 * 【为什么单独守护】`render.ts` 的 C 音名标签用的是 `bold 9px ${family}`。
 * 原实现的字号正则带 `^` 锚定，遇到 bold 前缀会解析失败并**静默回退 12**，
 * 而 dpr 放大函数也匹配失败、原样返回 —— 结果是字形按未放大的字号光栅化到
 * 一个按 12px 算出的槽位里，尺寸与度量全错，且不报任何错。
 */
describe("CSS 字体简写解析（含 bold 前缀）", () => {
    it("解析普通简写的字号", () => {
        expect(parseFontSizePx("9px sans-serif")).toBe(9);
        expect(parseFontSizePx('12px "Segoe UI", Roboto')).toBe(12);
        expect(parseFontSizePx("8px sans-serif")).toBe(8);
    });

    it("解析带 bold 前缀的简写字号（回归：曾静默回退 12）", () => {
        expect(parseFontSizePx("bold 9px sans-serif")).toBe(9);
        expect(parseFontSizePx('  bold 12px "Segoe UI", Roboto  ')).toBe(12);
    });

    it("解析带 italic / 数字字重前缀的简写", () => {
        expect(parseFontSizePx("italic 10px sans-serif")).toBe(10);
        expect(parseFontSizePx("600 10px sans-serif")).toBe(10);
    });

    it("按 dpr 放大字号（回归：bold 曾原样返回、未放大）", () => {
        expect(scaleFontKey("9px sans-serif", 2)).toBe("18px sans-serif");
        expect(scaleFontKey("bold 9px sans-serif", 2)).toBe("bold 18px sans-serif");
        expect(scaleFontKey('italic 10px "Segoe UI"', 3)).toBe('italic 30px "Segoe UI"');
    });

    it("无法解析时回退到 12（既有契约不变）", () => {
        expect(parseFontSizePx("garbage")).toBe(12);
    });

    it("只改字号，不动字体族里的数字", () => {
        // 字体族可能含数字（如 "Arial 2"）；只有带 px 的那一段才是字号。
        expect(scaleFontKey("9px Arial 2", 2)).toBe("18px Arial 2");
    });
});
