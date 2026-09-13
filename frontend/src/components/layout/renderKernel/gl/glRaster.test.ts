/**
 * GL 光栅化参数（./glRaster）行为自检。
 *
 * 【主要内容】
 * 1. 整数 DPR：物理尺寸 = CSS × DPR，绘制尺寸回算等于 CSS；
 * 2. 分数 DPR 与非整数 CSS：物理尺寸取整，绘制尺寸 = 物理 / DPR；
 * 3. 非法输入回退（DPR → 1，尺寸至少 1）。
 *
 * 【作用】`u_resolution` 必须用回算后的绘制尺寸，否则分数 DPR 下顶点坐标与物理
 * 像素不 1:1，网格线相位会随滚动漂移。本文件锁死这条换算契约。
 *
 * 【与其他模块的关系】覆盖 `glRaster.ts`；不依赖 DOM / WebGL。
 */

import { describe, expect, it } from "vitest";

import { resolveGlRasterTarget } from "./glRaster";

describe("resolveGlRasterTarget", () => {
    it("整数 DPR：物理尺寸 = CSS × DPR，绘制尺寸回算等于 CSS", () => {
        const target = resolveGlRasterTarget(1920, 1080, 2);
        expect(target.physicalWidthPx).toBe(3840);
        expect(target.physicalHeightPx).toBe(2160);
        expect(target.cssWidthPx).toBe(1920);
        expect(target.cssHeightPx).toBe(1080);
        expect(target.dpr).toBe(2);
    });

    it("分数 DPR：物理尺寸取整，绘制尺寸由物理尺寸回算", () => {
        const target = resolveGlRasterTarget(1500, 800, 1.25);
        expect(target.physicalWidthPx).toBe(1875);
        expect(target.physicalHeightPx).toBe(1000);
        expect(target.cssWidthPx).toBe(1500);
        expect(target.cssHeightPx).toBe(800);
    });

    it("非整数 CSS 尺寸：绘制尺寸 = 取整后的物理尺寸 / DPR", () => {
        const target = resolveGlRasterTarget(1500.5, 800.25, 2);
        // round(1500.5 × 2) = 3001；round(800.25 × 2) = round(1600.5) = 1601
        expect(target.physicalWidthPx).toBe(3001);
        expect(target.physicalHeightPx).toBe(1601);
        expect(target.cssWidthPx).toBe(1500.5);
        expect(target.cssHeightPx).toBe(800.5);
    });

    it("非法 DPR 回退为 1", () => {
        expect(resolveGlRasterTarget(800, 600, 0).dpr).toBe(1);
        expect(resolveGlRasterTarget(800, 600, Number.NaN).dpr).toBe(1);
        expect(resolveGlRasterTarget(800, 600, -2).dpr).toBe(1);
    });

    it("非法 / 过小尺寸回退到至少 1 物理像素", () => {
        const target = resolveGlRasterTarget(0, Number.NaN, 2);
        expect(target.physicalWidthPx).toBe(2);
        expect(target.physicalHeightPx).toBe(2);
        expect(target.cssWidthPx).toBe(1);
        expect(target.cssHeightPx).toBe(1);
    });
});
