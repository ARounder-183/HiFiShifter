/**
 * 网格实例构建（./gridInstances）行为自检。
 *
 * 【主要内容】
 * 1. 只产出构建窗口内的刻度；
 * 2. 强弱线按 **CSS 像素**区分宽度（1 / 2），与旧实现 SVG 的 `strokeWidth` 同源；
 * 3. x 按**居中**语义吸附到设备像素栅格（旧实现是居中描边）；
 * 4. 竖直方向覆盖 [0, contentBottomPx]（与视口无关，保证滚动零重建）；
 * 5. 退化输入（无内容高度 / 非法坐标）被安全处理。
 *
 * 【作用】网格是"滚动零重绘"的关键验证点：一旦竖直方向开始按视口裁剪，
 * 几何就会随滚动变化，滚动帧必须重建——第 4 条断言正是这条约束的守卫。
 * 线宽与居中语义则是"与旧实现观感一致"的守卫（曾按物理像素实现，Retina 下偏细）。
 *
 * 【与其他模块的关系】覆盖 `gridInstances.ts`；不依赖 DOM / WebGL / React。
 */

import { describe, expect, it } from "vitest";

import { buildGridInstances, type GridTickLike } from "./gridInstances";
import type { Rgba } from "../../../renderKernel/instanceTypes";

const WEAK: Rgba = [1, 1, 1, 0.1];
const STRONG: Rgba = [1, 1, 1, 0.2];

/** 线条整体透明度（与 `gridInstances` 内部的常量一致）。 */
const OPACITY = 0.9;

function makeTick(contentPx: number, isStrongGridLine = false): GridTickLike {
    return { contentPx, isStrongGridLine };
}

function build(
    ticks: GridTickLike[],
    overrides: Partial<Parameters<typeof buildGridInstances>[0]> = {},
) {
    return buildGridInstances({
        ticks,
        windowLeftPx: 0,
        windowWidthPx: 1000,
        contentBottomPx: 400,
        dpr: 2,
        weakRgba: WEAK,
        strongRgba: STRONG,
        ...overrides,
    });
}

describe("buildGridInstances", () => {
    it("只产出构建窗口内的网格线", () => {
        const out = build([makeTick(0, true), makeTick(100), makeTick(200)], {
            windowLeftPx: 150,
            windowWidthPx: 100,
        });
        // 窗口 [150, 250]：仅 contentPx=200 命中；弱线居中 → 左缘 = 200 − 0.5。
        expect(out.map((instance) => instance.x)).toEqual([199.5]);
    });

    it("强弱线按 CSS 像素区分宽度（1 / 2，与旧实现 strokeWidth 同源）", () => {
        const out = build([makeTick(100), makeTick(200, true)]);
        // dpr 不参与线宽：Retina 下线更粗才与旧实现观感一致。
        expect(out[0].w).toBe(1);
        expect(out[1].w).toBe(2);
    });

    it("线宽不随 DPR 变化（CSS 像素语义）", () => {
        const at1 = build([makeTick(100)], { dpr: 1 });
        const at2 = build([makeTick(100)], { dpr: 3 });
        expect(at1[0].w).toBe(at2[0].w);
    });

    it("同一位置的重复刻度只产出一条（避免 alpha 累积变亮）", () => {
        // 旧实现是 SVG path（重复坐标只覆盖）；GL 逐实例绘制会让 alpha 翻倍。
        const out = build([makeTick(100), makeTick(100)]);
        expect(out).toHaveLength(1);
    });

    it("同一位置强弱重合时保留强线", () => {
        const out = build([makeTick(100), makeTick(100, true)]);
        expect(out).toHaveLength(1);
        expect(out[0].w).toBe(2);
        expect(out[0].rgba).toEqual([1, 1, 1, 0.2 * OPACITY]);
    });

    it("吸附后落在同一物理像素列的刻度也归并", () => {
        // dpr=2：99.9 与 100.1 的居中左缘都吸附到 199/2 = 99.5
        const out = build([makeTick(99.9), makeTick(100.1)]);
        expect(out).toHaveLength(1);
    });

    it("强弱线使用各自的颜色，并叠加整体透明度", () => {
        const out = build([makeTick(100), makeTick(200, true)]);
        expect(out[0].rgba).toEqual([1, 1, 1, 0.1 * OPACITY]);
        expect(out[1].rgba).toEqual([1, 1, 1, 0.2 * OPACITY]);
    });

    it("x 按居中语义吸附到设备像素栅格", () => {
        const out = build([makeTick(100.3)], { dpr: 2 });
        // 居中：左缘 = 100.3 − 0.5 = 99.8 → round(99.8 × 2) / 2 = 100
        expect(out[0].x).toBe(100);
    });

    it("竖直方向覆盖 [0, contentBottomPx]（与视口无关）", () => {
        const out = build([makeTick(100)], { contentBottomPx: 720 });
        expect(out[0].y).toBe(0);
        expect(out[0].h).toBe(720);
    });

    it("没有轨道内容时不产出实例", () => {
        expect(build([makeTick(100)], { contentBottomPx: 0 })).toEqual([]);
    });

    it("非法坐标被忽略，非法 DPR 回退为 1", () => {
        const out = build([makeTick(Number.NaN), makeTick(100)], { dpr: 0 });
        expect(out).toHaveLength(1);
        expect(out[0].w).toBe(1);
    });
});
