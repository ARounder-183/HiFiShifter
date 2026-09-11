/**
 * 网格实例构建（./gridInstances）行为自检。
 *
 * 【主要内容】
 * 1. 只产出构建窗口内的刻度；
 * 2. 强弱线按物理像素区分宽度（1 / 2 物理像素）；
 * 3. x 吸附到设备像素栅格；
 * 4. 竖直方向覆盖 [0, contentBottomPx]（与视口无关，保证滚动零重建）；
 * 5. 退化输入（无内容高度 / 非法坐标）被安全处理。
 *
 * 【作用】网格是"滚动零重绘"的关键验证点：一旦竖直方向开始按视口裁剪，
 * 几何就会随滚动变化，滚动帧必须重建——本文件的第 4 条断言正是这条约束的守卫。
 *
 * 【与其他模块的关系】覆盖 `gridInstances.ts`；不依赖 DOM / WebGL / React。
 */

import { describe, expect, it } from "vitest";

import { buildGridInstances, type GridTickLike } from "./gridInstances";
import type { Rgba } from "./instanceTypes";

const WEAK: Rgba = [1, 1, 1, 0.1];
const STRONG: Rgba = [1, 1, 1, 0.2];

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
        // 窗口 [150, 250]：仅 contentPx=200 命中。
        expect(out.map((instance) => instance.x)).toEqual([200]);
    });

    it("强弱线按物理像素区分宽度", () => {
        const out = build([makeTick(100), makeTick(200, true)]);
        // dpr=2：弱线 1 物理像素 = 0.5 CSS px；强线 2 物理像素 = 1 CSS px。
        expect(out[0].w).toBe(0.5);
        expect(out[1].w).toBe(1);
    });

    it("强弱线使用各自的颜色", () => {
        const out = build([makeTick(100), makeTick(200, true)]);
        expect(out[0].rgba).toBe(WEAK);
        expect(out[1].rgba).toBe(STRONG);
    });

    it("x 吸附到设备像素栅格", () => {
        const out = build([makeTick(100.3)], { dpr: 2 });
        // round(100.3 × 2) / 2 = 201 / 2 = 100.5
        expect(out[0].x).toBe(100.5);
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
