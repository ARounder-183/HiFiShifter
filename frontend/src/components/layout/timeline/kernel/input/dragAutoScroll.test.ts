/**
 * 拖拽边缘自动滚屏（./dragAutoScroll）行为自检。
 *
 * 【主要内容】
 * 1. 带宽内线性加速、带宽外为 0；
 * 2. 方向符号：左缘为负、右缘为正（与 scrollLeft 增大方向一致）；
 * 3. 比例上限 1.5：拖出视口后不再继续加速；
 * 4. 步长与帧时长成正比（帧率无关），非法帧时长回退 1/60 秒；
 * 5. 非法 / 退化输入返回 0（不污染滚动位置）；
 * 6. `shouldAutoScrollForGesture` 只对横向位置类手势开放。
 *
 * 【作用】自动滚屏缺失是「使劲向右拖被卡住」的成因之一（内核对齐自绘滚动，
 * 浏览器不会替它滚屏）。这里锁死几何契约，避免以后调手感时改坏方向或让速度
 * 随刷新率漂移。
 *
 * 【与其他模块的关系】覆盖 `dragAutoScroll.ts`；不依赖 DOM 与 React。
 */

import { describe, expect, it } from "vitest";

import {
    DRAG_EDGE_SCROLL_BAND_PX,
    DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC,
    resolveDragEdgeScroll,
    shouldAutoScrollForGesture,
} from "./dragAutoScroll";

/** 视口：left=100、right=1100（宽 1000）。 */
const VIEW = { leftPx: 100, rightPx: 1100 };
const FRAME = 1000 / 60;

/** 以 60Hz 帧时长调用，返回本帧步长。 */
function step(clientX: number, frameMs = FRAME): number {
    return resolveDragEdgeScroll({ clientX, ...VIEW, frameMs });
}

describe("resolveDragEdgeScroll", () => {
    it("视口中央不滚屏", () => {
        expect(step(600)).toBe(0);
        expect(step(100 + DRAG_EDGE_SCROLL_BAND_PX)).toBe(0);
        expect(step(1100 - DRAG_EDGE_SCROLL_BAND_PX)).toBe(0);
    });

    it("左缘为负、右缘为正（方向与 scrollLeft 一致）", () => {
        expect(step(100 + DRAG_EDGE_SCROLL_BAND_PX - 4)).toBeLessThan(0);
        expect(step(1100 - DRAG_EDGE_SCROLL_BAND_PX + 4)).toBeGreaterThan(0);
    });

    it("带宽内越靠边越快（线性加速）", () => {
        const near = step(1100 - DRAG_EDGE_SCROLL_BAND_PX + 4);
        const mid = step(1100 - DRAG_EDGE_SCROLL_BAND_PX / 2);
        const edge = step(1100 - 1);
        expect(mid).toBeGreaterThan(near);
        expect(edge).toBeGreaterThan(mid);
    });

    it("指针拖出视口后不再继续加速（比例上限 1.5，在带外 16px 处饱和）", () => {
        // 比例 1.5 ⇒ 距边缘 1.5 × 带宽 = 48px 处饱和（右缘 1100 时即 clientX ≥ 1148）。
        const saturated = step(1100 + DRAG_EDGE_SCROLL_BAND_PX);
        expect(step(1100 + 5000)).toBeCloseTo(saturated, 6);
        // 饱和点之内的边缘（clientX = 1100，比例 1.0）仍小于饱和值。
        expect(step(1100)).toBeLessThan(saturated);
    });

    it("步长与帧时长成正比（帧率无关）", () => {
        const at60 = step(1100 - 1, 1000 / 60);
        const at120 = step(1100 - 1, 1000 / 120);
        const at30 = step(1100 - 1, 1000 / 30);
        expect(at120).toBeCloseTo(at60 / 2, 6);
        expect(at30).toBeCloseTo(at60 * 2, 6);
        // 同样的墙钟时间应滚过同样的距离。
        expect(at120 * 2).toBeCloseTo(at60, 6);
    });

    it("最大速度受常量约束（饱和后 = 1.5 倍速度 × 1 秒）", () => {
        const saturated = step(1100 + 5000, 1000);
        // 饱和比例 1.5、帧时长 1000ms（被夹到上限 100ms）⇒ 0.1 秒的量。
        expect(saturated).toBeCloseTo(DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC * 1.5 * 0.1, 6);
    });

    it("帧时长上限 100ms：挂起后恢复不会一次性跳过很长距离", () => {
        const longFrame = step(1100 + 5000, 5000);
        expect(longFrame).toBeCloseTo(step(1100 + 5000, 100), 6);
    });

    it("非法帧时长回退 1/60 秒，而不是停止滚屏", () => {
        const bad = step(1100 - 1, Number.NaN);
        const good = step(1100 - 1, 1000 / 60);
        expect(bad).toBeCloseTo(good, 6);
        expect(step(1100 - 1, 0)).toBeCloseTo(good, 6);
        expect(step(1100 - 1, -5)).toBeCloseTo(good, 6);
    });

    it("非有限指针坐标 / 退化视口返回 0", () => {
        expect(step(Number.NaN)).toBe(0);
        expect(step(Number.POSITIVE_INFINITY)).toBe(0);
        expect(
            resolveDragEdgeScroll({ clientX: 0, leftPx: 100, rightPx: 100, frameMs: FRAME }),
        ).toBe(0);
        expect(
            resolveDragEdgeScroll({ clientX: 0, leftPx: 200, rightPx: 100, frameMs: FRAME }),
        ).toBe(0);
    });

    it("带宽常量在正常视口宽度下不会互相重叠", () => {
        // 两侧带宽之和必须远小于常见视口宽度，否则中部全是"边缘"。
        expect(DRAG_EDGE_SCROLL_BAND_PX * 2).toBeLessThan(300);
    });
});

describe("shouldAutoScrollForGesture", () => {
    it("横向位置类手势需要自动滚屏", () => {
        expect(shouldAutoScrollForGesture("clip-drag")).toBe(true);
        expect(shouldAutoScrollForGesture("clip-trim")).toBe(true);
        expect(shouldAutoScrollForGesture("clip-fade")).toBe(true);
        expect(shouldAutoScrollForGesture("snap-offset-drag")).toBe(true);
    });

    it("纯纵向 / 短横向手势不滚屏（避免把视口带走）", () => {
        expect(shouldAutoScrollForGesture("gain-drag")).toBe(false);
        expect(shouldAutoScrollForGesture("crossfade-grip")).toBe(false);
        expect(shouldAutoScrollForGesture("none")).toBe(false);
        expect(shouldAutoScrollForGesture("pending-select")).toBe(false);
        expect(shouldAutoScrollForGesture("seek")).toBe(false);
    });

    it("未知种类不滚屏（保守：宁可不动，也不要误滚）", () => {
        expect(shouldAutoScrollForGesture("something-new")).toBe(false);
    });
});
