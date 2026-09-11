/**
 * wheel 输入归一化（./normalizeWheel）行为自检。
 *
 * 【主要内容】
 * 1. 三种 `deltaMode` 的换算：像素原样、行 × 行高、页 × 页高；
 * 2. 未知 `deltaMode` 按像素处理；
 * 3. 非法增量归零、非法上下文回退默认；
 * 4. `readWheelPixels` 双轴分别归一化。
 *
 * 【作用】滚轮是时间轴最高频输入；换算错误会让"滚一格"的手感在不同浏览器 /
 * 设备间分叉（Firefox 鼠标滚轮为 deltaMode=1），这里锁死量纲换算的契约。
 *
 * 【与其他模块的关系】覆盖 `normalizeWheel.ts`；不依赖 DOM 与 React。
 */

import { describe, expect, it } from "vitest";

import { normalizeWheelDelta, readWheelPixels } from "./normalizeWheel";

const CTX = { lineHeightPx: 16, pageHeightPx: 800 };

describe("normalizeWheelDelta", () => {
    it("deltaMode=0（像素）原样返回", () => {
        expect(normalizeWheelDelta(-12.5, 0, CTX)).toBe(-12.5);
        expect(normalizeWheelDelta(120, 0, CTX)).toBe(120);
    });

    it("deltaMode=1（行）按行高换算", () => {
        expect(normalizeWheelDelta(3, 1, CTX)).toBe(48);
        expect(normalizeWheelDelta(-2, 1, CTX)).toBe(-32);
    });

    it("deltaMode=2（页）按视口高度换算", () => {
        expect(normalizeWheelDelta(1, 2, CTX)).toBe(800);
        expect(normalizeWheelDelta(-1, 2, CTX)).toBe(-800);
    });

    it("未知 deltaMode 按像素处理", () => {
        expect(normalizeWheelDelta(7, 9, CTX)).toBe(7);
    });

    it("非法增量返回 0", () => {
        expect(normalizeWheelDelta(Number.NaN, 0, CTX)).toBe(0);
        expect(normalizeWheelDelta(Number.POSITIVE_INFINITY, 0, CTX)).toBe(0);
    });

    it("非法上下文回退到安全默认（行 16 / 页 800）", () => {
        expect(normalizeWheelDelta(1, 1, { lineHeightPx: 0, pageHeightPx: 0 })).toBe(16);
        expect(normalizeWheelDelta(1, 2, { lineHeightPx: 0, pageHeightPx: Number.NaN })).toBe(800);
    });
});

describe("readWheelPixels", () => {
    it("双轴分别归一化", () => {
        expect(readWheelPixels({ deltaX: 10, deltaY: 2, deltaMode: 1 }, CTX)).toEqual({
            x: 160,
            y: 32,
        });
    });

    it("像素模式双轴原样", () => {
        expect(readWheelPixels({ deltaX: -4, deltaY: 8, deltaMode: 0 }, CTX)).toEqual({
            x: -4,
            y: 8,
        });
    });
});
