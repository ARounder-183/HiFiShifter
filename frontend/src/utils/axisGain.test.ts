import { describe, expect, it } from "vitest";

import {
    advanceAxisGain,
    advanceAxisGain2D,
    createAxisGainState,
    createAxisGainState2D,
} from "./axisGain";

describe("advanceAxisGain", () => {
    it("starts at zero so the first frame produces no phantom movement", () => {
        const state = createAxisGainState();
        expect(advanceAxisGain(state, 0, 0.5)).toBe(0);
    });

    it("scales only the newly travelled segment", () => {
        const state = createAxisGainState();
        // 前 20px 按半速 → 10
        expect(advanceAxisGain(state, 20, 0.5)).toBe(10);
        // 再走 20px 按全速 → 10 + 20 = 30
        expect(advanceAxisGain(state, 40, 1)).toBe(30);
    });

    it("never rewrites the already accumulated amount when the gain changes", () => {
        /*
         * 这是本模块存在的全部理由：中途改倍率时，累计量必须连续。
         * 若实现是"总位移 × 当前倍率"，第二次调用会得到 40 × 1 = 40 而不是 30
         * —— 值会往前跳一段，正在进行的拖拽被打断。
         */
        const state = createAxisGainState();
        const first = advanceAxisGain(state, 20, 0.2);
        const second = advanceAxisGain(state, 20, 1);
        // 同一位置、倍率从 0.2 变到 1：不得凭空产生位移。
        expect(second).toBe(first);
    });

    it("stays continuous when the gain changes every frame", () => {
        const state = createAxisGainState();
        const gains = [0.3, 0.9, 0.1, 1.4, 0.5];
        let previous = 0;
        let raw = 0;
        for (const gain of gains) {
            raw += 10;
            const next = advanceAxisGain(state, raw, gain);
            // 单调向前：只可能走得更快或更慢，不可能倒退。
            expect(next).toBeGreaterThan(previous);
            // 且单帧的推进量绝不超过"全速走完这段"。
            expect(next - previous).toBeLessThanOrEqual(10 * Math.max(1, gain) + 1e-9);
            previous = next;
        }
    });

    it("treats a non-positive or non-finite gain as 1 rather than reversing", () => {
        // 反向移动比不动更糟：用户会看到值与手相反地走。
        const state = createAxisGainState();
        advanceAxisGain(state, 10, 0);
        expect(advanceAxisGain(state, 20, -3)).toBe(20);
        const other = createAxisGainState();
        advanceAxisGain(other, 10, 1);
        expect(advanceAxisGain(other, 20, Number.NaN)).toBe(20);
    });

    it("ignores a non-finite raw sample instead of poisoning the accumulator", () => {
        const state = createAxisGainState();
        advanceAxisGain(state, 10, 1);
        expect(advanceAxisGain(state, Number.NaN, 1)).toBe(10);
        // 后续合法采样仍然按"从 10 起"继续。
        expect(advanceAxisGain(state, 20, 1)).toBe(20);
    });

    it("supports dragging backwards", () => {
        const state = createAxisGainState();
        expect(advanceAxisGain(state, -10, 0.5)).toBe(-5);
        expect(advanceAxisGain(state, -20, 1)).toBe(-15);
    });
});

describe("advanceAxisGain2D", () => {
    it("applies the same scalar gain to both axes", () => {
        const state = createAxisGainState2D();
        expect(advanceAxisGain2D(state, 20, -40, 0.5)).toEqual({ x: 10, y: -20 });
    });

    it("keeps each axis's history independent", () => {
        const state = createAxisGainState2D();
        advanceAxisGain2D(state, 20, 0, 0.5);
        // 只有 X 继续走：Y 的累计不得被 X 的推进带动。
        expect(advanceAxisGain2D(state, 20, 0, 1)).toEqual({ x: 10, y: 0 });
    });

    it("is continuous across a gain change", () => {
        const state = createAxisGainState2D();
        const before = advanceAxisGain2D(state, 30, 30, 0.2);
        const after = advanceAxisGain2D(state, 30, 30, 1);
        expect(after).toEqual(before);
    });
});
