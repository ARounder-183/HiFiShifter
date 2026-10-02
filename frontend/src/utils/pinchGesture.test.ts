import { beforeEach, describe, expect, it, vi } from "vitest";

import {
    accumulatePinch,
    accumulatePinchSteps,
    clearPinchListeners,
    createPinchState,
    createPinchStepState,
    emitPinch,
    PINCH_FACTOR_PER_NOTCH,
    PINCH_GESTURE_GAP_MS,
    PINCH_UNITS_PER_NOTCH,
    pinchDeltaFromWheel,
    pinchZoomFactor,
    subscribePinch,
} from "./pinchGesture";

describe("pinchDeltaFromWheel", () => {
    it("only fires for ctrl / meta wheel (the platform's pinch signal)", () => {
        expect(pinchDeltaFromWheel({ deltaY: -50, ctrlKey: true, metaKey: false })).toBe(50);
        expect(pinchDeltaFromWheel({ deltaY: -50, ctrlKey: false, metaKey: true })).toBe(50);
        expect(pinchDeltaFromWheel({ deltaY: -50, ctrlKey: false, metaKey: false })).toBeNull();
    });

    it("treats spreading (negative deltaY) as zoom in", () => {
        expect(pinchDeltaFromWheel({ deltaY: -100, ctrlKey: true, metaKey: false })).toBe(100);
        expect(pinchDeltaFromWheel({ deltaY: 100, ctrlKey: true, metaKey: false })).toBe(-100);
    });

    it("normalizes line and page delta modes", () => {
        // deltaMode 1 = 行（16px/行），2 = 页（400px/页），与 normalizeWheel 同约定。
        expect(pinchDeltaFromWheel({ deltaY: -3, deltaMode: 1, ctrlKey: true, metaKey: false })).toBe(
            48,
        );
        expect(pinchDeltaFromWheel({ deltaY: -1, deltaMode: 2, ctrlKey: true, metaKey: false })).toBe(
            400,
        );
    });

    it("ignores zero and non-finite deltas", () => {
        expect(pinchDeltaFromWheel({ deltaY: 0, ctrlKey: true, metaKey: false })).toBeNull();
        expect(
            pinchDeltaFromWheel({ deltaY: Number.NaN, ctrlKey: true, metaKey: false }),
        ).toBeNull();
    });
});

describe("accumulatePinch", () => {
    it("accumulates within one gesture", () => {
        const state = createPinchState();
        expect(accumulatePinch(state, 30, 1000)).toBe(30);
        expect(accumulatePinch(state, 30, 1010)).toBe(60);
    });

    it("restarts after a gap so a stale remainder cannot leak in", () => {
        const state = createPinchState();
        accumulatePinch(state, 80, 1000);
        // 超时间隔 = 新手势：不得带着上一轮的 80 一起算。
        expect(accumulatePinch(state, 30, 1000 + PINCH_GESTURE_GAP_MS + 1)).toBe(30);
    });

    it("ignores non-finite deltas without corrupting the total", () => {
        const state = createPinchState();
        accumulatePinch(state, 50, 1000);
        expect(accumulatePinch(state, Number.NaN, 1010)).toBe(50);
        expect(accumulatePinch(state, 50, 1020)).toBe(100);
    });
});

describe("pinchZoomFactor", () => {
    it("reproduces the existing per-notch zoom exactly", () => {
        // 既有的离散规则是"每格 ×1.1"；连续版必须在整格处给出同一个值。
        expect(pinchZoomFactor(PINCH_UNITS_PER_NOTCH)).toBeCloseTo(PINCH_FACTOR_PER_NOTCH, 9);
        expect(pinchZoomFactor(PINCH_UNITS_PER_NOTCH * 2)).toBeCloseTo(
            PINCH_FACTOR_PER_NOTCH ** 2,
            9,
        );
    });

    it("is the identity at zero and inverts on the opposite direction", () => {
        expect(pinchZoomFactor(0)).toBe(1);
        expect(pinchZoomFactor(100) * pinchZoomFactor(-100)).toBeCloseTo(1, 9);
    });

    it("is monotonic and continuous (no per-notch steps)", () => {
        let previous = Number.NEGATIVE_INFINITY;
        for (let units = -300; units <= 300; units += 10) {
            const factor = pinchZoomFactor(units);
            expect(factor).toBeGreaterThan(previous);
            previous = factor;
        }
        // 半格处必须落在两格之间，而不是跳到整格的值。
        const half = pinchZoomFactor(PINCH_UNITS_PER_NOTCH / 2);
        expect(half).toBeGreaterThan(1);
        expect(half).toBeLessThan(PINCH_FACTOR_PER_NOTCH);
    });

    it("falls back to 1 for non-finite input", () => {
        expect(pinchZoomFactor(Number.NaN)).toBe(1);
    });
});

describe("accumulatePinchSteps", () => {
    it("emits nothing until a full notch accumulates", () => {
        // 这是修掉"触控板捏合让数值飞走"的关键：小 delta 不再每帧都算一步。
        const state = createPinchStepState();
        expect(accumulatePinchSteps(state, 20, 1000)).toBe(0);
        expect(accumulatePinchSteps(state, 20, 1010)).toBe(0);
        expect(accumulatePinchSteps(state, 20, 1020)).toBe(0);
        expect(accumulatePinchSteps(state, 20, 1030)).toBe(0);
        expect(accumulatePinchSteps(state, 20, 1040)).toBe(1);
    });

    it("keeps the remainder for the next notch", () => {
        const state = createPinchStepState();
        expect(accumulatePinchSteps(state, 150, 1000)).toBe(1);
        // 余 50，再来 50 就够第二格。
        expect(accumulatePinchSteps(state, 50, 1010)).toBe(1);
    });

    it("emits negative steps when pinching the other way", () => {
        const state = createPinchStepState();
        expect(accumulatePinchSteps(state, -250, 1000)).toBe(-2);
    });

    it("resets the remainder on a new gesture", () => {
        const state = createPinchStepState();
        accumulatePinchSteps(state, 90, 1000);
        // 新手势：余量归零，不得与上一轮凑成一格。
        expect(accumulatePinchSteps(state, 20, 1000 + PINCH_GESTURE_GAP_MS + 1)).toBe(0);
    });

    it("is rate-limited to the same speed as the discrete wheel", () => {
        // 一串小 delta 的总量决定步数，与"一格一个事件"的滚轮等价。
        const state = createPinchStepState();
        let steps = 0;
        for (let i = 0; i < 20; i += 1) {
            steps += accumulatePinchSteps(state, 10, 1000 + i * 8);
        }
        expect(steps).toBe(2);
    });
});

describe("pinch bus", () => {
    beforeEach(() => {
        clearPinchListeners();
    });

    it("delivers events to every subscriber", () => {
        const a = vi.fn();
        const b = vi.fn();
        subscribePinch(a);
        subscribePinch(b);
        emitPinch({ clientX: 10, clientY: 20, delta: 30 });
        expect(a).toHaveBeenCalledWith({ clientX: 10, clientY: 20, delta: 30 });
        expect(b).toHaveBeenCalledWith({ clientX: 10, clientY: 20, delta: 30 });
    });

    it("stops delivering after unsubscribe", () => {
        const listener = vi.fn();
        const off = subscribePinch(listener);
        emitPinch({ clientX: 0, clientY: 0, delta: 1 });
        off();
        emitPinch({ clientX: 0, clientY: 0, delta: 1 });
        expect(listener).toHaveBeenCalledTimes(1);
    });

    it("keeps going when one subscriber throws", () => {
        // 一个坏掉的表面不该让整个手势在其余表面上消失。
        const good = vi.fn();
        subscribePinch(() => {
            throw new Error("boom");
        });
        subscribePinch(good);
        expect(() => emitPinch({ clientX: 0, clientY: 0, delta: 1 })).not.toThrow();
        expect(good).toHaveBeenCalledTimes(1);
    });
});
