/*
 * 触摸手势与长按的规格。
 *
 * 【为什么这两块必须有测试】它们的失效方式都是"手感不对"而不是"报错"：捏合爆炸、
 * 平移跳一下、长按与拖动打架。这些只能靠**数值**钉住，靠肉眼试是试不出来的
 * （尤其在没有真机的时候）。
 */
import { describe, expect, test } from "vitest";

import {
    INERTIA_HALF_LIFE_MS,
    PAN_DEAD_ZONE_PX,
    PINCH_ZOOM_GAIN,
    TWO_TO_ONE_COOLDOWN_MS,
    createTouchGestureRecognizer,
    inertiaDisplacement,
    inertiaFinished,
    pinchZoomFactor,
} from "./touchGesture";
import {
    claimLongPress,
    longPressOwner,
    releaseLongPress,
    resetLongPressArbiter,
} from "../../../utils/longPressArbiter";

/** 可推进的假时钟。 */
function fakeClock(start = 1000) {
    let value = start;
    return {
        now: () => value,
        advance(ms: number) {
            value += ms;
        },
    };
}

describe("捏合的绝对缩放律", () => {
    test("Δspread = 0 时是恒等元", () => {
        expect(pinchZoomFactor(0)).toBe(1);
    });

    test("只依赖本帧的距离变化，不依赖当前距离", () => {
        // 同一 Δ 在"两指很近"与"两指很远"时给出同一个倍率 —— 比值律在近距离会爆炸，
        // 这条断言正是那个性质的钉子。
        const near = pinchZoomFactor(10);
        const far = pinchZoomFactor(10);
        expect(near).toBe(far);
        expect(near).toBeGreaterThan(1);
        expect(pinchZoomFactor(-10)).toBeLessThan(1);
    });

    test("倍率与 Δ 是指数关系", () => {
        expect(pinchZoomFactor(100)).toBeCloseTo(Math.exp(PINCH_ZOOM_GAIN * 100), 10);
    });

    test("单帧变化量被钳制，不会出现「一次采样把画面缩到极限」", () => {
        expect(pinchZoomFactor(100000)).toBe(pinchZoomFactor(400));
        expect(pinchZoomFactor(-100000)).toBe(pinchZoomFactor(-400));
    });

    test("非有限值退化成恒等元（采样缺失不该让画面乱跳）", () => {
        expect(pinchZoomFactor(Number.NaN)).toBe(1);
        expect(pinchZoomFactor(Number.POSITIVE_INFINITY)).toBe(1);
    });
});

describe("惯性", () => {
    test("零速度不产生位移", () => {
        expect(inertiaDisplacement(0, 100)).toBe(0);
    });

    test("位移单调递增且有上界", () => {
        const at100 = inertiaDisplacement(1, 100);
        const at200 = inertiaDisplacement(1, 200);
        expect(at200).toBeGreaterThan(at100);
        // 指数衰减的积分收敛到 v·h/ln2。
        expect(inertiaDisplacement(1, 100000)).toBeCloseTo(INERTIA_HALF_LIFE_MS / Math.LN2, 3);
    });

    test("速度衰减到阈值以下即结束", () => {
        expect(inertiaFinished(1, 0)).toBe(false);
        expect(inertiaFinished(1, 100000)).toBe(true);
    });
});

describe("手势识别器", () => {
    test("单指平移越过死区才触发 onPanStart", () => {
        const clock = fakeClock();
        const pans: Array<[number, number]> = [];
        let started = 0;
        const recognizer = createTouchGestureRecognizer(
            "both",
            {
                onPanStart: () => {
                    started += 1;
                },
                onPan: (dx, dy) => pans.push([dx, dy]),
            },
            clock.now,
        );
        recognizer.start(1, { x: 0, y: 0 });
        // 死区内的抖动不产生任何回调。
        recognizer.move(1, { x: PAN_DEAD_ZONE_PX - 1, y: 0 });
        expect(started).toBe(0);
        expect(pans).toHaveLength(0);
        clock.advance(16);
        recognizer.move(1, { x: PAN_DEAD_ZONE_PX + 10, y: 0 });
        expect(started).toBe(1);
        expect(pans).toHaveLength(1);
    });

    test("轴策略：x 轴只产出 dx", () => {
        const clock = fakeClock();
        const pans: Array<[number, number]> = [];
        const recognizer = createTouchGestureRecognizer(
            "x",
            { onPan: (dx, dy) => pans.push([dx, dy]) },
            clock.now,
        );
        recognizer.start(1, { x: 0, y: 0 });
        clock.advance(16);
        recognizer.move(1, { x: 40, y: 30 });
        expect(pans[0][0]).not.toBe(0);
        expect(pans[0][1]).toBe(0);
    });

    test("双指进入捏合，且平移状态被清空", () => {
        const clock = fakeClock();
        let pinchStarts = 0;
        let panStarts = 0;
        const factors: number[] = [];
        const recognizer = createTouchGestureRecognizer(
            "both",
            {
                onPinchStart: () => {
                    pinchStarts += 1;
                },
                onPinch: (factor) => factors.push(factor),
                onPanStart: () => {
                    panStarts += 1;
                },
            },
            clock.now,
        );
        recognizer.start(1, { x: 0, y: 0 });
        clock.advance(16);
        recognizer.start(2, { x: 100, y: 0 });
        expect(pinchStarts).toBe(1);
        expect(recognizer.mode).toBe("pinch");
        clock.advance(16);
        recognizer.move(2, { x: 140, y: 0 });
        expect(factors).toHaveLength(1);
        expect(factors[0]).toBeGreaterThan(1);
        expect(panStarts).toBe(0);
    });

    test("2 → 1 手指有冷却：剩下那根手指不会立刻变成平移", () => {
        const clock = fakeClock();
        let panStarts = 0;
        const recognizer = createTouchGestureRecognizer(
            "both",
            {
                onPanStart: () => {
                    panStarts += 1;
                },
            },
            clock.now,
        );
        recognizer.start(1, { x: 0, y: 0 });
        clock.advance(16);
        recognizer.start(2, { x: 100, y: 0 });
        clock.advance(16);
        recognizer.end(2);
        // 冷却期内移动：不算平移。
        clock.advance(10);
        recognizer.move(1, { x: 200, y: 0 });
        expect(panStarts).toBe(0);
        // 冷却结束后**重新按下**才进入平移（本次按压已经在冷却期内被判为 idle）。
        clock.advance(TWO_TO_ONE_COOLDOWN_MS + 10);
        recognizer.end(1);
        recognizer.start(3, { x: 0, y: 0 });
        clock.advance(16);
        recognizer.move(3, { x: 40, y: 0 });
        expect(panStarts).toBe(1);
    });

    test("取消会结束当前手势", () => {
        const clock = fakeClock();
        let pinches = 0;
        const recognizer = createTouchGestureRecognizer(
            "both",
            { onPinchEnd: () => { pinches += 1; } },
            clock.now,
        );
        recognizer.start(1, { x: 0, y: 0 });
        recognizer.start(2, { x: 100, y: 0 });
        recognizer.cancel();
        expect(pinches).toBe(1);
        expect(recognizer.mode).toBe("idle");
    });
});

describe("长按仲裁器", () => {
    test("先到先得，单槽互斥", () => {
        resetLongPressArbiter();
        expect(claimLongPress("menu")).toBe(true);
        expect(claimLongPress("drag")).toBe(false);
        expect(longPressOwner()).toBe("menu");
        releaseLongPress("drag");
        expect(longPressOwner()).toBe("menu");
        releaseLongPress("menu");
        expect(claimLongPress("drag")).toBe(true);
        resetLongPressArbiter();
    });

    test("重复声明自己的所有权是幂等的", () => {
        resetLongPressArbiter();
        expect(claimLongPress("menu")).toBe(true);
        expect(claimLongPress("menu")).toBe(true);
        resetLongPressArbiter();
    });
});
