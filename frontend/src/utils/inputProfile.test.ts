import { describe, expect, it } from "vitest";

import {
    INPUT_PROFILES,
    precisionRampGain,
    profileFor,
    scaledHitRadius,
    scaledThreshold,
    SECONDARY_LONG_PRESS_MS,
    TOUCH_PRECISION_RAMP_GAIN,
    TOUCH_PRECISION_RAMP_PX,
    tiltToSkew,
    trackpadInertiaGain,
} from "./inputProfile";

describe("profileFor", () => {
    it("maps known pointer types to their profile", () => {
        expect(profileFor({ pointerType: "mouse" }).kind).toBe("mouse");
        expect(profileFor({ pointerType: "pen" }).kind).toBe("pen");
        expect(profileFor({ pointerType: "touch" }).kind).toBe("touch");
    });

    it("falls back to the mouse-shaped unknown profile", () => {
        expect(profileFor({ pointerType: "" }).kind).toBe("unknown");
        expect(profileFor({}).kind).toBe("unknown");
        expect(profileFor({ pointerType: null }).kind).toBe("unknown");
    });
});

describe("mouse profile is the identity element", () => {
    /*
     * 这是零回归的锚点：mouse 剖面上所有缩放必须是恒等，否则引入本模块就改变了
     * 既有手感 —— 而这类改动不会抛错，只能靠断言钉住。
     */
    it("scales hit radii by exactly 1", () => {
        const mouse = INPUT_PROFILES.mouse;
        expect(scaledHitRadius(9, mouse)).toBe(9);
        expect(scaledHitRadius(10, mouse)).toBe(10);
        expect(scaledHitRadius(8, mouse)).toBe(8);
    });

    it("scales thresholds by exactly 1", () => {
        const mouse = INPUT_PROFILES.mouse;
        expect(scaledThreshold(3, mouse)).toBe(3);
        expect(scaledThreshold(4, mouse)).toBe(4);
        expect(scaledThreshold(5, mouse)).toBe(5);
    });

    it("keeps hover side effects and grants a free hand", () => {
        const mouse = INPUT_PROFILES.mouse;
        expect(mouse.allowsHoverSideEffects).toBe(true);
        expect(mouse.hasHover).toBe(true);
        expect(mouse.freeHand).toBe(true);
        expect(mouse.dragGain).toBe(1);
    });

    it("does not need a long-press fallback", () => {
        expect(INPUT_PROFILES.mouse.secondaryLongPressMs).toBeNull();
    });

    it("behaves identically to the unknown fallback", () => {
        // 合成事件 / 测试桩必须与鼠标同构，否则既有测试会莫名其妙地变。
        const { kind: _mk, ...mouse } = INPUT_PROFILES.mouse;
        const { kind: _uk, ...unknown } = INPUT_PROFILES.unknown;
        expect(unknown).toEqual(mouse);
    });
});

describe("pen profile", () => {
    it("enlarges hit radii only slightly", () => {
        const pen = INPUT_PROFILES.pen;
        expect(scaledHitRadius(9, pen)).toBeCloseTo(10.35, 5);
        // 明显小于触摸的放大倍数：笔尖本身精度高，只需补偿"离屏幕远"。
        expect(pen.hitRadiusScale).toBeLessThan(INPUT_PROFILES.touch.hitRadiusScale);
    });

    it("keeps drag thresholds at the mouse baseline", () => {
        // 笔的落点精度高于鼠标，不需要更大的起手阈值。
        expect(scaledThreshold(3, INPUT_PROFILES.pen)).toBe(3);
    });

    it("has pressure and tilt, but no free hand", () => {
        const pen = INPUT_PROFILES.pen;
        expect(pen.hasPressure).toBe(true);
        expect(pen.hasTilt).toBe(true);
        expect(pen.freeHand).toBe(false);
    });

    it("suppresses hover side effects but still has hover", () => {
        const pen = INPUT_PROFILES.pen;
        expect(pen.hasHover).toBe(true);
        expect(pen.allowsHoverSideEffects).toBe(false);
    });

    it("offers a long-press fallback for pens without a barrel button", () => {
        expect(INPUT_PROFILES.pen.secondaryLongPressMs).toBe(SECONDARY_LONG_PRESS_MS);
    });
});

describe("touch profile", () => {
    it("enlarges hit radii by an order of magnitude", () => {
        const touch = INPUT_PROFILES.touch;
        expect(scaledHitRadius(9, touch)).toBeCloseTo(21.6, 5);
        expect(scaledHitRadius(10, touch)).toBeCloseTo(24, 5);
    });

    it("raises the drag start threshold above finger jitter", () => {
        const touch = INPUT_PROFILES.touch;
        expect(scaledThreshold(3, touch)).toBe(8);
        expect(scaledThreshold(4, touch)).toBeCloseTo(10.6667, 3);
    });

    it("has no hover and no free hand", () => {
        const touch = INPUT_PROFILES.touch;
        expect(touch.hasHover).toBe(false);
        expect(touch.allowsHoverSideEffects).toBe(false);
        expect(touch.freeHand).toBe(false);
    });

    it("declares touch-action none for drag surfaces", () => {
        expect(INPUT_PROFILES.touch.touchAction).toBe("none");
        // 鼠标 / 笔保留默认，避免误伤滚动。
        expect(INPUT_PROFILES.mouse.touchAction).toBe("auto");
        expect(INPUT_PROFILES.pen.touchAction).toBe("auto");
    });

    it("damps the drag gain slightly to compensate for the missing cursor", () => {
        expect(INPUT_PROFILES.touch.dragGain).toBeLessThan(1);
        expect(INPUT_PROFILES.touch.dragGain).toBeGreaterThan(0);
    });
});

describe("precisionRampGain", () => {
    it("is fine at the very start of a touch gesture", () => {
        expect(precisionRampGain(INPUT_PROFILES.touch, 0)).toBe(TOUCH_PRECISION_RAMP_GAIN);
        expect(precisionRampGain(INPUT_PROFILES.touch, 4)).toBe(TOUCH_PRECISION_RAMP_GAIN);
    });

    it("switches to full gain once past the ramp", () => {
        expect(precisionRampGain(INPUT_PROFILES.touch, TOUCH_PRECISION_RAMP_PX)).toBe(1);
        expect(precisionRampGain(INPUT_PROFILES.touch, 400)).toBe(1);
    });

    it("uses the absolute distance so dragging backwards stays on the ramp", () => {
        expect(precisionRampGain(INPUT_PROFILES.touch, -5)).toBe(TOUCH_PRECISION_RAMP_GAIN);
        expect(precisionRampGain(INPUT_PROFILES.touch, -50)).toBe(1);
    });

    it("never applies to mouse or pen", () => {
        // 鼠标有修饰键、笔有压力通道 —— 斜坡只补触摸缺的那一样。
        expect(precisionRampGain(INPUT_PROFILES.mouse, 0)).toBe(1);
        expect(precisionRampGain(INPUT_PROFILES.pen, 0)).toBe(1);
        expect(precisionRampGain(INPUT_PROFILES.unknown, 0)).toBe(1);
    });
});

describe("tiltToSkew", () => {
    it("is symmetric at a vertical pen", () => {
        expect(tiltToSkew(0)).toBe(0.5);
    });

    it("maps the full tilt range onto the skew range", () => {
        expect(tiltToSkew(90)).toBeCloseTo(0.98, 6);
        expect(tiltToSkew(-90)).toBeCloseTo(0.02, 6);
    });

    it("leaning right raises the skew and leaning left lowers it", () => {
        expect(tiltToSkew(45)).toBeGreaterThan(0.5);
        expect(tiltToSkew(-45)).toBeLessThan(0.5);
    });

    it("is monotonic across the whole range", () => {
        let previous = Number.NEGATIVE_INFINITY;
        for (let tilt = -90; tilt <= 90; tilt += 5) {
            const skew = tiltToSkew(tilt);
            expect(skew).toBeGreaterThan(previous);
            previous = skew;
        }
    });

    it("clamps beyond the hardware range instead of leaving the valid skew band", () => {
        // 某些驱动会给出略超 ±90 的值。
        expect(tiltToSkew(180)).toBeCloseTo(0.98, 6);
        expect(tiltToSkew(-180)).toBeCloseTo(0.02, 6);
    });

    it("falls back to the symmetric skew for non-finite input", () => {
        expect(tiltToSkew(Number.NaN)).toBe(0.5);
        expect(tiltToSkew(Number.POSITIVE_INFINITY)).toBe(0.5);
    });

    it("never returns a skew the renderer would reject", () => {
        // `clampSkew` 的有效带是 0.02..0.98；越界会被静默钳回，值看起来"拖不动"。
        for (let tilt = -360; tilt <= 360; tilt += 7) {
            const skew = tiltToSkew(tilt);
            expect(skew).toBeGreaterThanOrEqual(0.02);
            expect(skew).toBeLessThanOrEqual(0.98);
        }
    });
});

describe("trackpadInertiaGain", () => {
    it("is inert unless the user declares a trackpad", () => {
        // 默认 auto 下必须完全不影响既有行为。
        expect(trackpadInertiaGain(false, 0.4)).toBe(1);
        expect(trackpadInertiaGain(false, 40)).toBe(1);
    });

    it("boosts only the slow sub-pixel segment once declared", () => {
        expect(trackpadInertiaGain(true, 0.4)).toBeGreaterThan(1);
        expect(trackpadInertiaGain(true, 1.5)).toBeGreaterThan(1);
        // 快划段交给系统加速度曲线，不再叠加。
        expect(trackpadInertiaGain(true, 2)).toBe(1);
        expect(trackpadInertiaGain(true, 40)).toBe(1);
    });

    it("ignores a stationary pointer", () => {
        expect(trackpadInertiaGain(true, 0)).toBe(1);
    });
});
