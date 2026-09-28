import { expect, test } from "vitest";

import {
    snapToDevicePx,
    verticalHairlineGeometry,
    wholeDevicePxLength,
} from "./devicePixelLine.ts";

test("snapToDevicePx pins positions onto device-pixel boundaries", () => {
    // dpr=1：回到整数像素。
    expect(snapToDevicePx(100.4, 1)).toBe(100);
    expect(snapToDevicePx(100.6, 1)).toBe(101);
    // dpr=1.25（Windows 125%）：设备像素边界落在 0.8 的倍数上。
    expect(snapToDevicePx(100.4, 1.25)).toBe(100.8); // 125.5 → round → 126 → 100.8
    expect(snapToDevicePx(100.0, 1.25)).toBe(100); // 已对齐则原样返回
    // dpr=1.5（150%）：边界落在 2/3 的倍数上。
    expect(snapToDevicePx(10.2, 1.5)).toBe(10); // 15.3 → 15 → 10
    expect(snapToDevicePx(10.5, 1.5)).toBe(10 + 2 / 3); // 15.75 → 16 → 10.666…
    // 负数（光标滚出视口左侧）同样吸附。
    expect(snapToDevicePx(-12.3, 1)).toBe(-12);
});

test("snapToDevicePx output always lands on a device-pixel boundary", () => {
    for (const dpr of [1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
        for (let x = -50; x <= 50; x += 0.37) {
            const snapped = snapToDevicePx(x, dpr);
            expect(Math.abs(snapped * dpr - Math.round(snapped * dpr))).toBeLessThan(1e-9);
        }
    }
});

test("snapToDevicePx guards non-finite input and bad dpr", () => {
    expect(snapToDevicePx(Number.NaN, 1.25)).toBe(0);
    expect(snapToDevicePx(Number.POSITIVE_INFINITY, 1.25)).toBe(0);
    expect(snapToDevicePx(10, Number.NaN)).toBe(10);
    expect(snapToDevicePx(10, 0)).toBe(10);
    expect(snapToDevicePx(10, -1)).toBe(10);
});

test("wholeDevicePxLength keeps the nearest whole physical-pixel width", () => {
    expect(wholeDevicePxLength(1, 1)).toBe(1); // 100%：与旧 w-px 一致
    expect(wholeDevicePxLength(1, 1.25)).toBe(0.8); // 1 物理像素
    expect(wholeDevicePxLength(1, 1.5)).toBe(4 / 3); // 2 物理像素（round(1.5)=2）
    expect(wholeDevicePxLength(1, 2)).toBe(1); // 200%：2 物理像素 = 1 CSS px
    expect(wholeDevicePxLength(2, 2)).toBe(2); // 4 物理像素
});

test("wholeDevicePxLength never collapses to zero", () => {
    expect(wholeDevicePxLength(1, 10)).toBe(1); // 10 物理像素 = 1 CSS px（宽度不变小）
    expect(wholeDevicePxLength(0, 2)).toBe(1); // 非法长度回退 1 CSS px 意图
    expect(wholeDevicePxLength(Number.NaN, 2)).toBe(1);
    expect(wholeDevicePxLength(1, Number.NaN)).toBe(1);
    expect(wholeDevicePxLength(1, 0)).toBe(1);
});

test("snapped position + whole-physical width exactly covers whole device columns", () => {
    // 修复的正确性组合不变量：left 与 left+width 都在设备像素边界上。
    for (const dpr of [1, 1.25, 1.5, 2]) {
        const width = wholeDevicePxLength(1, dpr);
        for (let x = 0; x <= 100; x += 0.43) {
            const left = snapToDevicePx(x, dpr);
            const right = left + width;
            expect(Math.abs(right * dpr - Math.round(right * dpr))).toBeLessThan(1e-9);
        }
    }
});

/**
 * 标尺竖线"粗细不一"的回归（系统缩放率 > 1 时最明显）。
 *
 * 曾经标尺的竖线用 `width: 1px` + 小数位置绘制：1px CSS 在 dpr=1.25 下等于 1.25
 * 个物理像素，且位置的小数部分使覆盖度逐条变化 —— 同一排线有的 1 个物理像素、有的
 * 2 个，肉眼就是"粗细不一"。`verticalHairlineGeometry` 把位置与宽度**成对**取整到
 * 设备像素，这里钉死"恰好覆盖整数个物理像素、且边缘落在边界上"。
 */
test("verticalHairlineGeometry covers whole device columns at any dpr", () => {
    for (const dpr of [1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
        for (const x of [0, 0.3, 12.5, 123.456, 1000.02, -17.7]) {
            const { left, width } = verticalHairlineGeometry(x, 1, dpr);
            const deviceSpan = Math.round((left + width) * dpr) - Math.round(left * dpr);
            // 覆盖的物理像素数 = `wholeDevicePxLength(1, dpr)` 的取整结果
            // （dpr=1→1、1.25→1、1.5→2、2→2、3→3），不能是"跨两列各一半"。
            expect({ dpr, x, deviceSpan }).toEqual({
                dpr,
                x,
                deviceSpan: Math.max(1, Math.round(1 * dpr)),
            });
        }
    }
});

test("verticalHairlineGeometry edges land on device-pixel boundaries", () => {
    for (const dpr of [1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
        for (const x of [0, 0.3, 12.5, 123.456, 1000.02, -17.7]) {
            const { left, width } = verticalHairlineGeometry(x, 1, dpr);
            expect(Math.abs(left * dpr - Math.round(left * dpr))).toBeLessThan(1e-9);
            expect(Math.abs((left + width) * dpr - Math.round((left + width) * dpr))).toBeLessThan(
                1e-9,
            );
        }
    }
});

test("verticalHairlineGeometry keeps the line centred within half a device pixel", () => {
    for (const dpr of [1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
        for (const x of [0, 0.3, 12.5, 123.456, 1000.02, -17.7]) {
            const { left, width } = verticalHairlineGeometry(x, 1, dpr);
            expect(Math.abs(left + width / 2 - x)).toBeLessThanOrEqual(0.5 / dpr + 1e-9);
        }
    }
});

test("verticalHairlineGeometry rounds bar-line width to whole physical pixels", () => {
    for (const dpr of [1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
        const { left, width } = verticalHairlineGeometry(10, 2, dpr);
        const physical = Math.round(width * dpr);
        // 与 `wholeDevicePxLength(2, dpr)` 同一规则（四舍五入到最接近的整数物理像素）。
        expect(physical).toBe(Math.max(1, Math.round(2 * dpr)));
        // 宽度取整后左右边缘仍落在设备像素边界上。
        expect(Math.abs(left * dpr - Math.round(left * dpr))).toBeLessThan(1e-9);
        expect(Math.abs((left + width) * dpr - Math.round((left + width) * dpr))).toBeLessThan(
            1e-9,
        );
    }
});

test("verticalHairlineGeometry guards bad dpr", () => {
    for (const bad of [Number.NaN, 0, -2, Number.POSITIVE_INFINITY]) {
        const { left, width } = verticalHairlineGeometry(5, 1, bad);
        expect(Number.isFinite(left)).toBe(true);
        expect(width).toBe(1);
    }
});
