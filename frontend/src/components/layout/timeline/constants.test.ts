/**
 * Clip 边缘交互几何的不变量。
 *
 * 覆盖两组常量：
 * 1. `fadeCornerReservePx` —— 左右边缘上"淡化角控件 vs 裁短/拉伸"的垂直
 *    所有权切分。这条规则被改过三轮，每轮的教训都留下：
 *      回归① 固定 48px，在典型 Clip 高度（74–90px）上吃掉 53%–65% 的边缘，
 *            用户想裁短却在边缘偏上按下时命中的是淡化控件；
 *      回归② 改 `body×0.38` 并封顶 34px，行高 >96px 后不再随高度缩放；
 *      回归③（Issue 141，用户实测）`body/3` 比例虽恒定，**绝对高度**却随行高
 *            从 20px 长到 57px（变化 2.85 倍），行高 192 时在边缘中部按下想裁短
 *            却命中渐变。
 *    现约定 = **恒等于横帽高度的定值**，不再随音频块高度缩放。
 *
 *    因此本文件的断言方向与回归②时期**相反**：那时要求"必须随高度增长、
 *    不得封顶"（`reserve > 14`），现在要求"必须是定值、且始终 ≤ 横帽高"。
 *    裁短区在支持范围内拿到 ≥76% 的边缘（`body/3` 时恒为 67%），回归①的
 *    意图被加强而非削弱。
 * 2. `fadeHitTargets` 的采样常量自洽性 —— 步长必须小于命中块边长，否则
 *    沿包络线会出现无法命中的空隙。
 */
import { test } from "vitest";

import { FADE_CORNER_CAP_HEIGHT_PX, fadeCornerReservePx } from "./constants";
import { FADE_LINE_HIT_SIZE, FADE_LINE_HIT_STEP_PX } from "./fadeHitTargets";

test("components/layout/timeline/constants.test.ts scripted checks", async () => {
    function assertTrue(condition: boolean, label: string): void {
        if (!condition) throw new Error(`${label}: expected true`);
    }

    // ── 1. 淡化保留区是定值，且始终小 ───────────────────────────────
    // body = rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT，
    // 支持的行高 80–192 → body 60–172。
    for (const bodyHeight of [60, 74, 76, 90, 100, 122, 172]) {
        const reserve = fadeCornerReservePx(bodyHeight);
        assertTrue(
            reserve === FADE_CORNER_CAP_HEIGHT_PX,
            `bodyHeight=${bodyHeight}: reserve is the fixed corner cap (received ${reserve})`,
        );
        const trimHeight = bodyHeight - reserve;
        assertTrue(
            trimHeight >= bodyHeight * 0.75 - 1e-9,
            `bodyHeight=${bodyHeight}: trim keeps >=75% of the edge (reserve=${reserve})`,
        );
    }

    // 超出支持范围（更矮 / 更高 / 非法值）也必须收敛：永远落在 [1, 横帽高]，
    // 且裁短至少拿到退化规则允许的那一半。
    for (const bodyHeight of [1, 2, 10, 20, 28, 50, 300, 4000]) {
        const reserve = fadeCornerReservePx(bodyHeight);
        assertTrue(
            reserve >= 1 && reserve <= FADE_CORNER_CAP_HEIGHT_PX,
            `bodyHeight=${bodyHeight}: reserve stays within [1, cap] (received ${reserve})`,
        );
        assertTrue(
            bodyHeight - reserve >= Math.floor(bodyHeight / 2),
            `bodyHeight=${bodyHeight}: trim keeps at least the degenerate half (reserve=${reserve})`,
        );
    }
    for (const bad of [0, Number.NaN, Number.POSITIVE_INFINITY, -5]) {
        assertTrue(
            fadeCornerReservePx(bad) === 1,
            `non-positive/non-finite body ${bad}: reserve collapses to a 1px landing spot`,
        );
    }

    // 反向断言（防回归②回潮）：保留区**不得**再随音频块高度增长。
    // 行高 80 与行高 192 的 body 相差近 3 倍，保留区必须一模一样。
    const shortClip = fadeCornerReservePx(60); // 行高 80
    const tallClip = fadeCornerReservePx(172); // 行高 192
    if (shortClip !== tallClip) {
        throw new Error(
            `reserve must not scale with clip height: body 60 -> ${shortClip}, body 172 -> ${tallClip}`,
        );
    }

    // 具体数值锚点：行高 80 / 96 / 120 / 192 全部落在同一个 14px。
    // （回归③之前这里是 20 / 25 / 33 / 57。）
    for (const [bodyHeight, expected] of [
        [60, 14],
        [76, 14],
        [100, 14],
        [172, 14],
    ] as const) {
        const reserve = fadeCornerReservePx(bodyHeight);
        if (reserve !== expected) {
            throw new Error(
                `body ${bodyHeight}: expected reserve ${expected}, received ${reserve}`,
            );
        }
    }

    // 退化矮 body（< 28）按 body/2 收敛，真边角仍有落点。
    for (const [bodyHeight, expected] of [
        [20, 10],
        [10, 5],
        [2, 1],
    ] as const) {
        const reserve = fadeCornerReservePx(bodyHeight);
        if (reserve !== expected) {
            throw new Error(
                `degenerate body ${bodyHeight}: expected ${expected}, received ${reserve}`,
            );
        }
    }

    // ── 2. 包络线采样自洽 ─────────────────────────────────────────
    assertTrue(
        FADE_LINE_HIT_STEP_PX < FADE_LINE_HIT_SIZE,
        "sampling step must be smaller than the hit block, or gaps appear along the envelope",
    );
});
