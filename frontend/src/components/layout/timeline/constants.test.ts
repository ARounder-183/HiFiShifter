/**
 * Clip 边缘交互几何的不变量。
 *
 * 覆盖两组常量：
 * 1. `fadeCornerReservePx` —— 左右边缘上"淡化角控件 vs 裁短/拉伸"的垂直
 *    所有权切分。回归背景：原先固定 48px，在典型 Clip 高度（74–90px）上吃掉
 *    53%–65% 的边缘，用户想裁短却在边缘偏上位置按下时命中的是淡化控件
 *    （"拉边界被判定成渐变"）。
 * 2. `fadeHitTargets` 的采样常量自洽性 —— 步长必须小于命中块边长，否则
 *    沿包络线会出现无法命中的空隙。
 */
import { test } from "vitest";

import { FADE_CORNER_CAP_HEIGHT_PX, FADE_CORNER_RESERVE_MAX_PX, fadeCornerReservePx } from "./constants";
import { FADE_LINE_HIT_SIZE, FADE_LINE_HIT_STEP_PX } from "./fadeHitTargets";

test("components/layout/timeline/constants.test.ts scripted checks", async () => {
    function assertTrue(condition: boolean, label: string): void {
        if (!condition) throw new Error(`${label}: expected true`);
    }

    // ── 1. trim 至少拿到 62% 的边缘高度 ──────────────────────────
    // Clip 实际高度 = rowHeight - CLIP_BODY_PADDING_Y，行高范围 80–192 →
    // 实际 74–186。62% 保证在 reserve 触到横帽下限（18px）之前都成立，
    // 即 clipHeight >= 18/0.38 ≈ 47.4 时；更矮的退化尺寸由下限兜底。
    for (const clipHeight of [50, 74, 90, 122, 186, 300]) {
        const reserve = fadeCornerReservePx(clipHeight);
        const trimHeight = clipHeight - reserve;
        assertTrue(
            trimHeight >= clipHeight * 0.62 - 1e-9,
            `clipHeight=${clipHeight}: trim keeps >=62% of the edge (reserve=${reserve})`,
        );
        assertTrue(
            reserve <= FADE_CORNER_RESERVE_MAX_PX + 1e-9,
            `clipHeight=${clipHeight}: reserve capped (got ${reserve})`,
        );
        assertTrue(
            reserve >= FADE_CORNER_CAP_HEIGHT_PX - 1e-9,
            `clipHeight=${clipHeight}: reserve never smaller than the corner cap`,
        );
    }
    // 退化矮 body：reserve 退化为横帽高度本身（真边角必须有落点）。
    if (Math.abs(fadeCornerReservePx(20) - FADE_CORNER_CAP_HEIGHT_PX) > 1e-9) {
        throw new Error("degenerate short body falls back to the corner cap height");
    }

    // 具体数值锚点：回归防护，防止未来调参时无意回到"吃掉大半边缘"。
    // rowHeight=80 → clipHeight=74 → 旧 48px 只给 trim 留 26px(35%)。
    const shortClip = fadeCornerReservePx(74);
    if (!(Math.abs(shortClip - 28.12) < 0.01)) {
        throw new Error(`short clip reserve: expected ~28.12, received ${shortClip}`);
    }
    const defaultClip = fadeCornerReservePx(90);
    if (Math.abs(defaultClip - FADE_CORNER_RESERVE_MAX_PX) > 1e-9) {
        throw new Error(
            `default-height clip hits the cap: expected ${FADE_CORNER_RESERVE_MAX_PX}, received ${defaultClip}`,
        );
    }

    // ── 2. 包络线采样自洽 ─────────────────────────────────────────
    assertTrue(
        FADE_LINE_HIT_STEP_PX < FADE_LINE_HIT_SIZE,
        "sampling step must be smaller than the hit block, or gaps appear along the envelope",
    );
});
