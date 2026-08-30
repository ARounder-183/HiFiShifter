/**
 * Clip 边缘交互几何的不变量。
 *
 * 覆盖两组常量：
 * 1. `fadeCornerReservePx` —— 左右边缘上"淡化角控件 vs 裁短/拉伸"的垂直
 *    所有权切分。回归背景①：原先固定 48px，在典型 Clip 高度（74–90px）上
 *    吃掉 53%–65% 的边缘，用户想裁短却在边缘偏上位置按下时命中的是淡化
 *    控件（"拉边界被判定成渐变"）；回归背景②：后改为 body×0.38 但封顶
 *    34px，行高 >96px 后拖拽区不再随轨道高度缩放（"看着是定长"）。
 *    现约定 = body 高度的 1/3，无全局封顶，行高 80–192 全覆盖。
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

    // ── 1. 淡化保留区 = body 的 1/3，裁短至少拿到 62% ───────────────
    // body = rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT，
    // 行高 80–192 → body 60–172。reserve = body/3 ⇒ trim 保住 2/3 ≥ 62%。
    for (const bodyHeight of [50, 60, 74, 90, 122, 172, 300]) {
        const reserve = fadeCornerReservePx(bodyHeight);
        const trimHeight = bodyHeight - reserve;
        assertTrue(
            trimHeight >= bodyHeight * 0.62 - 1e-9,
            `bodyHeight=${bodyHeight}: trim keeps >=62% of the edge (reserve=${reserve})`,
        );
        assertTrue(
            reserve >= FADE_CORNER_CAP_HEIGHT_PX - 1e-9,
            `bodyHeight=${bodyHeight}: reserve never smaller than the corner cap`,
        );
        // 无全局封顶：body 越大保留区越大（关键回归——曾被 34px 封死）。
        if (bodyHeight >= 3 * FADE_CORNER_CAP_HEIGHT_PX) {
            assertTrue(
                reserve > FADE_CORNER_CAP_HEIGHT_PX,
                `bodyHeight=${bodyHeight}: reserve exceeds the plain cap when body is tall enough`,
            );
        }
    }
    // 退化矮 body：reserve 退化为横帽高度本身（真边角必须有落点）。
    if (Math.abs(fadeCornerReservePx(20) - FADE_CORNER_CAP_HEIGHT_PX) > 1e-9) {
        throw new Error("degenerate short body falls back to the corner cap height");
    }

    // 具体数值锚点：行高 80 → body 60 → 20px；行高 96 → body 76 → 25px；
    // 行高 120 → body 100 → 33px；行高 192 → body 172 → 57px。
    // （被 34px 封顶压死的 120/192 两档必须显著大于 34px 之下沿。）
    const shortClip = fadeCornerReservePx(60);
    if (Math.abs(shortClip - 20) > 1e-9) {
        throw new Error(`short clip reserve: expected 20, received ${shortClip}`);
    }
    const defaultClip = fadeCornerReservePx(76);
    if (Math.abs(defaultClip - 25) > 1e-9) {
        throw new Error(`default-height clip reserve: expected 25, received ${defaultClip}`);
    }
    const tallClip = fadeCornerReservePx(172);
    if (Math.abs(tallClip - 57) > 1e-9) {
        throw new Error(`tall clip reserve: expected 57, received ${tallClip}`);
    }
    // 封顶回归：192 行高下保留区必须远超旧 34px 上限。
    if (tallClip <= 34 + 1e-9) {
        throw new Error(`tall clip reserve must exceed the old 34px cap, received ${tallClip}`);
    }

    // ── 2. 包络线采样自洽 ─────────────────────────────────────────
    assertTrue(
        FADE_LINE_HIT_STEP_PX < FADE_LINE_HIT_SIZE,
        "sampling step must be smaller than the hit block, or gaps appear along the envelope",
    );
});