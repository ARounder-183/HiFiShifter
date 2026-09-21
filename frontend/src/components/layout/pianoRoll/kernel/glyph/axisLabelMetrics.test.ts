/**
 * 纵轴刻度文字的画布下探量与**底部预留行**的约束回归。
 *
 * ## 背景（本测试要防的回归）
 *
 * 数值轴刻度标签以 `middle` 基准锚定在 `valueToY(v, heightPx)` 上，而值域下界
 * 恰好映射到 `heightPx` 本身 —— 最下方那条刻度（通常是 `0.0`）的文字中线正好
 * 落在绘图区下边缘，其字形槽位会向**下**超出一段（见
 * `glyphMiddleSlotDescentPx`）。因此轴画布必须比绘图区高出这一段，否则文字下半
 * 截被画布边界裁掉，观感是"最下面的刻度值被挡住了"。
 *
 * 而画布又必须落在纵轴列内（父层是 `overflow-hidden`），列比滚动视口高出的部分
 * 就是面板为自绘水平滚动条预留的那一行。于是形成一条**必须成立的量级约束**：
 *
 *     纵轴刻度文字的下探量  ≤  底部预留行高
 *
 * 这条约束此前不存在任何代码或测试上的连线：预留行写的是 Tailwind 的 `bottom-2`
 * （`0.5rem`，实际像素取决于根字号），画布高度写的是 `viewportHeightPx`。两者都
 * 改动过、却互不知情，才出现了"加了滚动条行以后最下面刻度值被挡"的问题。
 */

import { describe, expect, test } from "vitest";

import { PARAM_EDITOR_BOTTOM_BAR_PX } from "../../constants";
import {
    AXIS_TICK_LABEL_DESCENT_PX,
    AXIS_TICK_LABEL_FONT_SIZE_PX,
    glyphMiddleSlotDescentPx,
} from "./pianoRollGlyphs";
import { GLYPH_LINE_HEIGHT_RATIO } from "../../../renderKernel/glyph/glyphRasterizer";

describe("glyphMiddleSlotDescentPx", () => {
    test("等于 字号 × (行高比 − 0.5)", () => {
        for (const size of [8, 9, 10, 12, 16]) {
            expect(glyphMiddleSlotDescentPx(size)).toBeCloseTo(
                size * (GLYPH_LINE_HEIGHT_RATIO - 0.5),
                12,
            );
        }
    });

    test("10px 刻度字号的下探量为 7px", () => {
        // 10 × (1.2 − 0.5) = 7。写死这个数是为了让行高比变化时立刻被发现：
        // 它直接决定底部要预留多少像素。
        expect(glyphMiddleSlotDescentPx(10)).toBeCloseTo(7, 12);
    });

    test("非法字号返回 0（不产生负的预留量）", () => {
        expect(glyphMiddleSlotDescentPx(0)).toBe(0);
        expect(glyphMiddleSlotDescentPx(-10)).toBe(0);
        expect(glyphMiddleSlotDescentPx(Number.NaN)).toBe(0);
        expect(glyphMiddleSlotDescentPx(Number.POSITIVE_INFINITY)).toBe(0);
    });
});

describe("轴画布下探量与底部预留行的约束", () => {
    test("下探量必须 ≤ 底部预留行（否则画布超出纵轴列被裁掉）", () => {
        expect(AXIS_TICK_LABEL_DESCENT_PX).toBeLessThanOrEqual(PARAM_EDITOR_BOTTOM_BAR_PX);
    });

    test("下探量是向上取整的整数像素（画布尺寸必须是整数 CSS px）", () => {
        expect(Number.isInteger(AXIS_TICK_LABEL_DESCENT_PX)).toBe(true);
        expect(AXIS_TICK_LABEL_DESCENT_PX).toBe(
            Math.ceil(glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX)),
        );
    });

    test("字号的唯一真源：常量与使用点一致", () => {
        // 刻度字号同时出现在"字形请求的 fontKey"与"画布预留量"两处，
        // 必须来自同一个常量（曾分别是字面量 10 与独立的 8px 预留）。
        expect(AXIS_TICK_LABEL_FONT_SIZE_PX).toBe(10);
        expect(AXIS_TICK_LABEL_DESCENT_PX).toBeGreaterThan(0);
    });
});
