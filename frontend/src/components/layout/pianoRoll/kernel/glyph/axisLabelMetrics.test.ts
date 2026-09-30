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
 *
 * 【第二轮】加高画布只解决了"槽位有没有地方放"，没解决"槽位被放在哪"。锚点夹取
 * （`axisTickLabelAnchorBounds`）当时按 em 盒模型把下方内缩写成 `字号/2 + 1`，
 * 而字形管线的槽位向下延伸 `字号 × 0.7` —— 少算 2px，最下方那条标签的槽位仍然
 * 探出绘图区 1px。因此本文件同时钉住"夹取后的槽位必须落在绘图区内"。
 */

import { describe, expect, test } from "vitest";

import { PARAM_EDITOR_BOTTOM_BAR_PX } from "../../constants";
import {
    AXIS_TICK_LABEL_DESCENT_PX,
    AXIS_TICK_LABEL_EDGE_MARGIN_PX,
    AXIS_TICK_LABEL_FONT_SIZE_PX,
    axisTickLabelAnchorBounds,
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

describe("axisTickLabelAnchorBounds（刻度标签两端的内缩）", () => {
    const h = 400;

    test("锚点区间让字形槽位整体落在绘图区内（含 1px 余量）", () => {
        const { minY, maxY } = axisTickLabelAnchorBounds(AXIS_TICK_LABEL_FONT_SIZE_PX, h);
        const half = AXIS_TICK_LABEL_FONT_SIZE_PX / 2;
        // 上端：槽位上缘 = minY − 字号/2（字形管线的 middle 基准）不高于绘图区上缘。
        expect(minY - half).toBeGreaterThanOrEqual(0);
        // 下端：槽位下缘 = maxY + 下探量，必须落在绘图区内并留有余量。
        // 【为什么不是 `maxY + half`】槽位**不是**以锚点为中心的：字形管线画的槽位高
        // `字号 × 1.2`、上缘在锚点上方 `字号/2`，因此它向下延伸 `字号 × 0.7`
        // （见 glyphMiddleSlotDescentPx）。上一版按下方的 `half` 夹取，最下方那条
        // 标签的槽位就探出绘图区 1px。
        const descent = glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX);
        expect(maxY + descent).toBe(h - AXIS_TICK_LABEL_EDGE_MARGIN_PX);
    });

    test("最下方标签的槽位不越出绘图区（回归：dB 的 -∞ 被裁）", () => {
        /*
         * 【这是本测试存在的理由】音量 / 动态的值域下界恰为 0，因此绘图区底边那条
         * 刻度**必然**存在，dB 单位下它的标签就是 `-∞`。只要锚点夹取少算 1px，
         * 这条标签的槽位（进而可能连墨迹）就压在/越过绘图区下缘 —— 用户看到的是
         * "左下角的负无穷下半截被挡"。
         */
        for (const height of [100, 200, 400, 720, 1080]) {
            const { maxY } = axisTickLabelAnchorBounds(AXIS_TICK_LABEL_FONT_SIZE_PX, height);
            const slotBottom = maxY + glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX);
            expect(slotBottom, `h=${height} 时槽位下缘越出绘图区`).toBeLessThanOrEqual(height);
        }
    });

    test("两端内缩：上 半个字 + 余量，下 下探量 + 余量（两者不对称）", () => {
        const { minY, maxY } = axisTickLabelAnchorBounds(AXIS_TICK_LABEL_FONT_SIZE_PX, h);
        expect(minY).toBe(AXIS_TICK_LABEL_FONT_SIZE_PX / 2 + AXIS_TICK_LABEL_EDGE_MARGIN_PX);
        expect(maxY).toBe(
            h -
                glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX) -
                AXIS_TICK_LABEL_EDGE_MARGIN_PX,
        );
        // 本字号下具体是上 6px / 下 8px：写死它们是为了让行高比或字号变化时立刻被发现。
        expect(minY).toBe(6);
        expect(h - maxY).toBe(8);
        // 最密的一档刻度是 12 条（resolveAxisStep 的上界），间距 ≥ h/11，
        // 两端内缩之和远小于该间距，不可能让相邻标签重叠。
        expect(h / 11).toBeGreaterThan((minY + (h - maxY)) * 2);
    });

    test("值域两端的锚点（y=0 与 y=h）都被拉回绘图区内", () => {
        const { minY, maxY } = axisTickLabelAnchorBounds(AXIS_TICK_LABEL_FONT_SIZE_PX, h);
        // 最上面那条刻度（视口上界）与最下面那条（0 / dB 的 -∞）：
        expect(Math.min(Math.max(0, minY), maxY)).toBe(minY);
        expect(Math.min(Math.max(h, minY), maxY)).toBe(maxY);
        // 关键性质：夹取之后，槽位整体落在 [0, h] 内。
        expect(minY - AXIS_TICK_LABEL_FONT_SIZE_PX / 2).toBeGreaterThanOrEqual(0);
        expect(maxY + glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX)).toBeLessThanOrEqual(
            h,
        );
    });

    test("画布下方预留量足以容下字形槽位（夹取与预留同源）", () => {
        // 槽位下缘相对锚点的最大下探量必须不超过画布为它预留的那一条：
        // 若 `AXIS_TICK_LABEL_DESCENT_PX` 小于下探量，即便锚点夹对了也会被画布裁掉。
        expect(AXIS_TICK_LABEL_DESCENT_PX).toBeGreaterThanOrEqual(
            glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX),
        );
    });

    test("退化输入不产生反转区间", () => {
        for (const [fontSize, height] of [
            [0, 400],
            [-10, 400],
            [Number.NaN, 400],
            [10, 0],
            [10, Number.NaN],
            // 绘图区比一个字还矮：区间塌缩到 minY（标签落在顶端，仍可读）。
            [10, 4],
        ] as const) {
            const { minY, maxY } = axisTickLabelAnchorBounds(fontSize, height);
            expect(Number.isFinite(minY)).toBe(true);
            expect(maxY).toBeGreaterThanOrEqual(minY);
        }
    });
});
