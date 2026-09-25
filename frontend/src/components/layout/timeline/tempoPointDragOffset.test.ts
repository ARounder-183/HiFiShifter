/**
 * 变化点拖拽阈值自检（"双击后输入框偏移"的回归）。
 *
 * 关键性质：阈值内的位移一律折算为 0（点击不改变变化点）；越过阈值后从阈值处
 * 起算，因此不会出现"刚越过就跳一格"。
 */
import { describe, expect, it } from "vitest";

import {
    resolveTempoDragOffsetPx,
    TEMPO_DRAG_THRESHOLD_PX,
    tempoSecUnderCursor,
} from "./tempoPointDragOffset";

describe("resolveTempoDragOffsetPx（拖拽启动阈值）", () => {
    it("★ 阈值内的抖动折算为 0（双击的第一下不会挪动变化点）", () => {
        for (const dx of [0, 1, -1, 2, -2, TEMPO_DRAG_THRESHOLD_PX, -TEMPO_DRAG_THRESHOLD_PX]) {
            expect(resolveTempoDragOffsetPx(dx)).toBe(0);
        }
    });

    it("越过阈值后从阈值处起算（不跳一格）", () => {
        expect(resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX + 1)).toBe(1);
        expect(resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX + 10)).toBe(10);
        expect(resolveTempoDragOffsetPx(-TEMPO_DRAG_THRESHOLD_PX - 7)).toBe(-7);
    });

    it("位移连续（阈值两侧不出现跳变）", () => {
        const below = resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX);
        const above = resolveTempoDragOffsetPx(TEMPO_DRAG_THRESHOLD_PX + 0.001);
        expect(Math.abs(above - below)).toBeLessThan(0.01);
    });

    it("非有限位移折算为 0（不把 NaN 带进位置计算）", () => {
        expect(resolveTempoDragOffsetPx(Number.NaN)).toBe(0);
        expect(resolveTempoDragOffsetPx(Number.POSITIVE_INFINITY)).toBe(0);
    });
});

describe("tempoSecUnderCursor（从输入框拖出时的落点换算）", () => {
    it("★ 光标在标签右侧时，落点即光标所在时间（不保留初始偏移）", () => {
        // 标签左缘在 x=400（对应 4.0s），缩放 100px/s，光标已拖到 x=460。
        const sec = tempoSecUnderCursor({
            pointSec: 4,
            flagLeftPx: 400,
            clientX: 460,
            pxPerSec: 100,
        });
        expect(sec).toBeCloseTo(4.6, 9);
    });

    it("光标在标签左侧时同样按光标位置落点", () => {
        const sec = tempoSecUnderCursor({
            pointSec: 4,
            flagLeftPx: 400,
            clientX: 350,
            pxPerSec: 100,
        });
        expect(sec).toBeCloseTo(3.5, 9);
    });

    it("光标正好在标签左缘时即变化点原位置", () => {
        const sec = tempoSecUnderCursor({
            pointSec: 2.5,
            flagLeftPx: 250,
            clientX: 250,
            pxPerSec: 100,
        });
        expect(sec).toBe(2.5);
    });

    it("非法输入回退为变化点原位置（不产生 NaN 时间）", () => {
        expect(
            tempoSecUnderCursor({
                pointSec: 1,
                flagLeftPx: Number.NaN,
                clientX: 10,
                pxPerSec: 100,
            }),
        ).toBe(1);
        expect(tempoSecUnderCursor({ pointSec: 1, flagLeftPx: 0, clientX: 10, pxPerSec: 0 })).toBe(
            1,
        );
    });
});
