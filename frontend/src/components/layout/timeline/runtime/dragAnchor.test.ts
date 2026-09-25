/**
 * 拖拽锚点换算自检（"拖拽中滚动/缩放导致偏移"的回归）。
 *
 * 【要钉死的契约】落点与光标之间的**时间偏移**在拖拽全程恒定 —— 中途滚动或缩放
 * 都不改变它。这正是"抓在哪儿就一直在哪儿"。
 */
import { describe, expect, it } from "vitest";

import { deltaSecToContentPx, dragDeltaSec, pointerSecAt } from "./dragAnchor";

const RECT_LEFT = 256;

/** 模拟一次拖拽：返回给定视口下"对象应当落在的时间"。 */
function objectSecAt(args: {
    originObjectSec: number;
    startPointerSec: number;
    view: { scrollLeftPx: number; pxPerSec: number; clientX: number };
}): number {
    const pointerSec = pointerSecAt({
        scrollLeftPx: args.view.scrollLeftPx,
        pxPerSec: args.view.pxPerSec,
        clientX: args.view.clientX,
        rectLeft: RECT_LEFT,
    });
    return args.originObjectSec + dragDeltaSec(args.startPointerSec, pointerSec);
}

describe("拖拽锚点换算（pointerSecAt / dragDeltaSec）", () => {
    it("视口不变时，位移就是屏幕位移 / 缩放", () => {
        const start = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: 100,
            clientX: 500,
            rectLeft: RECT_LEFT,
        });
        const now = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: 100,
            clientX: 600,
            rectLeft: RECT_LEFT,
        });
        expect(dragDeltaSec(start, now)).toBeCloseTo(1, 9);
    });

    it("★ 中途缩放：对象与光标的时间偏移保持不变（不跳）", () => {
        // 按下：缩放 100px/s、未滚动、指针屏幕 x=500 → 指针时间 (500-256)/100 = 2.44s
        const startPointerSec = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: 100,
            clientX: 500,
            rectLeft: RECT_LEFT,
        });
        const originObjectSec = 4; // 抓在 clip 上，比光标晚 1.56s
        const grabOffset = originObjectSec - startPointerSec;

        // 放大到 200px/s（滚动不变）：指针没动，其时间变成 (500-256)/200 = 1.22s
        const afterZoom = objectSecAt({
            originObjectSec,
            startPointerSec,
            view: { scrollLeftPx: 0, pxPerSec: 200, clientX: 500 },
        });
        const pointerAfterZoom = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: 200,
            clientX: 500,
            rectLeft: RECT_LEFT,
        });
        // 关键断言：时间偏移不变 ⇒ 对象仍"粘"在光标上。
        expect(afterZoom - pointerAfterZoom).toBeCloseTo(grabOffset, 9);

        // 旧算法（起点是旧缩放下的内容像素、除以新缩放）：落点 = 4 + 244/200 = 5.22s，
        // 与光标的时间偏移变成 +4.0（本应 -1.56）—— 偏移 5.56s，正是用户看到的"严重偏移"。
        const legacy = originObjectSec + (500 - RECT_LEFT) / 200;
        expect(Math.abs(legacy - afterZoom)).toBeCloseTo(2.44, 6);
        expect(legacy - pointerAfterZoom).toBeCloseTo(4, 6);
        // 契约被破坏的量 = |旧偏移 − 应有偏移| = |4 − 1.56| = 2.44s。
        expect(Math.abs(legacy - pointerAfterZoom - grabOffset)).toBeCloseTo(2.44, 6);
    });

    it("★ 中途滚动：对象跟着光标走（时间偏移同样不变）", () => {
        const startPointerSec = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: 100,
            clientX: 500,
            rectLeft: RECT_LEFT,
        });
        const originObjectSec = 3;
        const grabOffset = originObjectSec - startPointerSec;

        for (const scrollLeftPx of [0, 37.5, 250, 1000.25]) {
            const objectSec = objectSecAt({
                originObjectSec,
                startPointerSec,
                view: { scrollLeftPx, pxPerSec: 100, clientX: 500 },
            });
            const pointerSec = pointerSecAt({
                scrollLeftPx,
                pxPerSec: 100,
                clientX: 500,
                rectLeft: RECT_LEFT,
            });
            expect(objectSec - pointerSec).toBeCloseTo(grabOffset, 9);
        }
    });

    it("★ 滚动与缩放同时发生：偏移依然不变", () => {
        const startPointerSec = pointerSecAt({
            scrollLeftPx: 120,
            pxPerSec: 80,
            clientX: 700,
            rectLeft: RECT_LEFT,
        });
        const originObjectSec = 6;
        const grabOffset = originObjectSec - startPointerSec;
        for (const [scrollLeftPx, pxPerSec] of [
            [120, 80],
            [-40, 240],
            [512.75, 33.3],
            [2000, 1600],
        ]) {
            const objectSec = objectSecAt({
                originObjectSec,
                startPointerSec,
                view: { scrollLeftPx, pxPerSec, clientX: 700 },
            });
            const pointerSec = pointerSecAt({
                scrollLeftPx,
                pxPerSec,
                clientX: 700,
                rectLeft: RECT_LEFT,
            });
            expect(objectSec - pointerSec).toBeCloseTo(grabOffset, 9);
        }
    });

    it("非有限输入不产生 NaN 落点", () => {
        expect(dragDeltaSec(Number.NaN, 1)).toBe(0);
        expect(dragDeltaSec(1, Number.NaN)).toBe(0);
        expect(pointerSecAt({ scrollLeftPx: 0, pxPerSec: 0, clientX: 100, rectLeft: 0 })).toBe(100);
        expect(deltaSecToContentPx(2, Number.NaN)).toBe(2);
    });
});
