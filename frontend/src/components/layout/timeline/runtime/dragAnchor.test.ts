/**
 * 拖拽锚点换算自检（"拖拽中滚动/缩放导致偏移"的回归）。
 *
 * 【要钉死的契约】
 * - `dragDeltaSec`：落点与光标之间的**时间偏移**恒定 —— 中途滚动或缩放都不改变它。
 * - `anchoredDeltaSec`：落点与光标之间的**像素偏移**恒定（抓取偏移按屏幕像素记），
 *   这才是"抓住对象的某处拖动"的预期 —— 缩放到 2 倍时，40px 的抓取偏移不应变成 80px。
 */
import { describe, expect, it } from "vitest";

import { anchoredDeltaSec, deltaSecToContentPx, dragDeltaSec, pointerSecAt } from "./dragAnchor";

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

/** 同上，但走"抓取偏移按像素恒定"的算法。 */
function anchoredObjectSecAt(args: {
    originObjectSec: number;
    startPointerSec: number;
    startPxPerSec: number;
    view: { scrollLeftPx: number; pxPerSec: number; clientX: number };
}): number {
    const pointerSec = pointerSecAt({
        scrollLeftPx: args.view.scrollLeftPx,
        pxPerSec: args.view.pxPerSec,
        clientX: args.view.clientX,
        rectLeft: RECT_LEFT,
    });
    return (
        args.originObjectSec +
        anchoredDeltaSec({
            anchorSec: args.originObjectSec,
            startPointerSec: args.startPointerSec,
            startPxPerSec: args.startPxPerSec,
            pointerSec,
            pxPerSec: args.view.pxPerSec,
        })
    );
}

/** 对象与光标之间的**屏幕像素**偏移。 */
function screenOffsetPx(args: {
    objectSec: number;
    view: { scrollLeftPx: number; pxPerSec: number; clientX: number };
}): number {
    return (
        args.view.clientX -
        RECT_LEFT -
        (args.objectSec * args.view.pxPerSec - args.view.scrollLeftPx)
    );
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

describe("抓取偏移按屏幕像素恒定（anchoredDeltaSec）", () => {
    it("缩放在拖拽中不变时，结果与 dragDeltaSec 逐位相同", () => {
        const startPointerSec = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: 100,
            clientX: 500,
            rectLeft: RECT_LEFT,
        });
        for (const clientX of [400, 500, 612.5, 900]) {
            const pointerSec = pointerSecAt({
                scrollLeftPx: 0,
                pxPerSec: 100,
                clientX,
                rectLeft: RECT_LEFT,
            });
            expect(
                anchoredDeltaSec({
                    anchorSec: 4,
                    startPointerSec,
                    startPxPerSec: 100,
                    pointerSec,
                    pxPerSec: 100,
                }),
            ).toBe(dragDeltaSec(startPointerSec, pointerSec));
        }
    });

    it("★ 中途缩放：对象与光标的**像素偏移**保持不变（旧算法会按缩放比放大）", () => {
        // 按下：100px/s、未滚动、指针屏幕 x=500 → 指针时间 2.44s；对象在 4s（抓取点偏右）
        const startPxPerSec = 100;
        const startPointerSec = pointerSecAt({
            scrollLeftPx: 0,
            pxPerSec: startPxPerSec,
            clientX: 500,
            rectLeft: RECT_LEFT,
        });
        const originObjectSec = 4;
        // 抓取偏移 = 指针 − 锚点（与 `anchoredDeltaSec` 同一约定）：
        // 对象在光标**右侧** (2.44 − 4) × 100 = −156px（即光标在对象左侧 156px 处）。
        const grabOffsetPx = (startPointerSec - originObjectSec) * startPxPerSec;
        expect(grabOffsetPx).toBeCloseTo(-156, 9);

        for (const pxPerSec of [100, 200, 40, 1600]) {
            const view = { scrollLeftPx: 0, pxPerSec, clientX: 500 };
            const anchored = anchoredObjectSecAt({
                originObjectSec,
                startPointerSec,
                startPxPerSec,
                view,
            });
            // 像素偏移恒定 —— 抓在哪儿就一直在哪儿。
            expect(screenOffsetPx({ objectSec: anchored, view })).toBeCloseTo(grabOffsetPx, 6);
            // 旧（纯时间）算法：像素偏移按缩放比伸缩 → 放大 2 倍时 156px 变 312px。
            const legacy = objectSecAt({ originObjectSec, startPointerSec, view });
            expect(screenOffsetPx({ objectSec: legacy, view })).toBeCloseTo(
                grabOffsetPx * (pxPerSec / startPxPerSec),
                6,
            );
        }
    });

    it("★ 中途滚动：像素偏移同样不变（指针时间按实时视口重算）", () => {
        const startPxPerSec = 80;
        const startPointerSec = pointerSecAt({
            scrollLeftPx: 120,
            pxPerSec: startPxPerSec,
            clientX: 700,
            rectLeft: RECT_LEFT,
        });
        const originObjectSec = 1.5;
        const grabOffsetPx = (startPointerSec - originObjectSec) * startPxPerSec;
        for (const [scrollLeftPx, pxPerSec] of [
            [120, 80],
            [420, 80],
            [120, 160],
            [-40, 240],
            [2048.5, 33.3],
        ]) {
            const view = { scrollLeftPx, pxPerSec, clientX: 700 };
            const anchored = anchoredObjectSecAt({
                originObjectSec,
                startPointerSec,
                startPxPerSec,
                view,
            });
            expect(screenOffsetPx({ objectSec: anchored, view })).toBeCloseTo(grabOffsetPx, 6);
        }
    });

    it("抓取偏移为 0（抓住对象本身/光标即锚点）时退化为时间算法", () => {
        const startPointerSec = 2;
        expect(
            anchoredDeltaSec({
                anchorSec: startPointerSec,
                startPointerSec,
                startPxPerSec: 100,
                pointerSec: 5,
                pxPerSec: 250,
            }),
        ).toBeCloseTo(3, 9);
    });
});
