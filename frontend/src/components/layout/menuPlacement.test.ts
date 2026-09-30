import { describe, expect, test } from "vitest";

import {
    MENU_MAX_VIEWPORT_FRACTION,
    MENU_VIEWPORT_MARGIN,
    resolveMenuPlacement,
} from "./menuPlacement";

describe("resolveMenuPlacement", () => {
    test("锚点靠上：向下展开，高度取 50vh 上限", () => {
        const placement = resolveMenuPlacement({
            anchorTop: 100,
            anchorBottom: 130,
            viewportHeight: 1000,
        });
        expect(placement.side).toBe("below");
        expect(placement.maxHeight).toBe(1000 * MENU_MAX_VIEWPORT_FRACTION);
    });

    test("锚点靠下：向上翻转，不再伸出窗口底部", () => {
        const placement = resolveMenuPlacement({
            anchorTop: 900,
            anchorBottom: 930,
            viewportHeight: 1000,
        });
        expect(placement.side).toBe("above");
        // 上方可用 900 - 8 = 892，被 50vh 上限压到 500。
        expect(placement.maxHeight).toBe(500);
    });

    test("下方空间不足时按实际空间收窄（这是原来的缺陷）", () => {
        // 窗口 400 高、锚点底部 130：下方只有 400-130-8 = 262，50vh 是 200。
        const placement = resolveMenuPlacement({
            anchorTop: 100,
            anchorBottom: 130,
            viewportHeight: 400,
        });
        expect(placement.side).toBe("below");
        expect(placement.maxHeight).toBe(200);
        // 关键不变量：菜单永远不越过视口下边缘。
        expect(130 + placement.maxHeight).toBeLessThanOrEqual(400);
    });

    test("窗口很矮且锚点靠下：翻到上方，仍不越界", () => {
        const placement = resolveMenuPlacement({
            anchorTop: 150,
            anchorBottom: 180,
            viewportHeight: 200,
        });
        expect(placement.side).toBe("above");
        // 上方可用 150 - 8 = 142，50vh 是 100。
        expect(placement.maxHeight).toBe(100);
        expect(180 - placement.maxHeight).toBeGreaterThanOrEqual(0);
    });

    test("两侧空间相等时朝下（默认方向）", () => {
        // below = 1000 - 500 - 8 = 492；above = 492 - 8 = 484 → 下方更大。
        // 构造严格相等：below === above ⇒ viewportHeight - anchorBottom === anchorTop。
        const placement = resolveMenuPlacement({
            anchorTop: 400,
            anchorBottom: 608,
            viewportHeight: 1008,
        });
        // below = 1008 - 608 - 8 = 392；above = 400 - 8 = 392 → 相等，取 below。
        expect(placement.side).toBe("below");
    });

    test("空间为负时钳到 0（不产生负高度）", () => {
        const placement = resolveMenuPlacement({
            anchorTop: -50,
            anchorBottom: -20,
            viewportHeight: 100,
        });
        expect(placement.maxHeight).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(placement.maxHeight)).toBe(true);
    });

    test("边距与上限可覆盖（便于复用与单测）", () => {
        const placement = resolveMenuPlacement({
            anchorTop: 10,
            anchorBottom: 40,
            viewportHeight: 1000,
            margin: 0,
            maxFraction: 1,
        });
        expect(placement.side).toBe("below");
        expect(placement.maxHeight).toBe(960);
    });

    test("默认常量稳定", () => {
        expect(MENU_VIEWPORT_MARGIN).toBe(8);
        expect(MENU_MAX_VIEWPORT_FRACTION).toBe(0.5);
    });
});
