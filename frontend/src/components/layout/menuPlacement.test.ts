import { describe, expect, test } from "vitest";

import {
    MENU_BOUNDARY_MARGIN,
    MENU_MAX_CONTAINER_FRACTION,
    resolveMenuMaxHeight,
} from "./menuPlacement";

/*
 * 契约：菜单**永远向下展开**，且只在面板（容器）内部铺开。
 *
 * 上一版按视口算并允许向上翻转 —— 参数编辑器是停靠窗口，上方没有展示区，翻上去
 * 只会盖住自己的工具栏。这里把"向下、且在容器内"两条都钉住。
 */
describe("resolveMenuMaxHeight", () => {
    const container = { containerTop: 100, containerBottom: 900 };

    test("锚点下方空间充足时取面板高度的一半", () => {
        const maxHeight = resolveMenuMaxHeight({ anchorBottom: 130, ...container });
        // 面板高 800，一半是 400；下方可用 900-130-8 = 762。
        expect(maxHeight).toBe(800 * MENU_MAX_CONTAINER_FRACTION);
    });

    test("锚点靠近面板底部时按剩余空间收窄（不会伸出面板）", () => {
        const maxHeight = resolveMenuMaxHeight({ anchorBottom: 820, ...container });
        // 900 - 820 - 8 = 72，比一半上限小。
        expect(maxHeight).toBe(72);
        expect(820 + maxHeight).toBeLessThanOrEqual(900);
    });

    test("锚点已经贴着底边时钳到 0（不产生负高度）", () => {
        const maxHeight = resolveMenuMaxHeight({ anchorBottom: 1000, ...container });
        expect(maxHeight).toBe(0);
    });

    test("面板很矮时也不会超过面板高度的一半", () => {
        const maxHeight = resolveMenuMaxHeight({
            anchorBottom: 30,
            containerTop: 20,
            containerBottom: 120,
        });
        // 面板高 100 → 上限 50；下方可用 120-30-8 = 82 → 取 50。
        expect(maxHeight).toBe(50);
    });

    test("边距与上限可覆盖", () => {
        const maxHeight = resolveMenuMaxHeight({
            anchorBottom: 130,
            ...container,
            margin: 0,
            maxFraction: 1,
        });
        expect(maxHeight).toBe(770);
    });

    test("默认常量稳定", () => {
        expect(MENU_BOUNDARY_MARGIN).toBe(8);
        expect(MENU_MAX_CONTAINER_FRACTION).toBe(0.5);
    });
});
