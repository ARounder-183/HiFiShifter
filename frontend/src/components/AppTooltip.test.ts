import { describe, expect, test } from "vitest";

import { clampTooltipPosition } from "./appTooltipPosition";

const VW = 1400;
const VH = 900;

describe("clampTooltipPosition（按气泡实测尺寸夹紧）", () => {
    test("远离边缘时气泡贴着光标右下方", () => {
        const p = clampTooltipPosition({ x: 400, y: 300 }, 41, 25, VW, VH);
        expect(p.x).toBe(414);
        expect(p.y).toBe(318);
    });

    test("右侧短文案气泡不再被固定余量甩离光标（回归：320px 固定余量）", () => {
        // 光标在 x=1232，"算法"气泡宽 41：旧实现钳到 innerWidth-320=1080，
        // 气泡落后光标 152px；新实现只收回必要距离。
        const p = clampTooltipPosition({ x: 1232, y: 418 }, 41, 25, VW, VH);
        expect(p.x).toBe(1246);
    });

    test("贴右缘时气泡右端不出窗（收回自身宽度而非固定余量）", () => {
        // 320px 宽长链接气泡：光标 1350 时 x+14 已越界，钳到窗宽-宽-间距。
        const p = clampTooltipPosition({ x: 1350, y: 418 }, 320, 60, VW, VH);
        expect(p.x).toBe(VW - 320 - 8);
        expect(p.x + 320).toBeLessThanOrEqual(VW - 8);
    });

    test("超宽气泡（窗宽不足）钳到左缘最小间距", () => {
        const p = clampTooltipPosition({ x: 500, y: 300 }, 5000, 60, 1400, 900);
        expect(p.x).toBe(8);
    });

    test("底部同样按实测高度夹紧", () => {
        const p = clampTooltipPosition({ x: 100, y: VH - 10 }, 41, 25, VW, VH);
        expect(p.y).toBe(VH - 25 - 8);
    });

    test("窗口极小时不低于最小间距", () => {
        const p = clampTooltipPosition({ x: 5, y: 5 }, 300, 60, 100, 50);
        expect(p.x).toBe(8);
        expect(p.y).toBe(8);
    });

    test("负锚点（指针越出视口）双向钳制不低于最小间距", () => {
        // clampAxisPosition 此前只夹上界：anchor+offset < edgeGap 时
        //（合成指针 / 拖拽中光标可为负），气泡会整段画出窗外左缘。
        const p = clampTooltipPosition({ x: -40, y: -30 }, 41, 25, VW, VH);
        expect(p.x).toBe(8);
        expect(p.y).toBe(8);
        expect(p.x).toBeGreaterThanOrEqual(8);
    });
});
