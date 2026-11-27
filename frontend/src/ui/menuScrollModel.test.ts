/*
 * 菜单滚动与定位的纯函数模型。
 *
 * 【为什么这些用例值这个价钱】它们是三处消费者（提示行可见性、键盘跟随、rAF 引擎）
 * 共用的唯一判据。任何一条算错，症状都长得像"偶尔"：提示行在边界上闪、
 * 键盘走到末项停在屏幕外、翻页在矮窗口里几乎不动。逐值钉住比在浏览器里截图可靠。
 */
import { describe, expect, test } from "vitest";

import { clampAxisPosition } from "../components/appTooltipPosition";
import {
    MENU_SCROLL_EPSILON_PX,
    clampMenuScrollTop,
    computeMenuPageItemCount,
    computeMenuPageStep,
    computeMenuScrollState,
    easeOutCubic,
    menuScrollDeltaToReveal,
    menuScrollPointerKind,
    placeMenuWithinViewport,
} from "./menuScrollModel";

describe("computeMenuScrollState", () => {
    test("装得下时两个方向都不提示", () => {
        expect(
            computeMenuScrollState({ scrollTop: 0, clientHeight: 500, scrollHeight: 500 }),
        ).toEqual({ canUp: false, canDown: false });
    });

    test("溢出小于容差也算装得下（隐藏 input 撑出的 1px 不该长出提示行）", () => {
        expect(
            computeMenuScrollState({ scrollTop: 0, clientHeight: 500, scrollHeight: 500.8 }),
        ).toEqual({ canUp: false, canDown: false });
    });

    test("在顶端时只有向下", () => {
        expect(
            computeMenuScrollState({ scrollTop: 0, clientHeight: 400, scrollHeight: 600 }),
        ).toEqual({ canUp: false, canDown: true });
    });

    test("在底端时只有向上", () => {
        expect(
            computeMenuScrollState({ scrollTop: 200, clientHeight: 400, scrollHeight: 600 }),
        ).toEqual({ canUp: true, canDown: false });
    });

    test("中间时两个方向都有", () => {
        expect(
            computeMenuScrollState({ scrollTop: 100, clientHeight: 400, scrollHeight: 600 }),
        ).toEqual({ canUp: true, canDown: true });
    });

    test("刚好差一个容差就是边界：容差本身不是可滚距离", () => {
        const state = computeMenuScrollState({
            scrollTop: 0,
            clientHeight: 100,
            scrollHeight: 100 + MENU_SCROLL_EPSILON_PX,
        });
        expect(state).toEqual({ canUp: false, canDown: false });
    });
});

describe("clampMenuScrollTop", () => {
    test("夹进 [0, maxScroll]", () => {
        expect(clampMenuScrollTop(-30, 200)).toBe(0);
        expect(clampMenuScrollTop(500, 200)).toBe(200);
        expect(clampMenuScrollTop(120, 200)).toBe(120);
    });

    test("不可滚时恒为 0", () => {
        expect(clampMenuScrollTop(50, 0)).toBe(0);
        expect(clampMenuScrollTop(50, -10)).toBe(0);
    });

    test("非有限值归零（NaN 会让 scrollTop 静默失效）", () => {
        expect(clampMenuScrollTop(Number.NaN, 200)).toBe(0);
    });
});

describe("menuScrollDeltaToReveal", () => {
    const base = { viewTop: 0, viewBottom: 400, insetTop: 18, insetBottom: 18 };

    test("项已经在窗口内 ⇒ 0", () => {
        expect(menuScrollDeltaToReveal({ ...base, itemTop: 100, itemBottom: 124 })).toBe(0);
    });

    test("项被上方提示行盖住 ⇒ 负值（要往上滚）", () => {
        expect(menuScrollDeltaToReveal({ ...base, itemTop: 4, itemBottom: 28 })).toBe(-14);
    });

    test("项被下方提示行盖住 ⇒ 正值（要往下滚）", () => {
        expect(menuScrollDeltaToReveal({ ...base, itemTop: 380, itemBottom: 404 })).toBe(22);
    });

    test("提示行不可见时按完整窗口算（inset 为 0）", () => {
        expect(
            menuScrollDeltaToReveal({
                ...base,
                insetTop: 0,
                insetBottom: 0,
                itemTop: 4,
                itemBottom: 28,
            }),
        ).toBe(0);
    });
});

describe("computeMenuPageStep / computeMenuPageItemCount", () => {
    test("矮窗口：翻页不小于三项高", () => {
        // clientHeight 100、项高 24 → 100 - 48 = 52 < 72，取 72。
        expect(computeMenuPageStep({ clientHeight: 100, itemHeight: 24 })).toBe(72);
    });

    test("高窗口：一页减两行", () => {
        expect(computeMenuPageStep({ clientHeight: 600, itemHeight: 24 })).toBe(552);
    });

    test("项高为 0（jsdom 读不到令牌）时仍然给出正数", () => {
        expect(computeMenuPageStep({ clientHeight: 0, itemHeight: 0 })).toBeGreaterThan(0);
    });

    test("PageUp/PageDown 的项数至少 1", () => {
        expect(computeMenuPageItemCount({ clientHeight: 0, itemHeight: 24 })).toBe(1);
        expect(computeMenuPageItemCount({ clientHeight: 240, itemHeight: 24 })).toBe(10);
    });
});

describe("placeMenuWithinViewport", () => {
    const viewport = { viewportWidth: 1000, viewportHeight: 800, edgeGap: 8 };

    test("装得下 ⇒ 贴锚点，不翻转", () => {
        expect(
            placeMenuWithinViewport({
                ...viewport,
                anchorX: 100,
                anchorY: 100,
                width: 200,
                height: 300,
            }),
        ).toEqual({ left: 100, top: 100, flippedX: false, flippedY: false });
    });

    test("右侧装不下、左侧装得下 ⇒ 翻到左侧", () => {
        const placement = placeMenuWithinViewport({
            ...viewport,
            anchorX: 900,
            anchorY: 100,
            width: 300,
            height: 200,
        });
        expect(placement.flippedX).toBe(true);
        expect(placement.left).toBe(600);
        expect(placement.top).toBe(100);
    });

    test("下方装不下、上方装得下 ⇒ 翻到上方（不再盖住右键点）", () => {
        const placement = placeMenuWithinViewport({
            ...viewport,
            anchorX: 100,
            anchorY: 700,
            width: 200,
            height: 300,
        });
        expect(placement.flippedY).toBe(true);
        expect(placement.top).toBe(400);
    });

    test("两侧都装不下 ⇒ 与 clampAxisPosition 逐值相等（退化路径没有新行为）", () => {
        const input = {
            ...viewport,
            anchorX: 500,
            anchorY: 400,
            width: 1400,
            height: 1200,
        };
        const placement = placeMenuWithinViewport(input);
        expect(placement.flippedX).toBe(false);
        expect(placement.flippedY).toBe(false);
        expect(placement.left).toBe(
            clampAxisPosition(input.anchorX, input.width, input.viewportWidth, 0, input.edgeGap),
        );
        expect(placement.top).toBe(
            clampAxisPosition(input.anchorY, input.height, input.viewportHeight, 0, input.edgeGap),
        );
    });
});

describe("缓动与指针分档", () => {
    test("easeOutCubic 从 0 到 1 且单调", () => {
        expect(easeOutCubic(0)).toBe(0);
        expect(easeOutCubic(1)).toBe(1);
        expect(easeOutCubic(0.5)).toBeGreaterThan(0.5);
        // 越界输入被夹住：rAF 的时间戳可能略微超过 duration。
        expect(easeOutCubic(-1)).toBe(0);
        expect(easeOutCubic(2)).toBe(1);
    });

    test("触摸走 touch 档，鼠标与笔走 pointer 档", () => {
        expect(menuScrollPointerKind("touch")).toBe("touch");
        expect(menuScrollPointerKind("mouse")).toBe("pointer");
        expect(menuScrollPointerKind("pen")).toBe("pointer");
        expect(menuScrollPointerKind(undefined)).toBe("pointer");
    });
});
