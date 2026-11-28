// 长按重复的**派发上下文**契约：合成派发永不布防。
//
// 【这条门防的是什么】`beginHoldRepeat` 只由 `keyup` / `blur` 终止，而移动端
// 抽屉、菜单、独立窗口转发走的是**同一条** `handleKeybindingAction`。触摸屏上
// 没有 `keyup`，于是"点一下撤销"在 400ms 后变成"每 50ms 撤一次，永不停止"——
// 用户报的「撤销/重做几乎不可用」，一半来自这里：一次点击撤掉了整个历史。
//
// 行为断言只能覆盖到"当前存在的那几个调用点"，而**新加一条合成入口**时它会
// 静默地退回旧行为。因此这里再加一条源码断言：App.tsx 里每一个
// `beginHoldRepeat(` 调用都必须被 `allowHoldRepeat` 挡住。
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
    DEFAULT_DISPATCH,
    KEYBOARD_DISPATCH,
    MOBILE_DISPATCH,
    SYNTHETIC_DISPATCH,
    shouldArmHoldRepeat,
} from "./dispatchContext";

describe("shouldArmHoldRepeat", () => {
    it("★ 只有真键盘布防长按重复", () => {
        expect(shouldArmHoldRepeat(KEYBOARD_DISPATCH)).toBe(true);
        expect(shouldArmHoldRepeat(MOBILE_DISPATCH)).toBe(false);
        expect(shouldArmHoldRepeat(SYNTHETIC_DISPATCH)).toBe(false);
    });

    it("缺省上下文 = 键盘（老调用点行为逐字节不变）", () => {
        expect(shouldArmHoldRepeat(undefined)).toBe(true);
        expect(DEFAULT_DISPATCH.repeat).toBe(true);
    });
});

describe("App.tsx 的源码契约", () => {
    const source = readFileSync("src/App.tsx", "utf8");

    it("★ 每一处 beginHoldRepeat 都被 allowHoldRepeat 挡住", () => {
        const lines = source.split("\n");
        const unguarded: number[] = [];
        for (let i = 0; i < lines.length; i += 1) {
            if (!lines[i].includes("beginHoldRepeat(")) continue;
            if (lines[i].includes("import ")) continue;
            // 允许两种形状：同一行带守卫，或紧邻的上一行/上一行括号里带守卫。
            const sameLine = lines[i].includes("allowHoldRepeat");
            const prev = (lines[i - 1] ?? "") + (lines[i - 2] ?? "");
            if (!sameLine && !prev.includes("allowHoldRepeat")) unguarded.push(i + 1);
        }
        expect(unguarded, `未受守卫的 beginHoldRepeat：${unguarded.join(", ")}`).toEqual([]);
    });

    it("移动端抽屉走 MOBILE_DISPATCH（repeat: false）", () => {
        expect(source).toContain("MOBILE_DISPATCH");
        expect(source).toContain("const allowHoldRepeat = dispatchContext.repeat;");
    });
});
