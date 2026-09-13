/**
 * 主画布内容签名（./mainCanvasSignature）行为自检。
 *
 * 【主要内容】
 * 1. 引用相同时判为相同（空闲帧走缓存快路径）；
 * 2. ★ 回归守卫：内容不同的对象 / 对象数组必须判为**不同**；
 * 3. 长度不同、原始值变化、首次比较 → 不同；
 * 4. `Object.is` 语义：NaN 与自身相同（避免某输入恒为 NaN 时每帧刷帧）。
 *
 * 【作用】签名是"要不要重绘主画布"的**唯一判据**，而它此前用 `[...].join("|")`
 * 构造——`join` 会把每个对象/数组元素串成字面量 `"[object Object]"`，于是两个
 * 完全不同的选区产生同一个签名，缓存命中、旧选区框留在画布上不消失（缺陷 #4）。
 * 本工程此前对签名层**零覆盖**（没有任何测试 import `drawPianoRoll`，也没有测试
 * 引用签名），因此这个缺陷能一路漏到真机。本文件是它的守卫——若比较实现退回
 * 字符串化，第 2、3、4 条会立刻失败。
 *
 * 【与其他模块的关系】覆盖 `mainCanvasSignature.ts`；被 `PianoRollPanel`（构造）
 * 与 `render.ts` 的 `drawPianoRoll`（消费）使用。不依赖 React / DOM / Redux。
 */
import { describe, expect, it } from "vitest";

import { isSameMainCanvasSignature } from "./mainCanvasSignature";

describe("isSameMainCanvasSignature", () => {
    it("引用相同 → 视为相同（空闲帧走缓存快路径）", () => {
        const selection = { aBeat: 1, bBeat: 2 };
        const a = [1920, 1.5, selection];
        const b = [1920, 1.5, selection];
        expect(isSameMainCanvasSignature(a, b)).toBe(true);
    });

    it("★ 回归守卫：内容不同的选区对象 → 必须判为不同", () => {
        // 这是 #4 的正向复现：若实现退回 join("|")，两项都会变成
        // "[object Object]" 而误判为相同，本断言随即失败。
        const a = [1, { aBeat: 4.1, bBeat: 6.7 }];
        const b = [1, { aBeat: 40, bBeat: 90 }];
        expect(isSameMainCanvasSignature(a, b)).toBe(false);
    });

    it("★ 回归守卫：选区从对象变为 null → 必须判为不同", () => {
        expect(isSameMainCanvasSignature([1, { aBeat: 1, bBeat: 2 }], [1, null])).toBe(false);
    });

    it("★ 回归守卫：对象数组内容不同的元素 → 必须判为不同", () => {
        const a = [[{ x: 1 }, { x: 2 }]];
        const b = [[{ x: 9 }, { x: 8 }]];
        expect(isSameMainCanvasSignature(a, b)).toBe(false);
    });

    it("长度不同 → 不同", () => {
        expect(isSameMainCanvasSignature([1, 2], [1, 2, 3])).toBe(false);
    });

    it("原始值变化 → 不同（滚动/缩放必须失效缓存）", () => {
        expect(isSameMainCanvasSignature([100, "pitch"], [101, "pitch"])).toBe(false);
    });

    it("数值 NaN 与自身：Object.is 语义下视为相同（避免 NaN 每帧刷帧）", () => {
        expect(isSameMainCanvasSignature([Number.NaN], [Number.NaN])).toBe(true);
    });

    it("首次比较（上次为 undefined）→ 不同", () => {
        expect(isSameMainCanvasSignature([1, 2], undefined)).toBe(false);
    });
});
