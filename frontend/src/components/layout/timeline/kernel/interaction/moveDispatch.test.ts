/**
 * 逐帧手势分派（./moveDispatch）行为自检。
 *
 * 【主要内容】
 * 1. **回归**：`snap-offset-drag`（吸附偏移 ◣ 手柄）必须分派到 `"snap-offset"`
 *    ——缺这一条就是「对齐标记无法正确的被移动」的根因；
 * 2. **穷尽性**：表驱动覆盖 `TimelineGestureKind` 全集，任何新增种类若被漏登记
 *    （返回了不该返回的 `null`）立即失败；
 * 3. 已工作的种类仍映射到各自预览器（防止修复时误伤兄弟分支）；
 * 4. `pending-select` 刻意返回 `null`（升级路径自行派发，避免重复派发）。
 *
 * 【作用】这是本次缺陷的**根因回归锁**。分派链漏分支不会编译失败、也不会被
 * 「预览器数学」类测试发现（数学是对的，只是从没被调用）——只有分派层测试能挡住。
 *
 * 【与其他模块的关系】覆盖 `moveDispatch.ts`；不依赖 DOM 与 React。
 */

import { describe, expect, it } from "vitest";

import { resolveMoveDispatch, type MoveDispatch, type TimelineGestureKind } from "./moveDispatch";

/**
 * 手势种类 → 期望分派目标。
 *
 * 【为什么用显式期望表而不是"只要不为 null"】"不为 null"会让把
 * `snap-offset-drag` 误接到 `"drag"` 这类错误通过——那会导致拖手柄时移动整个 clip。
 * 逐项写死期望值，才能同时守住"有没有分派"和"分派给谁"。
 */
const EXPECTED: Record<TimelineGestureKind, MoveDispatch> = {
    none: null,
    // 未升级：由 host 的阈值升级分支自行派发一次，避免与常规分派重复。
    "pending-select": null,
    seek: "seek",
    "box-select": "box-select",
    "clip-drag": "drag",
    "clip-trim": "trim",
    "clip-fade": "fade",
    "gain-drag": "gain",
    "crossfade-grip": "crossfade-grip",
    "snap-offset-drag": "snap-offset",
};

describe("resolveMoveDispatch", () => {
    it("【回归】吸附偏移手柄拖动每帧都分派到 snap-offset", () => {
        // 缺陷形态：该种类在分派链里缺失 ⇒ 返回 null ⇒ 预览只在跨越 4px 阈值
        // 时发生一次，标记随即冻结（实测拖动 120px 只移动 7px）。
        expect(resolveMoveDispatch("snap-offset-drag")).toBe("snap-offset");
        expect(resolveMoveDispatch("snap-offset-drag")).not.toBeNull();
    });

    it("【穷尽性】枚举手势种类全集并逐项断言（新增种类漏登记即失败）", () => {
        // 这份字面量数组是"全集"的第二道锁：与 EXPECTED 的键集合必须一致。
        const allKinds: TimelineGestureKind[] = [
            "none",
            "pending-select",
            "seek",
            "box-select",
            "clip-drag",
            "clip-trim",
            "clip-fade",
            "gain-drag",
            "crossfade-grip",
            "snap-offset-drag",
        ];
        expect(new Set(allKinds)).toEqual(new Set(Object.keys(EXPECTED)));
        for (const kind of allKinds) {
            expect(resolveMoveDispatch(kind), `kind=${kind}`).toBe(EXPECTED[kind]);
        }
    });

    it("除 none 与 pending-select 外，每种手势都有逐帧分派（不会静默失效）", () => {
        for (const [kind, dispatch] of Object.entries(EXPECTED)) {
            if (kind === "none" || kind === "pending-select") continue;
            expect(dispatch, `kind=${kind} 缺少逐帧分派`).not.toBeNull();
        }
    });

    it("每种手势种类映射到互不相同的分派目标（无意外合并）", () => {
        const dispatched = Object.values(EXPECTED).filter(
            (value): value is Exclude<MoveDispatch, null> => value !== null,
        );
        expect(new Set(dispatched).size).toBe(dispatched.length);
    });

    it("pending-select 不由常规分派处理（升级路径负责，避免同帧重复派发）", () => {
        expect(resolveMoveDispatch("pending-select")).toBeNull();
    });

    it("空闲状态无分派", () => {
        expect(resolveMoveDispatch("none")).toBeNull();
    });

    it("未登记的种类（运行时脏值）返回 null 而不抛错", () => {
        // 宿主在渲染热路径上；抛错会中断整个手势链。
        expect(resolveMoveDispatch("takes-lane-drag" as TimelineGestureKind)).toBeNull();
    });
});
