/**
 * 实例缓冲容量策略（./instanceBuffer）行为自检。
 *
 * 【主要内容】
 * 1. 容量足够时保持不变（不重新分配）；
 * 2. 容量不足时倍增，且至少容纳本次需求；
 * 3. 首次分配（current = 0）直接取需求值；
 * 4. 非法输入按 0 处理。
 *
 * 【作用】倍增策略是"滚动帧零分配"的前提：若退化为精确分配，窗口边界处
 * 每帧都会重新分配大缓冲，GC 压力回到热路径上。
 *
 * 【与其他模块的关系】覆盖 `instanceBuffer.ts`；被 `gl/sdfBoxProgram` 消费。
 */

import { describe, expect, it } from "vitest";

import { resolveBufferFloats } from "./instanceBuffer";

describe("resolveBufferFloats", () => {
    it("容量足够时保持不变", () => {
        expect(resolveBufferFloats(1000, 500)).toBe(1000);
        expect(resolveBufferFloats(1000, 1000)).toBe(1000);
    });

    it("容量不足时倍增（至少容纳本次需求）", () => {
        // 1000 → 2000（倍增后仍够）
        expect(resolveBufferFloats(1000, 1200)).toBe(2000);
        // 需求远超倍增结果时直接取需求值
        expect(resolveBufferFloats(1000, 5000)).toBe(5000);
    });

    it("首次分配（current = 0）直接取需求值", () => {
        expect(resolveBufferFloats(0, 250)).toBe(250);
    });

    it("非法输入按 0 处理", () => {
        expect(resolveBufferFloats(Number.NaN, 100)).toBe(100);
        expect(resolveBufferFloats(1000, Number.NaN)).toBe(1000);
        expect(resolveBufferFloats(1000, -5)).toBe(1000);
    });
});
