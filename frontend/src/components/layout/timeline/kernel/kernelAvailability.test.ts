/**
 * 内核可用性判定的单测。
 *
 * 【守的是什么】内核是唯一渲染路径，失败时无回退。"失败必须显式报错、绝不静默
 * 空白"是用户可感知的行为，必须有用例钉住——这也是 Phase 3 R8 教训（默认值曾
 * 跟随构建模式、打包后静默退回旧渲染器）在行为层面的延续。
 */
import { describe, expect, it } from "vitest";

import { isKernelAvailable } from "./kernelAvailability";

describe("isKernelAvailable", () => {
    it("★ 未失败时可用（渲染内核）", () => {
        expect(isKernelAvailable(false)).toBe(true);
    });

    it("★ 已失败时不可用（必须渲染失败界面，不得静默空白）", () => {
        expect(isKernelAvailable(true)).toBe(false);
    });

    it("返回值恒为布尔量（不得把 undefined 当可用传下去）", () => {
        expect(typeof isKernelAvailable(false)).toBe("boolean");
        expect(typeof isKernelAvailable(true)).toBe("boolean");
    });
});
