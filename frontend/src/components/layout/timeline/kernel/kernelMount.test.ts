/**
 * 渲染内核 · 挂载决策单测。
 *
 * 【本测试要补的盲区】`TimelinePanel` 的挂载分支此前内联在 JSX 里，而本工程
 * Vitest 跑在 node 环境（无 jsdom），**没有任何测试引用 `.tsx`**。实测变异：
 * - 把该分支改成恒 `true`（逃生门失效）→ 823 passed / 2 failed（与基线相同，测不出）
 * - 改成恒 `false`（内核永不挂载 —— 正是构建模式回归的老问题）→ 同样测不出
 *
 * 抽成纯函数后，下面这些用例就能钉住那两条行为。
 */
import { describe, expect, it } from "vitest";

import { resolveKernelMount } from "./kernelMount";

describe("resolveKernelMount（内核挂载决策）", () => {
    it("★ 未显式设置时默认挂载内核（与开关默认值一致）", () => {
        expect(resolveKernelMount().useKernel).toBe(true);
        expect(resolveKernelMount({}).useKernel).toBe(true);
    });

    it("★ 开关开启且未失败 → 挂载内核（默认路径）", () => {
        const d = resolveKernelMount({ enabled: true, unavailable: false });
        expect(d.useKernel).toBe(true);
        expect(d.disabledByUser).toBe(false);
        expect(d.disabledByFailure).toBe(false);
    });

    it("★ 运行期不可用 → 不挂载，且记为「因失败」而不是「用户关闭」", () => {
        // 这是 Critical：WebGL2 创建失败必须退回既有实现，绝不能空白面板。
        const d = resolveKernelMount({ enabled: true, unavailable: true });
        expect(d.useKernel).toBe(false);
        expect(d.disabledByFailure).toBe(true);
        expect(d.disabledByUser).toBe(false);
    });

    it("★ 用户主动关闭 → 不挂载，且记为「用户关闭」（不该尝试建 GL 上下文）", () => {
        const d = resolveKernelMount({ enabled: false, unavailable: false });
        expect(d.useKernel).toBe(false);
        expect(d.disabledByUser).toBe(true);
        // 关键：用户关闭时不得报"失败"——否则会打出一条误导性的"内核不可用"日志。
        expect(d.disabledByFailure).toBe(false);
    });

    it("用户关闭后即便内核曾失败，也只按用户意愿解释", () => {
        const d = resolveKernelMount({ enabled: false, unavailable: true });
        expect(d.useKernel).toBe(false);
        // 用户关闭时不报"失败"——否则会打出一条误导性的"内核不可用"日志
        //（用户是自己关的，环境未必有问题）。
        expect(d.disabledByUser).toBe(true);
        expect(d.disabledByFailure).toBe(false);
    });

    it("两个「为何不挂」标记互不冒充（只有「用户想用却失败」才算失败）", () => {
        // 这是本模块最有价值的一条约束：把"没试"与"试了失败"分开，直接决定
        // 调用方是否该记一条需要诊断的日志。
        expect(resolveKernelMount({ enabled: false, unavailable: true }).disabledByFailure).toBe(
            false,
        );
        expect(resolveKernelMount({ enabled: true, unavailable: true }).disabledByFailure).toBe(
            true,
        );
    });

    it("缺省入参等价于「开关开启、未失败」", () => {
        const withDefaults = resolveKernelMount();
        const explicit = resolveKernelMount({ enabled: true, unavailable: false });
        expect(withDefaults).toEqual(explicit);
    });

    it("无入参调用不抛异常", () => {
        expect(() => resolveKernelMount()).not.toThrow();
        expect(() => resolveKernelMount({})).not.toThrow();
    });
});
