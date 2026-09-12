/**
 * 渲染内核 · 挂载决策单测。
 *
 * 【本测试要补的盲区】`TimelinePanel` 的挂载分支此前内联在 JSX 里，而本工程
 * Vitest 跑在 node 环境（无 jsdom），**没有任何测试引用 `.tsx`**。实测变异：
 * - 把该分支改成恒 `true` → 与基线相同的通过数（测不出）
 * - 改成恒 `false`（内核永不挂载 —— 正是构建模式回归的老问题）→ 同样测不出
 *
 * 抽成纯函数后，下面这些用例就能钉住那两条行为。
 *
 * 【当前状态】内核收归唯一路径后，面板的判据已简化成只问"运行期是否可用"
 * （`kernelAvailability.isKernelAvailable`），本模块暂无生产消费者。用例保留的意义
 * 是钉住"没试"与"试了失败"必须可区分这一行为契约（见 `kernelMount.ts` 文件头）。
 */
import { describe, expect, it } from "vitest";

import { resolveKernelMount } from "./kernelMount";

describe("resolveKernelMount（内核挂载决策）", () => {
    it("★ 未显式设置时默认挂载内核（漏传不应静默不挂）", () => {
        expect(resolveKernelMount().useKernel).toBe(true);
        expect(resolveKernelMount({}).useKernel).toBe(true);
    });

    it("★ 要求使用且未失败 → 挂载内核（默认路径）", () => {
        const d = resolveKernelMount({ enabled: true, unavailable: false });
        expect(d.useKernel).toBe(true);
        expect(d.disabledByUser).toBe(false);
        expect(d.disabledByFailure).toBe(false);
    });

    it("★ 运行期不可用 → 不挂载，且记为「因失败」而不是「调用方关闭」", () => {
        // 这是 Critical：WebGL2 创建失败必须走到失败界面，绝不能空白面板。
        const d = resolveKernelMount({ enabled: true, unavailable: true });
        expect(d.useKernel).toBe(false);
        expect(d.disabledByFailure).toBe(true);
        expect(d.disabledByUser).toBe(false);
    });

    it("★ 调用方主动关闭 → 不挂载，且记为「调用方关闭」（不该尝试建 GL 上下文）", () => {
        const d = resolveKernelMount({ enabled: false, unavailable: false });
        expect(d.useKernel).toBe(false);
        expect(d.disabledByUser).toBe(true);
        // 关键：主动关闭时不得报"失败"——否则会打出一条误导性的"内核不可用"日志。
        expect(d.disabledByFailure).toBe(false);
    });

    it("关闭后即便内核曾失败，也只按主动意愿解释", () => {
        const d = resolveKernelMount({ enabled: false, unavailable: true });
        expect(d.useKernel).toBe(false);
        // 主动关闭时不报"失败"——否则会打出一条误导性的"内核不可用"日志
        //（是调用方自己不要的，环境未必有问题）。
        expect(d.disabledByUser).toBe(true);
        expect(d.disabledByFailure).toBe(false);
    });

    it("两个「为何不挂」标记互不冒充（只有「想用却失败」才算失败）", () => {
        // 这是本模块最有价值的一条约束：把"没试"与"试了失败"分开，直接决定
        // 调用方是否该记一条需要诊断的日志。
        expect(resolveKernelMount({ enabled: false, unavailable: true }).disabledByFailure).toBe(
            false,
        );
        expect(resolveKernelMount({ enabled: true, unavailable: true }).disabledByFailure).toBe(
            true,
        );
    });

    it("缺省入参等价于「要求使用、未失败」", () => {
        const withDefaults = resolveKernelMount();
        const explicit = resolveKernelMount({ enabled: true, unavailable: false });
        expect(withDefaults).toEqual(explicit);
    });

    it("无入参调用不抛异常", () => {
        expect(() => resolveKernelMount()).not.toThrow();
        expect(() => resolveKernelMount({})).not.toThrow();
    });
});
