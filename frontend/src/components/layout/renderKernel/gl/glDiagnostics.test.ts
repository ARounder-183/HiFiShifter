/**
 * WebGL2 可用性探测的单测。
 *
 * 【为什么要测】GL 失败界面是唯一路径下的**唯一**用户出口，其诊断信息必须准确且
 * 不得因环境差异抛错（它恰恰运行在"环境有问题"的机器上）。
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { collectGlDiagnostics } from "./glDiagnostics";

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("collectGlDiagnostics", () => {
    it("★ 完全不支持 WebGL2 时给出 false 且不抛异常", () => {
        vi.stubGlobal("document", {
            createElement: () => ({ getContext: () => null }),
        });
        const d = collectGlDiagnostics();
        expect(d.webgl2).toBe(false);
        expect(d.webgl1).toBe(false);
        expect(typeof d.userAgent).toBe("string");
    });

    it("支持 WebGL2 时 webgl2 为 true", () => {
        vi.stubGlobal("document", {
            createElement: () => ({
                getContext: (type: string) => (type === "webgl2" ? { fake: true } : null),
            }),
        });
        expect(collectGlDiagnostics().webgl2).toBe(true);
    });

    it("★ document 缺失（非浏览器环境）时不抛异常", () => {
        vi.stubGlobal("document", undefined);
        expect(() => collectGlDiagnostics()).not.toThrow();
        expect(collectGlDiagnostics().webgl2).toBe(false);
    });

    it("getContext 抛异常时按不可用处理（老驱动的常见表现）", () => {
        vi.stubGlobal("document", {
            createElement: () => ({
                getContext: () => {
                    throw new Error("blocked");
                },
            }),
        });
        expect(collectGlDiagnostics().webgl2).toBe(false);
    });

    it("★ 不尝试读取 renderer/vendor（失败环境拿不到，不假装能给出型号）", () => {
        // 【为什么单列一条】diagnostics 在**已经失败**的环境里运行，此时再建
        // webgl2 上下文同样返回 null，读不到 WEBGL_debug_renderer_info。与其给出
        // undefined 让人误以为拿到了，不如契约上就不提供该字段。
        vi.stubGlobal("document", {
            createElement: () => ({ getContext: () => null }),
        });
        const d = collectGlDiagnostics() as unknown as Record<string, unknown>;
        expect(d).not.toHaveProperty("renderer");
        expect(d).not.toHaveProperty("vendor");
    });
});
