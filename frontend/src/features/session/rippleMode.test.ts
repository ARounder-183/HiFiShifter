/**
 * 波纹编辑的模式词汇表：开关 × 方向 → 生效模式，以及旧版三值编码的迁移。
 *
 * 【为什么值得单独测】这一层是"UI 的两根轴"与"引擎的三值"之间唯一的换算点：
 * 拖拽预览（前端）与提交（后端）都靠它对齐。换算错一格，用户看到的就是
 * "预览推了三条轨道、松手只推了一条"这种两边都自洽、合起来错位的结果。
 */
import { expect, test } from "vitest";

import {
    DEFAULT_RIPPLE_SCOPE,
    hydrateRippleState,
    normalizeRippleScope,
    resolveRippleMode,
} from "./rippleMode";

test("normalizeRippleScope：非法值回落到按轨道", () => {
    expect(normalizeRippleScope("track")).toBe("track");
    expect(normalizeRippleScope("all")).toBe("all");
    for (const value of [undefined, null, "", "off", "ALL", 0, 1, {}, []]) {
        expect(normalizeRippleScope(value)).toBe(DEFAULT_RIPPLE_SCOPE);
    }
});

test("resolveRippleMode：关闭时无论方向是什么都折算成 off", () => {
    expect(resolveRippleMode(true, "track")).toBe("track");
    expect(resolveRippleMode(true, "all")).toBe("all");
    expect(resolveRippleMode(false, "track")).toBe("off");
    expect(resolveRippleMode(false, "all")).toBe("off");
});

test("hydrateRippleState：旧版三值编码按「开着 / 关着」还原", () => {
    // 旧编码里 track / all 就代表"开着"，off 代表"关着"。
    expect(hydrateRippleState({ rippleMode: "off" })).toEqual({ enabled: false, scope: "track" });
    expect(hydrateRippleState({ rippleMode: "track" })).toEqual({ enabled: true, scope: "track" });
    expect(hydrateRippleState({ rippleMode: "all" })).toEqual({ enabled: true, scope: "all" });
    // 手改坏的值 = 关着。
    expect(hydrateRippleState({ rippleMode: "sideways" })).toEqual({
        enabled: false,
        scope: "track",
    });
});

test("hydrateRippleState：新配置以开关为准，且关闭时仍记得方向", () => {
    expect(hydrateRippleState({ rippleEnabled: false, rippleMode: "all" })).toEqual({
        enabled: false,
        scope: "all",
    });
    expect(hydrateRippleState({ rippleEnabled: true, rippleMode: "track" })).toEqual({
        enabled: true,
        scope: "track",
    });
    // 开关只认真正的 true：字符串 "true" / 1 都不算（设置来自磁盘，不能靠真值性）。
    expect(hydrateRippleState({ rippleEnabled: "true", rippleMode: "all" }).enabled).toBe(false);
});

test("hydrateRippleState：什么都没有 → 出厂状态", () => {
    expect(hydrateRippleState({})).toEqual({ enabled: false, scope: DEFAULT_RIPPLE_SCOPE });
});
