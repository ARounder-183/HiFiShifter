import { describe, expect, it } from "vitest";

import {
    CHANNEL_TOLERANCE_PRESETS,
    DEFAULT_CHANNEL_IMPORT_POLICY,
    normalizeChannelImportPolicy,
    type ChannelImportPolicy,
} from "./settings";

/**
 * 导入声道策略设置的回归测试。
 *
 * 【为什么需要】容差下拉框曾经"显示不出内容"：`Select.Root` 的 `value` 用
 * `String(1e-3)` = `"0.001"`，而 `Select.Item` 的 `value` 写的是 `"1e-3"`。
 * Radix 找不到匹配项时 Trigger 渲染空白 —— 界面看起来是坏的，且没有任何断言
 * 会失败。下面第一条用例就是这个缺陷的守卫：下拉项与当前值必须用**同一套
 * 编码**，并且默认值必须落在档位里。
 */

describe("channel import policy", () => {
    it("keeps the default tolerance inside the preset list", () => {
        // 下拉项的取值必须由同一个函数产出；默认值必须能被选中，否则
        // Trigger 会显示空白（曾经的实际缺陷）。
        const values = CHANNEL_TOLERANCE_PRESETS.map(String);
        expect(values).toContain(String(DEFAULT_CHANNEL_IMPORT_POLICY.tolerance));
        expect(DEFAULT_CHANNEL_IMPORT_POLICY.tolerance).toBe(1e-3);
    });

    it("has strictly increasing, de-duplicated preset values", () => {
        // 顺序即滚轮方向（由严到松），重复值会让滚轮"卡住"。
        for (let i = 1; i < CHANNEL_TOLERANCE_PRESETS.length; i += 1) {
            expect(CHANNEL_TOLERANCE_PRESETS[i]).toBeGreaterThan(
                CHANNEL_TOLERANCE_PRESETS[i - 1],
            );
        }
        expect(new Set(CHANNEL_TOLERANCE_PRESETS.map(String)).size).toBe(
            CHANNEL_TOLERANCE_PRESETS.length,
        );
    });

    it("keeps every preset inside the backend clamp range", () => {
        // 后端 normalized() 把容差钳到 [0, 0.1]；越界的档位选完就会被改写，
        // 表现为"选了又跳回去"。
        for (const preset of CHANNEL_TOLERANCE_PRESETS) {
            expect(preset).toBeGreaterThanOrEqual(0);
            expect(preset).toBeLessThanOrEqual(0.1);
        }
    });

    it("preserves every preset through normalization", () => {
        for (const preset of CHANNEL_TOLERANCE_PRESETS) {
            const policy: ChannelImportPolicy = {
                ...DEFAULT_CHANNEL_IMPORT_POLICY,
                tolerance: preset,
            };
            expect(normalizeChannelImportPolicy(policy).tolerance).toBe(preset);
        }
    });

    it("clamps out-of-range values and falls back on bad enums", () => {
        const normalized = normalizeChannelImportPolicy({
            mode: "bogus" as ChannelImportPolicy["mode"],
            windowSec: 999,
            windowCount: 99_999,
            tolerance: 5,
            monoTargetMode: 7,
        });
        expect(normalized.mode).toBe("smart");
        expect(normalized.windowSec).toBeLessThanOrEqual(5);
        expect(normalized.windowCount).toBe(256);
        expect(normalized.tolerance).toBeLessThanOrEqual(0.1);
        expect(normalized.monoTargetMode).toBe(2);
    });

    it("falls back to defaults on non-finite numbers", () => {
        const normalized = normalizeChannelImportPolicy({
            ...DEFAULT_CHANNEL_IMPORT_POLICY,
            windowSec: Number.NaN,
            tolerance: Number.NaN,
            windowCount: Number.NaN,
        });
        expect(normalized.windowSec).toBe(DEFAULT_CHANNEL_IMPORT_POLICY.windowSec);
        expect(normalized.tolerance).toBe(DEFAULT_CHANNEL_IMPORT_POLICY.tolerance);
        expect(normalized.windowCount).toBe(DEFAULT_CHANNEL_IMPORT_POLICY.windowCount);
    });

    it("keeps valid target modes", () => {
        for (const mode of [2, 3, 4]) {
            const normalized = normalizeChannelImportPolicy({
                ...DEFAULT_CHANNEL_IMPORT_POLICY,
                monoTargetMode: mode,
            });
            expect(normalized.monoTargetMode).toBe(mode);
        }
    });
});
