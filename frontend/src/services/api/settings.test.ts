import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("../invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

import {
    DEFAULT_CHANNEL_IMPORT_POLICY,
    TOLERANCE_PERCENT_MAX,
    normalizeChannelImportPolicy,
    percentToTolerance,
    settingsApi,
    toleranceToPercent,
    type ChannelImportPolicy,
} from "./settings";

/**
 * 导入声道策略设置的回归测试。
 *
 * 容差在界面上以**满幅百分比**呈现（0.1% ↔ 1e-3），因此换算与格式化是这块
 * 最容易出错的地方：曾经容差下拉框因为"选项用 '1e-3' 字面量、当前值用
 * String(1e-3)='0.001'"而永久显示空白 —— 界面看起来是坏的，却没有任何断言
 * 会失败。下面守住的是同一类问题：换算必须精确、必须单调、必须不产生
 * 二进制表示残渣。
 */

/** 等待全部微任务（含 thenable 链）排空；setTimeout 让出事件循环即可。 */
const flushMicrotasks = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

describe("channel import policy", () => {
    it("maps the default tolerance to a clean percentage", () => {
        expect(DEFAULT_CHANNEL_IMPORT_POLICY.tolerance).toBe(1e-3);
        expect(toleranceToPercent(1e-3)).toBe(0.1);
        expect(toleranceToPercent(0)).toBe(0);
        // 后端钳制上限 0.1 ↔ 界面 10%。
        expect(toleranceToPercent(0.1)).toBe(TOLERANCE_PERCENT_MAX);
    });

    it("round-trips every representable percentage without drift", () => {
        for (const percent of [0, 0.01, 0.1, 0.5, 1, 2.5, 10]) {
            expect(toleranceToPercent(percentToTolerance(percent))).toBe(percent);
        }
    });

    it("does not leak binary float residue into the displayed value", () => {
        // 1e-6 × 100 在 IEEE754 下是 0.00009999999999999999；直接展示会让
        // 输入框里出现一串数字垃圾（旧配置里可能存着 1e-6）。
        expect(String(toleranceToPercent(1e-6))).toBe("0.0001");
        expect(String(toleranceToPercent(1e-5))).toBe("0.001");
        for (const tolerance of [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]) {
            const text = String(toleranceToPercent(tolerance));
            expect(text.length).toBeLessThan(10);
        }
    });

    it("survives non-finite inputs from an empty or malformed field", () => {
        expect(toleranceToPercent(Number.NaN)).toBe(0);
        expect(percentToTolerance(Number.NaN)).toBe(0);
    });

    it("preserves every preset-equivalent percentage through normalization", () => {
        for (const percent of [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]) {
            const policy: ChannelImportPolicy = {
                ...DEFAULT_CHANNEL_IMPORT_POLICY,
                tolerance: percentToTolerance(percent),
            };
            expect(normalizeChannelImportPolicy(policy).tolerance).toBe(
                percentToTolerance(percent),
            );
        }
    });

    it("clamps percentages above the backend limit", () => {
        // 超过 10% 的输入会被后端钳到 0.1；界面回读为 10%，不会"跳回去"。
        const normalized = normalizeChannelImportPolicy({
            ...DEFAULT_CHANNEL_IMPORT_POLICY,
            tolerance: percentToTolerance(25),
        });
        expect(normalized.tolerance).toBe(0.1);
        expect(toleranceToPercent(normalized.tolerance)).toBe(TOLERANCE_PERCENT_MAX);
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

describe("saveUiSettings write queue", () => {
    beforeEach(() => {
        invokeMock.mockReset();
    });

    it("serializes concurrent partial saves in dispatch order", async () => {
        // 每笔 invoke 挂起直到测试放行，模拟真实后端"读-改-写"期间的时间窗。
        const release: Array<() => void> = [];
        invokeMock.mockImplementation(
            () =>
                new Promise<void>((resolve) => {
                    release.push(resolve);
                }),
        );

        const first = settingsApi.saveUiSettings({ midiFillGaps: true });
        const second = settingsApi.saveUiSettings({ autoCrossfade: false });
        const third = settingsApi.saveUiSettings({ snapEnabled: true });

        // 队列的首笔在微任务里触发 invoke；排空后应只有第一笔真正发出，
        // 其余必须等前一笔落盘后才开始（否则后端的并发合并会互相覆盖丢字段）。
        await flushMicrotasks();
        expect(invokeMock).toHaveBeenCalledTimes(1);
        expect(invokeMock.mock.calls[0][0]).toBe("save_ui_settings");
        expect(invokeMock.mock.calls[0][1]).toEqual({ settings: { midiFillGaps: true } });

        release[0]?.();
        await first;
        await flushMicrotasks();
        expect(invokeMock).toHaveBeenCalledTimes(2);
        expect(invokeMock.mock.calls[1][1]).toEqual({ settings: { autoCrossfade: false } });

        release[1]?.();
        await second;
        await flushMicrotasks();
        expect(invokeMock).toHaveBeenCalledTimes(3);
        expect(invokeMock.mock.calls[2][1]).toEqual({ settings: { snapEnabled: true } });

        release[2]?.();
        await third;
    });

    it("keeps draining the queue after a failed save", async () => {
        invokeMock.mockImplementation(async () => ({ ok: true }));
        invokeMock.mockImplementationOnce(async () => {
            throw new Error("save failed");
        });

        const failing = settingsApi.saveUiSettings({ autoCrossfade: true });
        const following = settingsApi.saveUiSettings({ autoCrossfade: false });

        // 这一笔照常向调用方抛错（fire-and-forget 语义不变）。
        await expect(failing).rejects.toThrow("save failed");
        // 但失败不能断链：下一笔仍要执行，否则后续保存永远卡在队列里。
        await following;
        expect(invokeMock).toHaveBeenCalledTimes(2);
    });
});
