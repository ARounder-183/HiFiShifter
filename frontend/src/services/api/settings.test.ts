import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("../invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

import {
    DEFAULT_CHANNEL_IMPORT_POLICY,
    DEFAULT_PEN_INPUT_SETTINGS,
    TOLERANCE_PERCENT_MAX,
    normalizeChannelImportPolicy,
    normalizePenInputSettings,
    percentToTolerance,
    settingsApi,
    toleranceToPercent,
    type ChannelImportPolicy,
    type PenInputSettings,
} from "./settings";

/**
 * 导入声道策略设置的回归测试。
 *
 * 容差在界面上以**满幅百分比**呈现（默认 1% ↔ 1e-2），因此换算与格式化是这块
 * 最容易出错的地方：曾经容差下拉框因为"选项用 '1e-3' 字面量、当前值用
 * String(1e-3)='0.001'"而永久显示空白 —— 界面看起来是坏的，却没有任何断言
 * 会失败。下面守住的是同一类问题：换算必须精确、必须单调、必须不产生
 * 二进制表示残渣。
 */

/** 等待全部微任务（含 thenable 链）排空；setTimeout 让出事件循环即可。 */
const flushMicrotasks = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

describe("channel import policy", () => {
    it("maps the default tolerance to a clean percentage", () => {
        // 默认 1%（满幅的百分之一）：有损编码的左右残留就在这个量级，
        // 取 0.1% 会把大量"内容其实一致"的素材判成真立体声（漏判）。
        expect(DEFAULT_CHANNEL_IMPORT_POLICY.tolerance).toBe(1e-2);
        expect(toleranceToPercent(DEFAULT_CHANNEL_IMPORT_POLICY.tolerance)).toBe(1);
        expect(toleranceToPercent(1e-3)).toBe(0.1);
        expect(toleranceToPercent(0)).toBe(0);
        // 后端钳制上限 1（满幅）↔ 界面 100%。
        expect(toleranceToPercent(1)).toBe(TOLERANCE_PERCENT_MAX);
    });

    it("round-trips every representable percentage without drift", () => {
        for (const percent of [0, 0.01, 0.1, 0.5, 1, 2.5, 10, 50, 100]) {
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
        for (const percent of [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]) {
            const policy: ChannelImportPolicy = {
                ...DEFAULT_CHANNEL_IMPORT_POLICY,
                tolerance: percentToTolerance(percent),
            };
            expect(normalizeChannelImportPolicy(policy).tolerance).toBe(
                percentToTolerance(percent),
            );
        }
    });

    it("preserves in-range percentages up to the backend limit", () => {
        // 0~100% 全段合法：25% 不再被钳回 10%。
        const inRange = normalizeChannelImportPolicy({
            ...DEFAULT_CHANNEL_IMPORT_POLICY,
            tolerance: percentToTolerance(25),
        });
        expect(inRange.tolerance).toBe(percentToTolerance(25));
        // 超过 100% 的输入会被后端钳到满幅 1；界面回读为 100%，不会"跳回去"。
        const normalized = normalizeChannelImportPolicy({
            ...DEFAULT_CHANNEL_IMPORT_POLICY,
            tolerance: percentToTolerance(150),
        });
        expect(normalized.tolerance).toBe(1);
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
        // 5.0 超出 [0,1] → 钳到满幅 1。
        expect(normalized.tolerance).toBeLessThanOrEqual(1);
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

describe("pen input settings", () => {
    it("ships defaults that leave every device on the legacy path", () => {
        // `auto` 必须等价于"和引入本块之前完全一样"，否则升级即改手感。
        expect(DEFAULT_PEN_INPUT_SETTINGS.device).toBe("auto");
        // 压感默认开，但无压感设备（鼠标 / 触摸）会自动退化，因此不会误伤。
        expect(DEFAULT_PEN_INPUT_SETTINGS.pressureEnabled).toBe(true);
        // 倾斜默认关：它是最不可靠的一轴，绑一个会漂移的语义比留白更糟。
        expect(DEFAULT_PEN_INPUT_SETTINGS.tiltEnabled).toBe(false);
    });

    it("keeps every valid value untouched", () => {
        const settings: PenInputSettings = {
            ...DEFAULT_PEN_INPUT_SETTINGS,
            device: "trackpad",
            pressureDeadZone: 0.1,
            pressureCeiling: 0.8,
            pressureMinGain: 0.3,
            pressureMaxGain: 2,
            pressureGamma: 2,
            contactReadout: "always",
        };
        expect(normalizePenInputSettings(settings)).toEqual(settings);
    });

    it("falls back on an unknown device or readout enum", () => {
        const normalized = normalizePenInputSettings({
            ...DEFAULT_PEN_INPUT_SETTINGS,
            device: "joystick" as PenInputSettings["device"],
            contactReadout: "sometimes" as PenInputSettings["contactReadout"],
        });
        expect(normalized.device).toBe(DEFAULT_PEN_INPUT_SETTINGS.device);
        expect(normalized.contactReadout).toBe(DEFAULT_PEN_INPUT_SETTINGS.contactReadout);
    });

    it("keeps the dead zone strictly below the ceiling", () => {
        // 跨度归零会让映射退化成一条水平线 —— 压感彻底失效且无从察觉。
        const normalized = normalizePenInputSettings({
            ...DEFAULT_PEN_INPUT_SETTINGS,
            pressureDeadZone: 0.9,
            pressureCeiling: 0.2,
        });
        expect(normalized.pressureCeiling).toBeGreaterThan(normalized.pressureDeadZone);
    });

    it("keeps the max gain at or above the min gain", () => {
        const normalized = normalizePenInputSettings({
            ...DEFAULT_PEN_INPUT_SETTINGS,
            pressureMinGain: 2,
            pressureMaxGain: 0.5,
        });
        expect(normalized.pressureMaxGain).toBeGreaterThanOrEqual(normalized.pressureMinGain);
    });

    it("clamps out-of-range numbers and survives non-finite input", () => {
        const normalized = normalizePenInputSettings({
            ...DEFAULT_PEN_INPUT_SETTINGS,
            pressureDeadZone: Number.NaN,
            pressureGamma: -5,
            pressureMinGain: 999,
        });
        expect(normalized.pressureDeadZone).toBe(DEFAULT_PEN_INPUT_SETTINGS.pressureDeadZone);
        expect(normalized.pressureGamma).toBeGreaterThan(0);
        expect(normalized.pressureMinGain).toBeLessThanOrEqual(4);
    });

    it("never lets the floor exceed the dead zone", () => {
        const normalized = normalizePenInputSettings({
            ...DEFAULT_PEN_INPUT_SETTINGS,
            pressureDeadZone: 0.05,
            pressureFloor: 0.4,
        });
        expect(normalized.pressureFloor).toBeLessThanOrEqual(normalized.pressureDeadZone);
    });

    it("coerces the boolean switches rather than trusting the wire", () => {
        const normalized = normalizePenInputSettings({
            ...DEFAULT_PEN_INPUT_SETTINGS,
            pressureEnabled: 1 as unknown as boolean,
            trackpadPinchZoom: 0 as unknown as boolean,
        });
        expect(normalized.pressureEnabled).toBe(true);
        expect(normalized.trackpadPinchZoom).toBe(false);
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
