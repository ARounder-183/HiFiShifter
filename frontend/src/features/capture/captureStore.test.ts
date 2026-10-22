/**
 * Capture 共享状态 store 的契约。
 *
 * 【要钉死什么】
 * 1. `stop` 之后必须推进 `recordedToken` 并给出工程秒区间 —— 时间轴的"跳到刚录好的
 *    区间"完全依赖它（见 `useCaptureReveal`）；
 * 2. 内容没变时快照**引用不变**：`useSyncExternalStore` 靠引用相等判断，否则传输栏
 *    每 200ms 重渲染一次；
 * 3. 带参命令真的把参数发出去（`takeId`）—— 本仓已三次出现"前端乐观更新生效、后端
 *    调用从未到达"的静默分叉；
 * 4. 轮询是引用计数的：控件与面板各停一次才真正停。
 */
// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

import {
    getCaptureSnapshot,
    resetCaptureStoreForTests,
    startCapturePolling,
    captureActions,
} from "./captureStore";
import type { CaptureStatus } from "./captureApi";

function status(overrides: Partial<CaptureStatus> = {}): CaptureStatus {
    return {
        state: "empty",
        recording: null,
        capture: null,
        material: null,
        hasMaterial: false,
        span: null,
        takes: [],
        options: {
            maxDurationSec: 600,
            segmentSec: 8,
            monitor: true,
            outsidePassthrough: true,
        },
        takeCount: 0,
        activeTake: null,
        sawContext: true,
        sawInput: true,
        sawOffline: false,
        canCapture: true,
        hostPlaying: false,
        unsupportedPrecision: false,
        araBound: false,
        standby: false,
        limitWarning: false,
        stoppedReason: null,
        output: "passthrough",
        ...overrides,
    };
}

beforeEach(() => {
    invokeMock.mockReset();
    resetCaptureStoreForTests();
});

afterEach(() => {
    resetCaptureStoreForTests();
});

test("停止录制会把区间与票据交给视口", async () => {
    const recorded = status({
        state: "ready",
        takeCount: 1,
        activeTake: 3,
        span: { startFrame: 48_000, endFrame: 96_000, sampleRate: 48_000 },
    });
    invokeMock.mockImplementation(async (command: string) => {
        if (command === "capture_stop") return { take: 3, status: recorded };
        return recorded;
    });
    const before = getCaptureSnapshot().recordedToken;
    await captureActions.stop();
    const snapshot = getCaptureSnapshot();
    expect(snapshot.recordedToken).toBe(before + 1);
    // 48000 帧 @48k = 1.0 s；96000 帧 = 2.0 s。
    expect(snapshot.recordedSpanSec).toEqual([1, 2]);
    expect(snapshot.status?.state).toBe("ready");
});

test("没有区间时也推进票据，但不要求视口跳转", async () => {
    const recorded = status({ state: "ready", takeCount: 1 });
    invokeMock.mockImplementation(async (command: string) =>
        command === "capture_stop" ? { take: 1, status: recorded } : recorded,
    );
    const before = getCaptureSnapshot().recordedToken;
    await captureActions.stop();
    const snapshot = getCaptureSnapshot();
    expect(snapshot.recordedToken).toBe(before + 1);
    expect(snapshot.recordedSpanSec, "没有片段就没有可定位的区间").toBeNull();
});

test("带参命令把参数发出去（takeId）", async () => {
    invokeMock.mockResolvedValue(status({ takeCount: 2 }));
    await captureActions.deleteTake(9);
    expect(invokeMock).toHaveBeenCalledWith("capture_delete_take", 9);
});

test("选项补丁是逐字段的，不会顺手改别的字段", async () => {
    invokeMock.mockResolvedValue(status());
    await captureActions.setOptions({ monitor: false });
    expect(invokeMock).toHaveBeenCalledWith("capture_set_options", { monitor: false });
});

test("命令失败时把原因留在快照里，且不假装成功", async () => {
    invokeMock.mockRejectedValue(new Error("capture_not_applicable_ara"));
    await captureActions.start();
    const snapshot = getCaptureSnapshot();
    expect(snapshot.error).toContain("capture_not_applicable_ara");
    expect(snapshot.busy).toBe(false);
    expect(snapshot.status, "失败不该改动状态").toBeNull();
});

test("轮询按引用计数停止：两个订阅者各停一次才真正停", async () => {
    invokeMock.mockResolvedValue(status());
    const clearSpy = vi.spyOn(window, "clearInterval");
    const first = startCapturePolling();
    const second = startCapturePolling();
    first();
    expect(clearSpy, "还有一个订阅者，轮询不能停").not.toHaveBeenCalled();
    second();
    expect(clearSpy, "最后一个订阅者离开后必须停").toHaveBeenCalled();
    clearSpy.mockRestore();
});

test("轮询两次得到内容相同的状态时保留快照引用", async () => {
    const stable = status();
    invokeMock.mockResolvedValue(stable);
    const stop = startCapturePolling();
    await vi.waitFor(() => expect(getCaptureSnapshot().status).toBe(stable));
    const first = getCaptureSnapshot();
    await captureActions.refresh();
    expect(getCaptureSnapshot(), "内容没变就不该换引用").toBe(first);
    stop();
});
