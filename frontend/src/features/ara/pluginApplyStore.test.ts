/**
 * 插件宿主「自动应用」外部 store 的契约。
 *
 * 【要钉死什么】
 * 1. 轮询在停止后**真的**停下来（否则一个已关闭的状态片会永远每 250 ms 打一次
 *    IPC，并且不断把快照推给已经没人订阅的监听器集合）；
 * 2. 内容没变时快照**引用不变** —— 这是"250 ms 轮询不引起重渲染"的唯一依据
 *    （`useSyncExternalStore` 靠引用相等判断）；
 * 3. 宿主版本前进时重取时间轴，且重取失败**不推进**版本号（下一次轮询要重试）；
 * 4. `force` 原样透传给 `plugin_refresh`（强制重载会丢弃未应用的本地编辑，
 *    静默降级成非强制是不可接受的）。
 *
 * 【为什么 mock 掉 `invoke` 与 `hostEvents`】它们分别落到 Tauri IPC 与事件总线，
 * 在 jsdom 里没有后端。mock 掉之后本测试只验证 store 自己的时序与语义。
 */
// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();
const handlers = new Map<string, (event: { payload: unknown }) => void>();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

vi.mock("../../services/hostEvents", () => ({
    listen: async (event: string, handler: (event: { payload: unknown }) => void) => {
        handlers.set(event, handler);
        return () => handlers.delete(event);
    },
}));

import {
    getPluginApplySnapshot,
    reloadPluginHost,
    resetPluginApplyStoreForTests,
    startPluginApplyPolling,
    subscribePluginApply,
    type PluginApplyState,
} from "./pluginApplyStore";

/** 一份"已就绪、已应用"的宿主状态。 */
function readyState(overrides: Partial<PluginApplyState> = {}): PluginApplyState {
    return {
        generation: 4,
        applied_generation: 4,
        pending: false,
        host_version: 1,
        connected: true,
        ready: true,
        error: null,
        ...overrides,
    };
}

/**
 * 让排队的微任务跑完（`poll()` 里的 await 链）。
 *
 * 【为什么用 `advanceTimersByTimeAsync(0)` 而不是 `setTimeout(resolve, 0)`】本文件
 * 装了假定时器，`setTimeout` 也是假的 —— 用它做 flush 会永远等不到回调。
 */
const flush = () => vi.advanceTimersByTimeAsync(0);

let stop: (() => void) | null = null;

beforeEach(() => {
    vi.useFakeTimers();
    invokeMock.mockReset();
    handlers.clear();
    resetPluginApplyStoreForTests();
});

afterEach(() => {
    stop?.();
    stop = null;
    vi.useRealTimers();
});

test("polling stops for good once the returned stop function runs", async () => {
    invokeMock.mockResolvedValue(readyState());
    stop = startPluginApplyPolling(async () => undefined);
    await flush();
    const afterFirst = invokeMock.mock.calls.length;
    expect(afterFirst).toBeGreaterThan(0);

    stop();
    stop = null;
    await vi.advanceTimersByTimeAsync(2000);
    await flush();
    expect(invokeMock.mock.calls.length).toBe(afterFirst);
});

test("an unchanged poll keeps the same snapshot reference", async () => {
    invokeMock.mockResolvedValue(readyState());
    let notifications = 0;
    const unsubscribe = subscribePluginApply(() => {
        notifications++;
    });
    stop = startPluginApplyPolling(async () => undefined);

    await flush();
    const first = getPluginApplySnapshot();
    expect(first.state?.generation).toBe(4);
    expect(notifications).toBe(1);

    // 三次轮询返回完全相同的宿主状态：通知次数与快照引用都不该再变。
    await vi.advanceTimersByTimeAsync(1000);
    await flush();
    expect(notifications).toBe(1);
    expect(getPluginApplySnapshot()).toBe(first);
    unsubscribe();
});

test("a pushed state event updates the snapshot without waiting for the next poll", async () => {
    invokeMock.mockResolvedValue(readyState());
    stop = startPluginApplyPolling(async () => undefined);
    await flush();

    handlers.get("plugin_apply_state")?.({ payload: readyState({ generation: 9, pending: true }) });
    expect(getPluginApplySnapshot().state?.generation).toBe(9);
    expect(getPluginApplySnapshot().state?.pending).toBe(true);
});

test("host version progress refetches the timeline, and a failed refetch retries", async () => {
    invokeMock.mockResolvedValue(readyState({ host_version: 7 }));
    let refetches = 0;
    let failNext = true;
    stop = startPluginApplyPolling(async () => {
        refetches++;
        if (failNext) throw new Error("timeline unavailable");
    });

    await flush();
    expect(refetches).toBe(1);
    // 重取失败 ⇒ 版本号不推进 ⇒ 下一次轮询必须再试一次。
    await vi.advanceTimersByTimeAsync(250);
    await flush();
    expect(refetches).toBe(2);

    failNext = false;
    await vi.advanceTimersByTimeAsync(250);
    await flush();
    expect(refetches).toBe(3);
    // 版本号已推进 ⇒ 状态没变时不再重取。
    await vi.advanceTimersByTimeAsync(1000);
    await flush();
    expect(refetches).toBe(3);
});

test("a host change event refetches the timeline immediately", async () => {
    invokeMock.mockResolvedValue(readyState({ host_version: 3 }));
    let refetches = 0;
    stop = startPluginApplyPolling(async () => {
        refetches++;
    });
    await flush();
    const before = refetches;

    handlers.get("plugin_host_changed")?.({ payload: undefined });
    await flush();
    expect(refetches).toBe(before + 1);
});

test("reload passes force through to the host instead of silently downgrading it", async () => {
    invokeMock.mockResolvedValue(readyState());
    stop = startPluginApplyPolling(async () => undefined);
    await flush();
    invokeMock.mockClear();

    let refetches = 0;
    await reloadPluginHost(true, async () => {
        refetches++;
    });

    expect(invokeMock).toHaveBeenCalledWith("plugin_refresh", true);
    expect(refetches).toBe(1);
    expect(getPluginApplySnapshot().reloading).toBe(false);
});

test("a failed reload surfaces as a communication failure and clears the busy flag", async () => {
    invokeMock.mockResolvedValue(readyState());
    stop = startPluginApplyPolling(async () => undefined);
    await flush();

    invokeMock.mockRejectedValueOnce(new Error("host refused"));
    await reloadPluginHost(false, async () => undefined);

    expect(getPluginApplySnapshot().failure).toContain("host refused");
    expect(getPluginApplySnapshot().reloading).toBe(false);
});
