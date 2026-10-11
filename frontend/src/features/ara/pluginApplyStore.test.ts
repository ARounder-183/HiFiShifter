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
    // 生产实现读 `BackendInvokeError.kind`；这里按同样的语义从错误对象上取，
    // 让测试能构造"终态失败"（见下面的 terminal tests）。
    invokeErrorKind: (error: unknown) =>
        error && typeof error === "object" && (error as { kind?: unknown }).kind === "terminal"
            ? "terminal"
            : "transient",
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

/**
 * 【为什么必须钉死终态停止】后端在文档/组件关闭后会把同一个终态错误答给每一次
 * 轮询。实测在 Cubase 上这一个缺陷把一个终态错误放大成 19,056 条日志、轮转文件
 * 合计约 32 MB。轮询必须据此**永久停止**，而不是无限重试。
 */
test("a terminal failure stops polling for good and marks the session ended", async () => {
    const terminal = Object.assign(new Error("document closed"), { kind: "terminal" });
    invokeMock.mockRejectedValue(terminal);
    stop = startPluginApplyPolling(async () => undefined);

    await flush();
    expect(getPluginApplySnapshot().sessionEnded).toBe(true);
    expect(getPluginApplySnapshot().failure).toContain("document closed");
    const afterFailure = invokeMock.mock.calls.length;

    // 再推很久也不该有任何新请求：会话已结束，重试没有意义。
    await vi.advanceTimersByTimeAsync(60_000);
    await flush();
    expect(invokeMock.mock.calls.length).toBe(afterFailure);
});

/**
 * 【为什么要退避】暂时性失败（宿主正在重载、PCM 尚未就绪）确实值得重试，但按
 * 250 ms 的节奏重试 4 次/秒同样是白费 IPC。策略是"起步仍然快、随后成倍拉长"：
 * 第一次重试仍等基础间隔（保持状态栏响应），之后每失败一次 ×4，上限 30s。
 */
test("a transient failure backs off instead of retrying every 250ms", async () => {
    const transient = Object.assign(new Error("host busy"), { kind: "transient" });
    invokeMock.mockRejectedValue(transient);
    stop = startPluginApplyPolling(async () => undefined);

    await flush();
    expect(getPluginApplySnapshot().sessionEnded).toBe(false);
    const afterFirst = invokeMock.mock.calls.length;

    // 第一次重试仍等基础间隔（250 ms）：起步不牺牲响应。
    await vi.advanceTimersByTimeAsync(250);
    await flush();
    const afterSecond = invokeMock.mock.calls.length;
    expect(afterSecond).toBeGreaterThan(afterFirst);

    // 第二次失败后间隔翻到 1s（250×4）：再等 250 ms 不该有新请求。
    await vi.advanceTimersByTimeAsync(250);
    await flush();
    expect(invokeMock.mock.calls.length).toBe(afterSecond);

    // 满 1s 之后才重试。
    await vi.advanceTimersByTimeAsync(750);
    await flush();
    expect(invokeMock.mock.calls.length).toBeGreaterThan(afterSecond);
});

/** 成功一次之后退避必须复位，否则状态栏会永久停在慢节奏上。 */
test("a success after a transient failure resets the poll cadence", async () => {
    invokeMock.mockRejectedValueOnce(
        Object.assign(new Error("host busy"), { kind: "transient" }),
    );
    stop = startPluginApplyPolling(async () => undefined);
    await flush();
    const afterFailure = invokeMock.mock.calls.length;

    invokeMock.mockResolvedValue(readyState());
    await vi.advanceTimersByTimeAsync(1000);
    await flush();
    expect(getPluginApplySnapshot().failure).toBe("");

    // 复位后回到 250 ms 节奏：再等一个基础间隔就应有新请求。
    const afterSuccess = invokeMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(250);
    await flush();
    expect(invokeMock.mock.calls.length).toBeGreaterThan(afterSuccess);
});
