/**
 * 跨窗口状态桥的自检（独立窗口功能的地基）。
 *
 * 【要钉死的性质】
 * 1. 收到桥来的动作**不再回传**（否则两个窗口互相转发，形成无限回环）；
 * 2. 一次帧内的多个动作合并成一个批次发出（播放轮询这类高频路径不会把 IPC 打满）；
 * 3. 不可序列化的动作跳过广播而不是抛错（派发路径绝不能被桥打断）；
 * 4. 快照动作能把整份状态替换进 store（独立窗口的启动路径）。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const emitted: Array<{ event: string; payload: unknown }> = [];
/** 事件名 → 已注册的处理器（测试可手动触发，模拟另一个窗口的广播）。 */
const listeners = new Map<string, (event: { payload: unknown }) => void>();
vi.mock("@tauri-apps/api/event", () => ({
    emit: async (event: string, payload?: unknown) => {
        emitted.push({ event, payload });
    },
    listen: async (event: string, handler: (event: { payload: unknown }) => void) => {
        listeners.set(event, handler);
        return () => listeners.delete(event);
    },
}));

import {
    BRIDGE_ACTION_EVENT,
    BRIDGE_APPEARANCE_EVENT,
    BRIDGE_SNAPSHOT_EVENT,
    BRIDGE_SNAPSHOT_REQUEST_EVENT,
    createStoreBridgeMiddleware,
    installBridge,
    isSatelliteWindow,
    projectSnapshot,
    resolveWindowLabel,
    satelliteFormId,
    subscribeRemoteAppearance,
} from "./detachBridge";

function makeStore(role: "main" | "satellite" = "main") {
    // 极简 store：只需要 dispatch 能穿过中间件链。
    let state: unknown = { value: 0 };
    const listeners: Array<() => void> = [];
    const baseDispatch = (action: { type: string; payload?: unknown }) => {
        if (action.type === "set") state = { value: action.payload };
        for (const listener of listeners) listener();
        return action;
    };
    const middleware = createStoreBridgeMiddleware(role);
    const dispatch = middleware({
        getState: () => state,
        dispatch: (action: { type: string; payload?: unknown }) => dispatch(action as never),
    } as never)(baseDispatch as never) as (action: { type: string; payload?: unknown }) => unknown;
    return {
        dispatch,
        getState: () => state,
        subscribe: (listener: () => void) => {
            listeners.push(listener);
            return () => {
                const index = listeners.indexOf(listener);
                if (index >= 0) listeners.splice(index, 1);
            };
        },
    };
}

/** 让排队的 rAF 回调跑掉。 */
function flushFrame() {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

describe("detachBridge（跨窗口状态桥）", () => {
    beforeEach(() => {
        emitted.length = 0;
        listeners.clear();
        // node 环境没有 rAF：装一个立即在下一个宏任务触发的桩，让"帧合并"可确定性
        // 验证（中间件在无 rAF 时会退化为 16ms 定时器）。
        (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame = (
            callback: (time: number) => void,
        ) => setTimeout(() => callback(0), 0) as unknown as number;
    });

    afterEach(() => {
        delete (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
        vi.restoreAllMocks();
    });

    it("窗口标签：主窗口无参数，卫星窗口带窗体 id", () => {
        const original = globalThis.window;
        const setSearch = (search: string) => {
            (globalThis as { window?: unknown }).window = { location: { search } };
        };
        setSearch("");
        expect(resolveWindowLabel()).toBe("main");
        expect(isSatelliteWindow()).toBe(false);
        expect(satelliteFormId()).toBeNull();

        setSearch("?hsDetachedForm=notebook");
        expect(resolveWindowLabel()).toBe("detached:notebook");
        expect(isSatelliteWindow()).toBe(true);
        expect(satelliteFormId()).toBe("notebook");

        (globalThis as { window?: unknown }).window = original;
    });

    it("★ 收到桥来的动作不再回传（无回环）", async () => {
        const store = makeStore();
        await flushFrame();
        emitted.length = 0;
        store.dispatch({ type: "remote/thing", meta: { hsRemote: true } } as never);
        await flushFrame();
        expect(emitted).toHaveLength(0);
    });

    it("★ 快照动作绝不进广播通道（卫星应用快照不会把主窗口状态整体回滚）", async () => {
        // 回归：卫星 onSnapshot 派发的快照动作此前会被中间件广播回主窗口，主窗口
        // 的根 reducer 又对它做整体替换 —— 一次 detach 就把主窗口回滚到快照时刻。
        const store = makeStore();
        await flushFrame();
        emitted.length = 0;
        // 即便忘记标 hsRemote，类型本身也必须被拦下。
        store.dispatch({ type: "hs/bridgeSnapshot", payload: { value: 1 } } as never);
        store.dispatch({
            type: "hs/bridgeSnapshot",
            payload: { value: 1 },
            meta: { hsRemote: true },
        } as never);
        await flushFrame();
        expect(emitted.filter((item) => item.event === BRIDGE_ACTION_EVENT)).toHaveLength(0);
    });

    it("★ 一次帧内的多个动作合并成一批发出", async () => {
        const store = makeStore();
        await flushFrame();
        emitted.length = 0;
        store.dispatch({ type: "a" });
        store.dispatch({ type: "b" });
        store.dispatch({ type: "c" });
        await flushFrame();
        const batches = emitted.filter((item) => item.event === BRIDGE_ACTION_EVENT);
        expect(batches).toHaveLength(1);
        const payload = batches[0].payload as { origin: string; actions: Array<{ type: string }> };
        expect(payload.origin).toBe("main");
        expect(payload.actions.map((action) => action.type)).toEqual(["a", "b", "c"]);
    });

    it("动作仍然照常流向下游（中间件不吞动作）", () => {
        const store = makeStore();
        store.dispatch({ type: "set", payload: 42 });
        expect(store.getState()).toEqual({ value: 42 });
    });

    it("不可序列化的动作跳过广播，且不抛错", async () => {
        const store = makeStore();
        await flushFrame();
        emitted.length = 0;
        const circular: Record<string, unknown> = { type: "bad" };
        circular.self = circular;
        expect(() => store.dispatch(circular as never)).not.toThrow();
        await flushFrame();
        expect(emitted.filter((item) => item.event === BRIDGE_ACTION_EVENT)).toHaveLength(0);
    });
});

describe("快照投影（detachBridge 的重载荷裁剪）", () => {
    it("★ 剔除逐 clip 的波形数组（快照体积的决定项）", () => {
        const state = {
            session: {
                bpm: 120,
                clips: [
                    { id: "c0", waveform: [0.1, 0.2], waveformPreview: [0.3], gain: 1 },
                    { id: "c1", waveform: [0.4], waveformPreview: [], gain: 2 },
                ],
            },
            fileBrowser: { root: "/x" },
        };
        const projected = projectSnapshot(state) as {
            session: { bpm: number; clips: Array<Record<string, unknown>> };
            fileBrowser: unknown;
        };
        expect(projected.session.clips).toEqual([
            { id: "c0", gain: 1 },
            { id: "c1", gain: 2 },
        ]);
        // 其余字段原样保留。
        expect(projected.session.bpm).toBe(120);
        expect(projected.fileBrowser).toEqual({ root: "/x" });
    });

    it("结构异常时不抛错（投影是尽力而为）", () => {
        expect(projectSnapshot(null)).toBeNull();
        expect(projectSnapshot({ session: null })).toEqual({ session: null });
        expect(projectSnapshot({ session: { clips: "nope" } })).toEqual({
            session: { clips: "nope" },
        });
    });
});

describe("卫星窗口的快照握手（重试）", () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it("★ 未收到快照时反复重发请求（一次请求会丢，必须重试）", async () => {
        vi.useFakeTimers();
        const original = globalThis.window;
        (globalThis as { window?: unknown }).window = {
            location: { search: "?hsDetachedForm=notebook" },
        };
        try {
            const teardown = installBridge({
                role: "satellite",
                getState: () => ({}),
                dispatch: () => {},
                onSnapshot: () => {},
            });
            // 让动态 import 的事件 API 就绪。
            await vi.advanceTimersByTimeAsync(0);
            const countRequests = () =>
                emitted.filter((item) => item.event === BRIDGE_SNAPSHOT_REQUEST_EVENT).length;
            expect(countRequests()).toBe(1);
            await vi.advanceTimersByTimeAsync(1200);
            // 每 500ms 重发一次：1.2s 内至少多发两次。
            expect(countRequests()).toBeGreaterThanOrEqual(3);
            teardown();
            const before = countRequests();
            await vi.advanceTimersByTimeAsync(1200);
            expect(countRequests()).toBe(before);
        } finally {
            (globalThis as { window?: unknown }).window = original;
        }
    });

    it("★ 收到快照后停止重发，并把状态交给 onSnapshot", async () => {
        vi.useFakeTimers();
        const original = globalThis.window;
        (globalThis as { window?: unknown }).window = {
            location: { search: "?hsDetachedForm=notebook" },
        };
        try {
            const received: unknown[] = [];
            const teardown = installBridge({
                role: "satellite",
                getState: () => ({}),
                dispatch: () => {},
                onSnapshot: (state) => received.push(state),
            });
            await vi.advanceTimersByTimeAsync(0);
            const handler = listeners.get(BRIDGE_SNAPSHOT_EVENT);
            expect(handler).toBeDefined();
            handler?.({ payload: { target: "detached:notebook", state: { ok: 1 } } });
            expect(received).toEqual([{ ok: 1 }]);
            const before = emitted.filter(
                (item) => item.event === BRIDGE_SNAPSHOT_REQUEST_EVENT,
            ).length;
            await vi.advanceTimersByTimeAsync(2000);
            expect(
                emitted.filter((item) => item.event === BRIDGE_SNAPSHOT_REQUEST_EVENT).length,
            ).toBe(before);
            teardown();
        } finally {
            (globalThis as { window?: unknown }).window = original;
        }
    });
});

describe("外观下发（独立窗口继承主题与自定义字体）", () => {
    it("★ 订阅时立即回调已收到的外观，之后每次推送都回调", async () => {
        const original = globalThis.window;
        (globalThis as { window?: unknown }).window = {
            location: { search: "?hsDetachedForm=notebook" },
        };
        try {
            const received: unknown[] = [];
            const teardown = installBridge({
                role: "satellite",
                getState: () => ({}),
                dispatch: () => {},
                onSnapshot: () => {},
            });
            await new Promise((resolve) => setTimeout(resolve, 0));

            // 主窗口下发一次外观（模拟快照里带的外观）。
            const snapshotHandler = listeners.get(BRIDGE_SNAPSHOT_EVENT);
            expect(snapshotHandler).toBeDefined();
            snapshotHandler?.({
                payload: {
                    target: "detached:notebook",
                    state: {},
                    appearance: { fontFamily: "Georgia, serif" },
                },
            });

            // 之后挂载的订阅者应当**立即**拿到已收到的值（晚挂载也要能应用）。
            const unsubscribe = subscribeRemoteAppearance((appearance) =>
                received.push(appearance),
            );
            expect(received).toEqual([{ fontFamily: "Georgia, serif" }]);

            // 变更推送：再次回调。
            const appearanceHandler = listeners.get(BRIDGE_APPEARANCE_EVENT);
            expect(appearanceHandler).toBeDefined();
            appearanceHandler?.({ payload: { appearance: { fontFamily: "Noto Sans SC" } } });
            expect(received).toEqual([
                { fontFamily: "Georgia, serif" },
                { fontFamily: "Noto Sans SC" },
            ]);

            unsubscribe();
            appearanceHandler?.({ payload: { appearance: { fontFamily: "X" } } });
            expect(received).toHaveLength(2);
            teardown();
        } finally {
            (globalThis as { window?: unknown }).window = original;
        }
    });
});
