// @vitest-environment jsdom
/**
 * hostLinkPluginStore 的发布契约：**内容不变的轮询不得通知订阅者**。
 *
 * 【防的是什么】`hostLinkPluginApi.status()` 每 750ms 经 IPC 反序列化出全新对象。
 * 旧 `publish` 直接换引用，`useSyncExternalStore` 订阅者（状态栏控件与打开的详情
 * 对话框）于是每轮固定空重渲染。与 captureStore / pluginApplyStore 同一取舍。
 */
import { beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
    invokeErrorKind: () => "transient",
}));

import {
    getHostLinkPluginSnapshot,
    refreshHostLinkStatus,
    resetHostLinkPluginStoreForTests,
    startHostLinkPluginPolling,
    subscribeHostLinkPlugin,
} from "./hostLinkPluginStore";

const STATUS = {
    attached: true,
    phase: "ready",
    identity: "host:REAPER:doc",
    directory: "/workspaces/one",
    audio: { complete: 3, partial: 0, extern: 0 },
};

beforeEach(() => {
    resetHostLinkPluginStoreForTests();
    invokeMock.mockReset();
});

test("内容相同的轮询不重渲染订阅者，内容变化才通知", async () => {
    const listener = vi.fn();
    const unsubscribe = subscribeHostLinkPlugin(listener);
    const stopPolling = startHostLinkPluginPolling();
    invokeMock.mockResolvedValue(STATUS);

    await refreshHostLinkStatus();
    const first = getHostLinkPluginSnapshot();
    expect(first.status).toEqual(STATUS);
    expect(listener).toHaveBeenCalledTimes(1);

    // 第二轮返回**内容相同但引用不同**的对象：不得通知。
    invokeMock.mockResolvedValue({ ...STATUS });
    await refreshHostLinkStatus();
    expect(listener).toHaveBeenCalledTimes(1);
    expect(getHostLinkPluginSnapshot()).toBe(first);

    // 真正变化（哪怕只是一个字段）必须通知。
    invokeMock.mockResolvedValue({ ...STATUS, clips: 4 });
    await refreshHostLinkStatus();
    expect(listener).toHaveBeenCalledTimes(2);
    expect(getHostLinkPluginSnapshot().status?.clips).toBe(4);

    stopPolling();
    unsubscribe();
});
