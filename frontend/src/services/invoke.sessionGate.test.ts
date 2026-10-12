/**
 * 会话闸门的**调用侧**契约：终态一次判定、全局停止。
 *
 * 【为什么必须钉在 `invoke` 这一层】用户报障的 Cubase 场景里，`get_playback_state`
 * 在会话终结后仍被每 400ms 问一次，持续 5 分钟以上（实测 750 次），每次都跨一次 IPC
 * 并把整条错误对象（含最长 4000 字符的栈）序列化过去。前端错误日志的去重只能压住
 * **写日志**的那一半，压不住"每次失败都要跨一次 IPC"。
 *
 * 本用例断言：闸门一落，后续插件调用**不再产生任何 IPC**，并原样保留第一条终态的
 * 协议码（界面据它决定给哪句指引）。
 */
// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";

import type { WebViewMessagePort } from "./pluginHost";

function nativePort() {
    const listeners = new Set<(event: { data: unknown }) => void>();
    const sent: Record<string, unknown>[] = [];
    const port: WebViewMessagePort = {
        postMessage(message) {
            sent.push(message as Record<string, unknown>);
        },
        addEventListener(_name, listener) {
            listeners.add(listener);
        },
        removeEventListener(_name, listener) {
            listeners.delete(listener);
        },
    };
    return { port, sent, deliver: (data: unknown) => listeners.forEach((fn) => fn({ data })) };
}

/** 只数某条命令的 IPC：`reportFrontendError` 也会发一条 `log_frontend_error`，
 *  它是有意保留的（错误必须进日志），不属于"命令仍被发出"。 */
const commandCalls = (sent: Record<string, unknown>[], command: string) =>
    sent.filter((message) => message.command === command).length;

afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

it("a terminal response latches the gate and later calls never reach the IPC", async () => {
    vi.resetModules();
    const native = nativePort();
    const lifecycle = new Set<() => void>();
    vi.stubGlobal("window", {
        __HFS_PLUGIN_BOOTSTRAP__: { version: 1, viewId: "gate-view" },
        chrome: { webview: native.port },
        addEventListener: (_name: string, fn: () => void) => lifecycle.add(fn),
    });
    try {
        const { invoke, invokeErrorCode, invokeErrorKind } = await import("./invoke");
        const { ERROR_CODE, isBackendTerminal, sessionTerminal } = await import("./sessionGate");

        // 第一条命令：后端回一个终态失败（宿主从未绑定 ARA）。
        const first = invoke("get_timeline_state");
        expect(commandCalls(native.sent, "get_timeline_state")).toBe(1);
        native.deliver({
            version: 1,
            viewId: "gate-view",
            id: 1,
            ok: false,
            errorKind: "terminal",
            errorCode: ERROR_CODE.ARA_NOT_BOUND,
            error: "plug-in instance is not bound to an ARA document",
        });
        const error = await first.then(
            () => null,
            (cause: unknown) => cause,
        );
        expect(error).not.toBeNull();
        expect(invokeErrorKind(error)).toBe("terminal");
        expect(invokeErrorCode(error)).toBe(ERROR_CODE.ARA_NOT_BOUND);

        // 闸门已落，且保留了第一条终态的协议码（后到的连带失败不得覆盖）。
        expect(isBackendTerminal()).toBe(true);
        expect(sessionTerminal()?.code).toBe(ERROR_CODE.ARA_NOT_BOUND);

        // 后续 20 次调用**一次 IPC 都不该产生**。
        const before = commandCalls(native.sent, "get_timeline_state");
        for (let i = 0; i < 20; i += 1) {
            await invoke("get_timeline_state").then(
                () => undefined,
                () => undefined,
            );
        }
        expect(commandCalls(native.sent, "get_timeline_state")).toBe(before);
    } finally {
        lifecycle.forEach((close) => close());
    }
});

it("a transient failure leaves the gate open so polling can recover", async () => {
    vi.resetModules();
    const native = nativePort();
    const lifecycle = new Set<() => void>();
    vi.stubGlobal("window", {
        __HFS_PLUGIN_BOOTSTRAP__: { version: 1, viewId: "gate-view" },
        chrome: { webview: native.port },
        addEventListener: (_name: string, fn: () => void) => lifecycle.add(fn),
    });
    try {
        const { invoke } = await import("./invoke");
        const { isBackendTerminal } = await import("./sessionGate");

        const first = invoke("get_timeline_state");
        native.deliver({
            version: 1,
            viewId: "gate-view",
            id: 1,
            ok: false,
            errorKind: "transient",
            error: "FX editor not connected to its processor yet",
        });
        await first.then(
            () => undefined,
            () => undefined,
        );
        // 【为什么必须钉住】"编辑器尚未连上处理器"是**暂时**的：组件创建与绑定之间
        // 必然经历它。把瞬时失败也锁成终态会让界面刚打开就宣称"会话已结束"。
        expect(isBackendTerminal()).toBe(false);

        // 闸门仍然开着 ⇒ 下一条命令照常发出。
        void invoke("get_timeline_state").then(
            () => undefined,
            () => undefined,
        );
        expect(commandCalls(native.sent, "get_timeline_state")).toBe(2);
    } finally {
        lifecycle.forEach((close) => close());
    }
});
