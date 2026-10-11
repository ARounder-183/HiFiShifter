/**
 * 前端错误回传的**去重**契约。
 *
 * 【要钉死什么】后端限流能保住日志文件，却挡不住每次失败都要跨一次 IPC（错误对象
 * 含最长 4000 字符的栈）。所以前端也必须按"同一错误"去重：窗口内只上报一次，
 * 窗口过后再报一次并附上重复次数。实测一个终态错误被重试 4 次/秒，两侧都做才既
 * 不刷屏、也不白费 IPC。
 */
// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";

// 【为什么参数是 `unknown[]` 而不是空】`reportFrontendError` 会以
// `(command, { message, detail })` 调用宿主 invoke；测试要能读到那份载荷。
const invokeMock = vi.fn((..._args: unknown[]) => Promise.resolve(null));

vi.mock("./pluginHost", () => ({
    getPluginHost: () => ({
        invoke: (...args: unknown[]) => invokeMock(...args),
    }),
}));

import { reportFrontendError, resetFrontendErrorDedupForTests } from "./frontendErrorLog";

beforeEach(() => {
    invokeMock.mockClear();
    resetFrontendErrorDedupForTests();
});

test("the same error is reported once inside the dedup window", () => {
    reportFrontendError("Invoke failed: x", "stack trace");
    reportFrontendError("Invoke failed: x", "stack trace");
    reportFrontendError("Invoke failed: x", "stack trace");
    expect(invokeMock).toHaveBeenCalledTimes(1);
});

test("a different error is not suppressed by another error's dedup", () => {
    reportFrontendError("Invoke failed: a", "stack a");
    reportFrontendError("Invoke failed: b", "stack b");
    expect(invokeMock).toHaveBeenCalledTimes(2);
});

test("after the window elapses the error is reported again with a repeat count", () => {
    vi.useFakeTimers();
    try {
        reportFrontendError("Invoke failed: x", "stack");
        reportFrontendError("Invoke failed: x", "stack");
        reportFrontendError("Invoke failed: x", "stack");
        expect(invokeMock).toHaveBeenCalledTimes(1);

        vi.advanceTimersByTime(61_000);
        reportFrontendError("Invoke failed: x", "stack");
        expect(invokeMock).toHaveBeenCalledTimes(2);
        // 4 次调用里首次已上报，其余 3 次被抑制 —— 汇总必须如实说出这个数字，
        // 否则用户会以为"只重复了一次"。
        const args = invokeMock.mock.calls[1]?.[1] as { message: string };
        expect(args.message).toContain("repeated 3 time(s)");
    } finally {
        vi.useRealTimers();
    }
});

test("the dedup key is bounded so changing stack addresses cannot defeat it", () => {
    // 同一错误、栈尾不同（地址/行号每次变）：仍必须被认成同一类。
    const head = "Invoke failed: plugin_get_apply_state";
    reportFrontendError(head, `${"a".repeat(300)}-one`);
    reportFrontendError(head, `${"a".repeat(300)}-two`);
    expect(invokeMock).toHaveBeenCalledTimes(1);
});
