/**
 * frontendLog 的档位 / Off 短路 / 去重契约。
 */
// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn((..._args: unknown[]) => Promise.resolve(null));
vi.mock("./pluginHost", () => ({
    getPluginHost: () => ({ invoke: (...args: unknown[]) => invokeMock(...args) }),
}));

import {
    frontendLog,
    policyFromView,
    reportFrontendError,
    resetFrontendLogPolicyForTests,
    setFrontendLogPolicy,
} from "./frontendLog";
import type { LogSettingsView } from "./api/logSettings";

const view = (verbosity: LogSettingsView["verbosity"]): LogSettingsView =>
    ({
        ok: true,
        verbosity,
        persistedVerbosity: verbosity,
        source: "persisted",
        overridden: false,
        fileEnabled: verbosity !== "off",
        fileWritable: true,
        loggingHealthy: true,
        hostLoggerOwned: false,
        files: [],
        totalBytes: 0,
        maxFileBytes: 8 * 1024 * 1024,
        maxFiles: 3,
        policyVersion: 1,
        recentChanges: [],
    }) as LogSettingsView;

beforeEach(() => {
    invokeMock.mockClear();
    resetFrontendLogPolicyForTests();
});

test("off produces zero IPC and zero formatting work", () => {
    setFrontendLogPolicy(policyFromView(view("off")));
    frontendLog("error", "boom", new Error("stack"));
    frontendLog("debug", "trace", { a: 1 });
    expect(invokeMock).toHaveBeenCalledTimes(0);
});

test("low accepts only errors, high accepts every level", () => {
    setFrontendLogPolicy(policyFromView(view("low")));
    frontendLog("warn", "warn-1");
    expect(invokeMock).toHaveBeenCalledTimes(0);
    reportFrontendError("error-1");
    expect(invokeMock).toHaveBeenCalledTimes(1);
    const [, args] = invokeMock.mock.calls[0] as [string, { level: string }];
    expect(args.level).toBe("error");

    setFrontendLogPolicy(policyFromView(view("high")));
    frontendLog("debug", "debug-1");
    expect(invokeMock).toHaveBeenCalledTimes(2);
});

test("the same frontend error is deduplicated inside the window", () => {
    frontendLog("error", "Invoke failed: x", "stack trace");
    frontendLog("error", "Invoke failed: x", "stack trace");
    expect(invokeMock).toHaveBeenCalledTimes(1);
});

test("newlines cannot forge a second log line", () => {
    frontendLog("error", "first\n[frontend] forged");
    expect(invokeMock).toHaveBeenCalledTimes(1);
    const [, args] = invokeMock.mock.calls[0] as [string, { message: string }];
    expect(args.message).toContain("first\\n[frontend]");
});
