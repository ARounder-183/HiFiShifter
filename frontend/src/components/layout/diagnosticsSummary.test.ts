/**
 * 「复制诊断摘要」的纯模型契约：内容是稳定的、不含工程数据；日志被关闭时
 * 明确写 disabled，而不是顺手创建目录去打开一个不存在的文件夹。
 */
import { expect, test } from "vitest";

import type { LogSettingsView } from "../../services/api/logSettings";
import { diagnosticsSummary } from "./diagnosticsSummary";

function view(patch: Partial<LogSettingsView> = {}): LogSettingsView {
    return {
        ok: true,
        verbosity: "low",
        persistedVerbosity: "low",
        source: "persisted",
        overridden: false,
        fileEnabled: true,
        fileWritable: true,
        loggingHealthy: true,
        hostLoggerOwned: false,
        files: [],
        totalBytes: 0,
        maxFileBytes: 8 * 1024 * 1024,
        maxFiles: 3,
        policyVersion: 1,
        recentChanges: [],
        ...patch,
    };
}

test("the summary contains version, commit, host, verbosity and log path", () => {
    const text = diagnosticsSummary(
        { version: "0.2.0", commit: "abcdef", dirty: true },
        { device: "Host / ARA" },
        view({ verbosity: "medium", logFile: "C:\\logs\\hifishifter.log" }),
    );
    expect(text).toContain("HiFiShifter 0.2.0");
    expect(text).toContain("commit: abcdef (dirty)");
    expect(text).toContain("host: Host / ARA");
    expect(text).toContain("log verbosity: medium");
    expect(text).toContain("C:\\logs\\hifishifter.log");
    expect(text.split("\n")).toHaveLength(6);
});

test("the off level is reported explicitly without inventing a path", () => {
    const text = diagnosticsSummary(null, null, view({ verbosity: "off", fileEnabled: false }));
    expect(text).toContain("log verbosity: off");
    expect(text).toContain("log: disabled");
});

test("a failed read degrades every optional line to unknown", () => {
    const text = diagnosticsSummary(null, null, null);
    expect(text).toContain("HiFiShifter unknown");
    expect(text).toContain("commit: unknown");
    expect(text).toContain("host: unknown");
    expect(text).toContain("log verbosity: unknown");
    expect(text).toContain("log: unavailable");
});

test("an unhealthy logger is visible in the summary", () => {
    const text = diagnosticsSummary(null, null, view({ loggingHealthy: false }));
    expect(text).toContain("logging: unhealthy");
});
