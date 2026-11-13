/** 帮助菜单四档单选项的纯模型契约。 */
import { expect, test } from "vitest";

import type { LogSettingsView } from "../../services/api/logSettings";
import { logLevelLabelKey, resolveLogVerbosityMenu } from "./logVerbosityMenuModel";

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

test("four verbosity options are rendered in order", () => {
    expect(resolveLogVerbosityMenu(view()).options.map((o) => o.value)).toEqual([
        "off",
        "low",
        "medium",
        "high",
    ]);
});

test("the default low level is checked", () => {
    expect(resolveLogVerbosityMenu(view()).checked).toBe("low");
    expect(resolveLogVerbosityMenu(null).checked).toBe("low");
    expect(resolveLogVerbosityMenu(view({ verbosity: "high" })).checked).toBe("high");
});

test("a startup override disables every radio item", () => {
    const model = resolveLogVerbosityMenu(view({ overridden: true, source: "env" }));
    expect(model.overridden).toBe(true);
    expect(model.statusKey).toBe("menu_log_overridden");
    expect(model.options.every((o) => o.disabled)).toBe(true);
});

test("the status key follows logging health and the selected level", () => {
    expect(resolveLogVerbosityMenu(view()).statusKey).toBe("menu_log_current");
    expect(resolveLogVerbosityMenu(view({ verbosity: "off" })).statusVars).toEqual({
        level: "menu_log_level_off",
    });
    expect(resolveLogVerbosityMenu(view({ loggingHealthy: false })).statusKey).toBe(
        "menu_log_unavailable",
    );
    expect(resolveLogVerbosityMenu(view({ hostLoggerOwned: true })).statusKey).toBe(
        "menu_log_unavailable",
    );
});

test("every level has a stable label key for the submenu trigger", () => {
    expect(logLevelLabelKey("off")).toBe("menu_log_level_off");
    expect(logLevelLabelKey("low")).toBe("menu_log_level_low");
    expect(logLevelLabelKey("medium")).toBe("menu_log_level_medium");
    expect(logLevelLabelKey("high")).toBe("menu_log_level_high");
});
