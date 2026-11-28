/**
 * 日志详细程度的源码合同：五语键集合、默认档位、唯一入口、命令数量。
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
// vitest 的工作目录可能是仓库根也可能是 frontend/；用 __dirname 反推才稳定。
const ROOT = join(HERE, "..");
const REPO = join(ROOT, "..");
import { expect, test } from "vitest";

import { messages, type Locale } from "../src/i18n/messages";

const KEYS = [
    "menu_log_verbosity",
    "menu_log_level_off",
    "menu_log_level_low",
    "menu_log_level_medium",
    "menu_log_level_high",
    "menu_log_saved_failed",
];

test("all five catalogs contain the full menu_log key set", () => {
    for (const [locale, dict] of Object.entries(messages) as [Locale, Record<string, string>][]) {
        for (const key of KEYS) {
            expect(dict[key], locale + "." + key).toBeTruthy();
        }
    }
});

test("the single verbosity label carries the current level", () => {
    for (const [locale, dict] of Object.entries(messages) as [Locale, Record<string, string>][]) {
        expect(dict.menu_log_verbosity, locale).toContain("{level}");
    }
});

test("no diagnostics-summary or redundant status keys survive", () => {
    const forbidden = [
        "menu_copy_diagnostics_summary",
        "menu_copy_diagnostics_fallback",
        "menu_copy_diagnostics_failed_detail",
        "menu_copy_diagnostics_retry",
        "menu_log_verbosity_current",
        "menu_log_current",
        "menu_log_overridden",
        "menu_log_unavailable",
    ];
    for (const [locale, dict] of Object.entries(messages) as [Locale, Record<string, string>][]) {
        for (const key of forbidden) expect(dict[key], locale + "." + key).toBeUndefined();
    }
});

test("the only UI entry is one lean Help submenu", () => {
    const menuBar = readFileSync(join(ROOT, "src/components/layout/MenuBar.tsx"), "utf8");
    /*
     * 入口现在是**一段条目数据**（`useLogVerbosityMenu`）而不是一个 Radix 组件：
     * 菜单表面只有一套，条目只有一个渲染路径。判据随之从 "渲染了 DropdownMenu.Sub"
     * 变成"产出了恰好一个子菜单里的一组单选"，守的东西没变。
     */
    expect(menuBar).toContain("useLogVerbosityMenu(");
    expect(menuBar).not.toContain("menu_copy_diagnostics");
    expect(menuBar).not.toContain("diagnosticsSummary");
    const hook = readFileSync(join(ROOT, "src/components/layout/useLogVerbosityMenu.ts"), "utf8");
    // 四档在**一个**子菜单里，且每一项都是单选。
    expect(hook).toContain("items: model.options.map(");
    expect(hook).toContain("radio: true");
    expect(hook).toContain("checked: selected === option.value");
    // 子菜单里只允许四档单项；不得再塞标题、当前值、说明或分隔线。
    expect(hook).not.toContain("heading: true");
    expect(hook).not.toContain("separatorBefore");
    // 条目的构造模板只有一处（四个档位共用它）。
    expect((hook.match(/onSelect: \(\) => onChange/g) ?? []).length).toBe(1);
    const about = readFileSync(join(ROOT, "src/components/layout/AboutDialog.tsx"), "utf8");
    expect(about).not.toContain("LogVerbosity");
    const app = readFileSync(join(ROOT, "src/App.tsx"), "utf8");
    expect(app).not.toContain("hs:open-about-dialog");
});

test("the Help menu lists verbosity, log folder and diagnostics in that order", () => {
    const menuBar = readFileSync(join(ROOT, "src/components/layout/MenuBar.tsx"), "utf8");
    const start = menuBar.indexOf("const helpMenuItems");
    expect(start, "帮助菜单必须以 helpMenuItems 的形式定义").toBeGreaterThanOrEqual(0);
    const help = menuBar.slice(start, menuBar.indexOf("const languageMenuItems"));
    const verbosity = help.indexOf('key: "log-verbosity"');
    const openFolder = help.indexOf('key: "open-log-folder"');
    const exportDiagnostics = help.indexOf('key: "export-diagnostics"');
    expect(verbosity, "verbosity submenu must be present in the Help menu").toBeGreaterThanOrEqual(
        0,
    );
    expect(openFolder, "open log folder must be present in the Help menu").toBeGreaterThanOrEqual(
        0,
    );
    expect(
        exportDiagnostics,
        "export diagnostics must be present in the Help menu",
    ).toBeGreaterThanOrEqual(0);
    expect(verbosity).toBeLessThan(openFolder);
    expect(openFolder).toBeLessThan(exportDiagnostics);
});

test("log_settings.json has exactly one filename constant in Rust", () => {
    const source = readFileSync(
        join(REPO, "backend/hifishifter-kernel/src/log_settings.rs"),
        "utf8",
    );
    expect((source.match(/"log_settings[.]json"/g) ?? []).length).toBe(1);
    expect((source.match(/pub const LOG_SETTINGS_FILE_NAME/g) ?? []).length).toBe(1);
});

test("the frontend exposes exactly two log settings commands", () => {
    const api = readFileSync(join(ROOT, "src/services/api/logSettings.ts"), "utf8");
    expect(api).toContain('invoke("get_log_settings")');
    expect(api).toContain('invoke("set_log_settings"');
    expect(api).not.toContain("clear_history_logs");
    expect(api).not.toContain("read_log_tail");
});
