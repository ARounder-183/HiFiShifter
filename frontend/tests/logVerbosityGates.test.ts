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
    expect(menuBar).toContain("<LogVerbosityMenuItems");
    expect(menuBar).not.toContain("menu_copy_diagnostics");
    expect(menuBar).not.toContain("diagnosticsSummary");
    const component = readFileSync(
        join(ROOT, "src/components/layout/LogVerbosityMenuItems.tsx"),
        "utf8",
    );
    expect(component).toContain("<DropdownMenu.Sub>");
    expect(component).toContain("<DropdownMenu.SubTrigger");
    expect(component).toContain("<DropdownMenu.SubContent>");
    expect(component).toContain("<DropdownMenu.RadioGroup");
    // SubContent 里只允许四档单项；不得再塞标题、当前值、说明或分隔线。
    expect(component).not.toContain("<DropdownMenu.Label");
    expect(component).not.toContain("<DropdownMenu.Separator");
    expect((component.match(/<DropdownMenu.RadioItem/g) ?? []).length).toBe(1);
    const about = readFileSync(join(ROOT, "src/components/layout/AboutDialog.tsx"), "utf8");
    expect(about).not.toContain("LogVerbosity");
    const app = readFileSync(join(ROOT, "src/App.tsx"), "utf8");
    expect(app).not.toContain("hs:open-about-dialog");
});

test("the Help menu lists verbosity, log folder and diagnostics in that order", () => {
    const menuBar = readFileSync(join(ROOT, "src/components/layout/MenuBar.tsx"), "utf8");
    const help = menuBar.slice(menuBar.indexOf("{/* Help Menu */}"));
    const content = help.slice(0, help.indexOf("</DropdownMenu.Content>"));
    const verbosity = content.indexOf("<LogVerbosityMenuItems");
    const openFolder = content.indexOf('{tf("menu_open_log_folder")}');
    const exportDiagnostics = content.indexOf('{tf("menu_export_diagnostics")}');
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
