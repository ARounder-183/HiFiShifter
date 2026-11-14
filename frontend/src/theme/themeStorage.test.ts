/**
 * 主题持久化：外部存储值不能绕过主题文件导入时的白名单。
 */
// @vitest-environment jsdom
import { beforeEach, describe, expect, test } from "vitest";

import { importThemeFromJson, loadCustomThemes, saveCustomThemes } from "./themeStorage";

describe("themeStorage", () => {
    beforeEach(() => localStorage.clear());

    test("persisted custom themes are filtered before reaching CSS variables", () => {
        localStorage.setItem(
            "hifishifter.customThemes",
            JSON.stringify([
                {
                    id: "valid",
                    name: "Valid",
                    base: "dark",
                    colors: {
                        "qt-highlight": "#123456",
                        "qt-text": "url(javascript:alert(1))",
                        "not-a-token": "red",
                    },
                    waveformColors: { fill: "#fff", stroke: "rgba(0, 0, 0, 0.5)" },
                },
                { id: "bad", name: "Bad", base: "neon", colors: {} },
                "garbage",
            ]),
        );
        expect(loadCustomThemes()).toEqual([
            {
                id: "valid",
                name: "Valid",
                base: "dark",
                colors: { "qt-highlight": "#123456" },
                waveformColors: { fill: "#fff", stroke: "rgba(0, 0, 0, 0.5)" },
                accentColor: undefined,
                grayColor: undefined,
                radius: undefined,
            },
        ]);
    });

    test("a malformed stored list falls back to an empty theme list", () => {
        localStorage.setItem("hifishifter.customThemes", '{"themes":[]}');
        expect(loadCustomThemes()).toEqual([]);
    });

    test("imported theme ids and names are validated", () => {
        expect(
            importThemeFromJson(
                JSON.stringify({
                    name: "Imported",
                    base: "dark",
                    colors: {},
                    id: { nested: true },
                }),
            )?.theme.id,
        ).toMatch(/^[0-9a-f-]{36}$/i);
        expect(importThemeFromJson("[]")?.theme?.name).toBeUndefined();
    });

    test("saveCustomThemes is the inverse of loadCustomThemes for valid themes", () => {
        saveCustomThemes([
            { id: "one", name: "One", base: "light", colors: { "qt-window": "#abcdef" } },
        ]);
        expect(loadCustomThemes()[0].colors["qt-window"]).toBe("#abcdef");
    });
});
