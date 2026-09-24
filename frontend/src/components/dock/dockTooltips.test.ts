import { test } from "vitest";

import { dockDragHint, dockModifierLabel } from "./dockTooltips.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

test("components/dock/dockTooltips.test.ts scripted checks", async () => {
    // ── 修饰键名称不能落到"无绑定"占位符 ───────────────────────────
    //
    // 用户报告提示显示成"按住 - 可停靠"。根因是修饰键绑定用 `key: "__none__"` 构造，
    // 而 `isNoneBinding` 只看这一个字段 —— 命中后 `formatKeybinding` 走"无绑定"
    // 分支返回占位符（`—`），而不是修饰键名称。
    {
        const primary = dockModifierLabel("primary");
        assertEqual(primary, "Ctrl", "primary resolves to a modifier name, not the placeholder");
        assert(
            primary !== "—" && primary !== "-" && primary !== "__none__",
            "and definitely not a placeholder",
        );

        assertEqual(dockModifierLabel("alt"), "Alt", "alt label");
        assertEqual(dockModifierLabel("shift"), "Shift", "shift label");
        assertEqual(
            dockModifierLabel("none"),
            null,
            "no modifier → null (caller picks other copy)",
        );
        // 缺省（未设置）按 primary 处理。
        assertEqual(dockModifierLabel(undefined), "Ctrl", "undefined falls back to primary");
    }

    // ── 提示模板：两行 + 修饰键文本 ────────────────────────────────
    {
        const translate = (key: string) =>
            key === "dock_drag_hint"
                ? "拖拽以重排\n按住 {modifier} 可停靠"
                : "拖拽以重排\n拖到目标位置即可停靠";

        const hint = dockDragHint("primary", translate);
        assertEqual(hint, "拖拽以重排\n按住 Ctrl 可停靠", "modifier is interpolated");
        assert(hint.includes("\n"), "the hint is two lines (the custom tooltip renders pre-line)");
        assert(!hint.includes("{modifier}"), "no placeholder is left behind");

        // `dockModifier: "none"`（始终停靠）→ 换一句不带修饰键的提示。
        assertEqual(
            dockDragHint("none", translate),
            "拖拽以重排\n拖到目标位置即可停靠",
            "the always-dock variant omits the modifier",
        );
    }
});
