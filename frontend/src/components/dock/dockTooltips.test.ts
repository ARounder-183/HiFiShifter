import { test } from "vitest";

import { dockDragHint, dockModifierHint, dockModifierLabel } from "./dockTooltips.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

/** 与真实语言包同形的文案（键名一致，值取 zh-CN）。 */
function translate(key: string): string {
    switch (key) {
        case "dock_drag_rearrange":
            return "拖拽以重排";
        case "dock_dock_hint":
            return "按住 {modifier} 可停靠";
        case "dock_dock_hint_always":
            return "拖到目标位置即可停靠";
        default:
            throw new Error(`unexpected key: ${key}`);
    }
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

    // ── 抓手提示：两行 + 修饰键文本 ────────────────────────────────
    {
        const hint = dockDragHint("primary", translate);
        assertEqual(hint, "拖拽以重排" + "\n" + "按住 Ctrl 可停靠", "modifier is interpolated");
        assert(hint.includes("\n"), "the hint is two lines (the custom tooltip renders pre-line)");
        assert(!hint.includes("{modifier}"), "no placeholder is left behind");

        // `dockModifier: "none"`（始终停靠）→ 换一句不带修饰键的提示。
        assertEqual(
            dockDragHint("none", translate),
            "拖拽以重排" + "\n" + "拖到目标位置即可停靠",
            "the always-dock variant omits the modifier",
        );
    }

    // ── 幽灵提示的第二行与抓手**同源** ─────────────────────────────
    //
    // 【本用例要钉死的契约】拖拽幽灵必须**永远**给出"按住 {修饰键} 可停靠"：拖拽期间
    // 用户看不到抓手上的悬停提示，这里是唯一能告诉他怎么停靠的地方（用户明确要求）。
    // 两处共用 `dockModifierHint`，因此不可能分叉。
    {
        assertEqual(
            dockModifierHint("primary", translate),
            "按住 Ctrl 可停靠",
            "ghost hint is the modifier line",
        );
        assert(
            dockDragHint("primary", translate).endsWith(dockModifierHint("primary", translate)),
            "the grip hint's second line is exactly the ghost hint",
        );
        assertEqual(
            dockModifierHint("alt", translate),
            "按住 Alt 可停靠",
            "ghost hint follows the configured modifier",
        );
        assertEqual(
            dockModifierHint("none", translate),
            "拖到目标位置即可停靠",
            "no modifier → the always-dock phrasing",
        );
    }
});
