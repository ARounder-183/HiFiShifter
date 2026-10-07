// @vitest-environment jsdom
/*
 * 时间轴轨道区域右键菜单的快捷键提示。
 *
 * 【为什么必须有】这张菜单改用共享原语 `AppContextMenu` 时只填了 `label`/`onSelect`，
 * 把 `shortcut` 整条信息丢了 —— 于是同一张时间轴上，剪辑右键菜单显示 `Ctrl+V`，
 * 紧挨着的轨道区域菜单什么都不显示。共享原语本来就支持这一列（`AppMenuItemSpec.shortcut`），
 * 缺的是取绑定那一步（`ui/useMenuShortcut`）。
 *
 * 【为什么期望值由 `formatKeybinding` 算出而不是写 "Ctrl+V" 字面量】主修饰键在 macOS
 * 上渲染成 `⌘`。本测试要钉的契约是"菜单显示的就是当前绑定的文本"，而不是某个平台的
 * 具体字形 —— 写死字面量会让它在 macOS 上失败，却什么也没多证明。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import keybindingsReducer, {
    formatKeybinding,
    setKeybindings,
} from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { TrackAreaContextMenu } from "./TrackAreaContextMenu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

type PasteOrSplit = "clip.paste" | "clip.split";

/** 渲染菜单，返回每条菜单项的文案与生效绑定。 */
async function mountMenu(options?: {
    bindings?: Partial<Record<PasteOrSplit, Keybinding>>;
    canPaste?: boolean;
    canSplit?: boolean;
}): Promise<{
    items: string[];
    disabled: boolean[];
    binding: (id: PasteOrSplit) => Keybinding;
}> {
    const store = configureStore({ reducer: { keybindings: keybindingsReducer } });
    const bindings: Record<PasteOrSplit, Keybinding> = {
        "clip.paste": { key: "v", ctrl: true },
        "clip.split": { key: "s" },
        ...options?.bindings,
    };
    await act(async () => {
        store.dispatch(
            setKeybindings({ actionId: "clip.paste", bindings: [bindings["clip.paste"]] }),
        );
        store.dispatch(
            setKeybindings({ actionId: "clip.split", bindings: [bindings["clip.split"]] }),
        );
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <TrackAreaContextMenu
                        x={10}
                        y={10}
                        canPaste={options?.canPaste ?? true}
                        canSplit={options?.canSplit ?? true}
                        canCloseGaps
                        onPaste={() => undefined}
                        onSplit={() => undefined}
                        onCloseGaps={() => undefined}
                        onClose={() => undefined}
                    />
                </I18nProvider>
            </Provider>,
        );
    });
    const entries = [...document.body.querySelectorAll<HTMLElement>('[role="menuitem"]')];
    return {
        items: entries.map((item) => item.textContent ?? ""),
        disabled: entries.map((item) => item.hasAttribute("disabled")),
        binding: (id) => bindings[id],
    };
}

test("有绑定的项显示当前生效的快捷键文本", async () => {
    const { items, binding } = await mountMenu();
    expect(items).toHaveLength(3);
    expect(items[0]).toContain(formatKeybinding(binding("clip.paste"), ""));
    expect(items[1]).toContain(formatKeybinding(binding("clip.split"), ""));
});

test("没有绑定的「关闭间隙」不显示快捷键列", async () => {
    const { items } = await mountMenu();
    const closeGaps = items[2];
    expect(closeGaps).toContain("Close Gaps");
    expect(closeGaps, "未绑定的动作不应出现快捷键文本").not.toMatch(/Ctrl|Shift|Alt|⌘/);
});

test("快捷键随用户改绑实时变化（不是写死的字面量）", async () => {
    const rebind: Keybinding = { key: "b", ctrl: true, shift: true };
    const { items } = await mountMenu({ bindings: { "clip.paste": rebind } });
    expect(items[0]).toContain(formatKeybinding(rebind, ""));
    // 默认的 Ctrl+V 不应再出现（否则说明渲染的是常量而不是注册表）。
    expect(items[0]).not.toContain(formatKeybinding({ key: "v", ctrl: true }, ""));
});

test("禁用项仍然显示快捷键（用户靠它知道这个操作本来有键）", async () => {
    const { items, disabled, binding } = await mountMenu({ canPaste: false, canSplit: false });
    expect(disabled[0]).toBe(true);
    expect(items[0]).toContain(formatKeybinding(binding("clip.paste"), ""));
});

test("插件保留空白菜单，按真实能力开放分割，不误放开App粘贴或关闭间隙", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "blank-menu", clipSplitting: true };
    const { items, disabled } = await mountMenu();
    expect(items).toHaveLength(3);
    expect(disabled).toEqual([true, false, true]);
});

test("缺分割能力的插件仍显示空白菜单，但不发送未实现的几何命令", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "blank-menu" };
    const { items, disabled } = await mountMenu();
    expect(items).toHaveLength(3);
    expect(disabled).toEqual([true, true, true]);
});

test("原生剪贴板接通后空白区粘贴启用，不误开放关闭间隙", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "blank-menu", clipSplitting: true, clipClipboard: true };
    const { disabled } = await mountMenu();
    expect(disabled).toEqual([false, false, true]);
});
