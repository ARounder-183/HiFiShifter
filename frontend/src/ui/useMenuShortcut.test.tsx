// @vitest-environment jsdom
/*
 * 菜单快捷键文案的唯一取法（`useMenuShortcut`）。
 *
 * 【为什么必须有】这三行（读绑定 → 格式化）此前在 `EditContextMenu`、`TrackList`
 * 与 `MenuBar` 里各写了一遍，而时间轴那几个改用 `AppContextMenu` 的菜单干脆**没有**
 * 取 —— 同一张时间轴上，剪辑菜单有快捷键、轨道区域菜单没有。收敛成一个 hook 之后，
 * 这里锁住它对外承诺的两条契约：
 *
 * 1. 绑定了就给出可读文本（且跟随 store，用户改绑后自动变化）；
 * 2. 未绑定（`__none__`）返回 `undefined` —— 调用方据此**不渲染**那一列，
 *    而不是显示 `—`（那会被误读成"这个键就是短横线"）。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, expect, test } from "vitest";

import keybindingsReducer, { setKeybinding } from "../features/keybindings/keybindingsSlice";
import { useMenuShortcut } from "./useMenuShortcut";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;

afterEach(() => {
    host?.remove();
    host = null;
});

/** 把 hook 的返回值渲染成一行文本，便于断言（`undefined` 渲染成空）。 */
function renderShortcut(actionId: "clip.paste" | "clip.split" | "edit.deselect") {
    const store = configureStore({ reducer: { keybindings: keybindingsReducer } });
    host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);

    function Probe() {
        const shortcut = useMenuShortcut(actionId);
        return <span data-testid="out">{shortcut ?? ""}</span>;
    }

    act(() => {
        root.render(
            <Provider store={store}>
                <Probe />
            </Provider>,
        );
    });

    return {
        read: () => host?.querySelector('[data-testid="out"]')?.textContent ?? "",
        rebind: (binding: Parameters<typeof setKeybinding>[0]["binding"]) =>
            act(() => {
                store.dispatch(setKeybinding({ actionId, binding }));
            }),
        unmount: () => act(() => root.unmount()),
    };
}

test("绑定了就给可读文本，且跟随用户改绑", () => {
    // clip.paste 的默认绑定是 Ctrl+V；这里只断言"非空且含 V"，具体修饰键文本
    // 由 formatKeybinding 决定（macOS 上是 ⌘），不在此处重复它的规则。
    const probe = renderShortcut("clip.paste");
    expect(probe.read()).toContain("V");

    probe.rebind({ key: "b", ctrl: true, shift: true });
    expect(probe.read()).toContain("B");
    probe.unmount();
});

test("未绑定返回空（调用方不渲染快捷键列，而不是显示占位符）", () => {
    // edit.deselect 的默认绑定就是 `__none__`。
    const probe = renderShortcut("edit.deselect");
    expect(probe.read()).toBe("");

    probe.rebind({ key: "v", ctrl: true });
    expect(probe.read()).toContain("V");
    probe.unmount();
});
