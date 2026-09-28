// @vitest-environment jsdom
/*
 * 全局快捷键分发器与**复合控件**的方向键归属。
 *
 * 【为什么必须有】`playback.seekLeft/Right` 默认绑定在左右方向键上，
 * `track.selectUp/Down` 绑在上下方向键上，而分发器运行在 **window 捕获阶段**，
 * 命中后 `preventDefault()` + `stopPropagation()`。于是凡是声明了 ARIA 键盘契约的
 * 复合控件，方向键永远到不了控件自己：焦点停在停靠标签上按 ←/→ 会去 seek，
 * 而不是切标签 —— 标签条的键盘模型（`DockTabBar` 的 roving tabIndex + 方向键）
 * 因此**完全失效**，而它的单元测试全绿。
 *
 * 这是浏览器实测发现的：在真实应用里给标签派发 `ArrowLeft`，事件在 window 捕获
 * 阶段就被吞掉，标签自己的处理器收不到。所以这里锁定的契约是
 * **"焦点在复合控件里时，全局分发器必须让路"**，而不只是"标签处理器写对了"。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, expect, test } from "vitest";

import keybindingsReducer from "./keybindingsSlice";
import sessionReducer from "../session/sessionSlice";
import { useKeybindings } from "./useKeybindings";
import type { ActionId } from "./types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function createTestStore() {
    return configureStore({
        reducer: { keybindings: keybindingsReducer, session: sessionReducer },
    });
}

const fired: ActionId[] = [];

function Harness() {
    useKeybindings((actionId) => fired.push(actionId));
    return null;
}

const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    fired.length = 0;
    while (mounted.length) await mounted.pop()?.();
});

async function mount() {
    const store = createTestStore();
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <Harness />
            </Provider>,
        );
    });
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    return { store, host };
}

/** 在 `target` 上派发一个真实的（可冒泡、可取消的）keydown。 */
function press(target: EventTarget, key: string): { defaultPrevented: boolean } {
    const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
    target.dispatchEvent(event);
    return { defaultPrevented: event.defaultPrevented };
}

/** 建一个 `role="tablist"` + 可聚焦的 `role="tab"`，与停靠标签条同形。 */
function buildTablist(): { list: HTMLElement; tab: HTMLElement } {
    const list = document.createElement("div");
    list.setAttribute("role", "tablist");
    const tab = document.createElement("div");
    tab.setAttribute("role", "tab");
    tab.tabIndex = 0;
    list.append(tab);
    document.body.append(list);
    mounted.push(async () => {
        list.remove();
    });
    return { list, tab };
}

test("焦点在标签条里时，方向键不被全局分发器吞掉", async () => {
    await mount();
    const { tab } = buildTablist();
    tab.focus();

    const result = press(tab, "ArrowLeft");

    // 全局绑定（playback.seekLeft）不得触发……
    expect(fired).toEqual([]);
    // ……而且事件必须保持"未被消费"，否则标签自己的处理器也收不到
    // （分发器命中时正是靠 preventDefault + stopPropagation 让控件收不到）。
    expect(result.defaultPrevented).toBe(false);
});

test("焦点不在复合控件里时，方向键仍然走全局绑定", async () => {
    await mount();
    buildTablist(); // 存在但未聚焦

    press(document.body, "ArrowLeft");

    // 这条是上一条的对照：让路只在"焦点在复合控件内"时发生，
    // 否则时间轴的 ←/→ seek 就被整体破坏了。
    expect(fired).toEqual(["playback.seekLeft"]);
});

test("菜单里的方向键同样归菜单自己", async () => {
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    const item = document.createElement("button");
    item.setAttribute("role", "menuitem");
    item.tabIndex = 0;
    menu.append(item);
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });
    item.focus();

    expect(press(item, "ArrowDown").defaultPrevented).toBe(false);
    expect(fired).toEqual([]);
});

test("非方向键的全局快捷键在标签条里照常生效", async () => {
    await mount();
    const { tab } = buildTablist();
    tab.focus();

    // 让路只针对复合控件拥有的那一组键（方向键 + Home/End/PageUp/PageDown），
    // 其余全局快捷键不受影响 —— `k` 默认绑定在节拍器上。
    press(tab, "k");
    expect(fired).toEqual(["playback.metronome"]);
});
