// @vitest-environment jsdom
/*
 * 未绑定的键：全局分发器是**唯一**知道"插件不认识这个键"的地方。
 *
 * 【为什么这条要单独测】让路的入口 B 全靠这一层：命中绑定的路径早就有人守，而"一条
 * 绑定都没命中"此前是直接 `return` —— 插件不消费、宿主也收不到（Windows 的键盘消息
 * 不向父窗口冒泡），于是 Home / End / F2 这些 DAW 里几乎必用的键全部失效。
 *
 * 【为什么"排到微任务再问 defaultPrevented"】事件派发是同步的：微任务检查点在整条
 * 派发链跑完之后才执行，那时 `defaultPrevented` 才是"有没有人认领这个键"的最终答案。
 * 这条契约没有别的守卫 —— 写成同步检查会静默失效（组件本地消费者被绕过）。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterAll, afterEach, beforeEach, expect, test, vi } from "vitest";

const relayMock = vi.hoisted(() => vi.fn());

vi.mock("../../services/hostKeyRelay", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../services/hostKeyRelay")>();
    return { ...actual, relayHostKey: relayMock };
});

import keybindingsReducer from "./keybindingsSlice";
import sessionReducer from "../session/sessionSlice";
import { useKeybindings } from "./useKeybindings";
import { installFocusSurfaceTracking, resetActiveSurfaceForTests } from "../uiFocus/focusSurface";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const disposeFocusSurface = installFocusSurfaceTracking();

const fired: string[] = [];

function Harness() {
    useKeybindings((actionId) => fired.push(actionId));
    return null;
}

const mounted: Array<() => Promise<void>> = [];

afterEach(async () => {
    fired.length = 0;
    resetActiveSurfaceForTests();
    document.body.innerHTML = "";
    while (mounted.length) await mounted.pop()?.();
});

afterAll(() => {
    disposeFocusSurface();
});

beforeEach(() => {
    relayMock.mockReset();
    // 插件形态：没有 bootstrap 的话 `isPluginMode()` 是假，判定会直接返回 null。
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "unbound" } as never;
});

async function mount() {
    const store = configureStore({
        reducer: { keybindings: keybindingsReducer, session: sessionReducer },
    });
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
    });
}

/** 派发一次 keydown，并把排队的微任务跑完（让路判定就发生在那里）。 */
async function press(init: KeyboardEventInit): Promise<void> {
    await act(async () => {
        window.dispatchEvent(
            new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init }),
        );
        // 让 `queueMicrotask` 里的判定执行。
        await Promise.resolve();
        await Promise.resolve();
    });
}

test("未绑定的具名键被交给宿主，且不带动作名", async () => {
    await mount();
    await press({ key: "Home", code: "Home" });
    expect(fired).toEqual([]);
    expect(relayMock).toHaveBeenCalledTimes(1);
    expect(relayMock.mock.calls[0][0]).toEqual({
        action: null,
        stroke: { code: "Home", ctrl: false, shift: false, alt: false, meta: false },
    });
});

test("被组件 preventDefault 的键不算「没人要」", async () => {
    await mount();
    // 模拟一个本地消费者：它在冒泡阶段认领这个键（不 stopPropagation，只 preventDefault）。
    const claim = (event: Event) => event.preventDefault();
    window.addEventListener("keydown", claim);
    try {
        await press({ key: "F2", code: "F2" });
    } finally {
        window.removeEventListener("keydown", claim);
    }
    expect(relayMock).not.toHaveBeenCalled();
});

test("命中绑定的键不走让路", async () => {
    await mount();
    await press({ key: " ", code: "Space" });
    expect(fired).toEqual(["playback.toggle"]);
    expect(relayMock).not.toHaveBeenCalled();
});

test("自动重复只投一次（按住不放是一次意图）", async () => {
    await mount();
    await press({ key: "F2", code: "F2", repeat: true });
    expect(relayMock).not.toHaveBeenCalled();
});

test("可打印字符不投（打字不是让路）", async () => {
    await mount();
    await press({ key: "y", code: "KeyY" });
    expect(relayMock).not.toHaveBeenCalled();
});
