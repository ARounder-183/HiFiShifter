// @vitest-environment jsdom
/*
 * 全局分发器把**按键描述**一并交给应用层。
 *
 * 【为什么必须有】让路判定的全部输入就是"用户到底按了哪个键"。这一层若漏传（或把
 * Space 传成 `" "`），前端会认为"这个键不在表里"而**静默不接手** —— 用户看到的是
 * "按 Space 还是弹那句没出路的话"。而分发器本身的其它测试全绿。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterAll, afterEach, expect, test } from "vitest";

import keybindingsReducer from "./keybindingsSlice";
import sessionReducer from "../session/sessionSlice";
import { useKeybindings } from "./useKeybindings";
import type { ActionId } from "./types";
import type { HostKeyStroke } from "../../services/hostKeyRelay";
import { installFocusSurfaceTracking, resetActiveSurfaceForTests } from "../uiFocus/focusSurface";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const disposeFocusSurface = installFocusSurfaceTracking();

const fired: Array<{ action: ActionId; stroke: HostKeyStroke }> = [];

function Harness() {
    useKeybindings((actionId, stroke) => fired.push({ action: actionId, stroke }));
    return null;
}

const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    fired.length = 0;
    resetActiveSurfaceForTests();
    while (mounted.length) await mounted.pop()?.();
});

afterAll(() => {
    disposeFocusSurface();
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
        host.remove();
    });
    return { store };
}

function press(init: KeyboardEventInit): void {
    window.dispatchEvent(
        new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init }),
    );
}

test("Space 带上 DOM code（与键盘布局无关的物理键名），并带四个修饰位", async () => {
    await mount();
    press({ key: " ", code: "Space" });
    expect(fired).toEqual([
        {
            action: "playback.toggle",
            stroke: { code: "Space", ctrl: false, shift: false, alt: false, meta: false },
        },
    ]);
});

test("Enter 的 code 是 Enter（不是 NumpadEnter）", async () => {
    await mount();
    press({ key: "Enter", code: "Enter" });
    expect(fired.map((item) => item.stroke.code)).toEqual(["Enter"]);
});

test("修饰键如实上报（让路判定据此拒绝）", async () => {
    await mount();
    press({ key: "z", code: "KeyZ", ctrlKey: true });
    expect(fired).toHaveLength(1);
    // 具体命中哪个动作由焦点路由裁决（这里不钉），本用例只关心按键描述被如实带上。
    expect(fired[0].action).toMatch(/^(edit\.|pianoRoll\.)/);
    expect(fired[0].stroke).toEqual({
        code: "KeyZ",
        ctrl: true,
        shift: false,
        alt: false,
        meta: false,
    });
});

test("编辑 op 的路由分支也带按键描述", async () => {
    await mount();
    press({ key: "c", code: "KeyC", ctrlKey: true });
    expect(fired).toHaveLength(1);
    expect(fired[0].action).toMatch(/^(clip|pianoRoll)\.copy$/);
    expect(fired[0].stroke.code).toBe("KeyC");
    expect(fired[0].stroke.ctrl).toBe(true);
});
