// @vitest-environment jsdom
/*
 * `AppSelect` 的滚轮开关契约。
 *
 * 【为什么需要】滚轮调值是能力层的默认行为，但有些下拉的选项里含**极端值**：
 * 「渲染缓存管理」的容量/超龄预设末尾就是 `0`（不限 / 永不清理），滚轮滚下去会把
 * 4096 MB 直接变成"不限" —— 用户报告为"滚轮一滚就跳到 0"。`wheelAdjust={false}`
 * 让这类下拉把滚轮让给紧邻的数字输入框。
 *
 * 【为什么必须由测试钉住】它是"默认行为被关掉"的反向断言：日后有人删掉这个 prop
 * 或忘了传，功能会静默回到"滚轮改预设"，而人眼回归很难复现。
 */
import { configureStore } from "@reduxjs/toolkit";
import { Theme } from "@radix-ui/themes";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import sessionReducer from "../features/session/sessionSlice";
import { AppSelect } from "./Select";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom 没有 ResizeObserver，而 Radix 的 Tooltip/Popper 在布局 effect 里会构造它。
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(async () => {
    await act(async () => root.unmount());
    document.body.innerHTML = "";
});

const store = configureStore({
    reducer: { session: sessionReducer, keybindings: keybindingsReducer },
});

async function render(node: React.ReactNode) {
    await act(async () => {
        root.render(
            <Provider store={store}>
                <Theme>{node}</Theme>
            </Provider>,
        );
    });
}

/** 等两帧，让帧合并提交器落地（与 `wheelThrottle.test.tsx` 同一手法）。 */
async function nextFrame() {
    await act(async () => {
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    });
}

function wheel(el: Element, deltaY: number, init: WheelEventInit = {}) {
    el.dispatchEvent(new WheelEvent("wheel", { deltaY, bubbles: true, cancelable: true, ...init }));
}

// 预设列表**末项是 0**，与「渲染缓存管理」的两个列表同形。
const OPTIONS = [512, 1024, 2048, 4096, 8192, 0].map((value) => ({
    value: String(value),
    label: String(value),
}));

test("默认行为：滚轮改变选中项（能力层既有契约）", async () => {
    const onValueChange = vi.fn();
    await render(<AppSelect value="512" options={OPTIONS} onValueChange={onValueChange} />);
    const trigger = container.querySelector(".rt-SelectTrigger")!;

    await act(async () => {
        wheel(trigger, 100); // 向下
    });
    await nextFrame();

    expect(onValueChange).toHaveBeenCalled();
    expect(onValueChange.mock.calls.at(-1)?.[0]).toBe("1024");
});

test("wheelAdjust=false：滚轮不改变选中项（不会一滚就落到末项 0）", async () => {
    const onValueChange = vi.fn();
    await render(
        <AppSelect
            value="4096"
            options={OPTIONS}
            onValueChange={onValueChange}
            wheelAdjust={false}
        />,
    );
    const trigger = container.querySelector(".rt-SelectTrigger")!;

    await act(async () => {
        // 连续向下滚：默认行为下会走到末项 0（= 不限）。
        wheel(trigger, 100);
        wheel(trigger, 100);
        wheel(trigger, 100);
    });
    await nextFrame();

    expect(onValueChange).not.toHaveBeenCalled();
});
