// @vitest-environment jsdom
/*
 * `AppNumberField` 的步进契约（用户报告「滚轮一滚就跳到 0」时对照的"其他输入框"）。
 *
 * 指针在数字输入框上滚轮 / 按方向键 → 按**单位语义**走一格，按住精细调整修饰键
 * 走该单位的 fine 一档；**不得**跳到 0 或任何别处。
 *
 * 另外钉住"原生 `step` 不能是取值约束"这一条：用户报告渲染缓存里
 * 「音频块大小下限」一按保存就弹"请输入一个有效的值。最接近的两个有效值为 0 和 16。"
 * —— 根因是步长（16 KB）被写进了原生 `step`，而默认值 4 不是它的整数倍，
 * 浏览器的表单校验因此拦下提交。
 */
import { configureStore } from "@reduxjs/toolkit";
import { Theme } from "@radix-ui/themes";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import sessionReducer from "../features/session/sessionSlice";
import { AppNumberField } from "./NumberField";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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

async function nextFrame() {
    await act(async () => {
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    });
}

/** 输入框所在的可滚轮容器（`AppNumberField` 把非被动 wheel 监听挂在包裹层上）。 */
function wheelOnInput(input: HTMLInputElement, deltaY: number, init: WheelEventInit = {}) {
    const wrapper = input.parentElement!;
    wrapper.dispatchEvent(
        new WheelEvent("wheel", { deltaY, bubbles: true, cancelable: true, ...init }),
    );
}

function input(container: HTMLElement): HTMLInputElement {
    return container.querySelector("input")!;
}

test("整数单位：滚轮走一格（±1），不会跳到 0", async () => {
    const onCommit = vi.fn();
    await render(<AppNumberField value={4096} unit="integer" min={0} onCommit={onCommit} />);

    await act(async () => {
        wheelOnInput(input(container), -100);
    });
    await nextFrame();
    // 只断言提交值：组件按"外部值是真值"设计，测试替身不回灌 value 时会就地重新
    // 播种（显示回退），那是正确行为；真实调用方（对话框草稿）会回灌。
    expect(onCommit.mock.calls.at(-1)?.[0]).toBe(4097);
});

test("精细调整修饰键（Ctrl）：整数单位仍是 ±1", async () => {
    const onCommit = vi.fn();
    await render(<AppNumberField value={90} unit="integer" min={0} onCommit={onCommit} />);

    await act(async () => {
        wheelOnInput(input(container), -100, { ctrlKey: true });
    });
    await nextFrame();

    expect(onCommit.mock.calls.at(-1)?.[0]).toBe(91);
});

test("下界 0：继续向下滚停在 0，不会变成负数", async () => {
    const onCommit = vi.fn();
    await render(<AppNumberField value={1} unit="integer" min={0} onCommit={onCommit} />);

    await act(async () => {
        wheelOnInput(input(container), 100);
        wheelOnInput(input(container), 100);
    });
    await nextFrame();

    expect(onCommit.mock.calls.at(-1)?.[0]).toBe(0);
});

test("原生 step 是 any：步长不得变成浏览器的取值约束", async () => {
    /*
     * 【回归】`step={spec.coarse}` 会让"4 KB（默认值）"在 step=16 下成为
     * `stepMismatch`，于是提交对话框时浏览器弹气泡并拦下提交。步长是 UI 手感，
     * 不是合法性 —— 因此原生 step 必须是 `any`。
     */
    await render(<AppNumberField value={4} unit="kilobytes" min={0} onCommit={vi.fn()} />);
    const el = input(container);
    expect(el.getAttribute("step")).toBe("any");
    // jsdom 不实现约束校验（validity 恒为 true），因此再钉一条可静态核对的：
    // `min` 仍用于夹紧，但不再与 step 组合成 mismatch。
    expect(el.validity.stepMismatch).toBe(false);
});

test("方向键按单位语义走一格（不是原生的 ±1）", async () => {
    const onCommit = vi.fn();
    await render(<AppNumberField value={4096} unit="megabytes" min={0} onCommit={onCommit} />);

    await act(async () => {
        input(container).dispatchEvent(
            new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true, cancelable: true }),
        );
    });
    await nextFrame();
    expect(onCommit.mock.calls.at(-1)?.[0]).toBe(5120);
});

test("方向键 + 精细调整修饰键走 fine 一档", async () => {
    const onCommit = vi.fn();
    await render(<AppNumberField value={4096} unit="megabytes" min={0} onCommit={onCommit} />);

    await act(async () => {
        input(container).dispatchEvent(
            new KeyboardEvent("keydown", {
                key: "ArrowDown",
                ctrlKey: true,
                bubbles: true,
                cancelable: true,
            }),
        );
    });
    await nextFrame();
    expect(onCommit.mock.calls.at(-1)?.[0]).toBe(3968);
});
