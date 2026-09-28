// @vitest-environment jsdom
/*
 * 滚轮节流契约。
 *
 * 【为什么必须有】能力层把"滚轮调值"接给了**所有**下拉/数字框/滑块，但没有问过
 * 每个控件的变更代价。实测后果：`吸附/网格设置` 的「网格间距」下拉的 onValueChange
 * 会 dispatch 一个**同步 Tauri 命令**（跑在 UI 线程上，全量重建节拍器响点表），
 * 一次滚轮手势逐格提交 ⇒ 消息泵饥饿 ⇒ 窗口未响应。
 *
 * 因此这里锁两条：
 *   1. **一次手势每帧最多提交一次**（而不是每格一次）；
 *   2. **值一次走到位**（连续 N 格提交的最终值等于走 N 格，而不是反复算同一格）——
 *      第 2 条正是"外部值要等 IPC 往返才更新"时会退化成的错误行为。
 */
import { Theme } from "@radix-ui/themes";
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { keybindingsReducer } from "../features/keybindings";
import { AppNumberField } from "./NumberField";
import { AppSelect } from "./Select";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/*
 * jsdom 没有 ResizeObserver，而 Radix Select 的 Trigger 用它测量宽度。
 * 原语测试不需要真实测量，给一个空实现即可。
 */
if (typeof globalThis.ResizeObserver === "undefined") {
    class ResizeObserverStub {
        observe() {}
        unobserve() {}
        disconnect() {}
    }
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub;
}

/*
 * 原语跑在真实的 Radix Theme 与 Redux 之上：Radix `Select.Trigger` 读 theme context，
 * `AppNumberField` 读"精细调整"键位绑定。
 */
const store = configureStore({
    reducer: { keybindings: keybindingsReducer },
    middleware: (getDefault) => getDefault({ serializableCheck: false }),
});

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

function render(node: React.ReactNode) {
    return act(async () => {
        root.render(
            <Provider store={store}>
                <Theme>{node}</Theme>
            </Provider>,
        );
    });
}

const OPTIONS = ["a", "b", "c", "d", "e"].map((value) => ({ value, label: value }));

/** 等两帧，让帧合并提交器落地。 */
async function nextFrame() {
    await act(async () => {
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    });
}

function wheel(el: Element, deltaY: number) {
    el.dispatchEvent(new WheelEvent("wheel", { deltaY, bubbles: true, cancelable: true }));
}

test("同一帧内的多格滚轮只提交一次，且值一次走到位", async () => {
    const onChange = vi.fn();
    // 从 e 起步，向上滚三格 = e→d→c→b
    await render(<AppSelect value="e" options={OPTIONS} onValueChange={onChange} />);
    const trigger = container.querySelector(".rt-SelectTrigger")!;

    await act(async () => {
        wheel(trigger, -100);
        wheel(trigger, -100);
        wheel(trigger, -100);
    });
    await nextFrame();

    // 三格合并成一次提交，且值一次走到位（而不是反复算同一格）
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("b");
});

test("滚轮被接管（阻止祖先滚动）", async () => {
    await render(<AppSelect value="a" options={OPTIONS} onValueChange={() => {}} />);
    const trigger = container.querySelector(".rt-SelectTrigger")!;
    const event = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
    trigger.dispatchEvent(event);
    // 非被动监听里 preventDefault 才真正生效 —— 这是"滚轮调值不会连带滚动面板"的前提
    expect(event.defaultPrevented).toBe(true);
});

test("只有一项时不接管滚轮（没有可切的目标，不该吃掉滚动）", async () => {
    await render(
        <AppSelect value="a" options={[{ value: "a", label: "a" }]} onValueChange={() => {}} />,
    );
    const trigger = container.querySelector(".rt-SelectTrigger")!;
    const event = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
    trigger.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
});

test("滚轮不越过选项边界", async () => {
    const onChange = vi.fn();
    await render(<AppSelect value="a" options={OPTIONS} onValueChange={onChange} />);
    const trigger = container.querySelector(".rt-SelectTrigger")!;
    await act(async () => {
        // 约定：向上滚 = 上一项。已经在第一项，无路可走。
        wheel(trigger, -100);
    });
    await nextFrame();
    expect(onChange).not.toHaveBeenCalled();
});

test("数字框：同一帧多格只提交一次", async () => {
    const onCommit = vi.fn();
    await render(
        <AppNumberField
            value={10}
            unit="percent" /* 粗调 5 */
            min={0}
            max={100}
            onCommit={onCommit}
        />,
    );
    const wrap = container.querySelector("input")!.parentElement!;

    await act(async () => {
        wheel(wrap, -100);
        wheel(wrap, -100);
    });
    await nextFrame();

    // 两格 × 5 = +10，合并成一次提交
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(20);
});
