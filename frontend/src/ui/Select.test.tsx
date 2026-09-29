// @vitest-environment jsdom
/*
 * `AppSelect` 的取值契约。
 *
 * 【主要内容】
 * 1. 默认行为：滚轮改变选中项（能力层既有契约，别被守卫误伤）；
 * 2. **回归**：受控值变成"不在选项里"的值时，**不得**向调用方上报任何变更。
 *
 * 【为什么第 2 条必须钉住 —— 它是"滚轮一滚就跳到 0"的根因】
 * Radix 为表单兼容渲染一个隐藏的原生 `<select>`，并在受控值变化时把它镜像进去
 * （`SelectBubbleInput`：`setValue.call(select, value)` 后派发一个冒泡的 `change`）。
 * 受控值一旦**不在 `<option>` 里**，浏览器会把 `select.value` 归成 `""`，Radix 就把
 * 这个 `""` 原样转发给 `onValueChange` —— 调用方收到一次它从未请求过的空字符串变更。
 *
 * 真实事故：`RenderCacheDialog` 的两个数字框旁曾各配一个预设下拉，用 `"custom"` 表示
 * "当前值不是预设"。滚轮把 4096 调成 4097 ⇒ 下拉受控值变成 `"custom"` ⇒ 触发
 * `onValueChange("")` ⇒ 调用方 `Number("") === 0` ⇒ 占用上限被静默改成"不限"。
 *
 * 受控下拉**只可能**报告自己的选项，因此不属于选项集的值必须丢掉。这条断言就是
 * 那个守卫的回归锁：删掉守卫，它立刻变红。
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

// jsdom 没有 ResizeObserver，而 Radix 的 Popper 在布局 effect 里会构造它。
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
                <Theme>
                    {/*
                     * **必须包在 `<form>` 里**：Radix 只在"触发器位于表单内"时才渲染
                     * 那个隐藏的原生 `<select>`（`isFormControl = !!trigger.closest("form")`），
                     * 而它正是本文件要钉住的缺陷来源。真实场景满足这个条件 ——
                     * `AppDialog` 用 `<form>` 包住对话框正文（见 `Dialog.tsx`）。
                     * 不包的话下面的回归断言会**空过**（没有原生 select，自然没有空值）。
                     */}
                    <form>{node}</form>
                </Theme>
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

/** 预设列表**末项是 0**，与「渲染缓存管理」曾经的两个列表同形。 */
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

    expect(onValueChange.mock.calls.at(-1)?.[0]).toBe("1024");
});

test("【回归】受控值不在选项里时，不得上报空字符串变更", async () => {
    const onValueChange = vi.fn();
    await render(<AppSelect value="4096" options={OPTIONS} onValueChange={onValueChange} />);

    // 真实场景：紧邻的数字框被滚轮改成 4097，下拉的受控值随之变成不在选项里的哨兵。
    await render(<AppSelect value="custom" options={OPTIONS} onValueChange={onValueChange} />);
    await nextFrame();

    expect(onValueChange).not.toHaveBeenCalled();
});

test("【回归】隐藏原生 select 报来的空值同样被丢弃", async () => {
    const onValueChange = vi.fn();
    await render(<AppSelect value="4096" options={OPTIONS} onValueChange={onValueChange} />);
    const native = container.querySelector("select");
    expect(native, "Radix 会为表单兼容渲染隐藏原生 select").toBeTruthy();

    await act(async () => {
        // 直接复现 Radix 的镜像路径：受控值不在 <option> 里 ⇒ 浏览器归为 ""。
        native!.value = "not-an-option";
        native!.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await nextFrame();

    expect(onValueChange).not.toHaveBeenCalled();
});

test("守卫不误伤：选项内的值照常上报", async () => {
    const onValueChange = vi.fn();
    await render(<AppSelect value="4096" options={OPTIONS} onValueChange={onValueChange} />);
    const native = container.querySelector("select")!;

    await act(async () => {
        native.value = "8192";
        native.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await nextFrame();

    expect(onValueChange).toHaveBeenCalled();
    expect(onValueChange.mock.calls.at(-1)?.[0]).toBe("8192");
});
