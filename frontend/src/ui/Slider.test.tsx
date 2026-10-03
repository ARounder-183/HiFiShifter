// @vitest-environment jsdom
/*
 * 滑块包装盒的契约：把 Radix 滑块头的**装饰溢出**收在自己的盒子里。
 *
 * 【为什么必须有】Radix 的滑块头是绝对定位的装饰：可见部分比轨道高（size 2 是
 * 16 / 8），另有放大命中区（`::before`，滑块头 × 3）与焦点环。这份溢出**不会**
 * 被任何内在尺寸计算算进去（CSS 的 intrinsic sizing 从不含 scrollable overflow），
 * 却会被祖先的滚动容器算成"可滚动" —— 表现是每个含滑块的对话框都恒定挂着一条
 * **滚不动**的竖直滚动条。
 *
 * 用户报告的正是这个：参数编辑器右键菜单里"所有功能的弹窗都出现了竖直滚动条"。
 * 实测 8 个编辑对话框里，**有滑块的 7 个全部如此**，唯一没有滑块的「添加颤音」没有 ——
 * 对照关系与内容多少无关。
 *
 * 【为什么钉契约而不是现象】jsdom 量不出 `scrollHeight`（没有布局引擎），因此这里
 * 断言的是两件可静态核对的事：包装盒带 `.hs-slider-box`（`index.css` 里的裁切规则
 * 挂在它上面）、以及按密度算出的盒高足够容下可见滑块头。现象由浏览器里实测确认。
 * 横向余量（可见滑块头在 0% / 100% 会越过轨道两端 2px）与内嵌焦点环同样在
 * `index.css`，由 `designSystemGates.test.ts` 门禁钉住 —— jsdom 读不到样式表。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import { AppDensityProvider } from "./density";
import { AppSlider } from "./Slider";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Radix 的滑块用 `useSize` 观察轨道宽度，而 jsdom 没有 ResizeObserver。
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

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

function renderSlider(density: "form" | "compact") {
    const store = configureStore({ reducer: { keybindings: keybindingsReducer } });
    act(() => {
        root.render(
            <Provider store={store}>
                <AppDensityProvider value={density}>
                    <AppSlider
                        value={0}
                        unit="percent"
                        min={0}
                        max={100}
                        ariaLabel="test slider"
                        onChange={() => undefined}
                    />
                </AppDensityProvider>
            </Provider>,
        );
    });
    const box = container.querySelector<HTMLElement>(".hs-slider-box");
    expect(box, "滑块包装盒缺少 .hs-slider-box（裁切规则挂在它上面）").toBeTruthy();
    return box!;
}

test("包装盒带 hs-slider-box，且盒高容得下 size 2 的可见滑块头（16px）", () => {
    const box = renderSlider("form");
    // 16 = 滑块头 12 + 0.5 × 轨道 8（Radix 的尺寸公式，见 Slider.tsx 的推导注释）。
    expect(box.style.minHeight).toBe("16px");
    expect(box.className).toContain("hs-slider-box");
});

test("紧凑密度下用 size 1 的盒高（13px），不无谓撑高工具条", () => {
    const box = renderSlider("compact");
    // 13 = 滑块头 10 + 0.5 × 轨道 6。
    expect(box.style.minHeight).toBe("13px");
});

test("包装盒仍然铺满可用宽度（收住溢出不等于改变布局）", () => {
    const box = renderSlider("form");
    expect(box.className).toContain("flex-1");
    expect(box.className).toContain("items-center");
});
