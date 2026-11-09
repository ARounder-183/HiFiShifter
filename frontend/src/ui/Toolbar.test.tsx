// @vitest-environment jsdom
/*
 * 工具栏分隔线的设备像素契约。
 *
 * 【为什么锁 CSS 变量而不是最终 computed width】jsdom 不做布局，也解析不了
 * dpr 对 CSS 像素的映射；真正的几何由浏览器完成。这里锁定的是“工具栏把
 * 唯一正确的换算结果写到根节点、分隔线只消费该变量”这一结构，避免未来
 * 某个调用点又退回 `w-px`。
 */

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test } from "vitest";

import { AppToolbar, AppToolbarGroup } from "./Toolbar";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const originalDpr = window.devicePixelRatio;

function setDpr(value: number): void {
    Object.defineProperty(window, "devicePixelRatio", {
        configurable: true,
        value,
    });
}

afterEach(() => {
    setDpr(originalDpr);
    document.body.innerHTML = "";
});

test("工具栏分隔线的宽度取整数个物理像素（dpr=1.25 → 0.8px）", () => {
    setDpr(1.25);
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    act(() => {
        root.render(
            <AppToolbar
                leading={
                    <AppToolbarGroup>
                        <span>toolbar content</span>
                    </AppToolbarGroup>
                }
            />,
        );
    });

    const toolbar = container.firstElementChild as HTMLElement;
    // 分隔线本身用 CSS 变量继承；变量必须由 AppToolbar 统一写入，
    // 而不是每个 separator 各订阅一次 matchMedia。
    expect(toolbar.style.getPropertyValue("--hs-toolbar-hairline")).toBe("0.8px");

    act(() => root.unmount());
    container.remove();
});
