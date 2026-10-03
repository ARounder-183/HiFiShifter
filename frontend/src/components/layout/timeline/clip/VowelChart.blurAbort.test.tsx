// @vitest-environment jsdom
/**
 * VowelChart 的失焦守卫契约。
 *
 * 回归点：`onEnd` 同时挂在 window 的 pointerup/pointercancel 上（窗口中任意一次
 * 松手都会触发）。若在 onEnd 里注销 `registerDragAbort`，那么"打开浮窗的那次
 * 点击"就会把失焦守卫摘掉且 effect 依赖不变、不再重注册——之后一次被 blur 打断
 * 的拖拽会让 draggingRef 滞留为 true，切回后普通鼠标移动持续改写共振峰。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { VowelChart } from "./VowelChart";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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

function svgRect(): DOMRect {
    return {
        left: 0,
        top: 0,
        right: 420,
        bottom: 320,
        width: 420,
        height: 320,
        x: 0,
        y: 0,
        toJSON: () => ({}),
    } as DOMRect;
}

test("窗口内任意一次 pointerup 之后，blur 仍能终止后续 vowel-chart 拖拽", () => {
    const onChange = vi.fn();
    act(() => {
        root.render(<VowelChart targetF1Hz={500} targetF2Hz={1500} onChange={onChange} />);
    });
    const svg = container.querySelector("svg");
    if (svg === null) throw new Error("svg not rendered");
    svg.getBoundingClientRect = svgRect;

    // 打开浮窗的那次点击：窗口内任意一次松手（旧实现会在此注销失焦守卫）。
    act(() => {
        window.dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    });

    // 开始一次图表拖拽。
    act(() => {
        svg.dispatchEvent(
            new PointerEvent("pointerdown", {
                bubbles: true,
                cancelable: true,
                button: 0,
                clientX: 200,
                clientY: 200,
            }),
        );
    });
    onChange.mockClear();

    // 切屏（失焦）：守卫必须复位拖拽布尔。
    act(() => {
        window.dispatchEvent(new Event("blur"));
    });

    // 切回后普通鼠标移动不得继续改写共振峰。
    act(() => {
        window.dispatchEvent(
            new PointerEvent("pointermove", { bubbles: true, clientX: 260, clientY: 260 }),
        );
    });

    expect(onChange).not.toHaveBeenCalled();
});
