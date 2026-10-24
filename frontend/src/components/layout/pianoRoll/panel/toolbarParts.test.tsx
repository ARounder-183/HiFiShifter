// @vitest-environment jsdom
/*
 * `ToolbarSlot` 的显隐契约 —— 渲染侧与测量侧共用的那一个判据。
 *
 * 【为什么值得单独钉】工具栏的层级判据是"在真实行上逐级试显隐"（`toolbarGauge`），
 * 它靠这两个属性推断每个槽位的区间。属性与 `display` 一旦对不上，测量试的就不是
 * 用户看到的那一页 —— 而这类分叉在浏览器里表现为"怎么调都有 10px 的重叠"。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { NEVER_HIDE_TIER } from "../toolbarOverflow";
import { ToolbarSlot } from "./toolbarParts";

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

function renderSlot(props: { hideAt?: number; showAt?: number; tier: number }): HTMLElement {
    act(() => {
        root.render(
            <ToolbarSlot {...props}>
                <button type="button">x</button>
            </ToolbarSlot>,
        );
    });
    return container.firstElementChild as HTMLElement;
}

describe("ToolbarSlot", () => {
    it("hideAt：达到该级隐藏，之前显示（display: contents 不参与布局）", () => {
        expect(renderSlot({ hideAt: 3, tier: 2 }).style.display).toBe("contents");
        expect(renderSlot({ hideAt: 3, tier: 3 }).style.display).toBe("none");
        expect(renderSlot({ hideAt: 3, tier: 9 }).style.display).toBe("none");
    });

    it("showAt：阈值以下隐藏、达到后显示（「⋯」按钮靠它常挂载）", () => {
        expect(renderSlot({ showAt: 10, tier: 9 }).style.display).toBe("none");
        expect(renderSlot({ showAt: 10, tier: 10 }).style.display).toBe("contents");
        expect(renderSlot({ showAt: 10, tier: 11 }).style.display).toBe("contents");
    });

    it("两个属性同时写：只在区间内显示", () => {
        expect(renderSlot({ showAt: 5, hideAt: 8, tier: 4 }).style.display).toBe("none");
        expect(renderSlot({ showAt: 5, hideAt: 8, tier: 5 }).style.display).toBe("contents");
        expect(renderSlot({ showAt: 5, hideAt: 8, tier: 7 }).style.display).toBe("contents");
        expect(renderSlot({ showAt: 5, hideAt: 8, tier: 8 }).style.display).toBe("none");
    });

    it("NEVER_HIDE_TIER（负数）：任何区间都不隐藏", () => {
        expect(renderSlot({ hideAt: 1, tier: NEVER_HIDE_TIER }).style.display).toBe("contents");
        expect(renderSlot({ showAt: 10, tier: NEVER_HIDE_TIER }).style.display).toBe("contents");
    });

    it("把区间写进 data-* 供测量读取（不写 = 不参与让位）", () => {
        const hidden = renderSlot({ hideAt: 3, tier: 0 });
        expect(hidden.dataset.hideAt).toBe("3");
        expect(hidden.dataset.showAt).toBeUndefined();

        const shown = renderSlot({ showAt: 10, tier: 0 });
        expect(shown.dataset.showAt).toBe("10");
        expect(shown.dataset.hideAt).toBeUndefined();
    });
});
