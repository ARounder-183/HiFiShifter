// @vitest-environment jsdom
/*
 * 「点面板留白时把焦点收回列表」的判定。
 *
 * 【为什么值得单测】这条规则修的是一个真实故障：点一下工具条背景，焦点落到
 * `<body>`，面板的 keydown 处理器再也收不到事件 —— 输入字母跳转、方向键、Enter
 * 全部按不动。判定本身只看"点中的元素是不是自己管焦点的控件"，可以脱离面板单测；
 * 面板整体（依赖 store / Tauri / 音频上下文）本仓的键盘测试刻意不渲染。
 */

import { afterEach, describe, expect, test } from "vitest";

import { shouldPanelTakeFocus } from "./fileBrowserPanelFocus";

const host = document.createElement("div");
document.body.append(host);

afterEach(() => {
    host.innerHTML = "";
});

/** 建一个元素并返回它（可选带属性）。 */
function el(tag: string, attrs: Record<string, string> = {}): HTMLElement {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    host.append(node);
    return node;
}

describe("shouldPanelTakeFocus", () => {
    test("点留白：收回焦点", () => {
        expect(shouldPanelTakeFocus(el("div"))).toBe(true);
        // 工具条/搜索栏/路径栏的容器本身也是留白。
        expect(shouldPanelTakeFocus(el("div", { class: "px-2 py-1" }))).toBe(true);
        // 列表容器（tabindex=-1）点它的留白正是要聚焦它。
        expect(shouldPanelTakeFocus(el("div", { tabindex: "-1" }))).toBe(true);
    });

    test("点自己管焦点的控件：不抢", () => {
        for (const [tag, attrs] of [
            ["input", {}],
            ["textarea", {}],
            ["select", {}],
            ["button", {}],
            ["a", { href: "#" }],
        ] as const) {
            expect(shouldPanelTakeFocus(el(tag, attrs)), tag).toBe(false);
        }
        expect(shouldPanelTakeFocus(el("div", { contenteditable: "true" }))).toBe(false);
        // 可 Tab 进入的控件
        expect(shouldPanelTakeFocus(el("div", { tabindex: "0" }))).toBe(false);
    });

    test("点列表行 / 菜单项 / 滑杆 / 下拉：不抢", () => {
        expect(shouldPanelTakeFocus(el("div", { role: "option" }))).toBe(false);
        expect(shouldPanelTakeFocus(el("button", { role: "menuitem" }))).toBe(false);
        expect(shouldPanelTakeFocus(el("div", { role: "slider", tabindex: "0" }))).toBe(false);
        expect(shouldPanelTakeFocus(el("button", { role: "combobox" }))).toBe(false);
    });

    test("控件内部的子节点按最近的祖先判定", () => {
        // 搜索框里的放大镜图标是 `<svg>`：点它不该让面板抢走输入框的焦点。
        const button = el("button");
        const icon = document.createElement("span");
        button.append(icon);
        expect(shouldPanelTakeFocus(icon)).toBe(false);

        // 反过来：留白里的普通 span 仍然是留白。
        const blank = el("div");
        const label = document.createElement("span");
        blank.append(label);
        expect(shouldPanelTakeFocus(label)).toBe(true);
    });

    test("没有目标时不抢", () => {
        expect(shouldPanelTakeFocus(null)).toBe(false);
    });
});
