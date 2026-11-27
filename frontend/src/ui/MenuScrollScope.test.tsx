/**
 * MenuScrollScope 的渲染契约。
 *
 * 【为什么这几条必须有】提示行是"删掉不会有任何报错"的东西：漏渲染一层，菜单
 * 照常打开、照常能滚，只是**没有**任何"还能往下翻"的提示。反过来，内边距那一层
 * 漏了，菜单会失去左右内边距 —— 而这两个失败都只靠眼睛才能发现。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test } from "vitest";

import { MenuScrollScope } from "./MenuScrollScope";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let unmount: (() => void) | null = null;

afterEach(async () => {
    if (unmount) await act(async () => unmount?.());
    host?.remove();
    host = null;
    unmount = null;
});

async function render(ui: React.ReactNode): Promise<HTMLDivElement> {
    host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(ui));
    unmount = () => root.unmount();
    return host;
}

test("渲染两条提示行 + 一层内容容器", async () => {
    const container = await render(
        <div className="hs-menu">
            <MenuScrollScope>
                <span data-testid="child">x</span>
            </MenuScrollScope>
        </div>,
    );
    const shell = container.querySelector(".hs-menu")!;
    expect(shell.querySelectorAll(":scope > .hs-menu__scroll-hint").length).toBe(2);
    expect(shell.querySelector(":scope > .hs-menu__content")).not.toBeNull();
    expect(shell.querySelector("[data-testid='child']")?.textContent).toBe("x");
});

test("提示行不可聚焦、对读屏隐藏（它们不是菜单项）", async () => {
    const container = await render(
        <div className="hs-menu">
            <MenuScrollScope>content</MenuScrollScope>
        </div>,
    );
    for (const hint of container.querySelectorAll(".hs-menu__scroll-hint")) {
        expect(hint.getAttribute("aria-hidden")).toBe("true");
        expect(hint.getAttribute("tabindex")).toBeNull();
        // role="menuitem" 会把键盘导航的项数算错，且"滚一页"不是一条命令。
        expect(hint.getAttribute("role")).toBeNull();
    }
});

test("scrollable=false 时不渲染提示行，children 照常渲染", async () => {
    /*
     * 【children 无条件渲染】下面任何一步失败（找不到滚动容器、ResizeObserver
     * 不存在、令牌读不到），退化的都必须是"提示"，而不是"内容不见了"。
     */
    const container = await render(
        <div className="hs-menu hs-menu--no-scroll">
            <MenuScrollScope scrollable={false} contentClassName="flex flex-col">
                <span data-testid="child">x</span>
            </MenuScrollScope>
        </div>,
    );
    expect(container.querySelectorAll(".hs-menu__scroll-hint").length).toBe(0);
    const content = container.querySelector(".hs-menu__content")!;
    expect(content.className).toContain("flex");
    expect(container.querySelector("[data-testid='child']")?.textContent).toBe("x");
});

test("不可滚的壳里，提示行不会被点亮", async () => {
    /*
     * \`--no-scroll\` 的壳是 \`overflow: visible\`：MenuScrollScope 找不到滚动容器，
     * 整段逻辑关闭。少了这条判断，菜单会去量**面板**的滚动区，于是"这张菜单不滚"
     * 被误判成"它在滚整块面板"，提示行随面板滚动亮起。
     */
    const container = await render(
        <div className="hs-menu hs-menu--no-scroll">
            <MenuScrollScope>content</MenuScrollScope>
        </div>,
    );
    for (const hint of container.querySelectorAll(".hs-menu__scroll-hint")) {
        expect(hint.getAttribute("data-visible")).toBeNull();
    }
});
