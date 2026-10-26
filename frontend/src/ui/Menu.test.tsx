// @vitest-environment jsdom
/*
 * 菜单项的"为什么点不了"必须**看得见**。
 *
 * 【为什么这条必须有】`AppTooltipProvider` 在菜单弹出期间全局抑制气泡
 * （`components/AppTooltip.tsx`，见 `docs/i18n/style-guide.md` §3.5）。于是挂在菜单项
 * 上的 `data-tooltip` **一次都不会显示** —— 而 `dawControlledReason()` 与整族"插件里
 * 为什么不能做这件事"的解释正是挂在那里的。它们从来没被渲染过，没人发现。
 *
 * 修法是把通道换成正立文本：禁用项的第二行弱化小字。本用例钉住三件事：
 * 1. 禁用 + 有理由 ⇒ 理由可见；
 * 2. 可用项 ⇒ 不渲染第二行（否则每张菜单都变高）；
 * 3. 理由**不再**走 `data-tooltip`（那条通道是死的，留着只会让下一个人以为它在工作）。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test } from "vitest";

import { AppContextMenu, type AppMenuItemSpec } from "./Menu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

function mountMenu(items: AppMenuItemSpec[]): void {
    act(() => {
        root.render(
            <AppContextMenu
                x={10}
                y={10}
                items={items}
                onClose={() => {
                    /* 本用例不触发关闭 */
                }}
            />,
        );
    });
}

function menuItems(): HTMLButtonElement[] {
    return [...document.body.querySelectorAll<HTMLButtonElement>(".hs-menu__item")];
}

test("禁用项把理由渲染成第二行，而不是挂在死掉的 tooltip 通道上", () => {
    mountMenu([
        { key: "a", label: "Delete track", disabled: true, description: "The host owns this" },
    ]);
    const [item] = menuItems();
    expect(item.disabled).toBe(true);
    expect(item.querySelector(".hs-menu__description")?.textContent).toBe("The host owns this");
    // 通道换过之后不能再往 data-tooltip 上写：那条路在菜单里永远不显示。
    expect(item.getAttribute("data-tooltip")).toBeNull();
});

test("可用项不渲染第二行（理由只在点不了的时候才有意义）", () => {
    mountMenu([{ key: "a", label: "Delete track", description: "The host owns this" }]);
    const [item] = menuItems();
    expect(item.disabled).toBe(false);
    expect(item.querySelector(".hs-menu__description")).toBeNull();
});

test("禁用但没有理由时不留空行", () => {
    mountMenu([{ key: "a", label: "Delete track", disabled: true }]);
    const [item] = menuItems();
    expect(item.querySelector(".hs-menu__description")).toBeNull();
});
