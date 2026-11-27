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

/*
 * 按下期间冻结提示行可见性。
 *
 * 【为什么这条必须有】提示行滚到尽头会隐藏（data-visible 被清掉 → pointer-events:
 * none）。若手指还按着，元素变成不可命中会让 pointerup 落到别处 —— 于是"按住快滚
 * 到底"会留下一个永远收不到抬起的按下态，可见性从此**永久冻结**。
 *
 * 【怎么在没有布局的环境里测】jsdom 的 clientHeight / scrollHeight 恒为 0，这里
 * 直接把三个布局数字钉在壳上（scrollHeight 300 / clientHeight 100 → 可滚 200），
 * 然后手动派发 scroll 事件驱动判据。
 */
test("按下期间滚到边界不清提示行，抬起后补写", async () => {
    const container = await render(
        /*
         * 【为什么要写内联 overflowY】用例不加载 index.css，jsdom 的
         * getComputedStyle 对未声明的属性返回空串 —— 而"这张菜单可不可滚"的判据
         * 正是 overflow-y ∈ {auto, scroll}。不声明它，MenuScrollScope 会认为找不到
         * 滚动容器而整段关闭（那也正是 --no-scroll 壳的行为）。
         */
        <div className="hs-menu" style={{ overflowY: "auto" }}>
            <MenuScrollScope>content</MenuScrollScope>
        </div>,
    );
    const shell = container.querySelector<HTMLElement>(".hs-menu")!;
    const down = container.querySelector<HTMLElement>(".hs-menu__scroll-hint--down")!;

    Object.defineProperty(shell, "scrollHeight", { configurable: true, value: 300 });
    Object.defineProperty(shell, "clientHeight", { configurable: true, value: 100 });
    Object.defineProperty(shell, "scrollTop", {
        configurable: true,
        writable: true,
        value: 0,
    });

    const fireScroll = async () => {
        await act(async () => {
            shell.dispatchEvent(new Event("scroll"));
        });
    };

    // 在顶端：只有向下提示。
    await fireScroll();
    expect(down.getAttribute("data-visible")).toBe("1");

    // 按下（手指按住向下提示行）。
    await act(async () => {
        down.dispatchEvent(
            new PointerEvent("pointerdown", {
                bubbles: true,
                cancelable: true,
                pointerType: "touch",
                pointerId: 1,
                isPrimary: true,
            }),
        );
    });

    // 滚到尽头：判据变了，但按下期间**不许**把可见性写掉。
    shell.scrollTop = 200;
    await fireScroll();
    expect(
        down.getAttribute("data-visible"),
        "按下期间提示行不能隐藏 —— 隐藏会让 pointerup 落到别处",
    ).toBe("1");

    // 抬起：补写一次，提示行这才消失。
    await act(async () => {
        down.dispatchEvent(
            new PointerEvent("pointerup", {
                bubbles: true,
                pointerType: "touch",
                pointerId: 1,
                isPrimary: true,
            }),
        );
    });
    await fireScroll();
    expect(down.getAttribute("data-visible")).toBeNull();
});
