/**
 * 菜单栏下拉的三件套：点击开、点外面关、Esc 关、键盘打开时焦点进菜单。
 *
 * 【为什么这四条要单独锁】8 个下拉共用这一个小组件。任何一条错，症状都是"某一张
 * 菜单打不开 / 关不掉 / 键盘用户按了方向键界面没反应" —— 而这些都发生在真实
 * 菜单栏里，单看某一处代码看不出来。
 */
// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test } from "vitest";

import { AppMenuBarItem } from "./AppMenuBarItem";
import type { AppMenuItemSpec } from "./Menu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ITEMS: AppMenuItemSpec[] = [
    { key: "a", label: "Alpha" },
    { key: "b", label: "Beta" },
];

let host: HTMLDivElement | null = null;
let unmount: (() => void) | null = null;

afterEach(async () => {
    if (unmount) await act(async () => unmount?.());
    host?.remove();
    host = null;
    unmount = null;
    document.body.innerHTML = "";
});

/** 受控包装：真实用法里 open 住在菜单栏自己的一个 state 里。 */
function Harness({ align }: { align?: "start" | "end" }) {
    const [open, setOpen] = useState(false);
    return (
        <AppMenuBarItem
            open={open}
            onOpenChange={setOpen}
            items={ITEMS}
            ariaLabel="Menu"
            align={align}
        >
            <span>Menu</span>
        </AppMenuBarItem>
    );
}

async function render(align?: "start" | "end") {
    host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<Harness align={align} />));
    unmount = () => root.unmount();
    const trigger = host.querySelector<HTMLButtonElement>("button")!;
    const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
    const click = async (element: Element) => {
        await act(async () => {
            element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
        });
    };
    /*
     * 只在**目标元素**上派发一次。真实事件不会既在目标上派发一次、又在 document 上
     * 派发一次 —— 而"点外面关闭"用的是 document 的**捕获**监听，目标上的冒泡事件
     * 同样会经过它（捕获先于目标）。多派发一次 document 事件等于凭空造了一次
     * "点在 document 上"，那当然算外面。
     */
    const pointerDown = async (element: Element) => {
        await act(async () => {
            element.dispatchEvent(
                new PointerEvent("pointerdown", { bubbles: true, cancelable: true }),
            );
        });
    };
    const press = async (key: string) => {
        await act(async () => {
            trigger.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };
    return { host, trigger, menu, click, pointerDown, press };
}

test("点击触发器展开菜单，aria-expanded 如实反映状态", async () => {
    const t = await render();
    expect(t.menu()).toBeNull();
    expect(t.trigger.getAttribute("aria-haspopup")).toBe("menu");
    expect(t.trigger.getAttribute("aria-expanded")).toBe("false");

    await t.click(t.trigger);
    expect(t.menu()).not.toBeNull();
    expect(t.trigger.getAttribute("aria-expanded")).toBe("true");

    // 再点一次收起 —— 触发器是**切换**，不是"只开不关"。
    await t.click(t.trigger);
    expect(t.menu()).toBeNull();
});

test("点菜单外面关闭；点菜单里面不关闭", async () => {
    const t = await render();
    await t.click(t.trigger);
    const menu = t.menu()!;

    await t.pointerDown(menu);
    expect(t.menu(), "点菜单里面不该关掉菜单").not.toBeNull();

    await t.pointerDown(document.body);
    expect(t.menu(), "点菜单外面必须关掉菜单").toBeNull();
});

test("点触发器本身不算'点外面'（否则切换会闪一下又回来）", async () => {
    /*
     * 触发器的 pointerdown 会冒泡到 document 的捕获监听。若那里把它判成"外面"，
     * 菜单先关一次、触发器的 onClick 再开一次 —— 用户看到菜单闪一下又回来，
     * 而且永远关不掉。
     */
    const t = await render();
    await t.click(t.trigger);
    expect(t.menu()).not.toBeNull();

    await t.pointerDown(t.trigger);
    expect(t.menu(), "点触发器时菜单不该被'点外面'逻辑关掉").not.toBeNull();

    await t.click(t.trigger);
    expect(t.menu(), "触发器的切换必须真的收起菜单").toBeNull();
});

test("Escape 关闭菜单", async () => {
    const t = await render();
    await t.click(t.trigger);
    await act(async () => {
        document.dispatchEvent(
            new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }),
        );
    });
    expect(t.menu()).toBeNull();
});

test("键盘打开时焦点进菜单（鼠标打开时留在触发器上）", async () => {
    /*
     * 【为什么区分】鼠标打开时焦点留在触发器上 —— 与 VS Code / Windows 菜单栏一致，
     * 用户还要能 Tab 走开。键盘打开时焦点必须进菜单，否则方向键无处可去。
     */
    const t = await render();
    await t.click(t.trigger);
    expect(t.menu()).not.toBeNull();
    // 鼠标路径：焦点没被搬进菜单。
    expect(document.activeElement === t.trigger || document.activeElement === document.body).toBe(
        true,
    );
    await t.click(t.trigger);

    await t.press("ArrowDown");
    expect(t.menu()).not.toBeNull();
    // autoFocus 走的是"下一宏任务"（避开浏览器自己的焦点收尾）。
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(document.activeElement, "键盘打开后焦点必须在菜单里").toBe(t.menu());
});

test("菜单条目按声明渲染成 menuitem", async () => {
    const t = await render();
    await t.click(t.trigger);
    const items = Array.from(t.menu()!.querySelectorAll('[role="menuitem"]'));
    expect(items.map((item) => item.textContent)).toEqual(["Alpha", "Beta"]);
});
