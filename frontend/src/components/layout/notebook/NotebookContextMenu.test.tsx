// @vitest-environment jsdom
/*
 * 记事本右键菜单**表面**的行为测试。
 *
 * 【为什么必须有】这张菜单此前不在任何测试里 —— 而它承担着三条容易悄悄坏掉、
 * 又不容易在手工点测里发现的契约：
 *   1. **必须挂在 `document.body`**。留在面板的布局盒里会被 `.hs-scroll-gutter`
 *      之类的 `overflow` 裁掉，表现为"菜单没出现"（而不是报错）。
 *   2. **键盘可达**。原语有完整的方向键 / Home / End / Enter 模型，但它依赖
 *      调用方把 `disabled` 与 `heading` 如实标出来 —— 标错就会"方向键停在一条
 *      点不动的项上"。
 *   3. **禁用项真的点不动**，并且**点外部才关**。
 *
 * 这里只测表面；"什么时候出现哪一条"由 `notebookMenu.test.ts` 覆盖。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test } from "vitest";

import type { AppMenuItemSpec } from "../../../ui";
import { NotebookContextMenu } from "./NotebookContextMenu";

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

/** 菜单表面（原语用 `data-hs-context-menu="1"` 标记"已打开"）。 */
function surface(): HTMLElement | null {
    return document.querySelector<HTMLElement>('[data-hs-context-menu="1"]');
}

function itemByText(text: string): HTMLButtonElement {
    const button = Array.from(document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")).find(
        (element) => element.textContent?.includes(text),
    );
    if (!button) throw new Error(`menu item not found: ${text}`);
    return button;
}

async function mount(items: AppMenuItemSpec[], onClose: () => void = () => {}): Promise<void> {
    await act(async () => {
        root.render(
            <NotebookContextMenu
                x={40}
                y={60}
                items={items}
                ariaLabel="test menu"
                onClose={onClose}
            />,
        );
    });
}

/**
 * 让出一个宏任务，等菜单的**延后聚焦**落地。
 *
 * 【为什么不能只靠 `act`】菜单把焦点搬到自己是延后一个宏任务做的（同步聚焦会被
 * 浏览器在 `contextmenu` 手势收尾时回滚，见 `Menu.tsx` 的说明）。`act(async …)`
 * 只保证微任务与 React 自身的调度被冲刷，**不保证** `setTimeout(0)` 已经跑过 ——
 * 实测这条断言在并行负载下约 1/6 的概率读到"焦点还没搬"。显式等一个宏任务即可。
 */
async function settleFocus(): Promise<void> {
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
    });
}

const SAMPLE: AppMenuItemSpec[] = [
    { key: "copy", label: "Copy", shortcut: "Ctrl+C", onSelect: () => {} },
    { key: "group", label: "Format", heading: true },
    { key: "bold", label: "Bold", onSelect: () => {} },
    { key: "grey", label: "Cut", disabled: true, tooltip: "Select some text first" },
];

test("菜单挂在 document.body 上，不在触发它的容器里", async () => {
    await mount(SAMPLE);
    const menu = surface();
    expect(menu, "菜单没有渲染").not.toBeNull();
    expect(host.contains(menu), "菜单留在了面板容器里，会被 overflow 裁掉").toBe(false);
    expect(document.body.contains(menu)).toBe(true);
});

test("渲染出 aria 名称与全部条目（含分组标题与禁用项）", async () => {
    await mount(SAMPLE);
    expect(surface()?.getAttribute("aria-label")).toBe("test menu");
    expect(itemByText("Copy")).toBeTruthy();
    expect(itemByText("Bold")).toBeTruthy();
    expect(itemByText("Cut").disabled).toBe(true);
    // 分组标题不是可点的按钮。
    expect(
        Array.from(document.querySelectorAll(".hs-menu__item")).some(
            (element) => element.textContent?.trim() === "Format",
        ),
    ).toBe(false);
});

test("快捷键提示列只在有提示时渲染", async () => {
    await mount(SAMPLE);
    expect(itemByText("Copy").textContent).toContain("Ctrl+C");
    expect(itemByText("Bold").textContent).not.toContain("Ctrl");
});

test("禁用项点击不触发回调", async () => {
    let called = 0;
    await mount([{ key: "cut", label: "Cut", disabled: true, onSelect: () => (called += 1) }]);
    await act(async () => itemByText("Cut").click());
    expect(called).toBe(0);
});

test("方向键跳过禁用项与分组标题，Home / End 落到首尾可选项", async () => {
    await mount(SAMPLE);
    const active = () => document.querySelector('[data-active="1"]')?.textContent ?? "";

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    expect(active()).toContain("Copy");

    // 下一项应是 Bold —— 中间的 "Format" 标题与末尾的禁用项都不参与导航。
    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    expect(active()).toContain("Bold");

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "End" }));
    });
    expect(active()).toContain("Bold");

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Home" }));
    });
    expect(active()).toContain("Copy");
});

test("Enter 激活当前项并关闭菜单", async () => {
    let selected = 0;
    let closed = 0;
    await mount(
        [{ key: "copy", label: "Copy", onSelect: () => (selected += 1) }],
        () => (closed += 1),
    );
    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    });
    expect(selected).toBe(1);
    expect(closed).toBe(1);
});

test("Esc 关闭菜单", async () => {
    let closed = 0;
    await mount(SAMPLE, () => (closed += 1));
    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(closed).toBe(1);
});

test("点菜单外部关闭，点菜单内部不关闭", async () => {
    let closed = 0;
    await mount(SAMPLE, () => (closed += 1));

    await act(async () => {
        itemByText("Copy").dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    });
    expect(closed, "点在菜单内部不应关闭").toBe(0);

    await act(async () => {
        document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    });
    expect(closed, "点在菜单外部应关闭").toBe(1);
});

test("点击可选项触发回调（关闭由原语负责）", async () => {
    let selected = 0;
    let closed = 0;
    await mount(
        [{ key: "copy", label: "Copy", onSelect: () => (selected += 1) }],
        () => (closed += 1),
    );
    await act(async () => itemByText("Copy").click());
    expect(selected).toBe(1);
    expect(closed).toBe(1);
});

/*
 * 焦点必须收到菜单上 —— 这是"方向键能导航"的前提。
 *
 * 【要钉死什么】原语的方向键守卫是 `ownsArrowKeys(document.activeElement)`：
 * 焦点元素吞方向键就让路。记事本的触发面是 contenteditable，正好命中这条守卫，
 * 于是不收焦点时**方向键全被判给编辑器**，菜单高亮一格不动（浏览器实测：
 * `menuActive` 恒为 null，而光标在动）。键盘用户按 `ContextMenu` 键打开菜单后
 * 无法选择任何一项。
 *
 * 反过来也要钉住：关闭时焦点要**还给触发者**，而不是掉回 `<body>`。
 */
test("打开时把焦点收到菜单上，关闭时还给触发者", async () => {
    const opener = document.createElement("div");
    opener.tabIndex = 0;
    document.body.append(opener);
    opener.focus();
    expect(document.activeElement).toBe(opener);

    await mount([{ key: "copy", label: "Copy", onSelect: () => {} }]);
    await settleFocus();
    expect(document.activeElement, "焦点没有收到菜单上").toBe(surface());

    await act(async () => root.unmount());
    expect(document.activeElement, "关闭后焦点没有还给触发者").toBe(opener);
});

test("焦点原本在 contenteditable 上时，方向键仍然能导航", async () => {
    // 模拟"触发面本身就是可编辑元素"：焦点先落在 contenteditable 上。
    const editable = document.createElement("div");
    editable.contentEditable = "true";
    editable.tabIndex = 0;
    document.body.append(editable);
    editable.focus();

    await mount(SAMPLE);
    await settleFocus();
    // 焦点已被菜单收走 —— 这正是 `ownsArrowKeys` 不再让路的原因。
    // （不断言 `isContentEditable`：jsdom 没有实现它，断言只会测到 jsdom 的缺口。）
    expect(document.activeElement).toBe(surface());

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    expect(document.querySelector('[data-active="1"]')?.textContent).toContain("Copy");
});

/*
 * 子菜单（`items` 字段）。
 *
 * 【要钉死什么】三层契约，任一层坏了都表现为"子菜单看着有、其实用不了"：
 *   1. 触发项能展开出一个**独立**的 `role="menu"` 表面（不是把子项平铺进外层）；
 *   2. 触发项参与外层的方向键导航，并且**看得见**高亮（它由 `AppSubMenu` 渲染，
 *      不像普通项那样自带 `data-active` —— 少了这条，方向键走到它身上毫无反馈）；
 *   3. 子项点击照常触发回调并关闭整张菜单。
 */
const WITH_SUBMENU: AppMenuItemSpec[] = [
    { key: "copy", label: "Copy", onSelect: () => {} },
    {
        key: "format",
        label: "Format",
        items: [
            { key: "bold", label: "Bold", shortcut: "Ctrl+B", onSelect: () => {} },
            { key: "italic", label: "Italic", onSelect: () => {} },
        ],
    },
    { key: "find", label: "Find", onSelect: () => {} },
];

test("子菜单触发项展开出一个独立的菜单表面", async () => {
    await mount(WITH_SUBMENU);
    expect(document.querySelectorAll('[role="menu"]')).toHaveLength(1);
    // 子项在展开前不在 DOM 里。
    expect(document.querySelector(".hs-menu--submenu")).toBeNull();

    await act(async () => itemByText("Format").click());

    const panels = document.querySelectorAll('[role="menu"]');
    expect(panels, "展开后应是两个菜单表面（外层 + 子面板）").toHaveLength(2);
    const sub = document.querySelector(".hs-menu--submenu");
    expect(sub, "子面板没有出现").not.toBeNull();
    expect(sub?.textContent).toContain("Bold");
    expect(sub?.textContent).toContain("Italic");
    // 触发项自己声明了它会长出菜单。
    expect(itemByText("Format").getAttribute("aria-haspopup")).toBe("menu");
    expect(itemByText("Format").getAttribute("aria-expanded")).toBe("true");
});

test("子菜单触发项参与方向键导航，且高亮看得见", async () => {
    await mount(WITH_SUBMENU);
    const active = () => document.querySelector('[data-active="1"]')?.textContent ?? "";

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    expect(active()).toContain("Copy");

    // 关键：触发项必须像普通项一样被高亮 —— 它由 AppSubMenu 渲染，
    // 父层的 activeIndex 得显式传下去才会变成 data-active。
    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    expect(active(), "方向键走到子菜单触发项上没有高亮").toContain("Format");

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    expect(active()).toContain("Find");
});

test("点子项触发回调，并关掉整张菜单", async () => {
    let bolded = 0;
    let closed = 0;
    const items: AppMenuItemSpec[] = [
        {
            key: "format",
            label: "Format",
            items: [{ key: "bold", label: "Bold", onSelect: () => (bolded += 1) }],
        },
    ];
    await mount(items, () => (closed += 1));
    await act(async () => itemByText("Format").click());

    await act(async () => itemByText("Bold").click());
    expect(bolded).toBe(1);
    expect(closed, "子项执行后应关掉整张菜单").toBe(1);
});

test("禁用子项点不动", async () => {
    let bolded = 0;
    const items: AppMenuItemSpec[] = [
        {
            key: "format",
            label: "Format",
            items: [{ key: "bold", label: "Bold", disabled: true, onSelect: () => (bolded += 1) }],
        },
    ];
    await mount(items);
    await act(async () => itemByText("Format").click());
    await act(async () => itemByText("Bold").click());
    expect(bolded).toBe(0);
});

/*
 * 含子菜单的菜单壳**不得裁切内容**。
 *
 * 【为什么这条必须在**单测**里也钉一遍】子面板留在父壳内部（它要靠百分比相对触发项
 * 定位），而壳默认 `max-height` + `overflow-y: auto` 会把伸出右边的子面板裁掉 ——
 * 实测只露出 5px，等于**子菜单完全看不见**。jsdom 没有布局，量不出裁切，但
 * `hs-menu--no-scroll` 这个**类名**是可观察的，而它正是"别裁我"的唯一表达。
 *
 * 【为什么当初漏了】浏览器验证用的是 `getBoundingClientRect()`：它给的是布局几何，
 * 与裁切无关，照样报出完整矩形 —— 假绿灯。可见性必须用 `elementFromPoint` 判，
 * 那条检查留在 `scripts/notebook-menu-probe.mjs` 里（jsdom 做不到）。
 */
test("含子菜单时壳用 hs-menu--no-scroll（否则子面板被 overflow 裁掉）", async () => {
    await mount(WITH_SUBMENU);
    const shell = surface();
    expect(shell?.className, "壳会裁掉子面板").toContain("hs-menu--no-scroll");
});

test("不含子菜单时不加 no-scroll（长菜单仍要能自己滚）", async () => {
    await mount(SAMPLE);
    expect(surface()?.className).not.toContain("hs-menu--no-scroll");
});
