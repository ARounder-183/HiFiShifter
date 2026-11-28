// @vitest-environment jsdom
/*
 * 手写菜单的方向键导航。
 *
 * 【为什么必须有】本仓库有两套菜单：`AppContextMenu`（声明式，自带完整键盘模型）
 * 与几个手写菜单（需要内联输入框、滑杆等 `AppContextMenu` 装不下的内容）。
 * 后者此前只处理 Escape —— 声明了 `role="menu"` / `role="menuitem"` 却没有菜单的
 * 键盘契约：方向键不动、Home/End 不动。ARIA 角色是一份承诺，不是装饰。
 *
 * 这里锁定的契约（与 `AppContextMenu` 对齐）：
 * 1. 焦点不在菜单里时（右键点开就是这种状态）第一个 ↓ 进第一项，↑ 进最后一项；
 * 2. 之后在项之间**循环**移动；
 * 3. `Home` / `End` 到两端；
 * 4. 禁用的项不参与；
 * 5. 焦点在输入框里时方向键归输入框（移动文本光标），菜单不抢。
 */

import { act, useRef } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test } from "vitest";

import { useMenuKeyboard } from "./useMenuKeyboard";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** 一个最小的 `role="menu"` 表面：三个可用项 + 一个禁用项 + 一个输入框。 */
function TestMenu({ withInput = false }: { withInput?: boolean }) {
    const ref = useRef<HTMLDivElement | null>(null);
    useMenuKeyboard(ref);
    return (
        <div ref={ref} role="menu">
            {withInput ? <input data-testid="field" /> : null}
            <button type="button" role="menuitem" data-id="a">
                A
            </button>
            <button type="button" role="menuitem" data-id="b">
                B
            </button>
            <button type="button" role="menuitem" disabled data-id="disabled">
                D
            </button>
            <button type="button" role="menuitem" data-id="c">
                C
            </button>
        </div>
    );
}

/*
 * 必须真正卸载根，而不只是摘掉宿主元素：`useMenuKeyboard` 监听的是 document，
 * 只摘元素会让上一个用例的监听器留在 document 上，下一个用例按下方向键时
 * 两个菜单同时响应（实测会让焦点落到错误的地方）。
 */
const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.();
});

async function mount(withInput = false) {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(<TestMenu withInput={withInput} />);
    });
    const menu = host.querySelector<HTMLElement>('[role="menu"]');
    if (!menu) throw new Error("菜单未渲染");
    const items = Array.from(menu.querySelectorAll<HTMLElement>('[role="menuitem"]'));
    const focusedId = () =>
        document.activeElement instanceof HTMLElement
            ? (document.activeElement.getAttribute("data-id") ?? document.activeElement.tagName)
            : null;
    const press = async (key: string) => {
        await act(async () => {
            document.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    return { host, menu, items, focusedId, press };
}

test("焦点不在菜单里时，第一个 ↓ 进第一项、↑ 进最后一项", async () => {
    const m = await mount();

    // 右键点开菜单时的真实状态：焦点还在被点的元素上（这里是 body）。
    expect(document.activeElement).toBe(document.body);

    await m.press("ArrowDown");
    expect(m.focusedId()).toBe("a");

    document.body.focus();
    await m.press("ArrowUp");
    expect(m.focusedId()).toBe("c");
});

test("方向键在可用项之间循环，跳过禁用的项", async () => {
    const m = await mount();

    await m.press("ArrowDown");
    expect(m.focusedId()).toBe("a");

    await m.press("ArrowDown");
    expect(m.focusedId()).toBe("b");

    // 第三项是禁用项，必须被跳过。
    await m.press("ArrowDown");
    expect(m.focusedId()).toBe("c");

    // 循环回第一项。
    await m.press("ArrowDown");
    expect(m.focusedId()).toBe("a");

    await m.press("ArrowUp");
    expect(m.focusedId()).toBe("c");
});

test("Home / End 到两端", async () => {
    const m = await mount();

    await m.press("End");
    expect(m.focusedId()).toBe("c");

    await m.press("Home");
    expect(m.focusedId()).toBe("a");
});

/*
 * 嵌套菜单：子面板在**外层菜单的 DOM 子树里**（`ClipContextMenu` 的真实结构）。
 * 这正是"谁该响应"必须按焦点位置判断、而不能按挂载顺序判断的原因。
 */
function NestedMenu() {
    const outerRef = useRef<HTMLDivElement | null>(null);
    const innerRef = useRef<HTMLDivElement | null>(null);
    useMenuKeyboard(outerRef);
    useMenuKeyboard(innerRef);
    return (
        <div ref={outerRef} role="menu">
            <button type="button" role="menuitem" data-id="outer-1">
                O1
            </button>
            <button type="button" role="menuitem" data-id="outer-2">
                O2
            </button>
            <div ref={innerRef} role="menu">
                <button type="button" role="menuitem" data-id="inner-1">
                    I1
                </button>
                <button type="button" role="menuitem" data-id="inner-2">
                    I2
                </button>
            </div>
        </div>
    );
}

test("焦点在外层菜单时，子面板不抢方向键", async () => {
    /*
     * 【为什么这条必须有】若两层都监听 document 且不判断焦点，外层把焦点移到
     * 自己的下一项后，子面板会看到"当前项 = -1"而把焦点抢进自己 —— 用户按一次
     * 下键就被弹进了子菜单。
     */
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    await act(async () => root.render(<NestedMenu />));

    const outer2 = host.querySelector<HTMLElement>('[data-id="outer-1"]');
    const inner1 = host.querySelector<HTMLElement>('[data-id="inner-1"]');
    const outerMenu = host.querySelector<HTMLElement>('[role="menu"]');
    if (!outer2 || !inner1 || !outerMenu) throw new Error("菜单未渲染");

    await act(async () => outer2.focus());
    await act(async () => {
        document.dispatchEvent(
            new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true }),
        );
    });

    expect(document.activeElement?.getAttribute("data-id")).toBe("outer-2");
    expect(outerMenu.contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(inner1);
});

test("焦点在子面板里时，外层菜单不抢方向键", async () => {
    /*
     * 反方向的那一半：子面板在外层菜单的 DOM 子树里，所以"谁含焦点谁响应"不能
     * 只看 `contains` —— 外层也 contains 子面板里的元素。必须由**最内层**响应。
     * 少了这条判断，先注册的外层监听器会先把焦点挪到自己的第一项
     * （它看到的"当前项"是 -1），用户按一次下键就被弹出子菜单。
     */
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    await act(async () => root.render(<NestedMenu />));

    const inner1 = host.querySelector<HTMLElement>('[data-id="inner-1"]');
    if (!inner1) throw new Error("子面板未渲染");

    await act(async () => inner1.focus());
    await act(async () => {
        document.dispatchEvent(
            new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true }),
        );
    });

    expect(document.activeElement?.getAttribute("data-id")).toBe("inner-2");
});

test("焦点不在任何菜单里时，最后打开的那一层拿到焦点", async () => {
    // 右键点开的初始状态：焦点还在被点的元素上（body）。子面板更晚挂载，
    // 因此它应当是"当前菜单"。
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    await act(async () => root.render(<NestedMenu />));

    await act(async () => {
        document.dispatchEvent(
            new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true }),
        );
    });

    expect(document.activeElement?.getAttribute("data-id")).toBe("inner-1");
});

test("焦点在输入框里时方向键归输入框，菜单不抢", async () => {
    const m = await mount(true);
    const field = m.host.querySelector<HTMLInputElement>('[data-testid="field"]');
    if (!field) throw new Error("输入框未渲染");

    await act(async () => field.focus());
    await m.press("ArrowDown");

    // 若菜单抢走这个按键，焦点会跑到第一项 —— 用户就没法在输入框里移动光标了。
    expect(document.activeElement).toBe(field);
});

/*
 * PageUp / PageDown。
 *
 * 【为什么是"移动选中项"而不是"只滚"】只滚会让高亮停在屏幕外 —— 用户按了
 * PageDown，界面滚了，但"当前项"看不见，下一次 Enter 会激活一个他没看见的条目。
 * 菜单的键盘模型里"当前项"必须永远可见。
 *
 * jsdom 没有布局（clientHeight 恒为 0），因此这里锁的是**边界**：视口量不到时
 * 一页至少是一步；视口很大时一步走到尽头。精确的项数在
 * menuScrollModel.test.ts 里逐值断言。
 */
test("PageDown / PageUp 至少移动一项，且不越界", async () => {
    const m = await mount();

    await m.press("Home");
    expect(m.focusedId()).toBe("a");

    await m.press("PageDown");
    // clientHeight 为 0 时一页 = 1 项（下限），绝不能是 0 项。
    expect(m.focusedId()).toBe("b");

    await m.press("PageUp");
    expect(m.focusedId()).toBe("a");

    // 已经在第一项时再往上翻：停在第一项，不绕到末尾（翻页的语义是"再走一屏"，
    // 从末尾循环回开头会让"翻到底"变成一个没有尽头的动作）。
    await m.press("PageUp");
    expect(m.focusedId()).toBe("a");
});

test("视口很高时 PageDown 一步到末尾并停住", async () => {
    const m = await mount();
    Object.defineProperty(m.menu, "clientHeight", { configurable: true, value: 600 });

    await m.press("Home");
    await m.press("PageDown");
    expect(m.focusedId()).toBe("c");

    await m.press("PageDown");
    expect(m.focusedId()).toBe("c");
});

/*
 * 打字跳项。
 *
 * 【为什么必须在这里也有】方向键在"帮助"菜单里要走十几次才能到目标项；原生菜单
 * 与 VS Code 都支持直接打首字母跳过去。两族菜单（activeIndex 与 DOM 焦点）各写
 * 一套的话，手感和可用性会立刻分叉。
 */
function TypeaheadMenu() {
    const ref = useRef<HTMLDivElement | null>(null);
    useMenuKeyboard(ref);
    return (
        <div ref={ref} role="menu">
            <button type="button" role="menuitem" data-id="alpha">
                Alpha
            </button>
            <button type="button" role="menuitem" data-id="beta">
                Beta
            </button>
            <button type="button" role="menuitem" data-id="gamma">
                Gamma
            </button>
        </div>
    );
}

test("打字跳到以该前缀开头的项", async () => {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    await act(async () => root.render(<TypeaheadMenu />));

    const press = async (key: string) => {
        await act(async () => {
            document.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };

    await press("g");
    expect(document.activeElement?.getAttribute("data-id")).toBe("gamma");

    // 前缀继续拼：\`ga\` 仍然指向 gamma。
    await press("a");
    expect(document.activeElement?.getAttribute("data-id")).toBe("gamma");

    // 无匹配时**不动**（而不是跳回第一项）。
    await new Promise((resolve) => setTimeout(resolve, 600));
    await press("z");
    expect(document.activeElement?.getAttribute("data-id")).toBe("gamma");
});

/*
 * ArrowLeft 退出子面板。
 *
 * 【为什么只有子面板响应】外层菜单没有"左边"可去；子面板按左键应当收起自己、
 * 回到触发项 —— 那是原生菜单的行为。给所有层都响应会让一次左键关掉整棵树。
 */
test("ArrowLeft 只让子面板退出，外层不响应", async () => {
    let dismissed = 0;
    function NestedDismiss() {
        const outerRef = useRef<HTMLDivElement | null>(null);
        const innerRef = useRef<HTMLDivElement | null>(null);
        useMenuKeyboard(outerRef);
        useMenuKeyboard(innerRef, true, { nested: true, onDismiss: () => (dismissed += 1) });
        return (
            <div ref={outerRef} role="menu">
                <button type="button" role="menuitem" data-id="outer-1">
                    O1
                </button>
                <div ref={innerRef} role="menu">
                    <button type="button" role="menuitem" data-id="inner-1">
                        I1
                    </button>
                </div>
            </div>
        );
    }
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    await act(async () => root.render(<NestedDismiss />));

    const press = async (key: string) => {
        await act(async () => {
            document.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };

    // 焦点在外层：左键不该有任何反应。
    const outer1 = host.querySelector<HTMLElement>('[data-id="outer-1"]');
    await act(async () => outer1?.focus());
    await press("ArrowLeft");
    expect(dismissed).toBe(0);

    // 焦点在子面板里：左键退出本层。
    const inner1 = host.querySelector<HTMLElement>('[data-id="inner-1"]');
    await act(async () => inner1?.focus());
    await press("ArrowLeft");
    expect(dismissed).toBe(1);
});
