// @vitest-environment jsdom
/*
 * 全局快捷键分发器与**复合控件**的方向键归属。
 *
 * 【为什么必须有】`playback.seekLeft/Right` 默认绑定在左右方向键上，
 * `track.selectUp/Down` 绑在上下方向键上，而分发器运行在 **window 捕获阶段**，
 * 命中后 `preventDefault()` + `stopPropagation()`。于是凡是声明了 ARIA 键盘契约的
 * 复合控件，方向键永远到不了控件自己：焦点停在停靠标签上按 ←/→ 会去 seek，
 * 而不是切标签 —— 标签条的键盘模型（`DockTabBar` 的 roving tabIndex + 方向键）
 * 因此**完全失效**，而它的单元测试全绿。
 *
 * 这是浏览器实测发现的：在真实应用里给标签派发 `ArrowLeft`，事件在 window 捕获
 * 阶段就被吞掉，标签自己的处理器收不到。所以这里锁定的契约是
 * **"焦点在复合控件里时，全局分发器必须让路"**，而不只是"标签处理器写对了"。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterAll, afterEach, expect, test } from "vitest";

import keybindingsReducer from "./keybindingsSlice";
import sessionReducer, {setToolMode} from "../session/sessionSlice";
import { useKeybindings } from "./useKeybindings";
import type { ActionId } from "./types";
import {createPluginHost,type WebViewMessagePort} from "../../services/pluginHost";
import {
    installFocusSurfaceTracking,
    resetActiveSurfaceForTests,
    setActiveSurfaceExplicit,
} from "../uiFocus/focusSurface";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// 活动表面的跟踪由 App 安装；这里装上，才能覆盖"焦点归属"的真实链路。
const disposeFocusSurface = installFocusSurfaceTracking();

function createTestStore() {
    return configureStore({
        reducer: { keybindings: keybindingsReducer, session: sessionReducer },
    });
}

const fired: ActionId[] = [];

function Harness() {
    useKeybindings((actionId) => fired.push(actionId));
    return null;
}

const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    fired.length = 0;
    resetActiveSurfaceForTests();
    while (mounted.length) await mounted.pop()?.();
});

afterAll(() => {
    disposeFocusSurface();
});

async function mount() {
    const store = createTestStore();
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <Harness />
            </Provider>,
        );
    });
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });
    return { store, host };
}

/** 在 `target` 上派发一个真实的（可冒泡、可取消的）keydown。 */
function press(target: EventTarget, key: string): { defaultPrevented: boolean } {
    const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
    target.dispatchEvent(event);
    return { defaultPrevented: event.defaultPrevented };
}

/** 建一个 `role="tablist"` + 可聚焦的 `role="tab"`，与停靠标签条同形。 */
function buildTablist(): { list: HTMLElement; tab: HTMLElement } {
    const list = document.createElement("div");
    list.setAttribute("role", "tablist");
    const tab = document.createElement("div");
    tab.setAttribute("role", "tab");
    tab.tabIndex = 0;
    list.append(tab);
    document.body.append(list);
    mounted.push(async () => {
        list.remove();
    });
    return { list, tab };
}

test("焦点在标签条里时，方向键不被全局分发器吞掉", async () => {
    await mount();
    const { tab } = buildTablist();
    tab.focus();

    const result = press(tab, "ArrowLeft");

    // 全局绑定（playback.seekLeft）不得触发……
    expect(fired).toEqual([]);
    // ……而且事件必须保持"未被消费"，否则标签自己的处理器也收不到
    // （分发器命中时正是靠 preventDefault + stopPropagation 让控件收不到）。
    expect(result.defaultPrevented).toBe(false);
});

test("焦点不在复合控件里时，方向键仍然走全局绑定", async () => {
    await mount();
    buildTablist(); // 存在但未聚焦

    press(document.body, "ArrowLeft");

    // 这条是上一条的对照：让路只在"焦点在复合控件内"时发生，
    // 否则时间轴的 ←/→ seek 就被整体破坏了。
    expect(fired).toEqual(["playback.seekLeft"]);
});

test("菜单里的方向键同样归菜单自己", async () => {
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    const item = document.createElement("button");
    item.setAttribute("role", "menuitem");
    item.tabIndex = 0;
    menu.append(item);
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });
    item.focus();

    expect(press(item, "ArrowDown").defaultPrevented).toBe(false);
    expect(fired).toEqual([]);
});

test("右键菜单打开时（焦点不在菜单里）方向键也归菜单", async () => {
    /*
     * 右键菜单的真实状态：焦点还在被右键的元素上，菜单只是"打开了"。
     * 只检查焦点归属会漏掉这一整类 —— 菜单开着，方向键却去 seek 了。
     */
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    const item = document.createElement("button");
    item.setAttribute("role", "menuitem");
    item.tabIndex = 0;
    menu.append(item);
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });
    // 刻意**不**把焦点放进菜单：焦点留在 body 上，与右键点开时一致。
    document.body.focus();

    expect(press(document.body, "ArrowDown").defaultPrevented).toBe(false);
    expect(fired).toEqual([]);
});

test("关闭但**仍挂载**的菜单不得屏蔽方向键", async () => {
    /*
     * Radix 的菜单内容关闭后仍留在 DOM 里（`data-state="closed"`）。若把"存在"
     * 当作"打开"，方向键的全局绑定会被永久屏蔽 —— 时间轴的 ←/→ seek 直接失效。
     * 这条就是那个回归的守卫。
     */
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    menu.setAttribute("data-state", "closed");
    const item = document.createElement("button");
    item.setAttribute("role", "menuitem");
    menu.append(item);
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });

    press(document.body, "ArrowDown");
    expect(fired).toEqual(["track.selectDown"]);
});

test("焦点滞留在**已关闭**的菜单里时，方向键也回到全局绑定", async () => {
    /*
     * Radix 关闭菜单后会把内容留在 DOM 里，实测焦点也还停在里面。若"焦点在
     * `role="menu"` 内"就算拥有方向键，一个已经关掉的菜单会继续吞掉时间轴的
     * ←/→ seek。
     */
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    menu.setAttribute("data-state", "closed");
    const item = document.createElement("button");
    item.setAttribute("role", "menuitem");
    item.tabIndex = 0;
    menu.append(item);
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });
    item.focus();

    press(item, "ArrowDown");
    expect(fired).toEqual(["track.selectDown"]);
});

test("菜单关闭后，方向键回到全局绑定", async () => {
    // 对照：让路必须以"表面打开"为条件，否则时间轴的 ←/→ 会被永久屏蔽。
    await mount();
    press(document.body, "ArrowDown");
    expect(fired).toEqual(["track.selectDown"]);
});

test("非方向键的全局快捷键在标签条里照常生效", async () => {
    await mount();
    const { tab } = buildTablist();
    tab.focus();

    // 让路只针对复合控件拥有的那一组键（方向键 + Home/End/PageUp/PageDown），
    // 其余全局快捷键不受影响 —— `k` 默认绑定在节拍器上。
    press(tab, "k");
    expect(fired).toEqual(["playback.metronome"]);
});

// ── 输入式快速跳转（`data-hs-typeahead`）的表面 ────────────────────────────

/*
 * 【为什么要有这一组】全局绑定里有 15 个**无修饰的单键**（`d` 切参数拖拽方向、
 * `s` 分割片段、`k` 节拍器…）以及 Enter / Space / Backspace / Delete 等。分发器在
 * window **捕获**阶段命中后就 `preventDefault()` + `stopPropagation()`，于是焦点在
 * 文件列表里时：打字母被全局绑定截走（"输入字母快速跳转"永远进不来），Enter 去停
 * 播放而不是打开文件夹，Backspace 去初始化参数而不是回上级目录。
 *
 * 契约：**声明了 `data-hs-typeahead` 的表面拥有它自己实现了的那些键**，且事件必须
 * 保持"未被消费"，否则面板自己的处理器也收不到。
 */

/**
 * 建一个与文件列表同形的表面：`data-hs-surface` + `data-hs-typeahead` 容器 + 可聚焦的行。
 *
 * 两个属性缺一不可：前者让分发器知道"用户此刻在这个表面里工作"，后者声明
 * "这个表面自己拥有输入式跳转的按键"。
 */
function buildTypeAheadList(): { list: HTMLElement; row: HTMLElement } {
    const list = document.createElement("div");
    list.setAttribute("data-hs-surface", "fileBrowser");
    list.setAttribute("data-hs-typeahead", "1");
    const row = document.createElement("div");
    row.setAttribute("role", "option");
    row.tabIndex = 0;
    list.append(row);
    document.body.append(list);
    mounted.push(async () => {
        list.remove();
    });
    return { list, row };
}

/** 带修饰键的按键（`press` 只发 key，这里需要 ctrl/shift）。 */
function pressWith(
    target: EventTarget,
    key: string,
    modifiers: { ctrl?: boolean; shift?: boolean; alt?: boolean } = {},
): { defaultPrevented: boolean } {
    const event = new KeyboardEvent("keydown", {
        key,
        bubbles: true,
        cancelable: true,
        ctrlKey: modifiers.ctrl ?? false,
        shiftKey: modifiers.shift ?? false,
        altKey: modifiers.alt ?? false,
    });
    target.dispatchEvent(event);
    return { defaultPrevented: event.defaultPrevented };
}

test("列表里打字不被全局单键绑定截走（`d` 不再切参数拖拽方向）", async () => {
    await mount();
    const { row } = buildTypeAheadList();
    row.focus();

    const result = press(row, "d");

    expect(fired).toEqual([]);
    // 事件必须保持未消费：面板的输入式跳转就挂在这条链路上。
    expect(result.defaultPrevented).toBe(false);
});

test("列表里 Enter / Space / Backspace / Delete / Escape 都归列表自己", async () => {
    await mount();
    const { row } = buildTypeAheadList();
    row.focus();

    for (const key of ["Enter", " ", "Backspace", "Delete", "Escape"]) {
        fired.length = 0;
        const result = press(row, key);
        expect(fired, `${key} 被全局绑定截走了`).toEqual([]);
        expect(result.defaultPrevented, `${key} 被消费了`).toBe(false);
    }
});

test("列表里 Ctrl+A / Ctrl+C / Ctrl+Shift+N 归列表自己", async () => {
    await mount();
    const { row } = buildTypeAheadList();
    row.focus();

    for (const [key, mods] of [
        ["a", { ctrl: true }],
        ["c", { ctrl: true }],
        ["n", { ctrl: true, shift: true }],
    ] as const) {
        fired.length = 0;
        const result = pressWith(row, key, mods);
        expect(fired, `Ctrl+${key} 被全局绑定截走了`).toEqual([]);
        expect(result.defaultPrevented).toBe(false);
    }
});

test("列表里没实现语义的全局快捷键照常生效（Ctrl+S 保存）", async () => {
    /*
     * 让路只覆盖"面板实现了语义"的键。Ctrl+S（保存工程）面板没有对应动作，
     * 若一并吞掉，用户在文件列表里按保存会毫无反应 —— 那是新问题，不是修复。
     */
    await mount();
    const { row } = buildTypeAheadList();
    row.focus();

    pressWith(row, "s", { ctrl: true });
    expect(fired).toEqual(["project.save"]);
});

test("列表里 Shift+T / Shift+V 仍走全局绑定（面板没有对应动作）", async () => {
    /*
     * Shift+字母会产出可打印字符（Shift+T → "T"），不能因此落进"打字跳转"那一类：
     * 全局绑定里的 Shift+T（切换上一个 take）与 Shift+V（粘贴 vocal shifter）会被
     * 面板无声截走。面板若没有对应绑定，事件仍会照常流到它（这里只决定分发器是否
     * 提前让路），因此大小写不敏感的 type-ahead 不受影响。
     */
    await mount();
    const { row } = buildTypeAheadList();
    row.focus();

    pressWith(row, "T", { shift: true });
    pressWith(row, "V", { shift: true });
    expect(fired).toEqual(["clip.cycleTakePrev", "edit.pasteVocalShifter"]);
});

test("焦点不在列表里时，单键全局绑定不受影响", async () => {
    // 对照：让路必须以"焦点在该表面内"为条件，否则时间轴的 `s`（分割）等会整体失效。
    await mount();
    buildTypeAheadList(); // 存在但未聚焦

    press(document.body, "d");
    press(document.body, "Enter");
    expect(fired).toEqual(["pianoRoll.cycleDragDirection", "playback.stop"]);
});

test("DOM 焦点滞留在列表里、但活动表面是时间轴时，按键仍归时间轴", async () => {
    /*
     * 时间轴 / 参数编辑器刻意在 pointerdown 里 `preventDefault()` 自管焦点，点击它们
     * 之后 DOM 焦点会**滞留在上一次聚焦的元素**上（例如刚在文件列表里点过的行）。
     * 只按 target / activeElement 判断，时间轴的 `s`（分割）会被抢到文件列表里 ——
     * 用户点一下时间轴再按分割键，什么都不会发生。
     */
    await mount();
    const { row } = buildTypeAheadList();
    row.focus(); // 焦点确实在列表的行上
    setActiveSurfaceExplicit("timeline"); // ……但用户随后点的是时间轴

    press(row, "s");
    expect(fired).toEqual(["clip.split"]);
});

test("插件原生Ctrl+V只在对应view的keydown触发参数粘贴，keyup不再执行第二次",async()=>{
    const {store}=await mount();
    await act(async()=>{store.dispatch(setToolMode("select"));});
    setActiveSurfaceExplicit("pianoRoll");
    document.body.focus();
    const listeners=new Set<(event:{data:unknown})=>void>();
    const port:WebViewMessagePort={postMessage:()=>{},
        addEventListener:(_name,fn)=>listeners.add(fn),removeEventListener:(_name,fn)=>listeners.delete(fn)};
    const bridge=createPluginHost(port,{version:1,viewId:"keyboard-view"});
    const deliver=(viewId:string,type:string,repeat=false)=>{
        for(const fn of listeners) fn({data:{version:1,viewId,event:"plugin_keyboard",
            payload:{type,key:"v",ctrlKey:true,shiftKey:false,repeat}}});
    };
    try {
        deliver("other-view","keydown");expect(fired).toEqual([]);
        deliver("keyboard-view","keydown");expect(fired).toEqual(["pianoRoll.paste"]);
        deliver("keyboard-view","keyup");expect(fired).toEqual(["pianoRoll.paste"]);
        deliver("keyboard-view","keydown",true);expect(fired).toEqual(["pianoRoll.paste"]);
        const input=document.createElement("input");document.body.append(input);input.focus();
        deliver("keyboard-view","keydown");expect(fired).toEqual(["pianoRoll.paste"]);
        input.remove();
        fired.length=0;
        await act(async()=>{store.dispatch(setToolMode("draw"));});
        deliver("keyboard-view","keydown");expect(fired).toEqual(["clip.paste"]);
    } finally {bridge.dispose();}
});

/*
 * 激活键（Enter / Space）与弹出表面的归属。
 *
 * 【为什么必须有】与方向键那几条同源，但坏得更彻底：方向键至少有
 * `COMPOSITE_WIDGET_KEYS` 让路，**Enter 没有** —— 而裸 Enter 全局绑的是
 * `playback.stop`。于是"菜单开着按回车"的结果是**菜单项纹丝不动、播放却停了**：
 * 分发器在 window 捕获阶段命中并 `preventDefault` + `stopPropagation`，
 * `AppContextMenu` 的激活处理器看到 `defaultPrevented` 就让路。
 *
 * 这也是"测试全绿但功能不可用"的又一例：菜单的激活逻辑在 jsdom 单测里是对的
 * （那里没有全局分发器），只有真机实测才暴露。浏览器实测确认过：修复前回车既
 * 不展开子菜单也不激活菜单项，修复后两者都正常。
 */
test("弹出表面打开时，Enter 归它自己，不被全局绑定吞掉", async () => {
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    const item = document.createElement("button");
    item.setAttribute("role", "menuitem");
    item.tabIndex = 0;
    menu.append(item);
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });
    // 刻意不把焦点放进菜单：右键点开时焦点仍在被点的元素上。
    document.body.focus();

    const result = press(document.body, "Enter");

    // playback.stop 不得触发……
    expect(fired).toEqual([]);
    // ……而且事件必须保持"未被消费"，否则菜单自己的激活处理器收不到。
    expect(result.defaultPrevented).toBe(false);
});

test("Space 与 Enter 同等对待（菜单的另一个激活键）", async () => {
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });

    expect(press(document.body, " ").defaultPrevented).toBe(false);
});

test("没有弹出表面时，Enter 仍然走全局绑定", async () => {
    // 对照组：让路只在"弹出表面开着"时发生，否则裸 Enter 的停止播放就废了。
    await mount();

    press(document.body, "Enter");

    expect(fired).toEqual(["playback.stop"]);
});

test("关闭但**仍挂载**的菜单不得屏蔽 Enter", async () => {
    // 与方向键那条同构的回归守卫：把"存在"当"打开"会永久屏蔽全局 Enter。
    await mount();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    menu.setAttribute("data-state", "closed");
    document.body.append(menu);
    mounted.push(async () => {
        menu.remove();
    });

    press(document.body, "Enter");

    expect(fired).toEqual(["playback.stop"]);
});
