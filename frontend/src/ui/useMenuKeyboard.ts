/*
 * 弹出菜单的方向键导航。
 *
 * 【为什么需要它】本仓库的菜单分两族实现：`AppContextMenu`（声明式项列表，自带
 * 方向键、Home/End、Enter 激活）与**焦点导航**的表面（`AppAnchoredMenu`、
 * `AppSubMenu` 的子面板、以及若干手写菜单）。后者此前只处理 Escape —— 也就是说
 * 它们声明了 `role="menu"` / `role="menuitem"`，却没有实现菜单的键盘契约：
 * 方向键不动、Home/End 不动，键盘用户只能靠 Tab 逐个走。
 *
 * 【与 `AppContextMenu` 的行为对齐】语义完全一致：方向键在项之间循环、
 * Home/End 到两端、PageUp/PageDown 跨一页、打字跳项、Enter/Space 由 `<button>`
 * 原生处理。区别只是这里用**焦点**表示当前项（手写菜单的项本来就是真实
 * `<button>`，可以聚焦），而 `AppContextMenu` 用 `activeIndex` 表示。两者对用户
 * 是同一套手感 —— 而且现在连"打字跳项"这种细节也一致。
 *
 * 【为什么监听 document 而不是容器】右键菜单是"右键点开、焦点还在被点的元素上"
 * —— 焦点根本不在菜单里，监听容器就收不到第一个方向键。这与 `AppContextMenu`
 * 的捕获阶段 document 监听是同一个理由。
 */
import { useEffect, useLayoutEffect, useRef } from "react";
import type { RefObject } from "react";

import { menuItemHeightOf, revealMenuItem } from "./menuScrollDom";
import { computeMenuPageItemCount } from "./menuScrollModel";
import {
    EMPTY_MENU_TYPEAHEAD,
    MENU_ITEM_SELECTOR,
    matchMenuTypeahead,
    menuItemTypeaheadText,
    menuItemsOf,
    menuTypeaheadChar,
    pushMenuTypeahead,
    type MenuTypeaheadState,
} from "./menuTypeahead";

export { MENU_ITEM_SELECTOR, menuItemsOf };

/*
 * 已挂载菜单的栈。
 *
 * 【为什么需要它】同一时刻可以有多个菜单表面挂载：菜单里展开的子面板就是第二层
 * `role="menu"`。若每个都监听 document，一次 ArrowDown 会被两层各消费一次，
 * 焦点落在哪一层取决于监听顺序 —— 而"焦点该去哪"是有明确答案的，不该由注册顺序
 * 决定。
 *
 * 【判定规则】见 `shouldRespond`：谁**含**焦点谁响应；都不含时由最上层响应。
 */
const menuStack: HTMLElement[] = [];

/**
 * 本次按键该由哪个容器消费。
 *
 * 1. 焦点在某个容器内 → 由**最内层**那个含焦点的容器响应（子面板优先于外层菜单，
 *    因为子面板在外层菜单的 DOM 子树里）。
 * 2. 焦点不在任何容器内（右键点开的初始状态，焦点还在被点的元素上）→ 由**栈顶**
 *    响应，也就是最后打开的那一层。
 */
function shouldRespond(container: HTMLElement, focused: Element | null): boolean {
    const holds = focused instanceof HTMLElement && container.contains(focused);
    if (!holds) return menuStack[menuStack.length - 1] === container;
    return !menuStack.some(
        (other) =>
            other !== container &&
            container.contains(other) &&
            focused instanceof Node &&
            other.contains(focused),
    );
}

/**
 * 焦点落在这些元素上时，方向键属于它们自己（文本光标、滑杆取值）。
 *
 * 导出供同层的全局菜单（`AppContextMenu`）共用同一判定，两处口径必须一致：
 * 一处加守卫另一处不加，会出现"输入框里打字时这条菜单还跟着动"的分裂行为。
 */
export function ownsArrowKeys(element: Element | null): boolean {
    if (!(element instanceof HTMLElement)) return false;
    if (element.isContentEditable) return true;
    // 【为什么还要认角色】Radix 的滑块头是 `<span role="slider">`（不是 `<input>`），
    // 它用上下键调值；漏掉它，菜单的方向键会把滑块的上下键抢走 —— 节拍器音量条
    // 一旦放进菜单就会"键盘调不动"。数字输入同理（`spinbutton`）。
    if (element.closest('[role="slider"], [role="spinbutton"]')) return true;
    return ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

export interface MenuKeyboardOptions {
    /**
     * 请求退出本层（`ArrowLeft`）。
     *
     * 只有**子面板**会用到它：在外层菜单里按左键没有去处，而在子面板里按左键应当
     * 收起子面板、回到触发项 —— 那是原生菜单的行为。
     */
    onDismiss?: () => void;
    /** 本层是不是子面板（决定 `ArrowLeft` 是否有意义）。 */
    nested?: boolean;
}

/**
 * 给一个 `role="menu"` 容器接上键盘导航。
 *
 * @param containerRef 菜单容器（其内部所有 `[role="menuitem"]` 参与导航）。
 * @param active 菜单是否已挂载。传 `false` 可临时停用（例如重命名模式）。
 * @param options 退出回调与"是否子面板"。
 */
export function useMenuKeyboard(
    containerRef: RefObject<HTMLElement | null>,
    active = true,
    options?: MenuKeyboardOptions,
): void {
    /*
     * 回调与标志走 ref，**不进 effect 依赖**。
     *
     * 调用方普遍写成内联箭头函数（`onDismiss={() => setOpen(false)}`），它的身份
     * 每次渲染都变。把它放进依赖会让 effect 每渲染一次就重新入栈出栈 —— 而
     * `menuStack` 的顺序正是"哪一层该响应按键"的判据。
     */
    const dismissRef = useRef<MenuKeyboardOptions["onDismiss"]>(undefined);
    const nestedRef = useRef(false);
    useLayoutEffect(() => {
        dismissRef.current = options?.onDismiss;
        nestedRef.current = options?.nested ?? false;
    });
    const typeaheadRef = useRef<MenuTypeaheadState>(EMPTY_MENU_TYPEAHEAD);

    useEffect(() => {
        if (!active) return;
        const container = containerRef.current;
        if (!container) return;
        menuStack.push(container);

        /** 聚焦并让高亮完整可见（提示行会盖住边缘，`scrollIntoView` 不知道这件事）。 */
        const focusItem = (item: HTMLElement) => {
            item.focus();
            revealMenuItem(container, item);
        };

        /*
         * 用箭头函数而不是 `function` 声明：函数声明会被提升，TypeScript 因此
         * 不把上面 `if (!container) return` 的收窄带进闭包，`container` 在函数体
         * 里会退回 `HTMLElement | null`。赋值给 `const` 的箭头函数不提升，收窄保留。
         */
        const onKeyDown = (event: KeyboardEvent) => {
            // 已被内层处理器消费（例如滑杆的左右键）就不再插手。
            if (event.defaultPrevented) return;
            const focused = document.activeElement;
            if (!shouldRespond(container, focused)) return;
            // 菜单内的输入框/滑杆自己管方向键：正在输入时按 ↓ 应该移动光标，
            // 而不是跳到下一个菜单项。
            if (ownsArrowKeys(focused)) return;

            const items = menuItemsOf(container);
            if (items.length === 0) return;

            // -1 = 焦点不在菜单里（刚右键点开时就是这样）。
            const current = focused instanceof HTMLElement ? items.indexOf(focused) : -1;
            const pageCount = () =>
                computeMenuPageItemCount({
                    clientHeight: container.clientHeight,
                    itemHeight: menuItemHeightOf(container),
                });

            let next: number;
            switch (event.key) {
                case "ArrowDown":
                    next = current < 0 ? 0 : (current + 1) % items.length;
                    break;
                case "ArrowUp":
                    next =
                        current < 0
                            ? items.length - 1
                            : (current - 1 + items.length) % items.length;
                    break;
                case "Home":
                    next = 0;
                    break;
                case "End":
                    next = items.length - 1;
                    break;
                /*
                 * PageUp/PageDown = **移动选中项一页**，而不是只滚。
                 *
                 * 【为什么不只滚】只滚会让高亮停在屏幕外 —— 用户按了 PageDown，
                 * 界面滚了，但"当前项"看不见，下一次 Enter 会激活一个他没看见的
                 * 条目。菜单的键盘模型里"当前项"必须永远可见。
                 *
                 * 【为什么到端点就停而不是循环】翻页的语义是"再走一屏"，从末尾
                 * 循环回开头会让"翻到底"变成一个没有尽头的动作。
                 */
                case "PageDown": {
                    event.preventDefault();
                    const page = pageCount();
                    focusItem(items[current < 0 ? 0 : Math.min(items.length - 1, current + page)]);
                    return;
                }
                case "PageUp": {
                    event.preventDefault();
                    const page = pageCount();
                    focusItem(items[current < 0 ? items.length - 1 : Math.max(0, current - page)]);
                    return;
                }
                /*
                 * ArrowRight = 展开子菜单；ArrowLeft = 收起本层（仅子面板）。
                 *
                 * 【为什么 ArrowRight 是"点一下触发项"而不是自己开面板】展开状态住在
                 * `AppSubMenu` 内部，父层够不着。`.click()` 与鼠标走同一条路径，
                 * 两种入口因此不可能分叉。
                 *
                 * 【进去要再按一次 ↓】面板打开后焦点仍在触发项上；下一次 ↓ 由面板
                 * （`menuStack` 栈顶）接管并把焦点交给第一项。这与鼠标悬停展开后的
                 * 行为完全一致，不是特例。
                 */
                case "ArrowRight": {
                    if (current < 0) return;
                    const item = items[current];
                    if (item.getAttribute("aria-haspopup") !== "menu") return;
                    event.preventDefault();
                    if (item.getAttribute("aria-expanded") === "true") {
                        // 已经展开：把焦点交给面板的第一项（原生菜单的行为）。
                        // 面板此刻已经挂载，是触发项那个包装盒的子元素。
                        const panel =
                            item.parentElement?.querySelector<HTMLElement>('[role="menu"]');
                        const first = panel ? menuItemsOf(panel)[0] : null;
                        if (first) focusItem(first);
                        return;
                    }
                    item.click();
                    return;
                }
                case "ArrowLeft": {
                    /*
                     * 两种情况，顺序要紧：
                     *
                     * 1. 焦点停在**已展开的触发项**上 → 左键收起它（与右键展开对称）。
                     *    不处理这一条时，键盘用户用 ArrowRight 展开子菜单后没有任何
                     *    办法把它收回去（焦点并没有进入面板）。
                     * 2. 焦点在**子面板内部** → 左键退出本层。
                     *
                     * 焦点在外层菜单、而某个子面板恰好开着时，第 1 条会命中触发项；
                     * 若命中的是普通项，则什么都不做 —— 外层菜单没有"左边"可去。
                     */
                    if (current >= 0) {
                        const item = items[current];
                        if (
                            item.getAttribute("aria-haspopup") === "menu" &&
                            item.getAttribute("aria-expanded") === "true"
                        ) {
                            event.preventDefault();
                            item.click();
                            return;
                        }
                    }
                    if (!nestedRef.current) return;
                    // 焦点不在本层里时，左键属于**外层**（由外层的处理器裁决）。
                    if (!(focused instanceof HTMLElement) || !container.contains(focused)) return;
                    event.preventDefault();
                    dismissRef.current?.();
                    return;
                }
                case "Escape":
                    // Escape 由**表面自己**处理（dismiss 层栈保证只关最内层）。
                    // 在这里顺手关一层会让同一个按键关掉两层。
                    return;
                default: {
                    // 打字跳项。
                    const char = menuTypeaheadChar(event.key, event);
                    if (char === null) return;
                    const state = pushMenuTypeahead(typeaheadRef.current, char, performance.now());
                    typeaheadRef.current = state;
                    const match = matchMenuTypeahead(
                        items.map(menuItemTypeaheadText),
                        state.query,
                        current,
                    );
                    if (match < 0) return;
                    event.preventDefault();
                    focusItem(items[match]);
                    return;
                }
            }

            event.preventDefault();
            focusItem(items[next]);
        };

        document.addEventListener("keydown", onKeyDown, true);
        return () => {
            document.removeEventListener("keydown", onKeyDown, true);
            const at = menuStack.lastIndexOf(container);
            if (at >= 0) menuStack.splice(at, 1);
        };
    }, [active, containerRef]);
}
