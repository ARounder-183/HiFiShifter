/*
 * 弹出菜单的方向键导航。
 *
 * 【为什么需要它】本仓库有两套菜单：`AppContextMenu`（声明式项列表，自带方向键、
 * Home/End、Enter 激活）与若干**手写菜单**（需要内联输入框、滑杆、子菜单等
 * `AppContextMenu` 装不下的内容）。手写的那几个此前只处理 Escape —— 也就是说
 * 它们声明了 `role="menu"` / `role="menuitem"`，却没有实现菜单的键盘契约：
 * 方向键不动、Home/End 不动，键盘用户只能靠 Tab 逐个走。
 *
 * 【与 `AppContextMenu` 的行为对齐】语义完全一致：方向键在项之间循环、
 * Home/End 到两端、Enter/Space 由 `<button>` 原生处理。区别只是这里用**焦点**
 * 表示当前项（手写菜单的项本来就是真实 `<button>`，可以聚焦），而
 * `AppContextMenu` 用 `activeIndex` 表示。两者对用户是同一套手感。
 *
 * 【为什么监听 document 而不是容器】右键菜单是"右键点开、焦点还在被点的元素上"
 * —— 焦点根本不在菜单里，监听容器就收不到第一个方向键。这与 `AppContextMenu`
 * 的捕获阶段 document 监听是同一个理由。
 */
import { useEffect } from "react";
import type { RefObject } from "react";

/** 可导航的项：真实禁用（`disabled`）的项不参与，与鼠标行为一致。 */
const ITEM_SELECTOR = '[role="menuitem"]:not([disabled])';

/**
 * 本层菜单的项。
 *
 * 【为什么要排除子菜单的项】子菜单（`ClipContextMenu` 的子面板）在 DOM 上位于
 * 外层菜单的子树里，`querySelectorAll` 会把两层的项一起捞出来 —— 于是外层菜单按
 * 方向键会一路走进子菜单，而子菜单是**另一个**菜单表面，由它自己的钩子导航。
 * 用 `closest('[role="menu"]')` 判断"这一项属于哪一层"，只留属于本层的。
 */
function itemsOf(container: HTMLElement): HTMLElement[] {
    return Array.from(container.querySelectorAll<HTMLElement>(ITEM_SELECTOR)).filter(
        (item) => item.closest('[role="menu"]') === container,
    );
}

/*
 * 已挂载菜单的栈。
 *
 * 【为什么需要它】同一时刻可以有多个菜单表面挂载：`ClipContextMenu` 就同时有
 * 外层菜单和它展开的子面板，两者都是 `role="menu"`。若每个都监听 document，
 * 一次 ArrowDown 会被两层各消费一次，焦点落在哪一层取决于监听顺序 —— 而
 * "焦点该去哪"是有明确答案的，不该由注册顺序决定。
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

/** 焦点落在这些元素上时，方向键属于它们自己（文本光标、滑杆取值）。 */
function ownsArrowKeys(element: Element | null): boolean {
    if (!(element instanceof HTMLElement)) return false;
    if (element.isContentEditable) return true;
    return ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

/**
 * 给一个 `role="menu"` 容器接上方向键导航。
 *
 * @param containerRef 菜单容器（其内部所有 `[role="menuitem"]` 参与导航）。
 * @param active 菜单是否已挂载。传 `false` 可临时停用（例如重命名模式）。
 */
export function useMenuKeyboard(containerRef: RefObject<HTMLElement | null>, active = true): void {
    useEffect(() => {
        if (!active) return;
        const container = containerRef.current;
        if (!container) return;
        menuStack.push(container);

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

            const items = itemsOf(container);
            if (items.length === 0) return;

            // -1 = 焦点不在菜单里（刚右键点开时就是这样）。
            const current = focused instanceof HTMLElement ? items.indexOf(focused) : -1;

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
                default:
                    return;
            }

            event.preventDefault();
            items[next].focus();
        };

        document.addEventListener("keydown", onKeyDown, true);
        return () => {
            document.removeEventListener("keydown", onKeyDown, true);
            const at = menuStack.lastIndexOf(container);
            if (at >= 0) menuStack.splice(at, 1);
        };
    }, [active, containerRef]);
}
