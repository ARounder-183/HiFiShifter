/**
 * 菜单栏的一个下拉项：触发器 + 锚定菜单的**三件套**（anchor ref / menu ref / open 状态）。
 *
 * 【为什么必须有这个小组件】菜单栏有 8 个下拉。让每个作者各写一遍
 * `useRef` + `useState` + `<AppAnchoredMenu>`，就是 8 份"照着旁边抄"的代码 ——
 * 而本仓反复吃过这个亏（菜单项 class 曾经在 6 个文件里长出 6 套）。收敛到一处
 * 之后，"菜单栏的下拉怎么开、怎么关、键盘怎么进去"只有一个答案。
 *
 * 【为什么 open 是受控的】同一时刻只能有一张下拉展开。若每个下拉各持一份
 * `useState`，用键盘打开第二个时第一个不会关（指针打开时靠"点外面关闭"侥幸正确，
 * 键盘路径没有 pointerdown）。把"当前展开的是哪一个"收在菜单栏自己的一个 state 里，
 * 这条约束从结构上成立。
 */
import { useRef } from "react";
import type { ReactNode } from "react";

import { cx } from "./cx";
import { AppAnchoredMenu, type AppMenuItemSpec } from "./Menu";

export interface AppMenuBarItemProps {
    /** 触发器上显示的内容（图标 + 文本）。 */
    children: ReactNode;
    /** 菜单的无障碍名称（省略则用触发器的可读文本）。 */
    ariaLabel?: string;
    /** 是否展开（受控）。 */
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 声明式条目；与 `menuChildren` 二选一。 */
    items?: AppMenuItemSpec[];
    /** 手写内容（需要内联控件 / 非条目结构的菜单）；与 `items` 二选一。 */
    menuChildren?: ReactNode;
    /** 触发器追加类。默认值就是菜单栏触发器的既有外观。 */
    triggerClassName?: string;
    /** 菜单内容层的追加类。 */
    contentClassName?: string;
    /** 与触发器的水平对齐。菜单栏右端的下拉应当右对齐。 */
    align?: "start" | "end";
}

/** 菜单栏触发器的默认外观（与统一前逐字一致，只是元素从 `<span>` 换成了真按钮）。 */
const TRIGGER_CLASS =
    "shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white";

export function AppMenuBarItem({
    children,
    ariaLabel,
    open,
    onOpenChange,
    items,
    menuChildren,
    triggerClassName,
    contentClassName,
    align = "start",
}: AppMenuBarItemProps) {
    const anchorRef = useRef<HTMLButtonElement | null>(null);
    const menuRef = useRef<HTMLDivElement | null>(null);
    /**
     * 本次展开是不是**键盘**发起的。
     *
     * 【为什么要区分】鼠标点开时焦点留在触发器上（与 VS Code / Windows 菜单栏
     * 一致）；键盘打开时焦点必须进菜单，否则方向键无处可去 —— 那是 `AppAnchoredMenu`
     * 的方向键监听（按 `document.activeElement` 判归属）之外的一整套问题。
     * 这个值在 `onOpenChange(true)` **之前**写，因此展开后的第一次渲染就读得到。
     */
    const keyboardOpenRef = useRef(false);

    const openWith = (byKeyboard: boolean) => {
        keyboardOpenRef.current = byKeyboard;
        onOpenChange(true);
    };

    return (
        <>
            <button
                ref={anchorRef}
                type="button"
                // 【必须是真按钮 + aria-haspopup】触摸形态用
                // `[aria-haspopup="menu"]` 把菜单栏触发器抬到 44px（见 index.css）；
                // 声明 `aria-expanded` 之后读屏才知道这一项是展开着的。
                aria-haspopup="menu"
                aria-expanded={open}
                className={cx(TRIGGER_CLASS, triggerClassName)}
                onClick={() => {
                    // 鼠标路径：清掉上一次"键盘打开"留下的标记，否则用鼠标打开
                    // 也会把焦点搬进菜单（与 VS Code / Windows 菜单栏的手感不符）。
                    keyboardOpenRef.current = false;
                    onOpenChange(!open);
                }}
                onKeyDown={(event) => {
                    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                        event.preventDefault();
                        openWith(true);
                        return;
                    }
                    // Enter / 空格在原生 `<button>` 上会产生 click，click 会走到
                    // 上面的 toggle —— 但那条路径把焦点留在触发器上。菜单栏的约定是
                    // "键盘打开 ⇒ 焦点进菜单"，所以这里显式接管。
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openWith(true);
                    }
                }}
            >
                {children}
            </button>
            <AppAnchoredMenu
                open={open}
                anchorRef={anchorRef}
                menuRef={menuRef}
                onClose={() => onOpenChange(false)}
                align={align}
                autoFocus={keyboardOpenRef.current}
                ariaLabel={ariaLabel}
                contentClassName={contentClassName}
                items={items}
            >
                {menuChildren}
            </AppAnchoredMenu>
        </>
    );
}
