/**
 * 上下文菜单原语 —— 全应用手写右键菜单的**唯一**实现。
 *
 * 【为什么手写而不是用 Radix ContextMenu】这是既有决策，保留：时间轴与
 * 记事本的菜单需要与画布的手势系统协同（例如指针捕获、右键拖拽守卫），
 * Radix 的 portal 菜单会打断这条链路。项目已把该决策写在 `DockTabMenu` 头部。
 *
 * 【它解决的三个问题】
 *
 * 1. **漂移的菜单项外观**。审查发现同一个"菜单项"在 6 个文件里有 6 套
 *    class：`text-[12px]` / `text-qt-md` / `text-qt-xs`，`hover:bg-qt-button-hover` /
 *    `hover:bg-qt-hover` / `hover:bg-qt-highlight hover:text-white`，
 *    以及 `py-1` / `py-1.5` / `py-2` 三种行内边距。
 *
 * 2. **缺失的键盘可达性**。12 个手写菜单**没有一个有方向键导航**，只有 Esc。
 *    键盘用户打开右键菜单后只能再按 Esc 关掉，无法选择任何一项。
 *
 * 3. **各不相同的定位与层级**。有的按估算尺寸夹紧、有的不夹紧，
 *    z-index 从 `z-50` 到 `z-[9999]` 有 4 档（其中两处反而高过对话框）。
 *
 * 本组件把这三件事各做一次：单一 class 来源、完整键盘模型、按实测尺寸夹紧 +
 * `--qt-z-menu` 层级。
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { EDGE_GAP, clampAxisPosition } from "../components/appTooltipPosition";
import { cx } from "./cx";

export interface AppMenuItemSpec {
    key: string;
    label: ReactNode;
    onSelect: () => void;
    /** 展示用快捷键文本（不参与绑定，与 `data-tooltip` 同源）。 */
    shortcut?: string;
    /** 破坏性操作：悬停变红底红字。 */
    danger?: boolean;
    disabled?: boolean;
    /** 该项上方加一条分隔线，用于视觉分组。 */
    separatorBefore?: boolean;
    /** 右侧勾选标记（用于"当前选中项"这类菜单）。 */
    checked?: boolean;
}

export interface AppContextMenuProps {
    /** 视口坐标（`clientX` / `clientY`）。 */
    x: number;
    y: number;
    items: AppMenuItemSpec[];
    onClose: () => void;
    /** 最小宽度，默认 190px。各面板历史取值 140–220，新代码请省略以用默认值。 */
    minWidth?: number;
    /** 无障碍名称：菜单是弹出表面，需要有可读名称。 */
    ariaLabel?: string;
    /**
     * 是否标记为「时间轴浮动菜单」（`data-hs-floating-menu="1"`）。
     *
     * 这是时间轴侧的既有契约，有三个独立读取方，缺了它会静默失去豁免：
     *   - `AppTooltip`：悬停抑制（菜单打开时不该冒提示气泡）；
     *   - `TimelinePanel`：内联编辑器的失焦提交判断（点菜单不应提交输入框）；
     *   - `ClipHeader`：角标编辑的提交守卫。
     *
     * 因此凡是挂在时间轴/剪辑上的菜单都要显式打开它；普通面板菜单不需要。
     */
    floating?: boolean;
}

const ITEM_BASE =
    "hs-type-body flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left outline-none";

/**
 * 上下文菜单。
 *
 * @example
 * {menu ? (
 *   <AppContextMenu
 *     x={menu.x}
 *     y={menu.y}
 *     ariaLabel={t("ctx_clip")}
 *     items={[
 *       { key: "cut", label: t("cut"), shortcut: "Ctrl+X", onSelect: cut },
 *       { key: "del", label: t("delete"), danger: true, separatorBefore: true, onSelect: del },
 *     ]}
 *     onClose={() => setMenu(null)}
 *   />
 * ) : null}
 */
export function AppContextMenu({
    x,
    y,
    items,
    onClose,
    minWidth = 190,
    ariaLabel,
    floating = false,
}: AppContextMenuProps) {
    const ref = useRef<HTMLDivElement | null>(null);
    const [position, setPosition] = useState<{ left: number; top: number; ready: boolean }>({
        left: x,
        top: y,
        ready: false,
    });
    /** 键盘焦点所在的下标（指向**可选项**，跳过禁用项与分隔符）。 */
    const [activeIndex, setActiveIndex] = useState(-1);

    /**
     * 打开菜单前的焦点元素。
     *
     * 【为什么需要】菜单是弹出表面：关闭后若不归还焦点，键盘用户会被丢回
     * `<body>`，下一次 Tab 从文档头重新开始 —— 对右键菜单而言"关闭"等于"迷路"。
     * 卸载时归还；触发者可能已随菜单一起消失（例如菜单项删掉了它），故先查
     * `isConnected`，不存在就什么都不做。
     */
    const openerRef = useRef<HTMLElement | null>(null);
    useEffect(() => {
        openerRef.current =
            document.activeElement instanceof HTMLElement ? document.activeElement : null;
        return () => {
            const opener = openerRef.current;
            if (opener?.isConnected) opener.focus();
        };
    }, []);

    /** 可被键盘选中的项下标（禁用项不参与）。 */
    const selectableIndexes = useMemo(
        () =>
            items.reduce<number[]>(
                (acc, item, index) => (item.disabled ? acc : [...acc, index]),
                [],
            ),
        [items],
    );

    /**
     * 按实测尺寸夹紧。
     *
     * 【为什么推翻了原先"固定估算高度"的决策】被替换掉的 `NotebookContextMenu`
     * 里有一段注释，说明它刻意用估算而非测量，理由是"在 layout effect 里同步
     * setState 会触发级联渲染（React Compiler 会告警）"。那条理由针对的是**告警**，
     * 不是正确性；而估算的代价是菜单一旦出现换行项（长标签、勾选标记、快捷键列）
     * 就会被裁掉或跑出视口 —— 用户看到的菜单缺一项，且没有任何报错。
     *
     * 现在统一为测量：`useLayoutEffect` 在**绘制前**完成，因此没有视觉闪动；
     * 多付一次渲染的代价远小于"菜单被裁"。同时消除了同一仓库里两种做法并存
     * （本组件测量、`DockTabMenu` 估算），后者也已改为测量。
     */
    useLayoutEffect(() => {
        const el = ref.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        setPosition({
            left: clampAxisPosition(x, rect.width, window.innerWidth, 0, EDGE_GAP),
            top: clampAxisPosition(y, rect.height, window.innerHeight, 0, EDGE_GAP),
            ready: true,
        });
    }, [x, y, items.length]);

    const step = useCallback(
        (delta: 1 | -1) => {
            if (selectableIndexes.length === 0) return;
            setActiveIndex((current) => {
                const at = selectableIndexes.indexOf(current);
                if (at === -1)
                    return delta === 1 ? selectableIndexes[0] : selectableIndexes.at(-1)!;
                const next = (at + delta + selectableIndexes.length) % selectableIndexes.length;
                return selectableIndexes[next];
            });
        },
        [selectableIndexes],
    );

    /**
     * 全局监听：Esc 关闭、外部指针按下关闭、方向键导航。
     *
     * 捕获阶段监听：编辑器/画布自身的 `pointerdown` 会先改变选择，
     * 若用冒泡阶段，菜单可能先关闭又被下层重新打开。与既有
     * `NotebookContextMenu` 的处理一致。
     */
    useEffect(() => {
        function onPointerDown(event: PointerEvent) {
            if (ref.current && !ref.current.contains(event.target as Node)) onClose();
        }
        function onKeyDown(event: KeyboardEvent) {
            switch (event.key) {
                case "Escape":
                    event.preventDefault();
                    onClose();
                    return;
                case "ArrowDown":
                    event.preventDefault();
                    step(1);
                    return;
                case "ArrowUp":
                    event.preventDefault();
                    step(-1);
                    return;
                case "Home":
                    event.preventDefault();
                    if (selectableIndexes.length) setActiveIndex(selectableIndexes[0]);
                    return;
                case "End":
                    event.preventDefault();
                    if (selectableIndexes.length) setActiveIndex(selectableIndexes.at(-1)!);
                    return;
                default:
                    return;
            }
        }
        document.addEventListener("pointerdown", onPointerDown, true);
        document.addEventListener("keydown", onKeyDown, true);
        return () => {
            document.removeEventListener("pointerdown", onPointerDown, true);
            document.removeEventListener("keydown", onKeyDown, true);
        };
    }, [onClose, step, selectableIndexes]);

    /** 键盘激活：与鼠标点击走同一条路径，保证行为不分叉。 */
    useEffect(() => {
        function onKeyActivate(event: KeyboardEvent) {
            if (event.key !== "Enter" && event.key !== " ") return;
            const index = activeIndex;
            if (index < 0) return;
            const item = items[index];
            if (!item || item.disabled) return;
            event.preventDefault();
            item.onSelect();
            onClose();
        }
        document.addEventListener("keydown", onKeyActivate, true);
        return () => document.removeEventListener("keydown", onKeyActivate, true);
    }, [activeIndex, items, onClose]);

    return (
        <div
            ref={ref}
            role="menu"
            aria-label={ariaLabel}
            data-hs-context-menu="1"
            data-hs-floating-menu={floating ? "1" : undefined}
            className={cx(
                "fixed z-qt-menu rounded border border-qt-border bg-qt-window py-1 text-qt-text shadow-lg",
            )}
            style={{
                left: position.left,
                top: position.top,
                minWidth,
                visibility: position.ready ? undefined : "hidden",
            }}
            // 阻止冒泡到 document 的 pointerdown 关闭逻辑：面板自身的容器
            // 通常也监听 pointerdown 来清除选择，菜单内点击不应触发它。
            onPointerDown={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
        >
            {items.map((item, index) => (
                <AppContextMenuItem
                    key={item.key}
                    item={item}
                    active={index === activeIndex}
                    onHover={() => setActiveIndex(item.disabled ? -1 : index)}
                    onSelect={() => {
                        item.onSelect();
                        onClose();
                    }}
                />
            ))}
        </div>
    );
}

function AppContextMenuItem({
    item,
    active,
    onHover,
    onSelect,
}: {
    item: AppMenuItemSpec;
    active: boolean;
    onHover: () => void;
    onSelect: () => void;
}) {
    const tone = item.disabled
        ? "cursor-default text-qt-text-muted"
        : item.danger
          ? "hover:bg-qt-danger-bg hover:text-qt-danger-text"
          : "hover:bg-qt-hover";

    return (
        <button
            type="button"
            role="menuitem"
            disabled={item.disabled}
            aria-checked={item.checked}
            className={cx(
                ITEM_BASE,
                tone,
                item.separatorBefore && "mt-1 border-t border-qt-border pt-2.5",
                // 键盘焦点环与菜单自身的边框会叠在一起，改用底色表达高亮
                active && !item.disabled && "bg-qt-hover",
            )}
            style={{ paddingLeft: "var(--qt-space-5)", paddingRight: "var(--qt-space-5)" }}
            onMouseEnter={onHover}
            onClick={() => {
                if (item.disabled) return;
                onSelect();
            }}
        >
            <span className="truncate">{item.label}</span>
            <span className="flex shrink-0 items-center gap-2">
                {item.checked ? <span aria-hidden>✓</span> : null}
                {item.shortcut ? <span className="text-qt-text-muted">{item.shortcut}</span> : null}
            </span>
        </button>
    );
}
