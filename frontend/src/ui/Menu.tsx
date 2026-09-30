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
 *    class：`text-qt-sm` / `text-qt-md` / `text-qt-xs`，`hover:bg-qt-button-hover` /
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
import { useMenuKeyboard } from "./useMenuKeyboard";

export interface AppMenuItemSpec {
    key: string;
    label: ReactNode;
    /** 选择回调。`heading: true` 的标题行可省略。 */
    onSelect?: () => void;
    /** 展示用快捷键文本（不参与绑定，与 `data-tooltip` 同源）。 */
    shortcut?: string;
    /** 破坏性操作：悬停变红底红字。 */
    danger?: boolean;
    disabled?: boolean;
    /** 该项上方加一条分隔线，用于视觉分组。 */
    separatorBefore?: boolean;
    /** 右侧勾选标记（用于"当前选中项"这类菜单）。 */
    checked?: boolean;
    /** 条目左侧图标（take 操作、菜单按钮等）。 */
    icon?: ReactNode;
    /**
     * 悬停 / 禁用原因解释（走项目自定义 tooltip 通道）。收编自
     * `ClipContextMenu` 的 `title` prop —— "为什么点不了"应当可见。
     */
    tooltip?: string;
    /**
     * 分组标题行：渲染为不可选中的小标题（大写弱化色），不参与键盘导航。
     * 收编自 ActionBar 录音菜单的手写分组行 —— 平面菜单也常有分段需求。
     */
    heading?: boolean;
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
     * 顶部自定义区：渲染在条目列表之上、参与同一次定位测量。
     * `DockTabMenu` 的行内重命名、ActionBar 录音菜单的分组头由此承载。
     * 注意：这里是**非条目**内容，不参与键盘导航 —— 需要可聚焦控件的
     * 内容（如重命名输入框）自己处理焦点与 Enter/Esc。
     */
    header?: ReactNode;
    /**
     * 底部自定义区：渲染在条目列表之下、参与同一次定位测量。
     *
     * 【为什么需要它】`items` 是纯数据（`AppMenuItemSpec`），只表达得了扁平的
     * 一行。二级子菜单（`AppSubMenu`）与内联输入框这类需要真实 React 节点的
     * 内容装不进去，而"把它变成数据"意味着给 `AppMenuItemSpec` 加一棵递归树，
     * 那会波及全部 20 多处消费者 —— 只为了一处需要子菜单。
     *
     * 与 `header` 同构：这里是**非条目**内容，不参与 `activeIndex` 键盘导航；
     * `AppSubMenu` 自带 `useMenuKeyboard`，各层按 `role="menu"` 分层导航。
     */
    extraItems?: ReactNode;
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
    header,
    extraItems,
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

    /** 可被键盘选中的项下标（禁用项与标题行不参与）。 */
    const selectableIndexes = useMemo(
        () =>
            items.reduce<number[]>(
                (acc, item, index) => (item.disabled || item.heading ? acc : [...acc, index]),
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
     * 【pointerdown 用捕获阶段】编辑器/画布自身的 `pointerdown` 会先改变选择，
     * 若用冒泡阶段，菜单可能先关闭又被下层重新打开。
     *
     * 【keydown 用冒泡阶段】header 槽里的可交互内容（如 DockTabMenu 的重命名
     * 输入框）需要先吃到 Enter/Esc：监听在冒泡阶段时，目标元素的处理先于
     * 本监听，输入框 `stopPropagation()` 即可优先。全局快捷键分发器挂在
     * window 捕获层，无论如何都先于这里 —— 时序不受影响。
     */
    useEffect(() => {
        function onPointerDown(event: PointerEvent) {
            if (ref.current && !ref.current.contains(event.target as Node)) onClose();
        }
        function onKeyDown(event: KeyboardEvent) {
            // 内层表面（`AppSubMenu` 的子面板）用捕获阶段先处理并 preventDefault；
            // 这里必须让路，否则外层高亮会跟着内层一起动，出现两处高亮。
            if (event.defaultPrevented) return;
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
        document.addEventListener("keydown", onKeyDown);
        return () => {
            document.removeEventListener("pointerdown", onPointerDown, true);
            document.removeEventListener("keydown", onKeyDown);
        };
    }, [onClose, step, selectableIndexes]);

    /** 键盘激活：与鼠标点击走同一条路径，保证行为不分叉。（冒泡阶段，理由同上） */
    useEffect(() => {
        function onKeyActivate(event: KeyboardEvent) {
            if (event.key !== "Enter" && event.key !== " ") return;
            const index = activeIndex;
            if (index < 0) return;
            const item = items[index];
            if (!item || item.disabled) return;
            event.preventDefault();
            item.onSelect?.();
            onClose();
        }
        document.addEventListener("keydown", onKeyActivate);
        return () => document.removeEventListener("keydown", onKeyActivate);
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
            {header ? <div className="border-b border-qt-border px-2 py-1">{header}</div> : null}
            {extraItems}
            {items.map((item, index) => (
                <AppContextMenuItem
                    key={item.key}
                    item={item}
                    active={index === activeIndex}
                    onHover={() => setActiveIndex(item.disabled ? -1 : index)}
                    onSelect={() => {
                        item.onSelect?.();
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
    if (item.heading) {
        return (
            <div
                className="hs-type-caption px-3 py-1 font-semibold uppercase tracking-wide"
                style={{ paddingLeft: "var(--qt-space-5)" }}
            >
                {item.label}
            </div>
        );
    }
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
            data-tooltip={item.tooltip}
            onMouseEnter={onHover}
            onClick={() => {
                if (item.disabled) return;
                onSelect();
            }}
        >
            <span className="flex min-w-0 items-center gap-2">
                {item.icon ? (
                    <span className="shrink-0" aria-hidden>
                        {item.icon}
                    </span>
                ) : null}
                <span className="truncate">{item.label}</span>
            </span>
            <span className="flex shrink-0 items-center gap-2">
                {item.checked ? <span aria-hidden>✓</span> : null}
                {item.shortcut ? <span className="text-qt-text-muted">{item.shortcut}</span> : null}
            </span>
        </button>
    );
}

export interface AppSubMenuProps {
    /** 触发项文案。 */
    label: ReactNode;
    /** 触发项右侧的计数角标（如 Take 数量）。 */
    badge?: string;
    disabled?: boolean;
    /**
     * 子面板内容。
     *
     * 【为什么是 children 而不是 items】子面板里经常要放 `AppContextMenu` 装不下的
     * 东西 —— 带尾随按钮的行、滑杆、内联输入框。这与本文件顶部"手写菜单存在
     * 的理由"是同一条。
     */
    children: ReactNode;
}

/**
 * 一级菜单里的二级子菜单（悬停或点击展开）。
 *
 * 【为什么收进本文件】它原本是 `ClipContextMenu` 的局部组件，而"菜单里要有
 * 子菜单"并不是 Clip 特有的需求 —— 颤音预设列表同样需要（十几个预设平铺会把
 * 菜单撑得比屏幕高）。放进原语层，第二次需要它的人不必再写一遍定位夹紧与
 * 键盘导航，也顺带让 `useMenuKeyboard` 的嵌套菜单支持有了第二个消费者。
 *
 * 【导航】`useMenuKeyboard` 按 `closest('[role="menu"]')` 分层，因此外层菜单与
 * 子面板各按各的方向键走，互不串门。
 */
export function AppSubMenu({ label, badge, disabled = false, children }: AppSubMenuProps) {
    const [open, setOpen] = useState(false);
    const panelRef = useRef<HTMLDivElement>(null);
    useMenuKeyboard(panelRef);

    useLayoutEffect(() => {
        if (!open) return;
        const panel = panelRef.current;
        if (!panel) return;
        panel.style.left = "calc(100% - 4px)";
        panel.style.right = "auto";
        panel.style.top = "-5px";
        panel.style.bottom = "auto";
        // 宽度随内容展开：绝对定位面板的宽度默认被包含块（触发项宽度）封顶，
        // 长文本会因此换行。max-content 展开后若超出视口，按最终锚定侧的可用
        // 空间收口 —— 行内标签以 truncate 兜底。
        panel.style.width = "max-content";
        panel.style.maxWidth = "none";

        const vw = window.innerWidth;
        const vh = window.innerHeight;
        let rect = panel.getBoundingClientRect();
        // 右侧放不下就翻到左侧（菜单贴着视口右缘时必然如此）。
        if (rect.right > vw - 4) {
            panel.style.left = "auto";
            panel.style.right = "calc(100% - 4px)";
        }
        rect = panel.getBoundingClientRect();
        const anchoredLeft = panel.style.left !== "auto";
        const availableWidth = anchoredLeft ? vw - 8 - rect.left : rect.right - 8;
        if (rect.width > availableWidth) {
            panel.style.maxWidth = `${Math.max(160, Math.floor(availableWidth))}px`;
        }
        // 下方放不下就向上对齐（与触发项底边齐平）。
        rect = panel.getBoundingClientRect();
        if (rect.bottom > vh - 4) {
            panel.style.top = "auto";
            panel.style.bottom = "-5px";
        }
    }, [open]);

    return (
        <div
            className="relative"
            onMouseEnter={() => {
                if (!disabled) setOpen(true);
            }}
            onMouseLeave={() => setOpen(false)}
        >
            <button
                type="button"
                role="menuitem"
                className={cx(
                    "flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left text-qt-sm transition-colors",
                    disabled ? "cursor-default opacity-40" : "hover:bg-qt-hover",
                )}
                style={{ paddingLeft: "var(--qt-space-5)", paddingRight: "var(--qt-space-5)" }}
                disabled={disabled}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                    e.stopPropagation();
                    if (!disabled) setOpen((value) => !value);
                }}
                aria-haspopup="menu"
                aria-expanded={open}
            >
                <span className="flex min-w-0 items-center gap-2">
                    <span className="truncate">{label}</span>
                    {badge ? (
                        <span className="text-qt-micro leading-none rounded bg-black/20 px-1 py-0.5 opacity-70">
                            {badge}
                        </span>
                    ) : null}
                </span>
                <svg
                    width="12"
                    height="12"
                    viewBox="0 0 15 15"
                    fill="none"
                    aria-hidden="true"
                    className="shrink-0 opacity-50"
                >
                    <path
                        d="M6 3.5L10 7.5L6 11.5"
                        stroke="currentColor"
                        strokeWidth="1.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                </svg>
            </button>
            {open && !disabled ? (
                <div
                    ref={panelRef}
                    role="menu"
                    data-hs-context-menu="1"
                    className="absolute z-qt-menu min-w-[190px] rounded border border-qt-border bg-qt-window py-1 text-qt-text shadow-lg"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                >
                    {children}
                </div>
            ) : null}
        </div>
    );
}
