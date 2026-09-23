/*
 * 标签条：一个标签组内多窗体共处一格的界面。
 *
 * 这里承担三种手势，互不冲突：
 * 1. 单击标签 → 切换活动窗体；
 * 2. 拖标签 → 见 `dockDragController`（条内横移 = 重排，拖出 = 浮动，按住
 *    修饰键 = 停靠）；
 * 3. 右键标签 → 菜单（重命名 / 浮动 / 停靠 / 关闭）。
 */

import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
    ChevronDownIcon,
    ChevronUpIcon,
    Cross2Icon,
    DragHandleDots2Icon,
    ExternalLinkIcon,
} from "@radix-ui/react-icons";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import {
    getDockDragState,
    subscribeDockDrag,
    type DockDragState,
} from "../../features/dock/dockDragStore";
import {
    closeForm,
    floatForm,
    focusForm,
    setActiveTabOf,
    toggleTabsetCollapsed,
} from "../../features/dock/dockSlice";
import { getPanel } from "../../features/dock/panelRegistry";
import type { DockTabsetNode } from "../../features/dock/dockTypes";
import { useI18n } from "../../i18n/I18nProvider";
import { beginTabDrag } from "./dockDragController";
import { DockTabMenu } from "./DockTabMenu";

export interface DockTabBarProps {
    node: DockTabsetNode;
    onToggleFloat: (formId: string) => void;
    /**
     * 紧凑形态：单标签、未折叠、且用户没要求"总是显示标签条"时，只渲染一条
     * 极窄抓手（12px）而不是 26px 的完整标签条 —— 时间轴这类面板自带标题栏，
     * 再顶一条标签条纯属浪费垂直空间。
     */
    compact: boolean;
}

export function DockTabBar({ node, onToggleFloat, compact }: DockTabBarProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const barRef = useRef<HTMLDivElement | null>(null);
    const [menu, setMenu] = useState<{ formId: string; x: number; y: number } | null>(null);
    const showIcons = useAppSelector((s) => s.dock.settings.tabIcons);
    const forms = useAppSelector((s) => s.dock.layout.forms);

    // 拖拽中高亮"会插到哪个标签旁边"。只有本组自己在拖时才需要。
    const drag = useSyncExternalStore(subscribeDockDrag, getDockDragState, getDockDragState);
    const dragIndex = useMemo(() => resolveInsertIndex(drag, node, barRef.current), [drag, node]);

    const onTabPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>, formId: string) => {
            if (event.button !== 0) return;
            // 关闭按钮走自己的 onClick。
            if ((event.target as HTMLElement).closest("[data-dock-tab-close]")) return;
            dispatch(focusForm(formId));
            const panelId = forms[formId]?.panelId;
            beginTabDrag(event, {
                formId,
                panelId: panelId ?? formId,
                tabsetId: node.id,
                tabCount: node.tabs.length,
                tabBarElement: barRef.current,
            });
        },
        [dispatch, forms, node.id, node.tabs.length],
    );

    return (
        <>
            {!compact ? (
                <div
                    ref={barRef}
                    className="hs-dock-tabbar"
                    data-dock-tabbar={node.id}
                    data-collapsed={node.collapsed ? "true" : "false"}
                    role="tablist"
                >
                    {node.tabs.map((formId, index) => {
                        const form = forms[formId];
                        const definition = form ? getPanel(form.panelId) : undefined;
                        const title = form?.title ?? (definition ? tAny(definition.titleKey) : formId);
                        const Icon = definition?.icon;
                        const active = node.active === formId;
                        return (
                            <div
                                key={formId}
                                className="hs-dock-tab"
                                data-dock-tab={formId}
                                data-active={active ? "true" : "false"}
                                data-dock-target={dragIndex === index ? "true" : "false"}
                                data-dragging={drag?.started && drag.formId === formId ? "true" : "false"}
                                role="tab"
                                aria-selected={active}
                                title={title}
                                onPointerDown={(event) => onTabPointerDown(event, formId)}
                                onClick={() => dispatch(setActiveTabOf({ tabsetId: node.id, formId }))}
                                onContextMenu={(event) => {
                                    event.preventDefault();
                                    setMenu({ formId, x: event.clientX, y: event.clientY });
                                }}
                            >
                                {showIcons && Icon ? <Icon /> : null}
                                <span className="hs-dock-tab-label">{title}</span>
                                <span
                                    className="hs-dock-tab-close"
                                    data-dock-tab-close="1"
                                    role="button"
                                    aria-label={t("close")}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        dispatch(closeForm(formId));
                                    }}
                                >
                                    <Cross2Icon />
                                </span>
                            </div>
                        );
                    })}

                    <div className="hs-dock-tabbar-spacer" />

                    <div className="hs-dock-tabbar-actions">
                        <button
                            type="button"
                            className="hs-dock-tabbar-action"
                            title={tAny("dock_float_active")}
                            aria-label={tAny("dock_float_active")}
                            onClick={() => onToggleFloat(node.active)}
                        >
                            <ExternalLinkIcon />
                        </button>
                        <button
                            type="button"
                            className="hs-dock-tabbar-action"
                            title={tAny(node.collapsed ? "dock_expand" : "dock_collapse")}
                            aria-label={tAny(node.collapsed ? "dock_expand" : "dock_collapse")}
                            onClick={() => dispatch(toggleTabsetCollapsed({ tabsetId: node.id }))}
                        >
                            {node.collapsed ? <ChevronUpIcon /> : <ChevronDownIcon />}
                        </button>
                    </div>
                </div>
            ) : (
                // 紧凑形态：只留一条极窄的抓手。
                //
                // 【为什么不能什么都不渲染】面板需要一个可拖拽的着力点，否则
                // 用户再也没法把它拖出去、或与别的面板合并。
                <div
                    ref={barRef}
                    className="hs-dock-grip"
                    data-dock-tabbar={node.id}
                    data-dock-tab={node.active}
                    data-active="true"
                    role="tab"
                    aria-selected
                    title={tAny("dock_drag_hint")}
                    onPointerDown={(event) => onTabPointerDown(event, node.active)}
                    onContextMenu={(event) => {
                        event.preventDefault();
                        setMenu({ formId: node.active, x: event.clientX, y: event.clientY });
                    }}
                >
                    <DragHandleDots2Icon />
                </div>
            )}

            {menu ? (
                <DockTabMenu
                    formId={menu.formId}
                    x={menu.x}
                    y={menu.y}
                    onClose={() => setMenu(null)}
                    onFloat={() => {
                        dispatch(floatForm({ formId: menu.formId }));
                        setMenu(null);
                    }}
                    onCloseForm={() => {
                        dispatch(closeForm(menu.formId));
                        setMenu(null);
                    }}
                />
            ) : null}
        </>
    );
}

/**
 * 拖拽中"会插到第几个标签"的推算。
 *
 * 只在拖拽源与本组相同时给出提示：跨组拖拽的落点由覆盖层的半透明预览表达，
 * 两者同时高亮会让用户以为要发生两件事。
 */
function resolveInsertIndex(
    drag: DockDragState | null,
    node: DockTabsetNode,
    bar: HTMLElement | null,
): number | null {
    if (!drag?.started || drag.mode !== "tab" || !bar) return null;
    if (!node.tabs.includes(drag.formId)) return null;

    const rect = bar.getBoundingClientRect();
    if (
        drag.pointerX < rect.left - 12 ||
        drag.pointerX > rect.right + 12 ||
        drag.pointerY < rect.top - 12 ||
        drag.pointerY > rect.bottom + 12
    ) {
        return null;
    }

    const tabs = bar.querySelectorAll<HTMLElement>("[data-dock-tab]");
    for (let index = 0; index < tabs.length; index += 1) {
        const tabRect = tabs[index].getBoundingClientRect();
        if (drag.pointerX < tabRect.left + tabRect.width / 2) return index;
    }
    return tabs.length;
}
