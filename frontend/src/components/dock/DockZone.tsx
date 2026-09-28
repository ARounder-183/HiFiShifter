/*
 * 标签组（布局树的叶子）：标签条 + 内容槽位。
 *
 * 【这一层是"DOM 搬家"的落点】内容槽位渲染出来是空的，真实的面板 DOM 由
 * `useDockSlot` 从 `panelHostRegistry` 搬进来。因此布局树怎么变，面板的
 * React 组件都不卸载 —— 见 `panelHostRegistry` 的模块注释。
 */

import { useSyncExternalStore } from "react";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { getDockDragState, subscribeDockDrag } from "../../features/dock/dockDragStore";
import { floatForm } from "../../features/dock/dockSlice";
import type { DockTabsetNode } from "../../features/dock/dockTypes";
import { DockTabBar } from "./DockTabBar";
import { useDockSlot } from "./useDockSlot";

export interface DockZoneProps {
    node: DockTabsetNode;
}

export function DockZone({ node }: DockZoneProps) {
    const dispatch = useAppDispatch();
    // 拖拽期间关掉槽位内面板的指针事件，避免拖过时面板抢走 hover。
    const drag = useSyncExternalStore(subscribeDockDrag, getDockDragState, getDockDragState);
    const showTabBarWhenSingle = useAppSelector((s) => s.dock.settings.showTabBarWhenSingle);
    // 标签行位置是**布局级**偏好（持久化在 DockLayout 上），默认在下方。
    const tabPosition = useAppSelector((s) => s.dock.layout.tabPosition);

    const activeFormId =
        node.active && node.tabs.includes(node.active) ? node.active : (node.tabs[0] ?? null);
    const slotRef = useDockSlot(activeFormId);

    const onToggleFloat = (formId: string) => {
        if (!formId) return;
        dispatch(floatForm({ formId }));
    };

    // 折叠态只剩标签条时，槽位不渲染（CSS 也会隐藏它），但**仍然保持挂载**：
    // 宿主留在槽位里，展开即恢复，不需要重新挂载面板。
    return (
        <div
            className="hs-dock-tabset"
            data-dock-zone={node.id}
            data-collapsed={node.collapsed ? "true" : "false"}
            data-tab-position={tabPosition}
        >
            {/* 标签行与内容槽位的**视觉顺序**由 CSS `order` 决定（见 dock.css 的
                `data-position` 规则）：标签在下方时不必改动 DOM 顺序，焦点顺序与
                视觉顺序保持一致由 `data-tab-position` 上的 order 统一表达。 */}
            <DockTabBar
                node={node}
                onToggleFloat={onToggleFloat}
                compact={!showTabBarWhenSingle && node.tabs.length === 1 && !node.collapsed}
                tabPosition={tabPosition}
            />
            <div
                ref={slotRef}
                className="hs-dock-slot"
                data-dock-slot={node.id}
                style={drag?.started ? { pointerEvents: "none" } : undefined}
            />
        </div>
    );
}
