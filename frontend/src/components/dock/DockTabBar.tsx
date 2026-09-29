/*
 * 标签条：一个标签组内多窗体共处一格的界面。
 *
 * 这里承担三种手势，互不冲突：
 * 1. 单击标签 → 切换活动窗体；
 * 2. 拖标签 → 见 `dockDragController`（条内横移 = 重排，拖出 = 浮动，按住
 *    修饰键 = 停靠）；
 * 3. 右键标签 → 菜单（重命名 / 浮动 / 停靠 / 关闭）。
 */

import { useCallback, useState, useSyncExternalStore } from "react";
import {
    ChevronDownIcon,
    ChevronUpIcon,
    Cross2Icon,
    DragHandleDots2Icon,
    ExternalLinkIcon,
} from "@radix-ui/react-icons";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { getDockDragState, subscribeDockDrag } from "../../features/dock/dockDragStore";
import {
    closeForm,
    floatForm,
    focusForm,
    renameForm,
    setActiveTabOf,
    toggleTabsetCollapsed,
} from "../../features/dock/dockSlice";
import { maximizeActive } from "../../features/dock/dockApi";
import { displayTitleOf, synthesizePanelDefinition } from "../../features/dock/dockPanel";
import { isPanelForm } from "../../features/dock/dockTree";
import { dockDragHint } from "./dockTooltips";
import type { DockTabsetNode, DockTabPosition } from "../../features/dock/dockTypes";
import { useI18n } from "../../i18n/I18nProvider";
import { beginTabDrag } from "./dockDragController";
import { DockTabMenu } from "./DockTabMenu";
import { DockInlineRename } from "./DockInlineRename";
import { detachFormToWindow } from "../../features/dock/dockApi";
import { store } from "../../app/store";

export interface DockTabBarProps {
    node: DockTabsetNode;
    onToggleFloat: (formId: string) => void;
    /**
     * 紧凑形态：单标签、未折叠、且用户没要求"总是显示标签条"时，只渲染一条
     * 极窄抓手（12px）而不是 26px 的完整标签条 —— 时间轴这类面板自带标题栏，
     * 再顶一条标签条纯属浪费垂直空间。
     */
    compact: boolean;
    /** 标签行位置（与 `DockZone` 的 `data-tab-position` 同源）。 */
    tabPosition: DockTabPosition;
}

export function DockTabBar({ node, onToggleFloat, compact, tabPosition }: DockTabBarProps) {
    const dispatch = useAppDispatch();
    const { t, tf } = useI18n();
    // 元素用 state 持有而不是 ref：插入下标要在**渲染期**按标签矩形推算，
    // 而渲染期读 ref 违反 React Compiler 的引用规则。回调 ref 只在挂载/卸载
    // 时触发，不会带来额外渲染。
    const [barElement, setBarElement] = useState<HTMLDivElement | null>(null);
    const [menu, setMenu] = useState<{ formId: string; x: number; y: number } | null>(null);
    /** 正在行内重命名的窗体 id（双击面板标签进入，见 `onTabDoubleClick`）。 */
    const [renamingFormId, setRenamingFormId] = useState<string | null>(null);
    const showIcons = useAppSelector((s) => s.dock.settings.tabIcons);
    const dockModifier = useAppSelector((s) => s.dock.settings.dockModifier);
    // 提示里带上**当前生效的**修饰键文本（可被用户改），两行由自定义 tooltip 的
    // `white-space: pre-line` 渲染。原生 `title` 无法保证换行与主题一致。
    const dragHint = dockDragHint(dockModifier, tf);
    const doubleClickAction = useAppSelector((s) => s.dock.settings.doubleClickHeaderAction);
    const layout = useAppSelector((s) => s.dock.layout);
    const forms = useAppSelector((s) => s.dock.layout.forms);

    // 拖拽中的视觉反馈只有两处：被拖标签自身（`data-dragging` 置灰），以及
    // **实时的顺序变化** —— 指针在源标签条带内每跨过一个相邻标签的中线，控制器
    // 就提交一次顺序（见 `dockDragController.maybeLiveReorder`）。不再画"插入
    // 位置"指示线：顺序真的在动，指示线只会成为多余的第三种反馈。
    const drag = useSyncExternalStore(subscribeDockDrag, getDockDragState, getDockDragState);

    /** 双击标签的行为：面板 = 进入行内重命名（名称区域的专属交互，先于设置的
     *  默认动作）；其余窗体由设置决定（默认浮动/停靠切换）。 */
    const onTabDoubleClick = useCallback(
        (formId: string) => {
            if (isPanelForm(forms[formId])) {
                setRenamingFormId(formId);
                return;
            }
            if (doubleClickAction === "none") return;
            if (doubleClickAction === "toggleFloat") {
                onToggleFloat(formId);
                return;
            }
            if (doubleClickAction === "maximize") {
                dispatch(focusForm(formId));
                maximizeActive(dispatch);
                return;
            }
            dispatch(toggleTabsetCollapsed({ tabsetId: node.id }));
        },
        [dispatch, doubleClickAction, forms, node.id, onToggleFloat],
    );

    /**
     * 标签条的键盘模型。
     *
     * 【为什么必须有】标签条此前只有指针通道：`role="tablist"` / `role="tab"`
     * 已声明，但没有任何标签可聚焦，也没有方向键 —— 屏幕阅读器会播报一个
     * 永远进不去的标签组，键盘用户则完全切不了标签。声明 ARIA 角色却不实现
     * 其键盘契约，比不声明更糟。
     *
     * 【模型】roving tabIndex：整条标签条只占一个 Tab 停留点（活动标签），
     * 进入后用方向键在标签间移动，移动即激活（automatic activation）——
     * 面板内容都是本地渲染、切换无代价，不需要"先聚焦再回车"的两段式。
     * `Home` / `End` 到两端；`Delete` / `Backspace` 关闭当前标签（与 IDE 一致）。
     */
    const onTabKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLDivElement>, formId: string) => {
            // 关闭按钮在自己的处理器里消化键盘事件，不参与方向键移动。
            if (event.target !== event.currentTarget) return;
            const tabs = Array.from(
                event.currentTarget.parentElement?.querySelectorAll<HTMLElement>('[role="tab"]') ??
                    [],
            );
            const index = tabs.indexOf(event.currentTarget);
            if (index < 0) return;

            if (event.key === "Delete" || event.key === "Backspace") {
                event.preventDefault();
                dispatch(closeForm(formId));
                return;
            }

            let next: number;
            if (event.key === "ArrowRight") next = index + 1;
            else if (event.key === "ArrowLeft") next = index - 1;
            else if (event.key === "Home") next = 0;
            else if (event.key === "End") next = tabs.length - 1;
            else return;
            // 标签行只有上/下两种位置，始终是水平排布，因此不处理上下方向键。
            if (next < 0 || next >= tabs.length) return;

            event.preventDefault();
            const target = tabs[next];
            const nextFormId = target.getAttribute("data-dock-tab");
            if (nextFormId) dispatch(setActiveTabOf({ tabsetId: node.id, formId: nextFormId }));
            target.focus();
        },
        [dispatch, node.id],
    );

    const onTabPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>, formId: string) => {
            if (event.button !== 0) return;
            // 关闭按钮走自己的 onClick。
            if ((event.target as HTMLElement).closest("[data-dock-tab-close]")) return;
            // 行内重命名中：启动拖拽会 setPointerCapture 到标签元素，光标定位与
            // 文本选择会被指针捕获劫持 —— 编辑期间标签不参与拖拽。
            if (renamingFormId === formId) return;
            dispatch(focusForm(formId));
            const panelId = forms[formId]?.panelId;
            beginTabDrag(event, {
                formId,
                panelId: panelId ?? formId,
                tabsetId: node.id,
                tabCount: node.tabs.length,
                tabBarElement: barElement,
            });
        },
        [barElement, dispatch, forms, node.id, node.tabs.length, renamingFormId],
    );

    return (
        <>
            {!compact ? (
                <div
                    ref={setBarElement}
                    className="hs-dock-tabbar"
                    data-dock-tabbar={node.id}
                    data-position={tabPosition}
                    data-collapsed={node.collapsed ? "true" : "false"}
                    role="tablist"
                    aria-orientation="horizontal"
                >
                    {node.tabs.map((formId) => {
                        const form = forms[formId];
                        // 标题按**窗体**解析（重命名 > 面板按内容派生 > 注册表）：
                        // 面板不在注册表里，此前会退回显示原始窗体 id（用户报告）。
                        const definition = form
                            ? synthesizePanelDefinition(layout, formId)
                            : undefined;
                        const title = form ? displayTitleOf(layout, formId, tf) : formId;
                        const Icon = definition?.icon;
                        const active = node.active === formId;
                        return (
                            <div
                                key={formId}
                                className="hs-dock-tab"
                                data-dock-tab={formId}
                                data-active={active ? "true" : "false"}
                                data-dragging={
                                    drag?.started && drag.formId === formId ? "true" : "false"
                                }
                                role="tab"
                                aria-selected={active}
                                tabIndex={active ? 0 : -1}
                                data-tooltip={title}
                                onKeyDown={(event) => onTabKeyDown(event, formId)}
                                onPointerDown={(event) => onTabPointerDown(event, formId)}
                                onClick={() =>
                                    dispatch(setActiveTabOf({ tabsetId: node.id, formId }))
                                }
                                onDoubleClick={() => onTabDoubleClick(formId)}
                                onContextMenu={(event) => {
                                    event.preventDefault();
                                    setMenu({ formId, x: event.clientX, y: event.clientY });
                                }}
                            >
                                {showIcons && Icon ? <Icon /> : null}
                                {renamingFormId === formId && form ? (
                                    <DockInlineRename
                                        initial={form.title ?? ""}
                                        placeholder={title}
                                        ariaLabel={tf("dock_rename_tab")}
                                        onCommit={(next) => {
                                            dispatch(renameForm({ formId, title: next }));
                                            setRenamingFormId(null);
                                        }}
                                        onCancel={() => setRenamingFormId(null)}
                                    />
                                ) : (
                                    <span className="hs-dock-tab-label">{title}</span>
                                )}
                                <span
                                    className="hs-dock-tab-close"
                                    data-dock-tab-close="1"
                                    role="button"
                                    aria-label={t("close")}
                                    // 只有活动标签的关闭按钮进 Tab 停留点：一个标签
                                    // 对应一个停留点（与 Chrome / VS Code 一致），
                                    // 非活动标签仍可用 Delete 或右键菜单关闭。
                                    tabIndex={active ? 0 : -1}
                                    onKeyDown={(event) => {
                                        if (event.key !== "Enter" && event.key !== " ") return;
                                        event.preventDefault();
                                        event.stopPropagation();
                                        dispatch(closeForm(formId));
                                    }}
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
                            data-tooltip={tf("dock_float_active")}
                            aria-label={tf("dock_float_active")}
                            onClick={() => onToggleFloat(node.active)}
                        >
                            <ExternalLinkIcon />
                        </button>
                        <button
                            type="button"
                            className="hs-dock-tabbar-action"
                            data-tooltip={tf(node.collapsed ? "dock_expand" : "dock_collapse")}
                            aria-label={tf(node.collapsed ? "dock_expand" : "dock_collapse")}
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
                //
                // 【为什么这里没有 ARIA 角色】抓手只是**指针**的着力点，不是
                // 可激活的控件：它不可聚焦、没有键盘等价操作，而紧凑形态下这个
                // 标签组本来就只有唯一一个标签，没有"切换"可言。给它安上
                // `role="tab"` 等于向屏幕阅读器许诺一个永远进不去的标签组。
                <div
                    ref={setBarElement}
                    className="hs-dock-grip"
                    data-dock-tabbar={node.id}
                    data-dock-tab={node.active}
                    data-active="true"
                    data-position={tabPosition}
                    data-tooltip={dragHint}
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
                    detachAction={
                        // 只有**可拆**的窗体才给出这个入口 —— 否则用户会点到一个
                        // 开不出来的窗口（时间轴带着 WebGL 上下文，跨窗口必须重新
                        // 挂载，代价不可接受）。面板的可拆性是派生的（全体成员可
                        // 拆才可拆），走同一份解析。
                        synthesizePanelDefinition(layout, menu.formId)?.detachable
                            ? {
                                  labelKey: "dock_detach_to_window",
                                  run: () => {
                                      void detachFormToWindow(
                                          dispatch,
                                          store.getState,
                                          menu.formId,
                                      );
                                  },
                              }
                            : null
                    }
                />
            ) : null}
        </>
    );
}
