/*
 * 浮动层：浮在主界面之上的窗体。
 *
 * 【为什么默认不做成独立 OS 窗口】真实的 `WebviewWindow` 是另一个 JS 上下文，
 * 不共享 Redux / Context / 主题（独立窗口是另一个 JS 上下文，这正是
 * 这个原因）。要让时间轴或参数编辑器在独立窗口里工作，就得为每个面板重建
 * 状态桥 —— 而这两个面板恰恰是本应用最重的部分。走主窗口内的 portal 浮层
 * 则完全共享一切，且因为宿主是同一个 DOM 节点（见 `panelHostRegistry`），
 * 停靠 ⇄ 浮动切换是零成本的。
 *
 * 代价：浮窗不能拖到第二块显示器。这是刻意的取舍 —— 需要跨屏时，把整个主
 * 窗口拖过去即可；等状态桥成熟后再补 `floatMode: "osWindow"`。
 */

import { useCallback, useRef, useState, useSyncExternalStore } from "react";

import { store } from "../../app/store";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { EnterIcon, ExternalLinkIcon } from "@radix-ui/react-icons";

import { getDockDragState, subscribeDockDrag } from "../../features/dock/dockDragStore";
import {
    closeForm,
    dockFormTo,
    raiseFloat,
    renameForm,
    setFloatGeometry,
} from "../../features/dock/dockSlice";
import { findMainTabset } from "../../features/dock/dockSchema";
import { detachFormToWindow, maximizeActive } from "../../features/dock/dockApi";
import {
    displayTitleOf,
    isPanelDetachable,
    panelTitleOf,
    synthesizePanelDefinition,
} from "../../features/dock/dockPanel";
import { isPanelForm } from "../../features/dock/dockTree";
import type { DockForm, DockRect } from "../../features/dock/dockTypes";
import { useI18n } from "../../i18n/I18nProvider";
import { resolveFloatRect } from "../../features/dock/dockDropTarget";
import { beginFloatDrag } from "./dockDragController";
import { DockInlineRename } from "./DockInlineRename";
import { DockSubRoot } from "./DockSubRoot";
import { floatTitleBarActions } from "./floatTitleBar";
import { useDockSlot } from "./useDockSlot";

/** `geometry` 缺失（理论上不会发生）时的占位矩形，避免把 null 传进解析函数。 */
const ZERO_RECT = { x: 0, y: 0, w: 0, h: 0 };

export function DockFloatingLayer() {
    const layout = useAppSelector((s) => s.dock.layout);
    // `floatMode === "osWindow"` 的窗体由它自己的操作系统窗口渲染，这里不画 ——
    // 画了就会在主窗口里出现一个"幽灵浮窗"（内容为空，因为宿主已被搬走）。
    const floating = layout.floatOrder.filter(
        (formId) =>
            layout.forms[formId]?.floating === true &&
            layout.forms[formId]?.floatMode !== "osWindow",
    );
    if (floating.length === 0) return null;

    return (
        <>
            {floating.map((formId, index) => (
                <DockFloatWindow
                    key={formId}
                    form={layout.forms[formId]}
                    zIndex={100 + index}
                    active={layout.floatOrder.at(-1) === formId}
                />
            ))}
        </>
    );
}

/** 折叠（最小化）浮窗的标题条高度：与 `dock.css` 的 `[data-minimized="true"]` 同源。 */
const FLOAT_STRIP_HEIGHT_PX = 28;

function DockFloatWindow({
    form,
    zIndex,
    active,
}: {
    form: DockForm;
    zIndex: number;
    active: boolean;
}) {
    const dispatch = useAppDispatch();
    const layout = useAppSelector((s) => s.dock.layout);
    const { t, tf, tVars } = useI18n();
    // 浮动面板没有宿主 div（它渲染自己的树）：槽位机制只服务叶窗体。
    const floatingIsPanel = isPanelForm(form);
    const slotRef = useDockSlot(floatingIsPanel ? null : form.id);
    const geometry = form.float;
    const [dragging, setDragging] = useState(false);
    /** 行内重命名中（双击面板标题进入）：标题文本被输入框替换。 */
    const [renaming, setRenaming] = useState(false);
    const elementRef = useRef<HTMLDivElement | null>(null);

    // 拖拽中跟随实时几何，松手才落库（与分隔条同一策略）。
    const drag = useSyncExternalStore(subscribeDockDrag, getDockDragState, getDockDragState);
    const liveRect = drag?.started && drag.formId === form.id ? drag.floatRect : null;

    const definition = synthesizePanelDefinition(layout, form.id);
    /**
     * 标题：用户重命名 > 内容派生（面板）> 注册表标题。
     *
     * 【面板为什么从内容派生】组合出来的面板若永远叫"面板"，屏幕上同时浮着
     * 三个面板时无法分辨谁是谁。让它显示活动成员的标题，表现得像一个正常的
     * 窗口标题；成员多于一个时附带数量。设置可关（`panelTitleFromChild`）。
     */
    const titleFromChild = useAppSelector((s) => s.dock.settings.panelTitleFromChild);
    const title =
        form.title ??
        (isPanelForm(form)
            ? titleFromChild
                ? panelTitleOf(layout, form.id, tf)
                : tf("dock_panel_title")
            : definition
              ? tf(definition.titleKey)
              : form.panelId);
    /**
     * 标题栏动作矩阵：折叠 / 拆分 / 重停的可见性由定义给出 —— 叶窗体读注册表
     * 声明，面板读**派生**声明（全体成员可拆才可拆）。`redock` 与拖拽停靠共用
     * 同一条 `dockable` 声明 —— 拖不进去的面板也不能留一枚按钮绕道 dock 进去。
     */
    const actions = floatTitleBarActions(definition);
    /**
     * 不可拆时的解释：普通窗体是固定文案；面板指出**是谁挡住了**（递归到最内层
     * 的成员），否则用户面对一枚点不动的按钮无从下手。
     */
    const detachBlockedReason = isPanelForm(form)
        ? (() => {
              const verdict = isPanelDetachable(layout, form.id);
              if (verdict.ok) return null;
              const names = verdict.blockedBy
                  .map((id) => displayTitleOf(layout, id, tf))
                  .join(", ");
              return tVars("dock_detach_blocked_by", { names });
          })()
        : null;
    const doubleClickAction = useAppSelector((s) => s.dock.settings.doubleClickHeaderAction);

    /**
     * 本帧实际渲染用的矩形（带锚点时按当前视口推导）。
     *
     * 【交互必须以它为准，而不是 `geometry.x/y`】带锚点的浮窗（如首次打开的记事本）
     * 里 `geometry.x/y` 只是**占位值**（真实位置在渲染时才算出来）。早期实现把占位值
     * 交给拖拽/缩放作为起点，于是 `offsetX = clientX - 0`、拖拽目标 x 被算成约 0 ——
     * 表现为"一拖就跳到左上角"（用户报告的拖拽偏移）。缩放同理。
     */
    const rect =
        liveRect ??
        resolveFloatRect(geometry ?? ZERO_RECT, {
            w: window.innerWidth,
            h: window.innerHeight,
        });

    const onTitlePointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            if ((event.target as HTMLElement).closest("button")) return;
            dispatch(raiseFloat(form.id));

            const startDragAndTrack = (
                dragGeometry: DockRect,
                dropSize?: { w: number; h: number },
            ) => {
                setDragging(true);
                beginFloatDrag(event, {
                    formId: form.id,
                    panelId: form.panelId,
                    // 设置类面板（`dockable: false`）拖得动但停不进去。
                    dockable: actions.redock,
                    geometry: dragGeometry,
                    dropSize,
                });
                // 拖拽结束由控制器统一收尾；这里只负责把"正在拖"的视觉状态收回来。
                const onUp = () => {
                    setDragging(false);
                    window.removeEventListener("pointerup", onUp);
                    window.removeEventListener("pointercancel", onUp);
                };
                window.addEventListener("pointerup", onUp);
                window.addEventListener("pointercancel", onUp);
            };

            // ── 最大化：先还原，再"扯"下来 ──
            // 双击最大化后拖标题栏的意图是"把它拽下来"：先取消最大化，窗口以
            // 还原尺寸落到指针下继续拖（与资源管理器 / 浏览器同款）。抓取点在
            // 标题栏上的相对横向位置保持不变，纵向让标题条跟住指针。
            if (geometry?.maximized === true) {
                const restore = geometry.restore ?? { x: rect.x, y: rect.y, w: rect.w, h: rect.h };
                dispatch(
                    setFloatGeometry({
                        formId: form.id,
                        geometry: { maximized: false, ...restore },
                    }),
                );
                // resolveFloatRect 只吃几何字段（锚点等），状态标志不参与推导。
                const restoredRect = resolveFloatRect(
                    {
                        x: restore.x,
                        y: restore.y,
                        w: restore.w,
                        h: restore.h,
                        anchor: geometry.anchor,
                        anchorMarginPx: geometry.anchorMarginPx,
                        anchorOffsetX: geometry.anchorOffsetX,
                        anchorOffsetY: geometry.anchorOffsetY,
                    },
                    { w: window.innerWidth, h: window.innerHeight },
                );
                const ratioX = Math.min(
                    1,
                    Math.max(0, event.clientX / Math.max(1, window.innerWidth)),
                );
                startDragAndTrack({
                    ...restoredRect,
                    x: event.clientX - restoredRect.w * ratioX,
                });
                return;
            }

            // ── 折叠为标签条：拖的是"标题条"，不是整窗 ──
            // 拖拽预览与搬运都以标题条（28px，与 dock.css 的 [data-minimized]
            // 同源）为准，幽灵不再是原先整窗的大小；落库尺寸用 dropSize 保留
            // 展开后的高度，重新展开不丢尺寸。
            if (geometry?.minimized === true) {
                startDragAndTrack(
                    { x: rect.x, y: rect.y, w: rect.w, h: FLOAT_STRIP_HEIGHT_PX },
                    { w: rect.w, h: rect.h },
                );
                return;
            }

            startDragAndTrack(rect);
        },
        [dispatch, form.id, form.panelId, rect, geometry, actions.redock],
    );

    const onResizePointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>, edge: string) => {
            if (event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();
            dispatch(raiseFloat(form.id));
            setDragging(true);

            const start = { ...rect };
            const startX = event.clientX;
            const startY = event.clientY;
            const element = elementRef.current;
            let latest: DockRect = { x: start.x, y: start.y, w: start.w, h: start.h };

            const apply = (clientX: number, clientY: number) => {
                const dx = clientX - startX;
                const dy = clientY - startY;
                let { x, y, w, h } = start;
                if (edge.includes("e")) w = Math.max(180, start.w + dx);
                if (edge.includes("s")) h = Math.max(120, start.h + dy);
                if (edge.includes("w")) {
                    w = Math.max(180, start.w - dx);
                    x = start.x + (start.w - w);
                }
                if (edge.includes("n")) {
                    h = Math.max(120, start.h - dy);
                    y = start.y + (start.h - h);
                }
                latest = { x, y, w, h };
                if (element) {
                    element.style.left = `${x}px`;
                    element.style.top = `${y}px`;
                    element.style.width = `${w}px`;
                    element.style.height = `${h}px`;
                }
            };

            const onMove = (moveEvent: PointerEvent) => apply(moveEvent.clientX, moveEvent.clientY);
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                setDragging(false);
                dispatch(setFloatGeometry({ formId: form.id, geometry: latest }));
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [dispatch, form.id, rect],
    );

    if (!geometry) return null;

    const maximized = geometry.maximized === true;
    const minimized = geometry.minimized === true;

    return (
        <div
            ref={elementRef}
            className="hs-dock-float"
            data-active={active ? "true" : "false"}
            data-maximized={maximized ? "true" : "false"}
            data-minimized={minimized ? "true" : "false"}
            data-dragging={dragging ? "true" : "false"}
            /* 拖拽落点的第三个来源：浮窗自己是可停靠目标（拖到另一浮窗上 =
               组合成面板；拖到浮动面板上 = 停入它的树）。正在被拖的那枚由
               data-dragging 标记，采集时跳过。 */
            data-dock-float={form.id}
            data-dock-float-panel={isPanelForm(form) ? "true" : undefined}
            style={
                maximized
                    ? { left: 0, top: 0, width: "100vw", height: "100vh", zIndex }
                    : { left: rect.x, top: rect.y, width: rect.w, height: rect.h, zIndex }
            }
            onPointerDown={() => dispatch(raiseFloat(form.id))}
        >
            <div
                className="hs-dock-float-title"
                onPointerDown={onTitlePointerDown}
                onDoubleClick={() => {
                    // 面板的名称区域双击 = 进入行内重命名（先于设置的默认动作）。
                    if (floatingIsPanel) {
                        setRenaming(true);
                        return;
                    }
                    if (doubleClickAction === "none") return;
                    if (doubleClickAction === "maximize") {
                        dispatch(raiseFloat(form.id));
                        maximizeActive(dispatch);
                        return;
                    }
                    if (doubleClickAction === "collapse") {
                        dispatch(
                            setFloatGeometry({
                                formId: form.id,
                                geometry: { minimized: !minimized },
                            }),
                        );
                        return;
                    }
                    if (maximized) {
                        dispatch(
                            setFloatGeometry({
                                formId: form.id,
                                geometry: { maximized: false, ...(geometry.restore ?? {}) },
                            }),
                        );
                    } else {
                        dispatch(
                            setFloatGeometry({
                                formId: form.id,
                                geometry: {
                                    maximized: true,
                                    restore: {
                                        x: geometry.x,
                                        y: geometry.y,
                                        w: geometry.w,
                                        h: geometry.h,
                                    },
                                },
                            }),
                        );
                    }
                }}
                // 【刻意不挂 `data-tooltip`】浮动窗口的标题栏悬停时**不显示任何提示**：
                // 它是"拖起来"的着力点，拖拽开始后幽灵提示会立刻给出完整说明（含
                // 停靠修饰键），悬停时再弹一条更长的提示只会挡住标题栏本身。停靠态的
                // 抓手仍保留悬停提示（见 `DockTabBar`）。
            >
                {renaming && floatingIsPanel ? (
                    <DockInlineRename
                        initial={form.title ?? ""}
                        placeholder={title}
                        ariaLabel={tf("dock_rename_tab")}
                        onCommit={(next) => {
                            dispatch(renameForm({ formId: form.id, title: next }));
                            setRenaming(false);
                        }}
                        onCancel={() => setRenaming(false)}
                    />
                ) : (
                    <span className="hs-dock-tab-label">{title}</span>
                )}
                <div className="hs-dock-tabbar-spacer" />
                <button
                    type="button"
                    className="hs-dock-tabbar-action"
                    data-tooltip={tf(minimized ? "dock_expand" : "dock_collapse")}
                    aria-label={tf(minimized ? "dock_expand" : "dock_collapse")}
                    onClick={() =>
                        dispatch(
                            setFloatGeometry({
                                formId: form.id,
                                geometry: { minimized: !minimized },
                            }),
                        )
                    }
                >
                    {minimized ? "\u25B2" : "\u25BC"}
                </button>
                {actions.detach === "available" ? (
                    <button
                        type="button"
                        className="hs-dock-tabbar-action"
                        data-tooltip={tf("dock_detach_to_window")}
                        aria-label={tf("dock_detach_to_window")}
                        onClick={() => void detachFormToWindow(dispatch, store.getState, form.id)}
                    >
                        <ExternalLinkIcon />
                    </button>
                ) : actions.detach === "unsupported" ? (
                    // 不可拆的面板给出**解释**而不是一个点不动的按钮：普通窗体是
                    // 固定文案（时间轴与参数编辑器带着 WebGL 上下文与波形缓存，
                    // 跨窗口必须重新挂载，代价是数秒卡顿）；面板则指出是谁挡住了。
                    <span
                        className="hs-dock-tabbar-action"
                        data-tooltip={detachBlockedReason ?? tf("dock_detach_unsupported")}
                        aria-hidden
                        style={{ opacity: 0.4, cursor: "default" }}
                    >
                        <ExternalLinkIcon />
                    </span>
                ) : null}
                {actions.redock ? (
                    <button
                        type="button"
                        className="hs-dock-tabbar-action"
                        data-tooltip={tf("dock_redock")}
                        aria-label={tf("dock_redock")}
                        onClick={() => {
                            // 现读 store 而不是把树存进 ref：渲染期写 ref 违反
                            // React Compiler 的引用规则，而 store 随时可读。
                            const main = findMainTabset(store.getState().dock.layout);
                            if (!main) return;
                            dispatch(
                                dockFormTo({
                                    formId: form.id,
                                    target: { kind: "tab", tabsetId: main.id },
                                }),
                            );
                        }}
                    >
                        <EnterIcon />
                    </button>
                ) : null}
                <button
                    type="button"
                    className="hs-dock-tabbar-action"
                    data-tooltip={t("close")}
                    aria-label={t("close")}
                    onClick={() => dispatch(closeForm(form.id))}
                >
                    {"\u00D7"}
                </button>
            </div>

            <div className="hs-dock-float-body">
                {floatingIsPanel && form.childRootId ? (
                    // 浮动面板：body 里是它自己的布局根，不是宿主槽位。
                    <DockSubRoot rootId={form.childRootId} kind="panel" />
                ) : (
                    <div ref={slotRef} className="h-full w-full" />
                )}
            </div>

            {!maximized && !minimized
                ? (["n", "s", "w", "e", "nw", "ne", "sw", "se"] as const).map((edge) => (
                      <div
                          key={edge}
                          className="hs-dock-float-handle"
                          data-edge={edge}
                          onPointerDown={(event) => onResizePointerDown(event, edge)}
                      />
                  ))
                : null}
        </div>
    );
}
