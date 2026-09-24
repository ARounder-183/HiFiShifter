/*
 * 浮动层：浮在主界面之上的窗体。
 *
 * 【为什么默认不做成独立 OS 窗口】真实的 `WebviewWindow` 是另一个 JS 上下文，
 * 不共享 Redux / Context / 主题（`appearanceMain.tsx` 刻意不带 Redux 就是
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
import { EnterIcon } from "@radix-ui/react-icons";

import { getDockDragState, subscribeDockDrag } from "../../features/dock/dockDragStore";
import { closeForm, dockFormTo, raiseFloat, setFloatGeometry } from "../../features/dock/dockSlice";
import { findMainTabset } from "../../features/dock/dockSchema";
import { maximizeActive } from "../../features/dock/dockApi";
import { getPanel } from "../../features/dock/panelRegistry";
import type { DockForm, DockRect } from "../../features/dock/dockTypes";
import { useI18n } from "../../i18n/I18nProvider";
import { beginFloatDrag } from "./dockDragController";
import { useDockSlot } from "./useDockSlot";

export function DockFloatingLayer() {
    const layout = useAppSelector((s) => s.dock.layout);
    const floating = layout.floatOrder.filter((formId) => Boolean(layout.forms[formId]?.float));
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const slotRef = useDockSlot(form.id);
    const geometry = form.float;
    const [dragging, setDragging] = useState(false);
    const elementRef = useRef<HTMLDivElement | null>(null);

    // 拖拽中跟随实时几何，松手才落库（与分隔条同一策略）。
    const drag = useSyncExternalStore(subscribeDockDrag, getDockDragState, getDockDragState);
    const liveRect = drag?.started && drag.formId === form.id ? drag.floatRect : null;

    const definition = getPanel(form.panelId);
    const title = form.title ?? (definition ? tAny(definition.titleKey) : form.panelId);
    const doubleClickAction = useAppSelector((s) => s.dock.settings.doubleClickHeaderAction);

    const onTitlePointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            if ((event.target as HTMLElement).closest("button")) return;
            dispatch(raiseFloat(form.id));
            if (!geometry) return;
            setDragging(true);
            beginFloatDrag(event, {
                formId: form.id,
                panelId: form.panelId,
                geometry: { x: geometry.x, y: geometry.y, w: geometry.w, h: geometry.h },
            });
            // 拖拽结束由控制器统一收尾；这里只负责把"正在拖"的视觉状态收回来。
            const onUp = () => {
                setDragging(false);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
            };
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [dispatch, form.id, form.panelId, geometry],
    );

    const onResizePointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>, edge: string) => {
            if (event.button !== 0 || !geometry) return;
            event.preventDefault();
            event.stopPropagation();
            dispatch(raiseFloat(form.id));
            setDragging(true);

            const start = { ...geometry };
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
        [dispatch, form.id, geometry],
    );

    if (!geometry) return null;

    const rect = liveRect ?? geometry;
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
                title={tAny("dock_drag_hint")}
            >
                <span className="hs-dock-tab-label">{title}</span>
                <div className="hs-dock-tabbar-spacer" />
                <button
                    type="button"
                    className="hs-dock-tabbar-action"
                    title={tAny(minimized ? "dock_expand" : "dock_collapse")}
                    aria-label={tAny(minimized ? "dock_expand" : "dock_collapse")}
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
                <button
                    type="button"
                    className="hs-dock-tabbar-action"
                    title={tAny("dock_redock")}
                    aria-label={tAny("dock_redock")}
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
                <button
                    type="button"
                    className="hs-dock-tabbar-action"
                    title={t("close")}
                    aria-label={t("close")}
                    onClick={() => dispatch(closeForm(form.id))}
                >
                    {"\u00D7"}
                </button>
            </div>

            <div className="hs-dock-float-body">
                <div ref={slotRef} className="h-full w-full" />
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
