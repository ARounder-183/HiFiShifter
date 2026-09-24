/*
 * 停靠拖拽控制器。
 *
 * 【为什么是模块级命令式而不是 hook】拖拽期间要挂 window 级 pointermove/up、
 * 要读 DOM 里所有 Zone 的实时矩形、要在松手瞬间派发 Redux 动作。做成 hook
 * 就得把"谁在拖"穿过好几层组件传下去，而拖拽源（标签、浮窗标题栏）与落点
 * 判定（覆盖层）分处不同子树。做成模块级单例后，两边都只是调用同一个入口，
 * 状态经 `dockDragStore` 广播给唯一需要它的覆盖层。
 *
 * 【交互语义（对齐 VEGAS 的停靠修饰键）】
 * - 不按修饰键拖标签：拖出标签条即**浮动**；在标签条内横移 = **重排**。
 * - 按住修饰键拖：**优先停靠**，松手落在某个 Zone 上就并入/拆分，落空则回弹。
 * - 浮窗标题栏同理：不按修饰键只移动位置，按住修饰键拖到 Zone 上则重新停靠。
 *
 * 【重排与拆分的分界】用"指针是否还在标签条带内"判定，而不是靠距离阈值 ——
 * 前者与 Chrome/VS Code 的标签拖拽一致，用户不需要学习成本；后者会让"想重排
 * 却拆出去了"变成常见误操作。
 */

import { store } from "../../app/store";
import {
    beginDockDrag,
    endDockDrag,
    getDockDragState,
    updateDockDrag,
    type DockDropTargetState,
} from "../../features/dock/dockDragStore";
import {
    clampFloatRect,
    dropZoneToSide,
    pickDropTarget,
    resolveDropZone,
    snapFloatPosition,
    type DockZoneRect,
} from "../../features/dock/dockDropTarget";
import {
    dockFormTo,
    floatForm,
    mergeFormInto,
    setFloatGeometry,
    splitFormTo,
} from "../../features/dock/dockSlice";
import { findZone } from "../../features/dock/dockTree";
import type { DockDropZone, DockRect } from "../../features/dock/dockTypes";
import { isPrimaryModifierDown } from "../../utils/platform";

/** 启动拖拽的位移阈值（像素）：低于它视为点击，避免"点标签就抖出幽灵"。 */
const DRAG_THRESHOLD_PX = 6;

/** 标签条带的外扩量：稍微超出标签条仍算重排，手感更宽容。 */
const TAB_BAND_SLOP_PX = 10;

/** 浮动窗标题栏高度（夹紧时至少留这么多可见）。 */
const FLOAT_TITLE_BAR_PX = 28;

interface DragSession {
    mode: "tab" | "float";
    formId: string;
    panelId: string;
    pointerId: number;
    startX: number;
    startY: number;
    /** 指针相对窗体左上角的偏移（浮动搬运时保持抓取点不跳）。 */
    offsetX: number;
    offsetY: number;
    /** 标签拖拽：源标签组与标签条矩形。 */
    tabsetId: string | null;
    tabBarRect: DockRect | null;
    tabCount: number;
    /** 浮动拖拽：起始几何。 */
    floatStart: DockRect | null;
    /** 浮窗当前尺寸（拆成浮动窗时沿用）。 */
    sourceSize: { w: number; h: number };
}

let session: DragSession | null = null;

/** 拖拽开始时抓取的 Zone 矩形；窗口尺寸变化时刷新。 */
let zoneRects: DockZoneRect[] = [];

function collectZoneRects(): DockZoneRect[] {
    const out: DockZoneRect[] = [];
    for (const element of document.querySelectorAll<HTMLElement>("[data-dock-zone]")) {
        const zoneId = element.dataset.dockZone;
        if (!zoneId) continue;
        const rect = element.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) continue;
        out.push({ zoneId, rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height } });
    }
    return out;
}

function isDockModifierDown(event: PointerEvent | KeyboardEvent): boolean {
    const mode = store.getState().dock.settings.dockModifier;
    switch (mode) {
        case "none":
            return true;
        case "alt":
            return event.altKey;
        case "shift":
            return event.shiftKey;
        default:
            return isPrimaryModifierDown(event);
    }
}

function pointInInflated(rect: DockRect, x: number, y: number, slop: number): boolean {
    return (
        x >= rect.x - slop &&
        x <= rect.x + rect.w + slop &&
        y >= rect.y - slop &&
        y <= rect.y + rect.h + slop
    );
}

/** 解析指针落点：命中哪个 Zone、落在该 Zone 的哪一部位。 */
function resolveTarget(x: number, y: number): DockDropTargetState | null {
    const hit = pickDropTarget(zoneRects, { x, y });
    if (!hit) return null;
    const band = store.getState().dock.settings.edgeBandPx;
    const zone: DockDropZone = resolveDropZone(hit.rect, { x, y }, band) ?? "center";
    return { zoneId: hit.zoneId, zone, rect: hit.rect };
}

/** 计算标签拖拽的重排下标。 */
function resolveTabIndex(tabsetId: string, x: number): number | null {
    const tabs = document.querySelectorAll<HTMLElement>(
        `[data-dock-tabbar="${tabsetId}"] [data-dock-tab]`,
    );
    if (tabs.length === 0) return null;
    let index = tabs.length;
    for (let i = 0; i < tabs.length; i += 1) {
        const rect = tabs[i].getBoundingClientRect();
        if (x < rect.left + rect.width / 2) {
            index = i;
            break;
        }
    }
    return index;
}

function onPointerMove(event: PointerEvent): void {
    if (!session || event.pointerId !== session.pointerId) return;
    const state = getDockDragState();
    const x = event.clientX;
    const y = event.clientY;

    if (!state) {
        const distance = Math.hypot(x - session.startX, y - session.startY);
        if (distance < DRAG_THRESHOLD_PX) return;
        beginDockDrag({
            mode: session.mode,
            formId: session.formId,
            panelId: session.panelId,
            pointerX: x,
            pointerY: y,
            dockIntent: isDockModifierDown(event),
            floatRect: session.mode === "float" ? session.floatStart : null,
        });
        return;
    }

    const dockIntent = isDockModifierDown(event);
    const target = resolveTarget(x, y);

    if (session.mode === "float" && session.floatStart) {
        const settings = store.getState().dock.settings;
        const raw: DockRect = {
            x: x - session.offsetX,
            y: y - session.offsetY,
            w: session.floatStart.w,
            h: session.floatStart.h,
        };
        const snapped = settings.floatSnapEnabled
            ? snapFloatPosition(
                  raw,
                  zoneRects.map((zone) => zone.rect),
                  { w: window.innerWidth, h: window.innerHeight },
                  settings.floatSnapThresholdPx,
              )
            : { x: raw.x, y: raw.y };
        const next = clampFloatRect(
            { ...raw, x: snapped.x, y: snapped.y },
            { w: window.innerWidth, h: window.innerHeight },
            FLOAT_TITLE_BAR_PX,
        );
        updateDockDrag({ pointerX: x, pointerY: y, dockIntent, target, floatRect: next });
        return;
    }

    updateDockDrag({ pointerX: x, pointerY: y, dockIntent, target });
}

function onPointerUp(event: PointerEvent): void {
    if (!session || event.pointerId !== session.pointerId) return;
    const active = session;
    const state = getDockDragState();
    session = null;
    detach();

    if (!state) return; // 没越过阈值 = 一次点击，交给 onClick 处理

    const x = event.clientX;
    const y = event.clientY;
    const dockIntent = state.dockIntent;
    const target = state.target;
    endDockDrag();

    if (active.mode === "tab") {
        // 1) 按住修饰键 → 优先停靠；落空则回弹（什么都不做）。
        if (dockIntent) {
            if (!target) return;
            commitDock(active.formId, target);
            return;
        }
        // 2) 仍在标签条带内 → 同组重排。
        if (
            active.tabBarRect &&
            active.tabsetId &&
            active.tabCount > 1 &&
            pointInInflated(active.tabBarRect, x, y, TAB_BAND_SLOP_PX)
        ) {
            const index = resolveTabIndex(active.tabsetId, x);
            if (index !== null) {
                store.dispatch(
                    dockFormTo({
                        formId: active.formId,
                        target: { kind: "tab", tabsetId: active.tabsetId, index },
                    }),
                );
            }
            return;
        }
        // 3) 拖出标签条 → 浮动，落在指针处。
        store.dispatch(
            floatForm({
                formId: active.formId,
                geometry: {
                    x: Math.round(x - active.offsetX),
                    y: Math.round(y - active.offsetY),
                    w: active.sourceSize.w,
                    h: active.sourceSize.h,
                },
            }),
        );
        return;
    }

    // 浮动窗体
    if (dockIntent && target) {
        commitDock(active.formId, target);
        return;
    }
    const geometry = state.floatRect ?? active.floatStart;
    if (geometry) {
        store.dispatch(
            setFloatGeometry({
                formId: active.formId,
                geometry: { x: geometry.x, y: geometry.y, w: geometry.w, h: geometry.h },
            }),
        );
    }
}

/** 把落点提交为"并入标签组"或"在某一侧拆分"。 */
function commitDock(formId: string, target: DockDropTargetState): void {
    const side = dropZoneToSide(target.zone);
    const layout = store.getState().dock.layout;
    const zone = findZone(layout.tree, target.zoneId);
    if (!zone) return;

    if (zone.t === "tabset" && side === null) {
        store.dispatch(mergeFormInto({ formId, referenceFormId: zone.active || zone.tabs[0] }));
        return;
    }

    // 拆分：以目标 Zone 里当前活动的窗体作为参照物，落到它所在组的对应侧。
    const referenceFormId = zone.t === "tabset" ? zone.active || zone.tabs[0] : null;
    if (!referenceFormId || side === null) return;
    store.dispatch(splitFormTo({ formId, referenceFormId, side }));
}

function onKeyUp(event: KeyboardEvent): void {
    // 修饰键在拖拽中被松开：立刻把落点预览切回"浮动"语义，避免松手结果与
    // 屏幕上的提示不一致。
    if (!session || !getDockDragState()) return;
    if (
        event.key !== "Control" &&
        event.key !== "Meta" &&
        event.key !== "Alt" &&
        event.key !== "Shift"
    ) {
        return;
    }
    updateDockDrag({ dockIntent: isDockModifierDown(event), target: null });
}

function onWindowResize(): void {
    if (!session) return;
    zoneRects = collectZoneRects();
}

function attach(): void {
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("resize", onWindowResize);
}

function detach(): void {
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    window.removeEventListener("pointercancel", onPointerUp);
    window.removeEventListener("keyup", onKeyUp, true);
    window.removeEventListener("resize", onWindowResize);
}

export interface TabDragArgs {
    formId: string;
    panelId: string;
    tabsetId: string;
    tabCount: number;
    tabBarElement: HTMLElement | null;
}

/** 从一个标签开始拖拽。 */
export function beginTabDrag(event: React.PointerEvent, args: TabDragArgs): void {
    if (event.button !== 0) return;
    const tabBarRect = args.tabBarElement?.getBoundingClientRect() ?? null;
    const host = document.querySelector<HTMLElement>(
        `[data-dock-host="${cssEscape(args.formId)}"]`,
    );
    const hostRect = host?.getBoundingClientRect();
    const sourceSize = {
        w: Math.max(200, Math.round(hostRect?.width ?? 420)),
        h: Math.max(120, Math.round(hostRect?.height ?? 320)),
    };

    session = {
        mode: "tab",
        formId: args.formId,
        panelId: args.panelId,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        // 拆成浮窗时把指针放在标题栏左侧一点，符合"从标签上抓起来"的直觉。
        offsetX: 48,
        offsetY: 12,
        tabsetId: args.tabsetId,
        tabBarRect: tabBarRect
            ? { x: tabBarRect.x, y: tabBarRect.y, w: tabBarRect.width, h: tabBarRect.height }
            : null,
        tabCount: args.tabCount,
        floatStart: null,
        sourceSize,
    };
    zoneRects = collectZoneRects();
    attach();
}

export interface FloatDragArgs {
    formId: string;
    panelId: string;
    geometry: DockRect;
}

/** 从一个浮动窗的标题栏开始拖拽。 */
export function beginFloatDrag(event: React.PointerEvent, args: FloatDragArgs): void {
    if (event.button !== 0) return;
    session = {
        mode: "float",
        formId: args.formId,
        panelId: args.panelId,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        offsetX: event.clientX - args.geometry.x,
        offsetY: event.clientY - args.geometry.y,
        tabsetId: null,
        tabBarRect: null,
        tabCount: 0,
        floatStart: args.geometry,
        sourceSize: { w: args.geometry.w, h: args.geometry.h },
    };
    zoneRects = collectZoneRects();
    attach();
}

/** 拖拽是否正在进行（供 CSS 关掉指针事件等）。 */
export function isDockDragging(): boolean {
    return getDockDragState()?.started === true;
}

/** `CSS.escape` 在测试环境可能缺失，这里给一个最小实现。 */
function cssEscape(value: string): string {
    return value.replace(/["\\]/g, "\\$&");
}
