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
 * - 按住修饰键拖：**停靠**，松手落在某个 Zone 上就并入/拆分，落空则回弹。
 * - 浮窗标题栏同理：不按修饰键只移动位置，按住修饰键拖到 Zone 上则重新停靠。
 * - 优先级：源标签条带内的重排判定**先于**停靠意图 —— 指针明明还在这一组的
 *   标签条上，意图就是"调整这组的顺序"，按住修饰键也不例外（否则"始终停靠"
 *   模式下组内重排会被并入判定劫持成"移到组尾"）。
 *
 * 【重排与拆分的分界】用"指针是否还在标签条带内"判定，而不是靠距离阈值 ——
 * 前者与 Chrome/VS Code 的标签拖拽一致，用户不需要学习成本；后者会让"想重排
 * 却拆出去了"变成常见误操作。重排带内的反馈是**实时的顺序变化**（每跨过一个
 * 标签中点就提交一次顺序），同时压掉浮动幽灵 —— 用户只是想交换标签位置，
 * 冒出浮窗轮廓会让他以为拖错了（用户报告）。
 *
 * 【取消】pointercancel（掌压拒绝、IME 抢占等 OS 级取消）与 Escape 都会**中止**
 * 拖拽且不提交落点 —— 浮动预览与落点提示都只是 dockDragStore 的临时状态。唯一
 * 的例外是重排：它为了实时反馈会在拖拽中途提交顺序，取消时把源组恢复为拖拽前
 * 的顺序（起始下标在会话里留了底）。
 */

import { store } from "../../app/store";
import {
    beginDockDrag,
    endDockDrag,
    getDockDragState,
    updateDockDrag,
    type DockDragState,
    type DockDropTargetState,
} from "../../features/dock/dockDragStore";
import {
    buildRootEdgeZones,
    clampFloatRect,
    DOCK_ROOT_ZONE_ID,
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
import { getPanel } from "../../features/dock/panelRegistry";
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
    /**
     * 本次拖拽是否允许停靠。
     *
     * `false` 时拖拽仍然移动窗口，但**永不产生落点** —— 用于声明了
     * `dockable: false` 的面板（外观设置）：设置界面不该被编入工作布局，
     * 拖进主编辑区占一格既无意义又会污染用户排好的布局。
     */
    dockable: boolean;
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
    /**
     * 拖拽开始时被拖标签在源组内的下标。实时重排会在拖拽中途提交顺序，
     * 取消（Escape / pointercancel）时按它恢复原状；`null` 表示无从恢复
     * （源组此刻不存在等，理论不可达）。
     */
    reorderBaseIndex: number | null;
    /** 本次拖拽是否已经实时提交过顺序（决定取消时是否需要恢复）。 */
    reorderMoved: boolean;
    /** 浮动拖拽：起始几何。 */
    floatStart: DockRect | null;
    /** 浮窗当前尺寸（拆成浮动窗时沿用）。 */
    sourceSize: { w: number; h: number };
    /**
     * 松手落库的尺寸（默认 = sourceSize）。
     *
     * 【为什么与 sourceSize 分开】折叠为标签条（minimized）的浮窗拖拽时，视觉与
     * 预览都是"标题条"（28px 高），但落库必须保留**展开后**的尺寸 —— 否则重新
     * 展开时浮窗只剩一条标题（高度被 28px 覆写）。
     */
    dropSize: { w: number; h: number };
    /** 最后一次已知的指针视口坐标（修饰键变化时据此重算落点，见 `refreshIntent`）。 */
    lastX: number;
    lastY: number;
}

let session: DragSession | null = null;

/** 拖拽开始时抓取的 Zone 矩形；窗口尺寸变化时刷新。 */
let zoneRects: DockZoneRect[] = [];

// ── rAF 合并 ─────────────────────────────────────────────────────
//
// pointermove 每秒可触发上百次，而 dockDragStore 的订阅者（覆盖层、每个
// DockTabBar）会同步重渲染并量测 DOM —— 逐事件推送等于每条 move 强制一次
// 回流。因此把"最新一次算出的状态"存在模块里，每帧最多推送一次；pointerup
// 时同步清空积压，保证提交读到的是最终位置。

/** 越过启动阈值的那次移动要先 begin：与补丁同一帧推送。 */
let pendingBegin: Parameters<typeof beginDockDrag>[0] | null = null;
/** 待推送的最新补丁；新的直接覆盖旧的（中间位置没有渲染价值）。 */
let pendingPatch: Partial<DockDragState> | null = null;
let frameHandle: number | null = null;

/** 把积压的补丁同步推送进 dockDragStore（pointerup 提交前必须先调用）。 */
function flushDragUpdate(): void {
    if (frameHandle !== null) {
        cancelAnimationFrame(frameHandle);
        frameHandle = null;
    }
    if (pendingBegin) {
        beginDockDrag(pendingBegin);
        pendingBegin = null;
    }
    if (pendingPatch) {
        updateDockDrag(pendingPatch);
        pendingPatch = null;
    }
}

function scheduleDragUpdate(
    begin: Parameters<typeof beginDockDrag>[0] | null,
    patch: Partial<DockDragState>,
): void {
    if (begin) pendingBegin = begin;
    pendingPatch = patch;
    if (frameHandle === null) frameHandle = requestAnimationFrame(flushDragUpdate);
}

function discardScheduledUpdate(): void {
    if (frameHandle !== null) {
        cancelAnimationFrame(frameHandle);
        frameHandle = null;
    }
    pendingBegin = null;
    pendingPatch = null;
}

function collectZoneRects(): DockZoneRect[] {
    const out: DockZoneRect[] = [];
    for (const element of document.querySelectorAll<HTMLElement>("[data-dock-zone]")) {
        const zoneId = element.dataset.dockZone;
        if (!zoneId) continue;
        const rect = element.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) continue;
        out.push({ zoneId, rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height } });
    }
    // 根级边缘带：贴着整个停靠区外缘的合成 Zone，命中即"贯通该侧拆分"。
    // 与标签组 Zone 一起参与"面积最小者优先"的挑选，细带自然压过外缘处的
    // 大矩形（见 `buildRootEdgeZones`）。浮动拖拽同样受益 —— 浮窗可以贴边
    // 重新停靠到贯通全高/全宽的位置。
    const root = document.querySelector<HTMLElement>("[data-dock-root]");
    if (root) {
        const rect = root.getBoundingClientRect();
        if (rect.width >= 1 && rect.height >= 1) {
            const band = store.getState().dock.settings.edgeBandPx;
            out.push(
                ...buildRootEdgeZones(
                    { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
                    band,
                ),
            );
        }
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
    // 合成 Zone（根级边缘带）整条带就是一个部位，部位已在采集时固定；
    // 普通标签组 Zone 现场按指针在矩形内的位置解析。
    const band = store.getState().dock.settings.edgeBandPx;
    const zone: DockDropZone =
        hit.fixedZone ?? resolveDropZone(hit.rect, { x, y }, band) ?? "center";
    // 根级带的提交/预览基准是整个停靠区矩形（"贯通整侧"），不是那条细带。
    return { zoneId: hit.zoneId, zone, rect: hit.previewRect ?? hit.rect };
}

/**
 * 计算标签拖拽的重排下标（`addFormToTabset` 的**摘除后**下标）。
 *
 * DOM 里的标签矩形包含被拖标签自己，按"第一个中线在指针右侧的标签"得到的
 * 下标是**摘除前**的；先摘除再插入时，凡插入点在被拖标签原位置之后都要左移
 * 一格，否则跨过相邻标签中线就会一次跳两个位置（悬停右半 = 越过一个标签）。
 */
function resolveTabIndex(tabsetId: string, formId: string, x: number): number | null {
    const tabs = document.querySelectorAll<HTMLElement>(
        `[data-dock-tabbar="${tabsetId}"] [data-dock-tab]`,
    );
    if (tabs.length === 0) return null;
    let index = tabs.length;
    let currentIndex = -1;
    for (let i = 0; i < tabs.length; i += 1) {
        const rect = tabs[i].getBoundingClientRect();
        if (tabs[i].dataset.dockTab === formId) currentIndex = i;
        if (x < rect.left + rect.width / 2) {
            index = i;
            break;
        }
    }
    // 中点循环可能没走到被拖标签（index 已先命中）也没关系：它只影响"原位置
    // 在插入点之前"这一判断，-1（DOM 尚未更新）时按未后移处理。
    if (currentIndex !== -1 && currentIndex < index) index -= 1;
    return index;
}

/**
 * 判定指针此刻是否处于源标签条的**重排带**内（条带 ± 少量外扩）。
 *
 * 重排带内 = 松手是同组调整顺序（`tabCount > 1` 才有顺序可调），与
 * `onPointerUp` 的提交分支共用同一个判定，保证"看到的反馈"与"松手结果"一致。
 */
function inReorderBand(active: DragSession, x: number, y: number): boolean {
    return (
        active.mode === "tab" &&
        active.tabCount > 1 &&
        active.tabsetId !== null &&
        active.tabBarRect !== null &&
        pointInInflated(active.tabBarRect, x, y, TAB_BAND_SLOP_PX)
    );
}

/**
 * 实时重排：指针在重排带内每跨过一个相邻标签的中线，就立刻提交一次顺序。
 *
 * 【为什么中途提交而不是只画插入线】插入线仍要用户在脑中补完"松手后会怎样"，
 * 顺序本身动起来才是零学习成本的反馈（Chrome 的标签拖拽）。提交复用普通的
 * `dockFormTo`，与松手提交、快捷键等同一条路径。
 *
 * 【抖动防线】只有当算出的下标与 Redux 里**当前**下标不同才提交 —— DOM 重渲染
 * 与 Redux 更新之间存在一帧的时间差，指针在两次 move 之间没动时，按新矩形算出
 * 的下标必然等于已提交的下标，天然幂等；React 批量渲染导致的短暂 DOM 滞后也
 * 不会造成来回横跳。
 */
function maybeLiveReorder(active: DragSession, x: number, y: number): void {
    if (!inReorderBand(active, x, y) || active.tabsetId === null) return;
    const index = resolveTabIndex(active.tabsetId, active.formId, x);
    if (index === null) return;
    const zone = findZone(store.getState().dock.layout.tree, active.tabsetId);
    if (!zone || zone.t !== "tabset") return;
    const currentIndex = zone.tabs.indexOf(active.formId);
    if (index === currentIndex) return;
    active.reorderMoved = true;
    store.dispatch(
        dockFormTo({
            formId: active.formId,
            target: { kind: "tab", tabsetId: active.tabsetId, index },
        }),
    );
}

function onPointerMove(event: PointerEvent): void {
    if (!session || event.pointerId !== session.pointerId) return;
    const active = session;
    const x = event.clientX;
    const y = event.clientY;
    active.lastX = x;
    active.lastY = y;

    if (!getDockDragState()) {
        const distance = Math.hypot(x - active.startX, y - active.startY);
        if (distance < DRAG_THRESHOLD_PX) return;
        // 【刻意不同帧拆分】越过阈值的这一次移动要与 begin 同帧解析落点。否则：
        // 1) 首帧的落点提示会晚一帧才出现；
        // 2) 指针一次跨越大段距离（触控板快速甩动、事件合并）时就再没有第二次
        //    移动事件，落点永远是 null —— 表现为"按住修饰键拖了却没停靠"。
        scheduleDragUpdate(
            {
                mode: active.mode,
                formId: active.formId,
                panelId: active.panelId,
                pointerX: x,
                pointerY: y,
                dockIntent: isDockModifierDown(event),
                floatRect: active.mode === "float" ? active.floatStart : null,
            },
            computeMovePatch(active, x, y, event),
        );
        return;
    }

    // 实时重排先于本帧的预览补丁：顺序一变，标签条自己就会跟着动 —— 这是
    // 重排带内唯一的反馈，预览补丁同时会把浮动幽灵压掉（见 computeMovePatch）。
    maybeLiveReorder(active, x, y);
    scheduleDragUpdate(null, computeMovePatch(active, x, y, event));
}

/** 一次指针移动要推送的完整补丁：指针位置、停靠意图、落点与浮窗预览矩形。 */
function computeMovePatch(
    active: DragSession,
    x: number,
    y: number,
    intentSource: PointerEvent | KeyboardEvent,
): Partial<DockDragState> {
    /*
     * 不可停靠的面板（`dockable: false`，如外观设置）：拖拽照常移动窗口，但
     * **永不产生落点**，也不显示停靠预览 —— 否则用户会看到"这里能放"的暗示，
     * 松手却发现它停不进去（或更糟：真的停进去了，设置界面占了主编辑区一格）。
     */
    const dockIntent = active.dockable && isDockModifierDown(intentSource);
    // 重排带内没有"落点"可言：松手是调整顺序。落点置空让覆盖层的停靠预览
    // （含根级边缘带的贯通预览）退场，浮动预览被 `reorder` 标志显式压掉 ——
    // 否则用户只想交换标签位置，却看到浮窗轮廓跟着鼠标跑（用户报告）。
    const reorder = active.dockable && inReorderBand(active, x, y);
    const target = reorder || !active.dockable ? null : resolveTarget(x, y);
    // 浮窗搬运与标签拖拽都要算出"若此刻松手，浮窗会落在哪、多大" —— 覆盖层
    // 据此画虚线轮廓。没有它，不按修饰键拖拽时用户完全看不到结果。两种模式的
    // 换算相同：标签拖拽用记住的浮窗尺寸（`sourceSize`），搬运沿用当前几何。
    return {
        pointerX: x,
        pointerY: y,
        dockIntent,
        target,
        reorder,
        floatRect: reorder ? null : computePendingFloatRect(active, x, y, zoneRects),
    };
}

/** 计算"此刻松手会得到的浮窗矩形"（已夹紧到视口内）。 */
function computePendingFloatRect(
    active: DragSession,
    x: number,
    y: number,
    zones: readonly DockZoneRect[],
): DockRect {
    const settings = store.getState().dock.settings;
    const raw: DockRect = {
        x: x - active.offsetX,
        y: y - active.offsetY,
        w: active.sourceSize.w,
        h: active.sourceSize.h,
    };
    const snapped = settings.floatSnapEnabled
        ? snapFloatPosition(
              raw,
              zones.map((zone) => zone.rect),
              { w: window.innerWidth, h: window.innerHeight },
              settings.floatSnapThresholdPx,
          )
        : { x: raw.x, y: raw.y };
    return clampFloatRect(
        { ...raw, x: snapped.x, y: snapped.y },
        { w: window.innerWidth, h: window.innerHeight },
        FLOAT_TITLE_BAR_PX,
    );
}

function onPointerUp(event: PointerEvent): void {
    if (!session || event.pointerId !== session.pointerId) return;
    const active = session;
    // 帧合并中积压的补丁必须先同步落地：提交要从 dockDragStore 读最终落点
    // （state.target / state.floatRect），否则会用上一帧的位置提交。
    flushDragUpdate();
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
        // 1) 仍在标签条带内 → 同组重排（**先于**停靠意图判定：条带内的意图就是
        //    "调整这一组的顺序"，按住修饰键也不例外 —— 否则"始终停靠"模式下
        //    组内重排会被并入判定劫持成"移到组尾"，永远无法交换相邻标签）。
        if (inReorderBand(active, x, y) && active.tabsetId !== null) {
            const index = resolveTabIndex(active.tabsetId, active.formId, x);
            if (index === null) return;
            const zone = findZone(store.getState().dock.layout.tree, active.tabsetId);
            if (zone?.t === "tabset" && index !== zone.tabs.indexOf(active.formId)) {
                store.dispatch(
                    dockFormTo({
                        formId: active.formId,
                        target: { kind: "tab", tabsetId: active.tabsetId, index },
                    }),
                );
            }
            return;
        }
        // 2) 按住修饰键 → 停靠；落空则回弹（什么都不做）。
        if (dockIntent) {
            if (!target) return;
            commitDock(active.formId, target);
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
                geometry: {
                    x: geometry.x,
                    y: geometry.y,
                    // 尺寸用 dropSize 而不是拖拽预览矩形：折叠为标签条的浮窗
                    // 拖拽时预览是条状，落库要保留展开尺寸（见 DragSession）。
                    w: active.dropSize.w,
                    h: active.dropSize.h,
                },
            }),
        );
    }
}

/** 把落点提交为"并入标签组"、"在某一侧拆分"或"贯通整个停靠区的某一侧"。 */
function commitDock(formId: string, target: DockDropTargetState): void {
    const side = dropZoneToSide(target.zone);
    const layout = store.getState().dock.layout;

    // 根级边缘带：以整个停靠区为参照拆分（贯通全高/全宽），提交走
    // `{kind:"root"}`，与"打开面板时的默认落点"（`placeForm`）同一语义。
    if (target.zoneId === DOCK_ROOT_ZONE_ID) {
        if (side !== null) {
            store.dispatch(dockFormTo({ formId, target: { kind: "root", side } }));
        }
        return;
    }

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

/**
 * 取消拖拽：不提交任何落点，界面回到拖拽前的样子。
 *
 * 【为什么基本"什么都不提交"就是恢复】浮动预览与落点提示都只是 dockDragStore
 * 的临时状态，随 endDockDrag 一并消失。唯一的例外是**实时重排**：它为了把反馈
 * 做成"顺序真的在动"，拖拽中途就提交了顺序 —— 取消时按会话里留底的起始下标
 * 恢复原状，保证 Escape 的语义仍然是"就当我没拖过"。
 */
function cancelDockDrag(): void {
    if (!session) return;
    const active = session;
    session = null;
    // 帧合并里可能还积着一份补丁：取消后绝不能再推送（会把状态"复活"一帧）。
    discardScheduledUpdate();
    detach();
    endDockDrag();
    if (
        active.mode === "tab" &&
        active.reorderMoved &&
        active.tabsetId !== null &&
        active.reorderBaseIndex !== null
    ) {
        store.dispatch(
            dockFormTo({
                formId: active.formId,
                target: {
                    kind: "tab",
                    tabsetId: active.tabsetId,
                    index: active.reorderBaseIndex,
                },
            }),
        );
    }
}

function onPointerCancel(event: PointerEvent): void {
    if (!session || event.pointerId !== session.pointerId) return;
    // OS 级取消（掌压拒绝、IME 抢占、手势竞争）：绝不能走 onPointerUp ——
    // 那会把落点提交，用户明明没有松手却看到面板被停靠/浮走。
    cancelDockDrag();
}

function onKeyUp(event: KeyboardEvent): void {
    if (!isModifierKey(event.key)) return;
    refreshIntent(event);
}

function onKeyDown(event: KeyboardEvent): void {
    // Escape = 取消拖拽（对齐资源管理器 / VS Code 的拖拽语义）。捕获阶段监听
    // （见 attach），避免按键先被聚焦控件消费。Escape 与 pointerId 无关：
    // 会话还在就取消。
    if (session && event.key === "Escape") {
        event.preventDefault();
        cancelDockDrag();
        return;
    }
    if (!isModifierKey(event.key)) return;
    refreshIntent(event);
}

function isModifierKey(key: string): boolean {
    return key === "Control" || key === "Meta" || key === "Alt" || key === "Shift";
}

/**
 * 修饰键状态变化时，**不等下一次指针移动**就重算停靠意图与落点。
 *
 * 【为什么必须有】意图原先只在 `pointermove` 里解析，于是"拖到目标上方 → 按住
 * 停靠修饰键 → 松手"这条最自然的路径完全失效：按住修饰键本身不产生任何指针
 * 事件，幽灵不出现、`target` 仍是 null，松手时 `dockIntent && target` 判定失败
 * —— 表现为"按住修饰键后必须再动一下鼠标才会出现停靠幽灵"（用户报告）。
 * 指针位置是拖拽期间唯一会变且我们已经记录的输入，据此重算即可，无需真的移动。
 *
 * 松开修饰键同样走这里（而非只把 `target` 置空）：`dockModifier` 可以是
 * `shift`/`alt`，松开其中一个修饰键时另一个仍可能让意图成立，落点必须跟着重算。
 */
function refreshIntent(event: PointerEvent | KeyboardEvent): void {
    if (!session) return;
    const state = getDockDragState();
    if (!state) return;
    const dockIntent = isDockModifierDown(event);
    // 与 computeMovePatch 同一套重排带判定：悬停在源标签条上时没有"落点"，
    // 修饰键变化也不能把停靠预览"复活"到条带里。
    const reorder = inReorderBand(session, session.lastX, session.lastY);
    const target = reorder ? null : resolveTarget(session.lastX, session.lastY);
    // 比较基准要包含**尚未推送**的补丁：rAF 合并意味着 store 里的值可能落后于
    // "已经算出"的值，拿旧值去比会漏掉需要更新的情形（快速按下又松开修饰键）。
    const effective = pendingPatch ? { ...state, ...pendingPatch } : state;
    // 按住修饰键时键盘会**重复**触发 keydown（约 30Hz）：意图没变就不该惊动
    // 覆盖层。落点只看 zone 与部位，矩形由 Zone 决定，无需逐值比较。
    if (
        effective.dockIntent === dockIntent &&
        effective.reorder === reorder &&
        (effective.target?.zoneId ?? null) === (target?.zoneId ?? null) &&
        (effective.target?.zone ?? null) === (target?.zone ?? null)
    ) {
        return;
    }
    if (pendingPatch) {
        // 与指针移动共用同一份 rAF 补丁：这里只覆盖意图与落点，位置等字段保留
        // 指针移动已经算出的值（keydown 与 pointermove 可能落在同一帧内）。直接
        // 覆盖整个补丁会把位置回退到上一帧；不覆盖则下一帧的旧补丁又会把刚改
        // 的意图冲回去。
        pendingPatch = { ...pendingPatch, dockIntent, target, reorder };
        return;
    }
    updateDockDrag({ dockIntent, target, reorder });
}

function onWindowResize(): void {
    if (!session) return;
    zoneRects = collectZoneRects();
}

function attach(): void {
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerCancel);
    // 捕获阶段：修饰键的 keydown/keyup 可能先被聚焦控件消费（标签栏、编辑器
    // 等），冒泡阶段收不到就又会退回"必须动一下鼠标"的老问题。Escape 的取消
    // 也依赖这一点。
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("resize", onWindowResize);
}

function detach(): void {
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    window.removeEventListener("pointercancel", onPointerCancel);
    window.removeEventListener("keydown", onKeyDown, true);
    window.removeEventListener("keyup", onKeyUp, true);
    window.removeEventListener("resize", onWindowResize);
}

/**
 * 把后续指针事件捕获到拖拽源元素上。
 *
 * 【为什么需要】指针滑出源元素（甚至滑出窗口）后，事件仍定向到该元素并冒泡到
 * window，拖拽路径不会因为换了 hit-test 目标而中断。某些目标或旧实现不支持
 * 捕获：失败只是退化为纯 window 监听，不能让拖拽本身失败。
 */
function capturePointer(event: React.PointerEvent): void {
    try {
        event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
        // 忽略：window 级监听无论如何都能收到后续事件。
    }
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
    // 一次只允许一个拖拽会话：第二根手指按在另一个标签上时直接忽略 —— 会话是
    // 模块单例，被覆盖后第一个拖拽的 move/up 全部被 pointerId 不匹配拦下，
    // 两个手势会一起坏掉（卡在"拖拽中"且再也收不了尾）。
    if (session) return;
    const tabBarRect = args.tabBarElement?.getBoundingClientRect() ?? null;

    // 【拆成浮窗时用多大】用**记住的浮窗尺寸**（或面板默认值），而不是宿主的
    // 当前尺寸。后者是停靠态的尺寸 —— 用户把一个 300×200 的小浮窗停进一大片
    // 区域后，面板会被撑大；此时再拖出来若沿用宿主尺寸，他就会得到一个巨大的
    // 浮窗，而当初那个大小已经无从找回。浮动态与停靠态的尺寸是两个独立意图，
    // 必须分开保存（见 `DockForm.float`）。
    const form = store.getState().dock.layout.forms[args.formId];
    const definition = getPanel(form?.panelId ?? args.panelId);
    const sourceSize = {
        w: Math.max(200, Math.round(form?.float?.w ?? definition?.defaultWidth ?? 420)),
        h: Math.max(120, Math.round(form?.float?.h ?? definition?.defaultHeight ?? 320)),
    };
    // 实时重排会中途提交顺序，取消时按这个下标把源组恢复原状。
    const sourceZone = findZone(store.getState().dock.layout.tree, args.tabsetId);
    const reorderBaseIndex =
        sourceZone && sourceZone.t === "tabset" ? sourceZone.tabs.indexOf(args.formId) : null;

    session = {
        mode: "tab",
        formId: args.formId,
        panelId: args.panelId,
        // 标签拖拽永远可停靠：不可停靠的面板根本不会出现在标签组里。
        dockable: true,
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
        reorderBaseIndex,
        reorderMoved: false,
        floatStart: null,
        sourceSize,
        // 标签拖出去新建浮窗：没有"折叠"语义，落库尺寸 = 预览尺寸。
        dropSize: sourceSize,
        lastX: event.clientX,
        lastY: event.clientY,
    };
    capturePointer(event);
    zoneRects = collectZoneRects();
    attach();
}

export interface FloatDragArgs {
    formId: string;
    panelId: string;
    /** 是否允许停靠（见 `DragSession.dockable`）。缺省允许。 */
    dockable?: boolean;
    geometry: DockRect;
    /**
     * 松手落库的尺寸；缺省 = `geometry` 的尺寸。折叠为标签条的浮窗传
     * 展开后的尺寸（拖拽预览/搬运用条状 geometry，见上）。
     */
    dropSize?: { w: number; h: number };
}

/** 从一个浮动窗的标题栏开始拖拽。 */
export function beginFloatDrag(event: React.PointerEvent, args: FloatDragArgs): void {
    if (event.button !== 0) return;
    // 与 beginTabDrag 相同的会话互斥：拖拽进行中忽略第二根手指。
    if (session) return;
    session = {
        mode: "float",
        formId: args.formId,
        panelId: args.panelId,
        dockable: args.dockable ?? true,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        offsetX: event.clientX - args.geometry.x,
        offsetY: event.clientY - args.geometry.y,
        tabsetId: null,
        tabBarRect: null,
        tabCount: 0,
        reorderBaseIndex: null,
        reorderMoved: false,
        floatStart: args.geometry,
        sourceSize: { w: args.geometry.w, h: args.geometry.h },
        dropSize: {
            w: args.dropSize?.w ?? args.geometry.w,
            h: args.dropSize?.h ?? args.geometry.h,
        },
        lastX: event.clientX,
        lastY: event.clientY,
    };
    capturePointer(event);
    zoneRects = collectZoneRects();
    attach();
}

/** 拖拽是否正在进行（供 CSS 关掉指针事件等）。 */
export function isDockDragging(): boolean {
    return getDockDragState()?.started === true;
}
