/*
 * 停靠系统的命令式 API。
 *
 * 【为什么要单独一层】菜单、快捷键、工具栏按钮，以及将来的公开插件 API，
 * 需要的是"打开记事本""把当前窗体浮起来""套用混音布局"这类动作，而不是
 * Redux action 的形状。把它们收在这里，有三点好处：
 *
 * 1. 菜单与快捷键共享同一份行为，"菜单能做的快捷键做不了"这类分叉不会出现；
 * 2. 需要多步的动作用函数表达（如"最大化"要记住原树），不必往 reducer 里塞
 *    临时字段；
 * 3. 将来开放 API 时，暴露的就是这几个函数 —— 内部实现不需要为 API 再改一次。
 *
 * 所有函数都接受 `dispatch` / `getState` 而不是自己 import store，便于测试与
 * 复用（例如在一个未挂载到主 store 的场景里）。
 */

import type { AppDispatch, RootState } from "../../app/store";
import { reportFrontendError } from "../../services/frontendErrorLog";
import { translateOutsideReact } from "../../i18n/I18nProvider";
import { clampFloatRect, resolveFloatNearRect, resolveFloatRect } from "./dockDropTarget";
import { findMainTabset, normalizeDockLayout } from "./dockSchema";
import { collectTabsets, isFormVisible } from "./dockTree";
import { getPanel, listPanels } from "./panelRegistry";
import {
    applyDockPreset,
    closeForm,
    deleteDockPreset,
    dockFormTo,
    floatForm,
    focusForm,
    openPanel,
    resetDockLayout,
    saveDockPreset,
    setDockLayout,
    setFloatGeometry,
    setFormFloatMode,
    toggleMaximizeActive,
} from "./dockSlice";
import {
    clampScreenRectToMonitor,
    floatRectToWindowRect,
    windowRectToFloatRect,
    type GeometryRect,
    type MainWindowFrame,
} from "./detachedGeometry";
import {
    closeDetachedWindow,
    detachedWindowExists,
    openDetachedWindow,
    readDetachedClientRect,
    readMainMonitorRect,
    readMainWindowFrame,
    watchDetachedWindow,
    type DetachedClientRect,
} from "./detachedWindow";
import type { DockFloatGeometry, DockLayout } from "./dockTypes";

type GetState = () => RootState;

/**
 * 打开面板（已打开则聚焦它）。
 *
 * @param near 触发本次打开的控件矩形（主窗口客户区坐标）。声明了 `openAsFloating`
 *   的面板会优先落在它旁边 —— 见 `resolveFloatNearRect`。
 */
export function openPanelById(
    dispatch: AppDispatch,
    getState: GetState,
    panelId: string,
    near?: GeometryRect | null,
): void {
    const layout = getState().dock.layout;
    const formId = visibleFormFor(layout, panelId);
    if (formId) {
        dispatch(focusForm(formId));
        return;
    }
    dispatch(openPanel({ panelId, float: resolveOpenFloatGeometry(panelId, near) }));
}

/**
 * 按"从哪个控件打开"算出本次浮出的几何（不需要时返回 null，交给面板声明的锚点）。
 *
 * 【为什么放在命令层】要读视口尺寸（`window.inner*`）—— reducer 必须保持纯函数，
 * 这类环境读取只应出现在命令式 API 里。
 */
function resolveOpenFloatGeometry(
    panelId: string,
    near?: GeometryRect | null,
): DockFloatGeometry | null {
    if (!near) return null;
    const spec = getPanel(panelId)?.openAsFloating;
    if (!spec || typeof window === "undefined") return null;
    const rect = resolveFloatNearRect(
        near,
        { w: spec.width, h: spec.height },
        { w: window.innerWidth, h: window.innerHeight },
    );
    // 具体坐标 ⇒ 锚点必须清掉，否则渲染期会被"右下角"覆盖回去。
    return { ...rect, anchor: null };
}

/** 显隐切换：可见则关闭，不可见则打开。 */
export function togglePanelVisible(
    dispatch: AppDispatch,
    getState: GetState,
    panelId: string,
): void {
    const formId = visibleFormFor(getState().dock.layout, panelId);
    if (formId) {
        dispatch(closeForm(formId));
        return;
    }
    dispatch(openPanel({ panelId }));
}

/** 按面板 id 关闭它当前可见的窗体（面板自己的关闭按钮用）。 */
export function closeFormById(dispatch: AppDispatch, getState: GetState, panelId: string): void {
    const formId = visibleFormFor(getState().dock.layout, panelId);
    if (formId) dispatch(closeForm(formId));
}

export function isPanelVisible(getState: GetState, panelId: string): boolean {
    return visibleFormFor(getState().dock.layout, panelId) !== null;
}

/**
 * 选择器工厂：某面板当前是否可见。
 *
 * 面板显隐的**唯一事实源是布局树**（`isFormVisible` 由"在树上或有浮动几何"
 * 派生），不再有第二个布尔标志。这样"工具栏按钮亮着、面板却不在屏幕上"
 * 这类自相矛盾的状态从结构上不可能出现。
 */
export function selectPanelVisible(panelId: string) {
    return (state: RootState): boolean => {
        const layout = state.dock.layout;
        for (const formId of layout.order) {
            if (layout.forms[formId]?.panelId !== panelId) continue;
            if (isFormVisible(layout, formId)) return true;
        }
        return false;
    };
}

function visibleFormFor(layout: DockLayout, panelId: string): string | null {
    for (const formId of layout.order) {
        if (layout.forms[formId]?.panelId !== panelId) continue;
        if (isFormVisible(layout, formId)) return formId;
    }
    return null;
}

/** 当前活动窗体：优先用显式的活动窗体，否则取主标签组的活动标签。 */
export function activeFormId(getState: GetState): string | null {
    const { layout, activeFormId: active } = getState().dock;
    if (active && isFormVisible(layout, active)) return active;
    return findMainTabset(layout)?.active ?? null;
}

/** 停靠 ⇄ 浮动 切换。 */
export function toggleFloatActive(dispatch: AppDispatch, getState: GetState): void {
    const formId = activeFormId(getState);
    if (!formId) return;
    const form = getState().dock.layout.forms[formId];
    if (!form) return;
    if (form.floating) {
        const main = findMainTabset(getState().dock.layout);
        if (!main) return;
        dispatch(dockFormTo({ formId, target: { kind: "tab", tabsetId: main.id } }));
        return;
    }
    dispatch(floatForm({ formId }));
}

/** 聚焦下一个/上一个窗体（在可见窗体之间循环）。 */
export function cycleFocus(dispatch: AppDispatch, getState: GetState, delta: 1 | -1): void {
    const layout = getState().dock.layout;
    const visible = layout.order.filter((formId) => isFormVisible(layout, formId));
    if (visible.length === 0) return;
    const current = activeFormId(getState);
    const index = current ? visible.indexOf(current) : -1;
    const nextIndex = (index + delta + visible.length) % visible.length;
    dispatch(focusForm(visible[nextIndex]));
}

/** 把当前窗体的标签组放大到整片工作区，再按一次还原（REAPER 的最大化）。 */
export function maximizeActive(dispatch: AppDispatch): void {
    dispatch(toggleMaximizeActive());
}

/** 当前是否处于"最大化某窗体"的临时状态。 */
export function isMaximized(getState: GetState): boolean {
    return getState().dock.maximized !== null;
}

/**
 * 正在拆出独立窗口的窗体集合（防重入）。
 *
 * 【为什么必须】拆出是异步的（首个 await 之后才创建窗口）：双击拆出按钮会让两次
 * 调用都通过"窗口尚不存在"的检查并各自 `new WebviewWindow` —— 输的那次创建失败后
 * 会回退成进程内浮窗，最终独立窗口与进程内浮窗**各渲染一份面板**，且独立窗口
 * 从此无人回收。
 */
const detachingFormIds = new Set<string>();

/**
 * 把一个窗体拆到**独立操作系统窗口**（主窗口之外）。
 *
 * 【流程】先在布局里把它标成 `osWindow` 浮动态（这样主窗口立刻不再渲染它，且状态
 * 会随布局持久化），再创建独立窗口。创建失败时**回退**为进程内浮窗 —— 面板绝不会
 * 因为"窗口没开出来"而消失。
 *
 * 【为什么只有部分面板可用】独立窗口是另一个 JS 上下文，面板必须重新挂载；带
 * WebGL 上下文/波形缓存的面板（时间轴、参数编辑器）代价过高，见
 * `PanelDefinition.detachable`。未声明的面板在此直接拒绝，调用方据此提示用户。
 */
export async function detachFormToWindow(
    dispatch: AppDispatch,
    getState: GetState,
    formId: string,
): Promise<{ ok: true } | { ok: false; reason: string }> {
    const state = getState();
    const form = state.dock.layout.forms[formId];
    if (!form) return { ok: false, reason: "form-not-found" };
    const definition = getPanel(form.panelId);
    if (!definition?.detachable) return { ok: false, reason: "panel-not-detachable" };
    if (detachingFormIds.has(formId)) return { ok: false, reason: "detach-in-flight" };
    detachingFormIds.add(formId);
    try {
        return await detachFormToWindowInner(dispatch, getState, formId, definition);
    } finally {
        detachingFormIds.delete(formId);
    }
}

async function detachFormToWindowInner(
    dispatch: AppDispatch,
    getState: GetState,
    formId: string,
    definition: NonNullable<ReturnType<typeof getPanel>>,
): Promise<{ ok: true } | { ok: false; reason: string }> {
    const state = getState();
    const form = state.dock.layout.forms[formId];

    // 标题必须**翻译后**交给窗口：`titleKey` 是 i18n 键，直接塞进系统标题栏会显示成
    // "undo_history_title"（用户报告过）。窗体被重命名过时用用户的文本。
    const title = form?.title ?? translateOutsideReact(definition.titleKey);

    // 【原地转换】独立窗口落在浮窗**当前**位置：先把浮窗几何解析成具体矩形（带锚点时
    // 按视口推导，否则 x/y 只是占位值），再换算成屏幕坐标。
    const frame = await readMainWindowFrame();
    const viewport = { w: frame?.clientWidth ?? 0, h: frame?.clientHeight ?? 0 };
    const float = resolveFloatRect(
        form.float ?? {
            x: 0,
            y: 0,
            w: definition.defaultWidth,
            h: definition.defaultHeight,
        },
        viewport.w > 0 ? viewport : { w: 1280, h: 720 },
    );
    const width = float.w;
    const height = float.h;

    // 先落地布局：主窗口立刻停止渲染它，独立窗口随后接管。
    // 只写尺寸、**不写位置**：位置仍由浮窗几何（含锚点）表达，独立窗口只是它的另一种
    // 呈现；写死 x/y 会把"跟随右下角"的锚点语义提前清掉（用户原地拆出再关回来时，
    // 浮窗本该继续跟随窗口尺寸）。
    dispatch(floatForm({ formId, geometry: { w: width, h: height } }));
    dispatch(setFormFloatMode({ formId, floatMode: "osWindow" }));

    const screen = await resolveDetachedScreenPosition(float, frame);
    const result = await openDetachedWindow({ formId, title, width, height, screen });
    if (!result.ok) {
        // 回退：保持进程内浮窗（用户仍能看到并使用它）。
        // **必须留日志**：静默回退让"窗口没开出来"看起来像"功能没实现"（实际发生过）。
        reportFrontendError(`[detach] 独立窗口创建失败，已回退为进程内浮窗：${result.reason}`);
        // 创建"失败"也可能是竞态下另一个调用已把窗口开出来（label 已被占用）。
        // 只有窗口**确实不存在**时才回退进程内浮窗 —— 否则同一名窗体会两处各渲染
        // 一份，而独立窗口从此无人回收。
        if (!(await detachedWindowExists(formId))) {
            dispatch(setFormFloatMode({ formId, floatMode: "inApp" }));
        }
        return result;
    }

    // 窗口关闭即回收为进程内浮窗（最简单的"合回来"语义：关掉就等于收回）。
    // 用**轮询监视**而不是 `onCloseRequested`：后者从主窗口注册收不到事件，
    // 会让窗口永远关不掉（见 `watchDetachedWindow` 的说明）。
    startWatching(dispatch, getState, formId);
    return { ok: true };
}

/**
 * 把浮窗矩形换算成独立窗口的屏幕位置（外框坐标），并夹进显示器范围。
 *
 * 换算不出来（非 Tauri 环境、查询失败）时返回 null —— 此时窗口交给系统默认摆放，
 * 好过用一个猜出来的坐标。
 */
async function resolveDetachedScreenPosition(
    float: GeometryRect,
    frame: MainWindowFrame | null,
): Promise<{ x: number; y: number } | null> {
    if (frame === null) return null;
    const rect = floatRectToWindowRect(float, frame);
    const monitor = await readMainMonitorRect();
    if (monitor === null) return { x: rect.x, y: rect.y };
    // 主窗口贴着屏幕边缘（或跨显示器）时，换算结果可能整块落在显示器之外 —— 那样
    // 用户会得到一个看不见也点不到的窗口，而且它的位置会被记下来，下次依旧不可达。
    const clamped = clampScreenRectToMonitor(rect, monitor);
    return { x: clamped.x, y: clamped.y };
}

/** 把独立窗口里的窗体收回主窗口（停靠回原来的位置由布局决定）。 */
export async function reclaimDetachedForm(
    dispatch: AppDispatch,
    getState: GetState,
    formId: string,
): Promise<void> {
    const form = getState().dock.layout.forms[formId];
    if (!form || form.floatMode !== "osWindow") return;
    // 关窗前抓一次几何：用户可能刚移动/缩放完就直接关掉了窗口（轮询还没轮到）。
    const clientRect = await readDetachedClientRect(formId);
    await closeDetachedWindow(formId);
    if (clientRect) await persistDetachedClientRect(dispatch, getState, formId, clientRect);
    dispatch(setFormFloatMode({ formId, floatMode: "inApp" }));
}

/**
 * 启动时恢复上次拆出的独立窗口。
 *
 * 在布局 hydrate 之后调用一次（见 `finalizeDockHydration`）。只处理**仍然**标记为
 * `osWindow` 且处于浮动状态的窗体 —— 如果用户关掉了"启动时恢复浮窗"，那些窗体此时
 * 已经被收回停靠位，自然不会在这里被重新拆出去。
 */
export async function restoreDetachedWindows(
    dispatch: AppDispatch,
    getState: GetState,
): Promise<void> {
    const layout = getState().dock.layout;
    const detached = layout.order.filter((formId) => {
        const form = layout.forms[formId];
        return form?.floating === true && form.floatMode === "osWindow";
    });
    // 主窗口几何读一次即可：整轮恢复期间它不会变，而换算每个窗口都要用。
    const frame = await readMainWindowFrame();
    for (const formId of detached) {
        const form = layout.forms[formId];
        const definition = getPanel(form.panelId);
        if (!definition?.detachable) {
            // 面板不再支持（例如插件被换掉）：干净地回退为进程内浮窗。
            dispatch(setFormFloatMode({ formId, floatMode: "inApp" }));
            continue;
        }
        const result = await openDetachedWindow({
            formId,
            title: form.title ?? translateOutsideReact(definition.titleKey),
            width: form.float?.w ?? definition.defaultWidth,
            height: form.float?.h ?? definition.defaultHeight,
            // 启动恢复：位置由浮窗几何换算（独立窗口不单独记位置，见
            // `persistDetachedClientRect`）。
            screen: await resolveDetachedScreenPosition(
                resolveFloatRect(
                    form.float ?? {
                        x: 0,
                        y: 0,
                        w: definition.defaultWidth,
                        h: definition.defaultHeight,
                    },
                    frame === null
                        ? { w: 1280, h: 720 }
                        : { w: frame.clientWidth, h: frame.clientHeight },
                ),
                frame,
            ),
        });
        if (!result.ok) {
            reportFrontendError(
                `[detach] 启动恢复独立窗口失败，已回退为进程内浮窗：${result.reason}`,
            );
            dispatch(setFormFloatMode({ formId, floatMode: "inApp" }));
        } else {
            startWatching(dispatch, getState, formId);
        }
    }
}

/**
 * 每个独立窗体的监视停止函数（窗体被回收时停掉，避免重复回收）。
 *
 * 放在模块作用域：窗口的存活期跨越多次 dispatch，没有更合适的 React 宿主。
 */
const stopWatching = new Map<string, () => void>();

/** 开始监视独立窗口的存活；窗口消失即收回（并停掉自己的监视）。 */
function startWatching(dispatch: AppDispatch, getState: GetState, formId: string): void {
    stopWatching.get(formId)?.();
    stopWatching.set(
        formId,
        watchDetachedWindow(formId, {
            onGone: () => {
                stopWatching.get(formId)?.();
                stopWatching.delete(formId);
                // 复用同一回收路径：窗口此时已消失，`closeDetachedWindow` 是空操作，
                // 早退判定也保证重复回调不会重复 dispatch。
                void reclaimDetachedForm(dispatch, getState, formId);
            },
            onClientRect: (clientRect) =>
                void persistDetachedClientRect(dispatch, getState, formId, clientRect),
        }),
    );
}

/**
 * 把独立窗口当前的**客户区**几何换算成浮窗几何写回布局。
 *
 * 【为什么写回浮窗而不是另记一份屏幕坐标】独立窗口就是同一个浮窗的另一种呈现：
 * 关掉它时浮窗要接住窗口的位置与大小（原地转换），因此这里写的就是浮窗的 x/y/w/h，
 * 不引入第二份位置记录（两份记录必然要同步，也必然会有不一致的时候）。
 *
 * 【为什么由主窗口的轮询来写】用户拖动/缩放独立窗口时主窗口收不到事件（跨窗口事件
 * 语义不可靠，见 `watchDetachedWindow`）；轮询上报是唯一稳定时机。写入前先比一次，
 * 只在**真的变了**（>1px）时才 dispatch —— 否则每秒都会改一次布局，把持久化与
 * 订阅者都惊动一遍。
 */
async function persistDetachedClientRect(
    dispatch: AppDispatch,
    getState: GetState,
    formId: string,
    clientRect: DetachedClientRect,
): Promise<void> {
    // 已经被回收（或不再处于独立窗口）时不再写：晚到的轮询结果会把刚恢复的
    // 进程内浮窗位置改掉。
    if (getState().dock.layout.forms[formId]?.floatMode !== "osWindow") return;
    const frame = await readMainWindowFrame();
    if (frame === null) return;
    const raw = windowRectToFloatRect(clientRect, frame);
    // 【必须夹紧】浮窗必须留在主窗口客户区内（至少标题条可达），否则用户会把面板
    // 拖到"点不到"的地方，而且这个位置会被持久化，下次打开依旧不可达。
    const clamped = clampFloatRect(raw, { w: frame.clientWidth, h: frame.clientHeight }, 28);
    const current = getState().dock.layout.forms[formId]?.float ?? null;
    // 比较的是**解析后**的矩形：带锚点时 `float.x/y` 只是占位值，直接比它会每帧都
    // 判定为"变了"，于是第一轮轮询就把锚点清掉 —— 原地拆出再关回来时浮窗本该继续
    // 跟随窗口尺寸（只有用户真的挪动了窗口，才应该改成具体坐标）。
    const resolved =
        current === null
            ? null
            : resolveFloatRect(current, { w: frame.clientWidth, h: frame.clientHeight });
    if (
        resolved !== null &&
        Math.abs(resolved.x - clamped.x) < 1 &&
        Math.abs(resolved.y - clamped.y) < 1 &&
        Math.abs(resolved.w - clamped.w) < 1 &&
        Math.abs(resolved.h - clamped.h) < 1
    ) {
        return;
    }
    dispatch(setFloatGeometry({ formId, geometry: clamped }));
}

/** 某个窗体当前是否在独立窗口中。 */
export function isDetachedForm(getState: GetState, formId: string): boolean {
    return getState().dock.layout.forms[formId]?.floatMode === "osWindow";
}

/**
 * 布局与独立窗口的**结构性对账**。
 *
 * 【为什么需要】只有拆出/回收/启动恢复这三条路径知道独立窗口的存在；而布局可以被
 * 常规路径整体改写 —— 关闭窗体、停靠窗体、套用预设、导入布局。任何一条路径漏掉
 * 同步，就会出现"独立窗口还活着并渲染面板，而布局已不认账"的裂缝：表现为空白
 * 标签、面板双实例、或一个无人回收的幽灵窗口。与其给每条路径打补丁，不如在这里
 * 按最终状态收敛（幂等：无裂缝时不派发任何东西）。
 *
 * 由主窗口在状态变化后调用（见 `DockRoot` 的订阅）。
 */
export async function reconcileDetachedFormWindows(
    dispatch: AppDispatch,
    getState: GetState,
): Promise<void> {
    const layout = getState().dock.layout;

    // 正向裂缝：记录仍标着 `osWindow`，但窗体已不在浮动态（已被关闭/停靠/
    // 预设替换）。回收独立窗口并把 floatMode 复位成 inApp —— 窗体此后就是
    // 一个普通的停靠/已关闭窗体。
    const stale = layout.order.filter((formId) => {
        const form = layout.forms[formId];
        return form != null && form.floatMode === "osWindow" && form.floating !== true;
    });
    for (const formId of stale) {
        await reclaimDetachedForm(dispatch, getState, formId);
    }

    // 反向裂缝：仍有监视中的独立窗口，但布局已不再把该窗体标成 `osWindow` 浮动态
    // （整体替换类路径会把 floatMode 一并改写）。关掉窗口并停止监视，让主窗口
    // 独占面板的挂载权。
    for (const formId of [...stopWatching.keys()]) {
        const form = getState().dock.layout.forms[formId];
        if (form?.floating === true && form.floatMode === "osWindow") continue;
        stopWatching.get(formId)?.();
        stopWatching.delete(formId);
        await closeDetachedWindow(formId);
    }
}

/** 重置为出厂布局。 */
export function resetLayout(dispatch: AppDispatch): void {
    dispatch(resetDockLayout());
}

/** 把当前排布存为命名预设。 */
export function savePreset(dispatch: AppDispatch, name: string): void {
    dispatch(saveDockPreset(name));
}

export function applyPreset(dispatch: AppDispatch, name: string): void {
    dispatch(applyDockPreset(name));
}

export function deletePreset(dispatch: AppDispatch, name: string): void {
    dispatch(deleteDockPreset(name));
}

export function listPresetNames(getState: GetState): string[] {
    return listPresetNamesFromLayout(getState().dock.layout);
}

/**
 * 纯函数版本：按布局对象列出预设名（最近保存的在前）。
 *
 * 组件里应当用它 + `useAppSelector(s => s.dock.layout)`，而不是把 `getState`
 * 传进选择器 —— 后者每次都会构造新数组，`useSyncExternalStore` 会因快照不稳
 * 定而反复重渲染（React 会就此告警）。布局对象的引用只在真正变化时才变，
 * 因此"订阅布局 + 就地派生"是这里的正确形状。
 */
export function listPresetNamesFromLayout(layout: DockLayout): string[] {
    const presets = layout.presets ?? {};
    return Object.keys(presets).sort((a, b) => {
        const at = presets[a]?.createdAtMs ?? 0;
        const bt = presets[b]?.createdAtMs ?? 0;
        return bt - at;
    });
}

/** 导出布局为可分享的 JSON 文本。 */
export function exportLayoutJson(getState: GetState): string {
    return exportLayoutJsonFromLayout(getState().dock.layout);
}

export function exportLayoutJsonFromLayout(layout: DockLayout): string {
    return JSON.stringify(
        {
            kind: "hifishifter.layout",
            schema: layout.schema,
            tree: layout.tree,
            forms: layout.forms,
            order: layout.order,
            floatOrder: layout.floatOrder,
            gutters: layout.gutters,
        },
        null,
        2,
    );
}

/** 导入布局 JSON；返回是否成功（失败时布局不变）。 */
export function importLayoutJson(dispatch: AppDispatch, json: string): boolean {
    let parsed: unknown;
    try {
        parsed = JSON.parse(json);
    } catch {
        return false;
    }
    if (!parsed || typeof parsed !== "object") return false;
    const normalized = normalizeDockLayout(parsed);
    // 归一化一定会产出可用布局，所以"是否导入成功"由调用方通过布局是否变化
    // 之外的信息判断 —— 这里额外校验树非空，避免把垃圾 JSON 当成成功。
    if (collectTabsets(normalized.tree).length === 0) return false;
    dispatch(setDockLayout(normalized));
    return true;
}

export interface DockPanelEntry {
    panelId: string;
    titleKey: string;
    visible: boolean;
    formId: string | null;
}

/** 面板清单（"显示窗体"菜单用）。 */
export function listPanelEntries(getState: GetState): DockPanelEntry[] {
    return listPanelEntriesFromLayout(getState().dock.layout);
}

/** 纯函数版本，理由同 `listPresetNamesFromLayout`。 */
export function listPanelEntriesFromLayout(layout: DockLayout): DockPanelEntry[] {
    return listPanels().map((panel) => {
        const formId = visibleFormFor(layout, panel.id);
        return {
            panelId: panel.id,
            titleKey: panel.titleKey,
            visible: formId !== null,
            formId,
        };
    });
}

/** 供 UI 判断某面板是否注册过（菜单据此隐藏未注册项）。 */
export function isRegistered(panelId: string): boolean {
    return getPanel(panelId) !== undefined;
}
