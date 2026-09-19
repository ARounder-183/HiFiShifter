/**
 * 全局「右键拖拽收尾不得弹菜单」守卫。
 *
 * 【要修的问题】在任一处右键拖拽（时间轴内框选音频块、参数编辑器内右键拖拽
 * 调整曲线等），把指针移到**另一个表面**（左侧轨道头、时间轴标尺、参数编辑器
 * 标尺）再松开右键时，会弹出**那个表面**的右键菜单。
 *
 * 【根因：菜单由「松开时指针所在的元素」决定，而抑制标记只覆盖发起手势的表面】
 * Windows/WebView2 在**松开**右键时才派发 `contextmenu`，其 target 是指针松开
 * 位置下的元素 —— 也就是被划过的那个表面。各发起方本来都装了抑制（时间轴内核
 * 的 `suppressNextContextMenu` 挂在轨道区容器上、参数编辑器的 `suppressOnce`
 * 挂在 window 上且只活到 `setTimeout(0)`），但：
 * - 容器范围的监听收不到落在兄弟子树（标尺 / 轨道头）的事件；
 * - `setTimeout(0)` 拆除的监听在 Linux/WebKitGTK 上会输给 App.tsx
 *   `flushLinuxDeferredContextMenu` 的 `setTimeout(0)` **重派发**（同一宏任务
 *   队列，拆除先于重派发执行），于是抑制失效。
 *
 * 【为什么在 window 捕获阶段统一拦截】`contextmenu` 的传播顺序是
 * window → document → … → target。在 window 捕获阶段 `stopImmediatePropagation`
 * 能同时挡掉 App.tsx 的 document 级处理与目标表面的 React 处理器 —— 与手势
 * 从哪个组件发起无关，因此本守卫是**跨表面**的，不需要每个表面各自加判断。
 *
 * 【闸门如何收口（为什么不用定时器）】拖拽收尾时"武装"守卫，命中一次即吞掉并
 * 解除武装；**下一次 `pointerdown` 无条件解除武装**。右键拖拽的事件序列是
 * pointerdown → pointermove… → pointerup → contextmenu，因此"收尾到下一次
 * 按下之间"恰好就是要吞掉的那一次事件；而用户之后真正想开菜单的右键点击必然
 * 先经过 pointerdown（解除武装），不会被误吞。这比时间窗更确定：不依赖平台
 * 派发 `contextmenu` 的时机（按下即派发 / 松开即派发 / 延迟重派发都能覆盖），
 * 也不存在与 `setTimeout(0)` 抢跑的竞态。
 */

/**
 * 判定为「右键拖拽」的最小位移（CSS px），与时间轴框选阈值同源
 * （`BOX_SELECT_THRESHOLD_PX` / `TIMELINE_SELECTION_DRAG_THRESHOLD_PX`）。
 * 手抖级别的位移不应被判成拖拽，否则会把正常右键点击的菜单误吞。
 */
export const RIGHT_DRAG_THRESHOLD_PX = 5;

/**
 * 纯判定：右键按下到松开之间的位移是否已构成拖拽（超过阈值）。
 *
 * @param dx 水平位移（CSS px）。
 * @param dy 竖直位移（CSS px）。
 * @param thresholdPx 阈值，默认 `RIGHT_DRAG_THRESHOLD_PX`。
 * @returns 超过阈值返回 true。
 */
export function isRightDragBeyondThreshold(
    dx: number,
    dy: number,
    thresholdPx: number = RIGHT_DRAG_THRESHOLD_PX,
): boolean {
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return false;
    return dx * dx + dy * dy >= thresholdPx * thresholdPx;
}

/**
 * 纯判定：本次 `contextmenu` 是否应被吞掉（并据此解除武装）。
 *
 * @param armed 守卫是否已武装（右键拖拽刚收尾）。
 * @returns 应吞掉时返回 true；调用方负责 `preventDefault` /
 *   `stopImmediatePropagation` 以及解除武装。
 */
export function shouldSwallowContextMenu(armed: boolean): boolean {
    return armed;
}

/**
 * 显式武装守卫（供**自己知道**何时构成拖拽的手势调用）。
 *
 * 【为什么不只靠位移阈值】全局守卫只能按位移猜测"这次算不算拖拽"，而各手势的
 * 阈值并不相同（时间轴框选 5px、参数编辑器右键拖拽仅 2px）。手势在自己的阈值
 * 处显式调用本函数，判定就是**精确**的，不依赖猜测，也不会因为阈值取小而把
 * 正常右键点击的菜单误吞。
 *
 * 幂等；随后第一次 `contextmenu` 会吞掉它并自动解除（见 `onContextMenuCapture`）。
 */
export function armRightDragContextMenuGuard(): void {
    armed = true;
}

type ListenerTarget = {
    addEventListener(
        type: string,
        listener: EventListener,
        options?: boolean | AddEventListenerOptions,
    ): void;
    removeEventListener(
        type: string,
        listener: EventListener,
        options?: boolean | EventListenerOptions,
    ): void;
};

/** 当前是否已武装（右键拖拽刚收尾，下一次 contextmenu 应被吞掉）。 */
let armed = false;
/** 右键按下的起点；未按下时为 null。 */
let pressStart: { x: number; y: number } | null = null;
let installed = false;

function windowTarget(): ListenerTarget | null {
    const g = globalThis as { window?: ListenerTarget };
    return g.window ?? null;
}

function documentTarget(): ListenerTarget | null {
    const g = globalThis as { document?: ListenerTarget };
    return g.document ?? null;
}

function asMouseEvent(event: Event): MouseEvent | null {
    return event as MouseEvent;
}

function onPointerDownCapture(event: Event): void {
    // 新的指针交互开始：上一次右键拖拽遗留的武装必须失效，否则会误吞这一次
    // 交互真正需要的菜单（与内核 `suppressNextContextMenu` 在 pointerdown
    // 复位的既有约定同源）。
    armed = false;
    const e = asMouseEvent(event);
    pressStart = e && e.button === 2 ? { x: e.clientX, y: e.clientY } : null;
}

function onPointerMoveCapture(event: Event): void {
    // 拖拽一旦越过阈值就**立即**武装，而不是等到 pointerup：有些平台在右键
    // **按下**时就派发 `contextmenu`（App.tsx 在 Linux/WebKitGTK 上把它推迟到
    // pointerup 重派发），此刻"是否构成拖拽"只能靠位移判定。提前武装让重派发的
    // 那一次事件也落在"已武装"窗口内，不依赖平台派发时机。
    const start = pressStart;
    const e = asMouseEvent(event);
    if (!e || start === null) return;
    if (!isRightDragBeyondThreshold(e.clientX - start.x, e.clientY - start.y)) return;
    armed = true;
}

function onPointerUpCapture(event: Event): void {
    const start = pressStart;
    pressStart = null;
    const e = asMouseEvent(event);
    if (!e || e.button !== 2 || start === null) return;
    // 兜底：指针事件稀疏（快速甩动、事件合并）时 pointermove 可能没给出
    // 越阈值的样本，这里再判一次位移。
    if (isRightDragBeyondThreshold(e.clientX - start.x, e.clientY - start.y)) {
        armed = true;
    }
}

function onPointerCancelCapture(): void {
    pressStart = null;
}

function onContextMenuCapture(event: Event): void {
    if (!shouldSwallowContextMenu(armed)) return;
    armed = false;
    // 吞掉这次事件：同时挡掉 App.tsx 的 document 级处理与目标表面的
    // React 处理器（window 捕获阶段先于两者执行）。
    event.preventDefault();
    event.stopImmediatePropagation();
}

function onBlurCapture(): void {
    // 失焦即不再有"刚结束的拖拽"：解除武装，避免切回窗口后误吞一次菜单。
    armed = false;
    pressStart = null;
}

function onKeyDownCapture(event: Event): void {
    // 键盘请求菜单（Menu 键 / Shift+F10）同样会产生一次 `contextmenu`，而这条路
    // 不经过 `pointerdown`，得不到上面的复位。若不在这里解除，用户右键拖拽后
    // 用键盘请求的菜单会被这一次遗留的武装吞掉。
    //
    // 只认这两个键（而不是任意按键）：避免用一个过宽的复位条件削弱守卫。
    const e = event as KeyboardEvent;
    if (e.key === "ContextMenu" || (e.key === "F10" && e.shiftKey)) {
        armed = false;
        pressStart = null;
    }
}

/**
 * 挂载全局守卫（幂等）。由 App 的全局监听 effect 调用一次，卸载时调
 * `disposeRightDragContextMenuGuard`。
 */
export function installRightDragContextMenuGuard(): void {
    if (installed) return;
    const win = windowTarget();
    const doc = documentTarget();
    if (!win || !doc) return;
    installed = true;
    win.addEventListener("pointerdown", onPointerDownCapture, true);
    win.addEventListener("pointermove", onPointerMoveCapture, true);
    win.addEventListener("pointerup", onPointerUpCapture, true);
    win.addEventListener("pointercancel", onPointerCancelCapture, true);
    win.addEventListener("blur", onBlurCapture);
    win.addEventListener("contextmenu", onContextMenuCapture, true);
    win.addEventListener("keydown", onKeyDownCapture, true);
}

/** 卸载全局守卫（幂等）。 */
export function disposeRightDragContextMenuGuard(): void {
    if (!installed) return;
    installed = false;
    armed = false;
    pressStart = null;
    const win = windowTarget();
    if (!win) return;
    win.removeEventListener("pointerdown", onPointerDownCapture, true);
    win.removeEventListener("pointermove", onPointerMoveCapture, true);
    win.removeEventListener("pointerup", onPointerUpCapture, true);
    win.removeEventListener("pointercancel", onPointerCancelCapture, true);
    win.removeEventListener("blur", onBlurCapture);
    win.removeEventListener("contextmenu", onContextMenuCapture, true);
    win.removeEventListener("keydown", onKeyDownCapture, true);
}
