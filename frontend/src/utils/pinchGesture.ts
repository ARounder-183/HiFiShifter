/**
 * 触控板捏合手势的识别与累积（纯函数，可单测）。
 *
 * 【它解决什么】Web 平台上报触控板捏合的**唯一**方式是 `ctrlKey`（macOS 上是
 * `metaKey`）+ `wheel`。本工程此前在 `App.tsx` 里对这类事件无条件
 * `preventDefault()`（为了禁用 WebView 的页面缩放），于是**手势被静默丢弃**：
 * 触控板用户既没有浏览器缩放，也没有应用内缩放，捏合完全没反应。
 *
 * 正确做法是继续阻止页面缩放，但把同一批事件**识别成缩放手势**派发给当前表面。
 *
 * 【为什么不能只看 `ctrlKey`】真实鼠标按住 Ctrl 滚轮也会带 `ctrlKey`。两者在
 * Web 层**无法区分**（`WheelEvent` 不带 `pointerType`），因此这里不做区分 ——
 * 而是让"捏合 = 缩放"这件事对两者都成立：Ctrl+滚轮本来就该是缩放。
 *
 * 【单位与锚点】内部以"格"为单位，`PINCH_UNITS_PER_NOTCH = 100` 与既有滚轮路径
 * 的 `Math.max(1, Math.round(|delta| / 100))` 同源；每格的比例系数
 * `PINCH_FACTOR_PER_NOTCH = 1.1` 与 `PianoRollPanel` 既有的 `0.9 / 1.1` 一致。
 * 于是连续捏合在"一格"的粒度上**恰好复现**既有手感，只是不再有台阶。
 *
 * 【设计约束】纯函数 + 常量，不接触 DOM / React（总线除外 —— 它只做订阅转发），
 * 可在 node 环境完整单测。
 */

/** 一格捏合对应的 wheel delta 单位数（与既有滚轮步进同源）。 */
export const PINCH_UNITS_PER_NOTCH = 100;

/** 一格对应的缩放比例（与 `PianoRollPanel` 既有的 0.9 / 1.1 一致）。 */
export const PINCH_FACTOR_PER_NOTCH = 1.1;

/**
 * 两次捏合事件间隔超过这个时长即视为**新手势**（重新计比例）。
 *
 * 【为什么需要】捏合是"连续的若干个小事件"，而结束没有专门的信号 ——
 * 触控板不会上报 `pointerup`。用间隔判定是唯一可行的办法。
 */
export const PINCH_GESTURE_GAP_MS = 180;

/** 从 wheel 事件抽取捏合增量；非捏合返回 `null`。 */
export function pinchDeltaFromWheel(event: {
    deltaY: number;
    deltaMode?: number;
    ctrlKey: boolean;
    metaKey: boolean;
}): number | null {
    if (!event.ctrlKey && !event.metaKey) return null;
    if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return null;
    // deltaMode: 0 = 像素，1 = 行，2 = 页。后两者按行高 16px / 页 400px 折算，
    // 与 `normalizeWheel` 的既有约定一致（宁可粗一点，也不要漏掉整段手势）。
    const mode = event.deltaMode ?? 0;
    const scale = mode === 1 ? 16 : mode === 2 ? 400 : 1;
    const pixels = event.deltaY * scale;
    // 捏合张开（放大）时 deltaY 为负 → 取负号使"正 = 放大"，与直觉一致。
    return -pixels;
}

/** 一次捏合手势的累积状态。 */
export interface PinchState {
    /** 自本次手势开始累计的单位数（可为负 = 缩小）。 */
    accumulated: number;
    /** 上一次事件的时间戳（`performance.now()` 或 `Date.now()`，同源即可）。 */
    lastAt: number;
}

/** 新建一份捏合状态。 */
export function createPinchState(): PinchState {
    return { accumulated: 0, lastAt: 0 };
}

/**
 * 推进一次捏合，返回**自本次手势开始累计**的单位数。
 *
 * 超过 `PINCH_GESTURE_GAP_MS` 没有事件即视为新手势，累计量归零 ——
 * 否则上一轮捏合的余量会带进下一轮，表现为"刚碰板子就跳一下"。
 */
export function accumulatePinch(state: PinchState, delta: number, now: number): number {
    if (!Number.isFinite(delta)) return state.accumulated;
    if (!Number.isFinite(now)) now = state.lastAt;
    if (now - state.lastAt > PINCH_GESTURE_GAP_MS) {
        state.accumulated = 0;
    }
    state.accumulated += delta;
    state.lastAt = now;
    return state.accumulated;
}

/**
 * 累计单位数 → 缩放比例（1 = 未缩放）。
 *
 * 按"每格 1.1 倍"折算：`factor = 1.1 ** (units / 100)`。这是既有离散规则的
 * 连续推广，因此捏合与滚轮在同一台机器上不会给出两套手感。
 */
export function pinchZoomFactor(units: number): number {
    if (!Number.isFinite(units)) return 1;
    return Math.pow(PINCH_FACTOR_PER_NOTCH, units / PINCH_UNITS_PER_NOTCH);
}

/** 整步累积器（供只接受整数步的表面使用）。 */
export interface PinchStepState {
    /** 尚未凑满一格的余量。 */
    residual: number;
    lastAt: number;
}

/** 新建一份整步累积状态。 */
export function createPinchStepState(): PinchStepState {
    return { residual: 0, lastAt: 0 };
}

/**
 * 把捏合增量折算成**整步**，返回本帧应当走的步数（可为 0 / 负）。
 *
 * 【为什么必须累积而不是每帧至少一步】既有滚轮路径用
 * `Math.max(1, Math.round(|delta| / 100))` —— 那是给"一格一个事件"的机械滚轮
 * 设计的。触控板捏合吐出的是**一串**连续的小 delta，每个都 `max(1, …)` 会让
 * 深度/缩放以事件频率飞走。累积到满一格才走一步，与滚轮同速。
 */
export function accumulatePinchSteps(state: PinchStepState, delta: number, now: number): number {
    if (!Number.isFinite(delta)) return 0;
    if (!Number.isFinite(now)) now = state.lastAt;
    if (now - state.lastAt > PINCH_GESTURE_GAP_MS) {
        state.residual = 0;
    }
    state.residual += delta;
    state.lastAt = now;
    const steps = Math.trunc(state.residual / PINCH_UNITS_PER_NOTCH);
    if (steps !== 0) {
        state.residual -= steps * PINCH_UNITS_PER_NOTCH;
    }
    return steps;
}

// ── 捏合总线 ────────────────────────────────────────────────────────
//
// 【为什么用总线而不是让每个表面自己监听 wheel】全局的"禁用浏览器缩放"守卫只有
// 一个（`App.tsx` 的 capture 监听），它已经拿到了全部捏合事件。各表面再各挂一个
// capture 监听会重复、顺序也不确定。总线让 App 做唯一的识别点，表面只订阅。

/** 一次捏合事件（已识别为捏合，且页面缩放已被阻止）。 */
export interface PinchEvent {
    /** 指针位置（CSS 像素，视口坐标）。 */
    clientX: number;
    clientY: number;
    /** 本次事件的捏合增量（正 = 放大）。 */
    delta: number;
}

type PinchListener = (event: PinchEvent) => void;

const pinchListeners = new Set<PinchListener>();

/**
 * 订阅捏合事件。
 *
 * @returns 取消订阅的函数（组件卸载时必须调用）。
 */
export function subscribePinch(listener: PinchListener): () => void {
    pinchListeners.add(listener);
    return () => {
        pinchListeners.delete(listener);
    };
}

/** 派发一次捏合事件（由 App 的全局 wheel 守卫调用）。 */
export function emitPinch(event: PinchEvent): void {
    for (const listener of [...pinchListeners]) {
        try {
            listener(event);
        } catch {
            // 单个订阅者抛错不得打断其余订阅者（一个坏掉的表面不该让整个手势消失）。
        }
    }
}

/** 测试辅助：清空订阅者。 */
export function clearPinchListeners(): void {
    pinchListeners.clear();
}
