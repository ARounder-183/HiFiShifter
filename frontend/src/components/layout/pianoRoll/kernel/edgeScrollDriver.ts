/**
 * 参数编辑器内核 · 拖拽边缘自动滚屏驱动。
 *
 * 【主要内容】
 * 1. 时钟：本帧时长按「距**上一帧**」计算，离开触发带即复位；
 * 2. rAF 循环：指针停在边缘不动也持续滚，离开带 / 到边界 / 松手即停；
 * 3. 分派：把「指针位置 + 边界 + 帧时长」交给每个轴的 `apply`。
 *
 * 【它不做什么】
 * - 不知道单位（px 还是参数值）—— 那是 `EdgeScrollAxis` 的事；
 * - 不知道工具、不知道参数块、不写 React state；
 * - 不自己算速度、带宽、软启动（`edgeScrollAxis` / `valueAxisEdgeScroller` 的事）。
 *
 * 【为什么时钟必须「距上一帧」而不是「距上一次 step」】
 * 旧实现在 `step()` 里推进时钟。指针离开触发带时驱动**不排帧**，于是
 * `lastStepAtMs` 停在最后一次真正滚动的那一刻；指针重新进带的第一帧会拿到
 * 「带外空闲时长」（最多 200ms，被 `GESTURE_GAP_RESET_MS` 截断），一次补一大段。
 * 用户来回蹭边缘时看到的"每蹭一次跳一下"就是它。现在的规则是：
 * - 时钟只在带内推进；
 * - 离开带 ⇒ `lastFrameAtMs = null`；
 * - 复位后的第一帧 `frameMs = 0`（位移恒为 0），第二帧才是真实帧时长。
 * 代价是每次进带晚一帧起步（约 16ms），换来的是一条**连续**的速度曲线。
 *
 * 【为什么帧时长每帧只算一次】
 * `frameMsFor` 会推进 `lastFrameAtMs`。若每个轴各调一次，第二个轴永远拿到 0
 * —— 症状是"只有横向会滚、竖向怎么拖都不动"，或反之（取决于轴顺序）。
 *
 * 【违反边界的症状】
 * - 时钟按 step 计：来回蹭边缘时视图一跳一跳；
 * - 帧时长每轴各算一次：只有一个轴生效；
 * - 到达边界后仍排帧：松手前 CPU 空转、每帧标脏重绘。
 */
import type { EdgeScrollBounds } from "../../shared/edgeAutoScroll";
import {
    EDGE_SCROLL_MAX_AXIS_FRAME_MS,
    resolveEdgeScrollRatio,
    type EdgeScrollAxis,
} from "./edgeScrollAxis";

export type { EdgeScrollBounds };

export interface EdgeScrollDriverOptions {
    getBounds: () => EdgeScrollBounds | null;
    /** 本驱动负责的轴。每个轴自带单位、可滚性与手感。 */
    readonly axes: readonly EdgeScrollAxis[];
    /** 本帧确实滚动了。带指针位置，供调用方按同一位置重算。 */
    onScrolled?: (clientX: number, clientY?: number) => void;
    now?: () => number;
    requestFrame?: (cb: () => void) => number;
    cancelFrame?: (id: number) => void;
}

export interface EdgeScrollDriver {
    step: (
        clientX: number,
        clientY?: number,
        onScrolled?: (clientX: number, clientY?: number) => void,
    ) => boolean;
    track: (
        clientX: number,
        clientY?: number,
        onScrolled?: (clientX: number, clientY?: number) => void,
    ) => void;
    stop: () => void;
    isRunning: () => boolean;
}

export function createEdgeScrollDriver(options: EdgeScrollDriverOptions): EdgeScrollDriver {
    const now = options.now ?? (() => performance.now());
    const requestFrame = options.requestFrame ?? ((cb: () => void) => requestAnimationFrame(cb));
    const cancelFrame = options.cancelFrame ?? ((id: number) => cancelAnimationFrame(id));

    let rafId: number | null = null;
    let lastClientX: number | null = null;
    let lastClientY: number | null = null;
    /** 上一帧的时间戳；`null` = 时钟已复位（指针不在带内 / 手势刚开始）。 */
    let lastFrameAtMs: number | null = null;
    /** 最近一帧的时长，供 `tick` 判断「复位帧」是否需要继续排帧。 */
    let lastFrameMs = 0;
    let trackedOnScrolled: ((clientX: number, clientY?: number) => void) | undefined;

    /**
     * 本帧时长：距**上一帧**，不是距上一次 step。
     *
     * 复位后的第一帧返回 0（本帧不滚），让「带外空闲」永远不会被折算成位移。
     */
    const frameMsFor = (atMs: number, maxFrameMs: number): number => {
        if (lastFrameAtMs === null) {
            lastFrameAtMs = atMs;
            return 0;
        }
        const elapsed = atMs - lastFrameAtMs;
        lastFrameAtMs = atMs;
        if (!Number.isFinite(elapsed) || elapsed <= 0) return 0;
        return Math.min(elapsed, maxFrameMs);
    };

    /** 某个轴的指针坐标与边界；竖直轴缺边界时返回 null。 */
    const axisPointer = (
        axis: EdgeScrollAxis,
        clientX: number,
        clientY: number | null,
        bounds: EdgeScrollBounds,
    ): { pointerPx: number; minPx: number; maxPx: number } | null => {
        if (axis.id === "x") {
            return { pointerPx: clientX, minPx: bounds.left, maxPx: bounds.right };
        }
        if (clientY === null || bounds.top === undefined || bounds.bottom === undefined)
            return null;
        return { pointerPx: clientY, minPx: bounds.top, maxPx: bounds.bottom };
    };

    const axisRatio = (
        axis: EdgeScrollAxis,
        clientX: number,
        clientY: number | null,
        bounds: EdgeScrollBounds,
    ): number => {
        const geometry = axisPointer(axis, clientX, clientY, bounds);
        if (geometry === null) return 0;
        return resolveEdgeScrollRatio({ ...geometry, policy: axis.policy });
    };

    const emitScrolled = (
        clientX: number,
        clientY: number | null,
        callback: ((clientX: number, clientY?: number) => void) | undefined,
    ): void => {
        const fn = callback ?? options.onScrolled;
        if (fn === undefined) return;
        if (clientY === null) fn(clientX);
        else fn(clientX, clientY);
    };

    const step = (
        clientX: number,
        clientY?: number,
        onScrolled?: (clientX: number, clientY?: number) => void,
    ): boolean => {
        lastClientX = clientX;
        if (clientY !== undefined) lastClientY = clientY;
        const bounds = options.getBounds();
        if (!bounds) return false;
        // ★ 每帧只算一次：frameMsFor 会推进时钟，两个轴各调一次会让第二个轴拿到 0。
        const frameMs = frameMsFor(now(), EDGE_SCROLL_MAX_AXIS_FRAME_MS);
        lastFrameMs = frameMs;
        let moved = false;
        for (const axis of options.axes) {
            if (!axis.canScroll()) continue;
            const ratio = axisRatio(axis, clientX, lastClientY, bounds);
            if (ratio === 0) continue;
            const axisFrameMs = Math.min(frameMs, axis.maxFrameMs ?? EDGE_SCROLL_MAX_AXIS_FRAME_MS);
            if (axis.apply(ratio, axisFrameMs)) moved = true;
        }
        if (moved) emitScrolled(clientX, lastClientY, onScrolled);
        return moved;
    };

    /** 指针是否落在**任一可滚轴**的触发带内。 */
    const pointerInBand = (clientX: number, clientY: number | null): boolean => {
        const bounds = options.getBounds();
        if (!bounds) return false;
        return options.axes.some(
            (axis) => axis.canScroll() && axisRatio(axis, clientX, clientY, bounds) !== 0,
        );
    };

    const tick = () => {
        rafId = null;
        const clientX = lastClientX;
        if (clientX === null) return;
        const moved = step(clientX, lastClientY ?? undefined, trackedOnScrolled);
        if (!pointerInBand(clientX, lastClientY)) {
            // ★ 离开带 ⇒ 复位时钟。否则下一次进带的第一帧会拿到整段带外空闲。
            lastFrameAtMs = null;
            return;
        }
        // 复位帧（frameMs = 0）不会有位移，但必须继续排帧，否则循环在第一帧就停了。
        if (moved || lastFrameMs === 0) rafId = requestFrame(tick);
    };

    const track = (
        clientX: number,
        clientY?: number,
        onScrolled?: (clientX: number, clientY?: number) => void,
    ): void => {
        lastClientX = clientX;
        if (clientY !== undefined) lastClientY = clientY;
        trackedOnScrolled = onScrolled;
        if (!pointerInBand(clientX, lastClientY)) {
            // ★ 离开带 ⇒ 复位时钟并取消挂起帧（与 tick 同一规则，两处都必须做）。
            // 不取消的话，指针移出边缘后仍会多跑一帧才停 —— 那一帧的 frameMs 是
            // 真实帧时长，于是"离开边缘的瞬间"会补一段本不该有的位移。
            lastFrameAtMs = null;
            if (rafId !== null) {
                cancelFrame(rafId);
                rafId = null;
            }
            return;
        }
        if (rafId === null) rafId = requestFrame(tick);
    };

    const stop = (): void => {
        if (rafId !== null) {
            cancelFrame(rafId);
            rafId = null;
        }
        lastClientX = null;
        lastClientY = null;
        lastFrameAtMs = null;
        lastFrameMs = 0;
        trackedOnScrolled = undefined;
    };

    return {
        step,
        track,
        stop,
        isRunning: () => rafId !== null,
    };
}
