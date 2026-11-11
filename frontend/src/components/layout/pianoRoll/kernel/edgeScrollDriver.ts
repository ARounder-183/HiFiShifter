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
 * 【为什么时钟之外还需要 `ticking`：帧令牌的两个职责】
 * `rafId` 同时表达两件事：「已经排了一帧」与「这一帧正在执行」。旧实现在
 * `tick()` **入口**就把它置空，只完成了前者；于是任何在 `apply()` 内部发生的
 * 重入 `track()` 都会把「正在执行」误读成「空闲」，再排一帧，而 `tick()` 末尾
 * 又排一帧 —— 每帧**净增一个无人取消的孤儿回调**，且没有上界。
 *
 * 生产代码里这条重入是**必然发生**的：值轴的 `apply` → 提交值域视口 → 发布
 * 交互视口 → 订阅者合成 `pointermove` → 手势 `onMove` → 同一个驱动的 `track()`。
 * （Task 1 同时把那条广播改成微任务合帧，见 `pianoRollInteractionViewportBus`；
 * 但驱动本身必须自己站得住 —— 任何轴、任何调用方都可能重入。）
 *
 * 症状：拖得越久越卡。第 N 帧要跑 N 次 `tick()`，每次至少两回 `getBounds()`
 * （= `getBoundingClientRect()`，强制同步布局）⇒ O(N²)。
 *
 * 现在的规则（**不变式**，改代码前先读这三条）：
 * - **I-1** `rafId !== null` ⟺ **恰好一个**待执行帧；执行期间（`ticking`）不接受
 *   新的排帧请求，唯一的排帧入口是 `schedule()`。
 * - **I-2** 一帧只取一次 `getBounds()`，两轴与带内判定共用同一份矩形。
 * - **I-3** `stop()` 递增取消代次；`tick()` 收尾时若代次已变就不再续帧 ——
 *   这样「在 `apply()` 内部被 stop()」也不会留下一帧。
 *
 * 【违反边界的症状】
 * - 时钟按 step 计：来回蹭边缘时视图一跳一跳；
 * - 帧时长每轴各算一次：只有一个轴生效；
 * - 到达边界后仍排帧：松手前 CPU 空转、每帧标脏重绘；
 * - `tick()` 入口提前清 `rafId` 却不设 `ticking`：拖得越久越卡（见上）。
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
    /** 是否有待执行帧或正处在某一帧的执行中。 */
    isRunning: () => boolean;
}

export function createEdgeScrollDriver(options: EdgeScrollDriverOptions): EdgeScrollDriver {
    const now = options.now ?? (() => performance.now());
    const requestFrame = options.requestFrame ?? ((cb: () => void) => requestAnimationFrame(cb));
    const cancelFrame = options.cancelFrame ?? ((id: number) => cancelAnimationFrame(id));

    /** 唯一的帧令牌：`!== null` ⟺ 恰好一个待执行帧（I-1）。 */
    let rafId: number | null = null;
    /** 本帧回调正在执行。执行期间 `track()` 不得排帧（见模块头 I-1）。 */
    let ticking = false;
    /**
     * 取消代次。`stop()` 递增它；`tick()` 在入口记下自己的代次，收尾时若代次
     * 已变就不再续帧（I-3）。
     */
    let generation = 0;
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
        boundsIn?: EdgeScrollBounds | null,
    ): boolean => {
        lastClientX = clientX;
        if (clientY !== undefined) lastClientY = clientY;
        // I-2：`tick()` 传进来的矩形与带内判定共用；独立调用时才自己取。
        const bounds = boundsIn === undefined ? options.getBounds() : boundsIn;
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
    const pointerInBand = (
        clientX: number,
        clientY: number | null,
        boundsIn?: EdgeScrollBounds | null,
    ): boolean => {
        const bounds = boundsIn === undefined ? options.getBounds() : boundsIn;
        if (!bounds) return false;
        return options.axes.some(
            (axis) => axis.canScroll() && axisRatio(axis, clientX, clientY, bounds) !== 0,
        );
    };

    /**
     * 唯一的排帧入口（I-1）。
     *
     * 两个提前返回都是**不变式**，不是优化：`rafId !== null` 说明已经有一帧在
     * 队列里；`ticking` 说明我们正站在那一帧里面，它的收尾会自己续帧。
     */
    const schedule = (): void => {
        if (rafId !== null || ticking) return;
        rafId = requestFrame(tick);
    };

    function tick(): void {
        rafId = null; // 本帧已被消费
        ticking = true; // 但仍在执行：任何重入的 track() 都不得排帧
        const myGeneration = generation;
        let keep = false;
        try {
            const clientX = lastClientX;
            if (clientX !== null) {
                // I-2：一帧只取一次边界。getBoundingClientRect() 是强制同步布局，
                // 旧实现每帧取两次，在重入泄漏下被放大成 O(N²)。
                const bounds = options.getBounds();
                const moved = step(clientX, lastClientY ?? undefined, trackedOnScrolled, bounds);
                /*
                 * 用**最新**指针位置判定带内：`apply()` 期间可能有重入的 track()
                 * 把指针挪走（手势合成的 pointermove 就是这条路），此时本帧就该收尾。
                 */
                const latestX = lastClientX;
                if (latestX === null || !pointerInBand(latestX, lastClientY, bounds)) {
                    // ★ 离开带 ⇒ 复位时钟。否则下一次进带的第一帧会拿到整段带外空闲。
                    lastFrameAtMs = null;
                } else if (moved || lastFrameMs === 0) {
                    // 复位帧（frameMs = 0）不会有位移，但必须继续排帧，否则循环在第一帧就停了。
                    keep = true;
                }
            }
        } finally {
            ticking = false;
            if (keep && myGeneration === generation) schedule();
        }
    }

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
        schedule();
    };

    const stop = (): void => {
        // I-3：让正在执行的那一帧收尾时知道「我已经被取消了」。
        generation += 1;
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
        isRunning: () => rafId !== null || ticking,
    };
}
