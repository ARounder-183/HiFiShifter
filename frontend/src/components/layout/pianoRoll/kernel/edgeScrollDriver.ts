/**
 * 参数编辑器内核 · 拖拽边缘自动滚屏驱动（二维）。
 *
 * 指针停在视口左/右/上/下边缘时持续滚动，并在每次真实位移后回调调用方，
 * 让它按滚动后的投影重算笔画 / 选区 / 块预览。几何公式与时间轴共用
 * `shared/edgeAutoScroll` 的轴无关实现；本模块只负责时钟、rAF 与钳制。
 */
import {
    resolveAxisEdgeScrollDeltaPx,
    type EdgeScrollAxisPolicy,
    type EdgeScrollBounds,
} from "../../shared/edgeAutoScroll";
import { EDGE_SCROLL_MAX_SPEED_PX_PER_SEC } from "./dragArithmetic";

const MIN_STEP_PX = 0.01;
const GESTURE_GAP_RESET_MS = 200;
const DEFAULT_FRAME_MS = 1000 / 60;

export type { EdgeScrollBounds };

export interface EdgeScrollDriverOptions {
    getBounds: () => EdgeScrollBounds | null;
    getScrollLeft: () => number;
    setScrollLeft: (next: number) => void;
    getMaxScrollLeft: () => number;
    getMinScrollLeft: () => number;
    /** 竖直轴可选；缺省 = 只做水平滚动（旧调用）。 */
    getScrollTop?: () => number;
    setScrollTop?: (next: number) => void;
    getMaxScrollTop?: () => number;
    getMinScrollTop?: () => number;
    /** 本帧确实滚动了。带指针位置，供调用方按同一位置重算。 */
    onScrolled?: (clientX: number, clientY?: number) => void;
    maxSpeedPxPerSec?: number;
    maxSpeedYPxPerSec?: number;
    /** 水平轴手感策略；缺省 = 时间轴既有 32px / 1.5 / 线性。 */
    horizontalPolicy?: EdgeScrollAxisPolicy;
    /** 竖直轴（值轴）手感策略；参数编辑器传 18px / 1.0 / 1.6。 */
    verticalPolicy?: EdgeScrollAxisPolicy;
    /**
     * 动态竖向速度（px/秒）。参数编辑器的值域视口会缩放，速度必须按当前
     * span 现算；固定值只保留给不需要动态速度的调用方。
     */
    getMaxSpeedYPxPerSec?: () => number;
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
    const maxSpeedX = options.maxSpeedPxPerSec ?? EDGE_SCROLL_MAX_SPEED_PX_PER_SEC;
    const fixedMaxSpeedY = options.maxSpeedYPxPerSec ?? maxSpeedX;
    const resolveMaxSpeedY = (): number => {
        const value = options.getMaxSpeedYPxPerSec?.() ?? fixedMaxSpeedY;
        return Number.isFinite(value) && value > 0 ? value : 0;
    };

    let rafId: number | null = null;
    let lastClientX: number | null = null;
    let lastClientY: number | null = null;
    let lastStepAtMs: number | null = null;
    let trackedOnScrolled: ((clientX: number, clientY?: number) => void) | undefined;

    const frameMsFor = (atMs: number): number => {
        if (lastStepAtMs === null) return DEFAULT_FRAME_MS;
        const elapsed = atMs - lastStepAtMs;
        if (!Number.isFinite(elapsed) || elapsed <= 0 || elapsed > GESTURE_GAP_RESET_MS) {
            return DEFAULT_FRAME_MS;
        }
        return elapsed;
    };

    const hasVertical = (): boolean =>
        options.getScrollTop !== undefined &&
        options.setScrollTop !== undefined &&
        options.getMaxScrollTop !== undefined &&
        options.getMinScrollTop !== undefined;

    const horizontalCanScroll = (): boolean =>
        options.getMaxScrollLeft() > options.getMinScrollLeft();

    const verticalCanScroll = (): boolean =>
        hasVertical() && options.getMaxScrollTop!() > options.getMinScrollTop!();

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
        const atMs = now();
        const frameMs = frameMsFor(atMs);
        const deltaX = horizontalCanScroll()
            ? resolveAxisEdgeScrollDeltaPx({
                  pointerPx: clientX,
                  minPx: bounds.left,
                  maxPx: bounds.right,
                  frameMs,
                  maxSpeedPxPerSec: maxSpeedX,
                  ...options.horizontalPolicy,
              })
            : 0;
        const maxSpeedYNow = resolveMaxSpeedY();
        const deltaY =
            verticalCanScroll() &&
            maxSpeedYNow > 0 &&
            lastClientY !== null &&
            bounds.top !== undefined &&
            bounds.bottom !== undefined
                ? resolveAxisEdgeScrollDeltaPx({
                      pointerPx: lastClientY,
                      minPx: bounds.top,
                      maxPx: bounds.bottom,
                      frameMs,
                      maxSpeedPxPerSec: maxSpeedYNow,
                      ...options.verticalPolicy,
                  })
                : 0;
        lastStepAtMs = atMs;
        let moved = false;
        if (Math.abs(deltaX) >= MIN_STEP_PX) {
            const current = options.getScrollLeft();
            const max = options.getMaxScrollLeft();
            const min = options.getMinScrollLeft();
            if (max > min) {
                const next = Math.min(max, Math.max(min, current + deltaX));
                if (Math.abs(next - current) >= MIN_STEP_PX) {
                    options.setScrollLeft(next);
                    moved = true;
                }
            }
        }
        if (Math.abs(deltaY) >= MIN_STEP_PX && hasVertical()) {
            const current = options.getScrollTop!();
            const max = options.getMaxScrollTop!();
            const min = options.getMinScrollTop!();
            if (max > min) {
                const next = Math.min(max, Math.max(min, current + deltaY));
                if (Math.abs(next - current) >= MIN_STEP_PX) {
                    options.setScrollTop!(next);
                    moved = true;
                }
            }
        }
        if (moved) emitScrolled(clientX, lastClientY, onScrolled);
        return moved;
    };

    const pointerInBand = (clientX: number, clientY: number | null): boolean => {
        const bounds = options.getBounds();
        if (!bounds) return false;
        const canScrollX = horizontalCanScroll();
        const canScrollY = verticalCanScroll();
        if (!canScrollX && !canScrollY) return false;
        const horizontal =
            canScrollX &&
            resolveAxisEdgeScrollDeltaPx({
                pointerPx: clientX,
                minPx: bounds.left,
                maxPx: bounds.right,
                frameMs: DEFAULT_FRAME_MS,
                maxSpeedPxPerSec: maxSpeedX,
                ...options.horizontalPolicy,
            }) !== 0;
        const maxSpeedYNow = resolveMaxSpeedY();
        const vertical =
            canScrollY &&
            maxSpeedYNow > 0 &&
            clientY !== null &&
            bounds.top !== undefined &&
            bounds.bottom !== undefined &&
            resolveAxisEdgeScrollDeltaPx({
                pointerPx: clientY,
                minPx: bounds.top,
                maxPx: bounds.bottom,
                frameMs: DEFAULT_FRAME_MS,
                maxSpeedPxPerSec: maxSpeedYNow,
                ...options.verticalPolicy,
            }) !== 0;
        return horizontal || vertical;
    };

    const tick = () => {
        rafId = null;
        const clientX = lastClientX;
        if (clientX === null) return;
        const moved = step(clientX, lastClientY ?? undefined, trackedOnScrolled);
        // 指针离开边缘带，或仍在带内但两个轴都已到边界 → 停止排帧。
        if (moved && pointerInBand(clientX, lastClientY)) {
            rafId = requestFrame(tick);
        }
    };

    const track = (
        clientX: number,
        clientY?: number,
        onScrolled?: (clientX: number, clientY?: number) => void,
    ): void => {
        lastClientX = clientX;
        if (clientY !== undefined) lastClientY = clientY;
        trackedOnScrolled = onScrolled;
        if (rafId === null && pointerInBand(clientX, lastClientY)) {
            rafId = requestFrame(tick);
        }
    };

    const stop = (): void => {
        if (rafId !== null) {
            cancelFrame(rafId);
            rafId = null;
        }
        lastClientX = null;
        lastClientY = null;
        lastStepAtMs = null;
        trackedOnScrolled = undefined;
    };

    return {
        step,
        track,
        stop,
        isRunning: () => rafId !== null,
    };
}
