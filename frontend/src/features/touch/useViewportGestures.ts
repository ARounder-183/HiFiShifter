// 视口手势：把 `gestures/touchGesture.ts` 的识别器接到画布视口上。
//
// 【为什么复用识别器而不重写】那份 287 行的实现已经把三件容易写错的事做对了：
// 绝对缩放律（不会在两指平行时爆炸）、2→1 手指冷却（抬一根手指不会让画面跳）、
// 指数惯性。它此前是**死代码**（只被自己的单测引用）—— 本模块就是它的第一个
// 真实消费者。
//
// 【为什么捏合走「合成 ctrl+wheel」而不是直接改缩放】
// 两块画布都已有成熟的滚轮缩放通道（锚点换算、上下限钳制、与另一面板的同步、
// 同帧提交）。捏合与 ctrl+滚轮在语义上本来就是同一件事（浏览器自己也这么映射：
// 触控板捏合 = ctrl+wheel），`pinchDeltaFromWheel` / `accumulatePinchSteps`
// 已在仓库里按这个等价关系写好。合成一个事件交给既有通道，等于免费得到
// 「锚点不漂移」与「两个面板同步」这两条已经调好的行为；自己再写一遍缩放数学，
// 就是给同一个语义开第二个真值源。
//
// 【为什么在 capture 阶段拦】内核宿主与画布的编辑手势都挂在冒泡阶段。平移一旦
// 成立，就必须让编辑手势**完全看不到**这次按下 —— 否则会出现「边滚边拖 Clip」。
// capture 阶段 stopPropagation 是唯一能保证这件事的时机。
//
// 【为什么桌面必须逐字节不变】监听器在桌面上也装着（触摸屏笔记本会发
// pointerType === "touch"）。`enabled` 为假时本模块不注册任何监听器，
// 桌面路径因此完全没有变化。

import React from "react";

import {
    createTouchGestureRecognizer,
    inertiaDisplacement,
    inertiaFinished,
    type GestureAxis,
} from "../../components/layout/gestures/touchGesture";

/** 视口手势的落地端（每个画布各一份实现）。 */
export interface ViewportGestureAdapter {
    /** 平移增量（像素，正 = 内容向右/向下移动，即视口向左/向上滚）。 */
    panBy(dx: number, dy: number): void;
    /**
     * 捏合缩放。`factor > 1` = 放大。
     *
     * 实现约定：把这次倍率交给画布**既有的**滚轮缩放通道（合成 ctrl+wheel），
     * 锚点取 `(clientX, clientY)`。不要在适配器里重写缩放数学。
     */
    zoomBy(factor: number, clientX: number, clientY: number): void;
    /** 平移开始 / 结束（用于淡入滚动指示、暂停边缘自动滚屏）。 */
    onPanActiveChange?(active: boolean): void;
    /** 第二根手指落下：取消第一根手指正在进行的编辑手势。 */
    cancelActiveEdit?(): void;
}

export interface UseViewportGesturesArgs {
    /** 挂载点（画布或画布容器）。 */
    elementRef: React.MutableRefObject<HTMLElement | null>;
    /** 能力位 input.two_finger_gesture。为假时**不注册任何监听器**。 */
    enabled: boolean;
    adapter: ViewportGestureAdapter;
    /**
     * 这一下按下是否应当进入平移。
     *
     * 缺省返回 false（单指永远归编辑）。时间轴传「命中不是 Clip」，
     * 参数编辑器传「选择工具 + 空白」（由画布自己的命中测试决定）。
     */
    shouldPanOnPress?: (event: PointerEvent) => boolean;
    /** 允许的轴。默认两向。 */
    axis?: GestureAxis;
    /** 惯性开关（prefers-reduced-motion 下由调用方传 false）。 */
    inertia?: boolean;
}

function pointOf(event: PointerEvent): { x: number; y: number } {
    return { x: event.clientX, y: event.clientY };
}

/**
 * 挂载触摸视口手势。
 *
 * 生命周期与 `elementRef.current` 绑定：元素换掉时旧监听器全部摘除，
 * 因此面板重挂载不会留下幽灵手势。
 */
export function useViewportGestures({
    elementRef,
    enabled,
    adapter,
    shouldPanOnPress,
    axis = "both",
    inertia = true,
}: UseViewportGesturesArgs): void {
    // 适配器与判定函数放进 ref：监听器只在挂载时注册一次，但必须读到**最新**的
    // 回调（它们随渲染重建）。用 ref 而不是把它们塞进依赖数组，是为了避免
    // 「每次渲染都拆装一遍 window 监听器」这种热路径抖动。
    const adapterRef = React.useRef(adapter);
    adapterRef.current = adapter;
    const shouldPanRef = React.useRef(shouldPanOnPress);
    shouldPanRef.current = shouldPanOnPress;

    React.useEffect(() => {
        const element = elementRef.current;
        if (!enabled || element === null) return;

        /** 当前按下的触摸指针（只记触摸；鼠标/笔不参与）。 */
        const touches = new Map<number, { x: number; y: number }>();
        /** 识别器是否已被激活（激活后 move/up 都要转给它）。 */
        let engaged = false;
        let inertiaRaf: number | null = null;

        const stopInertia = () => {
            if (inertiaRaf !== null) {
                cancelAnimationFrame(inertiaRaf);
                inertiaRaf = null;
            }
        };

        const startInertia = (vx: number, vy: number) => {
            if (!inertia) return;
            if (Math.abs(vx) < 0.02 && Math.abs(vy) < 0.02) return;
            const startedAt = performance.now();
            const tick = () => {
                const elapsed = performance.now() - startedAt;
                if (inertiaFinished(vx, elapsed) && inertiaFinished(vy, elapsed)) {
                    inertiaRaf = null;
                    adapterRef.current.onPanActiveChange?.(false);
                    return;
                }
                // 逐帧位移 = 当前累计位移 − 上一帧累计位移
                // （inertiaDisplacement 返回的是从松手到此刻的累计量）。
                const prevElapsed = Math.max(0, elapsed - 16);
                const dx = inertiaDisplacement(vx, elapsed) - inertiaDisplacement(vx, prevElapsed);
                const dy = inertiaDisplacement(vy, elapsed) - inertiaDisplacement(vy, prevElapsed);
                if (dx !== 0 || dy !== 0) adapterRef.current.panBy(dx, dy);
                inertiaRaf = requestAnimationFrame(tick);
            };
            inertiaRaf = requestAnimationFrame(tick);
        };

        const recognizer = createTouchGestureRecognizer(
            axis,
            {
                onPanStart() {
                    adapterRef.current.onPanActiveChange?.(true);
                },
                onPan(dx, dy) {
                    adapterRef.current.panBy(dx, dy);
                },
                onPanEnd(vx, vy) {
                    startInertia(vx, vy);
                    if (inertiaRaf === null) adapterRef.current.onPanActiveChange?.(false);
                },
                onPinch(factor, anchorX, anchorY) {
                    // 捏合中心在视口坐标系里；合成滚轮事件要用 client 坐标。
                    const rect = element.getBoundingClientRect();
                    adapterRef.current.zoomBy(factor, rect.left + anchorX, rect.top + anchorY);
                },
            },
            () => performance.now(),
        );

        const releaseAll = () => {
            touches.clear();
            engaged = false;
            stopInertia();
            recognizer.cancel();
            adapterRef.current.onPanActiveChange?.(false);
        };

        const onPointerDown = (event: PointerEvent) => {
            if (event.pointerType !== "touch") return;
            touches.set(event.pointerId, pointOf(event));

            if (touches.size === 1) {
                const wantsPan = shouldPanRef.current?.(event) ?? false;
                if (!wantsPan) {
                    // 单指归编辑：不启动识别器，事件原样交给画布。
                    return;
                }
                engaged = true;
                event.preventDefault();
                event.stopPropagation();
                recognizer.start(event.pointerId, pointOf(event));
                return;
            }

            if (touches.size === 2) {
                // 第二根手指落下：先取消第一根手指正在进行的编辑，再进捏合。
                adapterRef.current.cancelActiveEdit?.();
                stopInertia();
                if (!engaged) {
                    // 第一根手指此前归编辑（识别器没启动）。这里补种两个点，
                    // 让它直接进入 pinch —— 两次 start 同步调用，中间不会产生
                    // 任何 move，因此不会先冒出一帧平移。
                    engaged = true;
                    const first = [...touches.entries()].find(([id]) => id !== event.pointerId);
                    if (first) recognizer.start(first[0], first[1]);
                }
                event.preventDefault();
                event.stopPropagation();
                recognizer.start(event.pointerId, pointOf(event));
            }
        };

        const onPointerMove = (event: PointerEvent) => {
            if (!touches.has(event.pointerId)) return;
            touches.set(event.pointerId, pointOf(event));
            if (!engaged) return;
            event.preventDefault();
            event.stopPropagation();
            recognizer.move(event.pointerId, pointOf(event));
        };

        const onPointerUp = (event: PointerEvent) => {
            if (!touches.delete(event.pointerId)) return;
            if (!engaged) return;
            event.preventDefault();
            event.stopPropagation();
            recognizer.end(event.pointerId);
            if (touches.size === 0) engaged = false;
        };

        const onPointerCancel = (event: PointerEvent) => {
            if (!touches.delete(event.pointerId)) return;
            if (!engaged) return;
            recognizer.cancel();
            engaged = false;
        };

        element.addEventListener("pointerdown", onPointerDown, { capture: true });
        element.addEventListener("pointermove", onPointerMove, { capture: true });
        element.addEventListener("pointerup", onPointerUp, { capture: true });
        element.addEventListener("pointercancel", onPointerCancel, { capture: true });
        window.addEventListener("pointerup", onPointerUp, { capture: true });
        window.addEventListener("pointercancel", onPointerCancel, { capture: true });
        // 失焦 / 页面隐藏时指针事件可能永远不来：必须收尾，否则识别器会一直
        // 停在一个「按着」的状态，下一次单指按下会被当成第二根手指。
        window.addEventListener("blur", releaseAll);

        return () => {
            element.removeEventListener("pointerdown", onPointerDown, { capture: true });
            element.removeEventListener("pointermove", onPointerMove, { capture: true });
            element.removeEventListener("pointerup", onPointerUp, { capture: true });
            element.removeEventListener("pointercancel", onPointerCancel, { capture: true });
            window.removeEventListener("pointerup", onPointerUp, { capture: true });
            window.removeEventListener("pointercancel", onPointerCancel, { capture: true });
            window.removeEventListener("blur", releaseAll);
            releaseAll();
        };
    }, [elementRef, enabled, axis, inertia]);
}

/** 合成一次「捏合」滚轮事件（各画布适配器共用）。 */
export function dispatchPinchWheel(
    target: EventTarget,
    clientX: number,
    clientY: number,
    factor: number,
): void {
    if (!Number.isFinite(factor) || factor <= 0 || factor === 1) return;
    // 倍率 → 滚轮增量的换算：缩放通道用的是「指数步长」
    // （resolveHorizontalWheelZoom 把 deltaY 折算成倍率），因此取自然对数即可
    // 得到等价增量。
    const deltaY = -Math.log(factor) * 300;
    const event = new WheelEvent("wheel", {
        bubbles: true,
        cancelable: true,
        clientX,
        clientY,
        deltaX: 0,
        deltaY,
        deltaMode: 0,
        // 捏合 ≡ ctrl+滚轮（浏览器自己也这么映射）。
        ctrlKey: true,
    });
    target.dispatchEvent(event);
}
