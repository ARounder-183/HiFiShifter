/**
 * 菜单提示行的自动滚动引擎（悬停慢滚 / 点击翻页 / 长按快滚）。
 *
 * 【为什么三种手势共用一个 rAF 循环】三种输入产生的是**同一种结果**：把
 * `scrollTop` 沿某个方向推一段距离。分成三套定时器就会出现三个"什么时候停"的
 * 答案 —— 而"停"恰恰是最容易错的地方（到边界、抬起手指、指针离开、滚轮介入、
 * 卸载、窗口失焦）。这里只有一个 `Motion` 描述 + 一个循环，停止条件写一遍。
 *
 * 【为什么"像素累积"而不是"每帧取整"】220 px/s 在 60fps 下是每帧 3.67px。直接
 * 取整会让实际速度变成 3px/帧（180 px/s，慢 18%），而慢速滚动时（30 px/s）
 * 会**一帧都不动**。累积余数之后，速度与帧率解耦。
 *
 * 【为什么"按下期间"要单独记】提示行滚到尽头会隐藏（`data-visible` 被清掉 →
 * `pointer-events: none`）。若手指还按着，元素变成不可命中会让 `pointerup` 落到
 * 别处 —— 于是"按住快滚到底"会留下一个永远收不到抬起的按下态。因此按下期间
 * 冻结可见性，抬起后再补写一次。
 */
import { useCallback, useEffect, useRef } from "react";
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from "react";

import {
    MENU_SCROLL_TUNING,
    clampMenuScrollTop,
    computeMenuPageStep,
    easeOutCubic,
    menuScrollPointerKind,
    type MenuScrollPointerKind,
} from "./menuScrollModel";

/** 一段进行中的运动。 */
type MenuScrollMotion =
    /** 速度驱动：悬停（带爬升）与长按（匀速，`rampMs = 0`）。 */
    | { kind: "velocity"; dir: 1 | -1; speed: number; rampMs: number; startedAt: number }
    /** 位置驱动：点击翻一页。 */
    | { kind: "tween"; from: number; to: number; durationMs: number; startedAt: number };

/** 单个方向提示行上的事件处理器。 */
export interface MenuAutoScrollHintHandlers {
    onPointerEnter: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerLeave: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerUp: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerCancel: (event: ReactPointerEvent<HTMLElement>) => void;
    onClick: (event: ReactMouseEvent<HTMLElement>) => void;
}

export interface MenuAutoScrollOptions {
    /** 滚动容器（菜单壳）。返回 null 表示这张菜单不可滚。 */
    getViewport: () => HTMLElement | null;
    /** 一个菜单项的高度（用于"翻一页"的步长）。 */
    getItemHeight: () => number;
    /**
     * 按下态。**由调用方持有**：可见性同步要读它（按下期间冻结），而可见性同步
     * 又必须能在引擎的回调里跑 —— 共用一个 ref 是唯一不会互相等待的形状。
     */
    pressingRef: { current: boolean };
    /** false = 不接管任何手势（`--no-scroll` 壳）。 */
    enabled: boolean;
    /** 滚动位置或运动状态变了，调用方据此重算提示行可见性。 */
    onStateChange: () => void;
}

export interface MenuAutoScroll {
    /** 绑定一个方向（`-1` = 向上提示行，`1` = 向下提示行）。 */
    bind: (dir: 1 | -1) => MenuAutoScrollHintHandlers;
    /** 外部滚动介入（滚轮 / 键盘）时调用：立刻停下自动滚动。 */
    interrupt: () => void;
}

/** 单帧允许的最大时间步长：后台标签页回来时 `dt` 会是几百毫秒，不夹就会瞬移。 */
const MAX_FRAME_DT_SECONDS = 0.064;

export function useMenuAutoScroll(options: MenuAutoScrollOptions): MenuAutoScroll {
    const optionsRef = useRef(options);
    optionsRef.current = options;

    const motionRef = useRef<MenuScrollMotion | null>(null);
    const rafRef = useRef<number | null>(null);
    const lastFrameRef = useRef(0);
    const carryRef = useRef(0);
    const hoverTimerRef = useRef<number | null>(null);
    const pressTimerRef = useRef<number | null>(null);
    const hoveringRef = useRef(false);
    /** 按下时那个提示行元素：`data-pressed` 写在它身上，抬起时按这个引用清掉。 */
    const pressedElementRef = useRef<HTMLElement | null>(null);

    const clearHoverTimer = useCallback(() => {
        if (hoverTimerRef.current !== null) {
            window.clearTimeout(hoverTimerRef.current);
            hoverTimerRef.current = null;
        }
    }, []);

    const clearPressTimer = useCallback(() => {
        if (pressTimerRef.current !== null) {
            window.clearTimeout(pressTimerRef.current);
            pressTimerRef.current = null;
        }
    }, []);

    const stop = useCallback(() => {
        motionRef.current = null;
        carryRef.current = 0;
        lastFrameRef.current = 0;
        if (rafRef.current !== null) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
        }
    }, []);

    const frameRef = useRef<(now: number) => void>(() => {});

    const schedule = useCallback(() => {
        if (rafRef.current !== null) return;
        rafRef.current = requestAnimationFrame((now) => frameRef.current(now));
    }, []);

    const frame = useCallback(
        (now: number) => {
            rafRef.current = null;
            const motion = motionRef.current;
            const viewport = optionsRef.current.getViewport();
            if (!motion || !viewport) {
                stop();
                return;
            }
            // 首帧不算时间步：`requestAnimationFrame` 的时间戳与启动时刻之间可能
            // 隔了十几毫秒（合成器排队），算进去就是"点一下先跳一小段"。
            if (lastFrameRef.current === 0) lastFrameRef.current = now;
            const dt = Math.min(now - lastFrameRef.current, MAX_FRAME_DT_SECONDS * 1000) / 1000;
            lastFrameRef.current = now;
            const maxScroll = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
            let next: number;
            if (motion.kind === "tween") {
                const t =
                    motion.durationMs <= 0
                        ? 1
                        : Math.min(1, (now - motion.startedAt) / motion.durationMs);
                next = motion.from + (motion.to - motion.from) * easeOutCubic(t);
                if (t >= 1) stop();
            } else {
                const ramp =
                    motion.rampMs > 0 ? Math.min(1, (now - motion.startedAt) / motion.rampMs) : 1;
                // 从 40% 爬到 100%：起步就有可见的位移（否则悬停 200ms 内像没反应），
                // 但爬升结束后才是"巡航速度"。
                const speed = motion.speed * (0.4 + 0.6 * ramp * ramp);
                carryRef.current += speed * dt;
                const step = Math.trunc(carryRef.current);
                carryRef.current -= step;
                if (step === 0) {
                    schedule();
                    return;
                }
                next = viewport.scrollTop + motion.dir * step;
            }
            const before = viewport.scrollTop;
            viewport.scrollTop = clampMenuScrollTop(next, maxScroll);
            // 到底了：位置没变就是唯一可靠的判据（不同浏览器对"超界赋值"的处理
            // 不完全一致，不能靠比较目标值）。
            if (viewport.scrollTop === before) {
                stop();
                optionsRef.current.onStateChange();
                return;
            }
            optionsRef.current.onStateChange();
            if (motionRef.current !== null) schedule();
        },
        [schedule, stop],
    );

    useEffect(() => {
        frameRef.current = frame;
    }, [frame]);

    const startMotion = useCallback(
        (motion: MenuScrollMotion) => {
            motionRef.current = motion;
            carryRef.current = 0;
            lastFrameRef.current = 0;
            schedule();
        },
        [schedule],
    );

    const startVelocity = useCallback(
        (dir: 1 | -1, speed: number, rampMs: number) => {
            startMotion({
                kind: "velocity",
                dir,
                speed,
                rampMs,
                startedAt: performance.now(),
            });
        },
        [startMotion],
    );

    const pressEnd = useCallback(() => {
        if (!optionsRef.current.pressingRef.current) return;
        optionsRef.current.pressingRef.current = false;
        clearPressTimer();
        const pressed = pressedElementRef.current;
        pressedElementRef.current = null;
        if (pressed) delete pressed.dataset.pressed;
        // 【只停 velocity】点击那一次补间要让它在抬起之后**走完** —— 否则 100ms
        // 的一次点击只滚了七成就停在半路，读起来像"卡住了"。
        if (motionRef.current?.kind === "velocity") stop();
        // 按下期间可见性被冻结，这里补写一次。
        optionsRef.current.onStateChange();
    }, [clearPressTimer, stop]);

    const hoverStart = useCallback(
        (dir: 1 | -1, kind: MenuScrollPointerKind) => {
            clearHoverTimer();
            const tuning = MENU_SCROLL_TUNING[kind];
            if (tuning.hoverDelayMs <= 0) {
                startVelocity(dir, tuning.hoverSpeed, tuning.hoverRampMs);
                return;
            }
            hoverTimerRef.current = window.setTimeout(() => {
                hoverTimerRef.current = null;
                if (!hoveringRef.current) return;
                startVelocity(dir, tuning.hoverSpeed, tuning.hoverRampMs);
            }, tuning.hoverDelayMs);
        },
        [clearHoverTimer, startVelocity],
    );

    const pressStart = useCallback(
        (dir: 1 | -1, kind: MenuScrollPointerKind, element: HTMLElement) => {
            const pressing = optionsRef.current.pressingRef;
            pressing.current = true;
            pressedElementRef.current = element;
            element.dataset.pressed = "1";
            // 悬停让位给按下：否则先悬停再按下的那一次会叠加两个运动。
            clearHoverTimer();
            hoveringRef.current = false;
            const tuning = MENU_SCROLL_TUNING[kind];
            const viewport = optionsRef.current.getViewport();
            if (viewport) {
                const step = computeMenuPageStep({
                    clientHeight: viewport.clientHeight,
                    itemHeight: optionsRef.current.getItemHeight(),
                });
                const from = viewport.scrollTop;
                const maxScroll = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
                startMotion({
                    kind: "tween",
                    from,
                    to: clampMenuScrollTop(from + dir * step, maxScroll),
                    durationMs: tuning.pageTweenMs,
                    startedAt: performance.now(),
                });
            }
            clearPressTimer();
            pressTimerRef.current = window.setTimeout(() => {
                pressTimerRef.current = null;
                if (!pressing.current) return;
                // 按住 → 匀速快滚。`rampMs: 0`：已经按住 420ms 了，再爬升就是拖延。
                startVelocity(dir, tuning.pressSpeed, 0);
            }, tuning.pressDelayMs);
        },
        [clearHoverTimer, clearPressTimer, startMotion, startVelocity],
    );

    const interrupt = useCallback(() => {
        clearHoverTimer();
        hoveringRef.current = false;
        stop();
    }, [clearHoverTimer, stop]);

    /*
     * 抬起 / 取消的安全网。
     *
     * 【为什么挂在 document 上】指针在按下期间可能离开提示行（滚到底时提示行会
     * 隐藏）。只监听元素自身的 `pointerup` 会让"按住快滚"永远停在按下态 ——
     * 而那会让可见性**永久冻结**（提示行再也不更新）。
     */
    useEffect(() => {
        if (!options.enabled) return;
        const onRelease = () => pressEnd();
        document.addEventListener("pointerup", onRelease, true);
        document.addEventListener("pointercancel", onRelease, true);
        return () => {
            document.removeEventListener("pointerup", onRelease, true);
            document.removeEventListener("pointercancel", onRelease, true);
        };
    }, [options.enabled, pressEnd]);

    /* 窗口失焦 / 标签页切走：不再有指针事件，必须主动停。 */
    useEffect(() => {
        if (!options.enabled) return;
        const onBlur = () => {
            pressEnd();
            interrupt();
        };
        window.addEventListener("blur", onBlur);
        document.addEventListener("visibilitychange", onBlur);
        return () => {
            window.removeEventListener("blur", onBlur);
            document.removeEventListener("visibilitychange", onBlur);
        };
    }, [interrupt, options.enabled, pressEnd]);

    /* 卸载：取消 rAF 与两个定时器。漏掉任何一条都会在菜单关闭后继续写已卸载的 DOM。 */
    useEffect(
        () => () => {
            clearHoverTimer();
            clearPressTimer();
            stop();
        },
        [clearHoverTimer, clearPressTimer, stop],
    );

    const bind = useCallback(
        (dir: 1 | -1): MenuAutoScrollHintHandlers => ({
            onPointerEnter: (event) => {
                if (!optionsRef.current.enabled) return;
                // 触摸没有悬停：`pointerenter` 在手指按下时也会来一次，若在这里
                // 启动慢滚，手机上一按就开始滚 —— 而用户只想点一下翻一页。
                if (event.pointerType === "touch") return;
                hoveringRef.current = true;
                hoverStart(dir, menuScrollPointerKind(event.pointerType));
            },
            onPointerLeave: () => {
                hoveringRef.current = false;
                clearHoverTimer();
                // 按下期间不打断：手指还按着，移开一点不该停。
                if (!optionsRef.current.pressingRef.current) stop();
            },
            onPointerDown: (event) => {
                if (!optionsRef.current.enabled) return;
                if (event.pointerType === "mouse" && event.button !== 0) return;
                // 不抢焦点：记事本菜单的 `autoFocus` 与 header 里的输入框都靠焦点
                // 活着，点一下提示行就把焦点搬走会让它们失焦提交。
                event.preventDefault();
                event.stopPropagation();
                pressStart(dir, menuScrollPointerKind(event.pointerType), event.currentTarget);
            },
            onPointerUp: () => pressEnd(),
            onPointerCancel: () => pressEnd(),
            onClick: (event) => {
                // 提示行不是命令：吞掉 click，别让它冒到菜单外层的"点外面关闭"。
                event.preventDefault();
                event.stopPropagation();
            },
        }),
        [clearHoverTimer, hoverStart, pressEnd, pressStart, stop],
    );

    return { bind, interrupt };
}
