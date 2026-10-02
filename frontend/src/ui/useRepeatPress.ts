/**
 * 按住重复触发。
 *
 * 【用途】「平滑」这类**可累积**的动作：点一下走一步，按住则连续走 —— 与键盘的自动
 * 重复同一套手感（先等一小段延迟，再以固定间隔重复）。抽成 hook 而不是写进某个按钮，
 * 是因为任何"点一下走一步"的动作都需要它，而每个调用方各自接一遍定时器必然漂移
 * （延迟、间隔、清理由谁负责）。
 *
 * 【为什么指针与键盘分两条路】指针按下要**立即**响应（"短按立刻执行一次"），而
 * `<button>` 的 `click` 要到抬起时才来 —— 两条都接就会重复触发一次。这里用
 * `event.detail` 区分：指针点击 `detail >= 1`（已在 pointerdown 触发过，跳过），
 * 键盘激活（Enter / Space）`detail === 0`（在这里补上）。不用"是否按过"的布尔标记，
 * 因此不存在"按下后没等到 click，标记卡住、吞掉下一次键盘操作"的边角。
 *
 * 【为什么要捕获指针】按住后把指针拖出按钮再松手，不捕获就收不到 `pointerup`，
 * 重复会一直跑下去。捕获让 `pointerup` 必定回到按钮上。
 *
 * 【调用方必须给元素加 `hs-touch-none`】按住重复的前提是"指针始终没离开元素"。
 * 触摸设备上，手指的轻微漂移会被浏览器解释成滚动并派发 `pointercancel`，重复
 * 因此中断 —— 而 `touch-action` 必须在触摸开始**之前**就设好，本 hook 无法
 * 事后补救（指针捕获也拦不住手势被浏览器接管）。类定义见 `index.css`。
 */
import {
    useCallback,
    useEffect,
    useRef,
    type PointerEvent as ReactPointerEvent,
    type MouseEvent as ReactMouseEvent,
} from "react";

/** 长按进入重复前的等待（ms）：短于它的按下视为"点一下"。 */
export const REPEAT_PRESS_DELAY_MS = 320;

/** 重复间隔（ms）。 */
export const REPEAT_PRESS_INTERVAL_MS = 70;

export interface RepeatPressOptions {
    /** 触发一次动作（短按一次、长按每帧一次）。 */
    onTrigger: () => void;
    disabled?: boolean;
    /** 首次重复前的等待；省略用 `REPEAT_PRESS_DELAY_MS`。 */
    delayMs?: number;
    /** 重复间隔；省略用 `REPEAT_PRESS_INTERVAL_MS`。 */
    intervalMs?: number;
}

/** 摊给按钮的事件处理器（`<AppButton {...useRepeatPress(...)} />`）。 */
export interface RepeatPressHandlers {
    onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
    onPointerUp: () => void;
    onPointerCancel: () => void;
    onLostPointerCapture: () => void;
    onClick: (event: ReactMouseEvent<HTMLElement>) => void;
}

export function useRepeatPress({
    onTrigger,
    disabled = false,
    delayMs = REPEAT_PRESS_DELAY_MS,
    intervalMs = REPEAT_PRESS_INTERVAL_MS,
}: RepeatPressOptions): RepeatPressHandlers {
    /*
     * 最新回调经 ref 转发：调用方几乎总是传内联箭头函数（每次渲染都是新身份），
     * 直接依赖它会让处理器反复重建、把进行中的重复打断。写入放在 effect 里
     * （而不是渲染期），与 `useFrameCommitter` 的处理一致。
     */
    const triggerRef = useRef(onTrigger);
    useEffect(() => {
        triggerRef.current = onTrigger;
    });

    const timersRef = useRef<{ delay: number | null; interval: number | null }>({
        delay: null,
        interval: null,
    });

    const stop = useCallback(() => {
        const timers = timersRef.current;
        if (timers.delay != null) {
            window.clearTimeout(timers.delay);
            timers.delay = null;
        }
        if (timers.interval != null) {
            window.clearInterval(timers.interval);
            timers.interval = null;
        }
    }, []);

    // 卸载时清干净：留着的 interval 会打到已卸载的父组件上。
    useEffect(() => stop, [stop]);

    const onPointerDown = useCallback(
        (event: ReactPointerEvent<HTMLElement>) => {
            if (disabled || event.button !== 0) return;
            // 短按立即执行一次：不等 click，手感才"跟手"。
            triggerRef.current();
            try {
                event.currentTarget.setPointerCapture(event.pointerId);
            } catch {
                // 环境不支持指针捕获（如测试环境）：退回普通 pointerup 路径。
            }
            stop();
            timersRef.current.delay = window.setTimeout(() => {
                timersRef.current.delay = null;
                timersRef.current.interval = window.setInterval(
                    () => triggerRef.current(),
                    intervalMs,
                );
            }, delayMs);
        },
        [disabled, delayMs, intervalMs, stop],
    );

    const onClick = useCallback(
        (event: ReactMouseEvent<HTMLElement>) => {
            if (disabled) return;
            // 指针点击（detail ≥ 1）已在 pointerdown 触发过；这里只接键盘激活。
            if (event.detail > 0) return;
            triggerRef.current();
        },
        [disabled],
    );

    return {
        onPointerDown,
        onPointerUp: stop,
        onPointerCancel: stop,
        onLostPointerCapture: stop,
        onClick,
    };
}
